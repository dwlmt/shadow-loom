"""
Step 8 — Directive Assembly.

Translates the mathematical state of a post-simulation sandbox (or a raw
ego-graph for observation/directive queries) into a structured CreativeBrief
that a downstream drafting LLM can consume.

This is a **template engine only** — no LLM calls happen here.  The output
is a fully typed Pydantic payload ready for Steps 9-10.
"""

from __future__ import annotations

import logging
import math
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Literal

import networkx as nx
from pydantic import BaseModel, Field

from shadow_loom.models import WorldStateV1
from shadow_loom.query_models import DirectiveQuery

logger = logging.getLogger(__name__)


# =====================================================================
# Analysis Models
# =====================================================================

class EpistemicGap(BaseModel):
    """Delta between a character's belief and objective reality."""
    entity_id: str
    belief_target_id: str
    believed_state: str
    actual_state: str
    gap_type: Literal["contradicted", "confirmed", "unknown", "quantitative"] = "unknown"
    gap_magnitude: float = Field(
        default=0.0,
        description="0.0 = aligned, 1.0 = maximally wrong",
    )


class TraitTrajectory(BaseModel):
    """Snapshot of a single trait with headroom analysis."""
    entity_id: str
    trait_name: str
    current_value: float
    inertia: float
    headroom_up: float    # 1.0 - current_value
    headroom_down: float  # current_value - 0.0


class RelationshipTension(BaseModel):
    """Snapshot of a relationship with asymmetry analysis."""
    source_id: str
    target_id: str
    affinity: float
    fear: float
    power_dynamic: float
    asymmetry_score: float = Field(
        default=0.0,
        description="How unbalanced the relationship is (0 = perfectly symmetric)",
    )


class NarrativeTension(BaseModel):
    """Fabula/syuzhet displacement for a single event.

    Measures narrative withholding — the distance between when something
    *happened* (fabula) and when it is *revealed* (syuzhet).
    """
    event_id: str
    fabula_time: int
    syuzhet_index: int
    description: str
    displacement: float = Field(
        description=(
            "Normalised rank displacement in [-1, 1]. "
            "Positive = delayed revelation (withheld); "
            "Negative = shown before its chronological turn (flash-forward)."
        ),
    )
    tension_type: Literal[
        "withheld_cause", "upcoming_revelation", "linear",
    ] = "linear"


class HiddenInformationChannel(BaseModel):
    """An InformationEdge whose existence has not yet been revealed to the reader."""
    source_id: str
    target_ids: List[str]
    medium: str
    discovered_at_syuzhet: int
    is_encrypted: bool = False


class CandidateResult(BaseModel):
    """Outcome of testing a candidate event through the causal physics envelope."""
    interventions: Dict[str, Any]
    valid: bool
    affective_score: float = Field(
        default=float("inf"),
        description=(
            "Distance from the target emotion (lower = better match). "
            "inf for pruned candidates."
        ),
    )
    blocked_reasons: List[str] = Field(default_factory=list)
    mutations: List[Dict[str, Any]] = Field(default_factory=list)


class ConstraintBlock(BaseModel):
    """A single instruction constraint for the drafting LLM."""
    constraint_type: Literal["mathematical", "epistemic", "spatial", "temporal", "narrative"]
    priority: Literal["hard", "soft"]
    instruction: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class CreativeBrief(BaseModel):
    """Structured output ready for a downstream drafting LLM."""
    target_effect: str
    target_entities: List[str]
    constraints: List[ConstraintBlock] = Field(default_factory=list)
    epistemic_gaps: List[EpistemicGap] = Field(default_factory=list)
    narrative_tensions: List[NarrativeTension] = Field(default_factory=list)
    hidden_channels: List[HiddenInformationChannel] = Field(default_factory=list)
    trait_trajectories: List[TraitTrajectory] = Field(default_factory=list)
    relationship_tensions: List[RelationshipTension] = Field(default_factory=list)
    physics_override: Optional[str] = None
    scene_context: Dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# Engine
# =====================================================================

class DirectiveAssembler:
    """
    Computes epistemic gaps, trait trajectories, and relationship tensions,
    then assembles them into a ``CreativeBrief`` for a given ``DirectiveQuery``.

    Usage::

        assembler = DirectiveAssembler(sandbox, ego_payload, world_state)
        brief = assembler.assemble(directive_query)
    """

    def __init__(
        self,
        sandbox: nx.MultiDiGraph | None,
        ego_payload: Dict[str, Any],
        world_state: WorldStateV1,
    ) -> None:
        self.sandbox = sandbox
        self.ego = ego_payload
        self.world_state = world_state

    # ------------------------------------------------------------------
    # Epistemic gap computation
    # ------------------------------------------------------------------
    def compute_epistemic_gaps(self, entity_ids: List[str]) -> List[EpistemicGap]:
        """Compare each entity's beliefs against the objective graph state."""
        gaps: List[EpistemicGap] = []

        for eid in entity_ids:
            ent_data = self._find_entity(eid)
            if not ent_data:
                continue

            for belief in ent_data.get("beliefs", []):
                target_id = belief.get("target_id", "")
                believed = belief.get("perceived_state", "")
                confidence = belief.get("confidence", 0.5)

                actual = self._resolve_actual_state(target_id)

                gap_type, magnitude = self._classify_gap(believed, actual, confidence)
                logger.debug("[DirectiveAssembly·Epistemic] %s belief about %s: believed=%r actual=%r → gap=%s mag=%.2f",
                             eid, target_id, believed, actual, gap_type, magnitude)
                gaps.append(EpistemicGap(
                    entity_id=eid,
                    belief_target_id=target_id,
                    believed_state=believed,
                    actual_state=actual,
                    gap_type=gap_type,
                    gap_magnitude=magnitude,
                ))
        return gaps

    # ------------------------------------------------------------------
    # Trait trajectory computation
    # ------------------------------------------------------------------
    def compute_trait_trajectories(self, entity_ids: List[str]) -> List[TraitTrajectory]:
        """Compute current value + headroom for each trait of given entities."""
        trajectories: List[TraitTrajectory] = []

        for eid in entity_ids:
            ent_data = self._find_entity(eid)
            if not ent_data:
                continue
            for trait_name, trait_data in ent_data.get("traits", {}).items():
                if not isinstance(trait_data, dict):
                    continue
                val = trait_data.get("value", 0.5)
                inertia = trait_data.get("inertia", 0.5)
                trajectories.append(TraitTrajectory(
                    entity_id=eid,
                    trait_name=trait_name,
                    current_value=val,
                    inertia=inertia,
                    headroom_up=1.0 - val,
                    headroom_down=val,
                ))
        return trajectories

    # ------------------------------------------------------------------
    # Relationship tension computation
    # ------------------------------------------------------------------
    def compute_relationship_tensions(self, entity_ids: List[str]) -> List[RelationshipTension]:
        """Compute relationship tensions involving the given entities."""
        tensions: List[RelationshipTension] = []
        eid_set = set(entity_ids)

        for rel in self.ego.get("relevant_relationships", []):
            src = rel.get("source_entity_id", "")
            tgt = rel.get("target_entity_id", "")
            if src not in eid_set and tgt not in eid_set:
                continue

            aff = rel.get("affinity", 0.0)
            fear = rel.get("fear", 0.0)
            power = rel.get("power_dynamic", 0.0)

            # Asymmetry: check for a reverse edge
            reverse_aff = 0.0
            reverse_fear = 0.0
            reverse_power = 0.0
            for rev in self.ego.get("relevant_relationships", []):
                if rev.get("source_entity_id") == tgt and rev.get("target_entity_id") == src:
                    reverse_aff = rev.get("affinity", 0.0)
                    reverse_fear = rev.get("fear", 0.0)
                    reverse_power = rev.get("power_dynamic", 0.0)
                    break

            asymmetry = (
                abs(aff - reverse_aff)
                + abs(fear - reverse_fear)
                + abs(power + reverse_power)  # power should be anti-symmetric
            ) / 3.0

            tensions.append(RelationshipTension(
                source_id=src, target_id=tgt,
                affinity=aff, fear=fear, power_dynamic=power,
                asymmetry_score=round(asymmetry, 3),
            ))
        return tensions

    # ------------------------------------------------------------------
    # Narrative tension computation (fabula/syuzhet displacement)
    # ------------------------------------------------------------------
    def compute_narrative_tension(
        self, syuzhet_anchor: Optional[int] = None,
    ) -> List[NarrativeTension]:
        """Compute rank displacement between fabula and syuzhet ordering.

        Parameters
        ----------
        syuzhet_anchor : int or None
            The reader's current position in the narrative.  Events with
            ``syuzhet_index > syuzhet_anchor`` are *not yet revealed*.
            If ``None``, all events are considered revealed.
        """
        events = self.world_state.events
        if not events:
            return []

        n = len(events)
        fabula_sorted = sorted(events, key=lambda e: e.fabula_time)
        syuzhet_sorted = sorted(events, key=lambda e: e.syuzhet_index)

        fabula_rank = {e.id: i for i, e in enumerate(fabula_sorted)}
        syuzhet_rank = {e.id: i for i, e in enumerate(syuzhet_sorted)}

        max_rank = max(n - 1, 1)  # avoid /0

        tensions: List[NarrativeTension] = []
        for evt in events:
            disp = (syuzhet_rank[evt.id] - fabula_rank[evt.id]) / max_rank

            if abs(disp) < 0.01:
                ttype = "linear"
            elif disp > 0:
                # Shown later than it happened — withheld
                ttype = "withheld_cause"
            else:
                # Shown before its chronological turn
                ttype = "upcoming_revelation"

            # If a syuzhet_anchor is given, only flag events that are
            # unrevealed (syuzhet_index > anchor) AND chronologically
            # already past (fabula_time <= max revealed fabula).
            if syuzhet_anchor is not None and evt.syuzhet_index <= syuzhet_anchor:
                ttype = "linear"  # already revealed — no active tension

            tensions.append(NarrativeTension(
                event_id=evt.id,
                fabula_time=evt.fabula_time,
                syuzhet_index=evt.syuzhet_index,
                description=evt.description,
                displacement=round(disp, 3),
                tension_type=ttype,
            ))

        return tensions

    # ------------------------------------------------------------------
    # Hidden information channels (discovered_at_syuzhet)
    # ------------------------------------------------------------------
    def compute_hidden_channels(
        self, syuzhet_anchor: Optional[int] = None,
    ) -> List[HiddenInformationChannel]:
        """Find InformationEdges whose existence is hidden from the reader.

        An edge is hidden when its ``discovered_at_syuzhet`` is greater
        than the current *syuzhet_anchor*.
        """
        if syuzhet_anchor is None:
            return []
        hidden: List[HiddenInformationChannel] = []
        for ie in self.world_state.information_topology:
            if ie.discovered_at_syuzhet > syuzhet_anchor:
                hidden.append(HiddenInformationChannel(
                    source_id=ie.source_id,
                    target_ids=ie.target_ids,
                    medium=ie.medium,
                    discovered_at_syuzhet=ie.discovered_at_syuzhet,
                    is_encrypted=ie.is_encrypted,
                ))
        return hidden

    # ------------------------------------------------------------------
    # Graph helpers for affective measures
    # ------------------------------------------------------------------
    def _build_causal_digraph(self) -> nx.DiGraph:
        """Build a weighted causal DiGraph from the world state topology."""
        _STRENGTH_W = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}
        g = nx.DiGraph()
        for ce in self.world_state.causal_topology:
            evidence_w = _STRENGTH_W.get(ce.evidence_strength, 0.5)
            force_scale = ce.causal_force / 10.0
            w = evidence_w * force_scale
            if g.has_edge(ce.source_id, ce.target_id):
                existing = g[ce.source_id][ce.target_id]["weight"]
                if w <= existing:
                    continue
            g.add_edge(ce.source_id, ce.target_id,
                       weight=w, mechanism=ce.mechanism)
        return g

    def _revealed_event_ids(self, syuzhet_anchor: Optional[int]) -> set[str]:
        """Return IDs of events the reader has seen by *syuzhet_anchor*."""
        if syuzhet_anchor is None:
            return {e.id for e in self.world_state.events}
        return {e.id for e in self.world_state.events
                if e.syuzhet_index <= syuzhet_anchor}

    # ------------------------------------------------------------------
    # Mystery  (Epistemic Gap — hidden causal ancestors)
    # ------------------------------------------------------------------
    def compute_mystery_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Mystery: ratio of hidden causal ancestors of known effects.

        Walks backward from each known effect node involving the target
        entities and counts how many of its causal predecessors are NOT
        yet revealed to the reader.  Returns a ratio in [0, 1].
        """
        causal_g = self._build_causal_digraph()
        revealed = self._revealed_event_ids(syuzhet_anchor)
        eid_set = set(entity_ids)

        # Identify "effect" nodes: revealed events involving target entities,
        # plus the entities themselves (which can be causal targets).
        effect_nodes: set[str] = set()
        for evt in self.world_state.events:
            if evt.id in revealed and (
                set(evt.actor_ids) & eid_set or set(evt.target_ids) & eid_set
            ):
                effect_nodes.add(evt.id)
        effect_nodes |= eid_set

        total_ancestors = 0
        hidden_ancestors = 0

        for eff in effect_nodes:
            if not causal_g.has_node(eff):
                continue
            ancestors = nx.ancestors(causal_g, eff)
            if not ancestors:
                continue
            total_ancestors += len(ancestors)
            hidden_ancestors += len(ancestors - revealed)

        if total_ancestors == 0:
            return 0.0

        score = hidden_ancestors / total_ancestors
        logger.debug(
            "[DirectiveAssembly·Mystery] hidden=%d / total=%d = %.3f",
            hidden_ancestors, total_ancestors, score,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Dramatic Irony  (Epistemic Asymmetry — reader > character)
    # ------------------------------------------------------------------
    def compute_dramatic_irony_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Dramatic Irony: information asymmetry where reader knows more.

        For each target entity, finds revealed causal edges (source event
        is in the reader's syuzhet graph) that point **to** the entity.
        If the source event is NOT in the character's belief targets, the
        reader sees a threat/secret the character cannot — dramatic irony.
        Returns a ratio in [0, 1].
        """
        if syuzhet_anchor is None:
            return 0.0

        revealed = self._revealed_event_ids(syuzhet_anchor)

        irony_gaps = 0
        total_connections = 0

        for eid in entity_ids:
            ent = self.world_state.entities.get(eid)
            if not ent:
                continue

            character_aware_of = {b.target_id for b in ent.beliefs}

            # Revealed causal edges targeting this entity
            for ce in self.world_state.causal_topology:
                if ce.target_id != eid:
                    continue
                if ce.source_id not in revealed:
                    continue  # Reader doesn't know this either
                total_connections += 1
                if ce.source_id not in character_aware_of:
                    irony_gaps += 1

            # Revealed information edges the character is unaware of
            for ie in self.world_state.information_topology:
                if ie.discovered_at_syuzhet > syuzhet_anchor:
                    continue
                if eid not in ie.target_ids:
                    continue
                total_connections += 1
                if ie.source_id not in character_aware_of:
                    irony_gaps += 1

        if total_connections == 0:
            return 0.0

        score = min(irony_gaps / total_connections, 1.0)
        logger.debug(
            "[DirectiveAssembly·DramaticIrony] gaps=%d / connections=%d = %.3f",
            irony_gaps, total_connections, score,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Suspense  (Probabilistic Valence — threat vs hope)
    # ------------------------------------------------------------------
    def compute_suspense_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Suspense: forward causal momentum between opposed outcomes.

        Examines unrevealed events involving the target entities and
        classifies them as *threat* (entity is target/victim) or *hope*
        (entity has agency/is actor).  Uses ``evidence_strength`` as a
        probability proxy — incoming causal edge weights when available,
        outgoing edge weights as fallback.

        Returns ``P(threat) - P(hope)`` clamped to [0, 1].
        Returns 0 when hope is entirely extinguished (despair, not suspense).
        """
        causal_g = self._build_causal_digraph()
        revealed = self._revealed_event_ids(syuzhet_anchor)
        all_evt_ids = {e.id for e in self.world_state.events}
        unrevealed = all_evt_ids - revealed
        eid_set = set(entity_ids)

        threat_prob = 0.0
        hope_prob = 0.0

        for evt_id in unrevealed:
            evt = next(
                (e for e in self.world_state.events if e.id == evt_id), None
            )
            if not evt:
                continue
            if not (set(evt.actor_ids) & eid_set) and not (set(evt.target_ids) & eid_set):
                continue

            # Probability proxy: prefer incoming edge weight, fall back to
            # outgoing edge weight (a strongly causal event is significant),
            # default to 0.5 (maximum entropy).
            prob = 0.5
            if causal_g.has_node(evt_id):
                in_edges = list(causal_g.in_edges(evt_id, data=True))
                out_edges = list(causal_g.out_edges(evt_id, data=True))
                if in_edges:
                    prob = max(d.get("weight", 0.5) for _, _, d in in_edges)
                elif out_edges:
                    prob = max(d.get("weight", 0.5) for _, _, d in out_edges)

            # Classify: entity acted upon → threat; entity acting → hope
            if (set(evt.target_ids) & eid_set) and not (set(evt.actor_ids) & eid_set):
                threat_prob = max(threat_prob, prob)
            elif set(evt.actor_ids) & eid_set:
                hope_prob = max(hope_prob, prob)

        if hope_prob <= 0.0:
            logger.debug(
                "[DirectiveAssembly·Suspense] No hope outcome — suspense=0 (despair)",
            )
            return 0.0

        score = max(0.0, threat_prob - hope_prob)
        logger.debug(
            "[DirectiveAssembly·Suspense] P(threat)=%.3f P(hope)=%.3f suspense=%.3f",
            threat_prob, hope_prob, score,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Surprise  (Prediction Error — KL Divergence)
    # ------------------------------------------------------------------
    def compute_surprise_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Surprise: KL divergence between reader's prior and actual state.

        Models each entity trait as a Bernoulli variable.

        * **Prior** — maximum-entropy baseline (0.5) adjusted toward the
          actual value for each *revealed* causal edge (what the reader
          can reasonably infer).  Unrevealed causes leave the prior at
          0.5, maximising the prediction error.
        * **Posterior** — actual trait values from the sandbox
          (post-simulation) or world state (truth).

        Returns a normalised KL divergence in [0, 1].
        """
        if syuzhet_anchor is None:
            return 0.0  # reader knows everything → no surprise

        EPS = 0.01
        _STRENGTH_W = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}
        revealed = self._revealed_event_ids(syuzhet_anchor)

        total_kl = 0.0
        trait_count = 0

        for eid in entity_ids:
            # Posterior: prefer sandbox, fall back to world_state
            if self.sandbox is not None and self.sandbox.has_node(eid):
                actual_traits = self.sandbox.nodes[eid].get("traits", {})
            else:
                actual_ent = self.world_state.entities.get(eid)
                if not actual_ent:
                    continue
                actual_traits = {
                    k: {"value": v.value} for k, v in actual_ent.traits.items()
                }

            for trait_name, actual_data in actual_traits.items():
                if not isinstance(actual_data, dict) or "value" not in actual_data:
                    continue

                actual_val = actual_data["value"]

                # Prior: start from maximum entropy, shift toward truth
                # for each revealed causal edge targeting this entity.
                prior_val = 0.5
                for ce in self.world_state.causal_topology:
                    if ce.target_id != eid:
                        continue
                    if ce.source_id not in revealed:
                        continue
                    w = _STRENGTH_W.get(ce.evidence_strength, 0.5)
                    prior_val += (actual_val - prior_val) * w * 0.5

                p = max(EPS, min(1 - EPS, actual_val))    # posterior
                q = max(EPS, min(1 - EPS, prior_val))     # prior

                # Binary KL: D_KL(p || q)
                kl = (
                    p * math.log(p / q)
                    + (1 - p) * math.log((1 - p) / (1 - q))
                )
                total_kl += max(0.0, kl)
                trait_count += 1

        if trait_count == 0:
            return 0.0

        max_kl = math.log(1 / EPS)
        avg_kl = total_kl / trait_count
        score = min(avg_kl / max_kl, 1.0)

        logger.debug(
            "[DirectiveAssembly·Surprise] avg_kl=%.4f max_kl=%.4f normalised=%.3f "
            "over %d traits",
            avg_kl, max_kl, score, trait_count,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Affective score computation (the "loss function")
    # ------------------------------------------------------------------
    def compute_affective_score(
        self,
        target_effect: str,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Compute how well the current world state matches *target_effect*.

        Returns a scalar "loss" — lower is better.  Negative scores
        indicate a strong match; positive scores indicate a poor match.

        Each of the four structural effects (mystery, dramatic_irony,
        suspense, surprise) dispatches to its dedicated graph-based
        method.  Emotion effects use trait-trajectory headroom.
        """
        score = 0.0
        logger.debug("[DirectiveAssembly·AffectiveScore] effect=%s entities=%s",
                     target_effect, entity_ids)

        # --- Structural effects (graph-based) ---
        if target_effect == "mystery":
            score -= self.compute_mystery_score(entity_ids, syuzhet_anchor)

        elif target_effect == "dramatic_irony":
            score -= self.compute_dramatic_irony_score(entity_ids, syuzhet_anchor)

        elif target_effect == "suspense":
            score -= self.compute_suspense_score(entity_ids, syuzhet_anchor)

        elif target_effect == "surprise":
            score -= self.compute_surprise_score(entity_ids, syuzhet_anchor)

        # --- Emotion effects (trait headroom) ---
        else:
            _EFFECT_TRAIT_MAP: Dict[str, List[str]] = {
                "grief": ["despair", "love", "hope"],
                "rage": ["anger", "rebelliousness", "resentment"],
                "joy": ["happiness", "hope", "contentment"],
                "fear": ["fear", "paranoia", "anxiety"],
                "love": ["love", "affection", "sensuality"],
                "regret": ["guilt", "remorse", "despair"],
            }
            _EFFECT_DECREASE: Dict[str, set] = {
                "grief": {"hope", "happiness", "contentment"},
                "rage": {"patience", "calm"},
                "joy": {"despair", "fear", "anxiety", "sadness"},
                "fear": {"courage", "hope", "calm"},
                "love": {"anger", "resentment"},
                "regret": {"happiness", "contentment", "hope"},
            }

            trajectories = self.compute_trait_trajectories(entity_ids)
            target_traits = _EFFECT_TRAIT_MAP.get(target_effect, [])
            decrease_set = _EFFECT_DECREASE.get(target_effect, set())
            relevant = [
                traj for traj in trajectories
                if traj.trait_name in target_traits
            ]
            if relevant:
                avg_headroom = sum(
                    traj.headroom_down if traj.trait_name in decrease_set
                    else traj.headroom_up
                    for traj in relevant
                ) / len(relevant)
                score -= avg_headroom
                logger.debug("[DirectiveAssembly·AffectiveScore] effect=%s avg_headroom=%.3f score=%.4f",
                             target_effect, avg_headroom, score)
            else:
                score += 0.5

        return round(score, 4)

    # ------------------------------------------------------------------
    # Candidate event pruning (the "envelope of possibilities")
    # ------------------------------------------------------------------
    def evaluate_candidate_events(
        self,
        candidates: List[Dict[str, Any]],
        target_effect: str,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> List[CandidateResult]:
        """Fork reality for each candidate intervention, prune the
        physically impossible ones, and rank survivors by affective score.

        This is the core "envelope of possibilities" method described in the
        Directive Assembly spec.  It requires a live sandbox (``self.sandbox``
        must not be ``None``).

        Parameters
        ----------
        candidates
            Each dict maps intervention paths to values, in the same format
            accepted by ``AMWNInstantiator.execute_interventions()``.
        target_effect
            The narrative emotion to optimise for.
        entity_ids
            Entities whose affective state is being scored.
        syuzhet_anchor
            Reader position for narrative tension scoring.

        Returns
        -------
        List[CandidateResult]
            Sorted by ``affective_score`` ascending (best first).
            Invalid candidates are appended at the end.
        """
        from shadow_loom.causal_physics import CausalPhysicsEngine

        if self.sandbox is None:
            raise ValueError(
                "evaluate_candidate_events requires a live sandbox. "
                "Pass one via DirectiveAssembler(sandbox=..., ...)."
            )

        results: List[CandidateResult] = []

        for candidate in candidates:
            forked = deepcopy(self.sandbox)
            engine = CausalPhysicsEngine(forked, self.world_state)
            physics = engine.execute(rung=2, interventions=candidate)

            # --- Pruning: intervention failed AND all propagations blocked ---
            # A candidate is physically impossible only if the surgery itself
            # was blocked (no intervened nodes) and no mutations occurred.
            surgery_applied = len(physics.intervened_nodes) > 0
            has_mutations = len(physics.mutations) > 0
            all_blocked = len(physics.blocked) > 0 and not has_mutations

            if not surgery_applied and all_blocked:
                logger.debug("[DirectiveAssembly·Candidate] PRUNED: no surgery applied and all blocked. reasons=%s",
                             [b.reason for b in physics.blocked])
                results.append(CandidateResult(
                    interventions=candidate,
                    valid=False,
                    blocked_reasons=[b.reason for b in physics.blocked],
                ))
                continue

            # --- Affective scoring on the post-intervention state ---
            # Build a fresh assembler on the forked sandbox
            forked_ego = self.ego  # ego is read-only, safe to share
            forked_assembler = DirectiveAssembler(
                sandbox=forked, ego_payload=forked_ego,
                world_state=self.world_state,
            )
            aff_score = forked_assembler.compute_affective_score(
                target_effect, entity_ids, syuzhet_anchor,
            )

            results.append(CandidateResult(
                interventions=candidate,
                valid=True,
                affective_score=aff_score,
                mutations=[m.model_dump() for m in physics.mutations],
            ))
            logger.debug("[DirectiveAssembly·Candidate] VALID: score=%.4f mutations=%d",
                         aff_score, len(physics.mutations))

        # Sort: valid candidates by score (ascending), invalid at the end
        valid = sorted(
            [r for r in results if r.valid],
            key=lambda r: r.affective_score,
        )
        invalid = [r for r in results if not r.valid]
        return valid + invalid

    # ------------------------------------------------------------------
    # Main assembler
    # ------------------------------------------------------------------
    def assemble(
        self,
        directive: DirectiveQuery,
        syuzhet_anchor: Optional[int] = None,
    ) -> CreativeBrief:
        """Build a ``CreativeBrief`` for the given directive.

        Parameters
        ----------
        directive : DirectiveQuery
            The narrative directive to fulfil.
        syuzhet_anchor : int or None
            The reader's position in the text (syuzhet_index).  Enables
            fabula/syuzhet displacement reasoning for suspense, surprise,
            mystery, and dramatic irony.
        """
        entity_ids = directive.target_entity_ids
        effect = directive.target_effect
        intensity = directive.intensity
        logger.debug("[DirectiveAssembly·Assemble] effect=%s entities=%s intensity=%.2f",
                     effect, entity_ids, intensity)

        gaps = self.compute_epistemic_gaps(entity_ids)
        trajectories = self.compute_trait_trajectories(entity_ids)
        rel_tensions = self.compute_relationship_tensions(entity_ids)
        narrative_tensions = self.compute_narrative_tension(syuzhet_anchor)
        hidden_channels = self.compute_hidden_channels(syuzhet_anchor)

        constraints: List[ConstraintBlock] = []

        # =============================================================
        # MYSTERY  (hidden causal ancestors)
        # =============================================================
        if effect == "mystery":
            mystery_score = self.compute_mystery_score(entity_ids, syuzhet_anchor)

            # Identify effect nodes and their hidden predecessors
            causal_g = self._build_causal_digraph()
            revealed = self._revealed_event_ids(syuzhet_anchor)
            eid_set = set(entity_ids)

            hidden_causes: List[tuple] = []
            for evt in self.world_state.events:
                if evt.id not in revealed:
                    continue
                if not (set(evt.actor_ids) & eid_set) and not (set(evt.target_ids) & eid_set):
                    continue
                if not causal_g.has_node(evt.id):
                    continue
                ancestors = nx.ancestors(causal_g, evt.id)
                hidden = ancestors - revealed
                if hidden:
                    hidden_causes.append((evt, hidden))

            if hidden_causes:
                most_mysterious_evt, most_hidden = max(
                    hidden_causes, key=lambda x: len(x[1]),
                )
                constraints.append(ConstraintBlock(
                    constraint_type="epistemic",
                    priority="hard",
                    instruction=(
                        f"[MYSTERY CONSTRAINT]: The event '{most_mysterious_evt.id}' "
                        f"({most_mysterious_evt.description}) has "
                        f"{len(most_hidden)} hidden causal predecessor(s) that "
                        f"the reader has NOT yet seen. You MUST NOT reveal or "
                        f"hint at these causes. The reader should feel the weight "
                        f"of the unknown (mystery_score={mystery_score:.2f})."
                    ),
                    evidence={
                        "effect_event_id": most_mysterious_evt.id,
                        "hidden_cause_count": len(most_hidden),
                        "hidden_cause_ids": sorted(most_hidden),
                        "mystery_score": mystery_score,
                    },
                ))
                for evt, hidden in hidden_causes:
                    if evt.id != most_mysterious_evt.id:
                        constraints.append(ConstraintBlock(
                            constraint_type="epistemic",
                            priority="soft",
                            instruction=(
                                f"Also mysterious: '{evt.id}' has "
                                f"{len(hidden)} hidden cause(s). "
                                f"Do not reveal."
                            ),
                            evidence={
                                "effect_event_id": evt.id,
                                "hidden_cause_count": len(hidden),
                            },
                        ))

            # Hidden information channels amplify mystery
            if hidden_channels:
                for hc in hidden_channels:
                    constraints.append(ConstraintBlock(
                        constraint_type="narrative",
                        priority="hard",
                        instruction=(
                            f"[HIDDEN CHANNEL]: A {hc.medium} link from "
                            f"{hc.source_id} to {hc.target_ids} exists but "
                            f"is not revealed until syuzhet_index="
                            f"{hc.discovered_at_syuzhet}. Do not reference it."
                        ),
                        evidence={
                            "source_id": hc.source_id,
                            "medium": hc.medium,
                            "discovered_at_syuzhet": hc.discovered_at_syuzhet,
                        },
                    ))

            # Fallback: if no hidden causal ancestors, use character belief
            # contradictions — the character's ignorance is itself a source
            # of mystery atmosphere for the scene.
            if not hidden_causes:
                contradicted = [g for g in gaps if g.gap_type == "contradicted"]
                if contradicted:
                    widest = max(contradicted, key=lambda g: g.gap_magnitude)
                    constraints.append(ConstraintBlock(
                        constraint_type="epistemic",
                        priority="hard",
                        instruction=(
                            f"[MYSTERY CONSTRAINT]: {widest.entity_id} "
                            f"believes '{widest.believed_state}' about "
                            f"{widest.belief_target_id}, but the truth is "
                            f"'{widest.actual_state}'. This ignorance is "
                            f"central to the mystery "
                            f"(magnitude={widest.gap_magnitude:.2f})."
                        ),
                        evidence={
                            "entity_id": widest.entity_id,
                            "belief_target_id": widest.belief_target_id,
                            "gap_magnitude": widest.gap_magnitude,
                        },
                    ))

        # =============================================================
        # DRAMATIC IRONY  (reader knows more than character)
        # =============================================================
        elif effect == "dramatic_irony":
            irony_score = self.compute_dramatic_irony_score(
                entity_ids, syuzhet_anchor,
            )
            revealed = self._revealed_event_ids(syuzhet_anchor)

            # Find specific irony points: reader sees cause → character,
            # but the character's beliefs lack the source event.
            irony_details: List[tuple] = []
            for eid in entity_ids:
                ent = self.world_state.entities.get(eid)
                if not ent:
                    continue
                character_aware_of = {b.target_id for b in ent.beliefs}
                for ce in self.world_state.causal_topology:
                    if ce.target_id != eid:
                        continue
                    if ce.source_id not in revealed:
                        continue
                    if ce.source_id not in character_aware_of:
                        src_evt = next(
                            (e for e in self.world_state.events
                             if e.id == ce.source_id),
                            None,
                        )
                        if src_evt:
                            irony_details.append((eid, src_evt))

            if irony_details:
                eid, src_evt = irony_details[0]
                constraints.append(ConstraintBlock(
                    constraint_type="epistemic",
                    priority="hard",
                    instruction=(
                        f"[DRAMATIC IRONY]: The reader knows about "
                        f"'{src_evt.id}' ({src_evt.description}) which "
                        f"causally affects {eid}, but {eid} is UNAWARE "
                        f"of this connection. Write the scene so the "
                        f"reader feels this information asymmetry. Show "
                        f"{eid} acting in ignorance while the reader "
                        f"knows the truth. The character MUST NOT learn "
                        f"about this connection during this scene "
                        f"(irony_score={irony_score:.2f})."
                    ),
                    evidence={
                        "entity_id": eid,
                        "secret_event_id": src_evt.id,
                        "irony_score": irony_score,
                    },
                ))
                for e_id, s_evt in irony_details[1:]:
                    constraints.append(ConstraintBlock(
                        constraint_type="epistemic",
                        priority="soft",
                        instruction=(
                            f"Also hidden from {e_id}: '{s_evt.id}' "
                            f"({s_evt.description})."
                        ),
                        evidence={
                            "entity_id": e_id,
                            "secret_event_id": s_evt.id,
                        },
                    ))

            # Character belief gaps reinforce the irony
            contradicted = [g for g in gaps if g.gap_type == "contradicted"]
            if contradicted:
                widest = max(contradicted, key=lambda g: g.gap_magnitude)
                constraints.append(ConstraintBlock(
                    constraint_type="epistemic",
                    priority="hard",
                    instruction=(
                        f"[EPISTEMIC CONSTRAINT]: {widest.entity_id} believes "
                        f"'{widest.believed_state}' about "
                        f"{widest.belief_target_id}, but the truth is "
                        f"'{widest.actual_state}'. The audience MUST see "
                        f"through this character's ignorance. "
                        f"You MUST NOT allow {widest.entity_id} to learn "
                        f"the truth "
                        f"(magnitude={widest.gap_magnitude:.2f})."
                    ),
                    evidence={
                        "entity_id": widest.entity_id,
                        "belief_target_id": widest.belief_target_id,
                        "gap_magnitude": widest.gap_magnitude,
                    },
                ))

        # =============================================================
        # SUSPENSE  (forward causal momentum — threat vs hope)
        # =============================================================
        elif effect == "suspense":
            suspense_score = self.compute_suspense_score(
                entity_ids, syuzhet_anchor,
            )

            # Layer 1: Forward probability tension
            if suspense_score > 0:
                constraints.append(ConstraintBlock(
                    constraint_type="mathematical",
                    priority="hard",
                    instruction=(
                        f"[SUSPENSE CONSTRAINT]: The causal graph shows "
                        f"opposing futures for the target entities. "
                        f"The threat probability exceeds the hope "
                        f"probability by {suspense_score:.2f}. You MUST "
                        f"maintain BOTH the sense of impending doom AND "
                        f"a sliver of hope. Do NOT resolve the tension "
                        f"in this scene."
                    ),
                    evidence={"suspense_score": suspense_score},
                ))

            # Layer 2: Epistemic gaps amplify uncertainty
            contradicted = [g for g in gaps if g.gap_type == "contradicted"]
            if contradicted:
                widest = max(contradicted, key=lambda g: g.gap_magnitude)
                constraints.append(ConstraintBlock(
                    constraint_type="epistemic",
                    priority="hard",
                    instruction=(
                        f"[EPISTEMIC CONSTRAINT]: {widest.entity_id} believes "
                        f"'{widest.believed_state}' about "
                        f"{widest.belief_target_id}, but the truth is "
                        f"'{widest.actual_state}'. You MUST NOT reveal "
                        f"the truth to this character. Write the scene so "
                        f"the reader feels the gap "
                        f"(magnitude={widest.gap_magnitude:.2f})."
                    ),
                    evidence={
                        "entity_id": widest.entity_id,
                        "belief_target_id": widest.belief_target_id,
                        "gap_magnitude": widest.gap_magnitude,
                    },
                ))
                for g in contradicted:
                    if (g.entity_id != widest.entity_id
                            or g.belief_target_id != widest.belief_target_id):
                        constraints.append(ConstraintBlock(
                            constraint_type="epistemic",
                            priority="soft",
                            instruction=(
                                f"{g.entity_id} also falsely believes "
                                f"'{g.believed_state}' about "
                                f"{g.belief_target_id}. Reinforce this "
                                f"ignorance."
                            ),
                            evidence={"gap_magnitude": g.gap_magnitude},
                        ))

            # Layer 3: Withheld events (narrative structure)
            withheld = [
                t for t in narrative_tensions
                if t.tension_type == "withheld_cause" and t.displacement > 0.1
            ]
            if withheld:
                most_displaced = max(withheld, key=lambda t: t.displacement)
                constraints.append(ConstraintBlock(
                    constraint_type="narrative",
                    priority="hard",
                    instruction=(
                        f"[NARRATIVE STRUCTURE]: The event "
                        f"'{most_displaced.event_id}' "
                        f"(fabula_time={most_displaced.fabula_time}) "
                        f"happened chronologically but is withheld from "
                        f"the reader until syuzhet_index="
                        f"{most_displaced.syuzhet_index} "
                        f"(displacement="
                        f"{most_displaced.displacement:+.2f}). "
                        f"You MUST NOT reference or spoil this event. "
                        f"The reader must not learn "
                        f"'{most_displaced.description}' yet."
                    ),
                    evidence={
                        "event_id": most_displaced.event_id,
                        "displacement": most_displaced.displacement,
                        "fabula_time": most_displaced.fabula_time,
                        "syuzhet_index": most_displaced.syuzhet_index,
                    },
                ))
                for t in withheld:
                    if t.event_id != most_displaced.event_id:
                        constraints.append(ConstraintBlock(
                            constraint_type="narrative",
                            priority="soft",
                            instruction=(
                                f"Also withheld: '{t.event_id}' "
                                f"(displacement="
                                f"{t.displacement:+.2f}). "
                                f"Do not reveal."
                            ),
                            evidence={
                                "event_id": t.event_id,
                                "displacement": t.displacement,
                            },
                        ))

            # Layer 4: Hidden information channels
            if hidden_channels:
                for hc in hidden_channels:
                    constraints.append(ConstraintBlock(
                        constraint_type="narrative",
                        priority="hard",
                        instruction=(
                            f"[HIDDEN CHANNEL]: A {hc.medium} link from "
                            f"{hc.source_id} to {hc.target_ids} exists but "
                            f"is not revealed until syuzhet_index="
                            f"{hc.discovered_at_syuzhet}. Do not reference "
                            f"it."
                        ),
                        evidence={
                            "source_id": hc.source_id,
                            "medium": hc.medium,
                            "discovered_at_syuzhet": hc.discovered_at_syuzhet,
                        },
                    ))

        # =============================================================
        # SURPRISE  (KL Divergence — prediction error)
        # =============================================================
        elif effect == "surprise":
            surprise_score = self.compute_surprise_score(
                entity_ids, syuzhet_anchor,
            )

            # Layer 1: KL divergence summary
            if surprise_score > 0:
                constraints.append(ConstraintBlock(
                    constraint_type="mathematical",
                    priority="hard",
                    instruction=(
                        f"[SURPRISE CONSTRAINT]: The prediction error "
                        f"between the reader's expectations and the "
                        f"actual revelation is {surprise_score:.2f} "
                        f"(normalised KL divergence). The reader's "
                        f"mental model must be forcefully updated. "
                        f"Maximise the shock of this moment."
                    ),
                    evidence={"surprise_kl_score": surprise_score},
                ))

            # Layer 2: Shatter contradicted beliefs
            high_confidence_wrong = [
                g for g in gaps
                if g.gap_type == "contradicted" and g.gap_magnitude >= 0.5
            ]
            for g in high_confidence_wrong:
                constraints.append(ConstraintBlock(
                    constraint_type="epistemic",
                    priority="hard",
                    instruction=(
                        f"[REVELATION]: Shatter {g.entity_id}'s belief "
                        f"that '{g.believed_state}'. The truth is "
                        f"'{g.actual_state}'. The revelation must land "
                        f"with intensity={intensity:.2f}."
                    ),
                    evidence={"gap_magnitude": g.gap_magnitude},
                ))

            # Layer 3: Flashback reveals
            upcoming = [
                t for t in narrative_tensions
                if t.tension_type == "withheld_cause"
                and t.displacement > 0.15
            ]
            if upcoming:
                biggest = max(upcoming, key=lambda t: t.displacement)
                constraints.append(ConstraintBlock(
                    constraint_type="narrative",
                    priority="hard",
                    instruction=(
                        f"[FLASHBACK REVEAL]: The event "
                        f"'{biggest.event_id}' "
                        f"(fabula_time={biggest.fabula_time}, "
                        f"displacement="
                        f"{biggest.displacement:+.2f}) is about to "
                        f"be revealed. This is a long-withheld cause — "
                        f"maximise the shock of revelation."
                    ),
                    evidence={
                        "event_id": biggest.event_id,
                        "displacement": biggest.displacement,
                    },
                ))

        # =============================================================
        # EMOTION effects (trait-shift)
        # =============================================================
        elif effect in ("grief", "rage", "joy", "fear", "love", "regret"):
            # Map effects to likely trait targets
            _effect_trait_map: Dict[str, List[str]] = {
                "grief": ["despair", "love", "hope"],
                "rage": ["anger", "rebelliousness", "resentment"],
                "joy": ["happiness", "hope", "contentment"],
                "fear": ["fear", "paranoia", "anxiety"],
                "love": ["love", "affection", "sensuality"],
                "regret": ["guilt", "remorse", "despair"],
            }
            # Traits that should DECREASE for each effect
            _effect_decrease: Dict[str, set] = {
                "grief": {"hope", "happiness", "contentment"},
                "rage": {"patience", "calm"},
                "joy": {"despair", "fear", "anxiety", "sadness"},
                "fear": {"courage", "hope", "calm"},
                "love": {"anger", "resentment"},
                "regret": {"happiness", "contentment", "hope"},
            }
            target_traits = _effect_trait_map.get(effect, [])
            decrease_set = _effect_decrease.get(effect, set())

            for traj in trajectories:
                if traj.trait_name in target_traits or not target_traits:
                    should_decrease = traj.trait_name in decrease_set
                    magnitude = intensity if intensity != 0 else 0.5
                    direction = -magnitude if should_decrease else magnitude
                    headroom = traj.headroom_down if should_decrease else traj.headroom_up
                    if headroom < 0.05:
                        continue  # Already at ceiling/floor
                    constraints.append(ConstraintBlock(
                        constraint_type="mathematical",
                        priority="hard",
                        instruction=(
                            f"[MATHEMATICAL CONSTRAINT]: {traj.entity_id}'s "
                            f"'{traj.trait_name}' is at {traj.current_value:.2f} "
                            f"(inertia={traj.inertia:.2f}, headroom={headroom:.2f}). "
                            f"The events of this scene MUST shift this trait by "
                            f"{direction:+.2f}. Focus the internal monologue on "
                            f"this psychological pivot."
                        ),
                        evidence={
                            "entity_id": traj.entity_id,
                            "trait": traj.trait_name,
                            "current": traj.current_value,
                            "inertia": traj.inertia,
                            "headroom": headroom,
                        },
                    ))

            # Also add relationship tension constraints if relevant
            for t in rel_tensions:
                if t.asymmetry_score > 0.3:
                    constraints.append(ConstraintBlock(
                        constraint_type="mathematical",
                        priority="soft",
                        instruction=(
                            f"The relationship {t.source_id}→{t.target_id} is "
                            f"asymmetric (score={t.asymmetry_score:.2f}): "
                            f"affinity={t.affinity:.2f}, fear={t.fear:.2f}, "
                            f"power={t.power_dynamic:.2f}. Use this tension."
                        ),
                        evidence={
                            "source_id": t.source_id,
                            "target_id": t.target_id,
                            "asymmetry": t.asymmetry_score,
                        },
                    ))

        # --- Specific vector target (from directive.target_vector_id) ---
        if directive.target_vector_id:
            vec_constraint = self._build_vector_constraint(
                directive.target_vector_id, intensity,
            )
            if vec_constraint:
                constraints.append(vec_constraint)

        # --- Physics override (split-screen) ---
        physics_override = self._detect_physics_override()

        if physics_override:
            constraints.append(ConstraintBlock(
                constraint_type="spatial",
                priority="hard",
                instruction=physics_override,
                evidence={},
            ))

        return CreativeBrief(
            target_effect=effect,
            target_entities=entity_ids,
            constraints=constraints,
            epistemic_gaps=gaps,
            narrative_tensions=narrative_tensions,
            hidden_channels=hidden_channels,
            trait_trajectories=trajectories,
            relationship_tensions=rel_tensions,
            physics_override=physics_override,
            scene_context=self.ego,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Look up an entity in the ego payload (focus or present)."""
        for ent in self.ego.get("focus_entities", []):
            if ent.get("id") == entity_id:
                return ent
        for ent in self.ego.get("present_entities", []):
            if ent.get("id") == entity_id:
                return ent
        return None

    def _resolve_actual_state(self, target_id: str) -> str:
        """Look up the objective state of a belief target in the world state."""
        # Entity status + key traits
        ent = self.world_state.entities.get(target_id)
        if ent:
            parts = [f"status={ent.status}"]
            if ent.location_id:
                parts.append(f"location={ent.location_id}")
            # Include traits with extreme values (far from 0.5 baseline)
            for tname, tv in ent.traits.items():
                if abs(tv.value - 0.5) >= 0.2:
                    parts.append(f"{tname}={tv.value:.2f}")
            return ", ".join(parts)

        # Object properties
        obj = self.world_state.objects.get(target_id)
        if obj:
            props = ", ".join(f"{k}={v}" for k, v in obj.properties.items())
            return f"owner={obj.owner_id}, {props}" if props else f"owner={obj.owner_id}"

        # Location ambient
        loc = self.world_state.locations.get(target_id)
        if loc:
            return f"location={loc.name}"

        # Event
        evt = next((e for e in self.world_state.events if e.id == target_id), None)
        if evt:
            return f"event_type={evt.event_type}, actors={evt.actor_ids}"

        return "unknown"

    def _classify_gap(
        self, believed: str, actual: str, confidence: float
    ) -> tuple[str, float]:
        """Classify the gap between believed and actual state.

        Returns (gap_type, gap_magnitude).
        """
        actual_lower = actual.lower()
        believed_lower = believed.lower()

        # Quantitative check: if both strings contain numeric values,
        # compute magnitude from their difference.
        actual_nums = re.findall(r"-?\d+(?:\.\d+)?", actual_lower)
        believed_nums = re.findall(r"-?\d+(?:\.\d+)?", believed_lower)
        if actual_nums and believed_nums:
            try:
                a_val = float(actual_nums[0])
                b_val = float(believed_nums[0])
                diff = abs(a_val - b_val)
                # Normalise magnitude: cap at 1.0 for unit-scale values,
                # otherwise use relative difference.
                denom = max(abs(a_val), abs(b_val), 1.0)
                magnitude = min(diff / denom, 1.0)
                logger.debug("[DirectiveAssembly·ClassifyGap] Quantitative branch: actual=%.3f believed=%.3f diff=%.3f mag=%.2f",
                             a_val, b_val, diff, magnitude)
                return "quantitative", round(magnitude, 2)
            except (ValueError, ZeroDivisionError):
                pass  # fall through to token-based heuristic

        # Quick heuristic: if the actual state keywords appear in the belief,
        # they're roughly aligned.
        actual_tokens = set(actual_lower.replace("=", " ").replace(",", " ").split())
        believed_tokens = set(believed_lower.replace("=", " ").replace(",", " ").split())

        overlap = actual_tokens & believed_tokens
        if len(overlap) >= len(actual_tokens) * 0.5 and actual_tokens:
            return "confirmed", round(1.0 - confidence, 2)

        # If essentially no overlap, they're contradicted
        if len(overlap) <= 1:
            return "contradicted", round(confidence, 2)

        return "unknown", 0.5

    def _build_vector_constraint(
        self, vector_target_id: str, intensity: float,
    ) -> Optional[ConstraintBlock]:
        """Build a constraint from a specific target_vector_id string."""
        if "." not in vector_target_id:
            return ConstraintBlock(
                constraint_type="mathematical",
                priority="hard",
                instruction=f"Alter the state of {vector_target_id} by {intensity:+.2f}.",
                evidence={},
            )

        node_id, vector_path = vector_target_id.split(".", 1)

        if vector_path.startswith("relationships."):
            parts = vector_path.split(".")
            if len(parts) != 3:
                return None
            _, target_entity, metric = parts
            current_val = "unknown"
            for rel in self.ego.get("relevant_relationships", []):
                if (rel.get("source_entity_id") == node_id
                        and rel.get("target_entity_id") == target_entity):
                    current_val = rel.get(metric, 0.0)
                    break
            return ConstraintBlock(
                constraint_type="mathematical",
                priority="hard",
                instruction=(
                    f"[MATHEMATICAL CONSTRAINT]: The {metric.upper()} between "
                    f"{node_id} and {target_entity} currently sits at {current_val}. "
                    f"You MUST write the prose such that this metric is forcefully "
                    f"shifted by {intensity:+.2f}."
                ),
                evidence={"node_id": node_id, "target": target_entity,
                          "metric": metric, "current": current_val},
            )

        if vector_path.startswith("traits."):
            trait_parts = vector_path.split(".")
            trait_name = trait_parts[1] if len(trait_parts) >= 2 else vector_path
            current_val = "unknown"
            ent_data = self._find_entity(node_id)
            if ent_data:
                current_val = ent_data.get("traits", {}).get(
                    trait_name, {}
                ).get("value", "unknown")
            return ConstraintBlock(
                constraint_type="mathematical",
                priority="hard",
                instruction=(
                    f"[MATHEMATICAL CONSTRAINT]: {node_id}'s internal "
                    f"'{trait_name}' trait currently sits at {current_val}. "
                    f"The events of this scene MUST shatter their inertia "
                    f"and shift this trait by {intensity:+.2f}."
                ),
                evidence={"node_id": node_id, "trait": trait_name,
                          "current": current_val},
            )

        if vector_path.startswith("beliefs."):
            parts = vector_path.split(".")
            belief_target = parts[1] if len(parts) >= 2 else "unknown"
            return ConstraintBlock(
                constraint_type="epistemic",
                priority="hard",
                instruction=(
                    f"[EPISTEMIC CONSTRAINT]: Force a realization. {node_id}'s "
                    f"confidence in their belief about {belief_target} must shift "
                    f"by {intensity:+.2f}. Shatter their current worldview."
                ),
                evidence={"node_id": node_id, "belief_target": belief_target},
            )

        return None

    def _detect_physics_override(self) -> Optional[str]:
        """Detect split-screen scenarios from the sandbox or ego payload."""
        # Prefer sandbox if available
        if self.sandbox is not None:
            entity_locs: set[str] = set()
            for _, data in self.sandbox.nodes(data=True):
                if data.get("node_type") == "Entity":
                    loc = data.get("location_id")
                    if loc:
                        entity_locs.add(loc)
            if len(entity_locs) <= 1:
                return None
            has_comms = any(
                d.get("edge_type") == "communicating_with"
                for _, _, d in self.sandbox.edges(data=True)
            )
            if has_comms:
                return (
                    "[PHYSICS OVERRIDE]: Characters are in SEPARATE locations "
                    "communicating remotely. You MUST NOT describe physical "
                    "touching, exchanging of items, or any direct physical "
                    "interaction. All interaction must be limited to the "
                    "communication medium."
                )
            return None

        # Fallback: infer from ego payload locations
        locs = {
            ent.get("location_id")
            for ent in self.ego.get("focus_entities", [])
            if ent.get("location_id")
        }
        locs |= {
            ent.get("location_id")
            for ent in self.ego.get("present_entities", [])
            if ent.get("location_id")
        }
        if len(locs) <= 1:
            return None
        return (
            "[PHYSICS OVERRIDE]: Characters span multiple locations. "
            "Constrain physical interaction to co-located characters only."
        )
