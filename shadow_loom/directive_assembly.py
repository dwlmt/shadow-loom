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
    rule3_pruned_interventions: List[str] = Field(
        default_factory=list,
        description=(
            "Intervention paths the AMWN ctf-calculus pre-flight (Rule 3) "
            "proved vacuous against ``target_node_ids`` for this candidate. "
            "A candidate whose only interventions are all rule3-pruned will "
            "be marked invalid because no physical surgery was applied."
        ),
    )
    rule2_redundant_evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Evidence nodes the ctf-calculus pre-flight (Rule 2) proved "
            "d-separated from this candidate's interventions on the AMWN."
        ),
    )


class ConstraintBlock(BaseModel):
    """A single instruction constraint for the drafting LLM."""
    constraint_type: Literal["mathematical", "epistemic", "spatial", "temporal", "narrative"]
    priority: Literal["hard", "soft"]
    instruction: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# Rendering directive models — carry effect-specific data the LLM needs
# =====================================================================

class RenderingDirective(BaseModel):
    """Specifies the exact stylistic and semantic actions the LLM must
    execute when rendering prose for a given query type + effect."""
    rendering_mode: str = Field(
        description=(
            "The rendering strategy: mystery, dramatic_irony, surprise, "
            "suspense, fear, joy, regret, grief, rage, love, "
            "observation, intervention, counterfactual, interrogation."
        ),
    )
    pov_lock: Optional[str] = Field(
        default=None,
        description="Entity ID to lock the narrative perspective to.",
    )
    pacing: Literal["dilated", "normal", "accelerated", "sharp_pivot"] = Field(
        default="normal",
        description="Temporal pacing of the prose.",
    )
    sensory_focus: Literal["wide", "normal", "tunnel", "absence"] = Field(
        default="normal",
        description=(
            "wide = expansive environment; tunnel = strip background, "
            "fixate on threat; absence = focus on what is missing."
        ),
    )
    tone_arc: Optional[str] = Field(
        default=None,
        description=(
            "A tonal trajectory description, e.g. "
            "'passive_sorrow → active_hostility' for rage."
        ),
    )
    stylistic_instructions: List[str] = Field(
        default_factory=list,
        description="Ordered list of specific prose instructions.",
    )


class CounterfactualBranch(BaseModel):
    """The actual vs. simulated outcome for regret rendering."""
    actual_outcome: str
    simulated_outcome: str
    divergence_event_id: Optional[str] = None
    divergence_description: Optional[str] = None


class ThreatProximity(BaseModel):
    """Threat information for fear/suspense rendering."""
    threat_event_id: Optional[str] = None
    threat_description: str = ""
    threat_probability: float = 0.5
    hope_probability: float = 0.5
    spatial_distance: Optional[int] = Field(
        default=None,
        description="Number of spatial hops between threat and target.",
    )
    damage_potential: float = Field(
        default=5.0,
        description="causal_force of the threat edge (0-10).",
    )


class CausalAttribution(BaseModel):
    """Who caused the loss — for rage rendering."""
    perpetrator_id: str
    perpetrator_name: Optional[str] = None
    loss_event_id: str
    loss_description: str
    causal_chain: List[str] = Field(
        default_factory=list,
        description="Ordered list of event IDs from perpetrator action to loss.",
    )


class EntanglementPair(BaseModel):
    """Structural coupling between two entities — for love rendering."""
    entity_a: str
    entity_b: str
    coupling_strength: float = Field(
        description="Normalised 0-1 coupling derived from relationship metrics.",
    )
    shared_location: bool = False


class InterventionMechanism(BaseModel):
    """How a do-operator state change must be physically rendered."""
    node_id: str
    old_state: str
    new_state: str
    mechanism_hint: str = Field(
        description=(
            "The physical/social mechanism that overcomes inertia, "
            "e.g. 'kinetic force on locked door', 'persuasion overcoming loyalty'."
        ),
    )
    inertia: float = 0.5


class AbductionTruth(BaseModel):
    """A hidden background variable inferred by Rung 3 abduction."""
    entity_id: str
    hidden_variable: str = Field(
        description="What must be true, e.g. 'stole the key yesterday'.",
    )
    weave_hint: str = Field(
        default="",
        description=(
            "How to surface this subtly in prose, e.g. "
            "'character reaches into pocket to feel the key'."
        ),
    )


class CreativeBrief(BaseModel):
    """Structured output ready for a downstream drafting LLM.

    Contains both the mathematical constraints (Step 9) and the
    rendering directives (Step 10 input) that control how the
    LLM translates the math into prose.
    """
    target_effect: str
    target_entities: List[str]
    original_query: Optional[str] = Field(
        default=None,
        description=(
            "The user's verbatim natural-language request that produced "
            "this brief. Surfaced to the generator and auditor so the "
            "rendered prose can honour the user's actual intent rather "
            "than only the engine's structured derivation."
        ),
    )
    constraints: List[ConstraintBlock] = Field(default_factory=list)
    epistemic_gaps: List[EpistemicGap] = Field(default_factory=list)
    narrative_tensions: List[NarrativeTension] = Field(default_factory=list)
    hidden_channels: List[HiddenInformationChannel] = Field(default_factory=list)
    trait_trajectories: List[TraitTrajectory] = Field(default_factory=list)
    relationship_tensions: List[RelationshipTension] = Field(default_factory=list)
    physics_override: Optional[str] = None
    scene_context: Dict[str, Any] = Field(default_factory=dict)

    # --- Rendering directives (Step 10 control layer) ---
    rendering: Optional[RenderingDirective] = None
    counterfactual_branch: Optional[CounterfactualBranch] = None
    threat_proximity: Optional[ThreatProximity] = None
    causal_attribution: Optional[CausalAttribution] = None
    entanglement_pairs: List[EntanglementPair] = Field(default_factory=list)
    intervention_mechanisms: List[InterventionMechanism] = Field(default_factory=list)
    abduction_truths: List[AbductionTruth] = Field(default_factory=list)


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

        # Strength-weighted mystery: each ancestor contributes its
        # *path strength* (product of edge weights along the strongest
        # path) so weak rumours don't count as much as eyewitness
        # causation. Falls back to 1.0 when no edge weight is available.
        total_mass = 0.0
        hidden_mass = 0.0

        for eff in effect_nodes:
            if not causal_g.has_node(eff):
                continue
            ancestors = nx.ancestors(causal_g, eff)
            if not ancestors:
                continue
            for anc in ancestors:
                # Strength of the strongest single-edge contribution from
                # this ancestor toward the effect (cheap proxy for path
                # strength; full path-product would be O(V*E) per query).
                if causal_g.has_edge(anc, eff):
                    w = causal_g[anc][eff].get("weight", 0.5)
                else:
                    # Multi-hop ancestor — use the max outgoing weight as
                    # an upper bound on its causal contribution.
                    out_ws = [
                        d.get("weight", 0.5)
                        for _, _, d in causal_g.out_edges(anc, data=True)
                    ]
                    w = max(out_ws) if out_ws else 0.5
                total_mass += w
                if anc not in revealed:
                    hidden_mass += w

        if total_mass == 0.0:
            return 0.0

        score = hidden_mass / total_mass
        logger.debug(
            "[DirectiveAssembly·Mystery] hidden_mass=%.3f / total_mass=%.3f = %.3f",
            hidden_mass, total_mass, score,
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

            # An entity is *aware* of an event when (a) they participate in
            # it (actor or target — direct experience), or (b) they are the
            # recipient of a revealed InformationEdge whose source carries
            # the event, or (c) they hold a Belief whose target_id matches
            # the event id. Belief.target_id is the *state* a character
            # believes about (entity/object/event), so events with a direct
            # belief entry are also counted.
            events_known_by_character: set[str] = {
                evt.id for evt in self.world_state.events
                if eid in evt.actor_ids or eid in evt.target_ids
            }
            for ie in self.world_state.information_topology:
                if ie.discovered_at_syuzhet > syuzhet_anchor:
                    continue
                if eid not in ie.target_ids:
                    continue
                # Treat the source of a revealed information edge as a
                # potential channel: if it names an event the character
                # learns about it.
                if ie.source_id.startswith("EVT_"):
                    events_known_by_character.add(ie.source_id)
            events_known_by_character |= {
                b.target_id for b in ent.beliefs
                if b.target_id.startswith("EVT_")
            }

            # Revealed causal edges whose *cause* is an event targeting
            # this entity. If the character has no awareness of that
            # event, the reader sees a threat/secret the character cannot.
            for ce in self.world_state.causal_topology:
                if ce.target_id != eid:
                    continue
                if not ce.source_id.startswith("EVT_"):
                    continue
                if ce.source_id not in revealed:
                    continue  # Reader doesn't know this either
                total_connections += 1
                if ce.source_id not in events_known_by_character:
                    irony_gaps += 1

            # Revealed information edges whose existence the character
            # cannot perceive (they are not a target). Counted as irony
            # only when the source is an event the reader has seen.
            for ie in self.world_state.information_topology:
                if ie.discovered_at_syuzhet > syuzhet_anchor:
                    continue
                if eid in ie.target_ids:
                    continue  # Character is on the channel — no asymmetry
                if not ie.source_id.startswith("EVT_") or ie.source_id not in revealed:
                    continue
                total_connections += 1
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

        # Noisy-OR aggregation: each unrevealed threat/hope event
        # contributes an independent failure probability ``1 - p_i``;
        # the combined probability is ``1 - ∏(1 - p_i)``. This means
        # multiple concurrent dangers compound rather than collapsing
        # to the single strongest one.
        threat_complement = 1.0
        hope_complement = 1.0

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

            prob = max(0.0, min(1.0, prob))

            # Classify: entity acted upon → threat; entity acting → hope
            if (set(evt.target_ids) & eid_set) and not (set(evt.actor_ids) & eid_set):
                threat_complement *= (1.0 - prob)
            elif set(evt.actor_ids) & eid_set:
                hope_complement *= (1.0 - prob)

        threat_prob = 1.0 - threat_complement
        hope_prob = 1.0 - hope_complement

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

        * **Prior** — starts from a corpus-marginal baseline (the average
          value of *that* trait across all entities in ``world_state``)
          rather than a hard-coded 0.5. The 0.5 default treats every
          trait as maximally uncertain, which is rarely true in practice
          (e.g. ``courage`` skews high in heroic casts; ``despair`` skews
          low). The marginal collapses to 0.5 only when the corpus is
          itself maximally split, otherwise it pulls the reader's
          expectation toward what the rest of the cast looks like.
          For each *revealed* causal edge targeting this entity we then
          apply a Bayesian-style additive update,
          ``prior += w_i · (actual - base_prior)`` (clipped to ``[ε, 1-ε]``),
          so the magnitude of the shift is tied directly to the edge
          weight rather than the previous geometric ``* 0.5`` halving.
        * **Posterior** — actual trait values from the sandbox
          (post-simulation) or world state (truth).

        Returns a normalised KL divergence in [0, 1].
        """
        if syuzhet_anchor is None:
            return 0.0  # reader knows everything → no surprise

        EPS = 0.01
        _STRENGTH_W = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}
        revealed = self._revealed_event_ids(syuzhet_anchor)

        # Pre-compute per-trait corpus marginals (average value of each
        # trait across every entity in the world state). This is the
        # "no causal evidence" baseline — what an uninformed reader
        # would guess given only knowledge of the world's overall trait
        # distribution.
        marginal_sum: Dict[str, float] = {}
        marginal_count: Dict[str, int] = {}
        for ent in self.world_state.entities.values():
            for tname, tdata in ent.traits.items():
                v = getattr(tdata, "value", None)
                if v is None:
                    continue
                marginal_sum[tname] = marginal_sum.get(tname, 0.0) + float(v)
                marginal_count[tname] = marginal_count.get(tname, 0) + 1

        def _trait_marginal(name: str) -> float:
            if marginal_count.get(name, 0) == 0:
                return 0.5
            return marginal_sum[name] / marginal_count[name]

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

                # Prior: start from the corpus marginal for this trait,
                # then accumulate additive Bayesian-style evidence from
                # revealed causal edges. ``prior = base + Σ w_i · (actual - base)``
                # is monotone in the number/strength of revealed edges
                # and clips cleanly into ``[ε, 1-ε]`` for the KL.
                base_prior = _trait_marginal(trait_name)
                prior_val = base_prior
                for ce in self.world_state.causal_topology:
                    if ce.target_id != eid:
                        continue
                    if ce.source_id not in revealed:
                        continue
                    w = _STRENGTH_W.get(ce.evidence_strength, 0.5)
                    prior_val += w * (actual_val - base_prior)
                # Explicit clipping so cumulative updates can't push the
                # prior outside the open unit interval used by the KL.
                prior_val = max(EPS, min(1 - EPS, prior_val))

                p = max(EPS, min(1 - EPS, actual_val))    # posterior
                q = prior_val                             # prior

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

        # --- Emotion effects (distance-to-target, NOT remaining headroom) ---
        # The previous implementation rewarded ``headroom_up`` for the
        # increase set, but ``headroom_up = 1 - current_value`` is the
        # ROOM still available to grow toward saturation — i.e. the
        # distance from the trait to its target (1.0). Subtracting that
        # from the score made an entity at value=0.0 score "best" for
        # grief/rage/etc., the exact opposite of what the affective
        # loss should reward. We now compute *closeness to target*:
        #   • increase set: target = 1.0, closeness = current_value
        #   • decrease set: target = 0.0, closeness = 1 - current_value
        # so an entity already saturated in the right direction yields
        # ``score ≈ -1`` (strong match) and an entity stuck on the wrong
        # side yields ``score ≈ 0`` (no match).
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
                # Closeness of each trait to its per-effect target.
                avg_match = sum(
                    (1.0 - traj.current_value) if traj.trait_name in decrease_set
                    else traj.current_value
                    for traj in relevant
                ) / len(relevant)
                score -= avg_match
                logger.debug("[DirectiveAssembly·AffectiveScore] effect=%s avg_match=%.3f score=%.4f",
                             target_effect, avg_match, score)
            else:
                # No traits in the entity match the per-effect target
                # set — we have nothing to score against. Treat this as
                # the *worst possible* match (``+1.0``) rather than the
                # midpoint (``+0.5``) so the loss is on the same scale
                # as the success path: structural effects subtract a
                # value in ``[0, 1]`` and emotion successes subtract
                # a closeness in ``[0, 1]``, giving a best-case score
                # of ``-1.0``. The fallback is the symmetric worst-case
                # of ``+1.0`` rather than a half-step that silently
                # ranked an unmeasurable trait set above genuinely
                # bad matches.
                score += 1.0

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
        from shadow_loom.amwn import build_causal_diagram

        if self.sandbox is None:
            raise ValueError(
                "evaluate_candidate_events requires a live sandbox. "
                "Pass one via DirectiveAssembler(sandbox=..., ...)."
            )

        # Build the static causal diagram ONCE — it depends only on
        # ``self.world_state`` which is invariant across candidates, and
        # the AMWN pre-flight inside each ``engine.execute()`` call would
        # otherwise rebuild it K times for K candidates.
        shared_diagram = build_causal_diagram(self.world_state)

        results: List[CandidateResult] = []

        for candidate in candidates:
            forked = deepcopy(self.sandbox)
            engine = CausalPhysicsEngine(forked, self.world_state)
            physics = engine.execute(
                rung=2,
                interventions=candidate,
                target_node_ids=entity_ids,
                causal_diagram=shared_diagram,
            )

            # --- Pruning: intervention failed AND all propagations blocked ---
            # A candidate is physically impossible only if the surgery itself
            # was blocked (no intervened nodes) and no mutations occurred.
            surgery_applied = len(physics.intervened_nodes) > 0
            has_mutations = len(physics.mutations) > 0
            all_blocked = len(physics.blocked) > 0 and not has_mutations
            # Rule-3 (ctf-calculus exclusion) can drop every intervention
            # silently before the do-surgery runs *when the engine is
            # configured to prune* (``physics.rule3_pruning_mode ==
            # "prune"``). Without an explicit check the candidate would
            # otherwise look "valid" against the untouched sandbox. In
            # advisory mode (the default) the surgery is still applied,
            # so ``surgery_applied`` will be True and this branch
            # naturally does not fire.
            all_rule3_pruned = (
                physics.rule3_pruning_mode == "prune"
                and len(physics.rule3_pruned_interventions) >= len(candidate)
                and not surgery_applied
                and not has_mutations
            )

            if (not surgery_applied and all_blocked) or all_rule3_pruned:
                reasons = [b.reason for b in physics.blocked]
                if all_rule3_pruned:
                    reasons.append("rule3_vacuous")
                logger.debug("[DirectiveAssembly·Candidate] PRUNED: no surgery applied. reasons=%s rule3=%s",
                             reasons, physics.rule3_pruned_interventions)
                results.append(CandidateResult(
                    interventions=candidate,
                    valid=False,
                    blocked_reasons=reasons,
                    rule3_pruned_interventions=list(physics.rule3_pruned_interventions),
                    rule2_redundant_evidence=list(physics.rule2_redundant_evidence),
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
                rule3_pruned_interventions=list(physics.rule3_pruned_interventions),
                rule2_redundant_evidence=list(physics.rule2_redundant_evidence),
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
        # USER INTENT  (verbatim NL request as a HARD constraint)
        # =============================================================
        # The user's natural-language request is the highest-priority
        # signal we have about what the scene must accomplish. We lift
        # it into a ``hard`` ConstraintBlock so:
        #   1) it appears in both the rendering prompt and the audit
        #      prompt (both consume ``brief.constraints`` directly),
        #   2) the LLM auditor can flag prose that ignores it as a
        #      typed violation rather than only a soft hint,
        #   3) it survives any future serialisation of the brief
        #      (constraints are first-class state).
        original_query = (getattr(directive, "original_query", None) or "").strip()
        if original_query:
            constraints.append(ConstraintBlock(
                constraint_type="narrative",
                priority="hard",
                instruction=(
                    f"[USER INTENT \u2014 verbatim]: {original_query}\n"
                    "The rendered scene MUST address the user's request "
                    "above. Engine-derived constraints below are "
                    "guard-rails, not substitutes for the user's intent."
                ),
                evidence={"original_query": original_query},
            ))

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
            # but the character has no perceptual access to the source
            # event. Mirrors the awareness model used by
            # ``compute_dramatic_irony_score`` — a character "knows" an
            # event only when (a) they participated in it, (b) they
            # received a revealed information edge whose source is the
            # event, or (c) they hold an explicit belief whose target
            # IS the event id. Comparing event IDs against
            # ``{b.target_id for b in ent.beliefs}`` directly (the
            # previous implementation) was wrong: belief targets are
            # almost always entity / object / world ids, never EVT_*,
            # so every revealed causal edge looked like irony and
            # saturated the constraint list.
            irony_details: List[tuple] = []
            for eid in entity_ids:
                ent = self.world_state.entities.get(eid)
                if not ent:
                    continue
                events_known_by_character: set[str] = {
                    evt.id for evt in self.world_state.events
                    if eid in evt.actor_ids or eid in evt.target_ids
                }
                for ie in self.world_state.information_topology:
                    if syuzhet_anchor is not None and ie.discovered_at_syuzhet > syuzhet_anchor:
                        continue
                    if eid not in ie.target_ids:
                        continue
                    if ie.source_id.startswith("EVT_"):
                        events_known_by_character.add(ie.source_id)
                events_known_by_character |= {
                    b.target_id for b in ent.beliefs
                    if b.target_id.startswith("EVT_")
                }
                for ce in self.world_state.causal_topology:
                    if ce.target_id != eid:
                        continue
                    if not ce.source_id.startswith("EVT_"):
                        continue
                    if ce.source_id not in revealed:
                        continue
                    if ce.source_id not in events_known_by_character:
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

        # =============================================================
        # Build RenderingDirective + effect-specific payloads
        # =============================================================
        rendering, counterfactual_branch, threat_proximity = None, None, None
        causal_attribution = None
        entanglement_pairs: List[EntanglementPair] = []

        pov_entity = entity_ids[0] if entity_ids else None

        if effect == "mystery":
            rendering = RenderingDirective(
                rendering_mode="mystery",
                pov_lock=pov_entity,
                pacing="normal",
                sensory_focus="wide",
                stylistic_instructions=[
                    "Focus heavily on sensory details and the aftermath of events.",
                    "Portray the characters' confusion and initial processing of the scene.",
                    "Suppress any omniscient narration that might hint at hidden causal ancestors.",
                    "Lock the prose strictly to the focal character's limited perspective.",
                    "Describe effects without causes — the reader must feel the weight of the unknown.",
                ],
            )

        elif effect == "dramatic_irony":
            rendering = RenderingDirective(
                rendering_mode="dramatic_irony",
                pov_lock=pov_entity,
                pacing="normal",
                sensory_focus="normal",
                stylistic_instructions=[
                    "Juxtapose the character's naive internal monologue against the looming threat the reader knows about.",
                    "Generate prose where the character feels a false sense of security.",
                    "Show the character making plans based on incomplete information.",
                    "Maximise the emotional friction between the reader's knowledge and the character's ignorance.",
                    "The character MUST NOT learn the truth during this scene.",
                ],
            )

        elif effect == "surprise":
            rendering = RenderingDirective(
                rendering_mode="surprise",
                pov_lock=pov_entity,
                pacing="sharp_pivot",
                sensory_focus="normal",
                tone_arc="comfortable_flow → abrupt_shock",
                stylistic_instructions=[
                    "Begin with flowing, comfortable prose that lulls the reader into the expected outcome.",
                    "Telegraph the reader's prior expectation through character thoughts and environmental cues.",
                    "At the moment of revelation, execute a sharp syntactical pivot.",
                    "Use a short, blunt sentence to reveal the hidden truth.",
                    "Force an immediate update to the reader's mental model — maximise prediction error.",
                ],
            )

        elif effect == "suspense":
            # Compute threat/hope for ThreatProximity
            suspense_data = self._compute_threat_hope_detail(entity_ids, syuzhet_anchor)
            threat_proximity = suspense_data

            rendering = RenderingDirective(
                rendering_mode="suspense",
                pov_lock=pov_entity,
                pacing="dilated",
                sensory_focus="normal",
                stylistic_instructions=[
                    "Dilate time — slow the pacing obsessively.",
                    "Focus on the mechanical, step-by-step progression of the threat (footsteps, ticking clocks, closing distance).",
                    "Keep the 'hopeful' escape route visible in the prose but physically just out of reach.",
                    "Force the reader to agonize over the closing window of opportunity.",
                    "Do NOT resolve the tension in this scene — maintain both doom and hope.",
                ],
            )

        elif effect == "fear":
            fear_data = self._compute_threat_hope_detail(entity_ids, syuzhet_anchor)
            threat_proximity = fear_data

            rendering = RenderingDirective(
                rendering_mode="fear",
                pov_lock=pov_entity,
                pacing="accelerated",
                sensory_focus="tunnel",
                stylistic_instructions=[
                    "Simulate tunnel vision — as the threat closes, strip away background descriptions.",
                    "Focus entirely on the imminent danger and the protagonist's visceral physiological reactions.",
                    "Describe racing heart, paralysis, shallow breathing, adrenaline.",
                    "Collapse the prose's descriptive scope as distance to the threat shrinks.",
                    "The environment fades; only the threat and the body's response remain.",
                ],
            )

        elif effect == "joy":
            rendering = RenderingDirective(
                rendering_mode="joy",
                pov_lock=pov_entity,
                pacing="normal",
                sensory_focus="wide",
                stylistic_instructions=[
                    "Reverse the tunnel vision of fear — expand the prose outward.",
                    "Describe the environment in brighter, broader, more vivid terms.",
                    "Focus on the physiological sensation of relief: unclenching muscles, deep breath, warmth.",
                    "Show the sudden opening of new, positive future pathways.",
                    "If a threat node was just eliminated, contrast the silence left behind with the character's flooding relief.",
                ],
            )

        elif effect == "regret":
            # Build counterfactual branch data
            counterfactual_branch = self._build_counterfactual_branch(entity_ids)

            rendering = RenderingDirective(
                rendering_mode="regret",
                pov_lock=pov_entity,
                pacing="dilated",
                sensory_focus="normal",
                tone_arc="harsh_reality ↔ agonizing_visualization",
                stylistic_instructions=[
                    "Weave the counterfactual graph directly into the character's internal monologue.",
                    "The prose MUST explicitly articulate 'if only...' logic.",
                    "Contrast the harsh sensory reality of the present with the character's visualization of the alternate timeline.",
                    "Do NOT simply state the character is sad — render the specific alternate path they failed to choose.",
                    "Alternate between the bleak present and the imagined better world, each making the other more painful.",
                ],
            )

        elif effect == "grief":
            rendering = RenderingDirective(
                rendering_mode="grief",
                pov_lock=pov_entity,
                pacing="dilated",
                sensory_focus="absence",
                stylistic_instructions=[
                    "Focus on ABSENCE — describe the physical space left behind by the lost entity.",
                    "Use fragmented or numb prose reflecting the system's loss of a structural pillar.",
                    "Render the silence where a voice used to be, the empty chair, the cold side of the bed.",
                    "The character's ego-graph has lost a central node — reflect this structural collapse in the prose's coherence.",
                    "Short sentences. Disconnected observations. The world feels wrong.",
                ],
            )

        elif effect == "rage":
            # Build causal attribution — who caused the loss
            causal_attribution = self._build_causal_attribution(entity_ids)

            rendering = RenderingDirective(
                rendering_mode="rage",
                pov_lock=pov_entity,
                pacing="accelerated",
                sensory_focus="tunnel",
                tone_arc="passive_sorrow → active_targeted_hostility",
                stylistic_instructions=[
                    "Execute a tonal shift from passive sorrow to active, targeted hostility.",
                    "The prose accelerates as focus narrows obsessively onto the perpetrator.",
                    "Reflect the character marshaling their damage_potential trait vectors.",
                    "Show the character preparing to initiate a retaliatory causal chain.",
                    "The grief doesn't disappear — it transmutes into directed kinetic energy.",
                ],
            )

        elif effect == "love":
            entanglement_pairs = self._build_entanglement_pairs(entity_ids)

            rendering = RenderingDirective(
                rendering_mode="love",
                pov_lock=pov_entity,
                pacing="normal",
                sensory_focus="normal",
                stylistic_instructions=[
                    "Demonstrate structural entanglement through mirrored reactions.",
                    "If Entity A takes a hit, Entity B reacts instantly — prioritizing A's safety over their own.",
                    "Highlight shared physical and emotional proximity.",
                    "Show harm-to-A equaling harm-to-B through the coupled entity's involuntary response.",
                    "Render the entanglement through action, not declaration — show, never tell.",
                ],
            )

        brief = CreativeBrief(
            target_effect=effect,
            target_entities=entity_ids,
            original_query=getattr(directive, "original_query", None),
            constraints=constraints,
            epistemic_gaps=gaps,
            narrative_tensions=narrative_tensions,
            hidden_channels=hidden_channels,
            trait_trajectories=trajectories,
            relationship_tensions=rel_tensions,
            physics_override=physics_override,
            scene_context=self.ego,
            rendering=rendering,
            counterfactual_branch=counterfactual_branch,
            threat_proximity=threat_proximity,
            causal_attribution=causal_attribution,
            entanglement_pairs=entanglement_pairs,
        )
        _log_creative_brief(brief)
        return brief

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

    # ------------------------------------------------------------------
    # Rendering-specific data builders
    # ------------------------------------------------------------------
    def _compute_threat_hope_detail(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> ThreatProximity:
        """Build a ThreatProximity payload for fear/suspense rendering."""
        causal_g = self._build_causal_digraph()
        revealed = self._revealed_event_ids(syuzhet_anchor)
        all_evt_ids = {e.id for e in self.world_state.events}
        unrevealed = all_evt_ids - revealed
        eid_set = set(entity_ids)

        best_threat_id: Optional[str] = None
        best_threat_desc = ""
        threat_prob = 0.0
        hope_prob = 0.0
        best_force = 0.0  # will be set from actual causal_force

        for evt_id in unrevealed:
            evt = next(
                (e for e in self.world_state.events if e.id == evt_id), None,
            )
            if not evt:
                continue
            if not (set(evt.actor_ids) & eid_set) and not (set(evt.target_ids) & eid_set):
                continue

            prob = 0.5
            force = 0.0
            if causal_g.has_node(evt_id):
                in_edges = list(causal_g.in_edges(evt_id, data=True))
                if in_edges:
                    prob = max(d.get("weight", 0.5) for _, _, d in in_edges)

            # Get max causal force of edges targeting the event
            for ce in self.world_state.causal_topology:
                if ce.target_id == evt_id:
                    force = max(force, ce.causal_force)

            if (set(evt.target_ids) & eid_set) and not (set(evt.actor_ids) & eid_set):
                if prob > threat_prob:
                    threat_prob = prob
                    best_threat_id = evt.id
                    best_threat_desc = evt.description
                    best_force = force
            elif set(evt.actor_ids) & eid_set:
                hope_prob = max(hope_prob, prob)

        # Compute spatial distance from threat to entity
        spatial_dist = None
        if best_threat_id and self.sandbox is not None:
            # Try to find the threat event's location and the entity's location
            threat_node = self.sandbox.nodes.get(best_threat_id, {})
            entity_node = self.sandbox.nodes.get(entity_ids[0], {}) if entity_ids else {}
            threat_loc = threat_node.get("location_id")
            entity_loc = entity_node.get("location_id")
            if threat_loc and entity_loc and threat_loc != entity_loc:
                # Build traversable spatial subgraph
                spatial_g = nx.Graph()
                for u, v, d in self.sandbox.edges(data=True):
                    if d.get("edge_type") == "connected_to" and not d.get("is_locked", False):
                        spatial_g.add_edge(u, v)
                try:
                    spatial_dist = nx.shortest_path_length(spatial_g, threat_loc, entity_loc)
                except nx.NetworkXNoPath:
                    spatial_dist = None

        return ThreatProximity(
            threat_event_id=best_threat_id,
            threat_description=best_threat_desc,
            threat_probability=round(threat_prob, 3),
            hope_probability=round(hope_prob, 3),
            spatial_distance=spatial_dist,
            damage_potential=best_force,
        )

    def _build_counterfactual_branch(
        self,
        entity_ids: List[str],
    ) -> Optional[CounterfactualBranch]:
        """Build the actual vs. simulated outcome for regret rendering.

        Uses the most recent negative event targeting the entities as the
        'actual outcome', and looks for the most recent choice event by the
        entities as the divergence point whose alternate path would have
        led to a better state.
        """
        eid_set = set(entity_ids)

        # Build a set of event IDs that have negative causal effects
        # (trait_delta < 0 on outgoing mutation edges).  Events with no
        # mutation edges are treated as *possibly* negative (we include them
        # to avoid false negatives when extraction didn't annotate deltas).
        negative_event_ids: set[str] = set()
        events_with_deltas: set[str] = set()
        for ce in self.world_state.causal_topology:
            if ce.trait_delta is not None:
                events_with_deltas.add(ce.source_id)
                if ce.trait_delta < 0:
                    negative_event_ids.add(ce.source_id)

        def _is_likely_negative(evt_id: str) -> bool:
            if evt_id in negative_event_ids:
                return True
            # If the event has no delta annotations at all, include it
            # conservatively (extraction may not have annotated deltas).
            return evt_id not in events_with_deltas

        # Find the most recent negative outcome targeting our entities
        negative_events = [
            e for e in sorted(self.world_state.events, key=lambda x: x.fabula_time, reverse=True)
            if e.event_type == "outcome" and (set(e.target_ids) & eid_set) and _is_likely_negative(e.id)
        ]
        if not negative_events:
            return None

        actual_evt = negative_events[0]

        # Find the most recent choice by our entities that preceded the outcome
        choices = [
            e for e in sorted(self.world_state.events, key=lambda x: x.fabula_time, reverse=True)
            if e.event_type == "choice"
            and (set(e.actor_ids) & eid_set)
            and e.fabula_time <= actual_evt.fabula_time
        ]
        divergence_evt = choices[0] if choices else None

        return CounterfactualBranch(
            actual_outcome=actual_evt.description,
            simulated_outcome=(
                f"If {divergence_evt.id} had gone differently, "
                f"the outcome '{actual_evt.id}' might have been averted."
                if divergence_evt
                else "An alternate choice might have prevented this outcome."
            ),
            divergence_event_id=divergence_evt.id if divergence_evt else None,
            divergence_description=divergence_evt.description if divergence_evt else None,
        )

    def _build_causal_attribution(
        self,
        entity_ids: List[str],
    ) -> Optional[CausalAttribution]:
        """For rage: trace the causal chain from a loss back to a perpetrator."""
        eid_set = set(entity_ids)
        causal_g = self._build_causal_digraph()

        # Identify events with negative causal effects (trait_delta < 0).
        negative_event_ids: set[str] = set()
        events_with_deltas: set[str] = set()
        for ce in self.world_state.causal_topology:
            if ce.trait_delta is not None:
                events_with_deltas.add(ce.source_id)
                if ce.trait_delta < 0:
                    negative_event_ids.add(ce.source_id)

        def _is_likely_negative(evt_id: str) -> bool:
            if evt_id in negative_event_ids:
                return True
            return evt_id not in events_with_deltas

        # Find loss events — negative outcomes targeting our entities
        loss_events = [
            e for e in sorted(self.world_state.events, key=lambda x: x.fabula_time, reverse=True)
            if e.event_type == "outcome" and (set(e.target_ids) & eid_set) and _is_likely_negative(e.id)
        ]
        if not loss_events:
            return None

        loss_evt = loss_events[0]

        # Trace back through the causal graph to find the originating actor
        if not causal_g.has_node(loss_evt.id):
            # Fallback: use the actor of the loss event itself
            if loss_evt.actor_ids:
                perp_id = loss_evt.actor_ids[0]
                perp_ent = self.world_state.entities.get(perp_id)
                return CausalAttribution(
                    perpetrator_id=perp_id,
                    perpetrator_name=perp_ent.name if perp_ent else None,
                    loss_event_id=loss_evt.id,
                    loss_description=loss_evt.description,
                    causal_chain=[loss_evt.id],
                )
            return None

        # Walk ancestors to find the originating entity (choice-maker)
        # Sort ancestors by graph distance (descending) to find the most
        # upstream perpetrator — the root cause, not a proximate relay.
        ancestors = nx.ancestors(causal_g, loss_evt.id)
        ancestors_by_depth: List[str] = []
        for anc_id in ancestors:
            try:
                dist = nx.shortest_path_length(causal_g, anc_id, loss_evt.id)
            except nx.NetworkXNoPath:
                dist = 0
            ancestors_by_depth.append((dist, anc_id))
        ancestors_by_depth.sort(reverse=True)  # most upstream first

        perpetrator_id: Optional[str] = None
        causal_chain: List[str] = []

        for _, anc_id in ancestors_by_depth:
            anc_evt = next(
                (e for e in self.world_state.events if e.id == anc_id), None,
            )
            if anc_evt and anc_evt.event_type == "choice" and anc_evt.actor_ids:
                # Found an entity who made a choice upstream of the loss
                actor = anc_evt.actor_ids[0]
                if actor not in eid_set:  # perpetrator is someone ELSE
                    perpetrator_id = actor
                    # Build the causal chain path
                    try:
                        path = nx.shortest_path(causal_g, anc_id, loss_evt.id)
                        causal_chain = path
                    except nx.NetworkXNoPath:
                        causal_chain = [anc_id, loss_evt.id]
                    break

        if not perpetrator_id and loss_evt.actor_ids:
            perpetrator_id = loss_evt.actor_ids[0]
            causal_chain = [loss_evt.id]

        if not perpetrator_id:
            return None

        perp_ent = self.world_state.entities.get(perpetrator_id)
        return CausalAttribution(
            perpetrator_id=perpetrator_id,
            perpetrator_name=perp_ent.name if perp_ent else None,
            loss_event_id=loss_evt.id,
            loss_description=loss_evt.description,
            causal_chain=causal_chain,
        )

    def _build_entanglement_pairs(
        self,
        entity_ids: List[str],
    ) -> List[EntanglementPair]:
        """For love: find structurally entangled entity pairs."""
        pairs: List[EntanglementPair] = []
        eid_set = set(entity_ids)

        for rel in self.ego.get("relevant_relationships", []):
            src = rel.get("source_entity_id", "")
            tgt = rel.get("target_entity_id", "")
            if src not in eid_set and tgt not in eid_set:
                continue

            affinity = rel.get("affinity", 0.0)
            if affinity <= 0.3:
                continue  # Not a love-candidate

            # Look for the reverse edge to compute coupling
            reverse_aff = 0.0
            for rev in self.ego.get("relevant_relationships", []):
                if rev.get("source_entity_id") == tgt and rev.get("target_entity_id") == src:
                    reverse_aff = rev.get("affinity", 0.0)
                    break

            # Coupling strength: geometric mean of mutual affinity
            coupling = (abs(affinity) * abs(reverse_aff)) ** 0.5 if reverse_aff > 0 else abs(affinity) * 0.5

            # Check co-location
            shared_loc = False
            src_data = self._find_entity(src)
            tgt_data = self._find_entity(tgt)
            if src_data and tgt_data:
                shared_loc = (
                    src_data.get("location_id") is not None
                    and src_data.get("location_id") == tgt_data.get("location_id")
                )

            pairs.append(EntanglementPair(
                entity_a=src,
                entity_b=tgt,
                coupling_strength=round(coupling, 3),
                shared_location=shared_loc,
            ))

        # Sort by coupling strength descending
        pairs.sort(key=lambda p: p.coupling_strength, reverse=True)
        return pairs

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


# =====================================================================
# Readable summary logger for CreativeBrief
# =====================================================================

def _log_creative_brief(brief: "CreativeBrief", *, max_items: int = 10) -> None:
    """Emit a multi-line, human-readable INFO summary of a CreativeBrief."""
    if not logger.isEnabledFor(logging.INFO):
        return

    lines: list[str] = []
    lines.append(
        f"[DirectiveAssembly·CreativeBrief] effect={brief.target_effect} "
        f"entities={brief.target_entities}"
    )
    if brief.original_query:
        lines.append(f"  Original query: {brief.original_query}")

    if brief.constraints:
        lines.append("  Constraints:")
        for c in brief.constraints[:max_items]:
            lines.append(f"    - {c}")
        if len(brief.constraints) > max_items:
            lines.append(f"    … (+{len(brief.constraints) - max_items} more)")

    if brief.epistemic_gaps:
        lines.append(f"  Epistemic gaps ({len(brief.epistemic_gaps)}):")
        for g in brief.epistemic_gaps[:max_items]:
            lines.append(f"    · {g}")
        if len(brief.epistemic_gaps) > max_items:
            lines.append(
                f"    … (+{len(brief.epistemic_gaps) - max_items} more)"
            )

    if brief.narrative_tensions:
        lines.append(f"  Narrative tensions ({len(brief.narrative_tensions)}):")
        for t in brief.narrative_tensions[:max_items]:
            lines.append(f"    · {t}")
        if len(brief.narrative_tensions) > max_items:
            lines.append(
                f"    … (+{len(brief.narrative_tensions) - max_items} more)"
            )

    if brief.hidden_channels:
        lines.append(f"  Hidden channels ({len(brief.hidden_channels)}):")
        for h in brief.hidden_channels[:max_items]:
            lines.append(f"    · {h}")

    if brief.trait_trajectories:
        lines.append(
            f"  Trait trajectories ({len(brief.trait_trajectories)}):"
        )
        for tr in brief.trait_trajectories[:max_items]:
            lines.append(f"    · {tr}")

    if brief.relationship_tensions:
        lines.append(
            f"  Relationship tensions ({len(brief.relationship_tensions)}):"
        )
        for rt in brief.relationship_tensions[:max_items]:
            lines.append(f"    · {rt}")

    if brief.physics_override:
        lines.append(f"  Physics override: {brief.physics_override}")

    if brief.counterfactual_branch:
        lines.append(
            f"  Counterfactual branch: {brief.counterfactual_branch}"
        )

    if brief.threat_proximity:
        lines.append(f"  Threat proximity: {brief.threat_proximity}")

    if brief.causal_attribution:
        lines.append(f"  Causal attribution: {brief.causal_attribution}")

    if brief.entanglement_pairs:
        lines.append(
            f"  Entanglement pairs ({len(brief.entanglement_pairs)}):"
        )
        for ep in brief.entanglement_pairs[:max_items]:
            lines.append(f"    · {ep}")

    if brief.rendering:
        lines.append("  Rendering directives:")
        for r in brief.rendering[:max_items]:
            lines.append(f"    → {r}")
        if len(brief.rendering) > max_items:
            lines.append(
                f"    … (+{len(brief.rendering) - max_items} more)"
            )

    logger.info("\n".join(lines))
