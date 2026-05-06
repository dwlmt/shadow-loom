# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

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
from typing import Any, Dict, List, Optional, Literal, Tuple

import networkx as nx
from pydantic import BaseModel, Field

from shadow_loom.models import WorldStateV1, NarrativeStyle, reconstruct_entity_at
from shadow_loom.query_models import DirectiveQuery
from shadow_loom.settings import get_settings as _get_settings

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


class HiddenChannel(BaseModel):
    """A communication signal whose existence is not yet on-page for the reader.

    Represents one of two cases:

    * **Hidden Channel** (``kind='channel'``): a standing :class:`Channel`
      capability that exists in the world but has not yet been disclosed
      via any on-page utterance with ``via_channel_id == channel.id``
      and ``syuzhet_index <= anchor``.
    * **Hidden Utterance** (``kind='utterance'``): a discrete
      ``EventNode(event_type='utterance')`` with ``syuzhet_index >
      anchor`` (the message itself happens later in narration order).
    """
    kind: Literal["channel", "utterance"]
    channel_id: Optional[str] = None
    utterance_event_id: Optional[str] = None
    medium: str
    participant_ids: List[str] = Field(default_factory=list)
    addressee_ids: List[str] = Field(default_factory=list)
    speaker_id: Optional[str] = None
    discovered_at_syuzhet: Optional[int] = Field(
        default=None,
        description=(
            "For ``kind='utterance'``, the syuzhet_index of the utterance "
            "event itself. For ``kind='channel'``, the syuzhet_index of "
            "the earliest utterance via this channel (None if no "
            "utterance ever surfaces)."
        ),
    )
    unintelligible_for: List[str] = Field(
        default_factory=list,
        description=(
            "Entity IDs whose per-recipient intelligibility on this "
            "channel is below the configured threshold. Even if the "
            "channel is on-page, these listeners cannot reliably parse "
            "its content — a fertile source of dramatic irony."
        ),
    )


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
    vacuous: bool = Field(
        default=False,
        description=(
            "True when ctf-calculus Rule 3 (Exclusion) proved this "
            "do-surgery has no directed path to the user's target nodes "
            "on the AMWN. The local change still occurs, but the "
            "renderer must not invent downstream causal ripples for it."
        ),
    )
    advisory: bool = Field(
        default=False,
        description=(
            "True when Rule 3 flagged the surgery as advisory-only "
            "(extracted topology may be missing latent confounders). "
            "Render downstream consequences cautiously — favour "
            "atmospheric echoes over loud causal chains."
        ),
    )


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


class ResearchHighlight(BaseModel):
    """One ``WorldFact`` rendered as background context for a brief."""
    fact_id: str
    topic: str
    summary: str
    confidence: str = Field(default="moderate")
    source_url_primary: Optional[str] = None


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
    hidden_channels: List[HiddenChannel] = Field(default_factory=list)
    trait_trajectories: List[TraitTrajectory] = Field(default_factory=list)
    relationship_tensions: List[RelationshipTension] = Field(default_factory=list)
    physics_override: Optional[str] = None
    scene_context: Dict[str, Any] = Field(default_factory=dict)

    # --- External research (segregated; off unless populated) ---
    external_research: List["ResearchHighlight"] = Field(
        default_factory=list,
        description=(
            "Optional ``WorldFact`` snippets selected by the assembler "
            "for the current scene focus. Carried as plain background "
            "context the renderer may consult; never authoritative for "
            "character traits, beliefs, events or world traits."
        ),
    )

    # --- Rendering directives (Step 10 control layer) ---
    rendering: Optional[RenderingDirective] = None
    counterfactual_branch: Optional[CounterfactualBranch] = None
    threat_proximity: Optional[ThreatProximity] = None
    causal_attribution: Optional[CausalAttribution] = None
    entanglement_pairs: List[EntanglementPair] = Field(default_factory=list)
    intervention_mechanisms: List[InterventionMechanism] = Field(default_factory=list)
    abduction_truths: List[AbductionTruth] = Field(default_factory=list)

    # --- AMWN branch context (Story-integration plan, Step 4) ---
    branch_world_id: Literal["factual", "shadow"] = Field(
        default="factual",
        description=(
            "Which AMWN branch this brief is being rendered onto. "
            "'factual' = the prose extends the canonical mainline; "
            "'shadow' = the prose lives on a counterfactual fork. The "
            "shadow scene is still rendered as the actual lived world "
            "in plain past-tense narration \u2014 the renderer must NOT "
            "surface this flag in the prose (no 'in this branch', no "
            "'timeline', no subjunctive author voice; see Rule 10 in "
            "prompts/generation.md). The flag is used internally to "
            "suppress factual-mainline continuity assumptions and to "
            "unlock the BRANCH CONTEXT block that feeds "
            "``factual_contrast_summary`` to the renderer as silent "
            "background."
        ),
    )
    branch_label: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable label for the shadow branch, when known "
            "(e.g. 'What if Duncan lived'). Surfaced to the generator "
            "so the rendered prose can echo the user's framing."
        ),
    )
    factual_contrast_summary: Optional[str] = Field(
        default=None,
        description=(
            "Brief prose summary of what *did* happen on the factual "
            "mainline at the same syuzhet horizon, included only when "
            "``branch_world_id == 'shadow'``. Used by the generator "
            "to keep the shadow branch in productive contrast with "
            "canon rather than re-narrating identical events."
        ),
    )
    preceding_prose: Optional[str] = Field(
        default=None,
        description=(
            "Concatenated story-so-far prose from prior versions in the "
            "current session's lineage, oldest \u2192 newest, truncated to a "
            "character budget. Threaded onto the brief by ``run_pipeline`` "
            "so a sequence of queries (e.g. counterfactual \u2192 counterfactual "
            "\u2192 intervention \u2192 observation) renders prose that is "
            "narratively continuous with everything that has come before, "
            "not just the accumulated world state. Filtered by branch: "
            "factual queries see only factual prose; shadow queries see "
            "the shadow lineage's contiguous tail plus factual ancestors. "
            "Surfaced verbatim into the renderer prompt's STORY SO FAR "
            "section but never authoritative \u2014 hard constraints and "
            "world-state still take precedence on conflict."
        ),
    )
    narrative_style: Optional[NarrativeStyle] = Field(
        default=None,
        description=(
            "Source-text style profile carried from ingestion. The "
            "renderer and auditor use this to keep the generated prose "
            "in the same form as the source (plot summary stays "
            "summary-length; short story stays scene-length; etc.)."
        ),
    )


# =====================================================================
# Emotion → trait vocabulary
# =====================================================================
# Each effect maps to ``(positive_indicators, inverse_indicators)``:
#   * positive — traits whose *high* value indicates the emotion
#   * inverse  — traits whose *low* value indicates the emotion
#
# Vocabulary is calibrated against the actual trait names used across
# the ``example_worlds/`` corpus (passion, tenderness, devotion, warmth,
# longing, obsession, vengefulness, cruelty, vindictiveness, …) — the
# previous narrow synonym lists (``["love", "affection", "sensuality"]``
# for love; ``["happiness", "hope", "contentment"]`` for joy) matched
# almost no real-world fixture, collapsing 50+ of the 96 emotion scores
# into the worst-case ``+1.0`` "no relevant traits" fallback. Treats
# ``positive`` and ``inverse`` symmetrically so the decrease arm is no
# longer dead code (previously decrease-set traits not duplicated in
# the increase list never entered the average).
_EFFECT_TRAITS: Dict[str, Tuple[List[str], List[str]]] = {
    "grief": (
        ["grief", "despair", "shame", "sadness", "longing",
         "vulnerability", "guilt"],
        ["hope", "happiness", "contentment", "joy", "warmth",
         "vitality", "resolve"],
    ),
    "rage": (
        ["rage", "anger", "aggression", "vengefulness", "cruelty",
         "vindictiveness", "resentment", "rebelliousness", "volatility",
         "ruthlessness"],
        ["calm", "patience", "composure", "compassion", "warmth",
         "kindness", "tenderness", "restraint"],
    ),
    "joy": (
        ["happiness", "hope", "contentment", "warmth", "joy",
         "vitality", "tenderness", "wit"],
        ["despair", "fear", "anxiety", "sadness", "grief", "shame",
         "paranoia", "guilt"],
    ),
    "fear": (
        ["fear", "paranoia", "anxiety", "vulnerability", "frailty",
         "desperation"],
        ["courage", "hope", "calm", "composure", "resolve", "bravado",
         "endurance", "resilience"],
    ),
    "love": (
        ["love", "affection", "passion", "tenderness", "devotion",
         "warmth", "longing", "obsession", "constancy", "sensuality",
         "compassion", "kindness"],
        ["coldness", "cruelty", "anger", "resentment", "vindictiveness",
         "rage"],
    ),
    "regret": (
        ["guilt", "remorse", "shame", "regret", "despair", "longing",
         "grief"],
        ["happiness", "contentment", "hope", "resolve", "vitality",
         "composure"],
    ),
}


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
        """Compute relationship tensions involving the given entities.

        Per-axis observed/evidence-aware:
          * Axes whose ``observed`` flag is False contribute nothing to
            the asymmetry score (an unmeasured axis is not a measurable
            asymmetry — the previous all-axes-equal-weight average
            silently inflated the score with hallucinated zeros).
          * Each contributing axis is weighted by the *minimum* of the
            two endpoints' evidence strengths (weak=1/3, moderate=2/3,
            strong=1) so a strongly-evidenced asymmetry anchors the
            score and a weak-vs-strong pair is not averaged into the
            middle. The reported per-axis values still come from the
            forward edge for backward compatibility.
        """
        tensions: List[RelationshipTension] = []
        eid_set = set(entity_ids)
        es_weight = {"weak": 1.0 / 3.0, "moderate": 2.0 / 3.0, "strong": 1.0}

        def _axis(rel: dict, axis: str) -> tuple[float, str, bool]:
            metrics = rel.get("metrics") if isinstance(rel.get("metrics"), dict) else None
            if metrics and isinstance(metrics.get(axis), dict):
                m = metrics[axis]
                return (
                    float(m.get("value", 0.0)),
                    str(m.get("evidence_strength", "moderate")),
                    bool(m.get("observed", True)),
                )
            # Legacy fallback: flat key, assume observed if present.
            if axis in rel and isinstance(rel.get(axis), (int, float)):
                return float(rel[axis]), "moderate", True
            return 0.0, "weak", False

        for rel in self.ego.get("relevant_relationships", []):
            src = rel.get("source_entity_id", "")
            tgt = rel.get("target_entity_id", "")
            if src not in eid_set and tgt not in eid_set:
                continue

            aff, aff_es, aff_obs = _axis(rel, "affinity")
            fear, fear_es, fear_obs = _axis(rel, "fear")
            power, power_es, power_obs = _axis(rel, "power_dynamic")

            # Locate the reverse edge once.
            reverse: dict = {}
            for rev in self.ego.get("relevant_relationships", []):
                if rev.get("source_entity_id") == tgt and rev.get("target_entity_id") == src:
                    reverse = rev
                    break
            r_aff, r_aff_es, r_aff_obs = _axis(reverse, "affinity")
            r_fear, r_fear_es, r_fear_obs = _axis(reverse, "fear")
            r_power, r_power_es, r_power_obs = _axis(reverse, "power_dynamic")

            contributions: list[tuple[float, float]] = []  # (delta, weight)
            if aff_obs and r_aff_obs:
                w = min(es_weight[aff_es], es_weight[r_aff_es])
                contributions.append((abs(aff - r_aff), w))
            if fear_obs and r_fear_obs:
                w = min(es_weight[fear_es], es_weight[r_fear_es])
                contributions.append((abs(fear - r_fear), w))
            if power_obs and r_power_obs:
                w = min(es_weight[power_es], es_weight[r_power_es])
                # power should be anti-symmetric (A dominates B ⇒ B
                # dominated by A), so the *sum* registers asymmetry.
                contributions.append((abs(power + r_power), w))

            if contributions:
                total_w = sum(w for _, w in contributions) or 1.0
                asymmetry = sum(d * w for d, w in contributions) / total_w
            else:
                asymmetry = 0.0

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
    # Hidden information signals (Channels + future utterances)
    # ------------------------------------------------------------------
    def compute_hidden_channels(
        self, syuzhet_anchor: Optional[int] = None,
    ) -> List[HiddenChannel]:
        """Find communication signals hidden from the reader at this anchor.

        Returns two kinds of hidden signal:

        * **Channels** that exist in the world but whose first on-page
          utterance comes after ``syuzhet_anchor`` (or which never
          surface). The mere existence of the channel is reader-secret.
        * **Utterance events** with ``syuzhet_index > syuzhet_anchor``.
          The message itself happens later in narration order, so its
          content must not leak.
        """
        if syuzhet_anchor is None:
            return []

        # Build channel_id → earliest revealed utterance syuzhet_index.
        channel_first_utt: Dict[str, Optional[int]] = {
            cid: None for cid in self.world_state.channels.keys()
        }
        for evt in self.world_state.events:
            if evt.event_type != "utterance" or not evt.via_channel_id:
                continue
            cid = evt.via_channel_id
            if cid not in channel_first_utt:
                continue
            current = channel_first_utt[cid]
            if current is None or evt.syuzhet_index < current:
                channel_first_utt[cid] = evt.syuzhet_index

        hidden: List[HiddenChannel] = []
        intel_thresh = _get_settings().physics.intelligibility_threshold
        for cid, ch in self.world_state.channels.items():
            unintel = sorted([
                pid for pid in ch.participant_ids
                if float(ch.intelligibility.get(pid, 1.0)) < intel_thresh
            ])
            first = channel_first_utt.get(cid)
            if first is None:
                # A channel that never carries an on-page utterance is
                # ambient capability, not a withheld secret. We still
                # surface it when at least one participant cannot
                # reliably parse what flows through it — that
                # asymmetry is itself a dramatic-irony lever even
                # though the channel is technically visible.
                if unintel:
                    hidden.append(HiddenChannel(
                        kind="channel",
                        channel_id=cid,
                        medium=ch.medium,
                        participant_ids=list(ch.participant_ids),
                        discovered_at_syuzhet=None,
                        unintelligible_for=unintel,
                    ))
                continue
            if first > syuzhet_anchor:
                hidden.append(HiddenChannel(
                    kind="channel",
                    channel_id=cid,
                    medium=ch.medium,
                    participant_ids=list(ch.participant_ids),
                    discovered_at_syuzhet=first,
                    unintelligible_for=unintel,
                ))

        for evt in self.world_state.events:
            if evt.event_type != "utterance":
                continue
            if evt.syuzhet_index <= syuzhet_anchor:
                continue
            ch = (
                self.world_state.channels.get(evt.via_channel_id)
                if evt.via_channel_id else None
            )
            hidden.append(HiddenChannel(
                kind="utterance",
                utterance_event_id=evt.id,
                channel_id=evt.via_channel_id,
                medium=ch.medium if ch else "unmediated",
                participant_ids=list(ch.participant_ids) if ch else [],
                addressee_ids=list(evt.addressee_ids),
                speaker_id=evt.speaker_id,
                discovered_at_syuzhet=evt.syuzhet_index,
            ))
        return hidden

    # ------------------------------------------------------------------
    # Graph helpers for affective measures
    # ------------------------------------------------------------------
    def _build_causal_digraph(self) -> nx.DiGraph:
        """Build a weighted causal DiGraph from the world state topology."""
        _STRENGTH_W = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}
        _scaling = _get_settings().physics.causal_force_scaling
        # Pre-index relationship per-axis evidence so ``mutation_social``
        # edges can attenuate their weight by the strength of the actual
        # measured relationship axis they target. Without this, chains
        # passing through a relationship metric (mutation_social →
        # mutation cascades) weight the relationship-touching link by
        # the CausalEdge's own evidence_strength only, ignoring whether
        # the underlying axis was strongly or weakly observed.
        rel_axis_es: Dict[Tuple[str, str, str], str] = {}
        for rel in self.world_state.social_topology:
            for axis_name, m in rel.metrics.items():
                if m.observed:
                    rel_axis_es[(rel.source_entity_id, rel.target_entity_id, axis_name)] = m.evidence_strength
        g = nx.DiGraph()
        for ce in self.world_state.causal_topology:
            evidence_w = _STRENGTH_W.get(ce.evidence_strength, 0.5)
            # For mutation_social edges, multiply by the per-axis
            # relationship evidence as a precision floor.
            if ce.causality_type == "mutation_social" and ce.rel_counterpart_id and ce.trait_target:
                rel_es = rel_axis_es.get(
                    (ce.target_id, ce.rel_counterpart_id, ce.trait_target)
                )
                if rel_es is not None:
                    evidence_w = min(evidence_w, _STRENGTH_W.get(rel_es, 0.5))
            force_scale = ce.causal_force / _scaling
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
    # Same saturation idea as suspense: a small surface (one or two
    # irony edges) shouldn't pin the gauge at 1.0 just because every
    # revealed cause happens to be hidden from the focal cast. Tuned
    # against ``example_worlds`` so canonical irony stories (Death on
    # the Nile, Gone Girl, Reservoir Dogs) sit in the 0.3–0.8 band at
    # their reveal-points rather than instantly saturating.
    _IRONY_SURFACE_K: float = 1.0

    def compute_dramatic_irony_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Dramatic Irony: information asymmetry where reader knows more.

        For each focal entity, counts *intensity-weighted* revealed
        events the focal entity does NOT know about, divided by the
        **total** event mass plus a saturation constant ``K``:

        .. math::

           \\text{irony} =
              \\frac{1}{|F|} \\sum_{c \\in F}
              \\frac{\\sum_{e \\in \\text{revealed}, e \\notin K_c} w_e}
                   {\\sum_{e \\in \\text{events}} w_e + K}

        where :math:`F` is the focal cast, :math:`K_c` is what
        character :math:`c` knows, and :math:`w_e` is event ``e``'s
        intensity (defaults to ``1.0``).

        Why this shape rather than the previous "cumulative
        revealed-only ratio":

        * The previous form normalised by the revealed-edge count, so
          numerator and denominator grew together and the score
          asymptoted to a story-specific plateau by anchor ~3 (e.g.
          Macbeth held 0.55–0.67 from anchor 3 onward; Reservoir
          Dogs *decayed* from 0.25 to 0.06 because the protagonist
          became actor-of-record on more revealed edges over time).
          Normalising by the **full event mass** (a fixed denominator)
          lets the curve rise smoothly with reveals and fall when
          characters acquire knowledge later, producing the
          dramatic-irony arc the gauge is supposed to depict.
        * The previous form scoped irony to causal edges whose target
          was the focal entity. On every example_world that is
          structurally degenerate: the focal cast (top-N by event
          degree) is precisely the cast that participates as actor or
          target in nearly every cause→focal edge, so the "character
          doesn't know the cause" condition rarely fires (Death on
          the Nile collapsed to flat-zero under that scoping). Real
          irony isn't about the focal entity's own incoming causal
          arrows — it's about *what the reader has been shown that
          the focal character has not seen*, regardless of whether
          that information happens to causally target them.
        * Character knowledge is bounded by the syuzhet anchor's
          fabula frontier, so a character isn't credited with
          knowing their own future arc from anchor 0. This is what
          allows late catch-ups (Macduff learning about his family,
          Poirot's denouement) to actually pull the curve down.

        The legacy "addressee-exclusion auto-counts as irony" branch
        is dropped: it added ``1/1`` per excluded utterance regardless
        of whether the excluded character had an epistemic gap,
        biasing the score upward by a fixed amount that never decayed.
        """
        if syuzhet_anchor is None:
            return 0.0

        revealed = self._revealed_event_ids(syuzhet_anchor)
        if not revealed:
            return 0.0

        # Fabula frontier: a character can plausibly know an event by
        # direct participation only once that event has happened in
        # narrative time. Use the latest fabula_time among revealed
        # events — for plots told in chronological order this matches
        # the syuzhet anchor exactly; for non-linear plots
        # (Reservoir Dogs flashbacks, Gone Girl diary entries) it
        # correctly admits earlier-fabula events the reader has just
        # been shown.
        events_by_id = {e.id: e for e in self.world_state.events}
        fabula_frontier = max(
            events_by_id[eid].fabula_time for eid in revealed
        )

        def _evt_w(evt) -> float:
            return float(getattr(evt, "intensity", None) or 1.0)

        total_mass = sum(_evt_w(e) for e in self.world_state.events)
        if total_mass <= 0.0:
            return 0.0

        revealed_mass = sum(
            _evt_w(events_by_id[eid]) for eid in revealed
        )

        per_character_gaps: list[float] = []

        for eid in entity_ids:
            ent = self.world_state.entities.get(eid)
            if not ent:
                continue

            # An entity is *aware* of an event when (a) they participate
            # in it (actor or target) AND it has happened by the fabula
            # frontier, (b) a revealed utterance addressed to them (or
            # spoken by them) refers to it, or (c) they hold a Belief
            # whose target_id matches the event id.
            #
            # Channel intelligibility gates (b): a recipient who cannot
            # parse the channel (intelligibility[recipient] below the
            # configured threshold) does NOT learn from the utterance
            # even if they were nominally addressed. Without this gate,
            # encrypted/coded/foreign-language messages would silently
            # close dramatic-irony surfaces that should remain open.
            intel_thresh = _get_settings().physics.intelligibility_threshold
            channels = self.world_state.channels

            known: set[str] = {
                evt.id for evt in self.world_state.events
                if (eid in evt.actor_ids or eid in evt.target_ids)
                and evt.fabula_time <= fabula_frontier
            }
            for utt in self.world_state.events:
                if utt.event_type != "utterance":
                    continue
                if utt.syuzhet_index > syuzhet_anchor:
                    continue
                if eid not in utt.addressee_ids and eid != utt.speaker_id:
                    continue
                # Speaker always understands what they themselves uttered.
                # Addressees only understand if the channel (when one
                # mediates the utterance) is intelligible to them.
                if eid != utt.speaker_id and utt.via_channel_id:
                    ch = channels.get(utt.via_channel_id)
                    if ch is not None:
                        intel = float(ch.intelligibility.get(eid, 1.0))
                        if intel < intel_thresh:
                            continue
                for tid in utt.target_ids:
                    if tid.startswith("EVT_"):
                        known.add(tid)
            known |= {
                b.target_id for b in ent.beliefs
                if b.target_id.startswith("EVT_")
            }

            # Intensity-weighted mass of revealed events this character
            # does NOT know — the per-character irony surface.
            gap_mass = sum(
                _evt_w(events_by_id[reid])
                for reid in revealed
                if reid not in known
            )
            per_character_gaps.append(
                gap_mass / (total_mass + self._IRONY_SURFACE_K)
            )

        if not per_character_gaps:
            return 0.0

        score = min(
            sum(per_character_gaps) / len(per_character_gaps), 1.0
        )
        logger.debug(
            "[DirectiveAssembly·DramaticIrony] revealed_mass=%.3f / "
            "(total_mass=%.3f + K=%.2f), per-char gaps=%s, score=%.3f",
            revealed_mass, total_mass, self._IRONY_SURFACE_K,
            [round(g, 3) for g in per_character_gaps], score,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Suspense  (Probabilistic Valence — threat vs hope)
    # ------------------------------------------------------------------
    # Suspense saturates as the *combined* unrevealed weight on either
    # side passes ~K. K=2 means "two strong unrevealed events on each
    # side" already counts as fully high-stakes; smaller K makes the
    # gauge twitchier. Tuned against the example_worlds corpus so a
    # mid-anchor on Macbeth / Gone Girl / Reservoir Dogs reads in the
    # 0.3–0.7 band rather than the previous flat-zero output.
    _SUSPENSE_STAKES_K: float = 2.0

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

        Returns ``balance × stakes`` clamped to ``[0, 1]``:

        * ``balance = 1 - |threat - hope| / (threat + hope)`` peaks at
          1.0 when the two sides are equally weighted (genuine
          uncertainty about the outcome) and decays to 0 when one side
          dominates the other.
        * ``stakes = total / (total + K)`` with ``K`` =
          :attr:`_SUSPENSE_STAKES_K`. Saturates so small unrevealed
          fragments don't pin the gauge at 1.0 just because they
          happen to be balanced.

        The previous ``max(0, (threat - hope) / total)`` form
        collapsed to zero on every real plot in ``example_worlds/``
        because the protagonist is the actor of most of their own
        forward events (Macbeth kills Duncan / Banquo / Macduff's
        family, all bumping ``hope_w`` over ``threat_w``) — leaving
        suspense pinned at 0.0 across the entire syuzhet axis even
        for canonical thrillers and tragedies.

        Returns 0 when hope is entirely extinguished (despair) or when
        no threat is present (safety) — both still degenerate to
        non-suspense as required by the test contract.
        """
        causal_g = self._build_causal_digraph()
        revealed = self._revealed_event_ids(syuzhet_anchor)
        all_evt_ids = {e.id for e in self.world_state.events}
        unrevealed = all_evt_ids - revealed
        eid_set = set(entity_ids)

        threat_weight = 0.0
        hope_weight = 0.0

        events_by_id = {e.id: e for e in self.world_state.events}

        for evt_id in unrevealed:
            evt = events_by_id.get(evt_id)
            if not evt:
                continue
            actor_set = set(evt.actor_ids)
            target_set = set(evt.target_ids)
            if not (actor_set & eid_set) and not (target_set & eid_set):
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

            # Per-entity classification: an entity acted upon (without
            # itself acting) contributes to threat; an entity acting
            # contributes to hope. A single event between two focused
            # entities legitimately raises both sides of the ledger.
            for eid in eid_set:
                is_actor = eid in actor_set
                is_target = eid in target_set
                if is_target and not is_actor:
                    threat_weight += prob
                elif is_actor:
                    hope_weight += prob

        if hope_weight <= 0.0:
            logger.debug(
                "[DirectiveAssembly·Suspense] No hope outcome — suspense=0 (despair)",
            )
            return 0.0
        if threat_weight <= 0.0:
            logger.debug(
                "[DirectiveAssembly·Suspense] No threat outcome — suspense=0 (safety)",
            )
            return 0.0

        total = threat_weight + hope_weight
        balance = 1.0 - abs(threat_weight - hope_weight) / total
        stakes = total / (total + self._SUSPENSE_STAKES_K)
        score = max(0.0, min(1.0, balance * stakes))
        logger.debug(
            "[DirectiveAssembly·Suspense] threat_w=%.3f hope_w=%.3f "
            "balance=%.3f stakes=%.3f suspense=%.3f",
            threat_weight, hope_weight, balance, stakes, score,
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
          apply a **geometric** Bayesian-style pull,
          ``prior += w_i · (actual - prior)`` (clipped to ``[ε, 1-ε]``),
          so each successive piece of evidence asymptotes the prior
          toward the truth without overshooting (an additive form
          ``prior += w · (actual - base_prior)`` summed past the
          actual value once ``Σw > 1``, producing a non-monotonic
          surprise curve that contradicted the
          "more-revealed → less-surprise" semantics).
        * **Posterior** — actual trait values from the sandbox
          (post-simulation) or world state (truth).

        Returns a normalised KL divergence in [0, 1].
        """
        if syuzhet_anchor is None:
            return 0.0  # reader knows everything → no surprise

        EPS = 0.01
        _STRENGTH_W = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}
        revealed = self._revealed_event_ids(syuzhet_anchor)

        # Determine the final fabula_time so we can resolve every
        # entity's *final* trait values (the syuzhet-axis posterior).
        # Reading raw ``Entity.traits`` was a silent bug: those are the
        # *baseline* (pre-story) values, so a character whose arc was
        # written entirely in ``state_timeline`` snapshots — which is
        # exactly how every example_world fixture encodes its
        # protagonist arcs (Macbeth's ambition 0.7→0.85, Lady
        # Macbeth's guilt 0.0→0.9, etc.) — would be compared against
        # itself, collapsing surprise to zero before the prior update
        # even ran.
        if self.world_state.events:
            t_max = max(e.fabula_time for e in self.world_state.events)
        else:
            t_max = 0

        def _final_traits(ent) -> Dict[str, float]:
            try:
                snap = reconstruct_entity_at(ent, t_max)
                return {
                    k: float(v["value"])
                    for k, v in snap.get("traits", {}).items()
                }
            except Exception:
                return {k: float(v.value) for k, v in ent.traits.items()}

        # Pre-compute per-trait corpus marginals using each entity's
        # *final* trait value (matches the posterior we'll be
        # comparing against). The marginal is computed leave-one-out
        # for every focal entity later, so we keep the per-entity
        # contributions rather than collapsing them up front.
        per_entity_finals: Dict[str, Dict[str, float]] = {}
        for ent_id, ent in self.world_state.entities.items():
            per_entity_finals[ent_id] = _final_traits(ent)

        marginal_sum: Dict[str, float] = {}
        marginal_count: Dict[str, int] = {}
        for ent_id, traits in per_entity_finals.items():
            for tname, tval in traits.items():
                marginal_sum[tname] = marginal_sum.get(tname, 0.0) + tval
                marginal_count[tname] = marginal_count.get(tname, 0) + 1

        def _trait_marginal(name: str, exclude_eid: str) -> float:
            # Leave-one-out: remove the focal entity's contribution so
            # the prior we're computing KL against isn't biased by the
            # very value we're trying to predict. With small casts
            # (every example_world averages 6–10 entities) the focal
            # entity carries 10–17% weight in the corpus marginal, so
            # the inclusive form was systematically pulling the prior
            # toward the posterior and squashing surprise toward 0.
            total = marginal_sum.get(name, 0.0)
            count = marginal_count.get(name, 0)
            excl = per_entity_finals.get(exclude_eid, {}).get(name)
            if excl is not None:
                total -= excl
                count -= 1
            # Need at least 2 *remaining* entities for the leave-one-out
            # marginal to be meaningful — with one or zero we fall back
            # to maximum entropy (0.5) so single-entity scenarios still
            # register surprise from extreme trait values.
            if count < 2:
                return 0.5
            return total / count

        total_kl = 0.0
        trait_count = 0

        for eid in entity_ids:
            # Posterior: prefer sandbox, fall back to the final-state
            # reconstruction computed above.
            if self.sandbox is not None and self.sandbox.has_node(eid):
                actual_traits = self.sandbox.nodes[eid].get("traits", {})
            else:
                actual_ent = self.world_state.entities.get(eid)
                if not actual_ent:
                    continue
                actual_traits = {
                    k: {"value": v}
                    for k, v in per_entity_finals.get(eid, {}).items()
                }

            for trait_name, actual_data in actual_traits.items():
                if not isinstance(actual_data, dict) or "value" not in actual_data:
                    continue

                actual_val = actual_data["value"]

                # Prior: start from the leave-one-out corpus marginal
                # for this trait, then pull toward the actual value
                # once for each revealed causal edge. Each edge applies
                # a geometric update ``prior += w · (actual - prior)``
                # so the prior asymptotes toward the truth as evidence
                # accumulates but cannot overshoot.
                base_prior = _trait_marginal(trait_name, eid)
                prior_val = base_prior
                for ce in self.world_state.causal_topology:
                    if ce.target_id != eid:
                        continue
                    if ce.source_id not in revealed:
                        continue
                    w = _STRENGTH_W.get(ce.evidence_strength, 0.5)
                    prior_val += w * (actual_val - prior_val)
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
                kl = max(0.0, kl)
                # Per-trait soft saturation. The previous form averaged
                # raw KL and divided by ``log(1/EPS) ≈ 4.6`` — the
                # *theoretical* maximum when one side sits at EPS and
                # the other at 1-EPS. In practice perceptually
                # meaningful binary KLs sit in the 0.2–1.5 band
                # (Macbeth's guilt 0.85 vs uninformed prior 0.5 yields
                # KL=0.27; despair 0.95 vs 0.5 yields 0.49) so dividing
                # by 4.6 compressed the entire signal into a 4% slice
                # of the gauge — flat-looking even on canonical
                # surprise plots like Death on the Nile and Gone Girl.
                #
                # ``1 - exp(-kl)`` keeps each per-trait contribution in
                # [0, 1] and maps perceptual KLs to perceptual gauge
                # positions: KL=0.27→0.24, KL=0.5→0.39, KL=1.0→0.63,
                # KL=2.0→0.86. Saturates smoothly so extreme reveals
                # still asymptote toward 1.0 without throwing away
                # information at the high end.
                total_kl += 1.0 - math.exp(-kl)
                trait_count += 1

        if trait_count == 0:
            return 0.0

        score = min(total_kl / trait_count, 1.0)

        logger.debug(
            "[DirectiveAssembly·Surprise] mean(1-exp(-kl))=%.4f over %d traits",
            score, trait_count,
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
        # loss should reward. We compute *closeness to target*:
        #   • positive trait: target = 1.0, closeness = current_value
        #   • inverse trait:  target = 0.0, closeness = 1 - current_value
        # so an entity already saturated in the right direction yields
        # ``score ≈ -1`` (strong match) and an entity stuck on the wrong
        # side yields ``score ≈ 0`` (no match). Both positive and
        # inverse traits enter the average — the previous form's
        # ``relevant`` filter only included the increase list, leaving
        # the decrease set as dead code (e.g. a brave character got no
        # fear-reducing credit because ``courage`` never entered the
        # average).
        else:
            positive, inverse = _EFFECT_TRAITS.get(target_effect, ([], []))
            positive_set = set(positive)
            inverse_set = set(inverse)

            trajectories = self.compute_trait_trajectories(entity_ids)
            contributions: List[float] = []
            for traj in trajectories:
                if traj.trait_name in positive_set:
                    contributions.append(traj.current_value)
                elif traj.trait_name in inverse_set:
                    contributions.append(1.0 - traj.current_value)

            if contributions:
                avg_match = sum(contributions) / len(contributions)
                score -= avg_match
                logger.debug(
                    "[DirectiveAssembly·AffectiveScore] effect=%s "
                    "contribs=%d avg_match=%.3f score=%.4f",
                    target_effect, len(contributions), avg_match, score,
                )
            else:
                # No traits in the entity match the per-effect target
                # set — we have nothing to score against. Treat this as
                # the *worst possible* match (``+1.0``) rather than the
                # midpoint (``+0.5``) so the loss is on the same scale
                # as the success path: structural effects subtract a
                # value in ``[0, 1]`` and emotion successes subtract
                # a closeness in ``[0, 1]``, giving a best-case score
                # of ``-1.0``. With the calibrated vocabulary above
                # this branch should now fire only for entities with
                # genuinely no emotional traits at all (e.g. inanimate
                # objects routed in by mistake).
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
                        f"have NOT been revealed on-page. You MUST NOT reveal or "
                        f"hint at these causes. Render the effect through aftermath "
                        f"and unanswered detail; do not name or hint at the missing "
                        f"causes on-page "
                        f"(mystery_score={mystery_score:.2f})."
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

            # Hidden information channels amplify mystery. Utterances
            # are hard constraints (revealing the content of an unspoken
            # line is a fidelity break) but standing channels are soft —
            # a sealed letter or unrevealed phone tap can be teased
            # on-page without naming what flows through it.
            if hidden_channels:
                for hc in hidden_channels:
                    if hc.kind == "channel":
                        instr = (
                            f"[HIDDEN CHANNEL]: A {hc.medium} link between "
                            f"{hc.participant_ids} exists but is not yet "
                            f"on-page. Do not reference it."
                        )
                        priority: Literal["hard", "soft"] = "soft"
                    else:
                        instr = (
                            f"[HIDDEN UTTERANCE]: A {hc.medium} message "
                            f"from {hc.speaker_id} to {hc.addressee_ids} "
                            f"happens at syuzhet_index="
                            f"{hc.discovered_at_syuzhet}. Do not reveal its content."
                        )
                        priority = "hard"
                    constraints.append(ConstraintBlock(
                        constraint_type="narrative",
                        priority=priority,
                        instruction=instr,
                        evidence={
                            "kind": hc.kind,
                            "channel_id": hc.channel_id,
                            "utterance_event_id": hc.utterance_event_id,
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
                for utt in self.world_state.events:
                    if utt.event_type != "utterance":
                        continue
                    if syuzhet_anchor is not None and utt.syuzhet_index > syuzhet_anchor:
                        continue
                    if eid not in utt.addressee_ids and eid != utt.speaker_id:
                        continue
                    for tid in utt.target_ids:
                        if tid.startswith("EVT_"):
                            events_known_by_character.add(tid)
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
                        f"[DRAMATIC IRONY]: The event "
                        f"'{src_evt.id}' ({src_evt.description}) has been "
                        f"revealed on-page and causally affects {eid}, but "
                        f"{eid} is UNAWARE of this connection. Render the "
                        f"information asymmetry through {eid}'s on-page "
                        f"behaviour and dialogue \u2014 their actions and "
                        f"choices must show the gap, never narrate it. Show "
                        f"{eid} acting in ignorance. {eid} MUST NOT learn "
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
                        f"'{widest.actual_state}'. The prose MUST keep "
                        f"{widest.entity_id} ignorant on-page; render their "
                        f"misplaced confidence through behaviour and dialogue. "
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
                        f"the truth to this character. Render the gap as "
                        f"{widest.entity_id}'s on-page actions and "
                        f"misplaced confidence "
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
                        f"happened chronologically but is withheld until "
                        f"syuzhet_index={most_displaced.syuzhet_index} "
                        f"(displacement="
                        f"{most_displaced.displacement:+.2f}). "
                        f"You MUST NOT reference, spoil, or hint at this event. "
                        f"Do not state '{most_displaced.description}' on-page yet."
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

            # Layer 4: Hidden information channels.  Same hard/soft
            # split as the mystery branch above: utterances are hard
            # (cannot be quoted before they happen), standing channels
            # are soft (a sealed letter may be teased on-page).
            if hidden_channels:
                for hc in hidden_channels:
                    if hc.kind == "channel":
                        instr = (
                            f"[HIDDEN CHANNEL]: A {hc.medium} link between "
                            f"{hc.participant_ids} exists but is not yet "
                            f"on-page. Do not reference it."
                        )
                        priority: Literal["hard", "soft"] = "soft"
                    else:
                        instr = (
                            f"[HIDDEN UTTERANCE]: A {hc.medium} message "
                            f"from {hc.speaker_id} to {hc.addressee_ids} "
                            f"happens at syuzhet_index="
                            f"{hc.discovered_at_syuzhet}. Do not reveal its content."
                        )
                        priority = "hard"
                    constraints.append(ConstraintBlock(
                        constraint_type="narrative",
                        priority=priority,
                        instruction=instr,
                        evidence={
                            "kind": hc.kind,
                            "channel_id": hc.channel_id,
                            "utterance_event_id": hc.utterance_event_id,
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
                        f"between the prior expected outcome and the "
                        f"actual revelation is {surprise_score:.2f} "
                        f"(normalised KL divergence). Render the moment "
                        f"of the pivot through a sharp syntactical break "
                        f"and a short, blunt sentence on-page \u2014 do not "
                        f"name 'the surprise', 'the shock', or 'the reader'."
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
                and t.displacement > 0.10
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
            # Use the shared module-level vocabulary (calibrated against
            # the example_worlds corpus). Both the increase
            # (``positive``) and decrease (``inverse``) lists drive
            # constraint generation so e.g. a brave character routed
            # through ``fear`` correctly produces a "courage must drop"
            # constraint, not silently nothing.
            positive, inverse = _EFFECT_TRAITS.get(effect, ([], []))
            target_traits = set(positive) | set(inverse)
            decrease_set = set(inverse)

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
                    "Render the focal character's confusion and initial processing of the scene.",
                    "Suppress all omniscient narration; do not hint at hidden causes.",
                    "Lock the prose strictly to the focal character's limited perspective.",
                    "Render effects without naming their causes; let absence carry the weight.",
                ],
            )

        elif effect == "dramatic_irony":
            rendering = RenderingDirective(
                rendering_mode="dramatic_irony",
                pov_lock=pov_entity,
                pacing="normal",
                sensory_focus="normal",
                stylistic_instructions=[
                    "Render the focal character's naive interior monologue against the on-page facts the character has not yet connected.",
                    "Show the focal character acting on a false sense of security — making plans, relaxing, feeling confident.",
                    "Show the focal character making decisions on incomplete information.",
                    "Render the gap as behaviour and dialogue, never as commentary.",
                    "The focal character MUST NOT learn the withheld truth during this scene.",
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
                    "Open with flowing, comfortable prose consistent with the prior expected outcome.",
                    "Telegraph the prior expectation through character thoughts and environmental cues.",
                    "At the moment of revelation, execute a sharp syntactical pivot.",
                    "Use a short, blunt sentence to render the hidden truth as it lands.",
                    "After the pivot, render the focal character's reorientation through behaviour, not commentary.",
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
                    "Keep the hopeful escape route visible in the prose but physically just out of reach.",
                    "Render the closing window of opportunity through concrete on-page detail, not commentary.",
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
            counterfactual_branch = self._build_counterfactual_branch(entity_ids, syuzhet_anchor=syuzhet_anchor)

            rendering = RenderingDirective(
                rendering_mode="regret",
                pov_lock=pov_entity,
                pacing="dilated",
                sensory_focus="normal",
                tone_arc="harsh_reality ↔ agonizing_visualization",
                stylistic_instructions=[
                    "Render the counterfactual content INSIDE the focal character's interior monologue as their own 'if only…' thought; never in author voice and never as a named structural object.",
                    "The focal character's interior monologue MUST articulate concrete 'if only…' logic about the specific choice they did not take.",
                    "Contrast the harsh sensory reality of the present moment with the focal character's interior visualisation of the path they did not choose.",
                    "Do NOT simply state the focal character is sad — render the specific choice they failed to make through their interior thought.",
                    "Alternate inside the focal character's POV between the bleak present and the imagined unchosen path; never step outside as a narrator pointing at the contrast.",
                ],
            )

        elif effect == "grief":
            rendering = RenderingDirective(
                rendering_mode="grief",
                pov_lock=pov_entity,
                pacing="dilated",
                sensory_focus="absence",
                stylistic_instructions=[
                    "Focus on ABSENCE — render the physical space left behind by the lost figure.",
                    "Use fragmented or numb prose that mirrors the focal character's disrupted interiority.",
                    "Render the silence where a voice used to be, the empty chair, the cold side of the bed.",
                    "Render the focal character's disorientation through concrete sensory absences and broken routine, not through structural commentary.",
                    "Short sentences. Disconnected observations. The world feels wrong.",
                ],
            )

        elif effect == "rage":
            # Build causal attribution — who caused the loss
            causal_attribution = self._build_causal_attribution(entity_ids, syuzhet_anchor=syuzhet_anchor)

            rendering = RenderingDirective(
                rendering_mode="rage",
                pov_lock=pov_entity,
                pacing="accelerated",
                sensory_focus="tunnel",
                tone_arc="passive_sorrow → active_targeted_hostility",
                stylistic_instructions=[
                    "Execute a tonal shift from passive sorrow to active, targeted hostility.",
                    "The prose accelerates as focus narrows obsessively onto the perpetrator.",
                    "Render the focal character's body marshalling the capacity to do harm — clenched hands, quickened pulse, hardened gaze — not abstract trait language.",
                    "Show the focal character preparing to act on the perpetrator (a step toward, a weapon picked up, a plan crystallising in dialogue or thought).",
                    "The grief does not disappear — render it transmuting into directed motion.",
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
                    "Render the bond through mirrored on-page reactions between the paired characters.",
                    "If one of the pair takes a hit, the other reacts instantly — prioritising the partner's safety over their own.",
                    "Render shared physical and emotional proximity through concrete blocking, gaze, touch.",
                    "Show harm-to-one as harm-to-the-other through the partner's involuntary response — never name it as 'entanglement' or 'coupling'.",
                    "Render the bond through action, never through declaration; show, never tell.",
                ],
            )

        # Stamp the active syuzhet anchor into scene_context so
        # downstream consumers (auditor leak-check, renderers) can
        # locate the brief on the timeline without having to re-derive
        # it from recent_memory (which is fabula-sorted and can sit
        # ahead of the reader's current syuzhet position).
        scene_context = dict(self.ego) if isinstance(self.ego, dict) else {}
        if syuzhet_anchor is not None:
            scene_context["syuzhet_anchor"] = syuzhet_anchor

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
            scene_context=scene_context,
            rendering=rendering,
            counterfactual_branch=counterfactual_branch,
            threat_proximity=threat_proximity,
            causal_attribution=causal_attribution,
            entanglement_pairs=entanglement_pairs,
            external_research=self._select_external_research(entity_ids),
            narrative_style=getattr(self.world_state, "narrative_style", None),
        )
        _log_creative_brief(brief)
        return brief

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_external_research(
        self, focus_entity_ids: List[str],
    ) -> List["ResearchHighlight"]:
        """Pick world facts whose ``related_node_ids`` intersect the scene focus.

        Facts with an empty ``related_node_ids`` are treated as
        project-wide and always included (they were attached to the
        project but not to any specific node).
        """
        facts = list(getattr(self.world_state, "world_facts", []) or [])
        if not facts:
            return []
        focus = set(focus_entity_ids or [])
        # Also include any locations / objects that are in scope via ego.
        for k in ("focus_locations", "focus_objects"):
            for item in self.ego.get(k, []) or []:
                if isinstance(item, dict) and "id" in item:
                    focus.add(item["id"])
        out: List[ResearchHighlight] = []
        for f in facts:
            related = list(getattr(f, "related_node_ids", []) or [])
            if related and focus and not (set(related) & focus):
                continue
            out.append(ResearchHighlight(
                fact_id=getattr(f, "id", ""),
                topic=getattr(f, "topic", ""),
                summary=getattr(f, "summary", ""),
                confidence=getattr(f, "confidence", "moderate"),
                source_url_primary=getattr(f, "source_url_primary", None),
            ))
        return out

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
        *,
        syuzhet_anchor: Optional[int] = None,
    ) -> Optional[CounterfactualBranch]:
        """Build the actual vs. simulated outcome for regret rendering.

        Uses the most recent negative event targeting the entities as the
        'actual outcome', and looks for the most recent choice event by the
        entities as the divergence point whose alternate path would have
        led to a better state.

        When ``syuzhet_anchor`` is provided, only events the reader has
        already encountered (``syuzhet_index <= anchor``) are considered.
        """
        eid_set = set(entity_ids)

        def _visible(evt) -> bool:
            return syuzhet_anchor is None or evt.syuzhet_index <= syuzhet_anchor

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
            and _visible(e)
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
            and _visible(e)
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
        *,
        syuzhet_anchor: Optional[int] = None,
    ) -> Optional[CausalAttribution]:
        """For rage: trace the causal chain from a loss back to a perpetrator.

        When ``syuzhet_anchor`` is provided, only events the reader has
        already encountered (``syuzhet_index <= anchor``) are considered
        as candidate loss events.
        """
        eid_set = set(entity_ids)
        causal_g = self._build_causal_digraph()

        def _visible(evt) -> bool:
            return syuzhet_anchor is None or evt.syuzhet_index <= syuzhet_anchor

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
            and _visible(e)
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
                    # Build the causal chain path. The shortest path may
                    # traverse non-event nodes (entities, world traits)
                    # because causal_topology is a mixed-node graph; the
                    # CausalAttribution.causal_chain field is documented
                    # as event IDs only, so we project the path down to
                    # the events on it.
                    event_ids = {e.id for e in self.world_state.events}
                    try:
                        path = nx.shortest_path(causal_g, anc_id, loss_evt.id)
                        causal_chain = [n for n in path if n in event_ids]
                    except nx.NetworkXNoPath:
                        causal_chain = [anc_id, loss_evt.id]
                    if not causal_chain:
                        causal_chain = [loss_evt.id]
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
        r = brief.rendering
        lines.append("  Rendering directive:")
        lines.append(f"    → mode={r.rendering_mode}")
        if r.pov_lock:
            lines.append(f"      pov_lock={r.pov_lock}")
        lines.append(
            f"      pacing={r.pacing}, sensory_focus={r.sensory_focus}"
        )
        if r.tone_arc:
            lines.append(f"      tone_arc={r.tone_arc}")
        if r.stylistic_instructions:
            lines.append(
                f"      stylistic_instructions ({len(r.stylistic_instructions)}):"
            )
            for instr in r.stylistic_instructions[:max_items]:
                lines.append(f"        · {instr}")
            if len(r.stylistic_instructions) > max_items:
                lines.append(
                    f"        … (+{len(r.stylistic_instructions) - max_items} more)"
                )

    logger.info("\n".join(lines))
