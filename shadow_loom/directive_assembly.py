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
from shadow_loom.settings import (
    DirectiveAssemblySettings,
    get_settings as _get_settings,
)

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
            "observation, intervention, counterfactual, manual_edit, "
            "fallback, default. (Interrogation queries return graph "
            "analysis, not prose, and never reach this directive.)"
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


def compute_hidden_channels_for(
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int],
) -> List["HiddenChannel"]:
    """Module-level shim around :meth:`DirectiveAssembler.compute_hidden_channels`.

    Lets non-directive brief builders (observation / intervention /
    counterfactual) populate ``CreativeBrief.hidden_channels`` without
    instantiating a full assembler. The Rung-1/2/3 paths need the same
    withheld-utterance/channel discipline the directive path enforces;
    without it the ``=== HIDDEN CHANNELS / UTTERANCES (HARD) ===``
    block never reaches the renderer/auditor for those queries.
    """
    if syuzhet_anchor is None:
        return []
    assembler = DirectiveAssembler(
        sandbox=None, ego_payload={}, world_state=world_state,
    )
    return assembler.compute_hidden_channels(syuzhet_anchor)


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
        settings: Optional["DirectiveAssemblySettings"] = None,
    ) -> None:
        self.sandbox = sandbox
        self.ego = ego_payload
        self.world_state = world_state
        # Resolve scorer tunables from settings (lazy import to avoid
        # any circular dependency at module-load time). Every class
        # attribute that begins with ``_MYSTERY_``, ``_IRONY_``,
        # ``_SUSPENSE_``, ``_SURPRISE_``, ``_HARM_KIND_SALIENCE``,
        # ``_DEFAULT_HARM_SALIENCE``, ``_TRAIT_NARRATIVE_SALIENCE`` or
        # ``_DEFAULT_TRAIT_SALIENCE`` is read here and shadowed onto
        # the instance, so existing ``self._XXX`` reads in the scorer
        # methods pick up the configured value transparently.
        if settings is None:
            settings = _get_settings().directive_assembly
        self._settings = settings
        s = settings
        # Mystery
        self._MYSTERY_PATH_DECAY_DEPTH = int(s.mystery_path_decay_depth)
        self._MYSTERY_PROXIMITY_TAU_SYUZHET = float(s.mystery_proximity_tau_syuzhet)
        # Irony
        self._IRONY_SURFACE_K = float(s.irony_surface_k)
        self._IRONY_FALSE_BELIEF_MULT = float(s.irony_false_belief_mult)
        self._IRONY_ACTION_ALPHA = float(s.irony_action_alpha)
        self._IRONY_ACTION_WEIGHT_CAP = float(s.irony_action_weight_cap)
        self._IRONY_AGGREGATOR_BETA = float(s.irony_aggregator_beta)
        self._IRONY_PROXIMITY_TAU_SYUZHET = float(s.irony_proximity_tau_syuzhet)
        self._IRONY_PROXIMITY_FLOOR = float(s.irony_proximity_floor)
        # Suspense
        self._SUSPENSE_STAKES_K = float(s.suspense_stakes_k)
        self._SUSPENSE_PROXIMITY_TAU_FABULA_GAPS = float(s.suspense_proximity_tau_fabula_gaps)
        self._SUSPENSE_PROXIMITY_TAU_SPATIAL = float(s.suspense_proximity_tau_spatial)
        self._SUSPENSE_PERSISTENCE_ALPHA = float(s.suspense_persistence_alpha)
        self._SUSPENSE_PERSISTENCE_CAP = float(s.suspense_persistence_cap)
        self._SUSPENSE_HOSTILE_AFFINITY = float(s.suspense_hostile_affinity)
        self._SUSPENSE_ALLY_AFFINITY = float(s.suspense_ally_affinity)
        # Surprise
        self._SURPRISE_TRAIT_KL_WEIGHT = float(s.surprise_trait_kl_weight)
        self._SURPRISE_ANACHRONY_WEIGHT = float(s.surprise_anachrony_weight)
        self._DEFAULT_TRAIT_SALIENCE = float(s.surprise_default_trait_salience)
        self._SURPRISE_SOURCE_EDGE_WEIGHT = float(s.surprise_source_edge_weight)
        self._SURPRISE_PRIOR_PSEUDOCOUNT = float(s.surprise_prior_pseudocount)
        # Salience tables (override class-level defaults so callers
        # can tune the appraisal hierarchy without code changes).
        self._HARM_KIND_SALIENCE = dict(s.harm_kind_salience)
        self._DEFAULT_HARM_SALIENCE = float(s.default_harm_salience)
        self._TRAIT_NARRATIVE_SALIENCE = dict(s.trait_narrative_salience)

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
        """Compute current value + headroom for each trait of given entities.

        Looks first in the ego payload (``focus_entities`` /
        ``present_entities``) where the orchestrator stages
        per-directive trait dicts; falls back to
        ``world_state.entities`` when the requested id has no ego
        entry. The fallback path is essential for any caller that
        constructs a ``DirectiveAssembler`` with an empty ego (the
        affective-curve plot, the audit script, the auditor's
        per-target rescore loop). Without it the emotion scorers
        silently returned ``+1.0`` (worst possible match) for every
        entity, since the trajectory list was empty and the
        no-contributions branch fired.
        """
        trajectories: List[TraitTrajectory] = []

        for eid in entity_ids:
            ent_data = self._find_entity(eid)
            if ent_data:
                trait_iter = (
                    (n, t) for n, t in ent_data.get("traits", {}).items()
                    if isinstance(t, dict)
                )
                resolve = lambda t, k, d: t.get(k, d)
            else:
                ent = self.world_state.entities.get(eid)
                if ent is None or not getattr(ent, "traits", None):
                    continue
                trait_iter = ent.traits.items()
                resolve = lambda t, k, d: getattr(t, k, d)

            for trait_name, trait_data in trait_iter:
                val = resolve(trait_data, "value", 0.5)
                inertia = resolve(trait_data, "inertia", 0.5)
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
    # Maximum reverse-walk depth for path-strength geometric decay
    # (Batch A.1). Truncating at 4 keeps the per-effect dijkstra
    # bounded on dense fixtures while still admitting the chains
    # narrative criticism actually attends to (a 4-link chain is
    # already at the edge of audience traceability per the Trabasso &
    # Sperry 1985 causal-network reading studies). Falls back to the
    # legacy single-edge weight beyond this depth.
    _MYSTERY_PATH_DECAY_DEPTH: int = 4

    # Curiosity proximity decay (Batch A.3 — Iser/Sternberg gap-
    # theory). The reader's curiosity sits over the *most recently
    # surfaced* effects, not the entire revealed cone equally. Effects
    # surfaced long ago are mentally filed as "resolved enough" and
    # no longer drive the gauge. Expressed in raw syuzhet-index
    # units — a value of 8 means "an effect 8 syuzhet beats behind
    # the anchor registers at 1/e ≈ 0.37 of full curiosity weight".
    # Symmetric mirror of the suspense forward-imminence kernel
    # (looking *forward* at upcoming threats vs. *backward* at
    # surfaced unexplained effects).
    _MYSTERY_PROXIMITY_TAU_SYUZHET: float = 8.0

    def compute_mystery_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Mystery: ratio of hidden causal ancestors of known effects.

        Walks backward from each known effect node involving the target
        entities and counts how many of its causal predecessors are NOT
        yet revealed to the reader. Each ancestor's contribution is
        weighted by three multiplicative factors:

        1. **Path strength** — the strongest reverse-path product of
           edge weights from ancestor to effect, depth-capped at
           ``_MYSTERY_PATH_DECAY_DEPTH``. A 3-hop weak chain
           (``0.25 × 0.25 × 0.25 ≈ 0.016``) contributes far less
           curiosity weight than a 1-hop strong link (``0.75``),
           matching the Trabasso & Sperry 1985 causal-network reading
           studies where deep chains decay in audience traceability.
           Replaces the legacy ``out_edges max`` fallback that gave a
           5-hop ancestor as much weight as a 1-hop link.
        2. **Harm-kind salience** — multiplied by
           ``_HARM_KIND_SALIENCE`` (the Lazarus-anchored existential >
           physical > betrayal > … hierarchy already used by suspense
           and dramatic-irony). A hidden murder is more mysterious
           than a hidden gossip exchange even when the path strengths
           are identical, capturing Sternberg's "expositional gap"
           weighting by the stake of the missing piece.
        3. **Per-effect proximity** to the syuzhet anchor — recently
           surfaced effects drive the gauge; long-resolved gaps do
           not (Iser 1976 *Akt des Lesens* / Sternberg 1992 curiosity
           taxonomy). Skipped for entity-as-effect nodes which carry
           no syuzhet position.

        Returns a ratio in [0, 1].
        """
        causal_g = self._build_causal_digraph()
        revealed = self._revealed_event_ids(syuzhet_anchor)
        eid_set = set(entity_ids)
        events_by_id = {e.id: e for e in self.world_state.events}

        # Identify "effect" nodes: revealed events involving target entities,
        # plus the entities themselves (which can be causal targets).
        effect_nodes: set[str] = set()
        for evt in self.world_state.events:
            if evt.id in revealed and (
                set(evt.actor_ids) & eid_set or set(evt.target_ids) & eid_set
            ):
                effect_nodes.add(evt.id)
        effect_nodes |= eid_set

        # Reverse causal graph for backward dijkstra: edge weight =
        # ``-log(forward_w)`` so the shortest path = the highest
        # product of forward weights (the strongest causal chain).
        # Computed once and reused across effect nodes.
        rev_g = nx.DiGraph()
        for u, v, d in causal_g.edges(data=True):
            w = max(float(d.get("weight", 0.5)), 1e-6)
            rev_g.add_edge(v, u, neglogw=-math.log(w))

        def _curiosity_proximity(eff_id: str) -> float:
            """Backward-looking decay weight on an effect's contribution."""
            if syuzhet_anchor is None:
                return 1.0
            evt = events_by_id.get(eff_id)
            if evt is None:
                return 1.0  # entity-as-effect (no syuzhet position)
            delta = max(0, syuzhet_anchor - evt.syuzhet_index)
            return math.exp(-delta / self._MYSTERY_PROXIMITY_TAU_SYUZHET)

        total_mass = 0.0
        hidden_mass = 0.0

        for eff in effect_nodes:
            if not causal_g.has_node(eff):
                continue
            if not rev_g.has_node(eff):
                continue
            # Bounded reverse dijkstra: per-ancestor min cumulative
            # ``-log w``, equivalent to max cumulative ``∏ w``. The
            # depth cutoff is enforced via ``cutoff`` on the
            # ``neglogw`` axis at ``-log(strongest_edge ** depth)`` =
            # ``depth · max(-log w)`` per safety; in practice the
            # per-hop count is bounded directly by switching to a
            # bounded BFS depth filter.
            try:
                # Two-pass: depths via unweighted BFS for the depth
                # filter, then dijkstra distances for path-product.
                depths = nx.single_source_shortest_path_length(
                    rev_g, eff, cutoff=self._MYSTERY_PATH_DECAY_DEPTH,
                )
                dists = nx.single_source_dijkstra_path_length(
                    rev_g, eff, weight="neglogw",
                )
            except (nx.NetworkXError, nx.NodeNotFound):
                continue
            eff_proximity = _curiosity_proximity(eff)
            for anc, depth in depths.items():
                if anc == eff or depth == 0:
                    continue
                neglogw = dists.get(anc)
                if neglogw is None:
                    continue
                path_w = math.exp(-neglogw)
                # Harm-kind salience reuse — Lazarus / OCC / Brewer
                # hierarchy. Only meaningful for events; entities use
                # the default mid-tier weight.
                if anc in events_by_id:
                    _, salience = self._harm_kind_for_event(anc, causal_g)
                else:
                    salience = self._DEFAULT_HARM_SALIENCE
                w = path_w * salience * eff_proximity
                total_mass += w
                if anc not in revealed:
                    hidden_mass += w

        if total_mass == 0.0:
            return 0.0

        score = hidden_mass / total_mass
        logger.debug(
            "[DirectiveAssembly·Mystery] hidden_mass=%.3f / total_mass=%.3f = %.3f "
            "(path-decay depth=%d, τ_curiosity=%.1f)",
            hidden_mass, total_mass, score,
            self._MYSTERY_PATH_DECAY_DEPTH,
            self._MYSTERY_PROXIMITY_TAU_SYUZHET,
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

    # Batch A.1 — false-belief multiplier (Pfister 1977 tragic-irony;
    # Cabanas Gonzalez 2024 ToM thesis). When the focal character
    # holds an explicit belief about an actor of a revealed event the
    # focal does NOT know about, the focal is acting under an outdated
    # picture of that actor — the canonical Iago→Othello / Jacqueline→
    # Linnet pattern where the audience watches the focal trust the
    # very person undermining them. Multiplied into the gap mass for
    # those events. Set above 1.0 to upweight false-belief gaps over
    # plain ignorance gaps; calibrated empirically — much above 2.0
    # the scorer pegs near 1.0 on every Iago-pattern fixture.
    _IRONY_FALSE_BELIEF_MULT: float = 1.5

    # Batch A.2 — action-weighting per character (Pfister 1977
    # protagonist-prominence; Sutherland 2013 *A Little History of
    # Literature* on irony as protagonist's blindness). Weights each
    # focal character's gap by their causal-out-degree on the syuzhet
    # axis up to the anchor — a character actively driving the plot
    # under false information carries more dramatic charge than a
    # bystander. Macduff's ignorance of the murders matters more than
    # Lennox's because Macduff is *acting* on a false picture.
    # Formula: ``c_weight = 1 + α · count(events where c is actor at
    # syuzhet ≤ anchor)``.
    _IRONY_ACTION_ALPHA: float = 0.15
    _IRONY_ACTION_WEIGHT_CAP: float = 3.0

    # Batch B.4 — weighted-max aggregator across the focal cast
    # (Sternberg's single-dominant-gap framing). The mean across
    # ``entity_ids`` was diluting a strong protagonist gap into a
    # cast average; the literature treats *one* character's tragic
    # blindness as carrying the irony charge, with the secondary
    # cast as backdrop. We now blend ``β · max(gaps) + (1-β) ·
    # mean(gaps)`` so the dominant gap drives the gauge while
    # secondary characters still register a residual contribution.
    # β = 0.6 calibrated against the example_worlds corpus to keep
    # ensemble fixtures (Reservoir Dogs cast, Tinker Tailor) within
    # the same band as solo-protagonist fixtures (Macbeth, Gone
    # Girl) with the same per-character maximum.
    _IRONY_AGGREGATOR_BETA: float = 0.6

    # Batch B.5 — proximity-to-closure decay (Booth 1974 stable vs
    # unstable irony). A gap that is about to be *closed* by a
    # near-future revelation is more dramatically charged than one
    # that will linger; conversely, a gap that closed long ago is
    # already discharged. We weight per-event gap mass by
    # ``exp(-Δ_to_closure / τ_irony)`` where Δ_to_closure is the
    # absolute distance in syuzhet-index units between the current
    # anchor and the syuzhet index at which the focal first comes
    # to know the event. Events the focal never learns within the
    # story's syuzhet axis contribute at the ``floor`` rate (0.4)
    # — they're permanently ironic but not pre-denouement
    # spike-worthy. Auto-scales: uses raw syuzhet-index units,
    # already integers, default τ ≈ 6 syuzhet beats.
    _IRONY_PROXIMITY_TAU_SYUZHET: float = 6.0
    _IRONY_PROXIMITY_FLOOR: float = 0.4

    def compute_dramatic_irony_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Dramatic Irony: information asymmetry where reader knows more.

        For each focal entity, counts *intensity-weighted* revealed
        events the focal entity does NOT know about, divided by the
        **revealed** event mass plus a saturation constant ``K``:

        .. math::

           \\text{irony} =
              \\frac{1}{|F|} \\sum_{c \\in F}
              \\frac{\\sum_{e \\in \\text{revealed}, e \\notin K_c} w_e}
                   {\\sum_{e \\in \\text{revealed}} w_e + K}

        Sternberg (1978; 1992) anchors the structural-affect framing
        in which dramatic irony is one of three reader-information
        states (alongside curiosity / mystery and surprise); Booth
        (1974), Muecke (1969) and Gerrig (1993) supply the specific
        "audience knows what the character does not" definition this
        scorer operationalises as the *gap* between what the reader
        has been shown and what the focal character knows.
        Normalising by ``revealed_mass`` (rather than the full
        story's event mass) makes the ratio dimensionally a *fraction
        of revealed material the character is in the dark about*. As
        the character participates in further events their ``known``
        set grows and the gap fraction *falls* — producing the
        canonical rise-then-fall arc theory predicts (peak around the
        midpoint when reader privilege is greatest, collapse near the
        climax when characters learn the truth: Macduff hearing of
        his family, Poirot's denouement, Nick's letter to Daisy).
        The earlier ``total_mass + K`` denominator was constant in
        the syuzhet anchor, so the numerator's monotone growth with
        reveals pinned the curve into a monotone *rise*, contradicting
        the rise-then-fall arc.

        where :math:`F` is the focal cast, :math:`K_c` is what
        character :math:`c` knows (participation, addressed
        utterances, beliefs with valid provenance), and :math:`w_e` is
        event ``e``'s intensity (defaults to ``1.0``). Character
        knowledge is bounded by the syuzhet anchor's fabula frontier,
        so a character isn't credited with knowing their own future
        arc from anchor 0.

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

        # Causal graph for harm-kind salience lookup (Batch A.3).
        # Built once and reused across focal entities.
        causal_g = self._build_causal_digraph()

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
            # whose target_id matches the event id, where the belief's
            # provenance (``acquired_via_event_id`` /
            # ``acquired_via_channel_id``) still resolves in this branch.
            #
            # Channel intelligibility gates (b): a recipient who cannot
            # parse the channel (intelligibility[recipient] below the
            # configured threshold) does NOT learn from the utterance
            # even if they were nominally addressed. Without this gate,
            # encrypted/coded/foreign-language messages would silently
            # close dramatic-irony surfaces that should remain open.
            #
            # Belief provenance gates (c): when a shadow-branch surgery
            # prunes the utterance or severs the channel that originally
            # posted a belief, that belief stops counting toward the
            # character's knowledge. This is what makes shadow-branch
            # irony deltas track the actual epistemic consequence of
            # the surgery rather than only the textual disappearance
            # of the belief node.
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
                # The utterance event itself is part of what the
                # speaker / addressee witnesses — being on either end
                # of a (intelligible) message is direct knowledge of
                # the message having been transmitted, not just of
                # what it referred to.
                known.add(utt.id)
                for tid in utt.target_ids:
                    if tid.startswith("EVT_"):
                        known.add(tid)
            # Reconstruct beliefs at the fabula frontier so we include
            # beliefs *acquired during the story* (via utterances,
            # observations, channel traffic) and not just the entity's
            # pre-story baseline. ``reconstruct_entity_at`` also filters
            # by ``established_at_fabula <= fabula_frontier`` so we never
            # credit a character with knowledge from their own future arc.
            recon_beliefs = reconstruct_entity_at(
                ent, fabula_frontier
            ).get("beliefs", [])

            # Provenance gate: a belief should only count toward the
            # character's knowledge in *this* branch when its causing
            # event / channel still exists and remains accessible.
            # Counterfactual surgery (shadow merges) frequently removes
            # the utterance or severs the channel that originally posted
            # the belief; without this gate, the irony scorer credits
            # the character with knowledge they no longer have any way
            # of holding, and shadow-branch irony deltas collapse.
            #
            # The gate fires when the provenance pointer is *non-null
            # and dangling*: a None pointer (legacy beliefs, baseline
            # knowledge, in-person observation) is left untouched so
            # this enhancement is fully backward-compatible with
            # fixtures predating the provenance fields.
            def _provenance_valid(b: dict) -> bool:
                via_evt = b.get("acquired_via_event_id")
                if via_evt is not None and via_evt not in events_by_id:
                    return False  # provenance event pruned in this branch
                via_ch = b.get("acquired_via_channel_id")
                if via_ch is not None:
                    ch = channels.get(via_ch)
                    if ch is None:
                        return False  # channel pruned (line cut, bond severed)
                    intel = float(ch.intelligibility.get(eid, 1.0))
                    if intel < intel_thresh:
                        return False  # channel exists but no longer intelligible
                    # Channel temporal bounds: if the channel was
                    # severed before the belief's nominal acquisition
                    # time, the belief is no longer reachable in this
                    # branch (e.g. shadow surgery brings the
                    # termination forward).
                    term = ch.terminated_at_fabula
                    est = b.get("established_at_fabula", 0)
                    if term is not None and term < est:
                        return False
                return True

            known |= {
                b["target_id"] for b in recon_beliefs
                if b["target_id"].startswith("EVT_")
                and _provenance_valid(b)
            }

            # Batch A.1 — false-belief surface. Build the set of
            # entity targets the focal holds beliefs about (and whose
            # provenance still resolves). A revealed event whose
            # actor set intersects this set triggers the false-belief
            # multiplier on the gap mass: the focal is operating
            # under an outdated model of someone whose latest
            # actions they have not been shown. This is the
            # mechanical proxy for tragic irony à la Iago→Othello,
            # Jacqueline→Linnet and Hero→Claudio (Pfister 1977,
            # Cabanas Gonzalez 2024). We deliberately do not try to
            # semantically compare ``Belief.perceived_state`` strings
            # to world truth — model entities the focal *has formed
            # an opinion about* are the tractable false-belief
            # surface, and the test is invariant under shadow surgery
            # via the same provenance gate as the event-belief side.
            believed_entity_targets: set[str] = {
                b["target_id"] for b in recon_beliefs
                if not b["target_id"].startswith("EVT_")
                and _provenance_valid(b)
            }

            # Batch A.2 — action weighting. Per-character prominence
            # weight: a focal who has authored more revealed events
            # is more dramatically central; their ignorance carries
            # more irony. Capped to keep ensemble fixtures balanced.
            action_count = sum(
                1 for evt in self.world_state.events
                if evt.id in revealed and eid in evt.actor_ids
            )
            action_weight = min(
                self._IRONY_ACTION_WEIGHT_CAP,
                1.0 + self._IRONY_ACTION_ALPHA * action_count,
            )

            # Batch B.5 — closure proximity per gap event. For each
            # revealed event ``e`` not in ``known``, find the
            # *earliest* later syuzhet position at which the focal
            # first witnesses an event with ``fabula_time >=
            # e.fabula_time`` — the dramatic moment the focal walks
            # into the scene that exposes the truth. A near closure
            # gives a sharp pre-denouement spike; a distant closure
            # is fully ironic but contributes at the imminence
            # floor. Events the focal never learns within the
            # syuzhet axis fall to the floor too. Cached per focal.
            participation_syuzhet: list[Tuple[int, int]] = sorted(
                (evt.syuzhet_index, evt.fabula_time)
                for evt in self.world_state.events
                if eid in evt.actor_ids or eid in evt.target_ids
            )

            def _closure_proximity(evt) -> float:
                target_fab = evt.fabula_time
                later = min(
                    (s for s, f in participation_syuzhet
                     if s > syuzhet_anchor and f >= target_fab),
                    default=None,
                )
                if later is None:
                    return self._IRONY_PROXIMITY_FLOOR
                delta = max(0, later - syuzhet_anchor)
                imminence = math.exp(
                    -delta / self._IRONY_PROXIMITY_TAU_SYUZHET
                )
                # Blend against the floor so even very-far closures
                # still carry the standing irony contribution.
                return max(self._IRONY_PROXIMITY_FLOOR, imminence)

            # Intensity-weighted mass of revealed events this character
            # does NOT know — the per-character irony surface. Each
            # gap event's contribution is now scaled by:
            #   * harm-kind salience (Batch A.3 — Lazarus / OCC
            #     hierarchy reuse: a hidden mortal threat is more
            #     dramatically ironic than a hidden small-talk
            #     exchange);
            #   * false-belief multiplier (Batch A.1) when the gap
            #     event's actor set intersects the focal's believed-
            #     entity targets — the focal is acting on an
            #     outdated picture of one of the perpetrators;
            #   * closure-proximity (Batch B.5) — sharper for
            #     pre-denouement events, floor-rated for permanently
            #     ironic ones (the focal never learns).
            gap_mass = 0.0
            for reid in revealed:
                if reid in known:
                    continue
                evt = events_by_id[reid]
                w = _evt_w(evt)
                _, salience = self._harm_kind_for_event(reid, causal_g)
                w *= salience
                if believed_entity_targets and (
                    set(evt.actor_ids) & believed_entity_targets
                ):
                    w *= self._IRONY_FALSE_BELIEF_MULT
                w *= _closure_proximity(evt)
                gap_mass += w

            # Normalise by *revealed* mass (Sternberg gap fraction)
            # rather than total mass: the gauge then expresses "what
            # share of the reader's privileged view the character is
            # blind to", a quantity that naturally falls as the
            # character catches up via late-story revelations. The
            # action-weight scales the per-character contribution
            # before averaging; characters with higher prominence
            # pull the score up more (Pfister 1977 protagonist-
            # blindness).
            per_character_gaps.append(
                action_weight * gap_mass
                / (revealed_mass + self._IRONY_SURFACE_K)
            )

        if not per_character_gaps:
            return 0.0

        # Batch B.4 — weighted-max combine across the focal cast.
        # Sternberg's structural-affect framing treats *one*
        # character's tragic blindness as carrying the irony charge
        # rather than the cast average. β · max + (1-β) · mean keeps
        # the secondary cast as a residual contribution while the
        # dominant gap drives the gauge. Final clamp at 1.0 since
        # action-weight × false-belief multiplier × harm-salience can
        # push individual gaps above the [0, 1] bound; the clamp is
        # the well-defined upper limit of the gauge itself, not a
        # silent truncation of an unbounded quantity.
        max_gap = max(per_character_gaps)
        mean_gap = sum(per_character_gaps) / len(per_character_gaps)
        score = min(
            self._IRONY_AGGREGATOR_BETA * max_gap
            + (1.0 - self._IRONY_AGGREGATOR_BETA) * mean_gap,
            1.0,
        )
        logger.debug(
            "[DirectiveAssembly·DramaticIrony] revealed_mass=%.3f "
            "(of total=%.3f, K=%.2f), per-char gaps=%s, "
            "max=%.3f mean=%.3f β=%.2f → %.3f",
            revealed_mass, total_mass, self._IRONY_SURFACE_K,
            [round(g, 3) for g in per_character_gaps],
            max_gap, mean_gap, self._IRONY_AGGREGATOR_BETA, score,
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
    #
    # Aggregate fall-back used when a kind has no per-kind override.
    _SUSPENSE_STAKES_K: float = 2.0

    # Per-kind saturation constants (improvement #6). Existential and
    # physical threats don't saturate quickly — one death threat does
    # not max out the gauge — so K_kind is large. Social / epistemic
    # threats do saturate quickly (three slights and the reader is
    # bored), so K_kind is small. Calibrated against the example_worlds
    # corpus so a single mid-anchor mortal threat lands in the 0.3-0.5
    # stakes band rather than flooring at near-zero or pegging to 1.0.
    _SUSPENSE_STAKES_K_BY_KIND: Dict[str, float] = {
        "existential": 4.0,
        "physical": 3.0,
        "betrayal": 2.5,
        "psychological": 2.0,
        "emotional": 2.0,
        "social": 1.5,
        "epistemic": 1.5,
        "informational": 1.5,
    }

    # Anticipatory-proximity decay (improvement #2 — Comisky & Bryant
    # 1982). Subjective probability of a threat rises with imminence.
    # ``imminence = exp(-Δ_fabula / τ_t) · exp(-Δ_spatial / τ_s)``.
    #
    # τ_fabula is expressed in *units of typical inter-event gaps*
    # (the median gap between consecutive event ``fabula_time``
    # values in the world), so it auto-scales whether the world uses
    # unit spacing or the default ``fabula_time_spacing=1000``. A
    # value of 6 means "an event 6 typical inter-event gaps in the
    # future registers at 1/e ≈ 0.37 of full weight". τ_spatial is
    # in raw graph hops (already unitless).
    _SUSPENSE_PROXIMITY_TAU_FABULA_GAPS: float = 6.0
    _SUSPENSE_PROXIMITY_TAU_SPATIAL: float = 4.0

    # Persistence multiplier (improvement #4 — Brewer & Lichtenstein
    # 1982 initiating-event arc). The longer a foreshadowed threat
    # lingers unresolved, the louder it gets. We measure persistence
    # as the count of *revealed* causal ancestors (proxy for "how long
    # the gun has been on the mantle"). Multiplier =
    # min(cap, 1 + α·persistence).
    _SUSPENSE_PERSISTENCE_ALPHA: float = 0.10
    _SUSPENSE_PERSISTENCE_CAP: float = 1.5

    # Disposition thresholds (improvement #1 — Zillmann 1996).
    # Affinity ≤ HOSTILE → actor counts as a hostile force on the
    # focal entity (their authored event registers as *threat*, not
    # hope). Affinity ≥ ALLY → actor counts as a rescuer / ally
    # (their authored event registers as *hope* even when the focal
    # entity is its target — improvement #5). The neutral band
    # (HOSTILE, ALLY) keeps the legacy actor=hope / target=threat
    # rule, so worlds without a populated social_topology degrade
    # gracefully to the prior behaviour.
    _SUSPENSE_HOSTILE_AFFINITY: float = -0.2
    _SUSPENSE_ALLY_AFFINITY: float = 0.2

    # Harm-kind salience weights for the suspense ledger.
    #
    # The threat/hope ledger remains a single scalar, but each
    # contributing event is now multiplied by a *salience weight* that
    # depends on the kind of harm (or hope) the event embodies. The
    # ranking follows the appraisal-theory hierarchy (Lazarus 1991
    # core relational themes; Ortony, Clore & Collins 1988 OCC
    # prospect-based emotions; Brewer & Lichtenstein 1982 structural-
    # affect): mortal/existential threats outweigh physical violence,
    # which outweighs betrayal-class moral injury, which outweighs
    # social/reputational shame, which outweighs informational/
    # epistemic destabilisation. The same ordering is used on the
    # hope side (rescue from death > physical safety > restored
    # loyalty > restored standing > restored knowledge), so a
    # high-stakes mortal threat pairs naturally with a high-stakes
    # mortal hope rather than being averaged into a flat "stuff
    # happens" weight.
    #
    # The kinds are inferred from the canonical ``mechanism`` strings
    # carried on incident causal edges (see
    # ``causal_physics.MECHANISM_TRAIT_MAP``). Each event takes the
    # MAX salience across its incident-edge mechanisms — the most
    # salient kind wins, so a "stab in the back" event wired with both
    # ``physical`` and ``betrayal`` edges registers at the higher of
    # the two rather than being averaged. Events with no resolvable
    # mechanism default to ``physical`` (the modal harm kind in the
    # corpus and the median salience), so the gauge degrades
    # gracefully on sparse fixtures.
    _HARM_KIND_SALIENCE: Dict[str, float] = {
        # Existential / mortal — irrevocable loss outranks all others
        # in OCC's prospect-based emotion hierarchy.
        "existential": 1.00,
        # Physical violence — Lazarus's "physical danger" theme.
        "physical": 0.85,
        # Betrayal / moral injury — Lazarus's "moral transgression";
        # narratively second only to mortal threat in the example
        # corpus (Gone Girl, Reservoir Dogs, Tinker Tailor).
        "betrayal": 0.75,
        # Psychological collapse (despair, breakdown) — OCC's
        # "distress about a self-relevant prospect" cluster.
        "psychological": 0.70,
        # Relational rupture — Lazarus's "relational loss"; the
        # romance / fellowship axis (Wuthering Heights, Persuasion).
        "emotional": 0.65,
        # Social / reputational — Lazarus's "social esteem / shame".
        "social": 0.55,
        # Epistemic destabilisation — discovery / exposure as a
        # *prospect* (the threat that the truth comes out). Lower
        # default salience because the cumulative-mystery scorer
        # already covers the epistemic surface; suspense should not
        # double-count it.
        "epistemic": 0.45,
        "informational": 0.45,
    }
    _DEFAULT_HARM_SALIENCE: float = _HARM_KIND_SALIENCE["physical"]

    def _harm_kind_for_event(
        self, evt_id: str, causal_g: nx.DiGraph,
    ) -> Tuple[str, float]:
        """Resolve the most-salient harm-kind for an unrevealed event.

        Reads the ``mechanism`` attribute on every incident causal
        edge (in or out) of ``evt_id`` and returns the
        ``(kind, salience)`` pair with the highest salience weight.
        Falls back to ``("physical", _DEFAULT_HARM_SALIENCE)`` when
        the event has no incident edges or none of the mechanisms
        resolve to a known kind.
        """
        if not causal_g.has_node(evt_id):
            return "physical", self._DEFAULT_HARM_SALIENCE
        best_kind = "physical"
        best_w = self._DEFAULT_HARM_SALIENCE
        seen_known = False
        for _, _, d in list(causal_g.in_edges(evt_id, data=True)) + list(
            causal_g.out_edges(evt_id, data=True)
        ):
            mech = (d.get("mechanism") or "").strip().lower()
            if not mech:
                continue
            # Map long-form aliases onto the salience table keys.
            if mech in ("physical_force",):
                mech = "physical"
            elif mech in ("epistemic_revelation",):
                mech = "epistemic"
            elif mech in ("social_coercion",):
                mech = "social"
            w = self._HARM_KIND_SALIENCE.get(mech)
            if w is None:
                continue
            if not seen_known or w > best_w:
                best_kind = mech
                best_w = w
                seen_known = True
        return best_kind, best_w

    def _build_affinity_index(self) -> Dict[Tuple[str, str], float]:
        """Build a directed ``(src,tgt) → affinity`` lookup.

        Used by the disposition-aware classification (improvement #1)
        and the rescue-propagation rule (improvement #5). Returns an
        empty dict when the world has no ``social_topology`` — callers
        treat missing entries as neutral (0.0) so worlds without a
        populated topology degrade to the legacy actor/target rule.
        """
        idx: Dict[Tuple[str, str], float] = {}
        for rel in self.world_state.social_topology:
            m = rel.metrics.get("affinity")
            if m is None:
                continue
            try:
                v = float(m.value)
            except (TypeError, ValueError):
                continue
            idx[(rel.source_entity_id, rel.target_entity_id)] = v
        return idx

    def _affinity_to(
        self,
        affinity_idx: Dict[Tuple[str, str], float],
        actor_id: str,
        focal_id: str,
    ) -> float:
        """Best-effort affinity from ``actor_id`` toward ``focal_id``.

        Uses directed (actor → focal) when present; falls back to the
        reverse direction (focal → actor); else 0.0 (neutral). Self-
        affinity (actor == focal) returns 1.0 so an entity is always
        treated as an ally of itself for rescue-propagation symmetry.
        """
        if actor_id == focal_id:
            return 1.0
        v = affinity_idx.get((actor_id, focal_id))
        if v is not None:
            return v
        v = affinity_idx.get((focal_id, actor_id))
        return v if v is not None else 0.0

    def _event_location_id(self, evt) -> Optional[str]:
        """Resolve the location an event occurs at.

        Prefers the first actor's current location; falls back to the
        first target's location. Returns ``None`` when no participant
        carries a ``location_id`` (e.g. abstract / world-level events).
        """
        for eid in list(evt.actor_ids) + list(evt.target_ids):
            ent = self.world_state.entities.get(eid)
            if ent is not None and getattr(ent, "location_id", None):
                return ent.location_id
        return None

    def _build_spatial_graph(self) -> Optional[nx.Graph]:
        """Build an undirected, traversable spatial graph for distance."""
        if not getattr(self.world_state, "spatial_topology", None):
            return None
        g = nx.Graph()
        for edge in self.world_state.spatial_topology:
            if getattr(edge, "is_locked", False):
                continue
            if getattr(edge, "destroyed_at_fabula", None) is not None:
                continue
            src = getattr(edge, "source_id", None)
            tgt = getattr(edge, "target_id", None)
            if src and tgt:
                g.add_edge(src, tgt)
        return g if g.number_of_nodes() > 0 else None

    def _fabula_now(self, revealed: set[str]) -> Optional[int]:
        """The latest revealed ``fabula_time`` — the reader's "now"."""
        events_by_id = {e.id: e for e in self.world_state.events}
        times = [
            events_by_id[eid].fabula_time
            for eid in revealed if eid in events_by_id
        ]
        return max(times) if times else None

    def _revealed_ancestor_count(
        self, evt_id: str, causal_g: nx.DiGraph, revealed: set[str],
    ) -> int:
        """Count revealed causal ancestors of ``evt_id`` (proxy for
        how long the threat has been foreshadowed). Used by the
        persistence multiplier (improvement #4)."""
        if not causal_g.has_node(evt_id):
            return 0
        try:
            ancestors = nx.ancestors(causal_g, evt_id)
        except (nx.NetworkXError, nx.NodeNotFound):
            return 0
        return sum(1 for a in ancestors if a in revealed)

    def _bucket_event_for_focal(
        self,
        evt,
        focal_id: str,
        affinity_idx: Dict[Tuple[str, str], float],
    ) -> Optional[str]:
        """Resolve the threat/hope bucket for an event w.r.t. one focal entity.

        Encapsulates the disposition-aware classification so it can be
        reused for both the unrevealed-event ledger (next-period
        belief perturbations) and the revealed-event ledger (the
        belief prior used by the EFK martingale aggregator).

        Returns ``"threat"``, ``"hope"``, or ``None``. Rules in
        priority order:

        1. Focal is target with no co-actor role and an *allied*
           actor is intervening → HOPE (rescue, #5).
        2. Focal is target with no co-actor role → THREAT (legacy /
           hostile-actor reinforced).
        3. Focal is itself an actor with a hostile co-actor present
           → THREAT (coerced participation).
        4. Focal is an actor → HOPE (legacy).
        5. Focal is neither but a hostile actor strikes an allied
           target → THREAT (third-party widening, #1).
        6. Otherwise → ``None``.
        """
        actor_set = set(evt.actor_ids)
        target_set = set(evt.target_ids)
        is_actor = focal_id in actor_set
        is_target = focal_id in target_set

        hostile_co_actor = any(
            self._affinity_to(affinity_idx, a, focal_id)
            <= self._SUSPENSE_HOSTILE_AFFINITY
            for a in actor_set if a != focal_id
        )
        allied_co_actor = any(
            self._affinity_to(affinity_idx, a, focal_id)
            >= self._SUSPENSE_ALLY_AFFINITY
            for a in actor_set if a != focal_id
        )

        if is_target and not is_actor:
            if allied_co_actor and not hostile_co_actor:
                return "hope"
            return "threat"
        if is_actor:
            return "threat" if hostile_co_actor else "hope"
        # Third-party widening (#1).
        if any(
            self._affinity_to(affinity_idx, a, focal_id)
            <= self._SUSPENSE_HOSTILE_AFFINITY
            for a in actor_set
        ) and any(
            self._affinity_to(affinity_idx, t, focal_id)
            >= self._SUSPENSE_ALLY_AFFINITY
            for t in target_set
        ):
            return "threat"
        return None

    def _event_base_weight(
        self, evt_id: str, causal_g: nx.DiGraph,
    ) -> Tuple[float, str, float]:
        """Return ``(prob, kind, salience)`` for an event.

        Shared by both the unrevealed-perturbation pass and the
        revealed-prior pass of the EFK aggregator.
        """
        prob = 0.5
        if causal_g.has_node(evt_id):
            in_edges = list(causal_g.in_edges(evt_id, data=True))
            out_edges = list(causal_g.out_edges(evt_id, data=True))
            if in_edges:
                prob = max(d.get("weight", 0.5) for _, _, d in in_edges)
            elif out_edges:
                prob = max(d.get("weight", 0.5) for _, _, d in out_edges)
        prob = max(0.0, min(1.0, prob))
        kind, salience = self._harm_kind_for_event(evt_id, causal_g)
        return prob, kind, salience

    def compute_suspense_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
        *,
        mode: str = "efk",
    ) -> float:
        """Suspense: forward causal momentum between opposed outcomes.

        Theory layer (improvements #1–#7 over the basic
        actor/target tally):

        * **#1 Disposition-aware classification** (Zillmann 1996
          disposition theory). An *actor*'s contribution flips to
          *threat* for the focal entity when the actor's affinity
          toward that entity is hostile (≤ ``_SUSPENSE_HOSTILE_AFFINITY``).
          Conversely, *#5 rescue propagation* — when the focal entity
          is the *target* and the acting party is an ally
          (affinity ≥ ``_SUSPENSE_ALLY_AFFINITY``), the event registers
          as *hope* for the focal entity.
        * **#2 Anticipatory proximity weighting** (Comisky & Bryant
          1982). Each event's contribution is multiplied by an
          imminence kernel ``exp(-Δ_fabula / τ_t) · exp(-Δ_spatial / τ_s)``,
          so threats that are temporally and spatially close weigh
          more heavily than equally-probable distant ones.
        * **#3 Per-kind balance × stakes** with a salience-weighted
          *max* combine (Brewer & Lichtenstein 1982 dominant-beat),
          available via ``mode='classic'``. A coherent single-kind
          tension dominates a diffuse multi-kind one rather than being
          averaged into it.
        * **#4 Persistence / exposure** — multiplier ``1 + α·a`` where
          ``a`` is the count of revealed causal ancestors of the
          unrevealed threat (proxy for how long it has been
          foreshadowed), capped at ``_SUSPENSE_PERSISTENCE_CAP``.
        * **#6 Per-kind saturation** — ``stakes^k = total^k /
          (total^k + K_k)`` where ``K_k`` rises with kind salience so
          existential threats saturate slowly while social ones
          saturate fast.
        * **#7 EFK expected-variance** (default, ``mode='efk'``) —
          aggregates per-kind ledgers as a salience-weighted average
          of Bernoulli outcome variances ``p(1-p)`` (Ely, Frankel &
          Kamenica 2015), giving suspense the same Bayesian-belief
          shape that the surprise scorer already has. Pass
          ``mode='classic'`` to recover the Brewer-Lichtenstein
          balance × stakes aggregator instead.

        Returns a clamped ``[0, 1]`` score. Returns 0 when hope is
        entirely extinguished (despair) or when no threat is present
        (safety) — both still degenerate to non-suspense as required
        by the test contract.
        """
        causal_g = self._build_causal_digraph()
        revealed = self._revealed_event_ids(syuzhet_anchor)
        all_evt_ids = {e.id for e in self.world_state.events}
        unrevealed = all_evt_ids - revealed
        eid_set = set(entity_ids)

        affinity_idx = self._build_affinity_index()
        spatial_g = self._build_spatial_graph()
        fabula_now = self._fabula_now(revealed)
        events_by_id = {e.id: e for e in self.world_state.events}

        # Auto-scale the fabula-time decay constant to the world's
        # actual inter-event spacing (#2). Use the median gap between
        # consecutive distinct fabula_times so a world using
        # ``fabula_time_spacing=1000`` and one using unit spacing both
        # decay over ~6 narrative beats rather than 6 raw ticks.
        sorted_fts = sorted({e.fabula_time for e in self.world_state.events})
        if len(sorted_fts) >= 2:
            gaps = [b - a for a, b in zip(sorted_fts, sorted_fts[1:]) if b > a]
            typical_gap = float(sorted(gaps)[len(gaps) // 2]) if gaps else 1.0
        else:
            typical_gap = 1.0
        tau_fabula = max(
            1.0, self._SUSPENSE_PROXIMITY_TAU_FABULA_GAPS * typical_gap,
        )

        # Per-kind ledgers (improvement #3) replace the single-bucket
        # threat_weight / hope_weight. Aggregate sums are still tracked
        # for legacy diagnostics / despair-and-safety degenerate
        # checks.
        threat_by_kind: Dict[str, float] = {}
        hope_by_kind: Dict[str, float] = {}
        threat_weight = 0.0
        hope_weight = 0.0

        # EFK belief-martingale ledgers (improvement #7, full form):
        # per (focal_entity, kind) we track
        #   * the *unrevealed* events that could perturb belief next,
        #     each as (bucket, weight, proximity), where ``proximity``
        #     is the un-normalised next-revelation prior; and
        #   * the *revealed* threat/hope mass already accumulated
        #     (A, B), used to form the Beta-posterior belief μ_t.
        # These are populated in mode='efk' only — they cost an extra
        # pass over the (much smaller) revealed-event set.
        unrevealed_by_focal_kind: Dict[
            Tuple[str, str], List[Tuple[str, float, float]],
        ] = {}
        revealed_by_focal_kind: Dict[Tuple[str, str], List[float]] = {}

        # Spatial-distance memo so we don't pay the BFS cost twice for
        # the same (evt_loc, focal_loc) pair across focal entities.
        spatial_memo: Dict[Tuple[str, str], Optional[int]] = {}

        def _spatial_dist(evt_loc: Optional[str], focal_loc: Optional[str]) -> Optional[int]:
            if not (evt_loc and focal_loc) or spatial_g is None:
                return None
            if evt_loc == focal_loc:
                return 0
            key = (evt_loc, focal_loc)
            if key in spatial_memo:
                return spatial_memo[key]
            try:
                d = nx.shortest_path_length(spatial_g, evt_loc, focal_loc)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = None
            spatial_memo[key] = d
            return d

        for evt_id in unrevealed:
            evt = events_by_id.get(evt_id)
            if not evt:
                continue
            actor_set = set(evt.actor_ids)
            target_set = set(evt.target_ids)
            if not (actor_set & eid_set) and not (target_set & eid_set):
                # Disposition-aware classification (#1) widens the net:
                # an event between two non-focal entities can still be
                # a *threat* to the focal entity if any actor is
                # hostile to it AND any target is allied with it
                # (e.g. the villain striking the hero's lover). For
                # symmetry with rescue propagation (#5), this only
                # fires when at least one *target* is allied with a
                # focal entity — pure off-screen action between
                # non-participants stays off the ledger to keep the
                # gauge from flooding on subplots.
                triggered = False
                for fid in eid_set:
                    if any(
                        self._affinity_to(affinity_idx, a, fid)
                        <= self._SUSPENSE_HOSTILE_AFFINITY
                        for a in actor_set
                    ) and any(
                        self._affinity_to(affinity_idx, t, fid)
                        >= self._SUSPENSE_ALLY_AFFINITY
                        for t in target_set
                    ):
                        triggered = True
                        break
                if not triggered:
                    continue

            # ------------- Probability proxy -------------
            prob = 0.5
            if causal_g.has_node(evt_id):
                in_edges = list(causal_g.in_edges(evt_id, data=True))
                out_edges = list(causal_g.out_edges(evt_id, data=True))
                if in_edges:
                    prob = max(d.get("weight", 0.5) for _, _, d in in_edges)
                elif out_edges:
                    prob = max(d.get("weight", 0.5) for _, _, d in out_edges)
            prob = max(0.0, min(1.0, prob))

            # ------------- Harm kind & salience (#3) -------------
            kind, salience = self._harm_kind_for_event(evt_id, causal_g)

            # ------------- Anticipatory proximity (#2) -------------
            if fabula_now is not None:
                dt = max(0, evt.fabula_time - fabula_now)
            else:
                dt = 0
            imminence_t = math.exp(-dt / tau_fabula)

            # ------------- Persistence multiplier (#4) -------------
            ancestors_revealed = self._revealed_ancestor_count(
                evt_id, causal_g, revealed,
            )
            persistence_mult = min(
                self._SUSPENSE_PERSISTENCE_CAP,
                1.0 + self._SUSPENSE_PERSISTENCE_ALPHA * ancestors_revealed,
            )

            evt_loc = self._event_location_id(evt)

            # ------------- Per-focal-entity classification (#1, #5) -------------
            # Compute an actor-disposition signal once (the most
            # extreme affinity any actor holds toward each focal
            # entity), then combine with the legacy actor/target rule
            # to bucket the contribution.
            for fid in eid_set:
                # Spatial proximity for this focal entity (#2).
                focal_loc = None
                fent = self.world_state.entities.get(fid)
                if fent is not None:
                    focal_loc = getattr(fent, "location_id", None)
                d_sp = _spatial_dist(evt_loc, focal_loc)
                if d_sp is None:
                    imminence_s = 1.0  # unknown → neutral, no penalty
                else:
                    imminence_s = math.exp(
                        -d_sp / self._SUSPENSE_PROXIMITY_TAU_SPATIAL,
                    )

                weighted_prob = (
                    prob * salience * imminence_t * imminence_s
                    * persistence_mult
                )
                if weighted_prob <= 0.0:
                    continue

                is_actor = fid in actor_set
                is_target = fid in target_set

                # Disposition signal across other actors of this event.
                hostile_actor = any(
                    self._affinity_to(affinity_idx, a, fid)
                    <= self._SUSPENSE_HOSTILE_AFFINITY
                    for a in actor_set if a != fid
                )
                allied_actor = any(
                    self._affinity_to(affinity_idx, a, fid)
                    >= self._SUSPENSE_ALLY_AFFINITY
                    for a in actor_set if a != fid
                )

                # Resolve bucket. Rules in priority order:
                # 1. Focal entity is target with no co-actor role and
                #    an allied actor is intervening → HOPE (rescue, #5).
                # 2. Focal entity is target with no co-actor role → THREAT
                #    (legacy; reinforced when hostile_actor is True).
                # 3. Focal entity is itself an actor with a hostile
                #    co-actor present → THREAT (coerced participation).
                # 4. Focal entity is an actor → HOPE (legacy).
                # 5. Focal entity is neither but disposition-triggered
                #    above (hostile actor on allied target) → THREAT.
                bucket: Optional[str] = None
                if is_target and not is_actor:
                    if allied_actor and not hostile_actor:
                        bucket = "hope"
                    else:
                        bucket = "threat"
                elif is_actor:
                    if hostile_actor:
                        bucket = "threat"
                    else:
                        bucket = "hope"
                else:
                    # Disposition-triggered third-party event (the
                    # widening branch above ensured at least one focal
                    # entity matches these conditions).
                    if any(
                        self._affinity_to(affinity_idx, a, fid)
                        <= self._SUSPENSE_HOSTILE_AFFINITY
                        for a in actor_set
                    ) and any(
                        self._affinity_to(affinity_idx, t, fid)
                        >= self._SUSPENSE_ALLY_AFFINITY
                        for t in target_set
                    ):
                        bucket = "threat"

                if bucket == "threat":
                    threat_weight += weighted_prob
                    threat_by_kind[kind] = (
                        threat_by_kind.get(kind, 0.0) + weighted_prob
                    )
                elif bucket == "hope":
                    hope_weight += weighted_prob
                    hope_by_kind[kind] = (
                        hope_by_kind.get(kind, 0.0) + weighted_prob
                    )

                # EFK belief-martingale ledger (#7 full): record the
                # unrevealed event as a candidate next-period belief
                # perturbation for this (focal, kind) belief variable.
                # The proximity term ``imminence_t * imminence_s`` is
                # the un-normalised prior over which event reveals
                # next; the Beta-posterior weight is the salience-and-
                # persistence-scaled per-event mass (no proximity, so
                # the same revealed-event uses the same weight whether
                # close or far when it actually fires).
                if mode == "efk" and bucket in ("threat", "hope"):
                    proximity = imminence_t * imminence_s
                    if proximity > 0.0:
                        belief_weight = prob * salience * persistence_mult
                        unrevealed_by_focal_kind.setdefault(
                            (fid, kind), [],
                        ).append((bucket, belief_weight, proximity))

        # ------------- Revealed-event prior pass (EFK only) -------------
        # Walk revealed events to build the (A_threat, B_hope) Beta-
        # posterior evidence per (focal, kind). Revealed events have
        # already happened, so we do *not* apply proximity (no
        # anticipation) or persistence (no foreshadowing arc). The
        # raw ``prob × salience`` is the audience's weight on that
        # past evidence.
        if mode == "efk":
            for evt_id in revealed:
                evt = events_by_id.get(evt_id)
                if evt is None:
                    continue
                prob_r, kind_r, salience_r = self._event_base_weight(
                    evt_id, causal_g,
                )
                w_r = prob_r * salience_r
                if w_r <= 0.0:
                    continue
                for fid in eid_set:
                    bucket_r = self._bucket_event_for_focal(
                        evt, fid, affinity_idx,
                    )
                    if bucket_r is None:
                        continue
                    revealed_by_focal_kind.setdefault(
                        (fid, kind_r), [0.0, 0.0],
                    )[0 if bucket_r == "threat" else 1] += w_r

        # The despair/safety degenerate cases (no hope-side or no
        # threat-side weight on the *unrevealed* set) only apply to
        # the classic balance × stakes aggregator, where they encode
        # the structural-affect prediction that one-sided futures
        # carry no suspense. EFK runs first because under the
        # belief-martingale formulation, accumulated *revealed*
        # evidence on one side combined with an upcoming reveal on
        # the other still produces non-trivial expected variance,
        # and the early returns would silence that signal.

        # ------------- EFK belief-martingale aggregator (#7 full) -------------
        # Per (focal, kind), treat the threat-vs-hope outcome as a
        # Bernoulli random variable. The audience's belief at the
        # current anchor is a Beta-posterior mean,
        #
        #     μ_t = (1 + A) / (2 + A + B)
        #
        # where A and B are the accumulated revealed threat/hope
        # weight on this (focal, kind). For each unrevealed event e
        # that could reveal *next*, with revelation prior π_e
        # (proportional to its proximity), and bucket b_e, the
        # would-be next-period belief is
        #
        #     μ_{t+1} | (e fires as threat) = (1 + A + w_e) / (2 + A + B + w_e),
        #     μ_{t+1} | (e fires as hope)   = (1 + A)       / (2 + A + B + w_e).
        #
        # Suspense for the (focal, kind) belief martingale is
        # E_t[(μ_{t+1} - μ_t)²] = Σ_e π_e (μ_{t+1}|e − μ_t)²
        # — the Ely-Frankel-Kamenica 2015 expected squared belief
        # change. We then aggregate across (focal, kind) by salience
        # × stakes, and rescale by ×4 since a single Bernoulli step
        # variance is bounded by 0.25.
        if mode == "efk":
            num = 0.0
            denom = 0.0
            per_kind_var: Dict[str, float] = {}
            for (fid, kind), unrev in unrevealed_by_focal_kind.items():
                if not unrev:
                    continue
                A, B = revealed_by_focal_kind.get((fid, kind), [0.0, 0.0])
                # Bilateral-mass guard (Brewer & Lichtenstein
                # structural-affect floor): require the *upcoming*
                # reveal set to contain both threat and hope
                # candidates on this (focal, kind). A purely one-
                # sided forward reveal set is despair (only threats
                # coming) or safety (only hopes coming), even
                # though strict EFK would still admit positive
                # variance from the magnitude-uncertainty alone.
                # This matches the test contract and the OCC
                # prospect-based-emotion taxonomy: suspense requires
                # outcome ambiguity, not merely magnitude ambiguity.
                unrev_threat = sum(w for b, w, _ in unrev if b == "threat")
                unrev_hope = sum(w for b, w, _ in unrev if b == "hope")
                if unrev_threat <= 0.0 or unrev_hope <= 0.0:
                    continue
                denom_mu = 2.0 + A + B
                if denom_mu <= 0.0:
                    continue
                mu_t = (1.0 + A) / denom_mu
                total_prox = sum(p for _, _, p in unrev)
                if total_prox <= 0.0:
                    continue
                var_fk = 0.0
                for bucket_u, w_u, prox_u in unrev:
                    pi_e = prox_u / total_prox
                    new_denom = denom_mu + w_u
                    if bucket_u == "threat":
                        mu_next = (1.0 + A + w_u) / new_denom
                    else:
                        mu_next = (1.0 + A) / new_denom
                    var_fk += pi_e * (mu_next - mu_t) ** 2
                # Per-(focal,kind) max-achievable variance at this
                # information state: collapse all unrevealed mass
                # into a single composite event landing on whichever
                # bucket gives the larger squared belief shift. This
                # is the EFK "maximum suspense" reference at the
                # current prior, and divides out the prior-strength
                # scale so the gauge stays in [0, 1] even as A+B
                # accumulate over the syuzhet axis. Without it,
                # var_fk shrinks like 1/(A+B)² as evidence builds
                # up, even when the remaining narrative is
                # genuinely suspenseful.
                w_total = sum(w for _, w, _ in unrev)
                if w_total <= 0.0:
                    continue
                composite_denom = denom_mu + w_total
                mu_if_threat = (1.0 + A + w_total) / composite_denom
                mu_if_hope = (1.0 + A) / composite_denom
                var_max = max(
                    (mu_if_threat - mu_t) ** 2,
                    (mu_if_hope - mu_t) ** 2,
                )
                if var_max <= 0.0:
                    continue
                # Normalised per-(focal,kind) suspense in [0, 1]:
                # the ratio of realised expected squared belief
                # change to the maximum a single composite reveal
                # of the same total mass would achieve. Peaks when
                # all upcoming reveals push belief in the same
                # direction (a coherent forward thrust toward one
                # outcome); decays when threat and hope upcoming
                # reveals cancel each other out. The per-event
                # alternative (sum of best-case per-event shifts)
                # was rejected because under our disposition-based
                # bucketing each event already lands on its own
                # extremising bucket, which would peg the gauge at
                # 1.0 for almost every fixture.
                var_norm = min(1.0, var_fk / var_max)
                # Stakes here uses the *unrevealed* mass only —
                # the mass actually still in play. Using total
                # mass (A + B + unrev) would keep stakes near 1.0
                # at the end of the story when only a single event
                # remains, even though that event is the *only*
                # tension left. With unrev mass alone, late-story
                # single-event leftovers correctly attenuate.
                T_fk = sum(w for _, w, _ in unrev)
                K_k = self._SUSPENSE_STAKES_K_BY_KIND.get(
                    kind, self._SUSPENSE_STAKES_K,
                )
                stakes_k = T_fk / (T_fk + K_k)
                sigma_k = self._HARM_KIND_SALIENCE.get(
                    kind, self._DEFAULT_HARM_SALIENCE,
                )
                weight_fk = sigma_k * stakes_k
                num += weight_fk * var_norm
                denom += weight_fk
                # Diagnostic per-kind aggregate (max over focal).
                per_kind_var[kind] = max(
                    per_kind_var.get(kind, 0.0), var_norm,
                )
            if denom <= 0.0:
                # No (focal, kind) had both a non-trivial belief and
                # a non-empty unrevealed candidate set — fall through
                # to the classic safety/despair-and-aggregate path.
                logger.debug(
                    "[DirectiveAssembly·Suspense·EFK] no belief variance "
                    "available — falling back to classic aggregator",
                )
            else:
                # var_norm is already in [0, 1] (per-(focal,kind)
                # max-normalised); the salience-stakes weighted
                # average preserves that bound.
                score_efk = max(0.0, min(1.0, num / denom))
                dom_kind = max(per_kind_var.items(), key=lambda kv: kv[1])[0] \
                    if per_kind_var else None
                logger.debug(
                    "[DirectiveAssembly·Suspense·EFK·full] suspense=%.3f "
                    "dominant_kind=%s per_kind_var=%s "
                    "threat_by_kind=%s hope_by_kind=%s",
                    score_efk, dom_kind,
                    {k: round(v, 4) for k, v in per_kind_var.items()},
                    {k: round(v, 3) for k, v in threat_by_kind.items()},
                    {k: round(v, 3) for k, v in hope_by_kind.items()},
                )
                return round(score_efk, 4)

        # Despair/safety guards: classic-only. (For EFK we already
        # returned above when the belief-martingale found
        # non-trivial expected variance; if it didn't, we fall
        # through here intentionally so the classic combiner can
        # still emit zero in the structurally-degenerate cases.)
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

        # ------------- Per-kind balance × stakes, weighted-max combine (#3, #6) -------------
        per_kind_scores: Dict[str, float] = {}
        for kind in set(threat_by_kind) | set(hope_by_kind):
            t_k = threat_by_kind.get(kind, 0.0)
            h_k = hope_by_kind.get(kind, 0.0)
            tot_k = t_k + h_k
            if tot_k <= 0.0 or t_k <= 0.0 or h_k <= 0.0:
                # Need both sides for genuine uncertainty.
                continue
            balance_k = 1.0 - abs(t_k - h_k) / tot_k
            K_k = self._SUSPENSE_STAKES_K_BY_KIND.get(
                kind, self._SUSPENSE_STAKES_K,
            )
            stakes_k = tot_k / (tot_k + K_k)
            w = self._HARM_KIND_SALIENCE.get(
                kind, self._DEFAULT_HARM_SALIENCE,
            )
            # Per-kind suspense, weighted by salience so the *max*
            # combine prefers narratively-major kinds over noise.
            per_kind_scores[kind] = w * balance_k * stakes_k

        if per_kind_scores:
            dominant_kind = max(per_kind_scores.items(), key=lambda kv: kv[1])[0]
            score = per_kind_scores[dominant_kind]
        else:
            # Fallback to legacy aggregate when no kind has both
            # threat and hope (e.g. each kind is one-sided but the
            # aggregate is mixed across kinds — rare but possible).
            total = threat_weight + hope_weight
            balance = 1.0 - abs(threat_weight - hope_weight) / total
            stakes = total / (total + self._SUSPENSE_STAKES_K)
            score = balance * stakes
            dominant_kind = None

        score = max(0.0, min(1.0, score))

        dom_threat = max(threat_by_kind.items(), key=lambda kv: kv[1])[0] \
            if threat_by_kind else None
        dom_hope = max(hope_by_kind.items(), key=lambda kv: kv[1])[0] \
            if hope_by_kind else None
        logger.debug(
            "[DirectiveAssembly·Suspense] threat_w=%.3f hope_w=%.3f "
            "suspense=%.3f dominant_kind=%s "
            "dominant_threat=%s dominant_hope=%s "
            "per_kind_scores=%s "
            "threat_by_kind=%s hope_by_kind=%s",
            threat_weight, hope_weight, score, dominant_kind,
            dom_threat, dom_hope,
            {k: round(v, 3) for k, v in per_kind_scores.items()},
            {k: round(v, 3) for k, v in threat_by_kind.items()},
            {k: round(v, 3) for k, v in hope_by_kind.items()},
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Surprise  (Prediction Error — KL Divergence)
    # ------------------------------------------------------------------
    # Batch A.1 — anachrony surprise component (Bae & Young 2008
    # plan-based narrative-surprise; Tobin 2018 *Anachronisms in
    # Narrative*; Bissell, Paulin & Piper 2025 multi-component
    # narrative-surprise framework). Trait-shift KL alone misses the
    # surprise generated by *temporal reordering* — flashbacks that
    # reframe earlier events, openers that drop the reader in medias
    # res. We compute a per-event anachrony score
    # ``|fabula_rank - syuzhet_rank| / N``, average over the relevant
    # event set, and combine with trait-KL via a convex weighting.
    # Default split (0.7 trait / 0.3 anachrony) preserves the
    # existing test contract dominance of trait shifts while letting
    # anachrony move the gauge where it should (Reservoir Dogs, Gone
    # Girl, Tinker Tailor) without overwhelming worlds with linear
    # tellings.
    _SURPRISE_TRAIT_KL_WEIGHT: float = 0.7
    _SURPRISE_ANACHRONY_WEIGHT: float = 0.3

    # Batch C.4 — per-trait narrative salience (Reagan et al. 2016
    # corpus emotional-arc analysis; Kim, Padó & Klinger 2017 genre-
    # conditioned arcs). Some traits carry more narrative-arc
    # signal — courage / love / loyalty / ambition / despair / guilt
    # are the dimensions on which protagonists *change* and which
    # readers track; literacy / fitness / wealth track too but
    # rarely dominate the arc. We mirror the harm-kind salience
    # table's structure: trait names checked case-insensitively as
    # substrings (so ``moral_courage``, ``physical_courage``,
    # ``courage`` all match the ``courage`` salience). Unmatched
    # traits get the median weight, so worlds without any salience-
    # tagged traits degrade gracefully to the prior unweighted
    # behaviour. Calibrated against the example_worlds corpus to
    # let canonical arc traits (Macbeth's ambition, Lady Macbeth's
    # guilt, Heathcliff's vengeance) drive the surprise gauge while
    # peripheral traits register a residual contribution.
    _TRAIT_NARRATIVE_SALIENCE: Dict[str, float] = {
        # Core arc dimensions — the things protagonists are *about*.
        "ambition": 1.00,
        "guilt": 0.95,
        "vengeance": 0.95,
        "despair": 0.95,
        "love": 0.90,
        "loyalty": 0.85,
        "courage": 0.85,
        "betrayal": 0.85,
        "honesty": 0.80,
        "morality": 0.80,
        "rage": 0.75,
        "fear": 0.75,
        "trust": 0.70,
        # Mid-tier — character but not usually the spine.
        "patience": 0.60,
        "wisdom": 0.60,
        "pride": 0.60,
        "compassion": 0.55,
        # Peripheral — informational backdrop.
        "literacy": 0.30,
        "fitness": 0.30,
        "wealth": 0.30,
        "health": 0.40,
    }
    _DEFAULT_TRAIT_SALIENCE: float = 0.55

    # Batch C.5 — source-edge contribution. Edges where the focal is
    # the *source* (Macbeth murders Duncan) update the focal's
    # traits too (ambition reinforced by acting on it), but at a
    # reduced weight: the canonical Bayesian update for "X did Y to
    # Z" speaks more strongly about Z's traits than X's. Multiplier
    # of 0.4 keeps the asymmetry while no longer ignoring the
    # source-side signal entirely (the previous target-only design
    # missed a substantial chunk of the per-character belief
    # update, especially in agent-centric fixtures like Macbeth and
    # Reservoir Dogs).
    _SURPRISE_SOURCE_EDGE_WEIGHT: float = 0.4

    def compute_surprise_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
        *,
        local: bool = False,
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

        ``local=True`` switches to **Bayesian Surprise** in the sense
        of Itti & Baldi (2009): the per-step belief update magnitude,
        ``mean_traits[ D_KL( q_s || q_{s-1} ) ]`` — the KL distance
        between the reader's prior immediately *after* and immediately
        *before* the current syuzhet anchor's revelations (posterior
        over prior, the canonical Itti-Baldi direction). This is
        formally identical to the Storck/Hochreiter/Schmidhuber (1995)
        RDIA formulation acknowledged on the iLab Bayesian-Surprise
        page. It produces the spike-and-decay trajectory the theory
        predicts (each revelation triggers a peak proportional to how
        much it shifts the reader's expectation; quiet stretches sit
        at zero). The timeseries view consumes this mode so the chart
        depicts moments-of-revelation rather than the integrated
        cumulative gap (which is what the cumulative form below
        measures, and which the existing test contract / directive
        optimiser expect as the default).
        """
        if syuzhet_anchor is None:
            return 0.0  # reader knows everything → no surprise

        EPS = 0.01
        _STRENGTH_W = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}

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

        def _prior_for(eid: str, trait_name: str, actual_val: float,
                       anchor: int) -> float:
            """Reader's prior for ``(eid, trait)`` at syuzhet ``anchor``.

            Batch B.3 — proper Beta-Bernoulli update replacing the
            legacy geometric pull ``p += w·(actual - p)``. The base
            corpus-marginal ``m`` seeds a Beta(s·m, s·(1-m)) prior
            (pseudo-count strength ``s = 2`` — weak enough to remain
            responsive to evidence, strong enough to anchor the
            posterior away from the EPS-clipped extremes when the
            evidence stream is empty). Each revealed causal edge
            targeting the focal entity contributes Bernoulli evidence
            with weight ``w = STRENGTH_W[evidence_strength]`` and
            outcome ``actual_val``:
                α += w · actual,    β += w · (1 - actual)
            and the posterior mean ``α / (α + β)`` is returned. This
            is the exact same Bayesian update pattern we now use for
            EFK suspense (Beta-posterior on threat/hope ledgers), so
            the surprise and suspense scorers share a coherent
            Bayesian core rather than relying on separate ad-hoc
            update rules. The geometric pull was a Storck/Hochreiter/
            Schmidhuber 1995 RDIA proxy — fine for monotone
            convergence but undefined posterior variance, which the
            optional Weber-Fechner / per-trait salience extensions in
            §3.3 require to behave well.
            """
            revealed_ids = self._revealed_event_ids(anchor)
            base = _trait_marginal(trait_name, eid)
            s = self._SURPRISE_PRIOR_PSEUDOCOUNT  # weak Beta pseudo-count anchor
            alpha = s * base
            beta = s * (1.0 - base)
            for ce in self.world_state.causal_topology:
                if ce.source_id not in revealed_ids:
                    continue
                w = _STRENGTH_W.get(ce.evidence_strength, 0.5)
                if ce.target_id == eid:
                    # Target-side: full Bernoulli update.
                    alpha += w * actual_val
                    beta += w * (1.0 - actual_val)
                elif ce.source_id == eid:
                    # Batch C.5 — source-side: reduced-weight
                    # update. "X did Y to Z" speaks more strongly
                    # about Z's traits than X's, but X's act
                    # itself is evidence about X's traits too
                    # (ambition reinforced by acting on it).
                    sw = w * self._SURPRISE_SOURCE_EDGE_WEIGHT
                    alpha += sw * actual_val
                    beta += sw * (1.0 - actual_val)
            denom = alpha + beta
            if denom <= 0.0:
                return 0.5
            p = alpha / denom
            return max(EPS, min(1 - EPS, p))

        def _binary_kl(p: float, q: float) -> float:
            p = max(EPS, min(1 - EPS, p))
            q = max(EPS, min(1 - EPS, q))
            return max(
                0.0,
                p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)),
            )

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

                if local:
                    # Bayesian Surprise (Itti & Baldi 2009): magnitude
                    # of the belief update at *this* step. Compare the
                    # reader's prior immediately before vs immediately
                    # after the current syuzhet anchor. Quiet stretches
                    # → ~0; revelations → spikes proportional to how
                    # much the new information shifts the prior.
                    q_prev = _prior_for(eid, trait_name, actual_val,
                                        syuzhet_anchor - 1)
                    q_now = _prior_for(eid, trait_name, actual_val,
                                       syuzhet_anchor)
                    kl = _binary_kl(q_now, q_prev)
                else:
                    # Cumulative form: KL between the reader's accumulated
                    # prior at this anchor and the true posterior. Falls
                    # monotonically as evidence accumulates — which is
                    # what the directive optimiser and existing test
                    # contract expect.
                    p = max(EPS, min(1 - EPS, actual_val))
                    q = _prior_for(eid, trait_name, actual_val, syuzhet_anchor)
                    kl = _binary_kl(p, q)

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
                # information at the high end. The decay constant
                # ``τ = 1`` matches the Weber-Fechner JND literature
                # for binary-distribution discrimination (Lu &
                # Dosher 2013, *Visual Psychophysics*), where the
                # subjective just-noticeable belief shift sits in
                # the ``0.5–1.0`` nat band — i.e. each ``1.0`` nat
                # of KL evidence delivers ``≈ 1 - 1/e ≈ 63%`` of
                # the perceptual range, which is exactly where this
                # saturation curve places it.
                #
                # Batch C.4 — per-trait narrative salience: weight
                # the per-trait contribution by Reagan-et-al. 2016
                # arc-relevance hierarchy. Substring match against
                # the salience table catches ``moral_courage``,
                # ``physical_courage`` and ``courage`` under the
                # same ``courage`` weight; unmatched traits get the
                # default mid-tier salience.
                trait_lc = trait_name.lower()
                trait_sal = self._DEFAULT_TRAIT_SALIENCE
                for key, weight in self._TRAIT_NARRATIVE_SALIENCE.items():
                    if key in trait_lc:
                        trait_sal = weight
                        break
                total_kl += trait_sal * (1.0 - math.exp(-kl))
                trait_count += trait_sal

        if trait_count == 0:
            return 0.0

        trait_kl_score = min(total_kl / trait_count, 1.0)

        # Batch A.1 — anachrony component (Bae & Young 2008; Bissell-
        # Paulin-Piper 2025). Per-event anachrony score
        # ``|fabula_rank - syuzhet_rank| / N`` for events relevant to
        # the focal entities (actor or target), averaged over the
        # event set the surprise mode considers:
        #   * cumulative mode → all revealed events at the anchor
        #     (the integrated anachrony footprint of the unfolding
        #     telling, mirroring the integrated trait-gap form);
        #   * local mode → events newly revealed at this anchor
        #     (the per-step anachrony spike, mirroring the per-step
        #     Itti-Baldi belief-update spike).
        # Events sharing a fabula-time / syuzhet-index are ranked
        # stably by ``(value, id)`` so the two ranks are
        # well-defined on every fixture without ties biasing the
        # score. Worlds with a single event or a perfectly linear
        # telling correctly contribute zero anachrony.
        anachrony_score = 0.0
        all_events = list(self.world_state.events)
        if all_events and len(all_events) >= 2:
            f_sorted = sorted(all_events, key=lambda e: (e.fabula_time, e.id))
            s_sorted = sorted(all_events, key=lambda e: (e.syuzhet_index, e.id))
            f_rank = {e.id: i for i, e in enumerate(f_sorted)}
            s_rank = {e.id: i for i, e in enumerate(s_sorted)}
            N = float(len(all_events) - 1)
            eid_set = set(entity_ids)
            revealed_ids = self._revealed_event_ids(syuzhet_anchor)
            if local:
                consider = {
                    e.id for e in all_events
                    if e.syuzhet_index == syuzhet_anchor
                }
            else:
                consider = revealed_ids
            relevant = [
                e for e in all_events
                if e.id in consider and (
                    set(e.actor_ids) & eid_set or set(e.target_ids) & eid_set
                )
            ]
            if relevant:
                anachrony_score = sum(
                    abs(f_rank[e.id] - s_rank[e.id]) / N
                    for e in relevant
                ) / len(relevant)
                anachrony_score = min(1.0, anachrony_score)

        # Convex weighted combine of trait-shift KL (the existing
        # Itti-Baldi / Storck quantity) and anachrony (Bae-Young
        # plan-based / Bissell-Paulin-Piper 2025 narrative-level
        # surprise). Weights sum to 1 so the result stays in [0, 1].
        score = (
            self._SURPRISE_TRAIT_KL_WEIGHT * trait_kl_score
            + self._SURPRISE_ANACHRONY_WEIGHT * anachrony_score
        )

        logger.debug(
            "[DirectiveAssembly·Surprise%s] trait_kl=%.4f anachrony=%.4f "
            "→ %.4f over %d traits",
            "·local" if local else "",
            trait_kl_score, anachrony_score, score, trait_count,
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

        # mystery / dramatic_irony / surprise collapse without a POV
        # anchor — the entire effect depends on locking the reader to
        # one consciousness and withholding what other minds know.
        # Without ``pov_lock`` set the auditor's POV check at
        # ``auditor.py::_format_constraints_for_audit`` is silently
        # skipped, so flag it loudly here so the caller sees the gap.
        if pov_entity is None and effect in ("mystery", "dramatic_irony", "surprise"):
            logger.warning(
                "[DirectiveAssembly] %s directive supplied no "
                "target_entity_ids; pov_lock will be None and the "
                "epistemic-control audit will lose its POV anchor. "
                "Pass at least one focus entity on the DirectiveQuery.",
                effect,
            )

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
        # Also include any locations / objects / co-present entities in
        # scope via the ego payload. The canonical ego shape uses
        # ``current_locations`` / ``present_objects`` / ``present_entities``
        # (see `Step8Engine._build_ego_payload`); the legacy
        # ``focus_locations`` / ``focus_objects`` keys are accepted for
        # back-compat with older callers. Without the canonical keys
        # being checked here, facts whose ``related_node_ids`` named a
        # scene location or object were silently dropped from the
        # external_research block of the prompt and audit.
        for k in (
            "focus_locations", "focus_objects",
            "current_locations", "present_objects", "present_entities",
        ):
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
