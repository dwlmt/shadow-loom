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
from typing import Any, Dict, List, Optional, Literal, Set, Tuple

import networkx as nx
from pydantic import BaseModel, Field

from shadow_loom.models import WorldStateV1, NarrativeStyle, reconstruct_entity_at, reconstruct_object_at, reconstruct_world_trait_at, event_location_at
from shadow_loom.causal_closure import (
    chain_reaction_parents_from_world_state,
    edges_within_closure,
    expand_chain_reaction_closure,
)
from shadow_loom.query_models import DirectiveQuery, DoTarget
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


class WorldTraitShift(BaseModel):
    """Recent shift in a WORLD_ trait the renderer should depict.

    Surfaces the trait's *latest* state-timeline movement (delta in
    magnitude.value at or before the brief's syuzhet anchor) so prose
    can foreground regime changes, mood reversals, prophecy
    resolutions and other ambient shifts the engine has folded onto
    ``GlobalTrait.state_timeline`` (per-chunk Consequences updates,
    Step-5 timeline reconciliation, Pearl-Rung-2 truth clamps with a
    linked ``proposition_id``). Empty when no recent movement exists.
    """
    trait_id: str
    trait_name: str
    previous_value: float
    current_value: float
    delta: float
    inertia: float
    affected_domains: List[str] = Field(default_factory=list)
    fabula_time: int
    triggered_by: Optional[str] = None
    proposition_id: Optional[str] = None
    description: Optional[str] = None


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
            "narrative_tension, observation, intervention, "
            "counterfactual, manual_edit, fallback, default. "
            "(Interrogation queries return graph analysis, not prose, "
            "and never reach this directive.)"
        ),
    )
    pov_lock: Optional[str] = Field(
        default=None,
        description=(
            "Primary entity ID to lock the narrative perspective to. "
            "When ``pov_policy == 'single'`` (default) this is the "
            "only licensed POV and head-hopping is a violation. When "
            "``pov_policy == 'rotating'`` or ``'ensemble'`` this is "
            "the *primary* / opening POV; ``additional_pov_locks`` "
            "lists the other licensed perspectives. "
            "\n\n"
            "**Role across the pipeline (round-7 audit 2026-05-26).** "
            "``pov_lock`` is the *single source of truth* for POV "
            "across rendering, refinement, and audit. Its enforcement "
            "is layered:\n"
            "\n"
            "  1. **Render-time contract** (``assemble_rendering_prompt`` "
            "     in ``shadow_loom/generation.py``). The brief's "
            "     ``pov_lock`` is emitted as a HARD constraint in the "
            "     rendering prompt and tells the generation LLM whose "
            "     consciousness it may narrate from.\n"
            "  2. **Scene-output mirror** (``GeneratedScene.pov_entity`` "
            "     in ``shadow_loom/generation.py``). The renderer must "
            "     mirror the lock back in its structured output. A\n"
            "     divergence here (``pov_lock`` set, ``pov_entity`` "
            "     ``None``) is the most common rewrite-pressure failure; "
            "     the feedback loop coerces it back AND records a "
            "     synthetic ``reasoning_failure`` violation so the next "
            "     iteration sees the breach explicitly. See "
            "     ``shadow_loom/auditor.py::_check_pov_lock_metadata``.\n"
            "  3. **Deterministic prose check** "
            "     (``shadow_loom/auditor.py::"
            "deterministic_prose_findings``). On every iteration of the "
            "     feedback loop, a regex pass counts sentences whose "
            "     grammatical subject is a *non-POV* proper noun "
            "     attached to a cognitive / perceptual verb. Above "
            "     ``AuditorConfig.pov_breach_threshold`` (default 3) "
            "     the loop synthesises a critical ``reasoning_failure`` "
            "     violation BEFORE the LLM auditor runs. This catches "
            "     the omniscient-narration drift that LLM auditors "
            "     miss under rewrite pressure.\n"
            "  4. **LLM auditor**. The auditor prompt is told the POV "
            "     lock and asked to flag head-hops, omniscient asides, "
            "     and non-POV interiority. It is the slowest and "
            "     most expressive check, but also the most "
            "     unreliable; layers 2 and 3 exist because the LLM "
            "     auditor empirically misses ~one-third of POV "
            "     breaches under iteration pressure.\n"
            "  5. **Refinement injection**. When the audit fails, the "
            "     refinement prompt re-emits the brief verbatim "
            "     (so ``pov_lock`` is restated) AND includes a "
            "     dedicated Rule 7 in ``shadow_loom/prompts/"
            "refinement.md`` forbidding ``pov_entity`` drops.\n"
            "\n"
            "Setting ``pov_lock=None`` deliberately disables ALL "
            "layers above and licenses omniscient narration. Setting "
            "``pov_lock`` and overriding "
            "``AuditorConfig.enable_deterministic_prose_checks=False`` "
            "keeps layers 1, 2, 4, 5 and skips layer 3 — useful for "
            "ensemble briefs where the regex would over-fire."
        ),
    )
    additional_pov_locks: List[str] = Field(
        default_factory=list,
        description=(
            "Additional licensed POV entity IDs beyond ``pov_lock``. "
            "Empty under ``pov_policy='single'``. Under "
            "``'rotating'`` each entity owns an internal-perception "
            "beat (no head-hopping within a beat). Under "
            "``'ensemble'`` the omniscient narrator may license "
            "interiority across the roster simultaneously."
        ),
    )
    pov_policy: Literal["single", "rotating", "ensemble"] = Field(
        default="single",
        description=(
            "How POV is licensed across the brief's target entities. "
            "``single`` = strict pov_lock, no head-hopping (default; "
            "back-compat). ``rotating`` = each entity in "
            "``[pov_lock] + additional_pov_locks`` gets its own beat. "
            "``ensemble`` = omniscient-constrained narrator licensed "
            "to render interiority across the roster."
        ),
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
    # --- Phase 7: typed Pearl-rung surgery metadata ---
    do_target: Optional[DoTarget] = Field(
        default=None,
        description=(
            "The typed Rung-3 surgery that produced this branch. "
            "``None`` for the legacy event-only path; populated when "
            "the brief was assembled from a typed counterfactual "
            "query (DoEvent / DoProposition / DoBelief / DoConcern / "
            "DoTrait). Lets the renderer pick rung-aware phrasing."
        ),
    )
    do_targets: List[DoTarget] = Field(
        default_factory=list,
        description=(
            "R3-2 (2026-05-29): the FULL list of typed Rung-3 surgeries "
            "in this brief. ``do_target`` above is the *primary* surgery "
            "(usually the first); ``do_targets`` carries every clamp so "
            "joint counterfactuals (\"if Macbeth had not killed Duncan "
            "AND Lady Macbeth had not goaded him\") can be rendered "
            "without silently dropping all but the first clamp. The "
            "renderer / auditor should consult this list when "
            "``len(do_targets) > 1`` and treat each entry as a "
            "simultaneous Pearl-3 surgery applied at the divergence "
            "point. Empty list means the brief came from the legacy "
            "untyped path."
        ),
    )
    do_target_gloss: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of what ``do_target`` refers "
            "to in the world (e.g. the EventNode.description, the "
            "Proposition.description, the Concern.description). "
            "Populated by the brief builder by resolving the "
            "``do_target`` id against ``WorldStateV1`` so the renderer "
            "sees the actual referent rather than only its opaque id. "
            "Without this gloss the renderer falls back to whatever "
            "adjacent context the SCENE CONTEXT block happens to "
            "carry, which can yield thematically-plausible but "
            "factually-wrong confabulations of the surgery target."
        ),
    )
    do_target_context: Optional[str] = Field(
        default=None,
        description=(
            "Pre-rendered multi-line block of the factual causal "
            "neighbourhood around the ``do_target`` (immediate causal "
            "predecessors, successors, affordance preconditions, with "
            "each event's description, fabula/syuzhet anchor, "
            "actors/targets, location). Only populated for event-kind "
            "surgeries on an outcome event \u2014 the renderer / auditor "
            "need this to write the counterfactual as a DIFFERENT "
            "outcome of the same attempt rather than erase the attempt "
            "entirely."
        ),
    )
    affected_propositions: List[str] = Field(
        default_factory=list,
        description=(
            "PROP_ ids whose truth flipped between the factual world "
            "and the counterfactual sandbox."
        ),
    )
    affected_beliefs: List[str] = Field(
        default_factory=list,
        description=(
            "Holder→target keys (``ENT_X→ENT_Y``) whose confidence "
            "shifted between factual and counterfactual."
        ),
    )
    affected_concerns: List[str] = Field(
        default_factory=list,
        description=(
            "CCN_ ids whose desire-satisfaction polarity flipped "
            "between factual and counterfactual."
        ),
    )
    # --- Parallel human-readable description lists ---
    # Populated at brief-build time by resolving the raw id lists
    # above against WorldStateV1. Without these the renderer sees
    # only opaque PROP_/CCN_/ENT_ ids and may confabulate.
    affected_proposition_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Proposition.description strings parallel to "
            "``affected_propositions`` (same order). Populated at "
            "brief-build time so the renderer sees the natural-language "
            "referent of each PROP_ id."
        ),
    )
    affected_belief_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable holder\u2192target labels parallel to "
            "``affected_beliefs`` (same order). "
            "E.g. \"Ken\u2019s beliefs about Mrs Coady\"."
        ),
    )
    affected_concern_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable concern labels parallel to "
            "``affected_concerns`` (same order). "
            "E.g. \"Ken\u2019s fear that Mrs Coady is alive\"."
        ),
    )
    # --- Object / world-trait / edge surgery side-effects (Round-6) ---
    # The Rung-3 surgery may also relocate props, clamp ambient world
    # traits, or sever/add topology edges. These id lists are produced
    # by ``narrative_physics._typed_target_payload`` from the engine's
    # ObjectMutation / WorldTraitMutation / EdgeMutation logs. The
    # auditor iterates them alongside the proposition/belief/concern
    # lists so prose that silently drops a prop relocation or topology
    # rewrite is flagged as a miracle step.
    affected_objects: List[str] = Field(
        default_factory=list,
        description=(
            "OBJ_ ids whose location / owner / properties were "
            "mutated between factual and counterfactual sandbox."
        ),
    )
    affected_world_traits: List[str] = Field(
        default_factory=list,
        description=(
            "WORLD_ ids whose ambient-force value changed between "
            "factual and counterfactual sandbox."
        ),
    )
    affected_edges: List[str] = Field(
        default_factory=list,
        description=(
            "Topology edges (``edge_type:action:source\u2192target``) "
            "added or severed between factual and counterfactual."
        ),
    )
    affected_entity_deletes: List[str] = Field(
        default_factory=list,
        description=(
            "ENT_ ids excised by a DoEntityDelete existence-"
            "counterfactual on this branch."
        ),
    )
    affected_object_deletes: List[str] = Field(
        default_factory=list,
        description=(
            "OBJ_ ids excised by a DoObjectDelete existence-"
            "counterfactual on this branch."
        ),
    )
    # --- Downstream consequence cascades (Phase-10: rich brief) ---
    # The Rung-3 surgery propagates through the AMWN. The engine
    # records what mutated (TraitMutation / SocialMutation /
    # Proposition/Belief/Concern mutations) and what was blocked by
    # inertia. Surfacing these as human-readable bullets gives the
    # renderer concrete consequences to ground prose in, instead of
    # bare ID lists.
    downstream_trait_changes: List[str] = Field(
        default_factory=list,
        description=(
            "Per-trait cascade lines (``ENT_X.fear: 0.30\u21920.65 "
            "(impact=0.40, inertia=0.20)``) from the engine's "
            "TraitMutation log under this counterfactual surgery."
        ),
    )
    downstream_relationship_changes: List[str] = Field(
        default_factory=list,
        description=(
            "Per-axis social-tie cascade lines from SocialMutation "
            "(``ENT_X\u2192ENT_Y trust: +0.50\u2192+0.20``)."
        ),
    )
    proposition_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "PropositionMutation detail lines including truth flip "
            "direction and number of cascaded belief updates."
        ),
    )
    belief_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "BeliefMutation detail lines (holder, target, old\u2192new "
            "confidence, triggering proposition)."
        ),
    )
    concern_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "ConcernMutation detail lines (holder, concern, field, "
            "old\u2192new value)."
        ),
    )
    object_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "ObjectMutation detail lines (object id, new_location_id, "
            "new_owner_id, properties_set / properties_unset diff). "
            "Surfaces DoNarrativeObject / DoObjectDelete propagation so "
            "the renderer dramatises prop relocations / ownership shifts "
            "rather than dropping them."
        ),
    )
    world_trait_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "WorldTraitMutation detail lines (world_trait_id, old\u2192new "
            "magnitude, affected_domains add/remove). Surfaces "
            "DoWorldTrait clamps so the renderer can ground ambient "
            "force shifts in atmosphere or institutional mood."
        ),
    )
    edge_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "EdgeMutation detail lines (edge_type, action, endpoints). "
            "Covers DoCausalEdge / DoSpatialEdge / DoChannel surgeries "
            "so topology rewrites are legible to the renderer / auditor "
            "rather than landing silently as a sandbox delta."
        ),
    )
    entity_delete_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "EntityDeleteMutation detail lines (entity_id, fabula_time, "
            "social/causal edges removed, beliefs removed, events "
            "scrubbed). Surfaces DoEntityDelete existence-counterfactual "
            "excisions so the renderer treats the character as never "
            "having been present and the auditor can flag any residual "
            "reference."
        ),
    )
    object_delete_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "ObjectDeleteMutation detail lines (object_id, fabula_time, "
            "cascade counts). Surfaces DoObjectDelete existence-"
            "counterfactual excisions so the renderer treats the prop "
            "as never having been present."
        ),
    )
    blocked_propagations_detail: List[str] = Field(
        default_factory=list,
        description=(
            "BlockedPropagation lines explaining why a cascade did "
            "NOT take effect (inertia, spatial affordance, cycle, "
            "noisy-OR absorbed) \u2014 the renderer must visibly "
            "dramatise the resistance rather than skip the node."
        ),
    )
    event_cascade_detail: List[str] = Field(
        default_factory=list,
        description=(
            "EventMutation detail lines from DoEventNode surgeries "
            "(spawn / suppress / time-shift of EventNodes), and from "
            "chain_reaction descendant closure (events marked "
            "``pruned=True`` by ``causal_physics`` Step B.6 after a "
            "do-surgery prevents their parent). The renderer must "
            "narrate each pruned descendant as not having occurred; "
            "the auditor's ``prevented_event_reenacted`` violation "
            "check is grounded in this list."
        ),
    )
    causal_chain: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of EVT_ ids the surgery propagates through "
            "on its way to the visible outcome \u2014 the prose must "
            "render each link as an on-page beat, not skip from "
            "surgery target to terminal consequence."
        ),
    )
    causal_chain_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "EventNode.description strings parallel to "
            "``causal_chain`` (same order). Annotates each EVT_ id in "
            "the chain so the renderer sees the actual event referent."
        ),
    )
    tragedy_form: Optional[Literal["tragic", "comic", "ironic", "neutral"]] = Field(
        default=None,
        description=(
            "Aristotelian / Frye narrative-form classification of the "
            "(actual − counterfactual) concern-satisfaction delta. "
            "``tragic`` = actual worse than counterfactual; ``comic`` "
            "= actual better; ``ironic`` = mixed-sign; ``neutral`` = "
            "no concern-load difference."
        ),
    )


class ThreatProximity(BaseModel):
    """Threat information for fear/suspense rendering.

    The ``threat_probability`` / ``hope_probability`` scalars are now
    computed with the same disposition-aware bucketing, harm-kind
    salience, fabula-time + spatial imminence, and persistence
    multiplier the suspense scorer uses (improvements A1-A4 / B8/B9),
    so the dashboard reading is no longer a regressed view of the
    same evidence the scorer already sees.
    """
    threat_event_id: Optional[str] = None
    threat_description: str = ""
    threat_probability: float = 0.5
    hope_probability: float = 0.5
    threat_kind: Optional[str] = Field(
        default=None,
        description=(
            "Dominant harm-kind of the highest-weight threat event "
            "(existential / physical / betrayal / psychological / "
            "emotional / social / epistemic / informational). ``None`` "
            "when no threat resolves."
        ),
    )
    hope_kind: Optional[str] = Field(
        default=None,
        description=(
            "Dominant kind of the highest-weight hope event. ``None`` "
            "when no hope resolves."
        ),
    )
    spatial_distance: Optional[int] = Field(
        default=None,
        description="Number of spatial hops between threat and target.",
    )
    damage_potential: float = Field(
        default=5.0,
        description="causal_force of the threat edge (0-10).",
    )
    # --- Phase 7: typed Pearl-rung surgery metadata (Rung-2) ---
    do_target: Optional[DoTarget] = Field(
        default=None,
        description=(
            "The typed Rung-2 surgery the threat reading was taken "
            "*under*. ``None`` for the legacy event-only path; "
            "populated when an InterventionQuery carried typed "
            "``do_targets`` (DoEvent / DoProposition / DoBelief / "
            "DoConcern / DoTrait). Lets the renderer phrase the "
            "intervention with the right epistemic / ontic register."
        ),
    )
    do_target_gloss: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of what ``do_target`` refers "
            "to. Mirrors the same field on :class:`InterventionBranch` "
            "so threat-keyed Rung-2 briefs surface the surgery's actual "
            "referent, not just its id."
        ),
    )
    do_target_context: Optional[str] = Field(
        default=None,
        description=(
            "Pre-rendered causal-neighbourhood block for the do_target. "
            "Mirrors the same field on :class:`InterventionBranch`."
        ),
    )
    affected_propositions: List[str] = Field(
        default_factory=list,
        description=(
            "PROP_ ids whose truth flipped between the factual world "
            "and the post-intervention sandbox."
        ),
    )
    affected_beliefs: List[str] = Field(
        default_factory=list,
        description=(
            "Holder→target keys (``ENT_X→ENT_Y``) whose confidence "
            "shifted between factual and post-intervention sandbox."
        ),
    )
    affected_concerns: List[str] = Field(
        default_factory=list,
        description=(
            "CCN_ ids whose desire-satisfaction polarity flipped or "
            "whose salience changed under the intervention."
        ),
    )
    # --- Parallel human-readable description lists ---
    affected_proposition_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Proposition.description strings parallel to "
            "``affected_propositions`` (same order)."
        ),
    )
    affected_belief_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable holder\u2192target labels parallel to "
            "``affected_beliefs`` (same order)."
        ),
    )
    affected_concern_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable concern labels parallel to "
            "``affected_concerns`` (same order)."
        ),
    )
    # --- Object / world-trait / edge surgery side-effects (Round-6) ---
    # See :class:`CounterfactualBranch` for semantics.
    affected_objects: List[str] = Field(default_factory=list)
    affected_world_traits: List[str] = Field(default_factory=list)
    affected_edges: List[str] = Field(default_factory=list)
    affected_entity_deletes: List[str] = Field(default_factory=list)
    affected_object_deletes: List[str] = Field(default_factory=list)
    # --- Downstream consequence cascades (Phase-10: rich brief) ---
    # See :class:`CounterfactualBranch` for field semantics; mirrored
    # here so an affective directive carrying Rung-2 surgery context
    # surfaces the same cascade detail as a plain intervention brief.
    downstream_trait_changes: List[str] = Field(default_factory=list)
    downstream_relationship_changes: List[str] = Field(default_factory=list)
    proposition_cascade_detail: List[str] = Field(default_factory=list)
    belief_cascade_detail: List[str] = Field(default_factory=list)
    concern_cascade_detail: List[str] = Field(default_factory=list)
    object_cascade_detail: List[str] = Field(default_factory=list)
    world_trait_cascade_detail: List[str] = Field(default_factory=list)
    edge_cascade_detail: List[str] = Field(default_factory=list)
    entity_delete_cascade_detail: List[str] = Field(default_factory=list)
    object_delete_cascade_detail: List[str] = Field(default_factory=list)
    blocked_propagations_detail: List[str] = Field(default_factory=list)
    event_cascade_detail: List[str] = Field(default_factory=list)
    causal_chain: List[str] = Field(default_factory=list)
    causal_chain_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "EventNode.description strings parallel to "
            "``causal_chain`` (same order)."
        ),
    )


class InterventionBranch(BaseModel):
    """Generic Rung-2 sandbox payload for intervention briefs.

    ``ThreatProximity`` historically carried Rung-2 surgery metadata
    *coupled* to a threat/hope reading because suspense and fear were
    the only directive effects that surfaced the do-operator. For
    plain intervention queries (no threat reading) we still need to
    surface the typed ``do_target`` and the
    ``affected_propositions / affected_beliefs / affected_concerns``
    sandbox so the renderer and auditor can verify that the prose
    actually grounds every flipped node — without reusing the
    threat-flavoured carrier.

    Mirrors the rung-2 fields on :class:`ThreatProximity` so the
    auditor's existing iteration over rung-2 side-effects can fold
    this in alongside fear/suspense readings.
    """
    do_target: Optional[DoTarget] = Field(
        default=None,
        description=(
            "The typed Rung-2 surgery the brief was assembled under "
            "(DoEvent / DoProposition / DoBelief / DoConcern / "
            "DoTrait / DoWorldTrait). ``None`` when the legacy "
            "event-only path supplied the intervention."
        ),
    )
    do_target_gloss: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of what ``do_target`` refers "
            "to in the world. Mirrors the same field on "
            ":class:`CounterfactualBranch` so the renderer sees the "
            "actual referent of the Rung-2 surgery, not just its id."
        ),
    )
    do_target_context: Optional[str] = Field(
        default=None,
        description=(
            "Pre-rendered causal-neighbourhood block for the do_target. "
            "Mirrors the same field on :class:`CounterfactualBranch`."
        ),
    )
    do_targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "All typed Rung-2 surgeries on the query, as plain dicts. "
            "Used when the query carries multiple do-targets and the "
            "renderer / auditor need to enumerate them rather than "
            "narrate a single primary surgery."
        ),
    )
    affected_propositions: List[str] = Field(
        default_factory=list,
        description=(
            "PROP_ ids whose truth flipped between the factual world "
            "and the post-intervention sandbox."
        ),
    )
    affected_beliefs: List[str] = Field(
        default_factory=list,
        description=(
            "Holder→target keys (``ENT_X→ENT_Y``) whose confidence "
            "shifted between factual and post-intervention sandbox."
        ),
    )
    affected_concerns: List[str] = Field(
        default_factory=list,
        description=(
            "CCN_ ids whose desire-satisfaction polarity flipped or "
            "whose salience changed under the intervention."
        ),
    )
    # --- Parallel human-readable description lists ---
    affected_proposition_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Proposition.description strings parallel to "
            "``affected_propositions`` (same order)."
        ),
    )
    affected_belief_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable holder\u2192target labels parallel to "
            "``affected_beliefs`` (same order)."
        ),
    )
    affected_concern_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable concern labels parallel to "
            "``affected_concerns`` (same order)."
        ),
    )
    # --- Object / world-trait / edge surgery side-effects (Round-6) ---
    # See :class:`CounterfactualBranch` for semantics.
    affected_objects: List[str] = Field(default_factory=list)
    affected_world_traits: List[str] = Field(default_factory=list)
    affected_edges: List[str] = Field(default_factory=list)
    affected_entity_deletes: List[str] = Field(default_factory=list)
    affected_object_deletes: List[str] = Field(default_factory=list)
    downstream_trait_changes: List[str] = Field(default_factory=list)
    downstream_relationship_changes: List[str] = Field(default_factory=list)
    proposition_cascade_detail: List[str] = Field(default_factory=list)
    belief_cascade_detail: List[str] = Field(default_factory=list)
    concern_cascade_detail: List[str] = Field(default_factory=list)
    object_cascade_detail: List[str] = Field(default_factory=list)
    world_trait_cascade_detail: List[str] = Field(default_factory=list)
    edge_cascade_detail: List[str] = Field(default_factory=list)
    entity_delete_cascade_detail: List[str] = Field(default_factory=list)
    object_delete_cascade_detail: List[str] = Field(default_factory=list)
    blocked_propagations_detail: List[str] = Field(default_factory=list)
    event_cascade_detail: List[str] = Field(default_factory=list)
    causal_chain: List[str] = Field(default_factory=list)
    causal_chain_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "EventNode.description strings parallel to "
            "``causal_chain`` (same order)."
        ),
    )
    tragedy_form: Optional[Literal["tragic", "comic", "ironic", "neutral"]] = Field(
        default=None,
        description=(
            "Aristotelian / Frye narrative-form classification of the "
            "(actual − intervened) concern-satisfaction delta."
        ),
    )


class SurpriseProfile(BaseModel):
    """Audience-belief revision payload — Itti-Baldi Bayesian surprise.

    Sibling to :class:`ThreatProximity`. Sourced from
    :func:`shadow_loom.affect_unification.compute_surprise_unified`,
    which sums ``KL(p_aud(P, t) || p_aud(P, t-1)) · stakes`` across
    every proposition whose audience confidence moved between the
    prior and current fabula anchor. See
    ``/memories/repo/affect-unification-plan.md`` Step 6.
    """
    score: float = Field(
        default=0.0,
        description=(
            "Total Bayesian-surprise score at the current fabula "
            "anchor. Unbounded above (sum of KL contributions); "
            "renderers should treat large jumps relative to recent "
            "history as strong revelation cues."
        ),
    )
    prior_fabula_t: Optional[int] = Field(
        default=None,
        description="Fabula time the belief revision is measured against.",
    )
    fabula_t: Optional[int] = Field(
        default=None,
        description="Fabula time at which surprise is measured.",
    )
    revealed_proposition_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Propositions whose audience confidence shifted by more "
            "than ``shift_threshold`` between the two anchors, "
            "ordered by KL contribution descending. Diagnostic — the "
            "renderer can grep these against ``world.propositions`` "
            "for human-readable descriptions of *what* surprised."
        ),
    )
    revealed_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable descriptions of the propositions in "
            "``revealed_proposition_ids``, in the same order. "
            "Surfaced into the renderer prompt so the model knows "
            "*what* the audience just learned without having to "
            "join against ``world.propositions``."
        ),
    )
    pleasant_score: float = Field(
        default=0.0,
        description=(
            "Tan (1996) / Ortony-Clore-Collins (1988) valence split: "
            "sum of KL contributions where the belief revision was "
            "*aligned* with the focal entity's concerns (a desired "
            "proposition turning true, or a feared one turning "
            "false). Renderers can use the ratio of pleasant vs. "
            "unpleasant to choose between 'windfall' and 'twist' "
            "surprise registers."
        ),
    )
    unpleasant_score: float = Field(
        default=0.0,
        description=(
            "Tan/Ortony valence split — KL contributions where the "
            "belief revision *opposed* the focal's concerns."
        ),
    )
    per_focal_score: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-entity surprise score, weighted by that entity's "
            ":class:`Concern` salience for each shifted proposition. "
            "Lets the renderer surface the character whose stake in "
            "the revelation is highest, even if the audience's raw "
            "KL is dominated by lower-stakes propositions."
        ),
    )


class IronyProfile(BaseModel):
    """Per-character audience-vs-focal divergence — Pfister/Sternberg KL.

    Sibling to :class:`ThreatProximity`. Sourced from
    :func:`shadow_loom.affect_unification.compute_irony_unified`,
    which sums ``KL(p_aud(P, t) || p_focal(P, t)) · stakes`` across
    propositions where the audience and the focal entity disagree.
    KL is asymmetric so audience-knows-more and focal-knows-more
    produce distinguishable scores (split into two siblings of the
    same shape).
    """
    focal_id: Optional[str] = Field(
        default=None,
        description=(
            "Entity id whose perspective the irony is measured "
            "against. ``None`` means the brief had no POV anchor."
        ),
    )
    audience_advantage_score: float = Field(
        default=0.0,
        description=(
            "Sum of ``KL(p_aud || p_focal) · stakes`` across all "
            "propositions — magnitude of dramatic irony where the "
            "audience knows something the focal does not."
        ),
    )
    focal_advantage_score: float = Field(
        default=0.0,
        description=(
            "Sum of ``KL(p_focal || p_aud) · stakes`` across all "
            "propositions — magnitude of mystery-from-the-audience-"
            "side where the focal knows something the audience does "
            "not."
        ),
    )
    fabula_t: Optional[int] = Field(
        default=None,
        description="Fabula time at which irony is measured.",
    )
    by_other_focal: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "``audience_advantage_score`` for every other non-"
            "audience entity whose score MATERIALLY differs from "
            "the primary focal's (|Δ| > 0.5). Suppressed when all "
            "siblings collapse to the same prior-baseline value, "
            "to keep the renderer prompt free of pseudo-signal. "
            "Useful when the brief targets multiple characters."
        ),
    )
    audience_advantage_propositions: List[str] = Field(
        default_factory=list,
        description=(
            "Top-k human-readable descriptions of propositions where "
            "the audience's confidence exceeds the focal's by more "
            "than 0.3 — i.e. the concrete things the audience knows "
            "that the focal does not. Ordered by stakes-weighted KL "
            "contribution descending. The renderer reads these to "
            "surface the irony as behaviour and dialogue (focal "
            "acting on a false sense of security) without naming "
            "the gap."
        ),
    )
    focal_advantage_propositions: List[str] = Field(
        default_factory=list,
        description=(
            "Top-k human-readable descriptions of propositions where "
            "the focal's confidence exceeds the audience's by more "
            "than 0.3 — i.e. things the focal knows that the audience "
            "does not (audience-side mystery). The renderer reads "
            "these to render the focal's interior knowledge as "
            "private texture without leaking it to the audience."
        ),
    )
    suspense_irony_score: float = Field(
        default=0.0,
        description=(
            "Sternberg (1978) three-mode split: audience-advantage KL "
            "restricted to ``kind='outcome'`` propositions — the "
            "audience knows how it ends and the focal does not."
        ),
    )
    curiosity_irony_score: float = Field(
        default=0.0,
        description=(
            "Sternberg three-mode split: audience-advantage KL on "
            "``kind in {identity_is, relation_holds}`` — the audience "
            "knows who/what someone is and the focal does not."
        ),
    )
    surprise_irony_score: float = Field(
        default=0.0,
        description=(
            "Sternberg three-mode split: audience-advantage KL on "
            "``kind in {event_occurs, trait_holds}`` — the audience "
            "knows a fact the focal will soon discover."
        ),
    )
    concern_weighted_score: float = Field(
        default=0.0,
        description=(
            "Pfister (1977/1988) felicity-conditions weighting: "
            "audience-advantage KL multiplied per-prop by the focal's "
            ":class:`Concern` salience for that proposition. Drops "
            "pseudo-irony on facts the focal does not care about."
        ),
    )
    most_ironised_entity_id: Optional[str] = Field(
        default=None,
        description=(
            "Wall (1983) discrepant-awareness gradient: the non-"
            "audience entity whose KL(audience || self) is highest "
            "at this anchor. Often but not always the focal — when "
            "different, the renderer should consider whether to "
            "shift the spotlight."
        ),
    )
    most_ironised_score: float = Field(
        default=0.0,
        description="KL(audience || most_ironised_entity).",
    )


class MysteryProfile(BaseModel):
    """Carroll erotetic mystery — entropy over hidden causes of known effects.

    Sibling to :class:`ThreatProximity`. Sourced from
    :func:`shadow_loom.affect_unification.compute_mystery_unified`,
    which for each effect proposition the audience confidently knows
    has happened sums Shannon entropy of the softmax-normalised
    causal_force distribution over its *unrevealed* ancestors in the
    causal graph.
    """
    score: float = Field(
        default=0.0,
        description=(
            "Total mystery score at the current fabula anchor. "
            "Higher = more known effects with diffuse, unresolved "
            "causal antecedents."
        ),
    )
    fabula_t: Optional[int] = Field(
        default=None,
        description="Fabula time at which mystery is measured.",
    )
    revelation_threshold: float = Field(
        default=0.7,
        description=(
            "Audience-confidence threshold above which a proposition "
            "counts as 'known' and so contributes its hidden-cause "
            "entropy."
        ),
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description=(
            "Top-k human-readable descriptions of effect propositions "
            "the audience confidently knows occurred but whose causal "
            "antecedents are not yet revealed. These are the open "
            "erotetic questions Carroll's mystery-as-question-set "
            "theory says drive the affect; the renderer surfaces them "
            "as *consequences without named causes* — show the effect "
            "on the page, suppress its 'why'."
        ),
    )
    plot_gap_score: float = Field(
        default=0.0,
        description=(
            "Carroll (1990) erotetic plot-gap score — the canonical "
            "mystery score (entropy over hidden causes of known "
            "effects). Identical to ``score`` and surfaced "
            "explicitly to mirror the breakdown structure."
        ),
    )
    character_gap_score: float = Field(
        default=0.0,
        description=(
            "Iser (1978) Leerstellen — sum of audience entropy on "
            "open ``identity_is`` / ``trait_holds`` propositions. "
            "Captures *who-is-X-really* mysteries that are invisible "
            "to the causal-ancestor scorer."
        ),
    )
    tellability_weighted_score: float = Field(
        default=0.0,
        description=(
            "Ryan (1991) tellability — open-question entropy weighted "
            "by the count and salience of :class:`Concern` records "
            "across all entities referencing each proposition. "
            "Mysteries many characters care about dominate."
        ),
    )
    governing_question_id: Optional[str] = Field(
        default=None,
        description=(
            "Carroll's macro-question — the open proposition whose "
            "resolution would commit the most other propositions "
            "(highest stakes × causal in-degree). The renderer "
            "should treat this as the spine the scene's mysteries "
            "orbit."
        ),
    )
    governing_question_description: Optional[str] = Field(
        default=None,
        description="Human-readable text for ``governing_question_id``.",
    )
    character_gap_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Top-k descriptions of open identity/trait propositions "
            "feeding ``character_gap_score``. The renderer surfaces "
            "these as ambiguity in characterisation rather than "
            "causal absence."
        ),
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
    causal_chain_descriptions: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable description of each event in ``causal_chain``, "
            "in the same order. Populated by the brief builder by resolving "
            "each id against ``WorldStateV1.events``. Empty entries (ids that "
            "could not be resolved) are left as empty strings. Surfaces so "
            "the renderer sees what each step in the causal chain *was*, "
            "not just its opaque id."
        ),
    )


# ---------------------------------------------------------------------------
# Character-felt emotion payloads (sourced from
# :mod:`shadow_loom.affect_unification`).
# ---------------------------------------------------------------------------

class FearProfile(BaseModel):
    """Lazarus appraisal × Öhman/LeDoux fear/anxiety split × Frijda dread.

    Sourced from
    :func:`shadow_loom.affect_unification.compute_fear_appraisal`.
    """
    object_fear_score: float = Field(
        default=0.0,
        description=(
            "Lazarus (1991) — focal's high-confidence fear-polarity "
            "concerns weighted by stakes \u00d7 salience \u00d7 (1 - coping)."
        ),
    )
    anxiety_score: float = Field(
        default=0.0,
        description=(
            "\u00d6hman & Mineka (2001) / LeDoux (1996) split: Shannon "
            "entropy across the focal's fear-concern beliefs. Diffuse, "
            "object-less dread of the unknown."
        ),
    )
    coping_score: float = Field(
        default=0.5,
        description=(
            "Lazarus secondary appraisal proxy in [0, 1] derived from "
            "the focal's resilience traits. Lower = the threat reads "
            "as overwhelming."
        ),
    )
    flight_available: bool = Field(
        default=True,
        description=(
            "Frijda action-readiness: True if focal's location has "
            "any spatial exit. False ⇒ no escape; combine with low "
            "coping for ``dread`` mode."
        ),
    )
    dread: bool = Field(
        default=False,
        description=(
            "Object-fear high AND coping low AND no flight. Render "
            "as paralytic dread (held breath, immobility) rather than "
            "active fear (running, fighting)."
        ),
    )
    primary_concern_id: Optional[str] = Field(
        default=None,
        description="The single dominant fear concern's id.",
    )
    primary_concern_description: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of the primary fear's "
            "proposition. Renderer dramatises *this* without naming "
            "the score."
        ),
    )


class JoyProfile(BaseModel):
    """Fredrickson broaden-and-build × OCC happy-for/gloating × Lazarus relief.

    Sourced from
    :func:`shadow_loom.affect_unification.compute_joy_appraisal`.
    """
    own_joy_score: float = Field(
        default=0.0,
        description=(
            "Fredrickson (2001) — focal's realised desire concerns "
            "(belief \u2192 1.0) summed and weighted by stakes \u00d7 "
            "salience. Drives broaden-and-build prose register."
        ),
    )
    happy_for_score: float = Field(
        default=0.0,
        description=(
            "OCC (Ortony et al., 1988) — joy on behalf of liked "
            "others (affinity > 0.3) whose desires are realised."
        ),
    )
    gloating_score: float = Field(
        default=0.0,
        description=(
            "OCC schadenfreude — disliked others' (affinity < -0.3) "
            "feared events realised. Renderer should treat as a "
            "morally complicating shade on the joy."
        ),
    )
    relief_score: float = Field(
        default=0.0,
        description=(
            "Lazarus relief — focal's feared concerns whose belief "
            "just dropped toward false. Distinct from joy proper; "
            "renderer should register it as physiological release "
            "(unclenching, exhale) rather than expansion."
        ),
    )
    primary_concern_id: Optional[str] = None
    primary_concern_description: Optional[str] = None


class RegretProfile(BaseModel):
    """Kahneman-Miller × Roese commission/omission × Gilovich downward.

    Sourced from
    :func:`shadow_loom.affect_unification.compute_regret_appraisal`.
    """
    agentive_regret_score: float = Field(
        default=0.0,
        description=(
            "Kahneman & Miller (1986) norm theory — regret weighted "
            "by closeness of the unchosen counterfactual (smaller "
            "fabula gap to divergence event \u2192 closer) and "
            "controllability (focal's choice event)."
        ),
    )
    disappointment_score: float = Field(
        default=0.0,
        description=(
            "Negative outcome with no controllable divergence \u2014 "
            "disappointment rather than regret. Renderer should use "
            "passive grief register, not 'if only' interiority."
        ),
    )
    commission_score: float = Field(
        default=0.0,
        description=(
            "Roese (1997) action asymmetry \u2014 hot, short-term regret "
            "over a deed done. Render acutely (vivid sensory recall "
            "of the act)."
        ),
    )
    omission_score: float = Field(
        default=0.0,
        description=(
            "Roese (1997) inaction asymmetry \u2014 cold, long-term "
            "regret over a deed not done. Render as brooding "
            "absence, the unspoken word."
        ),
    )
    downward_relief_score: float = Field(
        default=0.0,
        description=(
            "Gilovich & Medvec (1995) downward counterfactual \u2014 "
            "magnitude by which an averted sibling outcome was worse "
            "than what actually happened. Tinges the regret with "
            "'could have been worse'."
        ),
    )
    divergence_event_id: Optional[str] = None
    divergence_description: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of the divergence event — "
            "the choice the focal made (or failed to make) that produced "
            "the regret. Resolved from ``WorldStateV1`` at build time so "
            "the renderer knows *what* the unchosen path branched from."
        ),
    )
    loss_event_id: Optional[str] = None
    loss_description: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of the loss event. Resolved from "
            "``WorldStateV1`` at build time."
        ),
    )
    mode: str = Field(
        default="none",
        description=(
            "Classification: ``commission`` | ``omission`` | "
            "``disappointment`` | ``none``."
        ),
    )


class GriefProfile(BaseModel):
    """Bowlby attachment × Kübler-Ross stage × Worden tasks.

    Sourced from
    :func:`shadow_loom.affect_unification.compute_grief_appraisal`.
    """
    coupling_strength: float = Field(
        default=0.0,
        description=(
            "Bowlby (1969/1980) bond strength \u2014 geometric mean of "
            "mutual affinity between focal and the deceased. Drives "
            "grief intensity."
        ),
    )
    loss_event_id: Optional[str] = None
    loss_description: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of the loss event. Resolved from "
            "``WorldStateV1`` at build time."
        ),
    )
    lost_entity_id: Optional[str] = None
    stage: str = Field(
        default="none",
        description=(
            "K\u00fcbler-Ross (1969) detected stage: ``denial`` | "
            "``anger`` | ``bargaining`` | ``depression`` | "
            "``acceptance`` | ``none``. Renderer keys prose register "
            "to stage \u2014 denial reads as numb routine, anger as "
            "directed hostility, bargaining as intrusive 'if only' "
            "thinking, depression as fragmented absence, acceptance "
            "as quiet integration."
        ),
    )
    unfinished_concern_count: int = Field(
        default=0,
        description=(
            "Worden (1991) tasks \u2014 count of focal concerns "
            "referencing the lost entity that remain unrevised. "
            "Higher \u2192 more unfinished mourning."
        ),
    )


class RageProfile(BaseModel):
    """Berkowitz × Averill × Tedeschi-Felson coercive action.

    Sourced from
    :func:`shadow_loom.affect_unification.compute_rage_appraisal`.
    """
    blocked_concern_score: float = Field(
        default=0.0,
        description=(
            "Berkowitz (1989) frustration-aggression \u2014 focal's "
            "fear concerns whose proposition the loss event "
            "committed, summed weighted by stakes \u00d7 salience."
        ),
    )
    perpetrator_id: Optional[str] = None
    attribution_clarity: float = Field(
        default=0.0,
        description=(
            "Focal's confidence in the perpetrator-identity "
            "proposition. Diffuse blame (low) \u2192 frustration mode; "
            "sharp blame (high) \u2192 directed/retributive rage."
        ),
    )
    perpetrator_proximity: int = Field(
        default=-1,
        description=(
            "Spatial hops focal\u2192perpetrator. 0 = co-located, "
            "1 = elsewhere, -1 = unknown. Closer \u2192 more acute."
        ),
    )
    normative_violation: bool = Field(
        default=False,
        description=(
            "Averill (1982) \u2014 True when the violated concern's "
            "``kind`` is in {betrayal, abandonment, humiliation, "
            "injustice}. Required for rage as opposed to mere anger."
        ),
    )
    mode: str = Field(
        default="none",
        description=(
            "Tedeschi & Felson (1994) classification: "
            "``frustration`` | ``directed_rage`` | "
            "``retributive_rage`` | ``displaced_rage`` | ``none``."
        ),
    )


class LoveProfile(BaseModel):
    """Sternberg triangular × Berscheid-Hatfield × Bowlby attachment style.

    Sourced from
    :func:`shadow_loom.affect_unification.compute_love_appraisal`.
    """
    primary_partner_id: Optional[str] = None
    intimacy_score: float = Field(
        default=0.0,
        description=(
            "Sternberg (1986) intimacy \u2014 mutual high-confidence "
            "beliefs about each other. Wide proposition coverage."
        ),
    )
    passion_score: float = Field(
        default=0.0,
        description=(
            "Sternberg passion \u2014 sum of focal concern salience for "
            "propositions referring to the partner. High-salience "
            "narrow set."
        ),
    )
    commitment_score: float = Field(
        default=0.0,
        description=(
            "Sternberg commitment \u2014 affinity inertia \u00d7 normalised "
            "relationship age. Resistance to drift."
        ),
    )
    style: str = Field(
        default="none",
        description=(
            "Berscheid & Hatfield (1969/1974) register dominant: "
            "``passionate`` | ``companionate`` | ``balanced`` | "
            "``none``. Renderer picks yearning vs. quiet familiarity."
        ),
    )
    attachment_style: str = Field(
        default="unknown",
        description=(
            "Bowlby (1969) attachment classification from the "
            "relationship's affinity/fear metrics: ``secure`` | "
            "``anxious`` | ``avoidant`` | ``unknown``."
        ),
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
    world_trait_shifts: List[WorldTraitShift] = Field(
        default_factory=list,
        description=(
            "Recent WORLD_ trait shifts (regime changes, prophecy "
            "resolutions, ambient mood reversals) folded onto "
            "``GlobalTrait.state_timeline`` at or before the brief's "
            "syuzhet anchor. Empty when no movement is recent."
        ),
    )
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
    intervention_branch: Optional["InterventionBranch"] = Field(
        default=None,
        description=(
            "Rung-2 sandbox payload populated for intervention "
            "(do-calculus) briefs. Carries the typed do_target, the "
            "list of all do_targets, and the "
            "affected_propositions / affected_beliefs / "
            "affected_concerns the engine flipped under the "
            "intervention. Lets the renderer surface — and the auditor "
            "verify — every node the surgery touched, mirroring the "
            "Rung-3 ``counterfactual_branch`` channel."
        ),
    )
    surprise_profile: Optional["SurpriseProfile"] = Field(
        default=None,
        description=(
            "Audience-belief Bayesian-surprise reading at the brief's "
            "syuzhet anchor. Populated for ``surprise`` briefs and "
            "available as a sibling diagnostic for any other effect "
            "that wants to inspect concurrent revelation magnitude. "
            "See :class:`SurpriseProfile`."
        ),
    )
    irony_profile: Optional["IronyProfile"] = Field(
        default=None,
        description=(
            "Per-character audience-vs-focal divergence at the "
            "brief's syuzhet anchor. Populated for "
            "``dramatic_irony`` briefs. See :class:`IronyProfile`."
        ),
    )
    mystery_profile: Optional["MysteryProfile"] = Field(
        default=None,
        description=(
            "Carroll erotetic mystery score at the brief's syuzhet "
            "anchor. Populated for ``mystery`` briefs. See "
            ":class:`MysteryProfile`."
        ),
    )
    causal_attribution: Optional[CausalAttribution] = None
    entanglement_pairs: List[EntanglementPair] = Field(default_factory=list)
    fear_profile: Optional["FearProfile"] = Field(
        default=None,
        description=(
            "Lazarus/Öhman/Frijda fear appraisal at the brief's "
            "syuzhet anchor. Populated for ``fear`` briefs."
        ),
    )
    joy_profile: Optional["JoyProfile"] = Field(
        default=None,
        description=(
            "Fredrickson/OCC joy appraisal at the brief's syuzhet "
            "anchor. Populated for ``joy`` briefs."
        ),
    )
    regret_profile: Optional["RegretProfile"] = Field(
        default=None,
        description=(
            "Kahneman-Miller/Roese regret appraisal at the brief's "
            "syuzhet anchor. Populated for ``regret`` briefs."
        ),
    )
    grief_profile: Optional["GriefProfile"] = Field(
        default=None,
        description=(
            "Bowlby/Kübler-Ross/Worden grief appraisal at the brief's "
            "syuzhet anchor. Populated for ``grief`` briefs."
        ),
    )
    rage_profile: Optional["RageProfile"] = Field(
        default=None,
        description=(
            "Berkowitz/Averill/Tedeschi-Felson rage appraisal at the "
            "brief's syuzhet anchor. Populated for ``rage`` briefs."
        ),
    )
    love_profile: Optional["LoveProfile"] = Field(
        default=None,
        description=(
            "Sternberg/Berscheid-Hatfield/Bowlby love appraisal at "
            "the brief's syuzhet anchor. Populated for ``love`` briefs."
        ),
    )
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


# Improvement B10: assert that every mechanism string the canonical
# causal_physics map can produce resolves to a harm-kind salience
# weight in ``DirectiveAssembler._HARM_KIND_SALIENCE``. Without this,
# a freshly added mechanism would silently default to the "physical"
# salience and the suspense / fear / threat-proximity gauges would
# under-weight or mis-class the new harm category. The aliases must
# stay in sync with the lower-casing + alias map inside
# ``_harm_kind_for_event``.
def _validate_mechanism_salience_coverage() -> None:
    try:
        from shadow_loom.causal_physics import MECHANISM_TRAIT_MAP
    except Exception:
        # Avoid breaking the module if causal_physics changes shape;
        # the assertion is a build-time hygiene check, not a runtime
        # contract.
        return
    salience_keys = {
        "existential", "physical", "betrayal", "psychological",
        "emotional", "social", "epistemic", "informational",
    }
    aliases = {
        "physical_force": "physical",
        "epistemic_revelation": "epistemic",
        "social_coercion": "social",
    }
    missing: List[str] = []
    for mech in MECHANISM_TRAIT_MAP.keys():
        m = mech.strip().lower()
        m = aliases.get(m, m)
        if m not in salience_keys:
            missing.append(mech)
    if missing:
        raise AssertionError(
            "DirectiveAssembler._HARM_KIND_SALIENCE is missing entries "
            f"for causal_physics mechanisms: {missing}. Add them (or "
            "extend the alias table in _harm_kind_for_event) so the "
            "suspense / fear / threat-proximity gauges classify the "
            "new harm category instead of silently defaulting to "
            "'physical'."
        )


_validate_mechanism_salience_coverage()


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


# Event-types the instantiator treats as "this did not happen". Any
# event carrying one of these tags is part of the world's *negative*
# physics record — the renderer must not stage it as occurring.
_PREVENTED_EVENT_TYPES = frozenset({"prevented", "never_happened", "removed"})


def build_object_coherence_constraints(
    world_state: WorldStateV1,
    fabula_anchor: Optional[int],
    *,
    world_label: str = "this",
) -> List[ConstraintBlock]:
    """HARD constraints anchoring narrative objects to their physics state.

    For every :class:`NarrativeObject` reconstructed at
    ``fabula_anchor`` (using :func:`reconstruct_object_at`) emit a
    constraint that names:
      * the object's current ``location_id`` or ``owner_id`` (which-
        ever applies), so the renderer cannot place it elsewhere;
      * the object's ``affordances`` whitelist, so the renderer
        cannot have a character perform an action the object does
        not support (a poison can ``kill``, not ``read``).

    Without these blocks, prose generation receives objects only as
    free-text scene context, while entities receive *mathematical*
    trait-trajectory constraints. The asymmetry routinely produced
    drift — a dagger left in the kitchen reappearing in the bedroom,
    a letter being "read aloud" when its only affordance is
    ``inform`` (silent transmission). Mirrors the negative-physics
    pattern used by ``build_prevented_event_constraints``.
    """
    if not getattr(world_state, "objects", None):
        return []
    blocks: List[ConstraintBlock] = []
    for obj_id, obj in world_state.objects.items():
        if fabula_anchor is not None and obj.state_timeline:
            recon = reconstruct_object_at(obj, fabula_anchor)
            location_id = recon["location_id"]
            owner_id = recon["owner_id"]
            properties = recon["properties"]
        else:
            location_id = obj.location_id
            owner_id = obj.owner_id
            properties = dict(obj.properties)
        if owner_id:
            position_clause = f"is in the possession of `{owner_id}`"
        elif location_id:
            position_clause = f"is located at `{location_id}`"
        else:
            # No declared position at this anchor — skip rather than emit
            # a noisy "is somewhere" constraint that the renderer would
            # have to ignore.
            continue
        affordance_actions = sorted({
            (aff.action or "").strip().lower()
            for aff in (obj.affordances or [])
            if aff.action
        })
        affordance_clause = (
            f" Its declared affordances are: {affordance_actions!r}; "
            f"do NOT have any character perform an action on it that is "
            f"not in this list."
            if affordance_actions
            else ""
        )
        prop_clause = (
            f" Its current properties are {properties!r}; respect them "
            f"(if `state` is `poisoned`, drinking from it is fatal; etc.)."
            if properties else ""
        )
        blocks.append(ConstraintBlock(
            constraint_type="spatial",
            priority="hard",
            instruction=(
                f"In {world_label} world the object `{obj_id}` "
                f"({obj.name}) {position_clause} at the scene's anchor. "
                f"Do NOT place it elsewhere or have a character interact "
                f"with it from a different location.{affordance_clause}"
                f"{prop_clause}"
            ),
            evidence={
                "object_id": obj_id,
                "name": obj.name,
                "location_id": location_id,
                "owner_id": owner_id,
                "affordances": affordance_actions,
                "properties": properties,
                "fabula_anchor": fabula_anchor,
            },
        ))
    return blocks


def build_event_copresence_constraints(
    world_state: WorldStateV1,
    fabula_anchor: Optional[int],
    syuzhet_anchor: Optional[int],
    *,
    world_label: str = "this",
    window: int = 1,
) -> List[ConstraintBlock]:
    """HARD constraints binding the prose to the engine's event-location
    and co-presence ledger.

    For every event in the scene window (``|fabula_time - fabula_anchor|
    <= window`` AND ``syuzhet_index <= syuzhet_anchor`` when both are
    set) that carries an ``at_location_id``, emit a single
    ``ConstraintBlock`` naming three things the renderer must honour:

      1. ``MUST_DEPICT_AT`` \u2014 the event happens at the named
         location; do NOT relocate it.
      2. ``MUST_BE_PRESENT`` \u2014 every actor + non-channel target is
         physically present at that location at ``fabula_time``.
         Channel-mediated participants (utterance with
         ``via_channel_id`` set; addressees reached through that
         channel) are exempt and listed separately.
      3. ``MUST_NOT_BE_PRESENT`` \u2014 entities whose reconstructed
         location at ``fabula_time`` is NOT the event's location
         must NOT be staged at the event (no phantom witnesses).

    The auditor's deterministic ``_event_copresence_violations``
    pass consumes these constraints' ``evidence`` payloads.
    """
    if not getattr(world_state, "events", None):
        return []
    blocks: List[ConstraintBlock] = []
    entities = world_state.entities or {}
    locations = world_state.locations or {}
    for evt in world_state.events:
        loc_id = getattr(evt, "at_location_id", None)
        if not loc_id:
            continue
        if syuzhet_anchor is not None and evt.syuzhet_index > syuzhet_anchor:
            continue
        if (
            fabula_anchor is not None
            and abs(int(evt.fabula_time) - int(fabula_anchor)) > window
        ):
            continue
        loc_name = (
            locations[loc_id].name
            if loc_id in locations
            else loc_id
        )
        # Resolve channel-mediated exemption.
        channel_id = getattr(evt, "via_channel_id", None)
        addressees = list(getattr(evt, "addressee_ids", None) or [])
        actors = list(getattr(evt, "actor_ids", None) or [])
        targets = list(getattr(evt, "target_ids", None) or [])
        speaker = getattr(evt, "speaker_id", None)
        if speaker and speaker not in actors:
            actors = [speaker] + actors

        bound_present: list[str] = []
        channel_exempt: list[str] = []
        seen_b: set[str] = set()
        for pid in actors + targets + addressees:
            if not pid or pid in seen_b:
                continue
            seen_b.add(pid)
            # Speaker is always physically present at the event location;
            # only addressees / non-speaker targets get the channel
            # exemption.
            if (
                channel_id
                and pid != speaker
                and (pid in addressees or pid in targets)
            ):
                channel_exempt.append(pid)
            else:
                bound_present.append(pid)

        # Phantom-witness ledger: entities whose reconstructed location
        # at this event's fabula_time is NOT the event location.
        must_not_be_present: list[dict[str, str]] = []
        ft = int(evt.fabula_time)
        for eid, ent in (entities.items() if isinstance(entities, dict) else []):
            if eid in seen_b:
                continue
            try:
                snap = reconstruct_entity_at(ent, ft)
            except Exception:
                continue
            other_loc = snap.get("location_id") if isinstance(snap, dict) else None
            if other_loc and other_loc != loc_id:
                must_not_be_present.append({
                    "entity_id": eid,
                    "name": getattr(ent, "name", eid),
                    "elsewhere_id": other_loc,
                })

        present_clause = (
            f" The following must be physically present in the scene at "
            f"`{loc_id}`: {bound_present!r}."
            if bound_present else ""
        )
        channel_clause = (
            f" The following are reached over channel `{channel_id}` and "
            f"are NOT physically present at `{loc_id}`: {channel_exempt!r}; "
            f"render their participation as channel-mediated (call, letter, "
            f"telegram, mind-link, etc.)."
            if channel_exempt else ""
        )
        absent_ids = [r["entity_id"] for r in must_not_be_present]
        absent_clause = (
            f" Do NOT stage these characters as present in the scene "
            f"(they are elsewhere at fabula_time={ft}): {absent_ids!r}."
            if absent_ids else ""
        )
        blocks.append(ConstraintBlock(
            constraint_type="spatial",
            priority="hard",
            instruction=(
                f"In {world_label} world, event `{evt.id}` "
                f"({evt.event_type}) happens at `{loc_id}` ({loc_name}) "
                f"at fabula_time={ft}. Do NOT relocate it.{present_clause}"
                f"{channel_clause}{absent_clause}"
            ),
            evidence={
                "event_id": evt.id,
                "at_location_id": loc_id,
                "fabula_time": ft,
                "must_be_present": bound_present,
                "channel_exempt": channel_exempt,
                "via_channel_id": channel_id,
                "must_not_be_present": absent_ids,
                "must_not_be_present_detail": must_not_be_present,
            },
        ))
    return blocks


def build_prevented_event_constraints(
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int],
    *,
    world_label: str = "this",
) -> List[ConstraintBlock]:
    """HARD constraints for events the physics records as NOT occurring.

    Rendered prose must not stage any event whose ``event_type`` is in
    :data:`_PREVENTED_EVENT_TYPES` (``prevented`` / ``never_happened`` /
    ``removed``). These tags are emitted by the instantiator and the
    Rung-2/3 surgery paths to mark non-occurrences in the canonical
    record; without surfacing them on the brief the renderer reliably
    re-narrates them as having happened (the event row still carries a
    natural-language description).

    Applied to every brief builder (observation / intervention /
    counterfactual / directive) so the "what NOT to do" half of causal
    physics travels alongside the "what to do" mechanism block.
    """
    events = list(getattr(world_state, "events", []) or [])
    if not events:
        return []
    cap = syuzhet_anchor
    prevented = []
    for e in events:
        if (getattr(e, "event_type", None) or "") not in _PREVENTED_EVENT_TYPES:
            continue
        ft = getattr(e, "fabula_time", None)
        if cap is not None and ft is not None and ft > cap:
            continue
        prevented.append(e)
    if not prevented:
        return []
    lines: List[str] = []
    for e in prevented[:20]:
        eid = getattr(e, "id", "?")
        et = getattr(e, "event_type", "?")
        desc = (getattr(e, "description", "") or "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
        snippet = f" \u2014 {desc}" if desc else ""
        lines.append(f"  - {eid} [{et}]{snippet}")
    if len(prevented) > 20:
        lines.append(f"  - ...and {len(prevented) - 20} more.")
    return [ConstraintBlock(
        constraint_type="narrative",
        priority="hard",
        instruction=(
            f"=== PREVENTED EVENTS (HARD) === \u2014 the physics tags these "
            f"events as not occurring in the {world_label} world. Do "
            "NOT render any of them as having happened, do not stage "
            "them in real time, and do not have characters witness, "
            "remember, or react to them as past events. If a "
            "character's plan or expectation depended on one of them, "
            "render the consequence of its non-occurrence (the gap, "
            "the frustrated plan, the absence) \u2014 not the event "
            "itself.\n"
            + "\n".join(lines)
        ),
        evidence={"prevented_event_ids": [getattr(e, "id", "?") for e in prevented]},
    )]


def _expand_chain_reaction_closure(
    world_state: WorldStateV1,
    root_event_ids: List[str],
) -> tuple[set[str], List[tuple[str, str, str]]]:
    """Return ``(closure_event_ids, edges_in_closure)`` for the
    disjunctive chain_reaction descendant rule.

    Thin wrapper around
    :func:`shadow_loom.causal_closure.expand_chain_reaction_closure`
    and :func:`shadow_loom.causal_closure.edges_within_closure`.
    Retained as a local helper so the two call-sites below stay
    readable; the heavy lifting lives in ``causal_closure``.
    """
    roots = set(root_event_ids or [])
    if not roots:
        return set(), []
    parents_of = chain_reaction_parents_from_world_state(world_state)
    closure = expand_chain_reaction_closure(parents_of, roots)
    return closure, edges_within_closure(parents_of, closure)


def build_prune_cascade_context_constraints(
    world_state: WorldStateV1,
    pruned_root_event_ids: Optional[List[str]],
    *,
    world_label: str = "this",
) -> List[ConstraintBlock]:
    """HARD context block: explain the ORIGINAL causal chains the
    Rung-2/3 do-surgery severed, so the renderer knows what it must
    NOT confabulate a replacement for.

    Without this block the renderer sees only ``=== PREVENTED EVENTS
    (HARD) ===`` (a bare list of event ids it must not stage) and is
    free to invent a *novel* failure mode for any character whose
    plan depended on the pruned event (e.g. a counterfactual that
    prunes "Ken kills Mrs Coady's dogs" routinely produces prose
    where the prosecution loses for the wrong reason \u2014 "missing
    canine evidence" \u2014 because the renderer can see the trial
    event downstream but not the actual link). Surfacing the original
    parent\u2192child mechanisms lets the renderer route around the
    severed chains explicitly rather than guessing.

    Paired with :func:`build_dependent_state_substitution_constraints`
    so the renderer also sees the *positive* substitutions (entities
    whose state did NOT flip because their cause was pruned).
    """
    if not pruned_root_event_ids:
        return []
    closure, edges = _expand_chain_reaction_closure(
        world_state, pruned_root_event_ids,
    )
    if not closure:
        return []
    events_by_id = {
        getattr(e, "id", None): e
        for e in getattr(world_state, "events", []) or []
    }

    def _desc(eid: str) -> str:
        e = events_by_id.get(eid)
        if e is None:
            return ""
        d = (getattr(e, "description", "") or "").strip()
        if len(d) > 100:
            d = d[:97] + "..."
        return d

    lines: List[str] = []
    # Show roots first, then each chain_reaction edge in the closure
    # so the renderer sees the full \"originally caused\" forest.
    roots_sorted = sorted(set(pruned_root_event_ids) & closure)
    if roots_sorted:
        lines.append("  PRUNED ROOTS (the do-surgery removed these):")
        for rid in roots_sorted[:20]:
            snippet = _desc(rid)
            tail = f" \u2014 {snippet}" if snippet else ""
            lines.append(f"    - {rid}{tail}")
        if len(roots_sorted) > 20:
            lines.append(f"    - ...and {len(roots_sorted) - 20} more roots.")
    if edges:
        lines.append(
            "  ORIGINAL CHAIN-REACTION LINKS now SEVERED "
            "(parent \u2192 child, original mechanism):"
        )
        for parent, child, mech in edges[:30]:
            p_desc = _desc(parent)
            c_desc = _desc(child)
            mech_tag = f" [{mech}]" if mech else ""
            p_tail = f" \u2014 {p_desc}" if p_desc else ""
            c_tail = f" \u2014 {c_desc}" if c_desc else ""
            lines.append(
                f"    - {parent}{p_tail}\n"
                f"        \u2192 {child}{c_tail}{mech_tag}"
            )
        if len(edges) > 30:
            lines.append(f"    - ...and {len(edges) - 30} more links.")
    # Downstream-only (non-root) events in the closure, so the
    # renderer sees the whole forbidden set at a glance.
    downstream = sorted(closure - set(roots_sorted))
    if downstream:
        lines.append(
            "  DOWNSTREAM EVENTS in the closure (also DO NOT occur):"
        )
        for did in downstream[:20]:
            snippet = _desc(did)
            tail = f" \u2014 {snippet}" if snippet else ""
            lines.append(f"    - {did}{tail}")
        if len(downstream) > 20:
            lines.append(f"    - ...and {len(downstream) - 20} more.")
    if not lines:
        return []
    instruction = (
        f"=== SEVERED CAUSAL CHAINS (HARD, CONTEXT) === \u2014 the "
        f"following original-world causal chains were SEVERED by the "
        f"do-surgery. NONE of these events occur in the {world_label} "
        f"world, and the original parent\u2192child links DO NOT fire. "
        f"Use this map to understand WHICH downstream consequences "
        f"are absent; do NOT invent a substitute mechanism that "
        f"reaches the same outcome by another route, and do NOT "
        f"reframe the absence as an evidentiary gap (\"missing "
        f"evidence\", \"absent witness\", \"unexplained vacancy\") \u2014 "
        f"the chain simply did not happen. Render the world AS IT "
        f"NOW IS without the cascade.\n"
        + "\n".join(lines)
    )
    return [ConstraintBlock(
        constraint_type="narrative",
        priority="hard",
        instruction=instruction,
        evidence={
            "pruned_root_event_ids": list(roots_sorted),
            "pruned_closure_event_ids": sorted(closure),
            "severed_chain_reaction_edges": [
                {"source": p, "target": c, "mechanism": m}
                for p, c, m in edges
            ],
        },
    )]


def _holder_from_affected_belief_id(b: str) -> Optional[str]:
    """Parse an ``affected_beliefs`` entry. narrative_physics emits
    ``\"{holder_id}\u2192{target_id}\"``; older shapes may pass a bare
    belief id. Returns the holder id when parseable, else None."""
    if not b or not isinstance(b, str):
        return None
    if "\u2192" in b:
        return b.split("\u2192", 1)[0] or None
    return None


def _holder_from_affected_concern_id(c: str) -> Optional[str]:
    """Concern entries are ``\"{holder_id}.{concern_id}\"`` (dot-joined).
    Returns the holder id when parseable, else None."""
    if not c or not isinstance(c, str) or "." not in c:
        return None
    head = c.split(".", 1)[0]
    return head or None


def build_dependent_state_substitution_constraints(
    world_state: WorldStateV1,
    pruned_root_event_ids: Optional[List[str]],
    affected_beliefs: Optional[List[str]] = None,
    affected_concerns: Optional[List[str]] = None,
    *,
    world_label: str = "this",
) -> List[ConstraintBlock]:
    """HARD positive-substitution block: tell the renderer the
    CURRENT post-prune status of every entity implicated by a severed
    causal chain, and instruct it to render their plans as PROCEEDING
    under the changed conditions rather than invent a substitute
    failure mode.

    The implicated set is the union of:
      * ``actor_ids`` and ``target_ids`` of every event in the prune
        closure (these are the characters whose fate the cascade
        originally decided);
      * holders parsed from ``affected_beliefs`` /
        ``affected_concerns`` (these are the characters whose
        epistemic/motivational state would have flipped).

    For each implicated entity we surface their current ``status``,
    ``location_id``, and (briefly) their key traits as the
    ground-truth substitution. This is the positive complement to
    :func:`build_prune_cascade_context_constraints` (which surfaces
    only what is ABSENT). Together they give the renderer both the
    severed structure and the post-surgery world to render against,
    closing the gap that lets it confabulate (e.g. a counterfactual
    that prevents Mrs Coady\u2019s death must render her ALIVE and
    able to fulfil her original role \u2014 not invent a different
    reason the prosecution fails).
    """
    if not pruned_root_event_ids and not affected_beliefs and not affected_concerns:
        return []
    closure, _edges = _expand_chain_reaction_closure(
        world_state, list(pruned_root_event_ids or []),
    )
    events_by_id = {
        getattr(e, "id", None): e
        for e in getattr(world_state, "events", []) or []
    }
    implicated: set[str] = set()
    for eid in closure:
        e = events_by_id.get(eid)
        if e is None:
            continue
        for a in (getattr(e, "actor_ids", None) or []):
            if isinstance(a, str) and a.startswith("ENT_"):
                implicated.add(a)
        for t in (getattr(e, "target_ids", None) or []):
            if isinstance(t, str) and t.startswith("ENT_"):
                implicated.add(t)
    for b in (affected_beliefs or []):
        h = _holder_from_affected_belief_id(b)
        if h and h.startswith("ENT_"):
            implicated.add(h)
    for c in (affected_concerns or []):
        h = _holder_from_affected_concern_id(c)
        if h and h.startswith("ENT_"):
            implicated.add(h)
    if not implicated:
        return []
    entities = getattr(world_state, "entities", {}) or {}
    lines: List[str] = []
    for eid in sorted(implicated)[:20]:
        ent = entities.get(eid)
        if ent is None:
            lines.append(f"  - {eid}: (not in current world_state)")
            continue
        name = getattr(ent, "name", None) or eid
        status = getattr(ent, "status", None) or "?"
        loc = getattr(ent, "location_id", None) or "?"
        lines.append(
            f"  - {eid} ({name}): status={status!r}, location={loc} "
            f"\u2014 render in this CURRENT state."
        )
    if len(implicated) > 20:
        lines.append(f"  - ...and {len(implicated) - 20} more entities.")
    instruction = (
        f"=== DEPENDENT-STATE SUBSTITUTIONS (HARD) === \u2014 the entities "
        f"below were implicated by the severed causal chains. Their "
        f"post-surgery state is shown; render them in EXACTLY that "
        f"state in the {world_label} world. If a character\u2019s "
        f"established plan or social role depended on a pruned "
        f"event (e.g. a witness whose death was pruned remains "
        f"available to testify; a prosecution whose key fact was "
        f"pruned still has its other facts), render that plan / role "
        f"as PROCEEDING under the changed conditions \u2014 do NOT "
        f"invent a substitute failure mode (missing evidence, "
        f"alternative obstacle, contingency collapse) to recover the "
        f"original outcome by another route. The severed chain "
        f"simply did not happen; the entity continues from the "
        f"pre-cascade state shown here.\n"
        + "\n".join(lines)
    )
    return [ConstraintBlock(
        constraint_type="narrative",
        priority="hard",
        instruction=instruction,
        evidence={"implicated_entity_ids": sorted(implicated)},
    )]


def build_false_proposition_constraints(
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int],
    *,
    world_label: str = "this",
) -> List[ConstraintBlock]:
    """HARD constraints for propositions committed FALSE at the anchor.

    Walks ``Proposition.truth_at_fabula`` for every catalogued
    proposition and selects the latest commit at or before
    ``syuzhet_anchor``. Propositions whose latest applicable commit is
    ``False`` are surfaced as a "do-not-render-as-true" block so the
    renderer cannot stage their content as occurring in the scene.

    Characters may still *believe* them (and that gap powers
    dramatic-irony / surprise effects); the constraint targets the
    narration layer, not the belief layer.
    """
    props = getattr(world_state, "propositions", None) or []
    if not props:
        return []
    # ``propositions`` is canonically a List[Proposition] but historic
    # snapshots / fixtures sometimes deliver a dict keyed by PROP_ id.
    if isinstance(props, dict):
        prop_iter = list(props.values())
    else:
        prop_iter = list(props)
    cap = syuzhet_anchor if syuzhet_anchor is not None else None
    falsified: List[tuple] = []
    for p in prop_iter:
        pid = getattr(p, "id", None) or getattr(p, "proposition_id", None) or "?"
        truth_map = getattr(p, "truth_at_fabula", {}) or {}
        if not truth_map:
            continue
        applicable = [
            (int(t), bool(v))
            for t, v in truth_map.items()
            if cap is None or int(t) <= cap
        ]
        if not applicable:
            continue
        applicable.sort(key=lambda kv: kv[0])
        last_t, last_v = applicable[-1]
        if last_v is False:
            falsified.append((pid, p, last_t))
    if not falsified:
        return []
    lines: List[str] = []
    for pid, p, t in falsified[:20]:
        desc = (getattr(p, "description", "") or "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
        snippet = f" \u2014 {desc}" if desc else ""
        lines.append(f"  - {pid} (false @ T={t}){snippet}")
    if len(falsified) > 20:
        lines.append(f"  - ...and {len(falsified) - 20} more.")
    return [ConstraintBlock(
        constraint_type="narrative",
        priority="hard",
        instruction=(
            f"=== FALSE PROPOSITIONS (HARD) === \u2014 the physics commits "
            f"these propositions FALSE at or before this scene's anchor "
            f"in the {world_label} world. Do NOT stage their content "
            "as occurring or having occurred. Characters may still "
            "*believe* them \u2014 that mismatch is allowed and is often "
            "the point \u2014 but the narration must not enact them as "
            "fact.\n"
            + "\n".join(lines)
        ),
        evidence={"false_proposition_ids": [pid for pid, _, _ in falsified]},
    )]


def build_true_proposition_constraints(
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int],
    *,
    world_label: str = "this",
    max_lines: int = 20,
) -> List[ConstraintBlock]:
    """HARD constraints for propositions committed TRUE at the anchor.

    Mirror of :func:`build_false_proposition_constraints`. Without a
    positive counterpart the renderer is told what *not* to enact but
    nothing about which world-truths it MUST honour. Surfacing the
    true-at-anchor propositions makes the brief symmetric so a Rung-1
    observation or Rung-3 counterfactual that lands on a TRUE-committed
    fact cannot quietly omit / contradict it (AUDIT brief/auditor
    consistency).

    Characters may still *disbelieve* these (dramatic irony); the
    constraint targets the narration layer only.
    """
    props = getattr(world_state, "propositions", None) or []
    if not props:
        return []
    prop_iter = list(props.values()) if isinstance(props, dict) else list(props)
    cap = syuzhet_anchor
    committed_true: List[tuple] = []
    for p in prop_iter:
        pid = getattr(p, "id", None) or getattr(p, "proposition_id", None) or "?"
        truth_map = getattr(p, "truth_at_fabula", {}) or {}
        if not truth_map:
            continue
        applicable = [
            (int(t), bool(v))
            for t, v in truth_map.items()
            if cap is None or int(t) <= cap
        ]
        if not applicable:
            continue
        applicable.sort(key=lambda kv: kv[0])
        last_t, last_v = applicable[-1]
        if last_v is True:
            committed_true.append((pid, p, last_t))
    if not committed_true:
        return []
    lines: List[str] = []
    for pid, p, t in committed_true[:max_lines]:
        desc = (getattr(p, "description", "") or "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
        snippet = f" \u2014 {desc}" if desc else ""
        lines.append(f"  - {pid} (true @ T={t}){snippet}")
    if len(committed_true) > max_lines:
        lines.append(f"  - ...and {len(committed_true) - max_lines} more.")
    return [ConstraintBlock(
        constraint_type="narrative",
        priority="hard",
        instruction=(
            f"=== TRUE PROPOSITIONS (HARD) === \u2014 the physics commits "
            f"these propositions TRUE at or before this scene's anchor "
            f"in the {world_label} world. The narration MUST honour them "
            "(do not stage them as not-yet-the-case, undecided, or "
            "contradicted). Characters may still *disbelieve* them \u2014 "
            "that mismatch is allowed and often drives dramatic irony \u2014 "
            "but the prose may not enact the opposite as fact.\n"
            + "\n".join(lines)
        ),
        evidence={"true_proposition_ids": [pid for pid, _, _ in committed_true]},
    )]


def build_world_invariant_constraints(
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int],
    *,
    world_label: str = "this",
    intensity_floor: float = 0.5,
    max_lines: int = 12,
) -> List[ConstraintBlock]:
    """HARD constraint enumerating WORLD_ traits at or above ``intensity_floor``.

    Walks ``world_state.world_traits`` and emits the reconstructed
    magnitude *at* ``syuzhet_anchor`` (consulting ``state_timeline``).
    Only traits whose reconstructed intensity meets the floor are
    surfaced \u2014 these are the load-bearing world constraints the
    renderer cannot quietly violate (e.g. surveillance state, magic
    system rules, wartime economy). Without this block a Rung-3
    counterfactual brief carries no positive evidence of the world
    laws the new scene must continue to satisfy (AUDIT P0).
    """
    world_traits = getattr(world_state, "world_traits", None) or {}
    if not world_traits:
        return []
    cap = syuzhet_anchor
    invariants: List[tuple] = []
    for wid, wt in world_traits.items():
        base = getattr(wt, "magnitude", None)
        base_value = float(getattr(base, "value", 0.0) or 0.0) if base else 0.0
        effective_value = base_value
        effective_t: Optional[int] = None
        timeline = getattr(wt, "state_timeline", None) or []
        if timeline and cap is not None:
            applicable = [
                snap for snap in timeline
                if getattr(snap, "fabula_time", None) is not None
                and int(snap.fabula_time) <= cap
                and getattr(snap, "magnitude", None) is not None
            ]
            if applicable:
                applicable.sort(key=lambda s: int(s.fabula_time))
                latest = applicable[-1]
                effective_value = float(getattr(latest.magnitude, "value", base_value) or base_value)
                effective_t = int(latest.fabula_time)
        elif timeline:
            # No anchor: take the last snapshot in declaration order.
            last_snap = timeline[-1]
            if getattr(last_snap, "magnitude", None) is not None:
                effective_value = float(getattr(last_snap.magnitude, "value", base_value) or base_value)
                effective_t = int(getattr(last_snap, "fabula_time", 0) or 0)
        if effective_value < intensity_floor:
            continue
        invariants.append((wid, wt, effective_value, effective_t))
    if not invariants:
        return []
    invariants.sort(key=lambda row: -row[2])
    lines: List[str] = []
    for wid, wt, val, t in invariants[:max_lines]:
        name = (getattr(wt, "name", "") or "").strip() or wid
        category = (getattr(wt, "category", "") or "").strip()
        cat_str = f" [{category}]" if category else ""
        when = f" @ T={t}" if t is not None else ""
        lines.append(f"  - {wid}: {name}{cat_str} (intensity={val:.2f}{when})")
    if len(invariants) > max_lines:
        lines.append(f"  - ...and {len(invariants) - max_lines} more.")
    return [ConstraintBlock(
        constraint_type="narrative",
        priority="hard",
        instruction=(
            f"=== WORLD INVARIANTS (HARD) === \u2014 the {world_label} world "
            f"carries these load-bearing global traits at or above "
            f"intensity {intensity_floor:.2f} at this scene's anchor. The "
            "narration must remain consistent with them (do not quietly "
            "soften, contradict, or write past them; character agency must "
            "still bend to their constraints).\n"
            + "\n".join(lines)
        ),
        evidence={
            "world_invariant_ids": [wid for wid, _, _, _ in invariants],
        },
    )]


def _latest_proposition_truth_map(
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int],
) -> Dict[str, Optional[bool]]:
    """Return ``{proposition_id: latest_committed_truth_or_None}``
    sliced to ``syuzhet_anchor``.

    ``None`` entries flag propositions whose latest commit is *open*
    at the anchor (no truth value yet); callers can treat them as
    "uncommitted" and skip pink-elephant emission.
    """
    out: Dict[str, Optional[bool]] = {}
    props = getattr(world_state, "propositions", None) or []
    if isinstance(props, dict):
        prop_iter = list(props.values())
    else:
        prop_iter = list(props)
    cap = syuzhet_anchor
    for p in prop_iter:
        pid = getattr(p, "id", None) or getattr(p, "proposition_id", None)
        if not pid:
            continue
        truth_map = getattr(p, "truth_at_fabula", {}) or {}
        if not truth_map:
            out[pid] = None
            continue
        applicable = [
            (int(t), bool(v))
            for t, v in truth_map.items()
            if cap is None or int(t) <= cap
        ]
        if not applicable:
            out[pid] = None
            continue
        applicable.sort(key=lambda kv: kv[0])
        out[pid] = applicable[-1][1]
    return out


def build_unrealised_concern_constraints(
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int],
    *,
    world_label: str = "this",
) -> List[ConstraintBlock]:
    """HARD pink-elephant block for concerns whose underlying
    proposition will NOT have come true at this scene's anchor.

    The renderer\u2019s temptation, given a richly motivated character,
    is to grant the protagonist their desire (or vindicate their
    fear) on-page \u2014 even when the physics records the underlying
    proposition as committed FALSE. Surfacing the concern explicitly,
    keyed to the holding entity, blocks that drift in both renderer
    and auditor: they see ``Macbeth desires PROP_BECOMES_KING (false
    at this anchor)`` and know not to stage the coronation.

    Pairs with :func:`build_false_proposition_constraints` (which
    targets the narration layer at the proposition level) by adding
    an entity-level reading: *which character* is the one who
    must NOT be shown getting their wish / fear realised here.
    Belief-side mismatches (the character still *believes* the
    proposition is true) remain allowed and feed dramatic irony.
    """
    truth_map = _latest_proposition_truth_map(world_state, syuzhet_anchor)
    if not truth_map:
        return []
    entities = getattr(world_state, "entities", None) or {}
    if isinstance(entities, dict):
        entity_iter = list(entities.values())
    else:
        entity_iter = list(entities)
    cap = syuzhet_anchor
    rows: List[tuple] = []  # (entity_id, entity_name, concern, prop_truth)
    for ent in entity_iter:
        concerns = getattr(ent, "concerns", None) or []
        if not concerns:
            continue
        eid = getattr(ent, "id", "?")
        ename = getattr(ent, "name", eid)
        for c in concerns:
            pid = getattr(c, "proposition_id", None)
            if not pid or pid not in truth_map:
                continue
            latest = truth_map[pid]
            if latest is not False:
                continue  # only emit when proposition is committed FALSE
            # P0-FIX (P0-14): Window-gate concerns by activation_fabula_window (CRITICAL-006 audit).
            # Filter concerns outside their activation window to prevent temporal leakage
            # (e.g., Lear's irrelevance concern active before abdication event).
            window = getattr(c, "activation_fabula_window", None)
            if window and cap is not None:
                try:
                    start, end = int(window[0]), int(window[1])
                except (ValueError, TypeError, IndexError):
                    start, end = None, None
                if start is not None and cap < start:
                    continue
                if end is not None and cap > end:
                    continue
            rows.append((eid, ename, c, latest))
    if not rows:
        return []
    # Stable, salience-descending order so the highest-stakes
    # pink-elephants land first inside the prompt cap.
    rows.sort(key=lambda r: float(getattr(r[2], "salience", 0.0) or 0.0),
              reverse=True)
    lines: List[str] = []
    for eid, ename, c, _truth in rows[:20]:
        cid = getattr(c, "concern_id", "?")
        polarity = getattr(c, "polarity", "?")
        pid = getattr(c, "proposition_id", "?")
        kind = getattr(c, "kind", None) or ""
        kind_blob = f" [{kind}]" if kind else ""
        verb = "fulfilled" if polarity == "desire" else "realised"
        lines.append(
            f"  - {ename} ({eid}) {polarity}s {pid}{kind_blob} "
            f"\u2014 do NOT show this concern {verb} ({cid})"
        )
    if len(rows) > 20:
        lines.append(f"  - ...and {len(rows) - 20} more.")
    return [ConstraintBlock(
        constraint_type="narrative",
        priority="hard",
        instruction=(
            f"=== UNREALISED CONCERNS (HARD) === \u2014 these standing "
            f"desires / fears are tied to propositions the physics "
            f"commits FALSE at or before this scene's anchor in the "
            f"{world_label} world. Do NOT stage the desire as "
            "fulfilled, the fear as realised, or otherwise grant the "
            "holder the proposition's content on-page. The character "
            "may still *feel* the concern \u2014 longing, dread, "
            "anticipation \u2014 but the proposition itself must remain "
            "unrealised in the narration.\n"
            + "\n".join(lines)
        ),
        evidence={
            "unrealised_concern_ids": [
                getattr(c, "concern_id", "?") for _, _, c, _ in rows
            ],
        },
    )]


def build_false_belief_grounding_constraints(
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int],
    *,
    world_label: str = "this",
) -> List[ConstraintBlock]:
    """HARD pink-elephant block for beliefs whose linked proposition
    is committed FALSE at the anchor.

    Pairs with :func:`build_false_proposition_constraints` from the
    *belief* side: enumerates the entities who currently hold a
    proposition-linked belief whose proposition is false. The
    renderer must keep these as *believed* (interior life, dialogue
    presupposition, biased perception) but never as ground truth in
    narration. Without this entity-anchored reading, the renderer
    routinely \u201cresolves\u201d a sympathetic believer's stance into
    fact by accident.
    """
    truth_map = _latest_proposition_truth_map(world_state, syuzhet_anchor)
    if not truth_map:
        return []
    entities = getattr(world_state, "entities", None) or {}
    if isinstance(entities, dict):
        entity_iter = list(entities.values())
    else:
        entity_iter = list(entities)
    cap = syuzhet_anchor
    rows: List[tuple] = []  # (entity_id, entity_name, belief)
    for ent in entity_iter:
        beliefs = getattr(ent, "beliefs", None) or []
        if not beliefs:
            continue
        eid = getattr(ent, "id", "?")
        ename = getattr(ent, "name", eid)
        for b in beliefs:
            pid = getattr(b, "proposition_id", None)
            if not pid or pid not in truth_map:
                continue
            if truth_map[pid] is not False:
                continue
            # Time-gate: only beliefs already established at the anchor
            est = getattr(b, "established_at_fabula", None)
            if cap is not None and est is not None and int(est) > cap:
                continue
            rows.append((eid, ename, b))
    if not rows:
        return []
    rows.sort(key=lambda r: float(getattr(r[2], "confidence", 0.0) or 0.0),
              reverse=True)
    lines: List[str] = []
    for eid, ename, b in rows[:20]:
        pid = getattr(b, "proposition_id", "?")
        perc = (getattr(b, "perceived_state", "") or "").strip()
        if len(perc) > 100:
            perc = perc[:97] + "..."
        conf = float(getattr(b, "confidence", 0.0) or 0.0)
        perc_blob = f" \u2014 \"{perc}\"" if perc else ""
        lines.append(
            f"  - {ename} ({eid}) believes {pid} (conf={conf:.2f}){perc_blob}"
        )
    if len(rows) > 20:
        lines.append(f"  - ...and {len(rows) - 20} more.")
    return [ConstraintBlock(
        constraint_type="narrative",
        priority="hard",
        instruction=(
            f"=== FALSE-BELIEF GROUNDING (HARD) === \u2014 these entities "
            f"hold beliefs whose linked proposition is committed FALSE "
            f"at or before this scene's anchor in the {world_label} "
            f"world. Render the belief as *believed* (interior "
            "thought, biased dialogue, presupposed action) \u2014 NEVER "
            "as ground-truth narration. The narrator's voice must not "
            "endorse the believed content; the gap between belief and "
            "fact is exactly what generates dramatic irony / "
            "tragic-error / mistaken-identity beats and must be "
            "preserved on the page.\n"
            + "\n".join(lines)
        ),
        evidence={
            "false_belief_entity_ids": sorted({eid for eid, _, _ in rows}),
        },
    )]


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
    def compute_epistemic_gaps(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> List[EpistemicGap]:
        """Compare each entity's beliefs against the objective graph state.

        AUDIT (post-2026-05-26): when ``syuzhet_anchor`` is supplied,
        beliefs whose ``established_at_fabula`` is *strictly after* the
        anchor's fabula cut-off are filtered out. Treating future
        beliefs as already-held at the reader's current position
        contaminated dramatic-irony and mystery directives — the brief
        would compute gaps against knowledge the entity hasn't acquired
        yet at this point in the syuzhet.
        """
        gaps: List[EpistemicGap] = []
        anchor_t = self._syuzhet_anchor_to_fabula_time(syuzhet_anchor)

        for eid in entity_ids:
            ent_data = self._find_entity(eid)
            if not ent_data:
                continue

            for belief in ent_data.get("beliefs", []):
                if anchor_t is not None:
                    est = belief.get("established_at_fabula")
                    if est is not None and est > anchor_t:
                        continue
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
    def _syuzhet_anchor_to_fabula_time(
        self, syuzhet_anchor: Optional[int],
    ) -> Optional[int]:
        """Translate a syuzhet anchor into a fabula_time cut-off.

        Returns the maximum ``fabula_time`` among events whose
        ``syuzhet_index`` is ``<= syuzhet_anchor``. ``None`` if the
        anchor is ``None`` or no event qualifies, signalling
        "use latest available state" to callers.
        """
        if syuzhet_anchor is None:
            return None
        revealed_t = [
            e.fabula_time for e in self.world_state.events
            if e.syuzhet_index is not None
            and e.syuzhet_index <= syuzhet_anchor
            and e.fabula_time is not None
        ]
        return max(revealed_t) if revealed_t else None

    def compute_trait_trajectories(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> List[TraitTrajectory]:
        """Compute current value + headroom for each trait of given entities.

        When ``syuzhet_anchor`` is supplied, trait values are
        reconstructed from each entity's ``state_timeline`` at the
        fabula_time of the latest revealed event — so emotion-axis
        affective curves vary across the syuzhet (matching Reagan
        et al. 2016 / Vonnegut "shapes of stories" temporal arcs).
        When the anchor is ``None`` we keep the legacy behaviour:
        ego payload first (live, per-directive overrides), then a
        fallback to the entity's current ``traits`` map.

        The legacy fallback path is essential for any caller that
        constructs a ``DirectiveAssembler`` with an empty ego (the
        affective-curve plot, the audit script, the auditor's
        per-target rescore loop). Without it the emotion scorers
        silently returned ``+1.0`` (worst possible match) for every
        entity, since the trajectory list was empty and the
        no-contributions branch fired.
        """
        trajectories: List[TraitTrajectory] = []
        anchor_t = self._syuzhet_anchor_to_fabula_time(syuzhet_anchor)

        for eid in entity_ids:
            # When a syuzhet anchor is supplied, prefer the temporal
            # reconstruction over the (time-invariant) ego snapshot —
            # otherwise emotion curves collapse to flat lines.
            if anchor_t is not None:
                ent = self.world_state.entities.get(eid)
                if ent is None or not getattr(ent, "traits", None):
                    continue
                snap = reconstruct_entity_at(ent, anchor_t)
                trait_iter = snap.get("traits", {}).items()
                resolve = lambda t, k, d: t.get(k, d)
            else:
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
    # WORLD_ trait shift computation (rendering directive layer)
    # ------------------------------------------------------------------
    def compute_world_trait_shifts(
        self,
        syuzhet_anchor: Optional[int] = None,
        max_shifts: int = 6,
    ) -> List["WorldTraitShift"]:
        """Surface recent WORLD_ trait shifts at or before the brief's
        anchor.

        For each ``GlobalTrait`` we reconstruct its magnitude at the
        anchor's fabula_time (or the latest snapshot when the anchor
        is ``None``) and compare against the immediately-preceding
        snapshot. Traits with no movement are skipped. Returned list
        is sorted by absolute delta descending and capped at
        ``max_shifts`` so the renderer foregrounds the largest
        recent regime changes first.
        """
        anchor_t = self._syuzhet_anchor_to_fabula_time(syuzhet_anchor)
        shifts: List[WorldTraitShift] = []
        for wt_id, wt in (self.world_state.world_traits or {}).items():
            timeline = sorted(
                getattr(wt, "state_timeline", []) or [],
                key=lambda s: s.fabula_time,
            )
            if not timeline:
                continue
            cutoff = anchor_t if anchor_t is not None else max(s.fabula_time for s in timeline)
            visible = [s for s in timeline if s.fabula_time <= cutoff]
            if not visible:
                continue
            latest = visible[-1]
            if latest.magnitude is None:
                continue
            current = float(latest.magnitude.value)
            inertia = float(latest.magnitude.inertia)
            # Previous value: the snapshot before ``latest``, falling
            # back to the trait's baseline magnitude.
            if len(visible) >= 2 and visible[-2].magnitude is not None:
                previous = float(visible[-2].magnitude.value)
            elif wt.magnitude is not None:
                previous = float(wt.magnitude.value)
            else:
                previous = current
            delta = current - previous
            if abs(delta) < 1e-6:
                continue
            shifts.append(WorldTraitShift(
                trait_id=wt_id,
                trait_name=getattr(wt, "name", wt_id),
                previous_value=previous,
                current_value=current,
                delta=delta,
                inertia=inertia,
                affected_domains=list(getattr(wt, "affected_domains", []) or []),
                fabula_time=int(latest.fabula_time),
                triggered_by=getattr(latest, "triggered_by", None),
                proposition_id=getattr(wt, "proposition_id", None),
                description=getattr(latest, "description", None),
            ))
        shifts.sort(key=lambda s: abs(s.delta), reverse=True)
        return shifts[:max_shifts]

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

        # Derive an effective fabula upper bound from the revealed
        # syuzhet window so terminated channels are excluded once
        # the source text has reached / passed their termination tick.
        # ``fabula_anchor`` = max fabula_time among events with
        # ``syuzhet_index <= syuzhet_anchor``; ``None`` when no
        # event has been revealed yet (fall back to including all
        # channels regardless of termination).
        fabula_anchor: Optional[int] = None
        for evt in self.world_state.events:
            if evt.syuzhet_index > syuzhet_anchor:
                continue
            if fabula_anchor is None or evt.fabula_time > fabula_anchor:
                fabula_anchor = evt.fabula_time

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
            # Time-slice: skip channels that have already been
            # severed at this anchor. A dead channel cannot host
            # future utterances, so surfacing it as hidden
            # capability is misleading to the directive renderer.
            if (
                ch.terminated_at_fabula is not None
                and fabula_anchor is not None
                and ch.terminated_at_fabula <= fabula_anchor
            ):
                continue
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

    # When set on the instance, overrides the syuzhet-anchored reveal
    # set with a *fabula-anchored* one. Used by the fabula-sweep
    # timeseries view: a single ``syuzhet_anchor`` cannot represent
    # ``events with fabula_time \u2264 t`` when the author placed flashbacks
    # (early-fabula events at late-syuzhet indices), so the sweep would
    # incorrectly mark the whole story as "revealed" the moment the
    # snapshot first contains any flashback target. Setting this
    # attribute swaps the reveal predicate to ``fabula_time \u2264 fab``,
    # which matches the physical-resolution semantics the sweep wants.
    _fabula_anchor_override: Optional[int] = None

    def _revealed_event_ids(self, syuzhet_anchor: Optional[int]) -> set[str]:
        """Return IDs of events the reader has seen by *syuzhet_anchor*."""
        fab = getattr(self, "_fabula_anchor_override", None)
        if fab is not None:
            return {
                e.id for e in self.world_state.events
                if e.fabula_time is not None and e.fabula_time <= fab
            }
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
            """Backward-looking decay weight on an effect's contribution.

            Carroll 1990 *erotetic narrative*: live questions are the
            ones the story is currently asking. ``exp(-Δ/τ)`` decays
            the curiosity weight of effects whose syuzhet position is
            far behind the anchor — distinguishes *live mystery*
            (whodunit imminent) from *background mystery* (subplots
            trailing). Effects without a syuzhet position (entity-
            as-effect) get a neutral 1.0.
            """
            if syuzhet_anchor is None:
                return 1.0
            evt = events_by_id.get(eff_id)
            if evt is None:
                return 1.0  # entity-as-effect (no syuzhet position)
            delta = max(0, syuzhet_anchor - evt.syuzhet_index)
            return math.exp(-delta / self._MYSTERY_PROXIMITY_TAU_SYUZHET)

        # Sayers 1929 / Knox 1929 detective-fiction false-lead term.
        # Detective-genre theory says mystery intensity tracks not
        # only the *number* of unrevealed causes but the *number of
        # plausible-but-wrong* hypotheses the audience is actively
        # entertaining. We approximate this as the count of audience
        # belief assignments on event-shaped propositions whose
        # confidence sits in the ambiguity band (0.3–0.7) — the
        # reader has a hypothesis but isn't sure of it. Computed once
        # outside the per-effect loop and added as a bonus into the
        # final ratio so red-herring-rich worlds (Gone Girl, Death on
        # the Nile) read as more mysterious than worlds with the
        # same hidden-ancestor count but no plausible alternates.
        false_lead_count = 0
        try:
            from shadow_loom.affect_unification import (
                AUDIENCE_ID,
                BeliefState,
                synthesise_audience_entity,
                synthesise_propositions,
                backfill_character_belief_propositions,
            )
            ws_m = self.world_state
            if not ws_m.propositions:
                synthesise_propositions(ws_m)
                backfill_character_belief_propositions(ws_m)
            if AUDIENCE_ID not in ws_m.entities:
                synthesise_audience_entity(ws_m)
            bs_m = BeliefState(world=ws_m)
            ft_now_m = self._fabula_now(revealed)
            if ft_now_m is not None:
                for prop in ws_m.propositions:
                    # Mirror ``compute_suspense_unified`` — multi-flip
                    # propositions with a future commit are still open
                    # even if a prior commit exists. Only skip when
                    # every commit is at/before the cursor.
                    future_commits_m = [
                        t for t in prop.truth_at_fabula if t > ft_now_m
                    ]
                    if not future_commits_m and any(
                        t <= ft_now_m for t in prop.truth_at_fabula
                    ):
                        continue
                    p = bs_m.confidence(
                        AUDIENCE_ID, prop.proposition_id, ft_now_m,
                    )
                    if 0.3 <= p <= 0.7:
                        false_lead_count += 1
        except Exception:
            logger.debug("mystery false-lead pass failed", exc_info=True)

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
            base_score = 0.0
        else:
            base_score = hidden_mass / total_mass

        # Sayers/Knox false-lead boost: blend a saturating curve on
        # the live-hypothesis count into the base score. ``count /
        # (count + K)`` saturates at 1 with K=4 → 4 plausible alts
        # ⇒ 0.5 false-lead boost; 8 alts ⇒ 0.67. Convex-combined
        # with the base hidden-ratio at weight 0.25 so the gauge is
        # still dominated by structural hidden mass but red-herring-
        # rich fixtures (Gone Girl, Death on the Nile) lift visibly.
        if false_lead_count > 0:
            fl = false_lead_count / (false_lead_count + 4.0)
            score = 0.75 * base_score + 0.25 * fl
        else:
            score = base_score
        score = max(0.0, min(1.0, score))

        logger.debug(
            "[DirectiveAssembly·Mystery] hidden_mass=%.3f / total_mass=%.3f "
            "base=%.3f false_leads=%d → %.3f "
            "(path-decay depth=%d, τ_curiosity=%.1f)",
            hidden_mass, total_mass, base_score, false_lead_count, score,
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
        if syuzhet_anchor is None and getattr(
            self, "_fabula_anchor_override", None
        ) is None:
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
                # Bound character knowledge by the reveal-set rather
                # than a raw ``syuzhet_index`` compare, so the fabula
                # sweep (which leaves ``syuzhet_anchor`` as ``None`` and
                # uses ``_fabula_anchor_override`` instead) still gates
                # the character's utterance knowledge correctly.
                if utt.id not in revealed:
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
                # Closure proximity is a syuzhet-axis aesthetic
                # (distance between the reader's current reading
                # position and the future discourse position where the
                # focal will hear the truth). Under a fabula sweep
                # ``syuzhet_anchor`` is ``None`` and the concept does
                # not transfer — fall back to the standing floor so the
                # gap still contributes the base irony surface.
                if syuzhet_anchor is None:
                    return self._IRONY_PROXIMITY_FLOOR
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
                # Tan 1996 audience-concern gating (parity with
                # suspense). Dramatic irony about a character the
                # audience cares about lands harder than irony about
                # an indifferent stranger — the felt gap requires the
                # reader to be invested in the blind party's fate.
                w *= self._audience_concern_for_event(evt)
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
    # bored), so K_kind is small. Halved from the original
    # calibration (4.0/3.0/2.5/2.0/2.0/1.5/1.5/1.5) after broadening
    # the unified suspense filter to all uncommitted propositions —
    # the previous K's were tuned for a much larger ledger and were
    # crushing the score in the new regime.
    _SUSPENSE_STAKES_K_BY_KIND: Dict[str, float] = {
        "existential": 2.0,
        "physical": 1.5,
        "betrayal": 1.25,
        "psychological": 1.0,
        "emotional": 1.0,
        "social": 0.75,
        "epistemic": 0.75,
        "informational": 0.75,
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
    # Default fallback when the spatial graph is missing, trivially
    # small, or its diameter cannot be computed (disconnected with
    # only tiny components). Improvement B6 below replaces this
    # constant with an auto-scaled τ at runtime when the graph
    # supports it.
    _SUSPENSE_PROXIMITY_TAU_SPATIAL: float = 4.0
    # Improvement B6: target fraction of the spatial-graph diameter
    # at which the imminence kernel decays to 1/e ≈ 0.37. A τ_spatial
    # of ~30% of the diameter means a threat half-way across the world
    # already sits at ~e^(-1.7) ≈ 0.18 of full weight, while one
    # adjacent room over decays only marginally — matching the
    # narrative intuition that "across the keep" is far and "next
    # chamber" is close, regardless of whether the world has 6 or
    # 60 locations.
    _SUSPENSE_PROXIMITY_DIAMETER_FRACTION: float = 0.30

    # Improvement B7: probability prior for events with NO incident
    # causal edge in either direction. The previous default of 0.5
    # equated "explicitly modelled coin-flip" with "totally orphan
    # event", flattening the gauge. 0.25 is a deliberately
    # conservative prior — orphan events still register on the
    # ledger (so a sparse extraction doesn't zero out suspense), but
    # they no longer dominate over events with real causal weight.
    _SUSPENSE_ORPHAN_EVENT_PROB: float = 0.25

    # Persistence multiplier (improvement #4 — Brewer & Lichtenstein
    # 1982 initiating-event arc). The longer a foreshadowed threat
    # lingers unresolved, the louder it gets. We measure persistence
    # as the count of *revealed* causal ancestors (proxy for "how long
    # the gun has been on the mantle"). Multiplier =
    # min(cap, 1 + α·persistence).
    _SUSPENSE_PERSISTENCE_ALPHA: float = 0.10
    _SUSPENSE_PERSISTENCE_CAP: float = 1.5

    # Improvement B9: imminence damping by *unrevealed* causal
    # ancestors. An unrevealed event whose own setup is still pending
    # ("the murderer is mid-monologue, exposition still coming")
    # should feel less imminent than one whose setup is already on the
    # page ("the murderer is at the door now"). Imminence is divided
    # by ``1 + β · unrevealed_ancestors_count``. Small β so this is a
    # gentle delay, not a hard gate.
    _SUSPENSE_REMAINING_SETUP_BETA: float = 0.15

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

    # Soft penalty for a one-sided forward reveal set (only threats
    # coming → despair, or only hopes coming → safety). Strict
    # Brewer-Lichtenstein 1982 zeroes both, but Carroll 1990
    # (anomalous suspense) and Gerrig 1989 (pre-known outcomes still
    # arouse) give a robust empirical floor: readers feel something
    # on one-sided futures, just less than on balanced ones. ×0.3
    # preserves the theoretical ranking without the visual cliff at
    # the start/end of every story.
    _SUSPENSE_ONE_SIDED_MULT: float = 0.3

    # Convex blend weight for compute_suspense_score(mode='efk'):
    # final = w · efk + (1-w) · unified_entropy. EFK is expected
    # belief variance over kind ledgers (Ely-Frankel-Kamenica);
    # unified is current outcome-set entropy over uncommitted
    # propositions (Brewer-Lichtenstein audience-side substrate).
    # 0.6/0.4 mirrors the surprise scorer's blend and keeps the
    # legacy EFK contract dominant while adding a steady proposition-
    # entropy channel that doesn't collapse at the structural endpoints.
    _SUSPENSE_EFK_BLEND_WEIGHT: float = 0.6

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

    def _audience_concern_for_event(self, evt) -> float:
        """Tan 1996 F-emotion concern weight for a candidate event.

        Sum of audience-concern saliences referencing any proposition
        whose ``referent_ids`` intersect this event's actor/target set,
        plus a small floor so events with no concern hookup still
        register on the ledger. Returns a multiplier in roughly
        ``[0.2, 1.0]`` — concern dominates but never zeroes out.

        Brewer-Lichtenstein outcome ambiguity tells us *whether* a
        future event is suspenseful; Tan 1996 *Emotion and the
        Structure of Narrative Film* tells us *which* outcomes the
        reader cares about. Without this gate, "two non-focal NPCs
        having an argument" weighs as much as "the protagonist on
        trial for their life" — both have similar belief-variance,
        but the audience only feels suspense about the second.
        """
        from shadow_loom.affect_unification import AUDIENCE_ID
        audience = self.world_state.entities.get(AUDIENCE_ID)
        if audience is None or not audience.concerns:
            return 1.0  # no audience model → neutral, don't penalise
        evt_participants = set(evt.actor_ids) | set(evt.target_ids)
        if not evt_participants:
            return 0.5
        # Index propositions by their referent set for O(1) lookup.
        prop_index = {p.proposition_id: p for p in self.world_state.propositions}
        total = 0.0
        for c in audience.concerns:
            prop = prop_index.get(c.proposition_id)
            if prop is None:
                continue
            if not (set(prop.referent_ids) & evt_participants):
                continue
            total += float(c.salience)
        # Map concern total onto a [0.2, 1.0] multiplier — saturate
        # so a single high-salience concern is enough to hit the
        # ceiling, but events with no concern hookup still contribute
        # 20% so the gauge degrades gracefully on sparse fixtures.
        floor = 0.2
        return floor + (1.0 - floor) * (total / (total + 1.0))

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

    def _spatial_tau(self, spatial_g: Optional[nx.Graph]) -> float:
        """Auto-scaled τ for the spatial-imminence kernel (B6).

        Returns ``_SUSPENSE_PROXIMITY_DIAMETER_FRACTION × diameter``
        of the largest connected component (clamped to ≥ 2.0 hops so
        very small graphs still admit some near/far gradient).
        Falls back to the constant ``_SUSPENSE_PROXIMITY_TAU_SPATIAL``
        when no graph is available or its diameter is unmeasurable.
        """
        if spatial_g is None or spatial_g.number_of_nodes() < 2:
            return self._SUSPENSE_PROXIMITY_TAU_SPATIAL
        try:
            comps = list(nx.connected_components(spatial_g))
            if not comps:
                return self._SUSPENSE_PROXIMITY_TAU_SPATIAL
            biggest = max(comps, key=len)
            sub = spatial_g.subgraph(biggest)
            if sub.number_of_nodes() < 2:
                return self._SUSPENSE_PROXIMITY_TAU_SPATIAL
            d = nx.diameter(sub)
        except (nx.NetworkXError, nx.NetworkXNoPath, ValueError):
            return self._SUSPENSE_PROXIMITY_TAU_SPATIAL
        if d <= 0:
            return self._SUSPENSE_PROXIMITY_TAU_SPATIAL
        return max(2.0, d * self._SUSPENSE_PROXIMITY_DIAMETER_FRACTION)

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

    def _unrevealed_ancestor_count(
        self, evt_id: str, causal_g: nx.DiGraph, revealed: set[str],
    ) -> int:
        """Count *unrevealed* causal ancestors of ``evt_id`` (B9).

        Proxy for "how much setup is still pending before this event
        can fire". Used to damp imminence so a future event whose own
        causes have not yet appeared on the page feels less proximal
        than one whose setup is already revealed."""
        if not causal_g.has_node(evt_id):
            return 0
        try:
            ancestors = nx.ancestors(causal_g, evt_id)
        except (nx.NetworkXError, nx.NodeNotFound):
            return 0
        return sum(1 for a in ancestors if a not in revealed)

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
        # Improvement B7: distinguish "explicit coin-flip" from
        # "no causal model at all". Orphan events fall back to
        # ``_SUSPENSE_ORPHAN_EVENT_PROB`` (default 0.25) instead of
        # the neutral 0.5 so they no longer over-weight against
        # events with real edge evidence.
        prob = self._SUSPENSE_ORPHAN_EVENT_PROB
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

        ``mode='unified'`` delegates to
        :func:`shadow_loom.affect_unification.compute_suspense_unified`,
        which derives suspense as audience outcome-set entropy ×
        stakes × imminence over the shared ``Proposition`` /
        ``ENT_AUDIENCE`` substrate (see
        ``/memories/repo/affect-unification-plan.md``). Provided as a
        parity-flag for cross-validation against the legacy aggregator
        before the substrate replaces it; the result is normalised
        into [0, 1] by dividing by ``len(open_outcomes)·ln 2``.
        """
        if mode == "unified":
            from shadow_loom.affect_unification import (
                AUDIENCE_ID,
                BeliefState,
                compute_suspense_unified,
                synthesise_audience_entity,
                synthesise_propositions,
                backfill_character_belief_propositions,
                _binary_entropy,
                _auto_tau_fabula as _auto_tau_fabula_unified,
            )
            w = self.world_state
            if not w.propositions:
                synthesise_propositions(w)
                backfill_character_belief_propositions(w)
            if AUDIENCE_ID not in w.entities:
                synthesise_audience_entity(w)
            bs = BeliefState(world=w)
            revealed = self._revealed_event_ids(syuzhet_anchor)
            ft_now = self._fabula_now(revealed)
            if ft_now is None:
                return 0.0

            # ---- Enriched unified suspense (Step 7-prep) ----
            # Wrap the audience-entropy substrate with a subset of
            # the legacy EFK envelope: per-kind harm salience,
            # per-kind saturation curve, and persistence multiplier.
            # The substrate (audience entropy on open outcome
            # propositions) is the unified scorer's actual claim;
            # the envelope shapes the per-kind aggregation so the
            # output is on the same scale as the legacy 'efk' path
            # for cross-validation. No focal disposition gating —
            # that proved to suppress every world's signal in the
            # parity sweep.
            causal_g = self._build_causal_digraph()
            events_by_id = {e.id: e for e in w.events}

            tau_fabula = _auto_tau_fabula_unified(w)

            # Aggregate per-kind contribution. Each open outcome
            # proposition contributes
            #   H(p_aud) · stakes · salience · imminence · persistence
            # to its harm-kind ledger. We then apply the same per-
            # kind saturation curve as the legacy scorer
            #   stakes_k = total_k / (total_k + K_k)
            # and combine across kinds with a salience-weighted
            # average. This keeps the audience-entropy substrate
            # (the unified scorer's actual claim) but borrows the
            # legacy envelope so the two paths can be cross-
            # validated on like terms.
            per_kind_total: Dict[str, float] = {}

            for prop in w.propositions:
                # Iterate ALL propositions, not just ``outcome`` —
                # the legacy scorer treats every unrevealed event as
                # a candidate threat/hope, and ``_kind_for_event``
                # only assigns ``outcome`` to ``choice`` events. The
                # entropy substrate naturally collapses to zero for
                # propositions whose audience confidence is already
                # certain, so non-outcome props that are still
                # uncertain still contribute meaningfully.
                if not prop.referent_ids:
                    continue
                evt_id = prop.referent_ids[0]
                # Skip propositions the reader has already encountered
                # in syuzhet (the audience knows the outcome). This
                # matches the legacy scorer's ``unrevealed`` loop and
                # avoids zero-entropy noise from already-revealed
                # events whose audience confidence has collapsed to
                # 1.0. Note: events whose fabula_time is in the past
                # but whose syuzhet step is still upcoming (mystery
                # backstory) are intentionally INCLUDED — those are
                # exactly the suspenseful "what happened" beats.
                if evt_id in revealed:
                    continue
                evt = events_by_id.get(evt_id)
                if evt is None:
                    continue

                p_aud = bs.confidence(
                    AUDIENCE_ID, prop.proposition_id, ft_now,
                )
                h = _binary_entropy(p_aud)
                if h <= 0.0:
                    continue

                future_commits = [
                    t for t in prop.truth_at_fabula if t > ft_now
                ]
                if future_commits:
                    dt = min(future_commits) - ft_now
                    imminence_t = math.exp(-dt / max(1.0, tau_fabula))
                else:
                    # Event already happened in fabula but is still
                    # unrevealed in syuzhet (backstory mystery).
                    # Treat as fully imminent — the reveal could
                    # land any moment.
                    imminence_t = 1.0

                ancestors_revealed = self._revealed_ancestor_count(
                    evt_id, causal_g, revealed,
                )
                persistence_mult = min(
                    self._SUSPENSE_PERSISTENCE_CAP,
                    1.0
                    + self._SUSPENSE_PERSISTENCE_ALPHA
                    * ancestors_revealed,
                )

                kind, salience = self._harm_kind_for_event(
                    evt_id, causal_g,
                )

                contribution = (
                    h * prop.stakes * salience * imminence_t
                    * persistence_mult
                    * self._audience_concern_for_event(evt)
                )
                if contribution <= 0.0:
                    continue
                per_kind_total[kind] = (
                    per_kind_total.get(kind, 0.0) + contribution
                )

            if not per_kind_total:
                return 0.0

            num = 0.0
            denom = 0.0
            for kind, tot_k in per_kind_total.items():
                K_k = self._SUSPENSE_STAKES_K_BY_KIND.get(
                    kind, self._SUSPENSE_STAKES_K,
                )
                stakes_k = tot_k / (tot_k + K_k)
                sigma_k = self._HARM_KIND_SALIENCE.get(
                    kind, self._DEFAULT_HARM_SALIENCE,
                )
                num += sigma_k * stakes_k
                denom += sigma_k

            if denom <= 0.0:
                return 0.0
            return max(0.0, min(1.0, num / denom))

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
        # B6: auto-scale spatial decay to the spatial graph's actual
        # diameter so a 4-hop decay doesn't zero out a 60-location
        # world or under-decay a 6-room chamber piece.
        tau_spatial = self._spatial_tau(spatial_g)

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
            # B7: orphan events fall back to a deliberately
            # conservative prior so they don't out-weigh events with
            # actual edge evidence.
            prob = self._SUSPENSE_ORPHAN_EVENT_PROB
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

            # B9: damp imminence by *unrevealed* causal ancestors so a
            # future event whose own setup is still pending feels less
            # proximal than one whose causes are already on the page.
            ancestors_unrevealed = self._unrevealed_ancestor_count(
                evt_id, causal_g, revealed,
            )
            imminence_t /= (
                1.0 + self._SUSPENSE_REMAINING_SETUP_BETA
                * ancestors_unrevealed
            )

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
                        -d_sp / tau_spatial,
                    )

                weighted_prob = (
                    prob * salience * imminence_t * imminence_s
                    * persistence_mult
                    * self._audience_concern_for_event(evt)
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
                # structural-affect floor): the *upcoming* reveal
                # set ideally contains both threat and hope
                # candidates on this (focal, kind). Strict
                # Brewer-Lichtenstein zeroes one-sided futures, but
                # Carroll 1990 (anomalous suspense) and Gerrig 1989
                # (pre-known outcome physiological arousal) show
                # readers still feel dread on despair beats and
                # relieved tension on safety beats. We soften the
                # guard to a multiplier (×0.3) instead of a hard
                # skip, preserving the theoretical ranking
                # (balanced > one-sided) without the visual cliff
                # at the start/end of every story.
                unrev_threat = sum(w for b, w, _ in unrev if b == "threat")
                unrev_hope = sum(w for b, w, _ in unrev if b == "hope")
                one_sided_mult = 1.0
                if unrev_threat <= 0.0 or unrev_hope <= 0.0:
                    one_sided_mult = self._SUSPENSE_ONE_SIDED_MULT
                    if one_sided_mult <= 0.0:
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
                weight_fk = sigma_k * stakes_k * one_sided_mult
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
                # Blend in the unified audience-entropy substrate
                # (compute_suspense_unified, normalised by the
                # uncommitted-proposition count × ln 2). EFK
                # measures *expected* belief variance across kind
                # ledgers; unified measures *current* outcome-set
                # uncertainty over propositions. They're the same
                # theoretical quantity through different lenses
                # (Brewer & Lichtenstein outcome ambiguity → EFK
                # Bayesian variance), so a convex blend gives the
                # gauge a steadier signal — EFK alone goes to 0
                # at the structural endpoints (start/end of story)
                # even when there's still proposition entropy in
                # play. Default 0.6 EFK / 0.4 unified mirrors the
                # surprise scorer's blend.
                unified_score = 0.0
                try:
                    from shadow_loom.affect_unification import (
                        AUDIENCE_ID,
                        BeliefState,
                        compute_suspense_unified,
                        synthesise_audience_entity,
                        synthesise_propositions,
                        backfill_character_belief_propositions,
                    )
                    ws_u = self.world_state
                    if not ws_u.propositions:
                        synthesise_propositions(ws_u)
                        backfill_character_belief_propositions(ws_u)
                    if AUDIENCE_ID not in ws_u.entities:
                        synthesise_audience_entity(ws_u)
                    bs_u = BeliefState(world=ws_u)
                    revealed_u = self._revealed_event_ids(syuzhet_anchor)
                    ft_u = self._fabula_now(revealed_u)
                    if ft_u is not None and ws_u.propositions:
                        raw_u = compute_suspense_unified(bs_u, ft_u)
                        # Count uncommitted candidate propositions
                        # to normalise into [0, 1] — the unified
                        # sum scales with the open-proposition
                        # count, so divide by N · ln 2.
                        # "Open" mirrors ``compute_suspense_unified``:
                        # any future commit (t > cursor) keeps the
                        # proposition open even if there's a prior
                        # commit (multi-flip propositions).
                        n_open = sum(
                            1 for prop in ws_u.propositions
                            if any(
                                t > ft_u for t in prop.truth_at_fabula
                            ) or not any(
                                t <= ft_u for t in prop.truth_at_fabula
                            )
                        )
                        if n_open > 0:
                            unified_score = min(
                                1.0, raw_u / (n_open * math.log(2.0)),
                            )
                except Exception:
                    logger.debug(
                        "unified suspense blend failed", exc_info=True,
                    )
                w_efk = self._SUSPENSE_EFK_BLEND_WEIGHT
                blended = w_efk * score_efk + (1.0 - w_efk) * unified_score
                blended = max(0.0, min(1.0, blended))
                dom_kind = max(per_kind_var.items(), key=lambda kv: kv[1])[0] \
                    if per_kind_var else None
                logger.debug(
                    "[DirectiveAssembly·Suspense·EFK·full] efk=%.3f "
                    "unified=%.3f → blended=%.3f dominant_kind=%s "
                    "per_kind_var=%s threat_by_kind=%s hope_by_kind=%s",
                    score_efk, unified_score, blended, dom_kind,
                    {k: round(v, 4) for k, v in per_kind_var.items()},
                    {k: round(v, 3) for k, v in threat_by_kind.items()},
                    {k: round(v, 3) for k, v in hope_by_kind.items()},
                )
                return round(blended, 4)

        # Despair/safety guards: classic-only. EFK already handled
        # one-sided futures via ``_SUSPENSE_ONE_SIDED_MULT``; the
        # classic aggregator gets the same soft penalty here so
        # both modes share the Carroll/Gerrig empirical floor.
        classic_mult = 1.0
        if hope_weight <= 0.0 or threat_weight <= 0.0:
            classic_mult = self._SUSPENSE_ONE_SIDED_MULT
            if classic_mult <= 0.0:
                logger.debug(
                    "[DirectiveAssembly·Suspense] one-sided future "
                    "(threat=%.3f hope=%.3f) → suspense=0",
                    threat_weight, hope_weight,
                )
                return 0.0
            # Synthesise a tiny opposing-side weight so the
            # downstream balance/stakes maths stays well-defined.
            if hope_weight <= 0.0:
                hope_weight = 1e-6
            if threat_weight <= 0.0:
                threat_weight = 1e-6

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

        score = max(0.0, min(1.0, score)) * classic_mult

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
    _SURPRISE_TRAIT_KL_WEIGHT: float = 0.4
    _SURPRISE_ANACHRONY_WEIGHT: float = 0.2
    # Audience belief-revision component (Itti & Baldi 2009 Bayesian
    # surprise on the audience's posterior over event propositions).
    # The trait-KL form is mathematically correct but produces a
    # near-zero signal whenever a focal entity's traits sit close to
    # the corpus marginal — i.e. on most realistic protagonists. The
    # belief-revision form measures KL between the audience's
    # confidence-over-time on every uncommitted proposition and is
    # the direct analogue of what the unified suspense scorer reads
    # from the same substrate. Blending it in here gives the gauge a
    # responsive event-driven channel without breaking the existing
    # trait/anachrony test contract.
    _SURPRISE_BELIEF_KL_WEIGHT: float = 0.4

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
        if syuzhet_anchor is None and getattr(
            self, "_fabula_anchor_override", None
        ) is None:
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
            # Build event-id → actor-id set lookup so the source-side
            # branch below can correctly fire on entity participation
            # (round-3 audit fix). Previously the source-side check
            # was ``ce.source_id == eid``, which compared an event id
            # to an entity id and could never be true — every
            # source-side Bayesian update was silently dropped.
            _evt_actors: Dict[str, set[str]] = {
                evt.id: set(evt.actor_ids) for evt in self.world_state.events
            }
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
                elif eid in _evt_actors.get(ce.source_id, ()):
                    # Batch C.5 — source-side: reduced-weight
                    # update. "X did Y to Z" speaks more strongly
                    # about Z's traits than X's, but X's act
                    # itself is evidence about X's traits too
                    # (ambition reinforced by acting on it). The
                    # source-edge participant test is membership in
                    # the source event's actor_ids — NOT identity
                    # with the source-event id, which the legacy
                    # check used and which never matched.
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

        # Audience belief-revision component (Itti-Baldi 2009 over
        # the unified Proposition substrate). For ``local=True`` we
        # take the per-step KL between the audience's posterior at
        # this anchor's fabula time and the previous anchor's fabula
        # time; for cumulative we take the integrated audience-belief
        # entropy on still-open propositions (which collapses as the
        # plot resolves, mirroring the cumulative trait form). Both
        # are normalised by the open-proposition count × ln 2 so the
        # output stays in [0, 1].
        belief_kl_score = 0.0
        try:
            from shadow_loom.affect_unification import (
                AUDIENCE_ID,
                BeliefState,
                _binary_entropy,
                _binary_kl,
                _prop_stakes_at,
                backfill_character_belief_propositions,
                synthesise_audience_entity,
                synthesise_propositions,
            )
            ws = self.world_state
            if not ws.propositions:
                synthesise_propositions(ws)
                backfill_character_belief_propositions(ws)
            if AUDIENCE_ID not in ws.entities:
                synthesise_audience_entity(ws)
            bs = BeliefState(world=ws)
            revealed_now = self._revealed_event_ids(syuzhet_anchor)
            ft_now = self._fabula_now(revealed_now)
            if ft_now is not None and ws.propositions:
                if local:
                    revealed_prev = self._revealed_event_ids(
                        max(0, syuzhet_anchor - 1),
                    )
                    ft_prev = self._fabula_now(revealed_prev)
                    if ft_prev is None:
                        ft_prev = ft_now
                    # Correlation-aware aggregator (Friston 2010 free-
                    # energy predictive coding): two simultaneous
                    # reveals on causally-connected propositions are
                    # *jointly implied*, so the audience perceives them
                    # as a single information unit, not two independent
                    # surprises. Build a lookup of moved-this-step
                    # propositions, then for each we'll deflate its KL
                    # by the fraction of its immediate causal
                    # predecessors / successors that *also* moved this
                    # step. Mathematically: w_p = 1 / (1 + κ · n_kin)
                    # where n_kin counts in-step causal kin and κ
                    # controls how aggressively we collapse. κ=0.5
                    # halves the contribution when one kin moved with
                    # it; quarters when three did.
                    moved_this_step: Dict[str, float] = {}
                    for prop in ws.propositions:
                        p_now = bs.confidence(
                            AUDIENCE_ID, prop.proposition_id, ft_now,
                        )
                        p_prev = bs.confidence(
                            AUDIENCE_ID, prop.proposition_id, ft_prev,
                        )
                        if abs(p_now - p_prev) < 1e-9:
                            continue
                        moved_this_step[prop.proposition_id] = (
                            _binary_kl(p_now, p_prev) * _prop_stakes_at(
                                prop, ft_now,
                            )
                        )
                    # Build prop_id → referent event_id and an event-
                    # level causal-kin map (predecessors + successors)
                    # so we can count in-step kin per proposition.
                    prop_to_evt: Dict[str, str] = {}
                    for prop in ws.propositions:
                        if prop.referent_ids:
                            prop_to_evt[prop.proposition_id] = prop.referent_ids[0]
                    causal_kin: Dict[str, Set[str]] = {}
                    for ce in ws.causal_topology:
                        causal_kin.setdefault(ce.source_id, set()).add(
                            ce.target_id,
                        )
                        causal_kin.setdefault(ce.target_id, set()).add(
                            ce.source_id,
                        )
                    moved_evt_ids = {
                        prop_to_evt[pid] for pid in moved_this_step
                        if pid in prop_to_evt
                    }
                    KAPPA = 0.5
                    total = 0.0
                    n_terms = 0
                    for pid, raw_kl in moved_this_step.items():
                        evt_id = prop_to_evt.get(pid)
                        n_kin = 0
                        if evt_id is not None:
                            kin = causal_kin.get(evt_id, set())
                            n_kin = len(kin & moved_evt_ids)
                        deflate = 1.0 / (1.0 + KAPPA * n_kin)
                        total += raw_kl * deflate
                        n_terms += 1
                    if n_terms > 0:
                        belief_kl_score = min(
                            1.0, total / (n_terms * math.log(2.0)),
                        )
                else:
                    # Cumulative: integrated entropy on uncommitted
                    # outcome-shaped propositions at the current
                    # anchor. Mirrors the cumulative trait gap form.
                    total = 0.0
                    n_terms = 0
                    for prop in ws.propositions:
                        # Mirror ``compute_suspense_unified`` — keep
                        # multi-flip propositions open while a future
                        # commit is still pending.
                        future_commits_c = [
                            t for t in prop.truth_at_fabula if t > ft_now
                        ]
                        if not future_commits_c and any(
                            t <= ft_now for t in prop.truth_at_fabula
                        ):
                            continue
                        p_aud = bs.confidence(
                            AUDIENCE_ID, prop.proposition_id, ft_now,
                        )
                        h = _binary_entropy(p_aud)
                        if h <= 0.0:
                            continue
                        total += h * _prop_stakes_at(prop, ft_now)
                        n_terms += 1
                    if n_terms > 0:
                        belief_kl_score = min(
                            1.0, total / (n_terms * math.log(2.0)),
                        )
        except Exception:
            logger.debug(
                "surprise belief-KL component failed", exc_info=True,
            )

        # Convex weighted combine of trait-shift KL (Itti-Baldi /
        # Storck on entity traits), audience-belief revision (Itti-
        # Baldi on the unified Proposition substrate), and anachrony
        # (Bae-Young / Bissell-Paulin-Piper 2025). Weights sum to 1
        # so the result stays in [0, 1].
        score = (
            self._SURPRISE_TRAIT_KL_WEIGHT * trait_kl_score
            + self._SURPRISE_BELIEF_KL_WEIGHT * belief_kl_score
            + self._SURPRISE_ANACHRONY_WEIGHT * anachrony_score
        )

        logger.debug(
            "[DirectiveAssembly·Surprise%s] trait_kl=%.4f belief_kl=%.4f "
            "anachrony=%.4f → %.4f over %d traits",
            "·local" if local else "",
            trait_kl_score, belief_kl_score, anachrony_score,
            score, trait_count,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Narrative Tension (Brewer-Lichtenstein triad aggregator)
    # ------------------------------------------------------------------
    def compute_tension_score(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int] = None,
    ) -> float:
        """Composite tension reading across the structural-affect triad.

        Brewer & Lichtenstein 1982 frame narrative tension as a
        *triad* (suspense + curiosity + surprise) rather than four
        independent gauges; Sternberg 1978 *Expositional Modes*
        frames it as the disequilibrium between what the reader
        knows, suspects, and is owed; Vorderer-Wulff-Friedrichsen
        1996 *Suspense: Conceptualizations…* defines tension as the
        running integral of moment-to-moment uncertainty.

        We aggregate as

            T = α · suspense + β · mystery + γ · irony +
                δ · |Δ surprise| + ε · unpaid_setup_debt

        with weights ``(α,β,γ,δ,ε) = (0.40, 0.25, 0.20, 0.10, 0.05)``:
        suspense dominates per Brewer-Lichtenstein; mystery and
        irony contribute their current cumulative gaps; surprise
        contributes its *first derivative* as the disequilibrium
        kick (per Friston 2010 prediction error); unpaid setup debt
        adds the Chekhov's-gun overhang from foreshadowing arcs.

        Returns a clamped ``[0, 1]`` score.
        """
        suspense = self.compute_suspense_score(
            entity_ids, syuzhet_anchor,
        )
        mystery = self.compute_mystery_score(
            entity_ids, syuzhet_anchor,
        )
        irony = self.compute_dramatic_irony_score(
            entity_ids, syuzhet_anchor,
        )
        # Surprise derivative — local form is already the per-step
        # belief-update spike; that *is* dT/ds for surprise.
        surprise_delta = self.compute_surprise_score(
            entity_ids, syuzhet_anchor, local=True,
        )
        # Unpaid setup debt — count foreshadowing setups whose
        # payoff event (if any) hasn't been revealed yet at the
        # anchor. Saturating curve so debt-rich worlds (Chekhov,
        # Tinker Tailor) lift visibly without pegging the gauge.
        debt_score = 0.0
        try:
            from shadow_loom_ui.reasoning_helpers import (
                foreshadowing_arcs_data,
            )
            arcs = foreshadowing_arcs_data(self.world_state)
            revealed = self._revealed_event_ids(syuzhet_anchor)
            payoff_syuzhet: Dict[str, int] = {
                e.id: e.syuzhet_index for e in self.world_state.events
            }
            unpaid = 0
            for a in arcs:
                if a.get("is_loose"):
                    unpaid += 1
                    continue
                pid = a.get("payoff_id")
                if pid in payoff_syuzhet and pid not in revealed:
                    unpaid += 1
            debt_score = unpaid / (unpaid + 4.0)
        except Exception:
            logger.debug(
                "tension unpaid-setup-debt failed", exc_info=True,
            )

        score = (
            0.40 * suspense
            + 0.25 * mystery
            + 0.20 * irony
            + 0.10 * surprise_delta
            + 0.05 * debt_score
        )
        score = max(0.0, min(1.0, score))
        logger.debug(
            "[DirectiveAssembly·Tension] suspense=%.3f mystery=%.3f "
            "irony=%.3f Δsurprise=%.3f debt=%.3f → %.3f",
            suspense, mystery, irony, surprise_delta, debt_score, score,
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

        elif target_effect == "narrative_tension":
            score -= self.compute_tension_score(entity_ids, syuzhet_anchor)

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

            trajectories = self.compute_trait_trajectories(
                entity_ids, syuzhet_anchor=syuzhet_anchor,
            )
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
            # World-state must ALSO be forked per candidate. The typed
            # ``_apply_do_*`` handlers mirror surgical writes onto the
            # canonical world_state (object owner/location, event traits,
            # proposition truth, world-trait value, etc.). If we share
            # ``self.world_state`` across all K candidates each candidate
            # bleeds its mutations into the next candidate's "before"
            # state, and the scorer below would read from a polluted
            # snapshot rather than this candidate's post-intervention
            # state. Deep-copy isolates per-candidate mutations and
            # makes ``forked_world`` the actual post-intervention
            # world-state passed to ``compute_affective_score``.
            forked_world = deepcopy(self.world_state)
            engine = CausalPhysicsEngine(forked, forked_world)
            physics = engine.execute(
                rung=2,
                interventions=candidate,
                target_node_ids=entity_ids,
                causal_diagram=shared_diagram,
            )

            # --- Pruning: intervention failed AND all propagations blocked ---
            # A candidate is physically impossible only if the surgery itself
            # was blocked (no intervened nodes) and no mutations of ANY
            # mutation family occurred. Trait mutations alone are too
            # narrow: a candidate may legitimately produce only
            # object-, proposition-, belief-, concern-, world-trait-,
            # or social-relationship-mutations (e.g. a PROP truth clamp,
            # an OBJ owner clamp, a DoBelief surgery). Surface those as
            # effectful so the candidate isn't wrongly pruned.
            surgery_applied = len(physics.intervened_nodes) > 0
            has_mutations = (
                bool(physics.mutations)
                or bool(physics.social_mutations)
                or bool(getattr(physics, "object_mutations", None))
                or bool(getattr(physics, "proposition_mutations", None))
                or bool(getattr(physics, "belief_mutations", None))
                or bool(getattr(physics, "concern_mutations", None))
                or bool(getattr(physics, "world_trait_mutations", None))
            )
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
            # Build a fresh assembler on the forked sandbox AND the
            # candidate-mutated world projection so scorers that read
            # from ``world_state`` (object owner/location, event traits,
            # proposition truth, world traits) see this candidate's
            # actual post-intervention state rather than the canonical
            # pre-intervention baseline.
            forked_ego = self.ego  # ego is read-only, safe to share
            forked_assembler = DirectiveAssembler(
                sandbox=forked, ego_payload=forked_ego,
                world_state=forked_world,
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
        trajectories = self.compute_trait_trajectories(
            entity_ids, syuzhet_anchor=syuzhet_anchor,
        )
        rel_tensions = self.compute_relationship_tensions(entity_ids)
        narrative_tensions = self.compute_narrative_tension(syuzhet_anchor)
        hidden_channels = self.compute_hidden_channels(syuzhet_anchor)

        constraints: List[ConstraintBlock] = []

        # =============================================================
        # NEGATIVE PHYSICS  (what the prose must NOT stage)
        # =============================================================
        # The "what to do" half of the brief is the constraint /
        # mechanism / tension stack below. The "what NOT to do" half
        # comes from the world's negative-physics record: events the
        # instantiator (or a Rung-2/3 surgery) tagged as not occurring,
        # and propositions committed FALSE at or before the anchor.
        # Both must reach the renderer AND the auditor or the prose
        # silently re-narrates non-occurrences as fact.
        constraints.extend(build_prevented_event_constraints(
            self.world_state, syuzhet_anchor, world_label="this",
        ))
        constraints.extend(build_false_proposition_constraints(
            self.world_state, syuzhet_anchor, world_label="this",
        ))
        constraints.extend(build_unrealised_concern_constraints(
            self.world_state, syuzhet_anchor, world_label="this",
        ))
        constraints.extend(build_false_belief_grounding_constraints(
            self.world_state, syuzhet_anchor, world_label="this",
        ))

        # =============================================================
        # OBJECT COHERENCE  (where each prop is + what it can do)
        # =============================================================
        # Translate syuzhet_anchor → fabula_anchor (max fabula_time of
        # any event with syuzhet_index <= syuzhet_anchor) so
        # ``reconstruct_object_at`` can walk each object's
        # state_timeline to the correct tick. Falls back to None when
        # the anchor is unset; the builder then uses the static
        # initial position which is still better than nothing.
        fabula_anchor: Optional[int] = None
        if syuzhet_anchor is not None:
            for evt in self.world_state.events:
                if evt.syuzhet_index <= syuzhet_anchor and (
                    fabula_anchor is None or evt.fabula_time > fabula_anchor
                ):
                    fabula_anchor = evt.fabula_time
        constraints.extend(build_object_coherence_constraints(
            self.world_state, fabula_anchor, world_label="this",
        ))

        # =============================================================
        # EVENT CO-PRESENCE  (where each event happens + who is there)
        # =============================================================
        # PR 4 of EventNode.at_location_id. Pin every windowed event
        # to its declared spatial anchor and cascade the implicit
        # co-presence rule into MUST_BE_PRESENT / MUST_NOT_BE_PRESENT
        # ledgers the auditor consumes deterministically.
        constraints.extend(build_event_copresence_constraints(
            self.world_state,
            fabula_anchor,
            syuzhet_anchor,
            world_label="this",
        ))

        # =============================================================
        # SCENIC GROUNDING  (universal lived-present anchor)
        # =============================================================
        # Mirrors the hard ConstraintBlock that build_observation_brief,
        # build_intervention_brief, and build_counterfactual_brief all
        # ship. Without it the directive path's stylistic_instructions
        # (which vary per affect) are the only style anchor and the
        # auditor's stylistic checks have no shared HARD constraint to
        # bind to. Compatible with every effect because it constrains
        # narrator stance, not internal content.
        constraints.append(ConstraintBlock(
            constraint_type="narrative",
            priority="hard",
            instruction=(
                "Render this scene as the actual lived world \u2014 concrete "
                "physical action, sensory detail, and character behaviour, "
                "in plain past-tense narration. Do NOT use author-voice "
                "conditional or subjunctive framing (\"if he had\u2026\", "
                "\"would have\u2026\"). The events of this scene are what "
                "actually happened in this world."
            ),
            evidence={},
        ))

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
        surprise_profile: Optional[SurpriseProfile] = None
        irony_profile: Optional[IronyProfile] = None
        mystery_profile: Optional[MysteryProfile] = None
        fear_profile: Optional[FearProfile] = None
        joy_profile: Optional[JoyProfile] = None
        regret_profile: Optional[RegretProfile] = None
        grief_profile: Optional[GriefProfile] = None
        rage_profile: Optional[RageProfile] = None
        love_profile: Optional[LoveProfile] = None

        pov_entity = entity_ids[0] if entity_ids else None

        # P0 #2b (round-7 deeper audit 2026-05-27): when the
        # directive targets multiple entities, the primary entity is
        # the POV anchor and the rest are additional licensed POVs
        # under a rotating policy. Previously every RenderingDirective
        # constructor below set only ``pov_lock`` and left
        # ``pov_policy="single"`` and ``additional_pov_locks=[]`` at
        # their defaults, silently hard-locking even legitimately
        # multi-POV directives to a single perspective. Setting them
        # here once and wiring them into every constructor below
        # restores the documented contract on
        # ``RenderingDirective.pov_lock``.
        pov_additional_locks: List[str] = (
            [eid for eid in entity_ids[1:] if eid] if entity_ids else []
        )
        pov_policy: Literal["single", "rotating", "ensemble"] = (
            "rotating" if pov_additional_locks else "single"
        )

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
            try:
                payload = self._compute_unified_affect_payload(
                    entity_ids, syuzhet_anchor, "mystery",
                )
                if isinstance(payload, MysteryProfile):
                    mystery_profile = payload
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] mystery_profile build failed; "
                    "continuing without it.",
                )
            rendering = RenderingDirective(
                rendering_mode="mystery",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
            try:
                payload = self._compute_unified_affect_payload(
                    entity_ids, syuzhet_anchor, "irony",
                )
                if isinstance(payload, IronyProfile):
                    irony_profile = payload
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] irony_profile build failed; "
                    "continuing without it.",
                )
            rendering = RenderingDirective(
                rendering_mode="dramatic_irony",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
            try:
                payload = self._compute_unified_affect_payload(
                    entity_ids, syuzhet_anchor, "surprise",
                )
                if isinstance(payload, SurpriseProfile):
                    surprise_profile = payload
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] surprise_profile build failed; "
                    "continuing without it.",
                )
            rendering = RenderingDirective(
                rendering_mode="surprise",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
            try:
                fear_profile = self._compute_character_emotion_payload(
                    entity_ids, syuzhet_anchor, "fear",
                )
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] fear_profile build failed; "
                    "continuing without it.",
                )
                fear_profile = None

            rendering = RenderingDirective(
                rendering_mode="fear",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
            try:
                joy_profile = self._compute_character_emotion_payload(
                    entity_ids, syuzhet_anchor, "joy",
                )
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] joy_profile build failed; "
                    "continuing without it.",
                )
                joy_profile = None
            rendering = RenderingDirective(
                rendering_mode="joy",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
            try:
                regret_profile = self._compute_character_emotion_payload(
                    entity_ids, syuzhet_anchor, "regret",
                )
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] regret_profile build failed; "
                    "continuing without it.",
                )
                regret_profile = None

            rendering = RenderingDirective(
                rendering_mode="regret",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
            try:
                grief_profile = self._compute_character_emotion_payload(
                    entity_ids, syuzhet_anchor, "grief",
                )
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] grief_profile build failed; "
                    "continuing without it.",
                )
                grief_profile = None
            rendering = RenderingDirective(
                rendering_mode="grief",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
            try:
                rage_profile = self._compute_character_emotion_payload(
                    entity_ids, syuzhet_anchor, "rage",
                )
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] rage_profile build failed; "
                    "continuing without it.",
                )
                rage_profile = None

            rendering = RenderingDirective(
                rendering_mode="rage",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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
            try:
                love_profile = self._compute_character_emotion_payload(
                    entity_ids, syuzhet_anchor, "love",
                )
            except Exception:
                logger.exception(
                    "[DirectiveAssembly] love_profile build failed; "
                    "continuing without it.",
                )
                love_profile = None

            rendering = RenderingDirective(
                rendering_mode="love",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
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

        # =============================================================
        # NARRATIVE TENSION  (Brewer-Lichtenstein triad:
        # suspense + mystery + surprise + irony + unpaid setup debt)
        # =============================================================
        # Composite mode. The per-component renderer guidance
        # (mystery / suspense / surprise / dramatic_irony) is already
        # carried by the universal ``=== NARRATIVE TENSION ===`` block
        # that ``assemble_creative_brief`` (generation.py) writes from
        # ``brief.narrative_tensions``. The job of this branch is to
        # (a) emit a HARD composite envelope so the renderer treats
        # the whole triad as the primary affect, and (b) attach the
        # numeric tension reading so the auditor can score parity
        # against the rendered scene.
        elif effect == "narrative_tension":
            tension_score = self.compute_tension_score(
                entity_ids, syuzhet_anchor,
            )
            withheld = [
                t for t in narrative_tensions
                if t.tension_type == "withheld_cause"
            ]
            upcoming = [
                t for t in narrative_tensions
                if t.tension_type == "upcoming_revelation"
            ]
            constraints.append(ConstraintBlock(
                constraint_type="mathematical",
                priority="hard",
                instruction=(
                    f"[NARRATIVE TENSION CONSTRAINT]: Composite "
                    f"Brewer-Lichtenstein triad reading is "
                    f"{tension_score:.2f} (suspense + mystery + irony + "
                    f"|Δsurprise| + unpaid setup debt). This scene must "
                    f"SUSTAIN tension, not resolve it: keep open threats "
                    f"unresolved, hidden causes hidden, and dramatic-irony "
                    f"gaps unclosed. {len(withheld)} withheld cause(s) and "
                    f"{len(upcoming)} upcoming revelation(s) are in flight "
                    f"— render their pressure through aftermath, blocking, "
                    f"and sensory beats. Do not collapse the triad by "
                    f"explaining, naming, or paying off prematurely."
                ),
                evidence={
                    "tension_score": tension_score,
                    "withheld_count": len(withheld),
                    "upcoming_count": len(upcoming),
                    "withheld_event_ids": [t.event_id for t in withheld[:8]],
                    "upcoming_event_ids": [t.event_id for t in upcoming[:8]],
                },
            ))
            rendering = RenderingDirective(
                rendering_mode="narrative_tension",
                pov_lock=pov_entity,
                additional_pov_locks=pov_additional_locks,
                pov_policy=pov_policy,
                pacing="dilated",
                sensory_focus="normal",
                tone_arc="sustained_pressure",
                stylistic_instructions=[
                    "Sustain the composite envelope: suspense + mystery + irony + surprise overhang all live simultaneously.",
                    "Pace the beat so withheld causes pre-load consequence without being explained on-page.",
                    "Foreshadow upcoming revelations through environmental and behavioural cues, never through narration.",
                    "Keep dramatic-irony gaps open: characters act on their incomplete picture; the prose does not close it for the reader.",
                    "Do NOT resolve, name, or meta-narrate the tension — render only effects, blocking, and sensory pressure.",
                    "If a surprise pivot is queued, hold the comfortable register until the pivot lands; do not pre-shock.",
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

        # Universal lived-present grounding tail — append to whichever
        # affect-specific RenderingDirective was built above so every
        # directive brief carries the same scenic anchor that
        # build_observation_brief / build_intervention_brief /
        # build_counterfactual_brief end with. Compatible with every
        # affect (constrains narrator stance, not internal content).
        if rendering is not None:
            _grounding_tail = [
                "Ground the prose in concrete physical reality \u2014 "
                "what the POV character sees, hears, touches, and "
                "does, moment by moment.",
                "Render the scene as the lived present of this world. "
                "Do not stand outside it as a narrator commenting on "
                "its structure.",
            ]
            existing = list(rendering.stylistic_instructions or [])
            for line in _grounding_tail:
                if line not in existing:
                    existing.append(line)
            rendering = rendering.model_copy(update={
                "stylistic_instructions": existing,
            })

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
            world_trait_shifts=self.compute_world_trait_shifts(syuzhet_anchor),
            physics_override=physics_override,
            scene_context=scene_context,
            rendering=rendering,
            counterfactual_branch=counterfactual_branch,
            threat_proximity=threat_proximity,
            surprise_profile=surprise_profile,
            irony_profile=irony_profile,
            mystery_profile=mystery_profile,
            causal_attribution=causal_attribution,
            entanglement_pairs=entanglement_pairs,
            fear_profile=fear_profile,
            joy_profile=joy_profile,
            regret_profile=regret_profile,
            grief_profile=grief_profile,
            rage_profile=rage_profile,
            love_profile=love_profile,
            external_research=self._select_external_research(entity_ids),
            narrative_style=getattr(self.world_state, "narrative_style", None),
        )
        _resolve_pov_form_class_mutex(brief)
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
        """Build a ThreatProximity payload for fear/suspense rendering.

        Improvements A1-A5 / B8: this used to be a much weaker scorer
        than ``compute_suspense_score`` — raw actor/target buckets, no
        salience, no proximity, no persistence, ``hope_probability``
        was a bare edge-weight max with no symmetry to the threat side,
        and the spatial graph was rebuilt from the sandbox separately
        from the scorer's version (drifted on locked / destroyed
        edges). The dashboard / generation layer therefore read a
        regressed view of evidence the scorer already had.

        It now reuses the same primitives:

          * ``_bucket_event_for_focal`` (disposition-aware: rescue /
            coerced participation / third-party widening, A1)
          * harm-kind salience and per-event multiplier
            ``prob × salience × imminence_t × imminence_s ×
            persistence_mult / (1 + β · unrevealed_ancestors)`` (A2,
            B7, B9)
          * symmetric hope side built from the same product (A3)
          * pre-built ``events_by_id`` and ``force_by_target`` lookup
            tables — linear in events, no quadratic scan inside the
            loop (A4)
          * ``_build_spatial_graph`` reused so locked / destroyed
            passages are excluded uniformly (A5)
          * dominant ``threat_kind`` / ``hope_kind`` surfaced for the
            UI / auditor (B8)
        """
        causal_g = self._build_causal_digraph()
        revealed = self._revealed_event_ids(syuzhet_anchor)
        all_evt_ids = {e.id for e in self.world_state.events}
        unrevealed = all_evt_ids - revealed
        eid_set = set(entity_ids)

        # A4: O(events) lookup tables (events_by_id, force_by_target)
        # so the per-unrevealed-event inner loop is linear instead of
        # quadratic over events × causal_topology.
        events_by_id = {e.id: e for e in self.world_state.events}
        force_by_target: Dict[str, float] = {}
        for ce in self.world_state.causal_topology:
            cur = force_by_target.get(ce.target_id, 0.0)
            if ce.causal_force > cur:
                force_by_target[ce.target_id] = ce.causal_force

        # A5: reuse the same spatial graph the suspense scorer uses
        # (excludes locked / destroyed passages).
        spatial_g = self._build_spatial_graph()
        affinity_idx = self._build_affinity_index()

        fabula_now = self._fabula_now(revealed)
        sorted_fts = sorted({e.fabula_time for e in self.world_state.events})
        if len(sorted_fts) >= 2:
            gaps = [
                b - a for a, b in zip(sorted_fts, sorted_fts[1:]) if b > a
            ]
            typical_gap = (
                float(sorted(gaps)[len(gaps) // 2]) if gaps else 1.0
            )
        else:
            typical_gap = 1.0
        tau_fabula = max(
            1.0, self._SUSPENSE_PROXIMITY_TAU_FABULA_GAPS * typical_gap,
        )
        tau_spatial = self._spatial_tau(spatial_g)

        # Best (highest weighted-prob) event per side, plus the
        # dominant kind for B8.
        best_threat_id: Optional[str] = None
        best_threat_desc = ""
        best_threat_kind: Optional[str] = None
        threat_prob = 0.0
        best_force = 0.0
        best_threat_spatial_dist: Optional[int] = None

        best_hope_id: Optional[str] = None
        best_hope_kind: Optional[str] = None
        hope_prob = 0.0

        # Spatial distance memo across focal entities.
        spatial_memo: Dict[Tuple[str, str], Optional[int]] = {}

        def _spatial_dist(
            evt_loc: Optional[str], focal_loc: Optional[str],
        ) -> Optional[int]:
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
            # Cheap pre-filter: only keep events touching focal entities.
            # ``_bucket_event_for_focal`` handles disposition-aware
            # third-party widening on the suspense side; for the
            # ThreatProximity headline we deliberately stay focal-
            # touching so the UI single-event picker doesn't surface
            # unrelated subplots.
            if not (actor_set & eid_set) and not (target_set & eid_set):
                continue

            # B7: shared base-weight helper applies the orphan-event
            # prior and the per-event harm-kind salience.
            prob, kind, salience = self._event_base_weight(evt_id, causal_g)

            # Anticipatory proximity (A2 / B9 / B6).
            dt = (
                max(0, evt.fabula_time - fabula_now)
                if fabula_now is not None else 0
            )
            imminence_t = math.exp(-dt / tau_fabula)
            unrev_anc = self._unrevealed_ancestor_count(
                evt_id, causal_g, revealed,
            )
            imminence_t /= (
                1.0 + self._SUSPENSE_REMAINING_SETUP_BETA * unrev_anc
            )

            # Persistence multiplier (A2).
            rev_anc = self._revealed_ancestor_count(
                evt_id, causal_g, revealed,
            )
            persistence_mult = min(
                self._SUSPENSE_PERSISTENCE_CAP,
                1.0 + self._SUSPENSE_PERSISTENCE_ALPHA * rev_anc,
            )

            evt_loc = self._event_location_id(evt)
            force = force_by_target.get(evt_id, 0.0)

            # A1 / A3: disposition-aware bucketing per focal entity,
            # symmetric on hope side.
            for fid in eid_set:
                bucket = self._bucket_event_for_focal(
                    evt, fid, affinity_idx,
                )
                if bucket is None:
                    continue

                fent = self.world_state.entities.get(fid)
                focal_loc = (
                    getattr(fent, "location_id", None)
                    if fent is not None else None
                )
                d_sp = _spatial_dist(evt_loc, focal_loc)
                imminence_s = (
                    1.0 if d_sp is None
                    else math.exp(-d_sp / tau_spatial)
                )

                weighted = (
                    prob * salience * imminence_t * imminence_s
                    * persistence_mult
                )
                if weighted <= 0.0:
                    continue

                if bucket == "threat":
                    if weighted > threat_prob:
                        threat_prob = weighted
                        best_threat_id = evt.id
                        best_threat_desc = evt.description
                        best_threat_kind = kind
                        best_force = force
                        best_threat_spatial_dist = d_sp
                elif bucket == "hope":
                    if weighted > hope_prob:
                        hope_prob = weighted
                        best_hope_id = evt.id
                        best_hope_kind = kind

        # Clamp the surfaced probabilities into [0, 1]; the weighted
        # product can exceed 1.0 for a high-stakes existentially-
        # salient threat, but the UI consumer treats this as a [0, 1]
        # gauge.
        threat_prob_out = round(max(0.0, min(1.0, threat_prob)), 3)
        hope_prob_out = round(max(0.0, min(1.0, hope_prob)), 3)

        return ThreatProximity(
            threat_event_id=best_threat_id,
            threat_description=best_threat_desc,
            threat_probability=threat_prob_out,
            hope_probability=hope_prob_out,
            threat_kind=best_threat_kind,
            hope_kind=best_hope_kind,
            spatial_distance=best_threat_spatial_dist,
            damage_potential=best_force,
        )

    # ------------------------------------------------------------------
    # Unified affect siblings (Affect Unification, Step 6)
    # ------------------------------------------------------------------
    def _compute_unified_affect_payload(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int],
        which: Literal["surprise", "irony", "mystery"],
    ) -> Optional[BaseModel]:
        """Build one of the three sibling affect payloads from the
        shared ``BeliefState`` substrate.

        Wraps :func:`shadow_loom.affect_unification.compute_unified_affects`
        so the brief-build path stays decoupled from the unified
        scorers' synthesis side-effects (lazy proposition / audience
        synthesis on first call). Returns ``None`` if there is no
        fabula anchor available — the unified scorers all need one.
        """
        try:
            from shadow_loom.affect_unification import (
                AUDIENCE_ID,
                BeliefState,
                backfill_character_belief_propositions,
                compute_irony_breakdown,
                compute_irony_unified,
                compute_mystery_breakdown,
                compute_mystery_unified,
                compute_surprise_breakdown,
                compute_surprise_unified,
                synthesise_audience_entity,
                synthesise_propositions,
            )
        except Exception:
            logger.exception(
                "[DirectiveAssembly] affect_unification import failed; "
                "skipping %s payload.", which,
            )
            return None

        w = self.world_state
        if not w.propositions:
            synthesise_propositions(w)
            backfill_character_belief_propositions(w)
        if AUDIENCE_ID not in w.entities:
            synthesise_audience_entity(w)

        revealed = self._revealed_event_ids(syuzhet_anchor)
        ft_now = self._fabula_now(revealed)
        if ft_now is None:
            return None

        bs = BeliefState(world=w)

        if which == "surprise":
            # Prior anchor = previous syuzhet step's fabula time, or
            # ``ft_now - 1`` if anchor is 0 / unavailable.
            prior_anchor = (
                syuzhet_anchor - 1
                if syuzhet_anchor is not None and syuzhet_anchor > 0
                else None
            )
            if prior_anchor is not None:
                prior_revealed = self._revealed_event_ids(prior_anchor)
                ft_prev = self._fabula_now(prior_revealed)
                if ft_prev is None:
                    ft_prev = ft_now - 1
            else:
                ft_prev = ft_now - 1

            score = compute_surprise_unified(bs, ft_now, ft_prev)
            shifted = self._top_shifted_propositions(
                bs, ft_now, ft_prev, AUDIENCE_ID,
            )
            descs = self._proposition_descriptions(shifted)
            primary_focal = entity_ids[0] if entity_ids else None
            other_focals = [
                eid for eid in (entity_ids or [])
                if eid != AUDIENCE_ID
            ]
            breakdown = compute_surprise_breakdown(
                bs, ft_now, ft_prev,
                focal_id=primary_focal,
                other_focal_ids=other_focals or None,
            )
            return SurpriseProfile(
                score=score,
                prior_fabula_t=ft_prev,
                fabula_t=ft_now,
                revealed_proposition_ids=shifted,
                revealed_descriptions=descs,
                pleasant_score=breakdown.pleasant_score,
                unpleasant_score=breakdown.unpleasant_score,
                per_focal_score=breakdown.per_focal_score,
            )

        if which == "irony":
            focal = entity_ids[0] if entity_ids else None
            if focal is None or focal == AUDIENCE_ID:
                return None
            adv_aud = compute_irony_unified(bs, focal, ft_now)
            adv_focal = self._compute_focal_advantage_irony(
                bs, focal, ft_now,
            )
            # Sibling map: only carry entities whose score MATERIALLY
            # differs from the focal's. When every non-audience entity
            # holds zero proposition_id-bound beliefs (typical when
            # Step 5d clustering hasn't run on a hand-built world),
            # all siblings collapse to the same prior-baseline KL
            # value, which is pseudo-signal — suppress it.
            siblings_raw = {
                eid: compute_irony_unified(bs, eid, ft_now)
                for eid in w.entities
                if eid not in (AUDIENCE_ID, focal)
            }
            siblings = {
                eid: v for eid, v in siblings_raw.items()
                if abs(v - adv_aud) > 0.5
            }
            aud_props = self._top_divergent_propositions(
                bs, AUDIENCE_ID, focal, ft_now, k=5,
            )
            focal_props = self._top_divergent_propositions(
                bs, focal, AUDIENCE_ID, ft_now, k=5,
            )
            ib = compute_irony_breakdown(bs, focal, ft_now)
            return IronyProfile(
                focal_id=focal,
                audience_advantage_score=adv_aud,
                focal_advantage_score=adv_focal,
                fabula_t=ft_now,
                by_other_focal=siblings,
                audience_advantage_propositions=aud_props,
                focal_advantage_propositions=focal_props,
                suspense_irony_score=ib.suspense_irony_score,
                curiosity_irony_score=ib.curiosity_irony_score,
                surprise_irony_score=ib.surprise_irony_score,
                concern_weighted_score=ib.concern_weighted_score,
                most_ironised_entity_id=ib.most_ironised_entity_id,
                most_ironised_score=ib.most_ironised_score,
            )

        if which == "mystery":
            score = compute_mystery_unified(bs, ft_now)
            open_qs = self._top_mystery_questions(
                bs, ft_now, k=5, syuzhet_anchor=syuzhet_anchor,
            )
            mb = compute_mystery_breakdown(bs, ft_now)
            return MysteryProfile(
                score=score,
                fabula_t=ft_now,
                open_questions=open_qs,
                plot_gap_score=mb.plot_gap_score,
                character_gap_score=mb.character_gap_score,
                tellability_weighted_score=mb.tellability_weighted_score,
                governing_question_id=mb.governing_question_id,
                governing_question_description=mb.governing_question_description,
                character_gap_descriptions=mb.character_gap_descriptions,
            )

        return None

    # ------------------------------------------------------------------
    # Character-felt emotion payloads (OCC appraisal grid)
    # ------------------------------------------------------------------
    def _compute_character_emotion_payload(
        self,
        entity_ids: List[str],
        syuzhet_anchor: Optional[int],
        which: Literal["fear", "joy", "regret", "grief", "rage", "love"],
    ) -> Optional[BaseModel]:
        """Build one of the six character-emotion profile payloads.

        Wraps the appraisal scorers in
        :mod:`shadow_loom.affect_unification` so the brief-build path
        stays decoupled from substrate synthesis. Returns ``None`` if
        no focal entity or fabula anchor is available.
        """
        try:
            from shadow_loom.affect_unification import (
                AUDIENCE_ID,
                BeliefState,
                backfill_character_belief_propositions,
                compute_fear_appraisal,
                compute_grief_appraisal,
                compute_joy_appraisal,
                compute_love_appraisal,
                compute_rage_appraisal,
                compute_regret_appraisal,
                synthesise_audience_entity,
                synthesise_propositions,
            )
        except Exception:
            logger.exception(
                "[DirectiveAssembly] affect_unification import failed; "
                "skipping %s payload.", which,
            )
            return None

        focal = entity_ids[0] if entity_ids else None
        if focal is None or focal == AUDIENCE_ID:
            return None

        w = self.world_state
        if not w.propositions:
            synthesise_propositions(w)
            backfill_character_belief_propositions(w)
        if AUDIENCE_ID not in w.entities:
            synthesise_audience_entity(w)

        revealed = self._revealed_event_ids(syuzhet_anchor)
        ft_now = self._fabula_now(revealed)
        if ft_now is None:
            return None

        bs = BeliefState(world=w)

        if which == "fear":
            ap = compute_fear_appraisal(bs, focal, ft_now)
            return FearProfile(
                object_fear_score=ap.object_fear_score,
                anxiety_score=ap.anxiety_score,
                coping_score=ap.coping_score,
                flight_available=ap.flight_available,
                dread=ap.dread,
                primary_concern_id=ap.primary_concern_id,
                primary_concern_description=ap.primary_concern_description,
            )

        if which == "joy":
            prior_anchor = (
                syuzhet_anchor - 1
                if syuzhet_anchor is not None and syuzhet_anchor > 0
                else None
            )
            ft_prev = None
            if prior_anchor is not None:
                ft_prev = self._fabula_now(
                    self._revealed_event_ids(prior_anchor),
                )
            ap = compute_joy_appraisal(
                bs, focal, ft_now, prior_fabula_t=ft_prev,
            )
            return JoyProfile(
                own_joy_score=ap.own_joy_score,
                happy_for_score=ap.happy_for_score,
                gloating_score=ap.gloating_score,
                relief_score=ap.relief_score,
                primary_concern_id=ap.primary_concern_id,
                primary_concern_description=ap.primary_concern_description,
            )

        if which == "regret":
            ap = compute_regret_appraisal(bs, focal, ft_now)
            _divergence_desc: Optional[str] = None
            _loss_desc: Optional[str] = None
            if ap.divergence_event_id:
                _div_evt = next(
                    (e for e in self.world_state.events if e.id == ap.divergence_event_id),
                    None,
                )
                _divergence_desc = _div_evt.description if _div_evt else None
            if ap.loss_event_id:
                _loss_evt = next(
                    (e for e in self.world_state.events if e.id == ap.loss_event_id),
                    None,
                )
                _loss_desc = _loss_evt.description if _loss_evt else None
            return RegretProfile(
                agentive_regret_score=ap.agentive_regret_score,
                disappointment_score=ap.disappointment_score,
                commission_score=ap.commission_score,
                omission_score=ap.omission_score,
                downward_relief_score=ap.downward_relief_score,
                divergence_event_id=ap.divergence_event_id,
                divergence_description=_divergence_desc,
                loss_event_id=ap.loss_event_id,
                loss_description=_loss_desc,
                mode=ap.mode,
            )

        if which == "grief":
            ap = compute_grief_appraisal(bs, focal, ft_now)
            _grief_loss_desc: Optional[str] = None
            if ap.loss_event_id:
                _grief_loss_evt = next(
                    (e for e in self.world_state.events if e.id == ap.loss_event_id),
                    None,
                )
                _grief_loss_desc = _grief_loss_evt.description if _grief_loss_evt else None
            return GriefProfile(
                coupling_strength=ap.coupling_strength,
                loss_event_id=ap.loss_event_id,
                loss_description=_grief_loss_desc,
                lost_entity_id=ap.lost_entity_id,
                stage=ap.stage,
                unfinished_concern_count=ap.unfinished_concern_count,
            )

        if which == "rage":
            ap = compute_rage_appraisal(bs, focal, ft_now)
            return RageProfile(
                blocked_concern_score=ap.blocked_concern_score,
                perpetrator_id=ap.perpetrator_id,
                attribution_clarity=ap.attribution_clarity,
                perpetrator_proximity=ap.perpetrator_proximity,
                normative_violation=ap.normative_violation,
                mode=ap.mode,
            )

        if which == "love":
            ap = compute_love_appraisal(bs, focal, ft_now)
            return LoveProfile(
                primary_partner_id=ap.primary_partner_id,
                intimacy_score=ap.intimacy_score,
                passion_score=ap.passion_score,
                commitment_score=ap.commitment_score,
                style=ap.style,
                attachment_style=ap.attachment_style,
            )

        return None

    def _proposition_descriptions(self, prop_ids: List[str]) -> List[str]:
        """Resolve ``proposition_id``\u2009s to their human-readable descriptions
        in the same order, dropping unknown ids."""
        idx = {p.proposition_id: p for p in self.world_state.propositions}
        out: List[str] = []
        for pid in prop_ids:
            prop = idx.get(pid)
            if prop is None:
                continue
            out.append(prop.description or pid)
        return out

    def _top_divergent_propositions(
        self, bs, knower_id: str, ignorant_id: str,
        fabula_t: int, *, k: int = 5,
    ) -> List[str]:
        """Top-k proposition descriptions where ``knower_id``'s
        confidence exceeds ``ignorant_id``'s by > 0.3.

        Used to populate the ``audience_advantage_propositions`` /
        ``focal_advantage_propositions`` fields on
        :class:`IronyProfile`. Stakes-weighted ordering so high-
        stakes gaps surface first.
        """
        idx = {p.proposition_id: p for p in self.world_state.propositions}
        scored: List[tuple[float, str]] = []
        for prop in self.world_state.propositions:
            p_k = bs.confidence(knower_id, prop.proposition_id, fabula_t)
            p_i = bs.confidence(ignorant_id, prop.proposition_id, fabula_t)
            if p_k - p_i <= 0.3:
                continue
            score = (p_k - p_i) * prop.stakes
            scored.append((score, prop.proposition_id))
        scored.sort(reverse=True)
        out: List[str] = []
        for _, pid in scored[:k]:
            prop = idx.get(pid)
            if prop is None:
                continue
            out.append(prop.description or pid)
        return out

    def _top_mystery_questions(
        self, bs, fabula_t: int, *, k: int = 5,
        syuzhet_anchor: Optional[int] = None,
    ) -> List[str]:
        """Top-k 'effect known but causes hidden' propositions.

        These are the open erotetic questions in Carroll's mystery-
        as-question-set theory: facts the audience confidently
        registers but whose causal antecedents on
        ``world.causal_topology`` remain unrevealed at this anchor.
        Ordered by number of unrevealed ancestors descending so the
        most underdetermined effects surface first.

        ``syuzhet_anchor`` is the reader's current narrative
        position. When provided, the revealed-event set is restricted
        to events whose ``syuzhet_index`` has actually happened on-
        page at this anchor; without it the scorer would consider
        every cause revealed anywhere in the story as 'known', which
        collapses the open-question set to zero too early and
        silently degrades the mystery directive (round-3 audit fix).
        """
        from shadow_loom.affect_unification import AUDIENCE_ID
        causal_g = self._build_causal_digraph()
        revealed = self._revealed_event_ids(syuzhet_anchor)
        # Re-derive against the syuzhet position the bs was built
        # for: the BeliefState reads at fabula_t but the renderer is
        # at syuzhet_anchor. Use the same revealed set the suspense
        # branch uses by walking events whose syuzhet has happened.
        idx = {p.proposition_id: p for p in self.world_state.propositions}
        scored: List[tuple[int, str]] = []
        for prop in self.world_state.propositions:
            if not prop.referent_ids:
                continue
            evt_id = prop.referent_ids[0]
            p_aud = bs.confidence(
                AUDIENCE_ID, prop.proposition_id, fabula_t,
            )
            if p_aud < 0.7:
                continue
            if evt_id not in causal_g:
                continue
            try:
                ancestors = set(nx.ancestors(causal_g, evt_id))
            except Exception:  # noqa: BLE001
                continue
            unrevealed = [a for a in ancestors if a not in revealed]
            if len(unrevealed) < 2:
                continue
            scored.append((len(unrevealed), prop.proposition_id))
        scored.sort(reverse=True)
        out: List[str] = []
        for _, pid in scored[:k]:
            prop = idx.get(pid)
            if prop is None:
                continue
            out.append(prop.description or pid)
        return out

    def _top_shifted_propositions(
        self, bs, fabula_t: int, prior_fabula_t: int,
        agent_id: str, k: int = 5,
    ) -> List[str]:
        """Return up to ``k`` proposition_ids whose audience confidence
        shifted most (by absolute delta) between the two anchors."""
        deltas: List[tuple[float, str]] = []
        for prop in bs.world.propositions:
            p_now = bs.confidence(agent_id, prop.proposition_id, fabula_t)
            p_prev = bs.confidence(
                agent_id, prop.proposition_id, prior_fabula_t,
            )
            d = abs(p_now - p_prev)
            if d > 0.05:
                deltas.append((d, prop.proposition_id))
        deltas.sort(reverse=True)
        return [pid for _, pid in deltas[:k]]

    def _compute_focal_advantage_irony(
        self, bs, focal_id: str, fabula_t: int,
    ) -> float:
        """KL(p_focal || p_aud) · stakes — focal-knows-more divergence.

        Mirror of :func:`compute_irony_unified` with the agents
        swapped, so the irony payload can carry both directions of
        the asymmetric KL.
        """
        from shadow_loom.affect_unification import (
            AUDIENCE_ID, _binary_kl, _EPS,
        )
        if focal_id == AUDIENCE_ID:
            return 0.0
        total = 0.0
        for prop in bs.world.propositions:
            p_aud = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
            p_focal = bs.confidence(focal_id, prop.proposition_id, fabula_t)
            if abs(p_aud - p_focal) < _EPS:
                continue
            total += _binary_kl(p_focal, p_aud) * prop.stakes
        return total

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
                f"If \"{divergence_evt.description}\" had gone differently, "
                f"the outcome \"{actual_evt.description}\" might have been averted."
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
                    causal_chain_descriptions=[loss_evt.description],
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
        _evt_by_id = {e.id: e for e in self.world_state.events}
        _chain_descs = [
            _evt_by_id[eid].description if eid in _evt_by_id else ""
            for eid in causal_chain
        ]
        return CausalAttribution(
            perpetrator_id=perpetrator_id,
            perpetrator_name=perp_ent.name if perp_ent else None,
            loss_event_id=loss_evt.id,
            loss_description=loss_evt.description,
            causal_chain=causal_chain,
            causal_chain_descriptions=_chain_descs,
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

# Source-form classes that compress the entire scene into a few sentences
# of summary diction. When the renderer is *also* asked to lock to a
# single character's POV, the two constraints conflict: a sub-300-word
# summary cannot carry interior monologue or moment-by-moment perception
# and *also* honour third-person plot-summary register. Without a tie-
# breaker the auditor flags one violation per iteration and the
# refinement loop ping-pongs between expanding the prose (POV → 800
# words) and re-compressing it (synopsis → 200 words).
_SUMMARY_FORMATS: frozenset[str] = frozenset({
    "plot_summary", "synopsis", "outline",
})


def _resolve_pov_form_class_mutex(brief: "CreativeBrief") -> None:
    """Resolve the POV-lock vs. summary-form conflict in-place.

    When ``brief.rendering.pov_lock`` is set AND ``brief.narrative_style.
    format`` is one of the summary forms (``plot_summary``, ``synopsis``,
    ``outline``), the two constraints are mutually unsatisfiable. Pick a
    winner deterministically:

      * If the query explicitly named a focus entity AND the rendering
        mode is one of the epistemic / affective effects that *requires*
        a POV anchor (``mystery``, ``dramatic_irony``, ``surprise``,
        ``suspense``, ``fear``, ``regret``, ``grief``), POV lock is
        hard. Append a stylistic instruction telling the renderer to
        produce a *compressed POV scene* (≤target_word_max words but
        third-person POV-restricted observation, no novelistic
        interiority).
      * Otherwise POV lock becomes soft; the summary register wins.

    The decision is recorded in
    ``brief.scene_context["pov_form_resolution"]`` so the auditor can
    read which constraint is hard rather than flagging both.
    """
    rendering = getattr(brief, "rendering", None)
    style = getattr(brief, "narrative_style", None)
    if rendering is None or style is None:
        return
    if not getattr(rendering, "pov_lock", None):
        return
    fmt = getattr(style, "format", "unknown")
    if fmt not in _SUMMARY_FORMATS:
        return

    pov_required_modes = {
        "mystery", "dramatic_irony", "surprise",
        "suspense", "fear", "regret", "grief",
    }
    pov_is_hard = rendering.rendering_mode in pov_required_modes
    resolution: Dict[str, Any] = {
        "pov_lock": rendering.pov_lock,
        "format": fmt,
        "winner": "pov" if pov_is_hard else "summary",
        "reason": (
            "POV-anchored effect ({mode}) requires a perspective "
            "lock; downgrading summary register to a *compressed POV "
            "scene*.".format(mode=rendering.rendering_mode)
            if pov_is_hard else
            "Source form is a summary register ({fmt}); downgrading "
            "POV lock to a soft preference (third-person summary "
            "with optional POV bias).".format(fmt=fmt)
        ),
    }

    new_instructions = list(rendering.stylistic_instructions or [])
    if pov_is_hard:
        new_instructions.insert(0, (
            "[Composition rule | HARD] POV lock to "
            f"{rendering.pov_lock} is the binding constraint. The "
            f"source format is {fmt!r}; render a *compressed POV "
            "scene* — third-person limited to the POV character, no "
            "omniscient summary jumps, but obey the target word "
            "budget. Do NOT expand into novelistic interiority; do "
            "NOT switch to omniscient summary."
        ))
    else:
        new_instructions.insert(0, (
            f"[Composition rule | HARD] Source format {fmt!r} is the "
            "binding constraint: render as a summary in the source's "
            f"register. POV bias toward {rendering.pov_lock} is a "
            "*soft preference* — let the summary diction win when "
            "they conflict."
        ))
        # POV lock is now soft; clear it on the directive so the
        # auditor's hard POV-lock check doesn't enforce it.
        try:
            object.__setattr__(rendering, "pov_lock", None)
        except Exception:
            rendering.pov_lock = None  # type: ignore[assignment]

    try:
        object.__setattr__(rendering, "stylistic_instructions", new_instructions)
    except Exception:
        rendering.stylistic_instructions = new_instructions  # type: ignore[assignment]

    if isinstance(brief.scene_context, dict):
        brief.scene_context["pov_form_resolution"] = resolution

    logger.info(
        "[DirectiveAssembly] POV/form-class mutex: %s wins "
        "(mode=%s, format=%s, pov=%s).",
        resolution["winner"], rendering.rendering_mode, fmt,
        resolution["pov_lock"],
    )


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

    if brief.world_trait_shifts:
        lines.append(
            f"  World-trait shifts ({len(brief.world_trait_shifts)}):"
        )
        for ws in brief.world_trait_shifts[:max_items]:
            arrow = "↑" if ws.delta > 0 else "↓"
            lines.append(
                f"    {arrow} {ws.trait_id} {ws.previous_value:.2f}→{ws.current_value:.2f} "
                f"(Δ{ws.delta:+.2f}, inertia={ws.inertia:.2f}, "
                f"domains={ws.affected_domains}, ft={ws.fabula_time}, "
                f"trig={ws.triggered_by})"
            )

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
