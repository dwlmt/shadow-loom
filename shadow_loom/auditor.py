# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Steps 11–12 — Recursive Narrative Auditor & Inner-Loop Refinement.

The auditor evaluates LLM-rendered prose against the mathematical constraints
in the CreativeBrief and the causal physics graph.  When violations are found
it generates explicit feedback and forces re-generation until the prose
converges with the constraints or the retry budget is exhausted.

Key design:
  • The world-state graph is **deep-copied and versioned** before each audit
    cycle so the original (pre-audit) state is never mutated.
  • Each iteration produces an ``AuditCycleSnapshot`` capturing the versioned
    graph, the prose, and the audit verdict.
  • The loop terminates when the auditor passes or ``max_iterations`` is hit.
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import networkx as nx
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, NativeOutput, RunContext

from shadow_loom.causal_physics import CausalPhysicsResult, BlockedPropagation
from shadow_loom.directive_assembly import (
    CreativeBrief,
    ConstraintBlock,
    DirectiveAssembler,
    InterventionMechanism,
    _EFFECT_TRAITS,
)
from shadow_loom.generation import (
    GeneratedScene,
    GenerationConfig,
    IntroducedElements,
    assemble_rendering_prompt,
    format_scene_context_for_prompt,
    render_scene,
    _GenerationDeps,
    _build_generation_agent,
    _format_counterfactual,
    _format_causal_attribution,
    _format_threat_proximity,
    _format_intervention_branch,
    _format_entanglement,
    _format_interventions,
    _format_abduction,
    _annotate_ids,
)
from shadow_loom.models import (
    EventNode,
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_object_at,
)

from shadow_loom.settings import get_settings as _get_settings, resolve_model as _resolve_model
from shadow_loom._agent_logging import log_agent_output

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# =====================================================================
# Audit Category → target_effect mapping
# =====================================================================

#: Maps each target_effect to the audit categories the auditor must run.
#:
#: Two categories run universally regardless of effect:
#:   * ``meta`` (Category 4b) \u2014 meta-narration / pipeline leakage,
#:     the most common cross-mode failure.
#:   * ``style`` (Category 5) \u2014 source-style fidelity, only meaningful
#:     when ``brief.narrative_style`` is populated but cheap to leave on.
#: They're appended at the audit call-site so per-effect lookups stay
#: focused on the structural categories the effect actually requires.
EFFECT_AUDIT_CATEGORIES: Dict[str, List[str]] = {
    # Category 1: Epistemic Queries
    "mystery": ["epistemic", "physics"],
    "dramatic_irony": ["epistemic", "physics"],
    "surprise": ["epistemic", "physics"],
    # Category 2: Probabilistic Queries
    "suspense": ["probabilistic", "physics"],
    "fear": ["probabilistic", "physics"],
    "joy": ["probabilistic", "physics"],
    # Category 3: Counterfactual & Attribution
    "regret": ["counterfactual", "physics"],
    "grief": ["counterfactual", "physics"],
    "rage": ["counterfactual", "physics"],
    "love": ["counterfactual", "physics"],
    # Composite — narrative tension is the Brewer-Lichtenstein triad
    # aggregator over suspense + mystery + irony + Δsurprise + unpaid
    # setup debt. It needs every structural pass to fire because each
    # contributing scorer reads a different aspect of the world. We
    # union epistemic + probabilistic + physics; counterfactual is
    # not strictly required (the surprise term is local Δ, not a
    # rung-3 attribution) but kept off to bound prompt size.
    "narrative_tension": ["epistemic", "probabilistic", "physics"],
    # Non-directive — Pearl-rung-aware so the audit prompt explicitly
    # lists the rung whose semantics the prose must satisfy.
    # Round-14 audit (GAP-6): previously all three collapsed to
    # ``["physics"]``, so an "observation" audit and a "counterfactual"
    # audit listed the same category to the auditor LLM \u2014 erasing the
    # rung distinction that the rest of the pipeline goes to lengths
    # to preserve. The named rung categories are inert strings (they
    # surface in the prompt's "AUDIT CATEGORIES TO CHECK" header)
    # but give the auditor and any downstream log readers a clean
    # rung-1 / rung-2 / rung-3 read.
    "observation": ["physics", "observational"],
    "intervention": ["physics", "interventional"],
    "counterfactual": ["physics", "counterfactual"],
    # Note: ``general`` and ``interrogate`` queries short-circuit before
    # the auditor in pipeline.py — they answer questions *about* the
    # world model and never call the renderer, so they have no prose
    # to audit. ``resolve_audit_categories`` falls back to ["physics"]
    # for any unmapped effect, preserving safety if that ever changes.
}

#: Categories that always run, regardless of target_effect. Kept
#: separate from ``EFFECT_AUDIT_CATEGORIES`` so the per-effect maps
#: remain a focused declaration of which structural audits each
#: effect requires.
UNIVERSAL_AUDIT_CATEGORIES: List[str] = ["meta", "style"]


#: Effects the resolver is willing to silently fall back to a
#: physics-only baseline for. Anything else logs a warning so typos
#: surface fast instead of producing a quietly-under-powered audit.
#: Source of truth: ``query_models.target_effect`` Literal +
#: ``RenderingDirective.rendering_mode`` docstring +
#: ``EFFECT_AUDIT_CATEGORIES`` keys.
_KNOWN_TARGET_EFFECTS: frozenset[str] = frozenset({
    "observation", "intervention", "counterfactual",
    "mystery", "dramatic_irony", "surprise", "suspense",
    "fear", "joy", "regret", "grief", "rage", "love",
    "narrative_tension",
    "general", "interrogate",
    # Renderer-side fallback labels that occasionally surface in
    # briefs constructed by the orchestrator when the user-side
    # effect was untyped:
    "manual_edit", "fallback", "default",
})


def resolve_audit_categories(target_effect: str) -> List[str]:
    """Return the full audit-category list for a given target effect.

    Combines the per-effect categories with
    :data:`UNIVERSAL_AUDIT_CATEGORIES` (deduplicated, order-preserving).
    Use this single helper at every call-site so the auditor prompt and
    the run-audit log stay in sync.

    An unknown ``target_effect`` is a typo / drift signal: the per-effect
    map silently falls back to ``["physics"]`` which downgrades the audit
    to physics + universal only. We log a warning so the regression is
    visible rather than silently shipping a weakened audit.
    """
    if target_effect not in _KNOWN_TARGET_EFFECTS:
        logger.warning(
            "[Auditor] resolve_audit_categories received unknown "
            "target_effect=%r; falling back to physics-only base. "
            "This is usually a typo or a renamed effect that was not "
            "propagated to _KNOWN_TARGET_EFFECTS / "
            "EFFECT_AUDIT_CATEGORIES. Known effects: %s",
            target_effect, sorted(_KNOWN_TARGET_EFFECTS),
        )
    base = EFFECT_AUDIT_CATEGORIES.get(target_effect, ["physics"])
    seen: set[str] = set()
    out: List[str] = []
    for c in list(base) + UNIVERSAL_AUDIT_CATEGORIES:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


# =====================================================================
# Structured Feedback Models (NarrativeOrderObject)
# =====================================================================

class CausalPhysicsFeedback(BaseModel):
    """Audits the physical, spatial, and chronological integrity of the story.

    Metrics are computed from the causal physics engine and directive
    assembly — NOT LLM-judged.
    """
    miracle_steps_detected: List[str] = Field(
        default_factory=list,
        description=(
            "A log of abrupt jumps where a state changes without a "
            "valid preceding derivation or sufficient Impact > Inertia."
        ),
    )
    foreshadowing_payoff_score: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description=(
            "Score verifying that causal commitments or implicit events "
            "established earlier have been logically fulfilled."
        ),
    )
    cognitive_plausibility_score: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description=(
            "Ratio of entity actions consistent with their established "
            "beliefs (1.0 = perfectly plausible)."
        ),
    )
    cognitive_plausibility_details: str = Field(
        default="",
        description=(
            "Assessment of whether characters' actions accurately "
            "reflect their established beliefs without acting on "
            "unacquired knowledge."
        ),
    )
    cyclic_propagation_clusters: List[str] = Field(
        default_factory=list,
        description=(
            "Trait propagations refused because the source node sits inside "
            "a strongly-connected component of the causal subgraph. These "
            "indicate extraction problems (a true cycle in the static "
            "topology), not narrative miracles, and are reported separately "
            "from miracle steps so the rewrite loop doesn't try to 'fix' "
            "physics that is unfixable at the prose layer."
        ),
    )
    noisy_or_absorbed_propagations: List[str] = Field(
        default_factory=list,
        description=(
            "Trait propagations refused by the noisy-OR probabilistic gate "
            "because the aggregate per-edge probability fell below the "
            "configured firing threshold. These are *expected* probabilistic "
            "dampening of weak impulses (impact ≪ inertia), not narrative "
            "miracles, and are reported separately so the rewrite loop "
            "doesn't try to 'fix' the engine's correctly-absorbed weak "
            "nudges with prose-level changes. Inspect this field only "
            "when tuning ``settings.noisy_or_threshold`` or diagnosing "
            "why a deliberately-weak causal chain failed to mutate a trait."
        ),
    )
    rule3_pruned_interventions: List[str] = Field(
        default_factory=list,
        description=(
            "Intervention keys that the ctf-calculus pre-flight (Rule 3 "
            "Exclusion) proved vacuous against the user's query targets. "
            "Reported here so the auditor can warn when the engine had "
            "to drop a request the user explicitly asked for."
        ),
    )
    rule2_redundant_evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Evidence node IDs that the ctf-calculus pre-flight (Rule 2 "
            "Independence) proved d-separated from every intervention "
            "given the rest of the evidence. Abduction on these nodes "
            "is informational only — they cannot change the counter"
            "factual distribution."
        ),
    )


class AffectiveStateFeedback(BaseModel):
    """Measures the reader's psychological experience against the target directives.

    Metrics are computed from the DirectiveAssembler's scoring functions
    — NOT LLM-judged.
    """
    emotional_trajectory_scores: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Quantitative intensities achieved for targeted emotions "
            "(e.g., {'suspense': 0.8, 'mystery': 0.6})."
        ),
    )
    kl_divergence_prediction_error: Optional[float] = Field(
        default=None,
        description=(
            "Calculated divergence between the reader's prior expectation "
            "and actual revelation (critical for Surprise)."
        ),
    )
    affective_loss_mse: Optional[float] = Field(
        default=None,
        description=(
            "Signed affective loss for the brief's ``target_effect`` —"
            " lower is better, range ``[-1, +1]``. Negative values"
            " indicate a strong match (the structural-effect score, or"
            " trait closeness for emotion targets, is subtracted from"
            " zero); positive values indicate a poor match. Despite"
            " the legacy ``_mse`` suffix this is *not* a Mean Squared"
            " Error — no squaring is performed — kept for backward"
            " compatibility with serialized scorecards. ``None`` when"
            " the brief has no target entities or the assembler is"
            " unavailable, so downstream consumers can distinguish"
            " \"perfect fit\" from \"not measured\"."
        ),
    )


class StoryQualitySynthesis(BaseModel):
    """Translates the mathematical and structural metrics into a holistic
    literary critique.  This is the only LLM-judged section.
    """
    coherence_and_consistency_review: str = Field(
        default="",
        description=(
            "Synthesized explanation of how well the prose respected "
            "the high-level symbolic causal graph."
        ),
    )
    reward_hacking_diagnostics: Optional[str] = Field(
        default=None,
        description=(
            "Explanation of how the LLM attempted to cheat the physical "
            "constraints to achieve emotional targets, if applicable."
        ),
    )
    actionable_rewrite_directives: List[str] = Field(
        default_factory=list,
        description=(
            "Specific, targeted instructions for the inner-loop "
            "refinement to fix identified graph or affective errors."
        ),
    )


class NarrativeOrderObject(BaseModel):
    """The collated scorecard generated by the Auditor.

    Combines engine-computed metrics (causal + affective) with an
    LLM-judged literary critique (quality synthesis).  Determines if
    the text passes or must be recursively rewritten.
    """
    causal_feedback: CausalPhysicsFeedback = Field(
        default_factory=CausalPhysicsFeedback,
    )
    affective_feedback: AffectiveStateFeedback = Field(
        default_factory=AffectiveStateFeedback,
    )
    quality_synthesis: StoryQualitySynthesis = Field(
        default_factory=StoryQualitySynthesis,
    )
    overall_pass: bool = Field(
        default=False,
        description=(
            "True if the text meets all thresholds and requires no "
            "rewrite; False if the text must be sent back for refinement."
        ),
    )


class ChangeImpactMetrics(BaseModel):
    """Per-cycle delta metrics for the auditor feedback loop.

    Scoped to a single directive request — measures how the latest
    prose change affects the causal and affective state.  Unlike
    :class:`NarrativeOrderObject`, this contains **no** LLM literary
    critique and **no** overall pass/fail (that lives on AuditResult).
    """
    causal_feedback: CausalPhysicsFeedback = Field(
        default_factory=CausalPhysicsFeedback,
    )
    affective_feedback: AffectiveStateFeedback = Field(
        default_factory=AffectiveStateFeedback,
    )


# =====================================================================
# Output Models
# =====================================================================

class AuditViolation(BaseModel):
    """A single violation detected by the auditor."""
    violation_type: Literal[
        "epistemic_leakage",
        "knowledge_contamination",
        "low_kl_divergence",
        "suspense_threshold",
        "tonal_mismatch",
        "magnitude_too_low",
        "reasoning_failure",
        "affective_failure",
        "attribution_failure",
        "empathy_weight",
        "miracle_step",
        "abduction_failure",
        # Utterance / channel fidelity (Channels & Beliefs subsystem).
        "utterance_truth_contradiction",
        "channel_intelligibility_violation",
        "withheld_utterance_leak",
        "belief_provenance_contradiction",
        # ctf-calculus exclusion leaks (Rung-2 / Rung-3 do-surgery
        # bookkeeping). Each maps to a HARD ``ConstraintBlock`` the
        # brief builder emitted; the deterministic auditor pass scans
        # prose for verbatim/structural breaches so the refinement
        # loop cannot converge on output that contradicts the
        # engine's exclusion ledger.
        "pruned_utterance_leak",
        "disabled_channel_leak",
        "blocked_propagation_leak",
        # Defence-in-depth for do-prevention surgeries: prose stages a
        # NON-utterance event that the engine erased / cause-broke.
        # The auditor's other leak checks cover utterance content
        # (``pruned_utterance_leak``) and channel routing
        # (``disabled_channel_leak``); this one catches the case where
        # the renderer still enacts the underlying action / outcome
        # (e.g. \"Macbeth still stabs Duncan\" after a surgery
        # erased EVT_LADY_MACBETH_PERSUADES and Step B.6 cascaded the
        # prune to EVT_DUNCAN_MURDER). High-precision verbatim check
        # against the event id token and a substring of the event
        # description; paraphrase cases remain the LLM auditor's
        # remit. Critical \u2014 the prose contradicts the do-surgery
        # ledger directly. Fired by
        # :func:`_prevented_event_reenacted_violations`.
        "prevented_event_reenacted",
        # Inert-intervention aftermath (round-5 audit fix; the round-9
        # follow-up added the missing Literal value). Fired when the
        # engine flagged the surgery ``intervention_inert=True`` but
        # the prose still depicts the change "taking hold" ("still
        # steady", "as composed as ever", "unshaken"). Referenced by
        # ``shadow_loom/prompts/auditor.md`` and the UI
        # VIOLATION_EXPLANATIONS dict; without the Literal entry the
        # LLM's emission would fail Pydantic validation and the
        # violation would be silently dropped by the parser.
        "inert_intervention_aftermath",
        # Source-style fidelity (NarrativeStyle profile from ingestion).
        "style_mismatch",
        # Meta-narration: prose comments on its own structure (timelines,
        # divergences, counterfactual machinery) instead of rendering the
        # world as a lived scene.
        "meta_narration",
        # New top-level world element (entity / location / object /
        # world trait / proposition / concern) referenced in the prose
        # without either a SCENE CONTEXT record OR a corresponding
        # declaration in ``GeneratedScene.introduced_elements``. The
        # renderer is allowed to invent, but every invention MUST be
        # declared in the structured payload so the merge can spawn it
        # into the next ``WorldStateV1`` revision and downstream queries
        # know it exists. Free-floating prose names are a hard
        # violation \u2014 the refinement loop must either (a) remove the
        # name and use an existing referent or (b) move the name into
        # ``introduced_elements`` with a justification.
        "undeclared_element",
        # Object-coherence checks (added once NarrativeObject acquired a
        # full state_timeline). The deterministic auditor pass surfaces
        # three new failure modes that were previously invisible:
        #   * ``object_misuse`` — prose has a character perform an
        #     action on an OBJ_ whose declared ``affordances`` do not
        #     include that verb (e.g. "Macbeth read the dagger").
        #   * ``object_position_mismatch`` — prose places an OBJ_ in a
        #     room or in a character's hand that contradicts the
        #     reconstructed ``location_id`` / ``owner_id`` at the
        #     scene's fabula anchor (using
        #     :func:`reconstruct_object_at`).
        #   * ``entity_position_mismatch`` — prose places an ENT_ in a
        #     room that contradicts the reconstructed ``location_id``
        #     at the scene's fabula anchor (using
        #     :func:`reconstruct_entity_at`).
        # All three are HARD violations: they signal the renderer
        # contradicting the engine's authoritative world state.
        "object_misuse",
        "object_position_mismatch",
        "entity_position_mismatch",
        # Event-location / co-presence checks (PR 4 of
        # EventNode.at_location_id). The deterministic auditor pass
        # surfaces three failure modes once events carry an
        # ``at_location_id``:
        #   * ``event_location_mismatch`` \u2014 prose places an event at
        #     a location that contradicts the event's declared
        #     ``at_location_id`` (e.g. the Duncan murder is moved to
        #     the great hall when the schema names the bedchamber).
        #   * ``event_copresence_violation`` \u2014 prose has an actor /
        #     non-channel target *absent* at an event whose schema
        #     binds them as present (the implicit co-presence rule:
        #     every actor + non-channel target is present at
        #     ``at_location_id`` at ``fabula_time`` UNLESS the event is
        #     a channel-mediated utterance).
        #   * ``event_copresence_omission`` \u2014 prose adds a present
        #     character not bound to the event whose reconstructed
        #     location at ``fabula_time`` is NOT the event's
        #     ``at_location_id`` (a phantom witness).
        # All three are HARD violations: they signal the renderer
        # contradicting the engine's spatial / co-presence ledger.
        "event_location_mismatch",
        "event_copresence_violation",
        "event_copresence_omission",
        # Reuse-first / justification check on per-cycle introductions.
        # The renderer (and the user via ``query.introduce``) MAY mint
        # new top-level world elements (entities, locations, objects,
        # world traits, channels, propositions, concerns, events) when
        # the constraints or user request demand them, but every
        # introduction MUST carry a *concrete* ``justification`` that
        # names the existing candidates considered and why each was
        # insufficient. This violation fires when:
        #   * ``justification`` is empty / whitespace, OR
        #   * ``justification`` is generic boilerplate ("needed for
        #     the scene", "required by the prompt", "to advance the
        #     plot", "for narrative purposes", etc.), OR
        #   * an existing element with the same display name is
        #     already in ``WorldStateV1`` (the renderer is reinventing
        #     a referent that already exists).
        # Soft on the boilerplate axis (``major``) because the LLM
        # auditor remains responsible for paraphrase cases; hard
        # (``critical``) when the justification is missing entirely or
        # the name collides with an existing element.
        "unjustified_introduction",
        # AUDIT (post-2026-05-26): spurious_abduction \u2014 Rung-3 prose
        # introduces a *new* historical cause (a confession, a hidden
        # accomplice, an off-page event) that the engine never abduced.
        # Pearl Rung-3 abduction is monotone over the *engine's* U
        # ledger; the renderer cannot mint new exogenous antecedents
        # because the auditor's only ground truth for the past is the
        # engine's posterior. Violations of this kind silently fabricate
        # backstory and make the counterfactual unfalsifiable. Critical
        # by default \u2014 it breaks ctf-calculus Rule-3 (Exclusion)
        # for the introduced cause.
        "spurious_abduction",
        # AUDIT (post-2026-05-26): premature_payoff \u2014 directive prose
        # resolves a concern / proposition the brief left explicitly
        # open (``concern.activation_fabula_window`` still pending,
        # ``proposition.truth_at_fabula`` uncommitted at this anchor).
        # Authored payoffs that fire ahead of the engine schedule
        # collapse downstream suspense and contaminate the next merge
        # with a forced commit. Major by default; critical when the
        # affected proposition is part of an active SuspenseProfile.
        "premature_payoff",
        # AUDIT (round-12): world_trait_timeline_disorder \u2014 a
        # ``GlobalTrait.state_timeline`` whose snapshots are not in
        # monotonically non-decreasing ``fabula_time`` order. Fires
        # before the renderer is consulted: timeline disorder corrupts
        # ``reconstruct_world_trait_at`` replay so any ambient-force
        # grounding the prose draws on becomes unstable. Critical
        # \u2014 the engine's invariant is broken.
        "world_trait_timeline_disorder",
    ]
    severity: Literal["critical", "major", "minor"]
    description: str = Field(
        description="What went wrong — specific and actionable.",
    )
    evidence_quote: str = Field(
        default="",
        description="The exact passage from the prose that demonstrates the violation.",
    )
    feedback: str = Field(
        description=(
            "Explicit rewrite instruction for the generation LLM. "
            "Written as a direct command."
        ),
    )


class AuditResult(BaseModel):
    """Structured output of a single audit pass."""
    passed: bool = Field(
        description="True if all hard constraints are honoured and the target effect is achieved.",
    )
    violations: List[AuditViolation] = Field(default_factory=list)
    audit_summary: str = Field(
        default="",
        description="One-sentence summary of the overall audit result.",
    )
    failed_open: bool = Field(
        default=False,
        description="True if this result was produced by error fallback, not real audit.",
    )
    change_impact: Optional[ChangeImpactMetrics] = Field(
        default=None,
        description="Per-cycle engine-computed delta metrics. None if not computed.",
    )
    llm_raw_passed: Optional[bool] = Field(
        default=None,
        description=(
            "The LLM auditor's own verdict captured immediately after the "
            "model call and before any deterministic engine passes "
            "(position-mismatch, co-presence, etc.) append violations and "
            "flip ``passed``. The feedback loop uses this to distinguish "
            "'LLM passed but engine proxy fired a false positive' from "
            "'LLM itself found violations', preventing the 120-char "
            "proximity heuristic from blocking convergence on prose the "
            "LLM explicitly cleared."
        ),
    )
    llm_raw_violations: List[AuditViolation] = Field(
        default_factory=list,
        description=(
            "Snapshot of violations as returned by the LLM, captured at the "
            "same time as ``llm_raw_passed``, before deterministic engine "
            "passes append additional findings. Used by the minor-bypass "
            "logic so that engine-injected major violations (e.g. "
            "entity_position_mismatch) cannot prevent the bypass from "
            "firing when the LLM itself only flagged minor cosmetic issues."
        ),
    )


class AuditCycleSnapshot(BaseModel):
    """Immutable record of one audit-loop iteration."""
    iteration: int
    prose: str
    audit_result: AuditResult
    graph_version: int = Field(
        description=(
            "Monotonically increasing version of the world-state graph "
            "used during this cycle."
        ),
    )
    graph_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="nx.node_link_data snapshot of the versioned sandbox.",
    )
    change_impact: Optional[ChangeImpactMetrics] = Field(
        default=None,
        description="Per-cycle engine-computed delta metrics.",
    )
    engine_threshold_failures: List[str] = Field(
        default_factory=list,
        description=(
            "Round-9 F11: human-readable list of engine thresholds "
            "that did not pass for THIS cycle (recomputed from "
            "``change_impact`` and the active ``AuditorConfig``). "
            "Empty when the engine passed or no impact was scored. "
            "Persisted so per-iteration audit traces can report "
            "engine state on every row, not just the terminal one."
        ),
    )


class FeedbackLoopResult(BaseModel):
    """Final output of the full audit → refinement loop."""
    final_scene: GeneratedScene
    converged: bool = Field(
        description="True if the auditor passed before hitting max_iterations.",
    )
    iterations: int
    history: List[AuditCycleSnapshot] = Field(default_factory=list)
    final_graph_version: int
    change_impact: Optional[ChangeImpactMetrics] = Field(
        default=None,
        description="Per-cycle engine-computed delta metrics from the final cycle.",
    )
    correction_error: Optional[str] = Field(
        default=None,
        description=(
            "Populated when the loop exited because a refinement / rewrite "
            "LLM call raised, rather than because iterations were exhausted "
            "or convergence was reached. None on a normal exit."
        ),
    )
    engine_thresholds_passed: Optional[bool] = Field(
        default=None,
        description=(
            "Result of the deterministic engine-threshold check on the "
            "final cycle's ChangeImpactMetrics. None when no impact "
            "metrics were available to score."
        ),
    )
    engine_threshold_failures: List[str] = Field(
        default_factory=list,
        description="Human-readable list of engine thresholds that failed on the final cycle.",
    )


# =====================================================================
# Configuration
# =====================================================================

def _aud_defaults() -> dict:
    return _get_settings().auditor_config()


class AuditorConfig(BaseModel):
    """Runtime configuration for the audit/refinement loop."""
    auditor_model: str = Field(
        default="ollama:qwen3.6:35b",
        description="PydanticAI model string for the auditor LLM.",
    )
    generation_model: str = Field(
        default="ollama:qwen3.6:35b",
        description="PydanticAI model string for the generation LLM (re-renders).",
    )
    max_iterations: int = Field(
        default=4,
        ge=1,
        le=8,
        description=(
            "Maximum audit → rewrite cycles before giving up. Counts the "
            "initial render's audit as iteration 1, so default=4 yields "
            "up to 3 refinement attempts. The plausibility bypass "
            "(see ``run_feedback_loop``) fires after ceil(max_iterations/2) "
            "iterations when only soft-affective violations remain, so "
            "plausible prose typically exits after 2 cycles. "
            "Bounded to [1, 8] to keep wall-clock budgets sane."
        ),
    )
    output_retries: int = Field(
        default=5,
        description="Max retries for structured output parsing per LLM call.",
    )
    auditor_temperature: float = Field(
        default=0.2,
        description="Low temperature for deterministic auditing.",
    )
    generation_temperature: float = Field(
        default=0.7,
        description="Creative temperature for prose re-generation.",
    )
    max_tokens_audit: int = Field(
        default=8192,
        description=(
            "Maximum *output* tokens for the auditor LLM call. The audit "
            "result is structured JSON, not prose — 8 192 tokens is "
            "generous for any violations list and leaves the bulk of the "
            "256K context window for the (large) audit prompt input."
        ),
    )
    max_tokens_generation: int = Field(
        default=16000,
        description=(
            "Maximum *output* tokens for the auditor's prose re-write step. "
            "Mirrors GenerationConfig.max_tokens."
        ),
    )
    # --- Pass/fail thresholds for NarrativeOrderObject ---
    min_foreshadowing_score: float = Field(
        default=0.6,
        description="Minimum foreshadowing payoff score to pass.",
    )
    max_affective_loss: float = Field(
        default=0.3,
        description="Maximum affective loss MSE to pass.",
    )
    min_cognitive_plausibility: float = Field(
        default=0.7,
        description="Minimum cognitive plausibility ratio to pass.",
    )
    max_miracle_steps: int = Field(
        default=0,
        description=(
            "Maximum number of tolerated miracle steps before failing. "
            "0 preserves strict behavior. Mirrors the default in "
            "``AuditorSettings`` so direct construction matches the "
            "settings-driven path."
        ),
    )
    ignore_spatial_blocks: bool = Field(
        default=False,
        description=(
            "When true, blocked propagations tagged as spatial_affordance "
            "do not count toward miracle-step failure."
        ),
    )
    regression_retry_budget: int = Field(
        default=2,
        ge=0,
        description=(
            "How many anti-regression retries the feedback loop is "
            "allowed after the rollback detector fires. The previous "
            "behaviour (immediate break) corresponds to 0. Default "
            "1 grants one retry, which empirically recovers most "
            "regressions without burning the whole budget on a "
            "thrashing rewriter."
        ),
    )
    failed_open_tolerance: int = Field(
        default=2,
        ge=1,
        description=(
            "Number of consecutive failed-open audit calls tolerated "
            "before the loop refuses to mark prose as audited. Default "
            "2 mirrors the pre-round-7 hardcoded value."
        ),
    )
    enable_deterministic_prose_checks: bool = Field(
        default=True,
        description=(
            "When true, run the deterministic POV-lock and "
            "meta-narration regex checks "
            "(``deterministic_prose_findings``) alongside the LLM "
            "auditor and merge their findings into the violation list. "
            "This catches the two violation classes most likely to slip "
            "past the LLM auditor under rewrite pressure (round-7 "
            "audit 2026-05-26)."
        ),
    )
    pov_breach_threshold: int = Field(
        default=3,
        ge=1,
        description=(
            "Minimum number of non-POV cognitive/perceptual sentences "
            "the deterministic POV check must see before emitting a "
            "``reasoning_failure`` violation. Set higher to reduce "
            "false positives on ensemble scenes; set to 1 for strict "
            "single-consciousness rendering."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_from_settings(cls, data):
        if isinstance(data, dict):
            for k, v in _aud_defaults().items():
                data.setdefault(k, v)
        return data


# =====================================================================
# Dependencies
# =====================================================================

class _AuditorDeps(BaseModel):
    """Injected context for the auditor agent."""
    model_config = {"protected_namespaces": ()}
    audit_prompt: str = Field(
        description="The fully assembled audit prompt.",
    )


class _EvaluationDeps(BaseModel):
    """Injected context for the evaluation (literary critique) agent."""
    model_config = {"protected_namespaces": ()}
    evaluation_prompt: str = Field(
        description="The fully assembled evaluation prompt.",
    )


# =====================================================================
# Engine-Computed Metrics
# =====================================================================

def compute_causal_feedback(
    physics_result: Optional[CausalPhysicsResult],
    brief: CreativeBrief,
    world_state: Optional[WorldStateV1] = None,
) -> CausalPhysicsFeedback:
    """Derive causal physics metrics from engine data (no LLM call).

    Parameters
    ----------
    physics_result : CausalPhysicsResult or None
        The raw output of the causal physics simulation.
    brief : CreativeBrief
        The creative brief containing narrative tensions and epistemic gaps.
    world_state : WorldStateV1 or None
        The canonical world state (for resolving causal topology).
    """
    miracle_steps: List[str] = []
    cyclic_clusters: List[str] = []
    noisy_or_absorbed: List[str] = []
    rule3_pruned: List[str] = []
    rule2_redundant: List[str] = []
    foreshadowing_score = 1.0
    cog_plausibility_score = 1.0
    cog_details = ""
    # Total event count is used to normalise miracle-step penalties
    # below — a single block in a 5-event vignette should hurt more
    # than the same block in a 200-event saga.
    total_event_count = (
        len(world_state.events) if world_state is not None else 0
    )

    # --- Miracle steps: blocked propagations where impact failed inertia ---
    # Cycle blocks indicate static-topology extraction problems (the
    # causal subgraph contains an SCC), not narrative miracles, so they
    # are split out into ``cyclic_propagation_clusters`` and excluded
    # from miracle-step accounting.
    #
    # Noisy-OR absorptions are *expected* probabilistic dampening of
    # weak impulses by the configured per-edge gate — not narrative
    # miracles — and are routed into ``noisy_or_absorbed_propagations``
    # so the rewrite loop does not waste cycles trying to fix the
    # engine's correctly-absorbed sub-threshold nudges with prose
    # changes. Conflating them with inertia/spatial blocks was producing
    # spurious "miracle step" warnings on every plot run under the
    # default noisy-OR propagation mode (impact=0.04 < inertia=0.55,
    # etc. — the gate firing as designed, not a story problem).
    # Under noisy-OR propagation, plain ``reason="inertia"`` blocks are
    # the deterministic-fallback rendering of the same sub-threshold
    # absorption that the noisy-OR gate would have eaten on its own.
    # They are NOT narrative miracles (the rewriter cannot move them
    # via prose because the impulse never crossed the gate), and
    # treating them as veto-class violations was permanently pinning
    # ``engine_passed=False`` in the refinement loop. Route them into
    # ``noisy_or_absorbed_propagations`` whenever noisy-OR is the
    # active propagation mode so the loop's miracle-step accounting
    # mirrors the gate's actual semantics.
    try:
        _propagation_mode = _get_settings().physics.propagation_mode
    except Exception:
        _propagation_mode = "deterministic"
    _treat_inertia_as_absorbed = _propagation_mode == "noisy_or"
    # When the engine itself flagged the surgery inert (every requested
    # do-target Rule-3 pruned and/or every propagation absorbed by
    # cyclic SCC / inertia), the brief tells the renderer NOT to depict
    # any downstream consequence \u2014 see the
    # ``_build_inert_intervention_constraints`` HARD block. The blocks
    # the engine produced are *the correct outcome*, not narrative
    # miracles. Routing them to ``miracle_steps`` would pin
    # ``engine_passed=False`` for the entire refinement loop on
    # exactly the scenes where the prose was correctly inert.
    _inert = bool(
        physics_result is not None
        and getattr(physics_result, "intervention_inert", False)
    )

    if physics_result is not None:
        for b in physics_result.blocked:
            entry = (
                f"{b.node_id}.{b.trait}: impact={b.impact:.2f} < "
                f"inertia={b.inertia:.2f} ({b.reason})"
            )
            if b.reason == "cycle":
                cyclic_clusters.append(entry)
            elif b.reason == "noisy_or_absorbed":
                noisy_or_absorbed.append(entry)
            elif b.reason == "inertia" and (
                _treat_inertia_as_absorbed or _inert
            ):
                noisy_or_absorbed.append(entry)
            elif _inert:
                # Catch-all: any other block under an inert intervention
                # is also expected engine behaviour, not a miracle the
                # renderer is meant to repair.
                noisy_or_absorbed.append(entry)
            else:
                miracle_steps.append(entry)
        rule3_pruned = list(physics_result.rule3_pruned_interventions)
        rule2_redundant = list(physics_result.rule2_redundant_evidence)

    # --- Foreshadowing payoff: ratio of withheld narrative tensions resolved ---
    # A withheld_cause is "paid off" when one of its downstream
    # chain_reaction effects is REVEALED to the reader (i.e. the
    # effect event exists in the world AND its ``syuzhet_index``
    # appears earlier than the cause's reveal). The previous form
    # accepted *any* chain_reaction edge regardless of whether the
    # effect was on-page, so virtually every plot scored 1.0 — the
    # check did not actually verify pay-off, only causal fan-out.
    # See ``foreshadowing_arcs_data`` in shadow_loom_ui for the
    # equivalent syuzhet-aware logic the UI already uses.
    if brief.narrative_tensions:
        withheld = [
            nt for nt in brief.narrative_tensions
            if nt.tension_type == "withheld_cause"
        ]
        if withheld and world_state is not None:
            events_by_id = {e.id: e for e in world_state.events}
            withheld_by_id = {nt.event_id: nt for nt in withheld}
            resolved_ids: set[str] = set()
            for ce in world_state.causal_topology:
                if ce.causality_type != "chain_reaction":
                    continue
                if ce.source_id not in withheld_by_id:
                    continue
                effect = events_by_id.get(ce.target_id)
                if effect is None:
                    # Loose Chekhov's gun — effect was promised but
                    # never instantiated as an on-page event.
                    continue
                cause_nt = withheld_by_id[ce.source_id]
                # "Foreshadowing" requires the effect to have surfaced
                # before (or at) the cause's reveal — that is exactly
                # what makes it a set-up the reader can retro-fit.
                if effect.syuzhet_index <= cause_nt.syuzhet_index:
                    resolved_ids.add(ce.source_id)
            foreshadowing_score = (
                len(resolved_ids) / len(withheld_by_id)
                if withheld_by_id
                else 1.0
            )
        elif withheld:
            # No world state — score based on displacement magnitude
            avg_disp = sum(abs(nt.displacement) for nt in withheld) / len(withheld)
            foreshadowing_score = max(0.0, 1.0 - avg_disp)

    # --- Cognitive plausibility: penalise actual physical impossibilities
    # surfaced by the engine. Miracle-step blocks ARE the plausibility
    # signal we have access to deterministically — a state-change the
    # engine refused to derive is, by definition, an implausible jump.
    # Belief contradictions are dramatic irony (a feature of fiction)
    # so they enter only as informational ``cog_details``, never as a
    # numeric penalty. Previously this score was hard-coded to 1.0
    # which made one of the three hero tiles guaranteed-strong on
    # every report, biasing the verdict average upward.
    miracle_count = len(miracle_steps)
    if total_event_count > 0 and miracle_count > 0:
        cog_plausibility_score = max(
            0.0, 1.0 - miracle_count / total_event_count
        )
    else:
        cog_plausibility_score = 1.0

    if brief.epistemic_gaps:
        total = len(brief.epistemic_gaps)
        contradicted = sum(
            1 for g in brief.epistemic_gaps if g.gap_type == "contradicted"
        )
        if contradicted:
            cog_details = (
                f"{contradicted}/{total} entity beliefs are currently "
                f"contradicted by reality (dramatic irony — not a "
                f"plausibility violation by itself)."
            )
        else:
            cog_details = (
                f"All {total} tracked beliefs are consistent with reality."
            )
    if miracle_count > 0:
        miracle_note = (
            f"{miracle_count} miracle-step block(s) reduced plausibility "
            f"to {cog_plausibility_score:.2f}."
        )
        cog_details = f"{cog_details} {miracle_note}".strip()

    feedback = CausalPhysicsFeedback(
        miracle_steps_detected=miracle_steps,
        foreshadowing_payoff_score=round(foreshadowing_score, 4),
        cognitive_plausibility_score=cog_plausibility_score,
        cognitive_plausibility_details=cog_details,
        cyclic_propagation_clusters=cyclic_clusters,
        noisy_or_absorbed_propagations=noisy_or_absorbed,
        rule3_pruned_interventions=rule3_pruned,
        rule2_redundant_evidence=rule2_redundant,
    )
    log_agent_output(logger, "CausalPhysicsFeedback", feedback)
    return feedback


def compute_affective_feedback(
    brief: CreativeBrief,
    assembler: Optional[DirectiveAssembler],
    entity_ids: Optional[List[str]] = None,
) -> AffectiveStateFeedback:
    """Derive affective metrics from the DirectiveAssembler (no LLM call).

    Parameters
    ----------
    brief : CreativeBrief
        The creative brief (for target_effect and target_entities).
    assembler : DirectiveAssembler or None
        A configured assembler for computing scores.
    entity_ids : list[str] or None
        Override entity IDs (defaults to brief.target_entities).
    """
    eids = entity_ids or brief.target_entities
    trajectory_scores: Dict[str, float] = {}
    kl_divergence: Optional[float] = None
    affective_loss: Optional[float] = None

    if assembler is None or not eids:
        # No assembler OR no entities to score against → return a
        # "not measured" envelope rather than a misleading 0.0 (which
        # the UI would map to ``loss=0 → fit=strong``). Empty target
        # entities is a brief-quality issue, not a story-quality one.
        return AffectiveStateFeedback(
            emotional_trajectory_scores=trajectory_scores,
            kl_divergence_prediction_error=kl_divergence,
            affective_loss_mse=affective_loss,
        )

    # Compute the primary target effect score — but only when the
    # brief actually targets a measurable effect. Generic full-story
    # evaluations (target_effect == "observation") and unknown labels
    # have no defined "distance to target", and ``compute_affective_score``
    # falls into its no-contributions branch and returns ``+1.0`` —
    # the *worst possible* loss — which the UI then renders as a
    # spurious "poor fit" verdict on every quality report. Leave
    # ``affective_loss`` as ``None`` in that case so the hero tile
    # and findings card render "not measured" honestly.
    _MEASURABLE_TARGETS = (
        {"mystery", "dramatic_irony", "suspense", "surprise", "narrative_tension"}
        | set(_EFFECT_TRAITS.keys())
    )
    target = brief.target_effect
    if target in _MEASURABLE_TARGETS:
        try:
            affective_loss = assembler.compute_affective_score(target, eids)
        except Exception:
            logger.debug("[AffectiveFeedback] compute_affective_score failed for %s", target)
    else:
        logger.debug(
            "[AffectiveFeedback] target_effect=%r is not measurable; "
            "leaving affective_loss_mse=None (trajectory scores still computed).",
            target,
        )

    # Compute trajectory scores for all structural effects
    score_map = {
        "mystery": "compute_mystery_score",
        "dramatic_irony": "compute_dramatic_irony_score",
        "suspense": "compute_suspense_score",
        "surprise": "compute_surprise_score",
        "narrative_tension": "compute_tension_score",
    }
    for effect, method_name in score_map.items():
        try:
            method = getattr(assembler, method_name)
            trajectory_scores[effect] = round(method(eids), 4)
        except Exception:
            logger.debug("[AffectiveFeedback] %s failed", method_name)
    
    # P1-FIX: Add 6 emotion scorers (fear, joy, regret, grief, rage, love)
    emotion_map = {
        "fear": "compute_fear_score",
        "joy": "compute_joy_score",
        "regret": "compute_regret_score",
        "grief": "compute_grief_score",
        "rage": "compute_rage_score",
        "love": "compute_love_score",
    }
    for emotion, method_name in emotion_map.items():
        try:
            method = getattr(assembler, method_name, None)
            if method:
                trajectory_scores[emotion] = round(method(eids), 4)
        except Exception:
            logger.debug("[AffectiveFeedback] %s failed", method_name)

    # KL divergence is only meaningful when the brief actually targets
    # surprise — the field's docstring says "divergence between the
    # reader's prior expectation and actual revelation", which
    # ``compute_surprise_score`` computes. For other targets the
    # surprise-trajectory value is recorded in
    # ``emotional_trajectory_scores['surprise']`` (still useful) but
    # mislabelling it as a prediction-error here was confusing.
    if target == "surprise" and "surprise" in trajectory_scores:
        kl_divergence = trajectory_scores["surprise"]

    feedback = AffectiveStateFeedback(
        emotional_trajectory_scores=trajectory_scores,
        kl_divergence_prediction_error=kl_divergence,
        affective_loss_mse=(
            round(affective_loss, 4) if affective_loss is not None else None
        ),
    )
    log_agent_output(
        logger,
        f"AffectiveStateFeedback[target={brief.target_effect}, entities={eids}]",
        feedback,
    )
    return feedback


def compute_overall_pass(
    narrative_order: NarrativeOrderObject,
    config: AuditorConfig,
) -> bool:
    """Apply threshold checks to determine overall_pass."""
    cf = narrative_order.causal_feedback
    af = narrative_order.affective_feedback

    if cf.foreshadowing_payoff_score < config.min_foreshadowing_score:
        return False
    if cf.cognitive_plausibility_score < config.min_cognitive_plausibility:
        return False
    # ``affective_loss_mse`` may be ``None`` when the brief has no
    # target entities to score against — treat that as "not measured"
    # rather than "failed".
    if (
        af.affective_loss_mse is not None
        and af.affective_loss_mse > config.max_affective_loss
    ):
        return False
    if _count_miracle_step_failures(
        cf.miracle_steps_detected,
        ignore_spatial_blocks=config.ignore_spatial_blocks,
    ) > config.max_miracle_steps:
        return False
    return True


def _count_miracle_step_failures(
    miracle_steps: List[str],
    *,
    ignore_spatial_blocks: bool,
) -> int:
    """Count miracle-step failures under configurable filtering rules."""
    if not miracle_steps:
        return 0
    if not ignore_spatial_blocks:
        return len(miracle_steps)
    return sum(1 for s in miracle_steps if "(spatial_affordance)" not in s)


def _engine_thresholds_check(
    impact: Optional[ChangeImpactMetrics],
    config: AuditorConfig,
) -> Tuple[Optional[bool], List[str]]:
    """Apply :func:`compute_overall_pass` thresholds to per-cycle impact.

    Mirrors the full-story scorecard logic but operates on
    :class:`ChangeImpactMetrics` so the refinement loop can gate
    convergence on deterministic engine output rather than only the
    LLM auditor's self-report.

    Returns
    -------
    (passed, failures)
        ``passed`` is ``None`` when no impact was computed, otherwise
        a bool. ``failures`` is a list of human-readable threshold
        names that did not pass (empty when ``passed`` is True or None).
    """
    if impact is None:
        return None, []

    cf = impact.causal_feedback
    af = impact.affective_feedback
    failures: list[str] = []

    if cf is not None:
        if cf.foreshadowing_payoff_score < config.min_foreshadowing_score:
            failures.append(
                f"foreshadowing_payoff_score={cf.foreshadowing_payoff_score:.2f} "
                f"< min={config.min_foreshadowing_score:.2f}"
            )
        if cf.cognitive_plausibility_score < config.min_cognitive_plausibility:
            failures.append(
                f"cognitive_plausibility_score={cf.cognitive_plausibility_score:.2f} "
                f"< min={config.min_cognitive_plausibility:.2f}"
            )
        miracle_failures = _count_miracle_step_failures(
            cf.miracle_steps_detected,
            ignore_spatial_blocks=config.ignore_spatial_blocks,
        )
        if miracle_failures > config.max_miracle_steps:
            failures.append(
                "miracle_steps_detected="
                f"{list(cf.miracle_steps_detected)} "
                f"(counted={miracle_failures}, allowed={config.max_miracle_steps}, "
                f"ignore_spatial_blocks={config.ignore_spatial_blocks})"
            )

    if af is not None:
        if (
            af.affective_loss_mse is not None
            and af.affective_loss_mse > config.max_affective_loss
        ):
            failures.append(
                f"affective_loss_mse={af.affective_loss_mse:.3f} "
                f"> max={config.max_affective_loss:.3f}"
            )

    return (len(failures) == 0), failures


# =====================================================================
# Graph Versioning
# =====================================================================

class VersionedGraph:
    """Manages deep-copied, versioned snapshots of the sandbox graph.

    Each call to :meth:`fork` returns a fresh deep-copy of the current
    graph and increments the version counter.  The original graph passed
    to __init__ is stored as version 0 and is **never mutated**.
    """

    def __init__(self, sandbox: nx.MultiDiGraph) -> None:
        self._original = copy.deepcopy(sandbox)
        self._current = copy.deepcopy(sandbox)
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def current(self) -> nx.MultiDiGraph:
        """The current working copy (may be mutated by the physics engine)."""
        return self._current

    @property
    def original(self) -> nx.MultiDiGraph:
        """The immutable version-0 snapshot."""
        return self._original

    def fork(self) -> nx.MultiDiGraph:
        """Create a new deep-copy for the next iteration and bump the version."""
        self._version += 1
        self._current = copy.deepcopy(self._original)
        logger.debug(
            "[VersionedGraph] Forked to version %d (%d nodes, %d edges).",
            self._version, self._current.number_of_nodes(),
            self._current.number_of_edges(),
        )
        return self._current

    def snapshot_data(self) -> Dict[str, Any]:
        """Return nx.node_link_data of the current graph for serialisation."""
        return nx.node_link_data(self._current)


# =====================================================================
# Prompt Assembly
# =====================================================================

def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _format_constraints_for_audit(constraints: List[ConstraintBlock]) -> str:
    """Format constraints compactly for the auditor's reference."""
    if not constraints:
        return "(No constraints.)"
    lines: List[str] = []
    for i, c in enumerate(constraints, 1):
        priority_tag = "HARD" if c.priority == "hard" else "SOFT"
        lines.append(
            f"  {i}. [{priority_tag} | {c.constraint_type.upper()}] "
            f"{c.instruction}"
        )
    return "\n".join(lines)


def _format_affective_metrics_block(
    affective_feedback: Optional[AffectiveStateFeedback],
) -> List[str]:
    """Render the engine-grade affective ledger as prompt lines.

    Shared by :func:`assemble_audit_prompt` (refinement loop) and
    :func:`assemble_evaluation_prompt` (quality report) so both LLM
    surfaces see the *same* numbers under the *same* labels. Without a
    single source of truth for this block the two prompts drift out
    of step — the evaluator could see ``Affective loss MSE: 0.42`` on
    a scene whose audit cycle never showed the auditor any affective
    score at all, producing self-contradictory quality verdicts.
    """
    if affective_feedback is None:
        return []
    lines: List[str] = ["=== ENGINE: AFFECTIVE METRICS (ground truth) ==="]
    for effect, score in affective_feedback.emotional_trajectory_scores.items():
        lines.append(f"  {effect}: {score:.4f}")
    if affective_feedback.kl_divergence_prediction_error is not None:
        lines.append(
            f"  KL divergence (surprise): "
            f"{affective_feedback.kl_divergence_prediction_error:.4f}"
        )
    # ``affective_loss_mse`` is ``None`` when the brief had no
    # measurable target (e.g. observation queries without entities to
    # score against). Surface as "not measured" rather than 0.0 so the
    # LLM does not penalise prose against a metric that did not run.
    if affective_feedback.affective_loss_mse is not None:
        lines.append(
            f"  Affective loss MSE: "
            f"{affective_feedback.affective_loss_mse:.4f}"
        )
    else:
        lines.append(
            "  Affective loss MSE: not measured "
            "(no scorable target — ignore in evaluation)"
        )
    lines.append("")
    return lines


def _format_propositional_context(brief: CreativeBrief) -> List[str]:
    """Surface propositional / belief / concern context to the auditor.

    The renderer (``shadow_loom.generation``) already emits dedicated
    prompt blocks for ``surprise_profile``, ``irony_profile``,
    ``mystery_profile`` and the per-emotion appraisal payloads, plus
    the ``affected_propositions`` / ``affected_concerns`` /
    ``affected_beliefs`` lists carried on
    :class:`ThreatProximity` (Rung-2) and
    :class:`CounterfactualBranch` (Rung-3). The auditor must see the
    same data, otherwise it can flag prose against trait/belief gaps
    while staying blind to whether the rendered scene respects the
    propositional commitments and concern polarities the directive
    was assembled around. This helper renders only the populated
    fields so audits on briefs without a Phase A3 catalogue degrade
    gracefully to the legacy trait-anchored picture.
    """
    out: List[str] = []

    sp = getattr(brief, "surprise_profile", None)
    if sp is not None and (sp.revealed_proposition_ids or sp.score):
        out.append(
            "=== SURPRISE PROFILE (audience-belief revision — "
            "propositions whose audience prior just shifted) ==="
        )
        out.append(
            f"  total KL: {sp.score:.3f}  "
            f"pleasant: {sp.pleasant_score:.3f}  "
            f"unpleasant: {sp.unpleasant_score:.3f}"
        )
        descs = list(sp.revealed_descriptions or [])
        for i, pid in enumerate(sp.revealed_proposition_ids[:8]):
            desc = descs[i] if i < len(descs) else ""
            out.append(f"    - {pid}: {desc}")
        if sp.per_focal_score:
            top = sorted(
                sp.per_focal_score.items(), key=lambda kv: -kv[1]
            )[:5]
            out.append(
                "  per-focal (concern-weighted): "
                + ", ".join(f"{eid}={s:.2f}" for eid, s in top)
            )
        out.append(
            "  Rule: prose must render the on-page consequence of "
            "these belief shifts; do NOT have the focal voice the "
            "shift directly (no 'she suddenly realised...')."
        )
        out.append("")

    ip = getattr(brief, "irony_profile", None)
    if ip is not None and (
        ip.audience_advantage_score
        or ip.focal_advantage_score
        or ip.audience_advantage_propositions
        or ip.focal_advantage_propositions
    ):
        out.append(
            "=== IRONY PROFILE (audience vs focal belief gap) ==="
        )
        out.append(
            f"  focal: {ip.focal_id}  "
            f"audience-advantage KL: {ip.audience_advantage_score:.3f}  "
            f"focal-advantage KL: {ip.focal_advantage_score:.3f}"
        )
        if ip.audience_advantage_propositions:
            out.append("  audience knows (focal does NOT):")
            for desc in ip.audience_advantage_propositions[:6]:
                out.append(f"    - {desc}")
        if ip.focal_advantage_propositions:
            out.append("  focal knows (audience does NOT — keep hidden):")
            for desc in ip.focal_advantage_propositions[:6]:
                out.append(f"    - {desc}")
        if ip.most_ironised_entity_id and ip.most_ironised_entity_id != ip.focal_id:
            out.append(
                f"  most-ironised entity: {ip.most_ironised_entity_id} "
                f"(KL={ip.most_ironised_score:.2f})"
            )
        out.append(
            "  Rule: the focal MUST act consistent with their "
            "(false) belief state on every audience-advantage "
            "proposition. Flag prose where the focal silently "
            "absorbs an audience-advantage fact without an on-page "
            "trigger."
        )
        out.append("")

    mp = getattr(brief, "mystery_profile", None)
    if mp is not None and (mp.score or mp.open_questions):
        out.append(
            "=== MYSTERY PROFILE (open erotetic questions — known "
            "effects with hidden causes) ==="
        )
        out.append(
            f"  score: {mp.score:.3f}  "
            f"plot-gap: {mp.plot_gap_score:.3f}  "
            f"character-gap: {mp.character_gap_score:.3f}"
        )
        if mp.governing_question_description:
            out.append(
                f"  governing question: "
                f"{mp.governing_question_description}"
            )
        for q in (mp.open_questions or [])[:6]:
            out.append(f"    - {q}")
        for q in (mp.character_gap_descriptions or [])[:4]:
            out.append(f"    ~ {q}")
        out.append(
            "  Rule: the prose MUST render the effects on-page "
            "without naming or implying the hidden cause."
        )
        out.append("")

    # Rung-2 / Rung-3 surgery side-effects on the propositional /
    # concern / belief layer. These are populated only on briefs
    # built from typed DoTarget queries; the legacy event-only path
    # leaves the lists empty and the block is suppressed.
    for label, payload in (
        ("RUNG-2 INTERVENTION SIDE-EFFECTS", brief.threat_proximity),
        ("RUNG-2 INTERVENTION SIDE-EFFECTS", brief.intervention_branch),
        ("RUNG-3 COUNTERFACTUAL SIDE-EFFECTS", brief.counterfactual_branch),
    ):
        if payload is None:
            continue
        ap = list(getattr(payload, "affected_propositions", []) or [])
        ab = list(getattr(payload, "affected_beliefs", []) or [])
        ac = list(getattr(payload, "affected_concerns", []) or [])
        ao = list(getattr(payload, "affected_objects", []) or [])
        awt = list(getattr(payload, "affected_world_traits", []) or [])
        ae = list(getattr(payload, "affected_edges", []) or [])
        # Engine-emitted downstream cascade rails. The renderer
        # surfaces these in the brief; the auditor must enumerate
        # them too so prose that silently drops a prop relocation /
        # ambient-force shift / topology rewrite is flagged as a
        # miracle step rather than passed by a content-blind audit.
        dtc = list(getattr(payload, "downstream_trait_changes", []) or [])
        drc = list(getattr(payload, "downstream_relationship_changes", []) or [])
        pcd = list(getattr(payload, "proposition_cascade_detail", []) or [])
        bcd = list(getattr(payload, "belief_cascade_detail", []) or [])
        ccd = list(getattr(payload, "concern_cascade_detail", []) or [])
        ocd = list(getattr(payload, "object_cascade_detail", []) or [])
        wtcd = list(getattr(payload, "world_trait_cascade_detail", []) or [])
        ecd = list(getattr(payload, "edge_cascade_detail", []) or [])
        edcd = list(getattr(payload, "entity_delete_cascade_detail", []) or [])
        odcd = list(getattr(payload, "object_delete_cascade_detail", []) or [])
        aed = list(getattr(payload, "affected_entity_deletes", []) or [])
        aod = list(getattr(payload, "affected_object_deletes", []) or [])
        bpd = list(getattr(payload, "blocked_propagations_detail", []) or [])
        if not (ap or ab or ac or ao or awt or ae
                or dtc or drc or pcd or bcd or ccd
                or ocd or wtcd or ecd or edcd or odcd
                or aed or aod or bpd):
            continue
        out.append(f"=== {label} (typed DoTarget surgery) ===")
        if ap:
            out.append(
                "  PROPOSITIONS whose truth flipped: "
                + _annotate_ids(
                    ap[:12],
                    (getattr(payload, "affected_proposition_descriptions", None) or [])[:12],
                )
            )
        if ab:
            out.append(
                "  BELIEFS whose confidence shifted "
                "(holder\u2192target): " + _annotate_ids(
                    ab[:12],
                    (getattr(payload, "affected_belief_descriptions", None) or [])[:12],
                )
            )
        if ac:
            out.append(
                "  CONCERNS whose polarity / salience shifted: "
                + _annotate_ids(
                    ac[:12],
                    (getattr(payload, "affected_concern_descriptions", None) or [])[:12],
                )
            )
        if ao:
            out.append(
                "  OBJECTS relocated / owner-changed / property-set: "
                + ", ".join(ao[:12])
            )
        if awt:
            out.append(
                "  WORLD TRAITS clamped (ambient-force shifts): "
                + ", ".join(awt[:12])
            )
        if ae:
            out.append(
                "  EDGES rewritten (edge_type:action:source\u2192target): "
                + ", ".join(ae[:12])
            )
        if aed:
            out.append(
                "  ENTITIES EXCISED (existence-counterfactual \u2014 must NOT appear in prose): "
                + ", ".join(aed[:12])
            )
        if aod:
            out.append(
                "  OBJECTS EXCISED (existence-counterfactual \u2014 must NOT be referenced): "
                + ", ".join(aod[:12])
            )
        if dtc:
            out.append("  DOWNSTREAM TRAIT CASCADE:")
            for ln in dtc[:8]:
                out.append(f"    - {ln}")
        if drc:
            out.append("  DOWNSTREAM RELATIONSHIP CASCADE:")
            for ln in drc[:8]:
                out.append(f"    - {ln}")
        if pcd:
            out.append("  PROPOSITION CASCADE DETAIL:")
            for ln in pcd[:8]:
                out.append(f"    - {ln}")
        if bcd:
            out.append("  BELIEF CASCADE DETAIL:")
            for ln in bcd[:8]:
                out.append(f"    - {ln}")
        if ccd:
            out.append("  CONCERN CASCADE DETAIL:")
            for ln in ccd[:8]:
                out.append(f"    - {ln}")
        if ocd:
            out.append("  OBJECT CASCADE DETAIL:")
            for ln in ocd[:8]:
                out.append(f"    - {ln}")
        if wtcd:
            out.append("  WORLD-TRAIT CASCADE DETAIL:")
            for ln in wtcd[:8]:
                out.append(f"    - {ln}")
        if ecd:
            out.append("  EDGE CASCADE DETAIL:")
            for ln in ecd[:8]:
                out.append(f"    - {ln}")
        if edcd:
            out.append("  ENTITY DELETION CASCADE DETAIL:")
            for ln in edcd[:8]:
                out.append(f"    - {ln}")
        if odcd:
            out.append("  OBJECT DELETION CASCADE DETAIL:")
            for ln in odcd[:8]:
                out.append(f"    - {ln}")
        if bpd:
            out.append("  BLOCKED PROPAGATIONS (must dramatise resistance, not skip):")
            for ln in bpd[:8]:
                out.append(f"    - {ln}")
        tragedy = getattr(payload, "tragedy_form", None)
        if tragedy:
            out.append(f"  tragedy_form: {tragedy}")
        out.append(
            "  Rule: every flipped proposition / belief / concern / "
            "object / world-trait / edge above MUST be visibly "
            "grounded in an on-page event or utterance; flag silent "
            "off-page changes as miracle steps. Cascade detail lines "
            "(downstream trait/relationship/proposition/belief/concern/"
            "object/world-trait/edge) must also be reflected in prose "
            "\u2014 a beat per propagated effect, not a single summary "
            "sentence. Blocked propagations require a visible "
            "resistance beat, never a near-change."
        )
        out.append("")

    # Character-felt emotion appraisals — surface the concern-level
    # diagnostics so the auditor can validate prose against the
    # appraisal that drove the directive's stylistic instructions.
    emo_blocks = (
        ("FEAR APPRAISAL", brief.fear_profile, [
            ("object_fear", "object_fear_score"),
            ("anxiety", "anxiety_score"),
            ("coping", "coping_score"),
            ("flight_available", "flight_available"),
            ("dread", "dread"),
            ("primary_concern", "primary_concern_description"),
        ]),
        ("JOY APPRAISAL", brief.joy_profile, [
            ("own_joy", "own_joy_score"),
            ("happy_for", "happy_for_score"),
            ("gloating", "gloating_score"),
            ("relief", "relief_score"),
            ("primary_concern", "primary_concern_description"),
        ]),
        ("REGRET APPRAISAL", brief.regret_profile, [
            ("agentive_regret", "agentive_regret_score"),
            ("disappointment", "disappointment_score"),
            ("commission", "commission_score"),
            ("omission", "omission_score"),
            ("downward_relief", "downward_relief_score"),
            ("mode", "mode"),
            ("divergence_event", "divergence_description"),
            ("loss_event", "loss_description"),
        ]),
        ("GRIEF APPRAISAL", brief.grief_profile, [
            ("coupling_strength", "coupling_strength"),
            ("stage", "stage"),
            ("unfinished_concerns", "unfinished_concern_count"),
            ("lost_entity", "lost_entity_id"),
            ("loss_event", "loss_description"),
        ]),
        ("RAGE APPRAISAL", brief.rage_profile, [
            ("blocked_concern", "blocked_concern_score"),
            ("attribution_clarity", "attribution_clarity"),
            ("perpetrator", "perpetrator_id"),
            ("perpetrator_proximity", "perpetrator_proximity"),
            ("normative_violation", "normative_violation"),
            ("mode", "mode"),
        ]),
        ("LOVE APPRAISAL", brief.love_profile, [
            ("partner", "primary_partner_id"),
            ("intimacy", "intimacy_score"),
            ("passion", "passion_score"),
            ("commitment", "commitment_score"),
            ("style", "style"),
        ]),
    )
    for label, payload, fields in emo_blocks:
        if payload is None:
            continue
        parts: List[str] = []
        for human, attr in fields:
            v = getattr(payload, attr, None)
            if v is None or v == "":
                continue
            if isinstance(v, bool):
                parts.append(f"{human}={'yes' if v else 'no'}")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                parts.append(f"{human}={float(v):.2f}")
            else:
                parts.append(f"{human}={v}")
        if not parts:
            continue
        out.append(f"=== {label} ===")
        focal = getattr(payload, "focal_id", None)
        if focal:
            out.append(f"  focal: {focal}")
        out.append("  " + "  ".join(parts))
        cids = list(getattr(payload, "contributing_concern_ids", []) or [])
        if cids:
            out.append(
                "  driving concerns: " + ", ".join(cids[:8])
            )
        out.append("")

    return out


def assemble_audit_prompt(
    prose: str,
    brief: CreativeBrief,
    audit_categories: List[str],
    prior_feedback: Optional[List[str]] = None,
    causal_feedback: Optional[CausalPhysicsFeedback] = None,
    *,
    world_state: Optional[WorldStateV1] = None,
    affective_feedback: Optional[AffectiveStateFeedback] = None,
    introduced_elements: Optional["IntroducedElements"] = None,
) -> str:
    """Build the full prompt for the auditor LLM.

    Parameters
    ----------
    prose : str
        The rendered prose to audit.
    brief : CreativeBrief
        The mathematical constraints the prose must satisfy.
    audit_categories : list[str]
        Which audit categories to run (epistemic, probabilistic,
        counterfactual, physics).
    prior_feedback : list[str] or None
        Feedback from previous iterations (for inner-loop context).
    """
    sections: List[str] = []

    sections.append("=== PROSE TO AUDIT ===")
    sections.append(prose)
    sections.append("")

    # === Story so far (continuity context the generator saw) ===
    # Without this the auditor can raise false-positive continuity /
    # voice / thread judgments on prose the renderer was actually
    # instructed to continue from earlier versions.
    # Capped to the same ``preceding_prose_max_chars`` limit as the
    # renderer so the auditor sees an identical (tail-truncated) slice.
    if brief.preceding_prose:
        _pp_max = _get_settings().generation.preceding_prose_max_chars
        _pp = brief.preceding_prose.strip()
        if len(_pp) > _pp_max:
            _pp = "…" + _pp[-_pp_max:]
        # Shadow-branch audits: the joined tail mixes factual-canon
        # ancestors with the shadow continuation (see
        # ``_gather_preceding_prose``). Without a precedence rule the
        # auditor would flag any deliberate divergence-from-canon
        # (suppressed events, removed utterances, flipped
        # propositions) as a continuity violation, because the
        # factual prose blocks still narrate the original outcome.
        if brief.branch_world_id == "shadow":
            sections.append("=== STORY SO FAR (mixed: factual canon + shadow fork tail \u2014 do NOT re-audit) ===")
            sections.append(_pp)
            sections.append(
                "Blocks tagged ``(factual: \u2026)`` are the canon the shadow "
                "fork diverges FROM \u2014 they are NOT in force on this "
                "branch. Do NOT flag the PROSE TO AUDIT for contradicting "
                "factual details that the engine has explicitly "
                "superseded (a suppressed event, a removed utterance, a "
                "flipped proposition). Use the factual blocks only to "
                "judge tone / voice consistency; the BRANCH CONTEXT and "
                "the world state are authoritative on what is true on "
                "this branch. Blocks tagged ``(shadow: \u2026)`` are this "
                "fork's prior continuation and ARE in force for "
                "continuity judgments. Do NOT raise violations against "
                "the STORY SO FAR text itself \u2014 it is fixed history."
            )
        else:
            sections.append("=== STORY SO FAR (background continuity \u2014 do NOT re-audit) ===")
            sections.append(_pp)
            sections.append(
                "The prose above is the established narrative this scene "
                "continues from. Use it only to judge continuity / tone / "
                "voice consistency in the PROSE TO AUDIT above. Do NOT "
                "raise violations against the STORY SO FAR text itself \u2014 "
                "it is fixed history."
            )
        sections.append("")

    # === Branch context (AMWN shadow vs factual) ===
    # Mirrors the renderer prompt's BRANCH CONTEXT block so the auditor
    # judges the prose against the same canon-vs-fork framing the
    # generator was given. Without this the auditor can flag silent
    # divergences from canon as continuity errors.
    if brief.branch_world_id == "shadow":
        sections.append("=== BRANCH CONTEXT (background only) ===")
        sections.append("branch_world_id: shadow")
        if brief.branch_label:
            sections.append(f"branch_label: {brief.branch_label}")
        if brief.factual_contrast_summary:
            sections.append("factual_mainline_at_same_horizon:")
            sections.append(brief.factual_contrast_summary.strip())
        sections.append(
            "The prose lives on a shadow fork. It is expected to diverge "
            "from the factual mainline above; do NOT flag silent "
            "divergence from canon as a continuity error. Do flag the "
            "prose if it explicitly names the branch / mainline / "
            "contrast in the narration (Rule 10 in generation.md)."
        )
        sections.append("")

    # The user's verbatim request is part of the audit contract: prose
    # that ignores or contradicts what the user asked for is itself a
    # violation, even when all engine-derived constraints pass.
    if brief.original_query:
        sections.append("=== USER'S ORIGINAL REQUEST (verbatim) ===")
        sections.append(brief.original_query.strip())
        sections.append(
            "Flag the prose if it fails to address this request, even "
            "when no other constraint is violated."
        )
        sections.append("")

    # Source-style fidelity contract: the prose SHOULD match the form
    # of the source text. The band is loosened in the direction the
    # user query asks for ("in detail" widens, "briefly" tightens),
    # and only drifts >50% outside the loosened band are flagged as
    # `style_mismatch` violations.
    if brief.narrative_style is not None:
        from shadow_loom.narrative_style import (
            adjusted_word_band, format_narrative_style_block,
        )
        adj_min, adj_max, _intent = adjusted_word_band(
            brief.narrative_style, brief.original_query,
        )
        sections.append(format_narrative_style_block(
            brief.narrative_style,
            original_query=brief.original_query,
        ))
        sections.append(
            f"  Word-count gate: count the words in the prose above. "
            f"Only raise a `style_mismatch` violation when the count "
            f"falls outside the loosened band ({adj_min}\u2013{adj_max} "
            f"words) by more than \u00b150%."
        )
        sections.append(
            "  Severity rules for `style_mismatch`:\n"
            "    \u2022 `critical` \u2014 reserved for genuine "
            "form-class breaches that destroy the source register "
            "(e.g.\u00a0a `news_article` rendered as fictional scene "
            "work, a `synopsis` rendered as a 2,000-word short story, "
            "a `transcript` rendered as continuous narration). Always "
            "trigger a regeneration.\n"
            "    \u2022 `major` \u2014 reserved for word-count breaches "
            "outside the \u00b150% loosened band, OR a `prose_density` "
            "drift that has *also* dragged the prose across a form "
            "boundary. Trigger a regeneration.\n"
            "    \u2022 `minor` \u2014 default for everything else: "
            "pure prose-density drift (slightly too lush or too "
            "telegraphic) when the word count is inside the band and "
            "the form-class is intact. The refinement loop should NOT "
            "spend a regeneration cycle on this alone."
        )
        sections.append("")

    sections.append(f"=== TARGET EFFECT: {brief.target_effect.upper()} ===")
    sections.append(f"Target entities: {', '.join(brief.target_entities)}")
    sections.append("")

    sections.append("=== CONSTRAINTS (from Creative Brief) ===")
    sections.append(_format_constraints_for_audit(brief.constraints))
    sections.append("")

    # === Rendering directive (the same stylistic control the renderer
    # was given). Without this the auditor cannot validate POV-lock
    # breaches, pacing drift, or sensory-focus violations — it only
    # ever saw the constraints and reverse-engineered intent from
    # them. The directive is reference data: do NOT raise a violation
    # for the directive itself, only for prose that fails to honour it.
    if brief.rendering:
        r = brief.rendering
        sections.append(
            "=== RENDERING DIRECTIVE (the stylistic control the "
            "renderer was given \u2014 prose must honour it) ==="
        )
        sections.append(f"  rendering_mode: {r.rendering_mode}")
        sections.append(f"  pacing: {r.pacing}")
        sections.append(f"  sensory_focus: {r.sensory_focus}")
        if r.pov_lock:
            if r.pov_policy == "single" or not r.additional_pov_locks:
                sections.append(
                    f"  pov_lock: {r.pov_lock} \u2014 the prose MUST stay "
                    f"inside this entity's perception. Flag head-hopping "
                    f"or omniscient narration as a violation under the "
                    f"`physics` category (rationale prefix `pov_lock:`)."
                )
            else:
                roster = ", ".join([r.pov_lock, *r.additional_pov_locks])
                if r.pov_policy == "rotating":
                    sections.append(
                        f"  pov_policy: rotating \u2014 the prose MUST "
                        f"stay inside one of these entities' "
                        f"perceptions per beat (primary: {r.pov_lock}; "
                        f"roster: {roster}). Head-hopping *within* a "
                        f"single beat is a violation; rotating between "
                        f"beats is licensed. Flag violations under the "
                        f"`physics` category (rationale prefix "
                        f"`pov_lock:`)."
                    )
                else:  # ensemble
                    sections.append(
                        f"  pov_policy: ensemble (omniscient-"
                        f"constrained) \u2014 interiority is licensed "
                        f"across {roster} (primary: {r.pov_lock}). "
                        f"Entities OUTSIDE this roster remain "
                        f"externally observed; rendering their "
                        f"interiority is a violation under the "
                        f"`physics` category (rationale prefix "
                        f"`pov_lock:`)."
                    )
        if r.tone_arc:
            sections.append(f"  tone_arc: {r.tone_arc}")
        if r.stylistic_instructions:
            sections.append("  stylistic_instructions:")
            for j, si in enumerate(r.stylistic_instructions, 1):
                sections.append(f"    {j}. {si}")
        if r.pov_lock:
            sections.append(
                "  POV-temporal rule: the locked / rostered POV "
                "entity / entities know ONLY what they had "
                "perceived by the scene's fabula_time. Flag any "
                "interior thought or remembered detail that "
                "references events with fabula_time strictly "
                "GREATER than the scene anchor as a violation "
                "under the `physics` category (rationale prefix "
                "`pov_future_knowledge:`). A POV character "
                "\"remembering\" something that has not yet "
                "happened in their subjective timeline is the "
                "signature failure mode."
            )
        sections.append("")

    # === Physics override (engine-authored hard text the renderer was
    # told to honour verbatim). Mirrors the PHYSICS OVERRIDE block in
    # the rendering prompt so the auditor can flag prose that ignored
    # or contradicted it.
    if brief.physics_override:
        sections.append("=== PHYSICS OVERRIDE (HARD \u2014 prose must honour) ===")
        sections.append(brief.physics_override.strip())
        sections.append(
            "Flag the prose as a `reasoning_failure` violation if it "
            "ignores or contradicts the override above."
        )
        sections.append("")

    # === Utterance & channel fidelity (HARD)
    # The renderer received a HARD `UTTERANCE & CHANNEL FIDELITY` block
    # listing every withheld channel and every withheld utterance with
    # its `discovered_at_syuzhet`. Mirror it here so the auditor judges
    # against the same brief-level withheld set rather than only the
    # world-state-derived view further down.
    if brief.hidden_channels:
        sections.append(
            "=== HIDDEN CHANNELS / UTTERANCES (HARD \u2014 must NOT "
            "surface in the prose) ==="
        )
        for hc in brief.hidden_channels:
            if hc.kind == "channel":
                sections.append(
                    f"  - HIDDEN CHANNEL {hc.channel_id} ({hc.medium}, "
                    f"participants={hc.participant_ids}): exists in the "
                    f"world but the reader has not seen any utterance "
                    f"on it. Flag prose that names it, quotes from it, "
                    f"or implies its presence."
                )
            else:
                sections.append(
                    f"  - HIDDEN UTTERANCE {hc.utterance_event_id} "
                    f"({hc.medium} from {hc.speaker_id} to "
                    f"{hc.addressee_ids} at syuzhet="
                    f"{hc.discovered_at_syuzhet}): the message itself "
                    f"comes later in narration order. Flag prose that "
                    f"reveals its content (verbatim or paraphrased)."
                )
        sections.append(
            "  Violation type: `withheld_utterance_leak` for utterance "
            "leaks, `epistemic_leakage` for channel leaks."
        )
        sections.append("")

    # === Negative-physics record (HARD)
    # The brief's CONSTRAINTS section already carries the HARD
    # ``=== PREVENTED EVENTS (HARD) ===`` and ``=== FALSE PROPOSITIONS
    # (HARD) ===`` blocks (emitted by every brief builder \u2014 directive,
    # observation, intervention, counterfactual). Surface a dedicated
    # reminder here so the auditor flags negative-physics breaches
    # under a typed rationale rather than as a generic prose drift.
    has_prevented = any(
        "PREVENTED EVENTS (HARD)" in (c.instruction or "")
        for c in (brief.constraints or [])
    )
    has_false_props = any(
        "FALSE PROPOSITIONS (HARD)" in (c.instruction or "")
        for c in (brief.constraints or [])
    )
    has_severed_chains = any(
        "SEVERED CAUSAL CHAINS (HARD" in (c.instruction or "")
        for c in (brief.constraints or [])
    )
    has_dependent_subs = any(
        "DEPENDENT-STATE SUBSTITUTIONS (HARD)" in (c.instruction or "")
        for c in (brief.constraints or [])
    )
    if has_prevented or has_false_props or has_severed_chains or has_dependent_subs:
        sections.append(
            "=== NEGATIVE PHYSICS (the prose must NOT stage these "
            "as occurring) ==="
        )
        if has_prevented:
            sections.append(
                "  Prevented events: see the `=== PREVENTED EVENTS "
                "(HARD) ===` block in the constraints above. The "
                "physics tags those event ids as not occurring. Flag "
                "any prose that stages them as having happened, has "
                "characters witness or remember them as past events, "
                "or treats a downstream consequence as if the "
                "prevented event were canonical. Violation type: "
                "`reasoning_failure` with rationale prefix "
                "`prevented_event:`."
            )
        if has_false_props:
            sections.append(
                "  False propositions: see the `=== FALSE PROPOSITIONS "
                "(HARD) ===` block in the constraints above. The "
                "physics commits those propositions FALSE at or before "
                "this scene's anchor. Characters MAY believe them "
                "(belief\u2260fact is an allowed mismatch and often the "
                "point); the narration MUST NOT enact them as fact. "
                "Flag prose that asserts a false proposition as "
                "occurring/true. Violation type: `reasoning_failure` "
                "with rationale prefix `false_proposition:`."
            )
        if has_severed_chains:
            sections.append(
                "  Severed causal chains: see the `=== SEVERED CAUSAL "
                "CHAINS (HARD, CONTEXT) ===` block in the constraints "
                "above. The do-surgery removed the listed root events "
                "AND their disjunctive ``chain_reaction`` closure; "
                "neither the roots nor the listed downstream "
                "consequences occur in this world. Flag prose that "
                "(a) stages any closure event as happening, or "
                "(b) invents a SUBSTITUTE mechanism that reaches the "
                "original downstream outcome by a different route "
                "(e.g. \"missing evidence\", \"alternative witness\", "
                "\"unexplained vacancy\" framings used to recover a "
                "consequence whose original cause was pruned). "
                "Violation type: `reasoning_failure` with rationale "
                "prefix `severed_causal_chain:`."
            )
        if has_dependent_subs:
            sections.append(
                "  Dependent-state substitutions: see the `=== "
                "DEPENDENT-STATE SUBSTITUTIONS (HARD) ===` block in "
                "the constraints above. The listed entities must be "
                "rendered in EXACTLY the post-surgery status shown; "
                "their plans / roles continue from that state. Flag "
                "prose that depicts any listed entity in a "
                "pre-surgery status (e.g. as dead when the block "
                "lists status='alive', or as absent when the block "
                "lists a location) or that fabricates an alternative "
                "obstacle to recover the pre-surgery outcome. "
                "Violation type: `reasoning_failure` with rationale "
                "prefix `dependent_state_substitution:`."
            )
        sections.append("")

    sections.append(f"=== AUDIT CATEGORIES TO CHECK: {', '.join(audit_categories)} ===")
    sections.append(
        "Run ONLY the audit categories listed above. Do not surface "
        "violations for any category not on the list \u2014 those are "
        "out of scope for this audit pass and would be silently "
        "discarded by the loop. The categories surface as: "
        "`epistemic` \u2192 mystery / dramatic_irony / surprise audits; "
        "`probabilistic` \u2192 suspense / fear / joy audits; "
        "`counterfactual` \u2192 regret / grief / rage / love audits; "
        "`physics` \u2192 intervention (Rung 2) and abduction (Rung 3) "
        "audits; `meta` \u2192 universal meta-narration audit; "
        "`style` \u2192 source-style fidelity audit (only when a STYLE "
        "FIDELITY block is present above)."
    )
    sections.append("")

    # Full scene context — same rich ego-graph the renderer sees.
    # Without this the auditor was judging prose against world-traits
    # alone and could not verify presence of co-located characters,
    # objects, recent events, dialogue (with truth_value), causal
    # edges, or standing channels — i.e. exactly the details the
    # renderer was hallucinating because they were missing from its
    # own prompt as well.
    if brief.scene_context:
        sections.append(
            "=== SCENE CONTEXT (the same ego-graph the renderer saw — "
            "use it as the ground-truth world state for every "
            "category) ==="
        )
        sections.append(format_scene_context_for_prompt(brief.scene_context))
        sections.append("")

    # === Renderer-declared introduced elements ===
    # These are new world elements the renderer minted in *this draft*.
    # Show them here so the LLM auditor does not flag prose references
    # to their names/ids as ``undeclared_element`` violations. The
    # deterministic ``_undeclared_element_violations`` pass also accepts
    # them, but the LLM auditor runs independently and needs to see them
    # explicitly to avoid re-firing undeclared_element on every iteration.
    if introduced_elements is not None and not introduced_elements.is_empty():
        sections.append(
            "=== INTRODUCED ELEMENTS (declared by the renderer for this "
            "draft — treat as already part of the world) ==="
        )
        sections.append(
            "The renderer declared the following NEW elements while writing "
            "this scene. Treat every name and id listed below as a KNOWN, "
            "DECLARED referent. Do NOT flag them as ``undeclared_element`` "
            "violations — they are intentional additions that will be "
            "materialised into the next world-state revision after the audit "
            "passes. Only flag a declared element if its justification is "
            "boilerplate, it duplicates an existing element, or the prose "
            "uses it in a way that violates other physics constraints. "
            "Similarly, do NOT flag any sub-location or affordance of a "
            "declared or existing location (e.g. an aircraft cabin when the "
            "scene is aboard a flight, a courtroom at a courthouse, a hotel "
            "room in a hotel) — contextually entailed spaces are never "
            "``undeclared_element`` violations."
        )
        _ie_kind_map = (
            ("entity", introduced_elements.entities),
            ("location", introduced_elements.locations),
            ("object", introduced_elements.objects),
            ("world_trait", introduced_elements.world_traits),
            ("channel", introduced_elements.channels),
            ("proposition", introduced_elements.propositions),
            ("concern", introduced_elements.concerns),
            ("event", introduced_elements.events),
        )
        for _ie_kind, _ie_specs in _ie_kind_map:
            for _ie_spec in _ie_specs:
                _ie_id = getattr(_ie_spec, "id", "?")
                _ie_name = getattr(_ie_spec, "name", "") or ""
                _ie_just = (getattr(_ie_spec, "justification", "") or "").strip()
                sections.append(
                    f"  [{_ie_kind}] {_ie_id} \"{_ie_name}\": {_ie_just}"
                )
        sections.append("")

    # === External research (WorldFact) fidelity ===
    # Mirrors the renderer's EXTERNAL RESEARCH block so the auditor
    # judges the prose against the *same* background facts. Two
    # failure modes need flagging:
    #   * the prose contradicts a high-confidence WorldFact
    #     (e.g. invents a different aircraft, gets a date wrong);
    #   * the prose promotes a WorldFact into canonical story
    #     content (treats background as on-stage event / trait /
    #     belief), which the research layer explicitly forbids.
    # Both are reported as `reasoning_failure` violations with a
    # `world_fact_fidelity:` rationale prefix — no new violation_type
    # is added (the Literal already covers physics-style failures).
    research = getattr(brief, "external_research", None) or []
    if research:
        sections.append(
            "=== EXTERNAL RESEARCH (WorldFact background — NOT "
            "authoritative for plot) ==="
        )
        sections.append(
            "Use these only as fidelity checks for period / place / "
            "vocabulary detail. Flag the prose as a `reasoning_failure` "
            "violation (rationale prefix `world_fact_fidelity:`) when "
            "it (a) contradicts a high-confidence fact below, or "
            "(b) promotes a WorldFact into a canonical event / "
            "trait / belief / dialogue claim in the story."
        )
        for fact in research:
            fid = getattr(fact, "fact_id", None) or getattr(fact, "id", "?")
            conf = getattr(fact, "confidence", "moderate")
            topic = getattr(fact, "topic", "")
            summary = getattr(fact, "summary", "")
            sections.append(f"  - {fid} ({conf}) {topic}: {summary}")
        sections.append("")

    # Epistemic gaps for reference
    if brief.epistemic_gaps:
        sections.append("=== EPISTEMIC STATE (beliefs vs reality) ===")
        for g in brief.epistemic_gaps:
            sections.append(
                f"  {g.entity_id} re {g.belief_target_id}: "
                f"believes='{g.believed_state}' actual='{g.actual_state}' "
                f"({g.gap_type}, mag={g.gap_magnitude:.2f})"
            )
        sections.append("")

    # Trait trajectories for physics audit
    if brief.trait_trajectories:
        extreme = [
            t for t in brief.trait_trajectories
            if abs(t.current_value - 0.5) >= 0.15
        ]
        if extreme:
            sections.append("=== KEY TRAIT STATES ===")
            for t in extreme:
                sections.append(
                    f"  {t.entity_id}.{t.trait_name} = {t.current_value:.2f} "
                    f"(inertia={t.inertia:.2f})"
                )
            sections.append("")

    # Intervention mechanisms for physics audit \u2014 use the renderer's
    # formatter so the auditor sees the same VACUOUS / ADVISORY tags
    # the prose was written under (Rule-3 pruned vs unproven paths).
    if brief.intervention_mechanisms:
        sections.append(_format_interventions(brief.intervention_mechanisms))
        sections.append("")

    # Relationship tensions: the per-axis (affinity / fear /
    # power_dynamic) snapshot from directive assembly, with asymmetry
    # weighted by the *minimum* per-axis evidence_strength of the two
    # endpoints. The auditor needs this so it can flag prose that
    # violates the measured social geometry — without this section the
    # rendering LLM may freely invent affinity reversals or fear
    # spikes that the directive explicitly suppressed.
    if brief.relationship_tensions:
        sections.append(
            "=== RELATIONSHIP TENSIONS (per-axis social geometry that "
            "the rendered scene MUST respect) ==="
        )
        for rt in brief.relationship_tensions[:25]:
            sections.append(
                f"  {rt.source_id}→{rt.target_id}: "
                f"affinity={rt.affinity:+.2f} "
                f"fear={rt.fear:.2f} "
                f"power_dynamic={rt.power_dynamic:+.2f} "
                f"(asymmetry={rt.asymmetry_score:.2f})"
            )
        sections.append("")

    # Threat proximity for suspense/fear audit — use the renderer's
    # full formatter so the auditor sees identical text
    # (affected_propositions / affected_beliefs / affected_concerns,
    # tragedy_form, etc., not just the headline numbers).
    if brief.threat_proximity:
        sections.append("=== THREAT PROXIMITY ===")
        sections.append(_format_threat_proximity(brief.threat_proximity))
        sections.append("")

    # Intervention sandbox (Rung-2 do-calculus) for plain
    # intervention briefs that don't carry a threat reading. Surfaces
    # the typed do_target + affected_propositions / affected_beliefs /
    # affected_concerns so the auditor can validate that the prose
    # grounds every flipped sandbox node — same as it does for
    # threat-keyed briefs via ``threat_proximity`` above.
    if brief.intervention_branch:
        sections.append("=== INTERVENTION SANDBOX ===")
        sections.append(_format_intervention_branch(brief.intervention_branch))
        sections.append("")

    # Counterfactual branch — use the renderer's formatter so the
    # auditor sees the divergence point, the RUNG-3 SURGERY KIND
    # framing (ontic vs epistemic vs motivational), and the
    # Aristotelian / Frye tragedy_form hedge that the prose was
    # written under. Previously the auditor only saw
    # ``actual_outcome`` / ``simulated_outcome`` and could not
    # validate that the prose matched the brief's surgery register.
    if brief.counterfactual_branch:
        sections.append("=== COUNTERFACTUAL BRANCH ===")
        sections.append(_format_counterfactual(brief.counterfactual_branch))
        sections.append("")

    # Causal attribution for rage audit — reuse renderer formatter so
    # the causal_chain is visible to the auditor (it was previously
    # dropped, leaving the auditor blind to the chain the prose was
    # told to render).
    if brief.causal_attribution:
        sections.append("=== CAUSAL ATTRIBUTION ===")
        sections.append(_format_causal_attribution(brief.causal_attribution))
        sections.append("")

    # Entanglement pairs for love audit — reuse renderer formatter.
    if brief.entanglement_pairs:
        sections.append(_format_entanglement(brief.entanglement_pairs))
        sections.append("")

    # Abduction truths for physics audit — reuse renderer formatter
    # so the auditor sees the same hidden_variable phrasing it must
    # validate against (subtextual presence, no explicit mention).
    if brief.abduction_truths:
        sections.append(_format_abduction(brief.abduction_truths))
        sections.append("")

    # Propositional / belief / concern surfacing — the renderer was
    # given dedicated blocks for surprise / irony / mystery profiles
    # and the affected_propositions / affected_concerns / affected_beliefs
    # lists; the auditor must see the same data so it can flag prose
    # that fails to ground a flipped proposition or violates a focal's
    # concern polarity.
    sections.extend(_format_propositional_context(brief))

    # Prior feedback (for iteration > 0)
    if prior_feedback:
        sections.append("=== PRIOR AUDIT FEEDBACK (from previous iterations) ===")
        sections.append(
            "The prose was already rewritten based on this feedback. "
            "Check if the issues have been resolved:"
        )
        for i, fb in enumerate(prior_feedback, 1):
            sections.append(f"  {i}. {fb}")
        sections.append("")

    # Engine-computed physics ground truth — fed to the auditor so it
    # can flag prose against deterministic facts the engine already
    # established (miracle steps, cyclic clusters, ctf-calculus prunings).
    if causal_feedback is not None and (
        causal_feedback.miracle_steps_detected
        or causal_feedback.cyclic_propagation_clusters
        or causal_feedback.noisy_or_absorbed_propagations
        or causal_feedback.rule3_pruned_interventions
        or causal_feedback.rule2_redundant_evidence
    ):
        sections.append("=== ENGINE PHYSICS LEDGER (ground truth) ===")
        if causal_feedback.miracle_steps_detected:
            sections.append(
                f"  Miracle steps ({len(causal_feedback.miracle_steps_detected)}) "
                f"— prose MUST render the mechanism for each:"
            )
            for ms in causal_feedback.miracle_steps_detected:
                sections.append(f"    - {ms}")
        if causal_feedback.cyclic_propagation_clusters:
            sections.append(
                f"  Cyclic propagation clusters "
                f"({len(causal_feedback.cyclic_propagation_clusters)}) "
                f"— static-topology cycles, NOT miracle steps. "
                f"The (entity, trait) pairs below CANNOT move in this "
                f"sandbox; do NOT flag prose for failing to depict any "
                f"change in them, and do NOT request a mechanism that "
                f"would move them:"
            )
            for cc in causal_feedback.cyclic_propagation_clusters:
                sections.append(f"    - {cc}")
        if causal_feedback.noisy_or_absorbed_propagations:
            sections.append(
                f"  Noisy-OR absorbed propagations "
                f"({len(causal_feedback.noisy_or_absorbed_propagations)}) "
                f"— probabilistic gate dampened weak impulses (impact ≪ "
                f"inertia), NOT miracle steps. The displayed `impact=` "
                f"number is NOT a required Δ; it is the (insufficient) "
                f"push the engine rejected. Do NOT flag prose for "
                f"omitting these changes, do NOT request a mechanism for "
                f"them, and do NOT cite the impact value as a target "
                f"delta:"
            )
            for na in causal_feedback.noisy_or_absorbed_propagations:
                sections.append(f"    - {na}")
        if causal_feedback.rule3_pruned_interventions:
            sections.append(
                f"  Rule-3 pruned interventions "
                f"({len(causal_feedback.rule3_pruned_interventions)}) "
                f"— provably vacuous on the AMWN. The prose should not "
                f"claim these surgeries had downstream effects:"
            )
            for ip in causal_feedback.rule3_pruned_interventions:
                sections.append(f"    - {ip}")
        if causal_feedback.rule2_redundant_evidence:
            sections.append(
                f"  Rule-2 redundant evidence "
                f"({len(causal_feedback.rule2_redundant_evidence)}) "
                f"— d-separated from interventions; abduction skipped "
                f"(informational only):"
            )
            for ev in causal_feedback.rule2_redundant_evidence:
                sections.append(f"    - {ev}")
        sections.append("")

    # --- Utterance fidelity (truth_value & intelligibility) ---
    # The auditor needs to know which on-page utterances are explicitly
    # marked false / performative so it can flag prose that quietly
    # treats them as fact ("she said X" → narrator endorses X).  It
    # also needs the per-listener intelligibility of each channel so it
    # can flag prose where a listener with intelligibility < 0.5
    # nonetheless cleanly understands the message.
    if world_state is not None:
        non_factual_utts: List[EventNode] = [
            evt for evt in world_state.events
            if evt.event_type == "utterance"
            and evt.truth_value in {"false", "performative"}
        ]
        if non_factual_utts:
            sections.append(
                "=== UTTERANCE FIDELITY (truth_value rules — prose must "
                "NOT present these as narratorial fact) ==="
            )
            for evt in non_factual_utts[:25]:
                content = (evt.content or "").strip().replace("\n", " ")
                if len(content) > 120:
                    content = content[:117] + "..."
                sections.append(
                    f"  {evt.id} [{evt.truth_value}] "
                    f"speaker={evt.speaker_id} → "
                    f"addressees={list(evt.addressee_ids)}: "
                    f"\"{content}\""
                )
            sections.append(
                "  Rule: 'false' utterances are lies/errors; the "
                "narrator must not endorse their content. "
                "'performative' utterances (vows, declarations, "
                "promises) are neither true nor false — flag prose "
                "that treats them as factual claims about the world."
            )
            sections.append(
                "  Paraphrase leak rule: the narrator is ALSO "
                "endorsing a false utterance when it paraphrases the "
                "content without the speech-act frame. \"He insisted "
                "he was at the flat\" is acceptable (attribution is "
                "intact). \"He had been at the flat that night\" "
                "rendered as ground-truth narration of a "
                "truth_value='false' utterance is a "
                "**reasoning_failure** with rationale prefix "
                "`utterance_paraphrase_leak:` \u2014 fire it whenever "
                "the prose asserts the propositional content of a "
                "false / performative utterance from above without "
                "an attribution clause (\"said\", \"claimed\", "
                "\"testified\", interior monologue marker, etc.) "
                "anchoring it back to the speaker."
            )
            sections.append("")

        opaque_channels = [
            ch for ch in world_state.channels.values()
            if ch.intelligibility and any(
                v < 0.5 for v in ch.intelligibility.values()
            )
        ]
        if opaque_channels:
            sections.append(
                "=== CHANNEL INTELLIGIBILITY (listeners who cannot "
                "fully understand — prose must NOT let them cleanly "
                "comprehend) ==="
            )
            for ch in opaque_channels[:25]:
                opaque = [
                    f"{lid}={score:.2f}"
                    for lid, score in ch.intelligibility.items()
                    if score < 0.5
                ]
                sections.append(
                    f"  {ch.id} ({ch.medium}, "
                    f"participants={list(ch.participant_ids)}): "
                    f"low for {opaque}"
                )
            sections.append(
                "  Rule: a listener with intelligibility<0.5 hears "
                "the channel but cannot reliably parse it. Flag prose "
                "where such a listener nonetheless quotes, "
                "paraphrases, or acts on the content with full "
                "comprehension."
            )
            sections.append("")

    sections.append(
        "=== TASK ===\n"
        "Audit the prose above against the constraints and physics state. "
        "Return the structured audit result."
    )

    return "\n".join(sections)


# =====================================================================
# Round-7 design improvements (audit 2026-05-26)
# =====================================================================
#
# These helpers were extracted during the round-7 feedback-loop audit
# so the orchestrator in ``run_feedback_loop`` can reason in clean,
# testable units instead of carrying brittle inline rules. Each helper
# documents the specific failure mode it addresses.

#: Severity weights for the regression scorer. The previous regression
#: detector used a raw violation count, which silently treated a swap
#: of one ``critical`` violation for two ``minor`` ones as a regression
#: and treated promoting three ``minor`` violations to three
#: ``critical`` ones as *not* a regression (count was unchanged). The
#: weighted score fixes both directions: severity matters.
_SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 4.0,
    "major": 2.0,
    "minor": 1.0,
}


class Finding(BaseModel):
    """Unified view of any single issue the feedback loop is tracking.

    Three sources feed findings into the loop:

      * ``"auditor"`` \u2014 the LLM ``NarrativeAuditor`` ("style_mismatch",
        "epistemic_leakage", "meta_narration", \u2026).
      * ``"engine"`` \u2014 deterministic physics scorecard thresholds
        (miracle steps, cyclic clusters, foreshadowing floor, \u2026).
      * ``"deterministic"`` \u2014 cheap regex checks run alongside the
        LLM auditor (``deterministic_prose_findings``: POV breach,
        meta-narration trigger phrases).

    The regression scorer in ``run_feedback_loop`` historically only\n    consumed ``AuditViolation`` objects, which silently zero-weighted\n    engine-threshold failures \u2014 a draft could close every LLM\n    violation while breaking a hard physics threshold and the\n    regression detector would call it an improvement. ``Finding``\n    plus ``findings_from_sources`` unify the three streams behind a\n    single severity-weighted score so a regression in *any* signal\n    source is detected uniformly.\n\n    This type is intentionally non-invasive: it is a *view* over the\n    three existing streams, not a replacement for ``AuditViolation``\n    or the ``engine_failures`` list. Call sites that already work\n    with the typed streams are unchanged; the regression scorer is\n    the one consumer that benefits from the unified view.\n    """

    model_config = {"frozen": True}

    source: Literal["auditor", "engine", "deterministic"]
    finding_type: str
    severity: Literal["critical", "major", "minor"]
    feedback: str
    evidence_quote: str = ""


def findings_from_sources(
    audit: Optional["AuditResult"],
    engine_failures: Optional[List[str]] = None,
) -> List[Finding]:
    """Project an ``AuditResult`` + engine-failure list into a flat
    ``Finding`` list for unified regression scoring.

    Engine failures are mapped to ``severity="critical"`` because they
    are deterministic violations of hard physics thresholds (miracle
    step, cyclic cluster, etc.). The LLM auditor's per-violation
    severity is preserved verbatim. Deterministic regex findings are
    *already* on ``audit.violations`` by the time this is called (the
    feedback loop merges them in before this projection runs), so they
    are tagged with ``source="auditor"`` here \u2014 callers that need to
    distinguish them can re-run ``deterministic_prose_findings`` for a
    pure deterministic view.
    """
    out: List[Finding] = []
    if audit is not None:
        for v in audit.violations:
            out.append(Finding(
                source="auditor",
                finding_type=v.violation_type,
                severity=v.severity,
                feedback=v.feedback,
                evidence_quote=v.evidence_quote or "",
            ))
    for failure in engine_failures or []:
        out.append(Finding(
            source="engine",
            finding_type="engine_threshold",
            severity="critical",
            feedback=failure,
        ))
    return out


def _finding_severity_score(findings: Iterable[Finding]) -> float:
    """Severity-weighted score over the unified ``Finding`` stream.

    Used by the regression-rollback gate so a draft that closes every
    LLM violation but breaks an engine threshold (or vice versa) is
    correctly classified as a regression.
    """
    total = 0.0
    for f in findings:
        total += _SEVERITY_WEIGHTS.get(f.severity, _SEVERITY_WEIGHTS["minor"])
    return total


def _violation_severity_score(violations: List["AuditViolation"]) -> float:
    """Return a severity-weighted score for a list of violations.

    Used by the regression-rollback gate in ``run_feedback_loop``
    instead of ``len(violations)``. A rising score (with at least one
    newly-introduced violation type) indicates the rewriter is making
    things *worse*, not just flagging different aspects of the same
    issue. See ``_SEVERITY_WEIGHTS`` for the rationale.

    Unknown severity strings default to the ``minor`` weight so a
    future ``Literal`` extension cannot silently zero-out the scorer.
    """
    total = 0.0
    for v in violations:
        total += _SEVERITY_WEIGHTS.get(
            getattr(v, "severity", "minor"), _SEVERITY_WEIGHTS["minor"],
        )
    return total


# Regexes used by ``deterministic_prose_findings`` to short-circuit the
# two violation classes most likely to slip past the LLM auditor under
# rewrite pressure: omniscient narration of non-POV consciousnesses
# (``pov_breach``) and self-referential narrator commentary
# (``meta_narration``).
#
# These are deliberately conservative — they fire only on patterns
# that have no defensible authored use. The LLM auditor still runs
# afterwards and can add nuanced findings the regex misses.

#: Meta-narration trigger phrases. Each phrase is a hallmark of the
#: narrator stepping outside the diegesis to comment on the artifice
#: of the story (e.g. "in this diverged timeline", "the established
#: record shows"). All hits are case-insensitive whole-phrase matches.
_META_NARRATION_PHRASES: List[str] = [
    "established record",
    "diverged timeline",
    "in this universe",
    "in this timeline",
    "alternate history",
    "counterfactual record",
    "the reader",
    "dear reader",
    "as we have seen",
    "as noted earlier",
    "as established",
    "the narrator",
    "this story",
    "this narrative",
    "in our story",
]


def deterministic_prose_findings(
    prose: str,
    *,
    pov_entity: Optional[str] = None,
    pov_entity_aliases: Optional[Iterable[str]] = None,
    pov_breach_threshold: int = 3,
) -> List["AuditViolation"]:
    """Return synthetic violations the LLM auditor frequently misses.

    Two deterministic checks run here:

    1. **POV-lock breach** (``pov_breach``). When ``pov_entity`` is
       supplied, count sentences whose *grammatical subject* is a
       proper noun that is **not** the POV entity (or one of its
       aliases) and that is followed by a *cognitive / perceptual /
       affective* verb ("thought", "knew", "felt", "remembered",
       "wondered", "decided", "realised"). Each such sentence is
       direct evidence the narration has left the POV character's
       head. When the count exceeds ``pov_breach_threshold`` a single
       critical violation is returned with the first 2 offending
       sentences as evidence.
    2. **Meta-narration** (``meta_narration``). Any sentence
       containing a phrase from ``_META_NARRATION_PHRASES`` is
       collected. The first hit becomes the evidence quote on a
       critical violation.

    The function returns an empty list when no violations fire. It
    is intentionally cheap (no NLP dependency) so it can run on every
    iteration before the LLM auditor and pre-flag the cases the
    auditor most commonly misses under rewrite pressure (round-7 audit
    2026-05-26 \u2014 see ``docs/architecture.md`` POV-lock section).
    """
    findings: List["AuditViolation"] = []
    if not prose:
        return findings

    # Cheap sentence tokenizer: split on ., !, ?, but keep enough
    # context per sentence to attribute subject + verb. Avoids pulling
    # in an NLP dependency.
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[\.\!\?])\s+", prose)
        if s.strip()
    ]

    # --- POV-lock check ---
    if pov_entity:
        alias_set: set[str] = set()
        # Normalize the POV entity id (e.g. ``ENT_MRS_COADY``) into
        # surface forms the prose is likely to use. Without aliases
        # the check would over-fire on any third-person narrative.
        if pov_entity_aliases:
            alias_set.update(a.strip().lower() for a in pov_entity_aliases if a)
        # Strip a leading ``ENT_`` prefix and split underscores so
        # ``ENT_MRS_COADY`` -> {"mrs", "coady", "mrs coady"}. This is
        # a heuristic; callers SHOULD pass explicit aliases when
        # available.
        bare = re.sub(r"^ENT_", "", pov_entity, flags=re.IGNORECASE)
        parts = [p for p in re.split(r"[_\s]+", bare) if p]
        for p in parts:
            alias_set.add(p.lower())
        if parts:
            alias_set.add(" ".join(parts).lower())

        cognitive_verbs = (
            r"thought|knew|felt|remembered|wondered|decided|realised|"
            r"realized|believed|hoped|feared|noticed|saw|heard|"
            r"understood|recognized|recognised|suspected"
        )
        # Match "ProperNoun(s) [optional descriptor] cognitive_verb"
        subj_verb_re = re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
            r"(?:\s+\w+){0,3}?"
            r"\s+(?:" + cognitive_verbs + r")\b",
            re.IGNORECASE,
        )

        offending: List[str] = []
        # R19-M6: past-perfect retrospective narration ("had felt",
        # "had heard") is not a POV breach \u2014 the narrator is
        # looking back, not crossing into another character's
        # consciousness. Detect ``had`` immediately before the
        # cognitive verb and skip those matches.
        had_aux_re = re.compile(
            r"\bhad\s+(?:" + cognitive_verbs + r")\b",
            re.IGNORECASE,
        )
        for sent in sentences:
            m = subj_verb_re.search(sent)
            if not m:
                continue
            # If the verb in this match is immediately preceded by
            # ``had`` (past perfect / retrospective narration), skip.
            verb_start = m.end(1)
            tail = sent[verb_start:m.end()]
            if had_aux_re.search(tail):
                continue
            subject = m.group(1).strip().lower()
            # Skip if the subject is the POV entity (any alias).
            if any(alias in subject or subject in alias for alias in alias_set):
                continue
            # Skip common non-character proper nouns (place names
            # adjacent to a cognitive verb are extremely rare; this
            # filter is conservative).
            if subject in {"god", "heaven", "fate", "time"}:
                continue
            offending.append(sent)

        if len(offending) >= pov_breach_threshold:
            findings.append(AuditViolation(
                violation_type="reasoning_failure",
                severity="critical",
                description=(
                    f"POV lock broken: narration enters non-POV "
                    f"consciousness in {len(offending)} sentence(s) "
                    f"while brief locks POV to {pov_entity}."
                ),
                evidence_quote=" / ".join(offending[:2]),
                feedback=(
                    f"Restrict narration to {pov_entity}'s consciousness. "
                    f"Remove or externalize the cognitive/perceptual "
                    f"verbs attached to non-POV subjects in the "
                    f"offending sentences."
                ),
            ))

    # --- Meta-narration check ---
    lower_prose = prose.lower()
    meta_hits: List[Tuple[str, str]] = []
    for phrase in _META_NARRATION_PHRASES:
        idx = lower_prose.find(phrase)
        if idx < 0:
            continue
        # Recover the enclosing sentence for evidence.
        for sent in sentences:
            if phrase in sent.lower():
                meta_hits.append((phrase, sent))
                break
    if meta_hits:
        phrase, evidence = meta_hits[0]
        findings.append(AuditViolation(
            violation_type="meta_narration",
            severity="critical",
            description=(
                f"Meta-narration phrase \"{phrase}\" surfaces in the "
                f"prose. The narrator must remain inside the diegesis."
            ),
            evidence_quote=evidence,
            feedback=(
                f"Remove the phrase \"{phrase}\" and any surrounding "
                f"self-referential commentary about the story's "
                f"timeline / record / artifice. Render the same beat "
                f"diegetically."
            ),
        ))

    return findings


def _annotate_prose_with_violation_spans(
    prose: str, violations: List["AuditViolation"], max_chars: int = 6000,
) -> str:
    """Return ``prose`` with each violation's ``evidence_quote`` wrapped
    inline as a ``<<<VIOLATION:type>>>...<<</VIOLATION>>>`` marker.

    Span-locating ([A] in the round-7 audit improvements) turns the
    refinement prompt's "minimal surgical edit" instruction from a
    vibe into a directive the rewriter can mechanically execute: the
    spans it must touch are marked, and everything outside the
    markers should be preserved verbatim.

    Quotes that don't appear verbatim in the prose (the LLM
    paraphrased) are silently skipped — they remain available in the
    per-violation feedback list. Overlapping markers are flattened
    (the longest-quote-wins) so we don't emit malformed nested tags.

    The output is capped at ``max_chars`` to keep the refinement
    prompt under the model's working budget; the surrounding caller
    is responsible for appending the truncation marker.
    """
    if not prose:
        return prose

    # Build [start, end, type] triples for each quote that actually
    # appears verbatim in the draft.
    intervals: List[Tuple[int, int, str]] = []
    for v in violations:
        q = (v.evidence_quote or "").strip()
        if not q:
            continue
        idx = prose.find(q)
        if idx < 0:
            continue
        intervals.append((idx, idx + len(q), v.violation_type))

    if not intervals:
        return prose

    # Sort by start; resolve overlaps by keeping the longest span.
    intervals.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    flattened: List[Tuple[int, int, str]] = []
    for start, end, vtype in intervals:
        if flattened and start < flattened[-1][1]:
            # Overlap with previous. Keep whichever covers more.
            ps, pe, pt = flattened[-1]
            if (end - start) > (pe - ps):
                flattened[-1] = (start, end, vtype)
            continue
        flattened.append((start, end, vtype))

    # Splice in markers from end to start so earlier indices stay
    # valid.
    out = prose
    for start, end, vtype in reversed(flattened):
        out = (
            out[:start]
            + f"<<<VIOLATION:{vtype}>>>"
            + out[start:end]
            + "<<</VIOLATION>>>"
            + out[end:]
        )
    return out


def _check_pov_lock_metadata(
    brief_pov: Optional[str],
    scene_pov: Optional[str],
    *,
    additional_locks: Optional[Iterable[str]] = None,
    policy: str = "single",
) -> Optional[str]:
    """Return a violation feedback string if the scene's ``pov_entity``
    metadata diverges from the brief's POV contract.

    The orchestrator already coerces the metadata back, but a divergence
    here is a high-signal indicator the rewriter has internally
    abandoned the POV lock (even if the prose is then patched up). The
    return value is intended as a synthetic ``AuditViolation`` feedback
    line so the next refinement pass sees the breach explicitly
    listed.

    Policy semantics (round-7 deeper audit 2026-05-27 \u2014 P0 #2a):

    * ``"single"`` \u2014 ``scene_pov`` must equal ``brief_pov`` exactly.
    * ``"rotating"`` \u2014 ``scene_pov`` may be ``brief_pov`` OR any
      entry of ``additional_locks``.
    * ``"ensemble"`` \u2014 omniscient narration is licensed; no check
      fires (always returns ``None``).
    """
    if policy == "ensemble":
        return None
    if not brief_pov:
        return None
    allowed = {brief_pov}
    if additional_locks:
        allowed.update(a for a in additional_locks if a)
    if scene_pov in allowed:
        return None
    if policy == "rotating":
        return (
            f"POV-lock metadata diverged: brief licenses pov_entity in "
            f"{sorted(allowed)!r} under rotating policy but scene "
            f"returned pov_entity={scene_pov!r}. Mirror one of the "
            f"licensed entities exactly in the structured output and "
            f"keep all narration anchored to that consciousness for "
            f"the duration of the scene."
        )
    return (
        f"POV-lock metadata diverged: brief locks pov_entity="
        f"{brief_pov!r} but scene returned pov_entity={scene_pov!r}. "
        f"Mirror the brief's pov_entity exactly in the structured "
        f"output and keep all narration anchored to that "
        f"consciousness."
    )


_MIRACLE_INJECT_CAP = 12
_BLOCKED_KEY_RE = re.compile(
    r"(?P<entity>[A-Z][A-Z0-9_]+)\.(?P<trait>[a-zA-Z_][a-zA-Z0-9_]*)"
)


def _engine_blocked_keys(
    blocked: List[BlockedPropagation],
    causal_feedback: Optional[CausalPhysicsFeedback] = None,
) -> set[tuple[str, str]]:
    """Union of (entity, trait) pairs the engine declared off-limits.

    Combines ``BlockedPropagation`` entries with parsed
    ``cyclic_propagation_clusters`` / ``noisy_or_absorbed_propagations``
    so any downstream mechanism injection or auditor-flag matching can
    drop targets the engine already proved cannot move.
    """
    keys: set[tuple[str, str]] = set()
    for b in blocked or []:
        if getattr(b, "node_id", None) and getattr(b, "trait", None):
            keys.add((str(b.node_id), str(b.trait)))
    if causal_feedback is not None:
        for entry in (
            list(causal_feedback.cyclic_propagation_clusters or [])
            + list(causal_feedback.noisy_or_absorbed_propagations or [])
        ):
            m = _BLOCKED_KEY_RE.search(str(entry))
            if m:
                keys.add((m.group("entity"), m.group("trait")))
    return keys


def _violation_targets_blocked_key(
    v: AuditViolation,
    blocked_keys: set[tuple[str, str]],
) -> bool:
    """Best-effort check that a miracle_step violation is talking about
    a blocked propagation (so we should NOT inject a contradicting
    mechanism for it).

    Matching is deliberately conservative: only drop the violation when
    the auditor's text mentions the entity and trait in close proximity
    or in the canonical ``ENT_FOO.trait`` form. Loose "both words appear
    somewhere in the feedback" matching would silently suppress
    legitimate miracle-steps about an unrelated trait of the same
    entity, which is itself a regression. When in doubt, keep the
    injected directive — the cap (``_MIRACLE_INJECT_CAP``) bounds the
    blast radius if our match is too permissive.
    """
    if not blocked_keys:
        return False
    haystack = " ".join(
        [
            v.feedback or "",
            getattr(v, "evidence_quote", "") or "",
            v.description or "",
        ]
    )
    if not haystack:
        return False
    haystack_lower = haystack.lower()
    for ent, trait in blocked_keys:
        ent_l = ent.lower()
        trait_l = trait.lower()
        # 1) Canonical dotted form (``ENT_FOO.guilt``) — unambiguous.
        if f"{ent_l}.{trait_l}" in haystack_lower:
            return True
        # 2) Both tokens appear within 80 chars of each other AND the
        # trait word is bracketed by non-alphanumeric chars (so
        # ``guilt`` matches but ``guilty`` / ``guiltless`` don't, and
        # an unrelated paragraph mention of the entity doesn't drag in
        # a stray trait collision).
        ent_idx = haystack_lower.find(ent_l)
        if ent_idx < 0:
            continue
        # Use word-boundary regex for the trait to avoid partial-word
        # false positives.
        for m in re.finditer(rf"\b{re.escape(trait_l)}\b", haystack_lower):
            if abs(m.start() - ent_idx) <= 80:
                return True
    return False


def _inject_miracle_step_mechanisms(
    brief: CreativeBrief,
    blocked: List[BlockedPropagation],
    violations: List[AuditViolation],
    causal_feedback: Optional[CausalPhysicsFeedback] = None,
) -> int:
    """Append :class:`InterventionMechanism` entries to *brief* for every
    miracle-step the engine or auditor flagged.

    The audit/refinement loop previously could only re-prompt the LLM on
    a miracle step. That left the rewriter free to ignore the deficit and
    repeat the same Impact < Inertia mistake. Here we mutate the brief
    itself: each blocked propagation becomes an explicit
    ``InterventionMechanism`` directive carrying the entity, the trait,
    its old value, the inertia the rewrite must overcome, and a synthetic
    ``mechanism_hint`` reminding the renderer that a *visible physical
    or psychological force* is required on the page.

    Returns the number of mechanisms added (so the caller can decide
    whether to rebuild the rendering prompt).
    """
    if not blocked and not violations:
        return 0

    existing_keys = {
        (m.node_id, m.new_state) for m in brief.intervention_mechanisms
    }
    added = 0
    # Cap total mechanism count so multi-iteration runs cannot accumulate
    # contradictory "stage a beat for X" directives that flip-flop with
    # later auditor passes. Mirrors ``_MIRACLE_INJECT_CAP``.
    cap_remaining = max(0, _MIRACLE_INJECT_CAP - len(brief.intervention_mechanisms))
    # Engine ledger of (entity, trait) pairs the propagator already
    # declared off-limits (cycles, noisy-OR-absorbed, etc.). We must
    # NOT inject mechanism hints that ask the renderer to depict these
    # — that's the source of auditor↔engine contradictions.
    engine_blocked_keys = _engine_blocked_keys(blocked, causal_feedback)

    for b in blocked:
        if cap_remaining <= 0:
            break
        if b.reason != "inertia":
            # Spatial / cycle / noisy_or_absorbed blocks aren't narrative
            # miracles — their fix is in the world topology, not in the
            # prose. Injecting a mechanism for them creates a contradiction
            # with the brief's BLOCKED PROPAGATIONS (HARD) block.
            continue
        key = (b.node_id, f"{b.trait}\u2191" if b.impact > 0 else f"{b.trait}\u2193")
        if key in existing_keys:
            continue
        direction = "raise" if b.impact > 0 else "lower"
        brief.intervention_mechanisms.append(InterventionMechanism(
            node_id=b.node_id,
            old_state=b.trait,
            new_state=key[1],
            mechanism_hint=(
                f"Render an explicit on-page force that {direction}s "
                f"{b.node_id}.{b.trait}: the engine measured "
                f"|impact|={abs(b.impact):.2f} \u2264 inertia={b.inertia:.2f}, "
                "so the prose MUST stage a mechanism strong enough to "
                "overcome that inertia (described action, dialogue, "
                "perceived threat, etc.) instead of asserting the change."
            ),
            inertia=b.inertia,
        ))
        existing_keys.add(key)
        added += 1
        cap_remaining -= 1

    # The LLM auditor may also flag miracle-steps the engine missed
    # (e.g. an off-graph trait the renderer changed without justification).
    for v in violations:
        if cap_remaining <= 0:
            break
        if v.violation_type != "miracle_step":
            continue
        # Drop auditor-flagged miracle-steps whose subject overlaps
        # with engine-blocked propagations. Otherwise we'd be telling
        # the renderer to "stage a mechanism for X" while simultaneously
        # telling it (via BLOCKED PROPAGATIONS HARD) that X cannot
        # change — the loop never converges.
        if _violation_targets_blocked_key(v, engine_blocked_keys):
            continue
        # Synthetic key so the auditor-only flag still becomes a directive.
        key = ("AUDITOR", v.feedback[:40])
        if key in existing_keys:
            continue
        brief.intervention_mechanisms.append(InterventionMechanism(
            node_id="UNKNOWN",
            old_state="(auditor-flagged)",
            new_state="(auditor-flagged)",
            mechanism_hint=(
                "Auditor flagged miracle-step: " + v.feedback +
                " \u2014 stage an explicit causal mechanism in the next draft."
            ),
            inertia=0.5,
        ))
        existing_keys.add(key)
        added += 1
        cap_remaining -= 1

    return added


def _build_refinement_prompt(
    original_rendering_prompt: str,
    violations: List[AuditViolation],
    iteration: int,
    engine_failures: Optional[List[str]] = None,
    prior_violations: Optional[List[AuditViolation]] = None,
    brief: Optional[CreativeBrief] = None,
    previous_prose: Optional[str] = None,
    regression_warning: Optional[str] = None,
    introduced_elements: Optional["IntroducedElements"] = None,
) -> str:
    """Augment the original rendering prompt with auditor feedback.

    The feedback is injected as additional HARD constraints that override
    any conflicting soft constraints from the original brief.

    Engine-threshold failures (computed deterministically from the
    physics scorecard, not the LLM auditor) are also surfaced when
    provided, so the rewriter sees BOTH signal sources rather than only
    the LLM violations.

    ``prior_violations`` carries every violation flagged in earlier
    iterations of the loop. They are surfaced as **non-regression
    constraints** so the rewriter doesn't ping-pong between competing
    fixes (a classic failure mode where the model fixes
    ``style_mismatch`` by stripping voice, then regresses on
    ``meta_narration`` next iteration, then back).

    ``previous_prose`` (added in the round-7 audit, 2026-05-26) is the
    verbatim text of the draft the auditor just flagged. When supplied,
    the rewrite footer asks the model to **minimally edit** the prior
    draft rather than render from scratch — the from-scratch rewrite
    pattern routinely loses POV / style / anti-meta constraints the
    previous draft already got right and introduces fresh violation
    types. Without this anchor, a draft that was style-bad but had
    correct POV gets rewritten into style-better prose that breaks POV.
    """
    feedback_lines: List[str] = [
        "",
        f"=== AUDITOR FEEDBACK (Iteration {iteration}) ===",
        "The NarrativeAuditor found the following violations in your "
        "previous draft. You MUST address ALL of them in this rewrite:",
        "",
    ]

    # Surface the prior draft's introduced_elements declarations at the
    # top of the feedback block so the refiner knows exactly which new
    # elements were already declared and must be preserved in the new
    # GeneratedScene.introduced_elements output if the corresponding
    # prose references survive the rewrite. Without this, a refiner
    # performing a minimal surgical edit on unrelated violations
    # routinely drops or forgets the prior declarations, causing the
    # deterministic undeclared_element check to re-fire on the next
    # audit pass even though the prose is unchanged.
    if introduced_elements is not None and not introduced_elements.is_empty():
        feedback_lines.append(
            "=== PREVIOUSLY DECLARED INTRODUCED ELEMENTS "
            "(preserve in your new draft) ==="
        )
        feedback_lines.append(
            "The previous draft declared the following new world elements "
            "in its ``introduced_elements`` structured output. If your "
            "rewrite keeps any prose reference to a declared element's "
            "name or id, you MUST also include it in your new "
            "``introduced_elements`` output. Only drop a declaration if "
            "you are also removing the prose reference entirely. If you "
            "are fixing an ``undeclared_element`` violation for one of "
            "these, option (c) — re-declaring it with a better "
            "justification — is usually correct."
        )
        _ie_kind_map = (
            ("entity", introduced_elements.entities),
            ("location", introduced_elements.locations),
            ("object", introduced_elements.objects),
            ("world_trait", introduced_elements.world_traits),
            ("channel", introduced_elements.channels),
            ("proposition", introduced_elements.propositions),
            ("concern", introduced_elements.concerns),
            ("event", introduced_elements.events),
        )
        for _ie_kind, _ie_specs in _ie_kind_map:
            for _ie_spec in _ie_specs:
                _ie_id = getattr(_ie_spec, "id", "?")
                _ie_name = getattr(_ie_spec, "name", "") or ""
                _ie_just = (getattr(_ie_spec, "justification", "") or "").strip()
                feedback_lines.append(
                    f"  [{_ie_kind}] {_ie_id} \"{_ie_name}\": {_ie_just}"
                )
        feedback_lines.append("")

    # Round-7 audit (2026-05-26): when the feedback loop has just
    # rolled back from a regression and granted an anti-regression
    # retry, surface the warning at the very top of the feedback
    # block so it sits above the regular violation list. The rewriter
    # is being told: "your previous attempt broke things that were
    # already correct; do not do that again."
    if regression_warning:
        feedback_lines.extend([
            "=== REGRESSION ALERT (anti-regression retry) ===",
            regression_warning,
            "",
        ])

    # Anchor the rewrite to the prior draft so the model can perform a
    # surgical edit instead of re-rolling every surface choice from
    # scratch (audit 2026-05-26 — see docstring). Capped to keep the
    # refinement prompt under the model's working-attention budget;
    # the auditor's evidence quotes localize the offending spans even
    # when the cap truncates the tail of a long draft.
    _PREV_DRAFT_BUDGET_CHARS = 6000
    if previous_prose:
        prev = previous_prose.strip()
        if len(prev) > _PREV_DRAFT_BUDGET_CHARS:
            prev = prev[:_PREV_DRAFT_BUDGET_CHARS] + "\n\u2026 (truncated)"
        feedback_lines.extend([
            "=== PREVIOUS DRAFT (the prose the auditor just flagged) ===",
            prev,
            "",
        ])

    for i, v in enumerate(violations, 1):
        feedback_lines.append(
            f"  {i}. [{v.severity.upper()} | {v.violation_type}] "
            f"{v.feedback}"
        )
        if v.evidence_quote:
            feedback_lines.append(
                f"     OFFENDING PASSAGE: \"{v.evidence_quote}\""
            )
        feedback_lines.append("")

    if engine_failures:
        feedback_lines.append(
            "=== ENGINE-THRESHOLD FAILURES (deterministic scorecard) ==="
        )
        feedback_lines.append(
            "These are measured from the physics engine, not LLM-judged. "
            "Treat them as hard constraints alongside the violations above."
        )
        feedback_lines.append("")
        for i, f in enumerate(engine_failures, 1):
            feedback_lines.append(f"  E{i}. {f}")
        feedback_lines.append("")

    if prior_violations:
        # Deduplicate by (violation_type, feedback) so the same recurring
        # gripe doesn't bloat the prompt across iterations. Skip
        # violations whose ``violation_type`` already appears in the
        # current ``violations`` list \u2014 those are still active and
        # the rewriter is already looking at them above.
        active_types = {v.violation_type for v in violations}
        seen: set[tuple[str, str]] = set()
        unique_prior: List[AuditViolation] = []
        for v in prior_violations:
            if v.violation_type in active_types:
                continue
            key = (v.violation_type, v.feedback[:160])
            if key in seen:
                continue
            seen.add(key)
            unique_prior.append(v)

        if unique_prior:
            feedback_lines.append(
                "=== NON-REGRESSION CONSTRAINTS "
                "(fixed in earlier iterations \u2014 must remain fixed) ==="
            )
            feedback_lines.append(
                "These violations were raised against earlier drafts and "
                "have since been fixed. Do NOT reintroduce them while "
                "addressing the current violations above. Optimising the "
                "latest auditor note at the cost of regressing on a prior "
                "fix is the most common loop-thrash pattern \u2014 hold the "
                "line on each one."
            )
            feedback_lines.append("")
            for i, v in enumerate(unique_prior, 1):
                feedback_lines.append(
                    f"  N{i}. [{v.violation_type}] {v.feedback}"
                )
            feedback_lines.append("")

    feedback_lines.append(
        "=== REWRITE TASK ===\n"
        + (
            "You have the PREVIOUS DRAFT above. Produce a MINIMAL EDIT "
            "of it that fixes every violation listed above (and respects "
            "the non-regression constraints, when present). Preserve every "
            "surface choice the previous draft already got right — POV "
            "lock, rendering mode, opening framing, anti-meta-narration "
            "discipline, blocked-trait silence, style register — unless a "
            "specific violation requires changing it. Do NOT re-render "
            "from scratch; surgical edits only. The auditor will check "
            "again."
            if previous_prose else
            "Rewrite the prose passage from scratch, honouring ALL original "
            "constraints AND the auditor corrections above. The auditor will "
            "check again."
        )
    )
    
    # P3 #7 (round-7 deeper audit 2026-05-27): token-budget cap was
    # previously applied here, BEFORE the style re-anchor block was
    # potentially appended below, which let the final prompt exceed
    # MAX_FEEDBACK_CHARS by the size of the style block. Defer the
    # cap to the final assembled string so the guarantee actually
    # holds.

    # Style re-anchor: when a style violation is active (or has been
    # active in a prior iteration), the original STYLE FIDELITY block
    # is now buried thousands of tokens above the rewrite footer and
    # the rewriter reliably under-attends to it. Re-emit a compact
    # restatement of the brief's style contract immediately before the
    # rewrite footer so the contract sits in the same attention window
    # as the violations being fixed. Without this the loop converges
    # on local fixes (mode/POV/leak) but keeps drifting on density and
    # word-band, never matching the source register.
    style_types = {"style_mismatch", "register_drift", "word_band_violation"}
    style_active = any(v.violation_type in style_types for v in violations) or any(
        (pv.violation_type in style_types) for pv in (prior_violations or [])
    )
    if style_active and brief is not None and brief.narrative_style is not None:
        try:
            from shadow_loom.narrative_style import format_narrative_style_block
            style_block = format_narrative_style_block(
                brief.narrative_style,
                header=(
                    "STYLE FIDELITY (RE-EMITTED \u2014 the contract you are "
                    "still drifting from)"
                ),
                original_query=brief.original_query,
            )
            feedback_lines.append("")
            feedback_lines.append(
                "The style contract below is the SAME contract that was "
                "in your original prompt. The auditor has flagged a "
                "style violation across this iteration (and possibly "
                "earlier iterations). Restating it here so it sits in "
                "the same attention window as the violations above:"
            )
            feedback_lines.append("")
            feedback_lines.append(style_block)
        except Exception:
            logger.debug(
                "[FeedbackLoop] Style re-anchor skipped \u2014 "
                "format_narrative_style_block raised.",
                exc_info=True,
            )

    # P3 #7 (round-7 deeper audit 2026-05-27): apply the token-budget
    # cap to the FINAL assembled feedback text after every optional
    # section (PREVIOUS DRAFT, REGRESSION ALERT, violations,
    # non-regression constraints, rewrite task, style re-anchor) has
    # been appended. Truncation order: keep the leading sections (the
    # rewriter needs the previous draft and violation list more than
    # the trailing style block).
    MAX_FEEDBACK_CHARS = 8000
    feedback_text = "\n".join(feedback_lines)
    if len(feedback_text) > MAX_FEEDBACK_CHARS:
        logger.warning(
            "[Refinement] Feedback exceeds %d chars (%d), truncating",
            MAX_FEEDBACK_CHARS, len(feedback_text),
        )
        feedback_text = (
            feedback_text[:MAX_FEEDBACK_CHARS]
            + "\n... (feedback truncated due to length)"
        )

    return original_rendering_prompt + feedback_text


# =====================================================================
# Auditor Agent
# =====================================================================

def _build_auditor_agent(
    config: AuditorConfig,
) -> Agent[_AuditorDeps, AuditResult]:
    """Construct the Step 11 auditor LLM agent."""
    agent: Agent[_AuditorDeps, AuditResult] = Agent(
        _resolve_model(config.auditor_model, stage="auditor"),
        deps_type=_AuditorDeps,
        output_type=NativeOutput(AuditResult),
        system_prompt=_load_prompt("auditor.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_audit_prompt(ctx: RunContext[_AuditorDeps]) -> str:
        return ctx.deps.audit_prompt

    return agent


# =====================================================================
# Evaluation Agent (separate from auditor — literary critique only)
# =====================================================================

def _build_evaluation_agent(
    config: AuditorConfig,
) -> Agent[_EvaluationDeps, StoryQualitySynthesis]:
    """Construct the evaluation LLM agent for literary critique."""
    agent: Agent[_EvaluationDeps, StoryQualitySynthesis] = Agent(
        _resolve_model(config.auditor_model, stage="auditor"),
        deps_type=_EvaluationDeps,
        output_type=NativeOutput(StoryQualitySynthesis),
        system_prompt=_load_prompt("evaluation.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_evaluation_prompt(ctx: RunContext[_EvaluationDeps]) -> str:
        return ctx.deps.evaluation_prompt

    return agent


def assemble_evaluation_prompt(
    prose: str,
    brief: CreativeBrief,
    causal_feedback: Optional[CausalPhysicsFeedback] = None,
    affective_feedback: Optional[AffectiveStateFeedback] = None,
) -> str:
    """Build the full prompt for the evaluation LLM.

    Provides the prose, graph summary, brief constraints, and engine
    metrics so the LLM can produce a holistic literary critique.
    """
    sections: List[str] = []

    sections.append("=== PROSE TO EVALUATE ===")
    sections.append(prose)
    sections.append("")

    sections.append(f"=== TARGET EFFECT: {brief.target_effect.upper()} ===")
    sections.append(f"Target entities: {', '.join(brief.target_entities)}")
    sections.append("")

    sections.append("=== CONSTRAINTS (from Creative Brief) ===")
    sections.append(_format_constraints_for_audit(brief.constraints))
    sections.append("")

    # Style fidelity — the evaluator must grade the combined prose
    # against the same source-register contract the renderer/auditor
    # enforce on each chunk. Without this the full-story scorecard is
    # style-blind even when the brief carries a populated
    # ``narrative_style``.
    if brief.narrative_style is not None:
        ns = brief.narrative_style
        sections.append("=== STYLE FIDELITY (source register) ===")
        sections.append(f"  format: {ns.format}")
        sections.append(
            f"  target word count: {ns.target_word_min}\u2013{ns.target_word_max} "
            f"(per render; full-story prose may exceed this if multiple chunks)"
        )
        sections.append(f"  prose density: {ns.prose_density}")
        if ns.voice:
            sections.append(f"  voice: {ns.voice}")
        if ns.style_exemplar:
            sections.append(f"  exemplar: {ns.style_exemplar[:600]}")
        sections.append("")

    # Branch context — a shadow-branch evaluation must NOT be graded
    # as if it were factual canon. Mirror the renderer's
    # ``=== BRANCH CONTEXT ===`` block so the evaluator knows the
    # prose is a counterfactual world.
    if brief.branch_world_id == "shadow":
        sections.append("=== BRANCH CONTEXT ===")
        sections.append("branch_world_id: shadow")
        if getattr(brief, "branch_label", None):
            sections.append(f"branch_label: {brief.branch_label}")
        if getattr(brief, "factual_contrast_summary", None):
            sections.append(
                f"factual_contrast_summary: {brief.factual_contrast_summary}"
            )
        sections.append(
            "This evaluation is of a SHADOW branch \u2014 a Rung-2/3 fork "
            "produced by a do(\u00b7) intervention. Any divergence from "
            "``factual_contrast_summary`` (resurrected characters, "
            "prevented events, flipped propositions, missing canonical "
            "utterances) is INTENTIONAL and is what the branch exists "
            "to explore. Do NOT penalise these divergences as "
            "narrative inconsistency, plot-hole, or continuity "
            "failures. Grade the prose on its own internal coherence, "
            "tone, and craft \u2014 the factual contrast is a reference "
            "point for what was changed, not a rubric the shadow "
            "branch is expected to match."
        )
        sections.append("")

    # Epistemic state
    if brief.epistemic_gaps:
        sections.append("=== EPISTEMIC STATE (beliefs vs reality) ===")
        for g in brief.epistemic_gaps:
            sections.append(
                f"  {g.entity_id} re {g.belief_target_id}: "
                f"believes='{g.believed_state}' actual='{g.actual_state}' "
                f"({g.gap_type}, mag={g.gap_magnitude:.2f})"
            )
        sections.append("")

    # Trait trajectories
    if brief.trait_trajectories:
        sections.append("=== KEY TRAIT STATES ===")
        for t in brief.trait_trajectories:
            sections.append(
                f"  {t.entity_id}.{t.trait_name} = {t.current_value:.2f} "
                f"(inertia={t.inertia:.2f})"
            )
        sections.append("")

    # Propositional / belief / concern surfacing — same data the
    # renderer was given via the ``*_profile`` and
    # ``affected_propositions`` / ``affected_concerns`` /
    # ``affected_beliefs`` blocks. Without this the evaluator scores
    # full-story prose blind to the propositional commitments the
    # directive was assembled around.
    sections.extend(_format_propositional_context(brief))

    # Rung-2/3 surgery surfaces — mirror of the audit prompt's
    # threat_proximity / intervention_branch / counterfactual_branch
    # blocks. The evaluator must see the same do_target context,
    # multi-hop history chain, and surgery-kind hints the renderer
    # was given so the literary critique can grade the prose against
    # the actual Rung-2/3 instructions, not just the factual canon.
    if brief.threat_proximity:
        sections.append("=== THREAT PROXIMITY ===")
        sections.append(_format_threat_proximity(brief.threat_proximity))
        sections.append("")
    if brief.intervention_branch:
        sections.append("=== INTERVENTION SANDBOX ===")
        sections.append(_format_intervention_branch(brief.intervention_branch))
        sections.append("")
    if brief.counterfactual_branch:
        sections.append("=== COUNTERFACTUAL BRANCH ===")
        sections.append(_format_counterfactual(brief.counterfactual_branch))
        sections.append("")

    # Engine-computed metrics (ground truth for the evaluator)
    if causal_feedback is not None:
        sections.append("=== ENGINE: CAUSAL PHYSICS METRICS (ground truth) ===")
        sections.append(
            f"  Miracle steps detected: {len(causal_feedback.miracle_steps_detected)}"
        )
        for ms in causal_feedback.miracle_steps_detected:
            sections.append(f"    - {ms}")
        sections.append(
            f"  Foreshadowing payoff score: "
            f"{causal_feedback.foreshadowing_payoff_score:.2f}"
        )
        sections.append(
            f"  Cognitive plausibility: "
            f"{causal_feedback.cognitive_plausibility_score:.2f}"
        )
        if causal_feedback.cognitive_plausibility_details:
            sections.append(
                f"    {causal_feedback.cognitive_plausibility_details}"
            )
        # ctf-calculus pre-flight (Correa & Bareinboim 2025): flag the
        # auditor that these are *not* prose problems — they are static
        # graphical proofs the engine performed before simulation.
        if causal_feedback.cyclic_propagation_clusters:
            sections.append(
                f"  Cyclic propagation clusters: "
                f"{len(causal_feedback.cyclic_propagation_clusters)} "
                f"(static-topology cycles, NOT miracle steps — do not rewrite)"
            )
            for cc in causal_feedback.cyclic_propagation_clusters:
                sections.append(f"    - {cc}")
        if causal_feedback.noisy_or_absorbed_propagations:
            sections.append(
                f"  Noisy-OR absorbed propagations: "
                f"{len(causal_feedback.noisy_or_absorbed_propagations)} "
                f"(probabilistic gate dampened weak impulses, NOT miracle "
                f"steps — do not rewrite)"
            )
            for na in causal_feedback.noisy_or_absorbed_propagations:
                sections.append(f"    - {na}")
        if causal_feedback.rule3_pruned_interventions:
            sections.append(
                f"  Rule-3 pruned interventions: "
                f"{len(causal_feedback.rule3_pruned_interventions)} "
                f"(provably vacuous on the AMWN — surgery cannot reach targets)"
            )
            for ip in causal_feedback.rule3_pruned_interventions:
                sections.append(f"    - {ip}")
        if causal_feedback.rule2_redundant_evidence:
            sections.append(
                f"  Rule-2 redundant evidence: "
                f"{len(causal_feedback.rule2_redundant_evidence)} "
                f"(d-separated from interventions — abduction skipped)"
            )
            for ev in causal_feedback.rule2_redundant_evidence:
                sections.append(f"    - {ev}")
        sections.append("")

    sections.extend(_format_affective_metrics_block(affective_feedback))

    sections.append(
        "=== TASK ===\n"
        "Evaluate the prose above against the causal graph and constraints. "
        "Return the structured quality synthesis. Trust the engine metrics as "
        "ground truth — do not contradict them."
    )

    return "\n".join(sections)


def run_evaluation(
    prose: str,
    brief: CreativeBrief,
    config: AuditorConfig | None = None,
    causal_feedback: Optional[CausalPhysicsFeedback] = None,
    affective_feedback: Optional[AffectiveStateFeedback] = None,
) -> StoryQualitySynthesis:
    """Run a single evaluation pass — literary critique only.

    This is separate from the auditor: no violations, no refinement loop.
    Returns the LLM-judged quality synthesis.
    """
    config = config or AuditorConfig()

    eval_prompt = assemble_evaluation_prompt(
        prose, brief, causal_feedback, affective_feedback,
    )

    logger.info(
        "[Evaluation] Running literary critique: effect=%s prompt_len=%d",
        brief.target_effect, len(eval_prompt),
    )
    # Emit the full evaluation prompt at INFO so the literary-
    # critique instructions are visible alongside the renderer and
    # auditor prompts in the log stream.
    logger.info(
        "[Evaluation] Evaluation prompt (%d chars):\n"
        "========== BEGIN EVALUATION PROMPT ==========\n%s\n"
        "========== END EVALUATION PROMPT ==========",
        len(eval_prompt),
        eval_prompt,
    )

    agent = _build_evaluation_agent(config)
    deps = _EvaluationDeps(evaluation_prompt=eval_prompt)

    model_settings: Dict[str, Any] = {
        "max_tokens": config.max_tokens_audit,
    }
    if config.auditor_temperature != 0.2:
        model_settings["temperature"] = config.auditor_temperature

    try:
        result = agent.run_sync(
            f"Evaluate the prose quality for target effect: {brief.target_effect}.",
            deps=deps,
            model_settings=model_settings if model_settings else None,
        )
    except Exception as exc:
        logger.error("[Evaluation] LLM call failed: %s. Returning empty synthesis.", exc)
        return StoryQualitySynthesis(
            coherence_and_consistency_review=f"Evaluation skipped due to LLM error: {exc}",
        )

    synthesis = result.output
    log_agent_output(logger, "Evaluation", synthesis)
    logger.info(
        "[Evaluation] Quality synthesis complete: %d rewrite directives.",
        len(synthesis.actionable_rewrite_directives),
    )
    return synthesis


# =====================================================================
# Single Audit Pass
# =====================================================================


# Tokenization for the white-elephant semantic-similarity check.
# Strips punctuation, lowercases, drops a small closed set of function
# words so paraphrase detection isn't dominated by them. Keeping the
# stop-word list tiny and explicit (rather than pulling in NLTK) keeps
# the auditor dependency-free; the closed set covers the common
# English determiners / copulas / pronouns that show up in any
# utterance and would otherwise inflate Jaccard overlap.
_PARAPHRASE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "for",
    "in", "on", "at", "by", "with", "from", "as", "is", "am", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "have",
    "has", "had", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "this", "that", "these", "those", "so", "not", "no",
    "yes", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must",
})


def _paraphrase_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords. Returns the
    content-word token list used for shingle / Jaccard similarity.

    Normalisation handles the common cases that would otherwise cause
    false negatives:

    * Apostrophes / smart quotes are *removed* (not converted to
      space) so contractions collapse into their stem (``don't`` \u2192
      ``dont``, ``it's`` \u2192 ``its``). The bare stem is fine for
      shingle matching against another reworded sentence.
    * Unicode dashes (en/em/figure/horizontal-bar) collapse to space
      so hyphenated paraphrases split at the dash boundary.
    * Everything else non-alphanumeric collapses to space.
    """
    if not text:
        return []
    # Strip apostrophes (ASCII + smart) so contractions stem cleanly.
    cleaned = re.sub(r"[\u2018\u2019\u02bc']+", "", text.lower())
    # Collapse unicode dashes to spaces before the broad punctuation
    # pass so e.g. ``mother-in-law`` tokenises as three words.
    cleaned = re.sub(r"[\u2010-\u2015\u2212-]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    return [
        t for t in cleaned.split()
        if t and t not in _PARAPHRASE_STOPWORDS and len(t) > 1
    ]


def _shingle_jaccard(a: list[str], b: list[str], *, n: int = 3) -> float:
    """Compute Jaccard similarity over content n-grams (default
    trigrams) between two token lists. Returns 0.0 when either side
    has fewer than ``n`` tokens. Used as a cheap, dependency-free
    paraphrase detector for white-elephant exclusions."""
    if len(a) < n or len(b) < n:
        return 0.0
    sa = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    sb = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# Threshold above which a candidate sentence is flagged as a
# paraphrase of a withheld/pruned utterance. Two independent scorers
# are run and either tripping flags a leak:
#
# * Trigram-Jaccard \u2265 0.45 \u2014 catches near-verbatim
#   rewordings that share half the 3-word phrases on content tokens.
# * Content-word containment \u2265 0.6 with \u22654 shared tokens
#   \u2014 catches looser paraphrases where the distinctive
#   nouns/verbs of the canonical utterance all reappear in a single
#   sentence (e.g. "poisoned/chalice/Macbeth/drank" reordered into
#   author-voice prose).
#
# Pure verbatim leaks are caught earlier by the substring check; this
# layer exists for the white-elephant case where the renderer
# reworded the line to bypass the verbatim filter.
_PARAPHRASE_JACCARD_THRESHOLD: float = 0.45
_PARAPHRASE_CONTAINMENT_THRESHOLD: float = 0.6
_PARAPHRASE_CONTAINMENT_MIN_OVERLAP: int = 4
# Minimum content-token count on the candidate sentence before
# paraphrase scoring runs. Short fragments (<6 content tokens) are
# noisy and trip the shingle scorer on incidental overlap.
_PARAPHRASE_MIN_TOKENS: int = 6


def _paraphrase_match(
    canonical: str, prose_sentences: list[str],
) -> Optional[tuple[str, float]]:
    """Return ``(sentence, score)`` for the highest-scoring sentence
    that exceeds either the trigram-Jaccard or content-containment
    threshold, or ``None``. ``score`` is the larger of the two
    sub-scores so callers can surface a single number.

    Used for white-elephant paraphrase detection: catches the case
    where the renderer reworded a withheld / pruned utterance into a
    sentence that wouldn't trip the verbatim substring check.
    """
    canon_tokens = _paraphrase_tokens(canonical)
    if len(canon_tokens) < _PARAPHRASE_MIN_TOKENS:
        return None
    canon_set = set(canon_tokens)
    best: Optional[tuple[str, float]] = None
    for sent in prose_sentences:
        s_tokens = _paraphrase_tokens(sent)
        if len(s_tokens) < _PARAPHRASE_MIN_TOKENS:
            continue
        jaccard = _shingle_jaccard(canon_tokens, s_tokens)
        s_set = set(s_tokens)
        overlap = len(canon_set & s_set)
        containment = overlap / len(canon_set) if canon_set else 0.0
        flagged_by_jaccard = jaccard >= _PARAPHRASE_JACCARD_THRESHOLD
        flagged_by_containment = (
            containment >= _PARAPHRASE_CONTAINMENT_THRESHOLD
            and overlap >= _PARAPHRASE_CONTAINMENT_MIN_OVERLAP
        )
        if not (flagged_by_jaccard or flagged_by_containment):
            continue
        score = max(jaccard, containment)
        if best is None or score > best[1]:
            best = (sent.strip(), score)
    return best


def _withheld_utterance_leak_violations(
    prose: str,
    world_state: Optional[WorldStateV1],
    syuzhet_anchor: Optional[int],
) -> List[AuditViolation]:
    """Deterministic pre-check: scan prose for verbatim leaks of
    withheld utterance content.

    A withheld utterance is any ``EventNode(event_type='utterance')``
    whose ``syuzhet_index > syuzhet_anchor``. The check is intentionally
    conservative — it only flags substring matches of length >= 12
    characters, which avoids false positives on common phrases like
    "yes" or character names that happen to appear in withheld lines.
    The LLM auditor remains responsible for paraphrase-level leaks.
    """
    if world_state is None or syuzhet_anchor is None or not prose:
        return []
    issues: List[AuditViolation] = []
    prose_lower = prose.lower()
    prose_sentences = re.split(r"(?<=[.!?])\s+", prose)
    seen: set[str] = set()
    for evt in getattr(world_state, "events", []) or []:
        if getattr(evt, "event_type", None) != "utterance":
            continue
        if getattr(evt, "syuzhet_index", -1) <= syuzhet_anchor:
            continue
        content = (getattr(evt, "content", None) or "").strip()
        if len(content) < 12:
            continue
        needle = content.lower()
        if needle in prose_lower and needle not in seen:
            seen.add(needle)
            issues.append(AuditViolation(
                violation_type="withheld_utterance_leak",
                severity="critical",
                description=(
                    f"Prose verbatim quotes withheld utterance "
                    f"{evt.id} (syuzhet={evt.syuzhet_index} > anchor "
                    f"{syuzhet_anchor}); content must not surface yet."
                ),
                evidence_quote=content[:200],
                feedback=(
                    f"Remove the line attributed to {getattr(evt, 'speaker_id', 'unknown')}. "
                    f"This utterance happens later in narration order and "
                    f"the reader has not yet encountered it."
                ),
            ))
            continue
        # Paraphrase check: a trigram-overlap rewording also counts
        # as a leak when the verbatim substring isn't present.
        match = _paraphrase_match(content, prose_sentences)
        if match is not None and needle not in seen:
            sent, score = match
            seen.add(needle)
            issues.append(AuditViolation(
                violation_type="withheld_utterance_leak",
                severity="major",
                description=(
                    f"Prose paraphrases withheld utterance "
                    f"{evt.id} (syuzhet={evt.syuzhet_index} > anchor "
                    f"{syuzhet_anchor}; trigram Jaccard={score:.2f}); "
                    f"the content must not surface yet, even reworded."
                ),
                evidence_quote=sent[:200],
                feedback=(
                    f"Reword or remove the sentence: it conveys the "
                    f"same content as the withheld utterance attributed "
                    f"to {getattr(evt, 'speaker_id', 'unknown')}."
                ),
            ))
    return issues


def _cascade_exclusion_leak_violations(
    prose: str,
    brief: CreativeBrief,
    world_state: Optional[WorldStateV1],
) -> List[AuditViolation]:
    """Deterministic scan for ctf-calculus EXCLUSION leaks.

    Walks ``brief.constraints`` for the HARD blocks emitted by
    :func:`shadow_loom.generation._build_exclusion_constraints` and
    :func:`shadow_loom.generation._build_cascade_exclusion_constraints`
    and verifies the prose did not breach them. Three independent
    checks, each high-precision (substring length \u2265 12 / explicit
    id-token match) so the refinement loop never chases false
    positives:

    * **pruned_utterance_leak** \u2014 do-surgery severed the
      utterance's provenance; its canonical content must not
      surface verbatim in the {intervened, counterfactual} branch.
    * **disabled_channel_leak** \u2014 the channel id was severed by
      do-surgery; prose must not name the ``CHN_`` token nor route
      a line through it.
    * **blocked_propagation_leak** \u2014 the engine's propagation
      log marked ``(node, trait)`` as resisted; a sentence that
      co-mentions node+trait is flagged for triage.

    The check is a no-op when ``world_state`` is missing or the brief
    carries no exclusion constraints.
    """
    if not prose or brief is None or world_state is None:
        return []
    issues: List[AuditViolation] = []
    prose_lower = prose.lower()

    pruned_utt_ids: List[str] = []
    disabled_chan_ids: List[str] = []
    blocked_pairs: List[str] = []
    for c in (brief.constraints or []):
        ev = getattr(c, "evidence", None) or {}
        if not isinstance(ev, dict):
            continue
        pruned_utt_ids.extend(ev.get("pruned_utterance_event_ids", []) or [])
        disabled_chan_ids.extend(ev.get("disabled_channel_ids", []) or [])
        blocked_pairs.extend(ev.get("blocked_node_traits", []) or [])

    # 1. Pruned utterance content leaks (verbatim, >=12 chars).
    if pruned_utt_ids:
        events_by_id = {e.id: e for e in getattr(world_state, "events", []) or []}
        prose_sentences = re.split(r"(?<=[.!?])\s+", prose)
        seen: set[str] = set()
        for uid in pruned_utt_ids:
            evt = events_by_id.get(uid)
            if evt is None:
                continue
            content = (getattr(evt, "content", None) or "").strip()
            if len(content) < 12:
                continue
            needle = content.lower()
            if needle in prose_lower and needle not in seen:
                seen.add(needle)
                issues.append(AuditViolation(
                    violation_type="pruned_utterance_leak",
                    severity="critical",
                    description=(
                        f"Prose verbatim quotes do-surgery-pruned utterance "
                        f"{uid}; the line was severed by the do-calculus "
                        f"surgery and must not exist in this branch."
                    ),
                    evidence_quote=content[:200],
                    feedback=(
                        f"Remove the line attributed to "
                        f"{getattr(evt, 'speaker_id', 'unknown')}. "
                        f"This utterance's provenance was severed by the "
                        f"do-surgery; if the same speaker/addressee pair "
                        f"would still talk, invent a NEW line about a "
                        f"DIFFERENT subject consistent with the changed "
                        f"conditions."
                    ),
                ))
                continue
            # Paraphrase check (white-elephant exclusion): the
            # rewriter sometimes reworded the line to avoid the
            # verbatim filter; trigram-Jaccard catches that.
            match = _paraphrase_match(content, prose_sentences)
            if match is not None and needle not in seen:
                sent, score = match
                seen.add(needle)
                issues.append(AuditViolation(
                    violation_type="pruned_utterance_leak",
                    severity="major",
                    description=(
                        f"Prose paraphrases do-surgery-pruned utterance "
                        f"{uid} (trigram Jaccard={score:.2f}); the line "
                        f"was severed and must not exist in this branch, "
                        f"even reworded."
                    ),
                    evidence_quote=sent[:200],
                    feedback=(
                        f"Reword or remove the sentence: it conveys the "
                        f"same content as the pruned utterance "
                        f"attributed to "
                        f"{getattr(evt, 'speaker_id', 'unknown')}."
                    ),
                ))

    # 2. Disabled channel id / name leaks.
    if disabled_chan_ids:
        chans = getattr(world_state, "channels", {}) or {}
        seen_ch: set[str] = set()
        for cid in disabled_chan_ids:
            if not cid or cid in seen_ch:
                continue
            if cid in prose:  # case-sensitive: CHN_ is uppercase
                seen_ch.add(cid)
                issues.append(AuditViolation(
                    violation_type="disabled_channel_leak",
                    severity="critical",
                    description=(
                        f"Prose names disabled channel {cid}; the do-"
                        f"surgery severed this channel and it does not "
                        f"exist in this branch."
                    ),
                    evidence_quote=cid,
                    feedback=(
                        f"Remove every reference to {cid}. Route the "
                        f"affected speech act through a different "
                        f"channel (or omit it) consistent with the "
                        f"intervened/counterfactual world."
                    ),
                ))
                continue
            ch = chans.get(cid) if isinstance(chans, dict) else None
            name = getattr(ch, "name", None) if ch is not None else None
            if isinstance(name, str) and len(name) >= 6:
                low = name.lower()
                if low in prose_lower and low not in seen_ch:
                    seen_ch.add(low)
                    issues.append(AuditViolation(
                        violation_type="disabled_channel_leak",
                        severity="major",
                        description=(
                            f"Prose names disabled channel "
                            f"\"{name}\" ({cid}); the do-surgery severed "
                            f"this channel."
                        ),
                        evidence_quote=name,
                        feedback=(
                            f"Drop the reference to \"{name}\". The "
                            f"channel was severed by the do-surgery."
                        ),
                    ))

    # 3. Blocked-propagation leak: NARROW RESTORATION (R2-3, 2026-05-29).
    #
    # The original per-sentence node+trait co-mention scan was removed
    # because it false-positived on prose the brief explicitly asked
    # for (e.g. "Ken's stutter stayed silent, swallowed by his
    # controlled breathing"). The renderer is required to depict the
    # blocked propagation on-page; the deterministic scanner could
    # not distinguish "rendered as resisted" from "rendered as
    # propagated".
    #
    # The narrow restoration only fires when ALL of the following hold
    # in the same sentence:
    #   (a) the node identifier (ENT_/OBJ_/LOC_ id, OR its
    #       canonical name) appears, AND
    #   (b) the trait token appears, AND
    #   (c) NO resistance / negation marker appears anywhere in the
    #       sentence (a closed lexicon of high-precision tokens that
    #       the BLOCKED PROPAGATIONS directive instructs the renderer
    #       to use when staging resistance).
    # Severity is held at "minor" so even a mis-fire never blocks
    # refinement; the LLM auditor remains the source of truth for
    # this category.
    if blocked_pairs:
        prose_sentences = re.split(r"(?<=[.!?])\s+", prose)
        # High-precision resistance / negation lexicon. Any of these
        # in a sentence flips it to "rendered as resisted" and the
        # leak is suppressed.
        _RESISTANCE_MARKERS = (
            " not ", " no ", " never ", " none ", " n't ", " nor ",
            "without", "absent", "lacking",
            # explicit resistance verbs
            "resist", "withstood", "withstand", "swallow", "swallowed",
            "suppress", "suppressed", "controlled", "contained",
            "kept ", "stayed ", "remained ", "held ", "steady",
            "unchang", "unmoved", "unshaken", "unfaltering",
            "calm", "calmly", "still ", "silent",
        )
        # Build a node-id \u2192 lowercased canonical name lookup so a
        # sentence that mentions the name (not the raw id) still
        # qualifies as "mentions the node".
        ents = getattr(world_state, "entities", {}) or {}
        objs = getattr(world_state, "objects", {}) or {}
        locs = getattr(world_state, "locations", {}) or {}
        node_names: Dict[str, str] = {}
        for _m in (ents, objs, locs):
            if not isinstance(_m, dict):
                continue
            for _nid, _node in _m.items():
                _nm = getattr(_node, "name", None)
                if isinstance(_nm, str) and len(_nm.strip()) >= 3:
                    node_names[_nid] = _nm.strip().lower()
        seen_pairs: set[str] = set()
        for pair in blocked_pairs:
            if not isinstance(pair, str) or "." not in pair or pair in seen_pairs:
                continue
            node_id, _, trait = pair.partition(".")
            if not (node_id and trait) or len(trait) < 4:
                # Trait tokens shorter than 4 chars are too noisy
                # ("on", "in", "up", etc.) — skip rather than risk
                # mis-firing on common English words.
                continue
            trait_low = trait.lower()
            node_low = node_id.lower()
            name_low = node_names.get(node_id)
            for sentence in prose_sentences:
                sent_low = sentence.lower()
                # (a) node mention — either id or canonical name.
                mentions_node = (node_low in sent_low) or (
                    name_low is not None and name_low in sent_low
                )
                if not mentions_node:
                    continue
                # (b) trait mention.
                if trait_low not in sent_low:
                    continue
                # (c) ANY resistance / negation marker \u2192 suppress.
                padded = f" {sent_low} "
                if any(m in padded for m in _RESISTANCE_MARKERS):
                    continue
                seen_pairs.add(pair)
                issues.append(AuditViolation(
                    violation_type="blocked_propagation_leak",
                    severity="minor",
                    description=(
                        f"Prose co-mentions {node_id} and trait "
                        f"\"{trait}\" without any resistance / negation "
                        f"marker; the engine's BLOCKED PROPAGATIONS "
                        f"directive says this (node, trait) pair "
                        f"resisted the cascade and must be depicted as "
                        f"unchanged. The LLM auditor can override this "
                        f"if the sentence in fact stages resistance."
                    ),
                    evidence_quote=sentence.strip()[:200],
                    feedback=(
                        f"Render {node_id}'s {trait} as visibly RESISTED "
                        f"or UNCHANGED in this sentence (e.g. \"stayed\", "
                        f"\"held\", \"unmoved\", \"did not\"). The "
                        f"propagation was blocked by the engine, so the "
                        f"on-page depiction must show the resistance, "
                        f"not the propagation."
                    ),
                ))
                break

    return issues


def _undeclared_element_violations(
    prose: str,
    world_state: Optional[WorldStateV1],
    introduced: Optional[IntroducedElements],
) -> List[AuditViolation]:
    """Deterministic pre-check: flag prose references to elements that
    do not resolve to either ``world_state`` or
    ``GeneratedScene.introduced_elements``.

    Two flavours of reference are checked, both intentionally
    conservative to avoid false positives that would force the
    refinement loop to chase its tail:

      1. **Project-prefix ids in prose.** The renderer is instructed
         not to emit raw ids like ``ENT_FOO`` or ``LOC_BAR`` into
         prose, but if it does they MUST resolve. Any
         ``[A-Z]{2,5}_[A-Z0-9_]+`` token whose id is not in
         ``world_state`` or ``introduced_elements`` is a hard
         violation.

      2. **Multi-word proper-noun names.** Names like ``Roderigo
         Smith`` or ``Lady Macbeth`` are treated as candidate
         entity / location / object references. We deliberately do
         NOT flag bare single capitalised tokens \u2014 those collide
         too readily with sentence starts, days of the week, common
         vocatives, and well-known place adjectives, all of which
         the LLM auditor is better positioned to triage. The
         multi-word case is high-precision because few non-name
         capitalised bigrams survive normal English prose.

    The check returns no violations when ``world_state`` is not
    supplied (the caller has chosen not to enforce reference
    integrity in this audit pass).
    """
    if world_state is None or not prose:
        return []

    # ---------- canonical name + id sets ----------
    known_ids: set[str] = set()
    known_names_lower: set[str] = set()

    def _add_named_dict(d: Dict[str, Any] | None, name_attrs: tuple = ("name",)) -> None:
        if not d:
            return
        for nid, node in d.items():
            if nid:
                known_ids.add(nid)
            for attr in name_attrs:
                v = getattr(node, attr, None)
                if isinstance(v, str) and v.strip():
                    known_names_lower.add(v.strip().lower())
            # Aliases / display names if present.
            for alias_attr in ("aliases", "alternative_names"):
                aliases = getattr(node, alias_attr, None) or []
                for a in aliases:
                    if isinstance(a, str) and a.strip():
                        known_names_lower.add(a.strip().lower())

    _add_named_dict(getattr(world_state, "entities", None))
    _add_named_dict(getattr(world_state, "locations", None))
    _add_named_dict(getattr(world_state, "objects", None))
    _add_named_dict(getattr(world_state, "world_traits", None))
    _add_named_dict(getattr(world_state, "channels", None))
    # Propositions / concerns are referenced by id only \u2014 names not
    # in scope for the proper-noun pass but their ids count.
    for attr in ("propositions", "concerns"):
        coll = getattr(world_state, attr, None) or []
        if isinstance(coll, dict):
            coll = coll.values()
        for node in coll:
            nid = getattr(node, "id", None) or getattr(node, "proposition_id", None) \
                or getattr(node, "concern_id", None)
            if nid:
                known_ids.add(nid)
    for evt in getattr(world_state, "events", []) or []:
        eid = getattr(evt, "id", None)
        if eid:
            known_ids.add(eid)

    # ---------- declarations from this scene ----------
    if introduced is not None:
        known_ids.update(introduced.declared_ids())
        known_names_lower.update(n.lower() for n in introduced.declared_names())

    # ---------- common-English noise filter for proper nouns ----------
    # Title-cased tokens that legitimately appear at sentence starts or
    # as common vocatives without being names of world elements. Kept
    # conservative; the LLM auditor catches the rest.
    _STOP_TITLE_TOKENS = frozenset({
        "i", "the", "a", "an", "and", "but", "or", "so", "yet", "for", "nor",
        "he", "she", "it", "they", "we", "you", "his", "her", "its", "their",
        "this", "that", "these", "those", "there", "here", "now", "then",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "god", "lord", "sir", "madam", "mr", "mrs", "miss", "ms",
        "yes", "no", "ok", "okay",
    })

    issues: List[AuditViolation] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()

    # Pass 1 \u2014 raw ids in prose.
    for m in re.finditer(r"\b([A-Z]{2,5}_[A-Z0-9_]+)\b", prose):
        token = m.group(1)
        if token in known_ids or token in seen_ids:
            continue
        seen_ids.add(token)
        issues.append(AuditViolation(
            violation_type="undeclared_element",
            severity="critical",
            description=(
                f"Prose references id `{token}` which is not present in "
                f"the input world state and was not declared in "
                f"``introduced_elements``."
            ),
            evidence_quote=prose[max(0, m.start() - 40): m.end() + 40],
            feedback=(
                f"Either remove `{token}` from the prose (ids should "
                f"not appear in narrative text in any case), use an "
                f"existing referent, or \u2014 if the element is genuinely "
                f"new \u2014 add a corresponding entry under "
                f"``introduced_elements`` with a justification."
            ),
        ))

    # Pass 2 \u2014 multi-word proper-noun candidates (Title-cased
    # bigrams / trigrams). Skip if the leading token is a stop word
    # (handles "The Forest" at sentence starts where "Forest" alone
    # is the only canonical name).
    candidate_pattern = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"
    )
    for m in candidate_pattern.finditer(prose):
        full = m.group(1)
        full_lower = full.lower()
        if full_lower in known_names_lower or full_lower in seen_names:
            continue
        # Strip a leading stop-word title token if present
        # ("The Forest" \u2192 "Forest") and re-check.
        tokens = full.split()
        if tokens and tokens[0].lower() in _STOP_TITLE_TOKENS:
            tail = " ".join(tokens[1:])
            if tail.lower() in known_names_lower:
                continue
            # Single trailing token after stop \u2014 too noisy to flag.
            if len(tokens) <= 2:
                continue
        seen_names.add(full_lower)
        issues.append(AuditViolation(
            violation_type="undeclared_element",
            severity="major",
            description=(
                f"Prose references the name `{full}` which does not "
                f"resolve to any entity, location, object, world trait, "
                f"or channel in the input world state and was not "
                f"declared in ``introduced_elements``."
            ),
            evidence_quote=prose[max(0, m.start() - 40): m.end() + 40],
            feedback=(
                f"If `{full}` was meant to refer to an existing element, "
                f"correct the spelling to match the canonical name. If "
                f"`{full}` is a deliberate new addition, declare it in "
                f"``introduced_elements`` with a stable id, role, and "
                f"justification \u2014 the merge will then materialise it "
                f"into the next world-state revision. Otherwise remove "
                f"the name."
            ),
        ))
    return issues


# ----------------------------------------------------------------------
# Reuse-first / justification deterministic check
# ----------------------------------------------------------------------

# Boilerplate phrases that do NOT count as a real justification. Kept
# small and high-precision so the LLM auditor still owns the paraphrase
# cases; this list catches the most common renderer evasions.
_BOILERPLATE_JUSTIFICATIONS: tuple[str, ...] = (
    "needed for the scene",
    "needed for this scene",
    "required by the prompt",
    "required by the brief",
    "required by the constraints",
    "to advance the plot",
    "for narrative purposes",
    "for dramatic purposes",
    "for the story",
    "necessary for the scene",
    "necessary for the story",
    "context demands",
    "the scene calls for",
    "the prompt calls for",
    "n/a",
    "none",
    "no existing element fits",  # the *empty* claim with no reasoning
)

# Words / phrases the renderer is expected to use when actually
# articulating reuse-first reasoning ("no existing X because Y",
# "considered ENT_FOO but ...", "the existing locations are all
# indoors and the scene needs an outdoor ..."). Presence of any one
# token is treated as a weak signal that the justification engaged
# with the existing inventory.
_JUSTIFICATION_REUSE_TOKENS: tuple[str, ...] = (
    "existing", "considered", "candidate", "reused",
    "instead of", "rather than", "no ", "none of",
    "ent_", "loc_", "obj_", "wt_", "chn_", "prop_", "ccn_",
    # Sub-space / containment justifications for contextually entailed
    # locations (aircraft cabin within airport, courtroom within
    # courthouse, etc.) that don't reference a specific LOC_ id.
    "sub-space", "sub space", "part of", "within", "inside",
    "contained in", "section of", "area of", "room in", "cabin",
    "entailed", "implied by",
)


def _unjustified_introduction_violations(
    introduced: Optional[IntroducedElements],
    world_state: Optional[WorldStateV1],
) -> List[AuditViolation]:
    """Deterministic check that every per-cycle introduction carries a
    concrete reuse-first justification.

    Three failure modes are flagged:

      1. **Empty justification** (``critical``) — the spec's
         ``justification`` field is missing or whitespace.
      2. **Boilerplate justification** (``major``) — the
         justification matches one of ``_BOILERPLATE_JUSTIFICATIONS``
         AND contains none of ``_JUSTIFICATION_REUSE_TOKENS``. The
         renderer wrote a generic line that does not engage with the
         existing world inventory.
      3. **Name collision with an existing element** (``critical``)
         — the spec's ``name`` matches (case-insensitive) an
         existing entity / location / object / world-trait / channel
         in ``world_state``. The renderer should have reused the
         existing element instead of minting a duplicate id.

    The check runs only when ``introduced`` is non-empty. When
    ``world_state`` is ``None`` the name-collision pass is skipped
    (the boilerplate / empty passes still run).
    """
    if introduced is None or introduced.is_empty():
        return []

    issues: List[AuditViolation] = []

    # ---------- existing names (lower-cased) for collision pass ----------
    existing_names_lower: set[str] = set()
    if world_state is not None:
        for attr in ("entities", "locations", "objects", "world_traits", "channels"):
            coll = getattr(world_state, attr, None) or {}
            if isinstance(coll, dict):
                nodes = coll.values()
            else:
                nodes = coll
            for node in nodes:
                v = getattr(node, "name", None)
                if isinstance(v, str) and v.strip():
                    existing_names_lower.add(v.strip().lower())

    # The seven named-spec collections; events are excluded from the
    # name-collision pass (their ``name`` carries an event description,
    # not a noun referent).
    named_kinds: tuple[tuple[str, list, bool], ...] = (
        ("entity", introduced.entities, True),
        ("location", introduced.locations, True),
        ("object", introduced.objects, True),
        ("world_trait", introduced.world_traits, True),
        ("channel", introduced.channels, True),
        ("proposition", introduced.propositions, False),
        ("concern", introduced.concerns, False),
        ("event", introduced.events, False),
    )

    for kind, specs, check_name_collision in named_kinds:
        for spec in specs:
            sid = getattr(spec, "id", "<missing-id>")
            sname = getattr(spec, "name", "") or ""
            justification = (getattr(spec, "justification", "") or "").strip()

            # 1. Empty / whitespace.
            if not justification:
                issues.append(AuditViolation(
                    violation_type="unjustified_introduction",
                    severity="critical",
                    description=(
                        f"Introduced {kind} `{sid}` (\"{sname}\") has an "
                        f"empty ``justification``. Every per-cycle "
                        f"introduction must explain why no existing "
                        f"world-model element fits."
                    ),
                    evidence_quote="",
                    feedback=(
                        f"Populate ``introduced_elements.{kind}s[id={sid}]"
                        f".justification`` with a one-sentence rationale "
                        f"that names which existing {kind}(s) you "
                        f"considered and why each was insufficient. If "
                        f"an existing {kind} would have served, reuse "
                        f"its id and remove this declaration."
                    ),
                ))
                continue

            # 2. Boilerplate phrasing with no reuse-first reasoning.
            j_lower = justification.lower()
            looks_boilerplate = any(
                phrase in j_lower for phrase in _BOILERPLATE_JUSTIFICATIONS
            )
            engages_reuse = any(
                tok in j_lower for tok in _JUSTIFICATION_REUSE_TOKENS
            )
            if looks_boilerplate and not engages_reuse:
                issues.append(AuditViolation(
                    violation_type="unjustified_introduction",
                    severity="major",
                    description=(
                        f"Introduced {kind} `{sid}` (\"{sname}\") has a "
                        f"boilerplate ``justification`` (\"{justification}\") "
                        f"that does not engage with the existing world "
                        f"inventory. Reuse-first policy requires the "
                        f"renderer to name the existing candidates it "
                        f"considered."
                    ),
                    evidence_quote=justification,
                    feedback=(
                        f"Rewrite the justification for `{sid}` to name "
                        f"specific existing {kind}(s) you considered (by "
                        f"id or display name) and explain why each was "
                        f"insufficient — e.g. \"considered ENT_FOO but "
                        f"their location at fabula_time conflicts\", or "
                        f"\"no existing {kind} carries the required role\"."
                    ),
                ))

            # 3. Name collision with an existing element.
            if (
                check_name_collision
                and sname.strip()
                and sname.strip().lower() in existing_names_lower
            ):
                issues.append(AuditViolation(
                    violation_type="unjustified_introduction",
                    severity="critical",
                    description=(
                        f"Introduced {kind} `{sid}` reuses the display "
                        f"name \"{sname}\" of an existing {kind} already "
                        f"in the world state. Reuse the existing id "
                        f"instead of minting a duplicate."
                    ),
                    evidence_quote=sname,
                    feedback=(
                        f"Remove the ``introduced_elements.{kind}s`` "
                        f"entry for `{sid}` and refer to the existing "
                        f"{kind} by its canonical id in the prose. If "
                        f"the new element is *different* from the "
                        f"existing one despite sharing the name, "
                        f"disambiguate the display name (e.g. "
                        f"\"{sname} the Younger\")."
                    ),
                ))

    return issues


def _prevented_event_reenacted_violations(
    prose: str,
    brief: CreativeBrief,
    world_state: Optional[WorldStateV1],
) -> List[AuditViolation]:
    """Deterministic defence-in-depth for do-prevention surgeries.

    For every event id surfaced by the brief in
    ``pruned_utterance_event_ids`` whose ``EventNode.event_type`` is
    NOT ``"utterance"`` (utterance content leaks are already covered
    by :func:`_cascade_exclusion_leak_violations`), flag the prose
    if it either:

      * mentions the event's id token verbatim
        (e.g. ``EVT_DUNCAN_MURDER``), OR
      * verbatim-quotes a high-signal substring (\u2265 18 chars) of
        the event's ``description``.

    Paraphrase / pronoun cases remain the LLM auditor's remit. The
    check fires at ``critical`` severity because the prose is
    directly contradicting the engine's do-surgery ledger \u2014 the
    same event the brief lists under ``SEVERED CAUSAL CHAINS``
    is being staged on the page.

    Catches three failure modes:

      * Brief bug \u2014 a closure gap escapes Step B.6 in
        :mod:`shadow_loom.causal_physics`.
      * Renderer hallucination \u2014 the LLM ignores the SEVERED
        block.
      * Fixture data bug \u2014 a sufficient-cause edge that should
        have been a precondition keeps the descendant alive.
    """
    if not prose or brief is None or world_state is None:
        return []

    pruned_ids: set[str] = set()
    for c in (brief.constraints or []):
        ev = getattr(c, "evidence", None) or {}
        if not isinstance(ev, dict):
            continue
        for eid in (ev.get("pruned_utterance_event_ids", []) or []):
            if isinstance(eid, str):
                pruned_ids.add(eid)
    if not pruned_ids:
        return []

    events_by_id = {e.id: e for e in getattr(world_state, "events", []) or []}
    prose_lower = prose.lower()
    issues: List[AuditViolation] = []
    seen: set[str] = set()

    for eid in pruned_ids:
        evt = events_by_id.get(eid)
        if evt is None:
            continue
        # Utterance leaks are owned by _cascade_exclusion_leak_violations.
        if getattr(evt, "event_type", None) == "utterance":
            continue
        if eid in seen:
            continue

        # Check 1: verbatim id token (word-boundary aware).
        # Event ids are all-caps underscore tokens (EVT_ENTITY_ACTION);
        # they essentially never appear in natural prose, but use a
        # non-identifier-character boundary to avoid an EVT_FOO id
        # matching inside a longer token like EVT_FOOBAR.
        _eid_lower = eid.lower()
        _eid_pat = re.compile(
            r"(?<![a-z0-9_])" + re.escape(_eid_lower) + r"(?![a-z0-9_])"
        )
        if _eid_pat.search(prose_lower):
            seen.add(eid)
            issues.append(AuditViolation(
                violation_type="prevented_event_reenacted",
                severity="critical",
                description=(
                    f"Prose references do-surgery-pruned event {eid} by "
                    f"its id token; the event was severed by the "
                    f"do-calculus surgery (and the chain_reaction "
                    f"descendant closure) and must not appear staged "
                    f"in this branch."
                ),
                evidence_quote=eid,
                feedback=(
                    f"Remove the reference to {eid}. The event was "
                    f"erased by the do-surgery; if downstream "
                    f"characters need an explanation for the absence, "
                    f"render the gap (the action not happening, the "
                    f"frustrated plan) rather than the event itself."
                ),
            ))
            continue

        # Check 2: verbatim substring of the event description.
        desc = (getattr(evt, "description", None) or "").strip()
        if len(desc) < 18:
            continue
        needle = desc.lower()
        if needle in prose_lower:
            seen.add(eid)
            issues.append(AuditViolation(
                violation_type="prevented_event_reenacted",
                severity="critical",
                description=(
                    f"Prose verbatim quotes the description of "
                    f"do-surgery-pruned event {eid}; the event was "
                    f"severed by the do-calculus surgery and must not "
                    f"be staged in this branch."
                ),
                evidence_quote=desc[:200],
                feedback=(
                    f"Remove or rewrite the passage that stages {eid}. "
                    f"The event was erased by the do-surgery; render "
                    f"the consequence of its non-occurrence (the gap, "
                    f"the absence) rather than the event itself."
                ),
            ))

    return issues


def _event_copresence_violations(
    prose: str,
    brief: CreativeBrief,
    world_state: Optional[WorldStateV1],
    introduced: Optional[IntroducedElements] = None,
) -> List[AuditViolation]:
    """Deterministic co-presence + event-location auditor pass.

    For every spatial ConstraintBlock emitted by
    :func:`build_event_copresence_constraints` (identified by an
    ``evidence`` dict carrying ``event_id`` + ``at_location_id`` +
    ``must_be_present``), check three rules:

      * ``event_copresence_violation`` \u2014 a bound participant
        (``must_be_present``) is named in prose at a location that
        is not the event's ``at_location_id``. Conservative: only
        flagged when prose contains both the participant name AND a
        non-event location name within ~120 chars (verbatim phrase
        check).
      * ``event_copresence_omission`` \u2014 an entity in
        ``must_not_be_present`` is named in prose AT the event
        scene (verbatim mention of the entity name within ~120 chars
        of the event location's name).
      * ``event_location_mismatch`` \u2014 prose names the event's id /
        description AND a *different* canonical location name within
        ~120 chars.

    The LLM auditor remains responsible for paraphrase / pronoun
    cases. This pre-check guards the high-precision verbatim cases
    so the refinement loop cannot converge on prose that contradicts
    the engine's spatial ledger.
    """
    if world_state is None or not prose:
        return []
    prose_lower = prose.lower()
    locations = world_state.locations or {}
    entities = world_state.entities or {}
    issues: List[AuditViolation] = []
    seen: set[tuple[str, str, str]] = set()

    # Build supplemental name maps from introduced_elements so that
    # newly declared entities and locations resolve in _name() below.
    _ie_entity_names: Dict[str, str] = {}
    _ie_location_names: Dict[str, str] = {}
    if introduced is not None:
        for spec in introduced.entities:
            _ie_entity_names[spec.id] = spec.name or spec.id
        for spec in introduced.locations:
            _ie_location_names[spec.id] = spec.name or spec.id

    def _name(nid: str) -> str:
        if nid in entities:
            return getattr(entities[nid], "name", nid) or nid
        if nid in locations:
            return getattr(locations[nid], "name", nid) or nid
        if nid in _ie_entity_names:
            return _ie_entity_names[nid]
        if nid in _ie_location_names:
            return _ie_location_names[nid]
        return nid

    def _name_positions(name: str) -> list[int]:
        if not name:
            return []
        n = name.strip().lower()
        if len(n) < 3:
            return []
        out: list[int] = []
        start = 0
        while True:
            i = prose_lower.find(n, start)
            if i < 0:
                break
            out.append(i)
            start = i + len(n)
        return out

    for block in (brief.constraints or []):
        if block.constraint_type != "spatial":
            continue
        ev = block.evidence or {}
        evt_id = ev.get("event_id")
        loc_id = ev.get("at_location_id")
        bound = ev.get("must_be_present") or []
        absent = ev.get("must_not_be_present") or []
        if not evt_id or not loc_id:
            continue
        loc_name = _name(loc_id)
        loc_positions = _name_positions(loc_name)

        # Rule 1: bound participant named near a different location name.
        for pid in bound:
            pname = _name(pid)
            p_positions = _name_positions(pname)
            if not p_positions:
                continue
            triggered = False
            for pi in p_positions:
                if triggered:
                    break
                # Find the closest other-location name within +/-120
                for oloc_id, oloc in (locations.items() if isinstance(locations, dict) else []):
                    if oloc_id == loc_id:
                        continue
                    other_name = getattr(oloc, "name", oloc_id) or oloc_id
                    if len(other_name) < 3:
                        continue
                    for oi in _name_positions(other_name):
                        if abs(oi - pi) <= 120:
                            key = ("event_copresence_violation", evt_id, pid)
                            if key in seen:
                                triggered = True
                                break
                            seen.add(key)
                            issues.append(AuditViolation(
                                violation_type="event_copresence_violation",
                                severity="major",
                                description=(
                                    f"Prose stages `{pid}` ({pname}) at "
                                    f"`{oloc_id}` ({other_name}), but event "
                                    f"`{evt_id}` is anchored at `{loc_id}` "
                                    f"({loc_name}) and binds `{pid}` as "
                                    f"co-present there."
                                ),
                                evidence_quote=prose[max(0, pi - 40): pi + len(pname) + 40],
                                feedback=(
                                    f"Either re-locate `{pid}` to `{loc_id}` "
                                    f"({loc_name}) for event `{evt_id}` or "
                                    f"render their participation as channel-"
                                    f"mediated (only valid when the event has "
                                    f"a `via_channel_id`). Do NOT show them "
                                    f"acting at `{oloc_id}`."
                                ),
                            ))
                            triggered = True
                            break
                    if triggered:
                        break

        # Rule 2: phantom witness (must_not_be_present named near the
        # event location).
        if loc_positions:
            for pid in absent:
                pname = _name(pid)
                p_positions = _name_positions(pname)
                if not p_positions:
                    continue
                for pi in p_positions:
                    near = any(abs(li - pi) <= 120 for li in loc_positions)
                    if not near:
                        continue
                    key = ("event_copresence_omission", evt_id, pid)
                    if key in seen:
                        continue
                    seen.add(key)
                    issues.append(AuditViolation(
                        violation_type="event_copresence_omission",
                        severity="major",
                        description=(
                            f"Prose stages `{pid}` ({pname}) at the scene of "
                            f"event `{evt_id}` (`{loc_id}` / {loc_name}), but "
                            f"that character's reconstructed location at "
                            f"fabula_time differs. Phantom witness."
                        ),
                        evidence_quote=prose[max(0, pi - 40): pi + len(pname) + 40],
                        feedback=(
                            f"Remove `{pid}` from the scene at `{loc_id}` or "
                            f"justify their relocation by an explicit "
                            f"movement event before this beat."
                        ),
                    ))
                    break

    return issues


def _position_mismatch_violations(
    prose: str,
    brief: CreativeBrief,
    world_state: Optional[WorldStateV1],
    introduced: Optional[IntroducedElements] = None,
) -> List[AuditViolation]:
    """Deterministic object / entity position-mismatch auditor pass.

    Materialises the long-declared but never-emitted violation kinds
    ``object_position_mismatch`` and ``entity_position_mismatch``
    (see the ``violation_type`` enum at the top of this module). For
    every :class:`NarrativeObject` with a non-trivial state_timeline,
    and every :class:`Entity` with state_timeline movement snapshots,
    the function:

      1. Derives the scene's ``fabula_anchor`` from
         ``brief.scene_context.recent_memory`` (max ``fabula_time``).
      2. Reconstructs the canonical ``location_id`` at that anchor
         via :func:`reconstruct_object_at` / :func:`reconstruct_entity_at`.
      3. If prose verbatim names the object / entity AND a *different*
         canonical location name within ~120 chars, emits a HARD
         violation. The LLM auditor remains responsible for paraphrase
         / pronoun cases.

    Conservative on purpose \u2014 we only fire on verbatim, close-range
    co-mentions to keep false-positive rate low, mirroring the
    existing co-presence checker.
    """
    if world_state is None or not prose:
        return []
    locations = world_state.locations or {}
    if not locations:
        return []
    entities = world_state.entities or {}
    objects = (
        getattr(world_state, "objects", None)
        or getattr(world_state, "narrative_objects", None)
        or {}
    )

    sc = getattr(brief, "scene_context", {}) or {}
    fabula_anchor: Optional[int] = None
    if isinstance(sc, dict):
        # Prefer the explicit fabula_anchor stamped into scene_context by
        # DirectiveAssembler.assemble (computed via _syuzhet_to_fabula_cutoff
        # and already used by the constraint builders). This avoids anchor
        # drift when recent_memory contains flashback or prolepsis events
        # whose max(fabula_time) sits ahead of or behind the scene's actual
        # temporal anchor on the syuzhet axis.
        if isinstance(sc.get("fabula_anchor"), int):
            fabula_anchor = sc["fabula_anchor"]
        else:
            # Fall back to max(recent_memory.fabula_time) when the brief
            # was assembled without a syuzhet anchor (e.g. observation
            # queries with no explicit position) — in that case all events
            # are visible and the max fabula time is the correct anchor.
            recent = sc.get("recent_memory") or []
            if isinstance(recent, list):
                fts = [
                    e.get("fabula_time") for e in recent
                    if isinstance(e, dict) and isinstance(e.get("fabula_time"), int)
                ]
                if fts:
                    fabula_anchor = max(fts)
    if fabula_anchor is None:
        return []

    prose_lower = prose.lower()

    def _name_positions(name: str) -> list[int]:
        if not name:
            return []
        n = name.strip().lower()
        if len(n) < 3:
            return []
        out: list[int] = []
        start = 0
        while True:
            i = prose_lower.find(n, start)
            if i < 0:
                break
            out.append(i)
            start = i + len(n)
        return out

    # Precompute location name → positions map.
    loc_positions: Dict[str, list[int]] = {}
    loc_names: Dict[str, str] = {}
    for lid, loc in (locations.items() if isinstance(locations, dict) else []):
        lname = getattr(loc, "name", lid) or lid
        if len(lname) < 3:
            continue
        loc_names[lid] = lname
        positions = _name_positions(lname)
        if positions:
            loc_positions[lid] = positions

    # Augment with introduced_elements locations so that new locations
    # declared in this draft (e.g. LOC_AIRCRAFT_CABIN) participate in
    # the proximity check. Without this, a reconstructed_loc pointing
    # at an introduced id silently skips the check (loc_names miss),
    # and a wrong introduced location near an entity name is also
    # invisible to the checker.
    if introduced is not None:
        for spec in introduced.locations:
            lid = spec.id
            lname = spec.name or lid
            if len(lname) < 3 or lid in loc_names:
                continue
            loc_names[lid] = lname
            positions = _name_positions(lname)
            if positions:
                loc_positions[lid] = positions

    if not loc_positions:
        return []

    issues: List[AuditViolation] = []
    seen: set[tuple[str, str, str]] = set()

    def _check_subject(
        subject_id: str,
        subject_name: str,
        reconstructed_loc: Optional[str],
        kind: Literal["object_position_mismatch", "entity_position_mismatch"],
    ) -> None:
        if not subject_name or not reconstructed_loc:
            return
        if reconstructed_loc not in loc_names:
            return
        s_positions = _name_positions(subject_name)
        if not s_positions:
            return
        canonical_loc_name = loc_names[reconstructed_loc]
        for si in s_positions:
            for other_lid, other_positions in loc_positions.items():
                if other_lid == reconstructed_loc:
                    continue
                other_name = loc_names[other_lid]
                for oi in other_positions:
                    if abs(oi - si) > 120:
                        continue
                    key = (kind, subject_id, other_lid)
                    if key in seen:
                        return
                    seen.add(key)
                    issues.append(AuditViolation(
                        violation_type=kind,
                        severity="major",
                        description=(
                            f"Prose places `{subject_id}` ({subject_name}) "
                            f"at `{other_lid}` ({other_name}), but the "
                            f"reconstructed location at fabula_time="
                            f"{fabula_anchor} is `{reconstructed_loc}` "
                            f"({canonical_loc_name})."
                        ),
                        evidence_quote=prose[max(0, si - 40): si + len(subject_name) + 40],
                        feedback=(
                            f"Either re-locate `{subject_id}` to "
                            f"`{reconstructed_loc}` ({canonical_loc_name}) "
                            f"for this beat, or stage an explicit movement "
                            f"event before the mention."
                        ),
                    ))
                    return

    # Build the set of entity IDs named in TRUE propositions of
    # movement-asserting kinds (event_occurs, outcome) at fabula_anchor.
    # When a proposition like PROP_ARCHIE_WANDA_FLY_AWAY commits true,
    # the referenced entities may be physically somewhere their
    # state_timeline has not yet caught up to. Flagging those entities
    # as position-mismatched would be a false positive.
    # Restriction to event_occurs/outcome: trait_holds, relation_holds,
    # identity_is propositions don't assert location and must NOT suppress
    # the check (e.g. PROP_ARCHIE_IS_BARRISTER should not exempt Archie
    # from position checks).
    _MOVEMENT_PROP_KINDS = {"event_occurs", "outcome"}
    _prop_referenced_entity_ids: set[str] = set()
    if world_state is not None:
        from shadow_loom.models import reconstruct_proposition_at
        _props = world_state.propositions
        _prop_iter = (
            _props.values() if isinstance(_props, dict)
            else (_props or [])
        )
        for prop in _prop_iter:
            if getattr(prop, "kind", "") not in _MOVEMENT_PROP_KINDS:
                continue
            try:
                pstate = reconstruct_proposition_at(prop, fabula_anchor)
                if pstate.get("truth_at") is True:
                    for rid in (getattr(prop, "referent_ids", None) or []):
                        if isinstance(rid, str) and rid.startswith("ENT_"):
                            _prop_referenced_entity_ids.add(rid)
            except Exception:
                continue

    # Objects
    for oid, obj in (objects.items() if isinstance(objects, dict) else []):
        oname = getattr(obj, "name", None) or oid
        timeline = getattr(obj, "state_timeline", None) or []
        # Skip wholly-static objects with no timeline AND no initial
        # location \u2014 nothing to mismatch against.
        if not timeline and not getattr(obj, "location_id", None):
            continue
        try:
            state = reconstruct_object_at(obj, fabula_anchor)
        except Exception:
            continue
        _check_subject(
            oid, oname, state.get("location_id"),
            "object_position_mismatch",
        )

    # Entities
    for eid, ent in (entities.items() if isinstance(entities, dict) else []):
        ename = getattr(ent, "name", None) or eid
        # Skip entities whose position is asserted by a true proposition —
        # the reconstructed state_timeline location is stale in those cases.
        if eid in _prop_referenced_entity_ids:
            continue
        timeline = getattr(ent, "state_timeline", None) or []
        if not timeline and not getattr(ent, "location_id", None):
            continue
        try:
            state = reconstruct_entity_at(ent, fabula_anchor)
        except Exception:
            continue
        _check_subject(
            eid, ename, state.get("location_id"),
            "entity_position_mismatch",
        )

    return issues


def _world_trait_timeline_disorder_violations(
    world_state: Optional[WorldStateV1],
) -> List[AuditViolation]:
    """Deterministic monotonicity check on ``GlobalTrait.state_timeline``.

    Round-12 audit invariant. Any snapshot list whose consecutive
    ``fabula_time`` values descend corrupts ``reconstruct_*_at``
    replay and silently destabilises every renderer surface that
    relies on the reconstructed magnitude (atmosphere prose, ambient
    force grounding). Fires a HARD violation per offending trait.
    Independent of prose \u2014 this is a pure world-state invariant.
    """
    if world_state is None:
        return []
    out: List[AuditViolation] = []
    traits = getattr(world_state, "world_traits", None) or {}
    iterator = traits.values() if isinstance(traits, dict) else (traits or [])
    for wt in iterator:
        timeline = getattr(wt, "state_timeline", None) or []
        if len(timeline) < 2:
            continue
        prev = None
        for snap in timeline:
            ft = getattr(snap, "fabula_time", None)
            if ft is None:
                continue
            if prev is not None and ft < prev:
                wt_id = getattr(wt, "id", "") or getattr(wt, "world_trait_id", "?")
                out.append(AuditViolation(
                    violation_type="world_trait_timeline_disorder",
                    severity="critical",
                    description=(
                        f"World trait {wt_id} state_timeline is not "
                        f"monotonically ordered by fabula_time "
                        f"(saw {prev} then {ft})."
                    ),
                    evidence_quote="",
                    feedback=(
                        f"Engine invariant breach: sort or rebuild "
                        f"``{wt_id}.state_timeline`` so consecutive "
                        f"snapshots have non-decreasing fabula_time."
                    ),
                ))
                break
            prev = ft
    return out


def run_audit(
    prose: str,
    brief: CreativeBrief,
    config: AuditorConfig | None = None,
    prior_feedback: Optional[List[str]] = None,
    causal_feedback: Optional[CausalPhysicsFeedback] = None,
    *,
    world_state: Optional[WorldStateV1] = None,
    affective_feedback: Optional[AffectiveStateFeedback] = None,
    introduced_elements: Optional[IntroducedElements] = None,
) -> AuditResult:
    """Execute a single audit pass (Step 11).

    Parameters
    ----------
    prose : str
        The rendered prose to audit.
    brief : CreativeBrief
        The mathematical constraints from directive assembly.
    config : AuditorConfig or None
        Auditor configuration.
    prior_feedback : list[str] or None
        Feedback strings from previous iterations.

    Returns
    -------
    AuditResult
        The structured audit verdict.
    """
    config = config or AuditorConfig()

    categories = resolve_audit_categories(brief.target_effect)

    audit_prompt = assemble_audit_prompt(
        prose, brief, categories, prior_feedback, causal_feedback,
        world_state=world_state,
        affective_feedback=affective_feedback,
        introduced_elements=introduced_elements,
    )

    logger.info(
        "[Auditor] Running audit: effect=%s categories=%s prompt_len=%d",
        brief.target_effect, categories, len(audit_prompt),
    )
    # Emit the full audit prompt at DEBUG for log-based analysis,
    # mirroring the renderer's prompt logging so generator and
    # auditor instructions can be compared side-by-side. INFO would
    # leak prose under review and rich world-state slices to shared
    # log aggregators (round-3 audit).
    logger.debug(
        "[Auditor] Audit prompt (%d chars):\n"
        "========== BEGIN AUDIT PROMPT ==========\n%s\n"
        "========== END AUDIT PROMPT ==========",
        len(audit_prompt),
        audit_prompt,
    )
    logger.info(
        "[Auditor] Audit prompt prepared (%d chars)",
        len(audit_prompt),
    )

    agent = _build_auditor_agent(config)
    deps = _AuditorDeps(audit_prompt=audit_prompt)

    model_settings: Dict[str, Any] = {
        "max_tokens": config.max_tokens_audit,
    }
    if config.auditor_temperature != 0.2:
        model_settings["temperature"] = config.auditor_temperature

    try:
        result = agent.run_sync(
            f"Audit the prose for target effect: {brief.target_effect}.",
            deps=deps,
            model_settings=model_settings if model_settings else None,
        )
    except Exception as exc:
        # Fail SAFE, not OPEN. The previous behaviour returned
        # ``passed=True`` which let the feedback loop converge on a
        # malformed audit and silently merge possibly-bad prose. We
        # now mark the audit as failed so the loop can decide whether
        # to retry, fall through to its own safety net, or surface
        # the error to the caller.
        logger.error(
            "[Auditor] LLM call failed: %s. Returning failed_open audit "
            "(passed=False) so the loop does not silently converge.",
            exc,
        )
        return AuditResult(
            passed=False,
            violations=[
                AuditViolation(
                    violation_type="miracle_step",
                    severity="major",
                    description=f"Auditor LLM call raised: {exc!r}",
                    feedback=(
                        "The auditor could not be invoked. Treat this scene "
                        "as not-yet-validated."
                    ),
                ),
            ],
            audit_summary=f"Audit failed-open due to LLM error: {exc}",
            failed_open=True,
        )

    audit = result.output
    log_agent_output(logger, "Auditor", audit)

    # Snapshot the LLM's own verdict and violation list before any
    # deterministic engine passes mutate audit.passed / audit.violations.
    # The feedback loop reads these fields to correctly compute llm_passed
    # without being blocked by engine false-positives (e.g. the 120-char
    # proximity position-mismatch heuristic), and to restrict the minor-
    # bypass check to the LLM's own violations only (so an engine-injected
    # major violation cannot prevent the bypass from firing when the LLM
    # itself only flagged a minor style_mismatch).
    audit.llm_raw_passed = audit.passed
    audit.llm_raw_violations = list(audit.violations)

    # Deterministic withheld-utterance leak check. Runs even when the
    # LLM call succeeded — a verbatim leak is a hard failure that we
    # don't want to entrust to the auditor's free-form judgement. The
    # syuzhet anchor is read from brief.scene_context if present; we
    # fall back to the max syuzhet_index of any recent_memory event.
    syuzhet_anchor: Optional[int] = None
    sc = getattr(brief, "scene_context", {}) or {}
    if isinstance(sc, dict):
        if isinstance(sc.get("syuzhet_anchor"), int):
            syuzhet_anchor = sc["syuzhet_anchor"]
        else:
            recent = sc.get("recent_memory") or []
            if isinstance(recent, list):
                indices = [
                    e.get("syuzhet_index") for e in recent
                    if isinstance(e, dict) and isinstance(e.get("syuzhet_index"), int)
                ]
                if indices:
                    syuzhet_anchor = max(indices)
    leak_violations = _withheld_utterance_leak_violations(
        prose, world_state, syuzhet_anchor,
    )
    if leak_violations:
        audit.violations = list(audit.violations) + leak_violations
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(leak_violations)} deterministic "
            f"withheld-utterance leak(s)]"
        ).strip()

    # Deterministic undeclared-element check. Runs against the union of
    # ``world_state`` and the renderer's ``introduced_elements``
    # declaration. The LLM auditor remains responsible for paraphrase
    # / single-token cases; this catches the high-precision cases
    # (raw ids in prose, multi-word proper-noun bigrams) so the
    # refinement loop can never converge on prose that names something
    # the merge would have to drop.
    undeclared = _undeclared_element_violations(
        prose, world_state, introduced_elements,
    )
    if undeclared:
        audit.violations = list(audit.violations) + undeclared
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(undeclared)} undeclared "
            f"element(s)]"
        ).strip()

    # Deterministic reuse-first / justification check on every
    # per-cycle introduction. Flags empty or boilerplate
    # ``justification`` fields and display-name collisions with
    # existing world-state elements. The renderer (and the user via
    # ``query.introduce``) are free to mint new top-level elements,
    # but every introduction must concretely justify why no existing
    # element fit \u2014 see ``IntroducedElements`` and
    # ``prompts/auditor.md`` Category 4c.
    unjustified = _unjustified_introduction_violations(
        introduced_elements, world_state,
    )
    if unjustified:
        audit.violations = list(audit.violations) + unjustified
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(unjustified)} unjustified "
            f"introduction(s)]"
        ).strip()

    # Deterministic ctf-calculus EXCLUSION leak check. Pulls evidence
    # ids from the HARD ConstraintBlocks emitted by
    # ``_build_exclusion_constraints`` /
    # ``_build_cascade_exclusion_constraints`` and flags pruned
    # utterance content, severed channel references, and blocked
    # (node, trait) co-mentions. Mirrors the pink-elephant boundary
    # the renderer was forbidden from crossing so the auditor cannot
    # converge on prose that contradicts the engine's exclusion
    # ledger.
    cascade_leaks = _cascade_exclusion_leak_violations(
        prose, brief, world_state,
    )
    if cascade_leaks:
        audit.violations = list(audit.violations) + cascade_leaks
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(cascade_leaks)} ctf-calculus "
            f"exclusion leak(s)]"
        ).strip()

    # Defence-in-depth check for do-prevention surgeries that span
    # non-utterance events. The cascade-leak pass above covers
    # utterance content + channels; this one catches outcome / choice
    # events that the renderer staged despite being severed by the
    # do-surgery (or by the Step B.6 chain_reaction descendant
    # closure in :mod:`shadow_loom.causal_physics`).
    reenact_issues = _prevented_event_reenacted_violations(
        prose, brief, world_state,
    )
    if reenact_issues:
        audit.violations = list(audit.violations) + reenact_issues
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(reenact_issues)} prevented "
            f"event re-enacted]"
        ).strip()

    # Deterministic event co-presence / spatial-anchor check (PR 4 of
    # EventNode.at_location_id). Cross-references the spatial
    # ConstraintBlocks emitted by ``build_event_copresence_constraints``
    # against the prose for verbatim violations of MUST_BE_PRESENT
    # and MUST_NOT_BE_PRESENT ledgers.
    copresence_issues = _event_copresence_violations(
        prose, brief, world_state, introduced_elements,
    )
    if copresence_issues:
        audit.violations = list(audit.violations) + copresence_issues
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(copresence_issues)} event "
            f"co-presence violation(s)]"
        ).strip()

    # Deterministic object / entity position-mismatch check. Materialises
    # the long-declared ``object_position_mismatch`` and
    # ``entity_position_mismatch`` violation kinds against the
    # reconstructed location at the scene's fabula anchor.
    position_issues = _position_mismatch_violations(
        prose, brief, world_state, introduced_elements,
    )
    if position_issues:
        audit.violations = list(audit.violations) + position_issues
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(position_issues)} position "
            f"mismatch(es)]"
        ).strip()

    # Pure world-state invariant: world_trait state_timeline ordering.
    wt_disorder = _world_trait_timeline_disorder_violations(world_state)
    if wt_disorder:
        audit.violations = list(audit.violations) + wt_disorder
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(wt_disorder)} world-trait "
            f"timeline disorder]"
        ).strip()

    logger.info(
        "[Auditor] Audit complete: passed=%s violations=%d summary=%s",
        audit.passed, len(audit.violations), audit.audit_summary,
    )
    return audit


# =====================================================================
# NarrativeOrderObject Assembly
# =====================================================================

def _finalize_narrative_order(
    prose: str,
    brief: CreativeBrief,
    causal_feedback: CausalPhysicsFeedback,
    affective_feedback: AffectiveStateFeedback,
    config: AuditorConfig,
) -> NarrativeOrderObject:
    """Combine engine metrics with LLM evaluation into a NarrativeOrderObject.

    Runs the evaluation agent once for StoryQualitySynthesis, then applies
    threshold checks for overall_pass.
    """
    # LLM literary critique (single call)
    try:
        synthesis = run_evaluation(
            prose=prose,
            brief=brief,
            config=config,
            causal_feedback=causal_feedback,
            affective_feedback=affective_feedback,
        )
    except Exception as exc:
        logger.error(
            "[NarrativeOrder] Evaluation failed: %s. Using empty synthesis.", exc,
        )
        synthesis = StoryQualitySynthesis(
            coherence_and_consistency_review=(
                f"Evaluation skipped due to error: {exc}"
            ),
        )

    order = NarrativeOrderObject(
        causal_feedback=causal_feedback,
        affective_feedback=affective_feedback,
        quality_synthesis=synthesis,
        overall_pass=False,  # computed below
    )
    order.overall_pass = compute_overall_pass(order, config)
    return order


# =====================================================================
# Feedback Loop (Steps 11 + 12)
# =====================================================================

def run_feedback_loop(
    initial_scene: GeneratedScene,
    brief: CreativeBrief,
    sandbox: Optional[nx.MultiDiGraph] = None,
    world_state: Optional[WorldStateV1] = None,
    auditor_config: AuditorConfig | None = None,
    generation_config: GenerationConfig | None = None,
    query_type: str = "directive",
    physics_state: Optional[Dict[str, Any]] = None,
    physics_result: Optional[CausalPhysicsResult] = None,
    assembler: Optional[DirectiveAssembler] = None,
) -> FeedbackLoopResult:
    """Run the full audit → refinement loop (Steps 11–12).

    The loop:
      1. Deep-copy and version the sandbox graph.
      2. Audit the prose against the brief + graph.
      3. If the audit passes → return the current prose.
      4. If it fails → inject feedback into the rendering prompt
         and re-generate.  Repeat up to ``max_iterations``.

    Per-cycle engine metrics are computed and attached as
    ``ChangeImpactMetrics`` — scoped to the delta of that single
    directive request.  Full-story evaluation (``NarrativeOrderObject``)
    is **not** produced here; that is the Evaluator's responsibility
    (triggered via ``EvaluationQuery`` in the pipeline).

    Parameters
    ----------
    initial_scene : GeneratedScene
        The first prose rendering from Step 10.
    brief : CreativeBrief
        The mathematical constraints the prose must satisfy.
    sandbox : nx.MultiDiGraph or None
        The AMWN shadow graph.  Deep-copied and versioned so the
        original is never mutated.
    world_state : WorldStateV1 or None
        The canonical world state (for physics audit reference).
    auditor_config : AuditorConfig or None
        Configuration for the auditor LLM.
    generation_config : GenerationConfig or None
        Configuration for the generation LLM (re-renders).
    query_type : str
        The originating query type.
    physics_state : dict or None
        Raw physics state for generation context.
    physics_result : CausalPhysicsResult or None
        Output of the causal physics engine (for engine-computed metrics).
    assembler : DirectiveAssembler or None
        A configured assembler (for affective scoring).

    Returns
    -------
    FeedbackLoopResult
        Contains the final scene, convergence status, iteration count,
        the full history of audit snapshots, and per-cycle ChangeImpactMetrics.
    """
    auditor_config = auditor_config or AuditorConfig()
    generation_config = generation_config or GenerationConfig()

    # Override generation config with auditor config's generation model
    generation_config = generation_config.model_copy(
        update={"model": auditor_config.generation_model}
    )

    # D (round-7 audit 2026-05-26): isolate the brief from caller
    # mutation. ``_inject_miracle_step_mechanisms`` appends
    # InterventionMechanism entries to ``brief.intervention_mechanisms``
    # in place. Without this deep-copy, a caller that reuses the same
    # ``CreativeBrief`` for two pipeline calls (e.g. a UI retry
    # button) sees the second call start with the first call's
    # accumulated miracle-step mechanisms still attached — silently
    # changing the brief between runs.
    brief = brief.model_copy(deep=True)

    # Version the sandbox graph
    versioned: Optional[VersionedGraph] = None
    if sandbox is not None:
        versioned = VersionedGraph(sandbox)

    # Build the base rendering prompt (without feedback)
    base_rendering_prompt = assemble_rendering_prompt(
        brief, query_type, physics_state,
    )

    current_scene = initial_scene
    history: List[AuditCycleSnapshot] = []
    accumulated_feedback: List[str] = []
    # Violations from prior iterations, kept as structured objects so we
    # can re-surface them in the refinement prompt as
    # "non-regression constraints" \u2014 things the previous draft
    # already fixed and must not regress on. Without this, the rewriter
    # only sees the *latest* violation list and reliably ping-pongs
    # between competing constraints (e.g.\u00a0fixing
    # ``style_mismatch`` by stripping voice, then regressing on
    # ``meta_narration``, then back).
    accumulated_violations: List[AuditViolation] = []
    consecutive_failed_open = 0
    correction_error: Optional[str] = None

    # Anti-regression retry budget (round-7 audit 2026-05-26). When the
    # rewriter regresses (the severity-weighted score rises and a new
    # violation type appears), the old policy was to roll back to the
    # prior draft and immediately ``break``. That terminated the loop
    # on a single rewrite mistake even when iterations remained on
    # the ``max_iterations`` budget. We now grant
    # ``auditor_config.regression_retry_budget`` retries (default 1):
    # roll back to the prior draft, surface a REGRESSION ALERT in the
    # next refinement prompt naming the violation types just
    # introduced, and let the loop continue. Exhausting the budget
    # falls back to the original break-and-return path.
    regression_retries_remaining = auditor_config.regression_retry_budget
    pending_regression_warning: Optional[str] = None

    # Snapshot the rendering mode the brief asked for. The refinement
    # agent is forbidden from mutating it (mode-flip silently degrades
    # the audit because the auditor evaluates against the *brief's*
    # mode while the prose was rewritten under a different one). If
    # the agent flips the mode we treat its output as a generation
    # error, keep the previous scene, and exit the loop.
    expected_rendering_mode: Optional[str] = None
    if getattr(brief, "rendering", None) is not None:
        expected_rendering_mode = getattr(
            brief.rendering, "rendering_mode", None,
        )

    # Iteration-0 metadata coercion. The original (pre-refinement)
    # render is also free to mislabel ``rendering_mode`` on its
    # structured output: the LLM occasionally emits ``"observation"``
    # under a ``narrative_tension`` brief, etc. Previously the
    # refinement branch contained the only coercion site, which meant
    # that when iteration 1 already passed the audit, the returned
    # ``GeneratedScene.rendering_mode`` could differ from the brief's
    # mode \u2014 silently breaking the UI mode-filter, the next merge's
    # branch policy, and any downstream consumer keyed off the mode.
    # Coerce once here, with the same warning-then-continue semantics
    # the refinement branch uses, BEFORE the first audit runs.
    if (
        expected_rendering_mode is not None
        and current_scene.rendering_mode
        and current_scene.rendering_mode != expected_rendering_mode
    ):
        logger.warning(
            "[FeedbackLoop] Initial render emitted rendering_mode "
            "(%r != brief %r); coercing metadata to the brief's mode "
            "before the first audit. Prose is unchanged \u2014 only the "
            "structured label is normalised.",
            current_scene.rendering_mode, expected_rendering_mode,
        )
        current_scene = current_scene.model_copy(update={
            "rendering_mode": expected_rendering_mode,
        })

    # Track the prior iteration's violation count and scene so we can
    # roll back when the refinement agent INTRODUCES more violations
    # than it closes. Without this guard the loop happily accepts a
    # strictly-worse rewrite (e.g.\u00a0closes 1 minor density drift
    # while opening a major meta-narration leak) and the user sees the
    # regressed prose as the final output.
    prior_violation_count: Optional[int] = None
    prior_violation_score: Optional[float] = None
    prior_violation_keys: set[tuple[str, str]] = set()
    prior_scene: Optional[GeneratedScene] = None
    prior_audit: Optional[AuditResult] = None
    prior_cycle_impact: Optional[ChangeImpactMetrics] = None
    prior_graph_version: int = 0
    prior_graph_data: Dict[str, Any] = {}

    # Initial-scene generation failure short-circuit. ``render_scene``
    # returns a placeholder GeneratedScene with ``generation_error``
    # set when the LLM call raises. There is no useful prose to audit
    # or refine against \u2014 every retry would just be the auditor
    # complaining about the placeholder text \u2014 and downstream
    # re-extraction/merge must not absorb the placeholder into the
    # canonical world state. Exit immediately as non-converged with a
    # diagnostic, mirroring the refinement-failure exit below.
    if initial_scene.generation_error:
        logger.error(
            "[FeedbackLoop] Initial render failed (generation_error=%s) "
            "\u2014 skipping audit/refinement loop and returning the "
            "placeholder scene as non-converged.",
            initial_scene.generation_error,
        )
        return FeedbackLoopResult(
            final_scene=initial_scene,
            converged=False,
            iterations=0,
            history=[],
            final_graph_version=(versioned.version if versioned else 0),
            change_impact=None,
            correction_error=(
                f"Initial render produced fallback scene: "
                f"{initial_scene.generation_error}"
            ),
            engine_thresholds_passed=None,
            engine_threshold_failures=[],
        )

    # Up-front structural baseline. ``physics_result``, ``world_state``
    # and ``brief`` (the only inputs ``compute_*_feedback`` reads) are
    # invariant across iterations of this loop unless the brief is
    # mutated below by ``_inject_miracle_step_mechanisms``. So compute
    # the engine scorecard ONCE here, log it, and skip the per-iteration
    # recomputation unless the brief actually changes. This both saves
    # work and \u2014 more importantly \u2014 prevents an immutable
    # ``engine_passed=False`` from permanently vetoing convergence on
    # plots whose structural deficits the prose-rewriter has no power
    # to repair.
    baseline_causal = compute_causal_feedback(
        physics_result, brief, world_state,
    )
    baseline_affective = compute_affective_feedback(brief, assembler)
    baseline_impact = ChangeImpactMetrics(
        causal_feedback=baseline_causal,
        affective_feedback=baseline_affective,
    )
    baseline_engine_passed, baseline_engine_failures = (
        _engine_thresholds_check(baseline_impact, auditor_config)
    )
    if baseline_engine_passed is False:
        logger.warning(
            "[FeedbackLoop] Baseline engine thresholds FAIL before any "
            "rewrite (%d failures): %s. Convergence will not gate on "
            "the engine score because prose rewrites cannot move these "
            "structural metrics; only the LLM auditor's prose-level "
            "verdict will be required.",
            len(baseline_engine_failures),
            "; ".join(baseline_engine_failures),
        )
    cycle_causal = baseline_causal
    cycle_affective = baseline_affective
    cycle_impact = baseline_impact
    engine_passed = baseline_engine_passed
    engine_failures = baseline_engine_failures
    engine_invariant = True  # flips False once the brief is mutated

    for iteration in range(auditor_config.max_iterations):
        logger.info(
            "[FeedbackLoop] === Iteration %d/%d ===",
            iteration + 1, auditor_config.max_iterations,
        )

        # Recompute engine scorecard ONLY when the brief was mutated by
        # the previous iteration's ``_inject_miracle_step_mechanisms``
        # call. Otherwise it is bit-identical to the baseline and the
        # extra work would just thrash the logs.
        if not engine_invariant:
            cycle_causal = compute_causal_feedback(
                physics_result, brief, world_state,
            )
            cycle_affective = compute_affective_feedback(brief, assembler)
            cycle_impact = ChangeImpactMetrics(
                causal_feedback=cycle_causal,
                affective_feedback=cycle_affective,
            )
            engine_passed, engine_failures = _engine_thresholds_check(
                cycle_impact, auditor_config,
            )
            engine_invariant = True

        # --- Step 11: Audit ---
        # The auditor LLM still sees the engine's deterministic ledger
        # (miracle steps, cyclic clusters, ctf-calculus prunings) so it
        # can evaluate prose against ground truth.
        audit = run_audit(
            prose=current_scene.prose,
            brief=brief,
            config=auditor_config,
            prior_feedback=accumulated_feedback if iteration > 0 else None,
            causal_feedback=cycle_causal,
            world_state=world_state,
            affective_feedback=cycle_affective,
            introduced_elements=getattr(
                current_scene, "introduced_elements", None,
            ),
        )

        # R19-H10: short-circuit on world-state-invariant violations.
        # ``world_trait_timeline_disorder`` (and similar pure
        # world-state invariants) cannot be fixed by re-rendering
        # prose — the violation lives in the upstream world model.
        # Continuing the rewrite loop guarantees the same violation
        # re-fires every iteration, burning the entire iteration
        # budget on an unfixable problem. Bail out immediately,
        # routing the violation through ``engine_threshold_failures``
        # so callers see it as an engine-state breach, not a prose
        # failure.
        _UNREWRITEABLE_VIOLATION_TYPES = {
            "world_trait_timeline_disorder",
        }
        unrewriteable = [
            v for v in audit.violations
            if getattr(v, "violation_type", "") in _UNREWRITEABLE_VIOLATION_TYPES
        ]
        if unrewriteable:
            logger.error(
                "[FeedbackLoop] Detected %d world-state-invariant "
                "violation(s) at iteration %d that prose rewriting "
                "cannot fix (%s). Short-circuiting loop; the world "
                "model must be repaired upstream.",
                len(unrewriteable), iteration + 1,
                ", ".join(sorted({v.violation_type for v in unrewriteable})),
            )
            invariant_failures = [
                f"world_state_invariant: {v.violation_type}: {v.description}"
                for v in unrewriteable
            ]
            return FeedbackLoopResult(
                final_scene=current_scene,
                converged=False,
                iterations=iteration + 1,
                history=history,
                final_graph_version=(versioned.version if versioned else 0),
                change_impact=cycle_impact,
                engine_thresholds_passed=False,
                engine_threshold_failures=(
                    list(engine_failures or []) + invariant_failures
                ),
            )

        # run_audit() hard engine violation guard. ``llm_raw_passed``
        # correctly prevents 120-char proximity heuristics
        # (position-mismatch, co-presence, cascade-exclusion) from
        # blocking convergence when the LLM cleared the prose. But it
        # must NOT suppress convergence blocking for the *verbatim*
        # high-precision checks that the LLM structurally cannot
        # evaluate (it doesn't see the full world-state ledger):
        #   * withheld_utterance_leak  — secret dialogue in prose
        #   * pruned_utterance_leak    — do-surgery-severed dialogue
        #   * disabled_channel_leak    — do-surgery-severed channel ref
        #   * prevented_event_reenacted — do-surgery-pruned event staged
        # These are near-zero false-positive rate; when they fire the
        # prose is directly contradicting the engine's surgery ledger.
        # Detect them by comparing the engine-injected delta
        # (audit.violations[llm_raw_violations:]) against the type set.
        _HARD_ENGINE_VIOLATION_TYPES = {
            "withheld_utterance_leak",
            "pruned_utterance_leak",
            "disabled_channel_leak",
            "prevented_event_reenacted",
        }
        _llm_viol_count = len(audit.llm_raw_violations)
        _hard_engine_added = (
            len(audit.violations) > _llm_viol_count
            and any(
                getattr(v, "violation_type", "") in _HARD_ENGINE_VIOLATION_TYPES
                for v in audit.violations[_llm_viol_count:]
            )
        )

        # C (round-7 audit 2026-05-26): deterministic POV-lock and
        # meta-narration checks run alongside the LLM auditor and
        # merge their findings into the violation list. The LLM
        # auditor misses these classes under rewrite pressure (see
        # round-7 pipeline log: iter-2 dropped POV + emitted
        # "established record" meta-narration; the LLM auditor only
        # caught one of the two). The regex check is cheap and
        # deterministic, so it runs unconditionally when the brief
        # carries a POV lock or whenever the prose may contain
        # meta-narration triggers. The check is gated behind
        # ``enable_deterministic_prose_checks`` so callers can opt
        # out for genuine omniscient-narration briefs.
        #
        # Track whether this block adds NEW violations so the
        # convergence check can gate on det_findings separately from
        # the LLM's own verdict (``llm_raw_passed``).
        _violations_pre_det = len(audit.violations)
        _det_added_violations = False
        if auditor_config.enable_deterministic_prose_checks and not audit.failed_open:
            brief_pov = None
            pov_aliases: List[str] = []
            additional_locks: List[str] = []
            pov_policy = "single"
            if getattr(brief, "rendering", None) is not None:
                # The brief contract field is ``pov_lock``; the scene's
                # mirror field is ``pov_entity``. See the POV-lock
                # role documentation on ``RenderingDirective.pov_lock``.
                brief_pov = getattr(brief.rendering, "pov_lock", None)
                additional_locks = list(
                    getattr(brief.rendering, "additional_pov_locks", None) or []
                )
                pov_policy = getattr(brief.rendering, "pov_policy", None) or "single"
                # Use roster names as POV aliases when available. For
                # rotating policy, gather aliases for the primary lock
                # AND every additional lock so the deterministic
                # cognitive-verb check doesn't over-flag legitimate
                # interiority from any licensed POV.
                licensed_ids = {brief_pov, *additional_locks} - {None}
                roster = getattr(brief, "pov_roster", None) or []
                for entry in roster:
                    entity_id = getattr(entry, "entity_id", None)
                    if entity_id in licensed_ids:
                        names = getattr(entry, "alias_set", None) or []
                        pov_aliases.extend(str(n) for n in names if n)
                        canonical = getattr(entry, "canonical_name", None)
                        if canonical:
                            pov_aliases.append(str(canonical))
                # Round-9 A3+B4: when the brief carries no explicit
                # ``pov_roster`` (the common case — the brief contract
                # has no canonical roster field), fall back to deriving
                # aliases directly from ``world_state`` so the rotating
                # POV check doesn't over-flag legitimate interiority
                # from licensed characters. The deterministic checker
                # matches subject *names* in prose, so the entity ID
                # alone (e.g. ``ENT_MACBETH``) is useless; we need the
                # canonical name plus any name tokens.
                if not pov_aliases and world_state is not None:
                    entities = getattr(world_state, "entities", None) or {}
                    for entity_id in licensed_ids:
                        ent = entities.get(entity_id)
                        if ent is None:
                            continue
                        canonical = getattr(ent, "name", None)
                        if canonical:
                            pov_aliases.append(str(canonical))
                            # Split on whitespace/punctuation so
                            # "Macbeth (Thane of Glamis)" yields
                            # "Macbeth" as a single-token alias too.
                            for token in re.split(r"[\s,()\[\]/]+", str(canonical)):
                                token = token.strip()
                                if token and token.isalpha() and len(token) > 2:
                                    pov_aliases.append(token)
            # P0 #2a (round-7 deeper audit 2026-05-27): ensemble
            # licenses omniscient narration; skip the deterministic
            # POV check entirely. The meta-narration check still
            # runs because ensemble is not a licence for breaking
            # the fourth wall.
            if pov_policy == "ensemble":
                det_findings = deterministic_prose_findings(
                    current_scene.prose,
                    pov_entity=None,
                    pov_entity_aliases=None,
                    pov_breach_threshold=auditor_config.pov_breach_threshold,
                )
            else:
                # For rotating policy, treat additional locks as
                # additional licensed POVs by feeding their aliases
                # into the same allowed-subject set.
                det_findings = deterministic_prose_findings(
                    current_scene.prose,
                    pov_entity=brief_pov,
                    pov_entity_aliases=pov_aliases or None,
                    pov_breach_threshold=auditor_config.pov_breach_threshold,
                )
            # Pre-flag a POV-lock metadata divergence too — the
            # orchestrator below coerces it back but a divergence is
            # high-signal that the rewriter abandoned the POV lock.
            pov_meta_feedback = _check_pov_lock_metadata(
                brief_pov,
                getattr(current_scene, "pov_entity", None),
                additional_locks=additional_locks or None,
                policy=pov_policy,
            )
            if pov_meta_feedback:
                det_findings.append(AuditViolation(
                    violation_type="reasoning_failure",
                    severity="critical",
                    description="POV-lock metadata divergence.",
                    feedback=pov_meta_feedback,
                ))
            if det_findings:
                # De-duplicate against the LLM auditor's findings on
                # (type, evidence_quote[:160]) so the same breach
                # isn't double-counted.
                existing_keys = {
                    (v.violation_type, (v.evidence_quote or "")[:160])
                    for v in audit.violations
                }
                for f in det_findings:
                    k = (f.violation_type, (f.evidence_quote or "")[:160])
                    if k in existing_keys:
                        continue
                    audit.violations.append(f)
                    existing_keys.add(k)
                # If the deterministic check fired but the LLM said
                # passed, override: deterministic findings are by
                # construction true positives.
                if det_findings and audit.passed:
                    logger.info(
                        "[FeedbackLoop] Deterministic prose check added "
                        "%d finding(s) the LLM auditor missed at "
                        "iteration %d; flipping audit.passed=False.",
                        len(det_findings), iteration + 1,
                    )
                    audit.passed = False

        # Record whether the deterministic prose block added new
        # violations so the convergence check can distinguish
        # "engine position-heuristic false-positive" (should not block
        # convergence when LLM cleared the prose) from "det_findings
        # caught a real POV / meta-narration breach the LLM missed"
        # (should block convergence regardless of llm_raw_passed).
        _det_added_violations = len(audit.violations) > _violations_pre_det

        # Snapshot the current state
        graph_version = versioned.version if versioned else 0
        graph_data = versioned.snapshot_data() if versioned else {}

        audit.change_impact = cycle_impact

        history.append(AuditCycleSnapshot(
            iteration=iteration,
            prose=current_scene.prose,
            audit_result=audit,
            graph_version=graph_version,
            graph_data=graph_data,
            change_impact=cycle_impact,
            engine_threshold_failures=list(engine_failures or []),
        ))

        # --- Track failed-open audits ---
        # A failed-open audit is an LLM/transport error, not evidence
        # the prose is bad. Previously the loop returned
        # ``converged=True`` after two consecutive failed-opens, which
        # let infrastructure outages masquerade as clean audit passes
        # and silently allowed downstream merge of unaudited prose.
        # We now return ``converged=False`` with a clear error so
        # callers (notably the pipeline merge gate) refuse to commit
        # the prose by default. Callers can opt into a force-merge
        # path explicitly when the failure is known-benign.
        if audit.failed_open:
            consecutive_failed_open += 1
            logger.warning(
                "[FeedbackLoop] Audit failed-open (%d consecutive) \u2014 "
                "no usable verdict; merge will be blocked unless caller "
                "explicitly forces.",
                consecutive_failed_open,
            )
            if consecutive_failed_open >= auditor_config.failed_open_tolerance:
                correction_error = (
                    f"Auditor failed-open {consecutive_failed_open} times "
                    f"in a row; refusing to mark the prose as audited. "
                    f"Last summary: {audit.audit_summary}"
                )
                return FeedbackLoopResult(
                    final_scene=current_scene,
                    converged=False,
                    iterations=iteration + 1,
                    history=history,
                    final_graph_version=graph_version,
                    change_impact=cycle_impact,
                    correction_error=correction_error,
                    engine_thresholds_passed=engine_passed,
                    engine_threshold_failures=engine_failures,
                )
        else:
            consecutive_failed_open = 0

        # --- Refinement-regression rollback ---
        # If the *previous* iteration's refinement introduced strictly
        # more violations than it closed, the rewriter regressed.
        # Roll back to the prior scene + audit and exit. This keeps
        # the user's final output at the best draft seen rather than
        # the latest draft (which is often worse). Specifically, the
        # rewriter is "regressing" when:
        #   (a) total violation count strictly increased, AND
        #   (b) at least one *new* violation type appeared that was
        #       not in the prior iteration's audit (so we are not
        #       just seeing the same gripe re-flagged with extra
        #       evidence). Pure increases on the same violation type
        #       are tolerated as the auditor finding more instances.
        if (
            iteration > 0
            and not audit.failed_open
            and prior_audit is not None
            and prior_violation_score is not None
        ):
            # P0 #1 (round-7 deeper audit 2026-05-27): unify the
            # regression key set across the LLM auditor AND the
            # deterministic engine. Before this, engine-only
            # regressions (e.g. a refinement closes every LLM
            # violation but breaks a miracle-step threshold) raised
            # the severity score but produced an empty
            # ``introduced_types`` set, so the rollback gate never
            # fired. The system's core promise is that the LLM is a
            # constrained renderer downstream of deterministic
            # physics — engine regressions therefore MUST be at
            # least as authoritative as LLM regressions.
            current_keys = {
                (v.violation_type, (v.evidence_quote or "")[:160])
                for v in audit.violations
            }
            for failure in engine_failures or []:
                current_keys.add(("engine_threshold", str(failure)[:160]))
            new_keys = current_keys - prior_violation_keys
            new_types = {t for (t, _q) in new_keys}
            prior_types = {t for (t, _q) in prior_violation_keys}
            introduced_types = new_types - prior_types
            # B + F (round-7 audit 2026-05-26): severity-weighted
            # score over the *unified* Finding stream (auditor
            # violations + engine failures). The raw violation count
            # used to ignore engine-threshold regressions; the
            # unified score catches a draft that closes every LLM
            # violation while breaking a hard physics threshold.
            current_score = _finding_severity_score(
                findings_from_sources(audit, engine_failures)
            )
            if (
                current_score > prior_violation_score
                and introduced_types
            ):
                if regression_retries_remaining > 0:
                    regression_retries_remaining -= 1
                    logger.warning(
                        "[FeedbackLoop] Refinement REGRESSION at iteration "
                        "%d: severity-weighted score rose %.1f -> %.1f and "
                        "%d new violation type(s) appeared (%s). Rolling "
                        "back to iteration %d prose and granting one "
                        "anti-regression retry (remaining=%d).",
                        iteration + 1, prior_violation_score,
                        current_score, len(introduced_types),
                        ", ".join(sorted(introduced_types)),
                        iteration, regression_retries_remaining,
                    )
                    if prior_scene is not None:
                        current_scene = prior_scene
                    audit = prior_audit
                    cycle_impact = prior_cycle_impact or cycle_impact
                    graph_version = prior_graph_version
                    graph_data = prior_graph_data
                    # R19-L8: after rolling back ``cycle_impact`` to
                    # the prior cycle's value, re-run the engine
                    # threshold check so ``engine_passed`` /
                    # ``engine_failures`` reflect the rolled-back
                    # impact rather than the regressed cycle's
                    # values.
                    try:
                        engine_passed, engine_failures = _engine_thresholds_check(
                            cycle_impact, auditor_config,
                        )
                    except Exception:
                        pass
                    pending_regression_warning = (
                        f"Your previous rewrite REGRESSED by introducing "
                        f"new violation type(s): "
                        f"{sorted(introduced_types)}. This is an "
                        f"anti-regression retry from the rolled-back "
                        f"draft above. Hold the line on those constraints "
                        f"absolutely while addressing the current "
                        f"violations below — do NOT trade one fix for "
                        f"another. If this rewrite regresses again the "
                        f"loop will exit and the rolled-back draft will "
                        f"be returned as the final output."
                    )
                    # Fall through into the refinement path so the
                    # loop produces a retry draft from the rolled-back
                    # prose with the REGRESSION ALERT surfaced.
                else:
                    logger.warning(
                        "[FeedbackLoop] Refinement REGRESSION at iteration "
                        "%d (retry budget exhausted): severity-weighted "
                        "score rose %.1f -> %.1f and %d new violation "
                        "type(s) appeared (%s). Rolling back to iteration "
                        "%d prose and exiting loop.",
                        iteration + 1, prior_violation_score,
                        current_score, len(introduced_types),
                        ", ".join(sorted(introduced_types)),
                        iteration,
                    )
                    correction_error = (
                        f"Refinement regressed at iteration "
                        f"{iteration + 1} after anti-regression retry was "
                        f"exhausted: introduced {sorted(introduced_types)}; "
                        f"rolled back."
                    )
                    if prior_scene is not None:
                        current_scene = prior_scene
                    audit = prior_audit
                    cycle_impact = prior_cycle_impact or cycle_impact
                    graph_version = prior_graph_version
                    graph_data = prior_graph_data
                    # R19-L8: re-run engine threshold check after rollback.
                    try:
                        engine_passed, engine_failures = _engine_thresholds_check(
                            cycle_impact, auditor_config,
                        )
                    except Exception:
                        pass
                    break

        # --- Convergence rule ---
        # The LLM auditor's prose-level verdict is the only signal that
        # actually responds to a rewrite. The engine veto is folded in
        # ONLY when (a) the engine produced metrics and (b) those
        # metrics are not structurally pinned-failing from iteration 0
        # (in which case prose cannot move them and gating on them
        # would create the infinite-rejection loop documented above).
        #
        # Use llm_raw_passed (the LLM's verdict before run_audit's
        # deterministic engine passes mutate audit.passed) so that
        # engine-injected false positives — e.g. the 120-char proximity
        # position-mismatch heuristic firing on "Mrs. Coady recalled
        # seeing George at the Old Bailey" when she is in her flat —
        # cannot prevent convergence on prose the LLM explicitly
        # cleared.  det_findings (POV / meta-narration regex checks in
        # the feedback loop itself) are still gated via
        # _det_added_violations; hard verbatim engine checks (withheld
        # utterance leaks, do-surgery leaks, etc.) are gated via
        # _hard_engine_added.
        _llm_verdict = (
            audit.llm_raw_passed
            if audit.llm_raw_passed is not None
            else audit.passed
        )
        llm_passed = (
            _llm_verdict
            and not audit.failed_open
            and not _det_added_violations
            and not _hard_engine_added
        )
        # Treat an audit whose only verdict was minor violations as
        # effectively passing for refinement-loop purposes. The
        # rewriter is unlikely to address purely cosmetic drift
        # without regressing on something more important, and burning
        # an iteration on it is a common cause of pointless ping-pong
        # (e.g.\u00a0iter 1 flags a minor density drift, iter 2 fixes
        # it but reintroduces a meta-narration leak, iter 3 flags
        # *that*\u2026). The audit object itself still carries
        # ``passed=False`` and the violations so callers can render
        # them in the UI \u2014 we just don't gate the loop on them.
        #
        # SAFETY: the minor-only bypass is restricted to an explicit
        # allowlist of cosmetic violation types. Without the allowlist
        # any future violation that happens to be flagged ``minor``
        # (including semantic regressions a rewriter could fix) is
        # silently waved through. Anything outside the allowlist
        # keeps ``llm_passed`` False so the loop continues.
        #
        # The allowlist is intentionally narrow: only violation types
        # the auditor's own prose explicitly classifies as "do NOT
        # spend a regeneration cycle on this alone" belong here.
        # Prior versions of this set listed speculative future types
        # (``prose_density``, ``stylistic_drift``, ``register_drift``,
        # ``pacing``, ``diction``, ``tone``, ``wordcount``,
        # ``sentence_length``, ``repetition``, ``filler``) that were
        # never wired through ``AuditViolation.violation_type``. They
        # were dead config surface — the bypass could never fire on
        # them — and were pruned in the 2026-05-26 round-2 audit.
        # Re-add a value here only after it is added to the
        # ``AuditViolation.violation_type`` Literal above.
        _MINOR_BYPASS_ALLOWLIST = {
            # ``style_mismatch`` at minor severity is the canonical
            # "do NOT spend a regeneration cycle on this alone" case
            # documented in ``assemble_audit_prompt`` and ``auditor.md``
            # (the auditor classifies pure prose-density drift inside
            # the loosened word band + intact form-class as minor and
            # tells the loop to skip it). Without ``style_mismatch`` on
            # this allowlist the loop still burns iterations on the
            # exact case the prompt was written to skip.
            "style_mismatch",
        }
        # Soft-affective violation types. These represent fine-grained
        # emotional/psychological calibration that may genuinely be at
        # the limit of what a single LLM pass can achieve on plausible
        # prose. After the plausibility threshold (see below) they no
        # longer block convergence when no hard-physics / world-state
        # violation is also present.
        _SOFT_AFFECTIVE_TYPES = {
            # Affective tuning — the effect is present but not perfectly
            # calibrated; prose is still plausible.
            "tonal_mismatch",
            "magnitude_too_low",
            "suspense_threshold",
            "low_kl_divergence",
            # Psychological nuance — character interiority is adequate
            # but not maximally precise.
            "reasoning_failure",
            "affective_failure",
            "attribution_failure",
            "empathy_weight",
            # Implicit subtext — the abduction signal exists but is subtle.
            "abduction_failure",
            # Style / form drift that is not a physics break.
            "style_mismatch",
        }
        # Hard world-state / information-control types that ALWAYS block
        # convergence regardless of severity or iteration count.
        _ALWAYS_HARD_TYPES = {
            "undeclared_element",
            "unjustified_introduction",
            "spurious_abduction",
            "premature_payoff",
            "epistemic_leakage",
            "knowledge_contamination",
            "withheld_utterance_leak",
            "pruned_utterance_leak",
            "disabled_channel_leak",
            "blocked_propagation_leak",
            "meta_narration",
            "miracle_step",
            "entity_position_mismatch",
            "object_position_mismatch",
            "event_location_mismatch",
            "event_copresence_violation",
            "event_copresence_omission",
            "inert_intervention_aftermath",
            "utterance_truth_contradiction",
            "belief_provenance_contradiction",
            "channel_intelligibility_violation",
        }
        # Use the LLM's own violation snapshot for the bypass check, not
        # the full augmented list. Engine-injected major violations
        # (entity_position_mismatch, co-presence, etc.) must not prevent the
        # bypass from firing when the LLM itself only raised minor cosmetic
        # issues — those are structurally unfixable by rewriting and burning
        # an iteration on them is exactly the pattern the bypass was designed
        # to prevent. Fall back to audit.violations when llm_raw_violations
        # is empty (failed_open path where llm_raw_violations was not set).
        _bypass_violations = (
            audit.llm_raw_violations
            if audit.llm_raw_violations
            else list(audit.violations)
        )
        if (
            not llm_passed
            and not audit.failed_open
            and not _det_added_violations  # never bypass when det checks fired real violations
            and not _hard_engine_added     # never bypass when hard engine checks fired
            and _bypass_violations
            and all(v.severity == "minor" for v in _bypass_violations)
            and all(
                getattr(v, "violation_type", "") in _MINOR_BYPASS_ALLOWLIST
                for v in _bypass_violations
            )
        ):
            logger.info(
                "[FeedbackLoop] All %d violation(s) at iteration %d are "
                "'minor' and in the cosmetic allowlist — treating as "
                "effectively passed; not spending a regeneration cycle.",
                len(audit.violations), iteration + 1,
            )
            llm_passed = True
        # --- Plausibility bypass ---
        # When the LLM's remaining violations are ALL soft-affective
        # (emotional calibration, psychological nuance, subtle subtext)
        # AND none are hard world-state / physics violations, accept the
        # prose after we have spent at least half the iteration budget.
        # This prevents the loop from grinding indefinitely on violations
        # that represent fine-grained LLM taste rather than broken physics
        # or world-model integrity — prose that an informed reader would
        # consider plausible should not loop forever because the auditor
        # wanted marginally sharper suspense or a more visceral adjective.
        #
        # Conditions for plausibility bypass:
        #   1. LLM did not pass on its own (otherwise this branch is moot).
        #   2. Audit did not fail-open (structured output parse failure).
        #   3. No deterministic checks added violations (POV / meta regex).
        #   4. No hard engine checks fired (utterance leaks, surgery leaks).
        #   5. ALL LLM violations are in _SOFT_AFFECTIVE_TYPES.
        #   6. NONE of the LLM violations are in _ALWAYS_HARD_TYPES.
        #   7. All remaining violations are at most ``major`` severity
        #      (i.e. no ``critical`` left — a critical means something
        #      is genuinely broken, not just imperfect).
        #   8. We have spent at least ceil(max_iterations / 2) iterations
        #      (so a first-pass lucky-miss can still get one retry).
        _plausibility_threshold = (auditor_config.max_iterations + 1) // 2
        _vtype = lambda v: getattr(v, "violation_type", "")  # noqa: E731
        if (
            not llm_passed
            and not audit.failed_open
            and not _det_added_violations
            and not _hard_engine_added
            and _bypass_violations
            and iteration >= _plausibility_threshold
            and all(_vtype(v) in _SOFT_AFFECTIVE_TYPES for v in _bypass_violations)
            and not any(_vtype(v) in _ALWAYS_HARD_TYPES for v in _bypass_violations)
            and all(v.severity != "critical" for v in _bypass_violations)
        ):
            logger.info(
                "[FeedbackLoop] Plausibility bypass at iteration %d/%d: "
                "%d soft-affective violation(s) remain (%s) but no hard "
                "physics or world-state violations — accepting plausible "
                "prose rather than grinding further.",
                iteration + 1, auditor_config.max_iterations,
                len(_bypass_violations),
                ", ".join(
                    f"{_vtype(v)}({v.severity})" for v in _bypass_violations
                ),
            )
            llm_passed = True
        engine_blocks_convergence = (
            engine_passed is False
            and not (engine_invariant and baseline_engine_passed is False)
        )
        if llm_passed and not engine_blocks_convergence:
            logger.info(
                "[FeedbackLoop] CONVERGED at iteration %d "
                "(llm_passed=%s, engine_passed=%s, baseline_engine_passed=%s). "
                "%s",
                iteration + 1, llm_passed, engine_passed,
                baseline_engine_passed, audit.audit_summary,
            )
            return FeedbackLoopResult(
                final_scene=current_scene,
                converged=True,
                iterations=iteration + 1,
                history=history,
                final_graph_version=graph_version,
                change_impact=cycle_impact,
                engine_thresholds_passed=engine_passed,
                engine_threshold_failures=engine_failures,
            )

        if llm_passed and engine_blocks_convergence:
            logger.info(
                "[FeedbackLoop] LLM auditor passed but engine thresholds "
                "regressed since baseline (%s). Continuing refinement.",
                "; ".join(engine_failures),
            )

        # --- Step 11.5: Mutate the brief on miracle-step verdicts ---
        # Re-prompting alone gives the rewriter no new structured signal
        # \u2014 it just sees the same scene + a textual nag. Promote each
        # miracle step into an explicit ``InterventionMechanism`` so the
        # next render pass sees a *typed directive* in the brief, not
        # just a paragraph of feedback. The rendering prompt is rebuilt
        # below from the (now mutated) brief so the new directives flow
        # into the LLM context.
        engine_blocked: List[BlockedPropagation] = []
        if physics_result is not None:
            engine_blocked = list(physics_result.blocked)
        injected_mechanisms = _inject_miracle_step_mechanisms(
            brief, engine_blocked, audit.violations,
            causal_feedback=cycle_causal,
        )
        if injected_mechanisms:
            logger.info(
                "[FeedbackLoop] Injected %d InterventionMechanism entries "
                "into brief from miracle-step violations.", injected_mechanisms,
            )
            # The brief was just mutated, so the next iteration's
            # engine scorecard may differ from the cached one. Force
            # a recompute on the next pass.
            engine_invariant = False

        # --- Step 12: Refinement ---
        logger.info(
            "[FeedbackLoop] FAILED audit \u2014 %d violations, %d engine failures. "
            "Regenerating.",
            len(audit.violations), len(engine_failures),
        )

        # Collect feedback for this iteration. Include engine failures
        # as synthetic feedback strings so the rewriter sees them too.
        iteration_feedback = [v.feedback for v in audit.violations]
        for failure in engine_failures:
            iteration_feedback.append(
                f"[engine-threshold] {failure} \u2014 adjust prose to fix."
            )
        accumulated_feedback.extend(iteration_feedback)

        # Fork the graph for the next iteration
        if versioned is not None:
            versioned.fork()

        # Rebuild the base rendering prompt from the (possibly mutated)
        # brief so any newly-injected InterventionMechanisms make it
        # into the next render pass.
        if injected_mechanisms:
            base_rendering_prompt = assemble_rendering_prompt(
                brief, query_type, physics_state,
            )

        # Build the augmented rendering prompt with feedback. Pass the
        # engine threshold failures alongside the LLM violations so the
        # rewriter actually sees both signal sources \u2014 previously
        # ``engine_failures`` only landed in ``accumulated_feedback``
        # for the *auditor's* next-iteration prior context, never in
        # the rewrite prompt itself.
        # ``accumulated_violations`` is the structured history of every
        # violation flagged in earlier iterations. Pass it so the
        # refinement prompt can list non-regression constraints and
        # break the ping-pong cycle between competing fixes.
        # A (round-7 audit 2026-05-26): annotate the previous prose
        # with inline <<<VIOLATION:type>>>...<<</VIOLATION>>> markers
        # around each violation's evidence_quote. This converts the
        # "minimal surgical edit" instruction from a vibe into a
        # directive the rewriter can mechanically execute: spans it
        # must touch are marked, and everything outside the markers
        # should be preserved verbatim. Quotes that don't appear in
        # the prose (the auditor paraphrased) are silently skipped and
        # remain available in the per-violation feedback list.
        _prev_prose = getattr(current_scene, "prose", None)
        if _prev_prose:
            _prev_prose = _annotate_prose_with_violation_spans(
                _prev_prose, audit.violations,
            )
        refinement_prompt = _build_refinement_prompt(
            base_rendering_prompt,
            audit.violations,
            iteration + 1,
            engine_failures=engine_failures,
            prior_violations=list(accumulated_violations),
            brief=brief,
            # Round-7 audit (2026-05-26): pass the (span-annotated)
            # previous draft so the rewriter can perform a minimal
            # surgical edit instead of a full re-roll that loses
            # correct POV/style/anti-meta choices.
            previous_prose=_prev_prose,
            # Round-7 audit (2026-05-26): when the rollback path
            # granted an anti-regression retry, surface the warning
            # here so the next rewrite sees a REGRESSION ALERT block
            # naming the violation types that were just introduced.
            regression_warning=pending_regression_warning,
            # Pass the prior draft's introduced_elements so the refiner
            # knows to preserve any declarations whose prose references
            # survive the rewrite (prevents deterministic
            # undeclared_element re-fires after unrelated edits).
            introduced_elements=getattr(
                current_scene, "introduced_elements", None,
            ),
        )
        # Consume the one-shot warning so a non-regressing next
        # iteration does not re-warn the rewriter.
        pending_regression_warning = None

        # Now extend with this iteration's violations so the *next*
        # refinement pass sees them as non-regression constraints.
        accumulated_violations.extend(audit.violations)

        # Snapshot this iteration's scene + audit BEFORE refinement so
        # the regression-rollback at the top of the next iteration can
        # restore them if the rewriter makes things strictly worse.
        prior_scene = current_scene
        prior_audit = audit
        prior_violation_score = _finding_severity_score(
            findings_from_sources(audit, engine_failures)
        )
        prior_violation_keys = {
            (v.violation_type, (v.evidence_quote or "")[:160])
            for v in audit.violations
        }
        # P0 #1 (round-7 deeper audit 2026-05-27): keep engine
        # failures in the prior-key snapshot too so the next
        # iteration's ``current_keys - prior_violation_keys`` diff
        # can detect a *new* engine-threshold breach.
        for failure in engine_failures or []:
            prior_violation_keys.add(("engine_threshold", str(failure)[:160]))
        prior_cycle_impact = cycle_impact
        prior_graph_version = graph_version
        prior_graph_data = graph_data

        # Re-generate the scene under the refinement system prompt so
        # the LLM is explicitly in rewrite mode (rather than reusing
        # the generic generation prompt and relying on injected text).
        agent = _build_generation_agent(
            generation_config,
            prompt_filename="refinement.md",
            stage="auditor_generation",
        )
        deps = _GenerationDeps(rendering_prompt=refinement_prompt)

        model_settings: Dict[str, Any] = {
            "max_tokens": generation_config.max_tokens,
        }
        if generation_config.temperature != 0.7:
            model_settings["temperature"] = generation_config.temperature

        try:
            result = agent.run_sync(
                f"Rewrite the scene. Target effect: {brief.target_effect}. "
                f"Address ALL auditor violations.",
                deps=deps,
                model_settings=model_settings if model_settings else None,
            )
            current_scene = result.output
            log_agent_output(logger, "Refinement", current_scene)
        except Exception as exc:
            logger.error(
                "[FeedbackLoop] Re-generation LLM call failed: %s. "
                "Keeping previous scene and exiting loop.", exc,
            )
            correction_error = f"Refinement LLM call raised: {exc!r}"
            current_scene = prior_scene
            break

        # Round-8 audit (defensive): pydantic-ai typed output should
        # always be a ``GeneratedScene`` in production, but tests that
        # patch ``_build_generation_agent`` with a bare ``MagicMock``
        # cause ``result.output`` to be a MagicMock. Subsequent
        # ``.model_copy()`` calls then propagate MagicMocks through to
        # ``FeedbackLoopResult`` construction, which fails pydantic
        # validation with a confusing ``model_type`` error far from
        # the root cause. Roll back to the prior known-good scene and
        # exit cleanly when this happens.
        if not isinstance(current_scene, GeneratedScene):
            logger.error(
                "[FeedbackLoop] Refinement returned a non-GeneratedScene "
                "output (%s); rolling back to prior scene and exiting.",
                type(current_scene).__name__,
            )
            correction_error = (
                f"Refinement returned non-GeneratedScene output: "
                f"{type(current_scene).__name__}"
            )
            current_scene = prior_scene
            break

        # --- Refinement rendering_mode contract ---
        # The refinement agent is forbidden from mutating
        # ``rendering_mode`` (and ``pov_entity`` when the brief
        # locked one). The auditor evaluates against the brief's
        # mode; if the rewriter silently relabels the structured
        # output, downstream consumers that key off
        # ``GeneratedScene.rendering_mode`` (the prose-by-mode UI
        # filter, the next iteration's audit prompt, the merge's
        # branch policy) drift out of step with the brief.
        #
        # Previously this was a hard exit \u2014 we kept the prior scene
        # and bailed the loop. In practice the rewriter often produces
        # a correct rewrite under the brief's mode and only mislabels
        # the metadata (the auditor judges against the brief's mode
        # regardless), so aborting was discarding a usable improvement
        # and pinning the user on the pre-refinement draft. Coerce the
        # metadata back to the brief's expectation, log a warning so
        # the regression is visible, and continue iterating.
        if (
            expected_rendering_mode is not None
            and current_scene.rendering_mode
            and current_scene.rendering_mode != expected_rendering_mode
        ):
            logger.warning(
                "[FeedbackLoop] Refinement agent mutated rendering_mode "
                "(%r -> %r) at iteration %d; coercing back to the "
                "brief's mode and continuing. The auditor evaluates "
                "against the brief, so the prose itself is still "
                "judged correctly \u2014 only the metadata was wrong.",
                expected_rendering_mode, current_scene.rendering_mode,
                iteration + 1,
            )
            current_scene = current_scene.model_copy(update={
                "rendering_mode": expected_rendering_mode,
            })

        # Coerce ``pov_entity`` back when the brief locked one and the
        # rewriter dropped it (commonly nulled at the same time as the
        # mode flip above). Under multi-POV policies (rotating /
        # ensemble) accept any roster member; only coerce back when
        # the rewriter dropped POV entirely or selected an entity
        # outside the roster.
        expected_pov = None
        expected_roster: list[str] = []
        if getattr(brief, "rendering", None) is not None:
            expected_pov = getattr(brief.rendering, "pov_lock", None)
            expected_roster = [expected_pov] + list(
                getattr(brief.rendering, "additional_pov_locks", []) or []
            ) if expected_pov else []
        current_pov = getattr(current_scene, "pov_entity", None)
        if (
            expected_pov
            and (current_pov is None or current_pov not in expected_roster)
        ):
            logger.warning(
                "[FeedbackLoop] Refinement agent dropped pov_entity "
                "(%r -> %r) at iteration %d; coercing back to the "
                "brief's primary pov_lock (roster=%r).",
                expected_pov, current_pov, iteration + 1, expected_roster,
            )
            current_scene = current_scene.model_copy(update={
                "pov_entity": expected_pov,
            })

        if current_scene.generation_error:
            # The render itself fell back to a placeholder; don't keep
            # iterating against bogus prose.
            logger.error(
                "[FeedbackLoop] Refinement returned a fallback scene "
                "(generation_error=%s). Exiting loop.",
                current_scene.generation_error,
            )
            correction_error = (
                f"Refinement produced fallback scene: "
                f"{current_scene.generation_error}"
            )
            break

        logger.info(
            "[FeedbackLoop] Re-rendered: %d chars, mode=%s",
            len(current_scene.prose), current_scene.rendering_mode,
        )

    # --- Exhausted iterations or broke out with correction_error ---
    if correction_error:
        logger.warning(
            "[FeedbackLoop] Exited with correction_error: %s",
            correction_error,
        )
    else:
        logger.warning(
            "[FeedbackLoop] Did NOT converge after %d iterations.",
            auditor_config.max_iterations,
        )

    # Final snapshot
    graph_version = versioned.version if versioned else 0
    graph_data = versioned.snapshot_data() if versioned else {}

    # Compute final engine metrics on non-convergence
    final_causal = compute_causal_feedback(
        physics_result, brief, world_state,
    )
    final_affective = compute_affective_feedback(
        brief, assembler,
    )
    final_impact = ChangeImpactMetrics(
        causal_feedback=final_causal,
        affective_feedback=final_affective,
    )
    final_engine_passed, final_engine_failures = _engine_thresholds_check(
        final_impact, auditor_config,
    )

    # --- Terminal audit on the post-refinement draft ---
    # The loop structure is (audit \u2192 refine) per iteration: the
    # refinement at the bottom of the final iteration mutates
    # ``current_scene`` but the loop exits before the *next*
    # iteration's audit fires. Without a terminal audit here, the
    # returned ``final_scene`` is one generation step ahead of the
    # last snapshot in ``history`` \u2014 the metadata says
    # "audited N times" but the prose the caller sees was never
    # audited at all. Run one final audit, append the snapshot, and
    # tag it ``iteration=max_iterations`` so consumers can
    # distinguish a terminal pass from a regular loop pass.
    #
    # Guards:
    #   * Skip when ``correction_error`` is set (the prose IS the
    #     pre-refinement scene; auditing it again is redundant).
    #   * Skip when the last history entry already covers
    #     ``current_scene.prose`` verbatim (the loop exited mid-
    #     iteration BEFORE refinement \u2014 e.g. via the regression
    #     rollback path \u2014 in which case the existing snapshot is
    #     authoritative).
    #   * Skip when the scene is a generation_error placeholder.
    terminal_audit_appended = False
    if (
        not correction_error
        and not current_scene.generation_error
        and (
            not history
            or history[-1].prose != current_scene.prose
        )
    ):
        logger.info(
            "[FeedbackLoop] Running terminal audit on the "
            "post-refinement draft (iteration=%d) so the returned "
            "scene is never one step ahead of its last audit.",
            auditor_config.max_iterations,
        )
        try:
            # Round-9 A1: use the *final* causal/affective metrics
            # recomputed against the post-refinement draft, not the
            # last loop-iteration's ``cycle_*`` snapshots. Otherwise
            # the terminal auditor reasons against stale engine
            # signals that may have changed in the final refinement.
            terminal_audit = run_audit(
                prose=current_scene.prose,
                brief=brief,
                config=auditor_config,
                prior_feedback=accumulated_feedback or None,
                causal_feedback=final_causal,
                world_state=world_state,
                affective_feedback=final_affective,
                introduced_elements=getattr(
                    current_scene, "introduced_elements", None,
                ),
            )
            terminal_audit.change_impact = final_impact
            history.append(AuditCycleSnapshot(
                iteration=auditor_config.max_iterations,
                prose=current_scene.prose,
                audit_result=terminal_audit,
                graph_version=graph_version,
                graph_data=graph_data,
                change_impact=final_impact,
                engine_threshold_failures=list(final_engine_failures or []),
            ))
            terminal_audit_appended = True
        except Exception:
            logger.exception(
                "[FeedbackLoop] Terminal audit FAILED; returning the "
                "unaudited post-refinement draft as non-converged. "
                "Callers gating on convergence will still refuse the "
                "merge; this only affects the audit_summary surface.",
            )

    return FeedbackLoopResult(
        final_scene=current_scene,
        converged=False,
        # Round-9 A2: ``iterations`` counts *audit/refine cycles*, not
        # snapshots. The terminal audit appended above shares the
        # iteration index of the final loop pass (it re-audits the
        # post-refinement draft of that same cycle) and should not
        # inflate the cycle count beyond ``max_iterations``.
        iterations=len(history) - (1 if terminal_audit_appended else 0),
        history=history,
        final_graph_version=graph_version,
        change_impact=final_impact,
        correction_error=correction_error,
        engine_thresholds_passed=final_engine_passed,
        engine_threshold_failures=final_engine_failures,
    )


# =====================================================================
# Convenience: render + audit in one call
# =====================================================================

def render_and_audit(
    brief: CreativeBrief,
    sandbox: Optional[nx.MultiDiGraph] = None,
    world_state: Optional[WorldStateV1] = None,
    auditor_config: AuditorConfig | None = None,
    generation_config: GenerationConfig | None = None,
    query_type: str = "directive",
    physics_state: Optional[Dict[str, Any]] = None,
    physics_result: Optional[CausalPhysicsResult] = None,
    assembler: Optional[DirectiveAssembler] = None,
) -> FeedbackLoopResult:
    """End-to-end: render a scene (Step 10) then audit+refine (Steps 11–12).

    This is the main entry point for the full generation pipeline with
    quality assurance.

    Returns
    -------
    FeedbackLoopResult
        The final audited scene with full iteration history.
    """
    generation_config = generation_config or GenerationConfig()

    # Step 10: Initial rendering
    initial_scene = render_scene(
        brief=brief,
        config=generation_config,
        query_type=query_type,
        physics_state=physics_state,
    )

    # Steps 11–12: Audit + refinement loop
    return run_feedback_loop(
        initial_scene=initial_scene,
        brief=brief,
        sandbox=sandbox,
        world_state=world_state,
        auditor_config=auditor_config,
        generation_config=generation_config,
        query_type=query_type,
        physics_state=physics_state,
        physics_result=physics_result,
        assembler=assembler,
    )
