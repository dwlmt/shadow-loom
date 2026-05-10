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
from typing import Any, Dict, List, Literal, Optional, Tuple

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
)
from shadow_loom.models import EventNode, WorldStateV1

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
    # Non-directive
    "observation": ["physics"],
    "intervention": ["physics"],
    "counterfactual": ["physics"],
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


def resolve_audit_categories(target_effect: str) -> List[str]:
    """Return the full audit-category list for a given target effect.

    Combines the per-effect categories with
    :data:`UNIVERSAL_AUDIT_CATEGORIES` (deduplicated, order-preserving).
    Use this single helper at every call-site so the auditor prompt and
    the run-audit log stay in sync.
    """
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
        default=3,
        description="Maximum audit → rewrite cycles before giving up.",
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
        default=64000,
        description="Max tokens for auditor response.",
    )
    max_tokens_generation: int = Field(
        default=64000,
        description="Max tokens for generation response.",
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
            elif b.reason == "inertia" and _treat_inertia_as_absorbed:
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
        if not (ap or ab or ac):
            continue
        out.append(f"=== {label} (typed DoTarget surgery) ===")
        if ap:
            out.append(
                "  PROPOSITIONS whose truth flipped: "
                + ", ".join(ap[:12])
            )
        if ab:
            out.append(
                "  BELIEFS whose confidence shifted "
                "(holder→target): " + ", ".join(ab[:12])
            )
        if ac:
            out.append(
                "  CONCERNS whose polarity / salience shifted: "
                + ", ".join(ac[:12])
            )
        tragedy = getattr(payload, "tragedy_form", None)
        if tragedy:
            out.append(f"  tragedy_form: {tragedy}")
        out.append(
            "  Rule: every flipped proposition / belief / concern "
            "above MUST be visibly grounded in an on-page event "
            "or utterance; flag silent off-page changes as "
            "miracle steps."
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
        ]),
        ("GRIEF APPRAISAL", brief.grief_profile, [
            ("coupling_strength", "coupling_strength"),
            ("stage", "stage"),
            ("unfinished_concerns", "unfinished_concern_count"),
            ("lost_entity", "lost_entity_id"),
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
            sections.append(
                f"  pov_lock: {r.pov_lock} \u2014 the prose MUST stay "
                f"inside this entity's perception. Flag head-hopping "
                f"or omniscient narration as a violation under the "
                f"`physics` category (rationale prefix `pov_lock:`)."
            )
        if r.tone_arc:
            sections.append(f"  tone_arc: {r.tone_arc}")
        if r.stylistic_instructions:
            sections.append("  stylistic_instructions:")
            for j, si in enumerate(r.stylistic_instructions, 1):
                sections.append(f"    {j}. {si}")
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
    if has_prevented or has_false_props:
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
                f"Do not flag these as prose problems:"
            )
            for cc in causal_feedback.cyclic_propagation_clusters:
                sections.append(f"    - {cc}")
        if causal_feedback.noisy_or_absorbed_propagations:
            sections.append(
                f"  Noisy-OR absorbed propagations "
                f"({len(causal_feedback.noisy_or_absorbed_propagations)}) "
                f"— probabilistic gate dampened weak impulses (impact ≪ "
                f"inertia), NOT miracle steps. Do not flag these as prose "
                f"problems:"
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


def _inject_miracle_step_mechanisms(
    brief: CreativeBrief,
    blocked: List[BlockedPropagation],
    violations: List[AuditViolation],
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

    for b in blocked:
        if b.reason != "inertia":
            # Spatial / cycle blocks aren't narrative miracles \u2014
            # their fix is in the world topology, not in the prose.
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

    # The LLM auditor may also flag miracle-steps the engine missed
    # (e.g. an off-graph trait the renderer changed without justification).
    for v in violations:
        if v.violation_type != "miracle_step":
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

    return added


def _build_refinement_prompt(
    original_rendering_prompt: str,
    violations: List[AuditViolation],
    iteration: int,
    engine_failures: Optional[List[str]] = None,
    prior_violations: Optional[List[AuditViolation]] = None,
    brief: Optional[CreativeBrief] = None,
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
    """
    feedback_lines: List[str] = [
        "",
        f"=== AUDITOR FEEDBACK (Iteration {iteration}) ===",
        "The NarrativeAuditor found the following violations in your "
        "previous draft. You MUST address ALL of them in this rewrite:",
        "",
    ]

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
        "Rewrite the prose passage from scratch, honouring ALL original "
        "constraints AND the auditor corrections above. The auditor will "
        "check again."
    )

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

    return original_rendering_prompt + "\n".join(feedback_lines)


# =====================================================================
# Auditor Agent
# =====================================================================

def _build_auditor_agent(
    config: AuditorConfig,
) -> Agent[_AuditorDeps, AuditResult]:
    """Construct the Step 11 auditor LLM agent."""
    agent: Agent[_AuditorDeps, AuditResult] = Agent(
        _resolve_model(config.auditor_model),
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
        _resolve_model(config.auditor_model),
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

    agent = _build_evaluation_agent(config)
    deps = _EvaluationDeps(evaluation_prompt=eval_prompt)

    model_settings: Dict[str, Any] = {}
    if config.auditor_temperature != 0.2:
        model_settings["temperature"] = config.auditor_temperature
    if config.max_tokens_audit != 64000:
        model_settings["max_tokens"] = config.max_tokens_audit

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

    # 3. Blocked propagation leaks: per-sentence node+trait co-mention.
    if blocked_pairs:
        ent_names: Dict[str, str] = {}
        for store_attr in ("entities", "objects", "locations", "world_traits"):
            store = getattr(world_state, store_attr, None) or {}
            if isinstance(store, dict):
                for nid, node in store.items():
                    nm = getattr(node, "name", None)
                    if isinstance(nm, str) and nm.strip():
                        ent_names[nid] = nm.strip()
        sentences = re.split(r"(?<=[.!?])\s+", prose)
        seen_pairs: set[str] = set()
        for pair in blocked_pairs:
            if "." not in pair or pair in seen_pairs:
                continue
            node_id, trait = pair.split(".", 1)
            trait_low = trait.replace("_", " ").lower().strip()
            if len(trait_low) < 4:
                continue
            name_low = ent_names.get(node_id, "").lower()
            for sent in sentences:
                sl = sent.lower()
                node_hit = (
                    node_id in sent
                    or (len(name_low) >= 3 and name_low in sl)
                )
                if node_hit and trait_low in sl:
                    seen_pairs.add(pair)
                    issues.append(AuditViolation(
                        violation_type="blocked_propagation_leak",
                        severity="major",
                        description=(
                            f"Prose names blocked propagation target "
                            f"{pair}; the engine recorded this "
                            f"(node, trait) as resisted \u2014 the "
                            f"propagated state must not be depicted."
                        ),
                        evidence_quote=sent.strip()[:200],
                        feedback=(
                            f"Render the RESISTANCE for {pair}, not "
                            f"the propagated state. Show the force, "
                            f"inertia, or affordance constraint that "
                            f"stopped the change from taking hold."
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


def _event_copresence_violations(
    prose: str,
    brief: CreativeBrief,
    world_state: Optional[WorldStateV1],
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

    def _name(nid: str) -> str:
        if nid in entities:
            return getattr(entities[nid], "name", nid) or nid
        if nid in locations:
            return getattr(locations[nid], "name", nid) or nid
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
    )

    logger.info(
        "[Auditor] Running audit: effect=%s categories=%s prompt_len=%d",
        brief.target_effect, categories, len(audit_prompt),
    )

    agent = _build_auditor_agent(config)
    deps = _AuditorDeps(audit_prompt=audit_prompt)

    model_settings: Dict[str, Any] = {}
    if config.auditor_temperature != 0.2:
        model_settings["temperature"] = config.auditor_temperature
    if config.max_tokens_audit != 64000:
        model_settings["max_tokens"] = config.max_tokens_audit

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

    # Deterministic event co-presence / spatial-anchor check (PR 4 of
    # EventNode.at_location_id). Cross-references the spatial
    # ConstraintBlocks emitted by ``build_event_copresence_constraints``
    # against the prose for verbatim violations of MUST_BE_PRESENT
    # and MUST_NOT_BE_PRESENT ledgers.
    copresence_issues = _event_copresence_violations(
        prose, brief, world_state,
    )
    if copresence_issues:
        audit.violations = list(audit.violations) + copresence_issues
        audit.passed = False
        audit.audit_summary = (
            f"{audit.audit_summary} [+{len(copresence_issues)} event "
            f"co-presence violation(s)]"
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

    # Track the prior iteration's violation count and scene so we can
    # roll back when the refinement agent INTRODUCES more violations
    # than it closes. Without this guard the loop happily accepts a
    # strictly-worse rewrite (e.g.\u00a0closes 1 minor density drift
    # while opening a major meta-narration leak) and the user sees the
    # regressed prose as the final output.
    prior_violation_count: Optional[int] = None
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
        ))

        # --- Track failed-open audits ---
        # A failed-open audit is an LLM/transport error, not evidence
        # the prose is bad. Treat it as bypass-passed (the auditor has
        # nothing useful to say) so transient infra failures don't
        # masquerade as story-quality verdicts. We still bail after a
        # streak so the loop doesn't spin forever waiting on a broken
        # auditor.
        if audit.failed_open:
            consecutive_failed_open += 1
            logger.warning(
                "[FeedbackLoop] Audit failed-open (%d consecutive) \u2014 "
                "treating as bypass-passed (auditor produced no usable "
                "verdict).",
                consecutive_failed_open,
            )
            if consecutive_failed_open >= 2:
                correction_error = (
                    f"Auditor failed-open {consecutive_failed_open} times "
                    f"in a row; bypassing the prose-level audit. Last "
                    f"summary: {audit.audit_summary}"
                )
                # Bypass-pass: return the current scene as converged
                # under the bypass policy rather than a hard failure.
                return FeedbackLoopResult(
                    final_scene=current_scene,
                    converged=True,
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
            and prior_violation_count is not None
        ):
            current_keys = {
                (v.violation_type, (v.evidence_quote or "")[:160])
                for v in audit.violations
            }
            new_keys = current_keys - prior_violation_keys
            new_types = {t for (t, _q) in new_keys}
            prior_types = {t for (t, _q) in prior_violation_keys}
            introduced_types = new_types - prior_types
            if (
                len(audit.violations) > prior_violation_count
                and introduced_types
            ):
                logger.warning(
                    "[FeedbackLoop] Refinement REGRESSION at iteration %d: "
                    "violation count rose %d -> %d and %d new violation "
                    "type(s) appeared (%s). Rolling back to iteration %d "
                    "prose and exiting loop.",
                    iteration + 1, prior_violation_count,
                    len(audit.violations), len(introduced_types),
                    ", ".join(sorted(introduced_types)),
                    iteration,
                )
                correction_error = (
                    f"Refinement regressed at iteration {iteration + 1}: "
                    f"introduced {sorted(introduced_types)}; rolled back."
                )
                # Use the prior iteration's scene/audit as the final.
                if prior_scene is not None:
                    current_scene = prior_scene
                audit = prior_audit
                cycle_impact = prior_cycle_impact or cycle_impact
                graph_version = prior_graph_version
                graph_data = prior_graph_data
                break

        # --- Convergence rule ---
        # The LLM auditor's prose-level verdict is the only signal that
        # actually responds to a rewrite. The engine veto is folded in
        # ONLY when (a) the engine produced metrics and (b) those
        # metrics are not structurally pinned-failing from iteration 0
        # (in which case prose cannot move them and gating on them
        # would create the infinite-rejection loop documented above).
        llm_passed = audit.passed and not audit.failed_open
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
        if (
            not llm_passed
            and not audit.failed_open
            and audit.violations
            and all(v.severity == "minor" for v in audit.violations)
        ):
            logger.info(
                "[FeedbackLoop] All %d violation(s) at iteration %d are "
                "'minor' \u2014 treating as effectively passed; not "
                "spending a regeneration cycle.",
                len(audit.violations), iteration + 1,
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
        refinement_prompt = _build_refinement_prompt(
            base_rendering_prompt,
            audit.violations,
            iteration + 1,
            engine_failures=engine_failures,
            prior_violations=list(accumulated_violations),
            brief=brief,
        )

        # Now extend with this iteration's violations so the *next*
        # refinement pass sees them as non-regression constraints.
        accumulated_violations.extend(audit.violations)

        # Snapshot this iteration's scene + audit BEFORE refinement so
        # the regression-rollback at the top of the next iteration can
        # restore them if the rewriter makes things strictly worse.
        prior_scene = current_scene
        prior_audit = audit
        prior_violation_count = len(audit.violations)
        prior_violation_keys = {
            (v.violation_type, (v.evidence_quote or "")[:160])
            for v in audit.violations
        }
        prior_cycle_impact = cycle_impact
        prior_graph_version = graph_version
        prior_graph_data = graph_data

        # Re-generate the scene under the refinement system prompt so
        # the LLM is explicitly in rewrite mode (rather than reusing
        # the generic generation prompt and relying on injected text).
        agent = _build_generation_agent(
            generation_config, prompt_filename="refinement.md",
        )
        deps = _GenerationDeps(rendering_prompt=refinement_prompt)

        model_settings: Dict[str, Any] = {}
        if generation_config.temperature != 0.7:
            model_settings["temperature"] = generation_config.temperature
        if generation_config.max_tokens != 64000:
            model_settings["max_tokens"] = generation_config.max_tokens

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
        # mode flip above).
        expected_pov = None
        if getattr(brief, "rendering", None) is not None:
            expected_pov = getattr(brief.rendering, "pov_lock", None)
        if (
            expected_pov
            and getattr(current_scene, "pov_entity", None) != expected_pov
        ):
            logger.warning(
                "[FeedbackLoop] Refinement agent dropped pov_entity "
                "(%r -> %r) at iteration %d; coercing back to the "
                "brief's pov_lock.",
                expected_pov, getattr(current_scene, "pov_entity", None),
                iteration + 1,
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

    return FeedbackLoopResult(
        final_scene=current_scene,
        converged=False,
        iterations=len(history),
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
