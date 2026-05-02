# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
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
import os
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
)
from shadow_loom.generation import (
    GeneratedScene,
    GenerationConfig,
    assemble_rendering_prompt,
    render_scene,
    _GenerationDeps,
    _build_generation_agent,
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
    # Non-directive
    "observation": ["physics"],
    "intervention": ["physics"],
    "counterfactual": ["physics"],
    "general": ["physics"],
}


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
    affective_loss_mse: float = Field(
        default=0.0,
        description=(
            "The Mean Squared Error delta between the requested emotional "
            "intensity and the actual intensity achieved."
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
        default="ollama:qwen3.6:27b",
        description="PydanticAI model string for the auditor LLM.",
    )
    generation_model: str = Field(
        default="ollama:qwen3.6:27b",
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
            else:
                miracle_steps.append(entry)
        rule3_pruned = list(physics_result.rule3_pruned_interventions)
        rule2_redundant = list(physics_result.rule2_redundant_evidence)

    # --- Foreshadowing payoff: ratio of withheld narrative tensions resolved ---
    if brief.narrative_tensions:
        withheld = [
            nt for nt in brief.narrative_tensions
            if nt.tension_type == "withheld_cause"
        ]
        if withheld and world_state is not None:
            # A withheld event is "resolved" if any downstream chain_reaction
            # CausalEdge from it exists (it was eventually paid off).
            resolved_ids: set[str] = set()
            withheld_ids = {nt.event_id for nt in withheld}
            for ce in world_state.causal_topology:
                if (
                    ce.source_id in withheld_ids
                    and ce.causality_type == "chain_reaction"
                ):
                    resolved_ids.add(ce.source_id)
            # Denominator must match the numerator's set semantics —
            # using ``len(withheld)`` (the list) double-counts duplicate
            # ``event_id`` references and biases the score low.
            foreshadowing_score = (
                len(resolved_ids) / len(withheld_ids)
                if withheld_ids
                else 1.0
            )
        elif withheld:
            # No world state — score based on displacement magnitude
            avg_disp = sum(abs(nt.displacement) for nt in withheld) / len(withheld)
            foreshadowing_score = max(0.0, 1.0 - avg_disp)

    # --- Cognitive plausibility: entities NOT acting on false beliefs ---
    if brief.epistemic_gaps:
        total = len(brief.epistemic_gaps)
        contradicted = sum(
            1 for g in brief.epistemic_gaps if g.gap_type == "contradicted"
        )
        # Plausibility = fraction of beliefs that are NOT contradicted
        cog_plausibility_score = round(
            (total - contradicted) / total if total > 0 else 1.0, 4
        )
        if contradicted:
            cog_details = (
                f"{contradicted}/{total} entity beliefs are contradicted by "
                f"reality. Characters may be acting on false information."
            )
        else:
            cog_details = (
                f"All {total} tracked beliefs are consistent with reality."
            )

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
    affective_loss = 0.0

    if assembler is None:
        return AffectiveStateFeedback(
            emotional_trajectory_scores=trajectory_scores,
            kl_divergence_prediction_error=kl_divergence,
            affective_loss_mse=affective_loss,
        )

    # Compute the primary target effect score
    target = brief.target_effect
    try:
        affective_loss = assembler.compute_affective_score(target, eids)
    except Exception:
        logger.debug("[AffectiveFeedback] compute_affective_score failed for %s", target)

    # Compute trajectory scores for all structural effects
    score_map = {
        "mystery": "compute_mystery_score",
        "dramatic_irony": "compute_dramatic_irony_score",
        "suspense": "compute_suspense_score",
        "surprise": "compute_surprise_score",
    }
    for effect, method_name in score_map.items():
        try:
            method = getattr(assembler, method_name)
            trajectory_scores[effect] = round(method(eids), 4)
        except Exception:
            logger.debug("[AffectiveFeedback] %s failed", method_name)

    # KL divergence (surprise-specific)
    if "surprise" in trajectory_scores:
        kl_divergence = trajectory_scores["surprise"]

    feedback = AffectiveStateFeedback(
        emotional_trajectory_scores=trajectory_scores,
        kl_divergence_prediction_error=kl_divergence,
        affective_loss_mse=round(affective_loss, 4),
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
    if af.affective_loss_mse > config.max_affective_loss:
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
        if af.affective_loss_mse > config.max_affective_loss:
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


def assemble_audit_prompt(
    prose: str,
    brief: CreativeBrief,
    audit_categories: List[str],
    prior_feedback: Optional[List[str]] = None,
    causal_feedback: Optional[CausalPhysicsFeedback] = None,
    *,
    world_state: Optional[WorldStateV1] = None,
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

    sections.append(f"=== TARGET EFFECT: {brief.target_effect.upper()} ===")
    sections.append(f"Target entities: {', '.join(brief.target_entities)}")
    sections.append("")

    sections.append("=== CONSTRAINTS (from Creative Brief) ===")
    sections.append(_format_constraints_for_audit(brief.constraints))
    sections.append("")

    sections.append(f"=== AUDIT CATEGORIES TO CHECK: {', '.join(audit_categories)} ===")
    sections.append("")

    # World traits (structural constraints the prose must respect)
    world_traits = brief.scene_context.get("world_traits", []) if brief.scene_context else []
    if world_traits:
        sections.append(
            "=== WORLD TRAITS (structural constraints — prose must be "
            "consistent with these world-level facts) ==="
        )
        for wt in world_traits:
            wid = wt.get("id", "?")
            name = wt.get("name", wid)
            desc = wt.get("description", "")
            mag = wt.get("magnitude", {})
            mag_val = mag.get("value", 0.5) if isinstance(mag, dict) else 0.5
            domains = ", ".join(wt.get("affected_domains", []))
            sections.append(f"  {name} ({wid}): mag={mag_val:.2f}, domains=[{domains}]")
            if desc:
                sections.append(f"    {desc[:120]}")
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

    # Intervention mechanisms for physics audit
    if brief.intervention_mechanisms:
        sections.append(
            "=== INTERVENTION PHYSICS (state changes that MUST have "
            "mechanisms rendered) ==="
        )
        for m in brief.intervention_mechanisms:
            sections.append(
                f"  {m.node_id}: {m.old_state} → {m.new_state} "
                f"via {m.mechanism_hint} (inertia={m.inertia:.2f})"
            )
        sections.append("")

    # Abduction truths for physics audit
    if brief.abduction_truths:
        sections.append(
            "=== ABDUCTION BACKGROUND TRUTHS (must be subtextually "
            "present but NOT explicitly stated) ==="
        )
        for a in brief.abduction_truths:
            sections.append(f"  {a.entity_id}: {a.hidden_variable}")
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
                f"power={rt.power_dynamic:+.2f} "
                f"(asymmetry={rt.asymmetry_score:.2f})"
            )
        sections.append("")

    # Threat proximity for suspense/fear audit
    if brief.threat_proximity:
        tp = brief.threat_proximity
        sections.append("=== THREAT PROXIMITY ===")
        sections.append(f"  Threat: {tp.threat_description}")
        sections.append(
            f"  P(threat)={tp.threat_probability:.2f} "
            f"P(hope)={tp.hope_probability:.2f}"
        )
        if tp.spatial_distance is not None:
            sections.append(f"  Spatial distance: {tp.spatial_distance} hops")
        sections.append("")

    # Counterfactual branch for regret audit
    if brief.counterfactual_branch:
        cf = brief.counterfactual_branch
        sections.append("=== COUNTERFACTUAL BRANCH ===")
        sections.append(f"  Actual: {cf.actual_outcome}")
        sections.append(f"  Simulated: {cf.simulated_outcome}")
        sections.append("")

    # Causal attribution for rage audit
    if brief.causal_attribution:
        ca = brief.causal_attribution
        sections.append("=== CAUSAL ATTRIBUTION ===")
        sections.append(
            f"  Perpetrator: {ca.perpetrator_id}"
            + (f" ({ca.perpetrator_name})" if ca.perpetrator_name else "")
        )
        sections.append(f"  Loss: {ca.loss_event_id} — {ca.loss_description}")
        sections.append("")

    # Entanglement pairs for love audit
    if brief.entanglement_pairs:
        sections.append("=== ENTANGLEMENT PAIRS ===")
        for p in brief.entanglement_pairs:
            sections.append(
                f"  {p.entity_a} ↔ {p.entity_b}: "
                f"coupling={p.coupling_strength:.2f}"
            )
        sections.append("")

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
) -> str:
    """Augment the original rendering prompt with auditor feedback.

    The feedback is injected as additional HARD constraints that override
    any conflicting soft constraints from the original brief.
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

    feedback_lines.append(
        "=== REWRITE TASK ===\n"
        "Rewrite the prose passage from scratch, honouring ALL original "
        "constraints AND the auditor corrections above. The auditor will "
        "check again."
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

    if affective_feedback is not None:
        sections.append("=== ENGINE: AFFECTIVE METRICS (ground truth) ===")
        for effect, score in affective_feedback.emotional_trajectory_scores.items():
            sections.append(f"  {effect}: {score:.4f}")
        if affective_feedback.kl_divergence_prediction_error is not None:
            sections.append(
                f"  KL divergence (surprise): "
                f"{affective_feedback.kl_divergence_prediction_error:.4f}"
            )
        sections.append(
            f"  Affective loss MSE: {affective_feedback.affective_loss_mse:.4f}"
        )
        sections.append("")

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


def run_audit(
    prose: str,
    brief: CreativeBrief,
    config: AuditorConfig | None = None,
    prior_feedback: Optional[List[str]] = None,
    causal_feedback: Optional[CausalPhysicsFeedback] = None,
    *,
    world_state: Optional[WorldStateV1] = None,
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

    categories = EFFECT_AUDIT_CATEGORIES.get(
        brief.target_effect, ["physics"]
    )

    audit_prompt = assemble_audit_prompt(
        prose, brief, categories, prior_feedback, causal_feedback,
        world_state=world_state,
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
    consecutive_failed_open = 0
    correction_error: Optional[str] = None

    for iteration in range(auditor_config.max_iterations):
        logger.info(
            "[FeedbackLoop] === Iteration %d/%d ===",
            iteration + 1, auditor_config.max_iterations,
        )

        # --- Step 11: Audit ---
        # Compute the engine's deterministic physics ledger first so the
        # auditor LLM sees the same ground-truth (miracle steps,
        # cyclic clusters, ctf-calculus prunings) the cycle scorecard
        # below will use. Without this the LLM auditor was blind to the
        # facts the engine already proved.
        cycle_causal = compute_causal_feedback(
            physics_result, brief, world_state,
        )

        audit = run_audit(
            prose=current_scene.prose,
            brief=brief,
            config=auditor_config,
            prior_feedback=accumulated_feedback if iteration > 0 else None,
            causal_feedback=cycle_causal,
            world_state=world_state,
        )

        # Snapshot the current state
        graph_version = versioned.version if versioned else 0
        graph_data = versioned.snapshot_data() if versioned else {}

        # Compute per-cycle engine metrics (causal computed above; reuse).
        cycle_affective = compute_affective_feedback(
            brief, assembler,
        )
        cycle_impact = ChangeImpactMetrics(
            causal_feedback=cycle_causal,
            affective_feedback=cycle_affective,
        )
        audit.change_impact = cycle_impact

        # Deterministic engine-side scorecard. The previous loop only
        # honoured the LLM auditor's self-reported ``passed`` flag,
        # which made convergence vulnerable to a hallucinated
        # ``passed=True``. We now require both signals when the
        # engine produced impact metrics.
        engine_passed, engine_failures = _engine_thresholds_check(
            cycle_impact, auditor_config,
        )

        history.append(AuditCycleSnapshot(
            iteration=iteration,
            prose=current_scene.prose,
            audit_result=audit,
            graph_version=graph_version,
            graph_data=graph_data,
            change_impact=cycle_impact,
        ))

        # --- Track failed-open audits separately from real fails ---
        if audit.failed_open:
            consecutive_failed_open += 1
            logger.warning(
                "[FeedbackLoop] Audit failed-open (%d consecutive). "
                "Treating as NOT converged.",
                consecutive_failed_open,
            )
            if consecutive_failed_open >= 2:
                # The auditor LLM is repeatedly broken. Bail out so
                # the caller sees a non-converged result with a typed
                # error rather than spinning the loop forever.
                correction_error = (
                    f"Auditor failed-open {consecutive_failed_open} times "
                    f"in a row: {audit.audit_summary}"
                )
                break
        else:
            consecutive_failed_open = 0

        # --- Convergence: require BOTH the LLM pass and the engine
        #     thresholds (when the engine produced metrics). ---
        llm_passed = audit.passed and not audit.failed_open
        if llm_passed and engine_passed is not False:
            logger.info(
                "[FeedbackLoop] CONVERGED at iteration %d "
                "(llm_passed=%s, engine_passed=%s). %s",
                iteration + 1, llm_passed, engine_passed,
                audit.audit_summary,
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

        if llm_passed and engine_passed is False:
            logger.info(
                "[FeedbackLoop] LLM auditor passed but engine thresholds "
                "failed (%s). Continuing refinement.",
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

        # Build the augmented rendering prompt with feedback
        refinement_prompt = _build_refinement_prompt(
            base_rendering_prompt,
            audit.violations,
            iteration + 1,
        )

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
