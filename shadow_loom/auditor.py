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
from typing import Any, Dict, List, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, NativeOutput, RunContext

from shadow_loom.causal_physics import CausalPhysicsResult, BlockedPropagation
from shadow_loom.directive_assembly import (
    CreativeBrief,
    ConstraintBlock,
    DirectiveAssembler,
)
from shadow_loom.generation import (
    GeneratedScene,
    GenerationConfig,
    assemble_rendering_prompt,
    render_scene,
    _GenerationDeps,
    _build_generation_agent,
)
from shadow_loom.models import WorldStateV1

from shadow_loom.settings import get_settings as _get_settings

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _ollama_base_url() -> str:
    return _get_settings().core.ollama_base_url

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
        default=3,
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
        default=2048,
        description="Max tokens for auditor response.",
    )
    max_tokens_generation: int = Field(
        default=4096,
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
    foreshadowing_score = 1.0
    cog_plausibility_score = 1.0
    cog_details = ""

    # --- Miracle steps: blocked propagations where impact failed inertia ---
    if physics_result is not None:
        for b in physics_result.blocked:
            miracle_steps.append(
                f"{b.node_id}.{b.trait}: impact={b.impact:.2f} < "
                f"inertia={b.inertia:.2f} ({b.reason})"
            )

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
            foreshadowing_score = (
                len(resolved_ids) / len(withheld)
                if withheld
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

    return CausalPhysicsFeedback(
        miracle_steps_detected=miracle_steps,
        foreshadowing_payoff_score=round(foreshadowing_score, 4),
        cognitive_plausibility_score=cog_plausibility_score,
        cognitive_plausibility_details=cog_details,
    )


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

    return AffectiveStateFeedback(
        emotional_trajectory_scores=trajectory_scores,
        kl_divergence_prediction_error=kl_divergence,
        affective_loss_mse=round(affective_loss, 4),
    )


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
    if cf.miracle_steps_detected:
        return False
    return True


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


def _resolve_model(model_str: str):
    if model_str.startswith("ollama:"):
        base_url = _ollama_base_url()
        model_name = model_str.split(":", 1)[1]
        from pydantic_ai.models.ollama import OllamaModel
        from pydantic_ai.providers.ollama import OllamaProvider
        return OllamaModel(model_name, provider=OllamaProvider(base_url=base_url))
    return model_str


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

    sections.append(
        "=== TASK ===\n"
        "Audit the prose above against the constraints and physics state. "
        "Return the structured audit result."
    )

    return "\n".join(sections)


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
    if config.max_tokens_audit != 2048:
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
    logger.info(
        "[Evaluation] Quality synthesis complete: %d rewrite directives.",
        len(synthesis.actionable_rewrite_directives),
    )
    return synthesis


# =====================================================================
# Single Audit Pass
# =====================================================================

def run_audit(
    prose: str,
    brief: CreativeBrief,
    config: AuditorConfig | None = None,
    prior_feedback: Optional[List[str]] = None,
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
        prose, brief, categories, prior_feedback,
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
    if config.max_tokens_audit != 2048:
        model_settings["max_tokens"] = config.max_tokens_audit

    try:
        result = agent.run_sync(
            f"Audit the prose for target effect: {brief.target_effect}.",
            deps=deps,
            model_settings=model_settings if model_settings else None,
        )
    except Exception as exc:
        logger.error("[Auditor] LLM call failed: %s. Returning pass-through.", exc)
        return AuditResult(
            passed=True,
            violations=[],
            audit_summary=f"Audit skipped due to LLM error: {exc}",
            failed_open=True,
        )

    audit = result.output
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

    for iteration in range(auditor_config.max_iterations):
        logger.info(
            "[FeedbackLoop] === Iteration %d/%d ===",
            iteration + 1, auditor_config.max_iterations,
        )

        # --- Step 11: Audit ---
        audit = run_audit(
            prose=current_scene.prose,
            brief=brief,
            config=auditor_config,
            prior_feedback=accumulated_feedback if iteration > 0 else None,
        )

        # Snapshot the current state
        graph_version = versioned.version if versioned else 0
        graph_data = versioned.snapshot_data() if versioned else {}

        # Compute per-cycle engine metrics
        cycle_causal = compute_causal_feedback(
            physics_result, brief, world_state,
        )
        cycle_affective = compute_affective_feedback(
            brief, assembler,
        )
        cycle_impact = ChangeImpactMetrics(
            causal_feedback=cycle_causal,
            affective_feedback=cycle_affective,
        )
        audit.change_impact = cycle_impact

        history.append(AuditCycleSnapshot(
            iteration=iteration,
            prose=current_scene.prose,
            audit_result=audit,
            graph_version=graph_version,
            graph_data=graph_data,
            change_impact=cycle_impact,
        ))

        # --- Check convergence ---
        if audit.passed:
            logger.info(
                "[FeedbackLoop] CONVERGED at iteration %d. %s",
                iteration + 1, audit.audit_summary,
            )
            return FeedbackLoopResult(
                final_scene=current_scene,
                converged=True,
                iterations=iteration + 1,
                history=history,
                final_graph_version=graph_version,
                change_impact=cycle_impact,
            )

        # --- Step 12: Refinement ---
        logger.info(
            "[FeedbackLoop] FAILED audit — %d violations. Regenerating.",
            len(audit.violations),
        )

        # Collect feedback for this iteration
        iteration_feedback = [v.feedback for v in audit.violations]
        accumulated_feedback.extend(iteration_feedback)

        # Fork the graph for the next iteration
        if versioned is not None:
            versioned.fork()

        # Build the augmented rendering prompt with feedback
        refinement_prompt = _build_refinement_prompt(
            base_rendering_prompt,
            audit.violations,
            iteration + 1,
        )

        # Re-generate the scene
        agent = _build_generation_agent(generation_config)
        deps = _GenerationDeps(rendering_prompt=refinement_prompt)

        model_settings: Dict[str, Any] = {}
        if generation_config.temperature != 0.7:
            model_settings["temperature"] = generation_config.temperature
        if generation_config.max_tokens != 4096:
            model_settings["max_tokens"] = generation_config.max_tokens

        try:
            result = agent.run_sync(
                f"Rewrite the scene. Target effect: {brief.target_effect}. "
                f"Address ALL auditor violations.",
                deps=deps,
                model_settings=model_settings if model_settings else None,
            )
            current_scene = result.output
        except Exception as exc:
            logger.error("[FeedbackLoop] Re-generation LLM call failed: %s. Keeping previous scene.", exc)
            break

        logger.info(
            "[FeedbackLoop] Re-rendered: %d chars, mode=%s",
            len(current_scene.prose), current_scene.rendering_mode,
        )

    # --- Exhausted iterations ---
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

    return FeedbackLoopResult(
        final_scene=current_scene,
        converged=False,
        iterations=auditor_config.max_iterations,
        history=history,
        final_graph_version=graph_version,
        change_impact=final_impact,
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
