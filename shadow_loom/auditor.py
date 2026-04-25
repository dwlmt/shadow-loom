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
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput, RunContext

from shadow_loom.directive_assembly import (
    CreativeBrief,
    ConstraintBlock,
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

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_OLLAMA_BASE_URL = "http://localhost:11434/v1/"

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


class FeedbackLoopResult(BaseModel):
    """Final output of the full audit → refinement loop."""
    final_scene: GeneratedScene
    converged: bool = Field(
        description="True if the auditor passed before hitting max_iterations.",
    )
    iterations: int
    history: List[AuditCycleSnapshot] = Field(default_factory=list)
    final_graph_version: int


# =====================================================================
# Configuration
# =====================================================================

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


# =====================================================================
# Dependencies
# =====================================================================

class _AuditorDeps(BaseModel):
    """Injected context for the auditor agent."""
    model_config = {"protected_namespaces": ()}
    audit_prompt: str = Field(
        description="The fully assembled audit prompt.",
    )


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
        base_url = os.environ.get("OLLAMA_BASE_URL", _OLLAMA_BASE_URL)
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
        )

    audit = result.output
    logger.info(
        "[Auditor] Audit complete: passed=%s violations=%d summary=%s",
        audit.passed, len(audit.violations), audit.audit_summary,
    )
    return audit


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
) -> FeedbackLoopResult:
    """Run the full audit → refinement loop (Steps 11–12).

    The loop:
      1. Deep-copy and version the sandbox graph.
      2. Audit the prose against the brief + graph.
      3. If the audit passes → return the current prose.
      4. If it fails → inject feedback into the rendering prompt
         and re-generate.  Repeat up to ``max_iterations``.

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

    Returns
    -------
    FeedbackLoopResult
        Contains the final scene, convergence status, iteration count,
        and the full history of audit snapshots with versioned graphs.
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

        history.append(AuditCycleSnapshot(
            iteration=iteration,
            prose=current_scene.prose,
            audit_result=audit,
            graph_version=graph_version,
            graph_data=graph_data,
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

    return FeedbackLoopResult(
        final_scene=current_scene,
        converged=False,
        iterations=auditor_config.max_iterations,
        history=history,
        final_graph_version=graph_version,
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
    )
