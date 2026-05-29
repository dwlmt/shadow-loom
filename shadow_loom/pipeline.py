# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Complete end-to-end pipeline orchestrator for Shadow-Loom.

Connects all pipeline stages into a single ``run_pipeline()`` entry point.
The pipeline has **12 distinct execution paths** depending on query type:

  1. **Ingestion** (optional) — raw text → ``WorldStateV1``
  2. **Narrative Physics** — query routing, ego-graph, sandbox, causal engine (Rungs 1-3)
  3. **Brief Assembly** — ``CreativeBrief`` from physics result
  4. **Generation** — prose rendering with scene context
  5. **Audit** — narrative quality validation
  6. **Refinement** — recursive feedback loop (up to N iterations)
  7. **Re-extraction** — extract topology from generated prose
  8. **Merge** — merge topology into versioned deep-copy
  9. **Answer Step** (interrogate/general queries) — Q&A over physics state
  10. **Evaluate Step** — Monte Carlo affective trajectory analysis
  11. **Shadow Prune** — retrospective event pruning for counterfactuals
  12. **Async Worker** — thread-pool ingestion for concurrent execution

Execution paths:
  • **Standard prose** (observation/intervention/counterfactual): Steps 1-8
  • **Interrogate/General**: Steps 1-2, 9
  • **Evaluate**: Steps 1-2, 10
  • **With refinement**: Steps 1-6 repeated up to max_refinement_iterations

The pipeline is flexible:
  • An existing ``WorldStateV1`` can be passed in (skips ingestion).
  • A ``VersionedWorldModel`` can be passed in to continue an existing history.
  • Non-prose query types skip prose generation/audit/merge.
  • Every intermediate result is recorded in ``PipelineHistory``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from shadow_loom.settings import get_settings as _get_settings

from shadow_loom.auditor import (
    AuditorConfig,
    FeedbackLoopResult,
    compute_causal_feedback,
    compute_affective_feedback,
    _finalize_narrative_order,
    render_and_audit,
)
from shadow_loom.causal_closure import (
    chain_reaction_parents_from_world_state,
    expand_chain_reaction_closure,
)
from shadow_loom.directive_assembly import CreativeBrief, DirectiveAssembler
from shadow_loom.extract_graph import (
    VersionedWorldModel,
    extract_topology_from_prose,
    introduced_elements_to_spawns,
    promote_sandbox_spawns,
)
from shadow_loom.generation import (
    GeneratedScene,
    GenerationConfig,
    render_from_query,
)
from shadow_loom.ingestion import (
    ExtractionConfig,
    run_extraction_async,
    validate_and_correct_world_state,
    validate_and_correct_world_state_async,
)
from shadow_loom.ingestion_diagnostics import capture_ingestion_warnings

if TYPE_CHECKING:
    from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import WorldStateV1
from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import UserRequest, EvaluationQuery, EvaluationResult, ManualEditQuery

logger = logging.getLogger(__name__)


# Mirrors ``shadow_loom.ingestion._VALID_STATUSES`` (kept here as a
# tiny local copy to avoid an import cycle: ingestion already imports
# pipeline-adjacent helpers). Used by ``_augment_topology_with_sandbox_deltas``
# to coerce free-text observation_facts onto the EntityUpdate schema.
_VALID_OBSERVATION_STATUSES: set[str] = {
    "healthy", "injured", "ill", "dead", "unconscious",
}
_OBSERVATION_STATUS_ALIASES: Dict[str, str] = {
    "alive": "healthy", "well": "healthy", "fine": "healthy",
    "wounded": "injured", "hurt": "injured", "bleeding": "injured",
    "sick": "ill", "diseased": "ill", "infected": "ill",
    "deceased": "dead", "killed": "dead", "slain": "dead",
    "ko": "unconscious", "knocked_out": "unconscious",
    "asleep": "unconscious", "unconscious": "unconscious",
}


# =====================================================================
# Anchor resolution helper
# =====================================================================

def _isolate_ws_for_surgery(
    ws: WorldStateV1,
    query: "UserRequest",
) -> WorldStateV1:
    """Return a fresh deep-clone of ``ws`` when the query is going to
    invoke ``CausalPhysicsEngine`` typed do-target handlers.

    The Rung-2 / Rung-3 ``_apply_do_*`` methods on
    :class:`CausalPhysicsEngine` mutate ``world_state`` in-place
    (writing onto ``events``, ``channels``, ``social_topology``,
    ``spatial_topology``, ``causal_topology``, and individual
    ``Concern`` / ``Channel`` / ``RelationshipMetric`` objects) so
    that downstream brief construction sees the surgery. Those fields
    are shared by reference between factual and shadow projections
    (``projected_for_branch`` only forks ``entities`` /
    ``objects`` / ``propositions`` / ``world_traits``), so a single
    counterfactual query was permanently corrupting the user's
    factual world.

    This isolation runs *after* :func:`_apply_query_introductions`
    (which already deep-clones when introductions are present, so
    those cases pay one clone, not two) and only for query types
    that can actually fire do-target handlers — observation /
    interrogate / general / evaluate paths are read-only and skip
    the clone.
    """
    if query.query_type not in ("intervention", "counterfactual"):
        return ws
    # P1-FIX: Full state isolation - deep clone ALL shared topology
    return ws.model_copy(deep=True)


def _apply_query_introductions(
    ws: WorldStateV1,
    query: "UserRequest",
    *,
    world_id: str = "factual",
) -> WorldStateV1:
    """Pre-spawn user-side ``query.introduce`` declarations into the
    working world state before physics runs.

    Returns a new :class:`WorldStateV1` (caller should re-bind their
    variable). The pre-spawned nodes carry ``world_id`` from the
    active VWM branch (defaults to ``"factual"`` for legacy callers)
    and are written directly into the typed registries, so
    do-surgeries that target them resolve correctly during
    :func:`calculate_narrative_physics`. The same payload is passed
    through to the renderer's ``GeneratedScene.introduced_elements``
    by the merge step so the auditor and re-extraction see the
    declarations once, not twice.

    No-op when ``query.introduce`` is ``None`` or empty.
    """
    introduced = getattr(query, "introduce", None)
    if introduced is None or introduced.is_empty():
        return ws
    spawns = introduced_elements_to_spawns(
        introduced, ws,
        world_id=world_id,  # type: ignore[arg-type]
    )
    if not any(spawns.get(k) for k in (
        "entities", "objects", "locations", "world_traits",
        "channels", "propositions", "concerns", "events",
    )):
        return ws
    new_ws = ws.model_copy(deep=True)
    for nid, ent in spawns.get("entities", {}).items():
        new_ws.entities.setdefault(nid, ent)
    for nid, obj in spawns.get("objects", {}).items():
        new_ws.objects.setdefault(nid, obj)
    for nid, loc in spawns.get("locations", {}).items():
        new_ws.locations.setdefault(nid, loc)
    for nid, wt in spawns.get("world_traits", {}).items():
        new_ws.world_traits.setdefault(nid, wt)
    if spawns.get("channels"):
        if new_ws.channels is None:
            new_ws.channels = {}
        for nid, ch in spawns["channels"].items():
            new_ws.channels.setdefault(nid, ch)
    if spawns.get("propositions"):
        existing_prop_ids = {p.proposition_id for p in new_ws.propositions}
        for nid, prop in spawns["propositions"].items():
            if nid not in existing_prop_ids:
                new_ws.propositions.append(prop)
    for holder_id, clist in spawns.get("concerns", {}).items():
        if holder_id in new_ws.entities:
            holder = new_ws.entities[holder_id]
            existing_ccn_ids = {c.concern_id for c in holder.concerns}
            for c in clist:
                if c.concern_id not in existing_ccn_ids:
                    holder.concerns.append(c)
    if spawns.get("events"):
        existing_event_ids = {e.id for e in (new_ws.events or [])}
        for nid, evt in spawns["events"].items():
            if nid not in existing_event_ids:
                new_ws.events.append(evt)
    logger.info(
        "[Pipeline] query.introduce pre-spawn \u2014 +%d entities, +%d objects, "
        "+%d locations, +%d world_traits, +%d channels, +%d propositions, "
        "+%d concern-attachments, +%d events.",
        len(spawns.get("entities", {})), len(spawns.get("objects", {})),
        len(spawns.get("locations", {})), len(spawns.get("world_traits", {})),
        len(spawns.get("channels", {})), len(spawns.get("propositions", {})),
        sum(len(v) for v in spawns.get("concerns", {}).values()),
        len(spawns.get("events", {})),
    )
    return new_ws


def _resolve_query_anchors(
    query: UserRequest,
    cfg_temporal: Optional[int],
    cfg_syuzhet: Optional[int],
    world_state: WorldStateV1,
) -> tuple[Optional[int], Optional[int]]:
    """Resolve effective ``(temporal_anchor, syuzhet_anchor)`` for *query*.

    Precedence (highest first):
      1. Explicit ``query.temporal_anchor`` / ``query.syuzhet_anchor``.
      2. ``query.anchor_after_event_id`` resolved against
         ``world_state.events`` — fills whichever anchor wasn't set
         explicitly with the matching event's ``fabula_time`` /
         ``syuzhet_index``.
      3. The pipeline-level ``cfg_temporal`` / ``cfg_syuzhet`` defaults.

    A non-existent ``anchor_after_event_id`` is logged as a warning and
    treated as if it were absent (the query still runs against the
    config defaults rather than failing outright).
    """
    temporal = query.temporal_anchor if query.temporal_anchor is not None else None
    syuzhet = query.syuzhet_anchor if query.syuzhet_anchor is not None else None

    after_id = query.anchor_after_event_id
    if after_id and (temporal is None or syuzhet is None):
        evt = next((e for e in world_state.events if e.id == after_id), None)
        if evt is None:
            logger.warning(
                "[Pipeline] anchor_after_event_id=%r not found in world "
                "state; falling back to PipelineConfig anchors.", after_id,
            )
        else:
            if temporal is None:
                temporal = evt.fabula_time
            if syuzhet is None:
                syuzhet = getattr(evt, "syuzhet_index", None)

    if temporal is None:
        temporal = cfg_temporal
    if syuzhet is None:
        syuzhet = cfg_syuzhet
    return temporal, syuzhet


# =====================================================================
# Configuration
# =====================================================================

class PipelineConfig(BaseModel):
    """Unified configuration for all pipeline stages."""

    # --- Physics ---
    use_causal_engine: bool = Field(
        default=True,
        description="Enable the Rung 2/3 causal physics engine.",
    )
    temporal_anchor: Optional[int] = Field(
        default=None,
        description="Fabula-time horizon for the physics engine.",
    )
    syuzhet_anchor: Optional[int] = Field(
        default=None,
        description="Syuzhet-index horizon for affective calculus.",
    )

    # --- Generation ---
    generation_config: Optional[GenerationConfig] = Field(
        default=None,
        description="LLM configuration for prose generation.",
    )

    # --- Audit ---
    auditor_config: Optional[AuditorConfig] = Field(
        default=None,
        description="LLM configuration for the audit/refinement loop.",
    )
    skip_audit: bool = Field(
        default=False,
        description="Skip the audit loop (Step 11–12). Useful for fast iteration.",
    )

    # --- Re-extraction + Merge ---
    extraction_config: Optional[ExtractionConfig] = Field(
        default=None,
        description="LLM configuration for prose → topology re-extraction.",
    )
    skip_reextraction: bool = Field(
        default=False,
        description="Skip prose re-extraction and world-model merge (Step 6–7).",
    )
    merge_on_audit_failure: bool = Field(
        default=False,
        description=(
            "Allow merge (Steps 6–7) to proceed even when the auditor did "
            "not converge or failed-open. Default False — a failed/non-"
            "converged audit blocks the merge so unaudited prose cannot "
            "pollute the canonical world model. Set True only for "
            "explicit unsafe / debug flows."
        ),
    )
    defer_reextraction: bool = Field(
        default=False,
        description=(
            "Return immediately after audit with prose ready, deferring "
            "Steps 6–7 (re-extraction + merge) so the caller can show the "
            "prose to the user and run the heavy ingestion in the "
            "background. The pipeline stashes a closure on the result; "
            "call ``finish_reextraction(result)`` later to drive it. Mutually "
            "exclusive with ``skip_reextraction`` (skip wins). "
            "**Sync-only**: ``run_pipeline_async`` rejects this flag with "
            "``ValueError`` because no async ``finish_reextraction`` parallel "
            "exists yet."
        ),
    )

    # --- Ingestion ---
    ingestion_config: Optional[ExtractionConfig] = Field(
        default=None,
        description="LLM configuration for raw-text ingestion (Step 1). "
        "Only used when ``raw_text`` is provided instead of a WorldStateV1.",
    )

    # --- World-model versioning ---
    max_snapshots: int = Field(
        default=10,
        description="Maximum number of full world-state snapshots to retain "
        "in the VersionedWorldModel history (last-K).",
    )

    # --- Branch policy (Story-integration plan, Step 1) ---
    branch_policy: Literal["auto", "mainline", "shadow"] = Field(
        default="auto",
        description=(
            "Where the merged version lands in the AMWN version DAG. "
            "'auto' (default) routes counterfactual queries onto a shadow "
            "fork (world_id='shadow') and every other generative query "
            "onto the factual mainline. 'mainline' forces the merge onto "
            "the factual branch regardless of query type (use with care: "
            "promotes a counterfactual into canon). 'shadow' forces a "
            "shadow fork even for observation/intervention/directive "
            "queries (useful for 'what would it look like if we wrote "
            "the next scene this way' explorations)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_from_settings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k, v in _get_settings().pipeline_config_kwargs().items():
                data.setdefault(k, v)
        return data


# =====================================================================
# Pipeline step records
# =====================================================================

class PhysicsStepRecord(BaseModel):
    """Snapshot of the narrative-physics result."""
    query_type: str
    status: str
    physics_state: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query-type-specific extras (mutations, hidden_deltas, etc.).",
    )


class GenerationStepRecord(BaseModel):
    """Snapshot of a single generation pass."""
    scene: GeneratedScene
    brief: Optional[CreativeBrief] = None


class AuditStepRecord(BaseModel):
    """Snapshot of the full audit/refinement loop."""
    feedback_result: FeedbackLoopResult


class ReextractionStepRecord(BaseModel):
    """Snapshot of the prose → topology extraction + merge step."""
    events_added: int = 0
    causal_edges_added: int = 0
    entity_updates_applied: int = 0
    entity_updates_skipped: List[str] = Field(default_factory=list)
    new_version: int = 0


class IngestionStepRecord(BaseModel):
    """Snapshot of the raw-text ingestion step."""
    is_valid: bool = True
    num_events: int = 0
    num_entities: int = 0
    num_locations: int = 0


class PipelineHistory(BaseModel):
    """Ordered record of every step executed in a pipeline run."""
    steps: List[Dict[str, Any]] = Field(default_factory=list)

    def record(self, step_name: str, data: BaseModel | Dict[str, Any]) -> None:
        payload = data.model_dump() if isinstance(data, BaseModel) else data
        self.steps.append({"step": step_name, **payload})


# =====================================================================
# Pipeline result
# =====================================================================

class PipelineResult(BaseModel):
    """Everything the pipeline produces in a single run."""

    # --- The answer ---
    prose: Optional[str] = Field(
        default=None,
        description="The final generated prose (None for non-prose queries).",
    )
    scene: Optional[GeneratedScene] = Field(
        default=None,
        description="The full GeneratedScene object (None for non-prose queries).",
    )

    # --- Physics output ---
    physics_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw output of calculate_narrative_physics.",
    )

    # --- Audit ---
    converged: Optional[bool] = Field(
        default=None,
        description="Whether the audit loop converged (None if skipped).",
    )
    audit_iterations: Optional[int] = Field(
        default=None,
        description="Number of audit iterations run (None if skipped).",
    )
    feedback_result: Optional[FeedbackLoopResult] = Field(
        default=None,
        description="Full audit loop result (None if skipped/not applicable).",
    )

    # --- Evaluation ---
    evaluation_result: Optional[EvaluationResult] = Field(
        default=None,
        description="Full-story evaluation result (None if not an evaluate query).",
    )

    # --- World model ---
    world_model: Optional[VersionedWorldModel] = Field(
        default=None,
        description="The versioned world model after merge (or initial if no merge).",
    )

    # --- History ---
    history: PipelineHistory = Field(
        default_factory=PipelineHistory,
        description="Ordered record of every pipeline step executed.",
    )

    # --- Query metadata ---
    query_type: str = ""

    # --- Plausibility ---
    implausible: bool = Field(
        default=False,
        description="True when the query (typically Rung 2/3 or directive) "
        "could not be applied because its targets do not resolve against "
        "the current world state.  When True, the world model is left "
        "unchanged and ``prose`` contains a human-readable explanation.",
    )
    implausibility_reason: Optional[str] = Field(
        default=None,
        description="Short reason why the query was deemed implausible.",
    )
    implausibility_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured diagnostics: unresolved_targets list, etc.",
    )

    # --- Re-extraction status ---
    reextraction_failed: bool = Field(
        default=False,
        description=(
            "True when prose generation succeeded but Step 6\u20137 "
            "(re-extraction + merge into the world model) raised an "
            "exception.  When True, ``prose`` is present but "
            "``world_model`` is the *unmerged* prior version \u2014 callers "
            "MUST NOT persist this as a new canonical version, since prose "
            "and world state are out of sync."
        ),
    )
    reextraction_error: Optional[str] = Field(
        default=None,
        description="Short error message when reextraction_failed is True.",
    )
    reextraction_pending: bool = Field(
        default=False,
        description=(
            "True when ``cfg.defer_reextraction`` was set and the pipeline "
            "returned with prose ready but Steps 6–7 not yet executed. The "
            "caller MUST invoke ``finish_reextraction(result)`` (typically in "
            "a background task) to perform the merge before persisting the "
            "new version. While pending, ``world_model`` carries the "
            "*pre-merge* model."
        ),
    )

    # --- Continuation quality bridge (rung-1/2/3 + directive +
    # manual-edit re-extraction) ---
    continuation_quality_report: Optional[Any] = Field(
        default=None,
        description=(
            "ValidationReport produced by the continuation-quality "
            "bridge after re-extraction + merge. ``None`` when the "
            "bridge did not run (skip_reextraction, generation "
            "failure, or reextraction exception). ``is_valid=False`` "
            "means structural errors persist after auto-repair + the "
            "correction loop; the version is still persisted (current "
            "behaviour) but ``continuation_quarantined`` is set so "
            "downstream UIs/MCP can flag it."
        ),
    )
    continuation_quarantined: bool = Field(
        default=False,
        description=(
            "True when the continuation-quality bridge could not bring "
            "the merged world to a clean state. The world model is "
            "still returned (parity with current behaviour) but its "
            "VersionRow is tagged with a ``_quarantined`` source so "
            "operators can filter it out. When False, the merged world "
            "passed structural validation."
        ),
    )

    # Private closure that finishes Steps 6\u20137 when the pipeline ran in
    # ``defer_reextraction`` mode. ``finish_reextraction(result)`` invokes
    # it; UI / MCP wrappers should never touch this attr directly.
    _deferred_reextraction_fn: Optional[Any] = PrivateAttr(default=None)
    # P6 (2026-05-29 ninth-pass audit): serialize finish_reextraction so
    # two concurrent callers can't both observe ``reextraction_pending=True``
    # and double-invoke the closure (which would re-merge the deferred
    # ingestion onto the world model twice). A per-result re-entrant lock
    # keeps the pending-check + closure-invoke atomic.
    _finish_lock: Any = PrivateAttr(default_factory=threading.RLock)


# =====================================================================
# Deferred re-extraction entry point
# =====================================================================

def finish_reextraction(result: "PipelineResult") -> "PipelineResult":
    """Drive the deferred Steps 6–7 (prose re-extraction + merge).

    When ``run_pipeline`` was called with ``cfg.defer_reextraction=True``,
    it returns immediately after audit with ``reextraction_pending=True``
    and stashes a closure capturing the locals needed for the merge.
    Call this from a background task to perform the heavy ingestion;
    on return, ``result.world_model`` reflects the merged version and
    ``result.reextraction_pending`` is False.

    Idempotent: a second call after completion is a no-op. Safe to call
    on a result that was never deferred (returns unchanged). Thread-safe:
    concurrent callers serialize on ``result._finish_lock`` so the closure
    fires at most once even under contention.
    """
    if not result.reextraction_pending:
        return result
    with result._finish_lock:
        # Re-check inside the lock: another caller may have completed
        # the merge while we were waiting on it.
        if not result.reextraction_pending:
            return result
        fn = result._deferred_reextraction_fn
        if fn is None:
            # Pending but no closure — someone deferred without setting up
            # the closure. Treat as a programming error but stay resilient.
            result.reextraction_pending = False
            result.reextraction_failed = True
            result.reextraction_error = (
                "finish_reextraction called but no deferred closure was stashed."
            )
            return result
        try:
            fn()
        finally:
            result._deferred_reextraction_fn = None
            result.reextraction_pending = False
        return result


# =====================================================================
# Lay-user summary
# =====================================================================

def humanize_pipeline_result(
    result: "PipelineResult",
    *,
    requested_effect: str | None = None,
    requested_intensity: float | None = None,
) -> str:
    """Render a ``PipelineResult`` as plain-English bullet lines.

    Designed for end-users (chat UI, MCP envelopes, CLI logs) — avoids
    jargon, shows achieved-vs-target affective intensity, and surfaces
    deterministic engine threshold failures with friendly labels.

    Parameters
    ----------
    result
        The pipeline run to summarise.
    requested_effect
        The directive's ``target_effect`` (when known) so achieved
        intensity can be compared against the user's ask.
    requested_intensity
        The directive's ``intensity`` (0-1) so the gap to target can be
        reported.
    """
    lines: list[str] = []

    qt = result.query_type or "query"
    lines.append(f"Ran a {qt} query.")

    if result.implausible:
        reason = result.implausibility_reason or "unspecified"
        lines.append(f"⚠ The request couldn't be applied: {reason}.")
        unresolved = (result.implausibility_details or {}).get("unresolved_targets", [])
        if unresolved:
            for u in unresolved[:3]:
                lines.append(
                    f"  • {u.get('target','?')}: {u.get('reason','?')}"
                )

    if result.prose:
        lines.append(f"Generated {len(result.prose):,} characters of prose.")

    if result.reextraction_failed:
        lines.append(
            "⚠ Prose was generated but the world model couldn't be updated "
            "from it — treat the new version as a draft only."
        )
        if result.reextraction_error:
            lines.append(f"  Reason: {result.reextraction_error}")

    if result.converged is not None:
        status = "passed audit" if result.converged else "did not pass audit"
        lines.append(
            f"Audit: {status} after {result.audit_iterations or 0} "
            f"refinement {'pass' if (result.audit_iterations or 0) == 1 else 'passes'}."
        )

    fb = result.feedback_result
    if fb is not None:
        # Engine threshold gate (deterministic, separate from LLM auditor).
        if fb.engine_thresholds_passed is True:
            lines.append("✓ All quality thresholds met (foreshadowing, plausibility, affective fit).")
        elif fb.engine_thresholds_passed is False:
            lines.append("⚠ Some quality thresholds were not met:")
            for f in fb.engine_threshold_failures[:5]:
                lines.append(f"  • {_friendly_threshold(f)}")

        # Achieved intensity per emotional effect.
        ci = fb.change_impact
        if ci is not None and ci.affective_feedback is not None:
            af = ci.affective_feedback
            scores = af.emotional_trajectory_scores or {}
            if requested_effect and requested_effect in scores:
                achieved = scores[requested_effect]
                if requested_intensity is not None:
                    gap = achieved - requested_intensity
                    arrow = "✓" if abs(gap) <= 0.15 else ("↑" if gap > 0 else "↓")
                    lines.append(
                        f"{arrow} {requested_effect.replace('_',' ').title()}: "
                        f"asked for {requested_intensity:.2f}, achieved "
                        f"{achieved:.2f} (gap {gap:+.2f})."
                    )
                else:
                    lines.append(
                        f"{requested_effect.replace('_',' ').title()} "
                        f"intensity achieved: {achieved:.2f}."
                    )
            elif scores:
                top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
                pretty = ", ".join(f"{k.replace('_',' ')} {v:.2f}" for k, v in top)
                lines.append(f"Strongest emotional effects: {pretty}.")

            if af.affective_loss_mse is not None and af.affective_loss_mse > 0:
                lines.append(
                    f"Distance from requested feeling: "
                    f"{af.affective_loss_mse:.2f} (lower is better)."
                )

        if ci is not None and ci.causal_feedback is not None:
            cf = ci.causal_feedback
            bits: list[str] = []
            bits.append(f"foreshadowing {cf.foreshadowing_payoff_score:.2f}")
            bits.append(f"plausibility {cf.cognitive_plausibility_score:.2f}")
            miracles = len(cf.miracle_steps_detected or [])
            if miracles:
                bits.append(f"{miracles} unexplained jump{'s' if miracles != 1 else ''}")
            lines.append("Causal quality — " + ", ".join(bits) + ".")

    if result.world_model is not None:
        lines.append(f"World model is now at version {result.world_model.version}.")

    return "\n".join(lines) if lines else "Done."


_FRIENDLY_THRESHOLD_LABELS = {
    "foreshadowing_payoff_score": "foreshadowing pay-off was below the minimum",
    "cognitive_plausibility_score": "a character acted against their own established beliefs",
    "affective_loss_mse": "the scene's emotional fit was further from the target than allowed",
    "miracle_steps_detected": "the scene contained an unexplained leap the engine couldn't justify",
}


def _friendly_threshold(failure: str) -> str:
    """Translate an engine threshold failure string into lay English."""
    for key, label in _FRIENDLY_THRESHOLD_LABELS.items():
        if failure.startswith(key):
            return f"{label} ({failure})"
    return failure


def _resolve_branch_policy(
    query: "UserRequest",
    cfg: "PipelineConfig",
    vwm: Optional[VersionedWorldModel] = None,
) -> tuple[Literal["factual", "shadow"], Optional[str]]:
    """Map ``cfg.branch_policy`` + ``query.query_type`` onto the AMWN
    ``world_id`` and ``branch_label`` to attach to the merged version.

    Policy table (Story-integration plan, Step 1):
      * ``"auto"`` (default): counterfactual queries fork to shadow;
        every other query type *inherits* the active branch from
        ``vwm.history[-1]`` so manual edits / observations made while
        exploring a shadow fork stay on that fork rather than
        silently landing back on factual canon. Falls back to
        factual when ``vwm`` is None or has no history.
      * ``"mainline"``: force factual mainline regardless of query type.
      * ``"shadow"``: force a shadow fork regardless of query type.

    ``branch_label`` is derived from ``query.description`` /
    ``original_query`` when forking onto a new shadow branch, or
    inherited from the active version's label when continuing an
    existing fork under auto policy.
    """
    policy = cfg.branch_policy
    if policy == "shadow":
        world_id: Literal["factual", "shadow"] = "shadow"
    elif policy == "mainline":
        world_id = "factual"
    else:  # auto
        if query.query_type == "counterfactual":
            world_id = "shadow"
        elif (
            query.query_type == "intervention"
            and (
                vwm is None
                or not vwm.history
                or vwm.history[-1].world_id != "shadow"
            )
        ):
            # R19-UI-(4): auto-fork shadow for ``do(X)`` interventions
            # *launched from factual mainline* so that surgical
            # "what changes if X" probes don't silently mutate the
            # canonical timeline. A user who runs an intervention on
            # the factual head from the UI expects the result to
            # appear as a new shadow branch they can promote or
            # discard \u2014 matching the affordance the command bar's
            # "Intervene" tooltip implies.
            #
            # When the active head is *already* a shadow branch, we
            # fall through to the inheritance path below so chained
            # interventions iterate within the same scenario instead
            # of spawning a sibling fork on every step. Mainline-
            # intervention from factual is still available explicitly
            # via ``branch_policy="mainline"``.
            world_id = "shadow"
        elif vwm is not None and vwm.history:
            # Inherit the active branch so manual_edit / interrogate /
            # general / observation / intervention / directive queries
            # land on whichever branch the user is currently on. Without
            # this a manual edit made while exploring a shadow fork
            # would be silently re-tagged onto the factual mainline.
            latest = vwm.history[-1]
            world_id = getattr(latest, "world_id", "factual") or "factual"
        else:
            world_id = "factual"

    label: Optional[str] = None
    if world_id == "shadow":
        # Prefer the user's verbatim natural-language request (carried on
        # every query via _QueryBase.original_query); fall back to the
        # ManualEditQuery-only ``description`` field when present.
        candidate = getattr(query, "original_query", None) or getattr(query, "description", None)
        label = (candidate or "").strip() or None
        # When *continuing* on an active shadow branch under auto,
        # inherit its existing label so successive edits don't fragment
        # continuity into per-query mini-branches. A counterfactual
        # query is a deliberate NEW fork even when launched from a
        # shadow head, so it keeps its own query-derived label.
        if (
            policy == "auto"
            and query.query_type != "counterfactual"
            and vwm is not None
            and vwm.history
        ):
            inherited = getattr(vwm.history[-1], "branch_label", None)
            if inherited:
                label = inherited
        # Invariant: a shadow fork *must* have a non-empty branch_label.
        # An empty label silently breaks proposition truth commits at the
        # merge boundary (cross-branch writes are blocked when label is
        # missing) and makes ``projected_for_branch`` a no-op, so the
        # next observation reads factual baseline and contradicts the
        # counterfactual prose. Synthesize a stable label from the query
        # type and the current vwm depth when no natural-language source
        # is available — this is exceptional but possible for queries
        # constructed programmatically (tests, MCP, direct API calls).
        if not label:
            depth = len(vwm.history) if vwm is not None and vwm.history else 0
            label = f"shadow-{query.query_type}-{depth}"
            logger.warning(
                "[branch·policy] Shadow fork requested without an "
                "original_query/description — synthesized branch_label=%r "
                "(query_type=%s). A missing label would silently route "
                "shadow writes to the factual baseline.",
                label, query.query_type,
            )
    return world_id, label


def _stamp_brief_branch(
    brief: Optional["CreativeBrief"],
    world_id: Literal["factual", "shadow"],
    label: Optional[str],
) -> Optional["CreativeBrief"]:
    """Set ``branch_world_id`` / ``branch_label`` on a CreativeBrief in-place.

    Returns the same brief for chained-call ergonomics. No-op when
    ``brief`` is ``None`` (some non-directive paths skip brief assembly
    entirely). The ``factual_contrast_summary`` field is intentionally
    left for the assembler/auditor to populate downstream — this helper
    only stamps the cheap branch identifiers so the generator template
    can switch tense/framing without re-querying the policy.
    """
    if brief is None:
        return None
    brief.branch_world_id = world_id
    if label is not None:
        brief.branch_label = label
    return brief


# Default budget for the STORY SO FAR section of the renderer prompt.
# Sized to fit comfortably alongside the rest of the brief in a 32k
# context window without dominating it; raise via the helper's
# ``max_chars`` kwarg when running against larger models.
_PRECEDING_PROSE_BUDGET_CHARS = 8000


# =====================================================================
# Continuation Quality Bridge — pipeline glue
# =====================================================================
#
# After ``vwm.merge`` folds re-extracted topology into the parent world,
# pipe the merged world state through the same auto-repair +
# programmatic-validation + LLM-correction loop raw-text ingestion
# uses. Without this bridge, structural regressions introduced by the
# generator (orphan events, dangling ids, retrograde fabula_times,
# dead actors emitting events, …) would silently land on the canonical
# (or shadow) branch.
#
# Honours the user-stated constraint: NO merging into parent — the
# continuation already lives in ``vwm_next.current``, which is a fresh
# WorldStateV1 with ``ancestor_id`` lineage preserved by
# ``VersionedWorldModel.merge``. We only *correct* it in place; the
# version chain is unchanged.
#
# Uses the EXTRACTION model (``cfg.extraction_config.model`` =
# ``EXTRACTION_MODEL`` env var with ``DEFAULT_MODEL`` fallback) — the
# same LLM raw-text ingestion uses, never ``GENERATION_MODEL`` /
# ``AUDITOR_MODEL``.

def _run_continuation_quality_bridge_sync(
    vwm_next: "VersionedWorldModel",
    cfg: "PipelineConfig",
    result: "PipelineResult",
    *,
    log_prefix: str,
) -> "VersionedWorldModel":
    """Validate-and-correct the merged continuation world (sync path).

    Returns a possibly-rewritten ``VersionedWorldModel`` whose
    ``current`` field is the corrected world state. Records the
    quality report on ``result.continuation_quality_report`` and sets
    ``result.continuation_quarantined`` when errors persist.
    """
    if cfg.extraction_config is None:
        return vwm_next
    try:
        corrected_ws, report = validate_and_correct_world_state(
            vwm_next.current,
            cfg.extraction_config,
            log_prefix=log_prefix,
        )
    except Exception:
        logger.exception(
            "%s Continuation quality bridge raised; persisting "
            "un-validated merge result.",
            log_prefix,
        )
        return vwm_next
    result.continuation_quality_report = report
    result.continuation_quarantined = not report.is_valid
    if corrected_ws is vwm_next.current:
        return vwm_next
    return vwm_next.model_copy(update={"current": corrected_ws})


async def _run_continuation_quality_bridge_async(
    vwm_next: "VersionedWorldModel",
    cfg: "PipelineConfig",
    result: "PipelineResult",
    *,
    log_prefix: str,
) -> "VersionedWorldModel":
    """Async sibling of :func:`_run_continuation_quality_bridge_sync`."""
    if cfg.extraction_config is None:
        return vwm_next
    try:
        corrected_ws, report = await validate_and_correct_world_state_async(
            vwm_next.current,
            cfg.extraction_config,
            log_prefix=log_prefix,
        )
    except Exception:
        logger.exception(
            "%s Continuation quality bridge raised; persisting "
            "un-validated merge result.",
            log_prefix,
        )
        return vwm_next
    result.continuation_quality_report = report
    result.continuation_quarantined = not report.is_valid
    if corrected_ws is vwm_next.current:
        return vwm_next
    return vwm_next.model_copy(update={"current": corrected_ws})


def _gather_preceding_prose(
    vwm: Optional[VersionedWorldModel],
    *,
    branch_world_id: Literal["factual", "shadow"],
    branch_label: Optional[str] = None,
    max_chars: int = _PRECEDING_PROSE_BUDGET_CHARS,
) -> Optional[str]:
    """Collect prior-version prose from ``vwm.history`` for narrative continuity.

    Walks the linear in-memory history newest \u2192 oldest, picking
    versions whose ``prose`` is non-empty and whose branch is
    compatible with the current query:

    * ``branch_world_id == "factual"`` \u2014 include only factual-tagged
      prose (canonical mainline only; shadow forks must not bleed into
      factual continuity).
    * ``branch_world_id == "shadow"`` \u2014 include the *most recent
      contiguous tail* of shadow-tagged prose first (the active fork's
      own narrative), then any factual ancestors that came before any
      shadow appears in the walk (the canon the fork branched from).

    Entries are then re-ordered oldest \u2192 newest, joined with version
    markers, and truncated from the *front* to ``max_chars`` so the
    most recent prose always survives. Returns ``None`` when nothing
    is available so callers can splat unconditionally.
    """
    if vwm is None or not getattr(vwm, "history", None):
        return None

    picked: list[tuple[int, str, str, Optional[str]]] = []
    if branch_world_id == "factual":
        for v in vwm.history:
            if v.prose and getattr(v, "world_id", "factual") == "factual":
                picked.append((v.version, v.prose, "factual", v.branch_label))
    else:
        # Round-7 audit: anchor the shadow context to the fork point so
        # factual prose written AFTER the shadow forked can't leak into
        # the shadow prompt. The fork point is approximated as the
        # earliest version of the matching shadow branch_label (or the
        # earliest shadow version overall when no label is given).
        # Without this, any factual prose later in history would be
        # treated as "ancestor canon" and cross-pollinate the branch.
        shadow_versions = [
            v.version for v in vwm.history
            if getattr(v, "world_id", "factual") == "shadow"
            and (branch_label is None or v.branch_label == branch_label)
        ]
        fork_point = min(shadow_versions) if shadow_versions else None
        if fork_point is None:
            logger.warning(
                "[_gather_preceding_prose] No shadow versions found for branch_label=%s; "
                "including all factual prose (appropriate when creating first shadow version).",
                branch_label,
            )

        # Shadow: take the contiguous tail of shadow versions newest
        # \u2192 oldest, stop on the first non-shadow we encounter, then
        # include all factual prose that came before that point so the
        # shadow rendering still sees the canon it branched from.
        history_rev = list(reversed(vwm.history))
        idx = 0
        while idx < len(history_rev):
            v = history_rev[idx]
            if getattr(v, "world_id", "factual") != "shadow":
                break
            if v.prose:
                # Optional branch_label filter \u2014 if the caller
                # specified a label, only include shadow prose that
                # matches it (different forks shouldn't cross-pollinate).
                if branch_label is None or v.branch_label == branch_label:
                    picked.append((v.version, v.prose, "shadow", v.branch_label))
            idx += 1
        # Now collect factual ancestors from the rest of history,
        # gated by the fork-point so post-fork factual advancement is
        # excluded.
        for v in history_rev[idx:]:
            if v.prose and getattr(v, "world_id", "factual") == "factual":
                if fork_point is not None and v.version >= fork_point:
                    continue
                picked.append((v.version, v.prose, "factual", v.branch_label))

    if not picked:
        return None

    # Sort oldest \u2192 newest so the joined block reads in narrative order.
    picked.sort(key=lambda t: t[0])

    # Truncate from the front (drop oldest) until the joined block
    # fits the budget. Each block is wrapped with a small marker so
    # the LLM can tell continuity from the active scene's task. The
    # marker is always ``(world_id: label)`` form (factual versions
    # without an explicit label render as ``(factual: mainline)``) so
    # the downstream prompt wrappers in answer.py / generation.py /
    # auditor.py can match a single ``(factual: \u2026)`` / ``(shadow: \u2026)``
    # pattern when telling the LLM which blocks are in force on the
    # active branch.
    blocks: list[str] = []
    for version, prose, wid, label in picked:
        marker_label = label if label else (
            "mainline" if wid == "factual" else "unlabelled"
        )
        marker = f"--- v{version} ({wid}: {marker_label}) ---"
        blocks.append(f"{marker}\n{prose.strip()}")

    joined = "\n\n".join(blocks)
    if len(joined) <= max_chars:
        return joined

    # Drop oldest blocks one at a time until under budget; if even the
    # newest block exceeds the budget, hard-truncate it from the front.
    while len(blocks) > 1 and len(joined) > max_chars:
        blocks.pop(0)
        joined = "\n\n".join(blocks)
    if len(joined) > max_chars:
        joined = "\u2026" + joined[-(max_chars - 1):]
    return joined


def _stamp_brief_continuity(
    brief: Optional["CreativeBrief"],
    vwm: Optional[VersionedWorldModel],
    *,
    branch_world_id: Literal["factual", "shadow"],
    branch_label: Optional[str] = None,
) -> Optional["CreativeBrief"]:
    """Stamp ``preceding_prose`` onto a brief from the lineage in ``vwm``.

    No-op when ``brief`` is ``None`` or when no prose is available.
    Called immediately after :func:`_stamp_brief_branch` at every
    brief-construction site in the pipeline so a chain of queries
    (counterfactual \u2192 intervention \u2192 observation, etc.) renders
    prose that is narratively continuous with everything that came
    before, not just the accumulated world state.
    """
    if brief is None:
        return None
    preceding = _gather_preceding_prose(
        vwm,
        branch_world_id=branch_world_id,
        branch_label=branch_label,
    )
    if preceding:
        brief.preceding_prose = preceding
    return brief


# Default budget for the FACTUAL CONTRAST summary on shadow briefs.
# Smaller than the STORY SO FAR budget because contrast is meant as a
# concise reference point, not a full re-narration of canon.
_FACTUAL_CONTRAST_BUDGET_CHARS = 2000


def _compute_factual_contrast(
    vwm: Optional[VersionedWorldModel],
    *,
    branch_world_id: Literal["factual", "shadow"],
    branch_label: Optional[str] = None,
    max_chars: int = _FACTUAL_CONTRAST_BUDGET_CHARS,
) -> Optional[str]:
    """Return a brief excerpt of the most recent factual prose for shadow briefs.

    Returns ``None`` for factual queries (no contrast needed) and when
    no factual prose exists in ``vwm.history``. The latest factual
    version's prose at-or-before the shadow's fork point is preferred so
    the contrast reflects the canon the shadow forked from, not
    factual prose that was authored AFTER the fork (which would leak
    future canon into the shadow prompt — Round-7 audit).
    """
    if branch_world_id != "shadow":
        return None
    if vwm is None or not getattr(vwm, "history", None):
        return None
    # Round-7 audit: cap factual selection at the fork point.
    shadow_versions = [
        v.version for v in vwm.history
        if getattr(v, "world_id", "factual") == "shadow"
        and (branch_label is None or v.branch_label == branch_label)
    ]
    fork_point = min(shadow_versions) if shadow_versions else None
    for v in reversed(vwm.history):
        if v.prose and getattr(v, "world_id", "factual") == "factual":
            if fork_point is not None and v.version >= fork_point:
                continue
            text = v.prose.strip()
            if len(text) > max_chars:
                text = "\u2026" + text[-(max_chars - 1):]
            return text
    return None


def _inert_aware_auditor_config(
    base_cfg: Optional["AuditorConfig"],
    physics_result: Dict[str, Any],
) -> Optional["AuditorConfig"]:
    """When the engine has marked the intervention inert (all
    propagations absorbed by inertia / cycles / noisy-OR), there is
    nothing for the auditor to chase: any "missing downstream Δ"
    violation would directly contradict the engine. Cap iterations and
    disallow miracle-step injection so we don't burn LLM calls grinding
    against absorbed cascades. The renderer's HARD inert constraint
    still produces the resistance prose; this just stops the refinement
    loop from over-revising it.

    Returns a (possibly modified) AuditorConfig. Pass-through when the
    intervention is *not* inert so the normal audit budget applies.
    """
    if not physics_result:
        return base_cfg
    if not physics_result.get("intervention_inert"):
        return base_cfg
    try:
        from shadow_loom.auditor import AuditorConfig as _AC
    except Exception:
        return base_cfg
    cfg = base_cfg.model_copy(deep=True) if base_cfg is not None else _AC()
    # One render + one audit pass is enough — the inert outcome is a
    # short scene of attempt + resistance and shouldn't be rewritten.
    if cfg.max_iterations > 2:
        cfg.max_iterations = 2
    cfg.regression_retry_budget = 0
    # Block miracle-step injection: by definition every downstream Δ
    # the auditor might want has been absorbed.
    cfg.max_miracle_steps = 0
    return cfg


def _stamp_brief_full(
    brief: Optional["CreativeBrief"],
    vwm: Optional[VersionedWorldModel],
    *,
    branch_world_id: Literal["factual", "shadow"],
    branch_label: Optional[str],
    factual_contrast: Optional[str] = None,
) -> Optional["CreativeBrief"]:
    """One-shot stamping: branch ids + preceding_prose + factual_contrast.

    Replaces the earlier pattern of three separate stampings at every
    brief-construction site so a missed call can no longer drop branch
    framing on a shadow render. ``factual_contrast`` is set only when
    non-empty and the brief does not already carry one.
    """
    if brief is None:
        return None
    _stamp_brief_branch(brief, branch_world_id, branch_label)
    _stamp_brief_continuity(
        brief, vwm,
        branch_world_id=branch_world_id, branch_label=branch_label,
    )
    if factual_contrast and not brief.factual_contrast_summary:
        brief.factual_contrast_summary = factual_contrast
    return brief


def _log_feedback_outcome(
    feedback: "FeedbackLoopResult",
    *,
    async_path: bool,
) -> None:
    """Loudly surface refinement-loop failure modes.

    ``FeedbackLoopResult`` carries three distinct failure signals that
    were previously buried in a debug-string parenthetical
    (``audit=skipped/failed``) or in a result attribute the caller
    would have to introspect manually:

      * ``converged=False`` \u2014 the auditor never passed within
        ``max_iterations``. Final scene is the best-effort last
        iteration, not a clean pass.
      * ``engine_thresholds_passed=False`` \u2014 deterministic
        ChangeImpactMetrics checks failed on the final cycle even if
        the LLM auditor passed. ``engine_threshold_failures`` names
        which gates fell.
      * ``correction_error`` populated \u2014 a refinement/rewrite
        LLM call raised; the loop exited early with whatever scene
        the previous iteration produced.

    Emit one WARNING per signal so log greppers, CI assertions, and
    the UI debug view see them without parsing the PipelineHistory
    or FeedbackLoopResult by hand.
    """
    tag = "Pipeline\u00b7Async" if async_path else "Pipeline"
    if not feedback.converged:
        logger.warning(
            "[%s] Refinement loop did NOT converge after %d iteration(s) "
            "\u2014 returning best-effort last scene. Auditor never passed.",
            tag, feedback.iterations,
        )
    if feedback.engine_thresholds_passed is False:
        logger.warning(
            "[%s] Engine-threshold check FAILED on final audit cycle "
            "(thresholds=%s); ChangeImpactMetrics did not meet the "
            "deterministic gates even if the LLM auditor passed.",
            tag,
            feedback.engine_threshold_failures or ["<unspecified>"],
        )
    if feedback.correction_error:
        logger.warning(
            "[%s] Refinement loop exited via correction_error=%r after "
            "%d iteration(s); scene reflects pre-error state, not a "
            "clean audit pass.",
            tag, feedback.correction_error, feedback.iterations,
        )


def _render_engine_priors(
    physics_result: Dict[str, Any],
    *,
    max_items_per_section: int = 12,
) -> Optional[str]:
    """Compact one-shot summary of deterministic engine output.

    Renders the trait mutations, hidden-deltas (rung-3 abduction),
    social mutations, blocked interventions, and intervened nodes a
    :class:`CausalPhysicsEngine` produced into a short text block the
    Physics / Social extractors can use as ground-truth priors when
    parsing continuation prose. Returns ``None`` when ``physics_result``
    has no actionable engine output (rung-1 / observation, or sandbox
    skipped) so the caller can omit the priors block entirely rather
    than emit an empty section.
    """
    if not physics_result:
        return None

    mutations = physics_result.get("mutations") or []
    hidden = physics_result.get("hidden_deltas") or {}
    social_muts = physics_result.get("social_mutations") or []
    event_muts = physics_result.get("event_mutations") or []
    blocked = physics_result.get("blocked") or []
    intervened = physics_result.get("intervened_nodes") or []
    inert = bool(physics_result.get("intervention_inert"))
    inert_reason = physics_result.get("intervention_inert_reason")
    pruned_utt_ids = physics_result.get("pruned_utterance_event_ids") or []
    disabled_ch_ids = physics_result.get("disabled_channel_ids") or []
    skipped_ints = physics_result.get("skipped_interventions") or []

    sections: List[str] = []

    def _coerce_field(item: Any, name: str) -> Any:
        if isinstance(item, dict):
            return item.get(name)
        return getattr(item, name, None)

    if intervened:
        ids = [str(n) for n in intervened[:max_items_per_section]]
        sections.append(
            "Intervened nodes (do-operator targets): "
            + ", ".join(ids)
            + (
                ""
                if len(intervened) <= max_items_per_section
                else f" (+{len(intervened) - max_items_per_section} more)"
            )
        )

    if mutations:
        lines = ["Trait mutations declared by the engine:"]
        for m in mutations[:max_items_per_section]:
            node = _coerce_field(m, "node_id")
            trait = _coerce_field(m, "trait")
            new_val = _coerce_field(m, "new_value")
            try:
                new_val_str = f"{float(new_val):+.2f}"
            except (TypeError, ValueError):
                new_val_str = str(new_val)
            lines.append(f"  - {node}.{trait} \u2192 {new_val_str}")
        if len(mutations) > max_items_per_section:
            lines.append(
                f"  ... (+{len(mutations) - max_items_per_section} more)"
            )
        sections.append("\n".join(lines))

    if hidden:
        lines = [
            "Hidden deltas (Rung-3 abduction \u2014 backstory commits):"
        ]
        count = 0
        for node, trait_dict in hidden.items():
            if not isinstance(trait_dict, dict):
                continue
            for trait, val in trait_dict.items():
                try:
                    val_str = f"{float(val):+.2f}"
                except (TypeError, ValueError):
                    val_str = str(val)
                lines.append(f"  - {node}.{trait} \u2192 {val_str}")
                count += 1
                if count >= max_items_per_section:
                    break
            if count >= max_items_per_section:
                break
        sections.append("\n".join(lines))

    if event_muts:
        # 2026-05-29 (Reingest priors gap): surface DoEvent surgeries
        # (relocation / time_shift / averted / forced) so the
        # extractors don't re-extract the pre-surgery event location
        # or timing from the prose. Mirrors the deterministic
        # event-mutation bridge applied later in topology.
        lines = [
            "Event mutations declared by the engine (do NOT re-extract "
            "the pre-surgery event location / fabula_time from the prose):"
        ]
        for em in event_muts[:max_items_per_section]:
            ekind = _coerce_field(em, "kind") or "?"
            eid = _coerce_field(em, "event_id") or "?"
            bits: List[str] = [f"{ekind} {eid}"]
            new_loc = _coerce_field(em, "new_at_location_id")
            if new_loc:
                bits.append(f"\u2192 {new_loc}")
            new_ft = _coerce_field(em, "new_fabula_time")
            if new_ft is not None:
                bits.append(f"@t={new_ft}")
            lines.append("  - " + " ".join(bits))
        if len(event_muts) > max_items_per_section:
            lines.append(
                f"  ... (+{len(event_muts) - max_items_per_section} more)"
            )
        sections.append("\n".join(lines))

    if social_muts:
        lines = ["Social mutations declared by the engine:"]
        for s in social_muts[:max_items_per_section]:
            src = (
                _coerce_field(s, "source_entity_id")
                or _coerce_field(s, "source_id")
            )
            tgt = (
                _coerce_field(s, "target_entity_id")
                or _coerce_field(s, "target_id")
            )
            metric = _coerce_field(s, "metric")
            new_val = _coerce_field(s, "new_value")
            try:
                new_val_str = f"{float(new_val):+.2f}"
            except (TypeError, ValueError):
                new_val_str = str(new_val)
            lines.append(
                f"  - {src} \u2192 {tgt} ({metric}) = {new_val_str}"
            )
        if len(social_muts) > max_items_per_section:
            lines.append(
                f"  ... (+{len(social_muts) - max_items_per_section} more)"
            )
        sections.append("\n".join(lines))

    if blocked:
        lines = [
            "Interventions blocked by physics (do NOT extract events "
            "that contradict these blocks):"
        ]
        for b in blocked[:max_items_per_section]:
            node = _coerce_field(b, "node_id")
            reason = (
                _coerce_field(b, "reason")
                or _coerce_field(b, "detail")
                or "blocked"
            )
            lines.append(f"  - {node}: {reason}")
        if len(blocked) > max_items_per_section:
            lines.append(
                f"  ... (+{len(blocked) - max_items_per_section} more)"
            )
        sections.append("\n".join(lines))

    if pruned_utt_ids:
        ids = [str(n) for n in pruned_utt_ids[:max_items_per_section]]
        sections.append(
            "Pruned utterance events (do-surgery removed these "
            "speech-act events; do NOT re-extract them from the prose "
            "even if quoted): " + ", ".join(ids)
            + (
                ""
                if len(pruned_utt_ids) <= max_items_per_section
                else f" (+{len(pruned_utt_ids) - max_items_per_section} more)"
            )
        )

    if disabled_ch_ids:
        ids = [str(n) for n in disabled_ch_ids[:max_items_per_section]]
        sections.append(
            "Disabled channels (do-surgery removed; beliefs acquired "
            "via these channels are no longer supported): "
            + ", ".join(ids)
            + (
                ""
                if len(disabled_ch_ids) <= max_items_per_section
                else f" (+{len(disabled_ch_ids) - max_items_per_section} more)"
            )
        )

    if skipped_ints:
        ids = [str(n) for n in skipped_ints[:max_items_per_section]]
        sections.append(
            "Skipped interventions (engine could not apply \u2014 target "
            "absent from sandbox; do NOT extract any consequence for "
            "these): " + ", ".join(ids)
            + (
                ""
                if len(skipped_ints) <= max_items_per_section
                else f" (+{len(skipped_ints) - max_items_per_section} more)"
            )
        )

    if inert:
        sections.append(
            "INERT INTERVENTION \u2014 the engine produced ZERO downstream "
            "mutations for this surgery"
            + (f": {inert_reason}" if inert_reason else "")
            + ". Do NOT extract any new trait shift, relationship update, "
            "belief change, or world-state delta from the prose attributed "
            "to this intervention."
        )

    if not sections:
        return None
    return "\n\n".join(sections)


def _compute_shadow_prune_closure(
    world_state: WorldStateV1,
    root_event_ids: set[str],
    *,
    cause_disconnected_event_ids: Optional[set[str]] = None,
) -> set[str]:
    """Expand a do-surgery prune set to its causal descendant closure.

    Thin wrapper around
    :func:`shadow_loom.causal_closure.expand_chain_reaction_closure` —
    see that module for the full Pearl / Halpern-Pearl rationale.
    Retained here for call-site stability and to keep the
    ``Optional[set[str]]`` signature the merge-time bridge expects.

    Returns the *expanded* prune set (includes the original roots).
    Safe to call with an empty root set (returns empty).
    """
    if not root_event_ids and not cause_disconnected_event_ids:
        return set()

    parents_of = chain_reaction_parents_from_world_state(world_state)
    return expand_chain_reaction_closure(
        parents_of,
        root_event_ids,
        cause_disconnected_ids=cause_disconnected_event_ids,
    )


def _augment_topology_with_sandbox_deltas(
    topology: "ChunkTopology",
    *,
    world_state: WorldStateV1,
    physics_result: Dict[str, Any],
    fabula_time_now: int,
    fabula_time_historical: Optional[int] = None,
    world_id: Literal["factual", "shadow"] = "factual",
    branch_label: Optional[str] = None,
    query_type: Optional[str] = None,
) -> "ChunkTopology":
    """Inject sandbox-derived state changes directly into the topology so
    the merge step persists physics deltas even if the LLM-generated
    prose did not verbalise them clearly enough for the extractor to
    re-discover.

    Bridged channels:
      * ``physics_result["mutations"]`` (list of TraitMutation dicts) \u2014
        ENT_* nodes become :class:`EntityUpdate` records anchored at
        ``fabula_time_now``; WORLD_* nodes append a
        :class:`WorldTraitSnapshot` to a re-injected
        ``new_world_traits[wid]`` so the existing backfill path picks
        them up.
      * ``physics_result["hidden_deltas"]`` (Rung-3 abduction output) \u2014
        Same shape as ``mutations`` but anchored at
        ``fabula_time_historical`` (the inferred backstory point), or
        ``fabula_time_now`` when no historical anchor is available.
        Trait deltas are converted into absolute trait values by
        adding to the entity's current trait.value.
      * ``physics_result["social_mutations"]`` (list of SocialMutation
        dicts) \u2014 appended to ``topology.social_topology`` as full
        :class:`RelationshipEdge` records so the dedup path keeps the
        most-recent metric value.

    All injected records carry ``world_id`` so shadow merges keep
    branch isolation when the calling pipeline already knows it is
    forking. (The merge itself also re-tags everything, but tagging
    here keeps the topology self-consistent for any inspector that
    reads it before merge runs.)
    """
    from shadow_loom.models import (
        EntityStateSnapshot,
        RelationshipEdge,
        RelationshipMetric,
        TraitVector,
        WorldTraitSnapshot,
    )
    from shadow_loom.ingestion import EntityUpdate, ObjectUpdate

    mutations = physics_result.get("mutations") or []
    hidden = physics_result.get("hidden_deltas") or {}
    social_muts = physics_result.get("social_mutations") or []

    # ------------------------------------------------------------------
    # Helper: merge an EntityUpdate-style trait write into the topology.
    # If an EntityUpdate for (entity_id, fabula_time) already exists,
    # extend its ``trait_updates``; otherwise append a fresh record.
    # ------------------------------------------------------------------
    def _upsert_entity_trait(
        entity_id: str, trait: str, new_value: float, ft: int,
        triggered_by: Optional[str] = None,
    ) -> None:
        ent = world_state.entities.get(entity_id)
        if ent is None:
            return
        existing_tv = ent.traits.get(trait)
        inertia = existing_tv.inertia if existing_tv else 0.3
        # ``TraitVector.value`` is constrained to ``[0.0, 1.0]`` on the
        # schema (``models.py`` L196). The previous ``[-1.0, 1.0]``
        # clamp was a copy of the relationship-axis bound and let a
        # negative ``base + delta_f`` slip through, which then crashed
        # ``TraitVector(...)`` with a ``ge=0`` ValidationError during
        # re-extraction merge (audit 2026-05-26).
        tv = TraitVector(value=float(max(0.0, min(1.0, new_value))), inertia=inertia)
        for eu in topology.entity_updates:
            if eu.entity_id == entity_id and eu.fabula_time == ft:
                if trait not in eu.trait_updates:
                    eu.trait_updates[trait] = tv
                # Preserve first-seen provenance; only upgrade if the
                # existing record was unattributed (triggered_by=None).
                if triggered_by and not getattr(eu, "triggered_by", None):
                    eu.triggered_by = triggered_by
                return
        topology.entity_updates.append(EntityUpdate(
            entity_id=entity_id,
            fabula_time=ft,
            triggered_by=triggered_by,
            trait_updates={trait: tv},
        ))

    # ------------------------------------------------------------------
    # Helper: append a WorldTraitSnapshot to a working copy of an
    # existing GlobalTrait, then route that copy through
    # ``new_world_traits`` so ``_backfill_world_trait`` merges the
    # new timeline entry into the canonical record.
    # ------------------------------------------------------------------
    def _upsert_world_trait_snapshot(
        wid: str, trait: str, new_value: float, ft: int,
        triggered_by: Optional[str] = None,
    ) -> None:
        # Prefer an entry already accumulated in this topology so
        # multiple deltas for the same WORLD_ trait stack rather than
        # overwriting each other; fall back to the canonical record
        # in world_state when this is the first delta for ``wid``.
        base_wt = topology.new_world_traits.get(wid) or world_state.world_traits.get(wid)
        if base_wt is None:
            return
        inertia = base_wt.magnitude.inertia if base_wt.magnitude else 0.3
        snap = WorldTraitSnapshot(
            world_id=world_id,
            fabula_time=ft,
            triggered_by=triggered_by,
            magnitude=TraitVector(
                value=float(max(0.0, min(1.0, new_value))),
                inertia=inertia,
            ),
        )
        # Skip duplicate snapshots at the same (fabula_time, world_id,
        # triggered_by) so per-chunk Consequences-fold snapshots (already
        # appended to ``base_wt.state_timeline`` during
        # ``assemble_world_state``) are not re-stamped by post-assembly
        # engine cycles writing through this helper. Including
        # ``triggered_by`` lets two genuinely distinct events at the
        # same tick still both land.
        existing_keys = {
            (s.fabula_time, getattr(s, "world_id", "factual"), getattr(s, "triggered_by", None))
            for s in base_wt.state_timeline
        }
        if (snap.fabula_time, snap.world_id, snap.triggered_by) in existing_keys:
            return
        wt_copy = base_wt.model_copy(update={
            "state_timeline": list(base_wt.state_timeline) + [snap],
            "world_id": world_id,
        })
        topology.new_world_traits[wid] = wt_copy

    # --- Trait mutations (Rung 2 + propagation) ------------------------
    for m in mutations:
        node_id = m.get("node_id") if isinstance(m, dict) else getattr(m, "node_id", None)
        trait = m.get("trait") if isinstance(m, dict) else getattr(m, "trait", None)
        new_val = m.get("new_value") if isinstance(m, dict) else getattr(m, "new_value", None)
        trig = m.get("triggered_by") if isinstance(m, dict) else getattr(m, "triggered_by", None)
        if not node_id or not trait or new_val is None:
            continue
        if node_id.startswith("WORLD_"):
            _upsert_world_trait_snapshot(node_id, trait, float(new_val), fabula_time_now, triggered_by=trig)
        else:
            _upsert_entity_trait(node_id, trait, float(new_val), fabula_time_now, triggered_by=trig)

    # --- Hidden deltas (Rung 3 abduction) ------------------------------
    historical_anchor = (
        fabula_time_historical if fabula_time_historical is not None else fabula_time_now
    )
    for entity_id, deltas in hidden.items():
        ent = world_state.entities.get(entity_id)
        if ent is None and not entity_id.startswith("WORLD_"):
            continue
        for trait, delta in (deltas or {}).items():
            try:
                delta_f = float(delta)
            except (TypeError, ValueError):
                continue
            if entity_id.startswith("WORLD_"):
                # Use the most-recent stacked snapshot for this trait
                # (if any) as the abduction baseline so multiple hidden
                # deltas + mutations compose, rather than always
                # rebasing off the canonical magnitude.
                stacked = topology.new_world_traits.get(entity_id)
                wt = stacked or world_state.world_traits.get(entity_id)
                if wt is None or wt.magnitude is None:
                    continue
                base = (
                    stacked.state_timeline[-1].magnitude.value
                    if stacked and stacked.state_timeline
                    and stacked.state_timeline[-1].magnitude is not None
                    else wt.magnitude.value
                )
                _upsert_world_trait_snapshot(
                    entity_id, trait, base + delta_f, historical_anchor,
                )
            else:
                tv = ent.traits.get(trait)
                base = tv.value if tv else 0.0
                _upsert_entity_trait(
                    entity_id, trait, base + delta_f, historical_anchor,
                )

    # --- Social mutations (relationship axes) --------------------------
    for sm in social_muts:
        if isinstance(sm, dict):
            src = sm.get("source_entity_id")
            tgt = sm.get("target_entity_id")
            metric = sm.get("metric")
            new_val = sm.get("new_value")
            old_val = sm.get("old_value")
            inertia = sm.get("inertia", 0.3)
            triggered_by = sm.get("triggered_by")
        else:
            src = getattr(sm, "source_entity_id", None)
            tgt = getattr(sm, "target_entity_id", None)
            metric = getattr(sm, "metric", None)
            new_val = getattr(sm, "new_value", None)
            old_val = getattr(sm, "old_value", None)
            inertia = getattr(sm, "inertia", 0.3)
            triggered_by = getattr(sm, "triggered_by", None)
        if not src or not tgt or not metric or new_val is None:
            continue
        if metric not in ("affinity", "fear", "power_dynamic"):
            continue
        rm = RelationshipMetric(
            value=float(new_val),
            inertia=float(inertia),
            evidence_strength="moderate",
            # Anchor at the current fabula horizon. The dedup pass
            # (``_deduplicate_social``) uses ``>=`` on ties so this
            # bridged sandbox value wins against any same-tick metric
            # the prose extractor emitted for the same dyad/axis,
            # because the bridge runs *after* extraction and is
            # therefore appended later in ``social_topology``.
            last_updated_fabula=fabula_time_now,
            observed=True,
        )
        topology.social_topology.append(RelationshipEdge(
            source_entity_id=src,
            target_entity_id=tgt,
            metrics={metric: rm},
            world_id=world_id,
        ))
        # --- Causal-explainability twin: emit a mutation_social
        # CausalEdge alongside the relationship snapshot so the social
        # delta has the same provenance edge in causal_topology that a
        # prose-extracted social mutation would. Without this the next
        # query against the merged world model can see the relationship
        # value moved but cannot trace *why*, breaking the auditor's
        # foreshadowing-payoff and miracle-step checks.
        if triggered_by:
            try:
                from shadow_loom.models import CausalEdge
                delta = (
                    float(new_val) - float(old_val)
                    if old_val is not None
                    else 0.0
                )
                topology.causal_topology.append(CausalEdge(
                    source_id=triggered_by,
                    target_id=src,
                    causality_type="mutation_social",
                    causal_force=abs(delta) * 10.0 if delta else 1.0,
                    mechanism="social",
                    evidence_strength="moderate",
                    fabula_time=fabula_time_now,
                    trait_target=metric,
                    trait_delta=delta,
                    rel_counterpart_id=tgt,
                    world_id=world_id,
                ))
            except Exception as e:  # noqa: BLE001
                # Causal-edge construction is best-effort: a malformed
                # triggered_by id (e.g. not an EVT_) would fail the
                # mutation_social validator. Don't lose the
                # relationship snapshot just because we can't stamp
                # the provenance edge.
                # HIGH-FIX: Upgrade to warning so bridge failures are visible
                logger.warning(
                    "[Bridge] Could not emit mutation_social CausalEdge "
                    "for %s->%s.%s (triggered_by=%s): %s",
                    src, tgt, metric, triggered_by, str(e),
                )

    # --- Channel severance (do-surgery on standing capabilities) -------
    # Physics emits ``disabled_channel_ids`` whenever an intervention or
    # counterfactual cuts a communication channel (line tapped & cut,
    # cipher broken & abandoned, courier killed, bond severed). The
    # render path already surfaces these via HARD ConstraintBlocks, but
    # without bridging them into the topology the merge step never
    # touches the canonical Channel record \u2014 so the *next* query against
    # the same world model still sees the channel as live and dialogue
    # can route through it again. Stamp ``terminated_at_fabula`` on a
    # copy and append it to ``topology.channels``; the channel-dedup
    # path (``_deduplicate_channels_with_map``) collapses the copy with
    # the canonical record and picks the earliest non-null termination.
    disabled_chs = physics_result.get("disabled_channel_ids") or []
    for cid in disabled_chs:
        ch = (world_state.channels or {}).get(cid)
        if ch is None:
            continue
        existing_term = ch.terminated_at_fabula
        # Don't bump a channel that was already severed earlier.
        if existing_term is not None and existing_term <= fabula_time_now:
            continue
        topology.channels[cid] = ch.model_copy(update={
            "terminated_at_fabula": int(fabula_time_now),
            "world_id": world_id,
        })

    # --- Pruned utterances \u2192 belief invalidation cascade --------------
    # Physics' ``pruned_utterance_event_ids`` enumerates utterances
    # the do-surgery rendered epistemically inert. The sandbox
    # surgery already prunes the *sandbox* beliefs via
    # ``Belief.acquired_via_event_id``, but those changes never reach
    # the persisted model unless the bridge translates them into
    # belief-invalidation snapshots. Without this, the next query
    # against the merged world still sees characters confidently
    # holding beliefs they only acquired from a now-erased utterance.
    pruned_utts = set(physics_result.get("pruned_utterance_event_ids") or [])
    if pruned_utts or disabled_chs:
        disabled_set = set(disabled_chs)
        for entity_id, ent in (world_state.entities or {}).items():
            invalidated_targets: List[str] = []
            for b in ent.beliefs:
                via_evt = getattr(b, "acquired_via_event_id", None)
                via_chn = getattr(b, "acquired_via_channel_id", None)
                if (via_evt and via_evt in pruned_utts) or (
                    via_chn and via_chn in disabled_set
                ):
                    invalidated_targets.append(b.target_id)
            if not invalidated_targets:
                continue
            # Reuse / extend an EntityUpdate at this fabula tick if
            # one already exists, so we don't pile up redundant
            # snapshots when the bridge wires multiple deltas for
            # the same entity at the same horizon.
            existing = next(
                (
                    eu for eu in topology.entity_updates
                    if eu.entity_id == entity_id
                    and eu.fabula_time == fabula_time_now
                ),
                None,
            )
            if existing is not None:
                merged_inv = list(dict.fromkeys(
                    list(existing.invalidated_belief_targets) + invalidated_targets
                ))
                existing.invalidated_belief_targets = merged_inv
            else:
                topology.entity_updates.append(EntityUpdate(
                    entity_id=entity_id,
                    fabula_time=fabula_time_now,
                    triggered_by=None,
                    invalidated_belief_targets=invalidated_targets,
                ))

    # --- Typed Pearl Rung-2/3 surgeries (proposition / belief / concern) ---
    # CausalPhysicsResult emits these directly; without bridging into
    # the topology the merge step would never persist clamps applied by
    # ``apply_do_targets`` to the proposition / belief / concern
    # substrate. See _apply_affect_to_world / _apply_belief_confidence_updates
    # in extract_graph.py for the merge-side handlers.
    from shadow_loom.ingestion import (
        BeliefConfidenceUpdate,
        ChunkConcernSnapshot,
        PropositionTruthCommit,
    )
    from shadow_loom.models import Belief

    # Proposition mutations → truth commits.
    for pm in physics_result.get("proposition_mutations") or []:
        pid = pm.get("proposition_id") if isinstance(pm, dict) else getattr(pm, "proposition_id", None)
        ftt = pm.get("fabula_time") if isinstance(pm, dict) else getattr(pm, "fabula_time", None)
        new_truth = pm.get("new_truth") if isinstance(pm, dict) else getattr(pm, "new_truth", None)
        if not pid or ftt is None or new_truth is None:
            continue
        topology.proposition_truth_commits.append(PropositionTruthCommit(
            proposition_id=pid,
            fabula_time=int(ftt),
            truth=bool(new_truth),
            triggered_by="DO_OPERATOR",
        ))
        # Pearl-Rung-2 cross-link: if a GlobalTrait carries
        # ``proposition_id == pid`` (set during ingestion's reconcile_affect
        # step 8 lexical linking), the truth-clamp also moves the world
        # trait. Magnitude tracks proposition truth: 1.0 when true, 0.0
        # when false. The merge fold honours inertia attenuation when the
        # snapshot lands on the canonical timeline.
        for wt_id, wt in world_state.world_traits.items():
            if getattr(wt, "proposition_id", None) == pid:
                _upsert_world_trait_snapshot(
                    wt_id, "magnitude",
                    1.0 if bool(new_truth) else 0.0,
                    int(ftt),
                )

    # Belief mutations → either a new Belief or a confidence overwrite.
    for bm in physics_result.get("belief_mutations") or []:
        if isinstance(bm, dict):
            holder = bm.get("holder_id")
            target = bm.get("target_id")
            new_conf = bm.get("new_confidence")
            created = bm.get("created", False)
            prop_id = bm.get("proposition_id")
        else:
            holder = getattr(bm, "holder_id", None)
            target = getattr(bm, "target_id", None)
            new_conf = getattr(bm, "new_confidence", None)
            created = getattr(bm, "created", False)
            prop_id = getattr(bm, "proposition_id", None)
        if not holder or not target or new_conf is None:
            continue
        # Locate or create an EntityUpdate at fabula_time_now for this holder.
        eu = next(
            (
                e for e in topology.entity_updates
                if e.entity_id == holder and e.fabula_time == fabula_time_now
            ),
            None,
        )
        if eu is None:
            eu = EntityUpdate(
                entity_id=holder, fabula_time=fabula_time_now, triggered_by=None,
            )
            topology.entity_updates.append(eu)
        if created:
            eu.new_beliefs.append(Belief(
                target_id=target,
                perceived_state="(do-operator)",
                confidence=float(new_conf),
                inertia=0.5,
                established_at_fabula=fabula_time_now,
                proposition_id=prop_id,
            ))
        else:
            eu.belief_confidence_updates.append(BeliefConfidenceUpdate(
                target_id=target,
                proposition_id=prop_id,
                new_confidence=float(new_conf),
            ))

    # Concern mutations → ConcernSnapshot routed via topology.concern_snapshots.
    for cm in physics_result.get("concern_mutations") or []:
        if isinstance(cm, dict):
            ccn_id = cm.get("concern_id")
            field = cm.get("field")
            new_val = cm.get("new_value")
        else:
            ccn_id = getattr(cm, "concern_id", None)
            field = getattr(cm, "field", None)
            new_val = getattr(cm, "new_value", None)
        if not ccn_id or not field:
            continue
        snap_kwargs: Dict[str, Any] = {
            "concern_id": ccn_id,
            "fabula_time": fabula_time_now,
            "triggered_by": "DO_OPERATOR",
        }
        if field == "salience" and new_val is not None:
            snap_kwargs["salience"] = float(new_val)
        elif field == "polarity" and new_val in ("desire", "fear"):
            snap_kwargs["polarity"] = new_val
        elif field == "active":
            # ``DoConcern`` emits two shapes: ``[lo, hi]`` (active=False;
            # window collapsed past horizon) or ``None`` (active=True;
            # always-on — window cleared). Encode the latter as the
            # empty-list sentinel that ``reconstruct_concern_at`` already
            # treats as ``always-active``. Anything else is malformed
            # and skipped.
            if isinstance(new_val, (list, tuple)) and len(new_val) == 2:
                snap_kwargs["activation_fabula_window"] = list(new_val)
            elif new_val is None:
                snap_kwargs["activation_fabula_window"] = []
            else:
                continue
        else:
            continue
        try:
            topology.concern_snapshots.append(ChunkConcernSnapshot(**snap_kwargs))
        except Exception:
            logger.debug(
                "[Bridge] Could not build ChunkConcernSnapshot for %s.%s.", ccn_id, field,
            )

    # World-trait mutations → WorldTraitSnapshot via the merge-fold
    # helper. Per-chunk Consequences updates and Rung-2 ``DoWorldTrait``
    # surgeries land via the same code path so inertia attenuation and
    # ``(fabula_time, world_id, triggered_by)`` dedup are consistent.
    for wm in physics_result.get("world_trait_mutations") or []:
        if isinstance(wm, dict):
            wt_id = wm.get("world_trait_id")
            new_val = wm.get("new_value")
            ftt = wm.get("fabula_time")
        else:
            wt_id = getattr(wm, "world_trait_id", None)
            new_val = getattr(wm, "new_value", None)
            ftt = getattr(wm, "fabula_time", None)
        if not wt_id or new_val is None:
            continue
        _upsert_world_trait_snapshot(
            wt_id, "magnitude", float(new_val),
            int(ftt) if ftt is not None else fabula_time_now,
        )

    # Object mutations → ObjectUpdate on the chunk topology. Pearl
    # Rung-2 ``DoNarrativeObject`` surgeries land here so the merge
    # step can fold each into a single ``ObjectStateSnapshot`` on the
    # canonical ``NarrativeObject.state_timeline`` \u2014 keeping the
    # bridge symmetric with ``EntityUpdate`` for entities and
    # ``WorldTraitSnapshot`` for world traits.
    for om in physics_result.get("object_mutations") or []:
        if isinstance(om, dict):
            obj_id = om.get("object_id")
            ftt = om.get("fabula_time")
            new_loc = om.get("new_location_id")
            new_own = om.get("new_owner_id")
            clr_loc = bool(om.get("set_location_null", False))
            clr_own = bool(om.get("set_owner_null", False))
            props_set = dict(om.get("properties_set") or {})
            props_unset = list(om.get("properties_unset") or [])
            trig = om.get("triggered_by")
        else:
            obj_id = getattr(om, "object_id", None)
            ftt = getattr(om, "fabula_time", None)
            new_loc = getattr(om, "new_location_id", None)
            new_own = getattr(om, "new_owner_id", None)
            clr_loc = bool(getattr(om, "set_location_null", False))
            clr_own = bool(getattr(om, "set_owner_null", False))
            props_set = dict(getattr(om, "properties_set", None) or {})
            props_unset = list(getattr(om, "properties_unset", None) or [])
            trig = getattr(om, "triggered_by", None)
        if not obj_id:
            continue
        try:
            topology.object_updates.append(ObjectUpdate(
                object_id=obj_id,
                fabula_time=int(ftt) if ftt is not None else fabula_time_now,
                triggered_by=trig,
                new_location_id=new_loc,
                new_owner_id=new_own,
                set_location_null=clr_loc,
                set_owner_null=clr_own,
                properties_set=props_set,
                properties_unset=props_unset,
            ))
        except Exception:
            logger.debug(
                "[Bridge] Could not build ObjectUpdate for %s.", obj_id,
            )

    # --- Observation reveals (Rung 1) -----------------------------------
    # ``observation_facts`` is a Dict[node_id, value_str] declaring
    # facts the POV (or audience) has now observed. Without a bridge
    # these reveals only land if the rendered prose verbalises them
    # clearly enough for the extractor to re-discover; many subtle
    # reveals get lost. Materialise them deterministically:
    #   * ``ENT_*`` → EntityUpdate with ``new_status=value``;
    #     when the value names a known location (``LOC_...``) we
    #     instead set ``new_location_id``.
    #   * ``WORLD_*`` → WorldTraitSnapshot with no magnitude change
    #     (the reveal is epistemic, not in-world); we attach a
    #     status-style description on the snapshot's triggered_by
    #     when the value parses as a number it's promoted to
    #     magnitude.value (clamped to [0,1]).
    #   * ``OBJ_*`` / ``LOC_*`` → recorded as a ``new_status``
    #     entity_update on the carrier entity if any; otherwise a
    #     debug log (we don't currently model object-state timelines
    #     as snapshots).
    #   * ``EVT_*`` → rewrite the event's ``at_location_id`` in place
    #     when the value names a known LOC_ id (PR 6 of
    #     EventNode.at_location_id). Co-presence repair cascades to
    #     participants on the next merge_topology pass.
    obs_facts = physics_result.get("observation_facts") or {}
    for node_id, raw_value in obs_facts.items():
        if not isinstance(node_id, str) or raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        if node_id.startswith("ENT_"):
            ent = world_state.entities.get(node_id)
            if ent is None:
                continue
            update_kwargs: Dict[str, Any] = {}
            if value.startswith("LOC_") and value in (world_state.locations or {}):
                update_kwargs["new_location_id"] = value
            else:
                # ``EntityUpdate.new_status`` is a Literal —
                # observation_facts may carry free-text values from the
                # physics engine (e.g. "well", "wounded", "deceased").
                # Coerce common aliases; skip anything we cannot map
                # rather than blowing up the merge.
                coerced = str(value).lower().strip()
                if coerced in _VALID_OBSERVATION_STATUSES:
                    update_kwargs["new_status"] = coerced
                else:
                    alias = _OBSERVATION_STATUS_ALIASES.get(coerced)
                    if alias is not None:
                        update_kwargs["new_status"] = alias
                    else:
                        logger.debug(
                            "[_augment_topology_with_sandbox_deltas] "
                            "skipping observation_fact %r=%r — value is "
                            "neither a LOC_ id nor a recognised entity "
                            "status; nothing to anchor.",
                            node_id, value,
                        )
                        continue
            existing = next(
                (
                    eu for eu in topology.entity_updates
                    if eu.entity_id == node_id
                    and eu.fabula_time == fabula_time_now
                ),
                None,
            )
            if existing is None:
                topology.entity_updates.append(EntityUpdate(
                    entity_id=node_id,
                    fabula_time=fabula_time_now,
                    triggered_by=None,
                    **update_kwargs,
                ))
            else:
                # Don't clobber an explicit prior write at this tick.
                for k, v in update_kwargs.items():
                    if getattr(existing, k, None) in (None, "", []):
                        setattr(existing, k, v)
        elif node_id.startswith("WORLD_"):
            # Numeric values are routed through the magnitude
            # snapshot path; non-numeric reveals (descriptive
            # strings) are skipped because WorldTraitSnapshot has
            # no free-form status field today.
            try:
                num = float(value)
            except ValueError:
                continue
            _upsert_world_trait_snapshot(
                node_id, "magnitude", num, fabula_time_now,
            )
        elif node_id.startswith("OBJ_"):
            # Object reveals: parse the value as either an LOC_ id (the
            # POV now sees the object somewhere), an ENT_ id (now in
            # someone's possession), the literal "dropped" / "picked_up"
            # sentinels, or a "key=value" property string. Anything else
            # is logged and skipped \u2014 we will not silently drop or
            # mis-bucket reveals.
            obj = (world_state.objects or {}).get(node_id)
            if obj is None:
                continue
            kwargs: Dict[str, Any] = {
                "object_id": node_id,
                "fabula_time": fabula_time_now,
                "triggered_by": None,
            }
            v = value.strip()
            if v.startswith("LOC_") and v in (world_state.locations or {}):
                kwargs["new_location_id"] = v
                kwargs["set_owner_null"] = True
            elif v.startswith("ENT_") and v in (world_state.entities or {}):
                kwargs["new_owner_id"] = v
                kwargs["set_location_null"] = True
            elif v.lower() in ("dropped", "placed"):
                kwargs["set_owner_null"] = True
            elif v.lower() in ("picked_up", "taken", "held"):
                kwargs["set_location_null"] = True
            elif "=" in v:
                k, _, val = v.partition("=")
                k = k.strip(); val = val.strip()
                if k:
                    kwargs["properties_set"] = {k: val}
            else:
                logger.debug(
                    "[Bridge\u00b7observe] OBJ_ reveal %s=%r unrecognised "
                    "(expected LOC_, ENT_, picked_up/dropped, or k=v); "
                    "skipped.", node_id, raw_value,
                )
                continue
            try:
                topology.object_updates.append(ObjectUpdate(**kwargs))
            except Exception:
                logger.debug(
                    "[Bridge\u00b7observe] Could not build ObjectUpdate for %s.",
                    node_id,
                )
        elif node_id.startswith("EVT_"):
            # PR 6 of EventNode.at_location_id: a Rung-1 reveal of the
            # form ``EVT_X = LOC_Y`` (or the literal phrase ``at LOC_Y``)
            # rewrites the event's spatial anchor in place. We do NOT
            # touch the event's actor/target lists \u2014 co-presence
            # repair (PR 2 of this series) handles cascading the move
            # to participants on the next merge_topology pass. Anything
            # else is logged and skipped.
            ev = next(
                (e for e in (world_state.events or []) if e.id == node_id),
                None,
            )
            if ev is None:
                logger.debug(
                    "[Bridge\u00b7observe] EVT_ reveal %s=%r skipped \u2014 "
                    "event not in world_state.events.", node_id, raw_value,
                )
                continue
            v = value.strip()
            # Accept either bare ``LOC_X`` or ``at LOC_X`` / ``@ LOC_X``.
            for prefix in ("at ", "@ ", "@"):
                if v.lower().startswith(prefix):
                    v = v[len(prefix):].strip()
                    break
            if not v.startswith("LOC_") or v not in (world_state.locations or {}):
                logger.debug(
                    "[Bridge\u00b7observe] EVT_ reveal %s=%r unrecognised "
                    "(expected a known LOC_ id, optionally prefixed by "
                    "'at '); skipped.", node_id, raw_value,
                )
                continue
            old_loc = ev.at_location_id
            if old_loc == v:
                continue
            ev.at_location_id = v
            # ROUND-15 (M-8): also re-emit the post-mutation event
            # through topology.events so the merge persists the reveal.
            # Without this, the in-place mutation on the transient
            # world_state is lost after the snapshot is committed.
            if ev not in topology.events:
                topology.events.append(ev)
            logger.info(
                "[Bridge\u00b7observe\u00b7EventLocation] Relocated %s "
                "at_location_id=%r \u2192 %r (fabula_time=%s); co-presence "
                "repair will cascade on next merge_topology.",
                node_id, old_loc, v, ev.fabula_time,
            )
        else:
            logger.debug(
                "[Bridge·observe] Skipped reveal for non-ENT/WORLD/OBJ node %s "
                "(value=%r) — no snapshot path.", node_id, raw_value,
            )

    # --- Shadow-branch suppression cascade (Rung-2 intervention /
    # Rung-3 counterfactual only). ----------------------------------
    # Physics' ``pruned_utterance_event_ids`` enumerates events the
    # do-surgery rendered epistemically inert (event_type =
    # 'prevented' / truth_value = 'false'). For a shadow merge we
    # expand that set to its disjunctive ``chain_reaction``-closure
    # (Pearl's structural-equation reading: a downstream event is
    # pruned only when every direct-cause parent is pruned, so
    # over-determined effects survive) and stash it on
    # ``topology.suppressed_event_ids``. The merge step then deletes
    # those events + cascade from the shadow snapshot regardless of
    # their original ``world_id`` tag — so the persisted shadow
    # VersionRow's world_state_json matches the prose the renderer
    # generated (Mrs Coady's heart-attack event vanishes alongside
    # the dog-killing that caused it).
    #
    # Strictly gated on (a) ``query_type`` being a true do-surgery
    # query (intervention / counterfactual) and (b) the merge
    # actually being onto a shadow branch. Rung-1 paths
    # (observation, continuation, affective) and any factual save
    # must keep additive merge semantics; new events from those
    # queries join the parent world without ever deleting ancestors.
    if (
        world_id == "shadow"
        and query_type in {"intervention", "counterfactual"}
    ):
        prune_roots = set(
            physics_result.get("pruned_utterance_event_ids") or []
        )
        # Cause-disconnected seeds: EVT_ ids whose ``outcome`` /
        # ``event_type`` were do-flipped this run. The event itself
        # still occurs (do not delete it), but its chain_reaction
        # children should be re-evaluated under the new payload —
        # otherwise the prose can render "George was acquitted"
        # while the persisted world still carries
        # ``EVT_GEORGE_JAILED`` as if the conviction had stuck.
        intervened = set(physics_result.get("intervened_nodes") or [])
        # ROUND-15 (C-2): a DoEvent whose only effect was relocation or
        # fabula-time shift preserves the causal chain — its descendants
        # should NOT be cause-disconnected. Subtract EVT_ ids whose
        # event_mutations are exclusively relocation/time_shift records.
        _evt_muts = physics_result.get("event_mutations") or []
        _non_disruptive: set[str] = set()
        _disruptive: set[str] = set()
        for _m in _evt_muts:
            _mkind = (_m.get("kind") if isinstance(_m, dict)
                      else getattr(_m, "kind", None))
            _meid = (_m.get("event_id") if isinstance(_m, dict)
                     else getattr(_m, "event_id", None))
            if not _meid:
                continue
            if _mkind in ("relocation", "time_shift"):
                _non_disruptive.add(_meid)
            else:
                _disruptive.add(_meid)
        # An event with any disruptive mutation stays cause-broken.
        _preserve = _non_disruptive - _disruptive
        cause_broken: set[str] = {
            nid for nid in intervened
            if isinstance(nid, str)
            and nid.startswith("EVT_")
            and nid not in _preserve
        }
        if prune_roots or cause_broken:
            closure = _compute_shadow_prune_closure(
                world_state, prune_roots,
                cause_disconnected_event_ids=cause_broken,
            )
            if closure:
                # Preserve any caller-set suppressions (none today,
                # but cheap to be additive).
                existing = set(topology.suppressed_event_ids or [])
                topology.suppressed_event_ids = sorted(existing | closure)
                # Telemetry for redundancy auditing: since the
                # post-2026-05-29 work, ``CausalPhysicsEngine.execute_interventions``
                # Step B.6 already expands the chain_reaction
                # descendant closure at intervention time and surfaces
                # the result in ``pruned_utterance_event_ids``. The
                # merge-time pass below remains as belt-and-braces;
                # tagging which branch ran lets us audit how often
                # the engine-side closure was complete on its own.
                already_closed = closure <= (prune_roots | cause_broken)
                logger.info(
                    "[Bridge\u00b7shadow-prune] query_type=%s expanded %d "
                    "physics-pruned + %d cause-disconnected event "
                    "root(s) \u2192 %d-event deletion closure for "
                    "shadow merge (closure_already_complete=%s): %s",
                    query_type, len(prune_roots), len(cause_broken),
                    len(closure), already_closed, sorted(closure),
                )

    # ROUND-15 (H-6): bridge ``edge_mutations`` (DoCausalEdge /
    # DoSpatialEdge / DoChannel) into the topology's vocabulary so the
    # surgery persists into the merged snapshot regardless of whether
    # the renderer's prose re-articulated it. Attribute-only mutations
    # (lock/unlock/retune) have no clean topology counterpart today and
    # rely on the in-place mutations the physics engine already applied
    # to ``world_state``; they are logged but not bridged.
    try:
        _edge_muts = list(physics_result.get("edge_mutations") or [])
        _causal_added = 0
        _causal_severed = 0
        _spatial_added = 0
        _spatial_severed = 0
        _channels_added = 0
        _channels_removed = 0
        _attr_only_skipped = 0
        for _em in _edge_muts:
            _etype = _em.get("edge_type") if isinstance(_em, dict) else getattr(_em, "edge_type", None)
            _act = _em.get("action") if isinstance(_em, dict) else getattr(_em, "action", None)
            _src = _em.get("source_id") if isinstance(_em, dict) else getattr(_em, "source_id", None)
            _tgt = _em.get("target_id") if isinstance(_em, dict) else getattr(_em, "target_id", None)
            _chn = _em.get("channel_id") if isinstance(_em, dict) else getattr(_em, "channel_id", None)
            _ft = _em.get("fabula_time") if isinstance(_em, dict) else getattr(_em, "fabula_time", fabula_time_now)
            _det = _em.get("details") if isinstance(_em, dict) else getattr(_em, "details", {})
            _det = _det or {}
            if _etype == "causal" and _act == "sever" and _src and _tgt:
                _ctype = _det.get("causality_type") or "*"
                key = (_src, _tgt, _ctype, int(_ft))
                if key not in topology.removed_causal_edge_keys:
                    topology.removed_causal_edge_keys.append(key)
                    _causal_severed += 1
            elif _etype == "causal" and _act == "add" and _src and _tgt:
                # Pull the live edge from world_state (physics added it
                # in-place) and copy into topology.causal_topology so
                # the merge sees it.
                for ce in (world_state.causal_topology or []):
                    if ce.source_id == _src and ce.target_id == _tgt:
                        if ce not in topology.causal_topology:
                            topology.causal_topology.append(ce)
                            _causal_added += 1
                        break
            elif _etype == "spatial" and _act == "sever" and _src and _tgt:
                key = (_src, _tgt)
                if key not in topology.removed_spatial_keys:
                    topology.removed_spatial_keys.append(key)
                    _spatial_severed += 1
            elif _etype == "spatial" and _act == "add" and _src and _tgt:
                for se in (world_state.spatial_topology or []):
                    if se.source_id == _src and se.target_id == _tgt:
                        if se not in topology.spatial_topology:
                            topology.spatial_topology.append(se)
                            _spatial_added += 1
                        break
            elif _etype == "channel" and _act == "deactivate" and _chn:
                if _chn not in topology.removed_channel_ids:
                    topology.removed_channel_ids.append(_chn)
                    _channels_removed += 1
            elif _etype == "channel" and _act == "activate" and _chn:
                live_chn = (world_state.channels or {}).get(_chn)
                if live_chn is not None and _chn not in topology.channels:
                    topology.channels[_chn] = live_chn
                    _channels_added += 1
            else:
                # lock/unlock/retune — attribute mutations on existing
                # nodes/edges. The in-place mutation on world_state will
                # ride through merge_topology IFF the merge re-reads the
                # mutated node; today we lean on prose re-extraction to
                # re-materialise. Flagged for future explicit bridging.
                _attr_only_skipped += 1
        if (_causal_severed or _causal_added or _spatial_severed
                or _spatial_added or _channels_added or _channels_removed
                or _attr_only_skipped):
            logger.info(
                "[Bridge\u00b7edge-mut] causal +%d / -%d, spatial +%d / -%d, "
                "channels +%d / -%d, attr-only skipped=%d",
                _causal_added, _causal_severed,
                _spatial_added, _spatial_severed,
                _channels_added, _channels_removed,
                _attr_only_skipped,
            )
    except Exception:
        logger.exception(
            "[Bridge\u00b7edge-mut] failed to bridge edge_mutations into topology."
        )

    # ROUND-15 (H-7): bridge ``event_mutations`` (DoEvent relocation /
    # time_shift) into the topology so the surgery persists. The
    # physics engine has already mutated ``world_state.events`` in
    # place; we re-emit the post-mutation EventNode through
    # ``topology.events`` so the merge dedup path (matched by
    # ``EventNode.id``) replaces the parent-world copy.
    try:
        _evt_muts = list(physics_result.get("event_mutations") or [])
        _evt_ids: set[str] = set()
        for _m in _evt_muts:
            _eid = _m.get("event_id") if isinstance(_m, dict) else getattr(_m, "event_id", None)
            if _eid:
                _evt_ids.add(_eid)
        if _evt_ids:
            _evts_by_id = {e.id: e for e in (world_state.events or [])}
            _existing_topo_ids = {e.id for e in (topology.events or [])}
            _added = 0
            for _eid in _evt_ids:
                if _eid in _existing_topo_ids:
                    continue
                _live = _evts_by_id.get(_eid)
                if _live is not None:
                    topology.events.append(_live)
                    _added += 1
            if _added:
                logger.info(
                    "[Bridge\u00b7event-mut] bridged %d DoEvent surgeries "
                    "into topology.events (relocation/time_shift).", _added,
                )
    except Exception:
        logger.exception(
            "[Bridge\u00b7event-mut] failed to bridge event_mutations into topology."
        )

    # AUDIT P0-1: bridge typed DoEntityDelete / DoObjectDelete from the
    # physics sandbox into the topology's removed_*_ids so the merge
    # cascade scrubs every referential surface (mirrors the cascade
    # already applied in-place by ``causal_physics`` handlers).
    # ROUND-15 (C-1): payloads from ``_typed_target_payload`` arrive as
    # ``model_dump()`` dicts, so the old ``isinstance`` dispatch was a
    # silent no-op. Dispatch on the ``target_kind`` literal instead and
    # accept either dicts or typed models for forward-compat.
    try:
        from shadow_loom.query_models import (
            DoEntityDelete as _DoEntityDelete,
            DoObjectDelete as _DoObjectDelete,
        )
        _do_targets = list(physics_result.get("do_targets") or [])
        _do_targets += list(physics_result.get("historical_do_targets") or [])
        for _t in _do_targets:
            if isinstance(_t, dict):
                _kind = _t.get("target_kind")
                _eid = _t.get("entity_id")
                _oid = _t.get("object_id")
            else:
                _kind = getattr(_t, "target_kind", None)
                _eid = getattr(_t, "entity_id", None)
                _oid = getattr(_t, "object_id", None)
            if _kind == "entity_delete" or isinstance(_t, _DoEntityDelete):
                if _eid and _eid not in topology.removed_entity_ids:
                    topology.removed_entity_ids.append(_eid)
            elif _kind == "object_delete" or isinstance(_t, _DoObjectDelete):
                if _oid and _oid not in topology.removed_object_ids:
                    topology.removed_object_ids.append(_oid)
    except Exception:
        logger.exception(
            "[Bridge\u00b7do-delete] failed to bridge typed "
            "DoEntityDelete / DoObjectDelete targets onto "
            "topology removed_*_ids."
        )

    return topology


def _resolve_manual_edit_anchor(
    query: "ManualEditQuery",
    ws: WorldStateV1,
    cfg: PipelineConfig,
) -> int:
    """Return the ``fabula_time_base`` for a manual-edit re-extraction.

    Precedence:
    1. Explicit ``query.insert_at_fabula_time``.
    2. ``query.insert_after_event_id`` → that event's
       ``fabula_time + extraction.fabula_time_spacing``.
    3. ``max(events.fabula_time) + spacing`` (continuation).
    """
    spacing = cfg.extraction_config.fabula_time_spacing
    if query.insert_at_fabula_time is not None:
        return query.insert_at_fabula_time
    if query.insert_after_event_id:
        for evt in ws.events:
            if evt.id == query.insert_after_event_id:
                return evt.fabula_time + spacing
        logger.warning(
            "[Pipeline] Manual edit anchor event %s not found in world; "
            "falling back to chronological end.",
            query.insert_after_event_id,
        )
    existing_max = max(
        (e.fabula_time for e in ws.events), default=-spacing,
    )
    return existing_max + spacing


def _apply_manual_edit_replacements(
    ws: WorldStateV1, query: "ManualEditQuery",
) -> WorldStateV1:
    """Return a deep-copy of ``ws`` with the user's replace_* targets
    (and their dependent edges) removed, for *replace* manual-edit
    semantics.

    This runs **before** the new prose is re-extracted so the LLM
    doesn't see the about-to-be-replaced graph slice as "previous
    context" and try to keep it stitched in. The merge step's
    deletion pass repeats the same drop set on the post-extract
    side so cascades (beliefs targeting removed entities, snapshots
    referencing removed propositions, etc.) are cleaned consistently.
    """
    drop_events = set(query.replace_event_ids)
    drop_entities = set(query.replace_entity_ids)
    drop_objects = set(query.replace_object_ids)
    drop_locations = set(query.replace_location_ids)
    drop_traits = set(query.replace_world_trait_ids)
    drop_channels = set(query.replace_channel_ids)
    drop_props = set(query.replace_proposition_ids)
    drop_concerns = {tuple(p) for p in query.replace_concern_ids}
    if not (
        drop_events or drop_entities or drop_objects or drop_locations
        or drop_traits or drop_channels or drop_props or drop_concerns
    ):
        return ws
    new_ws = ws.model_copy(deep=True)
    if drop_events:
        new_ws.events = [e for e in new_ws.events if e.id not in drop_events]
        new_ws.causal_topology = [
            ce for ce in new_ws.causal_topology
            if ce.source_id not in drop_events and ce.target_id not in drop_events
        ]
        new_ws.social_topology = [
            re for re in new_ws.social_topology
            if not getattr(re, "evidence_event_ids", None)
            or not (set(re.evidence_event_ids) & drop_events)
        ]
        new_ws.spatial_topology = [
            se for se in new_ws.spatial_topology
            if not getattr(se, "established_by_event_id", None)
            or se.established_by_event_id not in drop_events
        ]
        # Cascade: scrub belief provenance pointing at removed events.
        for ent in new_ws.entities.values():
            for b in ent.beliefs:
                if getattr(b, "acquired_via_event_id", None) in drop_events:
                    b.acquired_via_event_id = None
    if drop_entities:
        new_ws.entities = {
            eid: e for eid, e in new_ws.entities.items()
            if eid not in drop_entities
        }
        # Round-6 audit: RelationshipEdge exposes ``source_entity_id`` /
        # ``target_entity_id`` (per shadow_loom/models.py); the prior
        # ``re.source_id`` / ``re.target_id`` access raised AttributeError
        # on every manual_edit replacement that dropped an entity, so the
        # cascade clean-up never ran and the merge aborted.
        new_ws.social_topology = [
            re for re in new_ws.social_topology
            if re.source_entity_id not in drop_entities
            and re.target_entity_id not in drop_entities
        ]
        for ent in new_ws.entities.values():
            ent.beliefs = [
                b for b in ent.beliefs if b.target_id not in drop_entities
            ]
    if drop_objects:
        new_ws.objects = {
            oid: o for oid, o in new_ws.objects.items()
            if oid not in drop_objects
        }
    if drop_locations:
        new_ws.locations = {
            lid: l for lid, l in new_ws.locations.items()
            if lid not in drop_locations
        }
        # Round-6 audit: SpatialEdge exposes ``source_id`` / ``target_id``
        # (per shadow_loom/models.py); the prior ``se.location_id`` access
        # raised AttributeError on every manual_edit replacement that
        # dropped a location, aborting the merge. Drop edges whose
        # endpoints reference a removed location.
        new_ws.spatial_topology = [
            se for se in new_ws.spatial_topology
            if se.source_id not in drop_locations
            and se.target_id not in drop_locations
        ]
    if drop_traits:
        new_ws.world_traits = {
            tid: t for tid, t in new_ws.world_traits.items()
            if tid not in drop_traits
        }
    if drop_channels:
        new_ws.channels = {
            cid: c for cid, c in new_ws.channels.items()
            if cid not in drop_channels
        }
        # Cascade: scrub channel pointers on events and beliefs.
        for evt in new_ws.events:
            if getattr(evt, "via_channel_id", None) in drop_channels:
                evt.via_channel_id = None
        for ent in new_ws.entities.values():
            for b in ent.beliefs:
                if getattr(b, "acquired_via_channel_id", None) in drop_channels:
                    b.acquired_via_channel_id = None
    if drop_props:
        new_ws.propositions = [
            p for p in new_ws.propositions
            if p.proposition_id not in drop_props
        ]
        # Cascade: clear proposition refs on events and beliefs.
        for evt in new_ws.events:
            if getattr(evt, "asserts_proposition_id", None) in drop_props:
                evt.asserts_proposition_id = None
            if getattr(evt, "denies_proposition_id", None) in drop_props:
                evt.denies_proposition_id = None
            resolves = getattr(evt, "resolves_proposition_ids", None)
            if resolves:
                evt.resolves_proposition_ids = [
                    pid for pid in resolves if pid not in drop_props
                ]
        for ent in new_ws.entities.values():
            ent.concerns = [
                c for c in ent.concerns if c.proposition_id not in drop_props
            ]
            for b in ent.beliefs:
                if getattr(b, "proposition_id", None) in drop_props:
                    b.proposition_id = None
    if drop_concerns:
        # Build a set of removed concern_ids regardless of entity for
        # scrubbing counter_concern_ids on surviving concerns.
        removed_cids = {cid for _eid, cid in drop_concerns}
        for eid, ent in new_ws.entities.items():
            ent.concerns = [
                c for c in ent.concerns
                if (eid, c.concern_id) not in drop_concerns
            ]
            for c in ent.concerns:
                ccids = getattr(c, "counter_concern_ids", None)
                if ccids:
                    c.counter_concern_ids = [
                        x for x in ccids if x not in removed_cids
                    ]
    return new_ws


def _populate_removed_fields_from_manual_edit(
    topology: "ChunkTopology", query: "ManualEditQuery",
) -> None:
    """Mirror the manual-edit replace_* selections onto the topology's
    removed_* fields so the merge's deletion pass cascades through
    anything the pre-extract trim missed (e.g. cross-entity beliefs
    referring to the removed proposition).
    """
    if query.replace_event_ids:
        topology.removed_event_ids.extend(query.replace_event_ids)
    if query.replace_entity_ids:
        topology.removed_entity_ids.extend(query.replace_entity_ids)
    if query.replace_object_ids:
        topology.removed_object_ids.extend(query.replace_object_ids)
    if query.replace_location_ids:
        topology.removed_location_ids.extend(query.replace_location_ids)
    if query.replace_world_trait_ids:
        topology.removed_world_trait_ids.extend(query.replace_world_trait_ids)
    if query.replace_channel_ids:
        topology.removed_channel_ids.extend(query.replace_channel_ids)
    if query.replace_proposition_ids:
        topology.removed_proposition_ids.extend(query.replace_proposition_ids)
    if query.replace_concern_ids:
        topology.removed_concern_ids.extend(
            tuple(p) for p in query.replace_concern_ids
        )


# =====================================================================
# The pipeline
# =====================================================================

def run_pipeline(
    query: UserRequest,
    *,
    world_state: WorldStateV1 | None = None,
    versioned_model: VersionedWorldModel | None = None,
    raw_text: str | None = None,
    config: PipelineConfig | None = None,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
) -> PipelineResult:
    """Run the full Shadow-Loom pipeline end-to-end.

    Supply exactly one of:
      • ``world_state`` — an existing ``WorldStateV1`` (wrapped in a new
        ``VersionedWorldModel`` automatically).
      • ``versioned_model`` — an existing ``VersionedWorldModel`` to
        continue building on (preserves history).
      • ``raw_text`` — raw narrative text to ingest first (Step 1).

    Parameters
    ----------
    query : UserRequest
        The user query (observation, intervention, counterfactual,
        directive, interrogate, or general).
    world_state : WorldStateV1 or None
        Pre-built world state. Mutually exclusive with the others.
    versioned_model : VersionedWorldModel or None
        Pre-built versioned world model with history.
    raw_text : str or None
        Raw narrative text to ingest into a WorldStateV1 first.
    config : PipelineConfig or None
        Unified configuration for all stages.

    Returns
    -------
    PipelineResult
        Contains the final prose, scene, physics result, audit result,
        updated versioned world model, and full step history.
    """
    cfg = config or PipelineConfig()
    history = PipelineHistory()
    result = PipelineResult(query_type=query.query_type, history=history)

    # =================================================================
    # Step 0: Resolve the world model
    # =================================================================
    sources = sum([world_state is not None, versioned_model is not None, raw_text is not None])
    if sources == 0:
        raise ValueError(
            "Supply one of: world_state, versioned_model, or raw_text."
        )
    if sources > 1:
        raise ValueError(
            "Supply only one of: world_state, versioned_model, or raw_text."
        )

    if raw_text is not None:
        # --- Step 1: Ingestion ---
        logger.info("[Pipeline] Step 1: Ingesting raw text (%d chars).", len(raw_text))
        ing_cfg = cfg.ingestion_config or ExtractionConfig()
        # Tier 4 #14: capture validator/auto-fix warnings for this project.
        with capture_ingestion_warnings(str(project_id) if project_id is not None else "_anon"):
            # Round-7 audit: ``asyncio.run`` raises ``RuntimeError`` when
            # invoked from a thread that already has a running event loop
            # (e.g. a Jupyter cell, a FastAPI route, the NiceGUI server
            # thread). Detect that and offload to a worker thread so the
            # sync pipeline stays usable from inside an async caller.
            def _run_ingest():
                return asyncio.run(run_extraction_async(
                    raw_text,
                    config=ing_cfg,
                    user_id=user_id,
                    project_id=project_id,
                    version_id=version_id,
                ))
            try:
                asyncio.get_running_loop()
                _loop_active = True
            except RuntimeError:
                _loop_active = False
            if _loop_active:
                import threading as _t
                _box: dict = {}
                def _worker():
                    try:
                        _box["r"] = _run_ingest()
                    except BaseException as _e:  # noqa: BLE001
                        _box["e"] = _e
                _th = _t.Thread(target=_worker, daemon=True)
                _th.start()
                _th.join()
                if "e" in _box:
                    raise _box["e"]
                if "r" not in _box:
                    raise RuntimeError(
                        "Worker thread completed without returning result or exception"
                    )
                ws, validation_report = _box["r"]
            else:
                ws, validation_report = _run_ingest()
        vwm = VersionedWorldModel.from_world_state(ws, max_snapshots=cfg.max_snapshots)
        history.record("ingestion", IngestionStepRecord(
            is_valid=validation_report.is_valid,
            num_events=len(ws.events),
            num_entities=len(ws.entities),
            num_locations=len(ws.locations),
        ))
        logger.info(
            "[Pipeline] Ingestion complete — %d events, %d entities, valid=%s.",
            len(ws.events), len(ws.entities), validation_report.is_valid,
        )
    elif versioned_model is not None:
        vwm = versioned_model
        ws = vwm.current
        logger.info(
            "[Pipeline] Using existing VersionedWorldModel at v%d.",
            vwm.version,
        )
    else:
        # world_state is not None
        vwm = VersionedWorldModel.from_world_state(world_state, max_snapshots=cfg.max_snapshots)
        ws = vwm.current
        logger.info("[Pipeline] Wrapped WorldStateV1 in VersionedWorldModel v0.")

    result.world_model = vwm

    # =================================================================
    # Step 2: Narrative Physics
    # =================================================================
    logger.info(
        "[Pipeline] Step 2: Narrative physics — query_type=%s, causal_engine=%s.",
        query.query_type, cfg.use_causal_engine,
    )
    # Inherit branch identity from the VWM head so renderer-introduced
    # propositions/concerns land in the same AMWN sub-graph as the
    # rest of the active branch (factual or shadow).
    _active_branch_world_id = (
        vwm.history[-1].world_id if vwm.history else "factual"
    )
    _active_branch_label = (
        vwm.history[-1].branch_label if vwm.history else None
    )
    ws = _apply_query_introductions(
        ws, query, world_id=_active_branch_world_id,
    )
    # AMWN-split projection: when the active branch is shadow,
    # swap ``ws.entities`` for the merged factual + shadow_entities
    # sidecar view so physics, interrogation, and brief-building all
    # see the per-branch split copies (with their independently
    # trimmed state_timelines and accumulated shadow snapshots)
    # instead of the factual entity records. Factual branch reads
    # short-circuit to ``self`` so there is zero overhead on the
    # mainline path. See docs/academic-foundations.md §2.2 (Correa
    # & Bareinboim 2025) for the AMWN construction.
    ws = ws.projected_for_branch(
        branch_world_id=_active_branch_world_id,
        branch_label=_active_branch_label,
    )
    # Per-query surgery isolation — deep-clones ws so engine-side
    # ``_apply_do_*`` handlers (which mutate channels / events /
    # social_topology / spatial_topology / causal_topology / concerns /
    # propositions in place) cannot leak back to the VWM-held world
    # state. Without this every Rung-2 / Rung-3 query permanently
    # contaminated the factual world for subsequent queries. No-op for
    # observation / interrogate / general / evaluate (read-only paths).
    ws = _isolate_ws_for_surgery(ws, query)
    eff_temporal, eff_syuzhet = _resolve_query_anchors(
        query, cfg.temporal_anchor, cfg.syuzhet_anchor, ws,
    )
    # CC-3 (2026-05-29): advisory world-schema pre-pass for ALL query
    # types. Surfaces dangling references / location plausibility /
    # temporal-coherence defects into the history ledger before
    # physics so the MCP envelope, the UI, and post-mortem audits see
    # the same warnings regardless of query type. Logged at INFO so
    # the pre-pass is visible in pipeline traces without blocking.
    try:
        from shadow_loom.world_schema_audit import (
            audit_world_schema as _audit_world_schema,
        )
        _pre_warnings = _audit_world_schema(ws)
        if _pre_warnings:
            history.record(
                "world_schema_audit",
                {
                    "warning_count": len(_pre_warnings),
                    "warnings": _pre_warnings[:50],
                    "truncated": len(_pre_warnings) > 50,
                },
            )
            logger.info(
                "[Pipeline\u00b7schema\u00b7pre] %d world-schema warning(s) "
                "surfaced before physics (advisory).",
                len(_pre_warnings),
            )
    except Exception:
        logger.exception(
            "[Pipeline\u00b7schema\u00b7pre] World-schema pre-pass "
            "failed; continuing."
        )
    physics_result = calculate_narrative_physics(
        request=query,
        global_world_state=ws,
        temporal_anchor=eff_temporal,
        syuzhet_anchor=eff_syuzhet,
        use_causal_engine=cfg.use_causal_engine,
    )
    result.physics_result = physics_result

    # Extract the keys that aren't common metadata. ``_causal_physics_result``
    # is a typed object stashed for the auditor handoff and must not be
    # serialised into PipelineHistory.
    common_keys = {"status", "query_type", "physics_state", "_causal_physics_result"}
    extras = {k: v for k, v in physics_result.items() if k not in common_keys}

    history.record("narrative_physics", PhysicsStepRecord(
        query_type=physics_result.get("query_type", query.query_type),
        status=physics_result.get("status", "unknown"),
        physics_state=physics_result.get("physics_state", {}),
        extra=extras,
    ))

    # Implausibility short-circuit — explain, don't mutate.
    if physics_result.get("status") == "implausible":
        _apply_implausibility_short_circuit(
            query=query, physics_result=physics_result,
            vwm=vwm, result=result, history=history,
            log_prefix="[Pipeline]",
        )
        return result

    # Forced-past-implausibility — still flag the result so callers can warn.
    if "implausibility_warning" in physics_result:
        result.implausible = True
        result.implausibility_reason = physics_result.get("implausibility_warning")
        result.implausibility_details = physics_result.get("implausibility_details", {}) or {}
        history.record("implausibility", {
            "forced": True,
            "reason": result.implausibility_reason,
            "details": result.implausibility_details,
        })
        logger.warning(
            "[Pipeline] Forced past implausibility gate — generating prose anyway: %s",
            result.implausibility_reason,
        )

    # =================================================================
    # Non-prose queries: run the LLM Q&A step, then stop
    # =================================================================
    if query.query_type in ("interrogate", "general"):
        logger.info(
            "[Pipeline] Query type '%s' — answering question over physics state.",
            query.query_type,
        )
        # AUDIT P0-7: compute the deterministic posterior table
        # BEFORE the LLM answer step so the answerer (and any
        # downstream auditor) reads off a verifiable ground truth
        # rather than hallucinating ranks. Surfaced into
        # ``physics_result['posterior']`` for the answer template
        # and into ``physics_result['posterior_warnings']`` for the
        # auditor / UI.
        try:
            from shadow_loom.interrogate_posterior import (
                interrogate_posterior as _interrogate_posterior,
                audit_posterior_consistency as _audit_posterior_consistency,
            )
            # R1-2 (2026-05-29): use the *resolved* temporal anchor
            # (``eff_temporal`` from ``_resolve_query_anchors``) so
            # interrogate / general posterior reads the truth table
            # in force at the user's requested anchor, not the latest
            # tick. ``query.fabula_time`` does not exist on the query
            # model — the field is ``temporal_anchor`` — so the old
            # ``getattr(query, 'fabula_time', None)`` always returned
            # ``None`` and silently fell back to the latest truth
            # bucket.
            _ft = eff_temporal
            _posterior = _interrogate_posterior(ws, fabula_time=_ft)
            physics_result["posterior"] = _posterior
            physics_result["posterior_warnings"] = (
                _audit_posterior_consistency(_posterior)
            )
            logger.info(
                "[Pipeline·posterior] %d propositions ranked; %d warnings (anchor=%s).",
                len(_posterior),
                len(physics_result["posterior_warnings"]),
                _ft,
            )
        except Exception:
            logger.exception(
                "[Pipeline·posterior] Failed to compute deterministic "
                "interrogate posterior; continuing without it."
            )
        # AUDIT P1-2: surface deterministic affective scorers onto
        # the physics_result so the answer template / auditor / UI
        # see them. Never raises; emits zeros on failure.
        try:
            from shadow_loom.affective_scorers import (
                compute_affective_scorers as _compute_affective_scorers,
            )
            physics_result["affective_scorers"] = _compute_affective_scorers(
                ws, fabula_time=eff_temporal,
            )
        except Exception:
            logger.exception(
                "[Pipeline·affective] Failed to compute affective "
                "scorers; continuing without them."
            )
        # AUDIT round-2 P0: surface immutable entity constants
        # (Feyre.mortal_origin, Edmund.forbidden_kin) as
        # ConstantPrerequisite candidates so abduction / Rung-3
        # readers see intrinsic causes alongside mutable ones.
        try:
            from shadow_loom.interrogate_posterior import (
                surface_entity_constants as _surface_entity_constants,
            )
            physics_result["entity_constants"] = _surface_entity_constants(ws)
        except Exception:
            logger.exception(
                "[Pipeline\u00b7constants] Failed to surface entity "
                "constants; continuing without them."
            )
        # CC-2 (2026-05-29): lightweight deterministic schema audit for
        # readonly query types. Without this pass, an interrogate /
        # general query reads the posterior off a world whose
        # dangling-reference / temporal-coherence defects only surface
        # later as silent empty results. Advisory-only: surfaces a
        # capped list of human-readable warnings into the physics
        # result so the answer template, the MCP envelope, and the UI
        # can show them; never blocks the answer step.
        try:
            from shadow_loom.world_schema_audit import (
                audit_world_schema as _audit_world_schema,
            )
            _schema_warnings = _audit_world_schema(ws)
            physics_result["world_schema_warnings"] = list(_schema_warnings)
            if _schema_warnings:
                logger.info(
                    "[Pipeline\u00b7schema] %d world-schema warning(s) "
                    "surfaced for readonly query (advisory).",
                    len(_schema_warnings),
                )
        except Exception:
            logger.exception(
                "[Pipeline\u00b7schema] World-schema audit failed for "
                "readonly query; continuing without warnings."
            )
        try:
            _run_answer_step(query=query, physics_result=physics_result, cfg=cfg, history=history, vwm=vwm)
        except Exception:
            logger.exception(
                "[Pipeline] Answer step failed for query_type=%s \u2014 "
                "returning a degraded result with no answer prose.",
                query.query_type,
            )
            physics_result["answer_failed"] = True
            physics_result.setdefault(
                "answer",
                "(The model could not produce an answer for this "
                "question; please retry or rephrase.)",
            )
        return result

    # =================================================================
    if query.query_type in ("intervention", "counterfactual"):
        logger.info(
            "[Pipeline] Query type '%s' \u2014 surfacing rung-aware Q&A "
            "answer alongside prose.",
            query.query_type,
        )
        try:
            _run_answer_step(query=query, physics_result=physics_result, cfg=cfg, history=history, vwm=vwm)
        except Exception:
            logger.exception(
                "[Pipeline] Rung-aware answer step failed; continuing "
                "with prose generation."
            )

    # =================================================================
    # Evaluation query — full-story quality audit
    # =================================================================
    if query.query_type == "evaluate":
        _run_evaluation_branch(
            query=query, ws=ws, vwm=vwm, cfg=cfg,
            physics_result=physics_result, result=result, history=history,
        )
        return result

    # =================================================================
    # Manual edit: skip physics/generation/audit — prose is user-supplied
    # =================================================================
    if query.query_type == "manual_edit":
        logger.info("[Pipeline] Manual edit — skipping generation, running re-extraction.")
        result.prose = query.edited_prose

        # Create a minimal GeneratedScene wrapper for consistency.
        # Note: GeneratedScene has no scene_summary field; the user
        # description is captured in the merge description below.
        from shadow_loom.generation import GeneratedScene
        result.scene = GeneratedScene(
            prose=query.edited_prose,
            rendering_mode="manual_edit",
        )
        history.record("generation", GenerationStepRecord(
            scene=result.scene,
            brief=None,
        ))

        # Always run re-extraction for manual edits (the whole point)
        logger.info("[Pipeline] Extracting topology from manually edited prose.")
        try:
            anchor_base = _resolve_manual_edit_anchor(query, ws, cfg)
            ws_for_extract = _apply_manual_edit_replacements(ws, query)
            _world_id, _branch_label = _resolve_branch_policy(query, cfg, vwm)
            _preceding_prose = _gather_preceding_prose(
                vwm,
                branch_world_id=_world_id,
                branch_label=_branch_label,
            )
            topology = extract_topology_from_prose(
                prose=result.prose,
                world_state=ws_for_extract,
                config=cfg.extraction_config,
                fabula_time_base=anchor_base,
                branch_world_id=_world_id,
                branch_label=_branch_label,
                preceding_prose=_preceding_prose,
            )
            _populate_removed_fields_from_manual_edit(topology, query)
            description = (
                f"Manual edit: {query.description}"
                if query.description
                else "Manual edit"
            )
            # If the user asked for replace semantics, start the merge
            # from the *trimmed* world so the dropped events don't
            # come back via deep-copy of ``vwm.current``.
            vwm_for_merge = (
                vwm.model_copy(update={"current": ws_for_extract})
                if (
                    query.replace_event_ids or query.replace_entity_ids
                    or query.replace_object_ids or query.replace_location_ids
                    or query.replace_world_trait_ids or query.replace_channel_ids
                    or query.replace_proposition_ids or query.replace_concern_ids
                )
                else vwm
            )
            vwm_next = vwm_for_merge.merge(
                topology,
                source="manual_edit",
                description=description,
                prose=result.prose,
                world_id=_world_id,
                branch_label=_branch_label,
            )
            changeset = vwm_next.history[-1].changeset
            history.record("reextraction_merge", ReextractionStepRecord(
                events_added=changeset.events_added if changeset else 0,
                causal_edges_added=changeset.causal_edges_added if changeset else 0,
                entity_updates_applied=changeset.entity_updates_applied if changeset else 0,
                entity_updates_skipped=changeset.entity_updates_skipped if changeset else [],
                new_version=vwm_next.version,
            ))
            # Continuation quality bridge — same auto-repair +
            # validation + correction loop as raw-text ingestion,
            # using the EXTRACTION_MODEL.
            vwm_next = _run_continuation_quality_bridge_sync(
                vwm_next, cfg, result,
                log_prefix="[Pipeline·ManualEdit·QualityBridge]",
            )
            result.world_model = vwm_next
            logger.info(
                "[Pipeline] Manual edit merge complete — v%d → v%d.",
                vwm.version, vwm_next.version,
            )
        except Exception as _e:
            logger.exception("[Pipeline] Manual edit re-extraction/merge failed.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})
            result.reextraction_failed = True
            result.reextraction_error = str(_e)

        return result

    # =================================================================
    # Step 3–4: Brief Assembly + Generation
    # =================================================================
    physics_state = physics_result.get("physics_state", {})

    # Resolve branch policy once so every brief-construction site below
    # can stamp the active branch onto the CreativeBrief.
    _branch_world_id, _branch_label = _resolve_branch_policy(query, cfg, vwm)

    # Compute the story-so-far excerpt once for this query so every
    # brief-construction site below can thread it onto the brief and
    # every render_from_query call can pass it to the brief builders.
    # Filtered by the query's branch (factual queries see only factual
    # prose; shadow queries see the active fork's tail + factual
    # ancestors). See ``_gather_preceding_prose`` for the policy.
    _preceding_prose = _gather_preceding_prose(
        vwm,
        branch_world_id=_branch_world_id,
        branch_label=_branch_label,
    )
    # Compute the factual contrast summary for shadow queries so the
    # renderer (and the auditor) can see what *did* happen on canon
    # at the same horizon. None for factual queries.
    _factual_contrast = _compute_factual_contrast(
        vwm,
        branch_world_id=_branch_world_id,
        branch_label=_branch_label,
    )

    # For directive queries with causal engine, the brief is already built
    brief: CreativeBrief | None = None
    if query.query_type == "directive" and "creative_brief" in physics_result:
        brief_data = physics_result["creative_brief"]
        brief = CreativeBrief(**brief_data) if isinstance(brief_data, dict) else brief_data
        _stamp_brief_full(
            brief, vwm,
            branch_world_id=_branch_world_id,
            branch_label=_branch_label,
            factual_contrast=_factual_contrast,
        )

    if cfg.skip_audit:
        # Generate once, no audit loop
        logger.info("[Pipeline] Steps 3–4: Generating prose (audit skipped).")
        gen_cfg = cfg.generation_config or GenerationConfig()
        scene = render_from_query(
            query, physics_result, ws, gen_cfg,
            preceding_prose=_preceding_prose,
            branch_world_id=_branch_world_id,
            branch_label=_branch_label,
            factual_contrast_summary=_factual_contrast,
            syuzhet_anchor=eff_syuzhet,
        )

        history.record("generation", GenerationStepRecord(
            scene=scene,
            brief=brief,
        ))
        result.scene = scene
        result.prose = scene.prose

    else:
        # =============================================================
        # Steps 3–5: Brief → Render → Audit loop
        # =============================================================
        logger.info("[Pipeline] Steps 3–5: Generation + audit loop.")

        if brief is not None:
            # Directive with pre-built brief — use render_and_audit
            # Try to reconstruct the assembler for engine metrics
            _assembler = None
            if query.query_type == "directive":
                try:
                    from shadow_loom.extract_graph import extract_ego_graph_from_memory
                    _ego = extract_ego_graph_from_memory(
                        ws, brief.target_entities,
                        syuzhet_anchor=eff_syuzhet,
                        shadow_path_seed_ids=set(brief.target_entities or []),
                    )
                    _assembler = DirectiveAssembler(
                        sandbox=None, ego_payload=_ego.model_dump(), world_state=ws,
                    )
                except Exception:
                    # Without the assembler, the auditor still runs but
                    # affective-feedback metrics (achieved-vs-target
                    # intensity, affective_loss_mse) won't be populated.
                    # Surface this at WARNING so the missing chat
                    # diagnostics aren't silently swallowed.
                    logger.warning(
                        "[Pipeline] Could not rebuild DirectiveAssembler for "
                        "engine metrics; affective feedback will be missing "
                        "from this query's audit.",
                        exc_info=True,
                    )

            feedback = render_and_audit(
                brief=brief,
                world_state=ws,
                auditor_config=_inert_aware_auditor_config(cfg.auditor_config, physics_result),
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
                physics_result=physics_result.get("_causal_physics_result"),
                assembler=_assembler,
            )
        else:
            # Non-directive: render first, then audit
            gen_cfg = cfg.generation_config or GenerationConfig()
            initial_scene = render_from_query(
                query, physics_result, ws, gen_cfg,
                preceding_prose=_preceding_prose,
                branch_world_id=_branch_world_id,
                branch_label=_branch_label,
                factual_contrast_summary=_factual_contrast,
                syuzhet_anchor=eff_syuzhet,
            )

            # Build a brief for the auditor from the query
            brief = _build_brief_for_query(query, physics_result, ws, syuzhet_anchor=eff_syuzhet)
            _stamp_brief_full(
                brief, vwm,
                branch_world_id=_branch_world_id,
                branch_label=_branch_label,
                factual_contrast=_factual_contrast,
            )

            from shadow_loom.auditor import run_feedback_loop
            feedback = run_feedback_loop(
                initial_scene=initial_scene,
                brief=brief,
                world_state=ws,
                auditor_config=_inert_aware_auditor_config(cfg.auditor_config, physics_result),
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
                physics_result=physics_result.get("_causal_physics_result"),
            )

        history.record("generation", GenerationStepRecord(
            scene=feedback.final_scene,
            brief=brief,
        ))
        history.record("audit", AuditStepRecord(feedback_result=feedback))

        result.scene = feedback.final_scene
        result.prose = feedback.final_scene.prose
        result.converged = feedback.converged
        result.audit_iterations = feedback.iterations
        result.feedback_result = feedback
        _log_feedback_outcome(feedback, async_path=False)

    # =================================================================
    # Steps 6–7: Prose re-extraction + merge
    # =================================================================
    def _do_steps_6_7() -> None:
        """Execute prose re-extraction + merge against the captured locals.

        Defined as a closure so it can either run inline (default) or be
        stashed for ``finish_reextraction`` to drive later when the
        caller passed ``cfg.defer_reextraction=True``. Mutates ``result``
        and ``history`` in place — both are captured by the closure.
        """
        if cfg.skip_reextraction:
            logger.info("[Pipeline] Steps 6–7: Re-extraction skipped.")
            return
        if (
            result.converged is False
            and not cfg.merge_on_audit_failure
        ):
            # The audit did not converge (or failed open). Merging
            # unaudited prose into the canonical world model would let
            # narrative drift past the contract-checking gate. Refuse
            # the merge by default; callers can set
            # ``merge_on_audit_failure=True`` to override for explicit
            # unsafe / debug flows.
            logger.error(
                "[Pipeline] Steps 6–7: Re-extraction blocked — "
                "auditor did not converge (iterations=%s). Set "
                "PipelineConfig.merge_on_audit_failure=True to force.",
                result.audit_iterations,
            )
            result.reextraction_failed = True
            result.reextraction_error = (
                "Skipped re-extraction: auditor did not converge."
            )
            history.record(
                "reextraction_merge",
                {"error": "audit_not_converged"},
            )
            return
        if getattr(result.scene, "generation_error", None):
            # The renderer fell back to a placeholder scene; merging that
            # text into the canonical world state would pollute the graph
            # with junk extracted from a "[Generation failed: ...]" string.
            # Surface the failure on the result so the UI can show it and
            # leave the world model untouched.
            logger.error(
                "[Pipeline] Steps 6–7: Re-extraction skipped — scene "
                "carries generation_error=%s; refusing to merge fallback "
                "prose into world state.",
                result.scene.generation_error,
            )
            result.reextraction_failed = True
            result.reextraction_error = (
                f"Skipped re-extraction: generation failed "
                f"({result.scene.generation_error})."
            )
            history.record(
                "reextraction_merge",
                {"error": "generation_failure_skipped"},
            )
            return
        logger.info("[Pipeline] Steps 6–7: Extracting topology from prose and merging.")
        try:
            spawns = promote_sandbox_spawns(ws, physics_state)
            _world_id, _branch_label = _resolve_branch_policy(query, cfg, vwm)
            engine_priors = _render_engine_priors(physics_result)
            topology = extract_topology_from_prose(
                prose=result.prose,
                world_state=ws,
                config=cfg.extraction_config,
                spawns=spawns,
                introduced_elements=getattr(
                    result.scene, "introduced_elements", None,
                ),
                branch_world_id=_world_id,
                branch_label=_branch_label,
                preceding_prose=_preceding_prose,
                engine_priors=engine_priors,
            )
            description = (
                f"Pipeline merge after {query.query_type} query"
                f" (audit={'converged' if result.converged else 'skipped/failed'})"
            )
            # Bridge sandbox-only physics deltas (do-surgery mutations,
            # social_mutations, abduction hidden_deltas) directly into
            # the topology so the merge persists them even if the
            # generated prose did not verbalise them clearly enough
            # for the extractor to re-discover.
            _spacing = (
                cfg.extraction_config.fabula_time_spacing
                if cfg.extraction_config is not None else 100
            )
            _ft_now = max(
                (e.fabula_time for e in ws.events),
                default=-_spacing,
            ) + _spacing
            # Rung-3 abduction returns ``past_anchor`` — the actual
            # Point-of-Divergence in fabula time. Use it so hidden
            # deltas land at the correct historical tick. Fall back
            # to the timeline minimum only when the physics result
            # didn't surface one (rung-1/2 paths).
            _ft_hist = physics_result.get("past_anchor")
            if _ft_hist is None:
                _ft_hist = min(
                    (e.fabula_time for e in ws.events),
                    default=0,
                )
            else:
                _ft_hist = int(_ft_hist)
            _augment_topology_with_sandbox_deltas(
                topology,
                world_state=ws,
                physics_result=physics_result,
                fabula_time_now=_ft_now,
                fabula_time_historical=_ft_hist,
                world_id=_world_id,
                branch_label=_branch_label,
                query_type=getattr(query, "query_type", None),
            )
            vwm_next = vwm.merge(
                topology,
                source="pipeline",
                description=description,
                prose=result.prose,
                world_id=_world_id,
                branch_label=_branch_label,
            )
            # Record the changeset
            changeset = vwm_next.history[-1].changeset
            history.record("reextraction_merge", ReextractionStepRecord(
                events_added=changeset.events_added if changeset else 0,
                causal_edges_added=changeset.causal_edges_added if changeset else 0,
                entity_updates_applied=changeset.entity_updates_applied if changeset else 0,
                entity_updates_skipped=changeset.entity_updates_skipped if changeset else [],
                new_version=vwm_next.version,
            ))
            # Continuation quality bridge.
            vwm_next = _run_continuation_quality_bridge_sync(
                vwm_next, cfg, result,
                log_prefix="[Pipeline·QualityBridge]",
            )
            result.world_model = vwm_next
            logger.info(
                "[Pipeline] Merge complete — v%d → v%d (+%d events).",
                vwm.version, vwm_next.version,
                changeset.events_added if changeset else 0,
            )
        except Exception as _e:
            logger.exception("[Pipeline] Re-extraction/merge failed — returning unmerged model.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})
            result.reextraction_failed = True
            result.reextraction_error = str(_e)

    if cfg.defer_reextraction and not cfg.skip_reextraction:
        # Stash the closure for ``finish_reextraction`` to drive later.
        # The caller (UI / MCP) gets prose immediately and runs the
        # heavy ingest in a background task so it can show the prose
        # to the user without blocking on extraction.
        result.reextraction_pending = True
        result._deferred_reextraction_fn = _do_steps_6_7
        logger.info(
            "[Pipeline] Steps 6–7 deferred — caller must invoke "
            "finish_reextraction(result) to complete the merge."
        )
    else:
        _do_steps_6_7()

    logger.info("[Pipeline] Complete — query_type=%s, prose=%s.",
                query.query_type, "yes" if result.prose else "no")
    return result


async def run_pipeline_async(
    query: UserRequest,
    *,
    world_state: WorldStateV1 | None = None,
    versioned_model: VersionedWorldModel | None = None,
    raw_text: str | None = None,
    config: PipelineConfig | None = None,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
) -> PipelineResult:
    """Async variant of :func:`run_pipeline`.

    Uses :func:`run_extraction_async` for parallel ingestion when
    ``raw_text`` is supplied.  All other stages are identical to the
    synchronous pipeline.
    """
    cfg = config or PipelineConfig()
    history = PipelineHistory()
    result = PipelineResult(query_type=query.query_type, history=history)

    # Round-5 audit: ``cfg.defer_reextraction`` requires a sync callable
    # closure (see ``finish_reextraction``) that an async caller cannot
    # drive without thread offload, and there is no
    # ``finish_reextraction_async`` parallel yet. Reject the combination
    # at the boundary instead of silently ignoring the flag (the prior
    # behaviour ran steps 6\u20137 inline anyway, blocking the prose
    # response on background ingest). Callers that need defer must use
    # the sync ``run_pipeline``.
    if cfg.defer_reextraction and not cfg.skip_reextraction:
        raise ValueError(
            "cfg.defer_reextraction=True is not supported by "
            "run_pipeline_async (no async finish_reextraction parallel "
            "exists). Use run_pipeline (sync) for deferred re-extraction, "
            "or set cfg.defer_reextraction=False to run steps 6\u20137 "
            "inline on the async pipeline."
        )

    # =================================================================
    # Step 0: Resolve the world model (async ingestion when raw_text)
    # =================================================================
    sources = sum([world_state is not None, versioned_model is not None, raw_text is not None])
    if sources == 0:
        raise ValueError("Supply one of: world_state, versioned_model, or raw_text.")
    if sources > 1:
        raise ValueError("Supply only one of: world_state, versioned_model, or raw_text.")

    if raw_text is not None:
        logger.info("[Pipeline·Async] Step 1: Ingesting raw text (%d chars).", len(raw_text))
        ing_cfg = cfg.ingestion_config or ExtractionConfig()
        # Tier 4 #14: capture validator/auto-fix warnings for this project.
        with capture_ingestion_warnings(str(project_id) if project_id is not None else "_anon"):
            ws, validation_report = await run_extraction_async(
                raw_text,
                config=ing_cfg,
                user_id=user_id,
                project_id=project_id,
                version_id=version_id,
            )
        vwm = VersionedWorldModel.from_world_state(ws, max_snapshots=cfg.max_snapshots)
        history.record("ingestion", IngestionStepRecord(
            is_valid=validation_report.is_valid,
            num_events=len(ws.events),
            num_entities=len(ws.entities),
            num_locations=len(ws.locations),
        ))
        logger.info(
            "[Pipeline·Async] Ingestion complete — %d events, %d entities, valid=%s.",
            len(ws.events), len(ws.entities), validation_report.is_valid,
        )
    elif versioned_model is not None:
        vwm = versioned_model
        ws = vwm.current
    else:
        vwm = VersionedWorldModel.from_world_state(world_state, max_snapshots=cfg.max_snapshots)
        ws = vwm.current

    result.world_model = vwm

    # Remaining steps are identical to sync pipeline — delegate
    # (physics, generation, audit, re-extraction are all sync)
    logger.info(
        "[Pipeline·Async] Step 2: Narrative physics — query_type=%s, causal_engine=%s.",
        query.query_type, cfg.use_causal_engine,
    )
    _active_branch_world_id = (
        vwm.history[-1].world_id if vwm.history else "factual"
    )
    _active_branch_label = (
        vwm.history[-1].branch_label if vwm.history else None
    )
    ws = _apply_query_introductions(
        ws, query, world_id=_active_branch_world_id,
    )
    # AMWN-split projection — mirrors run_pipeline (sync).
    ws = ws.projected_for_branch(
        branch_world_id=_active_branch_world_id,
        branch_label=_active_branch_label,
    )
    # Per-query surgery isolation — see run_pipeline (sync) for
    # rationale.
    ws = _isolate_ws_for_surgery(ws, query)
    eff_temporal, eff_syuzhet = _resolve_query_anchors(
        query, cfg.temporal_anchor, cfg.syuzhet_anchor, ws,
    )
    # CC-3 (2026-05-29): mirror sync pre-pass.
    try:
        from shadow_loom.world_schema_audit import (
            audit_world_schema as _audit_world_schema,
        )
        _pre_warnings = _audit_world_schema(ws)
        if _pre_warnings:
            history.record(
                "world_schema_audit",
                {
                    "warning_count": len(_pre_warnings),
                    "warnings": _pre_warnings[:50],
                    "truncated": len(_pre_warnings) > 50,
                },
            )
            logger.info(
                "[Pipeline\u00b7Async\u00b7schema\u00b7pre] %d "
                "world-schema warning(s) surfaced before physics.",
                len(_pre_warnings),
            )
    except Exception:
        logger.exception(
            "[Pipeline\u00b7Async\u00b7schema\u00b7pre] World-schema "
            "pre-pass failed; continuing."
        )
    # Round-12 R12-01: ``calculate_narrative_physics`` is the heaviest
    # CPU-bound step in the pipeline (Monte-Carlo propagation, causal-
    # engine resolution, lattice sweeps). Calling it directly from the
    # async pipeline blocked the NiceGUI event loop for the duration
    # of the simulation — exactly the regression that R10 fixed for
    # render_and_audit / run_feedback_loop. Offload to a worker
    # thread so concurrent UI updates and websocket pongs keep
    # flowing while physics runs.
    physics_result = await asyncio.to_thread(
        calculate_narrative_physics,
        request=query,
        global_world_state=ws,
        temporal_anchor=eff_temporal,
        syuzhet_anchor=eff_syuzhet,
        use_causal_engine=cfg.use_causal_engine,
    )
    result.physics_result = physics_result

    common_keys = {"status", "query_type", "physics_state", "_causal_physics_result"}
    extras = {k: v for k, v in physics_result.items() if k not in common_keys}
    history.record("narrative_physics", PhysicsStepRecord(
        query_type=physics_result.get("query_type", query.query_type),
        status=physics_result.get("status", "unknown"),
        physics_state=physics_result.get("physics_state", {}),
        extra=extras,
    ))

    # Implausibility short-circuit (mirrors sync pipeline)
    if physics_result.get("status") == "implausible":
        _apply_implausibility_short_circuit(
            query=query, physics_result=physics_result,
            vwm=vwm, result=result, history=history,
            log_prefix="[Pipeline·Async]",
        )
        return result

    if "implausibility_warning" in physics_result:
        result.implausible = True
        result.implausibility_reason = physics_result.get("implausibility_warning")
        result.implausibility_details = physics_result.get("implausibility_details", {}) or {}
        history.record("implausibility", {
            "forced": True,
            "reason": result.implausibility_reason,
            "details": result.implausibility_details,
        })
        logger.warning(
            "[Pipeline·Async] Forced past implausibility gate — generating prose anyway: %s",
            result.implausibility_reason,
        )

    if query.query_type in ("interrogate", "general"):
        logger.info(
            "[Pipeline\u00b7Async] Query type '%s' \u2014 answering question over physics state.",
            query.query_type,
        )
        # AUDIT P0-7: mirror the sync interrogate path so async
        # callers also receive a deterministic posterior table.
        try:
            from shadow_loom.interrogate_posterior import (
                interrogate_posterior as _interrogate_posterior,
                audit_posterior_consistency as _audit_posterior_consistency,
            )
            # R1-2 (2026-05-29): mirror sync path — use resolved
            # ``eff_temporal`` not the non-existent
            # ``query.fabula_time`` attribute.
            _ft = eff_temporal
            _posterior = _interrogate_posterior(ws, fabula_time=_ft)
            physics_result["posterior"] = _posterior
            physics_result["posterior_warnings"] = (
                _audit_posterior_consistency(_posterior)
            )
        except Exception:
            logger.exception(
                "[Pipeline\u00b7Async\u00b7posterior] Failed to "
                "compute deterministic interrogate posterior."
            )
        # AUDIT P1-2: mirror the sync affective-scorer surfacing.
        try:
            from shadow_loom.affective_scorers import (
                compute_affective_scorers as _compute_affective_scorers,
            )
            physics_result["affective_scorers"] = _compute_affective_scorers(
                ws, fabula_time=eff_temporal,
            )
        except Exception:
            logger.exception(
                "[Pipeline\u00b7Async\u00b7affective] Failed to "
                "compute affective scorers."
            )
        # AUDIT round-2 P0: mirror constants surfacing in async path.
        try:
            from shadow_loom.interrogate_posterior import (
                surface_entity_constants as _surface_entity_constants,
            )
            physics_result["entity_constants"] = _surface_entity_constants(ws)
        except Exception:
            logger.exception(
                "[Pipeline\u00b7Async\u00b7constants] Failed to surface "
                "entity constants."
            )
        # CC-2 (2026-05-29): mirror the readonly-query schema audit
        # in the async path so MCP / UI surfaces see schema warnings
        # regardless of which pipeline orchestrator served the query.
        try:
            from shadow_loom.world_schema_audit import (
                audit_world_schema as _audit_world_schema,
            )
            _schema_warnings = _audit_world_schema(ws)
            physics_result["world_schema_warnings"] = list(_schema_warnings)
            if _schema_warnings:
                logger.info(
                    "[Pipeline\u00b7Async\u00b7schema] %d world-schema "
                    "warning(s) surfaced for readonly query (advisory).",
                    len(_schema_warnings),
                )
        except Exception:
            logger.exception(
                "[Pipeline\u00b7Async\u00b7schema] World-schema audit "
                "failed; continuing without warnings."
            )
        try:
            _run_answer_step(query=query, physics_result=physics_result, cfg=cfg, history=history, vwm=vwm)
        except Exception:
            logger.exception(
                "[Pipeline\u00b7Async] Answer step failed for "
                "query_type=%s \u2014 returning a degraded result with "
                "no answer prose.",
                query.query_type,
            )
            physics_result["answer_failed"] = True
            physics_result.setdefault(
                "answer",
                "(The model could not produce an answer for this "
                "question; please retry or rephrase.)",
            )
        return result

    # Pearl-Rung-2/3 queries also get a rung-aware answer card
    # (Phase-9) alongside the prose pipeline below.
    if query.query_type in ("intervention", "counterfactual"):
        logger.info(
            "[Pipeline\u00b7Async] Query type '%s' \u2014 surfacing rung-aware "
            "Q&A answer alongside prose.",
            query.query_type,
        )
        try:
            _run_answer_step(query=query, physics_result=physics_result, cfg=cfg, history=history, vwm=vwm)
        except Exception:
            logger.exception(
                "[Pipeline\u00b7Async] Rung-aware answer step failed; "
                "continuing with prose generation."
            )

    # Evaluation: full-story quality audit (delegates to shared sync helper)
    if query.query_type == "evaluate":
        _run_evaluation_branch(
            query=query, ws=ws, vwm=vwm, cfg=cfg,
            physics_result=physics_result, result=result, history=history,
        )
        return result

    # Manual edit: same as sync path
    if query.query_type == "manual_edit":
        logger.info("[Pipeline·Async] Manual edit — skipping generation, running re-extraction.")
        result.prose = query.edited_prose
        from shadow_loom.generation import GeneratedScene
        result.scene = GeneratedScene(
            prose=query.edited_prose,
            rendering_mode="manual_edit",
        )
        history.record("generation", GenerationStepRecord(scene=result.scene, brief=None))
        try:
            anchor_base = _resolve_manual_edit_anchor(query, ws, cfg)
            ws_for_extract = _apply_manual_edit_replacements(ws, query)
            _world_id, _branch_label = _resolve_branch_policy(query, cfg, vwm)
            _preceding_prose = _gather_preceding_prose(
                vwm,
                branch_world_id=_world_id,
                branch_label=_branch_label,
            )
            topology = await asyncio.to_thread(
                extract_topology_from_prose,
                prose=result.prose, world_state=ws_for_extract,
                config=cfg.extraction_config,
                fabula_time_base=anchor_base,
                branch_world_id=_world_id,
                branch_label=_branch_label,
                preceding_prose=_preceding_prose,
            )
            _populate_removed_fields_from_manual_edit(topology, query)
            description = f"Manual edit: {query.description}" if query.description else "Manual edit"
            vwm_for_merge = (
                vwm.model_copy(update={"current": ws_for_extract})
                if (
                    query.replace_event_ids or query.replace_entity_ids
                    or query.replace_object_ids or query.replace_location_ids
                    or query.replace_world_trait_ids or query.replace_channel_ids
                    or query.replace_proposition_ids or query.replace_concern_ids
                )
                else vwm
            )
            vwm_next = vwm_for_merge.merge(
                topology, source="manual_edit", description=description, prose=result.prose,
                world_id=_world_id, branch_label=_branch_label,
            )
            changeset = vwm_next.history[-1].changeset
            history.record("reextraction_merge", ReextractionStepRecord(
                events_added=changeset.events_added if changeset else 0,
                causal_edges_added=changeset.causal_edges_added if changeset else 0,
                entity_updates_applied=changeset.entity_updates_applied if changeset else 0,
                entity_updates_skipped=changeset.entity_updates_skipped if changeset else [],
                new_version=vwm_next.version,
            ))
            # Continuation quality bridge (async).
            vwm_next = await _run_continuation_quality_bridge_async(
                vwm_next, cfg, result,
                log_prefix="[Pipeline·Async·ManualEdit·QualityBridge]",
            )
            result.world_model = vwm_next
        except Exception as _e:
            logger.exception("[Pipeline·Async] Manual edit re-extraction/merge failed.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})
            result.reextraction_failed = True
            result.reextraction_error = str(_e)
        return result

    # Steps 3–4: Brief + Generation (same as sync)
    physics_state = physics_result.get("physics_state", {})
    _branch_world_id, _branch_label = _resolve_branch_policy(query, cfg, vwm)
    _preceding_prose = _gather_preceding_prose(
        vwm,
        branch_world_id=_branch_world_id,
        branch_label=_branch_label,
    )
    _factual_contrast = _compute_factual_contrast(
        vwm,
        branch_world_id=_branch_world_id,
        branch_label=_branch_label,
    )
    brief: CreativeBrief | None = None
    if query.query_type == "directive" and "creative_brief" in physics_result:
        brief_data = physics_result["creative_brief"]
        brief = CreativeBrief(**brief_data) if isinstance(brief_data, dict) else brief_data
        _stamp_brief_full(
            brief, vwm,
            branch_world_id=_branch_world_id,
            branch_label=_branch_label,
            factual_contrast=_factual_contrast,
        )

    if cfg.skip_audit:
        gen_cfg = cfg.generation_config or GenerationConfig()
        # AUDIT P0-7: offload synchronous LLM-backed render to a thread
        # so the async pipeline doesn't block the event loop.
        scene = await asyncio.to_thread(
            render_from_query,
            query, physics_result, ws, gen_cfg,
            preceding_prose=_preceding_prose,
            branch_world_id=_branch_world_id,
            branch_label=_branch_label,
            factual_contrast_summary=_factual_contrast,
            syuzhet_anchor=eff_syuzhet,
        )
        history.record("generation", GenerationStepRecord(scene=scene, brief=brief))
        result.scene = scene
        result.prose = scene.prose
    else:
        if brief is not None:
            # Reconstruct an assembler so engine-side affective metrics
            # (trajectory_scores, KL, affective_loss) flow into the
            # per-cycle audit. Mirrors the sync ``run_pipeline`` branch
            # — without this the async loop loses the affective half of
            # ``ChangeImpactMetrics``.
            _assembler = None
            if query.query_type == "directive":
                try:
                    from shadow_loom.extract_graph import extract_ego_graph_from_memory
                    _ego = extract_ego_graph_from_memory(
                        ws, brief.target_entities,
                        syuzhet_anchor=eff_syuzhet,
                        shadow_path_seed_ids=set(brief.target_entities or []),
                    )
                    _assembler = DirectiveAssembler(
                        sandbox=None, ego_payload=_ego.model_dump(), world_state=ws,
                    )
                except Exception:
                    logger.warning(
                        "[Pipeline·Async] Could not rebuild DirectiveAssembler "
                        "for engine metrics; affective feedback will be missing "
                        "from this query's audit.",
                        exc_info=True,
                    )

            # Round-10 R10-09: render_and_audit is a long-running
            # synchronous LLM call. Inside run_pipeline_async we MUST
            # offload it to a worker thread or it blocks the event
            # loop for the duration of generation+audit, stalling
            # every other request on the NiceGUI/MCP server.
            feedback = await asyncio.to_thread(
                render_and_audit,
                brief=brief, world_state=ws,
                auditor_config=_inert_aware_auditor_config(cfg.auditor_config, physics_result),
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
                physics_result=physics_result.get("_causal_physics_result"),
                assembler=_assembler,
            )
        else:
            gen_cfg = cfg.generation_config or GenerationConfig()
            initial_scene = await asyncio.to_thread(
                render_from_query,
                query, physics_result, ws, gen_cfg,
                preceding_prose=_preceding_prose,
                branch_world_id=_branch_world_id,
                branch_label=_branch_label,
                factual_contrast_summary=_factual_contrast,
                syuzhet_anchor=eff_syuzhet,
            )
            brief = _build_brief_for_query(query, physics_result, ws, syuzhet_anchor=eff_syuzhet)
            _stamp_brief_full(
                brief, vwm,
                branch_world_id=_branch_world_id,
                branch_label=_branch_label,
                factual_contrast=_factual_contrast,
            )
            from shadow_loom.auditor import run_feedback_loop
            # Round-10 R10-09: same reasoning as above — keep the
            # event loop responsive while the auditor iterates.
            feedback = await asyncio.to_thread(
                run_feedback_loop,
                initial_scene=initial_scene, brief=brief, world_state=ws,
                auditor_config=_inert_aware_auditor_config(cfg.auditor_config, physics_result),
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
                physics_result=physics_result.get("_causal_physics_result"),
            )
        history.record("generation", GenerationStepRecord(scene=feedback.final_scene, brief=brief))
        history.record("audit", AuditStepRecord(feedback_result=feedback))
        result.scene = feedback.final_scene
        result.prose = feedback.final_scene.prose
        result.converged = feedback.converged
        result.audit_iterations = feedback.iterations
        result.feedback_result = feedback
        _log_feedback_outcome(feedback, async_path=True)

    # Steps 6–7: Re-extraction + merge (same as sync)
    # Round-5 audit: ``cfg.defer_reextraction`` is rejected at the top
    # of ``run_pipeline_async`` (see the ValueError guard there) because
    # no async ``finish_reextraction`` parallel exists; by the time we
    # reach this branch the flag is guaranteed False, so steps 6\u20137
    # run inline unconditionally.
    if cfg.skip_reextraction:
        logger.info("[Pipeline·Async] Steps 6–7: Re-extraction skipped.")
    elif (
        result.converged is False
        and not cfg.merge_on_audit_failure
    ):
        logger.error(
            "[Pipeline·Async] Steps 6–7: Re-extraction blocked — "
            "auditor did not converge (iterations=%s). Set "
            "PipelineConfig.merge_on_audit_failure=True to force.",
            result.audit_iterations,
        )
        result.reextraction_failed = True
        result.reextraction_error = (
            "Skipped re-extraction: auditor did not converge."
        )
        history.record(
            "reextraction_merge",
            {"error": "audit_not_converged"},
        )
    elif getattr(result.scene, "generation_error", None):
        logger.error(
            "[Pipeline·Async] Steps 6–7: Re-extraction skipped — scene "
            "carries generation_error=%s; refusing to merge fallback "
            "prose into world state.",
            result.scene.generation_error,
        )
        result.reextraction_failed = True
        result.reextraction_error = (
            f"Skipped re-extraction: generation failed "
            f"({result.scene.generation_error})."
        )
        history.record(
            "reextraction_merge",
            {"error": "generation_failure_skipped"},
        )
    else:
        logger.info("[Pipeline·Async] Steps 6–7: Extracting topology from prose and merging.")
        try:
            spawns = await asyncio.to_thread(promote_sandbox_spawns, ws, physics_state)
            _world_id, _branch_label = _resolve_branch_policy(query, cfg, vwm)
            engine_priors = _render_engine_priors(physics_result)
            topology = await asyncio.to_thread(
                extract_topology_from_prose,
                prose=result.prose, world_state=ws, config=cfg.extraction_config,
                spawns=spawns,
                introduced_elements=getattr(
                    result.scene, "introduced_elements", None,
                ),
                branch_world_id=_world_id,
                branch_label=_branch_label,
                preceding_prose=_preceding_prose,
                engine_priors=engine_priors,
            )
            description = (
                f"Pipeline merge after {query.query_type} query"
                f" (audit={'converged' if result.converged else 'skipped/failed'})"
            )
            _spacing = (
                cfg.extraction_config.fabula_time_spacing
                if cfg.extraction_config is not None else 100
            )
            _ft_now = max(
                (e.fabula_time for e in ws.events),
                default=-_spacing,
            ) + _spacing
            # Rung-3 abduction returns ``past_anchor`` — the actual
            # Point-of-Divergence in fabula time. Use it so hidden
            # deltas land at the correct historical tick. Fall back
            # to the timeline minimum only when the physics result
            # didn't surface one (rung-1/2 paths).
            _ft_hist = physics_result.get("past_anchor")
            if _ft_hist is None:
                _ft_hist = min(
                    (e.fabula_time for e in ws.events),
                    default=0,
                )
            else:
                _ft_hist = int(_ft_hist)
            _augment_topology_with_sandbox_deltas(
                topology,
                world_state=ws,
                physics_result=physics_result,
                fabula_time_now=_ft_now,
                fabula_time_historical=_ft_hist,
                world_id=_world_id,
                branch_label=_branch_label,
                query_type=getattr(query, "query_type", None),
            )
            vwm_next = vwm.merge(
                topology, source="pipeline", description=description, prose=result.prose,
                world_id=_world_id, branch_label=_branch_label,
            )
            changeset = vwm_next.history[-1].changeset
            history.record("reextraction_merge", ReextractionStepRecord(
                events_added=changeset.events_added if changeset else 0,
                causal_edges_added=changeset.causal_edges_added if changeset else 0,
                entity_updates_applied=changeset.entity_updates_applied if changeset else 0,
                entity_updates_skipped=changeset.entity_updates_skipped if changeset else [],
                new_version=vwm_next.version,
            ))
            # Continuation quality bridge (async).
            vwm_next = await _run_continuation_quality_bridge_async(
                vwm_next, cfg, result,
                log_prefix="[Pipeline·Async·QualityBridge]",
            )
            result.world_model = vwm_next
        except Exception as _e:
            logger.exception("[Pipeline·Async] Re-extraction/merge failed.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})
            result.reextraction_failed = True
            result.reextraction_error = str(_e)

    logger.info("[Pipeline·Async] Complete — query_type=%s, prose=%s.",
                query.query_type, "yes" if result.prose else "no")
    return result


# =====================================================================
# Internal: Q&A answer step for general / interrogate queries
# =====================================================================

def _run_answer_step(
    *,
    query: "UserRequest",
    physics_result: Dict[str, Any],
    cfg: "PipelineConfig",
    history: "PipelineHistory",
    vwm: Optional[VersionedWorldModel] = None,
) -> None:
    """Call the LLM Q&A agent and stash the result on ``physics_result``.

    Mutates ``physics_result`` in place so downstream consumers
    (chat card, Answer panel, MCP) read the answer from the same
    structured dict they were already reading.
    """
    from shadow_loom.answer import answer_question

    question = (
        getattr(query, "question", None)
        or getattr(query, "original_query", None)
        or ""
    )
    require_proof = bool(getattr(query, "require_proof", False))
    qtype = getattr(query, "query_type", "general")

    # Resolve the active branch so the answer agent sees the same
    # branch framing the rest of the pipeline would have used. Without
    # this, general/interrogate questions asked while on a shadow
    # fork were answered against a branch-agnostic world slice.
    _branch_world_id, _branch_label = _resolve_branch_policy(query, cfg, vwm)
    _preceding_prose = _gather_preceding_prose(
        vwm,
        branch_world_id=_branch_world_id,
        branch_label=_branch_label,
    )
    _factual_contrast = _compute_factual_contrast(
        vwm,
        branch_world_id=_branch_world_id,
        branch_label=_branch_label,
    )

    card = answer_question(
        question=question,
        physics_state=physics_result.get("physics_state"),
        query_type=qtype,
        require_proof=require_proof,
        config=cfg.generation_config,
        branch_world_id=_branch_world_id,
        branch_label=_branch_label,
        factual_contrast_summary=_factual_contrast,
        preceding_prose=_preceding_prose,
        # Pass the source register so the Q&A LLM mirrors the same
        # tone / diction the renderer/auditor enforce on each chunk.
        # R19-H6: resolve via ``projected_for_branch`` so a shadow
        # branch carrying its own narrative_style (future per-branch
        # sidecar) is honoured instead of always reading the factual
        # baseline. Today narrative_style isn't branched so this is
        # a no-op, but the resolution path is now correct and a
        # future ``shadow_narrative_style`` sidecar will Just Work.
        narrative_style=(
            getattr(
                vwm.current.projected_for_branch(
                    _branch_world_id, _branch_label,
                ),
                "narrative_style",
                None,
            )
            if getattr(vwm, "current", None) is not None
            else None
        ),
        # Phase-9: forward typed Pearl-rung surgery metadata so the
        # intervention / counterfactual answer agents can match the
        # right epistemic / ontic register and apply the
        # narrative-form hedge. Stamped into ``physics_result`` by
        # ``narrative_physics._typed_target_payload`` (Phase-7).
        do_targets=(
            list(physics_result.get("do_targets") or [])
            or list(physics_result.get("historical_do_targets") or [])
            or None
        ),
        affected_propositions=list(physics_result.get("affected_propositions") or []) or None,
        affected_beliefs=list(physics_result.get("affected_beliefs") or []) or None,
        affected_concerns=list(physics_result.get("affected_concerns") or []) or None,
        affected_objects=list(physics_result.get("affected_objects") or []) or None,
        affected_world_traits=list(physics_result.get("affected_world_traits") or []) or None,
        affected_edges=list(physics_result.get("affected_edges") or []) or None,
        affected_entity_deletes=list(physics_result.get("affected_entity_deletes") or []) or None,
        affected_object_deletes=list(physics_result.get("affected_object_deletes") or []) or None,
        affected_events=list(physics_result.get("affected_events") or []) or None,
        tragedy_form=physics_result.get("tragedy_form"),
        # Phase-10: forward engine-emitted downstream cascades so the
        # intervention / counterfactual Q&A answer agent can enumerate
        # actual propagation rather than guess. (Interrogate keeps
        # these as ``None`` because the physics path does not run a
        # do-surgery there.)
        mutations=list(physics_result.get("mutations") or []) or None,
        social_mutations=list(physics_result.get("social_mutations") or []) or None,
        proposition_mutations=list(physics_result.get("proposition_mutations") or []) or None,
        belief_mutations=list(physics_result.get("belief_mutations") or []) or None,
        concern_mutations=list(physics_result.get("concern_mutations") or []) or None,
        # Round-5 audit: forward object / world-trait / edge mutation
        # streams so the Q&A answer renderer can enumerate prop,
        # ambient-force, and topology surgeries instead of dropping
        # them at the answer hand-off.
        object_mutations=list(physics_result.get("object_mutations") or []) or None,
        world_trait_mutations=list(physics_result.get("world_trait_mutations") or []) or None,
        edge_mutations=list(physics_result.get("edge_mutations") or []) or None,
        entity_delete_mutations=list(physics_result.get("entity_delete_mutations") or []) or None,
        object_delete_mutations=list(physics_result.get("object_delete_mutations") or []) or None,
        event_mutations=list(physics_result.get("event_mutations") or []) or None,
        blocked=list(physics_result.get("blocked") or []) or None,
        causal_chain=list(physics_result.get("causal_chain") or []) or None,
        # 2026-05-29 deep-audit HIGH-1: the pipeline computes these
        # deterministic diagnostics on every call (see writes to
        # ``physics_result["posterior_warnings"]`` and
        # ``physics_result["world_schema_warnings"]`` in
        # ``calculate_narrative_physics`` / its async sibling) but
        # they were never forwarded. Surface them so the answer
        # agent cannot contradict the engine's own audit.
        posterior_warnings=(
            list(physics_result.get("posterior_warnings") or []) or None
        ),
        world_schema_warnings=(
            list(physics_result.get("world_schema_warnings") or []) or None
        ),
    )

    physics_result["answer"] = card.answer
    physics_result["confidence"] = card.confidence
    physics_result["caveats"] = list(card.caveats)
    physics_result["evidence_node_ids"] = list(card.evidence_node_ids)
    # R19-M11: deterministic confidence caps. LLM-declared
    # confidence is uncalibrated and routinely contradicts the
    # physics signals the pipeline already collected: an inert
    # intervention (no engine mutations) cannot warrant high
    # confidence; a high blocked-to-total ratio means the engine
    # refused most of what was asked; zero grounded evidence ids
    # means no causal proof. Apply per-source caps so confidence
    # stays bounded above by the most pessimistic deterministic
    # signal, even if the LLM declared 0.95.
    try:
        _caps: List[float] = [1.0]
        # Cap 1: intervention inertness \u2014 if the engine reports
        # the do-targets had no propagation effect, cap confidence
        # at 0.4. ``physics_result['intervention_inert']`` is the
        # explicit signal; if absent fall back to "no mutations and
        # no blocked entries on a do-flavored query".
        if qtype in ("intervention", "counterfactual"):
            if physics_result.get("intervention_inert") is True:
                _caps.append(0.4)
            else:
                _mut_total = sum(
                    len(physics_result.get(k) or [])
                    for k in (
                        "mutations", "social_mutations",
                        "proposition_mutations", "belief_mutations",
                        "concern_mutations", "object_mutations",
                        "world_trait_mutations", "edge_mutations",
                        "entity_delete_mutations",
                        "object_delete_mutations",
                        "event_mutations",
                    )
                )
                _blocked = len(physics_result.get("blocked") or [])
                if _mut_total == 0 and _blocked == 0:
                    _caps.append(0.4)
                elif _blocked and (_mut_total + _blocked) > 0:
                    # Cap 2: high blocked ratio \u2014 0.6 cap when
                    # half or more of the requested surgeries were
                    # refused by the engine.
                    if (_blocked / float(_mut_total + _blocked)) >= 0.5:
                        _caps.append(0.6)
        # Cap 3: zero grounded evidence \u2014 cap at 0.5. This
        # mirrors ``require_proof`` semantics in an advisory mode:
        # answers with no evidence cannot be more confident than
        # half.
        if not card.evidence_node_ids:
            _caps.append(0.5)
        _confidence_cap = min(_caps)
        if float(card.confidence or 0.0) > _confidence_cap:
            _caveats = list(card.caveats)
            _caveats.append(
                f"Confidence capped to {_confidence_cap:.2f} by "
                f"deterministic physics signals (R19-M11)."
            )
            from shadow_loom.answer import AnswerCard as _AnswerCard
            card = _AnswerCard(
                answer=card.answer,
                confidence=_confidence_cap,
                caveats=_caveats,
                evidence_node_ids=list(card.evidence_node_ids),
            )
            physics_result["answer"] = card.answer
            physics_result["confidence"] = card.confidence
            physics_result["caveats"] = list(card.caveats)
    except Exception:
        # Best-effort: confidence calibration must never break
        # the answer flow.
        pass
    # ``proof`` is the field ``structured_response_data`` already
    # iterates for evidence rows; mirror evidence_node_ids onto it
    # so the chat card surfaces them without further changes.
    physics_result["proof"] = [
        {"id": nid, "kind": "evidence"}
        for nid in card.evidence_node_ids
    ]

    # Enforce ``require_proof`` semantics: if the caller explicitly
    # demanded a causal proof (evidence-id grounding) and the answer
    # agent could not produce any, surface the failure rather than
    # silently returning an unsupported answer card. Without this
    # gate ``require_proof=True`` was advisory only — the LLM was
    # told "back claims with evidence" but the pipeline accepted
    # whatever came back even when no evidence ids were grounded.
    if require_proof and not card.evidence_node_ids:
        physics_result["status"] = "implausible"
        physics_result["implausibility_reason"] = "proof_required_but_unavailable"
        physics_result["implausibility_details"] = {
            "reason": "proof_required_but_unavailable",
            "require_proof": True,
            "evidence_node_ids": [],
            "message": (
                "Query specified require_proof=True but the answer agent "
                "did not produce any evidence node ids to ground its claims."
            ),
        }
        # Down-rank confidence so downstream consumers (renderer,
        # auditor, UI) do not surface the unbacked answer as
        # high-confidence.
        physics_result["confidence"] = min(
            float(card.confidence or 0.0), 0.1,
        )
        # Preface the caveat list so the chat card / Answer panel
        # always shows the proof-unavailable banner first.
        caveats = list(physics_result.get("caveats") or [])
        caveats.insert(
            0, "Proof required but no supporting evidence ids returned.",
        )
        physics_result["caveats"] = caveats

    history.record("answer", {
        "query_type": qtype,
        "question": question[:200],
        "answer_len": len(card.answer or ""),
        "confidence": card.confidence,
        "evidence_count": len(card.evidence_node_ids),
        "caveats": card.caveats,
    })


# =====================================================================
# Internal: full-story evaluation branch (shared by sync + async pipelines)
# =====================================================================

def _run_evaluation_branch(
    *,
    query: "EvaluationQuery",
    ws: WorldStateV1,
    vwm: "VersionedWorldModel",
    cfg: "PipelineConfig",
    physics_result: Dict[str, Any],
    result: "PipelineResult",
    history: "PipelineHistory",
) -> None:
    """Run a full-story quality audit and populate ``result.evaluation_result``."""
    logger.info("[Pipeline] Evaluation query — running full-story quality audit.")
    # Resolve branch policy FIRST so prose collection can be branch-scoped.
    # Without this, a shadow-branch evaluation pulls factual prose into
    # the scorecard (and vice versa), conflating canon with counterfactual.
    _eval_branch_id, _eval_branch_label = _resolve_branch_policy(
        query, cfg, vwm,
    )
    # Collect all prose from versioned model history — branch-scoped.
    all_prose_parts: list[str] = []
    if vwm.history:
        for entry in vwm.history:
            entry_world_id = getattr(entry, "world_id", None)
            # Treat unset world_id as factual (legacy rows predate the tag).
            if entry_world_id is None:
                entry_world_id = "factual"
            if entry_world_id != _eval_branch_id:
                continue
            # When evaluating a specific shadow label, isolate it from
            # other shadow branches on the same vwm.
            if (
                _eval_branch_id == "shadow"
                and _eval_branch_label is not None
                and getattr(entry, "branch_label", None) != _eval_branch_label
            ):
                continue
            entry_prose = getattr(entry, "prose", None)
            if entry_prose:
                all_prose_parts.append(entry_prose)
    # If no history prose, fall back to event-summary text
    if not all_prose_parts:
        events_summary = "; ".join(
            f"[{e.id}] {e.description}" for e in ws.events[:50]
        )
        all_prose_parts.append(f"Story events summary: {events_summary}")
    full_prose = "\n\n---\n\n".join(all_prose_parts)

    from shadow_loom.extract_graph import extract_ego_graph_from_memory
    # Per EvaluationQuery contract: empty focus_entity_ids means "score
    # the whole cast". The prior ``[:5]`` cap silently truncated the
    # ego graph for any story with more than five entities, biasing
    # the evaluator toward whatever five keys happened to come first.
    focus_ids = list(query.focus_entity_ids or ws.entities.keys())
    ego_graph = extract_ego_graph_from_memory(ws, focus_ids)
    eval_assembler = DirectiveAssembler(
        sandbox=None, ego_payload=ego_graph.model_dump(), world_state=ws,
    )

    from shadow_loom.generation import _user_intent_constraints
    _nl = getattr(query, "original_query", None)
    eval_brief = CreativeBrief(
        target_effect="observation",
        target_entities=focus_ids,
        original_query=_nl,
        constraints=_user_intent_constraints(_nl),
        scene_context=physics_result.get("physics_state", {}),
        # Carry the source-text register so the evaluator grades the
        # combined prose against the same fidelity contract the
        # renderer/auditor enforce on each chunk. Without this the
        # full-story scorecard is style-blind even when every chunk
        # was rendered with a STYLE FIDELITY block.
        narrative_style=getattr(ws, "narrative_style", None),
        # Tag the branch so a shadow-branch evaluation is not silently
        # graded as if it were factual canon.
        branch_world_id=_eval_branch_id,
        branch_label=_eval_branch_label,
    )
    # A6 (2026-05-29 tenth-pass audit): forward the query's
    # ``syuzhet_anchor`` to the epistemic-gap and trait-trajectory
    # computations so the evaluator sees the same reader-position
    # frame the directive assembler uses. Without this, beliefs and
    # trait snapshots from the entire timeline (including future
    # events the reader has not yet encountered) inflate gap
    # magnitudes and distort reader-effect metrics.
    _eval_anchor = getattr(query, "syuzhet_anchor", None)
    eval_brief.epistemic_gaps = eval_assembler.compute_epistemic_gaps(
        focus_ids, syuzhet_anchor=_eval_anchor,
    )
    eval_brief.narrative_tensions = eval_assembler.compute_narrative_tension()
    eval_brief.trait_trajectories = eval_assembler.compute_trait_trajectories(
        focus_ids, syuzhet_anchor=_eval_anchor,
    )

    causal_fb = compute_causal_feedback(None, eval_brief, ws)
    affective_fb = compute_affective_feedback(eval_brief, eval_assembler, focus_ids)

    from shadow_loom.auditor import AuditorConfig as _AC
    eval_cfg = cfg.auditor_config or _AC()
    narrative_order = _finalize_narrative_order(
        prose=full_prose,
        brief=eval_brief,
        causal_feedback=causal_fb,
        affective_feedback=affective_fb,
        config=eval_cfg,
    )

    eval_result = EvaluationResult(
        narrative_order=narrative_order,
        story_prose_evaluated=full_prose if query.include_full_prose else "",
        version_count=len(all_prose_parts),
    )
    result.evaluation_result = eval_result
    result.prose = full_prose

    history.record("evaluation", {
        "overall_pass": narrative_order.overall_pass,
        "foreshadowing_score": narrative_order.causal_feedback.foreshadowing_payoff_score,
        "affective_loss": narrative_order.affective_feedback.affective_loss_mse,
        "miracle_steps": len(narrative_order.causal_feedback.miracle_steps_detected),
    })


# =====================================================================
# Internal: build a brief for non-directive queries
# =====================================================================

def _build_brief_for_query(
    query: UserRequest,
    physics_result: Dict[str, Any],
    world_state: WorldStateV1,
    syuzhet_anchor: Optional[int] = None,
) -> CreativeBrief:
    """Build a CreativeBrief from a non-directive query's physics result."""
    from shadow_loom.generation import (
        build_counterfactual_brief,
        build_intervention_brief,
        build_observation_brief,
    )

    physics_state = physics_result.get("physics_state", {})

    if query.query_type == "observation":
        return build_observation_brief(
            query, physics_state, world_state,
            syuzhet_anchor=syuzhet_anchor,
            skipped_interventions=physics_result.get("skipped_interventions"),
        )
    elif query.query_type == "intervention":
        return build_intervention_brief(
            query, physics_state, world_state,
            mutations=physics_result.get("mutations"),
            blocked=physics_result.get("blocked"),
            rule3_pruned_interventions=physics_result.get("rule3_pruned_interventions"),
            rule2_redundant_evidence=physics_result.get("rule2_redundant_evidence"),
            rule3_pruning_mode=physics_result.get("rule3_pruning_mode", "advisory"),
            pruned_utterance_event_ids=physics_result.get("pruned_utterance_event_ids"),
            disabled_channel_ids=physics_result.get("disabled_channel_ids"),
            skipped_interventions=physics_result.get("skipped_interventions"),
            # Sandbox-coverage payload (Rung-2). Mirrors the
            # counterfactual plumbing below so the renderer and
            # auditor see the typed do_target + every flipped
            # proposition / belief / concern under the do-surgery.
            affected_propositions=physics_result.get("affected_propositions"),
            affected_beliefs=physics_result.get("affected_beliefs"),
            affected_concerns=physics_result.get("affected_concerns"),
            affected_objects=physics_result.get("affected_objects"),
            affected_world_traits=physics_result.get("affected_world_traits"),
            affected_edges=physics_result.get("affected_edges"),
            affected_entity_deletes=physics_result.get("affected_entity_deletes"),
            affected_object_deletes=physics_result.get("affected_object_deletes"),
            # Phase-10 downstream cascade payload — engine-emitted
            # mutations/social/proposition/belief/concern records so
            # the InterventionBranch sent to the renderer carries the
            # actual propagation, not just ID lists.
            social_mutations=physics_result.get("social_mutations"),
            proposition_mutations=physics_result.get("proposition_mutations"),
            belief_mutations=physics_result.get("belief_mutations"),
            concern_mutations=physics_result.get("concern_mutations"),
            # Round-5 audit: object / world-trait / edge mutation
            # streams produced by ``_typed_target_payload`` were
            # silently dropped here, so prop relocations, ambient-
            # force shifts, and topology surgeries never reached the
            # renderer brief. Forward them so InterventionBranch's
            # object/world_trait/edge cascade detail rails populate.
            object_mutations=physics_result.get("object_mutations"),
            world_trait_mutations=physics_result.get("world_trait_mutations"),
            edge_mutations=physics_result.get("edge_mutations"),
            entity_delete_mutations=physics_result.get("entity_delete_mutations"),
            object_delete_mutations=physics_result.get("object_delete_mutations"),
            event_mutations=physics_result.get("event_mutations"),
            causal_chain=physics_result.get("causal_chain"),
            syuzhet_anchor=syuzhet_anchor,
            intervention_inert=bool(physics_result.get("intervention_inert")),
            intervention_inert_reason=physics_result.get("intervention_inert_reason"),
        )
    elif query.query_type == "counterfactual":
        return build_counterfactual_brief(
            query, physics_state, world_state,
            hidden_deltas=physics_result.get("hidden_deltas"),
            rule3_pruned_interventions=physics_result.get("rule3_pruned_interventions"),
            rule2_redundant_evidence=physics_result.get("rule2_redundant_evidence"),
            rule3_pruning_mode=physics_result.get("rule3_pruning_mode", "advisory"),
            pruned_utterance_event_ids=physics_result.get("pruned_utterance_event_ids"),
            disabled_channel_ids=physics_result.get("disabled_channel_ids"),
            skipped_interventions=physics_result.get("skipped_interventions"),
            # Sandbox-coverage payload (Phase-9): the renderer and
            # auditor render these as "AFFECTED PROPOSITIONS / BELIEFS /
            # CONCERNS" and "RUNG-3 SURGERY KIND" off the
            # CounterfactualBranch. Without plumbing them through here
            # both surfaces saw empty lists and the structural shadow
            # path was uncovered — only the NL ``simulated_outcome``
            # string carried the surgery target.
            affected_propositions=physics_result.get("affected_propositions"),
            affected_beliefs=physics_result.get("affected_beliefs"),
            affected_concerns=physics_result.get("affected_concerns"),
            affected_objects=physics_result.get("affected_objects"),
            affected_world_traits=physics_result.get("affected_world_traits"),
            affected_edges=physics_result.get("affected_edges"),
            affected_entity_deletes=physics_result.get("affected_entity_deletes"),
            affected_object_deletes=physics_result.get("affected_object_deletes"),
            # Typed Rung-3 surgery target list — forward so the brief
            # builder can populate ``CounterfactualBranch.do_target``
            # and the renderer's "RUNG-3 SURGERY KIND" /
            # precursor-attempt-survival guidance fires.
            historical_do_targets=physics_result.get("historical_do_targets"),
            # Phase-10 downstream cascade payload (mirrors intervention
            # rung above — without these the counterfactual prose
            # silently skips engine-propagated consequences and lands
            # too short).
            mutations=physics_result.get("mutations"),
            social_mutations=physics_result.get("social_mutations"),
            proposition_mutations=physics_result.get("proposition_mutations"),
            belief_mutations=physics_result.get("belief_mutations"),
            concern_mutations=physics_result.get("concern_mutations"),
            # Round-5 audit: forward Rung-3 object / world-trait /
            # edge mutation streams so the counterfactual brief
            # surfaces them in the renderer prompt (parallel to the
            # intervention rung above).
            object_mutations=physics_result.get("object_mutations"),
            world_trait_mutations=physics_result.get("world_trait_mutations"),
            edge_mutations=physics_result.get("edge_mutations"),
            entity_delete_mutations=physics_result.get("entity_delete_mutations"),
            object_delete_mutations=physics_result.get("object_delete_mutations"),
            event_mutations=physics_result.get("event_mutations"),
            blocked=physics_result.get("blocked"),
            causal_chain=physics_result.get("causal_chain"),
            syuzhet_anchor=syuzhet_anchor,
            intervention_inert=bool(physics_result.get("intervention_inert")),
            intervention_inert_reason=physics_result.get("intervention_inert_reason"),
        )
    elif query.query_type == "directive":
        # Reached when the causal engine is disabled (or otherwise
        # didn't surface a ``creative_brief``). Build a real
        # directive brief via DirectiveAssembler so the auditor sees
        # the same epistemic gaps / tensions / trajectories it would
        # have seen in the engine-on path — NOT a degenerate
        # ``observation`` brief.
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        target_entities = list(query.target_entity_ids or [])
        try:
            ego = extract_ego_graph_from_memory(
                world_state, target_entities, syuzhet_anchor=syuzhet_anchor,
                shadow_path_seed_ids=set(target_entities),
            )
            assembler = DirectiveAssembler(
                sandbox=None,
                ego_payload=ego.model_dump(),
                world_state=world_state,
            )
            return assembler.assemble(query, syuzhet_anchor=syuzhet_anchor)
        except Exception:
            logger.exception(
                "[Pipeline] DirectiveAssembler fallback failed — "
                "returning minimal directive brief."
            )
            from shadow_loom.generation import _user_intent_constraints
            nl = getattr(query, "original_query", None)
            return CreativeBrief(
                target_effect=query.target_effect,
                target_entities=target_entities,
                original_query=nl,
                constraints=_user_intent_constraints(nl),
                scene_context=physics_state,
                narrative_style=getattr(world_state, "narrative_style", None),
            )
    else:
        # Fallback minimal brief
        from shadow_loom.generation import _user_intent_constraints
        nl = getattr(query, "original_query", None)
        return CreativeBrief(
            target_effect="observation",
            target_entities=[],
            original_query=nl,
            constraints=_user_intent_constraints(nl),
            scene_context=physics_state,
            narrative_style=getattr(world_state, "narrative_style", None),
        )


# =====================================================================
# Internal: implausibility short-circuit
# =====================================================================

def _format_implausibility_explanation(
    query: UserRequest,
    physics_result: Dict[str, Any],
) -> str:
    """Render a human-readable explanation when the engine deemed the
    query implausible.  No LLM call — purely template-based so the
    behaviour is deterministic, fast, and free of side-effects."""
    reason = physics_result.get("implausibility_reason", "Unknown reason.")
    details = physics_result.get("implausibility_details", {}) or {}
    unresolved = details.get("unresolved_targets", []) or []

    lines: List[str] = [
        f"The requested {query.query_type} could not be applied to the "
        "current world state, so the story, world model, and graph have "
        "been left unchanged.",
        "",
        f"Reason: {reason}",
    ]
    if unresolved:
        lines.append("")
        lines.append("Unresolved targets:")
        for item in unresolved:
            tgt = item.get("target", "?")
            why = item.get("reason", "unspecified")
            lines.append(f"  - {tgt}: {why}")
    lines.append("")
    lines.append(
        "Please revise the query so it references entities, events, "
        "objects, locations, or world traits that exist in the current "
        "state, then submit it again."
    )
    return "\n".join(lines)


def _apply_implausibility_short_circuit(
    *,
    query: UserRequest,
    physics_result: Dict[str, Any],
    vwm: VersionedWorldModel,
    result: PipelineResult,
    history: PipelineHistory,
    log_prefix: str,
) -> None:
    """Populate ``result`` with an explanation and bail out cleanly.

    Skips generation, audit, re-extraction, and merge.  The world model
    is left exactly as supplied (``vwm`` is *not* advanced).
    """
    from shadow_loom.generation import GeneratedScene

    explanation = _format_implausibility_explanation(query, physics_result)
    result.prose = explanation
    result.scene = GeneratedScene(
        prose=explanation,
        rendering_mode="implausible",
        constraints_violated=["implausible_query"],
    )
    result.implausible = True
    result.implausibility_reason = physics_result.get("implausibility_reason")
    result.implausibility_details = physics_result.get("implausibility_details", {}) or {}
    result.world_model = vwm  # explicitly unchanged

    history.record("generation", GenerationStepRecord(scene=result.scene, brief=None))
    history.record("implausibility", {
        "reason": result.implausibility_reason,
        "details": result.implausibility_details,
    })
    logger.info(
        "%s Implausible query — short-circuited (world model left at v%d).",
        log_prefix, vwm.version,
    )
