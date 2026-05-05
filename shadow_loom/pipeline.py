# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Complete end-to-end pipeline orchestrator for Shadow-Loom.

Connects all pipeline stages into a single ``run_pipeline()`` entry point:

  1. **Ingestion** (optional) — raw text → ``WorldStateV1``
  2. **Narrative Physics** — query routing, ego-graph, sandbox, causal engine
  3. **Brief Assembly** — ``CreativeBrief`` from physics result
  4. **Generation** — prose rendering (Step 10)
  5. **Audit + Refinement** — recursive feedback loop (Steps 11–12)
  6. **Prose Re-extraction** — extract topology from generated prose
  7. **Merge** — merge topology into a versioned deep-copy of the world model

The pipeline is flexible:
  • An existing ``WorldStateV1`` can be passed in (skips ingestion).
  • A ``VersionedWorldModel`` can be passed in to continue an existing history.
  • Non-prose query types (interrogate, general) skip steps 3–7.
  • Every intermediate result is recorded in ``PipelineHistory``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from shadow_loom.settings import get_settings as _get_settings

from shadow_loom.auditor import (
    AuditorConfig,
    FeedbackLoopResult,
    compute_causal_feedback,
    compute_affective_feedback,
    _finalize_narrative_order,
    render_and_audit,
)
from shadow_loom.directive_assembly import CreativeBrief, DirectiveAssembler
from shadow_loom.extract_graph import (
    VersionedWorldModel,
    extract_topology_from_prose,
    promote_sandbox_spawns,
)
from shadow_loom.generation import (
    GeneratedScene,
    GenerationConfig,
    render_from_query,
)
from shadow_loom.ingestion import ExtractionConfig, run_extraction, run_extraction_async
from shadow_loom.models import WorldStateV1
from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import UserRequest, EvaluationQuery, EvaluationResult, ManualEditQuery

logger = logging.getLogger(__name__)


# =====================================================================
# Anchor resolution helper
# =====================================================================

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
) -> tuple[Literal["factual", "shadow"], Optional[str]]:
    """Map ``cfg.branch_policy`` + ``query.query_type`` onto the AMWN
    ``world_id`` and ``branch_label`` to attach to the merged version.

    Policy table (Story-integration plan, Step 1):
      * ``"auto"`` (default): counterfactual queries route to a shadow
        fork; every other generative query (observation, intervention,
        directive, manual_edit) lands on the factual mainline.
      * ``"mainline"``: force factual mainline regardless of query type.
      * ``"shadow"``: force a shadow fork regardless of query type.

    ``branch_label`` is derived from ``query.description`` when forking
    onto a shadow branch, otherwise ``None``.
    """
    policy = cfg.branch_policy
    if policy == "shadow":
        world_id: Literal["factual", "shadow"] = "shadow"
    elif policy == "mainline":
        world_id = "factual"
    else:  # auto
        world_id = "shadow" if query.query_type == "counterfactual" else "factual"

    label: Optional[str] = None
    if world_id == "shadow":
        # Prefer the user's verbatim natural-language request (carried on
        # every query via _QueryBase.original_query); fall back to the
        # ManualEditQuery-only ``description`` field when present.
        candidate = getattr(query, "original_query", None) or getattr(query, "description", None)
        label = (candidate or "").strip() or None
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
        # Now collect factual ancestors from the rest of history
        for v in history_rev[idx:]:
            if v.prose and getattr(v, "world_id", "factual") == "factual":
                picked.append((v.version, v.prose, "factual", v.branch_label))

    if not picked:
        return None

    # Sort oldest \u2192 newest so the joined block reads in narrative order.
    picked.sort(key=lambda t: t[0])

    # Truncate from the front (drop oldest) until the joined block
    # fits the budget. Each block is wrapped with a small marker so
    # the LLM can tell continuity from the active scene's task.
    blocks: list[str] = []
    for version, prose, wid, label in picked:
        marker = f"--- v{version} ({wid}"
        if label:
            marker += f": {label}"
        marker += ") ---"
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
    ws: WorldStateV1, replace_event_ids: List[str],
) -> WorldStateV1:
    """Return a deep-copy of ``ws`` with the given events (and their
    dependent edges) removed, for *replace* manual-edit semantics.

    This runs **before** the new prose is re-extracted so the LLM
    doesn't see the events it's about to replace as "previous events"
    and try to keep them stitched in.
    """
    if not replace_event_ids:
        return ws
    drop = set(replace_event_ids)
    new_ws = ws.model_copy(deep=True)
    new_ws.events = [e for e in new_ws.events if e.id not in drop]
    new_ws.causal_topology = [
        ce for ce in new_ws.causal_topology
        if ce.source_id not in drop and ce.target_id not in drop
    ]
    new_ws.social_topology = [
        re for re in new_ws.social_topology
        if not getattr(re, "evidence_event_ids", None)
        or not (set(re.evidence_event_ids) & drop)
    ]
    new_ws.spatial_topology = [
        se for se in new_ws.spatial_topology
        if not getattr(se, "established_by_event_id", None)
        or se.established_by_event_id not in drop
    ]
    return new_ws


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
        ws, validation_report = run_extraction(
            raw_text, 
            config=ing_cfg, 
            user_id=user_id, 
            project_id=project_id, 
            version_id=version_id
        )
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
    eff_temporal, eff_syuzhet = _resolve_query_anchors(
        query, cfg.temporal_anchor, cfg.syuzhet_anchor, ws,
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

    # =================================================================
    # Implausibility short-circuit — explain, don't mutate.
    # =================================================================
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
        _run_answer_step(query=query, physics_result=physics_result, cfg=cfg, history=history)
        return result

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
            ws_for_extract = _apply_manual_edit_replacements(
                ws, query.replace_event_ids,
            )
            topology = extract_topology_from_prose(
                prose=result.prose,
                world_state=ws_for_extract,
                config=cfg.extraction_config,
                fabula_time_base=anchor_base,
            )
            description = (
                f"Manual edit: {query.description}"
                if query.description
                else "Manual edit"
            )
            _world_id, _branch_label = _resolve_branch_policy(query, cfg)
            # If the user asked for replace semantics, start the merge
            # from the *trimmed* world so the dropped events don't
            # come back via deep-copy of ``vwm.current``.
            vwm_for_merge = (
                vwm.model_copy(update={"current": ws_for_extract})
                if query.replace_event_ids
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
    _branch_world_id, _branch_label = _resolve_branch_policy(query, cfg)

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

    # For directive queries with causal engine, the brief is already built
    brief: CreativeBrief | None = None
    if query.query_type == "directive" and "creative_brief" in physics_result:
        brief_data = physics_result["creative_brief"]
        brief = CreativeBrief(**brief_data) if isinstance(brief_data, dict) else brief_data
        _stamp_brief_branch(brief, _branch_world_id, _branch_label)
        if _preceding_prose and not brief.preceding_prose:
            brief.preceding_prose = _preceding_prose

    if cfg.skip_audit:
        # Generate once, no audit loop
        logger.info("[Pipeline] Steps 3–4: Generating prose (audit skipped).")
        gen_cfg = cfg.generation_config or GenerationConfig()
        scene = render_from_query(
            query, physics_result, ws, gen_cfg,
            preceding_prose=_preceding_prose,
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
                    _ego = extract_ego_graph_from_memory(ws, brief.target_entities, syuzhet_anchor=eff_syuzhet)
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
                auditor_config=cfg.auditor_config,
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
            )

            # Build a brief for the auditor from the query
            brief = _build_brief_for_query(query, physics_result, ws, syuzhet_anchor=eff_syuzhet)
            _stamp_brief_branch(brief, _branch_world_id, _branch_label)
            if _preceding_prose and not brief.preceding_prose:
                brief.preceding_prose = _preceding_prose

            from shadow_loom.auditor import run_feedback_loop
            feedback = run_feedback_loop(
                initial_scene=initial_scene,
                brief=brief,
                world_state=ws,
                auditor_config=cfg.auditor_config,
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

    # =================================================================
    # Steps 6–7: Prose re-extraction + merge
    # =================================================================
    if cfg.skip_reextraction:
        logger.info("[Pipeline] Steps 6–7: Re-extraction skipped.")
    elif getattr(result.scene, "generation_error", None):
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
    else:
        logger.info("[Pipeline] Steps 6–7: Extracting topology from prose and merging.")
        try:
            spawns = promote_sandbox_spawns(ws, physics_state)
            topology = extract_topology_from_prose(
                prose=result.prose,
                world_state=ws,
                config=cfg.extraction_config,
                spawns=spawns,
            )
            description = (
                f"Pipeline merge after {query.query_type} query"
                f" (audit={'converged' if result.converged else 'skipped/failed'})"
            )
            _world_id, _branch_label = _resolve_branch_policy(query, cfg)
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
        ws, validation_report = await run_extraction_async(
            raw_text, 
            config=ing_cfg,
            user_id=user_id,
            project_id=project_id, 
            version_id=version_id
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
    eff_temporal, eff_syuzhet = _resolve_query_anchors(
        query, cfg.temporal_anchor, cfg.syuzhet_anchor, ws,
    )
    physics_result = calculate_narrative_physics(
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
        _run_answer_step(query=query, physics_result=physics_result, cfg=cfg, history=history)
        return result

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
            ws_for_extract = _apply_manual_edit_replacements(
                ws, query.replace_event_ids,
            )
            topology = extract_topology_from_prose(
                prose=result.prose, world_state=ws_for_extract,
                config=cfg.extraction_config,
                fabula_time_base=anchor_base,
            )
            description = f"Manual edit: {query.description}" if query.description else "Manual edit"
            _world_id, _branch_label = _resolve_branch_policy(query, cfg)
            vwm_for_merge = (
                vwm.model_copy(update={"current": ws_for_extract})
                if query.replace_event_ids
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
            result.world_model = vwm_next
        except Exception as _e:
            logger.exception("[Pipeline·Async] Manual edit re-extraction/merge failed.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})
            result.reextraction_failed = True
            result.reextraction_error = str(_e)
        return result

    # Steps 3–4: Brief + Generation (same as sync)
    physics_state = physics_result.get("physics_state", {})
    _branch_world_id, _branch_label = _resolve_branch_policy(query, cfg)
    _preceding_prose = _gather_preceding_prose(
        vwm,
        branch_world_id=_branch_world_id,
        branch_label=_branch_label,
    )
    brief: CreativeBrief | None = None
    if query.query_type == "directive" and "creative_brief" in physics_result:
        brief_data = physics_result["creative_brief"]
        brief = CreativeBrief(**brief_data) if isinstance(brief_data, dict) else brief_data
        _stamp_brief_branch(brief, _branch_world_id, _branch_label)
        if _preceding_prose and not brief.preceding_prose:
            brief.preceding_prose = _preceding_prose

    if cfg.skip_audit:
        gen_cfg = cfg.generation_config or GenerationConfig()
        scene = render_from_query(
            query, physics_result, ws, gen_cfg,
            preceding_prose=_preceding_prose,
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
                    _ego = extract_ego_graph_from_memory(ws, brief.target_entities, syuzhet_anchor=eff_syuzhet)
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

            feedback = render_and_audit(
                brief=brief, world_state=ws,
                auditor_config=cfg.auditor_config,
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
                physics_result=physics_result.get("_causal_physics_result"),
                assembler=_assembler,
            )
        else:
            gen_cfg = cfg.generation_config or GenerationConfig()
            initial_scene = render_from_query(
                query, physics_result, ws, gen_cfg,
                preceding_prose=_preceding_prose,
            )
            brief = _build_brief_for_query(query, physics_result, ws, syuzhet_anchor=eff_syuzhet)
            _stamp_brief_branch(brief, _branch_world_id, _branch_label)
            if _preceding_prose and not brief.preceding_prose:
                brief.preceding_prose = _preceding_prose
            from shadow_loom.auditor import run_feedback_loop
            feedback = run_feedback_loop(
                initial_scene=initial_scene, brief=brief, world_state=ws,
                auditor_config=cfg.auditor_config,
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

    # Steps 6–7: Re-extraction + merge (same as sync)
    if cfg.skip_reextraction:
        logger.info("[Pipeline·Async] Steps 6–7: Re-extraction skipped.")
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
            spawns = promote_sandbox_spawns(ws, physics_state)
            topology = extract_topology_from_prose(
                prose=result.prose, world_state=ws, config=cfg.extraction_config,
                spawns=spawns,
            )
            description = (
                f"Pipeline merge after {query.query_type} query"
                f" (audit={'converged' if result.converged else 'skipped/failed'})"
            )
            _world_id, _branch_label = _resolve_branch_policy(query, cfg)
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

    card = answer_question(
        question=question,
        physics_state=physics_result.get("physics_state"),
        query_type=qtype,
        require_proof=require_proof,
        config=cfg.generation_config,
    )

    physics_result["answer"] = card.answer
    physics_result["confidence"] = card.confidence
    physics_result["caveats"] = list(card.caveats)
    physics_result["evidence_node_ids"] = list(card.evidence_node_ids)
    # ``proof`` is the field ``structured_response_data`` already
    # iterates for evidence rows; mirror evidence_node_ids onto it
    # so the chat card surfaces them without further changes.
    physics_result["proof"] = [
        {"id": nid, "kind": "evidence"}
        for nid in card.evidence_node_ids
    ]

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
    # Collect all prose from versioned model history
    all_prose_parts: list[str] = []
    if vwm.history:
        for entry in vwm.history:
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
    focus_ids = (
        query.focus_entity_ids
        if query.focus_entity_ids
        else list(ws.entities.keys())[:5]
    )
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
    )
    eval_brief.epistemic_gaps = eval_assembler.compute_epistemic_gaps(focus_ids)
    eval_brief.narrative_tensions = eval_assembler.compute_narrative_tension()
    eval_brief.trait_trajectories = eval_assembler.compute_trait_trajectories(focus_ids)

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
        return build_observation_brief(query, physics_state, world_state)
    elif query.query_type == "intervention":
        return build_intervention_brief(
            query, physics_state, world_state,
            mutations=physics_result.get("mutations"),
            blocked=physics_result.get("blocked"),
            rule3_pruned_interventions=physics_result.get("rule3_pruned_interventions"),
            rule2_redundant_evidence=physics_result.get("rule2_redundant_evidence"),
            rule3_pruning_mode=physics_result.get("rule3_pruning_mode", "advisory"),
        )
    elif query.query_type == "counterfactual":
        return build_counterfactual_brief(
            query, physics_state, world_state,
            hidden_deltas=physics_result.get("hidden_deltas"),
            rule3_pruned_interventions=physics_result.get("rule3_pruned_interventions"),
            rule2_redundant_evidence=physics_result.get("rule2_redundant_evidence"),
            rule3_pruning_mode=physics_result.get("rule3_pruning_mode", "advisory"),
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
