# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audit tab — NarrativeOrderObject scorecard, evaluation, audit loop replay.

Provides:
- Full-story evaluation with structured scorecard visualization
- Per-query audit history with convergence tracking
- Audit loop iteration replay (prose + violations per cycle)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent
from shadow_loom_ui.task_helpers import run_query_as_task
from shadow_loom_ui.reasoning_helpers import (
    directive_for_violation,
    violation_explanation,
)
from shadow_loom_ui.reasoning_viz import render_convergence_trajectory
from shadow_loom_ui.viz import (
    render_audit_passrate_pictorial,
    render_emotional_gauges,
    with_expand,
)
from shadow_loom_ui.viz_helpers import audit_passrate_data

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_audit_tab(state: AppState) -> None:
    """Build the Audit tab layout."""

    with ui.column().classes("w-full h-full p-6 gap-4 bg-slate-50"):
        # ── NL prompt for evaluation ──────────────────────────────
        with ui.row().classes("w-full gap-2 flex-wrap"):
            ui.chip(
                "Run full evaluation",
                icon="fact_check",
                on_click=lambda: state.emit(
                    StateEvent.QUERY_STARTED,
                    suggestion="Evaluate the story quality comprehensively",
                    query_type="evaluate",
                ),
            ).props("dense outline clickable color=primary")
            ui.chip(
                "Check for miracle steps",
                icon="warning",
                on_click=lambda: state.emit(
                    StateEvent.QUERY_STARTED,
                    suggestion="Are there any miracle steps or impossible state changes in the story?",
                    query_type="evaluate",
                ),
            ).props("dense outline clickable")
            ui.chip(
                "Evaluate character consistency",
                icon="psychology",
                on_click=lambda: state.emit(
                    StateEvent.QUERY_STARTED,
                    suggestion="Evaluate character consistency and cognitive plausibility",
                    query_type="evaluate",
                ),
            ).props("dense outline clickable")

        # ── Full Story Evaluation ─────────────────────────────────
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("fact_check", size="md", color="primary")
                ui.label("Full Story Evaluation").classes(
                    "text-lg font-semibold text-slate-800"
                )

            ui.label(
                "Run a comprehensive narrative quality evaluation across the entire world model."
            ).classes("text-sm text-slate-500")

            eval_container = ui.column().classes("w-full q-mt-sm")
            eval_status = ui.label("").classes("text-sm text-slate-600")

            async def _run_evaluation():
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return

                eval_status.set_text("Running evaluation…")
                try:
                    from shadow_loom.query_models import EvaluationQuery
                    query = EvaluationQuery()
                    result, _task = await run_query_as_task(
                        state,
                        label="Full-story evaluation",
                        kind="evaluate",
                        runner=lambda: asyncio.to_thread(
                            state.run_structured_query, query,
                        ),
                        summary_fn=lambda r: (r.summary if r else "") or "Done",
                    )
                    eval_status.set_text("")
                    _render_evaluation_result(eval_container, result)
                except Exception as e:
                    logger.exception("Evaluation failed")
                    eval_status.set_text(f"Error: {e}")

            ui.button(
                "Run Evaluation", icon="play_arrow", on_click=_run_evaluation
            ).props("unelevated color=primary no-caps").classes("rounded-lg shadow-sm")

        # ── Per-Query Audit History ────────────────────────
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            ui.label("Query Audit History").classes(
                "text-lg font-semibold text-slate-800"
            )
            history_container = ui.column().classes("w-full")

            def _refresh_history(**kw):
                history_container.clear()
                if not state.query_history:
                    with history_container:
                        ui.label("No queries yet. Use the command bar to ask questions.").classes(
                            "text-sm text-slate-400 italic"
                        )
                    return

                with history_container:
                    for i, result in enumerate(reversed(state.query_history)):
                        _render_query_audit_entry(i, result, state)

            _refresh_history()
            state.on(StateEvent.PIPELINE_RESULT, _refresh_history)
            state.on(StateEvent.PROJECT_LOADED, _refresh_history)


# =====================================================================
# Evaluation result renderer (NarrativeOrderObject scorecard)
# =====================================================================

def _render_evaluation_result(container, result: NLQueryResult) -> None:
    """Render the full-story evaluation with structured scorecard."""
    container.clear()

    if result.error:
        with container:
            ui.label(f"Error: {result.error}").classes("text-negative")
        return

    pr = result.pipeline_result
    if pr is None:
        with container:
            ui.label("No evaluation result.").classes(
                "text-sm text-slate-400 italic"
            )
        return

    with container:
        # Evaluation scorecard (NarrativeOrderObject)
        eval_result = getattr(pr, "evaluation_result", None)

        if eval_result:
            _render_scorecard(container, eval_result)
        elif pr.prose:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ):
                ui.label("Evaluation Report").classes(
                    "text-sm font-semibold text-slate-700"
                )
                ui.markdown(pr.prose)

        # Convergence info
        if pr.converged is not None:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ):
                status = "Converged" if pr.converged else "Did not converge"
                color = "positive" if pr.converged else "warning"
                with ui.row().classes("items-center gap-2"):
                    ui.badge(status, color=color)
                    ui.label(f"Audit iterations: {pr.audit_iterations}").classes(
                        "text-xs text-slate-500"
                    )


def _render_scorecard(container, eval_result) -> None:
    """Render NarrativeOrderObject scorecard with gauges and metrics."""
    with container:
        # Extract scores
        narrative_order = getattr(eval_result, "narrative_order", None)
        if narrative_order is None:
            ui.label("No scorecard data.").classes(
                "text-sm text-slate-400 italic"
            )
            return

        # Overall pass/fail
        overall = getattr(narrative_order, "overall_pass", None)
        if overall is not None:
            color = "positive" if overall else "negative"
            ui.badge(
                "PASS" if overall else "FAIL",
                color=color,
            ).classes("text-base font-semibold mb-3")

        # Causal feedback
        causal = getattr(narrative_order, "causal_feedback", None)
        if causal:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ):
                ui.label("Causal Metrics").classes(
                    "text-sm font-semibold text-slate-700 mb-2"
                )
                scores = {}
                if hasattr(causal, "foreshadowing_payoff_score"):
                    scores["Foreshadowing"] = causal.foreshadowing_payoff_score
                if hasattr(causal, "cognitive_plausibility_score"):
                    scores["Plausibility"] = causal.cognitive_plausibility_score
                if scores:
                    with_expand(
                        lambda h, s=scores: render_emotional_gauges(
                            s, height=h
                        ),
                        title="Causal metrics",
                        height="150px",
                    )

                # Miracle steps
                miracles = getattr(causal, "miracle_steps_detected", [])
                if miracles:
                    ui.label(f"⚠️ Miracle steps detected: {len(miracles)}").classes(
                        "text-negative mt-2"
                    )
                    for m in miracles[:5]:
                        ui.label(f"  • {m}").classes("text-xs text-negative")
                else:
                    ui.label("✓ No miracle steps").classes("text-positive mt-2")

        # Affective feedback
        affective = getattr(narrative_order, "affective_feedback", None)
        if affective:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ):
                ui.label("Affective Metrics").classes(
                    "text-sm font-semibold text-slate-700 mb-2"
                )
                scores = {}
                if hasattr(affective, "emotional_trajectory_scores") and affective.emotional_trajectory_scores:
                    scores.update(affective.emotional_trajectory_scores)
                if hasattr(affective, "affective_loss_mse"):
                    scores["Affective Loss"] = affective.affective_loss_mse
                if scores:
                    with_expand(
                        lambda h, s=scores: render_emotional_gauges(
                            s, height=h
                        ),
                        title="Affective metrics",
                        height="150px",
                    )

                if hasattr(affective, "kl_divergence_prediction_error"):
                    ui.label(
                        f"KL Divergence (surprise): {affective.kl_divergence_prediction_error:.3f}"
                    ).classes("text-xs text-slate-500 mt-2")

        # Quality synthesis
        quality = getattr(narrative_order, "quality_synthesis", None)
        if quality:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ):
                ui.label("Quality Synthesis").classes(
                    "text-sm font-semibold text-slate-700 mb-2"
                )
                if hasattr(quality, "coherence_and_consistency_review") and quality.coherence_and_consistency_review:
                    with ui.expansion("Coherence Review", icon="check_circle").props("dense"):
                        ui.markdown(quality.coherence_and_consistency_review)
                if hasattr(quality, "reward_hacking_diagnostics") and quality.reward_hacking_diagnostics:
                    with ui.expansion("Reward-Hacking Diagnostics", icon="warning").props("dense"):
                        ui.markdown(quality.reward_hacking_diagnostics)
                if hasattr(quality, "actionable_rewrite_directives") and quality.actionable_rewrite_directives:
                    with ui.expansion("Rewrite Directives", icon="edit_note").props("dense"):
                        ui.markdown(quality.actionable_rewrite_directives)


# =====================================================================
# Per-query audit entry with loop replay
# =====================================================================

def _render_query_audit_entry(index: int, result: NLQueryResult, state: AppState) -> None:
    """Render a single query's audit information with iteration replay."""
    pr = result.pipeline_result
    query_type = pr.query_type if pr else "unknown"
    converged = pr.converged if pr else None

    icon = "check_circle" if converged else "warning" if converged is False else "help"
    with ui.expansion(
        f"#{index + 1} — {query_type}",
        icon=icon,
    ).classes("w-full").props("dense"):
        if result.summary:
            # Summary is now multi-line lay-user text (humanize_pipeline_result).
            ui.markdown(result.summary).classes("text-sm text-slate-600")

        if result.error:
            ui.label(f"Error: {result.error}").classes("text-xs text-negative")

        if pr is not None:
            # Convergence
            if pr.converged is not None:
                status = "Converged" if pr.converged else "Did not converge"
                with ui.row().classes("items-center gap-2"):
                    color = "positive" if pr.converged else "warning"
                    ui.badge(status, color=color).props("dense")
                    ui.label(f"{pr.audit_iterations} iterations").classes("text-xs text-slate-500")

            # Engine threshold gate + achieved-vs-target affective intensity.
            # Sourced from the deterministic CausalPhysicsFeedback /
            # AffectiveStateFeedback the auditor attaches to every cycle
            # (independent of the LLM verdict).
            feedback = getattr(pr, "feedback_result", None)
            if feedback is not None:
                if feedback.engine_thresholds_passed is not None:
                    with ui.row().classes("items-center gap-2 mt-1"):
                        thresh_ok = feedback.engine_thresholds_passed
                        ui.badge(
                            "Quality thresholds: passed" if thresh_ok
                            else "Quality thresholds: failed",
                            color="positive" if thresh_ok else "negative",
                        ).props("dense")
                    if (not feedback.engine_thresholds_passed
                            and feedback.engine_threshold_failures):
                        with ui.column().classes("gap-0 mt-1 ml-2"):
                            for f in feedback.engine_threshold_failures[:5]:
                                ui.label(f"• {f}").classes(
                                    "text-xs text-negative"
                                )
                ci = getattr(feedback, "change_impact", None)
                parsed = result.parse_result.parsed if result.parse_result else None
                req_effect = getattr(parsed, "target_effect", None) if parsed else None
                req_intensity = getattr(parsed, "intensity", None) if parsed else None
                if ci is not None and ci.affective_feedback is not None:
                    af = ci.affective_feedback
                    scores = af.emotional_trajectory_scores or {}
                    if req_effect and req_effect in scores:
                        achieved = scores[req_effect]
                        with ui.row().classes("items-center gap-2 mt-1"):
                            ui.label("Achieved intensity:").classes(
                                "text-xs text-slate-500"
                            )
                            if req_intensity is not None:
                                gap = achieved - req_intensity
                                gap_color = (
                                    "positive" if abs(gap) <= 0.15
                                    else "warning"
                                )
                                ui.badge(
                                    f"{req_effect}: {achieved:.2f} "
                                    f"(asked {req_intensity:.2f}, gap {gap:+.2f})",
                                    color=gap_color,
                                ).props("dense")
                            else:
                                ui.badge(
                                    f"{req_effect}: {achieved:.2f}",
                                    color="primary",
                                ).props("dense")
                    if af.affective_loss_mse is not None:
                        ui.label(
                            f"Affective loss (distance from target): "
                            f"{af.affective_loss_mse:.3f}"
                        ).classes("text-xs text-slate-500")

            # Prose excerpt
            if pr.prose:
                with ui.expansion("Prose", icon="article").props("dense"):
                    ui.markdown(pr.prose[:500] + ("…" if len(pr.prose) > 500 else ""))

            # Audit loop replay (if feedback_result has history)
            feedback = getattr(pr, "feedback_result", None)
            if feedback and hasattr(feedback, "history") and feedback.history:
                # Collect all violations from audit cycles
                all_violations = []
                for cycle in feedback.history:
                    audit = getattr(cycle, "audit_result", None)
                    if audit:
                        all_violations.extend(getattr(audit, "violations", []))

                with ui.expansion(
                    f"Audit Loop ({len(feedback.history)} iterations)", icon="replay"
                ).props("dense"):
                    # Build per-iteration audit history for the
                    # pass-rate pictorial chart + table.
                    history_rows: list[dict] = []
                    for cycle in feedback.history:
                        audit = getattr(cycle, "audit_result", None)
                        if not audit:
                            continue
                        viols = getattr(audit, "violations", []) or []
                        total = len(viols)
                        # ``passed`` flag on the audit means the *iteration*
                        # passed; for per-violation pass we count any with
                        # severity != critical/major as "ok".
                        passed = sum(
                            1 for v in viols
                            if getattr(v, "severity", "") not in ("critical", "major")
                        )
                        history_rows.append({
                            "label": f"iter {getattr(cycle, 'iteration', '?')}",
                            "passed": passed,
                            "total": total,
                            "converged": bool(getattr(audit, "passed", False)),
                            "issues": [
                                {"ok": getattr(v, "severity", "") not in ("critical", "major")}
                                for v in viols
                            ],
                        })

                    if history_rows:
                        with_expand(
                            lambda h, hr=history_rows: (
                                render_audit_passrate_pictorial(
                                    hr, height=h
                                )
                            ),
                            title="Audit pass-rate per iteration",
                            height="220px",
                        )
                        _, _, table_rows = audit_passrate_data(history_rows)
                        with ui.expansion(
                            "Pass-rate table", icon="table_view",
                        ).props("dense"):
                            ui.table(
                                columns=[
                                    {"name": "iteration", "label": "Iteration", "field": "iteration", "sortable": True},
                                    {"name": "passed", "label": "Passed", "field": "passed", "sortable": True},
                                    {"name": "total", "label": "Total", "field": "total", "sortable": True},
                                    {"name": "pass_rate", "label": "Pass-rate", "field": "pass_rate", "sortable": True},
                                    {"name": "converged", "label": "Converged?", "field": "converged"},
                                ],
                                rows=table_rows,
                                pagination={"rowsPerPage": 10},
                            ).props("dense flat bordered").classes("w-full")

                    for cycle in feedback.history:
                        _render_audit_cycle(cycle)

                    # Convergence trajectory line chart
                    with_expand(
                        lambda h, fb=feedback: render_convergence_trajectory(
                            fb, height=h,
                        ),
                        title="Convergence trajectory",
                        height="240px",
                    )

                # Violations summary from audit cycles
                if all_violations:
                    with ui.expansion(f"Violations ({len(all_violations)})").props("dense"):
                        for v in all_violations:
                            _render_violation(v, state=state)

        # Parse info
        if result.parse_result and result.parse_result.parsed:
            with ui.expansion("Parse Details", icon="code").props("dense"):
                parsed = result.parse_result.parsed
                if hasattr(parsed, "reasoning") and parsed.reasoning:
                    ui.label(f"Reasoning: {parsed.reasoning}").classes(
                        "text-xs text-slate-500"
                    )


def _render_audit_cycle(cycle) -> None:
    """Render a single audit cycle (iteration) in the replay view."""
    iteration = getattr(cycle, "iteration", "?")
    with ui.card().classes(
        "w-full p-3 mb-1 bg-slate-100 border border-slate-200 rounded-lg"
    ):
        ui.label(f"Iteration {iteration}").classes(
            "text-sm font-semibold text-slate-700"
        )

        # Prose excerpt for this iteration
        prose = getattr(cycle, "prose", "")
        if prose:
            ui.markdown(prose[:300] + ("…" if len(prose) > 300 else "")).classes(
                "text-xs text-slate-600"
            )

        # Audit result
        audit = getattr(cycle, "audit_result", None)
        if audit:
            passed = getattr(audit, "passed", None)
            if passed is not None:
                color = "positive" if passed else "warning"
                ui.badge("passed" if passed else "failed", color=color).props("dense")

            violations = getattr(audit, "violations", [])
            if violations:
                for v in violations[:3]:
                    _render_violation(v, state=None, compact=True)


def _render_violation(v, *, state: "AppState | None" = None, compact: bool = False) -> None:
    """Render one :class:`AuditViolation` with type, plain-English explanation,
    and (when applicable) a one-click directive chip that pre-fills a
    corrective query in the command bar.

    When ``state`` is provided, the "Fix this" button emits
    ``StateEvent.QUERY_STARTED`` so the chat command bar pre-fills the
    suggested directive. When ``state`` is None (e.g. inside the audit
    cycle compact view) the chip is omitted.
    """
    severity = getattr(v, "severity", "")
    severity_color = {
        "critical": "negative",
        "major": "warning",
        "minor": "info",
    }.get(severity, "grey")
    vtype = getattr(v, "violation_type", "")
    description = getattr(v, "description", str(v))
    feedback = getattr(v, "feedback", "")
    explanation = violation_explanation(vtype) if vtype else ""

    card_classes = (
        "w-full p-2 mb-1 bg-white border border-slate-200 rounded-md"
        if not compact
        else "w-full p-1 mb-1 bg-slate-50 rounded"
    )
    with ui.card().classes(card_classes):
        with ui.row().classes("items-center gap-2 flex-wrap"):
            if severity:
                ui.badge(severity, color=severity_color).props("dense")
            if vtype:
                ui.badge(vtype, color="grey").props("dense outline")
            ui.label(description[:200] if compact else description).classes(
                "text-xs text-slate-700"
            )
        if explanation and not compact:
            ui.label(explanation).classes(
                "text-[11px] text-slate-500 italic mt-1"
            )
        if not compact:
            template = directive_for_violation(vtype) if vtype else None
            with ui.row().classes("items-center gap-2 mt-1"):
                if template is not None and state is not None:
                    prompt_text, qtype = template

                    def _on_fix(p=prompt_text, q=qtype, s=state):
                        s.emit(
                            StateEvent.QUERY_STARTED,
                            suggestion=p,
                            query_type=q,
                        )
                        ui.notify(
                            "Pre-filled corrective directive in command bar.",
                            type="positive",
                        )

                    ui.button(
                        "Fix this", icon="auto_fix_high", on_click=_on_fix,
                    ).props(
                        "dense outline color=primary size=sm no-caps"
                    ).tooltip(prompt_text)
                if feedback:
                    with ui.expansion(
                        "Auditor's rewrite instruction", icon="edit_note",
                    ).props("dense").classes("w-full"):
                        ui.markdown(feedback)
