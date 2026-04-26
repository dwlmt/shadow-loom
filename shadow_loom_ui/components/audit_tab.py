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
from shadow_loom_ui.viz import render_emotional_gauges

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_audit_tab(state: AppState) -> None:
    """Build the Audit tab layout."""

    with ui.column().classes("w-full h-full q-pa-md gap-4"):
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
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("fact_check", size="md", color="primary")
                ui.label("Full Story Evaluation").classes("text-h6")

            ui.label(
                "Run a comprehensive narrative quality evaluation across the entire world model."
            ).classes("text-body2 text-grey")

            eval_container = ui.column().classes("w-full q-mt-sm")
            eval_status = ui.label("").classes("text-body2")

            async def _run_evaluation():
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return

                eval_status.set_text("Running evaluation…")
                try:
                    from shadow_loom.query_models import EvaluationQuery
                    query = EvaluationQuery()
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: state.run_structured_query(query),
                    )
                    eval_status.set_text("")
                    _render_evaluation_result(eval_container, result)
                except Exception as e:
                    logger.exception("Evaluation failed")
                    eval_status.set_text(f"Error: {e}")
                    ui.notify(f"Evaluation failed: {e}", type="negative")

            ui.button(
                "Run Evaluation", icon="play_arrow", on_click=_run_evaluation
            ).props("color=primary no-caps")

        # ── Per-Query Audit History ───────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Query Audit History").classes("text-h6")
            history_container = ui.column().classes("w-full")

            def _refresh_history(**kw):
                history_container.clear()
                if not state.query_history:
                    with history_container:
                        ui.label("No queries yet. Use the command bar to ask questions.").classes(
                            "text-body2 text-grey"
                        )
                    return

                with history_container:
                    for i, result in enumerate(reversed(state.query_history)):
                        _render_query_audit_entry(i, result)

            _refresh_history()
            state.on(StateEvent.PIPELINE_RESULT, _refresh_history)


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
            ui.label("No evaluation result.").classes("text-grey")
        return

    with container:
        # Evaluation scorecard (NarrativeOrderObject)
        eval_result = getattr(pr, "evaluation_result", None)

        if eval_result:
            _render_scorecard(container, eval_result)
        elif pr.prose:
            with ui.card().classes("w-full q-pa-md"):
                ui.label("Evaluation Report").classes("text-subtitle1")
                ui.markdown(pr.prose)

        # Convergence info
        if pr.converged is not None:
            with ui.card().classes("w-full q-pa-md"):
                status = "Converged" if pr.converged else "Did not converge"
                color = "positive" if pr.converged else "warning"
                with ui.row().classes("items-center gap-2"):
                    ui.badge(status, color=color)
                    ui.label(f"Audit iterations: {pr.audit_iterations}").classes("text-caption")


def _render_scorecard(container, eval_result) -> None:
    """Render NarrativeOrderObject scorecard with gauges and metrics."""
    with container:
        # Extract scores
        narrative_order = getattr(eval_result, "narrative_order", None)
        if narrative_order is None:
            ui.label("No scorecard data.").classes("text-grey")
            return

        # Overall pass/fail
        overall = getattr(narrative_order, "overall_pass", None)
        if overall is not None:
            color = "positive" if overall else "negative"
            ui.badge(
                "PASS" if overall else "FAIL",
                color=color,
            ).classes("text-h6 q-mb-md")

        # Causal feedback
        causal = getattr(narrative_order, "causal_feedback", None)
        if causal:
            with ui.card().classes("w-full q-pa-md"):
                ui.label("Causal Metrics").classes("text-subtitle1 q-mb-sm")
                scores = {}
                if hasattr(causal, "foreshadowing_payoff_score"):
                    scores["Foreshadowing"] = causal.foreshadowing_payoff_score
                if hasattr(causal, "cognitive_plausibility_score"):
                    scores["Plausibility"] = causal.cognitive_plausibility_score
                if scores:
                    render_emotional_gauges(scores, height="150px")

                # Miracle steps
                miracles = getattr(causal, "miracle_steps_detected", [])
                if miracles:
                    ui.label(f"⚠️ Miracle steps detected: {len(miracles)}").classes(
                        "text-negative q-mt-sm"
                    )
                    for m in miracles[:5]:
                        ui.label(f"  • {m}").classes("text-caption text-negative")
                else:
                    ui.label("✓ No miracle steps").classes("text-positive q-mt-sm")

        # Affective feedback
        affective = getattr(narrative_order, "affective_feedback", None)
        if affective:
            with ui.card().classes("w-full q-pa-md"):
                ui.label("Affective Metrics").classes("text-subtitle1 q-mb-sm")
                scores = {}
                if hasattr(affective, "emotional_trajectory_scores") and affective.emotional_trajectory_scores:
                    scores.update(affective.emotional_trajectory_scores)
                if hasattr(affective, "affective_loss_mse"):
                    scores["Affective Loss"] = affective.affective_loss_mse
                if scores:
                    render_emotional_gauges(scores, height="150px")

                if hasattr(affective, "kl_divergence_prediction_error"):
                    ui.label(
                        f"KL Divergence (surprise): {affective.kl_divergence_prediction_error:.3f}"
                    ).classes("text-caption q-mt-sm")

        # Quality synthesis
        quality = getattr(narrative_order, "quality_synthesis", None)
        if quality:
            with ui.card().classes("w-full q-pa-md"):
                ui.label("Quality Synthesis").classes("text-subtitle1 q-mb-sm")
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

def _render_query_audit_entry(index: int, result: NLQueryResult) -> None:
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
            ui.label(result.summary).classes("text-body2")

        if result.error:
            ui.label(f"Error: {result.error}").classes("text-negative text-caption")

        if pr is not None:
            # Convergence
            if pr.converged is not None:
                status = "Converged" if pr.converged else "Did not converge"
                with ui.row().classes("items-center gap-2"):
                    color = "positive" if pr.converged else "warning"
                    ui.badge(status, color=color).props("dense")
                    ui.label(f"{pr.audit_iterations} iterations").classes("text-caption")

            # Prose excerpt
            if pr.prose:
                with ui.expansion("Prose", icon="article").props("dense"):
                    ui.markdown(pr.prose[:500] + ("…" if len(pr.prose) > 500 else ""))

            # Audit loop replay (if feedback_result has cycles)
            feedback = getattr(pr, "feedback_result", None)
            if feedback and hasattr(feedback, "cycles") and feedback.cycles:
                with ui.expansion(
                    f"Audit Loop ({len(feedback.cycles)} iterations)", icon="replay"
                ).props("dense"):
                    for cycle in feedback.cycles:
                        _render_audit_cycle(cycle)

            # Violations
            if hasattr(pr, "violations") and pr.violations:
                with ui.expansion(f"Violations ({len(pr.violations)})").props("dense"):
                    for v in pr.violations:
                        severity_color = {
                            "critical": "negative",
                            "major": "warning",
                            "minor": "info",
                        }.get(getattr(v, "severity", ""), "grey")
                        with ui.row().classes("items-center gap-2"):
                            ui.badge(
                                getattr(v, "severity", ""), color=severity_color
                            ).props("dense")
                            ui.label(getattr(v, "message", str(v))).classes("text-caption")

        # Parse info
        if result.parse_result and result.parse_result.parsed:
            with ui.expansion("Parse Details", icon="code").props("dense"):
                parsed = result.parse_result.parsed
                if hasattr(parsed, "reasoning") and parsed.reasoning:
                    ui.label(f"Reasoning: {parsed.reasoning}").classes("text-caption text-grey")


def _render_audit_cycle(cycle) -> None:
    """Render a single audit cycle (iteration) in the replay view."""
    iteration = getattr(cycle, "iteration", "?")
    with ui.card().classes("w-full q-pa-sm q-mb-xs").style("background: #252530"):
        ui.label(f"Iteration {iteration}").classes("text-subtitle2")

        # Prose excerpt for this iteration
        prose = getattr(cycle, "prose", "")
        if prose:
            ui.markdown(prose[:300] + ("…" if len(prose) > 300 else "")).classes("text-caption")

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
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("error_outline", size="xs", color="warning")
                        ui.label(
                            getattr(v, "message", str(v))[:100]
                        ).classes("text-caption text-grey")
