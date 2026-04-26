"""Audit tab — quality evaluation scorecard + per-query feedback history.

Provides a full-story evaluation button (NarrativeOrderObject scorecard)
and a history of per-query audit cycles with violations and metrics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_audit_tab(state: AppState) -> None:
    """Build the Audit tab layout."""

    with ui.column().classes("w-full h-full q-pa-md gap-4"):
        # ---- Full Story Evaluation ----
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

                eval_status.set_text("Running evaluation...")
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

        # ---- Per-Query Audit History ----
        with ui.card().classes("w-full"):
            ui.label("Query Audit History").classes("text-h6")
            history_container = ui.column().classes("w-full")

            def _refresh_history():
                history_container.clear()
                if not state.query_history:
                    with history_container:
                        ui.label("No queries yet.").classes("text-body2 text-grey")
                    return

                with history_container:
                    for i, result in enumerate(reversed(state.query_history)):
                        _render_query_audit_entry(i, result)

            _refresh_history()
            state.on(StateEvent.PIPELINE_RESULT, lambda **kw: _refresh_history())


# =====================================================================
# Evaluation result renderer
# =====================================================================

def _render_evaluation_result(container, result: NLQueryResult) -> None:
    """Render the full-story NarrativeOrderObject evaluation."""
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
        # If there's a narrative order object in the result
        if pr.prose:
            with ui.card().classes("w-full q-pa-md"):
                ui.label("Evaluation Report").classes("text-subtitle1")
                ui.markdown(pr.prose)

        # Audit convergence info
        if pr.converged is not None:
            with ui.card().classes("w-full q-pa-md"):
                status = "Converged" if pr.converged else "Did not converge"
                color = "positive" if pr.converged else "warning"
                ui.badge(status, color=color).classes("q-mb-sm")
                ui.label(f"Audit iterations: {pr.audit_iterations}").classes("text-caption")

        # Change impact metrics
        if hasattr(pr, "change_impact") and pr.change_impact:
            _render_change_impact(container, pr.change_impact)

        # Physics state (for interrogate-style evaluations)
        if pr.physics_state and not pr.prose:
            with ui.expansion("Raw Physics State", icon="data_object").classes("w-full"):
                ui.code(pr.physics_state[:3000], language="json")


def _render_change_impact(container, impact) -> None:
    """Render change impact metrics with progress bars."""
    with container:
        with ui.card().classes("w-full q-pa-md"):
            ui.label("Change Impact").classes("text-subtitle1")
            if hasattr(impact, "causal_delta"):
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label("Causal Delta").classes("w-32 text-caption")
                    ui.linear_progress(
                        value=min(1, abs(impact.causal_delta)),
                        show_value=False,
                        color="red" if impact.causal_delta > 0.5 else "green",
                    ).classes("flex-grow")
                    ui.label(f"{impact.causal_delta:.3f}").classes("text-caption")

            if hasattr(impact, "affective_delta"):
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label("Affective Delta").classes("w-32 text-caption")
                    ui.linear_progress(
                        value=min(1, abs(impact.affective_delta)),
                        show_value=False,
                        color="orange" if impact.affective_delta > 0.5 else "green",
                    ).classes("flex-grow")
                    ui.label(f"{impact.affective_delta:.3f}").classes("text-caption")


# =====================================================================
# Per-query audit entry
# =====================================================================

def _render_query_audit_entry(index: int, result: NLQueryResult) -> None:
    """Render a single query's audit information."""
    pr = result.pipeline_result
    query_type = pr.query_type if pr else "unknown"
    converged = pr.converged if pr else None

    with ui.expansion(
        f"#{index + 1} — {query_type}",
        icon="check_circle" if converged else "warning" if converged is False else "help",
    ).classes("w-full").props("dense"):
        # Summary
        if result.summary:
            ui.label(result.summary).classes("text-body2")

        if result.error:
            ui.label(f"Error: {result.error}").classes("text-negative text-caption")

        if pr is not None:
            # Convergence
            if pr.converged is not None:
                status = "Converged" if pr.converged else "Did not converge"
                ui.label(
                    f"Audit: {status} ({pr.audit_iterations} iterations)"
                ).classes("text-caption")

            # Prose excerpt
            if pr.prose:
                with ui.expansion("Prose", icon="article").props("dense"):
                    ui.markdown(pr.prose[:500] + ("..." if len(pr.prose) > 500 else ""))

            # Violations (if available in audit data)
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
