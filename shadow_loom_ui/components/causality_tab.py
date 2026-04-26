"""Causality tab — causal topology, what-if workbench, affective dashboard.

Three sub-views:
  1. Causal Topology — Sankey + causal force graph
  2. What-If Workbench — intervention/counterfactual builders
  3. Affective Dashboard — emotional gauges, tension, trait trajectories
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent
from shadow_loom_ui.viz import (
    render_causal_force_graph,
    render_causal_sankey,
    render_emotional_gauges,
    render_event_timeline,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_causality_tab(state: AppState) -> None:
    """Build the Causality tab with three sub-views."""

    with ui.column().classes("w-full h-full"):
        with ui.tabs().classes("w-full").props("dense no-caps") as sub_tabs:
            ui.tab("topology", label="Causal Topology", icon="device_hub")
            ui.tab("whatif", label="What-If Workbench", icon="science")
            ui.tab("affective", label="Affective Dashboard", icon="favorite")

        with ui.tab_panels(sub_tabs, value="topology").classes("w-full flex-grow"):
            with ui.tab_panel("topology").classes("q-pa-sm"):
                _build_causal_topology(state)

            with ui.tab_panel("whatif").classes("q-pa-sm"):
                _build_whatif_workbench(state)

            with ui.tab_panel("affective").classes("q-pa-sm"):
                _build_affective_dashboard(state)


# =====================================================================
# 1. Causal Topology
# =====================================================================

def _build_causal_topology(state: AppState) -> None:
    """Sankey diagram + causal force graph."""

    with ui.column().classes("w-full h-full gap-3"):
        # Toggle between Sankey and force graph
        view_toggle = ui.toggle(
            {"sankey": "Sankey Flow", "force": "Force Graph"},
            value="sankey",
        ).props("dense no-caps")

        graph_container = ui.column().classes("w-full flex-grow")

        def _refresh(**kw):
            graph_container.clear()
            ws = state.world_state
            if ws is None:
                with graph_container:
                    ui.label("No world model loaded.").classes("text-body2 text-grey")
                return

            with graph_container:
                if view_toggle.value == "sankey":
                    render_causal_sankey(ws, height="100%")
                else:
                    render_causal_force_graph(ws, height="100%")

        _refresh()
        view_toggle.on("update:model-value", lambda: _refresh())
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)


# =====================================================================
# 2. What-If Workbench
# =====================================================================

def _build_whatif_workbench(state: AppState) -> None:
    """Intervention and counterfactual builders."""

    with ui.column().classes("w-full h-full gap-4"):
        # ── Intervention Builder ──────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("build", color="red")
                ui.label("Intervention (do-operator)").classes("text-subtitle1")

            ui.label(
                "Force a change to the world state and see how it propagates."
            ).classes("text-caption text-grey")

            with ui.row().classes("w-full gap-2 q-mt-sm"):
                interv_input = ui.textarea(
                    placeholder="e.g., 'What happens if Macbeth refuses to kill Duncan?'",
                ).classes("flex-grow").props("rows=2 autogrow outlined dense")

            result_container_interv = ui.column().classes("w-full")

            async def _run_intervention():
                text = interv_input.value.strip()
                if not text:
                    return
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return

                state.emit(StateEvent.QUERY_STARTED)
                try:
                    result = await state.run_nl_query_async(
                        text, query_type="intervention"
                    )
                    _render_whatif_result(result_container_interv, result)
                except Exception as e:
                    logger.exception("Intervention failed")
                    ui.notify(f"Error: {e}", type="negative")

            ui.button(
                "Run Intervention", icon="play_arrow", on_click=_run_intervention
            ).props("color=red no-caps")

        # ── Counterfactual Builder ────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("alt_route", color="pink")
                ui.label("Counterfactual (what-if)").classes("text-subtitle1")

            ui.label(
                "Explore alternate timelines: what would be different if an event hadn't happened?"
            ).classes("text-caption text-grey")

            with ui.row().classes("w-full gap-2 q-mt-sm"):
                cf_input = ui.textarea(
                    placeholder="e.g., 'What if Romeo never met Juliet at the party?'",
                ).classes("flex-grow").props("rows=2 autogrow outlined dense")

            result_container_cf = ui.column().classes("w-full")

            async def _run_counterfactual():
                text = cf_input.value.strip()
                if not text:
                    return
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return

                state.emit(StateEvent.QUERY_STARTED)
                try:
                    result = await state.run_nl_query_async(
                        text, query_type="counterfactual"
                    )
                    _render_whatif_result(result_container_cf, result)
                except Exception as e:
                    logger.exception("Counterfactual failed")
                    ui.notify(f"Error: {e}", type="negative")

            ui.button(
                "Run Counterfactual", icon="play_arrow", on_click=_run_counterfactual
            ).props("color=pink no-caps")


def _render_whatif_result(container, result: NLQueryResult) -> None:
    """Render intervention/counterfactual result."""
    container.clear()

    if result.error:
        with container:
            ui.label(f"Error: {result.error}").classes("text-negative")
        return

    pr = result.pipeline_result
    if pr is None:
        with container:
            ui.label("No result.").classes("text-grey")
        return

    with container:
        with ui.card().classes("w-full q-pa-md"):
            with ui.row().classes("items-center gap-2 q-mb-sm"):
                ui.badge(pr.query_type, color="primary").props("dense")
                if pr.world_model:
                    ui.badge(f"v{pr.world_model.version}", color="teal").props("dense")

            if pr.prose:
                ui.markdown(pr.prose[:2000])

            if pr.physics_state:
                with ui.expansion("Physics State", icon="data_object").props("dense"):
                    ui.code(pr.physics_state[:2000], language="json")

            if pr.converged is not None:
                status = "converged" if pr.converged else "did not converge"
                color = "positive" if pr.converged else "warning"
                ui.badge(f"Audit: {status} ({pr.audit_iterations} iters)", color=color).props(
                    "dense"
                )


# =====================================================================
# 3. Affective Dashboard
# =====================================================================

def _build_affective_dashboard(state: AppState) -> None:
    """Emotional gauges, narrative tension, and candidate events."""

    with ui.column().classes("w-full h-full gap-4"):
        gauge_container = ui.column().classes("w-full")
        timeline_container = ui.column().classes("w-full")

        def _refresh(**kw):
            gauge_container.clear()
            timeline_container.clear()
            ws = state.world_state
            if ws is None:
                with gauge_container:
                    ui.label("No world model loaded.").classes("text-body2 text-grey")
                return

            # Compute basic affective scores from world state
            scores = _compute_affective_scores(ws)

            with gauge_container:
                if scores:
                    ui.label("Narrative Affect Scores").classes("text-subtitle1 q-mb-sm")
                    render_emotional_gauges(scores, height="200px")
                else:
                    ui.label("No affective scores available.").classes("text-grey")

            with timeline_container:
                ui.label("Event Timeline").classes("text-subtitle1 q-mb-sm")
                render_event_timeline(ws, height="250px")

        _refresh()
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)


def _compute_affective_scores(ws) -> dict[str, float]:
    """Compute basic affective scores from world state structure."""
    scores: dict[str, float] = {}

    if not ws.events:
        return scores

    # Mystery: proportion of information edges with late discovery
    if ws.information_topology:
        late = sum(
            1 for ie in ws.information_topology
            if ie.discovered_at_syuzhet > 0
        )
        scores["mystery"] = min(1.0, late / max(1, len(ws.information_topology)))

    # Tension: fabula/syuzhet displacement
    if ws.events:
        displacements = []
        for evt in ws.events:
            if evt.fabula_time and evt.syuzhet_index:
                displacements.append(abs(evt.fabula_time - evt.syuzhet_index))
        if displacements:
            max_disp = max(displacements) or 1
            scores["narrative_tension"] = min(1.0, sum(displacements) / (len(displacements) * max_disp))

    # Conflict: proportion of negative affinity relationships
    if ws.social_topology:
        negative = sum(1 for r in ws.social_topology if r.affinity < 0)
        scores["conflict"] = min(1.0, negative / max(1, len(ws.social_topology)))

    # Danger: average fear across all relationships
    if ws.social_topology:
        avg_fear = sum(r.fear for r in ws.social_topology) / len(ws.social_topology)
        scores["danger"] = min(1.0, avg_fear)

    # Complexity: causal density (edges / events)
    if ws.events:
        scores["causal_density"] = min(1.0, len(ws.causal_topology) / max(1, len(ws.events) * 2))

    return scores
