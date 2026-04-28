"""Reasoning tab — interpretability surfaces for writers and fan-analysts.

Sub-views:

* **Trace**         — the rung-2/rung-3 reasoning trace for the most
                      recent query (do-set / abduction / cascade /
                      blocked).
* **Belief lens**   — chronological provenance of an entity's beliefs:
                      when each belief formed, what triggered it, when
                      it was invalidated.
* **Why this?**     — attribution graph for any outcome event,
                      reverse-walking causal edges with ranked
                      contributing paths.
* **Foreshadowing** — setup→payoff arcs across fabula time, with a
                      "loose threads only" toggle for unpaid setups.
* **Convergence**   — per-iteration audit metrics for the most recent
                      refinement loop.

All renderers are pure data functions in
:mod:`shadow_loom_ui.reasoning_helpers` and
:mod:`shadow_loom_ui.reasoning_viz` — this module is the layout glue.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nicegui import ui

from shadow_loom_ui.components.event_navigator import build_event_navigator
from shadow_loom_ui.components.reasoning_trace import render_reasoning_trace
from shadow_loom_ui.reasoning_helpers import (
    foreshadowing_arcs_data,
)
from shadow_loom_ui.reasoning_viz import (
    render_attribution_graph,
    render_belief_provenance,
    render_convergence_trajectory,
    render_foreshadowing_arcs,
)
from shadow_loom_ui.state import AppState, StateEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_reasoning_tab(state: AppState) -> None:
    """Build the Reasoning tab layout."""

    with ui.column().classes("w-full h-full bg-slate-50"):
        with ui.tabs().props(
            "dense no-caps indicator-color=primary active-color=primary align=left"
        ).classes(
            "w-full bg-white border-b border-slate-200 px-4"
        ) as sub_tabs:
            ui.tab("events", label="Events", icon="auto_stories")
            ui.tab("trace", label="Trace", icon="psychology")
            ui.tab("belief", label="Belief lens", icon="visibility")
            ui.tab("attribution", label="Why this?", icon="device_hub")
            ui.tab("foreshadow", label="Foreshadowing", icon="auto_fix_high")
            ui.tab("convergence", label="Convergence", icon="show_chart")

        with ui.tab_panels(sub_tabs, value="events").classes(
            "w-full flex-grow bg-slate-50"
        ):
            with ui.tab_panel("events").classes("q-pa-none h-full"):
                build_event_navigator(state)
            with ui.tab_panel("trace").classes("p-4"):
                _build_trace_panel(state)
            with ui.tab_panel("belief").classes("p-4"):
                _build_belief_panel(state)
            with ui.tab_panel("attribution").classes("p-4"):
                _build_attribution_panel(state)
            with ui.tab_panel("foreshadow").classes("p-4"):
                _build_foreshadow_panel(state)
            with ui.tab_panel("convergence").classes("p-4"):
                _build_convergence_panel(state)


# =====================================================================
# 1. Trace — rung-2 / rung-3 reasoning trace for the last query
# =====================================================================

def _build_trace_panel(state: AppState) -> None:
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        ui.label(
            "Shows what the engine actually did on your most recent "
            "intervention or counterfactual query."
        ).classes("text-sm text-slate-500")

        body = ui.column().classes("w-full gap-2 q-mt-sm")

        def _refresh(**_: Any) -> None:
            body.clear()
            with body:
                render_reasoning_trace(state)

        _refresh()
        state.on(StateEvent.PIPELINE_RESULT, _refresh)
        state.on(StateEvent.PROJECT_LOADED, _refresh)


# =====================================================================
# 2. Belief lens — entity-scoped belief provenance
# =====================================================================

def _build_belief_panel(state: AppState) -> None:
    if state.world_state is None or not state.world_state.entities:
        ui.label("No world model loaded.").classes(
            "text-sm text-slate-400 italic"
        )
        return

    entity_options = {
        eid: ent.name for eid, ent in state.world_state.entities.items()
    }
    initial = state.selected_node_id if (
        state.selected_node_id and state.selected_node_id in entity_options
    ) else next(iter(entity_options))

    with ui.row().classes("w-full items-center gap-3"):
        ui.label("Entity:").classes("text-sm text-slate-600")
        select = ui.select(
            options=entity_options, value=initial,
        ).props("dense outlined options-dense").classes("min-w-64")

    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4 q-mt-sm"
    ):
        ui.label(
            "Each row tracks one of this character's beliefs. Green ▲ "
            "marks formation; red ▼ marks invalidation. Tooltip shows "
            "the triggering event."
        ).classes("text-xs text-slate-500")
        body = ui.column().classes("w-full q-mt-sm")

        def _refresh(**_: Any) -> None:
            body.clear()
            ws = state.world_state
            if ws is None:
                return
            with body:
                render_belief_provenance(ws, select.value)

        _refresh()
        select.on("update:model-value", lambda _e: _refresh())
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)
        state.on(StateEvent.PROJECT_LOADED, _refresh)


# =====================================================================
# 3. Attribution — "Why did event X happen?"
# =====================================================================

def _build_attribution_panel(state: AppState) -> None:
    if state.world_state is None or not state.world_state.events:
        ui.label("No events in the world model.").classes(
            "text-sm text-slate-400 italic"
        )
        return

    event_options = {
        evt.id: f"t{evt.fabula_time}: {(evt.description or evt.id)[:50]}"
        for evt in sorted(state.world_state.events, key=lambda e: e.fabula_time)
    }
    initial = state.selected_node_id if (
        state.selected_node_id and state.selected_node_id in event_options
    ) else next(iter(event_options))

    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
        ui.label("Outcome:").classes("text-sm text-slate-600")
        select = ui.select(
            options=event_options, value=initial,
        ).props("dense outlined options-dense").classes("min-w-96")

        ui.label("Max depth:").classes("text-sm text-slate-600 ml-4")
        depth = ui.number(value=4, min=1, max=8).props(
            "dense outlined"
        ).classes("w-24")

        ui.label("Min force:").classes("text-sm text-slate-600 ml-4")
        min_force = ui.number(value=0.0, min=0.0, max=10.0, step=0.5).props(
            "dense outlined"
        ).classes("w-24")

    body = ui.column().classes("w-full q-mt-sm")

    def _refresh(**_: Any) -> None:
        body.clear()
        ws = state.world_state
        if ws is None:
            return
        with body:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ):
                ui.label(
                    "Reverse-walks causal edges from the chosen outcome. "
                    "Brighter nodes are nearer causes; arrow width "
                    "reflects causal force."
                ).classes("text-xs text-slate-500")
                render_attribution_graph(
                    ws,
                    select.value,
                    max_depth=int(depth.value or 4),
                    min_force=float(min_force.value or 0.0),
                )

            # Ranked paths table — derived from the same data fn
            from shadow_loom_ui.reasoning_helpers import attribution_graph_data
            _, _, _, paths = attribution_graph_data(
                ws, select.value,
                max_depth=int(depth.value or 4),
                min_force=float(min_force.value or 0.0),
            )
            if paths:
                with ui.card().classes(
                    "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
                ):
                    ui.label("Top contributing paths").classes(
                        "text-sm font-semibold text-slate-700"
                    )
                    rows = [
                        {
                            "rank": i + 1,
                            "path": "  →  ".join(p["path"]),
                            "force": f"{p['force']:.2f}",
                            "depth": p["depth"],
                        }
                        for i, p in enumerate(paths[:10])
                    ]
                    ui.table(
                        columns=[
                            {"name": "rank", "label": "#", "field": "rank", "align": "right"},
                            {"name": "path", "label": "Causal path", "field": "path", "align": "left"},
                            {"name": "force", "label": "Force", "field": "force", "align": "right"},
                            {"name": "depth", "label": "Hops", "field": "depth", "align": "right"},
                        ],
                        rows=rows,
                        pagination={"rowsPerPage": 10},
                    ).props("dense flat bordered").classes("w-full")

    _refresh()
    select.on("update:model-value", lambda _e: _refresh())
    depth.on("update:model-value", lambda _e: _refresh())
    min_force.on("update:model-value", lambda _e: _refresh())
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)


# =====================================================================
# 4. Foreshadowing — setup→payoff arcs and loose-threads inbox
# =====================================================================

def _build_foreshadow_panel(state: AppState) -> None:
    with ui.row().classes("w-full items-center gap-3"):
        loose_only = ui.switch("Show loose threads only", value=False).props(
            "dense color=warning"
        )

    body = ui.column().classes("w-full q-mt-sm")

    def _refresh(**_: Any) -> None:
        body.clear()
        ws = state.world_state
        if ws is None:
            with body:
                ui.label("No world model loaded.").classes(
                    "text-sm text-slate-400 italic"
                )
            return
        with body:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ):
                ui.label(
                    "Curved arcs link setup events to their delayed "
                    "payoffs. Orange arcs are unresolved — Chekhov's "
                    "guns the writer hasn't fired yet."
                ).classes("text-xs text-slate-500")
                render_foreshadowing_arcs(
                    ws, show_loose_only=loose_only.value,
                )

            arcs = foreshadowing_arcs_data(ws)
            if loose_only.value:
                arcs = [a for a in arcs if a["is_loose"]]
            if arcs:
                with ui.card().classes(
                    "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
                ):
                    ui.label(
                        "Loose threads inbox" if loose_only.value
                        else "All foreshadowing arcs"
                    ).classes("text-sm font-semibold text-slate-700")
                    rows = [
                        {
                            "setup": a["setup_label"],
                            "payoff": a["payoff_label"],
                            "span": a["span"],
                            "force": f"{a['force']:.2f}",
                            "loose": "⚠ unresolved" if a["is_loose"] else "✓ resolved",
                        }
                        for a in arcs[:50]
                    ]
                    ui.table(
                        columns=[
                            {"name": "setup", "label": "Setup", "field": "setup"},
                            {"name": "payoff", "label": "Payoff", "field": "payoff"},
                            {"name": "span", "label": "Span", "field": "span"},
                            {"name": "force", "label": "Force", "field": "force"},
                            {"name": "loose", "label": "Status", "field": "loose"},
                        ],
                        rows=rows,
                        pagination={"rowsPerPage": 15},
                    ).props("dense flat bordered").classes("w-full")

    _refresh()
    loose_only.on("update:model-value", lambda _e: _refresh())
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)
    state.on(StateEvent.PROJECT_LOADED, _refresh)


# =====================================================================
# 5. Convergence — refinement-loop trajectory
# =====================================================================

def _build_convergence_panel(state: AppState) -> None:
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        ui.label(
            "Per-iteration audit trajectory of the most recent refinement "
            "loop. Green dashes mark iterations where the auditor passed."
        ).classes("text-sm text-slate-500")
        body = ui.column().classes("w-full q-mt-sm")

        def _refresh(**_: Any) -> None:
            body.clear()
            pr = state.last_result
            feedback = None
            if pr is not None:
                feedback = getattr(pr, "feedback_result", None)
            with body:
                if feedback is None:
                    ui.label(
                        "No refinement loop history yet. Run a query that "
                        "engages the auditor (intervention, directive, or "
                        "counterfactual) to populate this view."
                    ).classes("text-sm text-slate-400 italic")
                    return
                render_convergence_trajectory(feedback)

        _refresh()
        state.on(StateEvent.PIPELINE_RESULT, _refresh)
        state.on(StateEvent.PROJECT_LOADED, _refresh)
