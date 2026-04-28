"""World tab — multi-view ECharts exploration with click-to-inspect.

View modes: Overview | Social | Spatial | Ego | Temporal | Composition |
            Epistemic | Comparison
Clicking a node updates the left-panel inspector.
Bottom expansion shows filterable topology edge tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import (
    render_chord_diagram,
    render_ego_graph,
    render_entity_state_timeline,
    render_epistemic_map,
    render_event_gantt,
    render_event_timeline,
    render_parallel_coords,
    render_relationship_heatmap,
    render_social_graph,
    render_spatial_map,
    render_sunburst,
    render_theme_river,
    render_trait_boxplot,
    render_world_graph,
    render_world_treemap,
)
from shadow_loom_ui.viz_helpers import (
    fabula_time_bounds,
    snapshot_world_at,
    ws_to_causal_rows,
    ws_to_entity_rows,
    ws_to_event_rows,
    ws_to_info_rows,
    ws_to_object_rows,
    ws_to_social_rows,
    ws_to_spatial_rows,
    ws_to_trait_stats_rows,
    ws_to_world_trait_rows,
)

if TYPE_CHECKING:
    from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)

_VIEW_MODES = {
    "overview": "Overview",
    "social": "Social",
    "spatial": "Spatial",
    "ego": "Ego-Graph",
    "temporal": "Temporal",
    "composition": "Composition",
    "epistemic": "Epistemic",
    "comparison": "Comparison",
}


def build_world_tab(state: AppState) -> None:
    """Build the World tab layout."""

    with ui.column().classes("w-full h-full"):
        # ── Toolbar ───────────────────────────────────────────────
        with ui.row().classes("w-full items-center q-pa-sm gap-2"):
            view_mode = ui.toggle(
                _VIEW_MODES,
                value="overview",
            ).props("dense no-caps")

            # Ego-graph entity selector
            ego_select = ui.select(
                options=[],
                label="Focus entities",
                multiple=True,
            ).classes("w-64")
            ego_select.set_visibility(False)

            # Temporal entity selector
            temporal_select = ui.select(
                options=[],
                label="Entity",
            ).classes("w-48")
            temporal_select.set_visibility(False)

            ui.button("Refresh", icon="refresh", on_click=lambda: _refresh()).props(
                "flat dense"
            )

            social_layout = ui.toggle(
                {"force": "Force", "circular": "Circular"},
                value="force",
            ).props("dense no-caps").tooltip("Social graph layout")
            social_layout.set_visibility(False)
            spatial_animated = ui.checkbox("Animated", value=True).tooltip(
                "Animate location nodes (rippleEffect)"
            )
            spatial_animated.set_visibility(False)

            def _on_mode_change():
                mode = view_mode.value
                ego_select.set_visibility(mode == "ego")
                temporal_select.set_visibility(mode == "temporal")
                social_layout.set_visibility(mode == "social")
                spatial_animated.set_visibility(mode == "spatial")
                _refresh()

            view_mode.on("update:model-value", _on_mode_change)
            social_layout.on("update:model-value", lambda _e: _refresh())
            spatial_animated.on("update:model-value", lambda _e: _refresh())

        # ── Fabula timeline slider ────────────────────────────────
        slider_row = ui.row().classes(
            "w-full items-center q-px-md q-pb-sm gap-3 "
            "bg-slate-50 border-b border-slate-200"
        )
        with slider_row:
            ui.icon("schedule", color="primary")
            ui.label("Fabula time:").classes("text-sm text-slate-600")
            time_label = ui.label("live").classes(
                "text-sm font-mono text-slate-700 w-12"
            )
            time_slider = ui.slider(min=0, max=1, value=0, step=1).props(
                "color=primary label-always dense"
            ).classes("flex-grow")
            live_btn = ui.button(
                "Live", icon="bolt", on_click=lambda: _set_live()
            ).props("flat dense no-caps color=secondary")

        def _set_live():
            state.fabula_cursor = None
            time_label.text = "live"
            _refresh()

        def _on_slider_change():
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            state.fabula_cursor = t
            time_label.text = f"t={t}"
            _refresh()

        time_slider.on("update:model-value", lambda: _on_slider_change())

        # ── Graph container ───────────────────────────────────────
        graph_container = ui.column().classes(
            "w-full flex-grow p-4"
        )

        # ── Raw data tables (expandable) ──────────────────────
        with ui.expansion("Raw Data", icon="table_chart").classes(
            "w-full mx-4 mb-4 bg-white border border-slate-200 rounded-xl"
        ):
            _build_data_tables(state)

        # ── Graph click → inspector ───────────────────────────────
        def _on_graph_click(e):
            data = e.args if isinstance(e.args, dict) else {}
            node_data = data.get("data", {})
            node_id = node_data.get("id") or data.get("name")
            node_type = node_data.get("_sl_node_type")
            if node_id and node_type:
                state.select_node(node_id, node_type)

        # ── Render function ───────────────────────────────────────
        def _refresh(**kw):
            graph_container.clear()
            ws = state.world_state
            if ws is None:
                slider_row.set_visibility(False)
                with graph_container:
                    ui.label("No world model loaded.").classes(
                        "text-sm text-slate-400 italic q-pa-lg"
                    )
                return

            # Update slider bounds
            tmin, tmax = fabula_time_bounds(ws)
            if tmax > tmin:
                slider_row.set_visibility(True)
                time_slider.props(f"min={tmin} max={tmax}")
                if state.fabula_cursor is None:
                    time_slider.value = tmax
                    time_label.text = "live"
                else:
                    capped = max(tmin, min(tmax, state.fabula_cursor))
                    time_slider.value = capped
                    time_label.text = f"t={capped}"
            else:
                slider_row.set_visibility(False)

            # Snapshot the world model if a cursor is active
            if state.fabula_cursor is not None and tmax > tmin:
                try:
                    ws = snapshot_world_at(ws, state.fabula_cursor)
                except Exception:
                    logger.exception("Snapshot failed; falling back to live")

            # Update selectors
            entity_opts = {eid: ent.name for eid, ent in ws.entities.items()}
            ego_select.options = entity_opts
            temporal_select.options = entity_opts

            mode = view_mode.value
            with graph_container:
                try:
                    if mode == "overview":
                        render_world_graph(ws, on_click=_on_graph_click, height="100%")
                    elif mode == "social":
                        with ui.row().classes("w-full gap-2"):
                            with ui.column().classes("flex-grow"):
                                render_social_graph(
                                    ws,
                                    on_click=_on_graph_click,
                                    height="50%",
                                    layout=social_layout.value or "force",
                                )
                                render_chord_diagram(ws, height="50%")
                            with ui.column().classes("w-1/3"):
                                render_relationship_heatmap(ws, height="100%")
                    elif mode == "spatial":
                        render_spatial_map(
                            ws,
                            on_click=_on_graph_click,
                            height="100%",
                            animated=bool(spatial_animated.value),
                        )
                    elif mode == "ego":
                        focus = ego_select.value
                        if focus:
                            ids = focus if isinstance(focus, list) else [focus]
                            render_ego_graph(ws, ids, on_click=_on_graph_click, height="100%")
                        else:
                            ui.label("Select focus entities above.").classes(
                                "text-sm text-slate-500"
                            )
                    elif mode == "temporal":
                        eid = temporal_select.value
                        if eid:
                            render_entity_state_timeline(eid, ws, height="250px")
                        # ThemeRiver for multi-entity trait flow
                        render_theme_river(ws, height="300px")
                        # Event swim lanes
                        render_event_gantt(ws, on_click=_on_graph_click, height="250px")
                    elif mode == "composition":
                        with ui.row().classes("w-full gap-2 h-full"):
                            with ui.column().classes("flex-grow h-full"):
                                render_sunburst(ws, on_click=_on_graph_click, height="100%")
                            with ui.column().classes("w-1/2 h-full"):
                                render_world_treemap(ws, on_click=_on_graph_click, height="100%")
                    elif mode == "epistemic":
                        render_epistemic_map(ws, height="100%")
                    elif mode == "comparison":
                        with ui.column().classes("w-full gap-2"):
                            render_parallel_coords(ws, height="320px")
                            render_trait_boxplot(ws, height="280px")
                except Exception as e:
                    logger.exception("Graph rendering failed")
                    ui.label(f"Render error: {e}").classes("text-negative")

        # Initial render + subscriptions
        _refresh()
        ego_select.on("update:model-value", lambda: _refresh())
        temporal_select.on("update:model-value", lambda: _refresh())
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)


# =====================================================================
# Data tables (nodes + edges + distributions)
# =====================================================================

def _build_data_tables(state: AppState) -> None:
    """Build node, edge, and distribution tables as sub-tabs."""

    with ui.tabs().classes("w-full").props("dense") as data_tabs:
        ui.tab("entities", label="Entities", icon="person")
        ui.tab("events", label="Events", icon="bolt")
        ui.tab("objects", label="Objects", icon="inventory_2")
        ui.tab("world_traits", label="World Traits", icon="public")
        ui.tab("trait_stats", label="Trait Stats", icon="bar_chart")
        ui.tab("causal", label="Causal Edges", icon="trending_up")
        ui.tab("spatial", label="Spatial Edges", icon="map")
        ui.tab("social", label="Social Edges", icon="people")
        ui.tab("info", label="Info Edges", icon="mail")

    _table_props = "dense flat bordered"

    with ui.tab_panels(data_tabs, value="entities").classes("w-full"):
        with ui.tab_panel("entities"):
            entity_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "status", "label": "Status", "field": "status", "sortable": True},
                    {"name": "location", "label": "Location", "field": "location", "sortable": True},
                    {"name": "traits", "label": "#Traits", "field": "traits", "sortable": True},
                    {"name": "beliefs", "label": "#Beliefs", "field": "beliefs", "sortable": True},
                    {"name": "constants", "label": "Constants", "field": "constants"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("events"):
            event_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "fabula_time", "label": "Fabula t", "field": "fabula_time", "sortable": True},
                    {"name": "syuzhet_index", "label": "Syuzhet", "field": "syuzhet_index", "sortable": True},
                    {"name": "type", "label": "Type", "field": "type", "sortable": True},
                    {"name": "actors", "label": "Actors", "field": "actors"},
                    {"name": "targets", "label": "Targets", "field": "targets"},
                    {"name": "description", "label": "Description", "field": "description"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("objects"):
            object_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "location", "label": "Location", "field": "location", "sortable": True},
                    {"name": "owner", "label": "Owner", "field": "owner", "sortable": True},
                    {"name": "affordances", "label": "Affordances", "field": "affordances"},
                    {"name": "properties", "label": "Properties", "field": "properties"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("world_traits"):
            world_trait_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "magnitude", "label": "Magnitude", "field": "magnitude", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                    {"name": "description", "label": "Description", "field": "description"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("trait_stats"):
            trait_stats_table = ui.table(
                columns=[
                    {"name": "trait", "label": "Trait", "field": "trait", "sortable": True},
                    {"name": "min", "label": "Min", "field": "min", "sortable": True},
                    {"name": "q1", "label": "Q1", "field": "q1", "sortable": True},
                    {"name": "median", "label": "Median", "field": "median", "sortable": True},
                    {"name": "q3", "label": "Q3", "field": "q3", "sortable": True},
                    {"name": "max", "label": "Max", "field": "max", "sortable": True},
                    {"name": "outliers", "label": "Outliers", "field": "outliers", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("causal"):
            causal_table = ui.table(
                columns=[
                    {"name": "source", "label": "Source", "field": "source", "sortable": True},
                    {"name": "target", "label": "Target", "field": "target", "sortable": True},
                    {"name": "type", "label": "Type", "field": "type", "sortable": True},
                    {"name": "mechanism", "label": "Mechanism", "field": "mechanism"},
                    {"name": "force", "label": "Force", "field": "force", "sortable": True},
                    {"name": "evidence", "label": "Evidence", "field": "evidence", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("spatial"):
            spatial_table = ui.table(
                columns=[
                    {"name": "source", "label": "From", "field": "source", "sortable": True},
                    {"name": "target", "label": "To", "field": "target", "sortable": True},
                    {"name": "locked", "label": "Locked", "field": "locked", "sortable": True},
                    {"name": "barrier", "label": "Barrier", "field": "barrier"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("social"):
            social_table = ui.table(
                columns=[
                    {"name": "source", "label": "From", "field": "source", "sortable": True},
                    {"name": "target", "label": "To", "field": "target", "sortable": True},
                    {"name": "affinity", "label": "Affinity", "field": "affinity", "sortable": True},
                    {"name": "fear", "label": "Fear", "field": "fear", "sortable": True},
                    {"name": "power", "label": "Power", "field": "power", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("info"):
            info_table = ui.table(
                columns=[
                    {"name": "source", "label": "Source", "field": "source", "sortable": True},
                    {"name": "targets", "label": "Targets", "field": "targets"},
                    {"name": "medium", "label": "Medium", "field": "medium", "sortable": True},
                    {"name": "encrypted", "label": "Encrypted", "field": "encrypted", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

    def _refresh_tables(**kw):
        ws = state.world_state
        if ws is None:
            return
        entity_table.rows = ws_to_entity_rows(ws)
        event_table.rows = ws_to_event_rows(ws)
        object_table.rows = ws_to_object_rows(ws)
        world_trait_table.rows = ws_to_world_trait_rows(ws)
        trait_stats_table.rows = ws_to_trait_stats_rows(ws)
        causal_table.rows = ws_to_causal_rows(ws)
        spatial_table.rows = ws_to_spatial_rows(ws)
        social_table.rows = ws_to_social_rows(ws)
        info_table.rows = ws_to_info_rows(ws)

    _refresh_tables()
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_tables)
