"""World tab — multi-view ECharts exploration with click-to-inspect.

View modes: Overview | Social | Spatial | Ego | Temporal | Epistemic
Clicking a node updates the left-panel inspector.
Bottom drawer shows filterable topology edge tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import (
    render_ego_graph,
    render_entity_state_timeline,
    render_epistemic_map,
    render_event_timeline,
    render_relationship_heatmap,
    render_social_graph,
    render_spatial_map,
    render_world_graph,
)
from shadow_loom_ui.viz_helpers import (
    ws_to_causal_rows,
    ws_to_info_rows,
    ws_to_social_rows,
    ws_to_spatial_rows,
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
    "epistemic": "Epistemic",
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

            def _on_mode_change():
                mode = view_mode.value
                ego_select.set_visibility(mode == "ego")
                temporal_select.set_visibility(mode == "temporal")
                _refresh()

            view_mode.on("update:model-value", _on_mode_change)

        # ── Graph container ───────────────────────────────────────
        graph_container = ui.column().classes("w-full flex-grow q-pa-sm")

        # ── Topology tables (expandable) ──────────────────────────
        with ui.expansion("Topology Edges", icon="device_hub").classes("w-full"):
            _build_topology_tables(state)

        # ── Graph click → inspector ───────────────────────────────
        def _on_graph_click(e):
            data = e.args if isinstance(e.args, dict) else {}
            node_id = data.get("name") or data.get("data", {}).get("id")
            node_type = data.get("data", {}).get("_sl_node_type")
            if node_id and node_type:
                state.select_node(node_id, node_type)

        # ── Render function ───────────────────────────────────────
        def _refresh(**kw):
            graph_container.clear()
            ws = state.world_state
            if ws is None:
                with graph_container:
                    ui.label("No world model loaded.").classes("text-body2 text-grey q-pa-lg")
                return

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
                        with ui.row().classes("w-full h-full gap-2"):
                            with ui.column().classes("flex-grow"):
                                render_social_graph(ws, on_click=_on_graph_click, height="100%")
                            with ui.column().classes("w-1/3"):
                                render_relationship_heatmap(ws, height="100%")
                    elif mode == "spatial":
                        render_spatial_map(ws, on_click=_on_graph_click, height="100%")
                    elif mode == "ego":
                        focus = ego_select.value
                        if focus:
                            ids = focus if isinstance(focus, list) else [focus]
                            render_ego_graph(ws, ids, on_click=_on_graph_click, height="100%")
                        else:
                            ui.label("Select focus entities above.").classes(
                                "text-body2 text-grey"
                            )
                    elif mode == "temporal":
                        eid = temporal_select.value
                        if eid:
                            render_entity_state_timeline(eid, ws, height="300px")
                        else:
                            ui.label("Select an entity above.").classes(
                                "text-body2 text-grey"
                            )
                        # Also show event timeline
                        render_event_timeline(ws, on_click=_on_graph_click, height="250px")
                    elif mode == "epistemic":
                        render_epistemic_map(ws, height="100%")
                except Exception as e:
                    logger.exception("Graph rendering failed")
                    ui.label(f"Render error: {e}").classes("text-negative")

        # Initial render + subscriptions
        _refresh()
        ego_select.on("update:model-value", lambda: _refresh())
        temporal_select.on("update:model-value", lambda: _refresh())
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)


# =====================================================================
# Topology tables (correct field names from viz_helpers)
# =====================================================================

def _build_topology_tables(state: AppState) -> None:
    """Build the 4 topology edge tables as sub-tabs."""

    with ui.tabs().classes("w-full").props("dense") as topo_tabs:
        ui.tab("causal", label="Causal", icon="bolt")
        ui.tab("spatial", label="Spatial", icon="map")
        ui.tab("social", label="Social", icon="people")
        ui.tab("info", label="Information", icon="mail")

    with ui.tab_panels(topo_tabs, value="causal").classes("w-full"):
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
            ).props("dense flat").classes("w-full")

        with ui.tab_panel("spatial"):
            spatial_table = ui.table(
                columns=[
                    {"name": "source", "label": "From", "field": "source", "sortable": True},
                    {"name": "target", "label": "To", "field": "target", "sortable": True},
                    {"name": "locked", "label": "Locked", "field": "locked", "sortable": True},
                    {"name": "barrier", "label": "Barrier", "field": "barrier"},
                ],
                rows=[],
            ).props("dense flat").classes("w-full")

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
            ).props("dense flat").classes("w-full")

        with ui.tab_panel("info"):
            info_table = ui.table(
                columns=[
                    {"name": "source", "label": "Source", "field": "source", "sortable": True},
                    {"name": "targets", "label": "Targets", "field": "targets"},
                    {"name": "medium", "label": "Medium", "field": "medium", "sortable": True},
                    {"name": "encrypted", "label": "Encrypted", "field": "encrypted", "sortable": True},
                ],
                rows=[],
            ).props("dense flat").classes("w-full")

    def _refresh_tables(**kw):
        ws = state.world_state
        if ws is None:
            return
        causal_table.rows = ws_to_causal_rows(ws)
        spatial_table.rows = ws_to_spatial_rows(ws)
        social_table.rows = ws_to_social_rows(ws)
        info_table.rows = ws_to_info_rows(ws)

    _refresh_tables()
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_tables)
