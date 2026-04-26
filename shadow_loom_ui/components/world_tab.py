"""World tab — interactive graph visualization, node inspector, topology tables.

Graph view modes: Full World | Ego-Graph | Causal Subgraph | Social | Spatial
Clicking a node opens an inspector drawer on the right.
Bottom drawer shows filterable topology edge tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from nicegui import ui

from shadow_loom_ui.graph_viz import (
    render_causal_subgraph,
    render_ego_graph,
    render_world_graph,
)
from shadow_loom_ui.state import AppState, StateEvent

if TYPE_CHECKING:
    from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)

_VIEW_MODES = [
    ("full", "Full World"),
    ("ego", "Ego-Graph"),
    ("causal", "Causal Subgraph"),
]


def build_world_tab(state: AppState) -> None:
    """Build the World tab layout."""

    with ui.column().classes("w-full h-full"):
        # ---- Toolbar ----
        with ui.row().classes("w-full items-center q-pa-sm gap-2"):
            view_mode = ui.toggle(
                {k: v for k, v in _VIEW_MODES},
                value="full",
            ).props("dense no-caps")

            # Ego-graph entity selector (shown only in ego mode)
            ego_select = ui.select(
                options=[],
                label="Focus entities",
                multiple=True,
            ).classes("w-64")
            ego_select.set_visibility(False)

            ui.button("Refresh", icon="refresh", on_click=lambda: _refresh_graph()).props(
                "flat dense"
            )

            def _on_mode_change():
                ego_select.set_visibility(view_mode.value == "ego")
                _refresh_graph()

            view_mode.on("update:model-value", _on_mode_change)

        # ---- Main content: graph + inspector ----
        with ui.splitter(value=75).classes("w-full flex-grow") as split:
            with split.before:
                graph_container = ui.html("").classes("w-full h-full")

            with split.after:
                inspector_container = ui.scroll_area().classes("w-full h-full q-pa-sm")
                with inspector_container:
                    ui.label("Click a node in the graph to inspect it.").classes(
                        "text-body2 text-grey"
                    )

        # ---- Bottom: topology tables ----
        with ui.expansion("Topology Edges", icon="device_hub").classes("w-full"):
            _build_topology_tables(state)

        # ---- Graph rendering ----
        def _refresh_graph():
            ws = state.world_state
            if ws is None:
                graph_container.content = (
                    '<div style="padding:2rem;color:grey;">No world model loaded</div>'
                )
                return

            # Update ego select options
            entity_opts = {e.id: e.name for e in ws.entities}
            ego_select.options = entity_opts

            mode = view_mode.value
            try:
                if mode == "ego" and ego_select.value:
                    focus_ids = ego_select.value if isinstance(ego_select.value, list) else [ego_select.value]
                    html_str = render_ego_graph(ws, focus_entity_ids=focus_ids)
                elif mode == "causal":
                    html_str = render_causal_subgraph(ws)
                else:
                    html_str = render_world_graph(ws)

                graph_container.content = html_str
            except Exception as e:
                logger.exception("Graph rendering failed")
                graph_container.content = f'<div style="color:red;">Render error: {e}</div>'

        # Initial render + subscribe
        _refresh_graph()
        ego_select.on("update:model-value", lambda: _refresh_graph())
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_graph)


# =====================================================================
# Node Inspector
# =====================================================================

def _render_entity_inspector(container, entity) -> None:
    """Render entity detail in the inspector panel."""
    container.clear()
    with container:
        ui.label(entity.name).classes("text-h6")
        with ui.row().classes("gap-2"):
            if entity.status:
                color = "negative" if entity.status == "dead" else "positive"
                ui.badge(entity.status, color=color)
            if entity.current_location_id:
                ui.badge(f"@ {entity.current_location_id}", color="blue")

        if entity.constants:
            ui.label("Constants").classes("text-subtitle2 q-mt-sm")
            for k, v in entity.constants.items():
                ui.label(f"{k}: {v}").classes("text-caption")

        if entity.traits:
            ui.label("Traits").classes("text-subtitle2 q-mt-sm")
            for name, tv in entity.traits.items():
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(name).classes("text-caption w-24")
                    ui.linear_progress(
                        value=max(0, min(1, (tv.value + 1) / 2)),
                        show_value=False,
                    ).classes("flex-grow")
                    ui.label(f"{tv.value:.2f}").classes("text-caption")
                    ui.label(f"(inertia: {tv.inertia:.2f})").classes("text-caption text-grey")

        if entity.beliefs:
            ui.label("Beliefs").classes("text-subtitle2 q-mt-sm")
            for belief in entity.beliefs:
                with ui.row().classes("gap-1"):
                    ui.label(f"about {belief.about_entity_id}:").classes("text-caption")
                    ui.label(belief.believed_trait).classes("text-caption text-grey")

        if hasattr(entity, "state_timeline") and entity.state_timeline:
            ui.label("State Timeline").classes("text-subtitle2 q-mt-sm")
            for entry in entity.state_timeline[-5:]:
                ui.label(f"t={entry.fabula_time}: {entry.description}").classes("text-caption text-grey")


def _render_location_inspector(container, location) -> None:
    """Render location detail in the inspector panel."""
    container.clear()
    with container:
        ui.label(location.name).classes("text-h6")
        if location.description:
            ui.label(location.description).classes("text-body2 text-grey")
        if hasattr(location, "ambient_state") and location.ambient_state:
            ui.label("Ambient State").classes("text-subtitle2 q-mt-sm")
            for k, v in location.ambient_state.items():
                ui.label(f"{k}: {v}").classes("text-caption")


def _render_event_inspector(container, event) -> None:
    """Render event detail in the inspector panel."""
    container.clear()
    with container:
        ui.label(event.id).classes("text-h6")
        if event.description:
            ui.label(event.description).classes("text-body2 text-grey")
        with ui.row().classes("gap-2"):
            if event.event_type:
                ui.badge(event.event_type, color="orange")
            ui.label(f"Fabula: t={event.fabula_time}").classes("text-caption")
            if hasattr(event, "syuzhet_index"):
                ui.label(f"Syuzhet: #{event.syuzhet_index}").classes("text-caption")
        if event.actors:
            ui.label(f"Actors: {', '.join(event.actors)}").classes("text-caption q-mt-sm")
        if event.targets:
            ui.label(f"Targets: {', '.join(event.targets)}").classes("text-caption")


# =====================================================================
# Topology Tables
# =====================================================================

def _build_topology_tables(state: AppState) -> None:
    """Build the 4 topology edge tables as tabs."""

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
                ],
                rows=[],
            ).props("dense flat").classes("w-full")

        with ui.tab_panel("spatial"):
            spatial_table = ui.table(
                columns=[
                    {"name": "from", "label": "From", "field": "from", "sortable": True},
                    {"name": "to", "label": "To", "field": "to", "sortable": True},
                    {"name": "locked", "label": "Locked", "field": "locked", "sortable": True},
                    {"name": "barrier", "label": "Barrier", "field": "barrier"},
                ],
                rows=[],
            ).props("dense flat").classes("w-full")

        with ui.tab_panel("social"):
            social_table = ui.table(
                columns=[
                    {"name": "from", "label": "From", "field": "from", "sortable": True},
                    {"name": "to", "label": "To", "field": "to", "sortable": True},
                    {"name": "affinity", "label": "Affinity", "field": "affinity", "sortable": True},
                    {"name": "fear", "label": "Fear", "field": "fear", "sortable": True},
                    {"name": "power", "label": "Power", "field": "power", "sortable": True},
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

    def _refresh_tables():
        ws = state.world_state
        if ws is None:
            return

        # Causal
        causal_rows = []
        for edge in getattr(ws, "causal_topology", []):
            causal_rows.append({
                "source": edge.source_event_id,
                "target": edge.target_event_id,
                "type": getattr(edge, "causal_type", ""),
                "mechanism": getattr(edge, "mechanism", ""),
                "force": getattr(edge, "force", 0),
            })
        causal_table.rows = causal_rows

        # Spatial
        spatial_rows = []
        for edge in getattr(ws, "spatial_topology", []):
            spatial_rows.append({
                "from": edge.location_a_id,
                "to": edge.location_b_id,
                "locked": getattr(edge, "locked", False),
                "barrier": getattr(edge, "barrier_item", ""),
            })
        spatial_table.rows = spatial_rows

        # Social
        social_rows = []
        for edge in getattr(ws, "social_topology", []):
            social_rows.append({
                "from": edge.from_entity_id,
                "to": edge.to_entity_id,
                "affinity": getattr(edge, "affinity", 0),
                "fear": getattr(edge, "fear", 0),
                "power": getattr(edge, "power_dynamic", ""),
            })
        social_table.rows = social_rows

        # Information
        info_rows = []
        for edge in getattr(ws, "information_topology", []):
            info_rows.append({
                "source": edge.source_entity_id,
                "targets": ", ".join(getattr(edge, "target_entity_ids", [])),
                "medium": getattr(edge, "medium", ""),
                "encrypted": getattr(edge, "encrypted", False),
            })
        info_table.rows = info_rows

    _refresh_tables()
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_tables)
