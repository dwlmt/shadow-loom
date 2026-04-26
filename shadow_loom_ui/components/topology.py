"""Bottom drawer: topology edge tables (causal, spatial, social, information)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState


def build_topology_drawer(state: AppState) -> None:
    """Build the bottom topology drawer with edge tables."""

    causal_container = None
    spatial_container = None
    social_container = None
    info_container = None

    def _refresh():
        ws = state.world_state
        if ws is None:
            return

        # Causal edges
        if causal_container is not None:
            causal_container.clear()
            with causal_container:
                columns = [
                    {"name": "src", "label": "Source", "field": "src", "sortable": True},
                    {"name": "tgt", "label": "Target", "field": "tgt", "sortable": True},
                    {"name": "type", "label": "Type", "field": "type", "sortable": True},
                    {"name": "mech", "label": "Mechanism", "field": "mech"},
                    {"name": "force", "label": "Force", "field": "force", "sortable": True},
                    {"name": "time", "label": "T", "field": "time", "sortable": True},
                ]
                rows = [
                    {"src": ce.source_id, "tgt": ce.target_id,
                     "type": ce.causality_type, "mech": ce.mechanism,
                     "force": f"{ce.causal_force:.1f}", "time": ce.fabula_time}
                    for ce in ws.causal_topology
                ]
                ui.table(columns=columns, rows=rows).classes("w-full").props(
                    "dense flat virtual-scroll"
                )

        # Spatial edges
        if spatial_container is not None:
            spatial_container.clear()
            with spatial_container:
                columns = [
                    {"name": "src", "label": "From", "field": "src", "sortable": True},
                    {"name": "tgt", "label": "To", "field": "tgt", "sortable": True},
                    {"name": "locked", "label": "Locked", "field": "locked"},
                    {"name": "barrier", "label": "Barrier", "field": "barrier"},
                ]
                rows = [
                    {"src": se.source_id, "tgt": se.target_id,
                     "locked": "Yes" if se.is_locked else "No",
                     "barrier": se.barrier_item_id or "-"}
                    for se in ws.spatial_topology
                ]
                ui.table(columns=columns, rows=rows).classes("w-full").props("dense flat")

        # Social edges
        if social_container is not None:
            social_container.clear()
            with social_container:
                columns = [
                    {"name": "src", "label": "From", "field": "src", "sortable": True},
                    {"name": "tgt", "label": "To", "field": "tgt", "sortable": True},
                    {"name": "aff", "label": "Affinity", "field": "aff", "sortable": True},
                    {"name": "fear", "label": "Fear", "field": "fear", "sortable": True},
                    {"name": "pwr", "label": "Power", "field": "pwr", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia"},
                ]
                rows = [
                    {"src": re.source_entity_id, "tgt": re.target_entity_id,
                     "aff": f"{re.affinity:.2f}", "fear": f"{re.fear:.2f}",
                     "pwr": f"{re.power_dynamic:.2f}", "inertia": f"{re.inertia:.2f}"}
                    for re in ws.social_topology
                ]
                ui.table(columns=columns, rows=rows).classes("w-full").props("dense flat")

        # Information edges
        if info_container is not None:
            info_container.clear()
            with info_container:
                columns = [
                    {"name": "src", "label": "Source", "field": "src", "sortable": True},
                    {"name": "tgts", "label": "Targets", "field": "tgts"},
                    {"name": "medium", "label": "Medium", "field": "medium"},
                    {"name": "enc", "label": "Encrypted", "field": "enc"},
                ]
                rows = [
                    {"src": ie.source_id, "tgts": ", ".join(ie.target_ids),
                     "medium": ie.medium,
                     "enc": "Yes" if ie.is_encrypted else "No"}
                    for ie in ws.information_topology
                ]
                ui.table(columns=columns, rows=rows).classes("w-full").props("dense flat")

    # ---- Build the UI ----

    with ui.column().classes("w-full"):
        ui.label("Topology").classes("text-h6 q-pa-sm")

        with ui.tabs().classes("w-full") as tabs:
            causal_tab = ui.tab(f"Causal", icon="device_hub")
            spatial_tab = ui.tab(f"Spatial", icon="map")
            social_tab = ui.tab(f"Social", icon="people")
            info_tab = ui.tab(f"Information", icon="wifi")

        with ui.tab_panels(tabs, value=causal_tab).classes("w-full"):
            with ui.tab_panel(causal_tab):
                causal_container = ui.column().classes("w-full overflow-auto")
            with ui.tab_panel(spatial_tab):
                spatial_container = ui.column().classes("w-full overflow-auto")
            with ui.tab_panel(social_tab):
                social_container = ui.column().classes("w-full overflow-auto")
            with ui.tab_panel(info_tab):
                info_container = ui.column().classes("w-full overflow-auto")

    # Register refresh
    prev_cb = state.on_world_state_changed

    def _on_ws_changed():
        if prev_cb:
            prev_cb()
        _refresh()

    state.on_world_state_changed = _on_ws_changed
