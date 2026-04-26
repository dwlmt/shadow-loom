"""Left panel: world model explorer with tree navigation and inspector forms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState


def build_explorer(state: AppState) -> None:
    """Build the left-panel explorer inside the current NiceGUI container."""

    selected_node = {"type": None, "id": None}
    inspector_container = None

    def _tree_items() -> list[dict]:
        """Build tree structure from current world state."""
        if state.world_state is None:
            return [{"id": "empty", "label": "No world loaded", "icon": "info"}]

        ws = state.world_state
        items = []

        # Entities
        ent_children = []
        for eid, ent in sorted(ws.entities.items()):
            status_icon = {
                "healthy": "person", "injured": "personal_injury",
                "dead": "skull", "ill": "sick", "unconscious": "hotel",
            }.get(ent.status, "person")
            ent_children.append({
                "id": f"ent:{eid}", "label": f"{ent.name}",
                "icon": status_icon,
            })
        items.append({
            "id": "entities", "label": f"Entities ({len(ws.entities)})",
            "icon": "groups", "children": ent_children,
        })

        # Locations
        loc_children = [
            {"id": f"loc:{lid}", "label": loc.name, "icon": "place"}
            for lid, loc in sorted(ws.locations.items())
        ]
        items.append({
            "id": "locations", "label": f"Locations ({len(ws.locations)})",
            "icon": "map", "children": loc_children,
        })

        # Objects
        obj_children = [
            {"id": f"obj:{oid}", "label": obj.name, "icon": "category"}
            for oid, obj in sorted(ws.objects.items())
        ]
        items.append({
            "id": "objects", "label": f"Objects ({len(ws.objects)})",
            "icon": "inventory_2", "children": obj_children,
        })

        # Events
        events_sorted = sorted(ws.events, key=lambda e: e.fabula_time)
        evt_children = [
            {"id": f"evt:{evt.id}", "label": f"T{evt.fabula_time}: {evt.description[:40]}",
             "icon": "event"}
            for evt in events_sorted
        ]
        items.append({
            "id": "events", "label": f"Events ({len(ws.events)})",
            "icon": "timeline", "children": evt_children,
        })

        # World Traits
        wt_children = [
            {"id": f"wt:{wid}", "label": wt.name, "icon": "public"}
            for wid, wt in sorted(ws.world_traits.items())
        ]
        if wt_children:
            items.append({
                "id": "world_traits", "label": f"World Traits ({len(ws.world_traits)})",
                "icon": "language", "children": wt_children,
            })

        return items

    def _on_select(e):
        """Handle tree node selection — show inspector."""
        node_id = e.value
        if not node_id or ":" not in node_id:
            return
        ntype, nid = node_id.split(":", 1)
        selected_node["type"] = ntype
        selected_node["id"] = nid
        _render_inspector()

    def _render_inspector():
        """Render the inspector panel for the selected node."""
        if inspector_container is None:
            return
        inspector_container.clear()

        ws = state.world_state
        if ws is None:
            return

        ntype = selected_node["type"]
        nid = selected_node["id"]

        with inspector_container:
            if ntype == "ent" and nid in ws.entities:
                _render_entity_inspector(ws, nid)
            elif ntype == "loc" and nid in ws.locations:
                _render_location_inspector(ws, nid)
            elif ntype == "obj" and nid in ws.objects:
                _render_object_inspector(ws, nid)
            elif ntype == "evt":
                evt = next((e for e in ws.events if e.id == nid), None)
                if evt:
                    _render_event_inspector(evt)
            elif ntype == "wt" and nid in ws.world_traits:
                _render_world_trait_inspector(ws, nid)

    # ---- Inspector renderers ----

    def _render_entity_inspector(ws, eid: str):
        ent = ws.entities[eid]
        ui.label(f"{ent.name}").classes("text-h6")
        ui.label(f"ID: {eid}").classes("text-caption text-grey")

        with ui.row().classes("items-center gap-2"):
            ui.badge(ent.status, color={
                "healthy": "green", "injured": "orange",
                "dead": "red", "ill": "yellow", "unconscious": "grey",
            }.get(ent.status, "grey"))
            ui.label(f"@ {ent.location_id}").classes("text-body2")

        if ent.constants:
            ui.label("Constants").classes("text-subtitle2 q-mt-sm")
            for c in ent.constants:
                ui.badge(c, color="blue-grey")

        if ent.traits:
            ui.label("Traits").classes("text-subtitle2 q-mt-sm")
            for tname, tv in ent.traits.items():
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(tname).classes("text-body2").style("min-width: 100px")
                    ui.linear_progress(value=tv.value, show_value=False).props(
                        f'color={"green" if tv.value > 0.5 else "red"}'
                    ).classes("flex-grow")
                    ui.label(f"{tv.value:.2f}").classes("text-caption")
                    ui.label(f"(i={tv.inertia:.2f})").classes("text-caption text-grey")

        if ent.beliefs:
            ui.label("Beliefs").classes("text-subtitle2 q-mt-sm")
            for b in ent.beliefs:
                with ui.row().classes("gap-1"):
                    ui.label(f"→ {b.target_id}: {b.perceived_state}").classes("text-body2")
                    ui.label(f"({b.confidence:.0%})").classes("text-caption text-grey")

        if ent.state_timeline:
            ui.label(f"State Timeline ({len(ent.state_timeline)} snapshots)").classes(
                "text-subtitle2 q-mt-sm"
            )
            for snap in ent.state_timeline[-5:]:
                with ui.row().classes("gap-1"):
                    ui.label(f"T={snap.fabula_time}").classes("text-caption text-blue")
                    if snap.triggered_by:
                        ui.label(f"← {snap.triggered_by}").classes("text-caption text-grey")

    def _render_location_inspector(ws, lid: str):
        loc = ws.locations[lid]
        ui.label(loc.name).classes("text-h6")
        ui.label(f"ID: {lid}").classes("text-caption text-grey")
        ui.label(loc.description).classes("text-body2 q-mt-sm")

        if loc.ambient_state:
            ui.label("Ambient State").classes("text-subtitle2 q-mt-sm")
            for aname, av in loc.ambient_state.items():
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(aname).classes("text-body2").style("min-width: 100px")
                    ui.linear_progress(value=av.value, show_value=False).classes("flex-grow")
                    ui.label(f"{av.value:.2f} (vol={av.volatility:.2f})").classes("text-caption")

        # Who is here?
        present = [
            ent.name for ent in ws.entities.values()
            if ent.location_id == lid
        ]
        if present:
            ui.label("Present Entities").classes("text-subtitle2 q-mt-sm")
            for name in present:
                ui.label(f"  • {name}").classes("text-body2")

        # Objects here
        here_objs = [
            obj.name for obj in ws.objects.values()
            if obj.location_id == lid
        ]
        if here_objs:
            ui.label("Objects Here").classes("text-subtitle2 q-mt-sm")
            for name in here_objs:
                ui.label(f"  • {name}").classes("text-body2")

    def _render_object_inspector(ws, oid: str):
        obj = ws.objects[oid]
        ui.label(obj.name).classes("text-h6")
        ui.label(f"ID: {oid}").classes("text-caption text-grey")

        with ui.row().classes("gap-2"):
            if obj.owner_id:
                ui.label(f"Owned by: {obj.owner_id}").classes("text-body2")
            if obj.location_id:
                ui.label(f"At: {obj.location_id}").classes("text-body2")

        if obj.properties:
            ui.label("Properties").classes("text-subtitle2 q-mt-sm")
            for k, v in obj.properties.items():
                ui.label(f"  {k}: {v}").classes("text-body2")

        if obj.affordances:
            ui.label("Affordances").classes("text-subtitle2 q-mt-sm")
            for aff in obj.affordances:
                ui.label(f"  {aff.action} → {aff.target_type}").classes("text-body2")

    def _render_event_inspector(evt):
        ui.label(f"Event: {evt.id}").classes("text-h6")
        ui.label(f"T={evt.fabula_time} | Syuzhet={evt.syuzhet_index}").classes("text-caption text-grey")
        ui.badge(evt.event_type, color={
            "choice": "blue", "outcome": "orange", "revelation": "purple",
        }.get(evt.event_type, "grey"))

        ui.label(evt.description).classes("text-body2 q-mt-sm")

        if evt.actor_ids:
            ui.label(f"Actors: {', '.join(evt.actor_ids)}").classes("text-body2 q-mt-xs")
        if evt.target_ids:
            ui.label(f"Targets: {', '.join(evt.target_ids)}").classes("text-body2 q-mt-xs")

    def _render_world_trait_inspector(ws, wid: str):
        wt = ws.world_traits[wid]
        ui.label(f"{wt.name}").classes("text-h6")
        ui.label(f"ID: {wid}").classes("text-caption text-grey")
        ui.badge(wt.category, color="teal")
        ui.label(wt.description).classes("text-body2 q-mt-sm")

        with ui.row().classes("items-center gap-4 q-mt-sm"):
            ui.label(f"Magnitude: {wt.magnitude.value:.2f}").classes("text-body2")
            ui.label(f"Inertia: {wt.magnitude.inertia:.2f}").classes("text-body2")

        if wt.affected_domains:
            ui.label(f"Domains: {', '.join(wt.affected_domains)}").classes("text-body2 q-mt-xs")

        if wt.state_timeline:
            ui.label("State Timeline").classes("text-subtitle2 q-mt-md")
            for snap in wt.state_timeline:
                parts = [f"T={snap.fabula_time}"]
                if snap.triggered_by:
                    parts.append(f"by {snap.triggered_by}")
                if snap.magnitude:
                    parts.append(f"mag={snap.magnitude.value:.2f}")
                if snap.description:
                    parts.append(snap.description[:60])
                ui.label(" | ".join(parts)).classes("text-body2 q-ml-sm")

    # ---- Build the UI ----

    with ui.column().classes("w-full h-full"):
        ui.label("World Explorer").classes("text-h6 q-pa-sm")

        tree = ui.tree(
            _tree_items(), label_key="label", node_key="id",
            children_key="children",
            on_select=_on_select,
        ).classes("w-full").props("dense")

        ui.separator()

        ui.label("Inspector").classes("text-subtitle1 q-pa-sm")
        inspector_container = ui.column().classes("w-full q-pa-sm overflow-auto")

    # Register refresh callback
    def _refresh():
        tree.update()
        tree._props["nodes"] = _tree_items()
        tree.update()
        _render_inspector()

    state.on_world_state_changed = _refresh
