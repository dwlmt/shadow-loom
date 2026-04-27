"""Left panel — version tree, world explorer tree, and quick-reference inspector.

Provides progressive disclosure: tree overview → click for detail.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui import db
from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import render_trait_radar, render_version_tree
from shadow_loom_ui.viz_helpers import ws_stats

if TYPE_CHECKING:
    from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)


def build_left_panel(state: AppState) -> None:
    """Build the left sidebar: version tree + world explorer + inspector."""

    with ui.column().classes("w-full h-full gap-0"):
        # ── Version Tree ──────────────────────────────────────────
        with ui.expansion("Version Tree", icon="account_tree", value=True).classes(
            "w-full"
        ).props("dense"):
            version_container = ui.column().classes("w-full")
            _render_versions(state, version_container)

        ui.separator()

        # ── World Explorer Tree ───────────────────────────────────
        with ui.expansion("World Explorer", icon="public", value=True).classes(
            "w-full"
        ).props("dense"):
            explorer_container = ui.column().classes("w-full")
            _render_explorer(state, explorer_container)

        ui.separator()

        # ── Quick Reference / Inspector ───────────────────────────
        with ui.expansion("Inspector", icon="info", value=True).classes(
            "w-full"
        ).props("dense"):
            inspector_container = ui.scroll_area().classes("w-full").style(
                "max-height: 40vh"
            )
            with inspector_container:
                ui.label("Click a node to inspect.").classes(
                    "text-caption text-grey q-pa-sm"
                )

    # ── Event subscriptions ───────────────────────────────────────
    def _on_ws_change(**kw):
        _render_explorer(state, explorer_container)

    def _on_version_change(**kw):
        _render_versions(state, version_container)

    def _on_node_selected(**kw):
        _render_inspector(state, inspector_container, kw.get("node_id"), kw.get("node_type"))

    state.on(StateEvent.WORLD_STATE_CHANGED, _on_ws_change)
    state.on(StateEvent.VERSION_CHANGED, _on_version_change)
    state.on(StateEvent.NODE_SELECTED, _on_node_selected)


# =====================================================================
# Version tree
# =====================================================================

def _render_versions(state: AppState, container) -> None:
    container.clear()
    if state.project_id is None:
        with container:
            ui.label("No project loaded.").classes("text-caption text-grey")
        return

    tree_data = db.get_version_tree(state.project_id)
    if not tree_data:
        with container:
            ui.label("No versions yet.").classes("text-caption text-grey")
        return

    def _on_version_click(e):
        data = e.args if isinstance(e.args, dict) else {}
        name = data.get("name", "")
        # Find matching version
        for v in tree_data:
            if f"v{v['version']}" == name:
                _load_version(state, container, v, tree_data)
                return

    with container:
        render_version_tree(
            tree_data,
            current_version_id=state.current_version_row_id,
            on_click=_on_version_click,
            height="180px",
        )


def _load_version(state: AppState, container, v: dict, tree_data: list[dict]) -> None:
    from shadow_loom.models import WorldStateV1

    if v["id"] == state.current_version_row_id:
        return
    ver = db.get_version_by_id(v["id"])
    if ver is None:
        ui.notify("Version not found", type="warning")
        return
    try:
        ws = WorldStateV1.model_validate_json(ver.world_state_json)
        state.load_world_state(ws)
        state.current_version_row_id = v["id"]
        state.emit(StateEvent.VERSION_CHANGED, version=v["version"])
        ui.notify(f"Loaded v{v['version']}")
    except Exception as e:
        ui.notify(f"Load failed: {e}", type="negative")


# =====================================================================
# World explorer tree
# =====================================================================

def _render_explorer(state: AppState, container) -> None:
    container.clear()
    ws = state.world_state
    if ws is None:
        with container:
            ui.label("No world model.").classes("text-caption text-grey")
        return

    stats = ws_stats(ws)

    with container:
        # Search/filter input
        search = ui.input(
            placeholder="Filter…",
        ).classes("w-full q-mb-xs").props("dense outlined clearable")

        # Container that gets filtered
        items_container = ui.column().classes("w-full")

        def _build_items(filter_text: str = ""):
            items_container.clear()
            ft = filter_text.lower().strip()
            with items_container:
                # Entities
                filtered_ents = [
                    (eid, ent) for eid, ent in ws.entities.items()
                    if not ft or ft in ent.name.lower()
                ]
                if filtered_ents:
                    with ui.expansion(
                        f"Entities ({len(filtered_ents)})", icon="people"
                    ).classes("w-full").props("dense"):
                        for eid, ent in filtered_ents:
                            ui.button(
                                ent.name,
                                on_click=lambda e_id=eid: state.select_node(e_id, "Entity"),
                            ).props("flat dense no-caps align=left").classes("w-full text-left")

                # Locations
                filtered_locs = [
                    (lid, loc) for lid, loc in ws.locations.items()
                    if not ft or ft in loc.name.lower()
                ]
                if filtered_locs:
                    with ui.expansion(
                        f"Locations ({len(filtered_locs)})", icon="place"
                    ).classes("w-full").props("dense"):
                        for lid, loc in filtered_locs:
                            ui.button(
                                loc.name,
                                on_click=lambda l_id=lid: state.select_node(l_id, "Location"),
                            ).props("flat dense no-caps align=left").classes("w-full text-left")

                # Events
                filtered_evts = [
                    evt for evt in sorted(ws.events, key=lambda e: e.fabula_time)
                    if not ft or ft in (evt.description or "").lower() or ft in evt.id.lower()
                ]
                if filtered_evts:
                    with ui.expansion(
                        f"Events ({len(filtered_evts)})", icon="bolt"
                    ).classes("w-full").props("dense"):
                        for evt in filtered_evts[:30]:
                            label = f"t{evt.fabula_time}: {(evt.description or '')[:40]}"
                            ui.button(
                                label,
                                on_click=lambda ev_id=evt.id: state.select_node(ev_id, "EventNode"),
                            ).props("flat dense no-caps align=left").classes(
                                "w-full text-left text-caption"
                            )

                # Objects
                filtered_objs = [
                    (oid, obj) for oid, obj in ws.objects.items()
                    if not ft or ft in obj.name.lower()
                ]
                if filtered_objs:
                    with ui.expansion(
                        f"Objects ({len(filtered_objs)})", icon="category"
                    ).classes("w-full").props("dense"):
                        for oid, obj in filtered_objs:
                            ui.button(
                                obj.name,
                                on_click=lambda o_id=oid: state.select_node(o_id, "NarrativeObject"),
                            ).props("flat dense no-caps align=left").classes("w-full text-left")

                # World Traits
                filtered_wts = [
                    (wid, wt) for wid, wt in ws.world_traits.items()
                    if not ft or ft in wt.name.lower()
                ]
                if filtered_wts:
                    with ui.expansion(
                        f"World Traits ({len(filtered_wts)})", icon="public"
                    ).classes("w-full").props("dense"):
                        for wid, wt in filtered_wts:
                            ui.button(
                                f"{wt.name} ({wt.magnitude.value:.2f})",
                                on_click=lambda w_id=wid: state.select_node(w_id, "WorldTrait"),
                            ).props("flat dense no-caps align=left").classes("w-full text-left")

        _build_items()
        search.on("update:model-value", lambda e: _build_items(search.value or ""))


# =====================================================================
# Inspector (detail view for selected node)
# =====================================================================

def _render_inspector(state: AppState, container, node_id: str | None, node_type: str | None) -> None:
    container.clear()
    ws = state.world_state
    if ws is None or node_id is None:
        with container:
            ui.label("Click a node to inspect.").classes("text-caption text-grey q-pa-sm")
        return

    with container:
        if node_type == "Entity" and node_id in ws.entities:
            _inspect_entity(ws, node_id)
            _inspector_suggestions(state, node_id, node_type, ws.entities[node_id].name)
        elif node_type == "Location" and node_id in ws.locations:
            _inspect_location(ws, node_id)
            _inspector_suggestions(state, node_id, node_type, ws.locations[node_id].name)
        elif node_type == "EventNode":
            evt = next((e for e in ws.events if e.id == node_id), None)
            if evt:
                _inspect_event(ws, evt)
                _inspector_suggestions(state, node_id, node_type, evt.description[:30])
        elif node_type == "NarrativeObject" and node_id in ws.objects:
            _inspect_object(ws, node_id)
            _inspector_suggestions(state, node_id, node_type, ws.objects[node_id].name)
        elif node_type == "WorldTrait" and node_id in ws.world_traits:
            _inspect_world_trait(ws, node_id)
            _inspector_suggestions(state, node_id, node_type, ws.world_traits[node_id].name)
        else:
            ui.label(f"Unknown: {node_id}").classes("text-caption text-grey q-pa-sm")


def _inspect_entity(ws: WorldStateV1, eid: str) -> None:
    ent = ws.entities[eid]
    ui.label(ent.name).classes("text-subtitle1 q-pa-xs")
    with ui.row().classes("gap-1 q-px-xs"):
        color = "negative" if ent.status == "dead" else "positive"
        ui.badge(ent.status, color=color).props("dense")
        if ent.location_id in ws.locations:
            ui.badge(f"@ {ws.locations[ent.location_id].name}", color="blue").props("dense")

    # Mini radar
    if ent.traits:
        render_trait_radar(eid, ws, height="180px")

    # Constants
    if ent.constants:
        ui.label("Constants").classes("text-caption text-bold q-mt-xs q-px-xs")
        for c in ent.constants:
            ui.label(f"• {c}").classes("text-caption q-px-sm")

    # Beliefs
    if ent.beliefs:
        ui.label("Beliefs").classes("text-caption text-bold q-mt-xs q-px-xs")
        for b in ent.beliefs[:8]:
            ui.label(
                f"about {b.target_id}: {b.perceived_state} (conf={b.confidence:.2f})"
            ).classes("text-caption q-px-sm")

    # State timeline (last 5)
    if ent.state_timeline:
        ui.label("Recent State Changes").classes("text-caption text-bold q-mt-xs q-px-xs")
        for snap in ent.state_timeline[-5:]:
            parts = [f"t={snap.fabula_time}"]
            if snap.triggered_by:
                parts.append(f"by {snap.triggered_by}")
            if snap.status:
                parts.append(f"→ {snap.status}")
            if snap.traits:
                trait_strs = [f"{k}={v:.2f}" for k, v in snap.traits.items()]
                parts.append(", ".join(trait_strs[:3]))
            ui.label(" | ".join(parts)).classes("text-caption q-px-sm")


def _inspect_location(ws: WorldStateV1, lid: str) -> None:
    loc = ws.locations[lid]
    ui.label(loc.name).classes("text-subtitle1 q-pa-xs")
    if loc.description:
        ui.label(loc.description[:120]).classes("text-caption text-grey q-px-xs")

    # Entities here
    here = [e.name for eid, e in ws.entities.items() if e.location_id == lid]
    if here:
        ui.label("Entities Present").classes("text-caption text-bold q-mt-xs q-px-xs")
        for name in here:
            ui.label(f"• {name}").classes("text-caption q-px-sm")

    # Ambient state
    if loc.ambient_state:
        ui.label("Ambient State").classes("text-caption text-bold q-mt-xs q-px-xs")
        for k, v in loc.ambient_state.items():
            ui.label(f"{k}: {v.value:.2f}").classes("text-caption q-px-sm")

    # Spatial connections
    connections = [
        se for se in ws.spatial_topology
        if se.source_id == lid or se.target_id == lid
    ]
    if connections:
        ui.label("Connections").classes("text-caption text-bold q-mt-xs q-px-xs")
        for se in connections:
            other = se.target_id if se.source_id == lid else se.source_id
            other_name = ws.locations[other].name if other in ws.locations else other
            lock = " 🔒" if se.is_locked else ""
            ui.label(f"↔ {other_name}{lock}").classes("text-caption q-px-sm")


def _inspect_event(ws: WorldStateV1, evt) -> None:
    ui.label(evt.id).classes("text-subtitle1 q-pa-xs")
    with ui.row().classes("gap-1 q-px-xs"):
        ui.badge(evt.event_type, color="orange").props("dense")
        ui.label(f"t={evt.fabula_time}").classes("text-caption")
        ui.label(f"s={evt.syuzhet_index}").classes("text-caption")

    ui.label(evt.description).classes("text-caption q-px-xs q-mt-xs")

    if evt.actor_ids:
        names = [ws.entities[a].name if a in ws.entities else a for a in evt.actor_ids]
        ui.label(f"Actors: {', '.join(names)}").classes("text-caption q-px-xs")
    if evt.target_ids:
        names = [ws.entities[t].name if t in ws.entities else t for t in evt.target_ids]
        ui.label(f"Targets: {', '.join(names)}").classes("text-caption q-px-xs")

    # Causal edges from/to this event
    causes = [ce for ce in ws.causal_topology if ce.target_id == evt.id]
    effects = [ce for ce in ws.causal_topology if ce.source_id == evt.id]
    if causes:
        ui.label("Caused by").classes("text-caption text-bold q-mt-xs q-px-xs")
        for ce in causes:
            ui.label(f"← {ce.source_id} ({ce.mechanism})").classes("text-caption q-px-sm")
    if effects:
        ui.label("Causes").classes("text-caption text-bold q-mt-xs q-px-xs")
        for ce in effects:
            ui.label(f"→ {ce.target_id} ({ce.mechanism})").classes("text-caption q-px-sm")


def _inspect_object(ws: WorldStateV1, oid: str) -> None:
    obj = ws.objects[oid]
    ui.label(obj.name).classes("text-subtitle1 q-pa-xs")
    if obj.owner_id and obj.owner_id in ws.entities:
        ui.label(f"Owner: {ws.entities[obj.owner_id].name}").classes("text-caption q-px-xs")
    if obj.location_id and obj.location_id in ws.locations:
        ui.label(f"Location: {ws.locations[obj.location_id].name}").classes("text-caption q-px-xs")
    if obj.properties:
        ui.label("Properties").classes("text-caption text-bold q-mt-xs q-px-xs")
        for k, v in obj.properties.items():
            ui.label(f"{k}: {v}").classes("text-caption q-px-sm")
    if obj.affordances:
        ui.label("Affordances").classes("text-caption text-bold q-mt-xs q-px-xs")
        for a in obj.affordances:
            ui.label(f"• {a.action} → {a.target_type}").classes("text-caption q-px-sm")


def _inspect_world_trait(ws: WorldStateV1, wid: str) -> None:
    wt = ws.world_traits[wid]
    ui.label(wt.name).classes("text-subtitle1 q-pa-xs")
    if wt.description:
        ui.label(wt.description[:120]).classes("text-caption text-grey q-px-xs")
    ui.label(f"Magnitude: {wt.magnitude.value:.2f} (inertia: {wt.magnitude.inertia:.2f})").classes(
        "text-caption q-px-xs"
    )
    if wt.affected_domains:
        ui.label(f"Domains: {', '.join(wt.affected_domains)}").classes("text-caption q-px-xs")
    if wt.state_timeline:
        ui.label("Timeline").classes("text-caption text-bold q-mt-xs q-px-xs")
        for snap in wt.state_timeline[-5:]:
            parts = [f"t={snap.fabula_time}"]
            if snap.magnitude is not None:
                parts.append(f"mag={snap.magnitude:.2f}")
            if snap.description:
                parts.append(snap.description[:40])
            ui.label(" | ".join(parts)).classes("text-caption q-px-sm")


def _inspector_suggestions(state: AppState, node_id: str, node_type: str, name: str) -> None:
    """Context-aware NL suggestion chips based on selected node."""
    suggestions: list[tuple[str, str]] = []

    if node_type == "Entity":
        suggestions = [
            ("What happens to this character next?", "interrogate"),
            (f"Kill {name}", "intervention"),
            (f"What if {name} never existed?", "counterfactual"),
            (f"Move {name} to a different location", "intervention"),
            (f"Make {name} feel suspense", "directive"),
        ]
    elif node_type == "Location":
        suggestions = [
            (f"What is happening at {name}?", "interrogate"),
            (f"Lock all exits from {name}", "intervention"),
            (f"Describe the atmosphere at {name}", "general"),
        ]
    elif node_type == "EventNode":
        suggestions = [
            ("What caused this event?", "interrogate"),
            ("What if this never happened?", "counterfactual"),
            ("What are the consequences?", "interrogate"),
        ]
    elif node_type == "NarrativeObject":
        suggestions = [
            (f"What role does {name} play?", "interrogate"),
            (f"Destroy {name}", "intervention"),
        ]
    elif node_type == "WorldTrait":
        suggestions = [
            (f"How does {name} affect the story?", "interrogate"),
            (f"Increase {name} dramatically", "intervention"),
        ]

    if suggestions:
        ui.separator().classes("q-my-xs")
        ui.label("Try:").classes("text-caption text-grey q-px-xs")
        with ui.column().classes("w-full gap-0 q-px-xs"):
            for text, qtype in suggestions[:4]:
                ui.chip(
                    text[:45],
                    icon="flash_on",
                    on_click=lambda t=text, q=qtype: state.emit(
                        StateEvent.QUERY_STARTED, suggestion=t, query_type=q
                    ),
                ).props("dense outline size=sm clickable").classes("q-my-none")
