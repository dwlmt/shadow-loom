"""Explorer tab — world tree + inspector.

The version-history tree was promoted to the always-visible left
sidebar (see ``version_sidebar.py``). This tab keeps the World
Explorer (filterable list of entities / locations / events / objects /
world traits) on the left and the Inspector on the right.

Clicking a node in the world tree updates the inspector via the shared
:data:`StateEvent.NODE_SELECTED` event.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import render_trait_radar
from shadow_loom_ui.viz_helpers import ws_stats

if TYPE_CHECKING:
    from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)


def build_explorer_tab(state: AppState) -> None:
    """Build the Explorer tab — world tree (left) + inspector (right)."""

    with ui.row().classes("w-full h-full no-wrap gap-0"):
        # ── Left column: world explorer ───────────────────────────
        with ui.column().classes(
            "h-full bg-slate-50 border-r border-slate-200"
        ).style("width: 36%; min-width: 320px;"):
            with ui.row().classes(
                "w-full items-center px-3 py-2 border-b border-slate-200 gap-2"
            ):
                ui.icon("public", color="primary")
                ui.label("World Explorer").classes(
                    "text-sm font-semibold text-slate-700"
                )
            with ui.scroll_area().classes("w-full flex-grow"):
                explorer_container = ui.column().classes("w-full p-2")
                _render_explorer(state, explorer_container)

        # ── Right column: inspector ───────────────────────────────
        with ui.column().classes("flex-grow h-full bg-white"):
            with ui.row().classes(
                "w-full items-center px-4 py-3 border-b border-slate-200 gap-2"
            ):
                ui.icon("info", color="primary")
                ui.label("Inspector").classes(
                    "text-base font-semibold text-slate-800"
                )
                ui.space()
                ui.label("Click a node on the left to inspect.").classes(
                    "text-xs text-slate-400"
                )
            with ui.scroll_area().classes("w-full flex-grow"):
                inspector_container = ui.column().classes("w-full p-4 gap-2")
                with inspector_container:
                    ui.label("No node selected.").classes(
                        "text-sm text-slate-400 italic"
                    )

    # ── Event subscriptions ───────────────────────────────────────
    def _on_ws_change(**_kw):
        _render_explorer(state, explorer_container)

    def _on_node_selected(**kw):
        _render_inspector(
            state,
            inspector_container,
            kw.get("node_id"),
            kw.get("node_type"),
        )

    state.on(StateEvent.WORLD_STATE_CHANGED, _on_ws_change)
    state.on(StateEvent.VERSION_CHANGED, _on_ws_change)
    state.on(StateEvent.NODE_SELECTED, _on_node_selected)


# =====================================================================
# World explorer tree
# =====================================================================

def _render_explorer(state: AppState, container) -> None:
    container.clear()
    ws = state.world_state
    if ws is None:
        with container:
            ui.label("No world model.").classes("text-xs text-slate-500 p-2")
        return

    stats = ws_stats(ws)

    with container:
        ui.label(
            f"{stats.get('entities', 0)} entities · "
            f"{stats.get('events', 0)} events · "
            f"{stats.get('locations', 0)} locations · "
            f"{stats.get('objects', 0)} objects · "
            f"{stats.get('world_traits', 0)} traits"
        ).classes("text-xs text-slate-500 mb-1 px-1")

        search = ui.input(placeholder="Filter…").classes(
            "w-full q-mb-xs"
        ).props("dense outlined clearable")

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
                                on_click=lambda e_id=eid: state.select_node(
                                    e_id, "Entity"
                                ),
                            ).props(
                                "flat dense no-caps align=left"
                            ).classes("w-full text-left")

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
                                on_click=lambda l_id=lid: state.select_node(
                                    l_id, "Location"
                                ),
                            ).props(
                                "flat dense no-caps align=left"
                            ).classes("w-full text-left")

                # Events
                filtered_evts = [
                    evt for evt in sorted(ws.events, key=lambda e: e.fabula_time)
                    if not ft
                    or ft in (evt.description or "").lower()
                    or ft in evt.id.lower()
                ]
                if filtered_evts:
                    with ui.expansion(
                        f"Events ({len(filtered_evts)})", icon="bolt"
                    ).classes("w-full").props("dense"):
                        for evt in filtered_evts[:60]:
                            label = (
                                f"t{evt.fabula_time}: "
                                f"{(evt.description or '')[:50]}"
                            )
                            ui.button(
                                label,
                                on_click=lambda ev_id=evt.id: state.select_node(
                                    ev_id, "EventNode"
                                ),
                            ).props(
                                "flat dense no-caps align=left"
                            ).classes("w-full text-left text-xs")

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
                                on_click=lambda o_id=oid: state.select_node(
                                    o_id, "NarrativeObject"
                                ),
                            ).props(
                                "flat dense no-caps align=left"
                            ).classes("w-full text-left")

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
                                on_click=lambda w_id=wid: state.select_node(
                                    w_id, "WorldTrait"
                                ),
                            ).props(
                                "flat dense no-caps align=left"
                            ).classes("w-full text-left")

        _build_items()
        search.on(
            "update:model-value",
            lambda e: _build_items(search.value or ""),
        )


# =====================================================================
# Inspector (detail view for selected node)
# =====================================================================

def _render_inspector(
    state: AppState,
    container,
    node_id: str | None,
    node_type: str | None,
) -> None:
    container.clear()
    ws = state.world_state
    if ws is None or node_id is None:
        with container:
            ui.label("No node selected.").classes(
                "text-sm text-slate-400 italic"
            )
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
            ui.label(f"Unknown: {node_id}").classes(
                "text-sm text-slate-500"
            )


def _inspect_entity(ws: "WorldStateV1", eid: str) -> None:
    ent = ws.entities[eid]
    ui.label(ent.name).classes("text-lg font-semibold text-slate-800")
    with ui.row().classes("gap-1"):
        color = "negative" if ent.status == "dead" else "positive"
        ui.badge(ent.status, color=color).props("dense")
        if ent.location_id in ws.locations:
            ui.badge(
                f"@ {ws.locations[ent.location_id].name}", color="secondary"
            ).props("dense")

    if ent.traits:
        render_trait_radar(eid, ws, height="260px")

    if ent.constants:
        ui.label("Constants").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for c in ent.constants:
            ui.label(f"• {c}").classes("text-sm text-slate-600 px-2")

    if ent.beliefs:
        ui.label("Beliefs").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for b in ent.beliefs[:8]:
            ui.label(
                f"about {b.target_id}: {b.perceived_state} "
                f"(conf={b.confidence:.2f})"
            ).classes("text-sm text-slate-600 px-2")

    if ent.state_timeline:
        ui.label("State Changes").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for snap in ent.state_timeline[-8:]:
            parts = [f"t={snap.fabula_time}"]
            if snap.triggered_by:
                parts.append(f"by {snap.triggered_by}")
            if snap.status:
                parts.append(f"→ {snap.status}")
            if snap.traits:
                trait_strs = [
                    f"{k}={v.value:.2f}" for k, v in snap.traits.items()
                ]
                parts.append(", ".join(trait_strs[:3]))
            ui.label(" | ".join(parts)).classes(
                "text-xs text-slate-600 px-2"
            )


def _inspect_location(ws: "WorldStateV1", lid: str) -> None:
    loc = ws.locations[lid]
    ui.label(loc.name).classes("text-lg font-semibold text-slate-800")
    if loc.description:
        ui.label(loc.description).classes("text-sm text-slate-500")

    here = [e.name for eid, e in ws.entities.items() if e.location_id == lid]
    if here:
        ui.label("Entities Present").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for name in here:
            ui.label(f"• {name}").classes("text-sm text-slate-600 px-2")

    if loc.ambient_state:
        ui.label("Ambient State").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for k, v in loc.ambient_state.items():
            ui.label(f"{k}: {v.value:.2f}").classes(
                "text-sm text-slate-600 px-2"
            )

    connections = [
        se for se in ws.spatial_topology
        if se.source_id == lid or se.target_id == lid
    ]
    if connections:
        ui.label("Connections").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for se in connections:
            other = se.target_id if se.source_id == lid else se.source_id
            other_name = (
                ws.locations[other].name if other in ws.locations else other
            )
            lock = " 🔒" if se.is_locked else ""
            ui.label(f"↔ {other_name}{lock}").classes(
                "text-sm text-slate-600 px-2"
            )


def _inspect_event(ws: "WorldStateV1", evt) -> None:
    ui.label(evt.id).classes("text-lg font-semibold text-slate-800")
    with ui.row().classes("gap-1"):
        ui.badge(evt.event_type, color="warning").props("dense")
        ui.label(f"t={evt.fabula_time}").classes("text-xs text-slate-500")
        ui.label(f"s={evt.syuzhet_index}").classes("text-xs text-slate-500")

    ui.label(evt.description).classes("text-sm text-slate-700 mt-1")

    if evt.actor_ids:
        names = [
            ws.entities[a].name if a in ws.entities else a
            for a in evt.actor_ids
        ]
        ui.label(f"Actors: {', '.join(names)}").classes(
            "text-sm text-slate-600 mt-1"
        )
    if evt.target_ids:
        names = [
            ws.entities[t].name if t in ws.entities else t
            for t in evt.target_ids
        ]
        ui.label(f"Targets: {', '.join(names)}").classes(
            "text-sm text-slate-600"
        )

    causes = [ce for ce in ws.causal_topology if ce.target_id == evt.id]
    effects = [ce for ce in ws.causal_topology if ce.source_id == evt.id]
    if causes:
        ui.label("Caused by").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for ce in causes:
            ui.label(f"← {ce.source_id} ({ce.mechanism})").classes(
                "text-sm text-slate-600 px-2"
            )
    if effects:
        ui.label("Causes").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for ce in effects:
            ui.label(f"→ {ce.target_id} ({ce.mechanism})").classes(
                "text-sm text-slate-600 px-2"
            )


def _inspect_object(ws: "WorldStateV1", oid: str) -> None:
    obj = ws.objects[oid]
    ui.label(obj.name).classes("text-lg font-semibold text-slate-800")
    if obj.owner_id and obj.owner_id in ws.entities:
        ui.label(f"Owner: {ws.entities[obj.owner_id].name}").classes(
            "text-sm text-slate-600"
        )
    if obj.location_id and obj.location_id in ws.locations:
        ui.label(f"Location: {ws.locations[obj.location_id].name}").classes(
            "text-sm text-slate-600"
        )
    if obj.properties:
        ui.label("Properties").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for k, v in obj.properties.items():
            ui.label(f"{k}: {v}").classes("text-sm text-slate-600 px-2")
    if obj.affordances:
        ui.label("Affordances").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for a in obj.affordances:
            ui.label(f"• {a.action} → {a.target_type}").classes(
                "text-sm text-slate-600 px-2"
            )


def _inspect_world_trait(ws: "WorldStateV1", wid: str) -> None:
    wt = ws.world_traits[wid]
    ui.label(wt.name).classes("text-lg font-semibold text-slate-800")
    if wt.description:
        ui.label(wt.description).classes("text-sm text-slate-500")
    ui.label(
        f"Magnitude: {wt.magnitude.value:.2f} "
        f"(inertia: {wt.magnitude.inertia:.2f})"
    ).classes("text-sm text-slate-600")
    if wt.affected_domains:
        ui.label(f"Domains: {', '.join(wt.affected_domains)}").classes(
            "text-sm text-slate-600"
        )
    if wt.state_timeline:
        ui.label("Timeline").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        for snap in wt.state_timeline[-8:]:
            parts = [f"t={snap.fabula_time}"]
            if snap.magnitude is not None:
                parts.append(f"mag={snap.magnitude.value:.2f}")
            if snap.description:
                parts.append(snap.description[:60])
            ui.label(" | ".join(parts)).classes(
                "text-xs text-slate-600 px-2"
            )


def _inspector_suggestions(
    state: AppState, node_id: str, node_type: str, name: str
) -> None:
    """Context-aware NL suggestion chips that drive the bottom command bar."""
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
        ui.separator().classes("q-my-sm")
        ui.label("Try:").classes("text-xs text-slate-500")
        with ui.row().classes("w-full gap-1 flex-wrap"):
            for text, qtype in suggestions[:6]:
                ui.chip(
                    text[:50],
                    icon="flash_on",
                    on_click=lambda t=text, q=qtype: state.emit(
                        StateEvent.QUERY_STARTED,
                        suggestion=t,
                        query_type=q,
                    ),
                ).props("dense outline clickable")
