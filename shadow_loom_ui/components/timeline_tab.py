"""Timeline tab — event timeline visualization + version tree.

Top: Events plotted by fabula_time with type-colored dots.
Bottom: DB version tree with branch/rollback support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom.models import WorldStateV1
from shadow_loom_ui import db
from shadow_loom_ui.state import AppState, StateEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_EVENT_TYPE_COLORS = {
    "choice": "#2196F3",
    "outcome": "#FF9800",
    "revelation": "#9C27B0",
    "action": "#4CAF50",
    "dialogue": "#00BCD4",
    "transition": "#607D8B",
}

_SOURCE_BADGE_COLORS = {
    "ingestion": "green",
    "pipeline": "blue",
    "manual_edit": "purple",
    "manual_save": "amber",
    "rollback": "orange",
    "observation": "cyan",
    "intervention": "red",
    "counterfactual": "pink",
    "directive": "teal",
    "interrogate": "indigo",
    "general": "grey",
}


def build_timeline_tab(state: AppState) -> None:
    """Build the Timeline tab layout."""

    with ui.column().classes("w-full h-full"):
        # ---- Event Timeline ----
        ui.label("Event Timeline").classes("text-h6 q-pa-sm")
        event_container = ui.column().classes("w-full q-pa-sm")
        _render_event_timeline(state, event_container)

        ui.separator()

        # ---- Version Tree ----
        ui.label("Version History").classes("text-h6 q-pa-sm")
        version_container = ui.column().classes("w-full q-pa-sm")
        _render_version_tree(state, version_container)

    # Subscribe to updates
    state.on(StateEvent.WORLD_STATE_CHANGED, lambda: _render_event_timeline(state, event_container))
    state.on(StateEvent.VERSION_CHANGED, lambda **kw: _render_version_tree(state, version_container))


# =====================================================================
# Event Timeline
# =====================================================================

def _render_event_timeline(state: AppState, container) -> None:
    """Render events as a horizontal timeline grouped by fabula_time."""
    container.clear()
    ws = state.world_state
    if ws is None:
        with container:
            ui.label("No world model loaded.").classes("text-body2 text-grey")
        return

    events = sorted(ws.events, key=lambda e: (e.fabula_time or 0, getattr(e, "syuzhet_index", 0) or 0))
    if not events:
        with container:
            ui.label("No events in world model.").classes("text-body2 text-grey")
        return

    with container:
        # Summary stats
        with ui.row().classes("gap-4 q-mb-sm"):
            ui.label(f"{len(events)} events").classes("text-caption text-grey")
            types = {}
            for ev in events:
                t = getattr(ev, "event_type", "unknown") or "unknown"
                types[t] = types.get(t, 0) + 1
            for t, count in sorted(types.items()):
                color = _EVENT_TYPE_COLORS.get(t, "#607D8B")
                ui.chip(f"{t}: {count}").props(f'dense outline style="color:{color}"')

        # Timeline — scrollable horizontal strip
        with ui.scroll_area().classes("w-full").props("horizontal"):
            with ui.row().classes("gap-1 flex-nowrap q-pa-sm"):
                for ev in events:
                    color = _EVENT_TYPE_COLORS.get(
                        getattr(ev, "event_type", ""), "#607D8B"
                    )
                    with ui.card().classes("w-48 cursor-pointer").props("dense").style(
                        f"border-left: 3px solid {color};"
                    ):
                        with ui.row().classes("items-center gap-1"):
                            ui.label(f"t={ev.fabula_time}").classes("text-caption text-bold")
                            if hasattr(ev, "event_type") and ev.event_type:
                                ui.badge(ev.event_type, color=color).props("dense")
                        desc = (ev.description or ev.id)[:60]
                        ui.label(desc).classes("text-caption")
                        if ev.actors:
                            ui.label(f"Actors: {', '.join(ev.actors[:3])}").classes(
                                "text-caption text-grey"
                            )

        # Entity tracks
        if events and ws.entities:
            with ui.expansion("Entity Tracks", icon="timeline").classes("w-full q-mt-sm"):
                for ent in ws.entities[:10]:
                    if not hasattr(ent, "state_timeline") or not ent.state_timeline:
                        continue
                    with ui.expansion(ent.name).classes("w-full").props("dense"):
                        for entry in ent.state_timeline:
                            with ui.row().classes("items-center gap-2"):
                                ui.label(f"t={entry.fabula_time}").classes(
                                    "text-caption text-bold w-16"
                                )
                                ui.label(entry.description).classes("text-caption")


# =====================================================================
# Version Tree
# =====================================================================

def _render_version_tree(state: AppState, container) -> None:
    """Render the DB version tree with load/rollback controls."""
    container.clear()

    if state.project_id is None:
        with container:
            ui.label("No project loaded.").classes("text-body2 text-grey")
        return

    tree_data = db.get_version_tree(state.project_id)
    if not tree_data:
        with container:
            ui.label("No versions yet.").classes("text-body2 text-grey")
        return

    with container:
        with ui.list().props("bordered separator").classes("w-full"):
            for v in tree_data:
                is_current = (v["id"] == state.current_version_row_id)
                with ui.item().classes("" if not is_current else "bg-primary bg-opacity-10"):
                    with ui.item_section().props("side"):
                        if is_current:
                            ui.icon("radio_button_checked", color="primary")
                        else:
                            ui.icon("radio_button_unchecked", color="grey")

                    with ui.item_section():
                        with ui.row().classes("items-center gap-2"):
                            ui.item_label(f"v{v['version']}").classes("text-bold")
                            source = v.get("source", "")
                            badge_color = _SOURCE_BADGE_COLORS.get(source, "grey")
                            ui.badge(source, color=badge_color).props("dense")
                            if v.get("label"):
                                ui.badge(v["label"], color="blue").props("dense outline")
                            if v.get("is_bookmarked"):
                                ui.icon("bookmark", color="amber", size="xs")

                        desc = v.get("description", "")
                        if desc:
                            ui.item_label(desc).props("caption")

                        # Changeset summary
                        changeset = v.get("changeset_json")
                        if changeset:
                            ui.item_label("Has changeset").props("caption").classes("text-grey")

                        ui.item_label(
                            v.get("created_at", "")
                        ).props("caption").classes("text-grey")

                    with ui.item_section().props("side"):
                        with ui.row().classes("gap-1"):
                            if not is_current:
                                def _load_version(vid=v["id"], vnum=v["version"]):
                                    ver = db.get_version_by_id(vid)
                                    if ver is None:
                                        ui.notify("Version not found", type="warning")
                                        return
                                    try:
                                        ws = WorldStateV1.model_validate_json(ver.world_state_json)
                                        state.load_world_state(ws)
                                        state.current_version_row_id = vid
                                        state.emit(StateEvent.VERSION_CHANGED, version=vnum)
                                        ui.notify(f"Loaded v{vnum}")
                                        _render_version_tree(state, container)
                                    except Exception as e:
                                        ui.notify(f"Load failed: {e}", type="negative")

                                ui.button("Load", on_click=_load_version).props("flat dense")

                            # Bookmark toggle
                            if state.user_id:
                                bm_icon = "bookmark" if v.get("is_bookmarked") else "bookmark_outline"

                                def _toggle_bm(vid=v["id"]):
                                    current = v.get("is_bookmarked", False)
                                    db.bookmark_version(vid, not current)
                                    _render_version_tree(state, container)

                                ui.button(icon=bm_icon, on_click=_toggle_bm).props("flat dense round")
