"""Project workspace — tabbed work surface with left navigation rail.

Loads a project from the DB and renders 5 tabs:
Story | World | Timeline | Audit | Export
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import app, ui

from shadow_loom.models import WorldStateV1
from shadow_loom_ui import db
from shadow_loom_ui.state import AppState, StateEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Tab definitions: (key, icon, label)
_TABS = [
    ("story", "menu_book", "Story"),
    ("world", "hub", "World"),
    ("timeline", "timeline", "Timeline"),
    ("audit", "fact_check", "Audit"),
    ("export", "ios_share", "Export"),
]


def build_workspace(state: AppState, project_id: int) -> None:
    """Load a project and render the tabbed workspace."""

    # ---- Load project from DB ----
    project = db.get_project(project_id)
    if project is None:
        ui.label("Project not found.").classes("text-h5 text-negative q-pa-lg")
        ui.button("Back to Dashboard", on_click=lambda: ui.navigate.to("/"))
        return

    # Check access (basic: owner or member)
    if state.user_id:
        role = db.get_user_project_role(project_id, state.user_id)
        if role is None and not project.is_public and project.user_id != state.user_id:
            ui.label("Access denied.").classes("text-h5 text-negative q-pa-lg")
            ui.button("Back to Dashboard", on_click=lambda: ui.navigate.to("/"))
            return

    # Load latest version
    latest = db.get_latest_version(project_id)
    if latest is not None:
        try:
            ws = WorldStateV1.model_validate_json(latest.world_state_json)
            state.load_project(
                project_id=project_id,
                project_name=project.name,
                world_state=ws,
                version_row_id=latest.id,
                raw_text=project.raw_text,
            )
        except Exception as e:
            logger.exception("Failed to load world state")
            ui.notify(f"Failed to load world state: {e}", type="negative")
            state.project_id = project_id
            state.project_name = project.name
    else:
        state.project_id = project_id
        state.project_name = project.name

    # ---- Toolbar ----
    with ui.row().classes("w-full items-center q-px-md q-pt-sm gap-2"):
        ui.label(project.name).classes("text-h5")
        if project.description:
            ui.label(f"— {project.description}").classes("text-subtitle1 text-grey")
        ui.space()

        # Save button
        async def _save():
            if state.world_state is None:
                ui.notify("Nothing to save", type="warning")
                return
            ver = db.save_version(
                project_id=project_id,
                world_state_json=state.to_json(),
                ancestor_id=state.current_version_row_id,
                source="manual_save",
                description="Manual save",
                user_id=state.user_id,
            )
            state.current_version_row_id = ver.id
            ui.notify(f"Saved v{ver.version}", type="positive")

        ui.button("Save", icon="save", on_click=_save).props("flat dense")

        # Star toggle
        if state.user_id:
            is_starred = db.is_starred(project_id, state.user_id)
            star_icon = "star" if is_starred else "star_outline"
            star_btn = ui.button(icon=star_icon).props("flat dense round")

            def _toggle_star():
                now_starred = db.toggle_star(project_id, state.user_id)
                star_btn.props(f'icon={"star" if now_starred else "star_outline"}')
                ui.notify("Starred!" if now_starred else "Unstarred")

            star_btn.on("click", _toggle_star)

    # ---- Tabbed layout ----
    with ui.tabs().classes("w-full") as tabs:
        tab_refs = {}
        for key, icon, label in _TABS:
            tab_refs[key] = ui.tab(key, label=label, icon=icon)

    with ui.tab_panels(tabs, value="story").classes("w-full flex-grow"):
        with ui.tab_panel("story").classes("q-pa-none"):
            from shadow_loom_ui.components.story_tab import build_story_tab
            build_story_tab(state)

        with ui.tab_panel("world").classes("q-pa-none"):
            from shadow_loom_ui.components.world_tab import build_world_tab
            build_world_tab(state)

        with ui.tab_panel("timeline").classes("q-pa-none"):
            from shadow_loom_ui.components.timeline_tab import build_timeline_tab
            build_timeline_tab(state)

        with ui.tab_panel("audit").classes("q-pa-none"):
            from shadow_loom_ui.components.audit_tab import build_audit_tab
            build_audit_tab(state)

        with ui.tab_panel("export").classes("q-pa-none"):
            from shadow_loom_ui.components.export_tab import build_export_tab
            build_export_tab(state)
