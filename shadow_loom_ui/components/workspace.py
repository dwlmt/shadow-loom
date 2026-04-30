"""Project workspace — left version sidebar + tabbed canvas + bottom chat drawer.

Layout:
  ┌──────┬───────────────────────────────────────────┐
  │ Vers │  Story │ Explorer │ World │ Causality │ Audit │  │
  │ Tree │  (tab content — full canvas)               │
  │      ├─────────────────────────────────────────────┤
  │      │  Bottom Drawer: Chat / Query Bar           │
  └──────┴───────────────────────────────────────────┘

The project no longer exposes an explicit "Save" button. Pipeline
results (interventions, counterfactuals, etc.) are autosaved as new
versions by ``AppState._save_version_to_db``. The Story tab still
offers a manual "Save & Re-ingest" button next to the source-text
editor for the heavyweight LLM re-ingestion path.
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

# Tab definitions: (key, material-icon-name, label).  The Material font is
# remapped to its Outlined variant in theme.py so these match Feather visually.
_TABS = [
    ("story", "auto_stories", "Story"),
    ("explorer", "travel_explore", "Explorer"),
    ("world", "hub", "World"),
    ("causality", "account_tree", "Causality"),
    ("reasoning", "psychology", "Reasoning"),
    ("audit", "fact_check", "Audit"),
    ("editor", "edit_note", "Edit"),
    ("export", "ios_share", "Export"),
]

# Hover-overlay help text shown on each tab. Kept short so the tooltip
# fits inside one or two lines on smaller viewports.
_TAB_HELP: dict[str, str] = {
    "story": (
        "Story \u2014 read the canonical narrative for the active version, "
        "scrub the syuzhet timeline, and inspect per-event affective scores."
    ),
    "explorer": (
        "Explorer \u2014 browse entities, locations, objects, and world traits "
        "in a navigable register. Drill into any node to see its state at a "
        "chosen fabula time."
    ),
    "world": (
        "World \u2014 the world-state-at-time inspector. Slide the time cursor "
        "to see how every entity's traits, beliefs, status, and location "
        "evolve across the fabula."
    ),
    "causality": (
        "Causality \u2014 visualise the typed causal/social/spatial/information "
        "graph and the affective-dashboard heatmaps (suspense, irony, "
        "surprise) over the syuzhet."
    ),
    "reasoning": (
        "Reasoning \u2014 run natural-language queries against the world model "
        "and inspect the engine's step-by-step reasoning trace, including "
        "abduction, intervention, and counterfactual replay."
    ),
    "audit": (
        "Audit \u2014 the activity log for this project: every ingestion, "
        "query, manual edit, and version save with diffs and the LLM "
        "auditor's verdict."
    ),
    "editor": (
        "Edit \u2014 manually add, remove, or modify any node or edge in the "
        "world model. Validate before saving \u2014 errors block, warnings can "
        "be acknowledged."
    ),
    "export": (
        "Export \u2014 download the current world state as JSON, render the "
        "story to plain prose, or export the causal graph for external "
        "analysis."
    ),
}


def build_workspace(state: AppState, project_id: int) -> None:
    """Load a project and render the workspace."""

    # ---- Load project from DB ----
    project = db.get_project(project_id)
    if project is None:
        ui.label("Project not found.").classes(
            "text-2xl text-negative q-pa-lg"
        )
        ui.button(
            "Back to Dashboard", on_click=lambda: ui.navigate.to("/")
        ).props("unelevated color=primary no-caps")
        return

    # Check access. Fail-closed when no user_id is present \u2014 only public
    # projects (or projects in fully-open dev mode) are reachable anonymously.
    if state.user_id is not None:
        role = db.get_user_project_role(project_id, state.user_id)
        if role is None and not project.is_public and project.owner_id != state.user_id:
            ui.label("Access denied.").classes(
                "text-2xl text-negative q-pa-lg"
            )
            ui.button(
                "Back to Dashboard", on_click=lambda: ui.navigate.to("/")
            ).props("unelevated color=primary no-caps")
            return
    elif not project.is_public:
        ui.label("Access denied.").classes(
            "text-2xl text-negative q-pa-lg"
        )
        ui.button(
            "Back to Dashboard", on_click=lambda: ui.navigate.to("/")
        ).props("unelevated color=primary no-caps")
        return

    # Load latest version (or the user's active-version pointer, so a
    # version selected in a previous session is restored).
    initial_ver = None
    if state.user_id is not None:
        try:
            initial_ver = db.get_active_version(project_id, state.user_id)
        except Exception:
            logger.exception("Failed to read active-version pointer")
            initial_ver = None
    if initial_ver is None:
        initial_ver = db.get_latest_version(project_id)

    if initial_ver is not None:
        try:
            ws = WorldStateV1.model_validate_json(initial_ver.world_state_json)
            state.load_project(
                project_id=project_id,
                project_name=project.name,
                world_state=ws,
                version_row_id=initial_ver.id,
                raw_text=project.raw_text,
            )
            # Mirror the loaded version into the active-version pointer so
            # MCP read tools default to the same view as the UI.
            if state.user_id is not None:
                try:
                    db.set_active_version(
                        project_id, state.user_id, initial_ver.id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to update active-version pointer on load"
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
    with ui.row().classes(
        "w-full items-center bg-white border-b border-slate-200 "
        "px-6 py-3 gap-3"
    ):
        ui.label(project.name).classes(
            "text-xl font-semibold text-slate-800"
        )
        if project.description:
            ui.label(f"— {project.description}").classes(
                "text-sm text-slate-500"
            )
        ui.space()

        # Delete button (owner-only) with confirmation
        is_owner = (
            state.user_id is not None and project.owner_id == state.user_id
        )
        if is_owner:
            async def _confirm_delete():
                confirm = ui.dialog()
                with confirm, ui.card().classes("p-4 gap-2 max-w-md"):
                    ui.label(f'Delete "{project.name}"?').classes(
                        "text-lg font-semibold text-slate-800"
                    )
                    ui.label(
                        "This permanently removes the project and ALL of its "
                        "versions, activity, stars, and member access. This "
                        "cannot be undone."
                    ).classes("text-sm text-slate-600")
                    with ui.row().classes("w-full justify-end gap-2 mt-2"):
                        ui.button(
                            "Cancel", on_click=confirm.close
                        ).props("flat dense no-caps")
                        ui.button(
                            "Delete",
                            on_click=lambda: confirm.submit("delete"),
                        ).props("unelevated dense color=negative no-caps")
                result = await confirm
                if result != "delete":
                    return
                try:
                    db.delete_project(project_id, state.user_id)
                except db.ProjectDeleteError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                except PermissionError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Project delete failed")
                    ui.notify(f"Delete failed: {exc}", type="negative")
                    return
                ui.notify("Project deleted", type="positive")
                ui.navigate.to("/")

            ui.button(
                "Delete", icon="delete", on_click=_confirm_delete
            ).props("flat dense color=negative no-caps")

        # Star toggle
        if state.user_id:
            is_starred = db.is_starred(project_id, state.user_id)
            star_icon = "star" if is_starred else "star_border"
            star_color = "warning" if is_starred else "secondary"
            star_btn = ui.button(icon=star_icon).props(
                f"flat dense round color={star_color}"
            )

            def _toggle_star():
                now_starred = db.toggle_star(project_id, state.user_id)
                new_icon = "star" if now_starred else "star_border"
                new_color = "warning" if now_starred else "secondary"
                star_btn.props(f"flat dense round color={new_color} icon={new_icon}")
                ui.notify("Starred!" if now_starred else "Unstarred")

            star_btn.on("click", _toggle_star)

    # ---- Main layout: left version sidebar + tabs + bottom drawer ----
    with ui.splitter(value=18).classes("w-full bg-white").style(
        "height: calc(100vh - 140px)"
    ) as main_split:
        # Left rail: version-history tree (always visible)
        with main_split.before:
            from shadow_loom_ui.components.version_sidebar import (
                build_version_sidebar,
            )
            build_version_sidebar(state)

        # Right: tabs + bottom drawer
        with main_split.after:
            with ui.column().classes("w-full h-full gap-0 bg-white"):
                # Tab bar
                with ui.tabs().props(
                    "dense indicator-color=primary active-color=primary "
                    "align=left no-caps"
                ).classes(
                    "w-full bg-white border-b border-slate-200 px-4"
                ) as tabs:
                    for key, icon_name, label in _TABS:
                        with ui.tab(key, label=label, icon=icon_name):
                            ui.tooltip(_TAB_HELP[key]).classes(
                                "max-w-xs text-xs"
                            )

                # Track which top-level tab is visible so component
                # panels can skip refreshing when they're off-screen.
                state.set_active_path("story")

                def _on_top_tab(e):
                    val = getattr(e, "args", None)
                    if isinstance(val, str):
                        state.set_active_path(val)

                tabs.on("update:model-value", _on_top_tab)

                # Tab panels
                with ui.tab_panels(tabs, value="story").classes(
                    "w-full flex-grow bg-slate-50"
                ).style("overflow: auto"):
                    with ui.tab_panel("story").classes("q-pa-none h-full"):
                        from shadow_loom_ui.components.story_tab import build_story_tab
                        build_story_tab(state)

                    with ui.tab_panel("explorer").classes("q-pa-none h-full"):
                        from shadow_loom_ui.components.explorer_tab import (
                            build_explorer_tab,
                        )
                        build_explorer_tab(state)

                    with ui.tab_panel("world").classes("q-pa-none h-full"):
                        from shadow_loom_ui.components.world_tab import build_world_tab
                        build_world_tab(state)

                    with ui.tab_panel("causality").classes("q-pa-none h-full"):
                        from shadow_loom_ui.components.causality_tab import build_causality_tab
                        build_causality_tab(state)

                    with ui.tab_panel("reasoning").classes("q-pa-none h-full"):
                        from shadow_loom_ui.components.reasoning_tab import (
                            build_reasoning_tab,
                        )
                        build_reasoning_tab(state)

                    with ui.tab_panel("audit").classes("q-pa-none h-full"):
                        from shadow_loom_ui.components.audit_tab import build_audit_tab
                        build_audit_tab(state)

                    with ui.tab_panel("editor").classes("q-pa-none h-full"):
                        from shadow_loom_ui.components.editor_tab import (
                            build_editor_tab,
                        )
                        build_editor_tab(state)

                    with ui.tab_panel("export").classes("q-pa-none h-full"):
                        from shadow_loom_ui.components.export_tab import build_export_tab
                        build_export_tab(state)

                # Bottom chat drawer
                from shadow_loom_ui.components.chat import build_chat_drawer
                build_chat_drawer(state)
