# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dashboard page — project gallery, examples, activity feed, and quick actions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui import db
from shadow_loom_ui.components.dialogs import build_ingest_dialog
from shadow_loom_ui.theme import (
    PAGE_TITLE_CLS,
    SECTION_TITLE_CLS,
    feather,
)

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState

logger = logging.getLogger(__name__)


# =====================================================================
# Dashboard
# =====================================================================

def build_dashboard(state: AppState) -> None:
    """Build the main dashboard layout."""

    with ui.column().classes("w-full max-w-6xl mx-auto p-6 md:p-8 gap-6"):
        # ---- Quick actions ----
        with ui.row().classes("w-full items-center justify-between mb-2"):
            with ui.column().classes("gap-0"):
                ui.label("Dashboard").classes(PAGE_TITLE_CLS)
                ui.label(
                    "Neuro-symbolic reasoning for ancestral and parallel worlds."
                ).classes("text-sm text-slate-500 italic -mt-1")
            with ui.row().classes("gap-2"):
                ingest_dlg = build_ingest_dialog(state)
                with ui.button(on_click=ingest_dlg.open).props(
                    "unelevated color=primary no-caps"
                ).classes("rounded-lg shadow-sm"):
                    with ui.row().classes("items-center gap-2"):
                        feather("plus")
                        ui.label("New Project")

        # ---- My Projects ----
        ui.label("My Projects").classes(SECTION_TITLE_CLS)
        projects_container = ui.row().classes("w-full gap-4 flex-wrap")
        _render_project_cards(state, projects_container)

        # ---- Usage Dashboard ----  
        if state.user_id:
            from shadow_loom_ui.components.usage_dashboard import render_usage_dashboard
            render_usage_dashboard(state)

        # ---- Starred Projects ----
        if state.user_id:
            starred = db.list_starred_projects(state.user_id)
            if starred:
                ui.label("Starred Projects").classes(
                    SECTION_TITLE_CLS + " mt-4"
                )
                starred_container = ui.row().classes("w-full gap-4 flex-wrap")
                _render_starred_cards(starred, starred_container)

        # ---- Example Projects (pre-built world models) ----
        examples = db.list_example_projects()
        if examples:
            ui.label("Example Plots").classes(SECTION_TITLE_CLS + " mt-4")
            ui.label(
                "Pre-built world models. Click an example to copy it into "
                "your projects \u2014 no ingestion required."
            ).classes("text-xs text-slate-500 -mt-2")
            with ui.row().classes("w-full gap-3 flex-wrap"):
                for ex in examples:
                    _example_project_chip(state, ex)

        # ---- Recent Activity ----
        if state.user_id:
            activities = db.get_user_activity(state.user_id, limit=20)
            if activities:
                ui.label("Recent Activity").classes(
                    SECTION_TITLE_CLS + " mt-4"
                )
                with ui.card().classes(
                    "w-full p-0 bg-white border border-slate-200 "
                    "rounded-xl shadow-sm overflow-hidden"
                ):
                    for act in activities[:10]:
                        with ui.row().classes(
                            "w-full items-center px-4 py-3 "
                            "border-b border-slate-100 last:border-b-0"
                        ):
                            with ui.column().classes("gap-0"):
                                ui.label(act["summary"] or act["action"]).classes(
                                    "text-sm text-slate-700"
                                )
                                ui.label(
                                    str(act["created_at"])[:16]
                                    if act["created_at"]
                                    else ""
                                ).classes("text-xs text-slate-400")


# =====================================================================
# Project cards
# =====================================================================

def _render_project_cards(state: AppState, container: ui.row) -> None:
    """Populate the project card gallery."""
    container.clear()
    user_id = state.user_id
    # Anonymous (logged-out) callers must not enumerate the global
    # project table: ``db.list_projects()`` with no user_id returns
    # every non-example project, including private ones owned by
    # other users. Example projects are surfaced separately via
    # ``list_example_projects`` and copied on selection.
    projects = db.list_projects(user_id=user_id) if user_id else []

    if not projects:
        with container:
            with ui.card().classes(
                "w-72 bg-white border border-slate-200 rounded-xl shadow-sm p-6 "
                "text-center"
            ):
                with ui.row().classes("justify-center w-full mb-2"):
                    feather("folder", size="xl", color="#94a3b8")
                ui.label("No projects yet").classes(
                    "text-sm font-medium text-slate-700"
                )
                ui.label("Create a new project to get started.").classes(
                    "text-xs text-slate-400"
                )
        return

    with container:
        for p in projects:
            with ui.card().classes(
                "w-72 bg-white border border-slate-200 rounded-xl shadow-sm "
                "p-5 cursor-pointer sl-card-hover"
            ).on(
                "click", lambda pid=p["id"]: ui.navigate.to(f"/project/{pid}")
            ):
                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    feather("book-open", color="#F26B5E")
                    ui.label(p["name"]).classes(
                        "text-base font-semibold text-slate-800 ellipsis flex-grow"
                    )
                    # Owner-only project actions menu (delete with confirm)
                    if user_id is not None and p.get("owner_id") == user_id:
                        with ui.button(icon="more_vert").props(
                            "flat dense round size=sm color=secondary"
                        ).on("click.stop", lambda: None):
                            with ui.menu().props("auto-close"):
                                ui.menu_item(
                                    "Delete project",
                                    on_click=lambda proj=p: _confirm_delete_project(
                                        state, container, proj
                                    ),
                                ).classes("text-negative")
                if p.get("description"):
                    ui.label(p["description"][:80]).classes(
                        "text-xs text-slate-500 ellipsis mt-1"
                    )
                with ui.row().classes("w-full justify-between mt-3"):
                    ui.label(f"v{p.get('version_count', 0)}").classes(
                        "text-xs text-slate-400"
                    )
                    ui.label(p.get("updated_at", "")).classes(
                        "text-xs text-slate-400"
                    )
                with ui.row().classes("gap-1 mt-2"):
                    if p.get("is_public"):
                        ui.badge("public", color="primary").props("dense")
                    if p.get("is_template"):
                        ui.badge("example", color="secondary").props("dense")


async def _confirm_delete_project(
    state: AppState, container: ui.row, project: dict
) -> None:
    """Show a confirmation dialog and delete the project on accept."""
    if state.user_id is None:
        ui.notify("Sign in required to delete projects.", type="warning")
        return

    confirm = ui.dialog()
    with confirm, ui.card().classes("p-4 gap-2 max-w-md"):
        ui.label(f'Delete "{project.get("name", "?")}"?').classes(
            "text-lg font-semibold text-slate-800"
        )
        ui.label(
            "This permanently removes the project and ALL of its versions, "
            "activity, stars, and member access. This cannot be undone."
        ).classes("text-sm text-slate-600")
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=confirm.close).props(
                "flat dense no-caps"
            )
            ui.button(
                "Delete", on_click=lambda: confirm.submit("delete")
            ).props("unelevated dense color=negative no-caps")
    result = await confirm
    if result != "delete":
        return
    try:
        db.delete_project(project["id"], state.user_id)
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
    _render_project_cards(state, container)


def _render_starred_cards(starred_ids: list[int], container: ui.row) -> None:
    """Render starred project cards."""
    container.clear()
    # starred_ids is a list of project IDs — fetch each project
    with container:
        for pid in starred_ids:
            proj = db.get_project(pid)
            if proj is None:
                continue
            with ui.card().classes(
                "w-72 bg-white border border-slate-200 rounded-xl shadow-sm "
                "p-5 cursor-pointer sl-card-hover"
            ).on(
                "click", lambda p_id=pid: ui.navigate.to(f"/project/{p_id}")
            ):
                with ui.row().classes("items-center gap-2"):
                    feather("star", color="#F5B43C")
                    ui.label(proj.name).classes(
                        "text-base font-semibold text-slate-800 ellipsis"
                    )


# =====================================================================
# Example project chip — fork pre-built world model into user account
# =====================================================================

def _example_project_chip(state: AppState, example: dict) -> None:
    """Render a chip for a pre-built example world model.

    Clicking forks the example project into the current user's account
    and navigates to it. No ingestion runs.
    """

    def _use_example():
        if state.user_id is None:
            ui.notify("Sign in to copy this example into your projects.",
                      type="warning")
            return
        try:
            forked = db.fork_project(
                source_project_id=example["id"],
                new_owner_id=state.user_id,
                new_name=example["name"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Forking example failed")
            ui.notify(f"Failed to copy example: {exc}", type="negative")
            return
        if forked is None:
            ui.notify("Example has no saved version yet.", type="warning")
            return
        ui.notify(f"Copied '{example['name']}' to your projects.",
                  type="positive")
        ui.navigate.to(f"/project/{forked.id}")

    ui.chip(example["name"], icon="auto_stories", on_click=_use_example).props(
        "clickable outline color=secondary"
    )
