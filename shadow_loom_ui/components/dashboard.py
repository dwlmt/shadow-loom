# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dashboard page — project gallery, examples, activity feed, and quick actions."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import app, ui

from shadow_loom.ingestion import ExtractionConfig, run_extraction
from shadow_loom_ui import config, db
from shadow_loom_ui.theme import (
    CARD_CLS,
    CARD_TIGHT_CLS,
    PAGE_TITLE_CLS,
    SECTION_TITLE_CLS,
    feather,
)

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState

logger = logging.getLogger(__name__)

_SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_plots"


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
                ingest_dlg = _build_ingest_dialog(state)
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
    projects = db.list_projects(user_id=user_id) if user_id else db.list_projects()

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


# =====================================================================
# Ingest dialog — create new project from raw text
# =====================================================================

def _build_ingest_dialog(state: AppState) -> ui.dialog:
    """Dialog for ingesting raw narrative text into a new project."""

    dialog = ui.dialog().props("persistent maximized")

    with dialog, ui.card().classes("w-full max-w-3xl"):
        ui.label("Ingest Narrative Text").classes(
            "text-xl font-semibold text-slate-800"
        )
        ui.label(
            "Paste or upload raw story text. The pipeline will extract "
            "entities, events, locations, objects, and all topology edges."
        ).classes("text-sm text-slate-500")

        project_name = ui.input("Project Name", value="New Story").classes("w-full")
        project_desc = ui.input("Description (optional)").classes("w-full")
        text_area = ui.textarea(
            "Story Text",
            placeholder="Paste the full narrative text here...",
        ).classes("w-full").props("rows=15")

        # File upload
        ui.label("Or upload a .txt file:").classes("text-xs text-slate-500 mt-2")

        async def _handle_upload(e):
            content = e.content.read().decode("utf-8")
            text_area.value = content
            ui.notify(f"Loaded {len(content)} characters")

        ui.upload(on_upload=_handle_upload, auto_upload=True).props(
            "accept=.txt flat dense"
        ).classes("w-full")

        # Sample plots
        if _SAMPLE_DIR.exists():
            sample_files = sorted(_SAMPLE_DIR.glob("*.txt"))
            if sample_files:
                ui.label("Or load a sample plot:").classes("text-xs text-slate-500 mt-2")
                sample_select = ui.select(
                    options={str(f): f.stem.replace("_", " ").title()
                             for f in sample_files},
                    label="Sample",
                ).classes("w-64")

                def _load_sample():
                    path = sample_select.value
                    if path:
                        text_area.value = Path(path).read_text(encoding="utf-8")
                        project_name.value = Path(path).stem.replace("_", " ").title()

                sample_select.on("update:model-value", _load_sample)

        status = ui.label("").classes("text-sm text-slate-600 mt-2")
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        progress.set_visibility(False)

        with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            async def _run_ingestion():
                text = text_area.value.strip()
                if not text:
                    ui.notify("Enter some text first", type="warning")
                    return

                status.set_text("Extracting world model from text...")
                progress.set_visibility(True)
                progress.value = 0.1

                try:
                    cfg = ExtractionConfig(
                        chunk_strategy="act_headings",
                        fabula_time_spacing=100,
                        output_retries=5,
                        max_correction_retries=1,
                    )

                    ws, report = await asyncio.to_thread(
                        run_extraction, text, cfg,
                    )
                    progress.value = 0.8

                    proj = db.create_project(
                        name=project_name.value or "Untitled",
                        raw_text=text,
                        description=project_desc.value or None,
                        owner_id=state.user_id,
                    )
                    ver = db.save_version(
                        project_id=proj.id,
                        world_state_json=ws.model_dump_json(),
                        source="ingestion",
                        description="Initial ingestion",
                        user_id=state.user_id,
                    )

                    progress.value = 1.0
                    status.set_text(
                        f"Done! {len(ws.entities)} entities, {len(ws.events)} events, "
                        f"{len(ws.locations)} locations. "
                        f"Validation: {'PASS' if report.is_valid else 'FAIL'}"
                    )

                    # Log activity
                    if state.user_id:
                        try:
                            db.log_activity(
                                project_id=proj.id,
                                action="ingestion",
                                user_id=state.user_id,
                                summary=f"Ingested story: {proj.name}",
                                version_id=ver.id,
                            )
                        except Exception:
                            pass

                    ui.notify("World model created!", type="positive")
                    await asyncio.sleep(0.5)
                    dialog.close()
                    ui.navigate.to(f"/project/{proj.id}")

                except Exception as e:
                    logger.exception("Ingestion failed")
                    status.set_text(f"Error: {e}")
                    ui.notify(f"Ingestion failed: {e}", type="negative")
                finally:
                    progress.set_visibility(False)

            ui.button("Ingest", on_click=_run_ingestion, icon="auto_fix_high").props(
                "unelevated color=primary no-caps"
            ).classes("rounded-lg shadow-sm")

    return dialog
