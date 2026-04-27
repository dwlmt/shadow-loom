"""Dashboard page — project gallery, examples, activity feed, and quick actions."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import app, ui

from shadow_loom.ingestion import ExtractionConfig, run_extraction
from shadow_loom_ui import config, db

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState

logger = logging.getLogger(__name__)

_SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_plots"


# =====================================================================
# Dashboard
# =====================================================================

def build_dashboard(state: AppState) -> None:
    """Build the main dashboard layout."""

    with ui.column().classes("w-full max-w-6xl mx-auto q-pa-lg gap-6"):
        # ---- Quick actions ----
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Dashboard").classes("text-h4")
            with ui.row().classes("gap-2"):
                ingest_dlg = _build_ingest_dialog(state)
                ui.button("New Project", icon="add", on_click=ingest_dlg.open).props(
                    "color=primary no-caps"
                )

        # ---- My Projects ----
        ui.label("My Projects").classes("text-h6")
        projects_container = ui.row().classes("w-full gap-4 flex-wrap")
        _render_project_cards(state, projects_container)

        # ---- Starred Projects ----
        if state.user_id:
            starred = db.list_starred_projects(state.user_id)
            if starred:
                ui.label("Starred Projects").classes("text-h6 q-mt-md")
                starred_container = ui.row().classes("w-full gap-4 flex-wrap")
                _render_starred_cards(starred, starred_container)

        # ---- Example Projects ----
        if _SAMPLE_DIR.exists():
            samples = sorted(_SAMPLE_DIR.glob("*.txt"))
            if samples:
                ui.label("Example Plots").classes("text-h6 q-mt-md")
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    for sample_file in samples:
                        title = sample_file.stem.replace("_", " ").title()
                        _example_chip(state, sample_file, title)

        # ---- Recent Activity ----
        if state.user_id:
            activities = db.get_user_activity(state.user_id, limit=20)
            if activities:
                ui.label("Recent Activity").classes("text-h6 q-mt-md")
                with ui.list().props("bordered separator").classes("w-full"):
                    for act in activities[:10]:
                        with ui.item():
                            with ui.item_section():
                                ui.item_label(act["summary"] or act["action"])
                                ui.item_label(
                                    str(act["created_at"])[:16] if act["created_at"] else ""
                                ).props("caption")


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
            with ui.card().classes("w-72 q-pa-md"):
                ui.icon("folder_open", size="xl", color="grey").classes("q-mb-sm")
                ui.label("No projects yet").classes("text-body1 text-grey")
                ui.label("Create a new project to get started.").classes("text-caption text-grey")
        return

    with container:
        for p in projects:
            with ui.card().classes("w-72 cursor-pointer hover:shadow-lg transition-shadow").on(
                "click", lambda pid=p["id"]: ui.navigate.to(f"/project/{pid}")
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("auto_stories", color="primary")
                    ui.label(p["name"]).classes("text-subtitle1 ellipsis")
                if p.get("description"):
                    ui.label(p["description"][:80]).classes("text-caption text-grey ellipsis")
                with ui.row().classes("w-full justify-between q-mt-sm"):
                    ui.label(f"v{p.get('version_count', 0)}").classes("text-caption text-grey")
                    ui.label(p.get("updated_at", "")).classes("text-caption text-grey")
                with ui.row().classes("gap-1"):
                    if p.get("is_public"):
                        ui.badge("public", color="teal").props("dense")
                    if p.get("is_template"):
                        ui.badge("example", color="blue-grey").props("dense")


def _render_starred_cards(starred_ids: list[int], container: ui.row) -> None:
    """Render starred project cards."""
    container.clear()
    # starred_ids is a list of project IDs — fetch each project
    with container:
        for pid in starred_ids:
            proj = db.get_project(pid)
            if proj is None:
                continue
            with ui.card().classes("w-72 cursor-pointer hover:shadow-lg transition-shadow").on(
                "click", lambda p_id=pid: ui.navigate.to(f"/project/{p_id}")
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("star", color="amber")
                    ui.label(proj.name).classes("text-subtitle1 ellipsis")


# =====================================================================
# Example chip — quick ingest from sample plots
# =====================================================================

def _example_chip(state: AppState, sample_file: Path, title: str) -> None:
    """Render a clickable chip that ingests a sample plot."""

    async def _ingest_sample():
        ui.notify(f"Ingesting {title}...", type="info")
        try:
            text = sample_file.read_text(encoding="utf-8")
            cfg = ExtractionConfig(
                chunk_strategy="act_headings",
                fabula_time_spacing=100,
                output_retries=5,
                max_correction_retries=1,
            )
            ws, report = await asyncio.to_thread(
                run_extraction, text, cfg,
            )
            proj = db.create_project(
                name=title,
                raw_text=text,
                owner_id=state.user_id,
                is_public=False,
            )
            db.save_version(
                project_id=proj.id,
                world_state_json=ws.model_dump_json(),
                source="ingestion",
                description="Initial ingestion from example",
                user_id=state.user_id,
            )
            ui.notify(f"Created {title}!", type="positive")
            ui.navigate.to(f"/project/{proj.id}")
        except Exception as e:
            logger.exception("Sample ingestion failed")
            ui.notify(f"Failed: {e}", type="negative")

    ui.chip(title, icon="menu_book", on_click=_ingest_sample).props("clickable outline")


# =====================================================================
# Ingest dialog — create new project from raw text
# =====================================================================

def _build_ingest_dialog(state: AppState) -> ui.dialog:
    """Dialog for ingesting raw narrative text into a new project."""

    dialog = ui.dialog().props("persistent maximized")

    with dialog, ui.card().classes("w-full max-w-3xl"):
        ui.label("Ingest Narrative Text").classes("text-h5")
        ui.label(
            "Paste or upload raw story text. The pipeline will extract "
            "entities, events, locations, objects, and all topology edges."
        ).classes("text-body2 text-grey")

        project_name = ui.input("Project Name", value="New Story").classes("w-full")
        project_desc = ui.input("Description (optional)").classes("w-full")
        text_area = ui.textarea(
            "Story Text",
            placeholder="Paste the full narrative text here...",
        ).classes("w-full").props("rows=15")

        # File upload
        ui.label("Or upload a .txt file:").classes("text-caption q-mt-sm")

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
                ui.label("Or load a sample plot:").classes("text-caption q-mt-sm")
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

        status = ui.label("").classes("text-body2 q-mt-sm")
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
                "color=primary"
            )

    return dialog
