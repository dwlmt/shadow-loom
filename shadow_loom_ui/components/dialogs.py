"""Dialog components: ingestion, project management, and add-node dialogs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom.ingestion import ExtractionConfig, run_extraction
from shadow_loom_ui import db

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState

logger = logging.getLogger(__name__)


def build_ingest_dialog(state: AppState) -> ui.dialog:
    """Create a dialog for ingesting raw narrative text into a world model."""

    dialog = ui.dialog().props("persistent maximized")

    with dialog, ui.card().classes("w-full max-w-3xl"):
        ui.label("Ingest Narrative Text").classes("text-h5")
        ui.label(
            "Paste or upload raw story text. The pipeline will extract "
            "entities, events, locations, objects, and all topology edges."
        ).classes("text-body2 text-grey")

        project_name = ui.input("Project Name", value="New Story").classes("w-full")
        text_area = ui.textarea(
            "Story Text",
            placeholder="Paste the full narrative text here...",
        ).classes("w-full").props("rows=15")

        # File upload option
        ui.label("Or upload a .txt file:").classes("text-caption q-mt-sm")

        async def _handle_upload(e):
            content = e.content.read().decode("utf-8")
            text_area.value = content
            ui.notify(f"Loaded {len(content)} characters")

        ui.upload(on_upload=_handle_upload, auto_upload=True).props(
            "accept=.txt flat dense"
        ).classes("w-full")

        # Sample plots
        sample_dir = Path(__file__).resolve().parent.parent.parent / "sample_plots"
        if sample_dir.exists():
            sample_files = sorted(sample_dir.glob("*.txt"))
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
                    config = ExtractionConfig(
                        chunk_strategy="act_headings",
                        fabula_time_spacing=100,
                        output_retries=5,
                        max_correction_retries=1,
                    )

                    ws, report = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: run_extraction(text, config),
                    )
                    progress.value = 0.8

                    # Save to DB
                    proj = db.create_project(
                        name=project_name.value or "Untitled",
                        raw_text=text,
                    )
                    db.save_snapshot(
                        project_id=proj.id,
                        version=0,
                        world_state_json=ws.model_dump_json(),
                        description="Initial ingestion",
                    )

                    # Load into state
                    state.project_id = proj.id
                    state.project_name = proj.name
                    state.raw_text = text
                    state.load_world_state(ws)

                    progress.value = 1.0
                    status.set_text(
                        f"Done! {len(ws.entities)} entities, {len(ws.events)} events, "
                        f"{len(ws.locations)} locations. "
                        f"Validation: {'PASS' if report.is_valid else 'FAIL'}"
                    )

                    ui.notify("World model created!", type="positive")
                    await asyncio.sleep(1)
                    dialog.close()

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


def build_project_dialog(state: AppState) -> ui.dialog:
    """Create a dialog for loading existing projects."""

    dialog = ui.dialog()

    with dialog, ui.card().classes("w-full max-w-2xl"):
        ui.label("Load Project").classes("text-h5")

        project_list = ui.column().classes("w-full")

        def _refresh_projects():
            project_list.clear()
            projects = db.list_projects()
            if not projects:
                with project_list:
                    ui.label("No projects yet").classes("text-grey")
                return

            with project_list:
                for p in projects:
                    with ui.row().classes("w-full items-center q-pa-sm"):
                        with ui.row().classes("items-center gap-2 flex-grow"):
                            ui.label(p["name"]).classes("text-body1")
                            if p.get("label"):
                                ui.badge(p["label"], color="blue-grey")
                            if p.get("is_example"):
                                ui.badge("example", color="teal")
                        ui.label(f"{p.get('version_count', 0)} versions").classes("text-caption text-grey")
                        ui.label(p["updated_at"]).classes("text-caption text-grey")

                        async def _load(pid=p["id"], pname=p["name"]):
                            snap = db.load_latest_snapshot(pid)
                            if snap is None:
                                ui.notify("No versions found", type="warning")
                                return
                            ws = state.from_json(snap.world_state_json)
                            state.project_id = pid
                            state.project_name = pname
                            state.load_world_state(ws)
                            state.current_version_row_id = snap.id
                            ui.notify(f"Loaded {pname} (v{snap.version})")
                            dialog.close()

                        ui.button("Load", on_click=_load).props("flat dense color=primary")

        _refresh_projects()

        with ui.row().classes("w-full justify-end q-mt-md"):
            ui.button("Close", on_click=dialog.close).props("flat")

    return dialog


def build_version_dialog(state: AppState) -> ui.dialog:
    """Dialog for viewing and rolling back world model versions.

    Shows both in-memory version history and DB version tree.
    """

    dialog = ui.dialog()

    with dialog, ui.card().classes("w-full max-w-2xl"):
        ui.label("Version History").classes("text-h5")

        version_list = ui.column().classes("w-full")

        def _refresh():
            version_list.clear()

            # Try DB version tree first
            db_tree = []
            if state.project_id is not None:
                db_tree = db.get_version_tree(state.project_id)

            if db_tree:
                with version_list:
                    ui.label("Version Tree (DB)").classes("text-subtitle2 q-mb-xs")
                    for entry in reversed(db_tree):
                        with ui.row().classes("w-full items-center q-pa-xs"):
                            # Indent branches
                            ancestor_label = (
                                f"← v-row {entry['ancestor_id']}"
                                if entry["ancestor_id"]
                                else "root"
                            )
                            ui.label(f"v{entry['version']}").classes("text-body1")
                            ui.badge(entry["source"], color={
                                "ingestion": "green", "pipeline": "blue",
                                "manual_edit": "purple", "rollback": "orange",
                            }.get(entry["source"], "grey")).classes("q-mx-xs")
                            ui.label(ancestor_label).classes("text-caption text-grey")
                            desc = entry.get("description") or ""
                            ui.label(desc[:50]).classes("text-caption flex-grow")
                            if entry.get("changeset_summary"):
                                cs = entry["changeset_summary"]
                                ui.label(
                                    f"+{cs.get('events_added', 0)}evt "
                                    f"+{cs.get('causal_edges_added', 0)}ce"
                                ).classes("text-caption text-blue")

                            def _load_version(v=entry["version"], pid=state.project_id):
                                ver = db.get_version(pid, v)
                                if ver is None:
                                    ui.notify("Version not found", type="warning")
                                    return
                                ws = state.from_json(ver.world_state_json)
                                state.load_world_state(ws)
                                state.current_version_row_id = ver.id
                                ui.notify(f"Loaded v{v}")

                            ui.button("Load", on_click=_load_version).props(
                                "flat dense color=primary"
                            )

            # Also show in-memory history if available
            if state.versioned_model is not None:
                with version_list:
                    if db_tree:
                        ui.separator().classes("q-my-sm")
                    ui.label("In-Memory History").classes("text-subtitle2 q-mb-xs")
                    for entry in reversed(state.versioned_model.history):
                        is_current = entry.version == state.versioned_model.version
                        with ui.row().classes("w-full items-center q-pa-xs"):
                            if is_current:
                                ui.icon("arrow_right", color="primary")
                            ui.label(f"v{entry.version}").classes(
                                "text-body1" + (" text-primary" if is_current else "")
                            )
                            ui.label(entry.source).classes("text-caption text-grey")
                            ui.label(entry.description[:60]).classes("text-caption flex-grow")
                            if entry.changeset:
                                cs = entry.changeset
                                ui.label(
                                    f"+{cs.events_added}evt +{cs.causal_edges_added}ce"
                                ).classes("text-caption text-blue")
                            if not is_current and entry.version in {
                                s.version for s in state.versioned_model.snapshots
                            }:
                                def _rollback(v=entry.version):
                                    state.rollback_to(v)
                                    ui.notify(f"Rolled back to v{v}")
                                    _refresh()

                                ui.button("Rollback", on_click=_rollback).props(
                                    "flat dense color=orange"
                                )

            if not db_tree and state.versioned_model is None:
                with version_list:
                    ui.label("No version history available").classes("text-grey")

        _refresh()

        with ui.row().classes("w-full justify-end q-mt-md"):
            ui.button("Close", on_click=dialog.close).props("flat")

    return dialog
