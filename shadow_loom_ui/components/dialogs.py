"""Dialog components: ingestion, project management, and add-node dialogs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom.ingestion import ExtractionConfig, run_extraction
from shadow_loom_ui import db
from shadow_loom_ui.state import StateEvent
from shadow_loom_ui.task_helpers import capture_logs_to_task, notify_task_complete
from shadow_loom_ui.theme import feather

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState

logger = logging.getLogger(__name__)


def build_ingest_dialog(state: AppState) -> ui.dialog:
    """Create a dialog for ingesting raw narrative text into a world model."""

    dialog = ui.dialog().props("persistent maximized")

    with dialog, ui.card().classes(
        "w-full max-w-3xl bg-white border border-slate-200 "
        "rounded-xl shadow-sm p-6"
    ):
        ui.label("Ingest Narrative Text").classes(
            "text-2xl font-bold text-slate-800"
        )
        ui.label(
            "Paste or upload raw story text. The pipeline will extract "
            "entities, events, locations, objects, and all topology edges."
        ).classes("text-sm text-slate-500 mb-2")

        project_name = ui.input("Project Name", value="New Story").classes("w-full")
        text_area = ui.textarea(
            "Story Text",
            placeholder="Paste the full narrative text here...",
        ).classes("w-full").props("rows=15")

        # File upload option
        ui.label("Or upload a .txt file:").classes("text-xs text-slate-500 mt-2")

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
                # Build an explicit allow-list keyed by the str(path) we hand
                # to the client; we never trust the value we get back without
                # checking it against this map (defends against arbitrary
                # file reads via crafted select payloads).
                sample_allowlist = {str(f): f for f in sample_files}
                ui.label("Or load a sample plot:").classes("text-xs text-slate-500 mt-2")
                sample_select = ui.select(
                    options={k: v.stem.replace("_", " ").title()
                             for k, v in sample_allowlist.items()},
                    label="Sample",
                ).classes("w-64")

                def _load_sample():
                    chosen = sample_select.value
                    safe_path = sample_allowlist.get(chosen)
                    if safe_path is None:
                        ui.notify("Invalid sample selection", type="warning")
                        return
                    text_area.value = safe_path.read_text(encoding="utf-8")
                    project_name.value = safe_path.stem.replace("_", " ").title()

                sample_select.on("update:model-value", _load_sample)

        status = ui.label("").classes("text-sm text-slate-600 mt-2")
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        progress.set_visibility(False)

        # Track the currently running ingestion task (for progress updates + cancel-safe close).
        current = {"task": None, "asyncio_task": None}

        # Subscribe to TASKS_CHANGED so dialog progress mirrors the registry
        # while the dialog is open. We unbind on close.
        def _on_tasks_changed(**kwargs):
            t = current["task"]
            if t is None:
                return
            if kwargs.get("task") is not None and kwargs["task"].id != t.id:
                return
            if t.status == "running":
                status.set_text(t.message or "Working…")

        state.on(StateEvent.TASKS_CHANGED, _on_tasks_changed)

        button_row = ui.row().classes("w-full justify-end gap-2 mt-4")
        with button_row:
            cancel_btn = ui.button("Cancel", on_click=lambda: dialog.close()).props(
                "flat color=secondary no-caps"
            )
            background_btn = ui.button(
                "Run in background",
                icon="visibility_off",
                on_click=lambda: dialog.close(),
            ).props("flat color=primary no-caps")
            background_btn.set_visibility(False)
            ingest_btn = ui.button(
                "Ingest", icon="auto_fix_high",
            ).props("unelevated color=primary no-caps").classes("rounded-lg shadow-sm")

        async def _run_ingestion():
            text = text_area.value.strip()
            if not text:
                ui.notify("Enter some text first", type="warning")
                return

            pname = (project_name.value or "Untitled").strip()
            task = state.start_task(
                label=f"Ingest: {pname}",
                kind="ingestion",
            )
            current["task"] = task

            status.set_text("Starting ingestion…")
            progress.set_visibility(True)
            progress.props("indeterminate")
            ingest_btn.props("loading")
            background_btn.set_visibility(True)
            cancel_btn.props("disable")

            extraction_config = ExtractionConfig(
                chunk_strategy="act_headings",
                fabula_time_spacing=100,
                output_retries=5,
                max_correction_retries=1,
            )

            async def _do_work():
                try:
                    with capture_logs_to_task(state, task):
                        ws, report = await asyncio.to_thread(
                            run_extraction, text, extraction_config,
                        )

                    proj = db.create_project(
                        name=pname,
                        raw_text=text,
                        owner_id=state.user_id,
                    )
                    db.save_version(
                        project_id=proj.id,
                        world_state_json=ws.model_dump_json(),
                        version=0,
                        source="ingestion",
                        description="Initial ingestion",
                        user_id=state.user_id,
                    )
                    state.load_project(
                        project_id=proj.id,
                        project_name=proj.name,
                        world_state=ws,
                        raw_text=text,
                    )

                    summary = (
                        f"{len(ws.entities)} entities, {len(ws.events)} events, "
                        f"{len(ws.locations)} locations · "
                        f"validation {'PASS' if report.is_valid else 'FAIL'}"
                    )
                    state.finish_task(task, result_summary=summary)

                    # Notification works even if the dialog has been closed.
                    notify_task_complete(
                        task,
                        on_open=lambda _: ui.navigate.to(f"/project/{proj.id}"),
                        open_label="Open project",
                    )

                    if dialog.value:  # still open
                        status.set_text(f"Done! {summary}")
                        progress.props(remove="indeterminate")
                        progress.value = 1.0
                        await asyncio.sleep(0.5)
                        dialog.close()
                except Exception as e:
                    logger.exception("Ingestion failed")
                    state.finish_task(task, error=str(e))
                    notify_task_complete(task)
                    if dialog.value:
                        status.set_text(f"Error: {e}")
                finally:
                    if dialog.value:
                        progress.set_visibility(False)
                        ingest_btn.props(remove="loading")
                        background_btn.set_visibility(False)
                        cancel_btn.props(remove="disable")
                    current["task"] = None
                    current["asyncio_task"] = None

            current["asyncio_task"] = asyncio.create_task(_do_work())

        ingest_btn.on("click", _run_ingestion)

        # Detach the listener when the dialog disappears so we don't leak callbacks.
        def _on_dialog_hide():
            state.off(StateEvent.TASKS_CHANGED, _on_tasks_changed)

        dialog.on("hide", _on_dialog_hide)

    return dialog


def build_project_dialog(state: AppState) -> ui.dialog:
    """Create a dialog for loading existing projects."""

    dialog = ui.dialog()

    with dialog, ui.card().classes(
        "w-full max-w-2xl bg-white border border-slate-200 "
        "rounded-xl shadow-sm p-6"
    ):
        ui.label("Load Project").classes("text-2xl font-bold text-slate-800")

        project_list = ui.column().classes("w-full")

        def _refresh_projects():
            project_list.clear()
            # Only list projects this user is allowed to see. When
            # ``state.user_id`` is None we fall back to public/example
            # projects only via the helper below \u2014 avoid exposing
            # private projects to anonymous sessions.
            projects = db.list_projects(user_id=state.user_id)
            if state.user_id is None:
                projects = [p for p in projects if p.get("is_public") or p.get("is_example")]
            if not projects:
                with project_list:
                    ui.label("No projects yet").classes(
                        "text-sm text-slate-400 italic"
                    )
                return

            with project_list:
                for p in projects:
                    with ui.row().classes(
                        "w-full items-center px-3 py-2 "
                        "border-b border-slate-100 last:border-b-0"
                    ):
                        with ui.row().classes("items-center gap-2 flex-grow"):
                            ui.label(p["name"]).classes(
                                "text-sm font-medium text-slate-800"
                            )
                            if p.get("label"):
                                ui.badge(p["label"], color="secondary")
                            if p.get("is_example"):
                                ui.badge("example", color="primary")
                        ui.label(f"{p.get('version_count', 0)} versions").classes(
                            "text-xs text-slate-400"
                        )
                        ui.label(p["updated_at"]).classes(
                            "text-xs text-slate-400"
                        )

                        async def _load(pid=p["id"], pname=p["name"]):
                            # Re-check access at load time \u2014 defends
                            # against stale UI state and tampered ids.
                            proj_row = db.get_project(pid)
                            if proj_row is None:
                                ui.notify("Project not found", type="warning")
                                return
                            allowed = (
                                proj_row.is_public
                                or (state.user_id is not None and proj_row.owner_id == state.user_id)
                                or (
                                    state.user_id is not None
                                    and db.get_user_project_role(pid, state.user_id) is not None
                                )
                            )
                            if not allowed:
                                ui.notify("Access denied", type="negative")
                                return
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

                        ui.button("Load", on_click=_load).props(
                            "flat dense color=primary no-caps"
                        )

        _refresh_projects()

        with ui.row().classes("w-full justify-end mt-4"):
            ui.button("Close", on_click=dialog.close).props(
                "flat color=secondary no-caps"
            )

    return dialog


def build_version_dialog(state: AppState) -> ui.dialog:
    """Dialog for viewing and rolling back world model versions.

    Shows both in-memory version history and DB version tree.
    """

    dialog = ui.dialog()

    with dialog, ui.card().classes("w-full max-w-2xl"):
        ui.label("Version History").classes(
            "text-xl font-semibold text-slate-800"
        )

        version_list = ui.column().classes("w-full")

        def _refresh():
            version_list.clear()

            # Try DB version tree first
            db_tree = []
            if state.project_id is not None:
                db_tree = db.get_version_tree(state.project_id)

            if db_tree:
                with version_list:
                    ui.label("Version Tree (DB)").classes(
                        "text-sm font-semibold text-slate-700 mb-1"
                    )
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
                            ui.label(ancestor_label).classes(
                                "text-xs text-slate-500"
                            )
                            desc = entry.get("description") or ""
                            ui.label(desc[:50]).classes("text-xs text-slate-600 flex-grow")
                            if entry.get("changeset_summary"):
                                cs = entry["changeset_summary"]
                                ui.label(
                                    f"+{cs.get('events_added', 0)}evt "
                                    f"+{cs.get('causal_edges_added', 0)}ce"
                                ).classes("text-xs text-secondary")

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
                    ui.label("In-Memory History").classes(
                        "text-sm font-semibold text-slate-700 mb-1"
                    )
                    for entry in reversed(state.versioned_model.history):
                        is_current = entry.version == state.versioned_model.version
                        with ui.row().classes("w-full items-center q-pa-xs"):
                            if is_current:
                                ui.icon("arrow_right", color="primary")
                            ui.label(f"v{entry.version}").classes(
                                "text-body1" + (" text-primary" if is_current else "")
                            )
                            ui.label(entry.source).classes(
                                "text-xs text-slate-500"
                            )
                            ui.label(entry.description[:60]).classes("text-xs text-slate-600 flex-grow")
                            if entry.changeset:
                                cs = entry.changeset
                                ui.label(
                                    f"+{cs.events_added}evt +{cs.causal_edges_added}ce"
                                ).classes("text-xs text-secondary")
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
                    ui.label("No version history available").classes(
                        "text-sm text-slate-400 italic"
                    )

        _refresh()

        with ui.row().classes("w-full justify-end q-mt-md"):
            ui.button("Close", on_click=dialog.close).props("flat")

    return dialog
