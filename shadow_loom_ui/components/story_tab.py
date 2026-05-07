# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Story tab — prose reader, generation results, and source text.

Reading pane shows all generated prose from DB + session history.
Collapsible source text with manual-edit + re-ingest.
Chat/query input lives in the bottom command bar (not here).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui import db
from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.task_helpers import capture_logs_to_task, notify_task_complete
from shadow_loom_ui.components._safe_md import safe_markdown

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_story_tab(state: AppState) -> None:
    """Build the Story tab — prose reading + generation display."""

    with ui.column().classes("w-full h-full p-6 gap-3 bg-slate-50"):
        # ── Header with help popover ──────────────────────────────
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("auto_stories", color="primary")
            ui.label("Story").classes(
                "text-sm font-semibold text-slate-700"
            )
            ui.space()
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Story — prose reader & manual editor",
                body_md=(
                    "The reading pane for this version's narrative.\n\n"
                    "### Sections\n"
                    "**Source text** (collapsible)\n"
                    "- The original prose the project was ingested"
                    " from.\n"
                    "- Owners and editors can edit it inline. **Save &"
                    " Re-ingest** runs the full extraction pipeline"
                    " (entities, events, causal/social/spatial graphs)"
                    " and saves the result as a new version branching"
                    " from the current one.\n"
                    "- Read-only viewers see the textarea but cannot"
                    " save.\n\n"
                    "**Generated prose**\n"
                    "- Every prose card produced by Continue /"
                    " Intervene / What-If / Direct / Write prose along"
                    " the **lineage from the root to the active"
                    " version**. Switching branches in the version tree"
                    " changes which cards appear here.\n"
                    "- Each card shows the **query type** badge plus an"
                    " audit-status badge (*converged* / *unconverged*)"
                    " indicating whether the auditor accepted the prose"
                    " within its iteration budget.\n\n"
                    "### What you can do\n"
                    "- **Read** the canonical narrative for any version"
                    " by clicking it in the version tree.\n"
                    "- **Re-ingest** edited source text to fork a new"
                    " canonical baseline.\n"
                    "- Use the command bar below to generate the next"
                    " scene; the new card appears here automatically.\n\n"
                    "### What does *not* appear here\n"
                    "- **Ask** and **Interrogation** answers appear in"
                    " the Answer panel at the bottom of this tab,"
                    " below the source text and any generated prose,"
                    " so Q&A doesn't perturb the prose feed.\n"
                    "- **Implausible** runs that the engine refused —"
                    " their explanation surfaces as a chat message and"
                    " no version is saved."
                ),
                tooltip="What is this tab?",
            )

        # ── Source text (collapsible, editable) ─────────────
        source_container = ui.column().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm"
        )
        _render_source_text(state, source_container)


        # ── Generated Prose ───────────────────────────────────────
        prose_container = ui.column().classes("w-full gap-3")
        _render_prose(state, prose_container)
        # ── Answer panel (Ask / Interrogation results) ──────────
        # Sits below the source text and any generated prose so
        # read-only Q&A answers are visible alongside the narrative
        # they refer to without disturbing the prose feed.
        from shadow_loom_ui.components.answer_panel import (
            build_answer_panel,
        )
        build_answer_panel(state)
        # ── Subscribe ─────────────────────────────────────────────
        def _on_pipeline_result(**kwargs):
            # Ask / Interrogation queries are read-only Q&A: they don't
            # advance the world model and don't produce prose. The
            # Answer panel handles their results; re-rendering the
            # Story tab on every Q&A run was causing the prose feed to
            # flicker and scroll back to the top, which felt to users
            # like the original story was being mutated by their
            # questions. Skip the redraw when there is nothing new to
            # show here.
            result = kwargs.get("result")
            pr = getattr(result, "pipeline_result", None) if result else None
            if pr is None:
                return
            if not pr.prose and pr.query_type in ("general", "interrogate"):
                return
            # Evaluate queries produce an audit report, not story
            # prose — they belong in the Audit tab, not the Story
            # reader feed. Skip the redraw so the prose pane doesn't
            # flicker / scroll on every full-story evaluation.
            if pr.query_type == "evaluate":
                return
            _render_prose(state, prose_container)

        state.on(StateEvent.PIPELINE_RESULT, _on_pipeline_result)
        state.on(
            StateEvent.PROJECT_LOADED,
            lambda **kw: (
                _render_source_text(state, source_container),
                _render_prose(state, prose_container),
            ),
        )
        # Branch switch / delete / promote all funnel through
        # ``load_db_version``, which fires WORLD_STATE_CHANGED +
        # VERSION_CHANGED in lockstep. Without these subscriptions the
        # prose reader keeps rendering the previous branch's prose
        # cards and the source-text expansion keeps the previous
        # branch's textarea contents, even though every other panel
        # has already swapped over. Re-render on either event so the
        # Story tab tracks the active branch.
        state.on(
            StateEvent.VERSION_CHANGED,
            lambda **kw: (
                _render_source_text(state, source_container),
                _render_prose(state, prose_container),
            ),
        )
        state.on(
            StateEvent.WORLD_STATE_CHANGED,
            lambda **kw: (
                _render_source_text(state, source_container),
                _render_prose(state, prose_container),
            ),
        )


def _build_prompt_starters(state: AppState) -> None:  # pragma: no cover - removed
    """Deprecated. Prompt starter chips were removed from the Story tab."""
    return


def _trigger_prompt(state: AppState, prompt_text: str, query_type: str) -> None:  # pragma: no cover - removed
    """Deprecated."""
    state.emit(StateEvent.QUERY_STARTED, suggestion=prompt_text, query_type=query_type)


def _render_source_text(state: AppState, container) -> None:
    """Editable source-text panel: bulk replace + trigger full re-ingestion."""
    container.clear()
    if not state.raw_text:
        with container:
            with ui.expansion("Source Text", icon="description").props("dense").classes("w-full"):
                ui.label(
                    "No source text on this project — manual edits are unavailable."
                ).classes("text-xs text-slate-500 p-3")
        return

    can_edit = (
        state.project_id is not None
        and state.user_id is not None
        and _user_can_edit(state)
    )

    with container:
        with ui.expansion("Source Text", icon="description").classes("w-full"):
            with ui.column().classes("w-full p-3 gap-2"):
                ui.label(
                    f"{len(state.raw_text):,} characters"
                ).classes("text-xs text-slate-500")
                text_area = ui.textarea(value=state.raw_text).classes(
                    "w-full font-mono text-sm"
                ).props(
                    "outlined input-style='min-height: 60vh; max-height: 80vh; overflow:auto;'"
                )
                if not can_edit:
                    text_area.props("readonly")

                with ui.row().classes("w-full justify-end gap-2"):
                    if can_edit:
                        ui.button(
                            "Reset",
                            icon="undo",
                            on_click=lambda: text_area.set_value(state.raw_text),
                        ).props("flat color=secondary no-caps")
                        ui.button(
                            "Save & Re-ingest",
                            icon="auto_fix_high",
                            on_click=lambda: _confirm_reingest(
                                state, text_area.value,
                            ),
                        ).props("unelevated color=primary no-caps")


def _user_can_edit(state: AppState) -> bool:
    """True if the current user owns or has editor access."""
    if state.project_id is None or state.user_id is None:
        return False
    proj = db.get_project(state.project_id)
    if proj is None:
        return False
    if proj.owner_id == state.user_id:
        return True
    role = db.get_user_project_role(state.project_id, state.user_id)
    return role in ("editor", "admin")


def _confirm_reingest(state: AppState, edited_text: str) -> None:
    """Show a confirmation dialog before destructively re-ingesting."""
    edited_text = (edited_text or "").strip()
    if not edited_text:
        ui.notify("Edited text is empty", type="warning")
        return
    if edited_text == (state.raw_text or "").strip():
        ui.notify("No changes detected", type="info")
        return

    with ui.dialog() as dialog, ui.card():
        ui.label("Replace source text & re-ingest?").classes(
            "text-base font-semibold"
        )
        ui.label(
            "This runs the full extraction pipeline on the edited text and "
            "saves the result as a new version branching from the current one."
        ).classes("text-xs text-slate-500")
        ui.label(
            "The project's source text will be replaced. Existing versions "
            "are preserved."
        ).classes("text-xs text-slate-500")
        with ui.row().classes("justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Re-ingest",
                on_click=lambda: (dialog.close(), _start_reingest(state, edited_text)),
            ).props("unelevated color=primary")
    dialog.open()


def _start_reingest(state: AppState, edited_text: str) -> None:
    """Kick off background re-ingestion task."""
    from shadow_loom.ingestion import ExtractionConfig, run_extraction_async

    project_id = state.project_id
    user_id = state.user_id
    if project_id is None:
        return

    parent_version_row_id = state.current_version_row_id
    pname = state.project_name or f"Project {project_id}"

    task = state.start_task(
        label=f"Re-ingest: {pname}",
        kind="ingestion",
    )

    extraction_config = ExtractionConfig(
        chunk_strategy="act_headings",
        fabula_time_spacing=100,
        output_retries=5,
        max_correction_retries=1,
    )

    async def _do_work():
        try:
            with capture_logs_to_task(state, task):
                # Use the async pipeline so chunk extraction runs in
                # parallel (gated by ``max_concurrent_chunks``); see the
                # ingest-dialog path for the longer rationale.
                ws, report = await run_extraction_async(
                    edited_text, extraction_config,
                )

            # Bail out cleanly if the user navigated to a different
            # project *or version* while the heavyweight extraction was
            # running — we must not overwrite their new context with stale
            # data. We compare against the version pinned at task-start
            # (parent_version_row_id) since the user may have explicitly
            # branched away.
            if state.project_id != project_id:
                logger.info(
                    "[Re-ingest] Project switched mid-run (%s → %s); "
                    "discarding result.",
                    project_id, state.project_id,
                )
                state.finish_task(
                    task,
                    result_summary="cancelled (project switched)",
                )
                notify_task_complete(task)
                return
            if (
                parent_version_row_id is not None
                and state.current_version_row_id != parent_version_row_id
            ):
                logger.info(
                    "[Re-ingest] Version switched mid-run (%s → %s); "
                    "discarding result.",
                    parent_version_row_id, state.current_version_row_id,
                )
                state.finish_task(
                    task,
                    result_summary="cancelled (version switched)",
                )
                notify_task_complete(task)
                return

            db.update_project(project_id, raw_text=edited_text)

            new_ver = db.save_version(
                project_id=project_id,
                world_state_json=ws.model_dump_json(),
                ancestor_id=parent_version_row_id,
                source="manual_reingest",
                description="Manual edit \u2014 full re-ingestion",
                user_id=user_id,
            )

            state.raw_text = edited_text
            # Atomic version swap: resets cursors + emits the matching
            # WORLD_STATE_CHANGED / VERSION_CHANGED so every panel
            # picks up the freshly re-ingested world without scrubbing
            # the previous version's cursor onto it.
            state.load_db_version(ws, new_ver.id, version_number=new_ver.version)
            # Mirror into the active-version pointer so MCP defaults
            # to the freshly re-ingested version.
            if user_id is not None:
                try:
                    db.set_active_version(project_id, user_id, new_ver.id)
                except Exception:
                    logger.exception(
                        "Failed to update active-version pointer after re-ingest"
                    )

            summary = (
                f"v{new_ver.version}: {len(ws.entities)} entities, "
                f"{len(ws.events)} events, {len(ws.locations)} locations · "
                f"validation {'PASS' if report.is_valid else 'FAIL'}"
            )
            state.finish_task(task, result_summary=summary)
            notify_task_complete(task)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Re-ingestion failed")
            state.finish_task(task, error=str(exc))
            notify_task_complete(task)

    state.spawn_task(_do_work(), name=f"reingest:{task.id}")


def _render_prose(state: AppState, container) -> None:
    """Render all prose from DB history + current session."""
    container.clear()

    prose_entries: list[tuple[str, str, bool | None, int]] = []

    # Load from DB if project is active. When a specific version is
    # active, restrict the prose feed to that version's lineage
    # (root → current) so switching branches actually shows different
    # prose; otherwise the reader pools every version's prose across
    # every fork of the project, which made branch deletes / selects
    # look like no-ops in the Story panel.
    if state.project_id:
        try:
            from shadow_loom_ui.db import get_all_prose, get_version_by_id, get_version_lineage
            branch_path: list[int] | None = None
            if state.current_version_row_id is not None:
                cur_row = get_version_by_id(state.current_version_row_id)
                if cur_row is not None:
                    lineage = get_version_lineage(
                        state.project_id, cur_row.version,
                    )
                    branch_path = [entry["id"] for entry in lineage]
            db_prose = get_all_prose(state.project_id, branch_path=branch_path)
            for entry in db_prose:
                # Defensive filter for legacy data: evaluation queries
                # used to be persisted as VersionRows with
                # ``source="evaluate"`` and the report in ``prose``.
                # That prose belongs in the Audit tab — never the
                # Story reader.
                if entry.get("source") == "evaluate":
                    continue
                # Skip if it'll be duplicated from session history
                prose_entries.append((
                    entry.get("source", "pipeline"),
                    entry["prose"],
                    None,  # DB doesn't store convergence directly
                    0,
                ))
        except Exception:
            logger.exception("Failed to load prose from DB")

    # Overlay session results (may duplicate some DB entries, but ensures freshness)
    session_prose_set = set()
    for result in state.query_history:
        pr = result.pipeline_result
        if pr and pr.prose:
            # Evaluation queries produce an audit *report* — not story
            # prose — and surfacing them in the reader's prose feed
            # makes the report look like newly-written narrative. The
            # full evaluation output (scorecard + textual summary)
            # lives in the Audit tab; skip it here.
            if pr.query_type == "evaluate":
                continue
            # De-duplicate against DB entries by content hash
            key = pr.prose[:200]
            if key not in {p[1][:200] for p in prose_entries}:
                prose_entries.append((pr.query_type, pr.prose, pr.converged, pr.audit_iterations))
            session_prose_set.add(key)

    if not prose_entries:
        return

    with container:
        for i, (qtype, prose, converged, iters) in enumerate(prose_entries):
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl "
                "shadow-sm p-6"
            ):
                # Header with badges
                with ui.row().classes("items-center gap-2 mb-3"):
                    ui.badge(qtype, color="primary").props("dense")
                    if converged is not None:
                        color = "positive" if converged else "warning"
                        label = "converged" if converged else f"unconverged ({iters} iters)"
                        ui.badge(label, color=color).props("dense outline")

                # Prose content
                safe_markdown(prose)
