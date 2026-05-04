# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Header tasks indicator — badge button showing in-flight + recent background tasks.

Subscribes to ``StateEvent.TASKS_CHANGED`` and re-renders a popover
listing each ``BackgroundTask`` with its status, elapsed time, and
progress message. Provides a one-click way for the user to see what
the app is doing across all dialogs and tabs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, BackgroundTask, StateEvent
from shadow_loom_ui.theme import feather

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_STATUS_ICON = {
    "running": ("clock", "primary"),
    "complete": ("check-circle", "positive"),
    "failed": ("x-circle", "negative"),
}


def build_tasks_indicator(state: AppState) -> None:
    """Render a header button + popover showing active and recent tasks."""

    btn = ui.button().props("flat dense round color=secondary")
    with btn:
        feather("clipboard")
    badge = ui.badge("0", color="primary").props("floating")
    badge.move(target_container=btn)

    with btn:
        with ui.menu().props("auto-close=false") as menu:
            with ui.card().classes(
                "w-96 bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ).style("max-height: 60vh; overflow-y: auto;"):
                with ui.row().classes("w-full items-center justify-between mb-2"):
                    ui.label("Background tasks").classes(
                        "text-base font-semibold text-slate-800"
                    )
                    with ui.button(
                        on_click=lambda: state.clear_completed_tasks(),
                    ).props("flat dense size=sm color=secondary no-caps"):
                        with ui.row().classes("items-center gap-1"):
                            feather("trash", size="sm")
                            ui.label("Clear done").classes("text-xs")
                task_list = ui.column().classes("w-full gap-1")

    def _render(**_kwargs) -> None:
        # Bail out if the owning client has been deleted (e.g. tab closed)
        # before the disconnect handler had a chance to unsubscribe us.
        try:
            _ = task_list.client
        except RuntimeError:
            state.off(StateEvent.TASKS_CHANGED, _render)
            return
        # Update badge: prefer running count, else show total recent if any
        running = state.running_task_count
        total = len(state.background_tasks)
        if running > 0:
            badge.text = str(running)
            badge.props(remove="color=grey")
            badge.props("color=primary")
            badge.set_visibility(True)
        elif total > 0:
            badge.text = str(total)
            badge.props(remove="color=primary")
            badge.props("color=grey")
            badge.set_visibility(True)
        else:
            badge.set_visibility(False)

        task_list.clear()
        with task_list:
            if not state.background_tasks:
                ui.label("No background tasks").classes(
                    "text-xs text-slate-400 italic q-pa-sm"
                )
                return
            # Newest first
            for task in reversed(state.background_tasks):
                _render_task(task)

    def _render_task(task: BackgroundTask) -> None:
        icon, color = _STATUS_ICON.get(task.status, ("help-circle", "grey"))
        with ui.row().classes(
            "w-full items-start q-pa-xs no-wrap border-b border-slate-200"
        ):
            with ui.element('div').classes("q-mt-xs"):
                feather(icon)
            with ui.column().classes("flex-grow gap-0"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.label(task.label).classes(
                        "text-sm text-slate-700 ellipsis"
                    ).style("max-width: 220px;")
                    ui.label(f"{task.elapsed:.0f}s").classes(
                        "text-xs text-slate-400"
                    )
                if task.status == "running":
                    if task.message:
                        ui.label(task.message).classes(
                            "text-xs text-primary ellipsis"
                        )
                    ui.linear_progress(show_value=False).props(
                        "indeterminate color=primary"
                    ).classes("w-full")
                elif task.status == "complete":
                    if task.result_summary:
                        ui.label(task.result_summary).classes(
                            "text-xs text-slate-500 ellipsis"
                        )
                else:  # failed
                    ui.label(task.error or "failed").classes(
                        "text-xs text-negative ellipsis"
                    )

    state.on(StateEvent.TASKS_CHANGED, _render)
    _render()

    # Detach the listener when this client disconnects so we don't try to
    # mutate elements whose owning client has been deleted.
    try:
        client = ui.context.client
    except Exception:
        client = None
    if client is not None:
        client.on_disconnect(
            lambda: state.off(StateEvent.TASKS_CHANGED, _render)
        )
