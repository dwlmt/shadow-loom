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

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_STATUS_ICON = {
    "running": ("hourglass_top", "primary"),
    "complete": ("check_circle", "positive"),
    "failed": ("error", "negative"),
}


def build_tasks_indicator(state: AppState) -> None:
    """Render a header button + popover showing active and recent tasks."""

    btn = ui.button(icon="pending_actions").props("flat dense round")
    badge = ui.badge("0", color="red").props("floating")
    badge.move(target_container=btn)

    with btn:
        with ui.menu().props("auto-close=false") as menu:
            with ui.card().classes("w-96").style("max-height: 60vh; overflow-y: auto;"):
                with ui.row().classes("w-full items-center justify-between q-mb-sm"):
                    ui.label("Background tasks").classes("text-subtitle1")
                    ui.button(
                        "Clear done",
                        icon="clear_all",
                        on_click=lambda: state.clear_completed_tasks(),
                    ).props("flat dense size=sm")
                task_list = ui.column().classes("w-full gap-1")

    def _render(**_kwargs) -> None:
        # Update badge: prefer running count, else show total recent if any
        running = state.running_task_count
        total = len(state.background_tasks)
        if running > 0:
            badge.text = str(running)
            badge.props(remove="color=grey")
            badge.props("color=red")
            badge.set_visibility(True)
            btn.props("color=primary")
        elif total > 0:
            badge.text = str(total)
            badge.props(remove="color=red")
            badge.props("color=grey")
            badge.set_visibility(True)
            btn.props(remove="color=primary")
        else:
            badge.set_visibility(False)
            btn.props(remove="color=primary")

        task_list.clear()
        with task_list:
            if not state.background_tasks:
                ui.label("No background tasks").classes("text-caption text-grey")
                return
            # Newest first
            for task in reversed(state.background_tasks):
                _render_task(task)

    def _render_task(task: BackgroundTask) -> None:
        icon, color = _STATUS_ICON.get(task.status, ("help", "grey"))
        with ui.row().classes("w-full items-start q-pa-xs no-wrap").style(
            "border-bottom: 1px solid #333;"
        ):
            ui.icon(icon, color=color).classes("q-mt-xs")
            with ui.column().classes("flex-grow gap-0"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.label(task.label).classes("text-body2 ellipsis").style(
                        "max-width: 220px;"
                    )
                    ui.label(f"{task.elapsed:.0f}s").classes("text-caption text-grey")
                if task.status == "running":
                    if task.message:
                        ui.label(task.message).classes(
                            "text-caption text-primary ellipsis"
                        )
                    ui.linear_progress(show_value=False).props(
                        "indeterminate"
                    ).classes("w-full")
                elif task.status == "complete":
                    if task.result_summary:
                        ui.label(task.result_summary).classes(
                            "text-caption text-grey ellipsis"
                        )
                else:  # failed
                    ui.label(task.error or "failed").classes(
                        "text-caption text-negative ellipsis"
                    )

    state.on(StateEvent.TASKS_CHANGED, _render)
    _render()
