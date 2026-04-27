"""Helpers for running long operations as tracked ``BackgroundTask`` instances.

Bridges Python ``logging`` records from the pipeline/ingestion modules
into live progress messages on the task, and provides a uniform
"sticky completion notification" so users always know when work is done.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from nicegui import ui

from shadow_loom_ui.state import AppState, BackgroundTask


class _TaskLogBridge(logging.Handler):
    """Forwards selected log records into a ``BackgroundTask.message``."""

    def __init__(self, state: AppState, task: BackgroundTask, prefixes: tuple[str, ...]):
        super().__init__(level=logging.INFO)
        self._state = state
        self._task = task
        self._prefixes = prefixes

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:
            return
        if self._prefixes and not any(p in msg for p in self._prefixes):
            return
        # Trim very long messages
        self._state.update_task(self._task, msg[:160])


@contextmanager
def capture_logs_to_task(
    state: AppState,
    task: BackgroundTask,
    *,
    logger_names: tuple[str, ...] = ("shadow_loom",),
    prefixes: tuple[str, ...] = ("[Step", "[Pipeline", "[Audit", "[Generation", "[Ingest"),
) -> Iterator[None]:
    """Attach a log handler that pushes step-level messages into the task."""
    handler = _TaskLogBridge(state, task, prefixes=prefixes)
    attached: list[logging.Logger] = []
    try:
        for name in logger_names:
            lg = logging.getLogger(name)
            lg.addHandler(handler)
            attached.append(lg)
        yield
    finally:
        for lg in attached:
            try:
                lg.removeHandler(handler)
            except Exception:
                pass


def notify_task_complete(
    task: BackgroundTask,
    *,
    on_open: Optional[callable] = None,
    open_label: str = "Open",
) -> None:
    """Show a sticky notification summarising a finished task."""
    if task.status == "complete":
        msg = f"✓ {task.label} ({task.elapsed:.0f}s)"
        if task.result_summary:
            msg += f" — {task.result_summary}"
        actions = []
        if on_open is not None:
            actions.append({"label": open_label, "color": "white", "handler": on_open})
        try:
            ui.notify(msg, type="positive", timeout=8000, actions=actions or None)
        except Exception:
            ui.notify(msg, type="positive", timeout=8000)
    else:
        msg = f"✗ {task.label} failed"
        if task.error:
            msg += f" — {task.error}"
        ui.notify(msg, type="negative", timeout=0, close_button="Dismiss")
