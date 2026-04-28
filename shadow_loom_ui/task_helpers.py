"""Helpers for running long operations as tracked ``BackgroundTask`` instances.

Bridges Python ``logging`` records from the pipeline/ingestion modules
into live progress messages on the task, and provides a uniform
"sticky completion notification" so users always know when work is done.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator, Optional

from nicegui import ui

from shadow_loom_ui.state import AppState, BackgroundTask, StateEvent


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


# ── Unified async query runner ───────────────────────────────────

async def run_query_as_task(
    state: AppState,
    *,
    label: str,
    kind: str,
    runner: Callable[[], Awaitable[Any]],
    summary_fn: Optional[Callable[[Any], str]] = None,
    start_toast: bool = True,
) -> tuple[Any, BackgroundTask]:
    """Run a query as a tracked task with start + completion notifications.

    Centralises the boilerplate every query-launching button used to
    duplicate (start_task → emit QUERY_STARTED → capture_logs_to_task
    → run → finish_task → notify_task_complete). Every entry point
    that triggers a pipeline run should go through this so the user
    sees the same toast/indicator behaviour everywhere.

    Parameters
    ----------
    state
        The application state.
    label
        Short human label shown in the task indicator and toasts.
    kind
        Task kind tag (e.g. ``"query"``, ``"manual_edit"``, ``"evaluate"``,
        ``"directive"``, ``"intervention"``).
    runner
        Zero-arg async callable that performs the actual work and
        returns its result. Wrap sync calls with
        ``lambda: asyncio.to_thread(state.run_xxx, ...)``.
    summary_fn
        Optional callable that turns the runner's return value into a
        short summary string for the task / completion toast.
    start_toast
        When ``True`` (the default) shows a transient "Started: <label>"
        toast immediately so the user gets feedback even before the
        first log message arrives.

    Returns
    -------
    (result, task)
        The runner's return value and the finished ``BackgroundTask``.
        Exceptions raised by ``runner`` are caught, surfaced via
        ``notify_task_complete`` as a sticky failure toast, and then
        re-raised so the caller can update its own UI on failure.
    """
    task = state.start_task(label=label, kind=kind)
    state.emit(StateEvent.QUERY_STARTED)

    if start_toast:
        try:
            ui.notify(
                f"Started: {label}",
                type="ongoing",
                timeout=2500,
                position="top-right",
                spinner=True,
            )
        except Exception:
            # NiceGUI/Quasar may reject some notification kwargs in older
            # versions — fall back to a plain info toast.
            try:
                ui.notify(f"Started: {label}", type="info", timeout=2500)
            except Exception:
                pass

    try:
        with capture_logs_to_task(state, task):
            result = await runner()
    except Exception as exc:
        state.finish_task(task, error=str(exc))
        notify_task_complete(task)
        raise

    summary = ""
    if summary_fn is not None:
        try:
            summary = summary_fn(result) or ""
        except Exception:
            logging.getLogger(__name__).exception(
                "summary_fn raised; using empty summary",
            )
    # Detect a result-level error (e.g. NLQueryResult.error) without
    # forcing every caller to write a custom summary_fn.
    err = getattr(result, "error", None)
    if err:
        state.finish_task(task, error=str(err))
    else:
        state.finish_task(task, result_summary=summary[:160])
    notify_task_complete(task)
    return result, task
