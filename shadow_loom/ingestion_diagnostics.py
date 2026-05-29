"""In-memory collector for ingestion validator/auto-fix log records.

Captures any log record from ``shadow_loom.ingestion`` whose message starts
with a ``[Bracket-Prefix]`` token (e.g. ``[Auto-Fix]``, ``[Validator·Social]``,
``[Scaffold-Drift]``) and exposes the records to UI / audit consumers.

Backend for Tier 4 #14 in /memories/repo/ingestion-improvements-plan-2026-05-06.md.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

_BRACKET_RE = re.compile(r"^\[([^\]]+)\]")
_LOGGER_NAME = "shadow_loom.ingestion"

# C2 (twelfth-pass audit): use a ContextVar so concurrent ingestions
# in different threads / asyncio tasks each see their own active
# project_id. The previous global LIFO stack cross-attributed
# diagnostics when two ingestions overlapped (the most recently
# pushed project_id would absorb all log records from every active
# scope).
_current_project: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "shadow_loom_ingestion_diagnostics_current_project",
    default=None,
)


class IngestionWarning(BaseModel):
    """Single captured diagnostic line."""

    category: str = Field(..., description="Bracket prefix, e.g. 'Auto-Fix'.")
    level: str = Field(..., description="Log level name (INFO/WARNING/ERROR).")
    message: str = Field(..., description="Fully formatted log message.")


@dataclass
class _ProjectBuffer:
    records: List[IngestionWarning] = field(default_factory=list)


class _DiagnosticsHandler(logging.Handler):
    """Routes bracket-tagged ingestion records to project-scoped buffers."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._lock = threading.Lock()
        self._buffers: Dict[str, _ProjectBuffer] = {}

    # ----- handler API -----
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:
            return
        m = _BRACKET_RE.match(msg)
        if not m:
            return
        category = m.group(1)
        # Route by the current context's project_id rather than a
        # shared LIFO stack, so concurrent scopes don't bleed into
        # each other.
        project_id = _current_project.get()
        if project_id is None:
            return
        with self._lock:
            buf = self._buffers.setdefault(project_id, _ProjectBuffer())
            buf.records.append(
                IngestionWarning(
                    category=category,
                    level=record.levelname,
                    message=msg,
                )
            )

    # ----- scoping API -----
    def reset_buffer(self, project_id: str) -> None:
        with self._lock:
            self._buffers[project_id] = _ProjectBuffer()

    def records_for(self, project_id: str) -> List[IngestionWarning]:
        with self._lock:
            buf = self._buffers.get(project_id)
            return list(buf.records) if buf else []

    def clear(self, project_id: str) -> None:
        with self._lock:
            self._buffers.pop(project_id, None)


_HANDLER: Optional[_DiagnosticsHandler] = None
_HANDLER_LOCK = threading.Lock()


def _get_handler() -> _DiagnosticsHandler:
    global _HANDLER
    with _HANDLER_LOCK:
        if _HANDLER is None:
            _HANDLER = _DiagnosticsHandler()
            logging.getLogger(_LOGGER_NAME).addHandler(_HANDLER)
        return _HANDLER


class capture_ingestion_warnings:
    """Context manager that scopes captured log records to a project id.

    Usage::

        with capture_ingestion_warnings(project_id="42"):
            extract_topology_async(...)
        warnings = get_ingestion_warnings("42")
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = str(project_id)
        self._handler = _get_handler()
        self._token: Optional[contextvars.Token] = None
        self._prev_level: Optional[int] = None

    def __enter__(self) -> "capture_ingestion_warnings":
        # Reset buffer for a fresh ingest run, then bind this scope's
        # project_id into the current context. ``__exit__`` releases
        # the binding via the stored token so nested scopes restore
        # cleanly without relying on a global stack.
        self._handler.reset_buffer(self.project_id)
        self._token = _current_project.set(self.project_id)
        # Ensure INFO-level diagnostics actually reach our handler even
        # when the application's root logger is at WARNING.
        ing_logger = logging.getLogger(_LOGGER_NAME)
        self._prev_level = ing_logger.level
        if ing_logger.level == logging.NOTSET or ing_logger.level > logging.INFO:
            ing_logger.setLevel(logging.INFO)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _current_project.reset(self._token)
            self._token = None
        if self._prev_level is not None:
            logging.getLogger(_LOGGER_NAME).setLevel(self._prev_level)


def get_ingestion_warnings(project_id: str) -> List[IngestionWarning]:
    """Return captured warnings for ``project_id`` (empty if none)."""
    return _get_handler().records_for(str(project_id))


def clear_ingestion_warnings(project_id: str) -> None:
    _get_handler().clear(str(project_id))
