# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared application state for the Shadow-Loom UI.

Holds the current project, world model, and pipeline runner.
Provides the natural-language query interface that ties
query_parsing → pipeline → UI updates together.

Uses an event-bus pattern so multiple components can subscribe
to state changes without overwriting each other's callbacks.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.models import WorldStateV1
from shadow_loom.pipeline import (
    PipelineConfig,
    PipelineResult,
    finish_reextraction,
    humanize_pipeline_result,
    run_pipeline,
)
import threading
from shadow_loom.query_models import ManualEditQuery, UserRequest
from shadow_loom.query_parsing import (
    QueryParseResult,
    QueryParsingConfig,
    parse_query,
    parse_query_async,
)
from shadow_loom.db import (
    save_version,
    get_version_by_id,
    get_version_lineage_rows,
    log_activity,
    set_active_version,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Event bus
# =====================================================================

class StateEvent(Enum):
    """Events emitted by AppState when data changes."""
    WORLD_STATE_CHANGED = "world_state_changed"
    PIPELINE_RESULT = "pipeline_result"
    PROJECT_LOADED = "project_loaded"
    VERSION_CHANGED = "version_changed"
    NODE_SELECTED = "node_selected"
    QUERY_STARTED = "query_started"
    QUERY_COMPLETE = "query_complete"
    TASKS_CHANGED = "tasks_changed"
    FABULA_CURSOR_CHANGED = "fabula_cursor_changed"
    SYUZHET_CURSOR_CHANGED = "syuzhet_cursor_changed"
    TIME_AXIS_CHANGED = "time_axis_changed"
    ACTIVE_PATH_CHANGED = "active_path_changed"
    WORLD_FACTS_CHANGED = "world_facts_changed"
    PROJECT_LIST_CHANGED = "project_list_changed"
    AUTOSAVE_FAILED = "autosave_failed"


# Debounce window (seconds) for cursor emit fan-out. Coalesces bursty
# slider drags / keyboard repeats so subscribers don't see every tick.
_CURSOR_DEBOUNCE_S = 0.12

# Bounded query history (round-3 audit): a single session that runs
# many queries could accumulate unlimited PipelineResult objects and
# leak memory. 500 is enough for the longest interactive sessions; the
# oldest entries fall off when the cap is hit.
import os as _os
try:
    _QUERY_HISTORY_MAX = max(1, int(_os.environ.get("SHADOW_LOOM_QUERY_HISTORY_MAX", "500")))
except ValueError:
    _QUERY_HISTORY_MAX = 500


@dataclass
class BackgroundTask:
    """A long-running operation tracked for UI progress + notifications."""
    id: str
    label: str
    kind: str  # "ingestion" | "query" | "manual_edit" | "save" | ...
    status: str = "running"  # "running" | "complete" | "failed"
    message: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result_summary: str = ""
    error: Optional[str] = None
    # Optional metadata (e.g., project_id to navigate to on completion)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)


@dataclass
class NLQueryResult:
    """Result of a natural-language query through the full pipeline."""
    # Parsing stage
    parse_result: QueryParseResult
    # Pipeline stage (None if parsing produced no valid query)
    pipeline_result: Optional[PipelineResult] = None
    # Human-readable summary
    summary: str = ""
    error: Optional[str] = None


@dataclass
class AppState:
    """Mutable per-session state shared across all UI components.

    Each browser session gets its own AppState instance.
    Components register for events via ``state.on(event, callback)``
    and all listeners are notified via ``state.emit(event, **kwargs)``.
    """

    # Authenticated user
    user_id: Optional[int] = None
    username: str = ""
    display_name: str = ""
    avatar_url: str = ""

    # Current project
    project_id: Optional[int] = None
    project_name: str = ""

    # World model
    world_state: Optional[WorldStateV1] = None
    versioned_model: Optional[VersionedWorldModel] = None

    # DB version row ID of the currently loaded version (for correct ancestor tracking)
    current_version_row_id: Optional[int] = None

    # Raw source text (if ingested)
    raw_text: Optional[str] = None

    # Pipeline configuration
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)

    # Query parsing configuration
    query_parsing_config: QueryParsingConfig = field(default_factory=QueryParsingConfig)

    # History of NL queries + results in this session
    query_history: List[NLQueryResult] = field(default_factory=list)

    # Last pipeline result (for UI display)
    last_result: Optional[PipelineResult] = None
    last_parse: Optional[QueryParseResult] = None

    # Currently selected node (for cross-component inspector sync)
    selected_node_id: Optional[str] = None
    selected_node_type: Optional[str] = None

    # Fabula timeline cursor (None = "live"/latest); affects time-aware charts
    fabula_cursor: Optional[int] = None

    # Syuzhet (reading-order) cursor (None = "live"); affects suspense/reveal views
    syuzhet_cursor: Optional[int] = None

    # Global time axis: "fabula" (chronological) or "syuzhet" (telling
    # order). Charts that bin on event time honour this so the user
    # can see Genette's order/anachrony with one switch.
    time_axis: str = "fabula"

    # NOTE: AMWN branch identity ("factual" vs "shadow") is sourced
    # from :meth:`head_branch` (driven by the loaded VWM head row),
    # NOT from a free-standing flag on AppState. A previous
    # ``world_id`` field plus ``set_world_id`` setter and
    # ``WORLD_ID_CHANGED`` event were unused by any subscriber and
    # would not have driven ``projected_for_branch`` (which keys off
    # the VWM head); they were removed in the T-9 audit pass to
    # prevent future contributors from wiring a branch toggle to a
    # no-op signal.

    # Background task registry (in-flight + recently completed)
    background_tasks: List[BackgroundTask] = field(default_factory=list)
    _max_completed_tasks: int = 20

    # Live asyncio task handles for in-flight background coroutines.
    # Tracking these prevents "Task was destroyed but it is pending"
    # warnings (asyncio holds only weak refs to created tasks) and lets
    # us cancel everything cleanly on session shutdown.
    _async_tasks: "set[asyncio.Task]" = field(default_factory=set)

    # Round-11 R11-02: deferred re-extraction work runs on raw
    # ``threading.Thread`` instances (not the asyncio loop) so we keep
    # a separate registry for them. ``teardown`` joins these
    # best-effort during session eviction so a closed-tab session
    # does not leak a worker that keeps mutating dead state.
    _deferred_threads: "list[threading.Thread]" = field(default_factory=list)

    # Per-panel task slots: each panel id maps to its in-flight refresh
    # task. Spawning a new task for a panel cancels the previous one,
    # so rapid cursor scrubs collapse to the latest render only.
    _panel_tasks: "Dict[str, asyncio.Task]" = field(default_factory=dict)

    # Debounce timers for cursor emits (one per axis).
    _cursor_emit_tasks: "Dict[str, asyncio.Task]" = field(default_factory=dict)

    # Active tab path, e.g. ``"causality.affective"``. Panels gate
    # their refreshes on this so off-screen tabs don't burn CPU on
    # every cursor / world-state change.
    active_path: str = ""

    # Event bus: multiple listeners per event
    _listeners: Dict[StateEvent, List[Callable]] = field(default_factory=dict)

    # Round-6 audit: serialize pipeline mutations of the versioned
    # model so concurrent NL queries (e.g. user fires a second query
    # before the first completes, or the deferred re-extraction worker
    # races a fresh foreground query) can't tear ``versioned_model`` /
    # ``world_state`` against each other. Use a reentrant lock — the
    # same thread may enter via ``run_structured_query`` and then via
    # ``_save_version_to_db`` without self-deadlocking.
    _query_lock: threading.RLock = field(default_factory=threading.RLock)

    # ---- Branch-aware accessors ----------------------------------
    # ``world_state`` is the AMWN-PROJECTED view (shadow clones
    # layered into ``entities`` etc. via :meth:`projected_for_branch`)
    # — correct for UI reads but unsafe to serialize, because dumping
    # it conflates the factual baseline with the shadow clones and
    # the factual record for any cloned id is lost on round-trip
    # (the clone overwrites the factual entry in ``entities`` while
    # ``shadow_entities`` keeps the same clone, so a reload mistakes
    # the clone for the canonical factual entity).
    #
    # ``raw_world_state`` returns the un-projected snapshot directly
    # from the VWM history (which :meth:`load_db_version` /
    # :meth:`load_project` keep raw). All save / export / JSON-editor
    # paths MUST route through this so the persisted snapshot keeps
    # the factual baseline and the sidecar intact. ``head_branch``
    # surfaces the loaded row's ``(world_id, branch_label)`` so
    # those paths can also pass branch identity to ``save_version``
    # and avoid the silent demote-to-factual lineage corruption that
    # the MCP equivalents were already fixed against.

    @property
    def raw_world_state(self) -> Optional[WorldStateV1]:
        """Un-projected world state for the loaded row — use for
        every save / export / JSON-editor path. UI reads should keep
        using :attr:`world_state` (the projected view).
        """
        if self.versioned_model is not None and self.versioned_model.history:
            return self.versioned_model.current
        return self.world_state

    def head_branch(self) -> tuple[str, Optional[str]]:
        """Return ``(world_id, branch_label)`` of the loaded VWM head.

        Defaults to ``("factual", None)`` when no VWM is loaded so
        callers can always splat the result into ``save_version``
        without conditional branching.
        """
        if self.versioned_model is not None and self.versioned_model.history:
            head = self.versioned_model.history[-1]
            return head.world_id, head.branch_label
        return "factual", None

    # ---- Event bus ----

    def on(self, event: StateEvent, callback: Callable) -> None:
        """Register a listener for a state event."""
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: StateEvent, callback: Callable) -> None:
        """Unregister a listener."""
        listeners = self._listeners.get(event, [])
        if callback in listeners:
            listeners.remove(callback)

    def emit(self, event: StateEvent, **kwargs) -> None:
        """Notify all listeners of an event.

        Listeners that raise ``RuntimeError`` (the typical signal that
        their owning NiceGUI client has been deleted) are auto-removed
        so we don't keep trying to mutate dead UI elements on every
        subsequent emission. Other exceptions are logged but the
        listener is kept registered — only dead-client failures are
        treated as terminal.

        For :data:`StateEvent.WORLD_STATE_CHANGED` the snapshot and
        physics caches in :mod:`shadow_loom_ui.viz_helpers` are
        invalidated *before* listeners run, so every panel sees fresh
        data without each one having to remember to call
        ``invalidate_snapshot_cache``.
        """
        if event == StateEvent.WORLD_STATE_CHANGED:
            try:
                from shadow_loom_ui.viz_helpers import (
                    invalidate_physics_trajectory_cache,
                    invalidate_snapshot_cache,
                )
                invalidate_snapshot_cache()
                invalidate_physics_trajectory_cache()
            except Exception:
                logger.debug(
                    "[AppState] cache invalidation failed", exc_info=True,
                )
        listeners = self._listeners.get(event, [])
        dead: list[Callable] = []
        for cb in list(listeners):
            try:
                cb(**kwargs)
            except RuntimeError as exc:
                # NiceGUI raises RuntimeError("client has been deleted")
                # when the originating tab was closed before the
                # listener was unregistered.
                logger.debug(
                    "[AppState] Auto-detaching dead listener for %s: %s",
                    event.value, exc,
                )
                dead.append(cb)
            except Exception:
                logger.exception(
                    "[AppState] Listener error for %s", event.value,
                )
        for cb in dead:
            self.off(event, cb)

    # ---- Background task registry ----

    def start_task(self, label: str, kind: str = "generic", **meta) -> BackgroundTask:
        """Register a new in-flight task and notify listeners."""
        task = BackgroundTask(
            id=uuid.uuid4().hex[:8],
            label=label,
            kind=kind,
            meta=meta,
        )
        self.background_tasks.append(task)
        self.emit(StateEvent.TASKS_CHANGED, task=task, change="started")
        return task

    def update_task(self, task: BackgroundTask, message: str) -> None:
        """Update a task's progress message."""
        if task.status != "running":
            return
        task.message = message
        self.emit(StateEvent.TASKS_CHANGED, task=task, change="updated")

    def finish_task(
        self,
        task: BackgroundTask,
        result_summary: str = "",
        error: Optional[str] = None,
    ) -> None:
        """Mark a task as complete or failed and trim the history."""
        task.status = "failed" if error else "complete"
        task.finished_at = time.time()
        task.result_summary = result_summary
        task.error = error
        # Trim completed tasks beyond the cap (keep all running ones)
        completed = [t for t in self.background_tasks if t.status != "running"]
        if len(completed) > self._max_completed_tasks:
            keep = set(t.id for t in completed[-self._max_completed_tasks:])
            self.background_tasks = [
                t for t in self.background_tasks
                if t.status == "running" or t.id in keep
            ]
        self.emit(StateEvent.TASKS_CHANGED, task=task, change="finished")

    def clear_completed_tasks(self) -> None:
        """Remove all non-running tasks from the registry."""
        self.background_tasks = [
            t for t in self.background_tasks if t.status == "running"
        ]
        self.emit(StateEvent.TASKS_CHANGED, change="cleared")

    @property
    def running_task_count(self) -> int:
        return sum(1 for t in self.background_tasks if t.status == "running")

    # ---- asyncio task lifecycle ----

    def spawn_task(self, coro, *, name: str | None = None) -> "asyncio.Task":
        """Schedule ``coro`` and retain a strong reference for cancellation.

        Use this in place of :func:`asyncio.create_task` for any work
        whose lifetime is tied to the UI session, so the task survives
        garbage collection and can be cancelled on session shutdown.

        The coroutine is wrapped so that any ``RuntimeError`` raised by
        a NiceGUI dead-slot / dead-client mutation (the user navigated
        away mid-task and the task tried to update an element on the
        original page) is logged at debug rather than propagating as
        an unhandled task exception. Any other exception is logged at
        error level so genuine bugs still surface in the console.
        """
        async def _guarded():
            try:
                return await coro
            except asyncio.CancelledError:
                raise
            except RuntimeError as exc:
                msg = str(exc)
                if (
                    "has been deleted" in msg
                    or "client has been deleted" in msg
                    or "no current slot" in msg.lower()
                ):
                    logger.debug(
                        "[AppState] background task %s touched dead "
                        "slot/client after page nav: %s", name, exc,
                    )
                    return None
                logger.exception(
                    "[AppState] background task %s raised RuntimeError", name,
                )
                return None
            except Exception:
                logger.exception(
                    "[AppState] background task %s raised", name,
                )
                return None

        task = asyncio.create_task(_guarded(), name=name)
        self._async_tasks.add(task)
        task.add_done_callback(self._async_tasks.discard)
        return task

    async def cancel_all_async_tasks(self, timeout: float = 2.0) -> None:
        """Cancel every tracked asyncio task and await their teardown.

        Best-effort: any task that does not honour cancellation within
        ``timeout`` seconds is logged and left to the event loop.
        Intended for session/app shutdown hooks.
        """
        tasks = [t for t in self._async_tasks if not t.done()]
        for t in tasks:
            t.cancel()
        if not tasks:
            return
        try:
            await asyncio.wait(tasks, timeout=timeout)
        except Exception:
            logger.exception("[AppState] Error awaiting cancelled tasks")
        still_alive = [t for t in tasks if not t.done()]
        if still_alive:
            logger.warning(
                "[AppState] %d background task(s) did not finish within %.1fs",
                len(still_alive), timeout,
            )

    def teardown(self, *, thread_timeout: float = 2.0) -> None:
        """Synchronously release session-owned background work.

        Round-11 R11-03: called from the session-eviction path in
        :mod:`shadow_loom_ui.app`, which runs in a sync NiceGUI handler
        and cannot await :meth:`cancel_all_async_tasks`. We do the
        thread-safe parts here:

        * ``task.cancel()`` is safe to call from any thread — the
          owning event loop will pick up the cancellation on its next
          iteration. We do *not* try to await the tasks (no loop in
          scope) and accept that a few may run a bit longer; the
          weak-ref drop on AppState ensures their I/O targets fade
          out as the session does.
        * Deferred ``threading.Thread`` workers (R11-02) get a
          best-effort ``join(timeout=...)`` so they finish their
          current write rather than racing GC of the popped state.
        """
        # Cancel any in-flight asyncio tasks (thread-safe).
        for t in list(self._async_tasks):
            if not t.done():
                try:
                    t.cancel()
                except Exception:
                    logger.debug(
                        "[AppState] teardown: cancel failed", exc_info=True,
                    )
        # Join deferred re-extraction threads best-effort.
        per_thread = max(0.05, thread_timeout / max(1, len(self._deferred_threads)))
        for h in list(self._deferred_threads):
            if h.is_alive():
                try:
                    h.join(timeout=per_thread)
                except Exception:
                    logger.debug(
                        "[AppState] teardown: thread join failed",
                        exc_info=True,
                    )
        self._deferred_threads = [
            h for h in self._deferred_threads if h.is_alive()
        ]

    # ---- Session setup ----

    def set_user(self, user_id: int, username: str, display_name: str = "", avatar_url: str = "") -> None:
        """Set the authenticated user for this session."""
        self.user_id = user_id
        self.username = username
        self.display_name = display_name or username
        self.avatar_url = avatar_url

    # ---- Natural language query interface ----

    def run_nl_query(
        self,
        natural_language: str,
        query_type: str = "general",
        force_implausible: bool = False,
    ) -> NLQueryResult:
        """Parse a natural-language query and run it through the pipeline."""
        if self.world_state is None:
            return NLQueryResult(
                parse_result=QueryParseResult(
                    query=None,
                    parsed=None,
                    is_valid=False,
                ),
                error="No world model loaded. Ingest a story first.",
                summary="Error: No world model loaded.",
            )

        logger.info("[AppState] Parsing NL query: %s", natural_language[:120])
        try:
            parse_result = parse_query(
                natural_language,
                query_type=query_type,
                world_state=self.world_state,
                config=self.query_parsing_config,
            )
        except Exception as e:
            logger.exception("[AppState] Query parsing failed")
            return NLQueryResult(
                parse_result=QueryParseResult(
                    query=None, parsed=None, is_valid=False,
                ),
                error=f"Query parsing failed: {e}",
                summary=f"Parse error: {e}",
            )

        self.last_parse = parse_result

        if not parse_result.is_valid or parse_result.query is None:
            errors = "; ".join(e.message for e in parse_result.validation_errors)
            result = NLQueryResult(
                parse_result=parse_result,
                error=f"Query validation failed: {errors}",
                summary=f"Validation failed: {errors}",
            )
            self._record_query_history(result)
            return result

        return self.run_structured_query(
            parse_result.query,
            parse_result=parse_result,
            force_implausible=force_implausible,
        )

    def _record_query_history(self, result: NLQueryResult) -> None:
        """Append to ``query_history`` and trim to ``_QUERY_HISTORY_MAX``.

        Bounded ring-buffer behaviour prevents long-lived sessions from
        leaking ``PipelineResult`` objects indefinitely. See round-3
        audit; cap is tunable via ``SHADOW_LOOM_QUERY_HISTORY_MAX``.
        """
        self.query_history.append(result)
        overflow = len(self.query_history) - _QUERY_HISTORY_MAX
        if overflow > 0:
            del self.query_history[:overflow]

    def run_structured_query(
        self,
        query: UserRequest,
        parse_result: Optional[QueryParseResult] = None,
        force_implausible: bool = False,
    ) -> NLQueryResult:
        """Execute a pre-built structured query through the pipeline."""
        # Round-6 audit: serialize the whole pipeline run. Two queries
        # firing in parallel (foreground + deferred-reextraction worker,
        # or two rapid clicks) could otherwise read/write
        # ``versioned_model`` between each other's pipeline steps and
        # persist a half-merged version.
        with self._query_lock:
            return self._run_structured_query_locked(
                query, parse_result, force_implausible,
            )

    def _run_structured_query_locked(
        self,
        query: UserRequest,
        parse_result: Optional[QueryParseResult] = None,
        force_implausible: bool = False,
    ) -> NLQueryResult:
        if force_implausible and hasattr(query, "force_implausible"):
            query = query.model_copy(update={"force_implausible": True})
        if self.world_state is None:
            return NLQueryResult(
                parse_result=parse_result or QueryParseResult(
                    query=query, parsed=None, is_valid=True,
                ),
                error="No world model loaded.",
                summary="Error: No world model loaded.",
            )

        # T-12 (P1-1): thread the user-visible cursor anchors into the
        # query before the pipeline reconstructs the world. The cursor
        # is what every other UI surface (causality, reasoning, world,
        # explorer) is already pinned to; without this the query
        # silently runs against the *latest* world state regardless of
        # what the user sees. Explicit per-query anchors set upstream
        # (manual_edit insert points, "after EVT_X" anchors from the
        # parser, etc.) win over the global cursor so author-side
        # precision overrides the chrome.
        anchor_updates: Dict[str, Any] = {}
        if (
            getattr(query, "temporal_anchor", None) is None
            and self.fabula_cursor is not None
        ):
            anchor_updates["temporal_anchor"] = int(self.fabula_cursor)
        if (
            getattr(query, "syuzhet_anchor", None) is None
            and self.syuzhet_cursor is not None
        ):
            anchor_updates["syuzhet_anchor"] = int(self.syuzhet_cursor)
        if anchor_updates:
            try:
                query = query.model_copy(update=anchor_updates)
            except Exception:
                logger.debug(
                    "[AppState] Failed to apply cursor anchors to query",
                    exc_info=True,
                )

        # Snapshot execution context up-front so a project switch (or a
        # version load) that happens while the pipeline is running in a
        # worker thread cannot corrupt the persisted ancestor lineage.
        ctx_project_id = self.project_id
        ctx_user_id = self.user_id
        ctx_ancestor_row_id = self.current_version_row_id

        logger.info("[AppState] Running pipeline: query_type=%s", query.query_type)
        # Defer Steps 6–7 (prose re-extraction + merge) for write-class
        # query types so the user sees the rendered prose immediately
        # and the heavy ingest runs in a background thread. Read-only
        # Q&A queries (general/interrogate/evaluate) never advance the
        # world model and so don't need deferral.
        readonly_query_types = ("general", "interrogate", "evaluate")
        defer_this_run = (
            query.query_type not in readonly_query_types
            and not self.pipeline_config.skip_reextraction
        )
        run_cfg = self.pipeline_config
        if defer_this_run and not run_cfg.defer_reextraction:
            run_cfg = run_cfg.model_copy(update={"defer_reextraction": True})
        # Activate the signed-in user's per-user model overrides
        # (default model, per-stage models, custom OpenAI-compat
        # providers) so every LLM call in this thread picks them up.
        # No explicit reset is needed: this method runs under
        # ``asyncio.to_thread`` which gives a fresh ContextVar context
        # per invocation.
        from shadow_loom.settings import set_user_context as _set_user_context
        _set_user_context(ctx_user_id)
        try:
            pipeline_result = run_pipeline(
                query,
                versioned_model=self.versioned_model,
                config=run_cfg,
            )
        except Exception as e:
            logger.exception("[AppState] Pipeline execution failed")
            result = NLQueryResult(
                parse_result=parse_result or QueryParseResult(
                    query=query, parsed=None, is_valid=True,
                ),
                error=f"Pipeline failed: {e}",
                summary=f"Pipeline error: {e}",
            )
            self._record_query_history(result)
            return result

        # Capture the *previous* version before we overwrite ``self.versioned_model``
        # — needed below to detect engine short-circuits where the pipeline
        # produced an explanation but did not actually advance the world model.
        prev_version = (
            self.versioned_model.version if self.versioned_model is not None else None
        )

        # Detect a project switch *or* a version switch that happened
        # during pipeline execution so we don't paint stale results onto
        # the new context.
        project_switched = self.project_id != ctx_project_id
        version_switched = (
            self.current_version_row_id != ctx_ancestor_row_id
            and ctx_ancestor_row_id is not None
        )
        context_switched = project_switched or version_switched

        # Update world model if pipeline produced a new versioned model.
        # Suppress the WORLD_STATE_CHANGED emit when the version hasn't
        # actually advanced (e.g. read-only Ask/Interrogation queries
        # return the same versioned model unchanged) so panels don't
        # needlessly re-render and snapshot caches stay warm.
        if pipeline_result.world_model is not None and not context_switched:
            world_advanced = (
                prev_version is None
                or pipeline_result.world_model.version != prev_version
            )
            self.versioned_model = pipeline_result.world_model
            self.world_state = pipeline_result.world_model.current
            self._reproject_world_state_to_vwm_head()
            if world_advanced:
                self.emit(StateEvent.WORLD_STATE_CHANGED)

        self.last_result = pipeline_result

        # Persist version to DB \u2014 but skip when the engine short-circuited
        # an implausible query (no world-state advancement, just an explanation).
        short_circuited = bool(
            pipeline_result.implausible
            and pipeline_result.world_model is not None
            and prev_version is not None
            and pipeline_result.world_model.version == prev_version
            and not pipeline_result.feedback_result
        )
        # Read-only Q&A queries (Ask / Interrogation) never advance the
        # world model — pipeline returns early before generation. Saving
        # them as new versions would litter the version tree with
        # identical world-state snapshots and make branch lineage and
        # the prose-by-lineage filter in the Story panel meaningless.
        # The structured response card in chat is the only artifact a
        # Q&A run should produce.
        #
        # ``evaluate`` is also read-only: it scores the existing prose
        # corpus and returns a report. Persisting that report as a new
        # VersionRow caused the report text to leak into the Story
        # tab's prose feed via ``get_all_prose`` (which reads
        # ``VersionRow.prose`` regardless of source). The Audit tab is
        # the canonical surface for evaluation results.
        readonly_query = query.query_type in (
            "general", "interrogate", "evaluate",
        )
        # Also skip when re-extraction failed: prose is present but the world
        # model was *not* advanced to reflect that prose. Persisting would
        # store divergent prose/world state under the same version row.
        # Skip persistence entirely when context switched mid-flight — the
        # ancestor lineage we captured is no longer the user's current view.
        # When Steps 6–7 are still pending we cannot persist the version
        # yet — the world_model has not been advanced. The background
        # thread spawned below will persist + emit WORLD_STATE_CHANGED
        # once finish_reextraction completes.
        if (
            not short_circuited
            and not readonly_query
            and not pipeline_result.reextraction_failed
            and not pipeline_result.reextraction_pending
            and not context_switched
        ):
            self._save_version_to_db(
                pipeline_result=pipeline_result,
                # Prefer the user's verbatim NL request (now carried on
                # every query type as ``original_query``) so the UI and
                # activity log show what the human asked for, not the
                # parser's restated reasoning. Fall back through the
                # legacy edited-prose / parser-reasoning sources for
                # queries that were built without an NL string.
                raw_query=getattr(query, "original_query", None)
                or getattr(query, 'edited_prose', None)
                or (
                    parse_result.parsed.reasoning if parse_result and parse_result.parsed else None
                ),
                parsed_query_json=query.model_dump_json() if query else None,
                source=query.query_type,
                project_id=ctx_project_id,
                user_id=ctx_user_id,
                ancestor_row_id=ctx_ancestor_row_id,
            )

        # Build summary — lay-user phrasing including audit thresholds
        # and achieved-vs-target affective intensity when available.
        requested_effect = getattr(query, "target_effect", None)
        requested_intensity = getattr(query, "intensity", None)
        summary_text = humanize_pipeline_result(
            pipeline_result,
            requested_effect=requested_effect,
            requested_intensity=requested_intensity,
        )

        result = NLQueryResult(
            parse_result=parse_result or QueryParseResult(
                query=query, parsed=None, is_valid=True,
            ),
            pipeline_result=pipeline_result,
            summary=summary_text,
        )
        self._record_query_history(result)

        self.emit(StateEvent.PIPELINE_RESULT, result=result)
        self.emit(StateEvent.QUERY_COMPLETE, result=result)

        # ----- Background Steps 6–7 (re-extraction + merge) ---------
        # Prose was already returned to the caller. Run the deferred
        # closure on a daemon thread so the UI gets updated world state
        # later without blocking the original query response.
        if (
            pipeline_result.reextraction_pending
            and not context_switched
            and not readonly_query
        ):
            self._spawn_deferred_reextraction(
                pipeline_result=pipeline_result,
                query=query,
                parse_result=parse_result,
                project_id=ctx_project_id,
                user_id=ctx_user_id,
                ancestor_row_id=ctx_ancestor_row_id,
            )

        return result

    def _spawn_deferred_reextraction(
        self,
        *,
        pipeline_result: PipelineResult,
        query: UserRequest,
        parse_result: Optional[QueryParseResult],
        project_id: Optional[int],
        user_id: Optional[int],
        ancestor_row_id: Optional[int],
    ) -> None:
        """Run deferred Steps 6–7 in a daemon thread.

        On completion: applies the new versioned world model, persists
        to the DB if the project context still matches, and emits
        ``WORLD_STATE_CHANGED`` so panels refresh.
        """
        # Capture the UI event loop now (we are still on it). The worker
        # runs on a daemon thread, and ``emit`` fans out to listeners that
        # mutate NiceGUI elements — those mutations must be marshalled back
        # onto the loop or they are not reliably flushed to the browser
        # (and can raise). Degrade to a direct call only if no loop is
        # running (e.g. unit tests driving this synchronously).
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def _emit_on_loop(event: "StateEvent", **kwargs) -> None:
            if loop is not None:
                loop.call_soon_threadsafe(lambda: self.emit(event, **kwargs))
            else:
                self.emit(event, **kwargs)

        def _worker() -> None:
            # Re-activate the user's model overrides inside this daemon
            # thread (threading.Thread does not propagate ContextVars).
            try:
                from shadow_loom.settings import (
                    set_user_context as _set_user_context,
                )
                _set_user_context(user_id)
            except Exception:  # noqa: BLE001 — never block re-extraction
                logger.debug(
                    "[AppState] Failed to activate user model overrides "
                    "for deferred worker", exc_info=True,
                )
            try:
                finish_reextraction(pipeline_result)
            except Exception:
                logger.exception(
                    "[AppState] Deferred re-extraction failed"
                )
                return
            # Bail if the user navigated away mid-flight.
            project_switched = self.project_id != project_id
            ctx_ancestor = ancestor_row_id
            version_switched = (
                ctx_ancestor is not None
                and self.current_version_row_id != ctx_ancestor
            )
            if project_switched or version_switched:
                logger.info(
                    "[AppState] Deferred re-extraction completed but "
                    "context switched (project=%s→%s, ancestor=%s→%s); "
                    "discarding background result.",
                    project_id, self.project_id,
                    ctx_ancestor, self.current_version_row_id,
                )
                return
            if (
                pipeline_result.world_model is not None
                and not pipeline_result.reextraction_failed
            ):
                # Round-6 audit: hold the query lock while applying the
                # deferred merge so a foreground query firing at the
                # same instant can't interleave its versioned_model
                # mutation with ours.
                with self._query_lock:
                    self.versioned_model = pipeline_result.world_model
                    self.world_state = pipeline_result.world_model.current
                    self._reproject_world_state_to_vwm_head()
                _emit_on_loop(StateEvent.WORLD_STATE_CHANGED)
                # Persist the now-advanced version, mirroring the
                # synchronous-flow gating (skip readonly / failed /
                # short-circuit cases).
                if (
                    query.query_type not in (
                        "general", "interrogate", "evaluate",
                    )
                ):
                    self._save_version_to_db(
                        pipeline_result=pipeline_result,
                        raw_query=getattr(query, "original_query", None)
                        or getattr(query, "edited_prose", None)
                        or (
                            parse_result.parsed.reasoning
                            if parse_result and parse_result.parsed
                            else None
                        ),
                        parsed_query_json=(
                            query.model_dump_json() if query else None
                        ),
                        source=query.query_type,
                        project_id=project_id,
                        user_id=user_id,
                        ancestor_row_id=ancestor_row_id,
                    )
            # Re-emit PIPELINE_RESULT so any tab listening (e.g. audit /
            # causality) refreshes against the now-merged world.
            # Mirror the synchronous path's payload shape: listeners
            # expect an NLQueryResult, not the raw PipelineResult that
            # ``self.last_result`` carries.
            requested_effect = getattr(query, "target_effect", None)
            requested_intensity = getattr(query, "intensity", None)
            summary_text = humanize_pipeline_result(
                pipeline_result,
                requested_effect=requested_effect,
                requested_intensity=requested_intensity,
            )
            deferred_result = NLQueryResult(
                parse_result=parse_result or QueryParseResult(
                    query=query, parsed=None, is_valid=True,
                ),
                pipeline_result=pipeline_result,
                summary=summary_text,
            )
            _emit_on_loop(StateEvent.PIPELINE_RESULT, result=deferred_result)

        # Round-11 R11-02: register the deferred worker on the state
        # so ``teardown`` can join it during session eviction. Without
        # tracking, evicting a session whose deferred re-extraction is
        # still in flight leaves a thread mutating now-orphaned state
        # (racing GC of the popped AppState). Trim already-finished
        # handles first so the list stays bounded for long-lived
        # sessions that fire many deferred re-extractions.
        self._deferred_threads = [
            h for h in self._deferred_threads if h.is_alive()
        ]
        t = threading.Thread(
            target=_worker,
            name="deferred-reextraction",
            daemon=True,
        )
        self._deferred_threads.append(t)
        t.start()

    async def run_nl_query_async(
        self,
        natural_language: str,
        query_type: str = "general",
        force_implausible: bool = False,
    ) -> NLQueryResult:
        """Async version of run_nl_query for use in NiceGUI event handlers."""
        if self.world_state is None:
            return NLQueryResult(
                parse_result=QueryParseResult(
                    query=None, parsed=None, is_valid=False,
                ),
                error="No world model loaded. Ingest a story first.",
                summary="Error: No world model loaded.",
            )

        try:
            parse_result = await parse_query_async(
                natural_language,
                query_type=query_type,
                world_state=self.world_state,
                config=self.query_parsing_config,
            )
        except Exception as e:
            logger.exception("[AppState] Async query parsing failed")
            return NLQueryResult(
                parse_result=QueryParseResult(
                    query=None, parsed=None, is_valid=False,
                ),
                error=f"Query parsing failed: {e}",
                summary=f"Parse error: {e}",
            )

        self.last_parse = parse_result

        if not parse_result.is_valid or parse_result.query is None:
            errors = "; ".join(e.message for e in parse_result.validation_errors)
            result = NLQueryResult(
                parse_result=parse_result,
                error=f"Query validation failed: {errors}",
                summary=f"Validation failed: {errors}",
            )
            self._record_query_history(result)
            return result

        # Pipeline is sync — run in executor to avoid blocking NiceGUI event loop
        return await asyncio.to_thread(
            self.run_structured_query,
            parse_result.query,
            parse_result,
            force_implausible,
        )

    # ---- Manual edit ----

    def run_manual_edit(
        self,
        edited_prose: str,
        description: str = "",
        focus_entity_ids: list[str] | None = None,
        *,
        insert_after_event_id: str | None = None,
        insert_at_fabula_time: int | None = None,
        replace_event_ids: list[str] | None = None,
        replace_entity_ids: list[str] | None = None,
        replace_object_ids: list[str] | None = None,
        replace_location_ids: list[str] | None = None,
        replace_world_trait_ids: list[str] | None = None,
        replace_channel_ids: list[str] | None = None,
        replace_proposition_ids: list[str] | None = None,
        replace_concern_ids: list[tuple[str, str]] | None = None,
    ) -> NLQueryResult:
        """Submit user-authored prose as a ManualEditQuery through the pipeline.

        ``replace_*`` lists drop the named graph nodes (and their
        dependent edges/snapshots/concerns) before merging the
        re-extracted topology, giving true *replace* semantics across
        every namespace surfaced by the merge deletion pass.
        """
        query = ManualEditQuery(
            edited_prose=edited_prose,
            description=description,
            focus_entity_ids=focus_entity_ids or [],
            insert_after_event_id=insert_after_event_id,
            insert_at_fabula_time=insert_at_fabula_time,
            replace_event_ids=replace_event_ids or [],
            replace_entity_ids=replace_entity_ids or [],
            replace_object_ids=replace_object_ids or [],
            replace_location_ids=replace_location_ids or [],
            replace_world_trait_ids=replace_world_trait_ids or [],
            replace_channel_ids=replace_channel_ids or [],
            replace_proposition_ids=replace_proposition_ids or [],
            replace_concern_ids=list(replace_concern_ids or []),
        )
        return self.run_structured_query(query)

    # ---- Structured patch (skips prose round-trip) ----

    def apply_world_state_patch(
        self,
        patch: "Any",
        *,
        description: str = "",
    ) -> tuple[bool, list[str]]:
        """Apply a typed ``WorldStatePatch`` directly to the active world.

        Bypasses the prose re-extraction round-trip when the change is
        already known structurally (e.g. backfilling
        ``Belief.proposition_id``, committing a proposition truth,
        renaming a channel). Persists a new DB version on success and
        emits ``VERSION_CHANGED`` so the UI refreshes.

        Returns ``(ok, change_log)`` — ``ok`` is False on validation /
        apply / persist failures and the change log is a list of
        human-readable strings produced by the patcher.
        """
        from shadow_loom.ingestion import (
            WorldStatePatch as _WorldStatePatch,
            _apply_world_state_patch,
            _auto_repair,
            _programmatic_validation,
        )

        if self.world_state is None:
            return False, ["No active world state."]

        if isinstance(patch, dict):
            try:
                typed_patch = _WorldStatePatch.model_validate(patch)
            except Exception as exc:
                logger.exception("[AppState] Invalid world-state patch payload")
                return False, [f"Invalid patch payload: {exc}"]
        else:
            typed_patch = patch

        # Apply the patch to the RAW (un-projected) world so the
        # factual baseline + shadow_* sidecars are preserved across
        # the save round-trip. Patching the projected ``self
        # .world_state`` would flatten the per-branch AMWN clones
        # into ``entities`` and the subsequent
        # ``new_ws.model_dump_json()`` would persist the conflated
        # dict, silently overwriting the canonical factual baseline
        # for every cloned id.
        ws_for_patch = self.raw_world_state
        if ws_for_patch is None:
            return False, ["No active world state."]
        try:
            new_ws, changes = _apply_world_state_patch(
                ws_for_patch, typed_patch,
            )
        except Exception as exc:
            logger.exception("[AppState] World-state patch apply failed")
            return False, [f"Patch application failed: {exc}"]

        # ── Quality bridge: route the patched world through the same
        # programmatic validators and auto-repair pass that raw-text
        # ingestion uses. Previously this code path was a "back door"
        # that skipped every consistency check (orphan events, dead
        # actors, time ordering, status coherence, …) and persisted
        # whatever the patcher emitted. The LLM-based validator is
        # *not* called here because patches are meant to be cheap and
        # synchronous; the programmatic checks alone catch the
        # structural classes of regression that matter (dangling ids,
        # type mismatches, broken edges).
        try:
            new_ws, repairs = _auto_repair(new_ws)
        except Exception:
            logger.exception("[AppState] _auto_repair raised on patched world")
            repairs = []
        if repairs:
            changes = list(changes) + [f"(auto-repair) {r}" for r in repairs]

        try:
            issues = _programmatic_validation(new_ws)
        except Exception:
            logger.exception(
                "[AppState] _programmatic_validation raised on patched world"
            )
            issues = []
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        quarantined = bool(errors)
        if errors:
            logger.warning(
                "[AppState] Patched world failed validation \u2014 quarantining "
                "(%d errors, %d warnings)",
                len(errors), len(warnings),
            )
            for err in errors[:10]:
                logger.warning(
                    "[AppState]   [%s] %s", err.category, err.detail,
                )
            changes = list(changes) + [
                f"(quarantined: {len(errors)} validation error(s)) "
                f"{errors[0].detail}"
            ]
        elif warnings:
            logger.info(
                "[AppState] Patched world has %d validation warning(s); "
                "persisting normally.",
                len(warnings),
            )

        # Re-seed the versioned model with the patched world so the
        # next merge sees the corrected baseline. We still load
        # quarantined worlds into session state so the UI can show
        # the user what the patch produced; the quarantine only
        # affects persistence (source label + active-pointer move).
        # Capture branch identity BEFORE load_world_state — that
        # call replaces ``versioned_model.history`` with a fresh
        # synthetic factual head via ``from_world_state``, which
        # would otherwise demote a shadow patch to factual.
        _branch_world_id, _branch_label = self.head_branch()
        self.load_world_state(new_ws)

        # Persist a new DB version so the patch is durable.
        proj_id = self.project_id
        if proj_id is not None:
            try:
                ws_json = new_ws.model_dump_json()
                desc = description or typed_patch.notes or (
                    f"Structured patch ({len(changes)} change(s))"
                )
                if quarantined:
                    desc = f"[QUARANTINED] {desc}"
                source_label = (
                    "patch_world_state_quarantined"
                    if quarantined else "patch_world_state"
                )
                # Carry the loaded row's branch identity so a patch
                # on a shadow row stays on that shadow branch instead
                # of silently demoting to factual via the
                # ``save_version`` default ``world_id='factual'`` —
                # mirror of the MCP-side fix in
                # ``shadow_loom_mcp/server.py::patch_world_state``.
                # (Captured above before ``load_world_state``.)
                ver = save_version(
                    project_id=proj_id,
                    world_state_json=ws_json,
                    ancestor_id=self.current_version_row_id,
                    source=source_label,
                    description=desc,
                    user_id=self.user_id,
                    world_id=_branch_world_id,
                    branch_label=_branch_label,
                    actor_id=self.user_id,
                )
                # Quarantined patches do not become the active version;
                # the user / agent stays anchored on the parent so the
                # next read tool returns the validated baseline rather
                # than the broken patch result.
                if not quarantined:
                    self.current_version_row_id = ver.id
                if self.user_id is not None and not quarantined:
                    try:
                        set_active_version(proj_id, self.user_id, ver.id)
                    except Exception:
                        logger.exception(
                            "[AppState] Failed to update active-version pointer"
                        )
                try:
                    log_activity(
                        project_id=proj_id,
                        action=source_label,
                        user_id=self.user_id,
                        summary=desc,
                        version_id=ver.id,
                    )
                except Exception:
                    pass
                # Only fire VERSION_CHANGED for non-quarantined patches
                # \u2014 a quarantined version exists in history but is
                # not the active version, so the UI should keep showing
                # the parent. Tagged ``source="query_save"`` because
                # structured patches land on the active branch like a
                # normal query result rather than navigating away from
                # it (see ``_save_version_to_db`` for the rationale).
                if not quarantined:
                    self.emit(
                        StateEvent.VERSION_CHANGED,
                        version=ver.version,
                        source="query_save",
                    )
            except Exception:
                logger.exception("[AppState] Failed to persist patched world")
                # Persistence failed \u2014 return False so callers cannot
                # mistake the in-memory mutation for a durable commit
                # (round-3 audit). The change list still carries the
                # warning string so the UI banner can explain what
                # happened.
                return False, changes + ["(error) DB persistence failed"]
        return True, changes

    # ---- DB persistence ----

    def _save_version_to_db(
        self,
        pipeline_result: PipelineResult,
        raw_query: str | None = None,
        parsed_query_json: str | None = None,
        source: str = "pipeline",
        *,
        project_id: int | None = None,
        user_id: int | None = None,
        ancestor_row_id: int | None = None,
    ) -> None:
        """Persist the pipeline result as a new DB version (if a project is active).

        ``project_id`` / ``user_id`` / ``ancestor_row_id`` may be passed
        explicitly to pin the save against a snapshot of the session
        captured at query-start. This guards against project-switch
        races where ``self.project_id`` has moved on by the time an
        async pipeline run completes.
        """
        proj_id = project_id if project_id is not None else self.project_id
        if proj_id is None or pipeline_result.world_model is None:
            return
        # Avoid persisting against the wrong project if the user
        # navigated away mid-run.
        if (
            project_id is not None
            and self.project_id is not None
            and project_id != self.project_id
        ):
            logger.info(
                "[AppState] Skipping autosave \u2014 project switched (%s \u2192 %s)",
                project_id, self.project_id,
            )
            return
        save_user_id = user_id if user_id is not None else self.user_id
        ancestor = (
            ancestor_row_id if ancestor_row_id is not None
            else self.current_version_row_id
        )
        try:
            # Use the pipeline's own world_state JSON so we don't rely
            # on ``self.world_state`` (which may have been mutated by a
            # concurrent project load).
            ws_json = pipeline_result.world_model.current.model_dump_json()
            changeset_json = None
            branch_world_id = "factual"
            branch_label = None
            if pipeline_result.world_model.history:
                last_entry = pipeline_result.world_model.history[-1]
                if last_entry.changeset:
                    changeset_json = last_entry.changeset.model_dump_json()
                branch_world_id = last_entry.world_id
                branch_label = last_entry.branch_label

            ver = save_version(
                project_id=proj_id,
                world_state_json=ws_json,
                ancestor_id=ancestor,
                source=(
                    f"{source}_quarantined"
                    if getattr(pipeline_result, "continuation_quarantined", False)
                    else source
                ),
                description=(
                    f"[QUARANTINED] {source} query"
                    if getattr(pipeline_result, "continuation_quarantined", False)
                    else f"{source} query"
                ),
                changeset_json=changeset_json,
                raw_query=raw_query,
                parsed_query_json=parsed_query_json,
                prose=pipeline_result.prose,
                user_id=save_user_id,
                world_id=branch_world_id,
                branch_label=branch_label,
                actor_id=save_user_id,
            )
            # Only mutate session state if the user is still on the same
            # project as when the query started.
            if proj_id == self.project_id:
                self.current_version_row_id = ver.id

            # Mirror into the active-version pointer so MCP read tools
            # called by the agent default to the same version.
            if save_user_id is not None:
                try:
                    set_active_version(proj_id, save_user_id, ver.id)
                except Exception:
                    logger.exception(
                        "[AppState] Failed to update active-version pointer"
                    )

            # Log activity
            try:
                log_activity(
                    project_id=proj_id,
                    action=source,
                    user_id=save_user_id,
                    summary=f"{source} query \u2192 v{ver.version}",
                    version_id=ver.id,
                )
            except Exception:
                pass  # Activity logging is best-effort

            # Notify version change (only when still on the same project).
            # ``source="query_save"`` lets subscribers distinguish a new
            # version landing on the same branch (the user's current
            # query just persisted) from a deliberate version-swap
            # (sidebar click, rollback, project load). Without the
            # discriminator, panels that clear per-branch ephemera on
            # VERSION_CHANGED (chat history, answer panel) wipe the
            # user's just-submitted question between the user-message
            # render and the assistant-message render.
            if proj_id == self.project_id:
                self.emit(
                    StateEvent.VERSION_CHANGED,
                    version=ver.version,
                    source="query_save",
                )
        except Exception as exc:
            logger.exception("[AppState] Failed to save version to DB")
            # Surface the failure so the UI can show a toast / banner
            # instead of leaving the user under the impression their
            # query was persisted (round-3 audit). Listeners decide
            # whether to retry or prompt the user.
            try:
                self.emit(
                    StateEvent.AUTOSAVE_FAILED,
                    source=source,
                    error=str(exc),
                    project_id=proj_id,
                )
            except Exception:
                logger.exception(
                    "[AppState] AUTOSAVE_FAILED emit raised"
                )

    # ---- World model management ----

    def load_world_state(self, ws: WorldStateV1, *, max_snapshots: int = 10) -> None:
        """Set a new world state (e.g. from ingestion or DB load)."""
        self.world_state = ws
        self.versioned_model = VersionedWorldModel.from_world_state(
            ws, max_snapshots=max_snapshots,
        )
        # Maintain the AMWN-projected invariant on ``self.world_state``
        # so shadow-branch consumers see layered clones instead of the
        # raw factual baseline (round-4 audit). Cheap when the VWM
        # head is the factual branch \u2014 ``projected_for_branch``
        # returns ``self``.
        self._reproject_world_state_to_vwm_head()
        self.emit(StateEvent.WORLD_STATE_CHANGED)

    def _reproject_world_state_to_vwm_head(self) -> None:
        """Re-project ``self.world_state`` onto the VWM head branch.

        ``self.world_state`` is documented (L205) as the
        AMWN-PROJECTED view (shadow clones layered into ``entities``
        etc. via :meth:`projected_for_branch`). After any swap that
        assigns ``self.world_state = self.versioned_model.current``
        — the post-pipeline callback, the background re-extraction
        handler, ``rollback_to`` — the assignment leaves
        ``world_state`` as the raw merged ``WorldStateV1`` (factual
        baseline only; shadow clones live in ``shadow_entities[branch_label]``
        and its peers). Without re-projecting, every UI consumer that
        reads ``state.world_state.entities[id]`` on a shadow head
        sees the factual record — producing prose↔inspector drift,
        invalid LLM query parsing (shadow-spawned entities absent
        from the dynamic schema), and viz panels rendering the
        wrong branch. Cheap: ``projected_for_branch`` returns
        ``self`` for factual rows.

        Mirrors the projection block at the end of ``load_db_version``
        and ``load_project`` so all four code paths that mutate
        ``self.world_state`` maintain the same invariant.
        """
        if self.versioned_model is None or not self.versioned_model.history:
            return
        _head = self.versioned_model.history[-1]
        _projected = self.world_state.projected_for_branch(
            branch_world_id=_head.world_id,
            branch_label=_head.branch_label,
        )
        if _projected is not self.world_state:
            self.world_state = _projected

    def _rehydrate_vwm_history_from_db(self, version_row_id: int) -> None:
        """Replace ``versioned_model.history`` with the DB lineage.

        ``VersionedWorldModel.from_world_state`` always seeds a single
        synthetic ``WorldModelVersion(version=0, world_id="factual")``
        which loses the loaded row's branch identity. If the user
        selected a shadow head, the next pipeline run would consult
        ``vwm.history[-1].world_id == "factual"`` via
        ``_resolve_branch_policy`` and silently re-tag the new
        version onto the factual mainline. ``_gather_preceding_prose``
        would also lose all real prose continuity.

        This rebuilds ``history`` by walking the DB ancestor chain and
        materialising one ``WorldModelVersion`` per row carrying the
        real ``world_id``, ``branch_label``, ``prose``, ``source``,
        ``description``, and timestamp. Snapshots stay as the loaded
        current world (older snapshots are not re-hydrated — they
        would require deserialising every ancestor's
        ``world_state_json`` and the rollback UI uses the DB version
        list directly anyway).

        **R19-L1 (documented invariant):** Because only the current
        snapshot is materialised, in-process ``rollback`` to an
        intermediate ``WorldModelVersion`` will *not* find a matching
        ``WorldSnapshot`` and will raise ``KeyError``. The supported
        rollback UX after a DB rehydrate is to pick an ancestor row
        from the version sidebar and re-load via ``load_db_version``,
        which goes through the DB and rebuilds the snapshot chain
        deterministically. Callers that need in-process rollback
        across all rehydrated history must first reload every ancestor
        ``world_state_json`` (expensive, not performed by default).
        """
        # Local import keeps the module import graph identical to
        # before this change (state.py historically only imported
        # VersionedWorldModel from extract_graph).
        from shadow_loom.extract_graph import (
            WorldModelVersion,
            WorldSnapshot,
        )
        if self.versioned_model is None:
            return
        try:
            lineage = get_version_lineage_rows(version_row_id)
        except Exception:
            logger.exception(
                "[AppState] Failed to load version lineage for "
                "version_row_id=%s; keeping synthetic v0 history.",
                version_row_id,
            )
            return
        if not lineage:
            return
        history: list[WorldModelVersion] = []
        for idx, row in enumerate(lineage):
            history.append(WorldModelVersion(
                # Use sequential 0,1,2 in-memory indices (matches
                # how from_world_state seeds v0). The DB-side
                # version number is independent and is consulted via
                # ``current_version_row_id`` when the next save runs.
                version=idx,
                timestamp=str(row.created_at) if row.created_at else "",
                source=row.source or "db_load",
                description=row.description or "",
                prose=row.prose,
                world_id=row.world_id if row.world_id in ("factual", "shadow") else "factual",
                branch_label=row.branch_label,
            ))
        # Only the current snapshot is materialised; rebuild it under
        # the new head version index so ``rollback`` can still find
        # version-0 if invoked.
        head_version = history[-1].version
        self.versioned_model = self.versioned_model.model_copy(update={
            "history": history,
            "snapshots": [
                WorldSnapshot(
                    version=head_version,
                    world_state=self.versioned_model.current,
                ),
            ],
        })

    def load_project(
        self,
        project_id: int,
        project_name: str,
        world_state: WorldStateV1,
        version_row_id: int | None = None,
        raw_text: str | None = None,
    ) -> None:
        """Load a full project into state (convenience method).

        **R19-L2 (documented invariant — lineage-only history):** The
        rehydrated ``versioned_model.history`` reflects only the
        *direct ancestor chain* of ``version_row_id`` (one parent per
        row, via ``ancestor_id``). Sibling shadow descendants of any
        ancestor — e.g. parallel ``what-if`` branches off the same
        factual row — are **not** included; the in-process
        ``versioned_model`` is therefore a single-path view, not a
        full DAG snapshot. The full version tree remains accessible
        via ``db.get_version_tree(project_id)`` and is used by the
        version sidebar to render sibling branches. Callers needing
        DAG awareness in-process should query the DB directly rather
        than walking ``versioned_model.history``.
        """
        self.project_id = project_id
        self.project_name = project_name
        self.current_version_row_id = version_row_id
        self.raw_text = raw_text
        self.query_history.clear()
        self.last_result = None
        self.last_parse = None
        self.fabula_cursor = None
        self.syuzhet_cursor = None
        self.load_world_state(world_state)
        # Re-hydrate VWM history from the DB lineage so the loaded
        # row's branch identity (world_id, branch_label, prose)
        # survives the swap. Without this, the next pipeline run on a
        # shadow head would silently re-tag onto factual mainline.
        if version_row_id is not None:
            self._rehydrate_vwm_history_from_db(version_row_id)
        # AMWN-split projection on load (same as ``load_db_version``):
        # swap ``world_state.entities`` for the per-branch layered
        # view so UI consumers see the do(\u00b7)-modified entities on
        # a shadow row instead of the factual baseline. No-op on
        # factual rows.
        if self.versioned_model is not None and self.versioned_model.history:
            _head = self.versioned_model.history[-1]
            _projected = self.world_state.projected_for_branch(
                branch_world_id=_head.world_id,
                branch_label=_head.branch_label,
            )
            if _projected is not self.world_state:
                self.world_state = _projected
        # Resolve the version *number* for the loaded row so the
        # version sidebar / header label can render "v{N}" without
        # waiting for the next save or rollback. Without this emit
        # the label keeps showing the previous project's version.
        if version_row_id is not None:
            try:
                row = get_version_by_id(version_row_id)
            except Exception:
                logger.exception(
                    "[AppState] Failed to resolve version_row_id=%s on load_project",
                    version_row_id,
                )
                row = None
            if row is not None:
                self.emit(
                    StateEvent.VERSION_CHANGED,
                    version=row.version,
                    source="navigation",
                )
        self.emit(StateEvent.PROJECT_LOADED, project_id=project_id)

    def rollback_to(self, version: int) -> None:
        """Rollback the versioned world model to a previous snapshot."""
        if self.versioned_model is None:
            raise ValueError("No versioned model to rollback.")
        self.versioned_model = self.versioned_model.rollback(version)
        self.world_state = self.versioned_model.current
        self._reproject_world_state_to_vwm_head()
        # A version swap repositions the world; previously-active
        # cursors point into a different timeline and would render
        # garbage on the new one. Reset both so every time-aware panel
        # snaps back to "live" until the user scrubs again. Per-branch
        # derived state (query history, last result/parse, selected
        # node) is also dropped for the same reason as in
        # ``load_db_version``.
        self.fabula_cursor = None
        self.syuzhet_cursor = None
        self.query_history.clear()
        self.last_result = None
        self.last_parse = None
        self.selected_node_id = None
        self.selected_node_type = None
        self.emit(StateEvent.WORLD_STATE_CHANGED)
        self.emit(StateEvent.FABULA_CURSOR_CHANGED, cursor=None)
        self.emit(StateEvent.SYUZHET_CURSOR_CHANGED, cursor=None)
        self.emit(StateEvent.NODE_SELECTED, node_id=None, node_type=None)
        # ``rollback()`` appends a NEW head version (target+1) carrying
        # the rolled-back world; emitting the requested target version
        # here would desync UI labels from ``versioned_model.version``
        # (round-5 audit).
        self.emit(
            StateEvent.VERSION_CHANGED,
            version=self.versioned_model.version,
            source="navigation",
        )

    def load_db_version(
        self,
        ws: WorldStateV1,
        version_row_id: int,
        version_number: int | None = None,
    ) -> None:
        """Replace the live world with a DB-loaded version snapshot.

        Use this from version-load UI paths (sidebar, dialogs, story
        re-ingest) instead of writing ``world_state``/
        ``current_version_row_id`` directly. Centralising the swap
        keeps four invariants intact:

        * cursors are reset so we don't render the new world through
          the previous version's timeline;
        * per-branch derived state — query history, last pipeline
          result, last parse, selected node — is dropped, since those
          all reference entity/event ids and fabula times from the
          previous branch and would leak into panels that read
          ``state`` fields directly;
        * ``WORLD_STATE_CHANGED`` and ``VERSION_CHANGED`` fire in
          lockstep so subscribers (Story / Audit / Sidebar) update
          atomically;
        * snapshot/physics caches are flushed via the existing
          ``WORLD_STATE_CHANGED`` invalidation hook.
        """
        # Set the version pointer BEFORE swapping world state so that
        # any subscriber handling ``WORLD_STATE_CHANGED`` sees a
        # consistent (new world_state, new version_row_id) pair. The
        # previous order (load_world_state → set id) emitted
        # WORLD_STATE_CHANGED while ``current_version_row_id`` still
        # held the *previous* version's row id — silently violating
        # the lockstep invariant the docstring promises and giving
        # any panel that keys on (world_state, version_row_id) a
        # one-frame view of stale lineage.
        #
        # Also drop per-branch derived state. Query history, the last
        # pipeline result, the last parse, and the selected-node
        # cursor are all keyed against the *previous* branch's
        # entity/event ids and fabula times; leaving them in place
        # makes panels that read them directly (audit_tab,
        # story_tab, reasoning_tab, reasoning_trace) display content
        # from the prior branch after the swap.
        self.fabula_cursor = None
        self.syuzhet_cursor = None
        self.query_history.clear()
        self.last_result = None
        self.last_parse = None
        self.selected_node_id = None
        self.selected_node_type = None
        self.current_version_row_id = version_row_id
        self.load_world_state(ws)
        # Carry the DB row's branch identity into the in-memory VWM
        # history so subsequent queries inherit the correct world_id /
        # branch_label and prose continuity walks the real lineage.
        self._rehydrate_vwm_history_from_db(version_row_id)
        # AMWN-split projection on load: when the loaded row is on a
        # shadow branch, swap ``world_state.entities`` (and the other
        # sidecar-managed dicts) for the per-branch layered view
        # before any UI consumer reads it. Without this, every
        # ``state.world_state.entities[id]`` lookup in viz.py,
        # explorer_tab, reasoning_helpers, etc. returns the FACTUAL
        # entity record \u2014 producing prose\u2194inspector drift on
        # Rung-2/3 rows (the prose narrates the do(\u00b7)-modified
        # world; the Inspector shows the factual baseline). Cheap:
        # ``projected_for_branch`` returns ``self`` for factual rows.
        if self.versioned_model is not None and self.versioned_model.history:
            _head = self.versioned_model.history[-1]
            _projected = self.world_state.projected_for_branch(
                branch_world_id=_head.world_id,
                branch_label=_head.branch_label,
            )
            if _projected is not self.world_state:
                self.world_state = _projected
        self.emit(StateEvent.FABULA_CURSOR_CHANGED, cursor=None)
        self.emit(StateEvent.SYUZHET_CURSOR_CHANGED, cursor=None)
        self.emit(StateEvent.NODE_SELECTED, node_id=None, node_type=None)
        if version_number is not None:
            self.emit(
                StateEvent.VERSION_CHANGED,
                version=version_number,
                source="navigation",
            )

    def to_json(self) -> str:
        """Serialize the current world state to JSON for persistence.

        Routes through :attr:`raw_world_state` so the dump preserves
        the factual baseline + ``shadow_*`` sidecars; dumping the
        projected :attr:`world_state` would conflate the two.
        """
        raw = self.raw_world_state
        if raw is None:
            return "{}"
        return raw.model_dump_json(indent=2)

    def select_node(self, node_id: str | None, node_type: str | None = None) -> None:
        """Select a node for cross-component inspector sync."""
        self.selected_node_id = node_id
        self.selected_node_type = node_type
        self.emit(StateEvent.NODE_SELECTED, node_id=node_id, node_type=node_type)

    def set_fabula_cursor(self, t: int | None, *, immediate: bool = False) -> None:
        """Set the global fabula cursor (None = live).

        The cursor field updates immediately (so any synchronous
        reader sees the latest value), but the
        :data:`StateEvent.FABULA_CURSOR_CHANGED` emit is debounced by
        ``_CURSOR_DEBOUNCE_S`` to coalesce bursty drags / keyboard
        repeats into a single fan-out. Pass ``immediate=True`` to
        bypass the debounce (e.g. on programmatic hydration).
        """
        if self.fabula_cursor == t:
            return
        self.fabula_cursor = t
        self._schedule_cursor_emit(
            "fabula", StateEvent.FABULA_CURSOR_CHANGED, t,
            immediate=immediate,
        )

    def set_syuzhet_cursor(self, s: int | None, *, immediate: bool = False) -> None:
        """Set the global syuzhet cursor (None = live). Debounced; see
        :meth:`set_fabula_cursor`."""
        if self.syuzhet_cursor == s:
            return
        self.syuzhet_cursor = s
        self._schedule_cursor_emit(
            "syuzhet", StateEvent.SYUZHET_CURSOR_CHANGED, s,
            immediate=immediate,
        )

    @property
    def active_cursor(self) -> int | None:
        """The cursor for the currently selected ``time_axis``.

        Cross-panel sliders read this so an axis flip automatically
        switches between :attr:`fabula_cursor` (Genette story order)
        and :attr:`syuzhet_cursor` (telling order) without each panel
        re-implementing the dispatch.
        """
        if self.time_axis == "syuzhet":
            return self.syuzhet_cursor
        return self.fabula_cursor

    def set_active_cursor(self, value: int | None, *, immediate: bool = False) -> None:
        """Set the cursor on the currently active axis.

        Routes to :meth:`set_fabula_cursor` or :meth:`set_syuzhet_cursor`
        based on :attr:`time_axis`. Used by the World and Social tab
        sliders so the same widget drives the correct axis cursor.
        """
        if self.time_axis == "syuzhet":
            self.set_syuzhet_cursor(value, immediate=immediate)
        else:
            self.set_fabula_cursor(value, immediate=immediate)

    def set_time_axis(self, axis: str) -> None:
        """Set the global time axis ("fabula" | "syuzhet").

        Charts that bin on event time read this to decide whether to
        use ``evt.fabula_time`` (Genette: story order) or
        ``evt.syuzhet_index`` (telling order). Emits
        :data:`StateEvent.TIME_AXIS_CHANGED`.
        """
        axis = (axis or "fabula").lower()
        if axis not in ("fabula", "syuzhet"):
            axis = "fabula"
        if self.time_axis == axis:
            return
        self.time_axis = axis
        self.emit(StateEvent.TIME_AXIS_CHANGED, axis=axis)

    def _schedule_cursor_emit(
        self, axis: str, event: "StateEvent", value: int | None,
        *, immediate: bool,
    ) -> None:
        prev = self._cursor_emit_tasks.pop(axis, None)
        if prev is not None and not prev.done():
            prev.cancel()
        if immediate:
            self.emit(event, cursor=value)
            return

        # Probe for a running loop *before* creating the coroutine; if
        # there is no loop (e.g. in unit tests) we emit synchronously
        # and never instantiate ``_delayed()`` \u2014 avoids the
        # ``coroutine was never awaited`` RuntimeWarning.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.emit(event, cursor=value)
            return

        async def _delayed():
            try:
                await asyncio.sleep(_CURSOR_DEBOUNCE_S)
            except asyncio.CancelledError:
                return
            self.emit(event, cursor=value)

        task = asyncio.create_task(_delayed(), name=f"cursor-{axis}")
        self._cursor_emit_tasks[axis] = task
        self._async_tasks.add(task)
        task.add_done_callback(self._async_tasks.discard)

    def set_active_path(self, path: str) -> None:
        """Record the currently visible tab path (e.g. ``"causality.affective"``).

        Emits :data:`StateEvent.ACTIVE_PATH_CHANGED` so panels that
        deferred work while hidden can refresh exactly once on
        becoming visible.
        """
        if self.active_path == path:
            return
        self.active_path = path
        self.emit(StateEvent.ACTIVE_PATH_CHANGED, path=path)

    def is_path_visible(self, path: str) -> bool:
        """True iff the given dotted tab path is the active one (or its
        parent). An empty ``active_path`` (initial state) is treated as
        visible so first-paint isn't deferred."""
        if not self.active_path:
            return True
        return self.active_path == path or self.active_path.startswith(path + ".")

    def spawn_panel_task(
        self, panel_id: str, coro, *, name: str | None = None,
    ) -> "asyncio.Task":
        """Schedule ``coro`` for a named panel, cancelling any prior
        in-flight task for the same panel id.

        Use this for slider/cursor-driven refreshes so rapid scrubs
        collapse to the latest render only — the previous half-built
        snapshot is cancelled before the next one starts.
        """
        prev = self._panel_tasks.pop(panel_id, None)
        if prev is not None and not prev.done():
            prev.cancel()
        task = asyncio.create_task(coro, name=name or f"panel:{panel_id}")
        self._panel_tasks[panel_id] = task
        self._async_tasks.add(task)

        def _cleanup(t: asyncio.Task) -> None:
            self._async_tasks.discard(t)
            # Only remove from the slot if we're still the latest task.
            if self._panel_tasks.get(panel_id) is t:
                self._panel_tasks.pop(panel_id, None)

        task.add_done_callback(_cleanup)
        return task

    @staticmethod
    def from_json(data: str) -> WorldStateV1:
        """Deserialize a world state from JSON."""
        return WorldStateV1.model_validate_json(data)
