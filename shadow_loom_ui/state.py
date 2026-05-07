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
    humanize_pipeline_result,
    run_pipeline,
)
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
    ACTIVE_PATH_CHANGED = "active_path_changed"
    WORLD_FACTS_CHANGED = "world_facts_changed"
    PROJECT_LIST_CHANGED = "project_list_changed"


# Debounce window (seconds) for cursor emit fan-out. Coalesces bursty
# slider drags / keyboard repeats so subscribers don't see every tick.
_CURSOR_DEBOUNCE_S = 0.12


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

    # Background task registry (in-flight + recently completed)
    background_tasks: List[BackgroundTask] = field(default_factory=list)
    _max_completed_tasks: int = 20

    # Live asyncio task handles for in-flight background coroutines.
    # Tracking these prevents "Task was destroyed but it is pending"
    # warnings (asyncio holds only weak refs to created tasks) and lets
    # us cancel everything cleanly on session shutdown.
    _async_tasks: "set[asyncio.Task]" = field(default_factory=set)

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
            self.query_history.append(result)
            return result

        return self.run_structured_query(
            parse_result.query,
            parse_result=parse_result,
            force_implausible=force_implausible,
        )

    def run_structured_query(
        self,
        query: UserRequest,
        parse_result: Optional[QueryParseResult] = None,
        force_implausible: bool = False,
    ) -> NLQueryResult:
        """Execute a pre-built structured query through the pipeline."""
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

        # Snapshot execution context up-front so a project switch (or a
        # version load) that happens while the pipeline is running in a
        # worker thread cannot corrupt the persisted ancestor lineage.
        ctx_project_id = self.project_id
        ctx_user_id = self.user_id
        ctx_ancestor_row_id = self.current_version_row_id

        logger.info("[AppState] Running pipeline: query_type=%s", query.query_type)
        try:
            pipeline_result = run_pipeline(
                query,
                versioned_model=self.versioned_model,
                config=self.pipeline_config,
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
            self.query_history.append(result)
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
        if (
            not short_circuited
            and not readonly_query
            and not pipeline_result.reextraction_failed
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
        self.query_history.append(result)

        self.emit(StateEvent.PIPELINE_RESULT, result=result)
        self.emit(StateEvent.QUERY_COMPLETE, result=result)

        return result

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
            self.query_history.append(result)
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
                source=source,
                description=f"{source} query",
                changeset_json=changeset_json,
                raw_query=raw_query,
                parsed_query_json=parsed_query_json,
                prose=pipeline_result.prose,
                user_id=save_user_id,
                world_id=branch_world_id,
                branch_label=branch_label,
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

            # Notify version change (only when still on the same project)
            if proj_id == self.project_id:
                self.emit(StateEvent.VERSION_CHANGED, version=ver.version)
        except Exception:
            logger.exception("[AppState] Failed to save version to DB")

    # ---- World model management ----

    def load_world_state(self, ws: WorldStateV1, *, max_snapshots: int = 10) -> None:
        """Set a new world state (e.g. from ingestion or DB load)."""
        self.world_state = ws
        self.versioned_model = VersionedWorldModel.from_world_state(
            ws, max_snapshots=max_snapshots,
        )
        self.emit(StateEvent.WORLD_STATE_CHANGED)

    def load_project(
        self,
        project_id: int,
        project_name: str,
        world_state: WorldStateV1,
        version_row_id: int | None = None,
        raw_text: str | None = None,
    ) -> None:
        """Load a full project into state (convenience method)."""
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
                self.emit(StateEvent.VERSION_CHANGED, version=row.version)
        self.emit(StateEvent.PROJECT_LOADED, project_id=project_id)

    def rollback_to(self, version: int) -> None:
        """Rollback the versioned world model to a previous snapshot."""
        if self.versioned_model is None:
            raise ValueError("No versioned model to rollback.")
        self.versioned_model = self.versioned_model.rollback(version)
        self.world_state = self.versioned_model.current
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
        self.emit(StateEvent.VERSION_CHANGED, version=version)

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
        self.emit(StateEvent.FABULA_CURSOR_CHANGED, cursor=None)
        self.emit(StateEvent.SYUZHET_CURSOR_CHANGED, cursor=None)
        self.emit(StateEvent.NODE_SELECTED, node_id=None, node_type=None)
        if version_number is not None:
            self.emit(StateEvent.VERSION_CHANGED, version=version_number)

    def to_json(self) -> str:
        """Serialize the current world state to JSON for persistence."""
        if self.world_state is None:
            return "{}"
        return self.world_state.model_dump_json(indent=2)

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
