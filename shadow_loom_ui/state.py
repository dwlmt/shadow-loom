"""Shared application state for the Shadow-Loom UI.

Holds the current project, world model, and pipeline runner.
Provides the natural-language query interface that ties
query_parsing → pipeline → UI updates together.

Uses an event-bus pattern so multiple components can subscribe
to state changes without overwriting each other's callbacks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.models import WorldStateV1
from shadow_loom.pipeline import PipelineConfig, PipelineResult, run_pipeline
from shadow_loom.query_models import ManualEditQuery, UserRequest
from shadow_loom.query_parsing import (
    QueryParseResult,
    QueryParsingConfig,
    parse_query,
    parse_query_async,
)
from shadow_loom.db import (
    save_version,
    get_latest_version,
    log_activity,
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

    # Event bus: multiple listeners per event
    _listeners: Dict[StateEvent, List[Callable]] = field(default_factory=dict)

    # Legacy callbacks (kept for backward compat during migration)
    on_world_state_changed: Optional[Callable] = None
    on_pipeline_result: Optional[Callable] = None

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
        """Notify all listeners of an event."""
        for cb in self._listeners.get(event, []):
            try:
                cb(**kwargs)
            except Exception:
                logger.exception("[AppState] Listener error for %s", event.value)

    # ---- Session setup ----

    def set_user(self, user_id: int, username: str, display_name: str = "", avatar_url: str = "") -> None:
        """Set the authenticated user for this session."""
        self.user_id = user_id
        self.username = username
        self.display_name = display_name or username
        self.avatar_url = avatar_url

    # ---- Natural language query interface ----

    def run_nl_query(self, natural_language: str, query_type: str = "general") -> NLQueryResult:
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

        return self.run_structured_query(parse_result.query, parse_result=parse_result)

    def run_structured_query(
        self,
        query: UserRequest,
        parse_result: Optional[QueryParseResult] = None,
    ) -> NLQueryResult:
        """Execute a pre-built structured query through the pipeline."""
        if self.world_state is None:
            return NLQueryResult(
                parse_result=parse_result or QueryParseResult(
                    query=query, parsed=None, is_valid=True,
                ),
                error="No world model loaded.",
                summary="Error: No world model loaded.",
            )

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

        # Update world model if pipeline produced a new versioned model
        if pipeline_result.world_model is not None:
            self.versioned_model = pipeline_result.world_model
            self.world_state = pipeline_result.world_model.current
            # Notify via event bus
            self.emit(StateEvent.WORLD_STATE_CHANGED)
            # Legacy callback
            if self.on_world_state_changed:
                self.on_world_state_changed()

        self.last_result = pipeline_result

        # Persist version to DB
        self._save_version_to_db(
            pipeline_result=pipeline_result,
            raw_query=getattr(query, 'edited_prose', None) or (
                parse_result.parsed.reasoning if parse_result and parse_result.parsed else None
            ),
            parsed_query_json=query.model_dump_json() if query else None,
            source=query.query_type,
        )

        # Build summary
        summary_parts = [f"Query type: {pipeline_result.query_type}"]
        if pipeline_result.prose:
            summary_parts.append(f"Prose generated ({len(pipeline_result.prose)} chars)")
        if pipeline_result.converged is not None:
            summary_parts.append(
                f"Audit: {'converged' if pipeline_result.converged else 'did not converge'}"
                f" ({pipeline_result.audit_iterations} iterations)"
            )
        if pipeline_result.world_model:
            summary_parts.append(f"World model v{pipeline_result.world_model.version}")

        result = NLQueryResult(
            parse_result=parse_result or QueryParseResult(
                query=query, parsed=None, is_valid=True,
            ),
            pipeline_result=pipeline_result,
            summary=" | ".join(summary_parts),
        )
        self.query_history.append(result)

        # Notify via event bus
        self.emit(StateEvent.PIPELINE_RESULT, result=result)
        # Legacy callback
        if self.on_pipeline_result:
            self.on_pipeline_result(result)

        return result

    async def run_nl_query_async(self, natural_language: str, query_type: str = "general") -> NLQueryResult:
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

        # Pipeline is sync — run in executor
        return self.run_structured_query(parse_result.query, parse_result=parse_result)

    # ---- Manual edit ----

    def run_manual_edit(
        self,
        edited_prose: str,
        description: str = "",
        focus_entity_ids: list[str] | None = None,
    ) -> NLQueryResult:
        """Submit user-authored prose as a ManualEditQuery through the pipeline."""
        query = ManualEditQuery(
            edited_prose=edited_prose,
            description=description,
            focus_entity_ids=focus_entity_ids or [],
        )
        return self.run_structured_query(query)

    # ---- DB persistence ----

    def _save_version_to_db(
        self,
        pipeline_result: PipelineResult,
        raw_query: str | None = None,
        parsed_query_json: str | None = None,
        source: str = "pipeline",
    ) -> None:
        """Persist the pipeline result as a new DB version (if a project is active)."""
        if self.project_id is None or self.world_state is None:
            return
        try:
            ancestor_id = self.current_version_row_id

            changeset_json = None
            if pipeline_result.world_model and pipeline_result.world_model.history:
                last_entry = pipeline_result.world_model.history[-1]
                if last_entry.changeset:
                    changeset_json = last_entry.changeset.model_dump_json()

            ver = save_version(
                project_id=self.project_id,
                world_state_json=self.world_state.model_dump_json(),
                ancestor_id=ancestor_id,
                source=source,
                description=f"{source} query",
                changeset_json=changeset_json,
                raw_query=raw_query,
                parsed_query_json=parsed_query_json,
                prose=pipeline_result.prose,
                user_id=self.user_id,
            )
            self.current_version_row_id = ver.id

            # Log activity
            try:
                log_activity(
                    project_id=self.project_id,
                    action=source,
                    user_id=self.user_id,
                    summary=f"{source} query → v{ver.version}",
                    version_id=ver.id,
                )
            except Exception:
                pass  # Activity logging is best-effort

            # Notify version change
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
        if self.on_world_state_changed:
            self.on_world_state_changed()

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
        self.load_world_state(world_state)
        self.emit(StateEvent.PROJECT_LOADED, project_id=project_id)

    def rollback_to(self, version: int) -> None:
        """Rollback the versioned world model to a previous snapshot."""
        if self.versioned_model is None:
            raise ValueError("No versioned model to rollback.")
        self.versioned_model = self.versioned_model.rollback(version)
        self.world_state = self.versioned_model.current
        self.emit(StateEvent.WORLD_STATE_CHANGED)
        self.emit(StateEvent.VERSION_CHANGED, version=version)
        if self.on_world_state_changed:
            self.on_world_state_changed()

    def to_json(self) -> str:
        """Serialize the current world state to JSON for persistence."""
        if self.world_state is None:
            return "{}"
        return self.world_state.model_dump_json(indent=2)

    @staticmethod
    def from_json(data: str) -> WorldStateV1:
        """Deserialize a world state from JSON."""
        return WorldStateV1.model_validate_json(data)
