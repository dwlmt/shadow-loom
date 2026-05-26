# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Round-7 UI-consistency audit fixes (2026-05-26).

Verifies that ``VERSION_CHANGED`` emits carry a ``source`` kwarg
distinguishing engine-driven version saves from user-driven
navigation, and that the ``chat`` / ``answer_panel`` clear-on-branch
guards correctly filter on that source so they don't wipe per-branch
ephemera (the user's just-submitted chat message, a still-relevant
Q&A answer) when the engine appends a new child version on the same
branch the user is already on.
"""

from __future__ import annotations

import pytest

from shadow_loom.db import (
    create_project,
    init_db,
    save_version,
    upsert_user,
)
from shadow_loom_ui.state import AppState, StateEvent


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db("sqlite://")
    yield


def _seed_project() -> tuple[int, int]:
    user = upsert_user(
        "local", "round7-test", "round7user",
        email="r7@example.com",
    )
    proj = create_project(
        name="Round7Test", owner_id=user.id, description="",
    )
    return user.id, proj.id


def _empty_ws():
    from tests.conftest import make_empty_world_state
    return make_empty_world_state()


def _capture_version_emits(state: AppState) -> list[dict]:
    """Subscribe to ``VERSION_CHANGED`` and return a list that the
    test mutates as events are emitted."""
    captured: list[dict] = []

    def _listener(**kwargs):
        captured.append(dict(kwargs))

    state.on(StateEvent.VERSION_CHANGED, _listener)
    return captured


# =====================================================================
# Source-tagging on emits
# =====================================================================


class TestVersionChangedSourceTagging:
    """Every VERSION_CHANGED emit must carry a ``source`` kwarg so
    subscribers can distinguish a query-driven version save from a
    user-driven navigation."""

    def test_load_db_version_tags_navigation(self):
        uid, pid = _seed_project()
        ws_json = _empty_ws().model_dump_json()
        v0 = save_version(
            project_id=pid, world_state_json=ws_json,
            version=0, source="ingestion", description="root",
            user_id=uid,
        )

        state = AppState()
        state.user_id = uid
        state.project_id = pid
        captured = _capture_version_emits(state)

        ws = _empty_ws()
        state.load_db_version(ws, v0.id, version_number=v0.version)

        assert captured, "load_db_version must emit VERSION_CHANGED"
        # Final emit (post-load) is the canonical navigation event.
        last = captured[-1]
        assert last.get("source") == "navigation"
        assert last.get("version") == v0.version

    def test_rollback_to_tags_navigation(self):
        from shadow_loom.extract_graph import VersionedWorldModel

        state = AppState()
        state.versioned_model = VersionedWorldModel.from_world_state(
            _empty_ws(),
        )
        captured = _capture_version_emits(state)

        state.rollback_to(0)

        assert captured
        assert captured[-1].get("source") == "navigation"

    def test_apply_world_state_patch_tags_query_save(self):
        from shadow_loom.ingestion import WorldStatePatch

        uid, pid = _seed_project()
        state = AppState()
        state.user_id = uid
        state.project_id = pid
        state.load_world_state(_empty_ws())
        captured = _capture_version_emits(state)

        ok, _changes = state.apply_world_state_patch(
            WorldStatePatch(notes="round7 no-op"),
            description="round7 regression",
        )
        assert ok

        # apply_world_state_patch is a "version landed on the active
        # branch" event, not a navigation away from it.
        assert captured
        assert captured[-1].get("source") == "query_save"

    def test_save_version_to_db_tags_query_save(self):
        """``_save_version_to_db`` is the post-pipeline persistence
        hook for every mutation query type (observation, intervention,
        counterfactual, directive). It must tag its VERSION_CHANGED
        emit ``source='query_save'`` so chat / answer-panel guards
        don't treat it as a navigation."""
        from shadow_loom.extract_graph import VersionedWorldModel
        from shadow_loom.pipeline import PipelineResult

        uid, pid = _seed_project()
        state = AppState()
        state.user_id = uid
        state.project_id = pid
        state.load_world_state(_empty_ws())

        vwm = VersionedWorldModel.from_world_state(_empty_ws())
        pr = PipelineResult(
            world_model=vwm,
            query_type="observation",
        )
        captured = _capture_version_emits(state)

        state._save_version_to_db(
            pipeline_result=pr,
            raw_query="round7 observation",
            source="observation",
        )

        assert captured, "_save_version_to_db must emit VERSION_CHANGED"
        assert captured[-1].get("source") == "query_save"


# =====================================================================
# Subscriber guard logic — verifies the filter contract used by
# chat._clear_chat_on_branch and answer_panel._clear
# =====================================================================


class TestClearOnBranchSourceFilter:
    """The chat and answer-panel clear handlers must clear on
    navigation (or unsourced emits, for PROJECT_LOADED and legacy
    callers) but NOT on engine-driven query saves."""

    def test_chat_clear_skipped_for_query_save(self):
        """End-to-end behavioural check: build a state, subscribe a
        chat-style clear handler that mirrors the production guard,
        emit a query_save event, and verify the handler did NOT
        clear."""
        state = AppState()
        messages: list[dict] = [{"role": "user", "text": "hi"}]
        cleared = {"v": False}

        def _clear(**kwargs):
            if kwargs.get("source") == "query_save":
                return
            messages.clear()
            cleared["v"] = True

        state.on(StateEvent.VERSION_CHANGED, _clear)

        # Query-save emit must NOT clear.
        state.emit(
            StateEvent.VERSION_CHANGED, version=1, source="query_save",
        )
        assert messages == [{"role": "user", "text": "hi"}]
        assert cleared["v"] is False

        # Navigation emit MUST clear.
        state.emit(
            StateEvent.VERSION_CHANGED, version=0, source="navigation",
        )
        assert messages == []
        assert cleared["v"] is True

    def test_chat_clear_runs_on_unsourced_emit(self):
        """Unsourced VERSION_CHANGED emits (PROJECT_LOADED bridges,
        any legacy caller) still trigger the clear so the default is
        safe."""
        state = AppState()
        cleared = {"v": False}

        def _clear(**kwargs):
            if kwargs.get("source") == "query_save":
                return
            cleared["v"] = True

        state.on(StateEvent.VERSION_CHANGED, _clear)
        state.emit(StateEvent.VERSION_CHANGED, version=0)
        assert cleared["v"] is True


# =====================================================================
# Integration: live chat / answer_panel handler bound by the
# production builders also honours the filter.
# =====================================================================


class TestProductionHandlerSourceFilter:
    """Pin the actual production source-check string so a future
    typo in ``chat._clear_chat_on_branch`` or
    ``answer_panel._clear`` would be caught here even without a
    NiceGUI render harness."""

    def test_chat_source_filter_string_present(self):
        import inspect
        from shadow_loom_ui.components import chat

        src = inspect.getsource(chat._build_command_bar)
        assert 'kwargs.get("source") == "query_save"' in src, (
            "chat._clear_chat_on_branch must skip query_save emits"
        )

    def test_answer_panel_source_filter_string_present(self):
        import inspect
        from shadow_loom_ui.components import answer_panel

        src = inspect.getsource(answer_panel.build_answer_panel)
        assert 'kwargs.get("source") == "query_save"' in src, (
            "answer_panel._clear must skip query_save emits"
        )
