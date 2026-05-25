# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the MCP `run_and_save` helper activates the per-user model
override ContextVar before invoking `run_pipeline`. Mirrors the wiring
already covered for the UI in `test_user_model_settings.py`."""

from __future__ import annotations

import os
import tempfile

import pytest

from shadow_loom.models import Entity, EventNode, Location, WorldStateV1
from shadow_loom.query_models import GeneralQuery

# IMPORTANT: import the MCP helpers module at top level so that the
# ``shadow_loom_mcp`` package __init__ (which imports server.py and
# unconditionally calls ``init_db("sqlite:///shadow_loom.db")``) runs
# BEFORE the ``_isolated_db`` fixture below points the engine at a
# per-test sqlite file. Otherwise the first call into ``helpers``
# inside a test body silently re-inits the engine and orphans the
# test's seeded rows.
import shadow_loom_mcp.helpers as _mcp_helpers_preload  # noqa: F401


def _empty_world_state() -> WorldStateV1:
    return WorldStateV1(
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_X",
                status="healthy", traits={},
            ),
        },
        events=[
            EventNode(
                id="EVT_1", fabula_time=0, syuzhet_index=0,
                event_type="outcome", actor_ids=["ENT_A"], target_ids=[],
                description="x",
            ),
        ],
        causal_topology=[],
    )


def _dummy_query() -> GeneralQuery:
    return GeneralQuery(original_query="dummy", question="dummy?")


@pytest.fixture
def _isolated_db(monkeypatch):
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    import shadow_loom.db as _db
    monkeypatch.setattr(_db, "_engine", None)
    _db.init_db(f"sqlite:///{db_file}")
    yield _db
    try:
        os.unlink(db_file)
    except FileNotFoundError:
        pass


def test_mcp_run_and_save_activates_user_context(
    _isolated_db, monkeypatch,
):
    """When `run_and_save` is called with a `user_row_id`, the
    per-user model override ContextVar must be set before
    `run_pipeline` is invoked, so any stage that reads
    `get_user_overrides()` inside the pipeline sees the user's prefs."""
    # Arrange: a user with a recognisable default-model override.
    u = _isolated_db.upsert_user("test", "test:mcp:1", "mcp_user")
    _isolated_db.set_user_model_settings(
        u.id, default_model="openrouter:meta-llama/llama-3-70b",
    )
    # Sanity: confirm row landed before we hand control to helpers.
    pre = _isolated_db.get_user_model_settings(u.id)
    assert pre["default_model"] == "openrouter:meta-llama/llama-3-70b", \
        f"DB precondition failed: {pre}"

    # Monkeypatch `run_pipeline` (imported into helpers' namespace)
    # to capture the user-override snapshot that was active at the
    # moment it was called.
    from shadow_loom.settings import get_user_overrides
    import shadow_loom_mcp.helpers as helpers

    captured: dict = {}

    def _fake_run_pipeline(*_args, **_kwargs):
        captured["overrides"] = dict(get_user_overrides())
        # Return a minimal stub that satisfies the post-pipeline
        # save/response code path below. We deliberately raise so we
        # short-circuit before the save_version() call (it would need
        # a real WorldStateV1 round-trip we don't care about here).
        raise RuntimeError("stop-after-context-capture")

    monkeypatch.setattr(helpers, "run_pipeline", _fake_run_pipeline)

    # Build minimal inputs.
    ws = _empty_world_state()
    query = _dummy_query()

    # Act
    resp = helpers.run_and_save(
        query=query,
        project_id=0,            # never used (we short-circuit)
        world_state=ws,
        ancestor_row_id=None,
        user_row_id=u.id,
        raw_query="dummy",
        skip_audit=True,
        skip_reextraction=True,
    )

    # Assert: helpers caught the RuntimeError and returned an error dict
    assert isinstance(resp, dict)
    assert "error" in resp
    # And, critically: the user override was active.
    assert captured["overrides"]["default_model"] == \
        "openrouter:meta-llama/llama-3-70b", (
            f"Expected user default to be active inside run_pipeline; "
            f"got {captured.get('overrides')!r}"
        )


def test_mcp_run_and_save_with_none_user_clears_context(
    _isolated_db, monkeypatch,
):
    """Calling with `user_row_id=None` (e.g. unauthenticated tool) must
    not leak a prior user's overrides — the resolver should fall back
    to env defaults."""
    # Seed an unrelated user just so the table isn't empty.
    _isolated_db.upsert_user("test", "test:mcp:2", "other_user")

    import shadow_loom_mcp.helpers as helpers
    from shadow_loom.settings import get_user_overrides, set_user_context

    # Simulate a previous request having set a non-empty context.
    set_user_context(12345)

    captured: dict = {}

    def _fake_run_pipeline(*_args, **_kwargs):
        captured["overrides"] = dict(get_user_overrides())
        raise RuntimeError("stop")

    monkeypatch.setattr(helpers, "run_pipeline", _fake_run_pipeline)

    helpers.run_and_save(
        query=_dummy_query(),
        project_id=0,
        world_state=_empty_world_state(),
        ancestor_row_id=None,
        user_row_id=None,
        raw_query="x",
        skip_audit=True,
        skip_reextraction=True,
    )

    # Setting context to None should produce an empty override snapshot.
    assert captured["overrides"]["default_model"] == ""
    assert captured["overrides"]["stage_models"] == {}
    assert captured["overrides"]["custom_providers"] == []
