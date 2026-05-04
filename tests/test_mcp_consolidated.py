# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the consolidated MCP dispatchers.

These verify that ``discover`` / ``trace`` / ``author`` / ``manage``
correctly delegate to the underlying granular tool functions. The
granular tools have their own coverage in ``test_mcp_server.py``;
this file just checks the routing layer.
"""

from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["MCP_ALLOW_OPEN_MODE"] = "true"

from shadow_loom.db import (
    create_project,
    init_db,
    save_version,
    upsert_user,
)
from shadow_loom.models import WorldStateV1

init_db("sqlite://")

from shadow_loom_mcp import auth as mcp_auth
from shadow_loom_mcp.server import (
    author,
    discover,
    manage,
    trace,
)
from example_worlds.macbeth import world_state as macbeth_ws


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db("sqlite://")
    mcp_auth._token_user_cache.clear()
    yield
    mcp_auth._token_user_cache.clear()


def _seed() -> tuple[int, int, int]:
    user = upsert_user("local", "test-1", "tester")
    proj = create_project(name="Macbeth", owner_id=user.id)
    ws = deepcopy(macbeth_ws)
    ver = save_version(
        project_id=proj.id,
        world_state_json=ws.model_dump_json(),
        version=0,
        source="ingestion",
        user_id=user.id,
    )
    mcp_auth._token_user_cache["test-token"] = {
        "user_id": user.id,
        "scopes": {"read", "write", "admin"},
        "key_id": 1,
    }
    return user.id, proj.id, ver.id


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.request_context = None
    ctx.report_progress = AsyncMock()
    return ctx


# ── discover ──────────────────────────────────────────────────────


class TestDiscover:
    def test_projects(self):
        _seed()
        result = discover(_ctx(), scope="projects")
        assert "projects" in result

    def test_world_facts_empty(self):
        _, pid, _ = _seed()
        result = discover(_ctx(), scope="world_facts", project_id=pid)
        assert result == {"project_id": pid, "facts": [], "count": 0}

    def test_branches(self):
        _, pid, _ = _seed()
        result = discover(_ctx(), scope="branches", project_id=pid)
        # Just a smoke check — branches structure verified in test_mcp_server.
        assert "error" not in result or result.get("branches") is not None

    def test_channels(self):
        _, pid, _ = _seed()
        result = discover(_ctx(), scope="channels", project_id=pid)
        assert "error" not in result

    def test_unknown_scope(self):
        _seed()
        result = discover(_ctx(), scope="bogus")
        assert "error" in result
        assert "bogus" in result["error"]


# ── trace ─────────────────────────────────────────────────────────


class TestTrace:
    def test_history_full_tree(self):
        _, pid, _ = _seed()
        result = trace(_ctx(), kind="history", project_id=pid)
        assert "versions" in result

    def test_causal_requires_node_id(self):
        _, pid, _ = _seed()
        result = trace(_ctx(), kind="causal", project_id=pid, payload={})
        assert "error" in result
        assert "node_id" in result["error"]

    def test_channel_requires_channel_id(self):
        _, pid, _ = _seed()
        result = trace(_ctx(), kind="channel", project_id=pid, payload={})
        assert "error" in result
        assert "channel_id" in result["error"]

    def test_unknown_kind(self):
        _, pid, _ = _seed()
        result = trace(_ctx(), kind="bogus", project_id=pid)
        assert "error" in result


# ── author ────────────────────────────────────────────────────────


class TestAuthor:
    def test_research_requires_topic(self):
        _, pid, _ = _seed()
        result = asyncio.run(
            author(_ctx(), action="research", project_id=pid, payload={})
        )
        assert "error" in result
        assert "topic" in result["error"]

    def test_forget_fact_requires_fact_id(self):
        _, pid, _ = _seed()
        result = asyncio.run(
            author(_ctx(), action="forget_fact", project_id=pid, payload={})
        )
        assert "error" in result
        assert "fact_id" in result["error"]

    def test_edit_requires_prose(self):
        _, pid, _ = _seed()
        result = asyncio.run(
            author(_ctx(), action="edit", project_id=pid, payload={})
        )
        assert "error" in result
        assert "prose" in result["error"]

    def test_ingest_requires_text(self):
        _seed()
        result = asyncio.run(author(_ctx(), action="ingest", payload={}))
        assert "error" in result
        assert "text" in result["error"]

    def test_unknown_action(self):
        _seed()
        result = asyncio.run(author(_ctx(), action="bogus", payload={}))
        assert "error" in result


# ── manage ────────────────────────────────────────────────────────


class TestManage:
    def test_research_status_no_project_required(self):
        _seed()
        result = manage(_ctx(), action="research_status")
        for key in ("enabled", "provider", "api_key_present"):
            assert key in result

    def test_get_settings_default(self):
        _, pid, _ = _seed()
        result = manage(_ctx(), action="get_settings", project_id=pid)
        assert result == {"project_id": pid, "research_topics": []}

    def test_set_settings_then_get(self):
        _, pid, _ = _seed()
        result = manage(
            _ctx(),
            action="set_settings",
            project_id=pid,
            payload={"research_topics": ["A", "B"]},
        )
        assert result["research_topics"] == ["A", "B"]

    def test_set_settings_requires_payload(self):
        _, pid, _ = _seed()
        result = manage(_ctx(), action="set_settings", project_id=pid, payload={})
        assert "error" in result
        assert "research_topics" in result["error"]

    def test_get_active_version_requires_project(self):
        _seed()
        result = manage(_ctx(), action="get_active_version")
        assert "error" in result

    def test_branch_requires_from_version(self):
        _, pid, _ = _seed()
        result = manage(_ctx(), action="branch", project_id=pid, payload={})
        assert "error" in result

    def test_unknown_action(self):
        _seed()
        result = manage(_ctx(), action="bogus")
        assert "error" in result
