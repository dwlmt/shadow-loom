# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ProjectSettings DB layer + MCP tools.

Covers:
* ``db.get_project_settings`` / ``db.set_project_settings`` roundtrip,
  validation, and de-duplication.
* MCP tools ``get_project_settings``, ``set_project_settings``,
  ``get_research_status`` register and respond correctly in open mode.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["MCP_ALLOW_OPEN_MODE"] = "true"

from shadow_loom import db
from shadow_loom.db import create_project, init_db, upsert_user

init_db("sqlite://")

from shadow_loom_mcp import auth as mcp_auth
from shadow_loom_mcp.server import (
    get_project_settings as mcp_get_project_settings,
    get_research_status as mcp_get_research_status,
    set_project_settings as mcp_set_project_settings,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db("sqlite://")
    mcp_auth._token_user_cache.clear()
    yield
    mcp_auth._token_user_cache.clear()


def _seed() -> tuple[int, int]:
    user = upsert_user("local", "test-1", "tester")
    proj = create_project(name="P", owner_id=user.id)
    mcp_auth._token_user_cache["test-token"] = {
        "user_id": user.id,
        "scopes": {"read", "write", "admin"},
        "key_id": 1,
    }
    return user.id, proj.id


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.request_context = None
    ctx.report_progress = AsyncMock()
    return ctx


# ── DB layer ──────────────────────────────────────────────────────


class TestProjectSettingsDB:
    def test_default_returns_empty_topics(self):
        _, pid = _seed()
        assert db.get_project_settings(pid) == {"research_topics": []}

    def test_roundtrip(self):
        _, pid = _seed()
        db.set_project_settings(pid, research_topics=["Roaring Twenties", "WWII"])
        assert db.get_project_settings(pid) == {
            "research_topics": ["Roaring Twenties", "WWII"],
        }

    def test_strips_and_dedupes(self):
        _, pid = _seed()
        db.set_project_settings(
            pid,
            research_topics=["  WWII ", "WWII", "", "Cold War"],
        )
        assert db.get_project_settings(pid) == {
            "research_topics": ["WWII", "Cold War"],
        }

    def test_clear(self):
        _, pid = _seed()
        db.set_project_settings(pid, research_topics=["X"])
        db.set_project_settings(pid, research_topics=[])
        assert db.get_project_settings(pid) == {"research_topics": []}

    def test_rejects_non_list(self):
        _, pid = _seed()
        with pytest.raises(ValueError):
            db.set_project_settings(pid, research_topics="nope")  # type: ignore[arg-type]

    def test_rejects_non_strings(self):
        _, pid = _seed()
        with pytest.raises(ValueError):
            db.set_project_settings(pid, research_topics=["ok", 42])  # type: ignore[list-item]


# ── MCP tools ─────────────────────────────────────────────────────


class TestMCPProjectSettings:
    def test_get_settings_default(self):
        _, pid = _seed()
        result = mcp_get_project_settings(_ctx(), project_id=pid)
        assert result == {"project_id": pid, "research_topics": []}

    def test_set_then_get(self):
        _, pid = _seed()
        result = mcp_set_project_settings(
            _ctx(), research_topics=["Topic A", "Topic B"], project_id=pid,
        )
        assert result == {
            "project_id": pid,
            "research_topics": ["Topic A", "Topic B"],
        }
        again = mcp_get_project_settings(_ctx(), project_id=pid)
        assert again["research_topics"] == ["Topic A", "Topic B"]

    def test_set_validation_error(self):
        _, pid = _seed()
        result = mcp_set_project_settings(
            _ctx(), research_topics=["ok", 1],  # type: ignore[list-item]
            project_id=pid,
        )
        assert "error" in result


class TestMCPResearchStatus:
    def test_returns_expected_keys(self):
        _seed()
        result = mcp_get_research_status(_ctx())
        for key in (
            "enabled",
            "provider",
            "provider_model",
            "max_results_per_query",
            "api_key_present",
            "default_topics",
        ):
            assert key in result, f"missing key {key!r} in {result!r}"

    def test_api_key_never_leaks(self):
        _seed()
        result = mcp_get_research_status(_ctx())
        # The bool field is allowed; the actual key string must not be returned.
        for v in result.values():
            if isinstance(v, str):
                # Tavily keys begin with 'tvly-'; we just guard against that prefix
                # leaking in any string field.
                assert not v.startswith("tvly-"), (
                    f"API key prefix leaked in field value: {v!r}"
                )
