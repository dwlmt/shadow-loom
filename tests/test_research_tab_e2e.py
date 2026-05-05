# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end data-flow tests for the Research tab.

These tests exercise the *backend wiring* the Research tab depends on
without standing up a NiceGUI client:

* per-project topic persistence (``get_project_settings`` /
  ``set_project_settings``)
* the cache layer (``get_cached_research`` / ``save_cached_research``)
* fact persistence (``upsert_world_fact`` / ``list_world_facts`` /
  ``delete_world_fact``)
* the orchestrator ``lookup_and_persist_topic`` with a stubbed provider
  + agent so no network or LLM call is made.

The tab itself binds these helpers to widget callbacks; the only
behaviour not covered here is the NiceGUI render path, which is a
thin wrapper validated separately by smoke imports.
"""
from __future__ import annotations

import os

import pytest

os.environ["DATABASE_URL"] = "sqlite://"

from shadow_loom import db, research
from shadow_loom.db import init_db


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db("sqlite://")
    yield


def _seed() -> tuple[int, int]:
    user = db.upsert_user("local", "research-test", "researcher")
    proj = db.create_project(name="Research Test", owner_id=user.id)
    return user.id, proj.id


# =====================================================================
# Topic persistence — chips editor + project settings
# =====================================================================
class TestTopicPersistence:
    def test_get_settings_returns_empty_for_fresh_project(self):
        _, pid = _seed()
        s = db.get_project_settings(pid)
        assert s == {"research_topics": []}

    def test_set_then_get_round_trips_topics(self):
        _, pid = _seed()
        db.set_project_settings(pid, research_topics=["Edinburgh 1040", "Scottish succession"])
        assert db.get_project_settings(pid)["research_topics"] == [
            "Edinburgh 1040", "Scottish succession",
        ]

    def test_set_strips_and_dedupes(self):
        _, pid = _seed()
        db.set_project_settings(
            pid,
            research_topics=["  duplicate ", "duplicate", "", "unique"],
        )
        assert db.get_project_settings(pid)["research_topics"] == [
            "duplicate", "unique",
        ]

    def test_set_rejects_non_string_topic(self):
        _, pid = _seed()
        with pytest.raises(ValueError):
            db.set_project_settings(pid, research_topics=["ok", 42])  # type: ignore[list-item]


# =====================================================================
# Fact CRUD — list / upsert / delete
# =====================================================================
class TestFactCRUD:
    def test_list_world_facts_empty_for_fresh_project(self):
        _, pid = _seed()
        assert db.list_world_facts(pid) == []

    def test_upsert_then_list_returns_fact(self):
        _, pid = _seed()
        db.upsert_world_fact(
            project_id=pid,
            fact_id="FACT_001",
            topic="Edinburgh 1040",
            summary="Seat of Scottish kings.",
            confidence="high",
            source_url_primary="https://example.org/edinburgh",
            provider="tavily",
            related_node_ids_json="[]",
            raw_snippets_json="[]",
        )
        rows = db.list_world_facts(pid)
        assert len(rows) == 1
        assert rows[0].topic == "Edinburgh 1040"
        assert rows[0].confidence == "high"

    def test_upsert_overwrites_existing(self):
        _, pid = _seed()
        kwargs = dict(
            project_id=pid, fact_id="FACT_001", topic="t",
            summary="v1", confidence="low",
            source_url_primary="", provider="tavily",
            related_node_ids_json="[]", raw_snippets_json="[]",
        )
        db.upsert_world_fact(**kwargs)
        kwargs["summary"] = "v2"
        kwargs["confidence"] = "high"
        db.upsert_world_fact(**kwargs)
        rows = db.list_world_facts(pid)
        assert len(rows) == 1
        assert rows[0].summary == "v2"
        assert rows[0].confidence == "high"

    def test_delete_returns_true_when_present(self):
        _, pid = _seed()
        db.upsert_world_fact(
            project_id=pid, fact_id="FACT_001", topic="t",
            summary="s", confidence="moderate",
            source_url_primary="", provider="tavily",
            related_node_ids_json="[]", raw_snippets_json="[]",
        )
        assert db.delete_world_fact(project_id=pid, fact_id="FACT_001") is True
        assert db.list_world_facts(pid) == []

    def test_delete_returns_false_when_absent(self):
        _, pid = _seed()
        assert db.delete_world_fact(project_id=pid, fact_id="FACT_NOPE") is False


# =====================================================================
# lookup_and_persist_topic — full orchestration with stubbed provider
# =====================================================================
class _StubProvider:
    """Returns a single fixed snippet without hitting any network."""

    def search(self, query, *, max_results=5, user_id=None, project_id=None):
        return [
            research.ResearchSnippet(
                title="Stub source",
                url="https://example.org/stub",
                content=f"Stub snippet for {query}",
            )
        ]


class _StubAgentResult:
    def __init__(self, fact):
        self.output = fact


class _StubAgent:
    def __init__(self, fact):
        self._fact = fact

    def run_sync(self, *_args, **_kwargs):
        return _StubAgentResult(self._fact)


class TestLookupAndPersistTopic:
    def _patch(self, monkeypatch, *, provider_name="tavily", api_key="stub-key"):
        # Force the tab to consider the provider usable.
        from shadow_loom.settings import get_settings
        s = get_settings()
        monkeypatch.setattr(s.extraction, "research_provider", provider_name, raising=False)
        monkeypatch.setattr(s.extraction, "enable_research_agent", True, raising=False)
        monkeypatch.setattr(s.core, "tavily_api_key", api_key, raising=False)
        # Replace the provider builder + research agent.
        monkeypatch.setattr(
            research, "build_provider",
            lambda name, **_kw: _StubProvider(),
        )
        fact = research.WorldFact(
            id="ignored",
            topic="ignored",
            summary="Distilled stub fact.",
            confidence="moderate",
            source_url_primary="https://example.org/stub",
            related_node_ids=[],
        )
        from shadow_loom import ingestion
        monkeypatch.setattr(
            ingestion, "_build_research_agent",
            lambda _cfg: _StubAgent(fact),
        )

    def test_empty_topic_returns_error_dict(self, monkeypatch):
        self._patch(monkeypatch)
        uid, pid = _seed()
        out = research.lookup_and_persist_topic(
            project_id=pid, user_id=uid, topic="   ",
        )
        assert "error" in out

    def test_provider_none_returns_error_dict(self, monkeypatch):
        self._patch(monkeypatch, provider_name="none")
        uid, pid = _seed()
        out = research.lookup_and_persist_topic(
            project_id=pid, user_id=uid, topic="anything",
        )
        assert "error" in out
        assert "provider" in out["error"].lower()

    def test_happy_path_persists_fact_and_returns_summary(self, monkeypatch):
        self._patch(monkeypatch)
        uid, pid = _seed()
        out = research.lookup_and_persist_topic(
            project_id=pid, user_id=uid, topic="Edinburgh 1040",
        )
        assert "error" not in out
        assert out["topic"] == "Edinburgh 1040"
        assert out["fact_id"] == "FACT_001"
        assert out["summary"] == "Distilled stub fact."
        assert out["snippet_count"] == 1
        # And the fact is now visible to the tab via list_world_facts.
        rows = db.list_world_facts(pid)
        assert len(rows) == 1
        assert rows[0].fact_id == "FACT_001"

    def test_second_call_uses_cache(self, monkeypatch):
        self._patch(monkeypatch)
        uid, pid = _seed()
        first = research.lookup_and_persist_topic(
            project_id=pid, user_id=uid, topic="Cached topic",
        )
        assert first["cached"] is False
        second = research.lookup_and_persist_topic(
            project_id=pid, user_id=uid, topic="Cached topic",
        )
        assert second["cached"] is True
        # And we now have two FACT rows (each call upserts a new id).
        assert len(db.list_world_facts(pid)) == 2

    def test_provider_failure_is_folded_into_error_dict(self, monkeypatch):
        self._patch(monkeypatch)

        class _BoomProvider:
            def search(self, *args, **kwargs):
                raise RuntimeError("network down")

        monkeypatch.setattr(
            research, "build_provider",
            lambda name, **_kw: _BoomProvider(),
        )
        uid, pid = _seed()
        out = research.lookup_and_persist_topic(
            project_id=pid, user_id=uid, topic="will fail",
        )
        assert "error" in out
        assert "Provider call failed" in out["error"]
        assert db.list_world_facts(pid) == []
