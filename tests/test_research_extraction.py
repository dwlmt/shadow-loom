# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the optional research-extraction layer.

Covers:
  * ``shadow_loom.research`` provider Protocol + NullProvider + cache_key
  * ``WorldStateV1.world_facts`` is a real, default-empty field
  * ``ExtractionConfig`` research defaults are off / 'none' / empty
  * ``_run_research_step`` is a no-op when disabled
  * ``DirectiveAssembler._select_external_research`` filters by
    ``related_node_ids`` and project-wide facts pass through.
"""

from __future__ import annotations

import pytest

from shadow_loom.directive_assembly import (
    CreativeBrief,
    DirectiveAssembler,
    ResearchHighlight,
)
from shadow_loom.ingestion import ExtractionConfig, _run_research_step
from shadow_loom.models import WorldStateV1
from shadow_loom.research import (
    NullProvider,
    ResearchSnippet,
    WorldFact,
    build_provider,
    cache_key,
)


# ---------------------------------------------------------------------
# Provider layer
# ---------------------------------------------------------------------

def test_cache_key_is_deterministic_and_sensitive_to_inputs() -> None:
    a = cache_key("tavily", "basic", "macbeth")
    b = cache_key("tavily", "basic", "macbeth")
    assert a == b
    assert len(a) == 64  # sha256 hex
    # any input change → different key
    assert cache_key("tavily", "basic", "Macbeth") != a
    assert cache_key("tavily", "advanced", "macbeth") != a
    assert cache_key("none", "basic", "macbeth") != a


def test_null_provider_refuses_to_fabricate() -> None:
    """NullProvider must raise; never silently return [] when called."""
    with pytest.raises(RuntimeError):
        NullProvider().search("anything")


def test_build_provider_none_returns_null_provider() -> None:
    prov = build_provider("none")
    assert isinstance(prov, NullProvider)


def test_build_provider_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_provider("not-a-real-provider")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# WorldStateV1 integration
# ---------------------------------------------------------------------

def test_world_state_has_world_facts_field_default_empty() -> None:
    ws = WorldStateV1(
        locations={}, objects={}, entities={}, events=[], causal_topology=[],
    )
    assert ws.world_facts == []


def test_world_state_accepts_world_facts() -> None:
    fact = WorldFact(
        id="FACT_001",
        topic="t",
        summary="s",
        confidence="moderate",
        source_url_primary="https://example.com",
        provider="tavily",
        related_node_ids=["ENT_001"],
        raw_snippets=[],
    )
    ws = WorldStateV1(
        locations={}, objects={}, entities={}, events=[], causal_topology=[],
        world_facts=[fact],
    )
    assert ws.world_facts[0].id == "FACT_001"


# ---------------------------------------------------------------------
# ExtractionConfig defaults — research is OFF
# ---------------------------------------------------------------------

def test_extraction_config_research_is_disabled_by_default() -> None:
    c = ExtractionConfig()
    assert c.enable_research_agent is False
    assert c.research_provider == "none"
    assert c.research_topics == []
    assert c.research_max_results_per_query == 5


def test_run_research_step_is_noop_when_disabled() -> None:
    """Disabled research must return the world state unchanged — no provider call."""
    ws = WorldStateV1(
        locations={}, objects={}, entities={}, events=[], causal_topology=[],
    )
    config = ExtractionConfig(enable_research_agent=False)
    out = _run_research_step(ws, config)
    assert out is ws
    assert out.world_facts == []


def test_run_research_step_skips_when_no_topics() -> None:
    """Enabled research with no topics must skip cleanly (no provider call)."""
    ws = WorldStateV1(
        locations={}, objects={}, entities={}, events=[], causal_topology=[],
    )
    config = ExtractionConfig(
        enable_research_agent=True,
        research_provider="tavily",  # would raise if it tried to build the provider
        research_topics=[],
    )
    out = _run_research_step(ws, config)
    assert out.world_facts == []


# ---------------------------------------------------------------------
# DirectiveAssembler._select_external_research
# ---------------------------------------------------------------------

def _make_world_with_facts(facts: list[WorldFact]) -> WorldStateV1:
    return WorldStateV1(
        locations={}, objects={}, entities={}, events=[], causal_topology=[],
        world_facts=facts,
    )


def test_select_external_research_filters_by_related_node_ids() -> None:
    fact_match = WorldFact(
        id="FACT_001", topic="t", summary="s", confidence="moderate",
        source_url_primary="", provider="tavily",
        related_node_ids=["ENT_HERO"], raw_snippets=[],
    )
    fact_unrelated = WorldFact(
        id="FACT_002", topic="t2", summary="s2", confidence="moderate",
        source_url_primary="", provider="tavily",
        related_node_ids=["ENT_OTHER"], raw_snippets=[],
    )
    ws = _make_world_with_facts([fact_match, fact_unrelated])
    da = DirectiveAssembler(sandbox=None, ego_payload={}, world_state=ws)
    out = da._select_external_research(["ENT_HERO"])
    assert [h.fact_id for h in out] == ["FACT_001"]


def test_select_external_research_includes_project_wide_facts() -> None:
    """A fact with empty related_node_ids is project-wide and always selected."""
    fact_global = WorldFact(
        id="FACT_GLOBAL", topic="period", summary="...", confidence="moderate",
        source_url_primary="", provider="tavily",
        related_node_ids=[], raw_snippets=[],
    )
    ws = _make_world_with_facts([fact_global])
    da = DirectiveAssembler(sandbox=None, ego_payload={}, world_state=ws)
    out = da._select_external_research(["ENT_HERO"])
    assert [h.fact_id for h in out] == ["FACT_GLOBAL"]


def test_select_external_research_empty_when_no_facts() -> None:
    ws = _make_world_with_facts([])
    da = DirectiveAssembler(sandbox=None, ego_payload={}, world_state=ws)
    assert da._select_external_research(["ENT_HERO"]) == []


# ---------------------------------------------------------------------
# Segregation: research never mutates entities/events/edges
# ---------------------------------------------------------------------

def test_run_research_disabled_never_touches_entities_or_events() -> None:
    """Sanity: with research off, entities/events/edges are bit-identical."""
    ws = WorldStateV1(
        locations={}, objects={}, entities={}, events=[], causal_topology=[],
    )
    before = ws.model_dump_json()
    out = _run_research_step(ws, ExtractionConfig())
    after = out.model_dump_json()
    assert before == after
