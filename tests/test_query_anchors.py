# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for per-query story-point anchors.

Covers the new ``temporal_anchor`` / ``syuzhet_anchor`` /
``anchor_after_event_id`` fields on ``_QueryBase`` and the
``pipeline._resolve_query_anchors`` helper that decides which value
takes precedence at pipeline entry.
"""

from __future__ import annotations

import pytest

from shadow_loom.models import EventNode, WorldStateV1
from shadow_loom.pipeline import _resolve_query_anchors
from shadow_loom.query_models import (
    DirectiveQuery,
    GeneralQuery,
    InterventionQuery,
    ObservationQuery,
)


@pytest.fixture
def world_with_event() -> WorldStateV1:
    return WorldStateV1(
        entities={},
        locations={},
        objects={},
        world_traits={},
        events=[
            EventNode(
                id="EVT_BANQUO_DEATH",
                fabula_time=4500,
                syuzhet_index=12,
                event_type="outcome",
                description="Banquo is murdered.",
            ),
            EventNode(
                id="EVT_BANQUET",
                fabula_time=5200,
                syuzhet_index=14,
                event_type="outcome",
                description="The banquet is held.",
            ),
        ],
        causal_topology=[],
    )


class TestQueryBaseFields:
    """The new anchor fields exist on every query type."""

    def test_directive_accepts_anchors(self):
        q = DirectiveQuery(
            target_entity_ids=["ENT_M"],
            target_effect="suspense",
            temporal_anchor=4500,
            syuzhet_anchor=12,
            anchor_after_event_id="EVT_BANQUO_DEATH",
        )
        assert q.temporal_anchor == 4500
        assert q.syuzhet_anchor == 12
        assert q.anchor_after_event_id == "EVT_BANQUO_DEATH"

    @pytest.mark.parametrize("cls,kwargs", [
        (ObservationQuery, {}),
        (InterventionQuery, {"interventions": {"ENT_X.status": "dead"}}),
        (GeneralQuery, {"question": "?"}),
    ])
    def test_other_types_accept_anchors(self, cls, kwargs):
        q = cls(temporal_anchor=999, **kwargs)
        assert q.temporal_anchor == 999
        # Anchors default to None when not provided.
        q2 = cls(**kwargs)
        assert q2.temporal_anchor is None
        assert q2.syuzhet_anchor is None
        assert q2.anchor_after_event_id is None


class TestResolveQueryAnchors:
    """Precedence: explicit query anchors > anchor_after_event_id > config."""

    def test_falls_back_to_config_when_query_silent(self, world_with_event):
        q = DirectiveQuery(target_entity_ids=["ENT_M"], target_effect="suspense")
        t, s = _resolve_query_anchors(q, 1000, 5, world_with_event)
        assert t == 1000
        assert s == 5

    def test_explicit_query_anchor_overrides_config(self, world_with_event):
        q = DirectiveQuery(
            target_entity_ids=["ENT_M"], target_effect="suspense",
            temporal_anchor=7777, syuzhet_anchor=20,
        )
        t, s = _resolve_query_anchors(q, 1000, 5, world_with_event)
        assert t == 7777
        assert s == 20

    def test_anchor_after_event_resolves_to_event_times(self, world_with_event):
        q = DirectiveQuery(
            target_entity_ids=["ENT_M"], target_effect="suspense",
            anchor_after_event_id="EVT_BANQUO_DEATH",
        )
        t, s = _resolve_query_anchors(q, None, None, world_with_event)
        assert t == 4500
        assert s == 12

    def test_explicit_temporal_wins_over_after_event(self, world_with_event):
        q = DirectiveQuery(
            target_entity_ids=["ENT_M"], target_effect="suspense",
            temporal_anchor=9999,
            anchor_after_event_id="EVT_BANQUO_DEATH",
        )
        t, s = _resolve_query_anchors(q, None, None, world_with_event)
        assert t == 9999
        # syuzhet wasn't set explicitly, so the event fills it.
        assert s == 12

    def test_unknown_after_event_falls_back_to_config(self, world_with_event, caplog):
        q = DirectiveQuery(
            target_entity_ids=["ENT_M"], target_effect="suspense",
            anchor_after_event_id="EVT_DOES_NOT_EXIST",
        )
        t, s = _resolve_query_anchors(q, 100, 1, world_with_event)
        assert t == 100
        assert s == 1


class TestQueryParserPlumbing:
    """The query-parser's ``_build_query`` propagates anchors from a
    ``ParsedQuery`` onto the resulting concrete query model."""

    def test_build_query_threads_anchors(self):
        from shadow_loom.query_parsing import ParsedQuery, _build_query

        parsed = ParsedQuery(
            query_type="directive",
            reasoning="t",
            target_entity_ids=["ENT_M"],
            target_effect="suspense",
            temporal_anchor=4500,
            syuzhet_anchor=12,
            anchor_after_event_id="EVT_BANQUO_DEATH",
        )
        q = _build_query(parsed, natural_language="after Banquo dies, build dread")
        assert isinstance(q, DirectiveQuery)
        assert q.temporal_anchor == 4500
        assert q.syuzhet_anchor == 12
        assert q.anchor_after_event_id == "EVT_BANQUO_DEATH"
        assert q.original_query == "after Banquo dies, build dread"

    def test_build_query_threads_anchors_for_observation(self):
        from shadow_loom.query_parsing import ParsedQuery, _build_query

        parsed = ParsedQuery(
            query_type="observation",
            reasoning="t",
            anchor_after_event_id="EVT_BANQUET",
        )
        q = _build_query(parsed)
        assert isinstance(q, ObservationQuery)
        assert q.anchor_after_event_id == "EVT_BANQUET"
        assert q.temporal_anchor is None
