# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PR 1: schema + helper tests for ``EventNode.at_location_id``.

Covers:
  * Round-trip serialization of the new optional field.
  * ``event_location_at`` resolution: explicit anchor wins; falls back
    to the primary actor's reconstructed location at fabula_time.
  * Validator warning when an utterance has neither ``via_channel_id``
    nor ``at_location_id`` but does have addressees.
"""

from __future__ import annotations

import logging

import pytest

from shadow_loom.models import (
    Entity,
    EntityStateSnapshot,
    EventNode,
    TraitVector,
    WorldStateV1,
    event_location_at,
)


def _ent(
    eid: str,
    *,
    location: str,
    timeline: list[EntityStateSnapshot] | None = None,
) -> Entity:
    return Entity(
        id=eid,
        name=eid,
        location_id=location,
        status="healthy",
        traits={"x": TraitVector(value=0.5, inertia=0.5)},
        beliefs=[],
        state_timeline=timeline or [],
    )


def _ws(entities: list[Entity], events: list[EventNode]) -> WorldStateV1:
    return WorldStateV1(
        entities={e.id: e for e in entities},
        locations={},
        objects={},
        world_traits={},
        events=events,
        causal_topology=[],
    )


class TestEventNodeSchema:
    def test_at_location_id_defaults_to_none(self):
        evt = EventNode(
            id="EVT_X",
            fabula_time=100,
            syuzhet_index=1,
            event_type="outcome",
            description="d",
        )
        assert evt.at_location_id is None

    def test_round_trip_with_anchor(self):
        evt = EventNode(
            id="EVT_X",
            fabula_time=100,
            syuzhet_index=1,
            event_type="outcome",
            description="d",
            at_location_id="LOC_HALL",
        )
        clone = EventNode.model_validate(evt.model_dump())
        assert clone.at_location_id == "LOC_HALL"


class TestEventLocationAt:
    def test_explicit_anchor_wins(self):
        macbeth = _ent("ENT_M", location="LOC_BEDCHAMBER")
        evt = EventNode(
            id="EVT_E",
            fabula_time=10,
            syuzhet_index=1,
            event_type="outcome",
            actor_ids=["ENT_M"],
            description="d",
            at_location_id="LOC_HALL",
        )
        ws = _ws([macbeth], [evt])
        assert event_location_at(evt, ws) == "LOC_HALL"

    def test_actor_fallback_uses_initial_location(self):
        macbeth = _ent("ENT_M", location="LOC_BEDCHAMBER")
        evt = EventNode(
            id="EVT_E",
            fabula_time=10,
            syuzhet_index=1,
            event_type="outcome",
            actor_ids=["ENT_M"],
            description="d",
        )
        ws = _ws([macbeth], [evt])
        assert event_location_at(evt, ws) == "LOC_BEDCHAMBER"

    def test_actor_fallback_uses_reconstructed_location(self):
        macbeth = _ent(
            "ENT_M",
            location="LOC_BEDCHAMBER",
            timeline=[
                EntityStateSnapshot(fabula_time=5, location_id="LOC_HALL"),
                EntityStateSnapshot(fabula_time=20, location_id="LOC_BATTLEMENT"),
            ],
        )
        evt = EventNode(
            id="EVT_E",
            fabula_time=10,
            syuzhet_index=1,
            event_type="outcome",
            actor_ids=["ENT_M"],
            description="d",
        )
        ws = _ws([macbeth], [evt])
        # At fabula 10 the entity is at LOC_HALL (post snap@5, pre snap@20).
        assert event_location_at(evt, ws) == "LOC_HALL"

    def test_speaker_id_takes_precedence_over_actor_ids(self):
        a = _ent("ENT_A", location="LOC_A")
        b = _ent("ENT_B", location="LOC_B")
        evt = EventNode(
            id="EVT_U",
            fabula_time=1,
            syuzhet_index=1,
            event_type="utterance",
            actor_ids=["ENT_A", "ENT_B"],
            speaker_id="ENT_B",
            addressee_ids=["ENT_A"],
            description="d",
            content="hi",
        )
        ws = _ws([a, b], [evt])
        assert event_location_at(evt, ws) == "LOC_B"

    def test_no_actor_no_anchor_returns_none(self):
        evt = EventNode(
            id="EVT_NAT",
            fabula_time=1,
            syuzhet_index=1,
            event_type="outcome",
            description="storm rolls in",
        )
        ws = _ws([], [evt])
        assert event_location_at(evt, ws) is None

    def test_fallback_none_disables_actor_lookup(self):
        macbeth = _ent("ENT_M", location="LOC_HALL")
        evt = EventNode(
            id="EVT_E",
            fabula_time=10,
            syuzhet_index=1,
            event_type="outcome",
            actor_ids=["ENT_M"],
            description="d",
        )
        ws = _ws([macbeth], [evt])
        assert event_location_at(evt, ws, fallback="none") is None


class TestUtteranceAnchorWarning:
    def test_warns_when_utterance_has_addressees_but_no_anchor(self, caplog):
        # Reset dedupe set so the warning fires for this test.
        EventNode.__dict__.get("_actorless_warn_seen", set()).discard(
            ("utterance_no_anchor", "EVT_TALK_ABC")
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.ingestion"):
            EventNode(
                id="EVT_TALK_ABC",
                fabula_time=1,
                syuzhet_index=1,
                event_type="utterance",
                actor_ids=["ENT_A"],
                speaker_id="ENT_A",
                addressee_ids=["ENT_B"],
                description="A talks to B",
                content="hi",
            )
        assert any(
            "neither via_channel_id nor at_location_id" in rec.message
            for rec in caplog.records
        )

    def test_no_warning_when_anchor_present(self, caplog):
        EventNode.__dict__.get("_actorless_warn_seen", set()).discard(
            ("utterance_no_anchor", "EVT_TALK_OK")
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.ingestion"):
            EventNode(
                id="EVT_TALK_OK",
                fabula_time=1,
                syuzhet_index=1,
                event_type="utterance",
                actor_ids=["ENT_A"],
                speaker_id="ENT_A",
                addressee_ids=["ENT_B"],
                description="A talks to B",
                content="hi",
                at_location_id="LOC_HALL",
            )
        assert not any(
            "neither via_channel_id nor at_location_id" in rec.message
            for rec in caplog.records
        )

    def test_no_warning_when_channel_present(self, caplog):
        EventNode.__dict__.get("_actorless_warn_seen", set()).discard(
            ("utterance_no_anchor", "EVT_TALK_CHN")
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.ingestion"):
            EventNode(
                id="EVT_TALK_CHN",
                fabula_time=1,
                syuzhet_index=1,
                event_type="utterance",
                actor_ids=["ENT_A"],
                speaker_id="ENT_A",
                addressee_ids=["ENT_B"],
                description="A phones B",
                content="hi",
                via_channel_id="CHN_PHONE",
            )
        assert not any(
            "neither via_channel_id nor at_location_id" in rec.message
            for rec in caplog.records
        )
