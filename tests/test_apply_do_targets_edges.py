# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Edge-layer typed do-targets and introduction-spec coverage.

Covers:
  * DoChannel — disable / re-enable / re-tune intelligibility
  * DoRelationship — clamp affinity / fear / power_dynamic, including
    spawn-on-missing-edge
  * DoCausalEdge — add and sever
  * DoSpatialEdge — add, sever, lock, unlock
  * IntroducedChannelSpec — pre-spawn from query.introduce.channels
  * IntroducedEventSpec — pre-spawn from query.introduce.events
  * Concern-introduction polarity translation regression (positive →
    desire, negative → fear) — previously fell through into the silent
    try/except because the canonical ``Concern`` model rejects the
    ``aversion`` polarity emitted by the legacy mapper.
"""
from __future__ import annotations

import pytest

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, Channel, Proposition,
    TraitVector, RelationshipEdge, RelationshipMetric, SpatialEdge,
)
from shadow_loom.extract_graph import (
    extract_ego_graph_from_memory,
    introduced_elements_to_spawns,
)
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.query_models import (
    DoChannel, DoRelationship, DoCausalEdge, DoSpatialEdge,
)
from shadow_loom.introduced_elements import (
    IntroducedElements,
    IntroducedChannelSpec, IntroducedEventSpec, IntroducedConcernSpec,
    IntroducedPropositionSpec,
)


def _make_world() -> WorldStateV1:
    return WorldStateV1(
        locations={
            "LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={}),
            "LOC_B": Location(
                id="LOC_B",
                name="B", description="B", ambient_state={}),
        },
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.2)},
                beliefs=[],
                concerns=[],
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB", name="Bob", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.7, inertia=0.3)},
                beliefs=[],
                concerns=[],
            ),
        },
        events=[
            EventNode(
                id="EVT_PUSH", fabula_time=10, syuzhet_index=10,
                event_type="choice", actor_ids=["ENT_BOB"],
                target_ids=["ENT_ALICE"], description="Bob shoves Alice.",
            ),
        ],
        causal_topology=[],
        spatial_topology=[
            SpatialEdge(source_id="LOC_A", target_id="LOC_B",
                        connection_type="doorway", bidirectional=True),
        ],
        social_topology=[],
        channels={
            "CHN_PIPELINE": Channel(
                id="CHN_PIPELINE", name="pipeline", medium="courier",
                participant_ids=["ENT_ALICE", "ENT_BOB"],
                directionality="duplex",
                intelligibility={},
            ),
        },
    )


def _build_engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE", "ENT_BOB"])
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    return CausalPhysicsEngine(sandbox, ws)


class TestDoChannel:
    def test_severs_active_channel(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([DoChannel(
            channel_id="CHN_PIPELINE", active=False, fabula_time=12,
        )])
        assert ws.channels["CHN_PIPELINE"].terminated_at_fabula == 12

    def test_reenables_severed_channel(self):
        ws = _make_world()
        ws.channels["CHN_PIPELINE"].terminated_at_fabula = 5
        eng = _build_engine(ws)
        eng.apply_do_targets([DoChannel(channel_id="CHN_PIPELINE", active=True)])
        assert ws.channels["CHN_PIPELINE"].terminated_at_fabula is None

    def test_intelligibility_override(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([DoChannel(
            channel_id="CHN_PIPELINE",
            intelligibility={"ENT_ALICE": 0.0},
        )])
        assert ws.channels["CHN_PIPELINE"].intelligibility["ENT_ALICE"] == 0.0


class TestDoRelationship:
    def test_spawns_relationship_when_missing(self):
        ws = _make_world()
        assert ws.social_topology == []
        eng = _build_engine(ws)
        eng.apply_do_targets([DoRelationship(
            source_entity_id="ENT_ALICE", target_entity_id="ENT_BOB",
            metric="affinity", value=-0.8,
        )])
        assert len(ws.social_topology) == 1
        rel = ws.social_topology[0]
        assert rel.metrics["affinity"].value == -0.8

    def test_clamps_existing_metric(self):
        ws = _make_world()
        ws.social_topology = [RelationshipEdge(
            source_entity_id="ENT_ALICE", target_entity_id="ENT_BOB",
            metrics={"affinity": RelationshipMetric(value=0.4)},
        )]
        eng = _build_engine(ws)
        eng.apply_do_targets([DoRelationship(
            source_entity_id="ENT_ALICE", target_entity_id="ENT_BOB",
            metric="affinity", value=-0.5,
        )])
        assert ws.social_topology[0].metrics["affinity"].value == -0.5


class TestDoCausalEdge:
    def test_add_creates_world_edge(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([DoCausalEdge(
            source_id="EVT_PUSH", target_id="ENT_ALICE",
            action="add",
            causality_type="mutation",
            mechanism="physical",
            trait_target="courage",
            trait_delta=-0.2,
            fabula_time=10,
        )])
        added = [e for e in ws.causal_topology
                 if e.source_id == "EVT_PUSH" and e.target_id == "ENT_ALICE"]
        assert len(added) == 1
        assert added[0].mechanism == "physical"

    def test_sever_removes_matching(self):
        ws = _make_world()
        from shadow_loom.models import CausalEdge
        ws.causal_topology = [CausalEdge(
            source_id="EVT_PUSH", target_id="ENT_ALICE",
            causality_type="mutation", mechanism="physical",
            fabula_time=10, trait_target="courage", trait_delta=-0.2,
        )]
        eng = _build_engine(ws)
        eng.apply_do_targets([DoCausalEdge(
            source_id="EVT_PUSH", target_id="ENT_ALICE", action="sever",
        )])
        assert ws.causal_topology == []


class TestDoSpatialEdge:
    def test_lock(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([DoSpatialEdge(
            source_id="LOC_A", target_id="LOC_B", action="lock",
        )])
        assert ws.spatial_topology[0].is_locked is True

    def test_unlock(self):
        ws = _make_world()
        ws.spatial_topology[0].is_locked = True
        eng = _build_engine(ws)
        eng.apply_do_targets([DoSpatialEdge(
            source_id="LOC_A", target_id="LOC_B", action="unlock",
        )])
        assert ws.spatial_topology[0].is_locked is False

    def test_sever_removes_world_edge(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([DoSpatialEdge(
            source_id="LOC_A", target_id="LOC_B", action="sever",
        )])
        # Round-4 audit fix: severed edges are tombstoned
        # (``destroyed_at_fabula`` set) rather than removed, so the
        # extraction / instantiator layers can still reason about
        # the historical passage.
        survivors = [
            e for e in ws.spatial_topology
            if e.source_id == "LOC_A" and e.target_id == "LOC_B"
        ]
        assert len(survivors) == 1
        assert survivors[0].destroyed_at_fabula is not None

    def test_add_appends_world_edge(self):
        ws = _make_world()
        ws.locations["LOC_C"] = Location(id="LOC_C", name="C", description="C", ambient_state={})
        eng = _build_engine(ws)
        eng.apply_do_targets([DoSpatialEdge(
            source_id="LOC_B", target_id="LOC_C", action="add",
            connection_type="passage",
        )])
        added = [e for e in ws.spatial_topology
                 if e.source_id == "LOC_B" and e.target_id == "LOC_C"]
        assert len(added) == 1


class TestIntroducedSpecs:
    def test_channel_spec_materialises(self):
        ws = _make_world()
        intro = IntroducedElements(channels=[IntroducedChannelSpec(
            id="CHN_NEW", name="back-channel",
            justification="needed for the wiretap intervention",
            medium="telephone",
            participant_ids=["ENT_ALICE", "ENT_BOB"],
        )])
        spawns = introduced_elements_to_spawns(intro, ws)
        assert "CHN_NEW" in spawns["channels"]
        assert spawns["channels"]["CHN_NEW"].medium == "telephone"

    def test_event_spec_materialises(self):
        ws = _make_world()
        intro = IntroducedElements(events=[IntroducedEventSpec(
            id="EVT_NEW", name="off-page revelation",
            justification="user-asserted historical occurrence",
            fabula_time=5, syuzhet_index=5,
            event_type="revelation",
            description="A previously untold revelation about Alice.",
        )])
        spawns = introduced_elements_to_spawns(intro, ws)
        assert "EVT_NEW" in spawns["events"]
        assert spawns["events"]["EVT_NEW"].event_type == "revelation"

    def test_concern_spec_polarity_no_longer_silently_dropped(self):
        """Regression: ``polarity='positive'`` used to be translated to
        ``aversion`` which is rejected by the canonical Concern model;
        the validation error was silently swallowed by the surrounding
        try/except, so introduced concerns never reached the world.
        """
        ws = _make_world()
        intro = IntroducedElements(
            propositions=[IntroducedPropositionSpec(
                id="PROP_X", name="Alice escapes",
                justification="anchor for the new concern",
            )],
            concerns=[IntroducedConcernSpec(
                id="CCN_NEW", name="alice-wants-out",
                justification="motivates the intervention",
                holder_entity_id="ENT_ALICE",
                proposition_id="PROP_X",
                polarity="positive",
                salience=0.8,
            )],
        )
        spawns = introduced_elements_to_spawns(intro, ws)
        attached = spawns["concerns"].get("ENT_ALICE") or []
        assert len(attached) == 1
        assert attached[0].polarity == "desire"
