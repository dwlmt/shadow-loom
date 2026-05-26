# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round-7 audit \u2014 the *sever / mutate* surface of the engine.

Symmetric to the round-6 ``create`` audit. Covers the four defects
identified during the round-7 focused audit:

* **R7-F1** \u2014 ``_apply_do_causal_edge`` (sever) ignored
  ``causality_type``, removing *every* edge between the pair when the
  caller only meant to sever one type. With a Bob\u2192Alice push the
  pair routinely carries both a ``mutation`` arrow (physical harm)
  and a ``mutation_social`` arrow (affinity hit); the original sever
  destroyed both.
* **R7-F2** \u2014 ``_apply_do_spatial_edge`` (sever) was asymmetric: a
  bidirectional connection added by the ``add`` branch registered the
  reverse arrow, but sever only cut the forward arrow \u2014 leaving the
  world traversable B\u2192A after A\u2192B had been cut.
* **R7-F3** \u2014 ``_apply_do_relationship`` spawned a
  ``RelationshipEdge`` even when the endpoints were typos, leaving a
  dangling dyad in ``social_topology``.
* **R7-F4** \u2014 ``_apply_do_object`` did not validate
  ``new_location_id`` / ``new_owner_id`` against the canonical world,
  silently pinning the object at a phantom location/owner.
"""
from __future__ import annotations

import logging

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    Location,
    NarrativeObject,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.query_models import (
    DoCausalEdge,
    DoNarrativeObject,
    DoRelationship,
    DoSpatialEdge,
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
        objects={
            "OBJ_KEY": NarrativeObject(
                id="OBJ_KEY", name="key", location_id="LOC_A",
                owner_id=None, properties={}, affordances=[],
            ),
        },
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.2)},
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB", name="Bob", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.7, inertia=0.3)},
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
            SpatialEdge(source_id="LOC_B", target_id="LOC_A",
                        connection_type="doorway", bidirectional=True),
        ],
        social_topology=[],
        channels={},
    )


def _engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE", "ENT_BOB"])
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    return CausalPhysicsEngine(sandbox, ws)


# ---------------------------------------------------------------------
# R7-F1: sever respects causality_type
# ---------------------------------------------------------------------
class TestDoCausalEdgeSeverTypeFilter:
    def _ws_two_types(self) -> WorldStateV1:
        ws = _make_world()
        ws.causal_topology = [
            CausalEdge(
                source_id="EVT_PUSH", target_id="ENT_ALICE",
                causality_type="mutation", mechanism="physical",
                fabula_time=10, trait_target="courage", trait_delta=-0.2,
            ),
            CausalEdge(
                source_id="EVT_PUSH", target_id="ENT_ALICE",
                causality_type="mutation_social", mechanism="emotional",
                fabula_time=10, trait_target="affinity", trait_delta=-0.4,
                rel_counterpart_id="ENT_BOB",
            ),
        ]
        return ws

    def test_typed_sever_keeps_other_type(self):
        ws = self._ws_two_types()
        eng = _engine(ws)
        eng.apply_do_targets([DoCausalEdge(
            source_id="EVT_PUSH", target_id="ENT_ALICE",
            action="sever", causality_type="mutation",
        )])
        remaining = [e.causality_type for e in ws.causal_topology]
        assert remaining == ["mutation_social"], (
            "sever with causality_type='mutation' must leave the "
            "co-existing mutation_social arrow untouched"
        )

    def test_untyped_sever_removes_all(self):
        # Back-compat: untyped sever still nukes the whole pair.
        ws = self._ws_two_types()
        eng = _engine(ws)
        eng.apply_do_targets([DoCausalEdge(
            source_id="EVT_PUSH", target_id="ENT_ALICE", action="sever",
        )])
        assert ws.causal_topology == []


# ---------------------------------------------------------------------
# R7-F2: bidirectional sever cuts the reverse arrow too
# ---------------------------------------------------------------------
class TestDoSpatialEdgeSeverBidirectional:
    def test_sever_forward_also_severs_reverse(self):
        ws = _make_world()
        # Both directions present \u2014 a typical add(bidirectional=True)
        # pair installed by the engine itself or by ingestion.
        assert any(
            e.source_id == "LOC_B" and e.target_id == "LOC_A"
            for e in ws.spatial_topology
        )
        eng = _engine(ws)
        eng.apply_do_targets([DoSpatialEdge(
            source_id="LOC_A", target_id="LOC_B", action="sever",
        )])
        # Round-4 audit fix: both directions are tombstoned, not
        # filtered out. Assert that every matching edge now carries
        # ``destroyed_at_fabula``.
        matched = [
            e for e in ws.spatial_topology
            if (e.source_id == "LOC_A" and e.target_id == "LOC_B")
            or (e.source_id == "LOC_B" and e.target_id == "LOC_A")
        ]
        assert matched, "severed edges must remain as tombstones"
        assert all(e.destroyed_at_fabula is not None for e in matched)

    def test_sever_unidirectional_does_not_touch_reverse(self):
        # If the world only contained the forward arrow, the reverse
        # cut must NOT fire (there is no reverse arrow to remove and
        # the original intent was strictly directional).
        ws = _make_world()
        ws.spatial_topology = [SpatialEdge(
            source_id="LOC_A", target_id="LOC_B",
            connection_type="doorway", bidirectional=False,
        )]
        eng = _engine(ws)
        eng.apply_do_targets([DoSpatialEdge(
            source_id="LOC_A", target_id="LOC_B", action="sever",
        )])
        # Round-4 audit fix: tombstone-preserve semantics. The
        # forward edge stays in the topology with destroyed_at_fabula
        # set; the reverse cut still must not fire (there was no
        # reverse edge to mark in the first place).
        assert len(ws.spatial_topology) == 1
        survivor = ws.spatial_topology[0]
        assert survivor.source_id == "LOC_A"
        assert survivor.target_id == "LOC_B"
        assert survivor.destroyed_at_fabula is not None


# ---------------------------------------------------------------------
# R7-F3: DoRelationship refuses dangling endpoints
# ---------------------------------------------------------------------
class TestDoRelationshipRefusesDanglingEndpoints:
    def test_typoed_target_does_not_persist(self, caplog):
        ws = _make_world()
        eng = _engine(ws)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
            eng.apply_do_targets([DoRelationship(
                source_entity_id="ENT_ALICE",
                target_entity_id="ENT_GHOST",  # not in world.entities
                metric="affinity", value=-0.7,
            )])
        assert ws.social_topology == [], (
            "world social_topology must not accumulate dangling-endpoint dyads"
        )
        assert any(
            "not in world.entities" in r.getMessage()
            for r in caplog.records
        )

    def test_known_endpoints_still_spawn(self):
        ws = _make_world()
        eng = _engine(ws)
        eng.apply_do_targets([DoRelationship(
            source_entity_id="ENT_ALICE", target_entity_id="ENT_BOB",
            metric="affinity", value=-0.3,
        )])
        assert len(ws.social_topology) == 1
        assert ws.social_topology[0].metrics["affinity"].value == -0.3


# ---------------------------------------------------------------------
# R7-F4: DoNarrativeObject refuses dangling location/owner
# ---------------------------------------------------------------------
class TestDoObjectRefusesDanglingReferences:
    def test_unknown_location_skipped(self, caplog):
        ws = _make_world()
        eng = _engine(ws)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
            eng.apply_do_targets([DoNarrativeObject(
                object_id="OBJ_KEY",
                new_location_id="LOC_GHOST",  # not in world.locations
            )])
        # Object's canonical location is unchanged.
        assert ws.objects["OBJ_KEY"].location_id == "LOC_A"
        assert any(
            "new_location_id" in r.getMessage()
            and "world.locations" in r.getMessage()
            for r in caplog.records
        )

    def test_unknown_owner_skipped(self, caplog):
        ws = _make_world()
        eng = _engine(ws)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
            eng.apply_do_targets([DoNarrativeObject(
                object_id="OBJ_KEY",
                new_owner_id="ENT_GHOST",  # not in world.entities
            )])
        assert ws.objects["OBJ_KEY"].owner_id is None
        assert any(
            "new_owner_id" in r.getMessage()
            and "world.entities" in r.getMessage()
            for r in caplog.records
        )

    def test_known_location_still_applies(self):
        ws = _make_world()
        eng = _engine(ws)
        eng.apply_do_targets([DoNarrativeObject(
            object_id="OBJ_KEY", new_location_id="LOC_B",
        )])
        assert ws.objects["OBJ_KEY"].location_id == "LOC_B"
