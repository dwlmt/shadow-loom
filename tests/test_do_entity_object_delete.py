"""Candidate B coverage: ``DoEntityDelete`` and ``DoObjectDelete``.

Excises an ``Entity`` or ``NarrativeObject`` from the world ("never
existed") and cascades references off social topology, beliefs,
concerns, causal edges, channels, and event actor lists.
"""
from __future__ import annotations

import networkx as nx
import pytest

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.models import (
    Belief, CausalEdge, Entity, EventNode, Location, NarrativeObject,
    RelationshipEdge, RelationshipMetric, WorldStateV1,
)
from shadow_loom.query_models import DoEntityDelete, DoObjectDelete


def _ws_with_entities() -> WorldStateV1:
    loc = Location(id="LOC_ROOM", name="Room", description="A room")
    amy = Entity(id="ENT_AMY", name="Amy", location_id="LOC_ROOM", status="healthy", traits={})
    desi = Entity(id="ENT_DESI", name="Desi", location_id="LOC_ROOM", status="healthy", traits={})
    # Belief on Amy targeting Desi (should be dropped).
    amy.beliefs.append(Belief(
        target_id="ENT_DESI",
        perceived_state="suspicious",
        confidence=0.8,
        inertia=0.5,
    ))
    rel = RelationshipEdge(
        source_entity_id="ENT_AMY",
        target_entity_id="ENT_DESI",
        metrics={"affinity": RelationshipMetric(value=-0.5, inertia=0.5)},
    )
    evt = EventNode(
        id="EVT_FIGHT",
        description="Amy and Desi fight",
        fabula_time=1000,
        at_location_id="LOC_ROOM",
        syuzhet_index=0,
        event_type="outcome",
        actor_ids=["ENT_AMY", "ENT_DESI"],
    )
    ce = CausalEdge(
        source_id="ENT_DESI",
        target_id="EVT_FIGHT",
        causality_type="affordance_gate",
        mechanism="presence",
        fabula_time=1000,
    )
    return WorldStateV1(
        locations={"LOC_ROOM": loc},
        entities={"ENT_AMY": amy, "ENT_DESI": desi},
        objects={},
        events=[evt],
        social_topology=[rel],
        causal_topology=[ce],
    )


def test_entity_delete_removes_entity():
    ws = _ws_with_entities()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine._apply_do_entity_delete(DoEntityDelete(entity_id="ENT_DESI"))
    assert "ENT_DESI" not in ws.entities


def test_entity_delete_drops_social_edges():
    ws = _ws_with_entities()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine._apply_do_entity_delete(DoEntityDelete(entity_id="ENT_DESI"))
    assert ws.social_topology == []


def test_entity_delete_drops_beliefs_targeting_deleted():
    ws = _ws_with_entities()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine._apply_do_entity_delete(DoEntityDelete(entity_id="ENT_DESI"))
    assert ws.entities["ENT_AMY"].beliefs == []


def test_entity_delete_scrubs_actor_ids():
    ws = _ws_with_entities()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine._apply_do_entity_delete(DoEntityDelete(entity_id="ENT_DESI"))
    evt = ws.events[0]
    assert "ENT_DESI" not in evt.actor_ids
    assert evt.actor_ids == ["ENT_AMY"]


def test_entity_delete_drops_causal_edges_referencing():
    ws = _ws_with_entities()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine._apply_do_entity_delete(DoEntityDelete(entity_id="ENT_DESI"))
    assert ws.causal_topology == []


def _ws_with_object() -> WorldStateV1:
    loc = Location(id="LOC_VAULT", name="Vault", description="A vault")
    amy = Entity(id="ENT_AMY", name="Amy", location_id="LOC_VAULT", status="healthy", traits={})
    obj = NarrativeObject(
        id="OBJ_DIAMONDS",
        name="Diamonds",
        location_id="LOC_VAULT",
        owner_id=None,
        affordances=[],
    )
    amy.beliefs.append(Belief(
        target_id="OBJ_DIAMONDS",
        perceived_state="in vault",
        confidence=0.9,
        inertia=0.5,
    ))
    evt = EventNode(
        id="EVT_THEFT",
        description="Steal the diamonds",
        fabula_time=500,
        at_location_id="LOC_VAULT",
        syuzhet_index=0,
        event_type="outcome",
        actor_ids=["ENT_AMY"],
    )
    ce = CausalEdge(
        source_id="OBJ_DIAMONDS",
        target_id="EVT_THEFT",
        causality_type="affordance_gate",
        mechanism="presence",
        fabula_time=500,
    )
    return WorldStateV1(
        locations={"LOC_VAULT": loc},
        entities={"ENT_AMY": amy},
        objects={"OBJ_DIAMONDS": obj},
        events=[evt],
        causal_topology=[ce],
    )


def test_object_delete_removes_object_and_cascades_beliefs_edges():
    ws = _ws_with_object()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine._apply_do_object_delete(DoObjectDelete(object_id="OBJ_DIAMONDS"))
    assert "OBJ_DIAMONDS" not in ws.objects
    assert ws.entities["ENT_AMY"].beliefs == []
    assert ws.causal_topology == []


def test_entity_delete_missing_id_is_noop_warning(caplog):
    ws = _ws_with_entities()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine._apply_do_entity_delete(DoEntityDelete(entity_id="ENT_GHOST"))
    assert "ENT_AMY" in ws.entities
    assert "ENT_DESI" in ws.entities
