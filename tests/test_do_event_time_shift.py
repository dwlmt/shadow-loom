"""Candidate A — event-time-shift do-surgery.

`do(EVT_X.new_fabula_time = Y)` rewrites an event's fabula_time and
re-stamps every snapshot triggered by that event so per-axis
`last_updated_fabula` timestamps and `reconstruct_*_at` replay stay
consistent at the new tick. Also re-stamps outgoing causal edges and
the matching social-metric `last_updated_fabula`.

Real-plot motivation: in Gone Girl, asking "what if Amy had killed
Desi later?" requires shifting `EVT_AMY_KILLS_DESI` without removing
it. This test exercises the same surgery on a synthetic world so the
assertions stay narrow.
"""
from __future__ import annotations

import pytest
import networkx as nx

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.models import (
    CausalEdge,
    Entity,
    EntityStateSnapshot,
    EventNode,
    Location,
    RelationshipEdge,
    RelationshipMetric,
    WorldStateV1,
)
from shadow_loom.query_models import DoEvent


def _ws_with_one_event() -> WorldStateV1:
    flat = Location(id="LOC_FLAT", name="The flat", description="A flat")
    amy = Entity(
        id="ENT_AMY",
        name="Amy",
        location_id="LOC_FLAT",
        status="healthy",
        traits={},
        beliefs=[],
        state_timeline=[
            EntityStateSnapshot(
                fabula_time=13000,
                triggered_by="EVT_AMY_KILLS_DESI",
                location_id="LOC_FLAT",
            ),
        ],
        world_id="factual",
    )
    desi = Entity(
        id="ENT_DESI",
        name="Desi",
        location_id="LOC_FLAT",
        status="dead",
        traits={},
        beliefs=[],
        world_id="factual",
    )
    evt = EventNode(
        id="EVT_AMY_KILLS_DESI",
        description="Amy kills Desi",
        fabula_time=13000,
        syuzhet_index=10,
        event_type="outcome",
        at_location_id="LOC_FLAT",
        actor_ids=["ENT_AMY"],
    )
    rel = RelationshipEdge(
        source_entity_id="ENT_AMY",
        target_entity_id="ENT_DESI",
        metrics={
            "affinity": RelationshipMetric(value=-0.9, last_updated_fabula=13000),
            "fear": RelationshipMetric(value=0.0, last_updated_fabula=0),
            "power_dynamic": RelationshipMetric(value=0.8, last_updated_fabula=13000),
        },
        world_id="factual",
    )
    cedge_aff = CausalEdge(
        source_id="EVT_AMY_KILLS_DESI",
        target_id="ENT_AMY",
        rel_counterpart_id="ENT_DESI",
        causality_type="mutation_social",
        trait_target="affinity",
        trait_delta=-0.4,
        mechanism="murder destroys affinity",
        fabula_time=13000,
    )
    return WorldStateV1(
        locations={"LOC_FLAT": flat},
        objects={},
        entities={"ENT_AMY": amy, "ENT_DESI": desi},
        events=[evt],
        propositions=[],
        social_topology=[rel],
        causal_topology=[cedge_aff],
    )


def _run_shift(ws: WorldStateV1, new_ft: int) -> WorldStateV1:
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(
            event_id="EVT_AMY_KILLS_DESI",
            occurred=True,
            new_fabula_time=new_ft,
        ),
    ])
    return ws


def test_event_fabula_time_is_rewritten():
    ws = _run_shift(_ws_with_one_event(), 18000)
    assert ws.events[0].fabula_time == 18000


def test_entity_snapshot_triggered_by_event_is_restamped():
    ws = _run_shift(_ws_with_one_event(), 18000)
    amy = ws.entities["ENT_AMY"]
    assert len(amy.state_timeline) == 1
    snap = amy.state_timeline[0]
    assert snap.fabula_time == 18000
    assert snap.triggered_by == "EVT_AMY_KILLS_DESI"


def test_causal_edge_outgoing_from_shifted_event_is_restamped():
    ws = _run_shift(_ws_with_one_event(), 18000)
    cedge = ws.causal_topology[0]
    assert cedge.source_id == "EVT_AMY_KILLS_DESI"
    assert cedge.fabula_time == 18000


def test_relationship_metric_last_updated_is_restamped_when_owned_by_event():
    ws = _run_shift(_ws_with_one_event(), 18000)
    rel = ws.social_topology[0]
    # affinity was driven by EVT_AMY_KILLS_DESI — should shift.
    assert rel.metrics["affinity"].last_updated_fabula == 18000
    # fear was not touched by the event — should remain at 0.
    assert rel.metrics["fear"].last_updated_fabula == 0


def test_noop_when_new_fabula_time_unset_or_event_did_not_occur():
    ws = _ws_with_one_event()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(event_id="EVT_AMY_KILLS_DESI", occurred=True),  # no new_fabula_time
    ])
    assert ws.events[0].fabula_time == 13000

    # And: occurred=False should suppress the shift even if new_fabula_time set.
    ws2 = _ws_with_one_event()
    engine2 = CausalPhysicsEngine(nx.MultiDiGraph(), ws2)
    engine2.apply_do_targets([
        DoEvent(
            event_id="EVT_AMY_KILLS_DESI",
            occurred=False,
            new_fabula_time=18000,
        ),
    ])
    assert ws2.events[0].fabula_time == 13000
