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
    Belief,
    CausalEdge,
    Concern,
    ConcernSnapshot,
    Entity,
    EntityStateSnapshot,
    EventNode,
    Location,
    Proposition,
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


def _ws_with_committed_proposition() -> WorldStateV1:
    """Synthetic world where the shifted event commits a Proposition's
    truth via ``resolves_proposition_ids``. The proposition's
    ``truth_at_fabula`` ledger is keyed at the event's old fabula tick,
    so any shift must relocate the key in lockstep (sixth-pass F1)."""
    flat = Location(id="LOC_FLAT", name="The flat", description="A flat")
    amy = Entity(
        id="ENT_AMY", name="Amy", location_id="LOC_FLAT", status="healthy",
        traits={}, beliefs=[], world_id="factual",
    )
    desi = Entity(
        id="ENT_DESI", name="Desi", location_id="LOC_FLAT", status="dead",
        traits={}, beliefs=[], world_id="factual",
    )
    evt = EventNode(
        id="EVT_AMY_KILLS_DESI",
        description="Amy kills Desi",
        fabula_time=13000,
        syuzhet_index=10,
        event_type="outcome",
        at_location_id="LOC_FLAT",
        actor_ids=["ENT_AMY"],
        resolves_proposition_ids=["PROP_DESI_DEAD"],
    )
    prop = Proposition(
        proposition_id="PROP_DESI_DEAD",
        kind="outcome",
        description="Desi is dead.",
        truth_at_fabula={13000: True},
        world_id="factual",
    )
    return WorldStateV1(
        locations={"LOC_FLAT": flat},
        objects={},
        entities={"ENT_AMY": amy, "ENT_DESI": desi},
        events=[evt],
        propositions=[prop],
        social_topology=[],
        causal_topology=[],
    )


def test_proposition_truth_ledger_key_is_relocated_when_committing_event_shifts():
    """F1 regression: shifting an event that commits a proposition's
    truth must move the ``truth_at_fabula`` key from the old fabula
    tick to the new one; otherwise the truth ledger anchors the
    commit at a tick when the event no longer occurs."""
    ws = _ws_with_committed_proposition()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(
            event_id="EVT_AMY_KILLS_DESI",
            occurred=True,
            new_fabula_time=18000,
        ),
    ])
    prop = ws.propositions[0]
    assert prop.truth_at_fabula == {18000: True}, (
        f"truth ledger should follow the shifted committing event; "
        f"got {prop.truth_at_fabula}"
    )


def test_proposition_truth_ledger_key_preserved_when_second_committer_anchors_old_tick():
    """If another event at the old tick also commits the same prop,
    keep the old-tick entry intact and copy the value forward."""
    ws = _ws_with_committed_proposition()
    # Second utterance that also commits PROP_DESI_DEAD at the same
    # original tick. After shifting the primary event the ledger
    # should retain {13000: True} (anchored by the second event) AND
    # add {18000: True} from the shifted event.
    ws.events.append(EventNode(
        id="EVT_OBSERVER_REPORTS_DEAD",
        description="Observer reports Desi dead",
        fabula_time=13000,
        syuzhet_index=11,
        event_type="utterance",
        at_location_id="LOC_FLAT",
        actor_ids=["ENT_AMY"],
        asserts_proposition_id="PROP_DESI_DEAD",
    ))
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(
            event_id="EVT_AMY_KILLS_DESI",
            occurred=True,
            new_fabula_time=18000,
        ),
    ])
    prop = ws.propositions[0]
    assert prop.truth_at_fabula == {13000: True, 18000: True}, (
        f"expected both ticks anchored; got {prop.truth_at_fabula}"
    )


def _ws_with_concern_closed_by_event() -> WorldStateV1:
    """Concern whose ``ConcernSnapshot`` closes at the shifted event's
    fabula_time. Mirrors ingestion Phase C auto-close output where
    ``activation_fabula_window=[lo, commit_fab]`` is set with
    ``commit_fab = triggering event.fabula_time``."""
    flat = Location(id="LOC_FLAT", name="The flat", description="A flat")
    amy = Entity(
        id="ENT_AMY",
        name="Amy",
        location_id="LOC_FLAT",
        status="healthy",
        traits={},
        beliefs=[],
        concerns=[
            Concern(
                concern_id="CCN_AMY_KILL_DESI",
                proposition_id="PROP_DESI_DEAD",
                polarity="desire",
                salience=0.8,
                activation_fabula_window=[5000, 20000],
                state_timeline=[
                    ConcernSnapshot(
                        fabula_time=13000,
                        triggered_by="EVT_AMY_KILLS_DESI",
                        salience=0.1,
                        activation_fabula_window=[5000, 13000],
                    ),
                ],
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
        resolves_proposition_ids=["PROP_DESI_DEAD"],
    )
    prop = Proposition(
        proposition_id="PROP_DESI_DEAD",
        kind="outcome",
        description="Desi is dead.",
        truth_at_fabula={13000: True},
        world_id="factual",
    )
    return WorldStateV1(
        locations={"LOC_FLAT": flat},
        objects={},
        entities={"ENT_AMY": amy, "ENT_DESI": desi},
        events=[evt],
        propositions=[prop],
        social_topology=[],
        causal_topology=[],
    )


def test_concern_snapshot_activation_window_endpoint_is_restamped_when_triggering_event_shifts():
    """F4 regression: ``ConcernSnapshot.activation_fabula_window``
    endpoints that equal the shifted event's old fabula_time must
    follow the event to the new tick. Otherwise the closure tick
    dangles and ``reconstruct_concern_at`` reports the concern as
    open at a tick when its triggering event no longer occurs."""
    ws = _ws_with_concern_closed_by_event()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(
            event_id="EVT_AMY_KILLS_DESI",
            occurred=True,
            new_fabula_time=18000,
        ),
    ])
    amy = ws.entities["ENT_AMY"]
    snap = amy.concerns[0].state_timeline[0]
    assert snap.fabula_time == 18000
    assert snap.activation_fabula_window == [5000, 18000], (
        f"closure endpoint should follow shifted event; got "
        f"{snap.activation_fabula_window}"
    )


def test_top_level_belief_established_at_fabula_is_restamped_when_acquiring_event_shifts():
    """F8 regression: ``Entity.beliefs`` written outside a state_timeline
    snapshot (e.g. by ``_apply_do_belief``) carry
    ``established_at_fabula = triggering event.fabula_time``. After a
    DoEventTimeShift on the acquiring event, the standalone belief
    must follow to the new tick or ``reconstruct_entity_at`` orders
    it at a tick when the event no longer occurs."""
    flat = Location(id="LOC_FLAT", name="The flat", description="A flat")
    amy = Entity(
        id="ENT_AMY",
        name="Amy",
        location_id="LOC_FLAT",
        status="healthy",
        traits={},
        beliefs=[
            Belief(
                target_id="EVT_AMY_KILLS_DESI",
                perceived_state="Desi is dead",
                confidence=0.9,
                inertia=0.3,
                evidence_strength="strong",
                acquired_via_event_id="EVT_AMY_KILLS_DESI",
                established_at_fabula=13000,
            ),
        ],
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
    ws = WorldStateV1(
        locations={"LOC_FLAT": flat},
        objects={},
        entities={"ENT_AMY": amy},
        events=[evt],
        propositions=[],
        social_topology=[],
        causal_topology=[],
    )
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(
            event_id="EVT_AMY_KILLS_DESI",
            occurred=True,
            new_fabula_time=18000,
        ),
    ])
    b = ws.entities["ENT_AMY"].beliefs[0]
    assert b.established_at_fabula == 18000, (
        f"top-level belief established_at_fabula should follow the "
        f"shifted acquiring event; got {b.established_at_fabula}"
    )


def test_proposition_tracking_belief_established_at_fabula_is_restamped_when_committing_event_shifts():
    """F6 regression: beliefs synthesised by
    ``_post_pass_synthesize_audience_beliefs`` (or any extractor pass)
    that carry ``proposition_id`` are keyed on the proposition's
    commit tick. When a DoEventTimeShift relocates the prop's truth
    ledger from ``old_ft`` to ``new_ft``, every belief whose
    ``proposition_id`` matches and whose ``established_at_fabula ==
    old_ft`` must follow, or the audience surrogate keeps citing a
    tick when the proposition is no longer committed."""
    ws = _ws_with_committed_proposition()
    # Synthesise an AUDIENCE belief tracking the proposition, keyed
    # on the commit tick (mirrors the ingestion post-pass).
    audience = Entity(
        id="ENT_AUDIENCE", name="Audience", location_id="LOC_FLAT",
        status="healthy", traits={},
        beliefs=[
            Belief(
                target_id="ENT_DESI",
                perceived_state="Desi is dead",
                confidence=0.9,
                inertia=0.5,
                evidence_strength="strong",
                proposition_id="PROP_DESI_DEAD",
                established_at_fabula=13000,
            ),
        ],
        world_id="factual",
    )
    ws.entities["ENT_AUDIENCE"] = audience
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(
            event_id="EVT_AMY_KILLS_DESI",
            occurred=True,
            new_fabula_time=18000,
        ),
    ])
    aud_belief = ws.entities["ENT_AUDIENCE"].beliefs[0]
    assert aud_belief.established_at_fabula == 18000, (
        f"audience belief tracking PROP_DESI_DEAD should follow the "
        f"shifted commit tick; got {aud_belief.established_at_fabula}"
    )
