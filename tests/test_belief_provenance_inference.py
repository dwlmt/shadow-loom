# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression coverage for ``_auto_repair``'s deterministic belief
provenance inference.

When a belief has neither ``acquired_via_event_id`` nor
``acquired_via_channel_id`` set, ``_auto_repair`` now backfills both
fields from the unique utterance event that plausibly produced the
belief. Conservative: skips whenever there is ambiguity (zero or >=2
candidates) so we never invent a false causal edge that abduction /
counterfactual rollback would later rely on.

Implemented in ``shadow_loom/ingestion.py`` ~L13708 onward (closure
``_infer_belief_provenance`` inside ``_auto_repair``).
"""

from __future__ import annotations

from shadow_loom.ingestion import _auto_repair
from shadow_loom.models import (
    Belief,
    Channel,
    Entity,
    EventNode,
    Location,
    WorldStateV1,
)


def _world_with_utterance(
    *,
    speakers_addressees_targets,
    via_channel_id=None,
) -> WorldStateV1:
    """Build a minimal world with one entity per actor plus the
    listed utterance event(s). ``speakers_addressees_targets`` is a
    list of ``(evt_id, speaker, addressees, targets, fabula_time)``
    tuples."""
    ws = WorldStateV1(
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        objects={},
        entities={},
        events=[],
        causal_topology=[],
    )
    ent_ids = set()
    for _, sp, addrs, tgts, _ft in speakers_addressees_targets:
        ent_ids.add(sp)
        for a in addrs:
            ent_ids.add(a)
        for t in tgts:
            ent_ids.add(t)
    for eid in ent_ids:
        ws.entities[eid] = Entity(
            id=eid, name=eid.split("_", 1)[-1].title(),
            location_id="LOC_X", status="healthy", traits={},
        )
    for evt_id, sp, addrs, tgts, ft in speakers_addressees_targets:
        ws.events.append(EventNode(
            id=evt_id, fabula_time=ft, syuzhet_index=ft,
            event_type="utterance",
            speaker_id=sp, addressee_ids=list(addrs),
            actor_ids=[sp], target_ids=list(tgts),
            description=f"{sp} tells {addrs} about {tgts}",
            via_channel_id=via_channel_id,
        ))
    if via_channel_id:
        ws.channels = {via_channel_id: Channel(
            id=via_channel_id,
            name=via_channel_id,
            medium="speech",
            participant_ids=sorted(ent_ids),
        )}
    return ws


def _get_belief(ws: WorldStateV1, holder: str, target: str) -> Belief:
    for b in ws.entities[holder].beliefs:
        if b.target_id == target:
            return b
    raise AssertionError(f"no belief on {holder} about {target}")


# ---------------------------------------------------------------------
# Positive cases: single matching utterance -> provenance inferred
# ---------------------------------------------------------------------


def test_single_matching_utterance_backfills_event_id():
    ws = _world_with_utterance(speakers_addressees_targets=[
        ("EVT_TELL", "ENT_ALICE", ["ENT_BOB"], ["ENT_CARL"], 10),
    ])
    ws.entities["ENT_BOB"].beliefs = [Belief(
        target_id="ENT_CARL", perceived_state="Carl is alive",
        confidence=0.7, inertia=0.5,
    )]
    repaired, _ = _auto_repair(ws)
    b = _get_belief(repaired, "ENT_BOB", "ENT_CARL")
    assert b.acquired_via_event_id == "EVT_TELL"


def test_single_matching_utterance_with_channel_backfills_both():
    ws = _world_with_utterance(
        speakers_addressees_targets=[
            ("EVT_TELL", "ENT_ALICE", ["ENT_BOB"], ["ENT_CARL"], 10),
        ],
        via_channel_id="CHN_RADIO",
    )
    ws.entities["ENT_BOB"].beliefs = [Belief(
        target_id="ENT_CARL", perceived_state="Carl is alive",
        confidence=0.7, inertia=0.5,
    )]
    repaired, _ = _auto_repair(ws)
    b = _get_belief(repaired, "ENT_BOB", "ENT_CARL")
    assert b.acquired_via_event_id == "EVT_TELL"
    assert b.acquired_via_channel_id == "CHN_RADIO"


def test_speaker_as_holder_direct_witness_backfills():
    """Direct-witness belief: holder IS the speaker. The candidate
    function accepts this as a valid match."""
    ws = _world_with_utterance(speakers_addressees_targets=[
        ("EVT_SAY", "ENT_ALICE", ["ENT_BOB"], ["ENT_CARL"], 10),
    ])
    ws.entities["ENT_ALICE"].beliefs = [Belief(
        target_id="ENT_CARL", perceived_state="Carl matters",
        confidence=0.8, inertia=0.5,
    )]
    repaired, _ = _auto_repair(ws)
    b = _get_belief(repaired, "ENT_ALICE", "ENT_CARL")
    assert b.acquired_via_event_id == "EVT_SAY"


# ---------------------------------------------------------------------
# Negative cases: ambiguous or absent matches -> no inference
# ---------------------------------------------------------------------


def test_two_matching_utterances_no_inference():
    """Conservative threshold: ambiguous matches are left for the LLM
    correction loop. Both utterances satisfy the candidate predicate
    so neither is chosen."""
    ws = _world_with_utterance(speakers_addressees_targets=[
        ("EVT_TELL_A", "ENT_ALICE", ["ENT_BOB"], ["ENT_CARL"], 10),
        ("EVT_TELL_B", "ENT_DANA",  ["ENT_BOB"], ["ENT_CARL"], 20),
    ])
    ws.entities["ENT_BOB"].beliefs = [Belief(
        target_id="ENT_CARL", perceived_state="Carl is alive",
        confidence=0.7, inertia=0.5,
    )]
    repaired, _ = _auto_repair(ws)
    b = _get_belief(repaired, "ENT_BOB", "ENT_CARL")
    assert b.acquired_via_event_id is None
    assert b.acquired_via_channel_id is None


def test_no_matching_utterance_no_inference():
    ws = _world_with_utterance(speakers_addressees_targets=[
        ("EVT_OTHER", "ENT_ALICE", ["ENT_DANA"], ["ENT_CARL"], 10),
    ])
    # Bob was never addressed; no inference candidate exists.
    ws.entities["ENT_BOB"] = Entity(
        id="ENT_BOB", name="Bob", location_id="LOC_X",
        status="healthy", traits={},
        beliefs=[Belief(
            target_id="ENT_CARL", perceived_state="Carl is alive",
            confidence=0.7, inertia=0.5,
        )],
    )
    repaired, _ = _auto_repair(ws)
    b = _get_belief(repaired, "ENT_BOB", "ENT_CARL")
    assert b.acquired_via_event_id is None


def test_belief_with_existing_provenance_is_not_overwritten():
    """Idempotence: if the belief already has provenance, inference
    must not touch it (regardless of what other utterances exist)."""
    ws = _world_with_utterance(speakers_addressees_targets=[
        ("EVT_TELL", "ENT_ALICE", ["ENT_BOB"], ["ENT_CARL"], 10),
    ])
    ws.entities["ENT_BOB"].beliefs = [Belief(
        target_id="ENT_CARL", perceived_state="Carl is alive",
        confidence=0.7, inertia=0.5,
        acquired_via_event_id="EVT_ALREADY_KNOWN",
    )]
    repaired, _ = _auto_repair(ws)
    b = _get_belief(repaired, "ENT_BOB", "ENT_CARL")
    # Note: dangling-ref repair will null this id because it's not in
    # the world; we only care that inference didn't overwrite it with
    # the OTHER candidate. The post-inference repair pass is allowed
    # to clear unknown ids \u2014 the invariant under test is "inference
    # respects existing provenance", not "no other pass touches it".
    assert b.acquired_via_event_id != "EVT_TELL"
