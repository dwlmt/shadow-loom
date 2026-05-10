# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PR 4: directive + auditor co-presence wiring tests.

Covers:
  * ``build_event_copresence_constraints`` emits one HARD spatial
    block per windowed event with ``at_location_id``; bound actors
    appear in ``must_be_present``; channel-mediated addressees
    appear in ``channel_exempt`` (not in ``must_be_present``);
    elsewhere-located entities surface in ``must_not_be_present``.
  * Deterministic auditor catches ``event_copresence_violation``
    (bound actor staged at a different LOC) and
    ``event_copresence_omission`` (phantom witness staged at the
    event scene).
"""

from __future__ import annotations

from shadow_loom.models import (
    Channel, Entity, EntityStateSnapshot, EventNode, Location,
    TraitVector, WorldStateV1,
)
from shadow_loom.directive_assembly import (
    CreativeBrief, build_event_copresence_constraints,
)
from shadow_loom.auditor import _event_copresence_violations


def _ent(eid, *, location, name=None, timeline=None):
    return Entity(
        id=eid, name=name or eid, location_id=location, status="healthy",
        traits={"x": TraitVector(value=0.5, inertia=0.5)},
        beliefs=[], state_timeline=timeline or [],
    )


def _ws(*, entities, locations, events, channels=None):
    if isinstance(locations, list):
        loc_map = {lid: lobj for lid, lobj in locations}
    else:
        loc_map = locations
    return WorldStateV1(
        entities={e.id: e for e in entities},
        locations=loc_map,
        objects={}, world_traits={}, events=events,
        causal_topology=[], channels=channels or {},
    )


def _loc(lid, name):
    return lid, Location(name=name, description="d", ambient_state={})


class TestBuildEventCopresenceConstraints:
    def test_face_to_face_event_emits_must_be_present(self):
        macbeth = _ent("ENT_MACBETH", location="LOC_BEDCHAMBER", name="Macbeth")
        duncan = _ent("ENT_DUNCAN", location="LOC_BEDCHAMBER", name="Duncan")
        evt = EventNode(
            id="EVT_MURDER", fabula_time=10, syuzhet_index=10,
            event_type="outcome", actor_ids=["ENT_MACBETH"],
            target_ids=["ENT_DUNCAN"], description="murder",
            at_location_id="LOC_BEDCHAMBER",
        )
        ws = _ws(
            entities=[macbeth, duncan],
            locations=[_loc("LOC_BEDCHAMBER", "Bedchamber"), _loc("LOC_HALL", "Hall")],
            events=[evt],
        )
        blocks = build_event_copresence_constraints(ws, fabula_anchor=10, syuzhet_anchor=10)
        assert len(blocks) == 1
        b = blocks[0]
        assert b.constraint_type == "spatial"
        assert b.priority == "hard"
        assert b.evidence["event_id"] == "EVT_MURDER"
        assert set(b.evidence["must_be_present"]) == {"ENT_MACBETH", "ENT_DUNCAN"}
        assert b.evidence["channel_exempt"] == []

    def test_channel_utterance_addressee_is_exempt(self):
        a = _ent("ENT_A", location="LOC_OCEANIA", name="Winston")
        b = _ent("ENT_B", location="LOC_AIRSTRIP", name="Julia")
        ch = Channel(
            id="CHN_TELESCREEN", name="ts", medium="electronic",
            participant_ids=["ENT_A", "ENT_B"], directionality="duplex",
            intelligibility={},
        )
        evt = EventNode(
            id="EVT_TALK", fabula_time=5, syuzhet_index=5,
            event_type="utterance", actor_ids=["ENT_A"],
            speaker_id="ENT_A", addressee_ids=["ENT_B"],
            via_channel_id="CHN_TELESCREEN",
            description="speaks", content="hi",
            at_location_id="LOC_OCEANIA",
        )
        ws = _ws(
            entities=[a, b],
            locations=[_loc("LOC_OCEANIA", "Oceania"), _loc("LOC_AIRSTRIP", "Airstrip One")],
            events=[evt], channels={ch.id: ch},
        )
        blocks = build_event_copresence_constraints(ws, fabula_anchor=5, syuzhet_anchor=5)
        assert len(blocks) == 1
        ev = blocks[0].evidence
        # Speaker present at the event location; addressee channel-exempt.
        assert "ENT_A" in ev["must_be_present"]
        assert "ENT_B" in ev["channel_exempt"]
        assert "ENT_B" not in ev["must_be_present"]

    def test_phantom_witness_listed_in_must_not_be_present(self):
        macbeth = _ent("ENT_MACBETH", location="LOC_BEDCHAMBER")
        # Banquo is at the hall at fabula_time=10 (his initial location).
        banquo = _ent("ENT_BANQUO", location="LOC_HALL", name="Banquo")
        evt = EventNode(
            id="EVT_MURDER", fabula_time=10, syuzhet_index=10,
            event_type="outcome", actor_ids=["ENT_MACBETH"],
            target_ids=[], description="murder",
            at_location_id="LOC_BEDCHAMBER",
        )
        ws = _ws(
            entities=[macbeth, banquo],
            locations=[_loc("LOC_BEDCHAMBER", "Bedchamber"), _loc("LOC_HALL", "Hall")],
            events=[evt],
        )
        blocks = build_event_copresence_constraints(ws, fabula_anchor=10, syuzhet_anchor=10)
        ev = blocks[0].evidence
        assert "ENT_BANQUO" in ev["must_not_be_present"]

    def test_event_outside_window_skipped(self):
        a = _ent("ENT_A", location="LOC_HALL")
        evt = EventNode(
            id="EVT_FAR", fabula_time=100, syuzhet_index=100,
            event_type="outcome", actor_ids=["ENT_A"],
            description="d", at_location_id="LOC_HALL",
        )
        ws = _ws(
            entities=[a], locations=[_loc("LOC_HALL", "Hall")], events=[evt],
        )
        blocks = build_event_copresence_constraints(
            ws, fabula_anchor=10, syuzhet_anchor=10, window=1,
        )
        assert blocks == []

    def test_event_without_at_location_skipped(self):
        a = _ent("ENT_A", location="LOC_HALL")
        evt = EventNode(
            id="EVT_PLACELESS", fabula_time=10, syuzhet_index=10,
            event_type="revelation", actor_ids=["ENT_A"],
            description="d",
        )
        ws = _ws(
            entities=[a], locations=[_loc("LOC_HALL", "Hall")], events=[evt],
        )
        blocks = build_event_copresence_constraints(ws, fabula_anchor=10, syuzhet_anchor=10)
        assert blocks == []


def _brief_with_blocks(blocks):
    return CreativeBrief(
        target_effect="mystery",
        target_entities=["ENT_TEST"],
        constraints=blocks,
        epistemic_gaps=[],
        trait_trajectories=[],
        intervention_mechanisms=[],
        abduction_truths=[],
        threat_proximity=None,
        counterfactual_branch=None,
        causal_attribution=None,
        entanglement_pairs=[],
        rendering=None,
        scene_context={},
    )


class TestEventCopresenceAuditor:
    def test_flags_actor_at_wrong_location(self):
        macbeth = _ent("ENT_MACBETH", location="LOC_BEDCHAMBER", name="Macbeth")
        duncan = _ent("ENT_DUNCAN", location="LOC_BEDCHAMBER", name="Duncan")
        evt = EventNode(
            id="EVT_MURDER", fabula_time=10, syuzhet_index=10,
            event_type="outcome", actor_ids=["ENT_MACBETH"],
            target_ids=["ENT_DUNCAN"], description="murder",
            at_location_id="LOC_BEDCHAMBER",
        )
        ws = _ws(
            entities=[macbeth, duncan],
            locations=[_loc("LOC_BEDCHAMBER", "Bedchamber"), _loc("LOC_HALL", "Hall")],
            events=[evt],
        )
        blocks = build_event_copresence_constraints(ws, fabula_anchor=10, syuzhet_anchor=10)
        brief = _brief_with_blocks(blocks)
        prose = "Macbeth strode through the Hall, his hands still shaking."
        issues = _event_copresence_violations(prose, brief, ws)
        assert any(v.violation_type == "event_copresence_violation" for v in issues)

    def test_flags_phantom_witness(self):
        macbeth = _ent("ENT_MACBETH", location="LOC_BEDCHAMBER", name="Macbeth")
        banquo = _ent("ENT_BANQUO", location="LOC_HALL", name="Banquo")
        evt = EventNode(
            id="EVT_MURDER", fabula_time=10, syuzhet_index=10,
            event_type="outcome", actor_ids=["ENT_MACBETH"],
            description="murder", at_location_id="LOC_BEDCHAMBER",
        )
        ws = _ws(
            entities=[macbeth, banquo],
            locations=[_loc("LOC_BEDCHAMBER", "Bedchamber"), _loc("LOC_HALL", "Hall")],
            events=[evt],
        )
        blocks = build_event_copresence_constraints(ws, fabula_anchor=10, syuzhet_anchor=10)
        brief = _brief_with_blocks(blocks)
        prose = "In the Bedchamber, Banquo watched in silence as the deed was done."
        issues = _event_copresence_violations(prose, brief, ws)
        assert any(v.violation_type == "event_copresence_omission" for v in issues)

    def test_clean_prose_no_violations(self):
        macbeth = _ent("ENT_MACBETH", location="LOC_BEDCHAMBER", name="Macbeth")
        duncan = _ent("ENT_DUNCAN", location="LOC_BEDCHAMBER", name="Duncan")
        evt = EventNode(
            id="EVT_MURDER", fabula_time=10, syuzhet_index=10,
            event_type="outcome", actor_ids=["ENT_MACBETH"],
            target_ids=["ENT_DUNCAN"], description="murder",
            at_location_id="LOC_BEDCHAMBER",
        )
        ws = _ws(
            entities=[macbeth, duncan],
            locations=[_loc("LOC_BEDCHAMBER", "Bedchamber"), _loc("LOC_HALL", "Hall")],
            events=[evt],
        )
        blocks = build_event_copresence_constraints(ws, fabula_anchor=10, syuzhet_anchor=10)
        brief = _brief_with_blocks(blocks)
        prose = "In the Bedchamber, Macbeth bent over Duncan in the dark."
        issues = _event_copresence_violations(prose, brief, ws)
        assert issues == []
