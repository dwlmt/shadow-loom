# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shadow-merge ``Proposition.truth_at_fabula`` provenance cascade.

When a counterfactual / intervention prunes events via
``ChunkTopology.suppressed_event_ids``, any proposition truth-value
entry whose ONLY justification was a now-suppressed event must be
dropped from the shadow row's ``Proposition.truth_at_fabula``.
Otherwise the proposition's ground-truth dict (deep-copied from
factual at ``VersionedWorldModel.merge()`` start) keeps reporting
the factual outcome on the shadow branch \u2014 producing the
documented prose\u2194interrogation drift where the prose narrates
Mrs Coady alive after the dog-killing is averted, but
``PROP_MRS_COADY_DIES.truth_at_fabula[t] == True`` remains in the
JSON snapshot.

Provenance rule (no schema change required): drop
``truth_at_fabula[ft]`` iff some suppressed event committed to the
prop at ``ft`` (via ``asserts_proposition_id`` /
``denies_proposition_id`` / ``resolves_proposition_ids``) AND no
surviving event commits to the same prop at the same ``ft``.
Entries with no event-level provenance at all (story-prior commits
without an originating event) are preserved.
"""

from __future__ import annotations

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    Entity,
    EventNode,
    Proposition,
    WorldStateV1,
)


def _ws_with_prop_committed_by(event_ids: list[str]) -> WorldStateV1:
    """Build a minimal world where one proposition becomes ``True`` at
    ``t=100`` and is asserted by the events in ``event_ids``."""
    coady = Entity(
        id="ENT_MRS_COADY", name="Mrs Coady",
        location_id="LOC_FLAT", status="healthy", traits={},
        world_id="factual",
    )
    events = [
        EventNode(
            id=eid, fabula_time=100, syuzhet_index=100,
            event_type="utterance",
            actor_ids=["ENT_GEORGE"], target_ids=["ENT_KEN"],
            description="George orders Ken: do Mrs Coady",
            asserts_proposition_id="PROP_MRS_COADY_DIES",
            truth_value="performative",
            world_id="factual",
        )
        for eid in event_ids
    ]
    prop = Proposition(
        proposition_id="PROP_MRS_COADY_DIES",
        description="Mrs Coady dies before she can testify",
        kind="outcome",
        truth_at_fabula={100: True},
    )
    return WorldStateV1(
        entities={"ENT_MRS_COADY": coady},
        events=events,
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[prop], channels={},
    )


def test_shadow_suppression_drops_unsupported_truth_at_fabula_entry():
    """Suppressing the sole asserting event drops the
    ``truth_at_fabula`` entry at the same fabula_time."""
    vwm = VersionedWorldModel.from_world_state(
        _ws_with_prop_committed_by(["EVT_UTT_GEORGE_ORDERS_KEN_KILL_COADY"]),
    )
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_UTT_GEORGE_ORDERS_KEN_KILL_COADY"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_no_dogs")
    prop = next(
        p for p in vwm2.current.propositions
        if p.proposition_id == "PROP_MRS_COADY_DIES"
    )
    assert 100 not in prop.truth_at_fabula, (
        "Entry whose only event-level provenance is a suppressed "
        "event must be dropped from the shadow row."
    )


def test_shadow_suppression_preserves_truth_with_surviving_commit():
    """If another non-suppressed event commits the same prop at the
    same fabula_time, the entry survives (over-determined)."""
    vwm = VersionedWorldModel.from_world_state(
        _ws_with_prop_committed_by([
            "EVT_UTT_SUPPRESSED",
            "EVT_UTT_SURVIVES",  # asserts same prop at same ft
        ]),
    )
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_UTT_SUPPRESSED"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    prop = next(
        p for p in vwm2.current.propositions
        if p.proposition_id == "PROP_MRS_COADY_DIES"
    )
    assert prop.truth_at_fabula.get(100) is True, (
        "Surviving event with same (prop_id, fabula_time) commit "
        "must keep the truth entry."
    )


def test_shadow_suppression_preserves_truth_without_event_provenance():
    """A story-prior ``truth_at_fabula`` entry not committed by ANY
    event (suppressed or not) is preserved \u2014 we cannot prove it
    was caused by suppressed events."""
    coady = Entity(
        id="ENT_MRS_COADY", name="Mrs Coady",
        location_id="LOC_FLAT", status="healthy", traits={},
        world_id="factual",
    )
    # An unrelated event we'll suppress.
    suppressed_evt = EventNode(
        id="EVT_KILL_DOGS", fabula_time=50, syuzhet_index=50,
        event_type="outcome",
        actor_ids=["ENT_KEN"], target_ids=["ENT_DOGS"],
        description="Ken kills the dogs",
        world_id="factual",
    )
    prop = Proposition(
        proposition_id="PROP_STORY_PRIOR",
        description="Some prior fact",
        kind="trait_holds",
        truth_at_fabula={100: True},  # no event ever asserted this
    )
    ws = WorldStateV1(
        entities={"ENT_MRS_COADY": coady},
        events=[suppressed_evt],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[prop], channels={},
    )
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_KILL_DOGS"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    prop_after = next(
        p for p in vwm2.current.propositions
        if p.proposition_id == "PROP_STORY_PRIOR"
    )
    assert prop_after.truth_at_fabula.get(100) is True, (
        "Entries with no event-level provenance are preserved "
        "(conservative: cannot be linked to suppressed events)."
    )


def test_shadow_suppression_drops_via_denies_proposition_id():
    """``denies_proposition_id`` provenance is also honoured."""
    coady = Entity(
        id="ENT_MRS_COADY", name="Mrs Coady",
        location_id="LOC_FLAT", status="healthy", traits={},
        world_id="factual",
    )
    deny_evt = EventNode(
        id="EVT_DENIAL", fabula_time=100, syuzhet_index=100,
        event_type="utterance",
        actor_ids=["ENT_LAWYER"], target_ids=["ENT_JURY"],
        description="The lawyer denies that Mrs Coady is dead.",
        denies_proposition_id="PROP_MRS_COADY_DIES",
        truth_value="true",
        world_id="factual",
    )
    prop = Proposition(
        proposition_id="PROP_MRS_COADY_DIES",
        description="Mrs Coady dies",
        kind="outcome",
        truth_at_fabula={100: False},
    )
    ws = WorldStateV1(
        entities={"ENT_MRS_COADY": coady},
        events=[deny_evt],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[prop], channels={},
    )
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_DENIAL"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    prop_after = next(
        p for p in vwm2.current.propositions
        if p.proposition_id == "PROP_MRS_COADY_DIES"
    )
    assert 100 not in prop_after.truth_at_fabula, (
        "Suppressed denying event must also drop its truth_at_fabula entry."
    )


def test_shadow_suppression_factual_proposition_dict_untouched():
    """The factual row's proposition truth dict is byte-equal after a
    shadow suppression (rows are isolated)."""
    vwm = VersionedWorldModel.from_world_state(
        _ws_with_prop_committed_by(["EVT_UTT_SUP"]),
    )
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_UTT_SUP"],
    )
    vwm.merge(topology, world_id="shadow", branch_label="cf")
    # Factual root row is untouched.
    factual_prop = next(
        p for p in vwm.current.propositions
        if p.proposition_id == "PROP_MRS_COADY_DIES"
    )
    assert factual_prop.truth_at_fabula == {100: True}


def test_shadow_suppression_via_referent_ids_drops_truth_entry():
    """Propositions tied to an event via ``referent_ids`` (not via
    ``asserts/denies/resolves_proposition_id``) must also have their
    matching-fabula-tick ``truth_at_fabula`` entries dropped on the
    shadow branch when the referenced event is suppressed.

    Regression for the Mrs Coady bug: example_worlds and author-
    authored worlds typically declare
    ``Proposition(referent_ids=[EVT_X, ENT_Y],
    truth_at_fabula={t: True})`` without setting
    ``EventNode.asserts_proposition_id``. The event-provenance
    cascade therefore missed these and the shadow-branch
    interrogate kept reporting the factual truth value even after
    the underlying event was counterfactually prevented.
    """
    coady = Entity(
        id="ENT_MRS_COADY", name="Mrs Coady",
        location_id="LOC_FLAT", status="healthy", traits={},
        world_id="factual",
    )
    # Event with NO asserts/denies/resolves provenance.
    death_evt = EventNode(
        id="EVT_MRS_COADY_DIES_HEART_ATTACK",
        fabula_time=14000, syuzhet_index=14,
        event_type="outcome",
        actor_ids=[], target_ids=["ENT_MRS_COADY"],
        description="Mrs Coady dies of a heart attack.",
        world_id="factual",
    )
    # Proposition tied to the event ONLY via referent_ids.
    prop = Proposition(
        proposition_id="PROP_MRS_COADY_DIES",
        description="Mrs Coady dies before testifying",
        kind="event_occurs",
        referent_ids=["EVT_MRS_COADY_DIES_HEART_ATTACK", "ENT_MRS_COADY"],
        truth_at_fabula={0: False, 14000: True},
    )
    ws = WorldStateV1(
        entities={"ENT_MRS_COADY": coady},
        events=[death_evt],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[prop], channels={},
    )
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_MRS_COADY_DIES_HEART_ATTACK"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_no_heart_attack")
    # Shadow projection must yield the cloned prop with the death
    # truth-tick dropped.
    projected = vwm2.current.projected_for_branch(
        branch_world_id="shadow", branch_label="cf_no_heart_attack",
    )
    prop_shadow = next(
        p for p in projected.propositions
        if p.proposition_id == "PROP_MRS_COADY_DIES"
    )
    assert 14000 not in prop_shadow.truth_at_fabula, (
        "referent_ids-linked event suppression must drop the "
        "matching truth_at_fabula entry from the shadow clone."
    )
    assert prop_shadow.truth_at_fabula.get(0) is False, (
        "Pre-event truth entries must survive on the shadow clone."
    )
    # Factual row is unchanged.
    factual_prop = next(
        p for p in vwm.current.propositions
        if p.proposition_id == "PROP_MRS_COADY_DIES"
    )
    assert factual_prop.truth_at_fabula == {0: False, 14000: True}


def test_shadow_suppression_via_referent_ids_preserves_unrelated_ticks():
    """Truth ticks whose fabula_time does NOT match any suppressed
    event are preserved \u2014 only the tick coinciding with the
    suppressed event is dropped."""
    actor = Entity(
        id="ENT_HERO", name="Hero",
        location_id="LOC_X", status="healthy", traits={},
        world_id="factual",
    )
    evt = EventNode(
        id="EVT_DEFEAT", fabula_time=500, syuzhet_index=5,
        event_type="outcome",
        actor_ids=["ENT_HERO"], target_ids=[],
        description="The hero is defeated.",
        world_id="factual",
    )
    prop = Proposition(
        proposition_id="PROP_HERO_LIVES",
        description="The hero lives.",
        kind="trait_holds",
        referent_ids=["EVT_DEFEAT", "ENT_HERO"],
        # Pre-event True, mid-story False at a non-event tick, and
        # False at the event tick.
        truth_at_fabula={0: True, 200: True, 500: False},
    )
    ws = WorldStateV1(
        entities={"ENT_HERO": actor},
        events=[evt],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[prop], channels={},
    )
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_DEFEAT"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    projected = vwm2.current.projected_for_branch(
        branch_world_id="shadow", branch_label="cf",
    )
    prop_shadow = next(
        p for p in projected.propositions
        if p.proposition_id == "PROP_HERO_LIVES"
    )
    assert prop_shadow.truth_at_fabula.get(0) is True
    assert prop_shadow.truth_at_fabula.get(200) is True
    assert 500 not in prop_shadow.truth_at_fabula


def test_shadow_suppression_widens_to_drop_intermediate_truth_when_all_referents_suppressed():
    """Gap #3: when EVERY event in ``referent_ids`` is suppressed,
    the cascade also drops every ``truth_at_fabula`` entry at-or-
    after the earliest suppressed referent's fabula_time \u2014 not
    only entries exactly at a suppressed event's tick.

    Macbeth ``PROP_FEUDAL_ORDER_INTACT`` shape: factual truth
    ``{0: True, 500: False, 5000: True}`` where ``500`` is the
    declarative ``False`` set at Macbeth's coup (caused by
    ``EVT_DUNCAN_MURDER``) and ``5000`` is the declarative ``True``
    restoration at Malcolm's coronation. Suppressing both events
    (e.g. via a witches-prophecy counterfactual) must restore the
    proposition to ``{0: True}`` on the shadow branch \u2014 not
    leave the orphan ``False`` between them.
    """
    duncan = Entity(
        id="ENT_DUNCAN", name="Duncan",
        location_id="LOC_INVERNESS", status="healthy", traits={},
        world_id="factual",
    )
    murder_evt = EventNode(
        id="EVT_DUNCAN_MURDER",
        fabula_time=500, syuzhet_index=5,
        event_type="outcome",
        actor_ids=["ENT_MACBETH"], target_ids=["ENT_DUNCAN"],
        description="Macbeth murders Duncan in his sleep.",
        world_id="factual",
    )
    crown_evt = EventNode(
        id="EVT_MALCOLM_CROWNED",
        fabula_time=5000, syuzhet_index=50,
        event_type="outcome",
        actor_ids=["ENT_MALCOLM"], target_ids=[],
        description="Malcolm is crowned king at Scone.",
        world_id="factual",
    )
    prop = Proposition(
        proposition_id="PROP_FEUDAL_ORDER_INTACT",
        description="The Scottish feudal order is intact.",
        kind="outcome",
        referent_ids=["EVT_DUNCAN_MURDER", "EVT_MALCOLM_CROWNED"],
        truth_at_fabula={0: True, 500: False, 5000: True},
    )
    ws = WorldStateV1(
        entities={"ENT_DUNCAN": duncan},
        events=[murder_evt, crown_evt],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[prop], channels={},
    )
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_DUNCAN_MURDER", "EVT_MALCOLM_CROWNED"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_no_prophecy")
    projected = vwm2.current.projected_for_branch(
        branch_world_id="shadow", branch_label="cf_no_prophecy",
    )
    prop_shadow = next(
        p for p in projected.propositions
        if p.proposition_id == "PROP_FEUDAL_ORDER_INTACT"
    )
    assert prop_shadow.truth_at_fabula == {0: True}, (
        "All-referents-suppressed must drop every truth tick at-or-"
        "after the earliest suppressed event's fabula_time."
    )


def test_shadow_suppression_partial_referents_keeps_exact_tick_only():
    """Gap #3 conservative side: if SOME event referents survive,
    we keep only the exact-tick rule \u2014 we cannot prove an
    intermediate declarative tick was caused by the suppressed
    subset (the surviving event may have set it)."""
    actor = Entity(
        id="ENT_HERO", name="Hero",
        location_id="LOC_X", status="healthy", traits={},
        world_id="factual",
    )
    evt_a = EventNode(
        id="EVT_FALL", fabula_time=300, syuzhet_index=3,
        event_type="outcome",
        actor_ids=["ENT_HERO"], target_ids=[],
        description="The hero falls.",
        world_id="factual",
    )
    evt_b = EventNode(
        id="EVT_RISE", fabula_time=700, syuzhet_index=7,
        event_type="outcome",
        actor_ids=["ENT_HERO"], target_ids=[],
        description="The hero rises again.",
        world_id="factual",
    )
    prop = Proposition(
        proposition_id="PROP_HERO_RULES",
        description="The hero rules.",
        kind="trait_holds",
        referent_ids=["EVT_FALL", "EVT_RISE"],
        truth_at_fabula={0: True, 300: False, 500: False, 700: True},
    )
    ws = WorldStateV1(
        entities={"ENT_HERO": actor},
        events=[evt_a, evt_b],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[prop], channels={},
    )
    vwm = VersionedWorldModel.from_world_state(ws)
    # Only suppress one of the two referents.
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_FALL"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    projected = vwm2.current.projected_for_branch(
        branch_world_id="shadow", branch_label="cf",
    )
    prop_shadow = next(
        p for p in projected.propositions
        if p.proposition_id == "PROP_HERO_RULES"
    )
    # Exact-tick rule: only fabula_time=300 dropped.
    assert prop_shadow.truth_at_fabula.get(0) is True
    assert 300 not in prop_shadow.truth_at_fabula
    assert prop_shadow.truth_at_fabula.get(500) is False
    assert prop_shadow.truth_at_fabula.get(700) is True
