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
