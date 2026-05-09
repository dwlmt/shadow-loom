# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Phase E validator extensions and patch handlers.

Covers the new affect-unification namespaces (Proposition / Concern /
Belief→PROP) added by ``_programmatic_validation`` and the six new
``WorldStatePatch`` ops applied by ``_apply_world_state_patch``.
"""

from __future__ import annotations

from shadow_loom.ingestion import (
    _BeliefPropAssignment,
    _apply_world_state_patch,
    _is_correction_regression,
    _programmatic_validation,
    WorldStatePatch,
)
from shadow_loom.models import (
    Belief,
    Concern,
    ConcernSnapshot,
    Entity,
    EventNode,
    Location,
    Proposition,
    PropositionSnapshot,
    WorldStateV1,
)


def _empty_world() -> WorldStateV1:
    return WorldStateV1(
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        objects={},
        entities={"ENT_A": Entity(
            id="ENT_A", name="A", location_id="LOC_X",
            status="healthy", traits={},
        )},
        events=[EventNode(
            id="EVT_1", fabula_time=10, syuzhet_index=0,
            event_type="outcome", actor_ids=["ENT_A"], target_ids=[],
            description="x",
        )],
        causal_topology=[],
    )


# ----------------------------------------------------------------------
# Validator
# ----------------------------------------------------------------------

def _categories(ws: WorldStateV1) -> list[str]:
    return [i.category for i in _programmatic_validation(ws)]


def test_validator_flags_unknown_proposition_on_concern():
    ws = _empty_world()
    ws.entities["ENT_A"].concerns = [Concern(
        concern_id="CCN_X", proposition_id="PROP_GHOST",
        polarity="fear", salience=0.5,
    )]
    cats = _categories(ws)
    assert "unknown_proposition_id" in cats


def test_validator_flags_unknown_proposition_on_belief():
    ws = _empty_world()
    ws.entities["ENT_A"].beliefs = [Belief(
        target_id="ENT_A", perceived_state="x",
        confidence=0.5, inertia=0.5,
        proposition_id="PROP_GHOST",
    )]
    cats = _categories(ws)
    assert "unknown_proposition_id" in cats


def test_validator_flags_orphan_snapshot_trigger():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
        state_timeline=[PropositionSnapshot(
            fabula_time=10, triggered_by="EVT_GHOST", stakes=0.7,
        )],
    )]
    cats = _categories(ws)
    assert "snapshot_no_event" in cats


def test_validator_flags_temporal_out_of_range_truth():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
        truth_at_fabula={9999: True},
    )]
    cats = _categories(ws)
    assert "temporal" in cats


def test_validator_flags_asymmetric_counter_concern():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    ws.entities["ENT_A"].concerns = [
        Concern(concern_id="CCN_A", proposition_id="PROP_X",
                polarity="fear", salience=0.5,
                counter_concern_ids=["CCN_B"]),
        Concern(concern_id="CCN_B", proposition_id="PROP_X",
                polarity="desire", salience=0.5,
                counter_concern_ids=[]),  # missing back-pointer
    ]
    cats = _categories(ws)
    assert "asymmetric_counter_concern" in cats


def test_validator_flags_event_resolves_unknown_prop():
    ws = _empty_world()
    ws.events[0].resolves_proposition_ids = ["PROP_GHOST"]
    cats = _categories(ws)
    assert "unknown_proposition_id" in cats


def test_validator_clean_world_has_no_phase_e_issues():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    ws.entities["ENT_A"].concerns = [Concern(
        concern_id="CCN_X", proposition_id="PROP_X",
        polarity="fear", salience=0.5,
    )]
    ws.entities["ENT_A"].beliefs = [Belief(
        target_id="ENT_A", perceived_state="x",
        confidence=0.5, inertia=0.5,
        proposition_id="PROP_X",
    )]
    cats = _categories(ws)
    for c in (
        "unknown_proposition_id", "snapshot_no_event",
        "asymmetric_counter_concern", "bad_proposition_id",
        "bad_concern_id",
    ):
        assert c not in cats, f"unexpected {c} in {cats}"


# ----------------------------------------------------------------------
# Patch handlers
# ----------------------------------------------------------------------

def test_patch_add_proposition_lands():
    ws = _empty_world()
    patch = WorldStatePatch(add_propositions={
        "PROP_X": Proposition(
            proposition_id="PROP_X", kind="event_occurs",
            referent_ids=["ENT_A"], description="x",
        ),
    })
    out, _ = _apply_world_state_patch(ws, patch)
    assert any(p.proposition_id == "PROP_X" for p in out.propositions)


def test_patch_add_proposition_dedup_on_existing():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    patch = WorldStatePatch(add_propositions={
        "PROP_X": Proposition(
            proposition_id="PROP_X", kind="event_occurs",
            referent_ids=["ENT_A"], description="dup",
        ),
    })
    out, changes = _apply_world_state_patch(ws, patch)
    assert sum(1 for p in out.propositions if p.proposition_id == "PROP_X") == 1
    # Original wins.
    assert next(p for p in out.propositions if p.proposition_id == "PROP_X").description == "x"
    assert any("already exists" in c for c in changes)


def test_patch_commit_proposition_truth():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    patch = WorldStatePatch(commit_proposition_truth={
        "PROP_X": {10: True, 20: False},
    })
    out, _ = _apply_world_state_patch(ws, patch)
    p = next(p for p in out.propositions if p.proposition_id == "PROP_X")
    assert p.truth_at_fabula == {10: True, 20: False}


def test_patch_update_proposition_snapshots_appends_and_normalises_trigger():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    patch = WorldStatePatch(
        event_renames={"EVT_OLD": "EVT_1"},
        update_proposition_snapshots={
            "PROP_X": [PropositionSnapshot(
                fabula_time=10, triggered_by="EVT_OLD", stakes=0.9,
            )],
        },
    )
    out, _ = _apply_world_state_patch(ws, patch)
    p = next(p for p in out.propositions if p.proposition_id == "PROP_X")
    assert len(p.state_timeline) == 1
    # Rename was forwarded onto the snapshot trigger.
    assert p.state_timeline[0].triggered_by == "EVT_1"


def test_patch_add_concerns_dedup_on_proposition_polarity():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    ws.entities["ENT_A"].concerns = [Concern(
        concern_id="CCN_OLD", proposition_id="PROP_X",
        polarity="fear", salience=0.5,
    )]
    patch = WorldStatePatch(add_concerns={
        "ENT_A": [Concern(
            concern_id="CCN_NEW", proposition_id="PROP_X",
            polarity="fear", salience=0.9,
        )],
    })
    out, _ = _apply_world_state_patch(ws, patch)
    # Same (proposition_id, polarity) — dedup keeps original.
    assert len(out.entities["ENT_A"].concerns) == 1
    assert out.entities["ENT_A"].concerns[0].concern_id == "CCN_OLD"


def test_patch_update_concern_snapshots_appends():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    ws.entities["ENT_A"].concerns = [Concern(
        concern_id="CCN_X", proposition_id="PROP_X",
        polarity="fear", salience=0.5,
    )]
    patch = WorldStatePatch(update_concern_snapshots={
        "ENT_A": {"CCN_X": [ConcernSnapshot(
            fabula_time=10, triggered_by="EVT_1", salience=1.0,
        )]},
    })
    out, _ = _apply_world_state_patch(ws, patch)
    ccn = out.entities["ENT_A"].concerns[0]
    assert len(ccn.state_timeline) == 1
    assert ccn.state_timeline[0].salience == 1.0


def test_patch_set_belief_proposition_id_backfill():
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    ws.entities["ENT_A"].beliefs = [Belief(
        target_id="ENT_A", perceived_state="x",
        confidence=0.5, inertia=0.5,
    )]
    patch = WorldStatePatch(set_belief_proposition_ids=[_BeliefPropAssignment(
        entity_id="ENT_A", target_id="ENT_A", proposition_id="PROP_X",
    )])
    out, _ = _apply_world_state_patch(ws, patch)
    assert out.entities["ENT_A"].beliefs[0].proposition_id == "PROP_X"


def test_patch_preserves_propositions_through_apply():
    """Round-trip an empty patch — propositions must survive."""
    ws = _empty_world()
    ws.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    out, _ = _apply_world_state_patch(ws, WorldStatePatch())
    assert any(p.proposition_id == "PROP_X" for p in out.propositions)


# ----------------------------------------------------------------------
# Regression circuit-breaker
# ----------------------------------------------------------------------

def test_regression_trips_on_proposition_loss():
    before = _empty_world()
    before.propositions = [
        Proposition(proposition_id=f"PROP_{i}", kind="event_occurs",
                    referent_ids=["ENT_A"], description="x")
        for i in range(4)
    ]
    after = before.model_copy(update={"propositions": before.propositions[:1]})
    reason = _is_correction_regression(before, after)
    assert reason is not None
    assert "propositions" in reason


def test_regression_trips_on_concern_loss():
    before = _empty_world()
    before.propositions = [Proposition(
        proposition_id="PROP_X", kind="event_occurs",
        referent_ids=["ENT_A"], description="x",
    )]
    before.entities["ENT_A"].concerns = [
        Concern(concern_id=f"CCN_{i}", proposition_id="PROP_X",
                polarity="fear", salience=0.5)
        for i in range(4)
    ]
    after = before.model_copy(deep=True)
    after.entities["ENT_A"].concerns = before.entities["ENT_A"].concerns[:1]
    reason = _is_correction_regression(before, after)
    assert reason is not None
    assert "concerns" in reason
