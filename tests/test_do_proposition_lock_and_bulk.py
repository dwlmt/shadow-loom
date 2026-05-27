"""Candidate D coverage: ``DoProposition.truth_at_fabula`` bulk
overwrite and ``hard_lock_forever`` pin.
"""
from __future__ import annotations

import networkx as nx

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.models import Location, Proposition, WorldStateV1
from shadow_loom.query_models import DoProposition


def _ws_with_prop(initial_truth: dict[int, bool] | None = None) -> WorldStateV1:
    loc = Location(id="LOC_0", name="loc", description="x")
    prop = Proposition(
        id="PROP_X",
        proposition_id="PROP_X",
        kind="outcome",
        description="X holds",
        truth_at_fabula=dict(initial_truth or {}),
    )
    return WorldStateV1(
        locations={"LOC_0": loc},
        entities={},
        objects={},
        events=[],
        causal_topology=[],
        propositions=[prop],
    )


def _engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    return CausalPhysicsEngine(nx.MultiDiGraph(), ws)


def test_bulk_truth_at_fabula_overwrites_ledger():
    ws = _ws_with_prop({0: False, 1000: True, 5000: True})
    eng = _engine(ws)
    eng._apply_do_proposition(DoProposition(
        proposition_id="PROP_X",
        truth=False,
        fabula_time=2000,
        truth_at_fabula={0: True, 5000: False},
        propagate_to_beliefs=False,
    ))
    prop = ws.propositions[0]
    # Bulk dict replaced the ledger; anchor ft=2000 also lands.
    assert prop.truth_at_fabula == {0: True, 2000: False, 5000: False}


def test_hard_lock_pins_proposition_in_engine_set():
    ws = _ws_with_prop({0: False})
    eng = _engine(ws)
    assert not getattr(eng, "_proposition_hard_locks", set())
    eng._apply_do_proposition(DoProposition(
        proposition_id="PROP_X",
        truth=True,
        fabula_time=100,
        hard_lock_forever=True,
        propagate_to_beliefs=False,
    ))
    assert "PROP_X" in eng._proposition_hard_locks


def test_hard_lock_without_inverse_does_not_create_phantom_entry():
    ws = _ws_with_prop({0: False})
    eng = _engine(ws)
    eng._apply_do_proposition(DoProposition(
        proposition_id="PROP_X",
        truth=True,
        fabula_time=100,
        hard_lock_forever=True,
        propagate_to_beliefs=False,
    ))
    assert eng._proposition_hard_locks == {"PROP_X"}


def test_default_clamp_without_hard_lock_leaves_lock_set_empty():
    ws = _ws_with_prop({0: False})
    eng = _engine(ws)
    eng._apply_do_proposition(DoProposition(
        proposition_id="PROP_X",
        truth=True,
        fabula_time=100,
        propagate_to_beliefs=False,
    ))
    assert not getattr(eng, "_proposition_hard_locks", set())


def test_bulk_truth_at_fabula_honours_primary_truth_anchor():
    """When the bulk dict omits the anchor tick, the primary ``truth``
    field still lands there so the do-target's headline assertion is
    never lost."""
    ws = _ws_with_prop({0: False})
    eng = _engine(ws)
    eng._apply_do_proposition(DoProposition(
        proposition_id="PROP_X",
        truth=True,
        fabula_time=3000,
        truth_at_fabula={0: False, 5000: False},  # omits 3000
        propagate_to_beliefs=False,
    ))
    assert ws.propositions[0].truth_at_fabula[3000] is True
