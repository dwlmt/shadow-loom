# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for proposition / concern conflict handling.

Covers four scenarios introduced by the May 2026 conflict-handling pass:

  * Same-tick true/false truth-write collision is logged as a warning.
  * ``Proposition.inverse_proposition_id`` mirrors truth commits with
    the opposite value during the Phase C truth-write sweep.
  * Phase C counter-concern auto-closure transitively closes partner
    concerns through ``Concern.counter_concern_ids``.
  * Do-surgery applying a PROP truth clamp + a contradictory concern
    clamp in the same query emits a consistency warning.
"""
from __future__ import annotations

import logging
from typing import List

import pytest

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, Proposition, Concern,
)


# =====================================================================
# Same-tick true/false truth-write collision
# =====================================================================

def test_same_tick_truth_collision_logs_warning(caplog):
    """When two events in the same Phase C sweep write opposing truths
    to the same (PROP, fabula_time), the reconciler picks last-write-
    wins AND emits a warning so the operator can see the conflict.
    """
    from shadow_loom.ingestion import reconcile_affect

    prop = Proposition(
        proposition_id="PROP_X",
        kind="event_occurs",
        description="X happened",
        referent_ids=["ENT_A"],
    )
    ws = WorldStateV1(
        locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_A",
                status="healthy", traits={},
            ),
        },
        events=[
            EventNode(
                id="EVT_AFFIRM", fabula_time=10, syuzhet_index=0,
                event_type="utterance", actor_ids=["ENT_A"],
                target_ids=[], description="A affirms X.",
                asserts_proposition_id="PROP_X", truth_value="true",
            ),
            EventNode(
                id="EVT_DENY", fabula_time=10, syuzhet_index=1,
                event_type="utterance", actor_ids=["ENT_A"],
                target_ids=[], description="A denies X.",
                denies_proposition_id="PROP_X",
            ),
        ],
        causal_topology=[], spatial_topology=[], social_topology=[],
        propositions=[prop],
    )

    with caplog.at_level(logging.WARNING, logger="shadow_loom.ingestion"):
        reconcile_affect(ws, None, [])

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "Conflicting truth writes within the same pass on PROP_X@fabula=10" in m
        for m in msgs
    ), f"expected same-tick collision warning, got: {msgs}"


# =====================================================================
# Inverse proposition auto-mirror
# =====================================================================

def test_inverse_proposition_truth_is_mirrored():
    """When PROP_X has ``inverse_proposition_id=PROP_NOT_X`` and an
    event commits PROP_X=true, the reconciler must auto-write
    PROP_NOT_X=false at the same fabula tick.
    """
    from shadow_loom.ingestion import reconcile_affect

    ws = WorldStateV1(
        locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_A",
                status="healthy", traits={},
            ),
        },
        events=[
            EventNode(
                id="EVT_COMMIT", fabula_time=20, syuzhet_index=0,
                event_type="revelation", actor_ids=["ENT_A"],
                target_ids=[], description="X is revealed true.",
                resolves_proposition_ids=["PROP_X"],
            ),
        ],
        causal_topology=[], spatial_topology=[], social_topology=[],
        propositions=[
            Proposition(
                proposition_id="PROP_X",
                kind="event_occurs",
                description="X holds",
                referent_ids=["ENT_A"],
                inverse_proposition_id="PROP_NOT_X",
            ),
            Proposition(
                proposition_id="PROP_NOT_X",
                kind="event_occurs",
                description="X does not hold",
                referent_ids=["ENT_A"],
                inverse_proposition_id="PROP_X",
            ),
        ],
    )

    out = reconcile_affect(ws, None, [])
    prop_x = next(p for p in out.propositions if p.proposition_id == "PROP_X")
    prop_not_x = next(
        p for p in out.propositions if p.proposition_id == "PROP_NOT_X"
    )
    assert prop_x.truth_at_fabula.get(20) is True
    assert prop_not_x.truth_at_fabula.get(20) is False


def test_inverse_proposition_conflict_does_not_overwrite(caplog):
    """When the inverse PROP already has an explicit truth at the same
    fabula tick that conflicts with the mirrored value, the mirror is
    skipped and a warning is logged.
    """
    from shadow_loom.ingestion import reconcile_affect

    ws = WorldStateV1(
        locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_A",
                status="healthy", traits={},
            ),
        },
        events=[
            EventNode(
                id="EVT_COMMIT", fabula_time=30, syuzhet_index=0,
                event_type="revelation", actor_ids=["ENT_A"],
                target_ids=[], description="X true.",
                resolves_proposition_ids=["PROP_X"],
            ),
        ],
        causal_topology=[], spatial_topology=[], social_topology=[],
        propositions=[
            Proposition(
                proposition_id="PROP_X",
                kind="event_occurs",
                description="X",
                referent_ids=["ENT_A"],
                inverse_proposition_id="PROP_NOT_X",
            ),
            Proposition(
                proposition_id="PROP_NOT_X",
                kind="event_occurs",
                description="not X",
                referent_ids=["ENT_A"],
                # Pre-existing TRUE at the same tick contradicts the
                # would-be inverse mirror (false).
                truth_at_fabula={30: True},
            ),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="shadow_loom.ingestion"):
        out = reconcile_affect(ws, None, [])
    prop_not_x = next(
        p for p in out.propositions if p.proposition_id == "PROP_NOT_X"
    )
    # Pre-existing value preserved; mirror was skipped.
    assert prop_not_x.truth_at_fabula[30] is True
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "Inverse-proposition consistency conflict" in m for m in msgs
    ), f"expected inverse-conflict warning, got: {msgs}"


# =====================================================================
# Counter-concern transitive auto-closure
# =====================================================================

def test_counter_concern_transitive_autoclose():
    """When PROP_X commits and there exists a desire-concern over
    PROP_X paired with a fear-concern over PROP_X via
    ``counter_concern_ids``, Phase C must inject closure snapshots
    onto BOTH concerns at the commit tick (with the partner snapshot
    propagated through the counter-concern link).
    """
    from shadow_loom.ingestion import reconcile_affect

    desire = Concern(
        concern_id="CCN_DESIRE",
        proposition_id="PROP_X",
        polarity="desire",
        salience=0.9,
        counter_concern_ids=["CCN_FEAR"],
    )
    fear = Concern(
        concern_id="CCN_FEAR",
        proposition_id="PROP_X",
        polarity="fear",
        salience=0.7,
        counter_concern_ids=["CCN_DESIRE"],
    )
    ws = WorldStateV1(
        locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_A",
                status="healthy", traits={},
                concerns=[desire],
            ),
            "ENT_B": Entity(
                id="ENT_B", name="B", location_id="LOC_A",
                status="healthy", traits={},
                concerns=[fear],
            ),
        },
        events=[
            EventNode(
                id="EVT_COMMIT", fabula_time=40, syuzhet_index=0,
                event_type="revelation", actor_ids=["ENT_A"],
                target_ids=[], description="X resolves true.",
                resolves_proposition_ids=["PROP_X"],
            ),
        ],
        causal_topology=[], spatial_topology=[], social_topology=[],
        propositions=[
            Proposition(
                proposition_id="PROP_X",
                kind="outcome",
                description="X",
                referent_ids=["ENT_A"],
            ),
        ],
    )

    out = reconcile_affect(ws, None, [])
    a_concerns = out.entities["ENT_A"].concerns or []
    b_concerns = out.entities["ENT_B"].concerns or []
    desire_out = next(c for c in a_concerns if c.concern_id == "CCN_DESIRE")
    fear_out = next(c for c in b_concerns if c.concern_id == "CCN_FEAR")

    def _has_closure_at(c, fabula_tick):
        for snap in (c.state_timeline or []):
            if snap.fabula_time == fabula_tick:
                # Closure = low salience or activation window cap.
                if (snap.salience is not None and snap.salience < 0.2):
                    return True
                if snap.activation_fabula_window is not None:
                    return True
        return False

    assert _has_closure_at(desire_out, 40), (
        f"desire concern not closed at commit: {desire_out.state_timeline}"
    )
    assert _has_closure_at(fear_out, 40), (
        f"fear concern (counter) not closed at commit: {fear_out.state_timeline}"
    )


# =====================================================================
# Do-surgery PROP + counter-concern consistency warning
# =====================================================================

def test_do_surgery_prop_concern_contradiction_warns(caplog):
    """Clamping PROP_X=true while flipping its desire-concern polarity
    to ``fear`` in the same query must emit a consistency warning from
    the causal-physics pre-flight (the surgery is still applied;
    operator simply gets a heads-up that the topology is inconsistent).
    """
    from shadow_loom.causal_physics import CausalPhysicsEngine
    from shadow_loom.query_models import DoProposition, DoConcern
    import networkx as nx

    ws = WorldStateV1(
        locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_A",
                status="healthy", traits={},
                concerns=[
                    Concern(
                        concern_id="CCN_DESIRE_X",
                        proposition_id="PROP_X",
                        polarity="desire",
                        salience=0.8,
                    ),
                ],
            ),
        },
        events=[],
        causal_topology=[], spatial_topology=[], social_topology=[],
        propositions=[
            Proposition(
                proposition_id="PROP_X",
                kind="event_occurs",
                description="X",
                referent_ids=["ENT_A"],
            ),
        ],
    )

    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    do_targets = [
        DoProposition(proposition_id="PROP_X", truth=True),
        # Flip the desire concern to fear over the same proposition
        # while truth is being clamped true — a textbook contradiction.
        DoConcern(
            holder_id="ENT_A",
            concern_id="CCN_DESIRE_X",
            polarity="fear",
        ),
    ]
    with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
        engine.apply_do_targets(do_targets)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("Consistency" in m and "CCN_DESIRE_X" in m for m in msgs), (
        f"expected do-surgery consistency warning, got: {msgs}"
    )
