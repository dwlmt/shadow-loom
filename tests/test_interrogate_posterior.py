# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the deterministic interrogate posterior helper (AUDIT P0-7)."""
from __future__ import annotations

import pytest

from shadow_loom.interrogate_posterior import (
    audit_posterior_consistency,
    interrogate_posterior,
)
from shadow_loom.models import (
    Belief,
    Entity,
    Location,
    Proposition,
    TraitVector,
    WorldStateV1,
)


def _ws_with_two_props() -> WorldStateV1:
    ent_a = Entity(
        id="ENT_A", name="A",
        location_id="LOC_X", status="healthy",
        traits={"loyalty": TraitVector(value=0.8, inertia=0.6, evidence_strength="moderate")},
        beliefs=[
            Belief(target_id="ENT_B", perceived_state="B is loyal",
                   proposition_id="PROP_B_LOYAL",
                   confidence=0.9, inertia=0.8, evidence_strength="strong"),
            Belief(target_id="ENT_B", perceived_state="B is alive",
                   proposition_id="PROP_B_ALIVE",
                   confidence=0.95, inertia=0.7, evidence_strength="strong"),
        ],
    )
    ent_b = Entity(
        id="ENT_B", name="B",
        location_id="LOC_X", status="healthy",
        traits={"loyalty": TraitVector(value=0.7, inertia=0.5, evidence_strength="moderate")},
    )
    return WorldStateV1(
        entities={"ENT_A": ent_a, "ENT_B": ent_b},
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        events=[],
        causal_topology=[],
        objects={},
        propositions=[
            Proposition(
                proposition_id="PROP_B_LOYAL",
                kind="trait_holds",
                referent_ids=["ENT_B"],
                description="B is loyal",
                audience_default_prior=0.6,
                stakes=0.8,
                truth_at_fabula={1000: True, 5000: False},
            ),
            Proposition(
                proposition_id="PROP_B_ALIVE",
                kind="trait_holds",
                referent_ids=["ENT_B"],
                description="B is alive",
                audience_default_prior=0.9,
                stakes=0.6,
                truth_at_fabula={1000: True},
            ),
        ],
    )


def test_posterior_picks_correct_tick():
    ws = _ws_with_two_props()
    rows = interrogate_posterior(ws, fabula_time=3000)
    by_id = {r["proposition_id"]: r for r in rows}
    # PROP_B_LOYAL: at ft=3000 the largest tick <= 3000 is 1000 → True.
    assert by_id["PROP_B_LOYAL"]["canonical_tick"] == 1000
    assert by_id["PROP_B_LOYAL"]["truth_at_query"] is True
    # PROP_B_ALIVE: same.
    assert by_id["PROP_B_ALIVE"]["canonical_tick"] == 1000
    assert by_id["PROP_B_ALIVE"]["truth_at_query"] is True


def test_posterior_flips_polarity_past_truth_change():
    ws = _ws_with_two_props()
    rows = interrogate_posterior(ws, fabula_time=7000)
    by_id = {r["proposition_id"]: r for r in rows}
    # PROP_B_LOYAL: at ft=7000 truth=False, so belief endorses
    # contradicts bucket.
    assert by_id["PROP_B_LOYAL"]["truth_at_query"] is False
    assert by_id["PROP_B_LOYAL"]["contradict_weight"] > 0
    assert "ENT_A" in by_id["PROP_B_LOYAL"]["contradictors"]


def test_posterior_default_picks_largest_tick():
    ws = _ws_with_two_props()
    rows = interrogate_posterior(ws, fabula_time=None)
    by_id = {r["proposition_id"]: r for r in rows}
    assert by_id["PROP_B_LOYAL"]["canonical_tick"] == 5000
    assert by_id["PROP_B_LOYAL"]["truth_at_query"] is False


def test_posterior_sorts_by_stakes_desc():
    ws = _ws_with_two_props()
    rows = interrogate_posterior(ws, fabula_time=1000)
    # PROP_B_LOYAL has stakes=0.8, PROP_B_ALIVE has stakes=0.6
    assert rows[0]["proposition_id"] == "PROP_B_LOYAL"
    assert rows[1]["proposition_id"] == "PROP_B_ALIVE"


def test_audit_warns_on_under_modelled_prop():
    ent = Entity(
        id="ENT_X", name="X",
        location_id="LOC_X", status="healthy",
        traits={"loyalty": TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")},
    )
    ws = WorldStateV1(
        entities={"ENT_X": ent},
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        events=[],
        causal_topology=[],
        objects={},
        propositions=[
            Proposition(
                proposition_id="PROP_HIGH_STAKES_NO_BELIEFS",
                kind="outcome",
                referent_ids=["ENT_X"],
                description="High-stakes claim with no belief evidence",
                audience_default_prior=0.5, stakes=0.9,
                truth_at_fabula={1000: True},
            ),
        ],
    )
    rows = interrogate_posterior(ws, fabula_time=1000)
    warnings = audit_posterior_consistency(rows)
    assert any("under-modelled" in w for w in warnings)
