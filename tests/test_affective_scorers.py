# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for deterministic affective scorers (AUDIT P1-2)."""
from __future__ import annotations

from shadow_loom.affective_scorers import compute_affective_scorers
from shadow_loom.models import (
    Belief,
    Concern,
    Entity,
    Location,
    Proposition,
    RelationshipEdge,
    RelationshipMetric,
    TraitVector,
    WorldStateV1,
)


def _ws(**overrides) -> WorldStateV1:
    defaults = dict(
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A",
                location_id="LOC_X", status="healthy",
                traits={"loyalty": TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")},
            ),
        },
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        objects={},
        events=[],
        causal_topology=[],
        propositions=[],
    )
    defaults.update(overrides)
    return WorldStateV1(**defaults)


def test_empty_world_returns_zeros():
    s = compute_affective_scorers(_ws())
    assert s == {
        "mystery": 0.0, "irony": 0.0, "suspense": 0.0,
        "surprise": 0.0, "tension": 0.0, "ambivalence": 0.0,
    }


def test_mystery_tracks_undecided_high_stakes_props():
    props = [
        Proposition(proposition_id="P1", kind="event_occurs", referent_ids=[],
                    description="x", audience_default_prior=0.5, stakes=0.8,
                    truth_at_fabula={}),
        Proposition(proposition_id="P2", kind="event_occurs", referent_ids=[],
                    description="y", audience_default_prior=0.5, stakes=0.6,
                    truth_at_fabula={}),
    ]
    s = compute_affective_scorers(_ws(propositions=props))
    assert s["mystery"] > 0.6


def test_irony_flags_high_conf_belief_against_canonical_truth():
    prop = Proposition(
        proposition_id="P1", kind="event_occurs", referent_ids=[],
        description="x", audience_default_prior=0.5, stakes=0.5,
        truth_at_fabula={1000: False},
    )
    ent = Entity(
        id="ENT_A", name="A", location_id="LOC_X", status="healthy",
        traits={"loyalty": TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")},
        beliefs=[
            Belief(target_id="ENT_A", perceived_state="X is true",
                   proposition_id="P1", confidence=0.9, inertia=0.5,
                   evidence_strength="strong"),
        ],
    )
    s = compute_affective_scorers(_ws(entities={"ENT_A": ent}, propositions=[prop]))
    assert s["irony"] == 1.0


def test_suspense_tracks_open_concerns_on_undecided_props():
    prop = Proposition(
        proposition_id="P1", kind="event_occurs", referent_ids=[],
        description="x", audience_default_prior=0.5, stakes=0.5,
        truth_at_fabula={},
    )
    ent = Entity(
        id="ENT_A", name="A", location_id="LOC_X", status="healthy",
        traits={"loyalty": TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")},
        concerns=[
            Concern(concern_id="CCN_1", proposition_id="P1",
                    polarity="fear", salience=0.8),
        ],
    )
    s = compute_affective_scorers(_ws(entities={"ENT_A": ent}, propositions=[prop]))
    assert s["suspense"] == 0.8


def test_tension_aggregates_dyad_fear():
    re = RelationshipEdge(
        source_entity_id="ENT_A", target_entity_id="ENT_A",
        established_at_fabula=0,
        metrics={
            "fear": RelationshipMetric(
                value=0.7, inertia=0.5, evidence_strength="moderate",
                last_updated_fabula=0,
            ),
        },
    )
    s = compute_affective_scorers(_ws(social_topology=[re]))
    assert s["tension"] == 0.7


def test_ambivalence_picks_up_counter_concern_pair():
    ent = Entity(
        id="ENT_A", name="A", location_id="LOC_X", status="healthy",
        traits={"loyalty": TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")},
        concerns=[
            Concern(concern_id="CCN_FEAR", proposition_id="P_X",
                    polarity="fear", salience=0.8,
                    counter_concern_ids=["CCN_DESIRE"]),
            Concern(concern_id="CCN_DESIRE", proposition_id="P_X",
                    polarity="desire", salience=0.9,
                    counter_concern_ids=["CCN_FEAR"]),
        ],
    )
    s = compute_affective_scorers(_ws(entities={"ENT_A": ent}))
    # min(0.8, 0.9) = 0.8
    assert s["ambivalence"] == 0.8


def test_ambivalence_zero_when_no_counter_links():
    ent = Entity(
        id="ENT_A", name="A", location_id="LOC_X", status="healthy",
        traits={"loyalty": TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")},
        concerns=[
            Concern(concern_id="CCN_FEAR", proposition_id="P_X",
                    polarity="fear", salience=0.8),
            Concern(concern_id="CCN_DESIRE", proposition_id="P_X",
                    polarity="desire", salience=0.9),
        ],
    )
    s = compute_affective_scorers(_ws(entities={"ENT_A": ent}))
    assert s["ambivalence"] == 0.0
