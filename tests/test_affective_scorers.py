# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for deterministic affective scorers (AUDIT P1-2)."""
from __future__ import annotations

from shadow_loom.affective_scorers import compute_affective_scorers
from shadow_loom.affect_unification import (
    AUDIENCE_ID,
    BeliefState,
    compute_surprise_unified,
    synthesise_audience_entity,
)
from shadow_loom.models import (
    Belief,
    Concern,
    Entity,
    EventNode,
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


def test_audience_learns_entity_anchored_proposition_via_truth_commits():
    """Regression: ``synthesise_audience_entity`` must emit audience
    beliefs for propositions anchored to *entities* (not events).

    A common authoring style attaches a proposition to the entities it
    is *about* (``referent_ids=[ENT_HERO, ENT_FOE]``) and records its
    ground truth in ``truth_at_fabula`` — no single event "carries" it.
    Pre-fix, the syuzhet-ordered belief loop only emitted beliefs for
    propositions reachable from a revealed event, so these
    entity-anchored propositions never produced an audience belief.
    Their confidence then sat frozen at ``audience_default_prior``
    (``_prop_audience_prior_at`` does not consult ``truth_at_fabula``),
    zeroing every surprise / mystery score on worlds authored this way
    (observed on brief_encounter and wuthering_heights, where 100% of
    propositions are entity-anchored).

    The fix emits one belief snapshot per truth commit, gated at the
    commit's own fabula tick, so the audience's confidence tracks the
    ground truth as the narrative reaches each fabula moment.
    """
    hero = Entity(id="ENT_HERO", name="Hero", location_id="LOC_X",
                  status="healthy", traits={})
    foe = Entity(id="ENT_FOE", name="Foe", location_id="LOC_X",
                 status="healthy", traits={})
    # Two events only give the fabula timeline / provenance anchors;
    # neither is referenced by the proposition under test.
    evt_early = EventNode(id="EVT_EARLY", description="they meet",
                          fabula_time=1000, syuzhet_index=0,
                          event_type="outcome")
    evt_late = EventNode(id="EVT_LATE", description="they part",
                         fabula_time=3000, syuzhet_index=1,
                         event_type="outcome")
    # Entity-anchored proposition: about the two entities, with a truth
    # that flips True -> False across the story.
    bond = Proposition(
        proposition_id="PROP_BOND", kind="relation_holds",
        description="The bond between Hero and Foe holds.",
        referent_ids=["ENT_HERO", "ENT_FOE"],
        audience_default_prior=0.5, stakes=0.9,
        truth_at_fabula={1000: True, 3000: False},
    )
    ws = WorldStateV1(
        entities={"ENT_HERO": hero, "ENT_FOE": foe},
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        objects={}, events=[evt_early, evt_late], causal_topology=[],
        propositions=[bond],
    )

    synthesise_audience_entity(ws)
    audience = ws.entities[AUDIENCE_ID]
    bond_beliefs = [
        b for snap in audience.state_timeline
        for b in snap.beliefs_added
        if b.proposition_id == "PROP_BOND"
    ]
    assert bond_beliefs, (
        "audience must hold beliefs about an entity-anchored proposition"
    )

    bs = BeliefState(world=ws)
    # Confidence tracks the ground-truth commits, not the frozen prior.
    assert bs.confidence(AUDIENCE_ID, "PROP_BOND", 1000) > 0.9
    assert bs.confidence(AUDIENCE_ID, "PROP_BOND", 3000) < 0.1

    # The True -> False flip is a genuine expectation violation: the
    # audience-belief movement must register as surprise.
    assert compute_surprise_unified(bs, 3000, 1000) > 0.0
