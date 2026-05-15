# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the post-OSS-audit deterministic helpers.

Covers:

* :func:`shadow_loom.ingestion._mint_world_trait_propositions`
* :func:`shadow_loom.ingestion._auto_pair_ambivalent_concerns`
* :func:`shadow_loom.ingestion._promote_sentient_objects`

These are pure deterministic passes so we hand-build a minimal
``WorldStateV1`` per case rather than running the LLM stack.
"""

from __future__ import annotations

from shadow_loom.ingestion import (
    _auto_pair_ambivalent_concerns,
    _mint_world_trait_propositions,
    _promote_sentient_objects,
)
from shadow_loom.models import (
    Affordance,
    Belief,
    BeliefConfidenceShift,
    CausalEdge,
    Channel,
    Concern,
    Entity,
    EntityStateSnapshot,
    EventNode,
    GlobalTrait,
    Location,
    NarrativeObject,
    ObjectStateSnapshot,
    Proposition,
    RelationshipEdge,
    RelationshipMetric,
    TraitVector,
    WorldStateV1,
)


def _empty_world(**overrides) -> WorldStateV1:
    base = dict(
        locations={
            "LOC_X": Location(id="LOC_X", name="X", description="x"),
        },
        objects={},
        entities={},
        events=[],
        causal_topology=[],
    )
    base.update(overrides)
    return WorldStateV1(**base)


# ---------------------------------------------------------------------------
# _mint_world_trait_propositions
# ---------------------------------------------------------------------------


class TestMintWorldTraitPropositions:
    def test_mints_one_prop_per_trait_lacking_link(self):
        wt = GlobalTrait(
            id="WORLD_PANOPTICON",
            name="Total Surveillance",
            description="The state watches everyone.",
            category="governance",
            magnitude=TraitVector(value=0.8, inertia=0.6),
            affected_domains=["social"],
        )
        ws = _empty_world(world_traits={"WORLD_PANOPTICON": wt})

        n = _mint_world_trait_propositions(ws)

        assert n == 1
        assert len(ws.propositions) == 1
        prop = ws.propositions[0]
        assert prop.proposition_id == "PROP_WORLD_PANOPTICON"
        assert prop.kind == "trait_holds"
        assert prop.referent_ids == ["WORLD_PANOPTICON"]
        # magnitude.value=0.8 propagates to both prior and stakes.
        assert prop.audience_default_prior == 0.8
        assert prop.stakes == 0.8
        # Trait gets back-linked.
        assert ws.world_traits["WORLD_PANOPTICON"].proposition_id == (
            "PROP_WORLD_PANOPTICON"
        )

    def test_idempotent(self):
        wt = GlobalTrait(
            id="WORLD_W", name="W", description="w",
            category="governance",
            magnitude=TraitVector(value=0.5, inertia=0.5),
            affected_domains=["social"],
        )
        ws = _empty_world(world_traits={"WORLD_W": wt})

        first = _mint_world_trait_propositions(ws)
        second = _mint_world_trait_propositions(ws)

        assert first == 1
        assert second == 0
        assert len(ws.propositions) == 1

    def test_skips_already_linked(self):
        existing = Proposition(
            proposition_id="PROP_PRE",
            kind="trait_holds",
            referent_ids=["WORLD_W"],
            description="pre-existing",
        )
        wt = GlobalTrait(
            id="WORLD_W", name="W", description="w",
            category="governance",
            magnitude=TraitVector(value=0.5, inertia=0.5),
            affected_domains=["social"],
            proposition_id="PROP_PRE",
        )
        ws = _empty_world(
            world_traits={"WORLD_W": wt},
            propositions=[existing],
        )

        n = _mint_world_trait_propositions(ws)

        assert n == 0
        assert len(ws.propositions) == 1
        assert ws.propositions[0].proposition_id == "PROP_PRE"

    def test_back_links_when_prop_exists_under_canonical_id(self):
        # World trait has no link, but the canonical PROP_WORLD_<suffix>
        # is already in the registry — should back-link without re-minting.
        existing = Proposition(
            proposition_id="PROP_WORLD_W",
            kind="trait_holds",
            referent_ids=["WORLD_W"],
            description="already there",
        )
        wt = GlobalTrait(
            id="WORLD_W", name="W", description="w",
            category="governance",
            magnitude=TraitVector(value=0.5, inertia=0.5),
            affected_domains=["social"],
        )
        ws = _empty_world(
            world_traits={"WORLD_W": wt},
            propositions=[existing],
        )

        n = _mint_world_trait_propositions(ws)

        assert n == 0
        assert len(ws.propositions) == 1
        assert ws.world_traits["WORLD_W"].proposition_id == "PROP_WORLD_W"


# ---------------------------------------------------------------------------
# _auto_pair_ambivalent_concerns
# ---------------------------------------------------------------------------


class TestAutoPairAmbivalentConcerns:
    def _ent_with_concerns(self, *concerns: Concern) -> WorldStateV1:
        ent = Entity(
            id="ENT_E", name="E", location_id="LOC_X",
            status="healthy", traits={},
            concerns=list(concerns),
        )
        return _empty_world(entities={"ENT_E": ent})

    def test_pairs_opposite_polarity(self):
        a = Concern(
            concern_id="CCN_A", proposition_id="PROP_X",
            polarity="desire", salience=0.6,
        )
        b = Concern(
            concern_id="CCN_B", proposition_id="PROP_X",
            polarity="fear", salience=0.7,
        )
        ws = self._ent_with_concerns(a, b)

        n = _auto_pair_ambivalent_concerns(ws)

        assert n == 2
        cs = ws.entities["ENT_E"].concerns
        # Cross-linked.
        assert "CCN_B" in cs[0].counter_concern_ids
        assert "CCN_A" in cs[1].counter_concern_ids

    def test_skips_same_polarity(self):
        a = Concern(
            concern_id="CCN_A", proposition_id="PROP_X",
            polarity="desire", salience=0.6,
        )
        b = Concern(
            concern_id="CCN_B", proposition_id="PROP_X",
            polarity="desire", salience=0.7,
        )
        ws = self._ent_with_concerns(a, b)

        n = _auto_pair_ambivalent_concerns(ws)

        assert n == 0
        assert ws.entities["ENT_E"].concerns[0].counter_concern_ids == []

    def test_skips_groups_of_three_or_more(self):
        a = Concern(concern_id="CCN_A", proposition_id="PROP_X",
                    polarity="desire", salience=0.5)
        b = Concern(concern_id="CCN_B", proposition_id="PROP_X",
                    polarity="fear", salience=0.5)
        c = Concern(concern_id="CCN_C", proposition_id="PROP_X",
                    polarity="fear", salience=0.5)
        ws = self._ent_with_concerns(a, b, c)

        n = _auto_pair_ambivalent_concerns(ws)

        assert n == 0

    def test_skips_closed_concerns(self):
        # A concern with a capped activation_fabula_window upper bound
        # is considered closed and must not be paired.
        a = Concern(
            concern_id="CCN_A", proposition_id="PROP_X",
            polarity="desire", salience=0.5,
            activation_fabula_window=[0, 100],  # closed at 100
        )
        b = Concern(
            concern_id="CCN_B", proposition_id="PROP_X",
            polarity="fear", salience=0.5,
        )
        ws = self._ent_with_concerns(a, b)

        n = _auto_pair_ambivalent_concerns(ws)

        assert n == 0
        assert ws.entities["ENT_E"].concerns[1].counter_concern_ids == []

    def test_idempotent(self):
        a = Concern(concern_id="CCN_A", proposition_id="PROP_X",
                    polarity="desire", salience=0.5)
        b = Concern(concern_id="CCN_B", proposition_id="PROP_X",
                    polarity="fear", salience=0.5)
        ws = self._ent_with_concerns(a, b)

        first = _auto_pair_ambivalent_concerns(ws)
        second = _auto_pair_ambivalent_concerns(ws)

        assert first == 2
        assert second == 0


# ---------------------------------------------------------------------------
# _promote_sentient_objects
# ---------------------------------------------------------------------------


class TestPromoteSentientObjects:
    def _talking_object_world(self) -> WorldStateV1:
        # R2-D2 modelled as OBJ_ but speaks and acts on a non-utterance event.
        r2 = NarrativeObject(
            id="OBJ_R2D2", name="R2-D2", location_id="LOC_X",
            owner_id=None, properties={}, affordances=[],
        )
        leia = Entity(
            id="ENT_LEIA", name="Leia", location_id="LOC_X",
            status="healthy", traits={},
        )
        evt_action = EventNode(
            id="EVT_R2_DELIVERS",
            fabula_time=10, syuzhet_index=0,
            event_type="outcome",
            actor_ids=["OBJ_R2D2"],          # OBJ as actor → sentient
            target_ids=["ENT_LEIA"],
            description="R2-D2 delivers the message.",
        )
        evt_speech = EventNode(
            id="EVT_R2_BEEPS",
            fabula_time=11, syuzhet_index=1,
            event_type="utterance",
            actor_ids=["OBJ_R2D2"],
            target_ids=[],
            speaker_id="OBJ_R2D2",            # OBJ as speaker → sentient
            addressee_ids=["ENT_LEIA"],
            description="R2-D2 beeps urgently.",
            content="Help us.",
            truth_value="true",
        )
        return WorldStateV1(
            locations={
                "LOC_X": Location(id="LOC_X", name="X", description="x"),
            },
            objects={"OBJ_R2D2": r2},
            entities={"ENT_LEIA": leia},
            events=[evt_action, evt_speech],
            causal_topology=[],
        )

    def test_basic_promotion(self):
        ws = self._talking_object_world()

        ws, repairs = _promote_sentient_objects(ws)

        assert "ENT_R2D2" in ws.entities
        assert "OBJ_R2D2" not in ws.objects
        ent = ws.entities["ENT_R2D2"]
        assert ent.status == "healthy"
        assert ent.location_id == "LOC_X"
        # Cross-refs rewritten.
        assert ws.events[0].actor_ids == ["ENT_R2D2"]
        assert ws.events[1].speaker_id == "ENT_R2D2"
        assert ws.events[1].actor_ids == ["ENT_R2D2"]
        assert any("OBJ_R2D2" in r for r in repairs)

    def test_no_promotion_when_only_referenced_in_target_or_utterance_actor(self):
        # OBJ that is only a target / only an utterance actor is NOT sentient.
        sword = NarrativeObject(
            id="OBJ_SWORD", name="Sword", location_id="LOC_X",
            owner_id=None, properties={},
            affordances=[Affordance(action="wield", target_type="Entity")],
        )
        macbeth = Entity(
            id="ENT_MACBETH", name="Macbeth", location_id="LOC_X",
            status="healthy", traits={},
        )
        evt = EventNode(
            id="EVT_GRAB", fabula_time=10, syuzhet_index=0,
            event_type="outcome",
            actor_ids=["ENT_MACBETH"],
            target_ids=["OBJ_SWORD"],
            description="Macbeth grabs the sword.",
        )
        ws = WorldStateV1(
            locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
            objects={"OBJ_SWORD": sword},
            entities={"ENT_MACBETH": macbeth},
            events=[evt],
            causal_topology=[],
        )

        ws, repairs = _promote_sentient_objects(ws)

        assert "OBJ_SWORD" in ws.objects
        assert "ENT_SWORD" not in ws.entities
        assert repairs == []

    def test_idempotent(self):
        ws = self._talking_object_world()

        ws, first = _promote_sentient_objects(ws)
        ws, second = _promote_sentient_objects(ws)

        assert first  # first call did work
        assert second == []  # second is a no-op

    def test_rewrites_topology_channels_and_propositions(self):
        ws = self._talking_object_world()
        # Add a channel, a causal edge, a social edge, and a proposition
        # that all reference OBJ_R2D2.
        ws.channels["CHN_BOND"] = Channel(
            id="CHN_BOND", name="droid-link", medium="signal",
            participant_ids=["OBJ_R2D2", "ENT_LEIA"],
            intelligibility={"OBJ_R2D2": 1.0, "ENT_LEIA": 1.0},
            established_at_fabula=0,
        )
        ws.causal_topology.append(CausalEdge(
            source_id="EVT_R2_DELIVERS",
            target_id="OBJ_R2D2",
            causality_type="mutation",
            mechanism="physical",
            fabula_time=10,
            evidence_strength="strong",
        ))
        ws.social_topology.append(RelationshipEdge(
            source_entity_id="OBJ_R2D2",
            target_entity_id="ENT_LEIA",
            metrics={"affinity": RelationshipMetric(
                value=0.8, inertia=0.5, observed=True,
                evidence_strength="strong", last_updated_fabula=0,
            )},
        ))
        ws.propositions.append(Proposition(
            proposition_id="PROP_R2_LOYAL",
            kind="trait_holds",
            referent_ids=["OBJ_R2D2"],
            description="R2-D2 is loyal.",
        ))

        ws, _ = _promote_sentient_objects(ws)

        assert ws.channels["CHN_BOND"].participant_ids == ["ENT_R2D2", "ENT_LEIA"]
        assert ws.channels["CHN_BOND"].intelligibility == {
            "ENT_R2D2": 1.0, "ENT_LEIA": 1.0,
        }
        assert ws.causal_topology[0].target_id == "ENT_R2D2"
        assert ws.social_topology[0].source_entity_id == "ENT_R2D2"
        assert ws.propositions[0].referent_ids == ["ENT_R2D2"]

    def test_rewrites_object_owner_and_belief_history(self):
        # A second object owned by the promoted droid; a witness entity
        # whose timeline carries a belief invalidation against the OBJ id.
        ws = self._talking_object_world()
        toolbox = NarrativeObject(
            id="OBJ_TOOLBOX", name="Toolbox", location_id="LOC_X",
            owner_id="OBJ_R2D2", properties={},
            affordances=[],
            state_timeline=[ObjectStateSnapshot(
                fabula_time=5, owner_id="OBJ_R2D2",
            )],
        )
        ws.objects["OBJ_TOOLBOX"] = toolbox
        leia = ws.entities["ENT_LEIA"]
        leia.beliefs.append(Belief(
            target_id="OBJ_R2D2", perceived_state="trustworthy",
            confidence=0.9, inertia=0.5,
        ))
        leia.state_timeline.append(EntityStateSnapshot(
            fabula_time=12,
            beliefs_added=[Belief(
                target_id="OBJ_R2D2", perceived_state="brave",
                confidence=0.8, inertia=0.4,
            )],
            beliefs_invalidated=[
                "OBJ_R2D2",                       # bare form
                "OBJ_R2D2::PROP_R2_LOYAL",        # composite form
            ],
            belief_confidence_updates=[
                BeliefConfidenceShift(
                    target_id="OBJ_R2D2", new_confidence=0.6,
                ),
            ],
        ))

        ws, _ = _promote_sentient_objects(ws)

        # Owner refs (current + historical) updated.
        assert ws.objects["OBJ_TOOLBOX"].owner_id == "ENT_R2D2"
        assert (
            ws.objects["OBJ_TOOLBOX"].state_timeline[0].owner_id
            == "ENT_R2D2"
        )
        # Beliefs and belief-history updated.
        leia2 = ws.entities["ENT_LEIA"]
        assert leia2.beliefs[0].target_id == "ENT_R2D2"
        snap = leia2.state_timeline[0]
        assert snap.beliefs_added[0].target_id == "ENT_R2D2"
        assert "ENT_R2D2" in snap.beliefs_invalidated
        assert "ENT_R2D2::PROP_R2_LOYAL" in snap.beliefs_invalidated
        assert snap.belief_confidence_updates[0].target_id == "ENT_R2D2"

    def test_collision_falls_back_to_ent_from_prefix(self):
        # ENT_DROID already exists; promoting OBJ_DROID must use a
        # collision-safe id.
        existing = Entity(
            id="ENT_DROID", name="Existing droid", location_id="LOC_X",
            status="healthy", traits={},
        )
        droid = NarrativeObject(
            id="OBJ_DROID", name="Other droid", location_id="LOC_X",
            owner_id=None, properties={}, affordances=[],
        )
        evt = EventNode(
            id="EVT_DROID_ACTS", fabula_time=1, syuzhet_index=0,
            event_type="outcome",
            actor_ids=["OBJ_DROID"],
            target_ids=[],
            description="The other droid acts.",
        )
        ws = WorldStateV1(
            locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
            objects={"OBJ_DROID": droid},
            entities={"ENT_DROID": existing},
            events=[evt],
            causal_topology=[],
        )

        ws, _ = _promote_sentient_objects(ws)

        # The pre-existing ENT_DROID is intact; the promoted record
        # picked the ENT_FROM_<suffix> fallback.
        assert "ENT_DROID" in ws.entities
        assert "ENT_FROM_DROID" in ws.entities
        assert ws.events[0].actor_ids == ["ENT_FROM_DROID"]
