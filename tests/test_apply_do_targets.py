# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 1 — CausalPhysicsEngine.apply_do_targets dispatcher tests.

Covers DoEvent + DoTrait delegation to the legacy path, and the new
DoBelief / DoConcern / DoProposition (with belief cascade) surgeries.
"""
import pytest
import networkx as nx

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, Belief, Concern, Proposition,
    TraitVector,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.causal_physics import (
    CausalPhysicsEngine,
    PropositionMutation, BeliefMutation, ConcernMutation,
)
from shadow_loom.query_models import (
    DoEvent, DoTrait, DoProposition, DoBelief, DoConcern,
)


def _make_world() -> WorldStateV1:
    return WorldStateV1(
        locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={})},
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.2)},
                beliefs=[
                    Belief(
                        target_id="ENT_BOB",
                        perceived_state="Bob is loyal",
                        confidence=0.8,
                        evidence_strength="strong",
                        proposition_id="PROP_BOB_LOYAL",
                        inertia=0.3,
                    ),
                ],
                concerns=[
                    Concern(
                        concern_id="CCN_ALICE_FEARS_BETRAYAL",
                        proposition_id="PROP_BOB_LOYAL",
                        polarity="fear",
                        kind="betrayal",
                        salience=0.7,
                    ),
                ],
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB", name="Bob", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.7, inertia=0.3)},
                beliefs=[
                    Belief(
                        target_id="ENT_ALICE",
                        perceived_state="Alice trusts me",
                        confidence=0.6,
                        evidence_strength="strong",
                        proposition_id="PROP_BOB_LOYAL",
                        inertia=0.3,
                    ),
                ],
            ),
        },
        events=[
            EventNode(
                id="EVT_BETRAYAL", fabula_time=10, syuzhet_index=10,
                event_type="choice", actor_ids=["ENT_BOB"],
                target_ids=["ENT_ALICE"], description="Bob betrays Alice",
            ),
        ],
        causal_topology=[],
        spatial_topology=[],
        social_topology=[],
        propositions=[
            Proposition(
                proposition_id="PROP_BOB_LOYAL",
                description="Bob remains loyal to Alice",
                kind="trait_holds",
                referent_ids=["ENT_BOB", "ENT_ALICE"],
                truth_at_fabula={1: True},
            ),
        ],
    )


def _build_engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE", "ENT_BOB"])
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    return CausalPhysicsEngine(sandbox, ws)


class TestDoEventDelegation:
    def test_do_event_routes_through_legacy_path(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([DoEvent(event_id="EVT_BETRAYAL", occurred=False)])
        # Event should now be marked prevented
        assert eng.sandbox.has_node("EVT_BETRAYAL")
        et = eng.sandbox.nodes["EVT_BETRAYAL"].get("event_type")
        assert et == "prevented"
        assert "EVT_BETRAYAL" in eng._intervened_nodes


class TestDoTraitDelegation:
    def test_do_trait_routes_through_legacy_path(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([DoTrait(holder_id="ENT_ALICE", trait_name="courage", value=0.99)])
        assert ("ENT_ALICE", "courage") in eng._intervened_traits


class TestDoBelief:
    def test_clamps_existing_belief_confidence(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoBelief(holder_id="ENT_ALICE", target_id="ENT_BOB",
                     proposition_id="PROP_BOB_LOYAL", confidence=0.0),
        ])
        beliefs = eng.sandbox.nodes["ENT_ALICE"]["beliefs"]
        b = next(b for b in beliefs if b.get("proposition_id") == "PROP_BOB_LOYAL")
        assert b["confidence"] == 0.0
        assert b["acquired_via_event_id"] == "DO_OPERATOR"

        # Mutation recorded
        assert len(eng._belief_mutations) == 1
        m = eng._belief_mutations[0]
        assert isinstance(m, BeliefMutation)
        assert m.holder_id == "ENT_ALICE"
        assert m.new_confidence == 0.0
        assert m.created is False
        assert m.old_confidence == pytest.approx(0.8)

    def test_creates_belief_when_missing(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoBelief(holder_id="ENT_BOB", target_id="ENT_NEWCOMER",
                     perceived_state="Newcomer is hostile", confidence=1.0),
        ])
        beliefs = eng.sandbox.nodes["ENT_BOB"]["beliefs"]
        b = next(b for b in beliefs if b.get("target_id") == "ENT_NEWCOMER")
        assert b["confidence"] == 1.0
        assert b["perceived_state"] == "Newcomer is hostile"
        assert eng._belief_mutations[0].created is True

    def test_missing_holder_is_warned_not_raised(self):
        ws = _make_world()
        eng = _build_engine(ws)
        # Should silently skip (warning logged)
        eng.apply_do_targets([
            DoBelief(holder_id="ENT_GHOST", target_id="ENT_BOB", confidence=0.5),
        ])
        assert eng._belief_mutations == []


class TestDoConcern:
    def test_clamps_salience(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoConcern(holder_id="ENT_ALICE",
                      concern_id="CCN_ALICE_FEARS_BETRAYAL",
                      salience=0.0),
        ])
        concerns = eng.sandbox.nodes["ENT_ALICE"]["concerns"]
        c = next(c for c in concerns if c.get("concern_id") == "CCN_ALICE_FEARS_BETRAYAL")
        assert c["salience"] == 0.0
        assert any(m.field == "salience" for m in eng._concern_mutations)

    def test_disables_concern_via_active_false(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoConcern(holder_id="ENT_ALICE",
                      concern_id="CCN_ALICE_FEARS_BETRAYAL",
                      active=False),
        ])
        c = next(c for c in eng.sandbox.nodes["ENT_ALICE"]["concerns"]
                 if c["concern_id"] == "CCN_ALICE_FEARS_BETRAYAL")
        window = c["activation_fabula_window"]
        assert window is not None and window[0] == window[1]

    def test_polarity_flip(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoConcern(holder_id="ENT_ALICE",
                      concern_id="CCN_ALICE_FEARS_BETRAYAL",
                      polarity="desire"),
        ])
        c = next(c for c in eng.sandbox.nodes["ENT_ALICE"]["concerns"]
                 if c["concern_id"] == "CCN_ALICE_FEARS_BETRAYAL")
        assert c["polarity"] == "desire"

    def test_unknown_concern_skipped(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoConcern(holder_id="ENT_ALICE", concern_id="CCN_NONEXISTENT", salience=0.5),
        ])
        assert eng._concern_mutations == []


class TestDoProposition:
    def test_clamps_world_state_truth_at_fabula(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoProposition(proposition_id="PROP_BOB_LOYAL",
                          truth=False, fabula_time=10,
                          propagate_to_beliefs=False),
        ])
        prop = next(p for p in ws.propositions if p.proposition_id == "PROP_BOB_LOYAL")
        assert prop.truth_at_fabula[10] is False
        assert len(eng._proposition_mutations) == 1
        pm = eng._proposition_mutations[0]
        assert isinstance(pm, PropositionMutation)
        assert pm.cascaded_belief_count == 0

    def test_clamp_recorded_on_sandbox_graph(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoProposition(proposition_id="PROP_BOB_LOYAL", truth=False,
                          fabula_time=10, propagate_to_beliefs=False),
        ])
        clamps = eng.sandbox.graph.get("proposition_clamps", [])
        assert any(c["proposition_id"] == "PROP_BOB_LOYAL" and c["truth"] is False
                   for c in clamps)

    def test_cascades_into_beliefs_when_enabled(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoProposition(proposition_id="PROP_BOB_LOYAL", truth=False,
                          fabula_time=10, propagate_to_beliefs=True),
        ])
        # Both Alice's and Bob's beliefs reference PROP_BOB_LOYAL → both cascade.
        pm = eng._proposition_mutations[0]
        assert pm.cascaded_belief_count == 2

        # Cascaded belief mutations should attribute provenance to the proposition.
        cascaded = [m for m in eng._belief_mutations if m.triggered_by == "PROP_BOB_LOYAL"]
        assert len(cascaded) == 2

        # The clamped truth was False; both beliefs assert loyalty (affirming),
        # so confidence should be driven to 0 (scaled by evidence_strength).
        alice_b = next(b for b in eng.sandbox.nodes["ENT_ALICE"]["beliefs"]
                       if b.get("proposition_id") == "PROP_BOB_LOYAL")
        assert alice_b["confidence"] == 0.0


class TestMixedDispatch:
    def test_mixed_targets_apply_all_kinds(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoEvent(event_id="EVT_BETRAYAL", occurred=False),
            DoTrait(holder_id="ENT_ALICE", trait_name="courage", value=0.9),
            DoBelief(holder_id="ENT_ALICE", target_id="ENT_BOB",
                     proposition_id="PROP_BOB_LOYAL", confidence=0.1),
            DoConcern(holder_id="ENT_ALICE",
                      concern_id="CCN_ALICE_FEARS_BETRAYAL", salience=0.0),
            DoProposition(proposition_id="PROP_BOB_LOYAL", truth=False,
                          fabula_time=10, propagate_to_beliefs=False),
        ])
        assert eng.sandbox.nodes["EVT_BETRAYAL"]["event_type"] == "prevented"
        assert ("ENT_ALICE", "courage") in eng._intervened_traits
        assert any(m.holder_id == "ENT_ALICE" and m.new_confidence == 0.1
                   for m in eng._belief_mutations)
        assert any(m.field == "salience" for m in eng._concern_mutations)
        assert any(m.proposition_id == "PROP_BOB_LOYAL"
                   for m in eng._proposition_mutations)


class TestResultRoundTrip:
    def test_result_carries_new_mutation_fields(self):
        ws = _make_world()
        eng = _build_engine(ws)
        eng.apply_do_targets([
            DoBelief(holder_id="ENT_ALICE", target_id="ENT_BOB",
                     proposition_id="PROP_BOB_LOYAL", confidence=0.0),
        ])
        result = eng.execute(rung=2, interventions={})
        assert len(result.belief_mutations) == 1
        assert result.belief_mutations[0].new_confidence == 0.0
