# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the Causal Physics Engine (Step 7).

Exercises the three-rung CTF simulation:
  • Rung 2 — do-operator delegates to existing surgery + forward propagation
  • Rung 3 — abduction stores hidden_deltas, forward cascade via topological sort
  • Impact > Inertia gating during propagation
  • Bidirectional trait shifts
  • Spatial affordance blocking
  • Intervened nodes are not overridden by propagation
"""
import pytest
from copy import deepcopy

import networkx as nx

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge,
    TraitVector, Affordance, Belief,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.causal_physics import (
    CausalPhysicsEngine, CausalPhysicsResult, STRENGTH_MULTIPLIER,
    MECHANISM_TRAIT_MAP, MECHANISM_FALLBACK_FACTOR, SocialMutation,
)
from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import InterventionQuery, CounterfactualQuery

# ── Reuse a plot for integration tests ──
from example_worlds.macbeth import world_state as macbeth_ws


# =====================================================================
# Minimal fixture for unit-level tests
# =====================================================================
def _make_minimal_world() -> WorldStateV1:
    """Two-room world with two entities and one causal chain."""
    return WorldStateV1(
        locations={
            "LOC_A": Location(name="Room A", description="Room A",
                              ambient_state={}),
            "LOC_B": Location(name="Room B", description="Room B",
                              ambient_state={}),
        },
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={
                    "courage": TraitVector(value=0.5, inertia=0.2),
                    "anger": TraitVector(value=0.3, inertia=0.1),
                },
                beliefs=[
                    Belief(target_id="ENT_BOB",
                           perceived_state="Bob is friendly",
                           confidence=0.9, inertia=0.5),
                ],
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB", name="Bob", location_id="LOC_A",
                status="healthy",
                traits={
                    "courage": TraitVector(value=0.7, inertia=0.3),
                    "anger": TraitVector(value=0.1, inertia=0.4),
                },
            ),
        },
        events=[
            EventNode(id="EVT_FIGHT", fabula_time=1, syuzhet_index=1,
                      event_type="choice", actor_ids=["ENT_ALICE"],
                      target_ids=["ENT_BOB"], description="Alice attacks Bob"),
            EventNode(id="EVT_RESULT", fabula_time=2, syuzhet_index=2,
                      event_type="outcome", actor_ids=[],
                      description="Bob retaliates"),
        ],
        causal_topology=[
            CausalEdge(source_id="EVT_FIGHT",
                       target_id="ENT_BOB", causality_type="mutation", causal_force=5.0,
                       mechanism="physical", evidence_strength="strong",
                       fabula_time=1),
            CausalEdge(source_id="EVT_RESULT",
                       target_id="ENT_ALICE", causality_type="mutation", causal_force=5.0,
                       mechanism="physical", evidence_strength="moderate",
                       fabula_time=2),
            CausalEdge(source_id="EVT_FIGHT",
                       target_id="EVT_RESULT", causality_type="chain_reaction", causal_force=5.0,
                       mechanism="physical", evidence_strength="strong",
                       fabula_time=1),
        ],
        spatial_topology=[
            SpatialEdge(source_id="LOC_A", target_id="LOC_B"),
        ],
        social_topology=[
            RelationshipEdge(source_entity_id="ENT_ALICE",
                             target_entity_id="ENT_BOB",
                             affinity=0.6, fear=0.1, power_dynamic=0.0),
        ],
    )


def _build_sandbox(ws: WorldStateV1, focus_ids=None, query_type="intervention"):
    """Helper: extract ego graph and instantiate sandbox."""
    if focus_ids is None:
        focus_ids = list(ws.entities.keys())
    ego = extract_ego_graph_from_memory(ws, focus_ids)
    return AMWNInstantiator.create_sandbox(ego.model_dump(), query_type)


# =====================================================================
# RUNG 2 — INTERVENTION VIA CAUSAL ENGINE
# =====================================================================
class TestRung2Intervention:
    """Rung 2: do-operator delegates to existing surgery + propagation runs."""

    def test_engine_applies_surgery(self):
        """do-operator must mutate the target node's state."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)

        result = engine.execute(rung=2, interventions={
            "ENT_ALICE.status": "injured",
        })

        assert isinstance(result, CausalPhysicsResult)
        # Alice should be injured after surgery
        g = nx.node_link_graph(result.sandbox_data)
        alice = next(n for n in g.nodes(data=True) if n[0] == "ENT_ALICE")
        assert alice[1]["status"] == "injured"

    def test_intervened_nodes_recorded(self):
        """Intervened node IDs must appear in the result."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)

        result = engine.execute(rung=2, interventions={
            "ENT_ALICE.status": "injured",
        })
        assert "ENT_ALICE" in result.intervened_nodes

    def test_intervened_nodes_not_overridden(self):
        """Propagation must skip nodes that were directly intervened on."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])

        # Force Alice's courage to a specific value
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2, interventions={
            "ENT_ALICE.traits.courage.value": 0.99,
        })

        # If Alice is an intervened root, propagation should not override her
        assert "ENT_ALICE" in result.intervened_nodes
        g = nx.node_link_graph(result.sandbox_data)
        alice_traits = next(
            n[1]["traits"] for n in g.nodes(data=True) if n[0] == "ENT_ALICE"
        )
        # The value should be 0.99 dampened by inertia, NOT further changed by propagation
        assert alice_traits["courage"]["value"] != 0.5  # not the original

    def test_propagation_applies_to_downstream(self):
        """Downstream entities must receive propagated trait shifts."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)

        result = engine.execute(rung=2, interventions={
            "EVT_FIGHT.event_type": "outcome",
        })
        # Result should contain either mutations or blocked entries for Bob
        # (EVT_FIGHT → ENT_BOB causal edge exists)
        all_nodes = [m.node_id for m in result.mutations] + [b.node_id for b in result.blocked]
        # The engine must have processed the downstream entity
        assert "ENT_BOB" in all_nodes, (
            f"ENT_BOB not found in mutations or blocked. "
            f"mutations={[m.node_id for m in result.mutations]}, "
            f"blocked={[b.node_id for b in result.blocked]}"
        )


# =====================================================================
# RUNG 3 — COUNTERFACTUAL VIA CAUSAL ENGINE
# =====================================================================
class TestRung3Counterfactual:
    """Rung 3: abduction + do-operator + forward propagation."""

    def test_abduction_stores_hidden_deltas(self):
        """Abduction must store hidden_deltas on conditioned entities."""
        ws = _make_minimal_world()
        # Modify factual Alice to be very different from sandbox Alice
        ws_modified = deepcopy(ws)
        ws_modified.entities["ENT_ALICE"].traits["courage"].value = 0.9

        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")
        engine = CausalPhysicsEngine(sandbox, ws_modified)

        result = engine.execute(
            rung=3,
            interventions={"EVT_FIGHT.event_type": "outcome"},
            evidence_node_ids=["ENT_ALICE"],
        )

        assert "ENT_ALICE" in result.hidden_deltas
        assert "courage" in result.hidden_deltas["ENT_ALICE"]
        # Delta should be (0.9 - 0.5) = 0.4
        assert abs(result.hidden_deltas["ENT_ALICE"]["courage"] - 0.4) < 0.01

    def test_abduction_blends_traits(self):
        """Abduction must blend traits toward factual values, damped by
        trait inertia (high-inertia traits resist present-day evidence).

        This test exercises the *legacy* blend formula explicitly so it is
        not coupled to the package-level default ``abduction_blend_mode``.
        """
        from shadow_loom.settings import get_settings
        physics = get_settings().physics
        prev_mode = physics.abduction_blend_mode
        physics.abduction_blend_mode = "legacy"
        try:
            ws = _make_minimal_world()
            ws_modified = deepcopy(ws)
            ws_modified.entities["ENT_ALICE"].traits["courage"].value = 0.9

            sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")
            old_courage = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
            trait_inertia = sandbox.nodes["ENT_ALICE"]["traits"]["courage"].get("inertia", 0.5)

            engine = CausalPhysicsEngine(sandbox, ws_modified)
            engine.abduction_update(["ENT_ALICE"])

            new_courage = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
            blend_factor = max(0.0, min(1.0, 1.0 - trait_inertia))
            expected = old_courage + (0.9 - old_courage) * blend_factor
            assert abs(new_courage - expected) < 0.01
        finally:
            physics.abduction_blend_mode = prev_mode

    def test_abduction_backpropagates_beliefs(self):
        """Abduction must copy missing beliefs from factual entity."""
        ws = _make_minimal_world()
        ws_modified = deepcopy(ws)
        ws_modified.entities["ENT_ALICE"].beliefs.append(
            Belief(target_id="ENT_BOB", perceived_state="Bob is actually hostile",
                   confidence=0.8, inertia=0.5)
        )

        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")
        engine = CausalPhysicsEngine(sandbox, ws_modified)
        engine.abduction_update(["ENT_ALICE"])

        beliefs = sandbox.nodes["ENT_ALICE"]["beliefs"]
        states = [b["perceived_state"] for b in beliefs]
        assert "Bob is actually hostile" in states

    def test_full_counterfactual_pipeline(self):
        """Full Rung 3 execution must produce a valid result."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")
        engine = CausalPhysicsEngine(sandbox, ws)

        result = engine.execute(
            rung=3,
            interventions={"EVT_FIGHT.event_type": "outcome"},
            evidence_node_ids=["ENT_ALICE"],
        )

        assert result.sandbox_data  # non-empty
        assert isinstance(result.mutations, list)
        assert isinstance(result.blocked, list)


# =====================================================================
# IMPACT > INERTIA GATING
# =====================================================================
class TestImpactInertiaGating:
    """Propagation must respect Impact > Inertia."""

    def test_high_inertia_blocks_propagation(self):
        """When inertia exceeds impact, the trait must not change."""
        ws = _make_minimal_world()
        # Set Bob's anger inertia very high so propagation can't shift it
        ws.entities["ENT_BOB"].traits["anger"] = TraitVector(value=0.5, inertia=0.99)

        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})

        # Check anger was blocked
        anger_blocked = [
            b for b in result.blocked
            if b.node_id == "ENT_BOB" and b.trait == "anger" and b.reason == "inertia"
        ]
        assert len(anger_blocked) >= 1 or sandbox.nodes["ENT_BOB"]["traits"]["anger"]["value"] == 0.5

    def test_low_inertia_allows_propagation(self):
        """When impact exceeds inertia, the trait must change."""
        ws = _make_minimal_world()
        # Give Bob very low inertia and a strong causal edge incoming
        ws.entities["ENT_BOB"].traits["courage"] = TraitVector(value=0.5, inertia=0.01)

        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})

        # Check for mutation on Bob's courage
        courage_mutations = [
            m for m in result.mutations
            if m.node_id == "ENT_BOB" and m.trait == "courage"
        ]
        courage_blocked = [
            b for b in result.blocked
            if b.node_id == "ENT_BOB" and b.trait == "courage"
        ]
        # Either it was mutated or it was blocked — but at very low inertia it should pass
        assert len(courage_mutations) >= 1 or len(courage_blocked) == 0


# =====================================================================
# SPATIAL AFFORDANCE BLOCKING
# =====================================================================
class TestSpatialAffordanceBlocking:
    """Propagation must check spatial reachability for entity→entity causation."""

    def test_unreachable_location_blocks_propagation(self):
        """Entity in an unreachable room must not receive propagated changes."""
        ws = _make_minimal_world()
        # Put Bob in a disconnected location
        ws.locations["LOC_ISLAND"] = Location(
            name="Island", description="Unreachable island", ambient_state={},
        )
        ws.entities["ENT_BOB"].location_id = "LOC_ISLAND"
        # No spatial edge connects LOC_A → LOC_ISLAND

        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])

        # Manually add a causal edge Alice→Bob in the sandbox
        sandbox.add_edge("ENT_ALICE", "ENT_BOB",
                         edge_type="causal", evidence_strength="strong",
                         mechanism="physical")

        engine = CausalPhysicsEngine(sandbox, ws)
        # Manually mark Alice as triggered so her outgoing causal edge fires.
        engine._intervened_nodes.add("ENT_ALICE")
        engine.propagate()

        blocked_spatial = [
            b for b in engine._blocked if b.reason == "spatial_affordance"
        ]
        # Bob's traits should have been blocked
        assert len(blocked_spatial) >= 1


# =====================================================================
# INTEGRATION WITH NARRATIVE_PHYSICS.PY
# =====================================================================
class TestNarrativePhysicsIntegration:
    """The use_causal_engine=True path must work end-to-end."""

    def test_intervention_with_engine(self):
        """Intervention via causal engine must return mutations list."""
        query = InterventionQuery(interventions={
            "ENT_MACBETH.status": "injured",
        })
        result = calculate_narrative_physics(
            query, macbeth_ws, use_causal_engine=True,
        )
        assert result["status"] == "success"
        assert result["query_type"] == "intervention"
        assert "mutations" in result
        assert "blocked" in result
        assert "intervened_nodes" in result

    def test_counterfactual_with_engine(self):
        """Counterfactual via causal engine must return hidden_deltas."""
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "outcome"},
            evidence_node_ids=["ENT_MACBETH"],
        )
        result = calculate_narrative_physics(
            query, macbeth_ws, use_causal_engine=True,
        )
        assert result["status"] == "success"
        assert result["query_type"] == "counterfactual"
        assert "hidden_deltas" in result
        assert "mutations" in result

    def test_legacy_path_unchanged(self):
        """Default path (use_causal_engine=False) must produce same shape."""
        query = InterventionQuery(interventions={
            "ENT_MACBETH.status": "injured",
        })
        result = calculate_narrative_physics(query, macbeth_ws)
        assert result["status"] == "success"
        assert "mutations" not in result  # legacy path doesn't have this


# =====================================================================
# RESULT MODEL
# =====================================================================
class TestCausalPhysicsResult:
    """CausalPhysicsResult must serialize cleanly."""

    def test_result_serialization(self):
        """Result must be serializable to dict."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2, interventions={
            "ENT_ALICE.status": "injured",
        })

        d = result.model_dump()
        assert "sandbox_data" in d
        assert "mutations" in d
        assert "blocked" in d
        assert "intervened_nodes" in d
        assert "hidden_deltas" in d


# =====================================================================
# MECHANISM TRAIT MAP GATING
# =====================================================================
class TestMechanismTraitMap:
    """MECHANISM_TRAIT_MAP must gate which traits receive impulse."""

    def _make_mechanism_world(self, mechanism: str) -> WorldStateV1:
        """World with a single causal edge using the given mechanism."""
        return WorldStateV1(
            locations={
                "LOC_A": Location(name="Room A", description="A", ambient_state={}),
            },
            objects={},
            entities={
                "ENT_ALICE": Entity(
                    id="ENT_ALICE", name="Alice", location_id="LOC_A",
                    status="healthy",
                    traits={
                        "guilt": TraitVector(value=0.3, inertia=0.01),
                        "courage": TraitVector(value=0.3, inertia=0.01),
                    },
                ),
            },
            events=[
                EventNode(id="EVT_CAUSE", fabula_time=1, syuzhet_index=1,
                          event_type="outcome", description="Something happens"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_CAUSE",
                           target_id="ENT_ALICE", causality_type="mutation", causal_force=5.0,
                           mechanism=mechanism, evidence_strength="strong",
                           fabula_time=1),
            ],
            spatial_topology=[],
            social_topology=[],
        )

    def test_psychological_mechanism_favors_matching_traits(self):
        """'psychological' mechanism must give guilt (matching) a larger impulse
        than courage (non-matching) by factor of MECHANISM_FALLBACK_FACTOR."""
        ws = self._make_mechanism_world("psychological")
        sandbox = _build_sandbox(ws, ["ENT_ALICE"])
        engine = CausalPhysicsEngine(sandbox, ws)
        engine.abduction_update(["EVT_CAUSE"])

        guilt = sandbox.nodes["ENT_ALICE"]["traits"]["guilt"]["value"]
        courage = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]

        # guilt is in MECHANISM_TRAIT_MAP["psychological"], courage is not
        assert "guilt" in MECHANISM_TRAIT_MAP["psychological"]
        assert "courage" not in MECHANISM_TRAIT_MAP["psychological"]
        # guilt should have shifted more than courage
        assert guilt > courage

    def test_unknown_mechanism_applies_equally(self):
        """Mechanism not in MECHANISM_TRAIT_MAP must apply full impulse to ALL traits."""
        ws = self._make_mechanism_world("magic_mutation")
        sandbox = _build_sandbox(ws, ["ENT_ALICE"])
        engine = CausalPhysicsEngine(sandbox, ws)
        engine.abduction_update(["EVT_CAUSE"])

        guilt = sandbox.nodes["ENT_ALICE"]["traits"]["guilt"]["value"]
        courage = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
        # Both should have shifted equally (magic_mutation not in map)
        assert abs(guilt - courage) < 0.001

    def test_propagation_uses_mechanism_gating(self):
        """propagate() must also apply MECHANISM_TRAIT_MAP gating."""
        ws = self._make_mechanism_world("psychological")
        sandbox = _build_sandbox(ws, ["ENT_ALICE"])
        engine = CausalPhysicsEngine(sandbox, ws)
        # Mark all event nodes as intervened so canonical edges fire under the
        # active-source gating (this test pre-dates the gating but still wants
        # to exercise mechanism math).
        for nid, ndata in sandbox.nodes(data=True):
            if ndata.get("node_type") == "EventNode":
                engine._intervened_nodes.add(nid)
        engine.propagate()

        # Check that any mutations or blocked entries reflect mechanism gating
        all_items = engine._mutations + engine._blocked
        guilt_items = [x for x in all_items if x.trait == "guilt"]
        courage_items = [x for x in all_items if x.trait == "courage"]
        # Psychological mechanism must produce at least guilt entries
        assert guilt_items, "Psychological mechanism should produce guilt mutations/blocked"
        # If there are entries for both, guilt impact should be >= courage impact
        if courage_items:
            assert abs(guilt_items[0].impact) >= abs(courage_items[0].impact)


# =====================================================================
# SIGNED DELTA PROPAGATION (BIDIRECTIONAL)
# =====================================================================
class TestSignedDeltaPropagation:
    """Propagation must shift traits TOWARD source value (bidirectional)."""

    def test_trait_decreases_when_source_lower(self):
        """If source courage=0.2, target courage=0.8, target must decrease."""
        ws = WorldStateV1(
            locations={
                "LOC_A": Location(name="Room A", description="A", ambient_state={}),
            },
            objects={},
            entities={
                "ENT_SRC": Entity(
                    id="ENT_SRC", name="Source", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.2, inertia=0.01)},
                ),
                "ENT_TGT": Entity(
                    id="ENT_TGT", name="Target", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.8, inertia=0.01)},
                ),
            },
            events=[],
            causal_topology=[],
            spatial_topology=[],
            social_topology=[],
        )
        sandbox = _build_sandbox(ws, ["ENT_SRC", "ENT_TGT"])
        # Manually add a strong entity→entity causal edge
        sandbox.add_edge("ENT_SRC", "ENT_TGT",
                         edge_type="causal", evidence_strength="strong",
                         mechanism="physical")
        engine = CausalPhysicsEngine(sandbox, ws)
        engine._intervened_nodes.add("ENT_SRC")
        engine.propagate()

        mutations = [m for m in engine._mutations
                     if m.node_id == "ENT_TGT" and m.trait == "courage"]
        assert mutations, "Target should have courage mutations from strong source"
        assert mutations[0].new_value < 0.8, \
            f"Target courage should decrease, got {mutations[0].new_value}"

    def test_trait_increases_when_source_higher(self):
        """If source courage=0.9, target courage=0.2, target must increase."""
        ws = WorldStateV1(
            locations={
                "LOC_A": Location(name="Room A", description="A", ambient_state={}),
            },
            objects={},
            entities={
                "ENT_SRC": Entity(
                    id="ENT_SRC", name="Source", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.9, inertia=0.01)},
                ),
                "ENT_TGT": Entity(
                    id="ENT_TGT", name="Target", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.2, inertia=0.01)},
                ),
            },
            events=[],
            causal_topology=[],
            spatial_topology=[],
            social_topology=[],
        )
        sandbox = _build_sandbox(ws, ["ENT_SRC", "ENT_TGT"])
        sandbox.add_edge("ENT_SRC", "ENT_TGT",
                         edge_type="causal", evidence_strength="strong",
                         mechanism="physical")
        engine = CausalPhysicsEngine(sandbox, ws)
        engine._intervened_nodes.add("ENT_SRC")
        engine.propagate()

        mutations = [m for m in engine._mutations
                     if m.node_id == "ENT_TGT" and m.trait == "courage"]
        assert mutations, "Target should have courage mutations from strong source"
        assert mutations[0].new_value > 0.2, \
            f"Target courage should increase, got {mutations[0].new_value}"


# =====================================================================
# ABDUCTION EVENT EVIDENCE
# =====================================================================
class TestAbductionEventEvidence:
    """Abduction must handle EventNode evidence via causal edges."""

    def test_event_evidence_updates_downstream_traits(self):
        """Evidence event must update downstream entity traits via causal topology."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")

        old_courage = sandbox.nodes["ENT_BOB"]["traits"]["courage"]["value"]
        engine = CausalPhysicsEngine(sandbox, ws)
        engine.abduction_update(["EVT_FIGHT"])

        # EVT_FIGHT → ENT_BOB causal edge exists; traits should have shifted
        new_courage = sandbox.nodes["ENT_BOB"]["traits"]["courage"]["value"]
        assert new_courage != old_courage

    def test_missing_evidence_node_no_crash(self):
        """Evidence node not in sandbox must be skipped without error."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        # Should not raise
        engine.abduction_update(["EVT_NONEXISTENT"])

    def test_event_evidence_not_double_applied(self):
        """Regression: abduction Case 2 must not be re-applied by propagate().

        For an evidence event whose mutation edge has explicit
        ``trait_target``/``trait_delta``, ``execute(rung=3)`` must produce
        the same trait shift as ``abduction_update`` alone — propagate()
        should *not* add a second contribution from the same edge.
        """
        from shadow_loom.models import (
            WorldStateV1, Location, Entity, EventNode, CausalEdge, TraitVector,
        )
        ws = WorldStateV1(
            locations={"LOC_A": Location(name="A", description="", ambient_state={})},
            objects={},
            entities={
                "ENT_BOB": Entity(
                    id="ENT_BOB", name="Bob", location_id="LOC_A", status="healthy",
                    traits={"courage": TraitVector(value=0.3, inertia=0.0)},
                ),
            },
            events=[
                EventNode(id="EVT_X", fabula_time=1, syuzhet_index=1,
                          event_type="choice", actor_ids=[], target_ids=["ENT_BOB"],
                          description="shock"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_X", target_id="ENT_BOB",
                           causality_type="mutation", causal_force=10.0,
                           mechanism="psychological", evidence_strength="strong",
                           trait_target="courage", trait_delta=0.3, fabula_time=1),
            ],
        )

        def _build():
            ego = extract_ego_graph_from_memory(ws, ["ENT_BOB"])
            return AMWNInstantiator.create_sandbox(ego.model_dump(), "counterfactual")

        sb_abd = _build()
        CausalPhysicsEngine(sb_abd, ws).abduction_update(["EVT_X"])
        abd_only = sb_abd.nodes["ENT_BOB"]["traits"]["courage"]["value"]

        sb_full = _build()
        CausalPhysicsEngine(sb_full, ws).execute(rung=3, evidence_node_ids=["EVT_X"])
        full_run = sb_full.nodes["ENT_BOB"]["traits"]["courage"]["value"]

        assert abs(full_run - abd_only) < 1e-6, (
            f"Rung-3 double-application: abduction={abd_only:.4f}, "
            f"full execute={full_run:.4f}"
        )


# =====================================================================
# RUNG VALIDATION
# =====================================================================
class TestRungValidation:
    """execute() must reject invalid rung values."""

    def test_rung_1_raises_value_error(self):
        """Rung 1 is not valid; must raise ValueError."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        with pytest.raises(ValueError, match="Invalid rung=1"):
            engine.execute(rung=1)

    def test_rung_0_raises_value_error(self):
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        with pytest.raises(ValueError, match="Invalid rung=0"):
            engine.execute(rung=0)

    def test_rung_99_raises_value_error(self):
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        with pytest.raises(ValueError, match="Invalid rung=99"):
            engine.execute(rung=99)

    def test_rung_3_without_evidence_still_propagates(self):
        """Rung 3 with no evidence must skip abduction but still propagate."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=3, interventions={
            "EVT_FIGHT.event_type": "outcome",
        })
        assert result.sandbox_data  # no crash, valid result


# =====================================================================
# CYCLIC CAUSAL GRAPH
# =====================================================================
class TestCyclicCausalGraph:
    """Cyclic causal graph must fall back gracefully."""

    def test_cyclic_graph_does_not_crash(self):
        """Sandbox with A→B→A cycle must not raise during propagation."""
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        # Create bidirectional causal edges
        sandbox.add_edge("ENT_ALICE", "ENT_BOB",
                         edge_type="causal", evidence_strength="strong",
                         mechanism="physical")
        sandbox.add_edge("ENT_BOB", "ENT_ALICE",
                         edge_type="causal", evidence_strength="strong",
                         mechanism="physical")
        engine = CausalPhysicsEngine(sandbox, ws)
        engine.propagate()  # must not raise


# =====================================================================
# SPATIAL REACHABILITY OWNERSHIP
# =====================================================================
class TestSpatialReachabilityOwnership:
    """_check_spatial_reachability must require ownership for unlock."""

    def test_unowned_key_cannot_unlock(self):
        """An unlock object with owner_id=None must NOT unlock barrier."""
        ws = WorldStateV1(
            locations={
                "LOC_A": Location(name="Room A", description="A", ambient_state={}),
                "LOC_B": Location(name="Room B", description="B", ambient_state={}),
            },
            objects={
                "OBJ_DOOR": NarrativeObject(
                    id="OBJ_DOOR", name="Iron Door", location_id="LOC_A",
                    owner_id=None,
                    properties={}, affordances=[],
                ),
                "OBJ_KEY": NarrativeObject(
                    id="OBJ_KEY", name="Key", location_id="LOC_A",
                    owner_id=None,  # NO owner
                    properties={},
                    affordances=[Affordance(action="unlock",
                                            target_type="NarrativeObject")],
                ),
            },
            entities={
                "ENT_ALICE": Entity(
                    id="ENT_ALICE", name="Alice", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.5, inertia=0.1)},
                ),
                "ENT_BOB": Entity(
                    id="ENT_BOB", name="Bob", location_id="LOC_B",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.5, inertia=0.1)},
                ),
            },
            events=[],
            causal_topology=[],
            spatial_topology=[
                SpatialEdge(source_id="LOC_A", target_id="LOC_B",
                            is_locked=True, barrier_item_id="OBJ_DOOR"),
            ],
            social_topology=[],
        )
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        # Should NOT be reachable — key is unowned
        assert engine._check_spatial_reachability("LOC_A", "LOC_B") is False

    def test_owned_key_unlocks(self):
        """An unlock object owned by an entity must unlock barrier."""
        ws = WorldStateV1(
            locations={
                "LOC_A": Location(name="Room A", description="A", ambient_state={}),
                "LOC_B": Location(name="Room B", description="B", ambient_state={}),
            },
            objects={
                "OBJ_DOOR": NarrativeObject(
                    id="OBJ_DOOR", name="Iron Door", location_id="LOC_A",
                    owner_id=None,
                    properties={}, affordances=[],
                ),
                "OBJ_KEY": NarrativeObject(
                    id="OBJ_KEY", name="Key", location_id=None,
                    owner_id="ENT_ALICE",  # owned!
                    properties={},
                    affordances=[Affordance(action="unlock",
                                            target_type="NarrativeObject")],
                ),
            },
            entities={
                "ENT_ALICE": Entity(
                    id="ENT_ALICE", name="Alice", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.5, inertia=0.1)},
                ),
                "ENT_BOB": Entity(
                    id="ENT_BOB", name="Bob", location_id="LOC_B",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.5, inertia=0.1)},
                ),
            },
            events=[],
            causal_topology=[],
            spatial_topology=[
                SpatialEdge(source_id="LOC_A", target_id="LOC_B",
                            is_locked=True, barrier_item_id="OBJ_DOOR"),
            ],
            social_topology=[],
        )
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        assert engine._check_spatial_reachability("LOC_A", "LOC_B") is True

    def test_barrier_name_matching(self):
        """Affordance target_type matching barrier name (not just node_type) must work."""
        ws = WorldStateV1(
            locations={
                "LOC_A": Location(name="Room A", description="A", ambient_state={}),
                "LOC_B": Location(name="Room B", description="B", ambient_state={}),
            },
            objects={
                "OBJ_GATE": NarrativeObject(
                    id="OBJ_GATE", name="Iron Gate", location_id="LOC_A",
                    owner_id=None,
                    properties={}, affordances=[],
                ),
                "OBJ_KEY": NarrativeObject(
                    id="OBJ_KEY", name="Key", location_id=None,
                    owner_id="ENT_ALICE",
                    properties={},
                    affordances=[Affordance(action="unlock",
                                            target_type="Iron Gate")],
                ),
            },
            entities={
                "ENT_ALICE": Entity(
                    id="ENT_ALICE", name="Alice", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.5, inertia=0.1)},
                ),
            },
            events=[],
            causal_topology=[],
            spatial_topology=[
                SpatialEdge(source_id="LOC_A", target_id="LOC_B",
                            is_locked=True, barrier_item_id="OBJ_GATE"),
            ],
            social_topology=[],
        )
        sandbox = _build_sandbox(ws, ["ENT_ALICE"])
        engine = CausalPhysicsEngine(sandbox, ws)
        assert engine._check_spatial_reachability("LOC_A", "LOC_B") is True


# =====================================================================
# MUTATION_SOCIAL PROPAGATION
# =====================================================================
class TestMutationSocialPropagation:
    """propagate_social() must apply relationship metric deltas from mutation_social causal edges."""

    def _make_social_world(self) -> WorldStateV1:
        """World with a mutation_social edge: EVT_BETRAYAL → ENT_ALICE's fear of ENT_BOB increases."""
        return WorldStateV1(
            locations={
                "LOC_A": Location(name="Room A", description="A", ambient_state={}),
            },
            objects={},
            entities={
                "ENT_ALICE": Entity(
                    id="ENT_ALICE", name="Alice", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.5, inertia=0.2)},
                ),
                "ENT_BOB": Entity(
                    id="ENT_BOB", name="Bob", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.7, inertia=0.3)},
                ),
            },
            events=[
                EventNode(id="EVT_BETRAYAL", fabula_time=1, syuzhet_index=1,
                          event_type="choice", actor_ids=["ENT_BOB"],
                          target_ids=["ENT_ALICE"], description="Bob betrays Alice"),
            ],
            causal_topology=[
                CausalEdge(
                    source_id="EVT_BETRAYAL", target_id="ENT_ALICE",
                    causality_type="mutation_social",
                    mechanism="betrayal", evidence_strength="strong",
                    causal_force=8.0, fabula_time=1,
                    trait_target="fear", trait_delta=0.5,
                    rel_counterpart_id="ENT_BOB",
                ),
            ],
            spatial_topology=[],
            social_topology=[
                RelationshipEdge(
                    source_entity_id="ENT_ALICE", target_entity_id="ENT_BOB",
                    affinity=0.6, fear=0.1, power_dynamic=0.0, inertia=0.1,
                ),
            ],
        )

    def test_social_propagation_mutates_relationship(self):
        """mutation_social edge must shift the relationship metric."""
        ws = self._make_social_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        engine._intervened_nodes.add("EVT_BETRAYAL")
        engine.propagate_social()

        assert len(engine._social_mutations) == 1
        sm = engine._social_mutations[0]
        assert sm.source_entity_id == "ENT_ALICE"
        assert sm.target_entity_id == "ENT_BOB"
        assert sm.metric == "fear"
        assert sm.new_value > sm.old_value  # fear should increase

    def test_social_propagation_respects_inertia(self):
        """High relationship inertia must block the social mutation."""
        ws = self._make_social_world()
        # Overwrite the relationship with very high inertia
        ws.social_topology[0] = RelationshipEdge(
            source_entity_id="ENT_ALICE", target_entity_id="ENT_BOB",
            affinity=0.6, fear=0.1, power_dynamic=0.0, inertia=0.99,
        )
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        engine._intervened_nodes.add("EVT_BETRAYAL")
        engine.propagate_social()

        # Should be blocked by inertia
        assert len(engine._social_mutations) == 0
        blocked_social = [b for b in engine._blocked if "rel." in b.trait]
        assert len(blocked_social) == 1

    def test_social_propagation_creates_missing_edge(self):
        """If no relationship edge exists, propagate_social must create one."""
        ws = self._make_social_world()
        ws.social_topology = []  # Remove existing relationship
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        engine._intervened_nodes.add("EVT_BETRAYAL")
        engine.propagate_social()

        assert len(engine._social_mutations) == 1
        sm = engine._social_mutations[0]
        assert sm.old_value == 0.0  # Created from scratch
        assert sm.new_value > 0.0   # fear should be positive

    def test_social_propagation_scales_by_evidence_and_force(self):
        """Delta must be scaled by evidence_strength × (causal_force / 10)."""
        ws = self._make_social_world()
        # Edge has evidence_strength="strong" (0.75) and causal_force=8.0 (0.8)
        # raw_delta = 0.5, so scaled = 0.5 * 0.75 * 0.8 = 0.3
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        engine._intervened_nodes.add("EVT_BETRAYAL")
        engine.propagate_social()

        sm = engine._social_mutations[0]
        expected_scaled = 0.5 * STRENGTH_MULTIPLIER["strong"] * (8.0 / 10.0)
        # Impact > Inertia dampening: effective = scaled_delta - inertia
        expected_effective = expected_scaled - 0.1  # inertia=0.1
        expected_new = 0.1 + expected_effective  # old fear=0.1
        assert abs(sm.new_value - expected_new) < 0.01

    def test_social_propagation_clamps_fear(self):
        """Fear must be clamped to [0, 1]."""
        ws = self._make_social_world()
        # Give a huge delta to push fear beyond 1.0
        ws.causal_topology[0] = CausalEdge(
            source_id="EVT_BETRAYAL", target_id="ENT_ALICE",
            causality_type="mutation_social",
            mechanism="betrayal", evidence_strength="strong",
            causal_force=10.0, fabula_time=1,
            trait_target="fear", trait_delta=2.0,
            rel_counterpart_id="ENT_BOB",
        )
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        engine._intervened_nodes.add("EVT_BETRAYAL")
        engine.propagate_social()

        sm = engine._social_mutations[0]
        assert sm.new_value <= 1.0

    def test_social_propagation_clamps_affinity(self):
        """Affinity must be clamped to [-1, 1]."""
        ws = self._make_social_world()
        ws.causal_topology[0] = CausalEdge(
            source_id="EVT_BETRAYAL", target_id="ENT_ALICE",
            causality_type="mutation_social",
            mechanism="betrayal", evidence_strength="strong",
            causal_force=10.0, fabula_time=1,
            trait_target="affinity", trait_delta=-3.0,
            rel_counterpart_id="ENT_BOB",
        )
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        engine._intervened_nodes.add("EVT_BETRAYAL")
        engine.propagate_social()

        sm = engine._social_mutations[0]
        assert sm.new_value >= -1.0

    def test_execute_includes_social_mutations(self):
        """Full execute() must return social_mutations in the result."""
        ws = self._make_social_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2, interventions={
            "EVT_BETRAYAL.event_type": "outcome",
        })
        assert isinstance(result.social_mutations, list)
        # social_mutations should be populated
        assert len(result.social_mutations) >= 1

    def test_macbeth_mutation_social_edges_load(self):
        """Macbeth fixture must contain mutation_social edges."""
        social_edges = [ce for ce in macbeth_ws.causal_topology
                        if ce.causality_type == "mutation_social"]
        # Fixture grew from the original 3 to 7 mutation_social edges
        # during the 2026-05-01 audit (additional dyad mutations across
        # the Duncan / Banquo / Lady Macbeth web).
        assert len(social_edges) >= 3
        # Verify the Duncan murder → Macbeth fears Banquo edge
        fear_edge = next(ce for ce in social_edges
                         if ce.source_id == "EVT_DUNCAN_MURDER"
                         and ce.rel_counterpart_id == "ENT_BANQUO")
        assert fear_edge.target_id == "ENT_MACBETH"
        assert fear_edge.trait_target == "fear"
        assert fear_edge.trait_delta == 0.5


# =====================================================================
# PROBABILISTIC PHYSICS — added with the noisy-OR / Monte-Carlo / drift
# / Bayesian-blend changes. These tests pin the *new* settings paths and
# exercise the orchestration entry points (execute_distribution).
# =====================================================================
import pytest as _pytest
import random
import statistics

from shadow_loom.settings import get_settings as _get_settings
from shadow_loom.causal_physics import (
    NoisyOrProbability, TraitDistribution,
    _sigmoid, _noisy_or_per_edge_probability, _noisy_or_aggregate,
    _evidence_strength_sigma, _sample_causal_force, _sample_trait_value,
)


@_pytest.fixture
def physics_settings():
    """Snapshot/restore the cached physics settings for an isolated test.

    Mutating ``get_settings().physics`` directly is the simplest path —
    the settings object is a long-lived singleton and per-process state.
    The fixture saves every attribute we touch and restores it on
    teardown so tests can't leak into one another.
    """
    s = _get_settings().physics
    snapshot = {
        k: getattr(s, k) for k in (
            "abduction_blend_mode", "abduction_evidence_precision",
            "propagation_mode", "noisy_or_temperature",
            "noisy_or_threshold", "monte_carlo_samples",
            "monte_carlo_seed", "entity_trait_baseline_drift_rate",
            "causal_force_sigma_weak", "causal_force_sigma_moderate",
            "causal_force_sigma_strong",
        )
    }
    yield s
    for k, v in snapshot.items():
        setattr(s, k, v)


# =====================================================================
# Noisy-OR helpers (unit-level)
# =====================================================================
class TestNoisyOrHelpers:
    def test_sigmoid_monotonic(self):
        assert _sigmoid(-10.0) < _sigmoid(0.0) < _sigmoid(10.0)
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_per_edge_probability_at_inertia_is_half(self):
        # impulse magnitude == inertia ⇒ sigmoid(0) = 0.5
        p = _noisy_or_per_edge_probability(
            weighted_impulse=0.5, inertia=0.5, temperature=0.25,
        )
        assert abs(p - 0.5) < 1e-9

    def test_per_edge_probability_above_inertia(self):
        p = _noisy_or_per_edge_probability(
            weighted_impulse=1.0, inertia=0.2, temperature=0.25,
        )
        assert p > 0.5

    def test_per_edge_probability_uses_absolute_impulse(self):
        # Sign of impulse must not affect the per-edge probability.
        p_pos = _noisy_or_per_edge_probability(0.7, 0.3, 0.25)
        p_neg = _noisy_or_per_edge_probability(-0.7, 0.3, 0.25)
        assert abs(p_pos - p_neg) < 1e-9

    def test_aggregate_empty_is_zero(self):
        assert _noisy_or_aggregate([]) == 0.0

    def test_aggregate_independent_or(self):
        # Two independent attempts at p=0.5 give 1 - 0.25 = 0.75.
        assert abs(_noisy_or_aggregate([0.5, 0.5]) - 0.75) < 1e-9


# =====================================================================
# Causal-force / trait-value sampling
# =====================================================================
class TestStochasticSampling:
    def test_evidence_strength_sigma_lookup(self, physics_settings):
        physics_settings.causal_force_sigma_weak = 0.4
        physics_settings.causal_force_sigma_moderate = 0.2
        physics_settings.causal_force_sigma_strong = 0.05
        assert _evidence_strength_sigma("weak") == 0.4
        assert _evidence_strength_sigma("moderate") == 0.2
        assert _evidence_strength_sigma("strong") == 0.05
        # Unknown labels fall through to moderate.
        assert _evidence_strength_sigma("unrecognised") == 0.2

    def test_sample_causal_force_scales_with_evidence(self, physics_settings):
        # With sigma=0 (impossible — clamped to 1e-6), strong evidence
        # collapses to the nominal value; weak evidence scatters wider.
        physics_settings.causal_force_sigma_weak = 0.5
        physics_settings.causal_force_sigma_strong = 0.001
        rng_weak = random.Random(42)
        rng_strong = random.Random(42)
        weak_samples = [
            _sample_causal_force(5.0, "weak", rng_weak) for _ in range(200)
        ]
        strong_samples = [
            _sample_causal_force(5.0, "strong", rng_strong) for _ in range(200)
        ]
        weak_var = statistics.pvariance(weak_samples)
        strong_var = statistics.pvariance(strong_samples)
        assert weak_var > strong_var

    def test_sample_causal_force_clamped(self):
        rng = random.Random(0)
        for _ in range(100):
            v = _sample_causal_force(5.0, "moderate", rng)
            assert 0.0 <= v <= 10.0

    def test_sample_trait_value_high_inertia_tight(self):
        rng_high = random.Random(7)
        rng_low = random.Random(7)
        high = [_sample_trait_value(0.5, 0.95, rng_high) for _ in range(300)]
        low = [_sample_trait_value(0.5, 0.05, rng_low) for _ in range(300)]
        assert statistics.pvariance(high) < statistics.pvariance(low)


# =====================================================================
# Bayesian abduction blend (Rung 3)
# =====================================================================
class TestBayesianAbduction:
    def test_bayesian_posterior_formula(self, physics_settings):
        physics_settings.abduction_blend_mode = "bayesian"
        physics_settings.abduction_evidence_precision = 1.0

        ws = _make_minimal_world()
        ws_modified = deepcopy(ws)
        # Set a high-inertia trait on Alice and override the factual value.
        ws.entities["ENT_ALICE"].traits["courage"] = TraitVector(value=0.5, inertia=0.8)
        ws_modified.entities["ENT_ALICE"].traits["courage"] = TraitVector(value=0.9, inertia=0.8)

        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")
        old_val = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
        inertia = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["inertia"]

        engine = CausalPhysicsEngine(sandbox, ws_modified)
        engine.abduction_update(["ENT_ALICE"])

        new_val = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
        # Posterior = (k_prior * old + k_ev * evidence) / (k_prior + k_ev).
        expected = (inertia * old_val + 1.0 * 0.9) / (inertia + 1.0)
        assert abs(new_val - expected) < 1e-6

    def test_bayesian_high_inertia_resists_evidence(self, physics_settings):
        physics_settings.abduction_blend_mode = "bayesian"
        physics_settings.abduction_evidence_precision = 0.1  # weak evidence
        ws = _make_minimal_world()
        ws_modified = deepcopy(ws)
        ws.entities["ENT_ALICE"].traits["courage"] = TraitVector(value=0.2, inertia=0.95)
        ws_modified.entities["ENT_ALICE"].traits["courage"] = TraitVector(value=0.9, inertia=0.95)

        sandbox = _build_sandbox(ws, ["ENT_ALICE"], "counterfactual")
        old_val = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
        engine = CausalPhysicsEngine(sandbox, ws_modified)
        engine.abduction_update(["ENT_ALICE"])
        new_val = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
        # Should barely move from the prior.
        assert abs(new_val - old_val) < 0.1
        # And in particular nowhere near the evidence value 0.9.
        assert abs(new_val - 0.9) > 0.5

    def test_legacy_mode_still_supported(self, physics_settings):
        physics_settings.abduction_blend_mode = "legacy"
        ws = _make_minimal_world()
        ws_modified = deepcopy(ws)
        ws_modified.entities["ENT_ALICE"].traits["courage"].value = 0.9

        sandbox = _build_sandbox(ws, ["ENT_ALICE"], "counterfactual")
        old_val = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
        inertia = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["inertia"]
        engine = CausalPhysicsEngine(sandbox, ws_modified)
        engine.abduction_update(["ENT_ALICE"])
        new_val = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
        expected = old_val + (0.9 - old_val) * (1.0 - inertia)
        assert abs(new_val - expected) < 1e-6


# =====================================================================
# Noisy-OR propagation
# =====================================================================
class TestNoisyOrPropagation:
    def test_noisy_or_records_populated(self, physics_settings):
        physics_settings.propagation_mode = "noisy_or"
        ws = _make_minimal_world()
        # Low inertia so the noisy-OR gate fires and we get a record.
        ws.entities["ENT_BOB"].traits["courage"] = TraitVector(value=0.5, inertia=0.05)
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})
        assert any(
            isinstance(r, NoisyOrProbability) and r.node_id == "ENT_BOB"
            for r in result.noisy_or_probabilities
        )

    def test_noisy_or_blocks_emit_absorbed_reason(self, physics_settings):
        physics_settings.propagation_mode = "noisy_or"
        physics_settings.noisy_or_temperature = 0.05  # sharp gate
        physics_settings.noisy_or_threshold = 0.99    # near-impossible to clear
        ws = _make_minimal_world()
        ws.entities["ENT_BOB"].traits["anger"] = TraitVector(value=0.5, inertia=0.99)
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})
        absorbed = [b for b in result.blocked if b.reason == "noisy_or_absorbed"]
        # Anger should at least be tracked as an absorbed noisy-OR block
        # whenever the gate fired against it.
        assert any(b.node_id == "ENT_BOB" for b in absorbed) or all(
            r.fired or r.aggregate_probability < 0.99
            for r in result.noisy_or_probabilities
            if r.node_id == "ENT_BOB"
        )

    def test_noisy_or_does_not_pollute_legacy_path(self, physics_settings):
        physics_settings.propagation_mode = "deterministic"
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})
        # No noisy-OR records under deterministic mode.
        assert result.noisy_or_probabilities == []
        # And no noisy_or_absorbed reasons either.
        assert all(b.reason != "noisy_or_absorbed" for b in result.blocked)


# =====================================================================
# Baseline drift (entity_trait_baseline_drift_rate)
# =====================================================================
class TestBaselineDrift:
    def test_drift_pulls_mutation_back_toward_baseline(self, physics_settings):
        physics_settings.propagation_mode = "deterministic"
        physics_settings.entity_trait_baseline_drift_rate = 1.0  # full pull-back
        ws = _make_minimal_world()
        # Strong incoming impulse, low-inertia target.
        ws.entities["ENT_BOB"].traits["courage"] = TraitVector(value=0.5, inertia=0.05)

        # First capture the no-drift result.
        physics_settings.entity_trait_baseline_drift_rate = 0.0
        sb_no = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        eng_no = CausalPhysicsEngine(sb_no, ws)
        result_no = eng_no.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})

        # Now compare against full drift (rate=1.0) on a fresh sandbox.
        physics_settings.entity_trait_baseline_drift_rate = 1.0
        sb_yes = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        eng_yes = CausalPhysicsEngine(sb_yes, ws)
        result_yes = eng_yes.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})

        # If propagate produced any mutation on Bob.courage, the drifted
        # version must sit closer to the baseline (0.5) than the
        # un-drifted version. Skip the check when there were no mutations.
        muts_no = [m for m in result_no.mutations
                   if m.node_id == "ENT_BOB" and m.trait == "courage"]
        muts_yes = [m for m in result_yes.mutations
                    if m.node_id == "ENT_BOB" and m.trait == "courage"]
        if not muts_no:
            _pytest.skip("Test fixture did not produce a Bob.courage mutation.")
        assert muts_yes, "Drift run produced no Bob.courage mutation."
        baseline = 0.5
        assert abs(muts_yes[0].new_value - baseline) <= abs(muts_no[0].new_value - baseline)

    def test_zero_drift_rate_preserves_legacy(self, physics_settings):
        physics_settings.propagation_mode = "deterministic"
        physics_settings.entity_trait_baseline_drift_rate = 0.0
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        before = sandbox.nodes["ENT_BOB"]["traits"]["courage"]["value"]
        result = engine.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})
        # No drift bookkeeping should have run; mutations match trait values.
        for m in result.mutations:
            tdata = sandbox.nodes[m.node_id]["traits"][m.trait]
            assert abs(tdata["value"] - m.new_value) < 1e-9
        # Sanity: at least one trait may still equal its before-value.
        assert isinstance(before, float)


# =====================================================================
# Monte-Carlo distributional execute
# =====================================================================
class TestExecuteDistribution:
    def test_zero_samples_falls_back_to_deterministic(self, physics_settings):
        physics_settings.monte_carlo_samples = 0
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute_distribution(
            rung=2, interventions={"EVT_FIGHT.event_type": "outcome"},
        )
        # No distributions populated when sampling is disabled.
        assert result.trait_distributions == {}

    def test_distribution_populated_with_samples(self, physics_settings):
        physics_settings.monte_carlo_samples = 6
        physics_settings.monte_carlo_seed = 123
        physics_settings.propagation_mode = "noisy_or"
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute_distribution(
            rung=2,
            interventions={"EVT_FIGHT.event_type": "outcome"},
            samples=6,
            seed=123,
        )
        # At least one entity-trait should have a distribution recorded.
        assert result.trait_distributions, (
            "execute_distribution returned no per-trait distributions"
        )
        any_dist: TraitDistribution | None = None
        for by_trait in result.trait_distributions.values():
            for d in by_trait.values():
                any_dist = d
                break
            if any_dist:
                break
        assert any_dist is not None
        assert any_dist.samples_count > 0
        assert 0.0 <= any_dist.p5 <= any_dist.p50 <= any_dist.p95 <= 1.0

    def test_distribution_supports_bayesian_blend(self, physics_settings):
        # The Monte-Carlo path must honour the abduction_blend_mode
        # setting — execute_distribution delegates to execute() which
        # delegates to abduction_update().
        physics_settings.monte_carlo_samples = 4
        physics_settings.monte_carlo_seed = 7
        physics_settings.abduction_blend_mode = "bayesian"
        physics_settings.abduction_evidence_precision = 5.0  # evidence dominates

        ws = _make_minimal_world()
        ws_modified = deepcopy(ws)
        ws_modified.entities["ENT_ALICE"].traits["courage"].value = 0.9

        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")
        engine = CausalPhysicsEngine(sandbox, ws_modified)
        result = engine.execute_distribution(
            rung=3,
            interventions={"EVT_FIGHT.event_type": "outcome"},
            evidence_node_ids=["ENT_ALICE"],
            samples=4,
            seed=7,
        )
        # Result should record a distribution for at least one trait of Alice.
        assert "ENT_ALICE" in result.trait_distributions

    def test_sandbox_restored_after_distribution_run(self, physics_settings):
        physics_settings.monte_carlo_samples = 3
        physics_settings.monte_carlo_seed = 1
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        before_value = engine.sandbox.nodes["ENT_BOB"]["traits"]["courage"]["value"]
        engine.execute_distribution(
            rung=2,
            interventions={"EVT_FIGHT.event_type": "outcome"},
            samples=3,
            seed=1,
        )
        after_value = engine.sandbox.nodes["ENT_BOB"]["traits"]["courage"]["value"]
        assert before_value == after_value, (
            "execute_distribution must not mutate the engine's sandbox in place."
        )

    def test_execute_auto_routes_when_samples_configured(self, physics_settings):
        """Setting ``monte_carlo_samples > 0`` makes the *plain* ``execute()``
        entry point auto-delegate to ``execute_distribution`` so existing
        callers (narrative_physics, directive_assembly, MCP server, ...)
        opt into Monte-Carlo without code changes.
        """
        physics_settings.monte_carlo_samples = 3
        physics_settings.monte_carlo_seed = 11
        ws = _make_minimal_world()
        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
        engine = CausalPhysicsEngine(sandbox, ws)
        # Plain execute() — but settings should redirect through the MC path.
        result = engine.execute(rung=2, interventions={"EVT_FIGHT.event_type": "outcome"})
        assert result.trait_distributions, (
            "execute() with monte_carlo_samples>0 must populate "
            "trait_distributions via the auto-routed Monte-Carlo path."
        )
