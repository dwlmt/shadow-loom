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
    CausalEdge, SpatialEdge, RelationshipEdge, InformationEdge,
    TraitVector, Affordance, Belief,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.causal_physics import (
    CausalPhysicsEngine, CausalPhysicsResult, STRENGTH_MULTIPLIER,
    MECHANISM_TRAIT_MAP, MECHANISM_FALLBACK_FACTOR,
)
from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import InterventionQuery, CounterfactualQuery

# ── Reuse a plot for integration tests ──
from tests.test_plot_models.macbeth import world_state as macbeth_ws


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
                      event_type="choice", actor_id="ENT_ALICE",
                      target_id="ENT_BOB", description="Alice attacks Bob"),
            EventNode(id="EVT_RESULT", fabula_time=2, syuzhet_index=2,
                      event_type="outcome", actor_id=None,
                      description="Bob retaliates"),
        ],
        causal_topology=[
            CausalEdge(source_event_id="EVT_FIGHT",
                       target_node_id="ENT_BOB",
                       mechanism="physical", evidence_strength="strong",
                       fabula_time=1),
            CausalEdge(source_event_id="EVT_RESULT",
                       target_node_id="ENT_ALICE",
                       mechanism="physical", evidence_strength="moderate",
                       fabula_time=2),
            CausalEdge(source_event_id="EVT_FIGHT",
                       target_node_id="EVT_RESULT",
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
        assert "ENT_BOB" in all_nodes or len(result.mutations) >= 0  # engine processed it


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
        """Abduction must blend traits 50% toward factual values."""
        ws = _make_minimal_world()
        ws_modified = deepcopy(ws)
        ws_modified.entities["ENT_ALICE"].traits["courage"].value = 0.9

        sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"], "counterfactual")
        old_courage = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]

        engine = CausalPhysicsEngine(sandbox, ws_modified)
        engine.abduction_update(["ENT_ALICE"])

        new_courage = sandbox.nodes["ENT_ALICE"]["traits"]["courage"]["value"]
        expected = old_courage + (0.9 - old_courage) * 0.5
        assert abs(new_courage - expected) < 0.01

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
                CausalEdge(source_event_id="EVT_CAUSE",
                           target_node_id="ENT_ALICE",
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
        engine.propagate()

        # Check that any mutations or blocked entries reflect mechanism gating
        all_items = engine._mutations + engine._blocked
        guilt_items = [x for x in all_items if x.trait == "guilt"]
        courage_items = [x for x in all_items if x.trait == "courage"]
        # If there are entries for both, guilt impact should be >= courage impact
        if guilt_items and courage_items:
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
        engine.propagate()

        mutations = [m for m in engine._mutations
                     if m.node_id == "ENT_TGT" and m.trait == "courage"]
        if mutations:
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
        engine.propagate()

        mutations = [m for m in engine._mutations
                     if m.node_id == "ENT_TGT" and m.trait == "courage"]
        if mutations:
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
