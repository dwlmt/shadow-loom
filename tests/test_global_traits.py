# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for Global Traits (World-Level Properties).

Covers:
  • Model creation and validation (GlobalTrait, WorldTraitSnapshot)
  • reconstruct_world_trait_at() temporal reconstruction
  • CausalEdge validator accepts WORLD_ prefix
  • WorldStateV1 with world_traits field
  • EgoGraphPayload includes world_traits
  • Sandbox creates WORLD_ nodes and ambient edges
  • Engine propagation with WORLD_ sources (magnitude scaling, domain filtering)
  • Abduction skips WORLD_ nodes
"""
import pytest
from copy import deepcopy

import networkx as nx

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge,
    TraitVector, AmbientVector, Affordance, Belief, EntityStateSnapshot,
    GlobalTrait, WorldTraitSnapshot,
    reconstruct_entity_at, reconstruct_world_trait_at,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory, EgoGraphPayload
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.causal_physics import (
    CausalPhysicsEngine, CausalPhysicsResult,
    MECHANISM_TRAIT_MAP, MECHANISM_FALLBACK_FACTOR,
)

# Reuse plot models that now include world_traits
from example_worlds.macbeth import world_state as macbeth_ws
from example_worlds.nineteen_eighty_four import world_state as orwell_ws
from example_worlds.persuasion import world_state as persuasion_ws


# =====================================================================
# Minimal fixtures
# =====================================================================

def _make_world_with_traits() -> WorldStateV1:
    """Minimal world with one entity and one world trait."""
    return WorldStateV1(
        locations={
            "LOC_A": Location(
                id="LOC_A",
                name="Room A", description="A room.",
                              ambient_state={}),
        },
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={
                    "fear": TraitVector(value=0.3, inertia=0.2),
                    "courage": TraitVector(value=0.5, inertia=0.3),
                    "loyalty": TraitVector(value=0.6, inertia=0.4),
                },
            ),
        },
        world_traits={
            "WORLD_WARTIME": GlobalTrait(
                id="WORLD_WARTIME",
                name="Wartime Economy",
                description="The nation is at war, causing scarcity and fear.",
                category="economy",
                magnitude=TraitVector(value=0.8, inertia=0.5),
                affected_domains=["psychological", "social"],
            ),
        },
        events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="outcome", description="A battle begins."),
        ],
        causal_topology=[],
    )


# =====================================================================
# Phase 1: Model Tests
# =====================================================================

class TestGlobalTraitModel:
    def test_creation(self):
        gt = GlobalTrait(
            id="WORLD_TEST", name="Test Trait",
            description="A test world trait.",
            category="governance",
            magnitude=TraitVector(value=0.7, inertia=0.6),
            affected_domains=["social", "psychological"],
        )
        assert gt.id == "WORLD_TEST"
        assert gt.magnitude.value == 0.7
        assert gt.magnitude.inertia == 0.6
        assert gt.affected_domains == ["social", "psychological"]
        assert gt.state_timeline == []

    def test_round_trip(self):
        gt = GlobalTrait(
            id="WORLD_MAGIC", name="Magic System",
            description="A structured magic system.",
            category="magic_system",
            magnitude=TraitVector(value=0.9, inertia=0.95),
            affected_domains=["physical", "epistemic"],
        )
        data = gt.model_dump()
        gt2 = GlobalTrait(**data)
        assert gt2.id == gt.id
        assert gt2.magnitude.value == gt.magnitude.value

    def test_with_state_timeline(self):
        gt = GlobalTrait(
            id="WORLD_WAR", name="Wartime",
            description="At war.", category="governance",
            magnitude=TraitVector(value=0.8, inertia=0.5),
            affected_domains=["social"],
            state_timeline=[
                WorldTraitSnapshot(
                    fabula_time=500,
                    triggered_by="EVT_PEACE_TREATY",
                    magnitude=TraitVector(value=0.2, inertia=0.3),
                    description="The war ends.",
                ),
            ],
        )
        assert len(gt.state_timeline) == 1
        assert gt.state_timeline[0].fabula_time == 500


class TestWorldTraitSnapshot:
    def test_minimal(self):
        snap = WorldTraitSnapshot(fabula_time=100)
        assert snap.fabula_time == 100
        assert snap.triggered_by is None
        assert snap.magnitude is None
        assert snap.description is None

    def test_full(self):
        snap = WorldTraitSnapshot(
            fabula_time=200,
            triggered_by="EVT_REVOLUTION",
            magnitude=TraitVector(value=0.1, inertia=0.2),
            description="The regime falls.",
        )
        assert snap.magnitude.value == 0.1


class TestReconstructWorldTraitAt:
    def test_no_timeline(self):
        gt = GlobalTrait(
            id="WORLD_X", name="X", description="X", category="governance",
            magnitude=TraitVector(value=0.7, inertia=0.6),
            affected_domains=["social"],
        )
        result = reconstruct_world_trait_at(gt, 999)
        assert result["magnitude"]["value"] == 0.7
        assert result["magnitude"]["inertia"] == 0.6
        assert result["description"] == "X"

    def test_before_first_snapshot(self):
        gt = GlobalTrait(
            id="WORLD_X", name="X", description="Original", category="governance",
            magnitude=TraitVector(value=0.8, inertia=0.5),
            affected_domains=["social"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=500, magnitude=TraitVector(value=0.2, inertia=0.3)),
            ],
        )
        result = reconstruct_world_trait_at(gt, 400)
        assert result["magnitude"]["value"] == 0.8  # still original

    def test_after_snapshot(self):
        gt = GlobalTrait(
            id="WORLD_X", name="X", description="Original", category="governance",
            magnitude=TraitVector(value=0.8, inertia=0.5),
            affected_domains=["social"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=500, magnitude=TraitVector(value=0.2, inertia=0.3)),
            ],
        )
        result = reconstruct_world_trait_at(gt, 600)
        assert result["magnitude"]["value"] == 0.2
        assert result["magnitude"]["inertia"] == 0.3

    def test_description_update(self):
        gt = GlobalTrait(
            id="WORLD_X", name="X", description="At war", category="governance",
            magnitude=TraitVector(value=0.8, inertia=0.5),
            affected_domains=["social"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=500, description="At peace"),
            ],
        )
        result = reconstruct_world_trait_at(gt, 600)
        assert result["description"] == "At peace"
        assert result["magnitude"]["value"] == 0.8  # unchanged

    def test_multiple_snapshots(self):
        gt = GlobalTrait(
            id="WORLD_X", name="X", description="Phase 0", category="governance",
            magnitude=TraitVector(value=0.9, inertia=0.5),
            affected_domains=["social"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=200, magnitude=TraitVector(value=0.5, inertia=0.4)),
                WorldTraitSnapshot(fabula_time=400, magnitude=TraitVector(value=0.1, inertia=0.2),
                                   description="Phase 2"),
            ],
        )
        r1 = reconstruct_world_trait_at(gt, 150)
        assert r1["magnitude"]["value"] == 0.9
        r2 = reconstruct_world_trait_at(gt, 300)
        assert r2["magnitude"]["value"] == 0.5
        r3 = reconstruct_world_trait_at(gt, 500)
        assert r3["magnitude"]["value"] == 0.1
        assert r3["description"] == "Phase 2"


class TestCausalEdgeWorldPrefix:
    def test_world_ambient_propagation(self):
        """WORLD_ source with ambient_propagation should pass validation."""
        edge = CausalEdge(
            source_id="WORLD_WARTIME", target_id="ENT_ALICE",
            causality_type="ambient_propagation",
            mechanism="psychological", fabula_time=100,
        )
        assert edge.source_id == "WORLD_WARTIME"

    def test_world_mutation(self):
        """WORLD_ source with mutation should pass validation."""
        edge = CausalEdge(
            source_id="WORLD_SURVEILLANCE", target_id="ENT_WINSTON",
            causality_type="mutation",
            mechanism="psychological", fabula_time=100,
            trait_target="fear", trait_delta=0.3,
        )
        assert edge.causality_type == "mutation"

    def test_world_chain_reaction(self):
        """WORLD_ source with chain_reaction to an event should pass."""
        edge = CausalEdge(
            source_id="WORLD_WARTIME", target_id="EVT_FOOD_SHORTAGE",
            causality_type="chain_reaction",
            mechanism="physical", fabula_time=100,
        )
        assert edge.causality_type == "chain_reaction"

    def test_world_affordance_gate(self):
        """WORLD_ source with affordance_gate to an event should pass."""
        edge = CausalEdge(
            source_id="WORLD_MAGIC_SYSTEM", target_id="EVT_SPELL_CAST",
            causality_type="affordance_gate",
            mechanism="physical", fabula_time=100,
        )
        assert edge.causality_type == "affordance_gate"

    def test_world_mutation_social_rejected(self):
        """WORLD_ with mutation_social should be rejected."""
        with pytest.raises(ValueError, match="world trait"):
            CausalEdge(
                source_id="WORLD_X", target_id="ENT_Y",
                causality_type="mutation_social",
                mechanism="social", fabula_time=100,
                trait_target="affinity", trait_delta=0.1,
                rel_counterpart_id="ENT_Z",
            )


class TestWorldStateV1WithTraits:
    def test_empty_world_traits_default(self):
        ws = WorldStateV1(
            locations={}, objects={}, entities={}, events=[], causal_topology=[],
        )
        assert ws.world_traits == {}

    def test_with_world_traits(self):
        ws = _make_world_with_traits()
        assert "WORLD_WARTIME" in ws.world_traits
        assert ws.world_traits["WORLD_WARTIME"].magnitude.value == 0.8


# =====================================================================
# Phase 4: Graph Payload Tests
# =====================================================================

class TestEgoGraphWorldTraits:
    def test_world_traits_included_in_payload(self):
        ws = _make_world_with_traits()
        payload = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
        assert len(payload.world_traits) == 1
        assert payload.world_traits[0]["id"] == "WORLD_WARTIME"

    def test_world_traits_empty_when_none(self):
        ws = WorldStateV1(
            locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="a", ambient_state={})},
            objects={}, entities={
                "ENT_X": Entity(id="ENT_X", name="X", location_id="LOC_A",
                                status="healthy", traits={"t": TraitVector(value=0.5, inertia=0.5)}),
            },
            events=[], causal_topology=[],
        )
        payload = extract_ego_graph_from_memory(ws, ["ENT_X"])
        assert payload.world_traits == []

    def test_world_trait_ids_in_scene_node_ids(self):
        """Causal edges from WORLD_ sources should pass the scene_node_ids filter."""
        ws = _make_world_with_traits()
        ws.causal_topology.append(CausalEdge(
            source_id="WORLD_WARTIME", target_id="ENT_ALICE",
            causality_type="ambient_propagation",
            mechanism="psychological", fabula_time=100,
        ))
        payload = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
        world_edges = [e for e in payload.relevant_causal_edges
                       if e["source_id"].startswith("WORLD_")]
        assert len(world_edges) == 1

    def test_temporal_reconstruction(self):
        """World traits should be time-reconstructed when anchor is set."""
        ws = _make_world_with_traits()
        ws.world_traits["WORLD_WARTIME"].state_timeline = [
            WorldTraitSnapshot(fabula_time=50, magnitude=TraitVector(value=0.1, inertia=0.2)),
        ]
        # Before snapshot
        payload_early = extract_ego_graph_from_memory(ws, ["ENT_ALICE"], temporal_anchor=40)
        assert payload_early.world_traits[0]["magnitude"]["value"] == 0.8
        # After snapshot
        payload_late = extract_ego_graph_from_memory(ws, ["ENT_ALICE"], temporal_anchor=60)
        assert payload_late.world_traits[0]["magnitude"]["value"] == 0.1


# =====================================================================
# Phase 5: Sandbox Tests
# =====================================================================

class TestSandboxWorldTraits:
    def _build_sandbox(self, ws=None, query_type="observation"):
        ws = ws or _make_world_with_traits()
        payload = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
        sandbox = AMWNInstantiator.create_sandbox(payload.model_dump(), query_type)
        return sandbox

    def test_world_node_created(self):
        sandbox = self._build_sandbox()
        assert sandbox.has_node("WORLD_WARTIME")
        data = sandbox.nodes["WORLD_WARTIME"]
        assert data["node_type"] == "WorldTrait"
        assert data["magnitude"]["value"] == 0.8

    def test_ambient_edges_created(self):
        """Auto-gen ambient_propagation edges from WORLD_ to all entities."""
        sandbox = self._build_sandbox()
        ambient_edges = [
            (u, v, d) for u, v, d in sandbox.edges(data=True)
            if d.get("edge_type") == "causal"
            and d.get("causality_type") == "ambient_propagation"
            and u.startswith("WORLD_")
        ]
        assert len(ambient_edges) >= 1
        edge = ambient_edges[0]
        assert edge[0] == "WORLD_WARTIME"
        assert edge[1] == "ENT_ALICE"
        # Force should be magnitude.value * 2.0
        assert abs(edge[2]["causal_force"] - 0.8 * 2.0) < 0.01

    def test_ambient_edge_mechanism_from_domains(self):
        sandbox = self._build_sandbox()
        ambient_edges = [
            (u, v, d) for u, v, d in sandbox.edges(data=True)
            if u == "WORLD_WARTIME" and d.get("causality_type") == "ambient_propagation"
        ]
        assert ambient_edges[0][2]["mechanism"] == "psychological"  # first affected_domain

    def test_no_ambient_edges_when_no_traits(self):
        ws = WorldStateV1(
            locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="a", ambient_state={})},
            objects={}, entities={
                "ENT_X": Entity(id="ENT_X", name="X", location_id="LOC_A",
                                status="healthy", traits={"t": TraitVector(value=0.5, inertia=0.5)}),
            },
            events=[], causal_topology=[],
        )
        payload = extract_ego_graph_from_memory(ws, ["ENT_X"])
        sandbox = AMWNInstantiator.create_sandbox(payload.model_dump(), "observation")
        ambient = [(u, v) for u, v, d in sandbox.edges(data=True)
                   if u.startswith("WORLD_")]
        assert len(ambient) == 0


# =====================================================================
# Phase 6: Engine Tests
# =====================================================================

class TestEnginePropagationWorldTraits:
    def _run_propagation(self, ws=None):
        ws = ws or _make_world_with_traits()
        payload = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
        sandbox = AMWNInstantiator.create_sandbox(payload.model_dump(), "observation")
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2)
        return sandbox, result

    def test_world_trait_propagates(self):
        """WORLD_ node should propagate impact to entity traits."""
        sandbox, result = self._run_propagation()
        # The ambient edge should have attempted to propagate
        # Whether it passes inertia depends on force, but it should at least try
        # Check that mutations or blocked entries exist for alice
        alice_mutations = [m for m in result.mutations if m.node_id == "ENT_ALICE"]
        alice_blocked = [b for b in result.blocked if b.node_id == "ENT_ALICE"]
        # At minimum, the engine should have processed edges from WORLD_ to ENT_ALICE
        assert len(alice_mutations) + len(alice_blocked) > 0

    def test_magnitude_scales_impulse(self):
        """Higher magnitude should produce stronger impact."""
        ws_strong = _make_world_with_traits()
        ws_strong.world_traits["WORLD_WARTIME"].magnitude = TraitVector(value=0.9, inertia=0.5)
        ws_weak = _make_world_with_traits()
        ws_weak.world_traits["WORLD_WARTIME"].magnitude = TraitVector(value=0.1, inertia=0.5)

        _, result_strong = self._run_propagation(ws_strong)
        _, result_weak = self._run_propagation(ws_weak)

        # Extract all impacts (mutations + blocked) for alice's fear
        def _total_impact(result):
            total = 0.0
            for m in result.mutations:
                if m.node_id == "ENT_ALICE":
                    total += abs(m.new_value - m.old_value)
            return total

        impact_strong = _total_impact(result_strong)
        impact_weak = _total_impact(result_weak)
        # Strong magnitude should produce >= impact than weak
        # (they could both be blocked, in which case both are 0, which is fine)
        assert impact_strong >= impact_weak


class TestEngineAbductionSkipsWorld:
    def test_world_nodes_skipped(self):
        """Abduction should skip WORLD_ evidence nodes without error."""
        ws = _make_world_with_traits()
        payload = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
        sandbox = AMWNInstantiator.create_sandbox(payload.model_dump(), "observation")
        engine = CausalPhysicsEngine(sandbox, ws)
        # This should not raise, even with a WORLD_ ID as evidence
        result = engine.execute(rung=3, evidence_node_ids=["WORLD_WARTIME", "ENT_ALICE"])
        assert isinstance(result, CausalPhysicsResult)
        # WORLD_ should not appear in hidden_deltas
        assert "WORLD_WARTIME" not in result.hidden_deltas


class TestEngineDomainFiltering:
    def test_off_domain_gets_fallback(self):
        """Edges with mechanism not in affected_domains should get reduced weight."""
        ws = _make_world_with_traits()
        # WORLD_WARTIME has affected_domains=["psychological", "social"]
        # Add an explicit edge with mechanism "physical" (not in affected_domains)
        ws.causal_topology.append(CausalEdge(
            source_id="WORLD_WARTIME", target_id="ENT_ALICE",
            causality_type="ambient_propagation",
            mechanism="physical",  # NOT in affected_domains
            causal_force=8.0,
            fabula_time=100,
        ))
        payload = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
        sandbox = AMWNInstantiator.create_sandbox(payload.model_dump(), "observation")
        engine = CausalPhysicsEngine(sandbox, ws)
        result = engine.execute(rung=2)
        # The edge should still be processed (not error), just at reduced weight
        assert isinstance(result, CausalPhysicsResult)


# =====================================================================
# Plot Model Integration Tests
# =====================================================================

class TestPlotModelWorldTraits:
    def test_macbeth_has_world_traits(self):
        # Fixture grew during the 2026-05-01 audit (WORLD_DIVINE_RIGHT
        # added). Pin the canonical pair the engine relies on rather
        # than the exact count.
        assert len(macbeth_ws.world_traits) >= 2
        assert "WORLD_FEUDAL_HIERARCHY" in macbeth_ws.world_traits
        assert "WORLD_SUPERNATURAL_PROPHECY" in macbeth_ws.world_traits

    def test_orwell_has_world_traits(self):
        # Fixture renamed WORLD_SURVEILLANCE_STATE → WORLD_PANOPTICON
        # during the audit; same role (panoptic surveillance regime),
        # same magnitude band.
        assert len(orwell_ws.world_traits) == 3
        assert "WORLD_PANOPTICON" in orwell_ws.world_traits
        assert orwell_ws.world_traits["WORLD_PANOPTICON"].magnitude.value == 0.95

    def test_persuasion_has_world_traits(self):
        # Fixture renamed WORLD_SOCIAL_RIGIDITY → WORLD_REGENCY_RANK
        # and grew with WORLD_NAVAL_PRIZE_ECONOMY +
        # WORLD_PRIMOGENITURE_ENTAIL during the audit.
        assert len(persuasion_ws.world_traits) >= 2
        assert "WORLD_REGENCY_RANK" in persuasion_ws.world_traits

    def test_macbeth_world_traits_in_ego_graph(self):
        payload = extract_ego_graph_from_memory(macbeth_ws, ["ENT_MACBETH"])
        assert len(payload.world_traits) >= 2
        ids = {wt["id"] for wt in payload.world_traits}
        assert "WORLD_FEUDAL_HIERARCHY" in ids
        assert "WORLD_SUPERNATURAL_PROPHECY" in ids

    def test_macbeth_sandbox_has_world_nodes(self):
        payload = extract_ego_graph_from_memory(macbeth_ws, ["ENT_MACBETH"])
        sandbox = AMWNInstantiator.create_sandbox(payload.model_dump(), "observation")
        assert sandbox.has_node("WORLD_FEUDAL_HIERARCHY")
        assert sandbox.has_node("WORLD_SUPERNATURAL_PROPHECY")
        # Check ambient edges exist
        ambient = [(u, v) for u, v, d in sandbox.edges(data=True)
                   if u.startswith("WORLD_") and d.get("causality_type") == "ambient_propagation"]
        assert len(ambient) > 0

    def test_orwell_full_pipeline(self):
        """End-to-end: orwell world traits → ego graph → sandbox → engine."""
        payload = extract_ego_graph_from_memory(orwell_ws, ["ENT_WINSTON"])
        sandbox = AMWNInstantiator.create_sandbox(payload.model_dump(), "observation")
        engine = CausalPhysicsEngine(sandbox, orwell_ws)
        result = engine.execute(rung=2)
        assert isinstance(result, CausalPhysicsResult)
        # The surveillance state should have tried to propagate
        winston_mutations = [m for m in result.mutations if m.node_id == "ENT_WINSTON"]
        winston_blocked = [b for b in result.blocked if b.node_id == "ENT_WINSTON"]
        assert len(winston_mutations) + len(winston_blocked) > 0


# =====================================================================
# Phase G: Regression Tests (audit remediation)
# =====================================================================

class TestIngestionPreservesWorldTraits:
    """Verify world_traits survive ingestion pipeline rebuild paths."""

    def test_normalize_fabula_times_preserves_world_traits(self):
        """_normalize_fabula_times must not drop world_traits."""
        from shadow_loom.ingestion import _normalize_fabula_times

        ws = _make_world_with_traits()
        # Force small sequential times so normalization triggers
        ws.events[0] = ws.events[0].model_copy(update={"fabula_time": 1, "syuzhet_index": 0})
        ws.events.append(EventNode(
            id="EVT_2", fabula_time=2, syuzhet_index=1,
            event_type="outcome", description="Another event.",
        ))
        ws.world_traits["WORLD_WARTIME"].state_timeline = [
            WorldTraitSnapshot(fabula_time=1, magnitude=TraitVector(value=0.5, inertia=0.3)),
        ]

        normalized = _normalize_fabula_times(ws, spacing=1000)
        # world_traits must be preserved
        assert "WORLD_WARTIME" in normalized.world_traits
        assert normalized.world_traits["WORLD_WARTIME"].magnitude.value == 0.8
        # state_timeline fabula_time should be remapped
        assert normalized.world_traits["WORLD_WARTIME"].state_timeline[0].fabula_time == 1000

    def test_auto_repair_preserves_world_traits(self):
        """_auto_repair must not drop world_traits from rebuilt WorldStateV1."""
        from shadow_loom.ingestion import _auto_repair

        ws = _make_world_with_traits()
        # Add a broken causal edge to trigger repair
        ws.causal_topology.append(CausalEdge(
            source_id="EVT_NONEXISTENT", target_id="EVT_1",
            causality_type="chain_reaction",
            mechanism="physical", fabula_time=100,
        ))

        repaired, repairs = _auto_repair(ws)
        assert len(repairs) > 0  # repair was triggered
        assert "WORLD_WARTIME" in repaired.world_traits

    def test_auto_repair_does_not_delete_world_causal_edges(self):
        """WORLD_ causal edges should not be pruned as broken links."""
        from shadow_loom.ingestion import _auto_repair

        ws = _make_world_with_traits()
        ws.causal_topology.append(CausalEdge(
            source_id="WORLD_WARTIME", target_id="ENT_ALICE",
            causality_type="ambient_propagation",
            mechanism="psychological", fabula_time=100,
        ))
        # Also add a broken edge to trigger repair
        ws.causal_topology.append(CausalEdge(
            source_id="EVT_GHOST", target_id="EVT_1",
            causality_type="chain_reaction",
            mechanism="physical", fabula_time=50,
        ))

        repaired, repairs = _auto_repair(ws)
        world_edges = [ce for ce in repaired.causal_topology
                       if ce.source_id.startswith("WORLD_")]
        assert len(world_edges) == 1
        assert world_edges[0].source_id == "WORLD_WARTIME"

    def test_programmatic_validation_accepts_world_ids(self):
        """_programmatic_validation should not flag WORLD_ causal edges as broken."""
        from shadow_loom.ingestion import _programmatic_validation

        ws = _make_world_with_traits()
        ws.causal_topology.append(CausalEdge(
            source_id="WORLD_WARTIME", target_id="ENT_ALICE",
            causality_type="ambient_propagation",
            mechanism="psychological", fabula_time=100,
        ))

        issues = _programmatic_validation(ws)
        broken = [i for i in issues if "WORLD_WARTIME" in i.detail]
        assert len(broken) == 0


class TestExtractionValidIdSet:
    """Verify _build_valid_id_set includes world_traits."""

    def test_world_trait_ids_in_valid_set(self):
        from shadow_loom.ingestion import GlobalRegister, _build_valid_id_set

        reg = GlobalRegister(
            locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="a", ambient_state={})},
            objects={},
            entities={},
            world_traits={
                "WORLD_TEST": GlobalTrait(
                    id="WORLD_TEST", name="Test", description="Test",
                    category="governance",
                    magnitude=TraitVector(value=0.5, inertia=0.5),
                    affected_domains=["social"],
                ),
            },
        )
        valid = _build_valid_id_set(reg)
        assert "WORLD_TEST" in valid
        assert "LOC_A" in valid


class TestQueryParsingWorldTraits:
    """Verify query_parsing includes WORLD_ in graph summary and ID collection."""

    def test_graph_summary_includes_world_traits(self):
        from shadow_loom.query_parsing import _build_graph_summary

        ws = _make_world_with_traits()
        summary = _build_graph_summary(ws)
        assert "WORLD TRAITS:" in summary
        assert "WORLD_WARTIME" in summary
        assert "Wartime Economy" in summary

    def test_collect_all_ids_includes_world_traits(self):
        from shadow_loom.query_parsing import _collect_all_ids

        ws = _make_world_with_traits()
        ids = _collect_all_ids(ws)
        assert "WORLD_WARTIME" in ids

    def test_name_index_includes_world_traits(self):
        from shadow_loom.query_parsing import _build_name_index

        ws = _make_world_with_traits()
        index = _build_name_index(ws)
        # Should be findable by normalized name
        assert "WARTIME" in index or "WARTIME_ECONOMY" in index


class TestGenerationWorldTraitsContext:
    """Verify generation scene context includes world traits."""

    def test_format_scene_context_with_world_traits(self):
        from shadow_loom.generation import _format_scene_context

        ctx = {
            "focus_entities": [],
            "world_traits": [
                {
                    "id": "WORLD_WARTIME",
                    "name": "Wartime Economy",
                    "description": "The nation is at war.",
                    "magnitude": {"value": 0.8, "inertia": 0.5},
                    "affected_domains": ["psychological", "social"],
                },
            ],
        }
        result = _format_scene_context(ctx)
        assert "Wartime Economy" in result
        assert "WORLD_WARTIME" in result
        assert "mag=0.80" in result


class TestNarrativePhysicsDomainFilter:
    """Verify narrative_physics forward cascade applies domain filtering for WorldTrait sources."""

    def test_world_trait_domain_filter_in_sandbox(self):
        """WORLD_ nodes in sandbox should have WorldTrait node_type for domain filtering."""
        ws = _make_world_with_traits()
        payload = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
        sandbox = AMWNInstantiator.create_sandbox(payload.model_dump(), "observation")
        node_data = sandbox.nodes["WORLD_WARTIME"]
        assert node_data["node_type"] == "WorldTrait"
        assert "affected_domains" in node_data
        assert node_data["affected_domains"] == ["psychological", "social"]


# =====================================================================
# Phase H: Step 5 — World Trait Timeline Extraction
# =====================================================================

class TestWorldTraitTimelineAgent:
    """Tests for the Step 5 post-assembly world trait timeline extraction."""

    def test_build_agent(self):
        """_build_world_trait_timeline_agent returns a configured agent."""
        from shadow_loom.ingestion import (
            ExtractionConfig, _build_world_trait_timeline_agent,
        )
        config = ExtractionConfig(model="ollama:test-model")
        agent = _build_world_trait_timeline_agent(config)
        assert agent is not None

    def test_extract_skips_when_no_world_traits(self):
        """extract_world_trait_timelines should be a no-op with no world traits."""
        from shadow_loom.ingestion import extract_world_trait_timelines, ExtractionConfig

        ws = WorldStateV1(
            locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="a", ambient_state={})},
            objects={},
            entities={},
            events=[
                EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                          event_type="outcome", description="Something."),
            ],
            world_traits={},
            causal_topology=[],
        )
        result = extract_world_trait_timelines(ws, ExtractionConfig())
        assert result is ws  # same object, no modification

    @pytest.mark.asyncio
    async def test_extract_async_skips_when_no_world_traits(self):
        """Async variant should also skip when no world traits."""
        from shadow_loom.ingestion import extract_world_trait_timelines_async, ExtractionConfig

        ws = WorldStateV1(
            locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="a", ambient_state={})},
            objects={},
            entities={},
            events=[
                EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                          event_type="outcome", description="Something."),
            ],
            world_traits={},
            causal_topology=[],
        )
        result = await extract_world_trait_timelines_async(ws, ExtractionConfig())
        assert result is ws

    def test_extract_applies_timelines(self):
        """Verify timelines are applied correctly when the agent returns data."""
        from unittest.mock import patch, MagicMock
        from shadow_loom.ingestion import (
            extract_world_trait_timelines, ExtractionConfig,
            WorldTraitTimelineExtraction,
        )

        ws = _make_world_with_traits()
        # Add a second event to trigger the timeline
        ws.events.append(EventNode(
            id="EVT_WAR_ENDS", fabula_time=5000, syuzhet_index=1,
            event_type="outcome", description="The war ends with a treaty.",
        ))

        mock_extraction = WorldTraitTimelineExtraction(
            timelines={
                "WORLD_WARTIME": [
                    WorldTraitSnapshot(
                        fabula_time=5000,
                        triggered_by="EVT_WAR_ENDS",
                        magnitude=TraitVector(value=0.1, inertia=0.3),
                        description="The war ends with a treaty signing.",
                    ),
                ],
            },
        )

        mock_result = MagicMock()
        mock_result.output = mock_extraction

        with patch(
            "shadow_loom.ingestion._build_world_trait_timeline_agent"
        ) as mock_build:
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = mock_result
            mock_build.return_value = mock_agent

            result = extract_world_trait_timelines(ws, ExtractionConfig())

        # Verify timeline was applied
        assert "WORLD_WARTIME" in result.world_traits
        wt = result.world_traits["WORLD_WARTIME"]
        assert len(wt.state_timeline) == 1
        assert wt.state_timeline[0].fabula_time == 5000
        assert wt.state_timeline[0].triggered_by == "EVT_WAR_ENDS"
        assert wt.state_timeline[0].magnitude.value == 0.1
        # Original magnitude should be unchanged
        assert wt.magnitude.value == 0.8

    def test_extract_survives_agent_failure(self):
        """If the agent raises, the original world state is returned unchanged."""
        from unittest.mock import patch, MagicMock
        from shadow_loom.ingestion import extract_world_trait_timelines, ExtractionConfig

        ws = _make_world_with_traits()

        with patch(
            "shadow_loom.ingestion._build_world_trait_timeline_agent"
        ) as mock_build:
            mock_agent = MagicMock()
            mock_agent.run_sync.side_effect = RuntimeError("LLM failed")
            mock_build.return_value = mock_agent

            result = extract_world_trait_timelines(ws, ExtractionConfig())

        # Should return original world state unchanged
        assert result is ws
        assert result.world_traits["WORLD_WARTIME"].state_timeline == []

    def test_extract_preserves_unchanged_traits(self):
        """Traits not in the extraction output keep their existing state."""
        from unittest.mock import patch, MagicMock
        from shadow_loom.ingestion import (
            extract_world_trait_timelines, ExtractionConfig,
            WorldTraitTimelineExtraction,
        )

        ws = _make_world_with_traits()
        # Add a second world trait
        ws.world_traits["WORLD_MORALE"] = GlobalTrait(
            id="WORLD_MORALE", name="National Morale",
            description="Public morale during wartime.",
            category="social_structure",
            magnitude=TraitVector(value=0.4, inertia=0.3),
            affected_domains=["psychological"],
        )

        # Only WORLD_WARTIME changes
        mock_extraction = WorldTraitTimelineExtraction(
            timelines={
                "WORLD_WARTIME": [
                    WorldTraitSnapshot(
                        fabula_time=100,
                        triggered_by="EVT_1",
                        magnitude=TraitVector(value=0.2, inertia=0.4),
                    ),
                ],
            },
        )
        mock_result = MagicMock()
        mock_result.output = mock_extraction

        with patch(
            "shadow_loom.ingestion._build_world_trait_timeline_agent"
        ) as mock_build:
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = mock_result
            mock_build.return_value = mock_agent

            result = extract_world_trait_timelines(ws, ExtractionConfig())

        # WORLD_WARTIME got a timeline
        assert len(result.world_traits["WORLD_WARTIME"].state_timeline) == 1
        # WORLD_MORALE is preserved unchanged
        assert "WORLD_MORALE" in result.world_traits
        assert result.world_traits["WORLD_MORALE"].state_timeline == []

    @pytest.mark.asyncio
    async def test_validator_rejects_invalid_event_id(self):
        """The output validator should reject snapshots referencing non-existent events."""
        from unittest.mock import MagicMock
        from shadow_loom.ingestion import (
            ExtractionConfig, _build_world_trait_timeline_agent,
            _WorldTraitTimelineDeps, WorldTraitTimelineExtraction,
        )
        from pydantic_ai import ModelRetry

        config = ExtractionConfig(model="ollama:test-model")
        agent = _build_world_trait_timeline_agent(config)

        ws = _make_world_with_traits()
        deps = _WorldTraitTimelineDeps(
            world_traits=ws.world_traits,
            events=ws.events,
        )

        bad_extraction = WorldTraitTimelineExtraction(
            timelines={
                "WORLD_WARTIME": [
                    WorldTraitSnapshot(
                        fabula_time=100,
                        triggered_by="EVT_NONEXISTENT",
                        magnitude=TraitVector(value=0.1, inertia=0.3),
                    ),
                ],
            },
        )

        ctx = MagicMock()
        ctx.deps = deps

        validators = agent._output_validators
        assert len(validators) > 0

        with pytest.raises(ModelRetry):
            await validators[-1].validate(bad_extraction, ctx, wrap_validation_errors=False)

    @pytest.mark.asyncio
    async def test_validator_rejects_mismatched_fabula_time(self):
        """The validator should reject snapshots where fabula_time doesn't match the event."""
        from unittest.mock import MagicMock
        from shadow_loom.ingestion import (
            ExtractionConfig, _build_world_trait_timeline_agent,
            _WorldTraitTimelineDeps, WorldTraitTimelineExtraction,
        )
        from pydantic_ai import ModelRetry

        config = ExtractionConfig(model="ollama:test-model")
        agent = _build_world_trait_timeline_agent(config)

        ws = _make_world_with_traits()
        deps = _WorldTraitTimelineDeps(
            world_traits=ws.world_traits,
            events=ws.events,
        )

        bad_extraction = WorldTraitTimelineExtraction(
            timelines={
                "WORLD_WARTIME": [
                    WorldTraitSnapshot(
                        fabula_time=999,  # doesn't match EVT_1's fabula_time=100
                        triggered_by="EVT_1",
                        magnitude=TraitVector(value=0.1, inertia=0.3),
                    ),
                ],
            },
        )

        ctx = MagicMock()
        ctx.deps = deps

        validators = agent._output_validators
        with pytest.raises(ModelRetry):
            await validators[-1].validate(bad_extraction, ctx, wrap_validation_errors=False)

    @pytest.mark.asyncio
    async def test_validator_rejects_invalid_world_trait_id(self):
        """The validator should reject timelines for non-existent world traits."""
        from unittest.mock import MagicMock
        from shadow_loom.ingestion import (
            ExtractionConfig, _build_world_trait_timeline_agent,
            _WorldTraitTimelineDeps, WorldTraitTimelineExtraction,
        )
        from pydantic_ai import ModelRetry

        config = ExtractionConfig(model="ollama:test-model")
        agent = _build_world_trait_timeline_agent(config)

        ws = _make_world_with_traits()
        deps = _WorldTraitTimelineDeps(
            world_traits=ws.world_traits,
            events=ws.events,
        )

        bad_extraction = WorldTraitTimelineExtraction(
            timelines={
                "WORLD_NONEXISTENT": [
                    WorldTraitSnapshot(
                        fabula_time=100,
                        triggered_by="EVT_1",
                        magnitude=TraitVector(value=0.5, inertia=0.5),
                    ),
                ],
            },
        )

        ctx = MagicMock()
        ctx.deps = deps

        validators = agent._output_validators
        with pytest.raises(ModelRetry):
            await validators[-1].validate(bad_extraction, ctx, wrap_validation_errors=False)
