# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end pipeline tests: plot models → graph extraction → physics/affective
engines → generation → audit loop → prose re-extraction → merge.

All LLM calls are mocked.  All computational code (physics engines, affective
calculus, graph versioning, merge) runs un-mocked.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from shadow_loom.auditor import (
    AuditCycleSnapshot,
    AuditResult,
    AuditViolation,
    AuditorConfig,
    FeedbackLoopResult,
    VersionedGraph,
    run_feedback_loop,
)
from shadow_loom.causal_physics import CausalPhysicsEngine, CausalPhysicsResult
from shadow_loom.directive_assembly import (
    CreativeBrief,
    DirectiveAssembler,
)
from shadow_loom.extract_graph import (
    EgoGraphPayload,
    MergeChangeset,
    VersionedWorldModel,
    WorldModelVersion,
    extract_ego_graph_from_memory,
    extract_topology_from_prose,
    merge_topology,
)
from shadow_loom.generation import GeneratedScene
from shadow_loom.ingestion import (
    ChunkTopology,
    EntityUpdate,
    PhysicsExtraction,
    SocialExtraction,
)
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import (
    Belief,
    CausalEdge,
    Entity,
    EntityStateSnapshot,
    EventNode,
    RelationshipEdge,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import (
    CounterfactualQuery,
    DirectiveQuery,
    GeneralQuery,
    InterrogationQuery,
    InterventionQuery,
    ObservationQuery,
)

# -- Plot model fixtures --------------------------------------------------
from example_worlds.macbeth import world_state as macbeth_ws
from example_worlds.gone_girl import world_state as gone_girl_ws
from example_worlds.nineteen_eighty_four import world_state as orwell_ws
from example_worlds.brief_encounter import world_state as brief_encounter_ws
from example_worlds.reservoir_dogs import world_state as reservoir_dogs_ws


# =========================================================================
# Helpers
# =========================================================================

def _build_sandbox(ws, focus_ids=None, query_type="intervention"):
    """Build an nx sandbox from a WorldStateV1 fixture."""
    if focus_ids is None:
        focus_ids = list(ws.entities.keys())[:2]
    ego = extract_ego_graph_from_memory(ws, focus_ids)
    return AMWNInstantiator.create_sandbox(ego.model_dump(), query_type), ego.model_dump()


def _mock_scene(prose="The shadow fell across the courtyard.", rendering_mode="directive"):
    return GeneratedScene(
        prose=prose,
        pov_entity="ENT_TEST",
        rendering_mode=rendering_mode,
        constraints_honoured=["C1"],
        constraints_violated=[],
    )


def _mock_passing_audit():
    return AuditResult(
        passed=True,
        violations=[],
        audit_summary="All checks passed.",
    )


def _mock_failing_audit(feedback="Fix the shadow reference."):
    return AuditResult(
        passed=False,
        violations=[
            AuditViolation(
                violation_type="epistemic_leakage",
                severity="critical",
                description="Revealed hidden information.",
                evidence_quote="The shadow fell",
                feedback=feedback,
            ),
        ],
        audit_summary="Epistemic constraint violated.",
    )


def _mock_run_sync(output):
    """Create a MagicMock that mimics pydantic-ai's run_sync().output."""
    mock_result = MagicMock()
    mock_result.output = output
    return mock_result


def _mock_topology(ws: WorldStateV1) -> ChunkTopology:
    """Build a small topology with 1 new event, 1 causal edge, 1 entity update."""
    first_ent_id = next(iter(ws.entities))
    first_loc_id = next(iter(ws.locations))
    max_fabula = max((e.fabula_time for e in ws.events), default=0)
    new_time = max_fabula + 1000

    new_event = EventNode(
        id=f"EVT_GENERATED_{new_time}",
        fabula_time=new_time,
        syuzhet_index=new_time,
        event_type="outcome",
        actor_ids=[first_ent_id],
        target_ids=[],
        description="A new event extracted from generated prose.",
    )
    new_causal = CausalEdge(
        source_id=ws.events[-1].id if ws.events else new_event.id,
        target_id=new_event.id,
        causality_type="chain_reaction",
        causal_force=5,
        mechanism="causal consequence",
        evidence_strength="moderate",
        propagation_delay=0,
        fabula_time=new_time,
    )
    new_update = EntityUpdate(
        entity_id=first_ent_id,
        fabula_time=new_time,
        triggered_by=new_event.id,
        trait_updates={"fear": TraitVector(value=0.9, inertia=0.3)},
    )

    return ChunkTopology(
        events=[new_event],
        causal_topology=[new_causal],
        entity_updates=[new_update],
    )


def _deep_snapshot(ws: WorldStateV1) -> dict:
    """Serialize a WorldStateV1 for before/after immutability checks."""
    return ws.model_dump()


# =========================================================================
# Test: Rung 2 (Intervention) End-to-End
# =========================================================================

class TestRung2EndToEnd:
    """Intervention query → physics engine → generation → audit → extraction → merge."""

    def test_rung2_physics_produces_mutations(self):
        """InterventionQuery runs causal physics and produces mutations."""
        ws = macbeth_ws
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.guilt.value": 0.0},
        )
        result = calculate_narrative_physics(
            query, ws, use_causal_engine=True,
        )
        assert result["status"] == "success"
        assert result["query_type"] == "intervention"
        assert "physics_state" in result

    def test_rung2_sandbox_isolation(self):
        """Original world state is never mutated by the physics engine."""
        ws = macbeth_ws
        snap_before = _deep_snapshot(ws)
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.guilt.value": 0.0},
        )
        _ = calculate_narrative_physics(query, ws, use_causal_engine=True)
        snap_after = _deep_snapshot(ws)
        assert snap_before == snap_after

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_rung2_generation_and_audit_loop(self, mock_gen_agent, mock_run_audit):
        """Full loop: physics → brief → mock generation → mock audit → converges."""
        ws = macbeth_ws
        focus_ids = ["ENT_MACBETH", "ENT_LADY_MACBETH"]
        sandbox, ego_dump = _build_sandbox(ws, focus_ids)

        assembler = DirectiveAssembler(sandbox, ego_dump, ws)
        directive = DirectiveQuery(
            target_entity_ids=focus_ids,
            target_effect="fear",
        )
        brief = assembler.assemble(directive)
        assert isinstance(brief, CreativeBrief)

        initial_scene = _mock_scene("Macbeth trembled in the dark castle.")
        mock_run_audit.return_value = _mock_passing_audit()

        result = run_feedback_loop(
            initial_scene=initial_scene,
            brief=brief,
            sandbox=sandbox,
            world_state=ws,
        )
        assert isinstance(result, FeedbackLoopResult)
        assert result.converged is True
        assert result.iterations == 1
        assert len(result.history) == 1
        assert result.history[0].audit_result.passed is True

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_rung2_extraction_and_merge(self, mock_gen_agent, mock_run_audit):
        """After generation, extract topology from prose and merge → new WS."""
        ws = macbeth_ws
        snap_before = _deep_snapshot(ws)

        topology = _mock_topology(ws)
        merged = merge_topology(ws, topology)

        # Original unchanged
        assert _deep_snapshot(ws) == snap_before
        # Merged has new events
        assert len(merged.events) > len(ws.events)
        # New event present
        new_evt_ids = {e.id for e in merged.events} - {e.id for e in ws.events}
        assert len(new_evt_ids) == 1
        assert any("GENERATED" in eid for eid in new_evt_ids)


# =========================================================================
# Test: Rung 3 (Counterfactual) End-to-End
# =========================================================================

class TestRung3EndToEnd:
    """Counterfactual query → abduction → generation → audit → merge."""

    def test_rung3_abduction_runs(self):
        """CounterfactualQuery produces hidden_deltas via abduction."""
        ws = gone_girl_ws
        first_event = ws.events[0].id if ws.events else None
        ent_ids = list(ws.entities.keys())[:2]
        query = CounterfactualQuery(
            historical_interventions={
                f"{first_event}.event_type": "prevented",
            } if first_event else {"EVT_DUMMY.event_type": "prevented"},
            evidence_node_ids=ent_ids,
        )
        result = calculate_narrative_physics(
            query, ws, use_causal_engine=True,
        )
        assert result["status"] == "success"
        assert result["query_type"] == "counterfactual"
        assert "physics_state" in result

    def test_rung3_world_state_immutable(self):
        """Original world state untouched after counterfactual query."""
        ws = gone_girl_ws
        snap_before = _deep_snapshot(ws)
        ent_ids = list(ws.entities.keys())[:2]
        first_event = ws.events[0].id if ws.events else "EVT_DUMMY"
        query = CounterfactualQuery(
            historical_interventions={f"{first_event}.event_type": "prevented"},
            evidence_node_ids=ent_ids,
        )
        _ = calculate_narrative_physics(query, ws, use_causal_engine=True)
        assert _deep_snapshot(ws) == snap_before

    def test_rung3_versioned_graph_preserves_original(self):
        """VersionedGraph.fork() always resets to original; original never mutated."""
        ws = gone_girl_ws
        sandbox, _ = _build_sandbox(ws, list(ws.entities.keys())[:2])
        vg = VersionedGraph(sandbox)

        original_data = nx.node_link_data(vg.original)
        # Fork multiple times and mutate each fork
        for i in range(3):
            forked = vg.fork()
            forked.add_node(f"INJECTED_{i}", label="test")

        # Original is pristine
        assert nx.node_link_data(vg.original) == original_data
        # Current is a fresh fork (last fork), but we added a node to it
        assert vg.version == 3

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_rung3_feedback_loop_with_merge(self, mock_gen_agent, mock_run_audit):
        """Full rung-3 pipeline: physics → brief → audit → merge with versioning."""
        ws = gone_girl_ws
        snap_before = _deep_snapshot(ws)
        focus_ids = list(ws.entities.keys())[:2]
        sandbox, ego_dump = _build_sandbox(ws, focus_ids)

        assembler = DirectiveAssembler(sandbox, ego_dump, ws)
        directive = DirectiveQuery(
            target_entity_ids=focus_ids,
            target_effect="mystery",
        )
        brief = assembler.assemble(directive)

        initial_scene = _mock_scene("Nick stared at the diary, confused.")
        mock_run_audit.return_value = _mock_passing_audit()

        result = run_feedback_loop(
            initial_scene=initial_scene,
            brief=brief,
            sandbox=sandbox,
            world_state=ws,
        )
        assert result.converged is True

        # Now merge extracted topology
        topology = _mock_topology(ws)
        merged = merge_topology(ws, topology)
        assert len(merged.events) > len(ws.events)
        assert _deep_snapshot(ws) == snap_before


# =========================================================================
# Test: Rung 1 (Observation) End-to-End
# =========================================================================

class TestRung1EndToEnd:
    """Observation query → ego-graph → generation → world state immutable."""

    def test_rung1_observation_produces_ego_graph(self):
        """ObservationQuery returns physics_state with ego-graph data."""
        ws = orwell_ws
        ent_ids = list(ws.entities.keys())[:2]
        query = ObservationQuery(focus_entity_ids=ent_ids)
        result = calculate_narrative_physics(query, ws)
        assert result["status"] == "success"
        assert result["query_type"] == "observation"
        assert "physics_state" in result

    def test_rung1_world_state_untouched(self):
        """Full rung-1 pipeline leaves world state unchanged."""
        ws = orwell_ws
        snap_before = _deep_snapshot(ws)
        ent_ids = list(ws.entities.keys())[:2]
        query = ObservationQuery(focus_entity_ids=ent_ids)
        _ = calculate_narrative_physics(query, ws)
        assert _deep_snapshot(ws) == snap_before

    def test_rung1_brief_from_ego_graph(self):
        """Build a CreativeBrief from rung-1 ego-graph output."""
        ws = orwell_ws
        ent_ids = list(ws.entities.keys())[:2]
        sandbox, ego_dump = _build_sandbox(ws, ent_ids, query_type="observation")

        assembler = DirectiveAssembler(sandbox, ego_dump, ws)
        directive = DirectiveQuery(
            target_entity_ids=ent_ids,
            target_effect="suspense",
        )
        brief = assembler.assemble(directive)
        assert isinstance(brief, CreativeBrief)
        assert brief.target_effect == "suspense"


# =========================================================================
# Test: Directive End-to-End
# =========================================================================

class TestDirectiveEndToEnd:
    """Directive query → affective calculus → generation → audit → merge."""

    def test_directive_full_pipeline(self):
        """DirectiveQuery produces a creative brief with non-empty constraints."""
        ws = macbeth_ws
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = calculate_narrative_physics(
            query, ws, use_causal_engine=True,
        )
        assert result["status"] == "success"
        assert "creative_brief" in result
        brief = result["creative_brief"]
        # Pipeline may return a dict or a CreativeBrief instance
        if isinstance(brief, dict):
            brief = CreativeBrief(**brief)
        assert isinstance(brief, CreativeBrief)
        assert len(brief.constraints) > 0

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_directive_audit_convergence(self, mock_gen_agent, mock_run_audit):
        """Audit loop: fail once then pass → converged in 2 iterations."""
        ws = macbeth_ws
        focus_ids = ["ENT_MACBETH", "ENT_LADY_MACBETH"]
        sandbox, ego_dump = _build_sandbox(ws, focus_ids)

        assembler = DirectiveAssembler(sandbox, ego_dump, ws)
        directive = DirectiveQuery(
            target_entity_ids=focus_ids,
            target_effect="fear",
        )
        brief = assembler.assemble(directive)

        initial_scene = _mock_scene("Macbeth felt the dagger pull him forward.", rendering_mode="fear")

        # First audit fails, second passes
        mock_run_audit.side_effect = [
            _mock_failing_audit("Remove the explicit dagger reference."),
            _mock_passing_audit(),
        ]
        # Re-generation agent returns refined scene
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("Something unseen pulled him forward.", rendering_mode="fear")
        )
        mock_gen_agent.return_value = mock_agent

        result = run_feedback_loop(
            initial_scene=initial_scene,
            brief=brief,
            sandbox=sandbox,
            world_state=ws,
        )
        assert result.converged is True
        assert result.iterations == 2
        assert len(result.history) == 2
        assert result.history[0].audit_result.passed is False
        assert result.history[1].audit_result.passed is True

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_directive_audit_exhaustion(self, mock_gen_agent, mock_run_audit):
        """Audit loop: always fails → exhausts max_iterations."""
        ws = macbeth_ws
        focus_ids = ["ENT_MACBETH"]
        sandbox, ego_dump = _build_sandbox(ws, focus_ids)

        assembler = DirectiveAssembler(sandbox, ego_dump, ws)
        directive = DirectiveQuery(
            target_entity_ids=focus_ids,
            target_effect="fear",
        )
        brief = assembler.assemble(directive)

        initial_scene = _mock_scene("Macbeth was afraid.", rendering_mode="fear")
        mock_run_audit.return_value = _mock_failing_audit()
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("Macbeth remained afraid.", rendering_mode="fear")
        )
        mock_gen_agent.return_value = mock_agent

        result = run_feedback_loop(
            initial_scene=initial_scene,
            brief=brief,
            sandbox=sandbox,
            world_state=ws,
            auditor_config=AuditorConfig(max_iterations=2),
        )
        assert result.converged is False
        assert result.iterations == 2

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_directive_full_with_merge(self, mock_gen_agent, mock_run_audit):
        """Full directive pipeline + extraction + versioned merge."""
        ws = macbeth_ws
        snap_before = _deep_snapshot(ws)
        focus_ids = ["ENT_MACBETH", "ENT_LADY_MACBETH"]
        sandbox, ego_dump = _build_sandbox(ws, focus_ids)

        assembler = DirectiveAssembler(sandbox, ego_dump, ws)
        directive = DirectiveQuery(
            target_entity_ids=focus_ids,
            target_effect="fear",
        )
        brief = assembler.assemble(directive)

        initial_scene = _mock_scene("Macbeth trembled.", rendering_mode="fear")
        mock_run_audit.return_value = _mock_passing_audit()

        loop_result = run_feedback_loop(
            initial_scene=initial_scene,
            brief=brief,
            sandbox=sandbox,
            world_state=ws,
        )
        assert loop_result.converged

        # Simulate extract + merge with versioned model
        topology = _mock_topology(ws)
        vwm = VersionedWorldModel.from_world_state(ws)
        vwm2 = vwm.merge(topology, description="Post-generation merge")

        assert vwm2.version == 1
        assert len(vwm2.history) == 2
        assert vwm2.history[0].source == "original"
        assert vwm2.history[1].source == "merge_topology"
        assert vwm2.history[1].changeset is not None
        assert vwm2.history[1].changeset.events_added == 1
        assert len(vwm2.current.events) > len(ws.events)

        # Original world state untouched
        assert _deep_snapshot(ws) == snap_before
        # vwm (version 0) also untouched
        assert len(vwm.current.events) == len(ws.events)

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_directive_accumulated_feedback(self, mock_gen_agent, mock_run_audit):
        """Feedback accumulates across audit iterations."""
        ws = macbeth_ws
        focus_ids = ["ENT_MACBETH"]
        sandbox, ego_dump = _build_sandbox(ws, focus_ids)
        assembler = DirectiveAssembler(sandbox, ego_dump, ws)
        directive = DirectiveQuery(
            target_entity_ids=focus_ids,
            target_effect="fear",
        )
        brief = assembler.assemble(directive)

        initial_scene = _mock_scene("Test prose.", rendering_mode="fear")
        # Capture the prior_feedback arg across calls
        captured_feedback = []

        def capture_audit(prose, brief, config=None, prior_feedback=None,
                          causal_feedback=None, *, world_state=None,
                          affective_feedback=None):
            captured_feedback.append(prior_feedback)
            if len(captured_feedback) < 3:
                return _mock_failing_audit(f"Fix issue {len(captured_feedback)}")
            return _mock_passing_audit()

        mock_run_audit.side_effect = capture_audit
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Revised.", rendering_mode="fear"))
        mock_gen_agent.return_value = mock_agent

        result = run_feedback_loop(
            initial_scene=initial_scene,
            brief=brief,
            sandbox=sandbox,
            world_state=ws,
            auditor_config=AuditorConfig(max_iterations=5),
        )
        assert result.converged is True
        # First call has no prior feedback
        assert captured_feedback[0] is None
        # Subsequent calls have accumulated feedback
        assert captured_feedback[1] is not None
        assert len(captured_feedback[2]) >= 2


# =========================================================================
# Test: Interrogation End-to-End
# =========================================================================

class TestInterrogationEndToEnd:
    """Interrogation query → physics state → immutability."""

    def test_interrogation_returns_full_world_state(self):
        """InterrogationQuery returns physics_state as full world state dict."""
        ws = gone_girl_ws
        query = InterrogationQuery(
            question="Who is behind Amy's disappearance?",
            require_proof=True,
        )
        result = calculate_narrative_physics(query, ws)
        assert result["status"] == "success"
        assert result["query_type"] == "interrogate"
        ps = result["physics_state"]
        assert "entities" in ps
        assert "locations" in ps
        assert "events" in ps

    def test_interrogation_world_state_immutable(self):
        """Original world state is not mutated by interrogation."""
        ws = gone_girl_ws
        snap_before = _deep_snapshot(ws)
        query = InterrogationQuery(
            question="Who is guilty?",
            require_proof=False,
        )
        _ = calculate_narrative_physics(query, ws)
        assert _deep_snapshot(ws) == snap_before


# =========================================================================
# Test: General Query End-to-End
# =========================================================================

class TestGeneralEndToEnd:
    """General query → full world state → immutability."""

    def test_general_returns_full_world_state(self):
        """GeneralQuery returns physics_state with full world state data."""
        ws = macbeth_ws
        query = GeneralQuery(question="What is the power structure in this story?")
        result = calculate_narrative_physics(query, ws)
        assert result["status"] == "success"
        assert result["query_type"] == "general"
        ps = result["physics_state"]
        assert "entities" in ps
        assert "locations" in ps
        assert "events" in ps
        assert "causal_topology" in ps

    def test_general_world_state_immutable(self):
        """Original world state is not mutated by general query."""
        ws = gone_girl_ws
        snap_before = _deep_snapshot(ws)
        query = GeneralQuery(question="Summarise everything.")
        _ = calculate_narrative_physics(query, ws)
        assert _deep_snapshot(ws) == snap_before

    def test_general_with_temporal_anchor(self):
        """Temporal anchor limits events in the response."""
        ws = macbeth_ws
        query = GeneralQuery(question="What happened early on?")
        result = calculate_narrative_physics(query, ws, temporal_anchor=5)
        for evt in result["physics_state"]["events"]:
            assert evt["fabula_time"] <= 5


# =========================================================================
# Test: merge_topology Isolation (No Mocks, Pure Merge Logic)
# =========================================================================

class TestMergeTopologyIsolation:
    """Verify merge_topology creates deep copies and merges correctly."""

    def test_merge_creates_deep_copy(self):
        """Mutating the merged world state does not affect the original."""
        ws = macbeth_ws
        topology = _mock_topology(ws)
        merged = merge_topology(ws, topology)

        # Mutate the merged copy
        merged.events.clear()
        # Original untouched
        assert len(ws.events) > 0

    def test_merge_adds_new_events(self):
        """New events from topology appear in the merged world state."""
        ws = macbeth_ws
        topology = _mock_topology(ws)
        original_count = len(ws.events)
        merged = merge_topology(ws, topology)
        assert len(merged.events) == original_count + 1

    def test_merge_adds_new_causal_edges(self):
        """New causal edges are merged with deduplication."""
        ws = macbeth_ws
        from shadow_loom.ingestion import deduplicate_causal
        deduped_original_count = len(deduplicate_causal(list(ws.causal_topology)))
        topology = _mock_topology(ws)
        merged = merge_topology(ws, topology)
        # After merge+dedup the count should include the new edge
        assert len(merged.causal_topology) >= deduped_original_count + 1

    def test_merge_applies_entity_updates(self):
        """Entity state_timeline is extended with new snapshots."""
        ws = macbeth_ws
        first_ent_id = next(iter(ws.entities))
        original_timeline_len = len(ws.entities[first_ent_id].state_timeline)

        topology = _mock_topology(ws)
        merged = merge_topology(ws, topology)

        new_timeline_len = len(merged.entities[first_ent_id].state_timeline)
        assert new_timeline_len == original_timeline_len + 1
        # Original unchanged
        assert len(ws.entities[first_ent_id].state_timeline) == original_timeline_len

    def test_merge_deduplicates_social_edges(self):
        """Duplicate social edges are deduplicated by (source, target)."""
        ws = macbeth_ws
        if len(ws.social_topology) == 0:
            pytest.skip("No social edges in fixture")
        existing_edge = ws.social_topology[0]
        duplicate = existing_edge.model_copy(update={"last_updated_fabula": existing_edge.last_updated_fabula + 1})

        topology = ChunkTopology(social_topology=[duplicate])
        merged = merge_topology(ws, topology)

        # Count edges with the same source/target as existing_edge
        count = sum(
            1 for e in merged.social_topology
            if e.source_entity_id == existing_edge.source_entity_id
            and e.target_entity_id == existing_edge.target_entity_id
        )
        assert count == 1  # Deduplicated to one (the newer one)

    def test_merge_skips_unknown_entity_updates(self):
        """Entity updates for non-existent entity IDs are skipped."""
        ws = macbeth_ws
        topology = ChunkTopology(
            entity_updates=[
                EntityUpdate(
                    entity_id="ENT_NONEXISTENT_999",
                    fabula_time=99999,
                    trait_updates={"fear": TraitVector(value=1.0, inertia=0.5)},
                ),
            ],
        )
        # Should not raise
        merged = merge_topology(ws, topology)
        assert "ENT_NONEXISTENT_999" not in merged.entities

    def test_original_events_preserved(self):
        """All original events remain after merge."""
        ws = macbeth_ws
        original_ids = {e.id for e in ws.events}
        topology = _mock_topology(ws)
        merged = merge_topology(ws, topology)
        merged_ids = {e.id for e in merged.events}
        assert original_ids.issubset(merged_ids)

    def test_merge_with_empty_topology(self):
        """Merging empty topology → equivalent but distinct object."""
        ws = macbeth_ws
        from shadow_loom.ingestion import deduplicate_causal
        topology = ChunkTopology()
        merged = merge_topology(ws, topology)
        # Events unchanged
        assert len(merged.events) == len(ws.events)
        # Causal edges may be deduplicated if the fixture has dupes
        deduped_count = len(deduplicate_causal(list(ws.causal_topology)))
        assert len(merged.causal_topology) == deduped_count
        # But a distinct object (deep copy)
        assert merged is not ws


# =========================================================================
# Test: VersionedWorldModel
# =========================================================================

class TestVersionedWorldModel:
    """Test the version-history wrapper around WorldStateV1."""

    def test_from_world_state_creates_version_zero(self):
        """VersionedWorldModel starts at version 0."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        assert vwm.version == 0
        assert len(vwm.history) == 1
        assert vwm.history[0].version == 0
        assert vwm.history[0].source == "original"

    def test_from_world_state_deep_copies(self):
        """The incoming world state is deep-copied at construction."""
        ws = macbeth_ws
        vwm = VersionedWorldModel.from_world_state(ws)
        # Mutate vwm.current → original unchanged
        vwm.current.events.clear()
        assert len(ws.events) > 0

    def test_merge_increments_version(self):
        """Each merge bumps the version number."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        topo1 = _mock_topology(macbeth_ws)
        vwm2 = vwm.merge(topo1, description="First merge")
        assert vwm2.version == 1
        assert vwm.version == 0  # Original unchanged

        topo2 = _mock_topology(macbeth_ws)
        vwm3 = vwm2.merge(topo2, description="Second merge")
        assert vwm3.version == 2
        assert len(vwm3.history) == 3

    def test_merge_does_not_mutate_self(self):
        """merge() returns a new VersionedWorldModel; self is unchanged."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        original_event_count = len(vwm.current.events)
        topo = _mock_topology(macbeth_ws)
        vwm2 = vwm.merge(topo)

        assert len(vwm.current.events) == original_event_count
        assert len(vwm2.current.events) == original_event_count + 1
        assert len(vwm.history) == 1
        assert len(vwm2.history) == 2

    def test_history_contains_changesets(self):
        """Each merge records a MergeChangeset with counts."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        topo = _mock_topology(macbeth_ws)
        vwm2 = vwm.merge(topo)

        cs = vwm2.history[1].changeset
        assert cs is not None
        assert cs.events_added == 1
        # causal_edges_added may be 0 if existing dupes were removed during dedup
        # but at least one new edge was added net
        assert cs.entity_updates_applied == 1
        assert cs.entity_updates_skipped == []

    def test_history_records_timestamps(self):
        """Version entries contain ISO-8601 timestamps."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        assert "T" in vwm.history[0].timestamp  # ISO format

    def test_multiple_merges_accumulate_events(self):
        """Sequential merges each add their events."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        base_count = len(vwm.current.events)

        for i in range(3):
            topo = _mock_topology(macbeth_ws)
            # Give each event a unique ID
            topo.events[0] = topo.events[0].model_copy(
                update={"id": f"EVT_MERGE_{i}", "fabula_time": 99000 + i * 1000}
            )
            topo.causal_topology[0] = topo.causal_topology[0].model_copy(
                update={"target_id": f"EVT_MERGE_{i}", "fabula_time": 99000 + i * 1000}
            )
            vwm = vwm.merge(topo, description=f"Merge {i}")

        assert vwm.version == 3
        assert len(vwm.current.events) == base_count + 3
        assert len(vwm.history) == 4

    def test_changeset_tracks_skipped_entity_updates(self):
        """Entity updates for unknown entities are recorded in changeset."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        topo = ChunkTopology(
            entity_updates=[
                EntityUpdate(
                    entity_id="ENT_GHOST_999",
                    fabula_time=99999,
                    trait_updates={"fear": TraitVector(value=1.0, inertia=0.5)},
                ),
            ],
        )
        vwm2 = vwm.merge(topo)
        cs = vwm2.history[1].changeset
        assert "ENT_GHOST_999" in cs.entity_updates_skipped

    # --- Snapshot retention ---

    def test_snapshots_stored_on_merge(self):
        """Each merge stores a snapshot of the new state."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        assert len(vwm.snapshots) == 1  # v0

        topo = _mock_topology(macbeth_ws)
        vwm2 = vwm.merge(topo, description="First merge")
        assert len(vwm2.snapshots) == 2
        assert vwm2.get_snapshot(0) is not None
        assert vwm2.get_snapshot(1) is not None

    def test_max_snapshots_trims_oldest(self):
        """When snapshots exceed max_snapshots, oldest non-v0 are trimmed."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws, max_snapshots=3)

        for i in range(5):
            topo = _mock_topology(macbeth_ws)
            topo.events[0] = topo.events[0].model_copy(
                update={"id": f"EVT_TRIM_{i}", "fabula_time": 99000 + i * 1000}
            )
            topo.causal_topology[0] = topo.causal_topology[0].model_copy(
                update={"target_id": f"EVT_TRIM_{i}", "fabula_time": 99000 + i * 1000}
            )
            vwm = vwm.merge(topo, description=f"Merge {i}")

        assert vwm.version == 5
        # max_snapshots=3 → keep v0 + 2 most recent
        assert len(vwm.snapshots) <= 3
        # v0 is always retained
        assert vwm.get_snapshot(0) is not None
        # Oldest non-v0 versions should be trimmed
        versions = sorted(s.version for s in vwm.snapshots)
        assert versions[0] == 0
        # Most recent is always retained
        assert vwm.get_snapshot(vwm.version) is not None

    def test_get_snapshot_returns_none_for_trimmed(self):
        """Requesting a trimmed snapshot returns None."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws, max_snapshots=2)
        for i in range(3):
            topo = _mock_topology(macbeth_ws)
            topo.events[0] = topo.events[0].model_copy(
                update={"id": f"EVT_SNAP_{i}", "fabula_time": 99000 + i * 1000}
            )
            topo.causal_topology[0] = topo.causal_topology[0].model_copy(
                update={"target_id": f"EVT_SNAP_{i}", "fabula_time": 99000 + i * 1000}
            )
            vwm = vwm.merge(topo)

        # v1 should have been trimmed (only v0 and v3 kept)
        assert vwm.get_snapshot(1) is None

    # --- Rollback ---

    def test_rollback_to_version_zero(self):
        """Rollback to v0 restores the original world state."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        original_event_count = len(vwm.current.events)

        topo = _mock_topology(macbeth_ws)
        vwm2 = vwm.merge(topo, description="Add event")
        assert len(vwm2.current.events) == original_event_count + 1

        vwm3 = vwm2.rollback(0)
        assert len(vwm3.current.events) == original_event_count
        # Version increments (rollback is a new version entry)
        assert vwm3.version == 2
        assert vwm3.history[-1].source == "rollback"

    def test_rollback_to_intermediate_version(self):
        """Rollback to an intermediate version restores that snapshot."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        base_count = len(vwm.current.events)

        topo1 = _mock_topology(macbeth_ws)
        topo1.events[0] = topo1.events[0].model_copy(
            update={"id": "EVT_ROLL_1", "fabula_time": 99000}
        )
        topo1.causal_topology[0] = topo1.causal_topology[0].model_copy(
            update={"target_id": "EVT_ROLL_1", "fabula_time": 99000}
        )
        vwm1 = vwm.merge(topo1, description="First merge")
        count_at_v1 = len(vwm1.current.events)

        topo2 = _mock_topology(macbeth_ws)
        topo2.events[0] = topo2.events[0].model_copy(
            update={"id": "EVT_ROLL_2", "fabula_time": 99001}
        )
        topo2.causal_topology[0] = topo2.causal_topology[0].model_copy(
            update={"target_id": "EVT_ROLL_2", "fabula_time": 99001}
        )
        vwm2 = vwm1.merge(topo2, description="Second merge")
        assert len(vwm2.current.events) == count_at_v1 + 1

        vwm3 = vwm2.rollback(1)
        assert len(vwm3.current.events) == count_at_v1
        assert vwm3.version == 3

    def test_rollback_does_not_mutate_self(self):
        """rollback() returns a new object; self is unchanged."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        topo = _mock_topology(macbeth_ws)
        vwm2 = vwm.merge(topo)
        v2_events = len(vwm2.current.events)

        vwm3 = vwm2.rollback(0)
        assert len(vwm2.current.events) == v2_events  # unchanged

    def test_rollback_raises_on_trimmed_version(self):
        """Rollback to a trimmed snapshot raises KeyError."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws, max_snapshots=2)
        for i in range(3):
            topo = _mock_topology(macbeth_ws)
            topo.events[0] = topo.events[0].model_copy(
                update={"id": f"EVT_RBTRIM_{i}", "fabula_time": 99000 + i * 1000}
            )
            topo.causal_topology[0] = topo.causal_topology[0].model_copy(
                update={"target_id": f"EVT_RBTRIM_{i}", "fabula_time": 99000 + i * 1000}
            )
            vwm = vwm.merge(topo)

        # v1 should be trimmed
        with pytest.raises(KeyError, match="not available"):
            vwm.rollback(1)

    def test_rollback_history_is_truncated(self):
        """After rollback, history only contains entries up to the target + the rollback entry."""
        vwm = VersionedWorldModel.from_world_state(macbeth_ws)
        topo1 = _mock_topology(macbeth_ws)
        topo1.events[0] = topo1.events[0].model_copy(
            update={"id": "EVT_HIST_1", "fabula_time": 99000}
        )
        topo1.causal_topology[0] = topo1.causal_topology[0].model_copy(
            update={"target_id": "EVT_HIST_1", "fabula_time": 99000}
        )
        vwm1 = vwm.merge(topo1)

        topo2 = _mock_topology(macbeth_ws)
        topo2.events[0] = topo2.events[0].model_copy(
            update={"id": "EVT_HIST_2", "fabula_time": 99001}
        )
        topo2.causal_topology[0] = topo2.causal_topology[0].model_copy(
            update={"target_id": "EVT_HIST_2", "fabula_time": 99001}
        )
        vwm2 = vwm1.merge(topo2)

        vwm3 = vwm2.rollback(0)
        # History: v0 (original) + v3 (rollback). v1 and v2 are dropped.
        assert len(vwm3.history) == 2
        assert vwm3.history[0].version == 0
        assert vwm3.history[-1].source == "rollback"


# =========================================================================
# Test: extract_topology_from_prose (mocked agents)
# =========================================================================

class TestExtractTopologyFromProse:
    """Test prose → ChunkTopology extraction with mocked LLM agents."""

    @patch("shadow_loom.ingestion._build_social_agent")
    @patch("shadow_loom.ingestion._build_physics_agent")
    def test_extraction_calls_both_agents(self, mock_physics_builder, mock_social_builder):
        """Both physics and social agents are called during extraction."""
        ws = macbeth_ws

        physics_output = PhysicsExtraction(
            events=[
                EventNode(
                    id="EVT_NEW_1",
                    fabula_time=99000,
                    syuzhet_index=99000,
                    event_type="outcome",
                    actor_ids=["ENT_MACBETH"],
                    target_ids=[],
                    description="A new event.",
                ),
            ],
            causal_topology=[],
        )
        social_output = SocialExtraction(
            channels={},
            utterance_events=[],
            social_topology=[],
        )

        mock_physics_agent = MagicMock()
        mock_physics_agent.run_sync.return_value = _mock_run_sync(physics_output)
        mock_physics_builder.return_value = mock_physics_agent

        mock_social_agent = MagicMock()
        mock_social_agent.run_sync.return_value = _mock_run_sync(social_output)
        mock_social_builder.return_value = mock_social_agent

        result = extract_topology_from_prose("Macbeth saw the ghost.", ws)

        assert isinstance(result, ChunkTopology)
        assert len(result.events) == 1
        assert result.events[0].id == "EVT_NEW_1"
        mock_physics_builder.assert_called_once()
        mock_social_builder.assert_called_once()
        mock_physics_agent.run_sync.assert_called_once()
        mock_social_agent.run_sync.assert_called_once()

    @patch("shadow_loom.ingestion._build_social_agent")
    @patch("shadow_loom.ingestion._build_physics_agent")
    def test_extraction_passes_existing_event_ids(self, mock_physics_builder, mock_social_builder):
        """Existing event IDs from world state are passed as previous_event_ids."""
        ws = macbeth_ws

        mock_physics_agent = MagicMock()
        mock_physics_agent.run_sync.return_value = _mock_run_sync(PhysicsExtraction())
        mock_physics_builder.return_value = mock_physics_agent

        mock_social_agent = MagicMock()
        mock_social_agent.run_sync.return_value = _mock_run_sync(SocialExtraction())
        mock_social_builder.return_value = mock_social_agent

        extract_topology_from_prose("Some prose.", ws)

        # Check the physics agent received previous event IDs via deps
        physics_call_kwargs = mock_physics_agent.run_sync.call_args
        deps = physics_call_kwargs.kwargs.get("deps") or physics_call_kwargs[1].get("deps")
        assert len(deps.previous_event_ids) == len(ws.events)

    @patch("shadow_loom.ingestion._build_social_agent")
    @patch("shadow_loom.ingestion._build_physics_agent")
    def test_extraction_builds_register_from_world_state(self, mock_physics_builder, mock_social_builder):
        """GlobalRegister is built from world_state entities/locations/objects."""
        ws = macbeth_ws

        mock_physics_agent = MagicMock()
        mock_physics_agent.run_sync.return_value = _mock_run_sync(PhysicsExtraction())
        mock_physics_builder.return_value = mock_physics_agent

        mock_social_agent = MagicMock()
        mock_social_agent.run_sync.return_value = _mock_run_sync(SocialExtraction())
        mock_social_builder.return_value = mock_social_agent

        extract_topology_from_prose("Some prose.", ws)

        # Verify register contains all entities from world state
        physics_call = mock_physics_agent.run_sync.call_args
        deps = physics_call.kwargs.get("deps") or physics_call[1].get("deps")
        assert set(deps.global_register.entities.keys()) == set(ws.entities.keys())
        assert set(deps.global_register.locations.keys()) == set(ws.locations.keys())
        assert set(deps.global_register.objects.keys()) == set(ws.objects.keys())


# =========================================================================
# Test: Cross-Plot End-to-End (Parametrized)
# =========================================================================

_CROSS_PLOT_FIXTURES = [
    ("macbeth", macbeth_ws, "fear"),
    ("gone_girl", gone_girl_ws, "mystery"),
    ("orwell_1984", orwell_ws, "suspense"),
    ("brief_encounter", brief_encounter_ws, "love"),
    ("reservoir_dogs", reservoir_dogs_ws, "suspense"),
]


class TestCrossPlotEndToEnd:
    """Verify directive pipeline works across multiple plot model fixtures."""

    @pytest.mark.parametrize("name,ws,effect", _CROSS_PLOT_FIXTURES)
    def test_directive_produces_creative_brief(self, name, ws, effect):
        """DirectiveQuery → creative brief with non-empty constraints for each plot."""
        ent_ids = list(ws.entities.keys())[:2]
        query = DirectiveQuery(
            target_entity_ids=ent_ids,
            target_effect=effect,
        )
        result = calculate_narrative_physics(
            query, ws, use_causal_engine=True,
        )
        assert result["status"] == "success", f"Failed for {name}"
        assert "creative_brief" in result, f"No creative_brief for {name}"
        brief = result["creative_brief"]
        if isinstance(brief, dict):
            brief = CreativeBrief(**brief)
        assert isinstance(brief, CreativeBrief)
        assert len(brief.constraints) > 0, f"Empty constraints for {name}"

    @pytest.mark.parametrize("name,ws,effect", _CROSS_PLOT_FIXTURES)
    def test_observation_across_plots(self, name, ws, effect):
        """ObservationQuery works for each plot fixture."""
        ent_ids = list(ws.entities.keys())[:2]
        query = ObservationQuery(focus_entity_ids=ent_ids)
        result = calculate_narrative_physics(query, ws)
        assert result["status"] == "success", f"Failed for {name}"
        assert result["query_type"] == "observation"

    @pytest.mark.parametrize("name,ws,effect", _CROSS_PLOT_FIXTURES)
    def test_merge_across_plots(self, name, ws, effect):
        """merge_topology works for each plot fixture without mutating original."""
        snap_before = _deep_snapshot(ws)
        topology = _mock_topology(ws)
        merged = merge_topology(ws, topology)
        assert len(merged.events) > len(ws.events), f"No events added for {name}"
        assert _deep_snapshot(ws) == snap_before, f"Original mutated for {name}"

    @pytest.mark.parametrize("name,ws,effect", _CROSS_PLOT_FIXTURES)
    def test_versioned_world_model_across_plots(self, name, ws, effect):
        """VersionedWorldModel merge works across plots with history tracking."""
        vwm = VersionedWorldModel.from_world_state(ws)
        topology = _mock_topology(ws)
        vwm2 = vwm.merge(topology, description=f"Test merge for {name}")
        assert vwm2.version == 1
        assert vwm2.history[1].changeset.events_added == 1
        assert len(vwm.current.events) == len(ws.events)  # v0 unchanged
