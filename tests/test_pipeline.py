# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the pipeline orchestrator (``shadow_loom.pipeline``).

Every LLM call (generation, audit, re-extraction) is mocked.
All computational code (physics engines, affective calculus, graph
versioning, merge) runs un-mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shadow_loom.auditor import (
    AuditResult,
    AuditViolation,
    AuditorConfig,
    FeedbackLoopResult,
)
from shadow_loom.directive_assembly import CreativeBrief
from shadow_loom.extract_graph import (
    MergeChangeset,
    VersionedWorldModel,
    WorldModelVersion,
)
from shadow_loom.generation import GeneratedScene, GenerationConfig
from shadow_loom.ingestion import (
    ChunkTopology,
    EntityUpdate,
    ExtractionConfig,
    PhysicsExtraction,
    SocialExtraction,
    ValidationReport,
)
from shadow_loom.models import (
    CausalEdge,
    EventNode,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.pipeline import (
    PipelineConfig,
    PipelineHistory,
    PipelineResult,
    run_pipeline,
)
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
# Mock helpers
# =========================================================================

def _mock_scene(prose="The shadow fell across the courtyard."):
    return GeneratedScene(
        prose=prose,
        pov_entity="ENT_TEST",
        rendering_mode="directive",
        constraints_honoured=["C1"],
        constraints_violated=[],
    )


def _mock_passing_audit():
    return AuditResult(
        passed=True,
        violations=[],
        audit_summary="All checks passed.",
    )


def _mock_failing_audit(feedback="Fix the issue."):
    return AuditResult(
        passed=False,
        violations=[
            AuditViolation(
                violation_type="epistemic_leakage",
                severity="critical",
                description="Problem detected.",
                evidence_quote="some text",
                feedback=feedback,
            ),
        ],
        audit_summary="Violation found.",
    )


def _mock_run_sync(output):
    mock_result = MagicMock()
    mock_result.output = output
    return mock_result


def _mock_topology(ws: WorldStateV1) -> ChunkTopology:
    """Small topology: 1 event + 1 causal edge + 1 entity update."""
    first_ent_id = next(iter(ws.entities))
    max_fabula = max((e.fabula_time for e in ws.events), default=0)
    new_time = max_fabula + 1000

    return ChunkTopology(
        events=[
            EventNode(
                id=f"EVT_PIPELINE_{new_time}",
                fabula_time=new_time,
                syuzhet_index=new_time,
                event_type="outcome",
                actor_ids=[first_ent_id],
                target_ids=[],
                description="Pipeline-generated event.",
            ),
        ],
        causal_topology=[
            CausalEdge(
                source_id=ws.events[-1].id if ws.events else f"EVT_PIPELINE_{new_time}",
                target_id=f"EVT_PIPELINE_{new_time}",
                causality_type="chain_reaction",
                causal_force=5,
                mechanism="consequence",
                evidence_strength="moderate",
                propagation_delay=0,
                fabula_time=new_time,
            ),
        ],
        entity_updates=[
            EntityUpdate(
                entity_id=first_ent_id,
                fabula_time=new_time,
                triggered_by=f"EVT_PIPELINE_{new_time}",
                trait_updates={"fear": TraitVector(value=0.9, inertia=0.3)},
            ),
        ],
    )


def _deep_snapshot(ws: WorldStateV1) -> dict:
    return ws.model_dump()


# =========================================================================
# Test: Pipeline input validation
# =========================================================================

class TestPipelineInputValidation:
    """Verify the pipeline rejects invalid input combinations."""

    def test_no_input_raises(self):
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        with pytest.raises(ValueError, match="Supply one of"):
            run_pipeline(query)

    def test_multiple_inputs_raises(self):
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        with pytest.raises(ValueError, match="Supply only one"):
            run_pipeline(query, world_state=macbeth_ws, raw_text="hello")


# =========================================================================
# Test: Non-prose queries (interrogate, general)
# =========================================================================

class TestNonProseQueries:
    """Interrogation and general queries return physics state, no prose."""

    def test_interrogation_returns_physics_no_prose(self):
        query = InterrogationQuery(question="Who is guilty?", require_proof=False)
        result = run_pipeline(query, world_state=gone_girl_ws)

        assert isinstance(result, PipelineResult)
        assert result.query_type == "interrogate"
        assert result.prose is None
        assert result.scene is None
        assert result.physics_result["status"] == "success"
        assert "physics_state" in result.physics_result
        assert result.world_model is not None
        assert result.world_model.version == 0

    def test_general_returns_physics_no_prose(self):
        query = GeneralQuery(question="What are all the relationships?")
        result = run_pipeline(query, world_state=macbeth_ws)

        assert result.query_type == "general"
        assert result.prose is None
        assert result.physics_result["status"] == "success"
        ps = result.physics_result["physics_state"]
        assert "entities" in ps
        assert "locations" in ps

    def test_non_prose_records_history(self):
        query = InterrogationQuery(question="Test", require_proof=False)
        result = run_pipeline(query, world_state=macbeth_ws)

        steps = result.history.steps
        assert len(steps) >= 1
        assert steps[0]["step"] == "narrative_physics"

    def test_interrogation_world_state_immutable(self):
        snap_before = _deep_snapshot(gone_girl_ws)
        query = InterrogationQuery(question="Test", require_proof=False)
        _ = run_pipeline(query, world_state=gone_girl_ws)
        assert _deep_snapshot(gone_girl_ws) == snap_before


# =========================================================================
# Test: Observation pipeline (Rung 1)
# =========================================================================

class TestObservationPipeline:
    """Observation → physics → generation → audit → result."""

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_observation_full_pipeline(self, mock_gen_builder, mock_audit, mock_extract):
        """Full observation pipeline with mocked LLM."""
        ws = orwell_ws
        snap_before = _deep_snapshot(ws)

        # Mock generation
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("Winston stared at the telescreen.")
        )
        mock_gen_builder.return_value = mock_agent

        # Mock audit → pass immediately
        mock_audit.return_value = _mock_passing_audit()

        # Mock re-extraction
        mock_extract.return_value = _mock_topology(ws)

        query = ObservationQuery(focus_entity_ids=list(ws.entities.keys())[:2])
        result = run_pipeline(query, world_state=ws)

        assert result.query_type == "observation"
        assert result.prose is not None
        assert result.converged is True
        assert result.audit_iterations == 1
        assert result.world_model is not None
        assert result.world_model.version == 1  # merged
        assert _deep_snapshot(ws) == snap_before

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_observation_skip_audit(self, mock_gen_builder, mock_audit, mock_extract):
        """skip_audit=True skips audit loop."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Test."))
        mock_gen_builder.return_value = mock_agent
        mock_extract.return_value = _mock_topology(orwell_ws)

        query = ObservationQuery(focus_entity_ids=list(orwell_ws.entities.keys())[:2])
        result = run_pipeline(
            query, world_state=orwell_ws,
            config=PipelineConfig(skip_audit=True),
        )

        assert result.prose is not None
        assert result.converged is None  # audit skipped
        assert result.audit_iterations is None
        mock_audit.assert_not_called()


# =========================================================================
# Test: Intervention pipeline (Rung 2)
# =========================================================================

class TestInterventionPipeline:
    """Intervention → causal physics → generation → audit → merge."""

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_intervention_full_pipeline(self, mock_gen_builder, mock_audit, mock_extract):
        ws = macbeth_ws
        snap_before = _deep_snapshot(ws)

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("Macbeth's guilt dissolved as he reached for the crown.")
        )
        mock_gen_builder.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(ws)

        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.guilt.value": 0.0}
        )
        result = run_pipeline(query, world_state=ws)

        assert result.query_type == "intervention"
        assert result.prose is not None
        assert result.converged is True
        assert result.world_model.version == 1
        assert len(result.world_model.current.events) > len(ws.events)
        assert _deep_snapshot(ws) == snap_before

    @patch("shadow_loom.generation._build_generation_agent")
    def test_intervention_physics_only(self, mock_gen_builder):
        """skip_audit + skip_reextraction → physics + generation, no audit/merge."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("Macbeth reaches for the crown.")
        )
        mock_gen_builder.return_value = mock_agent

        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.guilt.value": 0.0}
        )
        result = run_pipeline(
            query, world_state=macbeth_ws,
            config=PipelineConfig(skip_audit=True, skip_reextraction=True),
        )
        assert result.physics_result["status"] == "success"
        assert result.prose is not None
        assert result.converged is None
        assert result.world_model.version == 0  # no merge


# =========================================================================
# Test: Counterfactual pipeline (Rung 3)
# =========================================================================

class TestCounterfactualPipeline:
    """Counterfactual → abduction → generation → audit → merge."""

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_counterfactual_full_pipeline(self, mock_gen_builder, mock_audit, mock_extract):
        ws = gone_girl_ws
        snap_before = _deep_snapshot(ws)

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("Had Nick never returned home, Amy's plan would have unraveled.")
        )
        mock_gen_builder.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(ws)

        first_event = ws.events[0].id if ws.events else "EVT_DUMMY"
        ent_ids = list(ws.entities.keys())[:2]
        query = CounterfactualQuery(
            historical_interventions={f"{first_event}.event_type": "prevented"},
            evidence_node_ids=ent_ids,
        )
        result = run_pipeline(query, world_state=ws)

        assert result.query_type == "counterfactual"
        assert result.prose is not None
        assert result.converged is True
        assert result.world_model.version == 1
        assert _deep_snapshot(ws) == snap_before


# =========================================================================
# Test: Directive pipeline
# =========================================================================

class TestDirectivePipeline:
    """Directive → affective calculus → creative brief → audit → merge."""

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_directive_full_pipeline(self, mock_gen_builder, mock_audit, mock_extract):
        ws = macbeth_ws
        snap_before = _deep_snapshot(ws)

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("The darkness closed around Macbeth like a fist.")
        )
        mock_gen_builder.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(ws)

        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = run_pipeline(query, world_state=ws)

        assert result.query_type == "directive"
        assert result.prose is not None
        assert result.converged is True
        assert result.world_model.version == 1
        assert len(result.world_model.current.events) > len(ws.events)
        assert _deep_snapshot(ws) == snap_before

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_directive_audit_convergence(self, mock_gen_builder, mock_audit_gen, mock_audit, mock_extract):
        """Fail once then pass → converged in 2 iterations."""
        ws = macbeth_ws

        # Initial generation
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("First draft.")
        )
        mock_gen_builder.return_value = mock_agent

        # Audit re-generation agent
        mock_rewrite_agent = MagicMock()
        mock_rewrite_agent.run_sync.return_value = _mock_run_sync(
            _mock_scene("Refined draft.")
        )
        mock_audit_gen.return_value = mock_rewrite_agent

        mock_audit.side_effect = [_mock_failing_audit(), _mock_passing_audit()]
        mock_extract.return_value = _mock_topology(ws)

        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = run_pipeline(query, world_state=ws)

        assert result.converged is True
        assert result.audit_iterations == 2

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_directive_audit_exhaustion(self, mock_gen_builder, mock_audit_gen, mock_audit, mock_extract):
        """Always fails → exhausts iterations."""
        ws = macbeth_ws

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Draft."))
        mock_gen_builder.return_value = mock_agent

        mock_rewrite = MagicMock()
        mock_rewrite.run_sync.return_value = _mock_run_sync(_mock_scene("Still bad."))
        mock_audit_gen.return_value = mock_rewrite

        mock_audit.return_value = _mock_failing_audit()
        mock_extract.return_value = _mock_topology(ws)

        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = run_pipeline(
            query, world_state=ws,
            config=PipelineConfig(auditor_config=AuditorConfig(max_iterations=2)),
        )

        assert result.converged is False
        assert result.audit_iterations == 2


# =========================================================================
# Test: Existing VersionedWorldModel input
# =========================================================================

class TestVersionedWorldModelInput:
    """Pipeline accepts and continues an existing VersionedWorldModel."""

    def test_continues_version_history(self):
        """Passing a v1 model → pipeline merges to v2."""
        ws = macbeth_ws
        vwm = VersionedWorldModel.from_world_state(ws)
        topo = _mock_topology(ws)
        vwm = vwm.merge(topo, description="Pre-pipeline merge")
        assert vwm.version == 1

        # Non-prose query so no LLM calls needed
        query = InterrogationQuery(question="Test", require_proof=False)
        result = run_pipeline(query, versioned_model=vwm)

        # No merge for non-prose, but history is preserved
        assert result.world_model.version == 1
        assert len(result.world_model.history) == 2

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_versioned_model_merge_increments(self, mock_gen, mock_audit, mock_extract):
        """v1 model → directive → merge → v2."""
        ws = macbeth_ws
        vwm = VersionedWorldModel.from_world_state(ws)
        topo1 = _mock_topology(ws)
        vwm = vwm.merge(topo1, description="Manual merge")
        assert vwm.version == 1

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Scene."))
        mock_gen.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(vwm.current)

        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = run_pipeline(query, versioned_model=vwm)

        assert result.world_model.version == 2
        assert len(result.world_model.history) == 3
        assert result.world_model.history[0].source == "original"
        assert result.world_model.history[1].source == "merge_topology"
        assert result.world_model.history[2].source == "pipeline"

    def test_versioned_model_original_not_mutated(self):
        """The VersionedWorldModel passed in is never mutated."""
        ws = macbeth_ws
        vwm = VersionedWorldModel.from_world_state(ws)
        original_version = vwm.version
        original_event_count = len(vwm.current.events)

        query = GeneralQuery(question="Test")
        _ = run_pipeline(query, versioned_model=vwm)

        assert vwm.version == original_version
        assert len(vwm.current.events) == original_event_count


# =========================================================================
# Test: Pipeline history tracking
# =========================================================================

class TestPipelineHistory:
    """Verify every step is recorded in the history."""

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_full_history_recorded(self, mock_gen, mock_audit, mock_extract):
        """All steps appear in history for a full directive pipeline."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Prose."))
        mock_gen.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(macbeth_ws)

        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = run_pipeline(query, world_state=macbeth_ws)

        step_names = [s["step"] for s in result.history.steps]
        assert "narrative_physics" in step_names
        assert "generation" in step_names
        assert "audit" in step_names
        assert "reextraction_merge" in step_names

    def test_non_prose_history_minimal(self):
        """Non-prose queries only record physics step."""
        query = InterrogationQuery(question="Test", require_proof=False)
        result = run_pipeline(query, world_state=macbeth_ws)

        step_names = [s["step"] for s in result.history.steps]
        assert "narrative_physics" in step_names
        assert "generation" not in step_names
        assert "audit" not in step_names

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_skip_reextraction_no_merge_step(self, mock_gen, mock_audit, mock_extract):
        """skip_reextraction omits the merge step from history."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Prose."))
        mock_gen.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()

        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = run_pipeline(
            query, world_state=macbeth_ws,
            config=PipelineConfig(skip_reextraction=True),
        )

        step_names = [s["step"] for s in result.history.steps]
        assert "reextraction_merge" not in step_names
        assert result.world_model.version == 0  # no merge happened
        mock_extract.assert_not_called()


# =========================================================================
# Test: Skip options
# =========================================================================

class TestPipelineSkipOptions:
    """Verify skip_audit and skip_reextraction work correctly."""

    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_skip_audit_still_extracts(self, mock_gen, mock_extract):
        """skip_audit but NOT skip_reextraction → still merges."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Quick."))
        mock_gen.return_value = mock_agent
        mock_extract.return_value = _mock_topology(macbeth_ws)

        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = run_pipeline(
            query, world_state=macbeth_ws,
            config=PipelineConfig(skip_audit=True),
        )

        assert result.prose is not None
        assert result.converged is None
        assert result.world_model.version == 1  # merge happened

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_skip_reextraction_no_merge(self, mock_gen, mock_audit):
        """skip_reextraction → version stays at 0."""
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Scene."))
        mock_gen.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()

        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
        )
        result = run_pipeline(
            query, world_state=macbeth_ws,
            config=PipelineConfig(skip_reextraction=True),
        )

        assert result.prose is not None
        assert result.converged is True
        assert result.world_model.version == 0


# =========================================================================
# Test: World state immutability across all query types
# =========================================================================

_ALL_QUERIES = [
    ("observation", ObservationQuery(focus_entity_ids=["ENT_MACBETH"])),
    ("intervention", InterventionQuery(interventions={"ENT_MACBETH.traits.guilt.value": 0.0})),
    ("counterfactual", CounterfactualQuery(
        historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
        evidence_node_ids=["ENT_MACBETH"],
    )),
    ("directive", DirectiveQuery(target_entity_ids=["ENT_MACBETH"], target_effect="fear")),
    ("interrogation", InterrogationQuery(question="Test", require_proof=False)),
    ("general", GeneralQuery(question="Tell me everything")),
]


class TestWorldStateImmutability:
    """Original world state is never mutated regardless of query type."""

    @pytest.mark.parametrize("name,query", _ALL_QUERIES)
    @patch("shadow_loom.generation._build_generation_agent")
    def test_world_state_immutable(self, mock_gen, name, query):
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Test."))
        mock_gen.return_value = mock_agent

        snap_before = _deep_snapshot(macbeth_ws)
        run_pipeline(
            query, world_state=macbeth_ws,
            config=PipelineConfig(skip_audit=True, skip_reextraction=True),
        )
        assert _deep_snapshot(macbeth_ws) == snap_before


# =========================================================================
# Test: Cross-plot parametrized
# =========================================================================

_CROSS_PLOT = [
    ("macbeth", macbeth_ws),
    ("gone_girl", gone_girl_ws),
    ("orwell", orwell_ws),
    ("brief_encounter", brief_encounter_ws),
    ("reservoir_dogs", reservoir_dogs_ws),
]


class TestCrossPlotPipeline:
    """Pipeline works across different plot model fixtures."""

    @pytest.mark.parametrize("name,ws", _CROSS_PLOT)
    def test_interrogation_across_plots(self, name, ws):
        query = InterrogationQuery(question="What happened?", require_proof=False)
        result = run_pipeline(query, world_state=ws)
        assert result.physics_result["status"] == "success"
        assert result.world_model is not None

    @pytest.mark.parametrize("name,ws", _CROSS_PLOT)
    def test_general_across_plots(self, name, ws):
        query = GeneralQuery(question="Summarise the story.")
        result = run_pipeline(query, world_state=ws)
        assert result.physics_result["status"] == "success"
        assert "physics_state" in result.physics_result

    @pytest.mark.parametrize("name,ws", _CROSS_PLOT)
    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_observation_across_plots(self, mock_gen, mock_audit, mock_extract, name, ws):
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Observed."))
        mock_gen.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(ws)

        ent_ids = list(ws.entities.keys())[:2]
        query = ObservationQuery(focus_entity_ids=ent_ids)
        result = run_pipeline(query, world_state=ws)
        assert result.prose is not None, f"No prose for {name}"
        assert result.world_model.version == 1, f"No merge for {name}"

    @pytest.mark.parametrize("name,ws", _CROSS_PLOT)
    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_intervention_across_plots(self, mock_gen, mock_audit, mock_extract, name, ws):
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Intervened."))
        mock_gen.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(ws)

        first_ent = next(iter(ws.entities))
        query = InterventionQuery(
            interventions={f"{first_ent}.traits.fear.value": 0.9},
        )
        result = run_pipeline(query, world_state=ws)
        assert result.prose is not None, f"No prose for {name}"
        assert result.world_model.version == 1, f"No merge for {name}"

    @pytest.mark.parametrize("name,ws", _CROSS_PLOT)
    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_counterfactual_across_plots(self, mock_gen, mock_audit, mock_extract, name, ws):
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("What if."))
        mock_gen.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(ws)

        first_event = ws.events[0].id if ws.events else "EVT_DUMMY"
        ent_ids = list(ws.entities.keys())[:2]
        query = CounterfactualQuery(
            historical_interventions={f"{first_event}.event_type": "prevented"},
            evidence_node_ids=ent_ids,
        )
        result = run_pipeline(query, world_state=ws)
        assert result.prose is not None, f"No prose for {name}"
        assert result.world_model.version == 1, f"No merge for {name}"

    @pytest.mark.parametrize("name,ws", _CROSS_PLOT)
    @patch("shadow_loom.pipeline.extract_topology_from_prose")
    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.generation._build_generation_agent")
    def test_directive_across_plots(self, mock_gen, mock_audit, mock_extract, name, ws):
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Scene."))
        mock_gen.return_value = mock_agent
        mock_audit.return_value = _mock_passing_audit()
        mock_extract.return_value = _mock_topology(ws)

        ent_ids = list(ws.entities.keys())[:2]
        query = DirectiveQuery(
            target_entity_ids=ent_ids,
            target_effect="suspense",
        )
        result = run_pipeline(query, world_state=ws)
        assert result.prose is not None, f"No prose for {name}"
        assert result.world_model.version == 1, f"No merge for {name}"


# =========================================================================
# Test: PipelineResult structure
# =========================================================================

class TestPipelineResultStructure:
    """Verify the PipelineResult model has expected fields."""

    def test_result_fields_present(self):
        query = GeneralQuery(question="Test")
        result = run_pipeline(query, world_state=macbeth_ws)

        assert hasattr(result, "prose")
        assert hasattr(result, "scene")
        assert hasattr(result, "physics_result")
        assert hasattr(result, "converged")
        assert hasattr(result, "audit_iterations")
        assert hasattr(result, "feedback_result")
        assert hasattr(result, "world_model")
        assert hasattr(result, "history")
        assert hasattr(result, "query_type")

    def test_result_serializable(self):
        """PipelineResult can be serialized to dict."""
        query = GeneralQuery(question="Test")
        result = run_pipeline(query, world_state=macbeth_ws)
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "query_type" in d
        assert "history" in d


# =========================================================================
# Test: Implausibility short-circuit (Rung 2/3 + directive)
# =========================================================================

class TestImplausibilityShortCircuit:
    """When the engine cannot apply a Rung-2/3 (or directive) query
    because its targets do not exist, the pipeline must explain rather
    than fail, and must NOT mutate the world model."""

    def _assert_no_mutation(self, result, ws_before_dump):
        # World model is left at v0 and the underlying state is identical.
        assert result.world_model.version == 0
        assert result.world_model.current.model_dump() == ws_before_dump

    def test_intervention_unknown_target_short_circuits(self):
        ws_before = _deep_snapshot(macbeth_ws)
        query = InterventionQuery(
            interventions={"ENT_DOES_NOT_EXIST.status": "dead"}
        )
        result = run_pipeline(query, world_state=macbeth_ws)

        assert result.implausible is True
        assert result.implausibility_reason
        assert result.prose is not None
        assert "could not be applied" in result.prose
        assert "ENT_DOES_NOT_EXIST.status" in result.prose
        assert result.scene is not None
        assert result.scene.rendering_mode == "implausible"
        # No generation/audit/extraction LLM calls should have been needed.
        assert result.converged is None
        assert result.audit_iterations is None
        assert result.feedback_result is None
        self._assert_no_mutation(result, ws_before)
        # History records the implausibility.
        steps = [s["step"] for s in result.history.steps]
        assert "implausibility" in steps
        assert "reextraction_merge" not in steps

    def test_counterfactual_unknown_target_short_circuits(self):
        ws_before = _deep_snapshot(macbeth_ws)
        query = CounterfactualQuery(
            historical_interventions={"FAKE_NODE.status": "alive"},
            evidence_node_ids=[],
        )
        result = run_pipeline(query, world_state=macbeth_ws)

        assert result.implausible is True
        assert result.prose is not None
        assert "FAKE_NODE.status" in result.prose
        self._assert_no_mutation(result, ws_before)

    def test_directive_unknown_target_short_circuits(self):
        ws_before = _deep_snapshot(macbeth_ws)
        query = DirectiveQuery(
            target_entity_ids=["ENT_NO_SUCH_PERSON"],
            target_effect="grief",
        )
        result = run_pipeline(query, world_state=macbeth_ws)

        assert result.implausible is True
        assert "ENT_NO_SUCH_PERSON" in result.prose
        self._assert_no_mutation(result, ws_before)

    def test_plausible_intervention_not_flagged(self):
        """A well-formed intervention against a real entity is not flagged."""
        first_ent = next(iter(macbeth_ws.entities))
        query = InterventionQuery(
            interventions={f"{first_ent}.status": "altered"}
        )
        cfg = PipelineConfig(skip_audit=True, skip_reextraction=True)
        with patch("shadow_loom.generation._build_generation_agent") as mock_gen_builder:
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene())
            mock_gen_builder.return_value = mock_agent
            result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        assert result.implausible is False
        assert result.implausibility_reason is None


# =========================================================================
# Test: force_implausible override
# =========================================================================

class TestForceImplausibleOverride:
    """When ``force_implausible=True`` the engine still detects the
    problem but proceeds with degraded best-effort generation, and the
    pipeline reports the warning on the result."""

    def test_intervention_force_generates_prose(self):
        query = InterventionQuery(
            interventions={"ENT_DOES_NOT_EXIST.status": "dead"},
            force_implausible=True,
        )
        cfg = PipelineConfig(skip_audit=True, skip_reextraction=True)
        with patch("shadow_loom.generation._build_generation_agent") as mock_gen_builder:
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Forced prose."))
            mock_gen_builder.return_value = mock_agent
            result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        # Flagged as implausible but prose was still generated.
        assert result.implausible is True
        assert result.implausibility_reason
        assert result.prose == "Forced prose."
        # History records the forced step.
        steps = [s for s in result.history.steps if s["step"] == "implausibility"]
        assert steps and steps[0].get("forced") is True

    def test_counterfactual_force_anchors_at_horizon(self):
        query = CounterfactualQuery(
            historical_interventions={"FAKE_NODE.status": "alive"},
            evidence_node_ids=[],
            force_implausible=True,
        )
        cfg = PipelineConfig(skip_audit=True, skip_reextraction=True)
        with patch("shadow_loom.generation._build_generation_agent") as mock_gen_builder:
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Forced cf."))
            mock_gen_builder.return_value = mock_agent
            result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        assert result.implausible is True
        assert result.prose == "Forced cf."

    def test_directive_force_uses_fallback_pov(self):
        query = DirectiveQuery(
            target_entity_ids=["ENT_NO_SUCH_PERSON"],
            target_effect="grief",
            force_implausible=True,
        )
        cfg = PipelineConfig(skip_audit=True, skip_reextraction=True)
        with patch("shadow_loom.generation._build_generation_agent") as mock_gen_builder:
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = _mock_run_sync(_mock_scene("Forced directive."))
            mock_gen_builder.return_value = mock_agent
            result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        assert result.implausible is True
        assert result.implausibility_details.get("fallback_pov") in macbeth_ws.entities
        assert result.prose == "Forced directive."
