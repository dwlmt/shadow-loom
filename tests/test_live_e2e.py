"""
Live end-to-end integration tests for the full Shadow-Loom pipeline.

These tests run real LLM calls via Ollama against pre-built plot fixtures
from ``example_worlds``.  They exercise every query type through
the complete pipeline: narrative physics → generation → audit/feedback →
re-extraction → versioned merge.

**Requirements:**
  - A running Ollama instance at ``localhost:11434``
  - The ``qwen3.6:27b`` model pulled

Skipped automatically when Ollama is unreachable.
"""

from __future__ import annotations

import urllib.request
import pytest

from shadow_loom.auditor import AuditorConfig, FeedbackLoopResult
from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.generation import GenerationConfig
from shadow_loom.ingestion import ExtractionConfig
from shadow_loom.models import WorldStateV1
from shadow_loom.pipeline import PipelineConfig, PipelineResult, run_pipeline
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

# =========================================================================
# Skip if Ollama is not reachable
# =========================================================================

def _ollama_available() -> bool:
    try:
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return resp.status == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not reachable at localhost:11434",
)

# =========================================================================
# Shared config — keep max_iterations low for test speed
# =========================================================================

_MODEL = "ollama:qwen3.6:27b"


def _test_pipeline_config(*, skip_audit: bool = False, skip_reextraction: bool = False) -> PipelineConfig:
    return PipelineConfig(
        use_causal_engine=True,
        generation_config=GenerationConfig(model=_MODEL, max_tokens=1024),
        auditor_config=AuditorConfig(
            auditor_model=_MODEL,
            generation_model=_MODEL,
            max_iterations=1,
            max_tokens_audit=1024,
            max_tokens_generation=1024,
        ),
        skip_audit=skip_audit,
        skip_reextraction=skip_reextraction,
        extraction_config=ExtractionConfig(model=_MODEL),
    )


# =========================================================================
# Assertion helpers
# =========================================================================

def _assert_valid_pipeline_result(result: PipelineResult, *, expect_prose: bool):
    """Common assertions for all pipeline results."""
    assert result is not None
    assert result.physics_result is not None
    assert result.world_model is not None
    if expect_prose:
        assert result.prose is not None
        assert len(result.prose) > 20, f"Prose too short: {result.prose!r}"
        assert result.scene is not None
    else:
        # interrogate / general don't produce prose
        assert result.physics_result.get("status") is not None


# =========================================================================
# 1. OBSERVATION (Rung 1)
# =========================================================================

@requires_ollama
class TestObservationE2E:
    """Observation query — natural progression from the ego-graph."""

    def test_observation_macbeth(self):
        """Observe from Macbeth's POV, generate prose, audit, and merge."""
        query = ObservationQuery(
            focus_entity_ids=["ENT_MACBETH"],
            observations={"ENT_MACBETH": "troubled by guilt"},
        )
        cfg = _test_pipeline_config()
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        assert result.query_type == "observation"
        # Should have gone through audit
        assert result.converged is not None
        assert result.audit_iterations is not None
        assert result.audit_iterations >= 1

    def test_observation_skip_audit(self):
        """Observation with audit skipped — faster, still produces prose."""
        query = ObservationQuery(
            focus_entity_ids=["ENT_LADY_MACBETH"],
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        # Audit was skipped
        assert result.converged is None


# =========================================================================
# 2. INTERVENTION (Rung 2)
# =========================================================================

@requires_ollama
class TestInterventionE2E:
    """Intervention query — do-operator forcing state changes."""

    def test_intervention_macbeth_location(self):
        """Force Macbeth to a new location — generate prose for the move."""
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.location_id": "LOC_HEATH",
            },
        )
        cfg = _test_pipeline_config(skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        assert result.query_type == "intervention"

    def test_intervention_trait_shift(self):
        """Shift Macbeth's guilt trait — prose should reflect internal turmoil."""
        query = InterventionQuery(
            interventions={
                "ENT_MACBETH.traits.guilt": 0.99,
            },
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)


# =========================================================================
# 3. COUNTERFACTUAL (Rung 3)
# =========================================================================

@requires_ollama
class TestCounterfactualE2E:
    """Counterfactual query — abduction + historical intervention."""

    def test_counterfactual_duncan_lives(self):
        """What if Duncan was never murdered? Condition on present evidence."""
        query = CounterfactualQuery(
            historical_interventions={
                "EVT_DUNCAN_MURDER.outcome": "Duncan survives the night",
            },
            evidence_node_ids=["EVT_MACBETH_CROWNED"],
        )
        cfg = _test_pipeline_config(skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        assert result.query_type == "counterfactual"


# =========================================================================
# 4. DIRECTIVE (Full affective calculus + generation + audit)
# =========================================================================

@requires_ollama
class TestDirectiveE2E:
    """Directive query — mathematical optimization of narrative effects."""

    def test_directive_suspense(self):
        """Maximize suspense for Macbeth — full brief → generation → audit."""
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="suspense",
            target_vector_id="EVT_MACBETH_KILLED",
            intensity=0.9,
        )
        cfg = _test_pipeline_config()
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        assert result.query_type == "directive"
        # Directive goes through render_and_audit path
        assert result.converged is not None

    def test_directive_dramatic_irony(self):
        """Maximize dramatic irony — Duncan doesn't know what the reader knows."""
        query = DirectiveQuery(
            target_entity_ids=["ENT_DUNCAN"],
            target_effect="dramatic_irony",
            intensity=1.0,
        )
        cfg = _test_pipeline_config(skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)

    def test_directive_grief(self):
        """Maximize grief for Macduff after family slaughter."""
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACDUFF"],
            target_effect="grief",
            intensity=1.0,
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)


# =========================================================================
# 5. INTERROGATION (Graph RAG — no prose)
# =========================================================================

@requires_ollama
class TestInterrogationE2E:
    """Interrogation query — graph pathfinding, no generation."""

    def test_interrogation_spatial_path(self):
        """Ask about spatial reachability in the graph."""
        query = InterrogationQuery(
            question="Is there a physical path for Macbeth to reach Birnam Wood from Dunsinane Castle?",
            require_proof=True,
        )
        cfg = _test_pipeline_config()
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=False)
        assert result.query_type == "interrogate"
        # No prose or audit for interrogation
        assert result.prose is None
        assert result.converged is None


# =========================================================================
# 6. GENERAL (Full-Graph Q&A — no prose)
# =========================================================================

@requires_ollama
class TestGeneralE2E:
    """General query — open-ended graph Q&A."""

    def test_general_relationship_query(self):
        """Ask about relationships across the graph."""
        query = GeneralQuery(
            question="What are all the relationships between Macbeth and Lady Macbeth?",
        )
        cfg = _test_pipeline_config()
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=False)
        assert result.query_type == "general"


# =========================================================================
# 7. FULL PIPELINE WITH VERSIONED MERGE
# =========================================================================

@requires_ollama
class TestVersionedMergeE2E:
    """End-to-end with re-extraction and versioned world model merge."""

    def test_observation_with_merge(self):
        """Observe → generate → audit → re-extract → merge into versioned model."""
        query = ObservationQuery(
            focus_entity_ids=["ENT_MACBETH"],
        )
        cfg = _test_pipeline_config(skip_audit=True)  # skip audit for speed, keep re-extraction
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)

        # World model should be versioned
        vwm = result.world_model
        assert isinstance(vwm, VersionedWorldModel)
        # Should have at least the initial version
        assert vwm.version >= 0
        # The current world state should still validate
        ws = vwm.current
        assert isinstance(ws, WorldStateV1)
        assert len(ws.entities) > 0
        assert len(ws.events) > 0

    def test_directive_versioned_continuation(self):
        """Run two queries on the same versioned model — history accumulates."""
        # First query: observation
        q1 = ObservationQuery(focus_entity_ids=["ENT_BANQUO"])
        cfg = _test_pipeline_config(skip_audit=True)
        r1 = run_pipeline(q1, world_state=macbeth_ws, config=cfg)
        assert r1.world_model is not None
        v1 = r1.world_model.version

        # Second query: directive, continuing from the versioned model
        q2 = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
            intensity=0.8,
        )
        r2 = run_pipeline(q2, versioned_model=r1.world_model, config=cfg)
        _assert_valid_pipeline_result(r2, expect_prose=True)

        # Version should have advanced
        assert r2.world_model.version >= v1


# =========================================================================
# 8. CROSS-PLOT VALIDATION
# =========================================================================

@requires_ollama
class TestCrossPlotE2E:
    """Verify the pipeline works across different plot models."""

    def test_gone_girl_directive(self):
        """Directive query against Gone Girl world state."""
        ent_ids = list(gone_girl_ws.entities.keys())
        assert len(ent_ids) >= 1, "Gone Girl fixture should have entities"
        query = DirectiveQuery(
            target_entity_ids=[ent_ids[0]],
            target_effect="surprise",
            intensity=0.9,
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=gone_girl_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)

    def test_gone_girl_interrogation(self):
        """Interrogation query against Gone Girl world state."""
        query = InterrogationQuery(
            question="Who had access to the diary?",
        )
        cfg = _test_pipeline_config()
        result = run_pipeline(query, world_state=gone_girl_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=False)


# =========================================================================
# 9. INGESTION E2E (raw text → WorldStateV1)
# =========================================================================

@requires_ollama
class TestIngestionE2E:
    """Test raw text ingestion through the pipeline."""

    def test_ingest_short_plot(self):
        """Ingest a minimal plot summary and run an observation query."""
        short_text = (
            "Act I\n\n"
            "Romeo Montague meets Juliet Capulet at a masked ball in Verona. "
            "Despite their families' ancient feud, they fall in love at first sight. "
            "Romeo sneaks into the Capulet garden and they exchange vows under the balcony.\n\n"
            "Act II\n\n"
            "Romeo and Juliet are secretly married by Friar Lawrence. "
            "Later that day, Tybalt kills Mercutio in a street fight. "
            "Romeo, enraged, kills Tybalt and is banished from Verona by the Prince."
        )
        query = ObservationQuery(focus_entity_ids=[])
        cfg = PipelineConfig(
            use_causal_engine=False,
            ingestion_config=ExtractionConfig(model=_MODEL),
            generation_config=GenerationConfig(model=_MODEL, max_tokens=512),
            skip_audit=True,
            skip_reextraction=True,
        )
        result = run_pipeline(query, raw_text=short_text, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)

        # Verify ingestion produced a valid world state
        ws = result.world_model.current
        assert len(ws.entities) >= 2, f"Expected at least Romeo and Juliet, got {list(ws.entities.keys())}"
        assert len(ws.events) >= 2, f"Expected at least 2 events, got {len(ws.events)}"
        assert len(ws.locations) >= 1, f"Expected at least 1 location"


# =========================================================================
# 10. ASYNC PIPELINE E2E
# =========================================================================

@requires_ollama
class TestAsyncPipelineE2E:
    """Test the async pipeline variant."""

    def test_async_ingestion(self):
        """Async ingestion of a short plot — parallel ontology + chunk extraction."""
        import asyncio
        from shadow_loom.ingestion import run_extraction_async

        short_text = (
            "Part I\n\n"
            "Elizabeth Bennet meets Mr Darcy at a ball in Meryton. "
            "She finds him proud and disagreeable. Her sister Jane falls ill "
            "at Netherfield and Elizabeth walks through the mud to nurse her.\n\n"
            "Part II\n\n"
            "Mr Darcy proposes to Elizabeth but she rejects him, citing his "
            "pride and his interference with Jane and Mr Bingley. "
            "Darcy writes a letter explaining himself and Elizabeth's opinion begins to change."
        )
        cfg = ExtractionConfig(model=_MODEL)
        ws, report = asyncio.run(run_extraction_async(short_text, config=cfg))

        assert isinstance(ws, WorldStateV1)
        assert len(ws.entities) >= 2, f"Expected at least 2 entities, got {list(ws.entities.keys())}"
        assert len(ws.events) >= 2, f"Expected at least 2 events, got {len(ws.events)}"
        assert len(ws.locations) >= 1
        # Validation should have run
        assert report is not None
        assert isinstance(report.is_valid, bool)
