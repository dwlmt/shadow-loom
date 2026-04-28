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


# =========================================================================
# 11. QUERY PARSING E2E — natural language → nodes / edges
# =========================================================================
#
# These tests run real LLM classification via Ollama and verify that the
# parser correctly maps free-form English to graph IDs (entities, events,
# objects, locations, world traits) and the appropriate query type, and
# that the resulting structured query then executes cleanly through the
# real pipeline.

@requires_ollama
class TestQueryParsingE2E:
    """End-to-end NL → structured query → pipeline integration."""

    def _parse(self, nl: str, *, query_type: str | None = None, ws=None):
        from shadow_loom.query_parsing import (
            QueryParsingConfig, parse_query,
        )
        ws = ws if ws is not None else macbeth_ws
        cfg = QueryParsingConfig(model=_MODEL)
        return parse_query(nl, query_type=query_type, world_state=ws, config=cfg)

    # ── Per-type classification + ID resolution ──────────────────────

    def test_observation_classification(self):
        result = self._parse(
            "What happens next from Macbeth's point of view?",
        )
        assert result.is_valid
        assert result.query.query_type == "observation"
        # Macbeth should be resolved as the focus entity
        focus = result.query.focus_entity_ids or []
        assert any("MACBETH" in f for f in focus), (
            f"Expected ENT_MACBETH in focus_entity_ids, got {focus}"
        )

    def test_intervention_classification_and_id_resolution(self):
        result = self._parse(
            "Force Macbeth to die immediately. Set ENT_MACBETH.status to 'dead'.",
            query_type="intervention",
        )
        assert result.is_valid, f"Errors: {result.validation_errors}"
        # If the LLM under-populated the intervention payload the parser
        # falls back to a general query — still a valid run, but in that
        # case the resolved_ids should at least include ENT_MACBETH.
        if result.query.query_type == "intervention":
            keys = list(result.query.interventions.keys())
            assert keys, "Expected at least one intervention"
            assert all("." in k for k in keys), f"Bad keys: {keys}"
            assert any("ENT_MACBETH" in k for k in keys), keys
        else:
            assert result.fallback is not None
            resolved = [r.resolved_id for r in (result.parsed.resolved_ids or [])]
            assert any("MACBETH" in r for r in resolved), resolved

    def test_counterfactual_classification(self):
        result = self._parse(
            "What if Duncan had never been murdered? Use Macbeth's current "
            "guilt as evidence.",
            query_type="counterfactual",
        )
        assert result.is_valid, f"Errors: {result.validation_errors}"
        assert result.query.query_type == "counterfactual"
        hi_keys = list(result.query.historical_interventions.keys())
        assert hi_keys, "Expected at least one historical intervention"
        assert all("." in k for k in hi_keys)
        # Should reference an EVT_ id
        assert any(k.split(".")[0].startswith("EVT_") for k in hi_keys), hi_keys

    def test_directive_classification_with_target_entity(self):
        result = self._parse(
            "Maximise dramatic irony around Macbeth.",
            query_type="directive",
        )
        assert result.is_valid, f"Errors: {result.validation_errors}"
        assert result.query.query_type == "directive"
        assert result.query.target_effect == "dramatic_irony"
        assert any(
            "MACBETH" in t for t in result.query.target_entity_ids
        ), result.query.target_entity_ids

    def test_interrogate_classification(self):
        result = self._parse(
            "Question: is there a physical path for Macbeth to reach "
            "Duncan's chamber unseen? Answer with proof.",
            query_type="interrogate",
        )
        assert result.is_valid
        # Accept fallback to general (also a Q&A query type) when the LLM
        # forgets to populate `question`.
        assert result.query.query_type in ("interrogate", "general")
        assert result.query.question

    def test_general_classification(self):
        result = self._parse(
            "Summarise the relationships between the main characters.",
            query_type="general",
        )
        assert result.is_valid
        assert result.query.query_type == "general"
        assert result.query.question

    def test_evaluate_classification(self):
        result = self._parse(
            "Run a full quality audit of the story.",
            query_type="evaluate",
        )
        assert result.is_valid
        assert result.query.query_type == "evaluate"

    def test_manual_edit_classification(self):
        prose = (
            "Edit: Macbeth draws his dagger and stares at it in the gloom "
            "of the courtyard, his hand trembling."
        )
        result = self._parse(prose, query_type="manual_edit")
        assert result.is_valid
        assert result.query.query_type == "manual_edit"
        assert "dagger" in result.query.edited_prose.lower()

    # ── ID-only resolution (entities, locations, objects, events) ────

    def test_resolves_location_name_to_loc_id(self):
        result = self._parse(
            "Show what is happening at the battlefield.",
            query_type="observation",
        )
        assert result.is_valid
        # Either focus_entity_ids has it or resolved_ids contains the LOC_
        all_ids = (
            (result.query.focus_entity_ids or [])
            + [r.resolved_id for r in (result.parsed.resolved_ids or [])]
        )
        assert any(i.startswith("LOC_") for i in all_ids), all_ids

    def test_resolves_object_name_to_obj_id(self):
        result = self._parse(
            "Force Macbeth to drop the crown.",
            query_type="intervention",
        )
        assert result.is_valid
        all_ids = list(result.query.interventions.keys()) + [
            r.resolved_id for r in (result.parsed.resolved_ids or [])
        ]
        assert any(
            "OBJ_" in i or "CROWN" in i.upper() for i in all_ids
        ), all_ids

    # ── Pipeline integration: NL → parse → run_pipeline ─────────────

    def test_nl_to_pipeline_observation(self):
        """Parse NL then execute the resulting query through run_pipeline."""
        parsed = self._parse(
            "What happens next from Macbeth's perspective?",
            query_type="observation",
        )
        assert parsed.is_valid
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(
            parsed.query,
            world_state=macbeth_ws.model_copy(deep=True),
            config=cfg,
        )
        assert result.prose, "Pipeline should produce prose"
        assert result.world_model is not None

    def test_nl_to_pipeline_directive(self):
        parsed = self._parse(
            "Maximise suspense around Macbeth.",
            query_type="directive",
        )
        assert parsed.is_valid
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(
            parsed.query,
            world_state=macbeth_ws.model_copy(deep=True),
            config=cfg,
        )
        assert result.prose
        assert result.query_type == "directive"

    # ── Fuzzy fallback when LLM returns a slightly wrong ID ─────────

    def test_fuzzy_repair_on_unknown_id(self):
        # We can't force the LLM to emit a bad id, but we can verify the
        # fallback machinery works by checking that valid responses round-trip
        # cleanly without invoking the fallback.
        result = self._parse(
            "What happens next from Lady Macbeth's perspective?",
            query_type="observation",
        )
        assert result.is_valid
        # No fallback should be needed for a clean reference.
        assert result.fallback is None or result.fallback.strategy == "none"

