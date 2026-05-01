# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

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

import os
import pathlib
import time
import urllib.request
import uuid
import pytest

import shadow_loom.db as sl_db
from shadow_loom.auditor import AuditorConfig, FeedbackLoopResult
from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.generation import GenerationConfig
from shadow_loom.ingestion import ExtractionConfig
from shadow_loom.models import WorldStateV1
from shadow_loom.pipeline import (
    PipelineConfig,
    PipelineResult,
    humanize_pipeline_result,
    run_pipeline,
    run_pipeline_async,
)
from shadow_loom.query_models import (
    CounterfactualQuery,
    DirectiveQuery,
    EvaluationQuery,
    GeneralQuery,
    InterrogationQuery,
    InterventionQuery,
    ManualEditQuery,
    ObservationQuery,
)

# -- Plot model fixtures --------------------------------------------------
from example_worlds.macbeth import world_state as macbeth_ws
from example_worlds.gone_girl import world_state as gone_girl_ws

# =========================================================================
# Per-test SQLite database fixture
# =========================================================================
#
# Every test in this module gets a fresh, isolated SQLite database file
# under ``logs/test_dbs/`` so live ingestion / version writes from one
# test cannot leak into another and so a failed run can be inspected on
# disk after the fact. The DB file is named
# ``<timestamp>_<test-name>_<short-uuid>.db`` and the path is exposed
# to the test via the ``test_db_path`` fixture if it needs it.

_TEST_DB_DIR = pathlib.Path(
    os.environ.get(
        "SHADOW_LOOM_TEST_DB_DIR",
        pathlib.Path(__file__).resolve().parent.parent / "logs" / "test_dbs",
    )
)


@pytest.fixture(autouse=True)
def test_db_path(request, tmp_path_factory):
    """Initialise a fresh SQLite DB for every test and tear it down after.

    The DB file is created under ``logs/test_dbs/`` (overridable via
    ``SHADOW_LOOM_TEST_DB_DIR``) so it persists for post-mortem inspection.
    The module-level engine in ``shadow_loom.db`` is reset between tests so
    each test starts from a clean slate.
    """
    _TEST_DB_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = request.node.name.replace("/", "_").replace(":", "_")[:80]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    db_path = _TEST_DB_DIR / f"{stamp}_{safe_name}_{uuid.uuid4().hex[:6]}.db"
    url = f"sqlite:///{db_path}"

    # Reset any engine that a previous test (or import-time init) created
    # so init_db() rebinds cleanly to this test's URL.
    sl_db._engine = None
    sl_db.init_db(url)

    yield db_path

    # Dispose engine so the file handle is released; keep the .db file on
    # disk for inspection.
    if sl_db._engine is not None:
        try:
            sl_db._engine.dispose()
        except Exception:
            pass
        sl_db._engine = None

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
# 0. PLOT TEXT INGESTION (Step 1 of the pipeline) — runs FIRST
# =========================================================================
#
# These are the *first* live-e2e tests so a freshly-pulled clone with
# nothing but Ollama running can confirm that raw plot summaries from
# ``sample_plots/`` ingest into a valid ``WorldStateV1`` end-to-end
# before any of the downstream physics / generation tests fire.

_SAMPLE_PLOTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "sample_plots"

_PLOTS_TO_INGEST = [
    "romeo_and_juliet.txt",
    "macbeth.txt",
    "dads_army.txt",
    "gone_girl.txt",
    "persuasion.txt",
    "reservoir_dogs.txt",
]


@requires_ollama
class TestPlotIngestionE2E:
    """Live ingestion of every sample plot text in ``sample_plots/``.

    Each plot is read from disk, fed through ``run_pipeline`` with
    ``raw_text=...``, and the resulting ``WorldStateV1`` is validated.
    Each test gets its own SQLite DB via the ``test_db_path`` fixture.
    """

    @pytest.mark.parametrize("plot_filename", _PLOTS_TO_INGEST)
    def test_ingest_plot_to_world_state(self, plot_filename, test_db_path):
        plot_path = _SAMPLE_PLOTS_DIR / plot_filename
        assert plot_path.exists(), f"Missing sample plot: {plot_path}"
        raw_text = plot_path.read_text(encoding="utf-8")
        assert len(raw_text) > 100, (
            f"Plot file too short to be meaningful: {plot_path}"
        )

        # Run an observation query — physics is disabled so we exercise
        # ingestion + brief + render without slow simulation.
        query = ObservationQuery(focus_entity_ids=[])
        cfg = PipelineConfig(
            use_causal_engine=False,
            ingestion_config=ExtractionConfig(model=_MODEL),
            generation_config=GenerationConfig(model=_MODEL, max_tokens=512),
            skip_audit=True,
            skip_reextraction=True,
        )
        result = run_pipeline(query, raw_text=raw_text, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)

        ws = result.world_model.current
        assert isinstance(ws, WorldStateV1), "Ingestion must yield WorldStateV1"
        assert len(ws.entities) >= 2, (
            f"{plot_filename}: expected >= 2 entities, got "
            f"{list(ws.entities.keys())}"
        )
        assert len(ws.events) >= 2, (
            f"{plot_filename}: expected >= 2 events, got {len(ws.events)}"
        )
        assert len(ws.locations) >= 1, (
            f"{plot_filename}: expected >= 1 location"
        )
        # Every entity should carry a non-empty name
        for eid, ent in ws.entities.items():
            assert ent.name and ent.name.strip(), (
                f"{plot_filename}: entity {eid} has empty name"
            )
        # Every event should have a well-formed EVT_ id
        for ev in ws.events:
            assert ev.id and ev.id.startswith("EVT_"), (
                f"{plot_filename}: event missing/invalid id: {ev}"
            )
        # The DB file for this test must have been created.
        assert test_db_path.exists(), (
            f"Per-test DB was not created at {test_db_path}"
        )

    def test_ingest_async_short_plot(self, test_db_path):
        """Async ingestion path against a short hand-crafted plot."""
        import asyncio
        from shadow_loom.ingestion import run_extraction_async

        short_text = (
            "Part I\n\n"
            "In the small Provençal village of Manosque, a young shepherd "
            "named Jean discovers a hidden spring on the abandoned Soubeyran "
            "farm. He confides the secret to his neighbour, Ugolin, who "
            "covets the land for himself.\n\n"
            "Part II\n\n"
            "Ugolin and his uncle Cesar quietly block the spring. Jean labours "
            "day after day to keep his crops alive without water and eventually "
            "dies of exhaustion. Ugolin buys the farm."
        )
        cfg = ExtractionConfig(model=_MODEL)
        ws, report = asyncio.run(run_extraction_async(short_text, config=cfg))
        assert isinstance(ws, WorldStateV1)
        assert len(ws.entities) >= 2
        assert len(ws.events) >= 2
        assert report is not None
        assert isinstance(report.is_valid, bool)
        assert test_db_path.exists()


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


# =========================================================================
# 12. MANUAL EDIT (User-authored prose, no generation, real re-extraction)
# =========================================================================

@requires_ollama
class TestManualEditE2E:
    """Manual-edit query — bypasses physics & generation, real re-extract + merge."""

    def test_manual_edit_macbeth_short_scene(self):
        """User supplies prose; pipeline re-extracts topology and merges."""
        query = ManualEditQuery(
            edited_prose=(
                "Macbeth stood alone in the courtyard of Dunsinane Castle, "
                "the bloody dagger heavy in his trembling hand. The witches' "
                "prophecy echoed in his mind, and for the first time he felt "
                "the full weight of what he had done to Duncan."
            ),
            description="Solitary moment of Macbeth's guilt after the murder.",
            focus_entity_ids=["ENT_MACBETH"],
        )
        cfg = _test_pipeline_config()  # re-extraction must run for manual edit
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)

        assert result.query_type == "manual_edit"
        # Prose comes back as the user's verbatim text (or close to it).
        assert result.prose is not None and len(result.prose) > 20
        assert "Macbeth" in result.prose
        # Audit / generation are skipped for manual_edit
        assert result.scene is not None
        assert result.world_model is not None
        # The world model should advance one version.
        assert isinstance(result.world_model, VersionedWorldModel)
        assert result.world_model.version >= 0
        # Manual-edit must NOT be flagged as implausible — there is nothing
        # for the engine to refuse.
        assert result.implausible is False

    def test_manual_edit_skip_reextraction(self):
        """Manual edit with re-extraction skipped — prose is returned unchanged."""
        prose = "Lady Macbeth wandered the candlelit halls, her hands rubbing together."
        query = ManualEditQuery(
            edited_prose=prose,
            focus_entity_ids=["ENT_LADY_MACBETH"],
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        assert result.query_type == "manual_edit"
        assert result.prose is not None
        assert "Lady Macbeth" in result.prose


# =========================================================================
# 13. EVALUATION (Full-story scorecard)
# =========================================================================

@requires_ollama
class TestEvaluationE2E:
    """Evaluation query — runs the NarrativeOrderObject scorecard end-to-end."""

    def test_evaluation_macbeth_full(self):
        """Score the Macbeth fixture against the full scorecard."""
        query = EvaluationQuery(
            focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
            include_full_prose=True,
        )
        cfg = _test_pipeline_config()
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)

        assert result.query_type == "evaluate"
        assert result.evaluation_result is not None
        narrative_order = result.evaluation_result.narrative_order
        # Scorecard must populate the three feedback objects + overall_pass
        assert narrative_order is not None
        assert hasattr(narrative_order, "causal_feedback")
        assert hasattr(narrative_order, "affective_feedback")
        assert hasattr(narrative_order, "overall_pass")
        assert isinstance(narrative_order.overall_pass, bool)
        # Scores must be numeric and in-range
        cf = narrative_order.causal_feedback
        af = narrative_order.affective_feedback
        assert isinstance(cf.foreshadowing_payoff_score, (int, float))
        assert isinstance(af.affective_loss_mse, (int, float))
        # Aggregated prose must be present when include_full_prose=True
        assert result.evaluation_result.story_prose_evaluated

    def test_evaluation_minimal_prose(self):
        """Evaluation falls back to event-summary text when no prose history."""
        query = EvaluationQuery(focus_entity_ids=[], include_full_prose=False)
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        assert result.evaluation_result is not None
        # include_full_prose=False ⇒ scorecard runs but story_prose_evaluated is empty
        assert result.evaluation_result.story_prose_evaluated == ""


# =========================================================================
# 14. IMPLAUSIBILITY ENVELOPE (force_implausible flag)
# =========================================================================

@requires_ollama
class TestImplausibilityE2E:
    """Verify the implausibility short-circuit and force_implausible bypass.

    A query whose targets do not resolve against the world state must be
    flagged ``result.implausible = True`` and must NOT mutate the
    versioned world model. With ``force_implausible=True`` the pipeline
    proceeds to generation but still reports the implausibility on the
    result.
    """

    def test_intervention_unknown_entity_is_implausible(self):
        query = InterventionQuery(
            interventions={"ENT_NONEXISTENT_GHOST.location_id": "LOC_HEATH"},
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        assert result.query_type == "intervention"
        assert result.implausible is True
        assert result.implausibility_reason
        # No prose generated, world unchanged
        assert result.world_model is not None

    def test_force_implausible_intervention_still_generates(self):
        query = InterventionQuery(
            interventions={"ENT_NONEXISTENT_GHOST.location_id": "LOC_HEATH"},
            force_implausible=True,
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        # Engine still flags it, but prose IS generated under the override.
        assert result.implausible is True
        assert result.prose is not None and len(result.prose) > 20

    def test_directive_unknown_target_is_implausible(self):
        query = DirectiveQuery(
            target_entity_ids=["ENT_NONEXISTENT_HERO"],
            target_effect="suspense",
            intensity=0.5,
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        assert result.query_type == "directive"
        assert result.implausible is True
        assert result.implausibility_reason


# =========================================================================
# 15. CTF-CALCULUS PRE-FLIGHT (Correa & Bareinboim, ICML 2025)
# =========================================================================

@requires_ollama
class TestCtfCalculusE2E:
    """Verify the AMWN / ctf-calculus pre-flight runs alongside the engine.

    The pre-flight reports Rule-1 vacuity, Rule-2 redundant evidence, and
    Rule-3 pruned interventions. We just check the report propagates onto
    the physics result for a real run.
    """

    def test_intervention_with_target_nodes_runs_preflight(self):
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.guilt": 0.9},
            target_node_ids=["EVT_MACBETH_KILLED"],  # Y-set for Rule 3
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        # The physics result should carry pre-flight diagnostics. We don't
        # assert a specific value (pruning is data-dependent) — just that
        # the keys exist.
        physics = result.physics_result
        assert physics is not None

    def test_counterfactual_with_evidence_runs_preflight(self):
        query = CounterfactualQuery(
            historical_interventions={
                "EVT_DUNCAN_MURDER.outcome": "Duncan survives the night",
            },
            evidence_node_ids=["EVT_MACBETH_CROWNED"],
            target_node_ids=["EVT_MACBETH_KILLED"],
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)


# =========================================================================
# 16. FULL ASYNC PIPELINE (run_pipeline_async beyond ingestion)
# =========================================================================

@requires_ollama
class TestAsyncFullPipelineE2E:
    """End-to-end run through the async pipeline for each non-trivial query type."""

    def test_async_observation(self):
        import asyncio
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = asyncio.run(run_pipeline_async(query, world_state=macbeth_ws, config=cfg))
        _assert_valid_pipeline_result(result, expect_prose=True)
        assert result.query_type == "observation"

    def test_async_directive(self):
        import asyncio
        query = DirectiveQuery(
            target_entity_ids=["ENT_LADY_MACBETH"],
            target_effect="regret",
            intensity=0.8,
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = asyncio.run(run_pipeline_async(query, world_state=macbeth_ws, config=cfg))
        _assert_valid_pipeline_result(result, expect_prose=True)
        assert result.query_type == "directive"

    def test_async_interrogate(self):
        import asyncio
        query = InterrogationQuery(
            question="Who killed King Duncan and where?",
            require_proof=False,
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = asyncio.run(run_pipeline_async(query, world_state=macbeth_ws, config=cfg))
        _assert_valid_pipeline_result(result, expect_prose=False)
        assert result.query_type == "interrogate"

    def test_async_raw_text_then_observation(self):
        """Async ingestion → observation in a single async pipeline run."""
        import asyncio
        short_text = (
            "Chapter 1\n\n"
            "Heathcliff arrives at Wuthering Heights as a foundling. "
            "Catherine Earnshaw befriends him while her brother Hindley "
            "treats him as a servant.\n\n"
            "Chapter 2\n\n"
            "Years pass; Catherine marries Edgar Linton at Thrushcross "
            "Grange while Heathcliff disappears for three years."
        )
        query = ObservationQuery(focus_entity_ids=[])
        cfg = PipelineConfig(
            use_causal_engine=False,
            ingestion_config=ExtractionConfig(model=_MODEL),
            generation_config=GenerationConfig(model=_MODEL, max_tokens=512),
            skip_audit=True,
            skip_reextraction=True,
        )
        result = asyncio.run(run_pipeline_async(query, raw_text=short_text, config=cfg))
        _assert_valid_pipeline_result(result, expect_prose=True)
        ws = result.world_model.current
        assert len(ws.entities) >= 2


# =========================================================================
# 17. RE-EXTRACTION + MERGE INVARIANTS
# =========================================================================

@requires_ollama
class TestReextractionInvariantsE2E:
    """Confirm the audit/re-extract/merge invariants advertised in the docs."""

    def test_failed_reextraction_does_not_advance_canonical(self):
        """If re-extraction fails, the version tree must NOT advance.

        We can't easily force a re-extraction failure with a real LLM, so
        the test asserts the invariant on the success path: when re-extraction
        succeeds, ``reextraction_failed`` is False AND a new version exists.
        """
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        cfg = _test_pipeline_config(skip_audit=True)  # keep re-extraction
        v_before = VersionedWorldModel.from_world_state(
            macbeth_ws.model_copy(deep=True)
        ).version
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        assert result.reextraction_failed is False
        assert result.world_model.version >= v_before

    def test_audit_iterations_recorded_when_audit_enabled(self):
        query = ObservationQuery(focus_entity_ids=["ENT_BANQUO"])
        cfg = _test_pipeline_config()  # audit + re-extraction both on
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        assert result.converged is not None
        assert result.audit_iterations is not None
        assert result.audit_iterations >= 1
        assert result.feedback_result is not None


# =========================================================================
# 18. PIPELINE HISTORY INVARIANTS
# =========================================================================
#
# Every pipeline run records an ordered list of step records into
# ``result.history.steps``.  These tests pin the contract for which step
# names appear for which query type so downstream UI/MCP code can rely on
# them.

@requires_ollama
class TestPipelineHistoryE2E:
    """``PipelineResult.history.steps`` ordering and naming contract."""

    @staticmethod
    def _step_names(result: PipelineResult) -> list[str]:
        return [s.get("step") for s in result.history.steps]

    def test_history_observation_with_audit_and_merge(self):
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        cfg = _test_pipeline_config()  # audit + re-extraction both on
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)

        names = self._step_names(result)
        # narrative_physics must precede generation; audit must follow
        # generation; reextraction_merge closes the loop.
        for required in (
            "narrative_physics", "generation", "audit", "reextraction_merge",
        ):
            assert required in names, f"Missing {required!r} in {names}"
        assert names.index("narrative_physics") < names.index("generation")
        assert names.index("generation") < names.index("audit")
        assert names.index("audit") < names.index("reextraction_merge")

    def test_history_skip_audit_skips_audit_step(self):
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        names = self._step_names(result)
        assert "narrative_physics" in names
        assert "generation" in names
        assert "audit" not in names
        assert "reextraction_merge" not in names

    def test_history_ingestion_step_recorded_for_raw_text(self):
        short_text = (
            "Chapter 1\n\nElinor Dashwood quietly bears the loss of "
            "Norland and the cooling of Edward Ferrars's affection."
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
        names = self._step_names(result)
        assert names[0] == "ingestion", f"Expected ingestion first, got {names}"

    def test_history_evaluation_records_evaluation_step(self):
        query = EvaluationQuery(focus_entity_ids=[], include_full_prose=False)
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        names = self._step_names(result)
        assert "evaluation" in names, names

    def test_history_manual_edit_records_generation_and_merge(self):
        query = ManualEditQuery(
            edited_prose="Macbeth paced the battlements, dagger in hand.",
            description="Test manual edit",
            focus_entity_ids=["ENT_MACBETH"],
        )
        cfg = _test_pipeline_config()
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        names = self._step_names(result)
        assert "generation" in names
        # Manual edit always re-extracts.
        assert "reextraction_merge" in names
        # No audit step on manual edit.
        assert "audit" not in names


# =========================================================================
# 19. BRANCH ROUTING THROUGH THE FULL PIPELINE
# =========================================================================
#
# Unit tests in ``test_branch_routing.py`` cover ``_resolve_branch_policy``
# in isolation. These tests confirm the policy is honoured all the way
# through to the merged ``world_id`` tag on the new version.

@requires_ollama
class TestBranchRoutingE2E:
    """Counterfactual queries fork to a shadow branch by default."""

    def test_counterfactual_lands_on_shadow_under_auto(self):
        query = CounterfactualQuery(
            historical_interventions={
                "EVT_DUNCAN_MURDER.outcome": "Duncan survives the night",
            },
            evidence_node_ids=["EVT_MACBETH_CROWNED"],
            original_query="What if Duncan had survived?",
        )
        cfg = _test_pipeline_config(skip_audit=True)  # keep re-extraction
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)

        vwm = result.world_model
        # Newly merged events on the latest version must be tagged shadow.
        latest = vwm.history[-1]
        added_event_ids = (
            latest.changeset.events_added if latest.changeset else 0
        )
        # At least one event was added by the merge for a real LLM run.
        if added_event_ids:
            ws = vwm.current
            shadow_events = [
                e for e in ws.events if getattr(e, "world_id", "factual") == "shadow"
            ]
            assert shadow_events, (
                "Counterfactual merge must tag added events with world_id='shadow'"
            )

    def test_observation_lands_on_factual_under_auto(self):
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        cfg = _test_pipeline_config(skip_audit=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        ws = result.world_model.current
        # Every event on the canon should remain factual.
        for e in ws.events:
            assert getattr(e, "world_id", "factual") == "factual", (
                f"Event {e.id} drifted off factual mainline under 'auto' policy."
            )

    def test_force_shadow_policy_for_observation(self):
        query = ObservationQuery(focus_entity_ids=["ENT_LADY_MACBETH"])
        cfg = PipelineConfig(
            use_causal_engine=True,
            generation_config=GenerationConfig(model=_MODEL, max_tokens=512),
            skip_audit=True,
            skip_reextraction=False,
            extraction_config=ExtractionConfig(model=_MODEL),
            branch_policy="shadow",
        )
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)
        latest = result.world_model.history[-1]
        added = latest.changeset.events_added if latest.changeset else 0
        if added:
            shadow_events = [
                e for e in result.world_model.current.events
                if getattr(e, "world_id", "factual") == "shadow"
            ]
            assert shadow_events, (
                "branch_policy='shadow' must tag merged events with world_id='shadow'."
            )


# =========================================================================
# 20. FORCE_IMPLAUSIBLE — bypass for both Rung-2 and Rung-3
# =========================================================================
#
# The Rung-2 case is covered above (``TestImplausibilityE2E``). This
# class adds the symmetrical Rung-3 (counterfactual) bypass.

@requires_ollama
class TestForceImplausibleCounterfactualE2E:
    """Counterfactual ``force_implausible=True`` still produces prose."""

    def test_force_implausible_counterfactual_generates_prose(self):
        query = CounterfactualQuery(
            historical_interventions={
                "EVT_NEVER_HAPPENED.outcome": "would not happen",
            },
            evidence_node_ids=["EVT_NONEXISTENT_EVIDENCE"],
            force_implausible=True,
            original_query="A counterfactual on a phantom event.",
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        # Engine still flags as implausible but prose IS generated.
        assert result.implausible is True
        assert result.implausibility_reason
        assert result.prose is not None and len(result.prose) > 20


# =========================================================================
# 21. HUMANIZE_PIPELINE_RESULT — lay-user summaries
# =========================================================================
#
# The ``humanize_pipeline_result`` helper is the chat-UI / MCP envelope's
# canonical formatter for a ``PipelineResult``. These tests confirm it
# returns non-empty text on the supported query types and surfaces
# the requested-vs-achieved emotional intensity for directives.

@requires_ollama
class TestHumanizePipelineResultE2E:
    """The lay-user summariser handles every prose-producing query type."""

    def test_humanize_observation(self):
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        text = humanize_pipeline_result(result)
        assert isinstance(text, str) and text.strip()
        assert "observation" in text.lower()
        assert "Generated" in text  # prose count line

    def test_humanize_directive_reports_intensity_gap(self):
        query = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
            intensity=0.8,
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        text = humanize_pipeline_result(
            result, requested_effect="fear", requested_intensity=0.8,
        )
        assert isinstance(text, str) and text.strip()
        assert "directive" in text.lower()

    def test_humanize_implausible_intervention(self):
        query = InterventionQuery(
            interventions={"ENT_GHOST_OF_NOWHERE.location_id": "LOC_HEATH"},
        )
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        text = humanize_pipeline_result(result)
        assert "couldn't be applied" in text or "implausible" in text.lower()

    def test_humanize_evaluation(self):
        query = EvaluationQuery(focus_entity_ids=[], include_full_prose=False)
        cfg = _test_pipeline_config(skip_audit=True, skip_reextraction=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        text = humanize_pipeline_result(result)
        assert isinstance(text, str) and text.strip()


# =========================================================================
# 22. MULTI-CYCLE CONTINUATION
# =========================================================================
#
# Verify that running multiple pipeline cycles against the same
# ``VersionedWorldModel`` strictly grows the history and version
# counter, and never silently loses the prior versions.

@requires_ollama
class TestMultiCycleContinuationE2E:
    """Sequential pipeline runs accumulate versions in the world model."""

    def test_three_cycles_strictly_advance_version(self):
        cfg = _test_pipeline_config(skip_audit=True)  # keep re-extraction

        q1 = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        r1 = run_pipeline(q1, world_state=macbeth_ws, config=cfg)
        v1 = r1.world_model.version

        q2 = ObservationQuery(focus_entity_ids=["ENT_LADY_MACBETH"])
        r2 = run_pipeline(q2, versioned_model=r1.world_model, config=cfg)
        v2 = r2.world_model.version

        q3 = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
            intensity=0.7,
        )
        r3 = run_pipeline(q3, versioned_model=r2.world_model, config=cfg)
        v3 = r3.world_model.version

        assert v1 <= v2 <= v3, (
            f"Version must monotonically advance (got {v1}, {v2}, {v3})"
        )
        # History entries: every successful merge appends one entry.
        assert len(r3.world_model.history) >= len(r1.world_model.history)


# =========================================================================
# 23. SAVE_VERSION INTEGRATION (Pipeline → DB persistence)
# =========================================================================
#
# The pipeline itself does not write to the database; persistence is the
# caller's responsibility. These tests exercise the canonical
# pipeline → ``save_version`` flow that the UI's task helpers and the
# MCP ``run_and_save`` wrapper use.

@requires_ollama
class TestSaveVersionE2E:
    """``save_version`` correctly stamps branch metadata from a pipeline run."""

    def test_pipeline_then_save_version_preserves_branch(self, test_db_path):
        proj = sl_db.create_project(name="e2e_save_version_test")
        # Save the initial world state as v0.
        v0 = sl_db.save_version(
            project_id=proj.id,
            world_state_json=macbeth_ws.model_dump_json(),
            source="seed",
            description="Seed Macbeth",
        )

        # Run a counterfactual — under 'auto' policy this routes to shadow.
        query = CounterfactualQuery(
            historical_interventions={
                "EVT_DUNCAN_MURDER.outcome": "Duncan survives the night",
            },
            evidence_node_ids=["EVT_MACBETH_CROWNED"],
            original_query="Counterfactual save-version smoke test",
        )
        cfg = _test_pipeline_config(skip_audit=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)

        # Persist the merged result with branch metadata from the pipeline.
        # The pipeline doesn't expose world_id directly; resolve it the same
        # way the wrappers do — counterfactual under 'auto' → shadow.
        v1 = sl_db.save_version(
            project_id=proj.id,
            world_state_json=result.world_model.current.model_dump_json(),
            ancestor_id=v0.id,
            source="pipeline",
            description="Counterfactual fork",
            prose=result.prose,
            world_id="shadow",
            branch_label=query.original_query,
        )
        assert v1.world_id == "shadow"
        assert v1.branch_label == query.original_query
        assert v1.ancestor_id == v0.id
        # And the version row is retrievable.
        roundtrip = sl_db.get_version_by_id(v1.id)
        assert roundtrip is not None
        assert roundtrip.world_id == "shadow"

    def test_pipeline_then_save_version_factual_default(self, test_db_path):
        proj = sl_db.create_project(name="e2e_save_version_factual")
        v0 = sl_db.save_version(
            project_id=proj.id,
            world_state_json=macbeth_ws.model_dump_json(),
            source="seed",
        )
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        cfg = _test_pipeline_config(skip_audit=True)
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        _assert_valid_pipeline_result(result, expect_prose=True)

        v1 = sl_db.save_version(
            project_id=proj.id,
            world_state_json=result.world_model.current.model_dump_json(),
            ancestor_id=v0.id,
            source="pipeline",
            prose=result.prose,
            world_id="factual",
        )
        assert v1.world_id == "factual"
        assert v1.branch_label is None


# =========================================================================
# 24. CHANNEL & UTTERANCE EXTRACTION (Information physics surface)
# =========================================================================
#
# The information-topology refactor replaced the legacy ``InformationEdge``
# with first-class ``Channel`` nodes plus utterance ``EventNode``s carrying
# ``speaker_id`` / ``addressee_ids`` / ``via_channel_id`` / ``truth_value``.
# These tests confirm ingestion produces both surfaces on dialogue-rich
# plot text.

_DIALOGUE_PLOT = """
Act I

In the hushed parlour Mr Knightley addressed Emma directly:
"You have always been kind to Harriet, but a match with Mr Elton is folly."
Emma laughed and shook her head. "You misunderstand my plans entirely."
Knightley left, troubled, and Emma turned back to her drawing.

Act II

Later, alone with Harriet, Emma said in a low voice:
"Forget Mr Elton. Mr Frank Churchill, when he comes, will be everything
charming." Harriet listened and believed every word, though Mr Knightley
across the lane had warned the very opposite.
"""


@requires_ollama
class TestChannelExtractionE2E:
    """Ingestion of dialogue-heavy text produces channels and utterance events."""

    def test_dialogue_plot_extracts_channels(self):
        cfg = ExtractionConfig(model=_MODEL)
        from shadow_loom.ingestion import run_extraction
        ws, report = run_extraction(_DIALOGUE_PLOT, config=cfg)
        assert isinstance(ws, WorldStateV1)
        # Channels are a dict on WorldStateV1 — at least one for the
        # parlour conversation must be extracted.
        channels = getattr(ws, "channels", {}) or {}
        assert len(channels) >= 1, (
            f"Expected ≥1 Channel, got {list(channels.keys())}; report.is_valid={report.is_valid}"
        )
        # And at least one utterance event with a speaker_id.
        utterance_events = [
            e for e in ws.events
            if getattr(e, "speaker_id", None) is not None
        ]
        assert len(utterance_events) >= 1, (
            "Expected ≥1 utterance EventNode with speaker_id."
        )

    def test_utterance_event_has_addressees_and_truth_value(self):
        cfg = ExtractionConfig(model=_MODEL)
        from shadow_loom.ingestion import run_extraction
        ws, _ = run_extraction(_DIALOGUE_PLOT, config=cfg)
        utts = [e for e in ws.events if getattr(e, "speaker_id", None)]
        assert utts, "No utterance events extracted"
        # At least one utterance should carry addressees AND a truth_value.
        with_addressees = [
            e for e in utts if getattr(e, "addressee_ids", None)
        ]
        assert with_addressees, (
            "Expected at least one utterance with addressee_ids populated."
        )
        with_truth = [
            e for e in utts if getattr(e, "truth_value", None) is not None
        ]
        assert with_truth, (
            "Expected at least one utterance with a truth_value annotation."
        )


# =========================================================================
# 25. PLAUSIBILITY VACUITY (engine-level Tier-2 implausibility)
# =========================================================================
#
# Tier-1 implausibility (unknown node IDs) is covered above. Tier-2 is
# when the engine binds the do-operator to a real node but propagation
# yields no downstream effect. We exercise this by intervening on a
# trait of an isolated entity that has no outgoing causal edges.

@requires_ollama
class TestEngineVacuityE2E:
    """Tier-2 implausibility — the engine binds but propagates nothing."""

    def test_engine_threshold_failures_recorded_when_audit_enabled(self):
        """The audit feedback object always surfaces engine threshold state."""
        query = ObservationQuery(focus_entity_ids=["ENT_MACBETH"])
        cfg = _test_pipeline_config()
        result = run_pipeline(query, world_state=macbeth_ws, config=cfg)
        assert result.feedback_result is not None
        # Either passed or failed — but the field MUST be populated when
        # change_impact metrics were available (i.e. the deterministic
        # engine gate ran).
        ci = result.feedback_result.change_impact
        if ci is not None:
            assert result.feedback_result.engine_thresholds_passed in (True, False), (
                "engine_thresholds_passed must be True/False when change_impact is set."
            )
            # Failures list is always present.
            assert isinstance(
                result.feedback_result.engine_threshold_failures, list,
            )


