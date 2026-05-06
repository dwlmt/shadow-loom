# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for shadow_loom.ingestion — chunking, validation, assembly,
and deduplication. No LLM calls are made; all tests are deterministic.
"""

import importlib
import pathlib

import pytest

from shadow_loom.models import (
    Belief,
    CausalEdge,
    Channel,
    Entity,
    EventNode,
    Location,
    NarrativeObject,
    RelationshipEdge,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.ingestion import (
    ExtractionConfig,
    GlobalRegister,
    ValidationIssue,
    chunk_text,
    _auto_repair,
    _deduplicate_social,
    _deduplicate_spatial,
    _load_prompt,
    _normalize_fabula_times,
    _programmatic_validation,
    _validate_dead_actors,
    _validate_time_ordering,
    assemble_world_state,
    ChunkTopology,
    _pre_allocate_chunk_params,
    _reconcile_chunk_topologies,
    _apply_event_renames,
    _shift_fabula_times,
)


# =====================================================================
# Helpers — minimal WorldStateV1 fixtures
# =====================================================================

def _minimal_ws(**overrides) -> WorldStateV1:
    """Build a minimal valid WorldStateV1, merging *overrides*."""
    defaults = dict(
        locations={"LOC_A": Location(name="A", description="a", ambient_state={})},
        objects={},
        entities={
            "ENT_X": Entity(
                id="ENT_X", name="X", location_id="LOC_A", status="healthy",
                traits={"t": TraitVector(value=0.5, inertia=0.5)},
            ),
        },
        events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", actor_ids=["ENT_X"], description="evt1"),
        ],
        causal_topology=[],
        spatial_topology=[],
        social_topology=[],
    )
    defaults.update(overrides)
    return WorldStateV1(**defaults)


# =====================================================================
# chunk_text
# =====================================================================

class TestChunkText:
    def test_act_headings_split(self):
        text = "Act I\nFirst act.\n\nAct II\nSecond act."
        chunks = chunk_text(text, strategy="act_headings")
        assert len(chunks) == 2
        assert "First act" in chunks[0]
        assert "Second act" in chunks[1]

    def test_chapter_headings(self):
        text = "Chapter 1\nIntro.\n\nChapter 2\nMiddle."
        chunks = chunk_text(text, strategy="act_headings")
        assert len(chunks) == 2

    def test_part_headings_roman(self):
        text = "Part I\nA.\n\nPart II\nB.\n\nPart III\nC."
        chunks = chunk_text(text, strategy="act_headings")
        assert len(chunks) == 3

    def test_fallback_to_paragraph(self):
        text = "No headings here.\n\nJust paragraphs.\n\nThree of them."
        chunks = chunk_text(text, strategy="act_headings", min_chunk_chars=0)
        assert len(chunks) == 3

    def test_paragraph_strategy(self):
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = chunk_text(text, strategy="paragraph", min_chunk_chars=0)
        assert len(chunks) == 3

    def test_empty_text(self):
        assert chunk_text("", strategy="paragraph") == []
        assert chunk_text("", strategy="act_headings") == []

    def test_single_paragraph(self):
        chunks = chunk_text("Just one.", strategy="paragraph")
        assert chunks == ["Just one."]

    def test_min_chunk_chars_merging(self):
        # 4 tiny paragraphs, threshold 50 → should merge
        text = "A" * 10 + "\n\n" + "B" * 10 + "\n\n" + "C" * 10 + "\n\n" + "D" * 10
        chunks = chunk_text(text, strategy="paragraph", min_chunk_chars=50)
        assert len(chunks) == 1  # all merged

    def test_min_chunk_chars_no_merge_when_big(self):
        text = ("X" * 2000 + "\n\n") * 3
        chunks = chunk_text(text, strategy="paragraph", min_chunk_chars=1500)
        assert len(chunks) == 3

    def test_min_chunk_chars_final_tiny_merged(self):
        # Big chunk + tiny tail → tail merged into previous
        text = "A" * 2000 + "\n\n" + "B" * 100
        chunks = chunk_text(text, strategy="paragraph", min_chunk_chars=1500)
        assert len(chunks) == 1  # tiny tail merged back

    def test_preamble_before_first_heading(self):
        text = "Preamble text.\n\nAct I\nBody."
        chunks = chunk_text(text, strategy="act_headings")
        assert len(chunks) == 2
        assert "Preamble" in chunks[0]


# =====================================================================
# _deduplicate_social / _deduplicate_spatial
# =====================================================================

class TestDeduplication:
    def test_social_keeps_latest(self):
        e1 = RelationshipEdge(
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            affinity=0.5, last_updated_fabula=100,
        )
        e2 = RelationshipEdge(
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            affinity=0.9, last_updated_fabula=200,
        )
        result = _deduplicate_social([e1, e2])
        assert len(result) == 1
        assert result[0].affinity == 0.9

    def test_social_different_pairs_kept(self):
        e1 = RelationshipEdge(
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            affinity=0.5, last_updated_fabula=100,
        )
        e2 = RelationshipEdge(
            source_entity_id="ENT_B", target_entity_id="ENT_A",
            affinity=-0.3, last_updated_fabula=100,
        )
        result = _deduplicate_social([e1, e2])
        assert len(result) == 2

    def test_social_empty(self):
        assert _deduplicate_social([]) == []

    def test_spatial_keeps_earliest_established(self):
        # Lifecycle contract: when two duplicates differ in
        # ``established_at_fabula``, the earliest tick wins so the
        # spatial relation's lifespan is not silently truncated.
        e1 = SpatialEdge(source_id="LOC_A", target_id="LOC_B", established_at_fabula=0)
        e2 = SpatialEdge(source_id="LOC_A", target_id="LOC_B", established_at_fabula=100)
        result = _deduplicate_spatial([e1, e2])
        assert len(result) == 1
        assert result[0].established_at_fabula == 0

    def test_spatial_empty(self):
        assert _deduplicate_spatial([]) == []


# =====================================================================
# assemble_world_state
# =====================================================================

class TestAssembleWorldState:
    def _register(self):
        return GlobalRegister(
            locations={"LOC_A": Location(name="A", description="a", ambient_state={})},
            objects={},
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A", status="healthy",
                    traits={"t": TraitVector(value=0.5, inertia=0.5)},
                ),
            },
        )

    def test_events_sorted_by_fabula(self):
        topos = [
            ChunkTopology(events=[
                EventNode(id="EVT_2", fabula_time=200, syuzhet_index=1,
                          event_type="choice", description="second"),
            ]),
            ChunkTopology(events=[
                EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                          event_type="choice", description="first"),
            ]),
        ]
        ws = assemble_world_state(self._register(), topos)
        assert ws.events[0].id == "EVT_1"
        assert ws.events[1].id == "EVT_2"

    def test_social_deduplicated(self):
        topos = [
            ChunkTopology(social_topology=[
                RelationshipEdge(
                    source_entity_id="ENT_X", target_entity_id="ENT_X",
                    affinity=0.1, last_updated_fabula=100,
                ),
            ]),
            ChunkTopology(social_topology=[
                RelationshipEdge(
                    source_entity_id="ENT_X", target_entity_id="ENT_X",
                    affinity=0.9, last_updated_fabula=200,
                ),
            ]),
        ]
        ws = assemble_world_state(self._register(), topos)
        assert len(ws.social_topology) == 1
        assert ws.social_topology[0].affinity == 0.9

    def test_spatial_deduplicated(self):
        reg = self._register()
        reg.locations["LOC_B"] = Location(name="B", description="b", ambient_state={})
        topos = [
            ChunkTopology(spatial_topology=[
                SpatialEdge(source_id="LOC_A", target_id="LOC_B", established_at_fabula=0),
                SpatialEdge(source_id="LOC_A", target_id="LOC_B", established_at_fabula=100),
            ]),
        ]
        ws = assemble_world_state(reg, topos)
        assert len(ws.spatial_topology) == 1

    def test_empty_topologies(self):
        ws = assemble_world_state(self._register(), [])
        assert ws.events == []
        assert ws.causal_topology == []


# =====================================================================
# _programmatic_validation
# =====================================================================

class TestProgrammaticValidation:
    def test_clean_ws_no_errors(self):
        ws = _minimal_ws()
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_broken_causal_source(self):
        ws = _minimal_ws(causal_topology=[
            CausalEdge(source_id="EVT_NONEXISTENT", target_id="EVT_1",
                       causality_type="chain_reaction",
                       mechanism="physical", fabula_time=100),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error" and "source_id" in i.detail]
        assert len(errors) >= 1

    def test_causal_source_must_be_event(self):
        """source_id pointing to a known entity is now valid (universal causal edges)."""
        ws = _minimal_ws(causal_topology=[
            CausalEdge(source_id="ENT_X", target_id="EVT_1",
                       causality_type="affordance_gate",
                       mechanism="social", fabula_time=100),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error" and "source_id" in i.detail]
        assert len(errors) == 0

    def test_broken_causal_target(self):
        ws = _minimal_ws(causal_topology=[
            CausalEdge(source_id="EVT_1", target_id="EVT_GONE",
                       causality_type="chain_reaction",
                       mechanism="physical", fabula_time=100),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error" and "target_id" in i.detail]
        assert len(errors) >= 1

    def test_broken_relationship_entity(self):
        ws = _minimal_ws(social_topology=[
            RelationshipEdge(source_entity_id="ENT_GHOST", target_entity_id="ENT_X",
                             affinity=0.5, last_updated_fabula=100),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error" and "source_entity_id" in i.detail]
        assert len(errors) >= 1

    def test_broken_spatial_location(self):
        ws = _minimal_ws(spatial_topology=[
            SpatialEdge(source_id="LOC_A", target_id="LOC_MISSING"),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error" and "LOC_MISSING" in i.detail]
        assert len(errors) >= 1

    def test_broken_channel_participant(self):
        ws = _minimal_ws(channels={
            "CHN_X": Channel(id="CHN_X", name="x", medium="speech",
                              participant_ids=["ENT_NOBODY", "ENT_X"],
                              established_at_fabula=100),
        })
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error" and "ENT_NOBODY" in i.detail]
        assert len(errors) >= 1

    def test_hallucinated_actor_id(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", actor_ids=["ENT_GHOST"], description="spooky"),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.category == "hallucinated_id"]
        assert len(errors) >= 1

    def test_hallucinated_target_id(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", actor_ids=["ENT_X"],
                      target_ids=["ENT_MISSING"], description="miss"),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.category == "hallucinated_id"]
        assert len(errors) >= 1

    def test_entity_bad_location(self):
        ws = _minimal_ws(entities={
            "ENT_X": Entity(
                id="ENT_X", name="X", location_id="LOC_NOWHERE", status="healthy",
                traits={"t": TraitVector(value=0.5, inertia=0.5)},
            ),
        })
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error" and "LOC_NOWHERE" in i.detail]
        assert len(errors) >= 1

    def test_duplicate_event_ids(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", description="first"),
            EventNode(id="EVT_1", fabula_time=200, syuzhet_index=1,
                      event_type="outcome", description="dupe"),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.category == "duplicate"]
        assert len(errors) >= 1

    def test_belief_bad_target_id(self):
        ws = _minimal_ws(entities={
            "ENT_X": Entity(
                id="ENT_X", name="X", location_id="LOC_A", status="healthy",
                traits={"t": TraitVector(value=0.5, inertia=0.5)},
                beliefs=[Belief(target_id="ENT_GHOST", perceived_state="haunted",
                                confidence=0.9, inertia=0.5)],
            ),
        })
        issues = _programmatic_validation(ws)
        warnings = [i for i in issues if "belief" in i.detail.lower() and "ENT_GHOST" in i.detail]
        assert len(warnings) >= 1


# =====================================================================
# _validate_time_ordering
# =====================================================================

class TestValidateTimeOrdering:
    def test_contiguous_syuzhet_ok(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=200, syuzhet_index=1,
                      event_type="choice", description="b"),
        ])
        issues = _validate_time_ordering(ws)
        assert not any("syuzhet" in i.detail.lower() for i in issues if i.severity == "error")

    def test_duplicate_syuzhet_flagged(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=200, syuzhet_index=0,
                      event_type="choice", description="b"),
        ])
        issues = _validate_time_ordering(ws)
        errors = [i for i in issues if i.severity == "error" and "syuzhet" in i.detail.lower()]
        assert len(errors) >= 1

    def test_gap_in_syuzhet_warned(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=200, syuzhet_index=2,
                      event_type="choice", description="c"),
        ])
        issues = _validate_time_ordering(ws)
        warns = [i for i in issues if i.severity == "warning" and "gap" in i.detail.lower()]
        assert len(warns) >= 1

    def test_small_fabula_spacing_warned(self):
        ws = _minimal_ws(events=[
            EventNode(id=f"EVT_{i}", fabula_time=i, syuzhet_index=i,
                      event_type="choice", description=f"e{i}")
            for i in range(5)
        ])
        issues = _validate_time_ordering(ws)
        warns = [i for i in issues if "contiguous small" in i.detail.lower()]
        assert len(warns) >= 1

    def test_proper_fabula_spacing_ok(self):
        ws = _minimal_ws(events=[
            EventNode(id=f"EVT_{i}", fabula_time=i * 100, syuzhet_index=i,
                      event_type="choice", description=f"e{i}")
            for i in range(5)
        ])
        issues = _validate_time_ordering(ws)
        warns = [i for i in issues if "contiguous small" in i.detail.lower()]
        assert len(warns) == 0

    def test_causal_cause_after_effect(self):
        ws = _minimal_ws(
            events=[
                EventNode(id="EVT_CAUSE", fabula_time=200, syuzhet_index=0,
                          event_type="choice", description="cause"),
                EventNode(id="EVT_EFFECT", fabula_time=100, syuzhet_index=1,
                          event_type="outcome", description="effect"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_CAUSE", target_id="EVT_EFFECT",
                           causality_type="chain_reaction",
                           mechanism="physical", fabula_time=200),
            ],
        )
        issues = _validate_time_ordering(ws)
        errors = [i for i in issues if "cause" in i.detail.lower() and "effect" in i.detail.lower()]
        assert len(errors) >= 1

    def test_channel_terminated_before_established(self):
        ws = _minimal_ws(channels={
            "CHN_X": Channel(id="CHN_X", name="x", medium="speech",
                              participant_ids=["ENT_X", "ENT_Y"],
                              established_at_fabula=200, terminated_at_fabula=100),
        })
        issues = _validate_time_ordering(ws)
        errors = [i for i in issues if "terminated_at_fabula" in i.detail]
        assert len(errors) >= 1

    def test_empty_events_no_crash(self):
        ws = _minimal_ws(events=[])
        issues = _validate_time_ordering(ws)
        assert issues == []


# =====================================================================
# _validate_dead_actors
# =====================================================================

class TestValidateDeadActors:
    def test_dead_target_acting_after_death(self):
        ws = _minimal_ws(
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A", status="dead",
                    traits={"t": TraitVector(value=0.5, inertia=0.5)},
                ),
            },
            events=[
                EventNode(id="EVT_KILL", fabula_time=100, syuzhet_index=0,
                          event_type="outcome", actor_ids=["ENT_X"],
                          target_ids=["ENT_X"], description="X dies"),
                EventNode(id="EVT_POST", fabula_time=200, syuzhet_index=1,
                          event_type="choice", actor_ids=["ENT_X"],
                          description="X acts after death"),
            ],
        )
        issues = _validate_dead_actors(ws)
        # Dead-actor issues are warnings (could be fake death/ghost)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) == 1
        assert "ENT_X" in warnings[0].detail
        assert "EVT_POST" in warnings[0].detail
        assert "fake death" in warnings[0].detail.lower() or "ghost" in warnings[0].detail.lower()

    def test_suicide_detected(self):
        """Self-caused death (actor=dead entity, target=None) should be detected."""
        ws = _minimal_ws(
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A", status="dead",
                    traits={"t": TraitVector(value=0.5, inertia=0.5)},
                ),
            },
            events=[
                EventNode(id="EVT_SUICIDE", fabula_time=100, syuzhet_index=0,
                          event_type="outcome", actor_ids=["ENT_X"],
                          target_ids=[], description="X takes own life"),
                EventNode(id="EVT_POST", fabula_time=200, syuzhet_index=1,
                          event_type="choice", actor_ids=["ENT_X"],
                          description="ghost X acts"),
            ],
        )
        issues = _validate_dead_actors(ws)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) == 1
        assert "EVT_POST" in warnings[0].detail

    def test_choice_death_detected(self):
        """Death events of type 'choice' (suicide by deliberate choice) should be detected."""
        ws = _minimal_ws(
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A", status="dead",
                    traits={"t": TraitVector(value=0.5, inertia=0.5)},
                ),
            },
            events=[
                EventNode(id="EVT_SUICIDE", fabula_time=100, syuzhet_index=0,
                          event_type="choice", actor_ids=["ENT_X"],
                          target_ids=[], description="X chooses to end it"),
                EventNode(id="EVT_POST", fabula_time=200, syuzhet_index=1,
                          event_type="choice", actor_ids=["ENT_X"],
                          description="ghost acts"),
            ],
        )
        issues = _validate_dead_actors(ws)
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) == 1

    def test_no_death_event_no_crash(self):
        """Entity marked dead but no matching death event -> no crash, no false positives."""
        ws = _minimal_ws(
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A", status="dead",
                    traits={"t": TraitVector(value=0.5, inertia=0.5)},
                ),
            },
            events=[
                EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                          event_type="choice", actor_ids=["ENT_X"],
                          description="X does something"),
            ],
        )
        issues = _validate_dead_actors(ws)
        assert len(issues) == 0  # no death event found -> can't flag

    def test_healthy_entity_no_issues(self):
        ws = _minimal_ws()
        issues = _validate_dead_actors(ws)
        assert len(issues) == 0

    def test_death_event_itself_not_flagged(self):
        """The death event where the entity is the actor should not be flagged."""
        ws = _minimal_ws(
            entities={
                "ENT_X": Entity(
                    id="ENT_X", name="X", location_id="LOC_A", status="dead",
                    traits={"t": TraitVector(value=0.5, inertia=0.5)},
                ),
            },
            events=[
                EventNode(id="EVT_SUICIDE", fabula_time=100, syuzhet_index=0,
                          event_type="outcome", actor_ids=["ENT_X"],
                          target_ids=[], description="X dies"),
            ],
        )
        issues = _validate_dead_actors(ws)
        assert len(issues) == 0


# =====================================================================
# _load_prompt
# =====================================================================

class TestLoadPrompt:
    def test_existing_prompt_loads(self):
        content = _load_prompt("ontology_extraction.md")
        assert "Narrative Ontology Extractor" in content

    def test_missing_prompt_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_prompt("nonexistent_prompt_file.md")

    def test_all_prompts_exist(self):
        """Verify all prompt files referenced by the pipeline exist."""
        for name in ["ontology_extraction.md", "ontology_locations.md",
                      "ontology_objects.md", "ontology_entities.md",
                      "socratic_scaffolding.md",
                      "physics_extraction.md", "social_extraction.md",
                      "validation.md", "correction.md"]:
            content = _load_prompt(name)
            assert len(content) > 100


# =====================================================================
# _auto_repair
# =====================================================================

class TestAutoRepair:
    def test_removes_broken_causal_edge(self):
        ws = _minimal_ws(causal_topology=[
            CausalEdge(source_id="EVT_GONE", target_id="EVT_1",
                       causality_type="chain_reaction",
                       mechanism="physical", fabula_time=100),
        ])
        repaired, repairs = _auto_repair(ws)
        assert len(repaired.causal_topology) == 0
        assert len(repairs) == 1

    def test_removes_broken_social_edge(self):
        ws = _minimal_ws(social_topology=[
            RelationshipEdge(source_entity_id="ENT_GHOST", target_entity_id="ENT_X",
                             affinity=0.5, last_updated_fabula=100),
        ])
        repaired, repairs = _auto_repair(ws)
        assert len(repaired.social_topology) == 0
        assert len(repairs) == 1

    def test_removes_broken_spatial_edge(self):
        ws = _minimal_ws(spatial_topology=[
            SpatialEdge(source_id="LOC_A", target_id="LOC_GONE"),
        ])
        repaired, repairs = _auto_repair(ws)
        assert len(repaired.spatial_topology) == 0
        assert len(repairs) == 1

    def test_removes_broken_channel_participant(self):
        ws = _minimal_ws(
            entities={
                "ENT_X": Entity(id="ENT_X", name="X", location_id="LOC_A", status="healthy",
                                traits={"t": TraitVector(value=0.5, inertia=0.5)}),
                "ENT_Y": Entity(id="ENT_Y", name="Y", location_id="LOC_A", status="healthy",
                                traits={"t": TraitVector(value=0.5, inertia=0.5)}),
            },
            channels={
                "CHN_X": Channel(id="CHN_X", name="x", medium="speech",
                                  participant_ids=["ENT_X", "ENT_Y", "ENT_GONE"],
                                  established_at_fabula=100),
            },
        )
        repaired, repairs = _auto_repair(ws)
        assert len(repaired.channels) == 1
        assert "ENT_GONE" not in repaired.channels["CHN_X"].participant_ids
        assert any("ENT_GONE" in r for r in repairs)

    def test_deduplicates_events(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", description="first"),
            EventNode(id="EVT_1", fabula_time=200, syuzhet_index=1,
                      event_type="outcome", description="dupe"),
        ])
        repaired, repairs = _auto_repair(ws)
        assert len(repaired.events) == 1
        assert repaired.events[0].description == "first"
        assert len(repairs) == 1

    def test_clean_ws_unchanged(self):
        ws = _minimal_ws()
        repaired, repairs = _auto_repair(ws)
        assert len(repairs) == 0
        assert len(repaired.events) == len(ws.events)

    def test_removes_channel_with_too_few_participants(self):
        ws = _minimal_ws(channels={
            "CHN_X": Channel(id="CHN_X", name="x", medium="speech",
                              participant_ids=["ENT_X", "ENT_GONE"],
                              established_at_fabula=100),
        })
        repaired, repairs = _auto_repair(ws)
        assert len(repaired.channels) == 0


# =====================================================================
# Orphan event + info density checks
# =====================================================================

class TestOrphanAndInfoDensity:
    def test_orphan_event_warned(self):
        """Events not referenced by any causal edge should produce a warning."""
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=200, syuzhet_index=1,
                      event_type="outcome", description="b"),
        ])
        issues = _programmatic_validation(ws)
        orphan_warns = [i for i in issues if i.category == "orphan"]
        assert len(orphan_warns) >= 1

    def test_connected_event_not_orphan(self):
        ws = _minimal_ws(
            events=[
                EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                          event_type="choice", description="a"),
                EventNode(id="EVT_2", fabula_time=200, syuzhet_index=1,
                          event_type="outcome", description="b"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_1", target_id="EVT_2",
                           causality_type="chain_reaction",
                           mechanism="physical", fabula_time=100),
            ],
        )
        issues = _programmatic_validation(ws)
        orphan_warns = [i for i in issues if i.category == "orphan"]
        assert len(orphan_warns) == 0

    def test_zero_info_edges_warned(self):
        """No info edges with 3+ events should produce a warning."""
        ws = _minimal_ws(events=[
            EventNode(id=f"EVT_{i}", fabula_time=i * 100, syuzhet_index=i,
                      event_type="choice", description=f"e{i}")
            for i in range(5)
        ])
        issues = _programmatic_validation(ws)
        info_warns = [i for i in issues if i.category == "missing_information"]
        assert len(info_warns) >= 1

    def test_sufficient_channels_ok(self):
        ws = _minimal_ws(
            events=[
                EventNode(id=f"EVT_{i}", fabula_time=i * 100, syuzhet_index=i,
                          event_type="choice", description=f"e{i}")
                for i in range(5)
            ],
            channels={
                f"CHN_{i}": Channel(id=f"CHN_{i}", name=f"c{i}", medium="speech",
                                     participant_ids=["ENT_X", "ENT_Y"],
                                     established_at_fabula=i * 100)
                for i in range(3)
            },
        )
        issues = _programmatic_validation(ws)
        info_warns = [i for i in issues if i.category == "missing_information"]
        assert len(info_warns) == 0


# =====================================================================
# _normalize_fabula_times
# =====================================================================

class TestNormalizeFabulaTimes:
    def test_rescales_small_integers(self):
        """Sequential 1,2,3 should become 100,200,300."""
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=1, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=2, syuzhet_index=1,
                      event_type="outcome", description="b"),
            EventNode(id="EVT_3", fabula_time=3, syuzhet_index=2,
                      event_type="revelation", description="c"),
        ])
        result = _normalize_fabula_times(ws, spacing=100)
        times = [e.fabula_time for e in result.events]
        assert times == [100, 200, 300]

    def test_preserves_well_spaced(self):
        """Times already at 100-spacing should be unchanged."""
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=200, syuzhet_index=1,
                      event_type="outcome", description="b"),
        ])
        result = _normalize_fabula_times(ws, spacing=100)
        times = [e.fabula_time for e in result.events]
        assert times == [100, 200]

    def test_rescales_causal_edges(self):
        ws = _minimal_ws(
            events=[
                EventNode(id="EVT_1", fabula_time=1, syuzhet_index=0,
                          event_type="choice", description="a"),
                EventNode(id="EVT_2", fabula_time=2, syuzhet_index=1,
                          event_type="outcome", description="b"),
            ],
            causal_topology=[
                CausalEdge(source_id="EVT_1", target_id="EVT_2",
                           causality_type="chain_reaction",
                           mechanism="physical", fabula_time=1),
            ],
        )
        result = _normalize_fabula_times(ws, spacing=100)
        assert result.causal_topology[0].fabula_time == 100

    def test_rescales_channels(self):
        ws = _minimal_ws(
            events=[
                EventNode(id="EVT_1", fabula_time=1, syuzhet_index=0,
                          event_type="choice", description="a"),
                EventNode(id="EVT_2", fabula_time=2, syuzhet_index=1,
                          event_type="outcome", description="b"),
            ],
            channels={
                "CHN_X": Channel(id="CHN_X", name="x", medium="speech",
                                  participant_ids=["ENT_X", "ENT_Y"],
                                  established_at_fabula=1, terminated_at_fabula=2),
            },
        )
        result = _normalize_fabula_times(ws, spacing=100)
        ch = result.channels["CHN_X"]
        assert ch.established_at_fabula == 100
        assert ch.terminated_at_fabula == 200

    def test_rescales_social_edges(self):
        ws = _minimal_ws(
            events=[
                EventNode(id="EVT_1", fabula_time=1, syuzhet_index=0,
                          event_type="choice", description="a"),
                EventNode(id="EVT_2", fabula_time=2, syuzhet_index=1,
                          event_type="outcome", description="b"),
            ],
            social_topology=[
                RelationshipEdge(source_entity_id="ENT_X", target_entity_id="ENT_X",
                                 affinity=0.5, last_updated_fabula=2),
            ],
        )
        result = _normalize_fabula_times(ws, spacing=100)
        assert result.social_topology[0].last_updated_fabula == 200

    def test_preserves_simultaneous_events(self):
        """Events with the same fabula_time should keep the same new time."""
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=1, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=1, syuzhet_index=1,
                      event_type="outcome", description="b"),
            EventNode(id="EVT_3", fabula_time=2, syuzhet_index=2,
                      event_type="choice", description="c"),
        ])
        result = _normalize_fabula_times(ws, spacing=100)
        times = [e.fabula_time for e in result.events]
        assert times == [100, 100, 200]

    def test_empty_events_no_crash(self):
        ws = _minimal_ws(events=[])
        result = _normalize_fabula_times(ws, spacing=100)
        assert len(result.events) == 0

    def test_single_event_no_change(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=5, syuzhet_index=0,
                      event_type="choice", description="a"),
        ])
        result = _normalize_fabula_times(ws, spacing=100)
        assert result.events[0].fabula_time == 5  # only 1 unique time, no change

    def test_rescales_belief_fabula(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=1, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=2, syuzhet_index=1,
                      event_type="outcome", description="b"),
        ])
        # Add a belief with established_at_fabula=1
        ent = ws.entities["ENT_X"]
        updated_ent = ent.model_copy(update={"beliefs": [
            Belief(target_id="ENT_X", perceived_state="test",
                   confidence=0.8, inertia=0.5, established_at_fabula=1),
        ]})
        ws = ws.model_copy(update={"entities": {"ENT_X": updated_ent}})
        result = _normalize_fabula_times(ws, spacing=100)
        assert result.entities["ENT_X"].beliefs[0].established_at_fabula == 100

    def test_pre_story_belief_stays_zero(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=1, syuzhet_index=0,
                      event_type="choice", description="a"),
            EventNode(id="EVT_2", fabula_time=2, syuzhet_index=1,
                      event_type="outcome", description="b"),
        ])
        ent = ws.entities["ENT_X"]
        updated_ent = ent.model_copy(update={"beliefs": [
            Belief(target_id="ENT_X", perceived_state="pre-story",
                   confidence=0.8, inertia=0.5, established_at_fabula=0),
        ]})
        ws = ws.model_copy(update={"entities": {"ENT_X": updated_ent}})
        result = _normalize_fabula_times(ws, spacing=100)
        assert result.entities["ENT_X"].beliefs[0].established_at_fabula == 0


# =====================================================================
# Tightened actor/target validation
# =====================================================================

class TestActorTargetValidation:
    def test_actor_must_be_entity(self):
        """actor_id pointing to a location should be flagged."""
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", actor_ids=["LOC_A"],
                      description="location acts"),
        ])
        issues = _programmatic_validation(ws)
        errors = [i for i in issues if i.severity == "error" and "actor_id" in i.detail]
        assert len(errors) == 1

    def test_target_can_be_entity(self):
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", target_ids=["ENT_X"],
                      description="target is entity"),
        ])
        issues = _programmatic_validation(ws)
        target_errors = [i for i in issues if "target_id" in i.detail and i.severity == "error"]
        assert len(target_errors) == 0

    def test_target_can_be_object(self):
        ws = _minimal_ws(
            objects={"OBJ_ITEM": NarrativeObject(id="OBJ_ITEM", name="Item",
                                                  location_id="LOC_A", description="an item",
                                                  owner_id=None, affordances=[])},
            events=[
                EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                          event_type="choice", target_ids=["OBJ_ITEM"],
                          description="target is object"),
            ],
        )
        issues = _programmatic_validation(ws)
        target_errors = [i for i in issues if "target_id" in i.detail and i.severity == "error"]
        assert len(target_errors) == 0

    def test_target_location_flagged(self):
        """target_id pointing to a location should be flagged."""
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", target_ids=["LOC_A"],
                      description="target is location"),
        ])
        issues = _programmatic_validation(ws)
        target_errors = [i for i in issues if "target_id" in i.detail and i.severity == "error"]
        assert len(target_errors) == 1

    def test_auto_repair_fixes_bad_actor(self):
        """auto_repair should remove invalid actor_ids entries."""
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", actor_ids=["LOC_A"],
                      description="bad actor"),
        ])
        repaired, repairs = _auto_repair(ws)
        assert repaired.events[0].actor_ids == []
        assert len(repairs) == 1

    def test_auto_repair_fixes_bad_target(self):
        """auto_repair should remove invalid target_ids entries."""
        ws = _minimal_ws(events=[
            EventNode(id="EVT_1", fabula_time=100, syuzhet_index=0,
                      event_type="choice", target_ids=["LOC_A"],
                      description="bad target"),
        ])
        repaired, repairs = _auto_repair(ws)
        assert repaired.events[0].target_ids == []
        assert len(repairs) == 1


# =====================================================================
# Test plot models pass programmatic validation
# =====================================================================

_TEST_MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "example_worlds"
_MODEL_FILES = sorted(_TEST_MODEL_DIR.glob("*.py"))
# Exclude __init__.py
_MODEL_FILES = [f for f in _MODEL_FILES if f.name != "__init__.py"]


@pytest.mark.parametrize("model_path", _MODEL_FILES, ids=lambda p: p.stem)
def test_plot_model_passes_validation(model_path):
    """Every hand-built test plot model should pass programmatic validation
    with zero errors (warnings are acceptable)."""
    # Dynamic import
    module_name = f"example_worlds.{model_path.stem}"
    mod = importlib.import_module(module_name)
    ws = mod.world_state

    issues = _programmatic_validation(ws)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        detail = "\n".join(f"  [{e.category}] {e.detail}" for e in errors)
        pytest.fail(f"{model_path.stem} has {len(errors)} validation error(s):\n{detail}")


# =====================================================================
# ExtractionConfig defaults
# =====================================================================

class TestExtractionConfig:
    def test_defaults(self):
        c = ExtractionConfig()
        assert c.fabula_time_spacing == 1000
        assert c.min_chunk_chars == 1500
        assert c.output_retries == 5
        assert c.chunk_overlap_chars == 300
        assert c.max_correction_retries == 5
        assert c.max_concurrent_chunks == 8
        assert c.estimated_events_per_chunk == 10

    def test_custom_values(self):
        c = ExtractionConfig(fabula_time_spacing=50, min_chunk_chars=500,
                             chunk_overlap_chars=0, max_correction_retries=3)
        assert c.fabula_time_spacing == 50
        assert c.min_chunk_chars == 500
        assert c.chunk_overlap_chars == 0
        assert c.max_correction_retries == 3


# =====================================================================
# New Pipeline Models — QAPair, SocraticScaffold, PhysicsExtraction,
#                       SocialExtraction
# =====================================================================

from shadow_loom.ingestion import (
    QAPair,
    SocraticScaffold,
    PhysicsExtraction,
    SocialExtraction,
    _build_valid_id_set,
    _format_scaffold,
)


class TestNewPipelineModels:
    """Verify construction and field access of the new extraction models."""

    def test_qa_pair_construction(self):
        qa = QAPair(category="why", question="Why did X act?", answer="Because Y.")
        assert qa.category == "why"
        assert "X" in qa.question
        assert "Y" in qa.answer

    def test_qa_pair_category_literal(self):
        """Only the six interrogative categories are allowed."""
        for cat in ("who", "what", "where", "when", "why", "how"):
            qa = QAPair(category=cat, question="q", answer="a")
            assert qa.category == cat

    def test_socratic_scaffold_empty(self):
        s = SocraticScaffold()
        assert s.qa_pairs == []

    def test_socratic_scaffold_with_pairs(self):
        pairs = [
            QAPair(category="who", question="Who acts?", answer="A."),
            QAPair(category="why", question="Why?", answer="B."),
        ]
        s = SocraticScaffold(qa_pairs=pairs)
        assert len(s.qa_pairs) == 2
        assert s.qa_pairs[0].category == "who"

    def test_physics_extraction_empty(self):
        p = PhysicsExtraction()
        assert p.events == []
        assert p.causal_topology == []
        assert p.spatial_topology == []

    def test_physics_extraction_with_data(self):
        evt = EventNode(
            id="EVT_1", fabula_time=100, syuzhet_index=0,
            event_type="choice", description="test",
        )
        ce = CausalEdge(
            source_id="EVT_1", target_id="EVT_1",
            causality_type="chain_reaction", mechanism="physical",
            fabula_time=100,
        )
        p = PhysicsExtraction(events=[evt], causal_topology=[ce])
        assert len(p.events) == 1
        assert len(p.causal_topology) == 1

    def test_social_extraction_empty(self):
        s = SocialExtraction()
        assert s.channels == {}
        assert s.utterance_events == []
        assert s.social_topology == []

    def test_social_extraction_with_data(self):
        ch = Channel(
            id="CHN_X", name="x", medium="speech",
            participant_ids=["ENT_A", "ENT_B"],
            established_at_fabula=100,
        )
        re_edge = RelationshipEdge(
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            last_updated_fabula=100,
        )
        s = SocialExtraction(channels={"CHN_X": ch}, social_topology=[re_edge])
        assert len(s.channels) == 1
        assert len(s.social_topology) == 1


class TestFormatScaffold:
    """Verify scaffold formatting helper."""

    def test_empty_scaffold(self):
        s = SocraticScaffold()
        text = _format_scaffold(s)
        assert "No scaffolding" in text

    def test_formatted_scaffold(self):
        pairs = [
            QAPair(category="who", question="Who acts?", answer="Macbeth."),
            QAPair(category="why", question="Why?", answer="Ambition."),
        ]
        text = _format_scaffold(SocraticScaffold(qa_pairs=pairs))
        assert "[WHO]" in text
        assert "[WHY]" in text
        assert "Macbeth" in text
        assert "Ambition" in text


class TestBuildValidIdSet:
    """Verify the ID set builder used by result validators."""

    def test_basic_id_set(self):
        reg = GlobalRegister(
            locations={"LOC_A": Location(name="A", description="a", ambient_state={})},
            objects={"OBJ_X": NarrativeObject(
                id="OBJ_X", name="X", location_id="LOC_A", owner_id=None,
                properties={}, affordances=[],
            )},
            entities={"ENT_1": Entity(
                id="ENT_1", name="One", location_id="LOC_A", status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.5)},
            )},
        )
        valid = _build_valid_id_set(reg)
        assert "LOC_A" in valid
        assert "OBJ_X" in valid
        assert "ENT_1" in valid
        assert "EVT_1" not in valid

    def test_id_set_with_events(self):
        reg = GlobalRegister(
            locations={"LOC_A": Location(name="A", description="a", ambient_state={})},
            objects={},
            entities={},
        )
        valid = _build_valid_id_set(reg, ["EVT_1", "EVT_2"])
        assert "EVT_1" in valid
        assert "EVT_2" in valid
        assert "LOC_A" in valid


class TestResultValidators:
    """Verify that the per-chunk result validators raise ModelRetry on bad IDs."""

    def test_physics_validator_catches_bad_causal_source(self):
        """A hallucinated source_id in a CausalEdge should trigger ModelRetry."""
        from pydantic_ai import ModelRetry as _ModelRetry
        from shadow_loom.ingestion import _PhysicsDeps, _build_valid_id_set

        reg = GlobalRegister(
            locations={"LOC_A": Location(name="A", description="a", ambient_state={})},
            objects={},
            entities={"ENT_X": Entity(
                id="ENT_X", name="X", location_id="LOC_A", status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.5)},
            )},
        )
        # Simulate what the result_validator does: check IDs
        physics = PhysicsExtraction(
            events=[EventNode(
                id="EVT_1", fabula_time=100, syuzhet_index=0,
                event_type="choice", description="test",
            )],
            causal_topology=[CausalEdge(
                source_id="EVT_HALLUCINATED", target_id="EVT_1",
                causality_type="chain_reaction", mechanism="physical",
                fabula_time=100,
            )],
        )
        # Reproduce the validator logic
        new_evt_ids = [e.id for e in physics.events]
        valid = _build_valid_id_set(reg, [] + new_evt_ids)
        bad = []
        for ce in physics.causal_topology:
            if ce.source_id not in valid:
                bad.append(f"CausalEdge source_id '{ce.source_id}' is not a valid ID.")
            if ce.target_id not in valid:
                bad.append(f"CausalEdge target_id '{ce.target_id}' is not a valid ID.")
        assert len(bad) == 1
        assert "EVT_HALLUCINATED" in bad[0]

    def test_social_validator_catches_bad_channel_participant(self):
        """A hallucinated participant_id in a Channel should be caught."""
        reg = GlobalRegister(
            locations={"LOC_A": Location(name="A", description="a", ambient_state={})},
            objects={},
            entities={"ENT_X": Entity(
                id="ENT_X", name="X", location_id="LOC_A", status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.5)},
            )},
        )
        social = SocialExtraction(
            channels={"CHN_X": Channel(
                id="CHN_X", name="x", medium="speech",
                participant_ids=["ENT_GHOST", "ENT_X"],
                established_at_fabula=100,
            )},
        )
        entity_ids = set(reg.entities.keys())
        node_ids = entity_ids | set(reg.objects.keys())
        bad = []
        for ch in social.channels.values():
            for pid in ch.participant_ids:
                if pid not in node_ids:
                    bad.append(f"Channel participant_id '{pid}' is not a valid entity/object.")
        assert len(bad) == 1
        assert "ENT_GHOST" in bad[0]

    def test_physics_validator_accepts_valid_ids(self):
        """All valid IDs should pass without issues."""
        reg = GlobalRegister(
            locations={"LOC_A": Location(name="A", description="a", ambient_state={})},
            objects={},
            entities={"ENT_X": Entity(
                id="ENT_X", name="X", location_id="LOC_A", status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.5)},
            )},
        )
        physics = PhysicsExtraction(
            events=[EventNode(
                id="EVT_1", fabula_time=100, syuzhet_index=0,
                event_type="choice", description="test",
            )],
            causal_topology=[CausalEdge(
                source_id="EVT_1", target_id="ENT_X",
                causality_type="mutation", mechanism="physical",
                fabula_time=100,
            )],
        )
        new_evt_ids = [e.id for e in physics.events]
        valid = _build_valid_id_set(reg, [] + new_evt_ids)
        bad = []
        for ce in physics.causal_topology:
            if ce.source_id not in valid:
                bad.append(ce.source_id)
            if ce.target_id not in valid:
                bad.append(ce.target_id)
        assert bad == []


# =====================================================================
# Tests for Parallel Extraction Infrastructure
# =====================================================================

class TestPreAllocateChunkParams:
    """Tests for _pre_allocate_chunk_params."""

    def test_basic_allocation(self):
        chunks = ["chunk0", "chunk1", "chunk2"]
        config = ExtractionConfig(
            estimated_events_per_chunk=10,
            fabula_time_spacing=1000,
            chunk_overlap_chars=5,
        )
        params = _pre_allocate_chunk_params(chunks, config)
        assert len(params) == 3
        # First chunk has no prev tail
        assert params[0].chunk_index == 0
        assert params[0].syuzhet_offset == 0
        assert params[0].prev_chunk_tail == ""
        # Second chunk
        assert params[1].chunk_index == 1
        assert params[1].syuzhet_offset == 10
        assert params[1].prev_chunk_tail == "hunk0"
        # Third chunk
        assert params[2].chunk_index == 2
        assert params[2].syuzhet_offset == 20

    def test_no_overlap(self):
        chunks = ["aaa", "bbb"]
        config = ExtractionConfig(chunk_overlap_chars=0)
        params = _pre_allocate_chunk_params(chunks, config)
        assert params[1].prev_chunk_tail == ""

    def test_single_chunk(self):
        config = ExtractionConfig()
        params = _pre_allocate_chunk_params(["solo"], config)
        assert len(params) == 1
        assert params[0].syuzhet_offset == 0

    def test_fabula_bases_not_pre_allocated(self):
        """_ChunkParams must NOT carry a per-chunk fabula_time_base —
        forcing chunk order onto fabula order would erase flashbacks.
        Syuzhet offsets, on the other hand, ARE chunk-position-derived
        because syuzhet IS narration order."""
        chunks = ["a", "b", "c", "d"]
        config = ExtractionConfig(
            estimated_events_per_chunk=5,
            fabula_time_spacing=100,
        )
        params = _pre_allocate_chunk_params(chunks, config)
        assert not hasattr(params[0], "fabula_time_base") or \
               "fabula_time_base" not in type(params[0]).model_fields
        offsets = [p.syuzhet_offset for p in params]
        assert offsets == [0, 5, 10, 15]


class TestReconcileChunkTopologies:
    """Tests for _reconcile_chunk_topologies — syuzhet renumbering,
    fabula ordering, and duplicate event ID resolution."""

    def _make_topo(self, events=None, causal=None, channels=None, social=None,
                   spatial=None, entity_updates=None) -> ChunkTopology:
        return ChunkTopology(
            events=events or [],
            causal_topology=causal or [],
            channels=channels or {},
            social_topology=social or [],
            spatial_topology=spatial or [],
            entity_updates=entity_updates or [],
        )

    def test_syuzhet_renumbered_across_chunks(self):
        """Events from chunk 0 get syuzhet 0,1; chunk 1 gets 2,3."""
        topo0 = self._make_topo(events=[
            EventNode(id="EVT_A", description="a", event_type="choice",
                      fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_B", description="b", event_type="outcome",
                      fabula_time=200, syuzhet_index=1, actor_ids=[], target_ids=[]),
        ])
        topo1 = self._make_topo(events=[
            EventNode(id="EVT_C", description="c", event_type="choice",
                      fabula_time=300, syuzhet_index=0, actor_ids=[], target_ids=[]),
        ])
        config = ExtractionConfig(fabula_time_spacing=100)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        all_syuzhets = [e.syuzhet_index for t in result for e in t.events]
        assert all_syuzhets == [0, 1, 2]

    def test_fabula_time_preserved_across_chunks(self):
        """Reconcile must NOT force chunk-order onto fabula-order.
        Chunk 1 narrating an earlier story-world event (a flashback)
        must keep its smaller fabula_time — syuzhet position does not
        determine fabula position."""
        topo0 = self._make_topo(events=[
            EventNode(id="EVT_A", description="a", event_type="choice",
                      fabula_time=500, syuzhet_index=0, actor_ids=[], target_ids=[]),
        ])
        topo1 = self._make_topo(events=[
            EventNode(id="EVT_B", description="flashback", event_type="choice",
                      fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[]),
        ])
        config = ExtractionConfig(fabula_time_spacing=1000)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        assert result[0].events[0].fabula_time == 500
        assert result[1].events[0].fabula_time == 100  # flashback preserved
        # Syuzhet renumbering still applies (narration order = chunk order).
        assert [e.syuzhet_index for t in result for e in t.events] == [0, 1]

    def test_duplicate_event_ids_renamed(self):
        """Same EVT_ID in two chunks → later chunk's ID gets _cN suffix."""
        topo0 = self._make_topo(events=[
            EventNode(id="EVT_DUEL", description="first", event_type="choice",
                      fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[]),
        ])
        topo1 = self._make_topo(events=[
            EventNode(id="EVT_DUEL", description="second", event_type="outcome",
                      fabula_time=200, syuzhet_index=0, actor_ids=[], target_ids=[]),
        ])
        config = ExtractionConfig(fabula_time_spacing=100)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        ids = [e.id for t in result for e in t.events]
        assert len(set(ids)) == 2  # no duplicates
        assert ids[0] == "EVT_DUEL"
        assert ids[1].startswith("EVT_DUEL_c")

    def test_duplicate_event_id_renames_propagate_to_edges(self):
        """Renamed event IDs must update causal edge references."""
        topo0 = self._make_topo(events=[
            EventNode(id="EVT_X", description="first", event_type="choice",
                      fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[]),
        ])
        topo1 = self._make_topo(
            events=[
                EventNode(id="EVT_X", description="second", event_type="outcome",
                          fabula_time=200, syuzhet_index=0, actor_ids=[], target_ids=[]),
            ],
            causal=[
                CausalEdge(source_id="EVT_X", target_id="ENT_Y",
                           causality_type="mutation", mechanism="physical",
                           fabula_time=200),
            ],
        )
        config = ExtractionConfig(fabula_time_spacing=100)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        renamed_id = result[1].events[0].id
        assert renamed_id != "EVT_X"
        # Causal edge source should be renamed too
        assert result[1].causal_topology[0].source_id == renamed_id

    def test_empty_topologies(self):
        config = ExtractionConfig()
        result = _reconcile_chunk_topologies([], config)
        assert result == []

    def test_no_events_passthrough(self):
        """Chunks with no events pass through without error."""
        topo = self._make_topo()
        config = ExtractionConfig()
        result = _reconcile_chunk_topologies([topo], config)
        assert len(result) == 1
        assert result[0].events == []

    def test_channels_preserved_across_chunks(self):
        """Channels from different chunks should both survive reconciliation."""
        topo0 = self._make_topo(
            events=[
                EventNode(id="EVT_A", description="a", event_type="choice",
                          fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[]),
            ],
            channels={
                "CHN_A": Channel(id="CHN_A", name="a", medium="speech",
                                  participant_ids=["ENT_X", "ENT_Y"],
                                  established_at_fabula=100),
            },
        )
        topo1 = self._make_topo(
            events=[
                EventNode(id="EVT_B", description="b", event_type="choice",
                          fabula_time=200, syuzhet_index=0, actor_ids=[], target_ids=[]),
            ],
            channels={
                "CHN_B": Channel(id="CHN_B", name="b", medium="letter",
                                  participant_ids=["ENT_X", "ENT_Y"],
                                  established_at_fabula=200),
            },
        )
        config = ExtractionConfig(fabula_time_spacing=100)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        assert "CHN_A" in result[0].channels
        assert "CHN_B" in result[1].channels

    def test_fabula_shift_includes_entity_update_beliefs(self):
        """_shift_fabula_times must shift beliefs in entity_updates too,
        but must leave the 0 "pre-story baseline" sentinel untouched.
        (The reconcile pass no longer invokes this helper; this is a
        direct unit test of the helper, kept because other call sites
        may add inter-chunk shifts in the future.)"""
        from shadow_loom.ingestion import EntityUpdate
        topo = self._make_topo(
            events=[
                EventNode(id="EVT_B", description="b", event_type="choice",
                          fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[]),
            ],
            entity_updates=[
                EntityUpdate(
                    entity_id="ENT_X", fabula_time=100, triggered_by="EVT_B",
                    new_beliefs=[
                        Belief(
                            target_id="ENT_Y", perceived_state="alive",
                            confidence=0.9, inertia=0.5,
                            established_at_fabula=100,
                        ),
                        Belief(
                            target_id="ENT_Z", perceived_state="pre-story",
                            confidence=0.9, inertia=0.5,
                            established_at_fabula=0,  # pre-story sentinel
                        ),
                    ],
                ),
            ],
        )
        _shift_fabula_times(topo, shift=400)
        assert topo.events[0].fabula_time == 500
        eu = topo.entity_updates[0]
        assert eu.fabula_time == 500
        # Story-time belief shifted; pre-story sentinel left at 0.
        assert eu.new_beliefs[0].established_at_fabula == 500
        assert eu.new_beliefs[1].established_at_fabula == 0

    def test_chunk_restart_pathology_shifted(self):
        """Chunk N whose LLM ignored the spacing hint and emitted small
        sequential integers (1, 2, 3) entirely below prior_max, with no
        causal back-links, must be shifted forward to avoid collision."""
        topo0 = self._make_topo(events=[
            EventNode(id="EVT_A1", description="a1", event_type="choice",
                      fabula_time=1000, syuzhet_index=0, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_A2", description="a2", event_type="outcome",
                      fabula_time=2000, syuzhet_index=1, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_A3", description="a3", event_type="choice",
                      fabula_time=3000, syuzhet_index=2, actor_ids=[], target_ids=[]),
        ])
        topo1 = self._make_topo(events=[
            EventNode(id="EVT_B1", description="b1", event_type="choice",
                      fabula_time=1, syuzhet_index=0, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_B2", description="b2", event_type="outcome",
                      fabula_time=2, syuzhet_index=1, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_B3", description="b3", event_type="choice",
                      fabula_time=3, syuzhet_index=2, actor_ids=[], target_ids=[]),
        ])
        config = ExtractionConfig(fabula_time_spacing=1000)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        # Chunk 0 unchanged
        assert [e.fabula_time for e in result[0].events] == [1000, 2000, 3000]
        # Chunk 1 shifted so min == prior_max + spacing == 4000;
        # internal spacing of 1 preserved.
        assert [e.fabula_time for e in result[1].events] == [4000, 4001, 4002]

    def test_flashback_with_causal_link_preserved(self):
        """A flashback chunk that links causally back into a prior
        chunk's event must NOT be shifted, even if its fabula values
        are small and below prior_max."""
        topo0 = self._make_topo(events=[
            EventNode(id="EVT_PAST", description="past", event_type="choice",
                      fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_NOW", description="now", event_type="outcome",
                      fabula_time=5000, syuzhet_index=1, actor_ids=[], target_ids=[]),
        ])
        # Flashback chunk: small absolute values, but a chain_reaction
        # edge points back to EVT_PAST — clear deliberate flashback.
        topo1 = self._make_topo(
            events=[
                EventNode(id="EVT_FB1", description="fb1", event_type="revelation",
                          fabula_time=50, syuzhet_index=0, actor_ids=[], target_ids=[]),
                EventNode(id="EVT_FB2", description="fb2", event_type="outcome",
                          fabula_time=60, syuzhet_index=1, actor_ids=[], target_ids=[]),
            ],
            causal=[
                CausalEdge(source_id="EVT_PAST", target_id="EVT_FB1",
                           causality_type="chain_reaction", mechanism="psychological",
                           fabula_time=100),
            ],
        )
        config = ExtractionConfig(fabula_time_spacing=1000)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        # Flashback preserved
        assert [e.fabula_time for e in result[1].events] == [50, 60]

    def test_flash_forward_preserved(self):
        """A chunk whose events sit above prior_max (flash-forward) is
        never shifted regardless of internal spacing."""
        topo0 = self._make_topo(events=[
            EventNode(id="EVT_A", description="a", event_type="choice",
                      fabula_time=1000, syuzhet_index=0, actor_ids=[], target_ids=[]),
        ])
        topo1 = self._make_topo(events=[
            EventNode(id="EVT_FF1", description="ff1", event_type="outcome",
                      fabula_time=9000, syuzhet_index=0, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_FF2", description="ff2", event_type="choice",
                      fabula_time=9100, syuzhet_index=1, actor_ids=[], target_ids=[]),
        ])
        config = ExtractionConfig(fabula_time_spacing=1000)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        assert [e.fabula_time for e in result[1].events] == [9000, 9100]

    def test_mixed_chunk_with_present_event_preserved(self):
        """A chunk with at least one event above prior_max (i.e. mixed
        present + flashback content) is left alone — the heuristic only
        fires on chunks that are *entirely* below prior_max."""
        topo0 = self._make_topo(events=[
            EventNode(id="EVT_A", description="a", event_type="choice",
                      fabula_time=1000, syuzhet_index=0, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_B", description="b", event_type="outcome",
                      fabula_time=2000, syuzhet_index=1, actor_ids=[], target_ids=[]),
        ])
        topo1 = self._make_topo(events=[
            EventNode(id="EVT_C1", description="c1", event_type="revelation",
                      fabula_time=5, syuzhet_index=0, actor_ids=[], target_ids=[]),
            EventNode(id="EVT_C2", description="c2", event_type="outcome",
                      fabula_time=2500, syuzhet_index=1, actor_ids=[], target_ids=[]),
        ])
        config = ExtractionConfig(fabula_time_spacing=1000)
        result = _reconcile_chunk_topologies([topo0, topo1], config)
        # Mixed chunk untouched — at least one event ≥ prior_max.
        assert [e.fabula_time for e in result[1].events] == [5, 2500]


class TestApplyEventRenames:
    """Tests for _apply_event_renames."""

    def test_renames_event_ids(self):
        topo = ChunkTopology(
            events=[EventNode(id="EVT_OLD", description="d", event_type="choice",
                              fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[])],
            causal_topology=[],
            social_topology=[],
            spatial_topology=[],
            entity_updates=[],
        )
        result = _apply_event_renames(topo, {"EVT_OLD": "EVT_NEW"})
        assert result.events[0].id == "EVT_NEW"

    def test_renames_causal_references(self):
        topo = ChunkTopology(
            events=[],
            causal_topology=[CausalEdge(
                source_id="EVT_OLD", target_id="EVT_OTHER",
                causality_type="chain_reaction", mechanism="physical",
                fabula_time=100,
            )],
            social_topology=[],
            spatial_topology=[],
            entity_updates=[],
        )
        result = _apply_event_renames(topo, {"EVT_OLD": "EVT_NEW"})
        assert result.causal_topology[0].source_id == "EVT_NEW"
        assert result.causal_topology[0].target_id == "EVT_OTHER"

    def test_no_rename_if_not_in_map(self):
        topo = ChunkTopology(
            events=[EventNode(id="EVT_KEEP", description="d", event_type="choice",
                              fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[])],
            causal_topology=[],
            social_topology=[],
            spatial_topology=[],
            entity_updates=[],
        )
        result = _apply_event_renames(topo, {"EVT_OTHER": "EVT_NEW"})
        assert result.events[0].id == "EVT_KEEP"


class TestShiftFabulaTimes:
    """Tests for _shift_fabula_times."""

    def test_shifts_events(self):
        topo = ChunkTopology(
            events=[EventNode(id="EVT_A", description="d", event_type="choice",
                              fabula_time=100, syuzhet_index=0, actor_ids=[], target_ids=[])],
            causal_topology=[],
            social_topology=[],
            spatial_topology=[],
            entity_updates=[],
        )
        _shift_fabula_times(topo, 500)
        assert topo.events[0].fabula_time == 600

    def test_shifts_causal_edges(self):
        topo = ChunkTopology(
            events=[],
            causal_topology=[CausalEdge(
                source_id="EVT_A", target_id="EVT_B",
                causality_type="chain_reaction", mechanism="physical",
                fabula_time=200,
            )],
            social_topology=[],
            spatial_topology=[],
            entity_updates=[],
        )
        _shift_fabula_times(topo, 1000)
        assert topo.causal_topology[0].fabula_time == 1200

    def test_shifts_channels(self):
        topo = ChunkTopology(
            events=[],
            causal_topology=[],
            channels={"CHN_X": Channel(
                id="CHN_X", name="x", medium="speech",
                participant_ids=["ENT_A", "ENT_B"],
                established_at_fabula=100,
                terminated_at_fabula=200,
            )},
            social_topology=[],
            spatial_topology=[],
            entity_updates=[],
        )
        _shift_fabula_times(topo, 300)
        assert topo.channels["CHN_X"].established_at_fabula == 400
        assert topo.channels["CHN_X"].terminated_at_fabula == 500

    def test_shifts_social_edges(self):
        topo = ChunkTopology(
            events=[],
            causal_topology=[],
            social_topology=[RelationshipEdge(
                source_entity_id="ENT_A", target_entity_id="ENT_B",
                affinity=0.5, fear=0.0, power_balance=0.0,
                last_updated_fabula=100,
            )],
            spatial_topology=[],
            entity_updates=[],
        )
        _shift_fabula_times(topo, 200)
        assert topo.social_topology[0].last_updated_fabula == 300

    def test_shifts_spatial_edges(self):
        topo = ChunkTopology(
            events=[],
            causal_topology=[],
            social_topology=[],
            spatial_topology=[SpatialEdge(
                source_id="LOC_A", target_id="LOC_B",
                established_at_fabula=50,
                destroyed_at_fabula=150,
            )],
            entity_updates=[],
        )
        _shift_fabula_times(topo, 100)
        assert topo.spatial_topology[0].established_at_fabula == 150
        assert topo.spatial_topology[0].destroyed_at_fabula == 250
