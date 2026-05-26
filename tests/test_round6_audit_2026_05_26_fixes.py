# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the round-6 audit remediation batch dated
2026-05-26.

Each section pins one finding from the round-6 audit so a regression
in the manual-edit replacement cascade (RelationshipEdge /
SpatialEdge attribute names), the MCP re-extraction routing for
observation queries, or the fuzzy-repair handling of dotted
``target_vector_id`` references will fail loudly.
"""
from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from shadow_loom.models import (
    Entity,
    EventNode,
    Location,
    RelationshipEdge,
    RelationshipMetric,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.pipeline import _apply_manual_edit_replacements
from shadow_loom.query_models import ManualEditQuery, ObservationQuery
from shadow_loom.query_parsing import _try_fuzzy_repair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _world() -> WorldStateV1:
    return WorldStateV1(
        locations={
            "LOC_A": Location(
                id="LOC_A", name="A", description="A", ambient_state={},
            ),
            "LOC_B": Location(
                id="LOC_B", name="B", description="B", ambient_state={},
            ),
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
            EventNode(
                id="EVT_MEET", fabula_time=1, syuzhet_index=1,
                event_type="choice", actor_ids=["ENT_ALICE"],
                target_ids=["ENT_BOB"], description="They meet",
            ),
        ],
        social_topology=[
            RelationshipEdge(
                source_entity_id="ENT_ALICE",
                target_entity_id="ENT_BOB",
                metrics={
                    "affinity": RelationshipMetric(
                        value=0.2, inertia=0.3, observed=True,
                        last_updated_fabula=1,
                    ),
                },
                established_at_fabula=1,
            ),
        ],
        spatial_topology=[
            SpatialEdge(
                source_id="LOC_A", target_id="LOC_B",
                connection_type="doorway", bidirectional=True,
                established_at_fabula=0,
            ),
        ],
        causal_topology=[],
    )


# ===========================================================================
# #1 — replace_entity_ids drops dependent social edges via correct attrs
# ===========================================================================

class TestManualEditDropEntityCascadesSocialEdges:
    """The prior implementation referenced ``re.source_id`` /
    ``re.target_id`` on ``RelationshipEdge``; those attributes do not
    exist (the real names are ``source_entity_id`` /
    ``target_entity_id``), so every manual_edit that listed an entity
    in ``replace_entity_ids`` raised ``AttributeError`` and the merge
    aborted.
    """

    def test_dropping_endpoint_removes_relationship_edge(self):
        ws = _world()
        query = ManualEditQuery(
            reasoning="r",
            edited_prose="Bob is gone.",
            replace_entity_ids=["ENT_BOB"],
        )
        # Must not raise AttributeError.
        new_ws = _apply_manual_edit_replacements(ws, query)
        assert "ENT_BOB" not in new_ws.entities
        # Every surviving social edge whose endpoint was Bob must
        # have been cascaded away (the model validator may seed a
        # reverse edge, but both directions referenced ENT_BOB).
        assert new_ws.social_topology == []
        # Original ws untouched (deep copy).
        assert "ENT_BOB" in ws.entities
        assert ws.social_topology, "original ws social topology preserved"


# ===========================================================================
# #2 — replace_location_ids drops dependent spatial edges via correct attrs
# ===========================================================================

class TestManualEditDropLocationCascadesSpatialEdges:
    """``SpatialEdge`` exposes ``source_id`` / ``target_id`` (no
    ``location_id`` attribute exists). The prior filter raised
    ``AttributeError`` on every replacement.
    """

    def test_dropping_location_removes_spatial_edge(self):
        ws = _world()
        query = ManualEditQuery(
            reasoning="r",
            edited_prose="The doorway collapses.",
            replace_location_ids=["LOC_B"],
        )
        new_ws = _apply_manual_edit_replacements(ws, query)
        assert "LOC_B" not in new_ws.locations
        # Spatial edge whose endpoint referenced LOC_B is gone.
        assert new_ws.spatial_topology == []

    def test_dropping_location_leaves_unrelated_edges_intact(self):
        ws = _world()
        # Add a self-loop edge that does not touch the dropped location.
        ws.spatial_topology.append(SpatialEdge(
            source_id="LOC_A", target_id="LOC_A",
            connection_type="loop", bidirectional=False,
            established_at_fabula=0,
        ))
        query = ManualEditQuery(
            reasoning="r",
            edited_prose="The doorway collapses.",
            replace_location_ids=["LOC_B"],
        )
        new_ws = _apply_manual_edit_replacements(ws, query)
        # The LOC_A self-loop survives.
        survivors = [
            e for e in new_ws.spatial_topology
            if e.source_id == "LOC_A" and e.target_id == "LOC_A"
        ]
        assert len(survivors) == 1


# ===========================================================================
# #3 — MCP run_and_save enables re-extraction for observation queries
# ===========================================================================

class TestObservationTriggersReextraction:
    """``run_and_save`` previously omitted ``observation`` from the
    prose-rendering set, so the MCP ``narrate`` mode (which uses an
    ObservationQuery) saved prose against a stale topology.
    """

    def test_observation_in_prose_rendering_set(self):
        import shadow_loom_mcp.helpers as helpers
        src = inspect.getsource(helpers.run_and_save)
        # The set must include 'observation' alongside the
        # mutation-rendering types.
        assert '"observation"' in src
        assert '"intervention"' in src
        assert '"manual_edit"' in src

    def test_observation_docstring_updated(self):
        import shadow_loom_mcp.helpers as helpers
        doc = (helpers.run_and_save.__doc__ or "")
        assert "observation" in doc


# ===========================================================================
# #4 — _try_fuzzy_repair remaps dotted target_vector_id base id
# ===========================================================================

class TestFuzzyRepairDottedTargetVectorId:
    """``target_vector_id`` typically arrives as ``ENT_X.traits.fear``.
    The prior code only matched the full string; if the remapping
    contained only the base id (``ENT_X -> ENT_Y``) the dotted form
    was left stale and validation kept failing.
    """

    def test_source_handles_dotted_remap(self):
        # Source-inspection: the dotted-base branch must exist.
        src = inspect.getsource(_try_fuzzy_repair)
        # The remap logic for target_vector_id must include the
        # ``"." in tvid`` dotted-base path.
        assert "patched.target_vector_id" in src
        # The dotted-base branch mirrors _remap_dict_keys.
        assert '"." in tvid' in src
        assert "remappings[base]" in src

    def test_dotted_target_vector_id_remapped_directly(self):
        """End-to-end check: build a parsed query with a dotted
        ``target_vector_id`` whose base id is missing from the world,
        run fuzzy repair, and assert the dotted form is rewritten to
        the resolved base id.
        """
        from shadow_loom.query_parsing import ParsedQuery, ValidationError
        ws = _world()
        # ``ENT_ALIC`` is a fuzzy-near miss for ``ENT_ALICE``.
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="r",
            target_entity_ids=["ENT_ALIC"],
            target_vector_id="ENT_ALIC.traits.courage",
            target_effect="suspense",
        )
        errors = [
            ValidationError(
                field="target_entity_ids",
                message="ID 'ENT_ALIC' not found in world model.",
                severity="error",
            ),
        ]
        patched, remappings, remaining = _try_fuzzy_repair(parsed, errors, ws)
        assert remappings.get("ENT_ALIC") == "ENT_ALICE"
        # The dotted target_vector_id must be remapped on the base id.
        assert patched.target_vector_id == "ENT_ALICE.traits.courage"
