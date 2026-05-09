# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end tests for the new-element introduction cycle.

Covers:

* ``query.introduce`` pre-spawning a new entity into the sandbox so a
  do-surgery on it succeeds (Step 6).
* The engine's ``skipped_interventions`` ledger surfacing when a
  do-surgery targets a non-existent node WITHOUT pre-declaration
  (Step 3).
* The auditor's ``undeclared_element`` violation firing when prose
  names an entity the world doesn't know AND the renderer didn't
  declare it (Step 2).
* The renderer's ``GeneratedScene.introduced_elements`` flowing
  through re-extraction so the merge promotes the declared entity
  into the canonical world (Steps 1 + 4).
* The merge's referential-integrity pass populating
  ``MergeChangeset.events_with_dangling_refs`` when an extracted
  event names an unknown id (Step 5).

LLM agents (generation, audit, extraction) are mocked. Engine,
sandbox, merge, integrity check all run real.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx

from shadow_loom.auditor import _undeclared_element_violations
from shadow_loom.extract_graph import (
    VersionedWorldModel,
    introduced_elements_to_spawns,
)
from shadow_loom.generation import GeneratedScene
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.introduced_elements import (
    IntroducedElements,
    IntroducedEntitySpec,
    IntroducedLocationSpec,
)
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import EventNode
from shadow_loom.pipeline import PipelineConfig, run_pipeline
from shadow_loom.query_models import (
    InterventionQuery,
    ObservationQuery,
)

from example_worlds.macbeth import world_state as macbeth_ws


# ---------------------------------------------------------------
# Mock helpers (mirror the conventions in tests/test_pipeline.py)
# ---------------------------------------------------------------

def _mock_run_sync(output):
    m = MagicMock()
    m.output = output
    return m


def _scene(prose: str, introduced: IntroducedElements | None = None) -> GeneratedScene:
    return GeneratedScene(
        prose=prose,
        pov_entity="ENT_TEST",
        rendering_mode="intervention",
        constraints_honoured=["C1"],
        constraints_violated=[],
        introduced_elements=introduced or IntroducedElements(),
    )


def _empty_topology() -> ChunkTopology:
    return ChunkTopology(events=[], causal_topology=[],
                         spatial_topology=[], entity_updates=[])


# ---------------------------------------------------------------
# Step 6: query.introduce pre-spawn lets do-surgery target the new node
# ---------------------------------------------------------------

class TestQueryIntroducePreSpawn:

    def test_introduce_then_intervene_succeeds(self):
        """An InterventionQuery that targets a brand-new entity declared
        on ``query.introduce`` should NOT skip — pre-spawn puts it in
        the sandbox before do-surgery runs."""
        ws = macbeth_ws
        loc_id = next(iter(ws.locations))

        intro = IntroducedElements(
            entities=[IntroducedEntitySpec(
                id="ENT_NEW_MESSENGER",
                name="The messenger",
                justification="Carries a sealed letter to Macbeth.",
                located_in=loc_id,
            )],
        )
        query = InterventionQuery(
            interventions={"ENT_NEW_MESSENGER.traits.urgency.value": 0.9},
            introduce=intro,
        )

        with patch("shadow_loom.pipeline.calculate_narrative_physics") as mphy:
            mphy.return_value = {
                "status": "success",
                "query_type": "intervention",
                "physics_state": {"nodes": [], "edges": []},
                "skipped_interventions": [],
                "mutations": [],
                "blocked": [],
            }
            result = run_pipeline(
                query, world_state=ws,
                config=PipelineConfig(skip_audit=True, skip_reextraction=True),
            )

        # Pre-spawn ran before physics — the entity is on the working ws.
        assert "ENT_NEW_MESSENGER" in mphy.call_args.kwargs[
            "global_world_state"
        ].entities


# ---------------------------------------------------------------
# Step 3: engine ledger fires when intervention targets unknown node
# ---------------------------------------------------------------

class TestEngineSkippedInterventionLedger:

    def test_unknown_node_target_recorded(self):
        sandbox: nx.MultiDiGraph = nx.MultiDiGraph()
        sandbox.add_node("ENT_KNOWN")
        AMWNInstantiator.execute_interventions(sandbox, {
            "ENT_GHOST.traits.fear.value": 0.5,
        })
        skipped = sandbox.graph.get("skipped_interventions") or []
        assert len(skipped) == 1
        assert skipped[0]["node_id"] == "ENT_GHOST"
        assert skipped[0]["reason"] == "unknown_node"

    def test_malformed_key_recorded(self):
        sandbox: nx.MultiDiGraph = nx.MultiDiGraph()
        AMWNInstantiator.execute_interventions(sandbox, {
            "no_dot_here": 1,
        })
        skipped = sandbox.graph.get("skipped_interventions") or []
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "malformed_key"


# ---------------------------------------------------------------
# Step 2: undeclared_element auditor violation
# ---------------------------------------------------------------

class TestUndeclaredElementAudit:

    def test_undeclared_id_in_prose_flagged(self):
        ws = macbeth_ws
        prose = "Macbeth and ENT_PHANTOM crossed the courtyard."
        viols = _undeclared_element_violations(prose, ws, None)
        assert any(
            v.violation_type == "undeclared_element"
            and "ENT_PHANTOM" in (v.evidence_quote or "")
            for v in viols
        )

    def test_declared_id_in_prose_not_flagged(self):
        ws = macbeth_ws
        prose = "Macbeth and ENT_PHANTOM crossed the courtyard."
        intro = IntroducedElements(entities=[IntroducedEntitySpec(
            id="ENT_PHANTOM", name="The phantom",
            justification="A new spectral witness to the murder.",
            located_in=next(iter(ws.locations)),
        )])
        viols = _undeclared_element_violations(prose, ws, intro)
        assert not any(
            v.violation_type == "undeclared_element"
            and "ENT_PHANTOM" in (v.evidence_quote or "")
            for v in viols
        )


# ---------------------------------------------------------------
# Steps 1 + 4: introduced_elements flow through re-extraction as spawns
# ---------------------------------------------------------------

class TestIntroducedElementsToSpawns:

    def test_introduced_to_spawns_payload_shape(self):
        ws = macbeth_ws
        loc_id = next(iter(ws.locations))
        intro = IntroducedElements(
            locations=[IntroducedLocationSpec(
                id="LOC_NEW_GROVE",
                name="A hidden grove",
                justification="The witches summon Macbeth here for the third prophecy.",
                description="A clearing ringed by twisted oaks.",
            )],
            entities=[IntroducedEntitySpec(
                id="ENT_NEW_HERALD",
                name="The herald",
                justification="Announces Banquo's return from the hunt.",
                located_in=loc_id,
            )],
        )
        spawns = introduced_elements_to_spawns(intro, ws)
        assert "LOC_NEW_GROVE" in spawns["locations"]
        assert "ENT_NEW_HERALD" in spawns["entities"]
        assert spawns["entities"]["ENT_NEW_HERALD"].location_id == loc_id


# ---------------------------------------------------------------
# Step 5: merge populates events_with_dangling_refs
# ---------------------------------------------------------------

class TestMergeIntegrityPass:

    def test_dangling_ref_recorded(self):
        ws = macbeth_ws
        vwm = VersionedWorldModel.from_world_state(ws)

        # Build a topology whose new event references an unknown id.
        max_ft = max((e.fabula_time for e in ws.events), default=0)
        new_evt = EventNode(
            id=f"EVT_DANGLING_{max_ft + 1000}",
            fabula_time=max_ft + 1000,
            syuzhet_index=max_ft + 1000,
            event_type="outcome",
            actor_ids=["ENT_PHANTOM_NEVER_DECLARED"],
            target_ids=[],
            description="A phantom acts.",
        )
        topology = ChunkTopology(
            events=[new_evt], causal_topology=[],
            spatial_topology=[], entity_updates=[],
        )
        new_vwm = vwm.merge(topology)
        latest = new_vwm.history[-1].changeset
        assert latest is not None
        drefs = latest.events_with_dangling_refs
        assert len(drefs) == 1
        assert drefs[0]["event_id"] == new_evt.id
        assert "ENT_PHANTOM_NEVER_DECLARED" in drefs[0]["missing_ids"]

    def test_introduced_entity_resolves_no_dangling(self):
        """If the topology's ``new_entities`` includes the referenced
        id, the integrity pass should NOT flag it."""
        ws = macbeth_ws
        vwm = VersionedWorldModel.from_world_state(ws)

        loc_id = next(iter(ws.locations))
        intro = IntroducedElements(entities=[IntroducedEntitySpec(
            id="ENT_NEW_GHOST", name="A new ghost",
            justification="Witnesses Macbeth's downfall.",
            located_in=loc_id,
        )])
        spawns = introduced_elements_to_spawns(intro, ws)

        max_ft = max((e.fabula_time for e in ws.events), default=0)
        new_evt = EventNode(
            id=f"EVT_GHOSTED_{max_ft + 1000}",
            fabula_time=max_ft + 1000,
            syuzhet_index=max_ft + 1000,
            event_type="outcome",
            actor_ids=["ENT_NEW_GHOST"],
            target_ids=[],
            description="The new ghost speaks.",
        )
        topology = ChunkTopology(
            events=[new_evt], causal_topology=[],
            spatial_topology=[], entity_updates=[],
            new_entities=spawns["entities"],
        )
        new_vwm = vwm.merge(topology)
        latest = new_vwm.history[-1].changeset
        assert latest is not None
        assert latest.events_with_dangling_refs == []
        assert "ENT_NEW_GHOST" in new_vwm.current.entities
