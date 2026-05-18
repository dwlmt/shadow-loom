# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round-6 audit \u2014 the *create* surface of the query / pipeline.

Covers the four defects identified during the round-6 focused audit of
how the engine instantiates new world content from a user query:

* **R6-F1** \u2014 ``_apply_do_causal_edge`` action="add" used to append the
  edge to ``world_state.causal_topology`` unconditionally even when
  the sandbox endpoints were unknown (typo / pruned-from-ego). The
  sandbox add was correctly skipped, leaving the canonical world with
  a dangling-endpoint edge.
* **R6-F2** \u2014 Same defect in ``_apply_do_spatial_edge`` action="add"
  against ``world_state.spatial_topology``.
* **R6-F3** \u2014 ``introduced_elements_to_spawns`` Concern branch was
  missing the idempotency guard every other spawn branch has, so a
  duplicate spec doubled the holder's belief.
* **R6-F4** \u2014 Same branch never validated ``proposition_id`` against
  existing-or-co-introduced propositions; a dangling reference fell
  silently inert downstream. Now logs WARNING.
"""
from __future__ import annotations

import logging

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.extract_graph import (
    extract_ego_graph_from_memory,
    introduced_elements_to_spawns,
)
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.introduced_elements import (
    IntroducedConcernSpec,
    IntroducedElements,
    IntroducedPropositionSpec,
)
from shadow_loom.models import (
    Entity,
    EventNode,
    Location,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.query_models import DoCausalEdge, DoSpatialEdge


def _make_world() -> WorldStateV1:
    return WorldStateV1(
        locations={
            "LOC_A": Location(name="A", description="A", ambient_state={}),
            "LOC_B": Location(name="B", description="B", ambient_state={}),
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
                id="EVT_PUSH", fabula_time=10, syuzhet_index=10,
                event_type="choice", actor_ids=["ENT_BOB"],
                target_ids=["ENT_ALICE"], description="Bob shoves Alice.",
            ),
        ],
        causal_topology=[],
        spatial_topology=[
            SpatialEdge(source_id="LOC_A", target_id="LOC_B",
                        connection_type="doorway", bidirectional=True),
        ],
        social_topology=[],
        channels={},
    )


def _engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE", "ENT_BOB"])
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    return CausalPhysicsEngine(sandbox, ws)


# ---------------------------------------------------------------------
# R6-F1: DoCausalEdge add refuses dangling endpoints on world side
# ---------------------------------------------------------------------
class TestDoCausalEdgeAddRefusesDanglingEndpoints:
    def test_unknown_target_does_not_persist(self, caplog):
        ws = _make_world()
        eng = _engine(ws)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
            eng.apply_do_targets([DoCausalEdge(
                source_id="EVT_PUSH",
                target_id="ENT_GHOST",  # not in world.entities
                action="add",
                causality_type="mutation",
                mechanism="physical",
                trait_target="courage",
                trait_delta=-0.1,
                fabula_time=10,
            )])
        assert ws.causal_topology == [], (
            "world causal_topology must not accumulate dangling-endpoint edges"
        )
        assert any(
            "endpoint(s) unknown" in r.getMessage()
            for r in caplog.records
        )

    def test_unknown_source_does_not_persist(self):
        ws = _make_world()
        eng = _engine(ws)
        eng.apply_do_targets([DoCausalEdge(
            source_id="EVT_GHOST",  # not in world.events
            target_id="ENT_ALICE",
            action="add",
            causality_type="mutation",
            mechanism="psychological",
            trait_target="courage",
            trait_delta=-0.1,
            fabula_time=10,
        )])
        assert ws.causal_topology == []

    def test_known_endpoints_still_persist(self):
        # Regression guard: the happy path must remain unaffected.
        ws = _make_world()
        eng = _engine(ws)
        eng.apply_do_targets([DoCausalEdge(
            source_id="EVT_PUSH", target_id="ENT_ALICE",
            action="add",
            causality_type="mutation",
            mechanism="physical",
            trait_target="courage",
            trait_delta=-0.2,
            fabula_time=10,
        )])
        added = [
            e for e in ws.causal_topology
            if e.source_id == "EVT_PUSH" and e.target_id == "ENT_ALICE"
        ]
        assert len(added) == 1


# ---------------------------------------------------------------------
# R6-F2: DoSpatialEdge add refuses dangling location endpoints
# ---------------------------------------------------------------------
class TestDoSpatialEdgeAddRefusesDanglingEndpoints:
    def test_unknown_location_does_not_persist(self, caplog):
        ws = _make_world()
        before = list(ws.spatial_topology)
        eng = _engine(ws)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.causal_physics"):
            eng.apply_do_targets([DoSpatialEdge(
                source_id="LOC_A",
                target_id="LOC_GHOST",  # not in world.locations
                action="add",
                connection_type="passage",
            )])
        assert ws.spatial_topology == before, (
            "world spatial_topology must not accumulate dangling-LOC_ edges"
        )
        assert any(
            "not in world.locations" in r.getMessage()
            for r in caplog.records
        )

    def test_known_locations_still_persist(self):
        ws = _make_world()
        ws.locations["LOC_C"] = Location(name="C", description="C", ambient_state={})
        eng = _engine(ws)
        eng.apply_do_targets([DoSpatialEdge(
            source_id="LOC_B", target_id="LOC_C", action="add",
            connection_type="passage",
        )])
        added = [
            e for e in ws.spatial_topology
            if e.source_id == "LOC_B" and e.target_id == "LOC_C"
        ]
        assert len(added) == 1


# ---------------------------------------------------------------------
# R6-F3: Concern spawn is idempotent on duplicate id
# ---------------------------------------------------------------------
class TestIntroducedConcernIdempotent:
    def test_duplicate_spec_id_appended_once(self):
        ws = _make_world()
        intro = IntroducedElements(
            propositions=[IntroducedPropositionSpec(
                id="PROP_ESCAPE", name="Alice escapes",
                justification="anchor",
            )],
            concerns=[
                IntroducedConcernSpec(
                    id="CCN_DUP", name="alice-wants-out",
                    justification="motivation",
                    holder_entity_id="ENT_ALICE",
                    proposition_id="PROP_ESCAPE",
                    polarity="desire", salience=0.8,
                ),
                IntroducedConcernSpec(
                    id="CCN_DUP", name="alice-wants-out-again",
                    justification="duplicate emission",
                    holder_entity_id="ENT_ALICE",
                    proposition_id="PROP_ESCAPE",
                    polarity="desire", salience=0.9,
                ),
            ],
        )
        spawns = introduced_elements_to_spawns(intro, ws)
        attached = spawns["concerns"]["ENT_ALICE"]
        assert len(attached) == 1, "duplicate concern id must spawn at most once"
        assert attached[0].concern_id == "CCN_DUP"

    def test_id_already_on_holder_is_skipped(self):
        # If the holder entity already carries the concern, the spec
        # is a no-op (mirrors every other branch's existing-id guard).
        from shadow_loom.models import Concern
        ws = _make_world()
        ws.entities["ENT_ALICE"].concerns = [Concern(
            world_id="factual",
            concern_id="CCN_PREEXIST",
            proposition_id="PROP_X",
            polarity="desire", salience=0.5,
        )]
        intro = IntroducedElements(
            propositions=[IntroducedPropositionSpec(
                id="PROP_X", name="Anchor", justification="anchor",
            )],
            concerns=[IntroducedConcernSpec(
                id="CCN_PREEXIST", name="dup",
                justification="should be skipped",
                holder_entity_id="ENT_ALICE",
                proposition_id="PROP_X",
                polarity="desire", salience=0.9,
            )],
        )
        spawns = introduced_elements_to_spawns(intro, ws)
        assert spawns["concerns"].get("ENT_ALICE", []) == []


# ---------------------------------------------------------------------
# R6-F4: Concern spawn warns on dangling proposition_id
# ---------------------------------------------------------------------
class TestIntroducedConcernDanglingPropositionWarns:
    def test_dangling_proposition_logs_warning(self, caplog):
        ws = _make_world()
        intro = IntroducedElements(
            concerns=[IntroducedConcernSpec(
                id="CCN_DANGLE", name="dangler",
                justification="references a prop nobody declared",
                holder_entity_id="ENT_ALICE",
                proposition_id="PROP_NOWHERE",
                polarity="desire", salience=0.5,
            )],
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.extract_graph"):
            spawns = introduced_elements_to_spawns(intro, ws)
        # Concern still spawns (resilient) but a warning was emitted
        # so the operator can fix the dangling reference.
        assert len(spawns["concerns"].get("ENT_ALICE", [])) == 1
        assert any(
            "references" in r.getMessage() and "PROP_NOWHERE" in r.getMessage()
            for r in caplog.records
        )

    def test_co_introduced_proposition_does_not_warn(self, caplog):
        # When the proposition is declared in the SAME payload, the
        # reference is valid \u2014 the warning must not fire.
        ws = _make_world()
        intro = IntroducedElements(
            propositions=[IntroducedPropositionSpec(
                id="PROP_CO", name="Co-declared prop",
                justification="anchor",
            )],
            concerns=[IntroducedConcernSpec(
                id="CCN_OK", name="valid",
                justification="references a co-introduced prop",
                holder_entity_id="ENT_ALICE",
                proposition_id="PROP_CO",
                polarity="desire", salience=0.5,
            )],
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.extract_graph"):
            introduced_elements_to_spawns(intro, ws)
        assert not any(
            "references" in r.getMessage() and "PROP_CO" in r.getMessage()
            for r in caplog.records
        )
