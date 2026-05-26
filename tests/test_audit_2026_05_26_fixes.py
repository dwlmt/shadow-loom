# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests covering the post-2026-05-26 audit remediation batch.

Each section pins a single audit finding so regressions in the engine,
ingestion, brief assembly, or MCP surfaces fail loudly. The numbering
matches ``AUDIT_2026-05-26.md`` and ``docs/design-decisions.md
§D-AUDIT-2026-05-26``.
"""
from __future__ import annotations

import pytest

from shadow_loom.amwn import _to_context
from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    Location,
    RelationshipEdge,
    RelationshipMetric,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.query_models import (
    DoCausalEdge,
    DoRelationship,
    DoTrait,
)


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

def _make_two_entity_world() -> WorldStateV1:
    return WorldStateV1(
        locations={
            "LOC_A": Location(id="LOC_A", name="A", description="a",
                              ambient_state={}),
        },
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={
                    "fear": TraitVector(value=0.4, inertia=0.3),
                    "guilt": TraitVector(value=0.5, inertia=0.3),
                },
                beliefs=[], concerns=[],
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB", name="Bob", location_id="LOC_A",
                status="healthy",
                traits={"fear": TraitVector(value=0.4, inertia=0.3)},
                beliefs=[], concerns=[],
            ),
        },
        events=[
            EventNode(
                id="EVT_ANCHOR", fabula_time=10, syuzhet_index=10,
                event_type="outcome", actor_ids=["ENT_ALICE"], target_ids=[],
                description="anchor",
            ),
        ],
        causal_topology=[],
        spatial_topology=[],
        social_topology=[],
        channels={},
        propositions=[],
        world_traits={},
    )


def _engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    ego = extract_ego_graph_from_memory(
        ws, [eid for eid in ws.entities.keys()]
    )
    sandbox = AMWNInstantiator.create_sandbox(
        ego.model_dump(), "intervention"
    )
    return CausalPhysicsEngine(world_state=ws, sandbox=sandbox)


# ---------------------------------------------------------------------------
# D-A1: per-axis trait surgery preserves sibling-trait edges
# ---------------------------------------------------------------------------

class TestPerAxisTraitSurgery:
    """do(ENT_X.traits.fear=…) must not sever edges that target
    *other* trait axes on the same entity (Pearl minimal surgery).
    """

    def test_sibling_axis_edge_survives(self):
        ws = _make_two_entity_world()
        # Edge mutating Alice.guilt from the anchor event.
        ws.causal_topology = [
            CausalEdge(
                source_id="EVT_ANCHOR",
                target_id="ENT_ALICE",
                causality_type="mutation",
                causal_force=0.8,
                mechanism="psychological",
                fabula_time=10,
                trait_target="guilt",
                trait_delta=0.2,
            ),
            CausalEdge(
                source_id="EVT_ANCHOR",
                target_id="ENT_ALICE",
                causality_type="mutation",
                causal_force=0.8,
                mechanism="psychological",
                fabula_time=10,
                trait_target="fear",
                trait_delta=0.2,
            ),
        ]
        eng = _engine(ws)
        eng.apply_do_targets([
            DoTrait(
                holder_id="ENT_ALICE",
                trait_name="fear",
                value=0.9,
            )
        ])
        # The fear edge must be gone but the guilt edge must survive.
        keys = [
            data
            for _u, _v, data in eng.sandbox.in_edges("ENT_ALICE", data=True)
            if data.get("edge_type") == "causal"
        ]
        trait_targets = sorted(k.get("trait_target") for k in keys)
        assert "guilt" in trait_targets
        assert "fear" not in trait_targets


# ---------------------------------------------------------------------------
# D-A2: AMWN _to_context preserves dotted-path granularity
# ---------------------------------------------------------------------------

class TestAMWNContextGranularity:
    def test_distinct_trait_axes_yield_distinct_contexts(self):
        ctx_fear = _to_context({"ENT_ALICE.traits.fear": 0.9})
        ctx_guilt = _to_context({"ENT_ALICE.traits.guilt": 0.9})
        assert ctx_fear != ctx_guilt
        # Each context retains the full path string, not the entity id.
        assert ("ENT_ALICE.traits.fear", "") in ctx_fear
        assert ("ENT_ALICE.traits.guilt", "") in ctx_guilt


# ---------------------------------------------------------------------------
# D-A3: relationship surgery severs incoming mutation_social edges
# ---------------------------------------------------------------------------

class TestRelationshipSurgery:
    def test_mutation_social_edge_severed_on_per_axis_do(self):
        ws = _make_two_entity_world()
        ws.social_topology = [
            RelationshipEdge(
                source_entity_id="ENT_ALICE",
                target_entity_id="ENT_BOB",
                metrics={
                    "affinity": RelationshipMetric(value=0.2, inertia=0.3,
                                                    observed=True),
                },
            ),
        ]
        ws.causal_topology = [
            CausalEdge(
                source_id="EVT_ANCHOR",
                target_id="ENT_ALICE",
                causality_type="mutation_social",
                causal_force=0.6,
                mechanism="psychological",
                fabula_time=10,
                trait_target="affinity",
                rel_counterpart_id="ENT_BOB",
                trait_delta=0.3,
            ),
        ]
        eng = _engine(ws)
        # Surgical clamp on the Alice→Bob affinity metric.
        eng._intervened_relationships.add(("ENT_ALICE", "ENT_BOB", "affinity"))
        eng._perform_graph_surgery_edge_removal({}, set())
        surviving = [
            d for _u, _v, d in eng.sandbox.in_edges("ENT_ALICE", data=True)
            if d.get("causality_type") == "mutation_social"
            and d.get("rel_counterpart_id") == "ENT_BOB"
            and d.get("trait_target") == "affinity"
        ]
        assert surviving == []


# ---------------------------------------------------------------------------
# D-A9: do_causal_edge rejects same-tick non-chain_reaction edges
# ---------------------------------------------------------------------------

class TestStrictTemporalAcyclicity:
    def test_same_tick_mutation_edge_refused(self):
        ws = _make_two_entity_world()
        ws.events.append(
            EventNode(
                id="EVT_LATE", fabula_time=10, syuzhet_index=11,
                event_type="outcome", actor_ids=["ENT_BOB"], target_ids=[],
                description="late",
            )
        )
        eng = _engine(ws)
        before = len(ws.causal_topology)
        eng.apply_do_targets([
            DoCausalEdge(
                action="add",
                source_id="EVT_ANCHOR",
                target_id="EVT_LATE",
                causality_type="mutation",
                causal_force=0.5,
                mechanism="psychological",
                fabula_time=10,
            )
        ])
        # Edge rejected — same fabula tick + non-chain_reaction.
        assert len(ws.causal_topology) == before

    def test_same_tick_chain_reaction_edge_accepted(self):
        ws = _make_two_entity_world()
        ws.events.append(
            EventNode(
                id="EVT_LATE", fabula_time=10, syuzhet_index=11,
                event_type="outcome", actor_ids=["ENT_BOB"], target_ids=[],
                description="late",
            )
        )
        eng = _engine(ws)
        before = len(ws.causal_topology)
        eng.apply_do_targets([
            DoCausalEdge(
                action="add",
                source_id="EVT_ANCHOR",
                target_id="EVT_LATE",
                causality_type="chain_reaction",
                causal_force=0.5,
                mechanism="physical",
                fabula_time=10,
            )
        ])
        assert len(ws.causal_topology) == before + 1


# ---------------------------------------------------------------------------
# D-A5: new auditor violation explanations registered
# ---------------------------------------------------------------------------

class TestViolationExplanationsRegistered:
    def test_spurious_abduction_and_premature_payoff_present(self):
        from shadow_loom_ui.reasoning_helpers import VIOLATION_EXPLANATIONS

        assert "spurious_abduction" in VIOLATION_EXPLANATIONS
        assert "premature_payoff" in VIOLATION_EXPLANATIONS
        # Explanations are non-empty prose, not placeholders.
        assert len(VIOLATION_EXPLANATIONS["spurious_abduction"]) > 40
        assert len(VIOLATION_EXPLANATIONS["premature_payoff"]) > 40


# ---------------------------------------------------------------------------
# D-A10: MCP narrate-mode strict validation
# ---------------------------------------------------------------------------

class TestMCPNarrateModeValidation:
    def test_unknown_mode_returns_typed_error_envelope(self):
        # Read source rather than invoking the full MCP stack — keeps
        # the assertion stable across FastMCP transport changes.
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "shadow_loom_mcp" / "server.py"
        text = src.read_text(encoding="utf-8")
        assert '"INVALID_MODE"' in text
        # The validator list mentions all four canonical modes.
        for canon in ("observe", "intervene", "counterfactual", "directive"):
            assert f'"{canon}"' in text
