"""
Tests for the Ancestral Multi-World Network (AMWN) and ctf-calculus
(Correa & Bareinboim, ICML 2025) implementation in
``shadow_loom.amwn``, plus its pre-flight wiring inside
``CausalPhysicsEngine.execute()``.

Covers:
  • build_causal_diagram — structural projection of WorldStateV1
  • build_amwn — node shadowing across intervention contexts
  • Rule 1 (consistency) — vacuous do(X = observed(X))
  • Rule 2 (independence) — d-separation on the AMWN
  • Rule 3 (exclusion) — vacuous interventions on disconnected nodes
  • apply_ctf_calculus — pre-flight report
  • CausalPhysicsEngine integration — report propagates into result
"""
import pytest
import networkx as nx

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode,
    CausalEdge, SpatialEdge, TraitVector,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.amwn import (
    CounterfactualVar, CtfCalculusReport,
    build_causal_diagram, build_amwn,
    check_consistency, check_ctf_independence, check_exclusion,
    apply_ctf_calculus,
)


# =====================================================================
# Fixtures
# =====================================================================

def _linear_world() -> WorldStateV1:
    """Three-event chain: EVT_A → EVT_B → ENT_TARGET. ENT_OTHER disconnected."""
    return WorldStateV1(
        locations={
            "LOC_A": Location(name="A", description="A", ambient_state={}),
        },
        objects={},
        entities={
            "ENT_TARGET": Entity(
                id="ENT_TARGET", name="Target", location_id="LOC_A",
                status="healthy",
                traits={"fear": TraitVector(value=0.3, inertia=0.1)},
            ),
            "ENT_OTHER": Entity(
                id="ENT_OTHER", name="Other", location_id="LOC_A",
                status="healthy",
                traits={"fear": TraitVector(value=0.5, inertia=0.1)},
            ),
        },
        events=[
            EventNode(id="EVT_A", fabula_time=1, syuzhet_index=1,
                      event_type="choice", actor_ids=[], target_ids=[],
                      description="A happens"),
            EventNode(id="EVT_B", fabula_time=2, syuzhet_index=2,
                      event_type="outcome", actor_ids=[],
                      target_ids=["ENT_TARGET"],
                      description="B follows A"),
            EventNode(id="EVT_ISOLATED", fabula_time=3, syuzhet_index=3,
                      event_type="outcome", actor_ids=[],
                      description="Unrelated event"),
        ],
        causal_topology=[
            CausalEdge(source_id="EVT_A", target_id="EVT_B",
                       causality_type="chain_reaction", causal_force=5.0,
                       mechanism="physical", evidence_strength="strong",
                       fabula_time=1),
            CausalEdge(source_id="EVT_B", target_id="ENT_TARGET",
                       causality_type="mutation", causal_force=5.0,
                       mechanism="physical", evidence_strength="strong",
                       fabula_time=2, trait_target="fear", trait_delta=0.5),
        ],
        spatial_topology=[],
    )


# =====================================================================
# build_causal_diagram
# =====================================================================
class TestBuildCausalDiagram:

    def test_includes_all_node_types(self):
        ws = _linear_world()
        g = build_causal_diagram(ws)
        # Entities, events, locations all present
        for nid in ["ENT_TARGET", "ENT_OTHER", "EVT_A", "EVT_B",
                    "EVT_ISOLATED", "LOC_A"]:
            assert g.has_node(nid)

    def test_edges_match_causal_topology(self):
        ws = _linear_world()
        g = build_causal_diagram(ws)
        assert g.has_edge("EVT_A", "EVT_B")
        assert g.has_edge("EVT_B", "ENT_TARGET")
        # No spurious edges
        assert not g.has_edge("EVT_A", "ENT_TARGET")
        assert not g.has_edge("EVT_ISOLATED", "ENT_TARGET")

    def test_self_loops_dropped(self):
        ws = _linear_world()
        ws.causal_topology.append(CausalEdge(
            source_id="EVT_A", target_id="EVT_A",
            causality_type="chain_reaction", causal_force=1.0,
            mechanism="physical", evidence_strength="weak",
            fabula_time=1,
        ))
        g = build_causal_diagram(ws)
        assert not g.has_edge("EVT_A", "EVT_A")


# =====================================================================
# build_amwn — node shadowing
# =====================================================================
class TestBuildAmwn:

    def test_single_world_shape(self):
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        amwn = build_amwn(diag, [("ENT_TARGET", {"EVT_B": "fired"})])
        # Intervened node EVT_B is included with no incoming edges
        evt_b_nodes = [n for n in amwn.nodes if n.var_id == "EVT_B"]
        assert len(evt_b_nodes) == 1
        assert amwn.nodes[evt_b_nodes[0]]["intervened"] is True
        assert amwn.in_degree(evt_b_nodes[0]) == 0
        # ENT_TARGET is included downstream
        target_nodes = [n for n in amwn.nodes if n.var_id == "ENT_TARGET"]
        assert len(target_nodes) == 1

    def test_node_shadowing_across_compatible_contexts(self):
        """Two query vars with the *same projected context* on a
        common ancestor should yield ONE shared AMWN node — that is
        the heart of the ancestral-shadowing construction."""
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        # Both queries intervene on EVT_A in the same way.
        amwn = build_amwn(diag, [
            ("ENT_TARGET", {"EVT_A": "x"}),
            ("EVT_B", {"EVT_A": "x"}),
        ])
        evt_a_nodes = [n for n in amwn.nodes if n.var_id == "EVT_A"]
        # EVT_A is shadowed: a single node represents both worlds
        assert len(evt_a_nodes) == 1

    def test_distinct_contexts_split_nodes(self):
        """Two interventions on the same variable produce *the same*
        do-surgery (incoming edges cut) regardless of value — d-sep on
        the AMWN therefore treats them as a single node. Value-sensitive
        reasoning lives in :func:`check_consistency`, not in node
        identity."""
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        amwn = build_amwn(diag, [
            ("ENT_TARGET", {"EVT_A": "x"}),
            ("ENT_TARGET", {"EVT_A": "y"}),
        ])
        target_nodes = [n for n in amwn.nodes if n.var_id == "ENT_TARGET"]
        assert len(target_nodes) == 1


# =====================================================================
# Rule 1 — Consistency
# =====================================================================
class TestRule1Consistency:

    def test_consistent_values_redundant(self):
        assert check_consistency("ENT_X", "alive", "alive") is True

    def test_inconsistent_values_not_redundant(self):
        assert check_consistency("ENT_X", "dead", "alive") is False


# =====================================================================
# Rule 3 — Exclusion
# =====================================================================
class TestRule3Exclusion:

    def test_disconnected_intervention_excluded(self):
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        # EVT_ISOLATED has no path to ENT_TARGET → vacuous
        assert check_exclusion(
            diag, x_set={"EVT_ISOLATED"}, y_set={"ENT_TARGET"},
        ) is True

    def test_connected_intervention_not_excluded(self):
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        assert check_exclusion(
            diag, x_set={"EVT_A"}, y_set={"ENT_TARGET"},
        ) is False

    def test_blocking_with_z(self):
        """Cutting EVT_B from its parents should leave EVT_A
        with no path to ENT_TARGET."""
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        # With Z={EVT_B}, edges INTO EVT_B are removed, so EVT_A
        # loses its only path to ENT_TARGET.
        assert check_exclusion(
            diag, x_set={"EVT_A"}, y_set={"ENT_TARGET"}, z_set={"EVT_B"},
        ) is True


# =====================================================================
# Rule 2 — Independence (d-separation on AMWN)
# =====================================================================
class TestRule2Independence:

    def test_disconnected_vars_independent(self):
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        # EVT_ISOLATED ⊥ ENT_TARGET trivially (no shared ancestry)
        assert check_ctf_independence(
            diag,
            x_vars=[("EVT_ISOLATED", {})],
            y_vars=[("ENT_TARGET", {})],
        ) is True

    def test_causal_chain_not_independent(self):
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        # EVT_A → EVT_B → ENT_TARGET — clearly dependent
        assert check_ctf_independence(
            diag,
            x_vars=[("EVT_A", {})],
            y_vars=[("ENT_TARGET", {})],
        ) is False

    def test_conditioning_blocks_chain(self):
        ws = _linear_world()
        diag = build_causal_diagram(ws)
        # Conditioning on EVT_B blocks the EVT_A → ENT_TARGET path.
        assert check_ctf_independence(
            diag,
            x_vars=[("EVT_A", {})],
            y_vars=[("ENT_TARGET", {})],
            z_vars=[("EVT_B", {})],
        ) is True


# =====================================================================
# apply_ctf_calculus — pre-flight report
# =====================================================================
class TestApplyCtfCalculus:

    def test_rule3_prunes_isolated_intervention(self):
        ws = _linear_world()
        report = apply_ctf_calculus(
            ws,
            interventions={"EVT_ISOLATED": "fired"},
            target_node_ids=["ENT_TARGET"],
        )
        assert "EVT_ISOLATED" in report.rule3_pruned

    def test_rule3_keeps_relevant_intervention(self):
        ws = _linear_world()
        report = apply_ctf_calculus(
            ws,
            interventions={"EVT_A": "fired"},
            target_node_ids=["ENT_TARGET"],
        )
        assert "EVT_A" not in report.rule3_pruned

    def test_rule2_flags_disconnected_evidence(self):
        ws = _linear_world()
        report = apply_ctf_calculus(
            ws,
            interventions={"EVT_A": "fired"},
            evidence_node_ids=["ENT_OTHER"],   # disconnected entity
        )
        assert "ENT_OTHER" in report.rule2_redundant_evidence

    def test_empty_interventions_returns_empty_report(self):
        ws = _linear_world()
        report = apply_ctf_calculus(ws, interventions={})
        assert report.is_empty()


# =====================================================================
# Engine integration: ctf-calculus report appears in CausalPhysicsResult
# =====================================================================
class TestEnginePreflight:

    def _build_sandbox(self, ws: WorldStateV1):
        ego = extract_ego_graph_from_memory(ws, list(ws.entities.keys()))
        return AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")

    def test_rule2_flag_propagates_to_result(self):
        ws = _linear_world()
        sandbox = self._build_sandbox(ws)
        engine = CausalPhysicsEngine(sandbox, ws)

        # ENT_OTHER has no causal connection to EVT_A's effects.
        result = engine.execute(
            rung=3,
            interventions={"EVT_A": "fired"},
            evidence_node_ids=["ENT_OTHER"],
        )

        assert "ENT_OTHER" in result.rule2_redundant_evidence

    def test_no_interventions_no_report(self):
        ws = _linear_world()
        sandbox = self._build_sandbox(ws)
        engine = CausalPhysicsEngine(sandbox, ws)

        result = engine.execute(rung=2, interventions={})
        assert result.rule3_pruned_interventions == []
        assert result.rule2_redundant_evidence == []
