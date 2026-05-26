# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 2 + Phase 3 — sandbox utility-layer preservation and the public
typed-target Pearl-rung facades (``apply_intervention`` / ``find_pod``).
"""
import pytest
import networkx as nx

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, Belief, Concern, Proposition,
    TraitVector, CausalEdge,
)
from shadow_loom.narrative_physics import (
    _stamp_utility_layer,
    _load_propositions_from_sandbox,
    apply_intervention,
    find_pod,
    PointOfDivergence,
    calculate_narrative_physics,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.query_models import (
    DoEvent, DoTrait, DoBelief, DoConcern, DoProposition,
    InterventionQuery, CounterfactualQuery,
)


def _make_world() -> WorldStateV1:
    return WorldStateV1(
        locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={})},
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.2)},
                beliefs=[
                    Belief(
                        target_id="ENT_BOB",
                        perceived_state="Bob is loyal",
                        confidence=0.8,
                        evidence_strength="strong",
                        proposition_id="PROP_BOB_LOYAL",
                        inertia=0.3,
                    ),
                ],
                concerns=[
                    Concern(
                        concern_id="CCN_ALICE_FEARS_BETRAYAL",
                        proposition_id="PROP_BOB_LOYAL",
                        polarity="fear", kind="betrayal",
                        salience=0.7,
                        activation_fabula_window=[5, 100],
                    ),
                ],
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB", name="Bob", location_id="LOC_A",
                status="healthy",
                traits={"courage": TraitVector(value=0.7, inertia=0.3)},
            ),
        },
        events=[
            EventNode(id="EVT_MEET", fabula_time=1, syuzhet_index=1,
                      event_type="outcome", actor_ids=["ENT_ALICE", "ENT_BOB"],
                      description="Alice and Bob meet."),
            EventNode(id="EVT_BETRAYAL", fabula_time=10, syuzhet_index=10,
                      event_type="choice", actor_ids=["ENT_BOB"],
                      target_ids=["ENT_ALICE"], description="Bob betrays Alice"),
        ],
        causal_topology=[
            CausalEdge(source_id="EVT_MEET", target_id="EVT_BETRAYAL",
                       causality_type="chain_reaction", causal_force=3.0,
                       mechanism="social", evidence_strength="moderate",
                       fabula_time=1),
        ],
        spatial_topology=[],
        social_topology=[],
        propositions=[
            Proposition(
                proposition_id="PROP_BOB_LOYAL",
                description="Bob remains loyal to Alice",
                kind="trait_holds",
                referent_ids=["ENT_BOB", "ENT_ALICE"],
                truth_at_fabula={1: True, 10: False},
            ),
        ],
    )


# =====================================================================
# Phase 2 — Utility-layer preservation
# =====================================================================
class TestStampUtilityLayer:
    def test_stamp_writes_propositions_and_back_pointer(self):
        ws = _make_world()
        ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE", "ENT_BOB"])
        sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
        _stamp_utility_layer(sandbox, ws, fabula_anchor=10)
        assert sandbox.graph["parent_world_id"] == id(ws)
        assert sandbox.graph["fabula_anchor"] == 10
        props = sandbox.graph["propositions"]
        assert len(props) == 1
        assert props[0]["proposition_id"] == "PROP_BOB_LOYAL"

    def test_stamp_survives_node_link_round_trip(self):
        ws = _make_world()
        ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE", "ENT_BOB"])
        sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
        _stamp_utility_layer(sandbox, ws, fabula_anchor=10)
        data = nx.node_link_data(sandbox)
        recovered = _load_propositions_from_sandbox(data)
        assert len(recovered) == 1
        assert recovered[0]["proposition_id"] == "PROP_BOB_LOYAL"

    def test_load_from_empty_sandbox_returns_empty_list(self):
        assert _load_propositions_from_sandbox({}) == []
        assert _load_propositions_from_sandbox({"graph": {}}) == []


class TestRungBranchesStampSandbox:
    def test_intervention_branch_stamps_utility_layer(self):
        ws = _make_world()
        q = InterventionQuery(
            interventions={"ENT_ALICE.status": "injured"},
            original_query="test",
        )
        result = calculate_narrative_physics(q, ws, temporal_anchor=10)
        recovered = _load_propositions_from_sandbox(result["physics_state"])
        assert len(recovered) == 1
        assert recovered[0]["proposition_id"] == "PROP_BOB_LOYAL"

    def test_counterfactual_branch_stamps_utility_layer(self):
        ws = _make_world()
        q = CounterfactualQuery(
            historical_interventions={"EVT_BETRAYAL.event_type": "prevented"},
            evidence_node_ids=["EVT_BETRAYAL"],
            original_query="test",
        )
        result = calculate_narrative_physics(q, ws)
        recovered = _load_propositions_from_sandbox(result["physics_state"])
        assert len(recovered) == 1


# =====================================================================
# Phase 3 — apply_intervention
# =====================================================================
class TestApplyIntervention:
    def test_belief_target_via_engine(self):
        ws = _make_world()
        result = apply_intervention(
            [DoBelief(holder_id="ENT_ALICE", target_id="ENT_BOB",
                      proposition_id="PROP_BOB_LOYAL", confidence=0.0)],
            ws,
            fabula_anchor=10,
            use_causal_engine=True,
        )
        assert result["status"] == "success"
        assert len(result["belief_mutations"]) == 1
        bm = result["belief_mutations"][0]
        assert bm["holder_id"] == "ENT_ALICE"
        assert bm["new_confidence"] == 0.0

    def test_concern_target(self):
        ws = _make_world()
        result = apply_intervention(
            [DoConcern(holder_id="ENT_ALICE",
                       concern_id="CCN_ALICE_FEARS_BETRAYAL",
                       salience=0.0)],
            ws, fabula_anchor=10,
        )
        assert result["status"] == "success"
        assert any(m["field"] == "salience" for m in result["concern_mutations"])

    def test_proposition_target_with_cascade(self):
        ws = _make_world()
        result = apply_intervention(
            [DoProposition(proposition_id="PROP_BOB_LOYAL", truth=False,
                           fabula_time=10, propagate_to_beliefs=True)],
            ws, fabula_anchor=10,
        )
        assert result["status"] == "success"
        pm = result["proposition_mutations"][0]
        assert pm["new_truth"] is False
        assert pm["cascaded_belief_count"] >= 1

    def test_event_target_uses_legacy_path_via_engine(self):
        ws = _make_world()
        result = apply_intervention(
            [DoEvent(event_id="EVT_BETRAYAL", occurred=False)],
            ws, fabula_anchor=10,
        )
        assert result["status"] == "success"
        assert "EVT_BETRAYAL.event_type" in result["math_changes"]

    def test_mixed_targets(self):
        ws = _make_world()
        result = apply_intervention(
            [
                DoEvent(event_id="EVT_BETRAYAL", occurred=False),
                DoBelief(holder_id="ENT_ALICE", target_id="ENT_BOB",
                         proposition_id="PROP_BOB_LOYAL", confidence=0.1),
                DoConcern(holder_id="ENT_ALICE",
                          concern_id="CCN_ALICE_FEARS_BETRAYAL",
                          salience=0.0),
            ],
            ws, fabula_anchor=10,
        )
        assert result["status"] == "success"
        assert result["belief_mutations"]
        assert result["concern_mutations"]
        assert "EVT_BETRAYAL.event_type" in result["math_changes"]

    def test_empty_targets_implausible(self):
        ws = _make_world()
        result = apply_intervention([], ws)
        assert result["status"] == "implausible"

    def test_legacy_non_engine_path_event_only(self):
        ws = _make_world()
        result = apply_intervention(
            [DoEvent(event_id="EVT_BETRAYAL", occurred=False)],
            ws, use_causal_engine=False,
        )
        assert result["status"] == "success"
        # Sandbox round-trip carries the surgery
        g = nx.node_link_graph(result["physics_state"])
        assert g.nodes["EVT_BETRAYAL"]["event_type"] == "prevented"

    def test_result_carries_serialised_do_targets(self):
        ws = _make_world()
        result = apply_intervention(
            [DoConcern(holder_id="ENT_ALICE",
                       concern_id="CCN_ALICE_FEARS_BETRAYAL",
                       active=False)],
            ws, fabula_anchor=10,
        )
        assert len(result["do_targets"]) == 1
        assert result["do_targets"][0]["target_kind"] == "concern"


# =====================================================================
# Phase 3 — find_pod
# =====================================================================
class TestFindPod:
    def test_event_target_returns_event_fabula_time(self):
        ws = _make_world()
        pod = find_pod(DoEvent(event_id="EVT_BETRAYAL", occurred=False), ws)
        assert isinstance(pod, PointOfDivergence)
        assert pod.fabula_time == 10
        assert pod.target_kind == "event"
        assert "EVT_BETRAYAL" in pod.candidate_event_ids
        # Ancestor EVT_MEET should appear earlier in the candidate list
        assert pod.candidate_event_ids.index("EVT_MEET") < pod.candidate_event_ids.index("EVT_BETRAYAL")

    def test_event_target_unknown_raises(self):
        ws = _make_world()
        with pytest.raises(ValueError):
            find_pod(DoEvent(event_id="EVT_GHOST", occurred=False), ws)

    def test_proposition_target_anchors_at_latest_truth(self):
        ws = _make_world()
        pod = find_pod(
            DoProposition(proposition_id="PROP_BOB_LOYAL", truth=True),
            ws,
            fabula_anchor=10,
        )
        # truth_at_fabula has keys 1 and 10; horizon=10 picks 10.
        assert pod.fabula_time == 10
        assert pod.target_kind == "proposition"
        # Both events touch ENT_BOB / ENT_ALICE referents
        assert set(pod.candidate_event_ids) == {"EVT_MEET", "EVT_BETRAYAL"}

    def test_belief_target_anchors_at_latest_event_touching_target(self):
        ws = _make_world()
        pod = find_pod(
            DoBelief(holder_id="ENT_ALICE", target_id="ENT_BOB", confidence=0.0),
            ws,
        )
        # EVT_BETRAYAL touches ENT_BOB (actor); latest such event = ft 10.
        assert pod.fabula_time == 10
        assert pod.target_kind == "belief"

    def test_concern_target_anchors_at_window_start(self):
        ws = _make_world()
        pod = find_pod(
            DoConcern(holder_id="ENT_ALICE",
                      concern_id="CCN_ALICE_FEARS_BETRAYAL",
                      active=False),
            ws,
        )
        # Concern's activation_fabula_window=[5, 100] → anchor at 5.
        assert pod.fabula_time == 5
        assert pod.target_kind == "concern"

    def test_concern_target_unknown_raises(self):
        ws = _make_world()
        with pytest.raises(ValueError):
            find_pod(
                DoConcern(holder_id="ENT_ALICE", concern_id="CCN_GHOST"),
                ws,
            )

    def test_trait_target_anchors_at_earliest_holder_event(self):
        ws = _make_world()
        pod = find_pod(
            DoTrait(holder_id="ENT_ALICE", trait_name="courage", value=0.99),
            ws,
        )
        # EVT_MEET (ft=1) involves ENT_ALICE; that's the earliest.
        assert pod.fabula_time == 1
        assert pod.target_kind == "trait"

    def test_mutability_prior_flag_passes_through(self):
        ws = _make_world()
        pod = find_pod(
            DoEvent(event_id="EVT_BETRAYAL", occurred=False),
            ws,
            mutability_prior=True,
        )
        # Phase 5: flag activates the Kahneman-Miller × concern-load
        # re-ranking and is reported on the result.
        assert pod.mutability_prior_used is True

    def test_unsupported_target_raises(self):
        ws = _make_world()
        with pytest.raises(TypeError):
            find_pod("not a do target", ws)


# =====================================================================
# Phase 5 — Concern-mutability prior in find_pod
# =====================================================================
def _make_world_for_mutability() -> WorldStateV1:
    """Two ancestor events, only the *later* one touches a referent of
    a high-salience concern. Temporal ordering would put EVT_EARLY
    first; the mutability prior should re-rank EVT_LATE first.
    """
    return WorldStateV1(
        locations={"LOC_X": Location(
                id="LOC_X",
                name="X", description="X", ambient_state={})},
        objects={},
        entities={
            "ENT_HERO": Entity(
                id="ENT_HERO", name="Hero", location_id="LOC_X",
                status="healthy",
                traits={},
                concerns=[
                    Concern(
                        concern_id="CCN_HERO_FEARS_VILLAIN_RISES",
                        proposition_id="PROP_VILLAIN_RISES",
                        polarity="fear", kind="rise_to_power",
                        salience=0.9,
                        activation_fabula_window=[0, 100],
                    ),
                ],
            ),
            "ENT_VILLAIN": Entity(
                id="ENT_VILLAIN", name="Villain", location_id="LOC_X",
                status="healthy",
                traits={},
            ),
            "ENT_BYSTANDER": Entity(
                id="ENT_BYSTANDER", name="Bystander", location_id="LOC_X",
                status="healthy",
                traits={},
            ),
        },
        events=[
            # Earliest event — touches no concern referent.
            EventNode(id="EVT_EARLY", fabula_time=1, syuzhet_index=1,
                      event_type="outcome", actor_ids=["ENT_BYSTANDER"],
                      description="Bystander does something irrelevant."),
            # Later event — touches the concern's proposition referent.
            EventNode(id="EVT_LATE", fabula_time=5, syuzhet_index=5,
                      event_type="choice", actor_ids=["ENT_VILLAIN"],
                      description="Villain takes a step toward power."),
            # Target event.
            EventNode(id="EVT_CORONATION", fabula_time=10, syuzhet_index=10,
                      event_type="outcome", actor_ids=["ENT_VILLAIN"],
                      description="Villain crowned."),
        ],
        causal_topology=[
            CausalEdge(source_id="EVT_EARLY", target_id="EVT_CORONATION",
                       causality_type="chain_reaction", causal_force=1.0,
                       mechanism="social", evidence_strength="weak",
                       fabula_time=1),
            CausalEdge(source_id="EVT_LATE", target_id="EVT_CORONATION",
                       causality_type="chain_reaction", causal_force=3.0,
                       mechanism="social", evidence_strength="strong",
                       fabula_time=5),
        ],
        spatial_topology=[],
        social_topology=[],
        propositions=[
            Proposition(
                proposition_id="PROP_VILLAIN_RISES",
                description="The villain rises to power.",
                kind="event_occurs",
                referent_ids=["ENT_VILLAIN"],
                truth_at_fabula={10: True},
            ),
        ],
    )


class TestMutabilityPrior:
    def test_temporal_default_orders_earliest_first(self):
        ws = _make_world_for_mutability()
        pod = find_pod(
            DoEvent(event_id="EVT_CORONATION", occurred=False),
            ws,
            mutability_prior=False,
        )
        assert pod.mutability_prior_used is False
        # Earliest ancestor first.
        assert pod.candidate_event_ids[0] == "EVT_EARLY"
        assert pod.candidate_event_ids[1] == "EVT_LATE"

    def test_mutability_prior_reranks_concern_relevant_event_first(self):
        ws = _make_world_for_mutability()
        pod = find_pod(
            DoEvent(event_id="EVT_CORONATION", occurred=False),
            ws,
            mutability_prior=True,
        )
        assert pod.mutability_prior_used is True
        # EVT_LATE touches ENT_VILLAIN (a referent of PROP_VILLAIN_RISES,
        # which carries a high-salience concern); it should out-rank the
        # temporally-earlier but concern-irrelevant EVT_EARLY.
        assert pod.candidate_event_ids[0] == "EVT_LATE"

    def test_mutability_score_zero_when_no_concern_match(self):
        from shadow_loom.narrative_physics import _mutability_score
        ws = _make_world_for_mutability()
        early = next(e for e in ws.events if e.id == "EVT_EARLY")
        assert _mutability_score(early, ws) == 0.0

    def test_mutability_score_positive_when_concern_match(self):
        from shadow_loom.narrative_physics import _mutability_score
        ws = _make_world_for_mutability()
        late = next(e for e in ws.events if e.id == "EVT_LATE")
        # 0.9 salience × (1 - 0.5 typicality inside window) = 0.45
        assert _mutability_score(late, ws) == pytest.approx(0.45)

    def test_mutability_prior_stable_when_no_concerns(self):
        ws = _make_world_for_mutability()
        # Strip concerns — fall back to temporal ordering.
        ws.entities["ENT_HERO"].concerns = []
        pod = find_pod(
            DoEvent(event_id="EVT_CORONATION", occurred=False),
            ws,
            mutability_prior=True,
        )
        assert pod.mutability_prior_used is True
        assert pod.candidate_event_ids[0] == "EVT_EARLY"
