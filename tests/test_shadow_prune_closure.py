# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the counterfactual / intervention prune-closure cascade.

When a Rung-2 or Rung-3 do-surgery on a shadow branch renders an event
epistemically inert (``event_type='prevented'`` /
``truth_value='false'``), the persisted shadow VersionRow's
``world_state_json`` must reflect a world where:

* the pruned event no longer exists, and
* causally-downstream events whose every ``chain_reaction`` parent has
  been pruned also no longer exist (Pearl's disjunctive
  structural-equation reading; over-determined effects with a surviving
  sufficient cause must persist).

Before this fix, the prune set was only used to invalidate beliefs in
the sandbox — the events themselves stayed in the merged snapshot, so
a follow-up interrogation reading the JSON saw them as factual and
contradicted the prose the renderer just produced (e.g. Mrs Coady's
heart-attack death event persisting after the dog-killing that caused
it was intervened away in "A Fish Called Wanda").

Strict gating: Rung-1 paths (observation, continuation, affective)
must never prune ancestors — new events from those queries join the
parent world additively. Factual merges must also never prune
ancestors regardless of query type.
"""

from __future__ import annotations

from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    CausalEdge,
    EventNode,
    WorldStateV1,
)
from shadow_loom.pipeline import (
    _augment_topology_with_sandbox_deltas,
    _compute_shadow_prune_closure,
)


# ---------------------------------------------------------------------
# Closure helper — Pearl's disjunctive chain_reaction rule
# ---------------------------------------------------------------------


def _ws_with_chain(events: list[EventNode], edges: list[CausalEdge]) -> WorldStateV1:
    return WorldStateV1(
        entities={}, events=events,
        causal_topology=edges,
        spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[], channels={},
    )


def _chain(src: str, tgt: str) -> CausalEdge:
    return CausalEdge(
        source_id=src, target_id=tgt,
        causality_type="chain_reaction",
        mechanism="physical",
        fabula_time=0,
    )


def _evt(eid: str) -> EventNode:
    return EventNode(
        id=eid, fabula_time=0, syuzhet_index=0,
        event_type="outcome", actor_ids=[], target_ids=[],
        description=eid,
    )


def test_closure_empty_roots_returns_empty():
    ws = _ws_with_chain([], [])
    assert _compute_shadow_prune_closure(ws, set()) == set()


def test_closure_single_link_propagates():
    """A -> B; prune A => closure {A, B}."""
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B")],
        [_chain("EVT_A", "EVT_B")],
    )
    closure = _compute_shadow_prune_closure(ws, {"EVT_A"})
    assert closure == {"EVT_A", "EVT_B"}


def test_closure_disjunctive_survivor_blocks_prune():
    """A -> C, B -> C; prune only A => closure {A} (C survives because B
    is a surviving sufficient cause). This is the Halpern-Pearl
    over-determination test — Mrs Coady can still die of pneumonia
    even if the dog-killing is intervened away."""
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B"), _evt("EVT_C")],
        [_chain("EVT_A", "EVT_C"), _chain("EVT_B", "EVT_C")],
    )
    closure = _compute_shadow_prune_closure(ws, {"EVT_A"})
    assert closure == {"EVT_A"}, (
        "EVT_C must survive: EVT_B is a non-pruned sufficient cause"
    )


def test_closure_disjunctive_all_parents_pruned_propagates():
    """A -> C, B -> C; prune both A and B => closure {A, B, C}."""
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B"), _evt("EVT_C")],
        [_chain("EVT_A", "EVT_C"), _chain("EVT_B", "EVT_C")],
    )
    closure = _compute_shadow_prune_closure(ws, {"EVT_A", "EVT_B"})
    assert closure == {"EVT_A", "EVT_B", "EVT_C"}


def test_closure_transitive_chain_to_fixpoint():
    """A -> B -> C -> D; prune A => closure {A, B, C, D}."""
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B"), _evt("EVT_C"), _evt("EVT_D")],
        [
            _chain("EVT_A", "EVT_B"),
            _chain("EVT_B", "EVT_C"),
            _chain("EVT_C", "EVT_D"),
        ],
    )
    closure = _compute_shadow_prune_closure(ws, {"EVT_A"})
    assert closure == {"EVT_A", "EVT_B", "EVT_C", "EVT_D"}


def test_closure_ignores_non_chain_reaction_edges():
    """Only ``chain_reaction`` (Event→Event direct cause) edges count
    as sufficient causes for prune propagation. An ``affordance_gate``
    (State→Event modifier) from a state into EVT_B must NOT make
    pruning of an unrelated event cascade to EVT_B."""
    edges = [
        CausalEdge(
            source_id="ENT_X__location_id", target_id="EVT_B",
            causality_type="affordance_gate", mechanism="physical",
            fabula_time=0,
        ),
    ]
    ws = _ws_with_chain([_evt("EVT_A"), _evt("EVT_B")], edges)
    # Prune EVT_A; EVT_B has no chain_reaction parents at all, so it
    # cannot be added to the closure regardless of affordance gates.
    closure = _compute_shadow_prune_closure(ws, {"EVT_A"})
    assert closure == {"EVT_A"}


def test_closure_exogenous_event_never_pruned_via_closure():
    """An event with no chain_reaction incoming edges is exogenous and
    cannot be added to the closure (only the explicit root set
    contains it)."""
    ws = _ws_with_chain(
        [_evt("EVT_ROOT"), _evt("EVT_OTHER")],
        [],
    )
    closure = _compute_shadow_prune_closure(ws, {"EVT_ROOT"})
    assert closure == {"EVT_ROOT"}


# ---------------------------------------------------------------------
# _augment_topology_with_sandbox_deltas — gating + wiring
# ---------------------------------------------------------------------


def _bridge(world_state, physics_result, *, world_id, query_type):
    topology = ChunkTopology()
    _augment_topology_with_sandbox_deltas(
        topology,
        world_state=world_state,
        physics_result=physics_result,
        fabula_time_now=100,
        world_id=world_id,
        query_type=query_type,
    )
    return topology


def test_bridge_populates_suppression_for_counterfactual_shadow():
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B")],
        [_chain("EVT_A", "EVT_B")],
    )
    physics = {"pruned_utterance_event_ids": ["EVT_A"]}
    topo = _bridge(ws, physics, world_id="shadow", query_type="counterfactual")
    assert set(topo.suppressed_event_ids) == {"EVT_A", "EVT_B"}


def test_bridge_populates_suppression_for_intervention_shadow():
    """Rung-2 intervention on a shadow branch must trigger the same
    cascade as Rung-3 counterfactual."""
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B"), _evt("EVT_C")],
        [_chain("EVT_A", "EVT_B"), _chain("EVT_B", "EVT_C")],
    )
    physics = {"pruned_utterance_event_ids": ["EVT_A"]}
    topo = _bridge(ws, physics, world_id="shadow", query_type="intervention")
    assert set(topo.suppressed_event_ids) == {"EVT_A", "EVT_B", "EVT_C"}


def test_bridge_no_suppression_for_rung1_observation_even_with_prune_set():
    """Rung-1 observation must never prune ancestors. Even a stray
    non-empty ``pruned_utterance_event_ids`` from a misconfigured
    upstream path is ignored."""
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B")],
        [_chain("EVT_A", "EVT_B")],
    )
    physics = {"pruned_utterance_event_ids": ["EVT_A"]}
    topo = _bridge(ws, physics, world_id="shadow", query_type="observation")
    assert topo.suppressed_event_ids == []


def test_bridge_no_suppression_for_continuation():
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B")],
        [_chain("EVT_A", "EVT_B")],
    )
    physics = {"pruned_utterance_event_ids": ["EVT_A"]}
    # ``continue`` isn't a query_type literal — observation/general
    # cover the Rung-1 path. Verify neither triggers pruning.
    for qt in ("general", "interrogate", "directive", "manual_edit"):
        topo = _bridge(ws, physics, world_id="shadow", query_type=qt)
        assert topo.suppressed_event_ids == [], (
            f"query_type={qt!r} must not trigger prune cascade"
        )


def test_bridge_no_suppression_on_factual_branch_even_for_counterfactual():
    """Defensive: a factual merge must never prune ancestor events,
    even if (somehow) a counterfactual query routed there with a
    populated prune set. The shadow tag is the load-bearing guard."""
    ws = _ws_with_chain(
        [_evt("EVT_A"), _evt("EVT_B")],
        [_chain("EVT_A", "EVT_B")],
    )
    physics = {"pruned_utterance_event_ids": ["EVT_A"]}
    topo = _bridge(ws, physics, world_id="factual", query_type="counterfactual")
    assert topo.suppressed_event_ids == []


def test_bridge_empty_prune_set_no_suppression():
    ws = _ws_with_chain(
        [_evt("EVT_A")], [],
    )
    topo = _bridge(
        ws, {"pruned_utterance_event_ids": []},
        world_id="shadow", query_type="counterfactual",
    )
    assert topo.suppressed_event_ids == []
