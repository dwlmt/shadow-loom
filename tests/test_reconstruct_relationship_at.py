"""Gap #2 — RelationshipMetric fabula-time reconstruction.

Mirrors the per-entity / per-trait / per-proposition reconstruction
surface so AMWN sandboxes, ego graphs, and the auditor can read a
dyad's social metric values at any tick in the fabula, not only the
*current* terminal values.

The reconstruction is *delta-replay backwards*: start from the edge's
current per-axis values and undo every ``mutation_social`` causal edge
whose driving event fires after the requested tick. The result is
clamped per axis (affinity, power_dynamic in [-1, 1]; fear in [0, 1]).
"""
from __future__ import annotations

import pytest

from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    RelationshipEdge,
    RelationshipMetric,
    reconstruct_relationship_at,
)


def _edge(affinity_now: float, fear_now: float = 0.5) -> RelationshipEdge:
    return RelationshipEdge(
        source_entity_id="ENT_OTTO",
        target_entity_id="ENT_WANDA",
        metrics={
            "affinity": RelationshipMetric(value=affinity_now, last_updated_fabula=20000),
            "fear": RelationshipMetric(value=fear_now, last_updated_fabula=20000),
            "power_dynamic": RelationshipMetric(value=0.2, last_updated_fabula=20000),
        },
        world_id="factual",
    )


def _events() -> list[EventNode]:
    return [
        EventNode(
            id="EVT_BETRAYAL",
            description="Wanda betrays Otto",
            fabula_time=15000,
            location_id="LOC_FLAT",
            syuzhet_index=0,
            event_type="outcome",
        ),
        EventNode(
            id="EVT_RECONCILE",
            description="They make up",
            fabula_time=18000,
            location_id="LOC_FLAT",
            syuzhet_index=0,
            event_type="outcome",
        ),
    ]


def _causal_edges() -> list[CausalEdge]:
    return [
        CausalEdge(
            source_id="EVT_BETRAYAL",
            target_id="ENT_OTTO",
            rel_counterpart_id="ENT_WANDA",
            causality_type="mutation_social",
            trait_target="affinity",
            trait_delta=-0.6,
            mechanism="betrayal triggers loss of affinity",
            fabula_time=15000,
        ),
        CausalEdge(
            source_id="EVT_BETRAYAL",
            target_id="ENT_OTTO",
            rel_counterpart_id="ENT_WANDA",
            causality_type="mutation_social",
            trait_target="fear",
            trait_delta=0.3,
            mechanism="betrayal spikes fear",
            fabula_time=15000,
        ),
        CausalEdge(
            source_id="EVT_RECONCILE",
            target_id="ENT_OTTO",
            rel_counterpart_id="ENT_WANDA",
            causality_type="mutation_social",
            trait_target="affinity",
            trait_delta=0.4,
            mechanism="reconciliation restores some affinity",
            fabula_time=18000,
        ),
    ]


def test_reconstruct_after_all_mutations_returns_current_values():
    edge = _edge(affinity_now=0.3, fear_now=0.6)
    state = reconstruct_relationship_at(
        edge, 20000, causal_edges=_causal_edges(), events=_events(),
    )
    assert state["affinity"] == pytest.approx(0.3)
    assert state["fear"] == pytest.approx(0.6)


def test_reconstruct_between_mutations_rolls_back_future_only():
    # current affinity = 0.3. At fabula=17000 we are after BETRAYAL
    # (15000) but before RECONCILE (18000) — undo the +0.4 only.
    edge = _edge(affinity_now=0.3)
    state = reconstruct_relationship_at(
        edge, 17000, causal_edges=_causal_edges(), events=_events(),
    )
    assert state["affinity"] == pytest.approx(-0.1)


def test_reconstruct_before_all_mutations_rolls_back_everything():
    edge = _edge(affinity_now=0.3, fear_now=0.6)
    state = reconstruct_relationship_at(
        edge, 1000, causal_edges=_causal_edges(), events=_events(),
    )
    # affinity: 0.3 - 0.4 - (-0.6) = 0.5
    assert state["affinity"] == pytest.approx(0.5)
    # fear: 0.6 - 0.3 = 0.3
    assert state["fear"] == pytest.approx(0.3)


def test_reconstruct_clamps_to_axis_ranges():
    # Start affinity at 0.9, undo a -0.6 → would be 1.5; clamp to 1.0.
    edge = _edge(affinity_now=0.9)
    state = reconstruct_relationship_at(
        edge, 1000,
        causal_edges=[
            CausalEdge(
                source_id="EVT_BETRAYAL",
                target_id="ENT_OTTO",
                rel_counterpart_id="ENT_WANDA",
                causality_type="mutation_social",
                trait_target="affinity",
                trait_delta=-0.6,
                mechanism="betrayal",
                fabula_time=15000,
            ),
        ],
        events=_events(),
    )
    assert state["affinity"] == pytest.approx(1.0)


def test_reconstruct_filters_shadow_causal_edges_on_factual_holder():
    edge = _edge(affinity_now=0.3)
    shadow_edge = CausalEdge(
        source_id="EVT_BETRAYAL",
        target_id="ENT_OTTO",
        rel_counterpart_id="ENT_WANDA",
        causality_type="mutation_social",
        trait_target="affinity",
        trait_delta=-0.6,
        mechanism="betrayal",
                fabula_time=15000,
        world_id="shadow",
    )
    state = reconstruct_relationship_at(
        edge, 1000, causal_edges=[shadow_edge], events=_events(),
    )
    # Shadow edge is filtered out — current value is returned.
    assert state["affinity"] == pytest.approx(0.3)
