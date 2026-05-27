# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shadow-merge ``RelationshipEdge`` value rollback (Gap #1).

Closes the cross-branch leak where affective interrogations on a
counterfactual branch returned the post-mutation factual metric
value despite the mutating event having been intervened away.

When a ``mutation_social`` causal edge's source event is in
``ChunkTopology.suppressed_event_ids``, the merged ``WorldStateV1``
must materialise a shadow-tagged clone of the matching
``RelationshipEdge`` in ``shadow_social_topology[branch_label]``
with the suppressed ``trait_delta`` subtracted from the metric
value and ``evidence_strength`` demoted to ``weak``. The factual
``social_topology`` entry must be left untouched so factual reads
are unaffected.
"""

from __future__ import annotations

import pytest

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    RelationshipEdge,
    RelationshipMetric,
    WorldStateV1,
)


def _ws_with_social_mutation(
    *,
    initial_affinity: float,
    delta: float,
) -> WorldStateV1:
    """Two-entity world with one mutation_social CausalEdge bumping
    Otto→Wanda affinity by ``delta`` from ``initial_affinity``
    (which is the *post-mutation* value already accumulated on the
    RelationshipEdge — i.e. the factual final state)."""
    otto = Entity(
        id="ENT_OTTO", name="Otto",
        location_id="LOC_X", status="healthy", traits={},
        world_id="factual",
    )
    wanda = Entity(
        id="ENT_WANDA", name="Wanda",
        location_id="LOC_X", status="healthy", traits={},
        world_id="factual",
    )
    betrayal = EventNode(
        id="EVT_WANDA_BETRAYS_OTTO",
        fabula_time=1000, syuzhet_index=10,
        event_type="outcome",
        actor_ids=["ENT_WANDA"], target_ids=["ENT_OTTO"],
        description="Wanda double-crosses Otto.",
        world_id="factual",
    )
    edge = RelationshipEdge(
        source_entity_id="ENT_OTTO",
        target_entity_id="ENT_WANDA",
        metrics={
            "affinity": RelationshipMetric(
                value=initial_affinity,
                inertia=0.3,
                evidence_strength="strong",
                last_updated_fabula=1000,
                observed=True,
            ),
        },
        world_id="factual",
    )
    mutation = CausalEdge(
        source_id="EVT_WANDA_BETRAYS_OTTO",
        target_id="ENT_OTTO",
        rel_counterpart_id="ENT_WANDA",
        causality_type="mutation_social",
        trait_target="affinity",
        trait_delta=delta,
        fabula_time=1000,
        confidence=0.9,
        mechanism="Wanda's double-cross causes Otto's affinity to collapse.",
        world_id="factual",
    )
    return WorldStateV1(
        entities={"ENT_OTTO": otto, "ENT_WANDA": wanda},
        events=[betrayal],
        causal_topology=[mutation],
        spatial_topology=[],
        social_topology=[edge],
        locations={}, objects={}, world_traits={},
        propositions=[], channels={},
    )


def _get_affinity(ws: WorldStateV1, src: str, tgt: str) -> float:
    for e in ws.social_topology:
        if e.source_entity_id == src and e.target_entity_id == tgt:
            m = e.metrics.get("affinity")
            assert m is not None
            return float(m.value)
    raise AssertionError(f"no edge {src}->{tgt}")


def test_shadow_suppression_rolls_back_social_metric_value():
    """Suppressing the event behind a mutation_social edge must
    clone the dyad into ``shadow_social_topology`` and subtract
    the ``trait_delta`` from the affected metric value."""
    ws = _ws_with_social_mutation(initial_affinity=-0.5, delta=-0.8)
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_WANDA_BETRAYS_OTTO"],
    )
    vwm2 = vwm.merge(
        topology, world_id="shadow", branch_label="cf_no_betrayal",
    )
    projected = vwm2.current.projected_for_branch(
        branch_world_id="shadow", branch_label="cf_no_betrayal",
    )
    # Rolled back: -0.5 - (-0.8) = +0.3
    assert _get_affinity(projected, "ENT_OTTO", "ENT_WANDA") == pytest.approx(0.3)
    # Demoted evidence on the shadow clone.
    shadow_edge = next(
        e for e in projected.social_topology
        if (e.source_entity_id, e.target_entity_id)
        == ("ENT_OTTO", "ENT_WANDA")
    )
    assert shadow_edge.metrics["affinity"].evidence_strength == "weak"
    assert shadow_edge.world_id == "shadow"


def test_shadow_suppression_leaves_factual_social_topology_untouched():
    """The factual ``social_topology`` entry retains its
    post-mutation value and evidence_strength after a shadow
    merge \u2014 rows are isolated."""
    ws = _ws_with_social_mutation(initial_affinity=-0.5, delta=-0.8)
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_WANDA_BETRAYS_OTTO"],
    )
    vwm.merge(topology, world_id="shadow", branch_label="cf")
    factual_edge = next(
        e for e in vwm.current.social_topology
        if (e.source_entity_id, e.target_entity_id)
        == ("ENT_OTTO", "ENT_WANDA")
    )
    assert factual_edge.metrics["affinity"].value == -0.5
    assert factual_edge.metrics["affinity"].evidence_strength == "strong"
    assert factual_edge.world_id == "factual"


def test_shadow_rollback_clamps_to_valid_range():
    """Rollback that would push the metric outside [-1, 1] is
    clamped to the bound."""
    # Factual value at the floor; rolling back a negative delta
    # would push above the +1 ceiling.
    ws = _ws_with_social_mutation(initial_affinity=0.7, delta=-0.6)
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_WANDA_BETRAYS_OTTO"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    projected = vwm2.current.projected_for_branch(
        branch_world_id="shadow", branch_label="cf",
    )
    # 0.7 - (-0.6) = 1.3 → clamped to 1.0
    assert _get_affinity(projected, "ENT_OTTO", "ENT_WANDA") == 1.0


def test_factual_merge_does_not_create_shadow_social_clone():
    """A factual merge with no suppression must not populate the
    ``shadow_social_topology`` sidecar."""
    ws = _ws_with_social_mutation(initial_affinity=-0.5, delta=-0.8)
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(chunk_id="CHK_F")
    vwm2 = vwm.merge(topology, world_id="factual")
    assert vwm2.current.shadow_social_topology == {}
