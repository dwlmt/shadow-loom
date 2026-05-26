# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the trait-clamp bugs found 2026-05-26.

Two related defects shipped together because both clamps were
copy-pasted from the relationship-axis bound (``[-1, 1]``) onto
entity-trait writes whose schema requires ``[0, 1]``:

1. ``shadow_loom.pipeline._upsert_entity_trait`` clamped to
   ``[-1.0, 1.0]`` then constructed ``TraitVector(value=...)``.
   When ``_augment_topology_with_sandbox_deltas`` bridged a Rung-3
   hidden_delta large enough to drive ``base + delta_f`` negative,
   ``TraitVector``'s ``ge=0`` field validator raised and crashed
   the whole re-extraction merge.
2. ``shadow_loom.projections.reconstruct_entity_at_causal`` clamped
   accumulated trait values to ``[-1.0, 1.0]`` and rendered a dict
   straight into the MCP ``inspect()`` payload. No crash (the dict
   is never re-wrapped into a TraitVector) but the inspector
   silently produced incoherent negative trait readings.
"""

from __future__ import annotations

from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    Location,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.pipeline import _augment_topology_with_sandbox_deltas
from shadow_loom.projections import reconstruct_entity_at_causal


def _ws_with_low_trait(value: float = 0.05) -> WorldStateV1:
    return WorldStateV1(
        locations={
            "LOC_X": Location(
                id="LOC_X", name="X", description="d", ambient_state={},
            ),
        },
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_X", status="healthy",
                traits={
                    "courage": TraitVector(value=value, inertia=0.3),
                },
            ),
        },
        events=[],
        causal_topology=[],
        world_traits={},
    )


class TestPipelineEntityTraitClamp:
    """``_upsert_entity_trait`` must clamp the lower bound to 0.0 so a
    bridged delta that drives the running value below zero produces
    ``value=0.0`` instead of raising
    ``pydantic_core._pydantic_core.ValidationError``.
    """

    def test_negative_bridged_delta_does_not_crash(self):
        ws = _ws_with_low_trait(value=0.05)
        topo = ChunkTopology()
        # Hidden-delta lane: base 0.05 + (-0.5) = -0.45 (negative).
        # Pre-fix, this crashed inside TraitVector(value=-0.45) with a
        # ValidationError on ``ge=0``.
        _augment_topology_with_sandbox_deltas(
            topo,
            world_state=ws,
            physics_result={
                "hidden_deltas": {"ENT_A": {"courage": -0.5}},
            },
            fabula_time_now=100,
            fabula_time_historical=50,
        )
        # The bridge produced an EntityUpdate clamped to 0.0.
        eu = next(
            eu for eu in topo.entity_updates if eu.entity_id == "ENT_A"
        )
        assert eu.trait_updates["courage"].value == 0.0

    def test_above_one_mutation_clamped_to_one(self):
        """The upper bound must also be enforced (mutation lane)."""
        ws = _ws_with_low_trait(value=0.5)
        topo = ChunkTopology()
        _augment_topology_with_sandbox_deltas(
            topo,
            world_state=ws,
            physics_result={
                "mutations": [
                    {"node_id": "ENT_A", "trait": "courage", "new_value": 1.7},
                ],
            },
            fabula_time_now=100,
        )
        eu = next(
            eu for eu in topo.entity_updates if eu.entity_id == "ENT_A"
        )
        assert eu.trait_updates["courage"].value == 1.0


def _ws_for_projection() -> WorldStateV1:
    """Entity with NO pre-seeded ``courage`` trait so the projection's
    snapshot-replay step does not overwrite the causal running value
    we are exercising."""
    return WorldStateV1(
        locations={
            "LOC_X": Location(
                id="LOC_X", name="X", description="d", ambient_state={},
            ),
        },
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_X", status="healthy",
                traits={},
            ),
        },
        events=[
            EventNode(
                id="EVT_E1", fabula_time=10, syuzhet_index=10,
                event_type="outcome", actor_ids=["ENT_A"],
                description="e1", at_location_id="LOC_X",
            ),
            EventNode(
                id="EVT_E2", fabula_time=15, syuzhet_index=15,
                event_type="outcome", actor_ids=["ENT_A"],
                description="e2", at_location_id="LOC_X",
            ),
        ],
        causal_topology=[],
        world_traits={},
    )


class TestProjectionEntityTraitClamp:
    """``reconstruct_entity_at_causal`` must clamp accumulated trait
    values to ``[0, 1]`` so MCP ``inspect()`` callers never see
    negative trait readings.
    """

    def test_large_negative_delta_floor_at_zero(self):
        ws = _ws_for_projection()
        # Two negative mutation edges totaling -1.5 from an unseeded
        # baseline of 0.0. Pre-fix this clamped to -1.0; post-fix it
        # clamps to 0.0.
        for evt_id, delta in (("EVT_E1", -0.8), ("EVT_E2", -0.7)):
            ws.causal_topology.append(
                CausalEdge(
                    source_id=evt_id,
                    target_id="ENT_A",
                    causality_type="mutation",
                    fabula_time=int(evt_id[-1]) * 5 + 5,
                    trait_target="courage",
                    trait_delta=delta,
                    causal_force=1.0,
                    evidence_strength="moderate",
                    mechanism="psychological",
                )
            )
        snap = reconstruct_entity_at_causal(ws, "ENT_A", fabula_time=20)
        assert snap is not None
        courage = snap["traits"]["courage"]["value"]
        assert courage == 0.0

    def test_large_positive_delta_ceiled_at_one(self):
        ws = _ws_for_projection()
        for evt_id, delta in (("EVT_E1", 0.8), ("EVT_E2", 0.7)):
            ws.causal_topology.append(
                CausalEdge(
                    source_id=evt_id,
                    target_id="ENT_A",
                    causality_type="mutation",
                    fabula_time=int(evt_id[-1]) * 5 + 5,
                    trait_target="courage",
                    trait_delta=delta,
                    causal_force=1.0,
                    evidence_strength="moderate",
                    mechanism="psychological",
                )
            )
        snap = reconstruct_entity_at_causal(ws, "ENT_A", fabula_time=20)
        assert snap is not None
        courage = snap["traits"]["courage"]["value"]
        assert courage == 1.0


class TestProjectionCausalAccumulationSurvivesSnapshotReplay:
    """``reconstruct_entity_at_causal`` previously called
    :func:`reconstruct_entity_at` and unconditionally copied every trait
    from the resulting dict back into ``running`` — but that dict is
    seeded from ``ent.traits`` regardless of whether any
    ``state_timeline`` snapshot actually wrote the trait. The effect:
    every causal mutation accumulated above was silently discarded for
    any pre-seeded trait. Fix: only overwrite trait names that a
    snapshot actually touched.
    """

    def test_seeded_trait_keeps_causal_accumulation_with_no_snapshots(self):
        ws = _ws_with_low_trait(value=0.2)
        ws.events.append(EventNode(
            id="EVT_KICK", fabula_time=10, syuzhet_index=10,
            event_type="outcome", actor_ids=["ENT_A"],
            description="kick", at_location_id="LOC_X",
        ))
        ws.causal_topology.append(
            CausalEdge(
                source_id="EVT_KICK",
                target_id="ENT_A",
                causality_type="mutation",
                fabula_time=10,
                trait_target="courage",
                trait_delta=0.3,
                causal_force=1.0,
                evidence_strength="moderate",
                mechanism="psychological",
            )
        )
        snap = reconstruct_entity_at_causal(ws, "ENT_A", fabula_time=20)
        courage = snap["traits"]["courage"]["value"]
        # 0.2 baseline + 0.3 causal mutation = 0.5. Pre-fix this was
        # silently stomped back to 0.2 by the snapshot replay's
        # baseline copy.
        assert courage == 0.5

    def test_snapshot_at_tick_still_authoritative_when_present(self):
        """When a real ``EntityStateSnapshot`` writes the trait, its
        value must win — the causal accumulation is only used to fill
        gaps the timeline doesn't cover.
        """
        from shadow_loom.models import EntityStateSnapshot
        ws = _ws_with_low_trait(value=0.2)
        ws.events.append(EventNode(
            id="EVT_KICK", fabula_time=10, syuzhet_index=10,
            event_type="outcome", actor_ids=["ENT_A"],
            description="kick", at_location_id="LOC_X",
        ))
        ws.causal_topology.append(
            CausalEdge(
                source_id="EVT_KICK",
                target_id="ENT_A",
                causality_type="mutation",
                fabula_time=10,
                trait_target="courage",
                trait_delta=0.3,
                causal_force=1.0,
                evidence_strength="moderate",
                mechanism="psychological",
            )
        )
        ws.entities["ENT_A"].state_timeline.append(
            EntityStateSnapshot(
                fabula_time=10,
                triggered_by="EVT_KICK",
                traits={
                    "courage": TraitVector(value=0.85, inertia=0.3),
                },
            )
        )
        snap = reconstruct_entity_at_causal(ws, "ENT_A", fabula_time=20)
        # Authored snapshot wins.
        assert snap["traits"]["courage"]["value"] == 0.85
