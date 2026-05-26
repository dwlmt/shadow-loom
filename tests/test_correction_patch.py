# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for fourth-pass audit hardening of the correction-patch pipeline.

Covers small but load-bearing helpers introduced in the audit:

* ``_next_fact_index`` — collision-free numeric suffix generator.
* ``_snapshot_sort_key`` — deterministic tie-breaker for snapshot sort.
* ``_is_correction_regression`` — channel / world-trait loss guards.
* ``_apply_world_state_patch`` — channel_renames forwards
  ``via_channel_id`` references on EventNodes.
"""
from __future__ import annotations

import pytest

from shadow_loom.ingestion import (
    _apply_world_state_patch,
    _is_correction_regression,
    _next_fact_index,
    _snapshot_sort_key,
    WorldStatePatch,
)
from shadow_loom.models import (
    Channel,
    Entity,
    EntityStateSnapshot,
    EventNode,
    GlobalTrait,
    Location,
    TraitVector,
    WorldStateV1,
    WorldTraitSnapshot,
)
from shadow_loom.research import WorldFact


# ----------------------------------------------------------------------
# _next_fact_index
# ----------------------------------------------------------------------


class TestNextFactIndex:
    def test_empty_list_starts_at_one(self):
        assert _next_fact_index([]) == 1

    def test_dense_sequence(self):
        facts = [
            WorldFact(id="FACT_1", topic="t", summary="s"),
            WorldFact(id="FACT_2", topic="t", summary="s"),
        ]
        assert _next_fact_index(facts) == 3

    def test_sparse_sequence_uses_max_plus_one_not_len_plus_one(self):
        # FACT_5 exists but slots 2-4 are missing. len+1 would emit
        # FACT_3 and silently overwrite a hand-edited fact slot.
        facts = [
            WorldFact(id="FACT_1", topic="t", summary="s"),
            WorldFact(id="FACT_5", topic="t", summary="s"),
        ]
        assert _next_fact_index(facts) == 6

    def test_ignores_non_matching_ids(self):
        facts = [
            WorldFact(id="FACT_GIBBERISH", topic="t", summary="s"),
            WorldFact(id="FACT_2", topic="t", summary="s"),
        ]
        assert _next_fact_index(facts) == 3


# ----------------------------------------------------------------------
# _snapshot_sort_key
# ----------------------------------------------------------------------


class TestSnapshotSortKey:
    def test_primary_is_fabula_time(self):
        a = EntityStateSnapshot(fabula_time=10)
        b = EntityStateSnapshot(fabula_time=5)
        assert sorted([a, b], key=_snapshot_sort_key) == [b, a]

    def test_tie_breaks_on_triggered_by(self):
        # Same fabula_time → triggered_by sort applies, deterministically.
        a = EntityStateSnapshot(fabula_time=100, triggered_by="EVT_Z")
        b = EntityStateSnapshot(fabula_time=100, triggered_by="EVT_A")
        assert sorted([a, b], key=_snapshot_sort_key) == [b, a]
        # Reverse insertion order produces the same output.
        assert sorted([b, a], key=_snapshot_sort_key) == [b, a]

    def test_works_for_world_trait_snapshot_shape(self):
        # WorldTraitSnapshot has fewer fields than EntityStateSnapshot;
        # the key must not raise AttributeError.
        a = WorldTraitSnapshot(fabula_time=10, triggered_by="EVT_X")
        b = WorldTraitSnapshot(fabula_time=5, triggered_by="EVT_Y")
        assert sorted([a, b], key=_snapshot_sort_key) == [b, a]


# ----------------------------------------------------------------------
# _is_correction_regression — channels / world_traits guards
# ----------------------------------------------------------------------


def _ws_with_channels_and_traits(n_chan: int, n_trait: int) -> WorldStateV1:
    return WorldStateV1(
        locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="d", ambient_state={})},
        objects={},
        entities={
            "ENT_X": Entity(id="ENT_X", name="X", location_id="LOC_A",
                            status="healthy", traits={}),
            "ENT_Y": Entity(id="ENT_Y", name="Y", location_id="LOC_A",
                            status="healthy", traits={}),
        },
        events=[
            EventNode(id=f"EVT_{i}", fabula_time=i, syuzhet_index=i,
                      event_type="outcome", description="e")
            for i in range(3)
        ],
        causal_topology=[],
        channels={
            f"CHN_{i}": Channel(
                id=f"CHN_{i}", name=f"c{i}", medium="speech",
                participant_ids=["ENT_X", "ENT_Y"],
            )
            for i in range(n_chan)
        },
        world_traits={
            f"WORLD_T{i}": GlobalTrait(
                id=f"WORLD_T{i}", name=f"t{i}", description="d",
                category="governance",
                magnitude=TraitVector(value=0.5, inertia=0.5),
                affected_domains=["social"],
            )
            for i in range(n_trait)
        },
    )


class TestRegressionGuard:
    def test_channels_majority_loss_blocks(self):
        before = _ws_with_channels_and_traits(n_chan=4, n_trait=0)
        after = _ws_with_channels_and_traits(n_chan=1, n_trait=0)
        msg = _is_correction_regression(before, after)
        assert msg is not None
        assert "channels" in msg

    def test_channels_minor_loss_allowed(self):
        # 4 → 3 is a 25% loss; below the 50% threshold.
        before = _ws_with_channels_and_traits(n_chan=4, n_trait=0)
        after = _ws_with_channels_and_traits(n_chan=3, n_trait=0)
        assert _is_correction_regression(before, after) is None

    def test_world_traits_majority_loss_blocks(self):
        before = _ws_with_channels_and_traits(n_chan=0, n_trait=4)
        after = _ws_with_channels_and_traits(n_chan=0, n_trait=1)
        msg = _is_correction_regression(before, after)
        assert msg is not None
        assert "world_traits" in msg

    def test_no_loss_returns_none(self):
        before = _ws_with_channels_and_traits(n_chan=2, n_trait=2)
        after = _ws_with_channels_and_traits(n_chan=2, n_trait=2)
        assert _is_correction_regression(before, after) is None


# ----------------------------------------------------------------------
# _apply_world_state_patch — channel_renames
# ----------------------------------------------------------------------


class TestChannelRenamesPatch:
    def _ws_with_utterance(self) -> WorldStateV1:
        return WorldStateV1(
            locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="d", ambient_state={})},
            objects={},
            entities={
                "ENT_X": Entity(id="ENT_X", name="X", location_id="LOC_A",
                                status="healthy", traits={}),
                "ENT_Y": Entity(id="ENT_Y", name="Y", location_id="LOC_A",
                                status="healthy", traits={}),
            },
            events=[
                EventNode(
                    id="EVT_UTT_1", fabula_time=1, syuzhet_index=0,
                    event_type="utterance", description="X says hello",
                    speaker_id="ENT_X", addressee_ids=["ENT_Y"],
                    via_channel_id="CHN_OLD",
                ),
            ],
            causal_topology=[],
            channels={
                "CHN_OLD": Channel(
                    id="CHN_OLD", name="old name", medium="speech",
                    participant_ids=["ENT_X", "ENT_Y"],
                ),
            },
        )

    def test_rename_only_patch_is_applied(self):
        ws = self._ws_with_utterance()
        patch = WorldStatePatch(channel_renames={"CHN_OLD": "CHN_NEW"})
        new_ws, changes = _apply_world_state_patch(ws, patch)

        # Channel id was rewritten in the channels dict.
        assert "CHN_NEW" in new_ws.channels
        assert "CHN_OLD" not in new_ws.channels
        assert new_ws.channels["CHN_NEW"].id == "CHN_NEW"

        # Forward provenance: the utterance now points at the new id,
        # rather than being silently nulled by the auto-pruner.
        utt = next(e for e in new_ws.events if e.id == "EVT_UTT_1")
        assert utt.via_channel_id == "CHN_NEW"

        assert changes  # change-log is non-empty
