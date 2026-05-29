# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""F2 regression: channel-dedup forwarding must rewrite
``shadow_removed_channel_ids`` tombstone buckets.

When ``VersionedWorldModel.merge()`` collapses two channel ids via
``_deduplicate_channels_with_map``, every other carrier of the alias
(events, beliefs, sidecars) is rewritten in lockstep. The per-branch
tombstone list ``shadow_removed_channel_ids[branch]`` must also be
rewritten or the AMWN ``projected_for_branch`` reader will fail to
suppress the canonical id that the shadow do-surgery intended to
remove.
"""

from __future__ import annotations

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    Channel,
    Entity,
    EventNode,
    WorldStateV1,
)


def _ws_with_channel(channel_id: str) -> WorldStateV1:
    ent_a = Entity(
        id="ENT_A", name="A", location_id="LOC_X", status="healthy",
        traits={}, world_id="factual",
    )
    ent_b = Entity(
        id="ENT_B", name="B", location_id="LOC_X", status="healthy",
        traits={}, world_id="factual",
    )
    ch = Channel(
        id=channel_id, name="duplex", medium="speech",
        participant_ids=["ENT_A", "ENT_B"],
        directionality="duplex",
        established_at_fabula=10,
    )
    evt = EventNode(
        id="EVT_1", fabula_time=20, syuzhet_index=0,
        event_type="utterance", actor_ids=["ENT_A"],
        target_ids=["ENT_B"], description="hi",
        world_id="factual",
    )
    return WorldStateV1(
        entities={"ENT_A": ent_a, "ENT_B": ent_b},
        events=[evt],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[], channels={channel_id: ch},
    )


def test_dedup_rewrites_shadow_removed_channel_ids_tombstone():
    """A tombstone keyed on the alias id must become a tombstone
    keyed on the canonical id after dedup forwards the alias."""
    ws = _ws_with_channel("CHN_ALIAS")
    # Pre-seed a per-branch tombstone targeting the alias id.
    ws.shadow_removed_channel_ids = {"branch_x": ["CHN_ALIAS"]}
    vwm = VersionedWorldModel.from_world_state(ws)

    # New topology contains the same shape-keyed channel with a
    # different id; dedup collapses the alias onto the new canonical.
    new_ch = Channel(
        id="CHN_CANONICAL", name="duplex", medium="speech",
        participant_ids=["ENT_A", "ENT_B"],
        directionality="duplex",
        established_at_fabula=10,
        world_id="factual",
    )
    topology = ChunkTopology(
        chunk_id="CHK_REID",
        channels={"CHN_CANONICAL": new_ch},
    )
    vwm2 = vwm.merge(topology, source="test", description="dedup-tombstone")

    tombstones = vwm2.current.shadow_removed_channel_ids or {}
    bucket = tombstones.get("branch_x", [])
    assert "CHN_ALIAS" not in bucket, (
        "Tombstone bucket must be rewritten away from the now-aliased id."
    )
    # The canonical id may be either CHN_ALIAS or CHN_CANONICAL
    # depending on dedup ordering; whichever survives in
    # merged.channels is what the tombstone should now reference.
    surviving = set(vwm2.current.channels.keys())
    assert bucket and bucket[0] in surviving, (
        f"Tombstone {bucket!r} must reference a surviving channel id "
        f"(survivors={surviving!r})."
    )
