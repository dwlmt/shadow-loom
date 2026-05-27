# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gap #6: verify Belief / Concern updates routed via
``ChunkTopology.entity_updates`` on a shadow merge land on a
shadow-tagged clone (``shadow_entities[branch_label]``) AND are
visible to ``reconstruct_entity_at`` and to the factual baseline
without contamination.

Belief and Concern live nested inside ``Entity`` (not as
top-level WorldStateV1 fields). The routing therefore depends on
``_get_or_clone_shadow_entity`` materialising the clone before the
chunk's ``belief_updates`` / ``concern_updates`` are applied, and
on ``reconstruct_entity_at`` filtering snapshots by
``snap.world_id == holder.world_id`` so the clone reads its
shadow timeline while the factual entity stays clean.
"""

from __future__ import annotations

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology, EntityUpdate
from shadow_loom.models import (
    Belief,
    Entity,
    EventNode,
    Proposition,
    reconstruct_entity_at,
    WorldStateV1,
)


def _ws_with_one_belief() -> WorldStateV1:
    george = Entity(
        id="ENT_GEORGE", name="George",
        location_id="LOC_PRISON", status="healthy", traits={},
        beliefs=[
            Belief(
                target_id="ENT_WANDA",
                proposition_id="PROP_WANDA_LOVES_GEORGE",
                perceived_state="Wanda loves me.",
                confidence=0.9,
                inertia=0.5,
                established_at_fabula=0,
            ),
        ],
        world_id="factual",
    )
    return WorldStateV1(
        entities={"ENT_GEORGE": george},
        events=[
            EventNode(
                id="EVT_GEORGE_LEARNS_TRUTH",
                fabula_time=500, syuzhet_index=5,
                event_type="outcome",
                actor_ids=["ENT_GEORGE"], target_ids=[],
                description="George learns Wanda was conning him.",
                world_id="factual",
            ),
        ],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[
            Proposition(
                proposition_id="PROP_WANDA_LOVES_GEORGE",
                description="Wanda loves George",
                kind="trait_holds",
            ),
        ],
        channels={},
    )


def test_shadow_belief_update_lands_on_shadow_clone():
    """A shadow-branch belief confidence update via
    ``entity_updates`` materialises an entity clone in
    ``shadow_entities[branch_label]`` carrying the shadow-tagged
    snapshot; the factual entity is untouched."""
    vwm = VersionedWorldModel.from_world_state(_ws_with_one_belief())
    update = EntityUpdate(
        entity_id="ENT_GEORGE",
        fabula_time=500,
        triggered_by="EVT_GEORGE_LEARNS_TRUTH",
        invalidated_belief_targets=["ENT_WANDA::PROP_WANDA_LOVES_GEORGE"],
        new_beliefs=[
            Belief(
                target_id="ENT_WANDA",
                proposition_id="PROP_WANDA_LOVES_GEORGE",
                perceived_state="Wanda was using me.",
                confidence=0.05,
                inertia=0.7,
                established_at_fabula=500,
            ),
        ],
    )
    topology = ChunkTopology(
        chunk_id="CHK_SHADOW",
        entity_updates=[update],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_george_aware")
    # Shadow clone exists and carries the snapshot.
    sidecar = vwm2.current.shadow_entities.get("cf_george_aware") or {}
    clone = sidecar.get("ENT_GEORGE")
    assert clone is not None, "shadow merge must materialise the entity clone"
    assert clone.world_id == "shadow"
    assert any(
        s.triggered_by == "EVT_GEORGE_LEARNS_TRUTH"
        for s in clone.state_timeline
    )
    # Factual entity untouched.
    factual = vwm.current.entities["ENT_GEORGE"]
    assert factual.world_id == "factual"
    assert factual.state_timeline == []


def test_shadow_belief_reconstruction_reflects_clone_state():
    """``reconstruct_entity_at`` on the shadow clone reflects the
    invalidate+add Belief flow; on the factual entity it does not."""
    vwm = VersionedWorldModel.from_world_state(_ws_with_one_belief())
    topology = ChunkTopology(
        chunk_id="CHK_SHADOW",
        entity_updates=[
            EntityUpdate(
                entity_id="ENT_GEORGE",
                fabula_time=500,
                triggered_by="EVT_GEORGE_LEARNS_TRUTH",
                invalidated_belief_targets=[
                    "ENT_WANDA::PROP_WANDA_LOVES_GEORGE",
                ],
                new_beliefs=[
                    Belief(
                        target_id="ENT_WANDA",
                        proposition_id="PROP_WANDA_LOVES_GEORGE",
                        perceived_state="Wanda was using me.",
                        confidence=0.05,
                        inertia=0.7,
                        established_at_fabula=500,
                    ),
                ],
            ),
        ],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    projected = vwm2.current.projected_for_branch(
        branch_world_id="shadow", branch_label="cf",
    )
    george_shadow = projected.entities["ENT_GEORGE"]
    state = reconstruct_entity_at(george_shadow, fabula_time=600)
    # The shadow belief replaced the factual one.
    matched = [
        b for b in state["beliefs"]
        if b.get("proposition_id") == "PROP_WANDA_LOVES_GEORGE"
    ]
    assert len(matched) == 1
    assert matched[0]["confidence"] == 0.05
    # The factual entity at t=600 still reflects the pre-shadow belief.
    factual = vwm.current.entities["ENT_GEORGE"]
    factual_state = reconstruct_entity_at(factual, fabula_time=600)
    factual_matched = [
        b for b in factual_state["beliefs"]
        if b.get("proposition_id") == "PROP_WANDA_LOVES_GEORGE"
    ]
    assert len(factual_matched) == 1
    assert factual_matched[0]["confidence"] == 0.9


def test_shadow_branch_does_not_leak_into_sibling_branch():
    """Two sibling shadow branches must materialise independent
    clones; an update on one branch must not be visible on the
    other."""
    vwm = VersionedWorldModel.from_world_state(_ws_with_one_belief())
    update_a = EntityUpdate(
        entity_id="ENT_GEORGE",
        fabula_time=500,
        triggered_by="EVT_GEORGE_LEARNS_TRUTH",
        invalidated_belief_targets=["ENT_WANDA::PROP_WANDA_LOVES_GEORGE"],
        new_beliefs=[
            Belief(
                target_id="ENT_WANDA",
                proposition_id="PROP_WANDA_LOVES_GEORGE",
                perceived_state="Branch A: Wanda was using me.",
                confidence=0.01,
                inertia=0.7,
                established_at_fabula=500,
            ),
        ],
    )
    vwm_a = vwm.merge(
        ChunkTopology(
            chunk_id="CHK_A",
            entity_updates=[update_a],
        ),
        world_id="shadow", branch_label="branch_a",
    )
    projected_b = vwm_a.current.projected_for_branch(
        branch_world_id="shadow", branch_label="branch_b",
    )
    george_b = projected_b.entities["ENT_GEORGE"]
    state_b = reconstruct_entity_at(george_b, fabula_time=600)
    matched_b = [
        b for b in state_b["beliefs"]
        if b.get("proposition_id") == "PROP_WANDA_LOVES_GEORGE"
    ]
    # branch_b never wrote anything → still sees the original
    # factual belief at confidence 0.9.
    assert len(matched_b) == 1
    assert matched_b[0]["confidence"] == 0.9
