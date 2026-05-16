# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shadow-merge ``NarrativeObject.state_timeline`` cascade.

Parallel to the entity / world-trait / proposition / concern
timeline cascades: when a counterfactual / intervention prunes
events via ``ChunkTopology.suppressed_event_ids``, any
``ObjectStateSnapshot`` whose ``triggered_by`` references a
now-suppressed event must be dropped from the shadow row's
``NarrativeObject.state_timeline``. Otherwise an object that was
picked up / moved / poisoned by the suppressed event still appears
in the new owner's inventory (or at the new location, or in the new
state) on the shadow branch \u2014 producing the documented
prose\u2194interrogation drift (e.g. diamonds still "hidden in the
old workshop" on a branch where the heist that placed them there
was suppressed).
"""

from __future__ import annotations

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    Entity,
    EventNode,
    NarrativeObject,
    ObjectStateSnapshot,
    WorldStateV1,
)


def _ws_with_object_moved_by_event() -> WorldStateV1:
    """Diamonds at a workshop, then moved to a hideout by the heist
    event ``EVT_HEIST`` at t=200."""
    coady = Entity(
        id="ENT_MRS_COADY", name="Mrs Coady",
        location_id="LOC_FLAT", status="healthy", traits={},
        world_id="factual",
    )
    heist = EventNode(
        id="EVT_HEIST", fabula_time=200, syuzhet_index=200,
        event_type="outcome",
        actor_ids=["ENT_GEORGE"], target_ids=["OBJ_DIAMONDS"],
        description="George steals the diamonds",
        world_id="factual",
    )
    diamonds = NarrativeObject(
        id="OBJ_DIAMONDS", name="diamonds",
        location_id="LOC_WORKSHOP", owner_id=None,
        affordances=[],
        state_timeline=[
            ObjectStateSnapshot(
                world_id="factual",
                fabula_time=200,
                triggered_by="EVT_HEIST",
                location_id="LOC_HIDEOUT",
                owner_id="ENT_GEORGE",
            ),
        ],
    )
    return WorldStateV1(
        entities={"ENT_MRS_COADY": coady},
        events=[heist],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={"OBJ_DIAMONDS": diamonds},
        world_traits={}, propositions=[], channels={},
    )


def test_shadow_suppression_drops_object_snapshot_with_suppressed_trigger():
    """Suppressing ``EVT_HEIST`` drops the post-heist
    ``ObjectStateSnapshot`` from the shadow row's object timeline."""
    vwm = VersionedWorldModel.from_world_state(_ws_with_object_moved_by_event())
    topology = ChunkTopology(
        chunk_id="CHK_CF_NO_HEIST",
        suppressed_event_ids=["EVT_HEIST"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_no_heist")
    diamonds = vwm2.current.objects["OBJ_DIAMONDS"]
    triggers = {s.triggered_by for s in diamonds.state_timeline}
    assert "EVT_HEIST" not in triggers, (
        "Object snapshot whose triggered_by is suppressed must be dropped."
    )
    # Factual row untouched.
    factual_triggers = {s.triggered_by for s in vwm.current.objects["OBJ_DIAMONDS"].state_timeline}
    assert "EVT_HEIST" in factual_triggers


def test_shadow_suppression_preserves_unrelated_object_snapshots():
    """Snapshots whose ``triggered_by`` is NOT in the suppression
    set survive (e.g. earlier ownership transfers)."""
    ws = _ws_with_object_moved_by_event()
    diamonds = ws.objects["OBJ_DIAMONDS"]
    diamonds.state_timeline.insert(0, ObjectStateSnapshot(
        world_id="factual",
        fabula_time=100,
        triggered_by="EVT_EARLIER_TRANSFER",
        owner_id="ENT_OWNER_PRIOR",
    ))
    vwm = VersionedWorldModel.from_world_state(ws)
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        suppressed_event_ids=["EVT_HEIST"],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    triggers = {s.triggered_by for s in vwm2.current.objects["OBJ_DIAMONDS"].state_timeline}
    assert "EVT_EARLIER_TRANSFER" in triggers
    assert "EVT_HEIST" not in triggers
