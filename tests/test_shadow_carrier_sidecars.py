# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""AMWN node-splitting sidecars for non-Entity carriers.

Mirrors :mod:`tests.test_shadow_entity_amwn_split` for
``NarrativeObject``, ``Proposition``, ``GlobalTrait``, and
holder-entity concerns. Each test verifies the same three
properties:

1. A shadow-branch write targeting a factual-tagged carrier
   creates a per-branch split copy in the appropriate sidecar
   (lazy AMWN node-split) instead of being silently blocked.
2. The factual record is never mutated by the shadow write.
3. ``WorldStateV1.projected_for_branch`` swaps in the clone
   on layered reads so downstream consumers see the
   counterfactual outcome.
"""

from __future__ import annotations

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    Entity,
    EventNode,
    GlobalTrait,
    NarrativeObject,
    Proposition,
    TraitVector,
    WorldStateV1,
)


def _factual_world() -> WorldStateV1:
    coady = Entity(
        id="ENT_COADY", name="Mrs Coady",
        location_id="LOC_FLAT", status="healthy", traits={},
        world_id="factual",
    )
    dog_corpse = NarrativeObject(
        id="OBJ_DOG", name="dog",
        location_id="LOC_YARD", owner_id=None,
        affordances=[],
        world_id="factual",
    )
    prop = Proposition(
        proposition_id="PROP_COADY_DIES",
        description="Mrs Coady dies of fright",
        kind="outcome",
        truth_at_fabula={150: True},
        world_id="factual",
    )
    wt = GlobalTrait(
        id="WORLD_WEATHER",
        name="Weather",
        description="Weather over the moor",
        category="environment",
        magnitude=TraitVector(value=0.5, inertia=0.5),
        affected_domains=["physical"],
        world_id="factual",
    )
    return WorldStateV1(
        entities={"ENT_COADY": coady},
        objects={"OBJ_DOG": dog_corpse},
        locations={},
        events=[EventNode(
            id="EVT_KEN_KILLS_DOGS", fabula_time=100, syuzhet_index=100,
            event_type="outcome", actor_ids=["ENT_KEN"], target_ids=["OBJ_DOG"],
            description="Ken kills the dogs",
            world_id="factual",
        )],
        propositions=[prop],
        world_traits={"WORLD_WEATHER": wt},
        causal_topology=[], spatial_topology=[], social_topology=[],
        channels={},
    )


# ---------------------------------------------------------------------
# Proposition truth-commit routing
# ---------------------------------------------------------------------


def test_shadow_truth_commit_routes_to_sidecar():
    """A shadow-branch truth commit on a factual proposition
    materialises a sidecar clone instead of being skipped, and
    the factual ``truth_at_fabula`` dict is untouched."""
    vwm = VersionedWorldModel.from_world_state(_factual_world())
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        proposition_truth_commits=[
            {"proposition_id": "PROP_COADY_DIES", "fabula_time": 250, "truth": False, "triggered_by": "EVT_CF_INTERVIEW"},
        ],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")
    sidecar = vwm2.current.shadow_propositions["cf_dogs"]
    assert "PROP_COADY_DIES" in sidecar
    clone = sidecar["PROP_COADY_DIES"]
    assert clone.world_id == "shadow"
    assert clone.truth_at_fabula.get(250) is False
    factual = next(p for p in vwm2.current.propositions if p.proposition_id == "PROP_COADY_DIES")
    assert factual.world_id == "factual"
    assert 250 not in factual.truth_at_fabula
    projected = vwm2.current.projected_for_branch("shadow", "cf_dogs")
    pj = next(p for p in projected.propositions if p.proposition_id == "PROP_COADY_DIES")
    assert pj.truth_at_fabula.get(250) is False


def test_shadow_proposition_genesis_routes_to_sidecar():
    vwm = VersionedWorldModel.from_world_state(_factual_world())
    new_prop = Proposition(
        proposition_id="PROP_CF_ONLY",
        description="A counterfactual-only proposition",
        kind="outcome",
        truth_at_fabula={},
        world_id="factual",
    )
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        new_propositions={"PROP_CF_ONLY": new_prop},
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    assert not any(p.proposition_id == "PROP_CF_ONLY" for p in vwm2.current.propositions)
    sidecar = vwm2.current.shadow_propositions["cf"]
    assert "PROP_CF_ONLY" in sidecar
    assert sidecar["PROP_CF_ONLY"].world_id == "shadow"
    projected = vwm2.current.projected_for_branch("shadow", "cf")
    assert any(p.proposition_id == "PROP_CF_ONLY" for p in projected.propositions)


def test_shadow_proposition_framing_snapshot_routes_to_sidecar():
    vwm = VersionedWorldModel.from_world_state(_factual_world())
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        proposition_snapshots=[{
            "proposition_id": "PROP_COADY_DIES",
            "fabula_time": 220,
            "triggered_by": "EVT_CF_INTERVIEW",
            "stakes": 0.7,
            "audience_default_prior": 0.5,
            "description": "Shadow framing",
        }],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    sidecar = vwm2.current.shadow_propositions["cf"]
    assert "PROP_COADY_DIES" in sidecar
    clone = sidecar["PROP_COADY_DIES"]
    triggers = {s.triggered_by for s in clone.state_timeline}
    assert "EVT_CF_INTERVIEW" in triggers
    factual = next(p for p in vwm2.current.propositions if p.proposition_id == "PROP_COADY_DIES")
    f_triggers = {s.triggered_by for s in factual.state_timeline}
    assert "EVT_CF_INTERVIEW" not in f_triggers


# ---------------------------------------------------------------------
# NarrativeObject routing
# ---------------------------------------------------------------------


def test_shadow_object_update_routes_to_sidecar():
    vwm = VersionedWorldModel.from_world_state(_factual_world())
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        object_updates=[{
            "object_id": "OBJ_DOG",
            "fabula_time": 250,
            "triggered_by": "EVT_CF_INTERVIEW",
            "new_location_id": "LOC_PARK",
        }],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")
    sidecar = vwm2.current.shadow_objects["cf_dogs"]
    assert "OBJ_DOG" in sidecar
    clone = sidecar["OBJ_DOG"]
    assert clone.world_id == "shadow"
    clone_triggers = {s.triggered_by for s in clone.state_timeline}
    assert "EVT_CF_INTERVIEW" in clone_triggers
    factual = vwm2.current.objects["OBJ_DOG"]
    assert factual.world_id == "factual"
    f_triggers = {s.triggered_by for s in factual.state_timeline}
    assert "EVT_CF_INTERVIEW" not in f_triggers
    projected = vwm2.current.projected_for_branch("shadow", "cf_dogs")
    assert projected.objects["OBJ_DOG"].world_id == "shadow"


# ---------------------------------------------------------------------
# new_concerns routing (via entity sidecar)
# ---------------------------------------------------------------------


def test_shadow_new_concerns_routes_via_entity_sidecar():
    vwm = VersionedWorldModel.from_world_state(_factual_world())
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        new_concerns={
            "ENT_COADY": [{
                "concern_id": "CCN_COADY_CF",
                "proposition_id": "PROP_COADY_DIES",
                "polarity": "fear",
                "baseline_salience": 0.6,
            }],
        },
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf")
    sidecar = vwm2.current.shadow_entities["cf"]
    assert "ENT_COADY" in sidecar
    clone = sidecar["ENT_COADY"]
    concern_ids = {c.concern_id for c in clone.concerns}
    assert "CCN_COADY_CF" in concern_ids
    factual = vwm2.current.entities["ENT_COADY"]
    f_concern_ids = {c.concern_id for c in factual.concerns}
    assert "CCN_COADY_CF" not in f_concern_ids
