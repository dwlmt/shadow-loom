# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for sandbox-spawn promotion into the canonical world model.

Covers :func:`shadow_loom.extract_graph.promote_sandbox_spawns` and
``VersionedWorldModel.merge`` handling of the ``new_*`` topology fields.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from shadow_loom.extract_graph import (
    VersionedWorldModel,
    promote_sandbox_spawns,
)
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    Entity,
    GlobalTrait,
    Location,
    NarrativeObject,
    TraitVector,
)

from example_worlds.macbeth import world_state as macbeth_ws


# =====================================================================
# promote_sandbox_spawns — extraction from physics_state node-link dict
# =====================================================================
class TestPromoteSandboxSpawns:
    """Promotion is type-aware, idempotent, and tolerant of bad payloads."""

    def test_empty_physics_state_returns_empty_buckets(self):
        out = promote_sandbox_spawns(macbeth_ws, {})
        assert out == {
            "entities": {},
            "objects": {},
            "locations": {},
            "world_traits": {},
            "channels": {},
            "propositions": {},
            "concerns": {},
        }

    def test_none_physics_state_returns_empty_buckets(self):
        out = promote_sandbox_spawns(macbeth_ws, None)
        assert out["entities"] == {}

    def test_promotes_new_entity(self):
        # Pick an existing macbeth location so the entity is well-formed.
        loc_id = next(iter(macbeth_ws.locations))
        physics_state = {
            "nodes": [
                {
                    "id": "ENT_GHOST",
                    "world_id": "shadow",
                    "node_type": "Entity",
                    "name": "Banquo's Ghost",
                    "located_in": loc_id,
                    "status": "dead",
                }
            ],
            "links": [],
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        assert "ENT_GHOST" in out["entities"]
        ent = out["entities"]["ENT_GHOST"]
        assert isinstance(ent, Entity)
        assert ent.name == "Banquo's Ghost"
        assert ent.location_id == loc_id
        assert ent.status == "dead"

    def test_skips_entity_without_location(self):
        physics_state = {
            "nodes": [
                {
                    "id": "ENT_FLOATING",
                    "world_id": "shadow",
                    "node_type": "Entity",
                    "name": "Floating Spirit",
                }
            ]
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        assert "ENT_FLOATING" not in out["entities"]

    def test_promotes_new_object(self):
        physics_state = {
            "nodes": [
                {
                    "id": "OBJ_NEW_DAGGER",
                    "world_id": "shadow",
                    "node_type": "NarrativeObject",
                    "name": "Phantom Dagger",
                    "owner_id": "ENT_MACBETH",
                }
            ]
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        obj = out["objects"]["OBJ_NEW_DAGGER"]
        assert isinstance(obj, NarrativeObject)
        assert obj.owner_id == "ENT_MACBETH"
        assert obj.affordances == []

    def test_promotes_new_location(self):
        physics_state = {
            "nodes": [
                {
                    "id": "LOC_HIDDEN_TOWER",
                    "world_id": "shadow",
                    "node_type": "Location",
                    "name": "Hidden Tower",
                    "description": "A secret tower above Inverness.",
                }
            ]
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        loc = out["locations"]["LOC_HIDDEN_TOWER"]
        assert isinstance(loc, Location)
        assert loc.name == "Hidden Tower"
        assert loc.description.startswith("A secret tower")

    def test_promotes_new_world_trait_with_defaults(self):
        physics_state = {
            "nodes": [
                {
                    "id": "WORLD_NEW_CURSE",
                    "world_id": "shadow",
                    "node_type": "WorldTrait",
                    "name": "Banquo's Curse",
                    "description": "An ancestral curse.",
                }
            ]
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        wt = out["world_traits"]["WORLD_NEW_CURSE"]
        assert isinstance(wt, GlobalTrait)
        # Defaults applied when payload omits them.
        assert wt.category == "social_structure"
        assert wt.magnitude.value == 0.5
        assert wt.magnitude.inertia == 0.5

    def test_promotes_world_trait_with_explicit_magnitude_dict(self):
        physics_state = {
            "nodes": [
                {
                    "id": "WORLD_DECREE",
                    "world_id": "shadow",
                    "node_type": "GlobalTrait",
                    "name": "Royal Decree",
                    "description": "A new law.",
                    "category": "governance",
                    "magnitude": {
                        "value": 0.9,
                        "inertia": 0.7,
                        "evidence_strength": "strong",
                    },
                }
            ]
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        wt = out["world_traits"]["WORLD_DECREE"]
        assert wt.category == "governance"
        assert wt.magnitude.value == 0.9
        assert wt.magnitude.inertia == 0.7

    def test_skips_factual_world_id_nodes(self):
        physics_state = {
            "nodes": [
                {
                    "id": "ENT_NEW",
                    "world_id": "factual",
                    "node_type": "Entity",
                    "name": "Already canonical",
                    "located_in": next(iter(macbeth_ws.locations)),
                }
            ]
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        assert out["entities"] == {}

    def test_skips_existing_canonical_ids(self):
        # ENT_MACBETH already exists in the canonical world — must not
        # be re-promoted even if it appears as a shadow node.
        existing_id = next(iter(macbeth_ws.entities))
        loc_id = next(iter(macbeth_ws.locations))
        physics_state = {
            "nodes": [
                {
                    "id": existing_id,
                    "world_id": "shadow",
                    "node_type": "Entity",
                    "name": "Imposter",
                    "located_in": loc_id,
                }
            ]
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        assert existing_id not in out["entities"]

    def test_malformed_node_does_not_crash_promotion(self):
        loc_id = next(iter(macbeth_ws.locations))
        physics_state = {
            "nodes": [
                # Bad: status is not in the Literal allowlist.
                {
                    "id": "ENT_BAD",
                    "world_id": "shadow",
                    "node_type": "Entity",
                    "located_in": loc_id,
                    "status": "not_a_real_status",
                },
                # Good: should still be promoted alongside the bad one.
                {
                    "id": "ENT_OK",
                    "world_id": "shadow",
                    "node_type": "Entity",
                    "located_in": loc_id,
                    "name": "Good entity",
                },
            ]
        }
        out = promote_sandbox_spawns(macbeth_ws, physics_state)
        assert "ENT_BAD" not in out["entities"]
        assert "ENT_OK" in out["entities"]


# =====================================================================
# VersionedWorldModel.merge — new_* topology fields
# =====================================================================
class TestMergeNewNodes:
    """Merge promotes ``new_*`` topology fields into the canonical world."""

    def _empty_topology(self) -> ChunkTopology:
        return ChunkTopology()

    def test_merge_records_new_entity_in_changeset(self):
        loc_id = next(iter(macbeth_ws.locations))
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = self._empty_topology()
        topo.new_entities["ENT_NEWCOMER"] = Entity(
            id="ENT_NEWCOMER",
            name="Newcomer",
            location_id=loc_id,
            status="healthy",
            traits={},
        )
        vwm_next = vwm.merge(topo, source="test", description="spawn entity")
        assert "ENT_NEWCOMER" in vwm_next.current.entities
        cs = vwm_next.history[-1].changeset
        assert cs.entities_added == 1

    def test_merge_records_new_object_in_changeset(self):
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = self._empty_topology()
        topo.new_objects["OBJ_NEW"] = NarrativeObject(
            id="OBJ_NEW",
            name="New Object",
            location_id=None,
            owner_id=None,
            affordances=[],
        )
        vwm_next = vwm.merge(topo, source="test", description="spawn object")
        assert "OBJ_NEW" in vwm_next.current.objects
        assert vwm_next.history[-1].changeset.objects_added == 1

    def test_merge_records_new_location_in_changeset(self):
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = self._empty_topology()
        topo.new_locations["LOC_NEW"] = Location(
            name="New Location",
            description="A new place.",
        )
        vwm_next = vwm.merge(topo, source="test", description="spawn location")
        assert "LOC_NEW" in vwm_next.current.locations
        assert vwm_next.history[-1].changeset.locations_added == 1

    def test_merge_records_new_world_trait_in_changeset(self):
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = self._empty_topology()
        topo.new_world_traits["WORLD_NEW"] = GlobalTrait(
            id="WORLD_NEW",
            name="New Trait",
            description="A new fact.",
            category="governance",
            magnitude=TraitVector(value=0.5, inertia=0.5, evidence_strength="moderate"),
            affected_domains=["social"],
        )
        vwm_next = vwm.merge(topo, source="test", description="spawn trait")
        assert "WORLD_NEW" in vwm_next.current.world_traits
        assert vwm_next.history[-1].changeset.world_traits_added == 1

    def test_merge_is_idempotent_on_existing_id(self):
        existing_id = next(iter(macbeth_ws.entities))
        loc_id = next(iter(macbeth_ws.locations))
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = self._empty_topology()
        topo.new_entities[existing_id] = Entity(
            id=existing_id,
            name="Imposter",
            location_id=loc_id,
            status="healthy",
            traits={},
        )
        vwm_next = vwm.merge(topo, source="test", description="dup")
        # Existing entity is preserved (not overwritten with imposter).
        assert vwm_next.current.entities[existing_id].name != "Imposter"
        assert vwm_next.history[-1].changeset.entities_added == 0

    def test_merge_tags_spawn_with_world_id(self):
        loc_id = next(iter(macbeth_ws.locations))
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = self._empty_topology()
        topo.new_entities["ENT_SHADOW"] = Entity(
            id="ENT_SHADOW",
            name="Shadow Entity",
            location_id=loc_id,
            status="healthy",
            traits={},
        )
        vwm_next = vwm.merge(
            topo, source="test", description="branch spawn",
            world_id="shadow", branch_label="cf-branch-1",
        )
        assert vwm_next.current.entities["ENT_SHADOW"].world_id == "shadow"
