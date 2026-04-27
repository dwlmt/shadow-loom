from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from shadow_loom.models import (
    EntityStateSnapshot,
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
)

if TYPE_CHECKING:
    from shadow_loom.ingestion import ChunkTopology, ExtractionConfig

logger = logging.getLogger(__name__)

# ==========================================
# 1. THE OUTPUT SCHEMA
# ==========================================
class EgoGraphPayload(BaseModel):
    """The highly localized JSON payload sent to the LLM."""
    focus_entities: List[dict]
    current_locations: List[dict]
    present_entities: List[dict]
    present_objects: List[dict]
    relevant_relationships: List[dict]
    relevant_causal_edges: List[dict]
    relevant_spatial_edges: List[dict]
    relevant_information_edges: List[dict]
    recent_memory: List[dict]
    world_traits: List[dict] = Field(default_factory=list)

# ==========================================
# 2. THE IN-MEMORY EXTRACTION FUNCTION
# ==========================================
def extract_ego_graph_from_memory(
    world_state: WorldStateV1,
    focus_entity_ids: List[str],
    temporal_anchor: Optional[int] = None,
    memory_limit: int = 5
) -> EgoGraphPayload:
    """
    Calculates the union of localized Ego-Graphs for multiple entities.
    """
    logger.info("Initiating Multi-Ego GraphRAG for: %s", focus_entity_ids)

    # Use Sets to prevent duplicating nodes if two targets are in the same room
    focus_entities = []
    location_ids: Set[str] = set()
    current_locations = []

    # 1. Locate all Focus Entities and their Rooms
    for f_id in focus_entity_ids:
        ent = world_state.entities.get(f_id)
        if ent:
            ent_data = ent.model_dump()
            if temporal_anchor is not None:
                if ent.state_timeline:
                    reconstructed = reconstruct_entity_at(ent, temporal_anchor)
                    ent_data["traits"] = reconstructed["traits"]
                    ent_data["beliefs"] = reconstructed["beliefs"]
                    ent_data["status"] = reconstructed["status"]
                    ent_data["location_id"] = reconstructed["location_id"]
                else:
                    ent_data["beliefs"] = [
                        b for b in ent_data.get("beliefs", [])
                        if b.get("established_at_fabula", 0) <= temporal_anchor
                    ]
            focus_entities.append(ent_data)
            location_ids.add(ent_data["location_id"])
        else:
            logger.warning("Focus Entity '%s' not found.", f_id)

    if not focus_entities:
        raise ValueError(f"None of the focus entities {focus_entity_ids} found in WorldState.")

    # Also pull 1-hop neighbor locations via spatial_topology
    neighbor_location_ids: Set[str] = set()
    for se in world_state.spatial_topology:
        if se.source_id in location_ids and se.target_id not in location_ids:
            neighbor_location_ids.add(se.target_id)
        elif se.target_id in location_ids and se.source_id not in location_ids:
            neighbor_location_ids.add(se.source_id)

    all_location_ids = location_ids | neighbor_location_ids

    added_locs: Set[str] = set()
    for loc_id in all_location_ids:
        if loc_id not in added_locs:
            loc = world_state.locations.get(loc_id)
            if loc:
                loc_data = loc.model_dump()
                loc_data["id"] = loc_id
                current_locations.append(loc_data)
                added_locs.add(loc_id)

    # 2. The Spatial Filter (Who/What else is in ANY of these rooms?)
    present_entities = []
    present_entity_ids: Set[str] = set()
    focus_id_set = set(focus_entity_ids)

    for ent_id, entity in world_state.entities.items():
        if ent_id in focus_id_set:
            continue
        # Determine the entity's effective location at the anchor time
        if temporal_anchor is not None and entity.state_timeline:
            effective_loc = reconstruct_entity_at(entity, temporal_anchor)["location_id"]
        else:
            effective_loc = entity.location_id
        # If they are in one of the active rooms
        if effective_loc in location_ids:
            ent_data = entity.model_dump()
            if temporal_anchor is not None:
                if entity.state_timeline:
                    reconstructed = reconstruct_entity_at(entity, temporal_anchor)
                    ent_data["traits"] = reconstructed["traits"]
                    ent_data["beliefs"] = reconstructed["beliefs"]
                    ent_data["status"] = reconstructed["status"]
                    ent_data["location_id"] = reconstructed["location_id"]
                else:
                    ent_data["beliefs"] = [
                        b for b in ent_data.get("beliefs", [])
                        if b.get("established_at_fabula", 0) <= temporal_anchor
                    ]
            present_entities.append(ent_data)
            present_entity_ids.add(ent_id)

    present_objects = []
    for obj_id, obj in world_state.objects.items():
        is_on_active_floor = (obj.location_id in location_ids)
        is_held_by_active_local = (obj.owner_id in present_entity_ids or obj.owner_id in focus_id_set)

        if is_on_active_floor or is_held_by_active_local:
            present_objects.append(obj.model_dump())

    # 3. The Relational Filter
    relevant_relationships = []
    scene_entity_ids = focus_id_set | present_entity_ids
    for edge in world_state.social_topology:
        # Include relationships where at least one endpoint is a focus entity
        # and both endpoints are in the scene (focus + co-present)
        src_in_focus = edge.source_entity_id in focus_id_set
        tgt_in_focus = edge.target_entity_id in focus_id_set
        src_in_scene = edge.source_entity_id in scene_entity_ids
        tgt_in_scene = edge.target_entity_id in scene_entity_ids
        if (src_in_focus or tgt_in_focus) and src_in_scene and tgt_in_scene:
            # Time-slice: exclude relationships updated after the anchor
            if temporal_anchor is not None and edge.last_updated_fabula > temporal_anchor:
                logger.debug("[EgoGraph] Excluded relationship %s→%s: last_updated_fabula=%d > anchor=%d",
                             edge.source_entity_id, edge.target_entity_id, edge.last_updated_fabula, temporal_anchor)
                continue
            relevant_relationships.append(edge.model_dump())

    # 4. The Temporal Filter (Memory & Time-Slicing)
    valid_events = world_state.events
    if temporal_anchor is not None:
        valid_events = [evt for evt in valid_events if evt.fabula_time <= temporal_anchor]

    valid_events = sorted(valid_events, key=lambda x: x.fabula_time, reverse=True)
    recent_memory = [evt.model_dump() for evt in valid_events[:memory_limit]]

    # 5. The Spatial Navigation Filter (SpatialEdges connecting loaded locations)
    relevant_spatial_edges = []
    for se in world_state.spatial_topology:
        if se.source_id in all_location_ids and se.target_id in all_location_ids:
            # Skip paths not yet established at the anchor time
            if temporal_anchor is not None and se.established_at_fabula > temporal_anchor:
                continue
            # Skip destroyed paths (destroyed before or at anchor, or destroyed at all if no anchor)
            if se.destroyed_at_fabula is not None:
                if temporal_anchor is None or se.destroyed_at_fabula <= temporal_anchor:
                    continue
            relevant_spatial_edges.append(se.model_dump())

    # 6. The Information Filter (Comms links involving focus entities, time-sliced)
    relevant_information_edges = []
    for ie in world_state.information_topology:
        # Participant filter: source or any target must be a focus entity
        participants = {ie.source_id} | set(ie.target_ids)
        if not participants & focus_id_set:
            continue
        # Temporal filter: link must have been established by the anchor
        if temporal_anchor is not None and ie.established_at_fabula > temporal_anchor:
            continue
        # Skip terminated links (anchor past termination, or no anchor but link is terminated)
        if ie.terminated_at_fabula is not None:
            if temporal_anchor is not None and ie.terminated_at_fabula <= temporal_anchor:
                logger.debug("[EgoGraph] Excluded info edge %s→%s: terminated_at_fabula=%d <= anchor=%d",
                             ie.source_id, ie.target_ids, ie.terminated_at_fabula, temporal_anchor)
                continue
            if temporal_anchor is None:
                logger.debug("[EgoGraph] Excluded info edge %s→%s: terminated (no anchor)",
                             ie.source_id, ie.target_ids)
                continue
        relevant_information_edges.append(ie.model_dump())

    # 7. The Causal Filter (CausalEdges where both endpoints are in the scene)
    # Include WORLD_ IDs so causal edges from world traits pass the filter
    world_trait_ids = set(world_state.world_traits.keys())
    scene_node_ids = (
        focus_id_set
        | present_entity_ids
        | all_location_ids
        | {obj["id"] for obj in present_objects}
        | {evt["id"] for evt in recent_memory}
        | world_trait_ids
    )
    relevant_causal_edges = []
    for edge in world_state.causal_topology:
        if edge.source_id in scene_node_ids and edge.target_id in scene_node_ids:
            # Filter out future causal edges when a temporal anchor is set
            if temporal_anchor is not None and edge.fabula_time > temporal_anchor:
                continue
            relevant_causal_edges.append(edge.model_dump())

    # 8. World Traits (always include ALL — they are global, no spatial filtering)
    world_traits_payload: List[dict] = []
    for wt_id, wt in world_state.world_traits.items():
        wt_data = wt.model_dump()
        if temporal_anchor is not None and wt.state_timeline:
            reconstructed = reconstruct_world_trait_at(wt, temporal_anchor)
            wt_data["magnitude"] = reconstructed["magnitude"]
            wt_data["description"] = reconstructed["description"]
        world_traits_payload.append(wt_data)

    payload = EgoGraphPayload(
        focus_entities=focus_entities,
        current_locations=current_locations,
        present_entities=present_entities,
        present_objects=present_objects,
        relevant_relationships=relevant_relationships,
        relevant_causal_edges=relevant_causal_edges,
        relevant_spatial_edges=relevant_spatial_edges,
        relevant_information_edges=relevant_information_edges,
        recent_memory=recent_memory,
        world_traits=world_traits_payload,
    )

    logger.info("Multi-Ego GraphRAG complete — %d focus, %d locations, %d co-present, %d objects, %d relationships, %d causal, %d spatial, %d info, %d memory",
                 len(focus_entities), len(current_locations), len(present_entities),
                 len(present_objects), len(relevant_relationships), len(relevant_causal_edges),
                 len(relevant_spatial_edges), len(relevant_information_edges), len(recent_memory))
    return payload


# ==========================================
# 3. FULL WORLD-STATE EXTRACTION (for Interrogation)
# ==========================================
def extract_full_world_state(
    world_state: WorldStateV1,
    temporal_anchor: Optional[int] = None,
) -> dict:
    """
    Serialises the entire WorldStateV1 as a dictionary, optionally
    time-sliced to only include events at or before *temporal_anchor*.
    """
    dump = world_state.model_dump()

    if temporal_anchor is not None:
        pre_count = len(dump["events"])
        dump["events"] = [
            evt for evt in dump["events"]
            if evt["fabula_time"] <= temporal_anchor
        ]
        # Time-slice information_topology: exclude future and terminated comms
        dump["information_topology"] = [
            ie for ie in dump.get("information_topology", [])
            if ie["established_at_fabula"] <= temporal_anchor
            and (ie.get("terminated_at_fabula") is None or ie["terminated_at_fabula"] > temporal_anchor)
        ]
        logger.info("Omniscient Graph extracted — %d entities, %d locations, %d/%d events (anchor T=%d)",
                     len(dump["entities"]), len(dump["locations"]),
                     len(dump["events"]), pre_count, temporal_anchor)
    else:
        # Without an anchor, exclude terminated comms (they are dead links)
        dump["information_topology"] = [
            ie for ie in dump.get("information_topology", [])
            if ie.get("terminated_at_fabula") is None
        ]
        logger.info("Omniscient Graph extracted — %d entities, %d locations, %d events (no anchor)",
                     len(dump["entities"]), len(dump["locations"]), len(dump["events"]))

    return dump


# ==========================================
# 4. PROSE → TOPOLOGY EXTRACTION
# ==========================================

def extract_topology_from_prose(
    prose: str,
    world_state: WorldStateV1,
    config: "ExtractionConfig | None" = None,
) -> "ChunkTopology":
    """Extract graph topology from generated prose using the existing physics + social agents.

    Builds a ``GlobalRegister`` directly from the ``WorldStateV1`` (no LLM
    ontology extraction needed — the entities/locations/objects are already
    known) and then runs the physics and social extraction agents on the
    prose as a single chunk.

    Returns a :class:`ChunkTopology` containing new events, edges, and
    entity state updates found in the prose.
    """
    from shadow_loom.ingestion import (
        ChunkTopology,
        ExtractionConfig,
        GlobalRegister,
        PhysicsExtraction,
        SocialExtraction,
        SocraticScaffold,
        _PhysicsDeps,
        _SocialDeps,
        _build_physics_agent,
        _build_social_agent,
    )

    if config is None:
        config = ExtractionConfig()

    register = GlobalRegister(
        locations=world_state.locations,
        objects=world_state.objects,
        entities=world_state.entities,
        world_traits=world_state.world_traits,
    )

    empty_scaffold = SocraticScaffold(qa_pairs=[])

    # --- Physics extraction (events + causal + spatial + entity_updates) ---
    physics_agent = _build_physics_agent(config)
    physics_deps = _PhysicsDeps(
        global_register=register,
        scaffold=empty_scaffold,
        previous_event_ids=[evt.id for evt in world_state.events],
    )
    physics_result: PhysicsExtraction = physics_agent.run_sync(
        prose, deps=physics_deps,
    ).output

    # --- Social extraction (information + relationship edges) ---
    social_agent = _build_social_agent(config)
    social_deps = _SocialDeps(
        global_register=register,
        scaffold=empty_scaffold,
        chunk_event_ids=[evt.id for evt in physics_result.events],
        previous_event_ids=[evt.id for evt in world_state.events],
    )
    social_result: SocialExtraction = social_agent.run_sync(
        prose, deps=social_deps,
    ).output

    topology = ChunkTopology(
        events=physics_result.events,
        causal_topology=physics_result.causal_topology,
        spatial_topology=physics_result.spatial_topology,
        entity_updates=physics_result.entity_updates,
        information_topology=social_result.information_topology,
        social_topology=social_result.social_topology,
    )

    logger.info(
        "Prose extraction complete — %d events, %d causal, %d spatial, "
        "%d info, %d social edges, %d entity updates.",
        len(topology.events), len(topology.causal_topology),
        len(topology.spatial_topology), len(topology.information_topology),
        len(topology.social_topology), len(topology.entity_updates),
    )
    return topology


# ==========================================
# 5. TOPOLOGY MERGE INTO WORLD STATE
# ==========================================

class MergeChangeset(BaseModel):
    """Summary of what a single merge operation added to the world model."""
    events_added: int = 0
    causal_edges_added: int = 0
    spatial_edges_added: int = 0
    information_edges_added: int = 0
    social_edges_added: int = 0
    entity_updates_applied: int = 0
    entity_updates_skipped: List[str] = Field(default_factory=list)


class WorldModelVersion(BaseModel):
    """A single entry in the version history of a VersionedWorldModel."""
    version: int = Field(description="Monotonically increasing version number (0 = original).")
    timestamp: str = Field(description="ISO-8601 timestamp of when this version was created.")
    source: str = Field(
        description="What produced this version, e.g. 'original', 'merge_topology', 'feedback_loop'.",
    )
    description: str = Field(default="", description="Human-readable summary of changes.")
    changeset: Optional[MergeChangeset] = Field(
        default=None,
        description="Detailed counts of what was added. None for version 0 (original).",
    )
    prose: Optional[str] = Field(
        default=None,
        description="Generated/edited prose associated with this version, if any.",
    )


class WorldSnapshot(BaseModel):
    """A full deep-copy of a WorldStateV1 at a specific version."""
    version: int = Field(description="Version number this snapshot corresponds to.")
    world_state: WorldStateV1


class VersionedWorldModel(BaseModel):
    """Immutable-history wrapper around WorldStateV1.

    Every mutation produces a new deep-copied ``WorldStateV1`` and appends
    a version record.  The original world state is never modified.

    The ``snapshots`` list retains full deep-copies of the last *K*
    world states (controlled by ``max_snapshots``).  When the list
    exceeds ``max_snapshots``, the oldest non-original snapshots are
    trimmed.  Version 0 (the original) is always retained.
    """
    current: WorldStateV1
    history: List[WorldModelVersion] = Field(default_factory=list)
    snapshots: List[WorldSnapshot] = Field(
        default_factory=list,
        description="Last-K full world-state deep-copies for rollback/inspection.",
    )
    max_snapshots: int = Field(
        default=10,
        description="Maximum number of snapshots to retain.  Oldest non-original are trimmed first.",
    )

    @staticmethod
    def from_world_state(
        world_state: WorldStateV1,
        *,
        max_snapshots: int = 10,
    ) -> "VersionedWorldModel":
        """Create a new versioned world model from an existing WorldStateV1.

        The incoming ``world_state`` is deep-copied so that subsequent
        merges never mutate the caller's original.
        """
        frozen = copy.deepcopy(world_state)
        return VersionedWorldModel(
            current=copy.deepcopy(frozen),
            history=[
                WorldModelVersion(
                    version=0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="original",
                    description="Initial world state.",
                ),
            ],
            snapshots=[
                WorldSnapshot(version=0, world_state=frozen),
            ],
            max_snapshots=max_snapshots,
        )

    @property
    def version(self) -> int:
        """Current version number."""
        return self.history[-1].version if self.history else 0

    def get_snapshot(self, version: int) -> Optional[WorldSnapshot]:
        """Retrieve a stored snapshot by version number, or None if trimmed."""
        for snap in self.snapshots:
            if snap.version == version:
                return snap
        return None

    @property
    def original(self) -> Optional[WorldStateV1]:
        """The original (version-0) world state, if still retained."""
        snap = self.get_snapshot(0)
        return snap.world_state if snap else None

    def rollback(self, version: int) -> "VersionedWorldModel":
        """Create a new VersionedWorldModel rewound to a stored snapshot.

        The snapshot at *version* becomes the new ``current``.  History is
        truncated to entries <= *version*, and a new "rollback" entry is
        appended so the operation is auditable.  The snapshot list is
        carried forward (trimmed as usual).

        Raises ``KeyError`` if the requested version has been trimmed.
        """
        snap = self.get_snapshot(version)
        if snap is None:
            available = sorted(s.version for s in self.snapshots)
            raise KeyError(
                f"Snapshot for version {version} not available.  "
                f"Retained versions: {available}"
            )

        # History up to (and including) the target version
        kept_history = [h for h in self.history if h.version <= version]

        next_version = self.version + 1
        kept_history.append(WorldModelVersion(
            version=next_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="rollback",
            description=f"Rolled back to version {version}.",
        ))

        rolled_back = copy.deepcopy(snap.world_state)

        # Snapshots: keep everything up to the target, plus add the new current
        new_snapshots = [s for s in self.snapshots if s.version <= version]
        new_snapshots.append(WorldSnapshot(version=next_version, world_state=copy.deepcopy(rolled_back)))
        new_snapshots = self._trim_snapshots(new_snapshots, self.max_snapshots)

        logger.info(
            "[VersionedWorldModel] Rollback v%d → v%d (rewound to v%d).",
            self.version, next_version, version,
        )
        return VersionedWorldModel(
            current=rolled_back,
            history=kept_history,
            snapshots=new_snapshots,
            max_snapshots=self.max_snapshots,
        )

    def _trim_snapshots(self, snapshots: List[WorldSnapshot], max_k: int) -> List[WorldSnapshot]:
        """Trim snapshot list to at most *max_k*, preserving version 0."""
        if len(snapshots) <= max_k:
            return snapshots
        # Partition: keep version 0 always, trim oldest of the rest
        v0 = [s for s in snapshots if s.version == 0]
        rest = [s for s in snapshots if s.version != 0]
        # Keep the most recent (max_k - len(v0)) from rest
        keep = max_k - len(v0)
        if keep <= 0:
            return v0[:max_k]
        trimmed = v0 + rest[-keep:]
        trimmed.sort(key=lambda s: s.version)
        return trimmed

    def merge(
        self,
        topology: "ChunkTopology",
        *,
        source: str = "merge_topology",
        description: str = "",
        prose: Optional[str] = None,
    ) -> "VersionedWorldModel":
        """Merge a topology into the world model, returning a **new** VersionedWorldModel.

        The current ``WorldStateV1`` is deep-copied, the topology is merged
        into the copy, and a new version record is appended.  ``self`` is
        never mutated.
        """
        from shadow_loom.ingestion import (
            ChunkTopology,
            EntityUpdate,
            deduplicate_causal,
            deduplicate_info,
            deduplicate_social,
            deduplicate_spatial,
        )

        merged = copy.deepcopy(self.current)
        changeset = MergeChangeset()

        # --- Events (deduplicate by ID, keep existing) ---
        pre_events = len(merged.events)
        existing_event_ids = {e.id for e in merged.events}
        for evt in topology.events:
            if evt.id not in existing_event_ids:
                merged.events.append(evt)
                existing_event_ids.add(evt.id)
            else:
                logger.debug(
                    "[VersionedWorldModel·merge] Duplicate event %s — kept existing.",
                    evt.id,
                )
        merged.events.sort(key=lambda e: e.fabula_time)
        changeset.events_added = len(merged.events) - pre_events

        # --- Causal edges ---
        pre_causal = len(merged.causal_topology)
        merged.causal_topology.extend(topology.causal_topology)
        merged.causal_topology = deduplicate_causal(merged.causal_topology)
        merged.causal_topology.sort(key=lambda c: c.fabula_time)
        changeset.causal_edges_added = len(merged.causal_topology) - pre_causal

        # --- Spatial edges ---
        pre_spatial = len(merged.spatial_topology)
        merged.spatial_topology.extend(topology.spatial_topology)
        merged.spatial_topology = deduplicate_spatial(merged.spatial_topology)
        changeset.spatial_edges_added = len(merged.spatial_topology) - pre_spatial

        # --- Information edges ---
        pre_info = len(merged.information_topology)
        merged.information_topology.extend(topology.information_topology)
        merged.information_topology = deduplicate_info(merged.information_topology)
        changeset.information_edges_added = len(merged.information_topology) - pre_info

        # --- Social edges ---
        pre_social = len(merged.social_topology)
        merged.social_topology.extend(topology.social_topology)
        merged.social_topology = deduplicate_social(merged.social_topology)
        changeset.social_edges_added = len(merged.social_topology) - pre_social

        # --- Entity state updates ---
        for eu in topology.entity_updates:
            entity = merged.entities.get(eu.entity_id)
            if entity is None:
                logger.warning(
                    "Entity update for unknown entity '%s' — skipped.", eu.entity_id,
                )
                changeset.entity_updates_skipped.append(eu.entity_id)
                continue
            snap = EntityStateSnapshot(
                fabula_time=eu.fabula_time,
                triggered_by=eu.triggered_by,
                traits=eu.trait_updates,
                beliefs_added=eu.new_beliefs,
                beliefs_invalidated=eu.invalidated_belief_targets,
                status=eu.new_status,
                location_id=eu.new_location_id,
            )
            entity.state_timeline.append(snap)
            entity.state_timeline.sort(key=lambda s: s.fabula_time)
            changeset.entity_updates_applied += 1

        next_version = self.version + 1
        new_history = list(self.history) + [
            WorldModelVersion(
                version=next_version,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=source,
                description=description or f"Merged topology: +{changeset.events_added} events.",
                changeset=changeset,
                prose=prose,
            ),
        ]

        logger.info(
            "[VersionedWorldModel] v%d → v%d: +%d events, +%d causal, "
            "+%d spatial, +%d info, +%d social, %d entity updates (%d skipped).",
            self.version, next_version,
            changeset.events_added, changeset.causal_edges_added,
            changeset.spatial_edges_added, changeset.information_edges_added,
            changeset.social_edges_added, changeset.entity_updates_applied,
            len(changeset.entity_updates_skipped),
        )

        # Build snapshot list: carry forward existing + add current merged state
        new_snapshots = list(self.snapshots) + [
            WorldSnapshot(version=next_version, world_state=copy.deepcopy(merged)),
        ]
        new_snapshots = self._trim_snapshots(new_snapshots, self.max_snapshots)

        return VersionedWorldModel(
            current=merged,
            history=new_history,
            snapshots=new_snapshots,
            max_snapshots=self.max_snapshots,
        )


def merge_topology(
    base: WorldStateV1,
    topology: "ChunkTopology",
) -> WorldStateV1:
    """Convenience function: deep-copy ``base``, merge ``topology``, return new WorldStateV1.

    For full version tracking, use :class:`VersionedWorldModel` instead.
    """
    vwm = VersionedWorldModel.from_world_state(base)
    result = vwm.merge(topology)
    return result.current

