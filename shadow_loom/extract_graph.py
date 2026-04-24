from typing import List, Optional, Set
import logging
from pydantic import BaseModel, Field

from shadow_loom.models import WorldStateV1

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
                ent_data["beliefs"] = [
                    b for b in ent_data.get("beliefs", [])
                    if b.get("established_at_fabula", 0) <= temporal_anchor
                ]
            focus_entities.append(ent_data)
            location_ids.add(ent.location_id)
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
        # If they are in one of the active rooms AND not already a focus entity
        if entity.location_id in location_ids and ent_id not in focus_id_set:
            ent_data = entity.model_dump()
            if temporal_anchor is not None:
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
    for edge in world_state.social_topology:
        # Relationships starting from our Focus Entities toward anyone in the scene
        if edge.source_entity_id in focus_id_set and (
            edge.target_entity_id in present_entity_ids or edge.target_entity_id in focus_id_set
        ):
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
    scene_node_ids = (
        focus_id_set
        | present_entity_ids
        | all_location_ids
        | {obj["id"] for obj in present_objects}
        | {evt["id"] for evt in recent_memory}
    )
    relevant_causal_edges = []
    for edge in world_state.causal_topology:
        if edge.source_event_id in scene_node_ids and edge.target_node_id in scene_node_ids:
            relevant_causal_edges.append(edge.model_dump())

    payload = EgoGraphPayload(
        focus_entities=focus_entities,
        current_locations=current_locations,
        present_entities=present_entities,
        present_objects=present_objects,
        relevant_relationships=relevant_relationships,
        relevant_causal_edges=relevant_causal_edges,
        relevant_spatial_edges=relevant_spatial_edges,
        relevant_information_edges=relevant_information_edges,
        recent_memory=recent_memory
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

