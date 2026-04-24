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
    connected_locations: List[dict]
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
            focus_entities.append(ent.model_dump())
            location_ids.add(ent.location_id)
        else:
            logger.warning("Focus Entity '%s' not found.", f_id)

    if not focus_entities:
        raise ValueError(f"None of the focus entities {focus_entity_ids} found in WorldState.")

    # Also pull 1-hop neighbor locations so spatial connectivity can be wired
    neighbor_location_ids: Set[str] = set()
    for loc_id in location_ids:
        loc = world_state.locations.get(loc_id)
        if loc:
            current_locations.append(loc.model_dump())
            for neighbor_id in loc.connected_locations:
                if neighbor_id not in location_ids:
                    neighbor_location_ids.add(neighbor_id)

    for n_loc_id in neighbor_location_ids:
        n_loc = world_state.locations.get(n_loc_id)
        if n_loc:
            current_locations.append(n_loc.model_dump())

    all_location_ids = location_ids | neighbor_location_ids

    # 2. The Spatial Filter (Who/What else is in ANY of these rooms?)
    present_entities = []
    present_entity_ids: Set[str] = set()
    focus_id_set = set(focus_entity_ids)

    for ent_id, entity in world_state.entities.items():
        # If they are in one of the active rooms AND not already a focus entity
        if entity.location_id in location_ids and ent_id not in focus_id_set:
            present_entities.append(entity.model_dump())
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
            relevant_relationships.append(edge.model_dump())

    # 4. The Temporal Filter (Memory & Time-Slicing)
    valid_events = world_state.events
    if temporal_anchor is not None:
        valid_events = [evt for evt in valid_events if evt.fabula_time <= temporal_anchor]

    valid_events.sort(key=lambda x: x.fabula_time, reverse=True)
    recent_memory = [evt.model_dump() for evt in valid_events[:memory_limit]]

    # 5. The Spatial Navigation Filter (Location-to-Location connectivity)
    connected_locations_list = []
    for loc_id in all_location_ids:
        loc = world_state.locations.get(loc_id)
        if not loc:
            continue
        for neighbor_id in loc.connected_locations:
            if neighbor_id in all_location_ids:
                connected_locations_list.append({
                    "source_id": loc_id,
                    "target_id": neighbor_id,
                })

    # 6. The Causal Filter (CausalEdges where both endpoints are in the scene)
    scene_node_ids = (
        focus_id_set
        | present_entity_ids
        | all_location_ids
        | {obj["id"] for obj in present_objects}
        | {evt["id"] for evt in recent_memory}
    )
    relevant_causal_edges = []
    for edge in world_state.causal_topology:
        if edge.source_id in scene_node_ids and edge.target_id in scene_node_ids:
            relevant_causal_edges.append(edge.model_dump())

    payload = EgoGraphPayload(
        focus_entities=focus_entities,
        current_locations=current_locations,
        present_entities=present_entities,
        present_objects=present_objects,
        relevant_relationships=relevant_relationships,
        relevant_causal_edges=relevant_causal_edges,
        connected_locations=connected_locations_list,
        recent_memory=recent_memory
    )

    logger.info("Multi-Ego GraphRAG complete — %d focus, %d locations, %d co-present, %d objects, %d relationships, %d causal, %d spatial, %d memory",
                 len(focus_entities), len(current_locations), len(present_entities),
                 len(present_objects), len(relevant_relationships), len(relevant_causal_edges),
                 len(connected_locations_list), len(recent_memory))
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
        logger.info("Omniscient Graph extracted \u2014 %d entities, %d locations, %d/%d events (anchor T=%d)",
                     len(dump["entities"]), len(dump["locations"]),
                     len(dump["events"]), pre_count, temporal_anchor)
    else:
        logger.info("Omniscient Graph extracted \u2014 %d entities, %d locations, %d events (no anchor)",
                     len(dump["entities"]), len(dump["locations"]), len(dump["events"]))

    return dump

