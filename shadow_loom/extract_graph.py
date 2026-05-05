# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

from shadow_loom.models import (
    EntityStateSnapshot,
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
)

if TYPE_CHECKING:
    from shadow_loom.ingestion import ChunkTopology, ExtractionConfig

from shadow_loom._agent_logging import log_agent_output

logger = logging.getLogger(__name__)


# ==========================================
# 0. HELPERS
# ==========================================
def _time_slice_relationship_at(rel: Any, t: int) -> Optional[dict]:
    """Return a per-axis time-sliced copy of a relationship, or None.

    Per the per-axis refactor, each metric carries its own
    ``last_updated_fabula`` — an edge whose ``power_dynamic`` was last
    touched at T=2 but whose ``fear`` was touched at T=10 should still
    expose its affinity / power axes when sliced at T=5. The previous
    implementation keyed the whole-edge in/out decision off
    ``max(per-axis last_updated_fabula)``, dropping the entire edge if
    *any* axis was newer than the anchor — recreating the cross-axis
    discard bug the refactor was meant to eliminate.

    Behaviour:
      * ``rel`` may be a :class:`RelationshipEdge` instance or a dict
        produced by ``model_dump()`` / ``to_legacy_dict()``.
      * Axes whose ``last_updated_fabula > t`` are filtered out (the
        post-anchor mutation hasn't happened yet from t's perspective).
      * If every axis is filtered out the edge is dropped (return
        ``None``); otherwise a ``to_legacy_dict``-shaped copy is returned
        with the surviving axes only and the flat aggregate keys
        recomputed from the surviving subset.
    """
    if hasattr(rel, "to_legacy_dict"):
        data = rel.to_legacy_dict()
    elif isinstance(rel, dict):
        data = copy.deepcopy(rel)
    else:
        return None

    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    surviving: Dict[str, dict] = {}
    for name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        if m.get("last_updated_fabula", 0) <= t:
            surviving[name] = m

    if not surviving:
        return None

    data["metrics"] = surviving
    # Recompute the flat aggregate keys from the surviving subset so
    # legacy readers see a consistent view.
    for axis in ("affinity", "fear", "power_dynamic"):
        m = surviving.get(axis)
        data[axis] = float(m.get("value", 0.0)) if m and m.get("observed", True) else 0.0
    inertias = [m.get("inertia", 0.3) for m in surviving.values() if isinstance(m, dict)]
    data["inertia"] = min(inertias) if inertias else 0.3
    es_order = {"weak": 0, "moderate": 1, "strong": 2}
    rev = ("weak", "moderate", "strong")
    es_levels = [
        es_order.get(m.get("evidence_strength", "moderate"), 1)
        for m in surviving.values() if isinstance(m, dict)
    ]
    data["evidence_strength"] = rev[max(es_levels)] if es_levels else "moderate"
    data["last_updated_fabula"] = max(
        (m.get("last_updated_fabula", 0) for m in surviving.values() if isinstance(m, dict)),
        default=0,
    )
    return data


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
    relevant_channels: List[dict]
    relevant_utterance_events: List[dict] = Field(default_factory=list)
    recent_memory: List[dict]
    world_traits: List[dict] = Field(default_factory=list)

# ==========================================
# 2. THE IN-MEMORY EXTRACTION FUNCTION
# ==========================================
def extract_ego_graph_from_memory(
    world_state: WorldStateV1,
    focus_entity_ids: List[str],
    temporal_anchor: Optional[int] = None,
    memory_limit: int = 5,
    syuzhet_anchor: Optional[int] = None,
) -> EgoGraphPayload:
    
    """
    Calculates the union of localized Ego-Graphs for multiple entities.

    The ``syuzhet_anchor`` (if provided) gates events by the reader's
    position in the text: events whose ``syuzhet_index`` exceeds the
    anchor have not yet been narrated and must not appear in
    ``recent_memory`` or ``relevant_utterance_events``. The
    ``temporal_anchor`` (fabula time) gates by chronological time.
    Both filters are applied independently — an event must pass both
    to be included.
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
            # Time-slice per-axis: keep the edge if any axis was observed
            # at or before the anchor; drop axes whose own
            # last_updated_fabula > anchor.
            if temporal_anchor is not None:
                sliced = _time_slice_relationship_at(edge, temporal_anchor)
                if sliced is None:
                    logger.debug("[EgoGraph] Excluded relationship %s→%s: no axis observed at or before anchor=%d",
                                 edge.source_entity_id, edge.target_entity_id, temporal_anchor)
                    continue
                relevant_relationships.append(sliced)
            else:
                relevant_relationships.append(edge.to_legacy_dict())

    # 4. The Temporal Filter (Memory & Time-Slicing)
    valid_events = world_state.events
    if temporal_anchor is not None:
        valid_events = [evt for evt in valid_events if evt.fabula_time <= temporal_anchor]
    if syuzhet_anchor is not None:
        valid_events = [evt for evt in valid_events if evt.syuzhet_index <= syuzhet_anchor]

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

    # 6. The Channel Filter (standing comms capabilities involving focus entities)
    relevant_channels: List[dict] = []
    for cid, ch in world_state.channels.items():
        # Participant filter: any focus entity must be a participant
        if not set(ch.participant_ids) & focus_id_set:
            continue
        # Temporal filter: channel must have been established by the anchor
        if temporal_anchor is not None and ch.established_at_fabula > temporal_anchor:
            continue
        if ch.terminated_at_fabula is not None:
            if temporal_anchor is not None and ch.terminated_at_fabula <= temporal_anchor:
                logger.debug("[EgoGraph] Excluded channel %s: terminated_at_fabula=%d <= anchor=%d",
                             cid, ch.terminated_at_fabula, temporal_anchor)
                continue
            if temporal_anchor is None:
                logger.debug("[EgoGraph] Excluded channel %s: terminated (no anchor)", cid)
                continue
        relevant_channels.append(ch.model_dump())

    # 6b. Utterance events involving focus entities (time-sliced).
    relevant_utterance_events: List[dict] = []
    for evt in world_state.events:
        if evt.event_type != "utterance":
            continue
        if temporal_anchor is not None and evt.fabula_time > temporal_anchor:
            continue
        if syuzhet_anchor is not None and evt.syuzhet_index > syuzhet_anchor:
            continue
        participants = set(evt.actor_ids) | set(evt.target_ids) | set(evt.addressee_ids)
        if evt.speaker_id:
            participants.add(evt.speaker_id)
        if not participants & focus_id_set:
            continue
        relevant_utterance_events.append(evt.model_dump())

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
        relevant_channels=relevant_channels,
        relevant_utterance_events=relevant_utterance_events,
        recent_memory=recent_memory,
        world_traits=world_traits_payload,
    )

    logger.info("Multi-Ego GraphRAG complete — %d focus, %d locations, %d co-present, %d objects, %d relationships, %d causal, %d spatial, %d channels, %d utterances, %d memory",
                 len(focus_entities), len(current_locations), len(present_entities),
                 len(present_objects), len(relevant_relationships), len(relevant_causal_edges),
                 len(relevant_spatial_edges), len(relevant_channels),
                 len(relevant_utterance_events), len(recent_memory))
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
    time-sliced to only include events / topology valid at or before
    *temporal_anchor*.

    The slicing rules mirror those in :func:`extract_ego_graph_from_memory`
    so omniscient and ego views remain consistent at the same anchor:

      * ``events``: ``fabula_time <= t``
      * ``causal_topology``: ``fabula_time <= t``
      * ``social_topology``: ``last_updated_fabula <= t``
      * ``spatial_topology``: established by ``t`` and not yet destroyed at ``t``
      * ``channels``: established by ``t`` and not yet terminated at ``t``

    Without an anchor, only dead/terminated information edges are pruned;
    every other list comes through untouched.
    """
    dump = world_state.model_dump()

    if temporal_anchor is not None:
        t = temporal_anchor
        pre_count = len(dump["events"])
        dump["events"] = [
            evt for evt in dump["events"] if evt["fabula_time"] <= t
        ]
        dump["causal_topology"] = [
            ce for ce in dump.get("causal_topology", [])
            if ce.get("fabula_time", 0) <= t
        ]
        dump["social_topology"] = [
            sliced for sliced in (
                _time_slice_relationship_at(rel, t)
                for rel in dump.get("social_topology", [])
            )
            if sliced is not None
        ]
        dump["spatial_topology"] = [
            se for se in dump.get("spatial_topology", [])
            if se.get("established_at_fabula", 0) <= t
            and (
                se.get("destroyed_at_fabula") is None
                or se["destroyed_at_fabula"] > t
            )
        ]
        dump["channels"] = {
            cid: ch for cid, ch in dump.get("channels", {}).items()
            if ch.get("established_at_fabula", 0) <= t
            and (
                ch.get("terminated_at_fabula") is None
                or ch["terminated_at_fabula"] > t
            )
        }
        logger.info(
            "Omniscient Graph extracted — %d entities, %d locations, "
            "%d/%d events, %d causal, %d social, %d spatial, %d channels "
            "(anchor T=%d)",
            len(dump["entities"]), len(dump["locations"]),
            len(dump["events"]), pre_count,
            len(dump["causal_topology"]), len(dump["social_topology"]),
            len(dump["spatial_topology"]), len(dump["channels"]),
            t,
        )
    else:
        # Without an anchor, exclude terminated channels (dead links)
        dump["channels"] = {
            cid: ch for cid, ch in dump.get("channels", {}).items()
            if ch.get("terminated_at_fabula") is None
        }
        logger.info("Omniscient Graph extracted — %d entities, %d locations, %d events (no anchor)",
                     len(dump["entities"]), len(dump["locations"]), len(dump["events"]))

    return dump


# ==========================================
# 4. PROSE → TOPOLOGY EXTRACTION
# ==========================================

def promote_sandbox_spawns(
    world_state: WorldStateV1,
    physics_state: Dict[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    """Promote ``world_id="shadow"`` spawn nodes from the sandbox into
    typed canonical-world records.

    The Causal Engine handles ``<ID>.spawn`` interventions by adding
    new nodes to a Rung-2/3 sandbox tagged ``world_id="shadow"`` (see
    :func:`shadow_loom.instantiator._intervene_genesis`). Those nodes
    never enter the canonical :class:`WorldStateV1` unless we promote
    them here, *before* prose re-extraction runs — otherwise the
    re-extractor's :class:`GlobalRegister` will not know the new IDs
    and will either drop references or invent duplicate IDs.

    Returns a dict ``{"entities": {...}, "objects": {...},
    "locations": {...}, "world_traits": {...}}`` keyed by canonical
    ID. Caller is expected to attach these to a :class:`ChunkTopology`
    via its ``new_*`` fields and pre-register them in the
    extraction-time register.

    Channels (``Channel`` capabilities) are *not* promoted here because
    the social agent already creates standing :class:`Channel`
    capabilities from prose; sandbox channel spawns flow through
    ``ChunkTopology.channels`` in the normal merge path.

    The function never raises — malformed sandbox payloads are skipped
    and logged. Existing canonical IDs are skipped (idempotent).
    """
    from shadow_loom.models import (
        Entity,
        GlobalTrait,
        Location,
        NarrativeObject,
        TraitVector,
    )

    out: Dict[str, Dict[str, Any]] = {
        "entities": {},
        "objects": {},
        "locations": {},
        "world_traits": {},
    }
    if not physics_state:
        return out

    nodes = physics_state.get("nodes") or []
    for node in nodes:
        try:
            if node.get("world_id") != "shadow":
                continue
            node_id = node.get("id")
            if not node_id:
                continue
            node_type = (node.get("node_type") or "").strip()

            if node_type == "Entity":
                if node_id in world_state.entities or node_id in out["entities"]:
                    continue
                name = node.get("name") or node_id
                location_id = node.get("located_in") or node.get("location_id")
                if not location_id:
                    # Entity requires a location — skip if we cannot infer one.
                    logger.warning(
                        "[promote_sandbox_spawns] Skipping entity %s — no location_id.",
                        node_id,
                    )
                    continue
                ent = Entity(
                    id=node_id,
                    name=name,
                    location_id=location_id,
                    status=node.get("status", "healthy"),
                    traits=node.get("traits", {}) or {},
                    beliefs=node.get("beliefs", []) or [],
                    constants=node.get("constants", []) or [],
                    state_timeline=node.get("state_timeline", []) or [],
                )
                out["entities"][node_id] = ent

            elif node_type == "NarrativeObject":
                if node_id in world_state.objects or node_id in out["objects"]:
                    continue
                obj = NarrativeObject(
                    id=node_id,
                    name=node.get("name") or node_id,
                    location_id=node.get("location_id") or node.get("located_in"),
                    owner_id=node.get("owner_id"),
                    properties=node.get("properties", {}) or {},
                    affordances=node.get("affordances", []) or [],
                )
                out["objects"][node_id] = obj

            elif node_type == "Location":
                if node_id in world_state.locations or node_id in out["locations"]:
                    continue
                # Location's id is the dict key, not a model field.
                loc = Location(
                    name=node.get("name") or node_id,
                    description=node.get("description", "") or "",
                    ambient_state=node.get("ambient_state", {}) or {},
                )
                out["locations"][node_id] = loc

            elif node_type in ("WorldTrait", "GlobalTrait"):
                if node_id in world_state.world_traits or node_id in out["world_traits"]:
                    continue
                # GlobalTrait requires category + magnitude — supply
                # neutral defaults if the sandbox payload omitted them.
                magnitude_payload = node.get("magnitude")
                if isinstance(magnitude_payload, TraitVector):
                    magnitude = magnitude_payload
                elif isinstance(magnitude_payload, dict):
                    magnitude = TraitVector(**magnitude_payload)
                else:
                    magnitude = TraitVector(
                        value=0.5, inertia=0.5, evidence_strength="moderate",
                    )
                wt = GlobalTrait(
                    id=node_id,
                    name=node.get("name") or node_id,
                    description=node.get("description", "") or "",
                    category=node.get("category", "social_structure"),
                    magnitude=magnitude,
                    affected_domains=node.get("affected_domains", []) or [],
                    state_timeline=node.get("state_timeline", []) or [],
                )
                out["world_traits"][node_id] = wt
            # Channels and EventNodes are handled by the normal social /
            # physics extraction path; nothing to do here.
        except Exception:
            logger.exception(
                "[promote_sandbox_spawns] Failed to promote sandbox node %r — skipped.",
                node.get("id"),
            )
            continue

    if any(out[k] for k in out):
        logger.info(
            "[promote_sandbox_spawns] Promoted spawns — entities=%d, objects=%d, "
            "locations=%d, world_traits=%d.",
            len(out["entities"]), len(out["objects"]),
            len(out["locations"]), len(out["world_traits"]),
        )
    return out


def extract_topology_from_prose(
    prose: str,
    world_state: WorldStateV1,
    config: "ExtractionConfig | None" = None,
    *,
    spawns: Optional[Dict[str, Dict[str, Any]]] = None,
) -> "ChunkTopology":
    """Extract graph topology from generated prose using the existing physics + social agents.

    Builds a ``GlobalRegister`` directly from the ``WorldStateV1`` (no LLM
    ontology extraction needed — the entities/locations/objects are already
    known) and then runs the physics and social extraction agents on the
    prose as a single chunk.

    If ``spawns`` is provided (output of :func:`promote_sandbox_spawns`),
    the new entities / objects / locations / world_traits are pre-registered
    in the ``GlobalRegister`` so the extractor LLMs use the canonical
    spawn IDs instead of inventing duplicates, and the resulting
    :class:`ChunkTopology` carries them on its ``new_*`` fields so
    ``VersionedWorldModel.merge`` records the genesis in the changeset.

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

    # Merge sandbox spawns into the registry so the LLM agents see them
    # as canonical / known entities and refer to them by their spawn IDs.
    spawns = spawns or {}
    locations_known = {**world_state.locations, **spawns.get("locations", {})}
    objects_known = {**world_state.objects, **spawns.get("objects", {})}
    entities_known = {**world_state.entities, **spawns.get("entities", {})}
    world_traits_known = {**world_state.world_traits, **spawns.get("world_traits", {})}

    register = GlobalRegister(
        locations=locations_known,
        objects=objects_known,
        entities=entities_known,
        world_traits=world_traits_known,
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
    log_agent_output(logger, "PhysicsExtraction", physics_result)

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
    log_agent_output(logger, "SocialExtraction", social_result)

    topology = ChunkTopology(
        events=physics_result.events + social_result.utterance_events,
        causal_topology=physics_result.causal_topology,
        spatial_topology=physics_result.spatial_topology,
        entity_updates=physics_result.entity_updates,
        channels=social_result.channels,
        social_topology=social_result.social_topology,
        new_entities=spawns.get("entities", {}),
        new_objects=spawns.get("objects", {}),
        new_locations=spawns.get("locations", {}),
        new_world_traits=spawns.get("world_traits", {}),
    )

    logger.info(
        "Prose extraction complete — %d events, %d causal, %d spatial, "
        "%d channels, %d social edges, %d entity updates.",
        len(topology.events), len(topology.causal_topology),
        len(topology.spatial_topology), len(topology.channels),
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
    information_edges_added: int = 0  # retained for backwards-compat name in summaries
    social_edges_added: int = 0
    entity_updates_applied: int = 0
    entity_updates_skipped: List[str] = Field(default_factory=list)
    # Genesis-promoted nodes — populated when the upstream topology
    # carries ``new_entities`` / ``new_objects`` / ``new_locations``
    # / ``new_world_traits`` from a sandbox spawn. Default zero so
    # plain ingestion merges still serialize unchanged.
    entities_added: int = 0
    objects_added: int = 0
    locations_added: int = 0
    world_traits_added: int = 0


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
    world_id: Literal["factual", "shadow"] = Field(
        default="factual",
        description=(
            "AMWN branch this version belongs to. 'factual' = canonical "
            "mainline; 'shadow' = counterfactual fork. Set by the pipeline "
            "based on PipelineConfig.branch_policy. All AMWN nodes/edges "
            "added in this version inherit this tag."
        ),
    )
    branch_label: Optional[str] = Field(
        default=None,
        description=(
            "Optional human-readable name for the branch this version sits on "
            "(e.g. 'What if Duncan lived'). Typically only set on the first "
            "version of a shadow fork."
        ),
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
        world_id: Literal["factual", "shadow"] = "factual",
        branch_label: Optional[str] = None,
    ) -> "VersionedWorldModel":
        """Merge a topology into the world model, returning a **new** VersionedWorldModel.

        The current ``WorldStateV1`` is deep-copied, the topology is merged
        into the copy, and a new version record is appended.  ``self`` is
        never mutated.

        ``world_id`` tags all nodes/edges added by this merge (events,
        causal/spatial/social edges, channels, entity-state snapshots) so
        downstream readers can filter by AMWN branch. ``branch_label`` is
        a human-readable name carried on the version record (typically
        only set on the first version of a shadow fork).
        """
        from shadow_loom.ingestion import (
            ChunkTopology,
            EntityUpdate,
            _apply_channel_forwarding,
            _deduplicate_channels_with_map,
            deduplicate_causal,
            deduplicate_social,
            deduplicate_spatial,
        )

        merged = copy.deepcopy(self.current)
        # Deep-copy the incoming topology so re-tagging world_id below
        # never mutates the caller's object.
        topology = copy.deepcopy(topology)
        if world_id != "factual":
            for evt in topology.events:
                evt.world_id = world_id
            for ce in topology.causal_topology:
                ce.world_id = world_id
            for se in topology.spatial_topology:
                se.world_id = world_id
            for re in topology.social_topology:
                re.world_id = world_id
            for ch in topology.channels.values():
                ch.world_id = world_id
            for ent in topology.new_entities.values():
                ent.world_id = world_id
            for obj in topology.new_objects.values():
                obj.world_id = world_id
            for loc in topology.new_locations.values():
                loc.world_id = world_id
            for wt in topology.new_world_traits.values():
                wt.world_id = world_id
        changeset = MergeChangeset()

        # --- Genesis-promoted nodes (entities / objects / locations /
        # world_traits) — written first so subsequent edge / event /
        # entity_update merges can reference them. Existing IDs are
        # left untouched; only brand-new ones are added.
        for eid, ent in topology.new_entities.items():
            if eid in merged.entities:
                logger.debug(
                    "[VersionedWorldModel·merge] Spawn entity %s already exists — kept existing.",
                    eid,
                )
                continue
            merged.entities[eid] = ent
            changeset.entities_added += 1
        for oid, obj in topology.new_objects.items():
            if oid in merged.objects:
                continue
            merged.objects[oid] = obj
            changeset.objects_added += 1
        for lid, loc in topology.new_locations.items():
            if lid in merged.locations:
                continue
            merged.locations[lid] = loc
            changeset.locations_added += 1
        for wid, wt in topology.new_world_traits.items():
            if wid in merged.world_traits:
                continue
            merged.world_traits[wid] = wt
            changeset.world_traits_added += 1

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

        # --- Channels (with forwarding map for via_channel_id rewrites) ---
        pre_info = len(merged.channels)
        merged_channels, channel_forwarding = _deduplicate_channels_with_map(
            [merged.channels, topology.channels],
        )
        merged.channels = merged_channels
        if channel_forwarding:
            # Rewrite stale via_channel_id on every event in the merged
            # world AND on belief provenance carried by the incoming
            # entity_updates so the snapshots created below pick up the
            # canonical id.
            _apply_channel_forwarding(channel_forwarding, events=merged.events)
            _apply_channel_forwarding(
                channel_forwarding,
                events=[],
                entity_updates=topology.entity_updates,
            )
        changeset.information_edges_added = len(merged.channels) - pre_info

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
                world_id=world_id,
                branch_label=branch_label,
            ),
        ]

        logger.info(
            "[VersionedWorldModel] v%d → v%d: +%d events, +%d causal, "
            "+%d spatial, +%d info, +%d social, %d entity updates (%d skipped), "
            "+%d entities, +%d objects, +%d locations, +%d world_traits.",
            self.version, next_version,
            changeset.events_added, changeset.causal_edges_added,
            changeset.spatial_edges_added, changeset.information_edges_added,
            changeset.social_edges_added, changeset.entity_updates_applied,
            len(changeset.entity_updates_skipped),
            changeset.entities_added, changeset.objects_added,
            changeset.locations_added, changeset.world_traits_added,
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

