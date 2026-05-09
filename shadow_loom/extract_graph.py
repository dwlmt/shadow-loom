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
        if wt.state_timeline:
            # Resolve at ``temporal_anchor`` when supplied; otherwise pin to
            # the *latest* snapshot so anchor-less callers (notably shadow
            # merges and full-world dashboards) see the most recent shifted
            # value rather than the static baseline. The instantiator's
            # section H reads this same magnitude when generating
            # auto-ambient WORLD_ → Entity edges, so anchor-less callers
            # would otherwise propagate a stale baseline force.
            target_ft = temporal_anchor
            if target_ft is None:
                target_ft = max(int(s.fabula_time) for s in wt.state_timeline)
            reconstructed = reconstruct_world_trait_at(wt, int(target_ft))
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
    syuzhet_anchor: Optional[int] = None,
) -> dict:
    """
    Serialises the entire WorldStateV1 as a dictionary, optionally
    time-sliced to only include events / topology valid at or before
    *temporal_anchor*.

    The slicing rules mirror those in :func:`extract_ego_graph_from_memory`
    so omniscient and ego views remain consistent at the same anchor:

      * ``events``: ``fabula_time <= t`` AND, when ``syuzhet_anchor`` is
        provided, ``syuzhet_index <= s`` (events not yet narrated must
        not leak into the omniscient prompt — without this gate the
        renderer and auditor see future utterances and revelations).
      * ``causal_topology``: ``fabula_time <= t``
      * ``social_topology``: ``last_updated_fabula <= t``
      * ``spatial_topology``: established by ``t`` and not yet destroyed at ``t``
      * ``channels``: established by ``t`` and not yet terminated at ``t``

    Without an anchor, only dead/terminated information edges are pruned;
    every other list comes through untouched.

    The returned dict carries ``syuzhet_anchor`` at the top level when
    provided so downstream consumers (the deterministic withheld-utterance
    leak check) can locate the brief on the timeline without re-deriving
    it from event indices.
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

    if syuzhet_anchor is not None:
        s = syuzhet_anchor
        pre_evt = len(dump["events"])
        dump["events"] = [
            evt for evt in dump["events"]
            if evt.get("syuzhet_index", 0) <= s
        ]
        dump["syuzhet_anchor"] = s
        logger.info(
            "Omniscient Graph syuzhet-pruned: %d/%d events kept "
            "(syuzhet_anchor=%d)",
            len(dump["events"]), pre_evt, s,
        )

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
    "locations": {...}, "world_traits": {...}, "channels": {...}}``
    keyed by canonical ID. Caller is expected to attach these to a
    :class:`ChunkTopology` via its ``new_*`` fields (or
    ``ChunkTopology.channels`` for spawned :class:`Channel`
    capabilities) and pre-register them in the extraction-time
    register.

    :class:`Channel` spawns are promoted here too so that intervention
    queries that introduce a new standing capability (e.g. a witness
    placed within earshot, a new courier route, a magic mind-link
    formed mid-story) are persisted even when the renderer never
    surfaced the channel by name in prose. The social agent's
    re-extraction path will collapse duplicates if it does mention
    the channel.

    The function never raises — malformed sandbox payloads are skipped
    and logged. Existing canonical IDs are skipped (idempotent).
    """
    from shadow_loom.models import (
        Channel,
        Concern,
        Entity,
        GlobalTrait,
        Location,
        NarrativeObject,
        Proposition,
        TraitVector,
    )

    out: Dict[str, Dict[str, Any]] = {
        "entities": {},
        "objects": {},
        "locations": {},
        "world_traits": {},
        "channels": {},
        "propositions": {},
        # entity_id → list[Concern] so the merge step can attach each
        # spawned concern to its holder.
        "concerns": {},
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
            elif node_type == "Channel":
                # Spawned standing communication capability. Skip if the
                # canonical world already knows it (idempotent re-runs)
                # or if a prior promotion in the same call already
                # captured it.
                if node_id in (world_state.channels or {}) or node_id in out["channels"]:
                    continue
                participant_ids = (
                    node.get("participant_ids")
                    or node.get("participants")
                    or []
                )
                if not participant_ids:
                    logger.warning(
                        "[promote_sandbox_spawns] Skipping channel %s \u2014 no participants.",
                        node_id,
                    )
                    continue
                try:
                    ch = Channel(
                        id=node_id,
                        name=node.get("name") or node_id,
                        medium=node.get("medium") or "unspecified",
                        participant_ids=list(participant_ids),
                        directionality=node.get("directionality", "duplex"),
                        intelligibility=node.get("intelligibility", {}) or {},
                        established_at_fabula=int(node.get("established_at_fabula", 0)),
                        terminated_at_fabula=node.get("terminated_at_fabula"),
                        evidence_strength=node.get("evidence_strength", "moderate"),
                        world_id=node.get("world_id", "shadow"),
                    )
                except Exception:
                    logger.exception(
                        "[promote_sandbox_spawns] Channel %s payload invalid \u2014 skipped.",
                        node_id,
                    )
                    continue
                out["channels"][node_id] = ch
            elif node_type == "Proposition":
                if node_id in {p.proposition_id for p in world_state.propositions} \
                   or node_id in out["propositions"]:
                    continue
                try:
                    prop = Proposition(
                        world_id=node.get("world_id", "shadow"),
                        proposition_id=node_id,
                        kind=node.get("kind", "outcome"),
                        referent_ids=node.get("referent_ids", []) or [],
                        description=node.get("description", "") or node_id,
                        audience_default_prior=float(node.get("audience_default_prior", 0.5)),
                        stakes=float(node.get("stakes", 0.5)),
                        truth_at_fabula=node.get("truth_at_fabula", {}) or {},
                    )
                except Exception:
                    logger.exception(
                        "[promote_sandbox_spawns] Proposition %s payload invalid \u2014 skipped.",
                        node_id,
                    )
                    continue
                out["propositions"][node_id] = prop
            elif node_type == "Concern":
                holder_id = node.get("holder_id") or node.get("entity_id")
                if not holder_id:
                    logger.warning(
                        "[promote_sandbox_spawns] Skipping concern %s \u2014 no holder_id.",
                        node_id,
                    )
                    continue
                try:
                    concern = Concern(
                        world_id=node.get("world_id", "shadow"),
                        concern_id=node_id,
                        proposition_id=node.get("proposition_id"),
                        polarity=node.get("polarity", "desire"),
                        kind=node.get("kind"),
                        salience=float(node.get("salience", 0.5)),
                        activation_fabula_window=node.get("activation_fabula_window"),
                        counter_concern_ids=node.get("counter_concern_ids", []) or [],
                    )
                except Exception:
                    logger.exception(
                        "[promote_sandbox_spawns] Concern %s payload invalid \u2014 skipped.",
                        node_id,
                    )
                    continue
                out["concerns"].setdefault(holder_id, []).append(concern)
            # EventNodes are handled by the normal physics/social
            # extraction path; nothing to do here.
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
    fabula_time_base: Optional[int] = None,
    fabula_time_spacing: Optional[int] = None,
    branch_world_id: Literal["factual", "shadow"] = "factual",
    branch_label: Optional[str] = None,
    preceding_prose: Optional[str] = None,
    engine_priors: Optional[str] = None,
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

    ``fabula_time_base`` and ``fabula_time_spacing`` are passed to the
    physics agent so the new events land at the correct timeline
    position relative to the existing world. When ``fabula_time_base``
    is ``None`` it auto-resolves to ``max(world_state.events.fabula_time)
    + spacing`` so re-extractions of *continuation* prose append rather
    than clobber existing chronology. Pass an explicit value to insert
    a manual edit at a specific anchor in the timeline.

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

    # ── Resolve fabula timing context for the physics agent ──────────
    # Without this the LLM has no idea where in chronological time the
    # new prose sits and tends to assign fabula_times starting at 0,
    # which collides with existing events.
    spacing = (
        fabula_time_spacing
        if fabula_time_spacing is not None
        else config.fabula_time_spacing
    )
    if fabula_time_base is None:
        existing_max = max(
            (e.fabula_time for e in world_state.events),
            default=-spacing,
        )
        resolved_base = existing_max + spacing
    else:
        resolved_base = fabula_time_base
    prev_max_fabula = max(
        (e.fabula_time for e in world_state.events), default=0,
    )
    physics_msg = (
        f"Re-extraction of generated/edited prose "
        f"(fabula_time_base: {resolved_base}, "
        f"fabula_time_spacing: {spacing}, "
        f"max fabula_time so far: {prev_max_fabula}).\n\n"
        f"Use ABSOLUTE story-world chronology for fabula_time. Place "
        f"new chronological beats at or after fabula_time_base; "
        f"flashbacks may use SMALLER values, flash-forwards LARGER. "
        f"Existing event ids in the world model must keep their "
        f"original fabula_time if you reference them.\n\n"
    )
    # Branch framing — when re-extracting on a shadow fork, surface
    # the branch identity so the physics agent knows the new events
    # belong to a counterfactual world. The merge step will re-tag
    # everything anyway, but telling the agent up-front keeps it from
    # silently reconciling shadow events against factual canon.
    if branch_world_id == "shadow":
        physics_msg += (
            f"AMWN BRANCH CONTEXT: this prose lives on a SHADOW fork "
            f"(label: {branch_label or 'unspecified'}). Treat it as "
            f"the actual lived world on this branch — do NOT try to "
            f"reconcile contradictions with factual canon by dropping "
            f"events. Every new event you extract will be tagged "
            f"world_id=\"shadow\" by the merge step.\n\n"
        )
    if preceding_prose:
        # Surface prior on-branch prose as background continuity so
        # the agent can keep newly-extracted events / channels /
        # snapshots consistent with what came before instead of
        # re-extracting them as duplicates.
        snippet = preceding_prose.strip()
        if len(snippet) > 4000:
            snippet = "\u2026" + snippet[-3999:]
        physics_msg += (
            f"=== STORY SO FAR (prior prose for continuity \u2014 do NOT "
            f"re-extract events from this block) ===\n{snippet}\n\n"
            f"=== NEW PROSE TO EXTRACT FROM ===\n"
        )
    if engine_priors:
        # Engine-declared mutations / social shifts / sandbox spawns
        # are deterministic priors produced by the CausalPhysicsEngine
        # (rung-2 intervention, rung-3 abduction, directive
        # plausibility check). Surface them so the Physics agent
        # extracts events that are CONSISTENT with what the engine
        # already declared, rather than re-deriving an alternate
        # interpretation from prose alone. Without this hint, prose
        # that softly verbalises a hard mutation (e.g. "Anakin's anger
        # finally cracked his discipline") is often mis-extracted as
        # a single mood beat rather than as the trait-flip the engine
        # asserted.
        physics_msg += (
            f"\n=== ENGINE PRIORS (deterministic ground truth from "
            f"the causal engine \u2014 align your event extraction with "
            f"these; do NOT contradict them) ===\n{engine_priors}\n\n"
            f"=== END ENGINE PRIORS ===\n\n"
        )
    physics_msg += prose

    # --- Physics extraction (events + causal + spatial + entity_updates) ---
    physics_agent = _build_physics_agent(config)
    physics_deps = _PhysicsDeps(
        global_register=register,
        scaffold=empty_scaffold,
        previous_event_ids=[evt.id for evt in world_state.events],
    )
    physics_result: PhysicsExtraction = physics_agent.run_sync(
        physics_msg, deps=physics_deps,
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
    # Mirror the ingestion social-pass user message: feed the agent the
    # event summary the physics pass just produced + branch / story-so-far
    # framing so utterances and relationship metrics align with the
    # already-extracted events instead of being re-derived from prose
    # alone (which routinely missed cross-references and produced
    # branch-blind social topology on shadow forks).
    event_summary_lines = [
        f"  - {e.id} (fabula={e.fabula_time}, syuzhet={e.syuzhet_index}, "
        f"type={e.event_type}, actors={e.actor_ids}, targets={e.target_ids}): "
        f"{e.description}"
        for e in physics_result.events
    ]
    event_summary = "\n".join(event_summary_lines) or "  (no events extracted from this prose)"
    social_msg_parts: List[str] = [
        "Re-extraction of generated/edited prose.",
        f"\nEVENTS EXTRACTED FROM THIS PROSE:\n{event_summary}",
    ]
    if branch_world_id == "shadow":
        social_msg_parts.append(
            f"\nAMWN BRANCH CONTEXT: this prose lives on a SHADOW fork "
            f"(label: {branch_label or 'unspecified'}). Channels, "
            f"utterances, and relationship metrics you emit will be "
            f"tagged world_id=\"shadow\" by the merge step. Do not "
            f"reconcile against factual canon."
        )
    if preceding_prose:
        snippet = preceding_prose.strip()
        if len(snippet) > 4000:
            snippet = "\u2026" + snippet[-3999:]
        social_msg_parts.append(
            f"\n=== STORY SO FAR (prior prose for continuity \u2014 do NOT "
            f"re-extract channels/utterances/edges from this block) ===\n{snippet}"
        )
    if engine_priors:
        social_msg_parts.append(
            f"\n=== ENGINE PRIORS (deterministic ground truth from the "
            f"causal engine \u2014 align relationship metrics and any new "
            f"utterances with these; do NOT contradict them) ===\n"
            f"{engine_priors}\n=== END ENGINE PRIORS ==="
        )
    social_msg_parts.append(f"\nORIGINAL TEXT:\n{prose}")
    social_msg = "\n".join(social_msg_parts)
    social_result: SocialExtraction = social_agent.run_sync(
        social_msg, deps=social_deps,
    ).output
    log_agent_output(logger, "SocialExtraction", social_result)

    topology = ChunkTopology(
        events=physics_result.events + social_result.utterance_events,
        causal_topology=physics_result.causal_topology,
        spatial_topology=physics_result.spatial_topology,
        entity_updates=physics_result.entity_updates,
        # Merge social-agent channels with promoted sandbox channel
        # spawns; the social agent's view wins on collisions because
        # it has direct prose evidence, while the sandbox spawn is a
        # deterministic fallback for cases where the renderer never
        # surfaced the new channel by name.
        channels={**spawns.get("channels", {}), **social_result.channels},
        social_topology=social_result.social_topology,
        new_entities=spawns.get("entities", {}),
        new_objects=spawns.get("objects", {}),
        new_locations=spawns.get("locations", {}),
        new_world_traits=spawns.get("world_traits", {}),
        new_propositions=spawns.get("propositions", {}),
        new_concerns=spawns.get("concerns", {}),
    )

    # --- Optional Affect pass (Phase B4 ported to re-extraction) ----
    # When the world has a populated proposition catalogue and/or
    # standing concerns AND the chunk plausibly touches one of them,
    # run the affect agent so that proposition truth commits / framing
    # snapshots and concern drift implied by the generated prose are
    # captured. Without this, a counterfactual that flips a proposition
    # would never persist the truth-flip onto Proposition.truth_at_fabula.
    if config.enable_affect_agent and (
        world_state.propositions
        or any(e.concerns for e in world_state.entities.values())
    ):
        try:
            from shadow_loom.ingestion import (
                _AffectDeps,
                _build_affect_agent,
                _chunk_has_affect_signal,
                ChunkAffectExtraction,
                ConcernSeed,
            )
            catalogue_prop_ids = {p.proposition_id for p in world_state.propositions}
            seeded_entity_ids = {
                eid for eid, e in world_state.entities.items() if e.concerns
            }
            if _chunk_has_affect_signal(
                physics_result.events,
                physics_result.entity_updates,
                catalogue_prop_ids,
                seeded_entity_ids,
            ):
                # Materialise existing concerns as ConcernSeed payloads
                # so the agent sees the catalogue with full polarity /
                # baseline / kind detail.
                seeds: List[ConcernSeed] = []
                for ent_id, ent in world_state.entities.items():
                    for c in ent.concerns:
                        try:
                            seeds.append(ConcernSeed(
                                concern_id=c.concern_id,
                                entity_id=ent_id,
                                proposition_id=c.proposition_id,
                                polarity=c.polarity,
                                kind=c.kind,
                                baseline_salience=c.salience,
                            ))
                        except Exception:
                            continue
                affect_agent = _build_affect_agent(config)
                affect_deps = _AffectDeps(
                    global_register=register,
                    chunk_events=physics_result.events,
                    chunk_entity_updates=physics_result.entity_updates,
                    propositions=list(world_state.propositions),
                    concern_seeds=seeds,
                )
                affect_msg_parts: List[str] = [
                    "Re-extraction of generated/edited prose (affect pass).",
                    f"\nEVENTS EXTRACTED FROM THIS PROSE:\n{event_summary}",
                ]
                if branch_world_id == "shadow":
                    affect_msg_parts.append(
                        f"\nAMWN BRANCH CONTEXT: this prose lives on a SHADOW fork "
                        f"(label: {branch_label or 'unspecified'}). All snapshots "
                        f"and truth commits will be tagged world_id=\"shadow\"."
                    )
                affect_msg_parts.append(f"\nORIGINAL TEXT:\n{prose}")
                affect_msg = "\n".join(affect_msg_parts)
                affect_result: ChunkAffectExtraction = affect_agent.run_sync(
                    affect_msg, deps=affect_deps,
                ).output
                log_agent_output(logger, "ChunkAffectExtraction", affect_result)
                topology.proposition_snapshots.extend(affect_result.proposition_snapshots)
                topology.proposition_truth_commits.extend(affect_result.proposition_truth_commits)
                topology.concern_snapshots.extend(affect_result.concern_snapshots)
                topology.belief_snapshots.extend(
                    getattr(affect_result, "belief_snapshots", []) or []
                )
                topology.new_concern_seeds.extend(affect_result.new_concern_seeds)
        except Exception:
            logger.exception(
                "[extract_topology_from_prose] Affect pass failed \u2014 continuing without affect deltas."
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
    # Affect-layer additions (Phase 2026-05-07: prose-merge completeness).
    propositions_added: int = 0
    proposition_truths_committed: int = 0
    proposition_snapshots_added: int = 0
    concerns_added: int = 0
    concern_snapshots_added: int = 0
    belief_snapshots_added: int = 0
    belief_confidence_updates_applied: int = 0
    # Deletion counters (deletion pass runs before additive sections).
    events_removed: int = 0
    causal_edges_removed: int = 0
    spatial_edges_removed: int = 0
    social_edges_removed: int = 0
    channels_removed: int = 0
    entities_removed: int = 0
    objects_removed: int = 0
    locations_removed: int = 0
    world_traits_removed: int = 0
    propositions_removed: int = 0
    concerns_removed: int = 0
    events_superseded: int = 0


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


# ==========================================
# Genesis-node attribute backfill helpers
# ==========================================
# When a later chunk re-extracts a node that already exists in the
# merged world, we keep the original identity (id, location_id,
# status, owner_id, magnitude) but fill in any *missing* attributes
# from the incoming record. These helpers are intentionally
# additive-only — never overwrite a populated field — so the
# canonical first observation wins on every conflict.

def _prefer_longer(existing: str, incoming: str) -> str:
    """Return the longer non-empty string, preferring ``existing`` on ties."""
    if not incoming:
        return existing
    if not existing:
        return incoming
    return incoming if len(incoming) > len(existing) else existing


def _backfill_entity(existing, incoming):
    """Non-destructive merge of two ``Entity`` records sharing an id."""
    update: Dict[str, Any] = {}
    new_name = _prefer_longer(getattr(existing, "name", ""), getattr(incoming, "name", ""))
    if new_name != existing.name:
        update["name"] = new_name
    # Traits: existing wins on key conflict; add any new keys.
    new_traits = dict(existing.traits)
    for k, v in incoming.traits.items():
        if k not in new_traits:
            new_traits[k] = v
    if len(new_traits) != len(existing.traits):
        update["traits"] = new_traits
    # Beliefs: dedup by target_id, existing wins on conflict.
    existing_belief_targets = {b.target_id for b in existing.beliefs}
    new_beliefs = list(existing.beliefs) + [
        b for b in incoming.beliefs if b.target_id not in existing_belief_targets
    ]
    if len(new_beliefs) != len(existing.beliefs):
        update["beliefs"] = new_beliefs
    # Constants: set-union, preserve order.
    existing_constants = list(existing.constants)
    seen = set(existing_constants)
    for c in incoming.constants:
        if c not in seen:
            existing_constants.append(c)
            seen.add(c)
    if len(existing_constants) != len(existing.constants):
        update["constants"] = existing_constants
    return existing.model_copy(update=update) if update else existing


def _backfill_object(existing, incoming):
    """Non-destructive merge of two ``NarrativeObject`` records sharing an id."""
    update: Dict[str, Any] = {}
    new_name = _prefer_longer(existing.name, incoming.name)
    if new_name != existing.name:
        update["name"] = new_name
    # Properties: existing wins on key conflict.
    new_props = dict(existing.properties)
    for k, v in incoming.properties.items():
        if k not in new_props:
            new_props[k] = v
    if len(new_props) != len(existing.properties):
        update["properties"] = new_props
    # Affordances: dedup by (action, target_type), preserve order.
    seen_aff = {(a.action, a.target_type) for a in existing.affordances}
    new_affordances = list(existing.affordances)
    for a in incoming.affordances:
        key = (a.action, a.target_type)
        if key not in seen_aff:
            new_affordances.append(a)
            seen_aff.add(key)
    if len(new_affordances) != len(existing.affordances):
        update["affordances"] = new_affordances
    return existing.model_copy(update=update) if update else existing


def _backfill_location(existing, incoming):
    """Non-destructive merge of two ``Location`` records sharing an id."""
    update: Dict[str, Any] = {}
    new_name = _prefer_longer(existing.name, incoming.name)
    if new_name != existing.name:
        update["name"] = new_name
    new_desc = _prefer_longer(existing.description, incoming.description)
    if new_desc != existing.description:
        update["description"] = new_desc
    # Ambient state: existing wins on key conflict.
    new_ambient = dict(existing.ambient_state)
    for k, v in incoming.ambient_state.items():
        if k not in new_ambient:
            new_ambient[k] = v
    if len(new_ambient) != len(existing.ambient_state):
        update["ambient_state"] = new_ambient
    return existing.model_copy(update=update) if update else existing


def _backfill_world_trait(existing, incoming):
    """Non-destructive merge of two ``GlobalTrait`` records sharing an id."""
    update: Dict[str, Any] = {}
    new_name = _prefer_longer(existing.name, incoming.name)
    if new_name != existing.name:
        update["name"] = new_name
    new_desc = _prefer_longer(existing.description, incoming.description)
    if new_desc != existing.description:
        update["description"] = new_desc
    # Affected domains: set-union, preserve order.
    existing_domains = list(existing.affected_domains)
    seen_dom = set(existing_domains)
    for d in incoming.affected_domains:
        if d not in seen_dom:
            existing_domains.append(d)
            seen_dom.add(d)
    if len(existing_domains) != len(existing.affected_domains):
        update["affected_domains"] = existing_domains
    # Timeline: union by (fabula_time, world_id), preserve order.
    # Without this, sandbox-derived WorldTraitSnapshot entries routed
    # through ``new_world_traits`` for an already-existing trait would
    # be silently dropped because the merge path falls through to the
    # backfill branch instead of appending the trait fresh.
    existing_seen = {
        (s.fabula_time, getattr(s, "world_id", "factual"))
        for s in existing.state_timeline
    }
    appended_timeline = list(existing.state_timeline)
    for snap in incoming.state_timeline:
        key = (snap.fabula_time, getattr(snap, "world_id", "factual"))
        if key not in existing_seen:
            appended_timeline.append(snap)
            existing_seen.add(key)
    if len(appended_timeline) != len(existing.state_timeline):
        appended_timeline.sort(key=lambda s: s.fabula_time)
        update["state_timeline"] = appended_timeline
    return existing.model_copy(update=update) if update else existing


# =====================================================================
# Affect-layer merge helpers (Phase 2026-05-07)
# =====================================================================

def _apply_affect_to_world(
    merged: WorldStateV1,
    topology: "ChunkTopology",
    *,
    world_id: Literal["factual", "shadow"],
    changeset: "MergeChangeset",
) -> None:
    """Fold proposition / concern additions and snapshots into ``merged``.

    Idempotent: re-applying the same topology against the same world
    introduces no duplicates. Mutates ``merged`` in place and updates
    ``changeset`` counters.
    """
    from shadow_loom.models import Proposition, Concern  # local to keep import-time graph clean

    # Index existing propositions by id for O(1) lookup.
    prop_index: Dict[str, Proposition] = {p.proposition_id: p for p in merged.propositions}

    # 1. New propositions (genesis) — dedup on proposition_id.
    for pid, prop in topology.new_propositions.items():
        if pid in prop_index:
            continue
        new_prop = prop.model_copy(update={"world_id": world_id}) if world_id != "factual" else prop
        merged.propositions.append(new_prop)
        prop_index[pid] = new_prop
        changeset.propositions_added += 1

    # 2. Truth commits → Proposition.truth_at_fabula
    for commit in topology.proposition_truth_commits:
        prop = prop_index.get(commit.proposition_id)
        if prop is None:
            logger.warning(
                "[merge·affect] Truth commit for unknown PROP %s — skipped.",
                commit.proposition_id,
            )
            continue
        existing = prop.truth_at_fabula.get(commit.fabula_time)
        if existing is not None and existing == commit.truth:
            continue  # idempotent re-apply
        prop.truth_at_fabula[commit.fabula_time] = commit.truth
        changeset.proposition_truths_committed += 1

    # 3. Proposition framing snapshots → Proposition.state_timeline
    for snap in topology.proposition_snapshots:
        prop = prop_index.get(snap.proposition_id)
        if prop is None:
            logger.warning(
                "[merge·affect] Snapshot for unknown PROP %s — skipped.",
                snap.proposition_id,
            )
            continue
        from shadow_loom.models import PropositionSnapshot
        ps = PropositionSnapshot(
            world_id=world_id,
            fabula_time=snap.fabula_time,
            triggered_by=snap.triggered_by,
            stakes=snap.stakes,
            audience_default_prior=snap.audience_default_prior,
            description=snap.description,
        )
        # Dedup on (fabula_time, triggered_by, world_id)
        key = (ps.fabula_time, ps.triggered_by, ps.world_id)
        existing_keys = {
            (s.fabula_time, s.triggered_by, getattr(s, "world_id", "factual"))
            for s in prop.state_timeline
        }
        if key in existing_keys:
            continue
        prop.state_timeline.append(ps)
        prop.state_timeline.sort(key=lambda s: s.fabula_time)
        changeset.proposition_snapshots_added += 1

    # 4. New concerns (direct genesis) under a holder
    for entity_id, concerns in topology.new_concerns.items():
        ent = merged.entities.get(entity_id)
        if ent is None:
            logger.warning(
                "[merge·affect] new_concerns for unknown entity %s — skipped.",
                entity_id,
            )
            continue
        existing_keys = {
            (c.proposition_id, c.polarity) for c in ent.concerns
        }
        for concern in concerns:
            key = (concern.proposition_id, concern.polarity)
            if key in existing_keys:
                continue
            new_concern = (
                concern.model_copy(update={"world_id": world_id})
                if world_id != "factual" else concern
            )
            ent.concerns.append(new_concern)
            existing_keys.add(key)
            changeset.concerns_added += 1

    # 5. ConcernSeed materialisation (Phase B4 outputs already have a
    # holder pinned in the seed). Lazy-import the seed type to avoid
    # widening the import surface.
    try:
        from shadow_loom.ingestion import ConcernSeed  # type: ignore
    except Exception:
        ConcernSeed = None  # type: ignore
    if ConcernSeed is not None:
        for seed in topology.new_concern_seeds:
            holder_id = getattr(seed, "holder_id", None) or getattr(seed, "entity_id", None)
            if not holder_id:
                continue
            ent = merged.entities.get(holder_id)
            if ent is None:
                logger.warning(
                    "[merge·affect] ConcernSeed for unknown holder %s — skipped.",
                    holder_id,
                )
                continue
            prop_id = getattr(seed, "proposition_id", None)
            polarity = getattr(seed, "polarity", None)
            if not prop_id or not polarity:
                continue
            existing_keys = {(c.proposition_id, c.polarity) for c in ent.concerns}
            if (prop_id, polarity) in existing_keys:
                continue
            ccn_id = getattr(seed, "concern_id", None) or f"CCN_{holder_id}_{prop_id}_{polarity}".upper()
            try:
                concern = Concern(
                    world_id=world_id,
                    concern_id=ccn_id,
                    proposition_id=prop_id,
                    polarity=polarity,
                    kind=getattr(seed, "kind", None),
                    salience=getattr(seed, "baseline_salience", None) or getattr(seed, "salience", 0.5) or 0.5,
                    counter_concern_ids=list(getattr(seed, "counter_concern_ids", []) or []),
                )
            except Exception:
                logger.exception(
                    "[merge·affect] Failed to materialise ConcernSeed for %s.", holder_id,
                )
                continue
            ent.concerns.append(concern)
            changeset.concerns_added += 1

    # 6. Concern snapshots → Concern.state_timeline
    if topology.concern_snapshots:
        from shadow_loom.models import ConcernSnapshot
        # Build a (entity_id, concern_id) → Concern index (concerns may
        # be globally unique via concern_id but we still scope per-entity
        # so we don't accidentally write across holders).
        concern_index: Dict[str, "Concern"] = {}
        for ent in merged.entities.values():
            for c in ent.concerns:
                concern_index[c.concern_id] = c
        for snap in topology.concern_snapshots:
            target = concern_index.get(snap.concern_id)
            if target is None:
                logger.warning(
                    "[merge·affect] Concern snapshot for unknown CCN %s — skipped.",
                    snap.concern_id,
                )
                continue
            cs = ConcernSnapshot(
                world_id=world_id,
                fabula_time=snap.fabula_time,
                triggered_by=snap.triggered_by,
                salience=snap.salience,
                polarity=snap.polarity,
                activation_fabula_window=snap.activation_fabula_window,
                counter_concern_ids=snap.counter_concern_ids,
                kind=snap.kind,
            )
            existing_keys = {
                (s.fabula_time, s.triggered_by, getattr(s, "world_id", "factual"))
                for s in target.state_timeline
            }
            key = (cs.fabula_time, cs.triggered_by, cs.world_id)
            if key in existing_keys:
                continue
            target.state_timeline.append(cs)
            target.state_timeline.sort(key=lambda s: s.fabula_time)
            changeset.concern_snapshots_added += 1

    # 7. Belief snapshots → EntityStateSnapshot.belief_confidence_updates
    # Affect-side per-character confidence drift. Folded onto the
    # entity's state_timeline at the snapshot's fabula_time so
    # ``reconstruct_entity_at`` replays the drift in order. Distinct
    # from Consequences-side ``EntityUpdate.belief_confidence_updates``
    # which is a flat overwrite on the *current* belief list.
    if topology.belief_snapshots:
        from shadow_loom.models import EntityStateSnapshot, BeliefConfidenceShift
        for bs in topology.belief_snapshots:
            ent = merged.entities.get(bs.holder_id)
            if ent is None:
                logger.warning(
                    "[merge\u00b7affect] Belief snapshot for unknown holder %s \u2014 skipped.",
                    bs.holder_id,
                )
                continue
            # Locate-or-create an EntityStateSnapshot at this
            # (fabula_time, triggered_by) so multiple Affect snapshots
            # at the same tick stack onto a single timeline entry.
            snap_obj: Optional[EntityStateSnapshot] = None
            for s in ent.state_timeline:
                if (
                    s.fabula_time == bs.fabula_time
                    and s.triggered_by == bs.triggered_by
                    and getattr(s, "world_id", "factual") == world_id
                ):
                    snap_obj = s
                    break
            if snap_obj is None:
                snap_obj = EntityStateSnapshot(
                    world_id=world_id,
                    fabula_time=bs.fabula_time,
                    triggered_by=bs.triggered_by,
                )
                ent.state_timeline.append(snap_obj)
                ent.state_timeline.sort(key=lambda s: s.fabula_time)
            # Dedup on (target_id, proposition_id) within the snapshot
            # so re-applying the same topology doesn't pile up shifts.
            existing = {
                (u.target_id, u.proposition_id)
                for u in snap_obj.belief_confidence_updates
            }
            key = (bs.target_id, bs.proposition_id)
            if key in existing:
                continue
            snap_obj.belief_confidence_updates.append(BeliefConfidenceShift(
                target_id=bs.target_id,
                proposition_id=bs.proposition_id,
                new_confidence=bs.new_confidence,
                new_inertia=bs.new_inertia,
            ))
            changeset.belief_snapshots_added += 1


def _apply_belief_confidence_updates(
    merged: WorldStateV1,
    topology: "ChunkTopology",
    *,
    changeset: "MergeChangeset",
) -> None:
    """Apply each ``EntityUpdate.belief_confidence_updates`` entry by
    overwriting the matching existing :class:`Belief` on the entity.

    The match is by ``target_id`` (and ``proposition_id`` when set). If
    no matching belief exists the update is dropped with a warning —
    creation should go through ``new_beliefs`` on the same EntityUpdate.
    """
    for eu in topology.entity_updates:
        for upd in eu.belief_confidence_updates:
            ent = merged.entities.get(eu.entity_id)
            if ent is None:
                continue
            matched = False
            for belief in ent.beliefs:
                if belief.target_id != upd.target_id:
                    continue
                if upd.proposition_id and belief.proposition_id and belief.proposition_id != upd.proposition_id:
                    continue
                belief.confidence = float(upd.new_confidence)
                if upd.new_inertia is not None:
                    belief.inertia = float(upd.new_inertia)
                matched = True
                changeset.belief_confidence_updates_applied += 1
                break
            if not matched:
                logger.debug(
                    "[merge·belief] No existing belief on %s with target_id=%s; "
                    "confidence update skipped.",
                    eu.entity_id, upd.target_id,
                )


def _apply_deletions(
    merged: WorldStateV1,
    topology: "ChunkTopology",
    *,
    changeset: "MergeChangeset",
) -> None:
    """Run the deletion pass before additive merge sections.

    Cascades dependent edges/snapshots when a parent node is removed
    (e.g. removing an entity also drops its concerns and any social
    edges or beliefs naming it).
    """
    # --- Events
    if topology.removed_event_ids:
        drop = set(topology.removed_event_ids)
        before = len(merged.events)
        merged.events = [e for e in merged.events if e.id not in drop]
        removed_n = before - len(merged.events)
        changeset.events_removed += removed_n
        # Cascade: drop causal/spatial/social edges referencing dropped events.
        before_c = len(merged.causal_topology)
        merged.causal_topology = [
            c for c in merged.causal_topology
            if c.source_id not in drop and c.target_id not in drop
        ]
        changeset.causal_edges_removed += before_c - len(merged.causal_topology)

    # --- Causal edges (explicit keys)
    if topology.removed_causal_edge_keys:
        keys = {tuple(k) for k in topology.removed_causal_edge_keys}
        before = len(merged.causal_topology)
        merged.causal_topology = [
            c for c in merged.causal_topology
            if (c.source_id, c.target_id, c.causality_type, c.fabula_time) not in keys
        ]
        changeset.causal_edges_removed += before - len(merged.causal_topology)

    # --- Social dyads
    if topology.removed_social_dyad_keys:
        keys = {tuple(k) for k in topology.removed_social_dyad_keys}
        before = len(merged.social_topology)
        merged.social_topology = [
            r for r in merged.social_topology
            if (r.source_entity_id, r.target_entity_id) not in keys
        ]
        changeset.social_edges_removed += before - len(merged.social_topology)

    # --- Spatial edges
    if topology.removed_spatial_keys:
        keys = {tuple(k) for k in topology.removed_spatial_keys}
        before = len(merged.spatial_topology)
        merged.spatial_topology = [
            s for s in merged.spatial_topology
            if (s.source_id, s.target_id) not in keys
        ]
        changeset.spatial_edges_removed += before - len(merged.spatial_topology)

    # --- Channels
    if topology.removed_channel_ids:
        drop_chan = set(topology.removed_channel_ids)
        for cid in drop_chan:
            if cid in merged.channels:
                del merged.channels[cid]
                changeset.channels_removed += 1
        # Cascade: scrub channel pointers on events and beliefs.
        for evt in merged.events:
            if getattr(evt, "via_channel_id", None) in drop_chan:
                evt.via_channel_id = None
        for ent in merged.entities.values():
            for b in ent.beliefs:
                if getattr(b, "acquired_via_channel_id", None) in drop_chan:
                    b.acquired_via_channel_id = None

    # --- Entities (cascade beliefs/concerns/edges)
    if topology.removed_entity_ids:
        drop = set(topology.removed_entity_ids)
        for eid in drop:
            if eid in merged.entities:
                del merged.entities[eid]
                changeset.entities_removed += 1
        # Drop social edges naming a dropped entity.
        before_s = len(merged.social_topology)
        merged.social_topology = [
            r for r in merged.social_topology
            if r.source_entity_id not in drop and r.target_entity_id not in drop
        ]
        changeset.social_edges_removed += before_s - len(merged.social_topology)
        # Drop other entities' beliefs targeting a dropped entity.
        for ent in merged.entities.values():
            ent.beliefs = [b for b in ent.beliefs if b.target_id not in drop]

    # --- Objects
    for oid in topology.removed_object_ids:
        if oid in merged.objects:
            del merged.objects[oid]
            changeset.objects_removed += 1

    # --- Locations
    for lid in topology.removed_location_ids:
        if lid in merged.locations:
            del merged.locations[lid]
            changeset.locations_removed += 1

    # --- World traits
    for wid in topology.removed_world_trait_ids:
        if wid in merged.world_traits:
            del merged.world_traits[wid]
            changeset.world_traits_removed += 1

    # --- Propositions (also drop concerns referencing them; scrub
    # proposition pointers on surviving events and beliefs).
    if topology.removed_proposition_ids:
        drop = set(topology.removed_proposition_ids)
        before = len(merged.propositions)
        merged.propositions = [p for p in merged.propositions if p.proposition_id not in drop]
        changeset.propositions_removed += before - len(merged.propositions)
        for ent in merged.entities.values():
            before_c = len(ent.concerns)
            ent.concerns = [c for c in ent.concerns if c.proposition_id not in drop]
            changeset.concerns_removed += before_c - len(ent.concerns)
            for b in ent.beliefs:
                if getattr(b, "proposition_id", None) in drop:
                    b.proposition_id = None
        for evt in merged.events:
            if getattr(evt, "asserts_proposition_id", None) in drop:
                evt.asserts_proposition_id = None
            if getattr(evt, "denies_proposition_id", None) in drop:
                evt.denies_proposition_id = None
            resolves = getattr(evt, "resolves_proposition_ids", None)
            if resolves:
                evt.resolves_proposition_ids = [
                    pid for pid in resolves if pid not in drop
                ]

    # --- Concerns (entity_id, concern_id)
    if topology.removed_concern_ids:
        drop = set((eid, cid) for eid, cid in topology.removed_concern_ids)
        removed_cids = {cid for _eid, cid in drop}
        for eid, cid in drop:
            ent = merged.entities.get(eid)
            if ent is None:
                continue
            before_c = len(ent.concerns)
            ent.concerns = [c for c in ent.concerns if c.concern_id != cid]
            changeset.concerns_removed += before_c - len(ent.concerns)
        # Scrub counter_concern_ids on surviving concerns.
        for ent in merged.entities.values():
            for c in ent.concerns:
                ccids = getattr(c, "counter_concern_ids", None)
                if ccids:
                    c.counter_concern_ids = [
                        x for x in ccids if x not in removed_cids
                    ]


def _apply_supersession(
    merged: WorldStateV1,
    topology: "ChunkTopology",
    *,
    changeset: "MergeChangeset",
) -> None:
    """Stamp ``superseded_by_event_id`` on overridden events and rewrite
    cross-references on beliefs / concerns / propositions / new events."""
    if not topology.supersedes_event_ids:
        return
    mapping = dict(topology.supersedes_event_ids)
    event_index = {e.id: e for e in merged.events}
    for new_id, old_id in mapping.items():
        old_evt = event_index.get(old_id)
        if old_evt is None:
            logger.warning(
                "[merge·supersede] Old event %s not found — supersession skipped.", old_id,
            )
            continue
        if old_evt.superseded_by_event_id == new_id:
            continue
        old_evt.superseded_by_event_id = new_id
        changeset.events_superseded += 1

    # Rewrite belief provenance.
    for ent in merged.entities.values():
        for b in ent.beliefs:
            if b.acquired_via_event_id in mapping:
                b.acquired_via_event_id = mapping[b.acquired_via_event_id]

    # Rewrite Proposition.truth_at_fabula? No — those are keyed by time, not event.
    # But events' resolves_proposition_ids are still valid; we leave the
    # old event's list intact and rely on the supersede pointer for
    # downstream readers that prefer the override.


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
            for prop in topology.new_propositions.values():
                prop.world_id = world_id
            for concern_list in topology.new_concerns.values():
                for c in concern_list:
                    c.world_id = world_id
        changeset = MergeChangeset()

        # --- Deletion pass (P2 of prose-merge completeness) — runs
        # before additive sections so a single merge can replace-then-
        # add cleanly. Cascades dependent edges/snapshots.
        _apply_deletions(merged, topology, changeset=changeset)

        # --- Genesis-promoted nodes (entities / objects / locations /
        # world_traits) — written first so subsequent edge / event /
        # entity_update merges can reference them. Existing IDs keep
        # their identity; non-destructive attribute backfill fills in
        # missing fields (longer description, additional traits /
        # affordances / domains, set-union of constants) from the
        # incoming record so re-extractions enrich rather than
        # silently drop data.
        for eid, ent in topology.new_entities.items():
            existing = merged.entities.get(eid)
            if existing is not None:
                merged.entities[eid] = _backfill_entity(existing, ent)
                logger.debug(
                    "[VersionedWorldModel·merge] Spawn entity %s already exists — backfilled.",
                    eid,
                )
                continue
            merged.entities[eid] = ent
            changeset.entities_added += 1
        for oid, obj in topology.new_objects.items():
            existing_obj = merged.objects.get(oid)
            if existing_obj is not None:
                merged.objects[oid] = _backfill_object(existing_obj, obj)
                logger.debug(
                    "[VersionedWorldModel·merge] Spawn object %s already exists — backfilled.",
                    oid,
                )
                continue
            merged.objects[oid] = obj
            changeset.objects_added += 1
        for lid, loc in topology.new_locations.items():
            existing_loc = merged.locations.get(lid)
            if existing_loc is not None:
                merged.locations[lid] = _backfill_location(existing_loc, loc)
                logger.debug(
                    "[VersionedWorldModel·merge] Spawn location %s already exists — backfilled.",
                    lid,
                )
                continue
            merged.locations[lid] = loc
            changeset.locations_added += 1
        for wid, wt in topology.new_world_traits.items():
            existing_wt = merged.world_traits.get(wid)
            if existing_wt is not None:
                merged.world_traits[wid] = _backfill_world_trait(existing_wt, wt)
                logger.debug(
                    "[VersionedWorldModel·merge] World trait %s already exists — backfilled.",
                    wid,
                )
                continue
            merged.world_traits[wid] = wt
            changeset.world_traits_added += 1

        # --- Events (deduplicate by (id, world_id), keep existing) ---
        # Branch-aware: the same EVT_ id can legitimately exist on both
        # factual and shadow branches (Pearl Rung-2/3 fork). Collapsing
        # purely on ``id`` would silently drop the shadow variant of an
        # event whose factual sibling was already merged.
        pre_events = len(merged.events)
        existing_event_keys = {
            (e.id, getattr(e, "world_id", "factual")) for e in merged.events
        }
        for evt in topology.events:
            key = (evt.id, getattr(evt, "world_id", "factual"))
            if key not in existing_event_keys:
                merged.events.append(evt)
                existing_event_keys.add(key)
            else:
                logger.debug(
                    "[VersionedWorldModel·merge] Duplicate event %s (%s) — kept existing.",
                    evt.id,
                    key[1],
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
        merged.spatial_topology.sort(key=lambda s: s.established_at_fabula)
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
        merged.social_topology.sort(key=lambda r: r.last_updated_fabula)
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
                world_id=world_id,
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

        # --- Affect ledger (P2 of prose-merge completeness) ---
        # Folds new propositions, truth commits, proposition snapshots,
        # new concerns, ConcernSeed materialisations, and concern
        # snapshots into the merged world.
        _apply_affect_to_world(merged, topology, world_id=world_id, changeset=changeset)
        # Belief confidence overwrites (Pearl Rung-2 BeliefMutation bridge).
        _apply_belief_confidence_updates(merged, topology, changeset=changeset)
        # Supersession (mainline-promoted counterfactual override).
        _apply_supersession(merged, topology, changeset=changeset)

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

