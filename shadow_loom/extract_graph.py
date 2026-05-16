# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

from shadow_loom.models import (
    Entity,
    EntityStateSnapshot,
    GlobalTrait,
    NarrativeObject,
    ObjectStateSnapshot,
    Proposition,
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_object_at,
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

    # Edge-level lifecycle gate: the relationship has either not yet
    # begun (``established_at_fabula > t``) or has already been
    # severed (``ended_at_fabula <= t``). Drop the edge from the
    # sliced view in both cases — per-axis freshness is irrelevant
    # outside the active interval.
    est = data.get("established_at_fabula")
    if est is not None and est > t:
        return None
    end = data.get("ended_at_fabula")
    if end is not None and end <= t:
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
    relevant_propositions: List[dict] = Field(
        default_factory=list,
        description=(
            "Propositions whose ``referent_ids`` intersect the in-scene "
            "node set, time-sliced to the temporal anchor. Surfacing "
            "these into the ego-graph lets the renderer and auditor see "
            "the live propositional ledger \u2014 what's true, what's "
            "false, what's open \u2014 alongside the entity-level beliefs "
            "and concerns the same characters carry."
        ),
    )

# ==========================================
# 1b. SHADOW-PATH ANCESTOR CLOSURE
# ==========================================
def _causal_ancestors(
    world_state: WorldStateV1,
    seed_ids: Set[str],
    temporal_anchor: Optional[int],
    max_hops: int = 8,
) -> Set[str]:
    """Reverse-BFS over ``world_state.causal_topology``.

    Returns the set of node ids that can reach any seed via causal
    edges, anchor-time-sliced. Used to expand the ego-graph's scene
    set so Pearl-rung sandboxes (rungs 2 and 3) and directive briefs
    can reason over upstream causal lineage that lives outside the
    ``memory_limit`` recency window. Without this expansion an
    intervention/counterfactual at the climax of a long plot cannot
    reach the events that caused the present-day state, so the
    sandbox's "shadow paths from across the plot" are structurally
    severed at the recency boundary.
    """
    rev: Dict[str, List[str]] = {}
    for ce in world_state.causal_topology:
        if temporal_anchor is not None and ce.fabula_time > temporal_anchor:
            continue
        rev.setdefault(ce.target_id, []).append(ce.source_id)

    visited: Set[str] = set()
    frontier: Set[str] = set(seed_ids)
    for _ in range(max_hops):
        if not frontier:
            break
        next_frontier: Set[str] = set()
        for nid in frontier:
            for parent in rev.get(nid, []):
                if parent in visited or parent in seed_ids:
                    continue
                visited.add(parent)
                next_frontier.add(parent)
        frontier = next_frontier
    return visited


# ==========================================
# 2. THE IN-MEMORY EXTRACTION FUNCTION
# ==========================================
def extract_ego_graph_from_memory(
    world_state: WorldStateV1,
    focus_entity_ids: List[str],
    temporal_anchor: Optional[int] = None,
    memory_limit: int = 5,
    syuzhet_anchor: Optional[int] = None,
    shadow_path_seed_ids: Optional[Set[str]] = None,
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
        # Reconstruct object position/ownership at the temporal anchor
        # so picked-up / dropped / transferred objects are filtered into
        # the ego graph at the right place in fabula time. Without this
        # the AMWN sandbox sees only the object's *initial* location_id /
        # owner_id and a dagger that started in the kitchen would never
        # be visible in the bedroom scene where the murder takes place.
        if temporal_anchor is not None and obj.state_timeline:
            recon = reconstruct_object_at(obj, temporal_anchor)
            effective_loc = recon["location_id"]
            effective_owner = recon["owner_id"]
        else:
            effective_loc = obj.location_id
            effective_owner = obj.owner_id
        is_on_active_floor = (effective_loc in location_ids)
        is_held_by_active_local = (
            effective_owner in present_entity_ids
            or effective_owner in focus_id_set
        )

        if is_on_active_floor or is_held_by_active_local:
            obj_data = obj.model_dump()
            if temporal_anchor is not None and obj.state_timeline:
                obj_data["location_id"] = effective_loc
                obj_data["owner_id"] = effective_owner
                obj_data["properties"] = recon["properties"]
            present_objects.append(obj_data)

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
    # Drop superseded events from recency memory: when a counterfactual
    # is promoted onto the factual mainline the original event is
    # preserved on the world state for replay/audit but is no longer
    # the canonical version downstream consumers should reason over.
    # Retain when the named successor itself isn't (yet) in the valid
    # window so the recency feed doesn't go silent on the seam.
    valid_event_ids = {evt.id for evt in valid_events}
    valid_events = [
        evt for evt in valid_events
        if evt.superseded_by_event_id is None
        or evt.superseded_by_event_id not in valid_event_ids
    ]

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

    # 7a. Shadow-path expansion (Pearl-rung sandboxes + directive briefs).
    # When the caller is constructing a Rung-2/3 sandbox or a directive
    # brief, recency-only ``recent_memory`` truncation severs the
    # upstream causal lineage of intervention / evidence / focus
    # targets. Walk the reverse causal graph from those seeds and
    # admit any ancestor *events* into the scene so the sandbox can
    # carry the full event-to-event shadow path. Ancestor *entities*
    # are deliberately NOT injected: they would be admitted after
    # ``relevant_relationships`` had already been filtered, leaving
    # the entity present as a sandbox node but without its
    # relationship edges \u2014 which silently re-routes do-surgery
    # against a relationship from the inertia-checked path to the
    # "create new edge" path, and admits stale event\u2192entity causal
    # edges into the sandbox. Observation queries without a seed set
    # keep the original recency-only behaviour.
    if shadow_path_seed_ids:
        seeds: Set[str] = set(shadow_path_seed_ids) | focus_id_set
        ancestors = _causal_ancestors(world_state, seeds, temporal_anchor)
        if ancestors:
            existing_evt_ids = {e["id"] for e in recent_memory}
            event_by_id = {e.id: e for e in world_state.events}
            ancestor_events_added = 0
            for nid in ancestors:
                if nid in existing_evt_ids:
                    continue
                evt = event_by_id.get(nid)
                if evt is None:
                    continue
                if temporal_anchor is not None and evt.fabula_time > temporal_anchor:
                    continue
                if syuzhet_anchor is not None and evt.syuzhet_index > syuzhet_anchor:
                    continue
                recent_memory.append(evt.model_dump())
                existing_evt_ids.add(nid)
                ancestor_events_added += 1
            if ancestor_events_added:
                logger.info(
                    "[EgoGraph] Shadow-path expansion: +%d ancestor "
                    "events (seeds=%d, ancestors=%d)",
                    ancestor_events_added, len(seeds), len(ancestors),
                )

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

    # 9. Relevant propositions \u2014 propositions whose ``referent_ids``
    # intersect the in-scene node set, time-sliced. Surfacing these
    # alongside the entity-level beliefs/concerns lets the renderer
    # and auditor see what's true / false / open in the world model
    # for every prose-producing query type. Without this the
    # propositional ledger only reaches the prompt as flipped-id
    # lists on rung-2/3 deltas \u2014 the live propositions for the
    # focal scene are invisible.
    relevant_propositions: List[dict] = []
    prop_scene_ids = scene_node_ids  # already includes focus + present + locs + memory
    for prop in world_state.propositions:
        refs = prop.referent_ids or []
        if not refs:
            # Project-wide propositions (no specific referent) are
            # admitted unconditionally so the scene sees them.
            pass
        elif not any(r in prop_scene_ids for r in refs):
            continue
        prop_data = prop.model_dump()
        if temporal_anchor is not None and isinstance(prop_data.get("truth_at_fabula"), dict):
            prop_data["truth_at_fabula"] = {
                int(t): v
                for t, v in prop_data["truth_at_fabula"].items()
                if int(t) <= temporal_anchor
            }
        relevant_propositions.append(prop_data)

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
        relevant_propositions=relevant_propositions,
    )

    logger.info("Multi-Ego GraphRAG complete — %d focus, %d locations, %d co-present, %d objects, %d relationships, %d causal, %d spatial, %d channels, %d utterances, %d memory, %d propositions",
                 len(focus_entities), len(current_locations), len(present_entities),
                 len(present_objects), len(relevant_relationships), len(relevant_causal_edges),
                 len(relevant_spatial_edges), len(relevant_channels),
                 len(relevant_utterance_events), len(recent_memory),
                 len(relevant_propositions))
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

    # Drop superseded events from the omniscient view: when a successor
    # event is itself in the surviving set, the older (overridden)
    # event is no longer the canonical version downstream physics /
    # render / audit consumers should reason over. Preserved on the
    # raw ``world_state.events`` for replay/audit access; this is a
    # render-time projection only.
    surviving_ids = {evt.get("id") for evt in dump["events"]}
    pre_evt = len(dump["events"])
    dump["events"] = [
        evt for evt in dump["events"]
        if not evt.get("superseded_by_event_id")
        or evt["superseded_by_event_id"] not in surviving_ids
    ]
    if len(dump["events"]) < pre_evt:
        logger.info(
            "Omniscient Graph supersession-pruned: %d/%d events kept "
            "(removed %d superseded by surviving successors)",
            len(dump["events"]), pre_evt, pre_evt - len(dump["events"]),
        )

    return dump


# ==========================================
# 4. PROSE → TOPOLOGY EXTRACTION
# ==========================================

def introduced_elements_to_spawns(
    introduced: "Any",
    world_state: WorldStateV1,
    *,
    world_id: Literal["factual", "shadow"] = "factual",
) -> Dict[str, Dict[str, Any]]:
    """Materialise renderer-declared :class:`IntroducedElements` as
    a ``spawns``-shaped payload that ``extract_topology_from_prose``
    can mix with engine-side spawn nodes.

    The output mirrors :func:`promote_sandbox_spawns` exactly so the
    two channels can be merged with a simple per-key ``dict.update`` —
    engine-side spawns (deterministic ground truth from
    ``<ID>.spawn`` surgeries) win on collisions because they encode
    constraints the renderer is not allowed to override.

    Existing canonical IDs are skipped so the function is idempotent
    if the renderer re-declares an element already in the canonical
    world (which shouldn't happen, but is harmless if it does).

    ``world_id`` tags every spawned :class:`Proposition` and
    :class:`Concern`. Defaults to ``"factual"``; callers running on a
    shadow branch (counterfactual / what-if pipeline) must pass
    ``"shadow"`` so the introductions land in the same AMWN
    sub-graph as the rest of that branch's nodes — otherwise the
    instantiator's three-rule bookkeeping (consistency / independence
    / exclusion) sees factual props/concerns spawned inside a shadow
    sandbox and the next merge or rollback misroutes them.
    """
    from shadow_loom.models import (
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
        "concerns": {},
        "events": {},
    }
    if introduced is None or getattr(introduced, "is_empty", lambda: True)():
        return out

    # ── Locations first so entity/object ``located_in`` references resolve.
    for spec in getattr(introduced, "locations", []) or []:
        if spec.id in world_state.locations or spec.id in out["locations"]:
            continue
        try:
            out["locations"][spec.id] = Location(
                name=spec.name,
                description=spec.description or "",
                ambient_state={},
            )
        except Exception:
            logger.exception(
                "[introduced_elements_to_spawns] Location %s invalid \u2014 skipped.",
                spec.id,
            )

    for spec in getattr(introduced, "entities", []) or []:
        if spec.id in world_state.entities or spec.id in out["entities"]:
            continue
        location_id = spec.located_in
        if not location_id:
            logger.warning(
                "[introduced_elements_to_spawns] Entity %s missing located_in \u2014 skipped.",
                spec.id,
            )
            continue
        try:
            out["entities"][spec.id] = Entity(
                id=spec.id,
                name=spec.name,
                location_id=location_id,
                status="healthy",
                traits=dict(spec.initial_traits or {}),
                beliefs=[],
                constants=[],
                state_timeline=[],
            )
        except Exception:
            logger.exception(
                "[introduced_elements_to_spawns] Entity %s invalid \u2014 skipped.",
                spec.id,
            )

    for spec in getattr(introduced, "objects", []) or []:
        if spec.id in world_state.objects or spec.id in out["objects"]:
            continue
        try:
            out["objects"][spec.id] = NarrativeObject(
                id=spec.id,
                name=spec.name,
                location_id=spec.located_in,
                owner_id=None,
                properties={},
                affordances=[],
            )
        except Exception:
            logger.exception(
                "[introduced_elements_to_spawns] Object %s invalid \u2014 skipped.",
                spec.id,
            )

    for spec in getattr(introduced, "world_traits", []) or []:
        if spec.id in world_state.world_traits or spec.id in out["world_traits"]:
            continue
        magnitude = TraitVector(
            value=float(spec.value) if spec.value is not None else 0.5,
            inertia=0.5,
            evidence_strength="moderate",
        )
        try:
            out["world_traits"][spec.id] = GlobalTrait(
                id=spec.id,
                name=spec.name,
                description=spec.description or "",
                category="social_structure",
                magnitude=magnitude,
                affected_domains=[],
                state_timeline=[],
            )
        except Exception:
            logger.exception(
                "[introduced_elements_to_spawns] WorldTrait %s invalid \u2014 skipped.",
                spec.id,
            )

    existing_prop_ids = {p.proposition_id for p in world_state.propositions}
    for spec in getattr(introduced, "propositions", []) or []:
        if spec.id in existing_prop_ids or spec.id in out["propositions"]:
            continue
        try:
            out["propositions"][spec.id] = Proposition(
                world_id=world_id,
                proposition_id=spec.id,
                kind=spec.kind or "outcome",
                referent_ids=[],
                description=spec.name,
                audience_default_prior=0.5,
                stakes=0.5,
                truth_at_fabula={},
            )
        except Exception:
            logger.exception(
                "[introduced_elements_to_spawns] Proposition %s invalid \u2014 skipped.",
                spec.id,
            )

    for spec in getattr(introduced, "concerns", []) or []:
        try:
            # Spec polarity is "positive" / "negative"; the canonical
            # ``Concern`` model only accepts ``Literal["desire", "fear"]``.
            # Map directly without an intermediate vocabulary so the
            # validation error doesn't get swallowed silently below.
            polarity_da = "desire" if spec.polarity == "positive" else "fear"
            concern = Concern(
                world_id=world_id,
                concern_id=spec.id,
                proposition_id=spec.proposition_id,
                polarity=polarity_da,
                kind=None,
                salience=float(spec.salience),
                activation_fabula_window=None,
                counter_concern_ids=[],
            )
            out["concerns"].setdefault(spec.holder_entity_id, []).append(concern)
        except Exception:
            logger.exception(
                "[introduced_elements_to_spawns] Concern %s invalid \u2014 skipped.",
                spec.id,
            )

    # Channels \u2014 standing communication capabilities. Materialised
    # alongside entity/location spawns so utterance events introduced
    # in the same payload (or do-surgeries that target the channel)
    # can resolve their ``via_channel_id``.
    from shadow_loom.models import Channel, EventNode  # local import to avoid cycle
    for spec in getattr(introduced, "channels", []) or []:
        canonical_channels = world_state.channels or {}
        if spec.id in canonical_channels or spec.id in out["channels"]:
            continue
        try:
            out["channels"][spec.id] = Channel(
                id=spec.id,
                name=spec.name,
                medium=spec.medium,
                participant_ids=list(spec.participant_ids),
                directionality=spec.directionality,
                intelligibility=dict(spec.intelligibility),
            )
        except Exception:
            logger.exception(
                "[introduced_elements_to_spawns] Channel %s invalid \u2014 skipped.",
                spec.id,
            )

    # Events \u2014 query-authored EventNodes. Materialised so the
    # pipeline can append them to ``WorldStateV1.events`` before
    # physics, letting ``DoEvent`` clamp their occurrence and giving
    # the renderer a typed handle to cite.
    existing_event_ids = {e.id for e in (world_state.events or [])}
    for spec in getattr(introduced, "events", []) or []:
        if spec.id in existing_event_ids or spec.id in out["events"]:
            continue
        try:
            out["events"][spec.id] = EventNode(
                id=spec.id,
                fabula_time=int(spec.fabula_time),
                syuzhet_index=int(spec.syuzhet_index),
                event_type=spec.event_type,
                actor_ids=list(spec.actor_ids),
                target_ids=list(spec.target_ids),
                description=spec.description,
                content=spec.content,
                via_channel_id=spec.via_channel_id,
                speaker_id=spec.speaker_id,
                addressee_ids=list(spec.addressee_ids),
                truth_value=spec.truth_value,
            )
        except Exception:
            logger.exception(
                "[introduced_elements_to_spawns] Event %s invalid \u2014 skipped.",
                spec.id,
            )

    return out


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
    introduced_elements: "Any" = None,
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
    spawns = dict(spawns) if spawns else {}
    for k in (
        "entities", "objects", "locations", "world_traits",
        "channels", "propositions", "concerns",
    ):
        spawns.setdefault(k, {})

    # Mix renderer-declared introductions in. Engine spawns win on
    # collisions because they encode hard physics constraints the
    # renderer is not allowed to override.
    if introduced_elements is not None and not getattr(
        introduced_elements, "is_empty", lambda: True,
    )():
        rendered_spawns = introduced_elements_to_spawns(
            introduced_elements, world_state,
            world_id=branch_world_id,
        )
        for k in ("entities", "objects", "locations", "world_traits",
                  "channels", "propositions"):
            for nid, payload in rendered_spawns.get(k, {}).items():
                spawns[k].setdefault(nid, payload)
        # Concerns are list-valued per holder; extend without
        # clobbering engine-side concerns for the same holder.
        for holder_id, clist in rendered_spawns.get("concerns", {}).items():
            spawns["concerns"].setdefault(holder_id, []).extend(clist)

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
            # Surface renderer-/engine-introduced propositions to the
            # affect agent so it can commit truth values / framing
            # snapshots against them, not just against canonical
            # propositions.
            extra_props = list((spawns.get("propositions") or {}).values())
            for p in extra_props:
                catalogue_prop_ids.add(p.proposition_id)
            seeded_entity_ids = {
                eid for eid, e in world_state.entities.items() if e.concerns
            }
            # Holders that picked up a fresh concern via spawns also
            # qualify as "seeded" for affect-signal detection.
            for holder_id in (spawns.get("concerns") or {}).keys():
                seeded_entity_ids.add(holder_id)
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
                # Mirror the same for spawn-introduced concerns so
                # the agent sees them in its baseline catalogue.
                for holder_id, clist in (spawns.get("concerns") or {}).items():
                    for c in clist:
                        try:
                            seeds.append(ConcernSeed(
                                concern_id=c.concern_id,
                                entity_id=holder_id,
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
                    propositions=list(world_state.propositions) + extra_props,
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
    object_updates_applied: int = 0
    object_updates_skipped: List[str] = Field(default_factory=list)
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
    # Referential-integrity: events whose actor_ids/target_ids reference
    # ids that don't exist in the merged world AND were not declared via
    # ``introduced_elements`` / sandbox spawns. Each entry is a dict
    # ``{"event_id": str, "missing_ids": list[str], "field": str}``.
    # Empty by default. Populated by the merge integrity pass.
    events_with_dangling_refs: List[Dict[str, Any]] = Field(default_factory=list)

    # --- Spatial-anchor + co-presence repair (PR 2 of EventNode.at_location_id) ---
    # ``events_relocated``: ``{event_id: new_location_id}`` for every
    # event whose ``at_location_id`` was rewritten by the assembly
    # validator (either to repair a conflict with the actor's
    # reconstructed location, or to backfill from the actor when the
    # field was empty). ``copresence_repairs_applied`` counts
    # synthetic ``EntityStateSnapshot`` / ``ObjectStateSnapshot``
    # entries inserted to bring a participant to the event location.
    # ``copresence_repairs_skipped`` records cases where the validator
    # detected a co-presence violation but declined to auto-repair
    # (e.g. an entity is dead at fabula_time, or an object is
    # destroyed) — the auditor will flag these for human review.
    events_relocated: Dict[str, str] = Field(default_factory=dict)
    copresence_repairs_applied: int = 0
    copresence_repairs_skipped: List[Dict[str, Any]] = Field(default_factory=list)


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
    branch_label: Optional[str] = None,
) -> None:
    """Fold proposition / concern additions and snapshots into ``merged``.

    Idempotent: re-applying the same topology against the same world
    introduces no duplicates. Mutates ``merged`` in place and updates
    ``changeset`` counters.

    ``branch_label`` enables AMWN node-splitting for shadow merges:
    truth commits / framing snapshots / concerns / belief snapshots
    that would otherwise be blocked by the cross-branch guard are
    instead routed onto a per-branch split copy of the carrier in
    the appropriate sidecar (see ``_get_or_clone_shadow_proposition``
    and ``_get_or_clone_shadow_entity``).
    """
    from shadow_loom.models import Proposition, Concern  # local to keep import-time graph clean

    # Pre-compute suppressed-commit provenance once so per-call clone
    # helpers can trim the new sidecar copies consistently with the
    # ``_apply_deletions`` cascade (see same provenance rule there).
    _suppressed_event_ids: set[str] = set(getattr(topology, "suppressed_event_ids", None) or [])
    _suppressed_commits: set[tuple[str, int]] = set()
    _surviving_commits: set[tuple[str, int]] = set()
    if world_id == "shadow" and branch_label and _suppressed_event_ids:
        def _committed_props_at_local(evt: Any) -> List[tuple[str, int]]:
            ft = getattr(evt, "fabula_time", None)
            if ft is None:
                return []
            pairs: List[tuple[str, int]] = []
            pid_a = getattr(evt, "asserts_proposition_id", None)
            if pid_a:
                pairs.append((pid_a, int(ft)))
            pid_d = getattr(evt, "denies_proposition_id", None)
            if pid_d:
                pairs.append((pid_d, int(ft)))
            for pid_r in getattr(evt, "resolves_proposition_ids", None) or []:
                pairs.append((pid_r, int(ft)))
            return pairs
        for evt in merged.events:
            if evt.id in _suppressed_event_ids:
                _suppressed_commits.update(_committed_props_at_local(evt))
            else:
                _surviving_commits.update(_committed_props_at_local(evt))

    # Index existing propositions by id for O(1) lookup.
    prop_index: Dict[str, Proposition] = {p.proposition_id: p for p in merged.propositions}

    # 1. New propositions (genesis) — dedup on proposition_id.
    #
    # Branch routing: on shadow merges with a named branch, route
    # new propositions straight into the sidecar so factual reads
    # never see them. Falls back to the legacy shared-list append
    # for factual merges and for shadow merges without a named
    # branch (defensive — same fallback as entity routing).
    for pid, prop in topology.new_propositions.items():
        if pid in prop_index:
            continue
        if world_id == "shadow" and branch_label:
            sidecar = merged.shadow_propositions.setdefault(branch_label, {})
            if pid in sidecar:
                continue
            new_prop = prop.model_copy(update={"world_id": "shadow"})
            sidecar[pid] = new_prop
            # Make the sidecar entry visible to subsequent
            # ``prop_index`` lookups in this same call (truth
            # commits / framing snapshots in steps 2 & 3 below).
            prop_index[pid] = new_prop
            changeset.propositions_added += 1
            continue
        new_prop = prop.model_copy(update={"world_id": world_id}) if world_id != "factual" else prop
        merged.propositions.append(new_prop)
        prop_index[pid] = new_prop
        changeset.propositions_added += 1

    # 2. Truth commits → Proposition.truth_at_fabula
    #
    # Branch safety: a shadow merge must not flip the truth value of
    # a factual-tagged proposition (and vice versa); the truth-flip
    # would otherwise propagate into the canonical mainline via the
    # next read of ``truth_at_fabula``. Skipped commits are logged
    # at INFO with the ``[merge·affect]`` prefix so divergence is
    # auditable.
    for commit in topology.proposition_truth_commits:
        prop = prop_index.get(commit.proposition_id)
        if prop is None:
            logger.warning(
                "[merge·affect] Truth commit for unknown PROP %s — skipped.",
                commit.proposition_id,
            )
            continue
        prop_world = getattr(prop, "world_id", "factual") or "factual"
        if prop_world != world_id:
            if world_id == "shadow" and branch_label:
                # AMWN-split: materialise a per-branch clone of the
                # factual proposition and route the truth commit
                # onto the clone instead of mutating shared state.
                clone = _get_or_clone_shadow_proposition(
                    merged, commit.proposition_id,
                    branch_label=branch_label,
                    suppressed_commits=_suppressed_commits,
                    suppressed_event_ids=_suppressed_event_ids,
                    surviving_commits=_surviving_commits,
                )
                if clone is None:
                    logger.info(
                        "[merge·affect] Shadow truth commit on PROP %s "
                        "could not be cloned (no branch_label) — "
                        "cross-branch write blocked.",
                        commit.proposition_id,
                    )
                    continue
                prop = clone
                prop_index[commit.proposition_id] = clone
            else:
                logger.info(
                    "[merge·affect] Skipped truth commit on PROP %s "
                    "(prop.world_id=%s, merge_world_id=%s) — "
                    "cross-branch write blocked.",
                    commit.proposition_id, prop_world, world_id,
                )
                continue
        existing = prop.truth_at_fabula.get(commit.fabula_time)
        if existing is not None and existing == commit.truth:
            continue  # idempotent re-apply
        prop.truth_at_fabula[commit.fabula_time] = commit.truth
        changeset.proposition_truths_committed += 1

    # 3. Proposition framing snapshots → Proposition.state_timeline
    #
    # Branch safety: same rule as truth commits — a shadow merge
    # must not append snapshots onto a factual-tagged proposition.
    for snap in topology.proposition_snapshots:
        prop = prop_index.get(snap.proposition_id)
        if prop is None:
            logger.warning(
                "[merge·affect] Snapshot for unknown PROP %s — skipped.",
                snap.proposition_id,
            )
            continue
        prop_world = getattr(prop, "world_id", "factual") or "factual"
        if prop_world != world_id:
            if world_id == "shadow" and branch_label:
                clone = _get_or_clone_shadow_proposition(
                    merged, snap.proposition_id,
                    branch_label=branch_label,
                    suppressed_commits=_suppressed_commits,
                    suppressed_event_ids=_suppressed_event_ids,
                    surviving_commits=_surviving_commits,
                )
                if clone is None:
                    logger.info(
                        "[merge·affect] Shadow framing snapshot on PROP %s "
                        "could not be cloned (no branch_label) — "
                        "cross-branch write blocked.",
                        snap.proposition_id,
                    )
                    continue
                prop = clone
                prop_index[snap.proposition_id] = clone
            else:
                logger.info(
                    "[merge·affect] Skipped framing snapshot on PROP %s "
                    "(prop.world_id=%s, merge_world_id=%s) — "
                    "cross-branch write blocked.",
                    snap.proposition_id, prop_world, world_id,
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
    #
    # Branch safety: a shadow merge must not seed new concerns onto
    # a factual-tagged entity (and vice versa); the concerns would
    # otherwise become indistinguishable from canonical-mainline
    # affect once read back.
    for entity_id, concerns in topology.new_concerns.items():
        ent = merged.entities.get(entity_id)
        if ent is None:
            logger.warning(
                "[merge·affect] new_concerns for unknown entity %s — skipped.",
                entity_id,
            )
            continue
        ent_world = getattr(ent, "world_id", "factual") or "factual"
        if ent_world != world_id:
            if world_id == "shadow" and branch_label:
                # AMWN-split: route new concerns onto a per-branch
                # split copy of the entity so the factual entity
                # record is never mutated.
                clone = _get_or_clone_shadow_entity(
                    merged, entity_id,
                    branch_label=branch_label,
                    suppressed_event_ids=_suppressed_event_ids,
                )
                if clone is None:
                    logger.info(
                        "[merge·affect] Shadow new_concerns on %s "
                        "could not be cloned (no branch_label) — "
                        "cross-branch write blocked.",
                        entity_id,
                    )
                    continue
                ent = clone
            else:
                logger.info(
                    "[merge·affect] Skipped %d new_concern(s) on %s "
                    "(ent.world_id=%s, merge_world_id=%s) — "
                    "cross-branch write blocked.",
                    len(concerns), entity_id, ent_world, world_id,
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
            ent_world = getattr(ent, "world_id", "factual") or "factual"
            if ent_world != world_id:
                if world_id == "shadow" and branch_label:
                    clone = _get_or_clone_shadow_entity(
                        merged, holder_id,
                        branch_label=branch_label,
                        suppressed_event_ids=_suppressed_event_ids,
                    )
                    if clone is None:
                        logger.info(
                            "[merge·affect] Shadow ConcernSeed on %s "
                            "could not be cloned (no branch_label) — "
                            "cross-branch write blocked.",
                            holder_id,
                        )
                        continue
                    ent = clone
                else:
                    logger.info(
                        "[merge·affect] Skipped ConcernSeed on %s "
                        "(ent.world_id=%s, merge_world_id=%s) — "
                        "cross-branch write blocked.",
                        holder_id, ent_world, world_id,
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
                # Use explicit None checks so a legitimate 0.0
                # baseline (intentionally suppressed concern) is
                # preserved instead of silently coerced to 0.5 by
                # the truthiness-based ``or`` fallback chain.
                _baseline = getattr(seed, "baseline_salience", None)
                if _baseline is None:
                    _baseline = getattr(seed, "salience", None)
                if _baseline is None:
                    _baseline = 0.5
                concern = Concern(
                    world_id=world_id,
                    concern_id=ccn_id,
                    proposition_id=prop_id,
                    polarity=polarity,
                    kind=getattr(seed, "kind", None),
                    salience=float(_baseline),
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
        # Index concerns by ccn_id but keep a *list* so cross-entity
        # collisions (same CCN_ id appearing on two holders) route the
        # snapshot onto every matching concern instead of silently
        # binding to whichever was iterated last.
        #
        # On shadow merges with a named branch, we also need to find
        # the concern_id on entity *clones* in
        # ``merged.shadow_entities[branch_label]`` (whose ``world_id``
        # is ``"shadow"``) so a snapshot whose factual twin is being
        # masked by an AMWN-split clone lands on the clone instead.
        concern_index: Dict[str, List[tuple["Concern", str]]] = {}
        for ent in merged.entities.values():
            for c in ent.concerns:
                concern_index.setdefault(c.concern_id, []).append(
                    (c, ent.id),
                )
        if world_id == "shadow" and branch_label:
            for ent in (merged.shadow_entities.get(branch_label) or {}).values():
                for c in ent.concerns:
                    concern_index.setdefault(c.concern_id, []).append(
                        (c, ent.id),
                    )
        for snap in topology.concern_snapshots:
            targets = concern_index.get(snap.concern_id) or []
            if not targets:
                logger.warning(
                    "[merge·affect] Concern snapshot for unknown CCN %s — skipped.",
                    snap.concern_id,
                )
                continue
            # Branch safety: only stamp snapshots onto concerns whose
            # ``world_id`` matches the merge branch. On shadow merges
            # with a named branch, attempt to route a snapshot onto a
            # cross-branch concern via the entity sidecar before
            # giving up.
            in_branch = [
                (c, eid) for (c, eid) in targets
                if (getattr(c, "world_id", "factual") or "factual") == world_id
            ]
            if not in_branch and world_id == "shadow" and branch_label:
                # Try to clone the holder entity (any of the cross-
                # branch matches will do — pick the first) so its
                # concerns become shadow-tagged copies on the sidecar.
                _src_eid = targets[0][1]
                _clone = _get_or_clone_shadow_entity(
                    merged, _src_eid,
                    branch_label=branch_label,
                    suppressed_event_ids=_suppressed_event_ids,
                )
                if _clone is not None:
                    for cc in _clone.concerns:
                        if cc.concern_id == snap.concern_id and (
                            getattr(cc, "world_id", "factual") or "factual"
                        ) == "shadow":
                            in_branch.append((cc, _clone.id))
            if not in_branch:
                logger.info(
                    "[merge·affect] Skipped concern snapshot for CCN %s — "
                    "no concern with that id on branch=%s (found %d "
                    "cross-branch match(es)).",
                    snap.concern_id, world_id, len(targets),
                )
                continue
            for target, _eid in in_branch:
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
            # Branch safety: only stamp belief snapshots onto holders
            # whose ``world_id`` matches the merge branch. Without
            # this guard a shadow-fork affect snapshot would write
            # onto the factual entity timeline (or vice versa),
            # silently leaking counterfactual confidence drift onto
            # canonical character beliefs. Mirrors the
            # concern-snapshot guard a few blocks above.
            ent_world = getattr(ent, "world_id", "factual") or "factual"
            if ent_world != world_id:
                if world_id == "shadow" and branch_label:
                    clone = _get_or_clone_shadow_entity(
                        merged, bs.holder_id,
                        branch_label=branch_label,
                        suppressed_event_ids=_suppressed_event_ids,
                    )
                    if clone is None:
                        logger.info(
                            "[merge·affect] Shadow belief snapshot on %s "
                            "could not be cloned (no branch_label) — "
                            "cross-branch write blocked.",
                            bs.holder_id,
                        )
                        continue
                    ent = clone
                else:
                    logger.info(
                        "[merge\u00b7affect] Skipped belief snapshot on %s "
                        "(holder.world_id=%s, merge_world_id=%s) \u2014 "
                        "cross-branch write blocked.",
                        bs.holder_id, ent_world, world_id,
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
    merge_world_id: Literal["factual", "shadow"] = "factual",
    branch_label: Optional[str] = None,
) -> None:
    """Apply each ``EntityUpdate.belief_confidence_updates`` entry by
    overwriting the matching existing :class:`Belief` on the entity.

    The match is by ``target_id`` (and ``proposition_id`` when set). If
    no matching belief exists the update is dropped with a warning —
    creation should go through ``new_beliefs`` on the same EntityUpdate.

    Branch safety: a shadow merge must not rewrite confidence on a
    factual-tagged entity (and vice versa). Cross-branch writes are
    skipped at INFO level with the ``[merge·belief]`` prefix.
    """
    for eu in topology.entity_updates:
        for upd in eu.belief_confidence_updates:
            ent = merged.entities.get(eu.entity_id)
            if ent is None:
                continue
            ent_world = getattr(ent, "world_id", "factual") or "factual"
            if ent_world != merge_world_id:
                if merge_world_id == "shadow" and branch_label:
                    _sup = set(getattr(topology, "suppressed_event_ids", None) or [])
                    clone = _get_or_clone_shadow_entity(
                        merged, eu.entity_id,
                        branch_label=branch_label,
                        suppressed_event_ids=_sup,
                    )
                    if clone is None:
                        logger.info(
                            "[merge·belief] Shadow confidence update on %s "
                            "could not be cloned (no branch_label) — "
                            "cross-branch write blocked.",
                            eu.entity_id,
                        )
                        continue
                    ent = clone
                else:
                    logger.info(
                        "[merge·belief] Skipped confidence update on %s "
                        "(ent.world_id=%s, merge_world_id=%s) — "
                        "cross-branch write blocked.",
                        eu.entity_id, ent_world, merge_world_id,
                    )
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


def _get_or_clone_shadow_entity(
    merged: WorldStateV1,
    entity_id: str,
    *,
    branch_label: Optional[str],
    suppressed_event_ids: Optional[set[str]] = None,
) -> Optional["Entity"]:
    """Return the shadow-branch split copy of ``entity_id``, creating
    it on first touch via the AMWN node-splitting construction
    (Correa & Bareinboim 2025; see docs/academic-foundations.md §2.2).

    The split copy is materialised by deep-copying the factual entity
    into ``merged.shadow_entities[branch_label][entity_id]`` (lazy:
    only on the first shadow write that targets the entity — i.e.
    only when a ``do(·)`` makes the entity's ancestral context
    diverge from factual). Every snapshot whose ``triggered_by`` is
    in ``suppressed_event_ids`` is then trimmed off the clone's
    ``state_timeline`` because the structural-equation arc that
    produced it has been severed in this AMWN world. Beliefs whose
    ``acquired_via_event_id`` is in the suppressed set are likewise
    trimmed (same provenance argument). The clone is tagged
    ``world_id='shadow'``.

    Subsequent shadow merges read the existing clone (no re-clone,
    no re-trim) so accumulated shadow snapshots persist. Sibling
    shadow branches (different ``branch_label``) are independent
    AMWN worlds W*ₙ with their own split copies.

    Returns ``None`` when no factual entity with that id exists and
    no clone is present (caller logs as a normal unknown-entity skip).
    """
    if not branch_label:
        return None
    sidecar = merged.shadow_entities.setdefault(branch_label, {})
    if entity_id in sidecar:
        return sidecar[entity_id]
    factual = merged.entities.get(entity_id)
    if factual is None:
        return None
    clone = copy.deepcopy(factual)
    clone.world_id = "shadow"
    sup = set(suppressed_event_ids or [])
    if sup:
        clone.state_timeline = [
            s for s in clone.state_timeline
            if getattr(s, "triggered_by", None) not in sup
        ]
        clone.beliefs = [
            b for b in clone.beliefs
            if getattr(b, "acquired_via_event_id", None) not in sup
        ]
    sidecar[entity_id] = clone
    logger.info(
        "[merge\u00b7shadow-clone] AMWN-split entity %s on branch %r "
        "(suppressed=%d snapshot(s)/belief(s)).",
        entity_id, branch_label,
        len(sup) if sup else 0,
    )
    return clone


def _get_or_clone_shadow_object(
    merged: WorldStateV1,
    object_id: str,
    *,
    branch_label: Optional[str],
    suppressed_event_ids: Optional[set[str]] = None,
) -> Optional["NarrativeObject"]:
    """Return the shadow-branch split copy of ``object_id``.

    Mirror of :func:`_get_or_clone_shadow_entity` for
    ``NarrativeObject``: lazy AMWN-split clone of
    ``merged.objects[id]`` materialised in
    ``merged.shadow_objects[branch_label][object_id]`` on first
    shadow write. Snapshots on the clone whose ``triggered_by`` is
    in the suppressed-event set are trimmed off (severed causal
    arcs). Returns ``None`` when no factual object exists, or when
    no ``branch_label`` was provided (caller logs as usual).
    """
    if not branch_label:
        return None
    sidecar = merged.shadow_objects.setdefault(branch_label, {})
    if object_id in sidecar:
        return sidecar[object_id]
    factual = merged.objects.get(object_id)
    if factual is None:
        return None
    clone = copy.deepcopy(factual)
    clone.world_id = "shadow"
    sup = set(suppressed_event_ids or [])
    if sup:
        clone.state_timeline = [
            s for s in clone.state_timeline
            if getattr(s, "triggered_by", None) not in sup
        ]
    sidecar[object_id] = clone
    logger.info(
        "[merge\u00b7shadow-clone] AMWN-split object %s on branch %r "
        "(suppressed=%d snapshot(s)).",
        object_id, branch_label,
        len(sup) if sup else 0,
    )
    return clone


def _get_or_clone_shadow_world_trait(
    merged: WorldStateV1,
    trait_id: str,
    *,
    branch_label: Optional[str],
    suppressed_event_ids: Optional[set[str]] = None,
) -> Optional["GlobalTrait"]:
    """Mirror of :func:`_get_or_clone_shadow_entity` for
    ``GlobalTrait``. Trims ``state_timeline`` by ``triggered_by``.
    """
    if not branch_label:
        return None
    sidecar = merged.shadow_world_traits.setdefault(branch_label, {})
    if trait_id in sidecar:
        return sidecar[trait_id]
    factual = merged.world_traits.get(trait_id)
    if factual is None:
        return None
    clone = copy.deepcopy(factual)
    clone.world_id = "shadow"
    sup = set(suppressed_event_ids or [])
    if sup and hasattr(clone, "state_timeline") and clone.state_timeline:
        clone.state_timeline = [
            s for s in clone.state_timeline
            if getattr(s, "triggered_by", None) not in sup
        ]
    sidecar[trait_id] = clone
    logger.info(
        "[merge\u00b7shadow-clone] AMWN-split world_trait %s on branch %r "
        "(suppressed=%d snapshot(s)).",
        trait_id, branch_label,
        len(sup) if sup else 0,
    )
    return clone


def _get_or_clone_shadow_proposition(
    merged: WorldStateV1,
    proposition_id: str,
    *,
    branch_label: Optional[str],
    suppressed_commits: Optional[set[tuple[str, int]]] = None,
    suppressed_event_ids: Optional[set[str]] = None,
    surviving_commits: Optional[set[tuple[str, int]]] = None,
) -> Optional["Proposition"]:
    """Mirror of :func:`_get_or_clone_shadow_entity` for
    ``Proposition``.

    Lazy AMWN-split clone of ``merged.propositions[id]`` (located by
    id scan; the registry is a list) materialised in
    ``merged.shadow_propositions[branch_label][proposition_id]``.

    Trim semantics on the clone:
      * ``truth_at_fabula[ft]`` is dropped iff
        ``(proposition_id, ft) in suppressed_commits`` AND not in
        ``surviving_commits`` — same provenance rule used by the
        cascade trim in :func:`_apply_deletions`.
      * ``state_timeline`` entries whose ``triggered_by`` is in
        ``suppressed_event_ids`` are dropped.

    Returns ``None`` when the proposition does not exist or no
    ``branch_label`` was supplied.
    """
    if not branch_label:
        return None
    sidecar = merged.shadow_propositions.setdefault(branch_label, {})
    if proposition_id in sidecar:
        return sidecar[proposition_id]
    factual: Optional["Proposition"] = None
    for p in merged.propositions:
        if p.proposition_id == proposition_id:
            factual = p
            break
    if factual is None:
        return None
    clone = copy.deepcopy(factual)
    clone.world_id = "shadow"
    sup_commits = suppressed_commits or set()
    surv_commits = surviving_commits or set()
    if sup_commits and isinstance(clone.truth_at_fabula, dict):
        to_drop = {
            int(ft) for (pid, ft) in sup_commits
            if pid == proposition_id and (pid, ft) not in surv_commits
        }
        if to_drop:
            clone.truth_at_fabula = {
                int(k): v for k, v in clone.truth_at_fabula.items()
                if int(k) not in to_drop
            }
    sup_events = set(suppressed_event_ids or [])
    if sup_events and getattr(clone, "state_timeline", None):
        clone.state_timeline = [
            s for s in clone.state_timeline
            if getattr(s, "triggered_by", None) not in sup_events
        ]
    sidecar[proposition_id] = clone
    logger.info(
        "[merge\u00b7shadow-clone] AMWN-split proposition %s on branch %r "
        "(suppressed=%d commit(s)).",
        proposition_id, branch_label,
        len(sup_commits) if sup_commits else 0,
    )
    return clone


def _apply_deletions(
    merged: WorldStateV1,
    topology: "ChunkTopology",
    *,
    changeset: "MergeChangeset",
    merge_world_id: Literal["factual", "shadow"] = "factual",
) -> None:
    """Run the deletion pass before additive merge sections.

    Cascades dependent edges/snapshots when a parent node is removed
    (e.g. removing an entity also drops its concerns and any social
    edges or beliefs naming it).

    Branch safety: a merge running on ``world_id == "shadow"`` must
    not destroy factual-tagged events / channels / entities / objects
    / locations / propositions / concerns / edges, and vice versa.
    Without this guard a shadow-branch manual edit (which carries
    populated ``removed_*`` fields, e.g. retracted utterances) would
    silently delete canonical mainline state. Shadow merges therefore
    only delete ``world_id == "shadow"`` items; factual merges only
    delete ``world_id == "factual"`` items. Items with no world_id
    attribute (legacy data) are conservatively treated as factual.
    """

    def _wid(obj) -> str:
        # Tolerate elements missing the field (legacy / dict-coerced)
        # and treat them as factual so we never silently nuke them.
        return getattr(obj, "world_id", "factual") or "factual"

    def _branch_match(obj) -> bool:
        return _wid(obj) == merge_world_id

    # --- Events
    if topology.removed_event_ids:
        drop_req = set(topology.removed_event_ids)
        # Branch-filter: only delete events on the merge's branch.
        evt_index = {e.id: e for e in merged.events}
        drop = {
            eid for eid in drop_req
            if eid in evt_index and _branch_match(evt_index[eid])
        }
        skipped = drop_req - drop
        if skipped:
            logger.info(
                "[merge·delete] Skipped %d event deletion(s) on branch=%s "
                "because targets live on the other branch: %s",
                len(skipped), merge_world_id, sorted(skipped),
            )
        before = len(merged.events)
        merged.events = [e for e in merged.events if e.id not in drop]
        removed_n = before - len(merged.events)
        changeset.events_removed += removed_n
        # Cascade: drop causal/spatial/social edges referencing dropped events.
        # Cascade is unconditional on the dropped IDs (those events are gone
        # on this branch by definition); branch-mismatched edges naming a
        # surviving cross-branch event are left alone.
        before_c = len(merged.causal_topology)
        merged.causal_topology = [
            c for c in merged.causal_topology
            if not (
                (c.source_id in drop or c.target_id in drop)
                and _branch_match(c)
            )
        ]
        changeset.causal_edges_removed += before_c - len(merged.causal_topology)

    # --- Shadow suppression (counterfactual / intervention cascade).
    # Unlike ``removed_event_ids``, suppression bypasses the
    # ``_branch_match`` filter: a shadow merge is allowed to delete
    # factual-ancestor events whose causal preconditions no longer
    # hold under the do-intervention. Without this, the persisted
    # shadow VersionRow's snapshot would still list those factual
    # events (Mrs Coady's heart-attack death cascading from a
    # dog-killing that was intervened away), so a downstream
    # interrogation reading the JSON would contradict the prose.
    # Closure is computed caller-side in
    # ``_augment_topology_with_sandbox_deltas`` using Pearl's
    # disjunctive structural-equation reading on ``chain_reaction``
    # edges (an effect persists if any surviving sufficient cause
    # remains; pruned only when every ``chain_reaction`` parent is
    # suppressed).
    if topology.suppressed_event_ids:
        sup = set(topology.suppressed_event_ids)
        # Capture the suppressed event records (with their
        # ``asserts_proposition_id`` / ``denies_proposition_id`` /
        # ``resolves_proposition_ids`` provenance) BEFORE we drop them
        # from ``merged.events`` so the proposition-truth cascade below
        # has the data it needs.
        suppressed_events_records = [e for e in merged.events if e.id in sup]
        before_e = len(merged.events)
        merged.events = [e for e in merged.events if e.id not in sup]
        suppressed_n = before_e - len(merged.events)
        changeset.events_removed += suppressed_n
        # Cascade: causal edges touching suppressed events.
        before_c = len(merged.causal_topology)
        merged.causal_topology = [
            c for c in merged.causal_topology
            if c.source_id not in sup and c.target_id not in sup
        ]
        changeset.causal_edges_removed += before_c - len(merged.causal_topology)
        # Cascade: drop entity state_timeline snapshots whose
        # ``triggered_by`` references a suppressed event (the
        # entity-level mutation those snapshots applied is no longer
        # justified). Object, world-trait, proposition, and concern
        # timelines carry the same field so we walk them too.
        #
        # We must walk both the factual ``merged.entities`` dict AND
        # any previously-materialised AMWN-split copies in
        # ``merged.shadow_entities`` \u2014 a clone created on an
        # earlier shadow merge had its timeline trimmed against the
        # suppression set known at clone-time; the current merge may
        # be expanding that set (e.g. closure widened to include a
        # newly-suppressed descendant event), so existing clones
        # need a fresh trim pass.
        def _all_entity_records() -> List[Any]:
            records: List[Any] = list(merged.entities.values())
            for sidecar in (merged.shadow_entities or {}).values():
                records.extend(sidecar.values())
            return records

        def _all_object_records() -> List[Any]:
            records: List[Any] = list(merged.objects.values())
            for sidecar in (merged.shadow_objects or {}).values():
                records.extend(sidecar.values())
            return records

        def _all_world_trait_records() -> List[Any]:
            records: List[Any] = list(merged.world_traits.values())
            for sidecar in (merged.shadow_world_traits or {}).values():
                records.extend(sidecar.values())
            return records

        def _all_proposition_records() -> List[Any]:
            records: List[Any] = list(merged.propositions)
            for sidecar in (merged.shadow_propositions or {}).values():
                records.extend(sidecar.values())
            return records
        for ent in _all_entity_records():
            ent.state_timeline = [
                s for s in ent.state_timeline
                if getattr(s, "triggered_by", None) not in sup
            ]
        for obj in _all_object_records():
            obj.state_timeline = [
                s for s in obj.state_timeline
                if getattr(s, "triggered_by", None) not in sup
            ]
        for wt in _all_world_trait_records():
            wt.state_timeline = [
                s for s in wt.state_timeline
                if getattr(s, "triggered_by", None) not in sup
            ]
        for prop in _all_proposition_records():
            if getattr(prop, "state_timeline", None):
                prop.state_timeline = [
                    s for s in prop.state_timeline
                    if getattr(s, "triggered_by", None) not in sup
                ]
        for ent in _all_entity_records():
            for concern in ent.concerns:
                if getattr(concern, "state_timeline", None):
                    concern.state_timeline = [
                        s for s in concern.state_timeline
                        if getattr(s, "triggered_by", None) not in sup
                    ]
        # Cascade: drop beliefs whose ``acquired_via_event_id``
        # provenance points at a suppressed event. The bridge in
        # ``_augment_topology_with_sandbox_deltas`` already emits
        # belief-invalidation snapshots for the direct prune set;
        # this guards against closure-expansion drift where a
        # disjunctively-suppressed descendant event was the actual
        # provenance source. Walks sidecar clones too (same rationale
        # as the state_timeline trim above).
        for ent in _all_entity_records():
            ent.beliefs = [
                b for b in ent.beliefs
                if getattr(b, "acquired_via_event_id", None) not in sup
            ]
        # Cascade: drop ``Proposition.truth_at_fabula`` entries whose
        # only justification was an asserting/denying/resolving event
        # that has now been suppressed. Without this, the proposition
        # ground-truth dict (which was deep-copied from factual at
        # ``merge()`` start) keeps reporting the factual outcome on
        # the shadow branch \u2014 e.g. ``PROP_MRS_COADY_DIES`` remains
        # ``True`` even after the dog-killing root cause is severed,
        # so interrogation reads ``True`` while the prose narrates
        # her alive.
        #
        # Provenance rule (conservative, no schema change required):
        # an entry ``truth_at_fabula[ft]`` is dropped iff some
        # suppressed event committed to that prop at ``ft`` AND no
        # surviving event commits to the same prop at the same
        # ``ft``. Entries with no event-level provenance at all
        # (story-prior commits, physics clamps without an
        # originating event) are preserved \u2014 we cannot show they
        # were caused by suppressed events.
        if suppressed_events_records:
            def _committed_props_at(evt: Any) -> List[tuple[str, int]]:
                """Return ``(prop_id, fabula_time)`` pairs this event commits."""
                ft = getattr(evt, "fabula_time", None)
                if ft is None:
                    return []
                pairs: List[tuple[str, int]] = []
                pid_a = getattr(evt, "asserts_proposition_id", None)
                if pid_a:
                    pairs.append((pid_a, int(ft)))
                pid_d = getattr(evt, "denies_proposition_id", None)
                if pid_d:
                    pairs.append((pid_d, int(ft)))
                for pid_r in getattr(evt, "resolves_proposition_ids", None) or []:
                    pairs.append((pid_r, int(ft)))
                return pairs
            suppressed_commits: set[tuple[str, int]] = set()
            for evt in suppressed_events_records:
                suppressed_commits.update(_committed_props_at(evt))
            surviving_commits: set[tuple[str, int]] = set()
            for evt in merged.events:
                surviving_commits.update(_committed_props_at(evt))
            to_drop = suppressed_commits - surviving_commits
            if to_drop:
                drop_by_prop: Dict[str, set[int]] = {}
                for pid, ft in to_drop:
                    drop_by_prop.setdefault(pid, set()).add(ft)
                dropped_n = 0
                for prop in _all_proposition_records():
                    fts = drop_by_prop.get(prop.proposition_id)
                    if not fts:
                        continue
                    if not isinstance(prop.truth_at_fabula, dict):
                        continue
                    before_t = len(prop.truth_at_fabula)
                    prop.truth_at_fabula = {
                        int(k): v
                        for k, v in prop.truth_at_fabula.items()
                        if int(k) not in fts
                    }
                    dropped_n += before_t - len(prop.truth_at_fabula)
                if dropped_n:
                    logger.info(
                        "[merge\u00b7shadow-suppress] Dropped %d "
                        "Proposition.truth_at_fabula entr%s with "
                        "suppressed-event provenance "
                        "(no surviving commit at same fabula_time).",
                        dropped_n, "y" if dropped_n == 1 else "ies",
                    )

    # --- Causal edges (explicit keys)
    if topology.removed_causal_edge_keys:
        keys = {tuple(k) for k in topology.removed_causal_edge_keys}
        before = len(merged.causal_topology)
        merged.causal_topology = [
            c for c in merged.causal_topology
            if not (
                (c.source_id, c.target_id, c.causality_type, c.fabula_time) in keys
                and _branch_match(c)
            )
        ]
        changeset.causal_edges_removed += before - len(merged.causal_topology)

    # --- Social dyads
    if topology.removed_social_dyad_keys:
        keys = {tuple(k) for k in topology.removed_social_dyad_keys}
        before = len(merged.social_topology)
        merged.social_topology = [
            r for r in merged.social_topology
            if not (
                (r.source_entity_id, r.target_entity_id) in keys
                and _branch_match(r)
            )
        ]
        changeset.social_edges_removed += before - len(merged.social_topology)

    # --- Spatial edges
    if topology.removed_spatial_keys:
        keys = {tuple(k) for k in topology.removed_spatial_keys}
        before = len(merged.spatial_topology)
        merged.spatial_topology = [
            s for s in merged.spatial_topology
            if not (
                (s.source_id, s.target_id) in keys
                and _branch_match(s)
            )
        ]
        changeset.spatial_edges_removed += before - len(merged.spatial_topology)

    # --- Channels
    if topology.removed_channel_ids:
        drop_req = set(topology.removed_channel_ids)
        drop_chan = {
            cid for cid in drop_req
            if cid in merged.channels
            and _branch_match(merged.channels[cid])
        }
        skipped = drop_req - drop_chan
        if skipped:
            logger.info(
                "[merge·delete] Skipped %d channel deletion(s) on branch=%s "
                "(other-branch targets): %s",
                len(skipped), merge_world_id, sorted(skipped),
            )
        for cid in drop_chan:
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
        drop_req = set(topology.removed_entity_ids)
        drop = {
            eid for eid in drop_req
            if eid in merged.entities
            and _branch_match(merged.entities[eid])
        }
        skipped = drop_req - drop
        if skipped:
            logger.info(
                "[merge·delete] Skipped %d entity deletion(s) on branch=%s "
                "(other-branch targets): %s",
                len(skipped), merge_world_id, sorted(skipped),
            )
        for eid in drop:
            del merged.entities[eid]
            changeset.entities_removed += 1
        # Drop social edges naming a dropped entity (only on this branch).
        before_s = len(merged.social_topology)
        merged.social_topology = [
            r for r in merged.social_topology
            if not (
                (r.source_entity_id in drop or r.target_entity_id in drop)
                and _branch_match(r)
            )
        ]
        changeset.social_edges_removed += before_s - len(merged.social_topology)
        # Drop other entities' beliefs targeting a dropped entity.
        for ent in merged.entities.values():
            ent.beliefs = [b for b in ent.beliefs if b.target_id not in drop]

    # --- Objects
    for oid in topology.removed_object_ids:
        obj = merged.objects.get(oid)
        if obj is None:
            continue
        if not _branch_match(obj):
            logger.info(
                "[merge·delete] Skipped object deletion on branch=%s "
                "(other-branch target): %s",
                merge_world_id, oid,
            )
            continue
        del merged.objects[oid]
        changeset.objects_removed += 1

    # --- Locations
    for lid in topology.removed_location_ids:
        loc = merged.locations.get(lid)
        if loc is None:
            continue
        if not _branch_match(loc):
            logger.info(
                "[merge·delete] Skipped location deletion on branch=%s "
                "(other-branch target): %s",
                merge_world_id, lid,
            )
            continue
        del merged.locations[lid]
        changeset.locations_removed += 1

    # --- World traits
    for wid in topology.removed_world_trait_ids:
        wt = merged.world_traits.get(wid)
        if wt is None:
            continue
        if not _branch_match(wt):
            logger.info(
                "[merge·delete] Skipped world_trait deletion on branch=%s "
                "(other-branch target): %s",
                merge_world_id, wid,
            )
            continue
        del merged.world_traits[wid]
        changeset.world_traits_removed += 1

    # --- Propositions (also drop concerns referencing them; scrub
    # proposition pointers on surviving events and beliefs).
    if topology.removed_proposition_ids:
        drop_req = set(topology.removed_proposition_ids)
        prop_index = {p.proposition_id: p for p in merged.propositions}
        drop = {
            pid for pid in drop_req
            if pid in prop_index and _branch_match(prop_index[pid])
        }
        skipped = drop_req - drop
        if skipped:
            logger.info(
                "[merge·delete] Skipped %d proposition deletion(s) on "
                "branch=%s (other-branch targets): %s",
                len(skipped), merge_world_id, sorted(skipped),
            )
        before = len(merged.propositions)
        merged.propositions = [
            p for p in merged.propositions if p.proposition_id not in drop
        ]
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
        drop_pairs = set((eid, cid) for eid, cid in topology.removed_concern_ids)
        removed_cids: set[str] = set()
        for eid, cid in drop_pairs:
            ent = merged.entities.get(eid)
            if ent is None:
                continue
            kept: list = []
            for c in ent.concerns:
                if c.concern_id == cid and _branch_match(c):
                    changeset.concerns_removed += 1
                    removed_cids.add(cid)
                    continue
                kept.append(c)
            ent.concerns = kept
        # Scrub counter_concern_ids on surviving concerns.
        for ent in merged.entities.values():
            for c in ent.concerns:
                ccids = getattr(c, "counter_concern_ids", None)
                if ccids:
                    c.counter_concern_ids = [
                        x for x in ccids if x not in removed_cids
                    ]
                    c.counter_concern_ids = [
                        x for x in ccids if x not in removed_cids
                    ]


def _apply_supersession(
    merged: WorldStateV1,
    topology: "ChunkTopology",
    *,
    changeset: "MergeChangeset",
    merge_world_id: Literal["factual", "shadow"] = "factual",
) -> None:
    """Stamp ``superseded_by_event_id`` on overridden events and rewrite
    cross-references on beliefs / concerns / propositions / new events.

    Branch safety: a shadow merge must not stamp supersession on
    factual-tagged events (or rewrite belief provenance on factual-
    tagged entities). Cross-branch writes are skipped at INFO level
    with the ``[merge·supersede]`` prefix.
    """
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
        evt_world = getattr(old_evt, "world_id", "factual") or "factual"
        if evt_world != merge_world_id:
            logger.info(
                "[merge·supersede] Skipped supersession on %s "
                "(evt.world_id=%s, merge_world_id=%s) — "
                "cross-branch write blocked.",
                old_id, evt_world, merge_world_id,
            )
            continue
        if old_evt.superseded_by_event_id == new_id:
            continue
        old_evt.superseded_by_event_id = new_id
        changeset.events_superseded += 1

    # Rewrite belief provenance — only on holders that match the
    # merge branch, so a shadow supersession can't silently rewrite
    # factual belief provenance pointers.
    for ent in merged.entities.values():
        ent_world = getattr(ent, "world_id", "factual") or "factual"
        if ent_world != merge_world_id:
            continue
        for b in ent.beliefs:
            if b.acquired_via_event_id in mapping:
                b.acquired_via_event_id = mapping[b.acquired_via_event_id]

    # Rewrite Proposition.truth_at_fabula? No — those are keyed by time, not event.
    # But events' resolves_proposition_ids are still valid; we leave the
    # old event's list intact and rely on the supersede pointer for
    # downstream readers that prefer the override.


def _populate_dangling_ref_ledger(
    merged: WorldStateV1,
    changeset: "MergeChangeset",
) -> None:
    """Detect events that reference unknown ids and record them on the
    changeset's ``events_with_dangling_refs`` field.

    Resolution scope:

    - ``actor_ids`` and ``target_ids`` may name entities, objects, or
      locations (e.g. travel events target a location). All three
      registries are accepted.
    - Ids matching the project's reserved sentinel patterns
      (``ENV_*``, ``UNKNOWN_*``, ``ANON_*``) are tolerated because
      the physics agent legitimately emits them for ambient or
      anonymous referents.

    The ledger is purely diagnostic — the events stay in the world
    model. The UI can render a warning chip from this list and the
    user can promote dangling refs into a follow-up
    ``query.introduce`` payload.
    """
    known_ids = (
        set(merged.entities.keys())
        | set(merged.objects.keys())
        | set(merged.locations.keys())
        | set(merged.world_traits.keys())
        | {evt.id for evt in merged.events}
        | {p.proposition_id for p in merged.propositions}
        | set((merged.channels or {}).keys())
    )
    sentinel_prefixes = (
        "ENV_", "UNKNOWN_", "ANON_", "NARRATOR_",
        # Generic stand-ins the physics agent uses for unspecified
        # crowds / abstractions; tolerated rather than promoted.
        "GROUP_", "CROWD_", "AUDIENCE_",
    )

    def _is_sentinel(rid: str) -> bool:
        return any(rid.startswith(p) for p in sentinel_prefixes)

    for evt in merged.events:
        for field_name in ("actor_ids", "target_ids"):
            ids = getattr(evt, field_name, None) or []
            missing = [
                rid for rid in ids
                if rid and rid not in known_ids and not _is_sentinel(rid)
            ]
            if missing:
                changeset.events_with_dangling_refs.append({
                    "event_id": evt.id,
                    "field": field_name,
                    "missing_ids": missing,
                })


# Logger that the ``shadow_loom.ingestion_diagnostics`` handler captures.
_INGESTION_LOGGER = logging.getLogger("shadow_loom.ingestion")


def _apply_event_spatial_anchor_repairs(
    merged: WorldStateV1,
    changeset: "MergeChangeset",
) -> None:
    """Validate ``EventNode.at_location_id`` and auto-repair co-presence.

    For each event:
      * If ``at_location_id`` references an unknown LOC_ id → log
        ``[Validator·EventLocation] event_location_unknown`` and clear
        the field (the actor-fallback path will take over).
      * If ``at_location_id`` is set and disagrees with the primary
        actor's reconstructed location at ``fabula_time`` → log
        ``[Validator·EventLocation] event_location_conflict`` and
        insert an :class:`EntityStateSnapshot` moving the actor to the
        event's location at ``fabula_time``. The auditor's
        ``event_copresence_violation`` rule will pick up any remaining
        non-actor co-presence gaps; this pass intentionally only
        repairs the *primary* actor so we never silently teleport
        targets/witnesses around.
      * If ``at_location_id`` is empty BUT the event has actors and the
        primary actor has a known reconstructed location → backfill
        ``at_location_id`` from the actor (recorded on
        ``changeset.events_relocated``).

    Utterances with ``via_channel_id`` are exempt — channel-mediated
    speech-acts do not require physical co-location.
    """
    if not merged.events:
        return

    known_locs = set(merged.locations.keys())

    for evt in merged.events:
        # Channel-mediated utterances need no spatial anchor.
        if evt.event_type == "utterance" and evt.via_channel_id:
            continue

        # 1) Unknown LOC_ id → clear and log.
        if evt.at_location_id and evt.at_location_id not in known_locs:
            _INGESTION_LOGGER.warning(
                "[Validator·EventLocation] event_location_unknown: event %r "
                "names at_location_id=%r which is not in world.locations; "
                "clearing field and falling back to actor reconstruction.",
                evt.id, evt.at_location_id,
            )
            evt.at_location_id = None

        # Resolve primary actor (speaker_id wins for utterances).
        primary: Optional[str] = None
        if evt.speaker_id:
            primary = evt.speaker_id
        elif evt.actor_ids:
            primary = evt.actor_ids[0]

        actor_loc: Optional[str] = None
        actor_obj = merged.entities.get(primary) if primary else None
        if actor_obj is not None:
            try:
                actor_loc = reconstruct_entity_at(
                    actor_obj, evt.fabula_time,
                ).get("location_id")
            except Exception:
                actor_loc = getattr(actor_obj, "location_id", None)

        # 2) Conflict → repair by moving the primary actor.
        if evt.at_location_id and actor_loc and actor_loc != evt.at_location_id:
            # Skip auto-repair if the actor is dead at fabula_time —
            # surface as a skip so the auditor flags it.
            try:
                actor_status = reconstruct_entity_at(
                    actor_obj, evt.fabula_time,
                ).get("status")
            except Exception:
                actor_status = getattr(actor_obj, "status", "healthy")
            if actor_status == "dead":
                changeset.copresence_repairs_skipped.append({
                    "event_id": evt.id,
                    "participant_id": primary,
                    "reason": "actor_dead_at_fabula_time",
                    "fabula_time": evt.fabula_time,
                })
                _INGESTION_LOGGER.warning(
                    "[Validator·EventLocation] event_copresence_conflict: "
                    "event %r primary actor %r is dead at fabula=%d; "
                    "skipping co-presence repair.",
                    evt.id, primary, evt.fabula_time,
                )
                continue
            _INGESTION_LOGGER.info(
                "[Auto-Fix·EventLocation] event_location_conflict: event %r "
                "at %r but actor %r reconstructed at %r at fabula=%d; "
                "inserting EntityStateSnapshot to repair co-presence.",
                evt.id, evt.at_location_id, primary, actor_loc, evt.fabula_time,
            )
            actor_obj.state_timeline.append(
                EntityStateSnapshot(
                    fabula_time=evt.fabula_time,
                    triggered_by=evt.id,
                    location_id=evt.at_location_id,
                )
            )
            actor_obj.state_timeline.sort(key=lambda s: s.fabula_time)
            changeset.copresence_repairs_applied += 1
            continue

        # 3) Backfill from actor when empty.
        if not evt.at_location_id and actor_loc:
            evt.at_location_id = actor_loc
            changeset.events_relocated[evt.id] = actor_loc
            _INGESTION_LOGGER.info(
                "[Auto-Fix·EventLocation] event_location_backfilled: event "
                "%r had no at_location_id; backfilled %r from primary actor "
                "%r at fabula=%d.",
                evt.id, actor_loc, primary, evt.fabula_time,
            )


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
        world_id: Literal["factual", "shadow"] = "factual",
        branch_label: Optional[str] = None,
    ) -> "VersionedWorldModel":
        """Create a new versioned world model from an existing WorldStateV1.

        The incoming ``world_state`` is deep-copied so that subsequent
        merges never mutate the caller's original.

        ``world_id`` / ``branch_label`` seed the synthetic v0 history
        entry's branch identity. Pipeline branch routing (Continue,
        What-If, etc.) and the prose renderer's AMWN filter both
        consult ``history[-1].world_id``; without seeding these from
        the caller, every load of a shadow snapshot would silently
        present as factual — so a ``Continue`` issued from a shadow
        fork in the MCP would land back on the mainline. Defaults
        keep behaviour for callers that legitimately start from a new
        factual world (ingestion, tests).
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
                    world_id=world_id,
                    branch_label=branch_label,
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
        _apply_deletions(
            merged, topology, changeset=changeset,
            merge_world_id=world_id,
        )

        # --- Genesis-promoted nodes (entities / objects / locations /
        # world_traits) — written first so subsequent edge / event /
        # entity_update merges can reference them. Existing IDs keep
        # their identity; non-destructive attribute backfill fills in
        # missing fields (longer description, additional traits /
        # affordances / domains, set-union of constants) from the
        # incoming record so re-extractions enrich rather than
        # silently drop data.
        #
        # Branch safety: a shadow merge must NOT backfill a factual-
        # tagged node (and vice versa). Cross-branch backfill would
        # silently leak shadow-derived traits / beliefs / affordances
        # into the canonical mainline node. When the existing node
        # belongs to the other branch, the incoming record is dropped
        # at INFO level with a ``[merge\u00b7genesis]`` skip log; if
        # the caller wants the node on this branch, the topology must
        # tag it with ``world_id == merge_world_id``.
        def _genesis_branch_match(existing_obj) -> bool:
            existing_world = (
                getattr(existing_obj, "world_id", "factual") or "factual"
            )
            return existing_world == world_id

        for eid, ent in topology.new_entities.items():
            existing = merged.entities.get(eid)
            if existing is not None:
                if not _genesis_branch_match(existing):
                    logger.info(
                        "[merge\u00b7genesis] Skipped entity backfill on %s "
                        "(existing.world_id=%s, merge_world_id=%s) \u2014 "
                        "cross-branch write blocked.",
                        eid,
                        getattr(existing, "world_id", "factual"),
                        world_id,
                    )
                    continue
                merged.entities[eid] = _backfill_entity(existing, ent)
                logger.debug(
                    "[VersionedWorldModel\u00b7merge] Spawn entity %s already exists \u2014 backfilled.",
                    eid,
                )
                continue
            merged.entities[eid] = ent
            changeset.entities_added += 1
        for oid, obj in topology.new_objects.items():
            existing_obj = merged.objects.get(oid)
            if existing_obj is not None:
                if not _genesis_branch_match(existing_obj):
                    logger.info(
                        "[merge\u00b7genesis] Skipped object backfill on %s "
                        "(existing.world_id=%s, merge_world_id=%s) \u2014 "
                        "cross-branch write blocked.",
                        oid,
                        getattr(existing_obj, "world_id", "factual"),
                        world_id,
                    )
                    continue
                merged.objects[oid] = _backfill_object(existing_obj, obj)
                logger.debug(
                    "[VersionedWorldModel\u00b7merge] Spawn object %s already exists \u2014 backfilled.",
                    oid,
                )
                continue
            merged.objects[oid] = obj
            changeset.objects_added += 1
        for lid, loc in topology.new_locations.items():
            existing_loc = merged.locations.get(lid)
            if existing_loc is not None:
                if not _genesis_branch_match(existing_loc):
                    logger.info(
                        "[merge\u00b7genesis] Skipped location backfill on %s "
                        "(existing.world_id=%s, merge_world_id=%s) \u2014 "
                        "cross-branch write blocked.",
                        lid,
                        getattr(existing_loc, "world_id", "factual"),
                        world_id,
                    )
                    continue
                merged.locations[lid] = _backfill_location(existing_loc, loc)
                logger.debug(
                    "[VersionedWorldModel\u00b7merge] Spawn location %s already exists \u2014 backfilled.",
                    lid,
                )
                continue
            merged.locations[lid] = loc
            changeset.locations_added += 1
        for wid, wt in topology.new_world_traits.items():
            existing_wt = merged.world_traits.get(wid)
            if existing_wt is not None:
                if not _genesis_branch_match(existing_wt):
                    logger.info(
                        "[merge\u00b7genesis] Skipped world_trait backfill on %s "
                        "(existing.world_id=%s, merge_world_id=%s) \u2014 "
                        "cross-branch write blocked.",
                        wid,
                        getattr(existing_wt, "world_id", "factual"),
                        world_id,
                    )
                    continue
                merged.world_traits[wid] = _backfill_world_trait(existing_wt, wt)
                logger.debug(
                    "[VersionedWorldModel\u00b7merge] World trait %s already exists \u2014 backfilled.",
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
        # Shadow-branch writes that target a factual entity are
        # routed onto a per-branch SPLIT COPY in
        # ``merged.shadow_entities`` (AMWN node-splitting; Correa &
        # Bareinboim 2025 — see
        # :meth:`WorldStateV1.entities_for_branch` and
        # docs/academic-foundations.md §2.2). The split copy is
        # materialised lazily on first touch via
        # :func:`_get_or_clone_shadow_entity`, which also severs the
        # incoming structural-equation arcs corresponding to
        # ``topology.suppressed_event_ids`` from the clone's seeded
        # timeline so no pruned-cause snapshot survives on the
        # split. Factual writes (``world_id=='factual'``) keep their
        # original same-branch path and write directly onto
        # ``merged.entities[id]``.
        _suppressed_set = set(topology.suppressed_event_ids or [])
        for eu in topology.entity_updates:
            entity = merged.entities.get(eu.entity_id)
            if entity is None:
                logger.warning(
                    "Entity update for unknown entity '%s' — skipped.", eu.entity_id,
                )
                changeset.entity_updates_skipped.append(eu.entity_id)
                continue
            holder_world = getattr(entity, "world_id", "factual") or "factual"
            target_entity: Optional["Entity"] = entity
            if world_id == "shadow" and holder_world == "factual":
                # AMWN node-split: route onto a per-branch split copy
                # so the factual world is never mutated and the
                # shadow branch carries its own independently-
                # replayable state_timeline.
                target_entity = _get_or_clone_shadow_entity(
                    merged, eu.entity_id,
                    branch_label=branch_label,
                    suppressed_event_ids=_suppressed_set,
                )
                if target_entity is None:
                    logger.info(
                        "[merge\u00b7entity-update] Shadow write for %s "
                        "could not be cloned (no branch_label) \u2014 "
                        "cross-branch write blocked.",
                        eu.entity_id,
                    )
                    changeset.entity_updates_skipped.append(eu.entity_id)
                    continue
            elif holder_world != world_id:
                # Remaining mismatch (e.g. factual write onto a
                # shadow-tagged holder) stays blocked as before.
                logger.info(
                    "[merge\u00b7entity-update] Skipped snapshot on %s "
                    "(holder.world_id=%s, merge_world_id=%s) \u2014 "
                    "cross-branch write blocked.",
                    eu.entity_id,
                    holder_world,
                    world_id,
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
            target_entity.state_timeline.append(snap)
            target_entity.state_timeline.sort(key=lambda s: s.fabula_time)
            changeset.entity_updates_applied += 1

        # --- Object state updates ---
        # Mirror of the entity_updates loop above so Pearl Rung-2/3
        # ``DoNarrativeObject`` surgeries (and Rung-1 OBJ_ observation
        # reveals) routed through ``topology.object_updates`` land as
        # :class:`ObjectStateSnapshot` entries on the canonical
        # :attr:`NarrativeObject.state_timeline`. Without this loop the
        # bridge in ``pipeline._augment_topology_with_sandbox_deltas``
        # would silently drop every object clamp on re-extraction.
        for ou in getattr(topology, "object_updates", []) or []:
            obj = merged.objects.get(ou.object_id)
            if obj is None:
                logger.warning(
                    "Object update for unknown object '%s' \u2014 skipped.",
                    ou.object_id,
                )
                changeset.object_updates_skipped.append(ou.object_id)
                continue
            # Branch safety: mirror of the entity-update guard above.
            # An object's ``state_timeline`` is replayed in fabula
            # order without per-snapshot ``world_id`` filtering by the
            # object reconstruction helpers, so a shadow snapshot
            # written onto a factual-tagged object would pollute every
            # later factual reconstruction.
            holder_world = getattr(obj, "world_id", "factual") or "factual"
            target_obj: Optional["NarrativeObject"] = obj
            if world_id == "shadow" and holder_world == "factual":
                # AMWN node-split for objects: mirror of the entity
                # routing above. Lazy clone in
                # ``merged.shadow_objects[branch_label][ou.object_id]``.
                target_obj = _get_or_clone_shadow_object(
                    merged, ou.object_id,
                    branch_label=branch_label,
                    suppressed_event_ids=_suppressed_set,
                )
                if target_obj is None:
                    logger.info(
                        "[merge\u00b7object-update] Shadow write for %s "
                        "could not be cloned (no branch_label) \u2014 "
                        "cross-branch write blocked.",
                        ou.object_id,
                    )
                    changeset.object_updates_skipped.append(ou.object_id)
                    continue
            elif holder_world != world_id:
                logger.info(
                    "[merge\u00b7object-update] Skipped snapshot on %s "
                    "(holder.world_id=%s, merge_world_id=%s) \u2014 "
                    "cross-branch write blocked.",
                    ou.object_id,
                    holder_world,
                    world_id,
                )
                changeset.object_updates_skipped.append(ou.object_id)
                continue
            snap = ObjectStateSnapshot(
                world_id=world_id,
                fabula_time=ou.fabula_time,
                triggered_by=ou.triggered_by,
                location_id=ou.new_location_id,
                owner_id=ou.new_owner_id,
                set_location_null=bool(ou.set_location_null),
                set_owner_null=bool(ou.set_owner_null),
                properties_set=dict(ou.properties_set or {}),
                properties_unset=list(ou.properties_unset or []),
            )
            target_obj.state_timeline.append(snap)
            target_obj.state_timeline.sort(key=lambda s: s.fabula_time)
            changeset.object_updates_applied += 1

        # --- Affect ledger (P2 of prose-merge completeness) ---
        # Folds new propositions, truth commits, proposition snapshots,
        # new concerns, ConcernSeed materialisations, and concern
        # snapshots into the merged world.
        _apply_affect_to_world(
            merged, topology, world_id=world_id, changeset=changeset,
            branch_label=branch_label,
        )
        # Belief confidence overwrites (Pearl Rung-2 BeliefMutation bridge).
        _apply_belief_confidence_updates(
            merged, topology, changeset=changeset,
            merge_world_id=world_id, branch_label=branch_label,
        )
        # Supersession (mainline-promoted counterfactual override).
        _apply_supersession(
            merged, topology, changeset=changeset,
            merge_world_id=world_id,
        )

        # ── Referential-integrity pass ─────────────────────────────
        # Walk every event in the merged world and verify each
        # actor_id / target_id is resolvable. Anything missing means
        # the renderer named an entity / object / location that was
        # never declared via ``introduced_elements`` and never
        # spawned via ``<ID>.spawn`` — surface it on the changeset
        # so the UI can warn the user. The merge does NOT auto-drop
        # the dangling event because doing so would silently
        # discard prose; the auditor's ``undeclared_element`` rule
        # is the upstream gate.
        _populate_dangling_ref_ledger(merged, changeset)

        # PR 2 (EventNode.at_location_id): validate event spatial
        # anchors and auto-repair co-presence by inserting synthetic
        # EntityStateSnapshots when the primary actor's reconstructed
        # location disagrees with the event's. Runs *after* the
        # additive sections so it can read the merged event list.
        _apply_event_spatial_anchor_repairs(merged, changeset)

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

