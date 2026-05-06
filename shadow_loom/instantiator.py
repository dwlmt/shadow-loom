# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import networkx as nx
from typing import Dict, Any

from shadow_loom.settings import get_settings as _get_settings
from shadow_loom.models import default_relationship_metrics_dict


def _surgery_log_level() -> int:
    """Return the appropriate log level for surgery banner lines.

    Demoted to ``DEBUG`` while a Monte-Carlo sample is running so a
    128-sample sweep doesn't emit 128 \u00d7 N "Forced State" banners
    at INFO. Imported lazily to avoid a circular import with
    ``shadow_loom.causal_physics``.
    """
    try:
        from shadow_loom.causal_physics import is_in_mc_sample
    except ImportError:
        return logging.INFO
    return logging.DEBUG if is_in_mc_sample() else logging.INFO

logger = logging.getLogger(__name__)


def _physics_settings():
    return _get_settings().physics


def _relationship_inertia_default() -> float:
    return _physics_settings().relationship_inertia_default


def _ambient_force_multiplier() -> float:
    return _physics_settings().ambient_force_multiplier


def _default_causal_force() -> float:
    return _physics_settings().default_causal_force


def _inertia_epsilon() -> float:
    return _physics_settings().inertia_epsilon

class AMWNInstantiator:
    """
    Translates an Ego-Graph payload into a live, in-memory NetworkX sandbox.
    Handles the creation of the Shadow Timeline and executes topological do-calculus.
    """
    
    # ==========================================
    # CORE: BUILD THE SANDBOX
    # ==========================================
    @staticmethod
    def create_sandbox(ego_payload: Dict[str, Any], query_type: str) -> nx.MultiDiGraph:
        """
        Builds the NetworkX MultiDiGraph from the Stage 5 payload.
        """
        sandbox = nx.MultiDiGraph()
        
        # Determine the AMWN Multiverse Tag
        is_volatile = query_type in ["intervention", "counterfactual"]
        target_world_id = "shadow" if is_volatile else "factual"

        # --- HELPER: SAFE NODE ADDITION ---
        def add_amwn_node(node_data: Dict[str, Any], node_type: str):
            if not node_data or "id" not in node_data:
                return
                
            node_id = node_data["id"]
            attributes = node_data.copy()
            
            # OVERRIDE: Force the Multiverse Tag to protect the factual timeline
            attributes["world_id"] = target_world_id
            attributes["node_type"] = node_type
            
            sandbox.add_node(node_id, **attributes)

        # 1. BUILD THE NODES (The Vertices)
        location_ids = set()
        for loc_data in ego_payload.get("current_locations", []):
            add_amwn_node(loc_data, "Location")
            if loc_data.get("id"):
                location_ids.add(loc_data["id"])

        focus_entity_ids = set()
        for ent in ego_payload.get("focus_entities", []):
            add_amwn_node(ent, "Entity")
            if ent.get("id"):
                focus_entity_ids.add(ent["id"])

        for ent in ego_payload.get("present_entities", []):
            add_amwn_node(ent, "Entity")
            
        for obj in ego_payload.get("present_objects", []):
            add_amwn_node(obj, "NarrativeObject")

        for evt in ego_payload.get("recent_memory", []):
            add_amwn_node(evt, "EventNode")

        # Utterance events (event_type='utterance') from the ego-payload's
        # ``relevant_utterance_events`` channel. They may overlap with
        # ``recent_memory`` (then ``add_amwn_node`` is a no-op via
        # MultiDiGraph node identity) or be additional context
        # surfaced specifically because they belong to a relevant
        # channel. Either way the sandbox MUST carry their utterance
        # metadata (``via_channel_id``, ``speaker_id``, ``addressee_ids``,
        # ``truth_value``, ``content``) for the auditor's leak detector,
        # the directive assembler, and channel-surgery prune logic.
        for utt in ego_payload.get("relevant_utterance_events", []):
            add_amwn_node(utt, "EventNode")

        # World Trait nodes (always global — no spatial filtering)
        world_trait_ids = set()
        for wt in ego_payload.get("world_traits", []):
            add_amwn_node(wt, "WorldTrait")
            if wt.get("id"):
                world_trait_ids.add(wt["id"])

        # Collect all entity IDs in the sandbox for ambient edge generation
        all_entity_ids = set(focus_entity_ids)
        for ent in ego_payload.get("present_entities", []):
            if ent.get("id"):
                all_entity_ids.add(ent["id"])

        # 2. WEAVE THE TOPOLOGY (The Edges)
        
        # A. Spatial & Inventory Edges
        for focus_ent in ego_payload.get("focus_entities", []):
            f_id = focus_ent.get("id")
            f_loc = focus_ent.get("location_id")
            if f_id and f_loc and sandbox.has_node(f_loc):
                sandbox.add_edge(f_id, f_loc, edge_type="located_in")

        for ent in ego_payload.get("present_entities", []):
            e_id = ent.get("id")
            e_loc = ent.get("location_id")
            if e_id and e_loc and sandbox.has_node(e_loc):
                sandbox.add_edge(e_id, e_loc, edge_type="located_in")

        for obj in ego_payload.get("present_objects", []):
            obj_id = obj.get("id")
            if not obj_id:
                continue
            owner_id = obj.get("owner_id")
            if owner_id and sandbox.has_node(owner_id):
                sandbox.add_edge(obj_id, owner_id, edge_type="owned_by")
            else:
                obj_loc = obj.get("location_id")
                if obj_loc and sandbox.has_node(obj_loc):
                    sandbox.add_edge(obj_id, obj_loc, edge_type="located_in")

        # B. Social/Psychological Edges
        for rel in ego_payload.get("relevant_relationships", []):
            src = rel.get("source_entity_id")
            tgt = rel.get("target_entity_id")
            if src and tgt and sandbox.has_node(src) and sandbox.has_node(tgt):
                edge_attrs = rel.copy()
                edge_attrs["edge_type"] = "relationship"
                edge_attrs["world_id"] = target_world_id
                sandbox.add_edge(src, tgt, **edge_attrs)

        # C. Historical Causal Edges (actor → event fallback)
        # Only add if NOT already covered by the formal causal topology (Section D)
        formal_causal_pairs = set()
        for ce in ego_payload.get("relevant_causal_edges", []):
            src = ce.get("source_id")
            tgt = ce.get("target_id")
            if src and tgt:
                formal_causal_pairs.add((src, tgt))

        for evt in ego_payload.get("recent_memory", []):
            evt_id = evt.get("id")
            actor_ids = evt.get("actor_ids", [])
            for actor_id in actor_ids:
                if evt_id and actor_id and sandbox.has_node(actor_id):
                    if (actor_id, evt_id) not in formal_causal_pairs:
                        sandbox.add_edge(actor_id, evt_id, edge_type="causal", mechanism="physical")

        # D. Formal Causal Topology Edges (with real mechanism types)
        for ce in ego_payload.get("relevant_causal_edges", []):
            src = ce.get("source_id")
            tgt = ce.get("target_id")
            mech = ce.get("mechanism", "physical")
            strength = ce.get("evidence_strength", "moderate")
            force = ce.get("causal_force", _default_causal_force())
            ctype = ce.get("causality_type", "chain_reaction")
            delay = ce.get("propagation_delay", 0)
            ft = ce.get("fabula_time", 0)
            if src and tgt and sandbox.has_node(src) and sandbox.has_node(tgt):
                edge_attrs = dict(
                    edge_type="causal", mechanism=mech,
                    evidence_strength=strength,
                    causal_force=force,
                    causality_type=ctype,
                    propagation_delay=delay,
                    fabula_time=ft,
                    world_id=target_world_id,
                )
                # mutation / mutation_social metadata (trait_target, trait_delta)
                if ctype in ("mutation", "mutation_social"):
                    edge_attrs["trait_target"] = ce.get("trait_target")
                    edge_attrs["trait_delta"] = ce.get("trait_delta")
                if ctype == "mutation_social":
                    edge_attrs["target_id"] = tgt
                    edge_attrs["rel_counterpart_id"] = ce.get("rel_counterpart_id")
                sandbox.add_edge(src, tgt, **edge_attrs)

        # E. Spatial Navigation Edges (SpatialEdge — ALL edges wired, locked flagged)
        # Honors temporal lifecycle: an edge that was destroyed before the
        # ego-graph's frontier (max event fabula_time) is omitted entirely
        # — the passage no longer exists, so reachability must not see it.
        # An edge whose ``established_at_fabula`` is in the future is also
        # skipped: the passage hasn't been built yet at the slice we're
        # simulating. Counterfactual surgery that calls
        # ``spawn`` / ``destroy`` on a SpatialEdge can then take effect.
        # Compute the slice frontier here so spatial wiring can use it.
        _spatial_max_ft = 0
        for _evt in ego_payload.get("recent_memory", []):
            _ft = _evt.get("fabula_time", 0)
            if _ft > _spatial_max_ft:
                _spatial_max_ft = _ft

        for se in ego_payload.get("relevant_spatial_edges", []):
            src_loc = se.get("source_id")
            tgt_loc = se.get("target_id")
            is_locked = se.get("is_locked", False)
            barrier_item_id = se.get("barrier_item_id")
            established_at = se.get("established_at_fabula", 0) or 0
            destroyed_at = se.get("destroyed_at_fabula")
            # Lifecycle gating: skip edges outside the active window.
            if destroyed_at is not None and destroyed_at <= _spatial_max_ft:
                logger.debug(
                    "[Instantiator·Spatial] dropping %s↔%s — destroyed at "
                    "t=%s (frontier=%s)",
                    src_loc, tgt_loc, destroyed_at, _spatial_max_ft,
                )
                continue
            if established_at > _spatial_max_ft:
                logger.debug(
                    "[Instantiator·Spatial] dropping %s↔%s — not yet "
                    "established (t=%s, frontier=%s)",
                    src_loc, tgt_loc, established_at, _spatial_max_ft,
                )
                continue
            if src_loc and tgt_loc and sandbox.has_node(src_loc) and sandbox.has_node(tgt_loc):
                sandbox.add_edge(src_loc, tgt_loc, edge_type="connected_to",
                                 is_locked=is_locked, barrier_item_id=barrier_item_id,
                                 established_at_fabula=established_at,
                                 destroyed_at_fabula=destroyed_at,
                                 world_id=target_world_id)
                sandbox.add_edge(tgt_loc, src_loc, edge_type="connected_to",
                                 is_locked=is_locked, barrier_item_id=barrier_item_id,
                                 established_at_fabula=established_at,
                                 destroyed_at_fabula=destroyed_at,
                                 world_id=target_world_id)

        # F. Channels (standing comms capabilities) and on-page utterances.
        # Channels are nodes in the world model but materialise into the
        # sandbox as per-participant-pair ``communicating_with`` edges so
        # downstream graph queries (eavesdropping, mutation_social) can
        # use the same edge_type as before. ``intelligibility`` is
        # per-recipient: an edge from S→T carries ``intelligibility``
        # equal to the channel's intelligibility for T (default 1.0
        # — fully comprehensible).
        intel_thresh = _physics_settings().intelligibility_threshold
        for ch in ego_payload.get("relevant_channels", []):
            medium = ch.get("medium", "unknown")
            participants = ch.get("participant_ids", [])
            intelligibility = ch.get("intelligibility", {}) or {}
            directionality = ch.get("directionality", "duplex")
            channel_id = ch.get("id")
            for src in participants:
                if not sandbox.has_node(src):
                    continue
                for tgt in participants:
                    if src == tgt or not sandbox.has_node(tgt):
                        continue
                    # ``simplex`` channels carry information one way only;
                    # the convention is that participant_ids[0] is the
                    # sender. ``broadcast`` and ``duplex`` carry both
                    # directions.
                    if directionality == "simplex" and src != participants[0]:
                        continue
                    intel = float(intelligibility.get(tgt, 1.0))
                    sandbox.add_edge(
                        src, tgt,
                        edge_type="communicating_with",
                        medium=medium,
                        intelligibility=intel,
                        channel_id=channel_id,
                        world_id=target_world_id,
                    )

        # G. Epistemic Leakage (eavesdropping on intelligible channels).
        # An edge is eavesdroppable when its per-recipient intelligibility
        # meets the configured threshold
        # (``physics.intelligibility_threshold`` — default 0.3 — kept in
        # sync with ``directive_assembly`` and ``causal_physics`` so
        # hidden-channel detection sees the same edge set the runtime
        # engines reason over).
        comms_edges = [
            (u, v, d) for u, v, d in sandbox.edges(data=True)
            if d.get("edge_type") == "communicating_with"
            and float(d.get("intelligibility", 1.0)) >= intel_thresh
        ]
        for src, tgt, cdata in comms_edges:
            src_loc = sandbox.nodes.get(src, {}).get("location_id")
            tgt_loc = sandbox.nodes.get(tgt, {}).get("location_id")
            eavesdrop_locs = {loc for loc in (src_loc, tgt_loc) if loc}
            for node_id, node_data in sandbox.nodes(data=True):
                if node_data.get("node_type") != "Entity":
                    continue
                if node_id in (src, tgt):
                    continue
                if node_data.get("location_id") in eavesdrop_locs:
                    logger.debug("[Instantiator·Eavesdrop] %s can overhear %s→%s (medium=%s, location=%s)",
                                 node_id, src, tgt, cdata.get("medium"), node_data.get("location_id"))
                    sandbox.add_edge(src, node_id, edge_type="eavesdropped_by",
                                     medium=cdata.get("medium", "unknown"),
                                     world_id=target_world_id)

        # H. Auto-generated Ambient Propagation Edges (WORLD_ → Entity)
        # Weak baseline pressure from world-level facts to all entities in the scene.
        # These are runtime-only (not persisted in causal_topology).
        max_ft = _spatial_max_ft

        for wt_id in world_trait_ids:
            wt_node = sandbox.nodes.get(wt_id, {})
            mag = wt_node.get("magnitude", {})
            mag_value = mag.get("value", 0.5) if isinstance(mag, dict) else 0.5
            domains = wt_node.get("affected_domains", [])
            mechanism = domains[0] if domains else "psychological"
            ambient_force = mag_value * _ambient_force_multiplier()

            for ent_id in all_entity_ids:
                if sandbox.has_node(ent_id):
                    sandbox.add_edge(
                        wt_id, ent_id,
                        edge_type="causal",
                        causality_type="ambient_propagation",
                        causal_force=ambient_force,
                        mechanism=mechanism,
                        evidence_strength="moderate",
                        propagation_delay=0,
                        fabula_time=max_ft,
                        world_id=target_world_id,
                    )

        return sandbox

    # ==========================================
    # CORE: THE MASTER INTERVENTION ROUTER
    # ==========================================
    @classmethod
    def execute_interventions(cls, sandbox: nx.MultiDiGraph, interventions: Dict[str, Any]):
        """
        Parses the syntax of the intervention dictionary and routes it to 
        the correct topological surgery method.
        """
        for target_path, new_value in interventions.items():
            if '.' not in target_path:
                logger.warning("Malformed intervention key (no dot): %s. Skipping.", target_path)
                continue
            node_id, property_path = target_path.split('.', 1)

            # --- THE GENESIS CATCH ---
            if property_path == "spawn":
                cls._intervene_genesis(sandbox, node_id, new_value)
                continue

            # --- THE FAIL-SAFE ---
            if not sandbox.has_node(node_id):
                logger.warning("Node %s not in Ego-Graph. Skipping.", node_id)
                continue

            if property_path == "location_id":
                cls._intervene_spatial(sandbox, node_id, new_value)
            elif property_path == "owner_id":
                cls._intervene_inventory(sandbox, node_id, new_value)
            elif property_path.startswith("relationships."):
                cls._intervene_relationship(sandbox, node_id, property_path, new_value)
            elif property_path == "communicating_with":
                cls._intervene_comms(sandbox, node_id, new_value)
            else:
                cls._intervene_state(sandbox, node_id, property_path, new_value)

    # ==========================================
    # SPATIAL AFFORDANCE CHECK
    # ==========================================
    @staticmethod
    def _check_spatial_path(sandbox: nx.MultiDiGraph, entity_id: str,
                            old_loc: str, new_loc: str) -> bool:
        """Runs nx.has_path on a traversability sub-graph.
        An edge is traversable if it is unlocked, or if the entity holds
        an object whose affordances include 'unlock' targeting the barrier."""
        traversable = nx.DiGraph()
        for u, v, d in sandbox.edges(data=True):
            if d.get("edge_type") != "connected_to":
                continue
            if not d.get("is_locked", False):
                traversable.add_edge(u, v)
            else:
                barrier_id = d.get("barrier_item_id")
                if barrier_id and AMWNInstantiator._has_unlock_affordance(sandbox, entity_id, barrier_id):
                    traversable.add_edge(u, v)
        if not traversable.has_node(old_loc) or not traversable.has_node(new_loc):
            return False
        return nx.has_path(traversable, old_loc, new_loc)

    @staticmethod
    def _has_unlock_affordance(sandbox: nx.MultiDiGraph, actor_id: str, barrier_id: str) -> bool:
        """Check if the actor owns any object that can 'unlock' the barrier."""
        barrier_node = sandbox.nodes.get(barrier_id, {})
        barrier_node_type = barrier_node.get("node_type", "NarrativeObject")
        barrier_name = barrier_node.get("name", "")
        for node_id, data in sandbox.nodes(data=True):
            if data.get("node_type") != "NarrativeObject" or data.get("owner_id") != actor_id:
                continue
            for aff in data.get("affordances", []):
                if not isinstance(aff, dict) or aff.get("action") != "unlock":
                    continue
                aff_target = aff.get("target_type", "")
                if aff_target == barrier_node_type or aff_target == barrier_name:
                    logger.debug("[Instantiator·Unlock] %s owns item %s with affordance matching barrier %s (target_type=%s, barrier_type=%s, barrier_name=%s)",
                                 actor_id, node_id, barrier_id, aff_target, barrier_node_type, barrier_name)
                    return True
        return False

    # ==========================================
    # SURGERY 1: SPATIAL
    # ==========================================
    @staticmethod
    def _intervene_spatial(sandbox: nx.MultiDiGraph, entity_id: str, new_location_id: str):
        """Forces an entity into a new room, severing old spatial edges.
        Validates the path using nx.has_path on traversable edges and checks
        locked-barrier affordances before allowing the move."""
        old_location_id = sandbox.nodes[entity_id].get("location_id")

        # Spatial affordance gate: verify a valid path exists
        if old_location_id and old_location_id != new_location_id and sandbox.has_node(new_location_id):
            if not AMWNInstantiator._check_spatial_path(sandbox, entity_id, old_location_id, new_location_id):
                logger.warning("[Surgery] Spatial affordance BLOCKED: no traversable path from %s to %s for %s.",
                               old_location_id, new_location_id, entity_id)
                return

        edges_to_remove = []
        for u, v, key, data in sandbox.out_edges(entity_id, data=True, keys=True):
            if data.get("edge_type") == "located_in":
                edges_to_remove.append((u, v, key))
        sandbox.remove_edges_from(edges_to_remove)
        
        if sandbox.has_node(new_location_id):
            sandbox.nodes[entity_id]["location_id"] = new_location_id
            sandbox.add_edge(entity_id, new_location_id, edge_type="located_in", world_id="shadow")
        else:
            sandbox.nodes[entity_id]["location_id"] = new_location_id
            logger.warning("[Surgery] Target location %s not in sandbox; attribute set but no edge wired.", new_location_id)
            
        logger.info("[Surgery] Teleported %s to %s", entity_id, new_location_id)

    # ==========================================
    # SURGERY 2: INVENTORY
    # ==========================================
    @staticmethod
    def _intervene_inventory(sandbox: nx.MultiDiGraph, object_id: str, new_owner_id: str | None):
        """Forces an item to be picked up or dropped."""
        # Capture old owner BEFORE overwriting
        old_owner_id = sandbox.nodes[object_id].get("owner_id")
        
        edges_to_remove = []
        for u, v, key, data in sandbox.out_edges(object_id, data=True, keys=True):
            if data.get("edge_type") in ["owned_by", "located_in"]:
                edges_to_remove.append((u, v, key))
        sandbox.remove_edges_from(edges_to_remove)
        
        if new_owner_id and sandbox.has_node(new_owner_id):
            sandbox.nodes[object_id]["owner_id"] = new_owner_id
            # Owned items follow their owner — clear stale location
            sandbox.nodes[object_id]["location_id"] = None
            sandbox.add_edge(object_id, new_owner_id, edge_type="owned_by", world_id="shadow")
            logger.info("[Surgery] Gave %s to %s", object_id, new_owner_id)
        else:
            if new_owner_id and not sandbox.has_node(new_owner_id):
                logger.warning(
                    "[Surgery] Owner %s not in sandbox — treating as drop.",
                    new_owner_id,
                )
            sandbox.nodes[object_id]["owner_id"] = None
            # Resolve drop location: prefer the previous owner's current room
            drop_loc = None
            if old_owner_id and sandbox.has_node(old_owner_id):
                drop_loc = sandbox.nodes[old_owner_id].get("location_id")
            if not drop_loc:
                drop_loc = sandbox.nodes[object_id].get("location_id")
            if drop_loc and sandbox.has_node(drop_loc):
                sandbox.nodes[object_id]["location_id"] = drop_loc
                sandbox.add_edge(object_id, drop_loc, edge_type="located_in", world_id="shadow")
            logger.info("[Surgery] Dropped %s on the floor.", object_id)

    # ==========================================
    # SURGERY 3: SOCIAL (Impact > Inertia for relationships)
    # ==========================================
    @staticmethod
    def _intervene_relationship(sandbox: nx.MultiDiGraph, source_id: str, path: str, new_value: float):
        """Forces a relationship metric (Affinity/Fear/Power) to change.
        Applies Impact > Inertia: if |desired_shift| <= edge inertia, the
        relationship resists entirely; otherwise the shift is dampened.
        If no relationship edge exists, one is created with default metrics."""
        parts = path.split('.')
        if len(parts) != 3:
            logger.warning("Malformed relationship path: %s. Expected 'relationships.<target>.<metric>'.", path)
            return
        _, target_id, metric = parts
        
        if not sandbox.has_node(target_id):
            logger.warning("[Surgery] Relationship target %s not in sandbox. Skipping.", target_id)
            return
            
        # Find existing relationship edge
        found = False
        for u, v, key, data in sandbox.out_edges(source_id, data=True, keys=True):
            if v == target_id and data.get("edge_type") == "relationship":
                found = True
                current_val = data.get(metric, 0.0)
                if not isinstance(current_val, (int, float)):
                    current_val = 0.0
                # Prefer per-axis inertia from the new ``metrics`` dict
                # (carried through by RelationshipEdge.to_legacy_dict)
                # so surgery on ``fear`` honours fear's volatility band
                # rather than the edge-level min-aggregate.
                per_metric = (data.get("metrics") or {}).get(metric) or {}
                rel_inertia = per_metric.get(
                    "inertia",
                    data.get("inertia", _relationship_inertia_default()),
                )
                desired_shift = float(new_value) - current_val

                if abs(desired_shift) <= rel_inertia + _inertia_epsilon():
                    logger.info("[Surgery] Relationship inertia blocked: %s->%s %s shift=%.2f <= inertia=%.2f. No change.",
                                 source_id, target_id, metric, abs(desired_shift), rel_inertia)
                    return

                sign = 1 if desired_shift > 0 else -1
                effective_shift = desired_shift - sign * rel_inertia
                effective_val = current_val + effective_shift
                # Clamp: affinity/power_dynamic in [-1,1], fear in [0,1]
                if metric == "fear":
                    effective_val = max(0.0, min(1.0, effective_val))
                else:
                    effective_val = max(-1.0, min(1.0, effective_val))

                sandbox[u][v][key][metric] = effective_val
                # Mirror into the per-axis ``metrics`` dict so reads
                # via the new shape stay consistent post-surgery.
                if isinstance(data.get("metrics"), dict):
                    # Bump fabula timestamp past the most recent recorded
                    # axis mutation so per-axis time-slicing and the
                    # counterfactual-rollback machinery treat the surgical
                    # value as the latest observation. Without this bump
                    # ``time_slice_world_state`` and
                    # ``reconstruct_relationship_with_causal`` continue to
                    # serve the pre-surgery value at any t >= existing max.
                    existing_ts = [
                        m.get("last_updated_fabula", 0)
                        for m in data["metrics"].values()
                        if isinstance(m, dict)
                    ]
                    next_ft = (max(existing_ts) + 1) if existing_ts else 1
                    axis_state = data["metrics"].setdefault(metric, {
                        "value": current_val,
                        "inertia": rel_inertia,
                        "evidence_strength": "strong",
                        "last_updated_fabula": next_ft,
                        "observed": True,
                    })
                    axis_state["value"] = effective_val
                    axis_state["last_updated_fabula"] = next_ft
                    # Surgery is a deliberate intervention; mark the axis
                    # observed so downstream consumers stop treating the
                    # value as an unobserved default.
                    axis_state["observed"] = True
                logger.info("[Surgery] Relationship dampened: %s->%s %s desired=%.2f, inertia=%.2f, effective=%.2f",
                             source_id, target_id, metric, new_value, rel_inertia, effective_val)
                return

        # No existing edge — create a new relationship with full per-axis metrics
        if not found:
            if metric == "fear":
                primary_value = max(0.0, min(1.0, float(new_value)))
            else:
                primary_value = max(-1.0, min(1.0, float(new_value)))
            edge_attrs = {
                "edge_type": "relationship",
                "affinity": 0.0,
                "fear": 0.0,
                "power_dynamic": 0.0,
                "inertia": _relationship_inertia_default(),
                "evidence_strength": "weak",
                "last_updated_fabula": 0,
                "world_id": "shadow",
                "metrics": default_relationship_metrics_dict(
                    primary_metric=metric,
                    primary_value=primary_value,
                    fabula_time=0,
                    evidence_strength="weak",
                    inertia=_relationship_inertia_default(),
                ),
            }
            edge_attrs[metric] = primary_value
            sandbox.add_edge(source_id, target_id, **edge_attrs)
            logger.info("[Surgery] Created new relationship edge: %s->%s %s=%.2f",
                         source_id, target_id, metric, edge_attrs[metric])

    # ==========================================
    # SURGERY 4: STATE (Classic do-operator)
    # ==========================================
    @staticmethod
    def _intervene_state(sandbox: nx.MultiDiGraph, node_id: str, path: str, new_value: Any):
        """Forces a physical or psychological property and severs incoming causes.
        
        If the path targets a trait value (e.g., 'traits.ambition.value' or 'traits.ambition'),
        the Impact > Inertia check is applied: the effective shift is dampened by the trait's
        inertia. If |desired_shift| <= inertia, the trait resists entirely.
        """
        node_data = sandbox.nodes[node_id]
        keys = path.split('.')

        # --- IMPACT > INERTIA CHECK for trait mutations ---
        if keys[0] == "traits" and len(keys) >= 2:
            trait_name = keys[1]
            trait_data = node_data.get("traits", {}).get(trait_name)
            if isinstance(trait_data, dict) and "value" in trait_data and "inertia" in trait_data:
                current_val = trait_data["value"]
                trait_inertia = trait_data["inertia"]
                # Determine the target value
                if len(keys) == 2 and isinstance(new_value, (int, float)):
                    target_val = float(new_value)
                elif len(keys) == 3 and keys[2] == "value" and isinstance(new_value, (int, float)):
                    target_val = float(new_value)
                else:
                    target_val = None
                if target_val is not None:
                    desired_shift = target_val - current_val
                    if abs(desired_shift) <= trait_inertia + _inertia_epsilon():
                        logger.log(_surgery_log_level(),
                                   "[Surgery] Inertia blocked: %s.%s shift=%.2f <= inertia=%.2f. No change.",
                                   node_id, path, abs(desired_shift), trait_inertia)
                        return  # Trait resists — do NOT sever edges
                    # Dampen: effective shift = desired_shift - sign(shift)*inertia
                    sign = 1 if desired_shift > 0 else -1
                    effective_shift = desired_shift - sign * trait_inertia
                    effective_val = max(0.0, min(1.0, current_val + effective_shift))
                    new_value = effective_val
                    if len(keys) == 2:
                        # Promote shorthand "traits.X" → "traits.X.value" so the
                        # standard mutation updates the value inside the dict
                        # instead of replacing the entire TraitVector.
                        keys = [keys[0], keys[1], "value"]
                    logger.log(_surgery_log_level(),
                               "[Surgery] Inertia dampened: %s.%s desired=%.2f, inertia=%.2f, effective=%.2f",
                               node_id, path, target_val, trait_inertia, effective_val)

        # --- STANDARD STATE MUTATION ---
        current_level = node_data
        for key in keys[:-1]:
            if key not in current_level or not isinstance(current_level[key], dict):
                logger.warning("[Surgery] Path '%s' creates intermediate key '%s' on %s. "
                               "Verify this is intentional.", path, key, node_id)
                current_level[key] = {}
            current_level = current_level[key]
        current_level[keys[-1]] = new_value

        edges_to_remove = []
        for u, v, key, data in sandbox.in_edges(node_id, data=True, keys=True):
            if data.get("edge_type") == "causal":
                edges_to_remove.append((u, v, key))
        sandbox.remove_edges_from(edges_to_remove)
        
        logger.log(_surgery_log_level(),
                   "[Surgery] Forced State: do(%s.%s = %s)",
                   node_id, path, new_value)

    # ==========================================
    # SURGERY 5: GENESIS (Create from nothing)
    # ==========================================
    @staticmethod
    def _intervene_genesis(sandbox: nx.MultiDiGraph, new_node_id: str, payload: Dict[str, Any]):
        """Spawns a completely new node into the shadow timeline.
        Payload must define the 'node_type' and initial 'location_id'."""
        node_type = payload.get("node_type", "Entity")
        location_id = payload.get("location_id")

        # 1. Inject the node with the Shadow Multiverse tag
        attributes = payload.copy()
        attributes["id"] = new_node_id
        attributes["world_id"] = "shadow"
        attributes["node_type"] = node_type
        sandbox.add_node(new_node_id, **attributes)

        # 2. Wire it into physical reality immediately
        if location_id and sandbox.has_node(location_id):
            sandbox.add_edge(new_node_id, location_id, edge_type="located_in", world_id="shadow")

        logger.info("[Surgery] Genesis Event: Spawned %s into %s", new_node_id, location_id)

    # ==========================================
    # SURGERY 6: COMMS (Establish / Sever communication)
    # ==========================================
    @staticmethod
    def _intervene_comms(sandbox: nx.MultiDiGraph, source_id: str, target_ids: list | None):
        """Spawns or severs a communicating_with edge between entities."""
        # Coerce a bare string to a list
        if isinstance(target_ids, str):
            target_ids = [target_ids]

        # Sever all existing comms from this source. Capture which
        # channel_ids are being torn down so we can prune any
        # downstream beliefs whose provenance pointed at those edges.
        edges_to_remove = []
        severed_channel_ids: set[str] = set()
        severed_pairs: set[tuple[str, str]] = set()
        for u, v, key, data in sandbox.out_edges(source_id, data=True, keys=True):
            if data.get("edge_type") == "communicating_with":
                edges_to_remove.append((u, v, key))
                cid = data.get("channel_id")
                if cid:
                    severed_channel_ids.add(cid)
                severed_pairs.add((u, v))
        sandbox.remove_edges_from(edges_to_remove)

        # When a channel is torn down at one endpoint, the *standing
        # capability* itself is gone — every other ``communicating_with``
        # edge anywhere in the sandbox that carries the same
        # ``channel_id`` must also disappear, otherwise downstream
        # propagation (and the legacy cascades that read the global
        # causal_topology) would still treat the channel as live.
        if severed_channel_ids:
            extra_remove = [
                (u, v, key)
                for u, v, key, data in sandbox.edges(keys=True, data=True)
                if data.get("edge_type") == "communicating_with"
                and data.get("channel_id") in severed_channel_ids
            ]
            if extra_remove:
                sandbox.remove_edges_from(extra_remove)

            # Mark every utterance node whose ``via_channel_id`` rode on
            # a severed channel as ``pruned`` so the cascades skip
            # propagating its causal/social effects. The node itself
            # stays in the graph (the auditor's leak detector still
            # needs to see what *was* uttered) but its propagation
            # bridge is closed.
            pruned_utts: list[str] = []
            for nid, ndata in sandbox.nodes(data=True):
                if ndata.get("event_type") != "utterance":
                    continue
                via = ndata.get("via_channel_id")
                if via and via in severed_channel_ids:
                    ndata["pruned"] = True
                    pruned_utts.append(nid)
            if pruned_utts:
                logger.info(
                    "[Surgery] Marked %d utterance(s) pruned via severed "
                    "channel(s) %s: %s",
                    len(pruned_utts), sorted(severed_channel_ids), pruned_utts,
                )

        if not target_ids:
            logger.info("[Surgery] Severed all comms from %s", source_id)
            AMWNInstantiator._prune_beliefs_by_provenance(
                sandbox,
                removed_channel_ids=severed_channel_ids,
                severed_speaker_addressee_pairs=severed_pairs,
            )
            return

        for tgt in target_ids:
            if sandbox.has_node(tgt):
                sandbox.add_edge(source_id, tgt, edge_type="communicating_with",
                                 medium="unknown", world_id="shadow")
        logger.info("[Surgery] Opened comms: %s → %s", source_id, target_ids)
        # Beliefs that were acquired through channels we just severed
        # should be pruned regardless of whether new pairs were added.
        AMWNInstantiator._prune_beliefs_by_provenance(
            sandbox,
            removed_channel_ids=severed_channel_ids,
            severed_speaker_addressee_pairs=severed_pairs - {
                (source_id, t) for t in target_ids
            },
        )

    # ==========================================
    # PROVENANCE PRUNE HELPER
    # ==========================================
    @staticmethod
    def _prune_beliefs_by_provenance(
        sandbox: nx.MultiDiGraph,
        *,
        removed_event_ids: set[str] | None = None,
        removed_channel_ids: set[str] | None = None,
        severed_speaker_addressee_pairs: set[tuple[str, str]] | None = None,
    ) -> int:
        """Drop beliefs whose ``acquired_via_*`` provenance has been
        invalidated by graph surgery.

        A belief is removed when ANY of the following matches:
          * ``acquired_via_event_id`` is in ``removed_event_ids``;
          * ``acquired_via_channel_id`` is in ``removed_channel_ids``;
          * the belief's holder is the addressee of a (speaker, addressee)
            pair whose ``communicating_with`` edge has just been severed,
            and the belief's ``acquired_via_event_id`` resolves to an
            utterance whose ``speaker_id`` matches the severed sender.

        Mirrored on entity-node ``beliefs`` and on every snapshot in
        ``state_timeline[*].beliefs_added``. Returns the number of
        beliefs pruned.
        """
        removed_event_ids = set(removed_event_ids or ())
        removed_channel_ids = set(removed_channel_ids or ())
        severed_pairs = set(severed_speaker_addressee_pairs or ())

        if not (removed_event_ids or removed_channel_ids or severed_pairs):
            return 0

        # Build a quick lookup of utterance speakers from the sandbox
        # so we can resolve the third match condition without a full
        # WorldState pass.
        utterance_speaker: dict[str, str] = {}
        for n, ndata in sandbox.nodes(data=True):
            if ndata.get("event_type") == "utterance":
                sp = ndata.get("speaker_id")
                if sp:
                    utterance_speaker[n] = sp

        def _is_dangling(belief: dict, holder_id: str) -> bool:
            ev = belief.get("acquired_via_event_id")
            ch = belief.get("acquired_via_channel_id")
            if ev and ev in removed_event_ids:
                return True
            if ch and ch in removed_channel_ids:
                return True
            if ev and ev in utterance_speaker:
                speaker = utterance_speaker[ev]
                if (speaker, holder_id) in severed_pairs:
                    return True
            return False

        pruned = 0
        for n, ndata in sandbox.nodes(data=True):
            if ndata.get("node_type") != "Entity":
                continue
            beliefs = ndata.get("beliefs")
            if isinstance(beliefs, list):
                kept = []
                for b in beliefs:
                    if isinstance(b, dict) and _is_dangling(b, n):
                        pruned += 1
                        continue
                    kept.append(b)
                ndata["beliefs"] = kept
            timeline = ndata.get("state_timeline")
            if isinstance(timeline, list):
                for snap in timeline:
                    if not isinstance(snap, dict):
                        continue
                    added = snap.get("beliefs_added")
                    if not isinstance(added, list):
                        continue
                    kept_snap = []
                    for b in added:
                        if isinstance(b, dict) and _is_dangling(b, n):
                            pruned += 1
                            continue
                        kept_snap.append(b)
                    snap["beliefs_added"] = kept_snap
        if pruned:
            logger.info(
                "[Surgery·ProvenancePrune] Removed %d belief(s) whose "
                "provenance was invalidated (events=%d, channels=%d, "
                "severed_pairs=%d).",
                pruned, len(removed_event_ids), len(removed_channel_ids),
                len(severed_pairs),
            )
        return pruned