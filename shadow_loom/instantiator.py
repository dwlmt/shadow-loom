import logging
import networkx as nx
from typing import Dict, Any

logger = logging.getLogger(__name__)

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
            force = ce.get("causal_force", 5.0)
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
                # mutation_social metadata
                if ctype == "mutation_social":
                    edge_attrs["target_id"] = tgt
                    edge_attrs["rel_counterpart_id"] = ce.get("rel_counterpart_id")
                    edge_attrs["trait_target"] = ce.get("trait_target")
                    edge_attrs["trait_delta"] = ce.get("trait_delta")
                sandbox.add_edge(src, tgt, **edge_attrs)

        # E. Spatial Navigation Edges (SpatialEdge — ALL edges wired, locked flagged)
        for se in ego_payload.get("relevant_spatial_edges", []):
            src_loc = se.get("source_id")
            tgt_loc = se.get("target_id")
            is_locked = se.get("is_locked", False)
            barrier_item_id = se.get("barrier_item_id")
            if src_loc and tgt_loc and sandbox.has_node(src_loc) and sandbox.has_node(tgt_loc):
                sandbox.add_edge(src_loc, tgt_loc, edge_type="connected_to",
                                 is_locked=is_locked, barrier_item_id=barrier_item_id,
                                 world_id=target_world_id)
                sandbox.add_edge(tgt_loc, src_loc, edge_type="connected_to",
                                 is_locked=is_locked, barrier_item_id=barrier_item_id,
                                 world_id=target_world_id)

        # F. Information / Communication Edges (InformationEdge)
        for ie in ego_payload.get("relevant_information_edges", []):
            src = ie.get("source_id")
            medium = ie.get("medium", "unknown")
            is_encrypted = ie.get("is_encrypted", False)
            for tgt in ie.get("target_ids", []):
                if src and tgt and sandbox.has_node(src) and sandbox.has_node(tgt):
                    sandbox.add_edge(src, tgt, edge_type="communicating_with",
                                     medium=medium, is_encrypted=is_encrypted,
                                     world_id=target_world_id)

        # G. Epistemic Leakage (Eavesdropping on unencrypted comms)
        # Any entity co-located with a comms participant can overhear unencrypted channels.
        comms_edges = [
            (u, v, d) for u, v, d in sandbox.edges(data=True)
            if d.get("edge_type") == "communicating_with" and not d.get("is_encrypted", False)
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
            # Sync location_id to the new owner's location
            owner_loc = sandbox.nodes[new_owner_id].get("location_id")
            if owner_loc:
                sandbox.nodes[object_id]["location_id"] = None
            sandbox.add_edge(object_id, new_owner_id, edge_type="owned_by", world_id="shadow")
            logger.info("[Surgery] Gave %s to %s", object_id, new_owner_id)
        else:
            sandbox.nodes[object_id]["owner_id"] = new_owner_id
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
                rel_inertia = data.get("inertia", 0.3)
                desired_shift = float(new_value) - current_val

                if abs(desired_shift) <= rel_inertia:
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
                logger.info("[Surgery] Relationship dampened: %s->%s %s desired=%.2f, inertia=%.2f, effective=%.2f",
                             source_id, target_id, metric, new_value, rel_inertia, effective_val)
                return

        # No existing edge — create a new relationship with default metrics
        if not found:
            edge_attrs = {
                "edge_type": "relationship",
                "affinity": 0.0,
                "fear": 0.0,
                "power_dynamic": 0.0,
                "inertia": 0.3,
                "evidence_strength": "weak",
                "last_updated_fabula": 0,
                "world_id": "shadow",
            }
            # Clamp the target value
            if metric == "fear":
                edge_attrs[metric] = max(0.0, min(1.0, float(new_value)))
            else:
                edge_attrs[metric] = max(-1.0, min(1.0, float(new_value)))
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
                    if abs(desired_shift) <= trait_inertia:
                        logger.info("[Surgery] Inertia blocked: %s.%s shift=%.2f <= inertia=%.2f. No change.",
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
                    logger.info("[Surgery] Inertia dampened: %s.%s desired=%.2f, inertia=%.2f, effective=%.2f",
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
        
        logger.info("[Surgery] Forced State: do(%s.%s = %s)", node_id, path, new_value)

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

        # Sever all existing comms from this source
        edges_to_remove = []
        for u, v, key, data in sandbox.out_edges(source_id, data=True, keys=True):
            if data.get("edge_type") == "communicating_with":
                edges_to_remove.append((u, v, key))
        sandbox.remove_edges_from(edges_to_remove)

        if not target_ids:
            logger.info("[Surgery] Severed all comms from %s", source_id)
            return

        for tgt in target_ids:
            if sandbox.has_node(tgt):
                sandbox.add_edge(source_id, tgt, edge_type="communicating_with",
                                 medium="unknown", world_id="shadow")
        logger.info("[Surgery] Opened comms: %s → %s", source_id, target_ids)