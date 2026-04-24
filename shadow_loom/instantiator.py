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
            e_loc = ent.get("location_id")
            if e_loc and sandbox.has_node(e_loc):
                sandbox.add_edge(ent["id"], e_loc, edge_type="located_in")

        for obj in ego_payload.get("present_objects", []):
            obj_id = obj["id"]
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
            actor_id = evt.get("actor_id")
            if evt_id and actor_id and sandbox.has_node(actor_id):
                if (actor_id, evt_id) not in formal_causal_pairs:
                    sandbox.add_edge(actor_id, evt_id, edge_type="causal", mechanism="physical")

        # D. Formal Causal Topology Edges (with real mechanism types)
        for ce in ego_payload.get("relevant_causal_edges", []):
            src = ce.get("source_id")
            tgt = ce.get("target_id")
            mech = ce.get("mechanism", "physical")
            if src and tgt and sandbox.has_node(src) and sandbox.has_node(tgt):
                sandbox.add_edge(src, tgt, edge_type="causal", mechanism=mech,
                                 world_id=target_world_id)

        # E. Spatial Navigation Edges (SpatialEdge, bidirectional unless locked)
        for se in ego_payload.get("relevant_spatial_edges", []):
            src_loc = se.get("source_id")
            tgt_loc = se.get("target_id")
            is_locked = se.get("is_locked", False)
            if src_loc and tgt_loc and not is_locked:
                if sandbox.has_node(src_loc) and sandbox.has_node(tgt_loc):
                    sandbox.add_edge(src_loc, tgt_loc, edge_type="connected_to",
                                     world_id=target_world_id)
                    sandbox.add_edge(tgt_loc, src_loc, edge_type="connected_to",
                                     world_id=target_world_id)

        # F. Information / Communication Edges (InformationEdge)
        for ie in ego_payload.get("relevant_information_edges", []):
            src = ie.get("source_id")
            medium = ie.get("medium", "unknown")
            for tgt in ie.get("target_ids", []):
                if src and tgt and sandbox.has_node(src) and sandbox.has_node(tgt):
                    sandbox.add_edge(src, tgt, edge_type="communicating_with",
                                     medium=medium, world_id=target_world_id)

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
    # SURGERY 1: SPATIAL
    # ==========================================
    @staticmethod
    def _intervene_spatial(sandbox: nx.MultiDiGraph, entity_id: str, new_location_id: str):
        """Forces an entity into a new room, severing old spatial edges."""
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
                sandbox.add_edge(object_id, drop_loc, edge_type="located_in", world_id="shadow")
            logger.info("[Surgery] Dropped %s on the floor.", object_id)

    # ==========================================
    # SURGERY 3: SOCIAL
    # ==========================================
    @staticmethod
    def _intervene_relationship(sandbox: nx.MultiDiGraph, source_id: str, path: str, new_value: float):
        """Forces a relationship metric (Affinity/Friction) to change."""
        parts = path.split('.')
        if len(parts) != 3:
            logger.warning("Malformed relationship path: %s. Expected 'relationships.<target>.<metric>'.", path)
            return
        _, target_id, metric = parts
        
        if not sandbox.has_node(target_id):
            return
            
        for u, v, key, data in sandbox.out_edges(source_id, data=True, keys=True):
            if v == target_id and data.get("edge_type") == "relationship":
                sandbox[u][v][key][metric] = new_value
                logger.info("[Surgery] Forced %s->%s %s to %s", source_id, target_id, metric, new_value)
                return

    # ==========================================
    # SURGERY 4: STATE (Classic do-operator)
    # ==========================================
    @staticmethod
    def _intervene_state(sandbox: nx.MultiDiGraph, node_id: str, path: str, new_value: Any):
        """Forces a physical or psychological property and severs incoming causes."""
        node_data = sandbox.nodes[node_id]
        keys = path.split('.')
        
        current_level = node_data
        for key in keys[:-1]:
            if key not in current_level or not isinstance(current_level[key], dict):
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