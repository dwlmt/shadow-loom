from typing import Dict, Any, List, Optional
import logging

import networkx as nx

from shadow_loom.models import WorldStateV1
from shadow_loom.query_models import UserRequest
from shadow_loom.extract_graph import EgoGraphPayload, extract_ego_graph_from_memory, extract_full_world_state
from shadow_loom.instantiator import AMWNInstantiator

logger = logging.getLogger(__name__)


def calculate_narrative_physics(
    request: UserRequest,
    global_world_state: WorldStateV1,
    temporal_anchor: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes structural graph math and topological surgeries.
    Returns the serialized graph state purely in Python dictionaries.
    """
    # ==========================================
    # RUNG 1: OBSERVATION
    # ==========================================
    if request.query_type == "observation":
        # If no POV specified, fall back to the Omniscient Graph
        if not request.focus_entity_ids:
            logger.info("[Observation] No POV specified — extracting Omniscient Graph")
            full_state = extract_full_world_state(global_world_state, temporal_anchor)
            return {
                "status": "success",
                "query_type": "observation",
                "physics_state": full_state,
                "directives": request.observations
            }

        logger.info("[Observation] Multi-Ego extraction for POV: %s", request.focus_entity_ids)
        ego_graph = extract_ego_graph_from_memory(global_world_state, request.focus_entity_ids, temporal_anchor)

        return {
            "status": "success",
            "query_type": "observation",
            "physics_state": ego_graph.model_dump(),
            "directives": request.observations
        }

    # ==========================================
    # RUNG 2: INTERVENTION (do-calculus)
    # ==========================================
    elif request.query_type == "intervention":
        # Collect ALL affected entities from the intervention keys
        focus_ids = _resolve_focus_entities(request.interventions, global_world_state)
        logger.info("[Intervention] Resolved focus entities: %s from %d interventions", focus_ids, len(request.interventions))

        # 1. Time-Slice
        ego_graph = extract_ego_graph_from_memory(global_world_state, focus_ids, temporal_anchor)

        # 2. Build Sandbox & Apply Math
        shadow_graph = AMWNInstantiator.create_sandbox(ego_graph.model_dump(), "intervention")
        logger.info("[Intervention] Sandbox built — %d nodes, %d edges. Applying surgeries.",
                     shadow_graph.number_of_nodes(), shadow_graph.number_of_edges())
        AMWNInstantiator.execute_interventions(shadow_graph, request.interventions)
        logger.info("[Intervention] Surgeries complete — %d nodes, %d edges.",
                     shadow_graph.number_of_nodes(), shadow_graph.number_of_edges())

        result = {
            "status": "success",
            "query_type": "intervention",
            "physics_state": nx.node_link_data(shadow_graph),
            "math_changes": request.interventions
        }

        # Semantic Physics Override for split-screen (multi-room) scenarios
        physics_override = _generate_physics_override(shadow_graph)
        if physics_override:
            result["physics_override"] = physics_override

        return result

    # ==========================================
    # RUNG 3: COUNTERFACTUAL
    # ==========================================
    elif request.query_type == "counterfactual":
        # Determine the historical anchor point from the requested interventions
        past_anchor = _calculate_past_anchor(request.historical_interventions, global_world_state)

        focus_ids = _resolve_focus_entities(request.historical_interventions, global_world_state)
        logger.info("[Counterfactual] Point of Divergence: T=%d | Focus: %s", past_anchor, focus_ids)
        ego_graph = extract_ego_graph_from_memory(global_world_state, focus_ids, past_anchor)
        shadow_graph = AMWNInstantiator.create_sandbox(ego_graph.model_dump(), "counterfactual")
        logger.info("[Counterfactual] Historical sandbox built — %d nodes. Applying surgeries.",
                     shadow_graph.number_of_nodes())

        # --- ABDUCTION STEP: Update hidden variables from present evidence ---
        _apply_abduction(shadow_graph, request.evidence_node_ids, global_world_state)

        AMWNInstantiator.execute_interventions(shadow_graph, request.historical_interventions)

        # --- PREDICTION STEP: Forward cascade through causal topology ---
        _apply_forward_cascade(shadow_graph, global_world_state)

        result = {
            "status": "success",
            "query_type": "counterfactual",
            "physics_state": nx.node_link_data(shadow_graph),
            "math_changes": request.historical_interventions,
            "evidence_conditions": request.evidence_node_ids
        }

        physics_override = _generate_physics_override(shadow_graph)
        if physics_override:
            result["physics_override"] = physics_override

        return result

    # ==========================================
    # SEMANTIC: DIRECTIVE
    # ==========================================
    elif request.query_type == "directive":
        logger.info("[Directive] Target entities: %s | Effect: %s | Intensity: %.2f",
                     request.target_entity_ids, request.target_effect, request.intensity)
        ego_graph = extract_ego_graph_from_memory(global_world_state, request.target_entity_ids, temporal_anchor)

        injection_rules = _generate_directive_rules(
            ego_graph.model_dump(),
            request.target_vector_id,
            request.intensity
        )

        return {
            "status": "success",
            "query_type": "directive",
            "physics_state": ego_graph.model_dump(),
            "directives": injection_rules,
            "target_effect": request.target_effect
        }

    # ==========================================
    # GRAPH RAG: INTERROGATION
    # ==========================================
    elif request.query_type == "interrogate":
        logger.info("[Interrogation] Extracting Omniscient Graph for question: %s", request.question[:80])
        full_state = extract_full_world_state(global_world_state, temporal_anchor)

        return {
            "status": "success",
            "query_type": "interrogate",
            "physics_state": full_state,
            "question": request.question,
            "require_proof": request.require_proof
        }

    # Fallback
    raise ValueError(f"Unknown Query Type: {request.query_type}")

# ==========================================
# HELPER 0: RESOLVE ALL FOCUS ENTITIES FROM INTERVENTIONS
# ==========================================
def _resolve_focus_entities(interventions: Dict[str, Any], global_world_state: WorldStateV1) -> List[str]:
    """
    Given intervention keys like 'EVT_DUNCAN_MURDER.event_type' or 'ENT_MACBETH.status',
    collect ALL affected entities so the ego-graph is the union of their rooms.
    Genesis (.spawn) keys are skipped since their nodes don't exist yet.
    """
    seen = set()
    focus_ids = []

    for target_path in interventions:
        if '.' not in target_path:
            continue
        node_id, prop = target_path.split('.', 1)
        if prop == "spawn":
            continue

        resolved = None
        # Direct entity reference
        if node_id in global_world_state.entities:
            resolved = node_id
        else:
            # Event reference — use the event's actor
            event = next((e for e in global_world_state.events if e.id == node_id), None)
            if event and event.actor_id and event.actor_id in global_world_state.entities:
                resolved = event.actor_id
            else:
                # Object reference — use the object's owner
                obj = global_world_state.objects.get(node_id)
                if obj and obj.owner_id and obj.owner_id in global_world_state.entities:
                    resolved = obj.owner_id

        if resolved and resolved not in seen:
            seen.add(resolved)
            focus_ids.append(resolved)

        # Also resolve comms targets so their rooms are in the ego-graph
        if prop == "communicating_with":
            value = interventions[target_path]
            if isinstance(value, list):
                for tgt_id in value:
                    if tgt_id in global_world_state.entities and tgt_id not in seen:
                        seen.add(tgt_id)
                        focus_ids.append(tgt_id)

    # Last resort: first entity in the world state
    if not focus_ids and global_world_state.entities:
        focus_ids.append(next(iter(global_world_state.entities)))

    return focus_ids

# ==========================================
# HELPER 1: THE TIME MACHINE
# ==========================================
def _calculate_past_anchor(interventions: Dict[str, Any], global_world_state: WorldStateV1) -> int:
    """
    Finds the exact 'Point of Divergence' (fabula_time) for a Rung 3 Counterfactual.
    If multiple historical interventions are provided, it finds the oldest one,
    ensuring the simulation rolls back far enough to capture all cascading effects.
    """
    earliest_time = float('inf')
    
    for target_path in interventions.keys():
        # e.g., "EVT_MURDER_1.event_type" -> node_id = "EVT_MURDER_1"
        if '.' not in target_path:
            continue
        node_id, prop = target_path.split('.', 1)

        # Genesis spawns have no historical footprint — skip them
        if prop == "spawn":
            continue
        
        # Scenario A: The intervention explicitly targets a past EventNode
        event = next((evt for evt in global_world_state.events if evt.id == node_id), None)
        if event:
            earliest_time = min(earliest_time, event.fabula_time)
            continue
            
        # Scenario B: The intervention targets a Noun (Entity/Object) in the past
        # e.g., "OBJ_DAGGER.owner_id": "ENT_MACBETH"
        # We must find the last time this object was interacted with to anchor the timeline.
        relevant_events = [
            evt for evt in global_world_state.events 
            if evt.actor_id == node_id or (evt.description and node_id in evt.description)
        ]
        if relevant_events:
            # Find the most recent event involving this noun
            latest_relevant_time = max(evt.fabula_time for evt in relevant_events)
            earliest_time = min(earliest_time, latest_relevant_time)

    if earliest_time == float('inf'):
        raise ValueError(
            f"Temporal Paradox: Could not find a historical anchor for interventions: {interventions}"
        )
        
    return int(earliest_time)

# ==========================================
# HELPER 2: THE SEMANTIC COMPILER
# ==========================================
def _generate_directive_rules(ego_graph: Dict[str, Any], vector_target_id: str, shift: float) -> str:
    """
    Translates cold graph math into a "Creative Brief" for the drafting LLM.
    Calculates the current mathematical state of a trait or relationship and 
    instructs the LLM exactly how hard to push the narrative to achieve the target shift.
    """
    if not vector_target_id:
        # Fallback if no specific vector is targeted, just a general concept
        return f"NARRATIVE DIRECTIVE: Shift the emotional polarity of the scene by {shift:+.2f}."

    # Parse the target: e.g., "ENT_MACBETH.traits.ambition" -> "ENT_MACBETH", "traits.ambition"
    if '.' not in vector_target_id:
        return f"Alter the state of {vector_target_id} by a trajectory of {shift:+.2f}."
    node_id, vector_path = vector_target_id.split('.', 1)
    
    # 1. SOCIAL DIRECTIVE (Relationships)
    if vector_path.startswith("relationships."):
        # e.g., "relationships.ENT_DUNCAN.affinity"
        parts = vector_path.split('.')
        if len(parts) != 3:
            return f"Alter the state of {vector_target_id} by a trajectory of {shift:+.2f}."
        _, target_entity, metric = parts
        
        # Locate the current metric in the localized graph
        current_val = "unknown"
        for rel in ego_graph.get("relevant_relationships", []):
            if rel.get("source_entity_id") == node_id and rel.get("target_entity_id") == target_entity:
                current_val = rel.get(metric, 0.0)
                break
                
        return (
            f"[MATHEMATICAL CONSTRAINT]: The {metric.upper()} between {node_id} and {target_entity} "
            f"currently sits at {current_val}. You MUST write the prose such that this metric is "
            f"forcefully shifted by {shift:+.2f}. Provide clear, physical narrative evidence of this change "
            f"in their dialogue or body language."
        )

    # 2. PSYCHOLOGICAL DIRECTIVE (Traits)
    elif vector_path.startswith("traits."):
        # e.g., "traits.ambition" or "traits.ambition.value"
        trait_parts = vector_path.split('.')
        trait_name = trait_parts[1] if len(trait_parts) >= 2 else vector_path
        
        # Look up the trait on the POV character or present entities
        current_val = "unknown"
        for ent in ego_graph.get("focus_entities", []):
            if ent.get("id") == node_id:
                current_val = ent.get("traits", {}).get(trait_name, {}).get("value", "unknown")
                break
        if current_val == "unknown":
            for ent in ego_graph.get("present_entities", []):
                if ent.get("id") == node_id:
                    current_val = ent.get("traits", {}).get(trait_name, {}).get("value", "unknown")
                    break

        return (
            f"[MATHEMATICAL CONSTRAINT]: {node_id}'s internal '{trait_name}' trait "
            f"currently sits at {current_val}. The events of this scene MUST shatter their inertia "
            f"and shift this trait by {shift:+.2f}. Focus the internal monologue on this psychological pivot."
        )

    # 3. EPISTEMIC DIRECTIVE (Beliefs / Dramatic Irony)
    elif vector_path.startswith("beliefs."):
        return (
            f"[EPISTEMIC CONSTRAINT]: Force a realization. {node_id}'s confidence in their belief "
            f"about {vector_path.split('.')[1]} must shift by {shift:+.2f}. Shatter their current worldview."
        )

    return f"Alter the state of {vector_target_id} by a trajectory of {shift:+.2f}."


# ==========================================
# HELPER 3: THE PHYSICS OVERRIDE DETECTOR
# ==========================================
def _generate_physics_override(sandbox: nx.MultiDiGraph) -> Optional[str]:
    """
    Detects split-screen scenarios (entities in separate rooms communicating remotely).
    Returns a strict [PHYSICS OVERRIDE] string for the LLM, or None if single-room.
    """
    # Collect unique locations that Entity nodes are located_in
    entity_locations: set = set()
    for node_id, data in sandbox.nodes(data=True):
        if data.get("node_type") == "Entity":
            loc = data.get("location_id")
            if loc:
                entity_locations.add(loc)

    if len(entity_locations) <= 1:
        return None

    # Check if any communicating_with edges exist
    has_comms = any(
        d.get("edge_type") == "communicating_with"
        for _, _, d in sandbox.edges(data=True)
    )

    if has_comms:
        return (
            "[PHYSICS OVERRIDE]: Characters are in SEPARATE locations communicating remotely. "
            "You MUST NOT describe physical touching, exchanging of items, or any direct physical interaction. "
            "All interaction must be limited to the communication medium (speech, telepathy, etc.)."
        )

    return None


# ==========================================
# HELPER 4: ABDUCTION (Evidence → Latent Variable Update)
# ==========================================
def _apply_abduction(
    sandbox: nx.MultiDiGraph,
    evidence_node_ids: List[str],
    global_world_state: WorldStateV1,
) -> None:
    """
    Rung 3 Abduction: uses present-day evidence nodes to back-propagate
    latent trait/belief updates into the historical shadow graph.

    For each evidence node that exists in the CURRENT world state, we update
    the sandbox's entity traits and beliefs to reflect what MUST have been
    true at the point of divergence given the observed evidence.
    """
    if not evidence_node_ids:
        return

    evidence_strength_multiplier = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}

    for eid in evidence_node_ids:
        # Case 1: Evidence is an Entity — condition on its current factual state
        if eid in global_world_state.entities and sandbox.has_node(eid):
            factual_entity = global_world_state.entities[eid]
            node_data = sandbox.nodes[eid]
            # Back-propagate traits toward current evidence values (50% blend)
            for trait_name, tv in factual_entity.traits.items():
                sandbox_traits = node_data.get("traits", {})
                if trait_name in sandbox_traits and isinstance(sandbox_traits[trait_name], dict):
                    old_val = sandbox_traits[trait_name].get("value", 0.5)
                    shift = (tv.value - old_val) * 0.5
                    sandbox_traits[trait_name]["value"] = max(0.0, min(1.0, old_val + shift))
            # Back-propagate beliefs
            if "beliefs" not in node_data:
                node_data["beliefs"] = []
            existing_beliefs = node_data["beliefs"]
            for belief in factual_entity.beliefs:
                if not any(b.get("target_id") == belief.target_id and
                          b.get("perceived_state") == belief.perceived_state
                          for b in existing_beliefs):
                    existing_beliefs.append(belief.model_dump())
            logger.info("[Abduction] Conditioned entity %s on present-day evidence.", eid)

        # Case 2: Evidence is an Event — propagate through causal edges
        elif sandbox.has_node(eid):
            node_data = sandbox.nodes[eid]
            if node_data.get("node_type") == "EventNode":
                for ce in global_world_state.causal_topology:
                    if ce.source_event_id == eid:
                        mult = evidence_strength_multiplier.get(ce.evidence_strength, 0.5)
                        target_node = sandbox.nodes.get(ce.target_node_id)
                        if target_node and target_node.get("node_type") == "Entity":
                            traits = target_node.get("traits", {})
                            for trait_data in traits.values():
                                if isinstance(trait_data, dict) and "value" in trait_data:
                                    old_val = trait_data["value"]
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * 0.1))
                logger.info("[Abduction] Propagated evidence from event %s.", eid)
        else:
            logger.warning("[Abduction] Evidence node %s not in sandbox. Skipping.", eid)


# ==========================================
# HELPER 5: FORWARD CASCADE (CTF Step 3 — Prediction)
# ==========================================
def _apply_forward_cascade(
    sandbox: nx.MultiDiGraph,
    global_world_state: WorldStateV1,
) -> None:
    """
    CTF Step 3 — Prediction: After abduction and the do-operator, walk
    the causal topology forward from sandbox events, adjusting downstream
    entity traits proportionally to the edge's evidence_strength.

    The traversal is ordered by fabula_time so that early causes propagate
    before later ones.  When a causal edge targets an event not yet in the
    reachable set, it is added so that multi-hop cascades continue.
    """
    strength_mult = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}

    # Seed the reachable set with every event currently in the sandbox
    reachable_events = {
        nid for nid, d in sandbox.nodes(data=True)
        if d.get("node_type") == "EventNode"
    }

    if not reachable_events:
        return

    # Process causal edges in temporal order for proper forward propagation
    causal_sorted = sorted(
        global_world_state.causal_topology,
        key=lambda ce: ce.fabula_time,
    )

    for ce in causal_sorted:
        if ce.source_event_id not in reachable_events:
            continue

        target_id = ce.target_node_id
        mult = strength_mult.get(ce.evidence_strength, 0.5)

        target = sandbox.nodes.get(target_id)
        if target and target.get("node_type") == "Entity":
            traits = target.get("traits", {})
            for trait_data in traits.values():
                if isinstance(trait_data, dict) and "value" in trait_data:
                    old_val = trait_data["value"]
                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * 0.1))

        # If the target is itself an event, allow further cascading
        reachable_events.add(target_id)

    logger.info("[Forward Cascade] Processed %d reachable events.", len(reachable_events))