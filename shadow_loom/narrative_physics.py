from typing import Dict, Any, List, Optional
import logging

import networkx as nx

from shadow_loom.models import WorldStateV1, reconstruct_entity_at
from shadow_loom.query_models import UserRequest
from shadow_loom.extract_graph import EgoGraphPayload, extract_ego_graph_from_memory, extract_full_world_state
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.directive_assembly import DirectiveAssembler

logger = logging.getLogger(__name__)


def calculate_narrative_physics(
    request: UserRequest,
    global_world_state: WorldStateV1,
    temporal_anchor: Optional[int] = None,
    syuzhet_anchor: Optional[int] = None,
    use_causal_engine: bool = False,
) -> Dict[str, Any]:
    """
    Executes structural graph math and topological surgeries.
    Returns the serialized graph state purely in Python dictionaries.

    Parameters
    ----------
    temporal_anchor : int or None
        Fabula-time cutoff for time-slicing (causal physics layer).
    syuzhet_anchor : int or None
        Narrative-position cutoff for reader epistemic state
        (suspense / surprise layer).  Only used when
        ``use_causal_engine=True`` for directive queries.
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

        if use_causal_engine:
            engine = CausalPhysicsEngine(shadow_graph, global_world_state)
            physics_result = engine.execute(rung=2, interventions=request.interventions)
            result = {
                "status": "success",
                "query_type": "intervention",
                "physics_state": physics_result.sandbox_data,
                "math_changes": request.interventions,
                "mutations": [m.model_dump() for m in physics_result.mutations],
                "social_mutations": [m.model_dump() for m in physics_result.social_mutations],
                "blocked": [b.model_dump() for b in physics_result.blocked],
                "intervened_nodes": physics_result.intervened_nodes,
            }
        else:
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

        if use_causal_engine:
            engine = CausalPhysicsEngine(shadow_graph, global_world_state)
            physics_result = engine.execute(
                rung=3,
                interventions=request.historical_interventions,
                evidence_node_ids=request.evidence_node_ids,
            )
            result = {
                "status": "success",
                "query_type": "counterfactual",
                "physics_state": physics_result.sandbox_data,
                "math_changes": request.historical_interventions,
                "evidence_conditions": request.evidence_node_ids,
                "mutations": [m.model_dump() for m in physics_result.mutations],
                "social_mutations": [m.model_dump() for m in physics_result.social_mutations],
                "blocked": [b.model_dump() for b in physics_result.blocked],
                "hidden_deltas": physics_result.hidden_deltas,
            }
        else:
            # --- ABDUCTION STEP: Update hidden variables from present evidence ---
            _apply_abduction(shadow_graph, request.evidence_node_ids, global_world_state)

            AMWNInstantiator.execute_interventions(shadow_graph, request.historical_interventions)

            # --- PREDICTION STEP: Forward cascade through causal topology ---
            _apply_forward_cascade(shadow_graph, global_world_state)

            # --- SOCIAL PREDICTION: Propagate mutation_social edges ---
            _apply_social_cascade(shadow_graph, global_world_state)

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
        ego_dump = ego_graph.model_dump()

        if use_causal_engine:
            assembler = DirectiveAssembler(
                sandbox=None, ego_payload=ego_dump, world_state=global_world_state,
            )
            brief = assembler.assemble(request, syuzhet_anchor=syuzhet_anchor)
            return {
                "status": "success",
                "query_type": "directive",
                "physics_state": ego_dump,
                "creative_brief": brief.model_dump(),
                "target_effect": request.target_effect,
            }

        injection_rules = _generate_directive_rules(
            ego_dump,
            request.target_vector_id,
            request.intensity
        )

        return {
            "status": "success",
            "query_type": "directive",
            "physics_state": ego_dump,
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

    # ==========================================
    # FULL-GRAPH Q&A: GENERAL QUESTION
    # ==========================================
    elif request.query_type == "general":
        logger.info("[General] Full-graph Q&A for question: %s", request.question[:80])
        full_state = extract_full_world_state(global_world_state, temporal_anchor)

        return {
            "status": "success",
            "query_type": "general",
            "physics_state": full_state,
            "question": request.question,
            "include_topology": request.include_topology,
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
            if event and event.actor_ids:
                for _aid in event.actor_ids:
                    if _aid in global_world_state.entities:
                        resolved = _aid
                        break
            else:
                # Object reference — use the object's owner
                obj = global_world_state.objects.get(node_id)
                if obj and obj.owner_id and obj.owner_id in global_world_state.entities:
                    resolved = obj.owner_id

        if resolved and resolved not in seen:
            seen.add(resolved)
            focus_ids.append(resolved)
            logger.debug("[ResolveFocus] %s → resolved entity %s", target_path, resolved)

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
            if node_id in evt.actor_ids or node_id in evt.target_ids or (evt.description and node_id in evt.description)
        ]
        if relevant_events:
            # Find the most recent event involving this noun
            latest_relevant_time = max(evt.fabula_time for evt in relevant_events)
            earliest_time = min(earliest_time, latest_relevant_time)

    if earliest_time == float('inf'):
        raise ValueError(
            f"Temporal Paradox: Could not find a historical anchor for interventions: {interventions}"
        )

    logger.debug("[PastAnchor] earliest_time=%d from %d intervention keys", earliest_time, len(interventions))
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

# Mechanism → trait affinity mapping.  When an event's causal mechanism
# is known, only traits in the corresponding list are updated.  Traits
# not in any list receive a reduced fallback impulse.
_MECHANISM_TRAIT_MAP: Dict[str, List[str]] = {
    "physical": ["courage", "fear", "anger", "pain", "strength"],
    "physical_force": ["courage", "fear", "anger", "pain", "strength"],
    "psychological": ["guilt", "paranoia", "despair", "hope", "anxiety", "fear", "grief", "remorse"],
    "epistemic_revelation": ["suspicion", "curiosity", "paranoia", "guilt"],
    "epistemic": ["suspicion", "curiosity", "paranoia", "guilt"],
    "social_coercion": ["ambition", "fear", "rebelliousness", "loyalty", "obedience"],
    "social": ["ambition", "fear", "rebelliousness", "loyalty", "obedience"],
    "emotional": ["love", "affection", "grief", "despair", "hope", "anger", "fear"],
    "informational": ["suspicion", "curiosity", "paranoia"],
    "betrayal": ["anger", "grief", "fear", "loyalty", "affinity"],
}

# Fallback multiplier for traits not matching the mechanism.
_MECHANISM_FALLBACK_FACTOR = 0.2


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

            # Determine the temporal horizon of the sandbox for reconstruction
            max_ft = max(
                (d.get("fabula_time", 0) for _, d in sandbox.nodes(data=True) if d.get("fabula_time")),
                default=None,
            )

            # Reconstruct entity state at the sandbox's temporal horizon
            if max_ft is not None and factual_entity.state_timeline:
                reconstructed = reconstruct_entity_at(factual_entity, max_ft)
                target_traits = reconstructed["traits"]
                target_beliefs = reconstructed["beliefs"]
            else:
                target_traits = {k: {"value": v.value, "inertia": v.inertia} for k, v in factual_entity.traits.items()}
                target_beliefs = [b.model_dump() for b in factual_entity.beliefs]

            # Back-propagate traits toward reconstructed evidence values (50% blend)
            for trait_name, tv in target_traits.items():
                tv_value = tv["value"] if isinstance(tv, dict) else tv.value
                sandbox_traits = node_data.get("traits", {})
                if trait_name in sandbox_traits and isinstance(sandbox_traits[trait_name], dict):
                    old_val = sandbox_traits[trait_name].get("value", 0.5)
                    shift = (tv_value - old_val) * 0.5
                    sandbox_traits[trait_name]["value"] = max(0.0, min(1.0, old_val + shift))
            # Back-propagate beliefs
            if "beliefs" not in node_data:
                node_data["beliefs"] = []
            existing_beliefs = node_data["beliefs"]
            for belief in target_beliefs:
                b_target_id = belief.get("target_id") if isinstance(belief, dict) else belief.target_id
                b_state = belief.get("perceived_state") if isinstance(belief, dict) else belief.perceived_state
                if not any(b.get("target_id") == b_target_id and
                          b.get("perceived_state") == b_state
                          for b in existing_beliefs):
                    existing_beliefs.append(belief if isinstance(belief, dict) else belief.model_dump())
            logger.info("[Abduction] Conditioned entity %s on present-day evidence.", eid)

        # Case 2: Evidence is an Event — propagate through causal edges
        elif sandbox.has_node(eid):
            node_data = sandbox.nodes[eid]
            if node_data.get("node_type") == "EventNode":
                for ce in global_world_state.causal_topology:
                    if ce.source_id == eid:
                        # Respect propagation_delay
                        if ce.propagation_delay > 0:
                            target_node_data = sandbox.nodes.get(ce.target_id)
                            target_ft = target_node_data.get("fabula_time", float("inf")) if target_node_data else float("inf")
                            if target_ft < ce.fabula_time + ce.propagation_delay:
                                continue
                        mult = evidence_strength_multiplier.get(ce.evidence_strength, 0.5)
                        force_scale = ce.causal_force / 10.0
                        target_node = sandbox.nodes.get(ce.target_id)
                        if target_node and target_node.get("node_type") == "Entity":
                            traits = target_node.get("traits", {})

                            # Precise mutation: use trait_target/trait_delta
                            if ce.causality_type == "mutation" and ce.trait_target is not None:
                                td = traits.get(ce.trait_target)
                                if isinstance(td, dict) and "value" in td:
                                    delta = (ce.trait_delta if ce.trait_delta is not None else 1.0) * mult * force_scale
                                    td["value"] = max(0.0, min(1.0, td["value"] + delta))
                                continue

                            relevant = _MECHANISM_TRAIT_MAP.get(ce.mechanism, None)
                            for trait_name, trait_data in traits.items():
                                if not isinstance(trait_data, dict) or "value" not in trait_data:
                                    continue
                                old_val = trait_data["value"]
                                if relevant is None or trait_name in relevant:
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * force_scale))
                                else:
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * force_scale * _MECHANISM_FALLBACK_FACTOR))
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
    the causal topology forward, adjusting downstream entity traits
    proportionally to the edge's evidence_strength.

    Aligned with CausalPhysicsEngine.propagate():
      • Topological-sort ordering (falls back to fabula_time if cycles).
      • Impact > Inertia gating — small impulses are absorbed.
      • Bidirectional shifts — traits can decrease as well as increase.
      • Mechanism-targeted updates via _MECHANISM_TRAIT_MAP.
      • Spatial affordance checks (unlocked paths only).
    """
    strength_mult = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}

    # 1. Build a causal DiGraph from the global causal_topology,
    #    restricted to nodes present in the sandbox.
    causal_graph = nx.DiGraph()
    edge_meta: Dict[tuple, Dict[str, Any]] = {}  # (src, tgt) → {weight, mechanism}

    for ce in global_world_state.causal_topology:
        src = ce.source_id
        tgt = ce.target_id
        if not sandbox.has_node(src) and src not in {
            nid for nid, _ in sandbox.nodes(data=True)
        }:
            continue
        # Respect propagation_delay
        if ce.propagation_delay > 0:
            target_node_data = sandbox.nodes.get(tgt)
            target_ft = target_node_data.get("fabula_time", float("inf")) if target_node_data else float("inf")
            if target_ft < ce.fabula_time + ce.propagation_delay:
                continue
        evidence_w = strength_mult.get(ce.evidence_strength, 0.5)
        force_scale = ce.causal_force / 10.0
        weight = evidence_w * force_scale
        key = (src, tgt)
        if causal_graph.has_edge(src, tgt):
            existing_w = causal_graph[src][tgt].get("weight", 0.0)
            weight = max(existing_w, weight)
        causal_graph.add_edge(src, tgt, weight=weight)
        # Only overwrite edge_meta if this edge actually won the weight contest
        if key not in edge_meta or weight >= edge_meta[key]["weight"]:
            edge_meta[key] = {
                "weight": weight,
                "mechanism": ce.mechanism,
                "causality_type": ce.causality_type,
                "trait_target": ce.trait_target,
                "trait_delta": ce.trait_delta,
            }

    if causal_graph.number_of_edges() == 0:
        return

    # 2. Topological sort (fall back to fabula_time order if cycles)
    try:
        execution_order = list(nx.topological_sort(causal_graph))
    except nx.NetworkXUnfeasible:
        logger.warning("[Forward Cascade] Cyclic causal graph — falling back to fabula_time order.")
        execution_order = sorted(
            causal_graph.nodes(),
            key=lambda nid: next(
                (e.fabula_time for e in global_world_state.events if e.id == nid),
                float("inf"),
            ),
        )

    # 3. Build a spatial traversability sub-graph for affordance checks
    traversable = nx.DiGraph()
    for u, v, d in sandbox.edges(data=True):
        if d.get("edge_type") == "connected_to" and not d.get("is_locked", False):
            traversable.add_edge(u, v)

    # 4. Propagate
    for node_id in execution_order:
        target = sandbox.nodes.get(node_id)
        if not target or target.get("node_type") != "Entity":
            continue

        incoming = list(causal_graph.in_edges(node_id, data=True))
        if not incoming:
            continue

        traits = target.get("traits", {})
        if not traits:
            continue

        tgt_loc = target.get("location_id")

        for trait_name, trait_data in traits.items():
            if not isinstance(trait_data, dict) or "value" not in trait_data:
                continue

            current_val = trait_data["value"]
            trait_inertia = trait_data.get("inertia", 0.5)
            total_impact = 0.0
            spatial_ok = True

            for src, _, edata in incoming:
                w = edata.get("weight", 0.5)
                meta = edge_meta.get((src, node_id), {})
                mechanism = meta.get("mechanism", "physical")
                edge_ctype = meta.get("causality_type", "chain_reaction")
                edge_trait_target = meta.get("trait_target")
                edge_trait_delta = meta.get("trait_delta")

                # Precise mutation: if the edge specifies a trait_target,
                # only affect that specific trait (skip all others).
                if edge_ctype == "mutation" and edge_trait_target is not None:
                    if trait_name != edge_trait_target:
                        continue  # this edge doesn't affect this trait
                    if edge_trait_delta is not None:
                        total_impact += edge_trait_delta * w
                        continue

                relevant_traits = _MECHANISM_TRAIT_MAP.get(mechanism)

                # Mechanism-targeted gating
                if relevant_traits is not None and trait_name not in relevant_traits:
                    logger.debug("[ForwardCascade] %s→%s trait=%s: mechanism=%s fallback, w %.3f→%.3f",
                                 src, node_id, trait_name, mechanism, w, w * _MECHANISM_FALLBACK_FACTOR)
                    w *= _MECHANISM_FALLBACK_FACTOR

                src_data = sandbox.nodes.get(src)
                if not src_data:
                    continue

                if src_data.get("node_type") == "Entity":
                    src_trait = src_data.get("traits", {}).get(trait_name)
                    if isinstance(src_trait, dict) and "value" in src_trait:
                        # Signed delta: shift toward source trait value
                        total_impact += (src_trait["value"] - current_val) * w
                    else:
                        total_impact += w
                    # Spatial affordance
                    src_loc = src_data.get("location_id")
                    if (src_loc and tgt_loc and src_loc != tgt_loc
                            and traversable.has_node(src_loc)
                            and traversable.has_node(tgt_loc)):
                        if not nx.has_path(traversable, src_loc, tgt_loc):
                            spatial_ok = False
                else:
                    total_impact += w

            if not spatial_ok:
                logger.debug("[ForwardCascade] BLOCKED spatial: %s.%s", node_id, trait_name)
                continue

            if abs(total_impact) <= trait_inertia:
                logger.debug("[ForwardCascade] BLOCKED inertia: %s.%s |impact|=%.3f <= inertia=%.3f",
                             node_id, trait_name, abs(total_impact), trait_inertia)
                continue

            # Dampened shift: effective = impact - sign × inertia
            sign = 1 if total_impact > 0 else -1
            effective_shift = total_impact - sign * trait_inertia
            new_val = max(0.0, min(1.0, current_val + effective_shift))
            trait_data["value"] = new_val

    logger.info("[Forward Cascade] Propagation complete.")


# ==========================================
# HELPER 6: SOCIAL CASCADE (mutation_social propagation)
# ==========================================
def _apply_social_cascade(
    sandbox: nx.MultiDiGraph,
    global_world_state: WorldStateV1,
) -> None:
    """
    Walk ``mutation_social`` causal edges from the global causal_topology
    and apply relationship metric deltas to the sandbox's relationship edges.

    Mirrors CausalPhysicsEngine.propagate_social() for the legacy
    (use_causal_engine=False) counterfactual path.
    """
    strength_mult = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}

    for ce in global_world_state.causal_topology:
        if ce.causality_type != "mutation_social":
            continue

        target_id = ce.target_id          # perspective entity
        counterpart_id = ce.rel_counterpart_id  # other entity in dyad
        metric = ce.trait_target
        raw_delta = ce.trait_delta or 0.0

        if not counterpart_id or not metric:
            continue
        if not sandbox.has_node(target_id) or not sandbox.has_node(counterpart_id):
            continue

        # Respect propagation_delay
        if ce.propagation_delay > 0:
            target_node_data = sandbox.nodes.get(ce.target_id)
            target_ft = target_node_data.get("fabula_time", float("inf")) if target_node_data else float("inf")
            if target_ft < ce.fabula_time + ce.propagation_delay:
                continue

        # Scale delta
        evidence_w = strength_mult.get(ce.evidence_strength, 0.5)
        force_scale = ce.causal_force / 10.0
        scaled_delta = raw_delta * evidence_w * force_scale

        # Find the relationship edge target_id → counterpart_id
        rel_found = False
        for ru, rv, rkey, rdata in sandbox.out_edges(target_id, data=True, keys=True):
            if rv == counterpart_id and rdata.get("edge_type") == "relationship":
                rel_found = True
                current_val = rdata.get(metric, 0.0)
                if not isinstance(current_val, (int, float)):
                    current_val = 0.0
                rel_inertia = rdata.get("inertia", 0.3)

                # Impact > Inertia gating
                if abs(scaled_delta) <= rel_inertia:
                    logger.debug("[SocialCascade] Inertia blocked: %s→%s %s |delta|=%.3f <= inertia=%.3f",
                                 target_id, counterpart_id, metric, abs(scaled_delta), rel_inertia)
                    break

                # Dampened shift
                sign = 1 if scaled_delta > 0 else -1
                effective_shift = scaled_delta - sign * rel_inertia
                new_val = current_val + effective_shift

                # Clamp
                if metric == "fear":
                    new_val = max(0.0, min(1.0, new_val))
                else:
                    new_val = max(-1.0, min(1.0, new_val))

                sandbox[ru][rv][rkey][metric] = new_val
                logger.info("[SocialCascade] %s→%s %s: %.3f→%.3f (trigger=%s)",
                            target_id, counterpart_id, metric, current_val, new_val, ce.source_id)
                break

        # No existing relationship edge — create one with defaults
        if not rel_found:
            edge_attrs = {
                "edge_type": "relationship",
                "affinity": 0.0,
                "fear": 0.0,
                "power_dynamic": 0.0,
                "inertia": 0.3,
                "evidence_strength": "weak",
                "last_updated_fabula": ce.fabula_time,
                "world_id": "shadow",
            }
            if metric == "fear":
                edge_attrs[metric] = max(0.0, min(1.0, scaled_delta))
            else:
                edge_attrs[metric] = max(-1.0, min(1.0, scaled_delta))
            sandbox.add_edge(target_id, counterpart_id, **edge_attrs)
            logger.info("[SocialCascade] Created relationship %s→%s with %s=%.3f (trigger=%s)",
                        target_id, counterpart_id, metric, edge_attrs[metric], ce.source_id)

    logger.info("[Social Cascade] Propagation complete.")