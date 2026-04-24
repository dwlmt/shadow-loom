"""
Step 7 — Causal Physics Engine.

Formalises the three-rung CTF simulation (Observation / Intervention / Counterfactual)
on the NetworkX MultiDiGraph sandbox produced by AMWNInstantiator.create_sandbox().

Key improvements over the inline helpers in narrative_physics.py:
  • Topological-sort-based forward propagation (uses sandbox graph structure,
    not global causal_topology ordering by fabula_time).
  • Impact > Inertia gating during propagation (not just during surgery).
  • Bidirectional trait shifts (traits can decrease, not only increase).
  • Spatial affordance checks during propagation.
  • Auditable hidden_delta storage for counterfactual reasoning.
  • Structured CausalPhysicsResult output.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import networkx as nx
from pydantic import BaseModel, Field

from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)

# =====================================================================
# Result Model
# =====================================================================

STRENGTH_MULTIPLIER: Dict[str, float] = {
    "weak": 0.25,
    "moderate": 0.5,
    "strong": 0.75,
}

# Mechanism → trait affinity mapping.  When an event's causal mechanism
# is known, only traits in the corresponding list receive the full impulse.
# Traits not matching any mechanism family get a reduced fallback.
MECHANISM_TRAIT_MAP: Dict[str, List[str]] = {
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

MECHANISM_FALLBACK_FACTOR: float = 0.2


class TraitMutation(BaseModel):
    """Record of a single trait change applied during propagation."""
    node_id: str
    trait: str
    old_value: float
    new_value: float
    impact: float
    inertia: float


class BlockedPropagation(BaseModel):
    """Record of a propagation that was absorbed by inertia or blocked by affordance."""
    node_id: str
    trait: str
    impact: float
    inertia: float
    reason: str  # "inertia" or "spatial_affordance"


class CausalPhysicsResult(BaseModel):
    """Structured output of the causal physics simulation."""
    sandbox_data: dict = Field(description="nx.node_link_data(sandbox)")
    mutations: List[TraitMutation] = Field(default_factory=list)
    blocked: List[BlockedPropagation] = Field(default_factory=list)
    intervened_nodes: List[str] = Field(default_factory=list)
    hidden_deltas: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="node_id → {trait_name: delta} computed during abduction",
    )


# =====================================================================
# Engine
# =====================================================================

class CausalPhysicsEngine:
    """
    Runs the three-rung CTF simulation on an in-memory NetworkX sandbox.

    Usage::

        engine = CausalPhysicsEngine(sandbox, world_state)
        result = engine.execute(
            rung=3,
            interventions={"ENT_MACBETH.status": "dead"},
            evidence_node_ids=["ENT_MACBETH"],
        )
    """

    def __init__(self, sandbox: nx.MultiDiGraph, world_state: WorldStateV1) -> None:
        self.sandbox = sandbox
        self.world_state = world_state
        self._intervened_nodes: set[str] = set()
        self._hidden_deltas: Dict[str, Dict[str, float]] = {}
        self._mutations: List[TraitMutation] = []
        self._blocked: List[BlockedPropagation] = []

    # ------------------------------------------------------------------
    # Rung 3 — Abduction
    # ------------------------------------------------------------------
    def abduction_update(self, evidence_node_ids: List[str]) -> None:
        """
        Back-propagate present-day evidence into the historical sandbox.

        For entity evidence: computes per-trait hidden_delta, blends traits
        50 % toward factual values, and back-propagates missing beliefs.

        For event evidence: propagates through causal edges weighted by
        evidence_strength.
        """
        if not evidence_node_ids:
            return

        for eid in evidence_node_ids:
            # Case 1 — Evidence is an Entity
            if eid in self.world_state.entities and self.sandbox.has_node(eid):
                factual = self.world_state.entities[eid]
                node_data = self.sandbox.nodes[eid]
                deltas: Dict[str, float] = {}

                sandbox_traits = node_data.get("traits", {})
                for trait_name, tv in factual.traits.items():
                    if trait_name in sandbox_traits and isinstance(sandbox_traits[trait_name], dict):
                        old_val = sandbox_traits[trait_name].get("value", 0.5)
                        delta = tv.value - old_val
                        deltas[trait_name] = delta
                        blended = old_val + delta * 0.5
                        sandbox_traits[trait_name]["value"] = max(0.0, min(1.0, blended))
                        logger.debug("[CausalPhysics·Abduction] %s.%s: old=%.3f factual=%.3f delta=%.3f blended=%.3f",
                                     eid, trait_name, old_val, tv.value, delta, blended)

                if deltas:
                    self._hidden_deltas[eid] = deltas

                # Back-propagate beliefs
                if "beliefs" not in node_data:
                    node_data["beliefs"] = []
                existing = node_data["beliefs"]
                for belief in factual.beliefs:
                    if not any(
                        b.get("target_id") == belief.target_id
                        and b.get("perceived_state") == belief.perceived_state
                        for b in existing
                    ):
                        existing.append(belief.model_dump())

                logger.info("[CausalPhysics·Abduction] Conditioned entity %s (deltas: %s).", eid, deltas)

            # Case 2 — Evidence is an Event
            elif self.sandbox.has_node(eid):
                node_data = self.sandbox.nodes[eid]
                if node_data.get("node_type") == "EventNode":
                    for ce in self.world_state.causal_topology:
                        if ce.source_event_id != eid:
                            continue
                        mult = STRENGTH_MULTIPLIER.get(ce.evidence_strength, 0.5)
                        target_node = self.sandbox.nodes.get(ce.target_node_id)
                        if target_node and target_node.get("node_type") == "Entity":
                            traits = target_node.get("traits", {})
                            relevant = MECHANISM_TRAIT_MAP.get(ce.mechanism, None)
                            for trait_name, trait_data in traits.items():
                                if not isinstance(trait_data, dict) or "value" not in trait_data:
                                    continue
                                old_val = trait_data["value"]
                                if relevant is None or trait_name in relevant:
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * 0.1))
                                    logger.debug("[CausalPhysics·Abduction] Event %s → %s.%s: mechanism=%s matched, old=%.3f new=%.3f",
                                                 eid, ce.target_node_id, trait_name, ce.mechanism, old_val, trait_data["value"])
                                else:
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * 0.1 * MECHANISM_FALLBACK_FACTOR))
                                    logger.debug("[CausalPhysics·Abduction] Event %s → %s.%s: mechanism=%s fallback, old=%.3f new=%.3f",
                                                 eid, ce.target_node_id, trait_name, ce.mechanism, old_val, trait_data["value"])
                    logger.info("[CausalPhysics·Abduction] Propagated evidence from event %s.", eid)
            else:
                logger.warning("[CausalPhysics·Abduction] Evidence node %s not in sandbox. Skipping.", eid)

    # ------------------------------------------------------------------
    # Rung 2 — do-Operator
    # ------------------------------------------------------------------
    def apply_do_operator(self, interventions: Dict[str, Any]) -> None:
        """
        Delegate graph surgery to AMWNInstantiator.execute_interventions()
        and record which nodes were directly intervened on (so propagation
        will not override them).
        """
        AMWNInstantiator.execute_interventions(self.sandbox, interventions)

        for target_path in interventions:
            if "." not in target_path:
                continue
            node_id = target_path.split(".", 1)[0]
            if self.sandbox.has_node(node_id):
                self._intervened_nodes.add(node_id)

        logger.info("[CausalPhysics·do] Surgeries applied. Intervened roots: %s", self._intervened_nodes)

    # ------------------------------------------------------------------
    # Forward Propagation (the new physics)
    # ------------------------------------------------------------------
    def propagate(self) -> None:
        """
        Walk the causal sub-graph in topological order and propagate trait
        shifts downstream, gated by Impact > Inertia and spatial affordance.
        """
        # 1. Extract causal-only DiGraph from MultiDiGraph
        causal_graph = nx.DiGraph()
        for u, v, d in self.sandbox.edges(data=True):
            if d.get("edge_type") == "causal":
                weight = STRENGTH_MULTIPLIER.get(d.get("evidence_strength", "moderate"), 0.5)
                # DiGraph only keeps one edge per (u,v), take the max weight
                mechanism = d.get("mechanism", "physical")
                if causal_graph.has_edge(u, v):
                    existing_w = causal_graph[u][v].get("weight", 0.0)
                    if weight > existing_w:
                        causal_graph[u][v]["weight"] = weight
                        causal_graph[u][v]["mechanism"] = mechanism
                    continue
                causal_graph.add_edge(u, v, weight=weight, mechanism=mechanism)

        if causal_graph.number_of_edges() == 0:
            logger.info("[CausalPhysics·Propagate] No causal edges in sandbox. Skipping.")
            return

        # 2. Topological sort (fallback to BFS order if cycles)
        try:
            execution_order = list(nx.topological_sort(causal_graph))
        except nx.NetworkXUnfeasible:
            logger.warning("[CausalPhysics·Propagate] Cyclic causal graph — falling back to node order.")
            execution_order = list(causal_graph.nodes())

        # 3. Propagate
        for node_id in execution_order:
            if node_id in self._intervened_nodes:
                continue

            node_data = self.sandbox.nodes.get(node_id)
            if not node_data or node_data.get("node_type") != "Entity":
                continue

            incoming = list(causal_graph.in_edges(node_id, data=True))
            if not incoming:
                continue

            traits = node_data.get("traits", {})
            if not traits:
                continue

            # Collect per-trait accumulated impact from all upstream sources
            for trait_name, trait_data in traits.items():
                if not isinstance(trait_data, dict) or "value" not in trait_data:
                    continue

                current_val = trait_data["value"]
                trait_inertia = trait_data.get("inertia", 0.5)

                # Sum impact: each incoming causal edge contributes
                # source_trait_value × causal_weight.  For event→entity edges
                # the source is an EventNode (no traits), so we use the edge
                # weight directly as a fixed impulse.
                total_impact = 0.0
                spatial_ok = True

                for src, _, edata in incoming:
                    w = edata.get("weight", 0.5)
                    mechanism = edata.get("mechanism", "physical")
                    relevant_traits = MECHANISM_TRAIT_MAP.get(mechanism)

                    # Mechanism-targeted gating: reduce weight for non-matching traits
                    if relevant_traits is not None and trait_name not in relevant_traits:
                        logger.debug("[CausalPhysics·Propagate] %s→%s trait=%s: mechanism=%s not in target list, w %.3f→%.3f",
                                     src, node_id, trait_name, mechanism, w, w * MECHANISM_FALLBACK_FACTOR)
                        w *= MECHANISM_FALLBACK_FACTOR

                    src_data = self.sandbox.nodes.get(src)
                    if not src_data:
                        continue

                    if src_data.get("node_type") == "Entity":
                        src_trait = src_data.get("traits", {}).get(trait_name)
                        if isinstance(src_trait, dict) and "value" in src_trait:
                            # Signed delta: shift toward source trait value
                            total_impact += (src_trait["value"] - current_val) * w
                        else:
                            total_impact += w * 0.1
                    else:
                        # EventNode or other — fixed impulse from edge weight
                        total_impact += w * 0.1

                    # Spatial affordance: if the target entity's location is
                    # reachable from the source's location.  We only check when
                    # both sides have locations (entity nodes).
                    if src_data.get("node_type") == "Entity":
                        src_loc = src_data.get("location_id")
                        tgt_loc = node_data.get("location_id")
                        if src_loc and tgt_loc and src_loc != tgt_loc:
                            if not self._check_spatial_reachability(src_loc, tgt_loc):
                                spatial_ok = False

                if not spatial_ok:
                    logger.debug("[CausalPhysics·Propagate] BLOCKED spatial: %s.%s impact=%.3f", node_id, trait_name, total_impact)
                    self._blocked.append(BlockedPropagation(
                        node_id=node_id, trait=trait_name,
                        impact=total_impact, inertia=trait_inertia,
                        reason="spatial_affordance",
                    ))
                    continue

                if abs(total_impact) <= trait_inertia:
                    logger.debug("[CausalPhysics·Propagate] BLOCKED inertia: %s.%s |impact|=%.3f <= inertia=%.3f",
                                 node_id, trait_name, abs(total_impact), trait_inertia)
                    self._blocked.append(BlockedPropagation(
                        node_id=node_id, trait=trait_name,
                        impact=total_impact, inertia=trait_inertia,
                        reason="inertia",
                    ))
                    continue

                # Dampened shift (same formula as existing surgery)
                sign = 1 if total_impact > 0 else -1
                effective_shift = total_impact - sign * trait_inertia
                new_val = max(0.0, min(1.0, current_val + effective_shift))
                logger.debug("[CausalPhysics·Propagate] MUTATED %s.%s: %.3f→%.3f (impact=%.3f, inertia=%.3f, shift=%.3f)",
                             node_id, trait_name, current_val, new_val, total_impact, trait_inertia, effective_shift)

                self._mutations.append(TraitMutation(
                    node_id=node_id, trait=trait_name,
                    old_value=current_val, new_value=new_val,
                    impact=total_impact, inertia=trait_inertia,
                ))
                trait_data["value"] = new_val

        logger.info(
            "[CausalPhysics·Propagate] %d mutations applied, %d blocked.",
            len(self._mutations), len(self._blocked),
        )

    # ------------------------------------------------------------------
    # Spatial reachability helper
    # ------------------------------------------------------------------
    def _check_spatial_reachability(self, src_loc: str, tgt_loc: str) -> bool:
        """Check whether *tgt_loc* is reachable from *src_loc* via unlocked
        (or affordance-unlockable) connected_to edges in the sandbox."""
        traversable = nx.DiGraph()
        for u, v, d in self.sandbox.edges(data=True):
            if d.get("edge_type") != "connected_to":
                continue
            if not d.get("is_locked", False):
                traversable.add_edge(u, v)
            else:
                # Check if any entity in the sandbox owns an item that
                # can unlock the barrier (mirrors AMWNInstantiator logic).
                barrier_id = d.get("barrier_item_id")
                if barrier_id:
                    barrier_node = self.sandbox.nodes.get(barrier_id, {})
                    barrier_name = barrier_node.get("name", "")
                    barrier_node_type = barrier_node.get("node_type", "NarrativeObject")
                    for nid, ndata in self.sandbox.nodes(data=True):
                        if ndata.get("node_type") != "NarrativeObject":
                            continue
                        if ndata.get("owner_id") is None:
                            continue  # unowned items can't be used
                        for aff in ndata.get("affordances", []):
                            if not isinstance(aff, dict):
                                continue
                            if aff.get("action") != "unlock":
                                continue
                            # Match target_type against the barrier's
                            # node_type OR its name (domain term)
                            aff_target = aff.get("target_type", "")
                            if (aff_target == barrier_node_type
                                    or aff_target == barrier_name):
                                traversable.add_edge(u, v)
                                break
                        else:
                            continue
                        break
        if not traversable.has_node(src_loc) or not traversable.has_node(tgt_loc):
            logger.debug("[CausalPhysics·Spatial] %s or %s not in traversable graph", src_loc, tgt_loc)
            return False
        reachable = nx.has_path(traversable, src_loc, tgt_loc)
        logger.debug("[CausalPhysics·Spatial] %s→%s reachable=%s", src_loc, tgt_loc, reachable)
        return reachable

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------
    def execute(
        self,
        rung: int,
        interventions: Dict[str, Any] | None = None,
        evidence_node_ids: List[str] | None = None,
    ) -> CausalPhysicsResult:
        """
        Run the full CTF simulation.

        Parameters
        ----------
        rung : int
            2 for intervention, 3 for counterfactual.
        interventions : dict, optional
            do-operator targets (same format as InterventionQuery.interventions).
        evidence_node_ids : list[str], optional
            Present-day evidence nodes for Rung-3 abduction.
        """
        if rung not in (2, 3):
            raise ValueError(f"Invalid rung={rung}. Must be 2 (intervention) or 3 (counterfactual).")

        # Step A — Abduction (Rung 3 only)
        if rung == 3 and evidence_node_ids:
            self.abduction_update(evidence_node_ids)

        # Step B — do-operator (Rung 2+)
        if rung >= 2 and interventions:
            self.apply_do_operator(interventions)

        # Step C — Forward propagation
        self.propagate()

        return CausalPhysicsResult(
            sandbox_data=nx.node_link_data(self.sandbox),
            mutations=self._mutations,
            blocked=self._blocked,
            intervened_nodes=sorted(self._intervened_nodes),
            hidden_deltas=self._hidden_deltas,
        )
