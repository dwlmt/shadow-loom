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

from shadow_loom.amwn import CtfCalculusReport, apply_ctf_calculus
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import WorldStateV1, reconstruct_entity_at
from shadow_loom.settings import get_settings as _get_settings

logger = logging.getLogger(__name__)

# =====================================================================
# Result Model
# =====================================================================

def _physics_settings():
    return _get_settings().physics


def _strength_multiplier() -> Dict[str, float]:
    return _physics_settings().strength_multiplier


def _strength_weight(label: str) -> float:
    settings = _physics_settings()
    return settings.strength_multiplier.get(label, settings.strength_moderate)


def _force_scale(causal_force: float) -> float:
    settings = _physics_settings()
    if settings.causal_force_scaling == 0:
        return 0.0
    return causal_force / settings.causal_force_scaling


def _mechanism_fallback_factor() -> float:
    return _physics_settings().mechanism_fallback_factor


def _relationship_inertia_default() -> float:
    return _physics_settings().relationship_inertia_default


def _inertia_epsilon() -> float:
    return _physics_settings().inertia_epsilon


# Backward-compatible exports used by tests and downstream imports.
STRENGTH_MULTIPLIER: Dict[str, float] = _strength_multiplier()
MECHANISM_FALLBACK_FACTOR: float = _mechanism_fallback_factor()

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


class SocialMutation(BaseModel):
    """Record of a relationship metric change applied during social propagation."""
    source_entity_id: str
    target_entity_id: str
    metric: str
    old_value: float
    new_value: float
    impact: float
    inertia: float
    triggered_by: str  # source_id of the causal edge (usually EVT_)


class CausalPhysicsResult(BaseModel):
    """Structured output of the causal physics simulation."""
    sandbox_data: dict = Field(description="nx.node_link_data(sandbox)")
    mutations: List[TraitMutation] = Field(default_factory=list)
    social_mutations: List[SocialMutation] = Field(default_factory=list)
    blocked: List[BlockedPropagation] = Field(default_factory=list)
    intervened_nodes: List[str] = Field(default_factory=list)
    hidden_deltas: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="node_id → {trait_name: delta} computed during abduction",
    )
    # ctf-calculus pre-flight pruning (Correa & Bareinboim, ICML 2025).
    rule3_pruned_interventions: List[str] = Field(
        default_factory=list,
        description=(
            "Intervention keys excluded by Rule 3 (Exclusion): the target "
            "node has no directed path to any evidence/target variable in "
            "the mutilated diagram, so the do-surgery is provably vacuous."
        ),
    )
    rule2_redundant_evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Evidence node IDs flagged by Rule 2 (Independence): the node "
            "is d-separated from the intervened variables on the AMWN, so "
            "abduction on it cannot change the counterfactual distribution."
        ),
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
        self._social_mutations: List[SocialMutation] = []
        self._blocked: List[BlockedPropagation] = []
        # Set of node IDs whose outgoing causal edges are eligible to fire
        # in the next propagation step. Populated by _seed_active_sources()
        # at the start of propagate(); also consumed by propagate_social().
        self._active_sources: set[str] = set()
        # Event IDs already applied via Rung-3 abduction Case 2.
        # propagate() must skip edges whose source is in this set so that the
        # same evidence event does not contribute to a target trait twice
        # (once via abduction, once via forward propagation).
        self._abducted_event_evidence: set[str] = set()

    def _simulation_horizon(self) -> float:
        """Return the maximum fabula_time across all events (the 'now' of the story)."""
        if self.world_state.events:
            return max(e.fabula_time for e in self.world_state.events)
        return 0

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
            # Skip WORLD_ nodes — they are structural, not observable evidence.
            if eid.startswith("WORLD_"):
                logger.debug("[CausalPhysics·Abduction] Skipping WORLD_ node: %s", eid)
                continue
            # Case 1 — Evidence is an Entity
            if eid in self.world_state.entities and self.sandbox.has_node(eid):
                factual = self.world_state.entities[eid]
                node_data = self.sandbox.nodes[eid]
                deltas: Dict[str, float] = {}

                # Determine the temporal horizon of the sandbox for reconstruction
                max_ft = max(
                    (d.get("fabula_time", 0) for _, d in self.sandbox.nodes(data=True) if d.get("fabula_time")),
                    default=None,
                )

                # Reconstruct entity state at the sandbox's temporal horizon
                if max_ft is not None and factual.state_timeline:
                    reconstructed = reconstruct_entity_at(factual, max_ft)
                    target_traits = reconstructed["traits"]  # dict of {"value": float, "inertia": float}
                    target_beliefs = reconstructed["beliefs"]
                else:
                    # Legacy fallback: use initial state directly
                    target_traits = {k: {"value": v.value, "inertia": v.inertia} for k, v in factual.traits.items()}
                    target_beliefs = [b.model_dump() for b in factual.beliefs]

                sandbox_traits = node_data.get("traits", {})
                for trait_name, tv in target_traits.items():
                    tv_value = tv["value"] if isinstance(tv, dict) else tv.value
                    if trait_name in sandbox_traits and isinstance(sandbox_traits[trait_name], dict):
                        old_val = sandbox_traits[trait_name].get("value", 0.5)
                        delta = tv_value - old_val
                        deltas[trait_name] = delta
                        blended = old_val + delta * 0.5
                        sandbox_traits[trait_name]["value"] = max(0.0, min(1.0, blended))
                        logger.debug("[CausalPhysics·Abduction] %s.%s: old=%.3f target=%.3f delta=%.3f blended=%.3f",
                                     eid, trait_name, old_val, tv_value, delta, blended)

                if deltas:
                    self._hidden_deltas[eid] = deltas

                # Back-propagate beliefs
                if "beliefs" not in node_data:
                    node_data["beliefs"] = []
                existing = node_data["beliefs"]
                for belief in target_beliefs:
                    b_target_id = belief.get("target_id") if isinstance(belief, dict) else belief.target_id
                    b_state = belief.get("perceived_state") if isinstance(belief, dict) else belief.perceived_state
                    if not any(
                        b.get("target_id") == b_target_id
                        and b.get("perceived_state") == b_state
                        for b in existing
                    ):
                        existing.append(belief if isinstance(belief, dict) else belief.model_dump())

                logger.info("[CausalPhysics·Abduction] Conditioned entity %s (deltas: %s).", eid, deltas)

            # Case 2 — Evidence is an Event
            elif self.sandbox.has_node(eid):
                node_data = self.sandbox.nodes[eid]
                if node_data.get("node_type") == "EventNode":
                    # Mark this event so propagate() skips its outgoing edges
                    # — we have just applied them directly during abduction.
                    self._abducted_event_evidence.add(eid)
                    for ce in self.world_state.causal_topology:
                        if ce.source_id != eid:
                            continue
                        # Respect propagation_delay: skip edges whose effect hasn't elapsed
                        if ce.propagation_delay > 0:
                            target_node = self.sandbox.nodes.get(ce.target_id)
                            target_ft = target_node.get("fabula_time") if target_node else None
                            # Entity targets have no fabula_time — use simulation horizon
                            if target_ft is None:
                                target_ft = self._simulation_horizon()
                            if target_ft < ce.fabula_time + ce.propagation_delay:
                                logger.debug("[CausalPhysics·Abduction] Skipping edge %s→%s: delay=%d not elapsed.",
                                             ce.source_id, ce.target_id, ce.propagation_delay)
                                continue
                        mult = _strength_weight(ce.evidence_strength)
                        force_scale = _force_scale(ce.causal_force)
                        target_node = self.sandbox.nodes.get(ce.target_id)
                        if target_node and target_node.get("node_type") == "Entity":
                            traits = target_node.get("traits", {})

                            # Precise mutation: use trait_target/trait_delta
                            if ce.causality_type == "mutation" and ce.trait_target is not None:
                                td = traits.get(ce.trait_target)
                                if isinstance(td, dict) and "value" in td:
                                    delta = (ce.trait_delta if ce.trait_delta is not None else 1.0) * mult * force_scale
                                    td["value"] = max(0.0, min(1.0, td["value"] + delta))
                                    logger.debug("[CausalPhysics·Abduction] Event %s → %s.%s: mutation delta=%.3f",
                                                 eid, ce.target_id, ce.trait_target, delta)
                                continue

                            relevant = MECHANISM_TRAIT_MAP.get(ce.mechanism, None)
                            for trait_name, trait_data in traits.items():
                                if not isinstance(trait_data, dict) or "value" not in trait_data:
                                    continue
                                old_val = trait_data["value"]
                                if relevant is None or trait_name in relevant:
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * force_scale))
                                    logger.debug("[CausalPhysics·Abduction] Event %s → %s.%s: mechanism=%s matched, old=%.3f new=%.3f",
                                                 eid, ce.target_id, trait_name, ce.mechanism, old_val, trait_data["value"])
                                else:
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * force_scale * _mechanism_fallback_factor()))
                                    logger.debug("[CausalPhysics·Abduction] Event %s → %s.%s: mechanism=%s fallback, old=%.3f new=%.3f",
                                                 eid, ce.target_id, trait_name, ce.mechanism, old_val, trait_data["value"])
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
    def _seed_active_sources(self) -> set[str]:
        """Return the initial set of nodes whose outgoing causal edges fire.

        A canonical event's effects are already realised in each entity's
        ``state_timeline`` and baked into the sandbox via
        ``reconstruct_entity_at(temporal_anchor)``. Re-firing those events
        in propagate() would either double-count (mutation on top of a
        saturated trait) or produce false-positive blocks (impulse < inertia
        once the trait is already at the post-event value).

        To avoid that, only sources that were actually *triggered* this
        simulation step contribute impact:

          • Nodes the user surgically intervened on (Rung 2)
          • Nodes whose state was changed via abduction (Rung 3)
          • Persistent ambient/affordance sources (WorldTrait / Location /
            NarrativeObject) — these are continuous pressure, not consumed
            by being recorded in a snapshot

        Targets that get mutated during propagation are added to the active
        set incrementally so cascading effects propagate further downstream.
        """
        active: set[str] = set(self._intervened_nodes)
        # Abducted entities: their sandbox state was mutated in
        # abduction_update() and their hidden_deltas are now visible to
        # downstream propagation.
        active.update(self._hidden_deltas.keys())
        # Persistent ambient sources are always on.
        for nid, ndata in self.sandbox.nodes(data=True):
            nt = ndata.get("node_type")
            if nt in ("WorldTrait", "Location", "NarrativeObject"):
                active.add(nid)
        return active

    def propagate(self) -> None:
        """
        Walk the causal sub-graph in topological order and propagate trait
        shifts downstream, gated by Impact > Inertia and spatial affordance.
        """
        # 1. Extract causal-only DiGraph from MultiDiGraph
        causal_graph = nx.DiGraph()
        horizon = self._simulation_horizon()
        for u, v, d in self.sandbox.edges(data=True):
            if d.get("edge_type") == "causal":
                # Skip edges whose source event has already been applied via
                # Rung-3 abduction \u2014 prevents double-counting the same
                # evidence (once during abduction, once during propagation).
                if u in self._abducted_event_evidence:
                    logger.debug("[CausalPhysics\u00b7Propagate] Skipping edge %s\u2192%s: source already applied via abduction.",
                                 u, v)
                    continue
                # Respect propagation_delay: skip edges whose effect hasn't manifested yet
                delay = d.get("propagation_delay", 0)
                edge_ft = d.get("fabula_time", 0)
                if delay > 0:
                    target_node = self.sandbox.nodes.get(v, {})
                    # Entity targets have no fabula_time — use simulation horizon
                    target_ft = target_node.get("fabula_time") or horizon
                    if target_ft < edge_ft + delay:
                        logger.debug("[CausalPhysics·Propagate] Skipping edge %s→%s: delay=%d, edge_ft=%d, target_ft=%s",
                                     u, v, delay, edge_ft, target_ft)
                        continue

                evidence_w = _strength_weight(d.get("evidence_strength", "moderate"))
                force_scale = _force_scale(d.get("causal_force", 5.0))
                weight = evidence_w * force_scale
                # DiGraph only keeps one edge per (u,v), take the max weight
                mechanism = d.get("mechanism", "physical")
                trait_target = d.get("trait_target")
                trait_delta = d.get("trait_delta")
                causality_type = d.get("causality_type", "chain_reaction")
                if causal_graph.has_edge(u, v):
                    existing_w = causal_graph[u][v].get("weight", 0.0)
                    if weight > existing_w:
                        causal_graph[u][v]["weight"] = weight
                        causal_graph[u][v]["mechanism"] = mechanism
                        causal_graph[u][v]["trait_target"] = trait_target
                        causal_graph[u][v]["trait_delta"] = trait_delta
                        causal_graph[u][v]["causality_type"] = causality_type
                    continue
                causal_graph.add_edge(u, v, weight=weight, mechanism=mechanism,
                                      trait_target=trait_target, trait_delta=trait_delta,
                                      causality_type=causality_type)

        if causal_graph.number_of_edges() == 0:
            logger.info("[CausalPhysics·Propagate] No causal edges in sandbox. Skipping.")
            return

        # 2. Topological sort (fallback to BFS order if cycles)
        try:
            execution_order = list(nx.topological_sort(causal_graph))
        except nx.NetworkXUnfeasible:
            logger.warning("[CausalPhysics·Propagate] Cyclic causal graph — falling back to node order.")
            execution_order = list(causal_graph.nodes())

        # 2b. Seed the active-source set. Edges only fire when their source
        #     is in this set; downstream targets get added as they mutate.
        self._active_sources = self._seed_active_sources()
        logger.debug("[CausalPhysics·Propagate] Active sources seeded: %d nodes (intervened=%d, abducted=%d)",
                     len(self._active_sources), len(self._intervened_nodes), len(self._hidden_deltas))

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

                # Sum impact: each incoming causal edge contributes.
                # Edge weight already encodes evidence_strength × causal_force.
                # For mutation edges with trait_target/trait_delta, use the
                # precise delta if this trait matches; skip non-matching traits.
                # For entity→entity edges, use signed delta toward source.
                # For event→entity or other, use weight as fixed impulse.
                total_impact = 0.0
                spatial_ok = True

                for src, _, edata in incoming:
                    if src not in self._active_sources:
                        # Source wasn't triggered this step — its canonical
                        # effect is already realised in the entity's
                        # state_timeline (and thus in current_val). Re-firing
                        # would double-count or produce a false block.
                        logger.debug("[CausalPhysics·Propagate] Skipping inactive source %s→%s.%s",
                                     src, node_id, trait_name)
                        continue
                    w = edata.get("weight", 0.5)
                    mechanism = edata.get("mechanism", "physical")
                    edge_ctype = edata.get("causality_type", "chain_reaction")
                    edge_trait_target = edata.get("trait_target")
                    edge_trait_delta = edata.get("trait_delta")

                    # Precise mutation: if the edge specifies a trait_target,
                    # only affect that specific trait (skip all others).
                    if edge_ctype == "mutation" and edge_trait_target is not None:
                        if trait_name != edge_trait_target:
                            continue  # this edge doesn't affect this trait
                        if edge_trait_delta is not None:
                            total_impact += edge_trait_delta * w
                            continue

                    relevant_traits = MECHANISM_TRAIT_MAP.get(mechanism)

                    # Mechanism-targeted gating: reduce weight for non-matching traits
                    if relevant_traits is not None and trait_name not in relevant_traits:
                        fallback = _mechanism_fallback_factor()
                        logger.debug("[CausalPhysics·Propagate] %s→%s trait=%s: mechanism=%s not in target list, w %.3f→%.3f",
                                         src, node_id, trait_name, mechanism, w, w * fallback)
                        w *= fallback

                    src_data = self.sandbox.nodes.get(src)
                    if not src_data:
                        continue

                    # Domain filtering for WORLD_ sources: if edge mechanism
                    # is not in the world trait's affected_domains, apply fallback.
                    if src_data.get("node_type") == "WorldTrait":
                        affected = src_data.get("affected_domains", [])
                        if affected and mechanism not in affected:
                            fallback = _mechanism_fallback_factor()
                            logger.debug("[CausalPhysics·Propagate] WORLD_ domain filter: %s→%s mechanism=%s not in %s, w %.3f→%.3f",
                                     src, node_id, mechanism, affected, w, w * fallback)
                            w *= fallback

                    if src_data.get("node_type") == "Entity":
                        src_trait = src_data.get("traits", {}).get(trait_name)
                        if isinstance(src_trait, dict) and "value" in src_trait:
                            # Signed delta: shift toward source trait value
                            total_impact += (src_trait["value"] - current_val) * w
                        else:
                            total_impact += w
                    elif src_data.get("node_type") == "WorldTrait":
                        # World trait: scale impulse by magnitude intensity
                        mag = src_data.get("magnitude", {})
                        mag_value = mag.get("value", 0.5) if isinstance(mag, dict) else 0.5
                        total_impact += mag_value * w
                    else:
                        # EventNode or other — fixed impulse from edge weight
                        total_impact += w

                    # Spatial affordance: if the target entity's location is
                    # reachable from the source's location.  We only check when
                    # both sides have locations (entity nodes).
                    if src_data and src_data.get("node_type") == "Entity":
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

                if abs(total_impact) <= trait_inertia + _inertia_epsilon():
                    if total_impact == 0.0:
                        # No active source contributed any impulse this step —
                        # not a block, just nothing happened. Skip silently.
                        continue
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
                # Cascade: this target is now an active source for any
                # downstream edges processed later in topo order.
                self._active_sources.add(node_id)

        logger.info(
            "[CausalPhysics·Propagate] %d mutations applied, %d blocked.",
            len(self._mutations), len(self._blocked),
        )

    # ------------------------------------------------------------------
    # Social Propagation (mutation_social edges)
    # ------------------------------------------------------------------
    def propagate_social(self) -> None:
        """
        Walk ``mutation_social`` causal edges and apply relationship metric
        deltas to the sandbox's relationship edges.

        Each mutation_social CausalEdge specifies:
          - source_id:  the causal trigger (EVT_)
          - target_id:  the perspective entity (ENT_) whose relationship changes
          - rel_counterpart_id:  the other entity in the dyad (ENT_)
          - trait_target:  the metric name ('affinity', 'fear', 'power_dynamic')
          - trait_delta:  signed magnitude of the change

        Impact > Inertia gating is applied using the relationship edge's inertia.
        If no relationship edge exists yet, one is created with defaults.

        Uses the ``evidence_strength × (causal_force / 10)`` weight formula
        to scale the delta, matching the trait propagation convention.
        """
        # If propagate() didn't seed (e.g. no causal edges existed), seed now
        # so we still have intervened/abducted/ambient sources available.
        if not self._active_sources:
            self._active_sources = self._seed_active_sources()

        for u, v, d in self.sandbox.edges(data=True):
            if d.get("edge_type") != "causal":
                continue
            if d.get("causality_type") != "mutation_social":
                continue

            source_id = u  # EVT_ trigger
            target_id = d.get("target_id", v)  # perspective entity
            counterpart_id = d.get("rel_counterpart_id")
            metric = d.get("trait_target")
            raw_delta = d.get("trait_delta", 0.0)

            # Active-source gating: only fire if the triggering event was
            # actually invoked this step (intervention or cascade). Canonical
            # event effects are already realised in the relationship edge's
            # current values.
            if source_id not in self._active_sources:
                logger.debug("[CausalPhysics·SocialProp] Skipping inactive source %s→%s (%s)",
                             source_id, target_id, metric)
                continue

            if not counterpart_id or not metric:
                logger.warning("[CausalPhysics·SocialProp] Incomplete mutation_social edge %s→%s: "
                               "counterpart=%s, metric=%s. Skipping.", u, v, counterpart_id, metric)
                continue

            if not self.sandbox.has_node(target_id) or not self.sandbox.has_node(counterpart_id):
                logger.debug("[CausalPhysics·SocialProp] Endpoint missing: target=%s, counterpart=%s",
                             target_id, counterpart_id)
                continue

            # Scale delta by evidence_strength × causal_force
            evidence_w = _strength_weight(d.get("evidence_strength", "moderate"))
            force_scale = _force_scale(d.get("causal_force", 5.0))
            scaled_delta = raw_delta * evidence_w * force_scale

            # Find the relationship edge target_id → counterpart_id
            rel_found = False
            for ru, rv, rkey, rdata in self.sandbox.out_edges(target_id, data=True, keys=True):
                if rv == counterpart_id and rdata.get("edge_type") == "relationship":
                    rel_found = True
                    current_val = rdata.get(metric, 0.0)
                    if not isinstance(current_val, (int, float)):
                        current_val = 0.0
                    rel_inertia = rdata.get("inertia", _relationship_inertia_default())

                    # Impact > Inertia gating
                    if abs(scaled_delta) <= rel_inertia + _inertia_epsilon():
                        logger.debug("[CausalPhysics·SocialProp] Inertia blocked: %s→%s %s |delta|=%.3f <= inertia=%.3f",
                                     target_id, counterpart_id, metric, abs(scaled_delta), rel_inertia)
                        self._blocked.append(BlockedPropagation(
                            node_id=target_id, trait=f"rel.{counterpart_id}.{metric}",
                            impact=scaled_delta, inertia=rel_inertia, reason="inertia",
                        ))
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

                    self.sandbox[ru][rv][rkey][metric] = new_val
                    self._social_mutations.append(SocialMutation(
                        source_entity_id=target_id,
                        target_entity_id=counterpart_id,
                        metric=metric,
                        old_value=current_val,
                        new_value=new_val,
                        impact=scaled_delta,
                        inertia=rel_inertia,
                        triggered_by=source_id,
                    ))
                    logger.info("[CausalPhysics·SocialProp] %s→%s %s: %.3f→%.3f (trigger=%s, delta=%.3f, inertia=%.3f)",
                                target_id, counterpart_id, metric, current_val, new_val, source_id, scaled_delta, rel_inertia)
                    break

            # No existing relationship edge — create one with defaults
            if not rel_found:
                edge_attrs = {
                    "edge_type": "relationship",
                    "affinity": 0.0,
                    "fear": 0.0,
                    "power_dynamic": 0.0,
                    "inertia": _relationship_inertia_default(),
                    "evidence_strength": "weak",
                    "last_updated_fabula": d.get("fabula_time", 0),
                    "world_id": "shadow",
                }
                # Apply the delta directly (no inertia gating on creation)
                if metric == "fear":
                    edge_attrs[metric] = max(0.0, min(1.0, scaled_delta))
                else:
                    edge_attrs[metric] = max(-1.0, min(1.0, scaled_delta))
                self.sandbox.add_edge(target_id, counterpart_id, **edge_attrs)
                self._social_mutations.append(SocialMutation(
                    source_entity_id=target_id,
                    target_entity_id=counterpart_id,
                    metric=metric,
                    old_value=0.0,
                    new_value=edge_attrs[metric],
                    impact=scaled_delta,
                    inertia=_relationship_inertia_default(),
                    triggered_by=source_id,
                ))
                logger.info("[CausalPhysics·SocialProp] Created relationship %s→%s with %s=%.3f (trigger=%s)",
                            target_id, counterpart_id, metric, edge_attrs[metric], source_id)

        logger.info("[CausalPhysics·SocialProp] %d social mutations applied.", len(self._social_mutations))

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
    # ctf-calculus pre-flight (Correa & Bareinboim, ICML 2025)
    # ------------------------------------------------------------------
    def _apply_ctf_calculus_preflight(
        self,
        *,
        rung: int,
        interventions: Dict[str, Any],
        evidence_node_ids: List[str],
    ) -> CtfCalculusReport:
        """Run the static AMWN d-separation checks before simulation.

        Rule 3 (Exclusion) prunes intervention keys whose target node has
        no directed path to any evidence/target variable in the mutilated
        diagram. Rule 2 (Independence) flags evidence nodes that are
        d-separated from every intervened variable on the AMWN.

        The report is purely diagnostic at this layer; the engine still
        executes the requested surgery so the heuristic narrative
        physics layer remains unaffected.
        """
        if not interventions:
            return CtfCalculusReport()
        try:
            return apply_ctf_calculus(
                self.world_state,
                interventions=interventions,
                evidence_node_ids=evidence_node_ids,
                # Without explicit query targets, Rule 3 has nothing to
                # test against — pass None to disable that branch.
                target_node_ids=None,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("[CausalPhysics\u00b7ctf-calculus] Pre-flight failed; skipping report.")
            return CtfCalculusReport()

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

        # Step 0 — ctf-calculus pre-flight (Rules 2 & 3).
        # Static graphical reasoning on the AMWN: prune interventions that
        # are *provably* vacuous (Rule 3) and flag evidence that is
        # d-separated from every intervention (Rule 2). The simulation
        # below still runs on the unpruned set so heuristic narrative
        # propagation is preserved — the report is informational and the
        # pipeline can decide whether to short-circuit.
        ctf_report = self._apply_ctf_calculus_preflight(
            rung=rung,
            interventions=interventions or {},
            evidence_node_ids=evidence_node_ids or [],
        )

        # Step A — Abduction (Rung 3 only)
        if rung == 3 and evidence_node_ids:
            self.abduction_update(evidence_node_ids)

        # Step B — do-operator (Rung 2+)
        if rung >= 2 and interventions:
            self.apply_do_operator(interventions)

        # Step C — Forward propagation
        self.propagate()

        # Step D — Social propagation (mutation_social edges)
        self.propagate_social()

        result = CausalPhysicsResult(
            sandbox_data=nx.node_link_data(self.sandbox),
            mutations=self._mutations,
            social_mutations=self._social_mutations,
            blocked=self._blocked,
            intervened_nodes=sorted(self._intervened_nodes),
            hidden_deltas=self._hidden_deltas,
            rule3_pruned_interventions=ctf_report.rule3_pruned,
            rule2_redundant_evidence=ctf_report.rule2_redundant_evidence,
        )
        _log_physics_result(rung, interventions, evidence_node_ids, result)
        return result


# =====================================================================
# Readable summary logger
# =====================================================================

_RUNG_NAMES = {
    1: "Rung 1 (Observation)",
    2: "Rung 2 (Intervention / do-operator)",
    3: "Rung 3 (Counterfactual / abduction)",
}


def _log_physics_result(
    rung: int,
    interventions: Dict[str, Any] | None,
    evidence_node_ids: List[str] | None,
    result: CausalPhysicsResult,
    *,
    max_lines_per_section: int = 12,
) -> None:
    """Emit a multi-line, human-readable INFO summary of an engine run.

    The full structured payload is too large to dump verbatim
    (``sandbox_data`` alone is the entire NetworkX graph). Instead this
    helper prints just the parts a human cares about: how the engine was
    invoked, what was intervened on, what mutations happened, what was
    blocked, and what hidden counterfactual deltas were inferred — each
    truncated to a sensible cap.
    """
    if not logger.isEnabledFor(logging.INFO):
        return

    lines: list[str] = []
    label = _RUNG_NAMES.get(rung, f"Rung {rung}")
    lines.append(
        f"[CausalPhysics·Result] {label} — "
        f"{len(result.mutations)} mutations, "
        f"{len(result.social_mutations)} social mutations, "
        f"{len(result.blocked)} blocked, "
        f"{len(result.intervened_nodes)} intervened, "
        f"{len(result.hidden_deltas)} hidden-delta nodes"
    )

    if interventions:
        ivs = list(interventions.items())[:max_lines_per_section]
        lines.append("  Interventions (do-operator):")
        for nid, payload in ivs:
            lines.append(f"    - {nid} ← {payload}")
        if len(interventions) > max_lines_per_section:
            lines.append(
                f"    … (+{len(interventions) - max_lines_per_section} more)"
            )

    if evidence_node_ids:
        ev_preview = ", ".join(evidence_node_ids[:max_lines_per_section])
        more = (
            f" (+{len(evidence_node_ids) - max_lines_per_section} more)"
            if len(evidence_node_ids) > max_lines_per_section else ""
        )
        lines.append(f"  Abduction evidence: {ev_preview}{more}")

    if result.intervened_nodes:
        nodes = ", ".join(result.intervened_nodes[:max_lines_per_section])
        more = (
            f" (+{len(result.intervened_nodes) - max_lines_per_section} more)"
            if len(result.intervened_nodes) > max_lines_per_section else ""
        )
        lines.append(f"  Intervened nodes: {nodes}{more}")

    if result.mutations:
        # Sort by absolute shift descending so the most consequential
        # changes come first.
        sorted_muts = sorted(
            result.mutations,
            key=lambda m: abs(m.new_value - m.old_value),
            reverse=True,
        )
        lines.append("  Trait mutations (largest first):")
        for m in sorted_muts[:max_lines_per_section]:
            shift = m.new_value - m.old_value
            arrow = "↑" if shift > 0 else ("↓" if shift < 0 else "·")
            lines.append(
                f"    {arrow} {m.node_id}.{m.trait}: "
                f"{m.old_value:.3f} → {m.new_value:.3f} "
                f"(Δ={shift:+.3f}, impact={m.impact:.3f}, "
                f"inertia={m.inertia:.3f})"
            )
        if len(sorted_muts) > max_lines_per_section:
            lines.append(
                f"    … (+{len(sorted_muts) - max_lines_per_section} more)"
            )

    if result.social_mutations:
        lines.append("  Social mutations (relationships):")
        sorted_socs = sorted(
            result.social_mutations,
            key=lambda s: abs(s.new_value - s.old_value),
            reverse=True,
        )
        for s in sorted_socs[:max_lines_per_section]:
            shift = s.new_value - s.old_value
            arrow = "↑" if shift > 0 else ("↓" if shift < 0 else "·")
            lines.append(
                f"    {arrow} {s.source_entity_id}↔{s.target_entity_id}.{s.metric}: "
                f"{s.old_value:.3f} → {s.new_value:.3f} "
                f"(Δ={shift:+.3f}, via {s.triggered_by})"
            )
        if len(sorted_socs) > max_lines_per_section:
            lines.append(
                f"    … (+{len(sorted_socs) - max_lines_per_section} more)"
            )

    if result.blocked:
        lines.append("  Blocked propagations:")
        for b in result.blocked[:max_lines_per_section]:
            lines.append(
                f"    × {b.node_id}.{b.trait}: "
                f"impact={b.impact:.3f} ≤ inertia={b.inertia:.3f} "
                f"({b.reason})"
            )
        if len(result.blocked) > max_lines_per_section:
            lines.append(
                f"    … (+{len(result.blocked) - max_lines_per_section} more)"
            )

    if result.hidden_deltas:
        lines.append("  Hidden counterfactual deltas (abduction inferred):")
        items = list(result.hidden_deltas.items())[:max_lines_per_section]
        for nid, deltas in items:
            d_str = ", ".join(
                f"{t}{v:+.3f}" for t, v in list(deltas.items())[:6]
            )
            lines.append(f"    {nid}: {d_str}")
        if len(result.hidden_deltas) > max_lines_per_section:
            lines.append(
                f"    … (+{len(result.hidden_deltas) - max_lines_per_section} more)"
            )

    if result.rule3_pruned_interventions:
        lines.append("  ctf-calculus Rule 3 pruned (vacuous interventions):")
        for k in result.rule3_pruned_interventions[:max_lines_per_section]:
            lines.append(f"    × {k}")
    if result.rule2_redundant_evidence:
        lines.append("  ctf-calculus Rule 2 redundant evidence (d-separated):")
        for k in result.rule2_redundant_evidence[:max_lines_per_section]:
            lines.append(f"    × {k}")

    logger.info("\n".join(lines))
