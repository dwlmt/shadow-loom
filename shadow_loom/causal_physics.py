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

import copy
import logging
import math
import random
import statistics
from typing import Any, Dict, List, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field

from shadow_loom.amwn import CtfCalculusReport, apply_ctf_calculus, build_causal_diagram
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
    reason: str  # "inertia", "spatial_affordance", "cycle", or "noisy_or_absorbed"


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


class NoisyOrProbability(BaseModel):
    """Per-trait noisy-OR aggregate plus its per-edge components.

    ``per_edge`` lists ``{"source_id": str, "p": float, "weighted_impulse":
    float}`` dicts — one Bernoulli "attempt" per incoming causal edge — so
    the auditor can see which sources drove the joint probability and which
    were absorbed by inertia.
    """
    node_id: str
    trait: str
    inertia: float
    per_edge: List[Dict[str, Any]] = Field(default_factory=list)
    aggregate_probability: float = Field(
        description="1 - prod(1 - p_i) over the per-edge components."
    )
    fired: bool = Field(
        description=(
            "Under deterministic noisy-OR mode: aggregate_probability >= "
            "noisy_or_threshold. Under sampled mode: result of the Bernoulli "
            "draw."
        ),
    )


class TraitDistribution(BaseModel):
    """Sampled distribution over a post-propagation trait value.

    Populated by ``CausalPhysicsEngine.execute_distribution`` when the
    Monte-Carlo orchestration entry point is used. ``samples_count`` is
    the number of successful samples (excluding any that failed).
    """
    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    samples_count: int


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
            "the mutilated diagram, so the do-surgery is provably vacuous "
            "*relative to the extracted graph*. Whether the engine actually "
            "filtered these from the working set is governed by "
            "``CausalPhysicsSettings.rule3_pruning_mode``; see "
            "``rule3_pruning_mode`` below."
        ),
    )
    rule3_pruning_mode: Literal["advisory", "prune"] = Field(
        default="advisory",
        description=(
            "Mode the engine ran under. 'advisory': "
            "``rule3_pruned_interventions`` is reportage only — the do-"
            "surgery was still applied. 'prune': those keys were filtered "
            "out of the working set before simulation."
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
    # ------------------------------------------------------------------
    # Probabilistic outputs (populated only when the corresponding modes
    # are active in CausalPhysicsSettings; empty under default settings).
    # ------------------------------------------------------------------
    noisy_or_probabilities: List["NoisyOrProbability"] = Field(
        default_factory=list,
        description=(
            "Per (node, trait) noisy-OR aggregate probability that the trait "
            "shifted this step, plus the per-edge contributing probabilities. "
            "Populated under propagation_mode='noisy_or'."
        ),
    )
    trait_distributions: Dict[str, Dict[str, "TraitDistribution"]] = Field(
        default_factory=dict,
        description=(
            "node_id -> trait_name -> TraitDistribution. Populated by "
            "execute_distribution(); empty for single-shot execute() runs."
        ),
    )


# =====================================================================
# Probabilistic helpers (sampling + noisy-OR)
# =====================================================================


def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _evidence_strength_sigma(label: str) -> float:
    """Std-dev fraction of nominal causal_force for the given evidence_strength bucket."""
    s = _physics_settings()
    if label == "weak":
        return s.causal_force_sigma_weak
    if label == "strong":
        return s.causal_force_sigma_strong
    return s.causal_force_sigma_moderate


def _sample_causal_force(
    nominal_force: float,
    evidence_strength: str,
    rng: random.Random,
) -> float:
    """Draw causal_force ~ Normal(nominal, sigma * nominal), clamped to [0, 10].

    The point estimate emitted by the LLM is treated as the mean of a
    Normal whose std-dev scales with evidence_strength: weak evidence ⇒
    wider distribution, strong evidence ⇒ tight around the point estimate.
    """
    sigma_frac = _evidence_strength_sigma(evidence_strength)
    sigma = max(1e-6, sigma_frac * abs(nominal_force))
    sample = rng.gauss(nominal_force, sigma)
    return max(0.0, min(10.0, sample))


def _sample_trait_value(value: float, inertia: float, rng: random.Random) -> float:
    """Draw a trait value from Beta(alpha, beta) parameterised by inertia.

    Concentration kappa = 1 / (1 - inertia + eps) so high-inertia traits
    yield tight Betas (point-estimate-like), low-inertia traits yield
    diffuse Betas (more uncertainty around the point estimate).
    """
    eps = 1e-3
    value = max(eps, min(1.0 - eps, value))
    inertia = max(0.0, min(1.0, inertia))
    kappa = 1.0 / max(eps, 1.0 - inertia + eps)
    alpha = max(eps, value * kappa)
    beta = max(eps, (1.0 - value) * kappa)
    sample = rng.betavariate(alpha, beta)
    return max(0.0, min(1.0, sample))


def _noisy_or_per_edge_probability(
    weighted_impulse: float,
    inertia: float,
    temperature: float,
) -> float:
    """Per-edge p_i = sigmoid((|w*impulse| - inertia) / temperature).

    A weighted impulse exactly equal to the trait's inertia gives p_i =
    0.5 (the point at which the deterministic gate would barely flip);
    larger impulses saturate toward 1, smaller toward 0. Temperature
    controls how sharp the transition is.
    """
    temp = max(1e-6, temperature)
    return _sigmoid((abs(weighted_impulse) - inertia) / temp)


def _noisy_or_aggregate(per_edge_probs: List[float]) -> float:
    """1 - prod(1 - p_i). Independent-Bernoulli OR over per-edge attempts."""
    if not per_edge_probs:
        return 0.0
    survival = 1.0
    for p in per_edge_probs:
        survival *= max(0.0, 1.0 - p)
    return 1.0 - survival


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
        # Per-trait pinning: (node_id, trait_name) pairs that propagate()
        # must NOT overwrite. A path like ``ENT_X.traits.fear=0.9`` adds
        # ``("ENT_X", "fear")`` here so the surgical pin is honoured
        # without freezing the entity's other traits. Bare-node and spawn
        # interventions register the sentinel ``"*"`` to mean "every trait
        # of this node is pinned."
        self._intervened_traits: set[tuple[str, str]] = set()
        self._hidden_deltas: Dict[str, Dict[str, float]] = {}
        self._mutations: List[TraitMutation] = []
        self._social_mutations: List[SocialMutation] = []
        self._blocked: List[BlockedPropagation] = []
        # Noisy-OR per-trait records, populated only when
        # ``CausalPhysicsSettings.propagation_mode == "noisy_or"``.
        self._noisy_or_records: List[NoisyOrProbability] = []
        # RNG for the Monte-Carlo sampling path. Seeded explicitly when a
        # caller invokes ``execute_distribution``; unused under the default
        # deterministic path so the legacy code path stays bit-for-bit
        # reproducible.
        self._rng: Optional[random.Random] = None
        # When True, the noisy-OR gate draws a Bernoulli sample per trait
        # using ``self._rng`` instead of thresholding the aggregate
        # probability. Only flipped on inside execute_distribution().
        self._sample_noisy_or: bool = False
        # Marker set by ``execute_distribution`` on its per-sample sub-
        # engines so their ``execute()`` calls bypass the auto-route to
        # the Monte-Carlo orchestrator (otherwise we'd recurse).
        self._in_mc_sample: bool = False
        # Set of node IDs whose outgoing causal edges are eligible to fire
        # in the next propagation step. Populated by _seed_active_sources()
        # at the start of propagate(); also consumed by propagate_social().
        self._active_sources: set[str] = set()
        # Event IDs already applied via Rung-3 abduction Case 2.
        # propagate() must skip edges whose source is in this set so that the
        # same evidence event does not contribute to a target trait twice
        # (once via abduction, once via forward propagation).
        self._abducted_event_evidence: set[str] = set()
        # Cached traversable spatial subgraph; invariant per execute() so
        # we build it once and reuse for all _check_spatial_reachability
        # calls instead of rebuilding on every (src_loc, tgt_loc) pair.
        self._spatial_traversable: Optional[nx.DiGraph] = None

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
                ab_settings = _physics_settings()
                blend_mode = ab_settings.abduction_blend_mode
                ev_precision = ab_settings.abduction_evidence_precision
                for trait_name, tv in target_traits.items():
                    tv_value = tv["value"] if isinstance(tv, dict) else tv.value
                    if trait_name in sandbox_traits and isinstance(sandbox_traits[trait_name], dict):
                        old_val = sandbox_traits[trait_name].get("value", 0.5)
                        delta = tv_value - old_val
                        deltas[trait_name] = delta
                        trait_inertia = sandbox_traits[trait_name].get("inertia", 0.5)
                        if blend_mode == "bayesian":
                            # Treat inertia as the *precision* of the
                            # historical prior. Posterior mean is the
                            # precision-weighted combination of the prior
                            # (old_val) and the present-day evidence
                            # (tv_value). High-inertia traits shrink toward
                            # the historical baseline; low-inertia traits
                            # snap to the evidence.
                            denom = trait_inertia + ev_precision
                            if denom <= 0:
                                blended = old_val
                            else:
                                blended = (
                                    trait_inertia * old_val
                                    + ev_precision * tv_value
                                ) / denom
                            logger.debug(
                                "[CausalPhysics·Abduction·Bayes] %s.%s: prior=%.3f (k=%.3f) ev=%.3f (k=%.3f) post=%.3f",
                                eid, trait_name, old_val, trait_inertia,
                                tv_value, ev_precision, blended,
                            )
                        else:
                            # Legacy: blend toward factual value, damped
                            # by inertia. High-inertia traits resist;
                            # low-inertia absorb fully.
                            blend_factor = max(0.0, min(1.0, 1.0 - trait_inertia))
                            blended = old_val + delta * blend_factor
                            logger.debug(
                                "[CausalPhysics·Abduction] %s.%s: old=%.3f target=%.3f delta=%.3f blend=%.2f blended=%.3f",
                                eid, trait_name, old_val, tv_value, delta, blend_factor, blended,
                            )
                        sandbox_traits[trait_name]["value"] = max(0.0, min(1.0, blended))

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

        Both *path-style* keys (``ENT_X.traits.fear``) and *node-level*
        keys (``ENT_X``) record the underlying node id in
        ``_intervened_nodes`` so propagation cannot subsequently overwrite
        a value the user pinned by intervention.

        We additionally record per-trait pins in ``_intervened_traits`` so
        propagation can freeze only the specific trait the user surgically
        set instead of freezing every trait on the entity — a
        ``do(ENT_X.traits.fear=0.9)`` should leave ``loyalty`` and ``hope``
        free to evolve in response to other causal forces.
        """
        AMWNInstantiator.execute_interventions(self.sandbox, interventions)

        # First pass: collect every per-trait pin the user explicitly
        # named. We need this set up-front so a co-occurring wildcard
        # (``do(ENT_X.spawn)``) doesn't smother explicit per-trait
        # surgeries on the same node \u2014 the previous logic added
        # ``("ENT_X", "*")`` whenever spawn or a bare-node intervention
        # was present, and ``propagate()`` then froze every trait of the
        # entity, ignoring the fact that the user only wanted *some*
        # traits pinned and others free to evolve under propagation.
        per_trait_pins_by_node: Dict[str, set[str]] = {}
        for target_path in interventions:
            if "." not in target_path:
                continue
            node_id, sub_path = target_path.split(".", 1)
            sub = sub_path.strip()
            if sub.startswith("traits."):
                parts = sub.split(".")
                trait_name = parts[1] if len(parts) >= 2 else ""
                if trait_name:
                    per_trait_pins_by_node.setdefault(node_id, set()).add(trait_name)

        for target_path in interventions:
            if "." in target_path:
                node_id, sub_path = target_path.split(".", 1)
            else:
                node_id, sub_path = target_path, ""
            if not self.sandbox.has_node(node_id):
                continue
            self._intervened_nodes.add(node_id)

            # Decide which traits, if any, are pinned by this intervention.
            sub = sub_path.strip()
            if not sub or sub == "spawn":
                # Bare-node or genesis spawn. If the user *also* pinned
                # specific traits on this node, only honour those pins
                # (the wildcard would otherwise smother them and freeze
                # every other trait too). With no per-trait companion
                # surgery, the spawn pins everything via the wildcard.
                pinned = per_trait_pins_by_node.get(node_id)
                if pinned:
                    for trait_name in pinned:
                        self._intervened_traits.add((node_id, trait_name))
                else:
                    self._intervened_traits.add((node_id, "*"))
            elif sub.startswith("traits."):
                # ``traits.<name>`` or ``traits.<name>.value`` \u2014 pin the
                # specific trait only.
                parts = sub.split(".")
                trait_name = parts[1] if len(parts) >= 2 else ""
                if trait_name:
                    self._intervened_traits.add((node_id, trait_name))
            # status / location_id / beliefs / properties / etc. don't pin
            # any trait \u2014 propagation over the entity's traits is unaffected.

        logger.info(
            "[CausalPhysics·do] Surgeries applied. Intervened roots: %s; pinned traits: %s",
            self._intervened_nodes, self._intervened_traits,
        )

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
                force_scale = _force_scale(d.get("causal_force", _physics_settings().default_causal_force))
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

        # 2. Topological sort. If the causal sub-graph contains cycles
        #    (extraction occasionally produces them), fall back to an
        #    SCC-condensed ordering. *Within* a non-trivial SCC we refuse
        #    to propagate (every member is recorded as ``BlockedPropagation
        #    (reason="cycle")`` so the audit trail is explicit) — picking
        #    an arbitrary visit order would bake meaningless structure
        #    into the cycle's resolution.
        cyclic_blocked: set[str] = set()
        try:
            execution_order = list(nx.topological_sort(causal_graph))
        except nx.NetworkXUnfeasible:
            sccs = list(nx.strongly_connected_components(causal_graph))
            cyclic_sccs = [s for s in sccs if len(s) > 1]
            for s in cyclic_sccs:
                cyclic_blocked |= s
            logger.warning(
                "[CausalPhysics·Propagate] Cyclic causal graph: %d SCC(s) with "
                "%d node(s) total. Cyclic clusters are blocked from "
                "propagation; only acyclic spines fire.",
                len(cyclic_sccs), len(cyclic_blocked),
            )
            condensation = nx.condensation(causal_graph, sccs)
            execution_order = []
            for comp_idx in nx.topological_sort(condensation):
                # ``members`` is the set of original node ids in this SCC.
                members = condensation.nodes[comp_idx]["members"]
                # Sort for determinism so test runs are reproducible.
                execution_order.extend(sorted(members))

        # 2b. Seed the active-source set. Edges only fire when their source
        #     is in this set; downstream targets get added as they mutate.
        self._active_sources = self._seed_active_sources()
        logger.debug("[CausalPhysics·Propagate] Active sources seeded: %d nodes (intervened=%d, abducted=%d)",
                     len(self._active_sources), len(self._intervened_nodes), len(self._hidden_deltas))

        # 2c. Capture per-trait baselines (the pre-propagation value of every
        #     entity trait) so that ``entity_trait_baseline_drift_rate``
        #     can pull mutated traits back toward type after the
        #     propagation pass. Encodes "characters return to type" so a
        #     single off-screen shock doesn't permanently rewrite a high-
        #     inertia trait.
        trait_baselines: Dict[str, Dict[str, float]] = {}
        if _physics_settings().entity_trait_baseline_drift_rate > 0:
            for nid, ndata in self.sandbox.nodes(data=True):
                if ndata.get("node_type") != "Entity":
                    continue
                for tname, tdata in (ndata.get("traits") or {}).items():
                    if isinstance(tdata, dict) and "value" in tdata:
                        trait_baselines.setdefault(nid, {})[tname] = tdata["value"]

        # 3. Propagate
        for node_id in execution_order:
            # Per-trait pinning: an entity may be in _intervened_nodes
            # because the user surgically pinned ONE of its traits, but
            # we should still let other traits respond to causal forces.
            # Only skip the entity wholesale when every trait is pinned
            # (the wildcard "*" sentinel set by bare-node / spawn surgery).
            if (node_id, "*") in self._intervened_traits:
                continue

            # Refuse to propagate inside a cyclic SCC — record the block
            # explicitly so callers can see why the trait didn't move.
            if node_id in cyclic_blocked:
                node_data = self.sandbox.nodes.get(node_id)
                if node_data and node_data.get("node_type") == "Entity":
                    for trait_name, trait_data in (node_data.get("traits") or {}).items():
                        if not isinstance(trait_data, dict) or "value" not in trait_data:
                            continue
                        self._blocked.append(BlockedPropagation(
                            node_id=node_id, trait=trait_name,
                            impact=0.0, inertia=trait_data.get("inertia", 0.5),
                            reason="cycle",
                        ))
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

                # Per-trait pin: skip just this trait if the user pinned it.
                if (node_id, trait_name) in self._intervened_traits:
                    continue

                current_val = trait_data["value"]
                trait_inertia = trait_data.get("inertia", 0.5)

                # Sum impact: each incoming causal edge contributes.
                # Edge weight already encodes evidence_strength × causal_force.
                # For mutation edges with trait_target/trait_delta, use the
                # precise delta if this trait matches; skip non-matching traits.
                # For entity→entity edges, use signed delta toward source.
                # For event→entity or other, use weight as fixed impulse.
                #
                # We accumulate weighted contributions and a running
                # ``total_weight`` so the final aggregate is normalised
                # (weighted average), not a raw sum. The previous
                # ``total_impact - sign*inertia`` formulation was
                # scale-dependent: ten weak edges could outweigh a single
                # canonical force purely by stacking, defeating the
                # Impact > Inertia gate as a meaningful threshold.
                total_impact = 0.0
                total_weight = 0.0
                spatial_ok = True
                # Per-edge signed contributions: (source_id, signed_w*impulse)
                # tuples used by the noisy-OR aggregator. Always populated
                # so the cost is identical under the deterministic path
                # (the list is just unused).
                per_edge_contributions: List[tuple[str, float]] = []

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
                            contrib = edge_trait_delta * w
                            total_impact += contrib
                            total_weight += w
                            per_edge_contributions.append((src, contrib))
                            continue

                    relevant_traits = MECHANISM_TRAIT_MAP.get(mechanism)

                    # Mechanism-targeted gating: reduce weight for non-matching traits.
                    # We track the worst single fallback (mechanism mismatch OR
                    # WORLD_ domain mismatch) and apply it once, instead of
                    # multiplying both penalties — a trait that loses both a
                    # mechanism and a domain match shouldn't be ×fallback².
                    fallback_penalty = 1.0
                    if relevant_traits is not None and trait_name not in relevant_traits:
                        fb = _mechanism_fallback_factor()
                        fallback_penalty = min(fallback_penalty, fb)
                        logger.debug("[CausalPhysics·Propagate] %s→%s trait=%s: mechanism=%s not in target list, fb=%.3f",
                                         src, node_id, trait_name, mechanism, fb)

                    src_data = self.sandbox.nodes.get(src)
                    if not src_data:
                        continue

                    # Domain filtering for WORLD_ sources: if edge mechanism
                    # is not in the world trait's affected_domains, contribute
                    # to the (single) fallback penalty.
                    if src_data.get("node_type") == "WorldTrait":
                        affected = src_data.get("affected_domains", [])
                        if affected and mechanism not in affected:
                            fb = _mechanism_fallback_factor()
                            fallback_penalty = min(fallback_penalty, fb)
                            logger.debug("[CausalPhysics·Propagate] WORLD_ domain filter: %s→%s mechanism=%s not in %s, fb=%.3f",
                                     src, node_id, mechanism, affected, fb)

                    # Apply the (combined) fallback once.
                    if fallback_penalty < 1.0:
                        w *= fallback_penalty

                    if src_data.get("node_type") == "Entity":
                        src_trait = src_data.get("traits", {}).get(trait_name)
                        if isinstance(src_trait, dict) and "value" in src_trait:
                            # Signed delta: shift toward source trait value
                            contrib = (src_trait["value"] - current_val) * w
                        else:
                            contrib = w
                    elif src_data.get("node_type") == "WorldTrait":
                        # World trait: scale impulse by magnitude intensity
                        mag = src_data.get("magnitude", {})
                        mag_value = mag.get("value", 0.5) if isinstance(mag, dict) else 0.5
                        contrib = mag_value * w
                    else:
                        # EventNode or other — fixed impulse from edge weight
                        contrib = w
                    total_impact += contrib
                    total_weight += w
                    per_edge_contributions.append((src, contrib))

                    # Spatial affordance: if the target entity's location is
                    # reachable from the source's location.  We only check when
                    # both sides have locations (entity nodes).
                    if src_data and src_data.get("node_type") == "Entity":
                        src_loc = src_data.get("location_id")
                        tgt_loc = node_data.get("location_id")
                        if src_loc and tgt_loc and src_loc != tgt_loc:
                            if not self._check_spatial_reachability(src_loc, tgt_loc):
                                spatial_ok = False

                # Normalise the accumulated impact into a scale-invariant
                # aggregate. We divide by ``max(1.0, total_weight)`` rather
                # than ``total_weight`` so the gate is *scale-invariant
                # only above unit weight*: a single weak edge with
                # ``total_weight = 0.1`` and ``total_impact = 0.1`` should
                # not be inflated to ``norm_impact = 1.0`` — that would
                # let any single edge clear an arbitrarily high inertia.
                # Above unit weight (multiple stacking sources) the
                # divisor takes over and the aggregate behaves like a
                # weighted average, so ten weak edges no longer dominate
                # one canonical strong source.
                divisor = max(1.0, total_weight)
                norm_impact = total_impact / divisor if divisor else 0.0

                if not spatial_ok:
                    logger.debug("[CausalPhysics·Propagate] BLOCKED spatial: %s.%s impact=%.3f (norm=%.3f)", node_id, trait_name, total_impact, norm_impact)
                    self._blocked.append(BlockedPropagation(
                        node_id=node_id, trait=trait_name,
                        impact=norm_impact, inertia=trait_inertia,
                        reason="spatial_affordance",
                    ))
                    continue

                # ----------------------------------------------------------
                # Gating: deterministic (legacy) vs noisy-OR (probabilistic).
                # The noisy-OR path treats each contributing edge as an
                # independent Bernoulli attempt to overcome inertia, then
                # ORs them. Under default settings the legacy gate is
                # preserved bit-for-bit.
                # ----------------------------------------------------------
                settings = _physics_settings()
                if settings.propagation_mode == "noisy_or" and per_edge_contributions:
                    per_edge_probs: List[Dict[str, Any]] = []
                    raw_probs: List[float] = []
                    for src_id, contrib in per_edge_contributions:
                        p_i = _noisy_or_per_edge_probability(
                            weighted_impulse=contrib,
                            inertia=trait_inertia,
                            temperature=settings.noisy_or_temperature,
                        )
                        per_edge_probs.append({
                            "source_id": src_id,
                            "p": p_i,
                            "weighted_impulse": contrib,
                        })
                        raw_probs.append(p_i)
                    aggregate = _noisy_or_aggregate(raw_probs)

                    if self._sample_noisy_or and self._rng is not None:
                        fired = self._rng.random() < aggregate
                    else:
                        fired = aggregate >= settings.noisy_or_threshold

                    self._noisy_or_records.append(NoisyOrProbability(
                        node_id=node_id, trait=trait_name,
                        inertia=trait_inertia,
                        per_edge=per_edge_probs,
                        aggregate_probability=aggregate,
                        fired=fired,
                    ))
                    if not fired:
                        if norm_impact == 0.0:
                            continue
                        logger.debug(
                            "[CausalPhysics·Propagate] NOISY-OR ABSORBED %s.%s "
                            "p_agg=%.3f thr=%.3f inertia=%.3f",
                            node_id, trait_name, aggregate,
                            settings.noisy_or_threshold, trait_inertia,
                        )
                        self._blocked.append(BlockedPropagation(
                            node_id=node_id, trait=trait_name,
                            impact=norm_impact, inertia=trait_inertia,
                            reason="noisy_or_absorbed",
                        ))
                        continue

                    # Fired: scale shift by aggregate confidence so a
                    # marginal noisy-OR fire (p just over threshold) moves
                    # the trait less than a saturated one (p ~= 1.0).
                    sign = 1 if norm_impact > 0 else -1 if norm_impact < 0 else 0
                    if sign == 0:
                        continue
                    effective_shift = sign * abs(norm_impact) * aggregate
                    new_val = max(0.0, min(1.0, current_val + effective_shift))
                    logger.debug(
                        "[CausalPhysics·Propagate] NOISY-OR MUTATED %s.%s: "
                        "%.3f→%.3f (p_agg=%.3f, shift=%.3f, raw=%.3f)",
                        node_id, trait_name, current_val, new_val,
                        aggregate, effective_shift, total_impact,
                    )
                    self._mutations.append(TraitMutation(
                        node_id=node_id, trait=trait_name,
                        old_value=current_val, new_value=new_val,
                        impact=norm_impact, inertia=trait_inertia,
                    ))
                    trait_data["value"] = new_val
                    self._active_sources.add(node_id)
                    continue

                if abs(norm_impact) <= trait_inertia + _inertia_epsilon():
                    if norm_impact == 0.0:
                        # No active source contributed any impulse this step —
                        # not a block, just nothing happened. Skip silently.
                        continue
                    logger.debug("[CausalPhysics·Propagate] BLOCKED inertia: %s.%s |norm_impact|=%.3f <= inertia=%.3f (raw=%.3f, w=%.3f)",
                                 node_id, trait_name, abs(norm_impact), trait_inertia, total_impact, total_weight)
                    self._blocked.append(BlockedPropagation(
                        node_id=node_id, trait=trait_name,
                        impact=norm_impact, inertia=trait_inertia,
                        reason="inertia",
                    ))
                    continue

                # Dampened shift on the normalised impact (same dampening
                # formula as before, but in the scale-invariant space).
                sign = 1 if norm_impact > 0 else -1
                effective_shift = norm_impact - sign * trait_inertia
                new_val = max(0.0, min(1.0, current_val + effective_shift))
                logger.debug("[CausalPhysics·Propagate] MUTATED %s.%s: %.3f→%.3f (norm_impact=%.3f, inertia=%.3f, shift=%.3f, raw=%.3f, w=%.3f)",
                             node_id, trait_name, current_val, new_val, norm_impact, trait_inertia, effective_shift, total_impact, total_weight)

                self._mutations.append(TraitMutation(
                    node_id=node_id, trait=trait_name,
                    old_value=current_val, new_value=new_val,
                    impact=norm_impact, inertia=trait_inertia,
                ))
                trait_data["value"] = new_val
                # Cascade: this target is now an active source for any
                # downstream edges processed later in topo order.
                self._active_sources.add(node_id)

        # 4. Baseline drift — pull every mutated trait back toward its
        #    captured baseline by ``(1 - inertia) * rate``. High-inertia
        #    traits resist drift (they keep most of the propagated shift);
        #    low-inertia traits snap back almost entirely. Default rate is
        #    0 so legacy behaviour is preserved bit-for-bit.
        drift_rate = _physics_settings().entity_trait_baseline_drift_rate
        if drift_rate > 0 and trait_baselines and self._mutations:
            for mut in self._mutations:
                baseline = trait_baselines.get(mut.node_id, {}).get(mut.trait)
                if baseline is None:
                    continue
                node_data = self.sandbox.nodes.get(mut.node_id)
                if not node_data:
                    continue
                tdata = (node_data.get("traits") or {}).get(mut.trait)
                if not isinstance(tdata, dict) or "value" not in tdata:
                    continue
                current = tdata["value"]
                inertia = mut.inertia
                pull = (baseline - current) * max(0.0, 1.0 - inertia) * drift_rate
                if pull == 0.0:
                    continue
                drifted = max(0.0, min(1.0, current + pull))
                logger.debug(
                    "[CausalPhysics·Drift] %s.%s: %.3f → %.3f (baseline=%.3f, inertia=%.3f, rate=%.3f)",
                    mut.node_id, mut.trait, current, drifted,
                    baseline, inertia, drift_rate,
                )
                tdata["value"] = drifted
                # Reflect the post-drift value in the mutation record so
                # callers see the engine's final answer, not the
                # intermediate pre-drift number.
                mut.new_value = drifted

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

        # Cycle detection on the social-causal subgraph, mirroring
        # ``propagate()``. A ``mutation_social`` edge connects an event
        # source to a perspective entity (``target_id``) whose
        # relationship metric toward ``rel_counterpart_id`` is being
        # mutated. If those edges form an SCC the deltas would feed
        # back into themselves \u2014 the engine cannot decide a valid
        # firing order, so members of the cycle are skipped and a
        # ``BlockedPropagation(reason="cycle")`` entry is emitted for
        # each blocked metric, matching the reporting contract that
        # ``propagate()`` already establishes for trait cycles.
        social_subgraph = nx.DiGraph()
        for u, v, d in self.sandbox.edges(data=True):
            if d.get("edge_type") != "causal":
                continue
            if d.get("causality_type") != "mutation_social":
                continue
            tgt = d.get("target_id", v)
            counterpart = d.get("rel_counterpart_id")
            if not counterpart:
                continue
            # Edge from the perspective entity (whose relationship is
            # mutated) toward its counterpart \u2014 a cycle here means
            # mutual social mutations form a feedback loop.
            social_subgraph.add_edge(tgt, counterpart, source=u,
                                     metric=d.get("trait_target"))
        social_cyclic_blocked: set[str] = set()
        if social_subgraph.number_of_edges() > 0:
            try:
                nx.topological_sort(social_subgraph)
            except nx.NetworkXUnfeasible:
                for scc in nx.strongly_connected_components(social_subgraph):
                    if len(scc) > 1:
                        social_cyclic_blocked |= scc
                logger.warning(
                    "[CausalPhysics\u00b7SocialProp] Cyclic social-causal "
                    "subgraph: %d node(s) in non-trivial SCC(s). Members "
                    "are blocked from social propagation.",
                    len(social_cyclic_blocked),
                )

        # Snapshot the edge list before iterating: ``add_edge`` calls below
        # mutate the MultiDiGraph (creating a new relationship edge when
        # none exists) and iterating a live view is implementation-defined.
        edges_snapshot = list(self.sandbox.edges(data=True))
        for u, v, d in edges_snapshot:
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

            # Refuse to fire any social mutation whose perspective
            # entity or counterpart sits inside a cyclic SCC of the
            # social-causal subgraph (mirrors propagate() for trait
            # cycles). Record an explicit ``cycle`` block so the
            # auditor sees why the metric didn't move.
            if target_id in social_cyclic_blocked or counterpart_id in social_cyclic_blocked:
                logger.debug("[CausalPhysics·SocialProp] Cycle-blocked: %s→%s %s",
                             target_id, counterpart_id, metric)
                self._blocked.append(BlockedPropagation(
                    node_id=target_id,
                    trait=f"rel.{counterpart_id}.{metric}",
                    impact=raw_delta,
                    inertia=_relationship_inertia_default(),
                    reason="cycle",
                ))
                continue

            # Scale delta by evidence_strength × causal_force
            evidence_w = _strength_weight(d.get("evidence_strength", "moderate"))
            force_scale = _force_scale(d.get("causal_force", _physics_settings().default_causal_force))
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
        (or affordance-unlockable) connected_to edges in the sandbox.

        The traversable subgraph is invariant for the duration of a single
        ``execute()`` call (sandbox edges aren't mutated by propagation),
        so we build it lazily once and cache it on the engine instead of
        rebuilding it for every (src, tgt) pair.
        """
        traversable = self._spatial_traversable
        if traversable is None:
            traversable = self._build_spatial_traversable()
            self._spatial_traversable = traversable
        if not traversable.has_node(src_loc) or not traversable.has_node(tgt_loc):
            logger.debug("[CausalPhysics·Spatial] %s or %s not in traversable graph", src_loc, tgt_loc)
            return False
        reachable = nx.has_path(traversable, src_loc, tgt_loc)
        logger.debug("[CausalPhysics·Spatial] %s→%s reachable=%s", src_loc, tgt_loc, reachable)
        return reachable

    def _build_spatial_traversable(self) -> nx.DiGraph:
        """Build the cached traversable connected_to subgraph for this run.

        Iterates the sandbox edges once. Locked edges are admitted only when
        an entity in the sandbox owns an item with an ``unlock`` affordance
        whose ``target_type`` matches the barrier's node-type or name.
        """
        traversable = nx.DiGraph()
        for u, v, d in self.sandbox.edges(data=True):
            if d.get("edge_type") != "connected_to":
                continue
            if not d.get("is_locked", False):
                traversable.add_edge(u, v)
                continue
            barrier_id = d.get("barrier_item_id")
            if not barrier_id:
                continue
            barrier_node = self.sandbox.nodes.get(barrier_id, {})
            barrier_name = barrier_node.get("name", "")
            barrier_node_type = barrier_node.get("node_type", "NarrativeObject")
            for nid, ndata in self.sandbox.nodes(data=True):
                if ndata.get("node_type") != "NarrativeObject":
                    continue
                if ndata.get("owner_id") is None:
                    continue
                for aff in ndata.get("affordances", []):
                    if not isinstance(aff, dict):
                        continue
                    if aff.get("action") != "unlock":
                        continue
                    aff_target = aff.get("target_type", "")
                    if aff_target == barrier_node_type or aff_target == barrier_name:
                        traversable.add_edge(u, v)
                        break
                else:
                    continue
                break
        return traversable

    # ------------------------------------------------------------------
    # ctf-calculus pre-flight (Correa & Bareinboim, ICML 2025)
    # ------------------------------------------------------------------
    def _apply_ctf_calculus_preflight(
        self,
        *,
        rung: int,
        interventions: Dict[str, Any],
        evidence_node_ids: List[str],
        target_node_ids: List[str],
        diagram: Optional[nx.DiGraph] = None,
    ) -> CtfCalculusReport:
        """Run the static AMWN d-separation checks before simulation.

        Rule 3 (Exclusion) prunes intervention keys whose target node has
        no directed path to any *query target* in the mutilated diagram.
        Rule 2 (Independence) flags evidence nodes that are d-separated
        from every intervened variable on the AMWN given W*.

        The caller (execute) consumes the report to skip pruned
        interventions and redundant evidence — the heuristic propagation
        layer still runs on whatever survives.
        """
        if not interventions:
            return CtfCalculusReport()
        try:
            return apply_ctf_calculus(
                self.world_state,
                interventions=interventions,
                evidence_node_ids=evidence_node_ids,
                target_node_ids=target_node_ids or None,
                diagram=diagram,
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
        target_node_ids: List[str] | None = None,
        *,
        causal_diagram: Optional[nx.DiGraph] = None,
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
        target_node_ids : list[str], optional
            Downstream nodes the caller cares about. Used as the Y-set for
            the Rule 3 (Exclusion) ctf-calculus pre-flight: an intervention
            is provably vacuous if it has no directed path to any of these
            nodes in the mutilated diagram.
        causal_diagram : nx.DiGraph, optional
            Pre-built static causal diagram (output of ``build_causal_diagram``).
            Hoist this out of tight loops (e.g. candidate evaluation) where the
            world state is constant across many ``execute()`` calls.
        """
        if rung not in (2, 3):
            raise ValueError(f"Invalid rung={rung}. Must be 2 (intervention) or 3 (counterfactual).")

        # Auto-route to the Monte-Carlo orchestrator when sampling is
        # configured *and* this call isn't itself a sub-sample. Setting
        # ``CausalPhysicsSettings.monte_carlo_samples > 0`` makes
        # ``execute_distribution`` the effective engine for every call
        # site (narrative_physics, directive_assembly, MCP server, …)
        # without each caller having to know about the alternate entry
        # point. The flag ``_in_mc_sample`` is set on the per-sample
        # sub-engines inside ``execute_distribution`` to short-circuit
        # this redirect and prevent infinite recursion.
        if (
            not getattr(self, "_in_mc_sample", False)
            and _physics_settings().monte_carlo_samples > 0
        ):
            return self.execute_distribution(
                rung,
                interventions=interventions,
                evidence_node_ids=evidence_node_ids,
                target_node_ids=target_node_ids,
                causal_diagram=causal_diagram,
            )

        # Reset per-run caches. Sandbox edges aren't mutated by simulation,
        # so the spatial traversable graph is built once and reused for
        # every reachability check inside this execute() call.
        self._spatial_traversable = None

        interventions = interventions or {}
        evidence_node_ids = list(evidence_node_ids or [])
        target_node_ids = list(target_node_ids or [])

        # Step 0 — ctf-calculus pre-flight (Rules 2 & 3).
        # Static graphical reasoning on the AMWN: prune interventions
        # that are *provably* vacuous (Rule 3) and flag evidence that is
        # d-separated from every intervention given W* (Rule 2). The
        # engine drops both before stepping into abduction / surgery so
        # we don't waste cycles on simulation we've proven cannot move
        # the world.
        ctf_report = self._apply_ctf_calculus_preflight(
            rung=rung,
            interventions=interventions,
            evidence_node_ids=evidence_node_ids,
            target_node_ids=target_node_ids,
            diagram=causal_diagram,
        )

        # Filter Rule-3 pruned interventions out of the working set only
        # when the engine is configured to *prune* (the default is
        # *advisory*). Rationale: the AMWN is built from a latent-free
        # SCM (no bidirected confounder arcs). A missed common cause can
        # silently d-separate a real intervention from its query target,
        # so hard-deleting the do-surgery on Rule 3's word risks
        # converting a substantively meaningful intervention into a
        # no-op. In advisory mode we still surface
        # ``rule3_pruned_interventions`` for the auditor / UI but let
        # the heuristic propagation layer make the call. Switch to
        # ``rule3_pruning_mode='prune'`` only when the extracted causal
        # topology is known to be confounder-complete.
        # Rule-2 redundant evidence is reported but NOT filtered —
        # abduction may still populate ``hidden_deltas`` that downstream
        # consumers (introspection, the auditor, the UI) depend on.
        rule3_mode = _physics_settings().rule3_pruning_mode
        pruned_set = set(ctf_report.rule3_pruned)
        if pruned_set and rule3_mode == "prune":
            interventions = {
                k: v for k, v in interventions.items() if k not in pruned_set
            }
        elif pruned_set:
            logger.info(
                "[CausalPhysics\u00b7Rule3] %d intervention(s) flagged as "
                "vacuous by Rule 3 but kept (advisory mode): %s",
                len(pruned_set), sorted(pruned_set),
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
            rule3_pruning_mode=rule3_mode,
            rule2_redundant_evidence=ctf_report.rule2_redundant_evidence,
            noisy_or_probabilities=self._noisy_or_records,
        )
        _log_physics_result(rung, interventions, evidence_node_ids, result)
        return result

    # ------------------------------------------------------------------
    # Monte-Carlo distributional CTF
    # ------------------------------------------------------------------
    def execute_distribution(
        self,
        rung: int,
        interventions: Dict[str, Any] | None = None,
        evidence_node_ids: List[str] | None = None,
        target_node_ids: List[str] | None = None,
        *,
        causal_diagram: Optional[nx.DiGraph] = None,
        samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> CausalPhysicsResult:
        """Run ``execute()`` repeatedly under perturbed inputs.

        Each sample:
          * Restores the sandbox from a deep copy of the original.
          * Perturbs every causal edge's ``causal_force`` ~ Normal(force,
            sigma(evidence_strength)) so the per-edge weight becomes a
            random draw rather than the LLM's point estimate.
          * Perturbs every entity trait initial value ~ Beta(alpha, beta)
            with concentration ``kappa = 1/(1 - inertia + eps)`` so high-
            inertia traits stay tight and low-inertia ones diffuse.
          * Switches the noisy-OR gate from threshold mode to Bernoulli-
            sampling mode if ``propagation_mode == 'noisy_or'``.
          * Runs the standard ``execute()`` pipeline and records each
            entity-trait's post-propagation value.

        Returns a ``CausalPhysicsResult`` whose ``trait_distributions`` map
        is ``node_id -> trait_name -> TraitDistribution`` (mean/std/p5/p50/
        p95/samples_count). ``sandbox_data`` is taken from the last sample
        so the auditor still has a representative graph to inspect.
        """
        s = _physics_settings()
        n = samples if samples is not None else s.monte_carlo_samples
        if n <= 0:
            # Nothing to sample — fall back to a single deterministic run.
            return self.execute(
                rung,
                interventions=interventions,
                evidence_node_ids=evidence_node_ids,
                target_node_ids=target_node_ids,
                causal_diagram=causal_diagram,
            )

        rng_seed = seed if seed is not None else s.monte_carlo_seed
        rng = random.Random(rng_seed)

        # Snapshot the original sandbox so each sample starts from a
        # pristine copy. The original is restored at the end so callers
        # observing ``self.sandbox`` see no side-effects from sampling.
        original_sandbox = copy.deepcopy(self.sandbox)
        original_world_state = self.world_state

        # Collected: node_id -> trait_name -> [values...]
        collected: Dict[str, Dict[str, List[float]]] = {}
        last_result: Optional[CausalPhysicsResult] = None

        for i in range(n):
            sample_sandbox = copy.deepcopy(original_sandbox)

            # Perturb causal_force on every causal edge.
            for _u, _v, edata in sample_sandbox.edges(data=True):
                if edata.get("edge_type") != "causal":
                    continue
                nominal = edata.get("causal_force", s.default_causal_force)
                ev_label = edata.get("evidence_strength", "moderate")
                edata["causal_force"] = _sample_causal_force(nominal, ev_label, rng)

            # Perturb entity trait initial values.
            for _nid, ndata in sample_sandbox.nodes(data=True):
                if ndata.get("node_type") != "Entity":
                    continue
                traits = ndata.get("traits") or {}
                for _tname, tdata in traits.items():
                    if not isinstance(tdata, dict) or "value" not in tdata:
                        continue
                    inertia = tdata.get("inertia", 0.5)
                    tdata["value"] = _sample_trait_value(
                        tdata["value"], inertia, rng,
                    )

            # Build a fresh sub-engine bound to this sample's sandbox so
            # per-run state (mutations, blocked, hidden_deltas) is isolated.
            sub = CausalPhysicsEngine(sample_sandbox, original_world_state)
            sub._rng = rng
            # Mark this engine as already executing inside the Monte-Carlo
            # loop so its ``execute()`` call does not re-enter
            # ``execute_distribution`` and recurse forever.
            sub._in_mc_sample = True
            sub._sample_noisy_or = True
            try:
                last_result = sub.execute(
                    rung,
                    interventions=interventions,
                    evidence_node_ids=evidence_node_ids,
                    target_node_ids=target_node_ids,
                    causal_diagram=causal_diagram,
                )
            except Exception:
                logger.exception(
                    "[CausalPhysics·MC] Sample %d/%d failed; skipping.",
                    i + 1, n,
                )
                continue

            # Collect post-propagation trait values from this sample.
            for nid, ndata in sample_sandbox.nodes(data=True):
                if ndata.get("node_type") != "Entity":
                    continue
                traits = ndata.get("traits") or {}
                for tname, tdata in traits.items():
                    if not isinstance(tdata, dict) or "value" not in tdata:
                        continue
                    collected.setdefault(nid, {}).setdefault(tname, []).append(
                        float(tdata["value"])
                    )

        # Restore the original sandbox so the engine is reusable.
        self.sandbox = original_sandbox

        # Aggregate distributions.
        distributions: Dict[str, Dict[str, TraitDistribution]] = {}
        for nid, by_trait in collected.items():
            for tname, values in by_trait.items():
                if not values:
                    continue
                sorted_vals = sorted(values)
                k = len(sorted_vals)

                def _quantile(q: float) -> float:
                    if k == 1:
                        return sorted_vals[0]
                    pos = q * (k - 1)
                    lo = int(math.floor(pos))
                    hi = int(math.ceil(pos))
                    if lo == hi:
                        return sorted_vals[lo]
                    frac = pos - lo
                    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

                mean = statistics.fmean(values)
                std = statistics.pstdev(values) if k > 1 else 0.0
                distributions.setdefault(nid, {})[tname] = TraitDistribution(
                    mean=mean,
                    std=std,
                    p5=_quantile(0.05),
                    p50=_quantile(0.50),
                    p95=_quantile(0.95),
                    samples_count=k,
                )

        if last_result is None:
            # All samples failed — return an empty result rather than crash.
            return CausalPhysicsResult(
                sandbox_data=nx.node_link_data(self.sandbox),
                trait_distributions=distributions,
            )

        last_result.trait_distributions = distributions
        return last_result



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


# Resolve forward references in CausalPhysicsResult — the typed fields
# noisy_or_probabilities and trait_distributions reference NoisyOrProbability
# and TraitDistribution as strings (combined with `from __future__ import
# annotations`), so an explicit rebuild ensures pydantic v2 wires them up
# eagerly at import time rather than lazily on first instantiation.
CausalPhysicsResult.model_rebuild()
