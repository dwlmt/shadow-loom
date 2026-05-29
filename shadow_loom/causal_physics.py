# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

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

import contextvars
import copy
import logging
import math
import random
import statistics
from typing import Any, Dict, List, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field

from shadow_loom.amwn import CtfCalculusReport, apply_ctf_calculus
from shadow_loom.causal_closure import (
    chain_reaction_parents_from_sandbox,
    expand_chain_reaction_closure,
)
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import (
    WorldStateV1,
    default_relationship_metrics_dict,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
)
from shadow_loom.settings import get_settings as _get_settings

logger = logging.getLogger(__name__)

# Context flag: True while a Monte-Carlo sub-sample is running. Per-sample
# physics steps (abduction, do-surgery, propagate, social, result) demote
# their INFO logs to DEBUG so a 128-sample sweep does not spam 128×N
# banner blocks at INFO level. ``execute_distribution`` emits a single
# INFO summary at the end instead. Used both inside the engine and by
# ``shadow_loom.instantiator.AMWNInstantiator._intervene_state`` (a
# staticmethod called during do-surgery), which is why this is a
# module-level contextvar rather than an instance attribute.
_in_mc_sample_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "shadow_loom_causal_physics_in_mc_sample", default=False,
)


def is_in_mc_sample() -> bool:
    """Public read-only check for callers outside this module."""
    return _in_mc_sample_var.get()


def _physics_log(level_when_normal: int = logging.INFO) -> int:
    """Return the log level to use for per-step physics chatter.

    DEBUG inside a Monte-Carlo sample, ``level_when_normal`` otherwise.
    """
    if _in_mc_sample_var.get():
        return logging.DEBUG
    return level_when_normal


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


class PropositionMutation(BaseModel):
    """Record of a Proposition truth-clamp applied via Pearl Rung-2 surgery.

    Mirrors :class:`TraitMutation` for the audience-side utility layer.
    Cascaded ``BeliefMutation`` records (when ``propagate_to_beliefs`` is
    True) are recorded separately so the auditor can attribute downstream
    epistemic shifts to a specific proposition clamp.
    """
    proposition_id: str
    fabula_time: int
    old_truth: Optional[bool] = None
    new_truth: bool
    cascaded_belief_count: int = 0


class BeliefMutation(BaseModel):
    """Record of a Belief confidence clamp on a single character."""
    holder_id: str
    target_id: str
    proposition_id: Optional[str] = None
    old_confidence: Optional[float] = None
    new_confidence: float = Field(ge=0.0, le=1.0, description="Must be valid probability [0,1]")
    created: bool = False  # True when the belief did not previously exist
    triggered_by: str = "DO_OPERATOR"  # or PROP_X when cascaded


class ConcernMutation(BaseModel):
    """Record of a Concern clamp on a single character (utility layer)."""
    holder_id: str
    concern_id: str
    field: str  # 'salience' | 'polarity' | 'active'
    old_value: Any = None
    new_value: Any = None


class WorldTraitMutation(BaseModel):
    """Record of a WORLD_ ``GlobalTrait`` magnitude clamp applied via
    Pearl Rung-2 surgery.

    Mirrors :class:`PropositionMutation` for the global ambient-force
    layer. The pipeline adapter reads these and emits one
    :class:`WorldTraitSnapshot` per mutation, honouring the per-chunk
    merge fold's inertia attenuation so the timeline writes are
    consistent with per-chunk ``WorldTraitUpdate`` flows.
    """
    world_trait_id: str
    fabula_time: int
    old_value: Optional[float] = None
    new_value: float
    inertia: Optional[float] = None
    affected_domains_add: List[str] = Field(default_factory=list)
    affected_domains_remove: List[str] = Field(default_factory=list)
    triggered_by: Optional[str] = None


class ObjectMutation(BaseModel):
    """Record of a :class:`NarrativeObject` clamp applied via Pearl Rung-2
    surgery.

    Mirrors :class:`WorldTraitMutation` for the prop layer. The pipeline
    adapter reads these and emits one
    :class:`~shadow_loom.ingestion.ObjectUpdate` per mutation onto the
    chunk topology, which in turn folds into a single
    :class:`ObjectStateSnapshot` on the canonical
    :attr:`NarrativeObject.state_timeline` during the merge step.
    """
    object_id: str
    fabula_time: int
    new_location_id: Optional[str] = None
    new_owner_id: Optional[str] = None
    set_location_null: bool = False
    set_owner_null: bool = False
    properties_set: Dict[str, str] = Field(default_factory=dict)
    properties_unset: List[str] = Field(default_factory=list)
    triggered_by: Optional[str] = None


class EdgeMutation(BaseModel):
    """Record of a topological-edge surgery applied via Pearl Rung-2.

    Covers ``DoCausalEdge`` / ``DoSpatialEdge`` / ``DoChannel`` — three
    surgery families that rewrite edges (or edge-like channel attrs)
    rather than node-level magnitudes. Previously these surgeries only
    bumped ``_edge_do_targets_applied`` and mutated the sandbox/world
    silently; callers had no structured per-edge diff to render or
    audit. The collector lets the renderer say "severed PASSAGE_X→Y"
    instead of merely "intervention_inert=False, mutations=[]".

    ``edge_type`` namespaces the action vocabulary:

      * ``causal``   — ``action`` ∈ {"add", "sever"}
      * ``spatial``  — ``action`` ∈ {"add", "sever", "lock", "unlock"}
      * ``channel``  — ``action`` ∈ {"activate", "deactivate", "retune"}

    ``details`` carries per-action context (e.g. ``causality_type``
    filter on a sever, ``barrier_item_id`` on a lock, removed-edge
    count, intelligibility-axis diff) without forcing a discriminated
    union per family.
    """
    edge_type: Literal["causal", "spatial", "channel"]
    action: str
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    channel_id: Optional[str] = None
    fabula_time: int
    details: Dict[str, Any] = Field(default_factory=dict)


class EntityDeleteMutation(BaseModel):
    """Record of a :class:`DoEntityDelete` existence counterfactual.

    Pearl Rung-3 surgery excising an Entity ("never existed"). The
    surgery cascades through social topology, beliefs, causal edges,
    and event participation. Previously the cascade landed without a
    structured record, so the renderer / auditor / MCP envelope never
    knew the deletion had happened. Surfacing it lets prose dramatise
    the absence (\"as if Banquo had never lived\") and lets the auditor
    flag prose that mentions the excised entity as a miracle step.
    """
    entity_id: str
    fabula_time: int
    cascaded_social_edges_removed: int = 0
    cascaded_causal_edges_removed: int = 0
    cascaded_beliefs_removed: int = 0
    cascaded_events_scrubbed: int = 0
    triggered_by: Optional[str] = None


class ObjectDeleteMutation(BaseModel):
    """Record of a :class:`DoObjectDelete` existence counterfactual.

    Mirror of :class:`EntityDeleteMutation` for narrative objects.
    """
    object_id: str
    fabula_time: int
    cascaded_causal_edges_removed: int = 0
    cascaded_beliefs_removed: int = 0
    cascaded_events_scrubbed: int = 0
    triggered_by: Optional[str] = None


class EventMutation(BaseModel):
    """Record of a :class:`DoEvent` Pearl Rung-2 / Rung-3 surgery
    (relocation and/or fabula-time shift).

    Mirrors the other typed-mutation records so the renderer / auditor /
    MCP envelope / UI rails can surface event-level edits with the same
    structural fidelity as proposition / belief / object surgeries.

    ``kind`` discriminates between the two DoEvent actions; a single
    DoEvent that BOTH relocates and time-shifts emits two records
    (one per kind) so consumers can describe each effect independently.

      * ``relocation`` — fields: ``old_at_location_id`` / ``new_at_location_id``,
        ``cascaded_actor_snapshots``, ``skipped_dead_actors``.
      * ``time_shift`` — fields: ``old_fabula_time`` / ``new_fabula_time``,
        ``cascaded_snapshot_restamps``, ``cascaded_edge_restamps``.
    """
    event_id: str
    kind: Literal["relocation", "time_shift"]
    fabula_time: int
    old_at_location_id: Optional[str] = None
    new_at_location_id: Optional[str] = None
    old_fabula_time: Optional[int] = None
    new_fabula_time: Optional[int] = None
    cascaded_actor_snapshots: int = 0
    skipped_dead_actors: List[str] = Field(default_factory=list)
    cascaded_snapshot_restamps: int = 0
    cascaded_edge_restamps: int = 0
    triggered_by: Optional[str] = None


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
    # Pearl Rung-2 typed surgeries on the proposition / belief / concern
    # substrate. Populated by :meth:`CausalPhysicsEngine.apply_do_targets`;
    # empty under legacy event/trait-only do-surgery.
    proposition_mutations: List[PropositionMutation] = Field(default_factory=list)
    belief_mutations: List[BeliefMutation] = Field(default_factory=list)
    concern_mutations: List[ConcernMutation] = Field(default_factory=list)
    world_trait_mutations: List[WorldTraitMutation] = Field(
        default_factory=list,
        description=(
            "Pearl Rung-2 magnitude clamps on WORLD_ GlobalTraits. "
            "The pipeline adapter folds each into a "
            "``WorldTraitSnapshot`` on the canonical timeline."
        ),
    )
    object_mutations: List[ObjectMutation] = Field(
        default_factory=list,
        description=(
            "Pearl Rung-2 clamps on OBJ_ NarrativeObjects (location / "
            "owner / properties). The pipeline adapter folds each into "
            "an ``ObjectUpdate`` on the chunk topology, which becomes "
            "an ``ObjectStateSnapshot`` on the canonical timeline."
        ),
    )
    edge_mutations: List[EdgeMutation] = Field(
        default_factory=list,
        description=(
            "Pearl Rung-2 surgeries that rewrite topological edges "
            "(DoCausalEdge / DoSpatialEdge / DoChannel). Lets the "
            "renderer surface per-edge diffs (e.g. \"severed "
            "PASSAGE_X→Y\", \"locked DOOR with KEY_Y\", \"channel "
            "CHN_Z terminated\") instead of relying solely on the "
            "``_edge_do_targets_applied`` counter."
        ),
    )
    entity_delete_mutations: List[EntityDeleteMutation] = Field(
        default_factory=list,
        description=(
            "Pearl Rung-3 entity-excision surgeries (DoEntityDelete). "
            "Each record captures cascade counts for social edges, "
            "causal edges, beliefs, and event participation scrubs."
        ),
    )
    object_delete_mutations: List[ObjectDeleteMutation] = Field(
        default_factory=list,
        description=(
            "Pearl Rung-3 object-excision surgeries (DoObjectDelete). "
            "Mirror of ``entity_delete_mutations`` for props."
        ),
    )
    event_mutations: List[EventMutation] = Field(
        default_factory=list,
        description=(
            "Pearl Rung-2/3 DoEvent surgeries (event relocation and "
            "event time-shift). A single DoEvent that both relocates "
            "and time-shifts emits two records — one per kind."
        ),
    )
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
    pruned_beliefs_count: int = Field(
        default=0,
        description=(
            "Number of beliefs removed from sandbox entities because their "
            "``acquired_via_event_id`` / ``acquired_via_channel_id`` "
            "provenance pointed at an event or channel that the do-surgery "
            "removed. Surfaces the epistemic side-effect of channel / "
            "utterance interventions."
        ),
    )
    pruned_utterance_event_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Utterance event IDs that the do-surgery rendered "
            "epistemically inert (event_type set to 'prevented' or "
            "truth_value coerced to 'false'/'performative'). Used by "
            "narrative_physics and the auditor to know which on-page "
            "utterances must NOT propagate as factual evidence."
        ),
    )
    disabled_channel_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Channel IDs that the do-surgery severed (status='severed' "
            "or all participants removed). Downstream consumers should "
            "treat any utterance routed through these as non-occurring."
        ),
    )
    skipped_interventions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Intervention targets the engine could not apply because the "
            "named node was absent from the sandbox (and not introduced "
            "via ``<ID>.spawn`` or ``query.introduce``). Each entry "
            "carries ``target_path``, ``node_id``, ``property``, "
            "``reason``, and ``detail``. Surfaced into the brief so the "
            "renderer and the auditor know the intervention did NOT "
            "land \u2014 prose must not pretend the requested change took "
            "effect."
        ),
    )
    intervention_inert: bool = Field(
        default=False,
        description=(
            "True when every requested intervention was either Rule-3 "
            "pruned (no path to evidence in the mutilated diagram) or "
            "had all of its trait/social mutations absorbed by cyclic "
            "SCCs / inertia. In this state the do-surgery produced no "
            "observable change downstream. The pipeline MUST surface "
            "this to the brief so prose can disclose the inert outcome "
            "rather than fabricate consequences."
        ),
    )
    intervention_inert_reason: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable explanation of why the intervention was "
            "inert (e.g. 'all do-targets Rule-3 pruned and 4 trait "
            "propagations absorbed by cycle'). Populated alongside "
            "``intervention_inert``."
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
        # Per-axis relationship pins. A path like
        # ``ENT_A.relationships.ENT_B.affinity=-0.9`` adds the triple
        # ``("ENT_A", "ENT_B", "affinity")`` so ``propagate_social`` will
        # not silently overwrite the surgical value with a propagated
        # delta from an inbound ``mutation_social`` edge.
        self._intervened_relationships: set[tuple[str, str, str]] = set()
        self._hidden_deltas: Dict[str, Dict[str, float]] = {}
        self._mutations: List[TraitMutation] = []
        self._social_mutations: List[SocialMutation] = []
        self._blocked: List[BlockedPropagation] = []
        # Phase-1 Pearl Rung-2 typed-surgery collectors. These are appended
        # to by :meth:`apply_do_targets` (and by its proposition cascade
        # into beliefs); they round-trip into ``CausalPhysicsResult``.
        self._proposition_mutations: List[PropositionMutation] = []
        self._belief_mutations: List[BeliefMutation] = []
        self._concern_mutations: List[ConcernMutation] = []
        self._world_trait_mutations: List[WorldTraitMutation] = []
        self._object_mutations: List[ObjectMutation] = []
        # Pearl Rung-2 topological-edge surgeries (DoCausalEdge /
        # DoSpatialEdge / DoChannel) collected for caller-visible
        # rendering. Distinct from ``_edge_do_targets_applied`` which
        # is a flat counter used by the vacuity gate.
        self._edge_mutations: List[EdgeMutation] = []
        # Pearl Rung-3 existence-counterfactual collectors. Populated by
        # ``_apply_do_entity_delete`` / ``_apply_do_object_delete``.
        # Previously these surgeries cascaded silently — the renderer
        # / auditor / MCP envelope only saw a mutated ``world_state``
        # without a structured record.
        self._entity_delete_mutations: List[EntityDeleteMutation] = []
        self._object_delete_mutations: List[ObjectDeleteMutation] = []
        # Pearl Rung-2/3 DoEvent surgeries (relocation + time-shift).
        # Populated by ``_apply_do_event_relocation`` and
        # ``_apply_do_event_time_shift``; round-tripped into
        # ``CausalPhysicsResult.event_mutations``.
        self._event_mutations: List[EventMutation] = []
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
    def abduction_update(
        self,
        evidence_node_ids: List[str],
        *,
        pov_entity_id: Optional[str] = None,
    ) -> None:
        """
        Back-propagate present-day evidence into the historical sandbox.

        For entity evidence: computes per-trait hidden_delta, blends traits
        50 % toward factual values, and back-propagates missing beliefs.

        For event evidence: propagates through causal edges weighted by
        evidence_strength.

        R19-M7: when ``pov_entity_id`` is given, trait /
        relationship blends are skipped for evidence entities that
        share NO recorded event with the POV (no actor / addressee /
        referent overlap). The POV cannot infer hidden traits about
        a character they never witnessed; without this gate the
        abduction loop silently rewrites the historical state of
        off-screen characters using present-day evidence the POV
        had no access to.
        """
        if not evidence_node_ids:
            return

        # R19-M7: precompute the POV's event-overlap entity set.
        pov_overlap_ids: Optional[set] = None
        if pov_entity_id:
            pov_overlap_ids = set()
            for evt in (self.world_state.events or []):
                actors = set(getattr(evt, "actor_ids", None) or [])
                addressees = set(getattr(evt, "addressee_ids", None) or [])
                referents = set(getattr(evt, "referent_ids", None) or [])
                participants = actors | addressees | referents
                if pov_entity_id in participants:
                    pov_overlap_ids |= participants
            pov_overlap_ids.discard(pov_entity_id)

        for eid in evidence_node_ids:
            # Skip WORLD_ nodes — they are structural, not observable evidence.
            if eid.startswith("WORLD_"):
                logger.debug("[CausalPhysics·Abduction] Skipping WORLD_ node: %s", eid)
                continue
            # R19-M7: POV gate \u2014 skip trait / relationship blend
            # entirely for entities the POV never co-appeared with in
            # any event.
            if (
                pov_overlap_ids is not None
                and eid in self.world_state.entities
                and eid not in pov_overlap_ids
            ):
                logger.debug(
                    "[CausalPhysics\u00b7Abduction] Skipping %s: no event "
                    "overlap with POV=%s (R19-M7 gate).",
                    eid, pov_entity_id,
                )
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

                # Per-axis relationship abduction: blend each metric on
                # outgoing relationship edges from the evidence entity
                # toward the factual per-axis value, using
                # ``metrics[axis].evidence_strength`` as the precision
                # of the present-day prior — mirroring the per-trait
                # Bayes blend above. Without this, counterfactual
                # queries that treat measured present-day affinity /
                # fear / power as evidence cannot back-propagate.
                _es_precision = {"weak": 1.0 / 3.0, "moderate": 2.0 / 3.0, "strong": 1.0}
                for fact_rel in self.world_state.social_topology:
                    if fact_rel.source_entity_id != eid:
                        continue
                    other_id = fact_rel.target_entity_id
                    rel_deltas: Dict[str, float] = {}
                    for u, v, key, sandbox_data in self.sandbox.out_edges(eid, data=True, keys=True):
                        if v != other_id or sandbox_data.get("edge_type") != "relationship":
                            continue
                        sandbox_metrics = sandbox_data.get("metrics")
                        if not isinstance(sandbox_metrics, dict):
                            break
                        for axis_name, fact_metric in fact_rel.metrics.items():
                            if not fact_metric.observed:
                                continue
                            axis_state = sandbox_metrics.get(axis_name)
                            if not isinstance(axis_state, dict):
                                continue
                            old_val = float(axis_state.get("value", 0.0))
                            target_val = float(fact_metric.value)
                            ev_w = _es_precision.get(fact_metric.evidence_strength, 0.5)
                            inertia_w = float(axis_state.get("inertia", 0.3))
                            if blend_mode == "bayesian":
                                denom = inertia_w + ev_w
                                blended = (
                                    inertia_w * old_val + ev_w * target_val
                                ) / denom if denom > 0 else old_val
                            else:
                                blend_factor = max(0.0, min(1.0, 1.0 - inertia_w)) * ev_w
                                blended = old_val + (target_val - old_val) * blend_factor
                            if axis_name == "fear":
                                blended = max(0.0, min(1.0, blended))
                            else:
                                blended = max(-1.0, min(1.0, blended))
                            axis_state["value"] = blended
                            axis_state["observed"] = True
                            # Mirror to the flat aggregate so legacy
                            # readers stay consistent with the per-axis
                            # state post-blend.
                            sandbox_data[axis_name] = blended
                            rel_deltas[f"rel.{other_id}.{axis_name}"] = blended - old_val
                        break
                    if rel_deltas:
                        existing_deltas = self._hidden_deltas.setdefault(eid, {})
                        existing_deltas.update(rel_deltas)
                        logger.log(
                            _physics_log(),
                            "[CausalPhysics·Abduction·Rel] %s→%s deltas: %s",
                            eid, other_id, rel_deltas,
                        )

                # Back-propagate beliefs.
                #
                # Intelligibility guard: a belief whose provenance is a
                # channel where the holder cannot understand the medium
                # (per-recipient intelligibility < threshold) is not
                # epistemically valid evidence — the holder shouldn't
                # have learned it through that channel. We skip those
                # so abduction doesn't reinstate beliefs that the
                # current world topology says could not have been heard.
                if "beliefs" not in node_data:
                    node_data["beliefs"] = []
                existing = node_data["beliefs"]
                intel_thresh = _physics_settings().intelligibility_threshold
                for belief in target_beliefs:
                    b_dict = belief if isinstance(belief, dict) else belief.model_dump()
                    ch_id = b_dict.get("acquired_via_channel_id")
                    if ch_id:
                        ch = self.world_state.channels.get(ch_id)
                        if ch is not None:
                            intel_val = ch.intelligibility.get(eid, 1.0)
                            if float(intel_val) < intel_thresh:
                                logger.debug(
                                    "[CausalPhysics·Abduction·Belief] Skipping belief "
                                    "for %s acquired via low-intelligibility "
                                    "channel %s (intel=%.2f < %.2f).",
                                    eid, ch_id, intel_val, intel_thresh,
                                )
                                continue
                    b_target_id = b_dict.get("target_id")
                    b_state = b_dict.get("perceived_state")
                    if not any(
                        b.get("target_id") == b_target_id
                        and b.get("perceived_state") == b_state
                        for b in existing
                    ):
                        existing.append(b_dict)

                logger.log(
                    _physics_log(),
                    "[CausalPhysics·Abduction] Conditioned entity %s (deltas: %s).",
                    eid, deltas,
                )

            # Case 2 — Evidence is an Event
            elif self.sandbox.has_node(eid):
                node_data = self.sandbox.nodes[eid]
                if node_data.get("node_type") == "EventNode":
                    # P0-FIX (P0-5): Separate abduction from forward propagation (VIOLATION #6 audit).
                    # Pearl's abduction step (rung-3 counterfactual) should ONLY update exogenous
                    # variables (U terms), not propagate effects forward through causal edges.
                    # The original implementation propagated event evidence through the graph during
                    # abduction, violating temporal consistency (back-propagating future events).
                    #
                    # Correct Pearl abduction recipe (Causality 2009, §7):
                    #   1. Abduction: Update P(U|evidence) - infer hidden exogenous variables
                    #   2. Action: Apply do-operator (graph surgery)
                    #   3. Prediction: Forward propagate through modified graph
                    #
                    # This fix implements step 1 correctly: mark the event as evidence and update
                    # exogenous noise on affected entities, but do NOT forward-propagate during
                    # abduction. The propagate() method handles forward effects in step 3.
                    
                    # Truth-value guard for utterance events. An utterance whose truth_value is
                    # false or performative does NOT produce factual belief reinforcement.
                    if (
                        node_data.get("event_type") == "utterance"
                        and node_data.get("truth_value") in ("false", "performative")
                    ):
                        logger.log(
                            _physics_log(),
                            "[CausalPhysics·Abduction] Skipping event %s (truth_value=%s).",
                            eid, node_data.get("truth_value"),
                        )
                        self._abducted_event_evidence.add(eid)
                        continue
                    
                    # Mark this event as abducted so propagate() can decide whether to
                    # fire its outgoing edges. The original code marked it then immediately
                    # propagated, causing double-counting. Now we only mark it - propagate()
                    # will skip edges from abducted events to prevent double-application.
                    self._abducted_event_evidence.add(eid)
                    
                    # Update exogenous noise on affected entities based on event evidence.
                    # This implements Pearl's abduction: inferring U given evidence e.
                    # For each entity that this event causally affects, we update the
                    # exogenous_noise term to reflect the hidden variance that best
                    # explains the observed outcome.
                    for ce in self.world_state.causal_topology:
                        if ce.source_id != eid:
                            continue
                        
                        target_node = self.sandbox.nodes.get(ce.target_id)
                        if target_node and target_node.get("node_type") == "Entity":
                            traits = target_node.get("traits", {})
                            
                            # Update exogenous noise for the affected trait(s).
                            # High causal_force + strong evidence_strength suggests
                            # low unexplained variance (exogenous noise).
                            mult = _strength_weight(ce.evidence_strength)
                            force_scale = _force_scale(ce.causal_force)
                            explained_variance = mult * force_scale
                            unexplained_variance = max(0.0, 1.0 - explained_variance)
                            
                            if ce.causality_type == "mutation" and ce.trait_target:
                                td = traits.get(ce.trait_target)
                                if isinstance(td, dict):
                                    # Update exogenous noise (U_X) for this trait
                                    old_noise = td.get("exogenous_noise", 0.0)
                                    # Blend toward unexplained variance
                                    td["exogenous_noise"] = (old_noise + unexplained_variance) / 2.0
                                    logger.debug(
                                        "[CausalPhysics·Abduction·Exogenous] Event %s → %s.%s: "
                                        "updated exogenous_noise %.3f→%.3f (explained=%.3f)",
                                        eid, ce.target_id, ce.trait_target,
                                        old_noise, td["exogenous_noise"], explained_variance
                                    )
                            else:
                                # Update exogenous noise for all mechanism-relevant traits
                                relevant = MECHANISM_TRAIT_MAP.get(ce.mechanism, None)
                                for trait_name, trait_data in traits.items():
                                    if not isinstance(trait_data, dict):
                                        continue
                                    if relevant is None or trait_name in relevant:
                                        old_noise = trait_data.get("exogenous_noise", 0.0)
                                        trait_data["exogenous_noise"] = (old_noise + unexplained_variance) / 2.0
                    
                    logger.log(
                        _physics_log(),
                        "[CausalPhysics·Abduction] Marked event %s as abducted (exogenous vars updated).",
                        eid,
                    )
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
        # Phase 5b audit (Issue C) — snapshot the *pre-mutation* value of
        # every directly-clamped trait BEFORE we hand off to the
        # instantiator so we can emit a structured ``TraitMutation`` for
        # each direct clamp after the surgery lands. Previously, direct
        # trait clamps pinned ``_intervened_traits`` but never produced
        # a mutation row — when downstream propagation was blocked
        # (cyclic SCC, inertia), the result surfaced ``mutations=[]``
        # and looked inert despite the trait actually moving.
        _direct_trait_snapshot: List[tuple] = []
        for target_path, new_value in interventions.items():
            if "." not in target_path:
                continue
            node_id, sub_path = target_path.split(".", 1)
            sub = sub_path.strip()
            if not sub.startswith("traits."):
                continue
            parts = sub.split(".")
            if len(parts) < 2 or not parts[1]:
                continue
            trait_name = parts[1]
            if not self.sandbox.has_node(node_id):
                continue
            try:
                _new_f = float(new_value)
            except (TypeError, ValueError):
                continue
            traits = (self.sandbox.nodes[node_id].get("traits") or {})
            cur = traits.get(trait_name) or {}
            try:
                _old_f = float(cur.get("value")) if isinstance(cur, dict) and cur.get("value") is not None else None
            except (TypeError, ValueError):
                _old_f = None
            _direct_trait_snapshot.append((node_id, trait_name, _old_f, _new_f))

        AMWNInstantiator.execute_interventions(self.sandbox, interventions)

        # Emit one TraitMutation per direct clamp. ``inertia=0`` and
        # ``impact=0`` reflect the surgical bypass (we did not consume
        # any propagation budget); the observable diff lives in
        # ``old_value`` / ``new_value``.
        for node_id, trait_name, _old_f, _new_f in _direct_trait_snapshot:
            # Skip no-op clamps so the rendered diff list stays clean.
            if _old_f is not None and _old_f == _new_f:
                continue
            self._mutations.append(TraitMutation(
                node_id=node_id,
                trait=trait_name,
                old_value=(_old_f if _old_f is not None else float("nan")),
                new_value=_new_f,
                impact=0.0,
                inertia=0.0,
            ))

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
                # P0-FIX: Remove wildcard trait pinning (CRITICAL-004 audit).
                # Pearl's minimal surgery principle: only directly intervened
                # variables should be frozen. Bare-node or spawn interventions
                # (e.g., do(ENT_X.status="alive")) should NOT freeze all traits.
                # If the user explicitly pinned specific traits, honor those.
                pinned = per_trait_pins_by_node.get(node_id)
                if pinned:
                    for trait_name in pinned:
                        self._intervened_traits.add((node_id, trait_name))
                # No wildcard fallback - only pin what was explicitly set
            elif sub.startswith("traits."):
                # ``traits.<name>`` or ``traits.<name>.value`` \u2014 pin the
                # specific trait only.
                parts = sub.split(".")
                trait_name = parts[1] if len(parts) >= 2 else ""
                if trait_name:
                    self._intervened_traits.add((node_id, trait_name))
            elif sub.startswith("relationships."):
                # ``relationships.<other>.<metric>`` — pin the per-axis
                # relationship so propagate_social does not subsequently
                # overwrite the surgical value with a propagated delta.
                parts = sub.split(".")
                if len(parts) >= 3:
                    other_id = parts[1]
                    metric_name = parts[2]
                    if other_id and metric_name:
                        self._intervened_relationships.add(
                            (node_id, other_id, metric_name)
                        )
            # status / location_id / beliefs / properties / etc. don't pin
            # any trait \u2014 propagation over the entity's traits is unaffected.

        logger.log(
            _physics_log(),
            "[CausalPhysics·do] Surgeries applied. Intervened roots: %s; pinned traits: %s",
            self._intervened_nodes, self._intervened_traits,
        )

    # ------------------------------------------------------------------
    # Rung 2 — Typed do-target dispatch (Phase 1)
    # ------------------------------------------------------------------
    def apply_do_targets(self, do_targets: List[Any]) -> None:
        """Dispatch typed :class:`DoTarget` payloads to the correct surgery.

        ``DoEvent`` and ``DoTrait`` are translated into the legacy
        ``Dict[str, Any]`` shape and delegated to :meth:`apply_do_operator`
        (one delegation per dispatch call so all event / trait pins are
        applied atomically and the existing provenance invalidation logic
        downstream still sees the full intervention dict via
        ``self._last_legacy_interventions``).

        ``DoBelief``, ``DoConcern`` and ``DoProposition`` are applied
        directly to the sandbox (the proposition layer also lives on
        ``sandbox.graph['proposition_clamps']`` for Phase 2 readers, and
        cascades into matching beliefs when ``propagate_to_beliefs``).
        """
        # Local import to avoid a top-level cycle: query_models is
        # consumer-side and may import causal_physics types in future.
        from shadow_loom.query_models import (
            DoEvent, DoTrait, DoProposition, DoBelief, DoConcern, DoWorldTrait,
            DoChannel, DoRelationship, DoCausalEdge, DoSpatialEdge,
            DoNarrativeObject, DoEntityDelete, DoObjectDelete,
        )

        # Reset the per-call edge-surgery success counter. Vacuity-skip
        # gating in ``narrative_physics`` reads this counter to decide
        # whether an edge typed do-target genuinely mutated the sandbox
        # (count > 0) versus early-returned silently (count == 0). It
        # MUST start at zero for every ``apply_do_targets`` invocation.
        self._edge_do_targets_applied = 0

        # Pre-flight: warn on cross-target inconsistencies the engine
        # will silently apply but that almost always indicate operator
        # error. Detects two classes of conflict:
        #   1. PROP clamp + counter-concern clamp in the same query
        #      whose polarity / active state contradict the truth being
        #      written (e.g. clamp PROP_DUNCAN_DEAD=true while disarming
        #      the desire-throne concern over the same proposition).
        #   2. PROP clamp + DoConcern clamp on a concern anchored to the
        #      inverse proposition whose effective truth contradicts.
        # These remain warnings (not errors) because there are legitimate
        # narrative cases for them (a character whose ambition has burned
        # out before the regicide they once wanted is realised), but the
        # operator should see them in the log.
        prop_clamps: Dict[str, bool] = {
            t.proposition_id: bool(t.truth)
            for t in do_targets if isinstance(t, DoProposition)
        }
        if prop_clamps:
            world = self.world_state
            concern_owners: Dict[str, str] = {}
            concern_by_id: Dict[str, Any] = {}
            prop_by_id: Dict[str, Any] = {}
            for eid, ent in (world.entities or {}).items():
                for c in (ent.concerns or []):
                    concern_owners[c.concern_id] = eid
                    concern_by_id[c.concern_id] = c
            for p in (world.propositions or []):
                prop_by_id[p.proposition_id] = p
            for t in do_targets:
                if not isinstance(t, DoConcern):
                    continue
                c = concern_by_id.get(t.concern_id)
                if c is None:
                    continue
                # Effective proposition truth this concern sees once
                # the PROP clamps land: direct lookup, then inverse.
                effective_truth: Optional[bool] = prop_clamps.get(c.proposition_id)
                if effective_truth is None:
                    p = prop_by_id.get(c.proposition_id)
                    inv = getattr(p, "inverse_proposition_id", None) if p else None
                    if inv and inv in prop_clamps:
                        effective_truth = not prop_clamps[inv]
                if effective_truth is None:
                    continue
                # A desire-concern realised (truth==True) should usually
                # remain active (or be closed by the auto-closure pass)
                # — explicitly disarming it in the same query is a smell.
                disarming = (
                    t.active is False
                    or (t.salience is not None and t.salience < 0.1)
                )
                contradicts_polarity = (
                    t.polarity is not None
                    and ((t.polarity == "desire" and effective_truth is False)
                         or (t.polarity == "fear" and effective_truth is True))
                )
                if disarming and effective_truth is True and c.polarity == "desire":
                    logger.warning(
                        "[CausalPhysics\u00b7do_targets] Consistency: clamping "
                        "PROP %s=true while disarming desire-concern %s "
                        "anchored to it (holder=%s). Auto-closure would "
                        "normally fire on commit; the explicit disarm is "
                        "applied but may double-close.",
                        c.proposition_id, t.concern_id,
                        concern_owners.get(t.concern_id, "?"),
                    )
                if contradicts_polarity:
                    logger.warning(
                        "[CausalPhysics\u00b7do_targets] Consistency: clamping "
                        "concern %s polarity=%s while PROP %s effective "
                        "truth resolves to %s (holder=%s). The polarity flip "
                        "is applied but contradicts the truth surgery.",
                        t.concern_id, t.polarity, c.proposition_id,
                        effective_truth,
                        concern_owners.get(t.concern_id, "?"),
                    )

        legacy_dict: Dict[str, Any] = {}
        for t in do_targets:
            if isinstance(t, DoEvent):
                # Map the typed shape onto the legacy event-prevention
                # vocabulary the instantiator already understands.
                legacy_dict[f"{t.event_id}.event_type"] = (
                    "prevented" if t.occurred is False else "occurred"
                )
            elif isinstance(t, DoTrait):
                legacy_dict[f"{t.holder_id}.traits.{t.trait_name}"] = float(t.value)

        if legacy_dict:
            # Delegate event/trait surgeries through the existing path so
            # ``_intervened_nodes`` / ``_intervened_traits`` and provenance
            # tracking are populated consistently.
            self.apply_do_operator(legacy_dict)
            # Stash for the result-build site so the executor can pass
            # the same dict into ``_collect_provenance_invalidations``.
            existing = getattr(self, "_last_legacy_interventions", {}) or {}
            self._last_legacy_interventions = {**existing, **legacy_dict}
            # Track which legacy keys were ACTUALLY applied via
            # ``apply_do_operator`` (DoEvent / DoTrait subset) so the
            # caller can suppress a second pass in ``execute()`` and
            # avoid double-clamping under inertia dampening.
            existing_applied: set = getattr(self, "_legacy_applied_keys", set()) or set()
            self._legacy_applied_keys = existing_applied | set(legacy_dict.keys())

        # Also stash a legacy-shape mirror of the other three round-
        # trippable typed kinds (DoProposition / DoWorldTrait /
        # DoNarrativeObject) so the result-build site, Rule-3 preflight
        # reportage, and provenance invalidator see the same surgery
        # surface regardless of which entry point the caller used.
        # These are NOT re-routed through ``apply_do_operator`` — they
        # have dedicated ``_apply_do_*`` handlers invoked below — only
        # their legacy-dict shape is recorded.
        #
        # Important: ``DoNarrativeObject``'s legacy mirror keys
        # (``OBJ_X.owner_id`` / ``OBJ_X.location_id``) are intentionally
        # NOT added to ``_legacy_applied_keys``. The typed handler only
        # writes properties; the inventory / spatial edge surgery
        # (drop-at-owner's-room, owned_by edge rewiring) lives in
        # ``_intervene_inventory`` / ``_intervene_spatial`` and must run
        # on the legacy pass.
        reportage_only: Dict[str, Any] = {}
        proposition_world_trait_keys: set = set()
        channel_keys: set = set()
        for t in do_targets:
            if isinstance(t, DoProposition):
                k = f"{t.proposition_id}.truth"
                reportage_only[k] = bool(t.truth)
                proposition_world_trait_keys.add(k)
            elif isinstance(t, DoWorldTrait):
                k = f"{t.world_trait_id}.value"
                reportage_only[k] = float(t.value)
                proposition_world_trait_keys.add(k)
            elif isinstance(t, DoChannel):
                # Round-4 audit fix: legacy-mirror typed channel
                # activations so ``_collect_provenance_invalidations``
                # (narrative_physics ~L442) can prune utterance
                # provenance referencing the deactivated channel. The
                # canonical write happens in ``_apply_do_channel``;
                # this just exposes the surgery on the legacy envelope.
                if t.active is not None:
                    k = f"{t.channel_id}.active"
                    reportage_only[k] = bool(t.active)
                    channel_keys.add(k)
            elif isinstance(t, DoNarrativeObject):
                if t.new_location_id is not None:
                    reportage_only[f"{t.object_id}.location_id"] = t.new_location_id
                elif getattr(t, "set_location_null", False):
                    reportage_only[f"{t.object_id}.location_id"] = None
                if t.new_owner_id is not None:
                    reportage_only[f"{t.object_id}.owner_id"] = t.new_owner_id
                elif getattr(t, "set_owner_null", False):
                    reportage_only[f"{t.object_id}.owner_id"] = None
        if reportage_only:
            existing = getattr(self, "_last_legacy_interventions", {}) or {}
            self._last_legacy_interventions = {**existing, **reportage_only}
            # Proposition / world-trait typed handlers do the canonical
            # write; mark them applied so the legacy ``_intervene_state``
            # fallback doesn't redundantly clamp the same value.
            #
            # Phase 5b audit (Issue A) — also mark DoChannel mirror keys
            # applied. The typed ``_apply_do_channel`` is authoritative
            # for channel state; running the channel key through the
            # legacy ``execute_interventions`` either redundantly sets
            # an "active" attr on the sandbox node OR (when the channel
            # is outside the ego graph) records a spurious
            # ``unknown_node`` entry into ``skipped_interventions``,
            # which downstream surfaces as a false-positive
            # "intervention was skipped" report.
            applied_keys = proposition_world_trait_keys | channel_keys
            if applied_keys:
                existing_applied = getattr(self, "_legacy_applied_keys", set()) or set()
                self._legacy_applied_keys = existing_applied | applied_keys

        # Audit R18-25: normalise dispatch order so a same-batch
        # call cannot create a belief routed through a channel that
        # an earlier-in-the-batch DoChannel just severed (or vice
        # versa: a belief written by a DoEvent that a later same-
        # batch DoEntityDelete then orphans). Tier contract:
        #   (0) structural creates / mutations (objects, channels,
        #       relationships, causal / spatial edges)
        #   (1) destructive operations (entity/object delete, channel
        #       sever)
        #   (2) dependent epistemic / propositional writes (belief,
        #       concern, world-trait, proposition)
        # Within each tier the caller-supplied order is preserved so
        # deterministic same-tier sequences keep behaving as authored.
        def _tier(target: Any) -> int:
            if isinstance(target, (DoNarrativeObject, DoRelationship,
                                   DoCausalEdge, DoSpatialEdge)):
                return 0
            if isinstance(target, (DoEntityDelete, DoObjectDelete)):
                return 1
            if isinstance(target, DoChannel):
                return 1 if getattr(target, "active", None) is False else 0
            return 2

        do_targets_ordered = [
            t for _, t in sorted(
                enumerate(do_targets),
                key=lambda iv: (_tier(iv[1]), iv[0]),
            )
        ]

        for t in do_targets_ordered:
            if isinstance(t, DoProposition):
                self._apply_do_proposition(t)
            elif isinstance(t, DoBelief):
                self._apply_do_belief(t, triggered_by="DO_OPERATOR")
            elif isinstance(t, DoConcern):
                self._apply_do_concern(t)
            elif isinstance(t, DoWorldTrait):
                self._apply_do_world_trait(t)
            elif isinstance(t, DoChannel):
                # Edge-surgery success is signalled by the apply
                # method itself incrementing ``_edge_do_targets_applied``
                # (attribute-only mutations like channel-deactivation or
                # spatial-edge lock/unlock don't change edge / node
                # counts so a before/after diff here under-counts).
                self._apply_do_channel(t)
            elif isinstance(t, DoRelationship):
                self._apply_do_relationship(t)
            elif isinstance(t, DoCausalEdge):
                self._apply_do_causal_edge(t)
            elif isinstance(t, DoSpatialEdge):
                self._apply_do_spatial_edge(t)
            elif isinstance(t, DoNarrativeObject):
                self._apply_do_object(t)
            elif isinstance(t, DoEntityDelete):
                self._apply_do_entity_delete(t)
            elif isinstance(t, DoObjectDelete):
                self._apply_do_object_delete(t)

        # Event-relocation surgery (Rung-2/3): rewrites
        # ``EventNode.at_location_id`` and cascades EntityStateSnapshots
        # for every primary actor at the event's fabula_time so the
        # co-presence invariant continues to hold post-surgery. Run
        # AFTER the legacy event/trait dispatch so the relocated event
        # is still ``occurred`` when this runs.
        for t in do_targets:
            if isinstance(t, DoEvent) and t.new_at_location_id and t.occurred is True:
                self._apply_do_event_relocation(t)

        # Event-time-shift surgery (Rung-2/3): rewrites
        # ``EventNode.fabula_time`` and re-stamps any
        # EntityStateSnapshot / ObjectStateSnapshot whose
        # ``triggered_by`` matches so per-axis last-updated timestamps
        # and reconstruction replay continue to read consistently after
        # the shift. Independent of and composes with relocation.
        for t in do_targets:
            if (
                isinstance(t, DoEvent)
                and t.new_fabula_time is not None
                and t.occurred is True
            ):
                self._apply_do_event_time_shift(t)

        logger.log(
            _physics_log(),
            "[CausalPhysics·do_targets] %d targets applied "
            "(%d prop / %d belief / %d concern / %d world_trait / %d object mutations recorded).",
            len(do_targets),
            len(self._proposition_mutations),
            len(self._belief_mutations),
            len(self._concern_mutations),
            len(self._world_trait_mutations),
            len(self._object_mutations),
        )

    def _apply_do_proposition(self, target: Any) -> None:
        """Clamp a ``Proposition.truth_at_fabula`` and (optionally) cascade
        the clamp into every matching :class:`Belief` on every entity in
        the sandbox.

        The audience-side proposition layer is recorded on
        ``sandbox.graph['proposition_clamps']`` (a list of dicts), so
        Phase 2 readers can recover the surgery without re-walking the
        engine. The clamp is also written back to the live
        ``world_state.propositions`` list so downstream consumers reading
        the world state observe the pinned truth value.

        Three additional invariants beyond a bare timeline write:

        1. **Post-clamp suppression.** The surgery sets the canonical
           truth from ``ft`` forward; any pre-existing factual commits
           at ``fabula_time > ft`` would otherwise flicker the
           shadow-branch truth log (e.g. a counterfactual that pins
           ``X=False`` at T=30000 must not still surface the factual
           ``X=True`` at T=14000 followed by the new commit, because
           the shadow timeline diverges at ``ft``). Drop those later
           commits so the shadow brief shows a coherent monotone
           truth history.
        2. **Inverse-proposition mirror.** Mirrors the Phase C ingestion
           sweep (``ingestion.py`` "Phase C — Affect Reconciler"): when
           ``prop.inverse_proposition_id`` is set and the inverse
           exists in the catalogue, write the negated truth at the
           same tick onto the inverse, applying the same post-clamp
           suppression. Without this a do-surgery on ``PROP_X`` leaves
           ``PROP_NOT_X`` carrying stale factual commits that
           contradict the surgery.
        3. **Defensive copy-on-write.** The proposition (and its
           inverse) is replaced in ``world_state.propositions`` with a
           deep clone before mutation so callers that captured a
           reference to the canonical ``Proposition`` object before the
           shadow surgery (e.g. ``VersionedWorldModel.history``) do not
           observe the in-place edit.
        """
        ft = target.fabula_time
        if ft is None:
            ft = self._default_fabula_time()
        ft = int(ft)

        # Hard-lock + piecewise-dict surgery (Candidate D). Pinning a
        # proposition into the engine's ``_proposition_hard_locks``
        # set causes downstream resolvers (event-driven truth flips,
        # ingestion bridges) to refuse further mutations on this
        # proposition for the rest of the engine's life. The
        # ``truth_at_fabula`` dict overrides the canonical ledger
        # wholesale so an operator can write a piecewise truth arc
        # (e.g. ``{0: True, 5000: False}``) in one intervention
        # instead of issuing per-tick clamps.
        bulk_truth = getattr(target, "truth_at_fabula", None)
        hard_lock = bool(getattr(target, "hard_lock_forever", False))
        if hard_lock:
            locks = getattr(self, "_proposition_hard_locks", None)
            if locks is None:
                locks = set()
                self._proposition_hard_locks = locks
            locks.add(target.proposition_id)

        old_truth: Optional[bool] = None
        prop_index: Optional[int] = None
        inverse_pid: Optional[str] = None
        props_list = self.world_state.propositions or []
        for idx, prop in enumerate(props_list):
            if prop.proposition_id != target.proposition_id:
                continue
            prop_index = idx
            inverse_pid = getattr(prop, "inverse_proposition_id", None)
            if isinstance(prop.truth_at_fabula, dict):
                prior = [v for k, v in prop.truth_at_fabula.items() if int(k) <= ft]
                old_truth = prior[-1] if prior else None
                # Copy-on-write: replace the canonical Proposition with
                # a deep clone before mutating so external references
                # (UI snapshots, version history, sibling queries) keep
                # the unmodified object.
                if bulk_truth:
                    # Wholesale ledger overwrite (piecewise truth arc).
                    new_truth = {int(k): bool(v) for k, v in bulk_truth.items()}
                    # Always honour the primary ``truth`` field at ``ft``
                    # so the do-target's headline assertion lands even
                    # when the bulk dict omits the anchor.
                    new_truth.setdefault(ft, bool(target.truth))
                else:
                    new_truth = {
                        int(k): v for k, v in prop.truth_at_fabula.items() if int(k) < ft
                    }
                    new_truth[ft] = bool(target.truth)
                clone = prop.model_copy(update={"truth_at_fabula": new_truth})
                props_list[idx] = clone
            break

        # Missing-proposition guard (round-8 audit fix). Refuse to
        # record a PropositionMutation or proposition_clamps row when
        # the surgery targets a proposition that is not in the
        # canonical catalogue — the clamp would be unreadable to
        # downstream consumers and pollute the audit log with phantom
        # truth pins.
        if prop_index is None:
            logger.warning(
                "[CausalPhysics·do_proposition] Proposition %s not in "
                "world_state.propositions; skipping clamp.",
                target.proposition_id,
            )
            return

        # Inverse-proposition mirror (parity with Phase C ingestion).
        # Round-10 audit fix (R10-F1): track whether the mirror actually
        # landed so we can emit a paired ``PropositionMutation`` row for
        # the inverse below. Without that row, the pipeline's
        # ``PropositionTruthCommit`` bridge at pipeline.py only persists
        # the primary clamp on merge — the inverse mirror lives solely on
        # the deep-cloned shadow ``world_state.propositions`` (via
        # ``_isolate_ws_for_surgery``) and is discarded after physics.
        # The result was silent desync of every declared inverse pair
        # after any Do-surgery, even though the engine reported the
        # mirror as applied.
        inverse_mirror_applied: bool = False
        inverse_old_truth: Optional[bool] = None
        inverse_new_truth: Optional[bool] = None
        if inverse_pid:
            # HIGH-FIX: Validate inverse proposition exists before mirroring
            inverse_exists = any(p.proposition_id == inverse_pid for p in props_list)
            if not inverse_exists:
                logger.warning(
                    "[CausalPhysics·do_proposition] Inverse proposition %s not found in world_state; skipping mirror for %s",
                    inverse_pid, target.proposition_id
                )
                inverse_pid = None  # Disable mirror to avoid errors below
        
        if inverse_pid:
            for idx, inv_prop in enumerate(props_list):
                if inv_prop.proposition_id != inverse_pid:
                    continue
                inv_val = not bool(target.truth)
                if isinstance(inv_prop.truth_at_fabula, dict):
                    inv_existing = inv_prop.truth_at_fabula.get(ft)
                    if inv_existing is not None and inv_existing != inv_val:
                        # Round-7 audit: previously this branch logged
                        # the conflict and skipped, leaving the primary
                        # clamp committed while the inverse still
                        # carried its contradictory factual value —
                        # the catalogue ended in a self-inconsistent
                        # state (PROP_X=True AND PROP_NOT_X=True at the
                        # same tick). Resolve atomically by forcing the
                        # inverse to match the do-surgery intent; the
                        # primary clamp is the operator-issued
                        # intervention and wins by policy.
                        logger.warning(
                            "[CausalPhysics\u00b7do_proposition] Inverse "
                            "consistency conflict on %s@fabula=%d: existing "
                            "truth %s contradicts mirror from %s=%s "
                            "(would-be inverse=%s). Overwriting inverse to "
                            "restore catalogue consistency.",
                            inverse_pid, ft, inv_existing,
                            target.proposition_id, target.truth, inv_val,
                        )
                    inv_prior = [
                        v for k, v in inv_prop.truth_at_fabula.items()
                        if int(k) < ft
                    ]
                    inverse_old_truth = inv_prior[-1] if inv_prior else None
                    inv_new = {
                        int(k): v for k, v in inv_prop.truth_at_fabula.items()
                        if int(k) < ft
                    }
                    inv_new[ft] = inv_val
                    props_list[idx] = inv_prop.model_copy(
                        update={"truth_at_fabula": inv_new}
                    )
                    inverse_mirror_applied = True
                    inverse_new_truth = inv_val
                break

        if hard_lock and inverse_pid:
            locks = getattr(self, "_proposition_hard_locks", None)
            if locks is None:
                locks = set()
                self._proposition_hard_locks = locks
            locks.add(inverse_pid)

        # Refresh the sandbox's serialised proposition layer so
        # downstream consumers reading ``sandbox.graph['propositions']``
        # (Q&A compressor, brief renderer) observe the clamp + inverse
        # mirror instead of the pre-surgery snapshot taken by
        # ``_stamp_utility_layer``.
        if prop_index is not None:
            sand_props = self.sandbox.graph.get("propositions")
            if isinstance(sand_props, list):
                self.sandbox.graph["propositions"] = [
                    p.model_dump() for p in props_list
                ]

        # Persist the clamp on the sandbox graph for Phase-2 consumers.
        clamps = self.sandbox.graph.setdefault("proposition_clamps", [])
        clamps.append({
            "proposition_id": target.proposition_id,
            "fabula_time": ft,
            "truth": bool(target.truth),
        })

        cascaded = 0
        if target.propagate_to_beliefs:
            cascaded = self._cascade_proposition_to_beliefs(target)
            # P0-FIX: Cascade to inverse proposition beliefs as well
            if inverse_pid and inverse_mirror_applied:
                cascaded += self._cascade_inverse_proposition_to_beliefs(
                    target, inverse_pid, bool(inverse_new_truth)
                )

        self._proposition_mutations.append(PropositionMutation(
            proposition_id=target.proposition_id,
            fabula_time=ft,
            old_truth=old_truth,
            new_truth=bool(target.truth),
            cascaded_belief_count=cascaded,
        ))
        # Round-10 audit fix (R10-F1): emit a paired ``PropositionMutation``
        # for the inverse so the pipeline merge bridge emits a matching
        # ``PropositionTruthCommit`` and the
        # lockstep with the primary clamp. Phase C ingestion already does
        # this mirror on the canonical side; the Pearl Rung-2 path missed
        # it. Cascaded belief count is reported as 0 because the cascade
        # ran against the *primary* proposition's id only — beliefs tied
        # to the inverse keep their existing confidence (a future round
        # could cascade through the inverse too if needed).
        if inverse_mirror_applied and inverse_pid:
            self._proposition_mutations.append(PropositionMutation(
                proposition_id=inverse_pid,
                fabula_time=ft,
                old_truth=inverse_old_truth,
                new_truth=bool(inverse_new_truth),
                cascaded_belief_count=0,
            ))

    def _cascade_proposition_to_beliefs(self, target: Any) -> int:
        """Clamp every belief whose ``proposition_id`` matches the
        proposition being intervened on. Confidence is set to 1.0 when the
        belief's ``perceived_state`` aligns with the new truth, else 0.0,
        gated by the belief's ``evidence_strength`` (so weakly-evidenced
        beliefs swing less than confidently held ones).

        Returns the number of beliefs cascaded.
        """
        from shadow_loom.query_models import DoBelief
        count = 0
        for nid, ndata in self.sandbox.nodes(data=True):
            if ndata.get("node_type") != "Entity":
                continue
            beliefs = ndata.get("beliefs")
            if not isinstance(beliefs, list):
                continue
            for b in beliefs:
                if not isinstance(b, dict):
                    continue
                if b.get("proposition_id") != target.proposition_id:
                    continue
                # Default cascade rule: align belief confidence with the
                # clamped truth, scaled by evidence_strength (default 1.0).
                # Belief.evidence_strength is a Literal["weak","moderate",
                # "strong"]; map through the engine's strength multiplier.
                es_raw = b.get("evidence_strength")
                if isinstance(es_raw, str):
                    evidence = float(_strength_weight(es_raw))
                elif es_raw is None:
                    evidence = 1.0
                else:
                    evidence = float(es_raw)
                aligned = self._belief_aligned_with_truth(b, bool(target.truth))
                new_conf = (1.0 if aligned else 0.0) * max(0.0, min(1.0, evidence))
                # Use the belief-clamp helper so the mutation is recorded
                # consistently and provenance is set to the proposition.
                self._apply_do_belief(
                    DoBelief(
                        holder_id=nid,
                        target_id=b.get("target_id", ""),
                        confidence=new_conf,
                        proposition_id=target.proposition_id,
                    ),
                    triggered_by=target.proposition_id,
                )
                count += 1
        return count

    @staticmethod
    def _belief_aligned_with_truth(belief: Dict[str, Any], truth: bool) -> bool:
        """Best-effort alignment: a belief whose ``perceived_state`` /
        polarity affirms the proposition increases with truth=True; a
        belief that denies the proposition decreases with truth=True.

        Conservative default when polarity cannot be inferred: treat the
        belief as affirming (so confidence tracks the clamped truth).
        """
        polarity = belief.get("polarity") or belief.get("affirms")
        if isinstance(polarity, bool):
            return polarity == truth
        ps = (belief.get("perceived_state") or "").lower()
        if any(k in ps for k in ("not ", "no ", "false", "denies", "rejects")):
            return (not truth)
        return bool(truth)

    def _apply_do_belief(self, target: Any, *, triggered_by: str = "DO_OPERATOR") -> None:
        """Locate-or-create the belief on the holder entity and clamp
        confidence. Marks ``acquired_via_event_id`` so downstream
        propagation can see the do-operator provenance and so the
        provenance-pruner does not strip it.

        Does *not* cascade — belief surgeries are epistemic-only by
        design; downstream effects flow naturally through the standard
        propagate step which already reads beliefs.
        """
        # MODERATE-FIX (M-005): Validate belief channel provenance at creation.
        # Ensures acquired_via_channel_id references existing channel to maintain
        # provenance chain integrity.
        if target.acquired_via_channel_id:
            channel_exists = any(
                getattr(ch, "id", getattr(ch, "channel_id", None))
                == target.acquired_via_channel_id
                for ch in (self.world_state.channels or {}).values()
            )
            if not channel_exists:
                logger.warning(
                    "[CausalPhysics·do_belief] acquired_via_channel_id=%s not found. "
                    "Clearing provenance.",
                    target.acquired_via_channel_id
                )
                target.acquired_via_channel_id = None
        
        if not self.sandbox.has_node(target.holder_id):
            logger.warning(
                "[CausalPhysics·do_belief] Holder %s missing from sandbox; "
                "skipping clamp on %s.",
                target.holder_id, target.target_id,
            )
            return
        # Audit R18-11: validate the proposition referent before
        # mutating. Beliefs about *entities* may legitimately target
        # not-yet-registered nodes (off-stage referents, future
        # introductions); only the proposition link is strict because
        # propositions are first-class DB entries and a typoed
        # ``PROP_X`` always indicates a bug.
        if target.proposition_id:
            if not any(
                getattr(p, "proposition_id", None) == target.proposition_id
                for p in (self.world_state.propositions or [])
            ):
                logger.warning(
                    "[CausalPhysics·do_belief] proposition_id=%s is "
                    "not registered; refusing to attach belief on "
                    "holder=%s.",
                    target.proposition_id, target.holder_id,
                )
                return
        ndata = self.sandbox.nodes[target.holder_id]
        if ndata.get("node_type") != "Entity":
            return
        # Audit R17-6: derive the fabula tick from the trigger event
        # so newly-created beliefs carry ``established_at_fabula`` and
        # downstream reconstruction can order epistemic state correctly.
        trigger_ft: int = 0
        if triggered_by and triggered_by != "DO_OPERATOR":
            for _e in (self.world_state.events or []):
                if getattr(_e, "id", None) == triggered_by:
                    trigger_ft = int(getattr(_e, "fabula_time", 0) or 0)
                    break
        beliefs = ndata.setdefault("beliefs", [])
        # Locate by (target_id, proposition_id) — proposition_id wins when
        # both sides supply one.
        existing = None
        for b in beliefs:
            if not isinstance(b, dict):
                continue
            if target.proposition_id and b.get("proposition_id") == target.proposition_id:
                existing = b
                break
            if b.get("target_id") == target.target_id and not target.proposition_id:
                existing = b
                break

        old_conf: Optional[float] = None
        created = False
        if existing is None:
            created = True
            existing = {
                "target_id": target.target_id,
                "perceived_state": target.perceived_state or "",
                "confidence": float(target.confidence),
                "evidence_strength": 1.0,
                "proposition_id": target.proposition_id,
                "acquired_via_event_id": triggered_by,
                # Audit R18-10: persist the validated channel
                # provenance so a later DoChannel sever can prune
                # this belief (was dropped on the floor here and at
                # the canonical mirror below).
                "acquired_via_channel_id": target.acquired_via_channel_id,
                # Audit R17-6: timestamp the dict-shape belief so
                # reconstruct_entity_at orders sandbox-mirrored
                # beliefs alongside canonical ones.
                "established_at_fabula": trigger_ft,
            }
            beliefs.append(existing)
        else:
            old_conf = float(existing.get("confidence", 0.0))
            existing["confidence"] = float(target.confidence)
            existing["acquired_via_event_id"] = triggered_by
            if target.acquired_via_channel_id:
                # Audit R18-10: update channel provenance on existing
                # belief if the surgery specifies a (different) one.
                existing["acquired_via_channel_id"] = target.acquired_via_channel_id
            if target.perceived_state and not existing.get("perceived_state"):
                existing["perceived_state"] = target.perceived_state
            if target.proposition_id and not existing.get("proposition_id"):
                existing["proposition_id"] = target.proposition_id

        # Pin the holder so propagation does not silently overwrite the
        # belief via downstream cascades.
        self._intervened_nodes.add(target.holder_id)

        self._belief_mutations.append(BeliefMutation(
            holder_id=target.holder_id,
            target_id=target.target_id,
            proposition_id=target.proposition_id or existing.get("proposition_id"),
            old_confidence=old_conf,
            new_confidence=float(target.confidence),
            created=created,
            triggered_by=triggered_by,
        ))

        # Canonical mirror (round-8 audit fix). The sandbox dict-shape
        # belief lives only in the engine's working memory; re-extraction
        # reads :attr:`Entity.beliefs` from the canonical world, so
        # without this mirror the surgery would not survive a merge.
        canonical_holder = (self.world_state.entities or {}).get(target.holder_id)
        if canonical_holder is not None:
            from shadow_loom.models import Belief
            canonical_beliefs = list(getattr(canonical_holder, "beliefs", None) or [])
            mirrored = False
            for cb in canonical_beliefs:
                pid_match = (
                    target.proposition_id
                    and getattr(cb, "proposition_id", None) == target.proposition_id
                )
                tid_match = (
                    not target.proposition_id
                    and getattr(cb, "target_id", None) == target.target_id
                )
                if pid_match or tid_match:
                    cb.confidence = float(target.confidence)
                    cb.acquired_via_event_id = triggered_by
                    # Audit R18-10: mirror channel provenance onto
                    # the canonical Belief so the later sever can
                    # detect this belief on the prune walk.
                    if target.acquired_via_channel_id:
                        cb.acquired_via_channel_id = target.acquired_via_channel_id
                    mirrored = True
                    break
            if not mirrored:
                try:
                    canonical_beliefs.append(Belief(
                        target_id=target.target_id or "",
                        perceived_state=target.perceived_state or "",
                        confidence=float(target.confidence),
                        inertia=0.3,
                        evidence_strength="moderate",
                        proposition_id=target.proposition_id,
                        acquired_via_event_id=triggered_by,
                        # Audit R18-10: carry channel provenance into
                        # the canonical Belief so DoChannel severance
                        # can prune this belief later.
                        acquired_via_channel_id=target.acquired_via_channel_id,
                        # Audit R17-6: stamp the canonical belief so
                        # reconstruct_entity_at can sort epistemic
                        # state by ``established_at_fabula`` instead
                        # of falling back to insertion order.
                        established_at_fabula=trigger_ft,
                    ))
                except Exception:
                    logger.exception(
                        "[CausalPhysics·do_belief] Canonical belief mirror "
                        "validation failed for holder=%s target=%s.",
                        target.holder_id, target.target_id,
                    )
            canonical_holder.beliefs = canonical_beliefs

    def _cascade_inverse_proposition_to_beliefs(
        self, primary_target: Any, inverse_pid: str, inverse_new_truth: bool
    ) -> int:
        """P0-FIX: Cascade belief updates to the INVERSE proposition when
        the primary is clamped. Example: when PROP_DUNCAN_ALIVE=False is set,
        beliefs about PROP_DUNCAN_DEAD should also update.
        
        Returns the number of inverse beliefs cascaded.
        """
        from shadow_loom.query_models import DoBelief
        count = 0
        for nid, ndata in self.sandbox.nodes(data=True):
            if ndata.get("node_type") != "Entity":
                continue
            beliefs = ndata.get("beliefs")
            if not isinstance(beliefs, list):
                continue
            for b in beliefs:
                if not isinstance(b, dict):
                    continue
                if b.get("proposition_id") != inverse_pid:
                    continue
                # Inverse belief alignment: belief aligned with inverse_new_truth
                es_raw = b.get("evidence_strength")
                if isinstance(es_raw, str):
                    evidence = float(_strength_weight(es_raw))
                elif es_raw is None:
                    evidence = 1.0
                else:
                    evidence = float(es_raw)
                
                # Check if belief aligns with the inverse truth value
                aligned = self._belief_aligned_with_truth(b, inverse_new_truth)
                new_conf = (1.0 if aligned else 0.0) * max(0.0, min(1.0, evidence))
                
                # Use the belief-clamp helper for consistency
                self._apply_do_belief(
                    DoBelief(
                        holder_id=nid,
                        target_id=b.get("target_id", ""),
                        confidence=new_conf,
                        proposition_id=inverse_pid,
                    ),
                    triggered_by=inverse_pid,
                )
                count += 1
        return count

    def _apply_do_concern(self, target: Any) -> None:
        """Clamp salience / polarity / activation on a single
        :class:`Concern` belonging to ``target.holder_id``.

        Records one :class:`ConcernMutation` per field actually changed so
        the auditor can attribute downstream affect-unification deltas to
        a specific utility-layer surgery.
        """
        if not self.sandbox.has_node(target.holder_id):
            logger.warning(
                "[CausalPhysics·do_concern] Holder %s missing from sandbox; "
                "skipping clamp on concern %s.",
                target.holder_id, target.concern_id,
            )
            return
        ndata = self.sandbox.nodes[target.holder_id]
        if ndata.get("node_type") != "Entity":
            return
        concerns = ndata.get("concerns")
        if not isinstance(concerns, list):
            concerns = []
            ndata["concerns"] = concerns

        concern = next(
            (c for c in concerns if isinstance(c, dict)
             and c.get("concern_id") == target.concern_id),
            None,
        )
        if concern is None:
            logger.warning(
                "[CausalPhysics·do_concern] Concern %s not found on %s; "
                "skipping (creation requires polarity/proposition_id which "
                "the DoConcern surface does not carry).",
                target.concern_id, target.holder_id,
            )
            return

        ft = self._default_fabula_time()

        if target.polarity is not None and concern.get("polarity") != target.polarity:
            old = concern.get("polarity")
            concern["polarity"] = target.polarity
            self._concern_mutations.append(ConcernMutation(
                holder_id=target.holder_id, concern_id=target.concern_id,
                field="polarity", old_value=old, new_value=target.polarity,
            ))

        if target.salience is not None:
            old = concern.get("salience")
            if old != target.salience:
                concern["salience"] = float(target.salience)
                self._concern_mutations.append(ConcernMutation(
                    holder_id=target.holder_id, concern_id=target.concern_id,
                    field="salience", old_value=old, new_value=float(target.salience),
                ))

        if target.active is not None:
            old_window = concern.get("activation_fabula_window")
            if target.active is False:
                # Collapse the activation window past the query horizon.
                new_window = [int(ft) + 1, int(ft) + 1]
            else:
                # Always-on: clear any window.
                new_window = None
            concern["activation_fabula_window"] = new_window
            self._concern_mutations.append(ConcernMutation(
                holder_id=target.holder_id, concern_id=target.concern_id,
                field="active", old_value=old_window, new_value=new_window,
            ))

        # Pin the holder so downstream propagation does not regenerate
        # concerns that the surgery just suppressed.
        self._intervened_nodes.add(target.holder_id)

        # Canonical mirror (round-8 audit fix). Without this the
        # surgery only lives in the sandbox node attrs; the next
        # re-extraction pass reads concerns from
        # ``world_state.entities[holder].concerns`` and silently
        # restores the pre-surgery state.
        canonical_holder = (self.world_state.entities or {}).get(target.holder_id)
        if canonical_holder is not None:
            canonical_concerns = list(getattr(canonical_holder, "concerns", None) or [])
            for cc in canonical_concerns:
                if getattr(cc, "concern_id", None) != target.concern_id:
                    continue
                if target.polarity is not None:
                    cc.polarity = target.polarity
                if target.salience is not None:
                    cc.salience = float(target.salience)
                if target.active is not None:
                    if target.active is False:
                        cc.activation_fabula_window = [int(ft) + 1, int(ft) + 1]
                    else:
                        cc.activation_fabula_window = None
                break
            canonical_holder.concerns = canonical_concerns

    def _apply_do_world_trait(self, target: Any) -> None:
        """Clamp a WORLD_ ``GlobalTrait``'s magnitude — ambient-force
        intervention.

        Mutates the sandbox node's ``magnitude`` (so the current
        propagate step sees the new ambient value) and records a
        :class:`WorldTraitMutation` for the pipeline adapter to fold
        onto the canonical timeline as a
        :class:`WorldTraitSnapshot` (honouring inertia attenuation).
        Domain set-ops (add/remove) are applied directly to the
        sandbox node so the runtime domain gate routes correctly on
        the same propagate step. Pins the WORLD_ id in
        ``_intervened_nodes`` so it stays an active source.
        """
        wt_id = target.world_trait_id
        if not self.sandbox.has_node(wt_id):
            logger.warning(
                "[CausalPhysics·do_world_trait] Trait %s missing from sandbox; "
                "skipping clamp.", wt_id,
            )
            return
        ndata = self.sandbox.nodes[wt_id]
        if ndata.get("node_type") != "WorldTrait":
            logger.warning(
                "[CausalPhysics·do_world_trait] Node %s is not a WorldTrait "
                "(node_type=%s); skipping clamp.",
                wt_id, ndata.get("node_type"),
            )
            return
        mag = ndata.get("magnitude")
        if not isinstance(mag, dict):
            mag = {"value": 0.5, "inertia": 0.3}
            ndata["magnitude"] = mag
        old_value = float(mag.get("value", 0.5))
        new_value = float(max(0.0, min(1.0, target.value)))
        mag["value"] = new_value
        if target.inertia is not None:
            mag["inertia"] = float(max(0.0, min(0.99, target.inertia)))

        # Set-ops on affected_domains. Default to canonical-7 keep-order
        # by deduping while preserving insertion order.
        domains = list(ndata.get("affected_domains") or [])
        if target.affected_domains_add:
            for d in target.affected_domains_add:
                if d not in domains:
                    domains.append(str(d))
        if target.affected_domains_remove:
            domains = [d for d in domains if d not in set(target.affected_domains_remove)]
        ndata["affected_domains"] = domains

        ft = (
            int(target.fabula_time)
            if getattr(target, "fabula_time", None) is not None
            else self._default_fabula_time()
        )

        # Canonical mirror (round-8 audit fix). Every other Do-handler
        # mirrors its sandbox mutation onto ``world_state`` so the
        # re-extraction pass observes the surgery. WorldTrait used to
        # only mutate the sandbox node, leaving
        # ``world_state.world_traits[wt_id].magnitude`` stale and
        # propagating the pre-surgery value back on the next merge.
        canonical_wt = (self.world_state.world_traits or {}).get(wt_id)
        if canonical_wt is not None:
            try:
                from shadow_loom.models import TraitVector
                mag_obj = getattr(canonical_wt, "magnitude", None)
                if isinstance(mag_obj, TraitVector):
                    mag_obj.value = new_value
                    if target.inertia is not None:
                        mag_obj.inertia = float(max(0.0, min(0.99, target.inertia)))
                else:
                    canonical_wt.magnitude = TraitVector(
                        value=new_value,
                        inertia=float(target.inertia) if target.inertia is not None else 0.3,
                    )
                # Mirror domain set-ops onto canonical too.
                cdomains = list(getattr(canonical_wt, "affected_domains", None) or [])
                if target.affected_domains_add:
                    for d in target.affected_domains_add:
                        if d not in cdomains:
                            cdomains.append(str(d))
                if target.affected_domains_remove:
                    cdomains = [d for d in cdomains if d not in set(target.affected_domains_remove)]
                canonical_wt.affected_domains = cdomains
                # Round-12 audit: world-trait do-surgery used to mutate
                # only the canonical magnitude/domains but never
                # append a WorldTraitSnapshot, while DoNarrativeObject
                # writes ObjectStateSnapshot to canonical state_timeline.
                # That asymmetry left ``reconstruct_*_at`` replay blind
                # to the clamp. Append a snapshot here so engine
                # writes-canonical is uniform across object &
                # world-trait paths.
                try:
                    from shadow_loom.models import WorldTraitSnapshot
                    canonical_wt.state_timeline = list(
                        getattr(canonical_wt, "state_timeline", None) or []
                    )
                    canonical_wt.state_timeline.append(WorldTraitSnapshot(
                        fabula_time=ft,
                        triggered_by=getattr(target, "triggered_by", None),
                        magnitude=TraitVector(
                            value=new_value,
                            inertia=(
                                float(target.inertia)
                                if target.inertia is not None
                                else (mag.get("inertia") or 0.3)
                            ),
                        ),
                    ))
                    canonical_wt.state_timeline.sort(key=lambda s: s.fabula_time)
                except Exception:
                    logger.exception(
                        "[CausalPhysics\u00b7do_world_trait] Snapshot append failed for %s.",
                        wt_id,
                    )
            except Exception:
                logger.exception(
                    "[CausalPhysics·do_world_trait] Canonical mirror failed for %s.",
                    wt_id,
                )

        self._world_trait_mutations.append(WorldTraitMutation(
            world_trait_id=wt_id,
            fabula_time=ft,
            old_value=old_value,
            new_value=new_value,
            inertia=mag.get("inertia"),
            affected_domains_add=list(target.affected_domains_add or []),
            affected_domains_remove=list(target.affected_domains_remove or []),
            triggered_by=getattr(target, "triggered_by", None),
        ))
        # Pin so the trait remains an always-active ambient source for
        # the rest of this simulation step.
        self._intervened_nodes.add(wt_id)

    # ------------------------------------------------------------------
    # Edge-layer typed surgeries (DoChannel / DoRelationship /
    # DoCausalEdge / DoSpatialEdge). Lightweight implementations: each
    # mutates the sandbox graph (so the same propagate step sees the
    # change) and mirrors back to ``world_state`` where applicable so
    # downstream consumers reading the world directly observe the
    # surgery. None of these record a typed mutation row \u2014 the
    # auditor consumes the sandbox snapshot via ``physics_state`` and
    # the world-state mirror via the standard re-extraction path.
    # ------------------------------------------------------------------
    def _apply_do_channel(self, target: Any) -> None:
        """Toggle activation / re-tune intelligibility on a Channel.

        Channel creation is handled by ``query.introduce.channels``
        (pre-spawn into the world before physics); this surgery only
        mutates an *existing* channel.
        """
        ft = (
            int(target.fabula_time)
            if getattr(target, "fabula_time", None) is not None
            else self._default_fabula_time()
        )
        # Sandbox-side mutation: channel nodes carry ``node_type='Channel'``.
        if self.sandbox.has_node(target.channel_id):
            ndata = self.sandbox.nodes[target.channel_id]
            if target.active is True:
                ndata["terminated_at_fabula"] = None
            elif target.active is False:
                ndata["terminated_at_fabula"] = ft
            if target.intelligibility:
                intel = dict(ndata.get("intelligibility") or {})
                intel.update({k: float(v) for k, v in target.intelligibility.items()})
                ndata["intelligibility"] = intel
            self._intervened_nodes.add(target.channel_id)
        else:
            logger.warning(
                "[CausalPhysics\u00b7do_channel] Channel %s missing from sandbox; "
                "world-state mirror still applied.", target.channel_id,
            )
        # World-state mirror so ego-graph re-extraction picks up the change.
        canonical = self.world_state.channels or {}
        ch = canonical.get(target.channel_id)
        _channel_applied = False
        if ch is not None:
            if target.active is True:
                # Audit R18-26: re-enabling a previously terminated
                # channel used to just clear ``terminated_at_fabula``,
                # which retroactively re-opened the channel for the
                # entire gap (POV reads at ft \u2208 [old_term, ft_now)
                # would see the channel as available again). The flat
                # schema cannot express multi-segment availability,
                # so we approximate the new segment by bumping
                # ``established_at_fabula`` forward to the reopen
                # time. Downstream readers (channel-aware POV,
                # belief-acquisition gate) then see the channel as
                # closed during the gap and open from ft_now onward.
                prior_term = ch.terminated_at_fabula
                if prior_term is not None and prior_term <= ft:
                    ch.established_at_fabula = ft
                ch.terminated_at_fabula = None
            elif target.active is False:
                ch.terminated_at_fabula = ft
            if target.intelligibility:
                merged = dict(ch.intelligibility or {})
                merged.update({k: float(v) for k, v in target.intelligibility.items()})
                ch.intelligibility = merged
            self._edge_do_targets_applied += 1
            _channel_applied = True
        else:
            # Round-8 audit fix — every other Do-handler warns when the
            # canonical mirror cannot be applied; channel used to fail
            # silently which made debugging missing-mirror cases hard.
            logger.warning(
                "[CausalPhysics·do_channel] Channel %s missing from "
                "world.channels; canonical mirror skipped.",
                target.channel_id,
            )
            # Sandbox-only success still counts as work for vacuity gating.
            if self.sandbox.has_node(target.channel_id):
                self._edge_do_targets_applied += 1
                _channel_applied = True
        if _channel_applied:
            if target.active is False:
                _action = "deactivate"
            elif target.active is True:
                _action = "activate"
            else:
                _action = "retune"
            self._edge_mutations.append(EdgeMutation(
                edge_type="channel",
                action=_action,
                channel_id=target.channel_id,
                fabula_time=ft,
                details={
                    "active": target.active,
                    "intelligibility": (
                        {k: float(v) for k, v in target.intelligibility.items()}
                        if target.intelligibility else None
                    ),
                    "canonical_mirror": ch is not None,
                },
            ))
        
        # MODERATE-FIX (M-004): Channel termination cascades to invalidate beliefs.
        # When channel is severed (active=False), prune beliefs acquired via that channel
        # to maintain epistemic consistency.
        if target.active is False and target.channel_id:
            from shadow_loom.instantiator import AMWNInstantiator
            pruned_count = AMWNInstantiator._prune_beliefs_by_provenance(
                self.sandbox,
                removed_channel_ids={target.channel_id},
            )
            if pruned_count > 0:
                logger.info(
                    "[CausalPhysics·do_channel] Channel %s terminated, pruned %d belief(s).",
                    target.channel_id, pruned_count
                )
                # Mirror belief pruning to canonical world_state
                self._mirror_belief_pruning_to_canonical(
                    removed_event_ids=set(),
                    removed_channel_ids={target.channel_id},
                )

    def _apply_do_event_relocation(self, target: Any) -> None:
        """Rewrite an event's ``at_location_id`` and cascade
        ``EntityStateSnapshot(location_id=...)`` for every primary actor
        so the co-presence invariant continues to hold post-surgery.

        Skips dead actors (their ``status`` makes physical relocation
        meaningless) and logs the skip. Mirrors the change to both the
        sandbox node attrs and ``world_state.events`` so re-extraction
        observes the relocated event.
        """
        from shadow_loom.models import EntityStateSnapshot

        new_loc = target.new_at_location_id
        if not new_loc:
            return
        # Validate the target location exists in the world state.
        if new_loc not in (self.world_state.locations or {}):
            logger.warning(
                "[CausalPhysics\u00b7do_event_relocation] Unknown LOC_ id %r "
                "for event %s; skipping relocation.",
                new_loc, target.event_id,
            )
            return
        # Locate the event in world_state.events (a list).
        evt = None
        for e in (self.world_state.events or []):
            if e.id == target.event_id:
                evt = e
                break
        if evt is None:
            logger.warning(
                "[CausalPhysics\u00b7do_event_relocation] Event %s not in "
                "world_state.events; skipping relocation.", target.event_id,
            )
            return
        old_loc = evt.at_location_id
        evt.at_location_id = new_loc
        # Sandbox mirror.
        if self.sandbox.has_node(target.event_id):
            self.sandbox.nodes[target.event_id]["at_location_id"] = new_loc
            self._intervened_nodes.add(target.event_id)
        # Resolve primary actors: speaker first for utterances, else
        # actor_ids in declared order.
        actor_ids: list[str] = []
        if getattr(evt, "speaker_id", None):
            actor_ids.append(evt.speaker_id)
        for aid in (getattr(evt, "actor_ids", None) or []):
            if aid not in actor_ids:
                actor_ids.append(aid)
        ft = int(evt.fabula_time)
        cascaded = 0
        skipped_dead: list[str] = []
        entities = self.world_state.entities or {}
        for aid in actor_ids:
            actor = entities.get(aid) if isinstance(entities, dict) else None
            if actor is None:
                continue
            if str(getattr(actor, "status", "") or "").lower() == "dead":
                skipped_dead.append(aid)
                continue
            timeline = list(getattr(actor, "state_timeline", None) or [])
            timeline.append(EntityStateSnapshot(
                fabula_time=ft,
                triggered_by=evt.id,
                location_id=new_loc,
            ))
            timeline.sort(key=lambda s: s.fabula_time)
            actor.state_timeline = timeline
            # Sandbox mirror: also stash a flag so downstream consumers
            # see the relocation was applied.
            if self.sandbox.has_node(aid):
                self.sandbox.nodes[aid]["location_id"] = new_loc
                self._intervened_nodes.add(aid)
            cascaded += 1
        logger.info(
            "[CausalPhysics\u00b7do_event_relocation] Event %s relocated "
            "%s\u2192%s at fabula_time=%d; cascaded %d actor snapshot(s); "
            "skipped %d dead actor(s).",
            target.event_id, old_loc, new_loc, ft, cascaded, len(skipped_dead),
        )
        self._event_mutations.append(EventMutation(
            event_id=target.event_id,
            kind="relocation",
            fabula_time=ft,
            old_at_location_id=old_loc,
            new_at_location_id=new_loc,
            cascaded_actor_snapshots=cascaded,
            skipped_dead_actors=list(skipped_dead),
            triggered_by=getattr(target, "triggered_by", None),
        ))

    def _apply_do_event_time_shift(self, target: Any) -> None:
        """Rewrite an event's ``fabula_time`` and re-stamp every
        ``EntityStateSnapshot`` / ``ObjectStateSnapshot`` /
        ``WorldTraitSnapshot`` / ``PropositionSnapshot`` /
        ``ConcernSnapshot`` whose ``triggered_by`` matches the shifted
        event so per-axis ``last_updated_fabula`` timestamps and
        ``reconstruct_*_at`` replay stay consistent at the new tick.

        Also re-stamps:
          * any ``RelationshipMetric.last_updated_fabula`` whose axis
            was last touched by this event (best-effort: scans
            ``social_topology`` mutation_social ``CausalEdge`` entries);
          * the ``CausalEdge.fabula_time`` on every edge whose
            ``source_id`` is the shifted event.

        Mirrors changes into the sandbox node attrs so downstream
        propagation observes the new tick.
        """
        new_ft = int(target.new_fabula_time) if target.new_fabula_time is not None else None
        if new_ft is None:
            return
        # Locate the event in world_state.events (a list).
        evt = None
        for e in (self.world_state.events or []):
            if e.id == target.event_id:
                evt = e
                break
        if evt is None:
            logger.warning(
                "[CausalPhysics\u00b7do_event_time_shift] Event %s not in "
                "world_state.events; skipping time shift.", target.event_id,
            )
            return
        old_ft = int(evt.fabula_time)
        if old_ft == new_ft:
            return
        evt.fabula_time = new_ft
        # Sandbox mirror.
        if self.sandbox.has_node(target.event_id):
            self.sandbox.nodes[target.event_id]["fabula_time"] = new_ft
            self._intervened_nodes.add(target.event_id)

        cascaded = 0

        # Re-stamp snapshots triggered_by this event across every
        # holder kind. State-timeline snapshots are sorted by
        # fabula_time after re-stamping so reconstruction replay sees
        # them in the correct order.
        def _restamp_timeline(holder: Any, attr: str) -> int:
            timeline = list(getattr(holder, attr, None) or [])
            n = 0
            for snap in timeline:
                if getattr(snap, "triggered_by", None) == target.event_id:
                    snap.fabula_time = new_ft
                    # Audit R17-7: a snapshot triggered by the shifted
                    # event carries epistemic deltas
                    # (``beliefs_added``) that were stamped with the
                    # old tick. Re-stamp them in lockstep so the
                    # per-belief ``established_at_fabula`` stays
                    # coherent with the snapshot's ``fabula_time``;
                    # otherwise reconstruct_entity_at orders the
                    # belief at the old tick while the snapshot lives
                    # at the new one.
                    for _b in (getattr(snap, "beliefs_added", None) or []):
                        try:
                            _b.established_at_fabula = new_ft
                        except Exception:
                            pass
                    # Audit (sixth pass, F4): ``ConcernSnapshot``
                    # auto-close in ingestion Phase C sets
                    # ``activation_fabula_window = [lo, commit_fab]``
                    # where ``commit_fab`` is the triggering event's
                    # fabula_time. Shifting the event without
                    # remapping the window leaves the closure tick
                    # dangling at the old tick, so
                    # ``reconstruct_concern_at`` reports the concern
                    # as still-open at the now-vacated old tick and
                    # still-closed at the new tick.
                    _win = getattr(snap, "activation_fabula_window", None)
                    if _win:
                        try:
                            snap.activation_fabula_window = [
                                new_ft if int(t) == old_ft else int(t)
                                for t in _win
                            ]
                        except Exception:
                            pass
                    n += 1
            if n:
                timeline.sort(key=lambda s: s.fabula_time)
                setattr(holder, attr, timeline)
            return n

        entities = self.world_state.entities or {}
        for ent in (entities.values() if isinstance(entities, dict) else (entities or [])):
            cascaded += _restamp_timeline(ent, "state_timeline")
        objects = getattr(self.world_state, "objects", None) or {}
        for obj in (objects.values() if isinstance(objects, dict) else (objects or [])):
            cascaded += _restamp_timeline(obj, "state_timeline")
        for prop in (self.world_state.propositions or []):
            cascaded += _restamp_timeline(prop, "state_timeline")
        world_traits = getattr(self.world_state, "world_traits", None) or {}
        for trait in (
            world_traits.values()
            if isinstance(world_traits, dict)
            else (world_traits or [])
        ):
            cascaded += _restamp_timeline(trait, "state_timeline")
        # Concerns hang off Entity, not WorldStateV1 \u2014 iterate every
        # entity's concern list so their state-timelines are restamped
        # alongside trait/object/world-trait/proposition timelines.
        # (Previous loop iterated ``world_state.concerns`` which does
        # not exist, so concern snapshots silently kept the stale tick
        # after a DoEventTimeShift.)
        for ent in (entities.values() if isinstance(entities, dict) else (entities or [])):
            for concern in (getattr(ent, "concerns", None) or []):
                cascaded += _restamp_timeline(concern, "state_timeline")

        # Audit (sixth pass, F8): ``_apply_do_belief`` writes
        # top-level ``Entity.beliefs`` (outside any state_timeline
        # snapshot) with ``acquired_via_event_id=triggered_by`` and
        # ``established_at_fabula=trigger_ft`` (causal_physics.py
        # L1957/L1974). ``affect_unification`` and merge-time
        # reconcilers similarly stamp ``acquired_via_event_id``.
        # The snapshot loop above only catches beliefs nested inside
        # state_timeline; standalone top-level beliefs keep the old
        # tick after a DoEventTimeShift. Walk top-level beliefs and
        # restamp ``established_at_fabula`` when the belief was
        # acquired via the shifted event.
        beliefs_restamped = 0
        for ent in (entities.values() if isinstance(entities, dict) else (entities or [])):
            for b in (getattr(ent, "beliefs", None) or []):
                if getattr(b, "acquired_via_event_id", None) != target.event_id:
                    continue
                if int(getattr(b, "established_at_fabula", -1)) != old_ft:
                    continue
                try:
                    b.established_at_fabula = new_ft
                    beliefs_restamped += 1
                except Exception:
                    pass
        cascaded += beliefs_restamped

        # Re-stamp CausalEdge.fabula_time on outgoing edges of the
        # shifted event so the causal_topology stays ordered.
        edges_restamped = 0
        for cedge in (self.world_state.causal_topology or []):
            if getattr(cedge, "source_id", None) == target.event_id:
                cedge.fabula_time = new_ft
                edges_restamped += 1

        # Audit (fifth pass, M5): the propagation-delay gate at the
        # top of ``propagate`` reads ``d.get("fabula_time", 0)`` from
        # sandbox edge attrs, not from the canonical CausalEdge. Mirror
        # the restamp onto every outgoing sandbox edge of the shifted
        # event so the gate's ``edge_ft + delay`` arithmetic uses the
        # new tick. Without this, a Do-EventTimeShift that moves an
        # event LATER silently re-fires its (now-prematurely-released)
        # downstream effects, and one that moves it EARLIER fails to
        # release effects whose delay was already satisfied.
        if self.sandbox.has_node(target.event_id):
            for _src, _tgt, _key, _data in list(
                self.sandbox.out_edges(target.event_id, keys=True, data=True)
            ):
                if _data.get("edge_type") == "causal":
                    self.sandbox[_src][_tgt][_key]["fabula_time"] = new_ft

        # Audit (sixth pass, F1): ``Proposition.truth_at_fabula`` is
        # a ``Dict[int, bool]`` keyed by the committing event's
        # ``fabula_time`` (extract_graph reconciler folds Affect
        # commits with ``prop.truth_at_fabula[commit.fabula_time] =
        # commit.truth``; affect_unification synthesises new propos
        # with ``{evt.fabula_time: True}``). If we shift an event
        # that commits a proposition without relocating the key,
        # ``reconstruct_proposition_at(new_ft)`` reports the prop as
        # still-open at the moment the event now fires and
        # ``reconstruct_proposition_at(old_ft)`` reports a phantom
        # commit at a tick when the event no longer occurs. Walk
        # propositions, detect ones this event commits (via
        # ``resolves_proposition_ids`` / ``asserts_proposition_id``
        # / ``denies_proposition_id``), and relocate the ledger key.
        def _evt_commits_prop(_e: Any, _pid: str) -> bool:
            if _pid in (getattr(_e, "resolves_proposition_ids", None) or []):
                return True
            if getattr(_e, "asserts_proposition_id", None) == _pid:
                return True
            if getattr(_e, "denies_proposition_id", None) == _pid:
                return True
            return False

        # Audit (eighth pass, H2): ``Proposition.truth_at_fabula`` is
        # typed ``Dict[int, bool]`` and the model validator coerces
        # str keys to int at construction, but a ledger reconstructed
        # via ``model_construct`` / ``model_copy(update=...)`` / a raw
        # JSON or DB round-trip outside the validator can present
        # str keys. ``old_ft not in ledger`` then silently misses the
        # entry and relocation is skipped — stranding both the
        # primary and the inverse ledger at the old tick after the
        # event has been retimed. Normalize keys to int in-place
        # before the membership check so the relocation is robust to
        # transport-layer key drift.
        def _coerce_ledger_int_keys(_ledger: Any) -> None:
            if not isinstance(_ledger, dict):
                return
            _str_keys = [k for k in list(_ledger.keys()) if not isinstance(k, int)]
            for _k in _str_keys:
                try:
                    _ik = int(_k)
                except (TypeError, ValueError):
                    continue
                if _ik in _ledger and _ledger[_ik] != _ledger[_k]:
                    # Conflict — keep the existing int entry and drop
                    # the duplicate string one.
                    del _ledger[_k]
                    continue
                _ledger[_ik] = _ledger[_k]
                del _ledger[_k]

        truth_relocated = 0
        for prop in (self.world_state.propositions or []):
            _pid = getattr(prop, "proposition_id", None) or getattr(prop, "id", None)
            if _pid is None or not _evt_commits_prop(evt, _pid):
                continue
            ledger = getattr(prop, "truth_at_fabula", None)
            _coerce_ledger_int_keys(ledger)
            if not ledger or old_ft not in ledger:
                continue
            truth_value = ledger[old_ft]
            # If another event at old_ft also commits this prop the
            # old-tick entry still has an anchor — copy forward but
            # do NOT delete the old key.
            other_committer_at_old = False
            for _other in (self.world_state.events or []):
                if _other.id == evt.id:
                    continue
                if int(getattr(_other, "fabula_time", -1)) != old_ft:
                    continue
                if _evt_commits_prop(_other, _pid):
                    other_committer_at_old = True
                    break
            existing_at_new = ledger.get(new_ft)
            if existing_at_new is not None and existing_at_new != truth_value:
                logger.warning(
                    "[CausalPhysics\u00b7do_event_time_shift] Prop %s already "
                    "has truth_at_fabula[%d]=%s conflicting with shifted "
                    "event %s commit %s; leaving ledger untouched.",
                    _pid, new_ft, existing_at_new, target.event_id, truth_value,
                )
                continue
            ledger[new_ft] = truth_value
            if not other_committer_at_old and new_ft != old_ft:
                del ledger[old_ft]
            truth_relocated += 1

            # Audit (seventh pass, B4): inverse-proposition mirror.
            # ``_apply_do_proposition`` and the ingestion writers all
            # mirror committed truth onto ``inverse_proposition_id``
            # with the flipped value. When DoEventTimeShift relocates
            # the primary ledger key, the inverse's ledger entry at
            # ``old_ft`` is stranded — every consumer of the inverse
            # then sees ``not v`` at a tick when the primary is no
            # longer committed and ``None`` at the tick the primary
            # has moved to. Relocate the inverse entry in lockstep.
            inv_pid = getattr(prop, "inverse_proposition_id", None)
            if inv_pid:
                for inv_prop in (self.world_state.propositions or []):
                    if getattr(inv_prop, "proposition_id", None) != inv_pid:
                        continue
                    inv_ledger = getattr(inv_prop, "truth_at_fabula", None)
                    _coerce_ledger_int_keys(inv_ledger)
                    if not inv_ledger or old_ft not in inv_ledger:
                        break
                    inv_val = inv_ledger[old_ft]
                    inv_existing = inv_ledger.get(new_ft)
                    if inv_existing is not None and inv_existing != inv_val:
                        logger.warning(
                            "[CausalPhysics\u00b7do_event_time_shift] Inverse "
                            "prop %s already has truth_at_fabula[%d]=%s "
                            "conflicting with relocated mirror value %s; "
                            "leaving inverse ledger untouched.",
                            inv_pid, new_ft, inv_existing, inv_val,
                        )
                        break
                    inv_ledger[new_ft] = inv_val
                    if not other_committer_at_old and new_ft != old_ft:
                        del inv_ledger[old_ft]
                    truth_relocated += 1
                    break

            # Audit (sixth pass, F6): the ingestion post-pass
            # ``_post_pass_synthesize_audience_beliefs`` keys each
            # synthesised audience belief on ``min(prop.truth_at_fabula)``,
            # i.e. the earliest commit tick. When that tick is now
            # shifted away from ``old_ft``, every belief whose
            # ``proposition_id == _pid`` and
            # ``established_at_fabula == old_ft`` is left pointing at
            # a tick when the prop is no longer committed. Walk all
            # entities (not just AUDIENCE — extractor or affect-pass
            # beliefs can carry the same provenance) and restamp the
            # nested belief tick in lockstep. Only the old_ft entry
            # is touched; beliefs already keyed on a different tick
            # (e.g. a second commit at another time) are left alone.
            if not other_committer_at_old and new_ft != old_ft:
                for _ent in (self.world_state.entities or {}).values():
                    for _b in (getattr(_ent, "beliefs", None) or []):
                        if getattr(_b, "proposition_id", None) != _pid:
                            continue
                        try:
                            if int(getattr(_b, "established_at_fabula", -1)) == old_ft:
                                _b.established_at_fabula = new_ft
                                cascaded += 1
                        except (TypeError, ValueError):
                            continue

        # Best-effort: bump RelationshipMetric.last_updated_fabula on
        # any axis whose most-recent mutation_social edge is the
        # shifted event.
        metrics_restamped = 0
        for rel in (getattr(self.world_state, "social_topology", None) or []):
            for axis_name, metric in (rel.metrics or {}).items():
                # Find the most recent mutation_social edge targeting
                # this dyad+axis. If it matches the shifted event,
                # advance/retract the metric timestamp.
                latest_evt_id = None
                latest_ft = -1
                for cedge in (self.world_state.causal_topology or []):
                    if cedge.causality_type != "mutation_social":
                        continue
                    if cedge.target_id != rel.source_entity_id:
                        continue
                    if cedge.rel_counterpart_id != rel.target_entity_id:
                        continue
                    if cedge.trait_target != axis_name:
                        continue
                    if cedge.fabula_time > latest_ft:
                        latest_ft = cedge.fabula_time
                        latest_evt_id = cedge.source_id
                if latest_evt_id == target.event_id:
                    metric.last_updated_fabula = new_ft
                    metrics_restamped += 1

        logger.info(
            "[CausalPhysics\u00b7do_event_time_shift] Event %s shifted "
            "%d\u2192%d; cascaded %d snapshot(s), %d causal edge(s), "
            "%d social metric(s), %d truth ledger entry(ies).",
            target.event_id, old_ft, new_ft,
            cascaded, edges_restamped, metrics_restamped, truth_relocated,
        )
        self._event_mutations.append(EventMutation(
            event_id=target.event_id,
            kind="time_shift",
            fabula_time=new_ft,
            old_fabula_time=old_ft,
            new_fabula_time=new_ft,
            cascaded_snapshot_restamps=cascaded + truth_relocated,
            cascaded_edge_restamps=edges_restamped + metrics_restamped,
            triggered_by=getattr(target, "triggered_by", None),
        ))

    def _apply_do_relationship(self, target: Any) -> None:
        """Clamp a single per-axis :class:`RelationshipMetric`.

        Spawns a fresh metric record if the named pair has no existing
        edge, so the dyad does not have to be pre-modelled to be
        intervened on.
        """
        from shadow_loom.models import RelationshipMetric, RelationshipEdge
        ft = (
            int(target.fabula_time)
            if getattr(target, "fabula_time", None) is not None
            else self._default_fabula_time()
        )
        # Endpoint validation (round-7 audit fix): refuse to spawn a
        # dangling RelationshipEdge whose endpoints are not real
        # entities — a typo would otherwise leave the canonical
        # ``social_topology`` carrying an unresolvable dyad.
        entities = self.world_state.entities or {}
        if target.source_entity_id not in entities or target.target_entity_id not in entities:
            logger.warning(
                "[CausalPhysics·do_relationship] Skipping clamp — endpoint(s) not in world.entities (%s→%s).",
                target.source_entity_id, target.target_entity_id,
            )
            return
        # Sandbox-side: locate or create the relationship edge attrs.
        # Relationship edges are emitted with ``edge_type='relationship'``
        # by the instantiator and may carry per-axis metrics in either
        # the ``metrics`` dict or the legacy flat keys.
        sb = self.sandbox
        edge_key = None
        sandbox_ok = (
            sb.has_node(target.source_entity_id)
            and sb.has_node(target.target_entity_id)
        )
        if not sandbox_ok:
            # Round-4 audit fix: fail-closed on sandbox endpoint
            # absence. Previously we still mirrored to canonical and
            # incremented ``_edge_do_targets_applied``, producing
            # split-brain (canonical changed, sandbox didn't) and
            # masking vacuity signals because the operation was
            # counted as applied. If the sandbox doesn't carry the
            # dyad, the surgery can't be reasoned about — skip the
            # mirror entirely so downstream vacuity / provenance
            # checks fire correctly.
            logger.warning(
                "[CausalPhysics·do_relationship] Endpoint(s) missing "
                "from sandbox (%s -> %s); fail-closed: canonical "
                "mirror skipped and surgery not counted as applied.",
                target.source_entity_id, target.target_entity_id,
            )
            return
        if sandbox_ok:
            for k, attrs in sb.get_edge_data(
                target.source_entity_id, target.target_entity_id, default={}
            ).items() if sb.has_edge(target.source_entity_id, target.target_entity_id) else []:
                if attrs.get("edge_type") == "relationship":
                    edge_key = k
                    break
            if edge_key is None:
                # No existing relationship \u2014 create one carrying just
                # the clamped metric.
                edge_key = sb.add_edge(
                    target.source_entity_id,
                    target.target_entity_id,
                    edge_type="relationship",
                    source_entity_id=target.source_entity_id,
                    target_entity_id=target.target_entity_id,
                    metrics={},
                )
            attrs = sb[target.source_entity_id][target.target_entity_id][edge_key]
            metrics = attrs.setdefault("metrics", {})
            entry = metrics.get(target.metric) or {}
            old_value = entry.get("value") if entry else None
            entry["value"] = float(target.value)
            entry["observed"] = True
            entry["last_updated_fabula"] = ft
            if target.inertia is not None:
                entry["inertia"] = float(target.inertia)
            metrics[target.metric] = entry
            # Mirror onto the legacy flat key so consumers that only
            # know the legacy shape (renderer dump, audit prompts) see
            # the clamp without metric-dict expansion.
            attrs[target.metric] = float(target.value)
        else:
            # Round-4 audit fix: fail-closed on sandbox endpoint
            # absence. Previously the world-state mirror still ran
            # and ``_edge_do_targets_applied`` was still incremented,
            # producing split-brain (canonical changed, sandbox
            # didn't) and masking vacuity signals because the
            # operation was counted as applied. If the sandbox
            # doesn't carry the dyad, the surgery can't be reasoned
            # about -- skip the canonical mirror entirely so
            # downstream vacuity / provenance checks fire correctly.
            logger.warning(
                "[CausalPhysics\u00b7do_relationship] Endpoint missing from "
                "sandbox (%s -> %s); fail-closed: canonical mirror "
                "skipped and surgery not counted as applied.",
                target.source_entity_id, target.target_entity_id,
            )
            return
        # World-state mirror.
        topology = list(self.world_state.social_topology or [])
        rel = next(
            (r for r in topology
             if r.source_entity_id == target.source_entity_id
             and r.target_entity_id == target.target_entity_id),
            None,
        )
        if rel is None:
            # P0-FIX (P0-9): Set established_at_fabula on creation (CRITICAL-004 audit).
            # Without this timestamp, relationships are visible before they form
            # (time-slice leakage). New relationships must carry creation time.
            rel = RelationshipEdge(
                source_entity_id=target.source_entity_id,
                target_entity_id=target.target_entity_id,
                metrics={},
                established_at_fabula=ft,
            )
            # MODERATE-FIX (M-003): Initialize per-axis inertia for new relationships.
            # Each metric should have proper default inertia instead of using edge-level default.
            rel.metrics = {
                "affinity": RelationshipMetric(
                    value=0.0, inertia=0.3, observed=False, last_updated_fabula=ft
                ),
                "fear": RelationshipMetric(
                    value=0.0, inertia=0.4, observed=False, last_updated_fabula=ft
                ),
                "power_dynamic": RelationshipMetric(
                    value=0.0, inertia=0.5, observed=False, last_updated_fabula=ft
                ),
            }
            topology.append(rel)
            self.world_state.social_topology = topology
        metric_entry = rel.metrics.get(target.metric)
        if metric_entry is None:
            rel.metrics[target.metric] = RelationshipMetric(
                value=float(target.value),
                inertia=float(target.inertia) if target.inertia is not None else 0.3,
                observed=True,
                last_updated_fabula=ft,
            )
        else:
            metric_entry.value = float(target.value)
            metric_entry.observed = True
            metric_entry.last_updated_fabula = ft
            if target.inertia is not None:
                metric_entry.inertia = float(target.inertia)
        # Pin both endpoints so propagation does not silently overwrite
        # the clamped axis on the same step.
        self._intervened_nodes.add(target.source_entity_id)
        self._intervened_nodes.add(target.target_entity_id)
        # Round-4 audit fix: also register the per-axis dyad in the
        # relationship-intervention pin set the legacy path uses.
        # Without this, ``propagate_social`` and other relationship-
        # aware passes treated the typed clamp as ordinary edge data
        # and could overwrite it on the same step.
        self._intervened_relationships.add(
            (target.source_entity_id, target.target_entity_id, target.metric)
        )
        self._edge_do_targets_applied += 1
        # Phase 5b deep audit (DoRelationship Issue C-analogue): emit a
        # SocialMutation record for the direct clamp so the pipeline
        # adapter can fold a snapshot onto the canonical
        # ``RelationshipEdge.metrics`` timeline and the engine-inert
        # guard sees a registered surgery. Without this, a typed
        # relationship clamp wrote canonical+sandbox state but
        # ``result.social_mutations`` stayed empty, leaving downstream
        # consumers blind to the surgery and the inert check unable
        # to distinguish "no-op" from "ran but didn't propagate".
        try:
            old_f = float(old_value) if old_value is not None else float("nan")
        except (TypeError, ValueError):
            old_f = float("nan")
        new_f = float(target.value)
        if math.isnan(old_f) or abs(new_f - old_f) > 1e-12:
            self._social_mutations.append(SocialMutation(
                source_entity_id=target.source_entity_id,
                target_entity_id=target.target_entity_id,
                metric=str(target.metric),
                old_value=old_f,
                new_value=new_f,
                impact=0.0,
                inertia=float(target.inertia) if target.inertia is not None else 0.0,
                triggered_by="DO_OPERATOR",
            ))

    def _apply_do_causal_edge(self, target: Any) -> None:
        """Add or sever a :class:`CausalEdge`.

        ``add`` requires ``causality_type`` and ``mechanism``; missing
        either is logged and skipped. ``sever`` removes every matching
        edge between the named pair on both the sandbox and the
        world-state's ``causal_topology``.
        """
        from shadow_loom.models import CausalEdge
        ft = (
            int(target.fabula_time)
            if getattr(target, "fabula_time", None) is not None
            else self._default_fabula_time()
        )
        if target.action == "sever":
            # Optional ``causality_type`` filter (round-7 audit fix):
            # the EVT_PUSH→ENT_ALICE pair routinely carries multiple
            # edges with different causality_types (e.g. a
            # ``mutation`` arrow for physical harm AND a
            # ``mutation_social`` arrow for the affinity hit). The
            # original sever removed every edge between the pair,
            # silently destroying co-existing causal arrows the caller
            # never asked to sever.
            wanted_ct = getattr(target, "causality_type", None)

            def _ct_match(attrs: dict) -> bool:
                if wanted_ct is None:
                    return True
                return attrs.get("causality_type") == wanted_ct
            # Sandbox side.
            removed_count = 0
            if self.sandbox.has_edge(target.source_id, target.target_id):
                keys_to_drop = [
                    k for k, attrs in self.sandbox[target.source_id][target.target_id].items()
                    if attrs.get("edge_type") == "causal" and _ct_match(attrs)
                ]
                for k in keys_to_drop:
                    self.sandbox.remove_edge(target.source_id, target.target_id, key=k)
                removed_count += len(keys_to_drop)
            # World-state side.
            before = len(self.world_state.causal_topology or [])
            self.world_state.causal_topology = [
                e for e in (self.world_state.causal_topology or [])
                if not (
                    e.source_id == target.source_id
                    and e.target_id == target.target_id
                    and (wanted_ct is None or e.causality_type == wanted_ct)
                )
            ]
            removed_count += before - len(self.world_state.causal_topology)
            # Only count as a real intervention if at least one edge
            # was actually severed; sever-no-match would otherwise
            # bump the counter and make a semantic no-op look like a
            # successful surgery (round-4 audit).
            if removed_count > 0:
                self._edge_do_targets_applied += 1
                self._edge_mutations.append(EdgeMutation(
                    edge_type="causal",
                    action="sever",
                    source_id=target.source_id,
                    target_id=target.target_id,
                    fabula_time=ft,
                    details={
                        "removed_count": removed_count,
                        "causality_type_filter": wanted_ct,
                    },
                ))
            return
        # action == "add"
        if not target.causality_type or not target.mechanism:
            logger.warning(
                "[CausalPhysics\u00b7do_causal_edge] add missing "
                "causality_type/mechanism (%s\u2192%s); skipping.",
                target.source_id, target.target_id,
            )
            return
        try:
            edge = CausalEdge(
                source_id=target.source_id,
                target_id=target.target_id,
                causality_type=target.causality_type,
                causal_force=float(target.causal_force),
                mechanism=target.mechanism,
                fabula_time=ft,
                trait_target=target.trait_target,
                trait_delta=target.trait_delta,
                rel_counterpart_id=target.rel_counterpart_id,
            )
            
            # P0-FIX (P0-6): Validate temporal ordering (CRITICAL-001 audit).
            # Pearl's SCM requires cause to strictly precede effect. Check that
            # source event's fabula_time < target event's fabula_time (when both
            # are events).
            # AUDIT (post-2026-05-26): the previous guard only rejected
            # ``source_ft > target_ft`` which allowed same-tick (``==``) edges.
            # Same-tick causal edges can form directed cycles silently. We now
            # additionally reject ``source_ft == target_ft`` EXCEPT for
            # ``chain_reaction`` causality, where simultaneity is the modelling
            # intent (a stimulus and its immediate observable response sharing
            # one tick).
            source_node = self.sandbox.nodes.get(target.source_id, {})
            target_node = self.sandbox.nodes.get(target.target_id, {})
            source_ft = source_node.get("fabula_time")
            target_ft = target_node.get("fabula_time")
            if source_ft is not None and target_ft is not None:
                if source_ft > target_ft:
                    logger.error(
                        "[CausalPhysics·do_causal_edge] Temporal ordering violation: "
                        "cause %s (ft=%d) occurs AFTER effect %s (ft=%d). Refusing edge.",
                        target.source_id, source_ft, target.target_id, target_ft
                    )
                    return
                if (
                    source_ft == target_ft
                    and target.causality_type != "chain_reaction"
                ):
                    logger.error(
                        "[CausalPhysics·do_causal_edge] Same-tick causal edge "
                        "refused (cause %s ft=%d == effect %s ft=%d, causality_type=%s). "
                        "Only chain_reaction supports simultaneity; use a strictly "
                        "later fabula_time on the effect otherwise.",
                        target.source_id, source_ft, target.target_id, target_ft,
                        target.causality_type,
                    )
                    return
                    
            # P0-FIX (P0-7): Validate rel_counterpart_id exists (CRITICAL-002 audit).
            # For mutation_social edges, validate that the counterpart entity exists
            # to prevent orphaned references.
            if target.rel_counterpart_id:
                if target.rel_counterpart_id not in self.world_state.entities:
                    logger.error(
                        "[CausalPhysics·do_causal_edge] rel_counterpart_id=%s not found "
                        "in entities. Refusing mutation_social edge %s→%s.",
                        target.rel_counterpart_id, target.source_id, target.target_id
                    )
                    return
            
            # MODERATE-FIX (Graph Topology CRITICAL-007): Validate mechanism-trait mapping.
            # When trait_target is specified, ensure it's compatible with the mechanism.
            # Prevents silent propagation failures where edge fires but doesn't mutate.
            if target.trait_target and target.mechanism:
                valid_traits = MECHANISM_TRAIT_MAP.get(target.mechanism)
                if valid_traits is not None and target.trait_target not in valid_traits:
                    logger.warning(
                        "[CausalPhysics·do_causal_edge] Mechanism-trait mismatch: "
                        "mechanism=%s does not map to trait_target=%s (valid: %s). "
                        "Edge %s→%s may not propagate correctly.",
                        target.mechanism, target.trait_target, valid_traits,
                        target.source_id, target.target_id
                    )
                    # Don't refuse the edge - just warn. Mechanism might be valid but
                    # MECHANISM_TRAIT_MAP incomplete, or this is an exotic edge type.
            
            # MODERATE-FIX (M-006): Inherit evidence_strength from source EventNode.
            # Maintains consistency between event confidence and causal edge weights.
            if source_node.get("node_type") == "EventNode":
                source_evidence = source_node.get("evidence_strength", "moderate")
                if not target.evidence_strength or target.evidence_strength == "moderate":
                    # Inherit from source event if not explicitly set
                    edge.evidence_strength = source_evidence
                    logger.debug(
                        "[CausalPhysics·do_causal_edge] Inherited evidence_strength=%s from %s",
                        source_evidence, target.source_id
                    )
                    
        except Exception:
            logger.exception(
                "[CausalPhysics\u00b7do_causal_edge] CausalEdge validation failed for %s\u2192%s.",
                target.source_id, target.target_id,
            )
            return
        # Round-4 audit fix: validate canonical endpoints BEFORE
        # mutating the sandbox. Previously the sandbox add ran first;
        # if the canonical check below early-returned, the sandbox
        # edge persisted until reload and reasoning saw a phantom
        # causal arrow that no merge would ever surface.
        if not self._world_knows_node(target.source_id) or not self._world_knows_node(target.target_id):
            logger.warning(
                "[CausalPhysics·do_causal_edge] add skipped — endpoint(s) unknown to canonical world (%s→%s).",
                target.source_id, target.target_id,
            )
            return
        # Sandbox side.
        if self.sandbox.has_node(target.source_id) and self.sandbox.has_node(target.target_id):
            self.sandbox.add_edge(
                target.source_id, target.target_id,
                edge_type="causal",
                **edge.model_dump(),
            )
        topology = list(self.world_state.causal_topology or [])
        topology.append(edge)
        self.world_state.causal_topology = topology
        self._edge_do_targets_applied += 1
        self._edge_mutations.append(EdgeMutation(
            edge_type="causal",
            action="add",
            source_id=target.source_id,
            target_id=target.target_id,
            fabula_time=ft,
            details={
                "causality_type": target.causality_type,
                "mechanism": target.mechanism,
                "causal_force": float(target.causal_force),
                "trait_target": target.trait_target,
            },
        ))

    def _apply_do_spatial_edge(self, target: Any) -> None:
        """Add, sever, or lock-toggle a :class:`SpatialEdge`."""
        from shadow_loom.models import SpatialEdge
        ft = (
            int(target.fabula_time)
            if getattr(target, "fabula_time", None) is not None
            else self._default_fabula_time()
        )
        sb = self.sandbox

        def _matching_edge_keys() -> list:
            if not sb.has_edge(target.source_id, target.target_id):
                return []
            return [
                k for k, attrs in sb[target.source_id][target.target_id].items()
                if attrs.get("edge_type") == "connected_to"
            ]

        if target.action == "sever":
            matched = _matching_edge_keys()
            for k in matched:
                sb.remove_edge(target.source_id, target.target_id, key=k)
            
            # MODERATE-FIX (Graph Topology MODERATE-002): Mark destroyed_at_fabula
            # on canonical topology edges to distinguish permanent vs temporary 
            # passage removal. This allows downstream reasoning about whether a
            # severed passage can be restored.
            # Round-4 audit fix: count tombstones marked here so the
            # rebuilds below can be removed entirely without losing
            # the application-count signal. Filtering the topology
            # discards ``destroyed_at_fabula`` along with the edge,
            # which contradicts the tombstone semantics extract_graph
            # and instantiator rely on.
            _topo_removed = 0
            for e in (self.world_state.spatial_topology or []):
                if (e.source_id == target.source_id and e.target_id == target.target_id) or \
                   (e.source_id == target.target_id and e.target_id == target.source_id and 
                    getattr(e, "bidirectional", False)):
                    if getattr(e, "destroyed_at_fabula", None) is None:
                        e.destroyed_at_fabula = ft
                        _topo_removed += 1
            
            # Bidirectional sever (round-7 audit fix). The ``add``
            # branch below registers the reverse arrow for
            # bidirectional connections; an asymmetric sever leaves
            # the canonical world traversable B→A even after A→B was
            # cut, contradicting the renderer's view of the world.
            # Mirror the cut on the reverse edge for every direction
            # whose canonical record was bidirectional.
            reverse_was_bidi = any(
                e.source_id == target.target_id
                and e.target_id == target.source_id
                and getattr(e, "bidirectional", False)
                for e in (self.world_state.spatial_topology or [])
            )
            forward_was_bidi = any(
                e.source_id == target.source_id
                and e.target_id == target.target_id
                and getattr(e, "bidirectional", False)
                for e in (self.world_state.spatial_topology or [])
            )
            # Round-7 audit: capture pre-removal counts so a sever
            # against an already-missing edge (vacuous no-op) does
            # not inflate ``_edge_do_targets_applied``.
            _rev_removed = 0
            if reverse_was_bidi or forward_was_bidi:
                if sb.has_edge(target.target_id, target.source_id):
                    rev_keys = [
                        k for k, attrs in sb[target.target_id][target.source_id].items()
                        if attrs.get("edge_type") == "connected_to"
                    ]
                    _rev_removed = len(rev_keys)
                    for k in rev_keys:
                        sb.remove_edge(target.target_id, target.source_id, key=k)
            if matched or _rev_removed or _topo_removed:
                self._edge_do_targets_applied += 1
                self._edge_mutations.append(EdgeMutation(
                    edge_type="spatial",
                    action="sever",
                    source_id=target.source_id,
                    target_id=target.target_id,
                    fabula_time=ft,
                    details={
                        "sandbox_removed": matched,
                        "reverse_removed": _rev_removed,
                        "topology_tombstoned": _topo_removed,
                    },
                ))
            return

        if target.action in ("lock", "unlock"):
            new_locked = target.action == "lock"
            matched_keys = _matching_edge_keys()
            for k in matched_keys:
                attrs = sb[target.source_id][target.target_id][k]
                attrs["is_locked"] = new_locked
                if new_locked and target.barrier_item_id:
                    attrs["barrier_item_id"] = target.barrier_item_id
            canonical_hit = False
            forward_was_bidi = False
            for e in (self.world_state.spatial_topology or []):
                if e.source_id == target.source_id and e.target_id == target.target_id:
                    e.is_locked = new_locked
                    if new_locked and target.barrier_item_id:
                        e.barrier_item_id = target.barrier_item_id
                    canonical_hit = True
                    if getattr(e, "bidirectional", False):
                        forward_was_bidi = True
            # Round-4 audit: mirror lock/unlock onto the reverse
            # canonical edge when the forward edge is bidirectional.
            # The ``sever`` branch above already mirrors removal for
            # bidirectional connections; without mirroring lock-state
            # too, locking a bidirectional passage (e.g. Macbeth's
            # LOC_INVERNESS_CASTLE ↔ LOC_FORRES_COURT after the
            # murder, sealed by a guard cordon) leaves B→A traversable
            # while A→B is sealed, contradicting the renderer's view
            # of a symmetric barrier.
            reverse_hit = False
            if forward_was_bidi:
                for rev_e in (self.world_state.spatial_topology or []):
                    if (rev_e.source_id == target.target_id
                            and rev_e.target_id == target.source_id):
                        rev_e.is_locked = new_locked
                        if new_locked and target.barrier_item_id:
                            rev_e.barrier_item_id = target.barrier_item_id
                        reverse_hit = True
                # Mirror the sandbox lock-state too so propagation
                # within the same step sees the symmetric barrier.
                if sb.has_edge(target.target_id, target.source_id):
                    rev_keys = [
                        k for k, attrs in sb[target.target_id][target.source_id].items()
                        if attrs.get("edge_type") == "connected_to"
                    ]
                    for k in rev_keys:
                        rev_attrs = sb[target.target_id][target.source_id][k]
                        rev_attrs["is_locked"] = new_locked
                        if new_locked and target.barrier_item_id:
                            rev_attrs["barrier_item_id"] = target.barrier_item_id
            if matched_keys or canonical_hit:
                self._edge_do_targets_applied += 1
                self._edge_mutations.append(EdgeMutation(
                    edge_type="spatial",
                    action=target.action,
                    source_id=target.source_id,
                    target_id=target.target_id,
                    fabula_time=ft,
                    details={
                        "is_locked": new_locked,
                        "barrier_item_id": target.barrier_item_id,
                        "sandbox_edges_touched": len(matched_keys),
                        "canonical_hit": canonical_hit,
                        "reverse_hit": reverse_hit,
                    },
                ))
            return

        # action == "add"
        try:
            edge = SpatialEdge(
                source_id=target.source_id,
                target_id=target.target_id,
                connection_type=target.connection_type or "passage",
                bidirectional=bool(target.bidirectional),
                is_locked=False,
                barrier_item_id=target.barrier_item_id,
                established_at_fabula=ft,
            )
        except Exception:
            logger.exception(
                "[CausalPhysics\u00b7do_spatial_edge] SpatialEdge validation "
                "failed for %s\u2192%s.", target.source_id, target.target_id,
            )
            return
        
        # MODERATE-FIX (M-001): Validate barrier_item_id references existing object.
        # Prevents orphaned affordance gates that reference non-existent items.
        if edge.barrier_item_id:
            if edge.barrier_item_id not in (self.world_state.objects or {}):
                logger.warning(
                    "[CausalPhysics·do_spatial_edge] barrier_item_id=%s not found "
                    "in objects. Clearing affordance lock.",
                    edge.barrier_item_id
                )
                edge.barrier_item_id = None
            
        # P0-FIX (P0-8): Validate locations BEFORE adding to sandbox (CRITICAL-003 audit).
        # Moving validation before sb.add_edge prevents orphaned edges from persisting
        # in sandbox when validation fails. Original code added edge first, validated
        # second, leaving phantom passages on failure.
        locations = self.world_state.locations or {}
        if target.source_id not in locations or target.target_id not in locations:
            logger.warning(
                "[CausalPhysics·do_spatial_edge] add skipped — endpoint(s) not in world.locations (%s→%s).",
                target.source_id, target.target_id,
            )
            return
            
        # Validation passed - now safe to add to sandbox
        if sb.has_node(target.source_id) and sb.has_node(target.target_id):
            sb.add_edge(
                target.source_id, target.target_id,
                edge_type="connected_to",
                **edge.model_dump(),
            )
            if edge.bidirectional:
                sb.add_edge(
                    target.target_id, target.source_id,
                    edge_type="connected_to",
                    **edge.model_dump(),
                )
        # World-state side. Edge already validated above.
        topology = list(self.world_state.spatial_topology or [])
        topology.append(edge)
        self.world_state.spatial_topology = topology
        self._edge_do_targets_applied += 1
        self._edge_mutations.append(EdgeMutation(
            edge_type="spatial",
            action="add",
            source_id=target.source_id,
            target_id=target.target_id,
            fabula_time=ft,
            details={
                "connection_type": edge.connection_type,
                "bidirectional": bool(edge.bidirectional),
                "barrier_item_id": edge.barrier_item_id,
            },
        ))

    def _apply_do_object(self, target: Any) -> None:
        """Clamp a :class:`NarrativeObject`'s position / ownership / properties.

        Mutates the sandbox OBJ_ node in-place (so the same propagate
        step sees the new location / owner / properties) and records
        an :class:`ObjectMutation` for the pipeline adapter to fold
        into a :class:`ObjectStateSnapshot` on the canonical
        :attr:`NarrativeObject.state_timeline` via the
        :class:`~shadow_loom.ingestion.ObjectUpdate` bridge. Also
        mirrors the change onto :attr:`WorldStateV1.objects` so
        downstream readers consulting the world directly observe the
        clamp without re-running ingestion.

        ``set_*_null`` flags clear the corresponding field; otherwise
        an explicit ``new_*`` value overwrites and a missing one is
        a no-op (matching :class:`ObjectStateSnapshot` semantics).
        Pins the object id in ``_intervened_nodes`` so cascading
        physics treats the surgery as authoritative for this step.
        """
        ft = (
            int(target.fabula_time)
            if getattr(target, "fabula_time", None) is not None
            else self._default_fabula_time()
        )
        # Reference validation (round-7 audit fix). A typoed LOC_ /
        # ENT_ id would silently pin the object at a phantom
        # location/owner, leaving affordance gates unresolvable and
        # the renderer's place-the-object prompt dangling.
        locations = self.world_state.locations or {}
        entities = self.world_state.entities or {}
        if target.new_location_id is not None and target.new_location_id not in locations:
            logger.warning(
                "[CausalPhysics·do_object] new_location_id %r not in world.locations; skipping object %s.",
                target.new_location_id, target.object_id,
            )
            return
        if target.new_owner_id is not None and target.new_owner_id not in entities:
            logger.warning(
                "[CausalPhysics·do_object] new_owner_id %r not in world.entities; skipping object %s.",
                target.new_owner_id, target.object_id,
            )
            return
        # Sandbox-side mutation.
        if self.sandbox.has_node(target.object_id):
            ndata = self.sandbox.nodes[target.object_id]
            # NOTE: owner / location sandbox edits are intentionally
            # left to ``AMWNInstantiator.execute_interventions`` (which
            # ``engine.apply_do_operator`` invokes immediately after
            # ``apply_do_targets``). The legacy ``_intervene_inventory``
            # / ``_intervene_spatial`` paths capture the OLD owner /
            # location, rewrite the ``owned_by`` / ``located_in`` edge
            # topology, and resolve the drop-at-owner's-room semantics.
            # Mirroring those writes here would null ``owner_id`` /
            # ``location_id`` before the legacy surgery runs, breaking
            # the drop-after-teleport invariant (the dropped object
            # would settle at the dropper's PREVIOUS room because
            # ``_intervene_inventory`` could no longer recover the
            # owner's current location). Properties remain handled
            # here — execute_interventions has no ``properties.*``
            # dispatcher.
            if target.properties_set or target.properties_unset:
                props = dict(ndata.get("properties") or {})
                for k in target.properties_unset:
                    props.pop(k, None)
                for k, v in target.properties_set.items():
                    props[k] = str(v)
                ndata["properties"] = props
            self._intervened_nodes.add(target.object_id)
        else:
            logger.warning(
                "[CausalPhysics\u00b7do_object] Object %s missing from sandbox; "
                "world-state mirror still applied.", target.object_id,
            )
        # World-state mirror (so directive assembly / re-extraction see it).
        canonical = (self.world_state.objects or {}).get(target.object_id)
        if canonical is not None:
            if target.set_location_null:
                canonical.location_id = None
            elif target.new_location_id is not None:
                canonical.location_id = target.new_location_id
            if target.set_owner_null:
                canonical.owner_id = None
            elif target.new_owner_id is not None:
                canonical.owner_id = target.new_owner_id
            if target.properties_set or target.properties_unset:
                props = dict(canonical.properties or {})
                for k in target.properties_unset:
                    props.pop(k, None)
                for k, v in target.properties_set.items():
                    props[k] = str(v)
                canonical.properties = props

        # Fail-closed: if neither the sandbox graph nor the canonical
        # NarrativeObject knows this id, no mutation actually landed
        # \u2014 emitting an ObjectMutation would surface a phantom
        # cascade entry and suppress the vacuity guard. Skip the
        # append (the sandbox-missing logger.warning above is the
        # single signal).
        in_sandbox = target.object_id in self.sandbox.nodes
        if canonical is None and not in_sandbox:
            return

        self._object_mutations.append(ObjectMutation(
            object_id=target.object_id,
            fabula_time=ft,
            new_location_id=target.new_location_id,
            new_owner_id=target.new_owner_id,
            set_location_null=bool(target.set_location_null),
            set_owner_null=bool(target.set_owner_null),
            properties_set=dict(target.properties_set or {}),
            properties_unset=list(target.properties_unset or []),
            triggered_by=getattr(target, "triggered_by", None),
        ))
        
        # P1-FIX: Update object state_timeline for reconstruction.
        # AUDIT P0-4: ObjectStateSnapshot has no ``properties`` field; the
        # snapshot is a DIFF (``properties_set`` / ``properties_unset``)
        # plus location/owner changes from the incoming DoTarget. Tag the
        # snapshot with the active sandbox branch so consumers walking the
        # timeline can filter off-branch entries.
        if canonical is not None:
            from shadow_loom.models import ObjectStateSnapshot
            branch_world_id = self.sandbox.graph.get("world_id", "factual")
            if branch_world_id not in ("factual", "shadow"):
                branch_world_id = "factual"
            snapshot = ObjectStateSnapshot(
                world_id=branch_world_id,
                fabula_time=ft,
                triggered_by=getattr(target, "triggered_by", None) or "DO_OPERATOR",
                location_id=target.new_location_id,
                owner_id=target.new_owner_id,
                set_location_null=bool(target.set_location_null),
                set_owner_null=bool(target.set_owner_null),
                properties_set=dict(target.properties_set or {}),
                properties_unset=list(target.properties_unset or []),
            )
            canonical.state_timeline.append(snapshot)
            # Keep object timelines fabula-ordered so reconstruction is
            # deterministic (matches entity relocation path).
            canonical.state_timeline.sort(key=lambda s: s.fabula_time)

    def _apply_do_entity_delete(self, target: Any) -> None:
        """Excise an :class:`Entity` from the world ("never existed").

        Strict Pearl Rung-3 existence counterfactual. Removes the
        entity from :attr:`WorldStateV1.entities` and cascades:

        * drop every :class:`RelationshipEdge` naming the entity as
          source or target;
        * drop every belief in surviving entities whose ``target_id``
          is the deleted entity;
        * drop every concern owned by the deleted entity;
        * drop every causal edge whose source or target references the
          deleted entity;
        * strip the entity id from every surviving event's
          ``actor_ids``;
        * clear ``speaker_id`` on events where the deleted entity was
          the sole speaker;
        * mirror by removing the sandbox node (which incidentally
          drops all incident sandbox edges).

        Events with no remaining ``actor_ids`` (and no speaker) survive
        for audit visibility — operators must clamp ``occurred=False``
        on those explicitly if they want them suppressed. This keeps
        the surgery referent-safe without silently rewriting plot
        causality.
        """
        eid = target.entity_id
        ws = self.world_state
        entities = ws.entities or {}
        if eid not in entities:
            logger.warning(
                "[CausalPhysics\u00b7do_entity_delete] %s not in world.entities; skipping.",
                eid,
            )
            return
        del entities[eid]
        # Social topology.
        before_s = len(ws.social_topology or [])
        ws.social_topology = [
            r for r in (ws.social_topology or [])
            if r.source_entity_id != eid and r.target_entity_id != eid
        ]
        # Beliefs targeting the deleted entity in survivors.
        _beliefs_removed = 0
        for other in entities.values():
            before_b = len(other.beliefs or [])
            other.beliefs = [b for b in (other.beliefs or []) if b.target_id != eid]
            _beliefs_removed += before_b - len(other.beliefs)
        # Concerns owned by the entity (concerns live on entities;
        # popping the entity already dropped its concerns, but if any
        # downstream container holds dangling refs we'd cascade here).
        # Causal topology.
        before_c = len(ws.causal_topology or [])
        ws.causal_topology = [
            ce for ce in (ws.causal_topology or [])
            if ce.source_id != eid and ce.target_id != eid
        ]
        # Events: scrub from actor_ids / speaker_id.
        _events_scrubbed = 0
        for evt in (ws.events or []):
            touched = False
            if eid in (evt.actor_ids or []):
                evt.actor_ids = [a for a in evt.actor_ids if a != eid]
                touched = True
            if getattr(evt, "speaker_id", None) == eid:
                evt.speaker_id = None
                touched = True
            for fld in ("target_ids", "addressee_ids"):
                lst = getattr(evt, fld, None)
                if lst and eid in lst:
                    setattr(evt, fld, [x for x in lst if x != eid])
                    touched = True
            if touched:
                _events_scrubbed += 1
        # Propositions: scrub referent ids.
        for p in (ws.propositions or []):
            refs = getattr(p, "referent_ids", None)
            if refs and eid in refs:
                p.referent_ids = [r for r in refs if r != eid]
        # Channels: scrub participant_ids (Channel.participant_ids).
        for ch in (ws.channels or {}).values():
            parts = getattr(ch, "participant_ids", None)
            if parts and eid in parts:
                ch.participant_ids = [p for p in parts if p != eid]
        # Sandbox mirror.
        if self.sandbox.has_node(eid):
            self.sandbox.remove_node(eid)
        logger.info(
            "[CausalPhysics\u00b7do_entity_delete] Excised %s "
            "(social\u2212%d, causal\u2212%d).",
            eid, before_s - len(ws.social_topology),
            before_c - len(ws.causal_topology),
        )
        self._intervened_nodes.add(eid)
        # Structured record so the renderer / auditor / MCP envelope
        # can see the excision and ground prose against the cascade
        # rather than treat the entity as a phantom-mention candidate.
        self._entity_delete_mutations.append(EntityDeleteMutation(
            entity_id=eid,
            fabula_time=getattr(target, "fabula_time", None) or self._default_fabula_time(),
            cascaded_social_edges_removed=before_s - len(ws.social_topology),
            cascaded_causal_edges_removed=before_c - len(ws.causal_topology),
            cascaded_beliefs_removed=_beliefs_removed,
            cascaded_events_scrubbed=_events_scrubbed,
            triggered_by=getattr(target, "triggered_by", None),
        ))

    def _apply_do_object_delete(self, target: Any) -> None:
        """Excise a :class:`NarrativeObject` from the world ("never existed").

        Mirror of :meth:`_apply_do_entity_delete` for narrative
        objects. Cascades:

        * drop the object from :attr:`WorldStateV1.objects`;
        * scrub the object id from channel participant lists (an
          object-bearing channel like a phone-line collapses if its
          only handset is excised);
        * strip the object id from every event's ``object_ids`` (when
          present) so events that depended on the object survive but
          stop referencing it;
        * drop causal edges naming the object as source or target;
        * drop beliefs in any entity whose ``target_id`` is the
          deleted object;
        * mirror via sandbox node removal.

        As with entity deletion, events are not implicitly suppressed
        — operators clamp ``occurred=False`` themselves when the plot
        action depended on the missing prop.
        """
        oid = target.object_id
        ws = self.world_state
        objects = ws.objects or {}
        if oid not in objects:
            logger.warning(
                "[CausalPhysics\u00b7do_object_delete] %s not in world.objects; skipping.",
                oid,
            )
            return
        del objects[oid]
        # Channels: scrub participant_ids (Channel.participant_ids).
        for ch in (ws.channels or {}).values():
            participants = getattr(ch, "participant_ids", None)
            if participants and oid in participants:
                ch.participant_ids = [p for p in participants if p != oid]
        # Events: scrub object_ids if the field exists.
        _events_scrubbed = 0
        for evt in (ws.events or []):
            touched = False
            obj_ids = getattr(evt, "object_ids", None)
            if obj_ids and oid in obj_ids:
                evt.object_ids = [o for o in obj_ids if o != oid]
                touched = True
            for fld in ("target_ids",):
                lst = getattr(evt, fld, None)
                if lst and oid in lst:
                    setattr(evt, fld, [x for x in lst if x != oid])
                    touched = True
            if touched:
                _events_scrubbed += 1
        # Propositions: scrub referent ids.
        for p in (ws.propositions or []):
            refs = getattr(p, "referent_ids", None)
            if refs and oid in refs:
                p.referent_ids = [r for r in refs if r != oid]
        # Causal topology.
        before_c = len(ws.causal_topology or [])
        ws.causal_topology = [
            ce for ce in (ws.causal_topology or [])
            if ce.source_id != oid and ce.target_id != oid
        ]
        # Audit (eighth pass, M5): spatial edges may carry a
        # ``barrier_item_id`` pointing at the deleted object (a
        # locked door's key, a dragon guarding a pass, the One Ring
        # blocking egress). When the object is excised the lock
        # becomes a ghost barrier — traversal still appears blocked
        # by a non-existent item. Scrub the reference and normalize
        # the lock state: if ``is_locked`` was True solely because
        # of this item, leave the edge locked but barrier-less so
        # downstream readers can still see the original intent
        # while the dangling id is cleared.
        _spatial_scrubbed = 0
        for se in (ws.spatial_topology or []):
            if getattr(se, "barrier_item_id", None) == oid:
                se.barrier_item_id = None
                _spatial_scrubbed += 1
        # Beliefs targeting the object.
        _beliefs_removed = 0
        for ent in (ws.entities or {}).values():
            before_b = len(ent.beliefs or [])
            ent.beliefs = [b for b in (ent.beliefs or []) if b.target_id != oid]
            _beliefs_removed += before_b - len(ent.beliefs)
        # Sandbox mirror.
        if self.sandbox.has_node(oid):
            self.sandbox.remove_node(oid)
        logger.info(
            "[CausalPhysics\u00b7do_object_delete] Excised %s (causal\u2212%d).",
            oid, before_c - len(ws.causal_topology),
        )
        self._intervened_nodes.add(oid)
        # Structured record (mirror of EntityDeleteMutation).
        self._object_delete_mutations.append(ObjectDeleteMutation(
            object_id=oid,
            fabula_time=getattr(target, "fabula_time", None) or self._default_fabula_time(),
            cascaded_causal_edges_removed=before_c - len(ws.causal_topology),
            cascaded_beliefs_removed=_beliefs_removed,
            cascaded_events_scrubbed=_events_scrubbed,
            triggered_by=getattr(target, "triggered_by", None),
        ))

    def _default_fabula_time(self) -> int:
        """Best-effort fabula_time anchor when a DoTarget omits one.

        Falls back through (sandbox.graph['fabula_anchor'],
        world_state.global_anchor.fabula_time, 0) so the dispatcher does
        not require an explicit anchor on every typed target.
        """
        ft = self.sandbox.graph.get("fabula_anchor")
        if ft is not None:
            return int(ft)
        anchor = getattr(self.world_state, "global_anchor", None)
        ft = getattr(anchor, "fabula_time", None) if anchor is not None else None
        return int(ft) if ft is not None else 0

    def _world_knows_node(self, node_id: str) -> bool:
        """True iff ``node_id`` resolves to an EVT_/ENT_/OBJ_/LOC_/WORLD_
        node that already exists on the canonical
        :class:`WorldStateV1`. Used by the ``do_causal_edge`` add path
        to refuse persisting edges with dangling endpoints.
        """
        if not node_id:
            return False
        ws = self.world_state
        if node_id.startswith("EVT_"):
            return any(e.id == node_id for e in (ws.events or []))
        if node_id.startswith("ENT_"):
            return node_id in (ws.entities or {})
        if node_id.startswith("OBJ_"):
            return node_id in (ws.objects or {})
        if node_id.startswith("LOC_"):
            return node_id in (ws.locations or {})
        if node_id.startswith("WORLD_"):
            return node_id in (ws.world_traits or {})
        # Unknown prefix \u2014 conservatively reject so we never persist
        # an off-schema endpoint into the canonical topology.
        return False

    # ------------------------------------------------------------------
    # Provenance invalidation
    # ------------------------------------------------------------------
    def _collect_provenance_invalidations(
        self, interventions: Dict[str, Any],
    ) -> tuple[set[str], set[str]]:
        """Identify event/channel IDs that the do-surgery has rendered
        epistemically inert.

        Returns ``(removed_event_ids, removed_channel_ids)``.

        Heuristics, kept conservative (we only remove provenance when
        the surgery is unambiguously *destructive*; mere relabelling
        such as ``EVT_X.outcome="reworded"`` does not invalidate
        provenance):

        * Event ID ``EVT_*`` is removed when:
            - ``EVT_X.event_type`` is set to ``"prevented"``,
              ``"never_happened"``, or ``"removed"``;
            - ``EVT_X.truth_value`` is set to ``"false"`` or
              ``"performative"`` (utterance carries no factual signal).
        * Channel ID ``CHAN_*`` is removed when:
            - ``CHAN_Y.status`` is set to ``"severed"`` /
              ``"disabled"`` / ``"down"``;
            - ``CHAN_Y.participant_ids`` is set to ``[]`` (no listeners
              left, so the channel cannot deliver anything).
        * Entity-level ``ENT_X.communicating_with`` surgery is handled
          inline by ``AMWNInstantiator._intervene_comms``; we don't
          re-derive its prune set here.
        """
        removed_events: set[str] = set()
        removed_channels: set[str] = set()
        DESTRUCTIVE_EVENT_TYPES = {"prevented", "never_happened", "removed"}
        DESTRUCTIVE_TRUTH = {"false", "performative"}
        DESTRUCTIVE_STATUS = {"severed", "disabled", "down"}

        for path, value in (interventions or {}).items():
            if "." not in path:
                continue
            node_id, prop = path.split(".", 1)
            if node_id.startswith("EVT_"):
                if prop == "event_type" and isinstance(value, str) and value in DESTRUCTIVE_EVENT_TYPES:
                    removed_events.add(node_id)
                elif prop == "truth_value" and isinstance(value, str) and value in DESTRUCTIVE_TRUTH:
                    removed_events.add(node_id)
            elif node_id.startswith(("CHN_", "CHAN_")):
                if prop == "status" and isinstance(value, str) and value in DESTRUCTIVE_STATUS:
                    removed_channels.add(node_id)
                elif prop == "participant_ids" and isinstance(value, list) and len(value) == 0:
                    removed_channels.add(node_id)
                # Audit R18-12: typed ``DoChannel`` mirrors deactivation
                # as ``CHN_X.active = False`` (see ``_typed_target_payload``
                # at L1335). Without this branch the canonical
                # severance never reaches ``disabled_channel_ids`` and
                # belief-provenance pruning silently no-ops.
                elif prop == "active" and value is False:
                    removed_channels.add(node_id)
        return removed_events, removed_channels

    def _mirror_belief_pruning_to_canonical(
        self,
        *,
        removed_event_ids: Optional[set[str]] = None,
        removed_channel_ids: Optional[set[str]] = None,
        severed_speaker_addressee_pairs: Optional[set[tuple[str, str]]] = None,
    ) -> int:
        """Mirror sandbox belief-provenance pruning onto ``world_state``.

        ``AMWNInstantiator._prune_beliefs_by_provenance`` mutates the
        sandbox graph in place. For the canonical ``self.world_state``
        view to stay consistent (so downstream readers — auditors,
        scorers, MCP queries — see the same beliefs the engine reasoned
        over), the same predicate must be applied to
        :class:`Entity.beliefs` and every
        :class:`EntityStateSnapshot.beliefs_added` in
        ``state_timeline``.

        Returns the number of canonical beliefs removed.
        """
        removed_event_ids = set(removed_event_ids or ())
        removed_channel_ids = set(removed_channel_ids or ())
        severed_pairs = set(severed_speaker_addressee_pairs or ())

        if not (removed_event_ids or removed_channel_ids or severed_pairs):
            return 0

        # Resolve utterance speakers from the canonical event list so the
        # third match condition (severed speaker→addressee dyad) works
        # without having to walk the sandbox.
        utterance_speaker: Dict[str, str] = {}
        for evt in self.world_state.events or []:
            if getattr(evt, "event_type", None) == "utterance":
                sp = getattr(evt, "speaker_id", None)
                if sp:
                    utterance_speaker[evt.id] = sp

        def _is_dangling(belief: Any, holder_id: str) -> bool:
            ev = getattr(belief, "acquired_via_event_id", None)
            ch = getattr(belief, "acquired_via_channel_id", None)
            if ev and ev in removed_event_ids:
                return True
            if ch and ch in removed_channel_ids:
                return True
            if ev and ev in utterance_speaker:
                if (utterance_speaker[ev], holder_id) in severed_pairs:
                    return True
            return False

        pruned = 0
        for ent_id, ent in (self.world_state.entities or {}).items():
            beliefs = getattr(ent, "beliefs", None)
            if isinstance(beliefs, list):
                kept = []
                for b in beliefs:
                    if _is_dangling(b, ent_id):
                        pruned += 1
                        continue
                    kept.append(b)
                if len(kept) != len(beliefs):
                    ent.beliefs = kept
            timeline = getattr(ent, "state_timeline", None)
            if isinstance(timeline, list):
                for snap in timeline:
                    added = getattr(snap, "beliefs_added", None)
                    if not isinstance(added, list):
                        continue
                    kept_added = []
                    for b in added:
                        if _is_dangling(b, ent_id):
                            pruned += 1
                            continue
                        kept_added.append(b)
                    if len(kept_added) != len(added):
                        snap.beliefs_added = kept_added

        if pruned:
            logger.debug(
                "[CausalPhysics·_mirror_belief_pruning_to_canonical] "
                "Pruned %d canonical belief(s) (events=%d channels=%d pairs=%d).",
                pruned, len(removed_event_ids), len(removed_channel_ids),
                len(severed_pairs),
            )

        # AUDIT round-2 P0-1: canonical-side event-removal cascade.
        # When an event is destructively removed from the canonical
        # ``world_state.events`` list, also drop references to that
        # event id from propositions and causal_topology so downstream
        # readers (auditor, posterior, affective scorers, MCP queries)
        # don't see dangling refs. We gate strictly on ``world_state``
        # membership: sandbox-pruned events that remain in canonical
        # (e.g. counterfactual prevention markers) keep their canonical
        # edges intact so the shared fixture isn't structurally
        # damaged for downstream tests/readers.
        if removed_event_ids:
            canonical_event_ids = {
                getattr(e, "id", None) for e in (self.world_state.events or [])
            }
            truly_removed = {
                eid for eid in removed_event_ids if eid not in canonical_event_ids
            }
            if truly_removed:
                cleared_refs = 0
                cleared_topo = 0
                for prop in (self.world_state.propositions or []):
                    refs = getattr(prop, "referent_ids", None) or []
                    if any(r in truly_removed for r in refs):
                        prop.referent_ids = [r for r in refs if r not in truly_removed]
                        cleared_refs += 1
                topo = list(getattr(self.world_state, "causal_topology", None) or [])
                kept = []
                for ce in topo:
                    src = getattr(ce, "source_id", None)
                    tgt = getattr(ce, "target_id", None)
                    if (src in truly_removed) or (tgt in truly_removed):
                        cleared_topo += 1
                        continue
                    kept.append(ce)
                if cleared_topo:
                    self.world_state.causal_topology = kept
                if cleared_refs or cleared_topo:
                    logger.debug(
                        "[CausalPhysics\u00b7event-cascade] truly_removed=%d \u2192 "
                        "cleared_proposition_refs=%d, cleared_causal_edges=%d.",
                        len(truly_removed), cleared_refs, cleared_topo,
                    )

        return pruned

    def _perform_graph_surgery_edge_removal(
        self,
        interventions: Dict[str, Any],
        pruned_event_ids: set[str],
    ) -> None:
        """Remove incoming causal edges to intervened nodes (Pearl's graph surgery).
        
        Implements CRITICAL-003 audit fix: do(X=x) must physically sever all
        incoming edges to X, not just pin the value. Without edge removal,
        subsequent propagation can attempt to overwrite the surgical value,
        and d-separation reasoning on the AMWN sees phantom dependencies.
        
        For trait interventions (do(ENT_X.traits.fear=0.9)), removes incoming
        CausalEdges with trait_target=fear targeting ENT_X.
        
        For event interventions that mark events as prevented/never_happened,
        removes ALL incoming edges to that event node.
        
        For relationship interventions, removes incoming mutation_social edges
        targeting that specific metric axis.
        """
        edges_removed = 0
        
        # Trait interventions: remove incoming causal edges targeting the trait
        for (node_id, trait_name) in self._intervened_traits:
            if trait_name == "*":
                # Skip wildcard (should no longer be used after P0-2 fix)
                continue
            
            # Find all incoming causal edges that target this trait
            if not self.sandbox.has_node(node_id):
                continue
                
            incoming_edges_to_remove = []
            for pred in self.sandbox.predecessors(node_id):
                for key in self.sandbox[pred][node_id]:
                    edge_data = self.sandbox[pred][node_id][key]
                    if edge_data.get("edge_type") == "causal":
                        target_trait = edge_data.get("trait_target")
                        if target_trait == trait_name:
                            incoming_edges_to_remove.append((pred, node_id, key))
            
            for pred, node_id_tgt, key in incoming_edges_to_remove:
                self.sandbox.remove_edge(pred, node_id_tgt, key=key)
                edges_removed += 1
                logger.debug(
                    "[CausalPhysics·graph-surgery] Removed edge %s→%s (key=%s) "
                    "targeting trait %s (do-intervention)",
                    pred, node_id_tgt, key, trait_name
                )
        
        # Event interventions: remove all incoming edges to prevented events
        for evt_id in pruned_event_ids:
            if not self.sandbox.has_node(evt_id):
                continue
                
            incoming_edges_to_remove = []
            for pred in list(self.sandbox.predecessors(evt_id)):
                for key in list(self.sandbox[pred][evt_id].keys()):
                    incoming_edges_to_remove.append((pred, evt_id, key))
            
            for pred, evt_id_tgt, key in incoming_edges_to_remove:
                self.sandbox.remove_edge(pred, evt_id_tgt, key=key)
                edges_removed += 1
                logger.debug(
                    "[CausalPhysics·graph-surgery] Removed edge %s→%s (key=%s) "
                    "(event prevented/invalidated)",
                    pred, evt_id_tgt, key
                )
        
        # Relationship interventions: remove incoming mutation_social edges
        # AUDIT (post-2026-05-26): pinning alone is insufficient. The
        # sandbox MultiDiGraph carries ``mutation_social`` edges from
        # event nodes into the perspective entity, with edge data
        # ``rel_counterpart_id`` and ``trait_target`` identifying the
        # specific (perspective, counterpart, metric) arrow being
        # mutated (see ``propagate_social``). Per Pearl graph surgery
        # on a relationship variable we drop every ``mutation_social``
        # edge whose tuple matches the do-target so social propagation
        # cannot re-fire onto the pinned axis and the AMWN sees the
        # intervened metric as exogenous.
        for (rel_source_id, rel_target_id, metric) in self._intervened_relationships:
            if not self.sandbox.has_node(rel_source_id):
                continue
            incoming_edges_to_remove = []
            for pred in list(self.sandbox.predecessors(rel_source_id)):
                for key in list(self.sandbox[pred][rel_source_id].keys()):
                    edge_data = self.sandbox[pred][rel_source_id][key]
                    if edge_data.get("edge_type") != "causal":
                        continue
                    if edge_data.get("causality_type") != "mutation_social":
                        continue
                    if edge_data.get("rel_counterpart_id") != rel_target_id:
                        continue
                    if edge_data.get("trait_target") != metric:
                        continue
                    incoming_edges_to_remove.append((pred, rel_source_id, key))
            for pred, tgt, key in incoming_edges_to_remove:
                self.sandbox.remove_edge(pred, tgt, key=key)
                edges_removed += 1
                logger.debug(
                    "[CausalPhysics\u00b7graph-surgery] Removed mutation_social "
                    "%s\u2192%s (key=%s) for do(rel %s\u2192%s.%s)",
                    pred, tgt, key, rel_source_id, rel_target_id, metric,
                )
        
        if edges_removed > 0:
            logger.info(
                "[CausalPhysics·graph-surgery] Removed %d incoming edges "
                "to intervened nodes (Pearl's graph surgery)",
                edges_removed
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
            # ``affordance_gate`` edges (entity ENABLES event) refer to
            # the entity's *pre-event* state, while ``mutation`` edges
            # (event MUTATES entity) refer to the *post-event* state.
            # Collapsed onto a single entity node those two classes
            # form a temporal-collapse cycle that doesn't exist in
            # fabula time. Build a cycle-detection view that excludes
            # affordance_gate so genuine forward-causal cycles remain
            # visible while these temporal artefacts are dissolved.
            cycle_view = nx.DiGraph()
            for u, v, edata in causal_graph.edges(data=True):
                if edata.get("causality_type") == "affordance_gate":
                    continue
                cycle_view.add_edge(u, v, **edata)
            try:
                execution_order = list(nx.topological_sort(cycle_view))
                # The cycle was an affordance/mutation temporal artefact;
                # propagate every node in cycle_view's order, then append
                # any nodes that only appear as affordance sources.
                missing = [n for n in causal_graph.nodes if n not in cycle_view]
                execution_order.extend(sorted(missing))
                logger.info(
                    "[CausalPhysics\u00b7Propagate] Cycle dissolved by "
                    "excluding affordance_gate edges from cycle "
                    "detection (temporal-collapse artefact, not a "
                    "real causal loop)."
                )
            except nx.NetworkXUnfeasible:
                sccs = list(nx.strongly_connected_components(cycle_view))
                # A self-loop (n -> n) is technically a one-node SCC
                # that NetworkX classifies as "trivial" (len(s)==1),
                # but it is a genuine cycle — propagation through it
                # would oscillate. Treat self-looped nodes as cyclic
                # so they get blocked alongside multi-node SCCs.
                _selflooped = {
                    n for n in cycle_view.nodes
                    if cycle_view.has_edge(n, n)
                }
                cyclic_sccs = [s for s in sccs if len(s) > 1]
                for s in cyclic_sccs:
                    cyclic_blocked |= s
                cyclic_blocked |= _selflooped
                logger.warning(
                    "[CausalPhysics\u00b7Propagate] Cyclic causal graph: %d SCC(s) with "
                    "%d node(s) total (after excluding affordance_gate). "
                    "Cyclic clusters are blocked from propagation; only "
                    "acyclic spines fire.",
                    len(cyclic_sccs), len(cyclic_blocked),
                )
                condensation = nx.condensation(cycle_view, sccs)
                execution_order = []
                for comp_idx in nx.topological_sort(condensation):
                    members = condensation.nodes[comp_idx]["members"]
                    execution_order.extend(sorted(members))
                missing = [n for n in causal_graph.nodes if n not in cycle_view]
                execution_order.extend(sorted(missing))

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
                    # Affordance-gate / provenance prune honours: a
                    # source marked ``pruned=True`` (by
                    # ``_enforce_affordance_gates`` or by surgery's
                    # provenance sweep) must NOT propagate its
                    # downstream effects, even when it landed in the
                    # active set via interventions/abduction.
                    src_node_data = self.sandbox.nodes.get(src, {})
                    if src_node_data.get("pruned") is True:
                        logger.debug("[CausalPhysics·Propagate] Skipping pruned source %s→%s.%s",
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

                    # Mechanism-targeted gating: reduce weight for non-matching traits.
                    # We track the worst single fallback (mechanism mismatch OR
                    # WORLD_ domain mismatch) and apply it once, instead of
                    # multiplying both penalties — a trait that loses both a
                    # mechanism and a domain match shouldn't be ×fallback².
                    fallback_penalty = 1.0
                    # Multi-mechanism awareness: ambient edges from a
                    # multi-domain WorldTrait carry a ``mechanisms``
                    # list. A trait counts as mechanism-matched when ANY
                    # listed mechanism's MECHANISM_TRAIT_MAP entry
                    # contains it; only when none match do we apply the
                    # fallback.
                    edge_mechanisms_for_gate = edata.get("mechanisms") or [mechanism]
                    matched_any = False
                    for m in edge_mechanisms_for_gate:
                        rt = MECHANISM_TRAIT_MAP.get(m)
                        if rt is None or trait_name in rt:
                            matched_any = True
                            break
                    if not matched_any:
                        fb = _mechanism_fallback_factor()
                        fallback_penalty = min(fallback_penalty, fb)
                        logger.debug("[CausalPhysics·Propagate] %s→%s trait=%s: mechanisms=%s not in target lists, fb=%.3f",
                                         src, node_id, trait_name, edge_mechanisms_for_gate, fb)

                    src_data = self.sandbox.nodes.get(src)
                    if not src_data:
                        continue

                    # Domain filtering for WORLD_ sources: if edge mechanism
                    # is not in the world trait's affected_domains, contribute
                    # to the (single) fallback penalty.
                    if src_data.get("node_type") == "WorldTrait":
                        affected = src_data.get("affected_domains", [])
                        # The instantiator carries the full domain list
                        # on ambient edges under ``mechanisms`` so a
                        # multi-domain world trait can route into every
                        # MECHANISM_TRAIT_MAP family. Treat the trait as
                        # mechanism-matched when ANY declared mechanism
                        # covers the recipient trait.
                        edge_mechanisms = edata.get("mechanisms") or [mechanism]
                        if affected and not any(m in affected for m in edge_mechanisms):
                            fb = _mechanism_fallback_factor()
                            fallback_penalty = min(fallback_penalty, fb)
                            logger.debug("[CausalPhysics·Propagate] WORLD_ domain filter: %s→%s mechanisms=%s not in %s, fb=%.3f",
                                     src, node_id, edge_mechanisms, affected, fb)

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
                        # World trait: scale impulse by *time-resolved* magnitude
                        # intensity. The trait's ``magnitude.value`` may have
                        # shifted via per-chunk WorldTraitUpdate (folded onto
                        # ``GlobalTrait.state_timeline``) or the post-assembly
                        # Step-5 timeline pass. Reading the static node attr
                        # silently ignores those shifts. Look up the canonical
                        # GlobalTrait and reconstruct at the simulation horizon
                        # (the story's "now"); fall back to the sandbox copy
                        # when the trait isn't on world_state (shadow merges).
                        canonical = self.world_state.world_traits.get(src)
                        if canonical is not None and canonical.state_timeline:
                            try:
                                resolved = reconstruct_world_trait_at(
                                    canonical, int(horizon),
                                )
                                mag_value = float(resolved["magnitude"]["value"])
                            except Exception:
                                mag = src_data.get("magnitude", {})
                                mag_value = mag.get("value", 0.5) if isinstance(mag, dict) else 0.5
                        else:
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

        logger.log(
            _physics_log(),
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
                # Self-loop SCCs (single node with a self-edge) are also
                # cycles but ``len(scc) > 1`` misses them. A reciprocal
                # social mutation A\u2192A would silently propagate as if
                # acyclic without this check.
                for node in social_subgraph.nodes():
                    if social_subgraph.has_edge(node, node):
                        social_cyclic_blocked.add(node)
                logger.warning(
                    "[CausalPhysics\u00b7SocialProp] Cyclic social-causal "
                    "subgraph: %d node(s) in non-trivial SCC(s) or "
                    "self-loops. Members are blocked from social "
                    "propagation.",
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
            # Honour the prune flag for social cascade too — see the
            # corresponding check in ``propagate()``.
            if self.sandbox.nodes.get(source_id, {}).get("pruned") is True:
                logger.debug("[CausalPhysics·SocialProp] Skipping pruned source %s→%s (%s)",
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
            
            # HIGH-FIX: Check entity status - dead entities can't have relationships mutated
            target_node = self.sandbox.nodes.get(target_id, {})
            counterpart_node = self.sandbox.nodes.get(counterpart_id, {})
            target_status = target_node.get("status")
            counterpart_status = counterpart_node.get("status")
            if target_status == "dead" or counterpart_status == "dead":
                logger.debug(
                    "[CausalPhysics·SocialProp] Skipping dead entity: target=%s (status=%s), counterpart=%s (status=%s)",
                    target_id, target_status, counterpart_id, counterpart_status
                )
                continue

            # Per-axis relationship pin: a do(ENT_A.relationships.ENT_B.affinity=...)
            # surgery freezes that specific (target, counterpart, metric)
            # triple — propagation must not subsequently overwrite the
            # surgical value with a propagated delta. Symmetric to the
            # per-trait pinning logic in propagate().
            if (target_id, counterpart_id, metric) in self._intervened_relationships:
                logger.debug(
                    "[CausalPhysics·SocialProp] Pinned: %s→%s %s skipped",
                    target_id, counterpart_id, metric,
                )
                self._blocked.append(BlockedPropagation(
                    node_id=target_id,
                    trait=f"rel.{counterpart_id}.{metric}",
                    impact=raw_delta,
                    inertia=_relationship_inertia_default(),
                    reason="pinned",
                ))
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
                    # Prefer per-axis ``metrics[metric].value`` when present
                    # — the per-axis dict is the source of truth in the
                    # new schema; the flat key is a derived back-compat
                    # mirror that may be stale on edges that did not
                    # round-trip through ``to_legacy_dict``.
                    per_metric = (rdata.get("metrics") or {}).get(metric) or {}
                    if "value" in per_metric and isinstance(per_metric["value"], (int, float)):
                        current_val = float(per_metric["value"])
                    else:
                        current_val = rdata.get(metric, 0.0)
                        if not isinstance(current_val, (int, float)):
                            current_val = 0.0
                    rel_inertia = per_metric.get(
                        "inertia",
                        rdata.get("inertia", _relationship_inertia_default()),
                    )

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
                    # Mirror the change into the per-axis ``metrics``
                    # dict (when present) so downstream readers that
                    # use the new shape see the updated value too.
                    if isinstance(rdata.get("metrics"), dict):
                        axis_state = rdata["metrics"].setdefault(metric, {
                            "value": current_val,
                            "inertia": rel_inertia,
                            "evidence_strength": d.get("evidence_strength", "weak"),
                            "last_updated_fabula": d.get("fabula_time", 0),
                            "observed": True,
                        })
                        axis_state["value"] = new_val
                        axis_state["last_updated_fabula"] = max(
                            axis_state.get("last_updated_fabula", 0),
                            d.get("fabula_time", 0),
                        )
                        # Flip ``observed`` true: a propagated mutation is a
                        # measurement of the post-mutation state regardless
                        # of whether the pre-mutation axis was a measured
                        # neutral or an unobserved default.
                        axis_state["observed"] = True
                        # Carry the triggering edge's evidence_strength
                        # through so the new measurement claims no more
                        # confidence than the cause that produced it.
                        trigger_es = d.get("evidence_strength")
                        if trigger_es:
                            axis_state["evidence_strength"] = trigger_es
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
                    logger.log(
                        _physics_log(),
                        "[CausalPhysics·SocialProp] %s→%s %s: %.3f→%.3f (trigger=%s, delta=%.3f, inertia=%.3f)",
                        target_id, counterpart_id, metric, current_val, new_val, source_id, scaled_delta, rel_inertia,
                    )
                    break

            # No existing relationship edge — create one with full per-axis metrics
            if not rel_found:
                if metric == "fear":
                    primary_value = max(0.0, min(1.0, scaled_delta))
                else:
                    primary_value = max(-1.0, min(1.0, scaled_delta))
                edge_attrs = {
                    "edge_type": "relationship",
                    "affinity": 0.0,
                    "fear": 0.0,
                    "power_dynamic": 0.0,
                    "inertia": _relationship_inertia_default(),
                    "evidence_strength": "weak",
                    "last_updated_fabula": d.get("fabula_time", 0),
                    "world_id": "shadow",
                    "metrics": default_relationship_metrics_dict(
                        primary_metric=metric,
                        primary_value=primary_value,
                        fabula_time=d.get("fabula_time", 0),
                        evidence_strength=d.get("evidence_strength", "weak"),
                        inertia=_relationship_inertia_default(),
                    ),
                }
                edge_attrs[metric] = primary_value
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
                logger.log(
                    _physics_log(),
                    "[CausalPhysics·SocialProp] Created relationship %s→%s with %s=%.3f (trigger=%s)",
                    target_id, counterpart_id, metric, edge_attrs[metric], source_id,
                )

        logger.log(
            _physics_log(),
            "[CausalPhysics·SocialProp] %d social mutations applied.",
            len(self._social_mutations),
        )

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
            # N1 (2026-05-29 ninth-pass audit): only admit an unlocking
            # key whose holder is *physically at* one of the locked
            # edge's endpoints. The previous version treated any key
            # owned anywhere in the world (including by dead or off-map
            # characters) as making the locked edge globally traversable
            # — so a dropped/inherited key on the far side of the map
            # silently opened every matching door.
            for nid, ndata in self.sandbox.nodes(data=True):
                if ndata.get("node_type") != "NarrativeObject":
                    continue
                owner_id = ndata.get("owner_id")
                if owner_id is None:
                    continue
                owner_node = self.sandbox.nodes.get(owner_id, {})
                owner_loc = owner_node.get("location_id")
                if owner_loc not in (u, v):
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

        Note
        ----
        Keys recorded in ``self._legacy_applied_keys`` (populated by
        :meth:`apply_do_targets` when it routed DoEvent / DoTrait /
        DoProposition / DoWorldTrait surgeries through their canonical
        handlers) are filtered out of ``interventions`` before
        :meth:`apply_do_operator` re-applies the dict, so the same
        surgery is not committed twice (which would drift trait values
        under inertia dampening). The full dict is still used for
        Rule-3 preflight, provenance pruning, and reportage.
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
        # Per-key filter: any key recorded in ``_legacy_applied_keys``
        # was already committed by ``apply_do_targets`` via the
        # canonical typed handler (or via ``apply_do_operator`` internally
        # for the DoEvent/DoTrait lift). Skip those to avoid double-
        # application. Keys for DoNarrativeObject inventory / spatial
        # surgeries are NOT in this set — the typed handler only writes
        # properties; legacy ``_intervene_inventory`` /
        # ``_intervene_spatial`` must still run to rewire edges.
        applied_keys = getattr(self, "_legacy_applied_keys", set()) or set()
        if rung >= 2 and interventions and applied_keys:
            interventions_to_apply = {
                k: v for k, v in interventions.items() if k not in applied_keys
            }
        else:
            interventions_to_apply = interventions
        if rung >= 2 and interventions_to_apply:
            self.apply_do_operator(interventions_to_apply)

        # Step B.5 — Provenance prune.
        # Walk the (possibly filtered) intervention set and identify any
        # surgeries that *epistemically remove* an event or channel
        # (e.g. ``EVT_X.event_type='prevented'``,
        # ``EVT_X.truth_value='false'``, ``CHAN_Y.status='severed'``).
        # Beliefs whose ``acquired_via_*`` provenance pointed at those
        # IDs are no longer justifiable in the counterfactual world,
        # so we drop them from the sandbox before propagation.
        pruned_evt_ids, disabled_ch_ids = (
            self._collect_provenance_invalidations(interventions)
        )
        # P0-FIX: Implement graph surgery edge removal (CRITICAL-003 audit).
        # Pearl's do-operator requires removing all incoming causal edges to
        # intervened nodes. Pinning alone (via _intervened_traits) prevents
        # propagation from overwriting surgical values, but leaves structural
        # edges intact that can interfere with d-separation reasoning and
        # create spurious dependencies.
        self._perform_graph_surgery_edge_removal(interventions, pruned_evt_ids)
        
        # Pick up events that ``_enforce_affordance_gates`` (called
        # at the end of ``execute_interventions`` in the surgery step
        # above) marked ``pruned=True`` so beliefs acquired from
        # gate-blocked events are cleaned alongside provenance-pruned
        # ones.
        for _nid, _ndata in self.sandbox.nodes(data=True):
            if _ndata.get("node_type") == "EventNode" and _ndata.get("pruned") is True:
                pruned_evt_ids.add(_nid)

        # Step B.6 — Chain-reaction descendant closure for prevented events.
        #
        # When a surgery sets ``EVT_X.event_type ∈ {prevented,
        # never_happened, removed}`` or ``EVT_X.truth_value='false'``,
        # any event whose chain_reaction parents are ALL now pruned /
        # cause-disconnected must also be marked ``pruned`` — otherwise
        # the brief continues to list those descendants as canonical
        # and the renderer dutifully stages them on-page (e.g. Macbeth
        # still stabs Duncan after we erase the persuasion). Pearl's
        # disjunctive structural-equation rule: over-determined effects
        # with a surviving sufficient cause are preserved; ``enables``
        # / ``affordance_gate`` / ``ambient_propagation`` / ``mutation``
        # edges are modifiers, not sufficient causes, so they do not
        # participate in this closure.
        #
        # Mirrors :func:`shadow_loom.pipeline._compute_shadow_prune_closure`
        # but operates on the sandbox MultiDiGraph (we want descendants
        # marked ``pruned=True`` so ``propagate()`` and the cascade gates
        # skip them, AND we want them in ``pruned_evt_ids`` so the brief's
        # ``SEVERED CAUSAL CHAINS`` / ``DEPENDENT-STATE SUBSTITUTIONS``
        # blocks include the whole cascade).
        if pruned_evt_ids:
            chain_parents = chain_reaction_parents_from_sandbox(self.sandbox)
            closure = expand_chain_reaction_closure(chain_parents, pruned_evt_ids)
            newly_pruned = closure - pruned_evt_ids
            if newly_pruned:
                for nid in newly_pruned:
                    if self.sandbox.has_node(nid):
                        self.sandbox.nodes[nid]["pruned"] = True
                pruned_evt_ids |= newly_pruned
                logger.info(
                    "[CausalPhysics·ChainClosure] do-prevented surgery "
                    "expanded %d root event(s) → %d descendant(s) via "
                    "chain_reaction closure: %s",
                    len(closure) - len(newly_pruned),
                    len(newly_pruned),
                    sorted(newly_pruned),
                )

        # Audit (eighth pass, H1): mirror the merge-time
        # ``cause_disconnected_event_ids`` seeding done by
        # ``shadow_loom.pipeline.build_branch_topology`` so engine-time
        # and persistence-time chain-reaction closures agree. Without
        # this, a DoEvent that disruptively mutates (e.g. swaps
        # ``actor_ids`` / ``target_ids`` / wipes ``description`` /
        # changes ``event_type`` to a non-prevented value) without
        # marking the event ``pruned`` leaves descendants alive in
        # the engine sandbox — they fire during propagation, get
        # cited by the brief / answer layer — but the merge layer
        # then suppresses them, producing a turn-internal split
        # where reasoning and persistence disagree. A pure
        # relocation/time_shift preserves the chain (see ROUND-15
        # C-2 in pipeline.py) so we exclude those.
        _disruptive_evt_ids: set[str] = set()
        _relocation_only_evt_ids: set[str] = set()
        for _m in self._event_mutations:
            _meid = getattr(_m, "event_id", None)
            _mkind = getattr(_m, "kind", None)
            if not _meid:
                continue
            if _mkind in ("relocation", "time_shift"):
                _relocation_only_evt_ids.add(_meid)
            else:
                _disruptive_evt_ids.add(_meid)
        # Any event with a disruptive mutation stays cause-broken even
        # if it also has a relocation mutation in the same batch.
        cause_broken_evt_ids = _disruptive_evt_ids - (
            _relocation_only_evt_ids - _disruptive_evt_ids
        )
        # Subtract events already pruned (closure above already
        # handled their descendants) — we only need the
        # cause-disconnected delta as additional seeds.
        cause_broken_seeds = cause_broken_evt_ids - pruned_evt_ids
        if cause_broken_seeds:
            chain_parents = chain_reaction_parents_from_sandbox(self.sandbox)
            cd_closure = expand_chain_reaction_closure(
                chain_parents, cause_broken_seeds,
            )
            # Descendants of cause-broken events are themselves
            # cause-disconnected — mark and add to pruned_evt_ids so
            # provenance pruning + brief surfacing sees the full
            # cascade. Do NOT mark the cause_broken_seeds themselves
            # pruned (they still occurred; they just no longer cause
            # what they used to).
            cd_descendants = cd_closure - cause_broken_seeds
            cd_newly_pruned = cd_descendants - pruned_evt_ids
            if cd_newly_pruned:
                for nid in cd_newly_pruned:
                    if self.sandbox.has_node(nid):
                        self.sandbox.nodes[nid]["pruned"] = True
                pruned_evt_ids |= cd_newly_pruned
                logger.info(
                    "[CausalPhysics·ChainClosure] cause-disconnected "
                    "do-event mutation expanded %d root event(s) → %d "
                    "descendant(s) via chain_reaction closure: %s",
                    len(cause_broken_seeds),
                    len(cd_newly_pruned),
                    sorted(cd_newly_pruned),
                )

        beliefs_pruned = 0
        if pruned_evt_ids or disabled_ch_ids:
            beliefs_pruned = AMWNInstantiator._prune_beliefs_by_provenance(
                self.sandbox,
                removed_event_ids=pruned_evt_ids,
                removed_channel_ids=disabled_ch_ids,
            )
            # P1-FIX: Mirror belief pruning to canonical world_state
            self._mirror_belief_pruning_to_canonical(
                removed_event_ids=pruned_evt_ids,
                removed_channel_ids=disabled_ch_ids,
            )

        # Step C — Forward propagation
        self.propagate()

        # Step D — Social propagation (mutation_social edges)
        self.propagate_social()

        # Step D.5 — Inert-intervention detection.
        # Failure mode observed in the wild: every requested do-target is
        # Rule-3 pruned AND every downstream trait mutation is absorbed by
        # a cyclic SCC. The engine then produces zero observable change,
        # but the brief still hands a counterfactual prompt to the
        # renderer, which fabricates content ("professionalism held
        # steady\u2026 calm remained unshaken\u2026"). Detect this state
        # here so the pipeline can disclose it instead of papering over.
        _inert = False
        _inert_reason: Optional[str] = None
        # Track whether the caller *requested* a Rung-2/3 surgery at
        # all. ``interventions`` can be falsy in two distinct ways:
        # the caller passed nothing (Rung-1 observation \u2014 no inert
        # state possible) or every requested key was filtered out
        # upstream (type mismatch, wrong ID format, prune-mode Rule-3
        # drop). The latter case is itself an inert state we MUST
        # surface; the previous ``if interventions:`` guard
        # short-circuited it.
        _requested_surgery = rung >= 2 and (
            bool(interventions)
            or bool(self.sandbox.graph.get("skipped_interventions", []))
            or bool(ctf_report.rule3_pruned)
        )
        if _requested_surgery:
            _requested = set(interventions.keys())
            _pruned = set(ctf_report.rule3_pruned)
            _skipped = list(
                self.sandbox.graph.get("skipped_interventions", []) or []
            )
            _all_pruned = bool(_requested) and _requested.issubset(_pruned)
            _no_mutations = (
                not self._mutations
                and not self._social_mutations
                and not self._proposition_mutations
                and not self._belief_mutations
                and not self._concern_mutations
                and not self._world_trait_mutations
                and not self._object_mutations
                and not self._edge_mutations
                # A direct trait/node clamp via apply_do_operator pins
                # ``_intervened_traits`` / ``_intervened_nodes`` but
                # does NOT record a TraitMutation (mutation rows are
                # only emitted for downstream propagation, see
                # _propagate_*). If propagation is blocked (cyclic
                # SCC, inertia, etc.) but the clamp itself landed,
                # the world DID move — surfacing ``intervention_inert``
                # would be UX-misleading. Treat a non-empty trait
                # pin-set as observable movement.
                and not self._intervened_traits
            )
            _cycle_blocked = [
                b for b in self._blocked if b.reason == "cycle"
            ]
            # Additional absorbed-failure modes beyond pure cycle
            # blocks: inertia clamps that prevent any movement,
            # noisy-OR gate samples that absorbed every contribution,
            # spatial-affordance vetoes, and pinned targets. If the
            # caller's surgery produced no mutations AND every
            # downstream propagation was rejected for any of these
            # reasons, the request is observably inert and we MUST
            # surface that to the renderer/pipeline (the previous
            # classifier only counted "cycle" blocks, so an inertia-
            # or affordance-absorbed request silently looked like a
            # successful surgery with no consequences).
            _absorbed_blocked = [
                b for b in self._blocked
                if b.reason in (
                    "cycle",
                    "inertia",
                    "noisy_or_absorbed",
                    "spatial_affordance",
                    "pinned",
                )
            ]
            # Case 0: caller-requested surgery was emptied upstream.
            # ``_requested`` is the *post-prune* working set; if it is
            # empty but the original request was non-trivial (skipped
            # or rule-3-pruned), no do-surgery actually landed and
            # downstream is necessarily inert.
            if not _requested and (_skipped or _pruned):
                _inert = True
                _inert_reason = (
                    f"all requested intervention(s) filtered before "
                    f"surgery (skipped={len(_skipped)}, rule3_pruned="
                    f"{len(_pruned)}); no do-operator applied"
                )
            elif _all_pruned and _no_mutations:
                _inert = True
                _inert_reason = (
                    f"all {len(_requested)} intervention(s) Rule-3 pruned "
                    f"({sorted(_pruned)}) and no downstream mutations "
                    f"fired (cycle-blocked: {len(_cycle_blocked)})"
                )
            elif _no_mutations and _cycle_blocked:
                _inert = True
                _inert_reason = (
                    f"intervention(s) {sorted(_requested)} produced no "
                    f"mutations; {len(_cycle_blocked)} trait "
                    f"propagation(s) absorbed by cyclic SCC"
                )
            elif _no_mutations and _absorbed_blocked:
                # Broader absorbed-failure case: not a cycle, but
                # every propagation was rejected by inertia, noisy-OR
                # gate, spatial affordance, or pinning. Surface with
                # a reason breakdown so callers can disclose.
                from collections import Counter as _Counter
                _by_reason = _Counter(b.reason for b in _absorbed_blocked)
                _inert = True
                _inert_reason = (
                    f"intervention(s) {sorted(_requested)} produced no "
                    f"mutations; {len(_absorbed_blocked)} trait "
                    f"propagation(s) absorbed (by reason: "
                    f"{dict(_by_reason)})"
                )
            if _inert:
                logger.warning(
                    "[CausalPhysics\u00b7Inert] Intervention is inert "
                    "\u2014 %s. Pipeline should disclose this rather "
                    "than render fabricated consequences.",
                    _inert_reason,
                )

        result = CausalPhysicsResult(
            sandbox_data=nx.node_link_data(self.sandbox),
            mutations=self._mutations,
            social_mutations=self._social_mutations,
            blocked=self._blocked,
            intervened_nodes=sorted(self._intervened_nodes),
            proposition_mutations=self._proposition_mutations,
            belief_mutations=self._belief_mutations,
            concern_mutations=self._concern_mutations,
            world_trait_mutations=self._world_trait_mutations,
            object_mutations=self._object_mutations,
            edge_mutations=self._edge_mutations,
            entity_delete_mutations=self._entity_delete_mutations,
            object_delete_mutations=self._object_delete_mutations,
            event_mutations=self._event_mutations,
            hidden_deltas=self._hidden_deltas,
            rule3_pruned_interventions=ctf_report.rule3_pruned,
            rule3_pruning_mode=rule3_mode,
            rule2_redundant_evidence=ctf_report.rule2_redundant_evidence,
            noisy_or_probabilities=self._noisy_or_records,
            pruned_beliefs_count=beliefs_pruned,
            pruned_utterance_event_ids=sorted(pruned_evt_ids),
            disabled_channel_ids=sorted(disabled_ch_ids),
            skipped_interventions=list(
                self.sandbox.graph.get("skipped_interventions", []) or []
            ),
            intervention_inert=_inert,
            intervention_inert_reason=_inert_reason,
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
        # World-state must ALSO be isolated per sample. Many ``_apply_do_*``
        # handlers mirror surgical writes onto the canonical
        # ``world_state.objects`` / ``.entities`` / ``.events`` (so non-
        # sandbox readers like directive assembly see the clamp), and a
        # shared world_state would let sample N's mirror writes leak
        # into the priors seen by sample N+1, biasing the distribution
        # and violating i.i.d. sampling. Deep-copy once and bind every
        # sub-engine to its own copy. The caller's ``self.world_state``
        # reference is restored at the end so external readers are
        # unaffected.
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
            # The world_state is also deep-copied per sample so mirror
            # writes from typed do-handlers (object owner / location,
            # proposition truth, world-trait value, etc.) don't leak
            # across samples.
            sample_world_state = copy.deepcopy(original_world_state)
            sub = CausalPhysicsEngine(sample_sandbox, sample_world_state)
            sub._rng = rng
            # Mark this engine as already executing inside the Monte-Carlo
            # loop so its ``execute()`` call does not re-enter
            # ``execute_distribution`` and recurse forever.
            sub._in_mc_sample = True
            sub._sample_noisy_or = True
            # Replicate the parent engine's typed do-surgery state so the
            # sub-engine's ``CausalPhysicsResult`` faithfully reports the
            # proposition / belief / concern / world-trait / object
            # mutations that ``apply_do_targets()`` committed on the
            # parent BEFORE this Monte-Carlo sweep began. The structural
            # effects of those typed surgeries already live on
            # ``sample_sandbox`` / ``sample_world_state`` (via the deep
            # copies above) — but the mutation *records*, the intervened
            # node set, and the ``_legacy_applied_keys`` guard live on
            # the parent engine and would otherwise be lost: the sub-
            # engine only sees the legacy ``interventions`` dict, which
            # does not understand ``PROP_*.truth`` / belief / concern
            # keys, so its typed lists would stay empty and the final
            # result would falsely report ``intervention_inert=True``.
            # We also replicate ``_mutations`` / ``_social_mutations``
            # (legacy trait & relationship clamps that ``DoTrait`` /
            # ``DoRelationship`` already wrote on the parent),
            # ``_intervened_traits`` (so per-sample propagation respects
            # parent trait pins), and ``_last_legacy_interventions`` (so
            # provenance invalidation sees the full surgery surface).
            sub._proposition_mutations = list(self._proposition_mutations)
            sub._belief_mutations = list(self._belief_mutations)
            sub._concern_mutations = list(self._concern_mutations)
            sub._world_trait_mutations = list(self._world_trait_mutations)
            sub._object_mutations = list(self._object_mutations)
            sub._edge_mutations = list(self._edge_mutations)
            sub._entity_delete_mutations = list(self._entity_delete_mutations)
            sub._object_delete_mutations = list(self._object_delete_mutations)
            sub._mutations = list(self._mutations)
            sub._social_mutations = list(self._social_mutations)
            sub._intervened_nodes = set(self._intervened_nodes)
            sub._intervened_traits = set(self._intervened_traits)
            sub._intervened_relationships = set(self._intervened_relationships)
            sub._legacy_applied_keys = set(
                getattr(self, "_legacy_applied_keys", set()) or set()
            )
            sub._last_legacy_interventions = dict(
                getattr(self, "_last_legacy_interventions", {}) or {}
            )
            # Flip the module-level contextvar so per-step physics logs
            # (abduction, do-surgery, propagate, social, result) demote
            # to DEBUG for this sample. Reset in finally so we don't leak
            # the flag to the caller after the sweep.
            mc_token = _in_mc_sample_var.set(True)
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
            finally:
                _in_mc_sample_var.reset(mc_token)

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
        # ``self.world_state`` was never mutated (each sample bound to
        # its own deep copy), so no restore is needed for it.
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

        # Single INFO summary for the whole MC sweep (per-sample logs
        # were already demoted to DEBUG via the contextvar).
        if logger.isEnabledFor(logging.INFO):
            label = _RUNG_NAMES.get(rung, f"Rung {rung}")
            n_dist_traits = sum(len(v) for v in distributions.values())
            summary_lines = [
                f"[CausalPhysics·MC] {label} — Monte-Carlo sweep complete: "
                f"{n} sample(s), {len(distributions)} entity nodes, "
                f"{n_dist_traits} trait distributions."
            ]
            if interventions:
                summary_lines.append(
                    f"  Interventions: {sorted(interventions)}"
                )
            if evidence_node_ids:
                summary_lines.append(
                    f"  Abduction evidence: {evidence_node_ids[:8]}"
                    + (f" (+{len(evidence_node_ids) - 8} more)"
                       if len(evidence_node_ids) > 8 else "")
                )
            # Spotlight a few representative trait posteriors so the
            # operator can sanity-check magnitude without a 128-line dump.
            shown = 0
            for nid, by_trait in distributions.items():
                if shown >= 5:
                    summary_lines.append(
                        f"  … (+{len(distributions) - shown} more entities; "
                        "see CausalPhysicsResult.trait_distributions)"
                    )
                    break
                for tname, td in by_trait.items():
                    summary_lines.append(
                        f"  {nid}.{tname}: mean={td.mean:.3f} "
                        f"std={td.std:.3f} p5={td.p5:.3f} p95={td.p95:.3f} "
                        f"(N={td.samples_count})"
                    )
                shown += 1
            logger.info("\n".join(summary_lines))

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
    log_level = _physics_log()
    if not logger.isEnabledFor(log_level):
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

    logger.log(log_level, "\n".join(lines))


# Resolve forward references in CausalPhysicsResult — the typed fields
# noisy_or_probabilities and trait_distributions reference NoisyOrProbability
# and TraitDistribution as strings (combined with `from __future__ import
# annotations`), so an explicit rebuild ensures pydantic v2 wires them up
# eagerly at import time rather than lazily on first instantiation.
CausalPhysicsResult.model_rebuild()
