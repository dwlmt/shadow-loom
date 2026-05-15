# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Annotated, Any, List, Dict, Optional, Literal

_logger = logging.getLogger(__name__)


# Tier 5 #16: shared annotated alias for entity ids. Non-breaking — we
# expose the alias and use it on new fields; existing ``str`` fields
# remain ``str`` to preserve backwards compatibility with fixtures /
# example worlds that may emit looser ids during construction.
EntityId = Annotated[
    str,
    Field(
        pattern=r"^(?:ENT|OBJ|LOC|EVT|CHN)_[A-Z0-9_]+$",
        description="Canonical Shadow Loom topology id (ENT_/OBJ_/LOC_/EVT_/CHN_).",
    ),
]

# Sibling aliases for the affect-unification id namespaces. Kept as
# separate annotated types (rather than folded into ``EntityId``) so
# the topology-id pattern above stays a strict gate for entity /
# object / location / event / channel references; propositions and
# concerns live in their own namespaces and should not be cross-
# referenced from those fields.
PropositionId = Annotated[
    str,
    Field(
        pattern=r"^PROP_[A-Z0-9_]+$",
        description="Canonical Shadow Loom proposition id (PROP_*).",
    ),
]

ConcernId = Annotated[
    str,
    Field(
        pattern=r"^CCN_[A-Z0-9_]+$",
        description="Canonical Shadow Loom concern id (CCN_*).",
    ),
]

# =====================================================================
# PART 1: THE GRAPH DATABASE (The Reality Engine)
# =====================================================================


# Shared helper: coerce common LLM aliases for evidence_strength to the
# canonical {weak, moderate, strong} vocabulary *before* Pydantic's
# Literal validation runs. Without this the model would reject otherwise
# salvageable LLM output (e.g. "high", "medium") and force pydantic_ai
# to retry, wasting tokens.
_EVIDENCE_STRENGTH_ALIASES = {
    "weak": "weak",
    "moderate": "moderate",
    "strong": "strong",
    "high": "strong",
    "certain": "strong",
    "definite": "strong",
    "medium": "moderate",
    "med": "moderate",
    "average": "moderate",
    "low": "weak",
    "uncertain": "weak",
    "speculative": "weak",
    "implied": "weak",
}


def _coerce_evidence_strength(v: Any) -> Any:
    """Map common synonyms onto {weak, moderate, strong}; pass through
    anything we don't recognise so the Literal check still fires."""
    if not isinstance(v, str):
        return v
    return _EVIDENCE_STRENGTH_ALIASES.get(v.strip().lower(), v)


# Map common but off-vocabulary mechanism-domain labels onto canonical
# MECHANISM_TRAIT_MAP keys. Without this, a GlobalTrait whose
# affected_domains contains "economic" silently fails to match the
# pressure-routing gate (which keys on the canonical list), losing ~80%
# of its ambient pressure on dependent edges.
_DOMAIN_ALIASES = {
    "economic": "social",
    "economics": "social",
    "financial": "social",
    "political": "social",
    "moral": "psychological",
    "ethical": "psychological",
    "ideological": "epistemic",
    "religious": "epistemic",
    "spiritual": "epistemic",
    "supernatural": "psychological",
    "magical": "psychological",
    "environmental": "physical",
    "environment": "physical",
    "ecological": "physical",
    # Institutional / aesthetic / professional registers seen in OSS
    # extractions (e.g. The Grand Budapest Hotel, A Christmas Carol).
    # Without these, off-list mechanism labels silently bypass the
    # MECHANISM_TRAIT_MAP routing gate and shed ~80% of their pressure.
    "legal": "social",
    "juridical": "social",
    "bureaucratic": "social",
    "institutional": "social",
    "procedural": "social",
    "diplomatic": "social",
    "aesthetic": "emotional",
    "artistic": "emotional",
    "sentimental": "emotional",
    "nostalgic": "emotional",
    "reputational": "social",
    "honour": "social",
    "honor": "social",
    "familial": "social",
}


def _coerce_domain(v: Any) -> Any:
    """Map common synonyms onto canonical MECHANISM_TRAIT_MAP keys."""
    if not isinstance(v, str):
        return v
    return _DOMAIN_ALIASES.get(v.strip().lower(), v)


def _coerce_domain_list(v: Any) -> Any:
    if not isinstance(v, list):
        return v
    return [_coerce_domain(item) for item in v]


# Map common LLM / hand-authored ``mechanism`` labels onto the short
# canonical keys used throughout the codebase. The longer ``_revelation``
# / ``_coercion`` / ``_force`` suffixes are still recognised by
# ``causal_physics.MECHANISM_TRAIT_MAP`` for backward compatibility, but
# we canonicalize on the model boundary so that downstream pretty-prints,
# audit reports, and prompt-snippet round-trips all see the same token.
# Off-list mechanism labels (``kinetic``, ``chemical``, ``seduction`` …)
# are intentionally NOT mapped: they are a documented escape hatch that
# bypasses mechanism-trait routing.
_MECHANISM_ALIASES = {
    "physical_force": "physical",
    "epistemic_revelation": "epistemic",
    "social_coercion": "social",
    # Institutional / aesthetic registers (mirrors _DOMAIN_ALIASES on
    # the CausalEdge.mechanism boundary). OSS extractions for legal-
    # heavy plots (Grand Budapest, court dramas) routinely emit
    # 'legal'/'bureaucratic' here; map onto the canonical short keys.
    "legal": "social",
    "juridical": "social",
    "bureaucratic": "social",
    "institutional": "social",
    "procedural": "social",
    "diplomatic": "social",
    "reputational": "social",
    "honour": "social",
    "honor": "social",
    "familial": "social",
    "aesthetic": "emotional",
    "artistic": "emotional",
    "sentimental": "emotional",
    "nostalgic": "emotional",
    "environmental": "physical",
    "environment": "physical",
    "ecological": "physical",
    "economic": "social",
    "financial": "social",
    "political": "social",
    "ideological": "epistemic",
    "religious": "epistemic",
    "moral": "psychological",
    "ethical": "psychological",
    "supernatural": "psychological",
    "magical": "psychological",
}


def _coerce_mechanism(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    return _MECHANISM_ALIASES.get(v.strip().lower(), v)


# --- 0. AMWN (Ancestral Multiverse World Network) BASE CLASS (The Multiverse Tag) ---
class AMWNNode(BaseModel):
    world_id: Literal["factual", "shadow"] = Field(
        default="factual", 
        description="Tracks if this node belongs to the true timeline or a 'What If' branch."
    )

# --- 1. CORE PROPERTIES (The Physics & Soul) ---
class TraitVector(BaseModel):
    value: float = Field(description="0.0 to 1.0 (Current level of the trait)")
    inertia: float = Field(description="0.0 to 1.0 (Force required to shatter this trait. 1.0 = permanent)")
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description=(
            "Engine's certainty in the *extraction* of this trait from the source text. "
            "'strong' = directly stated/enacted, 'moderate' = inferred from behaviour, "
            "'weak' = abductive guess from sparse cues. Distinct from ``value`` "
            "(in-world magnitude) and ``inertia`` (resistance to change)."
        ),
    )

    _coerce_es = field_validator("evidence_strength", mode="before")(
        lambda v: _coerce_evidence_strength(v)
    )

class AmbientVector(BaseModel):
    value: float = Field(description="0.0 to 1.0 (How intense is this state?)")
    volatility: float = Field(description="0.0 to 1.0 (How fast does this change? 0.0 = immutable)")
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description=(
            "Engine's certainty in the *extraction* of this ambient condition. "
            "'strong' = explicit textual description, 'moderate' = inferred from "
            "scene context, 'weak' = assumed from genre/setting conventions. "
            "Distinct from ``value`` (intensity) and ``volatility`` (rate of change)."
        ),
    )

    _coerce_es = field_validator("evidence_strength", mode="before")(
        lambda v: _coerce_evidence_strength(v)
    )

class Affordance(BaseModel):
    action: str = Field(description="What this object can do (e.g., 'unlock', 'kill', 'read')")
    target_type: str = Field(description="What it acts upon (e.g., 'Door', 'Entity')")

class Belief(BaseModel):
    target_id: str = Field(description="ID of the object/entity/event they hold a belief about.")
    perceived_state: str = Field(description="What they THINK is true (e.g., 'Cup is safe').")
    confidence: float = Field(description="0.0 to 1.0 (How sure are they?)")
    inertia: float = Field(description="0.0 to 1.0 (How stubborn is this belief?)")
    established_at_fabula: int = Field(default=0, description="Fabula time when this belief was formed. Used for counterfactual time-slicing.")
    acquired_via_event_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional EVT_ id of the event (typically an utterance or revelation) "
            "that posted this belief. Lets counterfactual surgery prune downstream "
            "beliefs whose causing event no longer fires."
        ),
    )
    acquired_via_channel_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional CHN_ id of the standing communication channel through which "
            "the belief was acquired. Lets counterfactual surgery on a channel "
            "(e.g. line tapped, cipher broken, bond severed) prune downstream beliefs."
        ),
    )
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description=(
            "How reliably the engine should treat this belief as extracted. "
            "'strong' = unambiguous textual support, 'moderate' = inferred "
            "from context, 'weak' = abductive guess. Distinct from "
            "``confidence`` (which is the *character's* certainty); "
            "evidence_strength is the *engine's* certainty in the "
            "extraction. Feeds the Bayesian abduction variance."
        ),
    )
    proposition_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional PROP_ id joining this belief to a shared ``Proposition`` "
            "in ``WorldStateV1.propositions``. When set, the affect-unification "
            "layer can compute KL divergence between two agents' beliefs about "
            "the *same* proposition (dramatic irony) and Bayesian surprise "
            "between successive belief snapshots (Itti-Baldi). Backward-"
            "compatible: legacy beliefs without a ``proposition_id`` continue "
            "to render via ``perceived_state`` as before."
        ),
    )

    _coerce_es = field_validator("evidence_strength", mode="before")(
        lambda v: _coerce_evidence_strength(v)
    )


class Proposition(AMWNNode):
    """A first-class proposition that agents hold beliefs about.

    Backbone of the affect-unification layer (see
    ``/memories/repo/affect-unification-plan.md``). Lifts the opaque
    ``Belief.perceived_state`` string into a shared, joinable object
    so suspense, surprise, dramatic irony, and mystery can all be
    computed as queries on the same belief tensor.

    A ``Proposition`` is the storyworld *thing under discussion*. The
    audience (``ENT_AUDIENCE``) and each character hold per-agent
    confidence values about it. ``truth_at_fabula`` records when the
    proposition's ground-truth value commits in fabula time, so
    Bayesian surprise (``-log p(P)`` at reveal) is well-defined.

    Inherits ``world_id`` from :class:`AMWNNode` so a Rung-2/3
    intervention can introduce shadow-world propositions (e.g. a
    counterfactual ``PROP_DUNCAN_DEAD = false`` proposition tagged
    ``world_id='shadow'`` lives alongside the factual one) without
    overwriting the factual ledger.
    """
    proposition_id: str = Field(
        description="Unique PROP_ id, e.g. PROP_DUNCAN_DEAD.",
    )
    kind: Literal[
        "event_occurs", "trait_holds", "relation_holds",
        "identity_is", "outcome",
    ] = Field(
        description=(
            "Ontological kind. ``outcome`` is the Brewer-Lichtenstein "
            "resolution-of-an-open-question class that drives suspense; "
            "the others are background propositions whose belief-state "
            "feeds surprise / irony / mystery."
        ),
    )
    referent_ids: List[str] = Field(
        default_factory=list,
        description=(
            "EVT_/ENT_/OBJ_/LOC_/WORLD_/CHN_ ids the proposition is *about*. "
            "E.g. PROP_DUNCAN_DEAD references ['ENT_DUNCAN']; "
            "PROP_MACBETH_KILLS_DUNCAN references the murder event."
        ),
    )
    description: str = Field(
        description="Human-readable label, e.g. 'Duncan is dead'.",
    )
    audience_default_prior: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description=(
            "Audience's prior confidence in the proposition before any "
            "narrative evidence. Lower for blindsiding twists, higher "
            "for telegraphed inevitabilities."
        ),
    )
    stakes: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description=(
            "How much resolution of this proposition matters narratively. "
            "Multiplier on suspense / surprise / irony contributions."
        ),
    )
    truth_at_fabula: Dict[int, bool] = Field(
        default_factory=dict,
        description=(
            "Ground-truth commitments keyed by fabula_time. ``{1700: True}`` "
            "means the proposition becomes true at fabula 1700; before that "
            "the truth value is undetermined for surprise-scoring purposes."
        ),
    )
    state_timeline: List["PropositionSnapshot"] = Field(
        default_factory=list,
        description=(
            "Chronological snapshots of mutable framing fields (stakes, "
            "audience_default_prior, description) through the story. Empty "
            "= proposition's framing unchanged throughout narrative. The "
            "separate ``truth_at_fabula`` mapping continues to carry the "
            "ground-truth commitments."
        ),
    )


class Concern(AMWNNode):
    """A standing fear or desire on the part of a single entity.

    Replaces the inferred-at-runtime ``_bucket_event_for_focal``
    classification with a first-class ledger. Each concern names a
    ``Proposition`` whose realisation the entity desires or dreads,
    plus a per-entity salience that captures *this character's*
    weighting of *this concern* (Lear cares about irrelevance more
    than Banquo cares about it).

    ``counter_concern_ids`` lets us represent ambivalence as paired
    fear/desire concerns over the same proposition; the dashboard can
    surface 'torn between X and ¬X' when both members are above
    threshold.

    Inherits ``world_id`` from :class:`AMWNNode` so Rung-2/3
    interventions can install shadow-world concerns (e.g. a
    counterfactual Macbeth who *did not* desire the crown can be
    represented by a shadow concern with ``polarity='fear'`` over
    ``PROP_MACBETH_BECOMES_KING``) without mutating the factual
    character's ledger.
    """
    concern_id: str = Field(description="Unique CCN_ id.")
    proposition_id: str = Field(
        description="PROP_ id whose realisation realises or averts this concern.",
    )
    polarity: Literal["desire", "fear"] = Field(
        description="Whether this entity wants the proposition true (desire) or false (fear).",
    )
    kind: Optional[str] = Field(
        default=None,
        description=(
            "Optional harm/benefit-kind label (e.g. 'betrayal', 'abandonment', "
            "'irrelevance', 'death'). Drawn from the same vocabulary as the "
            "causal-physics ``MECHANISM_TRAIT_MAP`` plus narrative-affect "
            "extensions. Feeds per-entity salience overrides."
        ),
    )
    salience: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="This entity's personal weighting of this concern.",
    )
    activation_fabula_window: Optional[List[int]] = Field(
        default=None,
        description=(
            "Optional [start, end] fabula-time window during which this concern "
            "is active. None = always active. Lets us model concerns that arise "
            "mid-story (Lear's irrelevance only after the abdication)."
        ),
    )
    counter_concern_ids: List[str] = Field(
        default_factory=list,
        description="Other CCN_ ids forming an ambivalent pair with this concern.",
    )
    state_timeline: List["ConcernSnapshot"] = Field(
        default_factory=list,
        description=(
            "Chronological snapshots of mutable concern fields (salience, "
            "polarity, activation_fabula_window, counter_concern_ids, kind) "
            "through the story. Empty = concern unchanged throughout narrative."
        ),
    )

    @property
    def ambivalence_score(self) -> float:
        """Heuristic ambivalence weight derived from ``counter_concern_ids``.

        Returns ``salience`` when the concern carries at least one counter
        link (the concern is one half of an ambivalent pair) and 0.0
        otherwise. Downstream affect scorers multiply this onto inner-
        conflict suspense; concerns with no counter link contribute 0.
        """
        if not self.counter_concern_ids:
            return 0.0
        return float(self.salience)


class BeliefConfidenceShift(BaseModel):
    """Confidence/inertia overwrite on a single existing :class:`Belief`,
    folded onto :class:`EntityStateSnapshot` so the per-fabula timeline
    carries Affect-driven confidence drift alongside Consequences-driven
    creates / invalidates.

    Match is by ``target_id`` (and ``proposition_id`` when supplied). When
    no matching belief exists at the snapshot's fabula tick the entry is
    skipped during reconstruction — Affect must not forge new beliefs;
    that's Consequences' job via ``new_beliefs``.
    """
    target_id: str = Field(description="target_id of the belief to update.")
    proposition_id: Optional[str] = Field(
        default=None,
        description="Optional PROP_ id discriminator when target_id is ambiguous.",
    )
    new_confidence: float = Field(ge=0.0, le=1.0)
    new_inertia: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EntityStateSnapshot(BaseModel):
    """A point-in-time snapshot of an entity's mutable state.

    Stored on ``Entity.state_timeline`` in fabula_time order.
    Only *changed* fields need be populated — reconstruction merges
    each snapshot atop the previous accumulated state.
    """
    world_id: Literal["factual", "shadow"] = Field(
        default="factual",
        description=(
            "AMWN branch this snapshot belongs to. Snapshots produced by a "
            "shadow merge are tagged ``shadow`` so consumers walking a live "
            "entity's ``state_timeline`` can filter out off-branch entries."
        ),
    )
    fabula_time: int = Field(description="fabula_time this snapshot is valid from.")
    triggered_by: Optional[str] = Field(default=None, description="EVT_ ID that caused this state change.")
    traits: Dict[str, "TraitVector"] = Field(
        default_factory=dict,
        description="Trait values at this point. Only include traits that changed.",
    )
    beliefs_added: List["Belief"] = Field(
        default_factory=list,
        description="New beliefs formed at this point.",
    )
    beliefs_invalidated: List[str] = Field(
        default_factory=list,
        description="target_ids of beliefs shattered/superseded at this point.",
    )
    belief_confidence_updates: List["BeliefConfidenceShift"] = Field(
        default_factory=list,
        description=(
            "Per-belief confidence (and optional inertia) overwrites at "
            "this fabula tick. Authored by the Phase B4 Affect Agent's "
            "``belief_snapshots`` channel — confidence drift on existing "
            "beliefs triggered by emotional / framing events. Distinct "
            "from ``beliefs_added`` (Consequences creates a new belief) "
            "and ``beliefs_invalidated`` (Consequences shatters a belief). "
            "Replayed by :func:`reconstruct_entity_at` so per-character "
            "Bayesian-surprise diagnostics see the drift at the right "
            "fabula_time, not only the snapshot-flat overwrite."
        ),
    )
    status: Optional[Literal["healthy", "injured", "ill", "dead", "unconscious"]] = Field(
        default=None, description="New status if changed, else null.",
    )
    location_id: Optional[str] = Field(
        default=None, description="New location if entity moved, else null.",
    )


class ObjectStateSnapshot(BaseModel):
    """A point-in-time snapshot of a :class:`NarrativeObject`'s mutable state.

    Stored on ``NarrativeObject.state_timeline`` in fabula_time order.
    Mirrors :class:`EntityStateSnapshot`: only *changed* fields need
    be populated — reconstruction merges each snapshot atop the
    previous accumulated state.

    Object movement (``location_id``), ownership transfer
    (``owner_id``), and per-property mutations (``properties_set`` /
    ``properties_unset``) all flow through this single snapshot type
    so the per-fabula timeline can be reconstructed by
    :func:`reconstruct_object_at`. Without it,
    :class:`NarrativeObject` would carry only a static
    ``location_id`` / ``owner_id`` and downstream readers (AMWN
    sandbox, brief assembler, auditor) would see the *initial* state
    no matter what fabula_time they query — exactly the asymmetry
    that motivates this class relative to :class:`Entity` and
    :class:`GlobalTrait`.
    """
    world_id: Literal["factual", "shadow"] = Field(
        default="factual",
        description=(
            "AMWN branch this snapshot belongs to. Snapshots produced by a "
            "shadow merge are tagged ``shadow`` so consumers walking a live "
            "object's ``state_timeline`` can filter out off-branch entries."
        ),
    )
    fabula_time: int = Field(description="fabula_time this snapshot is valid from.")
    triggered_by: Optional[str] = Field(
        default=None,
        description="EVT_ ID that caused this object change (pickup, drop, transfer, mutation).",
    )
    location_id: Optional[str] = Field(
        default=None,
        description=(
            "New LOC_ id if the object moved (was dropped, placed, or relocated). "
            "Null when the object is now held — see ``owner_id``. The reconstruction "
            "helper uses an explicit ``set_location_null`` flag to disambiguate "
            "'no change' (this field omitted) from 'cleared because picked up' "
            "(``set_location_null=True``)."
        ),
    )
    owner_id: Optional[str] = Field(
        default=None,
        description=(
            "New ENT_ id if ownership transferred (pickup, gift, theft, "
            "inheritance). Null on drop — see ``set_owner_null``."
        ),
    )
    set_location_null: bool = Field(
        default=False,
        description=(
            "When True, explicitly clear ``NarrativeObject.location_id`` "
            "(object was picked up — it now lives in an inventory). "
            "Disambiguates from ``location_id=None`` meaning 'no change "
            "to location this tick'."
        ),
    )
    set_owner_null: bool = Field(
        default=False,
        description=(
            "When True, explicitly clear ``NarrativeObject.owner_id`` "
            "(object was dropped or placed). Disambiguates from "
            "``owner_id=None`` meaning 'no change to owner this tick'."
        ),
    )
    properties_set: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Property keys to overwrite (e.g. ``{'state': 'poisoned'}``). "
            "Merged atop the accumulated property dict."
        ),
    )
    properties_unset: List[str] = Field(
        default_factory=list,
        description="Property keys to remove from the accumulated property dict.",
    )


class WorldTraitSnapshot(BaseModel):
    """A point-in-time snapshot of a world trait's mutable state.

    Stored on ``GlobalTrait.state_timeline`` in fabula_time order.
    Only *changed* fields need be populated — reconstruction merges
    each snapshot atop the previous accumulated state.
    """
    world_id: Literal["factual", "shadow"] = Field(
        default="factual",
        description=(
            "AMWN branch this snapshot belongs to. Snapshots produced by a "
            "shadow merge are tagged ``shadow`` so consumers walking a live "
            "trait's ``state_timeline`` can filter out off-branch entries."
        ),
    )
    fabula_time: int = Field(description="fabula_time this snapshot is valid from.")
    triggered_by: Optional[str] = Field(default=None, description="EVT_ ID that caused this world change.")
    magnitude: Optional["TraitVector"] = Field(
        default=None,
        description="Updated magnitude (value + inertia) if changed, else null.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Updated prose description if the nature of the trait changed, else null.",
    )


class PropositionSnapshot(BaseModel):
    """A point-in-time snapshot of a proposition's mutable framing fields.

    Stored on ``Proposition.state_timeline`` in fabula_time order.
    Mirrors the snapshot/replay pattern used by ``Entity`` and
    ``GlobalTrait`` so audience-facing weights (``stakes``,
    ``audience_default_prior``) can evolve as the narrative escalates
    or de-escalates a question. Only *changed* fields need be
    populated — reconstruction merges each snapshot atop the previous
    accumulated state.

    The proposition's ground-truth value continues to live on the
    separate ``truth_at_fabula`` mapping; this timeline is for the
    *narrative weighting* of the proposition, not its truth.
    """
    world_id: Literal["factual", "shadow"] = Field(
        default="factual",
        description=(
            "AMWN branch this snapshot belongs to. Snapshots produced by a "
            "shadow merge are tagged ``shadow`` so consumers walking a live "
            "proposition's ``state_timeline`` can filter out off-branch entries."
        ),
    )
    fabula_time: int = Field(description="fabula_time this snapshot is valid from.")
    triggered_by: Optional[str] = Field(
        default=None,
        description="EVT_ ID that caused this framing change (revelation, escalation, …).",
    )
    stakes: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Updated narrative stakes if changed, else null.",
    )
    audience_default_prior: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Updated audience prior if the narrator has reframed the proposition, else null.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Updated human-readable label if the proposition's framing shifted, else null.",
    )


class ConcernSnapshot(BaseModel):
    """A point-in-time snapshot of a concern's mutable fields.

    Stored on ``Concern.state_timeline`` in fabula_time order.
    Mirrors the snapshot/replay pattern used by ``Entity`` and
    ``GlobalTrait`` so per-entity ``salience`` can ramp / decay,
    ``polarity`` can flip on a desire→fear reversal, and the
    activation window or counter-concern set can be rewritten
    mid-story without losing the original baseline. Only *changed*
    fields need be populated — reconstruction merges each snapshot
    atop the previous accumulated state.
    """
    world_id: Literal["factual", "shadow"] = Field(
        default="factual",
        description=(
            "AMWN branch this snapshot belongs to. Snapshots produced by a "
            "shadow merge are tagged ``shadow`` so consumers walking a live "
            "concern's ``state_timeline`` can filter out off-branch entries."
        ),
    )
    fabula_time: int = Field(description="fabula_time this snapshot is valid from.")
    triggered_by: Optional[str] = Field(
        default=None,
        description="EVT_ ID that caused this concern shift.",
    )
    salience: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Updated personal salience if changed, else null.",
    )
    polarity: Optional[Literal["desire", "fear"]] = Field(
        default=None,
        description="Updated polarity if a desire→fear (or vice versa) reversal occurred, else null.",
    )
    activation_fabula_window: Optional[List[int]] = Field(
        default=None,
        description="Updated [start, end] activation window if rewritten, else null.",
    )
    counter_concern_ids: Optional[List[str]] = Field(
        default=None,
        description="Updated set of counter-concern CCN_ ids if the ambivalence pairing changed, else null.",
    )
    kind: Optional[str] = Field(
        default=None,
        description="Updated harm/benefit-kind label if the concern was reclassified, else null.",
    )


class GlobalTrait(AMWNNode):
    """A world-level fact, law, or condition that constrains or enables characters.

    Represents Greimas' 'Power' actant — an abstract force that determines
    whether subjects can achieve their goals. Examples: surveillance state,
    magic system rules, wartime economy, social class rigidity.
    """
    id: str = Field(description="Unique ID with WORLD_ prefix, e.g., WORLD_SURVEILLANCE_STATE")
    name: str = Field(description="Human-readable name, e.g., 'Totalitarian Surveillance'")
    description: str = Field(description="Prose description of the world-level fact and its narrative role.")
    category: str = Field(
        description="Category of world trait: 'governance', 'magic_system', 'environment', "
                    "'social_structure', 'technology', 'ecology', 'economy', 'cosmology'."
    )
    magnitude: TraitVector = Field(
        description="value = intensity (0.0–1.0, how strongly this constrains characters), "
                    "inertia = resistance to change (0.0–1.0, how hard to shift this world fact. "
                    "Physics laws ~0.95, political situations ~0.4)."
    )
    affected_domains: List[str] = Field(
        description=(
            "Which causal mechanism categories this trait influences. "
            "Use the canonical short keys from MECHANISM_TRAIT_MAP: "
            "'physical', 'psychological', 'epistemic', 'social', 'emotional', "
            "'informational', 'betrayal'. Common off-list labels "
            "('economic', 'political', 'moral', 'ideological', 'religious', "
            "'supernatural', 'magical', …) are auto-mapped onto these via "
            "_DOMAIN_ALIASES so legacy fixtures keep routing correctly, but "
            "new content should emit the canonical form directly."
        ),
    )
    state_timeline: List[WorldTraitSnapshot] = Field(
        default_factory=list,
        description="Chronological snapshots of state changes through the story. "
                    "Empty = trait unchanged throughout narrative.",
    )
    proposition_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional PROP_ id this world trait reifies. When set, the trait"
            " doubles as a first-class proposition the audience can hold beliefs"
            " about (e.g. 'the prophecy is binding', 'the empire is watching')."
            " Pearl-Rung-2 BeliefMutation clamps on the linked proposition"
            " surface as additional WorldTraitSnapshot entries so propagation"
            " reflects the surgical change. Concerns/beliefs may target either id;"
            " the reconciler resolves cross-references at merge time."
        ),
    )

    _coerce_domains = field_validator("affected_domains", mode="before")(
        lambda v: _coerce_domain_list(v)
    )


# --- 2. THE NODES (The Nouns) ---
class Location(AMWNNode):
    node_type: Literal["Location"] = "Location"
    name: str
    description: str
    ambient_state: Dict[str, AmbientVector] = Field(default_factory=dict, description="e.g., {'temperature': AmbientVector(value=0.8, volatility=0.3)}")

class NarrativeObject(AMWNNode):
    id: str = Field(description="Unique ID, e.g., OBJ_DAGGER")
    name: str
    location_id: Optional[str] = Field(description="Where is it? Null if in an inventory.")
    owner_id: Optional[str] = Field(description="Who is holding it? Null if on the ground.")
    properties: Dict[str, str] = Field(default_factory=dict, description="e.g., {'state': 'poisoned'}")
    affordances: List[Affordance]
    state_timeline: List["ObjectStateSnapshot"] = Field(
        default_factory=list,
        description=(
            "Chronological snapshots of mutable object state through the "
            "story (movement, ownership transfers, property mutations). "
            "Empty = the object's location/owner/properties are unchanged "
            "throughout the narrative. Mirrors the snapshot pattern used "
            "by :class:`Entity` and :class:`GlobalTrait`; replayed by "
            ":func:`reconstruct_object_at` so AMWN sandboxes, ego graphs, "
            "the brief assembler and the auditor see the time-correct "
            "object position rather than only its initial value."
        ),
    )

class Entity(AMWNNode):
    id: str = Field(description="Unique ID, e.g., ENT_MACBETH")
    name: str
    location_id: str = Field(description="Initial location (pre-story or earliest known).")
    status: Literal["healthy", "injured", "ill", "dead", "unconscious"]
    traits: Dict[str, TraitVector] = Field(description="Initial multidimensional psychology (pre-story baseline).")
    beliefs: List[Belief] = Field(default_factory=list, description="Initial epistemic state.")
    concerns: List[Concern] = Field(
        default_factory=list,
        description=(
            "Standing fears and desires this entity carries through the "
            "narrative. Drives the affective-physics layer's threat / hope "
            "surfacing per entity. Empty for legacy worlds; populated by "
            "the optional concern-extraction ingestion step."
        ),
    )
    constants: List[str] = Field(default_factory=list, description="Immutable boolean tags, e.g., ['blind', 'undead']")
    state_timeline: List[EntityStateSnapshot] = Field(
        default_factory=list,
        description="Chronological snapshots of state changes through the story. "
                    "Empty = entity unchanged or legacy data.",
    )

class EventNode(AMWNNode):
    id: str = Field(description="Unique ID, e.g., EVT_DUNCAN_MURDER")
    fabula_time: int = Field(
        description="The strict chronological order (e.g., Year 1000). Used for Causal Physics."
    )
    syuzhet_index: int = Field(
        description="The sequence this appears in the text (e.g., Chapter 4, Paragraph 2). Used for Suspense."
    )
    event_type: Literal["choice", "outcome", "revelation", "utterance"] = Field(
        description=(
            "choice = a deliberate decision; outcome = a physical/situational "
            "happening; revelation = **reader-side narrator disclosure only** \u2014 "
            "a moment where the *narration* (not a character) lifts the veil "
            "on a previously-hidden fact (omniscient narrator reveal, "
            "unmasked-killer beat, chapter-end twist). Character-to-character "
            "disclosures are NOT revelations \u2014 they are utterances. "
            "utterance = a discrete speech-act / message transmitted between "
            "characters (fabula-side, the Social Agent's exclusive output). "
            "Utterance and revelation are distinct: an utterance can occur "
            "long before its content is revealed to the audience."
        ),
    )
    actor_ids: List[str] = Field(default_factory=list, description="Who did it? Empty if natural event. Supports joint actions (e.g., ['ENT_MACBETH', 'ENT_LADY_MACBETH']).")
    target_ids: List[str] = Field(default_factory=list, description=(
        "Who/what was acted upon? e.g., ['ENT_DUNCAN'] in a murder event. "
        "Supports diffuse effects. For ``event_type='utterance'``, "
        "``target_ids`` is the set of entities/objects/EVENTS the speech-act "
        "is *about* (the utterance's referents) — not its downstream causal "
        "effects. Any EVT_ id in an utterance's ``target_ids`` must have "
        "``fabula_time <= utterance.fabula_time`` UNLESS the utterance is "
        "``truth_value='performative'`` (prophecies, vows, orders may "
        "reference future events they posit/commit to). Causal effects of an "
        "utterance belong in ``causal_topology`` as ``chain_reaction`` edges."
    ))
    description: str

    # --- Utterance / revelation payload (optional, mainly for event_type='utterance' or 'revelation') ---
    content: Optional[str] = Field(
        default=None,
        description=(
            "For utterance/revelation events: the proposition transmitted or "
            "revealed (e.g., 'Kurtz has gone rogue in Cambodia'). Distinct "
            "from ``description`` (which narrates the event); ``content`` is "
            "the *epistemic payload* downstream beliefs reference. Null for "
            "choice/outcome events with no informational content."
        ),
    )
    via_channel_id: Optional[str] = Field(
        default=None,
        description=(
            "For utterance events: optional CHN_ id of the standing channel the "
            "message travelled over (telephone, mind-link, classified pipeline). "
            "Null when the event is its own channel — direct speech, in-person "
            "observation, an isolated letter."
        ),
    )
    speaker_id: Optional[str] = Field(
        default=None,
        description=(
            "For utterance events: the single ENT_/OBJ_ id of the speaker. "
            "Disambiguates among joint ``actor_ids`` (e.g. a chorus where "
            "only one character actually voices the line). Should be a member "
            "of ``actor_ids`` when both are populated."
        ),
    )
    addressee_ids: List[str] = Field(
        default_factory=list,
        description=(
            "For utterance events: ENT_ ids the speaker intends to reach. "
            "Distinct from ``target_ids`` (which is generic 'acted upon'); "
            "addressees are the *intended* recipients, while overhearers "
            "are picked up via channel intelligibility / location overlap."
        ),
    )
    truth_value: Optional[Literal["true", "false", "unknown", "performative"]] = Field(
        default=None,
        description=(
            "For utterance events: whether ``content`` is true in the storyworld. "
            "'true' = sincere accurate assertion, 'false' = lie / mistake / "
            "deception, 'unknown' = speaker themselves uncertain, 'performative' "
            "= speech-act not truth-apt (a vow, an order, a curse). Drives the "
            "Bayesian abduction layer's confidence on the resulting beliefs."
        ),
    )
    intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description=(
            "Narrative weight of the event in [0, 10]. Drives mass-weighted "
            "scoring downstream (dramatic-irony surface, suspense stakes, "
            "directive selection). 1.0 is the neutral default; 2.0–4.0 marks "
            "a pivotal beat (a confession, a death, a betrayal); 5.0+ is "
            "reserved for cataclysmic plot-turns (the murder of Duncan, the "
            "dropping of the bomb). Scenic / connective beats sit at 0.5–1.0."
        ),
    )

    # --- Proposition links (optional, populated by the affect sub-stage) ---
    resolves_proposition_ids: List[str] = Field(
        default_factory=list,
        description=(
            "PROP_ ids whose ``truth_at_fabula`` this event commits. Set by "
            "the per-chunk affect sub-stage when an outcome event resolves "
            "an open question (e.g. EVT_DUNCAN_MURDER resolves "
            "PROP_DUNCAN_DEAD = True). The reconciler reads this list to "
            "populate ``Proposition.truth_at_fabula``."
        ),
    )
    asserts_proposition_id: Optional[str] = Field(
        default=None,
        description=(
            "For utterance / revelation events: the PROP_ id whose truth "
            "the event asserts. Combined with ``truth_value`` the affect "
            "layer derives per-agent belief deltas without re-parsing "
            "``content``."
        ),
    )
    denies_proposition_id: Optional[str] = Field(
        default=None,
        description=(
            "For utterance / revelation events: the PROP_ id whose truth "
            "the event denies. Mutually exclusive with "
            "``asserts_proposition_id``; an utterance that both asserts P "
            "and denies Q should split into two events."
        ),
    )
    superseded_by_event_id: Optional[str] = Field(
        default=None,
        description=(
            "When non-null, this event has been overridden by another event "
            "(typically because a counterfactual was promoted onto the "
            "factual mainline via ``branch_policy='mainline'`` with an "
            "explicit supersession map). Downstream physics, audit and "
            "render readers should treat the named successor as canonical "
            "while preserving the historical event for replay / audit."
        ),
    )

    # --- Spatial anchor (optional, but the canonical answer to "where") ---
    at_location_id: Optional[str] = Field(
        default=None,
        description=(
            "LOC_ id where the event physically takes place. Optional for "
            "backwards compatibility — when null, downstream consumers fall "
            "back to the primary actor's reconstructed location at "
            "``fabula_time`` via :func:`event_location_at`. The implicit "
            "co-presence rule is: every actor and non-channel target of an "
            "event must be at ``at_location_id`` at ``fabula_time`` UNLESS "
            "the event is an utterance with ``via_channel_id`` and the "
            "participant is reached through that channel. The auditor and "
            "ingestion validator enforce this; the directive assembler "
            "emits MUST_DEPICT_AT / MUST_BE_PRESENT constraints from it; "
            "the Map sub-tab anchors the event glyph here."
        ),
    )

    # Tier 5 #17: ``choice`` events are by definition deliberate decisions —
    # they should have at least one actor. ``outcome`` events legitimately
    # admit empty ``actor_ids`` (natural disasters, ambient happenings).
    # ``utterance`` events may be authored by a speaker without populating
    # ``actor_ids`` (chorus-anonymous quotes), so we tolerate empty there
    # iff ``speaker_id`` is set. ``revelation`` events are narrator-side
    # disclosures and need no actor.
    #
    # We emit a logger warning rather than raising — auto-repair runs
    # downstream and the warning is captured by ingestion_diagnostics
    # for surface in the Causality / Ingestion-Warnings UI panel.
    @model_validator(mode="after")
    def _warn_on_actorless_intentional_event(self) -> "EventNode":
        import logging as _logging
        _log = _logging.getLogger("shadow_loom.ingestion")
        # Dedupe per (event_id, warning_kind) — pydantic's
        # ``model_copy`` re-runs validators, so a single ingestion
        # pass that remaps fabula_time / referent ids would otherwise
        # log the same warning 3-4× per event. The seen-set lives on
        # the class so it persists across instances within a process.
        seen = type(self).__dict__.get("_actorless_warn_seen")
        if seen is None:
            seen = set()
            try:
                setattr(type(self), "_actorless_warn_seen", seen)
            except Exception:
                seen = None  # frozen / locked class — fall back to noisy.
        if self.event_type == "choice" and not self.actor_ids:
            key = ("choice", self.id)
            if seen is None or key not in seen:
                _log.warning(
                    "[Validator·EventNode] choice event %r has no actor_ids "
                    "(a deliberate decision requires at least one decider).",
                    self.id,
                )
                if seen is not None:
                    seen.add(key)
        elif (
            self.event_type == "utterance"
            and not self.actor_ids
            and not self.speaker_id
        ):
            key = ("utterance", self.id)
            if seen is None or key not in seen:
                _log.warning(
                    "[Validator·EventNode] utterance event %r has neither "
                    "actor_ids nor speaker_id.",
                    self.id,
                )
                if seen is not None:
                    seen.add(key)
        # Spatial-anchor sanity for utterances: a face-to-face speech-act
        # needs *somewhere* to take place. If neither a channel nor an
        # explicit ``at_location_id`` is given but the speaker has named
        # addressees, downstream co-presence audit will have nothing to
        # check against — warn so ingestion can repair before the brief
        # is assembled.
        if (
            self.event_type == "utterance"
            and self.via_channel_id is None
            and self.at_location_id is None
            and (self.speaker_id or self.actor_ids)
            and self.addressee_ids
        ):
            key = ("utterance_no_anchor", self.id)
            if seen is None or key not in seen:
                _log.warning(
                    "[Validator·EventNode] utterance event %r has neither "
                    "via_channel_id nor at_location_id but has addressees "
                    "(downstream co-presence audit will be ungrounded).",
                    self.id,
                )
                if seen is not None:
                    seen.add(key)
        return self

class AMWNEdge(BaseModel):
    """Base class for all topology edges. Distinct from AMWNNode."""
    world_id: Literal["factual", "shadow"] = Field(
        default="factual",
        description="Allows edges to exist only in shadow branches.",
    )

# ==========================================
# 1. THE STANDING CAPABILITY: Channel
# ==========================================
class Channel(AMWNNode):
    """A persistent communication *capability* between participants.

     a Channel models who
    *can* communicate with whom over what medium for the duration of an
    interval. Discrete *messages* are first-class
    :class:`EventNode` instances with ``event_type='utterance'``,
    optionally referencing a Channel via ``via_channel_id``.

    Promoted to a node (not an edge) so that:
      * EventNodes and Beliefs can name a channel by id;
      * the channel itself can carry per-participant intelligibility
      * counterfactual surgery on the channel (cipher broken, line tapped,
        bond severed) cleanly cascades through ``Belief.acquired_via_channel_id``
        and ``EventNode.via_channel_id``.
    """
    id: str = Field(description="Unique ID with CHN_ prefix, e.g., CHN_DOSSIER_PIPELINE.")
    name: str = Field(description="Human-readable name, e.g., 'MACV-SOG Classified Dossier Pipeline'.")
    medium: str = Field(
        description=(
            "The channel medium: e.g., 'telephone', 'telepathy', 'classified_pipeline', "
            "'mail_correspondence', 'mind_link', 'shared_language', 'magic_mirror'."
        ),
    )
    participant_ids: List[str] = Field(
        description=(
            "ENT_ / OBJ_ ids of channel participants. n-ary; the channel is "
            "undirected by default — see ``directionality`` for asymmetric "
            "channels (broadcasts, simplex links)."
        ),
    )
    directionality: Literal["broadcast", "duplex", "simplex"] = Field(
        default="duplex",
        description=(
            "'duplex' = any participant can speak to any other (default for 2-party); "
            "'broadcast' = first participant speaks, the rest only listen "
            "(loudspeaker, telescreen, public proclamation); "
            "'simplex' = first→second only (one-way courier, dead-drop)."
        ),
    )
    intelligibility: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-participant decode probability \u2208 [0, 1]. A missing key "
            "means fully intelligible (1.0). 0.0 = opaque (foreign language, "
            "unbroken cipher, jargon outside the listener's competence); "
            "0.5 = partial (overhearing through a wall, lossy translation). "
            "Replaces the legacy ``is_encrypted`` boolean: encryption is "
            "modelled as low intelligibility for non-keyholders."
        ),
    )
    established_at_fabula: int = Field(
        default=0,
        description="Fabula tick the channel becomes available (0 = pre-story).",
    )
    terminated_at_fabula: Optional[int] = Field(
        default=None,
        description=(
            "Fabula tick the channel is severed (line cut, bond broken, courier killed). "
            "Null = still active at the end of the narrative."
        ),
    )
    discovered_at_syuzhet: Optional[int] = Field(
        default=None,
        description=(
            "Syuzhet index at which the audience (or POV entity) first "
            "becomes aware of this channel. ``None`` means the channel "
            "is overt from the start of the narrative; a positive value "
            "marks the reveal beat at which a previously-hidden channel "
            "(an undisclosed cipher line, a secret pact, a back-channel "
            "tip-off) is brought on-page. Distinct from "
            "``established_at_fabula`` (when the channel exists in the "
            "storyworld) so dramatic-irony and mystery scoring can "
            "reason about *audience knowledge* of the channel without "
            "touching its in-world lifecycle."
        ),
    )
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description=(
            "Engine's certainty in the *extraction* of this channel from the source text. "
            "'strong' = the channel is named/used on-page repeatedly, 'moderate' = "
            "the channel is implied by repeated communication, 'weak' = inferred from "
            "a single exchange. Feeds the directive-assembly epistemic abduction layer."
        ),
    )

    _coerce_es = field_validator("evidence_strength", mode="before")(
        lambda v: _coerce_evidence_strength(v)
    )

# ==========================================
# 2. THE UNIVERSAL CAUSAL LINK: CausalEdge
# ==========================================
class CausalEdge(AMWNEdge):
    """
    A universal causal link that can bridge Events, States, Traits, and Affordances.

    Supports four modalities of narrative causality:
      - chain_reaction:        Event → Event  (direct sequential triggers)
      - mutation:              Event → State   (actions leave marks on the world)
      - affordance_gate:       State → Event   (states enable or prevent events)
      - ambient_propagation:   State → State   (background physics without events)
    """
    source_id: str = Field(
        description="The cause. Can be an EVT_ (Event), ENT_ (Trait/State), LOC_ (Ambient State), OBJ_ (Affordance), or WORLD_ (Global Trait)."
    )
    target_id: str = Field(
        description="The effect. The node that is triggered or mutated."
    )

    causality_type: Literal[
        "chain_reaction",
        "mutation",
        "mutation_social",
        "affordance_gate",
        "ambient_propagation",
    ] = Field(
        description=(
            "The modality of the causal link: "
            "chain_reaction = Event→Event, "
            "mutation = Event→State (traits/status), "
            "mutation_social = Event→Relationship (affinity/fear/power), "
            "affordance_gate = State→Event, "
            "ambient_propagation = State→State."
        ),
    )

    causal_force: float = Field(
        default=5.0,
        description="0.0 to 10.0. The Impact magnitude this cause applies to the target.",
    )

    mechanism: str = Field(
        description=(
            "The 'how' of the causality. Canonical short keys (recommended): "
            "'physical', 'psychological', 'epistemic', 'social', 'emotional', "
            "'informational', 'betrayal' — these route the impulse onto the "
            "matching trait family via causal_physics.MECHANISM_TRAIT_MAP. "
            "Long-form aliases ('physical_force', 'epistemic_revelation', "
            "'social_coercion') are auto-canonicalized. Off-list labels "
            "('kinetic', 'chemical', 'seduction', 'coercion', 'deduction') "
            "are tolerated as an explicit escape hatch — they bypass "
            "mechanism-trait routing entirely (no 20% fallback penalty), so "
            "only use them when you specifically want the impulse to apply "
            "uniformly across all of the target's traits."
        ),
    )
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description="Statistical confidence for Bayes variance: weak=high variance, strong=low variance.",
    )

    _coerce_es = field_validator("evidence_strength", mode="before")(
        lambda v: _coerce_evidence_strength(v)
    )
    _coerce_mech = field_validator("mechanism", mode="before")(
        lambda v: _coerce_mechanism(v)
    )

    propagation_delay: int = Field(
        default=0,
        description="Number of fabula ticks between cause firing and effect manifesting. 0 = instantaneous.",
    )

    fabula_time: int = Field(description="The exact physics tick this cause took effect.")

    trait_target: Optional[str] = Field(
        default=None,
        description="For mutation edges: the specific trait affected, e.g., 'guilt'. "
                    "For mutation_social edges: the relationship metric, e.g., 'affinity', 'fear', 'power_dynamic'. "
                    "Null for non-mutation types.",
    )
    trait_delta: Optional[float] = Field(
        default=None,
        description="For mutation edges: signed magnitude of trait change (-1.0 to 1.0). "
                    "For mutation_social edges: signed magnitude of metric change. "
                    "Null for non-mutation types.",
    )
    rel_counterpart_id: Optional[str] = Field(
        default=None,
        description="For mutation_social edges only: the other entity in the relationship dyad. "
                    "source_id is the causal trigger (EVT_), target_id is the perspective entity (ENT_), "
                    "rel_counterpart_id is the other entity (ENT_). Null for non-social types.",
    )

    @model_validator(mode="after")
    def _check_causality_type_matches_ids(self) -> "CausalEdge":
        src_is_event = self.source_id.startswith("EVT_")
        src_is_world = self.source_id.startswith("WORLD_")
        tgt_is_event = self.target_id.startswith("EVT_")
        ct = self.causality_type

        if src_is_event and ct not in ("chain_reaction", "mutation", "mutation_social"):
            raise ValueError(
                f"source_id '{self.source_id}' is an event — causality_type must be "
                f"'chain_reaction', 'mutation', or 'mutation_social', got '{ct}'."
            )
        # WORLD_ nodes are state-like: allow affordance_gate and ambient_propagation
        if not src_is_event and not src_is_world and ct not in ("affordance_gate", "ambient_propagation"):
            raise ValueError(
                f"source_id '{self.source_id}' is a state node — causality_type must be "
                f"'affordance_gate' or 'ambient_propagation', got '{ct}'."
            )
        if src_is_world and ct not in ("affordance_gate", "ambient_propagation", "mutation", "chain_reaction"):
            raise ValueError(
                f"source_id '{self.source_id}' is a world trait — causality_type must be "
                f"'affordance_gate', 'ambient_propagation', 'mutation', or 'chain_reaction', got '{ct}'."
            )
        # WORLD_ → WORLD_ cross-trait dependencies (e.g. WARTIME → SURVEILLANCE_STATE).
        # The destination is a state node, not an event, so chain_reaction / mutation
        # both behave structurally; the runtime treats them like any other state-to-state
        # edge. Permit them explicitly so the validator does not block authored W→W edges.
        tgt_is_world = self.target_id.startswith("WORLD_")
        if src_is_world and tgt_is_world and ct not in ("chain_reaction", "mutation"):
            raise ValueError(
                f"WORLD_→WORLD_ edge '{self.source_id}'→'{self.target_id}' must use "
                f"causality_type 'chain_reaction' or 'mutation', got '{ct}'."
            )
        if tgt_is_event and ct not in ("chain_reaction", "affordance_gate"):
            raise ValueError(
                f"target_id '{self.target_id}' is an event — causality_type must be "
                f"'chain_reaction' or 'affordance_gate', got '{ct}'."
            )
        if not tgt_is_event and ct not in ("mutation", "mutation_social", "ambient_propagation", "chain_reaction"):
            raise ValueError(
                f"target_id '{self.target_id}' is a state node — causality_type must be "
                f"'mutation', 'mutation_social', 'ambient_propagation', or 'chain_reaction', got '{ct}'."
            )
        # ``chain_reaction`` against a state-node target is only legal when the
        # source is also a WORLD_ trait (cross-trait dependency, e.g.
        # WARTIME → SURVEILLANCE_STATE). Forbid it for entity/event sources to
        # keep the structural contract tight.
        if not tgt_is_event and ct == "chain_reaction" and not src_is_world:
            raise ValueError(
                f"chain_reaction into state-node target '{self.target_id}' is only "
                f"permitted from a WORLD_ source; got source '{self.source_id}'."
            )
        # mutation_social requires rel_counterpart_id
        if ct == "mutation_social" and not self.rel_counterpart_id:
            raise ValueError(
                "mutation_social edges require 'rel_counterpart_id' to identify the "
                "other entity in the relationship dyad."
            )
        return self

# ==========================================
# 3. THE ACCUMULATOR EDGE: RelationshipEdge
# ==========================================
class RelationshipMetric(BaseModel):
    """Per-axis state for one social metric on a RelationshipEdge.

    Each axis (``affinity``, ``fear``, ``power_dynamic``) drifts at its own
    rate, with its own evidence quality and its own staleness clock.
    Collapsing them onto a single edge-level inertia / evidence /
    timestamp \u2014 as the original schema did \u2014 forced a wrong-on-average
    compromise (``fear`` spikes in seconds, ``power_dynamic`` calcifies
    over years) and threw away per-metric Bayesian variance the engine
    already extracts for every ``mutation_social`` causal edge.
    """
    value: float = Field(
        description=(
            "Range depends on the metric: affinity \u2208 [-1, 1], fear \u2208 "
            "[0, 1], power_dynamic \u2208 [-1, 1]."
        ),
    )
    inertia: float = Field(
        default=0.3,
        description=(
            "0.0\u20131.0. Force required to shift this specific axis. "
            "Typical bands: fear ~0.2 (volatile), affinity ~0.4, "
            "power_dynamic ~0.6 (institutional inertia)."
        ),
    )
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description=(
            "Statistical confidence in this specific axis: "
            "weak=high variance, strong=low variance. Feeds the "
            "Bayesian abduction blend independently per metric."
        ),
    )
    last_updated_fabula: int = Field(
        default=0,
        description=(
            "Fabula tick of the most recent mutation to this axis. "
            "Counterfactual rollback uses this per-metric so that "
            "intervening on a long-stale ``power_dynamic`` does not "
            "discard recent ``fear`` mutations."
        ),
    )
    observed: bool = Field(
        default=True,
        description=(
            "True when this metric was actually extracted from the text. "
            "False = the metric is absent from the dyad and 0.0 should "
            "be read as 'unknown', not as a meaningful neutral. Defends "
            "the abduction reasoner against the zero-overload bug."
        ),
    )

    _coerce_es = field_validator("evidence_strength", mode="before")(
        lambda v: _coerce_evidence_strength(v)
    )


_REL_METRIC_NAMES = ("affinity", "fear", "power_dynamic")


def default_relationship_metrics_dict(
    *,
    primary_metric: Optional[str] = None,
    primary_value: float = 0.0,
    fabula_time: int = 0,
    evidence_strength: str = "weak",
    inertia: float = 0.3,
) -> Dict[str, dict]:
    """Build a fully-populated per-axis ``metrics`` dict for a fallback
    relationship edge created by the physics propagator or surgery.

    All three canonical axes are always emitted so downstream readers
    that iterate ``metrics.values()`` (auditor, viz, MCP, directive
    assembly) never have to special-case missing keys. The axis named
    in ``primary_metric`` carries ``primary_value`` and ``observed=True``;
    the other axes are seeded with ``value=0.0`` and ``observed=False``
    so the abduction layer can distinguish unobserved-zero from
    measured-zero.

    Returns plain dicts (not :class:`RelationshipMetric` instances)
    because callers write directly into the NetworkX edge-attr dict
    (``data["metrics"]``) without re-validating through Pydantic.
    """
    out: Dict[str, dict] = {}
    for name in _REL_METRIC_NAMES:
        is_primary = name == primary_metric
        out[name] = {
            "value": float(primary_value) if is_primary else 0.0,
            "inertia": float(inertia),
            "evidence_strength": evidence_strength if is_primary else "weak",
            "last_updated_fabula": int(fabula_time) if is_primary else 0,
            "observed": bool(is_primary),
        }
    return out


class RelationshipEdge(AMWNEdge):
    """Tracks continuous psychological and social metrics between two entities.

    Each metric (``affinity``, ``fear``, ``power_dynamic``) is stored as a
    :class:`RelationshipMetric` with its own value, inertia,
    evidence_strength, and staleness timestamp. The legacy flat fields
    (``affinity=0.7, fear=0.0, power_dynamic=0.0, inertia=0.3, ...``)
    remain accepted as constructor kwargs and as serialised JSON via a
    ``model_validator(mode="before")`` migration, and are exposed as
    read-only properties so existing readers (MCP, UI viz, generation,
    query parsing, directive assembly) continue to work unchanged.
    """
    source_entity_id: str = Field(description="Must be an ENT_ ID")
    target_entity_id: str = Field(description="Must be an ENT_ ID")

    metrics: Dict[
        Literal["affinity", "fear", "power_dynamic"],
        RelationshipMetric,
    ] = Field(
        default_factory=dict,
        description=(
            "Per-axis social metrics. Closed Literal vocabulary so the "
            "physics router (mutation_social.trait_target) and the "
            "ingestion sanitiser stay deterministic."
        ),
    )

    # --- Edge-level lifecycle (independent of per-axis freshness) ---
    established_at_fabula: Optional[int] = Field(
        default=None,
        description=(
            "Optional fabula tick at which this directed relationship "
            "first becomes active (acquaintance, alliance, marriage, "
            "etc.). ``None`` = active from the simulation start. "
            "Time-slicing readers (omniscient extract, ego graph) hide "
            "the edge from anchors before this tick."
        ),
    )
    ended_at_fabula: Optional[int] = Field(
        default=None,
        description=(
            "Optional fabula tick at which this directed relationship "
            "is permanently severed (rupture, death, exile, divorce). "
            "Distinct from per-axis staleness on "
            "``RelationshipMetric.last_updated_fabula``: a relationship "
            "with stale metrics is still *active*, just unobserved; an "
            "ended relationship no longer participates in propagation. "
            "Time-slicing readers drop the edge from anchors at or "
            "after this tick."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_fields(cls, data: Any) -> Any:
        """Coerce legacy flat constructor kwargs / serialised JSON into the
        per-metric ``metrics`` dict.

        Accepts either:
          * the new form: ``metrics={"affinity": {"value": 0.7, ...}, ...}``
          * the old form: ``affinity=0.7, fear=0.0, power_dynamic=0.0,
            inertia=0.3, evidence_strength="moderate",
            last_updated_fabula=0``

        Mixed forms are merged with new-form metrics taking precedence.
        """
        if not isinstance(data, dict):
            return data

        # Legacy edge-level fallbacks; consumed and removed so Pydantic
        # doesn't reject them as unknown fields.
        legacy_inertia = data.pop("inertia", None)
        legacy_es = data.pop("evidence_strength", None)
        legacy_ts = data.pop("last_updated_fabula", None)

        metrics: Dict[str, Any] = dict(data.get("metrics") or {})
        for name in _REL_METRIC_NAMES:
            if name not in data:
                continue
            v = data.pop(name)
            if name in metrics:
                # New-form already provided this axis; ignore the legacy value.
                continue
            entry: Dict[str, Any] = {"value": v, "observed": True}
            if legacy_inertia is not None:
                entry["inertia"] = legacy_inertia
            if legacy_es is not None:
                entry["evidence_strength"] = legacy_es
            if legacy_ts is not None:
                entry["last_updated_fabula"] = legacy_ts
            metrics[name] = entry

        # If the caller supplied edge-level fallbacks but no new-form keys,
        # backfill them onto every existing metric so behaviour matches the
        # old uniform-edge semantics.
        if metrics and (legacy_inertia is not None or legacy_es is not None or legacy_ts is not None):
            for name, m in metrics.items():
                if not isinstance(m, dict):
                    continue
                if legacy_inertia is not None and "inertia" not in m:
                    m["inertia"] = legacy_inertia
                if legacy_es is not None and "evidence_strength" not in m:
                    m["evidence_strength"] = legacy_es
                if legacy_ts is not None and "last_updated_fabula" not in m:
                    m["last_updated_fabula"] = legacy_ts

        if metrics:
            data["metrics"] = metrics
        return data

    # --- Per-metric helpers --------------------------------------------------

    def get_metric(self, name: str) -> Optional[RelationshipMetric]:
        """Return the per-metric record or ``None`` if the axis is unobserved."""
        return self.metrics.get(name)  # type: ignore[arg-type]

    def to_legacy_dict(self) -> dict:
        """Flatten to the legacy edge-attribute shape used by the
        NetworkX sandbox graph (``edge_type='relationship'`` attrs).

        Aggregation rules for the *flat* keys (consumed by code that
        only knows the legacy shape):
          * ``affinity`` / ``fear`` / ``power_dynamic`` \u2192 their per-axis
            ``value`` (0.0 when the axis is unobserved).
          * edge-level ``inertia`` \u2192 minimum across observed metrics
            (the most volatile axis dictates ease of mutation overall).
          * edge-level ``evidence_strength`` \u2192 strongest across observed
            axes (used only for visual emphasis and not for routing).
          * edge-level ``last_updated_fabula`` \u2192 maximum across observed
            axes (the most recent mutation determines the rollback horizon).

        The full per-axis ``metrics`` dict is also emitted so callers
        that *do* understand the new shape (e.g. the social
        propagator) can read per-axis inertia / evidence / staleness
        directly without losing precision to the aggregation.
        """
        d = self.model_dump()
        # Keep the per-axis structure for precision-sensitive readers.
        # ``model_dump()`` already serialises ``metrics`` as plain dicts.
        d["affinity"] = self.affinity
        d["fear"] = self.fear
        d["power_dynamic"] = self.power_dynamic
        d["inertia"] = self.inertia
        d["evidence_strength"] = self.evidence_strength
        d["last_updated_fabula"] = self.last_updated_fabula
        return d

    # --- Read-only legacy property accessors --------------------------------

    @property
    def affinity(self) -> float:
        m = self.metrics.get("affinity")  # type: ignore[arg-type]
        return m.value if m else 0.0

    @property
    def fear(self) -> float:
        m = self.metrics.get("fear")  # type: ignore[arg-type]
        return m.value if m else 0.0

    @property
    def power_dynamic(self) -> float:
        m = self.metrics.get("power_dynamic")  # type: ignore[arg-type]
        return m.value if m else 0.0

    @property
    def inertia(self) -> float:
        if not self.metrics:
            return 0.3
        return min(m.inertia for m in self.metrics.values())

    @property
    def evidence_strength(self) -> Literal["weak", "moderate", "strong"]:
        if not self.metrics:
            return "moderate"
        order = {"weak": 0, "moderate": 1, "strong": 2}
        rev = ("weak", "moderate", "strong")
        return rev[max(order[m.evidence_strength] for m in self.metrics.values())]  # type: ignore[return-value]

    @property
    def last_updated_fabula(self) -> int:
        if not self.metrics:
            return 0
        return max(m.last_updated_fabula for m in self.metrics.values())

# ==========================================
# 4. THE EPOCH EDGE: SpatialEdge
# ==========================================
class SpatialEdge(AMWNEdge):
    """Tracks the physical flow of matter (Architecture)."""
    source_id: str = Field(description="Must be a LOC_ ID")
    target_id: str = Field(description="Must be a LOC_ ID")
    
    # --- TOPOLOGY ---
    connection_type: str = Field(
        default="passage",
        description=(
            "Free-text classifier for the path (e.g. 'doorway', "
            "'corridor', 'stairs', 'one-way drop', 'window'). "
            "Used by the renderer/auditor for flavour and to flag "
            "implausible traversals; not consulted by physics."
        ),
    )
    bidirectional: bool = Field(
        default=True,
        description=(
            "When True the path is traversable A↔B and the "
            "instantiator emits an opposing connected_to edge. When "
            "False (e.g. a one-way drop, a magically sealed exit) "
            "only the declared source→target direction is wired into "
            "the sandbox so reachability/eavesdropping/spatial "
            "cascades respect the asymmetry."
        ),
    )

    # --- PHYSICAL CONSTRAINTS ---
    is_locked: bool = Field(default=False)
    barrier_item_id: Optional[str] = Field(
        default=None, 
        description="The ID of a NarrativeObject that dictates the 'locked' state (e.g., OBJ_IRON_DOOR)."
    )
    
    # --- TEMPORAL TRACKING ---
    established_at_fabula: int = Field(
        default=0, 
        description="Usually 0, unless the path was actively built during the story timeline."
    )
    destroyed_at_fabula: Optional[int] = Field(
        default=None, 
        description="T when the physical path was destroyed (e.g., a cave-in). Null if currently traversable."
    )
    

# --- 4. TEMPORAL RECONSTRUCTION ---

def reconstruct_entity_at(entity: "Entity", fabula_time: int) -> dict:
    """Reconstruct an entity's mutable state at a given fabula_time.

    Starts from the Entity's initial (pre-story) fields and replays
    EntityStateSnapshots up to *fabula_time* inclusive.

    Returns a dict with keys: traits, beliefs, status, location_id.
    Trait values are dicts ``{"value": float, "inertia": float}``.
    """
    # Seed from initial state
    traits: Dict[str, dict] = {
        k: {"value": v.value, "inertia": v.inertia}
        for k, v in entity.traits.items()
    }
    beliefs: list = [b.model_dump() for b in entity.beliefs]
    status: str = entity.status
    location_id: str = entity.location_id

    for snap in sorted(entity.state_timeline, key=lambda s: s.fabula_time):
        if snap.fabula_time > fabula_time:
            break
        # Merge trait updates
        for k, tv in snap.traits.items():
            traits[k] = {"value": tv.value, "inertia": tv.inertia}
        # Remove invalidated beliefs.
        # An invalidated entry may be either a bare ``target_id``
        # (legacy / coarse: drops every belief about that target)
        # or a ``"target_id::PROP_..."`` composite (fine-grained:
        # drops only the belief whose ``proposition_id`` matches).
        # The composite form keeps multi-belief same-target cases
        # (e.g. one belief about an entity per distinct
        # proposition) from being over-invalidated when only one of
        # them is contradicted.
        if snap.beliefs_invalidated:
            coarse: set = set()
            fine: set = set()
            for entry in snap.beliefs_invalidated:
                if "::" in entry:
                    fine.add(entry)
                else:
                    coarse.add(entry)
            kept = []
            for b in beliefs:
                tid = b.get("target_id")
                pid = b.get("proposition_id")
                if tid in coarse:
                    continue
                if pid and f"{tid}::{pid}" in fine:
                    continue
                kept.append(b)
            beliefs = kept
        # Add new beliefs
        for b in snap.beliefs_added:
            beliefs.append(b.model_dump())
        # Apply Affect-side confidence drift on existing beliefs.
        # Match by (target_id, proposition_id when set); silently skip
        # entries that don't match a current belief — Affect must not
        # forge new beliefs (that's Consequences' job).
        for shift in getattr(snap, "belief_confidence_updates", []) or []:
            for b in beliefs:
                if b.get("target_id") != shift.target_id:
                    continue
                if shift.proposition_id and b.get("proposition_id") not in (None, shift.proposition_id):
                    continue
                b["confidence"] = float(shift.new_confidence)
                if shift.new_inertia is not None:
                    b["inertia"] = float(shift.new_inertia)
                break
        if snap.status is not None:
            status = snap.status
        if snap.location_id is not None:
            location_id = snap.location_id

    # Filter beliefs by temporal anchor
    beliefs = [b for b in beliefs if b.get("established_at_fabula", 0) <= fabula_time]

    return {
        "traits": traits,
        "beliefs": beliefs,
        "status": status,
        "location_id": location_id,
    }


def reconstruct_world_trait_at(trait: "GlobalTrait", fabula_time: int) -> dict:
    """Reconstruct a world trait's state at a given fabula_time.

    Starts from the GlobalTrait's initial magnitude and replays
    WorldTraitSnapshots up to *fabula_time* inclusive.

    Returns a dict with keys: magnitude, description.
    magnitude is a dict ``{"value": float, "inertia": float}``.
    """
    magnitude = {"value": trait.magnitude.value, "inertia": trait.magnitude.inertia}
    description = trait.description

    for snap in sorted(trait.state_timeline, key=lambda s: s.fabula_time):
        if snap.fabula_time > fabula_time:
            break
        if snap.magnitude is not None:
            magnitude = {"value": snap.magnitude.value, "inertia": snap.magnitude.inertia}
        if snap.description is not None:
            description = snap.description

    return {
        "magnitude": magnitude,
        "description": description,
    }


def reconstruct_object_at(obj: "NarrativeObject", fabula_time: int) -> dict:
    """Reconstruct a :class:`NarrativeObject`'s mutable state at a given fabula_time.

    Starts from the object's initial fields and replays
    :class:`ObjectStateSnapshot` entries up to *fabula_time* inclusive.
    Mirrors :func:`reconstruct_entity_at` so AMWN sandboxes, ego
    graphs, the brief assembler, the auditor and any other downstream
    reader can see the object's correct position / ownership /
    properties at any point in the narrative rather than only its
    initial values.

    Returns a dict with keys ``location_id``, ``owner_id``,
    ``properties``. Snapshots are merged in fabula order; explicit
    null-clear flags (``set_location_null`` / ``set_owner_null``)
    distinguish "no change" from "cleared because picked up / dropped".
    """
    location_id: Optional[str] = obj.location_id
    owner_id: Optional[str] = obj.owner_id
    properties: Dict[str, str] = dict(obj.properties)

    for snap in sorted(obj.state_timeline, key=lambda s: s.fabula_time):
        if snap.fabula_time > fabula_time:
            break
        # Location: explicit clear wins, then explicit set, else no change.
        if snap.set_location_null:
            location_id = None
        elif snap.location_id is not None:
            location_id = snap.location_id
        # Owner: same precedence.
        if snap.set_owner_null:
            owner_id = None
        elif snap.owner_id is not None:
            owner_id = snap.owner_id
        # Property mutations: unset first, then set so an
        # author can rename a key in one snapshot.
        for k in snap.properties_unset:
            properties.pop(k, None)
        for k, v in snap.properties_set.items():
            properties[k] = v

    return {
        "location_id": location_id,
        "owner_id": owner_id,
        "properties": properties,
    }


def event_location_at(
    evt: "EventNode",
    ws: Any,
    *,
    fallback: Literal["actor", "none"] = "actor",
) -> Optional[str]:
    """Return the LOC_ id where *evt* takes place at ``evt.fabula_time``.

    Resolution order:
      1. ``evt.at_location_id`` when explicitly set;
      2. when ``fallback='actor'``: the reconstructed location of the
         primary actor (``speaker_id`` for utterances, else first
         ``actor_ids`` entry) at ``evt.fabula_time``;
      3. otherwise ``None``.

    ``ws`` is duck-typed: anything exposing an iterable ``entities``
    attribute of objects with ``id`` + ``location_id`` (and a
    ``state_timeline`` for replay) is acceptable. This keeps the
    helper usable from auditor / map / directive code without
    importing :class:`WorldStateV1`.
    """
    if evt.at_location_id:
        return evt.at_location_id
    if fallback != "actor":
        return None
    primary: Optional[str] = None
    if evt.speaker_id:
        primary = evt.speaker_id
    elif evt.actor_ids:
        primary = evt.actor_ids[0]
    if not primary:
        return None
    entities = getattr(ws, "entities", None)
    if entities is None:
        return None
    # ``WorldStateV1.entities`` is a dict ``{id: Entity}``; some lighter
    # ws-shaped objects pass a list. Normalise to an iterable of Entity.
    if isinstance(entities, dict):
        ent = entities.get(primary)
        ent_iter = [ent] if ent is not None else []
    else:
        ent_iter = list(entities)
    for ent in ent_iter:
        if getattr(ent, "id", None) != primary:
            continue
        try:
            return reconstruct_entity_at(ent, evt.fabula_time).get("location_id")
        except Exception:
            return getattr(ent, "location_id", None)
    return None


def reconstruct_proposition_at(prop: "Proposition", fabula_time: int) -> dict:
    """Reconstruct a proposition's mutable framing state at a given fabula_time.

    Starts from the Proposition's initial fields and replays
    ``PropositionSnapshot`` entries up to *fabula_time* inclusive.
    Mirrors :func:`reconstruct_entity_at` and
    :func:`reconstruct_world_trait_at` so engines can read
    fabula-time-aware ``stakes`` / ``audience_default_prior`` /
    ``description`` rather than only the static initial values.

    The returned ``truth_at`` value is the latest committed truth from
    ``Proposition.truth_at_fabula`` at or before *fabula_time*; ``None``
    if the proposition has not yet committed at that time.
    """
    stakes = prop.stakes
    audience_default_prior = prop.audience_default_prior
    description = prop.description

    for snap in sorted(prop.state_timeline, key=lambda s: s.fabula_time):
        if snap.fabula_time > fabula_time:
            break
        if snap.stakes is not None:
            stakes = snap.stakes
        if snap.audience_default_prior is not None:
            audience_default_prior = snap.audience_default_prior
        if snap.description is not None:
            description = snap.description

    truth_at: Optional[bool] = None
    for t in sorted(prop.truth_at_fabula.keys()):
        if t > fabula_time:
            break
        truth_at = prop.truth_at_fabula[t]

    return {
        "stakes": stakes,
        "audience_default_prior": audience_default_prior,
        "description": description,
        "truth_at": truth_at,
    }


def reconstruct_concern_at(concern: "Concern", fabula_time: int) -> dict:
    """Reconstruct a concern's mutable state at a given fabula_time.

    Starts from the Concern's initial fields and replays
    ``ConcernSnapshot`` entries up to *fabula_time* inclusive.
    Mirrors :func:`reconstruct_entity_at` so engines can read
    fabula-time-aware ``salience`` / ``polarity`` /
    ``activation_fabula_window`` / ``counter_concern_ids`` / ``kind``
    rather than only the static initial values.

    Returns an ``active`` flag honoring the (possibly updated)
    ``activation_fabula_window`` so callers don't need to repeat the
    window check.
    """
    salience = concern.salience
    polarity = concern.polarity
    activation_fabula_window = concern.activation_fabula_window
    counter_concern_ids = list(concern.counter_concern_ids)
    kind = concern.kind

    for snap in sorted(concern.state_timeline, key=lambda s: s.fabula_time):
        if snap.fabula_time > fabula_time:
            break
        if snap.salience is not None:
            salience = snap.salience
        if snap.polarity is not None:
            polarity = snap.polarity
        if snap.activation_fabula_window is not None:
            activation_fabula_window = snap.activation_fabula_window
        if snap.counter_concern_ids is not None:
            counter_concern_ids = list(snap.counter_concern_ids)
        if snap.kind is not None:
            kind = snap.kind

    active = True
    if activation_fabula_window and len(activation_fabula_window) == 2:
        lo, hi = activation_fabula_window
        active = lo <= fabula_time <= hi

    return {
        "salience": salience,
        "polarity": polarity,
        "activation_fabula_window": activation_fabula_window,
        "counter_concern_ids": counter_concern_ids,
        "kind": kind,
        "active": active,
    }


# --- 5. THE MASTER STATE (The Database Payload for Narrative structure) ---
class NarrativeStyle(BaseModel):
    """Captured profile of the source text's narrative *register*.

    The generation, refinement, and audit stages consult this so the
    rendered prose matches the *form* of the source — a plot-summary
    seed should produce summary-length condensed output, not a 2,000-
    word short story; a novel-excerpt seed should produce richly drawn
    prose, not a four-sentence beat sheet.

    Populated by ``shadow_loom.narrative_style.infer_narrative_style``
    during ingestion. Every field is optional so world-states loaded
    without a raw source still validate; downstream prompts fall back
    to their previous defaults when absent.
    """
    format: Literal[
        # Narrative fiction forms.
        "plot_summary", "synopsis", "outline", "scene",
        "short_story", "novel_excerpt", "screenplay", "verse",
        # Non-narrative / discursive forms — Shadow Loom is also used
        # for current-affairs reasoning, history, philosophy, etc.
        "news_article", "historical_account", "thought_experiment",
        "essay", "case_study", "transcript",
        "unknown",
    ] = Field(
        default="unknown",
        description=(
            "High-level form of the source text. Covers narrative "
            "fiction (plot_summary, scene, short_story, novel_excerpt, "
            "screenplay, verse) and non-narrative / discursive content "
            "(news_article, historical_account, thought_experiment, "
            "essay, case_study, transcript). Drives the target render "
            "length and prose density."
        ),
    )
    target_word_min: int = Field(
        default=500,
        ge=20,
        description="Lower bound (inclusive) of the per-render word budget.",
    )
    target_word_max: int = Field(
        default=2000,
        ge=20,
        description="Upper bound (inclusive) of the per-render word budget.",
    )
    prose_density: Literal["sparse", "moderate", "rich"] = Field(
        default="moderate",
        description=(
            "How much sensory / interior detail to render per beat. "
            "'sparse' = telegraphic summary diction (one sentence per "
            "story beat); 'moderate' = flowing scene prose; 'rich' = "
            "novelistic interiority and sensory texture."
        ),
    )
    voice: str = Field(
        default="",
        description=(
            "Free-form description of the narrative voice (POV, tense, "
            "tonal register, diction). Used by the renderer to mirror "
            "the source's voice."
        ),
    )
    style_exemplar: Optional[str] = Field(
        default=None,
        description=(
            "Up to ~600 characters lifted verbatim from the source so "
            "the renderer and auditor can pattern-match cadence and "
            "diction. May be None when no source text was supplied."
        ),
    )
    source_word_count: Optional[int] = Field(
        default=None,
        description="Total word count of the source text, when known.",
    )

    @model_validator(mode="after")
    def _check_word_range(self) -> "NarrativeStyle":
        if self.target_word_max < self.target_word_min:
            raise ValueError(
                "NarrativeStyle.target_word_max must be >= target_word_min"
            )
        return self


class WorldStateV1(BaseModel):
    locations: Dict[str, Location]
    objects: Dict[str, NarrativeObject]
    entities: Dict[str, Entity]
    events: List[EventNode]
    world_traits: Dict[str, "GlobalTrait"] = Field(
        default_factory=dict,
        description="World-level facts, laws, and conditions. Keyed by WORLD_ ID.",
    )
    narrative_style: Optional[NarrativeStyle] = Field(
        default=None,
        description=(
            "Profile of the source text's narrative register (format, "
            "target length, density, voice). Populated by ingestion "
            "when a raw source text is supplied; consulted by the "
            "directive assembler, renderer, and auditor so the "
            "generated prose preserves the source's form."
        ),
    )
    causal_topology: List[CausalEdge]
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)
    channels: Dict[str, Channel] = Field(
        default_factory=dict,
        description=(
            "Standing communication channels keyed by CHN_ id. Replaces the "
            "legacy ``information_topology`` collection; discrete messages "
            "that previously appeared as InformationEdges now appear as "
            "EventNodes with event_type='utterance'."
        ),
    )
    social_topology: List[RelationshipEdge] = Field(default_factory=list)
    propositions: List[Proposition] = Field(
        default_factory=list,
        description=(
            "Shared proposition registry for the affect-unification layer. "
            "Audience and characters hold per-agent ``Belief`` confidences "
            "about these propositions; suspense/surprise/irony/mystery are "
            "all computed as queries over this registry. Empty for legacy "
            "worlds; populated by ``synthesise_propositions`` after ingestion."
        ),
    )
    world_facts: List["WorldFact"] = Field(
        default_factory=list,
        description=(
            "Optional external research grounding (off by default). "
            "Populated only when ``ExtractionConfig.enable_research_agent`` "
            "is true. Each ``WorldFact`` carries its own ``source_url`` "
            "provenance and is intentionally segregated from entity "
            "traits, beliefs, events and global traits — the auditor "
            "never promotes a fact into those namespaces. See "
            "``shadow_loom.research`` for the provider layer and "
            "``docs/research-extraction-plan.md`` for the rationale."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_information_topology(cls, data: Any) -> Any:
        """Fail loudly when callers supply the removed ``information_topology`` field.

        The information-topology refactor split the legacy ``InformationEdge``
        into a ``Channel`` node (capability) and an ``EventNode(event_type=
        'utterance')`` (discrete message). There is intentionally no auto-
        migration: silent translation would mis-classify single-shot utterances
        as standing channels and vice-versa. Callers must explicitly rewrite.
        """
        if isinstance(data, dict) and "information_topology" in data:
            raise ValueError(
                "WorldStateV1.information_topology has been removed. Replace "
                "each legacy InformationEdge with either (a) a Channel node "
                "in WorldStateV1.channels (for standing capabilities) or "
                "(b) an EventNode(event_type='utterance', content=..., "
                "speaker_id=..., addressee_ids=[...]) appended to "
                "WorldStateV1.events (for discrete messages). See "
                "shadow_loom/models.py::Channel for the new schema."
            )
        return data

    @model_validator(mode="after")
    def _mirror_missing_relationship_directions(self) -> "WorldStateV1":
        """Synthesise the reverse-direction edge for any one-sided dyad.

        ``RelationshipEdge`` is *directed* — the metrics on edge
        ``A→B`` describe how *A* feels/stands toward *B*, and may
        differ from the reverse ``B→A`` edge (Heathcliff loves Cathy
        0.9; Cathy fears Heathcliff 0.7). The Social-extraction prompt
        asks the LLM for both directions, but in practice extractors
        and hand-authored fixtures frequently emit only the most
        salient half — leaving the reverse direction missing entirely
        and starving every downstream reader (physics social
        mutations, AMWN snapshots, the affective dashboard, the
        directive assembler, the renderer's relationship context, the
        auditor's asymmetry score) of the perspective entity's view.

        Rather than letting that gap propagate as silent zero-affinity
        / zero-fear, this validator synthesises the missing reverse
        edge as a *mirror*:

          * **affinity** and **fear** copy the forward value verbatim
            (default-symmetric prior — sensible when no contradicting
            signal exists; a real extraction posting a different
            reverse value will replace the mirror via the standard
            dedup paths).
          * **power_dynamic** is **negated** because a positive value
            on ``A→B`` means *A holds power over B*, so the dyad's
            other half necessarily reads as *B holds power under A* —
            same magnitude, flipped sign.
          * Every mirrored metric is marked ``observed=False``,
            ``evidence_strength="weak"``, has its ``inertia`` halved
            (so the abduction blender doesn't treat the prior as
            calcified), and resets ``last_updated_fabula`` to ``0``
            (the canonical "no recent observation" sentinel used by
            ``default_relationship_metrics_dict``). Together these
            ensure the mirror reads as a fallback prior rather than
            ground truth across every downstream consumer.

        Idempotent: a world that already has both directions of every
        dyad is unchanged. Runs at construction time so every
        consumer (ingestion, example_worlds, tests, snapshots, the UI)
        sees a consistent post-mirror world without having to call a
        helper.
        """
        edges = list(self.social_topology)
        if not edges:
            return self
        # Only mirror when *both* endpoints reference entities the world
        # actually knows about. Mirroring an edge whose source or target
        # is a hallucinated/unknown id would synthesise a second broken
        # edge that ``_auto_repair`` then has to clean up — silently
        # doubling the repair count and obscuring the original
        # extraction error. Skipping the mirror leaves the lone broken
        # edge intact for the existing ID-validation pass to flag.
        known_ids = set(self.entities.keys())
        indexed: dict[tuple[str, str], RelationshipEdge] = {
            (e.source_entity_id, e.target_entity_id): e for e in edges
        }
        mirrored: list[RelationshipEdge] = []
        for (src, tgt), edge in list(indexed.items()):
            if (tgt, src) in indexed:
                continue
            if src not in known_ids or tgt not in known_ids:
                continue
            new_metrics: dict[str, dict] = {}
            for name, m in edge.metrics.items():
                value = float(m.value)
                if name == "power_dynamic":
                    value = -value
                # Mirrored metrics are weak fallback priors, not
                # observations: halve the forward inertia (so the
                # abduction blender does not treat them as calcified)
                # and reset ``last_updated_fabula`` to 0 — the canonical
                # "no recent observation" sentinel used by
                # ``default_relationship_metrics_dict``.
                new_metrics[name] = {
                    "value": value,
                    "inertia": max(0.0, float(m.inertia) * 0.5),
                    "evidence_strength": "weak",
                    "last_updated_fabula": 0,
                    "observed": False,
                }
            if not new_metrics:
                continue
            try:
                mirror = RelationshipEdge(
                    world_id=edge.world_id,
                    source_entity_id=tgt,
                    target_entity_id=src,
                    metrics=new_metrics,  # type: ignore[arg-type]
                )
            except Exception:
                # Validation of the mirror should never fail in
                # practice — if it does, log and leave the gap rather
                # than blocking world construction.
                _logger.warning(
                    "Failed to mirror RelationshipEdge %s→%s; leaving "
                    "reverse direction missing.",
                    src, tgt, exc_info=True,
                )
                continue
            mirrored.append(mirror)
            indexed[(tgt, src)] = mirror
        if mirrored:
            self.social_topology = edges + mirrored
        return self


# ---------------------------------------------------------------------
# Forward-reference resolution
# ---------------------------------------------------------------------
# ``WorldFact`` lives in ``shadow_loom.research`` (a pure-pydantic /
# stdlib module that does not import from shadow_loom.models) so we can
# safely resolve the forward reference on ``WorldStateV1.world_facts``
# at import time without creating a cycle.
from shadow_loom.research import WorldFact  # noqa: E402

WorldStateV1.model_rebuild()