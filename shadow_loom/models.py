# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Any, List, Dict, Optional, Literal

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

    _coerce_es = field_validator("evidence_strength", mode="before")(
        lambda v: _coerce_evidence_strength(v)
    )


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
    status: Optional[Literal["healthy", "injured", "ill", "dead", "unconscious"]] = Field(
        default=None, description="New status if changed, else null.",
    )
    location_id: Optional[str] = Field(
        default=None, description="New location if entity moved, else null.",
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

class Entity(AMWNNode):
    id: str = Field(description="Unique ID, e.g., ENT_MACBETH")
    name: str
    location_id: str = Field(description="Initial location (pre-story or earliest known).")
    status: Literal["healthy", "injured", "ill", "dead", "unconscious"]
    traits: Dict[str, TraitVector] = Field(description="Initial multidimensional psychology (pre-story baseline).")
    beliefs: List[Belief] = Field(default_factory=list, description="Initial epistemic state.")
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
            "happening; revelation = the audience or a character learns a "
            "hitherto-hidden fact (syuzhet-side); utterance = a discrete "
            "speech-act / message transmitted between characters (fabula-side). "
            "Utterance and revelation are distinct: an utterance can occur "
            "long before its content is revealed to the audience."
        ),
    )
    actor_ids: List[str] = Field(default_factory=list, description="Who did it? Empty if natural event. Supports joint actions (e.g., ['ENT_MACBETH', 'ENT_LADY_MACBETH']).")
    target_ids: List[str] = Field(default_factory=list, description="Who/what was acted upon? e.g., ['ENT_DUNCAN'] in a murder event. Supports diffuse effects.")
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
        if tgt_is_event and ct not in ("chain_reaction", "affordance_gate"):
            raise ValueError(
                f"target_id '{self.target_id}' is an event — causality_type must be "
                f"'chain_reaction' or 'affordance_gate', got '{ct}'."
            )
        if not tgt_is_event and ct not in ("mutation", "mutation_social", "ambient_propagation"):
            raise ValueError(
                f"target_id '{self.target_id}' is a state node — causality_type must be "
                f"'mutation', 'mutation_social', or 'ambient_propagation', got '{ct}'."
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
        # Remove invalidated beliefs
        if snap.beliefs_invalidated:
            inv_set = set(snap.beliefs_invalidated)
            beliefs = [b for b in beliefs if b.get("target_id") not in inv_set]
        # Add new beliefs
        for b in snap.beliefs_added:
            beliefs.append(b.model_dump())
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


# ---------------------------------------------------------------------
# Forward-reference resolution
# ---------------------------------------------------------------------
# ``WorldFact`` lives in ``shadow_loom.research`` (a pure-pydantic /
# stdlib module that does not import from shadow_loom.models) so we can
# safely resolve the forward reference on ``WorldStateV1.world_facts``
# at import time without creating a cycle.
from shadow_loom.research import WorldFact  # noqa: E402

WorldStateV1.model_rebuild()