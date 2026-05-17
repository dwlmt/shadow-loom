# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Literal, Tuple, Union, Dict, List
from typing_extensions import Annotated

from shadow_loom.introduced_elements import IntroducedElements


# ---------------------------------------------------------------------
# DoTarget — typed, discriminated payloads for Pearl Rung-2 / Rung-3
# interventions. Replaces the legacy free-form ``Dict[str, Any]`` on
# ``InterventionQuery.interventions`` and
# ``CounterfactualQuery.historical_interventions``. The legacy dicts
# remain for backwards compatibility; the migration adapter
# ``coerce_legacy_interventions`` (in ``narrative_physics``) lifts them
# into typed ``DoTarget`` lists at dispatch time.
#
# Every variant carries a ``target_kind`` literal so Pydantic can
# discriminate the union without falling back to try-each-type.
# ---------------------------------------------------------------------
class DoEvent(BaseModel):
    """Clamp the occurrence of an ``EventNode``.

    Existing event-level surgery (the only intervention shape supported
    by the engine prior to this typed surface). ``occurred=False`` is
    the standard "what if X had not happened" form; ``occurred=True``
    forces an event that did not occur in the factual world.
    """
    target_kind: Literal["event"] = "event"
    event_id: str = Field(description="EVT_ id whose occurrence is clamped.")
    occurred: bool = Field(default=False, description="Clamped occurrence value.")
    new_at_location_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional LOC_ id to relocate the event to. When set (and "
            "``occurred`` remains True), the do-operator rewrites the "
            "event's ``at_location_id`` and cascades an "
            "``EntityStateSnapshot(location_id=new_at_location_id)`` for "
            "every primary actor at the event's ``fabula_time`` so the "
            "co-presence invariant continues to hold post-surgery. Has no "
            "effect when ``occurred=False``."
        ),
    )


class DoProposition(BaseModel):
    """Clamp a ``Proposition`` truth value at a fabula time.

    Cascades to every ``Belief`` whose ``proposition_id`` matches when
    ``propagate_to_beliefs`` is True (gated by the belief's
    ``evidence_strength``). Used for "suppose Banquo's line really
    inherits", "if it had been the case that O'Brien is genuinely
    Brotherhood", etc.
    """
    target_kind: Literal["proposition"] = "proposition"
    proposition_id: str = Field(description="PROP_ id whose truth is clamped.")
    truth: bool = Field(description="The clamped truth value.")
    fabula_time: Optional[int] = Field(
        default=None,
        description="Fabula time of the clamp. Defaults to the query's anchor when unset.",
    )
    propagate_to_beliefs: bool = Field(
        default=True,
        description=(
            "If True, cascade the clamp into every Belief whose proposition_id matches "
            "(adjusting confidence per evidence_strength). If False, only the "
            "audience-side ``truth_at_fabula`` is altered."
        ),
    )


class DoBelief(BaseModel):
    """Clamp a single character's belief — an epistemic intervention.

    Used for "if Macduff had believed Macbeth's grief was sincere", "if
    Otello believed Desdemona faithful". Distinct from ``DoProposition``
    because the underlying fact is unchanged — only the holder's
    epistemic state is forced.
    """
    target_kind: Literal["belief"] = "belief"
    holder_id: str = Field(description="ENT_ id of the believer.")
    target_id: str = Field(description="ENT_/EVT_/OBJ_/LOC_/WORLD_ id the belief is about.")
    perceived_state: Optional[str] = Field(
        default=None,
        description="Belief content (required when creating a belief that does not exist yet).",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Clamped confidence in the belief.",
    )
    proposition_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional PROP_ id this belief joins. Auto-resolved when a unique "
            "proposition references the (holder, target) pair."
        ),
    )


class DoConcern(BaseModel):
    """Clamp a single character's concern — a utility-layer intervention.

    Used for "if Lady Macbeth had no ambition", "without Heathcliff's
    desire for vengeance", "suppose Victor never feared the Creature".
    Any unset field is left at its factual value.
    """
    target_kind: Literal["concern"] = "concern"
    holder_id: str = Field(description="ENT_ id of the concern holder.")
    concern_id: str = Field(description="CCN_ id to clamp.")
    polarity: Optional[Literal["desire", "fear"]] = Field(
        default=None, description="Override polarity; None leaves it unchanged.",
    )
    salience: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Override salience; None leaves it unchanged.",
    )
    active: Optional[bool] = Field(
        default=None,
        description=(
            "Toggle the concern's activation. False collapses the activation_window "
            "to a single point past the query horizon (effectively disabling); True "
            "clears any window (always-active)."
        ),
    )


class DoTrait(BaseModel):
    """Clamp a single character trait. Equivalent to existing trait
    surgery in ``CausalPhysicsEngine.apply_do_operator`` but exposed as
    a typed payload on the query surface."""
    target_kind: Literal["trait"] = "trait"
    holder_id: str = Field(description="ENT_ id of the trait-bearer.")
    trait_name: str = Field(description="Trait name, e.g. 'ambition' or 'fear'.")
    value: float = Field(description="Clamped trait value.")
    inertia: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Override inertia; None leaves it unchanged.",
    )


class DoWorldTrait(BaseModel):
    """Clamp a WORLD_ ``GlobalTrait``'s magnitude — a global ambient
    intervention.

    Used for "if the war had ended at chapter 12", "suppose the
    surveillance state never tightened", "what if the prophecy had
    resolved early". Lands as a :class:`WorldTraitSnapshot` on the
    trait's ``state_timeline`` honouring the per-chunk merge fold's
    inertia attenuation, so high-inertia traits resist the clamp
    rather than snapping immediately.

    Distinct from :class:`DoTrait` (per-character) because WORLD_
    traits are shared common-cause anchors: every entity in the scene
    sees the shifted ambient force on its next propagation step.
    """
    target_kind: Literal["world_trait"] = "world_trait"
    world_trait_id: str = Field(
        description="WORLD_ id of the global trait whose magnitude is clamped.",
    )
    value: float = Field(
        ge=0.0, le=1.0,
        description="Clamped magnitude.value (0.0 absent, 1.0 maximally present).",
    )
    inertia: Optional[float] = Field(
        default=None, ge=0.0, le=0.99,
        description=(
            "Override magnitude.inertia; None leaves it unchanged. "
            "Capped at 0.99 (1.0 would make the trait literally "
            "unmovable)."
        ),
    )
    affected_domains_add: List[str] = Field(
        default_factory=list,
        description=(
            "Canonical domains to add to the trait's affected_domains "
            "set ('physical', 'psychological', 'epistemic', 'social', "
            "'emotional', 'informational', 'betrayal'). Set-additive, "
            "no-op if already present."
        ),
    )
    affected_domains_remove: List[str] = Field(
        default_factory=list,
        description=(
            "Canonical domains to drop from the trait's "
            "affected_domains set. Set-subtractive, no-op if absent."
        ),
    )
    fabula_time: Optional[int] = Field(
        default=None,
        description=(
            "Fabula time of the clamp. Defaults to the query's anchor "
            "(or the simulation horizon when neither is supplied)."
        ),
    )
    triggered_by: Optional[str] = Field(
        default=None,
        description=(
            "Optional EVT_ id whose occurrence motivates this clamp. "
            "When set, the snapshot's ``triggered_by`` field carries "
            "this id so the audit panel can attribute the shift."
        ),
    )


class DoChannel(BaseModel):
    """Mutate a standing :class:`Channel` \u2014 disable, re-enable, or
    re-tune intelligibility.

    Used for "if the wiretap had been discovered earlier", "suppose
    the courier had been turned", "what if the mind-link had been
    severed at chapter 8". The clamp lands on the sandbox's Channel
    node (so any utterance/belief that resolves through it on the
    same propagate step sees the new state) and is mirrored onto
    ``WorldStateV1.channels`` so downstream consumers reading the
    world directly observe the surgery.

    Channel *creation* is handled by ``query.introduce.channels`` /
    ``IntroducedChannelSpec`` so the brand-new birth path stays
    distinct from the in-place clamp path.
    """
    target_kind: Literal["channel"] = "channel"
    channel_id: str = Field(description="CHN_ id whose state is clamped.")
    active: Optional[bool] = Field(
        default=None,
        description=(
            "True = re-enable a previously-terminated channel "
            "(clears ``terminated_at_fabula``). False = sever the "
            "channel at this query's anchor."
        ),
    )
    intelligibility: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Per-participant decode-probability override \u2208 [0, 1]. "
            "Keys overwrite the channel's existing intelligibility map."
        ),
    )
    fabula_time: Optional[int] = Field(
        default=None,
        description="Fabula time of the clamp. Defaults to the query's anchor when unset.",
    )


class DoRelationship(BaseModel):
    """Clamp a single per-axis :class:`RelationshipMetric` between two
    entities \u2014 a social-fabric intervention.

    Targets the canonical ``(source, target)`` directed pair. The
    engine writes the clamp into both the sandbox edge attrs (so
    propagation sees it on the same step) and the world-state's
    ``social_topology`` (so directive assembly and re-extraction
    observe the pinned axis). Spawns a fresh edge carrying only the
    clamped metric when the named pair has no existing relationship.
    """
    target_kind: Literal["relationship"] = "relationship"
    source_entity_id: str = Field(description="ENT_ id of the perspective entity.")
    target_entity_id: str = Field(description="ENT_ id of the relationship counterpart.")
    metric: Literal["affinity", "fear", "power_dynamic"] = Field(
        description="Which per-axis metric to clamp.",
    )
    value: float = Field(
        ge=-1.0, le=1.0,
        description="Clamped metric value (typical range [-1.0, 1.0]).",
    )
    inertia: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Override per-axis inertia; None leaves it unchanged.",
    )
    fabula_time: Optional[int] = Field(
        default=None,
        description="Fabula time of the clamp. Defaults to the query's anchor when unset.",
    )


class DoCausalEdge(BaseModel):
    """Add or sever a :class:`CausalEdge` \u2014 mechanism-layer surgery.

    Clamps the *existence* of a causal arrow between two pre-existing
    nodes. Modifying force/mechanism on an existing edge is
    intentionally not supported \u2014 use re-extraction or sever+add for
    that. Distinct from :class:`DoEvent` (which clamps a node's
    occurrence) because removing a single cause leaves the effect
    free to be triggered by other parents, while preventing the
    effect node itself blocks every parent at once.
    """
    target_kind: Literal["causal_edge"] = "causal_edge"
    source_id: str = Field(description="EVT_/ENT_/OBJ_/LOC_/WORLD_ id of the cause.")
    target_id: str = Field(description="EVT_/ENT_/OBJ_/LOC_/WORLD_ id of the effect.")
    action: Literal["add", "sever"] = Field(
        description="``add`` instantiates a new edge; ``sever`` removes the matching edge(s).",
    )
    causality_type: Optional[Literal[
        "chain_reaction", "mutation", "mutation_social",
        "affordance_gate", "ambient_propagation",
    ]] = Field(
        default=None,
        description="Required for ``action='add'``. Mirrors :class:`CausalEdge.causality_type`.",
    )
    mechanism: Optional[str] = Field(
        default=None,
        description=(
            "Required for ``action='add'``. Canonical short key "
            "('physical', 'psychological', 'epistemic', 'social', "
            "'emotional', 'informational', 'betrayal') or an off-list label."
        ),
    )
    causal_force: float = Field(
        default=5.0, ge=0.0, le=10.0,
        description="Impact magnitude for ``action='add'``.",
    )
    trait_target: Optional[str] = Field(
        default=None,
        description="For mutation / mutation_social adds: the specific trait or metric affected.",
    )
    trait_delta: Optional[float] = Field(
        default=None,
        description="For mutation / mutation_social adds: signed magnitude of the change.",
    )
    rel_counterpart_id: Optional[str] = Field(
        default=None,
        description="For mutation_social adds: ENT_ id of the other entity in the dyad.",
    )
    fabula_time: Optional[int] = Field(
        default=None,
        description="Fabula time of the edge. Defaults to the query's anchor when unset.",
    )


class DoSpatialEdge(BaseModel):
    """Add, sever, or lock-toggle a :class:`SpatialEdge` \u2014 architecture
    surgery.

    Used for "if the postern had been bricked up", "suppose the secret
    passage from the tomb opened earlier", "what if the courtyard
    gate had been unlocked when Banquo arrived".
    """
    target_kind: Literal["spatial_edge"] = "spatial_edge"
    source_id: str = Field(description="LOC_ id of the source location.")
    target_id: str = Field(description="LOC_ id of the target location.")
    action: Literal["add", "sever", "lock", "unlock"] = Field(
        description=(
            "``add`` creates a new connection; ``sever`` removes the "
            "matching edge(s); ``lock``/``unlock`` toggles "
            "``is_locked`` on the existing edge."
        ),
    )
    connection_type: Optional[str] = Field(
        default=None,
        description="For ``action='add'``: free-text classifier (e.g. 'doorway', 'corridor').",
    )
    bidirectional: bool = Field(
        default=True,
        description="For ``action='add'``: whether the edge is traversable both ways.",
    )
    barrier_item_id: Optional[str] = Field(
        default=None,
        description="For ``action='add'`` or ``action='lock'``: optional OBJ_ id whose state determines the lock.",
    )
    fabula_time: Optional[int] = Field(
        default=None,
        description="Fabula time of the clamp. Defaults to the query's anchor when unset.",
    )


class DoNarrativeObject(BaseModel):
    """Clamp a :class:`NarrativeObject`'s position / ownership / properties \u2014
    prop-layer surgery.

    Used for "if the dagger had stayed in Macbeth's hand", "suppose the
    locket had never left the drawer", "what if the cup were poisoned
    when Duncan arrived". Unset fields are left at their factual value;
    ``set_location_null`` / ``set_owner_null`` toggle explicit clears
    so a pickup ("now in inventory; no location") and a drop ("now on
    the floor; no owner") are distinguishable from a no-op.

    Lands on the sandbox's OBJ_ node (so the same propagate step sees
    the new state) and is mirrored back via
    :class:`ObjectMutation` so the pipeline adapter can fold a
    :class:`ObjectStateSnapshot` onto the canonical
    :attr:`NarrativeObject.state_timeline`.
    """
    target_kind: Literal["object"] = "object"
    object_id: str = Field(description="OBJ_ id whose state is clamped.")
    new_location_id: Optional[str] = Field(
        default=None,
        description=(
            "New LOC_ id (placed / dropped / relocated). Use null + "
            "``set_location_null=True`` for pickup (clears location)."
        ),
    )
    new_owner_id: Optional[str] = Field(
        default=None,
        description=(
            "New ENT_ id (picked up / gifted / stolen / inherited). "
            "Use null + ``set_owner_null=True`` for drop (clears owner)."
        ),
    )
    set_location_null: bool = Field(
        default=False,
        description="Explicitly clear ``NarrativeObject.location_id`` (pickup).",
    )
    set_owner_null: bool = Field(
        default=False,
        description="Explicitly clear ``NarrativeObject.owner_id`` (drop).",
    )
    properties_set: Dict[str, str] = Field(
        default_factory=dict,
        description="Property keys to overwrite on the object.",
    )
    properties_unset: List[str] = Field(
        default_factory=list,
        description="Property keys to remove from the property dict.",
    )
    fabula_time: Optional[int] = Field(
        default=None,
        description="Fabula time of the clamp. Defaults to the query's anchor when unset.",
    )
    triggered_by: Optional[str] = Field(
        default=None,
        description=(
            "Optional EVT_ id whose occurrence motivates this clamp. "
            "Carried onto the resulting ObjectStateSnapshot for "
            "audit attribution."
        ),
    )


# Discriminated union — Pydantic v2 dispatches on ``target_kind``.
DoTarget = Annotated[
    Union[
        DoEvent, DoProposition, DoBelief, DoConcern, DoTrait, DoWorldTrait,
        DoChannel, DoRelationship, DoCausalEdge, DoSpatialEdge,
        DoNarrativeObject,
    ],
    Field(discriminator="target_kind"),
]


# ---------------------------------------------------------------------
# Shared base — every query type carries the user's verbatim natural-
# language request so it can be threaded into directive assembly,
# generation prompts, audit feedback, and the persisted version row.
# ---------------------------------------------------------------------
class _QueryBase(BaseModel):
    # ``validate_assignment=True`` re-runs all field validators (including
    # the typed-do_targets backfill model_validator on
    # InterventionQuery / CounterfactualQuery) on every post-construction
    # attribute assignment. Without it, a caller mutating
    # ``q.interventions[...] = ...`` after construction would desync
    # the legacy dict from ``q.do_targets`` and silently serve an
    # outdated typed surgery set to downstream consumers.
    model_config = {"validate_assignment": True}

    original_query: Optional[str] = Field(
        default=None,
        description=(
            "The user's verbatim natural-language request. Surfaced to "
            "the directive assembler, generator, auditor, and saved on "
            "the version row so the UI can show what the user asked for."
        ),
    )
    # ---------------------------------------------------------------
    # Story-point anchors. Per-query overrides for the pipeline-level
    # ``PipelineConfig.temporal_anchor`` / ``syuzhet_anchor``. When set,
    # the pipeline reconstructs the world / reader state at this point
    # in the story before executing the query, so directives, physics
    # and ego-graph extraction operate on the slice the user asked for
    # rather than the latest state.
    # ---------------------------------------------------------------
    temporal_anchor: Optional[int] = Field(
        default=None,
        description=(
            "Optional fabula-time horizon. Overrides "
            "``PipelineConfig.temporal_anchor`` for this query. The world "
            "state is reconstructed as of this fabula_time before the "
            "query runs (entities, beliefs, relationships, events, "
            "channels are all time-sliced)."
        ),
    )
    syuzhet_anchor: Optional[int] = Field(
        default=None,
        description=(
            "Optional syuzhet-index horizon for reader-effect calculus "
            "(suspense / surprise / mystery / dramatic-irony). Overrides "
            "``PipelineConfig.syuzhet_anchor`` for this query."
        ),
    )
    anchor_after_event_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional event ID (``EVT_*``) to anchor the query *immediately "
            "after*. The pipeline resolves this to the event's "
            "``fabula_time`` (and ``syuzhet_index`` if neither anchor is "
            "explicitly set) before execution. Convenient for 'apply this "
            "directive after EVT_BANQUO_DEATH' style requests without the "
            "caller looking the time up first. Explicit ``temporal_anchor`` "
            "/ ``syuzhet_anchor`` values take precedence over this."
        ),
    )
    introduce: Optional[IntroducedElements] = Field(
        default=None,
        description=(
            "User-side birth list of new top-level world elements this "
            "query should pre-declare into the world before physics. "
            "Use this when a do-surgery, observation, or directive needs "
            "to refer to an entity / location / object / world-trait / "
            "proposition / concern that doesn't exist yet (e.g. "
            "do(`ENT_NEW_AGENT.spawn` = {...}) where the agent is "
            "introduced for the first time). The pipeline pre-spawns "
            "these into the sandbox before the engine runs so do-surgeries "
            "can target them safely; they are then folded into the "
            "renderer's ``GeneratedScene.introduced_elements`` for the "
            "auditor and re-extraction. The renderer remains free to "
            "additionally introduce other elements that emerge "
            "organically from the prose."
        ),
    )


# ==========================================
# 1. THE OBSERVATION (Rung 1: Natural Progression)
# ==========================================
class ObservationQuery(_QueryBase):
    """
    Advances the clock natively, but allows conditioning the probability 
    engine on MULTIPLE observed facts before simulating the next step.
    """
    query_type: Literal["observation"] = "observation"
    observations: Dict[str, str] = Field(
        default_factory=dict, 
        description="Multiple facts observed right now. e.g., {'OBJ_CUP': 'empty', 'ENT_GUARD': 'asleep'}"
    )
    focus_entity_ids: List[str] = Field(
        default_factory=list, 
        description="Multiple characters to lock the POV onto."
    )

# ==========================================
# 2. THE INTERVENTION (Rung 2: God Mode)
# ==========================================
class InterventionQuery(_QueryBase):
    """
    Forces MULTIPLE variables to specific states simultaneously in the present moment,
    cutting incoming edges, and calculates the future from here.
    """
    query_type: Literal["intervention"] = "intervention"
    do_targets: List[DoTarget] = Field(
        default_factory=list,
        description=(
            "Typed Pearl Rung-2 do-operator targets (events, propositions, "
            "beliefs, concerns, traits). When non-empty, supersedes the legacy "
            "``interventions`` dict; the migration adapter lifts the legacy "
            "shape into typed targets when this list is empty."
        ),
    )
    interventions: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Legacy free-form do-operator dict, kept for backwards compatibility. "
            "Use ``do_targets`` for new code; this field is auto-coerced when "
            "``do_targets`` is empty."
        ),
    )
    target_node_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Optional downstream nodes the user cares about. When set, the "
            "ctf-calculus pre-flight uses these as the Y-set for Rule 3 "
            "(Exclusion): an intervention is provably vacuous if it has no "
            "directed path to any of these nodes in the mutilated diagram."
        ),
    )
    force_implausible: bool = Field(
        default=False,
        description="If True, generate prose even when the engine cannot resolve any "
        "intervention targets against the current world state. The implausibility "
        "reason is still reported on the result so the caller can warn the user.",
    )

    @model_validator(mode="after")
    def _backfill_typed_do_targets(self) -> "InterventionQuery":
        """Auto-coerce legacy ``interventions`` → typed ``do_targets``.

        Ensures every consumer of an :class:`InterventionQuery` sees a
        populated typed list as soon as the model is constructed,
        regardless of which surface (LLM parser, MCP, fixture, REST
        client) authored it. Without this, code paths that pulled
        ``query.do_targets`` directly silently saw an empty list whenever
        the caller used the legacy dotted-key dict, dropping the
        renderer's "RUNG-2 SURGERY KIND" hints and auditor coverage.
        """
        if not self.do_targets and self.interventions:
            # Use object.__setattr__ to bypass validate_assignment;
            # otherwise this triggers the validator recursively.
            object.__setattr__(
                self, "do_targets", _coerce_legacy_dict(self.interventions),
            )
        _warn_on_unhandled_legacy_keys(
            self.interventions, self.do_targets,
            field_label="interventions", model=self,
        )
        return self

# ==========================================
# 3. THE COUNTERFACTUAL (Rung 3: Abduction)
# ==========================================
class CounterfactualQuery(_QueryBase):
    """
    Goes back in time, updates hidden variables based on current evidence, 
    applies multiple interventions, and runs prediction.
    """
    query_type: Literal["counterfactual"] = "counterfactual"
    historical_do_targets: List[DoTarget] = Field(
        default_factory=list,
        description=(
            "Typed Pearl Rung-3 historical do-operator targets. When non-empty, "
            "supersedes the legacy ``historical_interventions`` dict; the migration "
            "adapter lifts the legacy shape into typed targets when this list is "
            "empty."
        ),
    )
    historical_interventions: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Legacy free-form historical do-operator dict, kept for backwards "
            "compatibility. Use ``historical_do_targets`` for new code."
        ),
    )
    evidence_node_ids: List[str] = Field(
        description="The facts from the present we must condition on to calculate latent traits."
    )
    target_node_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Optional downstream nodes the user cares about (Y-set for Rule 3 "
            "Exclusion in the ctf-calculus pre-flight)."
        ),
    )
    force_implausible: bool = Field(
        default=False,
        description="If True, generate prose even when no historical anchor can be "
        "located for the requested interventions. The implausibility reason is still "
        "surfaced on the result.",
    )

    @model_validator(mode="after")
    def _backfill_typed_historical_do_targets(self) -> "CounterfactualQuery":
        """Auto-coerce legacy ``historical_interventions`` → typed
        ``historical_do_targets`` on construction. See
        :meth:`InterventionQuery._backfill_typed_do_targets` for rationale."""
        if not self.historical_do_targets and self.historical_interventions:
            object.__setattr__(
                self,
                "historical_do_targets",
                _coerce_legacy_dict(self.historical_interventions),
            )
        _warn_on_unhandled_legacy_keys(
            self.historical_interventions,
            self.historical_do_targets,
            field_label="historical_interventions",
            model=self,
        )
        return self

# ==========================================
# 4. THE NARRATIVE DIRECTIVE (Merged Suspense & Emotion)
# ==========================================
class DirectiveQuery(_QueryBase):
    """
    Tells the engine to mathematically optimize the next event to maximize 
    a specific psychological or epistemic effect.
    """
    query_type: Literal["directive"] = "directive"
    target_entity_ids: List[str] = Field(description="The Entities experiencing the emotion or the ignorance.")
    target_effect: Literal["suspense", "surprise", "mystery", "dramatic_irony", "narrative_tension", "grief", "rage", "joy", "regret", "love", "fear"] = Field(
        description="The narrative effect to maximize."
    )
    target_vector_id: Optional[str] = Field(
        default=None,
        description="If suspense: the Objective Node ID they are blind to. If emotion: the Trait/Edge ID to shatter."
    )
    intensity: float = Field(default=1.0, description="0.0 to 1.0 multiplier for the Prompt injection.")
    force_implausible: bool = Field(
        default=False,
        description="If True, generate prose even when none of the target entities "
        "exist in the current world state (a fallback POV is used and the implausibility "
        "reason is reported on the result).",
    )

# ==========================================
# 5. THE INTERROGATOR (Graph RAG)
# ==========================================
class InterrogationQuery(_QueryBase):
    """Runs pathfinding on the AMWN without advancing time or writing prose."""
    query_type: Literal["interrogate"] = "interrogate"
    question: str = Field(description="e.g., 'Is there a physical path for Macbeth to reach the courtyard unseen?'")
    require_proof: bool = Field(default=True, description="Returns the Causal Bridges as mathematical proof.")

# ==========================================
# 6. GENERAL QUESTION (Full-Graph Q&A)
# ==========================================
class GeneralQuery(_QueryBase):
    """Open-ended question answered against the full world-state graph.

    Unlike InterrogationQuery (which targets pathfinding and requires proof),
    GeneralQuery accepts any natural-language question and returns the full
    omniscient graph so an LLM can reason freely over all entities, events,
    locations, topology, and timeline.
    """
    query_type: Literal["general"] = "general"
    question: str = Field(
        description="Any question about the world state, e.g. 'What are all the "
        "relationships between the Capulets and Montagues?'"
    )
    include_topology: bool = Field(
        default=True,
        description="Include causal, spatial, social, and information edges in the response.",
    )

# ==========================================
# 7. MANUAL EDIT (User-authored prose)
# ==========================================
class ManualEditQuery(_QueryBase):
    """User-supplied prose that bypasses generation.

    The engine skips physics simulation and LLM rendering.  Instead the
    user's text is treated as ground truth, re-extracted into topology,
    and merged into the world model.
    """
    query_type: Literal["manual_edit"] = "manual_edit"
    edited_prose: str = Field(
        description="The user's manually written or edited narrative prose.",
    )
    description: str = Field(
        default="",
        description="Optional human-readable description of the changes.",
    )
    focus_entity_ids: List[str] = Field(
        default_factory=list,
        description="Entities most affected by the edit (for ego-graph scoping).",
    )
    insert_after_event_id: Optional[str] = Field(
        default=None,
        description=(
            "Event id whose ``fabula_time`` anchors this edit. New "
            "events extracted from ``edited_prose`` are placed at "
            "``anchor.fabula_time + extraction.fabula_time_spacing`` "
            "and onwards, so the edit lands at the right point in "
            "chronology rather than colliding with existing events. "
            "When unset (and ``insert_at_fabula_time`` is also unset) "
            "the edit appends after the current chronological end."
        ),
    )
    insert_at_fabula_time: Optional[int] = Field(
        default=None,
        description=(
            "Explicit fabula_time anchor for the edit. Overrides "
            "``insert_after_event_id`` when both are set. Use this "
            "for inserting between known beats or backfilling "
            "history."
        ),
    )
    replace_event_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Existing event ids to remove before merging the "
            "re-extracted topology, for true *replace* semantics. "
            "Their dependent causal/social/spatial/info edges are "
            "removed transitively. Leave empty for additive edits."
        ),
    )
    # Extended deletion vocabulary (P6 of prose-merge completeness).
    # Each list flows into the matching ``ChunkTopology.removed_*``
    # field so the merge step's deletion pass cascades dependent
    # edges/snapshots/concerns. Leave empty for additive edits.
    replace_entity_ids: List[str] = Field(default_factory=list)
    replace_object_ids: List[str] = Field(default_factory=list)
    replace_location_ids: List[str] = Field(default_factory=list)
    replace_world_trait_ids: List[str] = Field(default_factory=list)
    replace_channel_ids: List[str] = Field(default_factory=list)
    replace_proposition_ids: List[str] = Field(default_factory=list)
    replace_concern_ids: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="(entity_id, concern_id) pairs to drop from the world.",
    )

# ==========================================
# 8. EVALUATION (Full-story quality audit)
# ==========================================
class EvaluationQuery(_QueryBase):
    """Runs a full-story evaluation using the NarrativeOrderObject scorecard.

    Collects all prose across versions, computes engine metrics from
    causal physics and directive assembly, and produces a structured
    quality report combining quantitative metrics with LLM literary critique.
    """
    query_type: Literal["evaluate"] = "evaluate"
    focus_entity_ids: List[str] = Field(
        default_factory=list,
        description="Entities to focus the evaluation on (empty = all).",
    )
    include_full_prose: bool = Field(
        default=True,
        description="Include the full reconstructed prose in the evaluation.",
    )


class EvaluationResult(BaseModel):
    """User-facing evaluation report for a full story."""
    narrative_order: Any = Field(
        description=(
            "The NarrativeOrderObject scorecard (CausalPhysicsFeedback + "
            "AffectiveStateFeedback + StoryQualitySynthesis + overall_pass)."
        ),
    )
    story_prose_evaluated: str = Field(
        default="",
        description="The full prose that was evaluated.",
    )
    version_count: int = Field(
        default=0,
        description="Number of versions whose prose was combined.",
    )


# ==========================================
# THE MASTER ROUTER
# ==========================================
UserRequest = Union[
    ObservationQuery, 
    InterventionQuery, 
    CounterfactualQuery, 
    DirectiveQuery, 
    InterrogationQuery,
    GeneralQuery,
    ManualEditQuery,
    EvaluationQuery,
]


# ---------------------------------------------------------------------
# Migration helper — lifts the legacy ``Dict[str, Any]`` intervention
# shape into typed ``DoTarget`` lists. Idempotent: if ``do_targets`` /
# ``historical_do_targets`` is already populated it is returned as-is.
#
# The legacy dict shape supports only event-level surgery, e.g.
# ``{"EVT_DUNCAN_MURDER": "averted"}``; values are treated as
# falsey-string-means-not-occurred.
# ---------------------------------------------------------------------
def _coerce_legacy_dict(legacy: Dict[str, Any]) -> List[DoTarget]:
    """Translate a legacy intervention dict into typed ``DoEvent`` targets."""
    targets: List[DoTarget] = []
    if not legacy:
        return targets
    # Falsey value-strings emitted by various legacy callers / LLM parsers
    # when the surgery is "prevent / avert / remove this event". Without
    # the broader set here a value like ``"prevented"`` was treated as
    # truthy and the resulting ``DoEvent`` was flipped to ``occurred=True``,
    # which silently inverted the counterfactual semantics and stopped the
    # renderer from emitting the precursor-attempt-survival hint.
    _NOT_OCCURRED_VALUES = {
        "averted",
        "false",
        "no",
        "not_occurred",
        "absent",
        "prevented",
        "removed",
        "blocked",
        "did_not_occur",
        "didnt_occur",
        "didn't_occur",
        "never_happened",
    }
    for key, value in legacy.items():
        if not isinstance(key, str):
            continue
        # Dotted-key form ("EVT_X.event_type") emitted by
        # _items_to_dotted_dict in query_parsing — strip the property
        # suffix so the DoEvent carries a clean event id rather than a
        # malformed "EVT_X.event_type" identifier the engine and prose
        # renderer cannot resolve back to a real event node.
        base_key, _, prop = key.partition(".")
        if base_key.startswith("EVT_"):
            value_str = value.lower() if isinstance(value, str) else None
            falsey = (
                value is False
                or value is None
                or (value_str is not None and value_str in _NOT_OCCURRED_VALUES)
            )
            # When the dotted property is ``event_type`` the LLM-style
            # value carries the semantic flip directly: "prevented" /
            # "averted" → did not occur; "occurred" / explicit event-type
            # string → did occur. Honour that here so the renderer sees
            # ``occurred=False`` and emits the precursor-attempt hint.
            if prop == "event_type" and value_str == "occurred":
                occurred = True
            else:
                occurred = not falsey
            targets.append(DoEvent(event_id=base_key, occurred=occurred))
            continue
        # ---- Proposition truth clamp: ``PROP_X.truth = bool`` -----
        if base_key.startswith("PROP_") and prop in ("truth", "is_true"):
            try:
                targets.append(DoProposition(
                    proposition_id=base_key, truth=bool(value),
                ))
            except Exception:
                pass
            continue
        # ---- World-trait magnitude clamp: ``WORLD_X.value`` / ``.magnitude`` ----
        if base_key.startswith("WORLD_") and prop in ("value", "magnitude", "strength"):
            if value is None:
                continue
            try:
                targets.append(DoWorldTrait(
                    world_trait_id=base_key, value=float(value),
                ))
            except (TypeError, ValueError):
                pass
            continue
        # ---- Object position / ownership: ``OBJ_X.location_id`` / ``.owner_id`` ----
        if base_key.startswith("OBJ_") and prop in ("location_id", "owner_id"):
            kwargs: Dict[str, Any] = {"object_id": base_key}
            if prop == "location_id":
                if value is None:
                    kwargs["set_location_null"] = True
                else:
                    kwargs["new_location_id"] = str(value)
            else:  # owner_id
                if value is None:
                    kwargs["set_owner_null"] = True
                else:
                    kwargs["new_owner_id"] = str(value)
            try:
                targets.append(DoNarrativeObject(**kwargs))
            except Exception:
                pass
            continue
        # ---- Entity trait clamp: ``ENT_X.traits.<name> = float`` ----
        # Mirrors the dotted shape emitted by ``_lift_do_targets_to_legacy_dict``
        # for ``DoTrait`` round-trips. Anything outside the ``traits.``
        # namespace stays in the legacy dict (handled by the engine's
        # dotted-key state interpreter).
        if base_key.startswith("ENT_") and prop.startswith("traits.") and value is not None:
            trait_name = prop[len("traits."):]
            if trait_name:
                try:
                    targets.append(DoTrait(
                        holder_id=base_key,
                        trait_name=trait_name,
                        value=float(value),
                    ))
                except (TypeError, ValueError):
                    pass
                continue
        # NOTE: ``DoBelief`` and ``DoConcern`` cannot be safely coerced
        # from a single dotted key because they require a (holder, target)
        # or (holder, concern_id) pair the legacy shape never carries.
        # Callers must supply those via the typed ``do_targets`` /
        # ``historical_do_targets`` channel.
    return targets


# Prefixes that ONLY round-trip through the typed ``DoTarget`` channel.
# Legacy dotted-key encodings for these are silently ignored by both the
# coercer above and the engine's dotted-key dispatcher (instantiator.py
# / causal_physics.py), so we surface a ``UserWarning`` at query
# construction time to make the desync visible.
_TYPED_ONLY_LEGACY_PREFIXES = (
    "BEL_",
    "CCN_",
    "REL_",
    "CAUSAL_EDGE_",
    "SPATIAL_EDGE_",
)

# Module-level cache for the strict-mode env var. ``validate_assignment``
# means ``_warn_on_unhandled_legacy_keys`` is called on every field
# assignment to an Intervention / Counterfactual query, so resolving the
# env var on each call adds noticeable overhead. Read once at import
# time; tests that need to flip the mode can call
# :func:`_refresh_strict_mode_from_env` (also invoked by a pytest
# autouse fixture in the test suite).
_STRICT_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _is_strict_legacy_prefixes_enabled() -> bool:
    import os
    return (
        os.environ.get("SHADOW_LOOM_STRICT_LEGACY_PREFIXES", "")
        .strip()
        .lower()
        in _STRICT_TRUE_VALUES
    )


_STRICT_LEGACY_PREFIXES = _is_strict_legacy_prefixes_enabled()


def _refresh_strict_mode_from_env() -> None:
    """Re-read the ``SHADOW_LOOM_STRICT_LEGACY_PREFIXES`` env var. Public
    for tests using ``monkeypatch.setenv`` that need the new value to
    take effect without re-importing the module."""
    global _STRICT_LEGACY_PREFIXES
    _STRICT_LEGACY_PREFIXES = _is_strict_legacy_prefixes_enabled()


def _warn_on_unhandled_legacy_keys(
    legacy: Dict[str, Any],
    coerced: List[DoTarget],
    *,
    field_label: str,
    model: Optional[BaseModel] = None,
) -> None:
    """Emit a ``UserWarning`` for legacy dict keys whose prefix indicates a
    surgery (belief / concern / relationship / causal-edge / spatial-edge)
    that has no single-key legacy encoding and was therefore dropped on
    the floor by both the coercer and the engine's dotted-key dispatcher.

    ``field_label`` is the public field name (``"interventions"`` /
    ``"historical_interventions"``) included in the warning text so the
    caller can locate the offending payload.

    ``model`` (optional): when provided, the helper de-duplicates
    warnings per-instance by stashing the last-warned offender signature
    in a private attribute on the model. ``validate_assignment=True``
    re-fires the model_validator on EVERY field write (even unrelated
    ones like ``target_node_ids``), so without this guard a caller
    setting N fields would emit the same warning N times.
    """
    if not legacy:
        return
    offenders: List[str] = []
    for key in legacy.keys():
        if not isinstance(key, str):
            continue
        base_key = key.partition(".")[0]
        if base_key.startswith(_TYPED_ONLY_LEGACY_PREFIXES):
            offenders.append(key)
    if not offenders:
        return
    signature = (field_label, frozenset(offenders))
    if model is not None:
        already = getattr(model, "__shadow_loom_warned_legacy_keys__", None)
        if already is None:
            already = set()
            # Bypass validate_assignment (and pydantic's field rejection
            # of unknown attrs) via object.__setattr__.
            object.__setattr__(
                model, "__shadow_loom_warned_legacy_keys__", already,
            )
        if signature in already:
            return
        already.add(signature)
    msg = (
        f"{field_label} contains keys with prefixes that have no "
        f"single-key legacy encoding and were ignored: {sorted(offenders)!r}. "
        "Encode belief / concern / relationship / causal-edge / "
        "spatial-edge surgeries via typed do_targets "
        "(DoBelief / DoConcern / DoRelationship / DoCausalEdge / "
        "DoSpatialEdge) instead."
    )
    # Opt-in strict mode: set ``SHADOW_LOOM_STRICT_LEGACY_PREFIXES=1``
    # in the environment to fail loudly (e.g. in CI) instead of merely
    # warning. Useful for shaking out LLM / MCP / REST callers that
    # silently emit unencodable surgeries.
    if _STRICT_LEGACY_PREFIXES:
        raise ValueError(msg)
    import warnings
    warnings.warn(msg, UserWarning, stacklevel=3)
