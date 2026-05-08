# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

from pydantic import BaseModel, Field
from typing import Any, Optional, Literal, Tuple, Union, Dict, List
from typing_extensions import Annotated


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


# Discriminated union — Pydantic v2 dispatches on ``target_kind``.
DoTarget = Annotated[
    Union[DoEvent, DoProposition, DoBelief, DoConcern, DoTrait],
    Field(discriminator="target_kind"),
]


# ---------------------------------------------------------------------
# Shared base — every query type carries the user's verbatim natural-
# language request so it can be threaded into directive assembly,
# generation prompts, audit feedback, and the persisted version row.
# ---------------------------------------------------------------------
class _QueryBase(BaseModel):
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
    for key, value in legacy.items():
        if not isinstance(key, str):
            continue
        if key.startswith("EVT_"):
            occurred = not (
                value is False
                or value is None
                or (isinstance(value, str) and value.lower() in {"averted", "false", "no", "not_occurred", "absent"})
            )
            targets.append(DoEvent(event_id=key, occurred=occurred))
        # Heuristic legacy support for other prefixes: leave to query parser
        # to emit typed targets going forward; legacy callers only ever set
        # event ids so we keep this conservative.
    return targets


def coerce_intervention_query(query: InterventionQuery) -> InterventionQuery:
    """Populate ``do_targets`` from ``interventions`` when empty."""
    if not query.do_targets and query.interventions:
        query.do_targets = _coerce_legacy_dict(query.interventions)
    return query


def coerce_counterfactual_query(query: CounterfactualQuery) -> CounterfactualQuery:
    """Populate ``historical_do_targets`` from ``historical_interventions`` when empty."""
    if not query.historical_do_targets and query.historical_interventions:
        query.historical_do_targets = _coerce_legacy_dict(query.historical_interventions)
    return query
