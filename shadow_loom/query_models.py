# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

from pydantic import BaseModel, Field
from typing import Any, Optional, Literal, Union, Dict, List


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
    interventions: Dict[str, Any] = Field(
        description="A dictionary of do-operator targets. Values are strings for state changes, or dicts for genesis spawns."
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
    historical_interventions: Dict[str, Any] = Field(
        description="The PAST events to change. e.g., {'EVT_GUARD_DUTY': 'slept'}"
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
    target_effect: Literal["suspense", "surprise", "mystery", "dramatic_irony", "grief", "rage", "joy", "regret", "love", "fear"] = Field(
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