from pydantic import BaseModel, Field
from typing import Any, Optional, Literal, Union, Dict, List

# ==========================================
# 1. THE OBSERVATION (Rung 1: Natural Progression)
# ==========================================
class ObservationQuery(BaseModel):
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
class InterventionQuery(BaseModel):
    """
    Forces MULTIPLE variables to specific states simultaneously in the present moment,
    cutting incoming edges, and calculates the future from here.
    """
    query_type: Literal["intervention"] = "intervention"
    interventions: Dict[str, Any] = Field(
        description="A dictionary of do-operator targets. Values are strings for state changes, or dicts for genesis spawns."
    )

# ==========================================
# 3. THE COUNTERFACTUAL (Rung 3: Abduction)
# ==========================================
class CounterfactualQuery(BaseModel):
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

# ==========================================
# 4. THE NARRATIVE DIRECTIVE (Merged Suspense & Emotion)
# ==========================================
class DirectiveQuery(BaseModel):
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

# ==========================================
# 5. THE INTERROGATOR (Graph RAG)
# ==========================================
class InterrogationQuery(BaseModel):
    """Runs pathfinding on the AMWN without advancing time or writing prose."""
    query_type: Literal["interrogate"] = "interrogate"
    question: str = Field(description="e.g., 'Is there a physical path for Macbeth to reach the courtyard unseen?'")
    require_proof: bool = Field(default=True, description="Returns the Causal Bridges as mathematical proof.")

# ==========================================
# 6. GENERAL QUESTION (Full-Graph Q&A)
# ==========================================
class GeneralQuery(BaseModel):
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
class ManualEditQuery(BaseModel):
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
class EvaluationQuery(BaseModel):
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