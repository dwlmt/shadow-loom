from pydantic import BaseModel, Field
from typing import Optional, Literal, Union, Dict, List

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
    time_steps: int = Field(default=1, description="How many sequential events to simulate.")
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
    interventions: Dict[str, str] = Field(
        description="A dictionary of multiple do-operator targets. e.g., {'EVT_MURDER': 'prevented', 'LOC_HALL.visibility': '0.0'}"
    )
    commit_to_factual: bool = Field(
        default=False, 
        description="If TRUE, permanently alters T_0. If FALSE, spawns a forward-looking shadow branch."
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
    historical_interventions: Dict[str, str] = Field(
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
    target_entity_id: str = Field(description="The Entity experiencing the emotion or the ignorance.")
    target_effect: Literal["suspense", "dramatic_irony", "grief", "rage", "joy", "regret", "love", "fear"] = Field(
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
# THE MASTER ROUTER
# ==========================================
UserRequest = Union[
    ObservationQuery, 
    InterventionQuery, 
    CounterfactualQuery, 
    DirectiveQuery, 
    InterrogationQuery
]