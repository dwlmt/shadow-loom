from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Union

# =====================================================================
# PART 1: THE GRAPH DATABASE (The Reality Engine)
# =====================================================================

# --- 0. AMWN BASE CLASS (The Multiverse Tag) ---
class AMWNNode(BaseModel):
    world_id: Literal["factual", "shadow"] = Field(
        default="factual", 
        description="Tracks if this node belongs to the true timeline or a 'What If' branch."
    )

# --- 1. CORE PROPERTIES (The Physics & Soul) ---
class TraitVector(BaseModel):
    value: float = Field(description="0.0 to 1.0 (Current level of the trait)")
    inertia: float = Field(description="0.0 to 1.0 (Force required to shatter this trait. 1.0 = permanent)")

class AmbientVector(BaseModel):
    value: float = Field(description="0.0 to 1.0 (How intense is this state?)")
    volatility: float = Field(description="0.0 to 1.0 (How fast does this change? 0.0 = immutable)")

class Affordance(BaseModel):
    action: str = Field(description="What this object can do (e.g., 'unlock', 'kill', 'read')")
    target_type: str = Field(description="What it acts upon (e.g., 'Door', 'Entity')")

class Belief(BaseModel):
    target_id: str = Field(description="ID of the object/entity/event they hold a belief about.")
    perceived_state: str = Field(description="What they THINK is true (e.g., 'Cup is safe').")
    confidence: float = Field(description="0.0 to 1.0 (How sure are they?)")
    inertia: float = Field(description="0.0 to 1.0 (How stubborn is this belief?)")

# --- 2. THE NODES (The Nouns) ---
class Location(AMWNNode):
    id: str = Field(description="Unique ID, e.g., LOC_COURTYARD")
    name: str
    connected_locations: List[str] = Field(description="IDs of adjacent rooms. Prevents teleporting.")
    ambient_states: Dict[str, AmbientVector] = Field(description="Hidden environmental U traits. e.g., {'visibility': AmbientVector}")
    constants: List[str] = Field(default_factory=list, description="Immutable boolean tags, e.g., ['underwater']")

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
    location_id: str = Field(description="Where are they right now?")
    status: Literal["healthy", "injured", "dead", "unconscious"]
    traits: Dict[str, TraitVector] = Field(description="Multidimensional psychology.")
    beliefs: List[Belief] = Field(default_factory=list, description="Epistemic state for Dramatic Irony/Suspense.")
    constants: List[str] = Field(default_factory=list, description="Immutable boolean tags, e.g., ['blind', 'undead']")

class EventNode(AMWNNode):
    id: str = Field(description="Unique ID, e.g., EVT_DUNCAN_MURDER")
    timestamp: int = Field(description="Chronological integer (1, 2, 3...)")
    event_type: Literal["choice", "outcome", "revelation"]
    actor_id: Optional[str] = Field(description="Who did it? Null if natural event.")
    description: str

# --- 3. THE EDGES (The Verbs & Bridges) ---
class CausalEdge(BaseModel):
    source_id: str = Field(description="ID of the Entity, Object, Location, or prior Event.")
    target_id: str = Field(description="ID of the resulting EventNode.")
    mechanism: Literal["physical", "psychological", "social", "epistemic"]

class RelationshipEdge(BaseModel):
    source_entity_id: str
    target_entity_id: str
    affinity: float = Field(description="-1.0 (Hatred) to 1.0 (Love/Adoration)")
    friction: float = Field(description="0.0 (Calm/Predictable) to 1.0 (Volatile/High-Energy)")
    power_dynamic: float = Field(description="-1.0 (Submission) to 1.0 (Dominance)")
    inertia: float = Field(description="0.0 to 1.0 (How hard is it to alter this dynamic?)")

# --- 4. THE MASTER STATE (The Database Payload for Narrative structure) ---
class WorldStateV1(BaseModel):
    locations: Dict[str, Location]
    objects: Dict[str, NarrativeObject]
    entities: Dict[str, Entity]
    events: List[EventNode]
    causal_topology: List[CausalEdge]
    social_topology: List[RelationshipEdge]