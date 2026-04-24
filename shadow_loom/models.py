from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional, Literal, Union

# =====================================================================
# PART 1: THE GRAPH DATABASE (The Reality Engine)
# =====================================================================


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
    node_type: Literal["Location"] = "Location"
    name: str
    description: str
    ambient_state: Optional[Dict[str, Any]] = Field(default_factory=dict, description="e.g., {'temperature': 'cold', 'lighting': 'dark'}")

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
    status: Literal["healthy", "injured", "ill", "dead", "unconscious"]
    traits: Dict[str, TraitVector] = Field(description="Multidimensional psychology.")
    beliefs: List[Belief] = Field(default_factory=list, description="Epistemic state for Dramatic Irony/Suspense.")
    constants: List[str] = Field(default_factory=list, description="Immutable boolean tags, e.g., ['blind', 'undead']")

class EventNode(AMWNNode):
    id: str = Field(description="Unique ID, e.g., EVT_DUNCAN_MURDER")
    fabula_time: int = Field(
        description="The strict chronological order (e.g., Year 1000). Used for Causal Physics."
    )
    syuzhet_index: int = Field(
        description="The sequence this appears in the text (e.g., Chapter 4, Paragraph 2). Used for Suspense."
    )
    event_type: Literal["choice", "outcome", "revelation"]
    actor_id: Optional[str] = Field(description="Who did it? Null if natural event.")
    description: str

# --- 3. THE EDGES (The Verbs & Bridges) ---
class CausalEdge(BaseModel):
    source_id: str = Field(description="ID of the Entity, Object, Location, or prior Event.")
    target_id: str = Field(description="ID of the resulting EventNode.")
    mechanism: Literal["physical", "psychological", "social", "epistemic"]

class SpatialEdge(BaseModel):
    """The physical flow of matter (Architecture)."""
    source_id: str = Field(description="Must be a LOC_ ID")
    target_id: str = Field(description="Must be a LOC_ ID")
    is_locked: bool = Field(default=False)
    barrier_item_id: Optional[str] = Field(default=None, description="ID of a NarrativeObject like a door or lock.")

class InformationEdge(BaseModel):
    """Upgraded to handle Broadcasts, Eavesdropping, and Time-Slicing."""
    
    source_id: str = Field(description="Must be an ENT_ or OBJ_ (e.g., a Radio beacon) ID")
    
    target_ids: List[str] = Field(description="List of ENT_ or LOC_ IDs receiving the signal.")
    
    medium: Literal["telephone", "telepathy", "radio", "shouting", "magic_mirror", "raven", "letter", "speech"]
    
    is_encrypted: bool = Field(
        default=False, 
        description="If false, entities in the same spatial Location as the source or target can intercept the payload."
    )
    
    established_at_fabula: int = Field(description="The timestamp when the comms link opened.")
    terminated_at_fabula: Optional[int] = Field(
        default=None, 
        description="The timestamp when the link closed. Null if currently active."
    )

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
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)
    information_topology: List[InformationEdge] = Field(default_factory=list)
    social_topology: List[RelationshipEdge]