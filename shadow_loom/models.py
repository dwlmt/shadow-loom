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
    established_at_fabula: int = Field(default=0, description="Fabula time when this belief was formed. Used for counterfactual time-slicing.")

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
    actor_id: Optional[str] = Field(default=None, description="Who did it? Null if natural event.")
    target_id: Optional[str] = Field(default=None, description="Who/what was acted upon? e.g., ENT_DUNCAN in a murder event.")
    description: str

class AMWNEdge(BaseModel):
    """Base class for all topology edges. Distinct from AMWNNode."""
    world_id: str = Field(default="factual", description="Allows edges to exist only in shadow branches.")

# ==========================================
# 1. THE VOLATILE INTERVAL: InformationEdge
# ==========================================
class InformationEdge(AMWNEdge):
    source_id: str = Field(description="Must be an ENT_ or OBJ_ ID")
    target_ids: List[str] = Field(description="Allows 1-to-Many broadcasting")
    
    # UPGRADE: Freeform string with suggestions
    medium: str = Field(
        description="The channel of communication. e.g., 'telephone', 'telepathy', 'shouting', 'magic_mirror', 'carrier_pigeon'"
    )
    is_encrypted: bool = Field(default=False, description="If False, triggers Eavesdropping Leakage.")
    
    established_at_fabula: int
    terminated_at_fabula: Optional[int] = None
    discovered_at_syuzhet: int = 0

# ==========================================
# 2. THE EXPLANATORY LINK: CausalEdge
# ==========================================
class CausalEdge(AMWNEdge):
    source_event_id: str = Field(description="Must be an EVT_ID")
    target_node_id: str = Field(description="The Node altered by the event.")
    
    # UPGRADE: The Mechanism
    mechanism: str = Field(
        description="The 'how' of the causality. e.g., 'physical_force', 'epistemic_revelation', 'social_coercion', 'magic_mutation'"
    )
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description="Statistical confidence for PyMC variance: weak=high variance, strong=low variance."
    )
    
    fabula_time: int = Field(description="The exact physics tick this cause took effect.")

# ==========================================
# 3. THE ACCUMULATOR EDGE: RelationshipEdge
# ==========================================
class RelationshipEdge(AMWNEdge):
    """Tracks continuous psychological and social metrics."""
    source_entity_id: str = Field(description="Must be an ENT_ ID")
    target_entity_id: str = Field(description="Must be an ENT_ ID")
    
    # --- SOCIAL DELTAS ---
    affinity: float = Field(default=0.0, description="-1.0 (Hate) to 1.0 (Love)")
    fear: float = Field(default=0.0, description="0.0 (None) to 1.0 (Terrified)")
    power_dynamic: float = Field(default=0.0, description="-1.0 (Subservient) to 1.0 (Dominant)")
    
    # --- INERTIA (Resistance to relationship mutation) ---
    inertia: float = Field(default=0.3, description="0.0 to 1.0 (Force required to shift this bond. 1.0 = unbreakable)")
    
    # --- STATISTICAL CONFIDENCE ---
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="moderate",
        description="Statistical confidence for Bayesian variance: weak=high variance, strong=low variance."
    )
    
    # --- TEMPORAL TRACKING ---
    last_updated_fabula: int = Field(
        default=0, 
        description="Relationships don't 'end', they just mutate. This timestamp dictates how far back to roll for counterfactuals."
    )

# ==========================================
# 4. THE EPOCH EDGE: SpatialEdge
# ==========================================
class SpatialEdge(AMWNEdge):
    """Tracks the physical flow of matter (Architecture)."""
    source_id: str = Field(description="Must be a LOC_ ID")
    target_id: str = Field(description="Must be a LOC_ ID")
    
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