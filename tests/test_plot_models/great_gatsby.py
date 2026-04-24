from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, InformationEdge, RelationshipEdge, TraitVector,
    Affordance, Belief,
)

# =============================================================================
# THE GREAT GATSBY — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
                "LOC_WEST_EGG": Location(
            name="West Egg (New Money)",
            description="West Egg (New Money)",
            ambient_state={"opulence": {"value": 0.9, "volatility": 0.2}, "artifice": {"value": 0.8, "volatility": 0.3}},
        ),
                "LOC_GATSBY_MANSION": Location(
            name="Gatsby's Mansion",
            description="Gatsby's Mansion",
            ambient_state={"opulence": {"value": 1.0, "volatility": 0.2}, "loneliness": {"value": 0.7, "volatility": 0.4}},
        ),
                "LOC_EAST_EGG": Location(
            name="East Egg (Old Money)",
            description="East Egg (Old Money)",
            ambient_state={"privilege": {"value": 0.95, "volatility": 0.05}},
        ),
                "LOC_BUCHANAN_MANSION": Location(
            name="Buchanan Mansion",
            description="Buchanan Mansion",
            ambient_state={"tension": {"value": 0.6, "volatility": 0.5}},
        ),
                "LOC_VALLEY_OF_ASHES": Location(
            name="Valley of Ashes",
            description="Valley of Ashes",
            ambient_state={"desolation": {"value": 0.9, "volatility": 0.1}, "poverty": {"value": 0.85, "volatility": 0.1}},
        ),
                "LOC_WILSON_GARAGE": Location(
            name="Wilson's Garage",
            description="Wilson's Garage",
            ambient_state={"despair": {"value": 0.8, "volatility": 0.4}},
        ),
                "LOC_NEW_YORK": Location(
            name="New York City",
            description="New York City",
            ambient_state={"energy": {"value": 0.8, "volatility": 0.3}},
        ),
                "LOC_PLAZA_HOTEL": Location(
            name="Plaza Hotel Suite",
            description="Plaza Hotel Suite",
            ambient_state={"tension": {"value": 0.9, "volatility": 0.5}},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_GREEN_LIGHT": NarrativeObject(
            id="OBJ_GREEN_LIGHT",
            name="The Green Light (Daisy's Dock)",
            location_id="LOC_EAST_EGG",
            owner_id=None,
            properties={"state": "perpetually_glowing", "symbolism": "gatsby_dream"},
            affordances=[
                Affordance(action="symbolize_hope", target_type="Entity"),
            ],
        ),
        "OBJ_GATSBY_CAR": NarrativeObject(
            id="OBJ_GATSBY_CAR",
            name="Gatsby's Yellow Car",
            location_id="LOC_VALLEY_OF_ASHES",
            owner_id="ENT_GATSBY",
            properties={"state": "involved_in_hit_and_run"},
            affordances=[
                Affordance(action="transport", target_type="Entity"),
                Affordance(action="kill", target_type="Entity"),
            ],
        ),
        "OBJ_GUN": NarrativeObject(
            id="OBJ_GUN",
            name="George Wilson's Gun",
            location_id="LOC_GATSBY_MANSION",
            owner_id=None,
            properties={"state": "discharged"},
            affordances=[
                Affordance(action="shoot", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_GATSBY": Entity(
            id="ENT_GATSBY",
            name="Jay Gatsby",
            location_id="LOC_GATSBY_MANSION",
            status="dead",
            traits={
                "romanticism": TraitVector(value=0.95, inertia=0.95),
                "ambition": TraitVector(value=0.9, inertia=0.8),
                "deception": TraitVector(value=0.7, inertia=0.5),
                "devotion": TraitVector(value=0.95, inertia=0.95),
                "naivete": TraitVector(value=0.7, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_DAISY", perceived_state="Daisy will choose me if I show enough wealth", confidence=0.85, inertia=0.9, established_at_fabula=0),
                Belief(target_id="OBJ_GREEN_LIGHT", perceived_state="The dream is still within reach", confidence=0.8, inertia=0.9, established_at_fabula=0),
                Belief(target_id="ENT_DAISY", perceived_state="Daisy never truly loved Tom — she was always waiting for me", confidence=0.8, inertia=0.85, established_at_fabula=0),
            ],
        ),
        "ENT_NICK": Entity(
            id="ENT_NICK",
            name="Nick Carraway",
            location_id="LOC_WEST_EGG",
            status="healthy",
            traits={
                "observation": TraitVector(value=0.85, inertia=0.8),
                "reserve": TraitVector(value=0.7, inertia=0.6),
                "moral_sense": TraitVector(value=0.75, inertia=0.7),
                "disillusionment": TraitVector(value=0.8, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_GATSBY", perceived_state="Gatsby is worth more than the whole damn bunch put together", confidence=0.8, inertia=0.7, established_at_fabula=0),
            ],
        ),
        "ENT_DAISY": Entity(
            id="ENT_DAISY",
            name="Daisy Buchanan",
            location_id="LOC_EAST_EGG",
            status="healthy",
            traits={
                "charm": TraitVector(value=0.9, inertia=0.7),
                "indecisiveness": TraitVector(value=0.85, inertia=0.7),
                "self_preservation": TraitVector(value=0.9, inertia=0.8),
                "carelessness": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_GATSBY", perceived_state="Gatsby will take the blame and protect me", confidence=0.9, inertia=0.5, established_at_fabula=0),
            ],
        ),
        "ENT_TOM": Entity(
            id="ENT_TOM",
            name="Tom Buchanan",
            location_id="LOC_EAST_EGG",
            status="healthy",
            traits={
                "aggression": TraitVector(value=0.85, inertia=0.8),
                "entitlement": TraitVector(value=0.9, inertia=0.9),
                "hypocrisy": TraitVector(value=0.85, inertia=0.8),
                "dominance": TraitVector(value=0.9, inertia=0.85),
            },
            beliefs=[
                Belief(target_id="ENT_GATSBY", perceived_state="Gatsby is a criminal bootlegger and a fraud", confidence=0.9, inertia=0.8, established_at_fabula=0),
                Belief(target_id="ENT_DAISY", perceived_state="Daisy will never leave me — she belongs to my world", confidence=0.9, inertia=0.85, established_at_fabula=0),
            ],
        ),
        "ENT_JORDAN": Entity(
            id="ENT_JORDAN",
            name="Jordan Baker",
            location_id="LOC_EAST_EGG",
            status="healthy",
            traits={
                "cynicism": TraitVector(value=0.7, inertia=0.6),
                "dishonesty": TraitVector(value=0.6, inertia=0.5),
            },
        ),
        "ENT_MYRTLE": Entity(
            id="ENT_MYRTLE",
            name="Myrtle Wilson",
            location_id="LOC_VALLEY_OF_ASHES",
            status="dead",
            traits={
                "ambition": TraitVector(value=0.8, inertia=0.5),
                "desperation": TraitVector(value=0.75, inertia=0.5),
                "vitality": TraitVector(value=0.8, inertia=0.4),
            },
            beliefs=[
                Belief(target_id="ENT_TOM", perceived_state="Tom will leave Daisy and take me into his world", confidence=0.6, inertia=0.4, established_at_fabula=0),
            ],
        ),
        "ENT_GEORGE": Entity(
            id="ENT_GEORGE",
            name="George Wilson",
            location_id="LOC_WILSON_GARAGE",
            status="dead",
            traits={
                "despair": TraitVector(value=0.9, inertia=0.6),
                "passivity": TraitVector(value=0.7, inertia=0.5),
                "grief": TraitVector(value=0.95, inertia=0.4),
                "jealousy": TraitVector(value=0.8, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_MYRTLE", perceived_state="Myrtle has a secret lover in New York", confidence=0.7, inertia=0.5, established_at_fabula=8),
                Belief(target_id="ENT_GATSBY", perceived_state="The owner of the yellow car is Myrtle's lover and killed her", confidence=1.0, inertia=0.9, established_at_fabula=12),
            ],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_NICK_ARRIVES", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_id="ENT_NICK", description="Nick Carraway arrives in West Egg, renting a bungalow next to Gatsby's mansion."),
        EventNode(id="EVT_DINNER_EAST_EGG", fabula_time=2, syuzhet_index=2, event_type="outcome", actor_id="ENT_NICK", description="Nick dines with Daisy and Tom in East Egg. Jordan reveals Tom has a mistress."),
        EventNode(id="EVT_GATSBY_STARES_GREEN_LIGHT", fabula_time=3, syuzhet_index=3, event_type="outcome", actor_id="ENT_GATSBY", description="Nick sees Gatsby standing alone staring at the green light across the bay."),
        EventNode(id="EVT_TOM_MYRTLE_PARTY", fabula_time=4, syuzhet_index=4, event_type="outcome", actor_id="ENT_TOM", description="Tom takes Nick to the Valley of Ashes; party in the New York apartment ends with Tom breaking Myrtle's nose."),
        EventNode(id="EVT_GATSBY_PARTY", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_id="ENT_GATSBY", description="Nick attends Gatsby's lavish party and meets Gatsby, who claims they served together in the war."),
        EventNode(id="EVT_JORDAN_REVEALS_PAST", fabula_time=6, syuzhet_index=6, event_type="revelation", actor_id="ENT_JORDAN", description="Jordan reveals that Gatsby and Daisy were in love in 1917 and that Gatsby's parties are all for her."),
        EventNode(id="EVT_GATSBY_DAISY_REUNION", fabula_time=7, syuzhet_index=7, event_type="choice", actor_id="ENT_GATSBY", description="Gatsby uses Nick to stage a reunion with Daisy. They begin an affair."),
        EventNode(id="EVT_TOM_DISCOVERS_AFFAIR", fabula_time=8, syuzhet_index=8, event_type="revelation", actor_id="ENT_TOM", description="Tom discovers Gatsby and Daisy's affair when Daisy addresses Gatsby with unguarded intimacy."),
        EventNode(id="EVT_PLAZA_CONFRONTATION", fabula_time=9, syuzhet_index=9, event_type="outcome", actor_id="ENT_TOM", description="At the Plaza Hotel, Tom and Gatsby argue. Tom reveals Gatsby's bootlegging. Daisy chooses to stay with Tom."),
        EventNode(id="EVT_MYRTLE_KILLED", fabula_time=10, syuzhet_index=10, event_type="outcome", actor_id="ENT_DAISY", target_id="ENT_MYRTLE", description="Driving back from the Plaza, Daisy (driving Gatsby's car) strikes and kills Myrtle Wilson."),
        EventNode(id="EVT_GATSBY_TAKES_BLAME", fabula_time=11, syuzhet_index=11, event_type="choice", actor_id="ENT_GATSBY", description="Gatsby tells Nick that Daisy was driving but he intends to take the blame."),
        EventNode(id="EVT_TOM_TELLS_GEORGE", fabula_time=12, syuzhet_index=12, event_type="choice", actor_id="ENT_TOM", description="Tom tells George Wilson that the car that killed Myrtle belongs to Gatsby."),
        EventNode(id="EVT_GATSBY_MURDERED", fabula_time=13, syuzhet_index=13, event_type="outcome", actor_id="ENT_GEORGE", target_id="ENT_GATSBY", description="George shoots Gatsby in his pool, then kills himself, believing Gatsby was Myrtle's lover and killer."),
        EventNode(id="EVT_FUNERAL", fabula_time=14, syuzhet_index=14, event_type="outcome", actor_id=None, description="Gatsby's funeral is sparsely attended. His father Henry Gatz arrives."),
        EventNode(id="EVT_NICK_LEAVES", fabula_time=15, syuzhet_index=15, event_type="choice", actor_id="ENT_NICK", description="Nick, disillusioned, decides to leave New York and return to the Midwest."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_event_id="EVT_NICK_ARRIVES", target_node_id="EVT_DINNER_EAST_EGG", mechanism="social", fabula_time=1),
        CausalEdge(source_event_id="EVT_DINNER_EAST_EGG", target_node_id="EVT_GATSBY_STARES_GREEN_LIGHT", mechanism="social", fabula_time=2),
        CausalEdge(source_event_id="EVT_GATSBY_PARTY", target_node_id="EVT_JORDAN_REVEALS_PAST", mechanism="epistemic", fabula_time=5),
        CausalEdge(source_event_id="EVT_JORDAN_REVEALS_PAST", target_node_id="EVT_GATSBY_DAISY_REUNION", mechanism="epistemic", fabula_time=6),
        CausalEdge(source_event_id="EVT_GATSBY_DAISY_REUNION", target_node_id="EVT_TOM_DISCOVERS_AFFAIR", mechanism="social", fabula_time=7),
        CausalEdge(source_event_id="EVT_TOM_DISCOVERS_AFFAIR", target_node_id="EVT_PLAZA_CONFRONTATION", mechanism="psychological", fabula_time=8),
        CausalEdge(source_event_id="EVT_PLAZA_CONFRONTATION", target_node_id="EVT_MYRTLE_KILLED", mechanism="psychological", fabula_time=9),
        CausalEdge(source_event_id="EVT_MYRTLE_KILLED", target_node_id="EVT_GATSBY_TAKES_BLAME", mechanism="psychological", fabula_time=10),
        CausalEdge(source_event_id="EVT_MYRTLE_KILLED", target_node_id="EVT_TOM_TELLS_GEORGE", mechanism="social", fabula_time=10),
        CausalEdge(source_event_id="EVT_TOM_TELLS_GEORGE", target_node_id="EVT_GATSBY_MURDERED", mechanism="epistemic", fabula_time=12),
        CausalEdge(source_event_id="EVT_GATSBY_MURDERED", target_node_id="EVT_FUNERAL", mechanism="social", fabula_time=13),
        CausalEdge(source_event_id="EVT_FUNERAL", target_node_id="EVT_NICK_LEAVES", mechanism="psychological", fabula_time=14),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_BUCHANAN_MANSION", target_id="LOC_EAST_EGG"),
        SpatialEdge(source_id="LOC_EAST_EGG", target_id="LOC_NEW_YORK"),
        SpatialEdge(source_id="LOC_EAST_EGG", target_id="LOC_WEST_EGG"),
        SpatialEdge(source_id="LOC_GATSBY_MANSION", target_id="LOC_WEST_EGG"),
        SpatialEdge(source_id="LOC_NEW_YORK", target_id="LOC_PLAZA_HOTEL"),
        SpatialEdge(source_id="LOC_NEW_YORK", target_id="LOC_VALLEY_OF_ASHES"),
        SpatialEdge(source_id="LOC_NEW_YORK", target_id="LOC_WEST_EGG"),
        SpatialEdge(source_id="LOC_VALLEY_OF_ASHES", target_id="LOC_WEST_EGG"),
        SpatialEdge(source_id="LOC_VALLEY_OF_ASHES", target_id="LOC_WILSON_GARAGE"),
    ],
    information_topology=[
        InformationEdge(
            source_id="ENT_JORDAN",
            target_ids=["ENT_NICK"],
            medium="conversation",
            established_at_fabula=6,
            terminated_at_fabula=6,
        ),
        InformationEdge(
            source_id="ENT_TOM",
            target_ids=["ENT_GEORGE"],
            medium="conversation",
            established_at_fabula=12,
            terminated_at_fabula=12,
        ),
    ],
    social_topology=[RelationshipEdge(source_entity_id="ENT_GATSBY", target_entity_id="ENT_DAISY", affinity=1.0, fear=0.3, power_dynamic=-0.3),
        RelationshipEdge(source_entity_id="ENT_DAISY", target_entity_id="ENT_GATSBY", affinity=0.5, fear=0.25, power_dynamic=0.3),
        RelationshipEdge(source_entity_id="ENT_TOM", target_entity_id="ENT_DAISY", affinity=0.4, fear=0.3, power_dynamic=0.7),
        RelationshipEdge(source_entity_id="ENT_TOM", target_entity_id="ENT_MYRTLE", affinity=0.3, fear=0.35, power_dynamic=0.8),
        RelationshipEdge(source_entity_id="ENT_TOM", target_entity_id="ENT_GATSBY", affinity=-0.8, fear=0.45, power_dynamic=0.4),
        RelationshipEdge(source_entity_id="ENT_GEORGE", target_entity_id="ENT_MYRTLE", affinity=0.7, fear=0.3, power_dynamic=-0.5),
        RelationshipEdge(source_entity_id="ENT_NICK", target_entity_id="ENT_GATSBY", affinity=0.7, fear=0.1, power_dynamic=-0.1),
        RelationshipEdge(source_entity_id="ENT_NICK", target_entity_id="ENT_JORDAN", affinity=0.5, fear=0.15, power_dynamic=0.0),
    ],
)
