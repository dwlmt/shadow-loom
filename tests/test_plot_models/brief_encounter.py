from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, InformationEdge, RelationshipEdge, TraitVector,
    Affordance, Belief,
)

# =============================================================================
# BRIEF ENCOUNTER — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
                "LOC_RAILWAY_STATION": Location(
            name="Milford Junction Railway Station",
            description="Milford Junction Railway Station",
            ambient_state={"transience": {"value": 0.8, "volatility": 0.3}, "routine": {"value": 0.7, "volatility": 0.2}},
        ),
                "LOC_REFRESHMENT_ROOM": Location(
            name="Station Refreshment Room",
            description="Station Refreshment Room",
            ambient_state={"intimacy": {"value": 0.6, "volatility": 0.5}, "public_exposure": {"value": 0.7, "volatility": 0.3}},
        ),
                "LOC_MILFORD_TOWN": Location(
            name="Milford Town (Cinema, Shops, Chemist)",
            description="Milford Town (Cinema, Shops, Chemist)",
            ambient_state={},
        ),
                "LOC_LAURA_HOME": Location(
            name="Laura's Home",
            description="Laura's Home",
            ambient_state={"domesticity": {"value": 0.8, "volatility": 0.1}, "emotional_distance": {"value": 0.6, "volatility": 0.4}},
        ),
                "LOC_STEPHEN_FLAT": Location(
            name="Stephen's Flat",
            description="Stephen's Flat",
            ambient_state={"guilt": {"value": 0.9, "volatility": 0.3}},
        ),
                "LOC_COUNTRYSIDE": Location(
            name="Countryside (Public Meeting Spots)",
            description="Countryside (Public Meeting Spots)",
            ambient_state={},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_GRIT": NarrativeObject(
            id="OBJ_GRIT",
            name="Piece of Grit (in Laura's Eye)",
            location_id=None,
            owner_id=None,
            properties={"state": "removed"},
            affordances=[
                Affordance(action="initiate_contact", target_type="Entity"),
            ],
        ),
        "OBJ_EXPRESS_TRAIN": NarrativeObject(
            id="OBJ_EXPRESS_TRAIN",
            name="Express Train",
            location_id="LOC_RAILWAY_STATION",
            owner_id=None,
            properties={"state": "passing"},
            affordances=[
                Affordance(action="kill", target_type="Entity"),
                Affordance(action="transport", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_LAURA": Entity(
            id="ENT_LAURA",
            name="Laura Jesson",
            location_id="LOC_LAURA_HOME",
            status="healthy",
            traits={
                "propriety": TraitVector(value=0.85, inertia=0.7),
                "passion": TraitVector(value=0.8, inertia=0.5),
                "guilt": TraitVector(value=0.85, inertia=0.6),
                "devotion_to_family": TraitVector(value=0.8, inertia=0.8),
                "emotional_depth": TraitVector(value=0.9, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_ALEC", perceived_state="I love him but our relationship is unworkable", confidence=0.95, inertia=0.7, established_at_fabula=0),
                Belief(target_id="ENT_FRED", perceived_state="I am betraying my good husband and family", confidence=0.95, inertia=0.7, established_at_fabula=0),
                Belief(target_id="ENT_ALEC", perceived_state="Our love is real but we can never act on it \u2014 duty comes first", confidence=0.9, inertia=0.7, established_at_fabula=5),
            ],
        ),
        "ENT_ALEC": Entity(
            id="ENT_ALEC",
            name="Alec Harvey",
            location_id="LOC_RAILWAY_STATION",
            status="healthy",
            traits={
                "idealism": TraitVector(value=0.75, inertia=0.6),
                "passion": TraitVector(value=0.8, inertia=0.5),
                "duty": TraitVector(value=0.7, inertia=0.6),
                "self_sacrifice": TraitVector(value=0.75, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_LAURA", perceived_state="I love her but must end it for my family's sake", confidence=0.9, inertia=0.6, established_at_fabula=0),
            ],
        ),
        "ENT_FRED": Entity(
            id="ENT_FRED",
            name="Fred Jesson",
            location_id="LOC_LAURA_HOME",
            status="healthy",
            traits={
                "steadiness": TraitVector(value=0.85, inertia=0.9),
                "patience": TraitVector(value=0.8, inertia=0.8),
                "perceptiveness": TraitVector(value=0.6, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_LAURA", perceived_state="Something has been troubling her \u2014 but I may not know the full truth", confidence=0.6, inertia=0.5, established_at_fabula=0),
                Belief(target_id="ENT_LAURA", perceived_state="Laura has been emotionally distant lately but I trust she will come back to me", confidence=0.7, inertia=0.6, established_at_fabula=7),
            ],
        ),
        "ENT_DOLLY": Entity(
            id="ENT_DOLLY",
            name="Dolly Messiter",
            location_id="LOC_REFRESHMENT_ROOM",
            status="healthy",
            traits={
                "obliviousness": TraitVector(value=0.9, inertia=0.8),
                "chattiness": TraitVector(value=0.9, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_LAURA", perceived_state="Laura and this man at the table are just casual acquaintances", confidence=0.95, inertia=0.9, established_at_fabula=0),
            ],
        ),
        "ENT_STEPHEN": Entity(
            id="ENT_STEPHEN",
            name="Stephen (Alec's Friend)",
            location_id="LOC_STEPHEN_FLAT",
            status="healthy",
            traits={
                "disapproval": TraitVector(value=0.6, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_ALEC", perceived_state="Alec is being unfaithful and using my flat for it", confidence=0.9, inertia=0.7, established_at_fabula=0),
            ],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_GRIT_IN_EYE", fabula_time=1, syuzhet_index=3, event_type="outcome", actor_id="ENT_ALEC", description="Alec removes a piece of grit from Laura's eye at the station, charming her."),
        EventNode(id="EVT_WEEKLY_MEETINGS", fabula_time=2, syuzhet_index=4, event_type="outcome", actor_id=None, description="Laura and Alec begin meeting on Thursdays — first casually at the chemist, then lunch and films."),
        EventNode(id="EVT_LOVE_ADMITTED", fabula_time=3, syuzhet_index=5, event_type="revelation", actor_id=None, description="Laura and Alec admit they love each other, despite both being married with children."),
        EventNode(id="EVT_FRIENDS_SPOTTED", fabula_time=4, syuzhet_index=6, event_type="outcome", actor_id=None, description="They run into Laura's friends in public, forcing the first of many deceptions."),
        EventNode(id="EVT_STEPHEN_FLAT_ATTEMPT", fabula_time=5, syuzhet_index=7, event_type="choice", actor_id="ENT_ALEC", description="Laura and Alec agree to make love at Stephen's flat, but Stephen returns unexpectedly and subtly chides Alec."),
        EventNode(id="EVT_LAURA_WANDERS", fabula_time=6, syuzhet_index=8, event_type="outcome", actor_id="ENT_LAURA", description="Distraught Laura wanders the streets for three hours until a police officer urges her home."),
        EventNode(id="EVT_RELATIONSHIP_UNWORKABLE", fabula_time=7, syuzhet_index=9, event_type="revelation", actor_id=None, description="Laura and Alec admit their relationship is unworkable."),
        EventNode(id="EVT_ALEC_TAKES_JOB", fabula_time=8, syuzhet_index=10, event_type="choice", actor_id="ENT_ALEC", description="For his family's sake, Alec decides to end the relationship by taking a job in Johannesburg."),
        EventNode(id="EVT_FINAL_MEETING", fabula_time=9, syuzhet_index=1, event_type="outcome", actor_id=None, description="Laura and Alec have their final meeting in the refreshment room. Dolly Messiter interrupts them."),
        EventNode(id="EVT_ALEC_DEPARTS", fabula_time=10, syuzhet_index=2, event_type="outcome", actor_id="ENT_ALEC", description="Alec's train arrives before a proper goodbye. He discreetly squeezes Laura's shoulder and departs."),
        EventNode(id="EVT_SUICIDE_ATTEMPT", fabula_time=11, syuzhet_index=11, event_type="choice", actor_id="ENT_LAURA", description="Overcome with emotion, Laura nearly jumps in front of an express train but gathers herself."),
        EventNode(id="EVT_LAURA_RETURNS_HOME", fabula_time=12, syuzhet_index=12, event_type="outcome", actor_id="ENT_LAURA", description="Laura returns home. Fred acknowledges her distance and thanks her for coming back. She weeps in his arms."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_event_id="EVT_GRIT_IN_EYE", target_node_id="EVT_WEEKLY_MEETINGS", mechanism="social", fabula_time=1),
        CausalEdge(source_event_id="EVT_WEEKLY_MEETINGS", target_node_id="EVT_LOVE_ADMITTED", mechanism="psychological", fabula_time=2),
        CausalEdge(source_event_id="EVT_LOVE_ADMITTED", target_node_id="EVT_FRIENDS_SPOTTED", mechanism="social", fabula_time=3),
        CausalEdge(source_event_id="EVT_LOVE_ADMITTED", target_node_id="EVT_STEPHEN_FLAT_ATTEMPT", mechanism="psychological", fabula_time=3),
        CausalEdge(source_event_id="EVT_STEPHEN_FLAT_ATTEMPT", target_node_id="EVT_LAURA_WANDERS", mechanism="psychological", fabula_time=5),
        CausalEdge(source_event_id="EVT_LAURA_WANDERS", target_node_id="EVT_RELATIONSHIP_UNWORKABLE", mechanism="psychological", fabula_time=6),
        CausalEdge(source_event_id="EVT_RELATIONSHIP_UNWORKABLE", target_node_id="EVT_ALEC_TAKES_JOB", mechanism="psychological", fabula_time=7),
        CausalEdge(source_event_id="EVT_ALEC_TAKES_JOB", target_node_id="EVT_FINAL_MEETING", mechanism="social", fabula_time=8),
        CausalEdge(source_event_id="EVT_FINAL_MEETING", target_node_id="EVT_ALEC_DEPARTS", mechanism="social", fabula_time=9),
        CausalEdge(source_event_id="EVT_ALEC_DEPARTS", target_node_id="EVT_SUICIDE_ATTEMPT", mechanism="psychological", fabula_time=10),
        CausalEdge(source_event_id="EVT_SUICIDE_ATTEMPT", target_node_id="EVT_LAURA_RETURNS_HOME", mechanism="psychological", fabula_time=11),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_COUNTRYSIDE", target_id="LOC_MILFORD_TOWN"),
        SpatialEdge(source_id="LOC_LAURA_HOME", target_id="LOC_RAILWAY_STATION"),
        SpatialEdge(source_id="LOC_MILFORD_TOWN", target_id="LOC_RAILWAY_STATION"),
        SpatialEdge(source_id="LOC_MILFORD_TOWN", target_id="LOC_STEPHEN_FLAT"),
        SpatialEdge(source_id="LOC_RAILWAY_STATION", target_id="LOC_REFRESHMENT_ROOM"),
    ],
    information_topology=[
        InformationEdge(
            source_id="ENT_LAURA",
            target_ids=["ENT_FRED"],
            medium="narration",
            established_at_fabula=12,
            terminated_at_fabula=12,
        ),
        InformationEdge(
            source_id="ENT_STEPHEN",
            target_ids=["ENT_ALEC"],
            medium="subtle_rebuke",
            established_at_fabula=5,
            terminated_at_fabula=5,
        ),
    ],
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_LAURA", target_entity_id="ENT_ALEC", affinity=0.9, fear=0.35, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_ALEC", target_entity_id="ENT_LAURA", affinity=0.85, fear=0.35, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_LAURA", target_entity_id="ENT_FRED", affinity=0.6, fear=0.1, power_dynamic=-0.1),
        RelationshipEdge(source_entity_id="ENT_FRED", target_entity_id="ENT_LAURA", affinity=0.7, fear=0.1, power_dynamic=0.1),
        RelationshipEdge(source_entity_id="ENT_LAURA", target_entity_id="ENT_DOLLY", affinity=0.2, fear=0.25, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_DOLLY", target_entity_id="ENT_LAURA", affinity=0.5, fear=0.0, power_dynamic=0.1),
    ],
)
