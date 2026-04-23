from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
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
            id="LOC_RAILWAY_STATION",
            name="Milford Junction Railway Station",
            connected_locations=["LOC_REFRESHMENT_ROOM", "LOC_MILFORD_TOWN"],
            ambient_states={
                "transience": AmbientVector(value=0.8, volatility=0.3),
                "routine": AmbientVector(value=0.7, volatility=0.2),
            },
        ),
        "LOC_REFRESHMENT_ROOM": Location(
            id="LOC_REFRESHMENT_ROOM",
            name="Station Refreshment Room",
            connected_locations=["LOC_RAILWAY_STATION"],
            ambient_states={
                "intimacy": AmbientVector(value=0.6, volatility=0.5),
                "public_exposure": AmbientVector(value=0.7, volatility=0.3),
            },
        ),
        "LOC_MILFORD_TOWN": Location(
            id="LOC_MILFORD_TOWN",
            name="Milford Town (Cinema, Shops, Chemist)",
            connected_locations=["LOC_RAILWAY_STATION"],
            ambient_states={},
        ),
        "LOC_LAURA_HOME": Location(
            id="LOC_LAURA_HOME",
            name="Laura's Home",
            connected_locations=["LOC_RAILWAY_STATION"],
            ambient_states={
                "domesticity": AmbientVector(value=0.8, volatility=0.1),
                "emotional_distance": AmbientVector(value=0.6, volatility=0.4),
            },
        ),
        "LOC_STEPHEN_FLAT": Location(
            id="LOC_STEPHEN_FLAT",
            name="Stephen's Flat",
            connected_locations=["LOC_MILFORD_TOWN"],
            ambient_states={
                "guilt": AmbientVector(value=0.9, volatility=0.3),
            },
        ),
        "LOC_COUNTRYSIDE": Location(
            id="LOC_COUNTRYSIDE",
            name="Countryside (Public Meeting Spots)",
            connected_locations=["LOC_MILFORD_TOWN"],
            ambient_states={},
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
                Belief(target_id="ENT_ALEC", perceived_state="I love him but our relationship is unworkable", confidence=0.95, inertia=0.7),
                Belief(target_id="ENT_FRED", perceived_state="I am betraying my good husband and family", confidence=0.95, inertia=0.7),
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
                Belief(target_id="ENT_LAURA", perceived_state="I love her but must end it for my family's sake", confidence=0.9, inertia=0.6),
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
                Belief(target_id="ENT_LAURA", perceived_state="Something has been troubling her — but I may not know the full truth", confidence=0.6, inertia=0.5),
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
                Belief(target_id="ENT_LAURA", perceived_state="Laura and this man at the table are just casual acquaintances", confidence=0.95, inertia=0.9),
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
                Belief(target_id="ENT_ALEC", perceived_state="Alec is being unfaithful and using my flat for it", confidence=0.9, inertia=0.7),
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
        CausalEdge(source_id="OBJ_GRIT", target_id="EVT_GRIT_IN_EYE", mechanism="physical"),
        CausalEdge(source_id="EVT_GRIT_IN_EYE", target_id="EVT_WEEKLY_MEETINGS", mechanism="social"),
        CausalEdge(source_id="EVT_WEEKLY_MEETINGS", target_id="EVT_LOVE_ADMITTED", mechanism="psychological"),
        CausalEdge(source_id="EVT_LOVE_ADMITTED", target_id="EVT_FRIENDS_SPOTTED", mechanism="social"),
        CausalEdge(source_id="EVT_LOVE_ADMITTED", target_id="EVT_STEPHEN_FLAT_ATTEMPT", mechanism="psychological"),
        CausalEdge(source_id="ENT_STEPHEN", target_id="EVT_STEPHEN_FLAT_ATTEMPT", mechanism="social"),
        CausalEdge(source_id="EVT_STEPHEN_FLAT_ATTEMPT", target_id="EVT_LAURA_WANDERS", mechanism="psychological"),
        CausalEdge(source_id="EVT_LAURA_WANDERS", target_id="EVT_RELATIONSHIP_UNWORKABLE", mechanism="psychological"),
        CausalEdge(source_id="EVT_RELATIONSHIP_UNWORKABLE", target_id="EVT_ALEC_TAKES_JOB", mechanism="psychological"),
        CausalEdge(source_id="EVT_ALEC_TAKES_JOB", target_id="EVT_FINAL_MEETING", mechanism="social"),
        CausalEdge(source_id="ENT_DOLLY", target_id="EVT_FINAL_MEETING", mechanism="social"),
        CausalEdge(source_id="EVT_FINAL_MEETING", target_id="EVT_ALEC_DEPARTS", mechanism="social"),
        CausalEdge(source_id="EVT_ALEC_DEPARTS", target_id="EVT_SUICIDE_ATTEMPT", mechanism="psychological"),
        CausalEdge(source_id="OBJ_EXPRESS_TRAIN", target_id="EVT_SUICIDE_ATTEMPT", mechanism="physical"),
        CausalEdge(source_id="EVT_SUICIDE_ATTEMPT", target_id="EVT_LAURA_RETURNS_HOME", mechanism="psychological"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_LAURA", target_entity_id="ENT_ALEC", affinity=0.9, friction=0.7, power_dynamic=0.0, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_ALEC", target_entity_id="ENT_LAURA", affinity=0.85, friction=0.7, power_dynamic=0.0, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_LAURA", target_entity_id="ENT_FRED", affinity=0.6, friction=0.2, power_dynamic=-0.1, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_FRED", target_entity_id="ENT_LAURA", affinity=0.7, friction=0.2, power_dynamic=0.1, inertia=0.85),
        RelationshipEdge(source_entity_id="ENT_LAURA", target_entity_id="ENT_DOLLY", affinity=0.2, friction=0.5, power_dynamic=0.0, inertia=0.4),
    ],
)
