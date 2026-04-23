from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
    Affordance, Belief,
)

# =============================================================================
# DAD'S ARMY — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
        "LOC_WALMINGTON": Location(
            id="LOC_WALMINGTON",
            name="Walmington-on-Sea",
            connected_locations=["LOC_CHURCH_HALL", "LOC_BANK", "LOC_CLIFFS"],
            ambient_states={
                "wartime_anxiety": AmbientVector(value=0.6, volatility=0.4),
                "community": AmbientVector(value=0.7, volatility=0.2),
            },
        ),
        "LOC_BANK": Location(
            id="LOC_BANK",
            name="Martins Bank (Walmington Branch)",
            connected_locations=["LOC_WALMINGTON"],
            ambient_states={
                "routine": AmbientVector(value=0.8, volatility=0.1),
            },
        ),
        "LOC_CHURCH_HALL": Location(
            id="LOC_CHURCH_HALL",
            name="Church Hall (Platoon HQ)",
            connected_locations=["LOC_WALMINGTON", "LOC_CHURCH_CRYPT"],
            ambient_states={
                "community": AmbientVector(value=0.7, volatility=0.3),
            },
        ),
        "LOC_CHURCH_CRYPT": Location(
            id="LOC_CHURCH_CRYPT",
            name="Church Crypt",
            connected_locations=["LOC_CHURCH_HALL"],
            ambient_states={
                "concealment": AmbientVector(value=0.8, volatility=0.2),
            },
        ),
        "LOC_TRAINING_GROUNDS": Location(
            id="LOC_TRAINING_GROUNDS",
            name="War Games Training Grounds",
            connected_locations=["LOC_WALMINGTON"],
            ambient_states={
                "chaos": AmbientVector(value=0.7, volatility=0.5),
            },
        ),
        "LOC_CLIFFS": Location(
            id="LOC_CLIFFS",
            name="White Cliffs (overlooking Channel)",
            connected_locations=["LOC_WALMINGTON"],
            ambient_states={
                "defiance": AmbientVector(value=0.8, volatility=0.2),
            },
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_JONES_VAN": NarrativeObject(
            id="OBJ_JONES_VAN",
            name="Jones's Van (Gas-Converted)",
            location_id="LOC_TRAINING_GROUNDS",
            owner_id="ENT_JONES",
            properties={"state": "gas_bag_punctured"},
            affordances=[
                Affordance(action="transport", target_type="Entity"),
            ],
        ),
        "OBJ_MAINWARING_REVOLVER": NarrativeObject(
            id="OBJ_MAINWARING_REVOLVER",
            name="Mainwaring's Revolver",
            location_id=None,
            owner_id="ENT_MAINWARING",
            properties={"state": "empty"},
            affordances=[
                Affordance(action="threaten", target_type="Entity"),
            ],
        ),
        "OBJ_COLLECTION_PLATE": NarrativeObject(
            id="OBJ_COLLECTION_PLATE",
            name="Collection Plate (concealing revolver)",
            location_id="LOC_CHURCH_HALL",
            owner_id=None,
            properties={"state": "used_as_concealment"},
            affordances=[
                Affordance(action="conceal", target_type="NarrativeObject"),
            ],
        ),
        "OBJ_UNION_FLAG": NarrativeObject(
            id="OBJ_UNION_FLAG",
            name="Union Flag",
            location_id="LOC_CLIFFS",
            owner_id=None,
            properties={"state": "waving"},
            affordances=[
                Affordance(action="symbolize_defiance", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_MAINWARING": Entity(
            id="ENT_MAINWARING",
            name="Captain George Mainwaring",
            location_id="LOC_CLIFFS",
            status="healthy",
            traits={
                "pomposity": TraitVector(value=0.9, inertia=0.85),
                "courage": TraitVector(value=0.8, inertia=0.7),
                "patriotism": TraitVector(value=0.95, inertia=0.9),
                "stubbornness": TraitVector(value=0.85, inertia=0.8),
                "leadership": TraitVector(value=0.7, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_FULLARD", perceived_state="We must face the enemy regardless — even with an empty gun", confidence=0.95, inertia=0.9),
            ],
        ),
        "ENT_WILSON": Entity(
            id="ENT_WILSON",
            name="Sergeant Arthur Wilson",
            location_id="LOC_CLIFFS",
            status="healthy",
            traits={
                "diffidence": TraitVector(value=0.85, inertia=0.7),
                "charm": TraitVector(value=0.7, inertia=0.6),
                "competence": TraitVector(value=0.7, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING", perceived_state="Both guns were empty — but his bluff was magnificent", confidence=1.0, inertia=0.8),
            ],
        ),
        "ENT_JONES": Entity(
            id="ENT_JONES",
            name="Lance-Corporal Jack Jones",
            location_id="LOC_CLIFFS",
            status="healthy",
            traits={
                "enthusiasm": TraitVector(value=0.9, inertia=0.8),
                "clumsiness": TraitVector(value=0.8, inertia=0.7),
                "loyalty": TraitVector(value=0.9, inertia=0.85),
            },
        ),
        "ENT_FRAZER": Entity(
            id="ENT_FRAZER",
            name="Private Frazer",
            location_id="LOC_CLIFFS",
            status="healthy",
            traits={
                "pessimism": TraitVector(value=0.85, inertia=0.8),
                "cunning": TraitVector(value=0.7, inertia=0.6),
            },
        ),
        "ENT_GODFREY": Entity(
            id="ENT_GODFREY",
            name="Private Godfrey",
            location_id="LOC_CLIFFS",
            status="healthy",
            traits={
                "gentleness": TraitVector(value=0.9, inertia=0.85),
                "frailty": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_PIKE": Entity(
            id="ENT_PIKE",
            name="Private Pike",
            location_id="LOC_CLIFFS",
            status="healthy",
            traits={
                "naivete": TraitVector(value=0.85, inertia=0.7),
                "youth": TraitVector(value=0.9, inertia=0.8),
            },
        ),
        "ENT_WALKER": Entity(
            id="ENT_WALKER",
            name="Private Joe Walker",
            location_id="LOC_CLIFFS",
            status="healthy",
            traits={
                "resourcefulness": TraitVector(value=0.85, inertia=0.7),
                "black_market": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_FULLARD": Entity(
            id="ENT_FULLARD",
            name="Major-General Fullard",
            location_id="LOC_CHURCH_HALL",
            status="healthy",
            traits={
                "authority": TraitVector(value=0.85, inertia=0.8),
                "irritability": TraitVector(value=0.8, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING", perceived_state="Mainwaring is a bumbling bank clerk playing at soldiers", confidence=0.85, inertia=0.6),
            ],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_EDEN_BROADCAST", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_id=None, description="Anthony Eden broadcasts a call for Local Defence Volunteers. Mainwaring and Wilson hear it at the bank."),
        EventNode(id="EVT_MAINWARING_TAKES_CHARGE", fabula_time=2, syuzhet_index=2, event_type="choice", actor_id="ENT_MAINWARING", description="Mainwaring commandeers the church hall and organises enrolment of the LDV platoon."),
        EventNode(id="EVT_PLATOON_FORMED", fabula_time=3, syuzhet_index=3, event_type="outcome", actor_id=None, description="The platoon is formed: Mainwaring as captain, Wilson as sergeant, Jones as lance-corporal, plus Frazer, Godfrey, Pike and Walker."),
        EventNode(id="EVT_IMPROVISED_WEAPONS", fabula_time=4, syuzhet_index=4, event_type="outcome", actor_id="ENT_JONES", description="With no weapons, the platoon uses Jones's improvised devices — a rocket launcher destroys a barn, a bathtub tank rolls into the river."),
        EventNode(id="EVT_UNIFORMS_AND_WEAPONS", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_id=None, description="The platoon secures uniforms and weapons. The LDV is renamed the Home Guard."),
        EventNode(id="EVT_WAR_GAMES_DISASTER", fabula_time=6, syuzhet_index=6, event_type="outcome", actor_id="ENT_JONES", description="At the training weekend, Jones's gas-converted van breaks down. A towed steam roller destroys tents and equipment, angering Major-General Fullard."),
        EventNode(id="EVT_BRIDGE_CHAOS", fabula_time=7, syuzhet_index=7, event_type="outcome", actor_id=None, description="The platoon oversleeps, misses breakfast, and the pontoon bridge exercise goes chaotically wrong with Jones adrift on a white horse."),
        EventNode(id="EVT_FULLARD_THREATENS", fabula_time=8, syuzhet_index=8, event_type="choice", actor_id="ENT_FULLARD", description="Fullard tells Mainwaring he will recommend his replacement due to the platoon's poor showing."),
        EventNode(id="EVT_LUFTWAFFE_CRASH", fabula_time=9, syuzhet_index=9, event_type="outcome", actor_id=None, description="A Luftwaffe reconnaissance aircraft is shot down. Its three-man crew parachutes to safety near Walmington."),
        EventNode(id="EVT_HOSTAGE_SITUATION", fabula_time=10, syuzhet_index=10, event_type="outcome", actor_id=None, description="The German crew enters the church hall during a Spitfire fundraiser, taking hostages including the mayor and vicar, and demanding a boat to France."),
        EventNode(id="EVT_PLATOON_INFILTRATES", fabula_time=11, syuzhet_index=11, event_type="choice", actor_id="ENT_MAINWARING", description="The platoon infiltrates through the crypt, enters in choir robes singing hymns, with rifles hidden beneath."),
        EventNode(id="EVT_GERMAN_STANDOFF", fabula_time=12, syuzhet_index=12, event_type="choice", actor_id="ENT_MAINWARING", description="Mainwaring confronts the Luftwaffe leader with a concealed revolver. Both agree to shoot at the count of three. The platoon draws their rifles."),
        EventNode(id="EVT_GERMANS_SURRENDER", fabula_time=13, syuzhet_index=13, event_type="outcome", actor_id=None, description="The Germans reluctantly surrender. Fullard is stunned. Wilson reveals the German's gun was empty — and so was Mainwaring's."),
        EventNode(id="EVT_HEROES_OF_TOWN", fabula_time=14, syuzhet_index=14, event_type="outcome", actor_id=None, description="Mainwaring and the Home Guard become the pride of the town. They stand on the cliffs looking toward France."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="EVT_EDEN_BROADCAST", target_id="EVT_MAINWARING_TAKES_CHARGE", mechanism="social"),
        CausalEdge(source_id="ENT_MAINWARING", target_id="EVT_MAINWARING_TAKES_CHARGE", mechanism="psychological"),
        CausalEdge(source_id="EVT_MAINWARING_TAKES_CHARGE", target_id="EVT_PLATOON_FORMED", mechanism="social"),
        CausalEdge(source_id="EVT_PLATOON_FORMED", target_id="EVT_IMPROVISED_WEAPONS", mechanism="physical"),
        CausalEdge(source_id="EVT_IMPROVISED_WEAPONS", target_id="EVT_UNIFORMS_AND_WEAPONS", mechanism="social"),
        CausalEdge(source_id="EVT_UNIFORMS_AND_WEAPONS", target_id="EVT_WAR_GAMES_DISASTER", mechanism="social"),
        CausalEdge(source_id="OBJ_JONES_VAN", target_id="EVT_WAR_GAMES_DISASTER", mechanism="physical"),
        CausalEdge(source_id="EVT_WAR_GAMES_DISASTER", target_id="EVT_BRIDGE_CHAOS", mechanism="social"),
        CausalEdge(source_id="EVT_BRIDGE_CHAOS", target_id="EVT_FULLARD_THREATENS", mechanism="social"),
        CausalEdge(source_id="ENT_FULLARD", target_id="EVT_FULLARD_THREATENS", mechanism="social"),
        CausalEdge(source_id="EVT_FULLARD_THREATENS", target_id="EVT_LUFTWAFFE_CRASH", mechanism="social"),
        CausalEdge(source_id="EVT_LUFTWAFFE_CRASH", target_id="EVT_HOSTAGE_SITUATION", mechanism="physical"),
        CausalEdge(source_id="EVT_HOSTAGE_SITUATION", target_id="EVT_PLATOON_INFILTRATES", mechanism="social"),
        CausalEdge(source_id="ENT_MAINWARING", target_id="EVT_PLATOON_INFILTRATES", mechanism="psychological"),
        CausalEdge(source_id="EVT_PLATOON_INFILTRATES", target_id="EVT_GERMAN_STANDOFF", mechanism="physical"),
        CausalEdge(source_id="OBJ_MAINWARING_REVOLVER", target_id="EVT_GERMAN_STANDOFF", mechanism="physical"),
        CausalEdge(source_id="EVT_GERMAN_STANDOFF", target_id="EVT_GERMANS_SURRENDER", mechanism="social"),
        CausalEdge(source_id="EVT_GERMANS_SURRENDER", target_id="EVT_HEROES_OF_TOWN", mechanism="social"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_MAINWARING", target_entity_id="ENT_WILSON", affinity=0.5, friction=0.5, power_dynamic=0.5, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_WILSON", target_entity_id="ENT_MAINWARING", affinity=0.5, friction=0.4, power_dynamic=-0.4, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_MAINWARING", target_entity_id="ENT_JONES", affinity=0.6, friction=0.5, power_dynamic=0.6, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_JONES", target_entity_id="ENT_MAINWARING", affinity=0.8, friction=0.3, power_dynamic=-0.5, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_MAINWARING", target_entity_id="ENT_PIKE", affinity=0.3, friction=0.6, power_dynamic=0.7, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_MAINWARING", target_entity_id="ENT_FULLARD", affinity=-0.3, friction=0.8, power_dynamic=-0.6, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_FULLARD", target_entity_id="ENT_MAINWARING", affinity=-0.4, friction=0.7, power_dynamic=0.7, inertia=0.5),
    ],
)
