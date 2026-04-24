from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, RelationshipEdge, TraitVector,
    Affordance, Belief,
)

# =============================================================================
# DEATH ON THE NILE — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
                "LOC_ASWAN": Location(
            name="Aswan",
            description="Aswan",
            ambient_state={"heat": {"value": 0.8, "volatility": 0.1}},
        ),
                "LOC_KARNAK_STEAMER": Location(
            name="Steamer Karnak",
            description="Steamer Karnak",
            ambient_state={"tension": {"value": 0.7, "volatility": 0.5}, "confinement": {"value": 0.8, "volatility": 0.1}},
        ),
                "LOC_KARNAK_LOUNGE": Location(
            name="Steamer Karnak — Lounge",
            description="Steamer Karnak — Lounge",
            ambient_state={"social": {"value": 0.7, "volatility": 0.4}},
        ),
                "LOC_KARNAK_CABINS": Location(
            name="Steamer Karnak — Passenger Cabins",
            description="Steamer Karnak — Passenger Cabins",
            ambient_state={"privacy": {"value": 0.6, "volatility": 0.5}},
        ),
                "LOC_ABU_SIMBEL": Location(
            name="Abu Simbel",
            description="Abu Simbel",
            ambient_state={"danger": {"value": 0.6, "volatility": 0.7}},
        ),
                "LOC_WADI_HALFA": Location(
            name="Wadi Halfa",
            description="Wadi Halfa",
            ambient_state={},
        ),
                "LOC_SHELLAL": Location(
            name="Shellal",
            description="Shellal",
            ambient_state={},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_PISTOL": NarrativeObject(
            id="OBJ_PISTOL",
            name="Jacqueline's Pistol",
            location_id=None,
            owner_id=None,
            properties={"state": "recovered_from_nile", "shots_fired": "3"},
            affordances=[
                Affordance(action="shoot", target_type="Entity"),
                Affordance(action="frame", target_type="Entity"),
            ],
        ),
        "OBJ_SECOND_PISTOL": NarrativeObject(
            id="OBJ_SECOND_PISTOL",
            name="Jacqueline's Second Pistol",
            location_id=None,
            owner_id="ENT_JACQUELINE",
            properties={"state": "concealed"},
            affordances=[
                Affordance(action="shoot", target_type="Entity"),
            ],
        ),
        "OBJ_PEARL_NECKLACE": NarrativeObject(
            id="OBJ_PEARL_NECKLACE",
            name="Linnet's Genuine Pearl Necklace",
            location_id=None,
            owner_id="ENT_TIM",
            properties={"state": "stolen", "value": "extremely_high"},
            affordances=[
                Affordance(action="sell", target_type="Entity"),
            ],
        ),
        "OBJ_IMITATION_PEARLS": NarrativeObject(
            id="OBJ_IMITATION_PEARLS",
            name="Imitation Pearl Necklace",
            location_id="LOC_KARNAK_CABINS",
            owner_id=None,
            properties={"state": "substituted"},
            affordances=[
                Affordance(action="deceive", target_type="Entity"),
            ],
        ),
        "OBJ_VELVET_STOLE": NarrativeObject(
            id="OBJ_VELVET_STOLE",
            name="Miss Van Schuyler's Velvet Stole",
            location_id=None,
            owner_id=None,
            properties={"state": "used_as_silencer"},
            affordances=[
                Affordance(action="muffle", target_type="NarrativeObject"),
            ],
        ),
        "OBJ_NAIL_POLISH_BOTTLE": NarrativeObject(
            id="OBJ_NAIL_POLISH_BOTTLE",
            name="Nail Polish Bottle (Red Ink)",
            location_id="LOC_KARNAK_CABINS",
            owner_id=None,
            properties={"content": "red_ink", "state": "planted"},
            affordances=[
                Affordance(action="simulate_blood", target_type="Entity"),
            ],
        ),
        "OBJ_PENNINGTON_REVOLVER": NarrativeObject(
            id="OBJ_PENNINGTON_REVOLVER",
            name="Pennington's Revolver",
            location_id=None,
            owner_id="ENT_PENNINGTON",
            properties={"state": "used_in_murder"},
            affordances=[
                Affordance(action="shoot", target_type="Entity"),
            ],
        ),
        "OBJ_BOULDER": NarrativeObject(
            id="OBJ_BOULDER",
            name="Boulder at Abu Simbel",
            location_id="LOC_ABU_SIMBEL",
            owner_id=None,
            properties={"state": "fallen"},
            affordances=[
                Affordance(action="crush", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_POIROT": Entity(
            id="ENT_POIROT",
            name="Hercule Poirot",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "intellect": TraitVector(value=0.98, inertia=0.95),
                "observation": TraitVector(value=0.95, inertia=0.9),
                "compassion": TraitVector(value=0.7, inertia=0.7),
                "moral_flexibility": TraitVector(value=0.4, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_JACQUELINE", perceived_state="She had a second pistol — I chose to let her end it", confidence=1.0, inertia=0.9),
            ],
        ),
        "ENT_LINNET": Entity(
            id="ENT_LINNET",
            name="Linnet Doyle (née Ridgeway)",
            location_id="LOC_KARNAK_CABINS",
            status="dead",
            traits={
                "wealth": TraitVector(value=0.95, inertia=0.9),
                "entitlement": TraitVector(value=0.8, inertia=0.7),
                "charm": TraitVector(value=0.75, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_SIMON", perceived_state="Simon genuinely loves me and married me for love", confidence=0.85, inertia=0.7),
                Belief(target_id="ENT_JACQUELINE", perceived_state="Jacqueline is a jealous ex who cannot accept losing Simon", confidence=0.9, inertia=0.7),
            ],
        ),
        "ENT_SIMON": Entity(
            id="ENT_SIMON",
            name="Simon Doyle",
            location_id="LOC_KARNAK_STEAMER",
            status="dead",
            traits={
                "cunning": TraitVector(value=0.5, inertia=0.4),
                "greed": TraitVector(value=0.8, inertia=0.6),
                "devotion_to_jacqueline": TraitVector(value=0.85, inertia=0.8),
                "impulsiveness": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_LINNET", perceived_state="Linnet is a means to wealth", confidence=0.9, inertia=0.7),
            ],
        ),
        "ENT_JACQUELINE": Entity(
            id="ENT_JACQUELINE",
            name="Jacqueline de Bellefort",
            location_id="LOC_KARNAK_STEAMER",
            status="dead",
            traits={
                "jealousy": TraitVector(value=0.9, inertia=0.7),
                "cunning": TraitVector(value=0.85, inertia=0.7),
                "devotion_to_simon": TraitVector(value=0.95, inertia=0.9),
                "desperation": TraitVector(value=0.9, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_SIMON", perceived_state="Simon and I are still lovers executing our plan — he married Linnet for her money", confidence=0.95, inertia=0.9),
            ],
        ),
        "ENT_RACE": Entity(
            id="ENT_RACE",
            name="Colonel Race",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "discipline": TraitVector(value=0.85, inertia=0.8),
                "observation": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_LOUISE": Entity(
            id="ENT_LOUISE",
            name="Louise Bourget",
            location_id="LOC_KARNAK_CABINS",
            status="dead",
            traits={
                "greed": TraitVector(value=0.7, inertia=0.5),
                "opportunism": TraitVector(value=0.75, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_SIMON", perceived_state="Simon killed Linnet — I can profit from this knowledge through blackmail", confidence=0.9, inertia=0.5),
            ],
        ),
        "ENT_PENNINGTON": Entity(
            id="ENT_PENNINGTON",
            name="Andrew Pennington",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "greed": TraitVector(value=0.8, inertia=0.6),
                "deception": TraitVector(value=0.7, inertia=0.5),
                "desperation": TraitVector(value=0.75, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_LINNET", perceived_state="I can trick Linnet into signing documents to cover my speculation losses", confidence=0.7, inertia=0.5),
            ],
        ),
        "ENT_TIM": Entity(
            id="ENT_TIM",
            name="Tim Allerton",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "cunning": TraitVector(value=0.7, inertia=0.5),
                "greed": TraitVector(value=0.6, inertia=0.4),
            },
            constants=["professional_thief"],
        ),
        "ENT_MRS_OTTERBOURNE": Entity(
            id="ENT_MRS_OTTERBOURNE",
            name="Salome Otterbourne",
            location_id="LOC_KARNAK_STEAMER",
            status="dead",
            traits={
                "curiosity": TraitVector(value=0.7, inertia=0.5),
            },
        ),
        "ENT_VAN_SCHUYLER": Entity(
            id="ENT_VAN_SCHUYLER",
            name="Marie Van Schuyler",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "entitlement": TraitVector(value=0.8, inertia=0.7),
            },
            constants=["kleptomaniac"],
        ),
        "ENT_CORNELIA": Entity(
            id="ENT_CORNELIA",
            name="Cornelia Robson",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "kindness": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="EVT_JACQUELINE_SHOOTS_SIMON", perceived_state="Jacqueline genuinely shot Simon in a jealous rage", confidence=0.95, inertia=0.8),
            ],
        ),
        "ENT_FANTHORP": Entity(
            id="ENT_FANTHORP",
            name="Jim Fanthorp",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "diligence": TraitVector(value=0.7, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="EVT_JACQUELINE_SHOOTS_SIMON", perceived_state="Jacqueline genuinely shot Simon in a jealous rage", confidence=0.95, inertia=0.8),
            ],
        ),
        "ENT_RICHETTI": Entity(
            id="ENT_RICHETTI",
            name="Guido Richetti",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "deception": TraitVector(value=0.7, inertia=0.6),
            },
            constants=["political_agitator"],
        ),
        "ENT_BESSNER": Entity(
            id="ENT_BESSNER",
            name="Dr Bessner",
            location_id="LOC_KARNAK_STEAMER",
            status="healthy",
            traits={
                "competence": TraitVector(value=0.8, inertia=0.7),
            },
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_POIROT_MEETS_LINNET", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_id="ENT_LINNET", description="Linnet approaches Poirot in Aswan, asking him to stop Jacqueline's stalking. Poirot refuses."),
        EventNode(id="EVT_POIROT_WARNS_JACQUELINE", fabula_time=2, syuzhet_index=2, event_type="choice", actor_id="ENT_POIROT", description="Poirot privately warns Jacqueline not to open her heart to evil."),
        EventNode(id="EVT_BOARDING_KARNAK", fabula_time=3, syuzhet_index=3, event_type="outcome", actor_id=None, description="Simon and Linnet board the Karnak to escape Jacqueline, but she anticipated them and boarded first."),
        EventNode(id="EVT_BOULDER_ATTACK", fabula_time=4, syuzhet_index=4, event_type="choice", actor_id="ENT_PENNINGTON", description="At Abu Simbel, Pennington pushes a boulder off a cliff, nearly crushing Linnet."),
        EventNode(id="EVT_RACE_BOARDS", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_id="ENT_RACE", description="Colonel Race boards at Wadi Halfa, informing Poirot of a political agitator among passengers."),
        EventNode(id="EVT_JACQUELINE_SHOOTS_SIMON", fabula_time=6, syuzhet_index=6, event_type="choice", actor_id="ENT_JACQUELINE", description="In the lounge, Jacqueline pretends to drunkenly shoot Simon. She deliberately misses; Simon fakes a leg injury with red ink."),
        EventNode(id="EVT_SIMON_MURDERS_LINNET", fabula_time=7, syuzhet_index=7, event_type="choice", actor_id="ENT_SIMON", description="While Jacqueline distracts Fanthorp and Cornelia, Simon takes the pistol, goes to Linnet's cabin, and shoots her dead."),
        EventNode(id="EVT_SIMON_SHOOTS_SELF", fabula_time=8, syuzhet_index=8, event_type="choice", actor_id="ENT_SIMON", description="Simon returns to the lounge, shoots himself in the leg for real, wraps the pistol in the velvet stole, and throws it overboard."),
        EventNode(id="EVT_PEARLS_STOLEN", fabula_time=9, syuzhet_index=9, event_type="choice", actor_id="ENT_TIM", description="Tim Allerton steals Linnet's genuine pearls and substitutes an imitation string."),
        EventNode(id="EVT_LINNET_FOUND_DEAD", fabula_time=10, syuzhet_index=10, event_type="revelation", actor_id=None, description="Linnet is found dead with a bullet in her brain; her pearl necklace is missing."),
        EventNode(id="EVT_LOUISE_BLACKMAIL", fabula_time=11, syuzhet_index=11, event_type="choice", actor_id="ENT_LOUISE", description="Louise hints to Simon that she witnessed him entering Linnet's cabin, planning blackmail."),
        EventNode(id="EVT_LOUISE_MURDERED", fabula_time=12, syuzhet_index=12, event_type="choice", actor_id="ENT_JACQUELINE", description="Jacqueline stabs Louise to death to protect Simon from blackmail."),
        EventNode(id="EVT_MRS_OTTERBOURNE_MURDERED", fabula_time=13, syuzhet_index=13, event_type="choice", actor_id="ENT_JACQUELINE", description="Mrs Otterbourne sees Jacqueline entering Louise's cabin. When she tries to tell Poirot, Simon alerts Jacqueline, who shoots her dead."),
        EventNode(id="EVT_POIROT_CONFRONTS_PENNINGTON", fabula_time=14, syuzhet_index=14, event_type="revelation", actor_id="ENT_POIROT", description="Poirot confronts Pennington about the boulder attack and his speculation with Linnet's inheritance."),
        EventNode(id="EVT_TIM_EXPOSED", fabula_time=15, syuzhet_index=15, event_type="revelation", actor_id="ENT_POIROT", description="Poirot recovers the genuine pearls from Tim and exposes him as a professional thief."),
        EventNode(id="EVT_RICHETTI_IDENTIFIED", fabula_time=16, syuzhet_index=16, event_type="revelation", actor_id="ENT_RACE", description="Race identifies Richetti as the political agitator he was looking for."),
        EventNode(id="EVT_POIROT_REVEALS_TRUTH", fabula_time=17, syuzhet_index=17, event_type="revelation", actor_id="ENT_POIROT", description="Poirot reveals that Simon killed Linnet; Jacqueline planned the entire murder scheme, and the pair are still lovers."),
        EventNode(id="EVT_SIMON_CONFESSES", fabula_time=18, syuzhet_index=18, event_type="outcome", actor_id="ENT_SIMON", description="Simon confesses. He and Jacqueline are arrested."),
        EventNode(id="EVT_MURDER_SUICIDE", fabula_time=19, syuzhet_index=19, event_type="choice", actor_id="ENT_JACQUELINE", description="As the steamer arrives at Shellal, Jacqueline shoots Simon and herself with a second pistol to escape the gallows."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="ENT_LINNET", target_id="EVT_POIROT_MEETS_LINNET", mechanism="social"),
        CausalEdge(source_id="ENT_JACQUELINE", target_id="EVT_POIROT_MEETS_LINNET", mechanism="psychological"),
        CausalEdge(source_id="EVT_POIROT_MEETS_LINNET", target_id="EVT_BOARDING_KARNAK", mechanism="social"),
        CausalEdge(source_id="EVT_BOARDING_KARNAK", target_id="EVT_BOULDER_ATTACK", mechanism="social"),
        CausalEdge(source_id="EVT_POIROT_MEETS_LINNET", target_id="EVT_POIROT_WARNS_JACQUELINE", mechanism="psychological"),
        CausalEdge(source_id="ENT_PENNINGTON", target_id="EVT_BOULDER_ATTACK", mechanism="physical"),
        CausalEdge(source_id="EVT_BOARDING_KARNAK", target_id="EVT_RACE_BOARDS", mechanism="social"),
        CausalEdge(source_id="ENT_JACQUELINE", target_id="EVT_JACQUELINE_SHOOTS_SIMON", mechanism="psychological"),
        CausalEdge(source_id="EVT_JACQUELINE_SHOOTS_SIMON", target_id="EVT_SIMON_MURDERS_LINNET", mechanism="physical"),
        CausalEdge(source_id="OBJ_PISTOL", target_id="EVT_SIMON_MURDERS_LINNET", mechanism="physical"),
        CausalEdge(source_id="OBJ_NAIL_POLISH_BOTTLE", target_id="EVT_JACQUELINE_SHOOTS_SIMON", mechanism="physical"),
        CausalEdge(source_id="EVT_SIMON_MURDERS_LINNET", target_id="EVT_SIMON_SHOOTS_SELF", mechanism="physical"),
        CausalEdge(source_id="OBJ_VELVET_STOLE", target_id="EVT_SIMON_SHOOTS_SELF", mechanism="physical"),
        CausalEdge(source_id="EVT_SIMON_MURDERS_LINNET", target_id="EVT_PEARLS_STOLEN", mechanism="social"),
        CausalEdge(source_id="ENT_TIM", target_id="EVT_PEARLS_STOLEN", mechanism="physical"),
        CausalEdge(source_id="EVT_SIMON_MURDERS_LINNET", target_id="EVT_LINNET_FOUND_DEAD", mechanism="physical"),
        CausalEdge(source_id="EVT_SIMON_MURDERS_LINNET", target_id="EVT_LOUISE_BLACKMAIL", mechanism="epistemic"),
        CausalEdge(source_id="ENT_LOUISE", target_id="EVT_LOUISE_BLACKMAIL", mechanism="psychological"),
        CausalEdge(source_id="EVT_LOUISE_BLACKMAIL", target_id="EVT_LOUISE_MURDERED", mechanism="psychological"),
        CausalEdge(source_id="ENT_JACQUELINE", target_id="EVT_LOUISE_MURDERED", mechanism="physical"),
        CausalEdge(source_id="EVT_LOUISE_MURDERED", target_id="EVT_MRS_OTTERBOURNE_MURDERED", mechanism="epistemic"),
        CausalEdge(source_id="ENT_SIMON", target_id="EVT_MRS_OTTERBOURNE_MURDERED", mechanism="social"),
        CausalEdge(source_id="ENT_JACQUELINE", target_id="EVT_MRS_OTTERBOURNE_MURDERED", mechanism="physical"),
        CausalEdge(source_id="OBJ_PENNINGTON_REVOLVER", target_id="EVT_MRS_OTTERBOURNE_MURDERED", mechanism="physical"),
        CausalEdge(source_id="EVT_POIROT_REVEALS_TRUTH", target_id="EVT_SIMON_CONFESSES", mechanism="epistemic"),
        CausalEdge(source_id="EVT_BOULDER_ATTACK", target_id="EVT_POIROT_CONFRONTS_PENNINGTON", mechanism="epistemic"),
        CausalEdge(source_id="EVT_PEARLS_STOLEN", target_id="EVT_TIM_EXPOSED", mechanism="epistemic"),
        CausalEdge(source_id="ENT_RACE", target_id="EVT_RICHETTI_IDENTIFIED", mechanism="epistemic"),
        CausalEdge(source_id="EVT_SIMON_CONFESSES", target_id="EVT_MURDER_SUICIDE", mechanism="psychological"),
        CausalEdge(source_id="OBJ_SECOND_PISTOL", target_id="EVT_MURDER_SUICIDE", mechanism="physical"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_ABU_SIMBEL", target_id="LOC_KARNAK_STEAMER"),
        SpatialEdge(source_id="LOC_ASWAN", target_id="LOC_KARNAK_STEAMER"),
        SpatialEdge(source_id="LOC_KARNAK_CABINS", target_id="LOC_KARNAK_LOUNGE"),
        SpatialEdge(source_id="LOC_KARNAK_CABINS", target_id="LOC_KARNAK_STEAMER"),
        SpatialEdge(source_id="LOC_KARNAK_LOUNGE", target_id="LOC_KARNAK_STEAMER"),
        SpatialEdge(source_id="LOC_KARNAK_STEAMER", target_id="LOC_SHELLAL"),
        SpatialEdge(source_id="LOC_KARNAK_STEAMER", target_id="LOC_WADI_HALFA"),
    ],
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_SIMON", target_entity_id="ENT_JACQUELINE", affinity=0.9, friction=0.6, power_dynamic=-0.2, inertia=0.85),
        RelationshipEdge(source_entity_id="ENT_JACQUELINE", target_entity_id="ENT_SIMON", affinity=0.95, friction=0.7, power_dynamic=0.3, inertia=0.9),
        RelationshipEdge(source_entity_id="ENT_SIMON", target_entity_id="ENT_LINNET", affinity=-0.3, friction=0.5, power_dynamic=-0.4, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_JACQUELINE", target_entity_id="ENT_LINNET", affinity=-0.9, friction=0.95, power_dynamic=-0.3, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_LINNET", target_entity_id="ENT_SIMON", affinity=0.7, friction=0.3, power_dynamic=0.5, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_PENNINGTON", target_entity_id="ENT_LINNET", affinity=-0.4, friction=0.7, power_dynamic=-0.3, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_LOUISE", target_entity_id="ENT_SIMON", affinity=-0.2, friction=0.6, power_dynamic=-0.5, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_POIROT", target_entity_id="ENT_RACE", affinity=0.7, friction=0.1, power_dynamic=0.1, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_CORNELIA", target_entity_id="ENT_BESSNER", affinity=0.5, friction=0.1, power_dynamic=-0.2, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_VAN_SCHUYLER", target_entity_id="ENT_CORNELIA", affinity=0.3, friction=0.4, power_dynamic=0.6, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_POIROT", target_entity_id="ENT_JACQUELINE", affinity=0.2, friction=0.5, power_dynamic=0.3, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_TIM", target_entity_id="ENT_LINNET", affinity=-0.2, friction=0.4, power_dynamic=-0.3, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_FANTHORP", target_entity_id="ENT_LINNET", affinity=0.3, friction=0.2, power_dynamic=-0.2, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_MRS_OTTERBOURNE", target_entity_id="ENT_POIROT", affinity=0.3, friction=0.3, power_dynamic=-0.2, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_RICHETTI", target_entity_id="ENT_RACE", affinity=-0.5, friction=0.7, power_dynamic=-0.3, inertia=0.4),
    ],
)
