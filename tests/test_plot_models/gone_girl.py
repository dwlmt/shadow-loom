from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, InformationEdge, RelationshipEdge, TraitVector,
    Affordance, Belief,
)

# =============================================================================
# GONE GIRL — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
                "LOC_NEW_YORK": Location(
            name="New York City (Brooklyn Brownstone)",
            description="New York City (Brooklyn Brownstone)",
            ambient_state={"happiness": {"value": 0.7, "volatility": 0.6}},
        ),
                "LOC_NORTH_CARTHAGE": Location(
            name="North Carthage, Missouri",
            description="North Carthage, Missouri",
            ambient_state={"stagnation": {"value": 0.7, "volatility": 0.3}, "tension": {"value": 0.8, "volatility": 0.5}},
        ),
                "LOC_DUNNE_HOUSE": Location(
            name="Nick & Amy's House",
            description="Nick & Amy's House",
            ambient_state={"crime_scene": {"value": 0.9, "volatility": 0.3}},
        ),
                "LOC_THE_BAR": Location(
            name="The Bar (Nick & Go's Bar)",
            description="The Bar (Nick & Go's Bar)",
            ambient_state={},
        ),
                "LOC_GO_WOODSHED": Location(
            name="Go's Woodshed",
            description="Go's Woodshed",
            ambient_state={"incrimination": {"value": 0.9, "volatility": 0.2}},
        ),
                "LOC_MOTEL_HIDEOUT": Location(
            name="Amy's Motel Hideout",
            description="Amy's Motel Hideout",
            ambient_state={"isolation": {"value": 0.8, "volatility": 0.5}},
        ),
                "LOC_DESI_LAKE_HOUSE": Location(
            name="Desi Collings' Lake House",
            description="Desi Collings' Lake House",
            ambient_state={"entrapment": {"value": 0.8, "volatility": 0.4}},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_FAKE_DIARY": NarrativeObject(
            id="OBJ_FAKE_DIARY",
            name="Amy's Fabricated Diary",
            location_id=None,
            owner_id=None,
            properties={"state": "planted", "content": "fabricated_abuse_chronicle"},
            affordances=[
                Affordance(action="frame", target_type="Entity"),
                Affordance(action="deceive", target_type="Entity"),
            ],
        ),
        "OBJ_TREASURE_HUNT_CLUES": NarrativeObject(
            id="OBJ_TREASURE_HUNT_CLUES",
            name="Anniversary Treasure Hunt Clues",
            location_id="LOC_NORTH_CARTHAGE",
            owner_id=None,
            properties={"state": "planted_as_evidence"},
            affordances=[
                Affordance(action="incriminate", target_type="Entity"),
                Affordance(action="reveal_plan", target_type="Entity"),
            ],
        ),
        "OBJ_PUNCH_JUDY_PUPPETS": NarrativeObject(
            id="OBJ_PUNCH_JUDY_PUPPETS",
            name="Punch and Judy Puppets",
            location_id="LOC_GO_WOODSHED",
            owner_id=None,
            properties={"state": "missing_handle", "handle_has_blood": "true"},
            affordances=[
                Affordance(action="incriminate", target_type="Entity"),
            ],
        ),
        "OBJ_CREDIT_CARDS": NarrativeObject(
            id="OBJ_CREDIT_CARDS",
            name="Credit Cards (in Nick's Name)",
            location_id="LOC_GO_WOODSHED",
            owner_id="ENT_NICK",
            properties={"state": "maxed_out_by_amy"},
            affordances=[
                Affordance(action="incriminate", target_type="Entity"),
            ],
        ),
        "OBJ_LIFE_INSURANCE": NarrativeObject(
            id="OBJ_LIFE_INSURANCE",
            name="Amy's Life Insurance Policy",
            location_id=None,
            owner_id="ENT_NICK",
            properties={"state": "recently_increased"},
            affordances=[
                Affordance(action="incriminate", target_type="Entity"),
            ],
        ),
        "OBJ_FROZEN_SEMEN": NarrativeObject(
            id="OBJ_FROZEN_SEMEN",
            name="Nick's Frozen Semen (Fertility Clinic)",
            location_id=None,
            owner_id="ENT_AMY",
            properties={"state": "used_for_insemination"},
            affordances=[
                Affordance(action="coerce", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_NICK": Entity(
            id="ENT_NICK",
            name="Nick Dunne",
            location_id="LOC_DUNNE_HOUSE",
            status="healthy",
            traits={
                "charm": TraitVector(value=0.7, inertia=0.5),
                "dishonesty": TraitVector(value=0.6, inertia=0.5),
                "passivity": TraitVector(value=0.7, inertia=0.6),
                "self_preservation": TraitVector(value=0.75, inertia=0.6),
                "resentment": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_AMY", perceived_state="Amy is a sociopath — I know the truth but can't prove it", confidence=1.0, inertia=0.9, established_at_fabula=0),
            ],
        ),
        "ENT_AMY": Entity(
            id="ENT_AMY",
            name="Amy Dunne (née Elliott)",
            location_id="LOC_DUNNE_HOUSE",
            status="healthy",
            traits={
                "intelligence": TraitVector(value=0.95, inertia=0.9),
                "manipulation": TraitVector(value=0.98, inertia=0.95),
                "narcissism": TraitVector(value=0.9, inertia=0.85),
                "vindictiveness": TraitVector(value=0.9, inertia=0.8),
                "perfectionism": TraitVector(value=0.9, inertia=0.85),
            },
            beliefs=[
                Belief(target_id="ENT_NICK", perceived_state="Nick and I uniquely understand each other", confidence=0.8, inertia=0.7, established_at_fabula=0),
                Belief(target_id="ENT_NICK", perceived_state="Nick deserves to be punished with death for wasting my life and betraying me", confidence=0.95, inertia=0.7, established_at_fabula=0),
            ],
        ),
        "ENT_GO": Entity(
            id="ENT_GO",
            name="Margo 'Go' Dunne",
            location_id="LOC_NORTH_CARTHAGE",
            status="healthy",
            traits={
                "loyalty": TraitVector(value=0.9, inertia=0.85),
                "skepticism": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_NICK", perceived_state="Nick is innocent — Amy set him up", confidence=0.95, inertia=0.9, established_at_fabula=0),
                Belief(target_id="ENT_AMY", perceived_state="Amy is a dangerous liar and manipulator", confidence=0.95, inertia=0.9, established_at_fabula=0),
            ],
        ),
        "ENT_BONEY": Entity(
            id="ENT_BONEY",
            name="Detective Rhonda Boney",
            location_id="LOC_NORTH_CARTHAGE",
            status="healthy",
            traits={
                "diligence": TraitVector(value=0.85, inertia=0.8),
                "suspicion": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_AMY", perceived_state="Amy is lying about the kidnapping but I cannot prove it", confidence=0.85, inertia=0.6, established_at_fabula=0),
            ],
        ),
        "ENT_DESI": Entity(
            id="ENT_DESI",
            name="Desi Collings",
            location_id="LOC_DESI_LAKE_HOUSE",
            status="dead",
            traits={
                "obsession": TraitVector(value=0.8, inertia=0.7),
                "possessiveness": TraitVector(value=0.85, inertia=0.7),
                "wealth": TraitVector(value=0.9, inertia=0.9),
            },
            beliefs=[
                Belief(target_id="ENT_AMY", perceived_state="Amy needs my protection — she will finally be mine", confidence=0.8, inertia=0.7, established_at_fabula=0),
            ],
        ),
        "ENT_TANNER": Entity(
            id="ENT_TANNER",
            name="Tanner Bolt (Lawyer)",
            location_id="LOC_NORTH_CARTHAGE",
            status="healthy",
            traits={
                "cunning": TraitVector(value=0.85, inertia=0.8),
                "media_savvy": TraitVector(value=0.9, inertia=0.8),
            },
        ),
        "ENT_NOELLE": Entity(
            id="ENT_NOELLE",
            name="Noelle Hawthorne",
            location_id="LOC_NORTH_CARTHAGE",
            status="healthy",
            traits={
                "gullibility": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_AMY", perceived_state="Amy was my best friend and was pregnant when she vanished", confidence=0.95, inertia=0.7, established_at_fabula=0),
                Belief(target_id="ENT_NICK", perceived_state="Nick murdered his pregnant wife", confidence=0.9, inertia=0.6, established_at_fabula=0),
            ],
        ),
        "ENT_ANDIE": Entity(
            id="ENT_ANDIE",
            name="Andie (Nick's Mistress)",
            location_id="LOC_NORTH_CARTHAGE",
            status="healthy",
            traits={
                "naivete": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_NICK", perceived_state="Nick will leave Amy for me", confidence=0.7, inertia=0.4, established_at_fabula=0),
            ],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_JOBS_LOST", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_id=None, description="Nick and Amy both lose their writing jobs in the 2009 recession."),
        EventNode(id="EVT_TRUST_FUND_DEPLETED", fabula_time=2, syuzhet_index=2, event_type="outcome", actor_id=None, description="Amy's parents ask for money from her trust fund due to financial troubles; Nick uses the remainder to open a bar with Go."),
        EventNode(id="EVT_RELOCATE_MISSOURI", fabula_time=3, syuzhet_index=3, event_type="choice", actor_id="ENT_NICK", description="Nick and Amy relocate to North Carthage, Missouri to care for Nick's sick mother."),
        EventNode(id="EVT_MARRIAGE_DETERIORATES", fabula_time=4, syuzhet_index=4, event_type="outcome", actor_id=None, description="Their marriage deteriorates. Amy resents suburban life; Nick begins an affair with student Andie."),
        EventNode(id="EVT_AMY_DISCOVERS_AFFAIR", fabula_time=5, syuzhet_index=5, event_type="revelation", actor_id="ENT_AMY", description="Amy discovers Nick's affair and begins planning an elaborate revenge over the next year."),
        EventNode(id="EVT_AMY_DISAPPEARS", fabula_time=6, syuzhet_index=6, event_type="choice", actor_id="ENT_AMY", description="On their fifth anniversary, Amy stages her disappearance with faked signs of struggle, planting evidence to frame Nick."),
        EventNode(id="EVT_NICK_SUSPECT", fabula_time=7, syuzhet_index=7, event_type="outcome", actor_id=None, description="Nick becomes the prime suspect. Media scrutiny, faked pregnancy, credit card debt, and increased life insurance all point to him."),
        EventNode(id="EVT_NICK_FINDS_CLUES", fabula_time=8, syuzhet_index=8, event_type="revelation", actor_id="ENT_NICK", description="Nick follows Amy's treasure hunt clues and discovers the woodshed full of incriminating evidence. He realizes Amy framed him."),
        EventNode(id="EVT_AMY_ROBBED", fabula_time=9, syuzhet_index=9, event_type="outcome", actor_id=None, description="Amy is robbed at the motel hideout, losing her cash reserves."),
        EventNode(id="EVT_AMY_SEEKS_DESI", fabula_time=10, syuzhet_index=10, event_type="choice", actor_id="ENT_AMY", description="Desperate, Amy contacts her wealthy ex Desi Collings, who hides her in his lake house."),
        EventNode(id="EVT_NICK_TV_INTERVIEW", fabula_time=11, syuzhet_index=11, event_type="choice", actor_id="ENT_NICK", description="With Tanner Bolt's help, Nick gives a TV interview performing the perfect apologetic husband."),
        EventNode(id="EVT_NICK_ARRESTED", fabula_time=12, syuzhet_index=12, event_type="outcome", actor_id=None, description="Police discover the woodshed and Amy's faked diary. Nick is arrested."),
        EventNode(id="EVT_AMY_WATCHES_INTERVIEW", fabula_time=13, syuzhet_index=13, event_type="revelation", actor_id="ENT_AMY", description="Amy sees Nick's TV interview and is impressed, convinced they are uniquely matched."),
        EventNode(id="EVT_DESI_MURDERED", fabula_time=14, syuzhet_index=14, event_type="choice", actor_id="ENT_AMY", target_id="ENT_DESI", description="Amy mutilates herself to fake captivity, seduces Desi, then murders him to escape and frame him as her kidnapper."),
        EventNode(id="EVT_AMY_RETURNS", fabula_time=15, syuzhet_index=15, event_type="outcome", actor_id="ENT_AMY", description="Amy returns to North Carthage with a fabricated kidnapping story. Nick is released."),
        EventNode(id="EVT_FORCED_MARRIAGE", fabula_time=16, syuzhet_index=16, event_type="outcome", actor_id=None, description="Nick, Go, and Boney know Amy is lying but have no proof. Nick is forced back into married life."),
        EventNode(id="EVT_AMY_INSEMINATES", fabula_time=17, syuzhet_index=17, event_type="choice", actor_id="ENT_AMY", description="Amy uses Nick's frozen semen to become pregnant, using the unborn child as leverage to keep Nick compliant."),
        EventNode(id="EVT_NICK_SUBMITS", fabula_time=18, syuzhet_index=18, event_type="outcome", actor_id="ENT_NICK", description="Amy forces Nick to delete his exposé. Nick dedicates himself to the role of perfect husband."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_event_id="EVT_JOBS_LOST", target_node_id="EVT_TRUST_FUND_DEPLETED", mechanism="social", fabula_time=1),
        CausalEdge(source_event_id="EVT_TRUST_FUND_DEPLETED", target_node_id="EVT_RELOCATE_MISSOURI", mechanism="social", fabula_time=2),
        CausalEdge(source_event_id="EVT_RELOCATE_MISSOURI", target_node_id="EVT_MARRIAGE_DETERIORATES", mechanism="psychological", fabula_time=3),
        CausalEdge(source_event_id="EVT_MARRIAGE_DETERIORATES", target_node_id="EVT_AMY_DISCOVERS_AFFAIR", mechanism="epistemic", fabula_time=4),
        CausalEdge(source_event_id="EVT_AMY_DISCOVERS_AFFAIR", target_node_id="EVT_AMY_DISAPPEARS", mechanism="psychological", fabula_time=5),
        CausalEdge(source_event_id="EVT_AMY_DISAPPEARS", target_node_id="EVT_NICK_SUSPECT", mechanism="epistemic", fabula_time=6),
        CausalEdge(source_event_id="EVT_AMY_ROBBED", target_node_id="EVT_AMY_SEEKS_DESI", mechanism="psychological", fabula_time=9),
        CausalEdge(source_event_id="EVT_NICK_TV_INTERVIEW", target_node_id="EVT_AMY_WATCHES_INTERVIEW", mechanism="psychological", fabula_time=11),
        CausalEdge(source_event_id="EVT_AMY_WATCHES_INTERVIEW", target_node_id="EVT_DESI_MURDERED", mechanism="psychological", fabula_time=13),
        CausalEdge(source_event_id="EVT_DESI_MURDERED", target_node_id="EVT_AMY_RETURNS", mechanism="social", fabula_time=14),
        CausalEdge(source_event_id="EVT_AMY_RETURNS", target_node_id="EVT_FORCED_MARRIAGE", mechanism="social", fabula_time=15),
        CausalEdge(source_event_id="EVT_AMY_INSEMINATES", target_node_id="EVT_NICK_SUBMITS", mechanism="psychological", fabula_time=17),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_DESI_LAKE_HOUSE", target_id="LOC_MOTEL_HIDEOUT"),
        SpatialEdge(source_id="LOC_DESI_LAKE_HOUSE", target_id="LOC_NORTH_CARTHAGE"),
        SpatialEdge(source_id="LOC_DUNNE_HOUSE", target_id="LOC_NORTH_CARTHAGE"),
        SpatialEdge(source_id="LOC_GO_WOODSHED", target_id="LOC_NORTH_CARTHAGE"),
        SpatialEdge(source_id="LOC_NEW_YORK", target_id="LOC_NORTH_CARTHAGE"),
        SpatialEdge(source_id="LOC_NORTH_CARTHAGE", target_id="LOC_THE_BAR"),
    ],
    information_topology=[
        InformationEdge(
            source_id="ENT_AMY",
            target_ids=["ENT_DESI"],
            medium="telephone",
            established_at_fabula=10,
            terminated_at_fabula=14,
        ),
    ],
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_NICK", target_entity_id="ENT_AMY", affinity=-0.6, fear=0.47, power_dynamic=-0.7),
        RelationshipEdge(source_entity_id="ENT_AMY", target_entity_id="ENT_NICK", affinity=0.3, fear=0.45, power_dynamic=0.8),
        RelationshipEdge(source_entity_id="ENT_NICK", target_entity_id="ENT_GO", affinity=0.9, fear=0.05, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_NICK", target_entity_id="ENT_ANDIE", affinity=0.4, fear=0.25, power_dynamic=0.4),
        RelationshipEdge(source_entity_id="ENT_AMY", target_entity_id="ENT_DESI", affinity=-0.3, fear=0.35, power_dynamic=0.5),
        RelationshipEdge(source_entity_id="ENT_DESI", target_entity_id="ENT_AMY", affinity=0.8, fear=0.3, power_dynamic=0.4),
        RelationshipEdge(source_entity_id="ENT_BONEY", target_entity_id="ENT_NICK", affinity=-0.3, fear=0.35, power_dynamic=0.5),
        RelationshipEdge(source_entity_id="ENT_NOELLE", target_entity_id="ENT_AMY", affinity=0.6, fear=0.1, power_dynamic=-0.3),
        RelationshipEdge(source_entity_id="ENT_TANNER", target_entity_id="ENT_NICK", affinity=0.5, fear=0.15, power_dynamic=0.3),
    ],
)
