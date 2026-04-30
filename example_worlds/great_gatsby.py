"""The Great Gatsby — high-fidelity WorldStateV1 test fixture.

Authored against the current ingestion prompts. Demonstrates all five
CausalEdge modalities, per-axis ``RelationshipMetric``, explicit
``evidence_strength`` everywhere, and named-latent WORLD_ traits
(Jazz Age class divide, Prohibition bootleg economy, Lost-Generation
post-war disillusion) wired as common-cause parents over the events
they jointly drive.
"""
from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge, RelationshipMetric, InformationEdge,
    TraitVector, AmbientVector, Affordance, Belief, EntityStateSnapshot,
    GlobalTrait, WorldTraitSnapshot,
)

world_state = WorldStateV1(
    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_WEST_EGG": Location(
            name="West Egg",
            description="The vulgar new-money side of the Long Island bay where Gatsby's mansion looms beside Nick's bungalow.",
            ambient_state={
                "ostentation": AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
                "newness": AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_EAST_EGG": Location(
            name="East Egg",
            description="The old-money village across the bay; the Buchanans' ancestral-feeling Georgian mansion.",
            ambient_state={
                "old_money_complacency": AmbientVector(value=0.9, volatility=0.1, evidence_strength="strong"),
                "tradition": AmbientVector(value=0.8, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_GATSBYS_MANSION": Location(
            name="Gatsby's Mansion",
            description="A French-château imitation in West Egg used as the stage for Gatsby's bootleg-funded Saturday parties.",
            ambient_state={
                "spectacle": AmbientVector(value=0.95, volatility=0.3, evidence_strength="strong"),
                "loneliness": AmbientVector(value=0.7, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_VALLEY_OF_ASHES": Location(
            name="Valley of Ashes",
            description="The sprawling refuse dump between West Egg and Manhattan, presided over by the eyes of Doctor T. J. Eckleburg.",
            ambient_state={
                "decay": AmbientVector(value=0.95, volatility=0.1, evidence_strength="strong"),
                "industrial_blight": AmbientVector(value=0.9, volatility=0.1, evidence_strength="strong"),
                "moral_emptiness": AmbientVector(value=0.85, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_WILSON_GARAGE": Location(
            name="Wilson's Garage",
            description="George Wilson's failing auto-repair shop in the Valley of Ashes; Myrtle's marital prison.",
            ambient_state={
                "poverty": AmbientVector(value=0.85, volatility=0.1, evidence_strength="strong"),
                "exhaustion": AmbientVector(value=0.8, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_NEW_YORK_APT": Location(
            name="Tom's Manhattan Apartment",
            description="The small flat Tom keeps for trysts with Myrtle; site of the broken-nose party.",
            ambient_state={
                "secrecy": AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
                "drunkenness": AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
            },
        ),
        "LOC_PLAZA_SUITE": Location(
            name="Plaza Hotel Suite",
            description="Sweltering hotel suite where Gatsby and Tom finally confront each other over Daisy.",
            ambient_state={
                "heat": AmbientVector(value=0.95, volatility=0.2, evidence_strength="strong"),
                "tension": AmbientVector(value=0.95, volatility=0.4, evidence_strength="strong"),
            },
        ),
        "LOC_MIDWEST": Location(
            name="The Midwest",
            description="Nick's home country; the moral standard against which the East is finally judged.",
            ambient_state={
                "settledness": AmbientVector(value=0.85, volatility=0.1, evidence_strength="strong"),
            },
        ),
    },

    # ── OBJECTS ────────────────────────────────────────────────────────
    objects={
        "OBJ_GREEN_LIGHT": NarrativeObject(
            id="OBJ_GREEN_LIGHT", name="Green Light at the End of Daisy's Dock",
            location_id="LOC_EAST_EGG", owner_id="ENT_DAISY",
            properties={"state": "burning", "function": "object_of_longing"},
            affordances=[Affordance(action="symbolise_unreachable_dream", target_type="Entity")],
        ),
        "OBJ_YELLOW_CAR": NarrativeObject(
            id="OBJ_YELLOW_CAR", name="Gatsby's Yellow Rolls-Royce",
            location_id="LOC_GATSBYS_MANSION", owner_id="ENT_GATSBY",
            properties={"state": "registered_to_gatsby"},
            affordances=[Affordance(action="serve_as_murder_weapon", target_type="Entity")],
        ),
        "OBJ_ECKLEBURG_BILLBOARD": NarrativeObject(
            id="OBJ_ECKLEBURG_BILLBOARD", name="Eyes of Doctor T. J. Eckleburg",
            location_id="LOC_VALLEY_OF_ASHES", owner_id=None,
            properties={"state": "weathered_billboard", "function": "ersatz_god"},
            affordances=[Affordance(action="haunt_witnesses", target_type="Entity")],
        ),
        "OBJ_GATSBYS_TELEPHONE": NarrativeObject(
            id="OBJ_GATSBYS_TELEPHONE", name="Gatsby's Bootleg-Trade Telephone",
            location_id="LOC_GATSBYS_MANSION", owner_id="ENT_GATSBY",
            properties={"state": "wired_to_chicago"},
            affordances=[Affordance(action="reveal_origin_of_fortune", target_type="Entity")],
        ),
        "OBJ_REVOLVER": NarrativeObject(
            id="OBJ_REVOLVER", name="George Wilson's Revolver",
            location_id="LOC_WILSON_GARAGE", owner_id="ENT_GEORGE",
            properties={"state": "loaded"},
            affordances=[Affordance(action="kill", target_type="Entity")],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    entities={
        "ENT_NICK": Entity(
            id="ENT_NICK", name="Nick Carraway",
            location_id="LOC_MIDWEST", status="healthy",
            traits={
                "reserve":      TraitVector(value=0.7, inertia=0.7, evidence_strength="strong"),
                "moral_judgement": TraitVector(value=0.65, inertia=0.65, evidence_strength="moderate"),
                "fascination":  TraitVector(value=0.5, inertia=0.5, evidence_strength="moderate"),
                "disillusion":  TraitVector(value=0.2, inertia=0.5, evidence_strength="weak"),
            },
            beliefs=[
                Belief(target_id="ENT_GATSBY",
                       perceived_state="he turned out all right at the end; worth more than the whole damn rotten bunch",
                       confidence=0.9, inertia=0.7, established_at_fabula=1500, evidence_strength="strong"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=100, triggered_by="EVT_NICK_MOVES_EAST",
                    location_id="LOC_WEST_EGG"),
                EntityStateSnapshot(fabula_time=1500, triggered_by="EVT_GATSBY_KILLED",
                    traits={
                        "disillusion":     TraitVector(value=0.95, inertia=0.8, evidence_strength="strong"),
                        "moral_judgement": TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=1700, triggered_by="EVT_NICK_RETURNS_WEST",
                    location_id="LOC_MIDWEST"),
            ],
        ),
        "ENT_GATSBY": Entity(
            id="ENT_GATSBY", name="Jay Gatsby (James Gatz)",
            location_id="LOC_GATSBYS_MANSION", status="healthy",
            traits={
                "longing":           TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "self_invention":    TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "romantic_idealism": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "secrecy":           TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "loyalty_to_daisy":  TraitVector(value=1.0,  inertia=0.9, evidence_strength="strong"),
                "hope":              TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_DAISY",
                       perceived_state="Daisy will leave Tom for me once she sees what I have become",
                       confidence=0.95, inertia=0.85, established_at_fabula=200, evidence_strength="strong"),
                Belief(target_id="OBJ_GREEN_LIGHT",
                       perceived_state="proof that Daisy is within reach across the bay",
                       confidence=0.9, inertia=0.85, established_at_fabula=200, evidence_strength="strong"),
            ],
            constants=["origin_north_dakota_farm", "wartime_officer"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=900, triggered_by="EVT_GATSBY_DAISY_REUNION",
                    traits={
                        "hope": TraitVector(value=1.0, inertia=0.85, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=1300, triggered_by="EVT_PLAZA_CONFRONTATION",
                    traits={
                        "hope": TraitVector(value=0.4, inertia=0.85, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_DAISY"],
                    beliefs_added=[
                        Belief(target_id="ENT_DAISY",
                               perceived_state="she may stay with Tom, but I will still take her blame",
                               confidence=0.7, inertia=0.85, established_at_fabula=1300, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=1500, triggered_by="EVT_GATSBY_KILLED",
                    status="dead", location_id="LOC_GATSBYS_MANSION"),
            ],
        ),
        "ENT_DAISY": Entity(
            id="ENT_DAISY", name="Daisy Buchanan",
            location_id="LOC_EAST_EGG", status="healthy",
            traits={
                "charm":            TraitVector(value=0.95, inertia=0.8, evidence_strength="strong"),
                "carelessness":     TraitVector(value=0.8,  inertia=0.7, evidence_strength="strong"),
                "vulnerability":    TraitVector(value=0.7,  inertia=0.5, evidence_strength="moderate"),
                "voice_of_money":   TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                "rekindled_love":   TraitVector(value=0.2,  inertia=0.4, evidence_strength="weak"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=900, triggered_by="EVT_GATSBY_DAISY_REUNION",
                    traits={
                        "rekindled_love": TraitVector(value=0.85, inertia=0.55, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=1300, triggered_by="EVT_PLAZA_CONFRONTATION",
                    traits={
                        "rekindled_love": TraitVector(value=0.3, inertia=0.55, evidence_strength="strong"),
                        "vulnerability":  TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=1450, triggered_by="EVT_MYRTLE_KILLED",
                    location_id="LOC_EAST_EGG",
                    traits={
                        "carelessness": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_TOM": Entity(
            id="ENT_TOM", name="Tom Buchanan",
            location_id="LOC_EAST_EGG", status="healthy",
            traits={
                "aggression":       TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
                "racial_arrogance": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "physicality":      TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "possessiveness":   TraitVector(value=0.9,  inertia=0.8, evidence_strength="strong"),
                "hypocrisy":        TraitVector(value=0.9,  inertia=0.8, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_GATSBY",
                       perceived_state="vulgar Mr Nobody from Nowhere, a bootlegger and a fraud",
                       confidence=0.95, inertia=0.8, established_at_fabula=1200, evidence_strength="strong"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1200, triggered_by="EVT_TOM_DISCOVERS_AFFAIR",
                    traits={
                        "possessiveness": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=1480, triggered_by="EVT_TOM_TELLS_GEORGE",
                    location_id="LOC_WILSON_GARAGE"),
            ],
        ),
        "ENT_JORDAN": Entity(
            id="ENT_JORDAN", name="Jordan Baker",
            location_id="LOC_EAST_EGG", status="healthy",
            traits={
                "insolence":      TraitVector(value=0.8, inertia=0.7, evidence_strength="strong"),
                "dishonesty":     TraitVector(value=0.7, inertia=0.65, evidence_strength="moderate"),
                "athletic_poise": TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
            },
            beliefs=[],
        ),
        "ENT_MYRTLE": Entity(
            id="ENT_MYRTLE", name="Myrtle Wilson",
            location_id="LOC_WILSON_GARAGE", status="healthy",
            traits={
                "vitality":       TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
                "social_aspiration": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "discontent":     TraitVector(value=0.9, inertia=0.75, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_TOM",
                       perceived_state="Tom will leave Daisy and rescue me from the ashes",
                       confidence=0.7, inertia=0.5, established_at_fabula=400, evidence_strength="moderate"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=500, triggered_by="EVT_TOM_BREAKS_NOSE",
                    status="ill",
                    traits={
                        "discontent": TraitVector(value=0.95, inertia=0.8, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=1450, triggered_by="EVT_MYRTLE_KILLED",
                    status="dead", location_id="LOC_VALLEY_OF_ASHES"),
            ],
        ),
        "ENT_GEORGE": Entity(
            id="ENT_GEORGE", name="George Wilson",
            location_id="LOC_WILSON_GARAGE", status="healthy",
            traits={
                "exhaustion":     TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
                "obliviousness":  TraitVector(value=0.7,  inertia=0.65, evidence_strength="moderate"),
                "love_for_myrtle": TraitVector(value=0.9, inertia=0.8, evidence_strength="strong"),
                "rage":           TraitVector(value=0.1,  inertia=0.4, evidence_strength="weak"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1300, triggered_by="EVT_GEORGE_SUSPECTS_AFFAIR",
                    traits={
                        "obliviousness": TraitVector(value=0.2, inertia=0.7, evidence_strength="strong"),
                        "rage":          TraitVector(value=0.7, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_MYRTLE",
                               perceived_state="my wife has been unfaithful — I shall lock her in",
                               confidence=0.9, inertia=0.7, established_at_fabula=1300, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=1450, triggered_by="EVT_MYRTLE_KILLED",
                    traits={
                        "rage":  TraitVector(value=1.0, inertia=0.7, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="OBJ_ECKLEBURG_BILLBOARD",
                               perceived_state="God sees everything",
                               confidence=0.95, inertia=0.85, established_at_fabula=1450, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=1490, triggered_by="EVT_TOM_TELLS_GEORGE",
                    beliefs_added=[
                        Belief(target_id="ENT_GATSBY",
                               perceived_state="the man who owns the yellow car must be Myrtle's lover and her killer",
                               confidence=0.95, inertia=0.6, established_at_fabula=1490, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=1500, triggered_by="EVT_GATSBY_KILLED",
                    status="dead", location_id="LOC_GATSBYS_MANSION"),
            ],
        ),
        "ENT_HENRY_GATZ": Entity(
            id="ENT_HENRY_GATZ", name="Henry Gatz",
            location_id="LOC_MIDWEST", status="healthy",
            traits={
                "paternal_pride": TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
                "humility":       TraitVector(value=0.8, inertia=0.8, evidence_strength="strong"),
                "grief":          TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1600, triggered_by="EVT_FUNERAL",
                    location_id="LOC_GATSBYS_MANSION"),
            ],
        ),
        "ENT_MEYER_WOLFSHIEM": Entity(
            id="ENT_MEYER_WOLFSHIEM", name="Meyer Wolfshiem",
            location_id="LOC_NEW_YORK_APT", status="healthy",
            traits={
                "criminality":   TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "discretion":    TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "patronage":     TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
            },
            beliefs=[],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_GATSBY_MEETS_DAISY_1917", fabula_time=50, syuzhet_index=10,
                  event_type="outcome", actor_ids=["ENT_GATSBY", "ENT_DAISY"], target_ids=[],
                  description="In Louisville, 1917, Lieutenant Gatsby and the debutante Daisy fall in love before he ships out for the war."),
        EventNode(id="EVT_DAISY_MARRIES_TOM", fabula_time=80, syuzhet_index=11,
                  event_type="choice", actor_ids=["ENT_DAISY", "ENT_TOM"], target_ids=[],
                  description="With Gatsby still overseas, Daisy reluctantly marries Tom Buchanan, the wealthy Yale football star from Chicago."),
        EventNode(id="EVT_NICK_MOVES_EAST", fabula_time=100, syuzhet_index=1,
                  event_type="choice", actor_ids=["ENT_NICK"], target_ids=[],
                  description="In spring 1922 Nick Carraway moves east to sell bonds and rents a bungalow in West Egg next to Gatsby's mansion."),
        EventNode(id="EVT_NICK_DINES_AT_BUCHANANS", fabula_time=200, syuzhet_index=2,
                  event_type="choice", actor_ids=["ENT_NICK", "ENT_DAISY", "ENT_TOM"], target_ids=["ENT_JORDAN"],
                  description="Nick dines in East Egg with the Buchanans and meets Jordan Baker, who reveals that Tom keeps a mistress."),
        EventNode(id="EVT_NICK_SEES_GREEN_LIGHT", fabula_time=250, syuzhet_index=3,
                  event_type="revelation", actor_ids=["ENT_NICK"], target_ids=["ENT_GATSBY"],
                  description="Returning home, Nick sees Gatsby alone on his lawn, stretching his arms toward a green light across the bay."),
        EventNode(id="EVT_TOM_TAKES_NICK_TO_NY", fabula_time=350, syuzhet_index=4,
                  event_type="choice", actor_ids=["ENT_TOM", "ENT_NICK"], target_ids=["ENT_MYRTLE"],
                  description="Tom drags Nick into the city by way of Wilson's garage, picking up Myrtle for a party at his Manhattan apartment."),
        EventNode(id="EVT_TOM_BREAKS_NOSE", fabula_time=500, syuzhet_index=5,
                  event_type="outcome", actor_ids=["ENT_TOM"], target_ids=["ENT_MYRTLE"],
                  description="At the apartment party Tom slaps Myrtle and breaks her nose for daring to mention Daisy's name."),
        EventNode(id="EVT_NICK_AT_GATSBYS_PARTY", fabula_time=700, syuzhet_index=6,
                  event_type="choice", actor_ids=["ENT_NICK"], target_ids=["ENT_GATSBY"],
                  description="Nick attends one of Gatsby's vast Saturday-night parties; Gatsby singles him out and introduces himself."),
        EventNode(id="EVT_JORDAN_TELLS_NICK_HISTORY", fabula_time=800, syuzhet_index=7,
                  event_type="revelation", actor_ids=["ENT_JORDAN"], target_ids=["ENT_NICK"],
                  description="At the Plaza, Jordan tells Nick that Gatsby and Daisy were lovers in 1917 and that the parties have all been bait to draw Daisy back."),
        EventNode(id="EVT_GATSBY_DAISY_REUNION", fabula_time=900, syuzhet_index=8,
                  event_type="choice", actor_ids=["ENT_GATSBY", "ENT_DAISY"], target_ids=[],
                  description="Nick stages a tea-time reunion at his bungalow; Gatsby and Daisy embark on an affair."),
        EventNode(id="EVT_TOM_DISCOVERS_AFFAIR", fabula_time=1200, syuzhet_index=9,
                  event_type="revelation", actor_ids=["ENT_DAISY"], target_ids=["ENT_TOM"],
                  description="Daisy carelessly addresses Gatsby with unmistakable intimacy in front of Tom, who realises the affair."),
        EventNode(id="EVT_GEORGE_SUSPECTS_AFFAIR", fabula_time=1280, syuzhet_index=12,
                  event_type="revelation", actor_ids=["ENT_GEORGE"], target_ids=["ENT_MYRTLE"],
                  description="George finds evidence that Myrtle has a lover, locks her upstairs, and resolves to take her west."),
        EventNode(id="EVT_PLAZA_CONFRONTATION", fabula_time=1300, syuzhet_index=13,
                  event_type="choice", actor_ids=["ENT_GATSBY", "ENT_TOM", "ENT_DAISY"], target_ids=[],
                  description="In a Plaza Hotel suite Gatsby demands Daisy say she never loved Tom; Tom exposes Gatsby's bootlegging; Daisy retreats to Tom."),
        EventNode(id="EVT_MYRTLE_KILLED", fabula_time=1450, syuzhet_index=14,
                  event_type="outcome", actor_ids=["ENT_DAISY"], target_ids=["ENT_MYRTLE"],
                  description="Driving Gatsby's yellow car back to East Egg, Daisy strikes and kills Myrtle, who has run into the road; Gatsby resolves to take the blame."),
        EventNode(id="EVT_TOM_TELLS_GEORGE", fabula_time=1480, syuzhet_index=15,
                  event_type="choice", actor_ids=["ENT_TOM"], target_ids=["ENT_GEORGE"],
                  description="Tom tells the grieving George that Gatsby owns the yellow car, knowing George will assume Gatsby was Myrtle's lover and killer."),
        EventNode(id="EVT_GATSBY_KILLED", fabula_time=1500, syuzhet_index=16,
                  event_type="outcome", actor_ids=["ENT_GEORGE"], target_ids=["ENT_GATSBY", "ENT_GEORGE"],
                  description="George shoots Gatsby in his swimming pool and then kills himself."),
        EventNode(id="EVT_FUNERAL", fabula_time=1600, syuzhet_index=17,
                  event_type="outcome", actor_ids=["ENT_NICK", "ENT_HENRY_GATZ"], target_ids=["ENT_GATSBY"],
                  description="Almost no one attends Gatsby's funeral; only Nick and the late-arriving Henry Gatz mourn him."),
        EventNode(id="EVT_NICK_RETURNS_WEST", fabula_time=1700, syuzhet_index=18,
                  event_type="choice", actor_ids=["ENT_NICK"], target_ids=[],
                  description="Disgusted with the East, Nick decides to return to the Midwest; he encounters Tom on Fifth Avenue and reluctantly shakes his hand."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction ──
        CausalEdge(source_id="EVT_GATSBY_MEETS_DAISY_1917", target_id="EVT_DAISY_MARRIES_TOM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=50, propagation_delay=30),
        CausalEdge(source_id="EVT_NICK_MOVES_EAST", target_id="EVT_NICK_DINES_AT_BUCHANANS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=100, propagation_delay=100),
        CausalEdge(source_id="EVT_NICK_DINES_AT_BUCHANANS", target_id="EVT_NICK_SEES_GREEN_LIGHT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=200, propagation_delay=50),
        CausalEdge(source_id="EVT_NICK_DINES_AT_BUCHANANS", target_id="EVT_TOM_TAKES_NICK_TO_NY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=200, propagation_delay=150),
        CausalEdge(source_id="EVT_TOM_TAKES_NICK_TO_NY", target_id="EVT_TOM_BREAKS_NOSE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=350, propagation_delay=150),
        CausalEdge(source_id="EVT_NICK_MOVES_EAST", target_id="EVT_NICK_AT_GATSBYS_PARTY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=100, propagation_delay=600),
        CausalEdge(source_id="EVT_NICK_AT_GATSBYS_PARTY", target_id="EVT_JORDAN_TELLS_NICK_HISTORY",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=6.0, fabula_time=700, propagation_delay=100),
        CausalEdge(source_id="EVT_JORDAN_TELLS_NICK_HISTORY", target_id="EVT_GATSBY_DAISY_REUNION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=800, propagation_delay=100),
        CausalEdge(source_id="EVT_GATSBY_DAISY_REUNION", target_id="EVT_TOM_DISCOVERS_AFFAIR",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=900, propagation_delay=300),
        CausalEdge(source_id="EVT_TOM_DISCOVERS_AFFAIR", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1200, propagation_delay=100),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="EVT_GEORGE_SUSPECTS_AFFAIR",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=500, propagation_delay=780),
        CausalEdge(source_id="EVT_GEORGE_SUSPECTS_AFFAIR", target_id="EVT_MYRTLE_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1280, propagation_delay=170),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="EVT_MYRTLE_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1300, propagation_delay=150),
        CausalEdge(source_id="EVT_MYRTLE_KILLED", target_id="EVT_TOM_TELLS_GEORGE",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=9.0, fabula_time=1450, propagation_delay=30),
        CausalEdge(source_id="EVT_TOM_TELLS_GEORGE", target_id="EVT_GATSBY_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1480, propagation_delay=20),
        CausalEdge(source_id="EVT_GATSBY_KILLED", target_id="EVT_FUNERAL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1500, propagation_delay=100),
        CausalEdge(source_id="EVT_FUNERAL", target_id="EVT_NICK_RETURNS_WEST",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1600, propagation_delay=100),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="EVT_GATSBY_DAISY_REUNION",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=80, propagation_delay=820),

        # ── mutation ──
        CausalEdge(source_id="EVT_GATSBY_MEETS_DAISY_1917", target_id="ENT_GATSBY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=50,
                   trait_target="longing", trait_delta=0.5),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_GATSBY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=80,
                   trait_target="self_invention", trait_delta=0.6),
        CausalEdge(source_id="EVT_GATSBY_DAISY_REUNION", target_id="ENT_GATSBY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=900,
                   trait_target="hope", trait_delta=0.2),
        CausalEdge(source_id="EVT_GATSBY_DAISY_REUNION", target_id="ENT_DAISY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=900,
                   trait_target="rekindled_love", trait_delta=0.65),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_GATSBY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1300,
                   trait_target="hope", trait_delta=-0.55),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_DAISY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1300,
                   trait_target="rekindled_love", trait_delta=-0.55),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_DAISY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1300,
                   trait_target="vulnerability", trait_delta=0.15),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="ENT_MYRTLE",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=500,
                   trait_target="discontent", trait_delta=0.05),
        CausalEdge(source_id="EVT_GEORGE_SUSPECTS_AFFAIR", target_id="ENT_GEORGE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1280,
                   trait_target="rage", trait_delta=0.6),
        CausalEdge(source_id="EVT_MYRTLE_KILLED", target_id="ENT_GEORGE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1450,
                   trait_target="rage", trait_delta=0.3),
        CausalEdge(source_id="EVT_MYRTLE_KILLED", target_id="ENT_DAISY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1450,
                   trait_target="carelessness", trait_delta=0.15),
        CausalEdge(source_id="EVT_GATSBY_KILLED", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1500,
                   trait_target="disillusion", trait_delta=0.75),
        CausalEdge(source_id="EVT_GATSBY_KILLED", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1500,
                   trait_target="moral_judgement", trait_delta=0.2),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_GATSBY_MEETS_DAISY_1917", target_id="ENT_GATSBY",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=50,
                   trait_target="affinity", trait_delta=1.0, rel_counterpart_id="ENT_DAISY"),
        CausalEdge(source_id="EVT_GATSBY_MEETS_DAISY_1917", target_id="ENT_DAISY",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=50,
                   trait_target="affinity", trait_delta=0.9, rel_counterpart_id="ENT_GATSBY"),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_GATSBY",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=80,
                   trait_target="affinity", trait_delta=-0.95, rel_counterpart_id="ENT_TOM"),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="ENT_MYRTLE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=500,
                   trait_target="fear", trait_delta=0.45, rel_counterpart_id="ENT_TOM"),
        CausalEdge(source_id="EVT_TOM_DISCOVERS_AFFAIR", target_id="ENT_TOM",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1200,
                   trait_target="affinity", trait_delta=-1.0, rel_counterpart_id="ENT_GATSBY"),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_DAISY",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1300,
                   trait_target="affinity", trait_delta=-0.4, rel_counterpart_id="ENT_GATSBY"),
        CausalEdge(source_id="EVT_GEORGE_SUSPECTS_AFFAIR", target_id="ENT_GEORGE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1280,
                   trait_target="affinity", trait_delta=-0.6, rel_counterpart_id="ENT_MYRTLE"),
        CausalEdge(source_id="EVT_TOM_TELLS_GEORGE", target_id="ENT_GEORGE",
                   causality_type="mutation_social", mechanism="informational", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1480,
                   trait_target="affinity", trait_delta=-1.0, rel_counterpart_id="ENT_GATSBY"),

        # ── affordance_gate ──
        CausalEdge(source_id="OBJ_GREEN_LIGHT", target_id="EVT_NICK_SEES_GREEN_LIGHT",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="strong",
                   causal_force=7.0, fabula_time=250),
        CausalEdge(source_id="OBJ_YELLOW_CAR", target_id="EVT_MYRTLE_KILLED",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=1450),
        CausalEdge(source_id="OBJ_YELLOW_CAR", target_id="EVT_TOM_TELLS_GEORGE",
                   causality_type="affordance_gate", mechanism="informational", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1480),
        CausalEdge(source_id="OBJ_REVOLVER", target_id="EVT_GATSBY_KILLED",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1500),
        CausalEdge(source_id="OBJ_GATSBYS_TELEPHONE", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="affordance_gate", mechanism="informational", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1300),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_VALLEY_OF_ASHES", target_id="ENT_GEORGE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=400),
        CausalEdge(source_id="LOC_VALLEY_OF_ASHES", target_id="ENT_MYRTLE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=400),
        CausalEdge(source_id="LOC_PLAZA_SUITE", target_id="ENT_TOM",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1300),
        CausalEdge(source_id="LOC_GATSBYS_MANSION", target_id="ENT_NICK",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=700),

        # ── WORLD_ → Event ──
        CausalEdge(source_id="WORLD_OLD_VS_NEW_MONEY", target_id="EVT_DAISY_MARRIES_TOM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=80),
        CausalEdge(source_id="WORLD_OLD_VS_NEW_MONEY", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1300),
        CausalEdge(source_id="WORLD_OLD_VS_NEW_MONEY", target_id="EVT_NICK_RETURNS_WEST",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1700),
        CausalEdge(source_id="WORLD_PROHIBITION_BOOTLEG", target_id="EVT_NICK_AT_GATSBYS_PARTY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=700),
        CausalEdge(source_id="WORLD_PROHIBITION_BOOTLEG", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1300),
        CausalEdge(source_id="WORLD_PROHIBITION_BOOTLEG", target_id="EVT_FUNERAL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1600),
        CausalEdge(source_id="WORLD_LOST_GENERATION", target_id="EVT_GATSBY_MEETS_DAISY_1917",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=50),
        CausalEdge(source_id="WORLD_LOST_GENERATION", target_id="EVT_NICK_MOVES_EAST",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=100),
        CausalEdge(source_id="WORLD_LOST_GENERATION", target_id="EVT_GATSBY_DAISY_REUNION",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=900),
        CausalEdge(source_id="WORLD_LOST_GENERATION", target_id="EVT_NICK_RETURNS_WEST",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1700),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_WEST_EGG", target_id="LOC_EAST_EGG"),
        SpatialEdge(source_id="LOC_EAST_EGG", target_id="LOC_WEST_EGG"),
        SpatialEdge(source_id="LOC_WEST_EGG", target_id="LOC_GATSBYS_MANSION"),
        SpatialEdge(source_id="LOC_GATSBYS_MANSION", target_id="LOC_WEST_EGG"),
        SpatialEdge(source_id="LOC_WEST_EGG", target_id="LOC_VALLEY_OF_ASHES"),
        SpatialEdge(source_id="LOC_VALLEY_OF_ASHES", target_id="LOC_WILSON_GARAGE"),
        SpatialEdge(source_id="LOC_VALLEY_OF_ASHES", target_id="LOC_NEW_YORK_APT"),
        SpatialEdge(source_id="LOC_NEW_YORK_APT", target_id="LOC_PLAZA_SUITE"),
        SpatialEdge(source_id="LOC_EAST_EGG", target_id="LOC_VALLEY_OF_ASHES"),
        SpatialEdge(source_id="LOC_MIDWEST", target_id="LOC_WEST_EGG"),
        SpatialEdge(source_id="LOC_WEST_EGG", target_id="LOC_MIDWEST"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    information_topology=[
        InformationEdge(source_id="ENT_JORDAN", target_ids=["ENT_NICK"],
                        medium="confidential_anecdote", is_encrypted=False,
                        established_at_fabula=200, terminated_at_fabula=200,
                        discovered_at_syuzhet=2, evidence_strength="strong"),
        InformationEdge(source_id="ENT_JORDAN", target_ids=["ENT_NICK", "ENT_GATSBY"],
                        medium="hotel_disclosure", is_encrypted=False,
                        established_at_fabula=800, terminated_at_fabula=800,
                        discovered_at_syuzhet=7, evidence_strength="strong"),
        InformationEdge(source_id="ENT_GATSBY", target_ids=["ENT_NICK"],
                        medium="speakeasy_confidence", is_encrypted=True,
                        established_at_fabula=750, terminated_at_fabula=750,
                        discovered_at_syuzhet=6, evidence_strength="moderate"),
        InformationEdge(source_id="ENT_DAISY", target_ids=["ENT_TOM"],
                        medium="unguarded_glance", is_encrypted=False,
                        established_at_fabula=1200, terminated_at_fabula=1200,
                        discovered_at_syuzhet=9, evidence_strength="strong"),
        InformationEdge(source_id="ENT_TOM", target_ids=["ENT_GATSBY", "ENT_DAISY"],
                        medium="public_accusation", is_encrypted=False,
                        established_at_fabula=1300, terminated_at_fabula=1300,
                        discovered_at_syuzhet=13, evidence_strength="strong"),
        InformationEdge(source_id="ENT_GATSBY", target_ids=["ENT_NICK"],
                        medium="post_accident_confession", is_encrypted=True,
                        established_at_fabula=1460, terminated_at_fabula=1460,
                        discovered_at_syuzhet=14, evidence_strength="strong"),
        InformationEdge(source_id="ENT_TOM", target_ids=["ENT_GEORGE"],
                        medium="lethal_disclosure", is_encrypted=False,
                        established_at_fabula=1480, terminated_at_fabula=1480,
                        discovered_at_syuzhet=15, evidence_strength="strong"),
        InformationEdge(source_id="OBJ_ECKLEBURG_BILLBOARD", target_ids=["ENT_GEORGE"],
                        medium="symbolic_perception", is_encrypted=False,
                        established_at_fabula=1450, terminated_at_fabula=1500,
                        discovered_at_syuzhet=14, evidence_strength="moderate"),
    ],

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_OLD_VS_NEW_MONEY": GlobalTrait(
            id="WORLD_OLD_VS_NEW_MONEY",
            name="Old Money vs New Money Divide",
            description="The Jazz Age caste line between inherited East Egg gentility and acquired West Egg fortune. Operates as common-cause parent over Daisy's marriage choice, the Plaza confrontation, and Nick's final flight west.",
            category="social_structure",
            magnitude=TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=1500, triggered_by="EVT_GATSBY_KILLED",
                    magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    description="The line is reaffirmed by Gatsby's death; old money escapes consequence."),
            ],
        ),
        "WORLD_PROHIBITION_BOOTLEG": GlobalTrait(
            id="WORLD_PROHIBITION_BOOTLEG",
            name="Prohibition Bootleg Economy",
            description="The 1920s Volstead-Act-driven shadow economy that funds Gatsby's mansion via Wolfshiem's drugstore-fronts and gives Tom the moral ammunition to destroy him. Operates as a high-magnitude latent that makes Gatsby's wealth and his vulnerability one and the same.",
            category="economy",
            magnitude=TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            affected_domains=["economic", "social"],
        ),
        "WORLD_LOST_GENERATION": GlobalTrait(
            id="WORLD_LOST_GENERATION",
            name="Post-War Disillusion (Lost Generation)",
            description="The pervasive American 1922 mood — restless veterans, broken promises of pre-war life, the willing suspension of moral seriousness — that puts Nick on an eastbound train, Gatsby in a Long Island mansion, and Daisy in a careless marriage.",
            category="cosmology",
            magnitude=TraitVector(value=0.8, inertia=0.85, evidence_strength="strong"),
            affected_domains=["psychological", "social"],
        ),
    },

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────
    social_topology=[
        # Gatsby ↔ Daisy — the obsession.
        RelationshipEdge(
            source_entity_id="ENT_GATSBY", target_entity_id="ENT_DAISY",
            metrics={
                "affinity": RelationshipMetric(value=1.0, inertia=0.6, evidence_strength="strong", last_updated_fabula=900),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_DAISY", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity": RelationshipMetric(value=0.55, inertia=0.45, evidence_strength="strong", last_updated_fabula=1300),
            },
        ),
        # Daisy ↔ Tom — wealthy marriage.
        RelationshipEdge(
            source_entity_id="ENT_DAISY", target_entity_id="ENT_TOM",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.55, evidence_strength="strong", last_updated_fabula=1300),
                "fear":          RelationshipMetric(value=0.35, inertia=0.2, evidence_strength="moderate", last_updated_fabula=1200),
                "power_dynamic": RelationshipMetric(value=-0.55, inertia=0.7, evidence_strength="strong", last_updated_fabula=1300),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_TOM", target_entity_id="ENT_DAISY",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.55, evidence_strength="strong", last_updated_fabula=1300),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.7, evidence_strength="strong", last_updated_fabula=1300),
            },
        ),
        # Tom ↔ Gatsby — class hatred.
        RelationshipEdge(
            source_entity_id="ENT_TOM", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity":      RelationshipMetric(value=-0.95, inertia=0.5, evidence_strength="strong", last_updated_fabula=1300),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.7, evidence_strength="strong", last_updated_fabula=1300),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_GATSBY", target_entity_id="ENT_TOM",
            metrics={
                "affinity": RelationshipMetric(value=-0.85, inertia=0.5, evidence_strength="strong", last_updated_fabula=1300),
            },
        ),
        # Tom ↔ Myrtle — affair.
        RelationshipEdge(
            source_entity_id="ENT_TOM", target_entity_id="ENT_MYRTLE",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.5, evidence_strength="strong", last_updated_fabula=500),
                "power_dynamic": RelationshipMetric(value=0.85, inertia=0.75, evidence_strength="strong", last_updated_fabula=500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MYRTLE", target_entity_id="ENT_TOM",
            metrics={
                "affinity":      RelationshipMetric(value=0.7, inertia=0.5, evidence_strength="strong", last_updated_fabula=500),
                "fear":          RelationshipMetric(value=0.45, inertia=0.2, evidence_strength="strong", last_updated_fabula=500),
                "power_dynamic": RelationshipMetric(value=-0.85, inertia=0.75, evidence_strength="strong", last_updated_fabula=500),
            },
        ),
        # Myrtle ↔ George — exhausted marriage.
        RelationshipEdge(
            source_entity_id="ENT_MYRTLE", target_entity_id="ENT_GEORGE",
            metrics={
                "affinity": RelationshipMetric(value=-0.4, inertia=0.55, evidence_strength="strong", last_updated_fabula=400),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_GEORGE", target_entity_id="ENT_MYRTLE",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.65, evidence_strength="strong", last_updated_fabula=400),
            },
        ),
        # George → Gatsby — fatal mistake.
        RelationshipEdge(
            source_entity_id="ENT_GEORGE", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity": RelationshipMetric(value=-0.95, inertia=0.5, evidence_strength="strong", last_updated_fabula=1490),
            },
        ),
        # Nick ↔ Gatsby — admiration.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.55, evidence_strength="strong", last_updated_fabula=1500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_GATSBY", target_entity_id="ENT_NICK",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.55, evidence_strength="strong", last_updated_fabula=1460),
            },
        ),
        # Nick ↔ Tom — moral repudiation.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_TOM",
            metrics={
                "affinity": RelationshipMetric(value=-0.85, inertia=0.55, evidence_strength="strong", last_updated_fabula=1700),
            },
        ),
        # Nick ↔ Jordan — flirtation cooled.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_JORDAN",
            metrics={
                "affinity": RelationshipMetric(value=0.3, inertia=0.45, evidence_strength="moderate", last_updated_fabula=1700),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JORDAN", target_entity_id="ENT_NICK",
            metrics={
                "affinity": RelationshipMetric(value=0.5, inertia=0.45, evidence_strength="moderate", last_updated_fabula=1500),
            },
        ),
        # Wolfshiem → Gatsby — patron.
        RelationshipEdge(
            source_entity_id="ENT_MEYER_WOLFSHIEM", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=700),
                "power_dynamic": RelationshipMetric(value=0.7, inertia=0.75, evidence_strength="moderate", last_updated_fabula=700),
            },
        ),
        # Henry Gatz → Gatsby — paternal pride.
        RelationshipEdge(
            source_entity_id="ENT_HENRY_GATZ", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity": RelationshipMetric(value=0.95, inertia=0.8, evidence_strength="strong", last_updated_fabula=1600),
            },
        ),
    ],
)
