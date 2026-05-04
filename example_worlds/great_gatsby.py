# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Great Gatsby — high-fidelity WorldStateV1 test fixture.

Authored against the current ingestion prompts. Demonstrates all five
CausalEdge modalities, per-axis ``RelationshipMetric``, explicit
``evidence_strength`` everywhere, and named-latent WORLD_ traits
(Jazz Age class divide, Prohibition bootleg economy, Lost-Generation
post-war disillusion) wired as common-cause parents over the events
they jointly drive.
"""
from shadow_loom.models import (
    Channel,
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge, RelationshipMetric, TraitVector, AmbientVector, Affordance, Belief, EntityStateSnapshot,
    GlobalTrait, WorldTraitSnapshot,
    NarrativeStyle,
)

world_state = WorldStateV1(
    narrative_style=NarrativeStyle(
        format='plot_summary',
        target_word_min=291,
        target_word_max=1090,
        prose_density='sparse',
        voice='third-person past-tense plot summary; condensed beat-by-beat diction; no quoted dialogue; uses temporal connectors ("Later,", "The next day,"); third-person POV; past tense',
        style_exemplar='n spring 1922, Nick Carraway—a Yale alumnus from the Midwest and a World War I veteran—journeys to New York City to obtain employment as a bond salesman. He rents a bungalow in the Long Island village of West Egg, next to a luxurious estate inhabited by Jay Gatsby, an enigmatic multi-millionaire who hosts dazzling soirées, yet does not partake in them.\n\nOne evening, Nick dines with a distant cousin, Daisy Buchanan, in the old money town of East Egg. Daisy is married to Tom Buchanan, formerly a Yale football star whom Nick knew during his college days.',
        source_word_count=727,
    ),
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
                       confidence=0.9, inertia=0.7, established_at_fabula=15000, evidence_strength="strong"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_NICK_MOVES_EAST",
                    location_id="LOC_WEST_EGG"),
                EntityStateSnapshot(fabula_time=15000, triggered_by="EVT_GATSBY_KILLED",
                    traits={
                        "disillusion":     TraitVector(value=0.95, inertia=0.8, evidence_strength="strong"),
                        "moral_judgement": TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=17000, triggered_by="EVT_NICK_RETURNS_WEST",
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
                       confidence=0.95, inertia=0.85, established_at_fabula=2000, evidence_strength="strong"),
                Belief(target_id="OBJ_GREEN_LIGHT",
                       perceived_state="proof that Daisy is within reach across the bay",
                       confidence=0.9, inertia=0.85, established_at_fabula=2000, evidence_strength="strong"),
            ],
            constants=["origin_north_dakota_farm", "wartime_officer"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=500, triggered_by="EVT_GATSBY_MEETS_DAISY_1917",
                    traits={
                        "longing": TraitVector(value=1.00, inertia=0.85, evidence_strength="moderate"),
                    }),
                EntityStateSnapshot(fabula_time=800, triggered_by="EVT_DAISY_MARRIES_TOM",
                    traits={
                        "self_invention": TraitVector(value=1.00, inertia=0.85, evidence_strength="moderate"),
                    }),EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_GATSBY_DAISY_REUNION",
                    traits={
                        "hope": TraitVector(value=1.0, inertia=0.85, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_PLAZA_CONFRONTATION",
                    traits={
                        "hope": TraitVector(value=0.4, inertia=0.85, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_DAISY"],
                    beliefs_added=[
                        Belief(target_id="ENT_DAISY",
                               perceived_state="she may stay with Tom, but I will still take her blame",
                               confidence=0.7, inertia=0.85, established_at_fabula=13000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=15000, triggered_by="EVT_GATSBY_KILLED",
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
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_GATSBY_DAISY_REUNION",
                    traits={
                        "rekindled_love": TraitVector(value=0.85, inertia=0.55, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_PLAZA_CONFRONTATION",
                    traits={
                        "rekindled_love": TraitVector(value=0.3, inertia=0.55, evidence_strength="strong"),
                        "vulnerability":  TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=14500, triggered_by="EVT_MYRTLE_KILLED",
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
                       confidence=0.95, inertia=0.8, established_at_fabula=12000, evidence_strength="strong"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=12000, triggered_by="EVT_TOM_DISCOVERS_AFFAIR",
                    traits={
                        "possessiveness": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=14800, triggered_by="EVT_TOM_TELLS_GEORGE",
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
            state_timeline=[
                EntityStateSnapshot(fabula_time=8200, triggered_by="EVT_UTT_JORDAN_REVEALS_HISTORY",
                    location_id="LOC_NEW_YORK_APT"),
            ],
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
                       confidence=0.7, inertia=0.5, established_at_fabula=4000, evidence_strength="moderate"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=5000, triggered_by="EVT_TOM_BREAKS_NOSE",
                    status="ill",
                    traits={
                        "discontent": TraitVector(value=0.95, inertia=0.8, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=14500, triggered_by="EVT_MYRTLE_KILLED",
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
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_GEORGE_SUSPECTS_AFFAIR",
                    traits={
                        "obliviousness": TraitVector(value=0.2, inertia=0.7, evidence_strength="strong"),
                        "rage":          TraitVector(value=0.7, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_MYRTLE",
                               perceived_state="my wife has been unfaithful — I shall lock her in",
                               confidence=0.9, inertia=0.7, established_at_fabula=13000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=14500, triggered_by="EVT_MYRTLE_KILLED",
                    traits={
                        "rage":  TraitVector(value=1.0, inertia=0.7, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="OBJ_ECKLEBURG_BILLBOARD",
                               perceived_state="God sees everything",
                               confidence=0.95, inertia=0.85, established_at_fabula=14500, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=14900, triggered_by="EVT_TOM_TELLS_GEORGE",
                    beliefs_added=[
                        Belief(target_id="ENT_GATSBY",
                               perceived_state="the man who owns the yellow car must be Myrtle's lover and her killer",
                               confidence=0.95, inertia=0.6, established_at_fabula=14900, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=15000, triggered_by="EVT_GATSBY_KILLED",
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
                EntityStateSnapshot(fabula_time=16000, triggered_by="EVT_FUNERAL",
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
            state_timeline=[
                EntityStateSnapshot(fabula_time=15000, triggered_by="EVT_GATSBY_KILLED",
                    location_id="LOC_NEW_YORK_APT"),
            ],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_GATSBY_MEETS_DAISY_1917", fabula_time=500, syuzhet_index=16,
                  event_type="outcome", actor_ids=["ENT_GATSBY", "ENT_DAISY"], target_ids=[],
                  description="In Louisville, 1917, Lieutenant Gatsby and the debutante Daisy fall in love before he ships out for the war."),
        EventNode(id="EVT_DAISY_MARRIES_TOM", fabula_time=800, syuzhet_index=18,
                  event_type="choice", actor_ids=["ENT_DAISY", "ENT_TOM"], target_ids=[],
                  description="With Gatsby still overseas, Daisy reluctantly marries Tom Buchanan, the wealthy Yale football star from Chicago."),
        EventNode(id="EVT_NICK_MOVES_EAST", fabula_time=1000, syuzhet_index=1,
                  event_type="choice", actor_ids=["ENT_NICK"], target_ids=[],
                  description="In spring 1922 Nick Carraway moves east to sell bonds and rents a bungalow in West Egg next to Gatsby's mansion."),
        EventNode(id="EVT_NICK_DINES_AT_BUCHANANS", fabula_time=2000, syuzhet_index=2,
                  event_type="choice", actor_ids=["ENT_NICK", "ENT_DAISY", "ENT_TOM"], target_ids=["ENT_JORDAN"],
                  description="Nick dines in East Egg with the Buchanans and meets Jordan Baker, who reveals that Tom keeps a mistress."),
        EventNode(id="EVT_NICK_SEES_GREEN_LIGHT", fabula_time=2500, syuzhet_index=4,
                  event_type="revelation", actor_ids=["ENT_NICK"], target_ids=["ENT_GATSBY"],
                  description="Returning home, Nick sees Gatsby alone on his lawn, stretching his arms toward a green light across the bay."),
        EventNode(id="EVT_TOM_TAKES_NICK_TO_NY", fabula_time=3500, syuzhet_index=5,
                  event_type="choice", actor_ids=["ENT_TOM", "ENT_NICK"], target_ids=["ENT_MYRTLE"],
                  description="Tom drags Nick into the city by way of Wilson's garage, picking up Myrtle for a party at his Manhattan apartment."),
        EventNode(id="EVT_TOM_BREAKS_NOSE", fabula_time=5000, syuzhet_index=6,
                  event_type="outcome", actor_ids=["ENT_TOM"], target_ids=["ENT_MYRTLE"],
                  description="At the apartment party Tom slaps Myrtle and breaks her nose for daring to mention Daisy's name."),
        EventNode(id="EVT_NICK_AT_GATSBYS_PARTY", fabula_time=7000, syuzhet_index=8,
                  event_type="choice", actor_ids=["ENT_NICK"], target_ids=["ENT_GATSBY"],
                  description="Nick attends one of Gatsby's vast Saturday-night parties; Gatsby singles him out and introduces himself."),
        EventNode(id="EVT_JORDAN_TELLS_NICK_HISTORY", fabula_time=8000, syuzhet_index=10,
                  event_type="revelation", actor_ids=["ENT_JORDAN"], target_ids=["ENT_NICK"],
                  description="At the Plaza, Jordan tells Nick that Gatsby and Daisy were lovers in 1917 and that the parties have all been bait to draw Daisy back."),
        EventNode(id="EVT_GATSBY_DAISY_REUNION", fabula_time=9000, syuzhet_index=12,
                  event_type="choice", actor_ids=["ENT_GATSBY", "ENT_DAISY"], target_ids=[],
                  description="Nick stages a tea-time reunion at his bungalow; Gatsby and Daisy embark on an affair."),
        EventNode(id="EVT_TOM_DISCOVERS_AFFAIR", fabula_time=12000, syuzhet_index=13,
                  event_type="revelation", actor_ids=["ENT_DAISY"], target_ids=["ENT_TOM"],
                  description="Daisy carelessly addresses Gatsby with unmistakable intimacy in front of Tom, who realises the affair."),
        EventNode(id="EVT_GEORGE_SUSPECTS_AFFAIR", fabula_time=12800, syuzhet_index=20,
                  event_type="revelation", actor_ids=["ENT_GEORGE"], target_ids=["ENT_MYRTLE"],
                  description="George finds evidence that Myrtle has a lover, locks her upstairs, and resolves to take her west."),
        EventNode(id="EVT_PLAZA_CONFRONTATION", fabula_time=13000, syuzhet_index=21,
                  event_type="choice", actor_ids=["ENT_GATSBY", "ENT_TOM", "ENT_DAISY"], target_ids=[],
                  description="In a Plaza Hotel suite Gatsby demands Daisy say she never loved Tom; Tom exposes Gatsby's bootlegging; Daisy retreats to Tom."),
        EventNode(id="EVT_MYRTLE_KILLED", fabula_time=14500, syuzhet_index=22,
                  event_type="outcome", actor_ids=["ENT_DAISY"], target_ids=["ENT_MYRTLE"],
                  description="Driving Gatsby's yellow car back to East Egg, Daisy strikes and kills Myrtle, who has run into the road; Gatsby resolves to take the blame."),
        EventNode(id="EVT_TOM_TELLS_GEORGE", fabula_time=14800, syuzhet_index=23,
                  event_type="choice", actor_ids=["ENT_TOM"], target_ids=["ENT_GEORGE"],
                  description="Tom tells the grieving George that Gatsby owns the yellow car, knowing George will assume Gatsby was Myrtle's lover and killer."),
        EventNode(id="EVT_GATSBY_KILLED", fabula_time=15000, syuzhet_index=24,
                  event_type="outcome", actor_ids=["ENT_GEORGE"], target_ids=["ENT_GATSBY", "ENT_GEORGE"],
                  description="George shoots Gatsby in his swimming pool and then kills himself."),
        EventNode(id="EVT_FUNERAL", fabula_time=16000, syuzhet_index=25,
                  event_type="outcome", actor_ids=["ENT_NICK", "ENT_HENRY_GATZ"], target_ids=["ENT_GATSBY"],
                  description="Almost no one attends Gatsby's funeral; only Nick and the late-arriving Henry Gatz mourn him."),
        EventNode(id="EVT_NICK_RETURNS_WEST", fabula_time=17000, syuzhet_index=26,
                  event_type="choice", actor_ids=["ENT_NICK"], target_ids=[],
                  description="Disgusted with the East, Nick decides to return to the Midwest; he encounters Tom on Fifth Avenue and reluctantly shakes his hand."),

        # ── UTTERANCES (on-page speech-acts) ──
        EventNode(id='EVT_UTT_JORDAN_REVEALS_MISTRESS', event_type='utterance',
                  description="At the Buchanan dinner Jordan confides to Nick that Tom keeps a mistress in the valley of ashes who brazenly telephones him at home.",
                  content="Tom's got some woman in New York — she lives down in the valley of ashes and has the gall to call him up at dinner-time.",
                  speaker_id='ENT_JORDAN', addressee_ids=['ENT_NICK'], actor_ids=['ENT_JORDAN'],
                  target_ids=['ENT_TOM', 'ENT_MYRTLE'],
                  via_channel_id=None, truth_value='true',
                  fabula_time=2200, syuzhet_index=3),
        EventNode(id='EVT_UTT_GATSBY_WAR_TALES', event_type='utterance',
                  description="Over a speakeasy lunch Gatsby tries to impress Nick with burnished tales of his war heroism and his Oxford days.",
                  content="I lived like a young rajah in all the capitals of Europe — and yes, I was at Oxford; in the war I was promoted to Major and every Allied government decorated me.",
                  speaker_id='ENT_GATSBY', addressee_ids=['ENT_NICK'], actor_ids=['ENT_GATSBY'],
                  target_ids=['ENT_GATSBY'],
                  via_channel_id=None, truth_value='false',
                  fabula_time=7500, syuzhet_index=7),
        EventNode(id='EVT_UTT_JORDAN_REVEALS_HISTORY', event_type='utterance',
                  description="At the Plaza Hotel Jordan tells Nick the secret history of Gatsby and Daisy and the true purpose of the parties across the bay.",
                  content="Gatsby and Daisy were lovers in Louisville back in 1917 before he shipped out — and the parties have all been bait to bring her over from East Egg.",
                  speaker_id='ENT_JORDAN', addressee_ids=['ENT_NICK'], actor_ids=['ENT_JORDAN'],
                  target_ids=['EVT_GATSBY_MEETS_DAISY_1917', 'EVT_DAISY_MARRIES_TOM'],
                  via_channel_id=None, truth_value='true',
                  fabula_time=8200, syuzhet_index=9),
        EventNode(id='EVT_UTT_DAISY_INTIMATE_ADDRESS', event_type='utterance',
                  description="Daisy carelessly addresses Gatsby with unmistakable intimacy in front of Tom, betraying the affair.",
                  content="Ah, you look so cool — you always look so cool; you resemble the advertisement of the man.",
                  speaker_id='ENT_DAISY', addressee_ids=['ENT_GATSBY'], actor_ids=['ENT_DAISY'],
                  target_ids=['EVT_GATSBY_DAISY_REUNION', 'ENT_GATSBY'],
                  via_channel_id=None, truth_value='performative',
                  fabula_time=12000, syuzhet_index=11),
        EventNode(id='EVT_UTT_GATSBY_DEMANDS_DENIAL', event_type='utterance',
                  description="In the Plaza suite Gatsby presses Daisy to renounce Tom outright and declare she never loved him.",
                  content="Just tell him the truth — that you never loved him — and it's all wiped out forever.",
                  speaker_id='ENT_GATSBY', addressee_ids=['ENT_DAISY', 'ENT_TOM'], actor_ids=['ENT_GATSBY'],
                  target_ids=['EVT_DAISY_MARRIES_TOM', 'ENT_DAISY', 'ENT_TOM'],
                  via_channel_id=None, truth_value='performative',
                  fabula_time=13000, syuzhet_index=14),
        EventNode(id='EVT_UTT_TOM_EXPOSES_BOOTLEG', event_type='utterance',
                  description="Tom destroys Gatsby in front of Daisy by exposing his fortune as bootlegging money funnelled through Wolfshiem.",
                  content="He and this Wolfshiem bought up a lot of side-street drug-stores and sold grain alcohol over the counter — that's just one of his little stunts.",
                  speaker_id='ENT_TOM', addressee_ids=['ENT_DAISY', 'ENT_GATSBY', 'ENT_NICK', 'ENT_JORDAN'], actor_ids=['ENT_TOM'],
                  target_ids=['ENT_GATSBY', 'ENT_MEYER_WOLFSHIEM'],
                  via_channel_id=None, truth_value='true',
                  fabula_time=13100, syuzhet_index=15),
        EventNode(id='EVT_UTT_GATSBY_CONFESSES_BLAME', event_type='utterance',
                  description="In the small hours after the accident Gatsby tells Nick that Daisy was driving but that he intends to take the blame for the killing.",
                  content="Of course I'll say I was driving. Daisy was at the wheel — but I'll take the blame for it.",
                  speaker_id='ENT_GATSBY', addressee_ids=['ENT_NICK'], actor_ids=['ENT_GATSBY'],
                  target_ids=['EVT_MYRTLE_KILLED', 'ENT_DAISY'],
                  via_channel_id=None, truth_value='true',
                  fabula_time=14600, syuzhet_index=17),
        EventNode(id='EVT_UTT_TOM_DIRECTS_GEORGE', event_type='utterance',
                  description="Tom tells the grieving George that the yellow car which struck Myrtle belongs to Gatsby — knowing George will infer Gatsby was the lover.",
                  content="That yellow car I drove down this afternoon wasn't mine — do you hear me? It was Gatsby's car.",
                  speaker_id='ENT_TOM', addressee_ids=['ENT_GEORGE'], actor_ids=['ENT_TOM'],
                  target_ids=['OBJ_YELLOW_CAR', 'ENT_GATSBY', 'EVT_MYRTLE_KILLED'],
                  via_channel_id=None, truth_value='true',
                  fabula_time=14850, syuzhet_index=19),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction ──
        CausalEdge(source_id="EVT_GATSBY_MEETS_DAISY_1917", target_id="EVT_DAISY_MARRIES_TOM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=500, propagation_delay=300),
        CausalEdge(source_id="EVT_NICK_MOVES_EAST", target_id="EVT_NICK_DINES_AT_BUCHANANS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1000, propagation_delay=1000),
        CausalEdge(source_id="EVT_NICK_DINES_AT_BUCHANANS", target_id="EVT_NICK_SEES_GREEN_LIGHT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=2000, propagation_delay=500),
        CausalEdge(source_id="EVT_NICK_DINES_AT_BUCHANANS", target_id="EVT_TOM_TAKES_NICK_TO_NY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2000, propagation_delay=1500),
        CausalEdge(source_id="EVT_TOM_TAKES_NICK_TO_NY", target_id="EVT_TOM_BREAKS_NOSE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3500, propagation_delay=1500),
        CausalEdge(source_id="EVT_NICK_MOVES_EAST", target_id="EVT_NICK_AT_GATSBYS_PARTY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000, propagation_delay=6000),
        CausalEdge(source_id="EVT_NICK_AT_GATSBYS_PARTY", target_id="EVT_JORDAN_TELLS_NICK_HISTORY",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=6.0, fabula_time=7000, propagation_delay=1000),
        CausalEdge(source_id="EVT_JORDAN_TELLS_NICK_HISTORY", target_id="EVT_GATSBY_DAISY_REUNION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=8000, propagation_delay=1000),
        CausalEdge(source_id="EVT_GATSBY_DAISY_REUNION", target_id="EVT_TOM_DISCOVERS_AFFAIR",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000, propagation_delay=3000),
        CausalEdge(source_id="EVT_TOM_DISCOVERS_AFFAIR", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=10.0, fabula_time=12000, propagation_delay=1000),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="EVT_GEORGE_SUSPECTS_AFFAIR",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=5000, propagation_delay=7800),
        CausalEdge(source_id="EVT_GEORGE_SUSPECTS_AFFAIR", target_id="EVT_MYRTLE_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=12800, propagation_delay=1700),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="EVT_MYRTLE_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000, propagation_delay=1500),
        CausalEdge(source_id="EVT_MYRTLE_KILLED", target_id="EVT_TOM_TELLS_GEORGE",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=9.0, fabula_time=14500, propagation_delay=300),
        CausalEdge(source_id="EVT_TOM_TELLS_GEORGE", target_id="EVT_GATSBY_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=14800, propagation_delay=200),
        CausalEdge(source_id="EVT_GATSBY_KILLED", target_id="EVT_FUNERAL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=15000, propagation_delay=1000),
        CausalEdge(source_id="EVT_FUNERAL", target_id="EVT_NICK_RETURNS_WEST",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=16000, propagation_delay=1000),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="EVT_GATSBY_DAISY_REUNION",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=800, propagation_delay=8200),

        # ── mutation ──
        CausalEdge(source_id="EVT_GATSBY_MEETS_DAISY_1917", target_id="ENT_GATSBY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=500,
                   trait_target="longing", trait_delta=0.5),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_GATSBY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=800,
                   trait_target="self_invention", trait_delta=0.6),
        CausalEdge(source_id="EVT_GATSBY_DAISY_REUNION", target_id="ENT_GATSBY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=9000,
                   trait_target="hope", trait_delta=0.2),
        CausalEdge(source_id="EVT_GATSBY_DAISY_REUNION", target_id="ENT_DAISY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=9000,
                   trait_target="rekindled_love", trait_delta=0.65),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_GATSBY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=13000,
                   trait_target="hope", trait_delta=-0.55),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_DAISY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000,
                   trait_target="rekindled_love", trait_delta=-0.55),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_DAISY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=13000,
                   trait_target="vulnerability", trait_delta=0.15),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="ENT_MYRTLE",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=5000,
                   trait_target="discontent", trait_delta=0.05),
        CausalEdge(source_id="EVT_GEORGE_SUSPECTS_AFFAIR", target_id="ENT_GEORGE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=12800,
                   trait_target="rage", trait_delta=0.6),
        CausalEdge(source_id="EVT_MYRTLE_KILLED", target_id="ENT_GEORGE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=14500,
                   trait_target="rage", trait_delta=0.3),
        CausalEdge(source_id="EVT_MYRTLE_KILLED", target_id="ENT_DAISY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=14500,
                   trait_target="carelessness", trait_delta=0.15),
        CausalEdge(source_id="EVT_GATSBY_KILLED", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=15000,
                   trait_target="disillusion", trait_delta=0.75),
        CausalEdge(source_id="EVT_GATSBY_KILLED", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=15000,
                   trait_target="moral_judgement", trait_delta=0.2),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_GATSBY_MEETS_DAISY_1917", target_id="ENT_GATSBY",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=500,
                   trait_target="affinity", trait_delta=1.0, rel_counterpart_id="ENT_DAISY"),
        CausalEdge(source_id="EVT_GATSBY_MEETS_DAISY_1917", target_id="ENT_DAISY",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=500,
                   trait_target="affinity", trait_delta=0.9, rel_counterpart_id="ENT_GATSBY"),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_GATSBY",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=800,
                   trait_target="affinity", trait_delta=-0.95, rel_counterpart_id="ENT_TOM"),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="ENT_MYRTLE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=5000,
                   trait_target="fear", trait_delta=0.45, rel_counterpart_id="ENT_TOM"),
        CausalEdge(source_id="EVT_TOM_DISCOVERS_AFFAIR", target_id="ENT_TOM",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=10.0, fabula_time=12000,
                   trait_target="affinity", trait_delta=-1.0, rel_counterpart_id="ENT_GATSBY"),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_DAISY",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000,
                   trait_target="affinity", trait_delta=-0.4, rel_counterpart_id="ENT_GATSBY"),
        CausalEdge(source_id="EVT_GEORGE_SUSPECTS_AFFAIR", target_id="ENT_GEORGE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=12800,
                   trait_target="affinity", trait_delta=-0.6, rel_counterpart_id="ENT_MYRTLE"),
        CausalEdge(source_id="EVT_TOM_TELLS_GEORGE", target_id="ENT_GEORGE",
                   causality_type="mutation_social", mechanism="informational", evidence_strength="strong",
                   causal_force=10.0, fabula_time=14800,
                   trait_target="affinity", trait_delta=-1.0, rel_counterpart_id="ENT_GATSBY"),

        # ── affordance_gate ──
        CausalEdge(source_id="OBJ_GREEN_LIGHT", target_id="EVT_NICK_SEES_GREEN_LIGHT",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2500),
        CausalEdge(source_id="OBJ_YELLOW_CAR", target_id="EVT_MYRTLE_KILLED",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=14500),
        CausalEdge(source_id="OBJ_YELLOW_CAR", target_id="EVT_TOM_TELLS_GEORGE",
                   causality_type="affordance_gate", mechanism="informational", evidence_strength="strong",
                   causal_force=8.0, fabula_time=14800),
        CausalEdge(source_id="OBJ_REVOLVER", target_id="EVT_GATSBY_KILLED",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=15000),
        CausalEdge(source_id="OBJ_GATSBYS_TELEPHONE", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="affordance_gate", mechanism="informational", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=13000),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_VALLEY_OF_ASHES", target_id="ENT_GEORGE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4000),
        CausalEdge(source_id="LOC_VALLEY_OF_ASHES", target_id="ENT_MYRTLE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4000),
        CausalEdge(source_id="LOC_PLAZA_SUITE", target_id="ENT_TOM",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=13000),
        CausalEdge(source_id="LOC_GATSBYS_MANSION", target_id="ENT_NICK",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=7000),

        # ── WORLD_ → Event ──
        CausalEdge(source_id="WORLD_OLD_VS_NEW_MONEY", target_id="EVT_DAISY_MARRIES_TOM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=800),
        CausalEdge(source_id="WORLD_OLD_VS_NEW_MONEY", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000),
        CausalEdge(source_id="WORLD_OLD_VS_NEW_MONEY", target_id="EVT_NICK_RETURNS_WEST",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=17000),
        CausalEdge(source_id="WORLD_PROHIBITION_BOOTLEG", target_id="EVT_NICK_AT_GATSBYS_PARTY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=7000),
        CausalEdge(source_id="WORLD_PROHIBITION_BOOTLEG", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=7.0, fabula_time=13000),
        CausalEdge(source_id="WORLD_PROHIBITION_BOOTLEG", target_id="EVT_FUNERAL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=16000),
        CausalEdge(source_id="WORLD_LOST_GENERATION", target_id="EVT_GATSBY_MEETS_DAISY_1917",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=500),
        CausalEdge(source_id="WORLD_LOST_GENERATION", target_id="EVT_NICK_MOVES_EAST",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_LOST_GENERATION", target_id="EVT_GATSBY_DAISY_REUNION",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=9000),
        CausalEdge(source_id="WORLD_LOST_GENERATION", target_id="EVT_NICK_RETURNS_WEST",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=17000),

        # ── orphan utterance wirings ──
        CausalEdge(source_id="EVT_NICK_DINES_AT_BUCHANANS", target_id="EVT_UTT_JORDAN_REVEALS_MISTRESS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2000, propagation_delay=200),
        CausalEdge(source_id="EVT_UTT_JORDAN_REVEALS_MISTRESS", target_id="EVT_TOM_TAKES_NICK_TO_NY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=2200, propagation_delay=1300),
        CausalEdge(source_id="EVT_UTT_GATSBY_WAR_TALES", target_id="EVT_UTT_JORDAN_REVEALS_HISTORY",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=7500, propagation_delay=700),
        CausalEdge(source_id="EVT_UTT_JORDAN_REVEALS_HISTORY", target_id="EVT_GATSBY_DAISY_REUNION",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=7.0, fabula_time=8200, propagation_delay=800),
        CausalEdge(source_id="EVT_UTT_DAISY_INTIMATE_ADDRESS", target_id="EVT_TOM_DISCOVERS_AFFAIR",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=12000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_GATSBY_DEMANDS_DENIAL", target_id="EVT_PLAZA_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_TOM_EXPOSES_BOOTLEG", target_id="EVT_MYRTLE_KILLED",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=13100, propagation_delay=1400),
        CausalEdge(source_id="EVT_UTT_GATSBY_CONFESSES_BLAME", target_id="EVT_NICK_RETURNS_WEST",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=14600, propagation_delay=2400),
        CausalEdge(source_id="EVT_UTT_TOM_DIRECTS_GEORGE", target_id="EVT_GATSBY_KILLED",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=10.0, fabula_time=14850, propagation_delay=150),
        # The Eckleburg billboard's blank god-eyes preside over the Valley
        # of Ashes; in George's broken theology they witness Myrtle's
        # death and license his vengeance. Modelled as an affordance gate
        # whose epistemic affordance ("God sees everything") supplies the
        # cosmic warrant for the Gatsby killing.
        CausalEdge(source_id="OBJ_ECKLEBURG_BILLBOARD", target_id="EVT_GATSBY_KILLED",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=14900),

        # ─── auto-patched mutation_social edges (per-axis coverage) ───
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_DAISY", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=800, propagation_delay=0),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_DAISY", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.1, mechanism="emotional", evidence_strength="strong", causal_force=5.0, fabula_time=13000, propagation_delay=0),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_TOM", rel_counterpart_id="ENT_DAISY", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=800, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_TAKES_NICK_TO_NY", target_id="ENT_TOM", rel_counterpart_id="ENT_MYRTLE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="ENT_TOM", rel_counterpart_id="ENT_MYRTLE", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.1, mechanism="physical", evidence_strength="strong", causal_force=4.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_TAKES_NICK_TO_NY", target_id="ENT_MYRTLE", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="affinity", trait_delta=0.7, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_GEORGE_SUSPECTS_AFFAIR", target_id="ENT_MYRTLE", rel_counterpart_id="ENT_GEORGE", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.4, mechanism="emotional", evidence_strength="strong", causal_force=5.0, fabula_time=12800, propagation_delay=0),
        CausalEdge(source_id="EVT_NICK_AT_GATSBYS_PARTY", target_id="ENT_NICK", rel_counterpart_id="ENT_GATSBY", causality_type="mutation_social", trait_target="affinity", trait_delta=0.4, mechanism="social", evidence_strength="strong", causal_force=5.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_GATSBY_KILLED", target_id="ENT_NICK", rel_counterpart_id="ENT_GATSBY", causality_type="mutation_social", trait_target="affinity", trait_delta=0.45, mechanism="emotional", evidence_strength="strong", causal_force=9.0, fabula_time=15000, propagation_delay=0),
        CausalEdge(source_id="EVT_NICK_AT_GATSBYS_PARTY", target_id="ENT_GATSBY", rel_counterpart_id="ENT_NICK", causality_type="mutation_social", trait_target="affinity", trait_delta=0.4, mechanism="social", evidence_strength="strong", causal_force=5.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_GATSBY_CONFESSES_BLAME", target_id="ENT_GATSBY", rel_counterpart_id="ENT_NICK", causality_type="mutation_social", trait_target="affinity", trait_delta=0.3, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=14600, propagation_delay=0),
        CausalEdge(source_id="EVT_NICK_DINES_AT_BUCHANANS", target_id="ENT_NICK", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.3, mechanism="social", evidence_strength="moderate", causal_force=3.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_TAKES_NICK_TO_NY", target_id="ENT_NICK", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.3, mechanism="psychological", evidence_strength="strong", causal_force=5.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_TOM_EXPOSES_BOOTLEG", target_id="ENT_NICK", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.25, mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=13100, propagation_delay=0),
        CausalEdge(source_id="EVT_NICK_DINES_AT_BUCHANANS", target_id="ENT_NICK", rel_counterpart_id="ENT_JORDAN", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="emotional", evidence_strength="moderate", causal_force=5.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_GATSBY_KILLED", target_id="ENT_NICK", rel_counterpart_id="ENT_JORDAN", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.2, mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=15000, propagation_delay=0),
        CausalEdge(source_id="EVT_NICK_DINES_AT_BUCHANANS", target_id="ENT_JORDAN", rel_counterpart_id="ENT_NICK", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="emotional", evidence_strength="moderate", causal_force=5.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_TOM_EXPOSES_BOOTLEG", target_id="ENT_MEYER_WOLFSHIEM", rel_counterpart_id="ENT_GATSBY", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=13100, propagation_delay=0),
        CausalEdge(source_id="EVT_FUNERAL", target_id="ENT_HENRY_GATZ", rel_counterpart_id="ENT_GATSBY", causality_type="mutation_social", trait_target="affinity", trait_delta=0.95, mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=16000, propagation_delay=0),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_DAISY", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="fear", trait_delta=0.2, mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=800, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_DISCOVERS_AFFAIR", target_id="ENT_DAISY", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="fear", trait_delta=0.15, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=12000, propagation_delay=0),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_DAISY", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.55, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=800, propagation_delay=0),
        CausalEdge(source_id="EVT_DAISY_MARRIES_TOM", target_id="ENT_TOM", rel_counterpart_id="ENT_DAISY", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.55, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=800, propagation_delay=0),
        CausalEdge(source_id="EVT_PLAZA_CONFRONTATION", target_id="ENT_TOM", rel_counterpart_id="ENT_GATSBY", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.6, mechanism="social", evidence_strength="strong", causal_force=8.0, fabula_time=13000, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_TAKES_NICK_TO_NY", target_id="ENT_TOM", rel_counterpart_id="ENT_MYRTLE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.6, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="ENT_TOM", rel_counterpart_id="ENT_MYRTLE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.25, mechanism="physical", evidence_strength="strong", causal_force=9.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_TAKES_NICK_TO_NY", target_id="ENT_MYRTLE", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.6, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_TOM_BREAKS_NOSE", target_id="ENT_MYRTLE", rel_counterpart_id="ENT_TOM", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.25, mechanism="physical", evidence_strength="strong", causal_force=9.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_TOM_EXPOSES_BOOTLEG", target_id="ENT_MEYER_WOLFSHIEM", rel_counterpart_id="ENT_GATSBY", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.7, mechanism="social", evidence_strength="moderate", causal_force=6.0, fabula_time=13100, propagation_delay=0),
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
    # Only standing communication capabilities live here. One-shot speech-acts
    # (Jordan's anecdote, Gatsby's confession, Tom's accusation, etc.) are
    # modelled as utterance events with via_channel_id=None — see EVT_UTT_*.
    channels={
        # Tom and Myrtle's affair telephone — recurring background calls ("his
        # mistress, who brazenly telephones him at his home"); not a single
        # speech-act but a standing capability throughout the affair.
        'CHN_TOM_MYRTLE_PHONE': Channel(
            id='CHN_TOM_MYRTLE_PHONE',
            name="Tom ↔ Myrtle Affair Telephone",
            medium='telephone',
            participant_ids=['ENT_TOM', 'ENT_MYRTLE'],
            directionality='duplex',
            intelligibility={},
            established_at_fabula=4000,
            terminated_at_fabula=14500,
            evidence_strength='strong',
        ),
        # Gatsby's wired-to-Chicago bootleg pipeline — the standing back-channel
        # to Wolfshiem and the drug-store front operation that funds the mansion.
        'CHN_GATSBY_BOOTLEG_PIPELINE': Channel(
            id='CHN_GATSBY_BOOTLEG_PIPELINE',
            name="Gatsby ↔ Wolfshiem Bootleg Pipeline",
            medium='classified_pipeline',
            participant_ids=['ENT_GATSBY', 'ENT_MEYER_WOLFSHIEM', 'OBJ_GATSBYS_TELEPHONE'],
            directionality='duplex',
            intelligibility={},
            established_at_fabula=2000,
            terminated_at_fabula=15000,
            evidence_strength='moderate',
        ),
    },

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
                WorldTraitSnapshot(fabula_time=15000, triggered_by="EVT_GATSBY_KILLED",
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
            affected_domains=["social"],
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
                "affinity": RelationshipMetric(value=1.0, inertia=0.6, evidence_strength="strong", last_updated_fabula=9000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_DAISY", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity": RelationshipMetric(value=0.55, inertia=0.45, evidence_strength="strong", last_updated_fabula=13000),
            },
        ),
        # Daisy ↔ Tom — wealthy marriage.
        RelationshipEdge(
            source_entity_id="ENT_DAISY", target_entity_id="ENT_TOM",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.55, evidence_strength="strong", last_updated_fabula=13000),
                "fear":          RelationshipMetric(value=0.35, inertia=0.2, evidence_strength="moderate", last_updated_fabula=12000),
                "power_dynamic": RelationshipMetric(value=-0.55, inertia=0.7, evidence_strength="strong", last_updated_fabula=13000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_TOM", target_entity_id="ENT_DAISY",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.55, evidence_strength="strong", last_updated_fabula=13000),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.7, evidence_strength="strong", last_updated_fabula=13000),
            },
        ),
        # Tom ↔ Gatsby — class hatred.
        RelationshipEdge(
            source_entity_id="ENT_TOM", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity":      RelationshipMetric(value=-0.95, inertia=0.5, evidence_strength="strong", last_updated_fabula=13000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.7, evidence_strength="strong", last_updated_fabula=13000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_GATSBY", target_entity_id="ENT_TOM",
            metrics={
                "affinity": RelationshipMetric(value=-0.85, inertia=0.5, evidence_strength="strong", last_updated_fabula=13000),
            },
        ),
        # Tom ↔ Myrtle — affair.
        RelationshipEdge(
            source_entity_id="ENT_TOM", target_entity_id="ENT_MYRTLE",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.5, evidence_strength="strong", last_updated_fabula=5000),
                "power_dynamic": RelationshipMetric(value=0.85, inertia=0.75, evidence_strength="strong", last_updated_fabula=5000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MYRTLE", target_entity_id="ENT_TOM",
            metrics={
                "affinity":      RelationshipMetric(value=0.7, inertia=0.5, evidence_strength="strong", last_updated_fabula=5000),
                "fear":          RelationshipMetric(value=0.45, inertia=0.2, evidence_strength="strong", last_updated_fabula=5000),
                "power_dynamic": RelationshipMetric(value=-0.85, inertia=0.75, evidence_strength="strong", last_updated_fabula=5000),
            },
        ),
        # Myrtle ↔ George — exhausted marriage.
        RelationshipEdge(
            source_entity_id="ENT_MYRTLE", target_entity_id="ENT_GEORGE",
            metrics={
                "affinity": RelationshipMetric(value=-0.4, inertia=0.55, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_GEORGE", target_entity_id="ENT_MYRTLE",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.65, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        # George → Gatsby — fatal mistake.
        RelationshipEdge(
            source_entity_id="ENT_GEORGE", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity": RelationshipMetric(value=-0.95, inertia=0.5, evidence_strength="strong", last_updated_fabula=14900),
            },
        ),
        # Nick ↔ Gatsby — admiration.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.55, evidence_strength="strong", last_updated_fabula=15000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_GATSBY", target_entity_id="ENT_NICK",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.55, evidence_strength="strong", last_updated_fabula=14600),
            },
        ),
        # Nick ↔ Tom — moral repudiation.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_TOM",
            metrics={
                "affinity": RelationshipMetric(value=-0.85, inertia=0.55, evidence_strength="strong", last_updated_fabula=17000),
            },
        ),
        # Nick ↔ Jordan — flirtation cooled.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_JORDAN",
            metrics={
                "affinity": RelationshipMetric(value=0.3, inertia=0.45, evidence_strength="moderate", last_updated_fabula=17000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JORDAN", target_entity_id="ENT_NICK",
            metrics={
                "affinity": RelationshipMetric(value=0.5, inertia=0.45, evidence_strength="moderate", last_updated_fabula=15000),
            },
        ),
        # Wolfshiem → Gatsby — patron.
        RelationshipEdge(
            source_entity_id="ENT_MEYER_WOLFSHIEM", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=7000),
                "power_dynamic": RelationshipMetric(value=0.7, inertia=0.75, evidence_strength="moderate", last_updated_fabula=7000),
            },
        ),
        # Henry Gatz → Gatsby — paternal pride.
        RelationshipEdge(
            source_entity_id="ENT_HENRY_GATZ", target_entity_id="ENT_GATSBY",
            metrics={
                "affinity": RelationshipMetric(value=0.95, inertia=0.8, evidence_strength="strong", last_updated_fabula=16000),
            },
        ),
    ],
)
