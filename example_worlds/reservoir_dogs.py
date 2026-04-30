"""Reservoir Dogs — high-fidelity WorldStateV1 test fixture.

Authored against the current ingestion prompts. Demonstrates all five
CausalEdge modalities, per-axis ``RelationshipMetric``, explicit
``evidence_strength`` everywhere, and named-latent WORLD_ traits
(criminal honour code, police infiltration) wired as common-cause
parents over the events they jointly drive. The two latents collide:
the Code demands silence and loyalty to the boss; the Infiltration
poisons every interaction with hidden allegiances, so that every
choice the crew makes is forced through both filters at once.
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
        "LOC_DINER": Location(
            name="Uncle Bob's Diner",
            description="Coffee-shop where the crew has breakfast before the heist; tipping debate exposes their personalities.",
            ambient_state={
                "camaraderie": AmbientVector(value=0.6, volatility=0.3, evidence_strength="moderate"),
                "fluorescent_banality": AmbientVector(value=0.85, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_WAREHOUSE": Location(
            name="Rendezvous Warehouse",
            description="Abandoned mortuary-warehouse where the surviving crew regroups after the botched heist; the slaughterhouse of the third act.",
            ambient_state={
                "tension": AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
                "danger": AmbientVector(value=0.8, volatility=0.4, evidence_strength="strong"),
                "blood_smell": AmbientVector(value=0.7, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_DIAMOND_STORE": Location(
            name="Karina's Diamond Store",
            description="The jewelry store targeted for the heist; a silent alarm trips and turns the job into a slaughter.",
            ambient_state={
                "danger": AmbientVector(value=0.95, volatility=0.3, evidence_strength="strong"),
                "panic": AmbientVector(value=0.9, volatility=0.5, evidence_strength="strong"),
            },
        ),
        "LOC_JOE_OFFICE": Location(
            name="Joe Cabot's Office",
            description="Crime boss Joe Cabot's office where the heist is planned and the crew is hand-picked from old contacts and prison parolees.",
            ambient_state={
                "authority": AmbientVector(value=0.8, volatility=0.2, evidence_strength="strong"),
                "old_school_loyalty": AmbientVector(value=0.75, volatility=0.15, evidence_strength="moderate"),
            },
        ),
        "LOC_ORANGE_CAR": Location(
            name="Getaway Car",
            description="The hijacked sedan in which White drives Orange from the diamond store, Orange bleeding out across the back seat.",
            ambient_state={
                "desperation": AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
                "siren_wail": AmbientVector(value=0.7, volatility=0.5, evidence_strength="moderate"),
            },
        ),
        "LOC_ORANGE_APARTMENT": Location(
            name="Mr Orange's Apartment",
            description="The cover apartment where undercover officer Freddy Newandyke is coached by Holdaway and rehearses his identity.",
            ambient_state={
                "deception": AmbientVector(value=0.75, volatility=0.4, evidence_strength="strong"),
                "rehearsal_anxiety": AmbientVector(value=0.7, volatility=0.4, evidence_strength="moderate"),
            },
        ),
    },

    # ── OBJECTS ────────────────────────────────────────────────────────
    objects={
        "OBJ_DIAMONDS": NarrativeObject(
            id="OBJ_DIAMONDS", name="Stolen Diamonds",
            location_id="LOC_DIAMOND_STORE", owner_id=None,
            properties={"state": "in_display_cases", "value": "high"},
            affordances=[Affordance(action="steal", target_type="Entity")],
        ),
        "OBJ_GUNS": NarrativeObject(
            id="OBJ_GUNS", name="Crew's Weapons",
            location_id="LOC_WAREHOUSE", owner_id=None,
            properties={"state": "loaded"},
            affordances=[
                Affordance(action="threaten", target_type="Entity"),
                Affordance(action="kill", target_type="Entity"),
            ],
        ),
        "OBJ_POLICE_BADGE": NarrativeObject(
            id="OBJ_POLICE_BADGE", name="Mr Orange's Police Badge",
            location_id="LOC_ORANGE_APARTMENT", owner_id="ENT_ORANGE",
            properties={"state": "hidden"},
            affordances=[Affordance(action="identify", target_type="Entity")],
        ),
        "OBJ_RAZOR": NarrativeObject(
            id="OBJ_RAZOR", name="Blonde's Straight Razor",
            location_id="LOC_WAREHOUSE", owner_id="ENT_BLONDE",
            properties={"state": "carried"},
            affordances=[
                Affordance(action="torture", target_type="Entity"),
                Affordance(action="mutilate", target_type="Entity"),
            ],
        ),
        "OBJ_RADIO": NarrativeObject(
            id="OBJ_RADIO", name="Warehouse Radio (K-Billy's Super Sounds of the 70s)",
            location_id="LOC_WAREHOUSE", owner_id=None,
            properties={"state": "playing", "song": "stuck_in_the_middle_with_you"},
            affordances=[Affordance(action="soundtrack_atrocity", target_type="Entity")],
        ),
        "OBJ_GASOLINE": NarrativeObject(
            id="OBJ_GASOLINE", name="Can of Gasoline",
            location_id="LOC_WAREHOUSE", owner_id="ENT_BLONDE",
            properties={"state": "full"},
            affordances=[Affordance(action="immolate", target_type="Entity")],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    entities={
        "ENT_WHITE": Entity(
            id="ENT_WHITE", name="Mr White (Larry Dimmick)",
            location_id="LOC_WAREHOUSE", status="healthy",
            traits={
                "loyalty":          TraitVector(value=0.8, inertia=0.55, evidence_strength="strong"),
                "professionalism":  TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
                "compassion":       TraitVector(value=0.6, inertia=0.4, evidence_strength="moderate"),
                "temper":           TraitVector(value=0.5, inertia=0.35, evidence_strength="moderate"),
                "paternal_instinct": TraitVector(value=0.7, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_ORANGE",
                       perceived_state="Orange is a good kid who can be saved; not a cop",
                       confidence=0.85, inertia=0.5, established_at_fabula=100, evidence_strength="strong"),
                Belief(target_id="ENT_BLONDE",
                       perceived_state="Blonde is a psychopath who blew the heist by shooting civilians",
                       confidence=0.85, inertia=0.5, established_at_fabula=500, evidence_strength="strong"),
                Belief(target_id="ENT_JOE",
                       perceived_state="Joe is an old friend whose judgment I trust — though employing Blonde was a mistake",
                       confidence=0.75, inertia=0.55, established_at_fabula=100, evidence_strength="moderate"),
            ],
            constants=["career_criminal", "joes_old_friend"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=400, triggered_by="EVT_ORANGE_SHOT",
                    location_id="LOC_ORANGE_CAR",
                    traits={
                        "compassion":        TraitVector(value=0.8, inertia=0.55, evidence_strength="strong"),
                        "paternal_instinct": TraitVector(value=0.9, inertia=0.65, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=800, triggered_by="EVT_MEXICAN_STANDOFF",
                    status="injured",
                    traits={
                        "temper": TraitVector(value=0.85, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=900, triggered_by="EVT_ORANGE_REVEALED",
                    traits={
                        "loyalty":    TraitVector(value=0.2, inertia=0.65, evidence_strength="strong"),
                        "compassion": TraitVector(value=0.3, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_ORANGE"],
                    beliefs_added=[
                        Belief(target_id="ENT_ORANGE",
                               perceived_state="he was the rat the whole time; my paternal love was a lie",
                               confidence=0.95, inertia=0.7, established_at_fabula=900, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_WHITE_KILLS_ORANGE",
                    status="dead", location_id="LOC_WAREHOUSE"),
            ],
        ),
        "ENT_ORANGE": Entity(
            id="ENT_ORANGE", name="Mr Orange (Freddy Newandyke)",
            location_id="LOC_WAREHOUSE", status="healthy",
            traits={
                "deceit":   TraitVector(value=0.65, inertia=0.4, evidence_strength="strong"),
                "guilt":    TraitVector(value=0.5, inertia=0.4, evidence_strength="moderate"),
                "courage":  TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                "duty":     TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "pain":     TraitVector(value=0.1, inertia=0.3, evidence_strength="weak"),
            },
            beliefs=[
                Belief(target_id="ENT_WHITE",
                       perceived_state="White genuinely cares about me — which makes the betrayal worse",
                       confidence=0.85, inertia=0.5, established_at_fabula=400, evidence_strength="strong"),
                Belief(target_id="ENT_HOLDAWAY",
                       perceived_state="my handler will extract me if I keep cover until the bust",
                       confidence=0.8, inertia=0.6, established_at_fabula=100, evidence_strength="strong"),
            ],
            constants=["undercover_cop", "uses_alias_freddy_newandyke"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=400, triggered_by="EVT_ORANGE_SHOT",
                    status="injured", location_id="LOC_ORANGE_CAR",
                    traits={
                        "pain":   TraitVector(value=0.95, inertia=0.4, evidence_strength="strong"),
                        "guilt":  TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=700, triggered_by="EVT_ORANGE_KILLS_BLONDE",
                    traits={
                        "courage": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                        "duty":    TraitVector(value=0.95, inertia=0.75, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=800, triggered_by="EVT_MEXICAN_STANDOFF",
                    status="injured",
                    traits={
                        "pain": TraitVector(value=1.0, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=900, triggered_by="EVT_ORANGE_REVEALED",
                    traits={
                        "deceit": TraitVector(value=0.1, inertia=0.55, evidence_strength="strong"),
                        "guilt":  TraitVector(value=0.95, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_WHITE",
                               perceived_state="he loves me enough to risk dying for me; he deserves the truth",
                               confidence=0.95, inertia=0.7, established_at_fabula=900, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_WHITE_KILLS_ORANGE",
                    status="dead"),
            ],
        ),
        "ENT_BLONDE": Entity(
            id="ENT_BLONDE", name="Mr Blonde (Vic Vega)",
            location_id="LOC_DIAMOND_STORE", status="healthy",
            traits={
                "sadism":          TraitVector(value=0.95, inertia=0.75, evidence_strength="strong"),
                "loyalty_to_joe":  TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
                "calm":            TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "unpredictability": TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_JOE",
                       perceived_state="Joe took care of me while I did time; I owe him absolutely",
                       confidence=0.95, inertia=0.75, established_at_fabula=50, evidence_strength="strong"),
            ],
            constants=["ex_convict", "vega_brother", "psychopath"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=300, triggered_by="EVT_HEIST_GOES_WRONG",
                    location_id="LOC_DIAMOND_STORE",
                    traits={
                        "sadism": TraitVector(value=1.0, inertia=0.8, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=600, triggered_by="EVT_BLONDE_TORTURES_COP",
                    location_id="LOC_WAREHOUSE"),
                EntityStateSnapshot(fabula_time=700, triggered_by="EVT_ORANGE_KILLS_BLONDE",
                    status="dead"),
            ],
        ),
        "ENT_PINK": Entity(
            id="ENT_PINK", name="Mr Pink",
            location_id="LOC_WAREHOUSE", status="healthy",
            traits={
                "self_preservation": TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
                "professionalism":   TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "paranoia":          TraitVector(value=0.8, inertia=0.55, evidence_strength="strong"),
                "stinginess":        TraitVector(value=0.7, inertia=0.6, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_JOE",
                       perceived_state="someone set us up — possibly from the inside",
                       confidence=0.85, inertia=0.55, established_at_fabula=500, evidence_strength="strong"),
            ],
            constants=["paranoid_neurotic", "tipping_refusenik"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=500, triggered_by="EVT_WAREHOUSE_REGROUP",
                    traits={
                        "paranoia": TraitVector(value=0.95, inertia=0.65, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_BLONDE",
                               perceived_state="Joe should never have hired this lunatic",
                               confidence=0.85, inertia=0.5, established_at_fabula=500, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=800, triggered_by="EVT_MEXICAN_STANDOFF",
                    location_id="LOC_WAREHOUSE",
                    traits={
                        "self_preservation": TraitVector(value=1.0, inertia=0.75, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_JOE": Entity(
            id="ENT_JOE", name="Joe Cabot",
            location_id="LOC_JOE_OFFICE", status="healthy",
            traits={
                "authority":      TraitVector(value=0.95, inertia=0.75, evidence_strength="strong"),
                "suspicion":      TraitVector(value=0.7, inertia=0.55, evidence_strength="strong"),
                "ruthlessness":   TraitVector(value=0.85, inertia=0.65, evidence_strength="strong"),
                "instinct":       TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_BLONDE",
                       perceived_state="Blonde is loyal — he did four years rather than name me",
                       confidence=0.95, inertia=0.7, established_at_fabula=50, evidence_strength="strong"),
            ],
            constants=["crime_boss", "old_school"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=800, triggered_by="EVT_MEXICAN_STANDOFF",
                    beliefs_added=[
                        Belief(target_id="ENT_ORANGE",
                               perceived_state="Orange is the rat — my instinct never lies",
                               confidence=0.95, inertia=0.7, established_at_fabula=800, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=800, triggered_by="EVT_MEXICAN_STANDOFF",
                    status="dead", location_id="LOC_WAREHOUSE"),
            ],
        ),
        "ENT_EDDIE": Entity(
            id="ENT_EDDIE", name="Nice Guy Eddie Cabot",
            location_id="LOC_JOE_OFFICE", status="healthy",
            traits={
                "loyalty_to_father": TraitVector(value=0.95, inertia=0.75, evidence_strength="strong"),
                "volatility":        TraitVector(value=0.75, inertia=0.5, evidence_strength="strong"),
                "bravado":           TraitVector(value=0.8, inertia=0.55, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_JOE",
                       perceived_state="Dad is always right; his judgment defines mine",
                       confidence=0.95, inertia=0.8, established_at_fabula=50, evidence_strength="strong"),
                Belief(target_id="ENT_BLONDE",
                       perceived_state="Vic is family — he proved his loyalty in prison",
                       confidence=0.9, inertia=0.7, established_at_fabula=50, evidence_strength="strong"),
            ],
            constants=["joes_son"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=800, triggered_by="EVT_MEXICAN_STANDOFF",
                    status="dead", location_id="LOC_WAREHOUSE"),
            ],
        ),
        "ENT_MARVIN": Entity(
            id="ENT_MARVIN", name="Officer Marvin Nash",
            location_id="LOC_DIAMOND_STORE", status="healthy",
            traits={
                "courage": TraitVector(value=0.6, inertia=0.4, evidence_strength="moderate"),
                "fear":    TraitVector(value=0.3, inertia=0.3, evidence_strength="weak"),
                "duty":    TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_ORANGE",
                       perceived_state="this man is one of ours; I will not blow his cover even under torture",
                       confidence=0.95, inertia=0.75, established_at_fabula=300, evidence_strength="strong"),
            ],
            constants=["police_officer", "lapd"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=500, triggered_by="EVT_BLONDE_TORTURES_COP",
                    status="injured", location_id="LOC_WAREHOUSE",
                    traits={
                        "fear": TraitVector(value=0.95, inertia=0.55, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=600, triggered_by="EVT_BLONDE_TORTURES_COP",
                    status="injured",
                    traits={
                        "fear": TraitVector(value=1.0, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=750, triggered_by="EVT_EDDIE_KILLS_NASH",
                    status="dead"),
            ],
        ),
        "ENT_BROWN": Entity(
            id="ENT_BROWN", name="Mr Brown",
            location_id="LOC_DIAMOND_STORE", status="healthy",
            traits={
                "glibness":        TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                "professionalism": TraitVector(value=0.65, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[],
            constants=["pseudo_philosopher"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=300, triggered_by="EVT_HEIST_GOES_WRONG",
                    status="dead", location_id="LOC_ORANGE_CAR"),
            ],
        ),
        "ENT_BLUE": Entity(
            id="ENT_BLUE", name="Mr Blue",
            location_id="LOC_DIAMOND_STORE", status="healthy",
            traits={
                "composure": TraitVector(value=0.85, inertia=0.65, evidence_strength="strong"),
                "loyalty":   TraitVector(value=0.75, inertia=0.55, evidence_strength="moderate"),
            },
            beliefs=[],
            constants=["even_tempered", "veteran_thief"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=750, triggered_by="EVT_BLUE_KILLED_OFFSCREEN",
                    status="dead"),
            ],
        ),
        "ENT_HOLDAWAY": Entity(
            id="ENT_HOLDAWAY", name="Sergeant Holdaway",
            location_id="LOC_ORANGE_APARTMENT", status="healthy",
            traits={
                "professionalism": TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
                "protectiveness":  TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
                "shrewdness":      TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_ORANGE",
                       perceived_state="Orange must commit to the cover totally — performance, not memorisation, is what survives a room of pros",
                       confidence=0.9, inertia=0.7, established_at_fabula=100, evidence_strength="strong"),
            ],
            constants=["police_handler", "lapd_sergeant"],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_HEIST_PLANNED", fabula_time=100, syuzhet_index=1,
                  event_type="choice",
                  actor_ids=["ENT_JOE", "ENT_EDDIE"],
                  target_ids=["ENT_WHITE", "ENT_ORANGE", "ENT_BLONDE", "ENT_PINK", "ENT_BROWN", "ENT_BLUE"],
                  description="Joe Cabot and Eddie assemble six experienced robbers — strangers to each other — under colour-coded aliases to rob Karina's Diamond Store."),
        EventNode(id="EVT_DINER_BREAKFAST", fabula_time=200, syuzhet_index=2,
                  event_type="outcome",
                  actor_ids=["ENT_WHITE", "ENT_ORANGE", "ENT_BLONDE", "ENT_PINK", "ENT_BROWN", "ENT_BLUE", "ENT_JOE", "ENT_EDDIE"],
                  target_ids=[],
                  description="The crew has breakfast at Uncle Bob's; the tipping debate exposes Pink's stinginess and the precarious cohesion of strangers."),
        EventNode(id="EVT_HEIST_GOES_WRONG", fabula_time=300, syuzhet_index=3,
                  event_type="outcome",
                  actor_ids=["ENT_BLONDE"],
                  target_ids=["ENT_WHITE", "ENT_ORANGE", "ENT_PINK", "ENT_BROWN"],
                  description="A silent alarm trips; Blonde starts murdering bystanders; police arrive immediately and Brown is killed in the getaway."),
        EventNode(id="EVT_ORANGE_SHOT", fabula_time=400, syuzhet_index=4,
                  event_type="outcome",
                  actor_ids=[],
                  target_ids=["ENT_ORANGE"],
                  description="Hijacking a car, Orange is shot in the abdomen by the panicked civilian driver; he kills her in return and bleeds out across the back seat as White drives him to the warehouse."),
        EventNode(id="EVT_WAREHOUSE_REGROUP", fabula_time=500, syuzhet_index=5,
                  event_type="choice",
                  actor_ids=["ENT_WHITE", "ENT_PINK"],
                  target_ids=["ENT_ORANGE"],
                  description="White and Pink regroup at the warehouse; Pink declares the job a setup and refuses to call a doctor; the two draw on each other before Blonde arrives with a kidnapped cop."),
        EventNode(id="EVT_BLONDE_TORTURES_COP", fabula_time=600, syuzhet_index=6,
                  event_type="choice",
                  actor_ids=["ENT_BLONDE"],
                  target_ids=["ENT_MARVIN"],
                  description="Left alone with Marvin Nash, Blonde slashes the cop's face and severs his ear with a straight razor while 'Stuck in the Middle with You' plays on the radio, and prepares to set him on fire."),
        EventNode(id="EVT_ORANGE_KILLS_BLONDE", fabula_time=700, syuzhet_index=7,
                  event_type="choice",
                  actor_ids=["ENT_ORANGE"],
                  target_ids=["ENT_BLONDE"],
                  description="Bleeding Orange empties his pistol into Blonde to save Nash, breaking cover for the audience and revealing his loyalties to the cop he was protecting."),
        EventNode(id="EVT_BLUE_KILLED_OFFSCREEN", fabula_time=750, syuzhet_index=8,
                  event_type="outcome",
                  actor_ids=[],
                  target_ids=["ENT_BLUE"],
                  description="Mr Blue is killed by the police in flight from the heist — reported by Joe at the warehouse, never shown."),
        EventNode(id="EVT_EDDIE_KILLS_NASH", fabula_time=760, syuzhet_index=9,
                  event_type="choice",
                  actor_ids=["ENT_EDDIE"],
                  target_ids=["ENT_MARVIN"],
                  description="Returning to the warehouse, Eddie executes the wounded cop to silence him, refusing to believe Orange's story that Blonde planned to murder them."),
        EventNode(id="EVT_MEXICAN_STANDOFF", fabula_time=800, syuzhet_index=10,
                  event_type="outcome",
                  actor_ids=["ENT_JOE", "ENT_EDDIE", "ENT_WHITE"],
                  target_ids=["ENT_ORANGE"],
                  description="Joe arrives, names Orange as the rat, and goes to execute him; White intervenes at gunpoint; Eddie aims at White; all three fire and Joe, Eddie, and (mortally) White fall."),
        EventNode(id="EVT_ORANGE_REVEALED", fabula_time=900, syuzhet_index=11,
                  event_type="revelation",
                  actor_ids=["ENT_ORANGE"],
                  target_ids=["ENT_WHITE"],
                  description="Cradled in White's arms, the dying Orange confesses that he is an undercover police officer, retroactively reframing every act of paternal tenderness."),
        EventNode(id="EVT_WHITE_KILLS_ORANGE", fabula_time=1000, syuzhet_index=12,
                  event_type="choice",
                  actor_ids=["ENT_WHITE"],
                  target_ids=["ENT_ORANGE"],
                  description="As LAPD storm the warehouse, White presses his pistol to Orange's head and pulls the trigger — and is cut down by police gunfire in the same beat."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction ──
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="EVT_DINER_BREAKFAST",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=100, propagation_delay=100),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="EVT_HEIST_GOES_WRONG",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=200, propagation_delay=100),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="EVT_ORANGE_SHOT",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=300, propagation_delay=100),
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="EVT_WAREHOUSE_REGROUP",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=400, propagation_delay=100),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="EVT_BLONDE_TORTURES_COP",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=500, propagation_delay=100),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="EVT_ORANGE_KILLS_BLONDE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=600, propagation_delay=100),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="EVT_EDDIE_KILLS_NASH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=700, propagation_delay=60),
        CausalEdge(source_id="EVT_EDDIE_KILLS_NASH", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=760, propagation_delay=40),
        CausalEdge(source_id="EVT_BLUE_KILLED_OFFSCREEN", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=750, propagation_delay=50),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="EVT_ORANGE_REVEALED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=800, propagation_delay=100),
        CausalEdge(source_id="EVT_ORANGE_REVEALED", target_id="EVT_WHITE_KILLS_ORANGE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=900, propagation_delay=100),

        # ── mutation ──
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="ENT_ORANGE",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=400,
                   trait_target="pain", trait_delta=0.85),
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="ENT_WHITE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=400,
                   trait_target="paternal_instinct", trait_delta=0.2),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_MARVIN",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=600,
                   trait_target="fear", trait_delta=0.7),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="ENT_ORANGE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=700,
                   trait_target="duty", trait_delta=0.1),
        CausalEdge(source_id="EVT_ORANGE_REVEALED", target_id="ENT_WHITE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=900,
                   trait_target="loyalty", trait_delta=-0.6),
        CausalEdge(source_id="EVT_ORANGE_REVEALED", target_id="ENT_ORANGE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=900,
                   trait_target="guilt", trait_delta=0.45),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="ENT_BLONDE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=300,
                   trait_target="sadism", trait_delta=0.05),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_PINK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=500,
                   trait_target="paranoia", trait_delta=0.15),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_ORANGE_REVEALED", target_id="ENT_WHITE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=10.0, fabula_time=900,
                   trait_target="affinity", trait_delta=-0.95, rel_counterpart_id="ENT_ORANGE"),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="ENT_WHITE",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=300,
                   trait_target="affinity", trait_delta=-0.75, rel_counterpart_id="ENT_BLONDE"),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="ENT_ORANGE",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=700,
                   trait_target="affinity", trait_delta=0.25, rel_counterpart_id="ENT_WHITE"),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="ENT_ORANGE",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=700,
                   trait_target="affinity", trait_delta=0.6, rel_counterpart_id="ENT_MARVIN"),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="ENT_WHITE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=800,
                   trait_target="affinity", trait_delta=-0.85, rel_counterpart_id="ENT_JOE"),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="ENT_JOE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=800,
                   trait_target="affinity", trait_delta=-0.9, rel_counterpart_id="ENT_WHITE"),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_MARVIN",
                   causality_type="mutation_social", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=600,
                   trait_target="fear", trait_delta=0.8, rel_counterpart_id="ENT_BLONDE"),

        # ── affordance_gate ──
        CausalEdge(source_id="ENT_JOE", target_id="EVT_HEIST_PLANNED",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=100),
        CausalEdge(source_id="ENT_BLONDE", target_id="EVT_HEIST_GOES_WRONG",
                   causality_type="affordance_gate", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=300),
        CausalEdge(source_id="OBJ_GUNS", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=800),
        CausalEdge(source_id="OBJ_RAZOR", target_id="EVT_BLONDE_TORTURES_COP",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=600),
        CausalEdge(source_id="OBJ_GASOLINE", target_id="EVT_ORANGE_KILLS_BLONDE",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=700),
        CausalEdge(source_id="OBJ_POLICE_BADGE", target_id="EVT_ORANGE_REVEALED",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=900),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_WAREHOUSE", target_id="ENT_PINK",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=500),
        CausalEdge(source_id="LOC_WAREHOUSE", target_id="ENT_WHITE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=500),
        CausalEdge(source_id="LOC_DIAMOND_STORE", target_id="ENT_BLONDE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=300),
        CausalEdge(source_id="LOC_ORANGE_CAR", target_id="ENT_ORANGE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=400),

        # ── WORLD_ → Event (named-latent common-cause wiring) ──
        CausalEdge(source_id="WORLD_CRIMINAL_CODE", target_id="EVT_HEIST_PLANNED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=100),
        CausalEdge(source_id="WORLD_CRIMINAL_CODE", target_id="EVT_BLONDE_TORTURES_COP",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=600),
        CausalEdge(source_id="WORLD_CRIMINAL_CODE", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=800),
        CausalEdge(source_id="WORLD_CRIMINAL_CODE", target_id="EVT_WHITE_KILLS_ORANGE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_POLICE_INFILTRATION", target_id="EVT_HEIST_GOES_WRONG",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=300),
        CausalEdge(source_id="WORLD_POLICE_INFILTRATION", target_id="EVT_ORANGE_KILLS_BLONDE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=700),
        CausalEdge(source_id="WORLD_POLICE_INFILTRATION", target_id="EVT_ORANGE_REVEALED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=900),
        CausalEdge(source_id="WORLD_POLICE_INFILTRATION", target_id="EVT_WHITE_KILLS_ORANGE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1000),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_JOE_OFFICE", target_id="LOC_DINER"),
        SpatialEdge(source_id="LOC_DINER", target_id="LOC_DIAMOND_STORE"),
        SpatialEdge(source_id="LOC_DIAMOND_STORE", target_id="LOC_ORANGE_CAR"),
        SpatialEdge(source_id="LOC_ORANGE_CAR", target_id="LOC_WAREHOUSE"),
        SpatialEdge(source_id="LOC_ORANGE_APARTMENT", target_id="LOC_JOE_OFFICE"),
        SpatialEdge(source_id="LOC_JOE_OFFICE", target_id="LOC_WAREHOUSE"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    information_topology=[
        InformationEdge(source_id="ENT_HOLDAWAY", target_ids=["ENT_ORANGE"],
                        medium="undercover_briefing", is_encrypted=True,
                        established_at_fabula=50, terminated_at_fabula=300,
                        discovered_at_syuzhet=1, evidence_strength="strong"),
        InformationEdge(source_id="ENT_ORANGE", target_ids=["ENT_HOLDAWAY"],
                        medium="undercover_report", is_encrypted=True,
                        established_at_fabula=50, terminated_at_fabula=300,
                        discovered_at_syuzhet=1, evidence_strength="strong"),
        InformationEdge(source_id="ENT_JOE", target_ids=["ENT_EDDIE", "ENT_WHITE"],
                        medium="planning_briefing", is_encrypted=True,
                        established_at_fabula=100, terminated_at_fabula=100,
                        discovered_at_syuzhet=1, evidence_strength="moderate"),
        InformationEdge(source_id="ENT_PINK", target_ids=["ENT_WHITE"],
                        medium="argument", is_encrypted=False,
                        established_at_fabula=500, terminated_at_fabula=500,
                        discovered_at_syuzhet=5, evidence_strength="strong"),
        InformationEdge(source_id="OBJ_RADIO", target_ids=["ENT_BLONDE", "ENT_MARVIN"],
                        medium="radio_song", is_encrypted=False,
                        established_at_fabula=600, terminated_at_fabula=700,
                        discovered_at_syuzhet=6, evidence_strength="strong"),
        InformationEdge(source_id="ENT_ORANGE", target_ids=["ENT_MARVIN"],
                        medium="badge_recognition", is_encrypted=True,
                        established_at_fabula=300, terminated_at_fabula=750,
                        discovered_at_syuzhet=7, evidence_strength="strong"),
        InformationEdge(source_id="ENT_ORANGE", target_ids=["ENT_WHITE", "ENT_PINK", "ENT_EDDIE"],
                        medium="false_accusation", is_encrypted=False,
                        established_at_fabula=750, terminated_at_fabula=760,
                        discovered_at_syuzhet=9, evidence_strength="moderate"),
        InformationEdge(source_id="ENT_JOE", target_ids=["ENT_WHITE", "ENT_EDDIE"],
                        medium="public_accusation", is_encrypted=False,
                        established_at_fabula=800, terminated_at_fabula=800,
                        discovered_at_syuzhet=10, evidence_strength="strong"),
        InformationEdge(source_id="ENT_ORANGE", target_ids=["ENT_WHITE"],
                        medium="deathbed_confession", is_encrypted=False,
                        established_at_fabula=900, terminated_at_fabula=900,
                        discovered_at_syuzhet=11, evidence_strength="strong"),
    ],

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_CRIMINAL_CODE": GlobalTrait(
            id="WORLD_CRIMINAL_CODE",
            name="Criminal Honour Code",
            description="The unwritten code among professional criminals: never rat, maintain professionalism, defer to the boss, and pay any debt of loyalty in blood. Operates as a common-cause parent over Joe's hiring of the loyal Blonde, the Mexican Standoff, and White's final execution of Orange.",
            category="social_structure",
            magnitude=TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=500, triggered_by="EVT_WAREHOUSE_REGROUP",
                    magnitude=TraitVector(value=0.55, inertia=0.5, evidence_strength="strong"),
                    description="Mutual suspicion and the drawn guns at the warehouse fracture the code from within."),
                WorldTraitSnapshot(fabula_time=1000, triggered_by="EVT_WHITE_KILLS_ORANGE",
                    magnitude=TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                    description="White's execution of Orange reaffirms the code at the price of his own life — the rat dies, no matter the cost."),
            ],
        ),
        "WORLD_POLICE_INFILTRATION": GlobalTrait(
            id="WORLD_POLICE_INFILTRATION",
            name="Undercover Infiltration",
            description="An undercover LAPD officer embedded inside the crew, poisoning every interaction with hidden allegiance and the threat of exposure. Operates as a common-cause parent over the heist's collapse, Orange's choice to kill Blonde, and the deathbed revelation that retroactively reframes every act of fellowship.",
            category="governance",
            magnitude=TraitVector(value=0.65, inertia=0.85, evidence_strength="strong"),
            affected_domains=["epistemic", "psychological", "social"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=900, triggered_by="EVT_ORANGE_REVEALED",
                    magnitude=TraitVector(value=1.0, inertia=0.95, evidence_strength="strong"),
                    description="Orange's identity as a cop is spoken aloud, retroactively colouring every prior interaction in the film."),
            ],
        ),
    },

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────
    social_topology=[
        # White ↔ Orange — paternal bond shattered by revelation.
        RelationshipEdge(
            source_entity_id="ENT_WHITE", target_entity_id="ENT_ORANGE",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.5, evidence_strength="strong", last_updated_fabula=900),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.65, evidence_strength="moderate", last_updated_fabula=400),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_WHITE",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=900),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=900),
                "power_dynamic": RelationshipMetric(value=-0.55, inertia=0.65, evidence_strength="moderate", last_updated_fabula=400),
            },
        ),
        # White ↔ Blonde — professional contempt → open hostility.
        RelationshipEdge(
            source_entity_id="ENT_WHITE", target_entity_id="ENT_BLONDE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=300),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=300),
            },
        ),
        # White ↔ Pink — wary professional alliance.
        RelationshipEdge(
            source_entity_id="ENT_WHITE", target_entity_id="ENT_PINK",
            metrics={
                "affinity":      RelationshipMetric(value=0.45, inertia=0.45, evidence_strength="moderate", last_updated_fabula=500),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_PINK", target_entity_id="ENT_WHITE",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.45, evidence_strength="moderate", last_updated_fabula=500),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=500),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=500),
            },
        ),
        # White ↔ Joe — old friendship destroyed at gunpoint.
        RelationshipEdge(
            source_entity_id="ENT_WHITE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.55, evidence_strength="strong", last_updated_fabula=800),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.7, evidence_strength="strong", last_updated_fabula=800),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_WHITE",
            metrics={
                "affinity":      RelationshipMetric(value=0.45, inertia=0.55, evidence_strength="strong", last_updated_fabula=800),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.7, evidence_strength="strong", last_updated_fabula=800),
            },
        ),
        # Blonde ↔ Joe — patron-client loyalty (the prison debt).
        RelationshipEdge(
            source_entity_id="ENT_BLONDE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.55, evidence_strength="strong", last_updated_fabula=50),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=50),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_BLONDE",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.55, evidence_strength="strong", last_updated_fabula=50),
                "power_dynamic": RelationshipMetric(value=0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=50),
            },
        ),
        # Joe ↔ Orange — earned trust → fatal accusation.
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_ORANGE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.5, inertia=0.5, evidence_strength="strong", last_updated_fabula=800),
                "power_dynamic": RelationshipMetric(value=0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=800),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.45, evidence_strength="moderate", last_updated_fabula=200),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=800),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=800),
            },
        ),
        # Eddie ↔ Joe — filial loyalty.
        RelationshipEdge(
            source_entity_id="ENT_EDDIE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.6, evidence_strength="strong", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_EDDIE",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.6, evidence_strength="strong", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        # Orange ↔ Blonde — natural enemies.
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_BLONDE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.85, inertia=0.5, evidence_strength="strong", last_updated_fabula=700),
                "fear":          RelationshipMetric(value=0.25, inertia=0.2, evidence_strength="strong", last_updated_fabula=600),
            },
        ),
        # Orange ↔ Marvin — buried solidarity.
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_MARVIN",
            metrics={
                "affinity": RelationshipMetric(value=0.6, inertia=0.55, evidence_strength="strong", last_updated_fabula=700),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MARVIN", target_entity_id="ENT_ORANGE",
            metrics={
                "affinity": RelationshipMetric(value=0.55, inertia=0.55, evidence_strength="strong", last_updated_fabula=300),
            },
        ),
        # Marvin ↔ Blonde — torturer/victim.
        RelationshipEdge(
            source_entity_id="ENT_MARVIN", target_entity_id="ENT_BLONDE",
            metrics={
                "affinity": RelationshipMetric(value=-0.95, inertia=0.5, evidence_strength="strong", last_updated_fabula=600),
                "fear":     RelationshipMetric(value=1.0, inertia=0.25, evidence_strength="strong", last_updated_fabula=600),
            },
        ),
        # Holdaway ↔ Orange — handler bond.
        RelationshipEdge(
            source_entity_id="ENT_HOLDAWAY", target_entity_id="ENT_ORANGE",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.55, evidence_strength="strong", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_HOLDAWAY",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.55, evidence_strength="strong", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        # Brown / Blue — minor edges to Joe.
        RelationshipEdge(
            source_entity_id="ENT_BROWN", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.45, inertia=0.45, evidence_strength="moderate", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=100),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_BLUE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.5, evidence_strength="moderate", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=100),
            },
        ),
    ],
)
