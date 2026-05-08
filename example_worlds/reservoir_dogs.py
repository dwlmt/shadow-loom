# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

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
    Channel,
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge, RelationshipMetric, TraitVector, AmbientVector, Affordance, Belief, EntityStateSnapshot,
    GlobalTrait, WorldTraitSnapshot,
    NarrativeStyle,
    Concern, Proposition, ConcernSnapshot, PropositionSnapshot,
)

world_state = WorldStateV1(
    narrative_style=NarrativeStyle(
        format='synopsis',
        target_word_min=270,
        target_word_max=1012,
        prose_density='sparse',
        voice='synoptic narration; no dialogue; condensed scene description; third-person POV; past tense',
        style_exemplar='Eight men planning to rob a jewelry store for a diamond shipment eat breakfast at a diner. To pull off the heist, boss Joe Cabot assembles six experienced robbers who are strangers to each other. Joe and his son, "Nice Guy" Eddie Cabot, have known some of the team for years, but to shield identities, the rest use aliases: Mr. White, a career criminal; Mr. Blonde, a trigger-happy ex-convict; Mr. Orange, a reputed drug dealer; Mr. Pink, a paranoid neurotic; Mr. Brown, a pseudo philosopher; and Mr. Blue, an even-tempered cohort.',
        source_word_count=675,
    ),
    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_DINER": Location(
            name="Coffee Shop",
            description="Coffee-shop where the crew has breakfast before the heist; tipping debate exposes their personalities.",
            ambient_state={
                "camaraderie": AmbientVector(value=0.6, volatility=0.3, evidence_strength="moderate"),
                "fluorescent_banality": AmbientVector(value=0.85, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_WAREHOUSE": Location(
            name="Rendezvous Warehouse",
            description="Abandoned warehouse where the surviving crew regroups after the botched heist; the slaughterhouse of the third act.",
            ambient_state={
                "tension": AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
                "danger": AmbientVector(value=0.8, volatility=0.4, evidence_strength="strong"),
                "blood_smell": AmbientVector(value=0.7, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_DIAMOND_STORE": Location(
            name="Jewelry Store",
            description="The jewelry store targeted for the heist; a silent alarm trips and turns the job into a slaughter.",
            ambient_state={
                "danger": AmbientVector(value=0.95, volatility=0.3, evidence_strength="strong"),
                "panic": AmbientVector(value=0.9, volatility=0.5, evidence_strength="strong"),
            },
        ),
        "LOC_JOE_OFFICE": Location(
            name="Joe Cabot's Planning Room",
            description="Crime boss Joe Cabot's planning room where the heist is laid out and the crew is hand-picked from old contacts and prison parolees.",
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
                       perceived_state="Orange is a good kid who can be saved; not a cop", proposition_id="PROP_WHITE_KNOWS_TRUTH",
                       confidence=0.85, inertia=0.5, established_at_fabula=1000, evidence_strength="strong"),
                Belief(target_id="ENT_BLONDE",
                       perceived_state="Blonde is a psychopath who blew the heist by shooting civilians", proposition_id="PROP_BLONDE_PSYCHO",
                       confidence=0.85, inertia=0.5, established_at_fabula=5000, evidence_strength="strong"),
                Belief(target_id="ENT_JOE",
                       perceived_state="Joe is an old friend whose judgment I trust — though employing Blonde was a mistake", proposition_id="PROP_HEIST_CLEAN",
                       confidence=0.75, inertia=0.55, established_at_fabula=1000, evidence_strength="moderate"),
            ],
            constants=["career_criminal", "joes_old_friend"],
            concerns=[
                # Sternberg passionate-bond — surrogate-paternal love that
                # White invests in the dying Orange in the back of the car.
                # Counter-link: when Orange is revealed AS the rat, his betrayal-fear
                # collides with this love — the central tragic dilemma.
                Concern(concern_id="CCN_WHITE_LOVES_ORANGE", proposition_id="PROP_ORANGE_LIVES",
                        polarity="desire", kind="love", salience=1.0,
                        activation_fabula_window=[4000, 10000],
                        counter_concern_ids=["CCN_WHITE_FEARS_RAT", "CCN_WHITE_DESIRES_VENGEANCE_ON_RAT"],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=9000, triggered_by="EVT_ORANGE_REVEALED",
                                            salience=0.25),
                        ]),
                Concern(concern_id="CCN_WHITE_DESIRES_HOSPITAL", proposition_id="PROP_ORANGE_HOSPITALISED",
                        polarity="desire", kind="survival", salience=0.9,
                        activation_fabula_window=[4000, 9500]),
                # Lazarus appraisal — the heist as professional pride.
                Concern(concern_id="CCN_WHITE_DESIRES_CLEAN_HEIST", proposition_id="PROP_HEIST_CLEAN",
                        polarity="desire", kind="loyalty", salience=0.7,
                        activation_fabula_window=[1000, 3000]),
                # Averill normative-violation rage — once the rat is named,
                # the moral injury that drives the standoff and the killing.
                # Counter-linked to White's love for Orange (the rat IS Orange).
                Concern(concern_id="CCN_WHITE_FEARS_RAT", proposition_id="PROP_RAT_EXISTS",
                        polarity="fear", kind="betrayal", salience=0.95,
                        activation_fabula_window=[3000, 10000],
                        counter_concern_ids=["CCN_WHITE_LOVES_ORANGE"]),
                Concern(concern_id="CCN_WHITE_DESIRES_VENGEANCE_ON_RAT", proposition_id="PROP_RAT_PUNISHED",
                        polarity="desire", kind="vengeance", salience=0.95,
                        activation_fabula_window=[8000, 10000],
                        counter_concern_ids=["CCN_WHITE_LOVES_ORANGE"],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=9000, triggered_by="EVT_ORANGE_REVEALED",
                                            salience=1.0),
                        ]),
                # OCC fear — Blonde is the agent of chaos in the warehouse.
                Concern(concern_id="CCN_WHITE_FEARS_BLONDE", proposition_id="PROP_BLONDE_PSYCHO",
                        polarity="fear", kind="mortal_threat", salience=0.7,
                        activation_fabula_window=[3000, 7000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=4000, triggered_by="EVT_ORANGE_SHOT",
                    location_id="LOC_ORANGE_CAR",
                    traits={
                        "compassion":        TraitVector(value=0.8, inertia=0.55, evidence_strength="strong"),
                        "paternal_instinct": TraitVector(value=0.9, inertia=0.65, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=8000, triggered_by="EVT_MEXICAN_STANDOFF",
                    status="injured",
                    traits={
                        "temper": TraitVector(value=0.85, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_ORANGE_REVEALED",
                    traits={
                        "loyalty":    TraitVector(value=0.2, inertia=0.65, evidence_strength="strong"),
                        "compassion": TraitVector(value=0.3, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_ORANGE"],
                    beliefs_added=[
                        Belief(target_id="ENT_ORANGE",
                               perceived_state="he was the rat the whole time; my paternal love was a lie",
                               proposition_id="PROP_RAT_IDENTIFIED",
                               confidence=0.95, inertia=0.7, established_at_fabula=9000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=9500, triggered_by="EVT_WHITE_KILLS_ORANGE",
                    traits={
                        "compassion": TraitVector(value=0.05, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=10000, triggered_by="EVT_WHITE_KILLS_ORANGE",
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
                       perceived_state="White genuinely cares about me — which makes the betrayal worse", proposition_id="PROP_WHITE_KNOWS_TRUTH",
                       confidence=0.85, inertia=0.5, established_at_fabula=4000, evidence_strength="strong"),
            ],
            constants=["undercover_cop", "uses_alias_freddy_newandyke"],
            concerns=[
                # Lazarus appraisal — LAPD duty drives the cover.
                Concern(concern_id="CCN_ORANGE_DESIRES_BUST", proposition_id="PROP_HEIST_BUSTED",
                        polarity="desire", kind="loyalty", salience=0.95,
                        activation_fabula_window=[500, 9000]),
                Concern(concern_id="CCN_ORANGE_FEARS_DISCOVERY", proposition_id="PROP_RAT_IDENTIFIED",
                        polarity="fear", kind="exposure", salience=0.95,
                        activation_fabula_window=[3000, 9000]),
                Concern(concern_id="CCN_ORANGE_LOVES_WHITE", proposition_id="PROP_WHITE_LIVES",
                        polarity="desire", kind="love", salience=0.85,
                        activation_fabula_window=[4000, 10000]),
                # Sternberg/Averill — Orange's confessional regret-as-injury
                # impels him to tell White the truth even at the cost of being shot.
                Concern(concern_id="CCN_ORANGE_DESIRES_CONFESS", proposition_id="PROP_WHITE_KNOWS_TRUTH",
                        polarity="desire", kind="truth", salience=0.9,
                        activation_fabula_window=[8000, 10000],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=8000, triggered_by="EVT_MEXICAN_STANDOFF",
                                            salience=1.0),
                        ]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=4000, triggered_by="EVT_ORANGE_SHOT",
                    status="injured", location_id="LOC_ORANGE_CAR",
                    traits={
                        "pain":   TraitVector(value=0.95, inertia=0.4, evidence_strength="strong"),
                        "guilt":  TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=7000, triggered_by="EVT_ORANGE_KILLS_BLONDE",
                    traits={
                        "courage": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                        "duty":    TraitVector(value=0.95, inertia=0.75, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=8000, triggered_by="EVT_MEXICAN_STANDOFF",
                    status="injured",
                    traits={
                        "pain": TraitVector(value=0.99, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_ORANGE_REVEALED",
                    traits={
                        "deceit": TraitVector(value=0.1, inertia=0.55, evidence_strength="strong"),
                        "guilt":  TraitVector(value=0.95, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_WHITE",
                               perceived_state="he loves me enough to risk dying for me; he deserves the truth",
                               proposition_id="PROP_WHITE_KNOWS_TRUTH",
                               confidence=0.95, inertia=0.7, established_at_fabula=9000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=10000, triggered_by="EVT_WHITE_KILLS_ORANGE",
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
                       perceived_state="Joe took care of me while I did time; I owe him absolutely", proposition_id="PROP_BLONDE_LOYAL",
                       confidence=0.95, inertia=0.75, established_at_fabula=500, evidence_strength="strong"),
            ],
            constants=["ex_convict", "vega_brother", "psychopath"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3000, triggered_by="EVT_HEIST_GOES_WRONG",
                    location_id="LOC_DIAMOND_STORE",
                    traits={
                        "sadism": TraitVector(value=0.99, inertia=0.8, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_BLONDE_TORTURES_COP",
                    location_id="LOC_WAREHOUSE"),
                EntityStateSnapshot(fabula_time=7000, triggered_by="EVT_ORANGE_KILLS_BLONDE",
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
                       proposition_id="PROP_RAT_EXISTS",
                       confidence=0.85, inertia=0.55, established_at_fabula=5000, evidence_strength="strong"),
            ],
            constants=["paranoid_neurotic", "tipping_refusenik"],
            concerns=[
                Concern(concern_id="CCN_PINK_DESIRES_ESCAPE", proposition_id="PROP_PINK_ESCAPES",
                        polarity="desire", kind="survival", salience=0.99,
                        activation_fabula_window=[3000, 8000]),
                Concern(concern_id="CCN_PINK_FEARS_RAT", proposition_id="PROP_RAT_EXISTS",
                        polarity="fear", kind="betrayal", salience=0.85,
                        activation_fabula_window=[3000, 8000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=5000, triggered_by="EVT_WAREHOUSE_REGROUP",
                    traits={
                        "paranoia": TraitVector(value=0.95, inertia=0.65, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_BLONDE",
                               perceived_state="Joe should never have hired this lunatic", proposition_id="PROP_BLONDE_PSYCHO",
                               confidence=0.85, inertia=0.5, established_at_fabula=5000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=8000, triggered_by="EVT_MEXICAN_STANDOFF",
                    location_id="LOC_WAREHOUSE",
                    traits={
                        "self_preservation": TraitVector(value=0.99, inertia=0.75, evidence_strength="strong"),
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
                       perceived_state="Blonde is loyal — he did four years rather than name me", proposition_id="PROP_BLONDE_LOYAL",
                       confidence=0.95, inertia=0.7, established_at_fabula=500, evidence_strength="strong"),
            ],
            constants=["crime_boss", "old_school"],
            concerns=[
                Concern(concern_id="CCN_JOE_LOVES_BLONDE", proposition_id="PROP_BLONDE_LOYAL",
                        polarity="desire", kind="love", salience=0.85,
                        activation_fabula_window=[1, 8000]),
                # Averill betrayal-rage — fires the standoff that kills him.
                Concern(concern_id="CCN_JOE_DESIRES_RAT_DEAD", proposition_id="PROP_RAT_PUNISHED",
                        polarity="desire", kind="vengeance", salience=0.95,
                        activation_fabula_window=[5000, 8000]),
                Concern(concern_id="CCN_JOE_DESIRES_DIAMONDS", proposition_id="PROP_HEIST_CLEAN",
                        polarity="desire", kind="power", salience=0.7,
                        activation_fabula_window=[1000, 3000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=8000, triggered_by="EVT_MEXICAN_STANDOFF",
                    beliefs_added=[
                        Belief(target_id="ENT_ORANGE",
                               perceived_state="Orange is the rat — my instinct never lies",
                               proposition_id="PROP_RAT_IDENTIFIED",
                               confidence=0.95, inertia=0.7, established_at_fabula=8000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=8000, triggered_by="EVT_MEXICAN_STANDOFF",
                    status="dead", location_id="LOC_WAREHOUSE",
                    traits={
                        # Parity with mutation edge EVT_MEXICAN_STANDOFF → ENT_JOE instinct -1.0.
                        "instinct": TraitVector(value=0.0, inertia=0.7, evidence_strength="strong"),
                    }),
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
                       perceived_state="Dad is always right; his judgment defines mine", proposition_id="PROP_BLONDE_LOYAL",
                       confidence=0.95, inertia=0.8, established_at_fabula=500, evidence_strength="strong"),
                Belief(target_id="ENT_BLONDE",
                       perceived_state="Vic is family — he proved his loyalty in prison", proposition_id="PROP_BLONDE_LOYAL",
                       confidence=0.9, inertia=0.7, established_at_fabula=500, evidence_strength="strong"),
            ],
            constants=["joes_son"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=8000, triggered_by="EVT_MEXICAN_STANDOFF",
                    status="dead", location_id="LOC_WAREHOUSE",
                    traits={
                        # Parity with mutation edge EVT_MEXICAN_STANDOFF → ENT_EDDIE bravado -1.0.
                        "bravado": TraitVector(value=0.0, inertia=0.55, evidence_strength="strong"),
                    }),
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
                       perceived_state="this man is one of ours; I will not blow his cover even under torture", proposition_id="PROP_RAT_IDENTIFIED",
                       confidence=0.95, inertia=0.75, established_at_fabula=3000, evidence_strength="strong"),
            ],
            constants=["police_officer", "lapd"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=5000, triggered_by="EVT_WAREHOUSE_REGROUP",
                    status="injured", location_id="LOC_WAREHOUSE",
                    traits={
                        "fear": TraitVector(value=0.95, inertia=0.55, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_BLONDE_TORTURES_COP",
                    status="unconscious",
                    traits={
                        "fear": TraitVector(value=0.99, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=7600, triggered_by="EVT_EDDIE_KILLS_NASH",
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
                EntityStateSnapshot(fabula_time=3000, triggered_by="EVT_HEIST_GOES_WRONG",
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
                EntityStateSnapshot(fabula_time=7500, triggered_by="EVT_BLUE_KILLED_OFFSCREEN",
                    status="dead"),
            ],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_HEIST_PLANNED", fabula_time=1000, syuzhet_index=1,
                  event_type="choice",
                  actor_ids=["ENT_JOE", "ENT_EDDIE"],
                  target_ids=["ENT_WHITE", "ENT_ORANGE", "ENT_BLONDE", "ENT_PINK", "ENT_BROWN", "ENT_BLUE"],
                  description="Joe Cabot and Eddie assemble six experienced robbers — strangers to each other — under colour-coded aliases to rob Karina's Diamond Store."),
        EventNode(id="EVT_DINER_BREAKFAST", fabula_time=2000, syuzhet_index=2,
                  event_type="outcome",
                  actor_ids=["ENT_WHITE", "ENT_ORANGE", "ENT_BLONDE", "ENT_PINK", "ENT_BROWN", "ENT_BLUE", "ENT_JOE", "ENT_EDDIE"],
                  target_ids=[],
                  description="The crew has breakfast at Uncle Bob's; the tipping debate exposes Pink's stinginess and the precarious cohesion of strangers."),
        EventNode(id="EVT_HEIST_GOES_WRONG", fabula_time=3000, syuzhet_index=3,
                  event_type="outcome",
                  actor_ids=["ENT_BLONDE"],
                  target_ids=["ENT_WHITE", "ENT_ORANGE", "ENT_PINK", "ENT_BROWN"],
                  description="A silent alarm trips; Blonde starts murdering bystanders; police arrive immediately and Brown is killed in the getaway."),
        EventNode(id="EVT_ORANGE_SHOT", fabula_time=4000, syuzhet_index=4,
                  event_type="outcome",
                  actor_ids=[],
                  target_ids=["ENT_ORANGE"],
                  description="Hijacking a car, Orange is shot in the abdomen by the panicked civilian driver; he kills her in return and bleeds out across the back seat as White drives him to the warehouse."),
        EventNode(id="EVT_WAREHOUSE_REGROUP", fabula_time=5000, syuzhet_index=5,
                  event_type="choice",
                  actor_ids=["ENT_WHITE", "ENT_PINK"],
                  target_ids=["ENT_ORANGE"],
                  description="White and Pink regroup at the warehouse; Pink declares the job a setup and refuses to call a doctor; the two draw on each other before Blonde arrives with a kidnapped cop."),
        EventNode(id="EVT_BLONDE_TORTURES_COP", fabula_time=6000, syuzhet_index=6,
                  event_type="choice",
                  actor_ids=["ENT_BLONDE"],
                  target_ids=["ENT_MARVIN"],
                  description="Left alone with Marvin Nash, Blonde slashes the cop's face and severs his ear with a straight razor while 'Stuck in the Middle with You' plays on the radio, and prepares to set him on fire."),
        EventNode(id="EVT_ORANGE_KILLS_BLONDE", fabula_time=7000, syuzhet_index=7,
                  event_type="choice",
                  actor_ids=["ENT_ORANGE"],
                  target_ids=["ENT_BLONDE"],
                  description="Bleeding Orange empties his pistol into Blonde to save Nash, breaking cover for the audience and revealing his loyalties to the cop he was protecting."),
        EventNode(id="EVT_BLUE_KILLED_OFFSCREEN", fabula_time=7500, syuzhet_index=8,
                  event_type="outcome",
                  actor_ids=[],
                  target_ids=["ENT_BLUE"],
                  description="Mr Blue is killed by the police in flight from the heist — reported by Joe at the warehouse, never shown."),
        EventNode(id="EVT_EDDIE_KILLS_NASH", fabula_time=7600, syuzhet_index=9,
                  event_type="choice",
                  actor_ids=["ENT_EDDIE"],
                  target_ids=["ENT_MARVIN"],
                  description="Returning to the warehouse, Eddie executes the wounded cop to silence him, refusing to believe Orange's story that Blonde planned to murder them."),
        EventNode(id="EVT_MEXICAN_STANDOFF", fabula_time=8000, syuzhet_index=10,
                  event_type="outcome",
                  actor_ids=["ENT_JOE", "ENT_EDDIE", "ENT_WHITE"],
                  target_ids=["ENT_ORANGE"],
                  description="Joe arrives, names Orange as the rat, and goes to execute him; White intervenes at gunpoint; Eddie aims at White; all three fire and Joe, Eddie, and (mortally) White fall."),
        EventNode(id="EVT_ORANGE_REVEALED", fabula_time=9000, syuzhet_index=11,
                  event_type="outcome",
                  actor_ids=["ENT_ORANGE"],
                  target_ids=["ENT_WHITE"],
                  description="Cradled in White's arms, the dying Orange confesses that he is an undercover police officer, retroactively reframing every act of paternal tenderness."),
        EventNode(id="EVT_WHITE_KILLS_ORANGE", fabula_time=10000, syuzhet_index=12,
                  event_type="choice",
                  actor_ids=["ENT_WHITE"],
                  target_ids=["ENT_ORANGE"],
                  description="As LAPD storm the warehouse, White presses his pistol to Orange's head and pulls the trigger — and is cut down by police gunfire in the same beat."),

        # ── UTTERANCE EVENTS (one-shot speech-acts; only those riding standing channels carry via_channel_id) ──
        EventNode(
            id="EVT_UTT_JOE_PITCHES_HEIST", fabula_time=1050, syuzhet_index=13,
            event_type="utterance", speaker_id="ENT_JOE",
            addressee_ids=["ENT_EDDIE", "ENT_WHITE", "ENT_BLONDE", "ENT_PINK", "ENT_ORANGE", "ENT_BLUE", "ENT_BROWN"],
            actor_ids=["ENT_JOE"],
            target_ids=["OBJ_DIAMONDS", "LOC_DIAMOND_STORE"],
            via_channel_id="CHN_CREW_PLANNING",
            truth_value="true",
            description="At his planning room Joe walks the hand-picked crew through the diamond-store score and lays down the colour-code aliases that keep them strangers to each other.",
            content="Joe outlines the diamond-store target, his vetted six-man roster, and the rule that nobody uses real names — only Mr White, Blonde, Orange, Pink, Brown, Blue.",
        ),
        EventNode(
            id="EVT_UTT_PINK_DEMANDS_NO_DOCTOR", fabula_time=5100, syuzhet_index=16,
            event_type="utterance", speaker_id="ENT_PINK",
            addressee_ids=["ENT_WHITE"],
            actor_ids=["ENT_PINK"],
            target_ids=["ENT_ORANGE", "ENT_JOE"],
            via_channel_id=None,
            truth_value="performative",
            description="At the warehouse, Pink draws on White and forbids him from calling a doctor for the dying Orange until Joe authorises it — he believes the heist was a setup.",
            content="Pink orders White to leave Orange bleeding rather than break protocol by summoning a hospital before Joe arrives.",
        ),
        EventNode(
            id="EVT_UTT_JOE_NAMES_RAT", fabula_time=7900, syuzhet_index=17,
            event_type="utterance", speaker_id="ENT_JOE",
            addressee_ids=["ENT_WHITE", "ENT_EDDIE", "ENT_PINK", "ENT_ORANGE"],
            actor_ids=["ENT_JOE"],
            target_ids=["ENT_ORANGE"],
            via_channel_id=None,
            truth_value="true",
            description="Joe arrives at the warehouse, points his pistol at Orange and declares him the police informant who set up the heist — triggering the Mexican standoff.",
            content="Joe accuses Orange of being an undercover police officer who fed the LAPD the heist plan and got Brown and Blue killed.",
        ),
        EventNode(
            id="EVT_UTT_ORANGE_COP_CONFESSION", fabula_time=9200, syuzhet_index=18,
            event_type="utterance", speaker_id="ENT_ORANGE",
            addressee_ids=["ENT_WHITE"],
            actor_ids=["ENT_ORANGE"],
            target_ids=["ENT_ORANGE"],
            via_channel_id=None,
            truth_value="true",
            description="Cradled in White's arms after the standoff, the dying Orange whispers the confession that retroactively reframes every paternal act of loyalty.",
            content="Orange tells White, 'I'm a cop, Larry. I'm so sorry,' confirming Joe's accusation.",
        ),
        # Orange discloses to Nash that he is an undercover police officer
        # — the first on-page airing of his identity, before Joe arrives.
        EventNode(
            id="EVT_UTT_ORANGE_DISCLOSES_TO_NASH", fabula_time=7050, syuzhet_index=19,
            event_type="utterance", speaker_id="ENT_ORANGE",
            addressee_ids=["ENT_MARVIN"],
            actor_ids=["ENT_ORANGE"],
            target_ids=["ENT_ORANGE", "ENT_JOE"],
            via_channel_id=None,
            truth_value="true",
            description="Orange tells the wounded Nash that he is an undercover police officer and that the LAPD will move when Joe arrives at the warehouse.",
            content="I'm a cop. The police will be here when Joe shows up — hold on.",
        ),
        # Nash reveals he had recognised Orange and protected his cover under torture.
        EventNode(
            id="EVT_UTT_NASH_RECOGNISED_ORANGE", fabula_time=7080, syuzhet_index=20,
            event_type="utterance", speaker_id="ENT_MARVIN",
            addressee_ids=["ENT_ORANGE"],
            actor_ids=["ENT_MARVIN"],
            target_ids=["ENT_ORANGE"],
            via_channel_id=None,
            truth_value="true",
            description="Nash replies that he recognised Orange from the start and refused to give him up under Blonde's razor.",
            content="I knew who you were. I wasn't going to give you up.",
        ),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction ──
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="EVT_DINER_BREAKFAST",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000, propagation_delay=1000),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="EVT_HEIST_GOES_WRONG",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2000, propagation_delay=1000),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="EVT_ORANGE_SHOT",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3000, propagation_delay=1000),
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="EVT_WAREHOUSE_REGROUP",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=4000, propagation_delay=1000),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="EVT_BLONDE_TORTURES_COP",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=5000, propagation_delay=1000),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="EVT_ORANGE_KILLS_BLONDE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=6000, propagation_delay=1000),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="EVT_EDDIE_KILLS_NASH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000, propagation_delay=600),
        CausalEdge(source_id="EVT_EDDIE_KILLS_NASH", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7600, propagation_delay=400),
        CausalEdge(source_id="EVT_BLUE_KILLED_OFFSCREEN", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=7500, propagation_delay=500),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="EVT_ORANGE_REVEALED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=8000, propagation_delay=1000),
        CausalEdge(source_id="EVT_ORANGE_REVEALED", target_id="EVT_WHITE_KILLS_ORANGE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=9000, propagation_delay=1000),

        # ── mutation ──
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="ENT_ORANGE",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=4000,
                   trait_target="pain", trait_delta=0.85),
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="ENT_WHITE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=4000,
                   trait_target="paternal_instinct", trait_delta=0.2),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_MARVIN",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=6000,
                   trait_target="fear", trait_delta=0.7),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="ENT_ORANGE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000,
                   trait_target="duty", trait_delta=0.4),
        CausalEdge(source_id="EVT_ORANGE_REVEALED", target_id="ENT_WHITE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=9000,
                   trait_target="loyalty", trait_delta=-0.6),
        CausalEdge(source_id="EVT_ORANGE_REVEALED", target_id="ENT_ORANGE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000,
                   trait_target="guilt", trait_delta=0.45),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="ENT_BLONDE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=6.0, fabula_time=3000,
                   trait_target="sadism", trait_delta=0.5),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_PINK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=5000,
                   trait_target="paranoia", trait_delta=0.15),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="ENT_WHITE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=8000,
                   trait_target="temper", trait_delta=0.4),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="ENT_JOE",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=8000,
                   trait_target="instinct", trait_delta=-1.0),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="ENT_EDDIE",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=8000,
                   trait_target="bravado", trait_delta=-1.0),
        CausalEdge(source_id="EVT_WHITE_KILLS_ORANGE", target_id="ENT_WHITE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=10000,
                   trait_target="compassion", trait_delta=-0.55),
        CausalEdge(source_id="EVT_EDDIE_KILLS_NASH", target_id="ENT_MARVIN",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=7600,
                   trait_target="fear", trait_delta=0.0),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_ORANGE_REVEALED", target_id="ENT_WHITE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=10.0, fabula_time=9000,
                   trait_target="affinity", trait_delta=-0.95, rel_counterpart_id="ENT_ORANGE"),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="ENT_WHITE",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3000,
                   trait_target="affinity", trait_delta=-0.75, rel_counterpart_id="ENT_BLONDE"),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="ENT_ORANGE",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000,
                   trait_target="affinity", trait_delta=0.25, rel_counterpart_id="ENT_WHITE"),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="ENT_ORANGE",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000,
                   trait_target="affinity", trait_delta=0.6, rel_counterpart_id="ENT_MARVIN"),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="ENT_WHITE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=8000,
                   trait_target="affinity", trait_delta=-0.85, rel_counterpart_id="ENT_JOE"),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="ENT_JOE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=8000,
                   trait_target="affinity", trait_delta=-0.9, rel_counterpart_id="ENT_WHITE"),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_MARVIN",
                   causality_type="mutation_social", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=6000,
                   trait_target="fear", trait_delta=0.8, rel_counterpart_id="ENT_BLONDE"),

        # ── affordance_gate ──
        CausalEdge(source_id="ENT_JOE", target_id="EVT_HEIST_PLANNED",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1000),
        CausalEdge(source_id="ENT_BLONDE", target_id="EVT_HEIST_GOES_WRONG",
                   causality_type="affordance_gate", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3000),
        CausalEdge(source_id="OBJ_GUNS", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=8000),
        CausalEdge(source_id="OBJ_RAZOR", target_id="EVT_BLONDE_TORTURES_COP",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000),
        CausalEdge(source_id="OBJ_GASOLINE", target_id="EVT_ORANGE_KILLS_BLONDE",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000),
        CausalEdge(source_id="OBJ_DIAMONDS", target_id="EVT_HEIST_PLANNED",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1000),
        CausalEdge(source_id="OBJ_RADIO", target_id="EVT_BLONDE_TORTURES_COP",
                   causality_type="affordance_gate", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=6000),

        # ── Utterance wiring (connect orphan EVT_UTT_ events to causal spine) ──
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="EVT_UTT_JOE_PITCHES_HEIST",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1000, propagation_delay=50),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="EVT_UTT_PINK_DEMANDS_NO_DOCTOR",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=5000, propagation_delay=100),
        CausalEdge(source_id="EVT_UTT_JOE_NAMES_RAT", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=9.0, fabula_time=7900, propagation_delay=100),
        CausalEdge(source_id="EVT_UTT_ORANGE_COP_CONFESSION", target_id="EVT_WHITE_KILLS_ORANGE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=9200, propagation_delay=800),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_WAREHOUSE", target_id="ENT_PINK",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=5000),
        CausalEdge(source_id="LOC_WAREHOUSE", target_id="ENT_WHITE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=5000),
        CausalEdge(source_id="LOC_DIAMOND_STORE", target_id="ENT_BLONDE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=3000),
        CausalEdge(source_id="LOC_ORANGE_CAR", target_id="ENT_ORANGE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=4000),

        # ── WORLD_ → Event (named-latent common-cause wiring) ──
        CausalEdge(source_id="WORLD_CRIMINAL_CODE", target_id="EVT_HEIST_PLANNED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_CRIMINAL_CODE", target_id="EVT_BLONDE_TORTURES_COP",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=6000),
        CausalEdge(source_id="WORLD_CRIMINAL_CODE", target_id="EVT_MEXICAN_STANDOFF",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=8000),
        CausalEdge(source_id="WORLD_CRIMINAL_CODE", target_id="EVT_WHITE_KILLS_ORANGE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=10000),
        CausalEdge(source_id="WORLD_POLICE_INFILTRATION", target_id="EVT_HEIST_GOES_WRONG",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=3000),
        CausalEdge(source_id="WORLD_POLICE_INFILTRATION", target_id="EVT_ORANGE_KILLS_BLONDE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=7000),
        CausalEdge(source_id="WORLD_POLICE_INFILTRATION", target_id="EVT_ORANGE_REVEALED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000),
        CausalEdge(source_id="WORLD_POLICE_INFILTRATION", target_id="EVT_WHITE_KILLS_ORANGE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=10000),

        # ─── auto-patched mutation_social edges (per-axis coverage) ───
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_WHITE", rel_counterpart_id="ENT_PINK", causality_type="mutation_social", trait_target="affinity", trait_delta=0.2, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_WHITE", rel_counterpart_id="ENT_PINK", causality_type="mutation_social", trait_target="affinity", trait_delta=0.15, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_WHITE", rel_counterpart_id="ENT_PINK", causality_type="mutation_social", trait_target="affinity", trait_delta=0.1, mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_PINK", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.15, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_PINK", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.15, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_PINK", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.1, mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_BLONDE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.55, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_ORANGE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.3, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_JOE", rel_counterpart_id="ENT_ORANGE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.1, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_JOE_NAMES_RAT", target_id="ENT_JOE", rel_counterpart_id="ENT_ORANGE", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.9, mechanism="betrayal", evidence_strength="strong", causal_force=9.0, fabula_time=7900, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_ORANGE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.2, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_ORANGE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.2, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_EDDIE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_EDDIE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="ENT_ORANGE", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.4, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_ORANGE", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.45, mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_NASH_RECOGNISED_ORANGE", target_id="ENT_MARVIN", rel_counterpart_id="ENT_ORANGE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.55, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=7080, propagation_delay=0),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_MARVIN", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.95, mechanism="physical", evidence_strength="strong", causal_force=10.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_BROWN", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.25, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_BROWN", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.2, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_BLUE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.3, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_BLUE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.2, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_ORANGE", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="fear", trait_delta=0.05, mechanism="epistemic", evidence_strength="weak", causal_force=3.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_ORANGE", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="fear", trait_delta=0.1, mechanism="emotional", evidence_strength="moderate", causal_force=5.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="ENT_ORANGE", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="fear", trait_delta=0.05, mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=8000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_WHITE", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="fear", trait_delta=0.05, mechanism="psychological", evidence_strength="weak", causal_force=3.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="ENT_WHITE", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="fear", trait_delta=0.15, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_PINK", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="fear", trait_delta=0.15, mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_PINK_DEMANDS_NO_DOCTOR", target_id="ENT_PINK", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="fear", trait_delta=0.05, mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=5100, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_ORANGE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="fear", trait_delta=0.05, mechanism="epistemic", evidence_strength="weak", causal_force=3.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_JOE_NAMES_RAT", target_id="ENT_ORANGE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="fear", trait_delta=0.15, mechanism="psychological", evidence_strength="strong", causal_force=8.0, fabula_time=7900, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="ENT_ORANGE", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="fear", trait_delta=0.1, mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_ORANGE", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="fear", trait_delta=0.15, mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="ENT_WHITE", rel_counterpart_id="ENT_ORANGE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.3, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=4000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_WHITE", rel_counterpart_id="ENT_ORANGE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.25, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="ENT_ORANGE", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.3, mechanism="physical", evidence_strength="strong", causal_force=6.0, fabula_time=4000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_ORANGE", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.25, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_WHITE", rel_counterpart_id="ENT_PINK", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.4, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_PINK_DEMANDS_NO_DOCTOR", target_id="ENT_WHITE", rel_counterpart_id="ENT_PINK", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.2, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=5100, propagation_delay=0),
        CausalEdge(source_id="EVT_WAREHOUSE_REGROUP", target_id="ENT_PINK", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.4, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_PINK_DEMANDS_NO_DOCTOR", target_id="ENT_PINK", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.2, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=5100, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_WHITE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.6, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_WHITE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.6, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_BLONDE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.7, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_BLONDE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.7, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_ORANGE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.65, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_ORANGE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.65, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_EDDIE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.65, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_EDDIE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.65, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_BROWN", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.6, mechanism="social", evidence_strength="moderate", causal_force=6.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_BLUE", rel_counterpart_id="ENT_JOE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.6, mechanism="social", evidence_strength="moderate", causal_force=6.0, fabula_time=1000, propagation_delay=0),
        # ── auto-backfilled per-axis mutation_social ──
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_BLONDE", rel_counterpart_id="ENT_WHITE",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.12,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_BLONDE", rel_counterpart_id="ENT_ORANGE",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.06,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_BLONDE", rel_counterpart_id="ENT_MARVIN",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.18,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_BROWN",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.09,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_BLUE",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.12,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_BLONDE", rel_counterpart_id="ENT_WHITE",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.09,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_DINER_BREAKFAST", target_id="ENT_BLONDE", rel_counterpart_id="ENT_ORANGE",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.12,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_BLONDE_TORTURES_COP", target_id="ENT_BLONDE", rel_counterpart_id="ENT_MARVIN",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.28,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_BROWN",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.18,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_HEIST_PLANNED", target_id="ENT_JOE", rel_counterpart_id="ENT_BLUE",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.18,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_JOE_OFFICE", target_id="LOC_DINER"),
        SpatialEdge(source_id="LOC_DINER", target_id="LOC_DIAMOND_STORE"),
        SpatialEdge(source_id="LOC_DIAMOND_STORE", target_id="LOC_ORANGE_CAR"),
        SpatialEdge(source_id="LOC_ORANGE_CAR", target_id="LOC_WAREHOUSE"),
        SpatialEdge(source_id="LOC_JOE_OFFICE", target_id="LOC_WAREHOUSE"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    # Only standing communication capabilities live here. One-shot speech-acts
    # (Pink's hospital ultimatum, Joe's accusation, Orange's deathbed confession,
    # Marvin's badge-recognition glance) are modelled as utterance EventNodes
    # with via_channel_id=None, not as one-instant Channels.
    channels={
        'CHN_CREW_PLANNING': Channel(
            id='CHN_CREW_PLANNING',
            name="Cabot crew heist-planning channel",
            medium='crew_planning',
            participant_ids=['ENT_JOE', 'ENT_EDDIE', 'ENT_WHITE', 'ENT_BLONDE',
                             'ENT_PINK', 'ENT_ORANGE', 'ENT_BLUE', 'ENT_BROWN'],
            directionality='broadcast',
            intelligibility={'ENT_JOE': 1.0, 'ENT_EDDIE': 1.0, 'ENT_WHITE': 1.0,
                             'ENT_BLONDE': 1.0, 'ENT_PINK': 1.0, 'ENT_ORANGE': 1.0,
                             'ENT_BLUE': 1.0, 'ENT_BROWN': 1.0},
            established_at_fabula=500,
            terminated_at_fabula=3000,
            evidence_strength='strong',
        ),
    },

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
                WorldTraitSnapshot(fabula_time=5000, triggered_by="EVT_WAREHOUSE_REGROUP",
                    magnitude=TraitVector(value=0.55, inertia=0.5, evidence_strength="strong"),
                    description="Mutual suspicion and the drawn guns at the warehouse fracture the code from within."),
                WorldTraitSnapshot(fabula_time=10000, triggered_by="EVT_WHITE_KILLS_ORANGE",
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
                WorldTraitSnapshot(fabula_time=9000, triggered_by="EVT_ORANGE_REVEALED",
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
                "affinity":      RelationshipMetric(value=0.6, inertia=0.5, evidence_strength="strong", last_updated_fabula=9000),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.65, evidence_strength="moderate", last_updated_fabula=4000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_WHITE",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=9000),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=9000),
                "power_dynamic": RelationshipMetric(value=-0.55, inertia=0.65, evidence_strength="moderate", last_updated_fabula=4000),
            },
        ),
        # White ↔ Blonde — professional contempt → open hostility.
        RelationshipEdge(
            source_entity_id="ENT_WHITE", target_entity_id="ENT_BLONDE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=3000),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=3000),
            },
        ),
        # Blonde → White — amused contempt for the moralising old-school criminal who shouts about not torturing the cop; no fear, mild dominance through unpredictability.
        RelationshipEdge(
            source_entity_id="ENT_BLONDE", target_entity_id="ENT_WHITE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.4, inertia=0.5, evidence_strength="strong", last_updated_fabula=3000),
                "power_dynamic": RelationshipMetric(value=0.3,  inertia=0.6, evidence_strength="moderate", last_updated_fabula=3000),
            },
        ),
        # White ↔ Pink — wary professional alliance.
        RelationshipEdge(
            source_entity_id="ENT_WHITE", target_entity_id="ENT_PINK",
            metrics={
                "affinity":      RelationshipMetric(value=0.45, inertia=0.45, evidence_strength="moderate", last_updated_fabula=5000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=5000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_PINK", target_entity_id="ENT_WHITE",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.45, evidence_strength="moderate", last_updated_fabula=5000),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=5000),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=5000),
            },
        ),
        # White ↔ Joe — old friendship destroyed at gunpoint.
        RelationshipEdge(
            source_entity_id="ENT_WHITE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.55, evidence_strength="strong", last_updated_fabula=8000),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.7, evidence_strength="strong", last_updated_fabula=8000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_WHITE",
            metrics={
                "affinity":      RelationshipMetric(value=0.45, inertia=0.55, evidence_strength="strong", last_updated_fabula=8000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.7, evidence_strength="strong", last_updated_fabula=8000),
            },
        ),
        # Blonde ↔ Joe — patron-client loyalty (the prison debt).
        RelationshipEdge(
            source_entity_id="ENT_BLONDE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.55, evidence_strength="strong", last_updated_fabula=500),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_BLONDE",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.55, evidence_strength="strong", last_updated_fabula=500),
                "power_dynamic": RelationshipMetric(value=0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=500),
            },
        ),
        # Joe ↔ Orange — earned trust → fatal accusation.
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_ORANGE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.5, inertia=0.5, evidence_strength="strong", last_updated_fabula=8000),
                "power_dynamic": RelationshipMetric(value=0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=8000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.45, evidence_strength="moderate", last_updated_fabula=2000),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=8000),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=8000),
            },
        ),
        # Eddie ↔ Joe — filial loyalty (Eddie devoted son; Joe paternal but business first).
        RelationshipEdge(
            source_entity_id="ENT_EDDIE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.85, inertia=0.6, evidence_strength="strong", last_updated_fabula=1000),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=1000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_EDDIE",
            metrics={
                "affinity":      RelationshipMetric(value=0.7, inertia=0.6, evidence_strength="strong", last_updated_fabula=1000),
                "power_dynamic": RelationshipMetric(value=0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=1000),
            },
        ),
        # Orange ↔ Blonde — natural enemies.
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_BLONDE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.85, inertia=0.5, evidence_strength="strong", last_updated_fabula=7000),
                "fear":          RelationshipMetric(value=0.25, inertia=0.2, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        # Blonde → Orange — the bleeding 'rookie' on the warehouse floor barely registers as a person; mild irritation at the noise, no fear, sociopathic indifference.
        RelationshipEdge(
            source_entity_id="ENT_BLONDE", target_entity_id="ENT_ORANGE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.2, inertia=0.5, evidence_strength="moderate", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=0.4,  inertia=0.6, evidence_strength="moderate", last_updated_fabula=6000),
            },
        ),
        # Orange ↔ Marvin — buried solidarity.
        RelationshipEdge(
            source_entity_id="ENT_ORANGE", target_entity_id="ENT_MARVIN",
            metrics={
                "affinity": RelationshipMetric(value=0.6, inertia=0.55, evidence_strength="strong", last_updated_fabula=7000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MARVIN", target_entity_id="ENT_ORANGE",
            metrics={
                "affinity": RelationshipMetric(value=0.55, inertia=0.55, evidence_strength="strong", last_updated_fabula=3000),
            },
        ),
        # Marvin ↔ Blonde — torturer/victim.
        RelationshipEdge(
            source_entity_id="ENT_MARVIN", target_entity_id="ENT_BLONDE",
            metrics={
                "affinity": RelationshipMetric(value=-0.95, inertia=0.5, evidence_strength="strong", last_updated_fabula=6000),
                "fear":     RelationshipMetric(value=1.0, inertia=0.25, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        # Blonde → Marvin — the cop tied to the chair is sport, not a person; the 'Stuck in the Middle With You' ear-cutting expresses pleasure in cruelty plus total dominion.
        RelationshipEdge(
            source_entity_id="ENT_BLONDE", target_entity_id="ENT_MARVIN",
            metrics={
                "affinity":      RelationshipMetric(value=-0.6, inertia=0.5, evidence_strength="strong", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=0.95, inertia=0.7, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        # Brown / Blue — minor edges to Joe.
        RelationshipEdge(
            source_entity_id="ENT_BROWN", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.45, inertia=0.45, evidence_strength="moderate", last_updated_fabula=1000),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=1000),
            },
        ),
        # Joe → Brown — the heister who dies first in the getaway; minor crew member he tolerated. Sign-flipped boss authority.
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_BROWN",
            metrics={
                "affinity":      RelationshipMetric(value=0.3, inertia=0.45, evidence_strength="moderate", last_updated_fabula=1000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=1000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_BLUE", target_entity_id="ENT_JOE",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.5, evidence_strength="moderate", last_updated_fabula=1000),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=1000),
            },
        ),
        # Joe → Blue — a quiet veteran heister Joe rates higher than the others; mild approval, sign-flipped boss authority.
        RelationshipEdge(
            source_entity_id="ENT_JOE", target_entity_id="ENT_BLUE",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.5, evidence_strength="moderate", last_updated_fabula=1000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=1000),
            },
        ),
    ],

    # ── PROPOSITIONS ────────────────────────────────────────────────────
    propositions=[
        Proposition(proposition_id="PROP_ORANGE_LIVES", kind="trait_holds",
                    referent_ids=["ENT_ORANGE"],
                    description="Mr. Orange survives his gut wound.",
                    audience_default_prior=0.3, stakes=0.85,
                    truth_at_fabula={10000: False}),
        Proposition(proposition_id="PROP_ORANGE_HOSPITALISED", kind="event_occurs",
                    referent_ids=["ENT_ORANGE"],
                    description="Mr. White gets Orange to a hospital.",
                    audience_default_prior=0.3, stakes=0.7,
                    truth_at_fabula={10000: False}),
        Proposition(proposition_id="PROP_HEIST_CLEAN", kind="outcome",
                    referent_ids=["EVT_HEIST_GOES_WRONG"],
                    description="The diamond heist goes off without complications.",
                    audience_default_prior=0.4, stakes=0.85,
                    truth_at_fabula={3000: False}),
        Proposition(proposition_id="PROP_RAT_EXISTS", kind="trait_holds",
                    referent_ids=["ENT_ORANGE"],
                    description="One of the crew is an undercover police informant.",
                    audience_default_prior=0.5, stakes=0.95,
                    truth_at_fabula={1: True}),
        Proposition(proposition_id="PROP_RAT_PUNISHED", kind="event_occurs",
                    referent_ids=["EVT_WHITE_KILLS_ORANGE"],
                    description="The rat is identified and executed.",
                    audience_default_prior=0.4, stakes=0.85,
                    truth_at_fabula={10000: True}),
        Proposition(proposition_id="PROP_RAT_IDENTIFIED", kind="event_occurs",
                    referent_ids=["EVT_ORANGE_REVEALED"],
                    description="The rat's true identity is exposed to the crew.",
                    audience_default_prior=0.55, stakes=0.9,
                    truth_at_fabula={9000: True}),
        Proposition(proposition_id="PROP_BLONDE_PSYCHO", kind="trait_holds",
                    referent_ids=["ENT_BLONDE"],
                    description="Mr. Blonde is uncontrollable and homicidal under pressure.",
                    audience_default_prior=0.6, stakes=0.7,
                    truth_at_fabula={6000: True}),
        Proposition(proposition_id="PROP_HEIST_BUSTED", kind="event_occurs",
                    referent_ids=["EVT_HEIST_GOES_WRONG"],
                    description="The LAPD busts the diamond heist (Orange's mission objective).",
                    audience_default_prior=0.45, stakes=0.85,
                    truth_at_fabula={3000: True}),
        Proposition(proposition_id="PROP_WHITE_LIVES", kind="trait_holds",
                    referent_ids=["ENT_WHITE"],
                    description="Mr. White survives the warehouse standoff.",
                    audience_default_prior=0.55, stakes=0.7,
                    truth_at_fabula={10000: False}),
        Proposition(proposition_id="PROP_WHITE_KNOWS_TRUTH", kind="event_occurs",
                    referent_ids=["EVT_ORANGE_REVEALED"],
                    description="Orange confesses his true identity to White before he dies.",
                    audience_default_prior=0.4, stakes=0.85,
                    truth_at_fabula={10000: True}),
        Proposition(proposition_id="PROP_PINK_ESCAPES", kind="event_occurs",
                    referent_ids=["ENT_PINK"],
                    description="Mr. Pink escapes the warehouse with the diamonds.",
                    audience_default_prior=0.4, stakes=0.6,
                    truth_at_fabula={10000: True}),
        Proposition(proposition_id="PROP_BLONDE_LOYAL", kind="trait_holds",
                    referent_ids=["ENT_BLONDE", "ENT_JOE"],
                    description="Vic Vega (Blonde) is the loyal ex-con Joe trusts most.",
                    audience_default_prior=0.7, stakes=0.55,
                    truth_at_fabula={7000: False}),
    ],
)
