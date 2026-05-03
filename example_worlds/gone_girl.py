# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gone Girl — high-fidelity WorldStateV1 test fixture.

Authored against the current ingestion prompts. Demonstrates all five
CausalEdge modalities, per-axis ``RelationshipMetric``, explicit
``evidence_strength`` everywhere, and named-latent WORLD_ traits
(Trial-by-Media, Marriage-as-Performance, Recession-Era Precarity)
wired as common-cause parents over the events they jointly drive.
The Trial-by-Media latent escalates with the news cycle and re-routes
both Nick's TV confession and the public reception of Amy's "rescue";
Marriage-as-Performance is the suburban suffocation that birthed the
fake diary and the final cage of staged domestic bliss.
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
        format='synopsis',
        target_word_min=318,
        target_word_max=600,
        prose_density='sparse',
        voice='synoptic narration; no dialogue; condensed scene description; third-person POV; past tense',
        style_exemplar="The narrative alternates between the point of view of Nick and Amy Dunne (née Elliott). Nick's narration begins shortly after arriving home on his fifth wedding anniversary to find Amy is missing from their home; there are signs of a struggle. Amy's narration comes in the form of her diaries and follows the earlier stages of their relationship.\n\nThe diary entries describe how Amy met Nick in New York City, where they both worked as writers. Nick was a journalist who wrote movie and TV reviews, while Amy wrote personality quizzes for women's magazines. After two years of dating, they married.",
        source_word_count=956,
    ),
    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_DUNNE_HOUSE": Location(
            name="Dunne House",
            description="Nick and Amy's rented McMansion in foreclosed-out North Carthage, Missouri — the staged scene of the disappearance.",
            ambient_state={
                "tension":     AmbientVector(value=0.7, volatility=0.4, evidence_strength="strong"),
                "stagnation":  AmbientVector(value=0.6, volatility=0.3, evidence_strength="strong"),
                "performance": AmbientVector(value=0.65, volatility=0.3, evidence_strength="moderate"),
            },
        ),
        "LOC_BAR": Location(
            name="The Bar",
            description="Nick and Margo's bar, purchased with the last of Amy's trust fund — Nick's only refuge from the press.",
            ambient_state={
                "refuge":     AmbientVector(value=0.5, volatility=0.3, evidence_strength="moderate"),
                "resentment": AmbientVector(value=0.5, volatility=0.3, evidence_strength="moderate"),
            },
        ),
        "LOC_HIDEOUT_OZARKS": Location(
            name="Ozarks Hideout",
            description="The cash-only motel in the Ozarks where Amy hides for weeks watching Nick on cable news.",
            ambient_state={
                "isolation": AmbientVector(value=0.8, volatility=0.2, evidence_strength="strong"),
                "control":   AmbientVector(value=0.7, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_DESI_LAKE_HOUSE": Location(
            name="Desi's Lake House",
            description="Desi Collings's gated, camera-rigged lake house — luxury that becomes a prison.",
            ambient_state={
                "luxury":     AmbientVector(value=0.7, volatility=0.2, evidence_strength="strong"),
                "entrapment": AmbientVector(value=0.6, volatility=0.4, evidence_strength="strong"),
                "surveillance": AmbientVector(value=0.85, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_POLICE_STATION": Location(
            name="North Carthage Police Station",
            description="Where Detectives Boney and Gilpin run the missing-persons-turned-homicide investigation.",
            ambient_state={
                "suspicion":   AmbientVector(value=0.7, volatility=0.4, evidence_strength="strong"),
                "procedural":  AmbientVector(value=0.8, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_MEDIA_CIRCUS": Location(
            name="Media Spotlight",
            description="The cable-news lawn camped outside the Dunne house; Ellen Abbott's pundit court of public opinion.",
            ambient_state={
                "hysteria":   AmbientVector(value=0.8, volatility=0.4, evidence_strength="strong"),
                "performance": AmbientVector(value=0.85, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_NEW_YORK": Location(
            name="New York City",
            description="The vanished Brooklyn-magazine life Nick and Amy left behind after the Recession layoffs.",
            ambient_state={
                "nostalgia":   AmbientVector(value=0.5, volatility=0.3, evidence_strength="moderate"),
                "lost_world":  AmbientVector(value=0.6, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_WOODLAWN": Location(
            name="Woodlawn",
            description="The crumbling Missouri suburb where Amy's anniversary clues lead Nick to Margo's woodshed.",
            ambient_state={
                "suburban_dread": AmbientVector(value=0.6, volatility=0.3, evidence_strength="moderate"),
                "decay":          AmbientVector(value=0.55, volatility=0.2, evidence_strength="moderate"),
            },
        ),
    },

    # ── OBJECTS ────────────────────────────────────────────────────────
    objects={
        "OBJ_DIARY": NarrativeObject(
            id="OBJ_DIARY", name="Amy's Fake Diary",
            location_id="LOC_DUNNE_HOUSE", owner_id="ENT_AMY",
            properties={"state": "planted", "purpose": "frame_nick"},
            affordances=[Affordance(action="incriminate", target_type="Entity")],
        ),
        "OBJ_TREASURE_HUNT": NarrativeObject(
            id="OBJ_TREASURE_HUNT", name="Anniversary Treasure Hunt Clues",
            location_id=None, owner_id="ENT_AMY",
            properties={"state": "staged", "purpose": "implicate_nick"},
            affordances=[Affordance(action="mislead", target_type="Entity")],
        ),
        "OBJ_BOX_CUTTER": NarrativeObject(
            id="OBJ_BOX_CUTTER", name="Box Cutter",
            location_id="LOC_DESI_LAKE_HOUSE", owner_id=None,
            properties={"state": "concealed", "purpose": "kill_desi"},
            affordances=[Affordance(action="kill", target_type="Entity")],
        ),
        "OBJ_PUPPET_STRINGS": NarrativeObject(
            id="OBJ_PUPPET_STRINGS", name="Punch-and-Judy Puppets (handle missing)",
            location_id="LOC_WOODLAWN", owner_id="ENT_AMY",
            properties={"state": "hidden_in_woodshed", "purpose": "evidence"},
            affordances=[Affordance(action="signal_death_penalty", target_type="Entity")],
        ),
        "OBJ_SPERM_SAMPLE": NarrativeObject(
            id="OBJ_SPERM_SAMPLE", name="Stored Fertility-Clinic Sample",
            location_id="LOC_NEW_YORK", owner_id="ENT_NICK",
            properties={"state": "frozen", "purpose": "binding_pregnancy"},
            affordances=[Affordance(action="conceive_under_duress", target_type="Entity")],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    entities={
        "ENT_NICK": Entity(
            id="ENT_NICK", name="Nick Dunne",
            location_id="LOC_DUNNE_HOUSE", status="healthy",
            traits={
                "charm":        TraitVector(value=0.7, inertia=0.4, evidence_strength="strong"),
                "cowardice":    TraitVector(value=0.6, inertia=0.4, evidence_strength="moderate"),
                "deception":    TraitVector(value=0.5, inertia=0.3, evidence_strength="moderate"),
                "resentment":   TraitVector(value=0.6, inertia=0.4, evidence_strength="moderate"),
                "adaptability": TraitVector(value=0.5, inertia=0.3, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_AMY",
                       perceived_state="Amy is controlling and impossible to please",
                       confidence=0.8, inertia=0.5,
                       established_at_fabula=1000, evidence_strength="moderate"),
                Belief(target_id="ENT_MARGO",
                       perceived_state="Margo is the only person who truly knows me",
                       confidence=0.9, inertia=0.6,
                       established_at_fabula=0, evidence_strength="strong"),
            ],
            constants=["writer", "twin_bond", "missouri_native"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=2000, triggered_by="EVT_AMY_DISAPPEARS",
                    traits={
                        "deception": TraitVector(value=0.7, inertia=0.4, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_NICK_AFFAIR_REVEALED",
                    traits={
                        "charm":      TraitVector(value=0.5, inertia=0.5, evidence_strength="strong"),
                        "cowardice":  TraitVector(value=0.65, inertia=0.5, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_ANDIE",
                               perceived_state="Andie has been a liability I cannot contain",
                               confidence=0.9, inertia=0.5,
                               established_at_fabula=6000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_NICK_TV_CONFESSION",
                    traits={
                        "adaptability": TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                        "deception":    TraitVector(value=0.8, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_AMY"],
                    beliefs_added=[
                        Belief(target_id="ENT_AMY",
                               perceived_state="Amy framed me and I must out-perform her on her own stage",
                               confidence=0.9, inertia=0.6,
                               established_at_fabula=9000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=14000, triggered_by="EVT_AMY_RETURNS",
                    traits={
                        "cowardice": TraitVector(value=0.75, inertia=0.55, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=16000, triggered_by="EVT_NICK_STAYS",
                    traits={
                        "adaptability": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                        "resentment":   TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_AMY",
                               perceived_state="I will play the perfect husband and write the truth in secret",
                               confidence=0.95, inertia=0.7,
                               established_at_fabula=16000, evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_AMY": Entity(
            id="ENT_AMY", name="Amy Elliott Dunne",
            location_id="LOC_DUNNE_HOUSE", status="healthy",
            traits={
                "intelligence":    TraitVector(value=0.95, inertia=0.6, evidence_strength="strong"),
                "narcissism":      TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
                "manipulation":    TraitVector(value=0.9, inertia=0.5, evidence_strength="strong"),
                "vindictiveness":  TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
                "control":         TraitVector(value=0.9, inertia=0.5, evidence_strength="strong"),
                "performativity":  TraitVector(value=0.9, inertia=0.6, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_NICK",
                       perceived_state="Nick has betrayed everything I gave him and must be punished publicly",
                       confidence=0.95, inertia=0.6,
                       established_at_fabula=1500, evidence_strength="strong"),
                Belief(target_id="ENT_DESI",
                       perceived_state="Desi is a useful pawn who still believes I belong to him",
                       confidence=0.8, inertia=0.4,
                       established_at_fabula=500, evidence_strength="moderate"),
            ],
            constants=["amazing_amy", "trust_fund_heiress", "harvard_psychology"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=2000, triggered_by="EVT_AMY_DISAPPEARS",
                    location_id="LOC_HIDEOUT_OZARKS",
                    traits={
                        "control": TraitVector(value=0.95, inertia=0.65, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=10000, triggered_by="EVT_AMY_ROBBED",
                    location_id="LOC_DESI_LAKE_HOUSE",
                    traits={
                        "control": TraitVector(value=0.7, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_DESI",
                               perceived_state="I need Desi's house and money — and then I need him gone",
                               confidence=0.85, inertia=0.5,
                               established_at_fabula=10000, evidence_strength="moderate"),
                    ]),
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_AMY_KILLS_DESI",
                    traits={
                        "vindictiveness": TraitVector(value=0.95, inertia=0.6, evidence_strength="strong"),
                        "control":        TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_DESI"]),
                EntityStateSnapshot(fabula_time=14000, triggered_by="EVT_AMY_RETURNS",
                    location_id="LOC_DUNNE_HOUSE",
                    traits={
                        "performativity": TraitVector(value=1.0, inertia=0.75, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_MARGO": Entity(
            id="ENT_MARGO", name="Margo Dunne",
            location_id="LOC_BAR", status="healthy",
            traits={
                "loyalty":         TraitVector(value=0.9, inertia=0.6, evidence_strength="strong"),
                "perceptiveness":  TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                "directness":      TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_NICK",
                       perceived_state="Nick is flawed but not a killer",
                       confidence=0.8, inertia=0.5,
                       established_at_fabula=0, evidence_strength="strong"),
            ],
            constants=["twin_bond", "co_owner_bar"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_NICK_AFFAIR_REVEALED",
                    traits={
                        "loyalty":        TraitVector(value=0.7, inertia=0.65, evidence_strength="strong"),
                        "perceptiveness": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_NICK"],
                    beliefs_added=[
                        Belief(target_id="ENT_NICK",
                               perceived_state="Nick lied to me — but he may still be innocent of murder",
                               confidence=0.6, inertia=0.4,
                               established_at_fabula=6000, evidence_strength="moderate"),
                    ]),
                EntityStateSnapshot(fabula_time=14000, triggered_by="EVT_AMY_RETURNS",
                    traits={
                        "perceptiveness": TraitVector(value=0.9, inertia=0.65, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_AMY",
                               perceived_state="Amy is lying and Nick is now her hostage",
                               confidence=0.95, inertia=0.7,
                               established_at_fabula=14000, evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_BONEY": Entity(
            id="ENT_BONEY", name="Detective Rhonda Boney",
            location_id="LOC_POLICE_STATION", status="healthy",
            traits={
                "thoroughness": TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
                "skepticism":   TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_NICK",
                       perceived_state="Nick is the prime suspect but the evidence is too clean",
                       confidence=0.7, inertia=0.4,
                       established_at_fabula=3000, evidence_strength="moderate"),
            ],
            constants=["lead_detective"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=14000, triggered_by="EVT_AMY_RETURNS",
                    traits={
                        "skepticism": TraitVector(value=0.9, inertia=0.6, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_NICK"],
                    beliefs_added=[
                        Belief(target_id="ENT_AMY",
                               perceived_state="Amy's kidnapping story does not survive a single careful question",
                               confidence=0.85, inertia=0.6,
                               established_at_fabula=14000, evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_DESI": Entity(
            id="ENT_DESI", name="Desi Collings",
            location_id="LOC_DESI_LAKE_HOUSE", status="healthy",
            traits={
                "obsession":      TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
                "possessiveness": TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                "wealth":         TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_AMY",
                       perceived_state="Amy finally needs me as she always should have",
                       confidence=0.8, inertia=0.4,
                       established_at_fabula=11000, evidence_strength="moderate"),
            ],
            constants=["old_money", "obsessive_ex"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=11000, triggered_by="EVT_AMY_GOES_TO_DESI",
                    location_id="LOC_DESI_LAKE_HOUSE",
                    traits={
                        "possessiveness": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_AMY_KILLS_DESI",
                    status="dead"),
            ],
        ),
        "ENT_TANNER": Entity(
            id="ENT_TANNER", name="Tanner Bolt",
            location_id="LOC_POLICE_STATION", status="healthy",
            traits={
                "cunning":     TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
                "showmanship": TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_NICK",
                       perceived_state="Nick is innocent but terrible at looking innocent on camera",
                       confidence=0.7, inertia=0.4,
                       established_at_fabula=8000, evidence_strength="strong"),
            ],
            constants=["celebrity_lawyer", "media_strategist"],
        ),
        "ENT_ANDIE": Entity(
            id="ENT_ANDIE", name="Andie Hardy",
            location_id="LOC_WOODLAWN", status="healthy",
            traits={
                "naivety":       TraitVector(value=0.7, inertia=0.4, evidence_strength="moderate"),
                "impulsiveness": TraitVector(value=0.6, inertia=0.3, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_NICK",
                       perceived_state="Nick loves me and will leave Amy",
                       confidence=0.7, inertia=0.3,
                       established_at_fabula=4000, evidence_strength="moderate"),
            ],
            constants=["former_student", "mistress"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_NICK_AFFAIR_REVEALED",
                    traits={
                        "impulsiveness": TraitVector(value=0.75, inertia=0.45, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_NICK"],
                    beliefs_added=[
                        Belief(target_id="ENT_NICK",
                               perceived_state="Nick used me and I will tell the cameras",
                               confidence=0.9, inertia=0.55,
                               established_at_fabula=6000, evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_GILPIN": Entity(
            id="ENT_GILPIN", name="Officer Jim Gilpin",
            location_id="LOC_POLICE_STATION", status="healthy",
            traits={
                "bluntness":   TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
                "suspicion":   TraitVector(value=0.85, inertia=0.5, evidence_strength="strong"),
                "impatience":  TraitVector(value=0.7, inertia=0.4, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_NICK",
                       perceived_state="Nick is guilty — the husband always is",
                       confidence=0.85, inertia=0.5,
                       established_at_fabula=3000, evidence_strength="strong"),
            ],
            constants=["partner_to_boney"],
        ),
        "ENT_NOELLE": Entity(
            id="ENT_NOELLE", name="Noelle Hawthorne",
            location_id="LOC_DUNNE_HOUSE", status="healthy",
            traits={
                "sentimentality":    TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
                "performativity":    TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                "hostility_to_nick": TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_AMY",
                       perceived_state="Amy was my best friend and Nick killed her",
                       confidence=0.9, inertia=0.6,
                       established_at_fabula=3000, evidence_strength="strong"),
                Belief(target_id="ENT_AMY",
                       perceived_state="Amy confided to me that she was pregnant",
                       confidence=0.7, inertia=0.5,
                       established_at_fabula=3000, evidence_strength="moderate"),
            ],
            constants=["neighbour", "vigil_organiser"],
        ),
        "ENT_MARYBETH": Entity(
            id="ENT_MARYBETH", name="Marybeth Elliott",
            location_id="LOC_NEW_YORK", status="healthy",
            traits={
                "protectiveness": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "composure":      TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                "public_polish":  TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_AMY",
                       perceived_state="Amy is the real Amazing Amy and our public legacy",
                       confidence=0.9, inertia=0.7,
                       established_at_fabula=0, evidence_strength="strong"),
            ],
            constants=["amazing_amy_author", "elliott_matriarch"],
        ),
        "ENT_RAND": Entity(
            id="ENT_RAND", name="Rand Elliott",
            location_id="LOC_NEW_YORK", status="healthy",
            traits={
                "affability":     TraitVector(value=0.75, inertia=0.5, evidence_strength="strong"),
                "public_polish":  TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                "obliviousness":  TraitVector(value=0.6, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_NICK",
                       perceived_state="Nick is family until proven otherwise",
                       confidence=0.6, inertia=0.4,
                       established_at_fabula=0, evidence_strength="moderate"),
            ],
            constants=["amazing_amy_author"],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_MARRIAGE_CRUMBLES", fabula_time=1000, syuzhet_index=1, event_type="outcome",
                  actor_ids=["ENT_NICK", "ENT_AMY"], target_ids=[],
                  description="Nick and Amy's marriage deteriorates after the Recession layoffs and the move to North Carthage."),
        EventNode(id="EVT_AMY_DISAPPEARS", fabula_time=2000, syuzhet_index=2, event_type="choice",
                  actor_ids=["ENT_AMY"], target_ids=["ENT_NICK"],
                  description="Amy stages her own disappearance on their fifth wedding anniversary."),
        EventNode(id="EVT_DIARY_FOUND", fabula_time=3000, syuzhet_index=3, event_type="outcome",
                  actor_ids=["ENT_BONEY"], target_ids=["ENT_NICK"],
                  description="Police find Amy's fabricated diary painting Nick as escalating from neglect to violence."),
        EventNode(id="EVT_INTERROGATION", fabula_time=4000, syuzhet_index=19, event_type="outcome",
                  actor_ids=["ENT_BONEY"], target_ids=["ENT_NICK"],
                  description="Detective Boney interrogates Nick, probing inconsistencies in his alibi and his flat affect."),
        EventNode(id="EVT_TREASURE_HUNT_CLUES", fabula_time=5000, syuzhet_index=4, event_type="outcome",
                  actor_ids=["ENT_NICK"], target_ids=[],
                  description="Nick follows Amy's anniversary treasure hunt; each clue points to damning planted evidence."),
        EventNode(id="EVT_NICK_AFFAIR_REVEALED", fabula_time=6000, syuzhet_index=5, event_type="revelation",
                  actor_ids=["ENT_ANDIE"], target_ids=["ENT_NICK"],
                  description="Nick's affair with Andie becomes public, destroying his credibility with the press and the police."),
        EventNode(id="EVT_MEDIA_TURNS", fabula_time=7000, syuzhet_index=21, event_type="outcome",
                  actor_ids=[], target_ids=["ENT_NICK"],
                  description="National cable news, led by Ellen Abbott, declares Nick guilty of his wife's murder."),
        EventNode(id="EVT_NICK_HIRES_TANNER", fabula_time=8000, syuzhet_index=8, event_type="choice",
                  actor_ids=["ENT_NICK"], target_ids=["ENT_TANNER"],
                  description="Nick hires celebrity defence lawyer Tanner Bolt to fight back through the media."),
        EventNode(id="EVT_NICK_TV_CONFESSION", fabula_time=9000, syuzhet_index=9, event_type="choice",
                  actor_ids=["ENT_NICK"], target_ids=["ENT_AMY"],
                  description="Nick goes on TV to apologise and beg Amy to come home — performing the husband she always wanted."),
        EventNode(id="EVT_AMY_ROBBED", fabula_time=10000, syuzhet_index=11, event_type="outcome",
                  actor_ids=[], target_ids=["ENT_AMY"],
                  description="Amy is robbed at the Ozarks motel, losing the cash reserve that funded her hideout."),
        EventNode(id="EVT_AMY_GOES_TO_DESI", fabula_time=11000, syuzhet_index=12, event_type="choice",
                  actor_ids=["ENT_AMY"], target_ids=["ENT_DESI"],
                  description="Desperate, Amy contacts Desi and moves into his lake house under his protection."),
        EventNode(id="EVT_NICK_FIGURES_OUT", fabula_time=12000, syuzhet_index=13, event_type="revelation",
                  actor_ids=["ENT_NICK"], target_ids=[],
                  description="Nick deduces from the puppet handle that Amy is alive and framed him; he plans the public counter-move."),
        EventNode(id="EVT_AMY_KILLS_DESI", fabula_time=13000, syuzhet_index=14, event_type="choice",
                  actor_ids=["ENT_AMY"], target_ids=["ENT_DESI"],
                  description="Amy mutilates herself, seduces Desi, then murders him with the box cutter — staged as escape from a kidnapper."),
        EventNode(id="EVT_AMY_RETURNS", fabula_time=14000, syuzhet_index=15, event_type="outcome",
                  actor_ids=["ENT_AMY"], target_ids=["ENT_NICK"],
                  description="Amy returns home bloodied, cast nationally as a survivor; Nick is publicly trapped beside her."),
        EventNode(id="EVT_AMY_PREGNANCY_TRAP", fabula_time=15000, syuzhet_index=24, event_type="revelation",
                  actor_ids=["ENT_AMY"], target_ids=["ENT_NICK"],
                  description="Amy reveals she has used Nick's stored fertility-clinic sample to become pregnant, binding him for life."),
        EventNode(id="EVT_NICK_STAYS", fabula_time=16000, syuzhet_index=18, event_type="choice",
                  actor_ids=["ENT_NICK"], target_ids=["ENT_AMY"],
                  description="Nick deletes his exposé, dedicates himself to the role of perfect husband, and stays for the child."),

        # ── UTTERANCES (discrete on-page speech-acts; woven into syuzhet at the
        #    point the reader actually encounters them) ────────────────────
        EventNode(id="EVT_UTT_NICK_CONFESS_AFFAIR_TO_MARGO", event_type="utterance",
                  description="Nick privately confides the affair with Andie to his twin Margo at the bar, asking her to help him hide it from investigators.",
                  speaker_id="ENT_NICK", addressee_ids=["ENT_MARGO"],
                  actor_ids=["ENT_NICK"], target_ids=["EVT_NICK_AFFAIR_REVEALED"],
                  content="I've been sleeping with Andie — a former student. I was going to leave Amy.",
                  via_channel_id="CHN_NICK_MARGO_CONFIDANT", truth_value="true",
                  fabula_time=2000, syuzhet_index=7),
        EventNode(id="EVT_UTT_ANDIE_GOES_PUBLIC", event_type="utterance",
                  description="Andie comes forward to investigators and the press, exposing the affair and detonating Nick's public credibility.",
                  speaker_id="ENT_ANDIE", addressee_ids=["ENT_BONEY"],
                  actor_ids=["ENT_ANDIE"], target_ids=["EVT_NICK_AFFAIR_REVEALED", "EVT_MEDIA_TURNS"],
                  content="I had an affair with Nick Dunne for over a year; he told me he was leaving Amy.",
                  via_channel_id=None, truth_value="true",
                  fabula_time=6000, syuzhet_index=6),
        EventNode(id="EVT_UTT_NOELLE_PREGNANCY_CLAIM", event_type="utterance",
                  description="Noelle Hawthorne tells detectives Boney and Gilpin that Amy had secretly confided to her that she was pregnant and afraid of Nick.",
                  speaker_id="ENT_NOELLE", addressee_ids=["ENT_BONEY", "ENT_GILPIN"],
                  actor_ids=["ENT_NOELLE"], target_ids=["EVT_INTERROGATION", "EVT_MEDIA_TURNS"],
                  content="Amy told me she was pregnant — and that Nick didn't want the baby.",
                  via_channel_id=None, truth_value="false",
                  fabula_time=5000, syuzhet_index=20),
        EventNode(id="EVT_UTT_NOELLE_VIGIL_DENUNCIATION", event_type="utterance",
                  description="At the candlelight vigil for Amy, Noelle publicly turns the gathered crowd against Nick on camera.",
                  speaker_id="ENT_NOELLE", addressee_ids=["ENT_NICK"],
                  actor_ids=["ENT_NOELLE"], target_ids=["EVT_MEDIA_TURNS"],
                  content="Where is our Amy? Nick — what have you done to her? Give her back to us!",
                  via_channel_id=None, truth_value="performative",
                  fabula_time=6000, syuzhet_index=23),
        EventNode(id="EVT_UTT_MARYBETH_TV_APPEAL", event_type="utterance",
                  description="Marybeth Elliott steps before the cameras at a press conference and pleads on national television for Amy's safe return.",
                  speaker_id="ENT_MARYBETH", addressee_ids=["ENT_NICK", "ENT_NOELLE"],
                  actor_ids=["ENT_MARYBETH"], target_ids=["EVT_MEDIA_TURNS"],
                  content="Please — whoever has our Amazing Amy — bring our daughter home safely.",
                  via_channel_id=None, truth_value="performative",
                  fabula_time=5000, syuzhet_index=22),
        EventNode(id="EVT_UTT_NICK_TV_APOLOGY", event_type="utterance",
                  description="On Sharon Schieber's talk show, Nick performs a Tanner-coached apology aimed at Amy and the watching nation.",
                  speaker_id="ENT_NICK", addressee_ids=["ENT_AMY", "ENT_TANNER", "ENT_MARGO"],
                  actor_ids=["ENT_NICK"], target_ids=["EVT_NICK_TV_CONFESSION", "EVT_AMY_RETURNS"],
                  content="I failed Amy as a husband — I beg her to come home so I can be the man she deserves.",
                  via_channel_id=None, truth_value="false",
                  fabula_time=9000, syuzhet_index=10),
        EventNode(id="EVT_UTT_AMY_STAGED_HOMECOMING", event_type="utterance",
                  description="Bloodied, Amy steps onto the porch of the Dunne house and greets Nick for the cameras as if returning from captivity.",
                  speaker_id="ENT_AMY", addressee_ids=["ENT_NICK"],
                  actor_ids=["ENT_AMY"], target_ids=["EVT_AMY_RETURNS"],
                  content="I'm home, Nick — I'm so glad to be back with you.",
                  via_channel_id=None, truth_value="false",
                  fabula_time=14000, syuzhet_index=17),
        EventNode(id="EVT_UTT_AMY_KIDNAP_STATEMENT", event_type="utterance",
                  description="At the police station, Amy gives detectives Boney and Gilpin her fabricated account of being abducted and held by Desi Collings.",
                  speaker_id="ENT_AMY", addressee_ids=["ENT_BONEY", "ENT_GILPIN"],
                  actor_ids=["ENT_AMY"], target_ids=["EVT_AMY_KILLS_DESI", "EVT_AMY_RETURNS"],
                  content="Desi Collings kidnapped me from the house and held me at his lake house — I killed him with a box cutter to escape.",
                  via_channel_id=None, truth_value="false",
                  fabula_time=14100, syuzhet_index=16),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction ──
        CausalEdge(source_id="EVT_MARRIAGE_CRUMBLES", target_id="EVT_AMY_DISAPPEARS",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=1000, propagation_delay=1000),
        CausalEdge(source_id="EVT_AMY_DISAPPEARS", target_id="EVT_DIARY_FOUND",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2000, propagation_delay=1000),
        CausalEdge(source_id="EVT_DIARY_FOUND", target_id="EVT_INTERROGATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3000, propagation_delay=1000),
        CausalEdge(source_id="EVT_DIARY_FOUND", target_id="EVT_TREASURE_HUNT_CLUES",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3000, propagation_delay=2000),
        CausalEdge(source_id="EVT_TREASURE_HUNT_CLUES", target_id="EVT_NICK_FIGURES_OUT",
                   causality_type="chain_reaction", mechanism="epistemic", evidence_strength="strong",
                   causal_force=8.0, fabula_time=5000, propagation_delay=7000),
        CausalEdge(source_id="EVT_NICK_AFFAIR_REVEALED", target_id="EVT_MEDIA_TURNS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000, propagation_delay=1000),
        CausalEdge(source_id="EVT_MEDIA_TURNS", target_id="EVT_NICK_HIRES_TANNER",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000, propagation_delay=1000),
        CausalEdge(source_id="EVT_NICK_HIRES_TANNER", target_id="EVT_NICK_TV_CONFESSION",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=8000, propagation_delay=1000),
        CausalEdge(source_id="EVT_AMY_ROBBED", target_id="EVT_AMY_GOES_TO_DESI",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=10000, propagation_delay=1000),
        CausalEdge(source_id="EVT_AMY_GOES_TO_DESI", target_id="EVT_AMY_KILLS_DESI",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=11000, propagation_delay=2000),
        CausalEdge(source_id="EVT_NICK_TV_CONFESSION", target_id="EVT_AMY_KILLS_DESI",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=6.0, fabula_time=9000, propagation_delay=4000),
        CausalEdge(source_id="EVT_AMY_KILLS_DESI", target_id="EVT_AMY_RETURNS",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=13000, propagation_delay=1000),
        CausalEdge(source_id="EVT_AMY_RETURNS", target_id="EVT_AMY_PREGNANCY_TRAP",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=14000, propagation_delay=1000),
        CausalEdge(source_id="EVT_AMY_PREGNANCY_TRAP", target_id="EVT_NICK_STAYS",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=15000, propagation_delay=1000),

        # ── mutation ──
        CausalEdge(source_id="EVT_AMY_DISAPPEARS", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2000,
                   trait_target="deception", trait_delta=0.2),
        CausalEdge(source_id="EVT_NICK_AFFAIR_REVEALED", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=6000,
                   trait_target="charm", trait_delta=-0.2),
        CausalEdge(source_id="EVT_NICK_TV_CONFESSION", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=9000,
                   trait_target="adaptability", trait_delta=0.2),
        CausalEdge(source_id="EVT_AMY_ROBBED", target_id="ENT_AMY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=6.0, fabula_time=10000,
                   trait_target="control", trait_delta=-0.2),
        CausalEdge(source_id="EVT_AMY_KILLS_DESI", target_id="ENT_AMY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000,
                   trait_target="vindictiveness", trait_delta=0.15),
        CausalEdge(source_id="EVT_AMY_RETURNS", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=14000,
                   trait_target="cowardice", trait_delta=0.15),
        CausalEdge(source_id="EVT_NICK_AFFAIR_REVEALED", target_id="ENT_MARGO",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=6000,
                   trait_target="loyalty", trait_delta=-0.2),
        CausalEdge(source_id="EVT_NICK_STAYS", target_id="ENT_NICK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=16000,
                   trait_target="resentment", trait_delta=0.25),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_NICK_TV_CONFESSION", target_id="ENT_AMY",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=9000,
                   trait_target="affinity", trait_delta=0.3, rel_counterpart_id="ENT_NICK"),
        CausalEdge(source_id="EVT_AMY_GOES_TO_DESI", target_id="ENT_DESI",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="moderate",
                   causal_force=6.0, fabula_time=11000,
                   trait_target="affinity", trait_delta=0.3, rel_counterpart_id="ENT_AMY"),
        CausalEdge(source_id="EVT_AMY_RETURNS", target_id="ENT_NICK",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=14000,
                   trait_target="fear", trait_delta=0.5, rel_counterpart_id="ENT_AMY"),
        CausalEdge(source_id="EVT_NICK_AFFAIR_REVEALED", target_id="ENT_MARGO",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=6000,
                   trait_target="affinity", trait_delta=-0.2, rel_counterpart_id="ENT_NICK"),
        CausalEdge(source_id="EVT_AMY_PREGNANCY_TRAP", target_id="ENT_NICK",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=15000,
                   trait_target="power_dynamic", trait_delta=-0.4, rel_counterpart_id="ENT_AMY"),
        CausalEdge(source_id="EVT_AMY_KILLS_DESI", target_id="ENT_AMY",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000,
                   trait_target="affinity", trait_delta=-0.9, rel_counterpart_id="ENT_DESI"),

        # ── affordance_gate ──
        CausalEdge(source_id="OBJ_DIARY", target_id="EVT_DIARY_FOUND",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="strong",
                   causal_force=9.0, fabula_time=3000),
        CausalEdge(source_id="OBJ_BOX_CUTTER", target_id="EVT_AMY_KILLS_DESI",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000),
        CausalEdge(source_id="OBJ_TREASURE_HUNT", target_id="EVT_TREASURE_HUNT_CLUES",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="strong",
                   causal_force=7.0, fabula_time=5000),
        CausalEdge(source_id="OBJ_PUPPET_STRINGS", target_id="EVT_NICK_FIGURES_OUT",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="strong",
                   causal_force=8.0, fabula_time=12000),
        CausalEdge(source_id="OBJ_SPERM_SAMPLE", target_id="EVT_AMY_PREGNANCY_TRAP",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=15000),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_MEDIA_CIRCUS", target_id="ENT_NICK",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=6000),
        CausalEdge(source_id="LOC_DESI_LAKE_HOUSE", target_id="ENT_AMY",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=11000),
        CausalEdge(source_id="LOC_DUNNE_HOUSE", target_id="ENT_AMY",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1000),
        CausalEdge(source_id="LOC_POLICE_STATION", target_id="ENT_NICK",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=4000),

        # ── WORLD_ → Event (named-latent common-cause wiring) ──
        CausalEdge(source_id="WORLD_MEDIA_TRIAL", target_id="EVT_INTERROGATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=4000),
        CausalEdge(source_id="WORLD_MEDIA_TRIAL", target_id="EVT_NICK_TV_CONFESSION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=9000),
        CausalEdge(source_id="WORLD_MEDIA_TRIAL", target_id="EVT_MEDIA_TURNS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000),
        CausalEdge(source_id="WORLD_MEDIA_TRIAL", target_id="EVT_AMY_RETURNS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=14000),
        CausalEdge(source_id="WORLD_SUBURBAN_PERFORMANCE", target_id="EVT_MARRIAGE_CRUMBLES",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_SUBURBAN_PERFORMANCE", target_id="EVT_DIARY_FOUND",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=3000),
        CausalEdge(source_id="WORLD_SUBURBAN_PERFORMANCE", target_id="EVT_AMY_DISAPPEARS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2000),
        CausalEdge(source_id="WORLD_SUBURBAN_PERFORMANCE", target_id="EVT_NICK_STAYS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=16000),
        CausalEdge(source_id="WORLD_RECESSION_PRECARITY", target_id="EVT_MARRIAGE_CRUMBLES",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_RECESSION_PRECARITY", target_id="EVT_AMY_GOES_TO_DESI",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=11000),

        # ── orphan utterance wirings ──
        CausalEdge(source_id="EVT_MARRIAGE_CRUMBLES", target_id="EVT_UTT_NICK_CONFESS_AFFAIR_TO_MARGO",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=4.0, fabula_time=1000, propagation_delay=1000),
        CausalEdge(source_id="EVT_INTERROGATION", target_id="EVT_UTT_NOELLE_PREGNANCY_CLAIM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=4000, propagation_delay=1000),
        CausalEdge(source_id="EVT_INTERROGATION", target_id="EVT_UTT_MARYBETH_TV_APPEAL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4000, propagation_delay=1000),
        CausalEdge(source_id="EVT_UTT_ANDIE_GOES_PUBLIC", target_id="EVT_NICK_AFFAIR_REVEALED",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_NOELLE_VIGIL_DENUNCIATION", target_id="EVT_MEDIA_TURNS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=6000, propagation_delay=1000),
        CausalEdge(source_id="EVT_NICK_TV_CONFESSION", target_id="EVT_UTT_NICK_TV_APOLOGY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=9000, propagation_delay=0),
        CausalEdge(source_id="EVT_AMY_RETURNS", target_id="EVT_UTT_AMY_STAGED_HOMECOMING",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=14000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_AMY_STAGED_HOMECOMING", target_id="EVT_UTT_AMY_KIDNAP_STATEMENT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=14000, propagation_delay=100),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_DUNNE_HOUSE", target_id="LOC_BAR"),
        SpatialEdge(source_id="LOC_DUNNE_HOUSE", target_id="LOC_WOODLAWN"),
        SpatialEdge(source_id="LOC_DUNNE_HOUSE", target_id="LOC_POLICE_STATION"),
        SpatialEdge(source_id="LOC_HIDEOUT_OZARKS", target_id="LOC_DESI_LAKE_HOUSE"),
        SpatialEdge(source_id="LOC_NEW_YORK", target_id="LOC_DUNNE_HOUSE"),
        SpatialEdge(source_id="LOC_POLICE_STATION", target_id="LOC_MEDIA_CIRCUS"),
        SpatialEdge(source_id="LOC_DESI_LAKE_HOUSE", target_id="LOC_DUNNE_HOUSE"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    channels={
        "CHN_DIARY_EVIDENCE": Channel(
            id="CHN_DIARY_EVIDENCE",
            name="Amy's Planted Diary as Evidence Pipeline",
            medium="diary",
            # Amy authors entries over a year; the diary is recovered by Boney
            # and seeds the police narrative across many sessions of reading.
            participant_ids=["ENT_AMY", "OBJ_DIARY", "ENT_BONEY"],
            directionality="simplex",
            intelligibility={},
            established_at_fabula=500,
            terminated_at_fabula=None,
            evidence_strength="strong",
        ),
        "CHN_NICK_MARGO_CONFIDANT": Channel(
            id="CHN_NICK_MARGO_CONFIDANT",
            name="Nick & Margo Sibling Confidant Channel",
            medium="private_conversation",
            participant_ids=["ENT_NICK", "ENT_MARGO"],
            directionality="duplex",
            intelligibility={},
            established_at_fabula=0,
            terminated_at_fabula=None,
            evidence_strength="strong",
        ),
        "CHN_LEGAL_STRATEGY": Channel(
            id="CHN_LEGAL_STRATEGY",
            name="Tanner Bolt Attorney-Client Strategy Sessions",
            medium="attorney_client",
            participant_ids=["ENT_TANNER", "ENT_NICK"],
            directionality="duplex",
            intelligibility={},
            established_at_fabula=8000,
            terminated_at_fabula=None,
            evidence_strength="strong",
        ),
        "CHN_DETECTIVE_PARTNERSHIP": Channel(
            id="CHN_DETECTIVE_PARTNERSHIP",
            name="Boney & Gilpin Detective Partnership",
            medium="partner_consultation",
            participant_ids=["ENT_BONEY", "ENT_GILPIN"],
            directionality="duplex",
            # Gilpin reads Nick as guilty earlier and more rigidly than Boney;
            # they don't fully share interpretive frame.
            intelligibility={"ENT_GILPIN": 0.7},
            established_at_fabula=2000,
            terminated_at_fabula=None,
            evidence_strength="moderate",
        ),
    },

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_MEDIA_TRIAL": GlobalTrait(
            id="WORLD_MEDIA_TRIAL",
            name="Trial by Media",
            description="The 24-hour cable-news cycle and its pundit-court (Ellen Abbott et al.) that runs a parallel justice system in which public opinion is weaponised and truth is irrelevant. Operates as common-cause parent over the interrogation, the affair-fuelled media turn, Nick's televised apology, and Amy's staged-survivor return.",
            category="social_structure",
            magnitude=TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
            affected_domains=["social", "psychological", "informational"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=7000, triggered_by="EVT_MEDIA_TURNS",
                    magnitude=TraitVector(value=0.9, inertia=0.6, evidence_strength="strong"),
                    description="Amy's disappearance becomes a national media spectacle that traps Nick in the role of presumed-guilty husband."),
                WorldTraitSnapshot(fabula_time=14000, triggered_by="EVT_AMY_RETURNS",
                    magnitude=TraitVector(value=1.0, inertia=0.7, evidence_strength="strong"),
                    description="Amy's bloodied return inverts the narrative — she now controls the cameras absolutely."),
            ],
        ),
        "WORLD_SUBURBAN_PERFORMANCE": GlobalTrait(
            id="WORLD_SUBURBAN_PERFORMANCE",
            name="Marriage as Performance",
            description="The cultural expectation that couples perform happiness for neighbours, families, and (eventually) cameras. Authenticity is sacrificed for appearances. The fake diary, the staged return, and Nick's final capitulation all draw their power from this latent.",
            category="social_structure",
            magnitude=TraitVector(value=0.8, inertia=0.7, evidence_strength="strong"),
            affected_domains=["psychological", "social"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=16000, triggered_by="EVT_NICK_STAYS",
                    magnitude=TraitVector(value=1.0, inertia=0.9, evidence_strength="strong"),
                    description="Nick agrees to stay married and perform the perfect-couple charade indefinitely for the unborn child."),
            ],
        ),
        "WORLD_RECESSION_PRECARITY": GlobalTrait(
            id="WORLD_RECESSION_PRECARITY",
            name="Post-2008 Economic Precarity",
            description="The Great Recession that erased the Dunnes' Brooklyn-magazine careers, drained Amy's trust fund into Nick's bar, and stranded the couple in foreclosed-out North Carthage. The latent that made the move, the resentment, and Amy's eventual cash-stranded retreat to Desi all materially possible.",
            category="economy",
            magnitude=TraitVector(value=0.75, inertia=0.8, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
        ),
    },

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────
    social_topology=[
        # Amy → Nick — the marriage-as-cage core dyad.
        RelationshipEdge(
            source_entity_id="ENT_AMY", target_entity_id="ENT_NICK",
            metrics={
                "affinity":      RelationshipMetric(value=-0.5, inertia=0.5, evidence_strength="strong", last_updated_fabula=14000),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="moderate", last_updated_fabula=9000),
                "power_dynamic": RelationshipMetric(value=0.75, inertia=0.7, evidence_strength="strong", last_updated_fabula=15000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_AMY",
            metrics={
                "affinity":      RelationshipMetric(value=-0.4, inertia=0.5, evidence_strength="strong", last_updated_fabula=9000),
                "fear":          RelationshipMetric(value=0.6, inertia=0.25, evidence_strength="strong", last_updated_fabula=14000),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=15000),
            },
        ),
        # Nick ↔ Margo — the only honest dyad in the book.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_MARGO",
            metrics={
                "affinity":      RelationshipMetric(value=0.9, inertia=0.55, evidence_strength="strong", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=0.0, inertia=0.6, evidence_strength="moderate", last_updated_fabula=6000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MARGO", target_entity_id="ENT_NICK",
            metrics={
                "affinity":      RelationshipMetric(value=0.8, inertia=0.5, evidence_strength="strong", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=0.0, inertia=0.6, evidence_strength="moderate", last_updated_fabula=6000),
            },
        ),
        # Amy ↔ Desi — instrumentalised obsession.
        RelationshipEdge(
            source_entity_id="ENT_AMY", target_entity_id="ENT_DESI",
            metrics={
                "affinity":      RelationshipMetric(value=-0.3, inertia=0.45, evidence_strength="strong", last_updated_fabula=13000),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=12000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.65, evidence_strength="strong", last_updated_fabula=13000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_DESI", target_entity_id="ENT_AMY",
            metrics={
                "affinity":      RelationshipMetric(value=0.7, inertia=0.5, evidence_strength="strong", last_updated_fabula=11000),
                "power_dynamic": RelationshipMetric(value=-0.4, inertia=0.65, evidence_strength="strong", last_updated_fabula=12000),
            },
        ),
        # Boney → Nick — investigator/suspect; cools on Amy's return.
        RelationshipEdge(
            source_entity_id="ENT_BONEY", target_entity_id="ENT_NICK",
            metrics={
                "affinity":      RelationshipMetric(value=-0.2, inertia=0.5, evidence_strength="moderate", last_updated_fabula=14000),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.7, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        # Nick → Andie — the affair.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_ANDIE",
            metrics={
                "affinity":      RelationshipMetric(value=0.3, inertia=0.4, evidence_strength="moderate", last_updated_fabula=6000),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.65, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ANDIE", target_entity_id="ENT_NICK",
            metrics={
                "affinity":      RelationshipMetric(value=-0.6, inertia=0.45, evidence_strength="strong", last_updated_fabula=6000),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="moderate", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=-0.55, inertia=0.65, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        # Nick ↔ Tanner — client/lawyer.
        RelationshipEdge(
            source_entity_id="ENT_NICK", target_entity_id="ENT_TANNER",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.45, evidence_strength="moderate", last_updated_fabula=9000),
                "power_dynamic": RelationshipMetric(value=-0.3, inertia=0.65, evidence_strength="strong", last_updated_fabula=8000),
            },
        ),
        # Margo ↔ Amy — sworn enemies behind the public smile.
        RelationshipEdge(
            source_entity_id="ENT_MARGO", target_entity_id="ENT_AMY",
            metrics={
                "affinity":      RelationshipMetric(value=-0.7, inertia=0.5, evidence_strength="strong", last_updated_fabula=14000),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=14000),
                "power_dynamic": RelationshipMetric(value=-0.4, inertia=0.65, evidence_strength="strong", last_updated_fabula=14000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_AMY", target_entity_id="ENT_MARGO",
            metrics={
                "affinity":      RelationshipMetric(value=-0.5, inertia=0.5, evidence_strength="strong", last_updated_fabula=14000),
                "power_dynamic": RelationshipMetric(value=0.4, inertia=0.65, evidence_strength="strong", last_updated_fabula=14000),
            },
        ),
        # Gilpin → Nick — uncomplicated hostility.
        RelationshipEdge(
            source_entity_id="ENT_GILPIN", target_entity_id="ENT_NICK",
            metrics={
                "affinity":      RelationshipMetric(value=-0.6, inertia=0.5, evidence_strength="strong", last_updated_fabula=4000),
                "power_dynamic": RelationshipMetric(value=0.5, inertia=0.7, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        # Boney ↔ Gilpin — partners.
        RelationshipEdge(
            source_entity_id="ENT_BONEY", target_entity_id="ENT_GILPIN",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.5, evidence_strength="moderate", last_updated_fabula=3000),
                "power_dynamic": RelationshipMetric(value=0.2, inertia=0.65, evidence_strength="moderate", last_updated_fabula=3000),
            },
        ),
        # Noelle → Amy / Nick.
        RelationshipEdge(
            source_entity_id="ENT_NOELLE", target_entity_id="ENT_AMY",
            metrics={
                "affinity":      RelationshipMetric(value=0.7, inertia=0.5, evidence_strength="moderate", last_updated_fabula=3000),
                "power_dynamic": RelationshipMetric(value=-0.3, inertia=0.65, evidence_strength="moderate", last_updated_fabula=3000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_NOELLE", target_entity_id="ENT_NICK",
            metrics={
                "affinity":      RelationshipMetric(value=-0.75, inertia=0.5, evidence_strength="strong", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=0.0, inertia=0.6, evidence_strength="moderate", last_updated_fabula=6000),
            },
        ),
        # Marybeth and Rand — the brand.
        RelationshipEdge(
            source_entity_id="ENT_MARYBETH", target_entity_id="ENT_AMY",
            metrics={
                "affinity":      RelationshipMetric(value=0.7, inertia=0.55, evidence_strength="strong", last_updated_fabula=2000),
                "power_dynamic": RelationshipMetric(value=0.4, inertia=0.7, evidence_strength="strong", last_updated_fabula=2000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MARYBETH", target_entity_id="ENT_NICK",
            metrics={
                "affinity":      RelationshipMetric(value=0.2, inertia=0.5, evidence_strength="moderate", last_updated_fabula=3000),
                "power_dynamic": RelationshipMetric(value=0.3, inertia=0.65, evidence_strength="moderate", last_updated_fabula=3000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_RAND", target_entity_id="ENT_NICK",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.5, evidence_strength="moderate", last_updated_fabula=3000),
                "power_dynamic": RelationshipMetric(value=0.2, inertia=0.65, evidence_strength="moderate", last_updated_fabula=3000),
            },
        ),
    ],
)
