# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Once Upon a Time in the West — high-fidelity WorldStateV1 fixture.

Sergio Leone (1968). Authored against the current ingestion prompts.
Demonstrates the five CausalEdge modalities, per-axis
``RelationshipMetric``, named-latent WORLD_ traits (the railroad as
inexorable cosmology, the dying Old West, the revenge contract that
binds Harmonica to Frank), and standing ``Channel``s for the Morton
commission, the bounty network, and Harmonica's wordless musical
identification.
"""
from shadow_loom.models import (
    Channel,
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge, RelationshipMetric,
    TraitVector, AmbientVector, Affordance, Belief, EntityStateSnapshot,
    GlobalTrait,
    NarrativeStyle,
    Concern, Proposition, ConcernSnapshot, PropositionSnapshot,
)

world_state = WorldStateV1(
    narrative_style=NarrativeStyle(
        format='synopsis',
        target_word_min=208,
        target_word_max=780,
        prose_density='sparse',
        voice='operatic Western synopsis; long held silences; ritual gunfight; third-person past tense',
        style_exemplar="A train arrives at the Old West town of Flagstone where a man with a harmonica kills three men attempting to ambush him. Meanwhile, Frank and his gang murder Brett McBain and his three children at his ranch Sweetwater. Shortly after, a former prostitute arrives at Sweetwater and reveals she is Jill McBain, who married McBain a month earlier in New Orleans.",
        source_word_count=520,
    ),

    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_FLAGSTONE_STATION": Location(
            name="Flagstone Railway Station",
            description="The dust-blown Western station where Harmonica steps off the first train; three of Frank's men wait to ambush him.",
            ambient_state={
                "ritual_violence": AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
                "iron_progress":   AmbientVector(value=0.8,  volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_SWEETWATER": Location(
            name="Sweetwater Ranch",
            description="Brett McBain's homestead, sitting on the only water for miles — and on the path the railroad must take.",
            ambient_state={
                "imminent_value": AmbientVector(value=0.95, volatility=0.4, evidence_strength="strong"),
                "blood_memory":   AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_MORTONS_TRAIN": Location(
            name="Morton's Private Railway Carriage",
            description="The luxurious carriage where the dying tycoon Morton issues commissions and counts the miles of track between him and the Pacific.",
            ambient_state={
                "industrial_capital":   AmbientVector(value=0.95, volatility=0.1, evidence_strength="strong"),
                "physical_decay":       AmbientVector(value=0.85, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_AUCTION_SQUARE": Location(
            name="Flagstone Auction Square",
            description="The dirt square where Sweetwater is auctioned and intimidated bidders fall silent under Frank's gaze.",
            ambient_state={
                "intimidation":  AmbientVector(value=0.9, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_DESERT_HANGING_ARCH": Location(
            name="Desert Hanging Arch (Flashback)",
            description="The arch in the desert where, years before, Frank made a young boy support his older brother on his shoulders, harmonica in mouth, until the boy collapsed.",
            ambient_state={
                "ritual_cruelty": AmbientVector(value=0.95, volatility=0.2, evidence_strength="strong"),
                "memory_charge":  AmbientVector(value=0.9,  volatility=0.1, evidence_strength="strong"),
            },
        ),
    },

    # ── OBJECTS ────────────────────────────────────────────────────────
    objects={
        "OBJ_HARMONICA": NarrativeObject(
            id="OBJ_HARMONICA", name="Harmonica (the instrument)",
            location_id="LOC_FLAGSTONE_STATION", owner_id="ENT_HARMONICA",
            properties={"state": "carried", "function": "memorial_weapon", "engraved_with": "brother's death"},
            affordances=[
                Affordance(action="identify_avenger", target_type="Entity"),
                Affordance(action="trigger_recognition", target_type="Entity"),
            ],
        ),
        "OBJ_BOUNTY_5000": NarrativeObject(
            id="OBJ_BOUNTY_5000", name="$5,000 Bounty on Cheyenne",
            location_id=None, owner_id=None,
            properties={"state": "outstanding", "function": "auction_capital"},
            affordances=[Affordance(action="purchase_sweetwater", target_type="Location")],
        ),
        "OBJ_FRAMING_EVIDENCE": NarrativeObject(
            id="OBJ_FRAMING_EVIDENCE", name="Cheyenne-Branded Duster",
            location_id="LOC_SWEETWATER", owner_id="ENT_FRANK",
            properties={"state": "planted", "function": "false_implication"},
            affordances=[Affordance(action="misdirect_blame", target_type="Entity")],
        ),
        "OBJ_RAILROAD_TRACKS": NarrativeObject(
            id="OBJ_RAILROAD_TRACKS", name="Pacific Railroad Right-of-Way",
            location_id="LOC_FLAGSTONE_STATION", owner_id="ENT_MORTON",
            properties={"state": "advancing", "destination": "Pacific Ocean"},
            affordances=[Affordance(action="reach_water_station_or_revert", target_type="Location")],
        ),
        "OBJ_CRUTCHES": NarrativeObject(
            id="OBJ_CRUTCHES", name="Morton's Crutches",
            location_id="LOC_MORTONS_TRAIN", owner_id="ENT_MORTON",
            properties={"state": "permanent", "condition": "spinal_tuberculosis"},
            affordances=[Affordance(action="immobilise_owner", target_type="Entity")],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    entities={
        "ENT_HARMONICA": Entity(
            id="ENT_HARMONICA", name="Harmonica",
            location_id="LOC_FLAGSTONE_STATION", status="healthy",
            traits={
                "vengeance":      TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                "patience":       TraitVector(value=0.95, inertia=0.9,  evidence_strength="strong"),
                "lethality":      TraitVector(value=0.95, inertia=0.9,  evidence_strength="strong"),
                "verbal_economy": TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                "compassion_for_jill": TraitVector(value=0.0, inertia=0.4, evidence_strength="weak"),
            },
            beliefs=[
                Belief(target_id="ENT_FRANK",
                       perceived_state="the man who hanged my brother — debt outstanding",
                       proposition_id="PROP_FRANK_HANGED_BROTHER",
                       confidence=0.99, inertia=0.95,
                       established_at_fabula=0, evidence_strength="strong"),
            ],
            concerns=[
                # The single defining standing concern — the entire film is the
                # discharge of this debt (Frijda action-readiness: revenge as
                # ritual, not impulse).
                Concern(concern_id="CCN_HARMONICA_AVENGES_BROTHER", proposition_id="PROP_HARMONICA_AVENGES_BROTHER",
                        polarity="desire", kind="revenge", salience=1.0,
                        state_timeline=[
                            ConcernSnapshot(fabula_time=3350, triggered_by="EVT_FRANK_DIES",
                                            salience=0.05),
                        ]),
                Concern(concern_id="CCN_HARMONICA_FRANK_RECOGNISES", proposition_id="PROP_HARMONICA_IDENTITY",
                        polarity="desire", kind="recognition", salience=0.85,
                        activation_fabula_window=[1100, 3300]),
                # Late-developing protective concern for Jill (compassion bumps
                # 0.0→0.6 at the auction).
                Concern(concern_id="CCN_HARMONICA_PROTECTS_JILL", proposition_id="PROP_JILL_KEEPS_SWEETWATER",
                        polarity="desire", kind="safety", salience=0.55,
                        activation_fabula_window=[2700, 3400]),
            ],
            state_timeline=[
                # Documents that Harmonica's baseline vengeance=0.95 is the
                # *result* of EVT_FLASHBACK_HANGING (fabula_time=0), satisfying
                # the mutation\u2194snapshot parity invariant. The backstory event
                # IS the trait's origin; without this anchor the snapshot
                # auditor (and counterfactual surgery) cannot tell that
                # rolling back the hanging should also roll back the obsession.
                EntityStateSnapshot(fabula_time=0, triggered_by="EVT_FLASHBACK_HANGING",
                    traits={
                        "vengeance": TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=2000, triggered_by="EVT_HARMONICA_SPIES_MORTON_TRAIN",
                    location_id="LOC_MORTONS_TRAIN",
                    beliefs_added=[
                        Belief(target_id="ENT_MORTON",
                               perceived_state="the railway tycoon who hired Frank",
                               proposition_id="PROP_MORTON_HIRES_FRANK",
                               confidence=0.9, inertia=0.8,
                               established_at_fabula=2000, evidence_strength="strong",
                               acquired_via_event_id="EVT_HARMONICA_SPIES_MORTON_TRAIN"),
                    ]),
                EntityStateSnapshot(fabula_time=2700, triggered_by="EVT_AUCTION",
                    traits={
                        "compassion_for_jill": TraitVector(value=0.6, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=3300, triggered_by="EVT_SHOWDOWN_FLASHBACK_REVEAL",
                    location_id="LOC_SWEETWATER",
                    traits={
                        "vengeance": TraitVector(value=0.0, inertia=0.95, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_FRANK": Entity(
            id="ENT_FRANK", name="Frank",
            location_id="LOC_SWEETWATER", status="healthy",
            traits={
                "cruelty":            TraitVector(value=0.95, inertia=0.9,  evidence_strength="strong"),
                "ambition":           TraitVector(value=0.85, inertia=0.8,  evidence_strength="strong"),
                "contempt_for_morton": TraitVector(value=0.7, inertia=0.7,  evidence_strength="moderate"),
                "curiosity_about_harmonica": TraitVector(value=0.0, inertia=0.4, evidence_strength="weak"),
            },
            beliefs=[
                Belief(target_id="ENT_MORTON",
                       perceived_state="my paymaster, soon to be obsolete",
                       proposition_id="PROP_MORTON_HIRES_FRANK",
                       confidence=0.85, inertia=0.7,
                       established_at_fabula=500, evidence_strength="strong"),
            ],
            concerns=[
                # Frank's defining ambition — to own Sweetwater outright,
                # eclipsing his employer.
                Concern(concern_id="CCN_FRANK_OWNS_SWEETWATER", proposition_id="PROP_FRANK_BUYS_SWEETWATER",
                        polarity="desire", kind="power", salience=0.95,
                        activation_fabula_window=[1000, 3300]),
                # The unspoken fear that becomes a question — "who are you?" —
                # the curiosity-as-anxiety that erodes him through the auction.
                Concern(concern_id="CCN_FRANK_KNOW_HARMONICA", proposition_id="PROP_HARMONICA_IDENTITY",
                        polarity="desire", kind="discovery", salience=0.65,
                        activation_fabula_window=[2700, 3300],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=3300, triggered_by="EVT_SHOWDOWN_FLASHBACK_REVEAL",
                                            salience=1.0, kind="mortal_threat"),
                        ]),
                # Pride / contempt for the dying tycoon who hired him.
                Concern(concern_id="CCN_FRANK_OUTLIVES_MORTON", proposition_id="PROP_MORTON_REACHES_PACIFIC",
                        polarity="fear", kind="irrelevance", salience=0.55,
                        activation_fabula_window=[500, 3100]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_MCBAIN_FAMILY_MURDER",
                    location_id="LOC_SWEETWATER",
                    traits={
                        "ambition": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=2700, triggered_by="EVT_AUCTION",
                    traits={
                        "curiosity_about_harmonica": TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=3300, triggered_by="EVT_SHOWDOWN_FLASHBACK_REVEAL",
                    beliefs_added=[
                        Belief(target_id="ENT_HARMONICA",
                               perceived_state="the boy from the desert arch — my own crime, returned",
                               proposition_id="PROP_HARMONICA_IDENTITY",
                               confidence=0.99, inertia=0.95,
                               established_at_fabula=3300, evidence_strength="strong",
                               acquired_via_event_id="EVT_SHOWDOWN_FLASHBACK_REVEAL"),
                    ]),
                EntityStateSnapshot(fabula_time=3350, triggered_by="EVT_FRANK_DIES",
                    status="dead"),
            ],
        ),
        "ENT_CHEYENNE": Entity(
            id="ENT_CHEYENNE", name="Cheyenne",
            location_id="LOC_FLAGSTONE_STATION", status="healthy",
            traits={
                "outlaw_code":      TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "loyalty":          TraitVector(value=0.75, inertia=0.7,  evidence_strength="strong"),
                "wry_humour":       TraitVector(value=0.85, inertia=0.8,  evidence_strength="strong"),
                "fugitive_anxiety": TraitVector(value=0.6,  inertia=0.5,  evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_FRANK",
                       perceived_state="the bastard who framed me",
                       proposition_id="PROP_FRANK_FRAMED_CHEYENNE",
                       confidence=0.9, inertia=0.85,
                       established_at_fabula=1500, evidence_strength="strong"),
            ],
            concerns=[
                # Cheyenne's outlaw-code injury: being framed for a massacre
                # he didn't commit (Averill normative_violation).
                Concern(concern_id="CCN_CHEYENNE_CLEAR_NAME", proposition_id="PROP_FRANK_FRAMED_CHEYENNE",
                        polarity="fear", kind="betrayal", salience=0.9,
                        activation_fabula_window=[1300, 3300]),
                # Loyalty to Harmonica after the rescue — the alliance that
                # carries Cheyenne to the train gunfight.
                Concern(concern_id="CCN_CHEYENNE_HELPS_HARMONICA", proposition_id="PROP_HARMONICA_AVENGES_BROTHER",
                        polarity="desire", kind="loyalty", salience=0.7,
                        activation_fabula_window=[2200, 3400]),
                # Background fugitive anxiety — the bounty network is always
                # closing.
                Concern(concern_id="CCN_CHEYENNE_AVOIDS_BOUNTY", proposition_id="PROP_CHEYENNE_GUILTY_OF_MASSACRE",
                        polarity="fear", kind="discovery", salience=0.5),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=2200, triggered_by="EVT_CHEYENNE_RESCUES_HARMONICA",
                    traits={
                        "loyalty": TraitVector(value=0.9, inertia=0.8, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=3400, triggered_by="EVT_CHEYENNE_RIDES_OFF_DYING",
                    status="injured", location_id=None),
            ],
        ),
        "ENT_JILL": Entity(
            id="ENT_JILL", name="Jill McBain",
            location_id="LOC_FLAGSTONE_STATION", status="healthy",
            traits={
                "resilience":       TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "pragmatism":       TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "grief":            TraitVector(value=0.0,  inertia=0.4,  evidence_strength="weak"),
                "moral_pliability": TraitVector(value=0.6,  inertia=0.5,  evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_BRETT_MCBAIN",
                       perceived_state="the husband who promised me a new life",
                       proposition_id="PROP_MCBAIN_FAMILY_SAFE",
                       confidence=0.95, inertia=0.85,
                       established_at_fabula=900, evidence_strength="strong"),
            ],
            concerns=[
                # The defining concern — keeping Sweetwater is keeping the
                # promise of a new life.
                Concern(concern_id="CCN_JILL_KEEPS_SWEETWATER", proposition_id="PROP_JILL_KEEPS_SWEETWATER",
                        polarity="desire", kind="survival", salience=1.0,
                        activation_fabula_window=[1200, 3400]),
                Concern(concern_id="CCN_JILL_FEARS_FRANK", proposition_id="PROP_FRANK_BUYS_SWEETWATER",
                        polarity="fear", kind="mortal_threat", salience=0.85,
                        activation_fabula_window=[1200, 2950],
                        counter_concern_ids=["CCN_JILL_KEEPS_SWEETWATER"],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=3350, triggered_by="EVT_FRANK_DIES",
                                            salience=0.0),
                        ]),
                Concern(concern_id="CCN_JILL_NEW_LIFE", proposition_id="PROP_RAILROAD_REACHES_SWEETWATER",
                        polarity="desire", kind="freedom", salience=0.75,
                        activation_fabula_window=[1200, 3400]),
                Concern(concern_id="CCN_JILL_GRIEVES_FAMILY", proposition_id="PROP_MCBAIN_FAMILY_SAFE",
                        polarity="desire", kind="loss_of_loved_one", salience=0.6,
                        activation_fabula_window=[1200, 1700]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1200, triggered_by="EVT_JILL_ARRIVES_AT_SWEETWATER",
                    location_id="LOC_SWEETWATER",
                    traits={
                        "grief": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_BRETT_MCBAIN",
                               perceived_state="murdered before I arrived",
                               proposition_id="PROP_MCBAIN_FAMILY_MURDERED",
                               confidence=0.95, inertia=0.85,
                               established_at_fabula=1200, evidence_strength="strong",
                               acquired_via_event_id="EVT_JILL_ARRIVES_AT_SWEETWATER"),
                    ]),
                EntityStateSnapshot(fabula_time=2500, triggered_by="EVT_FRANK_FORCES_JILL",
                    traits={
                        "moral_pliability": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=3400, triggered_by="EVT_BUILDING_STATION_AT_SWEETWATER",
                    location_id="LOC_SWEETWATER",
                    traits={
                        "resilience": TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_BRETT_MCBAIN": Entity(
            id="ENT_BRETT_MCBAIN", name="Brett McBain",
            location_id="LOC_SWEETWATER", status="healthy",
            traits={
                "frontier_vision": TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
                "fatherly_love":   TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="OBJ_RAILROAD_TRACKS",
                       perceived_state="will reach Sweetwater and make us rich",
                       proposition_id="PROP_RAILROAD_REACHES_SWEETWATER",
                       confidence=0.9, inertia=0.85,
                       established_at_fabula=400, evidence_strength="strong"),
            ],
            concerns=[
                Concern(concern_id="CCN_MCBAIN_FRONTIER_FORTUNE", proposition_id="PROP_RAILROAD_REACHES_SWEETWATER",
                        polarity="desire", kind="power", salience=0.95),
                Concern(concern_id="CCN_MCBAIN_FAMILY_SAFE", proposition_id="PROP_MCBAIN_FAMILY_SAFE",
                        polarity="desire", kind="love", salience=0.95),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_MCBAIN_FAMILY_MURDER",
                    status="dead", location_id="LOC_SWEETWATER"),
            ],
        ),
        "ENT_MORTON": Entity(
            id="ENT_MORTON", name="Mr. Morton",
            location_id="LOC_MORTONS_TRAIN", status="ill",
            traits={
                "industrial_will":   TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                "physical_frailty":  TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "longing_for_pacific": TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_FRANK",
                       perceived_state="useful tool, dangerous if unleashed too far",
                       proposition_id="PROP_MORTON_HIRES_FRANK",
                       confidence=0.85, inertia=0.7,
                       established_at_fabula=500, evidence_strength="strong"),
            ],
            concerns=[
                # The film's organising desire — Morton sees the Pacific from
                # his crutches.
                Concern(concern_id="CCN_MORTON_REACHES_PACIFIC", proposition_id="PROP_MORTON_REACHES_PACIFIC",
                        polarity="desire", kind="power", salience=1.0),
                Concern(concern_id="CCN_MORTON_FRANK_OBEYS", proposition_id="PROP_FRANK_EXCEEDS_BRIEF",
                        polarity="fear", kind="betrayal", salience=0.85,
                        activation_fabula_window=[500, 3100]),
                Concern(concern_id="CCN_MORTON_FEARS_DECLINE", proposition_id="PROP_MORTON_REACHES_PACIFIC",
                        polarity="fear", kind="death", salience=0.7,
                        counter_concern_ids=["CCN_MORTON_REACHES_PACIFIC"]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_MCBAIN_FAMILY_MURDER",
                    beliefs_added=[
                        Belief(target_id="ENT_FRANK",
                               perceived_state="my hireling has exceeded the brief",
                               proposition_id="PROP_FRANK_EXCEEDS_BRIEF",
                               confidence=0.95, inertia=0.8,
                               established_at_fabula=1000, evidence_strength="strong",
                               acquired_via_event_id="EVT_MCBAIN_FAMILY_MURDER",
                               acquired_via_channel_id="CHN_MORTON_FRANK_COMMISSION"),
                    ]),
                EntityStateSnapshot(fabula_time=3100, triggered_by="EVT_TRAIN_GUNFIGHT",
                    status="dead", location_id="LOC_MORTONS_TRAIN"),
            ],
        ),
        "ENT_FRANKS_GANG": Entity(
            id="ENT_FRANKS_GANG", name="Frank's Henchmen",
            location_id="LOC_FLAGSTONE_STATION", status="healthy",
            traits={
                "venality": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "lethality": TraitVector(value=0.7, inertia=0.6, evidence_strength="strong"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3100, triggered_by="EVT_TRAIN_GUNFIGHT",
                    status="dead"),
            ],
        ),
        "ENT_CHEYENNES_GANG": Entity(
            id="ENT_CHEYENNES_GANG", name="Cheyenne's Gang",
            location_id="LOC_FLAGSTONE_STATION", status="healthy",
            traits={
                "loyalty_to_cheyenne": TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3100, triggered_by="EVT_TRAIN_GUNFIGHT",
                    status="dead"),
            ],
        ),
        "ENT_HARMONICAS_BROTHER": Entity(
            id="ENT_HARMONICAS_BROTHER", name="Harmonica's Older Brother",
            location_id="LOC_DESERT_HANGING_ARCH", status="dead",
            traits={
                "stoic_acceptance": TraitVector(value=0.9, inertia=0.95, evidence_strength="strong"),
            },
            beliefs=[],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_FLASHBACK_HANGING", fabula_time=0, syuzhet_index=18,
                  event_type="outcome", actor_ids=["ENT_FRANK"],
                  target_ids=["ENT_HARMONICAS_BROTHER", "ENT_HARMONICA"],
                  description="Years before, Frank prepared to hang a man by making the man's younger brother support him on his shoulders, a harmonica forced into the boy's mouth; eventually the boy collapses, dooming his brother."),
        EventNode(id="EVT_MORTON_HIRES_FRANK", fabula_time=500, syuzhet_index=8,
                  event_type="choice", actor_ids=["ENT_MORTON"], target_ids=["ENT_FRANK"],
                  description="Morton, dying of spinal tuberculosis and obsessed with seeing his railroad reach the Pacific, hires Frank to intimidate — not murder — Brett McBain off Sweetwater."),
        EventNode(id="EVT_MCBAIN_FAMILY_MURDER", fabula_time=1000, syuzhet_index=2,
                  event_type="outcome", actor_ids=["ENT_FRANK", "ENT_FRANKS_GANG"],
                  target_ids=["ENT_BRETT_MCBAIN"],
                  description="Frank and his gang massacre Brett McBain and his three children at Sweetwater as they prepare for a celebration — exceeding Morton's brief."),
        EventNode(id="EVT_HARMONICA_ARRIVES", fabula_time=1100, syuzhet_index=1,
                  event_type="outcome", actor_ids=["ENT_HARMONICA"], target_ids=["ENT_FRANKS_GANG"],
                  description="A train arrives at Flagstone; the man with a harmonica steps off and kills three of Frank's men sent to ambush him, then waits."),
        EventNode(id="EVT_JILL_ARRIVES_AT_SWEETWATER", fabula_time=1200, syuzhet_index=3,
                  event_type="outcome", actor_ids=["ENT_JILL"], target_ids=["ENT_BRETT_MCBAIN"],
                  description="Jill McBain — a former New Orleans prostitute, married to Brett a month earlier — arrives at Sweetwater to find her new family murdered."),
        EventNode(id="EVT_FRANK_FRAMES_CHEYENNE", fabula_time=1300, syuzhet_index=4,
                  event_type="choice", actor_ids=["ENT_FRANK"], target_ids=["ENT_CHEYENNE"],
                  description="Frank plants evidence at the murder scene implicating the outlaw Cheyenne."),
        EventNode(id="EVT_HARMONICA_MEETS_CHEYENNE", fabula_time=1500, syuzhet_index=5,
                  event_type="choice", actor_ids=["ENT_HARMONICA", "ENT_CHEYENNE"], target_ids=[],
                  description="Harmonica encounters Cheyenne, now a fugitive; Cheyenne denies sending the ambushers and the two begin a wary alliance."),
        EventNode(id="EVT_HARMONICA_SAVES_JILL", fabula_time=1700, syuzhet_index=6,
                  event_type="outcome", actor_ids=["ENT_HARMONICA"], target_ids=["ENT_JILL", "ENT_FRANKS_GANG"],
                  description="Harmonica kills two of Frank's men sent to murder Jill at Sweetwater."),
        EventNode(id="EVT_HARMONICA_SPIES_MORTON_TRAIN", fabula_time=2000, syuzhet_index=7,
                  event_type="choice", actor_ids=["ENT_HARMONICA"], target_ids=["ENT_MORTON"],
                  description="Harmonica spies out Morton's private railway carriage and discovers the connection between Frank and the dying tycoon — but is captured."),
        EventNode(id="EVT_HARMONICA_CAPTURED", fabula_time=2100, syuzhet_index=9,
                  event_type="outcome", actor_ids=["ENT_FRANKS_GANG"], target_ids=["ENT_HARMONICA"],
                  description="Frank's men capture Harmonica; Frank is called away before he can deal with him personally."),
        EventNode(id="EVT_CHEYENNE_RESCUES_HARMONICA", fabula_time=2200, syuzhet_index=10,
                  event_type="outcome", actor_ids=["ENT_CHEYENNE"], target_ids=["ENT_HARMONICA"],
                  description="Cheyenne rescues Harmonica; the two formally agree to help Jill keep Sweetwater, using stockpiled materials to start building the water station."),
        EventNode(id="EVT_FRANK_FORCES_JILL", fabula_time=2500, syuzhet_index=11,
                  event_type="outcome", actor_ids=["ENT_FRANK"], target_ids=["ENT_JILL"],
                  description="Frank corners Jill in a sexually threatening encounter and forces her to put Sweetwater up for auction."),
        EventNode(id="EVT_AUCTION", fabula_time=2700, syuzhet_index=12,
                  event_type="choice", actor_ids=["ENT_HARMONICA", "ENT_CHEYENNE", "ENT_FRANK"], target_ids=["ENT_JILL"],
                  description="Frank's henchmen intimidate bidders so Frank can buy Sweetwater cheap; Harmonica appears with Cheyenne in tow and bids the $5,000 bounty on Cheyenne's head."),
        EventNode(id="EVT_MORTON_BRIBES_FRANKS_MEN", fabula_time=2900, syuzhet_index=13,
                  event_type="choice", actor_ids=["ENT_MORTON"], target_ids=["ENT_FRANK", "ENT_FRANKS_GANG"],
                  description="Morton, fearing Frank now wants Sweetwater for himself, bribes Frank's own men to kill him."),
        EventNode(id="EVT_HARMONICA_SAVES_FRANK", fabula_time=2950, syuzhet_index=14,
                  event_type="choice", actor_ids=["ENT_HARMONICA"], target_ids=["ENT_FRANK"],
                  description="Harmonica intervenes to save Frank from the bribed henchmen — the debt between them must be settled in person."),
        EventNode(id="EVT_TRAIN_GUNFIGHT", fabula_time=3100, syuzhet_index=15,
                  event_type="outcome", actor_ids=["ENT_CHEYENNES_GANG", "ENT_FRANKS_GANG"],
                  target_ids=["ENT_MORTON"],
                  description="Cheyenne's gang engages Frank's remaining men on Morton's train; everyone dies, including Morton, who crawls toward a muddy puddle in his last seconds."),
        EventNode(id="EVT_FRANK_RIDES_TO_SWEETWATER", fabula_time=3200, syuzhet_index=16,
                  event_type="choice", actor_ids=["ENT_FRANK"], target_ids=["ENT_HARMONICA"],
                  description="Surveying the aftermath, Frank rides to Sweetwater, where Harmonica is waiting."),
        EventNode(id="EVT_SHOWDOWN_FLASHBACK_REVEAL", fabula_time=3300, syuzhet_index=17,
                  event_type="outcome", actor_ids=["ENT_HARMONICA", "ENT_FRANK"], target_ids=["ENT_FRANK"],
                  description="In the showdown a flashback finally reveals Harmonica's identity; Harmonica beats Frank to the draw and pushes the harmonica into the dying man's mouth."),
        EventNode(id="EVT_FRANK_DIES", fabula_time=3350, syuzhet_index=19,
                  event_type="outcome", actor_ids=["ENT_HARMONICA"], target_ids=["ENT_FRANK"],
                  description="Frank dies in the dust outside Sweetwater, the harmonica between his teeth."),
        EventNode(id="EVT_BUILDING_STATION_AT_SWEETWATER", fabula_time=3400, syuzhet_index=20,
                  event_type="outcome", actor_ids=["ENT_JILL"], target_ids=["OBJ_RAILROAD_TRACKS"],
                  description="With Frank dead and Cheyenne riding off mortally wounded, Jill begins building the water station for the approaching railroad workers — the new Old West takes shape over the bones of the old."),
        EventNode(id="EVT_CHEYENNE_RIDES_OFF_DYING", fabula_time=3400, syuzhet_index=21,
                  event_type="outcome", actor_ids=["ENT_CHEYENNE"], target_ids=["ENT_CHEYENNE"],
                  description="Cheyenne, mortally wounded in the train gunfight, rides off into the desert to die alone rather than in front of Jill."),

        # ── UTTERANCES ──
        EventNode(id="EVT_UTT_MORTON_COMMISSIONS_FRANK", event_type="utterance",
                  description="Morton issues the commission that starts the whole chain.",
                  content="Frighten McBain off Sweetwater — but don't kill him. I want the land cleared, not a vendetta.",
                  speaker_id="ENT_MORTON", addressee_ids=["ENT_FRANK"], actor_ids=["ENT_MORTON"],
                  target_ids=["ENT_BRETT_MCBAIN", "EVT_MORTON_HIRES_FRANK"],
                  via_channel_id="CHN_MORTON_FRANK_COMMISSION", truth_value="performative",
                  fabula_time=500, syuzhet_index=22),
        EventNode(id="EVT_UTT_FRANK_NAMES_CHEYENNE", event_type="utterance",
                  description="Frank, in town after the massacre, casually drops Cheyenne's name to spread the framing.",
                  content="That looked like Cheyenne's work — those long dusters, that mark on the brand.",
                  speaker_id="ENT_FRANK", addressee_ids=["ENT_FRANKS_GANG"], actor_ids=["ENT_FRANK"],
                  target_ids=["ENT_CHEYENNE", "EVT_FRANK_FRAMES_CHEYENNE"],
                  via_channel_id=None, truth_value="false",
                  fabula_time=1300, syuzhet_index=23),
        EventNode(id="EVT_UTT_CHEYENNE_DENIES_AMBUSH", event_type="utterance",
                  description="Cheyenne, meeting Harmonica for the first time, denies sending the men at the station.",
                  content="Those weren't my men. I don't kill strangers for money — I kill them for reasons.",
                  speaker_id="ENT_CHEYENNE", addressee_ids=["ENT_HARMONICA"], actor_ids=["ENT_CHEYENNE"],
                  target_ids=["EVT_HARMONICA_ARRIVES"],
                  via_channel_id=None, truth_value="true",
                  fabula_time=1500, syuzhet_index=24),
        EventNode(id="EVT_UTT_FRANK_DEMANDS_NAME", event_type="utterance",
                  description="Across the showdown circle, Frank repeats the question he has been asking the whole film.",
                  content="Who are you?",
                  speaker_id="ENT_FRANK", addressee_ids=["ENT_HARMONICA"], actor_ids=["ENT_FRANK"],
                  target_ids=["ENT_HARMONICA", "EVT_SHOWDOWN_FLASHBACK_REVEAL"],
                  via_channel_id=None, truth_value="performative",
                  fabula_time=3300, syuzhet_index=25),
        EventNode(id="EVT_UTT_HARMONICA_PLAYS_THEME", event_type="utterance",
                  description="Harmonica answers the question not in words but by playing the theme from the day Frank hanged his brother.",
                  content="(plays the harmonica theme — the same notes the boy was forced to play under the desert arch)",
                  speaker_id="ENT_HARMONICA", addressee_ids=["ENT_FRANK"], actor_ids=["ENT_HARMONICA"],
                  target_ids=["ENT_FRANK", "EVT_FLASHBACK_HANGING", "EVT_SHOWDOWN_FLASHBACK_REVEAL"],
                  via_channel_id="CHN_HARMONICA_THEME_IDENTIFICATION", truth_value="performative",
                  fabula_time=3300, syuzhet_index=26),
        EventNode(id="EVT_UTT_AUCTION_CALL", event_type="utterance",
                  description="Harmonica, dragging Cheyenne by a rope, makes the impossible bid that breaks Frank's plan.",
                  content="Five thousand dollars. The bounty on this man's head — payable to the United States Marshal's account, here and now.",
                  speaker_id="ENT_HARMONICA", addressee_ids=["ENT_FRANK", "ENT_JILL"], actor_ids=["ENT_HARMONICA"],
                  target_ids=["ENT_CHEYENNE", "OBJ_BOUNTY_5000", "EVT_AUCTION"],
                  via_channel_id="CHN_BOUNTY_NETWORK", truth_value="performative",
                  fabula_time=2700, syuzhet_index=27),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction (the main spine) ──
        CausalEdge(source_id="EVT_FLASHBACK_HANGING", target_id="EVT_HARMONICA_ARRIVES",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=0, propagation_delay=1100),
        CausalEdge(source_id="EVT_MORTON_HIRES_FRANK", target_id="EVT_MCBAIN_FAMILY_MURDER",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=9.0, fabula_time=500, propagation_delay=500),
        CausalEdge(source_id="EVT_MCBAIN_FAMILY_MURDER", target_id="EVT_JILL_ARRIVES_AT_SWEETWATER",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000, propagation_delay=200),
        CausalEdge(source_id="EVT_MCBAIN_FAMILY_MURDER", target_id="EVT_FRANK_FRAMES_CHEYENNE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1000, propagation_delay=300),
        CausalEdge(source_id="EVT_HARMONICA_ARRIVES", target_id="EVT_HARMONICA_MEETS_CHEYENNE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1100, propagation_delay=400),
        CausalEdge(source_id="EVT_HARMONICA_MEETS_CHEYENNE", target_id="EVT_HARMONICA_SAVES_JILL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1500, propagation_delay=200),
        CausalEdge(source_id="EVT_HARMONICA_SAVES_JILL", target_id="EVT_HARMONICA_SPIES_MORTON_TRAIN",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1700, propagation_delay=300),
        CausalEdge(source_id="EVT_HARMONICA_SPIES_MORTON_TRAIN", target_id="EVT_HARMONICA_CAPTURED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2000, propagation_delay=100),
        CausalEdge(source_id="EVT_HARMONICA_CAPTURED", target_id="EVT_CHEYENNE_RESCUES_HARMONICA",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2100, propagation_delay=100),
        CausalEdge(source_id="EVT_FRANK_FORCES_JILL", target_id="EVT_AUCTION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2500, propagation_delay=200),
        CausalEdge(source_id="EVT_AUCTION", target_id="EVT_MORTON_BRIBES_FRANKS_MEN",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2700, propagation_delay=200),
        CausalEdge(source_id="EVT_MORTON_BRIBES_FRANKS_MEN", target_id="EVT_HARMONICA_SAVES_FRANK",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2900, propagation_delay=50),
        CausalEdge(source_id="EVT_MORTON_BRIBES_FRANKS_MEN", target_id="EVT_TRAIN_GUNFIGHT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2900, propagation_delay=200),
        CausalEdge(source_id="EVT_TRAIN_GUNFIGHT", target_id="EVT_FRANK_RIDES_TO_SWEETWATER",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3100, propagation_delay=100),
        CausalEdge(source_id="EVT_FRANK_RIDES_TO_SWEETWATER", target_id="EVT_SHOWDOWN_FLASHBACK_REVEAL",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=3200, propagation_delay=100),
        CausalEdge(source_id="EVT_SHOWDOWN_FLASHBACK_REVEAL", target_id="EVT_FRANK_DIES",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=3300, propagation_delay=50),
        CausalEdge(source_id="EVT_FRANK_DIES", target_id="EVT_BUILDING_STATION_AT_SWEETWATER",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3350, propagation_delay=50),
        CausalEdge(source_id="EVT_TRAIN_GUNFIGHT", target_id="EVT_CHEYENNE_RIDES_OFF_DYING",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3100, propagation_delay=300),

        # ── mutation ──
        CausalEdge(source_id="EVT_FLASHBACK_HANGING", target_id="ENT_HARMONICA",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=0,
                   trait_target="vengeance", trait_delta=0.95),
        CausalEdge(source_id="EVT_MCBAIN_FAMILY_MURDER", target_id="ENT_FRANK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000,
                   trait_target="ambition", trait_delta=0.1),
        CausalEdge(source_id="EVT_JILL_ARRIVES_AT_SWEETWATER", target_id="ENT_JILL",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=9.0, fabula_time=1200,
                   trait_target="grief", trait_delta=0.85),
        CausalEdge(source_id="EVT_FRANK_FORCES_JILL", target_id="ENT_JILL",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2500,
                   trait_target="moral_pliability", trait_delta=0.25),
        CausalEdge(source_id="EVT_AUCTION", target_id="ENT_FRANK",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=2700,
                   trait_target="curiosity_about_harmonica", trait_delta=0.7),
        CausalEdge(source_id="EVT_AUCTION", target_id="ENT_HARMONICA",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2700,
                   trait_target="compassion_for_jill", trait_delta=0.6),
        CausalEdge(source_id="EVT_SHOWDOWN_FLASHBACK_REVEAL", target_id="ENT_HARMONICA",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=3300,
                   trait_target="vengeance", trait_delta=-0.95),
        CausalEdge(source_id="EVT_BUILDING_STATION_AT_SWEETWATER", target_id="ENT_JILL",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=3400,
                   trait_target="resilience", trait_delta=0.1),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_MCBAIN_FAMILY_MURDER", target_id="ENT_MORTON",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1000,
                   trait_target="affinity", trait_delta=-0.5, rel_counterpart_id="ENT_FRANK"),
        CausalEdge(source_id="EVT_HARMONICA_SAVES_JILL", target_id="ENT_JILL",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1700,
                   trait_target="affinity", trait_delta=0.4, rel_counterpart_id="ENT_HARMONICA"),
        CausalEdge(source_id="EVT_CHEYENNE_RESCUES_HARMONICA", target_id="ENT_HARMONICA",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=2200,
                   trait_target="affinity", trait_delta=0.5, rel_counterpart_id="ENT_CHEYENNE"),
        CausalEdge(source_id="EVT_CHEYENNE_RESCUES_HARMONICA", target_id="ENT_CHEYENNE",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=2200,
                   trait_target="affinity", trait_delta=0.5, rel_counterpart_id="ENT_HARMONICA"),
        CausalEdge(source_id="EVT_FRANK_FORCES_JILL", target_id="ENT_JILL",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2500,
                   trait_target="fear", trait_delta=0.7, rel_counterpart_id="ENT_FRANK"),
        CausalEdge(source_id="EVT_AUCTION", target_id="ENT_FRANK",
                   causality_type="mutation_social", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2700,
                   trait_target="power_dynamic", trait_delta=-0.5, rel_counterpart_id="ENT_HARMONICA"),
        CausalEdge(source_id="EVT_MORTON_BRIBES_FRANKS_MEN", target_id="ENT_FRANK",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2900,
                   trait_target="affinity", trait_delta=-0.7, rel_counterpart_id="ENT_MORTON"),
        CausalEdge(source_id="EVT_HARMONICA_SAVES_FRANK", target_id="ENT_FRANK",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2950,
                   trait_target="power_dynamic", trait_delta=-0.5, rel_counterpart_id="ENT_HARMONICA"),
        CausalEdge(source_id="EVT_SHOWDOWN_FLASHBACK_REVEAL", target_id="ENT_FRANK",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=3300,
                   trait_target="fear", trait_delta=0.9, rel_counterpart_id="ENT_HARMONICA"),

        # ── affordance_gate ──
        CausalEdge(source_id="OBJ_HARMONICA", target_id="EVT_SHOWDOWN_FLASHBACK_REVEAL",
                   causality_type="affordance_gate", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=3300),
        CausalEdge(source_id="OBJ_FRAMING_EVIDENCE", target_id="EVT_FRANK_FRAMES_CHEYENNE",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1300),
        CausalEdge(source_id="OBJ_BOUNTY_5000", target_id="EVT_AUCTION",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2700),
        CausalEdge(source_id="OBJ_RAILROAD_TRACKS", target_id="EVT_MORTON_HIRES_FRANK",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=500),
        CausalEdge(source_id="OBJ_CRUTCHES", target_id="EVT_TRAIN_GUNFIGHT",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3100),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_DESERT_HANGING_ARCH", target_id="ENT_HARMONICA",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=0),
        CausalEdge(source_id="LOC_MORTONS_TRAIN", target_id="ENT_MORTON",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=500),
        CausalEdge(source_id="LOC_SWEETWATER", target_id="ENT_JILL",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1200),
        CausalEdge(source_id="LOC_AUCTION_SQUARE", target_id="ENT_FRANK",
                   causality_type="ambient_propagation", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2700),

        # ── WORLD_ → Event ──
        CausalEdge(source_id="WORLD_THE_RAILROAD", target_id="EVT_MORTON_HIRES_FRANK",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=9.0, fabula_time=500),
        CausalEdge(source_id="WORLD_THE_RAILROAD", target_id="EVT_BUILDING_STATION_AT_SWEETWATER",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3400),
        CausalEdge(source_id="WORLD_OLD_WEST_DYING", target_id="EVT_TRAIN_GUNFIGHT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3100),
        CausalEdge(source_id="WORLD_OLD_WEST_DYING", target_id="EVT_CHEYENNE_RIDES_OFF_DYING",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3400),
        CausalEdge(source_id="WORLD_REVENGE_CONTRACT", target_id="EVT_HARMONICA_ARRIVES",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1100),
        CausalEdge(source_id="WORLD_REVENGE_CONTRACT", target_id="EVT_HARMONICA_SAVES_FRANK",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=2950),
        CausalEdge(source_id="WORLD_REVENGE_CONTRACT", target_id="EVT_SHOWDOWN_FLASHBACK_REVEAL",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=3300),

        # ── orphan utterance wirings ──
        CausalEdge(source_id="EVT_MORTON_HIRES_FRANK", target_id="EVT_UTT_MORTON_COMMISSIONS_FRANK",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=500, propagation_delay=0),
        CausalEdge(source_id="EVT_FRANK_FRAMES_CHEYENNE", target_id="EVT_UTT_FRANK_NAMES_CHEYENNE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=4.0, fabula_time=1300, propagation_delay=0),
        CausalEdge(source_id="EVT_HARMONICA_MEETS_CHEYENNE", target_id="EVT_UTT_CHEYENNE_DENIES_AMBUSH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=4.0, fabula_time=1500, propagation_delay=0),
        CausalEdge(source_id="EVT_AUCTION", target_id="EVT_UTT_AUCTION_CALL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2700, propagation_delay=0),
        CausalEdge(source_id="EVT_SHOWDOWN_FLASHBACK_REVEAL", target_id="EVT_UTT_FRANK_DEMANDS_NAME",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=3300, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_FRANK_DEMANDS_NAME", target_id="EVT_UTT_HARMONICA_PLAYS_THEME",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3300, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_HARMONICA_PLAYS_THEME", target_id="EVT_FRANK_DIES",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3300, propagation_delay=50),

        # ─── auto-patched mutation_social edges (per-axis coverage) ───
        # Each observed affinity / power_dynamic dyad on the social topology
        # needs at least one ``mutation_social`` causal edge with the matching
        # ``trait_target`` so the corresponding gauge is not flat across the
        # fabula timeline. These edges anchor the static baselines to the
        # canonical events that establish them.
        CausalEdge(source_id="EVT_FLASHBACK_HANGING", target_id="ENT_HARMONICA", rel_counterpart_id="ENT_FRANK", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.95, mechanism="emotional", evidence_strength="strong", causal_force=10.0, fabula_time=3300, propagation_delay=0),
        CausalEdge(source_id="EVT_HARMONICA_ARRIVES", target_id="ENT_FRANK", rel_counterpart_id="ENT_HARMONICA", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.5, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=200, propagation_delay=0),
        CausalEdge(source_id="EVT_HARMONICA_SAVES_JILL", target_id="ENT_HARMONICA", rel_counterpart_id="ENT_JILL", causality_type="mutation_social", trait_target="affinity", trait_delta=0.55, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=1700, propagation_delay=0),
        CausalEdge(source_id="EVT_MCBAIN_FAMILY_MURDER", target_id="ENT_JILL", rel_counterpart_id="ENT_FRANK", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.85, mechanism="emotional", evidence_strength="strong", causal_force=9.0, fabula_time=300, propagation_delay=0),
        CausalEdge(source_id="EVT_HARMONICA_MEETS_CHEYENNE", target_id="ENT_CHEYENNE", rel_counterpart_id="ENT_JILL", causality_type="mutation_social", trait_target="affinity", trait_delta=0.65, mechanism="emotional", evidence_strength="moderate", causal_force=5.0, fabula_time=1400, propagation_delay=0),
        CausalEdge(source_id="EVT_MORTON_HIRES_FRANK", target_id="ENT_MORTON", rel_counterpart_id="ENT_FRANK", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.4, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=100, propagation_delay=0),
        CausalEdge(source_id="EVT_MORTON_HIRES_FRANK", target_id="ENT_FRANK", rel_counterpart_id="ENT_MORTON", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.4, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=100, propagation_delay=0),
        CausalEdge(source_id="EVT_FRANK_FORCES_JILL", target_id="ENT_JILL", rel_counterpart_id="ENT_FRANK", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.7, mechanism="physical", evidence_strength="strong", causal_force=8.0, fabula_time=1800, propagation_delay=0),
        CausalEdge(source_id="EVT_FRANK_FORCES_JILL", target_id="ENT_FRANK", rel_counterpart_id="ENT_JILL", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.7, mechanism="physical", evidence_strength="strong", causal_force=8.0, fabula_time=1800, propagation_delay=0),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_FLAGSTONE_STATION", target_id="LOC_SWEETWATER"),
        SpatialEdge(source_id="LOC_SWEETWATER", target_id="LOC_FLAGSTONE_STATION"),
        SpatialEdge(source_id="LOC_FLAGSTONE_STATION", target_id="LOC_MORTONS_TRAIN"),
        SpatialEdge(source_id="LOC_MORTONS_TRAIN", target_id="LOC_FLAGSTONE_STATION"),
        SpatialEdge(source_id="LOC_FLAGSTONE_STATION", target_id="LOC_AUCTION_SQUARE"),
        SpatialEdge(source_id="LOC_AUCTION_SQUARE", target_id="LOC_FLAGSTONE_STATION"),
        SpatialEdge(source_id="LOC_DESERT_HANGING_ARCH", target_id="LOC_FLAGSTONE_STATION"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    channels={
        "CHN_MORTON_FRANK_COMMISSION": Channel(
            id="CHN_MORTON_FRANK_COMMISSION",
            name="Morton–Frank Commission Pipeline",
            medium="private_meeting",
            participant_ids=["ENT_MORTON", "ENT_FRANK"],
            directionality="duplex",
            intelligibility={"ENT_MORTON": 0.95, "ENT_FRANK": 0.95},
            established_at_fabula=500, terminated_at_fabula=2900,
            evidence_strength="strong",
        ),
        "CHN_HARMONICA_THEME_IDENTIFICATION": Channel(
            id="CHN_HARMONICA_THEME_IDENTIFICATION",
            name="Harmonica's Wordless Theme",
            medium="musical_motif",
            participant_ids=["ENT_HARMONICA", "ENT_FRANK"],
            directionality="simplex",
            intelligibility={"ENT_HARMONICA": 0.95, "ENT_FRANK": 0.0},
            established_at_fabula=1100, terminated_at_fabula=None,
            evidence_strength="strong",
        ),
        "CHN_BOUNTY_NETWORK": Channel(
            id="CHN_BOUNTY_NETWORK",
            name="Marshal's Bounty Notice Network",
            medium="public_notice",
            participant_ids=["ENT_HARMONICA", "ENT_CHEYENNE", "ENT_FRANK", "ENT_JILL"],
            directionality="broadcast",
            intelligibility={"ENT_CHEYENNE": 0.95, "ENT_FRANK": 0.95, "ENT_JILL": 0.7},
            established_at_fabula=1500, terminated_at_fabula=None,
            evidence_strength="strong",
        ),
    },

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_THE_RAILROAD": GlobalTrait(
            id="WORLD_THE_RAILROAD",
            name="The Pacific Railroad as Inexorable Cosmology",
            description="The advancing railroad that values land by the inch, that turns water rights into murder warrants, and that absorbs every character's destiny into its westward schedule. The film's largest common-cause parent.",
            category="cosmology",
            magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
            affected_domains=["social", "economic"],
        ),
        "WORLD_OLD_WEST_DYING": GlobalTrait(
            id="WORLD_OLD_WEST_DYING",
            name="The Old West, Dying",
            description="The era of the lone gunfighter and the outlaw band, ending precisely as the rails arrive. Both Frank and Cheyenne are obsolete; only Harmonica's ritual debt remains to be paid before the land changes hands.",
            category="social_structure",
            magnitude=TraitVector(value=0.85, inertia=0.9, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
        ),
        "WORLD_REVENGE_CONTRACT": GlobalTrait(
            id="WORLD_REVENGE_CONTRACT",
            name="The Outstanding Revenge Contract",
            description="The unwritten law that binds Harmonica to Frank from the desert arch onward — a debt of identity that can only be discharged by the moment of recognition at the showdown.",
            category="moral_law",
            magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
            affected_domains=["psychological"],
        ),
    },

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────
    social_topology=[
        # Harmonica ↔ Frank — the central debt.
        RelationshipEdge(
            source_entity_id="ENT_HARMONICA", target_entity_id="ENT_FRANK",
            metrics={
                "affinity":      RelationshipMetric(value=-0.95, inertia=0.95, evidence_strength="strong", last_updated_fabula=3300),
                "fear":          RelationshipMetric(value=0.0, inertia=0.95, evidence_strength="strong", last_updated_fabula=0),
                "power_dynamic": RelationshipMetric(value=0.0, inertia=0.7, evidence_strength="moderate", last_updated_fabula=0),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_FRANK", target_entity_id="ENT_HARMONICA",
            metrics={
                "affinity":      RelationshipMetric(value=-0.5, inertia=0.7, evidence_strength="strong", last_updated_fabula=3300),
                "fear":          RelationshipMetric(value=0.9, inertia=0.6, evidence_strength="strong", last_updated_fabula=3300),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.6, evidence_strength="strong", last_updated_fabula=3300),
            },
        ),
        # Frank ↔ Morton — paymaster turning to enmity.
        RelationshipEdge(
            source_entity_id="ENT_MORTON", target_entity_id="ENT_FRANK",
            metrics={
                "affinity":      RelationshipMetric(value=-0.5, inertia=0.6, evidence_strength="strong", last_updated_fabula=2900),
                "power_dynamic": RelationshipMetric(value=0.4, inertia=0.7, evidence_strength="strong", last_updated_fabula=500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_FRANK", target_entity_id="ENT_MORTON",
            metrics={
                "affinity":      RelationshipMetric(value=-0.7, inertia=0.6, evidence_strength="strong", last_updated_fabula=2900),
                "power_dynamic": RelationshipMetric(value=-0.4, inertia=0.7, evidence_strength="strong", last_updated_fabula=500),
            },
        ),
        # Harmonica ↔ Cheyenne — uneasy alliance.
        RelationshipEdge(
            source_entity_id="ENT_HARMONICA", target_entity_id="ENT_CHEYENNE",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.55, evidence_strength="strong", last_updated_fabula=2200),
                "power_dynamic": RelationshipMetric(value=0.0, inertia=0.5, evidence_strength="moderate", last_updated_fabula=2200),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_CHEYENNE", target_entity_id="ENT_HARMONICA",
            metrics={
                "affinity": RelationshipMetric(value=0.6, inertia=0.55, evidence_strength="strong", last_updated_fabula=2200),
            },
        ),
        # Harmonica → Jill — protective compassion.
        RelationshipEdge(
            source_entity_id="ENT_HARMONICA", target_entity_id="ENT_JILL",
            metrics={
                "affinity": RelationshipMetric(value=0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=2700),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JILL", target_entity_id="ENT_HARMONICA",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.55, evidence_strength="strong", last_updated_fabula=2700),
            },
        ),
        # Jill ↔ Frank — predator and prey.
        RelationshipEdge(
            source_entity_id="ENT_JILL", target_entity_id="ENT_FRANK",
            metrics={
                "affinity":      RelationshipMetric(value=-0.85, inertia=0.6, evidence_strength="strong", last_updated_fabula=2500),
                "fear":          RelationshipMetric(value=0.85, inertia=0.6, evidence_strength="strong", last_updated_fabula=2500),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=2500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_FRANK", target_entity_id="ENT_JILL",
            metrics={
                "power_dynamic": RelationshipMetric(value=0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=2500),
            },
        ),
        # Cheyenne → Jill — protective.
        RelationshipEdge(
            source_entity_id="ENT_CHEYENNE", target_entity_id="ENT_JILL",
            metrics={
                "affinity": RelationshipMetric(value=0.65, inertia=0.55, evidence_strength="strong", last_updated_fabula=2200),
            },
        ),
        # Jill → Cheyenne — initially wary of the wanted outlaw who turns up at Sweetwater, then warms to him as the only honest man around. Mild affection by the end, low residual fear.
        RelationshipEdge(
            source_entity_id="ENT_JILL", target_entity_id="ENT_CHEYENNE",
            metrics={
                "affinity": RelationshipMetric(value=0.5, inertia=0.55, evidence_strength="strong",   last_updated_fabula=2200),
                "fear":     RelationshipMetric(value=0.2, inertia=0.2,  evidence_strength="moderate", last_updated_fabula=2200),
            },
        ),
    ],

    # ── PROPOSITIONS (affect-unification substrate) ─────────────────────
    # Catalogue of first-class propositions characters and the audience
    # hold beliefs about. Each Belief.proposition_id and each
    # Concern.proposition_id resolves into one of these. Calibrated for
    # Leone's operatic suspense (the Pacific railroad as inevitability,
    # the harmonica's identity as blindsiding final reveal).
    propositions=[
        # The opening atrocity that sets the spine in motion.
        Proposition(proposition_id="PROP_FRANK_HANGED_BROTHER", kind="event_occurs",
                    referent_ids=["EVT_FLASHBACK_HANGING", "ENT_FRANK", "ENT_HARMONICAS_BROTHER"],
                    description="Frank ritually hanged Harmonica's older brother in the desert.",
                    audience_default_prior=0.15, stakes=0.95,
                    truth_at_fabula={0: True}),
        # The blindsiding identity twist — held back until the showdown flashback.
        Proposition(proposition_id="PROP_HARMONICA_IDENTITY", kind="identity_is",
                    referent_ids=["ENT_HARMONICA", "ENT_FRANK"],
                    description="Harmonica is the boy from the desert arch — Frank's victim, returned.",
                    audience_default_prior=0.2, stakes=0.95,
                    truth_at_fabula={3300: True}),
        # The film's organising outcome question — does the debt get paid?
        Proposition(proposition_id="PROP_HARMONICA_AVENGES_BROTHER", kind="outcome",
                    referent_ids=["ENT_HARMONICA", "ENT_FRANK", "EVT_FRANK_DIES"],
                    description="Harmonica kills Frank in the showdown and avenges his brother.",
                    audience_default_prior=0.6, stakes=1.0,
                    truth_at_fabula={3350: True}),
        # The McBain massacre — opening catastrophe.
        Proposition(proposition_id="PROP_MCBAIN_FAMILY_MURDERED", kind="event_occurs",
                    referent_ids=["EVT_MCBAIN_FAMILY_MURDER", "ENT_BRETT_MCBAIN"],
                    description="Frank and his gang massacre Brett McBain and his children at Sweetwater.",
                    audience_default_prior=0.3, stakes=0.9,
                    truth_at_fabula={1000: True}),
        Proposition(proposition_id="PROP_MCBAIN_FAMILY_SAFE", kind="outcome",
                    referent_ids=["ENT_BRETT_MCBAIN", "LOC_SWEETWATER"],
                    description="Brett McBain and his children survive to enjoy the watering-station fortune.",
                    audience_default_prior=0.45, stakes=0.85,
                    truth_at_fabula={1000: False}),
        # Frank's lie that misdirects the bounty for half the film.
        Proposition(proposition_id="PROP_FRANK_FRAMED_CHEYENNE", kind="event_occurs",
                    referent_ids=["EVT_FRANK_FRAMES_CHEYENNE", "ENT_CHEYENNE", "ENT_FRANK"],
                    description="Frank planted evidence at Sweetwater implicating Cheyenne in the McBain massacre.",
                    audience_default_prior=0.4, stakes=0.7,
                    truth_at_fabula={1300: True}),
        Proposition(proposition_id="PROP_CHEYENNE_GUILTY_OF_MASSACRE", kind="identity_is",
                    referent_ids=["ENT_CHEYENNE", "EVT_MCBAIN_FAMILY_MURDER"],
                    description="Cheyenne ordered the McBain massacre.",
                    audience_default_prior=0.5, stakes=0.5,
                    truth_at_fabula={1500: False}),
        # The Morton-Frank commission — the social contract whose violation drives the second act.
        Proposition(proposition_id="PROP_MORTON_HIRES_FRANK", kind="relation_holds",
                    referent_ids=["ENT_MORTON", "ENT_FRANK", "CHN_MORTON_FRANK_COMMISSION"],
                    description="Morton hired Frank to clear Sweetwater for the railroad.",
                    audience_default_prior=0.3, stakes=0.7,
                    truth_at_fabula={500: True}),
        Proposition(proposition_id="PROP_FRANK_EXCEEDS_BRIEF", kind="trait_holds",
                    referent_ids=["ENT_FRANK", "ENT_MORTON"],
                    description="Frank has exceeded Morton's instructions and now wants Sweetwater for himself.",
                    audience_default_prior=0.55, stakes=0.7,
                    truth_at_fabula={1000: True}),
        Proposition(proposition_id="PROP_MORTON_BETRAYS_FRANK", kind="event_occurs",
                    referent_ids=["EVT_MORTON_BRIBES_FRANKS_MEN", "ENT_MORTON", "ENT_FRANK"],
                    description="Morton bribes Frank's own men to assassinate him.",
                    audience_default_prior=0.4, stakes=0.7,
                    truth_at_fabula={2900: True}),
        # The railroad as cosmology — the question that hangs over the whole film.
        Proposition(proposition_id="PROP_RAILROAD_REACHES_SWEETWATER", kind="outcome",
                    referent_ids=["OBJ_RAILROAD_TRACKS", "LOC_SWEETWATER", "WORLD_THE_RAILROAD"],
                    description="The Pacific Railroad reaches Sweetwater and uses it as the watering station.",
                    audience_default_prior=0.7, stakes=0.9,
                    truth_at_fabula={3400: True}),
        Proposition(proposition_id="PROP_MORTON_REACHES_PACIFIC", kind="outcome",
                    referent_ids=["ENT_MORTON", "OBJ_RAILROAD_TRACKS"],
                    description="Morton lives to see his rails touch the Pacific Ocean.",
                    audience_default_prior=0.45, stakes=0.7,
                    truth_at_fabula={3100: False}),
        # The Sweetwater ownership question — Jill's plot engine.
        Proposition(proposition_id="PROP_JILL_KEEPS_SWEETWATER", kind="outcome",
                    referent_ids=["ENT_JILL", "LOC_SWEETWATER"],
                    description="Jill McBain retains ownership of Sweetwater and completes the watering station.",
                    audience_default_prior=0.4, stakes=0.85,
                    truth_at_fabula={3400: True}),
        Proposition(proposition_id="PROP_FRANK_BUYS_SWEETWATER", kind="outcome",
                    referent_ids=["ENT_FRANK", "LOC_SWEETWATER", "EVT_AUCTION"],
                    description="Frank successfully buys Sweetwater at auction.",
                    audience_default_prior=0.55, stakes=0.8,
                    truth_at_fabula={2700: False}),
        Proposition(proposition_id="PROP_JILL_AUCTIONED_OUT", kind="event_occurs",
                    referent_ids=["EVT_FRANK_FORCES_JILL", "ENT_JILL", "ENT_FRANK"],
                    description="Frank coerces Jill into auctioning Sweetwater.",
                    audience_default_prior=0.5, stakes=0.7,
                    truth_at_fabula={2500: True}),
        # Cheyenne's mortal arc.
        Proposition(proposition_id="PROP_CHEYENNE_SURVIVES", kind="outcome",
                    referent_ids=["ENT_CHEYENNE"],
                    description="Cheyenne survives the train gunfight to ride out at the end.",
                    audience_default_prior=0.5, stakes=0.65,
                    truth_at_fabula={3400: False}),
        # The Old West as dying era — Leone's WORLD_-trait-as-proposition.
        Proposition(proposition_id="PROP_OLD_WEST_DYING", kind="trait_holds",
                    referent_ids=["WORLD_OLD_WEST_DYING", "WORLD_THE_RAILROAD"],
                    description="The age of gunfighters and outlaws ends as the rails arrive.",
                    audience_default_prior=0.7, stakes=0.6,
                    truth_at_fabula={3400: True}),
    ],
)
