"""Dad's Army — high-fidelity WorldStateV1 test fixture.

Authored against the current ingestion prompts. Demonstrates all five
CausalEdge modalities (chain_reaction, mutation, mutation_social,
affordance_gate, ambient_propagation), per-axis ``RelationshipMetric``,
explicit ``evidence_strength`` everywhere, and named-latent WORLD_
traits wired as common-cause parents over the events they jointly drive.

The two latents — Home-Front Spirit (the Blitz-era civilian determination
that puts a bank manager in a tin helmet) and British Class Comedy
(the platoon as microcosm of inter-war English caste tension) — operate
as the structural twin engines beneath every comic humiliation and every
modest act of courage in Walmington-on-Sea.
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
        "LOC_CHURCH_HALL": Location(
            name="Church Hall",
            description="HQ of the Walmington-on-Sea Home Guard platoon, requisitioned from the long-suffering Vicar.",
            ambient_state={
                "bumbling": AmbientVector(value=0.6, volatility=0.4, evidence_strength="strong"),
                "patriotism": AmbientVector(value=0.7, volatility=0.3, evidence_strength="strong"),
                "draughtiness": AmbientVector(value=0.6, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_HIGH_STREET": Location(
            name="High Street",
            description="Walmington-on-Sea's main parade of shops, a stage for ARP-versus-Home-Guard skirmishes.",
            ambient_state={
                "normality": AmbientVector(value=0.6, volatility=0.3, evidence_strength="strong"),
                "wartime_austerity": AmbientVector(value=0.7, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_JONES_SHOP": Location(
            name="Jones's Butcher Shop",
            description="Corporal Jones's butcher shop, doubling as informal armoury, mess hall, and rationing back-channel.",
            ambient_state={
                "warmth": AmbientVector(value=0.5, volatility=0.2, evidence_strength="moderate"),
                "ration_hustle": AmbientVector(value=0.6, volatility=0.3, evidence_strength="moderate"),
            },
        ),
        "LOC_BEACH": Location(
            name="Walmington Beach",
            description="The pebbled south-coast beach where invasion drills double as a comedy of misidentified buoys and sea-mines.",
            ambient_state={
                "wind": AmbientVector(value=0.6, volatility=0.3, evidence_strength="strong"),
                "tension": AmbientVector(value=0.3, volatility=0.4, evidence_strength="moderate"),
                "salt_spray": AmbientVector(value=0.7, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_VICARAGE": Location(
            name="Vicarage",
            description="Residence of the Vicar, adjacent to the church hall and perpetually under sonic siege from drilling boots.",
            ambient_state={
                "calm": AmbientVector(value=0.7, volatility=0.2, evidence_strength="strong"),
                "ecclesiastical_resentment": AmbientVector(value=0.6, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_AIRFIELD": Location(
            name="Local Airfield",
            description="Small grass airfield near Walmington commandeered for joint exercises and dressing-downs from regular-Army brass.",
            ambient_state={
                "danger": AmbientVector(value=0.4, volatility=0.4, evidence_strength="moderate"),
                "formality": AmbientVector(value=0.5, volatility=0.3, evidence_strength="strong"),
                "regular_army_disdain": AmbientVector(value=0.7, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_TOWN_SQUARE": Location(
            name="Town Square",
            description="Central square for parades, public addresses, and the platoon's eventual triumphant march.",
            ambient_state={
                "community": AmbientVector(value=0.6, volatility=0.3, evidence_strength="strong"),
                "civic_ritual": AmbientVector(value=0.7, volatility=0.2, evidence_strength="moderate"),
            },
        ),
    },

    # ── OBJECTS ────────────────────────────────────────────────────────
    objects={
        "OBJ_RIFLES": NarrativeObject(
            id="OBJ_RIFLES", name="Ancient Rifles",
            location_id="LOC_CHURCH_HALL", owner_id=None,
            properties={"state": "antiquated", "calibre": "mixed", "function": "platoon_armament"},
            affordances=[Affordance(action="shoot", target_type="Entity")],
        ),
        "OBJ_JONES_VAN": NarrativeObject(
            id="OBJ_JONES_VAN", name="Jones's Butcher Van",
            location_id="LOC_JONES_SHOP", owner_id="ENT_JONES",
            properties={"state": "running", "purpose": "transport", "fuel": "gas_converted"},
            affordances=[Affordance(action="transport", target_type="Entity")],
        ),
        "OBJ_WHISTLE": NarrativeObject(
            id="OBJ_WHISTLE", name="Parade Whistle",
            location_id="LOC_CHURCH_HALL", owner_id="ENT_MAINWARING",
            properties={"state": "polished", "function": "command_signal"},
            affordances=[Affordance(action="command", target_type="Entity")],
        ),
        "OBJ_SWASTIKA_FLAG": NarrativeObject(
            id="OBJ_SWASTIKA_FLAG", name="Captured Swastika Flag",
            location_id=None, owner_id=None,
            properties={"state": "captured", "function": "trophy_of_war"},
            affordances=[Affordance(action="display_as_trophy", target_type="Location")],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    entities={
        "ENT_MAINWARING": Entity(
            id="ENT_MAINWARING", name="Captain Mainwaring",
            location_id="LOC_CHURCH_HALL", status="healthy",
            traits={
                "pomposity":   TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "bravery":     TraitVector(value=0.6,  inertia=0.4, evidence_strength="moderate"),
                "insecurity":  TraitVector(value=0.6,  inertia=0.4, evidence_strength="strong"),
                "patriotism":  TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "class_anxiety": TraitVector(value=0.8, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_WILSON",
                       perceived_state="Wilson's languid charm undermines my proper authority",
                       confidence=0.7, inertia=0.5, established_at_fabula=100, evidence_strength="moderate"),
                Belief(target_id="ENT_HODGES",
                       perceived_state="Hodges is a petty rival who must not be allowed to win",
                       confidence=0.85, inertia=0.6, established_at_fabula=500, evidence_strength="strong"),
                Belief(target_id="ENT_FULLARD",
                       perceived_state="the Major-General persists in mistaking me for a mere bank clerk",
                       confidence=0.9, inertia=0.55, established_at_fabula=200, evidence_strength="strong"),
            ],
            constants=["bank_manager", "platoon_commander", "lower_middle_class"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=300, triggered_by="EVT_DRILL_FIASCO",
                    traits={"insecurity": TraitVector(value=0.7, inertia=0.45, evidence_strength="strong")}),
                EntityStateSnapshot(fabula_time=700, triggered_by="EVT_WILSON_OUTRANKS_REVELATION",
                    traits={
                        "insecurity":    TraitVector(value=0.8, inertia=0.55, evidence_strength="strong"),
                        "class_anxiety": TraitVector(value=0.95, inertia=0.8, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_WILSON",
                               perceived_state="he is socially my superior and I shall never live it down",
                               confidence=0.9, inertia=0.7, established_at_fabula=700, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=1100, triggered_by="EVT_MAINWARING_STANDS_FIRM",
                    traits={
                        "bravery":    TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                        "insecurity": TraitVector(value=0.55, inertia=0.5, evidence_strength="moderate"),
                    },
                    beliefs_invalidated=["ENT_FULLARD"]),
                EntityStateSnapshot(fabula_time=1500, triggered_by="EVT_PLATOON_MARCHES",
                    traits={"pomposity": TraitVector(value=0.95, inertia=0.8, evidence_strength="strong")}),
            ],
        ),
        "ENT_WILSON": Entity(
            id="ENT_WILSON", name="Sergeant Wilson",
            location_id="LOC_CHURCH_HALL", status="healthy",
            traits={
                "languid_charm": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "diffidence":    TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
                "wit":           TraitVector(value=0.75, inertia=0.6, evidence_strength="moderate"),
                "social_breeding": TraitVector(value=0.9, inertia=0.95, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="Mainwaring means well but is faintly absurd",
                       confidence=0.85, inertia=0.6, established_at_fabula=100, evidence_strength="strong"),
            ],
            constants=["public_school_educated", "former_officer_class"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=700, triggered_by="EVT_WILSON_OUTRANKS_REVELATION",
                    traits={"diffidence": TraitVector(value=0.55, inertia=0.55, evidence_strength="moderate")}),
            ],
        ),
        "ENT_JONES": Entity(
            id="ENT_JONES", name="Corporal Jones",
            location_id="LOC_JONES_SHOP", status="healthy",
            traits={
                "excitability": TraitVector(value=0.9,  inertia=0.55, evidence_strength="strong"),
                "loyalty":      TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "bravery":      TraitVector(value=0.75, inertia=0.5, evidence_strength="moderate"),
                "nostalgia":    TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="Captain Mainwaring is a great leader and deserves my unquestioning support",
                       confidence=0.95, inertia=0.85, established_at_fabula=100, evidence_strength="strong"),
            ],
            constants=["sudan_veteran", "butcher", "old_campaigner"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=800, triggered_by="EVT_JONES_BAYONET_CHARGE",
                    traits={
                        "excitability": TraitVector(value=0.95, inertia=0.55, evidence_strength="strong"),
                        "bravery":      TraitVector(value=0.8,  inertia=0.55, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_FRAZER": Entity(
            id="ENT_FRAZER", name="Private Frazer",
            location_id="LOC_CHURCH_HALL", status="healthy",
            traits={
                "doom":         TraitVector(value=0.9,  inertia=0.7, evidence_strength="strong"),
                "cunning":      TraitVector(value=0.7,  inertia=0.55, evidence_strength="moderate"),
                "stubbornness": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "thrift":       TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="Mainwaring is doomed to fail — we're all doomed",
                       confidence=0.7, inertia=0.55, established_at_fabula=100, evidence_strength="moderate"),
            ],
            constants=["undertaker", "scottish", "former_navy_rating"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_PARACHUTIST_CAPTURED",
                    traits={"doom": TraitVector(value=0.7, inertia=0.65, evidence_strength="moderate")}),
            ],
        ),
        "ENT_GODFREY": Entity(
            id="ENT_GODFREY", name="Private Godfrey",
            location_id="LOC_CHURCH_HALL", status="healthy",
            traits={
                "gentleness": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "frailty":    TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
                "courage":    TraitVector(value=0.5,  inertia=0.45, evidence_strength="moderate"),
                "mildness":   TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="Mainwaring is doing his best in trying circumstances",
                       confidence=0.7, inertia=0.5, established_at_fabula=100, evidence_strength="moderate"),
            ],
            constants=["conscientious_objector_medal", "first_world_war_medic", "elderly"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1200, triggered_by="EVT_GODFREY_HERO",
                    traits={"courage": TraitVector(value=0.75, inertia=0.5, evidence_strength="strong")},
                    beliefs_added=[
                        Belief(target_id="ENT_MAINWARING",
                               perceived_state="he understands now that quiet men can also be brave",
                               confidence=0.8, inertia=0.6, established_at_fabula=1200, evidence_strength="moderate"),
                    ]),
            ],
        ),
        "ENT_PIKE": Entity(
            id="ENT_PIKE", name="Private Pike",
            location_id="LOC_CHURCH_HALL", status="healthy",
            traits={
                "naivety":    TraitVector(value=0.8,  inertia=0.45, evidence_strength="strong"),
                "enthusiasm": TraitVector(value=0.75, inertia=0.5, evidence_strength="strong"),
                "youth":      TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_WILSON",
                       perceived_state="Uncle Arthur looks after me and Mum says I must do as he says",
                       confidence=0.95, inertia=0.6, established_at_fabula=100, evidence_strength="strong"),
            ],
            constants=["bank_clerk", "wears_scarf", "mothers_boy"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1300, triggered_by="EVT_PIKE_REFUSES_NAME",
                    traits={
                        "naivety":    TraitVector(value=0.6, inertia=0.45, evidence_strength="moderate"),
                        "enthusiasm": TraitVector(value=0.85, inertia=0.55, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_HODGES": Entity(
            id="ENT_HODGES", name="ARP Warden Hodges",
            location_id="LOC_HIGH_STREET", status="healthy",
            traits={
                "belligerence": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "pettiness":    TraitVector(value=0.75, inertia=0.55, evidence_strength="strong"),
                "civic_zeal":   TraitVector(value=0.7,  inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="Mainwaring's platoon is a joke that endangers proper civil defence",
                       confidence=0.85, inertia=0.55, established_at_fabula=500, evidence_strength="strong"),
            ],
            constants=["greengrocer", "arp_warden", "rival_authority"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1400, triggered_by="EVT_HODGES_GRUDGING_RESPECT",
                    traits={"belligerence": TraitVector(value=0.6, inertia=0.55, evidence_strength="moderate")},
                    beliefs_added=[
                        Belief(target_id="ENT_MAINWARING",
                               perceived_state="the bank manager actually held his nerve under fire",
                               confidence=0.65, inertia=0.4, established_at_fabula=1400, evidence_strength="moderate"),
                    ]),
            ],
        ),
        "ENT_WALKER": Entity(
            id="ENT_WALKER", name="Private Joe Walker",
            location_id="LOC_HIGH_STREET", status="healthy",
            traits={
                "cunning":     TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "spivvery":    TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "irreverence": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "loyalty":     TraitVector(value=0.5,  inertia=0.4, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="Mainwaring is a useful straight man for my schemes",
                       confidence=0.7, inertia=0.45, established_at_fabula=100, evidence_strength="moderate"),
            ],
            constants=["black_marketeer", "exempt_from_call_up", "wide_boy"],
        ),
        "ENT_FULLARD": Entity(
            id="ENT_FULLARD", name="Major-General Fullard",
            location_id="LOC_AIRFIELD", status="healthy",
            traits={
                "arrogance":    TraitVector(value=0.9,  inertia=0.8, evidence_strength="strong"),
                "competence":   TraitVector(value=0.7,  inertia=0.65, evidence_strength="moderate"),
                "irritability": TraitVector(value=0.85, inertia=0.55, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="Mainwaring is an incompetent bank clerk unfit to command",
                       confidence=0.85, inertia=0.6, established_at_fabula=200, evidence_strength="strong"),
            ],
            constants=["regular_army", "general_staff", "upper_class"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1400, triggered_by="EVT_HODGES_GRUDGING_RESPECT",
                    traits={"arrogance": TraitVector(value=0.75, inertia=0.7, evidence_strength="moderate")},
                    beliefs_added=[
                        Belief(target_id="ENT_MAINWARING",
                               perceived_state="the platoon may have been underestimated after all",
                               confidence=0.6, inertia=0.4, established_at_fabula=1400, evidence_strength="moderate"),
                    ]),
            ],
        ),
        "ENT_VICAR": Entity(
            id="ENT_VICAR", name="The Vicar",
            location_id="LOC_VICARAGE", status="healthy",
            traits={
                "fussiness": TraitVector(value=0.75, inertia=0.65, evidence_strength="strong"),
                "piety":     TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "anxiety":   TraitVector(value=0.65, inertia=0.45, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="Mainwaring monopolises my church hall with his absurd parading",
                       confidence=0.85, inertia=0.55, established_at_fabula=400, evidence_strength="strong"),
            ],
            constants=["clergyman", "civilian"],
        ),
        "ENT_GERMAN_OFFICER": Entity(
            id="ENT_GERMAN_OFFICER", name="Luftwaffe Officer",
            location_id="LOC_AIRFIELD", status="healthy",
            traits={
                "menace":      TraitVector(value=0.7,  inertia=0.55, evidence_strength="strong"),
                "discipline":  TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "frustration": TraitVector(value=0.5,  inertia=0.4, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_MAINWARING",
                       perceived_state="the British defenders are stupid and easy to outwit",
                       confidence=0.7, inertia=0.5, established_at_fabula=1000, evidence_strength="moderate"),
            ],
            constants=["wehrmacht_officer", "downed_airman"],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_PLATOON_FORMED", fabula_time=100, syuzhet_index=1, event_type="outcome",
                  actor_ids=["ENT_MAINWARING"], target_ids=["ENT_WILSON", "ENT_JONES"],
                  description="Mainwaring forms the Walmington-on-Sea Home Guard platoon at the church hall after Eden's wireless appeal."),
        EventNode(id="EVT_WEAPONS_DISTRIBUTED", fabula_time=200, syuzhet_index=2, event_type="outcome",
                  actor_ids=["ENT_MAINWARING"], target_ids=[],
                  description="The platoon receives ancient rifles and begins improvised drill in the church hall."),
        EventNode(id="EVT_DRILL_FIASCO", fabula_time=300, syuzhet_index=3, event_type="outcome",
                  actor_ids=["ENT_JONES"], target_ids=["ENT_MAINWARING"],
                  description="Jones panics during drill, causing a comedic chain-reaction of accidents that ends with a one-man bathtub-tank in the river."),
        EventNode(id="EVT_VICAR_COMPLAINS", fabula_time=400, syuzhet_index=13, event_type="outcome",
                  actor_ids=["ENT_VICAR"], target_ids=["ENT_MAINWARING"],
                  description="The Vicar complains formally about the platoon's monopolisation of the church hall for drilling."),
        EventNode(id="EVT_HODGES_CONFRONTATION", fabula_time=500, syuzhet_index=4, event_type="outcome",
                  actor_ids=["ENT_HODGES"], target_ids=["ENT_MAINWARING"],
                  description="ARP Warden Hodges challenges Mainwaring's authority during a blackout inspection on the High Street."),
        EventNode(id="EVT_PIKE_SCARF_INCIDENT", fabula_time=600, syuzhet_index=15, event_type="outcome",
                  actor_ids=["ENT_PIKE"], target_ids=["ENT_MAINWARING"],
                  description="Pike insists on wearing his scarf during drill, testing Mainwaring's patience and prompting the immortal 'stupid boy'."),
        EventNode(id="EVT_WILSON_OUTRANKS_REVELATION", fabula_time=700, syuzhet_index=5, event_type="revelation",
                  actor_ids=["ENT_WILSON"], target_ids=["ENT_MAINWARING"],
                  description="Wilson casually reveals his higher social standing, humiliating Mainwaring on the most sensitive of all axes."),
        EventNode(id="EVT_JONES_BAYONET_CHARGE", fabula_time=800, syuzhet_index=6, event_type="choice",
                  actor_ids=["ENT_JONES"], target_ids=[],
                  description="Jones leads a bayonet charge during a war-games exercise, shouting 'Don't panic!' as he goes."),
        EventNode(id="EVT_BEACH_PATROL", fabula_time=900, syuzhet_index=14, event_type="outcome",
                  actor_ids=["ENT_MAINWARING", "ENT_JONES"], target_ids=[],
                  description="Mainwaring and Jones patrol the beach after the exercises, mistaking a buoy for a sea-mine."),
        EventNode(id="EVT_PARACHUTIST_CAPTURED", fabula_time=1000, syuzhet_index=7, event_type="outcome",
                  actor_ids=["ENT_FRAZER", "ENT_GODFREY", "ENT_PIKE"], target_ids=["ENT_GERMAN_OFFICER"],
                  description="The platoon improbably captures a downed Luftwaffe crew on the beach after an aircraft is shot down."),
        EventNode(id="EVT_MAINWARING_STANDS_FIRM", fabula_time=1100, syuzhet_index=8, event_type="choice",
                  actor_ids=["ENT_MAINWARING"], target_ids=["ENT_FULLARD"],
                  description="Mainwaring stands up to Major-General Fullard, who is threatening to disband the platoon for poor showing on exercise."),
        EventNode(id="EVT_GODFREY_HERO", fabula_time=1200, syuzhet_index=9, event_type="revelation",
                  actor_ids=["ENT_GODFREY"], target_ids=["ENT_MAINWARING"],
                  description="Godfrey quietly reveals his Great-War medal for bravery as a stretcher-bearer, earning the platoon's astonished respect."),
        EventNode(id="EVT_PIKE_REFUSES_NAME", fabula_time=1300, syuzhet_index=10, event_type="choice",
                  actor_ids=["ENT_PIKE"], target_ids=["ENT_GERMAN_OFFICER"],
                  description="Pike refuses to give his name to the captured German officer; Mainwaring barks the immortal 'Don't tell him, Pike!'"),
        EventNode(id="EVT_HODGES_GRUDGING_RESPECT", fabula_time=1400, syuzhet_index=11, event_type="outcome",
                  actor_ids=["ENT_HODGES"], target_ids=["ENT_MAINWARING"],
                  description="Hodges grudgingly admits the platoon performed well during the real air-raid scare and the parachutist incident."),
        EventNode(id="EVT_PLATOON_MARCHES", fabula_time=1500, syuzhet_index=12, event_type="outcome",
                  actor_ids=["ENT_MAINWARING", "ENT_WILSON", "ENT_JONES", "ENT_FRAZER", "ENT_GODFREY", "ENT_PIKE"],
                  target_ids=[],
                  description="The platoon marches through Walmington town square to public cheers, finally respected by the community they defend."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction ──
        CausalEdge(source_id="EVT_PLATOON_FORMED", target_id="EVT_WEAPONS_DISTRIBUTED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=100, propagation_delay=100),
        CausalEdge(source_id="EVT_WEAPONS_DISTRIBUTED", target_id="EVT_DRILL_FIASCO",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=200, propagation_delay=100),
        CausalEdge(source_id="EVT_DRILL_FIASCO", target_id="EVT_VICAR_COMPLAINS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=300, propagation_delay=100),
        CausalEdge(source_id="EVT_DRILL_FIASCO", target_id="EVT_HODGES_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=300, propagation_delay=200),
        CausalEdge(source_id="EVT_HODGES_CONFRONTATION", target_id="EVT_WILSON_OUTRANKS_REVELATION",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=500, propagation_delay=200),
        # Pike scarf incident — Mainwaring's authority is needled, priming the Wilson rank revelation that follows.
        CausalEdge(source_id="EVT_HODGES_CONFRONTATION", target_id="EVT_PIKE_SCARF_INCIDENT",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=500, propagation_delay=100),
        CausalEdge(source_id="EVT_PIKE_SCARF_INCIDENT", target_id="EVT_WILSON_OUTRANKS_REVELATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=600, propagation_delay=100),
        CausalEdge(source_id="EVT_PIKE_SCARF_INCIDENT", target_id="EVT_PIKE_REFUSES_NAME",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=600, propagation_delay=700),
        CausalEdge(source_id="EVT_WILSON_OUTRANKS_REVELATION", target_id="EVT_JONES_BAYONET_CHARGE",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=700, propagation_delay=100),
        CausalEdge(source_id="EVT_JONES_BAYONET_CHARGE", target_id="EVT_BEACH_PATROL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=800, propagation_delay=100),
        CausalEdge(source_id="EVT_BEACH_PATROL", target_id="EVT_PARACHUTIST_CAPTURED",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=900, propagation_delay=100),
        CausalEdge(source_id="EVT_PARACHUTIST_CAPTURED", target_id="EVT_MAINWARING_STANDS_FIRM",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1000, propagation_delay=100),
        CausalEdge(source_id="EVT_MAINWARING_STANDS_FIRM", target_id="EVT_GODFREY_HERO",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1100, propagation_delay=100),
        CausalEdge(source_id="EVT_GODFREY_HERO", target_id="EVT_PIKE_REFUSES_NAME",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1200, propagation_delay=100),
        CausalEdge(source_id="EVT_PIKE_REFUSES_NAME", target_id="EVT_HODGES_GRUDGING_RESPECT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1300, propagation_delay=100),
        CausalEdge(source_id="EVT_HODGES_GRUDGING_RESPECT", target_id="EVT_PLATOON_MARCHES",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1400, propagation_delay=100),

        # ── mutation ──
        CausalEdge(source_id="EVT_DRILL_FIASCO", target_id="ENT_MAINWARING",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=300, trait_target="insecurity", trait_delta=0.1),
        CausalEdge(source_id="EVT_WILSON_OUTRANKS_REVELATION", target_id="ENT_MAINWARING",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=700, trait_target="insecurity", trait_delta=0.2),
        CausalEdge(source_id="EVT_WILSON_OUTRANKS_REVELATION", target_id="ENT_MAINWARING",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=700, trait_target="class_anxiety", trait_delta=0.15),
        CausalEdge(source_id="EVT_MAINWARING_STANDS_FIRM", target_id="ENT_MAINWARING",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1100, trait_target="bravery", trait_delta=0.25),
        CausalEdge(source_id="EVT_MAINWARING_STANDS_FIRM", target_id="ENT_MAINWARING",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1100, trait_target="insecurity", trait_delta=-0.05),
        CausalEdge(source_id="EVT_GODFREY_HERO", target_id="ENT_GODFREY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1200, trait_target="courage", trait_delta=0.25),
        CausalEdge(source_id="EVT_PIKE_REFUSES_NAME", target_id="ENT_PIKE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1300, trait_target="naivety", trait_delta=-0.2),
        CausalEdge(source_id="EVT_PIKE_REFUSES_NAME", target_id="ENT_PIKE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1300, trait_target="enthusiasm", trait_delta=0.1),
        CausalEdge(source_id="EVT_JONES_BAYONET_CHARGE", target_id="ENT_JONES",
                   causality_type="mutation", mechanism="emotional", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=800, trait_target="excitability", trait_delta=0.05),
        CausalEdge(source_id="EVT_HODGES_CONFRONTATION", target_id="ENT_HODGES",
                   causality_type="mutation", mechanism="emotional", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=500, trait_target="belligerence", trait_delta=0.05),
        CausalEdge(source_id="EVT_PLATOON_MARCHES", target_id="ENT_MAINWARING",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1500, trait_target="pomposity", trait_delta=0.1),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_HODGES_CONFRONTATION", target_id="ENT_MAINWARING",
                   causality_type="mutation_social", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=500,
                   trait_target="affinity", trait_delta=-0.3, rel_counterpart_id="ENT_HODGES"),
        CausalEdge(source_id="EVT_WILSON_OUTRANKS_REVELATION", target_id="ENT_MAINWARING",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=700,
                   trait_target="power_dynamic", trait_delta=-0.3, rel_counterpart_id="ENT_WILSON"),
        CausalEdge(source_id="EVT_HODGES_GRUDGING_RESPECT", target_id="ENT_HODGES",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1400,
                   trait_target="affinity", trait_delta=0.3, rel_counterpart_id="ENT_MAINWARING"),
        CausalEdge(source_id="EVT_MAINWARING_STANDS_FIRM", target_id="ENT_MAINWARING",
                   causality_type="mutation_social", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1100,
                   trait_target="power_dynamic", trait_delta=0.3, rel_counterpart_id="ENT_FULLARD"),
        CausalEdge(source_id="EVT_PARACHUTIST_CAPTURED", target_id="ENT_GERMAN_OFFICER",
                   causality_type="mutation_social", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1000,
                   trait_target="fear", trait_delta=0.4, rel_counterpart_id="ENT_MAINWARING"),
        CausalEdge(source_id="EVT_GODFREY_HERO", target_id="ENT_MAINWARING",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1200,
                   trait_target="affinity", trait_delta=0.25, rel_counterpart_id="ENT_GODFREY"),
        # Pike scarf incident — Mainwaring's affinity toward Pike dips ('stupid boy'); Pike's enthusiasm wilts briefly.
        CausalEdge(source_id="EVT_PIKE_SCARF_INCIDENT", target_id="ENT_MAINWARING",
                   causality_type="mutation_social", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=600,
                   trait_target="affinity", trait_delta=-0.15, rel_counterpart_id="ENT_PIKE"),
        CausalEdge(source_id="EVT_PIKE_SCARF_INCIDENT", target_id="ENT_PIKE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="moderate",
                   causal_force=3.0, fabula_time=600, trait_target="enthusiasm", trait_delta=-0.05),

        # ── affordance_gate ──
        CausalEdge(source_id="OBJ_RIFLES", target_id="EVT_WEAPONS_DISTRIBUTED",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=200),
        CausalEdge(source_id="OBJ_JONES_VAN", target_id="EVT_BEACH_PATROL",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=900),
        CausalEdge(source_id="ENT_MAINWARING", target_id="EVT_PLATOON_FORMED",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=100),
        CausalEdge(source_id="OBJ_WHISTLE", target_id="EVT_DRILL_FIASCO",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=300),
        CausalEdge(source_id="OBJ_RIFLES", target_id="EVT_PIKE_REFUSES_NAME",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1300),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_CHURCH_HALL", target_id="ENT_MAINWARING",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="weak",
                   causal_force=2.0, fabula_time=100),
        CausalEdge(source_id="LOC_BEACH", target_id="ENT_JONES",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="weak",
                   causal_force=2.0, fabula_time=900),
        CausalEdge(source_id="LOC_AIRFIELD", target_id="ENT_MAINWARING",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=3.0, fabula_time=1100),
        CausalEdge(source_id="LOC_TOWN_SQUARE", target_id="ENT_MAINWARING",
                   causality_type="ambient_propagation", mechanism="emotional", evidence_strength="moderate",
                   causal_force=3.0, fabula_time=1500),

        # ── WORLD_ → Event (named-latent common-cause wiring) ──
        CausalEdge(source_id="WORLD_HOME_FRONT", target_id="EVT_PLATOON_FORMED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=100),
        CausalEdge(source_id="WORLD_HOME_FRONT", target_id="EVT_PARACHUTIST_CAPTURED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_HOME_FRONT", target_id="EVT_MAINWARING_STANDS_FIRM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1100),
        CausalEdge(source_id="WORLD_HOME_FRONT", target_id="EVT_PIKE_REFUSES_NAME",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1300),
        CausalEdge(source_id="WORLD_HOME_FRONT", target_id="EVT_PLATOON_MARCHES",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1500),
        CausalEdge(source_id="WORLD_CLASS_COMEDY", target_id="EVT_HODGES_CONFRONTATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=500),
        CausalEdge(source_id="WORLD_CLASS_COMEDY", target_id="EVT_WILSON_OUTRANKS_REVELATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=700),
        CausalEdge(source_id="WORLD_CLASS_COMEDY", target_id="EVT_MAINWARING_STANDS_FIRM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=1100),
        CausalEdge(source_id="WORLD_CLASS_COMEDY", target_id="EVT_GODFREY_HERO",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1200),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_CHURCH_HALL", target_id="LOC_VICARAGE"),
        SpatialEdge(source_id="LOC_VICARAGE", target_id="LOC_CHURCH_HALL"),
        SpatialEdge(source_id="LOC_CHURCH_HALL", target_id="LOC_HIGH_STREET"),
        SpatialEdge(source_id="LOC_HIGH_STREET", target_id="LOC_CHURCH_HALL"),
        SpatialEdge(source_id="LOC_HIGH_STREET", target_id="LOC_JONES_SHOP"),
        SpatialEdge(source_id="LOC_JONES_SHOP", target_id="LOC_HIGH_STREET"),
        SpatialEdge(source_id="LOC_HIGH_STREET", target_id="LOC_TOWN_SQUARE"),
        SpatialEdge(source_id="LOC_TOWN_SQUARE", target_id="LOC_HIGH_STREET"),
        SpatialEdge(source_id="LOC_HIGH_STREET", target_id="LOC_BEACH"),
        SpatialEdge(source_id="LOC_TOWN_SQUARE", target_id="LOC_AIRFIELD"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    information_topology=[
        InformationEdge(source_id="ENT_MAINWARING", target_ids=["ENT_WILSON", "ENT_JONES", "ENT_PIKE"],
                        medium="parade_order", is_encrypted=False,
                        established_at_fabula=100, discovered_at_syuzhet=1, evidence_strength="strong"),
        InformationEdge(source_id="ENT_JONES", target_ids=["ENT_MAINWARING", "ENT_FRAZER"],
                        medium="war_anecdote", is_encrypted=False,
                        established_at_fabula=100, discovered_at_syuzhet=1, evidence_strength="moderate"),
        InformationEdge(source_id="ENT_WALKER", target_ids=["ENT_JONES", "ENT_PIKE"],
                        medium="black_market_offer", is_encrypted=True,
                        established_at_fabula=200, discovered_at_syuzhet=2, evidence_strength="moderate"),
        InformationEdge(source_id="ENT_VICAR", target_ids=["ENT_MAINWARING"],
                        medium="formal_complaint", is_encrypted=False,
                        established_at_fabula=400, terminated_at_fabula=400,
                        discovered_at_syuzhet=13, evidence_strength="strong"),
        InformationEdge(source_id="ENT_HODGES", target_ids=["ENT_MAINWARING"],
                        medium="shouting", is_encrypted=False,
                        established_at_fabula=500, terminated_at_fabula=500,
                        discovered_at_syuzhet=4, evidence_strength="strong"),
        InformationEdge(source_id="ENT_WILSON", target_ids=["ENT_MAINWARING"],
                        medium="understated_remark", is_encrypted=False,
                        established_at_fabula=700, terminated_at_fabula=700,
                        discovered_at_syuzhet=5, evidence_strength="strong"),
        InformationEdge(source_id="ENT_GERMAN_OFFICER", target_ids=["ENT_MAINWARING", "ENT_VICAR"],
                        medium="hostage_demand", is_encrypted=False,
                        established_at_fabula=1000, terminated_at_fabula=1100,
                        discovered_at_syuzhet=7, evidence_strength="strong"),
        InformationEdge(source_id="ENT_FULLARD", target_ids=["ENT_MAINWARING"],
                        medium="dressing_down", is_encrypted=False,
                        established_at_fabula=1100, terminated_at_fabula=1100,
                        discovered_at_syuzhet=8, evidence_strength="strong"),
        InformationEdge(source_id="ENT_GODFREY", target_ids=["ENT_MAINWARING", "ENT_JONES"],
                        medium="quiet_revelation", is_encrypted=False,
                        established_at_fabula=1200, terminated_at_fabula=1200,
                        discovered_at_syuzhet=9, evidence_strength="strong"),
        InformationEdge(source_id="ENT_MAINWARING", target_ids=["ENT_FULLARD"],
                        medium="radio_report", is_encrypted=False,
                        established_at_fabula=1400, terminated_at_fabula=1400,
                        discovered_at_syuzhet=11, evidence_strength="strong"),
    ],

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_HOME_FRONT": GlobalTrait(
            id="WORLD_HOME_FRONT",
            name="Home Front Spirit",
            description="The Blitz-era civilian determination to defend Britain despite inadequate training, equipment, and the absurdity of elderly volunteers facing a potential Nazi invasion. Operates as common-cause parent over the platoon's formation, every individual act of nerve, and the closing parade through Walmington.",
            category="social_structure",
            magnitude=TraitVector(value=0.7, inertia=0.65, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=1000, triggered_by="EVT_PARACHUTIST_CAPTURED",
                    magnitude=TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                    description="The improbable capture proves the platoon's volunteer spirit can produce real results."),
                WorldTraitSnapshot(fabula_time=1500, triggered_by="EVT_PLATOON_MARCHES",
                    magnitude=TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                    description="The platoon's parade through town crystallises Britain's defiant Home Front identity."),
            ],
        ),
        "WORLD_CLASS_COMEDY": GlobalTrait(
            id="WORLD_CLASS_COMEDY",
            name="British Class Comedy",
            description="The platoon as microcosm of inter-war English class tensions: a pompous lower-middle-class bank manager commands a public-school-bred sergeant, a Sudan-veteran butcher, an irreverent spiv, and a retired Edwardian gentleman — all jostling for status while pretending to defend the realm.",
            category="social_structure",
            magnitude=TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=700, triggered_by="EVT_WILSON_OUTRANKS_REVELATION",
                    magnitude=TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                    description="Wilson's casual disclosure makes the class structure visible even inside the platoon's uniformed equality."),
            ],
        ),
    },

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────
    social_topology=[
        # Mainwaring ↔ Wilson — the central comic dyad: official rank above, social rank below.
        RelationshipEdge(
            source_entity_id="ENT_MAINWARING", target_entity_id="ENT_WILSON",
            metrics={
                "affinity":      RelationshipMetric(value=0.45, inertia=0.45, evidence_strength="strong", last_updated_fabula=700),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=700),
                "power_dynamic": RelationshipMetric(value=0.3, inertia=0.7, evidence_strength="strong", last_updated_fabula=700),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_WILSON", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="moderate", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.2, inertia=0.65, evidence_strength="strong", last_updated_fabula=700),
            },
        ),
        # Mainwaring ↔ Jones — pomposity meets unconditional loyalty.
        RelationshipEdge(
            source_entity_id="ENT_MAINWARING", target_entity_id="ENT_JONES",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="weak", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.65, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_JONES", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.55, evidence_strength="strong", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.55, inertia=0.7, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        # Mainwaring ↔ Hodges — the petty rivalry that is Mainwaring's most active social wound.
        RelationshipEdge(
            source_entity_id="ENT_MAINWARING", target_entity_id="ENT_HODGES",
            metrics={
                "affinity":      RelationshipMetric(value=-0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=500),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=500),
                "power_dynamic": RelationshipMetric(value=0.1, inertia=0.6, evidence_strength="moderate", last_updated_fabula=500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_HODGES", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=-0.45, inertia=0.5, evidence_strength="strong", last_updated_fabula=1400),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="weak", last_updated_fabula=500),
                "power_dynamic": RelationshipMetric(value=-0.1, inertia=0.6, evidence_strength="moderate", last_updated_fabula=1400),
            },
        ),
        # Frazer → Mainwaring — sceptical Caledonian.
        RelationshipEdge(
            source_entity_id="ENT_FRAZER", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.45, evidence_strength="moderate", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="weak", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.25, inertia=0.6, evidence_strength="moderate", last_updated_fabula=100),
            },
        ),
        # Godfrey → Mainwaring — gentle deference.
        RelationshipEdge(
            source_entity_id="ENT_GODFREY", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="weak", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.3, inertia=0.6, evidence_strength="moderate", last_updated_fabula=100),
            },
        ),
        # Pike ↔ Wilson — the famous secret-uncle bond.
        RelationshipEdge(
            source_entity_id="ENT_PIKE", target_entity_id="ENT_WILSON",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.55, evidence_strength="strong", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="weak", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.35, inertia=0.65, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_WILSON", target_entity_id="ENT_PIKE",
            metrics={
                "affinity":      RelationshipMetric(value=0.55, inertia=0.55, evidence_strength="strong", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="weak", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=0.4, inertia=0.65, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        # Mainwaring → Pike — the 'stupid boy' axis.
        RelationshipEdge(
            source_entity_id="ENT_MAINWARING", target_entity_id="ENT_PIKE",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.45, evidence_strength="moderate", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="weak", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.65, evidence_strength="strong", last_updated_fabula=100),
            },
        ),
        # Walker ↔ Mainwaring — useful rogue.
        RelationshipEdge(
            source_entity_id="ENT_WALKER", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=0.45, inertia=0.45, evidence_strength="moderate", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="weak", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=-0.25, inertia=0.6, evidence_strength="moderate", last_updated_fabula=100),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MAINWARING", target_entity_id="ENT_WALKER",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.45, evidence_strength="moderate", last_updated_fabula=100),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="weak", last_updated_fabula=100),
                "power_dynamic": RelationshipMetric(value=0.3, inertia=0.6, evidence_strength="moderate", last_updated_fabula=100),
            },
        ),
        # Mainwaring ↔ Fullard — the regular-Army humiliation axis.
        RelationshipEdge(
            source_entity_id="ENT_MAINWARING", target_entity_id="ENT_FULLARD",
            metrics={
                "affinity":      RelationshipMetric(value=-0.35, inertia=0.5, evidence_strength="strong", last_updated_fabula=200),
                "fear":          RelationshipMetric(value=0.25, inertia=0.2, evidence_strength="strong", last_updated_fabula=200),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=1100),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_FULLARD", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=-0.55, inertia=0.5, evidence_strength="strong", last_updated_fabula=200),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="weak", last_updated_fabula=200),
                "power_dynamic": RelationshipMetric(value=0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=1100),
            },
        ),
        # Vicar → Mainwaring — ecclesiastical resentment.
        RelationshipEdge(
            source_entity_id="ENT_VICAR", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=-0.35, inertia=0.5, evidence_strength="moderate", last_updated_fabula=400),
                "fear":          RelationshipMetric(value=0.15, inertia=0.2, evidence_strength="weak", last_updated_fabula=400),
                "power_dynamic": RelationshipMetric(value=-0.2, inertia=0.6, evidence_strength="moderate", last_updated_fabula=400),
            },
        ),
        # German Officer ↔ Mainwaring — captor and captive.
        RelationshipEdge(
            source_entity_id="ENT_GERMAN_OFFICER", target_entity_id="ENT_MAINWARING",
            metrics={
                "affinity":      RelationshipMetric(value=-0.65, inertia=0.5, evidence_strength="strong", last_updated_fabula=1000),
                "fear":          RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="moderate", last_updated_fabula=1100),
                "power_dynamic": RelationshipMetric(value=-0.4, inertia=0.65, evidence_strength="strong", last_updated_fabula=1100),
            },
        ),
    ],
)
