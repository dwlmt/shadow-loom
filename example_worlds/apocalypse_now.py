# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apocalypse Now (1979, Coppola) — high-fidelity WorldStateV1 fixture.

Authored against the current ingestion prompts. Demonstrates all five
CausalEdge modalities, per-axis ``RelationshipMetric``, explicit
``evidence_strength`` everywhere, and four named-latent WORLD_ traits
wired as common-cause parents over the events they jointly drive:
``WORLD_VIETNAM_WAR`` (the Cold-War proxy theatre that produces both
Willard and Kurtz), ``WORLD_HEART_OF_DARKNESS`` (the Conradian moral
abyss the Nung river leads to), ``WORLD_CHAIN_OF_COMMAND`` (the army's
sanctioned-murder protocol that gives Willard his orders), and
``WORLD_RIVER_AS_FATE`` (the upriver vector as predestination).
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
        format='scene',
        target_word_min=437,
        target_word_max=1640,
        prose_density='moderate',
        voice='scene-level prose with some dialogue and sensory detail; third-person POV; past tense',
        style_exemplar='Apocalypse Now opens in Saigon in 1968. Army captain and special intelligence agent Benjamin Willard is holed up in a hotel room, heavily intoxicated and desperate to get back into action. He has completed one tour of duty in Vietnam, only to go home a changed man, miserable amid the confines of civilization. After agreeing to a divorce, he has returned to Vietnam for a second tour and now waits restlessly for a mission.',
        source_word_count=1093,
    ),
    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_SAIGON_HOTEL": Location(
            name="Saigon Hotel Room",
            description="Willard's stifling between-tour hotel room: ceiling-fan blades become helicopter rotors; broken mirror, broken man.",
            ambient_state={
                "claustrophobia":  AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
                "intoxication":    AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
                "dissociation":    AmbientVector(value=0.85, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_NHA_TRANG_HQ": Location(
            name="Nha Trang Command Compound",
            description="Air-conditioned MACV-SOG bungalow where two officers and a CIA man hand Willard his sanctioned-murder orders over roast beef and shrimp.",
            ambient_state={
                "bureaucratic_calm": AmbientVector(value=0.85, volatility=0.1, evidence_strength="strong"),
                "secrecy":           AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_PBR_RIVER": Location(
            name="The PBR on the Nung River",
            description="A Navy patrol boat — fibreglass hull, twin .50-cal — and the brown moving water that carries it inexorably upriver into Cambodia.",
            ambient_state={
                "isolation":     AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
                "humidity":      AmbientVector(value=0.95, volatility=0.1, evidence_strength="strong"),
                "drug_haze":     AmbientVector(value=0.7, volatility=0.5, evidence_strength="strong"),
                "fatalism":      AmbientVector(value=0.8, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_KILGORE_BEACH": Location(
            name="Vin Drin Dop / Kilgore's Beach",
            description="A Vietcong-held coastal village levelled by Kilgore's Air Cavalry to clear a surf-break; napalm at dawn, Wagner on loudspeakers.",
            ambient_state={
                "spectacle_violence": AmbientVector(value=0.95, volatility=0.4, evidence_strength="strong"),
                "absurdity":          AmbientVector(value=0.95, volatility=0.3, evidence_strength="strong"),
                "smell_of_napalm":    AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_HAU_PHAT_USO": Location(
            name="Hau Phat Supply Depot & USO Stage",
            description="A floodlit amphitheatre erected by the river: Playboy Playmates, pyrotechnics, and a mob of sex-starved GIs that overruns the stage.",
            ambient_state={
                "carnival_frenzy": AmbientVector(value=0.95, volatility=0.5, evidence_strength="strong"),
                "commodification": AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_DO_LUNG_BRIDGE": Location(
            name="Do Lung Bridge",
            description="The last American outpost before the Cambodian border; a bridge nightly destroyed and nightly rebuilt under tracer fire — no commanding officer, only chaos.",
            ambient_state={
                "anarchy":         AmbientVector(value=0.95, volatility=0.5, evidence_strength="strong"),
                "no_command":      AmbientVector(value=0.95, volatility=0.2, evidence_strength="strong"),
                "psychedelic_war": AmbientVector(value=0.9, volatility=0.4, evidence_strength="strong"),
            },
        ),
        "LOC_KURTZ_COMPOUND": Location(
            name="Kurtz's Compound",
            description="A ruined Khmer temple complex on the upper Nung: severed heads on stakes, hanged bodies in the trees, Montagnard worshippers at every door.",
            ambient_state={
                "horror":           AmbientVector(value=0.95, volatility=0.2, evidence_strength="strong"),
                "ritual_dread":     AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
                "moral_inversion":  AmbientVector(value=0.95, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_TEMPLE_INTERIOR": Location(
            name="The Temple Interior",
            description="Kurtz's lightless inner sanctum: shaved skull half-lit, Frazer's Golden Bough, T.S. Eliot, the muttered theology of unrestrained war.",
            ambient_state={
                "darkness":        AmbientVector(value=0.95, volatility=0.05, evidence_strength="strong"),
                "philosophical_dread": AmbientVector(value=0.9, volatility=0.1, evidence_strength="strong"),
            },
        ),
    },

    # ── OBJECTS ────────────────────────────────────────────────────────
    objects={
        "OBJ_MISSION_DOSSIER": NarrativeObject(
            id="OBJ_MISSION_DOSSIER", name="Kurtz Mission Dossier",
            location_id="LOC_PBR_RIVER", owner_id="ENT_WILLARD",
            properties={"state": "classified", "function": "biographical_seduction"},
            affordances=[Affordance(action="reveal_kurtz_humanity", target_type="Entity")],
        ),
        "OBJ_NAPALM": NarrativeObject(
            id="OBJ_NAPALM", name="Napalm Payload",
            location_id="LOC_KILGORE_BEACH", owner_id="ENT_KILGORE",
            properties={"state": "armed", "function": "treeline_clearance"},
            affordances=[Affordance(action="incinerate_jungle", target_type="Location")],
        ),
        "OBJ_MACHETE": NarrativeObject(
            id="OBJ_MACHETE", name="Sacrificial Machete",
            location_id="LOC_KURTZ_COMPOUND", owner_id=None,
            properties={"state": "honed"},
            affordances=[Affordance(action="execute_kurtz", target_type="Entity")],
        ),
        "OBJ_CARIBOU": NarrativeObject(
            id="OBJ_CARIBOU", name="Sacrificial Caribou",
            location_id="LOC_KURTZ_COMPOUND", owner_id=None,
            properties={"state": "garlanded_for_slaughter", "function": "ritual_substitute"},
            affordances=[Affordance(action="be_ritually_slaughtered", target_type="Entity")],
        ),
    },

    # ── ENTITIES ───────────────────────────────────────────────────────
    entities={
        "ENT_WILLARD": Entity(
            id="ENT_WILLARD", name="Captain Benjamin L. Willard",
            location_id="LOC_SAIGON_HOTEL", status="healthy",
            traits={
                "dissociation":    TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "intoxication":    TraitVector(value=0.85, inertia=0.4, evidence_strength="strong"),
                "obedience":       TraitVector(value=0.7,  inertia=0.7, evidence_strength="strong"),
                "lethality":       TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "moral_anaesthesia": TraitVector(value=0.6, inertia=0.6, evidence_strength="moderate"),
                "fascination_with_kurtz": TraitVector(value=0.1, inertia=0.4, evidence_strength="weak"),
            },
            beliefs=[
                Belief(target_id="ENT_KURTZ",
                       perceived_state="a name on a file — a problem to be terminated with extreme prejudice",
                       confidence=0.8, inertia=0.5, established_at_fabula=500, evidence_strength="strong"),
            ],
            constants=["us_army_special_intelligence", "second_tour", "divorced"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=500, triggered_by="EVT_NHA_TRANG_BRIEFING",
                    location_id="LOC_NHA_TRANG_HQ",
                    traits={
                        "obedience": TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_BOARD_PBR",
                    location_id="LOC_PBR_RIVER"),
                EntityStateSnapshot(fabula_time=3500, triggered_by="EVT_SAMPAN_MASSACRE",
                    traits={
                        "moral_anaesthesia": TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
                        "lethality":         TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_KURTZ",
                               perceived_state="we cut them in half with a machine gun and give them a Band-Aid; the line is a lie",
                               confidence=0.85, inertia=0.7, established_at_fabula=3500, evidence_strength="strong"),
                    ]),
                # Parity snapshots for EVT_DO_LUNG_BRIDGE and EVT_WILLARD_CAGED social mutations.
                EntityStateSnapshot(fabula_time=4500, triggered_by="EVT_DO_LUNG_BRIDGE",
                    location_id="LOC_DO_LUNG_BRIDGE",
                    beliefs_added=[
                        Belief(target_id="ENT_COLBY",
                               perceived_state="the army's previous assassin is now operating with Kurtz",
                               confidence=0.95, inertia=0.7, established_at_fabula=4500,
                               acquired_via_event_id="EVT_DO_LUNG_BRIDGE",
                               evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=6500, triggered_by="EVT_WILLARD_CAGED",
                    location_id="LOC_KURTZ_COMPOUND", status="injured"),
                EntityStateSnapshot(fabula_time=7000, triggered_by="EVT_WILLARD_READS_DOSSIER",
                    traits={
                        "fascination_with_kurtz": TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_KURTZ"],
                    beliefs_added=[
                        Belief(target_id="ENT_KURTZ",
                               perceived_state="a brilliant officer the army made and then could not unmake — perhaps the only sane man on this river",
                               confidence=0.85, inertia=0.8, established_at_fabula=7000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=8100, triggered_by="EVT_KURTZ_KILLED",
                    location_id="LOC_TEMPLE_INTERIOR",
                    traits={
                        "obedience":         TraitVector(value=0.3, inertia=0.85, evidence_strength="strong"),
                        "moral_anaesthesia": TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=8500, triggered_by="EVT_WILLARD_DEPARTS",
                    location_id="LOC_PBR_RIVER",
                    traits={
                        "dissociation": TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_KURTZ": Entity(
            id="ENT_KURTZ", name="Colonel Walter E. Kurtz",
            location_id="LOC_KURTZ_COMPOUND", status="healthy",
            traits={
                "intellect":         TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                "charisma":          TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "moral_collapse":    TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
                "lethality":         TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                "death_wish":        TraitVector(value=0.6,  inertia=0.7, evidence_strength="moderate"),
                "messianic_self_image": TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_WILLARD",
                       perceived_state="an errand boy sent by grocery clerks to collect a bill — but possibly my chosen executioner",
                       confidence=0.85, inertia=0.7, established_at_fabula=6000, evidence_strength="strong"),
            ],
            constants=["green_beret", "west_point", "rogue_command"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_ARRIVE_COMPOUND",
                    traits={
                        "death_wish": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_WILLARD"],
                    beliefs_added=[
                        Belief(target_id="ENT_WILLARD",
                               perceived_state="this is the one the army has finally sent to end me",
                               confidence=0.9, inertia=0.85, established_at_fabula=6000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=8100, triggered_by="EVT_KURTZ_KILLED",
                    status="dead", location_id="LOC_TEMPLE_INTERIOR"),
            ],
        ),
        "ENT_KILGORE": Entity(
            id="ENT_KILGORE", name="Lt. Colonel Bill Kilgore",
            location_id="LOC_KILGORE_BEACH", status="healthy",
            traits={
                "bravado":            TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                "love_of_surfing":    TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                "imperturbability":   TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                "casual_brutality":   TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="LOC_KILGORE_BEACH",
                       perceived_state="a six-foot peak that breaks both ways — Charlie don't surf, but we do",
                       confidence=0.95, inertia=0.95, established_at_fabula=1500, evidence_strength="strong"),
            ],
            constants=["air_cavalry", "stetson_and_yellow_scarf"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1800, triggered_by="EVT_NAPALM_SURF_RAID",
                    location_id="LOC_KILGORE_BEACH"),
            ],
        ),
        "ENT_CHIEF": Entity(
            id="ENT_CHIEF", name="Chief Petty Officer Phillips",
            location_id="LOC_PBR_RIVER", status="healthy",
            traits={
                "discipline":      TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
                "paternal_duty":   TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "rules_of_engagement": TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
                "distrust_of_willard": TraitVector(value=0.4, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_WILLARD",
                       perceived_state="a passenger with mysterious orders — not yet to be trusted with my boat",
                       confidence=0.7, inertia=0.6, established_at_fabula=1000, evidence_strength="strong"),
            ],
            constants=["us_navy", "career_sailor"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3500, triggered_by="EVT_SAMPAN_MASSACRE",
                    traits={
                        "distrust_of_willard": TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_WILLARD"],
                    beliefs_added=[
                        Belief(target_id="ENT_WILLARD",
                               perceived_state="he will kill anyone — civilians, the wounded — to keep his mission moving",
                               confidence=0.95, inertia=0.85, established_at_fabula=3500, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=5500, triggered_by="EVT_CHIEF_SPEARED",
                    status="dead", location_id="LOC_PBR_RIVER"),
            ],
        ),
        "ENT_LANCE": Entity(
            id="ENT_LANCE", name="Lance B. Johnson",
            location_id="LOC_PBR_RIVER", status="healthy",
            traits={
                "innocence":       TraitVector(value=0.7,  inertia=0.5, evidence_strength="strong"),
                "drug_use":        TraitVector(value=0.4,  inertia=0.5, evidence_strength="moderate"),
                "withdrawal":      TraitVector(value=0.2,  inertia=0.4, evidence_strength="weak"),
                "surfer_identity": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[],
            constants=["pro_surfer_california", "youngest_of_crew"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1800, triggered_by="EVT_NAPALM_SURF_RAID",
                    traits={
                        "drug_use":   TraitVector(value=0.7, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=3500, triggered_by="EVT_SAMPAN_MASSACRE",
                    traits={
                        "innocence":  TraitVector(value=0.3, inertia=0.7, evidence_strength="strong"),
                        "withdrawal": TraitVector(value=0.7, inertia=0.7, evidence_strength="strong"),
                        "drug_use":   TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_ARRIVE_COMPOUND",
                    location_id="LOC_KURTZ_COMPOUND",
                    traits={
                        "innocence":  TraitVector(value=0.05, inertia=0.85, evidence_strength="strong"),
                        "withdrawal": TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_KURTZ",
                               perceived_state="a god among the painted people; I belong here now",
                               confidence=0.9, inertia=0.85, established_at_fabula=6000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=8500, triggered_by="EVT_WILLARD_DEPARTS",
                    location_id="LOC_PBR_RIVER"),
            ],
        ),
        "ENT_CHEF": Entity(
            id="ENT_CHEF", name="Jay 'Chef' Hicks",
            location_id="LOC_PBR_RIVER", status="healthy",
            traits={
                "anxiety":      TraitVector(value=0.7, inertia=0.6, evidence_strength="strong"),
                "homesickness": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "irritability": TraitVector(value=0.6, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_WILLARD",
                       perceived_state="never should have got off the boat — and his mission is what got us off",
                       confidence=0.85, inertia=0.7, established_at_fabula=3000, evidence_strength="strong"),
            ],
            constants=["new_orleans_saucier", "draftee"],
            state_timeline=[
                # Witness belief from the on-page sampan massacre.
                EntityStateSnapshot(fabula_time=3500, triggered_by="EVT_SAMPAN_MASSACRE",
                    beliefs_added=[
                        Belief(target_id="ENT_WILLARD",
                               perceived_state="he killed the wounded woman in cold blood to keep us moving",
                               confidence=0.95, inertia=0.8, established_at_fabula=3500,
                               acquired_via_event_id="EVT_SAMPAN_MASSACRE",
                               evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=6500, triggered_by="EVT_WILLARD_CAGED",
                    status="dead", location_id="LOC_KURTZ_COMPOUND"),
            ],
        ),
        "ENT_CLEAN": Entity(
            id="ENT_CLEAN", name="Tyrone 'Mr Clean' Miller",
            location_id="LOC_PBR_RIVER", status="healthy",
            traits={
                "youth":         TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "trigger_happy": TraitVector(value=0.7,  inertia=0.6, evidence_strength="strong"),
                "homesickness":  TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[],
            constants=["seventeen_years_old", "south_bronx"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3500, triggered_by="EVT_SAMPAN_MASSACRE",
                    traits={
                        "trigger_happy": TraitVector(value=0.95, inertia=0.8, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=4800, triggered_by="EVT_MR_CLEAN_DEATH",
                    status="dead", location_id="LOC_PBR_RIVER"),
            ],
        ),
        "ENT_PHOTOJOURNALIST": Entity(
            id="ENT_PHOTOJOURNALIST", name="The Photojournalist",
            location_id="LOC_KURTZ_COMPOUND", status="healthy",
            traits={
                "mania":          TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
                "kurtz_worship":  TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                "logorrhoea":     TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_KURTZ",
                       perceived_state="a poet-warrior in the classical sense — the man enlarges the mind",
                       confidence=0.95, inertia=0.9, established_at_fabula=6000, evidence_strength="strong"),
            ],
            constants=["american_freelance", "amphetamines"],
            state_timeline=[
                # Parity for EVT_ARRIVE_COMPOUND → ENT_PHOTOJOURNALIST affinity toward Willard.
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_ARRIVE_COMPOUND",
                    location_id="LOC_KURTZ_COMPOUND"),
                EntityStateSnapshot(fabula_time=6050, triggered_by="EVT_UTT_PHOTOJOURNALIST_PROPHESIES",
                    location_id="LOC_KURTZ_COMPOUND"),
            ],
        ),
        "ENT_COLBY": Entity(
            id="ENT_COLBY", name="Captain Richard Colby",
            location_id="LOC_KURTZ_COMPOUND", status="healthy",
            traits={
                "catatonia":      TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                "former_obedience": TraitVector(value=0.1, inertia=0.4, evidence_strength="moderate"),
                "kurtz_loyalty":  TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_KURTZ",
                       perceived_state="my new and only commanding officer; the old chain of command is dead",
                       confidence=0.95, inertia=0.95, established_at_fabula=4500, evidence_strength="strong"),
            ],
            constants=["previous_assassin", "went_native"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_ARRIVE_COMPOUND",
                    location_id="LOC_KURTZ_COMPOUND"),
            ],
        ),
        # The Nha Trang briefing party (two officers + a CIA man) modelled as a
        # single composite entity — they speak with one voice, hand Willard the
        # dossier, and pronounce the kill-order with extreme prejudice.
        "ENT_NHA_TRANG_BRASS": Entity(
            id="ENT_NHA_TRANG_BRASS", name="Nha Trang Briefing Party (MACV-SOG officers + CIA man)",
            location_id="LOC_NHA_TRANG_HQ", status="healthy",
            traits={
                "institutional_authority": TraitVector(value=0.9, inertia=0.9, evidence_strength="strong"),
                "deniability":             TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[],
            constants=["chain_of_command", "cleared_for_kurtz_dossier"],
        ),
    },

    # ── EVENTS ─────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_SAIGON_HOTEL_BREAKDOWN", fabula_time=100, syuzhet_index=1,
                  event_type="outcome", actor_ids=["ENT_WILLARD"], target_ids=[],
                  description="Willard, drunk and stripped to his shorts in a Saigon hotel room, smashes a mirror with his fist and weeps for a mission."),
        EventNode(id="EVT_NHA_TRANG_BRIEFING", fabula_time=500, syuzhet_index=2,
                  event_type="outcome", actor_ids=["ENT_NHA_TRANG_BRASS"], target_ids=["ENT_WILLARD", "ENT_KURTZ"],
                  description="At Nha Trang two officers and a CIA man brief Willard on Colonel Kurtz, Green-Beret-gone-native in Cambodia, and order him to terminate Kurtz's command — with extreme prejudice."),
        EventNode(id="EVT_BOARD_PBR", fabula_time=1000, syuzhet_index=3,
                  event_type="choice", actor_ids=["ENT_WILLARD", "ENT_CHIEF", "ENT_LANCE", "ENT_CHEF", "ENT_CLEAN"], target_ids=[],
                  description="Willard joins the four-man crew of the Navy patrol boat that will ferry him up the Nung River to Cambodia."),
        EventNode(id="EVT_JOIN_KILGORE", fabula_time=1500, syuzhet_index=4,
                  event_type="choice", actor_ids=["ENT_WILLARD", "ENT_KILGORE"], target_ids=[],
                  description="The PBR rendezvouses with the 1/9th Air Cavalry; Willard meets Kilgore, who agrees to clear a Vietcong-held river-mouth — chiefly because it has a six-foot break."),
        EventNode(id="EVT_NAPALM_SURF_RAID", fabula_time=1800, syuzhet_index=5,
                  event_type="outcome", actor_ids=["ENT_KILGORE"], target_ids=["ENT_LANCE"],
                  description="At dawn, helicopters blasting Wagner's 'Ride of the Valkyries' assault the village; F-5s drop napalm on the treeline so Lance can surf — Kilgore loves the smell of napalm in the morning."),
        EventNode(id="EVT_USO_CHAOS", fabula_time=2500, syuzhet_index=6,
                  event_type="outcome", actor_ids=["ENT_LANCE", "ENT_CHEF", "ENT_CLEAN"], target_ids=[],
                  description="At the Hau Phat supply depot the crew watches a USO show; the sex-starved troops storm the stage and the Playmates are evacuated by helicopter."),
        EventNode(id="EVT_SAMPAN_MASSACRE", fabula_time=3500, syuzhet_index=7,
                  event_type="outcome", actor_ids=["ENT_CHIEF", "ENT_CHEF", "ENT_CLEAN", "ENT_WILLARD"], target_ids=[],
                  description="Chief insists on stopping a peasant sampan; a sudden movement panics Mr Clean into firing — every civilian dies; Willard finishes the surviving woman with a single shot to keep the mission moving."),
        EventNode(id="EVT_DO_LUNG_BRIDGE", fabula_time=4500, syuzhet_index=8,
                  event_type="outcome", actor_ids=["ENT_WILLARD"], target_ids=["ENT_COLBY"],
                  description="At Do Lung Bridge — the last American outpost — there is no commanding officer; in a packet of mail Willard reads a letter revealing that Captain Colby, sent on the same mission, is now operating with Kurtz."),
        EventNode(id="EVT_MR_CLEAN_DEATH", fabula_time=4800, syuzhet_index=9,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_CLEAN"],
                  description="As Mr Clean listens to a tape from his mother on the family Sears stereo, a Vietcong ambush rakes the boat; he is shot dead mid-tape."),
        EventNode(id="EVT_CHIEF_SPEARED", fabula_time=5500, syuzhet_index=10,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_CHIEF", "ENT_WILLARD"],
                  description="Primitive natives onshore loose a storm of arrows at the PBR; Chief is impaled with a thrown spear and, dying, tries to pull Willard onto the point with him."),
        EventNode(id="EVT_ARRIVE_COMPOUND", fabula_time=6000, syuzhet_index=11,
                  event_type="outcome", actor_ids=["ENT_WILLARD", "ENT_LANCE", "ENT_CHEF"], target_ids=["ENT_KURTZ", "ENT_PHOTOJOURNALIST"],
                  description="The PBR reaches Kurtz's macabre compound — bodies, severed heads, painted Montagnards on every shore; the manic Photojournalist greets the survivors and proclaims Kurtz's genius."),
        EventNode(id="EVT_WILLARD_CAGED", fabula_time=6500, syuzhet_index=12,
                  event_type="outcome", actor_ids=["ENT_KURTZ"], target_ids=["ENT_WILLARD", "ENT_CHEF"],
                  description="Kurtz's people drag Willard through the mud and lock him in a tiger cage; in the night Kurtz drops Chef's severed head into Willard's lap."),
        EventNode(id="EVT_WILLARD_READS_DOSSIER", fabula_time=7000, syuzhet_index=13,
                  event_type="outcome", actor_ids=["ENT_WILLARD"], target_ids=["ENT_KURTZ"],
                  description="Freed and given the run of the compound, Willard re-reads the Kurtz dossier and listens to days of Kurtz's philosophising; his contempt curdles into recognition."),
        EventNode(id="EVT_BUFFALO_RITUAL", fabula_time=8000, syuzhet_index=14,
                  event_type="outcome", actor_ids=["ENT_WILLARD"], target_ids=["ENT_KURTZ"],
                  description="As the Montagnards ritually slaughter a garlanded caribou, Willard, mud-streaked, rises from the river and enters Kurtz's chamber with a machete."),
        EventNode(id="EVT_KURTZ_KILLED", fabula_time=8100, syuzhet_index=15,
                  event_type="outcome", actor_ids=["ENT_WILLARD"], target_ids=["ENT_KURTZ"],
                  description="Willard hacks Kurtz down with the machete, intercut with the buffalo's slaughter; Kurtz's last words are 'the horror, the horror.'"),
        EventNode(id="EVT_WILLARD_DEPARTS", fabula_time=8500, syuzhet_index=16,
                  event_type="choice", actor_ids=["ENT_WILLARD", "ENT_LANCE"], target_ids=[],
                  description="Willard drops the machete; the natives lower their weapons in obeisance; he leads Lance back to the PBR, switches off the radio that would call in the air strike, and pulls away as rain begins to fall."),

        # ── UTTERANCE EVENTS (one-shot speech-acts; via_channel_id only when riding a standing channel) ──
        EventNode(
            id="EVT_UTT_NHA_TRANG_KILL_ORDER", fabula_time=550, syuzhet_index=17,
            event_type="utterance", speaker_id="ENT_NHA_TRANG_BRASS",
            addressee_ids=["ENT_WILLARD"],
            actor_ids=["ENT_NHA_TRANG_BRASS"],
            target_ids=["ENT_KURTZ"],
            via_channel_id=None,
            truth_value="performative",
            description="At the Nha Trang lunch the two officers and the CIA man hand Willard the dossier and pronounce the sanctioned-murder order: terminate Kurtz's command, with extreme prejudice.",
            content="Willard is ordered to proceed up the Nung River into Cambodia, locate Colonel Walter E. Kurtz, and terminate his command — with extreme prejudice.",
        ),
        EventNode(
            id="EVT_UTT_DOSSIER_PROFILES_KURTZ", fabula_time=3000, syuzhet_index=18,
            event_type="utterance", speaker_id="OBJ_MISSION_DOSSIER",
            addressee_ids=["ENT_WILLARD"],
            actor_ids=["OBJ_MISSION_DOSSIER"],
            target_ids=["ENT_KURTZ"],
            via_channel_id="CHN_MISSION_DOSSIER",
            truth_value="true",
            description="On the PBR Willard reads and re-reads the classified MACV-SOG dossier on Kurtz — service record, citations, the murder of four suspected double agents, the Montagnard god-king cult.",
            content="The dossier documents Kurtz's West Point pedigree, his Special Forces excellence, his unsanctioned execution of four ARVN intelligence agents, and his retreat to a Cambodian outpost where the Montagnards worship him.",
        ),
        EventNode(
            id="EVT_UTT_KILGORE_AIR_ASSAULT_ORDER", fabula_time=1780, syuzhet_index=19,
            event_type="utterance", speaker_id="ENT_KILGORE",
            addressee_ids=["ENT_WILLARD", "ENT_LANCE", "ENT_CHEF", "ENT_CHIEF", "ENT_CLEAN"],
            actor_ids=["ENT_KILGORE"],
            target_ids=["LOC_KILGORE_BEACH"],
            via_channel_id="CHN_AIR_CAV_RADIO",
            truth_value="performative",
            description="From the lead Huey, Kilgore calls the air assault on the VC-held river-mouth village over the Air Cav net, ordering Wagner on the loudspeakers and napalm on the treeline so Lance can surf the break.",
            content="Kilgore orders Big Duke Six's gunships to engage the village, cue 'Ride of the Valkyries' on the loudspeakers, and call in a napalm strike on the treeline.",
        ),
        EventNode(
            id="EVT_UTT_KILGORE_NAPALM_SOLILOQUY", fabula_time=2100, syuzhet_index=20,
            event_type="utterance", speaker_id="ENT_KILGORE",
            addressee_ids=["ENT_WILLARD"],
            actor_ids=["ENT_KILGORE"],
            target_ids=["ENT_KILGORE"],
            via_channel_id=None,
            truth_value="performative",
            description="On the smoking beach after the napalm strike, Kilgore squats beside Willard and delivers his unprompted aria on the smell of the morning air.",
            content="Kilgore tells Willard, 'I love the smell of napalm in the morning — it smells like victory,' framing the chemical incineration as sacrament.",
        ),
        EventNode(
            id="EVT_UTT_PHOTOJOURNALIST_PROPHESIES", fabula_time=6050, syuzhet_index=21,
            event_type="utterance", speaker_id="ENT_PHOTOJOURNALIST",
            addressee_ids=["ENT_WILLARD", "ENT_LANCE", "ENT_CHEF"],
            actor_ids=["ENT_PHOTOJOURNALIST"],
            target_ids=["ENT_KURTZ"],
            via_channel_id=None,
            truth_value="performative",
            description="At the boat ramp the manic American photojournalist accosts the survivors with a babbling, disjointed sermon on Kurtz's genius, quoting Kipling and Eliot at random.",
            content="The photojournalist proclaims that Kurtz is a poet-warrior who 'enlarges the mind' and warns the crew that ordinary categories cannot judge him.",
        ),
        EventNode(
            id="EVT_UTT_KURTZ_HORROR_DOCTRINE", fabula_time=7200, syuzhet_index=22,
            event_type="utterance", speaker_id="ENT_KURTZ",
            addressee_ids=["ENT_WILLARD"],
            actor_ids=["ENT_KURTZ"],
            target_ids=["ENT_KURTZ", "WORLD_VIETNAM_WAR"],
            via_channel_id=None,
            truth_value="performative",
            description="Across days in the compound Kurtz reads aloud his doctrine of horror to Willard, instructing him in the metaphysics of unrestrained war.",
            content="Kurtz expounds the doctrine that horror and moral terror are friends, that one must make a friend of horror, and that the army's failure is squeamishness disguised as virtue.",
        ),
        EventNode(
            id="EVT_UTT_KURTZ_LAST_WORDS", fabula_time=8095, syuzhet_index=23,
            event_type="utterance", speaker_id="ENT_KURTZ",
            addressee_ids=["ENT_WILLARD"],
            actor_ids=["ENT_KURTZ"],
            target_ids=["ENT_KURTZ"],
            via_channel_id=None,
            truth_value="true",
            description="As the machete-blows fall and the buffalo is slaughtered, Kurtz dies whispering his recognition of what the war and he have become.",
            content="Kurtz's dying breath is the twice-spoken phrase, 'The horror... the horror.'",
        ),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction (event → event) ──
        CausalEdge(source_id="EVT_SAIGON_HOTEL_BREAKDOWN", target_id="EVT_NHA_TRANG_BRIEFING",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=100, propagation_delay=400),
        CausalEdge(source_id="EVT_NHA_TRANG_BRIEFING", target_id="EVT_BOARD_PBR",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=500, propagation_delay=500),
        CausalEdge(source_id="EVT_BOARD_PBR", target_id="EVT_JOIN_KILGORE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000, propagation_delay=500),
        CausalEdge(source_id="EVT_JOIN_KILGORE", target_id="EVT_NAPALM_SURF_RAID",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=9.0, fabula_time=1500, propagation_delay=300),
        CausalEdge(source_id="EVT_NAPALM_SURF_RAID", target_id="EVT_USO_CHAOS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1800, propagation_delay=700),
        CausalEdge(source_id="EVT_USO_CHAOS", target_id="EVT_SAMPAN_MASSACRE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=2500, propagation_delay=1000),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="EVT_DO_LUNG_BRIDGE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=3500, propagation_delay=1000),
        CausalEdge(source_id="EVT_DO_LUNG_BRIDGE", target_id="EVT_MR_CLEAN_DEATH",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=4500, propagation_delay=300),
        CausalEdge(source_id="EVT_MR_CLEAN_DEATH", target_id="EVT_CHIEF_SPEARED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=4800, propagation_delay=700),
        CausalEdge(source_id="EVT_CHIEF_SPEARED", target_id="EVT_ARRIVE_COMPOUND",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=5500, propagation_delay=500),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="EVT_WILLARD_CAGED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000, propagation_delay=500),
        CausalEdge(source_id="EVT_WILLARD_CAGED", target_id="EVT_WILLARD_READS_DOSSIER",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=6500, propagation_delay=500),
        CausalEdge(source_id="EVT_WILLARD_READS_DOSSIER", target_id="EVT_BUFFALO_RITUAL",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000, propagation_delay=1000),
        CausalEdge(source_id="EVT_BUFFALO_RITUAL", target_id="EVT_KURTZ_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=8000, propagation_delay=100),
        CausalEdge(source_id="EVT_KURTZ_KILLED", target_id="EVT_WILLARD_DEPARTS",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=8100, propagation_delay=400),
        CausalEdge(source_id="EVT_DO_LUNG_BRIDGE", target_id="EVT_WILLARD_READS_DOSSIER",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=6.0, fabula_time=4500, propagation_delay=2500),

        # ── mutation (event → ENT, trait_target+trait_delta) ──
        CausalEdge(source_id="EVT_NHA_TRANG_BRIEFING", target_id="ENT_WILLARD",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=500,
                   trait_target="obedience", trait_delta=0.15),
        CausalEdge(source_id="EVT_NAPALM_SURF_RAID", target_id="ENT_LANCE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1800,
                   trait_target="drug_use", trait_delta=0.3),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_WILLARD",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3500,
                   trait_target="moral_anaesthesia", trait_delta=0.3),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_LANCE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3500,
                   trait_target="innocence", trait_delta=-0.4),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_LANCE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3500,
                   trait_target="withdrawal", trait_delta=0.5),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_CLEAN",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3500,
                   trait_target="trigger_happy", trait_delta=0.25),
        CausalEdge(source_id="EVT_WILLARD_READS_DOSSIER", target_id="ENT_WILLARD",
                   causality_type="mutation", mechanism="epistemic", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000,
                   trait_target="fascination_with_kurtz", trait_delta=0.75),
        CausalEdge(source_id="EVT_KURTZ_KILLED", target_id="ENT_WILLARD",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=8100,
                   trait_target="obedience", trait_delta=-0.4),
        CausalEdge(source_id="EVT_WILLARD_DEPARTS", target_id="ENT_WILLARD",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=8500,
                   trait_target="dissociation", trait_delta=0.1),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="ENT_LANCE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000,
                   trait_target="withdrawal", trait_delta=0.25),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="ENT_KURTZ",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=6000,
                   trait_target="death_wish", trait_delta=0.25),

        # ── mutation_social (event → ENT, trait_target+trait_delta+rel_counterpart_id) ──
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_CHIEF",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=3500,
                   trait_target="affinity", trait_delta=-0.6, rel_counterpart_id="ENT_WILLARD"),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_CHIEF",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3500,
                   trait_target="fear", trait_delta=0.4, rel_counterpart_id="ENT_WILLARD"),
        CausalEdge(source_id="EVT_DO_LUNG_BRIDGE", target_id="ENT_WILLARD",
                   causality_type="mutation_social", mechanism="informational", evidence_strength="strong",
                   causal_force=7.0, fabula_time=4500,
                   trait_target="affinity", trait_delta=0.3, rel_counterpart_id="ENT_KURTZ"),
        CausalEdge(source_id="EVT_WILLARD_READS_DOSSIER", target_id="ENT_WILLARD",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000,
                   trait_target="affinity", trait_delta=0.5, rel_counterpart_id="ENT_KURTZ"),
        CausalEdge(source_id="EVT_WILLARD_CAGED", target_id="ENT_WILLARD",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6500,
                   trait_target="fear", trait_delta=0.5, rel_counterpart_id="ENT_KURTZ"),
        CausalEdge(source_id="EVT_KURTZ_KILLED", target_id="ENT_LANCE",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=8100,
                   trait_target="affinity", trait_delta=0.3, rel_counterpart_id="ENT_WILLARD"),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="ENT_PHOTOJOURNALIST",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=6000,
                   trait_target="affinity", trait_delta=0.4, rel_counterpart_id="ENT_WILLARD"),

        # ── affordance_gate (state node → event) ──
        CausalEdge(source_id="OBJ_MISSION_DOSSIER", target_id="EVT_WILLARD_READS_DOSSIER",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000),
        CausalEdge(source_id="OBJ_NAPALM", target_id="EVT_NAPALM_SURF_RAID",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1800),
        CausalEdge(source_id="OBJ_MACHETE", target_id="EVT_KURTZ_KILLED",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=8100),
        CausalEdge(source_id="OBJ_CARIBOU", target_id="EVT_BUFFALO_RITUAL",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=8000),

        # ── ambient_propagation (LOC → ENT) ──
        CausalEdge(source_id="LOC_SAIGON_HOTEL", target_id="ENT_WILLARD",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=100),
        CausalEdge(source_id="LOC_PBR_RIVER", target_id="ENT_LANCE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=3000),
        CausalEdge(source_id="LOC_PBR_RIVER", target_id="ENT_CHEF",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=3000),
        CausalEdge(source_id="LOC_DO_LUNG_BRIDGE", target_id="ENT_WILLARD",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4500),
        CausalEdge(source_id="LOC_KURTZ_COMPOUND", target_id="ENT_WILLARD",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=6000),
        CausalEdge(source_id="LOC_KURTZ_COMPOUND", target_id="ENT_LANCE",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000),
        CausalEdge(source_id="LOC_TEMPLE_INTERIOR", target_id="ENT_KURTZ",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=6000),

        # ── WORLD_VIETNAM_WAR → events (governance) ──
        CausalEdge(source_id="WORLD_VIETNAM_WAR", target_id="EVT_NHA_TRANG_BRIEFING",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=500, propagation_delay=0),
        CausalEdge(source_id="WORLD_VIETNAM_WAR", target_id="EVT_NAPALM_SURF_RAID",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=9.0, fabula_time=1800, propagation_delay=0),
        CausalEdge(source_id="WORLD_VIETNAM_WAR", target_id="EVT_USO_CHAOS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=2500, propagation_delay=0),
        CausalEdge(source_id="WORLD_VIETNAM_WAR", target_id="EVT_SAMPAN_MASSACRE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3500, propagation_delay=0),

        # ── WORLD_HEART_OF_DARKNESS → events (cosmology) ──
        CausalEdge(source_id="WORLD_HEART_OF_DARKNESS", target_id="EVT_ARRIVE_COMPOUND",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="WORLD_HEART_OF_DARKNESS", target_id="EVT_WILLARD_READS_DOSSIER",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="WORLD_HEART_OF_DARKNESS", target_id="EVT_BUFFALO_RITUAL",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=8000, propagation_delay=0),
        CausalEdge(source_id="WORLD_HEART_OF_DARKNESS", target_id="EVT_KURTZ_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=8100, propagation_delay=0),

        # ── WORLD_CHAIN_OF_COMMAND → events (governance) ──
        CausalEdge(source_id="WORLD_CHAIN_OF_COMMAND", target_id="EVT_NHA_TRANG_BRIEFING",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=9.0, fabula_time=500, propagation_delay=0),
        CausalEdge(source_id="WORLD_CHAIN_OF_COMMAND", target_id="EVT_BOARD_PBR",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="WORLD_CHAIN_OF_COMMAND", target_id="EVT_SAMPAN_MASSACRE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="WORLD_CHAIN_OF_COMMAND", target_id="EVT_WILLARD_DEPARTS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=8500, propagation_delay=0),

        # ── WORLD_RIVER_AS_FATE → events (cosmology) ──
        CausalEdge(source_id="WORLD_RIVER_AS_FATE", target_id="EVT_BOARD_PBR",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="WORLD_RIVER_AS_FATE", target_id="EVT_DO_LUNG_BRIDGE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=4500, propagation_delay=0),
        CausalEdge(source_id="WORLD_RIVER_AS_FATE", target_id="EVT_CHIEF_SPEARED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=5500, propagation_delay=0),
        CausalEdge(source_id="WORLD_RIVER_AS_FATE", target_id="EVT_ARRIVE_COMPOUND",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000, propagation_delay=0),

        # ── orphan utterance wirings ──
        CausalEdge(source_id="EVT_NHA_TRANG_BRIEFING", target_id="EVT_UTT_NHA_TRANG_KILL_ORDER",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=500, propagation_delay=50),
        CausalEdge(source_id="EVT_UTT_NHA_TRANG_KILL_ORDER", target_id="EVT_BOARD_PBR",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=7.0, fabula_time=550, propagation_delay=450),
        CausalEdge(source_id="EVT_UTT_DOSSIER_PROFILES_KURTZ", target_id="EVT_WILLARD_READS_DOSSIER",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3000, propagation_delay=4000),
        CausalEdge(source_id="EVT_UTT_KILGORE_AIR_ASSAULT_ORDER", target_id="EVT_NAPALM_SURF_RAID",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1780, propagation_delay=20),
        CausalEdge(source_id="EVT_NAPALM_SURF_RAID", target_id="EVT_UTT_KILGORE_NAPALM_SOLILOQUY",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1800, propagation_delay=300),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="EVT_UTT_PHOTOJOURNALIST_PROPHESIES",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=4.0, fabula_time=6000, propagation_delay=50),
        CausalEdge(source_id="EVT_UTT_KURTZ_HORROR_DOCTRINE", target_id="EVT_BUFFALO_RITUAL",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=7200, propagation_delay=800),
        # Kurtz's last words are uttered DURING the killing (machete strikes), not after
        # his death snapshot. They are a parallel consequence of his mortal realization,
        # not caused by the death-finalisation. The earlier KURTZ_KILLED -> LAST_WORDS
        # edge would imply post-mortem speech, so we drop it.


        # ─── auto-patched mutation_social edges (per-axis coverage) ───
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="ENT_KURTZ", rel_counterpart_id="ENT_WILLARD", causality_type="mutation_social", trait_target="affinity", trait_delta=0.4, mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_BOARD_PBR", target_id="ENT_WILLARD", rel_counterpart_id="ENT_CHIEF", causality_type="mutation_social", trait_target="affinity", trait_delta=0.3, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_BOARD_PBR", target_id="ENT_LANCE", rel_counterpart_id="ENT_CHEF", causality_type="mutation_social", trait_target="affinity", trait_delta=0.55, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_BOARD_PBR", target_id="ENT_CHEF", rel_counterpart_id="ENT_LANCE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.55, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_CHEF", rel_counterpart_id="ENT_WILLARD", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.4, mechanism="betrayal", evidence_strength="strong", causal_force=8.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_BOARD_PBR", target_id="ENT_CLEAN", rel_counterpart_id="ENT_CHIEF", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_JOIN_KILGORE", target_id="ENT_KILGORE", rel_counterpart_id="ENT_WILLARD", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1500, propagation_delay=0),
        CausalEdge(source_id="EVT_NAPALM_SURF_RAID", target_id="ENT_WILLARD", rel_counterpart_id="ENT_KILGORE", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.2, mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=1800, propagation_delay=0),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="ENT_LANCE", rel_counterpart_id="ENT_KURTZ", causality_type="mutation_social", trait_target="affinity", trait_delta=0.85, mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="ENT_PHOTOJOURNALIST", rel_counterpart_id="ENT_KURTZ", causality_type="mutation_social", trait_target="affinity", trait_delta=0.95, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_WILLARD_DEPARTS", target_id="ENT_WILLARD", rel_counterpart_id="ENT_LANCE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="emotional", evidence_strength="moderate", causal_force=6.0, fabula_time=8500, propagation_delay=0),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_CHEF", rel_counterpart_id="ENT_WILLARD", causality_type="mutation_social", trait_target="fear", trait_delta=0.45, mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_WILLARD_CAGED", target_id="ENT_WILLARD", rel_counterpart_id="ENT_KURTZ", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.65, mechanism="physical", evidence_strength="strong", causal_force=9.0, fabula_time=6500, propagation_delay=0),
        CausalEdge(source_id="EVT_WILLARD_CAGED", target_id="ENT_KURTZ", rel_counterpart_id="ENT_WILLARD", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.65, mechanism="physical", evidence_strength="strong", causal_force=9.0, fabula_time=6500, propagation_delay=0),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_CHIEF", rel_counterpart_id="ENT_WILLARD", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.3, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="ENT_WILLARD", rel_counterpart_id="ENT_CHIEF", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.3, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=3500, propagation_delay=0),
        CausalEdge(source_id="EVT_BOARD_PBR", target_id="ENT_CLEAN", rel_counterpart_id="ENT_CHIEF", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.65, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_JOIN_KILGORE", target_id="ENT_KILGORE", rel_counterpart_id="ENT_WILLARD", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.65, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=1500, propagation_delay=0),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="ENT_LANCE", rel_counterpart_id="ENT_KURTZ", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.65, mechanism="social", evidence_strength="strong", causal_force=8.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="ENT_PHOTOJOURNALIST", rel_counterpart_id="ENT_KURTZ", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.65, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=6000, propagation_delay=0),

        # ─── per-axis coverage for newly-observed antagonism / bond axes ───
        CausalEdge(source_id="EVT_DO_LUNG_BRIDGE", target_id="ENT_COLBY", rel_counterpart_id="ENT_KURTZ", causality_type="mutation_social", trait_target="affinity", trait_delta=0.8, mechanism="epistemic", evidence_strength="strong", causal_force=6.0, fabula_time=4500, propagation_delay=0),
        CausalEdge(source_id="EVT_DO_LUNG_BRIDGE", target_id="ENT_COLBY", rel_counterpart_id="ENT_KURTZ", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.8, mechanism="epistemic", evidence_strength="strong", causal_force=6.0, fabula_time=4500, propagation_delay=0),
    ],

    # ── SPATIAL TOPOLOGY ───────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_SAIGON_HOTEL", target_id="LOC_NHA_TRANG_HQ"),
        SpatialEdge(source_id="LOC_NHA_TRANG_HQ", target_id="LOC_PBR_RIVER"),
        SpatialEdge(source_id="LOC_PBR_RIVER", target_id="LOC_KILGORE_BEACH"),
        SpatialEdge(source_id="LOC_KILGORE_BEACH", target_id="LOC_PBR_RIVER"),
        SpatialEdge(source_id="LOC_PBR_RIVER", target_id="LOC_HAU_PHAT_USO"),
        SpatialEdge(source_id="LOC_HAU_PHAT_USO", target_id="LOC_PBR_RIVER"),
        SpatialEdge(source_id="LOC_PBR_RIVER", target_id="LOC_DO_LUNG_BRIDGE"),
        SpatialEdge(source_id="LOC_DO_LUNG_BRIDGE", target_id="LOC_PBR_RIVER"),
        SpatialEdge(source_id="LOC_PBR_RIVER", target_id="LOC_KURTZ_COMPOUND"),
        SpatialEdge(source_id="LOC_KURTZ_COMPOUND", target_id="LOC_TEMPLE_INTERIOR"),
        SpatialEdge(source_id="LOC_TEMPLE_INTERIOR", target_id="LOC_KURTZ_COMPOUND"),
        SpatialEdge(source_id="LOC_KURTZ_COMPOUND", target_id="LOC_PBR_RIVER"),
    ],

    # ── INFORMATION TOPOLOGY ───────────────────────────────────────────
    # Standing communication capabilities only. The Nha Trang kill-order, the
    # Photojournalist's rant, Kilgore's napalm soliloquy, and Kurtz's last words
    # are one-shot utterances and live as utterance EventNodes with
    # via_channel_id=None. Willard's voice-over narration is internal monologue
    # and is NOT a Channel.
    channels={
        'CHN_MISSION_DOSSIER': Channel(
            id='CHN_MISSION_DOSSIER',
            name="MACV-SOG classified mission dossier on Colonel Kurtz",
            medium='classified_pipeline',
            # The dossier itself, the Nha Trang brass who authored it, and the
            # cleared reader Willard. Outsiders (the PBR crew) are non-participants
            # and so cannot decode it.
            participant_ids=['OBJ_MISSION_DOSSIER', 'ENT_NHA_TRANG_BRASS', 'ENT_WILLARD'],
            directionality='simplex',
            intelligibility={'ENT_WILLARD': 1.0, 'ENT_NHA_TRANG_BRASS': 1.0,
                             'OBJ_MISSION_DOSSIER': 1.0},
            established_at_fabula=500,
            terminated_at_fabula=8500,
            evidence_strength='strong',
        ),
        'CHN_AIR_CAV_RADIO': Channel(
            id='CHN_AIR_CAV_RADIO',
            name="1/9th Air Cavalry tactical radio net (and PBR comms)",
            medium='military_radio',
            participant_ids=['ENT_KILGORE', 'ENT_WILLARD', 'ENT_CHIEF',
                             'ENT_LANCE', 'ENT_CHEF', 'ENT_CLEAN'],
            directionality='duplex',
            intelligibility={'ENT_KILGORE': 1.0, 'ENT_WILLARD': 1.0,
                             'ENT_CHIEF': 1.0, 'ENT_LANCE': 0.8,
                             'ENT_CHEF': 0.8, 'ENT_CLEAN': 0.8},
            established_at_fabula=1500,
            terminated_at_fabula=8500,
            evidence_strength='strong',
        ),
    },

    # ── WORLD TRAITS ───────────────────────────────────────────────────
    world_traits={
        "WORLD_VIETNAM_WAR": GlobalTrait(
            id="WORLD_VIETNAM_WAR",
            name="The Vietnam War (1968)",
            description="The Cold-War proxy theatre whose machinery — MACV-SOG sanctioned-murder orders, Air Cavalry napalm strikes, USO morale shows, free-fire-zone civilian killings — produces both Willard's mission and Kurtz's collapse. Common-cause parent of the briefing, Kilgore's raid, the USO carnival, and the sampan massacre.",
            category="governance",
            magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
            affected_domains=["social", "psychological", "physical"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=8500, triggered_by="EVT_WILLARD_DEPARTS",
                    magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    description="The war grinds on regardless of one Captain's defection upriver."),
            ],
        ),
        "WORLD_HEART_OF_DARKNESS": GlobalTrait(
            id="WORLD_HEART_OF_DARKNESS",
            name="Heart of Darkness (the Conradian Abyss)",
            description="The cosmological fact that civilisation thins as one moves upriver — that horror is not the war's exception but its essence. Common-cause parent of the compound's horror, Willard's conversion, the buffalo ritual, and Kurtz's death.",
            category="cosmology",
            magnitude=TraitVector(value=0.9, inertia=0.95, evidence_strength="strong"),
            affected_domains=["psychological", "social", "emotional"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=8100, triggered_by="EVT_KURTZ_KILLED",
                    magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    description="The horror Kurtz named is confirmed by his executioner's silence."),
            ],
        ),
        "WORLD_CHAIN_OF_COMMAND": GlobalTrait(
            id="WORLD_CHAIN_OF_COMMAND",
            name="Army Chain of Command (Sanctioned-Murder Protocol)",
            description="The bureaucratic legitimacy that turns assassination into orders, civilians into permissible casualties, and Willard into the chosen instrument of Kurtz's termination. Common-cause parent of the briefing, the boarding of the PBR, the sampan finishing-shot, and Willard's final refusal of the radio.",
            category="governance",
            magnitude=TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            affected_domains=["social", "informational"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=8500, triggered_by="EVT_WILLARD_DEPARTS",
                    magnitude=TraitVector(value=0.5, inertia=0.85, evidence_strength="strong"),
                    description="For Willard the protocol is dead — he switches off the radio that would call in the airstrike."),
            ],
        ),
        "WORLD_RIVER_AS_FATE": GlobalTrait(
            id="WORLD_RIVER_AS_FATE",
            name="The River as Fate",
            description="The Nung River as one-way vector — geographical predestination dragging the PBR past every diminishing outpost of meaning toward Kurtz. Common-cause parent of the boarding, Do Lung, Chief's death, and the arrival at the compound.",
            category="cosmology",
            magnitude=TraitVector(value=0.85, inertia=0.95, evidence_strength="strong"),
            affected_domains=["physical", "psychological"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=6000, triggered_by="EVT_ARRIVE_COMPOUND",
                    magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    description="The river delivers what it always intended: Willard at Kurtz's door."),
            ],
        ),
    },

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────
    social_topology=[
        # Willard ↔ Kurtz — the fated dyad.
        RelationshipEdge(
            source_entity_id="ENT_WILLARD", target_entity_id="ENT_KURTZ",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.6, evidence_strength="strong", last_updated_fabula=7000),
                "fear":          RelationshipMetric(value=0.55, inertia=0.4, evidence_strength="strong", last_updated_fabula=6500),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=6500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_KURTZ", target_entity_id="ENT_WILLARD",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.7, evidence_strength="strong", last_updated_fabula=7000),
                "power_dynamic": RelationshipMetric(value=0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=6500),
            },
        ),
        # Chief ↔ Willard — the doomed authority dispute.
        RelationshipEdge(
            source_entity_id="ENT_CHIEF", target_entity_id="ENT_WILLARD",
            metrics={
                "affinity":      RelationshipMetric(value=-0.55, inertia=0.6, evidence_strength="strong", last_updated_fabula=3500),
                "fear":          RelationshipMetric(value=0.5, inertia=0.4, evidence_strength="strong", last_updated_fabula=3500),
                "power_dynamic": RelationshipMetric(value=-0.3, inertia=0.6, evidence_strength="moderate", last_updated_fabula=3500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_WILLARD", target_entity_id="ENT_CHIEF",
            metrics={
                "affinity":      RelationshipMetric(value=0.3, inertia=0.5, evidence_strength="moderate", last_updated_fabula=5500),
                "power_dynamic": RelationshipMetric(value=0.3, inertia=0.6, evidence_strength="moderate", last_updated_fabula=3500),
            },
        ),
        # Crew bonds.
        RelationshipEdge(
            source_entity_id="ENT_LANCE", target_entity_id="ENT_CHEF",
            metrics={
                "affinity": RelationshipMetric(value=0.55, inertia=0.5, evidence_strength="moderate", last_updated_fabula=3500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_CHEF", target_entity_id="ENT_LANCE",
            metrics={
                "affinity": RelationshipMetric(value=0.55, inertia=0.5, evidence_strength="moderate", last_updated_fabula=3500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_CHEF", target_entity_id="ENT_WILLARD",
            metrics={
                "affinity": RelationshipMetric(value=-0.4, inertia=0.5, evidence_strength="strong", last_updated_fabula=6000),
                "fear":     RelationshipMetric(value=0.45, inertia=0.4, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_CLEAN", target_entity_id="ENT_CHIEF",
            metrics={
                "affinity":      RelationshipMetric(value=0.6, inertia=0.55, evidence_strength="moderate", last_updated_fabula=3500),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=1000),
            },
        ),
        # Kilgore — distant authority over Willard.
        RelationshipEdge(
            source_entity_id="ENT_KILGORE", target_entity_id="ENT_WILLARD",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.55, evidence_strength="moderate", last_updated_fabula=1800),
                "power_dynamic": RelationshipMetric(value=0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=1800),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_WILLARD", target_entity_id="ENT_KILGORE",
            metrics={
                "affinity": RelationshipMetric(value=-0.2, inertia=0.5, evidence_strength="moderate", last_updated_fabula=1800),
            },
        ),
        # Lance under Kurtz's spell.
        RelationshipEdge(
            source_entity_id="ENT_LANCE", target_entity_id="ENT_KURTZ",
            metrics={
                "affinity":      RelationshipMetric(value=0.85, inertia=0.7, evidence_strength="strong", last_updated_fabula=7000),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.7, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        # Photojournalist's discipleship.
        RelationshipEdge(
            source_entity_id="ENT_PHOTOJOURNALIST", target_entity_id="ENT_KURTZ",
            metrics={
                "affinity":      RelationshipMetric(value=0.95, inertia=0.85, evidence_strength="strong", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=-0.65, inertia=0.85, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        # Colby — the predecessor.
        RelationshipEdge(
            source_entity_id="ENT_COLBY", target_entity_id="ENT_KURTZ",
            metrics={
                "affinity": RelationshipMetric(value=0.8, inertia=0.95, evidence_strength="strong", last_updated_fabula=4500),
                "power_dynamic": RelationshipMetric(value=-0.8, inertia=0.95, evidence_strength="strong", last_updated_fabula=4500),
            },
        ),
        # Kurtz → Colby — reverse dominance after Colby's defection.
        RelationshipEdge(
            source_entity_id="ENT_KURTZ", target_entity_id="ENT_COLBY",
            metrics={
                "power_dynamic": RelationshipMetric(value=0.8, inertia=0.95, evidence_strength="strong", last_updated_fabula=4500),
            },
        ),
        # Willard ↔ Lance — the surviving pair.
        RelationshipEdge(
            source_entity_id="ENT_WILLARD", target_entity_id="ENT_LANCE",
            metrics={
                "affinity": RelationshipMetric(value=0.5, inertia=0.5, evidence_strength="moderate", last_updated_fabula=8500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_LANCE", target_entity_id="ENT_WILLARD",
            metrics={
                "affinity": RelationshipMetric(value=0.5, inertia=0.5, evidence_strength="moderate", last_updated_fabula=8500),
            },
        ),
    ],
)
