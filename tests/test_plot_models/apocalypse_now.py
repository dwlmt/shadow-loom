from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, RelationshipEdge, TraitVector,
    Affordance, Belief,
)

# =============================================================================
# APOCALYPSE NOW — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
                "LOC_SAIGON_HOTEL": Location(
            name="Saigon Hotel Room",
            description="Saigon Hotel Room",
            ambient_state={"despair": {"value": 0.8, "volatility": 0.4}, "confinement": {"value": 0.7, "volatility": 0.3}},
        ),
                "LOC_NHA_TRANG": Location(
            name="Nha Trang (Military Briefing)",
            description="Nha Trang (Military Briefing)",
            ambient_state={"authority": {"value": 0.8, "volatility": 0.1}},
        ),
                "LOC_NUNG_RIVER": Location(
            name="Nung River (PBR Patrol Boat)",
            description="Nung River (PBR Patrol Boat)",
            ambient_state={"danger": {"value": 0.7, "volatility": 0.5}, "isolation": {"value": 0.6, "volatility": 0.4}},
        ),
                "LOC_KILGORE_VILLAGE": Location(
            name="Vietcong Village (Kilgore's Air Attack)",
            description="Vietcong Village (Kilgore's Air Attack)",
            ambient_state={"destruction": {"value": 0.9, "volatility": 0.3}},
        ),
                "LOC_JUNGLE": Location(
            name="Jungle (Tiger Encounter)",
            description="Jungle (Tiger Encounter)",
            ambient_state={"danger": {"value": 0.9, "volatility": 0.5}, "darkness": {"value": 0.8, "volatility": 0.3}},
        ),
                "LOC_SUPPLY_DEPOT": Location(
            name="U.S. Supply Depot / USO Stage",
            description="U.S. Supply Depot / USO Stage",
            ambient_state={"surrealism": {"value": 0.7, "volatility": 0.5}},
        ),
                "LOC_DO_LUNG_BRIDGE": Location(
            name="Do Lung Bridge (Last Outpost)",
            description="Do Lung Bridge (Last Outpost) (last_outpost)",
            ambient_state={"chaos": {"value": 0.95, "volatility": 0.3}, "danger": {"value": 0.9, "volatility": 0.2}},
        ),
                "LOC_KURTZ_COMPOUND": Location(
            name="Kurtz's Compound (Cambodia)",
            description="Kurtz's Compound (Cambodia)",
            ambient_state={"madness": {"value": 1.0, "volatility": 0.1}, "death": {"value": 0.95, "volatility": 0.1}},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_KURTZ_DOSSIER": NarrativeObject(
            id="OBJ_KURTZ_DOSSIER",
            name="Kurtz's Military Dossier",
            location_id=None,
            owner_id="ENT_WILLARD",
            properties={"state": "obsessively_reviewed"},
            affordances=[
                Affordance(action="inform", target_type="Entity"),
                Affordance(action="obsess", target_type="Entity"),
            ],
        ),
        "OBJ_PBR": NarrativeObject(
            id="OBJ_PBR",
            name="Navy Patrol Boat River (PBR)",
            location_id="LOC_KURTZ_COMPOUND",
            owner_id=None,
            properties={"state": "operational"},
            affordances=[
                Affordance(action="transport", target_type="Entity"),
            ],
        ),
        "OBJ_MACHETE": NarrativeObject(
            id="OBJ_MACHETE",
            name="Willard's Machete",
            location_id="LOC_KURTZ_COMPOUND",
            owner_id=None,
            properties={"state": "bloodied"},
            affordances=[
                Affordance(action="kill", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_WILLARD": Entity(
            id="ENT_WILLARD",
            name="Captain Benjamin Willard",
            location_id="LOC_KURTZ_COMPOUND",
            status="healthy",
            traits={
                "obsession": TraitVector(value=0.9, inertia=0.7),
                "detachment": TraitVector(value=0.85, inertia=0.6),
                "ruthlessness": TraitVector(value=0.8, inertia=0.6),
                "self_destruction": TraitVector(value=0.7, inertia=0.5),
                "discipline": TraitVector(value=0.75, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_KURTZ", perceived_state="Kurtz had to be terminated — his methods were unsound", confidence=0.9, inertia=0.7),
            ],
        ),
        "ENT_KURTZ": Entity(
            id="ENT_KURTZ",
            name="Colonel Walter E. Kurtz",
            location_id="LOC_KURTZ_COMPOUND",
            status="dead",
            traits={
                "intellect": TraitVector(value=0.95, inertia=0.9),
                "madness": TraitVector(value=0.9, inertia=0.7),
                "charisma": TraitVector(value=0.9, inertia=0.8),
                "brutality": TraitVector(value=0.9, inertia=0.7),
                "nihilism": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_WILLARD", perceived_state="You are the one they sent to kill me", confidence=1.0, inertia=0.9),
            ],
        ),
        "ENT_KILGORE": Entity(
            id="ENT_KILGORE",
            name="Lieutenant Colonel Bill Kilgore",
            location_id="LOC_KILGORE_VILLAGE",
            status="healthy",
            traits={
                "bravado": TraitVector(value=0.95, inertia=0.9),
                "recklessness": TraitVector(value=0.9, inertia=0.8),
                "war_love": TraitVector(value=0.9, inertia=0.9),
            },
            beliefs=[
                Belief(target_id="ENT_KURTZ", perceived_state="War is glorious — I love the smell of napalm in the morning", confidence=0.95, inertia=0.9),
            ],
        ),
        "ENT_CHIEF": Entity(
            id="ENT_CHIEF",
            name="Chief (PBR Commander)",
            location_id="LOC_NUNG_RIVER",
            status="dead",
            traits={
                "discipline": TraitVector(value=0.8, inertia=0.7),
                "duty": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_WILLARD", perceived_state="I am the boat commander — my authority governs this mission", confidence=0.85, inertia=0.7),
            ],
        ),
        "ENT_CHEF": Entity(
            id="ENT_CHEF",
            name="Chef (PBR Crew)",
            location_id="LOC_KURTZ_COMPOUND",
            status="dead",
            traits={
                "anxiety": TraitVector(value=0.8, inertia=0.5),
                "vulnerability": TraitVector(value=0.75, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_KURTZ", perceived_state="The jungle and this mission will kill us all", confidence=0.85, inertia=0.6),
            ],
        ),
        "ENT_LANCE": Entity(
            id="ENT_LANCE",
            name="Lance (PBR Crew)",
            location_id="LOC_KURTZ_COMPOUND",
            status="healthy",
            traits={
                "detachment": TraitVector(value=0.8, inertia=0.6),
                "drug_use": TraitVector(value=0.8, inertia=0.6),
            },
        ),
        "ENT_CLEAN": Entity(
            id="ENT_CLEAN",
            name="Clean (PBR Crew)",
            location_id="LOC_NUNG_RIVER",
            status="dead",
            traits={
                "youth": TraitVector(value=0.9, inertia=0.8),
                "innocence": TraitVector(value=0.7, inertia=0.4),
            },
        ),
        "ENT_PHOTOJOURNALIST": Entity(
            id="ENT_PHOTOJOURNALIST",
            name="American Photojournalist",
            location_id="LOC_KURTZ_COMPOUND",
            status="healthy",
            traits={
                "fanaticism": TraitVector(value=0.85, inertia=0.7),
                "mania": TraitVector(value=0.8, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_KURTZ", perceived_state="Kurtz is a genius — the man has enlarged my mind", confidence=0.95, inertia=0.8),
            ],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_WILLARD_IN_SAIGON", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_id="ENT_WILLARD", description="Willard is holed up in a Saigon hotel room, intoxicated and desperate for a mission, having returned for a second tour."),
        EventNode(id="EVT_MISSION_BRIEFING", fabula_time=2, syuzhet_index=2, event_type="choice", actor_id=None, description="Willard is briefed at Nha Trang: terminate Colonel Kurtz, a rogue Green Beret commanding a Montagnard army in Cambodia."),
        EventNode(id="EVT_KILGORE_ATTACK", fabula_time=3, syuzhet_index=3, event_type="choice", actor_id="ENT_KILGORE", description="Kilgore orders an air attack on a Vietcong village, playing Wagner's Ride of the Valkyries. The PBR is placed in the river."),
        EventNode(id="EVT_TIGER_ENCOUNTER", fabula_time=4, syuzhet_index=4, event_type="outcome", actor_id=None, description="Chef and Willard disembark in the jungle searching for mangoes. A tiger lunges at them; Chef has a nervous breakdown."),
        EventNode(id="EVT_USO_SHOW", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_id=None, description="At a supply depot, the crew watches a USO show with Playboy Playmates that descends into chaos as soldiers storm the stage."),
        EventNode(id="EVT_SAMPAN_MASSACRE", fabula_time=6, syuzhet_index=6, event_type="choice", actor_id="ENT_WILLARD", description="Chief orders a sampan search. Clean panics and kills the civilians. Willard executes the surviving woman to avoid a detour."),
        EventNode(id="EVT_CLEAN_KILLED", fabula_time=7, syuzhet_index=7, event_type="outcome", actor_id=None, description="Past Do Lung Bridge, the PBR comes under surprise Vietcong attack. Clean is fatally shot while listening to a tape from his mother."),
        EventNode(id="EVT_CHIEF_KILLED", fabula_time=8, syuzhet_index=8, event_type="outcome", actor_id=None, description="Natives attack the PBR with arrows. Chief is impaled with a spear and dies."),
        EventNode(id="EVT_ARRIVE_COMPOUND", fabula_time=9, syuzhet_index=9, event_type="outcome", actor_id=None, description="The surviving crew reaches Kurtz's macabre compound, strewn with corpses and severed heads. The photojournalist greets them."),
        EventNode(id="EVT_WILLARD_IMPRISONED", fabula_time=10, syuzhet_index=10, event_type="outcome", actor_id="ENT_KURTZ", description="Kurtz imprisons Willard in a tiger cage. During the night, Kurtz throws Chef's severed head into Willard's lap."),
        EventNode(id="EVT_KURTZ_PHILOSOPHIZES", fabula_time=11, syuzhet_index=11, event_type="revelation", actor_id="ENT_KURTZ", description="Willard is freed and given freedom to roam. He listens to Kurtz's philosophizing for several days."),
        EventNode(id="EVT_KURTZ_KILLED", fabula_time=12, syuzhet_index=12, event_type="choice", actor_id="ENT_WILLARD", description="Intercut with the ritual sacrifice of a caribou, Willard slaughters Kurtz with a machete. Kurtz's last words: 'The horror, the horror.'"),
        EventNode(id="EVT_WILLARD_DEPARTS", fabula_time=13, syuzhet_index=13, event_type="choice", actor_id="ENT_WILLARD", description="The natives acknowledge Willard as their new leader. He throws down the machete, collects Lance, and departs on the PBR."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="ENT_WILLARD", target_id="EVT_WILLARD_IN_SAIGON", mechanism="psychological"),
        CausalEdge(source_id="EVT_WILLARD_IN_SAIGON", target_id="EVT_MISSION_BRIEFING", mechanism="social"),
        CausalEdge(source_id="OBJ_KURTZ_DOSSIER", target_id="EVT_MISSION_BRIEFING", mechanism="epistemic"),
        CausalEdge(source_id="EVT_MISSION_BRIEFING", target_id="EVT_KILGORE_ATTACK", mechanism="social"),
        CausalEdge(source_id="ENT_KILGORE", target_id="EVT_KILGORE_ATTACK", mechanism="physical"),
        CausalEdge(source_id="EVT_KILGORE_ATTACK", target_id="EVT_TIGER_ENCOUNTER", mechanism="physical"),
        CausalEdge(source_id="EVT_TIGER_ENCOUNTER", target_id="EVT_USO_SHOW", mechanism="social"),
        CausalEdge(source_id="ENT_CHIEF", target_id="EVT_SAMPAN_MASSACRE", mechanism="social"),
        CausalEdge(source_id="ENT_WILLARD", target_id="EVT_SAMPAN_MASSACRE", mechanism="physical"),
        CausalEdge(source_id="EVT_SAMPAN_MASSACRE", target_id="EVT_CLEAN_KILLED", mechanism="physical"),
        CausalEdge(source_id="EVT_CLEAN_KILLED", target_id="EVT_CHIEF_KILLED", mechanism="physical"),
        CausalEdge(source_id="EVT_CHIEF_KILLED", target_id="EVT_ARRIVE_COMPOUND", mechanism="physical"),
        CausalEdge(source_id="ENT_KURTZ", target_id="EVT_WILLARD_IMPRISONED", mechanism="physical"),
        CausalEdge(source_id="EVT_ARRIVE_COMPOUND", target_id="EVT_WILLARD_IMPRISONED", mechanism="social"),
        CausalEdge(source_id="EVT_WILLARD_IMPRISONED", target_id="EVT_KURTZ_PHILOSOPHIZES", mechanism="psychological"),
        CausalEdge(source_id="EVT_KURTZ_PHILOSOPHIZES", target_id="EVT_KURTZ_KILLED", mechanism="psychological"),
        CausalEdge(source_id="OBJ_MACHETE", target_id="EVT_KURTZ_KILLED", mechanism="physical"),
        CausalEdge(source_id="EVT_KURTZ_KILLED", target_id="EVT_WILLARD_DEPARTS", mechanism="social"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_DO_LUNG_BRIDGE", target_id="LOC_NUNG_RIVER"),
        SpatialEdge(source_id="LOC_JUNGLE", target_id="LOC_NUNG_RIVER"),
        SpatialEdge(source_id="LOC_KILGORE_VILLAGE", target_id="LOC_NUNG_RIVER"),
        SpatialEdge(source_id="LOC_KURTZ_COMPOUND", target_id="LOC_NUNG_RIVER"),
        SpatialEdge(source_id="LOC_NHA_TRANG", target_id="LOC_NUNG_RIVER"),
        SpatialEdge(source_id="LOC_NHA_TRANG", target_id="LOC_SAIGON_HOTEL"),
        SpatialEdge(source_id="LOC_NUNG_RIVER", target_id="LOC_SUPPLY_DEPOT"),
    ],
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_WILLARD", target_entity_id="ENT_KURTZ", affinity=0.2, friction=0.9, power_dynamic=-0.3, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_KURTZ", target_entity_id="ENT_WILLARD", affinity=0.3, friction=0.7, power_dynamic=0.5, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_WILLARD", target_entity_id="ENT_CHIEF", affinity=0.3, friction=0.6, power_dynamic=0.2, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_WILLARD", target_entity_id="ENT_CHEF", affinity=0.3, friction=0.4, power_dynamic=0.3, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_WILLARD", target_entity_id="ENT_LANCE", affinity=0.3, friction=0.3, power_dynamic=0.3, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_CHIEF", target_entity_id="ENT_WILLARD", affinity=0.1, friction=0.7, power_dynamic=-0.2, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_PHOTOJOURNALIST", target_entity_id="ENT_KURTZ", affinity=0.9, friction=0.2, power_dynamic=-0.9, inertia=0.7),
    ],
)
