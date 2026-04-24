from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, InformationEdge, RelationshipEdge, TraitVector,
    Affordance, Belief,
)

# =============================================================================
# A COURT OF THORNS AND ROSES — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
                "LOC_HUMAN_REALM": Location(
            name="Human Realm (Feyre's Family Cottage)",
            description="Human Realm (Feyre's Family Cottage)",
            ambient_state={"poverty": {"value": 0.8, "volatility": 0.3}, "hostility_to_fae": {"value": 0.9, "volatility": 0.1}},
        ),
                "LOC_WOODS": Location(
            name="The Woods (Hunting Grounds)",
            description="The Woods (Hunting Grounds)",
            ambient_state={"danger": {"value": 0.7, "volatility": 0.5}},
        ),
                "LOC_WALL": Location(
            name="The Wall (Boundary between Realms)",
            description="The Wall (Boundary between Realms) (boundary)",
            ambient_state={"magic": {"value": 0.6, "volatility": 0.3}},
        ),
                "LOC_SPRING_COURT": Location(
            name="The Spring Court (Tamlin's Estate)",
            description="The Spring Court (Tamlin's Estate)",
            ambient_state={"beauty": {"value": 0.8, "volatility": 0.4}, "danger": {"value": 0.6, "volatility": 0.6}},
        ),
                "LOC_UNDER_THE_MOUNTAIN": Location(
            name="Under the Mountain (Amarantha's Court)",
            description="Under the Mountain (Amarantha's Court)",
            ambient_state={"oppression": {"value": 0.95, "volatility": 0.1}, "cruelty": {"value": 0.9, "volatility": 0.2}},
        ),
                "LOC_NIGHT_COURT": Location(
            name="The Night Court (Rhysand's Domain)",
            description="The Night Court (Rhysand's Domain)",
            ambient_state={"power": {"value": 0.9, "volatility": 0.2}, "mystery": {"value": 0.85, "volatility": 0.3}},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_ASH_ARROW": NarrativeObject(
            id="OBJ_ASH_ARROW",
            name="Ash Arrow",
            location_id=None,
            owner_id="ENT_FEYRE",
            properties={"state": "used", "material": "ash_wood"},
            affordances=[
                Affordance(action="kill_faerie", target_type="Entity"),
            ],
        ),
        "OBJ_ASH_DAGGERS": NarrativeObject(
            id="OBJ_ASH_DAGGERS",
            name="Ash Wood Daggers (Third Task)",
            location_id="LOC_UNDER_THE_MOUNTAIN",
            owner_id=None,
            properties={"state": "used_in_trials"},
            affordances=[
                Affordance(action="stab", target_type="Entity"),
            ],
        ),
        "OBJ_GLAMOUR": NarrativeObject(
            id="OBJ_GLAMOUR",
            name="Faerie Glamour (on Feyre's Family)",
            location_id="LOC_HUMAN_REALM",
            owner_id="ENT_TAMLIN",
            properties={"state": "active", "effect": "false_memories"},
            affordances=[
                Affordance(action="deceive", target_type="Entity"),
            ],
        ),
        "OBJ_MASKS": NarrativeObject(
            id="OBJ_MASKS",
            name="Permanent Masks (Curse Manifestation)",
            location_id="LOC_SPRING_COURT",
            owner_id=None,
            properties={"state": "broken", "cause": "Amarantha_curse"},
            affordances=[
                Affordance(action="conceal_identity", target_type="Entity"),
            ],
        ),
        "OBJ_RIDDLE": NarrativeObject(
            id="OBJ_RIDDLE",
            name="Amarantha's Riddle",
            location_id="LOC_UNDER_THE_MOUNTAIN",
            owner_id=None,
            properties={"answer": "love", "state": "unsolved"},
            affordances=[
                Affordance(action="solve", target_type="Entity"),
                Affordance(action="break_curse", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_FEYRE": Entity(
            id="ENT_FEYRE",
            name="Feyre",
            location_id="LOC_SPRING_COURT",
            status="healthy",
            traits={
                "courage": TraitVector(value=0.9, inertia=0.7),
                "determination": TraitVector(value=0.9, inertia=0.8),
                "hatred_of_fae": TraitVector(value=0.2, inertia=0.2),
                "love": TraitVector(value=0.9, inertia=0.8),
                "guilt": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_TAMLIN", perceived_state="I love him — his heart is literally made of stone", confidence=0.95, inertia=0.9, established_at_fabula=0),
                Belief(target_id="OBJ_MASKS", perceived_state="The permanent masks are caused by a mysterious blight or plague", confidence=0.8, inertia=0.6, established_at_fabula=0),
                Belief(target_id="ENT_TAMLIN", perceived_state="Tamlin sent me home only to protect me from the blight's danger", confidence=0.75, inertia=0.5, established_at_fabula=0),
                Belief(target_id="ENT_TAMLIN", perceived_state="Tamlin is under a curse that I do not understand", confidence=0.7, inertia=0.5, established_at_fabula=6),
            ],
            constants=["high_fae_transformed"],
        ),
        "ENT_TAMLIN": Entity(
            id="ENT_TAMLIN",
            name="Tamlin (High Lord of the Spring Court)",
            location_id="LOC_SPRING_COURT",
            status="healthy",
            traits={
                "protective": TraitVector(value=0.9, inertia=0.8),
                "duty": TraitVector(value=0.8, inertia=0.7),
                "love": TraitVector(value=0.85, inertia=0.8),
                "power": TraitVector(value=0.85, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_FEYRE", perceived_state="I must protect Feyre even if it means losing her — the curse forbids me from telling her the truth", confidence=0.95, inertia=0.9, established_at_fabula=0),
                Belief(target_id="ENT_FEYRE", perceived_state="Feyre must be sent home before Amarantha discovers her", confidence=0.9, inertia=0.7, established_at_fabula=5),
            ],
            constants=["high_fae", "shapeshifter"],
        ),
        "ENT_RHYSAND": Entity(
            id="ENT_RHYSAND",
            name="Rhysand (High Lord of the Night Court)",
            location_id="LOC_NIGHT_COURT",
            status="healthy",
            traits={
                "cunning": TraitVector(value=0.9, inertia=0.85),
                "compassion": TraitVector(value=0.7, inertia=0.6),
                "power": TraitVector(value=0.95, inertia=0.9),
            },
            beliefs=[
                Belief(target_id="ENT_FEYRE", perceived_state="This mortal is the key to defeating Amarantha — I must ensure she survives", confidence=0.75, inertia=0.6, established_at_fabula=0),
            ],
            constants=["high_fae"],
        ),
        "ENT_AMARANTHA": Entity(
            id="ENT_AMARANTHA",
            name="Amarantha (High Queen of Prythian)",
            location_id="LOC_UNDER_THE_MOUNTAIN",
            status="dead",
            traits={
                "cruelty": TraitVector(value=0.95, inertia=0.9),
                "possessiveness": TraitVector(value=0.9, inertia=0.8),
                "power": TraitVector(value=0.9, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_FEYRE", perceived_state="This mortal girl cannot possibly break my curse — I have already won", confidence=0.9, inertia=0.7, established_at_fabula=0),
                Belief(target_id="ENT_TAMLIN", perceived_state="Tamlin will eventually submit to me and be mine", confidence=0.85, inertia=0.8, established_at_fabula=0),
            ],
        ),
        "ENT_LUCIEN": Entity(
            id="ENT_LUCIEN",
            name="Lucien",
            location_id="LOC_SPRING_COURT",
            status="healthy",
            traits={
                "loyalty": TraitVector(value=0.8, inertia=0.7),
                "wit": TraitVector(value=0.75, inertia=0.6),
            },
        ),
        "ENT_ALIS": Entity(
            id="ENT_ALIS",
            name="Alis (Lady-in-Waiting)",
            location_id="LOC_SPRING_COURT",
            status="healthy",
            traits={
                "loyalty": TraitVector(value=0.85, inertia=0.8),
                "kindness": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_NESTA": Entity(
            id="ENT_NESTA",
            name="Nesta (Feyre's Sister)",
            location_id="LOC_HUMAN_REALM",
            status="healthy",
            traits={
                "stubbornness": TraitVector(value=0.85, inertia=0.8),
                "defiance": TraitVector(value=0.8, inertia=0.7),
                "anger": TraitVector(value=0.75, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_FEYRE", perceived_state="Feyre was taken by faeries — our father did nothing, just as he did nothing for our mother", confidence=0.95, inertia=0.8, established_at_fabula=0),
            ],
            constants=["glamour_resistant"],
        ),
        "ENT_ANDRAS": Entity(
            id="ENT_ANDRAS",
            name="Andras (Faerie Wolf)",
            location_id="LOC_WOODS",
            status="dead",
            traits={},
            constants=["faerie"],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_FEYRE_KILLS_WOLF", fabula_time=1, syuzhet_index=1, event_type="choice", actor_ids=["ENT_FEYRE"], target_ids=["ENT_ANDRAS"], description="Feyre kills a wolf in the woods with an ash arrow, suspecting it might be a faerie (Andras)."),
        EventNode(id="EVT_TAMLIN_DEMANDS_PAYMENT", fabula_time=2, syuzhet_index=2, event_type="choice", actor_ids=["ENT_TAMLIN"], description="Tamlin arrives at Feyre's cottage in beast form, demanding payment for Andras's death under the Treaty."),
        EventNode(id="EVT_FEYRE_GOES_TO_PRYTHIAN", fabula_time=3, syuzhet_index=3, event_type="choice", actor_ids=["ENT_FEYRE"], description="Feyre chooses to go to Prythian rather than die, and begins living at the Spring Court."),
        EventNode(id="EVT_FEYRE_LEARNS_PRYTHIAN", fabula_time=4, syuzhet_index=4, event_type="revelation", actor_ids=["ENT_FEYRE"], description="Feyre learns about Prythian's history, magic, and the blight. She bonds with Alis and Lucien."),
        EventNode(id="EVT_RHYSAND_THREATENS", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_ids=["ENT_RHYSAND"], description="Rhysand visits the Spring Court. Tamlin decides Feyre is no longer safe and sends her home."),
        EventNode(id="EVT_TAMLIN_FEYRE_LOVE", fabula_time=6, syuzhet_index=6, event_type="outcome", actor_ids=[], description="Tamlin and Feyre make love and declare their feelings, but Feyre cannot say 'I love you' aloud."),
        EventNode(id="EVT_FEYRE_RETURNS_HOME", fabula_time=7, syuzhet_index=7, event_type="outcome", actor_ids=["ENT_FEYRE"], description="Feyre returns to find her family wealthy (Tamlin's doing). Nesta reveals the glamour didn't work on her."),
        EventNode(id="EVT_FEYRE_RETURNS_PRYTHIAN", fabula_time=8, syuzhet_index=8, event_type="choice", actor_ids=["ENT_FEYRE"], description="Inspired by Nesta's words, Feyre returns to Prythian to fight alongside Tamlin."),
        EventNode(id="EVT_ALIS_REVEALS_TRUTH", fabula_time=9, syuzhet_index=9, event_type="revelation", actor_ids=["ENT_ALIS"], description="Alis reveals the blight is Amarantha. The curse requires a woman who hates fae to declare love for Tamlin. The curse wasn't broken."),
        EventNode(id="EVT_FEYRE_ENTERS_MOUNTAIN", fabula_time=10, syuzhet_index=10, event_type="choice", actor_ids=["ENT_FEYRE"], description="Feyre journeys Under the Mountain to rescue Tamlin and Lucien."),
        EventNode(id="EVT_TASK_ONE", fabula_time=11, syuzhet_index=11, event_type="outcome", actor_ids=["ENT_FEYRE"], description="Feyre escapes a giant worm in a labyrinth. Rhysand bets on her and heals her broken arm."),
        EventNode(id="EVT_RHYSAND_BARGAIN", fabula_time=12, syuzhet_index=12, event_type="choice", actor_ids=["ENT_RHYSAND"], description="In exchange for healing, Feyre agrees to spend one week per month with Rhysand in the Night Court."),
        EventNode(id="EVT_TASK_TWO", fabula_time=13, syuzhet_index=13, event_type="outcome", actor_ids=["ENT_FEYRE"], description="Feyre solves the puzzle and picks the correct lever, aided by Rhysand's telepathic voice."),
        EventNode(id="EVT_TASK_THREE", fabula_time=14, syuzhet_index=14, event_type="choice", actor_ids=["ENT_FEYRE"], description="Feyre must stab three faeries. She stabs two, then realizes the third is Tamlin — his heart of stone protects him."),
        EventNode(id="EVT_AMARANTHA_BEATS_FEYRE", fabula_time=15, syuzhet_index=15, event_type="outcome", actor_ids=["ENT_AMARANTHA"], target_ids=["ENT_FEYRE"], description="Amarantha refuses to honor the bargain and beats Feyre nearly to death."),
        EventNode(id="EVT_RIDDLE_SOLVED", fabula_time=16, syuzhet_index=16, event_type="revelation", actor_ids=["ENT_FEYRE"], description="Feyre realizes the riddle's answer is 'love' and speaks it aloud, breaking the curse as she dies."),
        EventNode(id="EVT_AMARANTHA_DESTROYED", fabula_time=17, syuzhet_index=17, event_type="outcome", actor_ids=["ENT_TAMLIN"], description="With the curse broken, Tamlin and Rhysand join forces to destroy Amarantha."),
        EventNode(id="EVT_FEYRE_RESURRECTED", fabula_time=18, syuzhet_index=18, event_type="outcome", actor_ids=[], description="The six High Lords gift Feyre with healing light and immortality. Tamlin places golden light on her heart, transforming her into High Fae."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="EVT_FEYRE_KILLS_WOLF", target_id="EVT_TAMLIN_DEMANDS_PAYMENT", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1),
        CausalEdge(source_id="EVT_TAMLIN_DEMANDS_PAYMENT", target_id="EVT_FEYRE_GOES_TO_PRYTHIAN", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=2),
        CausalEdge(source_id="EVT_FEYRE_GOES_TO_PRYTHIAN", target_id="EVT_FEYRE_LEARNS_PRYTHIAN", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=3),
        CausalEdge(source_id="EVT_RHYSAND_THREATENS", target_id="EVT_TAMLIN_FEYRE_LOVE", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=5),
        CausalEdge(source_id="EVT_TAMLIN_FEYRE_LOVE", target_id="EVT_FEYRE_RETURNS_HOME", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=6),
        CausalEdge(source_id="EVT_FEYRE_RETURNS_PRYTHIAN", target_id="EVT_ALIS_REVEALS_TRUTH", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=8),
        CausalEdge(source_id="EVT_ALIS_REVEALS_TRUTH", target_id="EVT_FEYRE_ENTERS_MOUNTAIN", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=9),
        CausalEdge(source_id="EVT_RHYSAND_BARGAIN", target_id="EVT_TASK_TWO", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=12),
        CausalEdge(source_id="EVT_TASK_THREE", target_id="EVT_AMARANTHA_BEATS_FEYRE", causality_type="chain_reaction", mechanism="physical", evidence_strength="moderate", causal_force=5.0, fabula_time=14),
        CausalEdge(source_id="EVT_AMARANTHA_BEATS_FEYRE", target_id="EVT_RIDDLE_SOLVED", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=15),
        CausalEdge(source_id="EVT_RIDDLE_SOLVED", target_id="EVT_AMARANTHA_DESTROYED", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=16),
        CausalEdge(source_id="EVT_AMARANTHA_DESTROYED", target_id="EVT_FEYRE_RESURRECTED", causality_type="chain_reaction", mechanism="physical", evidence_strength="moderate", causal_force=5.0, fabula_time=17),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_HUMAN_REALM", target_id="LOC_WALL"),
        SpatialEdge(source_id="LOC_HUMAN_REALM", target_id="LOC_WOODS"),
        SpatialEdge(source_id="LOC_NIGHT_COURT", target_id="LOC_UNDER_THE_MOUNTAIN"),
        SpatialEdge(source_id="LOC_SPRING_COURT", target_id="LOC_UNDER_THE_MOUNTAIN"),
        SpatialEdge(source_id="LOC_SPRING_COURT", target_id="LOC_WALL"),
    ],
    information_topology=[
        InformationEdge(
            source_id="ENT_RHYSAND",
            target_ids=["ENT_FEYRE"],
            medium="telepathy",
            established_at_fabula=12,
        ),
        InformationEdge(
            source_id="ENT_ALIS",
            target_ids=["ENT_FEYRE"],
            medium="whispered_confession",
            established_at_fabula=8,
            terminated_at_fabula=8,
        ),
    ],
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_FEYRE", target_entity_id="ENT_TAMLIN", affinity=0.9, fear=0.2, power_dynamic=-0.3),
        RelationshipEdge(source_entity_id="ENT_TAMLIN", target_entity_id="ENT_FEYRE", affinity=0.9, fear=0.15, power_dynamic=0.4),
        RelationshipEdge(source_entity_id="ENT_RHYSAND", target_entity_id="ENT_FEYRE", affinity=0.6, fear=0.25, power_dynamic=0.3),
        RelationshipEdge(source_entity_id="ENT_AMARANTHA", target_entity_id="ENT_TAMLIN", affinity=0.7, fear=0.4, power_dynamic=0.8),
        RelationshipEdge(source_entity_id="ENT_AMARANTHA", target_entity_id="ENT_FEYRE", affinity=-0.9, fear=0.47, power_dynamic=0.9),
        RelationshipEdge(source_entity_id="ENT_FEYRE", target_entity_id="ENT_LUCIEN", affinity=0.6, fear=0.1, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_TAMLIN", target_entity_id="ENT_LUCIEN", affinity=0.75, fear=0.1, power_dynamic=0.4),
        RelationshipEdge(source_entity_id="ENT_FEYRE", target_entity_id="ENT_NESTA", affinity=0.5, fear=0.3, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_FEYRE", target_entity_id="ENT_RHYSAND", affinity=-0.2, fear=0.4, power_dynamic=-0.4),
    ],
)
