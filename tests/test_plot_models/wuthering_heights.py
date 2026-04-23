from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
    Affordance, Belief,
)

# =============================================================================
# WUTHERING HEIGHTS — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
        "LOC_WUTHERING_HEIGHTS": Location(
            id="LOC_WUTHERING_HEIGHTS",
            name="Wuthering Heights (Earnshaw Farmhouse)",
            connected_locations=["LOC_MOORS", "LOC_THRUSHCROSS_GRANGE"],
            ambient_states={
                "wildness": AmbientVector(value=0.9, volatility=0.3),
                "hostility": AmbientVector(value=0.8, volatility=0.4),
                "decay": AmbientVector(value=0.7, volatility=0.3),
            },
        ),
        "LOC_THRUSHCROSS_GRANGE": Location(
            id="LOC_THRUSHCROSS_GRANGE",
            name="Thrushcross Grange (Linton Estate)",
            connected_locations=["LOC_WUTHERING_HEIGHTS", "LOC_MOORS"],
            ambient_states={
                "refinement": AmbientVector(value=0.8, volatility=0.3),
                "domesticity": AmbientVector(value=0.7, volatility=0.4),
            },
        ),
        "LOC_MOORS": Location(
            id="LOC_MOORS",
            name="The Yorkshire Moors",
            connected_locations=["LOC_WUTHERING_HEIGHTS", "LOC_THRUSHCROSS_GRANGE"],
            ambient_states={
                "wildness": AmbientVector(value=0.95, volatility=0.2),
                "freedom": AmbientVector(value=0.8, volatility=0.2),
            },
        ),
        "LOC_LIVERPOOL": Location(
            id="LOC_LIVERPOOL",
            name="Liverpool",
            connected_locations=["LOC_WUTHERING_HEIGHTS"],
            ambient_states={},
        ),
        "LOC_SOUTH": Location(
            id="LOC_SOUTH",
            name="Southern England (Isabella's Refuge)",
            connected_locations=[],
            ambient_states={},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_CATHERINE_DIARY": NarrativeObject(
            id="OBJ_CATHERINE_DIARY",
            name="Catherine Earnshaw's Diary Entries",
            location_id="LOC_WUTHERING_HEIGHTS",
            owner_id=None,
            properties={"state": "read_by_lockwood"},
            affordances=[
                Affordance(action="reveal_past", target_type="Entity"),
            ],
        ),
        "OBJ_GRANGE_DEED": NarrativeObject(
            id="OBJ_GRANGE_DEED",
            name="Thrushcross Grange Inheritance (Entailed)",
            location_id="LOC_THRUSHCROSS_GRANGE",
            owner_id=None,
            properties={"state": "inherited_by_cathy"},
            affordances=[
                Affordance(action="confer_ownership", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_HEATHCLIFF": Entity(
            id="ENT_HEATHCLIFF",
            name="Heathcliff",
            location_id="LOC_WUTHERING_HEIGHTS",
            status="dead",
            traits={
                "vengefulness": TraitVector(value=0.95, inertia=0.9),
                "passion": TraitVector(value=0.95, inertia=0.85),
                "cruelty": TraitVector(value=0.85, inertia=0.7),
                "obsession": TraitVector(value=0.95, inertia=0.9),
            },
            beliefs=[
                Belief(target_id="ENT_CATHERINE", perceived_state="Catherine is my soul — without her I am nothing", confidence=1.0, inertia=0.95),
                Belief(target_id="ENT_CATHERINE", perceived_state="Catherine despised me for my low status and chose Edgar over me", confidence=0.8, inertia=0.7),
            ],
        ),
        "ENT_CATHERINE": Entity(
            id="ENT_CATHERINE",
            name="Catherine Earnshaw (later Linton)",
            location_id="LOC_THRUSHCROSS_GRANGE",
            status="dead",
            traits={
                "passion": TraitVector(value=0.9, inertia=0.85),
                "pride": TraitVector(value=0.8, inertia=0.7),
                "selfishness": TraitVector(value=0.7, inertia=0.6),
                "wildness": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_HEATHCLIFF", perceived_state="I am Heathcliff", confidence=1.0, inertia=0.95),
                Belief(target_id="ENT_EDGAR", perceived_state="Marrying Edgar will elevate me — it would degrade me to marry Heathcliff", confidence=0.7, inertia=0.5),
            ],
        ),
        "ENT_EDGAR": Entity(
            id="ENT_EDGAR",
            name="Edgar Linton",
            location_id="LOC_THRUSHCROSS_GRANGE",
            status="dead",
            traits={
                "gentleness": TraitVector(value=0.75, inertia=0.7),
                "jealousy": TraitVector(value=0.6, inertia=0.5),
                "refinement": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_CATHERINE", perceived_state="I can make Catherine happy if Heathcliff would leave us in peace", confidence=0.7, inertia=0.5),
            ],
        ),
        "ENT_HINDLEY": Entity(
            id="ENT_HINDLEY",
            name="Hindley Earnshaw",
            location_id="LOC_WUTHERING_HEIGHTS",
            status="dead",
            traits={
                "cruelty": TraitVector(value=0.8, inertia=0.7),
                "alcoholism": TraitVector(value=0.9, inertia=0.8),
                "bitterness": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_HEATHCLIFF", perceived_state="Heathcliff stole my father's love and ruined my birthright", confidence=0.95, inertia=0.9),
            ],
        ),
        "ENT_ISABELLA": Entity(
            id="ENT_ISABELLA",
            name="Isabella Linton",
            location_id="LOC_SOUTH",
            status="dead",
            traits={
                "naivete": TraitVector(value=0.7, inertia=0.5),
                "bitterness": TraitVector(value=0.7, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_HEATHCLIFF", perceived_state="Heathcliff loves me — I can reform him", confidence=0.8, inertia=0.4),
            ],
        ),
        "ENT_CATHY": Entity(
            id="ENT_CATHY",
            name="Cathy Linton (daughter)",
            location_id="LOC_WUTHERING_HEIGHTS",
            status="healthy",
            traits={
                "spiritedness": TraitVector(value=0.8, inertia=0.7),
                "kindness": TraitVector(value=0.75, inertia=0.6),
            },
        ),
        "ENT_HARETON": Entity(
            id="ENT_HARETON",
            name="Hareton Earnshaw",
            location_id="LOC_WUTHERING_HEIGHTS",
            status="healthy",
            traits={
                "roughness": TraitVector(value=0.7, inertia=0.5),
                "potential": TraitVector(value=0.7, inertia=0.6),
                "loyalty": TraitVector(value=0.7, inertia=0.6),
            },
        ),
        "ENT_LINTON_H": Entity(
            id="ENT_LINTON_H",
            name="Linton Heathcliff",
            location_id="LOC_WUTHERING_HEIGHTS",
            status="dead",
            traits={
                "frailty": TraitVector(value=0.9, inertia=0.85),
                "petulance": TraitVector(value=0.7, inertia=0.6),
            },
        ),
        "ENT_NELLY": Entity(
            id="ENT_NELLY",
            name="Ellen 'Nelly' Dean",
            location_id="LOC_WUTHERING_HEIGHTS",
            status="healthy",
            traits={
                "practicality": TraitVector(value=0.8, inertia=0.7),
                "loyalty": TraitVector(value=0.75, inertia=0.7),
            },
        ),
        "ENT_LOCKWOOD": Entity(
            id="ENT_LOCKWOOD",
            name="Mr Lockwood",
            location_id="LOC_THRUSHCROSS_GRANGE",
            status="healthy",
            traits={
                "curiosity": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_CATHY", perceived_state="The young woman at the Heights must be Heathcliff's wife", confidence=0.7, inertia=0.3),
            ],
        ),
        "ENT_EARNSHAW": Entity(
            id="ENT_EARNSHAW",
            name="Mr Earnshaw",
            location_id="LOC_WUTHERING_HEIGHTS",
            status="dead",
            traits={
                "paternal_love": TraitVector(value=0.8, inertia=0.7),
            },
        ),
    },

    # ── EVENTS (Chronological — fabula order) ──────────────────────────────
    events=[
        EventNode(id="EVT_HEATHCLIFF_BROUGHT", fabula_time=1, syuzhet_index=3, event_type="outcome", actor_id="ENT_EARNSHAW", description="Earnshaw returns from Liverpool with an orphan he names Heathcliff, favouring him over his own children."),
        EventNode(id="EVT_HINDLEY_BULLIES", fabula_time=2, syuzhet_index=4, event_type="outcome", actor_id="ENT_HINDLEY", description="Hindley beats and humiliates Heathcliff. Catherine and Heathcliff become inseparable companions on the moors."),
        EventNode(id="EVT_EARNSHAW_DIES", fabula_time=3, syuzhet_index=5, event_type="outcome", actor_id=None, description="Mr Earnshaw dies. Hindley inherits Wuthering Heights and forces Heathcliff to live as a servant."),
        EventNode(id="EVT_CATHERINE_AT_GRANGE", fabula_time=4, syuzhet_index=6, event_type="outcome", actor_id=None, description="Catherine is bitten by the Lintons' dog while spying. She stays at the Grange for weeks, returning refined."),
        EventNode(id="EVT_HEATHCLIFF_HUMILIATED", fabula_time=5, syuzhet_index=7, event_type="outcome", actor_id=None, description="Hindley and Edgar mock Heathcliff. He is banished to an attic and swears revenge."),
        EventNode(id="EVT_FRANCES_DIES", fabula_time=6, syuzhet_index=8, event_type="outcome", actor_id=None, description="Frances dies after giving birth to Hareton."),
        EventNode(id="EVT_CATHERINE_ACCEPTS_EDGAR", fabula_time=7, syuzhet_index=9, event_type="choice", actor_id="ENT_CATHERINE", description="Catherine accepts Edgar's proposal, confessing to Nelly she loves Heathcliff deeply but cannot marry him for his low status."),
        EventNode(id="EVT_HEATHCLIFF_FLEES", fabula_time=8, syuzhet_index=10, event_type="choice", actor_id="ENT_HEATHCLIFF", description="Heathcliff overhears part of Catherine's confession, misunderstands, and flees. Catherine falls ill."),
        EventNode(id="EVT_LINTON_PARENTS_DIE", fabula_time=9, syuzhet_index=11, event_type="outcome", actor_id=None, description="Mr and Mrs Linton both die of fever. Edgar inherits Thrushcross Grange."),
        EventNode(id="EVT_HEATHCLIFF_RETURNS", fabula_time=10, syuzhet_index=12, event_type="outcome", actor_id="ENT_HEATHCLIFF", description="Three years later Heathcliff returns, mysteriously wealthy. He exploits Hindley's gambling and becomes mortgagee of Wuthering Heights."),
        EventNode(id="EVT_ISABELLA_ELOPEMENT", fabula_time=11, syuzhet_index=13, event_type="choice", actor_id="ENT_HEATHCLIFF", description="Heathcliff manipulates Isabella into eloping, using her infatuation as revenge on Edgar. He is banished from the Grange."),
        EventNode(id="EVT_CATHERINE_ILL", fabula_time=12, syuzhet_index=14, event_type="outcome", actor_id="ENT_CATHERINE", description="Catherine locks herself in her room and refuses food for three days, becoming gravely ill."),
        EventNode(id="EVT_CATHERINE_DIES", fabula_time=13, syuzhet_index=15, event_type="outcome", actor_id=None, description="Heathcliff visits the dying Catherine in secret. She dies giving birth to Cathy. Heathcliff rages, calling on Catherine's ghost to haunt him."),
        EventNode(id="EVT_ISABELLA_FLEES", fabula_time=14, syuzhet_index=16, event_type="choice", actor_id="ENT_ISABELLA", description="Isabella, embittered by abuse, flees south. She gives birth to Linton Heathcliff."),
        EventNode(id="EVT_HINDLEY_DIES", fabula_time=15, syuzhet_index=17, event_type="outcome", actor_id=None, description="Hindley dies of alcoholism. Hareton inherits Wuthering Heights in name, but Heathcliff takes possession."),
        EventNode(id="EVT_LINTON_TO_HEIGHTS", fabula_time=16, syuzhet_index=18, event_type="outcome", actor_id="ENT_HEATHCLIFF", description="After Isabella's death, the sickly Linton is brought north. Heathcliff insists his son live at Wuthering Heights."),
        EventNode(id="EVT_FORCED_MARRIAGE", fabula_time=17, syuzhet_index=19, event_type="choice", actor_id="ENT_HEATHCLIFF", description="Heathcliff schemes to marry Cathy to Linton, hoping to control Thrushcross Grange's inheritance."),
        EventNode(id="EVT_EDGAR_DIES", fabula_time=18, syuzhet_index=20, event_type="outcome", actor_id=None, description="Edgar dies. Cathy and Linton move to Wuthering Heights."),
        EventNode(id="EVT_LINTON_DIES", fabula_time=19, syuzhet_index=21, event_type="outcome", actor_id=None, description="Linton Heathcliff dies. Cathy is trapped at Wuthering Heights under Heathcliff's control."),
        EventNode(id="EVT_LOCKWOOD_ARRIVES", fabula_time=20, syuzhet_index=1, event_type="outcome", actor_id="ENT_LOCKWOOD", description="Lockwood arrives as tenant of Thrushcross Grange and visits Wuthering Heights. He reads Catherine's diary and has a nightmare about her ghost."),
        EventNode(id="EVT_NELLY_NARRATES", fabula_time=21, syuzhet_index=2, event_type="outcome", actor_id="ENT_NELLY", description="Nelly Dean tells Lockwood the full story of the Earnshaw and Linton families."),
        EventNode(id="EVT_CATHY_HARETON_RECONCILE", fabula_time=22, syuzhet_index=22, event_type="outcome", actor_id="ENT_CATHY", description="Cathy seeks Hareton's forgiveness. They reconcile and fall in love. She teaches him to read."),
        EventNode(id="EVT_HEATHCLIFF_DECLINES", fabula_time=23, syuzhet_index=23, event_type="outcome", actor_id="ENT_HEATHCLIFF", description="Overmastered, Heathcliff avoids the young couple, stops eating, and becomes increasingly fixated on Catherine's ghost."),
        EventNode(id="EVT_HEATHCLIFF_DIES", fabula_time=24, syuzhet_index=24, event_type="outcome", actor_id=None, description="Heathcliff is found dead in Catherine's old room. Locals report seeing the ghosts of Catherine and Heathcliff on the moors."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="EVT_HEATHCLIFF_BROUGHT", target_id="EVT_HINDLEY_BULLIES", mechanism="social"),
        CausalEdge(source_id="EVT_HINDLEY_BULLIES", target_id="EVT_HEATHCLIFF_HUMILIATED", mechanism="psychological"),
        CausalEdge(source_id="EVT_EARNSHAW_DIES", target_id="EVT_HEATHCLIFF_HUMILIATED", mechanism="social"),
        CausalEdge(source_id="EVT_CATHERINE_AT_GRANGE", target_id="EVT_CATHERINE_ACCEPTS_EDGAR", mechanism="social"),
        CausalEdge(source_id="EVT_HEATHCLIFF_HUMILIATED", target_id="EVT_CATHERINE_ACCEPTS_EDGAR", mechanism="social"),
        CausalEdge(source_id="EVT_CATHERINE_ACCEPTS_EDGAR", target_id="EVT_HEATHCLIFF_FLEES", mechanism="psychological"),
        CausalEdge(source_id="EVT_HEATHCLIFF_FLEES", target_id="EVT_HEATHCLIFF_RETURNS", mechanism="psychological"),
        CausalEdge(source_id="EVT_HEATHCLIFF_RETURNS", target_id="EVT_ISABELLA_ELOPEMENT", mechanism="psychological"),
        CausalEdge(source_id="EVT_ISABELLA_ELOPEMENT", target_id="EVT_CATHERINE_ILL", mechanism="psychological"),
        CausalEdge(source_id="EVT_CATHERINE_ILL", target_id="EVT_CATHERINE_DIES", mechanism="physical"),
        CausalEdge(source_id="EVT_CATHERINE_DIES", target_id="EVT_ISABELLA_FLEES", mechanism="psychological"),
        CausalEdge(source_id="EVT_HEATHCLIFF_RETURNS", target_id="EVT_HINDLEY_DIES", mechanism="social"),
        CausalEdge(source_id="EVT_FRANCES_DIES", target_id="EVT_HINDLEY_DIES", mechanism="psychological"),
        CausalEdge(source_id="EVT_ISABELLA_FLEES", target_id="EVT_LINTON_TO_HEIGHTS", mechanism="social"),
        CausalEdge(source_id="EVT_LINTON_TO_HEIGHTS", target_id="EVT_FORCED_MARRIAGE", mechanism="social"),
        CausalEdge(source_id="ENT_HEATHCLIFF", target_id="EVT_FORCED_MARRIAGE", mechanism="psychological"),
        CausalEdge(source_id="EVT_FORCED_MARRIAGE", target_id="EVT_EDGAR_DIES", mechanism="social"),
        CausalEdge(source_id="EVT_EDGAR_DIES", target_id="EVT_LINTON_DIES", mechanism="social"),
        CausalEdge(source_id="EVT_LINTON_DIES", target_id="EVT_CATHY_HARETON_RECONCILE", mechanism="social"),
        CausalEdge(source_id="EVT_CATHY_HARETON_RECONCILE", target_id="EVT_HEATHCLIFF_DECLINES", mechanism="psychological"),
        CausalEdge(source_id="ENT_CATHERINE", target_id="EVT_HEATHCLIFF_DECLINES", mechanism="psychological"),
        CausalEdge(source_id="EVT_HEATHCLIFF_DECLINES", target_id="EVT_HEATHCLIFF_DIES", mechanism="physical"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_HEATHCLIFF", target_entity_id="ENT_CATHERINE", affinity=1.0, friction=0.6, power_dynamic=0.0, inertia=0.95),
        RelationshipEdge(source_entity_id="ENT_CATHERINE", target_entity_id="ENT_HEATHCLIFF", affinity=1.0, friction=0.5, power_dynamic=0.0, inertia=0.95),
        RelationshipEdge(source_entity_id="ENT_CATHERINE", target_entity_id="ENT_EDGAR", affinity=0.5, friction=0.3, power_dynamic=-0.2, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_EDGAR", target_entity_id="ENT_CATHERINE", affinity=0.85, friction=0.4, power_dynamic=0.2, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_HEATHCLIFF", target_entity_id="ENT_HINDLEY", affinity=-0.9, friction=0.95, power_dynamic=0.3, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_HINDLEY", target_entity_id="ENT_HEATHCLIFF", affinity=-0.9, friction=0.9, power_dynamic=-0.3, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_HEATHCLIFF", target_entity_id="ENT_EDGAR", affinity=-0.8, friction=0.9, power_dynamic=0.4, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_HEATHCLIFF", target_entity_id="ENT_ISABELLA", affinity=-0.6, friction=0.7, power_dynamic=0.8, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_CATHY", target_entity_id="ENT_HARETON", affinity=0.8, friction=0.3, power_dynamic=0.1, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_HARETON", target_entity_id="ENT_CATHY", affinity=0.8, friction=0.3, power_dynamic=-0.1, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_HEATHCLIFF", target_entity_id="ENT_HARETON", affinity=-0.3, friction=0.6, power_dynamic=0.8, inertia=0.5),
    ],
)
