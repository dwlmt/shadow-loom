from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
    Affordance, Belief,
)

# =============================================================================
# MACBETH — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
        "LOC_HEATH": Location(
            id="LOC_HEATH",
            name="The Heath",
            connected_locations=["LOC_INVERNESS_CASTLE", "LOC_BATTLEFIELD"],
            ambient_states={
                "visibility": AmbientVector(value=0.3, volatility=0.6),
                "supernatural": AmbientVector(value=0.9, volatility=0.4),
            },
            constants=["desolate"],
        ),
        "LOC_BATTLEFIELD": Location(
            id="LOC_BATTLEFIELD",
            name="Battlefield (Scotland)",
            connected_locations=["LOC_HEATH", "LOC_INVERNESS_CASTLE"],
            ambient_states={
                "danger": AmbientVector(value=0.9, volatility=0.3),
            },
        ),
        "LOC_INVERNESS_CASTLE": Location(
            id="LOC_INVERNESS_CASTLE",
            name="Macbeth's Castle at Inverness",
            connected_locations=["LOC_HEATH", "LOC_DUNSINANE_CASTLE", "LOC_BATTLEFIELD"],
            ambient_states={
                "tension": AmbientVector(value=0.7, volatility=0.5),
                "visibility": AmbientVector(value=0.5, volatility=0.3),
            },
        ),
        "LOC_DUNSINANE_CASTLE": Location(
            id="LOC_DUNSINANE_CASTLE",
            name="Dunsinane Castle (Royal Palace)",
            connected_locations=["LOC_INVERNESS_CASTLE", "LOC_BIRNAM_WOOD", "LOC_MACDUFF_CASTLE"],
            ambient_states={
                "tension": AmbientVector(value=0.8, volatility=0.4),
                "paranoia": AmbientVector(value=0.9, volatility=0.3),
            },
        ),
        "LOC_BIRNAM_WOOD": Location(
            id="LOC_BIRNAM_WOOD",
            name="Birnam Wood",
            connected_locations=["LOC_DUNSINANE_CASTLE"],
            ambient_states={
                "concealment": AmbientVector(value=0.8, volatility=0.2),
            },
        ),
        "LOC_MACDUFF_CASTLE": Location(
            id="LOC_MACDUFF_CASTLE",
            name="Macduff's Castle at Fife",
            connected_locations=["LOC_DUNSINANE_CASTLE", "LOC_ENGLAND"],
            ambient_states={
                "safety": AmbientVector(value=0.6, volatility=0.7),
            },
        ),
        "LOC_ENGLAND": Location(
            id="LOC_ENGLAND",
            name="England (King Edward's Court)",
            connected_locations=["LOC_MACDUFF_CASTLE", "LOC_BIRNAM_WOOD"],
            ambient_states={
                "safety": AmbientVector(value=0.9, volatility=0.1),
            },
        ),
        "LOC_WITCHES_CAVERN": Location(
            id="LOC_WITCHES_CAVERN",
            name="The Witches' Cavern",
            connected_locations=["LOC_HEATH"],
            ambient_states={
                "supernatural": AmbientVector(value=1.0, volatility=0.2),
            },
            constants=["supernatural"],
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_DAGGERS": NarrativeObject(
            id="OBJ_DAGGERS",
            name="Bloody Daggers",
            location_id="LOC_INVERNESS_CASTLE",
            owner_id=None,
            properties={"state": "planted_on_servants"},
            affordances=[
                Affordance(action="kill", target_type="Entity"),
                Affordance(action="frame", target_type="Entity"),
            ],
        ),
        "OBJ_CROWN": NarrativeObject(
            id="OBJ_CROWN",
            name="Crown of Scotland",
            location_id="LOC_DUNSINANE_CASTLE",
            owner_id="ENT_MACBETH",
            properties={"state": "contested"},
            affordances=[
                Affordance(action="legitimize", target_type="Entity"),
            ],
        ),
        "OBJ_LETTER": NarrativeObject(
            id="OBJ_LETTER",
            name="Macbeth's Letter to Lady Macbeth",
            location_id="LOC_INVERNESS_CASTLE",
            owner_id="ENT_LADY_MACBETH",
            properties={"content": "witches_prophecy"},
            affordances=[
                Affordance(action="read", target_type="Entity"),
                Affordance(action="inform", target_type="Entity"),
            ],
        ),
        "OBJ_APPARITIONS": NarrativeObject(
            id="OBJ_APPARITIONS",
            name="Witches' Apparitions",
            location_id="LOC_WITCHES_CAVERN",
            owner_id=None,
            properties={"state": "prophetic"},
            affordances=[
                Affordance(action="prophesy", target_type="Entity"),
                Affordance(action="deceive", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_MACBETH": Entity(
            id="ENT_MACBETH",
            name="Macbeth",
            location_id="LOC_DUNSINANE_CASTLE",
            status="dead",
            traits={
                "ambition": TraitVector(value=0.95, inertia=0.8),
                "courage": TraitVector(value=0.85, inertia=0.7),
                "guilt": TraitVector(value=0.7, inertia=0.4),
                "paranoia": TraitVector(value=0.9, inertia=0.6),
                "cruelty": TraitVector(value=0.8, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="OBJ_APPARITIONS", perceived_state="I am invincible — no man born of woman can kill me", confidence=0.95, inertia=0.9),
                Belief(target_id="LOC_BIRNAM_WOOD", perceived_state="Birnam Wood cannot move to Dunsinane", confidence=0.95, inertia=0.9),
            ],
        ),
        "ENT_LADY_MACBETH": Entity(
            id="ENT_LADY_MACBETH",
            name="Lady Macbeth",
            location_id="LOC_DUNSINANE_CASTLE",
            status="dead",
            traits={
                "ambition": TraitVector(value=0.95, inertia=0.7),
                "ruthlessness": TraitVector(value=0.9, inertia=0.5),
                "guilt": TraitVector(value=0.85, inertia=0.3),
                "resolve": TraitVector(value=0.6, inertia=0.4),
            },
            beliefs=[
                Belief(target_id="ENT_MACBETH", perceived_state="Macbeth needs my strength to act", confidence=0.8, inertia=0.6),
                Belief(target_id="EVT_DUNCAN_MURDER", perceived_state="We can wash away the guilt and none will suspect us", confidence=0.8, inertia=0.3),
            ],
        ),
        "ENT_DUNCAN": Entity(
            id="ENT_DUNCAN",
            name="King Duncan",
            location_id="LOC_INVERNESS_CASTLE",
            status="dead",
            traits={
                "trust": TraitVector(value=0.9, inertia=0.8),
                "benevolence": TraitVector(value=0.85, inertia=0.9),
            },
            beliefs=[
                Belief(target_id="ENT_MACBETH", perceived_state="Macbeth is my loyal kinsman and brave hero", confidence=0.95, inertia=0.9),
            ],
        ),
        "ENT_BANQUO": Entity(
            id="ENT_BANQUO",
            name="Banquo",
            location_id="LOC_INVERNESS_CASTLE",
            status="dead",
            traits={
                "loyalty": TraitVector(value=0.85, inertia=0.8),
                "suspicion": TraitVector(value=0.7, inertia=0.5),
                "courage": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_MACBETH", perceived_state="Macbeth may have murdered Duncan to fulfill the witches' prophecy", confidence=0.7, inertia=0.5),
            ],
        ),
        "ENT_MACDUFF": Entity(
            id="ENT_MACDUFF",
            name="Macduff (Thane of Fife)",
            location_id="LOC_DUNSINANE_CASTLE",
            status="healthy",
            traits={
                "loyalty": TraitVector(value=0.9, inertia=0.9),
                "courage": TraitVector(value=0.9, inertia=0.8),
                "grief": TraitVector(value=0.85, inertia=0.4),
                "vengefulness": TraitVector(value=0.9, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="LOC_MACDUFF_CASTLE", perceived_state="My family is safe at home in Fife", confidence=0.8, inertia=0.6),
            ],
            constants=["caesarean_birth"],
        ),
        "ENT_MALCOLM": Entity(
            id="ENT_MALCOLM",
            name="Prince Malcolm",
            location_id="LOC_DUNSINANE_CASTLE",
            status="healthy",
            traits={
                "caution": TraitVector(value=0.8, inertia=0.7),
                "leadership": TraitVector(value=0.75, inertia=0.6),
                "loyalty": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_MACDUFF", perceived_state="Macduff might be a trap sent by Macbeth — I must test his loyalty", confidence=0.6, inertia=0.4),
            ],
        ),
        "ENT_FLEANCE": Entity(
            id="ENT_FLEANCE",
            name="Fleance",
            location_id="LOC_ENGLAND",
            status="healthy",
            traits={
                "innocence": TraitVector(value=0.9, inertia=0.8),
            },
        ),
        "ENT_WITCHES": Entity(
            id="ENT_WITCHES",
            name="The Three Witches",
            location_id="LOC_WITCHES_CAVERN",
            status="healthy",
            traits={
                "malice": TraitVector(value=0.9, inertia=1.0),
                "deception": TraitVector(value=0.95, inertia=1.0),
            },
            constants=["supernatural"],
        ),
        "ENT_LENNOX": Entity(
            id="ENT_LENNOX",
            name="Lennox",
            location_id="LOC_DUNSINANE_CASTLE",
            status="healthy",
            traits={
                "suspicion": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_MACBETH", perceived_state="Macbeth is a murdering tyrant who killed Duncan", confidence=0.8, inertia=0.6),
            ],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_REBELLION_DEFEATED", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_id="ENT_MACBETH", description="Macbeth and Banquo defeat the rebellion led by the traitorous Thane of Cawdor."),
        EventNode(id="EVT_WITCHES_PROPHECY_1", fabula_time=2, syuzhet_index=2, event_type="revelation", actor_id="ENT_WITCHES", description="The three witches prophesy Macbeth will be Thane of Cawdor and King, and Banquo will father kings."),
        EventNode(id="EVT_CAWDOR_TITLE", fabula_time=3, syuzhet_index=3, event_type="outcome", actor_id=None, description="Macbeth receives the title Thane of Cawdor, fulfilling the first prophecy."),
        EventNode(id="EVT_LETTER_SENT", fabula_time=4, syuzhet_index=4, event_type="choice", actor_id="ENT_MACBETH", description="Macbeth sends a letter about the witches' prophecy to Lady Macbeth."),
        EventNode(id="EVT_LADY_MACBETH_PERSUADES", fabula_time=5, syuzhet_index=5, event_type="choice", actor_id="ENT_LADY_MACBETH", description="Lady Macbeth persuades Macbeth to murder King Duncan that night."),
        EventNode(id="EVT_DUNCAN_MURDER", fabula_time=6, syuzhet_index=6, event_type="choice", actor_id="ENT_MACBETH", description="Macbeth stabs King Duncan to death in his sleep at Inverness Castle."),
        EventNode(id="EVT_SERVANTS_FRAMED", fabula_time=7, syuzhet_index=7, event_type="choice", actor_id="ENT_LADY_MACBETH", description="Lady Macbeth plants the bloody daggers on Duncan's sleeping servants to frame them."),
        EventNode(id="EVT_MACBETH_KILLS_SERVANTS", fabula_time=8, syuzhet_index=8, event_type="choice", actor_id="ENT_MACBETH", description="Macbeth impulsively kills Duncan's servants to prevent them from professing innocence."),
        EventNode(id="EVT_SONS_FLEE", fabula_time=9, syuzhet_index=9, event_type="choice", actor_id="ENT_MALCOLM", description="Duncan's sons Malcolm and Donalbain flee Scotland, making themselves suspects."),
        EventNode(id="EVT_MACBETH_CROWNED", fabula_time=10, syuzhet_index=10, event_type="outcome", actor_id="ENT_MACBETH", description="Macbeth assumes the throne as King of Scotland."),
        EventNode(id="EVT_BANQUO_MURDERED", fabula_time=11, syuzhet_index=11, event_type="choice", actor_id="ENT_MACBETH", description="Macbeth hires murderers who kill Banquo, but Fleance escapes."),
        EventNode(id="EVT_BANQUO_GHOST", fabula_time=12, syuzhet_index=12, event_type="revelation", actor_id=None, description="Banquo's ghost appears at the royal banquet, visible only to Macbeth, causing him to rave."),
        EventNode(id="EVT_WITCHES_PROPHECY_2", fabula_time=13, syuzhet_index=13, event_type="revelation", actor_id="ENT_WITCHES", description="The witches give Macbeth three new prophecies: beware Macduff, none born of woman can harm him, and he is safe until Birnam Wood moves."),
        EventNode(id="EVT_MACDUFF_FAMILY_SLAUGHTERED", fabula_time=14, syuzhet_index=14, event_type="choice", actor_id="ENT_MACBETH", description="Macbeth orders assassins to slaughter Macduff's wife and children at Fife."),
        EventNode(id="EVT_MALCOLM_MACDUFF_ALLIANCE", fabula_time=15, syuzhet_index=15, event_type="choice", actor_id="ENT_MALCOLM", description="Malcolm and Macduff ally in England and raise an army to overthrow Macbeth."),
        EventNode(id="EVT_LADY_MACBETH_SLEEPWALKING", fabula_time=16, syuzhet_index=16, event_type="outcome", actor_id="ENT_LADY_MACBETH", description="Lady Macbeth sleepwalks, trying to wash imaginary bloodstains, confessing guilt."),
        EventNode(id="EVT_BIRNAM_WOOD_MOVES", fabula_time=17, syuzhet_index=17, event_type="outcome", actor_id="ENT_MALCOLM", description="Malcolm's soldiers camouflage themselves with branches from Birnam Wood, fulfilling the prophecy."),
        EventNode(id="EVT_LADY_MACBETH_DEATH", fabula_time=18, syuzhet_index=18, event_type="outcome", actor_id="ENT_LADY_MACBETH", description="Lady Macbeth dies, implied suicide."),
        EventNode(id="EVT_MACBETH_KILLED", fabula_time=19, syuzhet_index=19, event_type="outcome", actor_id="ENT_MACDUFF", description="Macduff, born by Caesarean section, kills Macbeth in single combat, fulfilling the prophecy."),
        EventNode(id="EVT_MALCOLM_CROWNED", fabula_time=20, syuzhet_index=20, event_type="outcome", actor_id="ENT_MALCOLM", description="Malcolm is declared King of Scotland. Order is restored."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="EVT_CAWDOR_TITLE", mechanism="social"),
        CausalEdge(source_id="ENT_WITCHES", target_id="EVT_WITCHES_PROPHECY_1", mechanism="epistemic"),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="EVT_LETTER_SENT", mechanism="psychological"),
        CausalEdge(source_id="EVT_CAWDOR_TITLE", target_id="EVT_LETTER_SENT", mechanism="epistemic"),
        CausalEdge(source_id="OBJ_LETTER", target_id="EVT_LADY_MACBETH_PERSUADES", mechanism="epistemic"),
        CausalEdge(source_id="ENT_LADY_MACBETH", target_id="EVT_LADY_MACBETH_PERSUADES", mechanism="psychological"),
        CausalEdge(source_id="EVT_LADY_MACBETH_PERSUADES", target_id="EVT_DUNCAN_MURDER", mechanism="psychological"),
        CausalEdge(source_id="ENT_MACBETH", target_id="EVT_DUNCAN_MURDER", mechanism="physical"),
        CausalEdge(source_id="OBJ_DAGGERS", target_id="EVT_DUNCAN_MURDER", mechanism="physical"),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="EVT_SERVANTS_FRAMED", mechanism="physical"),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="EVT_MACBETH_KILLS_SERVANTS", mechanism="psychological"),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="EVT_SONS_FLEE", mechanism="psychological"),
        CausalEdge(source_id="EVT_SONS_FLEE", target_id="EVT_MACBETH_CROWNED", mechanism="social"),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="EVT_BANQUO_MURDERED", mechanism="psychological"),
        CausalEdge(source_id="ENT_MACBETH", target_id="EVT_BANQUO_MURDERED", mechanism="physical"),
        CausalEdge(source_id="EVT_BANQUO_MURDERED", target_id="EVT_BANQUO_GHOST", mechanism="psychological"),
        CausalEdge(source_id="ENT_WITCHES", target_id="EVT_WITCHES_PROPHECY_2", mechanism="epistemic"),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_2", target_id="EVT_MACDUFF_FAMILY_SLAUGHTERED", mechanism="psychological"),
        CausalEdge(source_id="EVT_MACDUFF_FAMILY_SLAUGHTERED", target_id="EVT_MALCOLM_MACDUFF_ALLIANCE", mechanism="psychological"),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="EVT_LADY_MACBETH_SLEEPWALKING", mechanism="psychological"),
        CausalEdge(source_id="EVT_LADY_MACBETH_SLEEPWALKING", target_id="EVT_LADY_MACBETH_DEATH", mechanism="psychological"),
        CausalEdge(source_id="EVT_MALCOLM_MACDUFF_ALLIANCE", target_id="EVT_BIRNAM_WOOD_MOVES", mechanism="physical"),
        CausalEdge(source_id="EVT_BIRNAM_WOOD_MOVES", target_id="EVT_MACBETH_KILLED", mechanism="physical"),
        CausalEdge(source_id="ENT_MACDUFF", target_id="EVT_MACBETH_KILLED", mechanism="physical"),
        CausalEdge(source_id="EVT_MACBETH_KILLED", target_id="EVT_MALCOLM_CROWNED", mechanism="social"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_MACBETH", target_entity_id="ENT_LADY_MACBETH", affinity=0.8, friction=0.6, power_dynamic=-0.3, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_MACBETH", target_entity_id="ENT_DUNCAN", affinity=0.3, friction=0.7, power_dynamic=-0.6, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_MACBETH", target_entity_id="ENT_BANQUO", affinity=0.4, friction=0.6, power_dynamic=0.2, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_MACBETH", target_entity_id="ENT_MACDUFF", affinity=-0.8, friction=0.9, power_dynamic=0.3, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_MACDUFF", target_entity_id="ENT_MALCOLM", affinity=0.8, friction=0.2, power_dynamic=-0.3, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_LADY_MACBETH", target_entity_id="ENT_DUNCAN", affinity=-0.5, friction=0.8, power_dynamic=-0.4, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_BANQUO", target_entity_id="ENT_MACBETH", affinity=0.2, friction=0.7, power_dynamic=-0.4, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_MALCOLM", target_entity_id="ENT_MACBETH", affinity=-0.9, friction=0.9, power_dynamic=-0.5, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_MACBETH", target_entity_id="ENT_WITCHES", affinity=0.3, friction=0.6, power_dynamic=-0.5, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_MACBETH", target_entity_id="ENT_FLEANCE", affinity=-0.6, friction=0.7, power_dynamic=0.5, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_LENNOX", target_entity_id="ENT_MACBETH", affinity=-0.5, friction=0.6, power_dynamic=-0.4, inertia=0.4),
    ],
)
