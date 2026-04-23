from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
    Affordance, Belief,
)

# =============================================================================
# 1984 — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
        "LOC_LONDON_AIRSTRIP_ONE": Location(
            id="LOC_LONDON_AIRSTRIP_ONE",
            name="London, Airstrip One (Oceania)",
            connected_locations=["LOC_MINISTRY_OF_TRUTH", "LOC_MINISTRY_OF_LOVE", "LOC_PROLE_QUARTER", "LOC_CHARRINGTON_SHOP", "LOC_VICTORY_MANSIONS"],
            ambient_states={
                "surveillance": AmbientVector(value=0.95, volatility=0.05),
                "oppression": AmbientVector(value=0.95, volatility=0.05),
            },
        ),
        "LOC_VICTORY_MANSIONS": Location(
            id="LOC_VICTORY_MANSIONS",
            name="Victory Mansions (Winston's Flat)",
            connected_locations=["LOC_LONDON_AIRSTRIP_ONE", "LOC_MINISTRY_OF_TRUTH"],
            ambient_states={
                "surveillance": AmbientVector(value=0.9, volatility=0.1),
                "privacy": AmbientVector(value=0.1, volatility=0.1),
            },
        ),
        "LOC_MINISTRY_OF_TRUTH": Location(
            id="LOC_MINISTRY_OF_TRUTH",
            name="Ministry of Truth",
            connected_locations=["LOC_LONDON_AIRSTRIP_ONE", "LOC_VICTORY_MANSIONS"],
            ambient_states={
                "surveillance": AmbientVector(value=1.0, volatility=0.0),
                "propaganda": AmbientVector(value=1.0, volatility=0.0),
            },
        ),
        "LOC_MINISTRY_OF_LOVE": Location(
            id="LOC_MINISTRY_OF_LOVE",
            name="Ministry of Love (Prison & Re-education)",
            connected_locations=["LOC_LONDON_AIRSTRIP_ONE", "LOC_ROOM_101"],
            ambient_states={
                "terror": AmbientVector(value=1.0, volatility=0.0),
                "surveillance": AmbientVector(value=1.0, volatility=0.0),
            },
        ),
        "LOC_ROOM_101": Location(
            id="LOC_ROOM_101",
            name="Room 101",
            connected_locations=["LOC_MINISTRY_OF_LOVE"],
            ambient_states={
                "terror": AmbientVector(value=1.0, volatility=0.0),
            },
            constants=["contains_worst_fear"],
        ),
        "LOC_CHARRINGTON_SHOP": Location(
            id="LOC_CHARRINGTON_SHOP",
            name="Mr Charrington's Antiques Shop (Rented Room)",
            connected_locations=["LOC_PROLE_QUARTER", "LOC_LONDON_AIRSTRIP_ONE"],
            ambient_states={
                "privacy": AmbientVector(value=0.7, volatility=0.8),
                "surveillance": AmbientVector(value=0.9, volatility=0.1),
            },
        ),
        "LOC_PROLE_QUARTER": Location(
            id="LOC_PROLE_QUARTER",
            name="Prole Quarter",
            connected_locations=["LOC_LONDON_AIRSTRIP_ONE", "LOC_CHARRINGTON_SHOP"],
            ambient_states={
                "surveillance": AmbientVector(value=0.4, volatility=0.3),
                "poverty": AmbientVector(value=0.8, volatility=0.1),
            },
        ),
        "LOC_COUNTRYSIDE": Location(
            id="LOC_COUNTRYSIDE",
            name="Countryside (Outside London)",
            connected_locations=["LOC_LONDON_AIRSTRIP_ONE"],
            ambient_states={
                "surveillance": AmbientVector(value=0.3, volatility=0.4),
                "freedom": AmbientVector(value=0.5, volatility=0.5),
            },
        ),
        "LOC_CHESTNUT_TREE_CAFE": Location(
            id="LOC_CHESTNUT_TREE_CAFE",
            name="Chestnut Tree Café",
            connected_locations=["LOC_LONDON_AIRSTRIP_ONE"],
            ambient_states={
                "despair": AmbientVector(value=0.8, volatility=0.2),
            },
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_DIARY": NarrativeObject(
            id="OBJ_DIARY",
            name="Winston's Diary",
            location_id=None,
            owner_id="ENT_WINSTON",
            properties={"content": "criticisms_of_party", "state": "hidden"},
            affordances=[
                Affordance(action="write", target_type="Entity"),
                Affordance(action="incriminate", target_type="Entity"),
            ],
        ),
        "OBJ_GOLDSTEIN_BOOK": NarrativeObject(
            id="OBJ_GOLDSTEIN_BOOK",
            name="The Theory and Practice of Oligarchical Collectivism",
            location_id=None,
            owner_id="ENT_WINSTON",
            properties={"author": "Goldstein (and Party members)", "state": "forbidden"},
            affordances=[
                Affordance(action="read", target_type="Entity"),
                Affordance(action="enlighten", target_type="Entity"),
            ],
        ),
        "OBJ_TELESCREEN": NarrativeObject(
            id="OBJ_TELESCREEN",
            name="Telescreen",
            location_id="LOC_VICTORY_MANSIONS",
            owner_id=None,
            properties={"state": "always_on"},
            affordances=[
                Affordance(action="surveil", target_type="Entity"),
                Affordance(action="broadcast_propaganda", target_type="Entity"),
            ],
        ),
        "OBJ_MEMORY_HOLES": NarrativeObject(
            id="OBJ_MEMORY_HOLES",
            name="Memory Holes",
            location_id="LOC_MINISTRY_OF_TRUTH",
            owner_id=None,
            properties={"state": "operational"},
            affordances=[
                Affordance(action="destroy", target_type="NarrativeObject"),
            ],
        ),
        "OBJ_RAT_CAGE": NarrativeObject(
            id="OBJ_RAT_CAGE",
            name="Rat Torture Device",
            location_id="LOC_ROOM_101",
            owner_id=None,
            properties={"state": "ready"},
            affordances=[
                Affordance(action="torture", target_type="Entity"),
                Affordance(action="break_will", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_WINSTON": Entity(
            id="ENT_WINSTON",
            name="Winston Smith",
            location_id="LOC_CHESTNUT_TREE_CAFE",
            status="healthy",
            traits={
                "rebelliousness": TraitVector(value=0.1, inertia=0.2),
                "intellect": TraitVector(value=0.8, inertia=0.8),
                "despair": TraitVector(value=0.9, inertia=0.7),
                "love_of_truth": TraitVector(value=0.1, inertia=0.1),
                "obedience": TraitVector(value=0.95, inertia=0.9),
            },
            beliefs=[
                Belief(target_id="ENT_BIG_BROTHER", perceived_state="I love Big Brother", confidence=1.0, inertia=1.0),
                Belief(target_id="ENT_OBRIEN", perceived_state="O'Brien is secretly a member of the Brotherhood resistance", confidence=0.7, inertia=0.6),
                Belief(target_id="ENT_CHARRINGTON", perceived_state="Charrington is a harmless old prole shopkeeper who sympathises with the past", confidence=0.85, inertia=0.7),
                Belief(target_id="LOC_CHARRINGTON_SHOP", perceived_state="The rented room is a safe refuge free from telescreens", confidence=0.85, inertia=0.7),
                Belief(target_id="OBJ_GOLDSTEIN_BOOK", perceived_state="This book reveals the genuine truth about how the Party maintains power", confidence=0.7, inertia=0.5),
            ],
        ),
        "ENT_JULIA": Entity(
            id="ENT_JULIA",
            name="Julia",
            location_id="LOC_LONDON_AIRSTRIP_ONE",
            status="healthy",
            traits={
                "rebelliousness": TraitVector(value=0.1, inertia=0.2),
                "pragmatism": TraitVector(value=0.8, inertia=0.7),
                "sensuality": TraitVector(value=0.7, inertia=0.5),
                "political_apathy": TraitVector(value=0.7, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_WINSTON", perceived_state="We have both betrayed each other", confidence=1.0, inertia=0.9),
                Belief(target_id="ENT_BIG_BROTHER", perceived_state="The Party is rotten but can only be resisted personally, not overthrown", confidence=0.8, inertia=0.7),
            ],
        ),
        "ENT_OBRIEN": Entity(
            id="ENT_OBRIEN",
            name="O'Brien",
            location_id="LOC_MINISTRY_OF_LOVE",
            status="healthy",
            traits={
                "intelligence": TraitVector(value=0.95, inertia=0.9),
                "cruelty": TraitVector(value=0.9, inertia=0.9),
                "deception": TraitVector(value=0.95, inertia=0.9),
                "devotion_to_party": TraitVector(value=1.0, inertia=1.0),
            },
            beliefs=[
                Belief(target_id="ENT_BIG_BROTHER", perceived_state="Power is the purpose — power for its own sake", confidence=1.0, inertia=1.0),
            ],
        ),
        "ENT_CHARRINGTON": Entity(
            id="ENT_CHARRINGTON",
            name="Mr Charrington",
            location_id="LOC_CHARRINGTON_SHOP",
            status="healthy",
            traits={
                "deception": TraitVector(value=0.95, inertia=0.9),
                "devotion_to_party": TraitVector(value=1.0, inertia=1.0),
            },
            constants=["thought_police_agent"],
        ),
        "ENT_BIG_BROTHER": Entity(
            id="ENT_BIG_BROTHER",
            name="Big Brother",
            location_id="LOC_LONDON_AIRSTRIP_ONE",
            status="healthy",
            traits={
                "authority": TraitVector(value=1.0, inertia=1.0),
                "cult_of_personality": TraitVector(value=1.0, inertia=1.0),
            },
            constants=["omnipresent", "possibly_nonexistent"],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_WINSTON_BUYS_DIARY", fabula_time=1, syuzhet_index=1, event_type="choice", actor_id="ENT_WINSTON", description="Winston buys a diary from Mr Charrington's shop and begins writing criticisms of the Party."),
        EventNode(id="EVT_JULIA_LOVE_NOTE", fabula_time=2, syuzhet_index=2, event_type="choice", actor_id="ENT_JULIA", description="Julia discreetly passes Winston a love note, initiating their secret affair."),
        EventNode(id="EVT_AFFAIR_BEGINS", fabula_time=3, syuzhet_index=3, event_type="choice", actor_id="ENT_WINSTON", description="Winston and Julia begin a secret affair, meeting first in the countryside, then in the rented room above Charrington's shop."),
        EventNode(id="EVT_OBRIEN_INVITATION", fabula_time=4, syuzhet_index=4, event_type="choice", actor_id="ENT_OBRIEN", description="O'Brien invites Winston to his flat, pretending to be a Brotherhood member, and gives him Goldstein's book."),
        EventNode(id="EVT_HATE_WEEK_ENEMY_SWITCH", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_id=None, description="During Hate Week, Oceania's enemy switches from Eurasia to Eastasia; records must be rewritten."),
        EventNode(id="EVT_GOLDSTEIN_BOOK_READ", fabula_time=6, syuzhet_index=6, event_type="choice", actor_id="ENT_WINSTON", description="Winston and Julia read parts of Goldstein's book, learning how the Party maintains power through perpetual war."),
        EventNode(id="EVT_CAPTURED", fabula_time=7, syuzhet_index=7, event_type="outcome", actor_id="ENT_CHARRINGTON", description="Winston and Julia are captured when Mr Charrington is revealed as a Thought Police agent."),
        EventNode(id="EVT_OBRIEN_REVEALS_TRUTH", fabula_time=8, syuzhet_index=8, event_type="revelation", actor_id="ENT_OBRIEN", description="O'Brien reveals himself as Thought Police, that the Brotherhood may not exist, and that power is the Party's sole purpose."),
        EventNode(id="EVT_TORTURE_REEDUCATION", fabula_time=9, syuzhet_index=9, event_type="outcome", actor_id="ENT_OBRIEN", description="Over months, Winston is starved, tortured, and re-educated to align his beliefs with the Party."),
        EventNode(id="EVT_ROOM_101", fabula_time=10, syuzhet_index=10, event_type="outcome", actor_id="ENT_OBRIEN", description="In Room 101, Winston is confronted with rat torture and begs for it to be done to Julia instead — his final betrayal."),
        EventNode(id="EVT_WINSTON_RELEASED", fabula_time=11, syuzhet_index=11, event_type="outcome", actor_id=None, description="Winston is released into public life, a broken man."),
        EventNode(id="EVT_ENCOUNTER_JULIA", fabula_time=12, syuzhet_index=12, event_type="outcome", actor_id=None, description="Winston encounters Julia; both admit they betrayed each other and are no longer in love."),
        EventNode(id="EVT_LOVES_BIG_BROTHER", fabula_time=13, syuzhet_index=13, event_type="outcome", actor_id="ENT_WINSTON", description="Winston accepts that he loves Big Brother. The Party's re-education is complete."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="ENT_WINSTON", target_id="EVT_WINSTON_BUYS_DIARY", mechanism="psychological"),
        CausalEdge(source_id="ENT_JULIA", target_id="EVT_JULIA_LOVE_NOTE", mechanism="psychological"),
        CausalEdge(source_id="EVT_JULIA_LOVE_NOTE", target_id="EVT_AFFAIR_BEGINS", mechanism="social"),
        CausalEdge(source_id="LOC_CHARRINGTON_SHOP", target_id="EVT_AFFAIR_BEGINS", mechanism="physical"),
        CausalEdge(source_id="ENT_OBRIEN", target_id="EVT_OBRIEN_INVITATION", mechanism="epistemic"),
        CausalEdge(source_id="EVT_OBRIEN_INVITATION", target_id="EVT_GOLDSTEIN_BOOK_READ", mechanism="epistemic"),
        CausalEdge(source_id="ENT_CHARRINGTON", target_id="EVT_CAPTURED", mechanism="epistemic"),
        CausalEdge(source_id="EVT_AFFAIR_BEGINS", target_id="EVT_CAPTURED", mechanism="physical"),
        CausalEdge(source_id="EVT_CAPTURED", target_id="EVT_OBRIEN_REVEALS_TRUTH", mechanism="epistemic"),
        CausalEdge(source_id="EVT_OBRIEN_REVEALS_TRUTH", target_id="EVT_TORTURE_REEDUCATION", mechanism="physical"),
        CausalEdge(source_id="EVT_TORTURE_REEDUCATION", target_id="EVT_ROOM_101", mechanism="psychological"),
        CausalEdge(source_id="OBJ_RAT_CAGE", target_id="EVT_ROOM_101", mechanism="physical"),
        CausalEdge(source_id="EVT_ROOM_101", target_id="EVT_WINSTON_RELEASED", mechanism="psychological"),
        CausalEdge(source_id="EVT_ROOM_101", target_id="EVT_ENCOUNTER_JULIA", mechanism="psychological"),
        CausalEdge(source_id="EVT_ENCOUNTER_JULIA", target_id="EVT_LOVES_BIG_BROTHER", mechanism="psychological"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_WINSTON", target_entity_id="ENT_JULIA", affinity=0.0, friction=0.3, power_dynamic=0.0, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_WINSTON", target_entity_id="ENT_OBRIEN", affinity=-0.8, friction=0.9, power_dynamic=-1.0, inertia=0.9),
        RelationshipEdge(source_entity_id="ENT_WINSTON", target_entity_id="ENT_BIG_BROTHER", affinity=1.0, friction=0.0, power_dynamic=-1.0, inertia=1.0),
        RelationshipEdge(source_entity_id="ENT_OBRIEN", target_entity_id="ENT_WINSTON", affinity=-0.3, friction=0.8, power_dynamic=1.0, inertia=0.9),
        RelationshipEdge(source_entity_id="ENT_CHARRINGTON", target_entity_id="ENT_WINSTON", affinity=-0.5, friction=0.2, power_dynamic=0.8, inertia=0.8),
    ],
)
