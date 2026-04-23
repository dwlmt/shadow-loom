from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
    Affordance, Belief,
)

# =============================================================================
# FRANKENSTEIN — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
        "LOC_GENEVA": Location(
            id="LOC_GENEVA",
            name="Geneva (Frankenstein Family Home)",
            connected_locations=["LOC_INGOLSTADT", "LOC_MER_DE_GLACE", "LOC_ALPS"],
            ambient_states={
                "domesticity": AmbientVector(value=0.7, volatility=0.4),
                "grief": AmbientVector(value=0.6, volatility=0.5),
            },
        ),
        "LOC_INGOLSTADT": Location(
            id="LOC_INGOLSTADT",
            name="University of Ingolstadt (Laboratory)",
            connected_locations=["LOC_GENEVA"],
            ambient_states={
                "obsession": AmbientVector(value=0.9, volatility=0.3),
            },
        ),
        "LOC_MER_DE_GLACE": Location(
            id="LOC_MER_DE_GLACE",
            name="Mer de Glace (Alpine Glacier)",
            connected_locations=["LOC_GENEVA"],
            ambient_states={
                "desolation": AmbientVector(value=0.8, volatility=0.2),
                "sublimity": AmbientVector(value=0.9, volatility=0.1),
            },
        ),
        "LOC_ALPS": Location(
            id="LOC_ALPS",
            name="Alpine Countryside",
            connected_locations=["LOC_GENEVA", "LOC_HOVEL"],
            ambient_states={},
        ),
        "LOC_HOVEL": Location(
            id="LOC_HOVEL",
            name="Hovel by the Cottage (Creature's Hiding Place)",
            connected_locations=["LOC_ALPS"],
            ambient_states={
                "isolation": AmbientVector(value=0.8, volatility=0.3),
            },
        ),
        "LOC_ORKNEY": Location(
            id="LOC_ORKNEY",
            name="Orkney Laboratory",
            connected_locations=["LOC_BRITAIN"],
            ambient_states={
                "isolation": AmbientVector(value=0.9, volatility=0.2),
                "dread": AmbientVector(value=0.8, volatility=0.4),
            },
        ),
        "LOC_BRITAIN": Location(
            id="LOC_BRITAIN",
            name="Britain (Victor & Clerval's Travels)",
            connected_locations=["LOC_GENEVA", "LOC_ORKNEY"],
            ambient_states={},
        ),
        "LOC_ARCTIC": Location(
            id="LOC_ARCTIC",
            name="The Arctic (Walton's Ship)",
            connected_locations=[],
            ambient_states={
                "desolation": AmbientVector(value=0.95, volatility=0.1),
                "death": AmbientVector(value=0.8, volatility=0.3),
            },
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_BODY_PARTS": NarrativeObject(
            id="OBJ_BODY_PARTS",
            name="Human Body Parts (Charnel Houses & Graves)",
            location_id="LOC_INGOLSTADT",
            owner_id=None,
            properties={"state": "assembled"},
            affordances=[
                Affordance(action="create_life", target_type="Entity"),
            ],
        ),
        "OBJ_PARADISE_LOST": NarrativeObject(
            id="OBJ_PARADISE_LOST",
            name="Paradise Lost (by Milton)",
            location_id=None,
            owner_id=None,
            properties={"state": "read_by_creature"},
            affordances=[
                Affordance(action="educate", target_type="Entity"),
            ],
        ),
        "OBJ_VICTORS_PAPERS": NarrativeObject(
            id="OBJ_VICTORS_PAPERS",
            name="Victor's Papers (in Creature's Clothes)",
            location_id=None,
            owner_id=None,
            properties={"state": "read_by_creature"},
            affordances=[
                Affordance(action="reveal_origin", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_VICTOR": Entity(
            id="ENT_VICTOR",
            name="Victor Frankenstein",
            location_id="LOC_ARCTIC",
            status="dead",
            traits={
                "obsession": TraitVector(value=0.9, inertia=0.8),
                "intellect": TraitVector(value=0.9, inertia=0.85),
                "guilt": TraitVector(value=0.85, inertia=0.6),
                "cowardice": TraitVector(value=0.7, inertia=0.5),
                "ambition": TraitVector(value=0.85, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_CREATURE", perceived_state="My creation is a monster that must be destroyed", confidence=0.95, inertia=0.9),
                Belief(target_id="ENT_CREATURE", perceived_state="The creature will kill ME on my wedding night", confidence=0.9, inertia=0.7),
                Belief(target_id="ENT_JUSTINE", perceived_state="Justine is innocent but I cannot reveal why without exposing my creation", confidence=1.0, inertia=0.9),
            ],
        ),
        "ENT_CREATURE": Entity(
            id="ENT_CREATURE",
            name="The Creature",
            location_id="LOC_ARCTIC",
            status="healthy",
            traits={
                "intelligence": TraitVector(value=0.8, inertia=0.7),
                "loneliness": TraitVector(value=0.95, inertia=0.8),
                "rage": TraitVector(value=0.85, inertia=0.6),
                "eloquence": TraitVector(value=0.8, inertia=0.7),
                "vengefulness": TraitVector(value=0.85, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_VICTOR", perceived_state="My creator abandoned me — misery made me a fiend", confidence=1.0, inertia=0.9),
                Belief(target_id="ENT_VICTOR", perceived_state="Victor will create a female companion for me", confidence=0.8, inertia=0.5),
            ],
        ),
        "ENT_ELIZABETH": Entity(
            id="ENT_ELIZABETH",
            name="Elizabeth Lavenza",
            location_id="LOC_GENEVA",
            status="dead",
            traits={
                "devotion": TraitVector(value=0.85, inertia=0.8),
                "gentleness": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_VICTOR", perceived_state="Our wedding will bring happiness and end Victor's melancholy", confidence=0.8, inertia=0.6),
            ],
        ),
        "ENT_HENRY": Entity(
            id="ENT_HENRY",
            name="Henry Clerval",
            location_id="LOC_BRITAIN",
            status="dead",
            traits={
                "friendship": TraitVector(value=0.9, inertia=0.8),
                "optimism": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_WILLIAM": Entity(
            id="ENT_WILLIAM",
            name="William Frankenstein",
            location_id="LOC_GENEVA",
            status="dead",
            traits={
                "innocence": TraitVector(value=0.9, inertia=0.9),
            },
        ),
        "ENT_JUSTINE": Entity(
            id="ENT_JUSTINE",
            name="Justine Moritz",
            location_id="LOC_GENEVA",
            status="dead",
            traits={
                "innocence": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_WILLIAM", perceived_state="I am innocent of William's murder", confidence=1.0, inertia=0.9),
            ],
        ),
        "ENT_ALPHONSE": Entity(
            id="ENT_ALPHONSE",
            name="Alphonse Frankenstein",
            location_id="LOC_GENEVA",
            status="dead",
            traits={
                "paternal_love": TraitVector(value=0.85, inertia=0.8),
            },
        ),
        "ENT_WALTON": Entity(
            id="ENT_WALTON",
            name="Captain Robert Walton",
            location_id="LOC_ARCTIC",
            status="healthy",
            traits={
                "ambition": TraitVector(value=0.8, inertia=0.6),
                "compassion": TraitVector(value=0.7, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_VICTOR", perceived_state="This expedition will bring glory and scientific discovery", confidence=0.8, inertia=0.5),
            ],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_CREATION", fabula_time=1, syuzhet_index=1, event_type="choice", actor_id="ENT_VICTOR", description="Victor assembles body parts and creates a large grotesque creature at the University of Ingolstadt. Repelled, he flees."),
        EventNode(id="EVT_CREATURE_FLEES", fabula_time=2, syuzhet_index=2, event_type="outcome", actor_id="ENT_CREATURE", description="The creature runs away, discovers fire, and hides in a hovel beside a cottage, learning language and reading."),
        EventNode(id="EVT_CREATURE_LEARNS_ORIGIN", fabula_time=3, syuzhet_index=3, event_type="revelation", actor_id="ENT_CREATURE", description="The creature reads Victor's papers and Paradise Lost, learning the truth of his origin and creator's identity."),
        EventNode(id="EVT_REJECTED_BY_FAMILY", fabula_time=4, syuzhet_index=4, event_type="outcome", actor_id=None, description="The creature reveals himself to the blind father, who is kind, but the rest of the family chase him away in horror."),
        EventNode(id="EVT_GIRL_RESCUE_SHOOTING", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_id=None, description="The creature saves a drowning girl but is shot by her father, who misunderstands."),
        EventNode(id="EVT_WILLIAM_MURDERED", fabula_time=6, syuzhet_index=6, event_type="choice", actor_id="ENT_CREATURE", description="Embittered, the creature travels to Geneva and kills Victor's younger brother William, then frames Justine."),
        EventNode(id="EVT_JUSTINE_EXECUTED", fabula_time=7, syuzhet_index=7, event_type="outcome", actor_id=None, description="Justine is tried and executed for William's murder. Victor suspects the creature but does not intervene."),
        EventNode(id="EVT_CREATURE_DEMANDS_MATE", fabula_time=8, syuzhet_index=8, event_type="choice", actor_id="ENT_CREATURE", description="On Mer de Glace, the creature tells Victor his story and asks for a female companion. Victor agrees."),
        EventNode(id="EVT_FEMALE_DESTROYED", fabula_time=9, syuzhet_index=9, event_type="choice", actor_id="ENT_VICTOR", description="In Orkney, Victor fears the consequences and destroys the incomplete female creature."),
        EventNode(id="EVT_CREATURE_WARNING", fabula_time=10, syuzhet_index=10, event_type="choice", actor_id="ENT_CREATURE", description="The creature warns Victor he will be with him on his wedding night."),
        EventNode(id="EVT_HENRY_MURDERED", fabula_time=11, syuzhet_index=11, event_type="choice", actor_id="ENT_CREATURE", description="The creature murders Henry Clerval in revenge for Victor destroying the female."),
        EventNode(id="EVT_VICTOR_BREAKDOWN", fabula_time=12, syuzhet_index=12, event_type="outcome", actor_id="ENT_VICTOR", description="Victor suffers a mental breakdown after Henry's death, then returns to Geneva."),
        EventNode(id="EVT_ELIZABETH_MURDERED", fabula_time=13, syuzhet_index=13, event_type="choice", actor_id="ENT_CREATURE", description="Victor marries Elizabeth. Fulfilling his threat, the creature murders her on the wedding night."),
        EventNode(id="EVT_ALPHONSE_DIES", fabula_time=14, syuzhet_index=14, event_type="outcome", actor_id=None, description="Victor's father Alphonse dies of grief after Elizabeth's murder."),
        EventNode(id="EVT_ARCTIC_PURSUIT", fabula_time=15, syuzhet_index=15, event_type="choice", actor_id="ENT_VICTOR", description="Victor vows revenge and pursues the creature across Europe to the Arctic."),
        EventNode(id="EVT_WALTON_RESCUE", fabula_time=16, syuzhet_index=16, event_type="outcome", actor_id="ENT_WALTON", description="Near death, Victor is rescued by Captain Walton's Arctic expedition. He recounts his story."),
        EventNode(id="EVT_VICTOR_DIES", fabula_time=17, syuzhet_index=17, event_type="outcome", actor_id=None, description="The crew decides to turn back. Victor, too weak to continue the chase, dies aboard the ship."),
        EventNode(id="EVT_CREATURE_MOURNS", fabula_time=18, syuzhet_index=18, event_type="outcome", actor_id="ENT_CREATURE", description="The creature boards the ship, mourns Victor, tells Walton he plans to burn himself on a pyre, and departs into the Arctic."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="ENT_VICTOR", target_id="EVT_CREATION", mechanism="physical"),
        CausalEdge(source_id="OBJ_BODY_PARTS", target_id="EVT_CREATION", mechanism="physical"),
        CausalEdge(source_id="EVT_CREATION", target_id="EVT_CREATURE_FLEES", mechanism="psychological"),
        CausalEdge(source_id="OBJ_VICTORS_PAPERS", target_id="EVT_CREATURE_LEARNS_ORIGIN", mechanism="epistemic"),
        CausalEdge(source_id="OBJ_PARADISE_LOST", target_id="EVT_CREATURE_LEARNS_ORIGIN", mechanism="epistemic"),
        CausalEdge(source_id="EVT_CREATURE_LEARNS_ORIGIN", target_id="EVT_REJECTED_BY_FAMILY", mechanism="social"),
        CausalEdge(source_id="EVT_REJECTED_BY_FAMILY", target_id="EVT_GIRL_RESCUE_SHOOTING", mechanism="social"),
        CausalEdge(source_id="EVT_GIRL_RESCUE_SHOOTING", target_id="EVT_WILLIAM_MURDERED", mechanism="psychological"),
        CausalEdge(source_id="ENT_CREATURE", target_id="EVT_WILLIAM_MURDERED", mechanism="physical"),
        CausalEdge(source_id="EVT_WILLIAM_MURDERED", target_id="EVT_JUSTINE_EXECUTED", mechanism="epistemic"),
        CausalEdge(source_id="EVT_JUSTINE_EXECUTED", target_id="EVT_CREATURE_DEMANDS_MATE", mechanism="psychological"),
        CausalEdge(source_id="ENT_CREATURE", target_id="EVT_CREATURE_DEMANDS_MATE", mechanism="psychological"),
        CausalEdge(source_id="EVT_CREATURE_DEMANDS_MATE", target_id="EVT_FEMALE_DESTROYED", mechanism="psychological"),
        CausalEdge(source_id="EVT_FEMALE_DESTROYED", target_id="EVT_CREATURE_WARNING", mechanism="psychological"),
        CausalEdge(source_id="EVT_FEMALE_DESTROYED", target_id="EVT_HENRY_MURDERED", mechanism="psychological"),
        CausalEdge(source_id="ENT_CREATURE", target_id="EVT_HENRY_MURDERED", mechanism="physical"),
        CausalEdge(source_id="EVT_HENRY_MURDERED", target_id="EVT_VICTOR_BREAKDOWN", mechanism="psychological"),
        CausalEdge(source_id="EVT_CREATURE_WARNING", target_id="EVT_ELIZABETH_MURDERED", mechanism="physical"),
        CausalEdge(source_id="ENT_CREATURE", target_id="EVT_ELIZABETH_MURDERED", mechanism="physical"),
        CausalEdge(source_id="EVT_ELIZABETH_MURDERED", target_id="EVT_ALPHONSE_DIES", mechanism="psychological"),
        CausalEdge(source_id="EVT_ALPHONSE_DIES", target_id="EVT_ARCTIC_PURSUIT", mechanism="psychological"),
        CausalEdge(source_id="EVT_ARCTIC_PURSUIT", target_id="EVT_WALTON_RESCUE", mechanism="physical"),
        CausalEdge(source_id="ENT_WALTON", target_id="EVT_WALTON_RESCUE", mechanism="social"),
        CausalEdge(source_id="EVT_WALTON_RESCUE", target_id="EVT_VICTOR_DIES", mechanism="physical"),
        CausalEdge(source_id="EVT_VICTOR_DIES", target_id="EVT_CREATURE_MOURNS", mechanism="psychological"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_VICTOR", target_entity_id="ENT_CREATURE", affinity=-0.8, friction=0.95, power_dynamic=0.2, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_CREATURE", target_entity_id="ENT_VICTOR", affinity=-0.5, friction=0.9, power_dynamic=-0.1, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_VICTOR", target_entity_id="ENT_ELIZABETH", affinity=0.8, friction=0.2, power_dynamic=0.1, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_VICTOR", target_entity_id="ENT_HENRY", affinity=0.85, friction=0.1, power_dynamic=0.0, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_CREATURE", target_entity_id="ENT_WILLIAM", affinity=-0.7, friction=0.9, power_dynamic=0.9, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_WALTON", target_entity_id="ENT_VICTOR", affinity=0.6, friction=0.3, power_dynamic=-0.1, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_ALPHONSE", target_entity_id="ENT_VICTOR", affinity=0.9, friction=0.2, power_dynamic=0.3, inertia=0.8),
    ],
)
