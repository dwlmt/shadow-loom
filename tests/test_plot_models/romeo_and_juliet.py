from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, InformationEdge, RelationshipEdge, TraitVector,
    Affordance, Belief,
)

# =============================================================================
# ROMEO AND JULIET — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
                "LOC_VERONA_STREETS": Location(
            name="Streets of Verona",
            description="Streets of Verona",
            ambient_state={"danger": {"value": 0.7, "volatility": 0.6}, "hostility": {"value": 0.8, "volatility": 0.5}},
        ),
                "LOC_CAPULET_HOUSE": Location(
            name="Capulet House (including Ballroom & Juliet's Chamber)",
            description="Capulet House (including Ballroom & Juliet's Chamber)",
            ambient_state={"opulence": {"value": 0.8, "volatility": 0.1}, "tension": {"value": 0.6, "volatility": 0.5}},
        ),
                "LOC_CAPULET_ORCHARD": Location(
            name="Capulet Orchard (Balcony Scene)",
            description="Capulet Orchard (Balcony Scene)",
            ambient_state={"romance": {"value": 0.9, "volatility": 0.3}, "secrecy": {"value": 0.8, "volatility": 0.4}},
        ),
                "LOC_MONTAGUE_HOUSE": Location(
            name="Montague House",
            description="Montague House",
            ambient_state={"concern": {"value": 0.6, "volatility": 0.4}},
        ),
                "LOC_FRIAR_CELL": Location(
            name="Friar Laurence's Cell",
            description="Friar Laurence's Cell",
            ambient_state={"sanctuary": {"value": 0.7, "volatility": 0.2}},
        ),
                "LOC_CAPULET_TOMB": Location(
            name="Capulet Family Crypt",
            description="Capulet Family Crypt",
            ambient_state={"death": {"value": 1.0, "volatility": 0.0}, "darkness": {"value": 0.9, "volatility": 0.1}},
        ),
                "LOC_MANTUA": Location(
            name="Mantua (Romeo's Exile)",
            description="Mantua (Romeo's Exile)",
            ambient_state={"isolation": {"value": 0.7, "volatility": 0.3}},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_POISON": NarrativeObject(
            id="OBJ_POISON",
            name="Poison (from Apothecary)",
            location_id=None,
            owner_id="ENT_ROMEO",
            properties={"state": "consumed"},
            affordances=[
                Affordance(action="kill", target_type="Entity"),
            ],
        ),
        "OBJ_SLEEPING_POTION": NarrativeObject(
            id="OBJ_SLEEPING_POTION",
            name="Friar Laurence's Sleeping Potion",
            location_id=None,
            owner_id=None,
            properties={"state": "consumed", "duration": "42_hours"},
            affordances=[
                Affordance(action="simulate_death", target_type="Entity"),
            ],
        ),
        "OBJ_DAGGER": NarrativeObject(
            id="OBJ_DAGGER",
            name="Romeo's Dagger",
            location_id="LOC_CAPULET_TOMB",
            owner_id=None,
            properties={"state": "bloodied"},
            affordances=[
                Affordance(action="kill", target_type="Entity"),
            ],
        ),
        "OBJ_LETTER_TO_ROMEO": NarrativeObject(
            id="OBJ_LETTER_TO_ROMEO",
            name="Friar Laurence's Letter to Romeo",
            location_id="LOC_FRIAR_CELL",
            owner_id="ENT_FRIAR_LAURENCE",
            properties={"state": "undelivered"},
            affordances=[
                Affordance(action="inform", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_ROMEO": Entity(
            id="ENT_ROMEO",
            name="Romeo Montague",
            location_id="LOC_CAPULET_TOMB",
            status="dead",
            traits={
                "passion": TraitVector(value=0.95, inertia=0.8),
                "impulsiveness": TraitVector(value=0.9, inertia=0.7),
                "romantic_devotion": TraitVector(value=0.95, inertia=0.9),
                "melancholy": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_JULIET", perceived_state="Juliet is dead", confidence=1.0, inertia=0.9, established_at_fabula=0),
            ],
        ),
        "ENT_JULIET": Entity(
            id="ENT_JULIET",
            name="Juliet Capulet",
            location_id="LOC_CAPULET_TOMB",
            status="dead",
            traits={
                "devotion": TraitVector(value=0.95, inertia=0.9),
                "courage": TraitVector(value=0.8, inertia=0.6),
                "defiance": TraitVector(value=0.8, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_ROMEO", perceived_state="Romeo is my true husband — I will die before marrying Paris", confidence=1.0, inertia=0.95, established_at_fabula=0),
            ],
        ),
        "ENT_TYBALT": Entity(
            id="ENT_TYBALT",
            name="Tybalt",
            location_id="LOC_VERONA_STREETS",
            status="dead",
            traits={
                "aggression": TraitVector(value=0.9, inertia=0.8),
                "family_honour": TraitVector(value=0.95, inertia=0.9),
            },
            beliefs=[
                Belief(target_id="ENT_ROMEO", perceived_state="Romeo came to our feast to insult and mock us", confidence=0.9, inertia=0.8, established_at_fabula=0),
            ],
        ),
        "ENT_MERCUTIO": Entity(
            id="ENT_MERCUTIO",
            name="Mercutio",
            location_id="LOC_VERONA_STREETS",
            status="dead",
            traits={
                "wit": TraitVector(value=0.9, inertia=0.8),
                "bravado": TraitVector(value=0.85, inertia=0.7),
                "loyalty": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_FRIAR_LAURENCE": Entity(
            id="ENT_FRIAR_LAURENCE",
            name="Friar Laurence",
            location_id="LOC_FRIAR_CELL",
            status="healthy",
            traits={
                "wisdom": TraitVector(value=0.75, inertia=0.7),
                "compassion": TraitVector(value=0.8, inertia=0.7),
                "risk_taking": TraitVector(value=0.6, inertia=0.4),
            },
            beliefs=[
                Belief(target_id="ENT_ROMEO", perceived_state="This marriage can reconcile the feuding families", confidence=0.7, inertia=0.5, established_at_fabula=0),
                Belief(target_id="EVT_SLEEPING_POTION", perceived_state="The sleeping potion plan will work and the message will reach Romeo", confidence=0.75, inertia=0.5, established_at_fabula=11),
            ],
        ),
        "ENT_CAPULET": Entity(
            id="ENT_CAPULET",
            name="Lord Capulet",
            location_id="LOC_CAPULET_HOUSE",
            status="healthy",
            traits={
                "authority": TraitVector(value=0.85, inertia=0.8),
                "stubbornness": TraitVector(value=0.8, inertia=0.7),
                "family_honour": TraitVector(value=0.9, inertia=0.9),
            },
            beliefs=[
                Belief(target_id="ENT_JULIET", perceived_state="Juliet grieves for Tybalt — marriage to Paris will cure her sorrow", confidence=0.8, inertia=0.7, established_at_fabula=0),
                Belief(target_id="ENT_JULIET", perceived_state="Juliet is dead", confidence=1.0, inertia=0.8, established_at_fabula=12),
            ],
        ),
        "ENT_PARIS": Entity(
            id="ENT_PARIS",
            name="Count Paris",
            location_id="LOC_CAPULET_TOMB",
            status="dead",
            traits={
                "nobility": TraitVector(value=0.7, inertia=0.6),
                "devotion": TraitVector(value=0.6, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_ROMEO", perceived_state="Romeo is a vandal who has desecrated Juliet's tomb", confidence=0.9, inertia=0.7, established_at_fabula=0),
            ],
        ),
        "ENT_PRINCE_ESCALUS": Entity(
            id="ENT_PRINCE_ESCALUS",
            name="Prince Escalus",
            location_id="LOC_VERONA_STREETS",
            status="healthy",
            traits={
                "authority": TraitVector(value=0.9, inertia=0.9),
                "justice": TraitVector(value=0.8, inertia=0.8),
            },
        ),
        "ENT_BENVOLIO": Entity(
            id="ENT_BENVOLIO",
            name="Benvolio",
            location_id="LOC_VERONA_STREETS",
            status="healthy",
            traits={
                "peacefulness": TraitVector(value=0.8, inertia=0.7),
                "loyalty": TraitVector(value=0.75, inertia=0.7),
            },
        ),
        "ENT_NURSE": Entity(
            id="ENT_NURSE",
            name="Nurse",
            location_id="LOC_CAPULET_HOUSE",
            status="healthy",
            traits={
                "loyalty": TraitVector(value=0.8, inertia=0.6),
                "pragmatism": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_JULIET", perceived_state="Romeo and Juliet's marriage can work", confidence=0.6, inertia=0.4, established_at_fabula=4),
            ],
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_STREET_BRAWL", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_ids=[], description="A street brawl erupts between Montague and Capulet servants. Prince Escalus declares further breaches punishable by death."),
        EventNode(id="EVT_CAPULET_BALL", fabula_time=2, syuzhet_index=2, event_type="outcome", actor_ids=[], description="Romeo attends the Capulet ball hoping to see Rosaline but instead meets and falls in love with Juliet."),
        EventNode(id="EVT_BALCONY_SCENE", fabula_time=3, syuzhet_index=3, event_type="choice", actor_ids=["ENT_ROMEO"], description="Romeo sneaks into the Capulet orchard; he and Juliet vow their love and agree to marry."),
        EventNode(id="EVT_SECRET_MARRIAGE", fabula_time=4, syuzhet_index=4, event_type="choice", actor_ids=["ENT_FRIAR_LAURENCE"], description="Friar Laurence secretly marries Romeo and Juliet, hoping to reconcile the families."),
        EventNode(id="EVT_TYBALT_CHALLENGES_ROMEO", fabula_time=5, syuzhet_index=5, event_type="choice", actor_ids=["ENT_TYBALT"], description="Tybalt challenges Romeo to a duel. Romeo refuses since Tybalt is now his kinsman."),
        EventNode(id="EVT_MERCUTIO_KILLED", fabula_time=6, syuzhet_index=6, event_type="outcome", actor_ids=["ENT_TYBALT"], description="Mercutio accepts the duel on Romeo's behalf and is fatally wounded when Romeo tries to intervene."),
        EventNode(id="EVT_TYBALT_KILLED", fabula_time=7, syuzhet_index=7, event_type="choice", actor_ids=["ENT_ROMEO"], description="Grief-stricken Romeo confronts and slays Tybalt."),
        EventNode(id="EVT_ROMEO_EXILED", fabula_time=8, syuzhet_index=8, event_type="outcome", actor_ids=["ENT_PRINCE_ESCALUS"], description="Prince Escalus exiles Romeo from Verona under penalty of death."),
        EventNode(id="EVT_CONSUMMATION", fabula_time=9, syuzhet_index=9, event_type="choice", actor_ids=["ENT_ROMEO"], description="Romeo secretly spends the night in Juliet's chamber; they consummate their marriage."),
        EventNode(id="EVT_PARIS_BETROTHAL", fabula_time=10, syuzhet_index=10, event_type="choice", actor_ids=["ENT_CAPULET"], description="Lord Capulet agrees to marry Juliet to Count Paris and threatens to disown her when she refuses."),
        EventNode(id="EVT_SLEEPING_POTION", fabula_time=11, syuzhet_index=11, event_type="choice", actor_ids=["ENT_FRIAR_LAURENCE"], description="Friar Laurence gives Juliet a potion to simulate death for 42 hours and plans to inform Romeo."),
        EventNode(id="EVT_JULIET_TAKES_POTION", fabula_time=12, syuzhet_index=12, event_type="choice", actor_ids=["ENT_JULIET"], description="Juliet takes the sleeping potion and is discovered apparently dead."),
        EventNode(id="EVT_MESSAGE_FAILS", fabula_time=13, syuzhet_index=13, event_type="outcome", actor_ids=[], description="Friar John cannot deliver the message to Romeo due to a plague quarantine."),
        EventNode(id="EVT_ROMEO_LEARNS_DEATH", fabula_time=14, syuzhet_index=14, event_type="revelation", actor_ids=[], description="Romeo's servant Balthasar tells him Juliet is dead. Romeo buys poison from an apothecary."),
        EventNode(id="EVT_PARIS_KILLED", fabula_time=15, syuzhet_index=15, event_type="outcome", actor_ids=["ENT_ROMEO"], target_ids=["ENT_PARIS"], description="Romeo encounters Paris at the crypt; they fight and Romeo kills Paris."),
        EventNode(id="EVT_ROMEO_DIES", fabula_time=16, syuzhet_index=16, event_type="choice", actor_ids=["ENT_ROMEO"], description="Believing Juliet dead, Romeo drinks poison and dies beside her."),
        EventNode(id="EVT_JULIET_DIES", fabula_time=17, syuzhet_index=17, event_type="choice", actor_ids=["ENT_JULIET"], target_ids=["ENT_JULIET"], description="Juliet awakens, finds Romeo dead, and stabs herself with his dagger."),
        EventNode(id="EVT_FAMILIES_RECONCILE", fabula_time=18, syuzhet_index=18, event_type="outcome", actor_ids=[], description="The Montagues and Capulets discover the dead lovers and agree to end their feud."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="EVT_STREET_BRAWL", target_id="EVT_CAPULET_BALL", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1),
        CausalEdge(source_id="EVT_CAPULET_BALL", target_id="EVT_BALCONY_SCENE", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=2),
        CausalEdge(source_id="EVT_BALCONY_SCENE", target_id="EVT_SECRET_MARRIAGE", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=3),
        CausalEdge(source_id="EVT_SECRET_MARRIAGE", target_id="EVT_TYBALT_CHALLENGES_ROMEO", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=4),
        CausalEdge(source_id="EVT_TYBALT_CHALLENGES_ROMEO", target_id="EVT_MERCUTIO_KILLED", causality_type="chain_reaction", mechanism="physical", evidence_strength="moderate", causal_force=5.0, fabula_time=5),
        CausalEdge(source_id="EVT_MERCUTIO_KILLED", target_id="EVT_TYBALT_KILLED", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=6),
        CausalEdge(source_id="EVT_TYBALT_KILLED", target_id="EVT_ROMEO_EXILED", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=7),
        CausalEdge(source_id="EVT_ROMEO_EXILED", target_id="EVT_CONSUMMATION", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=8),
        CausalEdge(source_id="EVT_CONSUMMATION", target_id="EVT_PARIS_BETROTHAL", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=9),
        CausalEdge(source_id="EVT_PARIS_BETROTHAL", target_id="EVT_SLEEPING_POTION", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=10),
        CausalEdge(source_id="EVT_JULIET_TAKES_POTION", target_id="EVT_MESSAGE_FAILS", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=12),
        CausalEdge(source_id="EVT_MESSAGE_FAILS", target_id="EVT_ROMEO_LEARNS_DEATH", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=13),
        CausalEdge(source_id="EVT_ROMEO_LEARNS_DEATH", target_id="EVT_PARIS_KILLED", causality_type="chain_reaction", mechanism="physical", evidence_strength="moderate", causal_force=5.0, fabula_time=14),
        CausalEdge(source_id="EVT_ROMEO_LEARNS_DEATH", target_id="EVT_ROMEO_DIES", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=14),
        CausalEdge(source_id="EVT_ROMEO_DIES", target_id="EVT_JULIET_DIES", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=16),
        CausalEdge(source_id="EVT_JULIET_DIES", target_id="EVT_FAMILIES_RECONCILE", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=17),
        CausalEdge(source_id="EVT_ROMEO_DIES", target_id="EVT_FAMILIES_RECONCILE", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=16),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_CAPULET_HOUSE", target_id="LOC_CAPULET_ORCHARD"),
        SpatialEdge(source_id="LOC_CAPULET_HOUSE", target_id="LOC_CAPULET_TOMB"),
        SpatialEdge(source_id="LOC_CAPULET_HOUSE", target_id="LOC_VERONA_STREETS"),
        SpatialEdge(source_id="LOC_CAPULET_ORCHARD", target_id="LOC_VERONA_STREETS"),
        SpatialEdge(source_id="LOC_FRIAR_CELL", target_id="LOC_VERONA_STREETS"),
        SpatialEdge(source_id="LOC_MANTUA", target_id="LOC_VERONA_STREETS"),
        SpatialEdge(source_id="LOC_MONTAGUE_HOUSE", target_id="LOC_VERONA_STREETS"),
    ],
    information_topology=[
        InformationEdge(
            source_id="ENT_FRIAR_LAURENCE",
            target_ids=["ENT_JULIET"],
            medium="confession",
            established_at_fabula=11,
            terminated_at_fabula=11,
        ),
        InformationEdge(
            source_id="ENT_FRIAR_LAURENCE",
            target_ids=["ENT_ROMEO"],
            medium="letter",
            established_at_fabula=13,
            terminated_at_fabula=13,
        ),
    ],
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_ROMEO", target_entity_id="ENT_JULIET", affinity=1.0, fear=0.35, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_JULIET", target_entity_id="ENT_ROMEO", affinity=1.0, fear=0.35, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_TYBALT", target_entity_id="ENT_ROMEO", affinity=-0.9, fear=0.47, power_dynamic=0.3),
        RelationshipEdge(source_entity_id="ENT_ROMEO", target_entity_id="ENT_MERCUTIO", affinity=0.85, fear=0.15, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_CAPULET", target_entity_id="ENT_JULIET", affinity=0.5, fear=0.4, power_dynamic=0.9),
        RelationshipEdge(source_entity_id="ENT_CAPULET", target_entity_id="ENT_PARIS", affinity=0.6, fear=0.1, power_dynamic=0.3),
        RelationshipEdge(source_entity_id="ENT_FRIAR_LAURENCE", target_entity_id="ENT_ROMEO", affinity=0.7, fear=0.1, power_dynamic=0.3),
        RelationshipEdge(source_entity_id="ENT_FRIAR_LAURENCE", target_entity_id="ENT_JULIET", affinity=0.6, fear=0.1, power_dynamic=0.3),
        RelationshipEdge(source_entity_id="ENT_PRINCE_ESCALUS", target_entity_id="ENT_ROMEO", affinity=-0.3, fear=0.35, power_dynamic=0.9),
        RelationshipEdge(source_entity_id="ENT_BENVOLIO", target_entity_id="ENT_ROMEO", affinity=0.8, fear=0.05, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_NURSE", target_entity_id="ENT_JULIET", affinity=0.85, fear=0.2, power_dynamic=0.1),
        RelationshipEdge(source_entity_id="ENT_JULIET", target_entity_id="ENT_CAPULET", affinity=0.4, fear=0.45, power_dynamic=-0.8),
    ],
)
