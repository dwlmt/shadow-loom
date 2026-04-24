from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, InformationEdge, RelationshipEdge, TraitVector,
    Affordance, Belief,
)

# =============================================================================
# A FISH CALLED WANDA — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
                "LOC_LONDON": Location(
            name="London",
            description="London",
            ambient_state={"intrigue": {"value": 0.7, "volatility": 0.5}},
        ),
                "LOC_OLD_WORKSHOP": Location(
            name="Old Workshop (Original Diamond Stash)",
            description="Old Workshop (Original Diamond Stash)",
            ambient_state={},
        ),
                "LOC_ARCHIE_HOUSE": Location(
            name="Archie Leach's House",
            description="Archie Leach's House",
            ambient_state={"domesticity": {"value": 0.6, "volatility": 0.5}},
        ),
                "LOC_COURTHOUSE": Location(
            name="Courthouse",
            description="Courthouse",
            ambient_state={"tension": {"value": 0.8, "volatility": 0.4}},
        ),
                "LOC_KEN_FLAT": Location(
            name="Ken's Flat",
            description="Ken's Flat",
            ambient_state={},
        ),
                "LOC_HEATHROW_HOTEL": Location(
            name="Hotel near Heathrow Airport",
            description="Hotel near Heathrow Airport",
            ambient_state={},
        ),
                "LOC_AIRPORT": Location(
            name="Heathrow Airport",
            description="Heathrow Airport",
            ambient_state={"escape": {"value": 0.8, "volatility": 0.5}},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_DIAMONDS": NarrativeObject(
            id="OBJ_DIAMONDS",
            name="Stolen Diamonds",
            location_id="LOC_HEATHROW_HOTEL",
            owner_id=None,
            properties={"state": "recovered_by_wanda_and_archie", "value": "extremely_high"},
            affordances=[
                Affordance(action="sell", target_type="Entity"),
                Affordance(action="motivate", target_type="Entity"),
            ],
        ),
        "OBJ_SAFE_KEY": NarrativeObject(
            id="OBJ_SAFE_KEY",
            name="Safe Deposit Box Key",
            location_id=None,
            owner_id="ENT_WANDA",
            properties={"state": "hidden_in_pendant"},
            affordances=[
                Affordance(action="unlock", target_type="NarrativeObject"),
            ],
        ),
        "OBJ_PENDANT": NarrativeObject(
            id="OBJ_PENDANT",
            name="Wanda's Pendant (contains key)",
            location_id=None,
            owner_id="ENT_WANDA",
            properties={"state": "recovered"},
            affordances=[
                Affordance(action="conceal", target_type="NarrativeObject"),
            ],
        ),
        "OBJ_KEN_FISH": NarrativeObject(
            id="OBJ_KEN_FISH",
            name="Ken's Pet Fish (including Wanda the fish)",
            location_id="LOC_KEN_FLAT",
            owner_id="ENT_KEN",
            properties={"state": "eaten_by_otto"},
            affordances=[
                Affordance(action="coerce", target_type="Entity"),
            ],
        ),
        "OBJ_STEAMROLLER": NarrativeObject(
            id="OBJ_STEAMROLLER",
            name="Steamroller",
            location_id="LOC_HEATHROW_HOTEL",
            owner_id=None,
            properties={"state": "used_for_vengeance"},
            affordances=[
                Affordance(action="crush", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_ARCHIE": Entity(
            id="ENT_ARCHIE",
            name="Archie Leach",
            location_id="LOC_AIRPORT",
            status="healthy",
            traits={
                "repression": TraitVector(value=0.7, inertia=0.5),
                "charm": TraitVector(value=0.7, inertia=0.6),
                "opportunism": TraitVector(value=0.7, inertia=0.5),
                "romantic_frustration": TraitVector(value=0.8, inertia=0.4),
            },
            beliefs=[
                Belief(target_id="ENT_WANDA", perceived_state="Wanda is genuinely attracted to me", confidence=0.8, inertia=0.5, established_at_fabula=0),
            ],
        ),
        "ENT_WANDA": Entity(
            id="ENT_WANDA",
            name="Wanda Gershwitz",
            location_id="LOC_AIRPORT",
            status="healthy",
            traits={
                "cunning": TraitVector(value=0.9, inertia=0.8),
                "seductiveness": TraitVector(value=0.9, inertia=0.7),
                "treachery": TraitVector(value=0.85, inertia=0.6),
                "self_interest": TraitVector(value=0.9, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_OTTO", perceived_state="Otto is a useful tool I will discard when I have the diamonds", confidence=0.9, inertia=0.8, established_at_fabula=0),
            ],
        ),
        "ENT_OTTO": Entity(
            id="ENT_OTTO",
            name="Otto West",
            location_id="LOC_AIRPORT",
            status="injured",
            traits={
                "aggression": TraitVector(value=0.9, inertia=0.8),
                "jealousy": TraitVector(value=0.85, inertia=0.7),
                "stupidity": TraitVector(value=0.7, inertia=0.7),
                "anglophobia": TraitVector(value=0.8, inertia=0.7),
                "pseudo_intellectualism": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_WANDA", perceived_state="Wanda is my lover and partner — she won't betray me", confidence=0.7, inertia=0.4, established_at_fabula=0),
            ],
        ),
        "ENT_KEN": Entity(
            id="ENT_KEN",
            name="Ken Pile",
            location_id="LOC_LONDON",
            status="healthy",
            traits={
                "kindness": TraitVector(value=0.8, inertia=0.7),
                "loyalty": TraitVector(value=0.7, inertia=0.6),
                "animal_love": TraitVector(value=0.95, inertia=0.9),
                "vengefulness": TraitVector(value=0.6, inertia=0.4),
            },
            constants=["stutter"],
            beliefs=[
                Belief(target_id="ENT_WANDA", perceived_state="Wanda and Otto are brother and sister, not lovers", confidence=0.9, inertia=0.7, established_at_fabula=0),
            ],
        ),
        "ENT_GEORGE": Entity(
            id="ENT_GEORGE",
            name="George Thomason",
            location_id="LOC_COURTHOUSE",
            status="healthy",
            traits={
                "cunning": TraitVector(value=0.75, inertia=0.6),
                "authority": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_WANDA", perceived_state="Wanda and Otto are brother and sister, not lovers", confidence=0.9, inertia=0.7, established_at_fabula=0),
                Belief(target_id="ENT_WANDA", perceived_state="Wanda is a loyal member of my gang", confidence=0.8, inertia=0.6, established_at_fabula=0),
            ],
        ),
        "ENT_WENDY": Entity(
            id="ENT_WENDY",
            name="Wendy Leach",
            location_id="LOC_ARCHIE_HOUSE",
            status="healthy",
            traits={
                "propriety": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="OBJ_PENDANT", perceived_state="This pendant is a gift from Archie for me", confidence=0.85, inertia=0.5, established_at_fabula=0),
            ],
        ),
        "ENT_MRS_COADY": Entity(
            id="ENT_MRS_COADY",
            name="Mrs Coady (Eyewitness)",
            location_id="LOC_LONDON",
            status="dead",
            traits={},
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_HEIST", fabula_time=1, syuzhet_index=1, event_type="choice", actor_id="ENT_GEORGE", description="George, Ken, Wanda, and Otto execute a successful jewel heist. Diamonds are hidden in a safe in the old workshop."),
        EventNode(id="EVT_WANDA_OTTO_BETRAY_GEORGE", fabula_time=2, syuzhet_index=2, event_type="choice", actor_id="ENT_WANDA", description="Wanda and Otto betray George to the police. He is arrested."),
        EventNode(id="EVT_DIAMONDS_MOVED", fabula_time=3, syuzhet_index=3, event_type="outcome", actor_id="ENT_GEORGE", description="George has already moved the diamonds. Wanda finds the safe deposit key in Ken's fish tank and hides it in her pendant."),
        EventNode(id="EVT_WANDA_SEDUCES_ARCHIE", fabula_time=4, syuzhet_index=4, event_type="choice", actor_id="ENT_WANDA", description="Wanda seduces George's barrister Archie Leach to get him to persuade George to plead guilty and reveal the diamonds' location."),
        EventNode(id="EVT_ARCHIE_FALLS_FOR_WANDA", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_id="ENT_ARCHIE", description="Archie, in a loveless marriage, falls for Wanda. Otto's jealous interference causes their meetings to go wrong."),
        EventNode(id="EVT_PENDANT_LOST", fabula_time=6, syuzhet_index=6, event_type="outcome", actor_id="ENT_WANDA", description="Wanda accidentally leaves her pendant at Archie's house. Wendy mistakes it for a gift. Archie stages a burglary to recover it."),
        EventNode(id="EVT_ARCHIE_ENDS_AFFAIR", fabula_time=7, syuzhet_index=7, event_type="choice", actor_id="ENT_ARCHIE", description="Feeling guilty, Archie ends the affair with Wanda."),
        EventNode(id="EVT_KEN_KILLS_DOGS", fabula_time=8, syuzhet_index=8, event_type="outcome", actor_id="ENT_KEN", target_id="ENT_MRS_COADY", description="Ken's attempts to kill witness Mrs Coady accidentally kill her three dogs. She ultimately dies of a heart attack."),
        EventNode(id="EVT_WANDA_TESTIFIES_AGAINST_GEORGE", fabula_time=9, syuzhet_index=9, event_type="choice", actor_id="ENT_WANDA", description="At trial, defence witness Wanda unexpectedly testifies against George."),
        EventNode(id="EVT_ARCHIE_EXPOSED", fabula_time=10, syuzhet_index=10, event_type="outcome", actor_id="ENT_ARCHIE", description="Archie calls Wanda 'darling' during cross-examination. Wendy realizes the affair and decides to divorce him."),
        EventNode(id="EVT_OTTO_EATS_FISH", fabula_time=11, syuzhet_index=11, event_type="choice", actor_id="ENT_OTTO", description="Otto eats Ken's pet fish one by one to force Ken to reveal the diamonds' location at the Heathrow hotel."),
        EventNode(id="EVT_ARCHIE_RESOLVES_STEAL", fabula_time=12, syuzhet_index=12, event_type="choice", actor_id="ENT_ARCHIE", description="With career and marriage over, Archie resolves to steal the diamonds and flee to South America."),
        EventNode(id="EVT_DIAMOND_CHASE", fabula_time=13, syuzhet_index=13, event_type="outcome", actor_id=None, description="Otto steals Archie's car with Wanda. They recover the diamonds, but Wanda double-crosses Otto and knocks him out."),
        EventNode(id="EVT_OTTO_VS_ARCHIE", fabula_time=14, syuzhet_index=14, event_type="outcome", actor_id="ENT_OTTO", description="Otto escapes and confronts Archie. Archie stalls by taunting Otto about American failures."),
        EventNode(id="EVT_STEAMROLLER", fabula_time=15, syuzhet_index=15, event_type="outcome", actor_id="ENT_KEN", description="Ken drives a steamroller over Otto (stuck in wet concrete) seeking vengeance for his fish. Otto survives."),
        EventNode(id="EVT_ESCAPE", fabula_time=16, syuzhet_index=16, event_type="outcome", actor_id="ENT_ARCHIE", description="Archie and Wanda board the plane with the diamonds. Otto clings to the window until blown off during takeoff."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_event_id="EVT_HEIST", target_node_id="EVT_WANDA_OTTO_BETRAY_GEORGE", mechanism="social", fabula_time=1),
        CausalEdge(source_event_id="EVT_WANDA_OTTO_BETRAY_GEORGE", target_node_id="EVT_DIAMONDS_MOVED", mechanism="epistemic", fabula_time=2),
        CausalEdge(source_event_id="EVT_WANDA_OTTO_BETRAY_GEORGE", target_node_id="EVT_WANDA_SEDUCES_ARCHIE", mechanism="social", fabula_time=2),
        CausalEdge(source_event_id="EVT_WANDA_SEDUCES_ARCHIE", target_node_id="EVT_ARCHIE_FALLS_FOR_WANDA", mechanism="psychological", fabula_time=4),
        CausalEdge(source_event_id="EVT_PENDANT_LOST", target_node_id="EVT_ARCHIE_ENDS_AFFAIR", mechanism="psychological", fabula_time=6),
        CausalEdge(source_event_id="EVT_KEN_KILLS_DOGS", target_node_id="EVT_WANDA_TESTIFIES_AGAINST_GEORGE", mechanism="social", fabula_time=8),
        CausalEdge(source_event_id="EVT_WANDA_TESTIFIES_AGAINST_GEORGE", target_node_id="EVT_ARCHIE_EXPOSED", mechanism="social", fabula_time=9),
        CausalEdge(source_event_id="EVT_OTTO_EATS_FISH", target_node_id="EVT_DIAMOND_CHASE", mechanism="epistemic", fabula_time=11),
        CausalEdge(source_event_id="EVT_ARCHIE_EXPOSED", target_node_id="EVT_ARCHIE_RESOLVES_STEAL", mechanism="psychological", fabula_time=10),
        CausalEdge(source_event_id="EVT_DIAMOND_CHASE", target_node_id="EVT_OTTO_VS_ARCHIE", mechanism="physical", fabula_time=13),
        CausalEdge(source_event_id="EVT_OTTO_EATS_FISH", target_node_id="EVT_STEAMROLLER", mechanism="psychological", fabula_time=11),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_AIRPORT", target_id="LOC_HEATHROW_HOTEL"),
        SpatialEdge(source_id="LOC_ARCHIE_HOUSE", target_id="LOC_COURTHOUSE"),
        SpatialEdge(source_id="LOC_ARCHIE_HOUSE", target_id="LOC_LONDON"),
        SpatialEdge(source_id="LOC_COURTHOUSE", target_id="LOC_LONDON"),
        SpatialEdge(source_id="LOC_HEATHROW_HOTEL", target_id="LOC_LONDON"),
        SpatialEdge(source_id="LOC_KEN_FLAT", target_id="LOC_LONDON"),
        SpatialEdge(source_id="LOC_LONDON", target_id="LOC_OLD_WORKSHOP"),
    ],
    information_topology=[
        InformationEdge(
            source_id="ENT_WANDA",
            target_ids=["ENT_ARCHIE"],
            medium="seduction",
            established_at_fabula=4,
            terminated_at_fabula=7,
        ),
    ],
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_WANDA", target_entity_id="ENT_OTTO", affinity=-0.2, fear=0.4, power_dynamic=0.3),
        RelationshipEdge(source_entity_id="ENT_OTTO", target_entity_id="ENT_WANDA", affinity=0.7, fear=0.4, power_dynamic=-0.1),
        RelationshipEdge(source_entity_id="ENT_WANDA", target_entity_id="ENT_ARCHIE", affinity=0.5, fear=0.2, power_dynamic=0.4),
        RelationshipEdge(source_entity_id="ENT_ARCHIE", target_entity_id="ENT_WANDA", affinity=0.8, fear=0.25, power_dynamic=-0.3),
        RelationshipEdge(source_entity_id="ENT_ARCHIE", target_entity_id="ENT_WENDY", affinity=0.1, fear=0.3, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_KEN", target_entity_id="ENT_GEORGE", affinity=0.6, fear=0.1, power_dynamic=-0.4),
        RelationshipEdge(source_entity_id="ENT_KEN", target_entity_id="ENT_OTTO", affinity=-0.8, fear=0.45, power_dynamic=-0.3),
        RelationshipEdge(source_entity_id="ENT_OTTO", target_entity_id="ENT_ARCHIE", affinity=-0.7, fear=0.45, power_dynamic=0.2),
    ],
)
