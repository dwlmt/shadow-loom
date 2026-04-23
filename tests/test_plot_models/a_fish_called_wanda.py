from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
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
            id="LOC_LONDON",
            name="London",
            connected_locations=["LOC_OLD_WORKSHOP", "LOC_COURTHOUSE", "LOC_ARCHIE_HOUSE", "LOC_KEN_FLAT"],
            ambient_states={
                "intrigue": AmbientVector(value=0.7, volatility=0.5),
            },
        ),
        "LOC_OLD_WORKSHOP": Location(
            id="LOC_OLD_WORKSHOP",
            name="Old Workshop (Original Diamond Stash)",
            connected_locations=["LOC_LONDON"],
            ambient_states={},
        ),
        "LOC_ARCHIE_HOUSE": Location(
            id="LOC_ARCHIE_HOUSE",
            name="Archie Leach's House",
            connected_locations=["LOC_LONDON", "LOC_COURTHOUSE"],
            ambient_states={
                "domesticity": AmbientVector(value=0.6, volatility=0.5),
            },
        ),
        "LOC_COURTHOUSE": Location(
            id="LOC_COURTHOUSE",
            name="Courthouse",
            connected_locations=["LOC_LONDON", "LOC_ARCHIE_HOUSE"],
            ambient_states={
                "tension": AmbientVector(value=0.8, volatility=0.4),
            },
        ),
        "LOC_KEN_FLAT": Location(
            id="LOC_KEN_FLAT",
            name="Ken's Flat",
            connected_locations=["LOC_LONDON"],
            ambient_states={},
        ),
        "LOC_HEATHROW_HOTEL": Location(
            id="LOC_HEATHROW_HOTEL",
            name="Hotel near Heathrow Airport",
            connected_locations=["LOC_LONDON", "LOC_AIRPORT"],
            ambient_states={},
        ),
        "LOC_AIRPORT": Location(
            id="LOC_AIRPORT",
            name="Heathrow Airport",
            connected_locations=["LOC_HEATHROW_HOTEL"],
            ambient_states={
                "escape": AmbientVector(value=0.8, volatility=0.5),
            },
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
                Belief(target_id="ENT_WANDA", perceived_state="Wanda is genuinely attracted to me", confidence=0.8, inertia=0.5),
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
                Belief(target_id="ENT_OTTO", perceived_state="Otto is a useful tool I will discard when I have the diamonds", confidence=0.9, inertia=0.8),
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
                Belief(target_id="ENT_WANDA", perceived_state="Wanda is my lover and partner — she won't betray me", confidence=0.7, inertia=0.4),
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
                Belief(target_id="ENT_WANDA", perceived_state="Wanda and Otto are brother and sister, not lovers", confidence=0.9, inertia=0.7),
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
                Belief(target_id="ENT_WANDA", perceived_state="Wanda and Otto are brother and sister, not lovers", confidence=0.9, inertia=0.7),
                Belief(target_id="ENT_WANDA", perceived_state="Wanda is a loyal member of my gang", confidence=0.8, inertia=0.6),
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
                Belief(target_id="OBJ_PENDANT", perceived_state="This pendant is a gift from Archie for me", confidence=0.85, inertia=0.5),
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
        EventNode(id="EVT_KEN_KILLS_DOGS", fabula_time=8, syuzhet_index=8, event_type="outcome", actor_id="ENT_KEN", description="Ken's attempts to kill witness Mrs Coady accidentally kill her three dogs. She ultimately dies of a heart attack."),
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
        CausalEdge(source_id="ENT_GEORGE", target_id="EVT_HEIST", mechanism="social"),
        CausalEdge(source_id="EVT_HEIST", target_id="EVT_WANDA_OTTO_BETRAY_GEORGE", mechanism="social"),
        CausalEdge(source_id="EVT_WANDA_OTTO_BETRAY_GEORGE", target_id="EVT_DIAMONDS_MOVED", mechanism="epistemic"),
        CausalEdge(source_id="OBJ_SAFE_KEY", target_id="EVT_DIAMONDS_MOVED", mechanism="physical"),
        CausalEdge(source_id="EVT_WANDA_OTTO_BETRAY_GEORGE", target_id="EVT_WANDA_SEDUCES_ARCHIE", mechanism="social"),
        CausalEdge(source_id="ENT_WANDA", target_id="EVT_WANDA_SEDUCES_ARCHIE", mechanism="psychological"),
        CausalEdge(source_id="EVT_WANDA_SEDUCES_ARCHIE", target_id="EVT_ARCHIE_FALLS_FOR_WANDA", mechanism="psychological"),
        CausalEdge(source_id="ENT_OTTO", target_id="EVT_ARCHIE_FALLS_FOR_WANDA", mechanism="social"),
        CausalEdge(source_id="OBJ_PENDANT", target_id="EVT_PENDANT_LOST", mechanism="physical"),
        CausalEdge(source_id="EVT_PENDANT_LOST", target_id="EVT_ARCHIE_ENDS_AFFAIR", mechanism="psychological"),
        CausalEdge(source_id="ENT_KEN", target_id="EVT_KEN_KILLS_DOGS", mechanism="physical"),
        CausalEdge(source_id="EVT_KEN_KILLS_DOGS", target_id="EVT_WANDA_TESTIFIES_AGAINST_GEORGE", mechanism="social"),
        CausalEdge(source_id="EVT_WANDA_TESTIFIES_AGAINST_GEORGE", target_id="EVT_ARCHIE_EXPOSED", mechanism="social"),
        CausalEdge(source_id="OBJ_KEN_FISH", target_id="EVT_OTTO_EATS_FISH", mechanism="psychological"),
        CausalEdge(source_id="EVT_OTTO_EATS_FISH", target_id="EVT_DIAMOND_CHASE", mechanism="epistemic"),
        CausalEdge(source_id="EVT_ARCHIE_EXPOSED", target_id="EVT_ARCHIE_RESOLVES_STEAL", mechanism="psychological"),
        CausalEdge(source_id="EVT_DIAMOND_CHASE", target_id="EVT_OTTO_VS_ARCHIE", mechanism="physical"),
        CausalEdge(source_id="OBJ_STEAMROLLER", target_id="EVT_STEAMROLLER", mechanism="physical"),
        CausalEdge(source_id="EVT_OTTO_EATS_FISH", target_id="EVT_STEAMROLLER", mechanism="psychological"),
        CausalEdge(source_id="OBJ_DIAMONDS", target_id="EVT_ESCAPE", mechanism="physical"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_WANDA", target_entity_id="ENT_OTTO", affinity=-0.2, friction=0.8, power_dynamic=0.3, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_OTTO", target_entity_id="ENT_WANDA", affinity=0.7, friction=0.8, power_dynamic=-0.1, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_WANDA", target_entity_id="ENT_ARCHIE", affinity=0.5, friction=0.4, power_dynamic=0.4, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_ARCHIE", target_entity_id="ENT_WANDA", affinity=0.8, friction=0.5, power_dynamic=-0.3, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_ARCHIE", target_entity_id="ENT_WENDY", affinity=0.1, friction=0.6, power_dynamic=0.0, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_KEN", target_entity_id="ENT_GEORGE", affinity=0.6, friction=0.2, power_dynamic=-0.4, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_KEN", target_entity_id="ENT_OTTO", affinity=-0.8, friction=0.9, power_dynamic=-0.3, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_OTTO", target_entity_id="ENT_ARCHIE", affinity=-0.7, friction=0.9, power_dynamic=0.2, inertia=0.5),
    ],
)
