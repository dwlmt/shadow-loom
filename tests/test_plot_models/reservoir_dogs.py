from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
    Affordance, Belief,
)

# =============================================================================
# RESERVOIR DOGS — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
        "LOC_DINER": Location(
            id="LOC_DINER",
            name="Uncle Bob's Pancake House (Diner)",
            connected_locations=["LOC_STREETS"],
            ambient_states={
                "camaraderie": AmbientVector(value=0.7, volatility=0.3),
            },
        ),
        "LOC_JEWELRY_STORE": Location(
            id="LOC_JEWELRY_STORE",
            name="Karina's Jewelry Store",
            connected_locations=["LOC_STREETS"],
            ambient_states={
                "tension": AmbientVector(value=0.9, volatility=0.5),
            },
        ),
        "LOC_STREETS": Location(
            id="LOC_STREETS",
            name="Los Angeles Streets",
            connected_locations=["LOC_DINER", "LOC_JEWELRY_STORE", "LOC_WAREHOUSE"],
            ambient_states={
                "danger": AmbientVector(value=0.8, volatility=0.6),
            },
        ),
        "LOC_WAREHOUSE": Location(
            id="LOC_WAREHOUSE",
            name="Warehouse Rendezvous Point",
            connected_locations=["LOC_STREETS"],
            ambient_states={
                "paranoia": AmbientVector(value=0.9, volatility=0.4),
                "dread": AmbientVector(value=0.85, volatility=0.5),
            },
        ),
        "LOC_JOES_OFFICE": Location(
            id="LOC_JOES_OFFICE",
            name="Joe Cabot's Office",
            connected_locations=["LOC_STREETS"],
            ambient_states={
                "authority": AmbientVector(value=0.8, volatility=0.2),
            },
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_DIAMONDS": NarrativeObject(
            id="OBJ_DIAMONDS",
            name="Stolen Diamond Shipment",
            location_id=None,
            owner_id=None,
            properties={"state": "hidden_by_pink"},
            affordances=[
                Affordance(action="motivate", target_type="Entity"),
            ],
        ),
        "OBJ_STRAIGHT_RAZOR": NarrativeObject(
            id="OBJ_STRAIGHT_RAZOR",
            name="Mr. Blonde's Straight Razor",
            location_id="LOC_WAREHOUSE",
            owner_id="ENT_BLONDE",
            properties={"state": "used_for_torture"},
            affordances=[
                Affordance(action="torture", target_type="Entity"),
            ],
        ),
        "OBJ_GASOLINE": NarrativeObject(
            id="OBJ_GASOLINE",
            name="Gasoline Can",
            location_id="LOC_WAREHOUSE",
            owner_id=None,
            properties={"state": "unused"},
            affordances=[
                Affordance(action="immolate", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_WHITE": Entity(
            id="ENT_WHITE",
            name="Mr. White (Larry Dimmick)",
            location_id="LOC_WAREHOUSE",
            status="dead",
            traits={
                "loyalty": TraitVector(value=0.9, inertia=0.8),
                "professionalism": TraitVector(value=0.8, inertia=0.7),
                "protectiveness": TraitVector(value=0.85, inertia=0.7),
                "temper": TraitVector(value=0.7, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_ORANGE", perceived_state="Orange is one of us — I got him into this and I owe him", confidence=0.9, inertia=0.85),
            ],
        ),
        "ENT_ORANGE": Entity(
            id="ENT_ORANGE",
            name="Mr. Orange (Freddy Newendyke)",
            location_id="LOC_WAREHOUSE",
            status="dead",
            traits={
                "courage": TraitVector(value=0.8, inertia=0.7),
                "deception": TraitVector(value=0.8, inertia=0.6),
                "guilt": TraitVector(value=0.7, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_WHITE", perceived_state="White protected me — I owe him the truth", confidence=0.9, inertia=0.7),
            ],
        ),
        "ENT_BLONDE": Entity(
            id="ENT_BLONDE",
            name="Mr. Blonde (Vic Vega)",
            location_id="LOC_WAREHOUSE",
            status="dead",
            traits={
                "psychopathy": TraitVector(value=0.95, inertia=0.9),
                "loyalty_to_joe": TraitVector(value=0.9, inertia=0.85),
                "sadism": TraitVector(value=0.9, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_JOE", perceived_state="I owe Joe everything — he's the only one who stood by me", confidence=0.95, inertia=0.9),
            ],
        ),
        "ENT_PINK": Entity(
            id="ENT_PINK",
            name="Mr. Pink",
            location_id="LOC_STREETS",
            status="healthy",
            traits={
                "paranoia": TraitVector(value=0.85, inertia=0.7),
                "pragmatism": TraitVector(value=0.9, inertia=0.8),
                "self_preservation": TraitVector(value=0.95, inertia=0.85),
            },
            beliefs=[
                Belief(target_id="ENT_JOE", perceived_state="The job was a setup — someone tipped off the cops", confidence=0.95, inertia=0.8),
            ],
        ),
        "ENT_JOE": Entity(
            id="ENT_JOE",
            name="Joe Cabot",
            location_id="LOC_WAREHOUSE",
            status="dead",
            traits={
                "authority": TraitVector(value=0.9, inertia=0.85),
                "shrewdness": TraitVector(value=0.85, inertia=0.8),
                "ruthlessness": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_ORANGE", perceived_state="Orange is the rat — this kid is the informer", confidence=0.9, inertia=0.8),
            ],
        ),
        "ENT_EDDIE": Entity(
            id="ENT_EDDIE",
            name="Nice Guy Eddie Cabot",
            location_id="LOC_WAREHOUSE",
            status="dead",
            traits={
                "loyalty": TraitVector(value=0.85, inertia=0.8),
                "volatility": TraitVector(value=0.7, inertia=0.6),
            },
            beliefs=[
                Belief(target_id="ENT_BLONDE", perceived_state="Blonde was loyal — he did four years and never talked", confidence=0.9, inertia=0.8),
            ],
        ),
        "ENT_BROWN": Entity(
            id="ENT_BROWN",
            name="Mr. Brown",
            location_id="LOC_STREETS",
            status="dead",
            traits={
                "talkativeness": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_BLUE": Entity(
            id="ENT_BLUE",
            name="Mr. Blue",
            location_id="LOC_STREETS",
            status="dead",
            traits={
                "composure": TraitVector(value=0.7, inertia=0.6),
            },
        ),
        "ENT_NASH": Entity(
            id="ENT_NASH",
            name="Officer Marvin Nash",
            location_id="LOC_WAREHOUSE",
            status="dead",
            traits={
                "endurance": TraitVector(value=0.8, inertia=0.7),
                "integrity": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_ORANGE", perceived_state="Orange is undercover — I must protect his cover and survive", confidence=1.0, inertia=0.9),
            ],
        ),
    },

    # ── EVENTS (Chronological — fabula order) ──────────────────────────────
    events=[
        EventNode(id="EVT_BLONDE_PAROLED", fabula_time=1, syuzhet_index=7, event_type="outcome", actor_id="ENT_BLONDE", description="Blonde is paroled after four years. He meets the Cabots, who reward his silence with a no-show job; he insists on real work."),
        EventNode(id="EVT_ORANGE_UNDERCOVER", fabula_time=2, syuzhet_index=11, event_type="choice", actor_id="ENT_ORANGE", description="Undercover officer Freddy Newendyke infiltrates as Mr. Orange, gaining Joe's and White's confidence with a rehearsed drug-deal anecdote."),
        EventNode(id="EVT_TEAM_ASSEMBLED", fabula_time=3, syuzhet_index=1, event_type="outcome", actor_id="ENT_JOE", description="Joe assembles six strangers under colour aliases at the diner to plan the diamond heist."),
        EventNode(id="EVT_HEIST_GOES_WRONG", fabula_time=4, syuzhet_index=2, event_type="outcome", actor_id=None, description="The heist goes wrong: an alarm is tripped, Blonde shoots bystanders, and police arrive immediately — confirming it was a setup."),
        EventNode(id="EVT_BROWN_KILLED", fabula_time=5, syuzhet_index=3, event_type="outcome", actor_id=None, description="Mr. Brown is killed by police during the escape."),
        EventNode(id="EVT_ORANGE_SHOT", fabula_time=6, syuzhet_index=4, event_type="outcome", actor_id="ENT_ORANGE", description="Orange is shot in the abdomen while hijacking a car; he kills the driver. White drives him to the warehouse."),
        EventNode(id="EVT_WAREHOUSE_RENDEZVOUS", fabula_time=7, syuzhet_index=5, event_type="outcome", actor_id=None, description="White and the bleeding Orange reach the warehouse. Pink arrives with the diamonds hidden nearby. They debate whether the job was a setup."),
        EventNode(id="EVT_BLONDE_BRINGS_NASH", fabula_time=8, syuzhet_index=6, event_type="outcome", actor_id="ENT_BLONDE", description="Blonde arrives with kidnapped Officer Marvin Nash. White and Pink stand down from their argument."),
        EventNode(id="EVT_NASH_TORTURED", fabula_time=9, syuzhet_index=8, event_type="outcome", actor_id=None, description="White and Pink rough up Nash. Eddie arrives and sends them to ditch the getaway cars, leaving Blonde in charge."),
        EventNode(id="EVT_EAR_SCENE", fabula_time=10, syuzhet_index=9, event_type="choice", actor_id="ENT_BLONDE", description="Blonde tortures Nash — slashing his face, cutting off his ear with a razor while dancing to 'Stuck in the Middle with You' — then prepares to set him on fire."),
        EventNode(id="EVT_ORANGE_KILLS_BLONDE", fabula_time=11, syuzhet_index=10, event_type="choice", actor_id="ENT_ORANGE", description="Orange shoots and kills Blonde. He reveals to Nash that he is an undercover cop. Nash confirms he recognised and protected Orange's cover."),
        EventNode(id="EVT_EDDIE_KILLS_NASH", fabula_time=12, syuzhet_index=12, event_type="choice", actor_id="ENT_EDDIE", description="Eddie, Pink, and White return. Orange claims Blonde planned to steal the diamonds. Eddie shoots Nash and accuses Orange of lying."),
        EventNode(id="EVT_JOE_ACCUSES_ORANGE", fabula_time=13, syuzhet_index=13, event_type="choice", actor_id="ENT_JOE", description="Joe arrives, reports Blue killed by police, and identifies Orange as the traitor. He draws on Orange to execute him."),
        EventNode(id="EVT_MEXICAN_STANDOFF", fabula_time=14, syuzhet_index=14, event_type="outcome", actor_id=None, description="White defends Orange at gunpoint against Joe. Eddie aims at White. All three fire — Joe and Eddie are killed, White and Orange wounded."),
        EventNode(id="EVT_PINK_FLEES", fabula_time=15, syuzhet_index=15, event_type="choice", actor_id="ENT_PINK", description="Pink grabs the diamonds and flees. A crash and gunshots are heard outside."),
        EventNode(id="EVT_ORANGE_CONFESSES", fabula_time=16, syuzhet_index=16, event_type="choice", actor_id="ENT_ORANGE", description="White cradles the dying Orange, who confesses he is a police officer."),
        EventNode(id="EVT_WHITE_KILLED", fabula_time=17, syuzhet_index=17, event_type="outcome", actor_id=None, description="White presses his gun to Orange's head. Police storm the warehouse and shoot White dead."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="EVT_BLONDE_PAROLED", target_id="EVT_TEAM_ASSEMBLED", mechanism="social"),
        CausalEdge(source_id="EVT_ORANGE_UNDERCOVER", target_id="EVT_TEAM_ASSEMBLED", mechanism="epistemic"),
        CausalEdge(source_id="ENT_JOE", target_id="EVT_TEAM_ASSEMBLED", mechanism="social"),
        CausalEdge(source_id="EVT_TEAM_ASSEMBLED", target_id="EVT_HEIST_GOES_WRONG", mechanism="social"),
        CausalEdge(source_id="ENT_ORANGE", target_id="EVT_HEIST_GOES_WRONG", mechanism="epistemic"),
        CausalEdge(source_id="ENT_BLONDE", target_id="EVT_HEIST_GOES_WRONG", mechanism="physical"),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="EVT_BROWN_KILLED", mechanism="physical"),
        CausalEdge(source_id="EVT_HEIST_GOES_WRONG", target_id="EVT_ORANGE_SHOT", mechanism="physical"),
        CausalEdge(source_id="EVT_ORANGE_SHOT", target_id="EVT_WAREHOUSE_RENDEZVOUS", mechanism="physical"),
        CausalEdge(source_id="OBJ_DIAMONDS", target_id="EVT_WAREHOUSE_RENDEZVOUS", mechanism="social"),
        CausalEdge(source_id="EVT_WAREHOUSE_RENDEZVOUS", target_id="EVT_BLONDE_BRINGS_NASH", mechanism="social"),
        CausalEdge(source_id="ENT_BLONDE", target_id="EVT_BLONDE_BRINGS_NASH", mechanism="physical"),
        CausalEdge(source_id="EVT_BLONDE_BRINGS_NASH", target_id="EVT_NASH_TORTURED", mechanism="physical"),
        CausalEdge(source_id="EVT_NASH_TORTURED", target_id="EVT_EAR_SCENE", mechanism="physical"),
        CausalEdge(source_id="OBJ_STRAIGHT_RAZOR", target_id="EVT_EAR_SCENE", mechanism="physical"),
        CausalEdge(source_id="EVT_EAR_SCENE", target_id="EVT_ORANGE_KILLS_BLONDE", mechanism="physical"),
        CausalEdge(source_id="ENT_ORANGE", target_id="EVT_ORANGE_KILLS_BLONDE", mechanism="psychological"),
        CausalEdge(source_id="EVT_ORANGE_KILLS_BLONDE", target_id="EVT_EDDIE_KILLS_NASH", mechanism="social"),
        CausalEdge(source_id="ENT_EDDIE", target_id="EVT_EDDIE_KILLS_NASH", mechanism="psychological"),
        CausalEdge(source_id="EVT_EDDIE_KILLS_NASH", target_id="EVT_JOE_ACCUSES_ORANGE", mechanism="social"),
        CausalEdge(source_id="ENT_JOE", target_id="EVT_JOE_ACCUSES_ORANGE", mechanism="epistemic"),
        CausalEdge(source_id="EVT_JOE_ACCUSES_ORANGE", target_id="EVT_MEXICAN_STANDOFF", mechanism="physical"),
        CausalEdge(source_id="ENT_WHITE", target_id="EVT_MEXICAN_STANDOFF", mechanism="psychological"),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="EVT_PINK_FLEES", mechanism="social"),
        CausalEdge(source_id="OBJ_DIAMONDS", target_id="EVT_PINK_FLEES", mechanism="psychological"),
        CausalEdge(source_id="EVT_MEXICAN_STANDOFF", target_id="EVT_ORANGE_CONFESSES", mechanism="psychological"),
        CausalEdge(source_id="EVT_ORANGE_CONFESSES", target_id="EVT_WHITE_KILLED", mechanism="psychological"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_WHITE", target_entity_id="ENT_ORANGE", affinity=0.8, friction=0.2, power_dynamic=0.3, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_ORANGE", target_entity_id="ENT_WHITE", affinity=0.5, friction=0.5, power_dynamic=-0.3, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_WHITE", target_entity_id="ENT_JOE", affinity=0.5, friction=0.5, power_dynamic=-0.4, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_WHITE", target_entity_id="ENT_BLONDE", affinity=-0.7, friction=0.9, power_dynamic=0.0, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_BLONDE", target_entity_id="ENT_JOE", affinity=0.85, friction=0.1, power_dynamic=-0.5, inertia=0.9),
        RelationshipEdge(source_entity_id="ENT_JOE", target_entity_id="ENT_BLONDE", affinity=0.7, friction=0.3, power_dynamic=0.5, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_JOE", target_entity_id="ENT_EDDIE", affinity=0.8, friction=0.2, power_dynamic=0.4, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_EDDIE", target_entity_id="ENT_JOE", affinity=0.85, friction=0.2, power_dynamic=-0.4, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_PINK", target_entity_id="ENT_WHITE", affinity=0.1, friction=0.7, power_dynamic=0.0, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_BLONDE", target_entity_id="ENT_NASH", affinity=-0.9, friction=1.0, power_dynamic=0.9, inertia=0.3),
    ],
)
