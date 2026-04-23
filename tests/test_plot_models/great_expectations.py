from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
    Affordance, Belief,
)

# =============================================================================
# GREAT EXPECTATIONS — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
        "LOC_KENT_MARSHES": Location(
            id="LOC_KENT_MARSHES",
            name="Kent Coastal Marshes",
            connected_locations=["LOC_FORGE", "LOC_SATIS_HOUSE", "LOC_CHURCHYARD"],
            ambient_states={
                "desolation": AmbientVector(value=0.7, volatility=0.3),
                "danger": AmbientVector(value=0.5, volatility=0.5),
            },
        ),
        "LOC_CHURCHYARD": Location(
            id="LOC_CHURCHYARD",
            name="Churchyard (Parents' Graves)",
            connected_locations=["LOC_KENT_MARSHES"],
            ambient_states={
                "isolation": AmbientVector(value=0.8, volatility=0.2),
            },
        ),
        "LOC_FORGE": Location(
            id="LOC_FORGE",
            name="Joe Gargery's Forge & Home",
            connected_locations=["LOC_KENT_MARSHES", "LOC_SATIS_HOUSE"],
            ambient_states={
                "warmth": AmbientVector(value=0.7, volatility=0.3),
                "simplicity": AmbientVector(value=0.8, volatility=0.1),
            },
        ),
        "LOC_SATIS_HOUSE": Location(
            id="LOC_SATIS_HOUSE",
            name="Satis House (Miss Havisham's Manor)",
            connected_locations=["LOC_KENT_MARSHES", "LOC_FORGE"],
            ambient_states={
                "decay": AmbientVector(value=0.9, volatility=0.1),
                "obsession": AmbientVector(value=0.95, volatility=0.05),
            },
            constants=["dilapidated"],
        ),
        "LOC_LONDON": Location(
            id="LOC_LONDON",
            name="London",
            connected_locations=["LOC_BARNARDS_INN", "LOC_JAGGERS_OFFICE", "LOC_KENT_MARSHES"],
            ambient_states={
                "grime": AmbientVector(value=0.7, volatility=0.2),
                "ambition": AmbientVector(value=0.7, volatility=0.4),
            },
        ),
        "LOC_BARNARDS_INN": Location(
            id="LOC_BARNARDS_INN",
            name="Barnard's Inn (Pip & Herbert's Rooms)",
            connected_locations=["LOC_LONDON"],
            ambient_states={
                "camaraderie": AmbientVector(value=0.7, volatility=0.3),
            },
        ),
        "LOC_JAGGERS_OFFICE": Location(
            id="LOC_JAGGERS_OFFICE",
            name="Mr Jaggers' Office",
            connected_locations=["LOC_LONDON"],
            ambient_states={
                "authority": AmbientVector(value=0.8, volatility=0.1),
            },
        ),
        "LOC_RIVER_THAMES": Location(
            id="LOC_RIVER_THAMES",
            name="River Thames (Escape Attempt)",
            connected_locations=["LOC_LONDON"],
            ambient_states={
                "danger": AmbientVector(value=0.8, volatility=0.5),
            },
        ),
        "LOC_SLUICE_HOUSE": Location(
            id="LOC_SLUICE_HOUSE",
            name="Sluice-house on the Marshes",
            connected_locations=["LOC_KENT_MARSHES"],
            ambient_states={
                "danger": AmbientVector(value=0.9, volatility=0.3),
            },
        ),
        "LOC_CAIRO": Location(
            id="LOC_CAIRO",
            name="Cairo, Egypt (Clarriker's Office)",
            connected_locations=[],
            ambient_states={
                "fresh_start": AmbientVector(value=0.7, volatility=0.3),
            },
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_FILE": NarrativeObject(
            id="OBJ_FILE",
            name="Joe's File (Stolen by Pip)",
            location_id=None,
            owner_id=None,
            properties={"state": "returned_via_convict"},
            affordances=[
                Affordance(action="cut_shackles", target_type="NarrativeObject"),
            ],
        ),
        "OBJ_LEG_IRON": NarrativeObject(
            id="OBJ_LEG_IRON",
            name="Convict's Leg Iron",
            location_id="LOC_FORGE",
            owner_id=None,
            properties={"state": "used_as_weapon"},
            affordances=[
                Affordance(action="strike", target_type="Entity"),
            ],
        ),
        "OBJ_WEDDING_DRESS": NarrativeObject(
            id="OBJ_WEDDING_DRESS",
            name="Miss Havisham's Wedding Dress",
            location_id="LOC_SATIS_HOUSE",
            owner_id="ENT_HAVISHAM",
            properties={"state": "decaying", "worn_since": "jilting"},
            affordances=[
                Affordance(action="symbolize_grief", target_type="Entity"),
            ],
        ),
        "OBJ_MONEY_ALLOWANCE": NarrativeObject(
            id="OBJ_MONEY_ALLOWANCE",
            name="Pip's Gentleman's Allowance",
            location_id=None,
            owner_id="ENT_PIP",
            properties={"source": "Magwitch", "amount": "500_per_annum"},
            affordances=[
                Affordance(action="fund", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_PIP": Entity(
            id="ENT_PIP",
            name="Philip 'Pip' Pirrip",
            location_id="LOC_CAIRO",
            status="healthy",
            traits={
                "ambition": TraitVector(value=0.8, inertia=0.6),
                "shame": TraitVector(value=0.7, inertia=0.5),
                "gratitude": TraitVector(value=0.7, inertia=0.6),
                "snobbery": TraitVector(value=0.5, inertia=0.3),
                "compassion": TraitVector(value=0.75, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_HAVISHAM", perceived_state="Miss Havisham was my secret benefactress", confidence=0.0, inertia=0.1),
                Belief(target_id="ENT_MAGWITCH", perceived_state="Magwitch is my true benefactor", confidence=1.0, inertia=0.9),
            ],
        ),
        "ENT_JOE": Entity(
            id="ENT_JOE",
            name="Joe Gargery",
            location_id="LOC_FORGE",
            status="healthy",
            traits={
                "kindness": TraitVector(value=0.95, inertia=0.95),
                "humility": TraitVector(value=0.9, inertia=0.9),
                "loyalty": TraitVector(value=0.95, inertia=0.95),
                "simplicity": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_PIP", perceived_state="Pip is the boy I raised — I love him regardless of what he becomes", confidence=0.95, inertia=0.95),
            ],
        ),
        "ENT_ESTELLA": Entity(
            id="ENT_ESTELLA",
            name="Estella",
            location_id="LOC_SATIS_HOUSE",
            status="healthy",
            traits={
                "coldness": TraitVector(value=0.6, inertia=0.4),
                "beauty": TraitVector(value=0.9, inertia=0.8),
                "emotional_growth": TraitVector(value=0.6, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_PIP", perceived_state="I ask for Pip's forgiveness — suffering has opened my heart", confidence=0.9, inertia=0.7),
            ],
        ),
        "ENT_HAVISHAM": Entity(
            id="ENT_HAVISHAM",
            name="Miss Havisham",
            location_id="LOC_SATIS_HOUSE",
            status="dead",
            traits={
                "bitterness": TraitVector(value=0.95, inertia=0.8),
                "manipulation": TraitVector(value=0.85, inertia=0.7),
                "remorse": TraitVector(value=0.7, inertia=0.4),
            },
            beliefs=[
                Belief(target_id="ENT_COMPEYSON", perceived_state="My fiancé betrayed me — all men are deceivers who break hearts", confidence=0.95, inertia=0.9),
                Belief(target_id="ENT_ESTELLA", perceived_state="Estella is my instrument of revenge against men", confidence=0.9, inertia=0.7),
            ],
        ),
        "ENT_MAGWITCH": Entity(
            id="ENT_MAGWITCH",
            name="Abel Magwitch",
            location_id="LOC_LONDON",
            status="dead",
            traits={
                "gratitude": TraitVector(value=0.9, inertia=0.9),
                "determination": TraitVector(value=0.85, inertia=0.8),
                "generosity": TraitVector(value=0.8, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_PIP", perceived_state="Pip is the gentleman I created — he is my life's purpose and motivation", confidence=1.0, inertia=0.95),
            ],
        ),
        "ENT_HERBERT": Entity(
            id="ENT_HERBERT",
            name="Herbert Pocket",
            location_id="LOC_CAIRO",
            status="healthy",
            traits={
                "friendship": TraitVector(value=0.85, inertia=0.8),
                "optimism": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_JAGGERS": Entity(
            id="ENT_JAGGERS",
            name="Mr Jaggers",
            location_id="LOC_JAGGERS_OFFICE",
            status="healthy",
            traits={
                "authority": TraitVector(value=0.9, inertia=0.85),
                "inscrutability": TraitVector(value=0.85, inertia=0.8),
            },
        ),
        "ENT_COMPEYSON": Entity(
            id="ENT_COMPEYSON",
            name="Compeyson",
            location_id="LOC_RIVER_THAMES",
            status="dead",
            traits={
                "villainy": TraitVector(value=0.9, inertia=0.8),
                "deception": TraitVector(value=0.85, inertia=0.7),
            },
        ),
        "ENT_ORLICK": Entity(
            id="ENT_ORLICK",
            name="Dolge Orlick",
            location_id="LOC_SLUICE_HOUSE",
            status="healthy",
            traits={
                "malice": TraitVector(value=0.85, inertia=0.7),
                "jealousy": TraitVector(value=0.8, inertia=0.6),
                "violence": TraitVector(value=0.85, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_PIP", perceived_state="Pip has always been favoured over me — he deserves punishment", confidence=0.9, inertia=0.8),
            ],
        ),
        "ENT_BIDDY": Entity(
            id="ENT_BIDDY",
            name="Biddy",
            location_id="LOC_FORGE",
            status="healthy",
            traits={
                "kindness": TraitVector(value=0.85, inertia=0.8),
                "intelligence": TraitVector(value=0.75, inertia=0.7),
            },
        ),
        "ENT_DRUMMLE": Entity(
            id="ENT_DRUMMLE",
            name="Bentley Drummle",
            location_id="LOC_LONDON",
            status="dead",
            traits={
                "brutality": TraitVector(value=0.8, inertia=0.7),
                "cruelty": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_MRS_JOE": Entity(
            id="ENT_MRS_JOE",
            name="Mrs Joe Gargery",
            location_id="LOC_FORGE",
            status="dead",
            traits={
                "temper": TraitVector(value=0.85, inertia=0.7),
            },
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_CONVICT_ENCOUNTER", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_id="ENT_MAGWITCH", description="Young Pip encounters escaped convict Magwitch in the churchyard. Magwitch threatens him for food and tools."),
        EventNode(id="EVT_PIP_STEALS_FOOD", fabula_time=2, syuzhet_index=2, event_type="choice", actor_id="ENT_PIP", description="Pip steals a file and food from home and delivers them to Magwitch."),
        EventNode(id="EVT_CONVICT_RECAPTURED", fabula_time=3, syuzhet_index=3, event_type="outcome", actor_id=None, description="Magwitch is recaptured fighting with Compeyson in the marshes. He claims he stole the food to protect Pip."),
        EventNode(id="EVT_SATIS_HOUSE_VISITS", fabula_time=4, syuzhet_index=4, event_type="outcome", actor_id="ENT_PIP", description="Pip visits Miss Havisham at Satis House and falls in love with Estella, who is cold and aloof."),
        EventNode(id="EVT_PIP_APPRENTICED", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_id="ENT_HAVISHAM", description="Miss Havisham gives Pip money to become Joe's apprentice blacksmith."),
        EventNode(id="EVT_MRS_JOE_ATTACKED", fabula_time=6, syuzhet_index=6, event_type="outcome", actor_id="ENT_ORLICK", description="Orlick attacks Mrs Joe with the convict's leg iron, leaving her unable to speak or work."),
        EventNode(id="EVT_GREAT_EXPECTATIONS", fabula_time=7, syuzhet_index=7, event_type="revelation", actor_id="ENT_JAGGERS", description="Jaggers informs Pip of an anonymous benefactor funding his life as a gentleman in London. Pip assumes it is Miss Havisham."),
        EventNode(id="EVT_PIP_GOES_TO_LONDON", fabula_time=8, syuzhet_index=8, event_type="choice", actor_id="ENT_PIP", description="Pip moves to London, rooms with Herbert Pocket, and begins his gentleman's education."),
        EventNode(id="EVT_PIP_ASHAMED_OF_JOE", fabula_time=9, syuzhet_index=9, event_type="outcome", actor_id="ENT_PIP", description="When Joe visits London, Pip is ashamed to be seen with him."),
        EventNode(id="EVT_ESTELLA_IN_SOCIETY", fabula_time=10, syuzhet_index=10, event_type="outcome", actor_id="ENT_ESTELLA", description="Estella is introduced into society in Richmond. Pip adores her; she warns him about Drummle."),
        EventNode(id="EVT_BENEFACTOR_REVEALED", fabula_time=11, syuzhet_index=11, event_type="revelation", actor_id="ENT_MAGWITCH", description="Magwitch returns to England and reveals himself as Pip's true benefactor, shattering Pip's assumptions."),
        EventNode(id="EVT_COMPEYSON_IDENTITY", fabula_time=12, syuzhet_index=12, event_type="revelation", actor_id="ENT_MAGWITCH", description="Magwitch reveals Compeyson was Miss Havisham's fraudulent fiancé."),
        EventNode(id="EVT_ESTELLA_MARRIES_DRUMMLE", fabula_time=13, syuzhet_index=13, event_type="outcome", actor_id="ENT_ESTELLA", description="Estella coldly tells Pip she plans to marry Drummle. Pip declares his love to no avail."),
        EventNode(id="EVT_HAVISHAM_REMORSE", fabula_time=14, syuzhet_index=14, event_type="outcome", actor_id="ENT_HAVISHAM", description="Miss Havisham shows remorse for raising Estella heartless. She funds Herbert's position. Her dress catches fire; Pip injures himself trying to save her."),
        EventNode(id="EVT_ESTELLA_PARENTAGE", fabula_time=15, syuzhet_index=15, event_type="revelation", actor_id="ENT_PIP", description="Pip deduces that Estella is the daughter of Magwitch and Molly (Jaggers' housekeeper, a former convict acquitted of murder)."),
        EventNode(id="EVT_ORLICK_AMBUSH", fabula_time=16, syuzhet_index=16, event_type="choice", actor_id="ENT_ORLICK", description="Orlick lures Pip to the sluice-house, confesses to attacking Mrs Joe, and attempts to murder Pip. Herbert and Startop rescue him."),
        EventNode(id="EVT_ESCAPE_ATTEMPT", fabula_time=17, syuzhet_index=17, event_type="choice", actor_id="ENT_PIP", description="Pip and Herbert attempt to smuggle Magwitch out of England by boat on the Thames."),
        EventNode(id="EVT_MAGWITCH_CAPTURED", fabula_time=18, syuzhet_index=18, event_type="outcome", actor_id="ENT_COMPEYSON", description="A police boat carrying Compeyson intercepts them. Magwitch and Compeyson fight in the river. Compeyson drowns; Magwitch is critically injured."),
        EventNode(id="EVT_MAGWITCH_DIES", fabula_time=19, syuzhet_index=19, event_type="outcome", actor_id=None, description="Pip visits dying Magwitch in prison, telling him his daughter Estella is alive."),
        EventNode(id="EVT_PIP_FALLS_ILL", fabula_time=20, syuzhet_index=20, event_type="outcome", actor_id=None, description="Pip falls gravely ill. Joe nurses him back to health and pays his debts."),
        EventNode(id="EVT_BIDDY_MARRIES_JOE", fabula_time=21, syuzhet_index=21, event_type="outcome", actor_id=None, description="Pip returns to propose to Biddy, but finds she has married Joe."),
        EventNode(id="EVT_PIP_TO_CAIRO", fabula_time=22, syuzhet_index=22, event_type="choice", actor_id="ENT_PIP", description="Pip moves to Cairo to work at Clarriker's with Herbert and Clara."),
        EventNode(id="EVT_REUNION_ESTELLA", fabula_time=23, syuzhet_index=23, event_type="outcome", actor_id=None, description="Eleven years later, Pip meets the widowed Estella at the ruins of Satis House. She asks forgiveness; they leave together."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="ENT_MAGWITCH", target_id="EVT_CONVICT_ENCOUNTER", mechanism="physical"),
        CausalEdge(source_id="EVT_CONVICT_ENCOUNTER", target_id="EVT_PIP_STEALS_FOOD", mechanism="psychological"),
        CausalEdge(source_id="EVT_PIP_STEALS_FOOD", target_id="EVT_CONVICT_RECAPTURED", mechanism="physical"),
        CausalEdge(source_id="ENT_HAVISHAM", target_id="EVT_SATIS_HOUSE_VISITS", mechanism="social"),
        CausalEdge(source_id="EVT_SATIS_HOUSE_VISITS", target_id="EVT_ESTELLA_IN_SOCIETY", mechanism="social"),
        CausalEdge(source_id="EVT_SATIS_HOUSE_VISITS", target_id="EVT_PIP_APPRENTICED", mechanism="social"),
        CausalEdge(source_id="ENT_ORLICK", target_id="EVT_MRS_JOE_ATTACKED", mechanism="physical"),
        CausalEdge(source_id="OBJ_LEG_IRON", target_id="EVT_MRS_JOE_ATTACKED", mechanism="physical"),
        CausalEdge(source_id="EVT_PIP_STEALS_FOOD", target_id="EVT_GREAT_EXPECTATIONS", mechanism="psychological"),
        CausalEdge(source_id="ENT_MAGWITCH", target_id="EVT_GREAT_EXPECTATIONS", mechanism="social"),
        CausalEdge(source_id="OBJ_MONEY_ALLOWANCE", target_id="EVT_PIP_GOES_TO_LONDON", mechanism="social"),
        CausalEdge(source_id="EVT_PIP_GOES_TO_LONDON", target_id="EVT_PIP_ASHAMED_OF_JOE", mechanism="psychological"),
        CausalEdge(source_id="EVT_GREAT_EXPECTATIONS", target_id="EVT_BENEFACTOR_REVEALED", mechanism="epistemic"),
        CausalEdge(source_id="ENT_MAGWITCH", target_id="EVT_BENEFACTOR_REVEALED", mechanism="epistemic"),
        CausalEdge(source_id="ENT_HAVISHAM", target_id="EVT_ESTELLA_MARRIES_DRUMMLE", mechanism="psychological"),
        CausalEdge(source_id="ENT_ESTELLA", target_id="EVT_ESTELLA_MARRIES_DRUMMLE", mechanism="social"),
        CausalEdge(source_id="EVT_BENEFACTOR_REVEALED", target_id="EVT_COMPEYSON_IDENTITY", mechanism="epistemic"),
        CausalEdge(source_id="EVT_BENEFACTOR_REVEALED", target_id="EVT_HAVISHAM_REMORSE", mechanism="psychological"),
        CausalEdge(source_id="ENT_MAGWITCH", target_id="EVT_ESTELLA_PARENTAGE", mechanism="epistemic"),
        CausalEdge(source_id="EVT_HAVISHAM_REMORSE", target_id="EVT_ESTELLA_PARENTAGE", mechanism="epistemic"),
        CausalEdge(source_id="ENT_ORLICK", target_id="EVT_ORLICK_AMBUSH", mechanism="physical"),
        CausalEdge(source_id="EVT_BENEFACTOR_REVEALED", target_id="EVT_ESCAPE_ATTEMPT", mechanism="social"),
        CausalEdge(source_id="ENT_COMPEYSON", target_id="EVT_MAGWITCH_CAPTURED", mechanism="physical"),
        CausalEdge(source_id="EVT_MAGWITCH_CAPTURED", target_id="EVT_MAGWITCH_DIES", mechanism="physical"),
        CausalEdge(source_id="EVT_MAGWITCH_DIES", target_id="EVT_PIP_FALLS_ILL", mechanism="psychological"),
        CausalEdge(source_id="ENT_JOE", target_id="EVT_PIP_FALLS_ILL", mechanism="social"),
        CausalEdge(source_id="EVT_PIP_FALLS_ILL", target_id="EVT_BIDDY_MARRIES_JOE", mechanism="social"),
        CausalEdge(source_id="EVT_BIDDY_MARRIES_JOE", target_id="EVT_PIP_TO_CAIRO", mechanism="psychological"),
        CausalEdge(source_id="EVT_PIP_TO_CAIRO", target_id="EVT_REUNION_ESTELLA", mechanism="social"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_JOE", affinity=0.8, friction=0.3, power_dynamic=0.1, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_ESTELLA", affinity=0.9, friction=0.7, power_dynamic=-0.5, inertia=0.85),
        RelationshipEdge(source_entity_id="ENT_ESTELLA", target_entity_id="ENT_PIP", affinity=0.3, friction=0.5, power_dynamic=0.5, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_MAGWITCH", affinity=0.6, friction=0.4, power_dynamic=0.1, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_MAGWITCH", target_entity_id="ENT_PIP", affinity=0.95, friction=0.1, power_dynamic=-0.2, inertia=0.9),
        RelationshipEdge(source_entity_id="ENT_HAVISHAM", target_entity_id="ENT_ESTELLA", affinity=0.6, friction=0.7, power_dynamic=0.8, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_HERBERT", affinity=0.85, friction=0.1, power_dynamic=0.0, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_MAGWITCH", target_entity_id="ENT_COMPEYSON", affinity=-0.95, friction=0.95, power_dynamic=-0.3, inertia=0.9),
        RelationshipEdge(source_entity_id="ENT_ORLICK", target_entity_id="ENT_PIP", affinity=-0.8, friction=0.9, power_dynamic=0.2, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_ORLICK", target_entity_id="ENT_MRS_JOE", affinity=-0.7, friction=0.8, power_dynamic=0.3, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_BIDDY", affinity=0.6, friction=0.3, power_dynamic=0.1, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_JOE", target_entity_id="ENT_MRS_JOE", affinity=0.6, friction=0.6, power_dynamic=-0.4, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_JAGGERS", target_entity_id="ENT_PIP", affinity=0.3, friction=0.3, power_dynamic=0.6, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_ESTELLA", target_entity_id="ENT_DRUMMLE", affinity=0.1, friction=0.5, power_dynamic=-0.3, inertia=0.3),
    ],
)
