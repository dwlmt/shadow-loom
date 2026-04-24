from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, SpatialEdge, InformationEdge, RelationshipEdge, TraitVector,
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
            name="Kent Coastal Marshes",
            description="Kent Coastal Marshes",
            ambient_state={"desolation": {"value": 0.7, "volatility": 0.3}, "danger": {"value": 0.5, "volatility": 0.5}},
        ),
                "LOC_CHURCHYARD": Location(
            name="Churchyard (Parents' Graves)",
            description="Churchyard (Parents' Graves)",
            ambient_state={"isolation": {"value": 0.8, "volatility": 0.2}},
        ),
                "LOC_FORGE": Location(
            name="Joe Gargery's Forge & Home",
            description="Joe Gargery's Forge & Home",
            ambient_state={"warmth": {"value": 0.7, "volatility": 0.3}, "simplicity": {"value": 0.8, "volatility": 0.1}},
        ),
                "LOC_SATIS_HOUSE": Location(
            name="Satis House (Miss Havisham's Manor)",
            description="Satis House (Miss Havisham's Manor) (dilapidated)",
            ambient_state={"decay": {"value": 0.9, "volatility": 0.1}, "obsession": {"value": 0.95, "volatility": 0.05}},
        ),
                "LOC_LONDON": Location(
            name="London",
            description="London",
            ambient_state={"grime": {"value": 0.7, "volatility": 0.2}, "ambition": {"value": 0.7, "volatility": 0.4}},
        ),
                "LOC_BARNARDS_INN": Location(
            name="Barnard's Inn (Pip & Herbert's Rooms)",
            description="Barnard's Inn (Pip & Herbert's Rooms)",
            ambient_state={"camaraderie": {"value": 0.7, "volatility": 0.3}},
        ),
                "LOC_JAGGERS_OFFICE": Location(
            name="Mr Jaggers' Office",
            description="Mr Jaggers' Office",
            ambient_state={"authority": {"value": 0.8, "volatility": 0.1}},
        ),
                "LOC_RIVER_THAMES": Location(
            name="River Thames (Escape Attempt)",
            description="River Thames (Escape Attempt)",
            ambient_state={"danger": {"value": 0.8, "volatility": 0.5}},
        ),
                "LOC_SLUICE_HOUSE": Location(
            name="Sluice-house on the Marshes",
            description="Sluice-house on the Marshes",
            ambient_state={"danger": {"value": 0.9, "volatility": 0.3}},
        ),
                "LOC_CAIRO": Location(
            name="Cairo, Egypt (Clarriker's Office)",
            description="Cairo, Egypt (Clarriker's Office)",
            ambient_state={"fresh_start": {"value": 0.7, "volatility": 0.3}},
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
                Belief(target_id="ENT_HAVISHAM", perceived_state="Miss Havisham is my secret benefactress and intends me for Estella", confidence=0.95, inertia=0.9, established_at_fabula=7),
                Belief(target_id="ENT_HAVISHAM", perceived_state="Miss Havisham was my secret benefactress", confidence=0.0, inertia=0.1, established_at_fabula=0),
                Belief(target_id="ENT_MAGWITCH", perceived_state="Magwitch is my true benefactor", confidence=1.0, inertia=0.9, established_at_fabula=0),
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
                Belief(target_id="ENT_PIP", perceived_state="Pip is the boy I raised — I love him regardless of what he becomes", confidence=0.95, inertia=0.95, established_at_fabula=0),
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
                Belief(target_id="ENT_PIP", perceived_state="I ask for Pip's forgiveness — suffering has opened my heart", confidence=0.9, inertia=0.7, established_at_fabula=0),
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
                Belief(target_id="ENT_COMPEYSON", perceived_state="My fiancé betrayed me — all men are deceivers who break hearts", confidence=0.95, inertia=0.9, established_at_fabula=0),
                Belief(target_id="ENT_ESTELLA", perceived_state="Estella is my instrument of revenge against all men", confidence=0.9, inertia=0.8, established_at_fabula=4),
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
                Belief(target_id="ENT_PIP", perceived_state="The boy who helped me will become a true gentleman worthy of my fortune", confidence=0.95, inertia=0.9, established_at_fabula=1),
                Belief(target_id="ENT_PIP", perceived_state="Pip is the gentleman I created — he is my life's purpose and motivation", confidence=1.0, inertia=0.95, established_at_fabula=0),
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
                Belief(target_id="ENT_PIP", perceived_state="Pip has always been favoured over me — he deserves punishment", confidence=0.9, inertia=0.8, established_at_fabula=0),
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
        EventNode(id="EVT_CONVICT_ENCOUNTER", fabula_time=1, syuzhet_index=1, event_type="outcome", actor_ids=["ENT_MAGWITCH"], description="Young Pip encounters escaped convict Magwitch in the churchyard. Magwitch threatens him for food and tools."),
        EventNode(id="EVT_PIP_STEALS_FOOD", fabula_time=2, syuzhet_index=2, event_type="choice", actor_ids=["ENT_PIP"], description="Pip steals a file and food from home and delivers them to Magwitch."),
        EventNode(id="EVT_CONVICT_RECAPTURED", fabula_time=3, syuzhet_index=3, event_type="outcome", actor_ids=[], description="Magwitch is recaptured fighting with Compeyson in the marshes. He claims he stole the food to protect Pip."),
        EventNode(id="EVT_SATIS_HOUSE_VISITS", fabula_time=4, syuzhet_index=4, event_type="outcome", actor_ids=["ENT_PIP"], description="Pip visits Miss Havisham at Satis House and falls in love with Estella, who is cold and aloof."),
        EventNode(id="EVT_PIP_APPRENTICED", fabula_time=5, syuzhet_index=5, event_type="outcome", actor_ids=["ENT_HAVISHAM"], description="Miss Havisham gives Pip money to become Joe's apprentice blacksmith."),
        EventNode(id="EVT_MRS_JOE_ATTACKED", fabula_time=6, syuzhet_index=6, event_type="outcome", actor_ids=["ENT_ORLICK"], target_ids=["ENT_MRS_JOE"], description="Orlick attacks Mrs Joe with the convict's leg iron, leaving her unable to speak or work."),
        EventNode(id="EVT_GREAT_EXPECTATIONS", fabula_time=7, syuzhet_index=7, event_type="revelation", actor_ids=["ENT_JAGGERS"], description="Jaggers informs Pip of an anonymous benefactor funding his life as a gentleman in London. Pip assumes it is Miss Havisham."),
        EventNode(id="EVT_PIP_GOES_TO_LONDON", fabula_time=8, syuzhet_index=8, event_type="choice", actor_ids=["ENT_PIP"], description="Pip moves to London, rooms with Herbert Pocket, and begins his gentleman's education."),
        EventNode(id="EVT_PIP_ASHAMED_OF_JOE", fabula_time=9, syuzhet_index=9, event_type="outcome", actor_ids=["ENT_PIP"], description="When Joe visits London, Pip is ashamed to be seen with him."),
        EventNode(id="EVT_ESTELLA_IN_SOCIETY", fabula_time=10, syuzhet_index=10, event_type="outcome", actor_ids=["ENT_ESTELLA"], description="Estella is introduced into society in Richmond. Pip adores her; she warns him about Drummle."),
        EventNode(id="EVT_BENEFACTOR_REVEALED", fabula_time=11, syuzhet_index=11, event_type="revelation", actor_ids=["ENT_MAGWITCH"], description="Magwitch returns to England and reveals himself as Pip's true benefactor, shattering Pip's assumptions."),
        EventNode(id="EVT_COMPEYSON_IDENTITY", fabula_time=12, syuzhet_index=12, event_type="revelation", actor_ids=["ENT_MAGWITCH"], description="Magwitch reveals Compeyson was Miss Havisham's fraudulent fiancé."),
        EventNode(id="EVT_ESTELLA_MARRIES_DRUMMLE", fabula_time=13, syuzhet_index=13, event_type="outcome", actor_ids=["ENT_ESTELLA"], description="Estella coldly tells Pip she plans to marry Drummle. Pip declares his love to no avail."),
        EventNode(id="EVT_HAVISHAM_REMORSE", fabula_time=14, syuzhet_index=14, event_type="outcome", actor_ids=["ENT_HAVISHAM"], description="Miss Havisham shows remorse for raising Estella heartless. She funds Herbert's position. Her dress catches fire; Pip injures himself trying to save her."),
        EventNode(id="EVT_ESTELLA_PARENTAGE", fabula_time=15, syuzhet_index=15, event_type="revelation", actor_ids=["ENT_PIP"], description="Pip deduces that Estella is the daughter of Magwitch and Molly (Jaggers' housekeeper, a former convict acquitted of murder)."),
        EventNode(id="EVT_ORLICK_AMBUSH", fabula_time=16, syuzhet_index=16, event_type="choice", actor_ids=["ENT_ORLICK"], target_ids=["ENT_PIP"], description="Orlick lures Pip to the sluice-house, confesses to attacking Mrs Joe, and attempts to murder Pip. Herbert and Startop rescue him."),
        EventNode(id="EVT_ESCAPE_ATTEMPT", fabula_time=17, syuzhet_index=17, event_type="choice", actor_ids=["ENT_PIP"], description="Pip and Herbert attempt to smuggle Magwitch out of England by boat on the Thames."),
        EventNode(id="EVT_MAGWITCH_CAPTURED", fabula_time=18, syuzhet_index=18, event_type="outcome", actor_ids=["ENT_COMPEYSON"], description="A police boat carrying Compeyson intercepts them. Magwitch and Compeyson fight in the river. Compeyson drowns; Magwitch is critically injured."),
        EventNode(id="EVT_MAGWITCH_DIES", fabula_time=19, syuzhet_index=19, event_type="outcome", actor_ids=[], description="Pip visits dying Magwitch in prison, telling him his daughter Estella is alive."),
        EventNode(id="EVT_PIP_FALLS_ILL", fabula_time=20, syuzhet_index=20, event_type="outcome", actor_ids=[], description="Pip falls gravely ill. Joe nurses him back to health and pays his debts."),
        EventNode(id="EVT_BIDDY_MARRIES_JOE", fabula_time=21, syuzhet_index=21, event_type="outcome", actor_ids=[], description="Pip returns to propose to Biddy, but finds she has married Joe."),
        EventNode(id="EVT_PIP_TO_CAIRO", fabula_time=22, syuzhet_index=22, event_type="choice", actor_ids=["ENT_PIP"], description="Pip moves to Cairo to work at Clarriker's with Herbert and Clara."),
        EventNode(id="EVT_REUNION_ESTELLA", fabula_time=23, syuzhet_index=23, event_type="outcome", actor_ids=[], description="Eleven years later, Pip meets the widowed Estella at the ruins of Satis House. She asks forgiveness; they leave together."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="EVT_CONVICT_ENCOUNTER", target_id="EVT_PIP_STEALS_FOOD", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=1),
        CausalEdge(source_id="EVT_PIP_STEALS_FOOD", target_id="EVT_CONVICT_RECAPTURED", causality_type="chain_reaction", mechanism="physical", evidence_strength="moderate", causal_force=5.0, fabula_time=2),
        CausalEdge(source_id="EVT_SATIS_HOUSE_VISITS", target_id="EVT_ESTELLA_IN_SOCIETY", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=4),
        CausalEdge(source_id="EVT_SATIS_HOUSE_VISITS", target_id="EVT_PIP_APPRENTICED", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=4),
        CausalEdge(source_id="EVT_PIP_STEALS_FOOD", target_id="EVT_GREAT_EXPECTATIONS", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=2),
        CausalEdge(source_id="EVT_PIP_GOES_TO_LONDON", target_id="EVT_PIP_ASHAMED_OF_JOE", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=8),
        CausalEdge(source_id="EVT_GREAT_EXPECTATIONS", target_id="EVT_BENEFACTOR_REVEALED", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=7),
        CausalEdge(source_id="EVT_BENEFACTOR_REVEALED", target_id="EVT_COMPEYSON_IDENTITY", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=11),
        CausalEdge(source_id="EVT_BENEFACTOR_REVEALED", target_id="EVT_HAVISHAM_REMORSE", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=11),
        CausalEdge(source_id="EVT_HAVISHAM_REMORSE", target_id="EVT_ESTELLA_PARENTAGE", causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate", causal_force=5.0, fabula_time=14),
        CausalEdge(source_id="EVT_BENEFACTOR_REVEALED", target_id="EVT_ESCAPE_ATTEMPT", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=11),
        CausalEdge(source_id="EVT_MAGWITCH_CAPTURED", target_id="EVT_MAGWITCH_DIES", causality_type="chain_reaction", mechanism="physical", evidence_strength="moderate", causal_force=5.0, fabula_time=18),
        CausalEdge(source_id="EVT_MAGWITCH_DIES", target_id="EVT_PIP_FALLS_ILL", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=19),
        CausalEdge(source_id="EVT_PIP_FALLS_ILL", target_id="EVT_BIDDY_MARRIES_JOE", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=20),
        CausalEdge(source_id="EVT_BIDDY_MARRIES_JOE", target_id="EVT_PIP_TO_CAIRO", causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=21),
        CausalEdge(source_id="EVT_PIP_TO_CAIRO", target_id="EVT_REUNION_ESTELLA", causality_type="chain_reaction", mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=22),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_BARNARDS_INN", target_id="LOC_LONDON"),
        SpatialEdge(source_id="LOC_CHURCHYARD", target_id="LOC_KENT_MARSHES"),
        SpatialEdge(source_id="LOC_FORGE", target_id="LOC_KENT_MARSHES"),
        SpatialEdge(source_id="LOC_FORGE", target_id="LOC_SATIS_HOUSE"),
        SpatialEdge(source_id="LOC_JAGGERS_OFFICE", target_id="LOC_LONDON"),
        SpatialEdge(source_id="LOC_KENT_MARSHES", target_id="LOC_LONDON"),
        SpatialEdge(source_id="LOC_KENT_MARSHES", target_id="LOC_SATIS_HOUSE"),
        SpatialEdge(source_id="LOC_KENT_MARSHES", target_id="LOC_SLUICE_HOUSE"),
        SpatialEdge(source_id="LOC_LONDON", target_id="LOC_RIVER_THAMES"),
        SpatialEdge(source_id="LOC_LONDON", target_id="LOC_CAIRO"),
    ],
    information_topology=[
        InformationEdge(
            source_id="ENT_JAGGERS",
            target_ids=["ENT_PIP"],
            medium="letter",
            established_at_fabula=7,
            terminated_at_fabula=7,
        ),
        InformationEdge(
            source_id="ENT_HERBERT",
            target_ids=["ENT_PIP"],
            medium="conversation",
            established_at_fabula=8,
            terminated_at_fabula=8,
        ),
        InformationEdge(
            source_id="ENT_MAGWITCH",
            target_ids=["ENT_PIP"],
            medium="confession",
            established_at_fabula=14,
        ),
    ],
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_JOE", affinity=0.8, fear=0.15, power_dynamic=0.1),
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_ESTELLA", affinity=0.9, fear=0.35, power_dynamic=-0.5),
        RelationshipEdge(source_entity_id="ENT_ESTELLA", target_entity_id="ENT_PIP", affinity=0.3, fear=0.25, power_dynamic=0.5),
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_MAGWITCH", affinity=0.6, fear=0.2, power_dynamic=0.1),
        RelationshipEdge(source_entity_id="ENT_MAGWITCH", target_entity_id="ENT_PIP", affinity=0.95, fear=0.05, power_dynamic=-0.2),
        RelationshipEdge(source_entity_id="ENT_HAVISHAM", target_entity_id="ENT_ESTELLA", affinity=0.6, fear=0.35, power_dynamic=0.8),
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_HERBERT", affinity=0.85, fear=0.05, power_dynamic=0.0),
        RelationshipEdge(source_entity_id="ENT_MAGWITCH", target_entity_id="ENT_COMPEYSON", affinity=-0.95, fear=0.47, power_dynamic=-0.3),
        RelationshipEdge(source_entity_id="ENT_ORLICK", target_entity_id="ENT_PIP", affinity=-0.8, fear=0.45, power_dynamic=0.2),
        RelationshipEdge(source_entity_id="ENT_ORLICK", target_entity_id="ENT_MRS_JOE", affinity=-0.7, fear=0.4, power_dynamic=0.3),
        RelationshipEdge(source_entity_id="ENT_PIP", target_entity_id="ENT_BIDDY", affinity=0.6, fear=0.15, power_dynamic=0.1),
        RelationshipEdge(source_entity_id="ENT_JOE", target_entity_id="ENT_MRS_JOE", affinity=0.6, fear=0.3, power_dynamic=-0.4),
        RelationshipEdge(source_entity_id="ENT_JAGGERS", target_entity_id="ENT_PIP", affinity=0.3, fear=0.15, power_dynamic=0.6),
        RelationshipEdge(source_entity_id="ENT_ESTELLA", target_entity_id="ENT_DRUMMLE", affinity=0.1, fear=0.25, power_dynamic=-0.3),
        RelationshipEdge(source_entity_id="ENT_HAVISHAM", target_entity_id="ENT_PIP", affinity=0.2, fear=0.1, power_dynamic=0.7),
        RelationshipEdge(source_entity_id="ENT_COMPEYSON", target_entity_id="ENT_MAGWITCH", affinity=-0.8, fear=0.3, power_dynamic=0.4),
    ],
)
