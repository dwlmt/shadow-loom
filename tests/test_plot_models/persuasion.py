from shadow_loom.models import (
    WorldStateV1, Location, NarrativeObject, Entity, EventNode,
    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,
    Affordance, Belief,
)

# =============================================================================
# PERSUASION — World State (Factual Timeline)
# Pass A: Ontology (Nouns) | Pass B: Chronology (Events) | Pass C: Topology (Edges)
# =============================================================================

world_state = WorldStateV1(

    # ── LOCATIONS ──────────────────────────────────────────────────────────
    locations={
        "LOC_KELLYNCH_HALL": Location(
            id="LOC_KELLYNCH_HALL",
            name="Kellynch Hall (Elliot Family Estate)",
            connected_locations=["LOC_UPPERCROSS", "LOC_BATH"],
            ambient_states={
                "decline": AmbientVector(value=0.7, volatility=0.3),
                "vanity": AmbientVector(value=0.8, volatility=0.1),
            },
        ),
        "LOC_UPPERCROSS": Location(
            id="LOC_UPPERCROSS",
            name="Uppercross Hall & Cottage (Musgrove Family)",
            connected_locations=["LOC_KELLYNCH_HALL", "LOC_LYME_REGIS"],
            ambient_states={
                "sociability": AmbientVector(value=0.7, volatility=0.3),
            },
        ),
        "LOC_LYME_REGIS": Location(
            id="LOC_LYME_REGIS",
            name="Lyme Regis (Cobb Seawall & Harville House)",
            connected_locations=["LOC_UPPERCROSS", "LOC_BATH"],
            ambient_states={
                "romance": AmbientVector(value=0.6, volatility=0.4),
                "danger": AmbientVector(value=0.5, volatility=0.6),
            },
        ),
        "LOC_BATH": Location(
            id="LOC_BATH",
            name="Bath",
            connected_locations=["LOC_KELLYNCH_HALL", "LOC_LYME_REGIS"],
            ambient_states={
                "social_performance": AmbientVector(value=0.8, volatility=0.2),
                "tension": AmbientVector(value=0.7, volatility=0.5),
            },
        ),
        "LOC_SHROPSHIRE": Location(
            id="LOC_SHROPSHIRE",
            name="Shropshire (Edward Wentworth's Home)",
            connected_locations=["LOC_BATH"],
            ambient_states={},
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────────
    objects={
        "OBJ_WENTWORTH_LETTER": NarrativeObject(
            id="OBJ_WENTWORTH_LETTER",
            name="Wentworth's Love Letter",
            location_id="LOC_BATH",
            owner_id="ENT_ANNE",
            properties={"state": "delivered", "content": "declaration_of_love"},
            affordances=[
                Affordance(action="declare_love", target_type="Entity"),
            ],
        ),
        "OBJ_PRIZE_WEALTH": NarrativeObject(
            id="OBJ_PRIZE_WEALTH",
            name="Wentworth's Prize Ship Wealth",
            location_id=None,
            owner_id="ENT_WENTWORTH",
            properties={"state": "accumulated"},
            affordances=[
                Affordance(action="legitimize_status", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────────
    entities={
        "ENT_ANNE": Entity(
            id="ENT_ANNE",
            name="Anne Elliot",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "constancy": TraitVector(value=0.95, inertia=0.95),
                "self_effacement": TraitVector(value=0.8, inertia=0.6),
                "intelligence": TraitVector(value=0.85, inertia=0.8),
                "emotional_depth": TraitVector(value=0.9, inertia=0.8),
                "composure": TraitVector(value=0.85, inertia=0.7),
            },
            beliefs=[
                Belief(target_id="ENT_WENTWORTH", perceived_state="I have always loved him — women do not give up their feelings", confidence=1.0, inertia=1.0),
                Belief(target_id="ENT_WILLIAM_ELLIOT", perceived_state="His character is opaque — I cannot judge him despite his refined manners", confidence=0.7, inertia=0.5),
            ],
        ),
        "ENT_WENTWORTH": Entity(
            id="ENT_WENTWORTH",
            name="Captain Frederick Wentworth",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "pride": TraitVector(value=0.8, inertia=0.6),
                "confidence": TraitVector(value=0.85, inertia=0.7),
                "ambition": TraitVector(value=0.8, inertia=0.7),
                "resentment": TraitVector(value=0.5, inertia=0.3),
                "devotion": TraitVector(value=0.9, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_ANNE", perceived_state="She was weak — she let herself be persuaded and lacked resolution", confidence=0.8, inertia=0.5),
            ],
        ),
        "ENT_SIR_WALTER": Entity(
            id="ENT_SIR_WALTER",
            name="Sir Walter Elliot",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "vanity": TraitVector(value=0.95, inertia=0.9),
                "extravagance": TraitVector(value=0.9, inertia=0.7),
                "snobbery": TraitVector(value=0.9, inertia=0.85),
            },
            beliefs=[
                Belief(target_id="ENT_WILLIAM_ELLIOT", perceived_state="William's attentions are genuine and will restore family fortunes", confidence=0.8, inertia=0.6),
            ],
        ),
        "ENT_ELIZABETH": Entity(
            id="ENT_ELIZABETH",
            name="Elizabeth Elliot",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "vanity": TraitVector(value=0.85, inertia=0.8),
                "snobbery": TraitVector(value=0.85, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_MRS_CLAY", perceived_state="Mrs Clay is my harmless companion — she has no designs on Father", confidence=0.85, inertia=0.7),
            ],
        ),
        "ENT_LADY_RUSSELL": Entity(
            id="ENT_LADY_RUSSELL",
            name="Lady Russell",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "prudence": TraitVector(value=0.85, inertia=0.7),
                "authority": TraitVector(value=0.7, inertia=0.6),
                "affection_for_anne": TraitVector(value=0.8, inertia=0.8),
            },
            beliefs=[
                Belief(target_id="ENT_WENTWORTH", perceived_state="He was an imprudent, unsuitable match for Anne", confidence=0.8, inertia=0.6),
            ],
        ),
        "ENT_WILLIAM_ELLIOT": Entity(
            id="ENT_WILLIAM_ELLIOT",
            name="William Elliot (Cousin / Heir)",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "charm": TraitVector(value=0.85, inertia=0.6),
                "calculation": TraitVector(value=0.9, inertia=0.8),
                "duplicity": TraitVector(value=0.85, inertia=0.7),
                "self_interest": TraitVector(value=0.9, inertia=0.85),
            },
            beliefs=[
                Belief(target_id="ENT_MRS_CLAY", perceived_state="She aims to marry Sir Walter — I must prevent this to protect my inheritance", confidence=0.85, inertia=0.7),
            ],
        ),
        "ENT_LOUISA": Entity(
            id="ENT_LOUISA",
            name="Louisa Musgrove",
            location_id="LOC_LYME_REGIS",
            status="healthy",
            traits={
                "impulsiveness": TraitVector(value=0.8, inertia=0.5),
                "determination": TraitVector(value=0.7, inertia=0.5),
            },
        ),
        "ENT_BENWICK": Entity(
            id="ENT_BENWICK",
            name="Captain Benwick",
            location_id="LOC_LYME_REGIS",
            status="healthy",
            traits={
                "sensitivity": TraitVector(value=0.85, inertia=0.7),
                "grief": TraitVector(value=0.7, inertia=0.4),
            },
        ),
        "ENT_HARVILLE": Entity(
            id="ENT_HARVILLE",
            name="Captain Harville",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "loyalty": TraitVector(value=0.8, inertia=0.8),
                "warmth": TraitVector(value=0.75, inertia=0.7),
            },
        ),
        "ENT_MRS_SMITH": Entity(
            id="ENT_MRS_SMITH",
            name="Mrs Smith (Anne's School Friend)",
            location_id="LOC_BATH",
            status="ill",
            traits={
                "resilience": TraitVector(value=0.8, inertia=0.7),
                "honesty": TraitVector(value=0.85, inertia=0.8),
            },
        ),
        "ENT_MRS_CLAY": Entity(
            id="ENT_MRS_CLAY",
            name="Mrs Clay (Elizabeth's Companion)",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "ambition": TraitVector(value=0.8, inertia=0.6),
                "manipulation": TraitVector(value=0.75, inertia=0.5),
            },
            beliefs=[
                Belief(target_id="ENT_SIR_WALTER", perceived_state="I can marry Sir Walter and secure my position", confidence=0.7, inertia=0.5),
            ],
        ),
        "ENT_CROFTS": Entity(
            id="ENT_CROFTS",
            name="Admiral & Mrs Croft",
            location_id="LOC_BATH",
            status="healthy",
            traits={
                "warmth": TraitVector(value=0.8, inertia=0.7),
            },
        ),
        "ENT_MARY": Entity(
            id="ENT_MARY",
            name="Mary Musgrove (née Elliot)",
            location_id="LOC_UPPERCROSS",
            status="healthy",
            traits={
                "self_pity": TraitVector(value=0.75, inertia=0.6),
                "hypochondria": TraitVector(value=0.7, inertia=0.5),
            },
        ),
    },

    # ── EVENTS (Chronological) ─────────────────────────────────────────────
    events=[
        EventNode(id="EVT_BROKEN_ENGAGEMENT", fabula_time=1, syuzhet_index=1, event_type="choice", actor_id="ENT_ANNE", description="Seven years ago, 19-year-old Anne broke her engagement to Wentworth, persuaded by Lady Russell that the match was imprudent."),
        EventNode(id="EVT_FINANCIAL_TROUBLE", fabula_time=9, syuzhet_index=2, event_type="outcome", actor_id=None, description="The Elliot family faces financial trouble from Sir Walter's lavish spending. They let Kellynch Hall."),
        EventNode(id="EVT_CROFTS_RENT_KELLYNCH", fabula_time=10, syuzhet_index=3, event_type="outcome", actor_id="ENT_CROFTS", description="Admiral Croft (Wentworth's brother-in-law) and his wife become tenants of Kellynch Hall."),
        EventNode(id="EVT_WENTWORTH_RETURNS", fabula_time=11, syuzhet_index=4, event_type="outcome", actor_id="ENT_WENTWORTH", description="Captain Wentworth, now wealthy from war prizes, visits his sister and crosses paths with Anne at Uppercross."),
        EventNode(id="EVT_LOUISA_HENRIETTA_INTEREST", fabula_time=12, syuzhet_index=5, event_type="outcome", actor_id=None, description="Both Louisa and Henrietta Musgrove show interest in Wentworth. Speculation about which he will marry."),
        EventNode(id="EVT_ANNE_OVERHEARS", fabula_time=13, syuzhet_index=6, event_type="revelation", actor_id="ENT_ANNE", description="Anne overhears Louisa tell Wentworth that Anne rejected Charles Musgrove. Anne realizes Wentworth hasn't forgiven her."),
        EventNode(id="EVT_LYME_VISIT", fabula_time=14, syuzhet_index=7, event_type="outcome", actor_id=None, description="The group visits Lyme Regis to see Captains Harville and Benwick. Anne attracts attention from a stranger (William Elliot)."),
        EventNode(id="EVT_LOUISA_FALLS", fabula_time=15, syuzhet_index=8, event_type="outcome", actor_id="ENT_LOUISA", description="Louisa insists on jumping from the Cobb seawall, falls and sustains a serious concussion. Anne organizes assistance."),
        EventNode(id="EVT_WENTWORTH_GUILT", fabula_time=16, syuzhet_index=9, event_type="revelation", actor_id="ENT_WENTWORTH", description="Wentworth is impressed by Anne's composure but feels guilty for encouraging Louisa's headstrong nature."),
        EventNode(id="EVT_ANNE_GOES_TO_BATH", fabula_time=17, syuzhet_index=10, event_type="outcome", actor_id="ENT_ANNE", description="Anne joins her father and sister in Bath. William Elliot flatters the family."),
        EventNode(id="EVT_WILLIAM_COURTS_ANNE", fabula_time=18, syuzhet_index=11, event_type="choice", actor_id="ENT_WILLIAM_ELLIOT", description="William Elliot pays special attention to Anne, but she finds his character opaque and difficult to judge."),
        EventNode(id="EVT_LOUISA_BENWICK_ENGAGED", fabula_time=19, syuzhet_index=12, event_type="revelation", actor_id=None, description="News arrives that Louisa is engaged to Captain Benwick, freeing Wentworth."),
        EventNode(id="EVT_WENTWORTH_TO_BATH", fabula_time=20, syuzhet_index=13, event_type="outcome", actor_id="ENT_WENTWORTH", description="Wentworth travels to Bath, where his jealousy is piqued by seeing William courting Anne."),
        EventNode(id="EVT_MRS_SMITH_REVEALS", fabula_time=21, syuzhet_index=14, event_type="revelation", actor_id="ENT_MRS_SMITH", description="Mrs Smith reveals William Elliot is a cold, calculating opportunist who exploited her late husband and refused to help her. His aim was to prevent Mrs Clay from marrying Sir Walter."),
        EventNode(id="EVT_HARVILLE_DEBATE", fabula_time=22, syuzhet_index=15, event_type="outcome", actor_id="ENT_ANNE", description="At the Musgroves' hotel, Anne and Harville debate the constancy of men vs women in love. Wentworth overhears."),
        EventNode(id="EVT_WENTWORTH_LETTER", fabula_time=23, syuzhet_index=16, event_type="choice", actor_id="ENT_WENTWORTH", description="Deeply moved, Wentworth writes Anne a letter declaring his enduring love."),
        EventNode(id="EVT_RECONCILIATION", fabula_time=24, syuzhet_index=17, event_type="outcome", actor_id=None, description="Anne and Wentworth reconcile, reaffirm their love, and renew their engagement."),
        EventNode(id="EVT_LADY_RUSSELL_ADMITS", fabula_time=25, syuzhet_index=18, event_type="outcome", actor_id="ENT_LADY_RUSSELL", description="Lady Russell admits she was wrong about Wentworth and endorses the engagement."),
        EventNode(id="EVT_WILLIAM_LEAVES", fabula_time=26, syuzhet_index=19, event_type="outcome", actor_id="ENT_WILLIAM_ELLIOT", description="William Elliot leaves Bath. Mrs Clay follows him and becomes his mistress, removing the danger of her marrying Sir Walter."),
        EventNode(id="EVT_MARRIAGE", fabula_time=27, syuzhet_index=20, event_type="outcome", actor_id=None, description="Anne and Wentworth marry. Wentworth helps Mrs Smith recover her assets."),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────────
    causal_topology=[
        CausalEdge(source_id="ENT_LADY_RUSSELL", target_id="EVT_BROKEN_ENGAGEMENT", mechanism="social"),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="EVT_WENTWORTH_RETURNS", mechanism="psychological"),
        CausalEdge(source_id="ENT_SIR_WALTER", target_id="EVT_FINANCIAL_TROUBLE", mechanism="social"),
        CausalEdge(source_id="EVT_FINANCIAL_TROUBLE", target_id="EVT_CROFTS_RENT_KELLYNCH", mechanism="social"),
        CausalEdge(source_id="EVT_CROFTS_RENT_KELLYNCH", target_id="EVT_WENTWORTH_RETURNS", mechanism="social"),
        CausalEdge(source_id="EVT_WENTWORTH_RETURNS", target_id="EVT_LOUISA_HENRIETTA_INTEREST", mechanism="social"),
        CausalEdge(source_id="EVT_WENTWORTH_RETURNS", target_id="EVT_ANNE_OVERHEARS", mechanism="epistemic"),
        CausalEdge(source_id="EVT_LOUISA_HENRIETTA_INTEREST", target_id="EVT_LYME_VISIT", mechanism="social"),
        CausalEdge(source_id="ENT_LOUISA", target_id="EVT_LOUISA_FALLS", mechanism="physical"),
        CausalEdge(source_id="EVT_LOUISA_FALLS", target_id="EVT_WENTWORTH_GUILT", mechanism="psychological"),
        CausalEdge(source_id="EVT_WENTWORTH_GUILT", target_id="EVT_WENTWORTH_TO_BATH", mechanism="psychological"),
        CausalEdge(source_id="EVT_LOUISA_FALLS", target_id="EVT_LOUISA_BENWICK_ENGAGED", mechanism="social"),
        CausalEdge(source_id="ENT_WILLIAM_ELLIOT", target_id="EVT_WILLIAM_COURTS_ANNE", mechanism="social"),
        CausalEdge(source_id="ENT_MRS_SMITH", target_id="EVT_MRS_SMITH_REVEALS", mechanism="epistemic"),
        CausalEdge(source_id="EVT_WENTWORTH_TO_BATH", target_id="EVT_HARVILLE_DEBATE", mechanism="social"),
        CausalEdge(source_id="EVT_HARVILLE_DEBATE", target_id="EVT_WENTWORTH_LETTER", mechanism="psychological"),
        CausalEdge(source_id="OBJ_WENTWORTH_LETTER", target_id="EVT_RECONCILIATION", mechanism="epistemic"),
        CausalEdge(source_id="EVT_RECONCILIATION", target_id="EVT_LADY_RUSSELL_ADMITS", mechanism="social"),
        CausalEdge(source_id="EVT_MRS_SMITH_REVEALS", target_id="EVT_WILLIAM_LEAVES", mechanism="social"),
        CausalEdge(source_id="ENT_MRS_CLAY", target_id="EVT_WILLIAM_LEAVES", mechanism="social"),
        CausalEdge(source_id="EVT_RECONCILIATION", target_id="EVT_MARRIAGE", mechanism="social"),
    ],

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────────
    social_topology=[
        RelationshipEdge(source_entity_id="ENT_ANNE", target_entity_id="ENT_WENTWORTH", affinity=0.95, friction=0.5, power_dynamic=-0.1, inertia=0.95),
        RelationshipEdge(source_entity_id="ENT_WENTWORTH", target_entity_id="ENT_ANNE", affinity=0.9, friction=0.5, power_dynamic=0.1, inertia=0.8),
        RelationshipEdge(source_entity_id="ENT_ANNE", target_entity_id="ENT_LADY_RUSSELL", affinity=0.7, friction=0.4, power_dynamic=-0.4, inertia=0.7),
        RelationshipEdge(source_entity_id="ENT_LADY_RUSSELL", target_entity_id="ENT_WENTWORTH", affinity=-0.3, friction=0.6, power_dynamic=0.3, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_WILLIAM_ELLIOT", target_entity_id="ENT_ANNE", affinity=0.6, friction=0.3, power_dynamic=0.2, inertia=0.3),
        RelationshipEdge(source_entity_id="ENT_ANNE", target_entity_id="ENT_WILLIAM_ELLIOT", affinity=0.2, friction=0.5, power_dynamic=-0.1, inertia=0.2),
        RelationshipEdge(source_entity_id="ENT_WENTWORTH", target_entity_id="ENT_WILLIAM_ELLIOT", affinity=-0.5, friction=0.7, power_dynamic=0.2, inertia=0.4),
        RelationshipEdge(source_entity_id="ENT_SIR_WALTER", target_entity_id="ENT_ANNE", affinity=0.2, friction=0.4, power_dynamic=0.7, inertia=0.6),
        RelationshipEdge(source_entity_id="ENT_ANNE", target_entity_id="ENT_MRS_SMITH", affinity=0.7, friction=0.1, power_dynamic=0.1, inertia=0.5),
        RelationshipEdge(source_entity_id="ENT_LOUISA", target_entity_id="ENT_WENTWORTH", affinity=0.6, friction=0.4, power_dynamic=-0.2, inertia=0.3),
    ],
)
