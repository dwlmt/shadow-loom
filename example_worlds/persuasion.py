# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persuasion — high-fidelity WorldStateV1 test fixture.

Authored against the current ingestion prompts. Demonstrates all five
CausalEdge modalities, per-axis ``RelationshipMetric``, explicit
``evidence_strength``, and named-latent WORLD_ traits (Regency rank,
the Napoleonic Wars / naval prize economy, primogeniture and entail)
wired as common-cause parents over the events they jointly drive.
"""
from shadow_loom.models import (
    Channel,
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge, RelationshipMetric, TraitVector, AmbientVector, Affordance, Belief, EntityStateSnapshot,
    GlobalTrait, WorldTraitSnapshot,
    NarrativeStyle,
    Concern, Proposition, ConcernSnapshot, PropositionSnapshot,
)

world_state = WorldStateV1(
    narrative_style=NarrativeStyle(
        format='synopsis',
        target_word_min=552,
        target_word_max=2068,
        prose_density='sparse',
        voice='synoptic narration; no dialogue; condensed scene description; third-person POV; past tense',
        style_exemplar="The story begins seven years after the broken engagement of Anne Elliot to Frederick Wentworth. Having just turned nineteen years old, Anne fell in love and had accepted a proposal of marriage from Wentworth, then a young and as yet undistinguished naval officer. Wentworth was considered clever, confident and ambitious, but his low social status and lack of wealth made Anne's family — her vain father Sir Walter Elliot and her older sister Elizabeth — view him as an unsuitable match for the daughter of a baronet.",
        source_word_count=1379,
    ),
    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_KELLYNCH_HALL": Location(
            name="Kellynch Hall",
            description="The Elliots' Somersetshire baronial seat; let to Admiral Croft when Sir Walter cannot afford to keep it.",
            ambient_state={
                "rank_consciousness": AmbientVector(value=0.95, volatility=0.1, evidence_strength="strong"),
                "fiscal_strain": AmbientVector(value=0.7, volatility=0.3, evidence_strength="strong"),
                # Frijda action-readiness: Anne can be removed to Uppercross or Bath; flight feasible.
                "connected_to": AmbientVector(value=1.0, volatility=0.0, evidence_strength="strong"),
            },
        ),
        "LOC_UPPERCROSS": Location(
            name="Uppercross Hall",
            description="The Musgroves' easy country house where Anne stays with her sister Mary.",
            ambient_state={
                "warmth": AmbientVector(value=0.7, volatility=0.2, evidence_strength="strong"),
                "informality": AmbientVector(value=0.7, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_LYME_REGIS": Location(
            name="Lyme Regis (the Cobb)",
            description="Coastal town and seawall where the Musgrove party visit Wentworth's naval friends — and where Louisa falls.",
            ambient_state={
                "naval_camaraderie": AmbientVector(value=0.7, volatility=0.2, evidence_strength="strong"),
                "physical_danger": AmbientVector(value=0.5, volatility=0.4, evidence_strength="moderate"),
            },
        ),
        "LOC_BATH": Location(
            name="Bath",
            description="Fashionable spa city where Sir Walter and Elizabeth retrench, and where the courtship plots converge.",
            ambient_state={
                "social_display": AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
                "marriage_market": AmbientVector(value=0.8, volatility=0.2, evidence_strength="strong"),
                # Anne is socially constrained — propriety, family duty, and the
                # marriage market keep her in William Elliot's orbit; physical
                # exits exist but social flight does not.
                "connected_to": AmbientVector(value=0.4, volatility=0.3, evidence_strength="moderate"),
            },
        ),
        "LOC_HARVILLE_LODGINGS": Location(
            name="The Harvilles' Lodgings",
            description="Modest seafront rooms in Lyme where Captain Harville's family receive the Musgrove party and where Louisa convalesces.",
            ambient_state={
                "domestic_warmth": AmbientVector(value=0.85, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_MRS_SMITH_LODGINGS": Location(
            name="Mrs Smith's Lodgings, Westgate Buildings",
            description="The shabby Bath rooms where Anne's impoverished old school friend lives.",
            ambient_state={
                "poverty": AmbientVector(value=0.8, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_WHITE_HART": Location(
            name="The White Hart Inn",
            description="Bath inn where the Musgroves stay and where Anne and Harville's overheard conversation reaches Wentworth.",
            ambient_state={
                "bustle": AmbientVector(value=0.7, volatility=0.3, evidence_strength="strong"),
            },
        ),
    },

    # ── OBJECTS ────────────────────────────────────────────────────────
    objects={
        "OBJ_BARONETAGE": NarrativeObject(
            id="OBJ_BARONETAGE", name="Sir Walter's Baronetage",
            location_id="LOC_KELLYNCH_HALL", owner_id="ENT_SIR_WALTER",
            properties={"state": "well_thumbed", "page_marked": "ELLIOT_OF_KELLYNCH"},
            affordances=[Affordance(action="display_rank", target_type="Entity")],
        ),
        "OBJ_KELLYNCH_LEASE": NarrativeObject(
            id="OBJ_KELLYNCH_LEASE", name="Lease of Kellynch Hall to Admiral Croft",
            location_id="LOC_KELLYNCH_HALL", owner_id="ENT_SIR_WALTER",
            properties={"state": "executed", "tenant": "ENT_ADMIRAL_CROFT"},
            affordances=[Affordance(action="bring_navy_into_county", target_type="Entity")],
        ),
        "OBJ_PRIZE_MONEY": NarrativeObject(
            id="OBJ_PRIZE_MONEY", name="Wentworth's Prize Money",
            location_id=None, owner_id="ENT_WENTWORTH",
            properties={"state": "twenty_five_thousand_pounds", "source": "captured_french_ships"},
            affordances=[Affordance(action="qualify_as_suitor", target_type="Entity")],
        ),
        "OBJ_SMITH_PROPERTY": NarrativeObject(
            id="OBJ_SMITH_PROPERTY", name="Mr Smith's West Indies Property",
            location_id=None, owner_id="ENT_MRS_SMITH",
            properties={"state": "neglected", "executor": "ENT_WILLIAM_ELLIOT"},
            affordances=[Affordance(action="restore_income", target_type="Entity")],
        ),
        "OBJ_WENTWORTHS_LETTER": NarrativeObject(
            id="OBJ_WENTWORTHS_LETTER", name="Captain Wentworth's Letter to Anne",
            location_id="LOC_WHITE_HART", owner_id="ENT_ANNE",
            properties={"state": "freshly_written", "content": "you_pierce_my_soul"},
            affordances=[Affordance(action="declare_love", target_type="Entity")],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    entities={
        "ENT_ANNE": Entity(
            id="ENT_ANNE", name="Anne Elliot",
            location_id="LOC_KELLYNCH_HALL", status="healthy",
            traits={
                "discernment":   TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
                "constancy":     TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "self_effacement": TraitVector(value=0.8, inertia=0.7,  evidence_strength="strong"),
                "regret":        TraitVector(value=0.75, inertia=0.6,  evidence_strength="strong"),
                "moral_courage": TraitVector(value=0.7,  inertia=0.65, evidence_strength="moderate"),
                "suspicion":     TraitVector(value=0.1,  inertia=0.4,  evidence_strength="moderate"),
                "loyalty":       TraitVector(value=0.7,  inertia=0.7,  evidence_strength="strong"),
                "hope":          TraitVector(value=0.2,  inertia=0.45, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_WENTWORTH",
                       perceived_state="he can never forgive me for the broken engagement", proposition_id="PROP_WENTWORTH_LOVES_ANNE",
                       confidence=0.85, inertia=0.5, established_at_fabula=3000, evidence_strength="strong"),
                Belief(target_id="ENT_LADY_RUSSELL",
                       perceived_state="she gave me bad counsel but she meant it as a mother would", proposition_id="PROP_PERSUASION_WAS_RIGHT",
                       confidence=0.8,  inertia=0.7, established_at_fabula=2000, evidence_strength="strong"),
            ],
            concerns=[
                # Sternberg passionate-bond — the constant, regretted love
                # that re-ignites when Wentworth returns to Somersetshire.
                Concern(concern_id="CCN_ANNE_DESIRES_WENTWORTH", proposition_id="PROP_WENTWORTH_LOVES_ANNE",
                        polarity="desire", kind="love", salience=1.0,
                        activation_fabula_window=[4000, 12000],
                        counter_concern_ids=["CCN_ANNE_FEARS_REJECTION"]),
                Concern(concern_id="CCN_ANNE_FEARS_REJECTION", proposition_id="PROP_WENTWORTH_LOVES_ANNE",
                        polarity="fear", kind="abandonment", salience=0.95,
                        activation_fabula_window=[4000, 11900],
                        counter_concern_ids=["CCN_ANNE_DESIRES_WENTWORTH"]),
                # Kahneman & Miller closeness-controllability — the foundational
                # regret that the engagement was broken at Lady Russell's urging.
                Concern(concern_id="CCN_ANNE_REGRETS_PERSUASION", proposition_id="PROP_PERSUASION_WAS_RIGHT",
                        polarity="fear", kind="injustice", salience=0.85,
                        activation_fabula_window=[1000, 11900]),
                # Lazarus secondary appraisal — anxiety over William Elliot's suit
                # before Mrs Smith's revelation gives her the truth.
                Concern(concern_id="CCN_ANNE_FEARS_WILLIAM", proposition_id="PROP_WILLIAM_ELLIOT_HONOURABLE",
                        polarity="fear", kind="exposure", salience=0.7,
                        activation_fabula_window=[8500, 10500],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=10500, triggered_by="EVT_MRS_SMITH_REVEALS_ELLIOT",
                                            salience=0.15),
                        ]),
                # Bowlby attachment — loyalty to Mrs Smith; the moral test of Bath.
                Concern(concern_id="CCN_ANNE_DESIRES_HELP_SMITH", proposition_id="PROP_MRS_SMITH_RELIEVED",
                        polarity="desire", kind="loyalty", salience=0.7,
                        activation_fabula_window=[10000, 14100]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_BROKEN_ENGAGEMENT",
                    traits={
                        "regret": TraitVector(value=1.00, inertia=0.60, evidence_strength="moderate"),
                        "self_effacement": TraitVector(value=1.00, inertia=0.70, evidence_strength="moderate"),
                    }),
                EntityStateSnapshot(fabula_time=3000, triggered_by="EVT_KELLYNCH_LET",
                    traits={
                        "self_effacement": TraitVector(value=0.95, inertia=0.70, evidence_strength="moderate"),
                    }),
                EntityStateSnapshot(fabula_time=4000, triggered_by="EVT_ANNE_VISITS_UPPERCROSS",
                    location_id="LOC_UPPERCROSS"),
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_LYME_VISIT",
                    location_id="LOC_LYME_REGIS"),
                EntityStateSnapshot(fabula_time=7000, triggered_by="EVT_LOUISA_FALLS",
                    traits={
                        "moral_courage": TraitVector(value=0.9, inertia=0.75, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=8000, triggered_by="EVT_ANNE_TO_BATH",
                    location_id="LOC_BATH"),
                EntityStateSnapshot(fabula_time=8600, triggered_by="EVT_WILLIAM_RETURNS_TO_FAMILY",
                    traits={
                        "suspicion": TraitVector(value=0.50, inertia=0.40, evidence_strength="moderate"),
                    }),
                EntityStateSnapshot(fabula_time=10100, triggered_by="EVT_ANNE_VISITS_MRS_SMITH",
                    traits={
                        "moral_courage": TraitVector(value=0.85, inertia=0.65, evidence_strength="moderate"),
                        "loyalty": TraitVector(value=0.95, inertia=0.70, evidence_strength="moderate"),
                    }),
                EntityStateSnapshot(fabula_time=10500, triggered_by="EVT_MRS_SMITH_REVEALS_ELLIOT",
                    beliefs_added=[
                        Belief(target_id="ENT_WILLIAM_ELLIOT",
                               perceived_state="cold, calculating; courts me only to forestall Mrs Clay", proposition_id="PROP_WILLIAM_ELLIOT_HONOURABLE",
                               confidence=0.95, inertia=0.7, established_at_fabula=10500, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=11920, triggered_by="EVT_WENTWORTHS_LETTER",
                    traits={
                        "hope": TraitVector(value=1.00, inertia=0.45, evidence_strength="moderate"),
                    }),EntityStateSnapshot(fabula_time=12000, triggered_by="EVT_RECONCILIATION",
                    traits={
                        "regret": TraitVector(value=0.1, inertia=0.6, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_WENTWORTH"],
                    beliefs_added=[
                        Belief(target_id="ENT_WENTWORTH",
                               perceived_state="he loves me as constantly as ever",
                               proposition_id="PROP_WENTWORTH_LOVES_ANNE",
                               confidence=1.0, inertia=0.85, established_at_fabula=12000, evidence_strength="strong"),
                    ]),
                
            ],
        ),
        "ENT_WENTWORTH": Entity(
            id="ENT_WENTWORTH", name="Captain Frederick Wentworth",
            location_id="LOC_KELLYNCH_HALL", status="healthy",
            traits={
                "ambition":      TraitVector(value=0.85, inertia=0.7,  evidence_strength="strong"),
                "professional_pride": TraitVector(value=0.9, inertia=0.8, evidence_strength="strong"),
                "resentment":    TraitVector(value=0.8,  inertia=0.55, evidence_strength="strong"),
                "constancy":     TraitVector(value=0.85, inertia=0.75, evidence_strength="moderate"),
                "self_knowledge": TraitVector(value=0.5, inertia=0.55, evidence_strength="moderate"),
                "hope":          TraitVector(value=0.2,  inertia=0.45, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_ANNE",
                       perceived_state="weak-charactered; could be persuaded out of love", proposition_id="PROP_PERSUASION_WAS_RIGHT",
                       confidence=0.85, inertia=0.5, established_at_fabula=4000, evidence_strength="strong"),
            ],
            concerns=[
                # Lazarus appraisal — naval ambition / prize-money fortune,
                # initially the substitute for the love he was refused.
                Concern(concern_id="CCN_WENTWORTH_DESIRES_FORTUNE", proposition_id="PROP_WENTWORTH_RICH",
                        polarity="desire", kind="ambition", salience=0.85,
                        activation_fabula_window=[1000, 4000]),
                # Averill normative-violation rage — the broken engagement reads
                # as betrayal until the Lyme conversation softens him.
                Concern(concern_id="CCN_WENTWORTH_FEARS_REJECTED_AGAIN", proposition_id="PROP_ANNE_LOVES_WENTWORTH",
                        polarity="fear", kind="betrayal", salience=0.95,
                        activation_fabula_window=[4000, 11800],
                        counter_concern_ids=["CCN_WENTWORTH_DESIRES_ANNE"],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=11800, triggered_by="EVT_HARVILLE_ANNE_CONVERSATION",
                                            salience=0.35),
                        ]),
                # Sternberg passionate-bond — re-emerges through the Lyme arc.
                Concern(concern_id="CCN_WENTWORTH_DESIRES_ANNE", proposition_id="PROP_ANNE_LOVES_WENTWORTH",
                        polarity="desire", kind="love", salience=1.0,
                        activation_fabula_window=[7000, 12000],
                        counter_concern_ids=["CCN_WENTWORTH_FEARS_REJECTED_AGAIN"],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=11900, triggered_by="EVT_WENTWORTHS_LETTER",
                                            salience=1.0),
                        ]),
                # Frijda guilt — once Louisa falls, his entanglement becomes a trap.
                Concern(concern_id="CCN_WENTWORTH_FEARS_LOUISA", proposition_id="PROP_LOUISA_WINS_WENTWORTH",
                        polarity="fear", kind="abandonment", salience=0.8,
                        activation_fabula_window=[7000, 9000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_BROKEN_ENGAGEMENT",
                    traits={
                        "resentment": TraitVector(value=1.00, inertia=0.55, evidence_strength="moderate"),
                        "ambition": TraitVector(value=1.00, inertia=0.70, evidence_strength="moderate"),
                    }),
                EntityStateSnapshot(fabula_time=4000, triggered_by="EVT_WENTWORTH_RETURNS",
                    location_id="LOC_UPPERCROSS"),
                EntityStateSnapshot(fabula_time=5500, triggered_by="EVT_WENTWORTH_LEARNS_ANNE_REFUSED_CHARLES",
                    traits={
                        "self_knowledge": TraitVector(value=0.65, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=7200, triggered_by="EVT_LOUISA_FALLS",
                    traits={
                        "self_knowledge": TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
                        "resentment":     TraitVector(value=0.3, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_ANNE"],
                    beliefs_added=[
                        Belief(target_id="ENT_ANNE",
                               perceived_state="composed and resolute under pressure; my error to encourage Louisa", proposition_id="PROP_ANNE_LOVES_WENTWORTH",
                               confidence=0.95, inertia=0.8, established_at_fabula=7200, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_LOUISA_BENWICK_ENGAGED",
                    traits={
                        "constancy": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=11000, triggered_by="EVT_WENTWORTH_TO_BATH",
                    location_id="LOC_BATH"),
                EntityStateSnapshot(fabula_time=11850, triggered_by="EVT_HARVILLE_ANNE_CONVERSATION",
                    traits={
                        "hope": TraitVector(value=0.90, inertia=0.45, evidence_strength="moderate"),
                        "resentment": TraitVector(value=0.40, inertia=0.55, evidence_strength="moderate"),
                    }),EntityStateSnapshot(fabula_time=12000, triggered_by="EVT_RECONCILIATION",
                    traits={
                        "resentment": TraitVector(value=0.05, inertia=0.6, evidence_strength="strong"),
                    }),
                
            ],
        ),
        "ENT_LADY_RUSSELL": Entity(
            id="ENT_LADY_RUSSELL", name="Lady Russell",
            location_id="LOC_KELLYNCH_HALL", status="healthy",
            traits={
                "prudence":      TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "rank_snobbery": TraitVector(value=0.7,  inertia=0.75, evidence_strength="strong"),
                "maternal_concern": TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_WENTWORTH",
                       perceived_state="a young, undistinguished, rash sailor — quite unsuitable", proposition_id="PROP_WENTWORTH_RICH",
                       confidence=0.9, inertia=0.7, established_at_fabula=1000, evidence_strength="strong"),
                Belief(target_id="ENT_WILLIAM_ELLIOT",
                       perceived_state="excellent match: rank, fortune, and renewed family connection",
                       proposition_id="PROP_ANNE_MARRIES_WILLIAM",
                       confidence=0.85, inertia=0.5, established_at_fabula=9000, evidence_strength="moderate"),
            ],
            concerns=[
                # Bowlby maternal-attachment — Anne's well-being filtered through rank.
                Concern(concern_id="CCN_LADY_RUSSELL_DESIRES_PRUDENT_MATCH", proposition_id="PROP_ANNE_MARRIES_WILLIAM",
                        polarity="desire", kind="social_status", salience=0.85,
                        activation_fabula_window=[8500, 11900]),
                Concern(concern_id="CCN_LADY_RUSSELL_FEARS_WENTWORTH", proposition_id="PROP_WENTWORTH_LOVES_ANNE",
                        polarity="fear", kind="social_status", salience=0.75,
                        activation_fabula_window=[1000, 12000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=12500, triggered_by="EVT_RECONCILIATION",
                    beliefs_invalidated=["ENT_WENTWORTH", "ENT_WILLIAM_ELLIOT"],
                    beliefs_added=[
                        Belief(target_id="ENT_WENTWORTH",
                               perceived_state="my judgement of him was wrong; he is worthy of Anne",
                               proposition_id="PROP_PERSUASION_WAS_RIGHT",
                               confidence=0.9, inertia=0.7, established_at_fabula=12500, evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_SIR_WALTER": Entity(
            id="ENT_SIR_WALTER", name="Sir Walter Elliot",
            location_id="LOC_KELLYNCH_HALL", status="healthy",
            traits={
                "vanity":        TraitVector(value=0.95, inertia=0.9,  evidence_strength="strong"),
                "rank_snobbery": TraitVector(value=0.95, inertia=0.9,  evidence_strength="strong"),
                "fiscal_imprudence": TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
                "paternal_affection": TraitVector(value=0.2, inertia=0.6, evidence_strength="moderate"),
                "humiliation":   TraitVector(value=0.0,  inertia=0.5, evidence_strength="weak"),
            },
            beliefs=[],
            concerns=[
                # OCC pride — rank above all, with the Mrs Clay menace as the
                # unspoken status anxiety the family will not name.
                Concern(concern_id="CCN_SIR_WALTER_DESIRES_RANK", proposition_id="PROP_KELLYNCH_KEPT",
                        polarity="desire", kind="social_status", salience=0.9,
                        activation_fabula_window=[1, 3000]),
                Concern(concern_id="CCN_SIR_WALTER_FEARS_HUMILIATION", proposition_id="PROP_MRS_CLAY_MARRIES_SIR_WALTER",
                        polarity="fear", kind="humiliation", salience=0.65,
                        activation_fabula_window=[3000, 13200]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3000, triggered_by="EVT_KELLYNCH_LET",
                    location_id="LOC_BATH"),
            ],
        ),
        "ENT_ELIZABETH": Entity(
            id="ENT_ELIZABETH", name="Elizabeth Elliot",
            location_id="LOC_KELLYNCH_HALL", status="healthy",
            traits={
                "vanity":        TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "rank_snobbery": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "self_regard":   TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_MRS_CLAY",
                       perceived_state="useful flatterer; no danger to my prospects", proposition_id="PROP_MRS_CLAY_MARRIES_SIR_WALTER",
                       confidence=0.85, inertia=0.65, established_at_fabula=3000, evidence_strength="moderate"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3000, triggered_by="EVT_KELLYNCH_LET",
                    location_id="LOC_BATH"),
                EntityStateSnapshot(fabula_time=8600, triggered_by="EVT_WILLIAM_RETURNS_TO_FAMILY",
                    traits={
                        "vanity": TraitVector(value=0.90, inertia=0.85, evidence_strength="moderate"),
                    }),
            ],
        ),
        "ENT_MARY": Entity(
            id="ENT_MARY", name="Mary Musgrove",
            location_id="LOC_UPPERCROSS", status="ill",
            traits={
                "self_pity":     TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
                "rank_snobbery": TraitVector(value=0.75, inertia=0.7,  evidence_strength="strong"),
                "querulousness": TraitVector(value=0.85, inertia=0.7,  evidence_strength="strong"),
            },
            beliefs=[],
        ),
        "ENT_CHARLES_MUSGROVE": Entity(
            id="ENT_CHARLES_MUSGROVE", name="Charles Musgrove",
            location_id="LOC_UPPERCROSS", status="healthy",
            traits={
                "good_humour":  TraitVector(value=0.8, inertia=0.7, evidence_strength="strong"),
                "sporting_distraction": TraitVector(value=0.75, inertia=0.65, evidence_strength="moderate"),
            },
            beliefs=[],
        ),
        "ENT_HENRIETTA": Entity(
            id="ENT_HENRIETTA", name="Henrietta Musgrove",
            location_id="LOC_UPPERCROSS", status="healthy",
            traits={
                "amiability":     TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                "suggestibility": TraitVector(value=0.7, inertia=0.55, evidence_strength="moderate"),
                "self_knowledge": TraitVector(value=0.25, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_CHARLES_HAYTER",
                       perceived_state="my settled love", proposition_id="PROP_HENRIETTA_MARRIES_HAYTER",
                       confidence=0.85, inertia=0.6, established_at_fabula=4500, evidence_strength="strong"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=5200, triggered_by="EVT_HAYTER_WITHDRAWS",
                    traits={
                        "suggestibility": TraitVector(value=0.5, inertia=0.6, evidence_strength="moderate"),
                    }),
                EntityStateSnapshot(fabula_time=5200, triggered_by="EVT_HAYTER_WITHDRAWS",
                    traits={
                        "self_knowledge": TraitVector(value=0.50, inertia=0.50, evidence_strength="moderate"),
                    }),
            ],
        ),
        "ENT_LOUISA": Entity(
            id="ENT_LOUISA", name="Louisa Musgrove",
            location_id="LOC_UPPERCROSS", status="healthy",
            traits={
                "high_spirits": TraitVector(value=0.9, inertia=0.65, evidence_strength="strong"),
                "wilfulness":   TraitVector(value=0.85, inertia=0.7,  evidence_strength="strong"),
                "infatuation":  TraitVector(value=0.4, inertia=0.4,  evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_WENTWORTH",
                       perceived_state="my decided suitor; I will be persuaded by no one", proposition_id="PROP_LOUISA_WINS_WENTWORTH",
                       confidence=0.85, inertia=0.5, established_at_fabula=5500, evidence_strength="moderate"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=7000, triggered_by="EVT_LOUISA_FALLS",
                    status="ill", location_id="LOC_HARVILLE_LODGINGS",
                    traits={
                        "high_spirits": TraitVector(value=0.2, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_LOUISA_BENWICK_ENGAGED",
                    status="healthy",
                    traits={
                        "infatuation": TraitVector(value=0.85, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_WENTWORTH"],
                    beliefs_added=[
                        Belief(target_id="ENT_BENWICK",
                               perceived_state="my poetic, grieving suitor", proposition_id="PROP_LOUISA_WINS_WENTWORTH",
                               confidence=0.85, inertia=0.55, established_at_fabula=9000, evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_BENWICK": Entity(
            id="ENT_BENWICK", name="Captain James Benwick",
            location_id="LOC_HARVILLE_LODGINGS", status="healthy",
            traits={
                "melancholy":    TraitVector(value=0.85, inertia=0.65, evidence_strength="strong"),
                "literary_taste": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "grief":         TraitVector(value=0.9,  inertia=0.7,  evidence_strength="strong"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_LOUISA_BENWICK_ENGAGED",
                    traits={
                        "grief":      TraitVector(value=0.3, inertia=0.65, evidence_strength="strong"),
                        "melancholy": TraitVector(value=0.4, inertia=0.6, evidence_strength="moderate"),
                    }),
            ],
        ),
        "ENT_HARVILLE": Entity(
            id="ENT_HARVILLE", name="Captain Harville",
            location_id="LOC_HARVILLE_LODGINGS", status="healthy",
            traits={
                "domestic_warmth": TraitVector(value=0.9, inertia=0.8, evidence_strength="strong"),
                "constancy":     TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
            },
            beliefs=[],
        ),
        "ENT_ADMIRAL_CROFT": Entity(
            id="ENT_ADMIRAL_CROFT", name="Admiral Croft",
            location_id="LOC_KELLYNCH_HALL", status="healthy",
            traits={
                "bluff_decency": TraitVector(value=0.9, inertia=0.8, evidence_strength="strong"),
                "uxoriousness":  TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
            },
            beliefs=[],
            constants=["royal_navy_admiral", "wentworths_brother_in_law"],
            state_timeline=[
                EntityStateSnapshot(
                    fabula_time=3000, triggered_by="EVT_KELLYNCH_LET",
                    location_id="LOC_KELLYNCH_HALL",
                    traits={
                        "bluff_decency": TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_SOPHIA_CROFT": Entity(
            id="ENT_SOPHIA_CROFT", name="Mrs Sophia Croft",
            location_id="LOC_KELLYNCH_HALL", status="healthy",
            traits={
                "competence":   TraitVector(value=0.9, inertia=0.8, evidence_strength="strong"),
                "frankness":    TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
            },
            beliefs=[],
        ),
        "ENT_WILLIAM_ELLIOT": Entity(
            id="ENT_WILLIAM_ELLIOT", name="Mr William Elliot",
            location_id="LOC_LYME_REGIS", status="healthy",
            traits={
                "polish":       TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "duplicity":    TraitVector(value=0.85, inertia=0.8,  evidence_strength="strong"),
                "calculation":  TraitVector(value=0.9,  inertia=0.85, evidence_strength="strong"),
                "self_interest": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="OBJ_BARONETAGE",
                       perceived_state="the Kellynch entail must come to me intact", proposition_id="PROP_WILLIAM_INHERITS",
                       confidence=0.95, inertia=0.85, established_at_fabula=8500, evidence_strength="strong"),
            ],
            concerns=[
                # Lazarus instrumental-goal appraisal — the entail is the only stake.
                Concern(concern_id="CCN_WILLIAM_DESIRES_BARONETAGE", proposition_id="PROP_WILLIAM_INHERITS",
                        polarity="desire", kind="ambition", salience=0.95,
                        activation_fabula_window=[8500, 13200]),
                # Averill normative violation — Mrs Clay would block the inheritance.
                Concern(concern_id="CCN_WILLIAM_FEARS_MRS_CLAY", proposition_id="PROP_MRS_CLAY_MARRIES_SIR_WALTER",
                        polarity="fear", kind="injustice", salience=0.9,
                        activation_fabula_window=[8500, 13200]),
                # Berscheid game-of-courtship — Anne is instrument and shield.
                Concern(concern_id="CCN_WILLIAM_DESIRES_ANNE", proposition_id="PROP_ANNE_MARRIES_WILLIAM",
                        polarity="desire", kind="social_status", salience=0.7,
                        activation_fabula_window=[8500, 12000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_WILLIAM_RETURNS_TO_FAMILY",
                    location_id="LOC_BATH"),
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_WILLIAM_LEAVES_BATH",
                    location_id=None),
                EntityStateSnapshot(fabula_time=13250, triggered_by="EVT_MRS_CLAY_FOLLOWS_WILLIAM",
                    traits={
                        "duplicity": TraitVector(value=0.90, inertia=0.80, evidence_strength="moderate"),
                    }),
            ],
        ),
        "ENT_MRS_CLAY": Entity(
            id="ENT_MRS_CLAY", name="Mrs Penelope Clay",
            location_id="LOC_KELLYNCH_HALL", status="healthy",
            traits={
                "ingratiation": TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
                "ambition":     TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[],
            concerns=[
                # Lazarus secondary appraisal — climbing via either Sir Walter or
                # William, whichever route the social arithmetic favours.
                Concern(concern_id="CCN_MRS_CLAY_DESIRES_SIR_WALTER", proposition_id="PROP_MRS_CLAY_MARRIES_SIR_WALTER",
                        polarity="desire", kind="ambition", salience=0.85,
                        activation_fabula_window=[3000, 13200]),
                Concern(concern_id="CCN_MRS_CLAY_FEARS_EXPOSURE", proposition_id="PROP_MRS_CLAY_EXPOSED",
                        polarity="fear", kind="exposure", salience=0.7,
                        activation_fabula_window=[3000, 13200]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3000, triggered_by="EVT_KELLYNCH_LET",
                    location_id="LOC_BATH"),
                EntityStateSnapshot(fabula_time=13200, triggered_by="EVT_MRS_CLAY_FOLLOWS_WILLIAM",
                    location_id=None),
            ],
        ),
        "ENT_MRS_SMITH": Entity(
            id="ENT_MRS_SMITH", name="Mrs Smith",
            location_id="LOC_MRS_SMITH_LODGINGS", status="ill",
            traits={
                "endurance":     TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
                "discernment":   TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_WILLIAM_ELLIOT",
                       perceived_state="cold opportunist who ruined my husband and refuses to act for me",
                       proposition_id="PROP_WILLIAM_ELLIOT_HONOURABLE",
                       confidence=0.95, inertia=0.85, established_at_fabula=9500, evidence_strength="strong"),
            ],
            constants=["widowed", "annes_old_school_friend"],
            state_timeline=[
                EntityStateSnapshot(
                    fabula_time=10000, triggered_by="EVT_ANNE_VISITS_MRS_SMITH",
                    location_id="LOC_MRS_SMITH_LODGINGS",
                    traits={
                        "endurance": TraitVector(value=0.88, inertia=0.78, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(
                    fabula_time=14100, triggered_by="EVT_WENTWORTH_HELPS_MRS_SMITH",
                    location_id="LOC_MRS_SMITH_LODGINGS", status="healthy",
                    traits={
                        "endurance": TraitVector(value=0.92, inertia=0.8, evidence_strength="strong"),
                        "gratitude": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_CHARLES_HAYTER": Entity(
            id="ENT_CHARLES_HAYTER", name="Charles Hayter",
            location_id="LOC_UPPERCROSS", status="healthy",
            traits={
                "diffidence":  TraitVector(value=0.7, inertia=0.6, evidence_strength="moderate"),
                "constancy":   TraitVector(value=0.8, inertia=0.7, evidence_strength="moderate"),
            },
            beliefs=[],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_BROKEN_ENGAGEMENT", fabula_time=1000, syuzhet_index=1,
                  event_type="choice", actor_ids=["ENT_ANNE", "ENT_LADY_RUSSELL"], target_ids=["ENT_WENTWORTH"],
                  description="Persuaded by Lady Russell that the match is imprudent for one so young, the nineteen-year-old Anne breaks her engagement to the unestablished Lieutenant Wentworth."),
        EventNode(id="EVT_WAR_AND_PRIZE_MONEY", fabula_time=2000, syuzhet_index=2,
                  event_type="outcome", actor_ids=["ENT_WENTWORTH"], target_ids=[],
                  description="Through the late Napoleonic Wars Wentworth captures French ships and accumulates roughly £25,000 in prize money, becoming a captain of fortune."),
        EventNode(id="EVT_KELLYNCH_LET", fabula_time=3000, syuzhet_index=3,
                  event_type="choice", actor_ids=["ENT_SIR_WALTER"], target_ids=["ENT_ADMIRAL_CROFT"],
                  description="Pressed by debts, Sir Walter is persuaded to let Kellynch Hall to Admiral and Mrs Croft and remove with Elizabeth and Mrs Clay to Bath."),
        EventNode(id="EVT_ANNE_VISITS_UPPERCROSS", fabula_time=4000, syuzhet_index=4,
                  event_type="choice", actor_ids=["ENT_ANNE", "ENT_CHARLES_MUSGROVE"], target_ids=[],
                  description="Anne goes to Uppercross to nurse Mary, putting her in the social orbit of the Crofts and the returning Wentworth."),
        EventNode(id="EVT_WENTWORTH_RETURNS", fabula_time=4200, syuzhet_index=5,
                  event_type="outcome", actor_ids=["ENT_WENTWORTH"], target_ids=[],
                  description="Wentworth visits the Crofts at Kellynch and is drawn into the Musgrove circle, openly looking to marry."),
        EventNode(id="EVT_HAYTER_WITHDRAWS", fabula_time=5000, syuzhet_index=6,
                  event_type="choice", actor_ids=["ENT_CHARLES_HAYTER"], target_ids=["ENT_HENRIETTA"],
                  description="Hurt by Henrietta's apparent interest in Wentworth, Charles Hayter stops visiting Uppercross."),
        EventNode(id="EVT_WENTWORTH_LEARNS_ANNE_REFUSED_CHARLES", fabula_time=5500, syuzhet_index=7,
                  event_type="outcome", actor_ids=["ENT_LOUISA"], target_ids=["ENT_WENTWORTH"],
                  description="Anne overhears Louisa telling Wentworth that Anne refused Charles Musgrove's first proposal, revealing Anne's continued unmarried state and disturbing Wentworth's settled view of her."),
        EventNode(id="EVT_LYME_VISIT", fabula_time=6000, syuzhet_index=8,
                  event_type="choice", actor_ids=["ENT_WENTWORTH", "ENT_ANNE"], target_ids=[],
                  description="The Uppercross party visits Captain Harville's family at Lyme Regis; Anne's looks attract the notice of a stranger who turns out to be William Elliot."),
        EventNode(id="EVT_WILLIAM_ADMIRES_ANNE_AT_LYME", fabula_time=6500, syuzhet_index=9,
                  event_type="outcome", actor_ids=["ENT_WILLIAM_ELLIOT"], target_ids=["ENT_ANNE"],
                  description="At Lyme, William Elliot — heir presumptive to Kellynch — sees and silently admires Anne, planting his later courtship."),
        EventNode(id="EVT_LOUISA_FALLS", fabula_time=7000, syuzhet_index=10,
                  event_type="outcome", actor_ids=["ENT_LOUISA"], target_ids=["ENT_LOUISA"],
                  description="On the Cobb Louisa insists on jumping a second time before Wentworth is ready to catch her, falls, and is concussed; Anne directs the response while the others panic."),
        EventNode(id="EVT_ANNE_TO_BATH", fabula_time=8000, syuzhet_index=11,
                  event_type="choice", actor_ids=["ENT_ANNE"], target_ids=[],
                  description="Anne joins Sir Walter and Elizabeth in Bath, where Lady Russell is also resident."),
        EventNode(id="EVT_WILLIAM_RETURNS_TO_FAMILY", fabula_time=8500, syuzhet_index=12,
                  event_type="choice", actor_ids=["ENT_WILLIAM_ELLIOT"], target_ids=["ENT_SIR_WALTER"],
                  description="William Elliot reconciles with Sir Walter and Elizabeth in Bath, courting Anne under the pretext of preserving the family line."),
        EventNode(id="EVT_LOUISA_BENWICK_ENGAGED", fabula_time=9000, syuzhet_index=13,
                  event_type="outcome", actor_ids=["ENT_LOUISA", "ENT_BENWICK"], target_ids=[],
                  description="During Louisa's long convalescence at the Harvilles', she and Captain Benwick fall in love and become engaged, freeing Wentworth from the obligation he had felt to her."),
        EventNode(id="EVT_ANNE_VISITS_MRS_SMITH", fabula_time=10000, syuzhet_index=14,
                  event_type="choice", actor_ids=["ENT_ANNE"], target_ids=["ENT_MRS_SMITH"],
                  description="Against her father's snobbish disapproval, Anne calls on her impoverished old school friend Mrs Smith in Westgate Buildings."),
        EventNode(id="EVT_MRS_SMITH_REVEALS_ELLIOT", fabula_time=10500, syuzhet_index=15,
                  event_type="outcome", actor_ids=["ENT_MRS_SMITH"], target_ids=["ENT_ANNE"],
                  description="Mrs Smith reveals William Elliot's history of cold opportunism, his refusal to act as her husband's executor, and that his real motive in courting Anne is to forestall Mrs Clay marrying Sir Walter and bearing a displacing male heir."),
        EventNode(id="EVT_WENTWORTH_TO_BATH", fabula_time=11000, syuzhet_index=16,
                  event_type="choice", actor_ids=["ENT_WENTWORTH"], target_ids=[],
                  description="Wentworth, freed by Louisa's engagement to Benwick, travels to Bath and is alarmed to find William Elliot seemingly courting Anne."),
        EventNode(id="EVT_HARVILLE_ANNE_CONVERSATION", fabula_time=11800, syuzhet_index=17,
                  event_type="choice", actor_ids=["ENT_ANNE", "ENT_HARVILLE"], target_ids=["ENT_WENTWORTH"],
                  description="At the White Hart, Anne discusses with Harville the comparative constancy of men and women in love, arguing that women love longest when all hope is gone — within Wentworth's hearing."),
        EventNode(id="EVT_WENTWORTHS_LETTER", fabula_time=11900, syuzhet_index=18,
                  event_type="choice", actor_ids=["ENT_WENTWORTH"], target_ids=["ENT_ANNE"],
                  description="Pierced by Anne's words, Wentworth writes the letter declaring he is half agony, half hope, and slips it where Anne will find it."),
        EventNode(id="EVT_RECONCILIATION", fabula_time=12000, syuzhet_index=19,
                  event_type="outcome", actor_ids=["ENT_ANNE", "ENT_WENTWORTH"], target_ids=[],
                  description="In the streets of Bath, Anne and Wentworth meet, declare themselves, and renew their engagement."),
        EventNode(id="EVT_WILLIAM_LEAVES_BATH", fabula_time=13000, syuzhet_index=20,
                  event_type="choice", actor_ids=["ENT_WILLIAM_ELLIOT"], target_ids=[],
                  description="His scheme exposed and his hopes of Anne dashed, William Elliot quietly leaves Bath."),
        EventNode(id="EVT_MRS_CLAY_FOLLOWS_WILLIAM", fabula_time=13200, syuzhet_index=21,
                  event_type="choice", actor_ids=["ENT_MRS_CLAY"], target_ids=["ENT_WILLIAM_ELLIOT"],
                  description="Mrs Clay slips away to become William Elliot's mistress, ending the threat that she might marry Sir Walter."),
        EventNode(id="EVT_WENTWORTH_HELPS_MRS_SMITH", fabula_time=14000, syuzhet_index=22,
                  event_type="outcome", actor_ids=["ENT_WENTWORTH"], target_ids=["ENT_MRS_SMITH"],
                  description="After his marriage to Anne, Wentworth uses his interest to help Mrs Smith recover her West Indies property and her income."),
    
        # ── UTTERANCES (on-page speech-acts) ──
        EventNode(id='EVT_UTT_LOUISA_TELLS_WENTWORTH', event_type='utterance',
                  description="Louisa tells Wentworth that Charles Musgrove first proposed to Anne, who refused him; Anne overhears.",
                  content="It was Anne he asked first — she would not have him; and so he turned to Mary.",
                  speaker_id='ENT_LOUISA', addressee_ids=['ENT_WENTWORTH'], actor_ids=['ENT_LOUISA'],
                  target_ids=['ENT_ANNE', 'EVT_BROKEN_ENGAGEMENT'],
                  via_channel_id=None, truth_value='true', fabula_time=5500, syuzhet_index=23),
        EventNode(id='EVT_UTT_WILLIAM_FLATTERS_ANNE', event_type='utterance',
                  description="William Elliot pays Anne refined, flattering attentions in Bath, hinting that someone had spoken fondly of her without revealing who.",
                  content="I was fascinated by the very name of Elliot — and someone, whom I shall not name, has long spoken of you with such warmth.",
                  speaker_id='ENT_WILLIAM_ELLIOT', addressee_ids=['ENT_ANNE'], actor_ids=['ENT_WILLIAM_ELLIOT'],
                  target_ids=['ENT_ANNE'],
                  via_channel_id='CHN_WILLIAM_FLATTERING_DISCOURSE', truth_value='false',
                  fabula_time=9500, syuzhet_index=24),
        EventNode(id='EVT_UTT_MRS_SMITH_REVEALS_WILLIAM', event_type='utterance',
                  description="Mrs Smith confides to Anne the true history of William Elliot — his ruin of her husband, his refusal to act as executor, and his calculating motives in courting Anne.",
                  content="He is a man without heart or conscience; he led my poor husband into ruin, and now courts you only to keep Mrs Clay from your father.",
                  speaker_id='ENT_MRS_SMITH', addressee_ids=['ENT_ANNE'], actor_ids=['ENT_MRS_SMITH'],
                  target_ids=['ENT_WILLIAM_ELLIOT', 'ENT_MRS_CLAY', 'ENT_SIR_WALTER'],
                  via_channel_id='CHN_MRS_SMITH_CONFIDANT', truth_value='true',
                  fabula_time=10500, syuzhet_index=25),
        EventNode(id='EVT_UTT_ADMIRAL_CROFT_BENWICK_NEWS', event_type='utterance',
                  description="Admiral Croft tells Anne in Bath that Louisa Musgrove is engaged to Captain Benwick.",
                  content="Did you ever hear the like? Louisa Musgrove and James Benwick — engaged, of all things!",
                  speaker_id='ENT_ADMIRAL_CROFT', addressee_ids=['ENT_ANNE'], actor_ids=['ENT_ADMIRAL_CROFT'],
                  target_ids=['EVT_LOUISA_BENWICK_ENGAGED', 'ENT_LOUISA', 'ENT_BENWICK'],
                  via_channel_id=None, truth_value='true', fabula_time=10800, syuzhet_index=26),
        EventNode(id='EVT_UTT_ANNE_WOMEN_CONSTANCY', event_type='utterance',
                  description="At the White Hart, Anne tells Captain Harville that women love longest, when existence or when hope is gone — within Wentworth's hearing.",
                  content="All the privilege I claim for my own sex — it is not a very enviable one — is that of loving longest, when existence or when hope is gone.",
                  speaker_id='ENT_ANNE', addressee_ids=['ENT_HARVILLE'], actor_ids=['ENT_ANNE'],
                  target_ids=['ENT_WENTWORTH', 'EVT_BROKEN_ENGAGEMENT'],
                  via_channel_id=None, truth_value='true', fabula_time=11800, syuzhet_index=27),
        EventNode(id='EVT_UTT_WENTWORTHS_LETTER', event_type='utterance',
                  description="Wentworth writes Anne the 'half agony, half hope' letter and slips it where she will find it.",
                  content="You pierce my soul. I am half agony, half hope... I have loved none but you.",
                  speaker_id='ENT_WENTWORTH', addressee_ids=['ENT_ANNE'], actor_ids=['ENT_WENTWORTH'],
                  target_ids=['ENT_ANNE', 'OBJ_WENTWORTHS_LETTER'],
                  via_channel_id=None, truth_value='true', fabula_time=11900, syuzhet_index=28),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction ──
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="EVT_WAR_AND_PRIZE_MONEY",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1000, propagation_delay=1000),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="EVT_WENTWORTH_RETURNS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3000, propagation_delay=1200),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="EVT_ANNE_VISITS_UPPERCROSS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3000, propagation_delay=1000),
        CausalEdge(source_id="EVT_WENTWORTH_RETURNS", target_id="EVT_HAYTER_WITHDRAWS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4200, propagation_delay=800),
        CausalEdge(source_id="EVT_WENTWORTH_RETURNS", target_id="EVT_WENTWORTH_LEARNS_ANNE_REFUSED_CHARLES",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=6.0, fabula_time=4200, propagation_delay=1300),
        CausalEdge(source_id="EVT_WENTWORTH_RETURNS", target_id="EVT_LYME_VISIT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=4200, propagation_delay=1800),
        CausalEdge(source_id="EVT_LYME_VISIT", target_id="EVT_WILLIAM_ADMIRES_ANNE_AT_LYME",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=6000, propagation_delay=500),
        CausalEdge(source_id="EVT_LYME_VISIT", target_id="EVT_LOUISA_FALLS",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000, propagation_delay=1000),
        CausalEdge(source_id="EVT_LOUISA_FALLS", target_id="EVT_LOUISA_BENWICK_ENGAGED",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000, propagation_delay=2000),
        CausalEdge(source_id="EVT_ANNE_TO_BATH", target_id="EVT_WILLIAM_RETURNS_TO_FAMILY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=8000, propagation_delay=500),
        CausalEdge(source_id="EVT_WILLIAM_ADMIRES_ANNE_AT_LYME", target_id="EVT_WILLIAM_RETURNS_TO_FAMILY",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=6500, propagation_delay=2000),
        CausalEdge(source_id="EVT_ANNE_TO_BATH", target_id="EVT_ANNE_VISITS_MRS_SMITH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=8000, propagation_delay=2000),
        CausalEdge(source_id="EVT_ANNE_VISITS_MRS_SMITH", target_id="EVT_MRS_SMITH_REVEALS_ELLIOT",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=8.0, fabula_time=10000, propagation_delay=500),
        CausalEdge(source_id="EVT_LOUISA_BENWICK_ENGAGED", target_id="EVT_WENTWORTH_TO_BATH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000, propagation_delay=2000),
        CausalEdge(source_id="EVT_WENTWORTH_TO_BATH", target_id="EVT_HARVILLE_ANNE_CONVERSATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=11000, propagation_delay=800),
        CausalEdge(source_id="EVT_HARVILLE_ANNE_CONVERSATION", target_id="EVT_WENTWORTHS_LETTER",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=11800, propagation_delay=100),
        CausalEdge(source_id="EVT_WENTWORTHS_LETTER", target_id="EVT_RECONCILIATION",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=11900, propagation_delay=100),
        CausalEdge(source_id="EVT_MRS_SMITH_REVEALS_ELLIOT", target_id="EVT_WILLIAM_LEAVES_BATH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=10500, propagation_delay=2500),
        CausalEdge(source_id="EVT_RECONCILIATION", target_id="EVT_WILLIAM_LEAVES_BATH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=12000, propagation_delay=1000),
        CausalEdge(source_id="EVT_WILLIAM_LEAVES_BATH", target_id="EVT_MRS_CLAY_FOLLOWS_WILLIAM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=13000, propagation_delay=200),
        CausalEdge(source_id="EVT_RECONCILIATION", target_id="EVT_WENTWORTH_HELPS_MRS_SMITH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=12000, propagation_delay=2000),

        # ── mutation ──
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=9.0, fabula_time=1000,
                   trait_target="regret", trait_delta=0.7),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000,
                   trait_target="self_effacement", trait_delta=0.3),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_WENTWORTH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1000,
                   trait_target="resentment", trait_delta=0.7),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_WENTWORTH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1000,
                   trait_target="ambition", trait_delta=0.3),
        CausalEdge(source_id="EVT_WENTWORTH_LEARNS_ANNE_REFUSED_CHARLES", target_id="ENT_WENTWORTH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=5500,
                   trait_target="self_knowledge", trait_delta=0.15),
        CausalEdge(source_id="EVT_LOUISA_FALLS", target_id="ENT_WENTWORTH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=7200,
                   trait_target="self_knowledge", trait_delta=0.4),
        CausalEdge(source_id="EVT_LOUISA_FALLS", target_id="ENT_WENTWORTH",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=9.0, fabula_time=7200,
                   trait_target="resentment", trait_delta=-0.5),
        CausalEdge(source_id="EVT_LOUISA_FALLS", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7200,
                   trait_target="moral_courage", trait_delta=0.2),
        CausalEdge(source_id="EVT_LOUISA_FALLS", target_id="ENT_LOUISA",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=7000,
                   trait_target="high_spirits", trait_delta=-0.7),
        CausalEdge(source_id="EVT_LOUISA_BENWICK_ENGAGED", target_id="ENT_BENWICK",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000,
                   trait_target="grief", trait_delta=-0.6),
        CausalEdge(source_id="EVT_LOUISA_BENWICK_ENGAGED", target_id="ENT_LOUISA",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=9000,
                   trait_target="infatuation", trait_delta=0.45),
        CausalEdge(source_id="EVT_RECONCILIATION", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=12000,
                   trait_target="regret", trait_delta=-0.65),
        CausalEdge(source_id="EVT_RECONCILIATION", target_id="ENT_WENTWORTH",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=12000,
                   trait_target="resentment", trait_delta=-0.25),

        # ── mutation: events newly covered ──
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_SIR_WALTER",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3000,
                   trait_target="vanity", trait_delta=-0.05),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_SIR_WALTER",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3000,
                   trait_target="humiliation", trait_delta=0.6),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3000,
                   trait_target="self_effacement", trait_delta=0.15),
        CausalEdge(source_id="EVT_HAYTER_WITHDRAWS", target_id="ENT_HENRIETTA",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=5200,
                   trait_target="suggestibility", trait_delta=-0.2),
        CausalEdge(source_id="EVT_HAYTER_WITHDRAWS", target_id="ENT_HENRIETTA",
                   causality_type="mutation", mechanism="emotional", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=5200,
                   trait_target="self_knowledge", trait_delta=0.25),
        CausalEdge(source_id="EVT_WILLIAM_RETURNS_TO_FAMILY", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=8600,
                   trait_target="suspicion", trait_delta=0.4),
        CausalEdge(source_id="EVT_WILLIAM_RETURNS_TO_FAMILY", target_id="ENT_ELIZABETH",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=8600,
                   trait_target="vanity", trait_delta=0.05),
        CausalEdge(source_id="EVT_ANNE_VISITS_MRS_SMITH", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=10100,
                   trait_target="moral_courage", trait_delta=0.15),
        CausalEdge(source_id="EVT_ANNE_VISITS_MRS_SMITH", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=10100,
                   trait_target="loyalty", trait_delta=0.25),
        CausalEdge(source_id="EVT_HARVILLE_ANNE_CONVERSATION", target_id="ENT_WENTWORTH",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=11850,
                   trait_target="hope", trait_delta=0.7),
        CausalEdge(source_id="EVT_HARVILLE_ANNE_CONVERSATION", target_id="ENT_WENTWORTH",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=11850,
                   trait_target="resentment", trait_delta=-0.4),
        CausalEdge(source_id="EVT_WENTWORTHS_LETTER", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=11920,
                   trait_target="hope", trait_delta=0.85),
        CausalEdge(source_id="EVT_MRS_CLAY_FOLLOWS_WILLIAM", target_id="ENT_WILLIAM_ELLIOT",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=13250,
                   trait_target="duplicity", trait_delta=0.05),
        CausalEdge(source_id="EVT_WENTWORTH_HELPS_MRS_SMITH", target_id="ENT_MRS_SMITH",
                   causality_type="mutation", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=14100,
                   trait_target="endurance", trait_delta=0.1),
        CausalEdge(source_id="EVT_WENTWORTH_HELPS_MRS_SMITH", target_id="ENT_MRS_SMITH",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=14100,
                   trait_target="gratitude", trait_delta=0.7),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_WENTWORTH",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=10.0, fabula_time=1000,
                   trait_target="affinity", trait_delta=-0.95, rel_counterpart_id="ENT_ANNE"),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=4.0, fabula_time=1000,
                   trait_target="affinity", trait_delta=-0.05, rel_counterpart_id="ENT_WENTWORTH"),
        CausalEdge(source_id="EVT_LOUISA_FALLS", target_id="ENT_WENTWORTH",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7200,
                   trait_target="affinity", trait_delta=0.65, rel_counterpart_id="ENT_ANNE"),
        CausalEdge(source_id="EVT_LOUISA_BENWICK_ENGAGED", target_id="ENT_LOUISA",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000,
                   trait_target="affinity", trait_delta=0.85, rel_counterpart_id="ENT_BENWICK"),
        CausalEdge(source_id="EVT_LOUISA_BENWICK_ENGAGED", target_id="ENT_BENWICK",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000,
                   trait_target="affinity", trait_delta=0.85, rel_counterpart_id="ENT_LOUISA"),
        CausalEdge(source_id="EVT_MRS_SMITH_REVEALS_ELLIOT", target_id="ENT_ANNE",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=9.0, fabula_time=10500,
                   trait_target="affinity", trait_delta=-0.85, rel_counterpart_id="ENT_WILLIAM_ELLIOT"),
        CausalEdge(source_id="EVT_RECONCILIATION", target_id="ENT_ANNE",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=12000,
                   trait_target="affinity", trait_delta=1.0, rel_counterpart_id="ENT_WENTWORTH"),
        CausalEdge(source_id="EVT_RECONCILIATION", target_id="ENT_WENTWORTH",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=12000,
                   trait_target="affinity", trait_delta=1.0, rel_counterpart_id="ENT_ANNE"),

        # ── affordance_gate ──
        CausalEdge(source_id="OBJ_KELLYNCH_LEASE", target_id="EVT_WENTWORTH_RETURNS",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=4200),
        CausalEdge(source_id="OBJ_PRIZE_MONEY", target_id="EVT_RECONCILIATION",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=12000),
        CausalEdge(source_id="OBJ_BARONETAGE", target_id="EVT_WILLIAM_RETURNS_TO_FAMILY",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=8500),
        CausalEdge(source_id="OBJ_WENTWORTHS_LETTER", target_id="EVT_RECONCILIATION",
                   causality_type="affordance_gate", mechanism="informational", evidence_strength="strong",
                   causal_force=10.0, fabula_time=12000),
        CausalEdge(source_id="OBJ_SMITH_PROPERTY", target_id="EVT_WENTWORTH_HELPS_MRS_SMITH",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=14000),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_KELLYNCH_HALL", target_id="ENT_SIR_WALTER",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=4.0, fabula_time=3000),
        CausalEdge(source_id="LOC_BATH", target_id="ENT_ELIZABETH",
                   causality_type="ambient_propagation", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=8500),
        CausalEdge(source_id="LOC_BATH", target_id="ENT_WILLIAM_ELLIOT",
                   causality_type="ambient_propagation", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=8500),
        CausalEdge(source_id="LOC_LYME_REGIS", target_id="ENT_LOUISA",
                   causality_type="ambient_propagation", mechanism="physical", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=7000),

        # ── WORLD_ → Event ──
        CausalEdge(source_id="WORLD_REGENCY_RANK", target_id="EVT_BROKEN_ENGAGEMENT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_REGENCY_RANK", target_id="EVT_KELLYNCH_LET",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3000),
        CausalEdge(source_id="WORLD_REGENCY_RANK", target_id="EVT_WILLIAM_RETURNS_TO_FAMILY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=8500),
        CausalEdge(source_id="WORLD_REGENCY_RANK", target_id="EVT_ANNE_VISITS_MRS_SMITH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=10000),
        CausalEdge(source_id="WORLD_NAVAL_PRIZE_ECONOMY", target_id="EVT_WAR_AND_PRIZE_MONEY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2000),
        CausalEdge(source_id="WORLD_NAVAL_PRIZE_ECONOMY", target_id="EVT_WENTWORTH_RETURNS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=4200),
        CausalEdge(source_id="WORLD_NAVAL_PRIZE_ECONOMY", target_id="EVT_KELLYNCH_LET",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=3000),
        CausalEdge(source_id="WORLD_PRIMOGENITURE_ENTAIL", target_id="EVT_WILLIAM_RETURNS_TO_FAMILY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=8500),
        CausalEdge(source_id="WORLD_PRIMOGENITURE_ENTAIL", target_id="EVT_MRS_SMITH_REVEALS_ELLIOT",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=6.0, fabula_time=10500),
        CausalEdge(source_id="WORLD_PRIMOGENITURE_ENTAIL", target_id="EVT_MRS_CLAY_FOLLOWS_WILLIAM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=13200),

        # ── orphan utterance wirings ──
        CausalEdge(source_id="EVT_UTT_LOUISA_TELLS_WENTWORTH", target_id="EVT_WENTWORTH_LEARNS_ANNE_REFUSED_CHARLES",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=6.0, fabula_time=5500, propagation_delay=0),
        CausalEdge(source_id="EVT_WILLIAM_RETURNS_TO_FAMILY", target_id="EVT_UTT_WILLIAM_FLATTERS_ANNE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=8500, propagation_delay=1000),
        CausalEdge(source_id="EVT_UTT_MRS_SMITH_REVEALS_WILLIAM", target_id="EVT_MRS_SMITH_REVEALS_ELLIOT",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=8.0, fabula_time=10500, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_ADMIRAL_CROFT_BENWICK_NEWS", target_id="EVT_WENTWORTH_TO_BATH",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=10800, propagation_delay=200),
        CausalEdge(source_id="EVT_UTT_ANNE_WOMEN_CONSTANCY", target_id="EVT_WENTWORTHS_LETTER",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=11800, propagation_delay=100),
        CausalEdge(source_id="EVT_UTT_WENTWORTHS_LETTER", target_id="EVT_RECONCILIATION",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=11900, propagation_delay=100),

        # ─── auto-patched mutation_social edges (per-axis coverage) ───
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE", rel_counterpart_id="ENT_LADY_RUSSELL", causality_type="mutation_social", trait_target="affinity", trait_delta=0.75, mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_LADY_RUSSELL", rel_counterpart_id="ENT_ANNE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.85, mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_SIR_WALTER", rel_counterpart_id="ENT_ANNE", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.1, mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_ANNE", rel_counterpart_id="ENT_SIR_WALTER", causality_type="mutation_social", trait_target="affinity", trait_delta=0.15, mechanism="psychological", evidence_strength="moderate", causal_force=3.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_ANNE_VISITS_UPPERCROSS", target_id="ENT_MARY", rel_counterpart_id="ENT_ANNE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.55, mechanism="emotional", evidence_strength="moderate", causal_force=5.0, fabula_time=4000, propagation_delay=0),
        CausalEdge(source_id="EVT_HAYTER_WITHDRAWS", target_id="ENT_HENRIETTA", rel_counterpart_id="ENT_CHARLES_HAYTER", causality_type="mutation_social", trait_target="affinity", trait_delta=0.85, mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_HAYTER_WITHDRAWS", target_id="ENT_CHARLES_HAYTER", rel_counterpart_id="ENT_HENRIETTA", causality_type="mutation_social", trait_target="affinity", trait_delta=0.85, mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_WENTWORTH_RETURNS", target_id="ENT_LOUISA", rel_counterpart_id="ENT_WENTWORTH", causality_type="mutation_social", trait_target="affinity", trait_delta=0.7, mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=4200, propagation_delay=0),
        CausalEdge(source_id="EVT_WENTWORTH_RETURNS", target_id="ENT_WENTWORTH", rel_counterpart_id="ENT_LOUISA", causality_type="mutation_social", trait_target="affinity", trait_delta=0.45, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=4200, propagation_delay=0),
        CausalEdge(source_id="EVT_WILLIAM_ADMIRES_ANNE_AT_LYME", target_id="ENT_WILLIAM_ELLIOT", rel_counterpart_id="ENT_ANNE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="social", evidence_strength="moderate", causal_force=6.0, fabula_time=6500, propagation_delay=0),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_MRS_CLAY", rel_counterpart_id="ENT_SIR_WALTER", causality_type="mutation_social", trait_target="affinity", trait_delta=0.7, mechanism="social", evidence_strength="moderate", causal_force=7.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_ANNE_VISITS_MRS_SMITH", target_id="ENT_ANNE", rel_counterpart_id="ENT_MRS_SMITH", causality_type="mutation_social", trait_target="affinity", trait_delta=0.85, mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=10000, propagation_delay=0),
        CausalEdge(source_id="EVT_ANNE_VISITS_MRS_SMITH", target_id="ENT_MRS_SMITH", rel_counterpart_id="ENT_ANNE", causality_type="mutation_social", trait_target="affinity", trait_delta=0.9, mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=10000, propagation_delay=0),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_ADMIRAL_CROFT", rel_counterpart_id="ENT_WENTWORTH", causality_type="mutation_social", trait_target="affinity", trait_delta=0.85, mechanism="social", evidence_strength="moderate", causal_force=6.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_SOPHIA_CROFT", rel_counterpart_id="ENT_WENTWORTH", causality_type="mutation_social", trait_target="affinity", trait_delta=0.95, mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_LADY_RUSSELL", rel_counterpart_id="ENT_ANNE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.6, mechanism="psychological", evidence_strength="strong", causal_force=9.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_ANNE_VISITS_MRS_SMITH", target_id="ENT_SIR_WALTER", rel_counterpart_id="ENT_ANNE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.55, mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=10000, propagation_delay=0),
        CausalEdge(source_id="EVT_WILLIAM_RETURNS_TO_FAMILY", target_id="ENT_WILLIAM_ELLIOT", rel_counterpart_id="ENT_ANNE", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.6, mechanism="social", evidence_strength="moderate", causal_force=7.0, fabula_time=8500, propagation_delay=0),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_MRS_CLAY", rel_counterpart_id="ENT_SIR_WALTER", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.5, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=3000, propagation_delay=0),

        # ─── added: negative-trait_delta mutations on Anne so Kahneman/Miller
        # regret + Berkowitz frustrated-rage scorers can attribute a loss event.
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=9.0, fabula_time=1000,
                   trait_target="hope", trait_delta=-0.7),
        CausalEdge(source_id="EVT_HAYTER_WITHDRAWS", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=3.0, fabula_time=5000,
                   trait_target="hope", trait_delta=-0.15),
        CausalEdge(source_id="EVT_WILLIAM_RETURNS_TO_FAMILY", target_id="ENT_ANNE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=8500,
                   trait_target="hope", trait_delta=-0.3),
        # ── auto-backfilled per-axis mutation_social ──
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE", rel_counterpart_id="ENT_ELIZABETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.06,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE", rel_counterpart_id="ENT_MARY",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.12,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_SIR_WALTER", rel_counterpart_id="ENT_MRS_CLAY",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.15,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAR_AND_PRIZE_MONEY", target_id="ENT_WENTWORTH", rel_counterpart_id="ENT_ADMIRAL_CROFT",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.24,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_WAR_AND_PRIZE_MONEY", target_id="ENT_WENTWORTH", rel_counterpart_id="ENT_SOPHIA_CROFT",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.27,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE", rel_counterpart_id="ENT_ELIZABETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.09,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_BROKEN_ENGAGEMENT", target_id="ENT_ANNE", rel_counterpart_id="ENT_MARY",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.06,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_KELLYNCH_LET", target_id="ENT_SIR_WALTER", rel_counterpart_id="ENT_MRS_CLAY",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.15,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=3000, propagation_delay=0),

        # ── WORLD_ → WORLD_ (named-latent forces destabilising one another) ──
        CausalEdge(source_id="WORLD_NAVAL_PRIZE_ECONOMY", target_id="WORLD_REGENCY_RANK",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4000,
                   description="Naval prize money lifts Wentworth across the rank gradient — the Napoleonic war is the mechanism by which a commoner becomes an eligible match."),
        CausalEdge(source_id="WORLD_PRIMOGENITURE_ENTAIL", target_id="WORLD_REGENCY_RANK",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1000,
                   description="The entail enforces the rank gradient by channelling baronetcies down the male line; Mrs Clay's threat to Sir Walter is a threat to that very mechanism."),
        CausalEdge(source_id="WORLD_REGENCY_RANK", target_id="WORLD_PRIMOGENITURE_ENTAIL",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=13000,
                   description="William Elliot's manoeuvres around the entail are themselves rank-coded — his courtship of Anne is calibrated to his place in the gradient."),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_KELLYNCH_HALL", target_id="LOC_UPPERCROSS"),
        SpatialEdge(source_id="LOC_UPPERCROSS", target_id="LOC_KELLYNCH_HALL"),
        SpatialEdge(source_id="LOC_UPPERCROSS", target_id="LOC_LYME_REGIS"),
        SpatialEdge(source_id="LOC_LYME_REGIS", target_id="LOC_HARVILLE_LODGINGS"),
        SpatialEdge(source_id="LOC_HARVILLE_LODGINGS", target_id="LOC_LYME_REGIS"),
        SpatialEdge(source_id="LOC_KELLYNCH_HALL", target_id="LOC_BATH"),
        SpatialEdge(source_id="LOC_UPPERCROSS", target_id="LOC_BATH"),
        SpatialEdge(source_id="LOC_LYME_REGIS", target_id="LOC_BATH"),
        SpatialEdge(source_id="LOC_BATH", target_id="LOC_MRS_SMITH_LODGINGS"),
        SpatialEdge(source_id="LOC_BATH", target_id="LOC_WHITE_HART"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    channels={
        # Mrs Smith's standing role as Anne's old-school-friend confidante in Bath —
        # she discloses information across multiple visits, not in one shot.
        'CHN_MRS_SMITH_CONFIDANT': Channel(
            id='CHN_MRS_SMITH_CONFIDANT',
            name="Mrs Smith ↔ Anne confidant relationship",
            medium='confidant',
            participant_ids=['ENT_MRS_SMITH', 'ENT_ANNE'],
            directionality='duplex',
            intelligibility={},
            established_at_fabula=10000, terminated_at_fabula=None,
            evidence_strength='strong',
        ),
        # William Elliot's standing register of refined, flattering, but opaque
        # discourse with Anne over the Bath weeks — Anne perceives only ~20 % of
        # its real intent (his cold instrumental motives).
        'CHN_WILLIAM_FLATTERING_DISCOURSE': Channel(
            id='CHN_WILLIAM_FLATTERING_DISCOURSE',
            name="William Elliot's flattering attentions to Anne",
            medium='flattering_discourse',
            participant_ids=['ENT_WILLIAM_ELLIOT', 'ENT_ANNE'],
            directionality='simplex',
            intelligibility={'ENT_ANNE': 0.2},
            established_at_fabula=8500, terminated_at_fabula=13000,
            evidence_strength='moderate',
        ),
    },

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_REGENCY_RANK": GlobalTrait(
            id="WORLD_REGENCY_RANK",
            name="Regency Rank Hierarchy",
            description="The fine-grained gradient of baronet, gentry, professional, naval officer, clergyman, and dependent that orders all introductions, marriages, and visiting practice in 1814 England. Operates as a common-cause parent over the original broken engagement, the Kellynch removal, and William Elliot's manoeuvres around the entail.",
            category="social_structure",
            magnitude=TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            proposition_id="PROP_RANK_BARS_LOVE",
            state_timeline=[
                WorldTraitSnapshot(fabula_time=12000, triggered_by="EVT_RECONCILIATION",
                    magnitude=TraitVector(value=0.7, inertia=0.85, evidence_strength="strong"),
                    description="Wentworth's marriage to Anne, sanctioned by Lady Russell, partially loosens the rank logic that originally divided them."),
            ],
        ),
        "WORLD_NAVAL_PRIZE_ECONOMY": GlobalTrait(
            id="WORLD_NAVAL_PRIZE_ECONOMY",
            name="Napoleonic Naval Prize Economy",
            description="The wartime British system under which captured enemy ships are sold and the proceeds shared among officers and crew. Wentworth's twenty-five thousand pounds — and the very fact of the Crofts' Kellynch tenancy — flow from this institution, which is the engine that returns Wentworth to Anne's social circle as an eligible match.",
            category="economy",
            magnitude=TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            affected_domains=["social"],
            proposition_id="PROP_WENTWORTH_RICH",
        ),
        "WORLD_PRIMOGENITURE_ENTAIL": GlobalTrait(
            id="WORLD_PRIMOGENITURE_ENTAIL",
            name="Primogeniture and the Kellynch Entail",
            description="The English law of inheritance that channels Kellynch Hall through the male line, making William Elliot heir presumptive and giving him a calculable interest in preventing Sir Walter's remarriage. The hidden structural cause of William's courtship of Anne and Mrs Clay's manoeuvres.",
            category="governance",
            magnitude=TraitVector(value=0.9, inertia=0.9, evidence_strength="strong"),
            affected_domains=["social"],
            proposition_id="PROP_WILLIAM_INHERITS",
        ),
    },

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────
    social_topology=[
        # Anne ↔ Wentworth — buried love.
        RelationshipEdge(
            source_entity_id="ENT_ANNE", target_entity_id="ENT_WENTWORTH",
            metrics={
                "affinity": RelationshipMetric(value=0.9, inertia=0.55, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_WENTWORTH", target_entity_id="ENT_ANNE",
            metrics={
                "affinity": RelationshipMetric(value=-0.4, inertia=0.5, evidence_strength="strong", last_updated_fabula=4200),
            },
        ),
        # Anne ↔ Lady Russell — daughter substitute.
        RelationshipEdge(
            source_entity_id="ENT_ANNE", target_entity_id="ENT_LADY_RUSSELL",
            metrics={
                "affinity": RelationshipMetric(value=0.75, inertia=0.7, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_LADY_RUSSELL", target_entity_id="ENT_ANNE",
            metrics={
                "affinity":      RelationshipMetric(value=0.85, inertia=0.7, evidence_strength="strong", last_updated_fabula=4000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.75, evidence_strength="strong", last_updated_fabula=1000),
            },
        ),
        # Sir Walter → Anne — neglect.
        RelationshipEdge(
            source_entity_id="ENT_SIR_WALTER", target_entity_id="ENT_ANNE",
            metrics={
                "affinity":      RelationshipMetric(value=-0.1, inertia=0.65, evidence_strength="strong", last_updated_fabula=4000),
                "power_dynamic": RelationshipMetric(value=0.55, inertia=0.75, evidence_strength="moderate", last_updated_fabula=1000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ANNE", target_entity_id="ENT_SIR_WALTER",
            metrics={
                "affinity": RelationshipMetric(value=0.15, inertia=0.6, evidence_strength="moderate", last_updated_fabula=4000),
            },
        ),
        # Elizabeth → Anne — disdain.
        RelationshipEdge(
            source_entity_id="ENT_ELIZABETH", target_entity_id="ENT_ANNE",
            metrics={
                "affinity": RelationshipMetric(value=0.0, inertia=0.65, evidence_strength="strong", last_updated_fabula=4000, observed=False),
            },
        ),
        # Anne → Elizabeth — patient resignation toward the elder sister who treats her as a non-person; muted negative affection, no fear, sister-of-elder-rank subordination.
        RelationshipEdge(
            source_entity_id="ENT_ANNE", target_entity_id="ENT_ELIZABETH",
            metrics={
                "affinity":      RelationshipMetric(value=-0.2, inertia=0.6,  evidence_strength="moderate", last_updated_fabula=4000),
                "power_dynamic": RelationshipMetric(value=-0.3, inertia=0.65, evidence_strength="moderate", last_updated_fabula=4000),
            },
        ),
        # Mary → Anne — clinging.
        RelationshipEdge(
            source_entity_id="ENT_MARY", target_entity_id="ENT_ANNE",
            metrics={
                "affinity": RelationshipMetric(value=0.55, inertia=0.55, evidence_strength="moderate", last_updated_fabula=4000),
            },
        ),
        # Anne → Mary — dutiful tolerance of the chronically self-pitying invalid sister at Uppercross; affectionate but exhausted.
        RelationshipEdge(
            source_entity_id="ENT_ANNE", target_entity_id="ENT_MARY",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.55, evidence_strength="moderate", last_updated_fabula=4000),
                "power_dynamic": RelationshipMetric(value=0.2, inertia=0.6,  evidence_strength="weak",     last_updated_fabula=4000),
            },
        ),
        # Henrietta ↔ Charles Hayter — settled love (Henrietta wavers toward Wentworth; Charles steadfast).
        RelationshipEdge(
            source_entity_id="ENT_HENRIETTA", target_entity_id="ENT_CHARLES_HAYTER",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.55, evidence_strength="strong", last_updated_fabula=5200),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_CHARLES_HAYTER", target_entity_id="ENT_HENRIETTA",
            metrics={
                "affinity": RelationshipMetric(value=0.9, inertia=0.6, evidence_strength="strong", last_updated_fabula=5000),
            },
        ),
        # Louisa ↔ Wentworth — flirtation.
        RelationshipEdge(
            source_entity_id="ENT_LOUISA", target_entity_id="ENT_WENTWORTH",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.5, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_WENTWORTH", target_entity_id="ENT_LOUISA",
            metrics={
                "affinity": RelationshipMetric(value=0.45, inertia=0.45, evidence_strength="moderate", last_updated_fabula=6000),
            },
        ),
        # Louisa ↔ Benwick — convalescent love (Louisa recovering bond; Benwick rebound from Fanny Harville).
        RelationshipEdge(
            source_entity_id="ENT_LOUISA", target_entity_id="ENT_BENWICK",
            metrics={
                "affinity": RelationshipMetric(value=0.8, inertia=0.55, evidence_strength="strong", last_updated_fabula=9000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_BENWICK", target_entity_id="ENT_LOUISA",
            metrics={
                "affinity": RelationshipMetric(value=0.9, inertia=0.55, evidence_strength="strong", last_updated_fabula=9000),
            },
        ),
        # William → Anne — instrumental courtship.
        RelationshipEdge(
            source_entity_id="ENT_WILLIAM_ELLIOT", target_entity_id="ENT_ANNE",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.45, evidence_strength="moderate", last_updated_fabula=9000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.65, evidence_strength="moderate", last_updated_fabula=9000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ANNE", target_entity_id="ENT_WILLIAM_ELLIOT",
            metrics={
                "affinity": RelationshipMetric(value=-0.6, inertia=0.5, evidence_strength="strong", last_updated_fabula=10500),
            },
        ),
        # Mrs Clay → Sir Walter — designs.
        RelationshipEdge(
            source_entity_id="ENT_MRS_CLAY", target_entity_id="ENT_SIR_WALTER",
            metrics={
                "affinity":      RelationshipMetric(value=0.7, inertia=0.55, evidence_strength="moderate", last_updated_fabula=3000),
                "power_dynamic": RelationshipMetric(value=-0.5, inertia=0.7, evidence_strength="moderate", last_updated_fabula=3000),
            },
        ),
        # Sir Walter → Mrs Clay — vain susceptibility to the flattering young widow he keeps as a household companion in Bath; unaware of her freckled designs on the title (sign-flipped power).
        RelationshipEdge(
            source_entity_id="ENT_SIR_WALTER", target_entity_id="ENT_MRS_CLAY",
            metrics={
                "affinity":      RelationshipMetric(value=0.5, inertia=0.55, evidence_strength="moderate", last_updated_fabula=3000),
                "power_dynamic": RelationshipMetric(value=0.5, inertia=0.7,  evidence_strength="moderate", last_updated_fabula=3000),
            },
        ),
        # Anne ↔ Mrs Smith — true friendship.
        RelationshipEdge(
            source_entity_id="ENT_ANNE", target_entity_id="ENT_MRS_SMITH",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.7, evidence_strength="strong", last_updated_fabula=10000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MRS_SMITH", target_entity_id="ENT_ANNE",
            metrics={
                "affinity": RelationshipMetric(value=0.9, inertia=0.7, evidence_strength="strong", last_updated_fabula=10500),
            },
        ),
        # Admiral & Sophia Croft → Wentworth — kin.
        RelationshipEdge(
            source_entity_id="ENT_ADMIRAL_CROFT", target_entity_id="ENT_WENTWORTH",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.75, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        # Wentworth → Admiral Croft — affectionate fellow-officer respect for his bluff brother-in-law who gives him Kellynch as a base for re-encountering Anne.
        RelationshipEdge(
            source_entity_id="ENT_WENTWORTH", target_entity_id="ENT_ADMIRAL_CROFT",
            metrics={
                "affinity": RelationshipMetric(value=0.8, inertia=0.75, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_SOPHIA_CROFT", target_entity_id="ENT_WENTWORTH",
            metrics={
                "affinity": RelationshipMetric(value=0.95, inertia=0.8, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        # Wentworth → Sophia Croft — deep sibling closeness with the sister who shipboard-sailed with her admiral; she is his moral compass on the question of marrying Louisa.
        RelationshipEdge(
            source_entity_id="ENT_WENTWORTH", target_entity_id="ENT_SOPHIA_CROFT",
            metrics={
                "affinity": RelationshipMetric(value=0.9, inertia=0.8, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
    ],

    # ── PROPOSITIONS ────────────────────────────────────────────────────
    propositions=[
        Proposition(proposition_id="PROP_WENTWORTH_LOVES_ANNE", kind="trait_holds",
                    referent_ids=["ENT_WENTWORTH", "ENT_ANNE"],
                    description="Captain Wentworth still loves Anne Elliot.",
                    audience_default_prior=0.55, stakes=0.95,
                    truth_at_fabula={11900: True, 12000: True}),
        Proposition(proposition_id="PROP_ANNE_LOVES_WENTWORTH", kind="trait_holds",
                    referent_ids=["ENT_ANNE", "ENT_WENTWORTH"],
                    description="Anne Elliot has loved Wentworth constantly since the broken engagement.",
                    audience_default_prior=0.7, stakes=0.9,
                    truth_at_fabula={1000: True, 12000: True}),
        Proposition(proposition_id="PROP_PERSUASION_WAS_RIGHT", kind="outcome",
                    referent_ids=["EVT_BROKEN_ENGAGEMENT"],
                    description="Lady Russell's counsel to break the engagement was the right judgement.",
                    audience_default_prior=0.3, stakes=0.85,
                    truth_at_fabula={12000: False}),
        Proposition(proposition_id="PROP_WILLIAM_ELLIOT_HONOURABLE", kind="trait_holds",
                    referent_ids=["ENT_WILLIAM_ELLIOT"],
                    description="William Elliot is the honourable suitor he presents himself as.",
                    audience_default_prior=0.45, stakes=0.85,
                    truth_at_fabula={10500: False}),
        Proposition(proposition_id="PROP_ANNE_MARRIES_WILLIAM", kind="event_occurs",
                    referent_ids=["ENT_ANNE", "ENT_WILLIAM_ELLIOT"],
                    description="Anne Elliot marries her cousin William Elliot.",
                    audience_default_prior=0.4, stakes=0.85,
                    truth_at_fabula={12000: False}),
        Proposition(proposition_id="PROP_LOUISA_WINS_WENTWORTH", kind="event_occurs",
                    referent_ids=["ENT_LOUISA", "ENT_WENTWORTH"],
                    description="Louisa Musgrove becomes engaged to Captain Wentworth.",
                    audience_default_prior=0.55, stakes=0.8,
                    truth_at_fabula={9000: False}),
        Proposition(proposition_id="PROP_HENRIETTA_MARRIES_HAYTER", kind="event_occurs",
                    referent_ids=["ENT_HENRIETTA", "ENT_CHARLES_HAYTER"],
                    description="Henrietta Musgrove marries her cousin Charles Hayter.",
                    audience_default_prior=0.65, stakes=0.5,
                    truth_at_fabula={2000: False}),
        Proposition(proposition_id="PROP_KELLYNCH_KEPT", kind="event_occurs",
                    referent_ids=["LOC_KELLYNCH_HALL", "ENT_SIR_WALTER"],
                    description="Sir Walter retains Kellynch Hall without leasing it.",
                    audience_default_prior=0.4, stakes=0.7,
                    truth_at_fabula={3000: False}),
        Proposition(proposition_id="PROP_MRS_CLAY_MARRIES_SIR_WALTER", kind="event_occurs",
                    referent_ids=["ENT_MRS_CLAY", "ENT_SIR_WALTER"],
                    description="Mrs Clay succeeds in marrying Sir Walter.",
                    audience_default_prior=0.45, stakes=0.85,
                    truth_at_fabula={13200: False}),
        Proposition(proposition_id="PROP_MRS_CLAY_EXPOSED", kind="event_occurs",
                    referent_ids=["ENT_MRS_CLAY"],
                    description="Mrs Clay's ambition is exposed to the Elliot family.",
                    audience_default_prior=0.4, stakes=0.6,
                    truth_at_fabula={13200: True}),
        Proposition(proposition_id="PROP_WILLIAM_INHERITS", kind="outcome",
                    referent_ids=["ENT_WILLIAM_ELLIOT", "OBJ_BARONETAGE"],
                    description="William Elliot inherits the Kellynch baronetcy unobstructed.",
                    audience_default_prior=0.6, stakes=0.7,
                    truth_at_fabula={13200: True}),
        Proposition(proposition_id="PROP_WENTWORTH_RICH", kind="trait_holds",
                    referent_ids=["ENT_WENTWORTH"],
                    description="Wentworth has made his fortune in naval prize money.",
                    audience_default_prior=0.5, stakes=0.6,
                    truth_at_fabula={4000: True}),
        Proposition(proposition_id="PROP_MRS_SMITH_RELIEVED", kind="event_occurs",
                    referent_ids=["ENT_MRS_SMITH"],
                    description="Mrs Smith's West Indian property is recovered through Wentworth's help.",
                    audience_default_prior=0.35, stakes=0.55,
                    truth_at_fabula={14100: True}),
        # WORLD_ trait Pearl-Rung-2 reification (Regency Rank).
        Proposition(proposition_id="PROP_RANK_BARS_LOVE", kind="trait_holds",
                    referent_ids=["WORLD_REGENCY_RANK", "ENT_ANNE", "ENT_WENTWORTH"],
                    description="The Regency rank hierarchy still bars a baronet's daughter from a naval commander — the original ground for the broken engagement remains intact.",
                    audience_default_prior=0.7, stakes=0.85,
                    truth_at_fabula={1000: True, 12000: False}),
    ],
)
