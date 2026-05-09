# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Brief Encounter — high-fidelity WorldStateV1 test fixture.

Authored against the current ingestion prompts. Demonstrates all five
CausalEdge modalities, per-axis ``RelationshipMetric``, explicit
``evidence_strength``, and named-latent WORLD_ traits (interwar
respectability, the Suburban-Married domesticity machine, the
railway timetable as cosmology) wired as common-cause parents over
the events they jointly drive.
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
        target_word_min=205,
        target_word_max=768,
        prose_density='sparse',
        voice='synoptic narration; no dialogue; condensed scene description; third-person POV; present tense',
        style_exemplar="In a railway station refreshment room, a man and a woman are sitting glumly at a table. One of the woman's friends comes in and immediately begins chatting away. After some terse pleasantries, the man leaves to catch his train. The woman explains that the man is about to move to Africa. After the man leaves, the woman abruptly disappears but soon returns, explaining that she wanted to see the express train pass by. After a short interval, the two women head for their own train.\n\nAt home, Laura Jesson, the woman at the table, sits in the living room with her husband Fred.",
        source_word_count=512,
    ),
    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_REFRESHMENT_ROOM": Location(
            name="Milford Junction Refreshment Room",
            description="The bright tea-room on the up platform where Laura and Alec snatch their weekly meetings — and where their farewell is wrecked by Dolly Messiter.",
            ambient_state={
                "respectability_under_glass": AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
                "constant_supervision":       AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
                # Frijda action-readiness: Dolly's interruption is the social
                # trap par excellence — physical exits exist but every gesture
                # is observed; flight from feeling, not from the room, denied.
                "connected_to": AmbientVector(value=0.4, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_PLATFORM": Location(
            name="Milford Junction Platform",
            description="The platform itself, with its express trains and timetabled separations.",
            ambient_state={
                "physical_danger":        AmbientVector(value=0.5, volatility=0.4, evidence_strength="moderate"),
                "imposed_separation":     AmbientVector(value=0.95, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_BOTANICAL_GARDENS": Location(
            name="Milford Botanical Gardens",
            description="Their first walk together; safely public, deniably innocent.",
            ambient_state={
                "deniable_innocence": AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_KARDOMAH": Location(
            name="Kardomah Café and Cinema",
            description="Where the relationship deepens over chops and matinées.",
            ambient_state={
                "rising_intimacy":    AmbientVector(value=0.7, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_STEPHENS_FLAT": Location(
            name="Stephen Lynn's Flat",
            description="The borrowed flat of Alec's friend Stephen where Laura and Alec almost cross the line — and where Stephen's early return makes their meeting impossible.",
            ambient_state={
                "borrowed_secrecy":   AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
                "humiliation_risk":   AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
                # Stephen's key is in the door; flight from exposure unavailable.
                "connected_to": AmbientVector(value=0.0, volatility=0.4, evidence_strength="strong"),
            },
        ),
        "LOC_MILFORD_STREETS": Location(
            name="Milford Streets in Rain",
            description="The streets along which Laura wanders in shame after the flat episode until a constable urges her home.",
            ambient_state={
                "exposure":           AmbientVector(value=0.8, volatility=0.4, evidence_strength="strong"),
                "moral_chill":        AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_LAURAS_HOME": Location(
            name="Laura's Suburban Home",
            description="The Jessons' Ketchworth sitting-room with its wireless and Fred's crossword.",
            ambient_state={
                "married_routine":    AmbientVector(value=0.95, volatility=0.05, evidence_strength="strong"),
                "domestic_safety":    AmbientVector(value=0.85, volatility=0.1, evidence_strength="strong"),
                # Marriage as gentle gaol — physically open but morally sealed.
                "connected_to": AmbientVector(value=0.4, volatility=0.05, evidence_strength="strong"),
            },
        ),
    },

    # ── OBJECTS ────────────────────────────────────────────────────────
    objects={
        "OBJ_GRIT_IN_EYE": NarrativeObject(
            id="OBJ_GRIT_IN_EYE", name="Speck of Grit in Laura's Eye",
            location_id="LOC_REFRESHMENT_ROOM", owner_id="ENT_LAURA",
            properties={"state": "needs_removing", "function": "ostensible_pretext"},
            affordances=[Affordance(action="legitimise_first_contact", target_type="Entity")],
        ),
        "OBJ_TIMETABLE": NarrativeObject(
            id="OBJ_TIMETABLE", name="Milford Junction Timetable",
            location_id="LOC_REFRESHMENT_ROOM", owner_id=None,
            properties={"state": "fixed", "thursday_pattern": "alec_at_hospital"},
            affordances=[Affordance(action="impose_separation", target_type="Entity")],
        ),
        "OBJ_STEPHENS_KEY": NarrativeObject(
            id="OBJ_STEPHENS_KEY", name="Stephen Lynn's Latchkey",
            location_id="LOC_STEPHENS_FLAT", owner_id="ENT_ALEC",
            properties={"state": "borrowed", "lender": "ENT_STEPHEN"},
            affordances=[Affordance(action="enable_planned_adultery", target_type="Entity")],
        ),
        "OBJ_DOLLY_VOICE": NarrativeObject(
            id="OBJ_DOLLY_VOICE", name="Dolly's Unstoppable Chatter",
            location_id="LOC_REFRESHMENT_ROOM", owner_id="ENT_DOLLY",
            properties={"state": "incessant", "obliviousness": "total"},
            affordances=[Affordance(action="prevent_proper_goodbye", target_type="Entity")],
        ),
        "OBJ_EXPRESS_TRAIN": NarrativeObject(
            id="OBJ_EXPRESS_TRAIN", name="Through-Express",
            location_id="LOC_PLATFORM", owner_id=None,
            properties={"state": "scheduled", "function": "near_suicide_temptation"},
            affordances=[Affordance(action="end_a_life", target_type="Entity")],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    entities={
        "ENT_LAURA": Entity(
            id="ENT_LAURA", name="Laura Jesson",
            location_id="LOC_LAURAS_HOME", status="healthy",
            traits={
                "respectability":   TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "longing":          TraitVector(value=0.4, inertia=0.45, evidence_strength="moderate"),
                "guilt":            TraitVector(value=0.3, inertia=0.45, evidence_strength="moderate"),
                "moral_seriousness": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "love_for_alec":    TraitVector(value=0.0, inertia=0.4, evidence_strength="weak"),
                "marital_loyalty":  TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_FRED",
                       perceived_state="kind, settled husband — utterly familiar", proposition_id="PROP_LAURA_LEAVES_FRED",
                       confidence=0.95, inertia=0.85, established_at_fabula=0, evidence_strength="strong"),
            ],
            concerns=[
                # Sternberg/Berscheid passionate attachment that ignites at the
                # cinema and never quite extinguishes.
                Concern(concern_id="CCN_LAURA_DESIRES_ALEC", proposition_id="PROP_ALEC_LOVES_LAURA",
                        polarity="desire", kind="love", salience=1.0,
                        activation_fabula_window=[3000, 12000],
                        counter_concern_ids=["CCN_LAURA_FEARS_ABANDONING_FRED"]),
                # Bowlby attachment + moral identity — Laura's marriage is the
                # counterweight that prevents the elopement.
                Concern(concern_id="CCN_LAURA_FEARS_ABANDONING_FRED", proposition_id="PROP_LAURA_LEAVES_FRED",
                        polarity="fear", kind="abandonment", salience=0.95,
                        activation_fabula_window=[4000, 12000],
                        counter_concern_ids=["CCN_LAURA_DESIRES_ALEC", "CCN_LAURA_DESIRES_ESCAPE"]),
                # Averill normative-violation rage substrate — exposure / shame
                # if their meetings are observed; intensified by the cinema sighting.
                Concern(concern_id="CCN_LAURA_FEARS_EXPOSURE", proposition_id="PROP_AFFAIR_EXPOSED",
                        polarity="fear", kind="humiliation", salience=0.9,
                        activation_fabula_window=[5000, 12000],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=5000, triggered_by="EVT_FRIENDS_SEE_THEM",
                                            salience=1.0),
                        ]),
                # Lazarus appraisal — Stephen's flat as the moral chasm that
                # Laura wants the affair to *not* cross.
                Concern(concern_id="CCN_LAURA_FEARS_CONSUMMATION", proposition_id="PROP_AFFAIR_CONSUMMATED",
                        polarity="fear", kind="injustice", salience=0.85,
                        activation_fabula_window=[6000, 9000]),
                # Kahneman/Miller foreshadowed regret — the road not taken.
                Concern(concern_id="CCN_LAURA_DESIRES_ESCAPE", proposition_id="PROP_LAURA_LEAVES_FRED",
                        polarity="desire", kind="freedom", salience=0.6,
                        activation_fabula_window=[6000, 10500],
                        counter_concern_ids=["CCN_LAURA_FEARS_ABANDONING_FRED"],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=9000, triggered_by="EVT_AGREE_TO_END",
                                            salience=0.2, kind="regret"),
                        ]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=1000, triggered_by="EVT_GRIT_IN_EYE",
                    location_id="LOC_REFRESHMENT_ROOM"),
                EntityStateSnapshot(fabula_time=4000, triggered_by="EVT_KARDOMAH_CINEMA",
                    location_id="LOC_KARDOMAH",
                    traits={
                        "longing":      TraitVector(value=0.7, inertia=0.55, evidence_strength="strong"),
                        "love_for_alec": TraitVector(value=0.55, inertia=0.5, evidence_strength="strong"),
                    }),
                # Parity snapshot for EVT_FRIENDS_SEE_THEM mutation → guilt +0.25.
                EntityStateSnapshot(fabula_time=5000, triggered_by="EVT_FRIENDS_SEE_THEM",
                    traits={
                        "guilt": TraitVector(value=0.5, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=7000, triggered_by="EVT_STEPHEN_RETURNS",
                    location_id="LOC_STEPHENS_FLAT",
                    traits={
                        "guilt":  TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
                        "respectability": TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=7500, triggered_by="EVT_WANDER_STREETS",
                    location_id="LOC_MILFORD_STREETS"),
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_AGREE_TO_END",
                    traits={
                        "moral_seriousness": TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=10000, triggered_by="EVT_FINAL_MEETING",
                    location_id="LOC_REFRESHMENT_ROOM"),
                EntityStateSnapshot(fabula_time=10500, triggered_by="EVT_NEAR_SUICIDE",
                    location_id="LOC_PLATFORM",
                    traits={
                        "longing":  TraitVector(value=0.95, inertia=0.6, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=12000, triggered_by="EVT_RETURN_TO_FRED",
                    location_id="LOC_LAURAS_HOME",
                    traits={
                        # Longing partly relaxes after she chooses Fred — matches the
                        # CausalEdge EVT_RETURN_TO_FRED → ENT_LAURA longing -0.4.
                        "longing":         TraitVector(value=0.55, inertia=0.6, evidence_strength="strong"),
                        # Love for Alec persists undimmed (no edge erases it); kept as
                        # the high-inertia residue she carries home.
                        "love_for_alec":   TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                        "marital_loyalty": TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_FRED",
                               perceived_state="he knows, more or less, and still wants me back", proposition_id="PROP_LAURA_LEAVES_FRED",
                               confidence=0.85, inertia=0.7, established_at_fabula=12000, evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_ALEC": Entity(
            id="ENT_ALEC", name="Dr Alec Harvey",
            location_id="LOC_REFRESHMENT_ROOM", status="healthy",
            traits={
                "professional_idealism": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "tenderness":           TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "self_division":        TraitVector(value=0.4, inertia=0.4, evidence_strength="moderate"),
                "love_for_laura":       TraitVector(value=0.0, inertia=0.4, evidence_strength="weak"),
                "marital_duty":         TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[],
            concerns=[
                Concern(concern_id="CCN_ALEC_DESIRES_LAURA", proposition_id="PROP_LAURA_LOVES_ALEC",
                        polarity="desire", kind="love", salience=1.0,
                        activation_fabula_window=[3000, 11000],
                        counter_concern_ids=["CCN_ALEC_FEARS_LOSING_FAMILY"]),
                Concern(concern_id="CCN_ALEC_FEARS_LOSING_FAMILY", proposition_id="PROP_ALEC_LEAVES_FAMILY",
                        polarity="fear", kind="abandonment", salience=0.9,
                        activation_fabula_window=[4000, 11000],
                        counter_concern_ids=["CCN_ALEC_DESIRES_LAURA"]),
                # Averill normative violation — Stephen's chastisement at the flat.
                Concern(concern_id="CCN_ALEC_FEARS_HUMILIATION", proposition_id="PROP_AFFAIR_EXPOSED",
                        polarity="fear", kind="humiliation", salience=0.85,
                        activation_fabula_window=[6000, 11000]),
                # Lazarus appraisal — the Johannesburg post as Frijda flight.
                Concern(concern_id="CCN_ALEC_DESIRES_NEW_LIFE", proposition_id="PROP_ALEC_TAKES_POST",
                        polarity="desire", kind="freedom", salience=0.7,
                        activation_fabula_window=[9000, 11000],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=9000, triggered_by="EVT_AGREE_TO_END",
                                            salience=0.95),
                        ]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=4000, triggered_by="EVT_KARDOMAH_CINEMA",
                    traits={
                        "love_for_laura": TraitVector(value=0.55, inertia=0.5, evidence_strength="strong"),
                        "self_division":  TraitVector(value=0.7, inertia=0.55, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=7000, triggered_by="EVT_STEPHEN_RETURNS",
                    traits={
                        "self_division": TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_AGREE_TO_END",
                    traits={
                        "marital_duty": TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_LAURA",
                               perceived_state="she must be allowed to keep her family",
                               proposition_id="PROP_LAURA_LEAVES_FRED",
                               confidence=0.95, inertia=0.85, established_at_fabula=9000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=11000, triggered_by="EVT_ALEC_DEPARTS",
                    location_id=None),
            ],
        ),
        "ENT_FRED": Entity(
            id="ENT_FRED", name="Fred Jesson",
            location_id="LOC_LAURAS_HOME", status="healthy",
            traits={
                "settled_kindness":  TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
                "obtuseness":        TraitVector(value=0.55, inertia=0.7, evidence_strength="moderate"),
                "tact":              TraitVector(value=0.7, inertia=0.7, evidence_strength="moderate"),
                "marital_loyalty":   TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=12000, triggered_by="EVT_RETURN_TO_FRED",
                    beliefs_added=[
                        Belief(target_id="ENT_LAURA",
                               perceived_state="something happened, but she has come back", proposition_id="PROP_AFFAIR_EXPOSED",
                               confidence=0.7, inertia=0.7, established_at_fabula=12000, evidence_strength="moderate"),
                    ]),
            ],
        ),
        "ENT_DOLLY": Entity(
            id="ENT_DOLLY", name="Dolly Messiter",
            location_id="LOC_REFRESHMENT_ROOM", status="healthy",
            traits={
                "obliviousness":   TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "garrulousness":   TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "social_currency": TraitVector(value=0.7, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[],
        ),
        "ENT_STEPHEN": Entity(
            id="ENT_STEPHEN", name="Stephen Lynn",
            location_id="LOC_STEPHENS_FLAT", status="healthy",
            traits={
                "discreet_disapproval": TraitVector(value=0.85, inertia=0.8, evidence_strength="strong"),
                "professional_loyalty": TraitVector(value=0.7, inertia=0.7, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_ALEC",
                       perceived_state="my colleague is breaking the rules and dragging me into it", proposition_id="PROP_AFFAIR_EXPOSED",
                       confidence=0.95, inertia=0.8, established_at_fabula=7000, evidence_strength="strong"),
            ],
        ),
        "ENT_MYRTLE_PALMER": Entity(
            id="ENT_MYRTLE_PALMER", name="Myrtle, Refreshment-Room Manageress",
            location_id="LOC_REFRESHMENT_ROOM", status="healthy",
            traits={
                "managerial_dignity": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "decorousness":       TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
                "stifled_warmth":     TraitVector(value=0.6, inertia=0.6, evidence_strength="moderate"),
            },
            beliefs=[],
        ),
        "ENT_ALBERT_GODBY": Entity(
            id="ENT_ALBERT_GODBY", name="Albert Godby, Ticket Inspector",
            location_id="LOC_REFRESHMENT_ROOM", status="healthy",
            traits={
                "cheerful_persistence": TraitVector(value=0.9, inertia=0.8, evidence_strength="strong"),
                "courtship_intent":     TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
            },
            beliefs=[],
        ),
        "ENT_BERYL": Entity(
            id="ENT_BERYL", name="Beryl Walters, Refreshment-Room Assistant",
            location_id="LOC_REFRESHMENT_ROOM", status="healthy",
            traits={
                "youthful_flirtation": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "shyness":             TraitVector(value=0.6, inertia=0.6, evidence_strength="moderate"),
            },
            beliefs=[],
        ),
        "ENT_STANLEY": Entity(
            id="ENT_STANLEY", name="Stanley, Pastry Boy",
            location_id="LOC_REFRESHMENT_ROOM", status="healthy",
            traits={
                "youthful_irreverence": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "playful_flirtation":   TraitVector(value=0.7, inertia=0.65, evidence_strength="moderate"),
            },
            beliefs=[],
        ),
        "ENT_POLICEMAN": Entity(
            id="ENT_POLICEMAN", name="Concerned Constable",
            location_id="LOC_MILFORD_STREETS", status="healthy",
            traits={
                "kindly_authority": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[],
        ),
        # Group entity: Laura's middle-class Milford acquaintances who catch
        # her and Alec coming out of the cinema. Distinct from Dolly Messiter,
        # who only enters at the final refreshment-room scene.
        "ENT_LAURAS_ACQUAINTANCES": Entity(
            id="ENT_LAURAS_ACQUAINTANCES", name="Laura's Milford Acquaintances",
            location_id="LOC_KARDOMAH", status="healthy",
            traits={
                "social_observation": TraitVector(value=0.8, inertia=0.7, evidence_strength="strong"),
                "gossip_propensity":  TraitVector(value=0.7, inertia=0.7, evidence_strength="moderate"),
            },
            beliefs=[],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_GRIT_IN_EYE", fabula_time=1000, syuzhet_index=3,
                  event_type="choice", actor_ids=["ENT_ALEC"], target_ids=["ENT_LAURA"],
                  description="In the refreshment room a piece of grit blows into Laura's eye and Alec, the doctor on the next stool, gently removes it — their innocent first contact."),
        EventNode(id="EVT_CHEMIST_MEETING", fabula_time=2000, syuzhet_index=4,
                  event_type="choice", actor_ids=["ENT_LAURA", "ENT_ALEC"], target_ids=[],
                  description="The following Thursday Laura runs into Alec at Boots' the chemist; they exchange a few sentences and arrange — without saying so — to meet again."),
        EventNode(id="EVT_BOTANICAL_WALK", fabula_time=3000, syuzhet_index=5,
                  event_type="choice", actor_ids=["ENT_LAURA", "ENT_ALEC"], target_ids=[],
                  description="They stroll together in the Botanical Gardens, ostensibly innocently, while Alec talks about his idealistic interest in industrial pneumonoconiosis."),
        EventNode(id="EVT_KARDOMAH_CINEMA", fabula_time=4000, syuzhet_index=6,
                  event_type="choice", actor_ids=["ENT_LAURA", "ENT_ALEC"], target_ids=[],
                  description="They have lunch at the Kardomah and a matinée together; Alec admits, and Laura half-admits, that they are in love."),
        EventNode(id="EVT_FRIENDS_SEE_THEM", fabula_time=5000, syuzhet_index=7,
                  event_type="outcome", actor_ids=["ENT_LAURAS_ACQUAINTANCES"], target_ids=["ENT_LAURA", "ENT_ALEC"],
                  description="Coming out of the cinema they are caught by Laura's acquaintances; Laura makes the first of many small, fluent lies."),
        EventNode(id="EVT_BORROW_FLAT_PLAN", fabula_time=6000, syuzhet_index=8,
                  event_type="choice", actor_ids=["ENT_ALEC"], target_ids=["ENT_STEPHEN"],
                  description="Alec arranges to borrow Stephen Lynn's flat on a Thursday evening; Laura agrees to meet him there."),
        EventNode(id="EVT_STEPHEN_RETURNS", fabula_time=7000, syuzhet_index=9,
                  event_type="outcome", actor_ids=["ENT_STEPHEN"], target_ids=["ENT_LAURA", "ENT_ALEC"],
                  description="Stephen returns unexpectedly from his cancelled dinner; Laura escapes down the back stairs but Stephen knows what was happening and quietly chastises Alec."),
        EventNode(id="EVT_WANDER_STREETS", fabula_time=7500, syuzhet_index=10,
                  event_type="choice", actor_ids=["ENT_LAURA"], target_ids=[],
                  description="Devastated, Laura wanders Milford in the rain for three hours until a constable persuades her to go home."),
        EventNode(id="EVT_AGREE_TO_END", fabula_time=9000, syuzhet_index=11,
                  event_type="choice", actor_ids=["ENT_LAURA", "ENT_ALEC"], target_ids=[],
                  description="At the railway station, the lovers acknowledge that they cannot make a life together; Alec resolves to take a hospital appointment in Johannesburg."),
        EventNode(id="EVT_FINAL_MEETING", fabula_time=10000, syuzhet_index=1,
                  event_type="outcome", actor_ids=["ENT_DOLLY"], target_ids=["ENT_LAURA", "ENT_ALEC"],
                  description="Their last meeting in the refreshment room is wrecked by Dolly Messiter's oblivious chatter; they cannot say what they meant to say."),
        EventNode(id="EVT_ALEC_DEPARTS", fabula_time=10200, syuzhet_index=2,
                  event_type="choice", actor_ids=["ENT_ALEC"], target_ids=["ENT_LAURA"],
                  description="Alec squeezes Laura's shoulder in lieu of a goodbye and boards his train, beginning the journey to Johannesburg."),
        EventNode(id="EVT_NEAR_SUICIDE", fabula_time=10500, syuzhet_index=12,
                  event_type="choice", actor_ids=["ENT_LAURA"], target_ids=["ENT_LAURA"],
                  description="On the platform Laura nearly steps in front of the through-express; she pulls back at the last moment."),
        EventNode(id="EVT_RETURN_TO_FRED", fabula_time=12000, syuzhet_index=13,
                  event_type="choice", actor_ids=["ENT_LAURA", "ENT_FRED"], target_ids=[],
                  description="Laura returns home; Fred, sensing something without naming it, thanks her for coming back to him and she weeps in his arms."),
        EventNode(id="EVT_STAFF_FLIRTATION_ARC", fabula_time=6000, syuzhet_index=14,
                  event_type="choice", actor_ids=["ENT_ALBERT_GODBY", "ENT_MYRTLE_PALMER", "ENT_BERYL", "ENT_STANLEY"], target_ids=[],
                  description="In counterpoint, the refreshment-room staff carry on their own working-class flirtations, openly visible but checked by the rules of decorum."),
    
        # ── UTTERANCES (on-page speech-acts) ──
        EventNode(id='EVT_UTT_LAURA_LIES_TO_FRIENDS', event_type='utterance',
                  description="Caught coming out of the cinema, Laura tells a small fluent lie to her acquaintances about who Alec is and why they are together.",
                  content="Oh — this is Dr Harvey, a friend; we ran into each other quite by chance.",
                  speaker_id='ENT_LAURA', addressee_ids=['ENT_LAURAS_ACQUAINTANCES'], actor_ids=['ENT_LAURA'],
                  target_ids=['ENT_ALEC', 'EVT_KARDOMAH_CINEMA'],
                  via_channel_id=None, truth_value='false', fabula_time=5000, syuzhet_index=15),
        EventNode(id='EVT_UTT_STEPHEN_CHASTISES_ALEC', event_type='utterance',
                  description="Stephen, having overheard Laura sneaking out of his flat, subtly chides Alec for the infidelity.",
                  content="I'm sorry, Alec — I had no idea you were using the flat for that sort of thing.",
                  speaker_id='ENT_STEPHEN', addressee_ids=['ENT_ALEC'], actor_ids=['ENT_STEPHEN'],
                  target_ids=['ENT_LAURA', 'EVT_STEPHEN_RETURNS'],
                  via_channel_id=None, truth_value='true', fabula_time=7000, syuzhet_index=16),
        EventNode(id='EVT_UTT_POLICEMAN_URGES_HOME', event_type='utterance',
                  description="A concerned constable, finding Laura wandering the rainy streets, kindly urges her to go home.",
                  content="You all right, madam? Best be getting home now — it's late and cold.",
                  speaker_id='ENT_POLICEMAN', addressee_ids=['ENT_LAURA'], actor_ids=['ENT_POLICEMAN'],
                  target_ids=['ENT_LAURA'],
                  via_channel_id=None, truth_value='performative', fabula_time=7500, syuzhet_index=17),
        EventNode(id='EVT_UTT_DOLLY_CHATTERS', event_type='utterance',
                  description="Dolly Messiter prattles obliviously over Laura and Alec's last few minutes together in the refreshment room.",
                  content="What a lovely surprise to find you here! Now do tell me everything — I haven't seen you for ages.",
                  speaker_id='ENT_DOLLY', addressee_ids=['ENT_LAURA', 'ENT_ALEC'], actor_ids=['ENT_DOLLY'],
                  target_ids=['EVT_FINAL_MEETING'],
                  via_channel_id=None, truth_value='true', fabula_time=10000, syuzhet_index=18),
        EventNode(id='EVT_UTT_ALEC_SHOULDER_SQUEEZE', event_type='utterance',
                  description="Unable to speak in front of Dolly, Alec discreetly squeezes Laura's shoulder as a wordless farewell before boarding his train.",
                  content="(a brief, deliberate squeeze of the shoulder — their entire goodbye)",
                  speaker_id='ENT_ALEC', addressee_ids=['ENT_LAURA'], actor_ids=['ENT_ALEC'],
                  target_ids=['ENT_LAURA', 'EVT_ALEC_DEPARTS'],
                  via_channel_id=None, truth_value='performative', fabula_time=10200, syuzhet_index=19),
        EventNode(id='EVT_UTT_FRED_THANKS_LAURA', event_type='utterance',
                  description="Fred, sensing Laura's distance without naming its cause, thanks her for coming back to him.",
                  content="You've been a long way away. Thank you for coming back to me.",
                  speaker_id='ENT_FRED', addressee_ids=['ENT_LAURA'], actor_ids=['ENT_FRED'],
                  target_ids=['ENT_LAURA', 'EVT_RETURN_TO_FRED'],
                  via_channel_id=None, truth_value='true', fabula_time=12000, syuzhet_index=20),
        EventNode(id='EVT_UTT_LAURA_INTERIOR_CONFESSION', event_type='utterance',
                  description="Sitting beside Fred, Laura silently 'confesses' the entire affair to him in her interior monologue — the framing voice-over of the film.",
                  content="Fred, dear Fred — there's so much I want to tell you, you're the only one in the world with enough wisdom and gentleness to understand…",
                  speaker_id='ENT_LAURA', addressee_ids=['ENT_FRED'], actor_ids=['ENT_LAURA'],
                  target_ids=['ENT_ALEC', 'EVT_GRIT_IN_EYE', 'EVT_KARDOMAH_CINEMA', 'EVT_STEPHEN_RETURNS', 'EVT_AGREE_TO_END'],
                  via_channel_id='CHN_LAURA_INTERIOR_CONFESSION', truth_value='true',
                  fabula_time=12000, syuzhet_index=21),
    ],

    # ── CAUSAL TOPOLOGY ────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction ──
        CausalEdge(source_id="EVT_GRIT_IN_EYE", target_id="EVT_CHEMIST_MEETING",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1000, propagation_delay=1000),
        CausalEdge(source_id="EVT_CHEMIST_MEETING", target_id="EVT_BOTANICAL_WALK",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2000, propagation_delay=1000),
        CausalEdge(source_id="EVT_BOTANICAL_WALK", target_id="EVT_KARDOMAH_CINEMA",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3000, propagation_delay=1000),
        CausalEdge(source_id="EVT_KARDOMAH_CINEMA", target_id="EVT_FRIENDS_SEE_THEM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=4000, propagation_delay=1000),
        CausalEdge(source_id="EVT_FRIENDS_SEE_THEM", target_id="EVT_BORROW_FLAT_PLAN",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=5000, propagation_delay=1000),
        CausalEdge(source_id="EVT_BORROW_FLAT_PLAN", target_id="EVT_STEPHEN_RETURNS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000, propagation_delay=1000),
        CausalEdge(source_id="EVT_STEPHEN_RETURNS", target_id="EVT_WANDER_STREETS",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000, propagation_delay=500),
        CausalEdge(source_id="EVT_WANDER_STREETS", target_id="EVT_AGREE_TO_END",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7500, propagation_delay=1500),
        CausalEdge(source_id="EVT_AGREE_TO_END", target_id="EVT_FINAL_MEETING",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=9000, propagation_delay=1000),
        CausalEdge(source_id="EVT_FINAL_MEETING", target_id="EVT_ALEC_DEPARTS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=10000, propagation_delay=200),
        CausalEdge(source_id="EVT_ALEC_DEPARTS", target_id="EVT_NEAR_SUICIDE",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=10200, propagation_delay=300),
        CausalEdge(source_id="EVT_NEAR_SUICIDE", target_id="EVT_RETURN_TO_FRED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=10500, propagation_delay=1500),

        # ── mutation ──
        CausalEdge(source_id="EVT_GRIT_IN_EYE", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=4.0, fabula_time=1000,
                   trait_target="longing", trait_delta=0.1),
        CausalEdge(source_id="EVT_KARDOMAH_CINEMA", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=4000,
                   trait_target="love_for_alec", trait_delta=0.55),
        CausalEdge(source_id="EVT_KARDOMAH_CINEMA", target_id="ENT_ALEC",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=4000,
                   trait_target="love_for_laura", trait_delta=0.55),
        CausalEdge(source_id="EVT_KARDOMAH_CINEMA", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4000,
                   trait_target="longing", trait_delta=0.3),
        CausalEdge(source_id="EVT_FRIENDS_SEE_THEM", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=5000,
                   trait_target="guilt", trait_delta=0.25),
        CausalEdge(source_id="EVT_STEPHEN_RETURNS", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=7000,
                   trait_target="guilt", trait_delta=0.65),
        CausalEdge(source_id="EVT_STEPHEN_RETURNS", target_id="ENT_ALEC",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=10.0, fabula_time=7000,
                   trait_target="self_division", trait_delta=0.55),
        CausalEdge(source_id="EVT_AGREE_TO_END", target_id="ENT_ALEC",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000,
                   trait_target="marital_duty", trait_delta=0.1),
        CausalEdge(source_id="EVT_AGREE_TO_END", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000,
                   trait_target="moral_seriousness", trait_delta=0.1),
        CausalEdge(source_id="EVT_NEAR_SUICIDE", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=10500,
                   trait_target="longing", trait_delta=0.55),
        CausalEdge(source_id="EVT_RETURN_TO_FRED", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=12000,
                   trait_target="marital_loyalty", trait_delta=0.1),
        CausalEdge(source_id="EVT_RETURN_TO_FRED", target_id="ENT_LAURA",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=12000,
                   trait_target="longing", trait_delta=-0.4),

        # ── mutation_social ──
        CausalEdge(source_id="EVT_KARDOMAH_CINEMA", target_id="ENT_LAURA",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=4000,
                   trait_target="affinity", trait_delta=0.7, rel_counterpart_id="ENT_ALEC"),
        CausalEdge(source_id="EVT_KARDOMAH_CINEMA", target_id="ENT_ALEC",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=4000,
                   trait_target="affinity", trait_delta=0.7, rel_counterpart_id="ENT_LAURA"),
        CausalEdge(source_id="EVT_STEPHEN_RETURNS", target_id="ENT_ALEC",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000,
                   trait_target="affinity", trait_delta=-0.4, rel_counterpart_id="ENT_STEPHEN"),
        CausalEdge(source_id="EVT_AGREE_TO_END", target_id="ENT_LAURA",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000,
                   trait_target="affinity", trait_delta=0.15, rel_counterpart_id="ENT_ALEC"),
        CausalEdge(source_id="EVT_FINAL_MEETING", target_id="ENT_LAURA",
                   causality_type="mutation_social", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=10000,
                   trait_target="affinity", trait_delta=-0.55, rel_counterpart_id="ENT_DOLLY"),
        CausalEdge(source_id="EVT_RETURN_TO_FRED", target_id="ENT_LAURA",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=12000,
                   trait_target="affinity", trait_delta=0.45, rel_counterpart_id="ENT_FRED"),
        CausalEdge(source_id="EVT_RETURN_TO_FRED", target_id="ENT_FRED",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=12000,
                   trait_target="affinity", trait_delta=0.4, rel_counterpart_id="ENT_LAURA"),

        # ── affordance_gate ──
        CausalEdge(source_id="OBJ_GRIT_IN_EYE", target_id="EVT_GRIT_IN_EYE",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=1000),
        CausalEdge(source_id="OBJ_TIMETABLE", target_id="EVT_CHEMIST_MEETING",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2000),
        CausalEdge(source_id="OBJ_STEPHENS_KEY", target_id="EVT_STEPHEN_RETURNS",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=7000),
        CausalEdge(source_id="OBJ_DOLLY_VOICE", target_id="EVT_FINAL_MEETING",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=10000),
        CausalEdge(source_id="OBJ_EXPRESS_TRAIN", target_id="EVT_NEAR_SUICIDE",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=10500),

        # ── ambient_propagation ──
        CausalEdge(source_id="LOC_REFRESHMENT_ROOM", target_id="ENT_LAURA",
                   causality_type="ambient_propagation", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4000),
        CausalEdge(source_id="LOC_LAURAS_HOME", target_id="ENT_LAURA",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=12000),
        CausalEdge(source_id="LOC_STEPHENS_FLAT", target_id="ENT_ALEC",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=7000),
        CausalEdge(source_id="LOC_PLATFORM", target_id="ENT_LAURA",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=10500),

        # ── WORLD_ → Event ──
        CausalEdge(source_id="WORLD_INTERWAR_RESPECTABILITY", target_id="EVT_FRIENDS_SEE_THEM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=5000),
        CausalEdge(source_id="WORLD_INTERWAR_RESPECTABILITY", target_id="EVT_STEPHEN_RETURNS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000),
        CausalEdge(source_id="WORLD_INTERWAR_RESPECTABILITY", target_id="EVT_AGREE_TO_END",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000),
        CausalEdge(source_id="WORLD_DOMESTICITY", target_id="EVT_AGREE_TO_END",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000),
        CausalEdge(source_id="WORLD_DOMESTICITY", target_id="EVT_RETURN_TO_FRED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=12000),
        CausalEdge(source_id="WORLD_DOMESTICITY", target_id="EVT_NEAR_SUICIDE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=10500),
        CausalEdge(source_id="WORLD_TIMETABLE", target_id="EVT_GRIT_IN_EYE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_TIMETABLE", target_id="EVT_CHEMIST_MEETING",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2000),
        CausalEdge(source_id="WORLD_TIMETABLE", target_id="EVT_FINAL_MEETING",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=10000),
        CausalEdge(source_id="WORLD_TIMETABLE", target_id="EVT_ALEC_DEPARTS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=10200),

        # ── Staff flirtation arc as parallel/contrast (wires the orphan event) ──
        CausalEdge(source_id="WORLD_INTERWAR_RESPECTABILITY", target_id="EVT_STAFF_FLIRTATION_ARC",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=6000, propagation_delay=10),
        CausalEdge(source_id="EVT_STAFF_FLIRTATION_ARC", target_id="ENT_ALBERT_GODBY",
                   causality_type="mutation_social", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=6000,
                   trait_target="affinity", trait_delta=0.2, rel_counterpart_id="ENT_MYRTLE_PALMER"),
        CausalEdge(source_id="EVT_STAFF_FLIRTATION_ARC", target_id="ENT_BERYL",
                   causality_type="mutation_social", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=6000,
                   trait_target="affinity", trait_delta=0.2, rel_counterpart_id="ENT_STANLEY"),

        # ── orphan utterance wirings ──
        CausalEdge(source_id="EVT_FRIENDS_SEE_THEM", target_id="EVT_UTT_LAURA_LIES_TO_FRIENDS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_LAURA_LIES_TO_FRIENDS", target_id="EVT_BORROW_FLAT_PLAN",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=5000, propagation_delay=1000),
        CausalEdge(source_id="EVT_STEPHEN_RETURNS", target_id="EVT_UTT_STEPHEN_CHASTISES_ALEC",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_STEPHEN_CHASTISES_ALEC", target_id="EVT_WANDER_STREETS",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=5.0, fabula_time=7000, propagation_delay=500),
        CausalEdge(source_id="EVT_WANDER_STREETS", target_id="EVT_UTT_POLICEMAN_URGES_HOME",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=4.0, fabula_time=7500, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_POLICEMAN_URGES_HOME", target_id="EVT_AGREE_TO_END",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=7500, propagation_delay=1500),
        CausalEdge(source_id="EVT_FINAL_MEETING", target_id="EVT_UTT_DOLLY_CHATTERS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=4.0, fabula_time=10000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_DOLLY_CHATTERS", target_id="EVT_ALEC_DEPARTS",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=10000, propagation_delay=200),
        CausalEdge(source_id="EVT_FINAL_MEETING", target_id="EVT_UTT_ALEC_SHOULDER_SQUEEZE",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=4.0, fabula_time=10000, propagation_delay=200),
        CausalEdge(source_id="EVT_UTT_ALEC_SHOULDER_SQUEEZE", target_id="EVT_NEAR_SUICIDE",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=10200, propagation_delay=300),
        CausalEdge(source_id="EVT_RETURN_TO_FRED", target_id="EVT_UTT_FRED_THANKS_LAURA",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=12000, propagation_delay=0),
        CausalEdge(source_id="EVT_RETURN_TO_FRED", target_id="EVT_UTT_LAURA_INTERIOR_CONFESSION",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=6.0, fabula_time=12000, propagation_delay=0),



        # ─── auto-patched mutation_social edges (per-axis coverage) ───
        CausalEdge(source_id="EVT_BORROW_FLAT_PLAN", target_id="ENT_ALEC", rel_counterpart_id="ENT_STEPHEN", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.15, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_STEPHEN_CHASTISES_ALEC", target_id="ENT_ALEC", rel_counterpart_id="ENT_STEPHEN", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.3, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_STAFF_FLIRTATION_ARC", target_id="ENT_ALBERT_GODBY", rel_counterpart_id="ENT_MYRTLE_PALMER", causality_type="mutation_social", trait_target="affinity", trait_delta=0.7, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_STAFF_FLIRTATION_ARC", target_id="ENT_BERYL", rel_counterpart_id="ENT_STANLEY", causality_type="mutation_social", trait_target="affinity", trait_delta=0.7, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_BORROW_FLAT_PLAN", target_id="ENT_STEPHEN", rel_counterpart_id="ENT_ALEC", causality_type="mutation_social", trait_target="fear", trait_delta=0.15, mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_STEPHEN_RETURNS", target_id="ENT_STEPHEN", rel_counterpart_id="ENT_ALEC", causality_type="mutation_social", trait_target="fear", trait_delta=0.15, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_STEPHEN_CHASTISES_ALEC", target_id="ENT_STEPHEN", rel_counterpart_id="ENT_ALEC", causality_type="mutation_social", trait_target="fear", trait_delta=0.15, mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_BORROW_FLAT_PLAN", target_id="ENT_ALEC", rel_counterpart_id="ENT_STEPHEN", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.2, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_STEPHEN_RETURNS", target_id="ENT_ALEC", rel_counterpart_id="ENT_STEPHEN", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.15, mechanism="social", evidence_strength="strong", causal_force=5.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_STEPHEN_CHASTISES_ALEC", target_id="ENT_ALEC", rel_counterpart_id="ENT_STEPHEN", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.15, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_STAFF_FLIRTATION_ARC", target_id="ENT_ALBERT_GODBY", rel_counterpart_id="ENT_MYRTLE_PALMER", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.5, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_STEPHEN_CHASTISES_ALEC", target_id="ENT_STEPHEN", rel_counterpart_id="ENT_ALEC", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.3, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_BORROW_FLAT_PLAN", target_id="ENT_STEPHEN", rel_counterpart_id="ENT_ALEC", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.15, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_STAFF_FLIRTATION_ARC", target_id="ENT_MYRTLE_PALMER", rel_counterpart_id="ENT_ALBERT_GODBY", causality_type="mutation_social", trait_target="affinity", trait_delta=0.7, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_STAFF_FLIRTATION_ARC", target_id="ENT_STANLEY", rel_counterpart_id="ENT_BERYL", causality_type="mutation_social", trait_target="affinity", trait_delta=0.7, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_STEPHEN_CHASTISES_ALEC", target_id="ENT_ALEC", rel_counterpart_id="ENT_STEPHEN", causality_type="mutation_social", trait_target="fear", trait_delta=0.3, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_STEPHEN_CHASTISES_ALEC", target_id="ENT_STEPHEN", rel_counterpart_id="ENT_ALEC", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.3, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_STAFF_FLIRTATION_ARC", target_id="ENT_MYRTLE_PALMER", rel_counterpart_id="ENT_ALBERT_GODBY", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.5, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=6000, propagation_delay=0),
        # ── auto-backfilled per-axis mutation_social ──
        CausalEdge(source_id="EVT_FINAL_MEETING", target_id="ENT_DOLLY", rel_counterpart_id="ENT_LAURA",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.21,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=10000, propagation_delay=0),
        CausalEdge(source_id="EVT_CHEMIST_MEETING", target_id="ENT_LAURA", rel_counterpart_id="ENT_ALEC",  # auto-backfill
                   causality_type="mutation_social", trait_target="fear", trait_delta=0.09,
                   mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_GRIT_IN_EYE", target_id="ENT_ALEC", rel_counterpart_id="ENT_LAURA",  # auto-backfill
                   causality_type="mutation_social", trait_target="fear", trait_delta=0.05,
                   mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),

        # ── WORLD_ → WORLD_ (named-latent forces destabilising one another) ──
        CausalEdge(source_id="WORLD_INTERWAR_RESPECTABILITY", target_id="WORLD_DOMESTICITY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=12000,
                   description="Respectability enforces domesticity — the social cost of leaving Fred is what returns Laura to the wireless and the crossword."),
        CausalEdge(source_id="WORLD_TIMETABLE", target_id="WORLD_DOMESTICITY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=11500,
                   description="The railway timetable structures the Thursdays — and, by ending Alec's last train, returns Laura to the domestic schedule on time."),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_REFRESHMENT_ROOM", target_id="LOC_PLATFORM"),
        SpatialEdge(source_id="LOC_PLATFORM", target_id="LOC_REFRESHMENT_ROOM"),
        SpatialEdge(source_id="LOC_PLATFORM", target_id="LOC_BOTANICAL_GARDENS"),
        SpatialEdge(source_id="LOC_PLATFORM", target_id="LOC_KARDOMAH"),
        SpatialEdge(source_id="LOC_PLATFORM", target_id="LOC_STEPHENS_FLAT"),
        SpatialEdge(source_id="LOC_STEPHENS_FLAT", target_id="LOC_MILFORD_STREETS"),
        SpatialEdge(source_id="LOC_PLATFORM", target_id="LOC_LAURAS_HOME"),
        SpatialEdge(source_id="LOC_LAURAS_HOME", target_id="LOC_PLATFORM"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    channels={
        # Laura's framing voice-over: a standing interior-monologue "confession"
        # to Fred that runs across the whole flashback. Fred hears almost none
        # of it (intelligibility 0.0); the audience hears all of it.
        'CHN_LAURA_INTERIOR_CONFESSION': Channel(
            id='CHN_LAURA_INTERIOR_CONFESSION',
            name="Laura's interior confession to Fred",
            medium='interior_monologue',
            participant_ids=['ENT_LAURA', 'ENT_FRED'],
            directionality='simplex',
            intelligibility={'ENT_FRED': 0.0},
            established_at_fabula=12000, terminated_at_fabula=None,
            evidence_strength='strong',
        ),
        # The station timetable as a standing one-to-many broadcast capability
        # whose schedule organises every meeting and parting in the film.
        'CHN_STATION_TIMETABLE': Channel(
            id='CHN_STATION_TIMETABLE',
            name="Milford Junction public timetable",
            medium='public_schedule',
            participant_ids=['OBJ_TIMETABLE', 'ENT_LAURA', 'ENT_ALEC'],
            directionality='broadcast',
            intelligibility={},
            established_at_fabula=1000, terminated_at_fabula=None,
            evidence_strength='strong',
        ),
    },

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_INTERWAR_RESPECTABILITY": GlobalTrait(
            id="WORLD_INTERWAR_RESPECTABILITY",
            name="Interwar Middle-Class Respectability",
            description="The 1938 English code under which a married middle-class woman cannot be alone with a man not her husband without explanation, cannot tell the truth to friends, and cannot leave her family without ruin. Operates as common-cause parent over every concealment, every lie, and the final surrender.",
            category="social_structure",
            magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            proposition_id="PROP_AFFAIR_CONSUMMATED",
        ),
        "WORLD_DOMESTICITY": GlobalTrait(
            id="WORLD_DOMESTICITY",
            name="Suburban-Married Domesticity",
            description="The settled English household — children, supper, the wireless, Fred's crossword — that constitutes the moral baseline against which Laura's longing is measured and to which she finally returns.",
            category="social_structure",
            magnitude=TraitVector(value=0.9, inertia=0.95, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            proposition_id="PROP_LAURA_RETURNS_TO_FRED",
        ),
        "WORLD_TIMETABLE": GlobalTrait(
            id="WORLD_TIMETABLE",
            name="The Railway Timetable as Cosmology",
            description="The Southern Railway timetable that allows the Thursdays to happen and that finally, on the last Thursday, ends the affair on schedule. Operates as a quietly maximal latent: nothing in the story happens off-timetable.",
            category="cosmology",
            magnitude=TraitVector(value=0.85, inertia=0.95, evidence_strength="strong"),
            affected_domains=["social"],
            proposition_id="PROP_ALEC_TAKES_POST",
        ),
    },

    # ── SOCIAL TOPOLOGY ────────────────────────────────────────────────
    social_topology=[
        # Laura ↔ Alec — the affair (asymmetric: Laura falls harder, fears scandal more).
        RelationshipEdge(
            source_entity_id="ENT_LAURA", target_entity_id="ENT_ALEC",
            metrics={
                "affinity": RelationshipMetric(value=0.95, inertia=0.55, evidence_strength="strong", last_updated_fabula=9000),
                "fear":     RelationshipMetric(value=0.3, inertia=0.4, evidence_strength="moderate", last_updated_fabula=9000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ALEC", target_entity_id="ENT_LAURA",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.55, evidence_strength="strong", last_updated_fabula=9000),
                "fear":     RelationshipMetric(value=0.15, inertia=0.4, evidence_strength="weak", last_updated_fabula=9000),
            },
        ),
        # Laura ↔ Fred — settled marriage.
        RelationshipEdge(
            source_entity_id="ENT_LAURA", target_entity_id="ENT_FRED",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=12000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_FRED", target_entity_id="ENT_LAURA",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.75, evidence_strength="strong", last_updated_fabula=12000),
            },
        ),
        # Alec ↔ Stephen — colleague who disapproves.
        RelationshipEdge(
            source_entity_id="ENT_STEPHEN", target_entity_id="ENT_ALEC",
            metrics={
                "affinity":      RelationshipMetric(value=-0.45, inertia=0.55, evidence_strength="strong", last_updated_fabula=7000),
                "power_dynamic": RelationshipMetric(value=0.5, inertia=0.7, evidence_strength="moderate", last_updated_fabula=7000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_ALEC", target_entity_id="ENT_STEPHEN",
            metrics={
                "affinity": RelationshipMetric(value=0.3, inertia=0.55, evidence_strength="moderate", last_updated_fabula=7000),
                "fear":     RelationshipMetric(value=0.45, inertia=0.2, evidence_strength="moderate", last_updated_fabula=7000),
            },
        ),
        # Laura → Dolly — chilly tolerance.
        RelationshipEdge(
            source_entity_id="ENT_LAURA", target_entity_id="ENT_DOLLY",
            metrics={
                "affinity": RelationshipMetric(value=-0.55, inertia=0.55, evidence_strength="strong", last_updated_fabula=10000),
            },
        ),
        # Dolly → Laura — oblivious chatty fondness for the friend whose final tea-room encounter she shatters with her arrival. Strong positive affinity, no fear, no power asymmetry.
        RelationshipEdge(
            source_entity_id="ENT_DOLLY", target_entity_id="ENT_LAURA",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.55, evidence_strength="strong", last_updated_fabula=10000),
            },
        ),
        # Refreshment-room flirtation.
        RelationshipEdge(
            source_entity_id="ENT_ALBERT_GODBY", target_entity_id="ENT_MYRTLE_PALMER",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.55, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MYRTLE_PALMER", target_entity_id="ENT_ALBERT_GODBY",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.55, evidence_strength="strong", last_updated_fabula=6000),
                "power_dynamic": RelationshipMetric(value=0.5, inertia=0.7, evidence_strength="moderate", last_updated_fabula=6000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_BERYL", target_entity_id="ENT_STANLEY",
            metrics={
                "affinity": RelationshipMetric(value=0.75, inertia=0.5, evidence_strength="moderate", last_updated_fabula=6000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_STANLEY", target_entity_id="ENT_BERYL",
            metrics={
                "affinity": RelationshipMetric(value=0.6, inertia=0.5, evidence_strength="moderate", last_updated_fabula=6000),
            },
        ),
    ],
    propositions=[
        # The mutual romantic attachment — central trait the affair tests.
        Proposition(proposition_id="PROP_ALEC_LOVES_LAURA", kind="trait_holds",
                    referent_ids=["ENT_ALEC", "ENT_LAURA"],
                    description="Alec is genuinely in love with Laura, not merely infatuated.",
                    audience_default_prior=0.7, stakes=0.9,
                    truth_at_fabula={3000: True}),
        Proposition(proposition_id="PROP_LAURA_LOVES_ALEC", kind="trait_holds",
                    referent_ids=["ENT_LAURA", "ENT_ALEC"],
                    description="Laura is genuinely in love with Alec, not merely flattered.",
                    audience_default_prior=0.7, stakes=0.9,
                    truth_at_fabula={3000: True}),
        # The Bowlby-attachment fork — Laura's marriage to Fred as the moral counterweight.
        Proposition(proposition_id="PROP_LAURA_LEAVES_FRED", kind="event_occurs",
                    referent_ids=["ENT_LAURA", "ENT_FRED"],
                    description="Laura leaves her husband Fred for Alec.",
                    audience_default_prior=0.2, stakes=0.95,
                    truth_at_fabula={12000: False}),
        # The mirroring fork on Alec's side.
        Proposition(proposition_id="PROP_ALEC_LEAVES_FAMILY", kind="event_occurs",
                    referent_ids=["ENT_ALEC"],
                    description="Alec leaves his wife and children for Laura.",
                    audience_default_prior=0.2, stakes=0.9,
                    truth_at_fabula={12000: False}),
        # Public exposure — the social-shame axis (humiliation kind).
        Proposition(proposition_id="PROP_AFFAIR_EXPOSED", kind="event_occurs",
                    referent_ids=["ENT_LAURA", "ENT_ALEC"],
                    description="The affair becomes publicly known and the social cost lands.",
                    audience_default_prior=0.3, stakes=0.85,
                    truth_at_fabula={12000: False}),
        # The Stephen Lynn flat scene — would have been physical consummation.
        Proposition(proposition_id="PROP_AFFAIR_CONSUMMATED", kind="event_occurs",
                    referent_ids=["ENT_LAURA", "ENT_ALEC"],
                    description="The affair is physically consummated at Stephen's flat.",
                    audience_default_prior=0.4, stakes=0.8,
                    truth_at_fabula={9000: False}),
        # Alec's escape — taking the medical post in Johannesburg ends the affair.
        Proposition(proposition_id="PROP_ALEC_TAKES_POST", kind="event_occurs",
                    referent_ids=["ENT_ALEC"],
                    description="Alec accepts the post in Johannesburg and emigrates.",
                    audience_default_prior=0.5, stakes=0.85,
                    truth_at_fabula={11500: True}),
        # WORLD_ trait Pearl-Rung-2 reification (Domesticity).
        Proposition(proposition_id="PROP_LAURA_RETURNS_TO_FRED", kind="trait_holds",
                    referent_ids=["WORLD_DOMESTICITY", "ENT_LAURA", "ENT_FRED"],
                    description="Laura returns to and stays within the suburban-married domesticity she briefly threatened to leave.",
                    audience_default_prior=0.85, stakes=0.85,
                    truth_at_fabula={12000: True}),
    ],
)
