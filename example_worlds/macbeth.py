# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Macbeth — high-fidelity WorldStateV1 test fixture.

Authored against the current ingestion prompts (ontology_locations,
ontology_objects, ontology_entities, ontology_world_traits,
physics_extraction, social_extraction, consequences_extraction,
world_trait_timeline). Demonstrates:

* All five CausalEdge modalities (chain_reaction, mutation,
  mutation_social, affordance_gate, ambient_propagation).
* Per-axis ``RelationshipMetric`` (only observed axes populated, with
  per-axis inertia bands: fear ~0.1–0.3, affinity ~0.3–0.6,
  power_dynamic ~0.5–0.8).
* Explicit ``evidence_strength`` everywhere it is supported
  (TraitVector, AmbientVector, Belief, CausalEdge,
  RelationshipMetric, GlobalTrait magnitude).
* ``state_timeline`` snapshots whose inertia *bumps up* (not down) when
  a trait is shocked away from baseline.
* WORLD_ named-latent traits wired as common-cause parents to the
  events they jointly drive.
* Initial entity locations match each character's first appearance
  (not their final stronghold).
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
        target_word_min=560,
        target_word_max=2100,
        prose_density='sparse',
        voice='synoptic narration; no dialogue; condensed scene description; third-person POV; past tense',
        style_exemplar='Act I\n\nMacbeth and Banquo encounter the witches for the first time.\nAmid thunder and lightning, three witches decide that their next meeting will be with Macbeth, the Thane (Lord) of Glamis. In the following scene, soldiers report to King Duncan of Scotland that his generals Banquo and Macbeth have just defeated a rebellion led by the traitorous Thane of Cawdor, allied with forces from Norway and Ireland. Duncan praises his kinsmen for their bravery and fighting prowess, announcing that the title of Thane of Cawdor shall be transferred to Macbeth.',
        source_word_count=1400,
    ),
    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_BATTLEFIELD": Location(
            id="LOC_BATTLEFIELD",
            name="Battlefield near Forres",
            description="Blood-soaked field where Macbeth and Banquo crush the Cawdor rebellion.",
            ambient_state={
                "danger": AmbientVector(value=0.9, volatility=0.4, evidence_strength="strong"),
                "tension": AmbientVector(value=0.8, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_HEATH": Location(
            id="LOC_HEATH",
            name="The Heath",
            description="Desolate fog-shrouded moor where Macbeth and Banquo first meet the witches.",
            ambient_state={
                "supernatural": AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
                "concealment": AmbientVector(value=0.7, volatility=0.3, evidence_strength="moderate"),
            },
        ),
        "LOC_FORRES_COURT": Location(
            id="LOC_FORRES_COURT",
            name="Duncan's Court at Forres",
            description="Royal hall where Duncan praises his generals and names Malcolm his heir.",
            ambient_state={
                "formality": AmbientVector(value=0.8, volatility=0.2, evidence_strength="moderate"),
                "tension": AmbientVector(value=0.4, volatility=0.4, evidence_strength="weak"),
            },
        ),
        "LOC_INVERNESS_CASTLE": Location(
            id="LOC_INVERNESS_CASTLE",
            name="Inverness Castle",
            description="Macbeth's ancestral keep; Duncan is murdered here in his sleep.",
            ambient_state={
                "tension": AmbientVector(value=0.7, volatility=0.5, evidence_strength="strong"),
                "concealment": AmbientVector(value=0.6, volatility=0.4, evidence_strength="moderate"),
                # Frijda action-readiness: castle has gates onto Forres / Dunsinane → flight feasible.
                "connected_to": AmbientVector(value=1.0, volatility=0.0, evidence_strength="strong"),
            },
        ),
        "LOC_DUNSINANE_CASTLE": Location(
            id="LOC_DUNSINANE_CASTLE",
            name="Dunsinane Castle",
            description="Royal seat of the crowned Macbeth; site of his banquet, raving, and final stand.",
            ambient_state={
                "danger": AmbientVector(value=0.8, volatility=0.3, evidence_strength="strong"),
                "tension": AmbientVector(value=0.9, volatility=0.2, evidence_strength="strong"),
                # Besieged in Act V — exits are technically present (Birnam road)
                # but politically/militarily sealed; Macbeth's *dread* fires from
                # the prophecy, not from physical entrapment.
                "connected_to": AmbientVector(value=1.0, volatility=0.4, evidence_strength="moderate"),
            },
        ),
        "LOC_WITCHES_CAVERN": Location(
            id="LOC_WITCHES_CAVERN",
            name="Witches' Cavern",
            description="Subterranean lair where Macbeth returns for the apparitions and second prophecies.",
            ambient_state={
                "supernatural": AmbientVector(value=0.95, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_MACDUFF_CASTLE": Location(
            id="LOC_MACDUFF_CASTLE",
            name="Macduff's Castle at Fife",
            description="Macduff's family seat; site of the slaughter of his wife and children.",
            ambient_state={
                "safety": AmbientVector(value=0.6, volatility=0.7, evidence_strength="moderate"),
                # Macduff has fled to England; the women and children are
                # without guard — Lady Macduff's flight is undermined by
                # her belief that her husband has *already* abandoned her.
                "connected_to": AmbientVector(value=0.4, volatility=0.5, evidence_strength="weak"),
            },
        ),
        "LOC_ENGLAND": Location(
            id="LOC_ENGLAND",
            name="English Court",
            description="Refuge of Malcolm and Macduff at King Edward's court; staging ground for the counter-invasion.",
            ambient_state={
                "safety": AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_BIRNAM_WOOD": Location(
            id="LOC_BIRNAM_WOOD",
            name="Birnam Wood",
            description="Forest west of Dunsinane; Malcolm's army camouflages itself with cut boughs from here.",
            ambient_state={
                "concealment": AmbientVector(value=0.8, volatility=0.3, evidence_strength="strong"),
            },
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────
    objects={
        "OBJ_BLOODY_DAGGERS": NarrativeObject(
            id="OBJ_BLOODY_DAGGERS", name="Bloody Daggers",
            location_id=None, owner_id="ENT_MACBETH",
            properties={"state": "blood-smeared", "stained_with": "duncan_blood"},
            affordances=[
                Affordance(action="kill", target_type="Entity"),
                Affordance(action="frame", target_type="Entity"),
            ],
        ),
        "OBJ_CROWN": NarrativeObject(
            id="OBJ_CROWN", name="Crown of Scotland",
            location_id="LOC_FORRES_COURT", owner_id="ENT_DUNCAN",
            properties={"state": "contested"},
            affordances=[Affordance(action="legitimize", target_type="Entity")],
        ),
        "OBJ_LETTER": NarrativeObject(
            id="OBJ_LETTER", name="Macbeth's Letter to Lady Macbeth",
            location_id="LOC_INVERNESS_CASTLE", owner_id="ENT_LADY_MACBETH",
            properties={"content": "witches_prophecy", "state": "delivered"},
            affordances=[Affordance(action="inform", target_type="Entity")],
        ),
        "OBJ_CAULDRON": NarrativeObject(
            id="OBJ_CAULDRON", name="Witches' Cauldron",
            location_id="LOC_WITCHES_CAVERN", owner_id=None,
            properties={"state": "bubbling"},
            affordances=[Affordance(action="prophesy", target_type="Entity")],
        ),
        "OBJ_APPARITIONS": NarrativeObject(
            id="OBJ_APPARITIONS", name="The Three Apparitions",
            location_id="LOC_WITCHES_CAVERN", owner_id=None,
            properties={"content": "armed_head_bloody_child_crowned_child"},
            affordances=[Affordance(action="prophesy", target_type="Entity"), Affordance(action="deceive", target_type="Entity")],
        ),
        "OBJ_BIRNAM_BOUGHS": NarrativeObject(
            id="OBJ_BIRNAM_BOUGHS", name="Birnam Wood Boughs",
            location_id="LOC_BIRNAM_WOOD", owner_id=None,
            properties={"state": "cut", "purpose": "camouflage"},
            affordances=[Affordance(action="conceal", target_type="Entity")],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    # Initial location = where the entity FIRST appears (per ontology_entities rules).
    entities={
        "ENT_MACBETH": Entity(
            id="ENT_MACBETH", name="Macbeth (Thane of Glamis)",
            location_id="LOC_BATTLEFIELD", status="healthy",
            traits={
                "ambition": TraitVector(value=0.7, inertia=0.55, evidence_strength="strong"),
                "courage": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "loyalty": TraitVector(value=0.7, inertia=0.5, evidence_strength="moderate"),
                "guilt": TraitVector(value=0.1, inertia=0.25, evidence_strength="weak"),
                "paranoia": TraitVector(value=0.2, inertia=0.3, evidence_strength="weak"),
                "ruthlessness": TraitVector(value=0.4, inertia=0.45, evidence_strength="moderate"),
                "despair": TraitVector(value=0.1, inertia=0.25, evidence_strength="weak"),
            },
            beliefs=[
                Belief(target_id="ENT_DUNCAN", perceived_state="Duncan is my kinsman and rightful king", proposition_id="PROP_THANE_LOYALTY",
                       confidence=0.9, inertia=0.55, evidence_strength="strong"),
                Belief(target_id="ENT_BANQUO", perceived_state="Banquo is my trusted comrade-in-arms", proposition_id="PROP_BANQUO_LINE_KINGS",
                       confidence=0.85, inertia=0.5, evidence_strength="strong"),
                Belief(target_id="ENT_WITCHES", perceived_state="The witches' prophecies may yet prove true", proposition_id="PROP_MACBETH_BECOMES_KING",
                       confidence=0.4, inertia=0.35, evidence_strength="moderate"),
            ],
            concerns=[
                # Lazarus appraisal: the prophecy plants a *desire-concern* whose
                # realisation requires regicide; OCC counter-concern with Banquo's line.
                Concern(concern_id="CCN_MACBETH_BECOMES_KING", proposition_id="PROP_MACBETH_BECOMES_KING",
                        polarity="desire", kind="ambition", salience=0.95,
                        activation_fabula_window=[2000, 19000],
                        counter_concern_ids=["CCN_MACBETH_FEARS_BANQUO_LINE", "CCN_MACBETH_FEARS_MACDUFF"]),
                Concern(concern_id="CCN_MACBETH_FEARS_BANQUO_LINE", proposition_id="PROP_BANQUO_LINE_KINGS",
                        polarity="fear", kind="heir_anxiety", salience=0.7,
                        activation_fabula_window=[2000, 19000],
                        counter_concern_ids=["CCN_MACBETH_BECOMES_KING"]),
                Concern(concern_id="CCN_MACBETH_FEARS_MACDUFF", proposition_id="PROP_MACDUFF_THREAT",
                        polarity="fear", kind="mortal_threat", salience=0.85,
                        activation_fabula_window=[13000, 19000],
                        counter_concern_ids=["CCN_MACBETH_BECOMES_KING", "CCN_MACBETH_DESIRES_BIRNAM_STILL"]),
                Concern(concern_id="CCN_MACBETH_DESIRES_INVINCIBILITY", proposition_id="PROP_MACBETH_INVINCIBLE",
                        polarity="desire", kind="power", salience=0.8,
                        activation_fabula_window=[13000, 19000]),
                Concern(concern_id="CCN_MACBETH_DESIRES_BIRNAM_STILL", proposition_id="PROP_BIRNAM_NEVER_MOVES",
                        polarity="desire", kind="fate", salience=0.7,
                        activation_fabula_window=[13000, 19000],
                        counter_concern_ids=["CCN_MACBETH_FEARS_MACDUFF"],
                        state_timeline=[
                            ConcernSnapshot(fabula_time=18500, triggered_by="EVT_BIRNAM_WOOD_MOVES",
                                            salience=0.95, polarity="fear", kind="doom",
                                            counter_concern_ids=["CCN_MACBETH_BECOMES_KING"]),
                        ]),
                # Sternberg passionate-bond concern — anchors grief at ft ≥ 17000.
                Concern(concern_id="CCN_MACBETH_LOVES_LADY", proposition_id="PROP_LADY_MACBETH_RESOLVE",
                        polarity="desire", kind="love", salience=0.7),
            ],
            constants=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=2000, triggered_by="EVT_WITCHES_PROPHECY_1",
                    traits={
                        "ambition": TraitVector(value=0.85, inertia=0.65, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=5000, triggered_by="EVT_LADY_MACBETH_PERSUADES",
                    traits={
                        "ruthlessness": TraitVector(value=0.80, inertia=0.45, evidence_strength="moderate"),
                    }),EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_DUNCAN_MURDER",
                    traits={
                        "guilt": TraitVector(value=0.7, inertia=0.4, evidence_strength="strong"),
                        "paranoia": TraitVector(value=0.55, inertia=0.4, evidence_strength="strong"),
                        "ruthlessness": TraitVector(value=0.65, inertia=0.55, evidence_strength="strong"),
                    },
                    location_id="LOC_INVERNESS_CASTLE"),
                EntityStateSnapshot(fabula_time=10000, triggered_by="EVT_MACBETH_CROWNED",
                    traits={
                        "paranoia": TraitVector(value=0.7, inertia=0.45, evidence_strength="strong"),
                    },
                    location_id="LOC_DUNSINANE_CASTLE"),
                EntityStateSnapshot(fabula_time=11000, triggered_by="EVT_BANQUO_MURDERED",
                    traits={
                        "guilt": TraitVector(value=0.85, inertia=0.5, evidence_strength="strong"),
                        "paranoia": TraitVector(value=0.85, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_WITCHES_PROPHECY_2",
                    traits={
                        "courage": TraitVector(value=0.95, inertia=0.75, evidence_strength="strong"),
                        "paranoia": TraitVector(value=0.55, inertia=0.45, evidence_strength="moderate"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_MACDUFF", perceived_state="Macduff threatens me but no man of woman born can harm me",
                               proposition_id="PROP_MACBETH_INVINCIBLE",
                               confidence=0.85, inertia=0.6, established_at_fabula=13000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=17000, triggered_by="EVT_LADY_MACBETH_DEATH",
                    traits={
                        "despair": TraitVector(value=0.9, inertia=0.5, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=18500, triggered_by="EVT_BIRNAM_WOOD_MOVES",
                    traits={
                        "courage": TraitVector(value=0.7, inertia=0.7, evidence_strength="moderate"),
                        "despair": TraitVector(value=0.95, inertia=0.55, evidence_strength="strong"),
                    },
                    beliefs_invalidated=["ENT_MACDUFF"]),
                EntityStateSnapshot(fabula_time=19000, triggered_by="EVT_MACBETH_KILLED",
                    status="dead"),
                
            ],
        ),
        "ENT_LADY_MACBETH": Entity(
            id="ENT_LADY_MACBETH", name="Lady Macbeth",
            location_id="LOC_INVERNESS_CASTLE", status="healthy",
            traits={
                "ambition": TraitVector(value=0.9, inertia=0.6, evidence_strength="strong"),
                "ruthlessness": TraitVector(value=0.85, inertia=0.55, evidence_strength="strong"),
                "resolve": TraitVector(value=0.9, inertia=0.55, evidence_strength="strong"),
                "guilt": TraitVector(value=0.05, inertia=0.2, evidence_strength="weak"),
            },
            beliefs=[
                Belief(target_id="ENT_MACBETH", perceived_state="Macbeth is too full of the milk of human kindness to seize the crown alone",
                       proposition_id="PROP_LADY_MACBETH_RESOLVE",
                       confidence=0.85, inertia=0.55, evidence_strength="strong"),
                Belief(target_id="OBJ_CROWN", perceived_state="The crown is within our grasp tonight",
                       proposition_id="PROP_MACBETH_BECOMES_KING",
                       confidence=0.9, inertia=0.5, evidence_strength="strong"),
            ],
            concerns=[
                Concern(concern_id="CCN_LADY_DESIRES_CROWN", proposition_id="PROP_MACBETH_BECOMES_KING",
                        polarity="desire", kind="ambition", salience=0.95,
                        activation_fabula_window=[4000, 17000],
                        counter_concern_ids=["CCN_LADY_FEARS_DISCOVERY"]),
                Concern(concern_id="CCN_LADY_FEARS_DISCOVERY", proposition_id="PROP_REGICIDE_DISCOVERED",
                        polarity="fear", kind="exposure", salience=0.85,
                        activation_fabula_window=[5000, 17000],
                        counter_concern_ids=["CCN_LADY_DESIRES_CROWN"]),
                Concern(concern_id="CCN_LADY_DESIRES_MACBETH", proposition_id="PROP_MARRIAGE_BOND",
                        polarity="desire", kind="love", salience=0.65,
                        activation_fabula_window=[1, 17000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_DUNCAN_MURDER",
                    traits={
                        "guilt": TraitVector(value=0.3, inertia=0.3, evidence_strength="moderate"),
                    }),
                EntityStateSnapshot(fabula_time=16000, triggered_by="EVT_LADY_MACBETH_SLEEPWALKING",
                    traits={
                        "guilt": TraitVector(value=0.95, inertia=0.55, evidence_strength="strong"),
                        "resolve": TraitVector(value=0.2, inertia=0.6, evidence_strength="strong"),
                    },
                    status="ill"),
                EntityStateSnapshot(fabula_time=17000, triggered_by="EVT_LADY_MACBETH_DEATH",
                    status="dead"),
            ],
        ),
        "ENT_DUNCAN": Entity(
            id="ENT_DUNCAN", name="King Duncan of Scotland",
            location_id="LOC_FORRES_COURT", status="healthy",
            traits={
                "benevolence": TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
                "trust": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "leadership": TraitVector(value=0.7, inertia=0.6, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_MACBETH", perceived_state="Macbeth is my loyal and valiant kinsman", proposition_id="PROP_THANE_LOYALTY",
                       confidence=0.95, inertia=0.65, evidence_strength="strong"),
                Belief(target_id="ENT_MALCOLM", perceived_state="Malcolm is fit to be my heir", proposition_id="PROP_MALCOLM_THRONE",
                       confidence=0.85, inertia=0.7, evidence_strength="strong"),
            ],
            concerns=[
                Concern(concern_id="CCN_DUNCAN_DESIRES_MALCOLM_HEIR", proposition_id="PROP_MALCOLM_THRONE",
                        polarity="desire", kind="succession", salience=0.85,
                        activation_fabula_window=[1, 6000]),
                Concern(concern_id="CCN_DUNCAN_DESIRES_LOYALTY", proposition_id="PROP_THANE_LOYALTY",
                        polarity="desire", kind="loyalty", salience=0.75,
                        activation_fabula_window=[1, 6000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=5500, triggered_by="EVT_DUNCAN_ARRIVES_INVERNESS",
                    location_id="LOC_INVERNESS_CASTLE"),
                EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_DUNCAN_MURDER",
                    status="dead"),
            ],
        ),
        "ENT_BANQUO": Entity(
            id="ENT_BANQUO", name="Banquo",
            location_id="LOC_BATTLEFIELD", status="healthy",
            traits={
                "loyalty": TraitVector(value=0.8, inertia=0.65, evidence_strength="strong"),
                "courage": TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                "suspicion": TraitVector(value=0.25, inertia=0.3, evidence_strength="weak"),
                "caution": TraitVector(value=0.6, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_WITCHES", perceived_state="The prophecies may hold truth — my line shall be kings",
                       proposition_id="PROP_BANQUO_LINE_KINGS",
                       confidence=0.5, inertia=0.4, evidence_strength="moderate"),
            ],
            concerns=[
                Concern(concern_id="CCN_BANQUO_DESIRES_FLEANCE_SAFE", proposition_id="PROP_FLEANCE_ALIVE",
                        polarity="desire", kind="love", salience=0.95,
                        activation_fabula_window=[1, 11000]),
                Concern(concern_id="CCN_BANQUO_FEARS_MACBETH_FOUL_PLAY", proposition_id="PROP_MACBETH_FOUL_PLAY",
                        polarity="fear", kind="betrayal", salience=0.7,
                        activation_fabula_window=[10000, 11000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=10500, triggered_by="EVT_MACBETH_CROWNED",
                    traits={
                        "suspicion": TraitVector(value=0.75, inertia=0.45, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_MACBETH", perceived_state="Macbeth played most foully for the crown",
                               proposition_id="PROP_MACBETH_FOUL_PLAY",
                               confidence=0.7, inertia=0.55, established_at_fabula=10500, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=11000, triggered_by="EVT_BANQUO_MURDERED",
                    status="dead"),
            ],
        ),
        "ENT_FLEANCE": Entity(
            id="ENT_FLEANCE", name="Fleance (son of Banquo)",
            location_id="LOC_DUNSINANE_CASTLE", status="healthy",
            traits={
                "innocence": TraitVector(value=0.9, inertia=0.6, evidence_strength="moderate"),
                "fear": TraitVector(value=0.4, inertia=0.25, evidence_strength="weak"),
            },
            beliefs=[],
            concerns=[
                Concern(concern_id="CCN_FLEANCE_FEARS_DEATH", proposition_id="PROP_FLEANCE_ALIVE",
                        polarity="desire", kind="survival", salience=0.95,
                        activation_fabula_window=[10500, 19000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=11000, triggered_by="EVT_BANQUO_MURDERED",
                    traits={
                        "fear": TraitVector(value=0.85, inertia=0.35, evidence_strength="strong"),
                    },
                    location_id="LOC_ENGLAND"),
            ],
        ),
        "ENT_MACDUFF": Entity(
            id="ENT_MACDUFF", name="Macduff (Thane of Fife)",
            location_id="LOC_INVERNESS_CASTLE", status="healthy",
            traits={
                "loyalty": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "courage": TraitVector(value=0.85, inertia=0.65, evidence_strength="strong"),
                "vengefulness": TraitVector(value=0.2, inertia=0.3, evidence_strength="weak"),
                "grief": TraitVector(value=0.1, inertia=0.25, evidence_strength="weak"),
                "suspicion": TraitVector(value=0.4, inertia=0.35, evidence_strength="moderate"),
            },
            beliefs=[],
            constants=["caesarean_birth"],
            concerns=[
                Concern(concern_id="CCN_MACDUFF_DESIRES_FAMILY_SAFE", proposition_id="PROP_MACDUFF_FAMILY_SAFE",
                        polarity="desire", kind="love", salience=0.95,
                        activation_fabula_window=[1, 14000]),
                Concern(concern_id="CCN_MACDUFF_FEARS_TYRANT", proposition_id="PROP_MACBETH_TYRANT",
                        polarity="fear", kind="loyalty", salience=0.75,
                        activation_fabula_window=[7000, 19000]),
                # Vengeance axis — Macduff's drive after the Fife slaughter (kind=vengeance,
                # not Averill betrayal-on-self since Macduff is the avenger not the betrayed).
                Concern(concern_id="CCN_MACDUFF_DESIRES_VENGEANCE", proposition_id="PROP_MACDUFF_AVENGED",
                        polarity="desire", kind="vengeance", salience=0.95,
                        activation_fabula_window=[14500, 19000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=7500, triggered_by="EVT_DUNCAN_DISCOVERED_MURDERED",
                    traits={
                        "suspicion": TraitVector(value=0.7, inertia=0.45, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_MACBETH", perceived_state="Macbeth's grief at Duncan's death rings false",
                               proposition_id="PROP_MACBETH_FOUL_PLAY",
                               confidence=0.6, inertia=0.5, established_at_fabula=7500, evidence_strength="moderate"),
                    ]),
                EntityStateSnapshot(fabula_time=12500, triggered_by="EVT_MACDUFF_FLEES_TO_ENGLAND",
                    location_id="LOC_ENGLAND"),
                EntityStateSnapshot(fabula_time=14500, triggered_by="EVT_MACDUFF_LEARNS_OF_MASSACRE",
                    traits={
                        "grief": TraitVector(value=0.95, inertia=0.55, evidence_strength="strong"),
                        "vengefulness": TraitVector(value=0.95, inertia=0.6, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_LADY_MACDUFF": Entity(
            id="ENT_LADY_MACDUFF", name="Lady Macduff",
            location_id="LOC_MACDUFF_CASTLE", status="healthy",
            traits={
                "courage": TraitVector(value=0.5, inertia=0.4, evidence_strength="moderate"),
                "anger": TraitVector(value=0.5, inertia=0.3, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_MACDUFF", perceived_state="My husband has abandoned us by fleeing to England", proposition_id="PROP_MACDUFF_FAMILY_SAFE",
                       confidence=0.7, inertia=0.5, evidence_strength="moderate"),
            ],
            concerns=[
                Concern(concern_id="CCN_LADY_MACDUFF_DESIRES_FAMILY_SAFE", proposition_id="PROP_MACDUFF_FAMILY_SAFE",
                        polarity="desire", kind="love", salience=0.95,
                        activation_fabula_window=[1, 14000]),
                Concern(concern_id="CCN_LADY_MACDUFF_FEARS_ABANDONMENT", proposition_id="PROP_MACDUFF_RETURNS",
                        polarity="fear", kind="abandonment", salience=0.8,
                        activation_fabula_window=[12500, 14000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=14000, triggered_by="EVT_MACDUFF_FAMILY_SLAUGHTERED",
                    status="dead"),
            ],
        ),
        "ENT_MALCOLM": Entity(
            id="ENT_MALCOLM", name="Prince Malcolm",
            location_id="LOC_FORRES_COURT", status="healthy",
            traits={
                "caution": TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
                "leadership": TraitVector(value=0.55, inertia=0.45, evidence_strength="moderate"),
                "loyalty": TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                "courage": TraitVector(value=0.6, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_MACBETH", perceived_state="Macbeth or his agents may seek to kill us next", proposition_id="PROP_MALCOLM_KILLED",
                       confidence=0.8, inertia=0.55, established_at_fabula=7500, evidence_strength="strong"),
            ],
            concerns=[
                Concern(concern_id="CCN_MALCOLM_FEARS_ASSASSINATION", proposition_id="PROP_MALCOLM_KILLED",
                        polarity="fear", kind="mortal_threat", salience=0.85,
                        activation_fabula_window=[7000, 15000]),
                Concern(concern_id="CCN_MALCOLM_DESIRES_THRONE", proposition_id="PROP_MALCOLM_THRONE",
                        polarity="desire", kind="duty", salience=0.7,
                        activation_fabula_window=[7000, 20000]),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=9000, triggered_by="EVT_SONS_FLEE",
                    location_id="LOC_ENGLAND",
                    traits={
                        "caution": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=15000, triggered_by="EVT_MALCOLM_MACDUFF_ALLIANCE",
                    traits={
                        "leadership": TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_WITCHES": Entity(
            id="ENT_WITCHES", name="The Three Witches",
            location_id="LOC_HEATH", status="healthy",
            traits={
                "malice": TraitVector(value=0.8, inertia=0.85, evidence_strength="strong"),
                "deception": TraitVector(value=0.85, inertia=0.85, evidence_strength="strong"),
            },
            beliefs=[],
            constants=["supernatural"],
        ),
        "ENT_LENNOX": Entity(
            id="ENT_LENNOX", name="Lennox",
            location_id="LOC_INVERNESS_CASTLE", status="healthy",
            traits={
                "loyalty": TraitVector(value=0.6, inertia=0.5, evidence_strength="moderate"),
                "suspicion": TraitVector(value=0.4, inertia=0.35, evidence_strength="moderate"),
                "caution": TraitVector(value=0.7, inertia=0.55, evidence_strength="moderate"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=12000, triggered_by="EVT_LENNOX_WHISPER",
                    traits={
                        "suspicion": TraitVector(value=0.8, inertia=0.5, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_MACBETH", perceived_state="Macbeth is a murdering tyrant", proposition_id="PROP_MACBETH_TYRANT",
                               confidence=0.8, inertia=0.55, established_at_fabula=12000, evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_DOCTOR": Entity(
            id="ENT_DOCTOR", name="Scottish Doctor",
            location_id="LOC_DUNSINANE_CASTLE", status="healthy",
            traits={
                "discretion": TraitVector(value=0.75, inertia=0.6, evidence_strength="moderate"),
                "fear": TraitVector(value=0.6, inertia=0.4, evidence_strength="moderate"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=16000, triggered_by="EVT_UTT_LADY_MACBETH_SLEEPWALK_CONFESSION",
                    beliefs_added=[
                        Belief(target_id="ENT_LADY_MACBETH", perceived_state="Her trouble is moral, not medical — she has confessed to murder", proposition_id="PROP_REGICIDE_DISCOVERED",
                               confidence=0.85, inertia=0.7, established_at_fabula=16000,
                               acquired_via_event_id="EVT_UTT_LADY_MACBETH_SLEEPWALK_CONFESSION",
                               evidence_strength="strong"),
                    ]),
            ],
        ),
        "ENT_GENTLEWOMAN": Entity(
            id="ENT_GENTLEWOMAN", name="Lady Macbeth's Gentlewoman",
            location_id="LOC_DUNSINANE_CASTLE", status="healthy",
            traits={
                "loyalty": TraitVector(value=0.7, inertia=0.55, evidence_strength="moderate"),
                "discretion": TraitVector(value=0.8, inertia=0.65, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_LADY_MACBETH", perceived_state="My mistress walks and talks in her sleep most nights", proposition_id="PROP_REGICIDE_DISCOVERED",
                       confidence=0.95, inertia=0.8, established_at_fabula=15500, evidence_strength="strong"),
            ],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_REBELLION_DEFEATED", fabula_time=1000, syuzhet_index=1,
                  event_type="outcome", actor_ids=["ENT_MACBETH", "ENT_BANQUO"], target_ids=[],
                  description="Macbeth and Banquo defeat the Cawdor rebellion on the battlefield.", at_location_id="LOC_BATTLEFIELD"),
        EventNode(id="EVT_WITCHES_PROPHECY_1", fabula_time=2000, syuzhet_index=2,
                  event_type="revelation", actor_ids=["ENT_WITCHES"],
                  target_ids=["ENT_MACBETH", "ENT_BANQUO"],
                  description="On the heath the witches hail Macbeth Thane of Cawdor and king hereafter, and tell Banquo his line shall be kings.", at_location_id="LOC_HEATH"),
        EventNode(id="EVT_CAWDOR_TITLE", fabula_time=3000, syuzhet_index=4,
                  event_type="outcome", actor_ids=["ENT_DUNCAN"], target_ids=["ENT_MACBETH"],
                  description="The Thane of Ross informs Macbeth that Duncan has granted him the title Thane of Cawdor, fulfilling the first prophecy.", at_location_id="LOC_FORRES_COURT"),
        EventNode(id="EVT_MALCOLM_NAMED_HEIR", fabula_time=3500, syuzhet_index=6,
                  event_type="choice", actor_ids=["ENT_DUNCAN"], target_ids=["ENT_MALCOLM"],
                  description="Duncan declares Malcolm his official heir, forcing Macbeth to confront the prophecy's obstacle.", at_location_id="LOC_FORRES_COURT"),
        EventNode(id="EVT_LETTER_SENT", fabula_time=4000, syuzhet_index=7,
                  event_type="choice", actor_ids=["ENT_MACBETH"], target_ids=["ENT_LADY_MACBETH"],
                  description="Macbeth sends a letter to Lady Macbeth telling her of the witches' prophecies.", at_location_id="LOC_BATTLEFIELD"),
        EventNode(id="EVT_LADY_MACBETH_PERSUADES", fabula_time=5000, syuzhet_index=8,
                  event_type="choice", actor_ids=["ENT_LADY_MACBETH"], target_ids=["ENT_MACBETH"],
                  description="Lady Macbeth persuades a wavering Macbeth to murder Duncan that very night.", at_location_id="LOC_INVERNESS_CASTLE"),
        EventNode(id="EVT_DUNCAN_ARRIVES_INVERNESS", fabula_time=5500, syuzhet_index=9,
                  event_type="outcome", actor_ids=["ENT_DUNCAN"], target_ids=[],
                  description="Duncan arrives at Inverness Castle as Macbeth's guest.", at_location_id="LOC_INVERNESS_CASTLE"),
        EventNode(id="EVT_DUNCAN_MURDER", fabula_time=6000, syuzhet_index=10,
                  event_type="choice", actor_ids=["ENT_MACBETH"], target_ids=["ENT_DUNCAN"],
                  description="Macbeth stabs the sleeping King Duncan to death.", at_location_id="LOC_INVERNESS_CASTLE"),
        EventNode(id="EVT_SERVANTS_FRAMED", fabula_time=6500, syuzhet_index=11,
                  event_type="choice", actor_ids=["ENT_LADY_MACBETH"], target_ids=[],
                  description="Lady Macbeth plants the bloody daggers on Duncan's drugged chamber attendants.", at_location_id="LOC_INVERNESS_CASTLE"),
        EventNode(id="EVT_DUNCAN_DISCOVERED_MURDERED", fabula_time=7000, syuzhet_index=12,
                  event_type="revelation", actor_ids=["ENT_MACDUFF"], target_ids=[],
                  description="Macduff discovers Duncan's body and raises the alarm at Inverness.", at_location_id="LOC_INVERNESS_CASTLE"),
        EventNode(id="EVT_MACBETH_KILLS_SERVANTS", fabula_time=7500, syuzhet_index=14,
                  event_type="choice", actor_ids=["ENT_MACBETH"], target_ids=[],
                  description="Macbeth impulsively kills the framed servants to silence them, claiming vengeance.", at_location_id="LOC_INVERNESS_CASTLE"),
        EventNode(id="EVT_SONS_FLEE", fabula_time=9000, syuzhet_index=15,
                  event_type="choice", actor_ids=["ENT_MALCOLM"], target_ids=[],
                  description="Duncan's sons Malcolm and Donalbain flee Scotland for England, fearing assassination.", at_location_id="LOC_ENGLAND"),
        EventNode(id="EVT_MACBETH_CROWNED", fabula_time=10000, syuzhet_index=16,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_MACBETH"],
                  description="With Duncan's heirs in flight, Macbeth assumes the throne as King of Scotland.", at_location_id="LOC_FORRES_COURT"),
        EventNode(id="EVT_BANQUO_MURDERED", fabula_time=11000, syuzhet_index=17,
                  event_type="choice", actor_ids=["ENT_MACBETH"], target_ids=["ENT_BANQUO"],
                  description="Macbeth's hired murderers ambush Banquo and kill him; Fleance escapes into the night.", at_location_id="LOC_DUNSINANE_CASTLE"),
        EventNode(id="EVT_BANQUO_GHOST", fabula_time=12000, syuzhet_index=18,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_MACBETH"],
                  description="Banquo's ghost appears at Macbeth's banquet, visible only to him; he raves in front of his lords.", at_location_id="LOC_FORRES_COURT"),
        EventNode(id="EVT_LENNOX_WHISPER", fabula_time=12500, syuzhet_index=19,
                  event_type="revelation", actor_ids=["ENT_LENNOX"], target_ids=[],
                  description="Lennox confides to another lord his suspicion that Macbeth is a murdering tyrant.", at_location_id="LOC_INVERNESS_CASTLE"),
        EventNode(id="EVT_WITCHES_PROPHECY_2", fabula_time=13000, syuzhet_index=20,
                  event_type="revelation", actor_ids=["ENT_WITCHES"], target_ids=["ENT_MACBETH"],
                  description="In the cavern the witches summon apparitions: beware Macduff, none of woman born can harm Macbeth, safe until Birnam Wood comes to Dunsinane.", at_location_id="LOC_HEATH"),
        EventNode(id="EVT_MACDUFF_FLEES_TO_ENGLAND", fabula_time=13500, syuzhet_index=22,
                  event_type="choice", actor_ids=["ENT_MACDUFF"], target_ids=[],
                  description="Macduff travels to England to seek out Malcolm and raise an army.", at_location_id="LOC_ENGLAND"),
        EventNode(id="EVT_MACDUFF_FAMILY_SLAUGHTERED", fabula_time=14000, syuzhet_index=23,
                  event_type="choice", actor_ids=["ENT_MACBETH"], target_ids=["ENT_LADY_MACDUFF"],
                  description="Macbeth orders his hired men to seize Macduff's castle and slaughter his wife and children.", at_location_id="LOC_DUNSINANE_CASTLE"),
        EventNode(id="EVT_MACDUFF_LEARNS_OF_MASSACRE", fabula_time=14500, syuzhet_index=24,
                  event_type="revelation", actor_ids=[], target_ids=["ENT_MACDUFF"],
                  description="The Thane of Ross arrives in England and tells Macduff that his family has been slaughtered.", at_location_id="LOC_ENGLAND"),
        EventNode(id="EVT_MALCOLM_MACDUFF_ALLIANCE", fabula_time=15000, syuzhet_index=25,
                  event_type="choice", actor_ids=["ENT_MALCOLM", "ENT_MACDUFF"], target_ids=[],
                  description="Provoked by grief, Macduff joins Malcolm's English-backed army to retake Scotland.", at_location_id="LOC_ENGLAND"),
        EventNode(id="EVT_LADY_MACBETH_SLEEPWALKING", fabula_time=16000, syuzhet_index=26,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_LADY_MACBETH"],
                  description="Lady Macbeth sleepwalks at Dunsinane, washing imaginary bloodstains from her hands while a doctor and gentlewoman watch.", at_location_id="LOC_DUNSINANE_CASTLE"),
        EventNode(id="EVT_LADY_MACBETH_DEATH", fabula_time=17000, syuzhet_index=28,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_LADY_MACBETH"],
                  description="Lady Macbeth dies — implied suicide — prompting Macbeth's 'tomorrow' soliloquy.", at_location_id="LOC_DUNSINANE_CASTLE"),
        EventNode(id="EVT_BIRNAM_WOOD_MOVES", fabula_time=18500, syuzhet_index=29,
                  event_type="outcome", actor_ids=["ENT_MALCOLM"], target_ids=[],
                  description="Malcolm's army cuts boughs from Birnam Wood and advances on Dunsinane camouflaged, fulfilling the prophecy.", at_location_id="LOC_ENGLAND"),
        EventNode(id="EVT_MACBETH_KILLED", fabula_time=19000, syuzhet_index=30,
                  event_type="outcome", actor_ids=["ENT_MACDUFF"], target_ids=["ENT_MACBETH"],
                  description="Macduff, born by caesarean section, kills Macbeth in single combat at Dunsinane.", at_location_id="LOC_ENGLAND"),
        EventNode(id="EVT_MALCOLM_CROWNED", fabula_time=20000, syuzhet_index=32,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_MALCOLM"],
                  description="Malcolm is crowned King of Scotland at Scone; legitimate feudal order is restored.", at_location_id="LOC_DUNSINANE_CASTLE"),

        # ── UTTERANCES (discrete on-page speech-acts; woven into syuzhet at the
        #    point the reader actually encounters them) ────────────────────
        EventNode(id="EVT_UTT_PROPHECY_HEATH", event_type="utterance",
                  description="On the heath the witches deliver their first prophetic salutation to Macbeth and Banquo.",
                  at_location_id="LOC_HEATH", speaker_id="ENT_WITCHES", addressee_ids=["ENT_MACBETH", "ENT_BANQUO"],
                  actor_ids=["ENT_WITCHES"], target_ids=["EVT_WITCHES_PROPHECY_1"],
                  content="All hail Macbeth — Thane of Glamis, Thane of Cawdor, king hereafter; and Banquo, lesser yet greater, shall father a line of kings though never wear the crown himself.",
                  via_channel_id="CHN_PROPHETIC_LINK", truth_value="performative",
                  fabula_time=2000, syuzhet_index=3),
        EventNode(id="EVT_UTT_DUNCAN_BESTOWS_CAWDOR", event_type="utterance",
                  description="At Forres, Duncan formally proclaims the title of Thane of Cawdor transferred to Macbeth, conveyed to him via Ross.",
                  at_location_id="LOC_FORRES_COURT", speaker_id="ENT_DUNCAN", addressee_ids=["ENT_MACBETH"],
                  actor_ids=["ENT_DUNCAN"], target_ids=["EVT_CAWDOR_TITLE"],
                  content="What the traitor Cawdor hath lost, noble Macbeth hath won — his title and lands are bestowed upon Macbeth.",
                  via_channel_id=None, truth_value="performative",
                  fabula_time=3000, syuzhet_index=5),
        EventNode(id="EVT_UTT_MACDUFF_ALARM", event_type="utterance",
                  description="Bursting back from the murder chamber, Macduff raises the general alarm at Inverness.",
                  at_location_id="LOC_INVERNESS_CASTLE", speaker_id="ENT_MACDUFF", addressee_ids=["ENT_LENNOX", "ENT_MALCOLM"],
                  actor_ids=["ENT_MACDUFF"], target_ids=["EVT_DUNCAN_DISCOVERED_MURDERED", "EVT_DUNCAN_MURDER"],
                  content="O horror, horror, horror — most sacrilegious murder hath broke ope the Lord's anointed temple; awake, awake, ring the alarum bell!",
                  via_channel_id=None, truth_value="true",
                  fabula_time=7000, syuzhet_index=13),
        EventNode(id="EVT_UTT_PROPHECY_CAVERN", event_type="utterance",
                  description="In the witches' cavern, the apparitions answer Macbeth's demand for further prophecy.",
                  at_location_id="LOC_HEATH", speaker_id="ENT_WITCHES", addressee_ids=["ENT_MACBETH"],
                  actor_ids=["ENT_WITCHES"], target_ids=["EVT_WITCHES_PROPHECY_2"],
                  content="Beware Macduff; none of woman born shall harm Macbeth; he is safe until Great Birnam Wood comes against him to Dunsinane Hill.",
                  via_channel_id="CHN_PROPHETIC_LINK", truth_value="performative",
                  fabula_time=13000, syuzhet_index=21),
        EventNode(id="EVT_UTT_LADY_MACBETH_SLEEPWALK_CONFESSION", event_type="utterance",
                  description="In a guilt-trance at Dunsinane, Lady Macbeth confesses the regicide and the slaughter of Macduff's wife while a doctor and gentlewoman watch unseen.",
                  at_location_id="LOC_INVERNESS_CASTLE", speaker_id="ENT_LADY_MACBETH", addressee_ids=["ENT_DOCTOR", "ENT_GENTLEWOMAN"],
                  actor_ids=["ENT_LADY_MACBETH"], target_ids=["EVT_DUNCAN_MURDER", "EVT_BANQUO_MURDERED", "EVT_MACDUFF_FAMILY_SLAUGHTERED"],
                  content="Out, damned spot — yet who would have thought the old man to have had so much blood in him? The Thane of Fife had a wife: where is she now?",
                  via_channel_id=None, truth_value="true",
                  fabula_time=16000, syuzhet_index=27),
        EventNode(id="EVT_UTT_MACBETH_INVINCIBILITY_TAUNT", event_type="utterance",
                  description="At Dunsinane, Macbeth taunts the closing Macduff with the witches' second prophecy, certain he cannot be killed by any man of woman born.",
                  at_location_id="LOC_DUNSINANE_CASTLE", speaker_id="ENT_MACBETH", addressee_ids=["ENT_MACDUFF"],
                  actor_ids=["ENT_MACBETH"], target_ids=["EVT_MACBETH_KILLED", "EVT_WITCHES_PROPHECY_2"],
                  content="Thou losest labour — I bear a charmèd life which must not yield to one of woman born.",
                  via_channel_id=None, truth_value="false",
                  fabula_time=19000, syuzhet_index=31),
    ],

    # ── CAUSAL TOPOLOGY ─────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction (Event → Event) ──
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="EVT_CAWDOR_TITLE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000, propagation_delay=2000),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="EVT_LETTER_SENT",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2000, propagation_delay=2000),
        CausalEdge(source_id="EVT_MALCOLM_NAMED_HEIR", target_id="EVT_LADY_MACBETH_PERSUADES",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=6.0, fabula_time=3500, propagation_delay=1500),
        CausalEdge(source_id="EVT_LETTER_SENT", target_id="EVT_LADY_MACBETH_PERSUADES",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=7.0, fabula_time=4000, propagation_delay=1000),
        CausalEdge(source_id="EVT_LADY_MACBETH_PERSUADES", target_id="EVT_DUNCAN_MURDER",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=5000, propagation_delay=1000),
        # NOTE: EVT_DUNCAN_ARRIVES_INVERNESS is a *precondition* for the
        # murder (Duncan must be co-located at Inverness), not a
        # sufficient cause. The schema requires Event→Event edges be
        # ``chain_reaction`` (which is treated as a sufficient cause by
        # Pearl's disjunctive prune rule), so we do not wire arrival →
        # murder. Co-location is established through the entity-location
        # snapshots / spatial topology instead, and the persuasion is
        # the murder's sole chain_reaction parent — letting
        # do(EVT_LADY_MACBETH_PERSUADES=prevented) correctly cascade
        # to suppress the murder via _compute_shadow_prune_closure.
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="EVT_SERVANTS_FRAMED",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=6000, propagation_delay=500),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="EVT_DUNCAN_DISCOVERED_MURDERED",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000, propagation_delay=1000),
        CausalEdge(source_id="EVT_DUNCAN_DISCOVERED_MURDERED", target_id="EVT_MACBETH_KILLS_SERVANTS",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=7000, propagation_delay=500),
        CausalEdge(source_id="EVT_DUNCAN_DISCOVERED_MURDERED", target_id="EVT_SONS_FLEE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000, propagation_delay=2000),
        CausalEdge(source_id="EVT_SONS_FLEE", target_id="EVT_MACBETH_CROWNED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=9000, propagation_delay=1000),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="EVT_BANQUO_MURDERED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=6.0, fabula_time=2000, propagation_delay=9000),
        CausalEdge(source_id="EVT_BANQUO_MURDERED", target_id="EVT_BANQUO_GHOST",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=11000, propagation_delay=1000),
        CausalEdge(source_id="EVT_BANQUO_GHOST", target_id="EVT_LENNOX_WHISPER",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=12000, propagation_delay=500),
        CausalEdge(source_id="EVT_BANQUO_GHOST", target_id="EVT_WITCHES_PROPHECY_2",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=12000, propagation_delay=1000),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_2", target_id="EVT_MACDUFF_FAMILY_SLAUGHTERED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000, propagation_delay=1000),
        CausalEdge(source_id="EVT_LENNOX_WHISPER", target_id="EVT_MACDUFF_FLEES_TO_ENGLAND",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=12500, propagation_delay=1000),
        CausalEdge(source_id="EVT_MACDUFF_FLEES_TO_ENGLAND", target_id="EVT_MACDUFF_FAMILY_SLAUGHTERED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=13500, propagation_delay=500),
        CausalEdge(source_id="EVT_MACDUFF_FAMILY_SLAUGHTERED", target_id="EVT_MACDUFF_LEARNS_OF_MASSACRE",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=7.0, fabula_time=14000, propagation_delay=500),
        CausalEdge(source_id="EVT_MACDUFF_LEARNS_OF_MASSACRE", target_id="EVT_MALCOLM_MACDUFF_ALLIANCE",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=9.0, fabula_time=14500, propagation_delay=500),
        CausalEdge(source_id="EVT_LADY_MACBETH_SLEEPWALKING", target_id="EVT_LADY_MACBETH_DEATH",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=16000, propagation_delay=1000),
        CausalEdge(source_id="EVT_MALCOLM_MACDUFF_ALLIANCE", target_id="EVT_BIRNAM_WOOD_MOVES",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=15000, propagation_delay=3500),
        CausalEdge(source_id="EVT_BIRNAM_WOOD_MOVES", target_id="EVT_MACBETH_KILLED",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=18500, propagation_delay=500),
        CausalEdge(source_id="EVT_MACBETH_KILLED", target_id="EVT_MALCOLM_CROWNED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=19000, propagation_delay=1000),

        # ── mutation (Event → Entity trait/status) ──
        # Each one has a matching state_timeline snapshot above.
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2000,
                   trait_target="ambition", trait_delta=0.4),
        CausalEdge(source_id="EVT_LADY_MACBETH_PERSUADES", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=5000,
                   trait_target="ruthlessness", trait_delta=0.4),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=6000,
                   trait_target="guilt", trait_delta=0.7),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=6000,
                   trait_target="paranoia", trait_delta=0.5),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="ENT_LADY_MACBETH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=6000,
                   trait_target="guilt", trait_delta=0.4),
        CausalEdge(source_id="EVT_MACBETH_CROWNED", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=10000,
                   trait_target="paranoia", trait_delta=0.3),
        CausalEdge(source_id="EVT_MACBETH_CROWNED", target_id="ENT_BANQUO",
                   causality_type="mutation", mechanism="epistemic", evidence_strength="strong",
                   causal_force=7.0, fabula_time=10000,
                   trait_target="suspicion", trait_delta=0.6, propagation_delay=500),
        CausalEdge(source_id="EVT_BANQUO_MURDERED", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=11000,
                   trait_target="paranoia", trait_delta=0.4),
        CausalEdge(source_id="EVT_BANQUO_MURDERED", target_id="ENT_FLEANCE",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=11000,
                   trait_target="fear", trait_delta=0.7),
        CausalEdge(source_id="EVT_LENNOX_WHISPER", target_id="ENT_LENNOX",
                   causality_type="mutation", mechanism="epistemic", evidence_strength="strong",
                   causal_force=5.0, fabula_time=12500,
                   trait_target="suspicion", trait_delta=0.5),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_2", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=13000,
                   trait_target="courage", trait_delta=0.3),
        CausalEdge(source_id="EVT_DUNCAN_DISCOVERED_MURDERED", target_id="ENT_MACDUFF",
                   causality_type="mutation", mechanism="epistemic", evidence_strength="strong",
                   causal_force=6.0, fabula_time=7500,
                   trait_target="suspicion", trait_delta=0.5),
        CausalEdge(source_id="EVT_MACDUFF_LEARNS_OF_MASSACRE", target_id="ENT_MACDUFF",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=14500,
                   trait_target="grief", trait_delta=1.0),
        CausalEdge(source_id="EVT_MACDUFF_LEARNS_OF_MASSACRE", target_id="ENT_MACDUFF",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=10.0, fabula_time=14500,
                   trait_target="vengefulness", trait_delta=1.0),
        CausalEdge(source_id="EVT_LADY_MACBETH_SLEEPWALKING", target_id="ENT_LADY_MACBETH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=16000,
                   trait_target="guilt", trait_delta=0.7),
        CausalEdge(source_id="EVT_LADY_MACBETH_DEATH", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=9.0, fabula_time=17000,
                   trait_target="despair", trait_delta=0.8),
        CausalEdge(source_id="EVT_MACBETH_KILLED", target_id="ENT_MACBETH",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=19000,
                   trait_target="courage", trait_delta=-1.0),
        CausalEdge(source_id="EVT_MACDUFF_FAMILY_SLAUGHTERED", target_id="ENT_LADY_MACDUFF",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=10.0, fabula_time=14000,
                   trait_target="courage", trait_delta=-1.0),

        # ── mutation_social (Event → Relationship metric) ──
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="ENT_MACBETH",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=2000,
                   trait_target="fear", trait_delta=0.4, rel_counterpart_id="ENT_BANQUO"),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="ENT_MACBETH",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=6000,
                   trait_target="fear", trait_delta=0.5, rel_counterpart_id="ENT_BANQUO"),
        CausalEdge(source_id="EVT_BANQUO_MURDERED", target_id="ENT_MALCOLM",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="moderate",
                   causal_force=6.0, fabula_time=11000,
                   trait_target="affinity", trait_delta=-0.5, rel_counterpart_id="ENT_MACBETH"),
        CausalEdge(source_id="EVT_MACDUFF_FAMILY_SLAUGHTERED", target_id="ENT_MACDUFF",
                   causality_type="mutation_social", mechanism="betrayal", evidence_strength="strong",
                   causal_force=10.0, fabula_time=14000,
                   trait_target="affinity", trait_delta=-1.0, rel_counterpart_id="ENT_MACBETH"),
        CausalEdge(source_id="EVT_MACDUFF_LEARNS_OF_MASSACRE", target_id="ENT_MACDUFF",
                   causality_type="mutation_social", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=14500,
                   trait_target="affinity", trait_delta=0.4, rel_counterpart_id="ENT_MALCOLM"),
        CausalEdge(source_id="EVT_LADY_MACBETH_PERSUADES", target_id="ENT_MACBETH",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=5000,
                   trait_target="power_dynamic", trait_delta=-0.3, rel_counterpart_id="ENT_LADY_MACBETH"),
        CausalEdge(source_id="EVT_BANQUO_GHOST", target_id="ENT_MACBETH",
                   causality_type="mutation_social", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=12000,
                   trait_target="fear", trait_delta=0.5, rel_counterpart_id="ENT_BANQUO"),

        # ── affordance_gate (State → Event) ──
        CausalEdge(source_id="ENT_LADY_MACBETH", target_id="EVT_LADY_MACBETH_PERSUADES",
                   causality_type="affordance_gate", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=5000),
        CausalEdge(source_id="OBJ_BLOODY_DAGGERS", target_id="EVT_DUNCAN_MURDER",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=6000),
        CausalEdge(source_id="ENT_WITCHES", target_id="EVT_WITCHES_PROPHECY_1",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2000),
        CausalEdge(source_id="OBJ_APPARITIONS", target_id="EVT_WITCHES_PROPHECY_2",
                   causality_type="affordance_gate", mechanism="epistemic", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000),
        CausalEdge(source_id="OBJ_BIRNAM_BOUGHS", target_id="EVT_BIRNAM_WOOD_MOVES",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=18500),
        CausalEdge(source_id="ENT_MACDUFF", target_id="EVT_MACBETH_KILLED",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=19000),

        # ── ambient_propagation (State → State) ──
        CausalEdge(source_id="LOC_DUNSINANE_CASTLE", target_id="ENT_MACBETH",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="weak",
                   causal_force=3.0, fabula_time=13000),
        CausalEdge(source_id="LOC_HEATH", target_id="ENT_MACBETH",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="weak",
                   causal_force=3.0, fabula_time=2000),
        CausalEdge(source_id="ENT_MACBETH", target_id="ENT_LADY_MACBETH",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="weak",
                   causal_force=2.5, fabula_time=12000),

        # ── WORLD_ → Event (named-latent common-cause wiring) ──
        CausalEdge(source_id="WORLD_FEUDAL_HIERARCHY", target_id="EVT_REBELLION_DEFEATED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=1000),
        CausalEdge(source_id="WORLD_FEUDAL_HIERARCHY", target_id="EVT_CAWDOR_TITLE",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=3000),
        CausalEdge(source_id="WORLD_FEUDAL_HIERARCHY", target_id="EVT_MALCOLM_NAMED_HEIR",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=3500),
        CausalEdge(source_id="WORLD_FEUDAL_HIERARCHY", target_id="EVT_MACBETH_CROWNED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=10000),
        CausalEdge(source_id="WORLD_FEUDAL_HIERARCHY", target_id="EVT_MALCOLM_CROWNED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=20000),
        CausalEdge(source_id="WORLD_SUPERNATURAL_PROPHECY", target_id="EVT_DUNCAN_MURDER",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=6000),
        CausalEdge(source_id="WORLD_SUPERNATURAL_PROPHECY", target_id="EVT_BANQUO_MURDERED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=11000),
        CausalEdge(source_id="WORLD_SUPERNATURAL_PROPHECY", target_id="EVT_MACDUFF_FAMILY_SLAUGHTERED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=14000),
        CausalEdge(source_id="WORLD_SUPERNATURAL_PROPHECY", target_id="EVT_BIRNAM_WOOD_MOVES",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=18500),
        CausalEdge(source_id="WORLD_SUPERNATURAL_PROPHECY", target_id="EVT_MACBETH_KILLED",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=19000),
        CausalEdge(source_id="WORLD_DIVINE_RIGHT", target_id="EVT_DUNCAN_MURDER",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=3.0, fabula_time=6000),
        CausalEdge(source_id="WORLD_DIVINE_RIGHT", target_id="EVT_LADY_MACBETH_SLEEPWALKING",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=16000),
        # AUDIT P0-8: tighten Lady Macbeth's sleepwalking onto the
        # regicide path so counterfactual "what if Duncan lived?"
        # surgeries propagate to the bedchamber scene rather than
        # only flowing through the cosmic-vengeance latent.
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="EVT_LADY_MACBETH_SLEEPWALKING",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=16000,
                   description="Lady Macbeth's guilt over goading her husband to regicide returns as somnambulistic hand-washing."),

        # ── WORLD_ → WORLD_ (named-latent forces destabilising one another) ──
        CausalEdge(source_id="WORLD_SUPERNATURAL_PROPHECY", target_id="WORLD_DIVINE_RIGHT",
                   causality_type="chain_reaction", mechanism="epistemic", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2000,
                   description="The prophecy seduces Macbeth into regicide, weaponising fate against the divine order it transgresses."),
        CausalEdge(source_id="WORLD_DIVINE_RIGHT", target_id="WORLD_FEUDAL_HIERARCHY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=6000,
                   description="Cosmic vengeance for regicide \u2014 madness, sleepwalking, sterile crown \u2014 cascades into the political collapse of the feudal compact."),
        CausalEdge(source_id="WORLD_SUPERNATURAL_PROPHECY", target_id="WORLD_FEUDAL_HIERARCHY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=6000,
                   description="The prophecy directly nominates a usurper, bypassing the feudal succession the hierarchy enforces."),

        # ── orphan utterance wirings ──
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="EVT_UTT_PROPHECY_HEATH",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=8.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_CAWDOR_TITLE", target_id="EVT_UTT_DUNCAN_BESTOWS_CAWDOR",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_DUNCAN_DISCOVERED_MURDERED", target_id="EVT_UTT_MACDUFF_ALARM",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_2", target_id="EVT_UTT_PROPHECY_CAVERN",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000, propagation_delay=0),
        CausalEdge(source_id="EVT_LADY_MACBETH_SLEEPWALKING", target_id="EVT_UTT_LADY_MACBETH_SLEEPWALK_CONFESSION",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=16000, propagation_delay=0),
        CausalEdge(source_id="EVT_UTT_MACBETH_INVINCIBILITY_TAUNT", target_id="EVT_MACBETH_KILLED",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=6.0, fabula_time=19000, propagation_delay=0),
        # Iconic-symbol affordance gates: each object materially enables
        # its anchor event (no letter, no remote persuasion; no crown, no
        # legitimate kingship to lose; no cauldron, no second prophecy
        # ritual to stage).
        CausalEdge(source_id="OBJ_LETTER", target_id="EVT_LETTER_SENT",
                   causality_type="affordance_gate", mechanism="informational", evidence_strength="strong",
                   causal_force=8.0, fabula_time=4000),
        CausalEdge(source_id="OBJ_CROWN", target_id="EVT_MACBETH_CROWNED",
                   causality_type="affordance_gate", mechanism="social", evidence_strength="strong",
                   causal_force=7.0, fabula_time=10000),
        CausalEdge(source_id="OBJ_CAULDRON", target_id="EVT_WITCHES_PROPHECY_2",
                   causality_type="affordance_gate", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=13000),


        # ─── auto-patched mutation_social edges (per-axis coverage) ───
        CausalEdge(source_id="EVT_LETTER_SENT", target_id="ENT_MACBETH", rel_counterpart_id="ENT_LADY_MACBETH", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=4000, propagation_delay=0),
        CausalEdge(source_id="EVT_LETTER_SENT", target_id="ENT_LADY_MACBETH", rel_counterpart_id="ENT_MACBETH", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=4000, propagation_delay=0),
        CausalEdge(source_id="EVT_CAWDOR_TITLE", target_id="ENT_MACBETH", rel_counterpart_id="ENT_DUNCAN", causality_type="mutation_social", trait_target="affinity", trait_delta=0.45, mechanism="social", evidence_strength="strong", causal_force=5.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="ENT_MACBETH", rel_counterpart_id="ENT_DUNCAN", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.5, mechanism="betrayal", evidence_strength="strong", causal_force=9.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="ENT_MACBETH", rel_counterpart_id="ENT_BANQUO", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="social", evidence_strength="strong", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="ENT_BANQUO", rel_counterpart_id="ENT_MACBETH", causality_type="mutation_social", trait_target="affinity", trait_delta=0.4, mechanism="social", evidence_strength="strong", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_MACBETH_CROWNED", target_id="ENT_BANQUO", rel_counterpart_id="ENT_MACBETH", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.4, mechanism="epistemic", evidence_strength="strong", causal_force=6.0, fabula_time=10000, propagation_delay=500),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="ENT_MACBETH", rel_counterpart_id="ENT_MACDUFF", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.3, mechanism="social", evidence_strength="moderate", causal_force=6.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_MACDUFF_FAMILY_SLAUGHTERED", target_id="ENT_MACBETH", rel_counterpart_id="ENT_MACDUFF", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.5, mechanism="physical", evidence_strength="strong", causal_force=9.0, fabula_time=14000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="ENT_MACBETH", rel_counterpart_id="ENT_WITCHES", causality_type="mutation_social", trait_target="affinity", trait_delta=0.2, mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_MALCOLM_MACDUFF_ALLIANCE", target_id="ENT_MALCOLM", rel_counterpart_id="ENT_MACDUFF", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=15000, propagation_delay=0),
        CausalEdge(source_id="EVT_MACBETH_CROWNED", target_id="ENT_BANQUO", rel_counterpart_id="ENT_MACBETH", causality_type="mutation_social", trait_target="fear", trait_delta=0.4, mechanism="epistemic", evidence_strength="strong", causal_force=6.0, fabula_time=10000, propagation_delay=500),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_2", target_id="ENT_MACBETH", rel_counterpart_id="ENT_MACDUFF", causality_type="mutation_social", trait_target="fear", trait_delta=0.6, mechanism="epistemic", evidence_strength="strong", causal_force=7.0, fabula_time=13000, propagation_delay=0),
        CausalEdge(source_id="EVT_MACDUFF_LEARNS_OF_MASSACRE", target_id="ENT_MACDUFF", rel_counterpart_id="ENT_MACBETH", causality_type="mutation_social", trait_target="fear", trait_delta=0.2, mechanism="emotional", evidence_strength="weak", causal_force=4.0, fabula_time=14500, propagation_delay=0),
        CausalEdge(source_id="EVT_SONS_FLEE", target_id="ENT_MALCOLM", rel_counterpart_id="ENT_MACBETH", causality_type="mutation_social", trait_target="fear", trait_delta=0.6, mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=9000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="ENT_MACBETH", rel_counterpart_id="ENT_WITCHES", causality_type="mutation_social", trait_target="fear", trait_delta=0.5, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_BANQUO_MURDERED", target_id="ENT_MACBETH", rel_counterpart_id="ENT_FLEANCE", causality_type="mutation_social", trait_target="fear", trait_delta=0.65, mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=11000, propagation_delay=0),
        CausalEdge(source_id="EVT_LADY_MACBETH_PERSUADES", target_id="ENT_LADY_MACBETH", rel_counterpart_id="ENT_MACBETH", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.4, mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_CAWDOR_TITLE", target_id="ENT_MACBETH", rel_counterpart_id="ENT_DUNCAN", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.4, mechanism="social", evidence_strength="strong", causal_force=5.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_DUNCAN_MURDER", target_id="ENT_MACBETH", rel_counterpart_id="ENT_DUNCAN", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.4, mechanism="physical", evidence_strength="strong", causal_force=9.0, fabula_time=6000, propagation_delay=0),
        CausalEdge(source_id="EVT_MACBETH_KILLED", target_id="ENT_MACDUFF", rel_counterpart_id="ENT_MACBETH", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.4, mechanism="physical", evidence_strength="strong", causal_force=10.0, fabula_time=19000, propagation_delay=0),
        CausalEdge(source_id="EVT_MALCOLM_MACDUFF_ALLIANCE", target_id="ENT_MACDUFF", rel_counterpart_id="ENT_MALCOLM", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.3, mechanism="social", evidence_strength="moderate", causal_force=6.0, fabula_time=15000, propagation_delay=0),
        CausalEdge(source_id="EVT_MALCOLM_MACDUFF_ALLIANCE", target_id="ENT_MALCOLM", rel_counterpart_id="ENT_MACDUFF", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.3, mechanism="social", evidence_strength="moderate", causal_force=6.0, fabula_time=15000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="ENT_MACBETH", rel_counterpart_id="ENT_WITCHES", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.6, mechanism="epistemic", evidence_strength="strong", causal_force=7.0, fabula_time=2000, propagation_delay=0),
        # ─── placeholder remediation: Lady Macbeth → Duncan covert hostility ───
        CausalEdge(source_id="EVT_LADY_MACBETH_PERSUADES", target_id="ENT_LADY_MACBETH", rel_counterpart_id="ENT_DUNCAN", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.9, mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_LADY_MACBETH_PERSUADES", target_id="ENT_LADY_MACBETH", rel_counterpart_id="ENT_DUNCAN", causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.6, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=5000, propagation_delay=0),
        # ── auto-backfilled per-axis mutation_social ──
        CausalEdge(source_id="EVT_CAWDOR_TITLE", target_id="ENT_DUNCAN", rel_counterpart_id="ENT_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.26,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="ENT_MACBETH", rel_counterpart_id="ENT_MALCOLM",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.21,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_CAWDOR_TITLE", target_id="ENT_DUNCAN", rel_counterpart_id="ENT_LADY_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.18,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="ENT_WITCHES", rel_counterpart_id="ENT_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.09,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="ENT_FLEANCE", rel_counterpart_id="ENT_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.27,
                   mechanism="emotional", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="ENT_MACBETH", rel_counterpart_id="ENT_MALCOLM",  # auto-backfill
                   causality_type="mutation_social", trait_target="fear", trait_delta=0.17,
                   mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="ENT_FLEANCE", rel_counterpart_id="ENT_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="fear", trait_delta=0.27,
                   mechanism="psychological", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_CAWDOR_TITLE", target_id="ENT_DUNCAN", rel_counterpart_id="ENT_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.21,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="ENT_MACBETH", rel_counterpart_id="ENT_MALCOLM",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.12,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_CAWDOR_TITLE", target_id="ENT_DUNCAN", rel_counterpart_id="ENT_LADY_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.18,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCHES_PROPHECY_1", target_id="ENT_WITCHES", rel_counterpart_id="ENT_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.18,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=2000, propagation_delay=0),
        CausalEdge(source_id="EVT_REBELLION_DEFEATED", target_id="ENT_FLEANCE", rel_counterpart_id="ENT_MACBETH",  # auto-backfill
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.21,
                   mechanism="social", evidence_strength="moderate", causal_force=4.0, fabula_time=1000, propagation_delay=0),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_BATTLEFIELD", target_id="LOC_HEATH"),
        SpatialEdge(source_id="LOC_BATTLEFIELD", target_id="LOC_FORRES_COURT"),
        SpatialEdge(source_id="LOC_HEATH", target_id="LOC_FORRES_COURT"),
        SpatialEdge(source_id="LOC_HEATH", target_id="LOC_WITCHES_CAVERN"),
        SpatialEdge(source_id="LOC_FORRES_COURT", target_id="LOC_INVERNESS_CASTLE"),
        SpatialEdge(source_id="LOC_INVERNESS_CASTLE", target_id="LOC_DUNSINANE_CASTLE"),
        SpatialEdge(source_id="LOC_DUNSINANE_CASTLE", target_id="LOC_INVERNESS_CASTLE"),
        SpatialEdge(source_id="LOC_DUNSINANE_CASTLE", target_id="LOC_MACDUFF_CASTLE"),
        SpatialEdge(source_id="LOC_DUNSINANE_CASTLE", target_id="LOC_BIRNAM_WOOD"),
        SpatialEdge(source_id="LOC_BIRNAM_WOOD", target_id="LOC_DUNSINANE_CASTLE"),
        SpatialEdge(source_id="LOC_MACDUFF_CASTLE", target_id="LOC_ENGLAND"),
        SpatialEdge(source_id="LOC_ENGLAND", target_id="LOC_BIRNAM_WOOD"),
        SpatialEdge(source_id="LOC_DUNSINANE_CASTLE", target_id="LOC_WITCHES_CAVERN"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    # Only STANDING communication capabilities live here. One-shot speech-acts
    # (Duncan's proclamation, Macduff's alarm cry, Macbeth's pre-duel taunt,
    # Lady Macbeth's sleepwalking confession) are modelled solely as utterance
    # EventNodes with via_channel_id=None.
    channels={
        "CHN_PROPHETIC_LINK": Channel(
            id="CHN_PROPHETIC_LINK",
            name="Witches' Prophetic Link to Macbeth",
            medium="prophetic_link",
            participant_ids=["ENT_WITCHES", "ENT_MACBETH", "ENT_BANQUO"],
            directionality="broadcast",
            # Riddling utterance: Macbeth and Banquo decode only the literal
            # surface, missing the equivocations that ultimately doom them.
            intelligibility={"ENT_MACBETH": 0.6, "ENT_BANQUO": 0.6},
            established_at_fabula=2000,
            terminated_at_fabula=13000,
            evidence_strength="strong",
        ),
        "CHN_NOBLE_DISSENT": Channel(
            id="CHN_NOBLE_DISSENT",
            name="Whispered Dissent Among the Loyalist Thanes",
            medium="classified_pipeline",
            participant_ids=["ENT_LENNOX", "ENT_MACDUFF", "ENT_MALCOLM"],
            directionality="duplex",
            intelligibility={},
            established_at_fabula=11000,
            terminated_at_fabula=None,
            evidence_strength="moderate",
        ),
    },

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_FEUDAL_HIERARCHY": GlobalTrait(
            id="WORLD_FEUDAL_HIERARCHY",
            name="Scottish Feudal Hierarchy",
            description="Scotland's feudal order in which thanes earn or forfeit royal favour through valour and bloodshed; succession is governed by the king's nominated heir. Drives ambition, loyalty oaths, and the legitimacy of seizure-by-violence.",
            category="governance",
            magnitude=TraitVector(value=0.8, inertia=0.7, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            proposition_id="PROP_FEUDAL_ORDER_INTACT",
            state_timeline=[
                WorldTraitSnapshot(fabula_time=6000, triggered_by="EVT_DUNCAN_MURDER",
                    magnitude=TraitVector(value=0.5, inertia=0.4, evidence_strength="strong"),
                    description="Regicide shatters the feudal compact; kingship is now seized by treachery, not bestowed by lineage."),
                WorldTraitSnapshot(fabula_time=20000, triggered_by="EVT_MALCOLM_CROWNED",
                    magnitude=TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
                    description="Malcolm's coronation at Scone restores legitimate succession, but the order is freshly re-installed and inertia is somewhat lower than the pre-murder baseline."),
            ],
        ),
        "WORLD_SUPERNATURAL_PROPHECY": GlobalTrait(
            id="WORLD_SUPERNATURAL_PROPHECY",
            name="Witches' Prophecy",
            description="A latent supernatural force operating through the witches' prophecies — never an event itself, but the named-latent common cause of every choice that the prophecy frames. Blurs the line between fate and agency.",
            category="cosmology",
            magnitude=TraitVector(value=0.55, inertia=0.9, evidence_strength="moderate"),
            affected_domains=["psychological", "epistemic"],
            proposition_id="PROP_PROPHECY_BINDING",
            state_timeline=[
                WorldTraitSnapshot(fabula_time=2000, triggered_by="EVT_WITCHES_PROPHECY_1",
                    magnitude=TraitVector(value=0.7, inertia=0.9, evidence_strength="strong"),
                    description="First prophecies overtly deliver the witches' design into Macbeth's psyche."),
                WorldTraitSnapshot(fabula_time=13000, triggered_by="EVT_WITCHES_PROPHECY_2",
                    magnitude=TraitVector(value=0.85, inertia=0.95, evidence_strength="strong"),
                    description="Second prophecies deepen Macbeth's false confidence with riddling assurances."),
                WorldTraitSnapshot(fabula_time=18500, triggered_by="EVT_BIRNAM_WOOD_MOVES",
                    magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    description="Prophecy fulfilled literally; supernatural fate is shown to be inescapable."),
            ],
        ),
        "WORLD_DIVINE_RIGHT": GlobalTrait(
            id="WORLD_DIVINE_RIGHT",
            name="Divine Right of Kings",
            description="The implicit Jacobean cosmological ordering in which the murder of an anointed king is a cosmic crime that returns as madness, sleeplessness and ill omens. Operates as a named-latent moral physics on the Macbeths.",
            category="cosmology",
            magnitude=TraitVector(value=0.5, inertia=0.85, evidence_strength="moderate"),
            affected_domains=["psychological", "social"],
            proposition_id="PROP_DIVINE_ORDER_AVENGES",
        ),
    },

    # ── SOCIAL TOPOLOGY (per-axis RelationshipMetric — only observed axes) ──
    social_topology=[
        # The Macbeth marriage: strong affinity and clear power inversion (she leads).
        RelationshipEdge(
            source_entity_id="ENT_MACBETH", target_entity_id="ENT_LADY_MACBETH",
            metrics={
                "affinity":      RelationshipMetric(value=0.8, inertia=0.5, evidence_strength="strong", last_updated_fabula=4000),
                "power_dynamic": RelationshipMetric(value=-0.4, inertia=0.6, evidence_strength="strong", last_updated_fabula=5000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_LADY_MACBETH", target_entity_id="ENT_MACBETH",
            metrics={
                "affinity":      RelationshipMetric(value=0.85, inertia=0.5, evidence_strength="strong", last_updated_fabula=4000),
                "power_dynamic": RelationshipMetric(value=0.4, inertia=0.6, evidence_strength="strong", last_updated_fabula=5000),
            },
        ),
        # Macbeth → Duncan: pre-murder loyalty, then guilty avoidance (only affinity inverts; fear unobserved).
        RelationshipEdge(
            source_entity_id="ENT_MACBETH", target_entity_id="ENT_DUNCAN",
            metrics={
                "affinity":      RelationshipMetric(value=0.4, inertia=0.4, evidence_strength="moderate", last_updated_fabula=1000),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=1000),
            },
        ),
        # Duncan → Macbeth: warm royal favour; the king elevates Macbeth to Thane of Cawdor and quarters at his castle as honoured guest. Affinity is high, power is the inverse of the above (king over thane), fear is absent — Duncan is fatally untroubled.
        RelationshipEdge(
            source_entity_id="ENT_DUNCAN", target_entity_id="ENT_MACBETH",
            metrics={
                "affinity":      RelationshipMetric(value=0.85, inertia=0.4, evidence_strength="strong", last_updated_fabula=1500),
                "power_dynamic": RelationshipMetric(value=0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=1000),
            },
        ),
        # Macbeth ↔ Banquo: comradeship souring into murderous fear.
        RelationshipEdge(
            source_entity_id="ENT_MACBETH", target_entity_id="ENT_BANQUO",
            metrics={
                "affinity": RelationshipMetric(value=0.5, inertia=0.4, evidence_strength="strong", last_updated_fabula=1000),
                "fear":     RelationshipMetric(value=0.6, inertia=0.2, evidence_strength="strong", last_updated_fabula=6000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_BANQUO", target_entity_id="ENT_MACBETH",
            metrics={
                "affinity": RelationshipMetric(value=0.4, inertia=0.4, evidence_strength="strong", last_updated_fabula=10500),
                "fear":     RelationshipMetric(value=0.4, inertia=0.2, evidence_strength="moderate", last_updated_fabula=10500),
            },
        ),
        # Macbeth ↔ Macduff: enmity hardening into mortal opposition.
        RelationshipEdge(
            source_entity_id="ENT_MACBETH", target_entity_id="ENT_MACDUFF",
            metrics={
                "affinity": RelationshipMetric(value=-0.7, inertia=0.4, evidence_strength="strong", last_updated_fabula=13000),
                "fear":     RelationshipMetric(value=0.55, inertia=0.2, evidence_strength="strong", last_updated_fabula=13000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MACDUFF", target_entity_id="ENT_MACBETH",
            metrics={
                "affinity": RelationshipMetric(value=-1.0, inertia=0.55, evidence_strength="strong", last_updated_fabula=14500),
                "fear":     RelationshipMetric(value=0.2, inertia=0.2, evidence_strength="weak", last_updated_fabula=14500),
                "power_dynamic": RelationshipMetric(value=0.4, inertia=0.6, evidence_strength="strong", last_updated_fabula=19000),
            },
        ),
        # Macduff ↔ Malcolm: alliance forged in shared loss.
        RelationshipEdge(
            source_entity_id="ENT_MACDUFF", target_entity_id="ENT_MALCOLM",
            metrics={
                "affinity":      RelationshipMetric(value=0.8, inertia=0.45, evidence_strength="strong", last_updated_fabula=15000),
                "power_dynamic": RelationshipMetric(value=-0.3, inertia=0.6, evidence_strength="moderate", last_updated_fabula=15000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_MALCOLM", target_entity_id="ENT_MACDUFF",
            metrics={
                "affinity":      RelationshipMetric(value=0.75, inertia=0.45, evidence_strength="strong", last_updated_fabula=15000),
                "power_dynamic": RelationshipMetric(value=0.3, inertia=0.6, evidence_strength="moderate", last_updated_fabula=15000),
            },
        ),
        # Malcolm → Macbeth: hatred and fear.
        RelationshipEdge(
            source_entity_id="ENT_MALCOLM", target_entity_id="ENT_MACBETH",
            metrics={
                "affinity": RelationshipMetric(value=-0.9, inertia=0.5, evidence_strength="strong", last_updated_fabula=11000),
                "fear":     RelationshipMetric(value=0.6, inertia=0.2, evidence_strength="strong", last_updated_fabula=9000),
            },
        ),
        # Macbeth → Malcolm: views the named heir as the prophecy's living obstacle — to be eliminated, then feared once Malcolm musters the English army.
        RelationshipEdge(
            source_entity_id="ENT_MACBETH", target_entity_id="ENT_MALCOLM",
            metrics={
                "affinity": RelationshipMetric(value=-0.7, inertia=0.45, evidence_strength="strong", last_updated_fabula=11000),
                "fear":     RelationshipMetric(value=0.55, inertia=0.2, evidence_strength="strong", last_updated_fabula=15000),
                "power_dynamic": RelationshipMetric(value=0.4, inertia=0.6, evidence_strength="moderate", last_updated_fabula=11000),
            },
        ),
        # Lady Macbeth → Duncan: cold contempt during the planning.
        RelationshipEdge(
            source_entity_id="ENT_LADY_MACBETH", target_entity_id="ENT_DUNCAN",
            metrics={
                "affinity": RelationshipMetric(value=-0.9, inertia=0.4, evidence_strength="strong", last_updated_fabula=5000),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.7, evidence_strength="moderate", last_updated_fabula=5000),
            },
        ),
        # Duncan → Lady Macbeth: gracious royal hospitality ("fair and noble hostess") — entirely unaware of her contempt, no fear, mild positive affinity, royal-over-subject power.
        RelationshipEdge(
            source_entity_id="ENT_DUNCAN", target_entity_id="ENT_LADY_MACBETH",
            metrics={
                "affinity": RelationshipMetric(value=0.6, inertia=0.4, evidence_strength="moderate", last_updated_fabula=5000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.7, evidence_strength="moderate", last_updated_fabula=5000),
            },
        ),
        # Macbeth → Witches: drawn to them with mingled fear and need.
        RelationshipEdge(
            source_entity_id="ENT_MACBETH", target_entity_id="ENT_WITCHES",
            metrics={
                "affinity": RelationshipMetric(value=0.2, inertia=0.4, evidence_strength="moderate", last_updated_fabula=13000),
                "fear":     RelationshipMetric(value=0.5, inertia=0.25, evidence_strength="strong", last_updated_fabula=2000),
                "power_dynamic": RelationshipMetric(value=-0.6, inertia=0.7, evidence_strength="strong", last_updated_fabula=2000),
            },
        ),
        # Witches → Macbeth: he is their chosen plaything — manipulative attraction, no fear, supernatural ascendancy. Power flips sign relative to the above.
        RelationshipEdge(
            source_entity_id="ENT_WITCHES", target_entity_id="ENT_MACBETH",
            metrics={
                "affinity": RelationshipMetric(value=0.3, inertia=0.4, evidence_strength="moderate", last_updated_fabula=2000),
                "power_dynamic": RelationshipMetric(value=0.6, inertia=0.7, evidence_strength="strong", last_updated_fabula=2000),
            },
        ),
        # Macbeth → Fleance: only fear is observed (a remaining heir of the prophecy).
        RelationshipEdge(
            source_entity_id="ENT_MACBETH", target_entity_id="ENT_FLEANCE",
            metrics={
                "fear": RelationshipMetric(value=0.65, inertia=0.2, evidence_strength="strong", last_updated_fabula=11000),
            },
        ),
        # Fleance → Macbeth: a child whose father has just been murdered on Macbeth's order — high fear, hatred, no power.
        RelationshipEdge(
            source_entity_id="ENT_FLEANCE", target_entity_id="ENT_MACBETH",
            metrics={
                "affinity": RelationshipMetric(value=-0.9, inertia=0.5, evidence_strength="strong", last_updated_fabula=11000),
                "fear":     RelationshipMetric(value=0.9, inertia=0.2, evidence_strength="strong", last_updated_fabula=11000),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=11000),
            },
        ),
    ],

    # ── PROPOSITIONS ────────────────────────────────────────────────────
    # Hand-authored proposition registry. Each Concern.proposition_id
    # above resolves into one of these. ``synthesise_propositions`` may
    # add further auto-derived PROP_FROM_<EVT_> entries at runtime;
    # those are additive and do not collide with these named atoms.
    propositions=[
        Proposition(proposition_id="PROP_MACBETH_BECOMES_KING", kind="event_occurs",
                    referent_ids=["EVT_MACBETH_CROWNED", "ENT_MACBETH"],
                    description="Macbeth seizes the Scottish crown.",
                    audience_default_prior=0.4, stakes=0.95,
                    truth_at_fabula={10000: True}),
        Proposition(proposition_id="PROP_DUNCAN_DEAD", kind="event_occurs",
                    referent_ids=["EVT_DUNCAN_MURDER", "ENT_DUNCAN"],
                    description="King Duncan is dead.",
                    audience_default_prior=0.2, stakes=0.95,
                    truth_at_fabula={6000: True}),
        Proposition(proposition_id="PROP_BANQUO_LINE_KINGS", kind="outcome",
                    referent_ids=["ENT_BANQUO", "ENT_FLEANCE"],
                    description="Banquo's bloodline will inherit the throne.",
                    audience_default_prior=0.5, stakes=0.85,
                    # AUDIT P1-5: the witches' prophecy is not falsified
                    # by Macbeth's reign \u2014 Fleance escapes the
                    # assassination (fabula 12500) and the line lives to
                    # eventually inherit. Defer the commit to the end
                    # of the play where the prophecy is implicitly
                    # vindicated rather than asserting False at 1000.
                    truth_at_fabula={19500: True}),
        Proposition(proposition_id="PROP_FLEANCE_ALIVE", kind="trait_holds",
                    referent_ids=["ENT_FLEANCE"],
                    description="Fleance survives the assassins.",
                    audience_default_prior=0.5, stakes=0.7,
                    truth_at_fabula={11000: True}),
        Proposition(proposition_id="PROP_MACBETH_FOUL_PLAY", kind="trait_holds",
                    referent_ids=["ENT_MACBETH", "EVT_DUNCAN_MURDER"],
                    description="Macbeth played foul to win the crown.",
                    audience_default_prior=0.95, stakes=0.7,
                    truth_at_fabula={6000: True}),
        Proposition(proposition_id="PROP_MACDUFF_THREAT", kind="trait_holds",
                    referent_ids=["ENT_MACDUFF", "ENT_MACBETH"],
                    description="Macduff is the prophesied threat to Macbeth.",
                    audience_default_prior=0.6, stakes=0.9,
                    truth_at_fabula={13000: True}),
        Proposition(proposition_id="PROP_MACBETH_INVINCIBLE", kind="trait_holds",
                    referent_ids=["ENT_MACBETH"],
                    description="No man of woman born can harm Macbeth.",
                    audience_default_prior=0.3, stakes=0.85,
                    truth_at_fabula={19000: False}),  # Macduff's caesarean birth refutes
        Proposition(proposition_id="PROP_BIRNAM_NEVER_MOVES", kind="outcome",
                    referent_ids=["LOC_BIRNAM_WOOD", "LOC_DUNSINANE_CASTLE"],
                    description="Birnam Wood will never come to Dunsinane.",
                    audience_default_prior=0.2, stakes=0.85,
                    truth_at_fabula={18500: False}),
        Proposition(proposition_id="PROP_REGICIDE_DISCOVERED", kind="event_occurs",
                    referent_ids=["EVT_DUNCAN_DISCOVERED_MURDERED"],
                    description="The regicide is publicly discovered.",
                    audience_default_prior=0.85, stakes=0.7,
                    truth_at_fabula={7000: True}),
        Proposition(proposition_id="PROP_LADY_MACBETH_RESOLVE", kind="trait_holds",
                    referent_ids=["ENT_LADY_MACBETH"],
                    description="Lady Macbeth's resolve and sanity hold.",
                    audience_default_prior=0.6, stakes=0.6,
                    truth_at_fabula={16000: False, 17000: False}),
        Proposition(proposition_id="PROP_MARRIAGE_BOND", kind="relation_holds",
                    referent_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
                    description="The Macbeth marriage bond endures.",
                    audience_default_prior=0.85, stakes=0.55,
                    truth_at_fabula={17000: False}),
        Proposition(proposition_id="PROP_MACDUFF_FAMILY_SAFE", kind="outcome",
                    referent_ids=["ENT_LADY_MACDUFF", "ENT_MACDUFF"],
                    description="Macduff's wife and children remain alive at Fife.",
                    audience_default_prior=0.55, stakes=0.85,
                    truth_at_fabula={14000: False}),
        Proposition(proposition_id="PROP_MACDUFF_RETURNS", kind="event_occurs",
                    referent_ids=["EVT_MACDUFF_FLEES_TO_ENGLAND", "ENT_MACDUFF"],
                    description="Macduff returns to defend his family.",
                    audience_default_prior=0.4, stakes=0.7,
                    truth_at_fabula={14000: False}),
        Proposition(proposition_id="PROP_MACDUFF_AVENGED", kind="event_occurs",
                    referent_ids=["EVT_MACBETH_KILLED", "ENT_MACDUFF"],
                    description="Macduff personally kills Macbeth and avenges his family.",
                    audience_default_prior=0.6, stakes=0.95,
                    truth_at_fabula={19000: True}),
        Proposition(proposition_id="PROP_MACBETH_TYRANT", kind="trait_holds",
                    referent_ids=["ENT_MACBETH"],
                    description="Macbeth rules Scotland as a murdering tyrant.",
                    audience_default_prior=0.9, stakes=0.7,
                    truth_at_fabula={12000: True}),
        Proposition(proposition_id="PROP_MALCOLM_KILLED", kind="event_occurs",
                    referent_ids=["ENT_MALCOLM"],
                    description="Macbeth's agents assassinate Malcolm.",
                    audience_default_prior=0.2, stakes=0.85,
                    truth_at_fabula={20000: False}),
        Proposition(proposition_id="PROP_MALCOLM_THRONE", kind="event_occurs",
                    referent_ids=["EVT_MALCOLM_CROWNED", "ENT_MALCOLM"],
                    description="Malcolm is crowned King of Scotland.",
                    audience_default_prior=0.5, stakes=0.85,
                    truth_at_fabula={20000: True}),
        Proposition(proposition_id="PROP_THANE_LOYALTY", kind="trait_holds",
                    referent_ids=["ENT_MACBETH", "ENT_BANQUO", "ENT_MACDUFF"],
                    description="The thanes remain loyal to Duncan's house.",
                    audience_default_prior=0.85, stakes=0.65,
                    truth_at_fabula={6000: False}),
        # Audience-question reifications of the three WORLD_ traits (Pearl-Rung-2 cross-link).
        Proposition(proposition_id="PROP_FEUDAL_ORDER_INTACT", kind="trait_holds",
                    referent_ids=["WORLD_FEUDAL_HIERARCHY"],
                    description="Scotland's feudal compact \u2014 succession through anointed lineage rather than seizure \u2014 still binds.",
                    audience_default_prior=0.8, stakes=0.85,
                    truth_at_fabula={6000: False, 20000: True}),
        Proposition(proposition_id="PROP_PROPHECY_BINDING", kind="trait_holds",
                    referent_ids=["WORLD_SUPERNATURAL_PROPHECY"],
                    description="The witches' prophecy is fate, not metaphor: what they foretell will literally come to pass.",
                    audience_default_prior=0.55, stakes=0.9,
                    truth_at_fabula={18500: True}),
        Proposition(proposition_id="PROP_DIVINE_ORDER_AVENGES", kind="trait_holds",
                    referent_ids=["WORLD_DIVINE_RIGHT"],
                    description="The cosmos itself avenges regicide \u2014 the murder of an anointed king returns as madness, sleeplessness and ill omens upon its perpetrators.",
                    audience_default_prior=0.5, stakes=0.7,
                    truth_at_fabula={17000: True}),
    ],
)
