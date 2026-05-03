# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Lion, the Witch, and the Wardrobe — high-fidelity WorldStateV1 test fixture.

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
)

world_state = WorldStateV1(
    narrative_style=NarrativeStyle(
        format='synopsis',
        target_word_min=450,
        target_word_max=600,
        prose_density='sparse',
        voice='synoptic narration; no dialogue; condensed scene description; third-person POV; past tense',
        style_exemplar='Peter, Susan, Edmund, and Lucy Pevensie are four siblings sent to live in the country with the eccentric Professor Kirke during World War II. The children explore the house on a rainy day and Lucy, the youngest, finds an enormous wardrobe. Lucy steps inside and finds herself in a strange, snowy wood. Lucy encounters the Faun Tumnus, who is surprised to meet a human girl.',
        source_word_count=1280,
    ),
    # ── LOCATIONS ──────────────────────────────────────────────────────
    locations={
        "LOC_PROFESSOR_HOUSE": Location(
            name="Professor Kirke's Country House",
            description="Sprawling English manor where the Pevensie children are evacuated during WWII.",
            ambient_state={
                "safety": AmbientVector(value=0.75, volatility=0.2, evidence_strength="moderate"),
                "mystery": AmbientVector(value=0.5, volatility=0.3, evidence_strength="moderate"),
            },
        ),
        "LOC_WARDROBE": Location(
            name="The Wardrobe",
            description="Threshold between worlds; fur-coat lined passage to Narnia.",
            ambient_state={
                "supernatural": AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
                "concealment": AmbientVector(value=0.7, volatility=0.3, evidence_strength="moderate"),
            },
        ),
        "LOC_LANTERN_WASTE": Location(
            name="Lantern Waste (Narnia Woods)",
            description="Snowy forest at Narnia's border with the lamppost; Lucy's first entry point.",
            ambient_state={
                "cold": AmbientVector(value=0.85, volatility=0.3, evidence_strength="strong"),
                "wonder": AmbientVector(value=0.65, volatility=0.3, evidence_strength="moderate"),
            },
        ),
        "LOC_TUMNUS_CAVE": Location(
            name="Tumnus's Cave",
            description="Faun's cozy underground home with warm fire and bookshelf.",
            ambient_state={
                "safety": AmbientVector(value=0.6, volatility=0.5, evidence_strength="moderate"),
                "warmth": AmbientVector(value=0.75, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_BEAVERS_DAM": Location(
            name="Beavers' Dam",
            description="Mr. and Mrs. Beaver's home; refuge and staging point for the journey to Aslan.",
            ambient_state={
                "safety": AmbientVector(value=0.7, volatility=0.4, evidence_strength="moderate"),
                "warmth": AmbientVector(value=0.7, volatility=0.2, evidence_strength="moderate"),
            },
        ),
        "LOC_STONE_TABLE": Location(
            name="The Stone Table",
            description="Ancient sacrificial altar on Aslan's hill; site of the Deep Magic ritual.",
            ambient_state={
                "supernatural": AmbientVector(value=0.9, volatility=0.1, evidence_strength="strong"),
                "danger": AmbientVector(value=0.7, volatility=0.5, evidence_strength="strong"),
            },
        ),
        "LOC_WITCH_CASTLE": Location(
            name="White Witch's Castle",
            description="Frozen fortress where the Witch holds court and turns creatures to stone.",
            ambient_state={
                "danger": AmbientVector(value=0.95, volatility=0.2, evidence_strength="strong"),
                "cold": AmbientVector(value=0.95, volatility=0.1, evidence_strength="strong"),
            },
        ),
        "LOC_ASLAN_CAMP": Location(
            name="Aslan's Camp",
            description="Pavilion encampment near the Stone Table; Aslan's temporary headquarters.",
            ambient_state={
                "safety": AmbientVector(value=0.8, volatility=0.4, evidence_strength="strong"),
                "hope": AmbientVector(value=0.85, volatility=0.3, evidence_strength="strong"),
            },
        ),
        "LOC_CAIR_PARAVEL": Location(
            name="Cair Paravel",
            description="Royal castle by the Eastern Sea; seat of the four thrones.",
            ambient_state={
                "formality": AmbientVector(value=0.8, volatility=0.2, evidence_strength="moderate"),
                "joy": AmbientVector(value=0.85, volatility=0.2, evidence_strength="strong"),
            },
        ),
        "LOC_BATTLEFIELD": Location(
            name="Battlefield near the Stone Table",
            description="Open field where Aslan's forces clash with the Witch's army.",
            ambient_state={
                "danger": AmbientVector(value=0.9, volatility=0.3, evidence_strength="strong"),
                "chaos": AmbientVector(value=0.85, volatility=0.4, evidence_strength="strong"),
            },
        ),
    },

    # ── OBJECTS ─────────────────────────────────────────────────────────
    objects={
        "OBJ_WARDROBE_PORTAL": NarrativeObject(
            id="OBJ_WARDROBE_PORTAL", name="The Wardrobe Portal",
            location_id="LOC_PROFESSOR_HOUSE", owner_id=None,
            properties={"state": "active", "destination": "Narnia"},
            affordances=[
                Affordance(action="portal_traversal", target_type="Entity"),
            ],
        ),
        "OBJ_TURKISH_DELIGHT": NarrativeObject(
            id="OBJ_TURKISH_DELIGHT", name="Enchanted Turkish Delight",
            location_id="LOC_WITCH_CASTLE", owner_id="ENT_WHITE_WITCH",
            properties={"state": "enchanted", "effect": "induce_craving"},
            affordances=[
                Affordance(action="induce_craving", target_type="Entity"),
            ],
        ),
        "OBJ_MAGIC_HORN": NarrativeObject(
            id="OBJ_MAGIC_HORN", name="Susan's Magic Horn",
            location_id="LOC_ASLAN_CAMP", owner_id="ENT_SUSAN",
            properties={"gift_from": "Father Christmas", "state": "ready"},
            affordances=[
                Affordance(action="summon_aid", target_type="Entity"),
            ],
        ),
        "OBJ_PETER_SWORD": NarrativeObject(
            id="OBJ_PETER_SWORD", name="Peter's Sword",
            location_id="LOC_ASLAN_CAMP", owner_id="ENT_PETER",
            properties={"gift_from": "Father Christmas", "state": "ready"},
            affordances=[
                Affordance(action="kill", target_type="Entity"),
            ],
        ),
        "OBJ_LUCY_CORDIAL": NarrativeObject(
            id="OBJ_LUCY_CORDIAL", name="Lucy's Healing Cordial",
            location_id="LOC_ASLAN_CAMP", owner_id="ENT_LUCY",
            properties={"gift_from": "Father Christmas", "state": "ready"},
            affordances=[
                Affordance(action="heal", target_type="Entity"),
            ],
        ),
        "OBJ_SUSAN_BOW": NarrativeObject(
            id="OBJ_SUSAN_BOW", name="Susan's Bow and Arrows",
            location_id="LOC_ASLAN_CAMP", owner_id="ENT_SUSAN",
            properties={"gift_from": "Father Christmas", "state": "ready"},
            affordances=[
                Affordance(action="kill", target_type="Entity"),
            ],
        ),
    },

    # ── ENTITIES ────────────────────────────────────────────────────────
    entities={
        "ENT_LUCY": Entity(
            id="ENT_LUCY", name="Lucy Pevensie",
            location_id="LOC_PROFESSOR_HOUSE", status="healthy",
            traits={
                "courage": TraitVector(value=0.7, inertia=0.5, evidence_strength="strong"),
                "faith": TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                "curiosity": TraitVector(value=0.85, inertia=0.5, evidence_strength="strong"),
                "compassion": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "innocence": TraitVector(value=0.9, inertia=0.65, evidence_strength="strong"),
            },
            beliefs=[
                Belief(target_id="ENT_TUMNUS", perceived_state="Tumnus is my friend and I must help him",
                       confidence=0.9, inertia=0.6, evidence_strength="strong"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=2000, triggered_by="EVT_LUCY_ENTERS_NARNIA",
                    traits={
                        "faith": TraitVector(value=0.9, inertia=0.65, evidence_strength="strong"),
                    },
                    location_id="LOC_LANTERN_WASTE"),
                EntityStateSnapshot(fabula_time=18000, triggered_by="EVT_ASLAN_RESURRECTION",
                    traits={
                        "faith": TraitVector(value=0.95, inertia=0.7, evidence_strength="strong"),
                        "courage": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_EDMUND": Entity(
            id="ENT_EDMUND", name="Edmund Pevensie",
            location_id="LOC_PROFESSOR_HOUSE", status="healthy",
            traits={
                "greed": TraitVector(value=0.5, inertia=0.4, evidence_strength="moderate"),
                "spite": TraitVector(value=0.6, inertia=0.4, evidence_strength="moderate"),
                "courage": TraitVector(value=0.4, inertia=0.45, evidence_strength="moderate"),
                "guilt": TraitVector(value=0.2, inertia=0.25, evidence_strength="weak"),
                "faith": TraitVector(value=0.3, inertia=0.4, evidence_strength="weak"),
            },
            beliefs=[
                Belief(target_id="ENT_LUCY", perceived_state="Lucy is making up stories to get attention",
                       confidence=0.6, inertia=0.4, evidence_strength="moderate"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=4000, triggered_by="EVT_EDMUND_TURKISH_DELIGHT",
                    traits={
                        "greed": TraitVector(value=0.85, inertia=0.5, evidence_strength="strong"),
                    },
                    location_id="LOC_LANTERN_WASTE"),
                EntityStateSnapshot(fabula_time=10000, triggered_by="EVT_EDMUND_BETRAYS",
                    traits={
                        "guilt": TraitVector(value=0.65, inertia=0.35, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=14000, triggered_by="EVT_EDMUND_RESCUED",
                    traits={
                        "guilt": TraitVector(value=0.8, inertia=0.4, evidence_strength="strong"),
                        "faith": TraitVector(value=0.75, inertia=0.5, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_PETER": Entity(
            id="ENT_PETER", name="Peter Pevensie",
            location_id="LOC_PROFESSOR_HOUSE", status="healthy",
            traits={
                "courage": TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
                "leadership": TraitVector(value=0.65, inertia=0.5, evidence_strength="moderate"),
                "protectiveness": TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                "responsibility": TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_PETER_KILLS_WOLF",
                    traits={
                        "courage": TraitVector(value=0.9, inertia=0.7, evidence_strength="strong"),
                        "leadership": TraitVector(value=0.85, inertia=0.65, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_SUSAN": Entity(
            id="ENT_SUSAN", name="Susan Pevensie",
            location_id="LOC_PROFESSOR_HOUSE", status="healthy",
            traits={
                "caution": TraitVector(value=0.75, inertia=0.55, evidence_strength="moderate"),
                "courage": TraitVector(value=0.55, inertia=0.5, evidence_strength="moderate"),
                "compassion": TraitVector(value=0.75, inertia=0.55, evidence_strength="strong"),
                "grief": TraitVector(value=0.1, inertia=0.25, evidence_strength="weak"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=17000, triggered_by="EVT_ASLAN_DEATH",
                    traits={
                        "grief": TraitVector(value=0.9, inertia=0.45, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=18000, triggered_by="EVT_ASLAN_RESURRECTION",
                    traits={
                        "grief": TraitVector(value=0.15, inertia=0.3, evidence_strength="moderate"),
                        "courage": TraitVector(value=0.75, inertia=0.6, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_TUMNUS": Entity(
            id="ENT_TUMNUS", name="Mr. Tumnus the Faun",
            location_id="LOC_TUMNUS_CAVE", status="healthy",
            traits={
                "kindness": TraitVector(value=0.85, inertia=0.6, evidence_strength="strong"),
                "guilt": TraitVector(value=0.7, inertia=0.4, evidence_strength="strong"),
                "fear": TraitVector(value=0.65, inertia=0.35, evidence_strength="strong"),
                "courage": TraitVector(value=0.6, inertia=0.5, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_WHITE_WITCH", perceived_state="The Witch will punish me if I disobey her orders",
                       confidence=0.9, inertia=0.6, evidence_strength="strong"),
            ],
            state_timeline=[
                EntityStateSnapshot(fabula_time=3000, triggered_by="EVT_TUMNUS_SPARES_LUCY",
                    traits={
                        "courage": TraitVector(value=0.8, inertia=0.6, evidence_strength="strong"),
                        "fear": TraitVector(value=0.8, inertia=0.4, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=7500, triggered_by="EVT_TUMNUS_ARRESTED",
                    status="injured"),
                EntityStateSnapshot(fabula_time=14500, triggered_by="EVT_ASLAN_FREES_STATUES",
                    status="healthy", location_id="LOC_WITCH_CASTLE"),
            ],
        ),
        "ENT_WHITE_WITCH": Entity(
            id="ENT_WHITE_WITCH", name="White Witch (Jadis)",
            location_id="LOC_WITCH_CASTLE", status="healthy",
            traits={
                "cruelty": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "ambition": TraitVector(value=0.9, inertia=0.8, evidence_strength="strong"),
                "deception": TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
                "fear": TraitVector(value=0.3, inertia=0.4, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="WORLD_PROPHECY_FOUR_THRONES", perceived_state="Four humans threaten my reign and must be eliminated",
                       confidence=0.95, inertia=0.8, evidence_strength="strong"),
            ],
            constants=["magical_power"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=10000, triggered_by="EVT_EDMUND_BETRAYS",
                    traits={
                        "fear": TraitVector(value=0.65, inertia=0.45, evidence_strength="strong"),
                    },
                    beliefs_added=[
                        Belief(target_id="ENT_ASLAN", perceived_state="Aslan has returned and threatens my rule",
                               confidence=0.95, inertia=0.7, established_at_fabula=10000, evidence_strength="strong"),
                    ]),
                EntityStateSnapshot(fabula_time=20000, triggered_by="EVT_WITCH_KILLED",
                    status="dead"),
            ],
        ),
        "ENT_ASLAN": Entity(
            id="ENT_ASLAN", name="Aslan the Great Lion",
            location_id="LOC_ASLAN_CAMP", status="healthy",
            traits={
                "majesty": TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                "compassion": TraitVector(value=0.9, inertia=0.85, evidence_strength="strong"),
                "resolve": TraitVector(value=0.95, inertia=0.85, evidence_strength="strong"),
                "grief": TraitVector(value=0.1, inertia=0.3, evidence_strength="weak"),
            },
            beliefs=[],
            constants=["divine_nature", "resurrection_power"],
            state_timeline=[
                EntityStateSnapshot(fabula_time=16000, triggered_by="EVT_ASLAN_PACT",
                    traits={
                        "grief": TraitVector(value=0.85, inertia=0.45, evidence_strength="strong"),
                    }),
                EntityStateSnapshot(fabula_time=17000, triggered_by="EVT_ASLAN_DEATH",
                    status="dead"),
                EntityStateSnapshot(fabula_time=18000, triggered_by="EVT_ASLAN_RESURRECTION",
                    status="healthy",
                    traits={
                        "grief": TraitVector(value=0.1, inertia=0.3, evidence_strength="weak"),
                        "resolve": TraitVector(value=0.98, inertia=0.9, evidence_strength="strong"),
                    }),
            ],
        ),
        "ENT_MR_BEAVER": Entity(
            id="ENT_MR_BEAVER", name="Mr. Beaver",
            location_id="LOC_BEAVERS_DAM", status="healthy",
            traits={
                "loyalty": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "wisdom": TraitVector(value=0.7, inertia=0.6, evidence_strength="moderate"),
                "caution": TraitVector(value=0.7, inertia=0.55, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_ASLAN", perceived_state="Aslan is the true king of Narnia",
                       confidence=0.95, inertia=0.85, evidence_strength="strong"),
            ],
        ),
        "ENT_MRS_BEAVER": Entity(
            id="ENT_MRS_BEAVER", name="Mrs. Beaver",
            location_id="LOC_BEAVERS_DAM", status="healthy",
            traits={
                "kindness": TraitVector(value=0.8, inertia=0.6, evidence_strength="moderate"),
                "practicality": TraitVector(value=0.75, inertia=0.6, evidence_strength="moderate"),
            },
            beliefs=[],
        ),
        "ENT_FATHER_CHRISTMAS": Entity(
            id="ENT_FATHER_CHRISTMAS", name="Father Christmas",
            location_id="LOC_LANTERN_WASTE", status="healthy",
            traits={
                "generosity": TraitVector(value=0.95, inertia=0.8, evidence_strength="strong"),
                "joy": TraitVector(value=0.9, inertia=0.75, evidence_strength="strong"),
            },
            beliefs=[],
            constants=["seasonal_power"],
        ),
        "ENT_MAUGRIM": Entity(
            id="ENT_MAUGRIM", name="Maugrim (Captain of Witch's Secret Police)",
            location_id="LOC_WITCH_CASTLE", status="healthy",
            traits={
                "cruelty": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
                "loyalty": TraitVector(value=0.8, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[],
            state_timeline=[
                EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_PETER_KILLS_WOLF",
                    status="dead"),
            ],
        ),
        "ENT_PROFESSOR_KIRKE": Entity(
            id="ENT_PROFESSOR_KIRKE", name="Professor Digory Kirke",
            location_id="LOC_PROFESSOR_HOUSE", status="healthy",
            traits={
                "wisdom": TraitVector(value=0.85, inertia=0.75, evidence_strength="strong"),
                "eccentricity": TraitVector(value=0.7, inertia=0.65, evidence_strength="moderate"),
            },
            beliefs=[
                Belief(target_id="ENT_LUCY", perceived_state="Lucy is telling the truth about Narnia",
                       confidence=0.8, inertia=0.6, established_at_fabula=5500, evidence_strength="moderate"),
            ],
        ),
        "ENT_WITCH_DWARF": Entity(
            id="ENT_WITCH_DWARF", name="The Witch's Dwarf",
            location_id="LOC_WITCH_CASTLE", status="healthy",
            traits={
                "cruelty": TraitVector(value=0.75, inertia=0.65, evidence_strength="moderate"),
                "loyalty": TraitVector(value=0.85, inertia=0.7, evidence_strength="strong"),
            },
            beliefs=[],
        ),
    },

    # ── EVENTS ──────────────────────────────────────────────────────────
    events=[
        EventNode(id="EVT_PEVENSIES_ARRIVE", fabula_time=1000, syuzhet_index=1,
                  event_type="outcome", actor_ids=["ENT_LUCY", "ENT_EDMUND", "ENT_PETER", "ENT_SUSAN"], target_ids=[],
                  description="The four Pevensie children arrive at Professor Kirke's country house during the war."),
        EventNode(id="EVT_LUCY_ENTERS_NARNIA", fabula_time=2000, syuzhet_index=2,
                  event_type="choice", actor_ids=["ENT_LUCY"], target_ids=[],
                  description="Lucy discovers the wardrobe and steps through into Narnia for the first time."),
        EventNode(id="EVT_LUCY_MEETS_TUMNUS", fabula_time=2500, syuzhet_index=3,
                  event_type="outcome", actor_ids=["ENT_LUCY"], target_ids=["ENT_TUMNUS"],
                  description="Lucy encounters Mr. Tumnus the Faun at the lamppost in the snowy woods."),
        EventNode(id="EVT_TUMNUS_SPARES_LUCY", fabula_time=3000, syuzhet_index=5,
                  event_type="choice", actor_ids=["ENT_TUMNUS"], target_ids=["ENT_LUCY"],
                  description="Tumnus confesses he was supposed to capture Lucy for the Witch, but chooses to let her go free."),
        EventNode(id="EVT_EDMUND_ENTERS_NARNIA", fabula_time=3500, syuzhet_index=6,
                  event_type="choice", actor_ids=["ENT_EDMUND"], target_ids=[],
                  description="Edmund follows Lucy into the wardrobe and enters Narnia alone."),
        EventNode(id="EVT_EDMUND_TURKISH_DELIGHT", fabula_time=4000, syuzhet_index=7,
                  event_type="choice", actor_ids=["ENT_WHITE_WITCH"], target_ids=["ENT_EDMUND"],
                  description="The White Witch feeds Edmund enchanted Turkish Delight, creating an insatiable craving."),
        EventNode(id="EVT_EDMUND_DENIES_NARNIA", fabula_time=5000, syuzhet_index=9,
                  event_type="choice", actor_ids=["ENT_EDMUND"], target_ids=["ENT_LUCY"],
                  description="Edmund spitefully denies Narnia's existence to Peter and Susan, betraying Lucy's trust."),
        EventNode(id="EVT_PROFESSOR_DEFENDS_LUCY", fabula_time=5500, syuzhet_index=10,
                  event_type="revelation", actor_ids=["ENT_PROFESSOR_KIRKE"], target_ids=[],
                  description="Professor Kirke argues logically that Lucy is likely telling the truth about Narnia."),
        EventNode(id="EVT_ALL_ENTER_NARNIA", fabula_time=6000, syuzhet_index=11,
                  event_type="outcome", actor_ids=["ENT_LUCY", "ENT_EDMUND", "ENT_PETER", "ENT_SUSAN"], target_ids=[],
                  description="All four children hide in the wardrobe and find themselves together in Narnia."),
        EventNode(id="EVT_TUMNUS_ARRESTED", fabula_time=7500, syuzhet_index=12,
                  event_type="outcome", actor_ids=["ENT_WHITE_WITCH"], target_ids=["ENT_TUMNUS"],
                  description="The children discover Tumnus has been arrested for treason; his cave ransacked."),
        EventNode(id="EVT_MEET_BEAVERS", fabula_time=8000, syuzhet_index=13,
                  event_type="outcome", actor_ids=["ENT_LUCY", "ENT_PETER", "ENT_SUSAN"], target_ids=["ENT_MR_BEAVER"],
                  description="Guided by a robin, the children meet Mr. Beaver, who offers to help them."),
        EventNode(id="EVT_BEAVERS_EXPLAIN_ASLAN", fabula_time=9000, syuzhet_index=15,
                  event_type="revelation", actor_ids=["ENT_MR_BEAVER"], target_ids=[],
                  description="At the Beavers' dam, Mr. Beaver explains Aslan's return and the prophecy of four thrones."),
        EventNode(id="EVT_EDMUND_BETRAYS", fabula_time=10000, syuzhet_index=16,
                  event_type="choice", actor_ids=["ENT_EDMUND"], target_ids=["ENT_WHITE_WITCH"],
                  description="Edmund slips away from the Beavers' dam to warn the White Witch about Aslan."),
        EventNode(id="EVT_JOURNEY_TO_STONE_TABLE", fabula_time=11000, syuzhet_index=18,
                  event_type="outcome", actor_ids=["ENT_LUCY", "ENT_PETER", "ENT_SUSAN", "ENT_MR_BEAVER", "ENT_MRS_BEAVER"], target_ids=[],
                  description="The remaining children and Beavers flee through the melting snow toward the Stone Table."),
        EventNode(id="EVT_FATHER_CHRISTMAS_GIFTS", fabula_time=11500, syuzhet_index=19,
                  event_type="outcome", actor_ids=["ENT_FATHER_CHRISTMAS"], target_ids=["ENT_LUCY", "ENT_PETER", "ENT_SUSAN"],
                  description="Father Christmas appears and gives magical gifts: Peter's sword, Susan's horn and bow, Lucy's cordial."),
        EventNode(id="EVT_SPRING_THAW", fabula_time=12000, syuzhet_index=20,
                  event_type="outcome", actor_ids=[], target_ids=[],
                  description="The enchanted winter breaks; snow melts and spring returns to Narnia."),
        EventNode(id="EVT_MEET_ASLAN", fabula_time=12500, syuzhet_index=21,
                  event_type="outcome", actor_ids=["ENT_LUCY", "ENT_PETER", "ENT_SUSAN"], target_ids=["ENT_ASLAN"],
                  description="The three children reach Aslan's camp and meet the Great Lion for the first time."),
        EventNode(id="EVT_PETER_KILLS_WOLF", fabula_time=13000, syuzhet_index=22,
                  event_type="choice", actor_ids=["ENT_PETER"], target_ids=["ENT_MAUGRIM"],
                  description="Peter kills Maugrim the wolf with his sword, rescuing Susan from attack."),
        EventNode(id="EVT_EDMUND_RESCUED", fabula_time=14000, syuzhet_index=23,
                  event_type="outcome", actor_ids=["ENT_ASLAN"], target_ids=["ENT_EDMUND"],
                  description="Aslan's forces rescue Edmund from the Witch just as she prepares to kill him."),
        EventNode(id="EVT_ASLAN_FREES_STATUES", fabula_time=14500, syuzhet_index=24,
                  event_type="outcome", actor_ids=["ENT_ASLAN"], target_ids=["ENT_TUMNUS"],
                  description="Aslan breathes on the stone statues in the Witch's castle, restoring them to life."),
        EventNode(id="EVT_WITCH_DEMANDS_EDMUND", fabula_time=15000, syuzhet_index=25,
                  event_type="choice", actor_ids=["ENT_WHITE_WITCH"], target_ids=["ENT_EDMUND"],
                  description="The Witch confronts Aslan, invoking the Deep Magic that grants her Edmund's life as a traitor."),
        EventNode(id="EVT_ASLAN_PACT", fabula_time=16000, syuzhet_index=27,
                  event_type="choice", actor_ids=["ENT_ASLAN"], target_ids=["ENT_WHITE_WITCH"],
                  description="Aslan privately agrees to a secret compromise with the Witch; he grows sad and withdrawn."),
        EventNode(id="EVT_ASLAN_DEATH", fabula_time=17000, syuzhet_index=29,
                  event_type="outcome", actor_ids=["ENT_WHITE_WITCH"], target_ids=["ENT_ASLAN"],
                  description="The Witch and her followers kill Aslan at the Stone Table; Susan and Lucy witness in hiding."),
        EventNode(id="EVT_ASLAN_RESURRECTION", fabula_time=18000, syuzhet_index=30,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_ASLAN"],
                  description="Aslan rises from death by the Deeper Magic; the Stone Table cracks and he returns triumphant."),
        EventNode(id="EVT_BATTLE", fabula_time=19000, syuzhet_index=31,
                  event_type="outcome", actor_ids=["ENT_ASLAN", "ENT_PETER", "ENT_EDMUND"], target_ids=[],
                  description="Aslan leads his army to join Peter and Edmund's forces in the great battle."),
        EventNode(id="EVT_WITCH_KILLED", fabula_time=20000, syuzhet_index=32,
                  event_type="outcome", actor_ids=["ENT_ASLAN"], target_ids=["ENT_WHITE_WITCH"],
                  description="Aslan kills the White Witch in battle, ending her reign over Narnia."),
        EventNode(id="EVT_CORONATION", fabula_time=21000, syuzhet_index=33,
                  event_type="outcome", actor_ids=[], target_ids=["ENT_LUCY", "ENT_EDMUND", "ENT_PETER", "ENT_SUSAN"],
                  description="The four Pevensie children are crowned as Kings and Queens of Narnia at Cair Paravel."),

        # ── UTTERANCES ──────────────────────────────────────────────────
        EventNode(id="EVT_UTT_TUMNUS_CONFESSION", event_type="utterance",
                  description="Tumnus confesses to Lucy that he is a servant of the White Witch who was supposed to capture her.",
                  speaker_id="ENT_TUMNUS", addressee_ids=["ENT_LUCY"],
                  actor_ids=["ENT_TUMNUS"], target_ids=["EVT_TUMNUS_SPARES_LUCY"],
                  content="I am a kidnapper for the White Witch, and I was supposed to betray you — but I cannot do it.",
                  via_channel_id=None, truth_value="true",
                  fabula_time=2800, syuzhet_index=4),
        EventNode(id="EVT_UTT_WITCH_TEMPTS_EDMUND", event_type="utterance",
                  description="The White Witch persuades Edmund to bring his siblings to her, promising Turkish Delight and power.",
                  speaker_id="ENT_WHITE_WITCH", addressee_ids=["ENT_EDMUND"],
                  actor_ids=["ENT_WHITE_WITCH"], target_ids=["EVT_EDMUND_TURKISH_DELIGHT"],
                  content="Bring your brother and sisters to me, and I will make you a prince and give you all the Turkish Delight you desire.",
                  via_channel_id=None, truth_value="false",
                  fabula_time=4000, syuzhet_index=8),
        EventNode(id="EVT_UTT_BEAVERS_PROPHECY", event_type="utterance",
                  description="Mr. Beaver tells the children about the prophecy that four humans will sit on the thrones at Cair Paravel.",
                  speaker_id="ENT_MR_BEAVER", addressee_ids=["ENT_LUCY", "ENT_PETER", "ENT_SUSAN"],
                  actor_ids=["ENT_MR_BEAVER"], target_ids=["EVT_BEAVERS_EXPLAIN_ASLAN"],
                  content="When Adam's flesh and Adam's bone sits at Cair Paravel in throne, the evil time will be over and done.",
                  via_channel_id="CHN_BEAVER_FIRESIDE", truth_value="true",
                  fabula_time=9000, syuzhet_index=14),
        EventNode(id="EVT_UTT_EDMUND_WARNS_WITCH", event_type="utterance",
                  description="Edmund betrays his siblings by reporting to the Witch that Aslan has returned and the children plan to meet him.",
                  speaker_id="ENT_EDMUND", addressee_ids=["ENT_WHITE_WITCH"],
                  actor_ids=["ENT_EDMUND"], target_ids=["EVT_EDMUND_BETRAYS"],
                  content="Aslan is at the Stone Table, and my brother and sisters are going there with the Beavers.",
                  via_channel_id=None, truth_value="true",
                  fabula_time=10000, syuzhet_index=17),
        EventNode(id="EVT_UTT_WITCH_CLAIMS_EDMUND", event_type="utterance",
                  description="The White Witch confronts Aslan, invoking the Deep Magic that entitles her to Edmund's blood.",
                  speaker_id="ENT_WHITE_WITCH", addressee_ids=["ENT_ASLAN"],
                  actor_ids=["ENT_WHITE_WITCH"], target_ids=["EVT_WITCH_DEMANDS_EDMUND"],
                  content="The boy Edmund is a traitor, and by the Deep Magic carved on the Stone Table, every traitor belongs to me as my lawful prey.",
                  via_channel_id=None, truth_value="performative",
                  fabula_time=15000, syuzhet_index=26),
        EventNode(id="EVT_UTT_ASLAN_SECRET_PACT", event_type="utterance",
                  description="Aslan and the Witch speak privately; he agrees to substitute himself for Edmund under the Deep Magic.",
                  speaker_id="ENT_ASLAN", addressee_ids=["ENT_WHITE_WITCH"],
                  actor_ids=["ENT_ASLAN"], target_ids=["EVT_ASLAN_PACT"],
                  content="I will take the traitor's place; release Edmund and take my life instead.",
                  via_channel_id=None, truth_value="performative",
                  fabula_time=16000, syuzhet_index=28),
    ],

    # ── CAUSAL TOPOLOGY ─────────────────────────────────────────────────
    causal_topology=[
        # ── chain_reaction (Event → Event) ──
        CausalEdge(source_id="EVT_PEVENSIES_ARRIVE", target_id="EVT_LUCY_ENTERS_NARNIA",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=6.0, fabula_time=1000, propagation_delay=1000),
        CausalEdge(source_id="EVT_LUCY_ENTERS_NARNIA", target_id="EVT_LUCY_MEETS_TUMNUS",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2000, propagation_delay=500),
        CausalEdge(source_id="EVT_LUCY_MEETS_TUMNUS", target_id="EVT_TUMNUS_SPARES_LUCY",
                   causality_type="chain_reaction", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=2500, propagation_delay=500),
        CausalEdge(source_id="EVT_TUMNUS_SPARES_LUCY", target_id="EVT_TUMNUS_ARRESTED",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=3000, propagation_delay=4500),
        CausalEdge(source_id="EVT_LUCY_ENTERS_NARNIA", target_id="EVT_EDMUND_ENTERS_NARNIA",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=2000, propagation_delay=1500),
        CausalEdge(source_id="EVT_EDMUND_ENTERS_NARNIA", target_id="EVT_EDMUND_TURKISH_DELIGHT",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3500, propagation_delay=500),
        CausalEdge(source_id="EVT_EDMUND_TURKISH_DELIGHT", target_id="EVT_EDMUND_DENIES_NARNIA",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=4000, propagation_delay=1000),
        CausalEdge(source_id="EVT_EDMUND_DENIES_NARNIA", target_id="EVT_PROFESSOR_DEFENDS_LUCY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=5000, propagation_delay=500),
        CausalEdge(source_id="EVT_ALL_ENTER_NARNIA", target_id="EVT_TUMNUS_ARRESTED",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=6.0, fabula_time=6000, propagation_delay=1500),
        CausalEdge(source_id="EVT_TUMNUS_ARRESTED", target_id="EVT_MEET_BEAVERS",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=7500, propagation_delay=500),
        CausalEdge(source_id="EVT_MEET_BEAVERS", target_id="EVT_BEAVERS_EXPLAIN_ASLAN",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="strong",
                   causal_force=7.0, fabula_time=8000, propagation_delay=1000),
        CausalEdge(source_id="EVT_BEAVERS_EXPLAIN_ASLAN", target_id="EVT_EDMUND_BETRAYS",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=9000, propagation_delay=1000),
        CausalEdge(source_id="EVT_EDMUND_BETRAYS", target_id="EVT_JOURNEY_TO_STONE_TABLE",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=10000, propagation_delay=1000),
        CausalEdge(source_id="EVT_JOURNEY_TO_STONE_TABLE", target_id="EVT_FATHER_CHRISTMAS_GIFTS",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=11000, propagation_delay=500),
        CausalEdge(source_id="EVT_FATHER_CHRISTMAS_GIFTS", target_id="EVT_SPRING_THAW",
                   causality_type="chain_reaction", mechanism="supernatural", evidence_strength="strong",
                   causal_force=7.0, fabula_time=11500, propagation_delay=500),
        CausalEdge(source_id="EVT_JOURNEY_TO_STONE_TABLE", target_id="EVT_MEET_ASLAN",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=11000, propagation_delay=1500),
        CausalEdge(source_id="EVT_MEET_ASLAN", target_id="EVT_PETER_KILLS_WOLF",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=12500, propagation_delay=500),
        CausalEdge(source_id="EVT_PETER_KILLS_WOLF", target_id="EVT_EDMUND_RESCUED",
                   causality_type="chain_reaction", mechanism="informational", evidence_strength="moderate",
                   causal_force=6.0, fabula_time=13000, propagation_delay=1000),
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="EVT_WITCH_DEMANDS_EDMUND",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=14000, propagation_delay=1000),
        CausalEdge(source_id="EVT_WITCH_DEMANDS_EDMUND", target_id="EVT_ASLAN_PACT",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=15000, propagation_delay=1000),
        CausalEdge(source_id="EVT_ASLAN_PACT", target_id="EVT_ASLAN_DEATH",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=10.0, fabula_time=16000, propagation_delay=1000),
        CausalEdge(source_id="EVT_ASLAN_DEATH", target_id="EVT_ASLAN_RESURRECTION",
                   causality_type="chain_reaction", mechanism="supernatural", evidence_strength="strong",
                   causal_force=10.0, fabula_time=17000, propagation_delay=1000),
        CausalEdge(source_id="EVT_ASLAN_RESURRECTION", target_id="EVT_BATTLE",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=18000, propagation_delay=1000),
        CausalEdge(source_id="EVT_BATTLE", target_id="EVT_WITCH_KILLED",
                   causality_type="chain_reaction", mechanism="physical", evidence_strength="strong",
                   causal_force=9.0, fabula_time=19000, propagation_delay=1000),
        CausalEdge(source_id="EVT_WITCH_KILLED", target_id="EVT_CORONATION",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=8.0, fabula_time=20000, propagation_delay=1000),

        # ── mutation (Event → Entity trait/status) ──
        CausalEdge(source_id="EVT_LUCY_ENTERS_NARNIA", target_id="ENT_LUCY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=2000,
                   trait_target="faith", trait_delta=0.3),
        CausalEdge(source_id="EVT_EDMUND_TURKISH_DELIGHT", target_id="ENT_EDMUND",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=4000,
                   trait_target="greed", trait_delta=0.6),
        CausalEdge(source_id="EVT_TUMNUS_SPARES_LUCY", target_id="ENT_TUMNUS",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=7.0, fabula_time=3000,
                   trait_target="courage", trait_delta=0.4),
        CausalEdge(source_id="EVT_TUMNUS_SPARES_LUCY", target_id="ENT_TUMNUS",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=3000,
                   trait_target="fear", trait_delta=0.3),
        CausalEdge(source_id="EVT_EDMUND_BETRAYS", target_id="ENT_EDMUND",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=10000,
                   trait_target="guilt", trait_delta=0.5),
        CausalEdge(source_id="EVT_BEAVERS_EXPLAIN_ASLAN", target_id="ENT_WHITE_WITCH",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=9000,
                   trait_target="fear", trait_delta=0.5),
        CausalEdge(source_id="EVT_PETER_KILLS_WOLF", target_id="ENT_PETER",
                   causality_type="mutation", mechanism="physical", evidence_strength="strong",
                   causal_force=7.0, fabula_time=13000,
                   trait_target="courage", trait_delta=0.4),
        CausalEdge(source_id="EVT_PETER_KILLS_WOLF", target_id="ENT_PETER",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=6.0, fabula_time=13000,
                   trait_target="leadership", trait_delta=0.4),
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="ENT_EDMUND",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=14000,
                   trait_target="guilt", trait_delta=0.3),
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="ENT_EDMUND",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=14000,
                   trait_target="faith", trait_delta=0.5),
        CausalEdge(source_id="EVT_ASLAN_PACT", target_id="ENT_ASLAN",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=16000,
                   trait_target="grief", trait_delta=0.75),
        CausalEdge(source_id="EVT_ASLAN_DEATH", target_id="ENT_SUSAN",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=9.0, fabula_time=17000,
                   trait_target="grief", trait_delta=0.8),
        CausalEdge(source_id="EVT_ASLAN_DEATH", target_id="ENT_LUCY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=9.0, fabula_time=17000,
                   trait_target="grief", trait_delta=0.7),
        CausalEdge(source_id="EVT_ASLAN_RESURRECTION", target_id="ENT_LUCY",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=9.0, fabula_time=18000,
                   trait_target="faith", trait_delta=0.4),
        CausalEdge(source_id="EVT_ASLAN_RESURRECTION", target_id="ENT_LUCY",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=18000,
                   trait_target="courage", trait_delta=0.3),
        CausalEdge(source_id="EVT_ASLAN_RESURRECTION", target_id="ENT_SUSAN",
                   causality_type="mutation", mechanism="emotional", evidence_strength="strong",
                   causal_force=8.0, fabula_time=18000,
                   trait_target="grief", trait_delta=-0.75),
        CausalEdge(source_id="EVT_ASLAN_RESURRECTION", target_id="ENT_SUSAN",
                   causality_type="mutation", mechanism="psychological", evidence_strength="strong",
                   causal_force=7.0, fabula_time=18000,
                   trait_target="courage", trait_delta=0.4),

        # ── mutation_social (Event → Relationship) — PER-AXIS COVERAGE ──
        # Lucy ↔ Tumnus: affinity
        CausalEdge(source_id="EVT_LUCY_MEETS_TUMNUS", target_id="ENT_LUCY", rel_counterpart_id="ENT_TUMNUS",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.7,
                   mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=2500, propagation_delay=0),
        CausalEdge(source_id="EVT_TUMNUS_SPARES_LUCY", target_id="ENT_LUCY", rel_counterpart_id="ENT_TUMNUS",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.3,
                   mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=3000, propagation_delay=0),
        CausalEdge(source_id="EVT_LUCY_MEETS_TUMNUS", target_id="ENT_TUMNUS", rel_counterpart_id="ENT_LUCY",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.6,
                   mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=2500, propagation_delay=0),
        # Edmund ↔ White Witch: affinity, fear, power_dynamic
        CausalEdge(source_id="EVT_EDMUND_TURKISH_DELIGHT", target_id="ENT_EDMUND", rel_counterpart_id="ENT_WHITE_WITCH",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.5,
                   mechanism="psychological", evidence_strength="moderate", causal_force=6.0, fabula_time=4000, propagation_delay=0),
        CausalEdge(source_id="EVT_EDMUND_BETRAYS", target_id="ENT_EDMUND", rel_counterpart_id="ENT_WHITE_WITCH",
                   causality_type="mutation_social", trait_target="fear", trait_delta=0.6,
                   mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=10000, propagation_delay=0),
        CausalEdge(source_id="EVT_EDMUND_TURKISH_DELIGHT", target_id="ENT_EDMUND", rel_counterpart_id="ENT_WHITE_WITCH",
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.7,
                   mechanism="psychological", evidence_strength="strong", causal_force=7.0, fabula_time=4000, propagation_delay=0),
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="ENT_EDMUND", rel_counterpart_id="ENT_WHITE_WITCH",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.8,
                   mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=14000, propagation_delay=0),
        # Edmund ↔ Lucy: affinity
        CausalEdge(source_id="EVT_EDMUND_DENIES_NARNIA", target_id="ENT_LUCY", rel_counterpart_id="ENT_EDMUND",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.6,
                   mechanism="betrayal", evidence_strength="strong", causal_force=7.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="ENT_LUCY", rel_counterpart_id="ENT_EDMUND",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.5,
                   mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=14000, propagation_delay=0),
        # Peter ↔ Edmund: affinity
        CausalEdge(source_id="EVT_EDMUND_DENIES_NARNIA", target_id="ENT_PETER", rel_counterpart_id="ENT_EDMUND",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=-0.4,
                   mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="ENT_PETER", rel_counterpart_id="ENT_EDMUND",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.6,
                   mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=14000, propagation_delay=0),
        # Susan ↔ Edmund: affinity
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="ENT_SUSAN", rel_counterpart_id="ENT_EDMUND",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.5,
                   mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=14000, propagation_delay=0),
        # Peter ↔ Lucy: affinity
        CausalEdge(source_id="EVT_PROFESSOR_DEFENDS_LUCY", target_id="ENT_PETER", rel_counterpart_id="ENT_LUCY",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.3,
                   mechanism="psychological", evidence_strength="moderate", causal_force=5.0, fabula_time=5500, propagation_delay=0),
        # Susan ↔ Lucy: affinity
        CausalEdge(source_id="EVT_PROFESSOR_DEFENDS_LUCY", target_id="ENT_SUSAN", rel_counterpart_id="ENT_LUCY",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.2,
                   mechanism="psychological", evidence_strength="weak", causal_force=4.0, fabula_time=5500, propagation_delay=0),
        # Peter ↔ Susan: affinity
        CausalEdge(source_id="EVT_PETER_KILLS_WOLF", target_id="ENT_SUSAN", rel_counterpart_id="ENT_PETER",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.5,
                   mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=13000, propagation_delay=0),
        # Lucy ↔ Aslan: affinity, power_dynamic
        CausalEdge(source_id="EVT_MEET_ASLAN", target_id="ENT_LUCY", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.8,
                   mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=12500, propagation_delay=0),
        CausalEdge(source_id="EVT_MEET_ASLAN", target_id="ENT_LUCY", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.7,
                   mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=12500, propagation_delay=0),
        CausalEdge(source_id="EVT_ASLAN_DEATH", target_id="ENT_LUCY", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.2,
                   mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=17000, propagation_delay=0),
        # Peter ↔ Aslan: affinity, power_dynamic
        CausalEdge(source_id="EVT_MEET_ASLAN", target_id="ENT_PETER", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.7,
                   mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=12500, propagation_delay=0),
        CausalEdge(source_id="EVT_MEET_ASLAN", target_id="ENT_PETER", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.6,
                   mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=12500, propagation_delay=0),
        # Susan ↔ Aslan: affinity, power_dynamic
        CausalEdge(source_id="EVT_MEET_ASLAN", target_id="ENT_SUSAN", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.7,
                   mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=12500, propagation_delay=0),
        CausalEdge(source_id="EVT_MEET_ASLAN", target_id="ENT_SUSAN", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.6,
                   mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=12500, propagation_delay=0),
        CausalEdge(source_id="EVT_ASLAN_DEATH", target_id="ENT_SUSAN", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.2,
                   mechanism="emotional", evidence_strength="strong", causal_force=6.0, fabula_time=17000, propagation_delay=0),
        # Edmund ↔ Aslan: affinity, power_dynamic
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="ENT_EDMUND", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="affinity", trait_delta=0.8,
                   mechanism="emotional", evidence_strength="strong", causal_force=8.0, fabula_time=14000, propagation_delay=0),
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="ENT_EDMUND", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=-0.7,
                   mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=14000, propagation_delay=0),
        # Aslan ↔ White Witch: fear, power_dynamic
        CausalEdge(source_id="EVT_BEAVERS_EXPLAIN_ASLAN", target_id="ENT_WHITE_WITCH", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="fear", trait_delta=0.7,
                   mechanism="psychological", evidence_strength="strong", causal_force=8.0, fabula_time=9000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCH_DEMANDS_EDMUND", target_id="ENT_WHITE_WITCH", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="power_dynamic", trait_delta=0.5,
                   mechanism="social", evidence_strength="strong", causal_force=7.0, fabula_time=15000, propagation_delay=0),
        CausalEdge(source_id="EVT_ASLAN_RESURRECTION", target_id="ENT_WHITE_WITCH", rel_counterpart_id="ENT_ASLAN",
                   causality_type="mutation_social", trait_target="fear", trait_delta=0.3,
                   mechanism="psychological", evidence_strength="strong", causal_force=8.0, fabula_time=18000, propagation_delay=0),

        # ── affordance_gate (State → Event) ──
        CausalEdge(source_id="OBJ_WARDROBE_PORTAL", target_id="EVT_LUCY_ENTERS_NARNIA",
                   causality_type="affordance_gate", mechanism="supernatural", evidence_strength="strong",
                   causal_force=9.0, fabula_time=2000),
        CausalEdge(source_id="OBJ_WARDROBE_PORTAL", target_id="EVT_EDMUND_ENTERS_NARNIA",
                   causality_type="affordance_gate", mechanism="supernatural", evidence_strength="strong",
                   causal_force=9.0, fabula_time=3500),
        CausalEdge(source_id="OBJ_WARDROBE_PORTAL", target_id="EVT_ALL_ENTER_NARNIA",
                   causality_type="affordance_gate", mechanism="supernatural", evidence_strength="strong",
                   causal_force=9.0, fabula_time=6000),
        CausalEdge(source_id="OBJ_TURKISH_DELIGHT", target_id="EVT_EDMUND_BETRAYS",
                   causality_type="affordance_gate", mechanism="psychological", evidence_strength="strong",
                   causal_force=8.0, fabula_time=10000),
        CausalEdge(source_id="OBJ_PETER_SWORD", target_id="EVT_PETER_KILLS_WOLF",
                   causality_type="affordance_gate", mechanism="physical", evidence_strength="strong",
                   causal_force=8.0, fabula_time=13000),
        CausalEdge(source_id="OBJ_MAGIC_HORN", target_id="EVT_PETER_KILLS_WOLF",
                   causality_type="affordance_gate", mechanism="supernatural", evidence_strength="strong",
                   causal_force=7.0, fabula_time=13000),
        CausalEdge(source_id="ENT_ASLAN", target_id="EVT_ASLAN_RESURRECTION",
                   causality_type="affordance_gate", mechanism="supernatural", evidence_strength="strong",
                   causal_force=10.0, fabula_time=18000),

        # ── ambient_propagation (State → State) ──
        CausalEdge(source_id="LOC_WITCH_CASTLE", target_id="ENT_EDMUND",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="weak",
                   causal_force=3.5, fabula_time=10000),
        CausalEdge(source_id="LOC_ASLAN_CAMP", target_id="ENT_LUCY",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="weak",
                   causal_force=3.0, fabula_time=12500),
        CausalEdge(source_id="LOC_STONE_TABLE", target_id="ENT_ASLAN",
                   causality_type="ambient_propagation", mechanism="psychological", evidence_strength="moderate",
                   causal_force=4.0, fabula_time=17000),

        # ── WORLD_ → Event (named-latent common-cause wiring) ──
        CausalEdge(source_id="WORLD_DEEP_MAGIC", target_id="EVT_WITCH_DEMANDS_EDMUND",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=8.0, fabula_time=15000),
        CausalEdge(source_id="WORLD_DEEP_MAGIC", target_id="EVT_ASLAN_PACT",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=9.0, fabula_time=16000),
        CausalEdge(source_id="WORLD_DEEP_MAGIC", target_id="EVT_ASLAN_DEATH",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=9.0, fabula_time=17000),
        CausalEdge(source_id="WORLD_DEEPER_MAGIC", target_id="EVT_ASLAN_RESURRECTION",
                   causality_type="chain_reaction", mechanism="supernatural", evidence_strength="strong",
                   causal_force=10.0, fabula_time=18000),
        CausalEdge(source_id="WORLD_PROPHECY_FOUR_THRONES", target_id="EVT_BEAVERS_EXPLAIN_ASLAN",
                   causality_type="chain_reaction", mechanism="epistemic", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=9000),
        CausalEdge(source_id="WORLD_PROPHECY_FOUR_THRONES", target_id="EVT_CORONATION",
                   causality_type="chain_reaction", mechanism="performative", evidence_strength="strong",
                   causal_force=8.0, fabula_time=21000),
        CausalEdge(source_id="WORLD_ENCHANTED_WINTER", target_id="EVT_SPRING_THAW",
                   causality_type="chain_reaction", mechanism="supernatural", evidence_strength="strong",
                   causal_force=7.0, fabula_time=12000),
        CausalEdge(source_id="WORLD_ENCHANTED_WINTER", target_id="EVT_FATHER_CHRISTMAS_GIFTS",
                   causality_type="chain_reaction", mechanism="supernatural", evidence_strength="moderate",
                   causal_force=5.0, fabula_time=11500),

        # ─── auto-patched mutation_social edges (per-axis coverage) ───
        CausalEdge(source_id="EVT_PEVENSIES_ARRIVE", target_id="ENT_EDMUND", rel_counterpart_id="ENT_LUCY", causality_type="mutation_social", trait_target="affinity", trait_delta=0.5, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_EDMUND_DENIES_NARNIA", target_id="ENT_EDMUND", rel_counterpart_id="ENT_LUCY", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.4, mechanism="psychological", evidence_strength="strong", causal_force=6.0, fabula_time=5000, propagation_delay=0),
        CausalEdge(source_id="EVT_PEVENSIES_ARRIVE", target_id="ENT_EDMUND", rel_counterpart_id="ENT_PETER", causality_type="mutation_social", trait_target="affinity", trait_delta=0.6, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_BATTLE", target_id="ENT_EDMUND", rel_counterpart_id="ENT_PETER", causality_type="mutation_social", trait_target="affinity", trait_delta=0.3, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=19000, propagation_delay=0),
        CausalEdge(source_id="EVT_PEVENSIES_ARRIVE", target_id="ENT_LUCY", rel_counterpart_id="ENT_PETER", causality_type="mutation_social", trait_target="affinity", trait_delta=0.7, mechanism="social", evidence_strength="strong", causal_force=6.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_MEET_ASLAN", target_id="ENT_LUCY", rel_counterpart_id="ENT_PETER", causality_type="mutation_social", trait_target="affinity", trait_delta=0.2, mechanism="emotional", evidence_strength="moderate", causal_force=5.0, fabula_time=12500, propagation_delay=0),
        CausalEdge(source_id="EVT_PEVENSIES_ARRIVE", target_id="ENT_LUCY", rel_counterpart_id="ENT_SUSAN", causality_type="mutation_social", trait_target="affinity", trait_delta=0.65, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_ASLAN_DEATH", target_id="ENT_LUCY", rel_counterpart_id="ENT_SUSAN", causality_type="mutation_social", trait_target="affinity", trait_delta=0.3, mechanism="emotional", evidence_strength="strong", causal_force=7.0, fabula_time=17000, propagation_delay=0),
        CausalEdge(source_id="EVT_PEVENSIES_ARRIVE", target_id="ENT_PETER", rel_counterpart_id="ENT_SUSAN", causality_type="mutation_social", trait_target="affinity", trait_delta=0.65, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=1000, propagation_delay=0),
        CausalEdge(source_id="EVT_JOURNEY_TO_STONE_TABLE", target_id="ENT_PETER", rel_counterpart_id="ENT_SUSAN", causality_type="mutation_social", trait_target="affinity", trait_delta=0.2, mechanism="social", evidence_strength="moderate", causal_force=5.0, fabula_time=11000, propagation_delay=0),
        CausalEdge(source_id="EVT_ASLAN_DEATH", target_id="ENT_WHITE_WITCH", rel_counterpart_id="ENT_ASLAN", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.5, mechanism="betrayal", evidence_strength="strong", causal_force=9.0, fabula_time=17000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCH_KILLED", target_id="ENT_WHITE_WITCH", rel_counterpart_id="ENT_ASLAN", causality_type="mutation_social", trait_target="affinity", trait_delta=-0.5, mechanism="physical", evidence_strength="strong", causal_force=10.0, fabula_time=20000, propagation_delay=0),

        # ── orphan wirings: free-the-statues + utterance manifestations ──
        CausalEdge(source_id="EVT_EDMUND_RESCUED", target_id="EVT_ASLAN_FREES_STATUES",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=6.0, fabula_time=14000, propagation_delay=500),
        CausalEdge(source_id="EVT_LUCY_MEETS_TUMNUS", target_id="EVT_UTT_TUMNUS_CONFESSION",
                   causality_type="chain_reaction", mechanism="psychological", evidence_strength="strong",
                   causal_force=5.0, fabula_time=2500, propagation_delay=300),
        CausalEdge(source_id="EVT_EDMUND_TURKISH_DELIGHT", target_id="EVT_UTT_WITCH_TEMPTS_EDMUND",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=4000, propagation_delay=0),
        CausalEdge(source_id="EVT_BEAVERS_EXPLAIN_ASLAN", target_id="EVT_UTT_BEAVERS_PROPHECY",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=9000, propagation_delay=0),
        CausalEdge(source_id="EVT_EDMUND_BETRAYS", target_id="EVT_UTT_EDMUND_WARNS_WITCH",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=10000, propagation_delay=0),
        CausalEdge(source_id="EVT_WITCH_DEMANDS_EDMUND", target_id="EVT_UTT_WITCH_CLAIMS_EDMUND",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=15000, propagation_delay=0),
        CausalEdge(source_id="EVT_ASLAN_PACT", target_id="EVT_UTT_ASLAN_SECRET_PACT",
                   causality_type="chain_reaction", mechanism="social", evidence_strength="strong",
                   causal_force=5.0, fabula_time=16000, propagation_delay=0),
    ],

    # ── SPATIAL TOPOLOGY ────────────────────────────────────────────────
    spatial_topology=[
        SpatialEdge(source_id="LOC_PROFESSOR_HOUSE", target_id="LOC_WARDROBE"),
        SpatialEdge(source_id="LOC_WARDROBE", target_id="LOC_LANTERN_WASTE"),
        SpatialEdge(source_id="LOC_LANTERN_WASTE", target_id="LOC_TUMNUS_CAVE"),
        SpatialEdge(source_id="LOC_LANTERN_WASTE", target_id="LOC_BEAVERS_DAM"),
        SpatialEdge(source_id="LOC_BEAVERS_DAM", target_id="LOC_STONE_TABLE"),
        SpatialEdge(source_id="LOC_STONE_TABLE", target_id="LOC_ASLAN_CAMP"),
        SpatialEdge(source_id="LOC_LANTERN_WASTE", target_id="LOC_WITCH_CASTLE"),
        SpatialEdge(source_id="LOC_WITCH_CASTLE", target_id="LOC_STONE_TABLE"),
        SpatialEdge(source_id="LOC_STONE_TABLE", target_id="LOC_BATTLEFIELD"),
        SpatialEdge(source_id="LOC_BATTLEFIELD", target_id="LOC_CAIR_PARAVEL"),
    ],

    # ── INFORMATION TOPOLOGY ────────────────────────────────────────────
    channels={
        "CHN_BEAVER_FIRESIDE": Channel(
            id="CHN_BEAVER_FIRESIDE",
            name="Beavers' Fireside Conversation",
            medium="conversation",
            participant_ids=["ENT_MR_BEAVER", "ENT_MRS_BEAVER", "ENT_LUCY", "ENT_PETER", "ENT_SUSAN"],
            directionality="duplex",
            intelligibility={},
            established_at_fabula=9000,
            terminated_at_fabula=10000,
            evidence_strength="strong",
        ),
    },

    # ── WORLD TRAITS ────────────────────────────────────────────────────
    world_traits={
        "WORLD_DEEP_MAGIC": GlobalTrait(
            id="WORLD_DEEP_MAGIC",
            name="The Deep Magic",
            description="Ancient law carved on the Stone Table: every traitor's life belongs to the Witch. Drives the Witch's claim on Edmund and Aslan's sacrifice.",
            category="cosmology",
            magnitude=TraitVector(value=0.85, inertia=0.9, evidence_strength="strong"),
            affected_domains=["social", "epistemic"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=15000, triggered_by="EVT_WITCH_DEMANDS_EDMUND",
                    magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    description="Deep Magic invoked explicitly; its claim is absolute."),
                WorldTraitSnapshot(fabula_time=17000, triggered_by="EVT_ASLAN_DEATH",
                    magnitude=TraitVector(value=0.75, inertia=0.9, evidence_strength="strong"),
                    description="Deep Magic satisfied by Aslan's substitution; its grip loosens."),
            ],
        ),
        "WORLD_DEEPER_MAGIC": GlobalTrait(
            id="WORLD_DEEPER_MAGIC",
            name="The Deeper Magic from Before the Dawn of Time",
            description="Secret law older than the Deep Magic: a willing innocent sacrifice reverses death. Drives Aslan's resurrection.",
            category="cosmology",
            magnitude=TraitVector(value=0.6, inertia=0.95, evidence_strength="moderate"),
            affected_domains=["supernatural", "psychological"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=18000, triggered_by="EVT_ASLAN_RESURRECTION",
                    magnitude=TraitVector(value=0.95, inertia=0.95, evidence_strength="strong"),
                    description="Deeper Magic activates; death itself runs backward."),
            ],
        ),
        "WORLD_PROPHECY_FOUR_THRONES": GlobalTrait(
            id="WORLD_PROPHECY_FOUR_THRONES",
            name="Prophecy of the Four Thrones",
            description="Ancient rhyme: when two Sons of Adam and two Daughters of Eve sit on the thrones at Cair Paravel, the Witch's reign will end.",
            category="cosmology",
            magnitude=TraitVector(value=0.7, inertia=0.85, evidence_strength="strong"),
            affected_domains=["social", "psychological"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=21000, triggered_by="EVT_CORONATION",
                    magnitude=TraitVector(value=0.95, inertia=0.9, evidence_strength="strong"),
                    description="Prophecy fulfilled; the four children crowned at Cair Paravel."),
            ],
        ),
        "WORLD_ENCHANTED_WINTER": GlobalTrait(
            id="WORLD_ENCHANTED_WINTER",
            name="The Enchanted Winter",
            description="The Witch's spell: always winter, never Christmas. Drives the frozen landscape and oppressive atmosphere until Aslan's return breaks it.",
            category="magic_system",
            magnitude=TraitVector(value=0.9, inertia=0.75, evidence_strength="strong"),
            affected_domains=["environmental", "psychological"],
            state_timeline=[
                WorldTraitSnapshot(fabula_time=11500, triggered_by="EVT_FATHER_CHRISTMAS_GIFTS",
                    magnitude=TraitVector(value=0.5, inertia=0.6, evidence_strength="strong"),
                    description="Christmas arrives; the spell is weakening."),
                WorldTraitSnapshot(fabula_time=12000, triggered_by="EVT_SPRING_THAW",
                    magnitude=TraitVector(value=0.1, inertia=0.4, evidence_strength="strong"),
                    description="Spring breaks through; the enchanted winter is shattered."),
            ],
        ),
    },

    # ── SOCIAL TOPOLOGY (per-axis RelationshipMetric) ──
    social_topology=[
        # Lucy ↔ Tumnus: strong affinity bond
        RelationshipEdge(
            source_entity_id="ENT_LUCY", target_entity_id="ENT_TUMNUS",
            metrics={
                "affinity": RelationshipMetric(value=0.95, inertia=0.55, evidence_strength="strong", last_updated_fabula=3000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_TUMNUS", target_entity_id="ENT_LUCY",
            metrics={
                "affinity": RelationshipMetric(value=0.9, inertia=0.55, evidence_strength="strong", last_updated_fabula=3000),
            },
        ),
        # Edmund ↔ White Witch: corrupted bond → revulsion
        RelationshipEdge(
            source_entity_id="ENT_EDMUND", target_entity_id="ENT_WHITE_WITCH",
            metrics={
                "affinity": RelationshipMetric(value=-0.3, inertia=0.4, evidence_strength="strong", last_updated_fabula=14000),
                "fear": RelationshipMetric(value=0.75, inertia=0.25, evidence_strength="strong", last_updated_fabula=10000),
                "power_dynamic": RelationshipMetric(value=-0.8, inertia=0.7, evidence_strength="strong", last_updated_fabula=4000),
            },
        ),
        # Edmund ↔ Lucy: sibling tension → reconciliation
        RelationshipEdge(
            source_entity_id="ENT_EDMUND", target_entity_id="ENT_LUCY",
            metrics={
                "affinity": RelationshipMetric(value=0.3, inertia=0.4, evidence_strength="moderate", last_updated_fabula=14000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_LUCY", target_entity_id="ENT_EDMUND",
            metrics={
                "affinity": RelationshipMetric(value=0.6, inertia=0.45, evidence_strength="strong", last_updated_fabula=14000),
            },
        ),
        # Peter ↔ Edmund: protective elder → forgiveness
        RelationshipEdge(
            source_entity_id="ENT_PETER", target_entity_id="ENT_EDMUND",
            metrics={
                "affinity": RelationshipMetric(value=0.7, inertia=0.5, evidence_strength="strong", last_updated_fabula=14000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_EDMUND", target_entity_id="ENT_PETER",
            metrics={
                "affinity": RelationshipMetric(value=0.65, inertia=0.45, evidence_strength="moderate", last_updated_fabula=14000),
            },
        ),
        # Susan ↔ Edmund: cautious concern
        RelationshipEdge(
            source_entity_id="ENT_SUSAN", target_entity_id="ENT_EDMUND",
            metrics={
                "affinity": RelationshipMetric(value=0.6, inertia=0.45, evidence_strength="moderate", last_updated_fabula=14000),
            },
        ),
        # Peter ↔ Lucy: protective care
        RelationshipEdge(
            source_entity_id="ENT_PETER", target_entity_id="ENT_LUCY",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.6, evidence_strength="strong", last_updated_fabula=5500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_LUCY", target_entity_id="ENT_PETER",
            metrics={
                "affinity": RelationshipMetric(value=0.8, inertia=0.55, evidence_strength="strong", last_updated_fabula=1000),
            },
        ),
        # Susan ↔ Lucy: sisterly bond
        RelationshipEdge(
            source_entity_id="ENT_SUSAN", target_entity_id="ENT_LUCY",
            metrics={
                "affinity": RelationshipMetric(value=0.75, inertia=0.55, evidence_strength="moderate", last_updated_fabula=5500),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_LUCY", target_entity_id="ENT_SUSAN",
            metrics={
                "affinity": RelationshipMetric(value=0.75, inertia=0.5, evidence_strength="moderate", last_updated_fabula=1000),
            },
        ),
        # Peter ↔ Susan: eldest siblings
        RelationshipEdge(
            source_entity_id="ENT_PETER", target_entity_id="ENT_SUSAN",
            metrics={
                "affinity": RelationshipMetric(value=0.75, inertia=0.55, evidence_strength="moderate", last_updated_fabula=1000),
            },
        ),
        RelationshipEdge(
            source_entity_id="ENT_SUSAN", target_entity_id="ENT_PETER",
            metrics={
                "affinity": RelationshipMetric(value=0.8, inertia=0.55, evidence_strength="strong", last_updated_fabula=13000),
            },
        ),
        # Lucy ↔ Aslan: devotion
        RelationshipEdge(
            source_entity_id="ENT_LUCY", target_entity_id="ENT_ASLAN",
            metrics={
                "affinity": RelationshipMetric(value=0.95, inertia=0.65, evidence_strength="strong", last_updated_fabula=17000),
                "power_dynamic": RelationshipMetric(value=-0.8, inertia=0.75, evidence_strength="strong", last_updated_fabula=12500),
            },
        ),
        # Peter ↔ Aslan: loyalty and respect
        RelationshipEdge(
            source_entity_id="ENT_PETER", target_entity_id="ENT_ASLAN",
            metrics={
                "affinity": RelationshipMetric(value=0.9, inertia=0.6, evidence_strength="strong", last_updated_fabula=12500),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=12500),
            },
        ),
        # Susan ↔ Aslan: cautious reverence
        RelationshipEdge(
            source_entity_id="ENT_SUSAN", target_entity_id="ENT_ASLAN",
            metrics={
                "affinity": RelationshipMetric(value=0.85, inertia=0.6, evidence_strength="strong", last_updated_fabula=17000),
                "power_dynamic": RelationshipMetric(value=-0.7, inertia=0.7, evidence_strength="strong", last_updated_fabula=12500),
            },
        ),
        # Edmund ↔ Aslan: gratitude and awe
        RelationshipEdge(
            source_entity_id="ENT_EDMUND", target_entity_id="ENT_ASLAN",
            metrics={
                "affinity": RelationshipMetric(value=0.9, inertia=0.55, evidence_strength="strong", last_updated_fabula=14000),
                "power_dynamic": RelationshipMetric(value=-0.8, inertia=0.7, evidence_strength="strong", last_updated_fabula=14000),
            },
        ),
        # White Witch ↔ Aslan: mortal enmity
        RelationshipEdge(
            source_entity_id="ENT_WHITE_WITCH", target_entity_id="ENT_ASLAN",
            metrics={
                "affinity": RelationshipMetric(value=-1.0, inertia=0.85, evidence_strength="strong", last_updated_fabula=9000),
                "fear": RelationshipMetric(value=0.85, inertia=0.35, evidence_strength="strong", last_updated_fabula=18000),
                "power_dynamic": RelationshipMetric(value=0.5, inertia=0.7, evidence_strength="strong", last_updated_fabula=15000),
            },
        ),
    ],
)