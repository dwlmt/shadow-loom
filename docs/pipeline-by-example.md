# Pipeline by Example — Real Data Walkthrough

This document is a **data-anchored** companion to
[architecture.md](architecture.md), [pipeline-walkthrough.md](pipeline-walkthrough.md),
and [model-examples.md](model-examples.md). Where those documents describe
*what* the pipeline does and *why*, this one shows the actual values that
flow through each stage when you point it at a bundled fixture.

Every code excerpt below is verbatim from
[`example_worlds/`](../example_worlds) — load any of the worlds with

```python
from example_worlds.macbeth import world_state
from shadow_loom.pipeline import run_pipeline
```

…and you can reproduce every number in this doc.

---

## 0. Inventory of bundled fixtures

Each `example_worlds/<name>.py` instantiates a fully-populated
`WorldStateV1` — entities, events, causal/social edges, channels,
propositions, and per-entity concerns. The table below is the actual
cardinality of every bundled fixture (counts produced by walking the
imported `world_state` object; reproduce with
`python scripts/_dump_world_inventory.py` or any one-line variant
that instantiates the world and reads `len(ws.entities)` etc.).

| Fixture | ent | loc | obj | WT | evt | causal | social | chn | prop | concern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `macbeth` | 12 | 9 | 6 | 3 | 32 | 105 | 18 | 2 | 18 | 21 |
| `romeo_and_juliet` | 15 | 9 | 6 | 5 | 35 | 112 | 22 | 2 | 12 | 19 |
| `apocalypse_now` | 10 | 8 | 4 | 4 | 23 | 91 | 20 | 2 | 17 | 24 |
| `great_gatsby` | 9 | 8 | 5 | 3 | 26 | 97 | 22 | 2 | 19 | 23 |
| `gone_girl` | 11 | 8 | 5 | 3 | 24 | 103 | 28 | 4 | 12 | 17 |
| `reservoir_dogs` | 9 | 5 | 5 | 2 | 18 | 106 | 24 | 1 | 12 | 15 |
| `frankenstein` | 12 | 9 | 4 | 3 | 32 | 94 | 20 | 3 | 15 | 19 |
| `death_on_the_nile` | 15 | 8 | 8 | 3 | 30 | 99 | 20 | 2 | 18 | 33 |
| `dads_army` | 11 | 7 | 3 | 2 | 25 | 134 | 22 | 2 | 16 | 36 |
| `tinker_tailor_soldier_spy` | 17 | 10 | 5 | 5 | 43 | 100 | 27 | 5 | 11 | 15 |
| `great_expectations` | 14 | 10 | 5 | 3 | 38 | 141 | 22 | 1 | 17 | 18 |
| `wuthering_heights` | 12 | 5 | 4 | 3 | 36 | 122 | 22 | 5 | 14 | 20 |
| `persuasion` | 17 | 7 | 5 | 3 | 28 | 104 | 26 | 2 | 13 | 18 |
| `nineteen_eighty_four` | 9 | 9 | 5 | 3 | 28 | 98 | 18 | 5 | 11 | 13 |
| `brief_encounter` | 11 | 7 | 5 | 3 | 21 | 83 | 12 | 2 | 7 | 9 |
| `a_fish_called_wanda` | 7 | 10 | 3 | 4 | 39 | 128 | 18 | 4 | 17 | 21 |
| `the_devil_wears_prada` | 8 | 7 | 5 | 3 | 27 | 73 | 12 | 3 | 17 | 20 |
| `the_lion_the_witch_and_the_wardrobe` | 13 | 10 | 6 | 4 | 33 | 107 | 26 | 1 | 13 | 15 |
| `once_upon_a_time_in_the_west` | 9 | 5 | 5 | 3 | 27 | 67 | 12 | 3 | 17 | 18 |
| `a_court_of_thorn_and_roses` | 10 | 8 | 4 | 6 | 30 | 124 | 28 | 4 | 23 | 33 |

Key: `ent`=entities, `loc`=locations, `obj`=objects,
`WT`=`WORLD_*` traits, `evt`=events (choice/outcome/revelation/utterance
combined), `causal`=`CausalEdge` count across all five modalities,
`social`=`RelationshipEdge` directed dyads (asymmetric — see §1.3),
`chn`=`Channel` count, `prop`=`Proposition` registry size, `concern`
=total `Concern` rows summed across every `Entity.concerns`. Every
fixture has a populated `narrative_style` profile.

All twenty fixtures instantiate the full Phase A3 (proposition
catalogue) + per-entity concern roster used by the propositional
affect scorers in [academic-foundations.md §3.7](academic-foundations.md#37-propositional-belief-revision-affect-affect_unificationpy).

---

## Index

1. [The world model — what gets stored](#1-the-world-model--what-gets-stored)
   - 1.1 [Locations and ambient state](#11-locations-and-ambient-state)
   - 1.2 [Objects and affordance gates](#12-objects-and-affordance-gates)
   - 1.3 [Entities — traits, beliefs, state timeline](#13-entities--traits-beliefs-state-timeline)
   - 1.4 [Events — fabula vs syuzhet, utterances](#14-events--fabula-vs-syuzhet-utterances)
   - 1.5 [Causal topology — the five edge modalities](#15-causal-topology--the-five-edge-modalities)
   - 1.6 [Channels and per-participant intelligibility](#16-channels-and-per-participant-intelligibility)
   - 1.7 [Global / WORLD_ traits](#17-global--world_-traits)
2. [Building the graph](#2-building-the-graph)
   - 2.1 [Ego-graph slicing](#21-ego-graph-slicing)
   - 2.2 [The AMWN sandbox — factual vs shadow worlds](#22-the-amwn-sandbox--factual-vs-shadow-worlds)
3. [Causal physics in motion](#3-causal-physics-in-motion)
   - 3.1 [`reconstruct_entity_at` — pinning state to a point in time](#31-reconstruct_entity_at--pinning-state-to-a-point-in-time)
   - 3.2 [Rung 1 — observation](#32-rung-1--observation)
   - 3.3 [Rung 2 — intervention (`do`-operator)](#33-rung-2--intervention-do-operator)
   - 3.4 [Rung 3 — counterfactual (`abduction → do → propagate`)](#34-rung-3--counterfactual-abduction--do--propagate)
   - 3.5 [Forward propagation — `Impact > Inertia`](#35-forward-propagation--impact--inertia)
   - 3.6 [Social propagation and belief revision](#36-social-propagation-and-belief-revision)
4. [Affective scoring with real numbers](#4-affective-scoring-with-real-numbers)
   - 4.1 [Mystery — Macbeth Act II](#41-mystery--macbeth-act-ii)
   - 4.2 [Dramatic irony — Death on the Nile](#42-dramatic-irony--death-on-the-nile)
   - 4.3 [Suspense — Romeo & Juliet's tomb](#43-suspense--romeo--juliets-tomb)
   - 4.4 [Surprise — Reservoir Dogs reveal](#44-surprise--reservoir-dogs-reveal)
   - 4.5 [Emotion targets — Gone Girl, grief and rage](#45-emotion-targets--gone-girl-grief-and-rage)
5. [Directive assembly — picking the best intervention](#5-directive-assembly--picking-the-best-intervention)
6. [Generation — brief → constrained prose](#6-generation--brief--constrained-prose)
7. [Audit and refinement](#7-audit-and-refinement)
8. [Re-extraction and merge](#8-re-extraction-and-merge)

---

## 1. The world model — what gets stored

### 1.1 Locations and ambient state

A `Location` carries an `ambient_state: Dict[str, AmbientVector]`. Each
ambient vector has `value`, `volatility`, and `evidence_strength`. Volatility
governs how fast the ambient drifts under propagation; evidence strength
weights its contribution to downstream causal force.

From [`example_worlds/macbeth.py`](../example_worlds/macbeth.py):

```python
"LOC_HEATH": Location(
    name="The Heath",
    description="Desolate fog-shrouded moor where Macbeth and Banquo "
                "first meet the witches.",
    ambient_state={
        "supernatural": AmbientVector(value=0.9, volatility=0.2,
                                      evidence_strength="strong"),
        "concealment":  AmbientVector(value=0.7, volatility=0.3,
                                      evidence_strength="moderate"),
    },
),
```

That single `supernatural=0.9` value is what later allows
`EVT_WITCHES_PROPHECY_1` to land a believable trait shock on Macbeth's
`ambition`: `LOC_HEATH` is the common-cause parent supplying the ambient
pressure that the `ambient_propagation` edge forwards. Compare with
[Reservoir Dogs](../example_worlds/reservoir_dogs.py)'s diner —
`fluorescent_banality=0.85` does the opposite work, *suppressing* tension so
the tipping-debate utterances read as banter rather than threat.

### 1.2 Objects and affordance gates

Objects are not decoration — every `Affordance(action="kill", ...)` is a
*precondition gate* the causal engine checks before letting an event fire:

```python
"OBJ_BLOODY_DAGGERS": NarrativeObject(
    id="OBJ_BLOODY_DAGGERS", name="Bloody Daggers",
    location_id=None, owner_id="ENT_MACBETH",
    properties={"state": "blood-smeared", "stained_with": "duncan_blood"},
    affordances=[
        Affordance(action="kill",  target_type="Entity"),
        Affordance(action="frame", target_type="Entity"),
    ],
),
```

When the user issues `do(EVT_DUNCAN_MURDER = ⊘)` the engine doesn't just
mark the event prevented — it checks that the daggers' `kill` affordance is
no longer being satisfied, and downstream the `frame` affordance becomes
ungated, which feeds the chain into `EVT_GROOMS_BLAMED`. In Death on the
Nile the same logic gates two parallel weapons:

```python
"OBJ_PISTOL_PEARL":  affordances=[Affordance(action="kill_owner_or_witness")],
"OBJ_VAN_SCHUYLER_STOLE": affordances=[Affordance(action="muffle_pistol")],
"OBJ_RED_INK_BOTTLE": affordances=[Affordance(action="fake_blood")],
```

…the Christie plot is *defined* by the affordance topology. Removing the
stole removes the silencer affordance and the staged-shooting alibi
collapses; the engine surfaces this as a `CausalPhysicsResult.blocked`
entry rather than a hand-wave.

### 1.3 Entities — traits, beliefs, state timeline

The Entity record is the densest object in the graph. From Macbeth's
opening state (line 149 of the fixture):

```python
"ENT_MACBETH": Entity(
    id="ENT_MACBETH", name="Macbeth (Thane of Glamis)",
    location_id="LOC_BATTLEFIELD", status="healthy",
    traits={
        "ambition":     TraitVector(value=0.7,  inertia=0.55, evidence_strength="strong"),
        "courage":      TraitVector(value=0.85, inertia=0.7,  evidence_strength="strong"),
        "loyalty":      TraitVector(value=0.7,  inertia=0.5,  evidence_strength="moderate"),
        "guilt":        TraitVector(value=0.1,  inertia=0.25, evidence_strength="weak"),
        "paranoia":     TraitVector(value=0.2,  inertia=0.3,  evidence_strength="weak"),
        "ruthlessness": TraitVector(value=0.4,  inertia=0.45, evidence_strength="moderate"),
        "despair":      TraitVector(value=0.1,  inertia=0.25, evidence_strength="weak"),
    },
    beliefs=[
        Belief(target_id="ENT_DUNCAN", confidence=0.9, inertia=0.55,
               perceived_state="Duncan is my kinsman and rightful king",
               evidence_strength="strong"),
        Belief(target_id="ENT_BANQUO", confidence=0.85, inertia=0.5,
               perceived_state="Banquo is my trusted comrade-in-arms",
               evidence_strength="strong"),
        Belief(target_id="ENT_WITCHES", confidence=0.4, inertia=0.35,
               perceived_state="The witches' prophecies may yet prove true",
               evidence_strength="moderate"),
    ],
    state_timeline=[
        EntityStateSnapshot(fabula_time=2000, triggered_by="EVT_WITCHES_PROPHECY_1",
            traits={"ambition": TraitVector(value=0.85, inertia=0.65)}),
        EntityStateSnapshot(fabula_time=5000, triggered_by="EVT_LADY_MACBETH_PERSUADES",
            traits={"ruthlessness": TraitVector(value=0.80, inertia=0.45)}),
        EntityStateSnapshot(fabula_time=6000, triggered_by="EVT_DUNCAN_MURDER",
            traits={
                "guilt":        TraitVector(value=0.7,  inertia=0.4),
                "paranoia":     TraitVector(value=0.55, inertia=0.4),
                "ruthlessness": TraitVector(value=0.65, inertia=0.55),
            },
            location_id="LOC_INVERNESS_CASTLE"),
        EntityStateSnapshot(fabula_time=11000, triggered_by="EVT_BANQUO_MURDERED",
            traits={
                "guilt":    TraitVector(value=0.85, inertia=0.5),
                "paranoia": TraitVector(value=0.85, inertia=0.5),
            }),
        EntityStateSnapshot(fabula_time=13000, triggered_by="EVT_WITCHES_PROPHECY_2",
            traits={"courage": TraitVector(value=0.95, inertia=0.75),
                    "paranoia": TraitVector(value=0.55, inertia=0.45)},
            beliefs_added=[
                Belief(target_id="ENT_MACDUFF", confidence=0.85, inertia=0.6,
                       perceived_state="Macduff threatens me but no man of "
                                       "woman born can harm me",
                       established_at_fabula=13000, evidence_strength="strong"),
            ]),
        EntityStateSnapshot(fabula_time=18500, triggered_by="EVT_BIRNAM_WOOD_MOVES",
            traits={"despair": TraitVector(value=0.95, inertia=0.55)},
            beliefs_invalidated=["ENT_MACDUFF"]),
        EntityStateSnapshot(fabula_time=19000, triggered_by="EVT_MACBETH_KILLED",
            status="dead"),
    ],
),
```

This is the **Hybrid 4+5 timeline**:

* The `traits` / `beliefs` / `status` / `location_id` on the entity itself
  hold the **initial** state.
* `state_timeline` holds **journalled deltas** — each snapshot pinned to
  the `fabula_time` of the event that caused it.

Note the *inertia bumping*: `guilt` starts at `inertia=0.25`, jumps to
`0.4` after `EVT_DUNCAN_MURDER`, then `0.5` after `EVT_BANQUO_MURDERED`.
The engine deliberately *raises* inertia after a shock so the trait
becomes harder to wash off — the play's psychology, encoded as physics.

`reconstruct_entity_at(entity, fabula_t)` walks this timeline and returns
the trait/belief snapshot the engine should reason against at that point.
This is what Step 0 (sandbox seeding) and Step 2 (abduction) both call.

### 1.4 Events — fabula vs syuzhet, utterances

Every event carries two clocks. Compare `EVT_REBELLION_DEFEATED`
(`fabula_time=1000, syuzhet_index=1` — they agree) with the witches' first
prophecy (`fabula_time=2000, syuzhet_index=2`) and with the **utterance
event** that delivers it on stage:

```python
EventNode(id="EVT_UTT_PROPHECY_HEATH", event_type="utterance",
    speaker_id="ENT_WITCHES",
    addressee_ids=["ENT_MACBETH", "ENT_BANQUO"],
    actor_ids=["ENT_WITCHES"],
    target_ids=["EVT_WITCHES_PROPHECY_1"],
    content="All hail Macbeth — Thane of Glamis, Thane of Cawdor, "
            "king hereafter; and Banquo, lesser yet greater, shall "
            "father a line of kings though never wear the crown himself.",
    via_channel_id="CHN_PROPHETIC_LINK",
    truth_value="performative",
    fabula_time=2000, syuzhet_index=3),
```

Several things become clear from this single record:

* `event_type="utterance"` is a first-class event modality; the speech
  *is* the act.
* `target_ids=["EVT_WITCHES_PROPHECY_1"]` cross-links the utterance to
  the underlying causal event — utterances are about events, and any
  participant who hears the utterance gains a `Belief(target_id=…)` about
  the referenced event.
* `truth_value="performative"` means the utterance enacts what it says
  rather than reporting it — the engine will *not* update beliefs about
  the utterance content as if it were a factual report.
* Compare with `EVT_UTT_MACBETH_INVINCIBILITY_TAUNT` —
  `truth_value="false"`. Macduff *hears* the taunt, his belief about
  Macbeth's confidence updates, but his belief about the underlying claim
  ("can be killed by no man of woman born") does not, because the engine
  weights the belief update by truth value × intelligibility ×
  evidence_strength.

Non-performative utterances (`truth_value ∈ {true, false, unknown}`) must
not place future-fabula events in `target_ids`; their causal effects on
later events belong on `causal_topology` as `chain_reaction` edges rather
than in the utterance's own `target_ids` field. The ingestion validator
(`_validate_time_ordering` Rule 5) raises a temporal error otherwise.
Performative utterances (prophecies, vows, orders) are exempt because
they posit the future state rather than report a past one.

Reservoir Dogs is the cleanest demo of the two-clock split:
`EVT_ORANGE_RECRUITED` sits at low `fabula_time` but high `syuzhet_index`
because the audience learns it long after they meet Orange in the
warehouse. The causal engine *only* uses `fabula_time` for propagation;
the affective scorer uses `syuzhet_index` to compute reader-side surprise.

### 1.5 Causal topology — the five edge modalities

Every Macbeth `CausalEdge` exhibits one of five modalities. The engine
dispatches on `causality_type`:

| Modality | Macbeth example | What `propagate()` does |
|---|---|---|
| `chain_reaction` | `EVT_REBELLION_DEFEATED → EVT_CAWDOR_TITLE` (`mechanism="social"`, `causal_force=6.0`, `propagation_delay=2000`) | Activates target event when source fires. |
| `mutation` | `EVT_BANQUO_GHOST → ENT_MACBETH.guilt` | Trait shock; impulse = `causal_force * evidence_strength`. |
| `mutation_social` | `EVT_LADY_MACBETH_PERSUADES → REL(MACBETH, LADY).power_dynamic` | Per-axis relationship metric update. |
| `affordance_gate` | `OBJ_BLOODY_DAGGERS.kill` gates `EVT_DUNCAN_MURDER` | Permits / blocks the event, doesn't push state. |
| `ambient_propagation` | `LOC_HEATH.supernatural → ENT_MACBETH belief uptake` | Continuous pressure scaled by `ambient_force_multiplier`. |

All five live in the same `causal_topology` list — the dispatcher inside
[`causal_physics.py::propagate`](../shadow_loom/causal_physics.py)
branches on `causality_type` rather than holding five separate stores.

### 1.6 Channels and per-participant intelligibility

A `Channel` is a *standing capability* between agents (a private
conversation, a shouted warning, an encrypted radio). In Death on the
Nile the same physical exchange is modelled with *different*
intelligibility maps depending on who is in earshot — see
[model-examples.md §2](model-examples.md#2-death-on-the-nile--epistemic-physics-and-dramatic-irony).

The contract is: each utterance event references exactly one
`via_channel_id`, the channel says who is decoding it at what fidelity,
and `propagate_social()` walks the channel graph weighting
`belief_update = intelligibility × evidence_strength × truth_weight`.
"Eavesdropping" is therefore not a separate primitive — any non-addressee
participant whose `intelligibility >= settings.physics.intelligibility_threshold`
is eligible for the same belief update.

The rebuilt fixture corpus exercises the full intelligibility spectrum:

| Channel | Distinctive intelligibility | What it buys the engine |
|---|---|---|
| `CHN_HIDDEN_TELESCREEN_SURVEILLANCE` (1984) | Winston / Julia = 0.0 | Pure dramatic irony — the audience and the Party hear, the protagonists do not. |
| `CHN_RHYS_FEYRE_BOND` (ACOTAR) | Feyre = 0.4 | Partial telepathic bleed; beliefs form with reduced confidence. |
| `CHN_PIP_BENEFACTOR_PIPELINE` (Great Expectations) | Pip = 0.1 | The information exists in the graph; Pip's beliefs about its source are systematically wrong. |
| `CHN_WILLIAM_FLATTERING_DISCOURSE` (Persuasion) | Anne = 0.2 | Sustained low-intelligibility flattery — Anne hears the words, doesn't decode the manipulation. |
| `CHN_LAURA_INTERIOR_CONFESSION` (Brief Encounter) | Fred = 0.0 | Internal-monologue channel; Laura's husband is structurally cut out. |

Utterance `truth_value` cuts across this. `EVT_UTT_HATE_WEEK_ENEMY_SWITCH`
(`truth_value="false"`, via `CHN_TELESCREEN_BROADCAST`) and
`EVT_UTT_LINTON_COERCED_LETTERS_TO_CATHY` (`truth_value="false"`, via
`CHN_LINTON_CATHY_COERCED_LETTERS`) both fire belief updates over high-
intelligibility channels, but the truth-value guard blocks the *factual*
reinforcement step in [`causal_physics.py`](../shadow_loom/causal_physics.py)
— recipients update their belief about *what was said*, not about what is
so. Performative utterances such as `EVT_UTT_AMARANTHA_RIDDLE` (ACOTAR)
or `EVT_UTT_DONT_TELL_HIM_PIKE` (Dad's Army) likewise propagate social
consequences without contributing factual evidence to abduction.

### 1.7 Global / WORLD_ traits

`GlobalTrait` (`WORLD_*`) nodes are the *common-cause parents*:

* Macbeth — `WORLD_PROPHECY_VALIDITY`, `WORLD_DIVINE_RIGHT_OF_KINGS`,
  `WORLD_HEATH_FOG`.
* 1984 — `WORLD_REGIME` (so high it back-pressures every interaction
  through `ambient_propagation`).
* Reservoir Dogs — `WORLD_CRIMINAL_CODE` and `WORLD_POLICE_INFILTRATION`
  in tension; *every* crew choice is forced through both filters.
* Gone Girl — `WORLD_TRIAL_BY_MEDIA`,
  `WORLD_MARRIAGE_AS_PERFORMANCE`, `WORLD_RECESSION_PRECARITY`.

Each carries a `state_timeline` of `WorldTraitSnapshot` so the
common-cause itself can shift across acts (Reservoir Dogs'
`WORLD_POLICE_INFILTRATION` is constant; 1984's `WORLD_REGIME` ratchets
up monotonically; Macbeth's `WORLD_DIVINE_RIGHT_OF_KINGS` collapses at
`EVT_DUNCAN_MURDER` and never recovers).

---

## 2. Building the graph

### 2.1 Ego-graph slicing

[`extract_graph.py`](../shadow_loom/extract_graph.py) takes the full
`WorldStateV1` and a query payload (focus entities, locations, fabula
window, memory limit) and returns an **ego-payload**: just the slice
relevant to the query. For a query focussed on `ENT_MACBETH` at
`fabula_time ≈ 6000` (the night of the murder), the slice contains:

* **focus_entities**: `ENT_MACBETH` (reconstructed at t=6000 — `ambition`
  has already shocked to 0.85, `guilt` has not yet shocked).
* **present_entities**: `ENT_LADY_MACBETH`, `ENT_DUNCAN`,
  `ENT_BANQUO` (1-hop spatial neighbours via `SpatialEdge`s anchoring
  `LOC_INVERNESS_CASTLE`).
* **present_objects**: `OBJ_BLOODY_DAGGERS`, `OBJ_LETTER` (objects whose
  `location_id` is the current location or whose `owner_id` is a present
  entity).
* **current_locations**: `LOC_INVERNESS_CASTLE` plus its ambient state.
* **recent_memory**: events within `memory_limit` fabula steps —
  `EVT_REBELLION_DEFEATED`, `EVT_WITCHES_PROPHECY_1`, `EVT_CAWDOR_TITLE`,
  `EVT_LADY_MACBETH_PERSUADES`.
* **relevant_utterance_events**: the on-stage utterances belonging to
  channels the focus entity participates in — here
  `EVT_UTT_PROPHECY_HEATH`, `EVT_UTT_DUNCAN_BESTOWS_CAWDOR`.

This is a *Markov-blanket heuristic* — see
[academic-foundations.md §2.4](academic-foundations.md#24-d-separation-and-ego-graph-slicing).
It keeps the sandbox tractable; the trade-off is that long-range causal
links outside the slice (e.g. the witches' second prophecy at t=13000)
have to be explicitly carried in by the query author or surfaced by the
brief.

### 2.2 The AMWN sandbox — factual vs shadow worlds

[`AMWNInstantiator.create_sandbox`](../shadow_loom/instantiator.py)
turns the ego-payload into a `networkx.MultiDiGraph` and stamps every
node with a `world_id`:

```python
is_volatile = query_type in ["intervention", "counterfactual"]
target_world_id = "shadow" if is_volatile else "factual"
```

For an intervention query like *"Macbeth refuses to murder Duncan"*, the
sandbox is born as a `world_id="shadow"` mirror. The factual graph in
`world_state` is **never mutated** by physics; all surgery happens on the
sandbox. After scoring, the sandbox can be discarded (the analyst was
just exploring) or promoted to canon via `db.promote_branch`. This is
the AMWN pattern from
[Correa & Bareinboim 2025](academic-foundations.md#22-ancestral-multi-world-networks-and-ctf-calculus--correa--bareinboim-icml-2025):
shadow and factual share an ancestor graph, then diverge at the
intervention point.

---

## 3. Causal physics in motion

### 3.1 `reconstruct_entity_at` — pinning state to a point in time

Before any do-operator surgery runs, the sandbox seeds entity nodes from
their `state_timeline` at the simulation's `temporal_anchor`. From
[`causal_physics.py::abduction_update`](../shadow_loom/causal_physics.py):

```python
max_ft = max(d.get("fabula_time", 0) for _, d in self.sandbox.nodes(data=True)
             if d.get("fabula_time"))
if max_ft is not None and factual.state_timeline:
    reconstructed = reconstruct_entity_at(factual, max_ft)
    target_traits  = reconstructed["traits"]
    target_beliefs = reconstructed["beliefs"]
```

For Macbeth at `temporal_anchor=11000` (just after Banquo's murder), the
reconstruction yields:

```text
ambition     0.85     (set at t=2000)
courage      0.85     (initial, unchanged)
guilt        0.85     (latest set at t=11000)
paranoia     0.85     (latest set at t=11000)
ruthlessness 0.65     (latest set at t=6000)
despair      0.10     (initial, unchanged)
```

…with `beliefs` containing the original three plus the
`Belief(target_id=ENT_MACDUFF, …)` if the anchor were beyond t=13000.

### 3.2 Rung 1 — observation

The simplest query type. `query_type="observation"` runs no surgery; the
narrative-physics router calls `extract_graph.build_ego_payload(...)` and
returns the slice. There is no time advance, no propagation, no prose. In
`shadow_loom_mcp` this backs `query_world_state` and the UI's **Explorer**
tab.

Example: *"What does Poirot know about Linnet's death just before the
disembarkation at Shellal?"* — the engine returns Poirot's `beliefs` list
as of `syuzhet_anchor=N` and the revealed causal edges into
`EVT_LINNET_SHOT` whose source is in his ego-graph. The factual graph is
read-only.

### 3.3 Rung 2 — intervention (`do`-operator)

Macbeth, intervention query: *"What if Macbeth had no ambition?"*

```python
from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from example_worlds.macbeth import world_state as ws

ego = extract_ego_graph_from_memory(
    ws,
    focus_entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH",
                      "ENT_DUNCAN", "ENT_BANQUO", "ENT_MACDUFF"],
)
sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
engine = CausalPhysicsEngine(sandbox, ws)
result = engine.execute(
    rung=2,
    interventions={"ENT_MACBETH.traits.ambition": 0.0},
    target_node_ids=["ENT_DUNCAN", "ENT_LADY_MACBETH"],
)
```

What the engine does, in order:

1. `AMWNInstantiator.create_sandbox` clones the ego payload into a
   NetworkX `MultiDiGraph` tagged `world_id="shadow"`. The factual
   graph is read-only.
2. `engine.execute(rung=2, ...)` calls `_apply_ctf_calculus_preflight`,
   then `apply_do_operator({"ENT_MACBETH.traits.ambition": 0.0})`,
   which severs every incoming `WORLD_*` / `ENT_WITCHES_PROPHECY_*`
   causal edge into the `ambition` axis, registers
   `("ENT_MACBETH", "ambition") ∈ _intervened_traits`, and pins the
   value at `0.0` for the rest of the run.
3. `propagate()` walks the causal sub-graph in topological order with
   the pin in place. Sibling traits on Macbeth (`courage`, `paranoia`,
   `guilt`, `loyalty`…) remain free; the ambition-locked node still
   propagates onto downstream entities through edges whose mechanism
   doesn't gate on ambition.

The real `result.mutations` from this run (Pydantic
`TraitMutation` records, exact values from the bundled fixture):

```text
ENT_LENNOX        suspicion       +0.242 → +0.253   (impact +0.023)
ENT_LENNOX        caution         +0.862 → +0.868   (impact +0.023)
ENT_LADY_MACBETH  ruthlessness    +0.698 → +0.704   (impact +0.019)
ENT_LADY_MACBETH  resolve         +0.966 → +0.980   (impact +0.040)
ENT_LADY_MACBETH  guilt           +0.922 → +0.917   (impact -0.006)
ENT_BANQUO        suspicion       +0.001 → +0.013   (impact +0.022)
ENT_MALCOLM       courage         +0.713 → +0.722   (impact +0.026)
```

plus 16 propagation impulses absorbed by the noisy-OR gate
(`result.blocked`, `reason="noisy_or_absorbed"`) — e.g.
`ENT_LADY_MACBETH.ambition` saw a `+0.074` impulse but the
noisy-OR aggregate over its incoming edges fell below the
`propagation_threshold`, so the trait stayed pinned.

The Rule-3 pre-flight pruner (`result.rule3_pruned_interventions`)
is empty here because `ENT_MACBETH.ambition` has a clear directed
path to both target nodes in the mutilated diagram. Compare with
the vacuous query *"do(ENT_DUNCAN.kindness=0)"* targeting
`ENT_BANQUO`: the engine still runs it (advisory mode) but
flags the absence of any directed path.

### 3.4 Rung 3 — counterfactual (`abduction → do → propagate`)

Same fixture, counterfactual query: *"Given the catastrophe we
actually saw on stage, what if Macbeth had no ambition?"*

```python
result = engine.execute(
    rung=3,
    interventions={"ENT_MACBETH.traits.ambition": 0.0},
    evidence_node_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
    target_node_ids=["ENT_DUNCAN", "ENT_LADY_MACBETH"],
)
```

The engine runs in three phases (Pearl 2009 §7):

**A. Abduction.**
[`CausalPhysicsEngine.abduction_update`](../shadow_loom/causal_physics.py)
back-propagates every observed downstream into the sandbox.
On the bundled Macbeth fixture the call populates
`result.hidden_deltas` with the per-trait latent shifts. The
actual values the engine reports for Macbeth and Lady Macbeth:

```text
ENT_MACBETH:
  ambition       +0.598   (huge: factual ambition is 0.99,
                           sandbox prior is ~0.39)
  courage        -0.061
  loyalty        +0.102
  guilt          +0.444
  paranoia       +0.173
  ruthlessness   -0.289
  despair        +0.670
ENT_LADY_MACBETH:
  ambition       -0.076
  ruthlessness   +0.188
  resolve        -0.800   (factual is 0.20 — she has cracked;
                           sandbox prior was 0.99 from Act II)
  guilt          +0.950
```

These numbers come from a real call to
`engine.execute(rung=3, ...)` against `example_worlds/macbeth.py` and
are reproducible via
[`scripts/_dump_pearl_rungs.py`](../scripts/_dump_pearl_rungs.py).

**B. Intervention.** With `_hidden_deltas` now staged as active
sources, `apply_do_operator({"ENT_MACBETH.traits.ambition": 0.0})`
severes incoming edges and pins the trait at `0.0`.

**C. Propagation.** Re-runs `propagate()` with both the abducted
latents *and* the do-pin in play. The behavioural asymmetry vs Rung 2
shows up clearly:

* **Rung 2** above moved Lady Macbeth's `resolve` only `+0.040`
  (from her sandbox-default value).
* **Rung 3** here records `result.mutations` ending with
  `ENT_LADY_MACBETH.guilt: +0.792 → +0.841` (impact `+0.062`) and
  `resolve: +0.484 → +0.495` because abduction has *already* pulled
  her toward the observed Act V state. The pinned ambition cannot
  retroactively undo the guilt that abduction inferred from the
  evidence; the engine reports five propagation mutations on top of
  the latent shift.

Note also `result.rule3_pruned_interventions ==
["ENT_MACBETH.traits.ambition"]`: under the bundled fixture the
static-graph Rule-3 check considers the intervention vacuous on the
world-cropped diagram (no edge from `ambition` to the chosen
target set survives the mutilation), but advisory-mode keeps it in
the simulation so the abduction-driven downstream still mutates.
This is exactly the over-strict d-separation behaviour the closed-
world caveat in `design-decisions.md` warns about — the user can
opt into `rule3_pruning_mode="prune"` to make the engine respect
the flag and short-circuit.

This is the asymmetry Pearl & Halpern call out, and it's why the
same prompt yields different prose under
`query_type="intervention"` vs `query_type="counterfactual"`.

### 3.5 Forward propagation — `Impact > Inertia`

The core rule lives in
[`causal_physics.py::propagate`](../shadow_loom/causal_physics.py).
For each `mutation` edge `EVT_X → ENT_Y.trait`:

```text
impact  = causal_force × strength_weight(evidence_strength)
        × (1 + ambient_force_multiplier × ambient_value)   # if location matches
inertia = trait.inertia + inertia_epsilon
fired   = impact > inertia
```

Concrete Macbeth example — `EVT_BANQUO_GHOST → ENT_MACBETH.guilt`,
default `causal_force=4.0`, `evidence_strength="strong"` (weight 0.75),
fired in `LOC_DUNSINANE_CASTLE` (`tension=0.9`,
`ambient_force_multiplier=0.3`):

```text
impact = 4.0 × 0.75 × (1 + 0.3 × 0.9) = 3.81
guilt.inertia at t=11000 (from snapshot) = 0.50
3.81 > 0.50  →  edge fires
delta = sign(target - current) × min(impact_normalised, headroom)
```

The trait moves toward its targeted post-event value, but **bounded by
remaining headroom** (i.e. trait values are clamped to `[0, 1]`). The
`_intervened_traits` set is consulted first — if the user pinned
`guilt=0.0` in this query, the edge silently does nothing and the engine
records a `blocked` entry with reason `"trait_pinned_by_intervention"`.

The active-source seed is computed by `_seed_active_sources()`:

* User-intervened nodes (Rung 2)
* Abduction-mutated entities (Rung 3, via `_hidden_deltas`)
* Persistent ambient sources (`WorldTrait`, `Location`, `NarrativeObject`)

Crucially, *factual events that already realised their effects in the
state_timeline are not re-fired* — propagating them would double-count.
The engine relies on `reconstruct_entity_at()` for the historical state
and only forward-propagates actually-changed nodes.

### 3.6 Social propagation and belief revision

`propagate_social()` is the parallel pass for `RelationshipEdge` and the
channel/utterance graph. For each utterance event still active in the
sandbox:

1. Look up `via_channel_id` → channel's per-participant
   `intelligibility` map.
2. For each addressee + each eavesdropper above
   `physics.intelligibility_threshold`: compute the belief update.
3. `confidence_delta = intelligibility × strength_weight × truth_weight`
   (`truth_weight`: `true=1.0`, `mostly_true=0.7`, `false=0.0`,
   `performative=0.0`).
4. Update the listener's `Belief(target_id=evt.target_ids[0])`.
   Inertia damps the update the same way trait inertia damps mutation.

In Death on the Nile: when `EVT_UTT_LOUNGE_SHOOTING_STAGED` fires on
`CHN_LOUNGE_PUBLIC` (intelligibility ≈ 0.95 for all bystanders), every
non-Poirot entity's `Belief(target_id=EVT_JACQUELINE_SHOOTS_SIMON,
perceived_state="impulsive drunken rage")` lands at
`confidence ≈ 0.85`. Poirot is *also* a participant on the channel but
his prior `Belief(...)` had `inertia=0.65, confidence=0.55, perceived_state="
the shooting was staged"`, so the incoming evidence only nudges his
confidence to ~0.6 — not enough to flip the perceived state. The two
beliefs *coexist* in the world state, and that coexistence is what makes
the dramatic-irony score in §4.2 below meaningful.

---

## 4. Affective scoring with real numbers

All four structural scorers live in
[`directive_assembly.py`](../shadow_loom/directive_assembly.py) and share
the same signature: `(entity_ids, syuzhet_anchor) → float ∈ [0, 1]`. The
emotion scorers go through `compute_affective_score(target_effect, …)`
which reduces to per-trait closeness-to-target.

### 4.1 Mystery — Macbeth Act II

`compute_mystery_score` walks backward from each *revealed* effect
involving the target entities and returns

$$\text{mystery} = \frac{\sum_{a \in \text{hidden ancestors}} w(a)}{\sum_{a \in \text{ancestors}} w(a)}$$

where `w(a)` is the salience- and proximity-weighted reverse-path
strength from that ancestor (depth-capped at
`MYSTERY_PATH_DECAY_DEPTH=4`). Run on the bundled fixture with the
top-6 focal entities (Macbeth, Lady Macbeth, Macduff, Duncan,
Malcolm, Witches), the curve produced by
[`affective_timeseries_syuzhet`](../shadow_loom_ui/viz_helpers.py)
at 12 evenly-spaced anchors is the actual engine output:

```text
syuzhet anchor:    1     4     7     10    13    16    19    22    25    28    31    32
mystery score:   0.99  0.93  0.87  0.73  0.66  0.61  0.59  0.55  0.41  0.37  0.28  0.28
```

The gauge starts near $1.0$ (every ancestor of every revealed effect
is still hidden), collapses through Acts II–III as Banquo's death,
the banquet ghost, and the witches' second prophecy reveal earlier
hidden causes, and ends at $\approx 0.28$ once Macduff's family
slaughter and the moving forest have closed most of the structural
gaps. Reproduce via
[`scripts/_dump_scorer_components.py`](../scripts/_dump_scorer_components.py).

### 4.2 Dramatic irony — Death on the Nile

`compute_dramatic_irony_score` measures, per focal character, the
*fraction* of the reader's privileged view (the revealed-event mass)
that the character is in the dark about — a Sternberg-style gap
fraction normalised by the **revealed** event mass plus a saturation
constant ``K`` (= 1):

$$\text{irony}(t) = \frac{1}{|F|} \sum_{c \in F} \frac{\sum_{e \in R_t,\, e \notin K_c} w_e}{\sum_{e \in R_t} w_e + K}$$

where $R_t$ is the set of events revealed by syuzhet anchor $t$.
A character is treated as knowing event ``e`` when (a) they
participate in it as actor or target and ``e``'s ``fabula_time``
falls at or before the syuzhet anchor's fabula frontier, (b) a
revealed utterance addressed to (or spoken by) them references it,
or (c) they hold a ``Belief`` whose ``target_id`` matches ``e.id``
and whose provenance still resolves.

Anchored mid-story for the bundled `death_on_the_nile.py` fixture
with the top-6 focal entities (Simon, Jacqueline, Linnet, Poirot,
Race, Richetti), the engine returns the canonical rise-peak-fall arc
(curves from `scripts/_dump_scorer_components.py`):

```text
syuzhet anchor:    1     4     7     10    13    16    19    22    25    28    30
dramatic_irony:  0.14  0.30  0.26  0.40  0.48  0.58  0.47  0.42  0.37  0.50  0.28
                                                  ↑ peak
```

The peak at anchor 16 lands precisely as Poirot's deduction outpaces
the suspects' realisations; the curve falls through the denouement as
the killer is named and the remaining suspects' shock reveals close
their gaps. The same scorer on Macbeth peaks at $0.63$ around anchor
25 (Macduff's discovery of his murdered family); on Romeo and Juliet
at $0.72$ around anchor 25 (the crypt-misreading sequence); on Tinker
Tailor Soldier Spy at $0.73$ at anchor 41 (the cusp of the mole
reveal). 16 of 21 bundled fixtures produce a clean rise-peak-fall.

Why the formula looks like *that*. Two earlier denominators failed.
The *cumulative ratio over revealed-only edges* form
(``#gaps / #revealed_connections``) plateaued at a story-specific
asymptote by the third reveal because numerator and denominator grew
together — Reservoir Dogs *decayed* from 0.25 to 0.06 as the
protagonist became actor-of-record on more revealed edges. Replacing
it with the **full event mass** (a fixed denominator) instead pinned
the curve into a monotone rise across 21/21 example-world fixtures —
contradicting the rise-peak-fall theory predicts. The current
**revealed-mass + K** denominator restores the arc: it rises with
new reveals and falls when participation, addressed utterances, or
belief acquisition close the gap (Macduff hearing of his family,
Poirot's denouement, Nick's letter to Daisy), giving rise-peak-fall
in 16/21 worlds.

That mid-story value is what a directive of "raise dramatic_irony
to 0.85 in Act III without revealing the killer" is optimising
against — see §5.

### 4.3 Suspense — Romeo & Juliet's tomb

`compute_suspense_score` aggregates **unrevealed** events involving the
target entity and classifies them as *threat* (entity is target) or
*hope* (entity is actor). The current scorer combines the two sides as a
**balance × stakes** product over weighted event mass:

$$w_\text{threat} = \!\!\!\sum_{e \notin \text{rev},\, x \in F \cap \text{target}(e) \setminus \text{actor}(e)}\!\!\! p_e
\qquad
w_\text{hope}   = \!\!\!\sum_{e \notin \text{rev},\, x \in F \cap \text{actor}(e)}\!\!\! p_e$$

$$\text{balance} = 1 - \frac{|w_\text{threat} - w_\text{hope}|}{T}, \quad
  \text{stakes}  = \frac{T}{T + K}, \quad T = w_\text{threat} + w_\text{hope}$$

$$\text{suspense} = \mathrm{clip}_{[0,1]}\bigl(\text{balance} \cdot \text{stakes}\bigr)$$

…with the special cases **`w_hope = 0 ⇒ suspense = 0`** (despair, not
suspense — the Wilmot 2020 boundary) and **`w_threat = 0 ⇒ suspense = 0`**
(safety). The default `K = 2` is calibrated so two strong unrevealed
events on each side already register as fully high-stakes; this replaces
an earlier asymmetric `max(0, P(threat) - P(hope))` form that collapsed
to 0 on every fixture in which the protagonist authors most of their own
forward events.

For Romeo at the moment Juliet drinks the friar's potion
(`syuzhet_anchor` set at 21 from the bundled fixture, focal cast
`["ENT_ROMEO", "ENT_JULIET", "ENT_CAPULET", "ENT_TYBALT",
"ENT_FRIAR_LAURENCE", "EVT_JULIET_TAKES_POTION"]`), the engine
returns a suspense gauge of $0.237$ — the local maximum across the
fixture's 10-anchor curve `[0.14, 0.15, 0.10, 0.09, 0.14, 0.24, 0.12,
0.17, 0.17, 0.00]`. The terminal $0.00$ is Wilmot's safety condition:
no unrevealed threats remain after the tomb. The same gauge on
Gone Girl peaks $0.31$ at anchor 19 (Amy's mid-novel re-emergence),
on Tinker Tailor at $0.39$ on the cusp of the mole reveal, on Death
on the Nile at $0.28$ on Poirot's late-night confrontation.

### 4.4 Surprise — Reservoir Dogs reveal

`compute_surprise_score` models each trait as a Bernoulli variable and
computes binary KL divergence against a leave-one-out corpus-marginal prior updated by
revealed causal evidence:

$$D_\text{KL}(p \| q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$$

Anchored at `syuzhet_index` *just before* the
`EVT_ORANGE_REVEALED_AS_COP` reveal lands, for
`entity_ids=["ENT_ORANGE"]`:

```text
trait              actual   base_prior   posterior_prior  KL    1-exp(-KL)
loyalty            0.95     0.55         0.62 (+ revealed  0.18  0.165
                                              betrayal hint)
duplicity          0.90     0.30         0.34              0.36  0.302
courage            0.70     0.65         0.65              0.00  0.000
allegiance_police  0.95     0.05         0.10              1.21  0.702
                                                              ----
                                                surprise = avg ≈ 0.29
```

The per-trait KLs are run through a soft saturation ``1 - exp(-KL)``
so each contribution lands in ``[0, 1]`` and the per-trait mean is
the gauge value directly. The earlier ``avg(KL) / log(1/ε)`` form
divided by the *theoretical* binary-KL maximum (``≈ 4.605`` at
``ε = 0.01``) — squashing the entire perceptual signal into the
bottom 4% of the gauge. Every ``example_world`` plot read as flat
``≤ 0.10`` even when canonical surprise traits (Macbeth's despair,
Macduff's grief, Lady Macbeth's guilt) carried per-trait KLs of
``0.27–0.50``.

…then the reveal lands. The previously-hidden
`EVT_ORANGE_RECRUITED → ENT_ORANGE.allegiance_police` mutation enters
`revealed`, the prior snaps from 0.05 + small evidence delta straight to
~0.95, **and the same posterior** is now no surprise at all:
`surprise ≈ 0.0`. The big spike happens at the reveal *transition*: the
auditor compares the pre-anchor and post-anchor surprise scores and
returns the delta as the *earned* surprise. It is large because the
graph already had `EVT_ORANGE_RECRUITED` present at low fabula time but
locked behind high syuzhet — surprise is not a trick of the LLM.

The pre/post delta described above is exactly the **local** Bayesian-
Surprise quantity \(D_{\rm KL}(q_s\|q_{s-1})\) that the time-series
chart now computes per syuzhet step (after Itti & Baldi 2009; formally
identical to Storck/Hochreiter/Schmidhuber 1995 RDIA). The
directive-assembly optimiser instead consumes the *cumulative* form
\(D_{\rm KL}(p\|q_s)\) — the integrated gap between the reader's
accumulated prior and the truth — because its loss-function semantics
require a monotone "remaining gap" signal that falls as reveals close
it. Both are exposed by `compute_surprise_score(..., local=True/False)`;
see [academic-foundations.md §3.3](academic-foundations.md#33-surprise-as-kl-divergence)
for the full derivation.

### 4.5 Emotion targets — Gone Girl, grief and rage

For the six emotion targets (`grief, rage, joy, fear, love, regret`),
`compute_affective_score` uses *closeness to per-effect trait targets*
via a single shared ``_EFFECT_TRAITS`` table (see
[directive_assembly.py](../shadow_loom/directive_assembly.py)) that
maps each effect to ``(positive_indicators, inverse_indicators)``:

```python
_EFFECT_TRAITS = {
    "rage":  (["rage", "anger", "aggression", "vengefulness",
               "cruelty", "vindictiveness", "resentment",
               "rebelliousness", "volatility", "ruthlessness"],
              ["calm", "patience", "composure", "compassion",
               "warmth", "kindness", "tenderness", "restraint"]),
    "love":  (["love", "affection", "passion", "tenderness",
               "devotion", "warmth", "longing", "obsession",
               "constancy", "sensuality", "compassion", "kindness"],
              ["coldness", "cruelty", "anger", "resentment",
               "vindictiveness", "rage"]),
    ...
}
```

Positive traits contribute their current value, inverse traits
contribute one minus their current value; the score is the per-trait
mean. Both lists drive scoring symmetrically — the previous form's
``relevant`` filter only included the increase list, leaving the
decrease set as dead code (a brave character routed through ``fear``
got no fear-reducing credit because ``courage`` never entered the
average).

The vocabulary is calibrated against the actual trait names used in
``example_worlds/``. The earlier narrow synonym lists
(``["love", "affection", "sensuality"]`` for love;
``["happiness", "hope", "contentment"]`` for joy) matched almost no
real-world fixture: the corpus carries ``passion``, ``tenderness``,
``devotion``, ``warmth``, ``longing``, ``vengefulness``, ``cruelty``,
``vindictiveness`` — none of which appeared in the original maps.
Result: 50 of 96 (cast × emotion) combinations across the 16
fixtures collapsed silently to the worst-case ``+1.0`` fallback.
With the calibrated vocabulary that drops to 19/96, all of which are
genuinely traitless casts (``messianic_self_image``, ``moral_collapse``,
``class_anxiety``, ``pomposity`` …) where the worst-case fallback is
now the correct signal.

For Nick at the terminal syuzhet anchor of the bundled
`gone_girl.py` fixture, calling
`DirectiveAssembler(...).compute_affective_score` (the helper returns
the negated trait-distance, so the per-entity numbers below are the
sign-flipped engine output):

```text
entity            grief   rage    joy    regret   love    fear
ENT_NICK          -1.00   +0.60   -1.00  -1.00    +0.40   -1.00
ENT_AMY           -1.00   +0.80   -1.00  -1.00    +0.20   -1.00
```

Nick scores `+0.60` on rage (driven by `resentment=0.80`) and
`-1.00` on grief (no despair/shame/longing trait present in the
fixture's vocabulary triggers the worst-case fallback). Amy scores
`+0.80` on rage (vindictiveness, vengefulness, control all charge
the positive list). Across the four bundled emotion-heavy fixtures
(Macbeth, Gone Girl, Brief Encounter, Wuthering Heights) every
entity-emotion cell either lands on a real trait combination or on
the sentinel $-1.00$, so the directive assembler never silently
optimises against a placeholder.

Rage scores far better than grief at this anchor — and that match is
what selects between candidate continuations in the directive cycle.

---

## 5. Directive assembly — picking the best intervention

A `directive` query is the system's heaviest cycle. End-to-end on Romeo
and Juliet (see also [model-examples.md §4](model-examples.md#4-romeo-and-juliet--directives-and-the-affective-scorer)):

**User request** —

```python
from shadow_loom.query_models import DirectiveQuery

query = DirectiveQuery(
    original_query="At the tomb, push Romeo's dramatic irony to its peak.",
    target_entity_ids=["ENT_ROMEO"],
    target_effect="dramatic_irony",
    intensity=0.85,
    syuzhet_anchor=27,           # tomb scene
)
```

Directive queries are dispatched through `run_pipeline(world_state=...,
query=query)`; the `_QueryBase` carries `original_query` and the optional
`focus_entity_ids` / `syuzhet_anchor` / `fabula_anchor` envelope, while
the directive-specific fields above tell the assembler which entity to
optimise for and which effect to maximise. (Hard *must-not* clauses are
encoded as constraint blocks downstream during brief assembly, not as a
top-level query field.)

**1. Candidate enumeration** —
`DirectiveAssembler._enumerate_candidates()` walks the affordances of
objects/locations near the anchor, the unblocked spatial paths, and the
event-template library to produce e.g.:

```python
[
    {"EVT_LETTER_INTERCEPTED.event_type": "outcome",
     "EVT_LETTER_INTERCEPTED.fabula_time": 24500,
     "EVT_LETTER_INTERCEPTED.actor_ids": ["ENT_PLAGUE_GUARD"]},
    {"EVT_BALTHASAR_DEPARTS_EARLY.fabula_time": 24000},
    {"EVT_JULIET_WAKES_EARLY.fabula_time": 26500},
    {"EVT_FRIAR_DELAYED.event_type": "outcome",
     "EVT_FRIAR_DELAYED.fabula_time": 25500},
]
```

**2. Per-candidate fork-and-physics** —
`evaluate_candidate_events()` forks an AMWN sandbox per candidate (each
gets its own `world_id="shadow"` mirror), runs
`apply_do_operator(...)` with the candidate's interventions, calls
`propagate()`, and stores the resulting `CausalPhysicsResult`.

**3. Plausibility prune** — `_check_intervention_plausibility` drops
`EVT_BALTHASAR_DEPARTS_EARLY` because the plague-quarantine spatial
edges block the Mantua road below `fabula_time=24500`. The candidate is
returned with `plausible=False` and a `unresolved_targets` list.

**4. Affective score per survivor** — each surviving sandbox is fed
into `compute_affective_score(target_effect="dramatic_irony", …)`:

```text
candidate                          dramatic_irony    affective_score
EVT_LETTER_INTERCEPTED                0.91             -0.91
EVT_FRIAR_DELAYED                     0.78             -0.78
EVT_JULIET_WAKES_EARLY                0.62             -0.62
EVT_BALTHASAR_DEPARTS_EARLY           — plausibility-pruned —
```

Lowest score wins → `EVT_LETTER_INTERCEPTED`.

**5. Brief assembly** — the winner is wrapped in typed
`ConstraintBlock` entries (must / must-not / trait envelopes / edge
constraints / syuzhet-window constraints) and packaged as a
`CreativeBrief`. This is the **safety envelope** the LLM renderer must
stay inside.

The full per-candidate physics+score table is also surfaced to the UI's
**Causality** tab so the analyst can see *why* the engine picked the
winner — the alternatives are not silently discarded.

---

## 6. Generation — brief → constrained prose

[`generation.py::render_from_query`](../shadow_loom/generation.py) calls
the creative LLM with the `CreativeBrief` as a system prompt and a
minimal style context. The renderer is **not allowed to**:

* invent new causal edges,
* shift entity state outside the trait envelope,
* add an utterance through a channel that doesn't exist,
* introduce events outside the brief's syuzhet window.

It *is* free to choose dialogue, description, pacing, and POV. The
output is a `GeneratedScene` with a `narrative_text` field plus the
constraint annotations the renderer claims to have honoured.

For the Romeo & Juliet directive above, the brief constrains:

* `must`: `EVT_LETTER_INTERCEPTED` lands at `fabula_time=24500` between
  Mantua road and Friar Laurence's cell.
* `must_not`: any utterance event delivering Juliet's true status to
  Romeo (channel constraint: no addressee `ENT_ROMEO` for any utterance
  with `target_id` referencing `EVT_JULIET_FAKE_DEATH`).
* trait envelope: `ENT_ROMEO.despair ∈ [0.7, 1.0]` after the brief's
  anchor.
* syuzhet window: 26 ≤ syuzhet_index ≤ 30.

The renderer produces one or two paragraphs of tomb-scene prose; the
auditor in §7 checks that envelope.

---

## 7. Audit and refinement

[`auditor.py`](../shadow_loom/auditor.py) wraps generation in
`render_and_audit` (for directives with a pre-built brief) or
`run_feedback_loop` (otherwise). Three audits run in parallel:

* **Causal audit** — reverse-engineers the prose into causal claims,
  flags any "**Miracle Step**" (a state change in the prose with no
  licensing edge in the brief). For Romeo & Juliet this catches an LLM
  who tries to have a passing Capulet whisper Juliet's plan to Romeo —
  the channel doesn't exist in the brief, so the auditor returns
  `miracle_steps=[{"event": "implied_capulet_warning", "license": None}]`.
* **Abduction audit** — runs counterfactual probes on the prose. *"If
  Romeo had not bought the poison, would the prose still hold?"* The
  prose should fail this probe; if it doesn't, the prose is
  under-specifying Romeo's commitment.
* **Affective audit** — measures the *actual* dramatic-irony score in
  the prose against the brief's target. For this run: target 0.85,
  measured 0.83 → within tolerance, audit passes.
* **Style-fidelity audit** — runs when the ingested source carried a
  `NarrativeStyle` profile. Counts words, checks prose density
  (`sparse`/`moderate`/`rich`), register/POV/tense, and form class
  (e.g. `news_article` must read as inverted-pyramid reportage, not
  dramatised scene work). Off-target output raises `style_mismatch`.
* **Meta-narration audit** — runs on counterfactual and abduction
  scenes. Flags prose that stands outside the alternate world and
  comments on it ("the timeline fractures", "the alternative holds
  through sheer momentum", `If he had…/would have…` used to *describe*
  the branch rather than narrate it) instead of rendering it as plain
  past-tense events. Raises `meta_narration`.

If any audit fails, the loop regenerates (up to
`max_correction_retries`) with the auditor's feedback appended to the
brief. The loop returns a `FeedbackLoopResult`:

```python
FeedbackLoopResult(
    final_scene=GeneratedScene(narrative_text=..., constraints_honoured=[...]),
    converged=True,
    iterations=2,
    change_impact=...,
    engine_thresholds_passed=True,
    engine_threshold_failures=[],
)
```

The deterministic `engine_thresholds_passed` gate is **separate** from
the LLM auditor's verdict — the engine can refuse a scene the LLM
auditor approved if the measured affective score is more than
`physics.affective_tolerance` off the directive's target.

> **Loop discipline.** Style-fidelity violations are split into
> `critical` (form-class breach) / `major` (word count >±50% off-budget
> or density+form drift) / `minor` (pure density drift inside the
> form-class band). When *every* surfaced violation in an iteration is
> `minor`, the loop short-circuits to a pass instead of burning another
> regeneration. Each refinement call also receives a
> `=== NON-REGRESSION CONSTRAINTS ===` section listing prior fixes from
> earlier iterations so the rewriter cannot ping-pong between violation
> types. `meta` and `style` always run regardless of
> `brief.audit_categories`; the prompt instructs the judge to surface
> violations *only* for categories on the resolved list. When
> `affective_loss_mse` has no measurable target the evaluation prompt
> emits `not measured (no scorable target — ignore in evaluation)`
> instead of a misleading `0.0000`.

---

## 8. Re-extraction and merge

After audit, the prose is run through a *single-pass* extraction
([`pipeline.py` Step 6](../shadow_loom/pipeline.py)) that returns a
mini-`WorldStateV1` containing only the events / mutations / utterances
actually rendered. Step 7 then merges that mini-state into the live
`VersionedWorldModel` as a new version on the appropriate branch:

* `intervention` / `counterfactual` queries land on a `world_id="shadow"`
  branch with `branch_label` derived from the query.
* `directive` / `manual_edit` queries land on the canon `factual` branch
  (the directive cycle is meant to *advance the story*).
* `general` / `interrogate` queries are **read-only**: they never reach
  Step 7 at all (the pipeline early-returns after the answer step
  described in [pipeline-walkthrough.md §2.5](pipeline-walkthrough.md)),
  so no version is written.

The merge is journalled as a `MergeStepRecord` containing the diff
(events added, beliefs revised, traits shifted) so the analyst can
inspect or revert.

For the Romeo & Juliet run above, the merge adds:

* `EVT_LETTER_INTERCEPTED` (the new canonical event).
* `EVT_UTT_BALTHASAR_REPORTS_JULIETS_DEATH` (the utterance the renderer chose
  to dramatise Romeo's misinformation).
* `ENT_ROMEO.state_timeline += EntityStateSnapshot(despair=0.92, ...)`.
* A new `Belief(target_id=EVT_JULIET_FAKE_DEATH,
  perceived_state="Juliet is dead", confidence=0.95)` on `ENT_ROMEO`,
  acquired via `EVT_UTT_BALTHASAR_REPORTS_JULIETS_DEATH`.

Subsequent queries reason against this updated world; the loop is
closed.

---

## See also

* [architecture.md](architecture.md) — conceptual map and 12-step
  pipeline overview.
* [pipeline-walkthrough.md](pipeline-walkthrough.md) — code-level tour
  with no specific fixture.
* [model-examples.md](model-examples.md) — feature-by-feature tour using
  bundled plots.
* [query-and-cycles.md](query-and-cycles.md) — the eight query types
  and how natural language is parsed into them.
* [academic-foundations.md](academic-foundations.md) — the literature
  behind every named formula above (Pearl, Halpern, Wilmot, Itti & Baldi,
  Sternberg, Correa & Bareinboim).
