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

Macbeth, intervention query: *"Macbeth refuses to murder Duncan."*

```python
engine.apply_do_operator({"EVT_DUNCAN_MURDER.event_type": "prevented"})
```

What [`apply_do_operator`](../shadow_loom/causal_physics.py) does in
order:

1. Calls `AMWNInstantiator.execute_interventions(sandbox, ...)` — the
   destructive surgery runs on the *sandbox*, not the factual graph.
2. Records `_intervened_nodes = {"EVT_DUNCAN_MURDER"}` so propagation
   later treats that node as **frozen** (a downstream `mutation` edge
   cannot push it back to its factual value).
3. Records per-trait pins in `_intervened_traits` so a sibling
   intervention on `ENT_MACBETH.traits.guilt=0.0` would freeze only
   `guilt`, leaving `courage` free to evolve.
4. Calls `_collect_provenance_invalidations` — `event_type="prevented"`
   triggers removal of `EVT_DUNCAN_MURDER` from the
   *epistemically-active* event set, so the social-cascade step will
   not propagate beliefs about it.

After surgery, `propagate()` walks the causal sub-graph in topological
order. `EVT_GROOMS_BLAMED`, `EVT_MALCOLM_FLEES`, `EVT_MACBETH_CROWNED`,
`EVT_BANQUO_MURDERED` all lose activation pressure; the corresponding
`mutation` edges into `guilt` and `paranoia` never fire. The result is a
sandbox where Macbeth ends with:

```text
ambition     0.85     (the prophecy still landed)
guilt        0.10     (no murder → no shock)
paranoia     0.20     (initial value preserved)
ruthlessness 0.40     (initial — Lady Macbeth's persuasion still
                       fires but no act follows it)
status       healthy
```

### 3.4 Rung 3 — counterfactual (`abduction → do → propagate`)

Same Macbeth fixture, counterfactual query: *"Given that Macbeth did
become tyrant, what if the witches had never appeared on the heath?"*

The engine runs in three phases (Pearl 2009 §7):

**A. Abduction** —
[`CausalPhysicsEngine.abduction_update`](../shadow_loom/causal_physics.py)
back-propagates *every observed downstream* into the sandbox. Evidence
node `ENT_MACBETH` is in `factual.state_timeline` at t=10000 with
`paranoia=0.7`; the sandbox node's prior is `paranoia=0.2` (initial).
Under the default `abduction_blend_mode="bayesian"`:

```text
prior_value = 0.2,  precision (inertia) = 0.45
evidence    = 0.7,  ev_precision = 0.5      # settings.physics.abduction_evidence_precision
posterior   = (0.45 * 0.2 + 0.5 * 0.7) / (0.45 + 0.5)
            = (0.09 + 0.35) / 0.95
            ≈ 0.463
```

Every observed trait/belief gets the same precision-weighted blend; the
high-inertia `courage` shrinks toward its sandbox prior, the low-inertia
`guilt` snaps to the evidence. The deltas land in `_hidden_deltas` so the
next propagation pass treats them as *active sources*.

**B. Intervention** — `apply_do_operator({"EVT_WITCHES_PROPHECY_1.event_type": "prevented"})`.

**C. Propagation** — re-runs `propagate()`. The crucial behavioural
asymmetry vs Rung 2:

* Rung 2 reads "remove the witches and replay forward — Macbeth's
  ambition never spikes, the murder never happens, the crown never
  falls." The whole downstream collapses cleanly.
* Rung 3 has *already committed* to the observed downstream via
  abduction. Removing the witches forces the engine to find an
  *alternative cause* for the ambition shock or report that the outcome
  was over-determined. In Macbeth this surfaces as the engine flagging
  `WORLD_PROPHECY_VALIDITY` and `LOC_HEATH.supernatural` as *partial
  substitute causes* — the prophecy is the proximate trigger, but the
  ambient pressure plus Lady Macbeth's persuasion would still fire
  `EVT_DUNCAN_MURDER` with reduced (but non-zero) `causal_force`.

This is exactly the asymmetry Pearl & Halpern call out, and it's why the
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

where `w(a)` is the strongest single-edge weight from that ancestor.
Anchored at `syuzhet_index=14` (immediately after the discovery of
Duncan's body) for `entity_ids=["ENT_DUNCAN", "ENT_MACBETH"]`:

* Revealed effect node: `EVT_DUNCAN_DISCOVERED_MURDERED` (syuzhet 12).
* Ancestors: `EVT_DUNCAN_MURDER`, `EVT_LADY_MACBETH_PERSUADES`,
  `EVT_WITCHES_PROPHECY_1`, `EVT_REBELLION_DEFEATED`,
  `WORLD_PROPHECY_VALIDITY`, `LOC_INVERNESS_CASTLE`.
* Hidden at this anchor (not yet in `revealed`): `EVT_DUNCAN_MURDER`
  itself (the on-stage murder happens off-stage, only its discovery is
  revealed), `EVT_LADY_MACBETH_PERSUADES`, `EVT_WITCHES_PROPHECY_1`.
* Weights: ≈ `0.75 + 0.5 + 0.75 = 2.0` hidden vs `2.0 + 0.5 + 0.5 = 3.0`
  total → `mystery ≈ 0.67`.

The "who actually did this and why" is exactly what the score is
measuring — and it climbs sharply at exactly the syuzhet point the play
intends.

### 4.2 Dramatic irony — Death on the Nile

`compute_dramatic_irony_score` counts revealed causal edges into the
target entity whose **source is unknown to that character**. Anchored at
`syuzhet_index=N` (just after `EVT_LINNET_SHOT`) for
`entity_ids=["ENT_PENNINGTON", "ENT_VAN_SCHUYLER", "ENT_ALLERTON"]`:

```text
total_connections = 8       (revealed causal edges into the cast)
irony_gaps        = 6       (events the reader has seen via Poirot's
                             POV but the suspect-pool has not)
score             = 6 / 8 = 0.75
```

That `0.75` is what a directive of "raise dramatic_irony to 0.85 in
Act III without revealing the killer" is optimising against — see §5.

### 4.3 Suspense — Romeo & Juliet's tomb

`compute_suspense_score` aggregates **unrevealed** events involving the
target entity and classifies them as *threat* (entity is target) or
*hope* (entity is actor). With **noisy-OR** combination:

$$P(\text{threat}) = 1 - \prod_i (1 - p_i^{\text{threat}}) \qquad P(\text{hope}) = 1 - \prod_i (1 - p_i^{\text{hope}})$$

$$\text{suspense} = \max(0, P(\text{threat}) - P(\text{hope}))$$

…with the special case **`hope_prob ≤ 0 ⇒ suspense = 0`** (despair, not
suspense — the Wilmot 2020 boundary).

For Romeo at the moment Juliet drinks the friar's potion
(`syuzhet_anchor` set just before the tomb scene),
`entity_ids=["ENT_ROMEO"]`:

* Unrevealed events with Romeo as **target**: `EVT_FRIAR_LETTER_LOST`
  (p=0.6), `EVT_BALTHASAR_REPORTS_DEATH` (p=0.8), `EVT_ROMEO_BUYS_POISON`
  (p=0.7), `EVT_ROMEO_DIES` (p=0.85).
* Unrevealed events with Romeo as **actor**: `EVT_ROMEO_RECONCILES`
  (p=0.2 — there is exactly one tenuous "reconciliation" path the
  fixture leaves alive).

```text
P(threat) = 1 - (0.4 × 0.2 × 0.3 × 0.15) ≈ 1 - 0.0036 ≈ 0.996
P(hope)   = 1 - (0.8)                             = 0.20
suspense  = 0.996 - 0.20                          ≈ 0.80
```

…which is exactly the regime the directive assembler will hand the
Friar's-letter-intercepted candidate (see §5) to push above 0.85.

### 4.4 Surprise — Reservoir Dogs reveal

`compute_surprise_score` models each trait as a Bernoulli variable and
computes binary KL divergence against a corpus-marginal prior updated by
revealed causal evidence:

$$D_\text{KL}(p \| q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$$

Anchored at `syuzhet_index` *just before* the
`EVT_ORANGE_REVEALED_AS_COP` reveal lands, for
`entity_ids=["ENT_ORANGE"]`:

```text
trait              actual   base_prior   posterior_prior  KL_contribution
loyalty            0.95     0.55         0.62 (+ revealed   0.18
                                              betrayal hint)
duplicity          0.90     0.30         0.34              0.36
courage            0.70     0.65         0.65              0.00
allegiance_police  0.95     0.05         0.10              1.21
                                                          ----
                                                     avg ≈ 0.44
max_kl = log(1 / 0.01) ≈ 4.605
surprise = min(0.44 / 4.605, 1.0) ≈ 0.10
```

…then the reveal lands. The previously-hidden
`EVT_ORANGE_RECRUITED → ENT_ORANGE.allegiance_police` mutation enters
`revealed`, the prior snaps from 0.05 + small evidence delta straight to
~0.95, **and the same posterior** is now no surprise at all:
`surprise ≈ 0.0`. The big spike happens at the reveal *transition*: the
auditor compares the pre-anchor and post-anchor surprise scores and
returns the delta as the *earned* surprise. It is large because the
graph already had `EVT_ORANGE_RECRUITED` present at low fabula time but
locked behind high syuzhet — surprise is not a trick of the LLM.

### 4.5 Emotion targets — Gone Girl, grief and rage

For the six emotion targets (`grief, rage, joy, fear, love, regret`),
`compute_affective_score` uses *closeness to per-effect trait targets*
(see [directive_assembly.py:1095+](../shadow_loom/directive_assembly.py)):

```python
_EFFECT_TRAIT_MAP = {
    "grief":  ["despair", "love", "hope"],
    "rage":   ["anger", "rebelliousness", "resentment"],
    ...
}
_EFFECT_DECREASE = {
    "grief":  {"hope", "happiness", "contentment"},
    ...
}
```

For Nick at the syuzhet point of Amy's televised "rescue" return,
`target_effect="rage"`, `entity_ids=["ENT_NICK"]`:

```text
trajectory     trait                current_value    in decrease set?
               anger                0.85             no   → 0.85
               resentment           0.80             no   → 0.80
               rebelliousness       0.20             no   → 0.20
avg_match = (0.85 + 0.80 + 0.20) / 3 ≈ 0.617
score = -0.617    (negative = strong match)
```

Compare with `target_effect="grief"`:

```text
trait    current_value     decrease?    contribution
despair  0.55              no            0.55
love     0.10              no            0.10
hope     0.20              YES           1 - 0.20 = 0.80
avg_match = 0.483    →    score = -0.483
```

Rage scores better than grief at this anchor — and that match is what
selects between candidate continuations in the directive cycle.

---

## 5. Directive assembly — picking the best intervention

A `directive` query is the system's heaviest cycle. End-to-end on Romeo
and Juliet (see also [model-examples.md §4](model-examples.md#4-romeo-and-juliet--directives-and-the-affective-scorer)):

**User request** —

```python
UserRequest(
    query_type="directive",
    target_effect="dramatic_irony",
    target_value=0.85,
    entity_ids=["ENT_ROMEO"],
    syuzhet_anchor=27,           # tomb scene
    constraint="no character may receive correct information about Juliet",
)
```

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

---

## 8. Re-extraction and merge

After audit, the prose is run through a *single-pass* extraction
([`pipeline.py` Step 6](../shadow_loom/pipeline.py)) that returns a
mini-`WorldStateV1` containing only the events / mutations / utterances
actually rendered. Step 7 then merges that mini-state into the live
`VersionedWorldModel` as a new version on the appropriate branch:

* `intervention` / `counterfactual` queries land on a `world_id="shadow"`
  branch with `branch_label` derived from the query.
* `directive` / `general` / `manual_edit` queries land on the canon
  `factual` branch (the directive cycle is meant to *advance the story*).

The merge is journalled as a `MergeStepRecord` containing the diff
(events added, beliefs revised, traits shifted) so the analyst can
inspect or revert.

For the Romeo & Juliet run above, the merge adds:

* `EVT_LETTER_INTERCEPTED` (the new canonical event).
* `EVT_UTT_BALTHASAR_REPORTS_DEATH` (the utterance the renderer chose
  to dramatise Romeo's misinformation).
* `ENT_ROMEO.state_timeline += EntityStateSnapshot(despair=0.92, ...)`.
* A new `Belief(target_id=EVT_JULIET_FAKE_DEATH,
  perceived_state="Juliet is dead", confidence=0.95)` on `ENT_ROMEO`,
  acquired via `EVT_UTT_BALTHASAR_REPORTS_DEATH`.

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
