# Model Examples — What Shadow-Loom Actually Does

This document walks through Shadow-Loom on **real story plots** rather than
toy fixtures. Each section picks one of the bundled
[`example_worlds/`](../example_worlds) / [`sample_plots/`](../sample_plots)
pairs and uses it to illustrate one or two parts of the engine — the data
model, causal physics, narrative physics, the affective scorer, generation,
or the audit/feedback loop.

It is meant to answer the question: *"OK, but what does this thing
actually do when you point it at a story?"*

For the conceptual map see [architecture.md](architecture.md); for the
code-level pipeline tour see
[pipeline-walkthrough.md](pipeline-walkthrough.md); for the per-query
mechanics see [query-and-cycles.md](query-and-cycles.md). This document
links into all three rather than re-explaining them.

> **How to follow along.** Every example here corresponds to a fixture
> file under [`example_worlds/`](../example_worlds) and a synopsis under
> [`sample_plots/`](../sample_plots). The fixtures are hand-authored
> against the live ingestion schema, so you can load any of them in the
> UI's example seeder, the MCP `seed_example_world` tool, or directly via
> `from example_worlds.macbeth import world_state` and run the same
> queries shown below.

---

## Index

| # | Example | Spotlights |
|---|---|---|
| 1 | [Macbeth](#1-macbeth--causal-physics-and-counterfactuals) | Edge taxonomy, intervention vs counterfactual, world-trait priors |
| 2 | [Death on the Nile](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony) | Beliefs, Channels, intelligibility, dramatic-irony scoring |
| 3 | [Reservoir Dogs](#3-reservoir-dogs--syuzhet-vs-fabula-and-information-flow) | Non-linear syuzhet, undercover identity, surprise as KL |
| 4 | [Romeo and Juliet](#4-romeo-and-juliet--directives-and-the-affective-scorer) | Directive enumeration, plausibility gate, brief assembly |
| 5 | [Gone Girl](#5-gone-girl--false-beliefs-and-the-audit-loop) | Truth-value utterances, miracle-step audit, refinement loop |
| 6 | [1984](#6-1984--global-traits-and-ambient-propagation) | World traits, regime as common cause, ambient edges |
| 7 | [A Fish Called Wanda](#7-a-fish-called-wanda--objects-affordances-and-chain-reactions) | Objects, affordance gates, chain-reaction cascades |
| 8 | [Frankenstein](#8-frankenstein--social-mutation-and-relationship-decay) | `mutation_social`, per-axis `RelationshipMetric`, inertia |

A consolidated **feature-to-example matrix** lives at the
[end of this document](#feature-to-example-matrix).

---

## 1. Macbeth — causal physics and counterfactuals

**Fixture:** [`example_worlds/macbeth.py`](../example_worlds/macbeth.py) ·
**Plot:** [`sample_plots/macbeth.txt`](../sample_plots/macbeth.txt)

Macbeth is the canonical fixture for the **causal layer** because the play
explicitly chains supernatural prophecy → ambition → murder → guilt →
paranoia → tyranny → downfall. Every link is a different `CausalEdge`
modality.

### What the data model captures

Looking at `world_state.entities["ENT_MACBETH"]` you can see the engine's
per-entity state primitives at work:

* **`TraitVector`** — `ambition`, `courage`, `loyalty`, `guilt`,
  `paranoia`, `ruthlessness`, `despair`, each with `value`, `inertia` and
  `evidence_strength`. Inertia bands matter: `courage` starts at `0.7`
  inertia (hard to shake), `guilt` starts at `0.25` (highly mutable). When
  a trait is shocked away from baseline (e.g. guilt jumping after Duncan's
  murder), the next `EntityStateSnapshot` *bumps inertia up* — the trait
  becomes harder to wash off, which is exactly the play's psychology.
* **`Belief`** — Macbeth begins with `confidence=0.4` in the witches'
  prophecy. Beliefs carry their own inertia and evidence strength so the
  causal engine can update them probabilistically rather than as boolean
  flags.
* **`state_timeline`** — an ordered list of `EntityStateSnapshot`s anchored
  to `triggered_by` event IDs. This is the **Hybrid 4+5 timeline**:
  current state lives on the entity, but every change is journalled with
  the event that caused it, so the causal engine can replay or rewind.

### Edge modalities — all five visible in one fixture

The Macbeth fixture deliberately exercises every causal modality the
engine knows about (see [architecture.md §3](architecture.md)):

| Edge | Macbeth example | Engine method |
|---|---|---|
| `chain_reaction` | `EVT_DUNCAN_MURDER → EVT_MALCOLM_FLEES` | `propagate()` activation |
| `mutation` | `EVT_BANQUO_GHOST → ENT_MACBETH.guilt += …` | `propagate()` trait shock |
| `mutation_social` | `EVT_LADY_MACBETH_PERSUADES → REL(MACBETH, LADY).power_dynamic` | `propagate_social()` |
| `affordance_gate` | `OBJ_BLOODY_DAGGERS.affordance("kill")` gates `EVT_DUNCAN_MURDER` | gate check inside `propagate()` |
| `ambient_propagation` | `LOC_HEATH.supernatural=0.9` boosts prophecy belief uptake | per-tick ambient sweep |

### Rung 2 vs Rung 3 — intervention and counterfactual

This is where the causal engine earns its name (see
[`shadow_loom/causal_physics.py::CausalPhysicsEngine`](../shadow_loom/causal_physics.py)
and [academic-foundations.md §2](academic-foundations.md)).

**Intervention (Pearl rung 2):**

> "Macbeth refuses to murder Duncan."

The router in
[`narrative_physics.py::calculate_narrative_physics`](../shadow_loom/narrative_physics.py)
calls `_check_intervention_plausibility` — Macbeth is alive, the daggers
exist, the affordance is satisfied, so the intervention is *plausible to
withhold*. It then forks an AMWN sandbox, calls
`apply_do_operator({"EVT_DUNCAN_MURDER": "⊘"})`, and runs `propagate()`
forward. The downstream `chain_reaction` edges into `EVT_MALCOLM_FLEES`,
`EVT_MACBETH_CROWNED`, `EVT_BANQUO_MURDERED` all lose activation pressure;
the corresponding `mutation` edges into `guilt` and `paranoia` never fire.

**Counterfactual (Pearl rung 3):**

> "Given that Macbeth *did* become tyrant, what if the witches had never
> appeared on the heath?"

Same fork, but now the sandbox runs `abduction_update` first to fix
*everything Macbeth observed* (Duncan dead, crown taken, Banquo's ghost)
as evidence, then applies `do(EVT_WITCHES_PROPHECY_1 = ⊘)`, then
re-propagates. The counterfactual answer is much weaker than the
intervention answer because abduction has already *committed* to the
observed downstream — the engine must find an alternative cause for the
ambition shock or report that Macbeth's tyranny was over-determined. This
is exactly the asymmetry Pearl & Halpern call out, and it's why the same
"what if" prompt produces different prose under
`query_type="intervention"` vs `query_type="counterfactual"`.

Under default `branch_policy="auto"` the counterfactual run lands on a
**shadow branch** with its own `world_id`, so the canon Macbeth graph is
never polluted; the analyst can diff the shadow against canon in the UI's
**Causality** tab and, if convinced, promote it via
`db.promote_branch`.

---

## 2. Death on the Nile — epistemic physics and dramatic irony

**Fixture:** [`example_worlds/death_on_the_nile.py`](../example_worlds/death_on_the_nile.py) ·
**Plot:** [`sample_plots/death_on_the_nile.txt`](../sample_plots/death_on_the_nile.txt)

A Christie mystery is the cleanest possible test of Shadow-Loom's
**epistemic layer** — who knows what, when did they learn it, through
which channel, and how confident are they?

### Beliefs as first-class objects

Each suspect's `beliefs` list contains `Belief` records keyed by
`target_id`. For example, after the staged shooting in the lounge,
`ENT_FANTHORP` holds:

```python
Belief(
    target_id="EVT_JACQUELINE_SHOOTS_SIMON",
    perceived_state="Jacqueline impulsively shot Simon in a drunken rage",
    confidence=0.9,
    inertia=0.6,
    evidence_strength="strong",
    acquired_via_event_id="EVT_LOUNGE_SHOOTING",
)
```

…while `ENT_POIROT` holds the same `target_id` with `perceived_state`
"the shooting was staged" and `confidence=0.55` building toward `0.95`
across the timeline. The two beliefs *coexist* in the world state — there
is no single "fact" the engine reasons over. This is what makes the
mystery tractable: the auditor measures the **gap** between what the
reader knows and what each character knows, rather than treating the
narrator as omniscient.

### Channels and per-participant intelligibility

Conversations and reveals are **`Channel`** nodes (see post-F18 design,
S1–S11). Each channel has a participants set and a per-participant
`intelligibility` map (`0.0`–`1.0`) — the fraction of an utterance that a
given participant actually decodes. In the Nile fixture:

* `CHN_LOUNGE_PUBLIC` — high intelligibility for all bystanders.
* `CHN_POIROT_LOUISE_PRIVATE` — Louise's hint is `intelligibility=0.4`
  for Poirot (she's deliberately oblique) and `1.0` for herself.
* `CHN_SIMON_JACQUELINE_PRIVATE` — Simon's shouted warning to Jacqueline
  through the cabin wall has `intelligibility=0.95` for Jacqueline and
  `0.05` for Mrs Otterbourne, who is *meant to overhear* but doesn't
  decode the signalling intent.

Utterance `EventNode`s carry `speaker_id`, `addressee_ids`,
`via_channel_id`, `truth_value` and `content`; the social cascade in
`propagate_social()` walks the channel graph, weights the belief update
by intelligibility × evidence_strength, and writes the result to each
participant's `Belief` set. Simon's shouted lie that Mrs Otterbourne
"saw who killed the maid" is a `truth_value="false"` utterance that
nevertheless updates Jacqueline's belief about the threat — exactly
because beliefs are about *perceived* state, not truth.

### Affective payoff — dramatic irony as epistemic asymmetry

Run `query_type="evaluate"` against the Nile world after Act IV and the
auditor's `compute_affective_feedback` returns a high
`dramatic_irony_score`: the reader's posterior over "who killed Linnet"
has collapsed to Simon+Jacqueline, while the on-graph beliefs of
Pennington, Van Schuyler, the Allertons et al still spread mass across
multiple suspects. The score is literally
`KL(reader_posterior || character_posterior)` averaged over POVs —
see [academic-foundations.md §3.4](academic-foundations.md#34-dramatic-irony-as-epistemic-asymmetry).

This is also why the **"raise dramatic irony to 0.8 without revealing the
killer"** directive is well-defined on Christie material: the directive
assembler can pick interventions that move the reader's posterior
(through utterance ordering on `syuzhet_index`) without touching any
character's belief set.

---

## 3. Reservoir Dogs — syuzhet vs fabula and information flow

**Fixture:** [`example_worlds/reservoir_dogs.py`](../example_worlds/reservoir_dogs.py) ·
**Plot:** [`sample_plots/reservoir_dogs.txt`](../sample_plots/reservoir_dogs.txt)

Reservoir Dogs is the system's **non-linear narration** test fixture.
The events in the diner happen *before* the heist, the heist itself
happens off-camera, and Mr Orange's identity reveal lands long after
the events that established it. Two clocks are needed — and Shadow-Loom
keeps them strictly separate.

### Two times, one graph

Every `EventNode` carries:

* `fabula_time` — *when in the story-world it happened* (integer
  monotonic per causal chain; ingestion spaces them by 1000, the engine
  by 100).
* `syuzhet_index` — *when the reader / viewer learns about it*
  (contiguous, dense, per-channel-of-narration).

Reservoir Dogs has `EVT_ORANGE_RECRUITED` at `fabula_time ≈ 1000` but
`syuzhet_index ≈ 42` — long after the warehouse standoff. The causal
engine *only* uses `fabula_time` for propagation; the auditor and the
affective scorer use `syuzhet_index` to compute reader-side surprise.
This separation means a flashback never accidentally rewires causality,
and a chronological replay never destroys the suspense the original
ordering creates.

### Surprise as KL divergence on reveal

When `EVT_ORANGE_REVEALED_AS_COP` fires at the high-syuzhet end,
`compute_affective_feedback` measures the reader's prior posterior over
"Orange's role" *before* the reveal vs the delta-spike *after*, and
returns the surprise as `KL(P_after || P_before)` (see
[academic-foundations.md §3.3](academic-foundations.md#33-surprise-as-kl-divergence)).
Because the graph already has `EVT_ORANGE_RECRUITED` and Nash's
"recognized Orange" utterance present at low fabula time but locked
behind high syuzhet, the surprise is *earned* — it is large, but it does
not violate causality.

This is also where **information physics** meets **causal physics**:
Orange's `Belief(target_id=ENT_WHITE, perceived_state="trusts me")` has
been climbing the whole film via `propagate_social()`. The reveal
doesn't change the past — it changes which `Channel`s now back-fill
their utterances into White's belief set, which is what makes the final
shot land.

---

## 4. Romeo and Juliet — directives and the affective scorer

**Fixture:** [`example_worlds/romeo_and_juliet.py`](../example_worlds/romeo_and_juliet.py) ·
**Plot:** [`sample_plots/romeo_and_juliet.txt`](../sample_plots/romeo_and_juliet.txt)

A `directive` query is the system's most sophisticated cycle: the user
asks for an *outcome* ("raise tension to 0.8", "kill off a minor
character without violating affordances", "introduce a misunderstanding
between X and Y") and the engine has to **enumerate candidate events**,
fork a sandbox per candidate, run physics, score with the affective
auditor, and return the survivor as a `CreativeBrief`. Shakespeare's
play is the cleanest test: every catastrophe is one missed message away.

### The directive cycle on this fixture

1. **User request** — `directive: "raise dramatic_irony to 0.85 in the
   tomb scene without giving Romeo correct information"`.
2. **`DirectiveAssembler.evaluate_candidate_events()`** enumerates
   candidates from the affordances and unblocked affordance gates of
   surrounding objects/locations: *Friar's letter is intercepted*,
   *Balthasar arrives faster*, *Juliet wakes one minute earlier*, …
3. For each candidate, `narrative_physics` forks an AMWN sandbox, runs
   `apply_do_operator` with the candidate event, and propagates. The
   plausibility gate (`_check_intervention_plausibility`) drops
   candidates that violate spatial reachability (Balthasar can't reach
   Mantua faster than the plague quarantine permits) or affordance
   constraints.
4. The survivors are scored by `compute_affective_feedback`. In Romeo
   and Juliet "Friar's letter intercepted" wins because it raises the
   reader's `KL(reader_posterior || romeo_posterior)` (irony) without
   adding any utterance event into Romeo's belief set.
5. The winner is wrapped in typed `ConstraintBlock`s — *must* events,
   *must-not* events, trait envelopes, edge constraints, syuzhet-window
   constraints — and passed to generation.

The brief is now a **safety envelope around the LLM**: the renderer can
write any prose it likes about the tomb, the messenger, the dagger, but
it cannot invent a causal edge, shift a trait outside its envelope, or
add an utterance through a channel that doesn't exist. This is the heart
of the constrained-generation approach — the LLM is a *renderer*, not a
simulator.

---

## 5. Gone Girl — false beliefs and the audit loop

**Fixture:** [`example_worlds/gone_girl.py`](../example_worlds/gone_girl.py) ·
**Plot:** [`sample_plots/gone_girl.txt`](../sample_plots/gone_girl.txt)

Gone Girl is the system's **lying narrator** fixture. Amy's diary is a
sequence of `EventNode`s with `truth_value="false"` propagated through
`CHN_DIARY_PUBLIC` to the police, the public, and (deliberately) to
Nick's belief set. The detective work is the *reader's* job, but
Shadow-Loom has to keep the fiction internally consistent.

### What the auditor catches

After generation, three audits run in parallel
(`shadow_loom/auditor.py::run_feedback_loop`):

* **Causal audit** — reverse-engineers prose into `(source, target,
  modality)` causal claims and matches them against the brief. Any
  state change in the prose that has no licensing edge in the brief is
  a **"miracle step"** and the loop regenerates. On Gone Girl this
  catches the LLM's frequent failure mode of having Nick "realise"
  something he was never given evidence for.
* **Abduction audit** — runs counterfactual probes. *"If the diary were
  truthful, would the prose still hold together?"* If yes, the prose
  has accidentally collapsed Amy's two faces into one — fail.
* **Affective audit** — measures the actual `dramatic_irony_score` and
  `suspense` in the prose vs the directive's target. On Gone Girl the
  target is *high* irony (reader knows Amy is alive long before Nick),
  and the auditor flags drift back toward neutral.

### The refinement loop in practice

`FeedbackLoopResult.iterations` typically settles at 1–3 on this
fixture. Each iteration:

1. Generation produces prose under the brief.
2. The three audits return `CausalPhysicsFeedback`,
   `AffectiveStateFeedback`, and an LLM literary critique.
3. `compute_overall_pass` checks `engine_thresholds_passed` (the
   deterministic gate — number of miracle steps, max trait drift,
   minimum affective deviation) *separately from* the LLM auditor's
   verdict. Either gate failing triggers regeneration with the
   feedback appended to the brief.
4. On convergence (`engine_thresholds_passed AND llm_pass`), the
   `final_scene` is written to `result.scene` and the prose moves to
   re-extraction (Step 6).

The deterministic gate is what stops the loop on adversarial cases
where the LLM auditor would happily approve plausible-sounding but
miracle-laden prose. The LLM gate is what stops it on cases where the
deterministic checks pass but the prose is dramatically inert.

---

## 6. 1984 — global traits and ambient propagation

**Fixture:** [`example_worlds/nineteen_eighty_four.py`](../example_worlds/nineteen_eighty_four.py) ·
**Plot:** [`sample_plots/nineteen_eighty_four.txt`](../sample_plots/nineteen_eighty_four.txt)

In *1984* the regime is the protagonist of every scene without ever
appearing in one. Shadow-Loom models this as a **`GlobalTrait`**
(`WORLD_PARTY_SURVEILLANCE`, `WORLD_DOUBLETHINK_PRESSURE`,
`WORLD_MATERIAL_SCARCITY`) with its own `state_timeline` of
`WorldTraitSnapshot`s — i.e. the regime *also* has inertia and
inflection points (the Two Minutes Hate intensifies surveillance, the
chocolate ration cut intensifies scarcity).

### World traits as common-cause parents

In the fixture, `WORLD_PARTY_SURVEILLANCE` is wired as a common-cause
parent to many `EventNode`s: every time Winston fails to mask a
microexpression in front of a telescreen, the engine treats the regime
trait as the *exogenous source* of that observation event. This matters
for counterfactuals: `do(WORLD_PARTY_SURVEILLANCE = 0.1)` is a
*regime-change* counterfactual and the engine correctly cascades it to
hundreds of downstream events without the LLM having to re-imagine each
one.

### Ambient edges — the weather of a scene

Every `Location` has an `ambient_state` (`fear`, `tension`,
`supervision`, `concealment`, …) and `CausalEdge(modality=
"ambient_propagation")` lets a high-ambient location *bias* the trait
shifts of any entity present. In *1984* the canteen has
`supervision=0.85, concealment=0.05`; Victory Mansions has
`supervision=0.4, concealment=0.7`. Winston's `paranoia` trait climbs
faster per fabula tick in the canteen than at home — same agent, same
narrative beat, different ambient. The renderer is told this in the
brief as *"location prior on `paranoia` for ENT_WINSTON: +0.15
(canteen, supervision=0.85)"* and the prose is constrained to honour
that bias.

This is also how the system models *atmosphere* without anthropomorphic
hand-waving: ambient propagation is just a per-tick scalar contribution
to the noisy-OR aggregation in
[`causal_physics.py::_noisy_or_aggregate`](../shadow_loom/causal_physics.py).

---

## 7. A Fish Called Wanda — objects, affordances, and chain reactions

**Fixture:** [`example_worlds/a_fish_called_wanda.py`](../example_worlds/a_fish_called_wanda.py) ·
**Plot:** [`sample_plots/a_fish_called_wanda.txt`](../sample_plots/a_fish_called_wanda.txt)

Farce is a stress test for the **object layer**. Every gag is a
chain reaction gated on whether someone has an object, whether the
object affords an action, and whether the right entities are in the
right place at the right time.

### Affordance gates — "you cannot stab without a knife"

`NarrativeObject` carries an explicit `affordances` list:

```python
NarrativeObject(
    id="OBJ_KEN_FISH_TANK", name="Ken's Fish Tank",
    affordances=[
        Affordance(action="house_pets", target_type="Object"),
        Affordance(action="be_swallowed_from", target_type="Entity"),
    ],
)
```

A `CausalEdge(modality="affordance_gate")` from
`OBJ_KEN_FISH_TANK → EVT_OTTO_EATS_FISH` *gates* the event: it only
fires if Otto and the tank are in the same location *and* the tank
still has the `be_swallowed_from` affordance *and* the fish are
present. The gate is enforced inside `propagate()` and a blocked
gate produces a `BlockedPropagation` record that surfaces in the
brief as a *must-not* constraint.

### Chain reactions across multiple gags

The Wanda fixture chains three affordance gates per scene: getting
the safe-deposit box → identifying Wanda's accent → tracing the
diamonds. Run `query_type="intervention"` with `do(EVT_KEN_LOSES_FISH
= ⊘)` and the engine walks the chain backwards, identifying *every*
downstream gag that depended on the lost fish through `chain_reaction`
edges. The output is a clean dependency tree the author can browse in
the UI's **Causality** tab — useful for screenwriters debugging which
beats can be cut without unravelling the third act.

### Why this matters for generation

Without affordance gates the renderer routinely invents props ("Otto
pulls out a knife and…") that have no licensing object in the world
state. The brief includes the *available affordances at this fabula
tick*, the auditor's miracle-step check rejects any state change that
required an absent affordance, and the loop regenerates. The end
result: the LLM can riff on dialogue and beats freely, but it cannot
spawn objects out of thin air.

---

## 8. Frankenstein — social mutation and relationship decay

**Fixture:** [`example_worlds/frankenstein.py`](../example_worlds/frankenstein.py) ·
**Plot:** [`sample_plots/frankenstein.txt`](../sample_plots/frankenstein.txt)

Frankenstein is the **`mutation_social`** showcase. The
Creator-Creature bond doesn't just shift one trait — it shifts an
entire `RelationshipMetric` along multiple axes (affinity, fear,
power_dynamic, obligation), each with its own `evidence_strength` and
inertia.

### Per-axis relationship metrics

```python
RelationshipEdge(
    source_id="ENT_FRANKENSTEIN", target_id="ENT_CREATURE",
    metrics={
        "affinity":      RelationshipMetric(value=-0.8, inertia=0.6, evidence_strength="strong"),
        "fear":          RelationshipMetric(value= 0.7, inertia=0.5, evidence_strength="strong"),
        "obligation":    RelationshipMetric(value= 0.4, inertia=0.4, evidence_strength="moderate"),
        "power_dynamic": RelationshipMetric(value= 0.3, inertia=0.5, evidence_strength="moderate"),
    },
)
```

Only *observed* axes are populated; the engine never invents a metric
just to fill the schema. When `EVT_CREATURE_KILLS_WILLIAM` fires, the
attached `mutation_social` edges target *specific axes* — affinity
plummets, fear spikes, obligation ticks up (Frankenstein now feels
responsible for the destruction). `propagate_social()` writes each
delta back to the relationship metric, with the per-axis inertia
controlling how much of the shock survives the next tick.

### Bidirectional asymmetry — the Creature loves what hates him

Two `RelationshipEdge`s exist between every dyad — Frankenstein →
Creature and Creature → Frankenstein. They evolve independently. The
Creature's `affinity` toward Frankenstein stays positive long after
Frankenstein's collapses to negative; the social cascade therefore
correctly produces *unilateral pursuit* rather than mutual hatred.
This asymmetry is what generates the novel's tragic structure, and
it falls out of the data model without any special-case code.

### Generation prompt fragment

The brief for a Creature-narrated scene includes:

> Per-axis relationship envelope (`ENT_CREATURE → ENT_FRANKENSTEIN`):
> affinity ∈ [+0.45, +0.60] (decreasing), fear ∈ [+0.10, +0.25],
> obligation ∈ [+0.30, +0.50], power_dynamic ∈ [-0.20, -0.05].
> Must-not events: `EVT_CREATURE_RECONCILES_WITH_FRANKENSTEIN`
> (would require affinity > 0.8).

The renderer can choose any prose realisation inside this envelope.
The miracle-step audit catches reconciliation language; the affective
audit catches prose that flattens the asymmetry.

---

## Cross-cutting: what every example shows about the pipeline

Independent of which fixture you load, **every** `run_pipeline` call
exercises the same seven-step shape ([architecture.md
§5](architecture.md), [pipeline-walkthrough.md](pipeline-walkthrough.md)).
What differs across the examples above is *which path through the
router runs* and *which audit signal dominates the feedback loop*.

| Pipeline step | Macbeth shows | Nile shows | Reservoir Dogs shows | R&J shows | Gone Girl shows | 1984 shows |
|---|---|---|---|---|---|---|
| 1 Ingestion | All five edge modalities | Channels + utterances | Two-clock event timing | Plausibility on first pass | `truth_value` extraction | `WorldTraitSnapshot` timelines |
| 2 Narrative physics | Rung 2 vs Rung 3 routing | Belief cascade via channels | `syuzhet_index` independence | Directive enumeration | Implausibility short-circuit | World-trait do-operator |
| 3 Brief assembly | Constraint envelope | Per-channel intelligibility | Reveal ordering | Typed `ConstraintBlock`s | Truth-value preservation | Ambient prior |
| 4 Generation | Constrained renderer | POV-locked utterances | Flashback positioning | LLM as renderer only | Lying-narrator scenes | Atmospheric biasing |
| 5 Audit + refinement | Miracle-step on causality | Dramatic irony score | Surprise as KL | Affective convergence | Three-way audit demo | Engine-threshold gate |
| 6 Re-extraction | Delta topology only | New beliefs added | New channels added | New events added | New utterances + truth | New world-trait snapshots |
| 7 Versioned merge | Shadow-branch fork | Mainline append | Mainline append | Shadow-branch fork | Mainline append | Mainline append |

---

## Feature-to-example matrix

A reverse index — *if you want to see feature X work, load fixture Y*:

| Feature | Best example | Where in this doc |
|---|---|---|
| `chain_reaction` edges | Macbeth, A Fish Called Wanda | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§7](#7-a-fish-called-wanda--objects-affordances-and-chain-reactions) |
| `mutation` edges (trait shock) | Macbeth | [§1](#1-macbeth--causal-physics-and-counterfactuals) |
| `mutation_social` edges | Frankenstein | [§8](#8-frankenstein--social-mutation-and-relationship-decay) |
| `affordance_gate` edges | A Fish Called Wanda | [§7](#7-a-fish-called-wanda--objects-affordances-and-chain-reactions) |
| `ambient_propagation` edges | 1984 | [§6](#6-1984--global-traits-and-ambient-propagation) |
| `Channel` + intelligibility | Death on the Nile | [§2](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony) |
| Utterances with `truth_value` | Gone Girl, Death on the Nile | [§5](#5-gone-girl--false-beliefs-and-the-audit-loop), [§2](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony) |
| Two-clock fabula vs syuzhet | Reservoir Dogs | [§3](#3-reservoir-dogs--syuzhet-vs-fabula-and-information-flow) |
| `GlobalTrait` + `WorldTraitSnapshot` | 1984 | [§6](#6-1984--global-traits-and-ambient-propagation) |
| Per-axis `RelationshipMetric` | Frankenstein | [§8](#8-frankenstein--social-mutation-and-relationship-decay) |
| `EntityStateSnapshot` timeline (Hybrid 4+5) | Macbeth | [§1](#1-macbeth--causal-physics-and-counterfactuals) |
| Pearl rung-2 intervention | Macbeth | [§1](#1-macbeth--causal-physics-and-counterfactuals) |
| Pearl rung-3 counterfactual | Macbeth, Romeo and Juliet | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer) |
| AMWN sandbox + shadow branch | Macbeth, Romeo and Juliet | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer) |
| Directive enumeration + scoring | Romeo and Juliet | [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer) |
| Affective scorer (dramatic irony) | Death on the Nile | [§2](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony) |
| Affective scorer (surprise / KL) | Reservoir Dogs | [§3](#3-reservoir-dogs--syuzhet-vs-fabula-and-information-flow) |
| Plausibility gate | Romeo and Juliet | [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer) |
| Brief assembly (`ConstraintBlock`) | Romeo and Juliet | [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer) |
| Constrained generation | A Fish Called Wanda, Frankenstein | [§7](#7-a-fish-called-wanda--objects-affordances-and-chain-reactions), [§8](#8-frankenstein--social-mutation-and-relationship-decay) |
| Causal audit (miracle step) | Gone Girl | [§5](#5-gone-girl--false-beliefs-and-the-audit-loop) |
| Abduction audit | Gone Girl | [§5](#5-gone-girl--false-beliefs-and-the-audit-loop) |
| Affective audit | Death on the Nile, Gone Girl | [§2](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony), [§5](#5-gone-girl--false-beliefs-and-the-audit-loop) |
| Engine-threshold deterministic gate | Gone Girl | [§5](#5-gone-girl--false-beliefs-and-the-audit-loop) |
| Re-extraction + versioned merge | All | [pipeline-walkthrough.md](pipeline-walkthrough.md#steps-67--prose-re-extraction--merge) |
| Branch-routing (`auto`/`mainline`/`shadow`) | Macbeth (counterfactual) | [§1](#1-macbeth--causal-physics-and-counterfactuals) |

---

## Trying these yourself

The fastest way to reproduce any of the runs above is the example seeder
or the MCP tool of the same name:

```python
from example_worlds.macbeth import world_state
from shadow_loom.pipeline import run_pipeline
from shadow_loom.query_models import InterventionQuery

result = run_pipeline(
    world_state=world_state,
    query=InterventionQuery(
        original_query="What if Macbeth had refused to kill Duncan?",
        interventions={"EVT_DUNCAN_MURDER": "⊘"},
    ),
)
print(result.prose)
print(result.physics_result["status"])
```

In the UI: open the example seeder dropdown (top of the **Story** tab),
pick the fixture, then drive queries from the **Causality** or
**Explorer** tabs. From an MCP client, call `seed_example_world` with
the fixture name and then any of the query tools (`intervene`,
`counterfactual`, `direct`, `narrate`, `evaluate`, `inspect`, `ask`).

---

## See also

* [architecture.md](architecture.md) — the conceptual map of the data model and the 12-step pipeline.
* [pipeline-walkthrough.md](pipeline-walkthrough.md) — code-level tour of one pipeline run.
* [query-and-cycles.md](query-and-cycles.md) — per-query-type mechanics.
* [academic-foundations.md](academic-foundations.md) — Pearl, Genette, Greimas, Sternberg, Halpern, Wilmot — the literature behind every named concept here.
* [use-cases.md](use-cases.md) — *what the system is for*, in audience-first terms.
* [ui-guide.md](ui-guide.md) — how to drive these queries from the NiceGUI workspace.
