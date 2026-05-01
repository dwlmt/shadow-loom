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
| 9 | [Persuasion](#9-persuasion--observation-queries-and-economic-common-causes) | `ObservationQuery` (Rung 1), overheard-conversation channel, `WORLD_PRIMOGENITURE` as economic constraint |
| 10 | [Great Gatsby](#10-great-gatsby--interrogation-queries-and-asymmetric-belief-topology) | `InterrogationQuery`, asymmetric bidirectional beliefs, ambient-as-licensor |
| 11 | [Wuthering Heights](#11-wuthering-heights--multi-generational-time-and-manualeditquery) | Multi-generational `state_timeline`, inherited grievance, `ManualEditQuery` |
| 12 | [Great Expectations](#12-great-expectations--abduction-with-hidden-benefactor--evaluationquery) | Hidden-benefactor abduction, moral over-determination, `EvaluationQuery` scorecard |
| 13 | [Apocalypse Now](#13-apocalypse-now--stacked-world_-traits-and-ambient-cognition) | Four parallel `WORLD_*` traits, noisy-OR ambient stacking, monotone spatial topology |
| 14 | [Brief Encounter](#14-brief-encounter--wilmot-suspense-the-despair-boundary-and-regret) | Suspense → despair collapse, `regret` emotion target, constant-supervision channel |
| 15 | [A Court of Thorns and Roses](#15-a-court-of-thorns-and-roses--magical-affordances-and-branch-promotion) | Magic-as-affordance, `WORLD_*` trait timelines, branch promotion |
| 16 | [Dad's Army](#16-dads-army--generalquery-and-the-comic-ensemble-graph) | `GeneralQuery`, repeated comic chain-reactions, sitcom suspense ceiling |

A query-type-to-fixture cross-reference lives
[just before the cross-cutting summary](#query-types-in-action--fixture-cross-reference);
a consolidated **feature-to-example matrix** lives at the
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

## 9. Persuasion — observation queries and economic common-causes

**Fixture:** [`example_worlds/persuasion.py`](../example_worlds/persuasion.py) ·
**Plot:** [`sample_plots/persuasion.txt`](../sample_plots/persuasion.txt)

Austen is the cleanest test of `ObservationQuery` — Pearl's Rung 1, the
"natural progression" cycle. The story turns on what people **observe and
infer** without anyone intervening, and on the economic and class
machinery (`WORLD_PRIMOGENITURE`, `WORLD_REGENCY_RANK`,
`WORLD_NAPOLEONIC_PRIZE_ECONOMY`) that *constrains every choice from
above*.

### `ObservationQuery` — Rung 1 in action

```python
from shadow_loom.query_models import ObservationQuery
from example_worlds.persuasion import world_state
from shadow_loom.pipeline import run_pipeline

result = run_pipeline(
    world_state=world_state,
    query=ObservationQuery(
        original_query="Anne sees Wentworth in the White Hart drawing room.",
        observations={
            "ENT_ANNE.location_id":      "LOC_WHITE_HART",
            "ENT_WENTWORTH.location_id": "LOC_WHITE_HART",
            "ENT_HARVILLE.location_id":  "LOC_WHITE_HART",
        },
        focus_entity_ids=["ENT_ANNE", "ENT_WENTWORTH"],
    ),
)
```

No `do(·)` operator runs. The router calls `extract_graph.build_ego_payload`
to compute the ego-graph at the observation, propagates the **ambient
state of `LOC_WHITE_HART`** into the focus entities (the `bustle=0.7`
ambient pushes audibility up; this is what makes the overheard
conversation possible), and lets the social cascade decide whether
Wentworth registers Anne's voice across the room.

The cycle still produces prose — but the brief carries no `must` events,
only a constraint envelope ("character locations pinned, channel
intelligibility recomputed, no new causal edges allowed").

### The overheard-conversation channel

The pivotal scene is a `Channel` with deliberately *asymmetric*
intelligibility:

```python
"CHN_ANNE_HARVILLE_OVERHEARD": Channel(
    id="CHN_ANNE_HARVILLE_OVERHEARD",
    participant_ids=["ENT_ANNE", "ENT_HARVILLE", "ENT_WENTWORTH"],
    intelligibility={
        "ENT_ANNE":      1.0,    # speaker
        "ENT_HARVILLE":  1.0,    # her direct interlocutor
        "ENT_WENTWORTH": 0.85,   # eavesdropping at his writing desk
    },
)
```

Wentworth is **not in `addressee_ids`** but his intelligibility is above
`physics.intelligibility_threshold`, so `propagate_social()` updates his
`Belief(target_id=ENT_ANNE, perceived_state="loves me yet")` from
`confidence=0.20` to `≈0.95` in a single pass. The result is the famous
hand-delivered letter — and the engine flags the belief flip as the
single mutation that licenses the climax. No murder, no surprise reveal:
just observation + channel physics + belief update. *That* is the Rung-1
cycle at full power.

### `WORLD_PRIMOGENITURE` as economic constraint

Sir Walter's debts are **not** an event in the timeline; they are a
standing pressure encoded as a `GlobalTrait` value of 0.8 on
`WORLD_PRIMOGENITURE` plus an `ambient_propagation` edge into
`LOC_KELLYNCH_HALL.fiscal_strain`. Every later choice — the let to the
Crofts, the move to Bath, Elizabeth's competition with Mrs Clay — has a
licensing edge that traces back to this single constant. Run
`do(WORLD_PRIMOGENITURE = 0.1)` and the engine cleanly cascades the
counterfactual ("Sir Walter inherits in trust, not absolutely") through
every downstream estate-decision event.

This is why the Austen fixture is the textbook example for the
`WORLD_*`-as-common-cause-parent pattern in
[architecture.md §3](architecture.md): the economic constraint is
*literally* the cause of every plot beat, and it is modelled as one node
with one timeline.

---

## 10. Great Gatsby — interrogation queries and asymmetric belief topology

**Fixture:** [`example_worlds/great_gatsby.py`](../example_worlds/great_gatsby.py) ·
**Plot:** [`sample_plots/great_gatsby.txt`](../sample_plots/great_gatsby.txt)

Gatsby is the **`InterrogationQuery`** showcase — graph-RAG-with-proof
over the AMWN, no time advance, no prose. The story is *about* who
knows what, who *thinks* the other knows what, and who is wrong on both
counts. Three of those four answers can be returned by a single query:

```python
from shadow_loom.query_models import InterrogationQuery

result = run_pipeline(
    world_state=world_state,
    query=InterrogationQuery(
        original_query="Who at the Plaza Suite knows that Daisy was driving?",
        require_proof=True,
    ),
)
for path in result.physics_result["proof_paths"]:
    print(path)
```

The router runs no `do(·)`, no abduction, no propagation, no scoring,
no LLM. It executes pure graph search:

1. Find `EVT_DAISY_DRIVES_INTO_MYRTLE` in the events table.
2. For each entity at `LOC_PLAZA_SUITE`, walk the channel graph and
   the `Belief` index for any node whose `target_id` references the
   event.
3. Return a `ProofPath` per entity: `(entity, belief, evidence_chain)`.

For Gatsby this returns: Gatsby (direct witness), Daisy (actor),
**not** Tom (his belief is the `truth_value="false"` cover from
Gatsby), **not** Wilson. The query *also* returns the missing edge —
the absence is what the next directive can target.

### Bidirectional asymmetric beliefs

The Gatsby fixture's centrepiece is the asymmetric Daisy/Gatsby
relationship: each holds a `Belief(target_id=other, …)` whose
`perceived_state` is in continuous tension with the *truth*. Per the
[Frankenstein pattern (§8)](#8-frankenstein--social-mutation-and-relationship-decay)
the `RelationshipEdge` is bidirectional and per-axis:

```python
RelationshipEdge(
    source_id="ENT_GATSBY", target_id="ENT_DAISY",
    metrics={
        "affinity":      RelationshipMetric(value=+0.95, inertia=0.85),
        "idealisation":  RelationshipMetric(value=+0.90, inertia=0.7),
    },
),
RelationshipEdge(
    source_id="ENT_DAISY", target_id="ENT_GATSBY",
    metrics={
        "affinity":      RelationshipMetric(value=+0.55, inertia=0.4),
        "fear_of_scandal": RelationshipMetric(value=+0.70, inertia=0.6),
    },
),
```

Gatsby's idealisation is high-inertia and survives every contrary signal;
Daisy's affinity is low-inertia and can be flipped by a single
`mutation_social` shock (Tom's revelations at the Plaza). The novel's
ending is exactly this asymmetry working out — and it is detectable from
the graph alone via two parallel `compute_trait_trajectories()` calls.

### The Valley of Ashes ambient

`LOC_VALLEY_OF_ASHES.moral_emptiness=0.85` is wired into every event
that physically passes through the location via
`ambient_propagation`. That is what makes Myrtle's death *causally* the
moral pivot it reads as: the `mutation` edge from
`EVT_DAISY_DRIVES_INTO_MYRTLE → ENT_TOM.callousness` is amplified by
the ambient, so Tom's small but real spike in callousness is what the
engine forwards into `EVT_TOM_LEADS_WILSON_TO_GATSBY`. Without the
ambient, the Tom→Wilson edge would not fire — Tom's prior callousness
is below the activation impulse alone. This is *atmosphere as physics*,
the same machinery as 1984's surveillance ambient
([§6](#6-1984--global-traits-and-ambient-propagation)) but inverted:
1984's ambient suppresses behaviour, Gatsby's enables it.

---

## 11. Wuthering Heights — multi-generational time and `ManualEditQuery`

**Fixture:** [`example_worlds/wuthering_heights.py`](../example_worlds/wuthering_heights.py) ·
**Plot:** [`sample_plots/wuthering_heights.txt`](../sample_plots/wuthering_heights.txt)

Wuthering Heights is the **time-depth** fixture: events span thirty
years, two generations, a death, an exhumation, and a posthumous
reconciliation. It is also the cleanest demonstration of
`ManualEditQuery` — the cycle where the user *supplies the prose* and
the engine reverses-engineers the world-state delta.

### `state_timeline` across decades

`ENT_HEATHCLIFF.state_timeline` carries snapshots at
`fabula_time = 5000` (childhood, brought from Liverpool),
`14000` (overhears Catherine's marriage decision, flees),
`22000` (returns rich and vengeful — `ruthlessness` jumps from 0.4 to
0.85, `inertia` is bumped to 0.7), `40000` (acquires Wuthering Heights
mortgage), `52000` (death, status="dead"). `reconstruct_entity_at` is
deterministic across *all* of those — the ego-graph for a query at
`fabula_time=45000` reconstructs Heathcliff as the rich, vengeful
landlord, not the orphan.

The crucial design choice ([architecture.md
§2](architecture.md#hybrid-45-timeline)): the long gaps are **not**
filled in. Snapshots are journalled only when a causal edge fires;
between snapshots the engine interpolates by holding the last value.
This is what keeps the graph small (Wuthering Heights has ~60 events
spanning 50 000 fabula ticks) while still letting causal physics rewind
to any point.

### Multi-generational `mutation_social` cascades

The second-generation plot (Cathy / Linton / Hareton) is licensed by
`mutation_social` edges whose source events are in the first
generation. `EVT_HEATHCLIFF_FORCES_LINTON_MARRIAGE → REL(CATHY,
HARETON).affinity` fires *across a generation* because Hareton's
`bitterness_at_dispossession` was set decades earlier by
`EVT_HINDLEY_DEGRADES_HEATHCLIFF`. The graph correctly treats
inherited grievance as a propagation chain — not a copied trait — so
the directive engine can ask "what if Hindley had treated the orphan
Heathcliff as a brother?" and watch the second-generation reconciliation
arrive *for free* via abduction + propagation, with no hand-coded
"Hareton is now nice" patch.

### `ManualEditQuery` — the user supplies prose

The user can write a chapter themselves and hand it to Shadow-Loom for
ingestion-into-canon:

```python
from shadow_loom.query_models import ManualEditQuery

result = run_pipeline(
    versioned_model=vwm,
    query=ManualEditQuery(
        original_query="Add a final chapter where Lockwood revisits Gimmerton",
        edited_prose=lockwood_revisit_text,   # user's own text
        description="Lockwood revisits the Heights eighteen months later",
        focus_entity_ids=["ENT_LOCKWOOD", "ENT_NELLY", "ENT_CATHY", "ENT_HARETON"],
    ),
)
```

The pipeline **skips** Steps 2–5 (no physics, no brief, no LLM
generation, no audit). The user's prose is the answer. Step 6
(prose → topology re-extraction) runs as normal: the chapter is parsed
into new `EventNode`s, `Channel`s, beliefs and trait deltas, and Step 7
merges them onto the canon branch. The `ManualEditStepRecord` notes
that this version's prose is *user-authored* so the Editor tab can
mark it accordingly and the auditor's miracle-step check is **off** for
this version (the user is the ground truth).

This is the cycle a human author uses when they want the engine to
keep their own prose in the world model — Shadow-Loom becomes a
graph-aware editor rather than a generator.

---

## 12. Great Expectations — abduction with hidden benefactor + `EvaluationQuery`

**Fixture:** [`example_worlds/great_expectations.py`](../example_worlds/great_expectations.py) ·
**Plot:** [`sample_plots/great_expectations.txt`](../sample_plots/great_expectations.txt)

The novel's structural engine is one fact Pip does not know — that his
benefactor is Magwitch, not Miss Havisham. Every page until the reveal is
*reader-side abduction* against missing evidence; every page after is
*character-side abduction* of why the early chapters happened. This is
the textbook **Pearl Rung 3** scenario.

### The hidden-benefactor counterfactual

```python
from shadow_loom.query_models import CounterfactualQuery

result = run_pipeline(
    world_state=world_state,
    query=CounterfactualQuery(
        original_query="What if Pip had been told Magwitch was his benefactor "
                       "from the start?",
        historical_interventions={
            "EVT_JAGGERS_VISITS_FORGE.actor_ids":
                ["ENT_JAGGERS", "ENT_MAGWITCH"],   # Magwitch arrives in person
            "EVT_JAGGERS_VISITS_FORGE.content":
                "Your great expectations come from a transported convict.",
        },
        evidence_node_ids=[
            "ENT_PIP",          # condition on what we eventually observed:
            "ENT_ESTELLA",      # Pip's snobbery, Estella's revealed parentage,
            "ENT_MAGWITCH",     # Magwitch's eventual capture and confession.
        ],
    ),
)
```

The engine runs:

1. **Abduction** — given the late-novel state of Pip (`snobbery=0.7`,
   `shame=0.85`), Estella (`reveal_parentage=Magwitch`), and Magwitch
   (`status=dying`), back-propagate hidden ancestor deltas onto the
   sandbox at `fabula_time=1000` (Pip's childhood).
2. **Intervention** — replace the Jaggers visit with the truthful
   variant.
3. **Re-propagate** forward.

The result is a sandbox where Pip's `snobbery` trajectory peaks at
`0.4` instead of `0.7` (low-inertia trait, snaps to evidence), but
Estella's parentage line and Magwitch's deportation chain are
**unchanged** (high-inertia historical structure). The engine
correctly reports *moral over-determination*: revealing the benefactor
early changes Pip but does not save Magwitch, because the legal-system
chain is forced by `WORLD_LAWS_REACH_OVER_CRIMINAL_CLASS` rather than
Pip's awareness.

### `EvaluationQuery` — the full-story scorecard

After enough Pip chapters have been ingested, run:

```python
from shadow_loom.query_models import EvaluationQuery

result = run_pipeline(
    versioned_model=vwm,
    query=EvaluationQuery(
        original_query="Score the current Pip arc.",
        focus_entity_ids=["ENT_PIP", "ENT_MAGWITCH", "ENT_ESTELLA"],
        include_full_prose=True,
    ),
)
print(result.physics_result["narrative_order"])
```

The router calls `_run_evaluation_branch`. No prose is written, no
version is committed. What comes back is a `NarrativeOrderObject`:

* **Causal physics feedback** — count of miracle steps, blocked
  propagations, average activation pressure across acts.
* **Affective feedback** — full mystery / dramatic-irony / suspense /
  surprise trajectories per focus entity, plus the six emotion targets
  scored across the syuzhet.
* **LLM literary critique** — a single Pydantic-AI agent comparing the
  prose against the brief targets.
* **`overall_pass`** — boolean gate combining the three.

For Great Expectations this is how you verify that the
hidden-benefactor abduction structure is *actually* delivering the
mystery score the novel needs — `compute_mystery_score(["ENT_PIP"],
syuzhet_anchor=N)` should climb monotonically until `EVT_MAGWITCH_REVEAL`
and then collapse to ≈0. The evaluation query is the test that
confirms it.

---

## 13. Apocalypse Now — stacked WORLD_ traits and ambient cognition

**Fixture:** [`example_worlds/apocalypse_now.py`](../example_worlds/apocalypse_now.py) ·
**Plot:** [`sample_plots/apocalypse_now.txt`](../sample_plots/apocalypse_now.txt)

Where 1984 has *one* dominant `WORLD_*` trait (the regime), Apocalypse
Now has **four in parallel**, each pressuring every event:

```python
"WORLD_VIETNAM_WAR":         GlobalTrait(...)   # the proxy theatre
"WORLD_HEART_OF_DARKNESS":   GlobalTrait(...)   # Conradian moral abyss
"WORLD_CHAIN_OF_COMMAND":    GlobalTrait(...)   # sanctioned-murder protocol
"WORLD_RIVER_AS_FATE":       GlobalTrait(...)   # upriver predestination
```

Every event downstream of the Saigon hotel has *all four* as
common-cause parents. This stresses the
`MECHANISM_TRAIT_MAP` gating in `causal_physics.py`: a `WORLD_*`
trait can only mutate traits whose family is in its
`affected_domains`. `WORLD_CHAIN_OF_COMMAND.affected_domains =
{"obedience", "duty", "moral_disengagement"}` so it is licensed to
shift Willard's `obedience` but **not** his `dissociation`; that
latter mutation is forced by `WORLD_HEART_OF_DARKNESS` instead. The
intersection lets the engine attribute each character beat to the
*right* common-cause for the brief.

### Ambient stacking on cognition

The PBR carries a stack of three ambients —
`isolation=0.85`, `humidity=0.95`, `drug_haze=0.7` — and Willard's
`paranoia` and `dissociation` traits are mutated by **the noisy-OR
aggregate** of the three ambient pressures over each fabula tick spent
on the boat:

$$\text{impulse}_{paranoia}(t) = 1 - \prod_{a \in \text{ambients}}\bigl(1 - w_a \cdot v_a(t)\bigr)$$

…where `w_a` is the trait's per-ambient weight from `_ambient_force_multiplier`
and `v_a(t)` is the ambient value at that tick. This is what produces the
film's monotone descent: every river-tick adds a small impulse, the
trait inertia is kept low so the impulses *accumulate*, and by Kurtz's
compound Willard is effectively a different person from the one in
Saigon. The engine can show this trajectory in the **Trait Trajectories**
tab without the LLM ever being asked "is Willard losing his mind?".

### The river as monotone spatial constraint

`SpatialEdge`s on the Nung river form a **strict total order** — the
PBR cannot return to Hau Phat once it has reached Do Lung Bridge.
`do(EVT_PBR_TURNS_BACK = trigger)` therefore fails the plausibility
gate (`_check_intervention_plausibility` finds no licensing
`SpatialEdge(direction="downstream")`). The renderer is forbidden the
fictional "they could just turn around" escape, and the brief surfaces
the spatial constraint as a `must-not` block. This is the first
fixture where pure spatial topology blocks a plot move — and it is the
right block: the film's *point* is that retreat is not an option.

---

## 14. Brief Encounter — Wilmot suspense, the despair boundary, and regret

**Fixture:** [`example_worlds/brief_encounter.py`](../example_worlds/brief_encounter.py) ·
**Plot:** [`sample_plots/brief_encounter.txt`](../sample_plots/brief_encounter.txt)

Lean's chamber drama is the cleanest test of the **Wilmot suspense →
despair boundary**. There is no murder, no chase, no reveal — just two
people who could choose to take the next train together and don't. The
forward causal graph eventually runs out of *hope* events for the
focal couple, and `compute_suspense_score` returns 0 not because the
characters are safe but because their futures have closed.

### Suspense climbing then collapsing

For `entity_ids=["ENT_LAURA", "ENT_ALEC"]` at successive
`syuzhet_anchor` values:

```text
syuzhet_anchor    P(threat)    P(hope)    suspense
   8 (botanical)     0.45        0.70       0.00     # hope dominates
  14 (kardomah)      0.62        0.65       0.00     # near-equal
  21 (flat)          0.85        0.55       0.30     # threat overtakes
  28 (returns home)  0.95        0.10       0.85     # peak suspense
  31 (Dolly arrives) 1.00        0.00       0.00     # despair: hope = 0
```

The final `0.0` is the engine *correctly* refusing to call this scene
suspenseful — see the `if hope_prob <= 0.0: return 0.0` early-return in
[`directive_assembly.py::compute_suspense_score`](../shadow_loom/directive_assembly.py)
and [academic-foundations.md
§3.1](academic-foundations.md#31-suspense-as-uncertainty-reduction--wilmot--keller-acl-2020).
The auditor flags any prose that *reads* as suspenseful at this
syuzhet point as having mistaken despair for tension.

### The `regret` emotion target

After the train scene the directive cycle's natural target is
`target_effect="regret"` (per the `_EFFECT_TRAIT_MAP` in
[directive_assembly.py:1095+](../shadow_loom/directive_assembly.py)
that maps regret onto `{guilt, remorse, despair}`). Brief Encounter's
strength is that the trait values are *already* close to their
regret-saturation targets after `EVT_DOLLY_INTERRUPTS_FAREWELL`, so the
directive enumerator has very little room to add events — the winning
candidate is usually `EVT_LAURA_HOME_BY_FIRESIDE_INTERIOR_MONOLOGUE`,
which adds zero new causal edges and just reweights the syuzhet. The
brief becomes almost pure constraint with one suggested beat. This is
what the engine *should* do when the world has converged on its target
emotion: not invent more plot.

### Constant-supervision channel

`LOC_REFRESHMENT_ROOM.constant_supervision=0.85` plus a
`CHN_TEAROOM_BYSTANDERS` channel with all bystanders as participants
at `intelligibility=0.6` means *every utterance* between Laura and
Alec is partly overheard. This is what makes their inability to speak
plainly a structural fact, not a stylistic choice — the directive
engine cannot raise `dramatic_irony` by picking utterances the
characters know are private, because no such utterance is licensed.

---

## 15. A Court of Thorns and Roses — magical affordances and branch promotion

**Fixture:** [`example_worlds/a_court_of_thorn_and_roses.py`](../example_worlds/a_court_of_thorn_and_roses.py) ·
**Plot:** [`sample_plots/a_court_of_thorn_and_roses.txt`](../sample_plots/a_court_of_thorn_and_roses.txt)

Fantasy is the test of *named magic-as-affordance*. Magic is not a free
pass — it is encoded as `WORLD_*` traits with `affected_domains`,
specific objects with magical `Affordance`s, and `Channel`s with
non-physical `intelligibility` semantics.

### Named magic systems as `WORLD_*` traits

Four magic-system latents drive every supernatural event:

```python
"WORLD_TREATY_WALL":      GlobalTrait(value=0.95, ...)   # ancient compact
"WORLD_AMARANTHAS_CURSE": GlobalTrait(value=0.85, ...,
                              state_timeline=[
                                  WorldTraitSnapshot(fabula_time=12000, value=0.95),
                                  WorldTraitSnapshot(fabula_time=22000, value=0.10),
                              ])
"WORLD_MATING_BOND":      GlobalTrait(value=0.7, ...)
"WORLD_FAE_MORTAL_DIVIDE":GlobalTrait(value=0.85, ...)
```

The `WORLD_AMARANTHAS_CURSE` *itself has a timeline* —
`WorldTraitSnapshot` entries at the moments the curse intensifies and
when it breaks. The engine therefore treats "the Blight is spreading"
as a first-class evolving state rather than as scenery, and a
`do(WORLD_AMARANTHAS_CURSE = 0.0)` counterfactual cleanly cascades the
removal of the curse-driven `mutation` edges *for every event after the
target fabula time*.

### Magic as affordance, not deus ex machina

Magical objects expose their power as ordinary `Affordance`s:

```python
NarrativeObject(
    id="OBJ_MATING_BOND_MARK", ...,
    affordances=[
        Affordance(action="bind_souls",       target_type="Entity"),
        Affordance(action="sense_distress",   target_type="Entity"),
    ],
)
```

A scene where Rhysand senses Feyre's distress across the continent is
licensed only if the bond-mark affordance is satisfied at both ends.
The auditor's miracle-step check rejects any prose where a character
"just knows" without a licensing affordance. Magic obeys the same
gating as a knife — exactly the design-intent in
[design-decisions.md](design-decisions.md).

### Branch promotion for "what if" canon

Counterfactual queries on this fixture default to landing on a
`world_id="shadow"` branch — useful because readers of fantasy often
want to explore "what if Feyre had refused the bargain?" without
losing canon. The MCP `list_branches` tool surfaces all such shadows;
the `promote_branch` tool moves one to `factual` if the author wants
it adopted. The full per-branch diff is journalled, so promotions are
reversible. (See [mcp-guide.md](mcp-guide.md) for the tool list.)

---

## 16. Dad's Army — `GeneralQuery` and the comic ensemble graph

**Fixture:** [`example_worlds/dads_army.py`](../example_worlds/dads_army.py) ·
**Plot:** [`sample_plots/dads_army.txt`](../sample_plots/dads_army.txt)

Sitcom is a stress test for the **`GeneralQuery`** — open-ended
omniscient Q&A that returns the full extracted world state for a
downstream LLM to riff over. It is also a low-stakes demo of repeated
`chain_reaction` cascades that resolve harmlessly (the Wilson suspense
gate is *meant* to collapse).

### `GeneralQuery` — full graph back to the caller

```python
from shadow_loom.query_models import GeneralQuery

result = run_pipeline(
    world_state=world_state,
    query=GeneralQuery(
        original_query="Summarise the platoon's hierarchy and mutual disdain.",
        include_topology=True,
    ),
)
# result.physics_result["world_state"] contains the full WorldStateV1
# result.physics_result["query_type"] == "general"
# No prose written, no version committed.
```

The router does **no** simulation. It returns the entire `WorldStateV1`
(or the topology slice when `include_topology=False`) for the calling
agent to reason over. Use this from an MCP client when a downstream
agent wants to ask the LLM five questions about the world without
spinning up the physics engine each time.

### Repeated comic chain-reactions

Dad's Army ingests cleanly because *every* episode is the same
`chain_reaction` shape: Mainwaring asserts authority → Wilson gently
questions → Pike says something stupid → Jones offers fixed-bayonet
solution → Frazer prophesies doom → resolution. The fixture wires this
as a recurring `chain_reaction` chain with `propagation_delay`
controlling the comic timing. Run `compute_suspense_score` on
`ENT_MAINWARING` and you'll see it climb to ≈0.4 mid-chain and collapse
back to ≈0.05 — exactly the low-stakes ceiling sitcom requires, and
exactly what `WORLD_HOME_FRONT_SPIRIT.value=0.7` (a *protective* world
trait) damps it to.

This is a useful negative-space example: the engine *can* score sitcom,
and what it scores is *correct* (sitcom suspense should not climb above
0.5). The directive engine refuses to push it higher because doing so
would require violating the home-front-spirit common-cause.

---

## Query types in action — fixture cross-reference

Every query type in [query-and-cycles.md §1](query-and-cycles.md#1-the-eight-query-types)
is best illustrated by a particular fixture. If you want to see one
type at full power:

| Query type | Best fixture | Section |
|---|---|---|
| `ObservationQuery` (Rung 1) | Persuasion | [§9](#9-persuasion--observation-queries-and-economic-common-causes) |
| `InterventionQuery` (Rung 2) | Macbeth | [§1](#1-macbeth--causal-physics-and-counterfactuals) |
| `CounterfactualQuery` (Rung 3) | Great Expectations, Macbeth | [§12](#12-great-expectations--abduction-with-hidden-benefactor--evaluationquery), [§1](#1-macbeth--causal-physics-and-counterfactuals) |
| `DirectiveQuery` (affective) | Romeo and Juliet, ACOTAR | [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer), [§15](#15-a-court-of-thorns-and-roses--magical-affordances-and-branch-promotion) |
| `InterrogationQuery` (graph RAG) | Great Gatsby | [§10](#10-great-gatsby--interrogation-queries-and-asymmetric-belief-topology) |
| `GeneralQuery` (Q&A passthrough) | Dad's Army | [§16](#16-dads-army--generalquery-and-the-comic-ensemble-graph) |
| `ManualEditQuery` (user prose) | Wuthering Heights | [§11](#11-wuthering-heights--multi-generational-time-and-manualeditquery) |
| `EvaluationQuery` (scorecard) | Great Expectations | [§12](#12-great-expectations--abduction-with-hidden-benefactor--evaluationquery) |

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
| `chain_reaction` edges | Macbeth, A Fish Called Wanda, Dad's Army | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§7](#7-a-fish-called-wanda--objects-affordances-and-chain-reactions), [§16](#16-dads-army--generalquery-and-the-comic-ensemble-graph) |
| `mutation` edges (trait shock) | Macbeth, Great Gatsby | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§10](#10-great-gatsby--interrogation-queries-and-asymmetric-belief-topology) |
| `mutation_social` edges | Frankenstein, Wuthering Heights (multi-generation) | [§8](#8-frankenstein--social-mutation-and-relationship-decay), [§11](#11-wuthering-heights--multi-generational-time-and-manualeditquery) |
| `affordance_gate` edges | A Fish Called Wanda, ACOTAR (magical) | [§7](#7-a-fish-called-wanda--objects-affordances-and-chain-reactions), [§15](#15-a-court-of-thorns-and-roses--magical-affordances-and-branch-promotion) |
| `ambient_propagation` edges | 1984, Apocalypse Now (stacked), Persuasion (Kellynch fiscal_strain) | [§6](#6-1984--global-traits-and-ambient-propagation), [§13](#13-apocalypse-now--stacked-world_-traits-and-ambient-cognition), [§9](#9-persuasion--observation-queries-and-economic-common-causes) |
| `Channel` + intelligibility | Death on the Nile, Persuasion (overheard), Brief Encounter (constant supervision) | [§2](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony), [§9](#9-persuasion--observation-queries-and-economic-common-causes), [§14](#14-brief-encounter--wilmot-suspense-the-despair-boundary-and-regret) |
| Utterances with `truth_value` | Gone Girl, Death on the Nile | [§5](#5-gone-girl--false-beliefs-and-the-audit-loop), [§2](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony) |
| Two-clock fabula vs syuzhet | Reservoir Dogs, Wuthering Heights (multi-decade) | [§3](#3-reservoir-dogs--syuzhet-vs-fabula-and-information-flow), [§11](#11-wuthering-heights--multi-generational-time-and-manualeditquery) |
| `GlobalTrait` + `WorldTraitSnapshot` | 1984, ACOTAR (curse timeline), Apocalypse Now | [§6](#6-1984--global-traits-and-ambient-propagation), [§15](#15-a-court-of-thorns-and-roses--magical-affordances-and-branch-promotion), [§13](#13-apocalypse-now--stacked-world_-traits-and-ambient-cognition) |
| Per-axis `RelationshipMetric` | Frankenstein, Great Gatsby (asymmetric) | [§8](#8-frankenstein--social-mutation-and-relationship-decay), [§10](#10-great-gatsby--interrogation-queries-and-asymmetric-belief-topology) |
| `EntityStateSnapshot` timeline (Hybrid 4+5) | Macbeth, Wuthering Heights | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§11](#11-wuthering-heights--multi-generational-time-and-manualeditquery) |
| Pearl Rung 1 observation | Persuasion | [§9](#9-persuasion--observation-queries-and-economic-common-causes) |
| Pearl Rung 2 intervention | Macbeth | [§1](#1-macbeth--causal-physics-and-counterfactuals) |
| Pearl Rung 3 counterfactual | Macbeth, Romeo and Juliet, Great Expectations | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer), [§12](#12-great-expectations--abduction-with-hidden-benefactor--evaluationquery) |
| Abduction (Rung 3 backward) | Macbeth, Great Expectations | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§12](#12-great-expectations--abduction-with-hidden-benefactor--evaluationquery) |
| AMWN sandbox + shadow branch | Macbeth, Romeo and Juliet, ACOTAR (branch promotion) | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer), [§15](#15-a-court-of-thorns-and-roses--magical-affordances-and-branch-promotion) |
| Directive enumeration + scoring | Romeo and Juliet, Brief Encounter (regret target) | [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer), [§14](#14-brief-encounter--wilmot-suspense-the-despair-boundary-and-regret) |
| Affective scorer (mystery) | Macbeth, Great Expectations | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§12](#12-great-expectations--abduction-with-hidden-benefactor--evaluationquery) |
| Affective scorer (dramatic irony) | Death on the Nile | [§2](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony) |
| Affective scorer (suspense, despair boundary) | Romeo and Juliet, Brief Encounter, Dad's Army (sitcom ceiling) | [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer), [§14](#14-brief-encounter--wilmot-suspense-the-despair-boundary-and-regret), [§16](#16-dads-army--generalquery-and-the-comic-ensemble-graph) |
| Affective scorer (surprise / KL) | Reservoir Dogs | [§3](#3-reservoir-dogs--syuzhet-vs-fabula-and-information-flow) |
| Plausibility gate | Romeo and Juliet, Apocalypse Now (spatial) | [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer), [§13](#13-apocalypse-now--stacked-world_-traits-and-ambient-cognition) |
| Brief assembly (`ConstraintBlock`) | Romeo and Juliet | [§4](#4-romeo-and-juliet--directives-and-the-affective-scorer) |
| Constrained generation | A Fish Called Wanda, Frankenstein | [§7](#7-a-fish-called-wanda--objects-affordances-and-chain-reactions), [§8](#8-frankenstein--social-mutation-and-relationship-decay) |
| Causal audit (miracle step) | Gone Girl, ACOTAR (magic-without-affordance) | [§5](#5-gone-girl--false-beliefs-and-the-audit-loop), [§15](#15-a-court-of-thorns-and-roses--magical-affordances-and-branch-promotion) |
| Abduction audit | Gone Girl | [§5](#5-gone-girl--false-beliefs-and-the-audit-loop) |
| Affective audit | Death on the Nile, Gone Girl | [§2](#2-death-on-the-nile--epistemic-physics-and-dramatic-irony), [§5](#5-gone-girl--false-beliefs-and-the-audit-loop) |
| Engine-threshold deterministic gate | Gone Girl | [§5](#5-gone-girl--false-beliefs-and-the-audit-loop) |
| `ObservationQuery` (Rung 1 cycle) | Persuasion | [§9](#9-persuasion--observation-queries-and-economic-common-causes) |
| `InterrogationQuery` (graph RAG) | Great Gatsby | [§10](#10-great-gatsby--interrogation-queries-and-asymmetric-belief-topology) |
| `GeneralQuery` (Q&A passthrough) | Dad's Army | [§16](#16-dads-army--generalquery-and-the-comic-ensemble-graph) |
| `ManualEditQuery` (user-supplied prose) | Wuthering Heights | [§11](#11-wuthering-heights--multi-generational-time-and-manualeditquery) |
| `EvaluationQuery` (`NarrativeOrderObject`) | Great Expectations | [§12](#12-great-expectations--abduction-with-hidden-benefactor--evaluationquery) |
| Re-extraction + versioned merge | All | [pipeline-walkthrough.md](pipeline-walkthrough.md#steps-67--prose-re-extraction--merge) |
| Branch-routing (`auto`/`mainline`/`shadow`) | Macbeth (counterfactual), ACOTAR (promotion) | [§1](#1-macbeth--causal-physics-and-counterfactuals), [§15](#15-a-court-of-thorns-and-roses--magical-affordances-and-branch-promotion) |

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
