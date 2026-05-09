# Architecture

This document is the technical reference for Shadow-Loom. It covers the data
model, the 12-step pipeline in implementation detail, the runtime modules, the
persistence layer, and the integration surfaces (UI + MCP).

For the conceptual / theoretical grounding of the ideas described here, see
[academic-foundations.md](academic-foundations.md). For the *why* behind each
choice, see [design-decisions.md](design-decisions.md).

---

## 1. The data model — `WorldStateV1`

Defined in [`shadow_loom/models.py`](../shadow_loom/models.py). All schemas use
Pydantic v2 with `model_validator` constraints.

### Nodes

| Class | Prefix | Purpose |
|---|---|---|
| `Location` | `LOC_` | Spatial container with `ambient_state: Dict[str, AmbientVector]` (each ambient owns `value`, `volatility`, `evidence_strength`). |
| `NarrativeObject` | `OBJ_` | Inanimate item with `affordances: List[Affordance]`. |
| `Entity` | `ENT_` | Character / agent with `traits` (per-trait `TraitVector{value, inertia, evidence_strength}`), `beliefs`, `status`, `state_timeline`. |
| `EventNode` | `EVT_` | Atomic happening anchored on both `fabula_time` and `syuzhet_index`. `event_type="utterance"` carries `content`, `speaker_id`, `addressee_ids`, `via_channel_id`, `truth_value`. |
| `GlobalTrait` | `WORLD_` | World-level fact / law / regime ("magic system", "surveillance state"). `magnitude` is a `TraitVector` (`value`, `inertia`, `evidence_strength`). |
| `Channel` | `CHN_` | Standing communication capability between participants. `medium`, `directionality` (`broadcast`/`duplex`/`simplex`), per-participant `intelligibility ∈ [0,1]` (replaces the legacy `is_encrypted` boolean), `established_at_fabula`. |

### Edges

| Class | Topology | Notes |
|---|---|---|
| `CausalEdge` | event⇄event / event→state / state→event / state→state | Single class with five `causality_type` modalities; validator enforces source/target type matches modality. |
| `RelationshipEdge` | entity⇄entity | Per-axis `metrics` dict (`affinity` / `fear` / `power_dynamic`); each axis owns its own `value`, `inertia`, `evidence_strength`, `last_updated_fabula`. Read via flat back-compat properties. |
| `SpatialEdge` | location→location | Optional `is_locked` + `barrier_item_id`. |

Note: communication is no longer modelled as an edge. Standing capability lives on the `Channel` *node*; discrete messages are first-class `EventNode`s with `event_type="utterance"` referencing a channel via `via_channel_id`.

### Temporal sub-models (the "Hybrid 4+5" design)

* **`EntityStateSnapshot`** lives on `Entity.state_timeline` — sparse delta of
  `traits / beliefs_added / beliefs_invalidated / status / location_id` valid
  from a given `fabula_time`, with an optional `triggered_by` event ID.
* **`WorldTraitSnapshot`** does the same for `GlobalTrait.state_timeline`.
* **`PropositionSnapshot`** lives on `Proposition.state_timeline` — sparse delta of `stakes / audience_default_prior / description` plus `truth_at_fabula` commitments emitted whenever a chunk's events resolve a proposition.
* **`ConcernSnapshot`** lives on `Concern.state_timeline` — sparse delta of `salience / polarity / activation_fabula_window / counter_concern_ids / kind`, capturing reversals such as Macbeth's desire→fear flip on `PROP_BANQUO_DEAD`.
* `Entity.traits / beliefs / status / location_id`,
  `GlobalTrait.magnitude`, `Proposition.stakes /
  audience_default_prior`, and `Concern.salience / polarity`
  represent the **initial pre-story baseline**; each timeline
  replays sparse deltas on top.
* Four pure reconstruction functions return the merged state at any chronological slice: `reconstruct_entity_at(entity, fabula_time)`, `reconstruct_world_trait_at(trait, fabula_time)`, `reconstruct_proposition_at(prop, fabula_time)`, and `reconstruct_concern_at(concern, fabula_time)`. They are used by the ego-graph extractor, the causal physics engine, the narrative physics layer, and the propositional affect scorers (`affect_unification.BeliefState`).

### Propositions and concerns

`WorldStateV1.propositions: List[Proposition]` is the world's typed
claim registry. Each `Proposition` carries `proposition_id`
(`PROP_*`), `kind ∈ {event_occurs, trait_holds, relation_holds,
identity_is, outcome}`, `referent_ids` (the ontology entities the
claim is about), `audience_default_prior ∈ [0, 1]`,
`stakes ∈ [0, 1]`, the time-indexed `truth_at_fabula: Dict[int,
bool]`, and a sparse `state_timeline` of `PropositionSnapshot`s.

`Entity.concerns: List[Concern]` carries each character's standing
fear / desire about a specific proposition: `concern_id` (`CCN_*`),
`proposition_id`, `polarity ∈ {desire, fear}`, `kind` (e.g.
`betrayal`), `salience ∈ [0, 1]`, `activation_fabula_window`, and
`counter_concern_ids` for ambivalence pairs (a character
simultaneously *desiring* and *fearing* the same outcome). Both
field families are populated either in Phase A3 (`Proposition
Catalogue`) of ingestion or in the per-chunk Affect sub-stage
(B4). Together they let the propositional / Bayesian affect
scorers in [`affect_unification.py`](../shadow_loom/affect_unification.py) reason
about suspense / surprise / irony / mystery as belief-revision
over named claims rather than only as graph-geometry of
trait-and-event mass — see
[academic-foundations.md §3.7](academic-foundations.md#37-propositional-belief-revision-affect-affect_unificationpy).

### Time

* `fabula_time : int` — strict chronological order (Russian Formalist *fabula*).
* `syuzhet_index : int` — order of presentation in the prose (Russian
  Formalist *syuzhet*). Must be contiguous and unique across all events.
* Convention: events are spaced by **100 ticks** of `fabula_time`. This gives
  `propagation_delay`, snapshot rebasing, and visual layout breathing room
  while keeping the ordering intuitive.

### AMWN tagging

Every node and edge inherits `world_id: Literal["factual", "shadow"]`. The
canonical graph is `factual`; counterfactual sandboxes spawn `shadow` mirror
nodes (see §4 below).

---

## 2. Phase 1 — World State & Initialisation (Steps 1–5)

### Step 1: Ontology ingestion ([`ingestion.py`](../shadow_loom/ingestion.py))

Five LLM agents extract a `GlobalRegister` from prose:

1. `extract_world_traits` → `WORLD_*` (governance, magic, environment, …)
2. `extract_entities` → `ENT_*` with initial trait vectors
3. `extract_locations` → `LOC_*` with ambient state
4. `extract_objects` → `OBJ_*` with affordances
5. fuzzy-resolve cross-chunk references (handles aliasing, partial names)

**A3 — Proposition catalogue.** A single global LLM pass
(`extract_proposition_catalogue_async`) extracts every `Proposition`
the narrative is *about* — outcome props (Will Macbeth become
king? Will Linnet die?), trait/relation/identity claims, and a
`ConcernSeed` list (per-entity polarity / salience anchored to a
PROP id). The catalogue is laid down once with full-text
attention so per-chunk extractors downstream can reference
canonical PROP / CCN ids without re-defining them. Truth
commitments and snapshot drift are deferred to the per-chunk
Affect sub-stage (B4 in §2 below).

### Step 2: Topology extraction (per-chunk)

For each text chunk the pipeline runs a **Socratic-QA scaffold** first
(Who/What/Where/When/Why/How pairs that articulate hidden motivations and
abductive inferences before structured extraction — see
[academic-foundations.md §6.5](academic-foundations.md#65-computational-narratology-and-story-understanding)
for the lineage), then dispatches three specialist agents:

* `PhysicsExtraction` — `EventNode`s (including `utterance` events), `CausalEdge`s, `SpatialEdge`s, fallback `EntityUpdate`s
* `SocialExtraction` — `RelationshipEdge`s, `Channel`s
* `ConsequencesExtraction` — authoritative `EntityUpdate`s anchored to the Physics events + mutation edges (default-on; overrides the Physics agent's own `entity_updates`). Toggle via `ExtractionConfig.enable_consequences_agent`.

In async mode Physics runs first; Social runs next (it is fed Physics's
event list and on-page entity ids); Consequences runs third so it can
wire belief provenance through Social's utterance / channel ids;
**Affect (B4)** runs last on chunks that touch a known
proposition / concern, emitting `PropositionSnapshot`s,
`truth_at_fabula` commits, `ConcernSnapshot`s, and weak-evidence
`new_concern_seeds` cited against this chunk's events. Chunks
are extracted in parallel under an `asyncio.Semaphore` gated by
`ExtractionConfig.max_concurrent_chunks` (default `12`); each chunk's
entire Socratic→Physics→Social→Consequences chain is wrapped in
`asyncio.wait_for(timeout=ExtractionConfig.per_agent_call_timeout_seconds)`
(default `600` s, `0` disables) so a wedged LLM call is cancelled
rather than holding the slot indefinitely. Each agent's output
validator runs a **sanitiser layer** that clamps numeric ranges, drops
self-loops, coerces status enum aliases, and fuzzy-fixes ID typos
before falling back to a `ModelRetry`. Per-chunk retries (axis, dyad,
mirror, anonymous-utterance, channel-zero, parity) **merge** their
outputs into the base via `_merge_physics_retry` /
`_merge_social_retry` / `_merge_anonymous_retry` /
`_merge_consequences_retry` rather than replacing them, so a retry that
targets one missing axis cannot silently drop the previously-extracted
edges. Results are merged in `assemble_world_state`, normalised
(`_normalize_fabula_times`), auto-repaired (`_auto_repair`), and
validated (`_programmatic_validation` → optional LLM correction
loop). Chunk order is treated as syuzhet order
only; each chunk's events keep their LLM-extracted `fabula_time`, so
flashbacks and flashforwards are preserved across chunk boundaries instead
of being re-sorted into reading order.

After assembly completes and before validation, three async passes run over the assembled state: `extract_entity_concerns_async` populates each entity's `concerns` list from the proposition catalogue; `cluster_belief_propositions_async` groups raw belief targets into canonical `PROP_*` references so downstream affect scorers can reason over named claims; and `_maybe_synthesise_audience_entity` injects a reserved `ENT_AUDIENCE` entity (the omniscient-reader perspective) when no audience entity was already present in the register.

`_programmatic_validation` runs `_validate_time_ordering`, which enforces four temporal invariants (contiguous unique `syuzhet_index`; reasonable `fabula_time` spacing; causal-edge cause-before-effect for `chain_reaction` edges; channel `established_at_fabula ≤ terminated_at_fabula`). A **fifth rule** (severity=error, category=temporal) was added for utterance temporal coherence: non-performative utterances (`truth_value ∈ {true, false, unknown}`) may not place `EVT_*` ids referring to future-fabula events in `target_ids` — if `target.fabula_time > utterance.fabula_time` the rule fires. Performative utterances (prophecies, vows, orders, declarations) are exempt because they posit or announce future states rather than report past ones; their downstream causal effects belong on `causal_topology` as `chain_reaction` edges, not in `target_ids`.

#### Step 3d — Optional external research (segregated, off by default)

After world-state assembly the pipeline can call an optional
`ResearchProvider` (currently Tavily; see
[shadow_loom/research.py](../shadow_loom/research.py)) once per topic in
`ExtractionConfig.research_topics`, distil each result through a
`research_extraction` agent, and append the result as a `WorldFact` to
`WorldStateV1.world_facts`. World facts are **structurally segregated**:
they live in their own field, never mutate `Entity` / `EventNode` /
`RelationshipEdge` / `GlobalTrait` namespaces, and only surface into
generation as a `BACKGROUND CONTEXT — NOT AUTHORITATIVE` block in the
Step 10 prompt. Off by default; opt in via
`EXTRACTION_ENABLE_RESEARCH_AGENT=true` plus
`EXTRACTION_RESEARCH_PROVIDER=tavily` plus `TAVILY_API_KEY`. Agents may
also add facts live via the `research_topic` MCP tool. Provider calls
are cached per-account (see
[CONTENT-POLICY.md §6.4 / §6.4a](../CONTENT-POLICY.md)).

### Step 3: Epistemic synchronisation

`Belief.established_at_fabula` records *when* a fact entered each character's
awareness; `Belief.acquired_via_event_id` and `Belief.acquired_via_channel_id`
track the provenance utterance and channel. The reader's awareness is
governed by `EventNode.syuzhet_index`. The ingestion prompts
(`prompts/social_extraction.md`) explicitly request both axes for any
communication event.

### Step 4: Director intent

`shadow_loom/query_models.py` defines eight query types — the user picks one to
drive a generation cycle:

| Query | Pearl rung | Purpose |
|---|---|---|
| `ObservationQuery` | 1 | "What happens next given the world as is?" |
| `InterventionQuery` | 2 | "What if X did Y?" — applies `do(X=y)`. |
| `CounterfactualQuery` | 3 | "What if the past had been different?" — abduction + intervention. |
| `DirectiveQuery` | — | High-level emotional target (`mystery`, `suspense`, `surprise`, `dramatic_irony`, plus emotion targets). |
| `InterrogationQuery` | — | Targeted Q&A over the typed graph ("who knows X?", "why does Y act?"). Read-only: dispatched through the same physics path as a rung-1 (Observation) query but the result is rendered by `shadow_loom/answer.py` as an `AnswerCard{answer, confidence, caveats, evidence_node_ids}` rather than persisted. |
| `GeneralQuery` | — | Free-form Q&A over the world model. Read-only, same `AnswerCard` flow as `InterrogationQuery`. |
| `ManualEditQuery` | — | User-supplied prose; routes to re-extraction without rendering. |
| `EvaluationQuery` | — | Recompute affective metrics across all versions for scoring. |

### Step 5: Ego-graph extraction ([`extract_graph.py`](../shadow_loom/extract_graph.py))

`extract_ego_graph_from_memory()` returns an `EgoGraphPayload` containing only
the slice relevant to the focal characters and time anchor:

* 1-hop spatial neighbours of focus locations
* co-located entities and objects
* causal edges within `memory_limit` recent events
* `RelationshipEdge`s and `Belief`s **time-sliced via
  `reconstruct_entity_at(temporal_anchor)`** so beliefs formed in the future
  do not leak in
* relevant `Channel`s and `WORLD_*` traits

This both saves LLM context window — addressing the well-documented
*lost-in-the-middle* effect in which long-context models
attend disproportionately to the start and end of the prompt
([Liu et al. 2024](https://aclanthology.org/2024.tacl-1.9/))
— and prevents temporal contamination of counterfactuals.

---

## 3. Phase 2 — Mathematical Simulation (Steps 6–8)

### Step 6: AMWN sandbox ([`instantiator.py`](../shadow_loom/instantiator.py))

`AMWNInstantiator.create_sandbox()` builds a NetworkX `MultiDiGraph` mirror of
the ego-graph. Mutations are confined to the sandbox until they are explicitly
committed back. Edge types laid down:

* `located_in`, `owned_by` — spatial / inventory topology
* `relationship` — psycho-social metrics, per-axis (`affinity` / `fear` / `power_dynamic` each with their own `value`, `inertia`, `evidence_strength`, `last_updated_fabula`)
* `causal` — `mechanism`, `evidence_strength`, `causal_force`,
  `propagation_delay`
* `connected_to` — spatial, with `is_locked` and `barrier_item_id`
* `communicating_with` — informational, derived from `Channel` participants and per-participant `intelligibility`. `Channel` carries `medium`, `directionality`, `participant_ids`, an `intelligibility` map, `established_at_fabula`, `terminated_at_fabula` (None while still open) and `evidence_strength`. (Hidden-channel discovery is tracked separately on `HiddenChannel.discovered_at_syuzhet` in the directive layer, not on `Channel` itself.)
* `eavesdropped_by` — auto-derived for participants whose `intelligibility >= physics.intelligibility_threshold` but who are *not* in an utterance's `addressee_ids` (epistemic leakage)

### Step 7: Causal Physics ([`causal_physics.py`](../shadow_loom/causal_physics.py))

`CausalPhysicsEngine` implements all three rungs of Pearl's Ladder of
Causation:

* **Abduction (rung 3 — Counterfactual).** Back-propagates present-day evidence into the
  historical sandbox. The default `abduction_blend_mode="bayesian"` blends
  each entity trait by a precision-weighted Bayesian update
  (trait inertia $\iota$ as the precision of the historical prior,
  evidence precision $\kappa$ on the present-day observation), with a
  legacy inertia-damped variant retained for ablation; beliefs propagate
  backward subject to per-channel intelligibility gating; relationship
  metrics use the same per-axis Bayes blend. `MECHANISM_TRAIT_MAP` gates
  which mechanisms can touch which trait families.
* **Action (rung 2 — Intervention).** Applies `do(X=x)` via six surgery types:
  spatial (with affordance path-checking), inventory, relationship, state
  mutation, genesis (spawn new nodes), and comms (open / sever channels).
  Incoming causal edges into the intervened node are severed; downstream
  edges are re-evaluated.
* **Propagation.** Topological sort over the causal sub-graph. Per-trait
  signed delta:

  $$\text{Impact} = (V_\text{source} - V_\text{current}) \times w$$

  A mutation only fires when $|\text{Impact}| > \text{Inertia}$. Spatial
  affordance gating blocks cross-location influence unless a path exists.
  Effective shift after dampening:

  $$\Delta_\text{effective} = \text{Impact} - \text{sign}(\text{Impact}) \times \text{Inertia}$$

Returned as `CausalPhysicsResult { mutations, blocked, hidden_deltas,
intervened_nodes }`.

### Step 8: Affective Calculus ([`directive_assembly.py`](../shadow_loom/directive_assembly.py))

Four structural-effect scorers operate purely on the graph geometry:

| Effect | Formula |
|---|---|
| **Mystery** | $\dfrac{\#\text{hidden ancestors}}{\#\text{total ancestors}}$ for each known effect; walks back through `causal_topology` and filters by ancestor events with `syuzhet_index > syuzhet_anchor`. |
| **Dramatic Irony** | Per-character mean of $\dfrac{\sum_{e \in R_t,\, e \notin K_c} w_e}{\sum_{e \in R_t} w_e + K}$, where $R_t$ is the set of events revealed to the reader by `syuzhet_anchor` $t$, $K_c$ is what character $c$ knows at that anchor, $w_e$ is event ``e``'s intensity (default 1.0), and $K = 1$ is a saturation constant. The score is therefore the *fraction of the reader's privileged view* the character is in the dark about — a Sternberg-style gap fraction. A character is treated as knowing an event when (a) they participate in it as actor or target *and* it has happened by the syuzhet anchor's fabula frontier, (b) a revealed utterance addressed to or spoken by them refers to it, or (c) they hold a `Belief` whose `target_id` matches the event id and whose provenance still resolves. Two earlier denominators failed: the *cumulative ratio over revealed-only edges* form plateaued by the third reveal because numerator and denominator grew together (Reservoir Dogs *decayed* from 0.25 to 0.06 as the protagonist became actor-of-record on more revealed edges); switching to the **full event mass** instead pinned the curve into a monotone rise across 21/21 example-world fixtures because the denominator stopped moving while the numerator kept growing — contradicting the rise-peak-fall arc Sternberg, Booth and Stanton predict for canonical irony plots (Macduff hearing of his family; Poirot's denouement; Nick's letter to Daisy). The current **revealed-mass + K** form restores the theoretical shape: it rises with new reveals and falls when participation, addressed utterances, or belief acquisition close the gap, producing rise-peak-fall in 16/21 worlds. |
| **Suspense** | $\text{balance} \times \text{stakes}$ clamped to $[0, 1]$, where $\text{balance} = 1 - \dfrac{|w_\text{threat} - w_\text{hope}|}{w_\text{threat} + w_\text{hope}}$ peaks at genuine outcome uncertainty and decays under one-sided dominance, and $\text{stakes} = \dfrac{w_\text{threat} + w_\text{hope}}{w_\text{threat} + w_\text{hope} + K}$ saturates so balanced fragments don't pin the gauge ($K = 2$ by default). For each focal entity, every unrevealed event in which the entity is a non-acting target contributes its `evidence_strength`-derived probability $p$ to $w_\text{threat}$, and every unrevealed event in which the entity is an actor contributes $p$ to $w_\text{hope}$. The probability proxy is the strongest incoming causal-edge weight on the event (outgoing as fallback, 0.5 default). Returns 0 at the **despair** boundary ($w_\text{hope} = 0$) and the **safety** boundary ($w_\text{threat} = 0$). The earlier asymmetric $\max(0, (w_\text{threat} - w_\text{hope})/(w_\text{threat}+w_\text{hope}))$ form collapsed to 0 on every fixture in which the protagonist authors most of their own forward events; balance × stakes follows Brewer & Lichtenstein's structural-affect framing of suspense as a response to outcome ambiguity. Inspired by Wilmot & Keller (2020); see [academic-foundations.md §3.1](academic-foundations.md#31-suspense-as-hopefear-here-hopethreat-anticipation--structural-affect-lineage). |
| **Surprise** | Per-trait binary KL divergence $D_\text{KL}(p \| q) = p\log\tfrac{p}{q} + (1-p)\log\tfrac{1-p}{1-q}$. Posterior $p$ is the entity's *final-state* trait value resolved via `reconstruct_entity_at(ent, t_max)` so authored `state_timeline` arcs are honoured (sandbox-preferred when running counterfactuals). Prior $q$ starts at the **leave-one-out** per-trait corpus marginal (mean across every *other* entity, falling back to 0.5 when fewer than two other entities carry the trait) — leave-one-out prevents the focal entity from biasing its own prior, which would otherwise collapse KL on the small casts typical of the example fixtures. The prior is then pulled toward the actual value by a geometric update $q \mathrel{+}= w \cdot (\text{actual} - q)$ for each revealed causal edge whose target is the entity, monotonically converging on the truth as evidence accumulates rather than overshooting. Each per-trait KL is run through a soft-saturation $1 - e^{-\text{KL}}$ (so perceptually meaningful KLs in the 0.2–1.5 band map to 0.18–0.78 of the gauge) and the result is averaged across the focal traits. The earlier $\text{avg}(\text{KL})/\log(1/\varepsilon)$ form divided by the *theoretical* binary-KL maximum ($\approx 4.6$ at $\varepsilon = 0.01$), squashing the entire perceptual signal into the bottom 4% of the gauge — every ``example_world`` plot read as flat ≤ 0.10 even when canonical surprise traits (Macbeth's despair, Macduff's grief) carried per-trait KLs of 0.27–0.50. See [academic-foundations.md §3.3](academic-foundations.md#33-surprise-as-kl-divergence). |

Six emotional effects (`grief`, `rage`, `joy`, `regret`, `love`, `fear`) use
trait-trajectory **closeness-to-target**: a single shared
``_EFFECT_TRAITS`` table maps each effect to ``(positive_indicators,
inverse_indicators)`` calibrated against the actual trait vocabulary of
the ``example_worlds/`` corpus (passion, tenderness, devotion, warmth,
longing, vengefulness, cruelty, vindictiveness, …). Positive traits
contribute their current value, inverse traits contribute one minus
their current value, and the score is the per-trait mean. Both lists
drive scoring symmetrically — the previous form's ``relevant`` filter
only included the increase list, leaving the decrease set as dead code
(a brave character routed through ``fear`` got no fear-reducing credit
because ``courage`` never entered the average). Falls back to the
worst-case ``+1.0`` only when an entity carries no traits matching
either list at all. Casts entirely lacking a measurable emotional
trait (``messianic_self_image``, ``moral_collapse``, ``class_anxiety``,
``pomposity`` …) — a deliberate stylistic choice on a few worlds —
still surface as the worst-case loss rather than silently scoring at
the midpoint.

`compute_affective_score()` returns the weighted combination requested by the
`DirectiveQuery`.

#### Propositional / Bayesian affect (`affect_unification.py`)

Alongside the trait-anchored scorers above, the post-2026
ingestion pipeline lays down a typed `Proposition` registry
(with `audience_default_prior`, `stakes`, time-indexed
`truth_at_fabula`, and a `state_timeline`) plus per-entity
`Concern` rows (polarity `desire`/`fear`, salience, activation
window, counter-concern pairs). On top of those,
[`affect_unification.py`](../shadow_loom/affect_unification.py)
exposes a second scoring layer that reasons over **belief
revision** rather than over graph geometry:

| Effect | Form (propositional) |
|---|---|
| **Suspense** | $\sum_P H(p_{\rm aud}(P, t_f)) \cdot \text{stakes} \cdot e^{-\Delta t / \tau}$ over open `outcome` propositions whose truth has not committed at $t_f$. |
| **Surprise** | $\sum_P D_{\rm KL}(p_{\rm aud}(P, t_f) \,\|\, p_{\rm aud}(P, t_f')) \cdot \text{stakes}$ — Itti–Baldi Bayesian surprise on the audience prior between two anchors. |
| **Dramatic irony** | $\sum_P D_{\rm KL}(p_{\rm aud}(P, t_f) \,\|\, p_c(P, t_f)) \cdot \text{stakes}$ — Pfister/Sternberg asymmetric KL between audience and focal character $c$. |
| **Mystery** | Erotetic mystery: Shannon entropy over softmax-normalised candidate causes of *known-to-audience* effect propositions. |

The trait layer (above) is what the **directive-assembly
optimiser** consumes — its loss semantics need a monotonically
declining graph-geometry signal that exists for every world.
The propositional layer is what the **affective dashboard** and
any Bayesian-narratology consumer surfaces — it answers *"how
surprised should the reader have been by this beat, given the
stated audience prior on `PROP_DUNCAN_DEAD`?"* — a question that
requires named propositions and is undefined for worlds whose
ingestion never produced a Phase A3 catalogue. See
[academic-foundations.md §3.7](academic-foundations.md#37-propositional-belief-revision-affect-affect_unificationpy).

#### What the four structural effects *mean*

The formulas above operationalise four narrative information-states. Each
asks a different question about the gap between what the reader knows,
what the character knows, and what is true:

* **Mystery — *the reader knows the effect but not the cause.*** Visible
  consequences whose causal ancestors are still off-page. High when the
  reader is staring at a corpse with no known killer; falls to zero once
  every cause behind every visible effect has also been revealed.
* **Dramatic irony — *the reader knows something the character does
  not.*** A sideways knowledge asymmetry. Counted when a causal source
  the reader has already seen is *not* in the focal entity's belief set
  at their `temporal_anchor`. Oedipus's audience knows the prophecy he
  doesn't.
* **Suspense — *the reader fears for someone whose outcome is still
  uncertain.*** Forward-looking. Threats are unrevealed events that
  *happen to* the entity; hopes are unrevealed events the entity itself
  *authors*. The score is **balance × stakes**: balance peaks under
  genuine outcome uncertainty (threat ≈ hope) and collapses under
  one-sided dominance, while stakes saturates so a tiny balanced
  fragment doesn't pin the gauge. The score is 0 at both the
  **despair** boundary (hope = 0, only inevitability remains) and
  the **safety** boundary (threat = 0, nothing left to fear).
* **Surprise — *the truth is not what the reader expected.*** Backward-
  looking prediction error. KL divergence between a prior built from
  the leave-one-out corpus marginal + revealed causes and the posterior
  given by the entity's reconstructed final-state trait values.

#### Behaviour along the two time axes

The four scorers are sampled along two independent axes by
[`viz_helpers.py`](../shadow_loom_ui/viz_helpers.py). The *cumulative*
surprise form (used by the directive optimiser) is monotone in the
syuzhet anchor; the *local* form (used by the time-series chart) is
not — see [academic-foundations.md §3.3](academic-foundations.md#33-surprise-as-kl-divergence)
for the dual-mode rationale.

* **Syuzhet axis** — the reader's progress through the *told* order.
  Advancing the syuzhet anchor only ever reveals more events.
  *Cumulative surprise* can therefore only move toward zero (the prior's
  geometric pull never overshoots the truth); *local surprise*
  $D_{\rm KL}(q_s\|q_{s-1})$ behaves Itti-Baldi-style instead, $\sim 0$
  on quiet stretches and spiking proportionally to each anchor's
  belief-update magnitude. Mystery decays as more causal ancestors come
  into view. Dramatic irony rises with each new reveal and falls as
  characters subsequently acquire knowledge — producing the
  rise-peak-fall shape the structural-affect literature predicts.
  Suspense falls as unrevealed threats / hopes are consumed.
* **Fabula axis** — the *story* order, sampled via
  `snapshot_world_at(t)`. Here the posterior trait values themselves
  change with the snapshot, so cumulative surprise *can* spike too —
  the fabula curve answers *which moments in the story are intrinsically
  surprising* (Macbeth's ambition flipping after the prophecy,
  Jacqueline's cruelty post-murder), whereas the syuzhet local-mode
  curve answers *which revelation moments most shift the reader's
  beliefs* (the Itti-Baldi 2009 reading).

---

## 4. Phase 3 — Generative Constraint (Step 9)

`DirectiveAssembler.evaluate_candidate_events()` forks the sandbox for every
candidate intervention, runs the physics engine, prunes anything that
violated affordance / inertia / propagation constraints, then ranks survivors
by the affective scorer. The winner is wrapped in a `CreativeBrief` with
typed `ConstraintBlock` entries:

* For **mystery** — hidden-predecessor counts and explicit "do not reveal" lines
* For **dramatic irony** — reader-vs-character asymmetry summary and "MUST NOT learn" guards
* For **suspense** — threat-vs-hope tension table, withheld-event protection list, hidden-channel list
* For **surprise** — per-trait KL magnitudes and belief-shattering revelation triggers
* For **emotions** — per-trait mathematical shift constraints with headroom and inertia evidence

The brief is the **only** thing handed to the renderer LLM in Step 10.

---

## 5. Phase 4 — Prose Generation (Step 10)

[`generation.py`](../shadow_loom/generation.py) calls a creative LLM with the
brief as system prompt + a small amount of style context. The renderer is
explicitly **not** allowed to invent causal edges or shift entity state — its
output is constrained to dialogue, description, and pacing within the
mathematical envelope.

### 5.1 Context flow into the renderer and the auditor

The same `CreativeBrief` is consumed by `assemble_rendering_prompt`
(generation) and `assemble_audit_prompt` (auditor). To keep the two views
in lock-step, every brief-construction site stamps the full set of
context fields, including:

* `original_query` — the user's verbatim NL request, lifted into a HARD
  constraint by all four brief paths (directive, observation,
  intervention, counterfactual) so the auditor flags prose that ignores
  it.
* `narrative_style` — the source-form profile that drives both the
  renderer's `STYLE FIDELITY` block and the auditor's `style_mismatch`
  word-band gate.
* `preceding_prose` + `branch_world_id` / `branch_label` /
  `factual_contrast_summary` — the story-so-far excerpt and the
  shadow-vs-canon framing, surfaced verbatim under matching headers in
  both prompts.
* `scene_context` — the ego-graph payload the renderer will see.
  `format_scene_context_for_prompt` normalises three input shapes
  (ego-graph dict, full `WorldStateV1.model_dump()`, sandbox
  `node_link_data`) so the auditor reads the same world the renderer
  did, including post-`do`-surgery state on Rung-2/3 paths.
* `scene_context["syuzhet_anchor"]` — the reader's narration position,
  stamped by every brief builder when an anchor is available. Used by
  the deterministic `withheld_utterance_leak` check inside `run_audit`
  so it can identify future utterances without re-deriving the anchor
  from `recent_memory`.
* `rendering` (RenderingDirective) and `physics_override` — surfaced
  to the auditor under `=== RENDERING DIRECTIVE ===` and
  `=== PHYSICS OVERRIDE (HARD) ===` so it can validate POV-lock
  breaches, pacing drift, and engine-authored hard text the renderer
  was told to honour verbatim.
* `hidden_channels` — the same brief-level withheld set the renderer
  receives is mirrored under `=== HIDDEN CHANNELS / UTTERANCES (HARD) ===`
  in the audit prompt so the auditor flags leaks against the
  syuzhet-aware view, not only the world-state-derived view.

For omniscient flows (POV-less observation, `interrogate`, `general`,
`manual_edit`, `evaluate`), `extract_full_world_state` accepts
`syuzhet_anchor` and prunes events whose `syuzhet_index` exceeds it,
so future-narration content cannot leak into the renderer or auditor
prompts even on the omniscient path.

For Q&A flows (`interrogate` / `general`), `answer.answer_question`
also threads `narrative_style` through the LLM context under
`=== SOURCE REGISTER (for tone / diction only) ===`, so the
`AnswerCard.answer` text mirrors the source's diction (formality,
character-name register, voice). Word-budget enforcement is
deliberately omitted: the `AnswerCard` is structured output, not a
prose chunk.

For the full-story `evaluate` flow, `assemble_evaluation_prompt`
mirrors the renderer's `=== STYLE FIDELITY ===` block (so the
combined prose is graded against the same source-register contract
each chunk was held to) and surfaces the `=== BRANCH CONTEXT ===`
block on shadow-branch evaluations (so a counterfactual fork is not
silently graded as if it were factual canon).

### 5.2 Engine-side exclusions (intervention & counterfactual)

For both Rung-2 interventions and Rung-3 counterfactuals the engine
surfaces `pruned_utterance_event_ids` and `disabled_channel_ids` —
utterances and channels whose provenance the do-surgery severed (e.g.
intervening on a speaker's `status` removes the lines they would have
spoken; severing a channel removes the messages it would have carried).
`build_intervention_brief` and `build_counterfactual_brief` both lift
each into a HARD constraint (`=== ERASED UTTERANCES ===` /
`=== DISABLED CHANNELS ===`) via the shared helper
`_build_exclusion_constraints`, so the renderer knows what *no longer
exists* in the intervened/counterfactual world, not just what does.
Without this, canonical lines kept in `preceding_prose` /
`factual_contrast_summary` reliably bled back into the prose. The
auditor reads the same blocks and flags leaks as `physics`-category
violations with rationale prefix `counterfactual_canon_bleed:`
(distinct from `withheld_utterance_leak`, which covers *future* lines,
not *erased* ones).

---

## 6. Phase 5 — Audit & Refinement (Steps 11–12)

[`auditor.py`](../shadow_loom/auditor.py) runs the LLM-as-judge over the
generated prose:

* **Causal audit** — reverse-engineers prose → causal claims; flags "Miracle
  Steps" (state changes with no licensing edge in the brief).
* **Abduction audit** — runs counterfactual probes against the prose to check
  implicit events hold up.
* **Affective audit** — measures the actual epistemic gap in the prose vs the
  intended target (e.g. did the renderer accidentally spoil a twist?).
* **Style-fidelity audit** — when the ingested source declared a
  `NarrativeStyle` profile, checks the rendered prose against the target word
  budget, prose density, register/POV/tense, and form class
  (`news_article`, `historical_account`, `thought_experiment`, `essay`,
  `case_study`, `transcript`); raises `style_mismatch` violations on drift.
* **Meta-narration audit** — runs on counterfactual / abduction-driven
  scenes. Flags prose that comments on its own counterfactual structure
  ("timeline", "divergence", "the alternative holds", `If he had…/would have…`
  framings, abstract aphorisms about fate or possibility) instead of rendering
  the alternate world as a lived past-tense scene; raises `meta_narration`.

If the auditor returns non-zero loss the refinement loop in
`pipeline.py::run_pipeline()` regenerates with the auditor's feedback
appended to the brief, up to `max_correction_retries`. On success the new
world state is committed back to the canonical graph.

**Audit-loop discipline.** The loop is engineered to avoid the classic
ping-pong failure mode where each iteration fixes one category and
regresses on the previous one:

* **Universal categories.** `meta` (meta-narration) and `style`
  (style-fidelity) audits run for every `target_effect` regardless of the
  brief's `audit_categories`; per-effect categories layer on top via
  `auditor.resolve_audit_categories(target_effect)`. The auditor prompt
  also instructs the judge to surface violations *only* for categories on
  the resolved list — anything else would be silently discarded by the
  loop.
* **Non-regression constraints.** Each refinement call passes the full
  list of *prior* violations from earlier iterations into the regeneration
  prompt under a dedicated `=== NON-REGRESSION CONSTRAINTS ===` section
  (deduplicated by `(violation_type, feedback[:160])`, excluding still-
  active types). The renderer is told explicitly to keep those fixes
  intact while addressing the current iteration's feedback.
* **Severity-aware short-circuit.** Style-fidelity is split into three
  severities: `critical` (form-class breach, e.g. `synopsis` rendered as a
  scene), `major` (word count >±50% off-budget or density+form drift), and
  `minor` (pure prose-density drift inside the form-class band). When
  *all* surfaced violations are `minor`, the loop short-circuits to
  `llm_passed=True` instead of burning another regeneration cycle on
  cosmetic refinement.
* **Form-class aware rendering.** For `summary` forms
  (`plot_summary` / `synopsis` / `outline`) and non-narrative forms
  (`news_article` / `historical_account` / `thought_experiment` / `essay`
  / `case_study` / `transcript`), `generation.assemble_rendering_prompt`
  emits a HARD override block ahead of the rendering directive so the
  scenic counterfactual / observation templates do not silently demand a
  lived past-tense scene the form-class forbids.
* **Evaluation "not measured" signal.** When `affective_loss_mse` is
  `None` (no scorable target exists), the evaluation prompt prints
  `Affective loss MSE: not measured (no scorable target — ignore in
  evaluation)` instead of silently formatting a misleading `0.0000`.

### 6.1 Sandbox-delta merge bridge

After audit converges, the pipeline re-extracts a `ChunkTopology`
from the rendered prose and merges it back into the
`VersionedWorldModel`. Re-extraction is lossy by design — the LLM
prose may not verbalise every physics-derived state change. To stop
those silent drops, `pipeline._augment_topology_with_sandbox_deltas`
bridges sandbox-only physics outputs directly into the topology
**before** merge. It currently bridges:

* `mutations` (`TraitMutation`) — entity / world-trait scalar
  updates anchored at the current fabula horizon; `WORLD_*` writes
  stack as `WorldTraitSnapshot` entries on the trait timeline.
* `hidden_deltas` (Rung-3 abduction) — entity / world-trait deltas
  anchored at the inferred historical past horizon, composed atop
  any same-tick mutations so multiple deltas accumulate rather
  than overwrite.
* `social_mutations` (`SocialMutation`) — per-axis
  `RelationshipEdge` writes plus, when the originating EVT id is
  known, a twin `mutation_social` `CausalEdge` so the relationship
  delta has the same provenance edge in `causal_topology` that a
  prose-extracted social mutation would.
* `disabled_channel_ids` — stamps `terminated_at_fabula` on a
  `Channel` copy so the channel-dedup pass in
  `_deduplicate_channels_with_map` collapses the canonical record
  to its severed lifecycle (next-query dialogue cannot route
  through a dead channel).
* `pruned_utterance_event_ids` + `disabled_channel_ids` → entity
  belief-invalidation cascade — for every belief whose
  `acquired_via_event_id` / `acquired_via_channel_id` references a
  pruned event or severed channel, an `EntityUpdate` with
  `invalidated_belief_targets` is emitted so the entity timeline
  records that the supporting evidence no longer exists in this
  branch.

`extract_graph.promote_sandbox_spawns` additionally promotes any
sandbox-spawned `Channel` capabilities (alongside `Entity` /
`NarrativeObject` / `Location` / `GlobalTrait`) into the merge so
that intervention queries introducing a new courier route or
mind-link survive even when the renderer never names the channel
in prose. Trait-mutation `triggered_by` provenance and Rung-3
counterfactual past-anchor propagation remain TODO; see the
audit notes in `/memories/repo/`.

---

## 7. Modules at a glance

| Module | Lines | Role |
|---|---|---|
| [`models.py`](../shadow_loom/models.py) | 440 | `WorldStateV1` schema + temporal reconstruction. |
| [`ingestion.py`](../shadow_loom/ingestion.py) | 8 649 | LLM-driven world extraction, normalisation, programmatic validation, correction loop with oscillation guard, error-relevant subgraph payloads, and final-pass validation snapshot. |
| [`extract_graph.py`](../shadow_loom/extract_graph.py) | 724 | Ego-graph slicing with temporal filters. |
| [`instantiator.py`](../shadow_loom/instantiator.py) | 592 | AMWN sandbox builder. |
| [`causal_physics.py`](../shadow_loom/causal_physics.py) | — | 3-rung CTF engine. |
| [`directive_assembly.py`](../shadow_loom/directive_assembly.py) | — | Affective scorers + brief assembly. |
| [`narrative_physics.py`](../shadow_loom/narrative_physics.py) | 1 216 | Pipeline orchestrator routing the eight query types. |
| [`generation.py`](../shadow_loom/generation.py) | 982 | LLM render + brief formatter. |
| [`auditor.py`](../shadow_loom/auditor.py) | — | LLM-as-judge with structured loss. |
| [`pipeline.py`](../shadow_loom/pipeline.py) | 1 223 | End-to-end runner with versioning. |
| [`query_parsing.py`](../shadow_loom/query_parsing.py) | 2 101 | Natural-language query → typed `Query*`. |
| [`db.py`](../shadow_loom/db.py) | — | SQLModel rows + `save_version` / `set_active_version` / version tree. |

---

## 8. Persistence

[`shadow_loom/db.py`](../shadow_loom/db.py) backs the system with SQLite by
default (`shadow_loom.db`).

* `ProjectRow` — name + owner + raw text.
* `VersionRow` — JSON-serialised `WorldStateV1`, with `ancestor_id` forming
  a **directed acyclic version tree**. `source` records how the version was
  created (`ingestion`, `pipeline`, `manual_edit`, etc.). Two extra columns,
  `world_id` (`factual` / `shadow`) and `branch_label`, tag every row with
  its AMWN branch so counterfactual forks can be browsed and promoted
  independently of the factual mainline.
* `ActiveVersionRow(project_id, user_id) → version_row_id` — the per-user
  pointer that the UI and MCP read by default.

Version writes go through `save_version(...)`; pointer updates through
`set_active_version(...)`. Deletion is rejoin-aware (`delete_version`
re-parents children atomically). `db.list_branches(project_id)` walks the
DAG and yields one summary per branch (root, head, fork-point ancestor,
version count); `db.promote_branch(version_row_id)` copies a shadow version
onto a new factual `VersionRow` whose ancestor is the current factual head.

Which branch a pipeline run lands on is decided by
`PipelineConfig.branch_policy: Literal["auto", "mainline", "shadow"]`
(default `"auto"`). Under `auto`, counterfactual queries fork to a fresh
shadow `world_id`; everything else stays on the factual mainline.
`VersionedWorldModel.merge(world_id=..., branch_label=...)` re-tags the
merged nodes/edges so per-branch retrieval stays clean.

---

## 9. The UI — `shadow_loom_ui/`

NiceGUI workspace ([`app.py`](../shadow_loom_ui/app.py)) composed of a
left-hand version sidebar (vertical tree only — the legacy radial layout
has been removed), a top-level **chat / command bar**, a dedicated
**Answer panel** above the chat bar for read-only Q&A results, and nine
cross-linked tabs in
[`components/workspace.py`](../shadow_loom_ui/components/workspace.py):

```
story · explorer · world · causality · reasoning · audit · research · editor · export
```

Every panel header carries an info-icon **help popover**
([`components/help_popover.py`](../shadow_loom_ui/components/help_popover.py))
opening a Markdown reference for that surface (what the panel does, how
to read its diagrams, what the controls mean). The Answer panel
([`components/answer_panel.py`](../shadow_loom_ui/components/answer_panel.py))
renders the result of a `GeneralQuery` or `InterrogationQuery` as a card
with claim, confidence badge, evidence-id list, and caveats; it clears on
`VERSION_CHANGED` and `PROJECT_LOADED` so a stale answer never persists
across versions.

`AppState` ([`state.py`](../shadow_loom_ui/state.py)) is the central
pub/sub. It debounces cursor scrubs (120 ms), gates panels by `active_path`,
and serialises panel rebuilds via `spawn_panel_task`. See
[ui-guide.md](ui-guide.md) for tab-by-tab walkthrough.

---

## 10. The MCP server — `shadow_loom_mcp/`

A FastMCP-based server ([`server.py`](../shadow_loom_mcp/server.py)) exposes
the world model as tools and resources. Highlights:

* `open_project` / `set_active_version` / `get_active_version`
* `run_and_save` — executes a query against the active version and
  auto-advances the active pointer to the new version row. Every response
  carries a `branch` envelope (`{world_id, branch_label, ancestor_id}`).
* Channel-aware reads: `list_channels`, `get_channel_history`, `who_can_hear`.
* Branch lifecycle: `list_branches`, `promote_branch`, `export_prose`
  (which can walk a specific lineage through the AMWN DAG).
  `promote_branch` rejects cross-project promotions: the source row's
  `project_id` must match the resolved project id from the caller's
  context, otherwise the tool returns an `error` without touching the DB.
* `get_entity`, `get_event`, `get_world_trait`, etc. — typed read tools.
* `helpers.load_world_state(pid, version, *, ctx=None)` resolves the active
  version per-user when `ctx` is provided; resources skip auth.
* The `async` tools `narrate` and `direct` dispatch the synchronous
  `run_and_save` call through `asyncio.to_thread` so other tools (and
  progress polling) keep being serviced while a multi-second pipeline
  run is in flight.
* `auth._token_user_cache` is a process-local cache mapping bearer
  tokens to `{user_id, scopes, key_id}`. `db.revoke_api_key` calls
  `auth.invalidate_token_cache(key_id=...)` so revocations take effect
  immediately rather than at process restart.

---

## 11. Tests

`tests/` contains roughly 1,300 pytest tests across 26 files covering models, ingestion, AMWN,
causal physics, directive assembly, narrative physics, version mutations,
reasoning helpers, viz helpers, the MCP server, and end-to-end pipeline
integration. `test_live_e2e.py` is excluded by default — it requires a local
Ollama instance.

```bash
python -m pytest tests/ --ignore=tests/test_live_e2e.py -q
```

---

## See also

* [pipeline-walkthrough.md](pipeline-walkthrough.md) — the same pipeline followed step-by-step at the **code** level, with every helper, short-circuit, and failure mode named.
* [query-and-cycles.md](query-and-cycles.md) — the eight query types and exactly how each one is realised inside the router described here in §2 (`narrative_physics.calculate_narrative_physics`).
* [mcp-guide.md](mcp-guide.md) — the agent-facing surface for everything in §10.
* [ui-guide.md](ui-guide.md) — the human-facing surface for everything in §9.
* [design-decisions.md](design-decisions.md) — *why* the schema looks the way it does.
* [academic-foundations.md](academic-foundations.md) — citations for fabula/syuzhet ([§1.1](academic-foundations.md#11-fabula-vs-syuzhet-fabula_time--syuzhet_index)), Pearl's ladder ([§2.1](academic-foundations.md#21-three-rungs-of-causation-observationquery-interventionquery-counterfactualquery)), Wilmot suspense ([§3.1](academic-foundations.md#31-suspense-as-hopefear-here-hopethreat-anticipation--structural-affect-lineage)), KL surprise ([§3.3](academic-foundations.md#33-surprise-as-kl-divergence)), AMWN, ctf-calculus ([§2.2](academic-foundations.md#22-ancestral-multi-world-networks-and-ctf-calculus--correa--bareinboim-icml-2025)).
* [settings.md](settings.md) — every runtime knob (model strings, token budgets, physics constants, audit thresholds) and how to override them.
