# Key Design Decisions

This document records the choices that shape Shadow-Loom and — equally
important — what we deliberately rejected. Each decision lists the
**alternative we considered**, the **tradeoff**, and the **invariant** it
locks in.

---

## D1. Graph-first, not prose-first

**Decision.** The canonical representation of a story is a typed Pydantic
graph (`WorldStateV1`), not a sequence of tokens.

**Alternative.** Pure-LLM systems (long-context generation, RAG over prose
chunks) keep prose as the source of truth.

**Tradeoff.** We spend extra LLM calls turning prose ↔ graph and we have to
maintain a schema. In return we get deterministic counterfactuals,
batch-friendly analytics, and the ability to validate consistency without
an LLM in the loop.

**Invariant.** Every generation cycle round-trips through the graph: prose →
graph → simulation → constraint → prose.

---

## D2. Fabula vs syuzhet are distinct first-class fields

**Decision.** Every event carries both `fabula_time` (chronological) and
`syuzhet_index` (presentation order); reader-side discovery is derived from
the utterance event's `syuzhet_index` while character-side belief uptake is
recorded on `Belief.established_at_fabula`.

**Alternative.** A single timestamp would be simpler.

**Tradeoff.** Schema complexity, mandatory normalisation in
`_normalize_fabula_times`. In return we can compute mystery, dramatic irony
and surprise as graph queries (they reduce to set operations between the
fabula and syuzhet projections).

**Invariant.** `syuzhet_index` is contiguous + unique across all events;
`fabula_time` is spaced by 100 ticks per distinct chronological position.

---

## D3. The "Hybrid 4+5" temporal entity state model

**Decision.** `Entity.traits / beliefs / status / location_id` represent the
**initial pre-story baseline**; mutations are stored as sparse deltas in
`Entity.state_timeline: List[EntityStateSnapshot]`. Pure-function
`reconstruct_entity_at(entity, fabula_time)` replays the deltas on demand.

**Alternative.** A) Store only the latest state and lose history. B) Store
full snapshots at every timestep (event sourcing without compression).

**Tradeoff.** Replays cost O(n) per query, but n is tiny in practice
(<100 events for a chapter) and the LLM only needs to emit the *changed*
fields, which keeps the extraction prompt size linear in mutations rather
than in entity count.

**Invariant.** `state_timeline` is monotonic in `fabula_time`. Reconstruction
is deterministic and side-effect-free.

See [architecture.md](architecture.md) and [pipeline-by-example.md§ 1.3](pipeline-by-example.md#13-entities--traits-beliefs-state-timeline)
for the on-disk shape and a worked Macbeth example.

---

## D4. Single `CausalEdge` class with five modalities

**Decision.** One `CausalEdge` model carries `causality_type ∈
{chain_reaction, mutation, mutation_social, affordance_gate,
ambient_propagation}`. A `model_validator` enforces that the prefix of
`source_id` / `target_id` matches the modality (e.g. `mutation` requires
`EVT_*` source and non-event target).

**Alternative.** Five separate edge classes.

**Tradeoff.** A bigger validator function and a slightly fatter prompt for
the LLM. In return: a single iteration loop in the physics engine, single
storage container in `WorldStateV1.causal_topology`, and single visualisation
codepath.

**Invariant.** `causality_type` is the source of truth — every consumer
branches on it; the prefix rule is a sanity check, not the dispatch.

---

## D5. `WORLD_*` (`GlobalTrait`) nodes are intervenable, not immutable

**Decision.** World-level facts (a war, a magic system, a regime) are
modelled as nodes with their own `state_timeline`, can be queried by
`do(WORLD_X = …)`, and **are skipped by abduction** (treated as exogenous
context, not as evidence).

**Alternative.** Hard-code world rules as Python constants.

**Tradeoff.** The graph gets bigger and the modal-validator on `CausalEdge`
has to special-case `WORLD_` source IDs.

**Invariant.** Counterfactuals can ask "what if the war ended?" and propagate
the consequences through `affected_domains` and `MECHANISM_TRAIT_MAP`. See
[academic-foundations.md §2.2](academic-foundations.md#22-ancestral-multi-world-networks-and-ctf-calculus--correa--bareinboim-icml-2025)
for the AMWN / ctf-calculus framework this implements.

---

## D5b. Decomposed Step 3 ingestion (Physics + Social + Consequences)

**Decision.** Per-chunk topology extraction is split across three
specialist agents, preceded by a **Socratic-QA scaffold** (see
[academic-foundations.md §6.5](academic-foundations.md#socratic-qa-scaffolding-ingestion-step-2)):

* **Step 3a — Physics agent**: events, causal edges, spatial edges.
* **Step 3b — Social agent**: relationship edges plus `Channel` nodes (the
  speech-act surface area; standing communication capability between
  participants, with a per-recipient `intelligibility` map). Discrete
  utterances themselves are extracted as `EventNode`s with
  `event_type="utterance"`, carrying `speaker_id`, `addressee_ids`,
  `via_channel_id`, `truth_value`, and `content` so beliefs can record
  explicit `acquired_via_event_id` / `acquired_via_channel_id` provenance.
  This replaces the legacy `InformationEdge` collection — see
  [scripts/migrate_information_edges.py](../scripts/migrate_information_edges.py)
  for the one-shot migration of older databases.
* **Step 3c — Consequences agent**: `EntityUpdate`s (trait/belief/status/location deltas) anchored to the events + mutation edges Physics produced. Default-on; overrides Physics's own `entity_updates` when enabled. Toggle: `ExtractionConfig.enable_consequences_agent`.

**Alternative.** A single "do everything" agent per chunk — the original
design — or a two-agent split (Physics + Social) where Physics also
emitted `EntityUpdate`s as a side-task.

**Tradeoff.** Three LLM calls per chunk instead of one or two. Mitigated
by running Steps 3b and 3c concurrently in async mode (both depend only
on the Physics output) and by the dramatic quality improvement: each
agent now has a single focused contract with a contract-headed prompt
and a sanitiser-equipped output validator that clamps numeric ranges,
drops self-loops, coerces status enum aliases, and fuzzy-fixes ID typos.

**Invariant.** Every Physics `mutation` / `mutation_social` edge that
targets an entity should have a corresponding `EntityUpdate` from the
Consequences agent. The Consequences validator audits this parity and
logs gaps without retrying (the agent already has every mutation listed
in its system prompt).

---

## D5c. Per-metric `RelationshipEdge` (axis-level inertia, evidence, staleness)

**Decision.** A `RelationshipEdge` no longer carries flat top-level
`affinity` / `fear` / `power_dynamic` numbers each governed by a single
edge-level `inertia` and `evidence_strength`. Instead it carries a
`metrics: dict[Literal["affinity","fear","power_dynamic"],
RelationshipMetric]` where every observed axis owns its own
`value`, `inertia`, `evidence_strength`, `last_updated_fabula`, and
`observed` flag. The axis vocabulary stays closed so the physics router
in `causal_physics.propagate_social` and the ingestion sanitiser in
`ingestion._sanitize_relationship_edge` remain deterministic.

**Why.** The old shape collapsed three independently-evidenced channels
into one inertia/evidence pair, throwing away signal the Bayesian
abduction layer was already capable of using:

- A scream is **strong** evidence of a `fear` spike but only **weak**
  evidence of a `power_dynamic` shift. The engine routes per-axis
  `mutation_social` CausalEdges with their own evidence weights, but
  they were being homogenised when written back to the edge.
- `fear` is **volatile** (settles within scenes), `affinity` is
  **moderate**, and `power_dynamic` is **institutional** (calcifies
  over arcs). Forcing one inertia value penalised either fear (stuck
  too high) or power dynamic (oscillates noisily).
- Counterfactual time-slicing intervening on a long-stale `power_dynamic`
  event was incorrectly discarding fresh `fear` mutations on the same
  edge because both were keyed off a single `last_updated_fabula`.

Distinguishing observed-zero from unobserved is also useful: an axis
the extractor never witnessed simply omits its key from `metrics`,
which the propagator and viz layer can treat differently from a
neutral `value=0.0`.

**Backward compatibility.** A `model_validator(mode="before")` on
`RelationshipEdge` accepts the legacy flat kwargs (and serialised legacy
JSON) and migrates them into `metrics`. Read-only `@property`
accessors expose `.affinity` / `.fear` / `.power_dynamic` / `.inertia`
/ `.evidence_strength` / `.last_updated_fabula` for every legacy
consumer (UI viz readers, MCP `get_relationships`, generation /
directive prompts, query parsing). A `to_legacy_dict()` flattens the
edge for the NetworkX sandbox so `causal_physics.propagate_social`
keeps reading flat dict keys. Setter sites
(`shadow_loom_ui/viz_helpers.py` time-slicing reconstructor, two
narrative-physics test mutators) had to be rewritten to mutate
`rel.metrics[axis].value` in place.

**Trade-off.** Prompts and the UI legend are slightly more verbose, and
the LLM extractor must now reason about per-axis evidence asymmetry.
That ergonomic cost is small relative to the variance signal recovered
downstream.

---

## D5d. Per-axis `mutation_social` coverage across all 20 fixtures (May 2026 rebuild)

**Decision.** Every observed `RelationshipMetric` axis (`affinity`,
`fear`, `power_dynamic`) on every relationship in the bundled
`example_worlds/*.py` corpus must be touched by at least one
`mutation_social` `CausalEdge` whose `trait_target` matches that axis.
A relationship that is purely inferential (no on-page event involves
both parties) is encoded with `observed=False, value=0.0` rather than a
static non-zero metric.

**Why.** The D5c migration left fixtures unchanged. Many edges
therefore carried *static* per-axis values --- a baseline `affinity` or
`power_dynamic` floated in from hand-authoring with no causal event
behind it. The
`shadow_loom_ui/viz_helpers.py` aggregator, which folds metrics over
`observed=True` axes only, then produced **flat** danger/conflict/power
gauges on roughly two-thirds of the corpus: the values were non-zero
but never moved, so the curve had nothing to plot. The regression test
`tests/test_affective_curve_evolution.py` was extended to enforce the
invariant axis-by-axis (previously fear-only, with twelve fixtures
grandfathered as `xfail`).

**What changed.**

- The three extraction prompts
  (`shadow_loom/prompts/{physics_extraction, social_extraction,
  validation}.md`) gained a *symmetric per-axis coverage rule*: when
  the social extractor emits a `RelationshipMetric` on any of the
  three axes, the physics extractor must emit at least one
  `mutation_social` edge with the matching `trait_target`. The
  validation prompt enforces the parity.
- An audit script
  ([`scripts/audit_static_axes.py`](../scripts/audit_static_axes.py))
  enumerates per-fixture *static dyads* (observed axes with no matching
  mutation) and lists candidate events involving both parties.
- A patch applier
  ([`scripts/apply_axis_patches.py`](../scripts/apply_axis_patches.py))
  performs idempotent text-surgery on a fixture `.py` to insert
  per-axis `CausalEdge` lines before the `causal_topology` block close
  (anchored on a sentinel comment) and to flip purely-inferential
  metrics to `observed=False`.
- All 20 fixtures were rebuilt: 120+ `mutation_social` edges added
  across the corpus, 14 inferential metrics flipped to unobserved.
  `KNOWN_FLAT_FEAR_FIXTURES` in the regression test is now empty;
  60/60 affective-curve tests pass with no `xfail`.

**Before/after qualitative example (Frankenstein).** Pre-rebuild, the
Creator-Creature dyad's `affinity = -0.8` and `power_dynamic = 0.3`
sat as static non-zero values: the UI showed a flat horizontal line
across the entire syuzhet timeline because no causal event ever wrote
to those axes. Post-rebuild, `EVT_CREATION` mutates the bidirectional
`power_dynamic` (Victor +0.40 / Creature -0.40) at fabula 3000;
`EVT_WILLIAM_MURDERED` collapses the Creature → William affinity to
-0.60 at fabula 4000; `EVT_CREATURE_REJECTED` writes the Felix →
Creature affinity (-0.90) and fear (+0.90) at fabula 8000. The conflict
gauge now traces the canonical hinge from rural pastoral through the
post-creation rupture to the Arctic chase, instead of showing a single
flat band.

**Trade-off.** Per-fixture surgery is artisanal --- the audit lists
candidate events but cannot decide on its own whether to write a
mutation or flip the dyad to unobserved. The applier amortises the
mechanical work; semantic placement was driven by per-fixture
re-reading of the `sample_plots/*.txt` source against existing
`EVT_*` ids.

---

## D5d. `evidence_strength` on `TraitVector` and `AmbientVector`

**Decision.** `TraitVector` (used for entity `traits` and
`GlobalTrait.magnitude`) and `AmbientVector` (used for
`Location.ambient_state`) carry an optional
`evidence_strength: Literal["weak", "moderate", "strong"] = "moderate"`
alongside `value`/`inertia` (or `value`/`volatility`). The same
`_coerce_evidence_strength` alias map used by `Belief`,
`Channel`, `CausalEdge`, and `RelationshipMetric` is applied,
so LLM synonyms (`high`/`low`/`certain`/…) are normalised before
Literal validation.

**Why.** The same argument that motivated `evidence_strength` on
relationship metrics applies to traits and ambient state: the engine's
confidence in the *extraction* is independent from the in-world
quantity (`value`) and the in-world stickiness
(`inertia`/`volatility`). A directly stated trait
("Macbeth is ambitious") deserves `"strong"`; one abduced from a single
hedged line ("seemed uneasy" → `low_courage=0.3`) is `"weak"`. Likewise
"the moor was bleak and cold" (strong) versus "probably damp given the
season" (weak) for ambient. Without this field the contradiction
checker, the abduction variance, and any future audit weighting cannot
distinguish a load-bearing extraction from a hedge.

**Backward compatibility.** The field is optional with default
`"moderate"`. Every existing fixture, snapshot, prompt response, and
serialised JSON payload continues to validate unchanged. Ingestion's
`_sanitize_entity_update` coerces aliases on `trait_updates` so LLM
output can hand back `"high"`/`"low"`/etc. without retry. MCP
`inspect_*` and the UI viz time-slicer surface the new key.

---
## D5e — Utterance temporal-coherence invariant

**Decision.** A non-performative utterance (`truth_value ∈ {true, false, unknown}`) may not list `EVT_*` ids in its `target_ids` whose `fabula_time` exceeds the utterance's own `fabula_time`. Performative utterances (`truth_value="performative"`) — prophecies, vows, orders, declarations — are exempt because they announce or posit future events rather than report past ones.

**Provenance.** The rule is enforced in three co-located places:
* [shadow_loom/ingestion.py](../shadow_loom/ingestion.py) — `_validate_time_ordering` Rule 5 (severity=error, category=temporal).
* [shadow_loom/prompts/social_extraction.md](../shadow_loom/prompts/social_extraction.md) — Rule 8 instructs the Social agent not to place future-fabula events in `target_ids` of non-performative utterances.
* [shadow_loom/models.py](../shadow_loom/models.py) — `EventNode.target_ids` field docstring notes the performative exemption.

**Why not allow it?** A non-performative utterance is a *report* about what has already happened — it is evidence for character beliefs about past events. Placing a future event in `target_ids` would cause the belief-propagation cascade to treat the future event as a known, reportable fact, contaminating entity belief timelines and violating the fabula-time ordering that the affect scorers and d-separation checks depend on.

**Downstream causal effects.** When an utterance *causes* a future event — Amy's lie triggering Nick's actions days later — that relationship belongs on `causal_topology` as a `CausalEdge(causality_type='chain_reaction')`, not in the utterance's `target_ids`. `target_ids` on an utterance should reference only the events the utterance *describes* (already occurred or simultaneous) — or, for performatives, the prospective commitment being announced.

**Invariant.** For all utterance events `u` where `u.truth_value ≠ "performative"` and all `tid ∈ u.target_ids` where `tid.startswith("EVT_")`: `events[tid].fabula_time ≤ u.fabula_time`.

---
## D6. AMWN sandboxing for all simulation

**Decision.** Every Pearl rung-2 (Intervention) / rung-3 (Counterfactual) query runs against a NetworkX
`MultiDiGraph` mirror with `world_id="shadow"` tags, never the canonical
`WorldStateV1` directly. Commits are explicit.

**Alternative.** Mutate `WorldStateV1` in place and roll back on failure.

**Tradeoff.** A copy on every query.

**Invariant.** A failed simulation cannot corrupt the canonical graph.
Concurrent counterfactual exploration is safe by construction (each query
gets its own sandbox).

**Branch isolation at the merge boundary.** Sandbox commits go through
[`VersionedWorldModel.merge(world_id=...)`](../shadow_loom/extract_graph.py),
which enforces the invariant on every persisted write — not only on the
sandbox round-trip. A merge running under one branch may not delete,
backfill, snapshot, supersede, or affect-update a holder tagged with the
*other* branch. Each blocked write is logged at INFO level with a tagged
prefix (`[merge·delete]` / `[merge·genesis]` /
`[merge·entity-update]` / `[merge·object-update]` / `[merge·belief]` /
`[merge·affect]` / `[merge·supersede]`). Replay helpers
(`reconstruct_entity_at`, `reconstruct_object_at`,
`reconstruct_entity_at_causal`, `reconstruct_world_trait_at_causal`)
also filter snapshots and causal mutations whose `world_id` differs from
the holder's, providing a defence-in-depth read path for any data that
predates the write-side guards. The full enforcement matrix is exercised
by [`tests/test_rung_audit_fixes.py`](../tests/test_rung_audit_fixes.py).

---

## D7. The LLM is a renderer, not an author

**Decision.** Steps 1–9 are deterministic Python (with LLM-driven
extraction in step 1). Step 10 (prose) is the only place a creative LLM is
trusted with output, and it is constrained by a typed `CreativeBrief` that
encodes:

* every mandatory state mutation,
* every belief that must / must-not form,
* every event that must remain hidden,
* every information channel that must remain undiscovered.

**Alternative.** Let an LLM both reason and render in one call.

**Tradeoff.** Multi-stage pipeline, worse latency.

**Invariant.** No new causal edges, traits or events appear in prose that
were not licensed by the brief. The Step-11 auditor is the enforcement
mechanism.

---

## D8. LLM-as-judge nested learning loop (Phase 5)

**Decision.** After every render, an auditor LLM reverse-engineers the prose
and computes loss against the brief; if non-zero, the renderer is run again
with the auditor's notes appended.

**Alternative.** Trust the renderer to follow instructions on the first try.

**Tradeoff.** Up to `max_correction_retries` extra LLM calls per scene.

**Invariant.** A scene cannot be committed to the canonical graph until the
auditor reports zero loss.

---

## D9. Inertia-gated mutation (`|Impact| > Inertia`)

**Decision.** Causal propagation only fires a mutation when the signed
delta exceeds the inertia of the target trait. The effective shift after
firing is dampened by inertia: $\Delta_\text{eff} = \text{Impact} -
\text{sign}(\text{Impact}) \cdot \text{Inertia}$.

**Alternative.** Linear propagation that always fires.

**Tradeoff.** Some "intuitive" mutations get blocked and must be authored
explicitly. In return: trait values stop drifting under chains of weak
causes (no exploding emotion values), and `inertia` gives authors a knob to
make a character "stubborn" / "fragile" without rewriting events.

**Invariant.** Inertia bounds the noise floor; the system converges.

---

## D10. Manual editing is first-class — versioning, not destructive

**Decision.** The Editor tab persists every save as a new `VersionRow` with
`ancestor_id` pointing at the previous version, `source="manual_edit"`. The
version tree is browsable in the version sidebar. Same `_programmatic_validation`
that gates ingestion gates manual saves.

**Alternative.** Inline mutation of the active row.

**Tradeoff.** Storage grows linearly with edits.

**Invariant.** No edit, however large, is destructive. `ancestor_id` is
never null for non-root versions. Validation errors block save; warnings can
be acknowledged.

See [ui-guide.md](ui-guide.md) §"Editor tab" for the user-facing design.

---

## D11. Active-version pointer separates "what is shown" from "what was last written"

**Decision.** Per-(project, user) `ActiveVersionRow` points at one
`VersionRow.id`. Both the UI and the MCP server resolve "current world" via
this pointer.

**Alternative.** Always show "latest version".

**Tradeoff.** A second table to maintain.

**Invariant.** The UI and MCP read tools always agree on which version is
"the world" for a given user. Users can pin an older version without losing
newer ones.

---

## D12. Schema invariants enforced at three levels

**Decision.** Three rings of validation:

1. **Pydantic** — types, enums, required fields, `model_validator`s
   (e.g. `CausalEdge` modality consistency, `mutation_social` requires
   `rel_counterpart_id`).
2. **`_programmatic_validation`** in [`ingestion.py`](../shadow_loom/ingestion.py)
   — cross-reference checks, broken edge endpoints, snapshot
   `triggered_by` resolution, monotonic timeline, syuzhet contiguity,
   dead-actor references.
3. **LLM auditor / validation agent** — semantic contradictions humans care
   about (does a character's behaviour match their stated trait?).

**Why three?** Each ring catches what the previous cannot:

* Pydantic catches *malformed* data.
* Programmatic catches *internally inconsistent* data.
* LLM catches *narratively wrong* data.

**Invariant.** The same `_programmatic_validation` is run by ingestion **and**
the manual editor — guaranteeing edits are held to the same bar as machine
extraction.

---

## D13. NiceGUI + AppState pub/sub instead of a JS framework

**Decision.** Pure-Python UI via NiceGUI, with `AppState.on(StateEvent.X)`
subscriptions for cross-tab updates. Rapid cursor scrubs are debounced
(120 ms). Heavy panel rebuilds go through `spawn_panel_task` which cancels
the previous task per panel id.

**Alternative.** React / Vue + a Python API.

**Tradeoff.** Less idiomatic for front-end developers.

**Invariant.** No JavaScript build step. Every UI change ships in a single
PR with the Python it depends on. State changes are observable from
backend tests.

---

## D14. SQLite + JSON column for world state, not a graph DB

**Decision.** `VersionRow.world_state_json` stores the entire serialised
`WorldStateV1` per version. Graph queries happen in NetworkX in process.

**Alternative.** Neo4j / a graph DB.

**Tradeoff.** No graph queries at the storage layer; entire world is
loaded per query.

**Invariant.** A single SQLite file is the entire database. Easy backups,
easy CI fixtures, no separate service.

---

## D15. Test fixtures are real classic plots

**Decision.** [`example_worlds/`](../example_worlds/) hosts hand-crafted
`WorldStateV1` fixtures for 20 well-known stories (Macbeth, Romeo & Juliet,
1984, Apocalypse Now, Gone Girl, Tinker Tailor Soldier Spy, The Lion the
Witch and the Wardrobe, …) — each with corresponding raw text in
[`sample_plots/`](../sample_plots/). Every fixture now ships standing
`Channel` objects (e.g. `CHN_TELESCREEN_BROADCAST`, `CHN_RHYS_FEYRE_BOND`,
`CHN_PIP_BENEFACTOR_PIPELINE`) plus first-class utterance `EventNode`s
with `content` / `target_ids` / `truth_value` / `via_channel_id` and
`Belief.acquired_via_event_id` / `acquired_via_channel_id` provenance,
so the regression suite covers deception, low-intelligibility leakage,
and performative speech-acts as well as plain causal physics.

**Why?** Validation feedback is meaningful when run against real narrative
shapes (mystery, betrayal, dual-timeline, ensemble, romance, conspiracy,
portal-fantasy). Synthetic fixtures hide the asymmetries the system is
designed to handle.

**Invariant.** All 20 fixtures pass `_programmatic_validation` with zero
errors and zero warnings — they are the regression baseline for any change
to the validator.

---

## D16. Channels and utterances are first-class — beliefs carry provenance

**Decision.** Speech, broadcasts, letters, eavesdropped conversations and
internal monologue are modelled as `EventNode`s with
`event_type ∈ {utterance, …}`, `content`, `target_ids`, `truth_value` and
`via_channel_id`, plus a `Channel` model with
`(participants, intelligibility, persistence, secrecy, observed_by)`.
Beliefs that arise from observation record **provenance** as
`Belief.acquired_via_event_id` and `acquired_via_channel_id`, and the
`compute_dramatic_irony_score` knows-set treats addressed/spoken-to
characters and beliefs-with-valid-provenance as having closed the gap.

**Alternative.** Treat dialogue as opaque prose only the renderer sees,
and infer character knowledge from a flat `participants` list on the
event.

**Tradeoff.** The schema gains two model classes and the extractor must
decide *who heard what through which medium*. In return:

* Deception is computable: a low-`truth_value` utterance on a
  high-`intelligibility` channel still produces a belief — just one
  whose ground-truth-mismatch the auditor flags.
* Eavesdropping and miscommunication are graph operations, not narrative
  hand-waving — `Channel.intelligibility=0.4` formally licenses the
  *partial* belief uptake that drives Tinker-Tailor-Soldier-Spy-style
  conspiracies and the Macbeth dagger soliloquy.
* The dramatic-irony scorer can credit a character with knowing an
  event the moment a revealed utterance addresses them, producing the
  rise-peak-fall arc Sternberg's theory predicts (see
  [academic-foundations.md §3.4](academic-foundations.md#34-dramatic-irony-as-epistemic-asymmetry)).

**Invariant.** Every `Belief` whose `acquired_via_event_id` is set must
reference an `EventNode` whose `target_ids`/`participants`/channel
membership actually places the believer in the audience; the validator
rejects orphan provenance.

---

## D17. Two affect layers: engine-grade vs heuristic

**Decision.** Affective scoring is split across two layers with
different cost/precision tradeoffs:

* **Engine-grade** (`directive_assembly.py::DirectiveAssembler`):
  `mystery`, `dramatic_irony`, `suspense`, `surprise`. These walk the
  full causal graph, consult the syuzhet anchor's revealed/unrevealed
  partition, and reconstruct entity state via `reconstruct_entity_at`.
  They are the scorers the directive-optimiser actually targets, and
  they are the ones with formal definitions in
  [academic-foundations.md §§3.1–3.4](academic-foundations.md#3-computational-models-of-suspense-surprise-and-curiosity).
* **Heuristic** (`shadow_loom_ui/viz_helpers.py::compute_affective_scores`):
  `conflict`, `danger`, `narrative_tension`, `causal_density`. These
  read snapshot-local relationship state and event ledger statistics,
  cost ~milliseconds, and feed every gauge / slider / ego-graph the UI
  shows. They are intentionally cheap and robust on sparse data — at
  the cost of being *correlates* of the engine quantities rather than
  drop-in substitutes.

**Alternative.** Single affect layer everywhere — either pay the
engine cost for every UI tick, or run heuristics only and lose the
optimiser's ability to target named structural effects.

**Tradeoff.** Two implementations means two code paths to keep aligned;
the engine layer leaks complexity (focus sets, anchor pairs,
sandbox-vs-truth) into any caller that wants a "real" score. In
return the UI stays interactive at 60 fps even on 200-event worlds
while the optimiser gets the precise, theory-grounded signal it
needs to choose between candidate interventions.

**Invariant.** The engine scorers are the source of truth for any
artefact persisted to the DB (briefs, audit reports, exports). The
heuristic scorers are display-only and never feed back into a
write.

---

## D18. Surprise has two formally distinct modes

**Decision.** `compute_surprise_score(..., local: bool = False)` exposes
two surprise quantities corresponding to the two distinct quantities
[Itti & Baldi (2009)](https://doi.org/10.1016/j.visres.2008.09.007)
distinguish:

* **Cumulative** (`local=False`, the default consumed by the
  directive optimiser): $D_{\rm KL}(p \,\|\, q_s)$ between the
  reader's accumulated prior $q_s$ and the truth $p$. Monotone
  non-increasing on the syuzhet axis as evidence accumulates — the
  loss-function semantics the optimiser needs.
* **Local** (`local=True`, used by the time-series chart and any
  per-step "what happened *here*" view): $D_{\rm KL}(q_s \,\|\, q_{s-1})$,
  the canonical Itti-Baldi *Bayesian Surprise* (KL between the
  reader's belief immediately after and immediately before the
  current syuzhet anchor's revelations). Formally identical to the
  Storck/Hochreiter/Schmidhuber 1995 RDIA formulation acknowledged on
  the iLab page. Quiet stretches contribute ~0; revelations spike.

**Alternative.** Pick one mode and live with it. (Earlier code did
exactly that — the cumulative form alone, with `surprise →
catching-up gap` semantics — and the time-series chart consequently
displayed monotone declines that contradicted the spike-and-decay
shape the underlying narrative theory predicts.)

**Tradeoff.** Two formulas with different shapes — easy to misread
which one is "the" surprise score. Mitigated by always plumbing the
flag through cache keys and by labelling the chart explicitly.

**Invariant.** The directive optimiser never consumes local-mode
surprise (would invert the gradient direction); time-series builders
never consume cumulative-mode surprise (would hide the spikes the
chart exists to surface). See
[academic-foundations.md §3.3](academic-foundations.md#33-surprise-as-kl-divergence)
for the full derivation.

---

## D19. Bayesian-style geometric prior pull, not additive

**Decision.** When updating a per-trait Bernoulli prior with a
revealed causal edge of weight $w$, the surprise scorer applies the
*geometric* pull
$q \mathrel{+}= w\,(\mathrm{actual} - q)$
clipped to $[\varepsilon, 1-\varepsilon]$, rather than the additive
$q \mathrel{+}= w\,(\mathrm{actual} - q_0)$ that uses the original
base prior.

**Alternative.** The additive form is what a naive linear interpolation
suggests; it is also what a "weighted majority" reading of the
evidence would imply.

**Tradeoff.** The geometric form costs an extra subtraction per edge.
In return it *asymptotes on* the truth: each new piece of evidence
moves $q$ a fraction of the remaining gap. The additive form
**summits past** the actual value once $\sum w > 1$, producing a
non-monotonic surprise curve that contradicts the
"more-revealed → less-surprise" semantics the cumulative form (D18)
is built around. This was a silent bug in an earlier revision; the
fix is recorded in
[academic-foundations.md §3.3](academic-foundations.md#33-surprise-as-kl-divergence).

**Invariant.** $q$ is monotone in the direction of the truth across a
causally-consistent revelation sequence, and never crosses
$\mathrm{actual}$.

---

## D20. The directive optimiser owns affordance/inertia/propagation pruning, not the renderer

**Decision.** `DirectiveAssembler.evaluate_candidate_events()` forks
the AMWN sandbox per candidate intervention, runs the causal physics,
and **prunes** any candidate that violates inertia, affordance, or
propagation constraints *before* the surviving candidate is wrapped
in a `CreativeBrief` and handed to the renderer LLM. The renderer
sees only constraints that have been mathematically guaranteed to
satisfy the world's physics.

**Alternative.** Let the renderer attempt every candidate as prose
and rely on the auditor to reject implausible ones.

**Tradeoff.** Forking sandboxes per candidate is the most expensive
step of the pipeline (D6's AMWN cost multiplied by the candidate
count). In return:

* The renderer is never asked to dramatise an event the world
  cannot actually produce — saving the much larger cost of a
  failed render + audit cycle.
* The brief is a *closed-form envelope*: every constraint in it
  has a witness in the sandbox, so the auditor's role narrows to
  "did the prose stay inside the envelope?" rather than "is the
  envelope itself valid?".
* Compositional safety: a candidate that satisfies the surface
  affordance but propagates to a contradiction three causal hops
  later is caught here, not in the prose.

**Invariant.** Every `ConstraintBlock` in a `CreativeBrief` has a
verified witness in at least one branch of the sandbox; the auditor
never has to re-verify physics, only fidelity.

---

## D21. Audit-loop convergence safeguards and temporal-collapse cycle handling

**Decision.** Six safeguards short-circuit the two non-convergence
modes that surfaced on densely-constrained counterfactual queries
(May 2026 Star Wars audit), and the propagator's cycle detection
distinguishes *temporal-collapse* artefacts from real causal loops:

1. **Refinement-regression rollback.** Each iteration's violation
   set `{(type, evidence_quote)}` is snapshotted. If iteration
   *n+1* both strictly increases the violation count *and*
   introduces a violation type unseen at iteration *n*, the
   `FeedbackLoop` rolls back to iteration *n*'s prose and exits
   with a structured `correction_error`. Without this, the
   rewriter regularly closes a minor density drift while opening
   a major meta-narration leak and the user sees the regressed
   draft as the final output.
2. **`rendering_mode` is immutable across refinement.** The
   refinement agent is forbidden from mutating the brief's
   rendering mode; if it returns a mismatching mode the orchestrator
   rejects the rewrite as a generation error and exits the loop.
   Without this, mode flips silently desynchronise the auditor's
   rubric from the prose's intent and convergence becomes accidental.
3. **POV vs. summary-form mutex.** When a brief simultaneously
   requests a POV-locked perspective *and* a summary source-form
   (`plot_summary`, `synopsis`, `outline`) the two constraints
   are mutually unsatisfiable. The directive assembler picks a
   winner deterministically (POV wins for mystery / dramatic_irony
   / surprise / suspense / fear / regret / grief; summary wins
   otherwise) and records the decision in
   `brief.scene_context["pov_form_resolution"]` so the auditor
   enforces exactly one.
4. **Quantitative form-class rubric** (auditor + generation
   prompts). Each source format now carries explicit per-beat
   sentence-count, dialogue-token, and interior-monologue-token
   thresholds (e.g. `synopsis`: ≤2 sentences/beat, 0% dialogue,
   ≤5% interior monologue) so generator and auditor share a
   measurable contract instead of competing prose-style instincts.
5. **Flat-abduction skip.** When the Rung-3 abduction posterior
   returns shifts with `|delta| < 0.10` on every trait (typical
   when the propagator hits a noisy-OR-absorbed or cyclic-blocked
   cluster), the renderer's "render abducted shifts as observable
   cues" sub-directive is dropped. The prose is no longer asked
   to invent body-language for shifts the simulator itself called
   negligible — and which the POV auditor would immediately flag
   as diagnostic gloss.
6. **Erased-utterance fingerprint, not text.** The `ERASED
   UTTERANCES (HARD)` constraint block (counterfactual / intervention
   briefs) carries a *structural fingerprint* — speaker,
   addressees, target-ids, truth-value, channel — but **not** the
   canonical `content` of the erased utterance. Showing the
   verbatim text in a "do not echo this" instruction is the
   classic pink-elephant anti-pattern: it makes the line the
   most salient phrase in the renderer's context, and the rewriter
   either reproduces it or paraphrases its evidentiary logic. The
   deterministic
   `_withheld_utterance_leak_violations` check still reads
   `world_state.events[].content` directly, and the LLM auditor
   pattern-matches on the act-shape, so leak detection is
   unaffected; only the renderer's exposure is removed.
7. **Temporal-collapse-aware SCC handling.** An `affordance_gate`
   edge (`Entity → Event`) refers to the entity's *pre-event*
   state; a `mutation` edge (`Event → Entity`) refers to the
   *post-event* state. Collapsed onto a single entity node the
   two form a strongly-connected component that does not exist
   in fabula time. Cycle detection (in both `causal_physics.propagate`
   and ingestion's `_auto_repair`) therefore excludes
   `affordance_gate` edges from the cycle-detection view; any SCC
   that survives is a real defect. The ingestion repair pass
   iteratively breaks the lowest-`causal_force` edge in each
   surviving SCC until the propagation graph is acyclic, capped
   at 50 iterations.

**Alternative.** Iterate to `max_iterations` and accept whatever
the rewriter last produced; treat all SCCs as physics blockers
and emit a flat-distribution counterfactual.

**Tradeoff.** Each safeguard adds a small amount of bookkeeping
in the inner loop and a small amount of explicit metadata on
the brief. The combined effect on the May 2026 Star Wars
counterfactual sweep is to dissolve a 34-node SCC + 3-node SCC
into a single genuine 3-node loop (`Tarkin orders → Alderaan
destroyed → DEATH_STAR_TERROR → Tarkin orders`) which is then
broken by the iterative repair pass; abduction recovers a
non-flat posterior; the feedback loop converges or rolls back
deterministically rather than ping-ponging between competing
fixes.

**Invariant.** (i) The final scene returned from the feedback
loop is the best draft seen, never a strictly-worse rewrite.
(ii) The auditor's rendering rubric matches the prose's rendering
mode for every iteration. (iii) After ingestion repair, every
SCC remaining in the propagation graph (with `affordance_gate`
edges excluded) reflects a genuine extraction defect, not a
temporal-collapse artefact.

---

## D22. Events have an explicit spatial anchor (`EventNode.at_location_id`)

**The decision.** Every `EventNode` carries an explicit
`at_location_id: Optional[str]` naming the LOC_ where the event physically
happens. The implicit invariant the field encodes:

> *If something happens at a location the characters and objects involved
> are present together — unless they are communicating over a `Channel`.*

When `at_location_id` is `None`, the helper
`event_location_at(evt, ws, fallback="actor")` resolves an effective
location by reconstructing the primary actor's `location_id` at
`evt.fabula_time`. Backfill is therefore deterministic and never
hallucinated.

**Why explicit, not derived.** A location field on `EventNode` looks like
schema duplication of "where the actors are standing". It isn't:

* **Channel-mediated participation** breaks the derived view. Macbeth and
  Banquo can both *participate* in an utterance event over a courier
  channel without being co-located. The event has one physical site (where
  the words were spoken); the addressee is somewhere else, present only
  through the channel. Without `at_location_id`, the renderer cannot tell
  which participants belong on the page at the event's location.
* **Co-presence violations** become checkable. The auditor's
  `event_copresence_violation` and `event_copresence_omission` rules need
  a single source of truth for "where the event happened" so they can flag
  prose that stages a bound participant somewhere else, or that adds a
  phantom witness to the event's location. With derived locations, the
  ground-truth target shifts under the auditor's feet whenever an actor's
  snapshot timeline is rewritten.
* **Counterfactual surgery on space** becomes a typed operation.
  `DoEvent.new_at_location_id` rewrites the anchor and cascades
  `EntityStateSnapshot(location_id=...)` for bound participants in a
  single step — the same do-operator semantics already used for traits and
  status, generalised to space.
* **The Map sub-tab** can render a star (★) glyph at every windowed
  event's anchor and outline bound participants in yellow (correct
  co-presence) or red dashed (phantom witness / displaced actor). Without
  a single anchor, "where to draw the star" is ambiguous when actors split.

**Implementation invariants.**

* `at_location_id` is **optional** for backwards compatibility. Existing
  fixtures and re-ingested worlds continue to pass without explicit values
  — `event_location_at(..., fallback="actor")` resolves the effective
  anchor from the primary actor.
* On merge, `_apply_event_spatial_anchor_repairs` (in `extract_graph.py`)
  validates every `at_location_id` against `world.locations`, reconciles
  conflicts with the primary actor's reconstructed location at
  `fabula_time`, auto-inserts `EntityStateSnapshot(location_id=...)` for
  bound participants who were elsewhere, and records every rewrite on
  `MergeChangeset.events_relocated` /
  `MergeChangeset.copresence_repairs_applied`.
* `DirectiveAssembler` emits one HARD spatial `ConstraintBlock` per
  windowed event with `at_location_id`. Bound participants go to
  `must_be_present`; channel-mediated addressees go to `channel_exempt`;
  every other entity reconstructed elsewhere at `fabula_time` goes to
  `must_not_be_present`.
* The renderer's universal Rule 11 (in `prompts/generation.md`) makes
  **implicit co-location the default** — the prose does not have to name
  the location every paragraph; it only has to *not contradict* it.
  Explicit naming of a different location for a bound participant during
  the event is the violation, not the silence.
* Rung-1 reveals of the form `EVT_X = LOC_Y` (or `at LOC_Y`) on
  `observation_facts` are folded back into the event's anchor by the
  pipeline bridge, with co-presence cascade re-run on the next merge.

**Why optional rather than required.** Many extracted events (interior
monologue, an entity's status change inferred at a distance, a world-trait
mutation) have no obvious physical site. Forcing `at_location_id` would
either silently default to misleading values or block ingestion on
genuinely placeless events. Optionality + a deterministic fallback +
auditor-enforced consistency where the field *is* set hits the same
correctness target without the false positives.

---

## D-AUDIT-2026-05-26. Post-audit theory + safety hardening

A May 2026 audit (`AUDIT_2026-05-26.md`) reviewed the system against
Pearl 2009, Bareinboim/Correa/Ibeling/Icard 2022, Correa & Bareinboim
ICML 2025 (AMWN + ctf-calculus), and OWASP Top 10. The remediation
batch below was applied as a single coordinated set.

### D-A1. Pearl minimal surgery is axis-scoped

**Decision.** `do(ENT_X.traits.fear=0.9)` only severs incoming causal
edges whose `trait_target` is `fear` (or untyped legacy edges that
could plausibly carry mutations onto `fear`). Edges that explicitly
target a *different* trait axis on the same entity (e.g. `guilt`,
`loyalty`) are preserved.

**Alternative considered & rejected.** The previous behaviour cut
*every* incoming causal edge to the intervened entity, regardless of
which trait the intervention targeted. This violated Pearl's $G_{\bar
X}$ construction (the do-operator severs edges into $X$, not into the
parent node of $X$).

**Invariant.** Per-axis trait surgery preserves the rest of the
entity's causal parentage, so siblings of the intervened axis remain
causally responsive to the rest of the graph.

### D-A2. AMWN context preserves variable-level granularity

**Decision.** `_to_context()` keeps the full dotted intervention path
(`ENT_alice.traits.fear`), so AMWN node-shadowing distinguishes
trait-level interventions on the same entity. Without this Rule 2/3
ctf-reasoning silently collapsed non-overlapping interventions.

### D-A3. Relationship surgery severs `mutation_social` edges

**Decision.** `do(rel(A→B).affinity=…)` removes every incoming
`mutation_social` edge whose
`(rel_counterpart_id, trait_target)` matches the do-target, in
addition to pinning the metric. Pinning alone left phantom edges that
the AMWN saw as continuing dependencies.

### D-A4. Anchor-filter beliefs by `established_at_fabula`

**Decision.** `compute_epistemic_gaps` excludes beliefs whose
`established_at_fabula` is strictly later than the brief's fabula
frontier. Anachronistic beliefs no longer leak into briefs (a
character cannot "know" a future fact).

### D-A5. Two new auditor violation types

**Decision.** The renderer-output auditor now emits
`spurious_abduction` (Rung-2 prose invents background premises to
explain an intervention's aftermath that the brief never licensed)
and `premature_payoff` (a `withheld_cause` event is staged before its
`syuzhet_index`). Both reuse the existing rationale-prefix machinery
in `auditor.md` and reference text in
`shadow_loom_ui/reasoning_helpers.py::VIOLATION_EXPLANATIONS`.

### D-A6. Ingestion lanes restored end-to-end

Three silent drops at validator boundaries were repaired:
* affect `belief_snapshots` flow through to Phase-C reconcile;
* consequences `object_updates` and `world_trait_updates` reach the
  chunk topology;
* `ObjectStateSnapshot` is constructed with the correct
  `properties_set` / `properties_unset` fields.

### D-A7. MCP error envelopes + API key hashing

**Decision.** All MCP tools route errors through `_sanitised_error`,
returning a typed `{error, code}` envelope rather than the raw
exception text. API keys are stored with HMAC-SHA-256 + per-key salt
+ server-side pepper (`SHADOW_LOOM_API_KEY_PEPPER`), with an explicit
index on `key_hash`. `list_projects(user_id=None)` no longer leaks
non-example project metadata to unauthenticated callers.

### D-A8. Cascade delete obeys ancestry, not row id

`delete_version(cascade=True)` peels leaves from the descendant
parent-map iteratively instead of sorting by descending row id; this
remains correct after `reparent_version` rewrites ancestry across
insertion order.

### D-A9. Temporal acyclicity is strict

`do_causal_edge` refuses `source_ft == target_ft` for every causality
type except `chain_reaction` (the one type where simultaneity is the
intended modelling primitive). Previously equal-tick edges silently
formed directed cycles.

### D-A10. MCP contract drift cleaned

`narrate(mode=…)` now validates `mode` strictly (matching `ask`);
`manage(action="fork")` accepts the same defaulted name as the
granular `fork` tool; `set_active_version` requires editor scope on
both the granular and the manage-action surfaces.

These changes are exercised by additions to `tests/test_*.py` and do
not change the public Pydantic schema. See
[architecture.md §post-audit](architecture.md) and
[academic-foundations.md §2.x](academic-foundations.md) for the
underlying theory pointers.

---

## What we rejected

| Rejected | Why |
|---|---|
| **Token-level constrained decoding.** | Doesn't compose with structural narrative constraints (a forbidden *event* can be expressed in infinitely many surface forms). The audit loop catches what decoding can't. |
| **Reinforcement-learning fine-tuning of the renderer.** | Out of scope; we treat the renderer as a black box so the system runs against any frontier model. |
| **A custom DSL for queries.** | Pydantic models are the DSL. Users write Python; agents emit JSON. |
| **Storing snapshots for every entity at every event.** | Quadratic blowup. The "Hybrid 4+5" delta + reconstruct compromise is empirically much smaller. |
| **Soft warnings only in the editor.** | Manual edits could silently break references. We block on errors; warnings still allow save with explicit "Save Anyway". |

---

## See also

* [architecture.md](architecture.md) — the schema and runtime that *implement* the decisions above.
* [pipeline-walkthrough.md](pipeline-walkthrough.md) — see the implausibility short-circuit, the re-extraction failure mode, and the auditor / renderer split in action.
* [query-and-cycles.md](query-and-cycles.md) — *why* the eight query types exist as a closed set rather than a free-form prompt.
* [academic-foundations.md](academic-foundations.md) — the literature behind every named decision (graph-first, fabula/syuzhet, Pearl, AMWN, Wilmot, Halpern). Key anchors: [§1.1 fabula/syuzhet](academic-foundations.md#11-fabula-vs-syuzhet-fabula_time--syuzhet_index), [§2.1 Pearl's ladder](academic-foundations.md#21-three-rungs-of-causation-observationquery-interventionquery-counterfactualquery), [§2.2 AMWN](academic-foundations.md#22-ancestral-multi-world-networks-and-ctf-calculus--correa--bareinboim-icml-2025), [§3.1 Wilmot suspense](academic-foundations.md#31-suspense-as-hopefear-here-hopethreat-anticipation--structural-affect-lineage), [§4.3 LLM-as-judge](academic-foundations.md#43-llm-as-judge-audit-loop).
* [settings.md](settings.md) — runtime defaults that operationalise these decisions.
* [use-cases.md](use-cases.md) — the user-facing capabilities each decision unlocks (and the ones it forecloses).
