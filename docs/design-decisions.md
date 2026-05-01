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
keeps reading flat dict keys. The 16 `example_worlds/*.py` fixtures
were not touched. Only setter sites
(`shadow_loom_ui/viz_helpers.py` time-slicing reconstructor, two
narrative-physics test mutators) had to be rewritten to mutate
`rel.metrics[axis].value` in place.

**Trade-off.** Prompts and the UI legend are slightly more verbose, and
the LLM extractor must now reason about per-axis evidence asymmetry.
That ergonomic cost is small relative to the variance signal recovered
downstream.

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

## D6. AMWN sandboxing for all simulation

**Decision.** Every Pearl rung-2 / rung-3 query runs against a NetworkX
`MultiDiGraph` mirror with `world_id="shadow"` tags, never the canonical
`WorldStateV1` directly. Commits are explicit.

**Alternative.** Mutate `WorldStateV1` in place and roll back on failure.

**Tradeoff.** A copy on every query.

**Invariant.** A failed simulation cannot corrupt the canonical graph.
Concurrent counterfactual exploration is safe by construction (each query
gets its own sandbox).

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
`WorldStateV1` fixtures for 16 well-known stories (Macbeth, Romeo & Juliet,
1984, Apocalypse Now, Gone Girl, …) — each with corresponding raw text in
[`sample_plots/`](../sample_plots/).

**Why?** Validation feedback is meaningful when run against real narrative
shapes (mystery, betrayal, dual-timeline, ensemble, romance). Synthetic
fixtures hide the asymmetries the system is designed to handle.

**Invariant.** All 16 fixtures pass `_programmatic_validation` with zero
errors and zero warnings — they are the regression baseline for any change
to the validator.

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
* [academic-foundations.md](academic-foundations.md) — the literature behind every named decision (graph-first, fabula/syuzhet, Pearl, AMWN, Wilmot, Halpern). Key anchors: [§1.1 fabula/syuzhet](academic-foundations.md#11-fabula-vs-syuzhet-fabula_time--syuzhet_index), [§2.1 Pearl's ladder](academic-foundations.md#21-three-rungs-of-causation-observationquery-interventionquery-counterfactualquery), [§2.2 AMWN](academic-foundations.md#22-ancestral-multi-world-networks-and-ctf-calculus--correa--bareinboim-icml-2025), [§3.1 Wilmot suspense](academic-foundations.md#31-suspense-as-uncertainty-reduction--wilmot--keller-acl-2020), [§4.3 LLM-as-judge](academic-foundations.md#43-llm-as-judge-audit-loop).
* [settings.md](settings.md) — runtime defaults that operationalise these decisions.
* [use-cases.md](use-cases.md) — the user-facing capabilities each decision unlocks (and the ones it forecloses).
