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
`syuzhet_index` (presentation order); information edges carry
`discovered_at_syuzhet` separately from `established_at_fabula`.

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

See [`/memories/repo/hybrid-45-implementation.md`](../memories/repo/hybrid-45-implementation.md)
for implementation notes.

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
[`/memories/repo/architecture.md`](../memories/repo/architecture.md) §"CTF-Calculus".

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
