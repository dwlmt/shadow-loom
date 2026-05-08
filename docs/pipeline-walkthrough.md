# Pipeline Walkthrough — End to End

This document follows a single request through every stage of Shadow-Loom,
from raw prose all the way back to a committed, versioned world model. It is
the "code-tour" companion to [architecture.md](architecture.md).

The orchestrator is
[`shadow_loom/pipeline.py::run_pipeline`](../shadow_loom/pipeline.py). It
accepts **exactly one** of:

* `raw_text=` — a brand-new story (runs ingestion first)
* `world_state=` — an existing `WorldStateV1` (wrapped fresh in a versioned model)
* `versioned_model=` — an existing `VersionedWorldModel` (preserves history)

…plus a `query: UserRequest` (see [query-and-cycles.md](query-and-cycles.md))
and a `PipelineConfig`.

It returns a `PipelineResult` containing the prose, the physics state, the
auditor verdict, the versioned world model after merge, and a
`PipelineHistory` recording every step.

```
                ┌──────────────────────────────────────────────┐
raw_text  ───▶  │ Step 1: Ingestion (5-pass extraction)        │
                └──────────────────┬───────────────────────────┘
                                   ▼
world_state  ─▶ ┌──────────────────────────────────────────────┐
                │ Step 0: Wrap in VersionedWorldModel          │
                └──────────────────┬───────────────────────────┘
                                   ▼
                ┌──────────────────────────────────────────────┐
                │ Step 2: Narrative Physics (query routing)    │
                │   • observation/general/interrogate → return │
                │   • intervention/counterfactual → causal eng.│
                │   • directive → causal + affective + brief   │
                │   • manual_edit → skip to Step 6             │
                │   • evaluate → full-story scorecard, return  │
                └──────────────────┬───────────────────────────┘
                          implausible? ──── return early ─────▶
                                   ▼
                ┌──────────────────────────────────────────────┐
                │ Steps 3–4: Brief Assembly + LLM Generation   │
                └──────────────────┬───────────────────────────┘
                                   ▼
                ┌──────────────────────────────────────────────┐
                │ Step 5: Audit + Refinement Loop              │
                └──────────────────┬───────────────────────────┘
                                   ▼
                ┌──────────────────────────────────────────────┐
                │ Steps 6–7: Prose → Topology → Merge          │
                └──────────────────┬───────────────────────────┘
                                   ▼
                            PipelineResult
```

---

## Step 1 — Ingestion (raw text → `WorldStateV1`)

Only runs when the caller supplies `raw_text=`. Implemented in
[`shadow_loom/ingestion.py::run_extraction_async`](../shadow_loom/ingestion.py)
(async, parallel chunks).

The extraction is a **5-step LLM cascade** with a programmatic safety net.

### 1a. Global Coreference Pre-Pass — `extract_ontology_async`

Step 1a (locations) runs first. Steps 1b (objects) and 1d (world traits)
then run in parallel; Step 1c (entities) chains after 1b so the entity
agent receives the resolved Object Register for belief grounding.

| Sub-step | Agent | Output | Prompt |
|---|---|---|---|
| 1a | `_extract_locations` | `LocationRegister` (`LOC_*` with `ambient_state`) | `prompts/ontology_locations.md` |
| 1b | `_extract_objects` | `ObjectRegister` (`OBJ_*` with `affordances`) | `prompts/ontology_objects.md` |
| 1c | `_extract_entities` | `EntityRegister` (`ENT_*` with initial `TraitVector`s, `Belief`s, `status`, `location_id`) | `prompts/ontology_entities.md` |
| 1d | `_extract_world_traits` | `WorldTraitsRegister` (`WORLD_*` — magic system, regime, climate, …) | (focused world-trait prompt) |

Each register feeds into the next so entities can reference real `LOC_` /
`OBJ_` IDs at extraction time. Cross-chunk fuzzy resolution merges aliases
("Mr Darcy" / "Fitzwilliam Darcy") under a single canonical ID.

### 1b. Text Chunking — `chunk_text`

Splits the prose by `act_headings` (default) or `paragraph` strategy, padding
each chunk with the previous chunk's last `chunk_overlap_chars` characters so
references survive boundaries. Tunable via `ExtractionConfig.min_chunk_chars`.

### 1c. Per-Chunk Topology — `extract_topology_async`

For each chunk the pipeline runs a **Socratic scaffold** first, then
decomposes the structured extraction across **three specialist agents**
in sequence (Step 3a → 3b → 3c). Consequences depends on Social's
extracted utterance events and channels for belief provenance, so the
two run sequentially within a chunk; chunk-level parallelism (gated by
`ExtractionConfig.max_concurrent_chunks`, default 12) provides the
throughput. A per-chunk soft timeout
(`ExtractionConfig.per_chunk_timeout_seconds`, default `600` s) wraps
the entire Socratic→Physics→Social→Consequences chain in
`asyncio.wait_for`; a wedged LLM call is cancelled and that chunk
yields an empty `ChunkTopology` (with all stage flags marked failed) so
the rest of the run can proceed. Set to `0` to disable.

| Step | Agent | Output | Prompt |
|---|---|---|---|
| 2 | Socratic scaffold | `SocraticScaffold` — Who/What/Where/When/Why/How QA pairs that surface implicit motivations, hidden state, and abductive inferences before structured extraction. Inspired by the Socratic method (see [academic-foundations.md §6.5](academic-foundations.md#65-computational-narratology-and-story-understanding)) and modern Socratic-QA / chain-of-thought prompting. | `prompts/socratic_scaffolding.md` |
| 3a | Physics agent | `PhysicsExtraction` — `EventNode`s (including `utterance` events with `content`/`speaker_id`/`addressee_ids`/`via_channel_id`/`truth_value`), `CausalEdge`s, `SpatialEdge`s, plus a fallback set of `EntityUpdate`s | `prompts/physics_extraction.md` |
| 3b | Social agent | `SocialExtraction` — `RelationshipEdge`s, `Channel`s (with per-participant `intelligibility`) | `prompts/social_extraction.md` |
| 3c | Consequences agent | `ConsequencesExtraction` — authoritative `EntityUpdate`s (trait/belief/status/location deltas) anchored to Physics events + mutation edges. Toggle via `ExtractionConfig.enable_consequences_agent` (default **on**); when enabled it overrides Physics's own `entity_updates` output. | `prompts/consequences_extraction.md` |

Each agent's output validator runs a **sanitiser layer** that auto-clamps
numeric ranges (`causal_force ∈ [0, 10]`, `trait_delta ∈ [-1, 1]`,
relationship metrics ∈ schema bounds, `inertia` capped at `0.99` so
traits never freeze permanently), drops self-loops, coerces status
aliases (`deceased → dead`, `wounded → injured`, …), and fuzzy-fixes ID
typos. Only structurally unfixable IDs trigger a `ModelRetry`. The
Consequences validator additionally performs a mutation⇄`EntityUpdate`
parity audit and a dead-actor warning that surface silent quality losses
in the log.

`EntityUpdate` is the per-chunk delta that becomes an `EntityStateSnapshot`
on the entity's `state_timeline` — this is where the **Hybrid 4+5 timeline**
is born.

### 1d. Assembly — `assemble_world_state`

Merges all chunks' topologies with the global register into a single
`WorldStateV1`. Fabula times are pre-allocated per chunk so parallel
extraction can't collide.

### 1e. Normalisation — `_normalize_fabula_times`

If the LLM emitted small integers (1, 2, 3 …) or duplicates, this
re-spaces all events by `fabula_time_spacing` (default 1000 — the engine
itself uses 100; ingestion leaves wider gaps so flashbacks can be inserted
later).

### 1f. Auto-Repair — `_auto_repair`

Pure-Python pass that fixes mechanical issues without touching the LLM:
broken `source_id`/`target_id` references resolved by fuzzy match, exact
duplicates dropped, orphan edges deleted, syuzhet indices made
contiguous. Invalid `Entity.location_id` values are **cleared to `None`**
(rather than silently re-pointed at "the first location"), since
plausible-but-wrong geography is a worse failure mode than "unknown
location". Returns the list of repairs, which is later attached to
`ValidationReport.repairs` so downstream UIs can audit silent fixes.

### 1g. World-Trait Timelines — `extract_world_trait_timelines`

Step 5b: a **single focused LLM pass** that, given the assembled state,
emits inflection points for each `WORLD_*` trait (regime change, war ends,
seasons turn). These become `WorldTraitSnapshot` entries on
`GlobalTrait.state_timeline`.

### 1h. Programmatic Validation + Correction Loop

`validate_world_state` runs `_programmatic_validation` (hallucinated IDs,
broken links, contradictions, duplicates, orphans). If errors remain, a
**correction agent** is invoked with the error summary + the current
state. When the serialised state exceeds
`correction_subgraph_threshold_chars` (default 400 KB) the prompt is
switched to an **error-relevant subgraph** that includes the events,
causal neighbours, channels, spatial edges and social edges referenced
by the error ids — not just events — so non-event errors
(channel/spatial/social) still get the right context.

Up to `max_correction_retries` passes are run. Each pass returns a
structured status (`applied` / `empty_patch` / `regression` /
`agent_failed` / `apply_failed`):

* `empty_patch` — the agent intentionally signalled "no safe fix";
  the loop stops because re-asking will return the same answer.
* `regression` / `agent_failed` / `apply_failed` — transient; the loop
  burns one retry slot but continues so the **oscillation guard** can
  decide when to stop.
* The oscillation guard hashes the outstanding error set after each
  iteration and breaks the loop the moment the same set recurs (the
  patch keeps fixing X and breaking Y, then fixing Y and breaking X).

Each iteration's compact log (errors, status, applied changes) is fed
back into the next correction prompt under a `PRIOR CORRECTION ATTEMPTS`
block so the LLM can change strategy instead of re-emitting the same
patch shape. After every applied patch, `_normalize_fabula_times` and
`_auto_repair` re-run and validation re-checks.

A **regression guard** (`_is_correction_regression`) rejects any patch
that drops more than 50 % of any topology, any entities, any channels,
or any world traits — silent destructive hallucinations are kept out of
the persisted state.

The result is `(WorldStateV1, ValidationReport)`. After research and
narrative-style inference run, a **final-pass `validate_world_state`**
is invoked once more so the returned report always describes the
returned state (the accumulated `repairs` log is preserved across the
refresh). The pipeline records an `IngestionStepRecord` (event/entity/
location counts + `is_valid`) and wraps the world state in a fresh
`VersionedWorldModel` at version 0.

> **In short:** ingestion is five LLM passes plus a programmatic safety
> net plus a correction loop — designed so that the topology handed to
> the simulator is already well-formed before any physics runs.

---

## Step 0 — Resolve the world model

Whichever input was supplied:

* `raw_text` → just-built world state, fresh `VersionedWorldModel.from_world_state(...)`
* `world_state` → wrapped into a fresh `VersionedWorldModel` (no prior history)
* `versioned_model` → used as-is (history preserved, `vwm.current` becomes `ws`)

`PipelineResult.world_model` is set immediately so callers always have a
pointer back to the model even if a later step fails.

---

## Step 2 — Narrative Physics (`calculate_narrative_physics`)

Lives in [`shadow_loom/narrative_physics.py`](../shadow_loom/narrative_physics.py).
This is the **router**: each query type takes a different path through the
engine. See [query-and-cycles.md](query-and-cycles.md) for the full per-type
walkthrough; the headline is:

| Query | What this step does |
|---|---|
| `observation` | Multi-ego graph extraction (or full omniscient state if no POV). |
| `intervention` | Plausibility gate → AMWN sandbox → Rung-2 (Intervention) do-calculus → propagation. |
| `counterfactual` | Plausibility gate → AMWN sandbox → Rung-3 (Counterfactual) abduction → Rung-2 (Intervention) do → propagation. |
| `directive` | Sandbox → candidate enumeration → physics per candidate → affective scoring → `CreativeBrief`. |
| `interrogate` / `general` | Read-only Q&A: graph RAG / pathfinding only (no time advance, no prose, no version). |
| `manual_edit` | Bypassed entirely. |
| `evaluate` | Full-story scorecard branch (handled below). |

The router writes a `PhysicsStepRecord` into the history and returns a
`physics_result` dict containing:

* `status` — `"success"` or `"implausible"`
* `physics_state` — serialised graph snapshot
* `query_type`
* (optional) `creative_brief` for directives
* (optional) `_causal_physics_result` — the typed `CausalPhysicsResult` stashed
  for the auditor (not serialised into history)

### Implausibility short-circuit

If `status == "implausible"` and the user did **not** pass
`force_implausible=true`, the pipeline calls
`_apply_implausibility_short_circuit`:

* sets `result.implausible = True`
* surfaces `result.implausibility_reason` and `implausibility_details`
  (including `unresolved_targets`)
* records an `"implausibility"` history entry
* **returns immediately — the world model is not mutated, no version is
  written.**

If `force_implausible=true`, the engine returns an `implausibility_warning`
rather than a hard failure; the pipeline flags the result as implausible but
continues into generation.

---

## Step 2.5 — Early-return query types

Two query types stop here without rendering prose:

* `interrogate` and `general` — read-only Q&A. The pipeline calls
  `_run_answer_step`, which dispatches
  [`shadow_loom.answer.answer_question`](../shadow_loom/answer.py) on the
  compressed physics state. The resulting `AnswerCard` is folded back
  into `physics_result` as the keys `answer`, `confidence`, `caveats`,
  `evidence_node_ids`, plus a `proof` list of `{id, kind: "evidence"}`
  entries so `structured_response_data` picks the answer up. The MCP
  server's `ask` tool and the UI's **Answer panel** both surface those
  fields directly. **No version is written.**
* `evaluate` — the dedicated `_run_evaluation_branch` runs the
  `NarrativeOrderObject` scorecard (causal physics feedback + affective
  feedback + LLM literary critique) and returns. No new version is written
  because no narrative was produced.

A third branch, `manual_edit`, also skips Steps 3–5 — the user's prose **is**
the answer — and jumps directly to Step 6.

---

## Steps 3–4 — Brief Assembly + Generation

For prose-producing queries:

### Step 3 — `CreativeBrief`

For `directive` queries the brief is **already built** by
`DirectiveAssembler.evaluate_candidate_events()` inside Step 2 (it forks the
sandbox per candidate, runs physics, prunes constraint violators, ranks
survivors by the affective scorer, wraps the winner in typed
`ConstraintBlock` entries — see [architecture.md §4](architecture.md)).

For other prose queries the pipeline calls `_build_brief_for_query` to
synthesise a brief from the physics result (intervention deltas, mutations,
blocked propagations, hidden deltas).

Every brief-construction site stamps the same context fields so the
renderer (Step 4) and the auditor (Step 5) read the same world:
`original_query` (verbatim NL request as a HARD constraint),
`narrative_style`, `preceding_prose`, branch context
(`branch_world_id` / `branch_label` / `factual_contrast_summary`),
`scene_context` (with `syuzhet_anchor` stamped when available),
`rendering` (RenderingDirective), `physics_override`, and
`hidden_channels`. For both Rung-2 interventions and Rung-3
counterfactuals the engine's `pruned_utterance_event_ids` and
`disabled_channel_ids` are lifted into HARD `=== ERASED UTTERANCES ===`
and `=== DISABLED CHANNELS ===` constraints so the renderer knows what
*no longer exists* in the intervened/counterfactual world (canon lines
kept in `preceding_prose` / `factual_contrast_summary` would otherwise
bleed back in).

### Step 4 — Generation

[`shadow_loom/generation.py::render_from_query`](../shadow_loom/generation.py)
calls a creative LLM with the brief as the system prompt + minimal style
context. The renderer is **not** allowed to invent causal edges or shift
entity state outside the envelope; its output is constrained to dialogue,
description, and pacing.

When `cfg.skip_audit=True`, generation runs once and the loop is skipped.

---

## Step 5 — Audit + Refinement Loop

[`shadow_loom/auditor.py`](../shadow_loom/auditor.py) wraps generation and
audit into `render_and_audit` (for directives with a pre-built brief) or
`run_feedback_loop` (for non-directive queries). Three audits run in
parallel:

* **Causal audit** — reverse-engineers prose → causal claims; flags any
  "Miracle Step" (state change with no licensing edge in the brief).
* **Abduction audit** — runs counterfactual probes to verify implicit events
  hold up.
* **Affective audit** — measures the actual epistemic gap / emotional
  intensity in the prose vs the directive's target.

The audit prompt mirrors the renderer's view of the brief: alongside
constraints, scene context, and prior feedback it surfaces
`=== RENDERING DIRECTIVE ===`, `=== PHYSICS OVERRIDE (HARD) ===`, and
`=== HIDDEN CHANNELS / UTTERANCES (HARD) ===` so the auditor can
validate POV-lock breaches, engine-authored hard text, and
syuzhet-aware leak rules against the same brief the renderer
consumed. For counterfactual queries the
`=== ERASED UTTERANCES ===` / `=== DISABLED CHANNELS ===` blocks are
also surfaced; leaks against them are `physics`-category violations
with rationale prefix `counterfactual_canon_bleed:` (distinct from
`withheld_utterance_leak`, which covers *future* lines, not *erased*
ones).

The result is a `FeedbackLoopResult` containing:

* `final_scene` — `GeneratedScene`
* `converged` — bool
* `iterations` — count
* `change_impact` (`causal_feedback` + `affective_feedback`)
* `engine_thresholds_passed` and `engine_threshold_failures` —
  deterministic gate, separate from the LLM auditor's verdict

If the auditor returns non-zero loss the loop regenerates (up to
`max_correction_retries`) with the auditor's feedback appended to the brief.

The pipeline records a `GenerationStepRecord` and an `AuditStepRecord`,
populates `result.scene / prose / converged / audit_iterations /
feedback_result`, and proceeds.

---

## Steps 6–7 — Prose Re-extraction + Merge

This is what closes the loop and updates the world model from the prose the
LLM just wrote.

### Step 6 — `extract_topology_from_prose`

A focused LLM pass extracts only the **delta** topology from the new prose:
new events, new causal edges, entity updates, new beliefs. The prompt
explicitly forbids re-extracting nodes that already exist.

**Genesis spawns.** Before Step 6 runs, the pipeline calls
[`promote_sandbox_spawns(ws, physics_state)`](../shadow_loom/extract_graph.py)
to walk the Rung-2/3 sandbox for nodes tagged `world_id="shadow"` and
promote them into typed canonical records (`Entity`, `NarrativeObject`,
`Location`, `WorldTrait`). The resulting bucket is passed to
`extract_topology_from_prose(..., spawns=...)`, which (a) pre-registers
the new IDs in the `GlobalRegister` so the extractor LLMs use the
canonical IDs instead of inventing duplicates, and (b) attaches them to
the returned `ChunkTopology.new_*` fields so Step 7's merge records the
genesis in its `MergeChangeset`. This closes the loop on
`*.spawn` interventions: a directive that says "spawn a new character
called Banquo's son" creates `ENT_FLEANCE` in the sandbox, promotes it
to canonical via this step, and the rendered prose is re-extracted with
`ENT_FLEANCE` as a known ID.

**Generation-failure skip.** If `result.scene.generation_error` is set
(the renderer fell back to a placeholder scene because the LLM call
raised), Step 6 is skipped entirely — merging a `[Generation failed:
...]` placeholder into the canonical world state would pollute the
graph with junk. The pipeline sets `result.reextraction_failed = True`
with an explanatory `reextraction_error` so the UI and MCP can surface
the failure, and the prior world model is preserved unchanged.

### Step 7 — `VersionedWorldModel.merge`

Merges the topology delta into a versioned **deep copy** of the model:

* Tagged with `source="pipeline"` (or `"manual_edit"` for `write` tool calls).
* Stores the prose alongside the new version.
* Computes a `Changeset` (events_added, causal_edges_added,
  entity_updates_applied, entity_updates_skipped, plus the genesis
  counters `entities_added` / `objects_added` / `locations_added` /
  `world_traits_added` populated by `promote_sandbox_spawns`) recorded
  as a `ReextractionStepRecord`.
* Bumps `vwm.version`; the new `VersionedWorldModel` replaces
  `result.world_model`.
* **Branch routing.** `PipelineConfig.branch_policy`
  (`"auto"` / `"mainline"` / `"shadow"`, default `"auto"`) decides whether
  the merged version lands on the factual mainline or on a shadow fork.
  Under `"auto"`, counterfactual queries fork to a fresh shadow `world_id`
  and everything else stays factual; `merge(world_id=..., branch_label=...)`
  re-tags the merged nodes/edges so per-branch retrieval stays clean. The
  resolved `(world_id, branch_label, ancestor_id)` is surfaced on every
  pipeline result so the persistence layer can stamp it onto the new
  `VersionRow`.

If re-extraction or merge raises, the pipeline:

* logs the exception
* sets `result.reextraction_failed = True` + `reextraction_error = str(e)`
* leaves `result.world_model` pointing at the **prior** unmerged version
* returns the prose so the caller can show it, but the contract is that
  callers **MUST NOT persist this as a new canonical version** — prose and
  world state are out of sync.

The MCP `run_and_save` helper enforces this by refusing to call
`save_version` when `reextraction_failed` is true.

---

## After the pipeline — versioning at the boundary

The pipeline itself does **not** write to the database. Persistence is the
caller's responsibility:

* The UI's task helpers and the MCP `run_and_save` wrapper both call
  `save_version(...)` with the new world state JSON, the ancestor row id
  (the version the user was on), and `source="pipeline"`. The new row's
  `world_id` and `branch_label` come from the pipeline result, so a
  shadow-branch counterfactual lands on its own fork rather than
  overwriting factual canon.
* MCP responses additionally include a `branch` envelope
  (`{world_id, branch_label, ancestor_id}`); when the merged world model
  is byte-identical to the ancestor (e.g. a `direct` call that produced
  prose without graph advancement) `run_and_save` returns a
  `world_model_unchanged` / `version_skipped` flag instead of writing a
  duplicate row.
* `set_active_version(...)` then advances the per-user `ActiveVersionRow`
  pointer so subsequent calls resolve against the new tip.
* The `original_query` field on every query type is preserved and stored on
  the version row so the UI can show "what the user asked for".

This is what gives Shadow-Loom its **directed acyclic version tree**:
every cycle creates a new node, descendants always point back to a real
ancestor, and the auditor / re-extraction failure modes never silently
corrupt the canonical history.

---

## What ends up in `PipelineResult`

| Field | Source step |
|---|---|
| `prose` / `scene` | Step 4–5 |
| `physics_result` | Step 2 |
| `converged` / `audit_iterations` / `feedback_result` | Step 5 |
| `evaluation_result` | Evaluation branch |
| `world_model` | Step 0 (then overwritten by Step 7 on success) |
| `query_type` | Caller-supplied `query` |
| `implausible` / `implausibility_reason` / `implausibility_details` | Step 2 short-circuit |
| `reextraction_failed` / `reextraction_error` | Step 7 |
| `history` | Every step records into this |

Use `humanize_pipeline_result(result, requested_effect=…, requested_intensity=…)`
to render the result as plain-English bullets for end users (chat UI, MCP
envelopes, CLI logs).

---

## Async variant — `run_pipeline_async`

Behaves identically except that **Step 1 ingestion** uses
`run_extraction_async`, which parallelises per-chunk topology extraction
gated by `ExtractionConfig.max_concurrent_chunks`. All other stages
(physics, generation, audit, re-extraction) remain synchronous because they
already coordinate their own concurrency internally.

---

## See also

* [architecture.md](architecture.md) — the conceptual map of the same 12-step pipeline (data model, modules, persistence).
* [model-examples.md](model-examples.md) — each step illustrated on real bundled plots (Macbeth, Death on the Nile, Reservoir Dogs, …).
* [query-and-cycles.md](query-and-cycles.md) — the **per-query-type** walkthrough of Step 2 (router) through Step 7 (merge).
* [mcp-guide.md](mcp-guide.md) §5 — the `run_and_save` versioning contract that wraps every `run_pipeline` call from MCP.
* [design-decisions.md](design-decisions.md) — *why* re-extraction is mandatory, *why* implausibility short-circuits, *why* the auditor is separate from the renderer.
* [academic-foundations.md](academic-foundations.md) — the literature behind ingestion's Socratic scaffold ([§6.5](academic-foundations.md#65-computational-narratology-and-story-understanding)), the auditor's LLM-as-judge protocol ([§4.3](academic-foundations.md#43-llm-as-judge-audit-loop)), and the merge's changeset model.
* [settings.md](settings.md) — every per-step `*Config` value (`GENERATION_*`, `EXTRACTION_*`, `AUDITOR_*`, `PHYSICS_*`) and the env vars that override them.
* [paper/shadow_loom.pdf](../paper/shadow_loom.pdf) **Appendix B** (`app:walkthrough`) — the same loop described as long-form prose on the *Macbeth* fixture, with every intermediate object (`GlobalRegister`, ego-graph, `CausalPhysicsResult`, `CreativeBrief`, `AuditResult`) shown step by step.
* [paper/shadow_loom.pdf](../paper/shadow_loom.pdf) **Appendix A** (`app:defs`) — the formal definitions and equations underlying each step (Eq. `eq:impact` for propagation, Eq. `eq:mystery` for mystery, Eq. `eq:surprise` for surprise).
