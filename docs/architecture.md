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
* `Entity.traits / beliefs / status / location_id` represent the **initial
  pre-story baseline**; the timeline replays deltas on top.
* `reconstruct_entity_at(entity, fabula_time)` and
  `reconstruct_world_trait_at(trait, fabula_time)` are pure functions that
  return the merged state at any chronological slice. They are used by the
  ego-graph extractor, the causal physics engine, and the narrative physics
  layer.

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

### Step 2: Topology extraction (per-chunk)

For each text chunk the pipeline runs a **Socratic-QA scaffold** first
(Who/What/Where/When/Why/How pairs that articulate hidden motivations and
abductive inferences before structured extraction — see
[academic-foundations.md §6.5](academic-foundations.md#65-computational-narratology-and-story-understanding)
for the lineage), then dispatches three specialist agents:

* `PhysicsExtraction` — `EventNode`s (including `utterance` events), `CausalEdge`s, `SpatialEdge`s, fallback `EntityUpdate`s
* `SocialExtraction` — `RelationshipEdge`s, `Channel`s
* `ConsequencesExtraction` — authoritative `EntityUpdate`s anchored to the Physics events + mutation edges (default-on; overrides the Physics agent's own `entity_updates`). Toggle via `ExtractionConfig.enable_consequences_agent`.

In async mode Physics runs first; Social and Consequences are dispatched
concurrently via `asyncio.gather` since both depend only on the Physics
output. Each agent's output validator runs a **sanitiser layer** that
clamps numeric ranges, drops self-loops, coerces status enum aliases, and
fuzzy-fixes ID typos before falling back to a `ModelRetry`. Results are
merged in `assemble_world_state`, normalised (`_normalize_fabula_times`),
auto-repaired (`_auto_repair`), and validated (`_programmatic_validation`
→ optional LLM correction loop). Chunk order is treated as syuzhet order
only; each chunk's events keep their LLM-extracted `fabula_time`, so
flashbacks and flashforwards are preserved across chunk boundaries instead
of being re-sorted into reading order.

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
[docs/research-extraction-plan.md](research-extraction-plan.md) and
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
| `InterrogationQuery` | — | Graph RAG over the world model for Q&A. |
| `GeneralQuery` | — | Free-form NL Q&A over the graph (ungrounded discussion). |
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

This both saves LLM context window and prevents temporal contamination of
counterfactuals.

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
* `communicating_with` — informational, derived from `Channel` participants and per-participant `intelligibility`. `Channel` carries `medium`, `directionality`, `participant_ids`, an `intelligibility` map, `established_at_fabula`, `terminated_at_fabula` (None while still open), `discovered_at_syuzhet` (for hidden-channel detection) and `evidence_strength`.
* `eavesdropped_by` — auto-derived for participants whose `intelligibility >= physics.intelligibility_threshold` but who are *not* in an utterance's `addressee_ids` (epistemic leakage)

### Step 7: Causal Physics ([`causal_physics.py`](../shadow_loom/causal_physics.py))

`CausalPhysicsEngine` implements all three rungs of Pearl's Ladder of
Causation:

* **Abduction (rung 3).** Back-propagates present-day evidence into the
  historical sandbox. Entity traits blend 50 % toward observed factual
  values; beliefs propagate backward; causal edges weighted by
  `evidence_strength`. `MECHANISM_TRAIT_MAP` gates which mechanisms can touch
  which trait families.
* **Action (rung 2).** Applies `do(X=x)` via six surgery types:
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
| **Dramatic Irony** | $\dfrac{\#\text{irony gaps}}{\#\text{total connections}}$; an irony gap is a revealed causal edge where the source event is not in the focal entity's belief set at `temporal_anchor`. |
| **Suspense** | $\text{balance} \times \text{stakes}$ clamped to $[0, 1]$, where $\text{balance} = 1 - \dfrac{|w_\text{threat} - w_\text{hope}|}{w_\text{threat} + w_\text{hope}}$ peaks at genuine outcome uncertainty and decays under one-sided dominance, and $\text{stakes} = \dfrac{w_\text{threat} + w_\text{hope}}{w_\text{threat} + w_\text{hope} + K}$ saturates so balanced fragments don't pin the gauge ($K = 2$ by default). For each focal entity, every unrevealed event in which the entity is a non-acting target contributes its `evidence_strength`-derived probability $p$ to $w_\text{threat}$, and every unrevealed event in which the entity is an actor contributes $p$ to $w_\text{hope}$. The probability proxy is the strongest incoming causal-edge weight on the event (outgoing as fallback, 0.5 default). Returns 0 at the **despair** boundary ($w_\text{hope} = 0$) and the **safety** boundary ($w_\text{threat} = 0$). The earlier asymmetric $\max(0, (w_\text{threat} - w_\text{hope})/(w_\text{threat}+w_\text{hope}))$ form collapsed to 0 on every fixture in which the protagonist authors most of their own forward events; balance × stakes follows Brewer & Lichtenstein's structural-affect framing of suspense as a response to outcome ambiguity. Inspired by Wilmot & Keller (2020); see [academic-foundations.md §3.1](academic-foundations.md#31-suspense-as-uncertainty-reduction--wilmot--keller-acl-2020). |
| **Surprise** | Per-trait binary KL divergence $D_\text{KL}(p \| q) = p\log\tfrac{p}{q} + (1-p)\log\tfrac{1-p}{1-q}$. Posterior $p$ is the entity's *final-state* trait value resolved via `reconstruct_entity_at(ent, t_max)` so authored `state_timeline` arcs are honoured (sandbox-preferred when running counterfactuals). Prior $q$ starts at the **leave-one-out** per-trait corpus marginal (mean across every *other* entity, falling back to 0.5 when fewer than two other entities carry the trait) — leave-one-out prevents the focal entity from biasing its own prior, which would otherwise collapse KL on the small casts typical of the example fixtures. The prior is then pulled toward the actual value by a geometric update $q \mathrel{+}= w \cdot (\text{actual} - q)$ for each revealed causal edge whose target is the entity, monotonically converging on the truth as evidence accumulates rather than overshooting. The result is the average per-trait KL across the focal entities, normalised by $\log(1/\varepsilon)$ to land in $[0, 1]$. See [academic-foundations.md §3.3](academic-foundations.md#33-surprise-as-kl-divergence). |

Six emotional effects (`grief`, `rage`, `joy`, `regret`, `love`, `fear`) use
trait-trajectory headroom analysis: each effect declares which traits should
move in which direction; the score is the available headroom weighted by
inertia evidence.

`compute_affective_score()` returns the weighted combination requested by the
`DirectiveQuery`.

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

#### Monotonicity along the two time axes

The four scorers are sampled along two independent axes by
[`viz_helpers.py`](../shadow_loom_ui/viz_helpers.py):

* **Syuzhet axis** — the reader's progress through the *told* order.
  Advancing the syuzhet anchor only ever reveals more events, so the
  prior in `surprise` can only move *toward* the truth (never past it,
  thanks to the geometric pull). Surprise is therefore monotonically
  non-increasing on this axis. Mystery and dramatic irony likewise
  fall as more sources come into view; suspense falls as unrevealed
  threats / hopes are consumed.
* **Fabula axis** — the *story* order, sampled via
  `snapshot_world_at(t)`. Here the posterior trait values themselves
  change with the snapshot, so surprise *can* spike — that is the
  intended behaviour. The fabula curve answers *which moments in the
  story are intrinsically surprising* (Macbeth's ambition flipping
  after the prophecy, Jacqueline's cruelty post-murder), whereas the
  syuzhet curve answers *how much catching-up the reader still has to
  do*.

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

---

## 7. Modules at a glance

| Module | Lines | Role |
|---|---|---|
| [`models.py`](../shadow_loom/models.py) | 440 | `WorldStateV1` schema + temporal reconstruction. |
| [`ingestion.py`](../shadow_loom/ingestion.py) | 2 978 | LLM-driven world extraction, normalisation, programmatic validation, correction loop. |
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

NiceGUI workspace ([`app.py`](../shadow_loom_ui/app.py)) composed of tabs in
[`components/workspace.py`](../shadow_loom_ui/components/workspace.py):

```
story · explorer · world · causality · reasoning · audit · editor · export
```

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
* `get_entity`, `get_event`, `get_world_trait`, etc. — typed read tools.
* `helpers.load_world_state(pid, version, *, ctx=None)` resolves the active
  version per-user when `ctx` is provided; resources skip auth.

---

## 11. Tests

`tests/` contains 950+ pytest tests covering models, ingestion, AMWN,
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
* [academic-foundations.md](academic-foundations.md) — citations for fabula/syuzhet ([§1.1](academic-foundations.md#11-fabula-vs-syuzhet-fabula_time--syuzhet_index)), Pearl's ladder ([§2.1](academic-foundations.md#21-three-rungs-of-causation-observationquery-interventionquery-counterfactualquery)), Wilmot suspense ([§3.1](academic-foundations.md#31-suspense-as-uncertainty-reduction--wilmot--keller-acl-2020)), KL surprise ([§3.3](academic-foundations.md#33-surprise-as-kl-divergence)), AMWN, ctf-calculus ([§2.2](academic-foundations.md#22-ancestral-multi-world-networks-and-ctf-calculus--correa--bareinboim-icml-2025)).
* [settings.md](settings.md) — every runtime knob (model strings, token budgets, physics constants, audit thresholds) and how to override them.
