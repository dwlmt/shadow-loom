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
| `Location` | `LOC_` | Spatial container with `ambient_state: Dict[str, AmbientVector]`. |
| `NarrativeObject` | `OBJ_` | Inanimate item with `affordances: List[Affordance]`. |
| `Entity` | `ENT_` | Character / agent with `traits`, `beliefs`, `status`, `state_timeline`. |
| `EventNode` | `EVT_` | Atomic happening anchored on both `fabula_time` and `syuzhet_index`. |
| `GlobalTrait` | `WORLD_` | World-level fact / law / regime ("magic system", "surveillance state"). |

### Edges

| Class | Topology | Notes |
|---|---|---|
| `CausalEdge` | event⇄event / event→state / state→event / state→state | Single class with five `causality_type` modalities; validator enforces source/target type matches modality. |
| `RelationshipEdge` | entity⇄entity | Continuous `affinity`, `fear`, `power_dynamic`, `inertia`. |
| `SpatialEdge` | location→location | Optional `is_locked` + `barrier_item_id`. |
| `InformationEdge` | (entity\|object)→entities | Communication channel with `medium`, `is_encrypted`, `discovered_at_syuzhet`. |

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

* `PhysicsExtraction` — `EventNode`s, `CausalEdge`s, `SpatialEdge`s, fallback `EntityUpdate`s
* `SocialExtraction` — `RelationshipEdge`s, `InformationEdge`s
* `ConsequencesExtraction` — authoritative `EntityUpdate`s anchored to the Physics events + mutation edges (default-on; overrides the Physics agent's own `entity_updates`). Toggle via `ExtractionConfig.enable_consequences_agent`.

In async mode Physics runs first; Social and Consequences are dispatched
concurrently via `asyncio.gather` since both depend only on the Physics
output. Each agent's output validator runs a **sanitiser layer** that
clamps numeric ranges, drops self-loops, coerces status enum aliases, and
fuzzy-fixes ID typos before falling back to a `ModelRetry`. Results are
merged in `assemble_world_state`, normalised (`_normalize_fabula_times`),
auto-repaired (`_auto_repair`), and validated (`_programmatic_validation`
→ optional LLM correction loop).

### Step 3: Epistemic synchronisation

`Belief.established_at_fabula` and `InformationEdge.discovered_at_syuzhet`
record *when* a fact entered each character's awareness vs the reader's. The
ingestion prompts (`prompts/social_extraction.md`) explicitly request both
axes for any communication event.

### Step 4: Director intent

`shadow_loom/query_models.py` defines five query types — the user picks one to
drive a generation cycle:

| Query | Pearl rung | Purpose |
|---|---|---|
| `ObservationQuery` | 1 | "What happens next given the world as is?" |
| `InterventionQuery` | 2 | "What if X did Y?" — applies `do(X=y)`. |
| `CounterfactualQuery` | 3 | "What if the past had been different?" — abduction + intervention. |
| `DirectiveQuery` | — | High-level emotional target (`mystery`, `suspense`, `surprise`, `dramatic_irony`, plus emotion targets). |
| `InterrogationQuery` | — | Graph RAG over the world model for Q&A. |

### Step 5: Ego-graph extraction ([`extract_graph.py`](../shadow_loom/extract_graph.py))

`extract_ego_graph_from_memory()` returns an `EgoGraphPayload` containing only
the slice relevant to the focal characters and time anchor:

* 1-hop spatial neighbours of focus locations
* co-located entities and objects
* causal edges within `memory_limit` recent events
* `RelationshipEdge`s and `Belief`s **time-sliced via
  `reconstruct_entity_at(temporal_anchor)`** so beliefs formed in the future
  do not leak in
* relevant `InformationEdge`s and `WORLD_*` traits

This both saves LLM context window and prevents temporal contamination of
counterfactuals.

---

## 3. Phase 2 — Mathematical Simulation (Steps 6–8)

### Step 6: AMWN sandbox ([`instantiator.py`](../shadow_loom/instantiator.py))

`AMWNInstantiator.create_sandbox()` builds a NetworkX `MultiDiGraph` mirror of
the ego-graph. Mutations are confined to the sandbox until they are explicitly
committed back. Edge types laid down:

* `located_in`, `owned_by` — spatial / inventory topology
* `relationship` — psycho-social metrics (affinity / fear / power_dynamic / inertia)
* `causal` — `mechanism`, `evidence_strength`, `causal_force`,
  `propagation_delay`
* `connected_to` — spatial, with `is_locked` and `barrier_item_id`
* `communicating_with` — informational, with `medium` and `is_encrypted`
* `eavesdropped_by` — auto-derived for unencrypted channels (epistemic leakage)

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
| **Mystery** | $\dfrac{\#\text{hidden ancestors}}{\#\text{total ancestors}}$ for each known effect; walks back through `causal_topology` and filters by `discovered_at_syuzhet > syuzhet_anchor`. |
| **Dramatic Irony** | $\dfrac{\#\text{irony gaps}}{\#\text{total connections}}$; an irony gap is a revealed causal edge where the source event is not in the focal entity's belief set at `temporal_anchor`. |
| **Suspense** | $P(\text{threat}) - P(\text{hope})$; threat = unrevealed events targeting the entity, hope = unrevealed events authored by the entity. `evidence_strength` is the probability proxy. Returns 0 when hope is extinguished (despair, not suspense). Inspired directly by Wilmot & Keller (2020); see [academic-foundations.md §3.1](academic-foundations.md#31-suspense-as-uncertainty-reduction--wilmot--keller-acl-2020). |
| **Surprise** | Per-trait binary KL divergence $D_\text{KL}(p \| q) = p\log\tfrac{p}{q} + (1-p)\log\tfrac{1-p}{1-q}$. Prior $q$ starts at maximum entropy 0.5 and is updated toward truth for each revealed causal edge; posterior $p$ is the actual trait. See [academic-foundations.md §3.3](academic-foundations.md#33-surprise-as-kl-divergence). |

Six emotional effects (`grief`, `rage`, `joy`, `regret`, `love`, `fear`) use
trait-trajectory headroom analysis: each effect declares which traits should
move in which direction; the score is the available headroom weighted by
inertia evidence.

`compute_affective_score()` returns the weighted combination requested by the
`DirectiveQuery`.

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
| [`narrative_physics.py`](../shadow_loom/narrative_physics.py) | 1 216 | Pipeline orchestrator routing the five query types. |
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
  created (`ingestion`, `pipeline`, `manual_edit`, etc.).
* `ActiveVersionRow(project_id, user_id) → version_row_id` — the per-user
  pointer that the UI and MCP read by default.

Version writes go through `save_version(...)`; pointer updates through
`set_active_version(...)`. Deletion is rejoin-aware (`delete_version`
re-parents children atomically).

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
  auto-advances the active pointer to the new version row.
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
