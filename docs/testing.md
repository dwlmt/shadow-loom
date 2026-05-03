# Testing

Shadow-Loom ships with a large pytest suite spread across **19 files** in
[`tests/`](../tests/). The suite is organised by pipeline phase: every major
module in [`shadow_loom/`](../shadow_loom/) has a focused unit-test file, and
several integration files exercise the engines together end-to-end. All LLM
calls are mocked by default; one optional file ([`test_live_e2e.py`](../tests/test_live_e2e.py))
runs against a real Ollama instance.

---

## Running the suite

```bash
# Fast path — everything except live LLM tests
python -m pytest tests/ --ignore=tests/test_live_e2e.py -q

# A single module
python -m pytest tests/test_amwn.py -v

# A single test
python -m pytest tests/test_narrative_physics.py::test_intervention_query_macbeth -v

# With coverage
python -m pytest tests/ --ignore=tests/test_live_e2e.py --cov=shadow_loom --cov=shadow_loom_mcp --cov=shadow_loom_ui

# The live tier (requires Ollama at localhost:11434 with qwen3.6:35b pulled).
# scripts/run_live_e2e.sh is a convenience wrapper that spawns it detached and
# logs to logs/live_e2e_<ts>.log so the suite can run in the background.
python -m pytest tests/test_live_e2e.py -v
bash scripts/run_live_e2e.sh
```

The mocked tiers are deterministic and run in roughly a minute on a developer
laptop. The live tier is slow (real model inference per query type) and is
skipped automatically when Ollama is unreachable.

---

## File index

Counts below are the number of `test_*` functions in each file.

### Core engine — unit tests

| File | Tests | What it covers | Module under test |
|---|---:|---|---|
| [test_ingestion.py](../tests/test_ingestion.py) | 114 | Chunking, validation, assembly, deduplication. No LLM calls — fully deterministic. | [`ingestion.py`](../shadow_loom/ingestion.py) |
| [test_amwn.py](../tests/test_amwn.py) | 20 | Ancestral Multi-World Network construction; ctf-calculus rules 1 (consistency), 2 (independence via d-separation), 3 (exclusion); pre-flight wiring inside `CausalPhysicsEngine.execute()`. Reference: Correa & Bareinboim, ICML 2025. | [`amwn.py`](../shadow_loom/amwn.py) |
| [test_causal_physics.py](../tests/test_causal_physics.py) | 62 | Three-rung CTF simulation: rung-2 do-operator + graph surgery, rung-3 abduction with `hidden_deltas` and topological-sort cascade, Impact > Inertia gating, bidirectional trait shifts, spatial-affordance blocking, intervened-node preservation. | [`causal_physics.py`](../shadow_loom/causal_physics.py) |
| [test_directive_assembly.py](../tests/test_directive_assembly.py) | 67 | Epistemic-gap computation, trait trajectories, relationship tensions, full `CreativeBrief` assembly for each directive effect. | [`directive_assembly.py`](../shadow_loom/directive_assembly.py) |
| [test_narrative_physics.py](../tests/test_narrative_physics.py) | 156 | All five core query types (observation, intervention, counterfactual, directive, interrogation) against real plot models; verifies returned graph structures reflect expected mutations. Largest unit-test file. | [`narrative_physics.py`](../shadow_loom/narrative_physics.py) |
| [test_branch_routing.py](../tests/test_branch_routing.py) | 10 | `_resolve_branch_policy`, `VersionedWorldModel.merge(world_id=…)` re-tagging, `db.list_branches` DAG walk, and `db.promote_branch` lifecycle. Covers the factual / shadow split that backs the AMWN persisted branches. | [`pipeline.py`](../shadow_loom/pipeline.py), [`extract_graph.py`](../shadow_loom/extract_graph.py), [`db.py`](../shadow_loom/db.py) |
| [test_channel_belief_integration.py](../tests/test_channel_belief_integration.py) | 22 | End-to-end coverage of the channel/utterance refactor: parser handling of `channel.*` / `utterance_event_ids` interventions, instantiator preservation of utterance attrs through sandboxing, causal-physics belief-provenance pruning, AMWN channel-as-node d-separation, generation-prompt fidelity block, auditor leak detection, and intelligibility-weighted hidden channels in directive assembly. | cross-cutting |
| [test_query_parsing.py](../tests/test_query_parsing.py) | 115 | Natural-language → typed query, dynamic Literal ID grounding, parser fallbacks. | [`query_parsing.py`](../shadow_loom/query_parsing.py) |
| [test_generation.py](../tests/test_generation.py) | 22 | Helper functions in the constrained renderer — prompt assembly, retry logic, output validation. | [`generation.py`](../shadow_loom/generation.py) |
| [test_auditor.py](../tests/test_auditor.py) | 79 | Audit prompt assembly; graph versioning via deep-copy isolation; `AuditResult` / `AuditViolation` construction; mocked feedback-loop orchestration; effect → audit-category mapping. | [`auditor.py`](../shadow_loom/auditor.py) |
| [test_global_traits.py](../tests/test_global_traits.py) | 50 | World-level traits: `GlobalTrait` / `WorldTraitSnapshot` models, `reconstruct_world_trait_at()`, `WORLD_` prefix validation, sandbox `WORLD_` node creation, ambient-edge propagation, abduction skipping `WORLD_` nodes. | [`models.py`](../shadow_loom/models.py), [`instantiator.py`](../shadow_loom/instantiator.py) |
| [test_version_mutations.py](../tests/test_version_mutations.py) | 18 | `delete_version` and `reparent_version` DB functions; in-memory SQLite. | [`db.py`](../shadow_loom/db.py) |
| [test_affective_curve_evolution.py](../tests/test_affective_curve_evolution.py) | 60 | Per-axis `mutation_social` coverage invariant: every observed `affinity` / `fear` / `power_dynamic` axis on every relationship in every bundled fixture must be touched by at least one matching `mutation_social` `CausalEdge`. Prevents flat danger / conflict / power gauges. See [design-decisions.md §D5d](design-decisions.md#d5d-per-axis-mutation_social-coverage-across-all-16-fixtures-may-2026-rebuild). | cross-cutting |

### Pipeline — integration tests

| File | Tests | What it covers |
|---|---:|---|
| [test_pipeline.py](../tests/test_pipeline.py) | 38 | The orchestrator (`shadow_loom.pipeline`) with mocked LLM. All computational code (physics engines, affective calculus, graph versioning, merge) runs un-mocked. |
| [test_pipeline_integration.py](../tests/test_pipeline_integration.py) | 34 | Full Steps 5–8: causal physics + affective calculus + directive assembly working as a coherent system; epistemic / fabula / syuzhet tracking; affective scoring of target emotions. |
| [test_end_to_end.py](../tests/test_end_to_end.py) | 52 | Plot models → graph extraction → physics/affective engines → generation → audit loop → prose re-extraction → merge. LLM mocked; everything else real. |
| [test_live_e2e.py](../tests/test_live_e2e.py) | 29 | **Optional live tier.** Real Ollama calls (`qwen3.6:35b`) against `example_worlds` fixtures, exercising every query type through the complete pipeline. Skipped automatically when Ollama is unreachable. |

### Surface tests — MCP & UI

| File | Tests | What it covers |
|---|---:|---|
| [test_mcp_server.py](../tests/test_mcp_server.py) | 63 | FastMCP server tool routing and resource serving against an in-memory SQLite DB seeded with the Macbeth fixture. LLM-calling paths mocked; computational paths run un-mocked. Auth tested in open mode (no bearer token). |
| [test_reasoning_helpers.py](../tests/test_reasoning_helpers.py) | 33 | `shadow_loom_ui.reasoning_helpers` — formatting and shaping of pipeline output for the UI. |
| [test_viz_helpers.py](../tests/test_viz_helpers.py) | 26 | `shadow_loom_ui.viz_helpers` — Sankey DAG safety, fabula snapshot construction. |

---

## How tests are scoped

A few conventions to be aware of when reading or adding tests:

- **LLM is mocked at the seam.** Unit and integration tests patch the model
  resolver / agent calls (`unittest.mock.patch` against `shadow_loom.generation`,
  `shadow_loom.auditor`, `shadow_loom.ingestion.run_extraction_pass`, etc.) and
  feed back deterministic Pydantic objects. Everything past the mock — graph
  surgery, AMWN, ctf-calculus, affective calculus, version merge — runs for
  real.
- **Fixtures come from [`example_worlds/`](../example_worlds/).** Each scripted
  world (Macbeth, Persuasion, *Reservoir Dogs*, …) returns a hand-built
  `WorldStateV1` so tests can assert against known graph topology rather than
  LLM-extracted topology.
- **DB tests use SQLite in-memory.** Set `DATABASE_URL=sqlite://` (the
  `test_version_mutations.py` and `test_mcp_server.py` files do this at import
  time before importing the modules under test).
- **No test mutates a canonical version.** Tests that exercise the audit /
  re-extract / merge pipeline create child versions and assert ancestor
  linkage; this matches the production invariant that nothing canonical is
  mutated until audit passes.
- **Live tests are gated.** [`test_live_e2e.py`](../tests/test_live_e2e.py)
  calls `pytest.importorskip` / connectivity checks against
  `localhost:11434` and exits cleanly when Ollama is missing, so CI doesn't
  fail on developer machines without a model server.

---

## Adding tests

When you add a feature to a module in [`shadow_loom/`](../shadow_loom/):

1. **Add unit tests to the matching `tests/test_<module>.py`** — mock at the
   LLM seam; let physics, graph surgery, and version operations run real.
2. **If the feature crosses engine boundaries** (e.g. a new directive effect
   that affects both the affective calculus and the auditor), add an
   integration test to [test_pipeline_integration.py](../tests/test_pipeline_integration.py)
   or [test_end_to_end.py](../tests/test_end_to_end.py).
3. **If the feature is exposed over MCP**, add a tool-routing test to
   [test_mcp_server.py](../tests/test_mcp_server.py).
4. **If you add a new scripted world to [`example_worlds/`](../example_worlds/)**,
   the existing query-parsing and narrative-physics tests will pick it up via
   their parametrised world list — no extra wiring needed.

---

## See also

- [pipeline-walkthrough.md](pipeline-walkthrough.md) — the code paths that
  these tests exercise.
- [architecture.md](architecture.md) — what each engine is responsible for and
  therefore what its test file should cover.
- [mcp-guide.md](mcp-guide.md) — the tool/resource surface tested by
  `test_mcp_server.py`.
