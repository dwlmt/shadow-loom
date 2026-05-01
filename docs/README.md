# Shadow-Loom Documentation

Shadow-Loom is a **neuro-symbolic causal narrative AI framework**. It treats a
story as a typed graph of entities, events and edges with explicit physics
(causality, beliefs, information flow, spatial topology) and uses a Large
Language Model only as a constrained renderer once mathematical simulation has
fixed what is allowed to happen next.

It integrates classical narratology (fabula vs syuzhet, Greimas, Genette),
Pearl's ladder of causation (observation, intervention, counterfactual),
information theory (KL surprise, Wilmot suspense), and modern LLM orchestration
into a single end-to-end pipeline that ingests prose, simulates over it,
generates new prose under provable constraints, and audits its own output —
all inside a versioned world model.

The top-level [`../README.md`](../README.md) is a one-screen pointer; this
file is the canonical entry point.

---

## Documentation index

| Document | Purpose |
|---|---|
| [architecture.md](architecture.md) | Deep technical walkthrough of the 12-step pipeline, data model, modules, and runtime flow. |
| [pipeline-walkthrough.md](pipeline-walkthrough.md) | End-to-end code-level tour of one pipeline run — ingestion, physics, generation, audit, re-extraction, merge. |
| [model-examples.md](model-examples.md) | Worked examples on real bundled plots (Macbeth, Death on the Nile, Reservoir Dogs, …) showing each pipeline stage and feature in action. |
| [query-and-cycles.md](query-and-cycles.md) | The eight query types, how natural language is parsed into them, and how each is realised in a pipeline cycle. |
| [mcp-guide.md](mcp-guide.md) | The `shadow_loom_mcp` server — 31 tools, 5 resources, auth, scopes, versioning contract, agent workflow. |
| [ui-guide.md](ui-guide.md) | NiceGUI workspace walkthrough, including the manual-editing **Editor** tab. |
| [testing.md](testing.md) | Test-suite organisation, what each file covers, how to run the live-LLM tier. |
| [use-cases.md](use-cases.md) | What the system is *for* — author tooling, AI-assisted fiction, narrative QA, simulation research. |
| [design-decisions.md](design-decisions.md) | The key choices that shape the architecture and what we deliberately rejected. |
| [settings.md](settings.md) | Every runtime knob: env vars, defaults, tuning recipes, model-provider switching. |
| [railway-deployment.md](railway-deployment.md) | Step-by-step recipe for deploying to Railway with managed Postgres, OAuth, and OpenRouter. |
| [academic-foundations.md](academic-foundations.md) | The literature behind every named concept — Pearl, Genette, Greimas, Sternberg, Halpern, Wilmot, Correa & Bareinboim, etc. |

Quick links by intent:

| If you want to … | Read |
|---|---|
| Understand the data model and the 12-step pipeline at a high level | [architecture.md](architecture.md) |
| Follow one request end-to-end through the code (ingestion → merge) | [pipeline-walkthrough.md](pipeline-walkthrough.md) |
| See the engine working on real story plots, feature by feature | [model-examples.md](model-examples.md) |
| Learn the eight query types and how natural language becomes one | [query-and-cycles.md](query-and-cycles.md) |
| Drive Shadow-Loom from Claude Desktop, Cursor, or any MCP client | [mcp-guide.md](mcp-guide.md) |
| Use the NiceGUI workspace (story / explorer / world / causality / …) | [ui-guide.md](ui-guide.md) |
| Run the test suite or extend it | [testing.md](testing.md) |
| See what Shadow-Loom is *for* | [use-cases.md](use-cases.md) |
| Understand *why* the architecture is the way it is | [design-decisions.md](design-decisions.md) |
| Tune model defaults, swap providers, or tweak physics constants | [settings.md](settings.md) |
| Deploy to production (Railway / Postgres / OAuth / OpenRouter) | [railway-deployment.md](railway-deployment.md) |
| Trace every named concept back to its literature | [academic-foundations.md](academic-foundations.md) |

---

## Reading paths

Pick the path that matches what you're trying to do.

**"I just want to understand the system."**
[architecture.md](architecture.md) → [academic-foundations.md](academic-foundations.md) → [design-decisions.md](design-decisions.md).

**"I want to drive it from an agent / build a client."**
[mcp-guide.md](mcp-guide.md) → [query-and-cycles.md](query-and-cycles.md) → [pipeline-walkthrough.md](pipeline-walkthrough.md).

**"I want to use the workspace as a human author."**
[use-cases.md](use-cases.md) → [ui-guide.md](ui-guide.md) → [query-and-cycles.md](query-and-cycles.md).

**"I want to extend the engine or contribute code."**
[architecture.md](architecture.md) → [pipeline-walkthrough.md](pipeline-walkthrough.md) → [testing.md](testing.md) → [design-decisions.md](design-decisions.md) → the module the change touches.

**"I'm writing a paper or comparing to prior work."**
[academic-foundations.md](academic-foundations.md) → [design-decisions.md](design-decisions.md) → [architecture.md](architecture.md).

If none of these fit, the safe default is [architecture.md](architecture.md) —
it links into every other document. Each doc also has a **See also** footer
that cross-links its closest neighbours.

---

## What it does, in one paragraph

You hand Shadow-Loom raw narrative text. It runs a five-pass extraction to
build a typed `WorldStateV1` — entities with `TraitVector`s, events anchored
on both `fabula_time` and `syuzhet_index`, beliefs (with explicit
`acquired_via_event_id` / `acquired_via_channel_id` provenance), locations
with ambient state, three kinds of edge (causal, social, spatial), and
`Channel` nodes plus `utterance` events that carry the speech-act layer
(replacing the legacy `InformationEdge`; see
[`scripts/migrate_information_edges.py`](../scripts/migrate_information_edges.py)).
Versions are persisted on a DAG with a `world_id` / `branch_label` so that
counterfactual explorations fork to a *shadow* branch and can be promoted
to factual canon. You
then issue a query: an observation, a do-calculus intervention, an abductive
counterfactual, an affective directive ("maximise dramatic irony for character
X"), or a graph Q&A. Shadow-Loom forks an Ancestral Multi-World Network
sandbox, runs Pearl's ladder over it, scores survivors with mathematical
mystery / irony / suspense / surprise / emotion functions, packages the
winner as a `CreativeBrief`, hands it to a constrained-LLM renderer, and runs
a recursive auditor against the prose to catch "Miracle Steps" and broken
beliefs. The new prose is re-extracted into topology and merged into a new
version of the world model. Every step is logged, every version is
ancestor-linked, and nothing canonical is mutated until the audit passes.

For the full technical breakdown of each phase, see
[architecture.md](architecture.md). For the per-query-type cycle, see
[query-and-cycles.md](query-and-cycles.md).

---

## TL;DR architecture

```
       (narrative text)
              │
              ▼
   ┌──────────────────────┐         Phase 1: Initialisation
   │  Ingestion (LLM)     │  ─────  Steps 1–5
   │  → WorldStateV1      │         (ontology, fabula, syuzhet,
   └──────────┬───────────┘          beliefs, ego-graph)
              │
              ▼
   ┌──────────────────────┐         Phase 2: Simulation
   │  AMWN sandbox        │  ─────  Steps 6–8
   │  Causal Physics (CTF)│         (Pearl rungs 2 & 3,
   │  Affective Calculus  │          Wilmot suspense, KL surprise)
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐         Phase 3: Constraint
   │  Directive Assembly  │  ─────  Step 9
   │  → CreativeBrief     │         (envelope of possibilities)
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐         Phase 4: Generation
   │  Constrained LLM     │  ─────  Step 10
   │  prose render        │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐         Phase 5: Audit
   │  LLM-as-judge        │  ─────  Steps 11–12
   │  + refinement loop   │         (causal / abductive / affective)
   └──────────┬───────────┘
              │
              ▼
       (verified prose +
        committed world state)
```

---

## The five phases (and where they live)

| Phase | Steps | Module | Doc |
|---|---|---|---|
| 1. World State & Initialisation | 1–5 | [`shadow_loom/ingestion.py`](../shadow_loom/ingestion.py), [`models.py`](../shadow_loom/models.py), [`extract_graph.py`](../shadow_loom/extract_graph.py) | [architecture §2](architecture.md), [pipeline §1](pipeline-walkthrough.md) |
| 2. Mathematical Simulation | 6–8 | [`instantiator.py`](../shadow_loom/instantiator.py), [`causal_physics.py`](../shadow_loom/causal_physics.py) | [architecture §3](architecture.md) |
| 3. Generative Constraint | 9 | [`directive_assembly.py`](../shadow_loom/directive_assembly.py) | [architecture §4](architecture.md) |
| 4. Prose Generation | 10 | [`generation.py`](../shadow_loom/generation.py) | [architecture §5](architecture.md) |
| 5. Nested Learning Audit | 11–12 | [`auditor.py`](../shadow_loom/auditor.py), [`pipeline.py`](../shadow_loom/pipeline.py) | [architecture §6](architecture.md), [pipeline §5–7](pipeline-walkthrough.md) |

The query router that ties them all together is
[`shadow_loom/narrative_physics.py`](../shadow_loom/narrative_physics.py); the
end-to-end orchestrator is
[`shadow_loom/pipeline.py::run_pipeline`](../shadow_loom/pipeline.py).

---

## Quick start

```bash
# Environment (the repo is packaged via pyproject.toml; no environment.yml)
conda create -n shadow-loom python=3.13 -y
conda activate shadow-loom
pip install -e .

# Tests (live LLM e2e excluded by default)
python -m pytest tests/ --ignore=tests/test_live_e2e.py -q

# UI
python -m shadow_loom_ui                 # NiceGUI workspace on http://localhost:7860

# MCP server (FastMCP, stdio)
python -m shadow_loom_mcp
```

See [ui-guide.md](ui-guide.md) for the workspace tour, [mcp-guide.md](mcp-guide.md)
for the MCP tool catalogue and the Claude Desktop / Cursor configuration
snippet, and [testing.md](testing.md) for the test-suite layout.

---

## Project layout

```
shadow_loom/                # core engine
  models.py                 # WorldStateV1 schema (Pydantic v2)
  ingestion.py              # 5-pass prose → WorldStateV1 extraction
  extract_graph.py          # ego-graph slicing + VersionedWorldModel
  instantiator.py           # AMWN sandbox (NetworkX MultiDiGraph)
  amwn.py                   # ctf-calculus pre-flight (Correa & Bareinboim 2025)
  causal_physics.py         # 3-rung CTF engine (Pearl's ladder)
  directive_assembly.py     # affective calculus + CreativeBrief
  narrative_physics.py      # query router (8 query types)
  generation.py             # constrained LLM renderer
  auditor.py                # recursive narrative auditor
  pipeline.py               # end-to-end orchestrator
  query_models.py           # the 8 typed query schemas
  query_parsing.py          # NL → typed query, with ID grounding
  db.py                     # SQLModel persistence + version tree
  settings.py               # config loader

shadow_loom_mcp/            # FastMCP server (31 tools, 5 resources)
shadow_loom_ui/             # NiceGUI workspace (8 tabs)
example_worlds/             # 16 scripted worlds for tests + demos
sample_plots/               # raw plot summaries for ingestion demos
tests/                      # pytest suite — see testing.md
docs/                       # this folder
```

---

## Status & licence

Research project; APIs are stable enough to use but evolve between minor
versions. See [design-decisions.md](design-decisions.md) for the choices that
shape the public surface and what we deliberately rejected.
