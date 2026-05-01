# Shadow Loom

A **neuro-symbolic causal narrative AI framework**. Shadow-Loom treats a story
as a typed graph of entities, events and edges with explicit physics
(causality, beliefs, information flow, spatial topology) and uses a Large
Language Model only as a constrained renderer once mathematical simulation has
fixed what is allowed to happen next.

It integrates classical narratology (fabula vs syuzhet, Greimas, Genette),
Pearl's ladder of causation (observation, intervention, counterfactual),
information theory (KL surprise, suspense), and modern LLM orchestration
into a single end-to-end pipeline that ingests prose, simulates over it,
generates new prose under provable constraints, and audits its own output —
all inside a versioned world model.

```
prose ──► graph ──► AMWN sandbox ──► causal physics ──► creative brief
                                                              │
                                                              ▼
        versioned world model ◄── re-extract ◄── audit ◄── LLM render
```

---

## Documentation

The long-form technical and conceptual reference lives in [`docs/`](docs/).
Every doc has a **See also** footer cross-linking its closest neighbours.

| Document | Purpose |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Deep technical walkthrough of the 12-step pipeline, data model, modules, and runtime flow. |
| [docs/pipeline-walkthrough.md](docs/pipeline-walkthrough.md) | End-to-end code-level tour of one pipeline run — ingestion, physics, generation, audit, re-extraction, merge. |
| [docs/pipeline-by-example.md](docs/pipeline-by-example.md) | Data-anchored walkthrough — every stage of the pipeline (world model, AMWN, Pearl rungs 1–3, propagation, suspense / surprise / mystery / dramatic-irony / emotion scoring, directive assembly, generation, audit, merge) illustrated with verbatim values from the bundled fixtures. |
| [docs/model-examples.md](docs/model-examples.md) | Worked examples on real bundled plots (Macbeth, Death on the Nile, Reservoir Dogs, …) showing each pipeline stage and feature in action. |
| [docs/query-and-cycles.md](docs/query-and-cycles.md) | The eight query types, how natural language is parsed into them, and how each is realised in a pipeline cycle. |
| [docs/mcp-guide.md](docs/mcp-guide.md) | The `shadow_loom_mcp` server — 31 tools, 5 resources, auth, scopes, versioning contract, agent workflow. |
| [docs/ui-guide.md](docs/ui-guide.md) | NiceGUI workspace walkthrough, including the manual-editing **Editor** tab. |
| [docs/testing.md](docs/testing.md) | Test-suite organisation, what each file covers, how to run the live-LLM tier. |
| [docs/use-cases.md](docs/use-cases.md) | What the system is *for* — author tooling, AI-assisted fiction, narrative QA, simulation research. |
| [docs/design-decisions.md](docs/design-decisions.md) | The key choices that shape the architecture and what we deliberately rejected. |
| [docs/settings.md](docs/settings.md) | Every runtime knob: env vars, defaults, tuning recipes, model-provider switching. |
| [docs/railway-deployment.md](docs/railway-deployment.md) | Step-by-step recipe for deploying to Railway with managed Postgres, OAuth, and OpenRouter. |
| [docs/render-deployment.md](docs/render-deployment.md) | Equivalent Render Blueprint deploy — single-click via [`render.yaml`](render.yaml). |
| [docs/academic-foundations.md](docs/academic-foundations.md) | The literature behind every named concept — Pearl, Genette, Greimas, Sternberg, Halpern, Wilmot, Correa & Bareinboim, etc. |

### Reading paths

Pick the path that matches what you're trying to do.

**"I just want to understand the system."**
[architecture.md](docs/architecture.md) → [academic-foundations.md](docs/academic-foundations.md) → [design-decisions.md](docs/design-decisions.md).

**"I want to drive it from an agent / build a client."**
[mcp-guide.md](docs/mcp-guide.md) → [query-and-cycles.md](docs/query-and-cycles.md) → [pipeline-walkthrough.md](docs/pipeline-walkthrough.md).

**"I want to use the workspace as a human author."**
[use-cases.md](docs/use-cases.md) → [ui-guide.md](docs/ui-guide.md) → [query-and-cycles.md](docs/query-and-cycles.md).

**"I want to extend the engine or contribute code."**
[architecture.md](docs/architecture.md) → [pipeline-walkthrough.md](docs/pipeline-walkthrough.md) → [testing.md](docs/testing.md) → [design-decisions.md](docs/design-decisions.md) → the module the change touches.

**"I'm writing a paper or comparing to prior work."**
[academic-foundations.md](docs/academic-foundations.md) → [design-decisions.md](docs/design-decisions.md) → [architecture.md](docs/architecture.md).

**"I want to see the engine working on real story plots, feature by feature."**
[model-examples.md](docs/model-examples.md) → [pipeline-by-example.md](docs/pipeline-by-example.md).

**"I want to deploy / tune / configure."**
[settings.md](docs/settings.md) → [railway-deployment.md](docs/railway-deployment.md) or [render-deployment.md](docs/render-deployment.md).

If none of these fit, the safe default is [architecture.md](docs/architecture.md) —
it links into every other document.

---

## What it does, in one paragraph

You hand Shadow-Loom raw narrative text. It runs a five-pass extraction to
build a typed `WorldStateV1` — entities with `TraitVector`s, events anchored
on both `fabula_time` and `syuzhet_index`, beliefs, locations with ambient
state, and three kinds of edge (causal, social, spatial), plus `Channel`
nodes and `utterance` events that carry the speech-act / information layer
(replacing the legacy `InformationEdge`). You
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
[docs/architecture.md](docs/architecture.md). For the per-query-type cycle,
see [docs/query-and-cycles.md](docs/query-and-cycles.md).

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
| 1. World State & Initialisation | 1–5 | [`shadow_loom/ingestion.py`](shadow_loom/ingestion.py), [`models.py`](shadow_loom/models.py), [`extract_graph.py`](shadow_loom/extract_graph.py) | [architecture §2](docs/architecture.md), [pipeline §1](docs/pipeline-walkthrough.md) |
| 2. Mathematical Simulation | 6–8 | [`instantiator.py`](shadow_loom/instantiator.py), [`causal_physics.py`](shadow_loom/causal_physics.py) | [architecture §3](docs/architecture.md) |
| 3. Generative Constraint | 9 | [`directive_assembly.py`](shadow_loom/directive_assembly.py) | [architecture §4](docs/architecture.md) |
| 4. Prose Generation | 10 | [`generation.py`](shadow_loom/generation.py) | [architecture §5](docs/architecture.md) |
| 5. Nested Learning Audit | 11–12 | [`auditor.py`](shadow_loom/auditor.py), [`pipeline.py`](shadow_loom/pipeline.py) | [architecture §6](docs/architecture.md), [pipeline §5–7](docs/pipeline-walkthrough.md) |

The query router that ties them all together is
[`shadow_loom/narrative_physics.py`](shadow_loom/narrative_physics.py); the
end-to-end orchestrator is
[`shadow_loom/pipeline.py::run_pipeline`](shadow_loom/pipeline.py).

---

## Quick start

### Prerequisites

* **Python 3.13+** (3.14 recommended; see [.python-version](.python-version)).
* **An LLM backend.** One of:
  * **[Ollama](https://ollama.com/download)** for local inference (default).
    After install: `ollama serve` then `ollama pull qwen3.6:27b`.
    Smaller models work too — set `DEFAULT_MODEL` in `.env`.
  * **OpenRouter** — set `OPENROUTER_API_KEY` and a `*_MODEL` value
    starting with `openrouter:` in `.env`.
  * **OpenAI** — set `OPENAI_API_KEY` and use `openai:gpt-…` model strings.
* **Optional:** Docker / Docker Compose for the containerised stack;
  Postgres if you don't want SQLite.

### Option A — one-shot bootstrap (recommended)

```bash
git clone https://github.com/dwlmt/shadow-loom.git
cd shadow-loom
make setup            # creates .venv, installs in editable mode, copies .env, checks Ollama
source .venv/bin/activate
make ui               # http://localhost:7860
```

`make setup` is idempotent — re-run any time. See `make help` for all
targets (`ui`, `mcp`, `pipeline`, `test`, `lint`, `docker-up`, …).

### Option B — manual

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env                # then edit if needed

ollama pull qwen3.6:27b              # if using Ollama (default)

python -m shadow_loom_ui              # NiceGUI workspace → http://localhost:7860
python -m shadow_loom_mcp             # MCP server (stdio)
python run_pipeline.py                # end-to-end demo
python -m pytest tests/ --ignore=tests/test_live_e2e.py -q   # ~978 tests
```

### Option C — Docker Compose (Postgres included)

```bash
cp .env.example .env                  # set STORAGE_SECRET to anything random
docker compose up --build             # → http://localhost:7860 backed by Postgres
```

The compose stack reaches an Ollama running on the host via
`host.docker.internal`. To run fully containerised, switch
`DEFAULT_MODEL` to a hosted provider in `.env`.

### Configuration

Every runtime knob lives in [`.env`](.env.example) (local) or
real environment variables (production). Defaults are baked into
[`shadow_loom/settings.py`](shadow_loom/settings.py) so a blank `.env`
already works for local Ollama. The full reference is in
[docs/settings.md](docs/settings.md).

See [docs/ui-guide.md](docs/ui-guide.md) for the workspace tour,
[docs/mcp-guide.md](docs/mcp-guide.md) for the MCP tool catalogue and the
Claude Desktop / Cursor configuration snippet, and
[docs/testing.md](docs/testing.md) for the test-suite layout.

---

## Deploy to a PaaS

Shadow-Loom ships infrastructure-as-code for two managed targets that
share the same [`Dockerfile`](Dockerfile). Pick whichever you prefer
— there is no functional difference between the two.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy?template=https://github.com/dwlmt/shadow-loom)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/dwlmt/shadow-loom)

| Target | Config | Walkthrough |
| --- | --- | --- |
| **Railway** | [`railway.toml`](railway.toml) | [docs/railway-deployment.md](docs/railway-deployment.md) |
| **Render** | [`render.yaml`](render.yaml) | [docs/render-deployment.md](docs/render-deployment.md) |

Both blueprints provision managed Postgres 16, generate a stable
`STORAGE_SECRET`, and inject `$PORT` into the container. You only
need to fill in `OPENROUTER_API_KEY`, `OAUTH_REDIRECT_BASE` and at
least one OAuth provider's `_CLIENT_ID` / `_CLIENT_SECRET` in the
platform dashboard.

On first boot the example seeder
([`shadow_loom_ui/example_seeder.py`](shadow_loom_ui/example_seeder.py))
automatically loads every fixture from
[`example_worlds/`](example_worlds/) with the matching prose from
[`sample_plots/`](sample_plots/), so the dashboard is populated with
ready-to-fork example projects (Macbeth, Death on the Nile, Reservoir
Dogs, …) the moment the deploy goes live.

For production you'll typically want OpenRouter (Render and Railway
web services have no GPU) — copy [`.env.production.example`](.env.production.example)
into the dashboard or use the defaults baked into the blueprint.

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
tests/                      # ~978 pytest tests — see docs/testing.md
docs/                       # long-form documentation
```

---

## Upgrading from pre-Channel projects

If you have stored projects that pre-date the `Channel` / `utterance`
refactor (their `world_state_json` blobs contain
`information_topology=[InformationEdge(...)]`), `WorldStateV1` will now
refuse to load them with a clear error. Run the one-shot migrator
once to rewrite each fixture in place:

```bash
python scripts/migrate_information_edges.py path/to/file_or_dir
```

The script splits each legacy `InformationEdge` into a `Channel`
(standing capability) plus an `EventNode(event_type='utterance', ...)`
(discrete message) and updates the relevant imports.

---

## Status & licence

Research project; APIs are stable enough to use but evolve between minor
versions. See [docs/design-decisions.md](docs/design-decisions.md) for the
choices that shape the public surface and what we deliberately rejected.

**Copyright © 2026 David Rae Wilmot.** Shadow Loom is **dual-licensed**:

* **Open source** under the [GNU Affero General Public License v3.0](LICENSE)
  (AGPLv3). If you modify Shadow Loom or make it available to users over a
  network (SaaS, hosted MCP server, internal web service, etc.), AGPLv3
  § 5 and § 13 require you to release the **complete corresponding source
  code of the entire combined work** to those users under AGPLv3.
* **Commercial license (AGPLv3 exception)** — see
  [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md). Organisations that
  cannot or do not wish to comply with AGPLv3's network-use disclosure
  obligation can obtain a paid commercial licence from the copyright
  holder that removes those obligations and permits proprietary
  derivatives and closed-source SaaS deployments.

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/)
plus a copyright licence-back to the maintainer that lets contributions
be redistributed under both licences — see
[COMMERCIAL-LICENSE.md § 5](COMMERCIAL-LICENSE.md#5-contributor-licensing).

### What can I do with Shadow Loom?

Quick guide — **not legal advice**. The authoritative texts are
[LICENSE](LICENSE), [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

**Open-source use (AGPLv3) plus Non-Commercial Clause:**

- [x] Use it for personal projects, research, and learning.
- [x] Read, modify, and fork the source code.
- [x] Run it on your own machine without restriction.
- [x] Redistribute it — as long as you keep it under AGPL-3.0-or-later.
- [ ] ⚠ Host it as a network service (SaaS, MCP server, hosted UI):
      you must offer the complete corresponding source code (including
      your modifications) to your users under AGPLv3.
- [ ] ⚠ Embed or link it into a larger product: the whole combined
      work must also be released under AGPLv3.
- [ ] ✗ Use it commercially without complying with AGPLv3 § 13
      (network-use disclosure) — that requires a paid commercial
      licence.
- [ ] ✗ Re-license it under a more permissive licence, or ship it
      inside closed-source software, without a commercial licence.

**Contributing:**

- [x] Open issues and pull requests on GitHub.
- [x] Contributions are accepted under the Developer Certificate of
      Origin (DCO) plus a copyright licence-back so they can ship
      under both the AGPL and the commercial licence.
- [x] Sign your commits with `git commit -s` to certify the DCO.

Contact: `david.wilmot@gmail.com`
