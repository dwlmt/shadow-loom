# Settings & Configuration

Every runtime knob in Shadow-Loom lives in one place:
[`shadow_loom/settings.py`](../shadow_loom/settings.py). The defaults baked
into that file are mirrored in [`config.env`](../config.env) — copy that
file to `.env` (or export the variables) to override them.

Settings are loaded once per process via `get_settings()` (cached). Each
section is a `pydantic-settings` `BaseSettings` subclass with an
`env_prefix`, so changing `GENERATION_MAX_TOKENS=128000` in your
environment overrides `GenerationSettings.max_tokens`.

---

## 1. How configuration flows

```
config.env / shell env
         │
         ▼
shadow_loom/settings.py          ← single source of truth
    ├─ CoreSettings              (DATABASE_URL, *_API_KEY, *_BASE_URL)
    ├─ GenerationSettings        (GENERATION_*)
    ├─ QueryParsingSettings      (QUERY_PARSING_*)
    ├─ AuditorSettings           (AUDITOR_*)
    ├─ ExtractionSettings        (EXTRACTION_*)
    ├─ CausalPhysicsSettings     (PHYSICS_*)
    ├─ MCPSettings               (MCP_*)
    ├─ PipelineSettings          (PIPELINE_*)
    ├─ UISettings                (UI_*)
    └─ OAuthSettings             (STORAGE_SECRET, *_CLIENT_ID/SECRET)
         │
         ▼
*_config_kwargs() builders       ← hand the right slice to each module
         │
         ▼
GenerationConfig, AuditorConfig, ExtractionConfig, QueryParsingConfig
(per-module Pydantic dataclasses with the same defaults)
```

The dataclass `*Config` objects (in [`generation.py`](../shadow_loom/generation.py),
[`auditor.py`](../shadow_loom/auditor.py), [`ingestion.py`](../shadow_loom/ingestion.py),
[`query_parsing.py`](../shadow_loom/query_parsing.py)) carry the same
defaults as the settings module — keep them in sync if you change one.

---

## 2. Core / providers

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///shadow_loom.db` | SQLModel connection string. Swap for Postgres in production. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1/` | Ollama OpenAI-compatible endpoint. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint. |
| `OPENROUTER_API_KEY` | *(empty)* | Required when any `*_MODEL` uses the `openrouter:` prefix. |
| `OPENAI_API_KEY` | *(empty)* | Required when any `*_MODEL` uses the `openai:` prefix. |
| `LANGFUSE_*` | *(public demo keys)* | Optional tracing — replace with your own project keys or blank to disable. |
| `DEFAULT_MODEL` | `ollama:qwen3.6:35b` | Fallback model string used when a sub-section's `*_MODEL` is unset. |

Model strings are parsed by PydanticAI: the prefix selects the provider
(`ollama:`, `openrouter:`, `openai:`) and the suffix is the model id.

---

## 3. Generation (Step 10 — prose rendering)

| Variable | Default | Notes |
|---|---|---|
| `GENERATION_MODEL` | `ollama:qwen3.6:35b` | The "creative" model. Larger models pay off here. |
| `GENERATION_MAX_TOKENS` | `64000` | Tuned for the 256K-context qwen3.6:35b. Lower for smaller-context models. |
| `GENERATION_TEMPERATURE` | `0.7` | Creative temperature for prose. |
| `GENERATION_OUTPUT_RETRIES` | `5` | PydanticAI structured-output validation retries per call. |

See [pipeline-walkthrough.md §Step 10](pipeline-walkthrough.md) for how the
brief becomes prose, and [academic-foundations.md §4.1](academic-foundations.md#41-the-brief-as-constraint-pattern-creativebrief)
for the brief-as-constraint pattern.

---

## 4. Query parsing (Step 0 — natural-language → typed query)

| Variable | Default | Notes |
|---|---|---|
| `QUERY_PARSING_MODEL` | `ollama:qwen3.6:35b` | Classification model — needs to be reliable on structured output, not creative. |
| `QUERY_PARSING_MAX_TOKENS` | `64000` | Generous to allow long chain-of-thought. |
| `QUERY_PARSING_TEMPERATURE` | `0.1` | Near-deterministic classification. |
| `QUERY_PARSING_OUTPUT_RETRIES` | `5` | Retries if the parser returns invalid JSON. |

See [query-and-cycles.md](query-and-cycles.md) for the eight query types
this stage discriminates between.

---

## 5. Auditor (Step 11 — LLM-as-judge audit/refine loop)

| Variable | Default | Notes |
|---|---|---|
| `AUDITOR_MODEL` | `ollama:qwen3.6:35b` | The judging model. |
| `AUDITOR_GENERATION_MODEL` | `ollama:qwen3.6:35b` | The model used for re-renders inside the loop. |
| `AUDITOR_MAX_ITERATIONS` | `3` | Hard cap on audit → rewrite cycles. Each iteration costs two LLM calls. |
| `AUDITOR_OUTPUT_RETRIES` | `5` | Structured-output retries per call. |
| `AUDITOR_TEMPERATURE` | `0.2` | Low temperature for deterministic auditing. |
| `AUDITOR_GENERATION_TEMPERATURE` | `0.7` | Creative temperature for re-renders. |
| `AUDITOR_MAX_TOKENS` | `64000` | Token budget for audit verdict. |
| `AUDITOR_MAX_TOKENS_GENERATION` | `64000` | Token budget for re-renders. |
| `AUDITOR_MIN_FORESHADOWING_SCORE` | `0.6` | Pass threshold for foreshadowing payoff. |
| `AUDITOR_MAX_AFFECTIVE_LOSS` | `0.3` | Max allowed drift between requested and delivered affective score. |
| `AUDITOR_MIN_COGNITIVE_PLAUSIBILITY` | `0.7` | Minimum cognitive-plausibility score to pass. |
| `AUDITOR_MAX_MIRACLE_STEPS` | `0` | Number of unmechanism'd jumps tolerated before failing. |
| `AUDITOR_IGNORE_SPATIAL_BLOCKS` | `false` | If true, suppresses affordance/co-location violations (debugging only). |

The thresholds map directly onto the auditor categories described in
[architecture.md §Step 11](architecture.md) and the LLM-as-judge protocol in
[academic-foundations.md §4.3](academic-foundations.md#43-llm-as-judge-audit-loop).

> **Cost knob.** `AUDITOR_MAX_ITERATIONS` is the single biggest LLM-cost
> lever in the pipeline: each iteration runs one audit + one re-render, so
> raising it from 3 to 5 can ~70% the audit-step cost in the worst case.

---

## 6. Extraction / ingestion (Steps 1–2 — text → graph)

| Variable | Default | Notes |
|---|---|---|
| `EXTRACTION_MODEL` | `ollama:qwen3.6:35b` | Topology extractor. |
| `EXTRACTION_CHUNK_STRATEGY` | `act_headings` | `act_headings` splits on `Act N` / `Chapter N` markers, `paragraph` packs by size. |
| `EXTRACTION_OUTPUT_RETRIES` | `5` | First-pass structured-output retries. |
| `EXTRACTION_FABULA_TIME_SPACING` | `1000` | Initial gap between fabula-time stamps; leaves room for flashbacks/inserts. |
| `EXTRACTION_MIN_CHUNK_CHARS` | `1500` | Minimum chunk size before merging adjacent paragraphs. |
| `EXTRACTION_CHUNK_OVERLAP_CHARS` | `300` | Trailing context prepended to the next chunk for coreference. |
| `EXTRACTION_MAX_CORRECTION_RETRIES` | `5` | Max validation-feedback repair passes (after the initial extract). |
| `EXTRACTION_MAX_CONCURRENT_CHUNKS` | `8` | Parallel async LLM extraction concurrency. Tune to your model's throughput. |
| `EXTRACTION_ESTIMATED_EVENTS_PER_CHUNK` | `10` | Pre-allocates syuzhet/fabula-time ranges for parallel extraction. |
| `EXTRACTION_ENABLE_RESEARCH_AGENT` | `false` | Opt-in: run Step 3d external research after world-state assembly. Off by default. |
| `EXTRACTION_RESEARCH_PROVIDER` | `none` | `none` (no-op) or `tavily` (requires `TAVILY_API_KEY` and `pip install -e ".[research]"`). |
| `EXTRACTION_RESEARCH_PROVIDER_MODEL` | *(empty)* | Provider-specific search depth (e.g. Tavily `basic` vs `advanced`). Hashed into the per-user cache key. |
| `EXTRACTION_RESEARCH_MAX_RESULTS_PER_QUERY` | `5` | Cap on snippets returned per provider call. |
| `EXTRACTION_RESEARCH_TOPICS` | *(empty)* | Pre-configured topics looked up at extraction time. Topics may also be added live via the `research_topic` MCP tool. |
| `TAVILY_API_KEY` | *(empty)* | Required only when `EXTRACTION_RESEARCH_PROVIDER=tavily`. See [research-extraction-plan.md](research-extraction-plan.md). |

The Socratic-scaffold extraction protocol is described in
[pipeline-walkthrough.md §Steps 1–2](pipeline-walkthrough.md), with the
academic backing in
[academic-foundations.md §6.5 (Computational narratology)](academic-foundations.md#65-computational-narratology-and-story-understanding).

---

## 7. Causal physics (Step 7)

| Variable | Default | Notes |
|---|---|---|
| `PHYSICS_STRENGTH_WEAK` | `0.25` | Multiplier for `causal_strength="weak"` edges. |
| `PHYSICS_STRENGTH_MODERATE` | `0.5` | Moderate multiplier. |
| `PHYSICS_STRENGTH_STRONG` | `0.75` | Strong multiplier. |
| `PHYSICS_MECHANISM_FALLBACK_FACTOR` | `0.2` | Damping when no `mechanism` is declared on an edge. |
| `PHYSICS_DEFAULT_TRAIT_BASELINE` | `0.5` | Maximum-entropy starting belief for unobserved traits. |
| `PHYSICS_DEFAULT_CAUSAL_FORCE` | `5.0` | Baseline force per causal edge. |
| `PHYSICS_CAUSAL_FORCE_SCALING` | `10.0` | Scales the impact → trait-delta conversion. |
| `PHYSICS_RELATIONSHIP_INERTIA_DEFAULT` | `0.3` | Resistance applied to relationship-edge mutations. |
| `PHYSICS_AMBIENT_FORCE_MULTIPLIER` | `2.0` | Multiplier for ambient-state forces (weather, mood, era). |
| `PHYSICS_INERTIA_EPSILON` | `0.0` | Small perturbation added to inertia to prevent stalling. |
| `PHYSICS_EGO_MEMORY_LIMIT` | `5` | How many prior events the ego-graph keeps for the focal entity. |

### Probabilistic propagation, Monte-Carlo and Bayesian abduction

These knobs were added with the channel/utterance and Bayesian-abduction
refactor; defaults preserve previous behaviour for everything except
`abduction_blend_mode` (which now defaults to `bayesian`).

| Variable | Default | Notes |
|---|---|---|
| `PHYSICS_PROPAGATION_MODE` | `noisy_or` | `deterministic` keeps the legacy weighted-average + `\|impact\| > inertia` gate. `noisy_or` treats each incoming edge as an independent Bernoulli attempt to overcome inertia. |
| `PHYSICS_NOISY_OR_TEMPERATURE` | `0.25` | Sigmoid temperature for the per-edge gate. Lower = sharper threshold. |
| `PHYSICS_NOISY_OR_THRESHOLD` | `0.5` | Aggregate noisy-OR probability above which a trait is considered to have shifted (deterministic noisy-OR path). |
| `PHYSICS_CAUSAL_FORCE_SIGMA_WEAK` | `0.30` | Std-dev (fraction of nominal force) for `evidence_strength="weak"` edges in Monte-Carlo CTF. |
| `PHYSICS_CAUSAL_FORCE_SIGMA_MODERATE` | `0.15` | Same for `evidence_strength="moderate"`. |
| `PHYSICS_CAUSAL_FORCE_SIGMA_STRONG` | `0.05` | Same for `evidence_strength="strong"`. |
| `PHYSICS_MONTE_CARLO_SAMPLES` | `128` | When > 0, the plain `CausalPhysicsEngine.execute()` auto-routes through `execute_distribution`, which returns a per-trait distribution (mean / p5 / p50 / p95) instead of a point estimate. Default of 128 gives a posterior-mean Monte-Carlo standard error of `sigma/sqrt(N) <= 0.044` for [0,1] traits while keeping per-call cost bounded; raise to 500–1000 for tight tail estimation, set to `0` to disable Monte-Carlo entirely. Cost scales linearly: every `execute()` call runs N propagations (affects `directive_assembly.evaluate_candidate_events` and `narrative_physics`). |
| `PHYSICS_MONTE_CARLO_SEED` | unset | Optional RNG seed for reproducible Monte-Carlo runs. |
| `PHYSICS_ABDUCTION_BLEND_MODE` | `bayesian` | `legacy`: `blended = old + delta * (1 - inertia)`. `bayesian`: `posterior = (inertia*old + ev_precision*evidence) / (inertia + ev_precision)`. |
| `PHYSICS_ABDUCTION_EVIDENCE_PRECISION` | `1.0` | Precision (1/variance) of present-day evidence in the Bayesian abduction blend. |
| `PHYSICS_ENTITY_TRAIT_INERTIA_DEFAULT` | `0.5` | Fallback inertia for an Entity trait when extraction is silent. |
| `PHYSICS_BELIEF_INERTIA_DEFAULT` | `0.3` | Fallback inertia for a Belief. |
| `PHYSICS_WORLD_TRAIT_INERTIA_DEFAULT` | `0.8` | Fallback inertia for a `WORLD_` trait magnitude. |
| `PHYSICS_EVENT_STATE_INERTIA_DEFAULT` | `0.0` | Quoted by the auditor when contrasting event volatility with character stability. |
| `PHYSICS_ENTITY_TRAIT_BASELINE_DRIFT_RATE` | `0.0` | If > 0, after each step entity traits drift back toward their `state_timeline` baseline by `(1 - inertia) * rate`. |
| `PHYSICS_RULE3_PRUNING_MODE` | `advisory` | `advisory` reports Rule-3 pruned interventions but still executes the do-surgery; `prune` drops them. Use `prune` only when the extracted causal topology is known to be confounder-complete. |
| `PHYSICS_ALLOW_UNOBSERVED_CONFOUNDERS` | `false` | Safety-net debugging toggle that injects synthetic latent parents for every pair of nodes sharing an observed cause. Combinatorial — leave off in normal use; the extraction prompts elicit named latents as `WORLD_*` traits instead. |
| `PHYSICS_INTELLIGIBILITY_THRESHOLD` | `0.3` | Per-recipient channel intelligibility below which a belief acquired through that channel is treated as epistemically invalid (used in abduction belief back-prop and directive-assembly leak risk). |

### Ingress word cap (UI / chat / MCP)

| Variable | Default | Notes |
|---|---|---|
| `PHYSICS_MAX_INGEST_WORDS` | `10000` | Maximum whitespace tokens accepted by any user-facing ingest (NiceGUI Story tab, chat box, MCP `ingest` / `narrate` / `write`). Enforced uniformly so a payload that's too large fails fast at the boundary instead of mid-pipeline. |

These constants are the dial-board of the
[`CausalPhysicsEngine`](../shadow_loom/causal_physics.py). Their motivation
is laid out in
[academic-foundations.md §2.3 (Abduction)](academic-foundations.md#23-abduction-causalphysicsengineabduction_update)
and [§5.1 (Trait + inertia model)](academic-foundations.md#51-trait--inertia-model).

> **Tuning rule of thumb.** If physics-driven mutations feel "too weak"
> (auditor reports flat affect), raise `PHYSICS_CAUSAL_FORCE_SCALING`. If
> they feel "too explosive" (out-of-character swings), lower it or raise
> `PHYSICS_RELATIONSHIP_INERTIA_DEFAULT`.

---

## 8. MCP server

| Variable | Default | Notes |
|---|---|---|
| `MCP_SKIP_AUDIT` | `true` | MCP defaults to no audit (latency-sensitive). Override per-call when needed. |
| `MCP_INGEST_FABULA_TIME_SPACING` | `1000` | Initial gap between fabula-time stamps for MCP ingest (matches the pipeline default; leaves room for flashbacks/inserts). |
| `MCP_INGEST_MAX_CORRECTION_RETRIES` | `5` | Validation-repair passes during MCP ingest. |
| `MCP_ALLOW_OPEN_MODE` | `false` | **Production must keep this false.** When true, scope checks pass when no scopes are resolved (dev/local mode only). |

Full MCP tool/resource catalogue and auth flow:
[mcp-guide.md](mcp-guide.md).

---

## 9. Pipeline

| Variable | Default | Notes |
|---|---|---|
| `PIPELINE_USE_CAUSAL_ENGINE` | `true` | When false, intervention/counterfactual queries skip Step 7 (debug only). |
| `PIPELINE_SKIP_AUDIT` | `false` | When true, bypasses the Step-11 audit loop. |
| `PIPELINE_SKIP_REEXTRACTION` | `false` | When true, skips Step 12 re-extraction (the new prose isn't merged back into the graph). |
| `PIPELINE_MAX_SNAPSHOTS` | `10` | Maximum versioned snapshots retained per world. |

These are the whole-pipeline switches consumed by
[`shadow_loom/pipeline.py`](../shadow_loom/pipeline.py). For what each step
actually does see [pipeline-walkthrough.md](pipeline-walkthrough.md).

---

## 10. UI

| Variable | Default | Notes |
|---|---|---|
| `UI_HOST` | `0.0.0.0` | NiceGUI bind address. |
| `UI_PORT` | `7860` | NiceGUI port. |
| `UI_TITLE` | `Shadow Loom` | Browser tab title. |
| `UI_DARK_MODE` | `false` | Default theme. |
| `UI_RELOAD` | `false` | Hot-reload (development only). |

For the workspace tour see [ui-guide.md](ui-guide.md).

---

## 11. OAuth (UI auth)

Auth is **disabled by default** — the UI runs as a single-user local app.
To enable an OAuth provider, set its client id + secret pair (and provide
`STORAGE_SECRET` for cookie signing). Any non-empty client id flips
`OAuthSettings.auth_enabled` to true.

| Variable | Notes |
|---|---|
| `STORAGE_SECRET` | Cookie/session signing secret. Auto-generated per process if blank — set this in production so sessions survive restarts. |
| `OAUTH_REDIRECT_BASE` | Public base URL for callback URLs (e.g. `https://shadow.example.com`). |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | GitHub OAuth app credentials. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google OAuth credentials. |
| `DISCORD_CLIENT_ID` / `DISCORD_CLIENT_SECRET` | Discord OAuth credentials. |
| `MICROSOFT_CLIENT_ID` / `MICROSOFT_CLIENT_SECRET` | Microsoft OAuth credentials. |

---

## 12. Tuning recipes

**"I want to use a smaller-context model (e.g. 8K-context Llama)."**
Drop every `*_MAX_TOKENS` to ~4000, drop `EXTRACTION_MIN_CHUNK_CHARS` to
~600, and lower `EXTRACTION_CHUNK_OVERLAP_CHARS` proportionally so you
have headroom for the prompt scaffold.

**"I want OpenRouter / OpenAI everywhere."**
Set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) and replace each
`*_MODEL` value with `openrouter:<provider>/<model>` (or
`openai:<model>`). `DEFAULT_MODEL` is the fallback for any unset
section.

**"I want fast iteration during development."**
`PIPELINE_SKIP_AUDIT=true`, `PIPELINE_SKIP_REEXTRACTION=true`,
`AUDITOR_MAX_ITERATIONS=1`, `EXTRACTION_MAX_CONCURRENT_CHUNKS=4`. Re-enable
audit for any run you'll evaluate.

**"I want maximum quality at any cost."**
`AUDITOR_MAX_ITERATIONS=5`, `AUDITOR_MIN_FORESHADOWING_SCORE=0.75`,
`AUDITOR_MIN_COGNITIVE_PLAUSIBILITY=0.85`,
`AUDITOR_MAX_AFFECTIVE_LOSS=0.15`. Expect ~2× the LLM cost.

**"My physics-driven trait deltas feel wrong."**
First check `PHYSICS_CAUSAL_FORCE_SCALING` (raise/lower together with
`PHYSICS_DEFAULT_CAUSAL_FORCE`). Then `PHYSICS_RELATIONSHIP_INERTIA_DEFAULT`
to soften / sharpen response. See
[academic-foundations.md §2.3](academic-foundations.md#23-abduction-causalphysicsengineabduction_update)
for the underlying model.

---

## See also

* [`shadow_loom/settings.py`](../shadow_loom/settings.py) — the canonical source for every default value.
* [`config.env`](../config.env) — copy-pasteable environment template.
* [architecture.md](architecture.md) — how each setting flows into the 12-step pipeline.
* [pipeline-walkthrough.md](pipeline-walkthrough.md) — code-level walkthrough showing where each `*Config` is consumed.
* [academic-foundations.md](academic-foundations.md) — the literature behind the threshold defaults (Wilmot suspense, Halpern actual causality, Pearl ladder, AMWN).
* [mcp-guide.md](mcp-guide.md) — MCP-specific overrides and scope rules.
