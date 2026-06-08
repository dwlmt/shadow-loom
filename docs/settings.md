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
    ├─ DirectiveAssemblySettings (DIRECTIVE_ASSEMBLY_*)
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

Shadow-Loom understands three families of LLM providers:

1. **Ollama** — local-first default, addressed via `ollama:<model-id>`.
2. **OpenAI-compatible HTTP endpoints** — addressed via
   `<prefix>:<model-id>`. The built-in registry covers 23 providers:

   | Group  | Prefixes |
   |---|---|
   | Cloud SaaS | `openrouter`, `openai`, `fireworks`, `featherless`, `together`, `deepinfra`, `groq`, `anyscale`, `perplexity`, `huggingface`, `mistral`, `xai`, `deepseek`, `moonshot`, `cerebras`, `sambanova`, `nebius`, `novita`, `hyperbolic` |
   | Local runtimes | `llamacpp` (port 8080), `vllm` (port 8000), `lmstudio` (port 1234), `localai` (port 8080), `unsloth` (Unsloth Studio, port 7860) |

   Local prefixes do not require an API key — the resolver injects a
   placeholder. Every cloud prefix expects a `<PREFIX>_API_KEY` env var
   (e.g. `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`) and optionally a
   `<PREFIX>_BASE_URL` override. You may register additional providers
   at runtime with `SHADOW_LOOM_PROVIDERS="prefix=https://host/v1,..."`.

3. **Per-user overrides (UI)** — see [§2a](#2a-per-user-overrides). Any
   signed-in user can set their own default model, per-stage models, and
   private OpenAI-compatible providers from the **Settings → Models &
   Providers** card. User overrides take precedence over the env vars
   below, so a deployment can ship with no `DEFAULT_MODEL` at all and
   let each user wire up their own keys.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///shadow_loom.db` | SQLModel connection string. Swap for Postgres in production. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1/` | Ollama OpenAI-compatible endpoint. |
| `OLLAMA_NUM_CTX` | `262144` | Context-window size (in tokens) requested for every `ollama:` model call. Ollama's OpenAI-compat endpoint defaults to **2 048 tokens** and silently truncates anything larger, producing empty/malformed JSON. This value is forwarded as `extra_body={"options": {"num_ctx": <N>}}` on every request; Ollama ≥0.6.x honours it at model-load time. Defaults to 256K to match the reference qwen3.6:35b model — lower it on smaller models (e.g. `32768` for an 8B model). Set `0` to disable forwarding. **Cloud providers (OpenAI, OpenRouter, Anthropic, …) are completely unaffected** — the `extra_body` is only set on the `ollama:` provider branch. See the [Local Ollama context size](#local-ollama-context-size) section below for belt-and-braces fallbacks on older Ollama versions. |
| `<PREFIX>_BASE_URL` | *(provider-specific default)* | Override the HTTP endpoint for any built-in provider. |
| `<PREFIX>_API_KEY` | *(empty)* | Required when any model string uses that prefix (cloud only). |
| `SHADOW_LOOM_PROVIDERS` | *(empty)* | Comma-separated `prefix=base_url` pairs to extend the built-in registry without touching code. |
| `LANGFUSE_*` | *(public demo keys)* | Optional tracing — replace with your own project keys or blank to disable. |
| `DEFAULT_MODEL` | `ollama:qwen3.6:35b` | Deployment-wide fallback. Every stage-specific `*_MODEL` below inherits this value when left unset, and every user without an explicit override inherits it too. May be left blank if every user is expected to configure their own default in the UI. |

Model strings are parsed by PydanticAI: the prefix selects the provider
and the suffix is the model id.

### 2a. Per-user overrides

The **Settings → Models & Providers** UI lets each signed-in user
persist their own:

- **Default model** — overrides `DEFAULT_MODEL` for that user.
- **Per-stage models** — separately override Generation, Auditor,
  Auditor re-renders, Extraction, and Query parsing.
- **Custom OpenAI-compatible providers** — prefix, base URL, API key,
  and "local" flag. Once saved, they appear in every model field as
  `<prefix>:<model-id>` and override the built-in registry entry of the
  same name (so a user can bring their own API key for a built-in
  provider too).

Saved values live in the `user_model_settings` table and are activated
at the start of every pipeline call (UI submission, deferred
re-extraction worker, and MCP API requests) via a ContextVar. Stages
that were started against the env default are transparently re-routed
to the user's default; stages that were given an explicit non-default
model string are left alone.

---

## 3. Generation (Step 10 — prose rendering)

| Variable | Default | Notes |
|---|---|---|
| `GENERATION_MODEL` | *(inherits `DEFAULT_MODEL`)* | The "creative" model. Larger models pay off here. Set only to override `DEFAULT_MODEL` for this stage. |
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
| `QUERY_PARSING_MODEL` | *(inherits `DEFAULT_MODEL`)* | Classification model — needs to be reliable on structured output, not creative. Set only to override `DEFAULT_MODEL` for this stage. |
| `QUERY_PARSING_MAX_TOKENS` | `64000` | Generous to allow long chain-of-thought. |
| `QUERY_PARSING_TEMPERATURE` | `0.1` | Near-deterministic classification. |
| `QUERY_PARSING_OUTPUT_RETRIES` | `5` | Retries if the parser returns invalid JSON. |

See [query-and-cycles.md](query-and-cycles.md) for the eight query types
this stage discriminates between.

---

## 5. Auditor (Step 11 — LLM-as-judge audit/refine loop)

| Variable | Default | Notes |
|---|---|---|
| `AUDITOR_MODEL` | *(inherits `DEFAULT_MODEL`)* | The judging model. Set only to override `DEFAULT_MODEL` for this stage. |
| `AUDITOR_GENERATION_MODEL` | *(inherits `DEFAULT_MODEL`)* | The model used for re-renders inside the loop. Set only to override `DEFAULT_MODEL` for this stage. |
| `AUDITOR_MAX_ITERATIONS` | `6` | Hard cap on audit → rewrite cycles. Each iteration costs two LLM calls. **Worst-case LLM-call budget per scene** ≈ `AUDITOR_MAX_ITERATIONS × (1 + AUDITOR_OUTPUT_RETRIES) × _PROVIDER_RETRY_ATTEMPTS` = `6 × 6 × 6 = 216` calls. In practice the loop short-circuits as soon as a clean scene is produced; reduce this or `AUDITOR_OUTPUT_RETRIES` first if your provider has tight rate limits. |
| `AUDITOR_OUTPUT_RETRIES` | `5` | Structured-output retries per call. Multiplies with `AUDITOR_MAX_ITERATIONS` and `_PROVIDER_RETRY_ATTEMPTS` (=6, hardcoded) to bound the per-scene LLM call budget. |
| `AUDITOR_TEMPERATURE` | `0.2` | Low temperature for deterministic auditing. |
| `AUDITOR_GENERATION_TEMPERATURE` | `0.7` | Creative temperature for re-renders. |
| `AUDITOR_MAX_TOKENS` | `64000` | Token budget for audit verdict. |
| `AUDITOR_MAX_TOKENS_GENERATION` | `64000` | Token budget for re-renders. |
| `AUDITOR_MIN_FORESHADOWING_SCORE` | `0.6` | Pass threshold for foreshadowing payoff. |
| `AUDITOR_MAX_AFFECTIVE_LOSS` | `0.3` | Max allowed drift between requested and delivered affective score. |
| `AUDITOR_MIN_COGNITIVE_PLAUSIBILITY` | `0.7` | Minimum cognitive-plausibility score to pass. Counts entities whose *actions* contradict their *own* established beliefs; merely holding a belief that turns out to be wrong (dramatic irony) is **not** penalised. |
| `AUDITOR_MAX_MIRACLE_STEPS` | `0` | Number of unmechanism'd jumps tolerated before failing. |
| `AUDITOR_IGNORE_SPATIAL_BLOCKS` | `false` | If true, suppresses affordance/co-location violations (debugging only). |

The thresholds map directly onto the auditor categories described in
[architecture.md §Step 11](architecture.md) and the LLM-as-judge protocol in
[academic-foundations.md §4.3](academic-foundations.md#43-llm-as-judge-audit-loop).

> **Cost knob.** `AUDITOR_MAX_ITERATIONS` is the single biggest LLM-cost
> lever in the pipeline: each iteration runs one audit + one re-render, so
> raising it from 3 to 5 can ~70% the audit-step cost in the worst case.

> **Loop discipline (non-tunable, but worth knowing).** The refinement
> loop is hardened against the classic ping-pong failure where each
> iteration fixes one category and regresses on a previous one:
>
> * `meta` (meta-narration) and `style` (style-fidelity) audits run for
>   *every* `target_effect` regardless of the brief's `audit_categories`;
>   per-effect categories layer on top via
>   `auditor.resolve_audit_categories(...)`. The judge prompt is told to
>   surface violations *only* for resolved categories — anything else
>   would be silently discarded by the loop.
> * Each refinement call passes prior violations as
>   `=== NON-REGRESSION CONSTRAINTS ===` so earlier fixes are preserved.
> * `style_mismatch` violations are graded `critical` (form-class
>   breach), `major` (word count >±50% off-budget or density+form drift),
>   or `minor` (pure prose-density drift inside band). When *all*
>   surfaced violations in an iteration are `minor`, the loop short-
>   circuits to `llm_passed=True` instead of burning another regeneration.
> * When `affective_loss_mse` has no scorable target the evaluation
>   prompt prints `not measured (no scorable target — ignore in
>   evaluation)` instead of formatting a misleading `0.0000`.

---

## 6. Extraction / ingestion (Steps 1–2 — text → graph)

| Variable | Default | Notes |
|---|---|---|
| `EXTRACTION_MODEL` | *(inherits `DEFAULT_MODEL`)* | Topology extractor. Set only to override `DEFAULT_MODEL` for this stage. |
| `EXTRACTION_CHUNK_STRATEGY` | `act_headings` | `act_headings` splits on `Act N` / `Chapter N` markers, `paragraph` packs by size. |
| `EXTRACTION_OUTPUT_RETRIES` | `5` | First-pass structured-output retries. |
| `EXTRACTION_DEFAULT_MAX_TOKENS` | `16384` | Baseline `max_tokens` (output-token cap) applied to **every** extraction agent that does not pass its own per-call `model_settings`. Wired in at `Agent(...)` construction time so it propagates through PydanticAI's `merge_model_settings` shallow merge (per-call overrides like the catalogue's 65 536 still win). Prevents silent-truncation cascades on providers that enforce a low default output cap (Together, Fireworks, some OpenRouter routings). On the OpenAI provider this is automatically remapped to `max_completion_tokens` for reasoning models, so the value works uniformly across OpenAI / Anthropic / Google / OpenRouter / Ollama. |
| `EXTRACTION_DEFAULT_TEMPERATURE` | `0.1` | Baseline sampling temperature for extraction agents that don't pass their own `model_settings`. Extraction is determinism-preferred (low temperature reduces id-minting variance across re-runs). Per-call overrides still win via shallow merge. |
| `EXTRACTION_FABULA_TIME_SPACING` | `1000` | Initial gap between fabula-time stamps; leaves room for flashbacks/inserts. |
| `EXTRACTION_MIN_CHUNK_CHARS` | `800` | Minimum chunk size before merging adjacent paragraphs. |
| `EXTRACTION_CHUNK_OVERLAP_CHARS` | `300` | Trailing context prepended to the next chunk for coreference. |
| `EXTRACTION_MAX_CORRECTION_RETRIES` | `5` | Max validation-feedback repair passes (after the initial extract). |
| `EXTRACTION_VALIDATION_PAYLOAD_MAX_CHARS` | `600000` | Hard cap on the WorldStateV1 JSON sent to the LLM validator (Step 3 Phase B). When the serialised state exceeds this, the validator switches to a compact projection (timeline-stripped) before falling back to truncation. Sized for a 256K-token context window with headroom for system prompt and structured-output response — lower this for smaller-context models. |
| `EXTRACTION_CORRECTION_SUBGRAPH_THRESHOLD_CHARS` | `400000` | When the WorldStateV1 JSON sent to the correction-patch agent exceeds this, fall back to an error-relevant subgraph (events, entities, channels, locations, objects, and world traits whose ids appear in the error details, plus one-hop causal neighbours and any spatial/social edges touching them) instead of the full state. The patch contract still applies to the full world on the way out. |
| `EXTRACTION_MAX_CONCURRENT_CHUNKS` | `12` | Parallel async LLM extraction concurrency. Tune to your model's throughput. |
| `EXTRACTION_PER_CHUNK_TIMEOUT_SECONDS` | `0` | Outer per-chunk safety timeout (seconds). **Off by default** — the per-agent-call timeout below is the primary mechanism. Set >0 only as a last-resort guard against pathological retry loops. When >0 and triggered, the chunk yields an empty `ChunkTopology` with all stage flags marked failed. |
| `EXTRACTION_PER_AGENT_CALL_TIMEOUT_SECONDS` | `600` | Per-agent-call soft timeout (seconds). Each `X_agent.run(...)` invocation inside the per-chunk pipeline (Socratic / Physics / Social / Consequences / Affect, plus all chunk-level retries) is wrapped in `asyncio.wait_for`. On timeout only the wedged call is cancelled; the surrounding per-stage `try/except` records the failure and the next stage still runs on whatever earlier stages produced. Replaces the legacy whole-chunk timeout — a single wedged agent no longer voids the other stages' work. `0` disables wrapping. |
| `EXTRACTION_ESTIMATED_EVENTS_PER_CHUNK` | `10` | Pre-allocates syuzhet/fabula-time ranges for parallel extraction. |
| `EXTRACTION_ENABLE_RESEARCH_AGENT` | `false` | Opt-in: run Step 3d external research after world-state assembly. Off by default. |
| `EXTRACTION_RESEARCH_PROVIDER` | `none` | `none` (no-op) or `tavily` (requires `TAVILY_API_KEY` and `pip install -e ".[research]"`). |
| `EXTRACTION_RESEARCH_PROVIDER_MODEL` | *(empty)* | Provider-specific search depth (e.g. Tavily `basic` vs `advanced`). Hashed into the per-user cache key. |
| `EXTRACTION_RESEARCH_MAX_RESULTS_PER_QUERY` | `5` | Cap on snippets returned per provider call. |
| `EXTRACTION_RESEARCH_TOPICS` | *(empty)* | Pre-configured topics looked up at extraction time. Topics may also be added live via the `research_topic` MCP tool. |
| `TAVILY_API_KEY` | *(empty)* | Required only when `EXTRACTION_RESEARCH_PROVIDER=tavily`. See the research section of [docs/architecture.md](architecture.md#step-3d--optional-external-research-segregated-off-by-default). |

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
| `PHYSICS_NOISY_OR_TEMPERATURE` | `0.25` | Sigmoid temperature for the per-edge gate. Lower = sharper threshold. **Bounded** `(0, 10]` (D5, thirteenth-pass audit) — zero divides the sigmoid, values above 10 produce a near-uniform gate that silently breaks every causal-physics regression. |
| `PHYSICS_NOISY_OR_THRESHOLD` | `0.5` | Aggregate noisy-OR probability above which a trait is considered to have shifted (deterministic noisy-OR path). **Bounded** `[0, 1]` (D5, thirteenth-pass audit) — it is a probability; out-of-range values used to silently disable or always-fire the gate. |
| `PHYSICS_CAUSAL_FORCE_SIGMA_WEAK` | `0.30` | Std-dev (fraction of nominal force) for `evidence_strength="weak"` edges in Monte-Carlo CTF. |
| `PHYSICS_CAUSAL_FORCE_SIGMA_MODERATE` | `0.15` | Same for `evidence_strength="moderate"`. |
| `PHYSICS_CAUSAL_FORCE_SIGMA_STRONG` | `0.05` | Same for `evidence_strength="strong"`. |
| `PHYSICS_MONTE_CARLO_SAMPLES` | `24` | **On by default (> 0), but seeded so results are stable** (see `PHYSICS_MONTE_CARLO_SEED`). When > 0, the plain `CausalPhysicsEngine.execute()` auto-routes through `execute_distribution`, which returns a per-trait distribution (mean / p5 / p50 / p95) instead of a point estimate. The default of 24 gives a posterior-mean Monte-Carlo standard error of `sigma/sqrt(N) <= 0.10` for [0,1] traits — cheap enough to leave enabled while still capturing narrative-level uncertainty; raise to 500–1000 for tight tail estimation, set to `0` to disable Monte-Carlo entirely (deterministic point-estimate). Cost scales linearly: every `execute()` call runs N propagations (affects `directive_assembly.evaluate_candidate_events` and `narrative_physics`). |
| `PHYSICS_MONTE_CARLO_SEED` | `0` | RNG seed for the Monte-Carlo sampler. **Defaults to a fixed int (0) so Monte-Carlo results — and the boundary-case plausibility/vacuity verdicts derived from them — are STABLE and reproducible run-to-run** even though Monte-Carlo is on by default. Set to unset/None for unseeded, genuinely stochastic sampling (verdict-sensitivity studies, cross-run bootstrap CIs). |
| *(logging — non-tunable)* | — | Inside a Monte-Carlo sweep the per-sample physics traces (`[CausalPhysics·Abduction]`, `[CausalPhysics·Result]`, `[Surgery]` inertia messages) are demoted from `INFO` to `DEBUG` via a `ContextVar` set by `execute_distribution`. A single aggregated `[CausalPhysics·MC]` summary banner at `INFO` level is emitted once per sweep with the trait-posterior spotlight (means / std / p5 / p95 for up to 5 entities), so console logs stay readable while DEBUG retains full per-sample detail for forensics. |
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
| `PHYSICS_MAX_INGEST_WORDS` | `10000` | Maximum whitespace tokens accepted by ingested narrative text (NiceGUI Story tab ingest dialog, MCP `ingest` / `narrate` / `write`). Enforced at the ingestion boundary so an oversized payload fails fast instead of mid-pipeline. The chat / channel input is not capped by this setting. |

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

## 8. Directive assembly (Step 8 — affective scorers)

The four structural-affect scorers (`mystery`, `dramatic_irony`,
`suspense`, `surprise`) are tuned through `DirectiveAssemblySettings`
(env prefix `DIRECTIVE_ASSEMBLY_`). Defaults match the published
formulations in
[academic-foundations.md §§3.1–3.4](academic-foundations.md#3-the-four-structural-affects)
and `paper/shadow_loom.tex`. All values are also exposed as Python
attributes on `DirectiveAssembler._settings` for callers that build
their own settings objects programmatically.

### Mystery (Sternberg curiosity gap)

| Variable | Default | Notes |
|---|---|---|
| `DIRECTIVE_ASSEMBLY_MYSTERY_PATH_DECAY_DEPTH` | `4` | Reverse-causal traversal depth cap (Trabasso & Sperry 1985 4-hop traceability). Ancestors beyond this depth drop out of the gap aggregation. |
| `DIRECTIVE_ASSEMBLY_MYSTERY_PROXIMITY_TAU_SYUZHET` | `8.0` | Curiosity-proximity decay constant in syuzhet-index units (Iser 1976 reader-gap recency). |

### Dramatic irony (knowledge asymmetry)

| Variable | Default | Notes |
|---|---|---|
| `DIRECTIVE_ASSEMBLY_IRONY_SURFACE_K` | `1.0` | Saturation constant K in the per-character gap-fraction denominator. |
| `DIRECTIVE_ASSEMBLY_IRONY_FALSE_BELIEF_MULT` | `1.5` | Boost when the focal holds a provenance-valid belief about the gap event's actor (Iago→Othello / Jacqueline→Linnet pattern). |
| `DIRECTIVE_ASSEMBLY_IRONY_ACTION_ALPHA` | `0.15` | Per-actor-event linear term in the focal-prominence weight `a_c = min(cap, 1 + α · #actor-events)`. |
| `DIRECTIVE_ASSEMBLY_IRONY_ACTION_WEIGHT_CAP` | `3.0` | Upper cap on the focal-prominence weight. |
| `DIRECTIVE_ASSEMBLY_IRONY_AGGREGATOR_BETA` | `0.6` | Convex aggregator: `β·max + (1-β)·mean` across focal characters (β closer to 1 = single-dominant-gap framing). |
| `DIRECTIVE_ASSEMBLY_IRONY_PROXIMITY_TAU_SYUZHET` | `6.0` | Closure-proximity decay constant: how sharply the gap discharges as the syuzhet approaches the truth-revealing scene. |
| `DIRECTIVE_ASSEMBLY_IRONY_PROXIMITY_FLOOR` | `0.4` | Minimum closure-proximity weight (prevents distant events from going to zero). |

### Suspense (hope/threat ledger)

| Variable | Default | Notes |
|---|---|---|
| `DIRECTIVE_ASSEMBLY_SUSPENSE_STAKES_K` | `2.0` | Saturation constant K in the stakes denominator. |
| `DIRECTIVE_ASSEMBLY_SUSPENSE_PROXIMITY_TAU_FABULA_GAPS` | `6.0` | Fabula-gap proximity decay (closer threats feel sharper). |
| `DIRECTIVE_ASSEMBLY_SUSPENSE_PROXIMITY_TAU_SPATIAL` | `4.0` | Spatial-distance proximity decay (in location hops). |
| `DIRECTIVE_ASSEMBLY_SUSPENSE_PERSISTENCE_ALPHA` | `0.10` | Per-revealed-edge persistence amplifier. |
| `DIRECTIVE_ASSEMBLY_SUSPENSE_PERSISTENCE_CAP` | `1.5` | Upper cap on the cumulative persistence multiplier. |
| `DIRECTIVE_ASSEMBLY_SUSPENSE_HOSTILE_AFFINITY` | `-0.2` | Affinity ≤ this value flags a relationship as hostile (threat side). |
| `DIRECTIVE_ASSEMBLY_SUSPENSE_ALLY_AFFINITY` | `0.2` | Affinity ≥ this value flags a relationship as allied (hope side). |

### Surprise (Beta-Bernoulli + anachrony)

| Variable | Default | Notes |
|---|---|---|
| `DIRECTIVE_ASSEMBLY_SURPRISE_TRAIT_KL_WEIGHT` | `0.7` | Convex weight on the trait-KL surprise component. |
| `DIRECTIVE_ASSEMBLY_SURPRISE_ANACHRONY_WEIGHT` | `0.3` | Convex weight on the plan-based anachrony component (Bae & Young 2008). Should sum to 1.0 with the KL weight. |
| `DIRECTIVE_ASSEMBLY_SURPRISE_DEFAULT_TRAIT_SALIENCE` | `0.55` | Salience for traits not in the per-trait narrative-salience table. |
| `DIRECTIVE_ASSEMBLY_SURPRISE_SOURCE_EDGE_WEIGHT` | `0.4` | Multiplier on causal edges where the focal is the source (vs target). Reflects "X did Y to Z" speaks more strongly about Z. |
| `DIRECTIVE_ASSEMBLY_SURPRISE_PRIOR_PSEUDOCOUNT` | `2.0` | Beta-Bernoulli prior pseudo-count `s`. The prior is `Beta(s·m, s·(1-m))` where `m` is the corpus baseline. |

### Salience tables (shared across scorers)

The harm-kind and trait-narrative salience tables can be replaced
wholesale via JSON env vars (Pydantic parses dict-typed `Field`s
from JSON strings):

| Variable | Default | Notes |
|---|---|---|
| `DIRECTIVE_ASSEMBLY_HARM_KIND_SALIENCE` | *(see code)* | Lazarus 1991 / OCC 1988 appraisal hierarchy. Defaults: `existential=1.00, physical=0.85, betrayal=0.75, psychological=0.70, emotional=0.65, social=0.55, epistemic=0.45, informational=0.45`. |
| `DIRECTIVE_ASSEMBLY_DEFAULT_HARM_SALIENCE` | `0.85` | Fallback salience when an event's mechanism does not resolve to any harm-kind. Defaults to `physical` (modal harm in the corpus). |
| `DIRECTIVE_ASSEMBLY_TRAIT_NARRATIVE_SALIENCE` | *(see code)* | Reagan et al. 2016 corpus arc-relevance hierarchy. Substring match against trait names. Highlights: `ambition=1.00, guilt=0.95, vengeance=0.95, despair=0.95, love=0.90, loyalty=0.85, courage=0.85`. |

> **Calibration note.** These defaults reproduce the canonical
> rise-peak-fall arcs against the 20-fixture
> [example_worlds/](../example_worlds) corpus — see
> [academic-foundations.md §3.5](academic-foundations.md#35-corpus-scale-audit)
> for the full per-scorer scale summary. Override individual
> constants only when targeting a non-canonical genre (e.g. raise
> `IRONY_AGGREGATOR_BETA` toward 1.0 for ensemble-cast tragedies
> where a single dominant blindness should overwhelm the average).

---

## 9. MCP server

| Variable | Default | Notes |
|---|---|---|
| `MCP_SKIP_AUDIT` | `true` | MCP defaults to no audit (latency-sensitive). Override per-call when needed. |
| `MCP_INGEST_FABULA_TIME_SPACING` | `1000` | Initial gap between fabula-time stamps for MCP ingest (matches the pipeline default; leaves room for flashbacks/inserts). |
| `MCP_INGEST_MAX_CORRECTION_RETRIES` | `5` | Validation-repair passes during MCP ingest. |
| `MCP_ALLOW_OPEN_MODE` | `false` | **Production must keep this false.** When true, scope checks pass when no scopes are resolved (dev/local mode only). |

Full MCP tool/resource catalogue and auth flow:
[mcp-guide.md](mcp-guide.md).

---

## 10. Pipeline

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

## 11. UI

| Variable | Default | Notes |
|---|---|---|
| `UI_HOST` | `0.0.0.0` | NiceGUI bind address. |
| `UI_PORT` | `7860` | NiceGUI port. |
| `UI_TITLE` | `Shadow Loom` | Browser tab title. |
| `UI_DARK_MODE` | `false` | Default theme. |
| `UI_RELOAD` | `false` | Hot-reload (development only). |

For the workspace tour see [ui-guide.md](ui-guide.md).

---

## 12. OAuth (UI auth)

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

## 13. Tuning recipes

**"I want to use a smaller-context model (e.g. 8K-context Llama)."**
Drop every `*_MAX_TOKENS` to ~4000, drop `EXTRACTION_MIN_CHUNK_CHARS` to
~600, and lower `EXTRACTION_CHUNK_OVERLAP_CHARS` proportionally so you
have headroom for the prompt scaffold.

**"I want OpenRouter / OpenAI everywhere."**
Set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) and set `DEFAULT_MODEL`
to `openrouter:<provider>/<model>` (or `openai:<model>`). Every stage
inherits `DEFAULT_MODEL` automatically — only set the per-stage
`*_MODEL` env vars when you want a specific stage to differ.

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

## 14. Local Ollama context size

Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`) defaults to
a **2 048-token context window** regardless of what the underlying model
advertises. On Shadow-Loom's multi-thousand-token extraction and
generation prompts this silently truncates the input, the model returns
empty / malformed JSON, and the pipeline cascades into
`JSONDecodeError`s with no obvious upstream cause.

Shadow-Loom mitigates this in three layered ways — set **all three** in
production for full belt-and-braces coverage:

1. **`OLLAMA_NUM_CTX` (default `262144`)** — every `ollama:` model call
   is constructed with
   `extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}}`. Ollama
   ≥0.6.x honours `options` on the OpenAI-compat endpoint and resizes
   the KV cache at model-load time. Older Ollama versions silently
   ignore the field, so:

2. **Server-side `OLLAMA_CONTEXT_LENGTH` env var** — export this on the
   machine that launches the Ollama server (e.g.
   `OLLAMA_CONTEXT_LENGTH=262144 ollama serve`) so every loaded model
   gets the larger window regardless of what the client requested.

3. **Modelfile `PARAMETER num_ctx`** — bake the context size into a
   custom Modelfile and `ollama create my-model -f Modelfile`. This is
   the most reliable option because it persists across server restarts
   and is independent of both environment variables and request fields.

All three approaches are documented at
[docs.ollama.com/api/openai-compatibility](https://docs.ollama.com/api/openai-compatibility)
under "Setting the context size".

> **Cloud providers are unaffected.** The `extra_body.options.num_ctx`
> field is only set on the `ollama:` provider branch in
> [`resolve_model`](../shadow_loom/settings.py). OpenAI, OpenRouter,
> Anthropic, Google, Azure, and every custom OpenAI-compatible provider
> use the standard `max_tokens` (auto-remapped to
> `max_completion_tokens` for OpenAI reasoning models by PydanticAI)
> with no Ollama-specific payload.

> **Memory caveat.** A 256K KV cache on a 35B-parameter model needs
> tens of GB of VRAM. If you're on a smaller GPU (or running an 8B
> model), drop `OLLAMA_NUM_CTX` to a value your hardware can actually
> load — `32768` is a safe ceiling for an 8B model on a single 24GB
> GPU.

---

## See also

* [`shadow_loom/settings.py`](../shadow_loom/settings.py) — the canonical source for every default value.
* [`config.env`](../config.env) — copy-pasteable environment template.
* [architecture.md](architecture.md) — how each setting flows into the 8-step pipeline.
* [pipeline-walkthrough.md](pipeline-walkthrough.md) — code-level walkthrough showing where each `*Config` is consumed.
* [academic-foundations.md](academic-foundations.md) — the literature behind the threshold defaults (Wilmot suspense, Halpern actual causality, Pearl ladder, AMWN).
* [mcp-guide.md](mcp-guide.md) — MCP-specific overrides and scope rules.
