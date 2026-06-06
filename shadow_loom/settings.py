# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Centralised settings for Shadow-Loom.

All hardcoded defaults are externalised here as Pydantic Settings objects.
Values are read (in priority order) from:

  1. Explicit keyword arguments
  2. Environment variables (e.g. ``GENERATION_MODEL``)
  3. A ``.env`` file (auto-discovered) or ``config.env`` at project root
  4. The defaults defined below

Every per-module Config dataclass (``GenerationConfig``, ``AuditorConfig``,
``ExtractionConfig``, ``QueryParsingConfig``, ``PipelineConfig``) can be
constructed from these settings via the ``get_*`` helpers, so callers that
already build their own Config objects are unaffected.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =====================================================================
# OpenAI-compatible provider registry
# =====================================================================
#
# Any provider that exposes an OpenAI-compatible Chat Completions endpoint
# can be plugged in via this registry. For each entry, the resolver reads:
#
#   * ``<PREFIX>_API_KEY``    — credential (required, except for unauthenticated
#                               local servers like Ollama)
#   * ``<PREFIX>_BASE_URL``   — overrides the default base URL below
#
# Users can register additional providers without code changes via the
# ``SHADOW_LOOM_PROVIDERS`` env var, e.g.::
#
#     SHADOW_LOOM_PROVIDERS=mistral=https://api.mistral.ai/v1,xai=https://api.x.ai/v1
#
# Then use ``mistral:mistral-large-latest`` as a model string and set
# ``MISTRAL_API_KEY``.
#
_BUILTIN_OPENAI_COMPAT_PROVIDERS: dict[str, str] = {
    # ── Cloud aggregators / multi-model ─────────────────────────
    "openrouter":    "https://openrouter.ai/api/v1",
    "openai":        "https://api.openai.com/v1",
    "fireworks":     "https://api.fireworks.ai/inference/v1",
    "featherless":   "https://api.featherless.ai/v1",
    "together":      "https://api.together.ai/v1",
    "deepinfra":     "https://api.deepinfra.com/v1/openai",
    "groq":          "https://api.groq.com/openai/v1",
    "perplexity":    "https://api.perplexity.ai",
    "huggingface":   "https://router.huggingface.co/v1",
    # ── First-party frontier APIs (OpenAI-compatible endpoints) ─
    "mistral":       "https://api.mistral.ai/v1",
    "xai":           "https://api.x.ai/v1",
    "deepseek":      "https://api.deepseek.com/v1",
    "moonshot":      "https://api.moonshot.ai/v1",
    # ── Hardware-accelerated inference (custom silicon / GPU clouds) ─
    "cerebras":      "https://api.cerebras.ai/v1",       # Cerebras CS-3 wafer-scale
    "sambanova":     "https://api.sambanova.ai/v1",      # SambaNova RDU chips
    # ── Fast/cheap inference clouds ─────────────────────────────
    "nebius":        "https://api.studio.nebius.ai/v1",
    "novita":        "https://api.novita.ai/v3/openai",
    "hyperbolic":    "https://api.hyperbolic.xyz/v1",
    # ── Local OpenAI-compatible servers (no API key required) ───
    # llama.cpp:  llama-server -m <gguf> --port 8080
    # vLLM:       python -m vllm.entrypoints.openai.api_server ...
    # LM Studio:  Local Server tab (default port 1234)
    # LocalAI:    docker run -p 8080:8080 localai/localai
    # Unsloth:    Unsloth Studio / unsloth-zoo inference (vLLM-backed,
    #             default Gradio port 7860; override via UNSLOTH_BASE_URL)
    "llamacpp":      "http://localhost:8080/v1",
    "vllm":          "http://localhost:8000/v1",
    "lmstudio":      "http://localhost:1234/v1",
    "localai":       "http://localhost:8080/v1",
    "unsloth":       "http://localhost:7860/v1",
}

# Providers that run locally and don't require an API key. Users may
# still set ``<PREFIX>_API_KEY`` if their server enforces auth (e.g. a
# llama.cpp server started with --api-key), in which case it's honoured.
_LOCAL_OPENAI_COMPAT_PROVIDERS: frozenset[str] = frozenset({
    "llamacpp",
    "vllm",
    "lmstudio",
    "localai",
    "unsloth",
})


def _parse_custom_providers(spec: str) -> dict[str, str]:
    """Parse ``name=url,name2=url2`` into a dict. Silently skips bad entries."""
    out: dict[str, str] = {}
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, url = chunk.split("=", 1)
        name = name.strip().lower()
        url = url.strip()
        if name and url:
            out[name] = url
    return out


def get_openai_compat_providers() -> dict[str, str]:
    """Return the merged provider registry.

    Merge order (later wins): built-ins → ``SHADOW_LOOM_PROVIDERS`` env
    var → per-user custom providers from the active user context (see
    :func:`set_user_context`).
    """
    merged = dict(_BUILTIN_OPENAI_COMPAT_PROVIDERS)
    merged.update(_parse_custom_providers(os.environ.get("SHADOW_LOOM_PROVIDERS", "")))
    overrides = _user_overrides_var.get()
    for prov in overrides.get("custom_providers", []):
        name = (prov.get("prefix") or "").strip().lower()
        url = (prov.get("base_url") or "").strip()
        if name and url:
            merged[name] = url
    return merged


# =====================================================================
# Per-user override context (UI-driven settings take precedence over env)
# =====================================================================
#
# When a user opens the UI and signs in, ``set_user_context(user_id)``
# loads their saved model preferences (default model, per-stage model
# overrides, custom OpenAI-compatible providers) into a ContextVar that
# lives for the duration of the request / worker thread. The settings
# resolver consults this context before falling back to environment
# variables, so an empty deployment (no ``DEFAULT_MODEL`` set in env)
# becomes valid as long as every signed-in user has saved their own
# preferences via the Settings page.
#
# Schema produced by ``shadow_loom.db.get_user_model_settings``::
#
#     {
#         "default_model":          str,   # "" → fall back to env
#         "stage_models": {                 # all values "" → fall back
#             "generation":   str,
#             "auditor":      str,
#             "auditor_generation": str,
#             "extraction":   str,
#             "query_parsing": str,
#         },
#         "custom_providers": [
#             {"prefix": str, "base_url": str, "api_key": str, "is_local": bool},
#             ...
#         ],
#     }

from contextvars import ContextVar  # noqa: E402  — kept near usage for clarity

_EMPTY_OVERRIDES: dict = {
    "default_model": "",
    "stage_models": {},
    "custom_providers": [],
}

_user_overrides_var: ContextVar[dict] = ContextVar(
    "shadow_loom_user_overrides", default=_EMPTY_OVERRIDES,
)


def set_user_context(user_id: Optional[int]) -> object:
    """Activate per-user model overrides for the current context.

    Returns an opaque token suitable for ``reset_user_context(token)`` so
    callers can restore the previous state (e.g. on worker thread exit).
    Passing ``user_id=None`` is a no-op that activates the empty
    overrides dict — useful for tests.
    """
    if user_id is None:
        return _user_overrides_var.set(_EMPTY_OVERRIDES)
    try:
        # Import lazily to avoid a settings ↔ db import cycle at module
        # load time.
        from shadow_loom.db import get_user_model_settings  # noqa: WPS433
        overrides = get_user_model_settings(int(user_id))
    except Exception:  # noqa: BLE001 — never let a stale DB block the LLM call
        overrides = _EMPTY_OVERRIDES
    return _user_overrides_var.set(overrides or _EMPTY_OVERRIDES)


def reset_user_context(token: object) -> None:
    """Restore a previous user-context token returned by :func:`set_user_context`."""
    try:
        _user_overrides_var.reset(token)  # type: ignore[arg-type]
    except (LookupError, ValueError):
        # Token from a different context (e.g. thread re-use) — fall back
        # to clearing the slot rather than raising.
        _user_overrides_var.set(_EMPTY_OVERRIDES)


def get_user_overrides() -> dict:
    """Return the active per-user overrides dict (read-only)."""
    return _user_overrides_var.get()


def _user_default_model() -> str:
    """Return the active user's chosen default model, or ``""`` if unset."""
    return (_user_overrides_var.get().get("default_model") or "").strip()


def _user_stage_model(stage: str) -> str:
    """Return the active user's per-stage override, or ``""`` if unset."""
    return (_user_overrides_var.get().get("stage_models", {}).get(stage) or "").strip()


def _user_custom_provider(prefix: str) -> Optional[dict]:
    """Return the active user's custom provider entry for ``prefix``, or ``None``."""
    plc = prefix.lower()
    for prov in _user_overrides_var.get().get("custom_providers", []):
        if (prov.get("prefix") or "").lower() == plc:
            return prov
    return None


# ── Discover config.env next to this file's package root ──────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_ENV = _PROJECT_ROOT / "config.env"
_DOT_ENV = _PROJECT_ROOT / ".env"
_ENV_FILE: tuple[Path, ...] = tuple(
    p for p in (_DOT_ENV, _CONFIG_ENV) if p.exists()
)


# =====================================================================
# Core / shared
# =====================================================================

class CoreSettings(BaseSettings):
    """Database, LLM provider, and shared defaults."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="sqlite:///shadow_loom.db",
        description="SQLAlchemy database URL.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1/",
        description="Base URL for the local Ollama API (OpenAI-compat).",
    )
    tavily_api_key: str = Field(
        default="",
        description=(
            "API key for the Tavily web-research provider. Required only "
            "when ``ExtractionConfig.enable_research_agent=True`` and "
            "``research_provider='tavily'``. Off by default."
        ),
    )
    langfuse_secret_key: str = Field(
        default="",
        description="Secret key for Langfuse tracing.",
    )
    langfuse_public_key: str = Field(
        default="",
        description="Public key for Langfuse tracing.",
    )
    langfuse_base_url: str = Field(
        default="",
        description="Base URL for Langfuse API. Example: https://cloud.langfuse.com",
    )
    default_model: str = Field(
        default="ollama:qwen3.6:35b",
        description=(
            "Fallback PydanticAI model string when a stage-specific model "
            "is not set. Format is ``<provider>:<model>``. Built-in providers: "
            "ollama, openrouter, openai, fireworks, featherless, together, "
            "deepinfra, groq, anyscale, perplexity, huggingface, mistral, xai, "
            "deepseek, moonshot, cerebras, sambanova, nebius, novita, "
            "hyperbolic, and the local servers llamacpp, vllm, lmstudio, "
            "localai (no API key required). Add custom OpenAI-compat providers "
            "via the SHADOW_LOOM_PROVIDERS env var. Strings without a recognised "
            "prefix are passed through to PydanticAI for native resolution "
            "(e.g. anthropic:claude-3-5-sonnet, google-gla:gemini-1.5-pro)."
        ),
    )
    openrouter_provider_sort: str = Field(
        default="throughput",
        description=(
            "Default OpenRouter provider-routing ``sort`` strategy. Sent in "
            "the request body as ``provider: {sort: <value>}`` for every "
            "``openrouter:`` model. Accepted values: ``throughput`` (highest "
            "tokens/sec — best for long ingestion runs), ``price`` (cheapest), "
            "``latency`` (fastest first token), or empty string to disable "
            "sorting and let OpenRouter use its default price-based load "
            "balancer. See https://openrouter.ai/docs/features/provider-routing."
        ),
    )


# =====================================================================
# Generation
# =====================================================================

class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GENERATION_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = Field(
        default="",
        description=(
            "PydanticAI model string for narrative generation. Leave "
            "empty to fall back to ``CoreSettings.default_model``."
        ),
    )
    max_tokens: int = Field(default=32000)
    temperature: float = Field(default=0.7)
    output_retries: int = Field(default=5)
    # ── Scene-context trimming ────────────────────────────────────
    # Controls how much world-state data is injected into each LLM
    # prompt.  Reducing these keeps input tokens well inside GPT-class
    # context windows without losing the information that matters for
    # a single scene.  Raise them back toward the old values if you
    # switch to a model with a large context window.
    scene_context_recent_events: int = Field(default=20)
    scene_context_max_beliefs: int = Field(default=6)
    scene_context_loc_desc_chars: int = Field(default=240)
    scene_context_obj_desc_chars: int = Field(default=200)
    scene_context_utterance_chars: int = Field(default=300)
    # ── Preceding-prose cap ──────────────────────────────────────
    # ``STORY SO FAR`` is the concatenation of all prior rendered prose
    # in a session lineage.  Without a cap it grows unboundedly and
    # can easily consume thousands of tokens.  Only the tail (most
    # recent content) is kept.
    preceding_prose_max_chars: int = Field(default=6000)
    # ── Answer-agent compress limits ───────────────────────────────────
    # Controls how many entities/events are sent to the Q&A answer
    # agent (_compress_world_state). Large worlds can easily exceed
    # GPT-class context windows with the old unlimited defaults.
    answer_max_entities: int = Field(default=60)
    answer_max_events: int = Field(default=80)


# =====================================================================
# Query Parsing
# =====================================================================

class QueryParsingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUERY_PARSING_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = Field(
        default="",
        description=(
            "PydanticAI model string for query parsing. Leave empty to "
            "fall back to ``CoreSettings.default_model``."
        ),
    )
    max_tokens: int = Field(default=32000)
    temperature: float = Field(default=0.1)
    output_retries: int = Field(default=5)


# =====================================================================
# Auditor
# =====================================================================

class AuditorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUDITOR_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = Field(
        default="",
        alias="AUDITOR_MODEL",
        description=(
            "Model for the auditor. Leave empty to fall back to "
            "``CoreSettings.default_model``."
        ),
    )
    generation_model: str = Field(
        default="",
        description=(
            "Model used by the auditor's regeneration step. Leave empty "
            "to fall back to ``CoreSettings.default_model``."
        ),
    )
    max_iterations: int = Field(default=6, ge=1, le=8)
    output_retries: int = Field(default=5)
    temperature: float = Field(default=0.2)
    generation_temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=32000)
    max_tokens_generation: int = Field(default=16000)
    min_foreshadowing_score: float = Field(default=0.6)
    max_affective_loss: float = Field(default=0.3)
    min_cognitive_plausibility: float = Field(default=0.7)
    max_miracle_steps: int = Field(default=0)
    ignore_spatial_blocks: bool = Field(default=False)
    regression_retry_budget: int = Field(default=2)
    failed_open_tolerance: int = Field(default=2)
    enable_deterministic_prose_checks: bool = Field(default=True)
    pov_breach_threshold: int = Field(default=3)


# =====================================================================
# Extraction / Ingestion
# =====================================================================

class ExtractionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EXTRACTION_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = Field(
        default="",
        description=(
            "PydanticAI model string for extraction. Leave empty to fall "
            "back to ``CoreSettings.default_model``."
        ),
    )
    chunk_strategy: Literal["act_headings", "paragraph"] = Field(default="act_headings")
    output_retries: int = Field(default=5)
    fabula_time_spacing: int = Field(default=1000)
    min_chunk_chars: int = Field(default=800)
    chunk_overlap_chars: int = Field(default=300)
    max_correction_retries: int = Field(default=5)
    validation_payload_max_chars: int = Field(default=200_000)
    correction_subgraph_threshold_chars: int = Field(default=120_000)
    max_concurrent_chunks: int = Field(default=12)
    per_chunk_timeout_seconds: float = Field(default=0.0)
    per_agent_call_timeout_seconds: float = Field(default=600.0)
    estimated_events_per_chunk: int = Field(default=10)
    enable_consequences_agent: bool = Field(default=True)
    chunk_consistency_audit: bool = Field(
        default=True,
        description=(
            "When true, run a deterministic post-extraction audit on "
            "each chunk's assembled topology that flags id-validity "
            "and cross-stage parity defects (orphan trait updates, "
            "dead-then-acting actor resurrections, same-tick location "
            "conflicts, mutation_social edges with no matching "
            "RelationshipEdge reading). Defects are logged as a "
            "structured warning but do NOT fail the chunk. Cheap "
            "(deterministic; no LLM call); leave on unless you are "
            "diagnosing a noisy log."
        ),
    )

    # ------------------------------------------------------------------
    # Optional research extraction (off by default)
    # ------------------------------------------------------------------
    enable_research_agent: bool = Field(
        default=False,
        description=(
            "If True, ``run_extraction_async`` will call the configured "
            "``research_provider`` once per topic in ``research_topics`` "
            "and append distilled ``WorldFact`` records to "
            "``WorldStateV1.world_facts``. Off by default — research is an "
            "opt-in, segregated layer that never mutates Entities/Events/Edges."
        ),
    )
    research_provider: Literal["none", "tavily"] = Field(
        default="none",
        description=(
            "Which ``ResearchProvider`` to use. 'none' selects the "
            "``NullProvider`` (returns []); 'tavily' requires "
            "``CoreSettings.tavily_api_key`` and the optional "
            "``tavily-python`` dependency."
        ),
    )
    research_provider_model: str = Field(
        default="",
        description=(
            "Optional provider-specific model / search-depth identifier "
            "(e.g. Tavily 'basic' vs 'advanced'). Empty selects provider "
            "default. Hashed into the cache key."
        ),
    )
    research_max_results_per_query: int = Field(
        default=5,
        description="Cap on snippets returned per provider call.",
    )
    research_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Pre-configured topics to look up at extraction time. Topics "
            "may also be added live via the ``research_topic`` MCP tool."
        ),
    )


# =====================================================================
# Causal Physics
# =====================================================================

class CausalPhysicsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PHYSICS_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    strength_weak: float = Field(default=0.25)
    strength_moderate: float = Field(default=0.5)
    strength_strong: float = Field(default=0.75)
    mechanism_fallback_factor: float = Field(default=0.2)
    default_trait_baseline: float = Field(default=0.5)
    default_causal_force: float = Field(default=5.0)
    causal_force_scaling: float = Field(default=10.0)
    relationship_inertia_default: float = Field(default=0.3)
    ambient_force_multiplier: float = Field(default=2.0)
    inertia_epsilon: float = Field(default=0.0)
    ego_memory_limit: int = Field(default=5)

    # ------------------------------------------------------------------
    # Probabilistic propagation
    # ------------------------------------------------------------------
    propagation_mode: Literal["deterministic", "noisy_or"] = Field(
        default="noisy_or",
        description=(
            "How incoming causal impulses are aggregated and gated against "
            "trait inertia. 'deterministic' (default) keeps the legacy "
            "weighted-average + |impact| > inertia gate so existing "
            "behaviour is preserved bit-for-bit. 'noisy_or' treats each "
            "incoming edge as an independent Bernoulli attempt to overcome "
            "inertia: per-edge success probability p_i = sigmoid((|w_i * "
            "impulse_i| - inertia) / noisy_or_temperature), and the trait "
            "shifts iff a noisy-OR draw 1 - prod(1 - p_i) > 0.5 (or, in "
            "Monte-Carlo mode, with that probability)."
        ),
    )
    noisy_or_temperature: float = Field(
        default=0.25,
        gt=0.0,
        le=10.0,
        description=(
            "Sigmoid temperature for the noisy-OR per-edge gate. Lower = "
            "sharper (closer to a hard threshold at |w*impulse| == inertia); "
            "higher = softer (more probability mass even when impulse is "
            "below inertia). D5 (thirteenth-pass audit): must be > 0 (zero "
            "triggers a divide-by-zero in the sigmoid) and \u2264 10 (above "
            "that the gate is effectively uniform random and silently "
            "breaks every causal-physics regression)."
        ),
    )
    noisy_or_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Aggregate noisy-OR probability above which the trait is "
            "considered to have shifted in the *deterministic* "
            "propagation_mode='noisy_or' path. Ignored under Monte-Carlo "
            "sampling, where the noisy-OR probability is drawn directly. "
            "D5 (thirteenth-pass audit): bounded to [0, 1] because it is a "
            "probability \u2014 values outside the unit interval used to "
            "silently disable the gate (>1) or fire on every edge (<0)."
        ),
    )

    # ------------------------------------------------------------------
    # Per-edge causal-force noise (Normal model around the point estimate)
    # ------------------------------------------------------------------
    causal_force_sigma_weak: float = Field(
        default=0.30,
        description="std-dev (fraction of nominal causal_force) for evidence_strength='weak'.",
    )
    causal_force_sigma_moderate: float = Field(
        default=0.15,
        description="std-dev fraction for evidence_strength='moderate'.",
    )
    causal_force_sigma_strong: float = Field(
        default=0.05,
        description="std-dev fraction for evidence_strength='strong'.",
    )

    # ------------------------------------------------------------------
    # Monte-Carlo distributional CTF
    # ------------------------------------------------------------------
    monte_carlo_samples: int = Field(
        default=24,
        description=(
            "If >0, ``CausalPhysicsEngine.execute_distribution`` will draw "
            "this many samples by perturbing causal_force ~ Normal(force, "
            "sigma(evidence_strength)) and trait values ~ Beta(alpha, beta) "
            "with concentration kappa = 1/(1 - inertia + eps). The aggregate "
            "result is a distribution over post-propagation trait values "
            "(mean / p5 / p50 / p95) instead of a single point. With "
            "``monte_carlo_samples > 0`` the plain ``execute()`` entry "
            "point auto-routes through ``execute_distribution`` so every "
            "downstream caller (narrative_physics, directive_assembly, "
            "MCP server) inherits the Bayesian Monte-Carlo treatment "
            "without code changes. "
            "Sample-size rationale (default=128): trait values live in "
            "[0,1] so the worst-case standard deviation is sigma <= 0.5; "
            "the Monte-Carlo standard error of the posterior mean is "
            "SE = sigma / sqrt(N), giving SE <= 0.044 at N=128. The 5th "
            "and 95th empirical percentiles have asymptotic SE "
            "~ sqrt(p(1-p) / N) / f(x_p) ~ 0.02 / f(x_p) for p=0.05, "
            "which is informative for narrative-level uncertainty without "
            "the 500-2000-sample budget needed for tight tail estimation. "
            "Set to 0 to disable Monte-Carlo entirely (deterministic "
            "point-estimate execute()); raise to 500-1000 for "
            "publication-quality posterior summaries (cost scales "
            "linearly: every execute() call runs N full propagations)."
        ),
    )
    monte_carlo_seed: Optional[int] = Field(
        default=None,
        description="Optional RNG seed for reproducible Monte-Carlo runs.",
    )

    # ------------------------------------------------------------------
    # Abduction blending (Rung 3)
    # ------------------------------------------------------------------
    abduction_blend_mode: Literal["legacy", "bayesian"] = Field(
        default="bayesian",
        description=(
            "How abduction reconciles a sandbox trait value with the "
            "factual present-day evidence. 'legacy' (default): blended = "
            "old + delta * (1 - inertia). 'bayesian': posterior = "
            "(inertia * old + ev_precision * evidence) / (inertia + "
            "ev_precision), i.e. inertia is interpreted as the *precision* "
            "of the historical prior and characters with high inertia "
            "shrink toward the historical baseline rather than the present "
            "evidence."
        ),
    )
    abduction_evidence_precision: float = Field(
        default=1.0,
        description=(
            "Precision (inverse variance) of the present-day evidence in "
            "the bayesian abduction blend. Larger values let the evidence "
            "dominate the prior even for high-inertia traits."
        ),
    )

    # ------------------------------------------------------------------
    # Type-aware inertia priors (used as fallbacks where extraction is silent)
    # ------------------------------------------------------------------
    entity_trait_inertia_default: float = Field(
        default=0.5,
        description=(
            "Fallback inertia for an Entity trait when the extracted "
            "TraitVector is missing one. Characters are stickier than "
            "events: defaults sit higher than situation-level facts."
        ),
    )
    belief_inertia_default: float = Field(
        default=0.3,
        description=(
            "Fallback inertia for a Belief when extraction omits it. "
            "Beliefs flip faster than personality, so the prior is lower "
            "than entity_trait_inertia_default."
        ),
    )
    world_trait_inertia_default: float = Field(
        default=0.8,
        description=(
            "Fallback inertia for a WORLD_ trait magnitude when extraction "
            "omits it. World-level facts are the stickiest tier."
        ),
    )
    event_state_inertia_default: float = Field(
        default=0.0,
        description=(
            "Conceptual inertia of an *event* — events are discrete, "
            "either-they-happened-or-they-didn't, so they have effectively "
            "zero stickiness. Exposed as a field so the auditor / reasoning "
            "layer can quote it explicitly when contrasting event volatility "
            "with character stability."
        ),
    )
    entity_trait_baseline_drift_rate: float = Field(
        default=0.0,
        description=(
            "If >0, after each propagation step entity traits drift back "
            "toward their state_timeline baseline by (1 - inertia) * rate. "
            "Encodes 'characters return to type' so a single off-screen "
            "shock doesn't permanently rewrite a high-inertia trait. "
            "Default 0 keeps legacy behaviour."
        ),
    )
    rule3_pruning_mode: Literal["advisory", "prune"] = Field(
        default="advisory",
        description=(
            "How the ctf-calculus pre-flight should treat Rule 3 (Exclusion) "
            "verdicts. 'advisory' (default): report pruned interventions but "
            "still execute the do-surgery — d-separation on the latent-free "
            "AMWN can spuriously mark an intervention vacuous when the LLM "
            "extraction missed a confounder. 'prune': drop the interventions "
            "before simulation. Use 'prune' only when the extracted causal "
            "topology is known to be confounder-complete."
        ),
    )
    allow_unobserved_confounders: bool = Field(
        default=False,
        description=(
            "Safety-net only. When True, the AMWN builder injects a "
            "synthetic ``U_<a>__<b>`` latent parent for *every* pair of "
            "nodes that share an observed cause, so d-separation refuses "
            "to mark the siblings independent. This is combinatorial and "
            "will collapse Rule 2 / Rule 3 reasoning to 'everything is "
            "dependent on everything' on richly-extracted graphs — use "
            "only as a debugging / research toggle when extraction is "
            "known to under-emit named latents.\n\n"
            "Preferred path: the extraction prompts "
            "(``ontology_world_traits.md`` rule 7, "
            "``physics_extraction.md`` rule 16) elicit named latent "
            "forces — fate, prophecy, ambient ideology, offstage war, "
            "family curse — as first-class ``WORLD_*`` traits and wire "
            "them as explicit ``chain_reaction`` causes of the events "
            "they jointly produce. With confounders modelled as "
            "observed ``WORLD_*`` parents the AMWN already routes the "
            "shared dependency correctly without needing this flag."
        ),
    )

    intelligibility_threshold: float = Field(
        default=0.3,
        description=(
            "Per-recipient channel intelligibility below which a belief "
            "acquired through that channel is considered epistemically "
            "invalid for the holder. Used in abduction belief back-prop "
            "(skip the belief) and in directive assembly's hidden-channel "
            "leak risk. Range 0.0 (anything goes) to 1.0 (only fully "
            "intelligible channels carry beliefs)."
        ),
    )

    max_ingest_words: int = Field(
        default=10_000,
        description=(
            "Maximum number of whitespace-separated tokens accepted by any "
            "ingestion entry point (UI textarea, sample loader, MCP "
            "``ingest`` / ``write`` tools, chat input). Shadow-Loom is "
            "designed for short summaries, synopses, and scenario sketches; "
            "long inputs make the per-chunk LLM passes prohibitively slow "
            "and produce graphs too dense for interactive counterfactual "
            "exploration. Enforced uniformly so the UI and MCP tools "
            "cannot disagree on what counts as oversized."
        ),
    )

    @property
    def strength_multiplier(self) -> dict[str, float]:
        return {
            "weak": self.strength_weak,
            "moderate": self.strength_moderate,
            "strong": self.strength_strong,
        }


# =====================================================================
# Directive Assembly (affective scorers: mystery, irony, suspense, surprise)
# =====================================================================

class DirectiveAssemblySettings(BaseSettings):
    """Tunables for the four structural-affect scorers.

    All defaults match the values used in the published paper
    (``paper/shadow_loom.tex``) and the academic-foundations doc
    (``docs/academic-foundations.md``). Override via env vars
    prefixed ``DIRECTIVE_ASSEMBLY_`` (e.g.
    ``DIRECTIVE_ASSEMBLY_MYSTERY_PATH_DECAY_DEPTH=6``).
    """

    model_config = SettingsConfigDict(
        env_prefix="DIRECTIVE_ASSEMBLY_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Mystery (Sternberg curiosity gap) ──────────────────────────
    mystery_path_decay_depth: int = Field(
        default=4,
        description=(
            "Reverse-causal traversal depth cap (Trabasso & Sperry "
            "1985 4-hop traceability). Ancestors beyond this depth "
            "are dropped from the gap aggregation."
        ),
    )
    mystery_proximity_tau_syuzhet: float = Field(
        default=8.0,
        description=(
            "Curiosity-proximity decay constant in syuzhet-index "
            "units (Iser 1976 reader-gap recency). Recent reveals "
            "carry more curiosity weight than distant ones."
        ),
    )

    # ── Dramatic Irony (knowledge asymmetry) ───────────────────────
    irony_surface_k: float = Field(
        default=1.0,
        description="Saturation constant K in the gap fraction denominator.",
    )
    irony_false_belief_mult: float = Field(
        default=1.5,
        description=(
            "Multiplicative boost when the focal holds a provenance-"
            "valid belief about the gap event's actor (Iago→Othello, "
            "Jacqueline→Linnet pattern). Pfister 1988 / Cabanas 2024."
        ),
    )
    irony_action_alpha: float = Field(
        default=0.15,
        description=(
            "Per-actor-event linear term in the focal-prominence "
            "weight a_c = min(cap, 1 + α · #actor-events)."
        ),
    )
    irony_action_weight_cap: float = Field(
        default=3.0,
        description="Upper cap on the focal-prominence weight a_c.",
    )
    irony_aggregator_beta: float = Field(
        default=0.6,
        description=(
            "Convex aggregator weight: β·max + (1-β)·mean across "
            "focal characters. β closer to 1 weights the dominant "
            "character; β closer to 0 weights the ensemble."
        ),
    )
    irony_proximity_tau_syuzhet: float = Field(
        default=6.0,
        description=(
            "Closure-proximity decay constant: how sharply the gap "
            "discharges as the syuzhet approaches the moment the "
            "focal walks into the truth-revealing scene."
        ),
    )
    irony_proximity_floor: float = Field(
        default=0.4,
        description=(
            "Minimum closure-proximity weight (events too far from "
            "any closure event still contribute at least this much)."
        ),
    )

    # ── Suspense (hope/threat ledger) ──────────────────────────────
    suspense_stakes_k: float = Field(
        default=2.0,
        description="Saturation constant K in the stakes denominator.",
    )
    suspense_proximity_tau_fabula_gaps: float = Field(
        default=6.0,
        description="Fabula-gap proximity decay (closer threats feel sharper).",
    )
    suspense_proximity_tau_spatial: float = Field(
        default=4.0,
        description="Spatial-distance proximity decay (in location hops).",
    )
    suspense_persistence_alpha: float = Field(
        default=0.10,
        description="Per-revealed-edge persistence amplifier.",
    )
    suspense_persistence_cap: float = Field(
        default=1.5,
        description="Upper cap on the cumulative persistence multiplier.",
    )
    suspense_hostile_affinity: float = Field(
        default=-0.2,
        description="Affinity threshold below which a relationship is hostile.",
    )
    suspense_ally_affinity: float = Field(
        default=0.2,
        description="Affinity threshold above which a relationship is allied.",
    )

    # ── Surprise (Beta-Bernoulli + anachrony) ──────────────────────
    surprise_trait_kl_weight: float = Field(
        default=0.7,
        description="Convex weight on the trait-KL surprise component.",
    )
    surprise_anachrony_weight: float = Field(
        default=0.3,
        description=(
            "Convex weight on the plan-based anachrony component "
            "(Bae & Young 2008). Must sum to 1.0 with trait-KL weight."
        ),
    )
    surprise_default_trait_salience: float = Field(
        default=0.55,
        description=(
            "Salience for traits not in the per-trait narrative-"
            "salience table. Median weight over corpus arc traits."
        ),
    )
    surprise_source_edge_weight: float = Field(
        default=0.4,
        description=(
            "Multiplier on causal edges where the focal is the "
            "source (vs target). Acting on a trait reinforces it but "
            "less than being on the receiving end."
        ),
    )
    surprise_prior_pseudocount: float = Field(
        default=2.0,
        description=(
            "Beta-Bernoulli prior pseudo-count strength s. The prior "
            "is Beta(s·m, s·(1-m)) where m is the corpus baseline."
        ),
    )

    # ── Harm-kind salience (shared by mystery, irony, suspense) ───
    harm_kind_salience: dict[str, float] = Field(
        default_factory=lambda: {
            "existential": 1.00,
            "physical": 0.85,
            "betrayal": 0.75,
            "psychological": 0.70,
            "emotional": 0.65,
            "social": 0.55,
            "epistemic": 0.45,
            "informational": 0.45,
        },
        description=(
            "Salience weights per harm-kind (Lazarus 1991 core "
            "relational themes; OCC 1988 prospect-based emotions). "
            "Override the whole dict to retune the appraisal hierarchy."
        ),
    )
    default_harm_salience: float = Field(
        default=0.85,
        description=(
            "Fallback salience for events whose mechanism does not "
            "resolve to any harm-kind. Defaults to physical (modal "
            "harm kind in the example corpus)."
        ),
    )

    # ── Trait narrative salience (used by Surprise) ──────────────
    trait_narrative_salience: dict[str, float] = Field(
        default_factory=lambda: {
            "ambition": 1.00, "guilt": 0.95, "vengeance": 0.95,
            "despair": 0.95, "love": 0.90, "loyalty": 0.85,
            "courage": 0.85, "betrayal": 0.85, "honesty": 0.80,
            "morality": 0.80, "rage": 0.75, "fear": 0.75,
            "trust": 0.70, "patience": 0.60, "wisdom": 0.60,
            "pride": 0.60, "compassion": 0.55,
            "literacy": 0.30, "fitness": 0.30, "wealth": 0.30,
            "health": 0.40,
        },
        description=(
            "Per-trait narrative-arc salience (Reagan et al. 2016 "
            "corpus arc analysis). Keys are checked case-insensitively "
            "as substrings against trait names."
        ),
    )

    # ── R19-L7: scorer tunables previously hardcoded in
    # ``shadow_loom/affective_scorers.py``. Surfaced here so the four
    # structural-affect scorers respect the same config plane as the
    # mystery / irony / suspense / surprise parameters above.
    scorer_recent_window: int = Field(
        default=2000,
        description=(
            "Fabula-time window (in ticks) used by suspense/surprise "
            "scorers to bound 'recent' propositions when fabula_time "
            "is supplied. Should be calibrated against world fabula "
            "cadence; 2000 ticks ~= one act in the example corpus."
        ),
    )
    scorer_surprise_flip_norm: float = Field(
        default=5.0,
        description=(
            "Normalisation denominator for surprise flip count: "
            "``min(flips / surprise_flip_norm, 1.0)``. Lower values "
            "saturate the scorer faster."
        ),
    )
    scorer_min_belief_confidence: float = Field(
        default=0.6,
        description=(
            "Minimum belief.confidence required for a belief to "
            "count toward dramatic irony / suspense knowledge-asymmetry "
            "calculations. Filters out tentative / speculative beliefs."
        ),
    )


# =====================================================================
# MCP Server
# =====================================================================

class MCPSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    skip_audit: bool = Field(default=True)
    ingest_fabula_time_spacing: int = Field(default=1000)
    ingest_max_correction_retries: int = Field(default=5)
    # When True, scope checks pass when no scopes are resolved (dev/local mode).
    # In production this MUST stay False so that misconfigured auth fails closed.
    allow_open_mode: bool = Field(default=False)


# =====================================================================
# Pipeline
# =====================================================================

class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    use_causal_engine: bool = Field(default=True)
    skip_audit: bool = Field(default=False)
    skip_reextraction: bool = Field(default=False)
    max_snapshots: int = Field(default=10)


# =====================================================================
# UI
# =====================================================================

class UISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UI_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7860)
    title: str = Field(default="Shadow Loom")
    dark_mode: bool = Field(default=False)
    reload: bool = Field(default=False)
    # AGPLv3 § 13: hosted instances must offer users the corresponding
    # source code. The footer link points here. Operators running a
    # modified build MUST update this URL to point at the modified
    # source for compliance.
    source_url: str = Field(default="https://github.com/dwlmt/shadow-loom")

    # Round-8 audit (UI-P2-02): per-request soft cap on chat / command-bar
    # query length. The pipeline accepts arbitrary input but a >N-character
    # query usually indicates the user pasted prose into the query box
    # (which should go through ``write`` instead). Setting to ``0`` disables
    # the preflight guard entirely.
    max_query_chars: int = Field(
        default=2000,
        ge=0,
        description=(
            "Soft cap on a single chat/command-bar query length. "
            "Set to 0 to disable the preflight guard."
        ),
    )

    @model_validator(mode="after")
    def _honour_platform_port(self) -> "UISettings":
        """Allow the unprefixed ``PORT`` env var (Railway, Heroku, Cloud
        Run, Fly.io …) to override ``UI_PORT`` so the same image runs on
        every PaaS without a per-host wrapper."""
        import os

        platform_port = os.environ.get("PORT")
        if platform_port:
            try:
                self.port = int(platform_port)
            except ValueError:
                pass
        return self


class OAuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    storage_secret: str = Field(default="")
    oauth_redirect_base: str = Field(default="http://localhost:7860")
    # When true, the UI middleware refuses to serve protected routes
    # unless at least one OAuth provider is configured. Without this
    # flag, an accidental misconfiguration (provider env vars missing
    # in production) silently flips ``auth_enabled`` to False and
    # opens every route to anonymous traffic. Set
    # ``SHADOW_LOOM_OAUTH__AUTH_REQUIRED=true`` in non-dev environments.
    auth_required: bool = Field(default=False)
    github_client_id: str = Field(default="")
    github_client_secret: str = Field(default="")
    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    discord_client_id: str = Field(default="")
    discord_client_secret: str = Field(default="")
    microsoft_client_id: str = Field(default="")
    microsoft_client_secret: str = Field(default="")
    # Sign in with Apple (OIDC). Two configuration paths:
    #   (a) supply a pre-minted ES256 JWT in apple_client_secret, OR
    #   (b) supply apple_team_id + apple_key_id + apple_private_key and
    #       Shadow-Loom will mint and refresh the JWT automatically.
    apple_client_id: str = Field(default="")
    apple_client_secret: str = Field(default="")
    apple_team_id: str = Field(default="")
    apple_key_id: str = Field(default="")
    apple_private_key: str = Field(default="")

    @property
    def resolved_storage_secret(self) -> str:
        """Return the configured secret, or generate one if blank."""
        return self.storage_secret or secrets.token_urlsafe(32)

    @property
    def auth_enabled(self) -> bool:
        return bool(
            self.github_client_id
            or self.google_client_id
            or self.discord_client_id
            or self.microsoft_client_id
            or self.apple_client_id
        )

    @property
    def oauth_providers(self) -> list[dict]:
        providers: list[dict] = []
        if self.github_client_id:
            providers.append({"name": "github", "label": "GitHub", "icon": "code"})
        if self.google_client_id:
            providers.append({"name": "google", "label": "Google", "icon": "mail"})
        if self.discord_client_id:
            providers.append({"name": "discord", "label": "Discord", "icon": "forum"})
        if self.microsoft_client_id:
            providers.append({"name": "microsoft", "label": "Microsoft", "icon": "window"})
        if self.apple_client_id:
            providers.append({"name": "apple", "label": "Apple", "icon": "apple"})
        return providers


# =====================================================================
# Singleton accessor — cached per process
# =====================================================================

@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Return the global settings singleton (constructed once)."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached ``Settings`` so the next ``get_settings()`` re-reads env.

    Round-13 R13-06: ``get_settings`` is memoised for the whole
    process lifetime, which is fine in production but causes silent
    stale-config bugs in tests / hot-reload paths that mutate
    ``os.environ`` after import. Operators (and tests) can now call
    this to invalidate the cache deterministically. We also clear the
    Fernet cache because its derived key is gated by the same env
    variable.
    """
    get_settings.cache_clear()
    try:
        from shadow_loom.db import reset_fernet_cache
        reset_fernet_cache()
    except Exception:
        # db module may not be importable in every test context.
        pass


class Settings:
    """Aggregated facade over all setting groups.

    Constructed once per process (via ``get_settings()``) and cached.
    Individual groups can also be instantiated directly if needed.
    """

    def __init__(self) -> None:
        self.core = CoreSettings()
        self.generation = GenerationSettings()
        self.query_parsing = QueryParsingSettings()
        self.auditor = AuditorSettings()
        self.extraction = ExtractionSettings()
        self.physics = CausalPhysicsSettings()
        self.directive_assembly = DirectiveAssemblySettings()
        self.mcp = MCPSettings()
        self.pipeline = PipelineSettings()
        self.ui = UISettings()
        self.oauth = OAuthSettings()

        # Fall back to core.default_model when a stage-specific model
        # is not explicitly configured. This lets users set a single
        # DEFAULT_MODEL (e.g. ``openrouter:qwen/qwen3-...``) and have
        # every pipeline stage inherit it without enumerating each
        # ``*_MODEL`` env var.
        fallback = self.core.default_model
        if not self.generation.model:
            self.generation.model = fallback
        if not self.query_parsing.model:
            self.query_parsing.model = fallback
        if not self.auditor.model:
            self.auditor.model = fallback
        if not self.auditor.generation_model:
            self.auditor.generation_model = fallback
        if not self.extraction.model:
            self.extraction.model = fallback

    # ── Convenience builders for per-module Config objects ──────────

    def generation_config(self) -> dict:
        """Return kwargs suitable for ``GenerationConfig(**...)``."""
        return {
            "model": self.generation.model,
            "output_retries": self.generation.output_retries,
            "max_tokens": self.generation.max_tokens,
            "temperature": self.generation.temperature,
            "scene_context_recent_events": self.generation.scene_context_recent_events,
            "scene_context_max_beliefs": self.generation.scene_context_max_beliefs,
            "scene_context_loc_desc_chars": self.generation.scene_context_loc_desc_chars,
            "scene_context_obj_desc_chars": self.generation.scene_context_obj_desc_chars,
            "scene_context_utterance_chars": self.generation.scene_context_utterance_chars,
            "preceding_prose_max_chars": self.generation.preceding_prose_max_chars,
            "answer_max_entities": self.generation.answer_max_entities,
            "answer_max_events": self.generation.answer_max_events,
        }

    def query_parsing_config(self) -> dict:
        """Return kwargs suitable for ``QueryParsingConfig(**...)``."""
        return {
            "model": self.query_parsing.model,
            "output_retries": self.query_parsing.output_retries,
            "max_tokens": self.query_parsing.max_tokens,
            "temperature": self.query_parsing.temperature,
        }

    def auditor_config(self) -> dict:
        """Return kwargs suitable for ``AuditorConfig(**...)``."""
        return {
            "auditor_model": self.auditor.model,
            "generation_model": self.auditor.generation_model,
            "max_iterations": self.auditor.max_iterations,
            "output_retries": self.auditor.output_retries,
            "auditor_temperature": self.auditor.temperature,
            "generation_temperature": self.auditor.generation_temperature,
            "max_tokens_audit": self.auditor.max_tokens,
            "max_tokens_generation": self.auditor.max_tokens_generation,
            "min_foreshadowing_score": self.auditor.min_foreshadowing_score,
            "max_affective_loss": self.auditor.max_affective_loss,
            "min_cognitive_plausibility": self.auditor.min_cognitive_plausibility,
            "max_miracle_steps": self.auditor.max_miracle_steps,
            "ignore_spatial_blocks": self.auditor.ignore_spatial_blocks,
            "regression_retry_budget": self.auditor.regression_retry_budget,
            "failed_open_tolerance": self.auditor.failed_open_tolerance,
            "enable_deterministic_prose_checks": (
                self.auditor.enable_deterministic_prose_checks
            ),
            "pov_breach_threshold": self.auditor.pov_breach_threshold,
        }

    def extraction_config(self) -> dict:
        """Return kwargs suitable for ``ExtractionConfig(**...)``."""
        return {
            "model": self.extraction.model,
            "chunk_strategy": self.extraction.chunk_strategy,
            "output_retries": self.extraction.output_retries,
            "fabula_time_spacing": self.extraction.fabula_time_spacing,
            "min_chunk_chars": self.extraction.min_chunk_chars,
            "chunk_overlap_chars": self.extraction.chunk_overlap_chars,
            "max_correction_retries": self.extraction.max_correction_retries,
            "validation_payload_max_chars": self.extraction.validation_payload_max_chars,
            "correction_subgraph_threshold_chars": self.extraction.correction_subgraph_threshold_chars,
            "max_concurrent_chunks": self.extraction.max_concurrent_chunks,
            "per_chunk_timeout_seconds": self.extraction.per_chunk_timeout_seconds,
            "per_agent_call_timeout_seconds": self.extraction.per_agent_call_timeout_seconds,
            "estimated_events_per_chunk": self.extraction.estimated_events_per_chunk,
            "enable_consequences_agent": self.extraction.enable_consequences_agent,
            "chunk_consistency_audit": self.extraction.chunk_consistency_audit,
            "enable_research_agent": self.extraction.enable_research_agent,
            "research_provider": self.extraction.research_provider,
            "research_provider_model": self.extraction.research_provider_model,
            "research_max_results_per_query": self.extraction.research_max_results_per_query,
            "research_topics": list(self.extraction.research_topics),
        }

    def pipeline_config_kwargs(self) -> dict:
        """Return kwargs suitable for ``PipelineConfig(**...)``."""
        return {
            "use_causal_engine": self.pipeline.use_causal_engine,
            "skip_audit": self.pipeline.skip_audit,
            "skip_reextraction": self.pipeline.skip_reextraction,
            "max_snapshots": self.pipeline.max_snapshots,
        }


# =====================================================================
# Shared model resolver
# =====================================================================

class _MergedSystemPromptsModel:
    """Thin mixin/wrapper that merges consecutive leading system messages.

    Some OpenAI-compatible providers (e.g. Parasail serving qwen models
    via OpenRouter) enforce the invariant that there must be *exactly one*
    system message and it must be the very first message.  PydanticAI
    emits one ``{"role": "system"}`` entry per static ``system_prompt=``
    argument **plus** one per ``@agent.system_prompt`` decorator, so a
    typical ingestion agent sends two system messages back-to-back.

    This subclass post-processes the mapped OpenAI message list and
    collapses all leading ``role="system"`` entries into a single
    message (joining their content with ``"\\n\\n"``).  Providers that
    already accept multiple system messages are unaffected in practice
    because the combined text is semantically identical.
    """

    async def _map_messages(self, messages, model_request_parameters):  # type: ignore[override]
        from pydantic_ai.models.openai import OpenAIChatModel
        openai_messages = await OpenAIChatModel._map_messages(  # type: ignore[arg-type]
            self, messages, model_request_parameters
        )
        # Separate leading system messages from the rest
        system_contents: list[str] = []
        rest: list = []
        for msg in openai_messages:
            if not rest and msg.get("role") == "system":
                content = msg.get("content", "")
                system_contents.append(content if isinstance(content, str) else str(content))
            else:
                rest.append(msg)
        if len(system_contents) <= 1:
            return openai_messages  # nothing to merge
        return [{"role": "system", "content": "\n\n".join(system_contents)}] + rest

    async def request(self, *args, **kwargs):  # type: ignore[override]
        """Retry transient ``UnexpectedModelBehavior`` / ``ModelHTTPError``.

        OpenRouter occasionally returns HTTP 200 with an upstream-error
        body that has ``{id, choices, model, object}`` all set to
        ``None`` (the provider crashed mid-completion). The OpenAI SDK
        parses this as a malformed ``ChatCompletion`` and pydantic-ai
        raises ``UnexpectedModelBehavior`` — fatal by default,
        because pydantic-ai's retry logic only catches ``ModelRetry``.

        We retry such transients up to ``_PROVIDER_RETRY_ATTEMPTS``
        times with a short backoff. Genuine schema errors and 4xx
        client errors (``ModelHTTPError`` with status < 500) are
        re-raised after the first attempt — they won't get better.

        **Provider-routing fallback.** OpenRouter model names may carry
        a routing-strategy suffix such as ``:nitro`` (high-throughput,
        single high-rate provider), ``:floor`` (cheapest provider) or
        ``:online`` (web-search augmented). When the chosen provider
        crashes, every retry hits the same dead provider and times out
        the chunk. After the first half of the retry budget has been
        burned on a malformed-completion stream, we strip the routing
        suffix on the remaining attempts so OpenRouter falls back to
        its full provider list (any healthy provider may answer).
        ``self._model_name`` is restored afterwards so the next call
        from the agent loop starts on the original routing strategy.
        """
        import asyncio
        import logging
        from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
        from pydantic_ai.models.openai import OpenAIChatModel

        log = logging.getLogger(__name__)
        original_model_name = getattr(self, "_model_name", None)
        # OpenRouter's documented per-model routing suffixes — when one
        # of these is present we can safely fall back to the bare
        # model id on later retries to escape a sticky bad provider.
        # See https://openrouter.ai/docs/features/provider-routing.
        _ROUTING_SUFFIXES = (":nitro", ":floor", ":online")
        # Drop the suffix on the very next attempt after the first
        # malformed-completion / 5xx \u2014 a stuck provider rarely
        # recovers within a single backoff window, and OpenRouter's
        # bare-id route shops every healthy provider for the model.
        fallback_after = 1
        last_exc: Exception | None = None
        try:
            for attempt in range(_PROVIDER_RETRY_ATTEMPTS):
                # On later attempts, drop a provider-routing suffix so
                # OpenRouter re-shops the request across all providers
                # for the bare model id.
                if (
                    attempt >= fallback_after
                    and original_model_name
                    and isinstance(original_model_name, str)
                    and any(original_model_name.endswith(s) for s in _ROUTING_SUFFIXES)
                ):
                    bare = original_model_name
                    for suffix in _ROUTING_SUFFIXES:
                        if bare.endswith(suffix):
                            bare = bare[: -len(suffix)]
                            break
                    if getattr(self, "_model_name", None) != bare:
                        log.warning(
                            "[Provider Retry] Dropping routing suffix on %s "
                            "\u2014 falling back to bare %s for retry %d/%d.",
                            original_model_name, bare,
                            attempt + 1, _PROVIDER_RETRY_ATTEMPTS,
                        )
                        self._model_name = bare  # type: ignore[attr-defined]
                try:
                    return await OpenAIChatModel.request(self, *args, **kwargs)
                except UnexpectedModelBehavior as exc:
                    last_exc = exc
                    if attempt + 1 >= _PROVIDER_RETRY_ATTEMPTS:
                        raise
                    log.warning(
                        "[Provider Retry] Malformed completion from %s "
                        "(attempt %d/%d): %s",
                        getattr(self, "model_name", "<unknown>"),
                        attempt + 1, _PROVIDER_RETRY_ATTEMPTS, exc,
                    )
                except ModelHTTPError as exc:
                    # Only retry on 5xx / 429; 4xx client errors won't recover.
                    status = getattr(exc, "status_code", None) or 0
                    retryable = status >= 500 or status == 429
                    if not retryable or attempt + 1 >= _PROVIDER_RETRY_ATTEMPTS:
                        raise
                    last_exc = exc
                    log.warning(
                        "[Provider Retry] HTTP %d from %s (attempt %d/%d): %s",
                        status, getattr(self, "model_name", "<unknown>"),
                        attempt + 1, _PROVIDER_RETRY_ATTEMPTS, exc,
                    )
                await asyncio.sleep(_PROVIDER_RETRY_BACKOFF_S * (attempt + 1))
            # Unreachable in practice — the loop either returns or re-raises.
            assert last_exc is not None
            raise last_exc
        finally:
            # Restore the original routing strategy so subsequent calls
            # from the agent loop start fresh on the user's chosen tier.
            if (
                original_model_name is not None
                and getattr(self, "_model_name", None) != original_model_name
            ):
                self._model_name = original_model_name  # type: ignore[attr-defined]


# Number of times to retry a transient malformed response or 5xx HTTP
# error from an OpenAI-compatible provider before giving up. Five
# in-loop retries (6 attempts total) absorbs the brief Parasail / Together
# outages we have seen in practice without masking persistent issues —
# qwen3.6-35b-a3b:nitro on Parasail can return null-body completions for
# several consecutive requests when its upstream is saturated.
_PROVIDER_RETRY_ATTEMPTS = 6
_PROVIDER_RETRY_BACKOFF_S = 1.5


def resolve_model(model_str: str, *, stage: Optional[str] = None):
    """Resolve a ``<provider>:<model>`` string to a PydanticAI model instance.

    Special-cased prefixes:
      ``ollama:<name>``      → :class:`OllamaModel` against the configured
                                local Ollama base URL (no API key required).

    OpenAI-compatible prefixes (any entry in
    :func:`get_openai_compat_providers`):
      ``<prefix>:<name>``    → :class:`OpenAIChatModel` via
                                :class:`OpenAIProvider`. The API key is read
                                from the ``<PREFIX>_API_KEY`` env var and the
                                base URL from ``<PREFIX>_BASE_URL`` (falling
                                back to the registered default).

    Anything else is returned as-is for PydanticAI's native resolver
    (e.g. ``anthropic:claude-3-sonnet``, ``google-gla:gemini-1.5-pro``).

    Per-user overrides (set via :func:`set_user_context`) take precedence
    over both env vars and the built-in registry. When the prefix matches
    a user-configured custom provider, its ``base_url`` and ``api_key``
    are used in place of the registry / env values.

    ``stage`` is an optional stage label (``"generation"``, ``"auditor"``,
    ``"auditor_generation"``, ``"extraction"``, ``"query_parsing"``); when
    set and the active user has a per-stage override, that wins over both
    the incoming ``model_str`` and the user's default.
    """
    core = get_settings().core

    # ── Empty model string → fall back chain ─────────────────────────
    # 1. user-context per-stage override
    # 2. user-context default
    # 3. env-derived core.default_model
    user_stage = _user_stage_model(stage) if stage else ""
    user_default = _user_default_model()
    env_default = (core.default_model or "").strip()
    incoming = (model_str or "").strip()
    if user_stage:
        # Per-stage user override wins outright.
        model_str = user_stage
    elif not incoming:
        fallback = user_default or env_default
        if not fallback:
            raise ValueError(
                "No model configured. Set DEFAULT_MODEL in the environment "
                "or save a default in Settings → Models & Providers."
            )
        model_str = fallback
    elif user_default and incoming == env_default:
        # Stage config inherited the env default but the user has saved
        # their own override — honour the user's choice.
        model_str = user_default
    else:
        model_str = incoming

    if model_str.startswith("ollama:"):
        model_name = model_str.split(":", 1)[1]
        from pydantic_ai.models.ollama import OllamaModel
        from pydantic_ai.providers.ollama import OllamaProvider
        return OllamaModel(model_name, provider=OllamaProvider(base_url=core.ollama_base_url))

    providers = get_openai_compat_providers()
    if ":" in model_str:
        prefix, model_name = model_str.split(":", 1)
        prefix_lc = prefix.lower()
        if prefix_lc in providers:
            env_prefix = prefix_lc.upper()
            user_prov = _user_custom_provider(prefix_lc)
            # User-configured credentials win over env vars.
            api_key = (
                (user_prov or {}).get("api_key", "").strip()
                or os.environ.get(f"{env_prefix}_API_KEY", "").strip()
            )
            is_local = (
                prefix_lc in _LOCAL_OPENAI_COMPAT_PROVIDERS
                or bool((user_prov or {}).get("is_local"))
            )
            if not api_key:
                if is_local:
                    # Local OpenAI-compatible servers (llama.cpp, vLLM,
                    # LM Studio, LocalAI, Unsloth Studio) don't require
                    # auth out of the box. Most still demand a non-empty
                    # Authorization header, so we send a placeholder.
                    api_key = "sk-no-key-required"
                else:
                    raise ValueError(
                        f"{env_prefix}_API_KEY must be set to use '{prefix_lc}:' "
                        f"models. Set it in config.env, .env, an environment "
                        f"variable, or Settings → Models & Providers."
                    )
            base_url = (
                (user_prov or {}).get("base_url", "").strip()
                or os.environ.get(f"{env_prefix}_BASE_URL", "").strip()
                or providers[prefix_lc]
            )
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            # OpenRouter-specific: forward the configured provider-routing
            # ``sort`` strategy as ``extra_body`` so every request includes
            # ``{"provider": {"sort": <value>}}`` in its body. See
            # https://openrouter.ai/docs/features/provider-routing.
            settings: dict | None = None
            if prefix_lc == "openrouter":
                sort = (core.openrouter_provider_sort or "").strip().lower()
                if sort:
                    settings = {"extra_body": {"provider": {"sort": sort}}}

            # Build a subclass that merges multiple leading system messages
            # into one (required by strict providers such as Parasail/qwen).
            _MergedModel = type(
                "_MergedOpenAIChatModel",
                (_MergedSystemPromptsModel, OpenAIChatModel),
                {},
            )
            return _MergedModel(
                model_name,
                provider=OpenAIProvider(base_url=base_url, api_key=api_key),
                settings=settings,  # type: ignore[arg-type]
            )

    # Fallback: pass through for PydanticAI native model resolution.
    return model_str
