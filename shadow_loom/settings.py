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
    "openrouter":    "https://openrouter.ai/api/v1",
    "openai":        "https://api.openai.com/v1",
    "fireworks":     "https://api.fireworks.ai/inference/v1",
    "featherless":   "https://api.featherless.ai/v1",
    "together":      "https://api.together.xyz/v1",
    "deepinfra":     "https://api.deepinfra.com/v1/openai",
    "groq":          "https://api.groq.com/openai/v1",
    "anyscale":      "https://api.endpoints.anyscale.com/v1",
    "perplexity":    "https://api.perplexity.ai",
    "huggingface":   "https://api-inference.huggingface.co/v1",
}


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
    """Return the merged provider registry (built-ins + ``SHADOW_LOOM_PROVIDERS``)."""
    merged = dict(_BUILTIN_OPENAI_COMPAT_PROVIDERS)
    merged.update(_parse_custom_providers(os.environ.get("SHADOW_LOOM_PROVIDERS", "")))
    return merged

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
            "deepinfra, groq, anyscale, perplexity, huggingface. Add custom OpenAI-compat "
            "providers via the SHADOW_LOOM_PROVIDERS env var. Strings without "
            "a recognised prefix are passed through to PydanticAI for native "
            "resolution."
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
    max_tokens: int = Field(default=128000)
    temperature: float = Field(default=0.7)
    output_retries: int = Field(default=5)


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
    max_iterations: int = Field(default=3)
    output_retries: int = Field(default=5)
    temperature: float = Field(default=0.2)
    generation_temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=32000)
    max_tokens_generation: int = Field(default=128000)
    min_foreshadowing_score: float = Field(default=0.6)
    max_affective_loss: float = Field(default=0.3)
    min_cognitive_plausibility: float = Field(default=0.7)
    max_miracle_steps: int = Field(default=0)
    ignore_spatial_blocks: bool = Field(default=False)


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
    validation_payload_max_chars: int = Field(default=600_000)
    correction_subgraph_threshold_chars: int = Field(default=400_000)
    max_concurrent_chunks: int = Field(default=12)
    per_chunk_timeout_seconds: float = Field(default=600.0)
    estimated_events_per_chunk: int = Field(default=10)
    enable_consequences_agent: bool = Field(default=True)

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
        description=(
            "Sigmoid temperature for the noisy-OR per-edge gate. Lower = "
            "sharper (closer to a hard threshold at |w*impulse| == inertia); "
            "higher = softer (more probability mass even when impulse is "
            "below inertia)."
        ),
    )
    noisy_or_threshold: float = Field(
        default=0.5,
        description=(
            "Aggregate noisy-OR probability above which the trait is "
            "considered to have shifted in the *deterministic* "
            "propagation_mode='noisy_or' path. Ignored under Monte-Carlo "
            "sampling, where the noisy-OR probability is drawn directly."
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
        default=128,
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
            "estimated_events_per_chunk": self.extraction.estimated_events_per_chunk,
            "enable_consequences_agent": self.extraction.enable_consequences_agent,
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

def resolve_model(model_str: str):
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
    """
    core = get_settings().core

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
            api_key = os.environ.get(f"{env_prefix}_API_KEY", "").strip()
            if not api_key:
                raise ValueError(
                    f"{env_prefix}_API_KEY must be set to use '{prefix_lc}:' "
                    f"models. Set it in config.env, .env, or as an environment "
                    f"variable."
                )
            base_url = os.environ.get(
                f"{env_prefix}_BASE_URL", providers[prefix_lc]
            ).strip() or providers[prefix_lc]
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
            return OpenAIChatModel(
                model_name,
                provider=OpenAIProvider(base_url=base_url, api_key=api_key),
                settings=settings,  # type: ignore[arg-type]
            )

    # Fallback: pass through for PydanticAI native model resolution.
    return model_str
