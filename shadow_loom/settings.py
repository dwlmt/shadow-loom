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

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the OpenRouter API.",
    )
    openrouter_api_key: str = Field(
        default="",
        description="API key for OpenRouter. Required when using 'openrouter:' model prefix.",
    )
    openai_api_key: str = Field(
        default="",
        description="API key for OpenAI. Required when using 'openai:' model prefix.",
    )
    default_model: str = Field(
        default="ollama:qwen3.6:27b",
        description=(
            "Fallback PydanticAI model string when a stage-specific model "
            "is not set. Prefix determines provider: 'ollama:', 'openrouter:', "
            "'openai:', or any PydanticAI model string."
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

    model: str = Field(default="ollama:qwen3.6:27b")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    output_retries: int = Field(default=3)


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

    model: str = Field(default="ollama:qwen3.6:27b")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.1)
    output_retries: int = Field(default=3)


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

    model: str = Field(default="ollama:qwen3.6:27b", alias="AUDITOR_MODEL")
    generation_model: str = Field(default="ollama:qwen3.6:27b")
    max_iterations: int = Field(default=3)
    output_retries: int = Field(default=3)
    temperature: float = Field(default=0.2)
    generation_temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    max_tokens_generation: int = Field(default=4096)
    min_foreshadowing_score: float = Field(default=0.6)
    max_affective_loss: float = Field(default=0.3)
    min_cognitive_plausibility: float = Field(default=0.7)


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

    model: str = Field(default="ollama:qwen3.6:27b")
    chunk_strategy: Literal["act_headings", "paragraph"] = Field(default="act_headings")
    output_retries: int = Field(default=5)
    fabula_time_spacing: int = Field(default=1000)
    min_chunk_chars: int = Field(default=1500)
    chunk_overlap_chars: int = Field(default=300)
    max_correction_retries: int = Field(default=3)
    max_concurrent_chunks: int = Field(default=8)
    estimated_events_per_chunk: int = Field(default=10)


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
    ego_memory_limit: int = Field(default=5)

    @property
    def strength_multiplier(self) -> dict[str, float]:
        return {
            "weak": self.strength_weak,
            "moderate": self.strength_moderate,
            "strong": self.strength_strong,
        }


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
    ingest_fabula_time_spacing: int = Field(default=100)
    ingest_max_correction_retries: int = Field(default=1)
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
        self.mcp = MCPSettings()
        self.pipeline = PipelineSettings()
        self.ui = UISettings()
        self.oauth = OAuthSettings()

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
            "max_concurrent_chunks": self.extraction.max_concurrent_chunks,
            "estimated_events_per_chunk": self.extraction.estimated_events_per_chunk,
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
    """Resolve a model string to a PydanticAI model instance.

    Prefixes:
      ``ollama:<name>``      → OllamaModel with configured base URL
      ``openrouter:<name>``  → OpenAIChatModel via OpenRouter (OpenAI-compat)
      ``openai:<name>``      → OpenAIChatModel via OpenAI directly
      anything else          → returned as-is for PydanticAI native resolution
    """
    core = get_settings().core

    if model_str.startswith("ollama:"):
        model_name = model_str.split(":", 1)[1]
        from pydantic_ai.models.ollama import OllamaModel
        from pydantic_ai.providers.ollama import OllamaProvider
        return OllamaModel(model_name, provider=OllamaProvider(base_url=core.ollama_base_url))

    if model_str.startswith("openrouter:"):
        if not core.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set to use 'openrouter:' models. "
                "Set it in config.env, .env, or as an environment variable."
            )
        model_name = model_str.split(":", 1)[1]
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                base_url=core.openrouter_base_url,
                api_key=core.openrouter_api_key,
            ),
        )

    if model_str.startswith("openai:"):
        if not core.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set to use 'openai:' models. "
                "Set it in config.env, .env, or as an environment variable."
            )
        model_name = model_str.split(":", 1)[1]
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(api_key=core.openai_api_key),
        )

    # Fallback: pass through for PydanticAI native model resolution
    return model_str
