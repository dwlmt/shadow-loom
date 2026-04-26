"""Environment-based configuration for the Shadow-Loom UI."""

from __future__ import annotations

import os
import secrets


DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///shadow_loom.db")
STORAGE_SECRET: str = os.environ.get("STORAGE_SECRET", secrets.token_urlsafe(32))

# OAuth providers (optional — UI works without auth)
GITHUB_CLIENT_ID: str = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET: str = os.environ.get("GITHUB_CLIENT_SECRET", "")
GOOGLE_CLIENT_ID: str = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET: str = os.environ.get("GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_BASE: str = os.environ.get("OAUTH_REDIRECT_BASE", "http://localhost:7860")

# Pipeline defaults
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "ollama:qwen3.6:27b")

AUTH_ENABLED: bool = bool(GITHUB_CLIENT_ID or GOOGLE_CLIENT_ID)
