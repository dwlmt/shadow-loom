"""Environment-based configuration for the Shadow-Loom UI.

Thin adapter over the centralised ``shadow_loom.settings`` module.
All values come from PydanticSettings (env vars → config.env → defaults).
Module-level attributes are preserved for backward compatibility.
"""

from __future__ import annotations

from shadow_loom.settings import get_settings as _get_settings

_settings = _get_settings()

# ── Database ──────────────────────────────────────────────────────
DATABASE_URL: str = _settings.core.database_url

# ── Auth / OAuth ──────────────────────────────────────────────────
STORAGE_SECRET: str = _settings.oauth.resolved_storage_secret
GITHUB_CLIENT_ID: str = _settings.oauth.github_client_id
GITHUB_CLIENT_SECRET: str = _settings.oauth.github_client_secret
GOOGLE_CLIENT_ID: str = _settings.oauth.google_client_id
GOOGLE_CLIENT_SECRET: str = _settings.oauth.google_client_secret
DISCORD_CLIENT_ID: str = _settings.oauth.discord_client_id
DISCORD_CLIENT_SECRET: str = _settings.oauth.discord_client_secret
MICROSOFT_CLIENT_ID: str = _settings.oauth.microsoft_client_id
MICROSOFT_CLIENT_SECRET: str = _settings.oauth.microsoft_client_secret
OAUTH_REDIRECT_BASE: str = _settings.oauth.oauth_redirect_base

AUTH_ENABLED: bool = _settings.oauth.auth_enabled
OAUTH_PROVIDERS: list[dict] = _settings.oauth.oauth_providers
