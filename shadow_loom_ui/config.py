# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

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
APPLE_CLIENT_ID: str = _settings.oauth.apple_client_id
APPLE_CLIENT_SECRET: str = _settings.oauth.apple_client_secret
APPLE_TEAM_ID: str = _settings.oauth.apple_team_id
APPLE_KEY_ID: str = _settings.oauth.apple_key_id
APPLE_PRIVATE_KEY: str = _settings.oauth.apple_private_key
OAUTH_REDIRECT_BASE: str = _settings.oauth.oauth_redirect_base

AUTH_ENABLED: bool = _settings.oauth.auth_enabled
AUTH_REQUIRED: bool = _settings.oauth.auth_required
OAUTH_PROVIDERS: list[dict] = _settings.oauth.oauth_providers

# Session-cookie hardening passed to ui.run(session_middleware_kwargs=...).
# ``same_site=lax`` is required so the OAuth redirect (a top-level
# navigation) still carries the session cookie. ``https_only`` marks the
# cookie Secure so it is never sent over plain HTTP — but that would break
# local HTTP dev, so it is only forced in hosted mode (AUTH_REQUIRED).
SESSION_MIDDLEWARE_KWARGS: dict = {
    "same_site": "lax",
    "https_only": bool(AUTH_REQUIRED),
}

# Hard fail-closed: if AUTH_REQUIRED is set but no provider is
# configured, refuse to start rather than silently opening every
# route to anonymous traffic (round-3 audit).
if AUTH_REQUIRED and not AUTH_ENABLED:
    raise RuntimeError(
        "AUTH_REQUIRED is set but no OAuth provider "
        "is configured (github/google/discord/microsoft/apple). Refusing "
        "to start; configure a provider or unset auth_required."
    )

# ── Input limits ──────────────────────────────────────────────────
# Maximum number of whitespace-separated tokens accepted by the
# ingestion textarea, file upload, sample loader, and chat input.
# Shadow-Loom is designed for short summaries, synopses, and scenario
# sketches \u2014 not full novels or shooting scripts. Long inputs make
# the per-chunk LLM passes prohibitively slow and produce graphs that
# are too dense for interactive counterfactual exploration.
#
# Sourced from ``shadow_loom.settings.PhysicsSettings.max_ingest_words``
# so the MCP tools enforce the same cap.
MAX_INGEST_WORDS: int = _settings.physics.max_ingest_words


def count_words(text: str) -> int:
    """Whitespace-tokenised word count used by the input gates."""
    if not text:
        return 0
    return len(text.split())
