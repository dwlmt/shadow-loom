# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Auth helpers for Shadow-Loom MCP — bearer token → user resolution, scope checks.

All tools resolve the authenticated user from the bearer token implicitly.
No ``user_id`` parameter is needed on any tool.

Scopes:
  read  — ORIENT + EXPLORE + REASON + JUDGE
  write — CREATE + MANAGE (branch, fork)
  admin — share, update_project
"""

from __future__ import annotations

import logging
from typing import Optional

from fastmcp import Context
from fastmcp.server.auth.providers.debug import DebugTokenVerifier

from shadow_loom.db import (
    get_project,
    get_user_project_role,
    validate_api_key,
)

logger = logging.getLogger(__name__)

# ── Token cache (token → resolved user info) ─────────────────────

_token_user_cache: dict[str, dict] = {}


def invalidate_token_cache(*, key_id: int | None = None, user_id: int | None = None) -> int:
    """Drop cached token entries matching *key_id* or *user_id*.

    Without this, ``revoke_api_key`` only marks the DB row inactive but
    in-process bearer-token validation keeps returning the cached user,
    so a revoked token remains usable for the lifetime of the server.
    Pass no arguments to clear the entire cache. Returns the number of
    entries removed.
    """
    if key_id is None and user_id is None:
        n = len(_token_user_cache)
        _token_user_cache.clear()
        return n
    to_drop = [
        tok for tok, info in _token_user_cache.items()
        if (key_id is not None and info.get("key_id") == key_id)
        or (user_id is not None and info.get("user_id") == user_id)
    ]
    for tok in to_drop:
        _token_user_cache.pop(tok, None)
    return len(to_drop)


def _validate_bearer_token(token: str) -> bool:
    """Validate an API key against the DB and cache the resolved user."""
    key_row = validate_api_key(token)
    if key_row is None:
        return False
    _token_user_cache[token] = {
        "user_id": key_row.user_id,
        "scopes": set(key_row.scopes.split(",")) if key_row.scopes else set(),
        "key_id": key_row.id,
    }
    return True


verifier = DebugTokenVerifier(validate=_validate_bearer_token)


# ── User resolution from Context ─────────────────────────────────

def _open_mode_enabled() -> bool:
    """Return True if the MCP server is in open (dev/test) mode.

    Reads ``MCP_ALLOW_OPEN_MODE`` directly from the environment rather
    than via :func:`shadow_loom.settings.get_settings`, which is
    ``@lru_cache``'d for the lifetime of the process. The cached form
    captures the env at first access, so any test that imports settings
    before this module sets the env var would freeze the flag at
    ``False`` and silently break every later open-mode test in the same
    pytest session. Going straight to ``os.environ`` keeps the toggle
    honest while still falling back to the cached settings value when
    the env var is unset.
    """
    import os

    raw = os.environ.get("MCP_ALLOW_OPEN_MODE")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    try:
        from shadow_loom.settings import get_settings
        return bool(getattr(get_settings().mcp, "allow_open_mode", False))
    except Exception:
        return False


def _last_cached_entry() -> Optional[dict]:
    """Return the most recently inserted token cache entry, or None."""
    if not _token_user_cache:
        return None
    # dict preserves insertion order in CPython 3.7+
    last_token = next(reversed(_token_user_cache))
    return _token_user_cache[last_token]


def get_user_id(ctx: Context) -> Optional[int]:
    """Resolve the authenticated user_id from the bearer token in Context.

    Fail-closed in production: if the token cannot be resolved from the
    active request context, return ``None``. In open mode (dev/test only),
    fall back to the most recently cached token entry so MagicMock-based
    tests can drive the tools without simulating a full transport layer.
    """
    rc = ctx.request_context
    if rc is None:
        if _open_mode_enabled():
            entry = _last_cached_entry()
            return entry["user_id"] if entry else None
        return None
    token_info = getattr(rc, "access_token", None)
    if token_info is None:
        return None
    raw_token = getattr(token_info, "claims", {}).get("token")
    if raw_token and raw_token in _token_user_cache:
        return _token_user_cache[raw_token]["user_id"]
    return None


def get_scopes(ctx: Context) -> set[str]:
    """Get the scopes for the authenticated user. Fail-closed (empty set).

    In open mode (dev/test only), fall back to the most recently cached
    token entry's scopes — see ``get_user_id`` for rationale.
    """
    rc = ctx.request_context
    if rc is None:
        if _open_mode_enabled():
            entry = _last_cached_entry()
            return set(entry["scopes"]) if entry else set()
        return set()
    token_info = getattr(rc, "access_token", None)
    if token_info is None:
        return set()
    raw_token = getattr(token_info, "claims", {}).get("token")
    if raw_token and raw_token in _token_user_cache:
        return _token_user_cache[raw_token]["scopes"]
    return set()


# ── Scope enforcement ─────────────────────────────────────────────

def require_scope(ctx: Context, scope: str) -> Optional[str]:
    """Check the authenticated user has *scope*. Returns error string or None.

    Fail-closed: if no scopes are resolved, deny access unless the MCP
    server has been explicitly placed in open mode via
    ``MCP_ALLOW_OPEN_MODE=true`` (development only).
    """
    scopes = get_scopes(ctx)
    if not scopes:
        if _open_mode_enabled():
            return None  # Explicit dev override
        return f"Access denied: scope '{scope}' required (no auth context)."
    if scope not in scopes:
        return f"Access denied: scope '{scope}' required."
    return None


# ── Project access check ─────────────────────────────────────────

def check_project_access(
    project_id: int,
    ctx: Context,
) -> Optional[str]:
    """Check if the authenticated user can access a project.

    Returns None if allowed, or an error string if denied.
    """
    user_id = get_user_id(ctx)
    proj = get_project(project_id)
    if proj is None:
        return "Project not found."
    # Owner always has access
    if proj.owner_id == user_id:
        return None
    # Public projects are readable by all
    if proj.is_public:
        return None
    # Check membership
    if user_id is not None:
        role = get_user_project_role(project_id, user_id)
        if role is not None:
            return None
    return "Access denied: you do not have permission to access this project."
