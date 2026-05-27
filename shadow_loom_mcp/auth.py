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
    """Return the single cached token entry, or None.

    Round-10 R10-01: previously returned the *most recently inserted*
    entry whenever any tokens had been cached. With more than one
    cached identity that behaviour is an impersonation surface — any
    open-mode caller without a request context inherits whichever
    token was validated last (a different user's). Now we only fall
    back when **exactly one** token has been seen, so the canonical
    "single test client" dev pattern still works but ambiguous
    multi-tenant fallback is denied (and logged) instead of being
    silently resolved to the wrong identity.
    """
    n = len(_token_user_cache)
    if n == 0:
        return None
    if n > 1:
        logger.warning(
            "[MCP·auth] Open-mode fallback refused: %d cached tokens "
            "in process, cannot choose unambiguously. Resolving as "
            "no-identity (returns None / empty scopes).",
            n,
        )
        return None
    last_token = next(iter(_token_user_cache))
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

# Role hierarchy used for ``min_role`` enforcement. Higher index =
# stronger permission. ``viewer`` is read-only; ``editor`` may mutate
# project content; ``admin`` may manage members and project settings.
_ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2}


def check_project_access(
    project_id: int,
    ctx: Context,
    *,
    min_role: str = "viewer",
) -> Optional[str]:
    """Check if the authenticated user can access a project at *min_role*.

    ``min_role`` defaults to ``"viewer"`` (read access). Pass
    ``"editor"`` for mutating tools and ``"admin"`` for project-
    management tools. The project owner always passes regardless of
    *min_role*.

    Returns ``None`` if allowed, or an error string if denied.
    """
    user_id = get_user_id(ctx)
    proj = get_project(project_id)
    if proj is None:
        return "Project not found."
    # Owner always has access at every level.
    # AUDIT (post-2026-05-26): require an authenticated user_id before
    # matching owner equality. ``None == None`` was previously granting
    # full owner privileges on ownerless / seeded projects to any
    # unauthenticated caller.
    if user_id is not None and proj.owner_id == user_id:
        return None
    required_rank = _ROLE_RANK.get(min_role, 0)
    # Public projects are readable by all, but writes still require an
    # explicit member role at the requested level.
    if proj.is_public and required_rank == 0:
        return None
    # Check membership and role rank.
    if user_id is not None:
        role = get_user_project_role(project_id, user_id)
        if role is not None:
            actual_rank = _ROLE_RANK.get(role, 0)
            if actual_rank >= required_rank:
                return None
            return (
                f"Access denied: project role '{role}' is below required "
                f"'{min_role}' for this operation."
            )
    return "Access denied: you do not have permission to access this project."


# ── Rate limiting ────────────────────────────────────────────────

# Round-12 R12-08: simple in-process token-bucket on expensive
# generation paths (``narrate``, ``direct``) to bound LLM/compute
# spend per identity. This is a *single-process* guard — for a
# multi-worker deployment a shared store (Redis) is the right
# next step, but in-process is strictly better than the zero
# defence we had before and matches the threat model of a
# single self-hosted MCP server.
import os as _os
import time as _time
from collections import deque as _deque
from threading import Lock as _Lock

_rate_buckets: dict[tuple[str, int | None], _deque] = {}
_rate_lock = _Lock()


def _rate_limit_per_minute() -> int:
    raw = _os.environ.get("SHADOW_LOOM_MCP_RATE_PER_MIN")
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return 10


def check_rate_limit(ctx: Context, kind: str) -> Optional[str]:
    """Enforce a per-(user, kind) rate limit on expensive MCP tools.

    Returns ``None`` if the call is allowed, else an error message
    naming the limit. ``kind`` is a free-form bucket label (e.g.
    ``"narrate"``) so different expensive endpoints maintain
    independent windows. Unauthenticated (open-mode) callers share
    a single ``None`` bucket which still applies the same cap.
    """
    cap = _rate_limit_per_minute()
    user_id = get_user_id(ctx)
    key = (kind, user_id)
    now = _time.monotonic()
    window = 60.0
    with _rate_lock:
        bucket = _rate_buckets.get(key)
        if bucket is None:
            bucket = _deque()
            _rate_buckets[key] = bucket
        # Drop timestamps that have aged out of the window.
        while bucket and (now - bucket[0]) > window:
            bucket.popleft()
        if len(bucket) >= cap:
            retry_in = max(0.0, window - (now - bucket[0]))
            return (
                f"Rate limit exceeded: at most {cap} '{kind}' calls per "
                f"minute per identity. Retry in ~{retry_in:.0f}s."
            )
        bucket.append(now)
        return None
