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

def get_user_id(ctx: Context) -> Optional[int]:
    """Resolve the authenticated user_id from the bearer token in Context."""
    rc = ctx.request_context
    if rc is not None:
        token_info = getattr(rc, "access_token", None)
        if token_info is not None:
            raw_token = getattr(token_info, "claims", {}).get("token")
            if raw_token and raw_token in _token_user_cache:
                return _token_user_cache[raw_token]["user_id"]
    # Fallback: return last cached user (single-request model)
    if _token_user_cache:
        return list(_token_user_cache.values())[-1]["user_id"]
    return None


def get_scopes(ctx: Context) -> set[str]:
    """Get the scopes for the authenticated user."""
    rc = ctx.request_context
    if rc is not None:
        token_info = getattr(rc, "access_token", None)
        if token_info is not None:
            raw_token = getattr(token_info, "claims", {}).get("token")
            if raw_token and raw_token in _token_user_cache:
                return _token_user_cache[raw_token]["scopes"]
    if _token_user_cache:
        return list(_token_user_cache.values())[-1]["scopes"]
    return set()


# ── Scope enforcement ─────────────────────────────────────────────

def require_scope(ctx: Context, scope: str) -> Optional[str]:
    """Check the authenticated user has *scope*. Returns error string or None."""
    scopes = get_scopes(ctx)
    if not scopes:
        return None  # No auth configured — allow (open mode)
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
