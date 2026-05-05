# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OAuth authentication middleware and routes for Shadow-Loom UI.

Supports GitHub, Google, Discord, Microsoft, and Apple OAuth providers
via Authlib. Also supports bearer-token API key authentication for
MCP/API access. When no OAuth credentials are configured, the app runs
without auth.
"""

from __future__ import annotations

import logging
import time

from authlib.integrations.starlette_client import OAuth
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

from shadow_loom_ui import config
from shadow_loom_ui.db import upsert_user, validate_api_key

logger = logging.getLogger(__name__)


# =====================================================================
# Apple Sign In — ES256 client_secret JWT (auto-minted, cached, refreshed)
# =====================================================================
# Apple requires the OAuth `client_secret` to be a JWT signed with the
# .p8 private key issued in the Apple Developer console. Spec:
# https://developer.apple.com/documentation/sign_in_with_apple/generate_and_validate_tokens
# Max expiry is 6 months; we mint a fresh 1-hour JWT on demand.

_APPLE_AUDIENCE = "https://appleid.apple.com"
_apple_jwt_cache: dict[str, object] = {"token": "", "expires_at": 0.0}


def _mint_apple_client_secret() -> str:
    """Return a cached or freshly-minted Apple client_secret JWT."""
    if config.APPLE_CLIENT_SECRET:
        # Operator supplied a pre-minted JWT; trust them to rotate it.
        return config.APPLE_CLIENT_SECRET
    if not (
        config.APPLE_TEAM_ID
        and config.APPLE_KEY_ID
        and config.APPLE_PRIVATE_KEY
        and config.APPLE_CLIENT_ID
    ):
        return ""

    now = time.time()
    cached = _apple_jwt_cache
    if cached["token"] and float(cached["expires_at"]) - now > 60:
        return str(cached["token"])

    # authlib.jose is deprecated in favour of joserfc but still ships
    # with authlib<2.0; using it avoids pulling in another dependency.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        from authlib.jose import jwt as _jwt

    expires_at = now + 3600  # 1 hour
    header = {"alg": "ES256", "kid": config.APPLE_KEY_ID}
    payload = {
        "iss": config.APPLE_TEAM_ID,
        "iat": int(now),
        "exp": int(expires_at),
        "aud": _APPLE_AUDIENCE,
        "sub": config.APPLE_CLIENT_ID,
    }
    private_key = config.APPLE_PRIVATE_KEY.replace("\\n", "\n")
    token = _jwt.encode(header, payload, private_key).decode("ascii")
    cached["token"] = token
    cached["expires_at"] = expires_at
    return token


oauth = OAuth()

# Register providers (only if credentials are set)
if config.GITHUB_CLIENT_ID:
    oauth.register(
        name="github",
        client_id=config.GITHUB_CLIENT_ID,
        client_secret=config.GITHUB_CLIENT_SECRET,
        authorize_url="https://github.com/login/oauth/authorize",
        access_token_url="https://github.com/login/oauth/access_token",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "read:user user:email"},
    )

if config.GOOGLE_CLIENT_ID:
    oauth.register(
        name="google",
        client_id=config.GOOGLE_CLIENT_ID,
        client_secret=config.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if config.DISCORD_CLIENT_ID:
    oauth.register(
        name="discord",
        client_id=config.DISCORD_CLIENT_ID,
        client_secret=config.DISCORD_CLIENT_SECRET,
        authorize_url="https://discord.com/api/oauth2/authorize",
        access_token_url="https://discord.com/api/oauth2/token",
        api_base_url="https://discord.com/api/",
        client_kwargs={"scope": "identify email"},
    )

if config.MICROSOFT_CLIENT_ID:
    oauth.register(
        name="microsoft",
        client_id=config.MICROSOFT_CLIENT_ID,
        client_secret=config.MICROSOFT_CLIENT_SECRET,
        server_metadata_url=(
            "https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration"
        ),
        client_kwargs={"scope": "openid email profile"},
    )

if config.APPLE_CLIENT_ID and _mint_apple_client_secret():
    # Apple POSTs back to the redirect_uri with `response_mode=form_post`
    # whenever scopes other than `openid` are requested.
    oauth.register(
        name="apple",
        client_id=config.APPLE_CLIENT_ID,
        client_secret=_mint_apple_client_secret,  # callable so JWTs auto-refresh
        server_metadata_url=(
            "https://appleid.apple.com/.well-known/openid-configuration"
        ),
        client_kwargs={
            "scope": "name email",
            "response_mode": "form_post",
        },
    )


# =====================================================================
# Middleware: require auth on pages, support bearer tokens on /api/*
# =====================================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated users to login page.

    Only active when AUTH_ENABLED is True.
    API requests with a valid Bearer token are always allowed.
    """

    OPEN_PREFIXES = ("/auth/", "/_nicegui/", "/static/", "/favicon")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Always allow open prefixes
        if any(path.startswith(p) for p in self.OPEN_PREFIXES):
            return await call_next(request)

        # Bearer token auth for API / MCP requests
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # validate_api_key hits the DB synchronously \u2014 offload so we
            # do not stall the event loop under concurrent API traffic.
            import asyncio as _asyncio
            key_row = await _asyncio.to_thread(validate_api_key, token)
            if key_row is not None:
                # Attach user info to request state for downstream use
                request.state.api_user_id = key_row.user_id
                request.state.api_key_scopes = key_row.scopes.split(",")
                return await call_next(request)
            # Invalid token on API routes → 401
            if path.startswith("/api/"):
                return JSONResponse(
                    {"error": "Invalid or expired API key"}, status_code=401
                )

        if not config.AUTH_ENABLED:
            return await call_next(request)

        # Check NiceGUI app.storage.user for auth flag
        from nicegui import app
        storage = app.storage.user
        if storage.get("authenticated"):
            return await call_next(request)

        # Not authenticated — allow login page and main page (login UI renders there)
        if path in ("/", "/login"):
            return await call_next(request)

        return RedirectResponse("/login")


# =====================================================================
# Route handlers (mounted in app.py)
# =====================================================================

async def auth_login(request: Request):
    """Redirect to OAuth provider."""
    provider_name = request.path_params.get("provider", "github")
    client = oauth.create_client(provider_name)
    if client is None:
        return JSONResponse({"error": f"Provider {provider_name} not configured"}, status_code=400)
    redirect_uri = f"{config.OAUTH_REDIRECT_BASE}/auth/{provider_name}/callback"
    return await client.authorize_redirect(request, redirect_uri)


async def auth_callback(request: Request):
    """Handle OAuth callback and create session."""
    from nicegui import app

    provider_name = request.path_params.get("provider", "github")
    client = oauth.create_client(provider_name)
    if client is None:
        return JSONResponse({"error": f"Provider {provider_name} not configured"}, status_code=400)

    token = await client.authorize_access_token(request)

    if provider_name == "github":
        resp = await client.get("user", token=token)
        profile = resp.json()
        user_id = str(profile.get("id", ""))
        username = profile.get("login", "unknown")
        display_name = profile.get("name")
        email = profile.get("email")
        avatar = profile.get("avatar_url")
    elif provider_name == "google":
        userinfo = token.get("userinfo", {})
        user_id = userinfo.get("sub", "")
        username = userinfo.get("email", "unknown").split("@")[0]
        display_name = userinfo.get("name")
        email = userinfo.get("email")
        avatar = userinfo.get("picture")
    elif provider_name == "discord":
        resp = await client.get("users/@me", token=token)
        profile = resp.json()
        user_id = str(profile.get("id", ""))
        username = profile.get("username", "unknown")
        display_name = profile.get("global_name")
        email = profile.get("email")
        disc_avatar = profile.get("avatar")
        avatar = (
            f"https://cdn.discordapp.com/avatars/{user_id}/{disc_avatar}.png"
            if disc_avatar else None
        )
    elif provider_name == "microsoft":
        userinfo = token.get("userinfo", {})
        user_id = userinfo.get("sub", "") or userinfo.get("oid", "")
        username = (userinfo.get("preferred_username", "unknown").split("@")[0])
        display_name = userinfo.get("name")
        email = userinfo.get("email") or userinfo.get("preferred_username")
        avatar = None  # Microsoft Graph photo requires separate API call
    elif provider_name == "apple":
        # Apple's identity is delivered entirely in the id_token; the
        # `user` form field (sent only on first auth) carries the name.
        userinfo = token.get("userinfo") or {}
        if not userinfo:
            try:
                userinfo = await client.parse_id_token(request, token)
            except Exception:  # noqa: BLE001 — fall back to raw token
                userinfo = {}
        user_id = userinfo.get("sub", "")
        email = userinfo.get("email")
        # First-time consent: Apple sends `user={"name":{...},"email":...}`
        # in the form body; subsequent logins omit it.
        try:
            form = await request.form()
            user_blob = form.get("user")
            if user_blob:
                import json as _json
                user_payload = _json.loads(user_blob)
                name = user_payload.get("name") or {}
                display_name = (
                    " ".join(
                        part for part in (name.get("firstName"), name.get("lastName")) if part
                    ).strip()
                    or None
                )
            else:
                display_name = None
        except Exception:  # noqa: BLE001
            display_name = None
        username = (email.split("@")[0] if email else f"apple-{user_id[:8]}") or "apple-user"
        avatar = None  # Apple does not expose a profile picture endpoint
    else:
        return JSONResponse({"error": "Unsupported provider"}, status_code=400)

    # Persist user
    db_user = upsert_user(
        provider=provider_name,
        provider_id=f"{provider_name}:{user_id}",
        username=username,
        email=email,
        avatar_url=avatar,
        display_name=display_name,
    )

    # Set session
    storage = app.storage.user
    storage["authenticated"] = True
    storage["user_id"] = db_user.id
    storage["username"] = username
    storage["display_name"] = display_name or username
    storage["avatar_url"] = avatar or ""
    storage["provider"] = provider_name

    logger.info("[Auth] User logged in: %s (%s)", username, provider_name)
    return RedirectResponse("/")


async def auth_logout(request: Request):
    """Clear session and redirect to home."""
    from nicegui import app
    storage = app.storage.user
    storage.clear()
    return RedirectResponse("/login")
