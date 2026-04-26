"""OAuth authentication middleware and routes for Shadow-Loom UI.

Supports GitHub and Google OAuth providers via Authlib.
When no OAuth credentials are configured, the app runs without auth.
"""

from __future__ import annotations

import logging
from typing import Optional

from authlib.integrations.starlette_client import OAuth
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

from shadow_loom_ui import config
from shadow_loom_ui.db import upsert_user

logger = logging.getLogger(__name__)

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


# =====================================================================
# Middleware: require auth on all pages (except /auth/* and /api/*)
# =====================================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated users to login page.

    Only active when AUTH_ENABLED is True.
    """

    OPEN_PREFIXES = ("/auth/", "/api/", "/_nicegui/", "/static/", "/favicon")

    async def dispatch(self, request: Request, call_next):
        if not config.AUTH_ENABLED:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(p) for p in self.OPEN_PREFIXES):
            return await call_next(request)

        # Check NiceGUI app.storage.user for auth flag
        from nicegui import app
        storage = app.storage.user
        if storage.get("authenticated"):
            return await call_next(request)

        # Not authenticated — if it's the main page, show login
        if path == "/":
            return await call_next(request)  # Login UI renders on /

        return RedirectResponse("/")


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
        email = profile.get("email")
        avatar = profile.get("avatar_url")
    elif provider_name == "google":
        userinfo = token.get("userinfo", {})
        user_id = userinfo.get("sub", "")
        username = userinfo.get("name", "unknown")
        email = userinfo.get("email")
        avatar = userinfo.get("picture")
    else:
        return JSONResponse({"error": "Unsupported provider"}, status_code=400)

    # Persist user
    db_user = upsert_user(
        provider=provider_name,
        provider_id=f"{provider_name}:{user_id}",
        username=username,
        email=email,
        avatar_url=avatar,
    )

    # Set session
    storage = app.storage.user
    storage["authenticated"] = True
    storage["user_id"] = db_user.id
    storage["username"] = username
    storage["avatar_url"] = avatar or ""
    storage["provider"] = provider_name

    logger.info("[Auth] User logged in: %s (%s)", username, provider_name)
    return RedirectResponse("/")


async def auth_logout(request: Request):
    """Clear session and redirect to home."""
    from nicegui import app
    storage = app.storage.user
    storage.clear()
    return RedirectResponse("/")
