"""Shadow-Loom NiceGUI web application — main entry point.

Multi-page routing:
  /login          – Authentication page
  /               – Dashboard (project gallery)
  /project/{id}   – Workspace (tabbed work surface)
  /settings       – Account, API keys, preferences

Run with:  python -m shadow_loom_ui.app
Or:        nicegui run shadow_loom_ui/app.py
"""

from __future__ import annotations

import logging

from nicegui import app, ui

from shadow_loom_ui import config
from shadow_loom_ui.auth import AuthMiddleware, auth_callback, auth_login, auth_logout
from shadow_loom_ui.db import init_db
from shadow_loom_ui.state import AppState
from shadow_loom.settings import get_settings as _get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_ui_settings = _get_settings().ui


# =====================================================================
# Startup
# =====================================================================

def _on_startup():
    """Initialise database and auth on app startup."""
    init_db(config.DATABASE_URL)
    logger.info("[App] Shadow-Loom UI starting on port %s", _ui_settings.port)


app.on_startup(_on_startup)

# Auth middleware
app.add_middleware(AuthMiddleware)

# Auth routes
app.add_route("/auth/{provider}", auth_login, methods=["GET"])
app.add_route("/auth/{provider}/callback", auth_callback, methods=["GET"])
app.add_route("/auth/logout", auth_logout, methods=["GET"])


# =====================================================================
# Helpers
# =====================================================================

def _get_session_state() -> AppState:
    """Return the per-session AppState, creating it if needed.

    Stored in NiceGUI's app.storage.user under ``_state``.
    """
    storage = app.storage.user
    state: AppState | None = storage.get("_state")
    if state is None:
        state = AppState()
        # Populate user info from session if authenticated
        if storage.get("authenticated"):
            state.set_user(
                user_id=storage.get("user_id", 0),
                username=storage.get("username", ""),
                display_name=storage.get("display_name", ""),
                avatar_url=storage.get("avatar_url", ""),
            )
        storage["_state"] = state
    return state


def _build_app_header(state: AppState, *, show_back: bool = False):
    """Render the shared top navigation bar."""
    with ui.header().classes("items-center justify-between q-pa-sm"):
        with ui.row().classes("items-center gap-2"):
            if show_back:
                ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
                    "flat dense round"
                )
            ui.icon("auto_stories", size="md", color="primary")
            ui.label("Shadow Loom").classes("text-h6 cursor-pointer").on(
                "click", lambda: ui.navigate.to("/")
            )

            if state.project_name:
                ui.label(f"— {state.project_name}").classes("text-subtitle1 text-grey")

        with ui.row().classes("items-center gap-1"):
            storage = app.storage.user
            if storage.get("authenticated"):
                avatar = storage.get("avatar_url", "")
                name = storage.get("display_name") or storage.get("username", "User")
                if avatar:
                    ui.avatar().props(f'src="{avatar}"').classes(
                        "cursor-pointer"
                    ).on("click", lambda: ui.navigate.to("/settings"))
                ui.label(name).classes("text-body2 cursor-pointer").on(
                    "click", lambda: ui.navigate.to("/settings")
                )
                ui.button(icon="settings", on_click=lambda: ui.navigate.to("/settings")).props(
                    "flat dense round"
                )
                ui.button(icon="logout", on_click=lambda: ui.navigate.to("/auth/logout")).props(
                    "flat dense round"
                )
            elif config.AUTH_ENABLED:
                ui.button("Sign in", icon="login", on_click=lambda: ui.navigate.to("/login")).props(
                    "flat dense"
                )


# =====================================================================
# Pages
# =====================================================================

@ui.page("/login")
def login_page():
    """Authentication page with OAuth provider buttons."""
    from shadow_loom_ui.components.login import build_login_page

    ui.dark_mode(_ui_settings.dark_mode)
    build_login_page()


@ui.page("/")
def dashboard_page():
    """Dashboard — project gallery, examples, activity feed."""
    from shadow_loom_ui.components.dashboard import build_dashboard

    state = _get_session_state()
    ui.dark_mode(_ui_settings.dark_mode)
    _build_app_header(state)
    build_dashboard(state)


@ui.page("/project/{project_id}")
def workspace_page(project_id: int):
    """Project workspace — tabbed work surface."""
    from shadow_loom_ui.components.workspace import build_workspace

    state = _get_session_state()
    ui.dark_mode(_ui_settings.dark_mode)
    _build_app_header(state, show_back=True)
    build_workspace(state, project_id)


@ui.page("/settings")
def settings_page():
    """Account settings, API keys, preferences."""
    from shadow_loom_ui.components.settings import build_settings

    state = _get_session_state()
    ui.dark_mode(_ui_settings.dark_mode)
    _build_app_header(state, show_back=True)
    build_settings(state)


# =====================================================================
# Entry point
# =====================================================================

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host=_ui_settings.host,
        port=_ui_settings.port,
        title=_ui_settings.title,
        storage_secret=config.STORAGE_SECRET,
        dark=_ui_settings.dark_mode,
        reload=_ui_settings.reload,
    )
