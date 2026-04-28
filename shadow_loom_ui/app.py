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
from shadow_loom_ui.theme import apply_theme, feather
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
    try:
        from shadow_loom_ui.example_seeder import seed_examples
        seed_examples()
    except Exception:
        logger.exception("[App] Example seeding failed")
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

# In-memory cache of AppState keyed by browser session id.
# AppState contains non-JSON-serialisable objects (PipelineConfig, callbacks,
# world models, etc.), so it cannot live in app.storage.user which is
# persisted to disk as JSON.
_SESSION_STATES: dict[str, AppState] = {}


def _get_session_state() -> AppState:
    """Return the per-session AppState, creating it if needed.

    Kept in a module-level dict keyed by the browser session id from
    ``app.storage.browser``. Auth info is sourced from ``app.storage.user``
    (which is JSON-safe) on first access.
    """
    storage = app.storage.user
    session_id = app.storage.browser.get("id", "")
    # Drop any legacy non-serialisable state that older builds may have left
    # in the persisted user storage.
    if "_state" in storage:
        storage.pop("_state", None)
    state = _SESSION_STATES.get(session_id)
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
        elif not config.AUTH_ENABLED:
            # No OAuth providers configured \u2014 attach the built-in
            # local user so the UI behaves like a single-user app.
            from shadow_loom.db import ensure_local_user
            local = ensure_local_user()
            state.set_user(
                user_id=local.id or 0,
                username=local.username,
                display_name=local.display_name or "Local User",
            )
        _SESSION_STATES[session_id] = state
    return state


def _build_app_header(state: AppState, *, show_back: bool = False):
    """Render the shared top navigation bar."""
    with ui.header().classes(
        "bg-white border-b border-slate-200 px-6 py-3 "
        "flex items-center justify-between text-slate-800"
    ).props("elevated=false flat"):
        with ui.row().classes("items-center gap-3"):
            if show_back:
                with ui.button(on_click=lambda: ui.navigate.to("/")).props(
                    "flat dense round color=secondary"
                ):
                    feather("arrow-left")
            with ui.row().classes("items-center gap-2 cursor-pointer").on(
                "click", lambda: ui.navigate.to("/")
            ):
                feather("book-open", size="lg", color=_brand_copper())
                ui.label("Shadow Loom").classes(
                    "text-xl font-bold tracking-tight text-slate-800"
                )

            if state.project_name:
                ui.label(f"— {state.project_name}").classes(
                    "text-sm font-medium text-slate-500"
                )

        with ui.row().classes("items-center gap-1"):
            storage = app.storage.user
            authed = storage.get("authenticated")
            local_mode = (not config.AUTH_ENABLED) and state.user_id is not None
            if authed or local_mode:
                # Background tasks indicator
                from shadow_loom_ui.components.tasks_indicator import build_tasks_indicator
                build_tasks_indicator(state)

                avatar = storage.get("avatar_url", "") if authed else ""
                name = (
                    storage.get("display_name") or storage.get("username", "User")
                    if authed
                    else (state.display_name or state.username or "Local User")
                )
                if avatar:
                    ui.avatar().props(f'src="{avatar}"').classes(
                        "cursor-pointer"
                    ).on("click", lambda: ui.navigate.to("/settings"))
                ui.label(name).classes(
                    "text-sm font-medium text-slate-700 cursor-pointer hidden sm:block"
                ).on("click", lambda: ui.navigate.to("/settings"))
                with ui.button(on_click=lambda: ui.navigate.to("/settings")).props(
                    "flat dense round color=secondary"
                ):
                    feather("settings")
                if authed:
                    with ui.button(
                        on_click=lambda: ui.navigate.to("/auth/logout")
                    ).props("flat dense round color=secondary"):
                        feather("log-out")
            elif config.AUTH_ENABLED:
                with ui.button(on_click=lambda: ui.navigate.to("/login")).props(
                    "flat dense color=primary no-caps"
                ):
                    with ui.row().classes("items-center gap-2"):
                        feather("log-in")
                        ui.label("Sign in")


def _brand_copper() -> str:
    """Late import so theme module is initialised."""
    from shadow_loom_ui.theme import PRIMARY
    return PRIMARY


# =====================================================================
# Pages
# =====================================================================

@ui.page("/login")
def login_page():
    """Authentication page with OAuth provider buttons."""
    from shadow_loom_ui.components.login import build_login_page

    apply_theme(_ui_settings.dark_mode)
    build_login_page()


@ui.page("/")
def dashboard_page():
    """Dashboard — project gallery, examples, activity feed."""
    from shadow_loom_ui.components.dashboard import build_dashboard

    state = _get_session_state()
    apply_theme(_ui_settings.dark_mode)
    _build_app_header(state)
    build_dashboard(state)


@ui.page("/project/{project_id}")
def workspace_page(project_id: int):
    """Project workspace — tabbed work surface."""
    from shadow_loom_ui.components.workspace import build_workspace

    state = _get_session_state()
    apply_theme(_ui_settings.dark_mode)
    _build_app_header(state, show_back=True)
    build_workspace(state, project_id)


@ui.page("/settings")
def settings_page():
    """Account settings, API keys, preferences."""
    from shadow_loom_ui.components.settings import build_settings

    state = _get_session_state()
    apply_theme(_ui_settings.dark_mode)
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
