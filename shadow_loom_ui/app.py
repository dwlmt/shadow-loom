# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

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


async def _on_shutdown() -> None:
    """Cancel any in-flight per-session background tasks before exit.

    Each ``AppState`` tracks its own asyncio task handles; without this
    hook a long-running ingestion can leak past app shutdown and emit
    "Task was destroyed but it is pending" warnings.
    """
    for state in list(_SESSION_STATES.values()):
        try:
            await state.cancel_all_async_tasks()
        except Exception:
            logger.exception("[App] Error cancelling session tasks on shutdown")


app.on_shutdown(_on_shutdown)

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


def _build_app_footer() -> None:
    """Render the shared footer with AGPLv3 § 13 source-code link.

    Shadow Loom is licensed under AGPL-3.0-or-later; § 13 of that
    licence requires hosted instances to offer the corresponding
    source code to interacting users. The link target is configurable
    via the ``UI_SOURCE_URL`` environment variable so operators of a
    modified build can point users at *their* corresponding source,
    as the licence requires.
    """
    source_url = _ui_settings.source_url
    with ui.footer().classes(
        "bg-white border-t border-slate-200 px-6 py-2 "
        "flex items-center justify-between text-xs text-slate-500"
    ).props("elevated=false flat"):
        ui.label(
            "Shadow Loom — AGPL-3.0-or-later. "
            "This is free software with NO WARRANTY."
        )
        with ui.row().classes("items-center gap-3"):
            ui.link("Source", source_url, new_tab=True).classes(
                "text-slate-600 hover:text-slate-900 underline"
            )
            ui.link(
                "Licence",
                f"{source_url.rstrip('/')}/blob/main/LICENSE",
                new_tab=True,
            ).classes("text-slate-600 hover:text-slate-900 underline")
            licence_dialog = _build_licence_guide_dialog(source_url)
            ui.button(
                "What can I do?",
                on_click=licence_dialog.open,
            ).props("flat dense no-caps size=sm color=primary").classes(
                "text-xs"
            )


def _build_licence_guide_dialog(source_url: str):
    """Build a popup with a checkbox guide to AGPLv3 + commercial use.

    Plain-language summary of what users can and can't do under
    Shadow Loom's dual-licence model. Not legal advice — links out
    to the canonical LICENSE, COMMERCIAL-LICENSE.md, CONTRIBUTING.md
    and README for the authoritative text.
    """
    base = source_url.rstrip("/")
    readme_url = f"{base}#readme"
    licence_url = f"{base}/blob/main/LICENSE"
    commercial_url = f"{base}/blob/main/COMMERCIAL-LICENSE.md"
    contributing_url = f"{base}/blob/main/CONTRIBUTING.md"

    # ✓ allowed under AGPLv3, ✗ requires commercial licence / forbidden,
    # ⚠ allowed but with obligations.
    permissions = [
        ("check_circle", "positive", "Use it for personal projects, research, and learning."),
        ("check_circle", "positive", "Read, modify, and fork the source code."),
        ("check_circle", "positive", "Run it on your own machine without restriction."),
        ("check_circle", "positive", "Redistribute it — as long as you keep it under AGPL-3.0-or-later."),
        (
            "warning",
            "warning",
            "Host it as a network service (SaaS, MCP server, hosted UI): "
            "you must offer the complete corresponding source code "
            "(including your modifications) to your users under AGPLv3.",
        ),
        (
            "warning",
            "warning",
            "Embed or link it into a larger product: the whole combined "
            "work must also be released under AGPLv3.",
        ),
        (
            "cancel",
            "negative",
            "Use it commercially without complying with AGPLv3 § 13 "
            "(network-use disclosure) — that requires a paid commercial licence.",
        ),
        (
            "cancel",
            "negative",
            "Re-license it under a more permissive licence, or ship it "
            "inside closed-source software, without a commercial licence.",
        ),
    ]

    contributing_items = [
        ("check_circle", "positive", "Open issues and pull requests on GitHub."),
        (
            "info",
            "info",
            "Contributions are accepted under the Developer Certificate of "
            "Origin (DCO) plus a copyright licence-back so they can ship "
            "under both the AGPL and the commercial licence.",
        ),
        (
            "info",
            "info",
            "Sign your commits with `git commit -s` to certify the DCO.",
        ),
    ]

    with ui.dialog() as dialog, ui.card().classes(
        "w-full max-w-2xl bg-white rounded-xl"
    ):
        with ui.row().classes("items-center gap-2 w-full"):
            ui.icon("gavel", color="primary")
            ui.label("What can I do with Shadow Loom?").classes(
                "text-lg font-semibold text-slate-800"
            )
            ui.space()
            ui.button(icon="close", on_click=dialog.close).props(
                "flat dense round size=sm color=secondary"
            )

        ui.label(
            "Shadow Loom is dual-licensed under AGPL-3.0-or-later and a "
            "commercial licence. Quick guide — not legal advice."
        ).classes("text-xs text-slate-500")

        ui.separator()

        ui.label("Open-source use (AGPLv3)").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        with ui.column().classes("gap-1 w-full"):
            for icon_name, color, text in permissions:
                with ui.row().classes("items-start gap-2 w-full no-wrap"):
                    ui.icon(icon_name, color=color).classes("mt-0.5")
                    ui.label(text).classes("text-sm text-slate-700 flex-1")

        ui.separator().classes("mt-2")

        ui.label("Contributing").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        with ui.column().classes("gap-1 w-full"):
            for icon_name, color, text in contributing_items:
                with ui.row().classes("items-start gap-2 w-full no-wrap"):
                    ui.icon(icon_name, color=color).classes("mt-0.5")
                    ui.label(text).classes("text-sm text-slate-700 flex-1")

        ui.separator().classes("mt-2")

        with ui.row().classes("items-center gap-3 mt-2 flex-wrap"):
            ui.link("README", readme_url, new_tab=True).classes(
                "text-sm text-primary underline"
            )
            ui.link("Full licence (AGPLv3)", licence_url, new_tab=True).classes(
                "text-sm text-primary underline"
            )
            ui.link(
                "Commercial licence",
                commercial_url,
                new_tab=True,
            ).classes("text-sm text-primary underline")
            ui.link(
                "Contributing guide",
                contributing_url,
                new_tab=True,
            ).classes("text-sm text-primary underline")

    return dialog


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
    _build_app_footer()


@ui.page("/")
def dashboard_page():
    """Dashboard — project gallery, examples, activity feed."""
    from shadow_loom_ui.components.dashboard import build_dashboard

    state = _get_session_state()
    apply_theme(_ui_settings.dark_mode)
    _build_app_header(state)
    build_dashboard(state)
    _build_app_footer()


@ui.page("/project/{project_id}")
def workspace_page(project_id: int):
    """Project workspace — tabbed work surface."""
    from shadow_loom_ui.components.workspace import build_workspace

    state = _get_session_state()
    apply_theme(_ui_settings.dark_mode)
    _build_app_header(state, show_back=True)
    build_workspace(state, project_id)
    _build_app_footer()


@ui.page("/settings")
def settings_page():
    """Account settings, API keys, preferences."""
    from shadow_loom_ui.components.settings import build_settings

    state = _get_session_state()
    apply_theme(_ui_settings.dark_mode)
    _build_app_header(state, show_back=True)
    build_settings(state)
    _build_app_footer()


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
