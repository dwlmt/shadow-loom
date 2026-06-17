# SPDX-FileCopyrightText: 2026 David Rae Wilmot
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
app.add_route("/auth/{provider}/callback", auth_callback, methods=["GET", "POST"])
app.add_route("/auth/logout", auth_logout, methods=["POST"])


# =====================================================================
# Helpers
# =====================================================================

# In-memory cache of AppState keyed by browser session id.
# AppState contains non-JSON-serialisable objects (PipelineConfig, callbacks,
# world models, etc.), so it cannot live in app.storage.user which is
# persisted to disk as JSON.
_SESSION_STATES: dict[str, AppState] = {}
# Round-6 audit: track per-session last-access timestamps so we can
# evict idle sessions. Without this, ``_SESSION_STATES`` grows
# unboundedly — every anon visitor adds a new ``AppState`` (carrying a
# loaded world model) that never gets reclaimed for the life of the
# process.
_SESSION_LAST_TOUCH: dict[str, float] = {}
# Soft cap; oldest-touched sessions evicted past this number.
_SESSION_STATES_MAX = 256
# Idle TTL: sessions untouched for longer than this are evicted on the
# next ``_get_session_state`` call.
_SESSION_IDLE_TTL_SECONDS = 60 * 60 * 6  # 6h
# Round-13 R13-01: serialise mutations to the session registry. NiceGUI
# routes can run concurrently on the event loop and in worker threads
# (auth callbacks, background renders), so the previous check-then-act
# pattern in ``_get_session_state`` could race two requests for the
# same browser id into creating two ``AppState`` objects, or have an
# eviction sweep drop a state that another handler had just created
# but not yet stored in ``_SESSION_LAST_TOUCH``. A single re-entrant
# lock around the registry is enough — the critical sections are very
# short (dict ops + a cheap ``AppState()`` constructor), and AppState
# itself owns its own internal locking for runtime mutations.
import threading as _threading
_SESSION_REGISTRY_LOCK = _threading.RLock()


def _evict_idle_sessions(now: float) -> None:
    """Drop sessions untouched past the idle TTL, then cap total count."""
    with _SESSION_REGISTRY_LOCK:
        stale = [
            sid for sid, t in _SESSION_LAST_TOUCH.items()
            if now - t > _SESSION_IDLE_TTL_SECONDS
        ]
        for sid in stale:
            # Round-11 R11-03: tear down session-owned background work
            # (asyncio tasks + deferred-reextraction threads) before
            # dropping the AppState. Without this, evicted sessions leak
            # workers that keep writing to now-orphaned state.
            st = _SESSION_STATES.get(sid)
            if st is not None:
                try:
                    st.teardown()
                except Exception:
                    logger.debug("session teardown failed for %s", sid, exc_info=True)
            _SESSION_STATES.pop(sid, None)
            _SESSION_LAST_TOUCH.pop(sid, None)
        overflow = len(_SESSION_STATES) - _SESSION_STATES_MAX
        if overflow > 0:
            # Evict oldest-touched first.
            victims = sorted(
                _SESSION_LAST_TOUCH.items(), key=lambda kv: kv[1]
            )[:overflow]
            for sid, _ in victims:
                st = _SESSION_STATES.get(sid)
                if st is not None:
                    try:
                        st.teardown()
                    except Exception:
                        logger.debug(
                            "session teardown failed for %s", sid, exc_info=True,
                        )
                _SESSION_STATES.pop(sid, None)
                _SESSION_LAST_TOUCH.pop(sid, None)


def _get_session_state() -> AppState:
    """Return the per-session AppState, creating it if needed.

    Kept in a module-level dict keyed by the browser session id from
    ``app.storage.browser``. Auth info is sourced from ``app.storage.user``
    (which is JSON-safe) on first access.

    **R19-L6 (documented invariant — tab sharing):** This intentionally
    keys on ``app.storage.browser['id']`` (one entry per browser
    session) rather than per-tab/per-client. Opening the same project
    in two tabs therefore points both tabs at the *same* AppState
    instance; mutations in tab B (e.g. running a new query, swapping
    project, promoting a branch) propagate to tab A on its next
    render/refresh. This is by design — concurrent independent
    workflows in two tabs are NOT supported, and users wanting
    isolated workspaces should use a private/incognito window
    (separate browser session id). Switch to per-client keying via
    ``app.storage.client`` / a Quasar tab id if independent tab
    workflows ever become a product requirement.
    """
    storage = app.storage.user
    session_id = app.storage.browser.get("id", "")
    # Reject / replace an empty session id. Without this, every caller
    # that arrives before ``app.storage.browser`` has been initialised
    # collapses onto the SAME AppState entry under the ``""`` key,
    # cross-leaking user/project/version state between sessions
    # (round-3 audit \u2014 multi-user collapse).
    if not session_id:
        import uuid as _uuid
        session_id = f"anon-{_uuid.uuid4().hex}"
        try:
            app.storage.browser["id"] = session_id
        except Exception:  # noqa: BLE001
            # Browser storage may be read-only in some contexts \u2014
            # the generated id still keeps THIS request isolated.
            pass
    # Drop any legacy non-serialisable state that older builds may have left
    # in the persisted user storage.
    if "_state" in storage:
        storage.pop("_state", None)
    # Round-6 audit: opportunistic eviction sweep before touch.
    import time as _time
    _now = _time.monotonic()
    _evict_idle_sessions(_now)
    # Round-13 R13-01: hold the registry lock across the
    # check-then-create sequence so two concurrent requests for the
    # same session cannot each build a fresh AppState (which would
    # silently discard one of them when the second .setdefault won).
    with _SESSION_REGISTRY_LOCK:
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
        _SESSION_LAST_TOUCH[session_id] = _now
    return state


def _build_app_header(state: AppState, *, show_back: bool = False):
    """Render the shared top navigation bar."""
    with ui.header().classes(
        "bg-white border-b border-slate-200 px-4 py-1 min-h-0 "
        "flex items-center justify-between text-slate-800"
    ).props("elevated=false flat"):
        with ui.row().classes("items-center gap-2"):
            if show_back:
                with ui.button(on_click=lambda: ui.navigate.to("/")).props(
                    "flat dense round color=secondary size=sm"
                ):
                    feather("arrow-left")
            with ui.row().classes("items-center gap-2 cursor-pointer").on(
                "click", lambda: ui.navigate.to("/")
            ):
                feather("book-open", size="sm", color=_brand_copper())
                ui.label("Shadow Loom").classes(
                    "text-sm font-semibold text-slate-800 leading-none"
                )

            if state.project_name:
                ui.label("/").classes("text-sm text-slate-400 leading-none")
                ui.label(state.project_name).classes(
                    "text-sm font-medium text-slate-700 leading-none"
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
                    ui.avatar(size="sm").props(f'src="{avatar}"').classes(
                        "cursor-pointer"
                    ).on("click", lambda: ui.navigate.to("/settings"))
                ui.label(name).classes(
                    "text-xs font-medium text-slate-700 cursor-pointer hidden sm:block leading-none"
                ).on("click", lambda: ui.navigate.to("/settings"))
                with ui.button(on_click=lambda: ui.navigate.to("/settings")).props(
                    "flat dense round color=secondary size=sm"
                ):
                    feather("settings")
                if authed:
                    with ui.button(
                        on_click=lambda: ui.run_javascript(
                            "fetch('/auth/logout', {method: 'POST', "
                            "credentials: 'same-origin'}).then("
                            "() => window.location.href = '/login');"
                        )
                    ).props("flat dense round color=secondary size=sm"):
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
            "Shadow Loom — AGPL-3.0-or-later, with a commercial licence "
            "available. This is free software with NO WARRANTY."
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
    content_policy_url = f"{base}/blob/main/CONTENT-POLICY.md"

    # ✓ allowed under AGPLv3, ✗ requires commercial licence / forbidden,
    # ⚠ allowed but with obligations.
    permissions = [
        ("check_circle", "positive", "Use it for personal projects, research, learning, and commercial work — provided you comply with AGPLv3."),
        ("check_circle", "positive", "Read, modify, and fork the source code."),
        ("check_circle", "positive", "Run it on your own machine without restriction."),
        ("check_circle", "positive", "Redistribute it — as long as you keep it under AGPL-3.0-or-later and convey the corresponding source under AGPLv3 § 5."),
        (
            "warning",
            "warning",
            "Host it as a network service (SaaS, MCP server, hosted UI): "
            "AGPLv3 § 13 requires you to offer the complete corresponding "
            "source code (including your modifications) to your users "
            "under AGPLv3.",
        ),
        (
            "warning",
            "warning",
            "Embed or link it into a larger product you convey: the whole "
            "combined work must also be released under AGPLv3.",
        ),
        (
            "cancel",
            "negative",
            "Deploy it as a network service without complying with AGPLv3 § 13 "
            "(network-use disclosure) — that requires a paid commercial licence.",
        ),
        (
            "cancel",
            "negative",
            "Re-license it under a more permissive licence, or ship it "
            "inside closed-source software, without a commercial licence.",
        ),
    ]

    creator_items = [
        (
            "check_circle",
            "positive",
            "Own everything you create — your inputs, world models, "
            "creative briefs, rendered scenes, audit reports, and exports. "
            "The maintainer asserts no copyright over your output.",
        ),
        (
            "check_circle",
            "positive",
            "Publish and sell your output commercially. No royalty, no "
            "attribution to Shadow Loom required.",
        ),
        (
            "check_circle",
            "positive",
            "Write the full range of adult fiction — graphic violence, "
            "sexuality between fictional adults, crime, war, drug use, "
            "dark and morally complex themes, real public figures in "
            "fictional / satirical / historical contexts. The auditor "
            "checks consistency, not taste.",
        ),
        (
            "warning",
            "warning",
            "You must be 13 or over to use the hosted service; under-18s "
            "require parent or guardian permission. Individual LLM "
            "providers may impose stricter age limits.",
        ),
        (
            "warning",
            "warning",
            "You are responsible for the rights to anything you ingest. "
            "The hosted service does not screen inputs and accepts no "
            "liability for any copyright, defamation, privacy, or "
            "publicity claim arising from your inputs or outputs.",
        ),
        (
            "warning",
            "warning",
            "Disclose AI assistance when your publication venue requires "
            "it — KDP, many literary magazines, SFWA guidance, EU AI Act "
            "Art. 50 from August 2026. Disclosure is your call.",
        ),
        (
            "warning",
            "warning",
            "Review every output before publication. Generated text may "
            "be inaccurate, contradictory, or derivative. The auditor's "
            "\"passes\" verdict is a narrative-consistency check, not a "
            "legal or copyright clearance.",
        ),
        (
            "cancel",
            "negative",
            "No sexual content involving minors, in any framing.",
        ),
        (
            "cancel",
            "negative",
            "No working CBRN / explosive synthesis instructions or "
            "functional malware dressed as fiction. Depicting such things "
            "exist in your story is fine; providing a working recipe is not.",
        ),
        (
            "cancel",
            "negative",
            "No non-consensual sexual content, deepfakes, doxxing, or "
            "fabricated criminal accusations targeting a real, identifiable "
            "living person.",
        ),
        (
            "cancel",
            "negative",
            "No direct, credible incitement to violence against a real, "
            "identifiable person or group.",
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

        ui.label("For creators — your content").classes(
            "text-sm font-semibold text-slate-700 mt-2"
        )
        with ui.column().classes("gap-1 w-full"):
            for icon_name, color, text in creator_items:
                with ui.row().classes("items-start gap-2 w-full no-wrap"):
                    ui.icon(icon_name, color=color).classes("mt-0.5")
                    ui.label(text).classes("text-sm text-slate-700 flex-1")
        ui.label(
            "The hosted service does not train models on your content. "
            "Prompts are routed to a third-party LLM provider and are "
            "subject to that provider's retention policy."
        ).classes("text-xs text-slate-500 mt-1")

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
                "Content policy",
                content_policy_url,
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
        session_middleware_kwargs=config.SESSION_MIDDLEWARE_KWARGS,
        dark=_ui_settings.dark_mode,
        reload=_ui_settings.reload,
    )
