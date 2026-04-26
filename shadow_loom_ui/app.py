"""Shadow-Loom NiceGUI web application — main entry point.

Run with:  python -m shadow_loom_ui.app
Or:        nicegui run shadow_loom_ui/app.py
"""

from __future__ import annotations

import logging

from nicegui import app, ui

from shadow_loom_ui import config
from shadow_loom_ui.auth import AuthMiddleware, auth_callback, auth_login, auth_logout
from shadow_loom_ui.components import (
    build_center_panel,
    build_chat_panel,
    build_explorer,
    build_ingest_dialog,
    build_project_dialog,
    build_topology_drawer,
    build_version_dialog,
)
from shadow_loom_ui.db import init_db
from shadow_loom_ui.state import AppState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Startup
# =====================================================================

def _on_startup():
    """Initialise database and auth on app startup."""
    init_db(config.DATABASE_URL)
    logger.info("[App] Shadow-Loom UI starting on port 7860")


app.on_startup(_on_startup)

# Auth middleware (no-op when AUTH_ENABLED=False)
app.add_middleware(AuthMiddleware)

# Auth routes
app.add_route("/auth/{provider}", auth_login, methods=["GET"])
app.add_route("/auth/{provider}/callback", auth_callback, methods=["GET"])
app.add_route("/auth/logout", auth_logout, methods=["GET"])


# =====================================================================
# Main page
# =====================================================================

@ui.page("/")
def main_page():
    """Render the main 3-panel layout."""

    # Per-session state
    state = AppState()

    # Dark mode
    ui.dark_mode(True)

    # ---- Header ----
    with ui.header().classes("items-center justify-between q-pa-sm"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("auto_stories", size="md", color="primary")
            ui.label("Shadow Loom").classes("text-h6")

            if state.project_name:
                ui.label(f"— {state.project_name}").classes("text-subtitle1 text-grey")

        with ui.row().classes("items-center gap-1"):
            # Toolbar buttons
            ingest_dlg = build_ingest_dialog(state)
            ui.button("Ingest", icon="upload_file", on_click=ingest_dlg.open).props(
                "flat dense"
            )

            project_dlg = build_project_dialog(state)
            ui.button("Projects", icon="folder_open", on_click=project_dlg.open).props(
                "flat dense"
            )

            version_dlg = build_version_dialog(state)
            ui.button("History", icon="history", on_click=version_dlg.open).props(
                "flat dense"
            )

            # Save button
            async def _save():
                if state.world_state is None or state.project_id is None:
                    ui.notify("Nothing to save", type="warning")
                    return
                from shadow_loom.db import save_version as db_save_version
                ver = db_save_version(
                    project_id=state.project_id,
                    world_state_json=state.to_json(),
                    ancestor_id=state.current_version_row_id,
                    source="manual_save",
                    description="Manual save",
                )
                state.current_version_row_id = ver.id
                ui.notify(f"Saved v{ver.version}", type="positive")

            ui.button("Save", icon="save", on_click=_save).props("flat dense")

            # Auth display
            if config.AUTH_ENABLED:
                storage = app.storage.user
                if storage.get("authenticated"):
                    avatar = storage.get("avatar_url", "")
                    name = storage.get("username", "User")
                    if avatar:
                        ui.avatar().props(f'src="{avatar}"').classes("cursor-pointer")
                    ui.label(name).classes("text-body2")
                    ui.button("Logout", on_click=lambda: ui.navigate.to("/auth/logout")).props(
                        "flat dense"
                    )
                else:
                    if config.GITHUB_CLIENT_ID:
                        ui.button(
                            "Login with GitHub",
                            on_click=lambda: ui.navigate.to("/auth/github"),
                        ).props("flat dense")
                    if config.GOOGLE_CLIENT_ID:
                        ui.button(
                            "Login with Google",
                            on_click=lambda: ui.navigate.to("/auth/google"),
                        ).props("flat dense")

    # ---- Main 3-panel layout ----
    with ui.splitter(value=20).classes("w-full h-full") as main_split:
        with main_split.before:
            # Left panel: Explorer
            with ui.scroll_area().classes("w-full h-full"):
                build_explorer(state)

        with main_split.after:
            with ui.splitter(value=70).classes("w-full h-full") as right_split:
                with right_split.before:
                    # Center panel: Query + Prose + Graph + Audit
                    with ui.column().classes("w-full h-full"):
                        build_center_panel(state)

                        # Bottom topology drawer
                        with ui.expansion("Topology Edges", icon="device_hub").classes("w-full"):
                            build_topology_drawer(state)

                with right_split.after:
                    # Right panel: Chat
                    with ui.scroll_area().classes("w-full h-full"):
                        build_chat_panel(state)

    # ---- Footer ----
    with ui.footer().classes("q-pa-xs items-center"):
        ui.label("Shadow Loom — Causal Narrative Engine").classes("text-caption text-grey")
        ui.space()
        if state.versioned_model:
            ui.label(f"v{state.versioned_model.version}").classes("text-caption text-grey")


# =====================================================================
# Entry point
# =====================================================================

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=7860,
        title="Shadow Loom",
        storage_secret=config.STORAGE_SECRET,
        dark=True,
        reload=False,
    )
