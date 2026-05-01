# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Login page component — OAuth provider buttons + guest mode."""

from __future__ import annotations

from nicegui import app, ui

from shadow_loom_ui import config
from shadow_loom_ui.theme import PRIMARY, feather


def build_login_page() -> None:
    """Render a centered login card with configured OAuth providers."""

    # If already authenticated, redirect to dashboard
    storage = app.storage.user
    if storage.get("authenticated"):
        ui.navigate.to("/")
        return

    with ui.column().classes("absolute-center items-center gap-6"):
        # Branding
        with ui.row().classes("items-center gap-3"):
            feather("book-open", size="xl", color=PRIMARY)
            ui.label("Shadow Loom").classes(
                "text-4xl font-bold tracking-tight text-slate-800"
            )

        ui.label("Causal Narrative Engine").classes(
            "text-sm font-medium text-slate-500"
        )

        # Provider buttons
        with ui.card().classes(
            "w-80 bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            ui.label("Sign in to continue").classes(
                "text-base font-semibold text-slate-800 mb-4"
            )

            providers = config.OAUTH_PROVIDERS
            if providers:
                for prov in providers:
                    name = prov["name"]
                    label = prov["label"]
                    icon = prov["icon"]
                    with ui.button(
                        on_click=lambda n=name: ui.navigate.to(f"/auth/{n}"),
                    ).props("outline no-caps color=primary").classes(
                        "w-full mb-2 rounded-lg"
                    ):
                        with ui.row().classes("items-center gap-2 w-full justify-center"):
                            feather(icon)
                            ui.label(f"Continue with {label}")
            else:
                ui.label("No OAuth providers configured.").classes(
                    "text-sm text-slate-500"
                )

            ui.separator().classes("q-my-md")

            # Guest mode (when auth is optional)
            if not config.AUTH_ENABLED:
                with ui.button(
                    on_click=lambda: ui.navigate.to("/"),
                ).props("flat no-caps color=secondary").classes("w-full"):
                    with ui.row().classes("items-center gap-2 w-full justify-center"):
                        feather("user")
                        ui.label("Continue as Guest")
            else:
                ui.label(
                    "Contact your administrator if you need access."
                ).classes("text-xs text-slate-400 text-center")
