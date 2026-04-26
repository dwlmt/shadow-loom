"""Login page component — OAuth provider buttons + guest mode."""

from __future__ import annotations

from nicegui import app, ui

from shadow_loom_ui import config


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
            ui.icon("auto_stories", size="xl", color="primary")
            ui.label("Shadow Loom").classes("text-h3")

        ui.label("Causal Narrative Engine").classes("text-subtitle1 text-grey")

        # Provider buttons
        with ui.card().classes("w-80 q-pa-lg"):
            ui.label("Sign in to continue").classes("text-h6 q-mb-md")

            providers = config.OAUTH_PROVIDERS
            if providers:
                for prov in providers:
                    name = prov["name"]
                    label = prov["label"]
                    icon = prov["icon"]
                    ui.button(
                        f"Continue with {label}",
                        icon=icon,
                        on_click=lambda n=name: ui.navigate.to(f"/auth/{n}"),
                    ).props("outline no-caps").classes("w-full q-mb-sm")
            else:
                ui.label("No OAuth providers configured.").classes("text-body2 text-grey")

            ui.separator().classes("q-my-md")

            # Guest mode (when auth is optional)
            if not config.AUTH_ENABLED:
                ui.button(
                    "Continue as Guest",
                    icon="person_outline",
                    on_click=lambda: ui.navigate.to("/"),
                ).props("flat no-caps").classes("w-full")
            else:
                ui.label(
                    "Contact your administrator if you need access."
                ).classes("text-caption text-grey text-center")
