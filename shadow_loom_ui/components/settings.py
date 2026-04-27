"""Settings page — profile, API keys, connected accounts, preferences."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import app, ui

from shadow_loom_ui import config, db
from shadow_loom_ui.state import AppState

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_settings(state: AppState) -> None:
    """Build the settings page layout."""

    if not state.user_id:
        ui.label("Sign in to access settings.").classes("text-h5 q-pa-lg text-grey")
        return

    user = db.get_user(state.user_id)
    if user is None:
        ui.label("User not found.").classes("text-h5 q-pa-lg text-negative")
        return

    with ui.column().classes("w-full max-w-3xl mx-auto q-pa-lg gap-6"):
        ui.label("Settings").classes("text-h4")

        # ---- Profile ----
        with ui.card().classes("w-full"):
            ui.label("Profile").classes("text-h6")

            with ui.row().classes("items-center gap-4 q-mb-md"):
                if user.avatar_url:
                    ui.avatar(size="xl").props(f'src="{user.avatar_url}"')
                with ui.column():
                    ui.label(user.username).classes("text-subtitle1")
                    ui.label(user.email or "").classes("text-caption text-grey")
                    if user.provider:
                        ui.badge(f"via {user.provider}", color="blue-grey").props("dense")

            display_name_input = ui.input(
                "Display Name", value=user.display_name or ""
            ).classes("w-full")
            bio_input = ui.textarea("Bio", value=user.bio or "").classes("w-full").props("rows=3")

            def _save_profile():
                db.update_user_profile(
                    state.user_id,
                    display_name=display_name_input.value or None,
                    bio=bio_input.value or None,
                )
                state.display_name = display_name_input.value or state.username
                storage = app.storage.user
                storage["display_name"] = state.display_name
                ui.notify("Profile saved!", type="positive")

            ui.button("Save Profile", icon="save", on_click=_save_profile).props(
                "color=primary no-caps"
            )

        # ---- Connected Accounts ----
        with ui.card().classes("w-full"):
            ui.label("Connected Accounts").classes("text-h6")
            ui.label(
                "Your account is linked to the provider you signed in with. "
                "Additional provider linking is not yet available."
            ).classes("text-body2 text-grey")

            with ui.row().classes("gap-2"):
                if user.provider:
                    ui.chip(user.provider.title(), icon="link").props("color=primary")

        # ---- API Keys ----
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("vpn_key", size="md", color="primary")
                ui.label("API Keys").classes("text-h6")

            ui.label(
                "API keys allow external tools and MCP clients to access your projects "
                "via Bearer token authentication."
            ).classes("text-body2 text-grey q-mb-md")

            # Create new key
            with ui.row().classes("items-center gap-2"):
                key_name_input = ui.input("Key name", placeholder="e.g., My MCP Client").classes(
                    "w-64"
                )
                scope_select = ui.select(
                    ["read", "read,write", "read,write,admin"],
                    value="read,write",
                    label="Scopes",
                ).classes("w-48")

                def _create_key():
                    name = key_name_input.value.strip()
                    if not name:
                        ui.notify("Enter a key name", type="warning")
                        return
                    _row, raw_key = db.create_api_key(
                        user_id=state.user_id,
                        name=name,
                        scopes=scope_select.value,
                    )
                    key_name_input.value = ""
                    # Show the key once
                    _show_new_key_dialog(raw_key)
                    _refresh_keys()

                ui.button("Generate Key", icon="add", on_click=_create_key).props(
                    "no-caps color=primary"
                )

            # Key list
            keys_container = ui.column().classes("w-full q-mt-md")

            def _refresh_keys():
                keys_container.clear()
                keys = db.list_api_keys(state.user_id)
                if not keys:
                    with keys_container:
                        ui.label("No API keys yet.").classes("text-body2 text-grey")
                    return

                with keys_container:
                    with ui.list().props("bordered separator").classes("w-full"):
                        for key in keys:
                            with ui.item():
                                with ui.item_section():
                                    with ui.row().classes("items-center gap-2"):
                                        ui.item_label(key["name"]).classes("text-bold")
                                        ui.badge(key["key_prefix"], color="blue-grey").props(
                                            "dense"
                                        )
                                        for scope in (key["scopes"] or "").split(","):
                                            if scope:
                                                ui.badge(scope, color="teal").props("dense outline")
                                        if not key["is_active"]:
                                            ui.badge("revoked", color="negative").props("dense")

                                    meta_parts = []
                                    if key.get("created_at"):
                                        meta_parts.append(
                                            f"Created: {str(key['created_at'])[:10]}"
                                        )
                                    if key.get("last_used_at"):
                                        meta_parts.append(
                                            f"Last used: {str(key['last_used_at'])[:10]}"
                                        )
                                    if meta_parts:
                                        ui.item_label(" · ".join(meta_parts)).props("caption")

                                with ui.item_section().props("side"):
                                    if key["is_active"]:
                                        def _revoke(kid=key["id"]):
                                            db.revoke_api_key(kid, state.user_id)
                                            _refresh_keys()
                                            ui.notify("Key revoked")

                                        ui.button(
                                            "Revoke", icon="delete", on_click=_revoke
                                        ).props("flat dense color=negative")

            _refresh_keys()

        # ---- Preferences ----
        with ui.card().classes("w-full"):
            ui.label("Preferences").classes("text-h6")
            ui.label("Coming soon: default model, theme, pipeline configuration.").classes(
                "text-body2 text-grey"
            )


def _show_new_key_dialog(raw_key: str) -> None:
    """Show a dialog with the newly created API key (shown only once)."""
    with ui.dialog() as dlg, ui.card().classes("w-96"):
        ui.label("API Key Created").classes("text-h6")
        ui.label(
            "Copy this key now — it will not be shown again."
        ).classes("text-body2 text-warning q-mb-sm")

        with ui.row().classes("items-center gap-2 w-full"):
            key_field = ui.input(value=raw_key).classes("flex-grow").props("readonly outlined dense")

            def _copy():
                import json as _json
                ui.run_javascript(
                    f"navigator.clipboard.writeText({_json.dumps(raw_key)})"
                )
                ui.notify("Copied!", type="positive")

            ui.button(icon="content_copy", on_click=_copy).props("flat dense")

        ui.button("Done", on_click=dlg.close).props("flat").classes("q-mt-sm")

    dlg.open()
