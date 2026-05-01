# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Settings page — profile, API keys, connected accounts, preferences."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import app, ui

from shadow_loom_ui import config, db
from shadow_loom_ui.state import AppState
from shadow_loom_ui.theme import (
    CARD_CLS,
    PAGE_TITLE_CLS,
    SECTION_TITLE_CLS,
    feather,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_settings(state: AppState) -> None:
    """Build the settings page layout."""

    if not state.user_id:
        ui.label("Sign in to access settings.").classes(
            "text-2xl text-slate-500 q-pa-lg"
        )
        return

    user = db.get_user(state.user_id)
    if user is None:
        ui.label("User not found.").classes(
            "text-2xl text-negative q-pa-lg"
        )
        return

    with ui.column().classes("w-full max-w-3xl mx-auto p-6 md:p-8 gap-6"):
        ui.label("Settings").classes(PAGE_TITLE_CLS)

        # ---- Profile ----
        with ui.card().classes("w-full " + CARD_CLS):
            ui.label("Profile").classes(SECTION_TITLE_CLS)

            with ui.row().classes("items-center gap-4 q-mb-md"):
                if user.avatar_url:
                    ui.avatar(size="xl").props(f'src="{user.avatar_url}"')
                with ui.column():
                    ui.label(user.username).classes(
                        "text-base font-semibold text-slate-800"
                    )
                    ui.label(user.email or "").classes(
                        "text-xs text-slate-500"
                    )
                    if user.provider:
                        ui.badge(f"via {user.provider}", color="secondary").props("dense")

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

            with ui.button(on_click=_save_profile).props(
                "unelevated color=primary no-caps"
            ).classes("rounded-lg shadow-sm"):
                with ui.row().classes("items-center gap-2"):
                    feather("save")
                    ui.label("Save Profile")

        # ---- Connected Accounts ----
        with ui.card().classes("w-full " + CARD_CLS):
            ui.label("Connected Accounts").classes(SECTION_TITLE_CLS)
            ui.label(
                "Your account is linked to the provider you signed in with. "
                "Additional provider linking is not yet available."
            ).classes("text-sm text-slate-500")

            with ui.row().classes("gap-2"):
                if user.provider:
                    ui.chip(user.provider.title(), icon="link").props(
                        "color=primary outline"
                    )

        # ---- API Keys ----
        with ui.card().classes("w-full " + CARD_CLS):
            with ui.row().classes("items-center gap-2"):
                feather("key", size="lg", color="#F26B5E")
                ui.label("API Keys").classes(SECTION_TITLE_CLS)

            ui.label(
                "API keys allow external tools and MCP clients to access your projects "
                "via Bearer token authentication."
            ).classes("text-sm text-slate-500 mb-4")

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

                with ui.button(on_click=_create_key).props(
                    "unelevated color=primary no-caps"
                ).classes("rounded-lg shadow-sm"):
                    with ui.row().classes("items-center gap-2"):
                        feather("plus")
                        ui.label("Generate Key")

            # Key list
            keys_container = ui.column().classes("w-full q-mt-md")

            def _refresh_keys():
                keys_container.clear()
                keys = db.list_api_keys(state.user_id)
                if not keys:
                    with keys_container:
                        ui.label("No API keys yet.").classes(
                            "text-sm text-slate-400 italic"
                        )
                    return

                with keys_container:
                    with ui.list().props("bordered separator").classes("w-full"):
                        for key in keys:
                            with ui.item():
                                with ui.item_section():
                                    with ui.row().classes("items-center gap-2"):
                                        ui.item_label(key["name"]).classes(
                                            "text-bold text-slate-800"
                                        )
                                        ui.badge(key["key_prefix"], color="secondary").props(
                                            "dense"
                                        )
                                        for scope in (key["scopes"] or "").split(","):
                                            if scope:
                                                ui.badge(scope, color="primary").props(
                                                    "dense outline"
                                                )
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

                                        with ui.button(on_click=_revoke).props(
                                            "flat dense color=negative no-caps"
                                        ):
                                            with ui.row().classes("items-center gap-1"):
                                                feather("trash-2", size="sm")
                                                ui.label("Revoke").classes("text-xs")

            _refresh_keys()

        # ---- Preferences ----
        with ui.card().classes("w-full " + CARD_CLS):
            ui.label("Preferences").classes(SECTION_TITLE_CLS)
            ui.label(
                "Coming soon: default model, theme, pipeline configuration."
            ).classes("text-sm text-slate-500")

        # ---- Pipeline configuration (read-only view of resolved settings) ----
        with ui.card().classes("w-full " + CARD_CLS):
            ui.label("Pipeline configuration").classes(SECTION_TITLE_CLS)
            ui.label(
                "Resolved values from environment / config.env. Edit your "
                "config.env or the matching env vars and restart to change."
            ).classes("text-xs text-slate-500 mb-2")
            from shadow_loom.settings import get_settings as _get_settings
            _s = _get_settings()
            rows = [
                {
                    "name": "physics.intelligibility_threshold",
                    "value": f"{_s.physics.intelligibility_threshold:.2f}",
                    "help": (
                        "Per-recipient channel intelligibility below which "
                        "a belief acquired through that channel is considered "
                        "epistemically invalid."
                    ),
                },
                {
                    "name": "physics.max_ingest_words",
                    "value": f"{_s.physics.max_ingest_words:,}",
                    "help": (
                        "Maximum whitespace-tokenised words accepted by any "
                        "ingestion entry point (UI, sample loader, MCP)."
                    ),
                },
                {
                    "name": "core.default_model",
                    "value": _s.core.default_model,
                    "help": "Fallback PydanticAI model string.",
                },
            ]
            ui.table(
                columns=[
                    {"name": "name", "label": "Setting", "field": "name", "align": "left"},
                    {"name": "value", "label": "Value", "field": "value", "align": "left"},
                    {"name": "help", "label": "Description", "field": "help", "align": "left"},
                ],
                rows=rows,
            ).props("dense flat bordered").classes("w-full")


def _show_new_key_dialog(raw_key: str) -> None:
    """Show a dialog with the newly created API key (shown only once)."""
    with ui.dialog() as dlg, ui.card().classes(
        "w-96 bg-white border border-slate-200 rounded-xl shadow-sm p-6"
    ):
        ui.label("API Key Created").classes("text-lg font-semibold text-slate-800")
        ui.label(
            "Copy this key now \u2014 it will not be shown again."
        ).classes("text-sm text-warning mb-3")

        with ui.row().classes("items-center gap-2 w-full"):
            key_field = ui.input(value=raw_key).classes("flex-grow").props(
                "readonly outlined dense"
            )

            def _copy():
                import json as _json
                ui.run_javascript(
                    f"navigator.clipboard.writeText({_json.dumps(raw_key)})"
                )
                ui.notify("Copied!", type="positive")

            with ui.button(on_click=_copy).props("flat dense color=secondary"):
                feather("copy")

        ui.button("Done", on_click=dlg.close).props(
            "unelevated color=primary no-caps"
        ).classes("rounded-lg mt-3")

    dlg.open()
