# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Settings page — profile, API keys, connected accounts, preferences."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import app, ui

from shadow_loom_ui import db
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

        # ---- Models & Providers ----
        _build_models_and_providers(state)

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


# =====================================================================
# Models & Providers — per-user LLM configuration
# =====================================================================
#
# The deployment ships with a ``DEFAULT_MODEL`` env var as a fallback;
# every signed-in user can override it (and any per-stage model) from
# this card. Saved values persist in ``UserModelSettingsRow`` and are
# activated for every pipeline call via
# :func:`shadow_loom.settings.set_user_context`.
#
# Users may also register their own OpenAI-compatible providers (e.g. a
# private llama.cpp endpoint or a paid Mistral key). Once added, models
# can be referenced as ``<prefix>:<model-id>`` in any of the model
# fields below.


_RECOGNISED_STAGE_LABELS: list[tuple[str, str]] = [
    ("generation",         "Generation (prose rendering)"),
    ("auditor",            "Auditor (LLM-as-judge)"),
    ("auditor_generation", "Auditor re-renders"),
    ("extraction",         "Extraction (Steps 3, 6, 7)"),
    ("query_parsing",      "Query parsing (Step 0)"),
]


def _build_models_and_providers(state: AppState) -> None:
    """Card with default-model / per-stage / custom-provider editors."""
    from shadow_loom.settings import (
        get_settings as _get_settings,
        get_openai_compat_providers as _get_providers,
    )

    saved = db.get_user_model_settings(state.user_id)
    env_default = _get_settings().core.default_model or "(unset)"

    with ui.card().classes("w-full " + CARD_CLS):
        with ui.row().classes("items-center gap-2"):
            feather("cpu", size="lg", color="#3E63DD")
            ui.label("Models & Providers").classes(SECTION_TITLE_CLS)

        ui.label(
            "Your settings override the deployment's environment defaults. "
            "Leave a field blank to inherit the deployment value."
        ).classes("text-sm text-slate-500 mb-3")

        # ── Default model ──────────────────────────────────────────
        default_input = ui.input(
            "Default model",
            value=saved["default_model"],
            placeholder=f"e.g. openrouter:openai/gpt-4o-mini  (env: {env_default})",
        ).classes("w-full").props("outlined dense")
        ui.label(
            "Format: <provider>:<model-id>. Built-in providers: "
            + ", ".join(sorted(_get_providers().keys()))
            + ". Add custom providers below to extend this list."
        ).classes("text-xs text-slate-400 mb-3")

        # ── Per-stage overrides ────────────────────────────────────
        with ui.expansion("Per-stage overrides", icon="tune").classes(
            "w-full border border-slate-200 rounded-lg q-mb-md"
        ).props("dense"):
            ui.label(
                "Override the default model for individual pipeline stages. "
                "Leave blank to use your Default model above."
            ).classes("text-xs text-slate-500 mb-2")
            stage_inputs: dict[str, "ui.input"] = {}
            for key, label in _RECOGNISED_STAGE_LABELS:
                stage_inputs[key] = ui.input(
                    label,
                    value=saved["stage_models"].get(key, ""),
                    placeholder="(use default)",
                ).classes("w-full").props("outlined dense")

        # ── Custom providers ───────────────────────────────────────
        ui.separator().classes("my-3")
        ui.label("Custom providers").classes("font-semibold text-slate-700")
        ui.label(
            "Register any OpenAI-compatible endpoint. Once saved, reference "
            "the provider as ``<prefix>:<model-id>`` in any model field."
        ).classes("text-xs text-slate-500 mb-2")

        providers_container = ui.column().classes("w-full gap-2")
        provider_rows: list[dict] = list(saved["custom_providers"])

        def _render_providers() -> None:
            providers_container.clear()
            with providers_container:
                if not provider_rows:
                    ui.label("No custom providers yet.").classes(
                        "text-xs text-slate-400 italic"
                    )
                for idx, prov in enumerate(provider_rows):
                    with ui.row().classes("w-full items-center gap-2"):
                        prefix_input = ui.input(
                            "prefix",
                            value=prov.get("prefix", ""),
                            placeholder="myprov",
                        ).classes("w-32").props("outlined dense")
                        url_input = ui.input(
                            "base URL",
                            value=prov.get("base_url", ""),
                            placeholder="https://api.example.com/v1",
                        ).classes("flex-grow").props("outlined dense")
                        key_input = ui.input(
                            "API key",
                            value=prov.get("api_key", ""),
                            password=True,
                            password_toggle_button=True,
                        ).classes("w-48").props("outlined dense")
                        local_cb = ui.checkbox(
                            "local",
                            value=bool(prov.get("is_local", False)),
                        )

                        # Bind on change so unsaved edits are captured
                        # in ``provider_rows`` before Save fires.
                        def _bind(i=idx, p=prefix_input, u=url_input,
                                  k=key_input, lc=local_cb):
                            def _update(*_a):
                                provider_rows[i] = {
                                    "prefix": p.value or "",
                                    "base_url": u.value or "",
                                    "api_key": k.value or "",
                                    "is_local": bool(lc.value),
                                }
                            for w in (p, u, k, lc):
                                w.on("update:model-value", _update)
                            return _update
                        _bind()

                        def _remove(i=idx):
                            provider_rows.pop(i)
                            _render_providers()

                        ui.button(icon="delete", on_click=_remove).props(
                            "flat dense color=negative"
                        )

        _render_providers()

        def _add_provider():
            provider_rows.append({
                "prefix": "",
                "base_url": "",
                "api_key": "",
                "is_local": False,
            })
            _render_providers()

        with ui.row().classes("items-center gap-2 mt-2"):
            with ui.button(on_click=_add_provider).props(
                "outline dense color=primary no-caps"
            ):
                with ui.row().classes("items-center gap-1"):
                    feather("plus")
                    ui.label("Add provider")

        # ── Save ───────────────────────────────────────────────────
        def _save_models():
            stage_models = {
                key: (inp.value or "").strip()
                for key, inp in stage_inputs.items()
                if (inp.value or "").strip()
            }
            try:
                db.set_user_model_settings(
                    state.user_id,
                    default_model=(default_input.value or "").strip(),
                    stage_models=stage_models,
                    custom_providers=provider_rows,
                )
            except ValueError as e:
                ui.notify(f"Save failed: {e}", type="negative")
                return
            ui.notify("Models & providers saved.", type="positive")

        with ui.button(on_click=_save_models).props(
            "unelevated color=primary no-caps"
        ).classes("rounded-lg shadow-sm mt-3"):
            with ui.row().classes("items-center gap-2"):
                feather("save")
                ui.label("Save Models & Providers")
