# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Export tab — export prose/world state, share project, fork, invite collaborators."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui import db
from shadow_loom_ui.state import AppState, StateEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_export_tab(state: AppState) -> None:
    """Build the Export tab layout."""

    with ui.column().classes("w-full h-full p-6 gap-4 bg-slate-50"):
        # ---- Export Prose ----
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("article", size="md", color="primary")
                ui.label("Export Prose").classes(
                    "text-lg font-semibold text-slate-800"
                )

            ui.label("Download all generated prose as Markdown.").classes(
                "text-sm text-slate-500"
            )

            async def _export_prose():
                if state.project_id is None:
                    ui.notify("No project loaded", type="warning")
                    return
                prose_list = db.get_all_prose(state.project_id)
                if not prose_list:
                    ui.notify("No prose found", type="info")
                    return
                md = f"# {state.project_name}\n\n"
                for i, entry in enumerate(prose_list):
                    md += f"## Version {entry['version']} ({entry['source']})\n\n{entry['prose']}\n\n---\n\n"

                ui.download(
                    md.encode("utf-8"),
                    filename=f"{state.project_name.replace(' ', '_')}_prose.md",
                )
                ui.notify("Prose exported!", type="positive")

            ui.button("Download Prose (Markdown)", icon="download", on_click=_export_prose).props(
                "unelevated no-caps color=primary"
            ).classes("rounded-lg shadow-sm")

        # ---- Export Current Version (Prose + World Model) ----
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("inventory_2", size="md", color="primary")
                ui.label("Export Current Version (JSON)").classes(
                    "text-lg font-semibold text-slate-800"
                )

            ui.label(
                "Download the currently loaded world model version with both "
                "its prose and world model JSON in a single file."
            ).classes("text-sm text-slate-500")

            def _build_version_payload() -> dict | None:
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return None
                row = None
                if state.current_version_row_id is not None:
                    row = db.get_version_by_id(state.current_version_row_id)
                elif state.project_id is not None:
                    row = db.get_latest_version(state.project_id)
                payload: dict = {
                    "project_id": state.project_id,
                    "project_name": state.project_name,
                    "world_model": state.world_state.model_dump(mode="json"),
                }
                if row is not None:
                    payload["version"] = {
                        "version_row_id": row.id,
                        "version": row.version,
                        "ancestor_id": row.ancestor_id,
                        "source": row.source,
                        "description": row.description,
                        "label": row.label,
                        "is_bookmarked": row.is_bookmarked,
                        "world_id": row.world_id,
                        "branch_label": row.branch_label,
                        "raw_query": row.raw_query,
                        "created_at": str(row.created_at),
                    }
                    payload["prose"] = row.prose
                else:
                    payload["version"] = None
                    payload["prose"] = None
                return payload

            def _export_version_json():
                payload = _build_version_payload()
                if payload is None:
                    return
                data = json.dumps(payload, indent=2, default=str)
                version_label = ""
                if payload.get("version"):
                    version_label = f"_v{payload['version']['version']}"
                ui.download(
                    data.encode("utf-8"),
                    filename=(
                        f"{state.project_name.replace(' ', '_')}"
                        f"{version_label}_version.json"
                    ),
                )
                ui.notify("Version exported!", type="positive")

            def _copy_version_json():
                payload = _build_version_payload()
                if payload is None:
                    return
                data = json.dumps(payload, indent=2, default=str)
                ui.run_javascript(
                    f"navigator.clipboard.writeText({json.dumps(data)})"
                )
                ui.notify("Copied to clipboard!", type="positive")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Download Version (JSON)",
                    icon="download",
                    on_click=_export_version_json,
                ).props("unelevated no-caps color=primary").classes(
                    "rounded-lg shadow-sm"
                )
                ui.button(
                    "Copy to Clipboard",
                    icon="content_copy",
                    on_click=_copy_version_json,
                ).props("no-caps outline color=secondary").classes("rounded-lg")

        # ---- Export World State ----
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("data_object", size="md", color="primary")
                ui.label("Export World State").classes(
                    "text-lg font-semibold text-slate-800"
                )

            ui.label("Download the current world model as JSON.").classes(
                "text-sm text-slate-500"
            )

            def _export_json():
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return
                data = state.world_state.model_dump_json(indent=2)
                ui.download(
                    data.encode("utf-8"),
                    filename=f"{state.project_name.replace(' ', '_')}_world.json",
                )
                ui.notify("World state exported!", type="positive")

            with ui.row().classes("gap-2"):
                ui.button("Download JSON", icon="download", on_click=_export_json).props(
                    "unelevated no-caps color=primary"
                ).classes("rounded-lg shadow-sm")

                # Copy to clipboard
                def _copy_json():
                    if state.world_state is None:
                        ui.notify("No world model", type="warning")
                        return
                    data = state.world_state.model_dump_json(indent=2)
                    ui.run_javascript(
                        f"navigator.clipboard.writeText({json.dumps(data)})"
                    )
                    ui.notify("Copied to clipboard!", type="positive")

                ui.button("Copy to Clipboard", icon="content_copy", on_click=_copy_json).props(
                    "no-caps outline color=secondary"
                ).classes("rounded-lg")

        # ---- Summary Stats ----
        stats_container = ui.column().classes("w-full")

        def _refresh_stats(**kw):
            stats_container.clear()
            ws = state.world_state
            if ws is None:
                return
            with stats_container:
                with ui.card().classes(
                    "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
                ):
                    ui.label("World Model Summary").classes(
                        "text-lg font-semibold text-slate-800"
                    )
                    stats = [
                        ("Entities", len(ws.entities)),
                        ("Locations", len(ws.locations)),
                        ("Events", len(ws.events)),
                        ("Objects", len(ws.objects)),
                        ("World Traits", len(ws.world_traits)),
                        ("Causal Edges", len(ws.causal_topology)),
                        ("Spatial Edges", len(ws.spatial_topology)),
                        ("Social Edges", len(ws.social_topology)),
                        ("Channels", len(ws.channels)),
                        (
                            "Utterances",
                            sum(1 for e in ws.events if e.event_type == "utterance"),
                        ),
                    ]
                    with ui.row().classes("gap-4 flex-wrap mt-2"):
                        for label, count in stats:
                            with ui.column().classes("items-center min-w-20"):
                                ui.label(str(count)).classes(
                                    "text-3xl font-bold text-slate-800"
                                )
                                ui.label(label).classes(
                                    "text-xs text-slate-500"
                                )

        _refresh_stats()
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_stats)

        # ---- Share / Collaborate ----
        if state.user_id and state.project_id:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("share", size="md", color="primary")
                    ui.label("Share & Collaborate").classes(
                        "text-lg font-semibold text-slate-800"
                    )

                project = db.get_project(state.project_id)
                is_owner = project and project.owner_id == state.user_id

                if is_owner:
                    # Visibility toggle
                    is_public = project.is_public if project else False

                    vis_switch = ui.switch(
                        "Public",
                        value=is_public,
                    )

                    def _toggle_visibility(e):
                        new_val = vis_switch.value
                        db.update_project(state.project_id, is_public=new_val)
                        ui.notify(
                            "Project is now public" if new_val else "Project is now private"
                        )

                    vis_switch.on("update:model-value", _toggle_visibility)

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Visibility:").classes("text-sm text-slate-600")

                    # Invite collaborators
                    ui.separator().classes("q-my-sm")
                    ui.label("Invite Collaborators").classes(
                        "text-sm font-semibold text-slate-700"
                    )

                    with ui.row().classes("items-center gap-2"):
                        invite_input = ui.input("Username or email").classes("w-64")
                        role_select = ui.select(
                            ["viewer", "editor", "admin"],
                            value="viewer",
                            label="Role",
                        ).classes("w-32")

                        def _invite():
                            username = invite_input.value.strip()
                            if not username:
                                return
                            users = db.search_users(username)
                            if not users:
                                ui.notify("User not found", type="warning")
                                return
                            target_user = users[0]
                            db.add_project_member(
                                state.project_id, target_user["id"], role_select.value
                            )
                            ui.notify(f"Invited {target_user['username']} as {role_select.value}")
                            invite_input.value = ""
                            _refresh_members()

                        ui.button("Invite", icon="person_add", on_click=_invite).props(
                            "unelevated no-caps color=primary"
                        ).classes("rounded-lg shadow-sm")

                    # Current members
                    members_container = ui.column().classes("w-full q-mt-sm")

                    def _refresh_members():
                        members_container.clear()
                        members = db.list_project_members(state.project_id)
                        if not members:
                            with members_container:
                                ui.label("No collaborators yet.").classes(
                                    "text-sm text-slate-400 italic"
                                )
                            return
                        with members_container:
                            for m in members:
                                with ui.row().classes("items-center gap-2"):
                                    ui.label(m["username"]).classes(
                                        "text-sm text-slate-700"
                                    )
                                    ui.badge(m["role"], color="secondary").props("dense")

                                    def _remove(uid=m["user_id"]):
                                        db.remove_project_member(state.project_id, uid)
                                        _refresh_members()

                                    ui.button(
                                        icon="close", on_click=_remove
                                    ).props("flat dense round size=xs")

                    _refresh_members()

                # Fork project
                ui.separator().classes("q-my-sm")

                def _fork():
                    if state.project_id is None:
                        return
                    new_proj = db.fork_project(
                        state.project_id,
                        state.user_id,
                        f"{state.project_name} (fork)",
                    )
                    if new_proj is None:
                        ui.notify("Fork failed — source not found", type="negative")
                        return
                    ui.notify(f"Forked as '{new_proj.name}'!", type="positive")
                    ui.navigate.to(f"/project/{new_proj.id}")

                ui.button("Fork Project", icon="call_split", on_click=_fork).props(
                    "no-caps outline color=secondary"
                ).classes("rounded-lg")
