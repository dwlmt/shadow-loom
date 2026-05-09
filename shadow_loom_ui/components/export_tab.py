# SPDX-FileCopyrightText: 2026 David Rae Wilmot
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
        # ── Header with help popover ────────────────────────────────────
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("ios_share", color="primary")
            ui.label("Export").classes(
                "text-sm font-semibold text-slate-700"
            )
            ui.space()
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Export — download world state, prose, & bundles",
                body_md=(
                    "Get your data out of Shadow Loom in standard"
                    " formats for use in other tools or for archival.\n\n"
                    "### Three export cards\n"
                    "- **Prose (Markdown)** — pick a scope: just the"
                    " currently loaded version, the full root\u2192current"
                    " branch lineage (default), or every version in"
                    " the project. Branch-aware so a shadow fork's"
                    " export doesn't pull in sibling branches.\n"
                    "- **World model (JSON)** — the full"
                    " `WorldStateV1` payload of the currently loaded"
                    " version: entities, events, all topology layers,"
                    " channels, beliefs, world traits. Round-trips"
                    " cleanly back into Shadow Loom and is the"
                    " canonical interchange format.\n"
                    "- **Version bundle (JSON)** — one file with the"
                    " version row metadata (id, ancestor, branch"
                    " label, raw query, timestamp), the world model,"
                    " and the prose for the loaded version. Useful"
                    " for archival or moving a single version between"
                    " projects.\n\n"
                    "### Notes\n"
                    "- Exports always reflect the **currently loaded"
                    " version**. Switch versions in the left tree"
                    " before exporting to capture a different branch"
                    " or point in history.\n"
                    "- Research facts are **not** included in any"
                    " export \u2014 they live in a separate store. Pull"
                    " them from the Research tab if needed.\n"
                    "- Exports are read-only and never alter the"
                    " project state."
                ),
                tooltip="What is this tab?",
            )

        # ---- Export Prose ----
        # Single prose-export card with a scope selector. Replaces the
        # old "Export Prose" (project-wide, branch-blind) and the
        # per-version "Download Prose" button which together produced
        # three near-identical Markdown actions on this tab.
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("article", size="md", color="primary")
                ui.label("Export prose").classes(
                    "text-lg font-semibold text-slate-800"
                )
            ui.label(
                "Download generated prose as Markdown. Pick the scope: "
                "just this version, the full root\u2192current branch "
                "lineage, or every version in the project."
            ).classes("text-sm text-slate-500")

            prose_scope = ui.toggle(
                {
                    "current": "Current version",
                    "lineage": "This branch (root \u2192 current)",
                    "project": "All versions in project",
                },
                value="lineage",
            ).props("dense unelevated")

            async def _export_prose():
                if state.project_id is None:
                    ui.notify("No project loaded", type="warning")
                    return
                scope = prose_scope.value or "lineage"
                stem = (
                    state.project_name.replace(" ", "_")
                    if state.project_name else "project"
                )
                if scope == "current":
                    row = None
                    if state.current_version_row_id is not None:
                        row = db.get_version_by_id(
                            state.current_version_row_id,
                        )
                    if row is None or not row.prose:
                        ui.notify(
                            "This version has no prose", type="info",
                        )
                        return
                    md = (
                        f"# {state.project_name} \u2014 v{row.version}\n\n"
                        f"{row.prose}\n"
                    )
                    filename = f"{stem}_v{row.version}_prose.md"
                else:
                    branch_path: list[int] | None = None
                    if scope == "lineage" and state.current_version_row_id:
                        cur = db.get_version_by_id(
                            state.current_version_row_id,
                        )
                        if cur is not None:
                            from shadow_loom_ui.db import (
                                get_version_lineage,
                            )
                            lineage = get_version_lineage(
                                state.project_id, cur.version,
                            )
                            branch_path = [e["id"] for e in lineage]
                    prose_list = db.get_all_prose(
                        state.project_id, branch_path=branch_path,
                    )
                    if not prose_list:
                        ui.notify("No prose found", type="info")
                        return
                    md = f"# {state.project_name}\n\n"
                    for entry in prose_list:
                        md += (
                            f"## v{entry['version']} ({entry['source']})\n\n"
                            f"{entry['prose']}\n\n---\n\n"
                        )
                    suffix = "branch" if scope == "lineage" else "all"
                    filename = f"{stem}_prose_{suffix}.md"

                ui.download(md.encode("utf-8"), filename=filename)
                ui.notify("Prose exported", type="positive")

            ui.button(
                "Download Markdown",
                icon="download",
                on_click=_export_prose,
            ).props("unelevated no-caps color=primary").classes(
                "rounded-lg shadow-sm"
            )

        # ---- Export World Model ----
        # The world model is always the *currently loaded* snapshot
        # \u2014 there is no meaningful "all versions" or "lineage" scope
        # for a single JSON object, so this card stays single-purpose
        # (download + copy) and is no longer duplicated by the old
        # combined-bundle card below.
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("data_object", size="md", color="primary")
                ui.label("Export world model").classes(
                    "text-lg font-semibold text-slate-800"
                )
            ui.label(
                "The full WorldStateV1 JSON for the currently loaded "
                "version. Round-trips cleanly back into Shadow Loom."
            ).classes("text-sm text-slate-500")

            def _world_filename_stem() -> str:
                stem = (
                    state.project_name.replace(" ", "_")
                    if state.project_name else "project"
                )
                row = None
                if state.current_version_row_id is not None:
                    row = db.get_version_by_id(state.current_version_row_id)
                elif state.project_id is not None:
                    row = db.get_latest_version(state.project_id)
                if row is not None:
                    stem += f"_v{row.version}"
                return stem

            def _export_world():
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return
                data = state.world_state.model_dump_json(indent=2)
                ui.download(
                    data.encode("utf-8"),
                    filename=f"{_world_filename_stem()}_world.json",
                )
                ui.notify("World model exported", type="positive")

            def _copy_world():
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return
                data = state.world_state.model_dump_json(indent=2)
                ui.run_javascript(
                    f"navigator.clipboard.writeText({json.dumps(data)})"
                )
                ui.notify("Copied to clipboard", type="positive")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Download JSON",
                    icon="download",
                    on_click=_export_world,
                ).props("unelevated no-caps color=primary").classes(
                    "rounded-lg shadow-sm"
                )
                ui.button(
                    "Copy to clipboard",
                    icon="content_copy",
                    on_click=_copy_world,
                ).props("no-caps outline color=secondary").classes(
                    "rounded-lg"
                )

        # ---- Export Version Bundle (metadata + world + prose) ----
        # The unique value-add over the two cards above: a single
        # JSON containing the version row metadata, the world model,
        # and the prose for the currently loaded version. Useful for
        # archival or for moving a single version between projects.
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("inventory_2", size="md", color="primary")
                ui.label("Export version bundle").classes(
                    "text-lg font-semibold text-slate-800"
                )
            ui.label(
                "One JSON file with the version metadata (row id, "
                "ancestor, branch label, raw query, timestamp), the "
                "world model, and the prose. Currently loaded version "
                "only."
            ).classes("text-sm text-slate-500")

            def _build_bundle() -> dict | None:
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

            def _export_bundle():
                payload = _build_bundle()
                if payload is None:
                    return
                data = json.dumps(payload, indent=2, default=str)
                ui.download(
                    data.encode("utf-8"),
                    filename=f"{_world_filename_stem()}_bundle.json",
                )
                ui.notify("Bundle exported", type="positive")

            def _copy_bundle():
                payload = _build_bundle()
                if payload is None:
                    return
                data = json.dumps(payload, indent=2, default=str)
                ui.run_javascript(
                    f"navigator.clipboard.writeText({json.dumps(data)})"
                )
                ui.notify("Copied to clipboard", type="positive")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Download JSON",
                    icon="download",
                    on_click=_export_bundle,
                ).props("unelevated no-caps color=primary").classes(
                    "rounded-lg shadow-sm"
                )
                ui.button(
                    "Copy to clipboard",
                    icon="content_copy",
                    on_click=_copy_bundle,
                ).props("no-caps outline color=secondary").classes(
                    "rounded-lg"
                )

        # ---- Research Dataset (JSONL preset) ----
        # One-click bundling of every persisted version into a flat
        # JSONL file: one row per generation event with the raw query
        # ("prompt"), the parsed brief, the generated prose, the
        # changeset summary, and any audit / source provenance the
        # database has on the row. Targeted at researchers who want
        # to slice the project's lineage into a fine-tunable / scored
        # dataset without writing a custom dump script.
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("dataset", size="md", color="primary")
                ui.label("Research dataset (JSONL)").classes(
                    "text-lg font-semibold text-slate-800"
                )
            ui.label(
                "One JSON object per persisted version: prompt + parsed "
                "brief + generated prose + changeset summary + audit / "
                "source provenance. Suitable for fine-tuning datasets "
                "or for offline evaluation pipelines."
            ).classes("text-sm text-slate-500")

            def _build_dataset_rows() -> list[dict] | None:
                if state.project_id is None:
                    ui.notify("No project loaded", type="warning")
                    return None
                summaries = db.list_versions(state.project_id)
                rows: list[dict] = []
                for s in summaries:
                    row = db.get_version_by_id(s["id"])
                    if row is None:
                        continue
                    parsed_brief = None
                    if row.parsed_query_json:
                        try:
                            parsed_brief = json.loads(row.parsed_query_json)
                        except (TypeError, ValueError):
                            parsed_brief = row.parsed_query_json
                    changeset = None
                    if row.changeset_json:
                        try:
                            changeset = json.loads(row.changeset_json)
                        except (TypeError, ValueError):
                            changeset = None
                    rows.append({
                        "version_row_id": row.id,
                        "version": row.version,
                        "ancestor_id": row.ancestor_id,
                        "world_id": row.world_id,
                        "branch_label": row.branch_label,
                        "source": row.source,
                        "description": row.description,
                        "prompt": row.raw_query,
                        "brief": parsed_brief,
                        "prose": row.prose,
                        "changeset": changeset,
                        "created_at": str(row.created_at),
                    })
                return rows

            def _export_dataset():
                rows = _build_dataset_rows()
                if rows is None:
                    return
                if not rows:
                    ui.notify(
                        "No versions to export", type="warning",
                    )
                    return
                lines = "\n".join(
                    json.dumps(r, default=str, ensure_ascii=False)
                    for r in rows
                )
                ui.download(
                    lines.encode("utf-8"),
                    filename=(
                        f"{_world_filename_stem()}_research.jsonl"
                    ),
                )
                ui.notify(
                    f"Exported {len(rows)} rows", type="positive",
                )

            def _copy_dataset():
                rows = _build_dataset_rows()
                if rows is None:
                    return
                if not rows:
                    ui.notify(
                        "No versions to export", type="warning",
                    )
                    return
                lines = "\n".join(
                    json.dumps(r, default=str, ensure_ascii=False)
                    for r in rows
                )
                ui.run_javascript(
                    f"navigator.clipboard.writeText({json.dumps(lines)})"
                )
                ui.notify(
                    f"Copied {len(rows)} rows to clipboard",
                    type="positive",
                )

            with ui.row().classes("gap-2"):
                ui.button(
                    "Download JSONL",
                    icon="download",
                    on_click=_export_dataset,
                ).props("unelevated no-caps color=primary").classes(
                    "rounded-lg shadow-sm"
                )
                ui.button(
                    "Copy to clipboard",
                    icon="content_copy",
                    on_click=_copy_dataset,
                ).props("no-caps outline color=secondary").classes(
                    "rounded-lg"
                )

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
                    # Background facts — not canon, shown muted + segregated.
                    fact_count = 0
                    if state.project_id is not None:
                        try:
                            from shadow_loom import db as _db
                            fact_count = len(_db.list_world_facts(state.project_id))
                        except Exception:
                            fact_count = 0
                    ui.separator().classes("my-2")
                    ui.label(
                        f"Background facts (not canon, segregated): {fact_count}"
                    ).classes("text-xs text-slate-400 italic")

        _refresh_stats()
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_stats)
        state.on(StateEvent.VERSION_CHANGED, _refresh_stats)

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
