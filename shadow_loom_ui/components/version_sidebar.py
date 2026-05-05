# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Left sidebar — version-history tree.

Always-visible left rail showing the project's version DAG. Clicking a
node loads that version into the rest of the UI (Story / World /
Causality / Audit / Export tabs all subscribe to
:data:`StateEvent.VERSION_CHANGED`). Owners and editors can also delete
the currently-loaded version (children are re-parented onto the deleted
version's ancestor) or graft it under a different ancestor.

Selecting a version also writes the active-version pointer in the DB,
which the MCP server consults so subsequent tool calls default to the
same version the user is looking at in the UI.
"""

from __future__ import annotations

import logging

from nicegui import ui

from shadow_loom_ui import db
from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import render_version_tree

logger = logging.getLogger(__name__)


def _resolve_clicked_version(
    args: dict, tree_data: list[dict],
) -> dict | None:
    """Map an ECharts node-click payload to its DB version row.

    ECharts click events deliver the whole event payload in
    ``e.args``; the per-node fields baked in by
    ``version_tree_to_echart_data`` (``_vid``, ``_version``, decorated
    ``name``) live under ``e.args["data"]``, NOT at the top level. An
    earlier version of this handler read ``args.get("_vid")`` directly,
    so ``_vid`` was always ``None`` and the handler fell through to the
    label-matching fallback — which itself failed on shadow rows
    because ``version_tree_to_echart_data`` appends an em-dash + branch
    label suffix (``"v3 — What if Duncan lived"``), so clicking a
    shadow branch did nothing at all and Story / Audit / Source all
    kept rendering the previous branch's contents.

    Returns the matching dict from ``tree_data`` or ``None`` (e.g. for
    the synthetic multi-root wrapper, which has no ``_vid``).
    """
    node = args.get("data") if isinstance(args.get("data"), dict) else {}
    # Prefer the per-node ``_vid`` payload — robust against label
    # decoration. Fall back to the top-level name in case a synthetic
    # root or future ECharts version nests differently.
    vid = node.get("_vid")
    if vid is None and isinstance(args.get("_vid"), int):
        vid = args["_vid"]
    if vid is not None:
        for v in tree_data:
            if v["id"] == vid:
                return v
    # Fallback: legacy label match against the *undecorated* name
    # (``"v{version}"``). Only matches factual rows by design; shadow
    # rows always carry ``_vid`` so they reach the handler via the
    # path above.
    name = node.get("name") or args.get("name", "")
    for v in tree_data:
        if f"v{v['version']}" == name:
            return v
    return None


def build_version_sidebar(state: AppState) -> None:
    """Build the left version-tree sidebar."""

    with ui.column().classes(
        "w-full h-full bg-slate-50 border-r border-slate-200 gap-0"
    ):
        with ui.row().classes(
            "w-full items-center px-3 py-2 border-b border-slate-200 gap-2"
        ):
            ui.icon("account_tree", color="primary")
            ui.label("Versions").classes(
                "text-sm font-semibold text-slate-700"
            )
            ui.space()
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Versions — the project's branching DAG",
                body_md=(
                    "Every generative query (Continue, Intervene,"
                    " What-If, Direct, Write prose) saves a **new"
                    " version** branching from the version that was"
                    " active when it ran. Read-only modes (Ask,"
                    " Interrogation, Evaluate) never create versions.\n\n"
                    "### Reading the tree\n"
                    "- **Highlighted node** — the version currently"
                    " loaded in every panel.\n"
                    "- **Edges** — ancestor relationships. A child"
                    " version's world state is the merge of its"
                    " ancestor plus the changeset produced by the query"
                    " that created it.\n"
                    "- **Node colour / decoration** — distinguishes"
                    " factual mainline from shadow branches.\n\n"
                    "### Branches\n"
                    "- **Factual** — your canonical story. Default for"
                    " Continue / Intervene / Direct / Write prose.\n"
                    "- **Shadow** — a What-If experiment, kept separate"
                    " so it doesn't pollute canon. Promote it to make"
                    " it factual; diff it against the factual head to"
                    " see what would change.\n\n"
                    "### Interactions\n"
                    "- **Click** any node to load that version. Story"
                    " prose, World state, Causality graph, Audit log,"
                    " Reasoning trace, and the MCP active-version"
                    " pointer all switch in lockstep.\n"
                    "- **Active-version pointer** — your selection is"
                    " remembered per project, so MCP tool calls and"
                    " your next session default to the same view.\n\n"
                    "### Owner / editor controls (top-row icons)\n"
                    "- 🗑 **Delete** — remove the current version."
                    " Children rejoin its parent (or cascade if you"
                    " tick the box). The root v0 cannot be deleted.\n"
                    "- 🔀 **Reparent** — graft the current version"
                    " under a different ancestor (useful when a branch"
                    " was started from the wrong point).\n"
                    "- ⏫ **Promote** — copy this shadow version onto"
                    " the factual mainline as a new canonical version.\n"
                    "- ⇄ **Diff** — side-by-side prose comparison"
                    " between this shadow version and the factual head."
                ),
                tooltip="What is the version tree?",
            )

        with ui.scroll_area().classes("w-full flex-grow"):
            container = ui.column().classes("w-full p-1")
            _render_versions(state, container)

    def _on_version_change(**_kw):
        _render_versions(state, container)

    def _on_project_loaded(**_kw):
        _render_versions(state, container)

    state.on(StateEvent.VERSION_CHANGED, _on_version_change)
    state.on(StateEvent.PROJECT_LOADED, _on_project_loaded)


# =====================================================================
# Version tree rendering + mutations
# =====================================================================


def _render_versions(state: AppState, container) -> None:
    container.clear()
    if state.project_id is None:
        with container:
            ui.label("No project loaded.").classes(
                "text-xs text-slate-400 italic p-3"
            )
        return

    tree_data = db.get_version_tree(state.project_id)
    if not tree_data:
        with container:
            ui.label("No versions yet.").classes(
                "text-xs text-slate-400 italic p-3"
            )
        return

    def _on_version_click(e):
        args = e.args if isinstance(e.args, dict) else {}
        v = _resolve_clicked_version(args, tree_data)
        if v is not None:
            _load_version(state, v)

    with container:
        if _can_mutate_versions(state):
            with ui.row().classes("w-full items-center gap-1 px-1 pt-1"):
                ui.button(
                    icon="delete",
                    on_click=lambda: _open_delete_version_dialog(
                        state, tree_data, container,
                    ),
                ).props("flat dense color=negative").tooltip(
                    "Delete current version (rejoins children to its parent)"
                )
                ui.button(
                    icon="alt_route",
                    on_click=lambda: _open_reparent_dialog(
                        state, tree_data, container,
                    ),
                ).props("flat dense color=primary").tooltip(
                    "Move current version under a different ancestor"
                )
                # Branch-aware actions (only meaningful on a shadow
                # version; cheap to render the buttons unconditionally
                # and disable when not applicable).
                current = next(
                    (v for v in tree_data
                     if v["id"] == state.current_version_row_id),
                    None,
                )
                is_shadow = bool(
                    current and (current.get("world_id") == "shadow")
                )
                ui.button(
                    icon="publish",
                    on_click=lambda: _open_promote_dialog(
                        state, current, container,
                    ),
                ).props(
                    f"flat dense color=secondary {'' if is_shadow else 'disable'}"
                ).tooltip(
                    "Promote this shadow branch onto the factual mainline"
                )
                ui.button(
                    icon="compare",
                    on_click=lambda: _open_diff_dialog(
                        state, current, tree_data,
                    ),
                ).props(
                    f"flat dense {'' if is_shadow else 'disable'}"
                ).tooltip("Diff this shadow version against factual head")

        tree_holder = ui.column().classes("w-full")

        def _draw_tree():
            tree_holder.clear()
            with tree_holder:
                render_version_tree(
                    tree_data,
                    current_version_id=state.current_version_row_id,
                    on_click=_on_version_click,
                    height="calc(100vh - 280px)",
                    orient="vertical",
                )

        _draw_tree()


def _can_mutate_versions(state: AppState) -> bool:
    if state.project_id is None or state.user_id is None:
        return False
    proj = db.get_project(state.project_id)
    if proj is None:
        return False
    if proj.owner_id == state.user_id:
        return True
    role = db.get_user_project_role(state.project_id, state.user_id)
    return role in ("editor", "admin")


def _load_version(state: AppState, v: dict) -> None:
    from shadow_loom.models import WorldStateV1

    if v["id"] == state.current_version_row_id:
        return
    ver = db.get_version_by_id(v["id"])
    if ver is None:
        ui.notify("Version not found", type="warning")
        return
    try:
        ws = WorldStateV1.model_validate_json(ver.world_state_json)
        # Atomic swap: resets cursors, emits WORLD_STATE_CHANGED +
        # VERSION_CHANGED + cursor-reset events in lockstep.
        state.load_db_version(ws, v["id"], version_number=v["version"])

        # Mirror the selection into the MCP active-version pointer so
        # subsequent agent tool calls default to the same version.
        if state.user_id is not None and state.project_id is not None:
            try:
                db.set_active_version(
                    state.project_id, state.user_id, v["id"],
                )
            except Exception:
                logger.exception("Failed to update active-version pointer")

        ui.notify(f"Loaded v{v['version']}")
    except Exception as e:
        ui.notify(f"Load failed: {e}", type="negative")


def _open_delete_version_dialog(
    state: AppState, tree_data: list[dict], container,
) -> None:
    if state.current_version_row_id is None:
        ui.notify("No version selected", type="warning")
        return
    current = next(
        (v for v in tree_data if v["id"] == state.current_version_row_id),
        None,
    )
    if current is None:
        ui.notify("Current version not found in tree", type="warning")
        return
    if current.get("ancestor_id") is None:
        ui.notify("The root version (v0) cannot be deleted", type="warning")
        return

    cascade_holder = {"value": False}

    with ui.dialog() as dialog, ui.card():
        ui.label(f"Delete version v{current['version']}?").classes(
            "text-base font-semibold"
        )
        ui.label(
            "Children will be re-parented onto this version's ancestor "
            "(rejoin), so the tree stays connected."
        ).classes("text-xs text-slate-500")
        ui.checkbox(
            "Cascade — also delete all descendants",
            on_change=lambda e: cascade_holder.update(value=bool(e.value)),
        ).props("dense")
        with ui.row().classes("justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Delete",
                on_click=lambda: _do_delete_version(
                    state, current, cascade_holder["value"],
                    container, dialog,
                ),
            ).props("color=negative unelevated")
    dialog.open()


def _do_delete_version(
    state: AppState,
    current: dict,
    cascade: bool,
    container,
    dialog,
) -> None:
    try:
        result = db.delete_version(
            current["id"], state.user_id, cascade=cascade,
        )
    except db.VersionMutationError as exc:
        ui.notify(f"Delete failed: {exc}", type="negative")
        return
    except PermissionError as exc:
        ui.notify(str(exc), type="negative")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Version delete failed")
        ui.notify(f"Delete failed: {exc}", type="negative")
        return

    dialog.close()
    deleted_count = len(result.get("deleted", []))
    reparented = result.get("reparented", {})
    msg = f"Deleted {deleted_count} version(s)"
    if reparented:
        msg += f"; rejoined {len(reparented)} child branch(es)"
    ui.notify(msg, type="positive")

    if state.current_version_row_id in result.get("deleted", []):
        # Pick a fallback version so the rest of the UI doesn't keep
        # rendering against deleted data: prefer the deleted row's
        # ancestor (the rejoin target), else fall back to the project's
        # current latest.
        from shadow_loom.models import WorldStateV1

        fallback_id = current.get("ancestor_id")
        fallback_ver = None
        if fallback_id is not None:
            fallback_ver = db.get_version_by_id(fallback_id)
        if fallback_ver is None and state.project_id is not None:
            fallback_ver = db.get_latest_version(state.project_id)

        if fallback_ver is not None:
            try:
                ws = WorldStateV1.model_validate_json(
                    fallback_ver.world_state_json
                )
                state.load_db_version(
                    ws, fallback_ver.id,
                    version_number=fallback_ver.version,
                )
                if state.user_id is not None and state.project_id is not None:
                    try:
                        db.set_active_version(
                            state.project_id, state.user_id, fallback_ver.id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to update active-version pointer after delete"
                        )
            except Exception:
                logger.exception(
                    "Failed to load fallback world state after delete"
                )
                # Fallback row exists in the DB but we couldn't
                # deserialise it. Drop the world payload entirely so
                # downstream panels don't keep rendering against the
                # just-deleted version's content.
                state.world_state = None
                state.fabula_cursor = None
                state.syuzhet_cursor = None
                state.query_history.clear()
                state.last_result = None
                state.last_parse = None
                state.selected_node_id = None
                state.selected_node_type = None
                state.current_version_row_id = None
                if state.user_id is not None and state.project_id is not None:
                    try:
                        db.clear_active_version(state.project_id, state.user_id)
                    except Exception:
                        logger.exception(
                            "Failed to clear active-version pointer"
                        )
                state.emit(StateEvent.WORLD_STATE_CHANGED)
                state.emit(StateEvent.VERSION_CHANGED, version=None)
        else:
            # No fallback ancestor or latest available (root would have
            # been the only option and it's protected from deletion, so
            # this branch is essentially defensive). Same teardown as
            # the load-failure path above — never leave panels reading
            # the deleted version's world state.
            state.world_state = None
            state.fabula_cursor = None
            state.syuzhet_cursor = None
            state.query_history.clear()
            state.last_result = None
            state.last_parse = None
            state.selected_node_id = None
            state.selected_node_type = None
            state.current_version_row_id = None
            if state.user_id is not None and state.project_id is not None:
                try:
                    db.clear_active_version(state.project_id, state.user_id)
                except Exception:
                    logger.exception(
                        "Failed to clear active-version pointer"
                    )
            state.emit(StateEvent.WORLD_STATE_CHANGED)
            state.emit(StateEvent.VERSION_CHANGED, version=None)
    else:
        state.emit(StateEvent.VERSION_CHANGED, version=None)
    _render_versions(state, container)


def _open_reparent_dialog(
    state: AppState, tree_data: list[dict], container,
) -> None:
    if state.current_version_row_id is None:
        ui.notify("No version selected", type="warning")
        return
    current = next(
        (v for v in tree_data if v["id"] == state.current_version_row_id),
        None,
    )
    if current is None:
        ui.notify("Current version not found in tree", type="warning")
        return
    if current.get("ancestor_id") is None:
        ui.notify("The root version cannot be reparented", type="warning")
        return

    descendants = _collect_descendants_in_tree(tree_data, current["id"])
    options: dict[int, str] = {}
    for v in tree_data:
        if v["id"] == current["id"] or v["id"] in descendants:
            continue
        label = f"v{v['version']}"
        if v.get("description"):
            label += f" — {v['description'][:40]}"
        options[v["id"]] = label

    if not options:
        ui.notify("No valid ancestor candidates", type="warning")
        return

    selected = {"value": current.get("ancestor_id")}

    with ui.dialog() as dialog, ui.card():
        ui.label(f"Reparent v{current['version']}").classes(
            "text-base font-semibold"
        )
        ui.label("Select the new ancestor:").classes(
            "text-xs text-slate-500"
        )
        ui.select(
            options=options,
            value=selected["value"],
            on_change=lambda e: selected.update(value=e.value),
        ).classes("w-full").props("dense outlined")
        with ui.row().classes("justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Move",
                on_click=lambda: _do_reparent_version(
                    state, current, selected["value"], container, dialog,
                ),
            ).props("color=primary unelevated")
    dialog.open()


def _collect_descendants_in_tree(
    tree_data: list[dict], root_id: int,
) -> set[int]:
    seen: set[int] = {root_id}
    frontier = [root_id]
    while frontier:
        next_frontier: list[int] = []
        for v in tree_data:
            if v.get("ancestor_id") in frontier and v["id"] not in seen:
                seen.add(v["id"])
                next_frontier.append(v["id"])
        frontier = next_frontier
    return seen


def _do_reparent_version(
    state: AppState,
    current: dict,
    new_ancestor_id: int | None,
    container,
    dialog,
) -> None:
    try:
        db.reparent_version(current["id"], new_ancestor_id, state.user_id)
    except db.VersionMutationError as exc:
        ui.notify(f"Reparent failed: {exc}", type="negative")
        return
    except PermissionError as exc:
        ui.notify(str(exc), type="negative")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Version reparent failed")
        ui.notify(f"Reparent failed: {exc}", type="negative")
        return

    dialog.close()
    ui.notify(
        f"v{current['version']} moved under "
        f"{'(detached)' if new_ancestor_id is None else f'version row {new_ancestor_id}'}",
        type="positive",
    )
    _render_versions(state, container)


# =====================================================================
# Branch promotion + diff (Story-integration plan, Step 3)
# =====================================================================


def _open_promote_dialog(
    state: AppState, current: dict | None, container,
) -> None:
    """Confirm promoting a shadow branch onto the factual mainline."""
    if current is None or current.get("world_id") != "shadow":
        ui.notify("Only shadow versions can be promoted", type="warning")
        return

    label = current.get("branch_label") or "(unlabelled)"
    desc_holder = {"value": ""}
    with ui.dialog() as dialog, ui.card():
        ui.label(f"Promote shadow v{current['version']} to canon?").classes(
            "text-base font-semibold"
        )
        ui.label(
            f"Branch: {label}. A new factual version will be appended "
            "to the mainline, copying this version's world state and "
            "prose. The original shadow version stays browsable."
        ).classes("text-xs text-slate-500")
        ui.input(
            label="Promotion note (optional)",
            on_change=lambda e: desc_holder.update(value=str(e.value or "")),
        ).classes("w-full").props("dense")
        with ui.row().classes("justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Promote",
                on_click=lambda: _do_promote(
                    state, current, desc_holder["value"], container, dialog,
                ),
            ).props("color=secondary unelevated")
    dialog.open()


def _do_promote(
    state: AppState, current: dict, description: str, container, dialog,
) -> None:
    try:
        promoted = db.promote_branch(
            current["id"],
            user_id=state.user_id,
            description=description or None,
        )
    except db.VersionMutationError as exc:
        ui.notify(f"Promote failed: {exc}", type="negative")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Branch promotion failed")
        ui.notify(f"Promote failed: {exc}", type="negative")
        return

    dialog.close()
    ui.notify(
        f"Promoted shadow v{current['version']} \u2192 factual v{promoted.version}",
        type="positive",
    )

    # Hop the UI onto the new factual head so the user sees the result.
    from shadow_loom.models import WorldStateV1
    try:
        ws = WorldStateV1.model_validate_json(promoted.world_state_json)
        state.load_db_version(
            ws, promoted.id, version_number=promoted.version,
        )
        if state.user_id is not None and state.project_id is not None:
            try:
                db.set_active_version(
                    state.project_id, state.user_id, promoted.id,
                )
            except Exception:
                logger.exception(
                    "Failed to update active-version pointer after promote"
                )
    except Exception:
        logger.exception("Failed to load promoted version")

    _render_versions(state, container)


def _open_diff_dialog(
    state: AppState, current: dict | None, tree_data: list[dict],
) -> None:
    """Side-by-side comparison of a shadow version against factual head."""
    if current is None or current.get("world_id") != "shadow":
        ui.notify("Diff is only available on shadow versions", type="warning")
        return
    if state.project_id is None:
        return

    factual_head = None
    for v in sorted(
        tree_data, key=lambda r: r.get("version", 0), reverse=True,
    ):
        if (v.get("world_id") or "factual") == "factual":
            factual_head = v
            break
    if factual_head is None:
        ui.notify("No factual version to compare against", type="warning")
        return

    shadow_row = db.get_version_by_id(current["id"])
    factual_row = db.get_version_by_id(factual_head["id"])
    if shadow_row is None or factual_row is None:
        ui.notify("Version row missing", type="warning")
        return

    label = current.get("branch_label") or "(unlabelled)"
    with ui.dialog() as dialog, ui.card().classes("min-w-[80vw]"):
        with ui.row().classes("w-full items-baseline gap-3"):
            ui.label("Branch diff").classes("text-base font-semibold")
            ui.label(
                f"shadow v{current['version']} ({label}) "
                f"vs factual v{factual_head['version']}"
            ).classes("text-xs text-slate-500")
        with ui.row().classes("w-full gap-3 mt-2"):
            with ui.column().classes("flex-1 gap-1"):
                ui.label("Factual head").classes(
                    "text-xs font-semibold text-emerald-700"
                )
                ui.markdown(
                    factual_row.prose or "_(no prose)_",
                ).classes(
                    "text-sm bg-emerald-50 p-2 rounded border "
                    "border-emerald-200 max-h-[60vh] overflow-auto"
                )
            with ui.column().classes("flex-1 gap-1"):
                ui.label("Shadow branch").classes(
                    "text-xs font-semibold text-violet-700"
                )
                ui.markdown(
                    shadow_row.prose or "_(no prose)_",
                ).classes(
                    "text-sm bg-violet-50 p-2 rounded border "
                    "border-violet-200 max-h-[60vh] overflow-auto"
                )
        with ui.row().classes("justify-end mt-2"):
            ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()
