# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Editor tab — manual editing of the live WorldStateV1.

Provides three integrated editing surfaces backed by a single JSON
textarea (the source of truth):

* **Validate** button — runs the same consistency checks used by the
  ingestion pipeline against the current textarea contents and renders
  errors / warnings in an inline feedback panel (no save).
* **Structural editor** — collapsible per-collection panels that let
  the user add / remove entities, events, locations, objects, world
  traits, and topology edges via small dialogs. Mutations are applied
  to the parsed JSON and synced back into the textarea.
* **JSON textarea** — full-fidelity manual edit of every field, for
  the cases the structural editor does not cover (trait values,
  snapshots, beliefs, descriptions, etc.).

Saving validates schema + cross-references and persists a new version
branching from the current one — no LLM re-ingestion is run.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from nicegui import ui
from pydantic import ValidationError

from shadow_loom.ingestion import _programmatic_validation
from shadow_loom.models import WorldStateV1
from shadow_loom_ui import db
from shadow_loom_ui.state import AppState, StateEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_editor_tab(state: AppState) -> None:
    """Build the Editor tab — view & manually edit the world model JSON."""

    with ui.column().classes("w-full h-full p-6 gap-3 bg-slate-50"):
        # ── Header with help popover ────────────────────────────────────
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("edit_note", color="primary")
            ui.label("World-Model Editor").classes(
                "text-sm font-semibold text-slate-700"
            )
            ui.space()
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Editor — hand-edit the world model JSON",
                body_md=(
                    "Direct read/write access to the underlying"
                    " `WorldStateV1` JSON for surgical fixes the LLM"
                    " pipeline can't make. Use this when you need to"
                    " correct an extraction error or patch in a node"
                    " the ingestion missed.\n\n"
                    "### What you can do\n"
                    "- **Add, remove, or modify** any node or edge:"
                    " entities, locations, objects, events, causal /"
                    " social / spatial / information edges, world"
                    " traits.\n"
                    "- **Validate** — type-checks the JSON against"
                    " the Pydantic schema; errors **block** save,"
                    " warnings can be acknowledged.\n"
                    "- **Save** — creates a new version branching from"
                    " the active one (just like a generative query),"
                    " so the change is reversible by switching"
                    " versions.\n\n"
                    "### Cautions\n"
                    "- Hand-edits **do not** trigger re-extraction or"
                    " auditing. Inconsistent state (e.g. an event"
                    " referencing a deleted entity) will surface as a"
                    " validation error or, worse, propagate into"
                    " downstream queries.\n"
                    "- Prefer the natural-language **Intervene** mode"
                    " for routine state changes — it audits and"
                    " re-extracts, keeping prose and graph in sync.\n"
                    "- Use this editor for graph topology fixes that"
                    " don't have a natural prose representation"
                    " (renaming an id, adding a missing causal edge)."
                ),
                tooltip="What is this tab?",
            )

        container = ui.column().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm "
            "p-4 gap-2"
        )
        _render_editor(state, container)

        # ── Ingestion warnings panel (moved from the removed Causality tab)
        from shadow_loom_ui.components.causality_tab import (
            _build_ingestion_warnings_panel,
        )
        _build_ingestion_warnings_panel(state)

        # Re-render on project / version swaps so the textarea always
        # reflects the live world.
        state.on(
            StateEvent.PROJECT_LOADED,
            lambda **kw: _render_editor(state, container),
        )
        state.on(
            StateEvent.WORLD_STATE_CHANGED,
            lambda **kw: _render_editor(state, container),
        )


def _render_editor(state: AppState, container) -> None:
    container.clear()

    if state.world_state is None:
        with container:
            ui.label(
                "No world model loaded — open or create a project first."
            ).classes("text-sm text-slate-500")
        return

    can_edit = _user_can_edit(state)
    current_json = state.world_state.model_dump_json(indent=2)

    with container:
        # --- Header -----------------------------------------------------
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("edit_note").classes("text-primary text-2xl")
            ui.label("Manual World Model Editor").classes(
                "text-base font-semibold text-slate-800"
            )
            ui.space()
            char_label = ui.label(f"{len(current_json):,} characters").classes(
                "text-xs text-slate-500"
            )

        ui.label(
            "Use the structural editor to add or remove items, or edit "
            "the JSON directly. Validate runs the same consistency checks "
            "as ingestion (ID validity, edge endpoints, snapshot "
            "triggered_by, timeline ordering). Save creates a new version "
            "branched from the current one — no re-ingestion runs."
        ).classes("text-xs text-slate-500")

        if not can_edit:
            ui.label(
                "Read-only — you need editor or owner access to save changes."
            ).classes("text-xs text-warning")

        # --- Textarea (the source of truth) — created up-front so
        #     handlers can refer to it via a holder, then visually
        #     re-attached in the desired position below. ---------------
        ta_holder: dict = {}

        def _refresh_char_count() -> None:
            ta = ta_holder.get("ta")
            if ta is not None:
                char_label.set_text(f"{len(ta.value or ''):,} characters")

        # --- Feedback panel (populated by Validate / add / remove) ----
        feedback_container = ui.column().classes("w-full gap-1")

        # --- Structural editor ----------------------------------------
        if can_edit:
            with ui.expansion(
                "Structural editor — add / remove items",
                icon="construction",
            ).classes("w-full").props("dense"):
                structural_container = ui.column().classes("w-full gap-2")

                def _refresh_structural() -> None:
                    ta = ta_holder.get("ta")
                    if ta is None:
                        return
                    _render_structural(
                        structural_container,
                        ta,
                        feedback_container,
                        _refresh_structural,
                        _refresh_char_count,
                    )

        # --- Textarea (rendered here, in its final visual position) ---
        text_area = ui.textarea(value=current_json).classes(
            "w-full font-mono text-xs"
        ).props(
            "outlined input-style='min-height: 60vh; max-height: 75vh; "
            "overflow:auto; white-space: pre;'"
        )
        if not can_edit:
            text_area.props("readonly")
        text_area.on_value_change(lambda _e: _refresh_char_count())
        ta_holder["ta"] = text_area

        if can_edit:
            _refresh_structural()

        # --- Toolbar --------------------------------------------------
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button(
                "Validate",
                icon="fact_check",
                on_click=lambda: _run_validation(text_area, feedback_container),
            ).props("flat color=secondary no-caps")
            ui.button(
                "Reset",
                icon="undo",
                on_click=lambda: (
                    text_area.set_value(current_json),
                    feedback_container.clear(),
                    _refresh_char_count(),
                ),
            ).props("flat color=secondary no-caps")
            if can_edit:
                ui.button(
                    "Save",
                    icon="save",
                    on_click=lambda: _confirm_save(
                        state, text_area.value, current_json,
                    ),
                ).props("unelevated color=primary no-caps")


# =====================================================================
# Validation feedback
# =====================================================================

def _parse_or_notify(text: str) -> Optional[dict]:
    """Parse JSON; return dict or None (showing a notify on failure)."""
    try:
        return json.loads(text or "")
    except (ValueError, TypeError) as exc:
        ui.notify(f"Invalid JSON: {exc}", type="negative")
        return None


def _validate_or_notify(text: str) -> tuple[Optional[WorldStateV1], list]:
    """Schema-validate JSON. Returns (model_or_none, issues_list).

    On JSON or schema failure, a toast is shown and (None, []) is returned.
    """
    try:
        ws = WorldStateV1.model_validate_json(text or "")
    except ValidationError as exc:
        errs = exc.errors()
        head = errs[:3]
        msg_lines = [
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in head
        ]
        more = f"\n(+{len(errs) - len(head)} more)" if len(errs) > len(head) else ""
        ui.notify(
            "Schema validation failed:\n" + "\n".join(msg_lines) + more,
            type="negative",
            multi_line=True,
            timeout=8000,
        )
        return None, []
    except ValueError as exc:
        ui.notify(f"Invalid JSON: {exc}", type="negative")
        return None, []
    try:
        issues = _programmatic_validation(ws)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Consistency validation crashed")
        ui.notify(f"Consistency check failed: {exc}", type="negative")
        return ws, []
    return ws, issues


def _run_validation(text_area, feedback_container) -> None:
    """Validate the current textarea contents, render results inline."""
    ws, issues = _validate_or_notify(text_area.value)
    if ws is None:
        feedback_container.clear()
        return
    _render_feedback(feedback_container, issues, ws)


def _render_feedback(feedback_container, issues, ws: WorldStateV1) -> None:
    feedback_container.clear()
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    with feedback_container:
        if not errors and not warnings:
            with ui.row().classes(
                "w-full items-center gap-2 p-2 rounded "
                "border border-emerald-200 bg-emerald-50"
            ):
                ui.icon("check_circle").classes("text-positive")
                ui.label(
                    f"Valid — {len(ws.entities)} entities, "
                    f"{len(ws.events)} events, "
                    f"{len(ws.locations)} locations, "
                    f"{len(ws.causal_topology)} causal edges. "
                    "No issues found."
                ).classes("text-xs text-emerald-900")
            return

        if errors:
            with ui.column().classes(
                "w-full gap-1 p-2 rounded "
                "border border-red-200 bg-red-50 max-h-[28vh] overflow-auto"
            ):
                ui.label(f"Errors ({len(errors)}) — these block saving").classes(
                    "text-xs font-semibold text-red-900"
                )
                for issue in errors:
                    ui.label(f"• [{issue.category}] {issue.detail}").classes(
                        "text-xs text-red-900"
                    )

        if warnings:
            with ui.column().classes(
                "w-full gap-1 p-2 rounded "
                "border border-amber-200 bg-amber-50 max-h-[28vh] overflow-auto"
            ):
                ui.label(
                    f"Warnings ({len(warnings)}) — review but do not block saving"
                ).classes("text-xs font-semibold text-amber-900")
                for issue in warnings:
                    ui.label(f"• [{issue.category}] {issue.detail}").classes(
                        "text-xs text-amber-900"
                    )


# =====================================================================
# Structural editor — add / remove items per collection
# =====================================================================

def _render_structural(
    container,
    text_area,
    feedback_container,
    refresh: Callable[[], None],
    refresh_char_count: Callable[[], None],
) -> None:
    """Render add/remove panels for every editable collection."""
    container.clear()
    data = _parse_or_notify(text_area.value)
    if data is None:
        with container:
            with ui.row().classes(
                "w-full items-center gap-2 p-2 rounded "
                "border border-amber-200 bg-amber-50"
            ):
                ui.icon("warning").classes("text-warning")
                ui.label(
                    "JSON parse error — fix the syntax in the textarea, "
                    "then click Refresh."
                ).classes("text-xs text-amber-900")
                ui.button("Refresh", icon="refresh", on_click=refresh).props(
                    "flat color=secondary no-caps dense"
                )
        return

    def commit(new_data: dict) -> None:
        text_area.set_value(json.dumps(new_data, indent=2))
        refresh_char_count()
        # Re-validate inline so the user sees the impact of their edit.
        _run_validation(text_area, feedback_container)
        refresh()

    with container:
        for kind in _STRUCTURAL_KINDS:
            _render_kind_panel(kind, data, commit)


# Each kind: (key in WorldStateV1 dict/list, label, container_type, prefix)
_STRUCTURAL_KINDS = [
    ("locations", "Locations", "dict", "LOC_"),
    ("entities", "Entities", "dict", "ENT_"),
    ("objects", "Objects", "dict", "OBJ_"),
    ("events", "Events", "list", "EVT_"),
    ("world_traits", "World Traits", "dict", "WORLD_"),
    ("causal_topology", "Causal Edges", "list", None),
    ("social_topology", "Relationship Edges", "list", None),
    ("spatial_topology", "Spatial Edges", "list", None),
    ("channels", "Channels", "dict", "CHN_"),
]


def _render_kind_panel(kind: tuple, data: dict, commit: Callable) -> None:
    key, label, ctype, prefix = kind
    items = data.get(key)
    if items is None:
        items = {} if ctype == "dict" else []
        data[key] = items
    count = len(items)

    with ui.expansion(f"{label} ({count})", icon="folder_open").classes(
        "w-full"
    ).props("dense"):
        with ui.column().classes("w-full gap-1"):
            # Item list
            if ctype == "dict":
                for item_id in sorted(items.keys()):
                    _render_item_row(
                        item_id,
                        summary=_dict_item_summary(items[item_id]),
                        on_remove=lambda iid=item_id: _remove_dict_item(
                            data, key, iid, commit,
                        ),
                    )
            else:
                for idx, item in enumerate(items):
                    _render_item_row(
                        _list_item_id(key, item, idx),
                        summary=_list_item_summary(key, item),
                        on_remove=lambda i=idx: _remove_list_item(
                            data, key, i, commit,
                        ),
                    )

            if not items:
                ui.label("(none)").classes("text-xs text-slate-400 italic")

            # Add button
            ui.button(
                f"Add {label[:-1] if label.endswith('s') else label}",
                icon="add",
                on_click=lambda k=key, c=ctype, p=prefix: _open_add_dialog(
                    k, c, p, data, commit,
                ),
            ).props("flat color=primary no-caps dense")


def _render_item_row(
    item_id: str, summary: str, on_remove: Callable,
) -> None:
    with ui.row().classes(
        "w-full items-center gap-2 p-1 border-b border-slate-100"
    ):
        ui.label(item_id).classes("text-xs font-mono text-slate-800 w-64 truncate")
        ui.label(summary).classes("text-xs text-slate-500 flex-1 truncate")
        ui.button(icon="delete", on_click=lambda: _confirm_remove(item_id, on_remove)).props(
            "flat dense color=negative round size=sm"
        )


def _confirm_remove(item_id: str, on_remove: Callable) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label(f"Remove {item_id}?").classes("text-base font-semibold")
        ui.label(
            "References to this ID elsewhere in the world model will become "
            "broken links — Validate after removal to see what needs fixing."
        ).classes("text-xs text-slate-500")
        with ui.row().classes("justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button(
                "Remove",
                on_click=lambda: (dialog.close(), on_remove()),
            ).props("unelevated color=negative no-caps")
    dialog.open()


def _dict_item_summary(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    name = item.get("name") or ""
    desc = item.get("description") or ""
    if name and desc:
        return f"{name} — {desc[:80]}"
    return (name or desc)[:120]


def _list_item_id(key: str, item: Any, idx: int) -> str:
    if not isinstance(item, dict):
        return f"#{idx}"
    if key == "events":
        return item.get("id") or f"#{idx}"
    if key == "causal_topology":
        return f"{item.get('source_id', '?')} → {item.get('target_id', '?')}"
    if key == "social_topology":
        return (
            f"{item.get('source_entity_id', '?')} ↔ "
            f"{item.get('target_entity_id', '?')}"
        )
    if key == "spatial_topology":
        return f"{item.get('source_id', '?')} → {item.get('target_id', '?')}"
    if key == "channels":
        pids = item.get("participant_ids") or []
        return f"{item.get('medium', '?')} — {', '.join(pids) or '?'}"
    return f"#{idx}"


def _list_item_summary(key: str, item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    if key == "events":
        ft = item.get("fabula_time")
        sy = item.get("syuzhet_index")
        et = item.get("event_type", "")
        desc = item.get("description", "")
        return f"[{et}] fabula={ft} syuzhet={sy} — {desc[:80]}"
    if key == "causal_topology":
        return (
            f"[{item.get('causality_type', '?')}] "
            f"mechanism={item.get('mechanism', '?')} "
            f"fabula={item.get('fabula_time', '?')}"
        )
    if key == "social_topology":
        # Per-metric schema: affinity / fear / power_dynamic now live
        # under ``metrics.<axis>.value`` instead of as flat keys. Fall
        # back to the legacy flat keys for backward-compatible payloads.
        # Also surface per-axis evidence_strength + observed so users
        # don't silently lose the new signal when editing.
        metrics = item.get("metrics") or {}
        def _axis(name: str) -> str:
            m = metrics.get(name) or {}
            if not m:
                if name in item:
                    return f"{name[0]}={item.get(name, 0)}"
                return f"{name[0]}=–"
            value = m.get("value", 0)
            es = m.get("evidence_strength", "moderate")
            es_short = {"weak": "w", "moderate": "m", "strong": "s"}.get(es, "m")
            obs = "" if m.get("observed", True) else " unobs"
            return f"{name[0]}={value} [{es_short}{obs}]"
        return f"{_axis('affinity')} {_axis('fear')} {_axis('power_dynamic')}"
    if key == "spatial_topology":
        return (
            f"locked={item.get('is_locked', False)} "
            f"established={item.get('established_at_fabula', 0)}"
        )
    if key == "channels":
        return (
            f"medium={item.get('medium', '?')} "
            f"directionality={item.get('directionality', 'duplex')} "
            f"established={item.get('established_at_fabula', '?')}"
        )
    return ""


def _remove_dict_item(data: dict, key: str, item_id: str, commit: Callable) -> None:
    items = data.get(key) or {}
    if item_id in items:
        items.pop(item_id)
        commit(data)
        ui.notify(f"Removed {item_id}", type="info")


def _remove_list_item(data: dict, key: str, idx: int, commit: Callable) -> None:
    items = data.get(key) or []
    if 0 <= idx < len(items):
        removed = items.pop(idx)
        commit(data)
        ident = _list_item_id(key, removed, idx)
        ui.notify(f"Removed {ident}", type="info")


# ---------- Add dialogs --------------------------------------------------

def _open_add_dialog(
    key: str, ctype: str, prefix: Optional[str], data: dict, commit: Callable,
) -> None:
    """Show a small dialog collecting the minimum fields, insert a skeleton."""
    builder = _ADD_BUILDERS.get(key)
    if builder is None:
        ui.notify(f"Add not implemented for {key}", type="warning")
        return
    builder(data, commit, prefix)


def _next_fabula_time(data: dict) -> int:
    events = data.get("events") or []
    times = [e.get("fabula_time") for e in events if isinstance(e, dict)]
    times = [t for t in times if isinstance(t, int)]
    return (max(times) + 100) if times else 100


def _next_syuzhet_index(data: dict) -> int:
    events = data.get("events") or []
    idxs = [e.get("syuzhet_index") for e in events if isinstance(e, dict)]
    idxs = [i for i in idxs if isinstance(i, int)]
    return (max(idxs) + 1) if idxs else 1


def _normalise_id(raw: str, prefix: Optional[str]) -> str:
    raw = (raw or "").strip().upper().replace(" ", "_")
    if not raw:
        return ""
    if prefix and not raw.startswith(prefix):
        raw = prefix + raw
    return raw


def _add_dialog_dict(
    title: str,
    prefix: str,
    extra_inputs: list,
    build_skeleton: Callable[[str, dict], dict],
    data: dict,
    collection_key: str,
    commit: Callable,
) -> None:
    with ui.dialog() as dialog, ui.card().classes("min-w-[480px]"):
        ui.label(title).classes("text-base font-semibold")
        id_input = ui.input(label=f"ID (prefix: {prefix})").classes("w-full")
        extras: dict[str, Any] = {}
        for label, key, default in extra_inputs:
            inp = ui.input(label=label, value=default or "").classes("w-full")
            extras[key] = inp

        def _do_add() -> None:
            new_id = _normalise_id(id_input.value, prefix)
            if not new_id:
                ui.notify("ID required", type="warning")
                return
            items = data.setdefault(collection_key, {})
            if new_id in items:
                ui.notify(f"{new_id} already exists", type="warning")
                return
            values = {k: (v.value or "").strip() for k, v in extras.items()}
            items[new_id] = build_skeleton(new_id, values)
            commit(data)
            dialog.close()
            ui.notify(f"Added {new_id}", type="positive")

        with ui.row().classes("justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Add", on_click=_do_add).props("unelevated color=primary")
    dialog.open()


def _add_dialog_list(
    title: str,
    inputs: list,
    build_skeleton: Callable[[dict], dict],
    data: dict,
    collection_key: str,
    commit: Callable,
) -> None:
    with ui.dialog() as dialog, ui.card().classes("min-w-[520px]"):
        ui.label(title).classes("text-base font-semibold")
        widgets: dict[str, Any] = {}
        for label, key, kind, default in inputs:
            if kind == "text":
                w = ui.input(label=label, value=default or "").classes("w-full")
            elif kind == "int":
                w = ui.number(
                    label=label,
                    value=default if default is not None else 0,
                    format="%.0f",
                ).classes("w-full")
            elif kind == "float":
                # Free-form text so the user can leave the field empty
                # to mean "not applicable" (e.g. trait_delta on a
                # chain_reaction edge); the builder validates and
                # coerces. ``ui.number`` would force a value of 0.0
                # which is semantically different from "unset".
                w = ui.input(
                    label=label,
                    value="" if default is None else str(default),
                ).classes("w-full")
            else:
                w = ui.input(label=label, value=default or "").classes("w-full")
            widgets[key] = (w, kind)

        def _do_add() -> None:
            values: dict[str, Any] = {}
            for key, (w, kind) in widgets.items():
                v = w.value
                if kind == "int":
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        ui.notify(f"{key} must be an integer", type="warning")
                        return
                elif kind == "float":
                    raw = (v or "").strip() if isinstance(v, str) else v
                    if raw in ("", None):
                        v = None
                    else:
                        try:
                            v = float(raw)
                        except (TypeError, ValueError):
                            ui.notify(
                                f"{key} must be a number (or blank)",
                                type="warning",
                            )
                            return
                else:
                    v = (v or "").strip()
                values[key] = v
            try:
                skeleton = build_skeleton(values)
            except ValueError as exc:
                ui.notify(str(exc), type="warning")
                return
            items = data.setdefault(collection_key, [])
            items.append(skeleton)
            commit(data)
            dialog.close()
            ui.notify("Added", type="positive")

        with ui.row().classes("justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Add", on_click=_do_add).props("unelevated color=primary")
    dialog.open()


# ---- Per-kind builders --------------------------------------------------

def _add_location(data: dict, commit: Callable, prefix: Optional[str]) -> None:
    _add_dialog_dict(
        title="Add Location",
        prefix=prefix or "LOC_",
        extra_inputs=[
            ("Name", "name", ""),
            ("Description", "description", ""),
        ],
        build_skeleton=lambda lid, v: {
            "world_id": "factual",
            "node_type": "Location",
            "name": v.get("name") or lid,
            "description": v.get("description") or "",
            "ambient_state": {},
        },
        data=data, collection_key="locations", commit=commit,
    )


def _add_entity(data: dict, commit: Callable, prefix: Optional[str]) -> None:
    _add_dialog_dict(
        title="Add Entity",
        prefix=prefix or "ENT_",
        extra_inputs=[
            ("Name", "name", ""),
            ("Initial location_id (LOC_…)", "location_id", ""),
        ],
        build_skeleton=lambda eid, v: {
            "world_id": "factual",
            "id": eid,
            "name": v.get("name") or eid,
            "location_id": v.get("location_id") or "",
            "status": "healthy",
            "traits": {},
            "beliefs": [],
            "constants": [],
            "state_timeline": [],
        },
        data=data, collection_key="entities", commit=commit,
    )


def _add_object(data: dict, commit: Callable, prefix: Optional[str]) -> None:
    _add_dialog_dict(
        title="Add Object",
        prefix=prefix or "OBJ_",
        extra_inputs=[
            ("Name", "name", ""),
            ("location_id (LOC_…) or blank", "location_id", ""),
            ("owner_id (ENT_…) or blank", "owner_id", ""),
        ],
        build_skeleton=lambda oid, v: {
            "world_id": "factual",
            "id": oid,
            "name": v.get("name") or oid,
            "location_id": v.get("location_id") or None,
            "owner_id": v.get("owner_id") or None,
            "properties": {},
            "affordances": [],
        },
        data=data, collection_key="objects", commit=commit,
    )


def _add_world_trait(data: dict, commit: Callable, prefix: Optional[str]) -> None:
    _add_dialog_dict(
        title="Add World Trait",
        prefix=prefix or "WORLD_",
        extra_inputs=[
            ("Name", "name", ""),
            ("Description", "description", ""),
            ("Category", "category", "social_structure"),
        ],
        build_skeleton=lambda wid, v: {
            "world_id": "factual",
            "id": wid,
            "name": v.get("name") or wid,
            "description": v.get("description") or "",
            "category": v.get("category") or "social_structure",
            "magnitude": {"value": 0.5, "inertia": 0.5},
            "affected_domains": [],
            "state_timeline": [],
        },
        data=data, collection_key="world_traits", commit=commit,
    )


def _add_event(data: dict, commit: Callable, prefix: Optional[str]) -> None:
    _add_dialog_list(
        title="Add Event",
        inputs=[
            ("ID (EVT_…)", "id", "text", ""),
            ("Description", "description", "text", ""),
            (
                "event_type (choice/outcome/revelation/utterance)",
                "event_type", "text", "outcome",
            ),
            ("fabula_time (auto)", "fabula_time", "int", _next_fabula_time(data)),
            ("syuzhet_index (auto)", "syuzhet_index", "int", _next_syuzhet_index(data)),
            # Utterance-only fields. Ignored for non-utterance events;
            # required when event_type == "utterance".
            ("utterance: content", "content", "text", ""),
            ("utterance: speaker_id (ENT_…)", "speaker_id", "text", ""),
            (
                "utterance: addressee_ids (comma-separated ENT_…)",
                "addressee_ids", "text", "",
            ),
            ("utterance: via_channel_id (CHN_…, optional)", "via_channel_id", "text", ""),
            (
                "utterance: truth_value (true/false/unknown/performative)",
                "truth_value", "text", "true",
            ),
        ],
        build_skeleton=lambda v: _build_event_skeleton(v, prefix),
        data=data, collection_key="events", commit=commit,
    )


def _build_event_skeleton(v: dict, prefix: Optional[str]) -> dict:
    eid = _normalise_id(v.get("id", ""), prefix or "EVT_")
    if not eid:
        raise ValueError("Event ID required")
    et = (v.get("event_type") or "outcome").strip()
    if et not in ("choice", "outcome", "revelation", "utterance"):
        raise ValueError(
            "event_type must be choice/outcome/revelation/utterance"
        )
    skel = {
        "world_id": "factual",
        "id": eid,
        "fabula_time": v["fabula_time"],
        "syuzhet_index": v["syuzhet_index"],
        "event_type": et,
        "actor_ids": [],
        "target_ids": [],
        "description": v.get("description", ""),
    }
    if et == "utterance":
        speaker = (v.get("speaker_id") or "").strip()
        if not speaker:
            raise ValueError("utterance events require speaker_id")
        addressees = [
            a.strip() for a in (v.get("addressee_ids") or "").split(",")
            if a.strip()
        ]
        if not addressees:
            raise ValueError(
                "utterance events require at least one addressee_id"
            )
        truth = (v.get("truth_value") or "true").strip()
        if truth not in ("true", "false", "unknown", "performative"):
            raise ValueError(
                "truth_value must be true/false/unknown/performative"
            )
        channel = (v.get("via_channel_id") or "").strip() or None
        skel["content"] = v.get("content") or ""
        skel["speaker_id"] = speaker
        skel["addressee_ids"] = addressees
        skel["via_channel_id"] = channel
        skel["truth_value"] = truth
        # Mirror the speaker into actor_ids so causal traversal still
        # finds them; the speaker is by definition acting.
        skel["actor_ids"] = [speaker]
        skel["target_ids"] = list(addressees)
    return skel


def _add_causal_edge(data: dict, commit: Callable, prefix: Optional[str]) -> None:
    _add_dialog_list(
        title="Add Causal Edge",
        inputs=[
            ("source_id", "source_id", "text", ""),
            ("target_id", "target_id", "text", ""),
            (
                "causality_type (chain_reaction/mutation/mutation_social/"
                "affordance_gate/ambient_propagation)",
                "causality_type", "text", "chain_reaction",
            ),
            ("mechanism", "mechanism", "text", "physical"),
            ("fabula_time", "fabula_time", "int", _next_fabula_time(data) - 100 or 0),
            # mutation / mutation_social only. trait_target =
            # affected trait name (for mutation, e.g. 'guilt') OR
            # the social axis (for mutation_social: 'affinity',
            # 'fear', 'power_dynamic'). trait_delta is the signed
            # change. rel_counterpart_id is the *other* entity in
            # the dyad and is REQUIRED for mutation_social by the
            # CausalEdge model validator.
            (
                "trait_target (mutation only — trait name or social axis)",
                "trait_target", "text", "",
            ),
            (
                "trait_delta (mutation only — signed magnitude, blank=N/A)",
                "trait_delta", "float", None,
            ),
            (
                "rel_counterpart_id (mutation_social only — other ENT_ in dyad)",
                "rel_counterpart_id", "text", "",
            ),
        ],
        build_skeleton=_build_causal_skeleton,
        data=data, collection_key="causal_topology", commit=commit,
    )


def _build_causal_skeleton(v: dict) -> dict:
    if not v.get("source_id") or not v.get("target_id"):
        raise ValueError("source_id and target_id are required")
    ct = v.get("causality_type") or "chain_reaction"
    valid_ct = {
        "chain_reaction", "mutation", "mutation_social",
        "affordance_gate", "ambient_propagation",
    }
    if ct not in valid_ct:
        raise ValueError(f"causality_type must be one of {sorted(valid_ct)}")
    trait_target = (v.get("trait_target") or "").strip() or None
    trait_delta = v.get("trait_delta")  # already float|None from dialog
    rel_counterpart = (v.get("rel_counterpart_id") or "").strip() or None
    if ct == "mutation_social":
        if not rel_counterpart:
            raise ValueError(
                "mutation_social requires rel_counterpart_id (the other "
                "ENT_ in the dyad)."
            )
        if not trait_target:
            # The model accepts None but the social propagator and
            # auditor both key on this; warn the user up-front.
            raise ValueError(
                "mutation_social requires trait_target — one of "
                "'affinity', 'fear', 'power_dynamic'."
            )
        if trait_target not in ("affinity", "fear", "power_dynamic"):
            raise ValueError(
                "mutation_social trait_target must be one of "
                "'affinity', 'fear', 'power_dynamic'."
            )
    return {
        "world_id": "factual",
        "source_id": v["source_id"],
        "target_id": v["target_id"],
        "causality_type": ct,
        "causal_force": 5.0,
        "mechanism": v.get("mechanism") or "physical",
        "evidence_strength": "moderate",
        "propagation_delay": 0,
        "fabula_time": v["fabula_time"],
        "trait_target": trait_target,
        "trait_delta": trait_delta,
        "rel_counterpart_id": rel_counterpart,
    }


def _add_social_edge(data: dict, commit: Callable, prefix: Optional[str]) -> None:
    _add_dialog_list(
        title="Add Relationship Edge",
        inputs=[
            ("source_entity_id (ENT_…)", "source_entity_id", "text", ""),
            ("target_entity_id (ENT_…)", "target_entity_id", "text", ""),
        ],
        build_skeleton=lambda v: _ensure_endpoints(v, ["source_entity_id", "target_entity_id"]) or {
            "world_id": "factual",
            "source_entity_id": v["source_entity_id"],
            "target_entity_id": v["target_entity_id"],
            # Build the edge in the new per-axis shape with an empty
            # ``metrics`` dict — the user adds the axes they actually
            # observed via the editor. Seeding flat ``affinity=0.0`` etc.
            # would trip the legacy migrator into materialising every
            # axis with ``observed=True``, indistinguishable from an
            # LLM extraction that genuinely measured all three at zero.
            "metrics": {},
        },
        data=data, collection_key="social_topology", commit=commit,
    )


def _add_spatial_edge(data: dict, commit: Callable, prefix: Optional[str]) -> None:
    _add_dialog_list(
        title="Add Spatial Edge",
        inputs=[
            ("source_id (LOC_…)", "source_id", "text", ""),
            ("target_id (LOC_…)", "target_id", "text", ""),
        ],
        build_skeleton=lambda v: _ensure_endpoints(v, ["source_id", "target_id"]) or {
            "world_id": "factual",
            "source_id": v["source_id"],
            "target_id": v["target_id"],
            "is_locked": False,
            "barrier_item_id": None,
            "established_at_fabula": 0,
            "destroyed_at_fabula": None,
        },
        data=data, collection_key="spatial_topology", commit=commit,
    )


def _add_channel(
    data: dict, commit: Callable, prefix: Optional[str],
) -> None:
    _add_dialog_dict(
        title="Add Channel",
        prefix=prefix or "CHN_",
        extra_inputs=[
            ("name", "name", ""),
            ("medium", "medium", "speech"),
            ("participant_ids (comma-separated)", "participant_ids", ""),
            ("directionality (broadcast|duplex|simplex)", "directionality", "duplex"),
            ("established_at_fabula", "established_at_fabula", "0"),
            (
                "intelligibility (JSON, e.g. {\"ENT_X\": 0.4})",
                "intelligibility", "{}",
            ),
        ],
        build_skeleton=_build_channel_skeleton,
        data=data, collection_key="channels", commit=commit,
    )


def _build_channel_skeleton(new_id: str, v: dict) -> dict:
    raw_pids = v.get("participant_ids") or ""
    pids = [p.strip() for p in raw_pids.split(",") if p.strip()]
    if len(pids) < 2:
        raise ValueError("at least two participant_ids required")
    direction = v.get("directionality") or "duplex"
    if direction not in ("broadcast", "duplex", "simplex"):
        raise ValueError("directionality must be broadcast | duplex | simplex")
    try:
        established = int(v.get("established_at_fabula") or 0)
    except (TypeError, ValueError):
        established = 0
    raw_intel = v.get("intelligibility") or "{}"
    try:
        intel = json.loads(raw_intel) if isinstance(raw_intel, str) else dict(raw_intel)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"intelligibility must be JSON: {exc}") from exc
    if not isinstance(intel, dict):
        raise ValueError("intelligibility must be a JSON object")
    norm_intel: dict[str, float] = {}
    for k, val in intel.items():
        try:
            f = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"intelligibility[{k}] must be numeric, got {val!r}"
            ) from exc
        if not 0.0 <= f <= 1.0:
            raise ValueError(
                f"intelligibility[{k}]={f} out of range [0.0, 1.0]"
            )
        norm_intel[str(k)] = f
    return {
        "id": new_id,
        "world_id": "factual",
        "name": v.get("name") or new_id,
        "medium": v.get("medium") or "speech",
        "participant_ids": pids,
        "directionality": direction,
        "intelligibility": norm_intel,
        "established_at_fabula": established,
        "terminated_at_fabula": None,
        "evidence_strength": "moderate",
    }


def _ensure_endpoints(v: dict, keys: list) -> None:
    for k in keys:
        if not v.get(k):
            raise ValueError(f"{k} required")
    return None  # falsy → triggers `or {…}` build


_ADD_BUILDERS: dict[str, Callable] = {
    "locations": _add_location,
    "entities": _add_entity,
    "objects": _add_object,
    "world_traits": _add_world_trait,
    "events": _add_event,
    "causal_topology": _add_causal_edge,
    "social_topology": _add_social_edge,
    "spatial_topology": _add_spatial_edge,
    "channels": _add_channel,
}


def _user_can_edit(state: AppState) -> bool:
    """True if the current user owns or has editor access on the project."""
    if state.project_id is None or state.user_id is None:
        return False
    proj = db.get_project(state.project_id)
    if proj is None:
        return False
    if proj.owner_id == state.user_id:
        return True
    role = db.get_user_project_role(state.project_id, state.user_id)
    return role in ("editor", "admin")


def _confirm_save(state: AppState, edited_json: str, current_json: str) -> None:
    """Validate edited JSON and prompt before persisting a new version.

    Runs in two phases:
      1. Pydantic schema validation (`WorldStateV1.model_validate_json`).
      2. Cross-reference / consistency checks via the same
         ``_programmatic_validation`` used by the ingestion pipeline:
         hallucinated/missing IDs, broken edge endpoints, snapshot
         ``triggered_by`` pointing at unknown events, monotonic
         state_timeline ordering, syuzhet/fabula time consistency,
         dead-actor references, etc.

    Any **error**-severity consistency issue blocks the save. Warnings
    are shown in the confirmation dialog and the user can choose to
    save anyway (e.g. orphan events, low info-edge density).
    """
    edited_json = (edited_json or "").strip()
    if not edited_json:
        ui.notify("Editor is empty", type="warning")
        return
    if edited_json == current_json.strip():
        ui.notify("No changes detected", type="info")
        return

    # --- Phase 1: Pydantic schema validation ----------------------------------
    try:
        new_ws = WorldStateV1.model_validate_json(edited_json)
    except ValidationError as exc:
        errs = exc.errors()
        head = errs[:3]
        msg_lines = [
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in head
        ]
        more = f"\n(+{len(errs) - len(head)} more)" if len(errs) > len(head) else ""
        ui.notify(
            "Schema validation failed:\n" + "\n".join(msg_lines) + more,
            type="negative",
            multi_line=True,
            timeout=8000,
        )
        return
    except ValueError as exc:  # JSON parse error
        ui.notify(f"Invalid JSON: {exc}", type="negative")
        return

    # --- Phase 2: Cross-reference / consistency checks -----------------------
    try:
        issues = _programmatic_validation(new_ws)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Consistency validation crashed")
        ui.notify(f"Consistency check failed to run: {exc}", type="negative")
        return

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if errors:
        # Hard block — show every error so the user can fix them.
        _show_issues_dialog(
            title=f"{len(errors)} consistency error(s) — cannot save",
            subtitle=(
                "IDs, edge endpoints or timeline references are broken. "
                "Fix these and try again."
            ),
            errors=errors,
            warnings=warnings,
            blocking=True,
            on_save=None,
        )
        return

    # No errors — show summary + any warnings, allow save.
    def _proceed() -> None:
        _do_save(state, new_ws)

    _show_issues_dialog(
        title="Save edited world model as new version?",
        subtitle=(
            f"Validated successfully — {len(new_ws.entities)} entities, "
            f"{len(new_ws.events)} events, {len(new_ws.locations)} locations. "
            "A new version will branch from the current one."
        ),
        errors=[],
        warnings=warnings,
        blocking=False,
        on_save=_proceed,
    )


def _show_issues_dialog(
    *,
    title: str,
    subtitle: str,
    errors: list,
    warnings: list,
    blocking: bool,
    on_save,
) -> None:
    """Render an issues dialog. If ``blocking`` is True, no save button is shown."""
    with ui.dialog() as dialog, ui.card().classes("min-w-[600px] max-w-[800px]"):
        ui.label(title).classes("text-base font-semibold")
        ui.label(subtitle).classes("text-xs text-slate-600")

        if errors:
            ui.label(f"Errors ({len(errors)}):").classes(
                "text-sm font-semibold text-negative mt-2"
            )
            with ui.column().classes(
                "w-full max-h-[30vh] overflow-auto gap-1 "
                "border border-red-200 rounded p-2 bg-red-50"
            ):
                for issue in errors:
                    ui.label(f"• [{issue.category}] {issue.detail}").classes(
                        "text-xs text-red-900"
                    )

        if warnings:
            ui.label(f"Warnings ({len(warnings)}):").classes(
                "text-sm font-semibold text-warning mt-2"
            )
            with ui.column().classes(
                "w-full max-h-[30vh] overflow-auto gap-1 "
                "border border-amber-200 rounded p-2 bg-amber-50"
            ):
                for issue in warnings:
                    ui.label(f"• [{issue.category}] {issue.detail}").classes(
                        "text-xs text-amber-900"
                    )
            if not blocking:
                ui.label(
                    "Warnings will not block saving but should be reviewed."
                ).classes("text-xs text-slate-500 italic")

        with ui.row().classes("justify-end gap-2 mt-2"):
            ui.button("Close" if blocking else "Cancel", on_click=dialog.close).props(
                "flat"
            )
            if not blocking and on_save is not None:
                save_label = (
                    "Save Anyway" if warnings else "Save Version"
                )
                ui.button(
                    save_label,
                    on_click=lambda: (dialog.close(), on_save()),
                ).props("unelevated color=primary")
    dialog.open()


def _do_save(state: AppState, new_ws: WorldStateV1) -> None:
    """Persist the validated world state as a new branched version."""
    project_id = state.project_id
    user_id = state.user_id
    if project_id is None:
        ui.notify("No active project", type="negative")
        return

    parent_version_row_id = state.current_version_row_id

    try:
        new_ver = db.save_version(
            project_id=project_id,
            world_state_json=new_ws.model_dump_json(),
            ancestor_id=parent_version_row_id,
            source="manual_edit",
            description="Manual world-model edit",
            user_id=user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual edit save failed")
        ui.notify(f"Save failed: {exc}", type="negative")
        return

    # Atomic swap: resets cursors and emits WORLD_STATE_CHANGED /
    # VERSION_CHANGED so every panel updates in lockstep.
    state.load_db_version(new_ws, new_ver.id, version_number=new_ver.version)

    # Mirror into the active-version pointer so MCP read tools default
    # to the freshly saved version.
    if user_id is not None:
        try:
            db.set_active_version(project_id, user_id, new_ver.id)
        except Exception:
            logger.exception(
                "Failed to update active-version pointer after manual edit"
            )

    ui.notify(
        f"Saved as v{new_ver.version} (manual edit)",
        type="positive",
    )
