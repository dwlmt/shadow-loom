# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Research tab — segregated background facts + per-project topic config.

Surfaces the optional web-research feature (``shadow_loom.research``):

* Status strip — provider, enabled flag, API-key presence (read from
  :func:`shadow_loom.settings.get_settings` via the MCP-shared layer).
* Per-project topics editor (chip list + add input). Persisted to
  :class:`shadow_loom.db.ProjectSettingsRow`. **Editing topics never
  mutates a version** — research config is intentionally outside
  ``WorldStateV1``.
* "Research now" actions — iterate the topic list (or run an ad-hoc
  topic) calling :func:`shadow_loom.research.lookup_and_persist_topic`
  off the UI thread via :mod:`asyncio`.
* Facts list — cards rendered from :class:`shadow_loom.db.WorldFactRow`.
  Each card carries source URL, confidence chip, expandable raw
  snippets, and a delete button.

Visual treatment is deliberately distinct from the version-tree green /
shadow violet — facts are **background-only context** and must not be
mistaken for canonical narrative state. See ``CONTENT-POLICY.md`` §6.4a
and ``docs/architecture.md`` Step 3d.
"""

from __future__ import annotations

import asyncio
import json
import logging

from nicegui import ui

from shadow_loom import db
from shadow_loom.research import lookup_and_persist_topic
from shadow_loom.settings import get_settings
from shadow_loom_ui.state import AppState, StateEvent

logger = logging.getLogger(__name__)

# Muted parchment accent — distinct from version-tree green / shadow violet.
_ACCENT = "#8a6d3b"
_ACCENT_BG = "#fdf8ef"
_ACCENT_BORDER = "#e8d9b3"

_RESEARCH_TIMEOUT_S = 120.0


def build_research_tab(state: AppState) -> None:
    """Build the Research tab.

    Layout: status strip → topics editor → run actions → facts list.
    All long-running provider calls are routed through
    :meth:`AppState.spawn_panel_task` with a hard timeout so the user
    can navigate away without orphaning a spinner.
    """
    with ui.column().classes("w-full h-full p-4 gap-4 bg-slate-50"):
        # ── Status strip ─────────────────────────────────────────
        status_row = ui.row().classes(
            "w-full items-center gap-3 p-3 rounded border"
        ).style(
            f"background-color: {_ACCENT_BG}; border-color: {_ACCENT_BORDER};"
        )
        with status_row:
            ui.icon("menu_book").style(f"color: {_ACCENT}")
            ui.label("Research / Background Facts").classes(
                "text-base font-semibold"
            ).style(f"color: {_ACCENT}")
            ui.space()
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Research — background facts, never canon",
                body_md=(
                    "Per-project reference material looked up from"
                    " external providers (web, knowledge bases). Used"
                    " by the engine as *background context* when"
                    " generating prose, **never** written into the"
                    " world model as canonical events.\n\n"
                    "### What you can do\n"
                    "- **Manage topics** — add, remove, or edit the"
                    " research topics for this project (e.g."
                    " *Edinburgh in 1040*, *Scottish succession"
                    " law*). Each topic is queried independently.\n"
                    "- **Run lookup** for one or all topics. Provider"
                    " calls run as background tasks — you can"
                    " navigate away and come back.\n"
                    "- **Browse facts** — each retrieved fact carries"
                    " its source URL and confidence; click to expand.\n\n"
                    "### Segregation policy\n"
                    "Research facts and the world model are kept in"
                    " **separate stores**:\n"
                    "- The pipeline injects research as *flavour"
                    " notes* into the prompt for prose generation.\n"
                    "- The re-extraction step that runs on the"
                    " generated prose **does not** read from research,"
                    " so research can never accidentally become a new"
                    " canonical event or relationship.\n"
                    "- Deleting a research fact does not delete any"
                    " version or canon prose.\n\n"
                    "### Status strip\n"
                    "Shows whether a provider lookup is currently"
                    " in-flight and the elapsed wall-clock time."
                ),
                tooltip="What is this tab?",
            )
            status_label = ui.label("").classes("text-xs text-slate-600")

        # Empty-state copy + segregation reminder.
        ui.label(
            "Research facts are project-wide reference material. They are "
            "stored separately from entities, events, and channels and are "
            "never written into the canonical narrative."
        ).classes("text-xs text-slate-500 italic")

        # ── Topics editor ────────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("label").style(f"color: {_ACCENT}")
                ui.label("Project topics").classes("text-sm font-semibold")
                ui.space()
                ui.label("Editing topics does not change the world state.").classes(
                    "text-xs text-slate-400 italic"
                )
            chips_row = ui.row().classes("w-full gap-1 flex-wrap")
            with ui.row().classes("w-full items-center gap-2"):
                topic_input = ui.input(
                    placeholder="Add a topic (e.g. 'Roaring Twenties')",
                ).classes("flex-grow")
                add_topic_btn = ui.button(
                    "Add", icon="add",
                ).props("dense color=primary")
                run_all_btn = ui.button(
                    "Research all topics now", icon="travel_explore",
                ).props("dense color=accent")

        # ── Ad-hoc lookup ────────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("search").style(f"color: {_ACCENT}")
                ui.label("Ad-hoc lookup").classes("text-sm font-semibold")
            with ui.row().classes("w-full items-center gap-2"):
                adhoc_input = ui.input(
                    placeholder="One-off topic — does not get added to the project list",
                ).classes("flex-grow")
                adhoc_btn = ui.button(
                    "Look up", icon="public",
                ).props("dense color=primary")

        # ── Facts list ───────────────────────────────────────────
        with ui.card().classes("w-full flex-grow"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("collections_bookmark").style(f"color: {_ACCENT}")
                facts_header = ui.label("Background facts").classes(
                    "text-sm font-semibold"
                )
                ui.space()
                ui.label("Not audited.").classes("text-xs text-slate-400 italic")
            facts_container = ui.column().classes("w-full gap-2")

    # ── Helpers ───────────────────────────────────────────────────
    def _refresh_status() -> None:
        try:
            settings = get_settings()
            ext = settings.extraction
            provider = ext.research_provider
            api_key_present = bool(getattr(settings.core, "tavily_api_key", "") or "")
            enabled = bool(ext.enable_research_agent)
        except Exception:
            logger.exception("[Research tab] failed to read settings")
            status_label.text = "status unavailable"
            return
        bits = [
            f"provider={provider}",
            f"enabled={'yes' if enabled else 'no'}",
            f"key={'set' if api_key_present else 'missing'}",
        ]
        status_label.text = " · ".join(bits)
        # Disable run actions if not usable.
        usable = enabled and provider != "none" and api_key_present
        for b in (run_all_btn, add_topic_btn, adhoc_btn):
            b.enabled = True  # always allow editing topics; only run buttons need provider
        run_all_btn.enabled = usable
        adhoc_btn.enabled = usable
        if not usable:
            tip = "Research provider not configured (see docs/render-deployment.md or docs/railway-deployment.md)."
            run_all_btn.tooltip(tip)
            adhoc_btn.tooltip(tip)

    def _current_topics() -> list[str]:
        if state.project_id is None:
            return []
        try:
            return list(db.get_project_settings(state.project_id).get("research_topics", []))
        except Exception:
            logger.exception("[Research tab] get_project_settings failed")
            return []

    def _refresh_chips() -> None:
        chips_row.clear()
        topics = _current_topics()
        if not topics:
            with chips_row:
                ui.label("No topics yet — add one below.").classes(
                    "text-xs text-slate-400 italic"
                )
            return
        with chips_row:
            for t in topics:
                chip = ui.chip(t, removable=True, color="grey-3").props("dense")
                chip.on("remove", lambda _e, _t=t: _remove_topic(_t))

    def _persist_topics(topics: list[str]) -> None:
        if state.project_id is None:
            ui.notify("No project loaded.", type="warning")
            return
        try:
            db.set_project_settings(state.project_id, research_topics=topics)
        except Exception as e:
            logger.exception("[Research tab] set_project_settings failed")
            ui.notify(f"Save failed: {e}", type="negative")

    def _add_topic() -> None:
        new_t = (topic_input.value or "").strip()
        if not new_t:
            return
        topics = _current_topics()
        if new_t in topics:
            ui.notify(f"Already in list: {new_t}", type="info")
            topic_input.set_value("")
            return
        topics.append(new_t)
        _persist_topics(topics)
        topic_input.set_value("")
        _refresh_chips()

    def _remove_topic(topic: str) -> None:
        topics = [t for t in _current_topics() if t != topic]
        _persist_topics(topics)
        _refresh_chips()

    def _refresh_facts() -> None:
        facts_container.clear()
        if state.project_id is None:
            with facts_container:
                ui.label("No project loaded.").classes(
                    "text-xs text-slate-400 italic"
                )
            facts_header.text = "Background facts"
            return
        try:
            rows = db.list_world_facts(state.project_id)
        except Exception:
            logger.exception("[Research tab] list_world_facts failed")
            rows = []
        facts_header.text = f"Background facts ({len(rows)})"
        if not rows:
            with facts_container:
                ui.label(
                    "No facts yet. Use 'Research all topics now' or the "
                    "ad-hoc lookup above to populate this list."
                ).classes("text-xs text-slate-400 italic")
            return
        with facts_container:
            for row in rows:
                _render_fact_card(row)

    def _render_fact_card(row) -> None:  # noqa: ANN001
        confidence = (row.confidence or "moderate").lower()
        conf_color = {
            "high": "positive",
            "moderate": "primary",
            "low": "warning",
        }.get(confidence, "secondary")
        with ui.card().classes("w-full").style(
            f"border-left: 4px solid {_ACCENT};"
        ):
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("article").style(f"color: {_ACCENT}")
                ui.label(row.topic).classes("text-sm font-semibold")
                ui.chip(confidence, color=conf_color).props("dense")
                ui.space()
                if row.source_url_primary:
                    ui.link("source", row.source_url_primary, new_tab=True).props(
                        'rel="noopener noreferrer"'
                    ).classes("text-xs")
                ui.button(
                    icon="delete",
                    on_click=lambda _e, fid=row.fact_id: _delete_fact(fid),
                ).props("flat dense round size=sm color=negative")
            ui.label(row.summary).classes("text-sm text-slate-700")
            try:
                snippets = json.loads(row.raw_snippets_json or "[]")
            except Exception:
                snippets = []
            if snippets:
                with ui.expansion(
                    f"{len(snippets)} source snippet(s)", icon="expand_more",
                ).classes("w-full text-xs"):
                    for s in snippets:
                        with ui.column().classes("gap-0 mb-2"):
                            ui.label(s.get("title", "")).classes(
                                "text-xs font-semibold"
                            )
                            url = s.get("url", "")
                            if url:
                                ui.link(url, url, new_tab=True).props(
                                    'rel="noopener noreferrer"'
                                ).classes("text-xs text-slate-500")
                            ui.label(s.get("content", "")).classes(
                                "text-xs text-slate-600 whitespace-pre-line"
                            )

    def _delete_fact(fact_id: str) -> None:
        if state.project_id is None:
            return
        try:
            db.delete_world_fact(project_id=state.project_id, fact_id=fact_id)
        except Exception as e:
            logger.exception("[Research tab] delete_world_fact failed")
            ui.notify(f"Delete failed: {e}", type="negative")
            return
        ui.notify(f"Deleted {fact_id}", type="info")
        state.emit(StateEvent.WORLD_FACTS_CHANGED)
        _refresh_facts()

    async def _run_lookup(topic: str) -> None:
        """Run a single provider+agent lookup off the UI thread."""
        if state.project_id is None or state.user_id is None:
            ui.notify("No project / user loaded.", type="warning")
            return
        pid = state.project_id
        uid = state.user_id
        ui.notify(f"Researching: {topic}…", type="info")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    lookup_and_persist_topic,
                    project_id=pid,
                    user_id=uid,
                    topic=topic,
                ),
                timeout=_RESEARCH_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            ui.notify(
                f"Timed out after {int(_RESEARCH_TIMEOUT_S)}s: {topic}",
                type="negative",
            )
            return
        except Exception as e:
            logger.exception("[Research tab] lookup failed for %r", topic)
            ui.notify(f"Lookup failed: {e}", type="negative")
            return
        # Drop result if the user switched projects while it ran.
        if state.project_id != pid:
            return
        if "error" in result:
            ui.notify(result["error"], type="negative")
            return
        ui.notify(
            f"Saved {result.get('fact_id', 'fact')}: {topic}",
            type="positive",
        )
        state.emit(StateEvent.WORLD_FACTS_CHANGED)
        _refresh_facts()

    async def _run_all_topics() -> None:
        topics = _current_topics()
        if not topics:
            ui.notify("No topics configured for this project.", type="info")
            return
        for t in topics:
            await _run_lookup(t)

    def _on_add_clicked() -> None:
        _add_topic()

    def _on_run_all_clicked() -> None:
        state.spawn_panel_task("research", _run_all_topics())

    def _on_adhoc_clicked() -> None:
        topic = (adhoc_input.value or "").strip()
        if not topic:
            return
        adhoc_input.set_value("")
        state.spawn_panel_task("research", _run_lookup(topic))

    add_topic_btn.on("click", lambda _e: _on_add_clicked())
    topic_input.on("keydown.enter", lambda _e: _on_add_clicked())
    run_all_btn.on("click", lambda _e: _on_run_all_clicked())
    adhoc_btn.on("click", lambda _e: _on_adhoc_clicked())
    adhoc_input.on("keydown.enter", lambda _e: _on_adhoc_clicked())

    # ── Subscriptions ─────────────────────────────────────────────
    def _on_project_loaded(**_kw):
        _refresh_status()
        _refresh_chips()
        _refresh_facts()

    def _on_facts_changed(**_kw):
        _refresh_facts()

    state.on(StateEvent.PROJECT_LOADED, _on_project_loaded)
    state.on(StateEvent.WORLD_FACTS_CHANGED, _on_facts_changed)

    # Initial paint.
    _refresh_status()
    _refresh_chips()
    _refresh_facts()
