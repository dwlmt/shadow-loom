"""Story tab — prose reader, generation results, and source text.

Reading pane shows all generated prose from DB + session history.
Collapsible source text. Prompt starters for common writer actions.
Chat/query input lives in the bottom command bar (not here).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent
from shadow_loom_ui.theme import feather

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_story_tab(state: AppState) -> None:
    """Build the Story tab — prose reading + generation display."""

    with ui.column().classes("w-full h-full p-6 gap-3 bg-slate-50"):
        # ── Source text (collapsible) ───────────────────────
        if state.raw_text:
            with ui.expansion("Source Text", icon="description").classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm"
            ):
                ui.markdown(
                    state.raw_text[:8000]
                    + ("..." if len(state.raw_text) > 8000 else "")
                )

        # ── Quick NL prompts for writers ──────────────────────────
        with ui.row().classes("w-full gap-2 flex-wrap q-pa-xs"):
            _build_prompt_starters(state)

        # ── Generated Prose ───────────────────────────────────────
        prose_container = ui.column().classes("w-full gap-3")
        _render_prose(state, prose_container)

        # ── Subscribe ─────────────────────────────────────────────
        state.on(
            StateEvent.PIPELINE_RESULT,
            lambda **kw: _render_prose(state, prose_container),
        )
        state.on(
            StateEvent.PROJECT_LOADED,
            lambda **kw: _render_prose(state, prose_container),
        )


def _build_prompt_starters(state: AppState) -> None:
    """Writer-friendly prompt starter chips that populate the command bar."""

    starters = [
        ("Continue the story\u2026", "observation", "book-open"),
        ("What would happen if\u2026", "counterfactual", "git-branch"),
        ("Make this scene more suspenseful", "directive", "film"),
        ("Why does this character\u2026", "interrogate", "cpu"),
        ("Evaluate the story quality", "evaluate", "check-square"),
    ]

    for label, qtype, icon in starters:
        ui.button(
            label,
            icon=icon,
            on_click=lambda l=label, q=qtype: _trigger_prompt(state, l, q),
        ).props("outline dense no-caps size=sm color=primary").classes(
            "rounded-lg"
        )


def _trigger_prompt(state: AppState, prompt_text: str, query_type: str) -> None:
    """Emit a prompt suggestion that the command bar will pick up."""
    state.emit(StateEvent.QUERY_STARTED, suggestion=prompt_text, query_type=query_type)


def _render_prose(state: AppState, container) -> None:
    """Render all prose from DB history + current session."""
    container.clear()

    prose_entries: list[tuple[str, str, bool | None, int]] = []

    # Load from DB if project is active
    if state.project_id:
        try:
            from shadow_loom_ui.db import get_all_prose
            db_prose = get_all_prose(state.project_id)
            for entry in db_prose:
                # Skip if it'll be duplicated from session history
                prose_entries.append((
                    entry.get("source", "pipeline"),
                    entry["prose"],
                    None,  # DB doesn't store convergence directly
                    0,
                ))
        except Exception:
            logger.exception("Failed to load prose from DB")

    # Overlay session results (may duplicate some DB entries, but ensures freshness)
    session_prose_set = set()
    for result in state.query_history:
        pr = result.pipeline_result
        if pr and pr.prose:
            # De-duplicate against DB entries by content hash
            key = pr.prose[:200]
            if key not in {p[1][:200] for p in prose_entries}:
                prose_entries.append((pr.query_type, pr.prose, pr.converged, pr.audit_iterations))
            session_prose_set.add(key)

    if not prose_entries:
        with container:
            with ui.card().classes(
                "w-full items-center bg-white border border-slate-200 "
                "rounded-xl shadow-sm p-12 gap-2"
            ):
                feather("book-open", size="xl", color="#94a3b8")
                ui.label("No prose generated yet").classes(
                    "text-base font-medium text-slate-600"
                )
                ui.label(
                    "Use the prompt starters above or the command bar below to "
                    "ask questions, run directives, or write continuations."
                ).classes("text-sm text-slate-400 text-center max-w-md")
        return

    with container:
        for i, (qtype, prose, converged, iters) in enumerate(prose_entries):
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl "
                "shadow-sm p-6"
            ):
                # Header with badges
                with ui.row().classes("items-center gap-2 mb-3"):
                    ui.badge(qtype, color="primary").props("dense")
                    if converged is not None:
                        color = "positive" if converged else "warning"
                        label = "converged" if converged else f"unconverged ({iters} iters)"
                        ui.badge(label, color=color).props("dense outline")

                # Prose content
                ui.markdown(prose)
