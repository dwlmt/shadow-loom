"""Story tab — prose reader, generation results, and source text.

Reading pane shows all generated prose with constraint badges.
Collapsible source text and quick-reference sidebar.
Chat/query input lives in the bottom drawer (not here).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_story_tab(state: AppState) -> None:
    """Build the Story tab — prose reading + generation display."""

    with ui.column().classes("w-full h-full q-pa-md gap-3"):
        # ── Source text (collapsible) ─────────────────────────────
        if state.raw_text:
            with ui.expansion("Source Text", icon="description").classes("w-full"):
                ui.markdown(
                    state.raw_text[:8000]
                    + ("..." if len(state.raw_text) > 8000 else "")
                )

        # ── Generated Prose ───────────────────────────────────────
        prose_container = ui.column().classes("w-full gap-3")
        _render_prose(state, prose_container)

        # ── Subscribe ─────────────────────────────────────────────
        state.on(
            StateEvent.PIPELINE_RESULT,
            lambda **kw: _render_prose(state, prose_container),
        )


def _render_prose(state: AppState, container) -> None:
    """Render all prose from query history."""
    container.clear()

    prose_entries = []
    for result in state.query_history:
        pr = result.pipeline_result
        if pr and pr.prose:
            prose_entries.append((pr.query_type, pr.prose, pr.converged, pr.audit_iterations))

    if not prose_entries:
        with container:
            with ui.column().classes("w-full items-center q-pa-xl"):
                ui.icon("menu_book", size="xl", color="grey")
                ui.label("No prose generated yet").classes("text-body1 text-grey")
                ui.label(
                    "Use the Chat bar below to ask questions, run directives, "
                    "or write continuations."
                ).classes("text-caption text-grey")
        return

    with container:
        for i, (qtype, prose, converged, iters) in enumerate(prose_entries):
            with ui.card().classes("w-full"):
                # Header with badges
                with ui.row().classes("items-center gap-2 q-mb-sm"):
                    ui.badge(qtype, color="primary").props("dense")
                    if converged is not None:
                        color = "positive" if converged else "warning"
                        label = "converged" if converged else f"unconverged ({iters} iters)"
                        ui.badge(label, color=color).props("dense outline")

                # Prose content
                ui.markdown(prose)
