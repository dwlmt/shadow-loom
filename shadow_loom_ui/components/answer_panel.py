# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Answer panel — surfaces the latest Ask / Interrogation result.

Read-only Q&A queries (``general`` / ``interrogate``) intentionally do
not advance the world model and do not write prose. Without a
dedicated surface their answers used to be buried in transient toast
notifications, which made them feel like the engine was ignoring the
user's question. This panel lives at the bottom of the Story tab,
beneath the source text and any generated prose, and shows the
structured response card (claim · evidence · confidence · caveats)
for the most recent Q&A run.

The panel clears whenever:
  * a new project is loaded (``PROJECT_LOADED``)
  * the user picks a different version in the left sidebar
    (``VERSION_CHANGED``)

so it never lingers with stale answers that no longer match the
viewer's current world-state context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.reasoning_helpers import structured_response_data
from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent
from shadow_loom_ui.components._safe_md import safe_markdown

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_READONLY_TYPES = {"general", "interrogate"}


def build_answer_panel(state: AppState) -> None:
    """Render the Answer panel inside the Story tab.

    The panel is hidden entirely when there is no answer to show, so
    it never wastes vertical real estate. It re-appears whenever an
    Ask / Interrogation query completes, and is cleared on project /
    version switches so a stale answer never lingers next to a world
    state it no longer matches.
    """
    container = ui.column().classes(
        "w-full gap-0 rounded-xl border border-slate-200 bg-white shadow-sm"
    )

    # Mutable state captured by closures — the latest result and
    # whether the panel is currently expanded.
    last_result: dict[str, NLQueryResult | None] = {"value": None}

    def _render() -> None:
        try:
            container.clear()
            result = last_result["value"]
            if result is None:
                container.set_visibility(False)
                return
            container.set_visibility(True)
            with container:
                _render_card(state, result, last_result, _render)
        except RuntimeError as exc:
            # Background tasks may complete after the user navigated
            # away from the workspace (the owning client is dead).
            logger.debug(
                "[answer_panel] render skipped (dead client): %s", exc,
            )

    def _on_pipeline_result(**kwargs) -> None:
        result: NLQueryResult | None = kwargs.get("result")
        if result is None or result.pipeline_result is None:
            return
        if result.pipeline_result.query_type not in _READONLY_TYPES:
            # Generative queries surface their results in the Story
            # tab; the Answer panel is reserved for Q&A runs so the
            # two never compete for the same screen real estate.
            return
        last_result["value"] = result
        _render()

    def _clear(**kwargs) -> None:
        # Only clear on navigation (sidebar click, rollback, project
        # load). A ``source="query_save"`` VERSION_CHANGED is emitted
        # when ``_save_version_to_db`` lands a new child version on
        # the active branch — clearing on it would wipe a still-valid
        # Q&A answer the moment the user runs any unrelated
        # generative query. Default (no source kwarg) clears, so
        # PROJECT_LOADED and legacy emits behave as before.
        if kwargs.get("source") == "query_save":
            return
        last_result["value"] = None
        _render()

    state.on(StateEvent.PIPELINE_RESULT, _on_pipeline_result)
    state.on(StateEvent.VERSION_CHANGED, _clear)
    state.on(StateEvent.PROJECT_LOADED, _clear)

    # Round-12 R12-09: detach listeners on client disconnect to avoid
    # leaking callbacks that mutate destroyed UI elements (matches
    # ``tasks_indicator``).
    try:
        client = ui.context.client
    except Exception:
        client = None
    if client is not None:
        def _cleanup():
            state.off(StateEvent.PIPELINE_RESULT, _on_pipeline_result)
            state.off(StateEvent.VERSION_CHANGED, _clear)
            state.off(StateEvent.PROJECT_LOADED, _clear)
        client.on_disconnect(_cleanup)

    _render()


def _render_card(
    state: AppState,
    result: NLQueryResult,
    last_result: dict[str, NLQueryResult | None],
    rerender,
) -> None:
    """Render a single Q&A response card inside ``container``."""
    pr = result.pipeline_result
    if pr is None:
        return

    card = structured_response_data(pr.physics_result, ws=state.world_state)
    qtype = pr.query_type or "general"

    with ui.row().classes(
        "w-full items-start gap-3 px-4 py-3"
    ):
        ui.icon(
            "psychology" if qtype == "interrogate" else "chat",
            color="primary",
        ).classes("text-2xl")
        with ui.column().classes("flex-grow gap-1"):
            with ui.row().classes("items-center gap-2"):
                ui.label(
                    "Interrogation" if qtype == "interrogate" else "Ask"
                ).classes("text-sm font-semibold text-slate-700")
                conf_pct = int((card.get("confidence") or 0.0) * 100)
                conf_color = (
                    "positive" if conf_pct >= 70
                    else "warning" if conf_pct >= 40
                    else "negative"
                )
                ui.badge(
                    f"{conf_pct}% confidence", color=conf_color,
                ).props("dense outline")
                # R19-UI: branch badge so a counterfactual answer is
                # visually distinct from a factual one. Reads the
                # active VWM head; falls back to no badge on factual
                # mainline to keep the default case uncluttered.
                _vwm = getattr(state, "versioned_model", None)
                if _vwm is not None and getattr(_vwm, "history", None):
                    _head = _vwm.history[-1]
                    _world_id = getattr(_head, "world_id", "factual")
                    _branch_label = getattr(_head, "branch_label", None)
                    if _world_id == "shadow":
                        _badge_text = (
                            f"shadow: {_branch_label}"
                            if _branch_label
                            else "shadow"
                        )
                        ui.badge(_badge_text, color="purple").props(
                            "dense outline"
                        ).tooltip(
                            "Answer derived from a shadow branch "
                            "(counterfactual / intervention fork); "
                            "factual mainline is unchanged."
                        )
                from shadow_loom_ui.components._pearl_chip import (
                    render_pearl_chip,
                )
                render_pearl_chip(qtype)
                from shadow_loom_ui.components.help_popover import (
                    help_popover,
                )
                help_popover(
                    title="Answer panel — read-only Q&A responses",
                    body_md=(
                        "Surfaces the latest **Interrogation** result."
                        " Interrogation is read-only: it queries the"
                        " world model without generating prose or"
                        " saving a version, so the Story tab and"
                        " version tree stay unchanged.\n\n"
                        "### Card layout\n"
                        "- **Mode badge** — *Interrogation*"
                        " (diagnostic causal Q&A).\n"
                        "- **Confidence** — how well the world state"
                        " supports the claim.\n"
                        "  - 🟢 **Green ≥ 70%** — directly stated in"
                        " the graph.\n"
                        "  - 🟡 **Amber 40–69%** — inferred from"
                        " multiple nodes.\n"
                        "  - 🔴 **Red < 40%** — speculative or"
                        " unsupported.\n"
                        "- **Claim** — the engine's plain-language"
                        " answer.\n"
                        "- **Evidence** (expandable) — the world-state"
                        " node ids the LLM consulted (`ENT_*`,"
                        " `EVT_*`, `OBJ_*`, `LOC_*`, `CHN_*`). Cross"
                        "-reference these in Explorer / World tabs to"
                        " verify.\n"
                        "- **Caveats** — missing information,"
                        " ambiguity, or inferences that go beyond what"
                        " the graph explicitly records.\n\n"
                        "### Lifecycle\n"
                        "- New Ask/Interrogation runs **replace** the"
                        " current card.\n"
                        "- Switching versions or projects **clears**"
                        " the panel — an old answer no longer matches"
                        " the visible world state, so it would mislead.\n"
                        "- Click ✕ to dismiss manually."
                    ),
                    tooltip="What does this panel show?",
                )
                ui.space()
                from shadow_loom_ui.components._copy_button import (
                    copy_button,
                )
                copy_button(
                    card.get("claim") or "",
                    tooltip="Copy answer to clipboard",
                )
                def _do_dismiss():
                    last_result["value"] = None
                    rerender()
                ui.button(
                    icon="close",
                    on_click=_do_dismiss,
                ).props(
                    'flat dense round size=sm '
                    'aria-label="Dismiss answer"'
                ).tooltip("Dismiss")

            claim = card.get("claim") or "(no answer returned)"
            safe_markdown(claim).classes(
                "text-sm text-slate-800 leading-snug"
            )

            evidence = card.get("evidence") or []
            if evidence:
                with ui.expansion(
                    f"Evidence ({len(evidence)})", icon="fact_check",
                ).props("dense").classes("w-full"):
                    with ui.column().classes("gap-1 p-2"):
                        for ev in evidence[:20]:
                            label_txt = ev.get("label") or ev.get("node_id", "?")
                            ui.label(
                                f"• {label_txt}  ({ev.get('node_id', '?')})"
                            ).classes("text-xs text-slate-600 font-mono")

            caveats = card.get("caveats") or []
            if caveats:
                with ui.column().classes("gap-1"):
                    for cav in caveats:
                        with ui.row().classes("items-start gap-1"):
                            ui.icon("info", color="warning").classes(
                                "text-sm mt-0.5"
                            )
                            ui.label(cav).classes(
                                "text-xs text-slate-600"
                            )
