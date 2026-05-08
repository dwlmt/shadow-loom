# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Affective tab — emotional gauges, tension, and affective time-series.

Promoted out of the Causality tab in May 2026 because the affective
calculus is conceptually independent of the causal-graph workbench
and deserves its own top-level surface.

The dashboard implementation itself still lives in
:mod:`shadow_loom_ui.components.causality_tab` (as
:func:`build_affective_dashboard`); this module is the thin top-level
shell that hosts it.
"""

from __future__ import annotations

from nicegui import ui

from shadow_loom_ui.state import AppState
from shadow_loom_ui.components._subtab_help import subtab_help
from shadow_loom_ui.components.causality_tab import build_affective_dashboard


def build_affective_tab(state: AppState) -> None:
    """Render the top-level Affective tab."""

    with ui.column().classes("w-full h-full bg-slate-50"):
        with ui.row().classes(
            "w-full items-center px-4 py-2 border-b border-slate-200 "
            "gap-2 bg-white"
        ):
            ui.icon("favorite", color="primary")
            ui.label("Affective Dashboard").classes(
                "text-sm font-semibold text-slate-700"
            )
            ui.space()
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Affective \u2014 narrative-affect calculus over the syuzhet",
                body_md=(
                    "Heatmaps and gauges of *suspense*, *surprise*,"
                    " *dramatic irony*, *mystery*, and per-emotion"
                    " scores (love, regret, rage, fear, joy, grief)"
                    " evaluated by the engine over the reading order"
                    " (syuzhet).\n\n"
                    "### What you see\n"
                    "- **Gauges** \u2014 at-a-glance qualitative state at"
                    " the active cursor (live or scrubbed).\n"
                    "- **Affective Metrics over time** \u2014 multi-line"
                    " time-series, one line per metric, with the"
                    " active cursor as a needle.\n"
                    "- **Event Timeline** \u2014 every scored event pinned"
                    " to its syuzhet (or fabula) position so you can"
                    " correlate spikes with what just happened.\n"
                    "- **Raw Data** \u2014 expandable tables of scored"
                    " events and per-metric scores.\n\n"
                    "### How to read it\n"
                    "- Spikes mark dramatic peaks; long sags often"
                    " indicate pacing problems the audit will flag.\n"
                    "- Toggle **Fabula \u2194 Syuzhet** to see how reorder"
                    " (flashbacks, in-medias-res) reshapes the"
                    " trajectory.\n"
                    "- Toggle **Normalize** to compare metric *shapes*"
                    " when their absolute magnitudes differ widely.\n\n"
                    "### Provenance\n"
                    "All scores come from the same"
                    " ``DirectiveAssembler`` /"
                    " ``affective_timeseries_syuzhet`` pipeline used by"
                    " the engine when ranking candidate interventions"
                    " in the Causality tab \u2014 nothing here is"
                    " LLM-generated."
                ),
                tooltip="What is this tab?",
            )

        with ui.column().classes("w-full h-full p-4 gap-4 overflow-auto"):
            subtab_help("affective")
            build_affective_dashboard(state)


