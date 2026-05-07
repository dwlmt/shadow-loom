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

from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.components._subtab_help import subtab_help
from shadow_loom_ui.components.causality_tab import build_affective_dashboard
from shadow_loom.models import reconstruct_proposition_at, reconstruct_concern_at


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
            with ui.tabs().classes("w-full") as _affective_tabs:
                _t_dash = ui.tab("dashboard", label="Dashboard", icon="dashboard")
                _t_props = ui.tab("propositions", label="Propositions", icon="forum")
                _t_concerns = ui.tab("concerns", label="Concerns", icon="psychology")
            with ui.tab_panels(_affective_tabs, value=_t_dash).classes("w-full"):
                with ui.tab_panel(_t_dash):
                    build_affective_dashboard(state)
                with ui.tab_panel(_t_props):
                    _build_propositions_panel(state)
                with ui.tab_panel(_t_concerns):
                    _build_concerns_panel(state)


# =====================================================================
# Propositions panel
# =====================================================================

def _max_fabula_time(state: AppState) -> int:
    ws = state.world_state
    if ws is None or not ws.events:
        return 0
    return max((e.fabula_time for e in ws.events), default=0)


def _build_propositions_panel(state: AppState) -> None:
    """Per-proposition timeline browser.

    Lets the user scrub a fabula-time cursor and see each proposition's
    framing (``stakes``, ``audience_default_prior``, ``description``) and
    committed truth value at that moment, via
    :func:`shadow_loom.models.reconstruct_proposition_at`.
    """
    cursor = {"t": _max_fabula_time(state)}

    with ui.row().classes("w-full items-center gap-3"):
        ui.icon("forum", color="primary")
        ui.label("Proposition timeline").classes(
            "text-sm font-semibold text-slate-700"
        )
        ui.space()
        cursor_label = ui.label(f"fabula t={cursor['t']}").classes(
            "text-xs text-slate-500 font-mono"
        )

    max_t_state = {"v": _max_fabula_time(state)}
    slider = ui.slider(
        min=0, max=max(max_t_state["v"], 1), value=cursor["t"], step=1,
    ).props("label-always").classes("w-full")

    table = ui.table(
        columns=[
            {"name": "id", "label": "ID", "field": "id", "sortable": True},
            {"name": "kind", "label": "Kind", "field": "kind", "sortable": True},
            {"name": "description", "label": "Description@t", "field": "description"},
            {"name": "truth_at", "label": "Truth@t", "field": "truth_at", "sortable": True},
            {"name": "stakes", "label": "Stakes@t", "field": "stakes", "sortable": True},
            {"name": "prior", "label": "Aud. prior@t", "field": "prior", "sortable": True},
            {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
        ],
        rows=[],
        pagination={"rowsPerPage": 15},
    ).classes("w-full")

    def _refresh(**_: object) -> None:
        ws = state.world_state
        if ws is None:
            table.rows = []
            return
        new_max = _max_fabula_time(state)
        if new_max != max_t_state["v"]:
            max_t_state["v"] = new_max
            slider.props(f"max={max(new_max, 1)}")
            if cursor["t"] > new_max:
                cursor["t"] = new_max
                slider.value = new_max
        rows = []
        for prop in ws.propositions:
            snap = reconstruct_proposition_at(prop, cursor["t"])
            rows.append({
                "id": prop.proposition_id,
                "kind": prop.kind,
                "description": snap["description"],
                "truth_at": (
                    "true" if snap["truth_at"] is True
                    else "false" if snap["truth_at"] is False
                    else "—"
                ),
                "stakes": round(float(snap["stakes"]), 3),
                "prior": round(float(snap["audience_default_prior"]), 3),
                "world_id": prop.world_id,
            })
        table.rows = rows
        cursor_label.text = f"fabula t={cursor['t']}"

    def _on_slider(e: object) -> None:
        cursor["t"] = int(getattr(e, "value", cursor["t"]) or 0)
        _refresh()

    slider.on("update:model-value", _on_slider)
    _refresh()
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)


# =====================================================================
# Concerns panel
# =====================================================================

def _build_concerns_panel(state: AppState) -> None:
    """Per-entity concern browser with fabula-time replay.

    Lists every (entity, concern) pair, showing the concern's
    fabula-time-aware state (``salience``, ``polarity``,
    ``activation_fabula_window``, ``kind``, ``active``) via
    :func:`shadow_loom.models.reconstruct_concern_at`.
    """
    cursor = {"t": _max_fabula_time(state)}

    with ui.row().classes("w-full items-center gap-3"):
        ui.icon("psychology", color="primary")
        ui.label("Concern timeline").classes(
            "text-sm font-semibold text-slate-700"
        )
        ui.space()
        only_active = ui.switch(
            "Only active@t", value=False,
        ).props("dense").classes("text-xs")
        cursor_label = ui.label(f"fabula t={cursor['t']}").classes(
            "text-xs text-slate-500 font-mono"
        )

    max_t_state = {"v": _max_fabula_time(state)}
    slider = ui.slider(
        min=0, max=max(max_t_state["v"], 1), value=cursor["t"], step=1,
    ).props("label-always").classes("w-full")

    table = ui.table(
        columns=[
            {"name": "entity", "label": "Entity", "field": "entity", "sortable": True},
            {"name": "concern_id", "label": "Concern", "field": "concern_id", "sortable": True},
            {"name": "proposition_id", "label": "Proposition", "field": "proposition_id"},
            {"name": "polarity", "label": "Polarity@t", "field": "polarity", "sortable": True},
            {"name": "salience", "label": "Salience@t", "field": "salience", "sortable": True},
            {"name": "kind", "label": "Kind@t", "field": "kind"},
            {"name": "active", "label": "Active@t", "field": "active", "sortable": True},
            {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
        ],
        rows=[],
        pagination={"rowsPerPage": 20},
    ).classes("w-full")

    def _refresh(**_: object) -> None:
        ws = state.world_state
        if ws is None:
            table.rows = []
            return
        new_max = _max_fabula_time(state)
        if new_max != max_t_state["v"]:
            max_t_state["v"] = new_max
            slider.props(f"max={max(new_max, 1)}")
            if cursor["t"] > new_max:
                cursor["t"] = new_max
                slider.value = new_max
        rows = []
        for ent in ws.entities.values():
            for concern in ent.concerns:
                snap = reconstruct_concern_at(concern, cursor["t"])
                if only_active.value and not snap["active"]:
                    continue
                rows.append({
                    "entity": ent.name,
                    "concern_id": concern.concern_id,
                    "proposition_id": concern.proposition_id,
                    "polarity": snap["polarity"],
                    "salience": round(float(snap["salience"]), 3),
                    "kind": snap.get("kind") or "—",
                    "active": "✓" if snap["active"] else "—",
                    "world_id": concern.world_id,
                })
        table.rows = rows
        cursor_label.text = f"fabula t={cursor['t']}"

    def _on_slider(e: object) -> None:
        cursor["t"] = int(getattr(e, "value", cursor["t"]) or 0)
        _refresh()

    slider.on("update:model-value", _on_slider)
    only_active.on("update:model-value", lambda _e: _refresh())
    _refresh()
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)
