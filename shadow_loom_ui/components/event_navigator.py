# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Event navigator — narrative-order (syuzhet) event browser.

Left rail: every event in syuzhet_index order, clickable.
Right pane: a *dossier* for the selected event combining

  * Event details (description, type, fabula vs syuzhet, neighbours).
  * Actors and targets, with per-character traits + beliefs
    reconstructed at this moment in fabula time.
  * Incoming/outgoing causal edges (mechanism, force, evidence).
  * Locations involved.
  * Objects in scope.
  * World traits active at this fabula tick.
  * Trait-radar charts for each actor.
  * Cross-link buttons to Explorer, Causality, and the Reasoning
    sub-tabs (Why this? / Belief lens).

Selecting an event also emits :data:`StateEvent.NODE_SELECTED` so the
Explorer's inspector and any other subscribers stay in sync.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from nicegui import ui

from shadow_loom_ui.reasoning_helpers import (
    event_context_data,
    syuzhet_event_index,
)
from shadow_loom_ui.reasoning_viz import render_attribution_graph
from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import render_trait_radar

logger = logging.getLogger(__name__)


# Map from event_type to a Material icon + colour so the rail reads
# like a story-beat list rather than a flat sequence of rows.
_EVENT_TYPE_STYLE: Dict[str, tuple[str, str]] = {
    "choice": ("alt_route", "primary"),
    "outcome": ("flag", "warning"),
    "revelation": ("auto_awesome", "secondary"),
}


def build_event_navigator(state: AppState) -> None:
    """Build the Events sub-tab inside the Reasoning tab."""
    if state.world_state is None or not state.world_state.events:
        ui.label("No events in the current world model.").classes(
            "text-sm text-slate-400 italic"
        )
        return

    selected_event = {"id": ""}
    # User-toggled order: "syuzhet" (narrative-text order) vs
    # "fabula" (in-world chronological order). They differ for any
    # story with flashbacks, framing devices, or in medias res.
    order_state = {"mode": "syuzhet"}

    with ui.splitter(value=28).classes("w-full").style(
        "height: calc(100vh - 280px); min-height: 480px"
    ) as split:
        # ── Left: ordered event list ──────────────────────────────
        with split.before:
            with ui.column().classes("w-full h-full bg-white border-r border-slate-200 overflow-auto"):
                with ui.row().classes("w-full items-center px-3 py-2 bg-slate-50 border-b border-slate-200"):
                    ui.icon("auto_stories", color="primary")
                    order_label = ui.label("Narrative order").classes(
                        "text-sm font-semibold text-slate-700"
                    )
                    ui.space()
                    order_caption = ui.label("syuzhet").classes(
                        "text-[10px] text-slate-400 uppercase"
                    )

                # Order toggle: syuzhet (text) vs fabula (chronology)
                order_toggle = ui.toggle(
                    {"syuzhet": "Narrative", "fabula": "Chronology"},
                    value="syuzhet",
                ).props("dense no-caps spread").classes("w-full px-2 q-mt-xs").tooltip(
                    "Narrative = syuzhet (order events appear in the text). "
                    "Chronology = fabula (order events happen in the world)."
                )

                # Search box
                search = ui.input(
                    placeholder="Filter events…",
                ).props("dense outlined clearable").classes("w-full px-2 q-mt-xs")

                list_container = ui.column().classes("w-full gap-0 q-mt-sm")

                def _refresh_list() -> None:
                    list_container.clear()
                    rows = syuzhet_event_index(state.world_state) if state.world_state else []
                    # Re-sort if the user picked fabula chronology.
                    # syuzhet_event_index already returns syuzhet-sorted
                    # rows, so we only need to overwrite the order for
                    # the fabula case.
                    if order_state["mode"] == "fabula":
                        rows = sorted(
                            rows,
                            key=lambda r: (r["fabula_time"], r["syuzhet_index"]),
                        )
                    # Header label reflects current mode
                    if order_state["mode"] == "fabula":
                        order_label.text = "Chronological order"
                        order_caption.text = "fabula"
                    else:
                        order_label.text = "Narrative order"
                        order_caption.text = "syuzhet"

                    ft = (search.value or "").strip().lower()
                    if ft:
                        rows = [
                            r for r in rows
                            if ft in r["description"].lower()
                            or ft in r["id"].lower()
                            or ft in r["event_type"].lower()
                        ]

                    if not rows:
                        with list_container:
                            ui.label("No matching events.").classes(
                                "text-xs text-slate-400 italic px-3"
                            )
                        return

                    with list_container:
                        for r in rows:
                            icon, color = _EVENT_TYPE_STYLE.get(
                                r["event_type"], ("circle", "grey"),
                            )
                            is_selected = (selected_event["id"] == r["id"])
                            row_classes = (
                                "w-full items-start gap-2 px-3 py-2 cursor-pointer "
                                "border-l-2 "
                                + ("border-primary bg-primary/5" if is_selected
                                   else "border-transparent hover:bg-slate-50")
                            )
                            # Whichever ordering is active gets the
                            # prominent leading marker; the other is
                            # shown smaller alongside so the user can
                            # always see the relationship between
                            # narrative position and fabula tick.
                            if order_state["mode"] == "fabula":
                                primary = f"t{r['fabula_time']}"
                                secondary = f"#{r['syuzhet_index']}"
                            else:
                                primary = f"#{r['syuzhet_index']}"
                                secondary = f"t{r['fabula_time']}"
                            with ui.row().classes(row_classes).on(
                                "click",
                                lambda _e, eid=r["id"]: _select(eid),
                            ):
                                ui.icon(icon, color=color, size="sm")
                                with ui.column().classes("gap-0 flex-grow min-w-0"):
                                    with ui.row().classes("items-baseline gap-2"):
                                        ui.label(primary).classes(
                                            "text-[11px] text-slate-600 font-mono font-semibold"
                                        )
                                        ui.label(r["event_type"]).classes(
                                            "text-[10px] text-slate-500 uppercase"
                                        )
                                        ui.label(secondary).classes(
                                            "text-[10px] text-slate-400 font-mono"
                                        )
                                    ui.label(r["description"][:80] or r["id"]).classes(
                                        "text-xs text-slate-700 leading-tight"
                                    ).style("overflow-wrap:anywhere")

                def _on_order_change() -> None:
                    order_state["mode"] = order_toggle.value or "syuzhet"
                    _refresh_list()

                order_toggle.on("update:model-value", lambda _e: _on_order_change())
                search.on("update:model-value", lambda _e: _refresh_list())

        # ── Right: detail dossier ─────────────────────────────────
        with split.after:
            detail = ui.column().classes("w-full h-full overflow-auto bg-slate-50 p-4")

            def _select(event_id: str) -> None:
                selected_event["id"] = event_id
                _refresh_list()
                _refresh_detail()
                # Cross-link with the rest of the UI
                if state.world_state and any(
                    e.id == event_id for e in state.world_state.events
                ):
                    state.select_node(event_id, "EventNode")

            def _refresh_detail() -> None:
                detail.clear()
                ws = state.world_state
                if ws is None or not selected_event["id"]:
                    with detail:
                        ui.label(
                            "Select an event from the left rail to see "
                            "its full dossier."
                        ).classes("text-sm text-slate-400 italic")
                    return
                ctx = event_context_data(ws, selected_event["id"])
                if not ctx:
                    with detail:
                        ui.label("Event not found.").classes(
                            "text-sm text-slate-500"
                        )
                    return
                with detail:
                    _render_event_dossier(state, ctx)

            # Initial selection: first event
            initial_rows = syuzhet_event_index(state.world_state)
            if initial_rows:
                _select(initial_rows[0]["id"])

            # Keep in sync if the world model changes (e.g. new version loaded)
            state.on(StateEvent.WORLD_STATE_CHANGED, lambda **_kw: (
                _refresh_list(), _refresh_detail(),
            ))
            # Honour cross-component selections (e.g. clicking a node
            # in the Explorer or causality graph).
            def _on_external_select(**kw):
                nid = kw.get("node_id")
                if not nid or nid == selected_event["id"]:
                    return
                if state.world_state and any(
                    e.id == nid for e in state.world_state.events
                ):
                    _select(nid)
            state.on(StateEvent.NODE_SELECTED, _on_external_select)

            _refresh_list()


# =====================================================================
# Detail dossier renderer
# =====================================================================

def _render_event_dossier(state: AppState, ctx: Dict[str, Any]) -> None:
    evt = ctx["event"]
    icon, color = _EVENT_TYPE_STYLE.get(evt["event_type"], ("circle", "grey"))

    # ── Header card with cross-link buttons ───────────────────────
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.icon(icon, color=color, size="md")
            ui.label(f"#{evt['syuzhet_index']}").classes(
                "text-xs text-slate-400 font-mono"
            )
            ui.badge(evt["event_type"], color=color).props("dense")
            ui.label(f"fabula t={evt['fabula_time']}").classes(
                "text-xs text-slate-500"
            )
            ui.space()
            ui.label(evt["id"]).classes(
                "text-[10px] text-slate-400 font-mono"
            )

        ui.label(evt["description"] or "(no description)").classes(
            "text-base text-slate-800 mt-2"
        )

        # Cross-links bar
        with ui.row().classes("items-center gap-2 mt-3 flex-wrap"):
            ui.button(
                "Open in Explorer",
                icon="travel_explore",
                on_click=lambda eid=evt["id"]: _jump_to(state, "explorer", eid, "EventNode"),
            ).props("dense outline color=primary size=sm no-caps")
            ui.button(
                "Why this? (attribution)",
                icon="device_hub",
                on_click=lambda eid=evt["id"]: _jump_to(state, "reasoning.attribution", eid, "EventNode"),
            ).props("dense outline color=primary size=sm no-caps")
            ui.button(
                "Causality view",
                icon="account_tree",
                on_click=lambda eid=evt["id"]: _jump_to(state, "causality", eid, "EventNode"),
            ).props("dense outline color=primary size=sm no-caps")
            ui.button(
                "Run counterfactual…",
                icon="alt_route",
                on_click=lambda eid=evt["id"]: state.emit(
                    StateEvent.QUERY_STARTED,
                    suggestion=f"What if {ctx['event']['description'][:80]} had not happened?",
                    query_type="counterfactual",
                ),
            ).props("dense outline color=primary size=sm no-caps")

        # Syuzhet neighbours
        nb = ctx.get("syuzhet_neighbours", {})
        with ui.row().classes("items-center gap-2 mt-3 text-xs text-slate-500"):
            if nb.get("prev"):
                p = nb["prev"]
                ui.button(
                    f"← #{p['syuzhet_index']} {p['description']}",
                    on_click=lambda pid=p["id"]: state.select_node(pid, "EventNode"),
                ).props("flat dense no-caps").classes("text-xs text-left")
            ui.space()
            if nb.get("next"):
                n = nb["next"]
                ui.button(
                    f"#{n['syuzhet_index']} {n['description']} →",
                    on_click=lambda nid=n["id"]: state.select_node(nid, "EventNode"),
                ).props("flat dense no-caps").classes("text-xs text-right")

    # ── Two-column layout for the rest ────────────────────────────
    with ui.row().classes("w-full gap-3 flex-wrap items-start q-mt-md"):
        with ui.column().classes("flex-grow gap-3 min-w-96"):
            _render_actors_panel(state, ctx)
            _render_targets_panel(state, ctx)
            _render_objects_panel(state, ctx)

        with ui.column().classes("flex-grow gap-3 min-w-96"):
            _render_causal_panel(state, ctx)
            _render_locations_panel(state, ctx)
            _render_world_traits_panel(state, ctx)
            _render_attribution_panel(state, ctx)


def _jump_to(
    state: AppState, path: str, node_id: str, node_type: str,
) -> None:
    """Select a node and switch tabs.

    Tab-switch in NiceGUI is best done by setting the q-tabs value via
    JS — but the lightweight version here just emits a NODE_SELECTED
    plus an active-path hint. Components subscribed to NODE_SELECTED
    pick it up.
    """
    state.select_node(node_id, node_type)
    state.set_active_path(path)
    ui.notify(
        f"Selected {node_id} — switch to the '{path.split('.')[0]}' tab to view.",
        type="info",
    )


# =====================================================================
# Sub-panels
# =====================================================================

def _render_actors_panel(state: AppState, ctx: Dict[str, Any]) -> None:
    actors = ctx["actors"]
    if not actors:
        return
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("group", color="primary")
            ui.label(f"Actors ({len(actors)})").classes(
                "text-sm font-semibold text-slate-700"
            )
        ui.label(
            "Each actor's traits and beliefs as they were at this fabula tick."
        ).classes("text-[11px] text-slate-500")

        for actor in actors:
            with ui.expansion(
                f"{actor['name']} — {actor.get('status', '?')}",
                icon="person",
            ).props("dense").classes("w-full"):
                # Quick chips: id, location, status
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    ui.badge(actor["id"]).props("dense")
                    if actor.get("location_id"):
                        ui.badge(
                            f"@ {actor['location_id']}", color="secondary",
                        ).props("dense")
                    ui.button(
                        "Belief lens", icon="visibility",
                        on_click=lambda eid=actor["id"]: _jump_to(
                            state, "reasoning.belief", eid, "Entity",
                        ),
                    ).props("flat dense color=primary size=sm no-caps")
                    ui.button(
                        "Inspect", icon="travel_explore",
                        on_click=lambda eid=actor["id"]: _jump_to(
                            state, "explorer", eid, "Entity",
                        ),
                    ).props("flat dense color=primary size=sm no-caps")

                # Trait radar
                if actor.get("traits") and state.world_state is not None:
                    render_trait_radar(actor["id"], state.world_state, height="240px")

                # Beliefs as a small table
                beliefs = actor.get("beliefs") or []
                if beliefs:
                    rows = [
                        {
                            "target": b.get("target_id", ""),
                            "perceived": b.get("perceived_state", ""),
                            "conf": f"{float(b.get('confidence', 0)):.2f}",
                            "inert": f"{float(b.get('inertia', 0)):.2f}",
                            "since": b.get("established_at_fabula", 0),
                        }
                        for b in beliefs
                    ]
                    ui.table(
                        columns=[
                            {"name": "target", "label": "About", "field": "target"},
                            {"name": "perceived", "label": "Believes", "field": "perceived"},
                            {"name": "conf", "label": "Conf.", "field": "conf"},
                            {"name": "inert", "label": "Inertia", "field": "inert"},
                            {"name": "since", "label": "Since", "field": "since"},
                        ],
                        rows=rows,
                        pagination={"rowsPerPage": 8},
                    ).props("dense flat bordered").classes("w-full")
                else:
                    ui.label("No beliefs recorded for this character at this moment.").classes(
                        "text-xs text-slate-400 italic"
                    )


def _render_targets_panel(state: AppState, ctx: Dict[str, Any]) -> None:
    targets = ctx["targets"]
    if not targets:
        return
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("crisis_alert", color="warning")
            ui.label(f"Targets ({len(targets)})").classes(
                "text-sm font-semibold text-slate-700"
            )
        for t in targets:
            kind = t.get("_kind", "?")
            with ui.row().classes("items-baseline gap-2 q-mt-xs"):
                ui.badge(kind, color="grey").props("dense outline")
                ui.label(t.get("name", t.get("id", ""))).classes(
                    "text-sm text-slate-700"
                )
                ui.label(t.get("id", "")).classes(
                    "text-[10px] text-slate-400 font-mono"
                )
                ui.button(
                    "Inspect", icon="travel_explore",
                    on_click=lambda eid=t["id"], k=kind: _jump_to(
                        state, "explorer", eid, k,
                    ),
                ).props("flat dense color=primary size=sm no-caps")


def _render_objects_panel(state: AppState, ctx: Dict[str, Any]) -> None:
    objects = ctx["objects"]
    if not objects:
        return
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("category", color="secondary")
            ui.label(f"Objects in scope ({len(objects)})").classes(
                "text-sm font-semibold text-slate-700"
            )
        rows = [
            {
                "name": o["name"],
                "id": o["id"],
                "owner": o.get("owner_id") or "—",
                "loc": o.get("location_id") or "—",
            }
            for o in objects
        ]
        ui.table(
            columns=[
                {"name": "name", "label": "Object", "field": "name"},
                {"name": "id", "label": "ID", "field": "id"},
                {"name": "owner", "label": "Owner", "field": "owner"},
                {"name": "loc", "label": "Location", "field": "loc"},
            ],
            rows=rows,
            pagination={"rowsPerPage": 8},
            on_select=lambda e: state.select_node(
                e.selection[0]["id"], "NarrativeObject"
            ) if e.selection else None,
            selection="single",
        ).props("dense flat bordered").classes("w-full")


def _render_causal_panel(state: AppState, ctx: Dict[str, Any]) -> None:
    incoming = ctx["incoming"]
    outgoing = ctx["outgoing"]
    if not incoming and not outgoing:
        return
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("account_tree", color="primary")
            ui.label("Causal relationships").classes(
                "text-sm font-semibold text-slate-700"
            )
        ui.label(
            "Direct causes feeding this event and effects it propagates."
        ).classes("text-[11px] text-slate-500")

        if incoming:
            with ui.expansion(
                f"Incoming ({len(incoming)})", icon="south_east",
                value=True,
            ).props("dense").classes("w-full"):
                _render_edge_table(state, incoming, side="incoming")
        if outgoing:
            with ui.expansion(
                f"Outgoing ({len(outgoing)})", icon="north_east",
                value=True,
            ).props("dense").classes("w-full"):
                _render_edge_table(state, outgoing, side="outgoing")


def _render_edge_table(
    state: AppState, edges: list, *, side: str,
) -> None:
    other_key = "source_id" if side == "incoming" else "target_id"
    other_label = "From" if side == "incoming" else "To"
    rows = [
        {
            "other_id": e[other_key],
            "other_label": e[f"{other_key.split('_')[0]}_label"],
            "type": e["type"],
            "mech": e["mechanism"],
            "force": f"{e['force']:.2f}",
            "evidence": e["evidence"],
            "delay": e["delay"],
            "trait": (e.get("trait_target") or "—") + (
                f" {e['trait_delta']:+.2f}" if e.get("trait_delta") is not None else ""
            ),
        }
        for e in edges
    ]
    ui.table(
        columns=[
            {"name": "other_label", "label": other_label, "field": "other_label"},
            {"name": "type", "label": "Type", "field": "type"},
            {"name": "mech", "label": "Mechanism", "field": "mech"},
            {"name": "force", "label": "Force", "field": "force"},
            {"name": "evidence", "label": "Evidence", "field": "evidence"},
            {"name": "delay", "label": "Delay", "field": "delay"},
            {"name": "trait", "label": "Trait Δ", "field": "trait"},
        ],
        rows=rows,
        pagination={"rowsPerPage": 10},
        on_select=lambda e: state.select_node(
            e.selection[0]["other_id"], None,
        ) if e.selection else None,
        selection="single",
    ).props("dense flat bordered").classes("w-full")


def _render_locations_panel(state: AppState, ctx: Dict[str, Any]) -> None:
    locations = ctx["locations"]
    if not locations:
        return
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("place", color="secondary")
            ui.label(f"Locations ({len(locations)})").classes(
                "text-sm font-semibold text-slate-700"
            )
        for loc in locations:
            with ui.row().classes("items-baseline gap-2"):
                ui.button(
                    loc["name"],
                    on_click=lambda lid=loc["id"]: _jump_to(
                        state, "explorer", lid, "Location",
                    ),
                ).props("flat dense no-caps color=primary size=sm")
                ui.label((loc.get("description") or "")[:120]).classes(
                    "text-xs text-slate-500"
                )


def _render_world_traits_panel(state: AppState, ctx: Dict[str, Any]) -> None:
    traits = ctx["world_traits_active"]
    if not traits:
        return
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("public", color="primary")
            ui.label(f"Global factors ({len(traits)})").classes(
                "text-sm font-semibold text-slate-700"
            )
        ui.label(
            "World-level constraints (governance, magic, environment, etc) "
            "and their magnitude at this moment."
        ).classes("text-[11px] text-slate-500")

        rows = [
            {
                "id": t["id"],
                "name": t["name"],
                "category": t["category"],
                "magnitude": f"{t['magnitude']['value']:.2f}",
                "inertia": f"{t['magnitude']['inertia']:.2f}",
                "description": (t.get("description") or "")[:80],
            }
            for t in traits
        ]
        ui.table(
            columns=[
                {"name": "name", "label": "Trait", "field": "name"},
                {"name": "category", "label": "Category", "field": "category"},
                {"name": "magnitude", "label": "Magnitude", "field": "magnitude"},
                {"name": "inertia", "label": "Inertia", "field": "inertia"},
                {"name": "description", "label": "Note", "field": "description"},
            ],
            rows=rows,
            pagination={"rowsPerPage": 8},
            on_select=lambda e: state.select_node(
                e.selection[0]["id"], "WorldTrait",
            ) if e.selection else None,
            selection="single",
        ).props("dense flat bordered").classes("w-full")


def _render_attribution_panel(state: AppState, ctx: Dict[str, Any]) -> None:
    """Mini attribution graph for this event."""
    if not ctx["incoming"]:
        return
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("device_hub", color="primary")
            ui.label("Why did this happen?").classes(
                "text-sm font-semibold text-slate-700"
            )
        ui.label(
            "Reverse-walk of the causal graph from this event."
        ).classes("text-[11px] text-slate-500")
        if state.world_state is not None:
            render_attribution_graph(
                state.world_state, ctx["event"]["id"],
                max_depth=3, height="280px",
            )
