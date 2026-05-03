# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Causality tab — causal topology, what-if workbench, directive builder, affective dashboard.

Four sub-views:
  1. Causal Topology — Sankey + causal force graph
  2. What-If Workbench — NL-driven intervention/counterfactual with propagation waterfall
  3. Directive Builder — visual emotional target builder + NL hybrid
  4. Affective Dashboard — emotional gauges, tension, trait trajectories
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent
from shadow_loom_ui.task_helpers import run_query_as_task
from shadow_loom_ui.viz import (
    render_affective_timeseries,
    render_causal_force_graph,
    render_causal_sankey,
    render_chart_skeleton,
    render_emotional_gauges,
    render_emotional_gauges_graded,
    render_empty_state,
    render_entity_state_timeline,
    render_event_timeline,
    render_physics_trajectory,
    render_propagation_graph,
    render_propagation_waterfall,
    render_trait_radar_compare,
    render_relationship_timeline,
    render_world_trait_timeline,
    open_explain_dialog,
    with_expand,
)
from shadow_loom_ui.viz_helpers import (
    mutations_to_propagation_rows,
    entity_radar_compare_rows,
)

import json as _json


# =====================================================================
# URL-sync helpers
#
# We mutate ``window.location`` via ``history.replaceState`` so links
# to a specific cursor / aspect are bookmarkable, but the back button
# is not polluted with every slider tick.
# =====================================================================

def _url_set_param(key: str, value: "str | int | None") -> None:
    """Set/remove a single query-string param without reloading the page.

    Safe to call from event handlers; uses ``json.dumps`` to escape
    both keys and values so user-controlled aspect strings cannot
    inject JavaScript.
    """
    k = _json.dumps(str(key))
    if value is None or value == "":
        ui.run_javascript(
            "(()=>{const u=new URL(window.location);"
            f"u.searchParams.delete({k});"
            "window.history.replaceState({},'',u);})();"
        )
        return
    v = _json.dumps(str(value))
    ui.run_javascript(
        "(()=>{const u=new URL(window.location);"
        f"u.searchParams.set({k},{v});"
        "window.history.replaceState({},'',u);})();"
    )


async def _url_get_params() -> dict:
    """Read ``window.location.search`` into a flat dict (str→str)."""
    try:
        raw = await ui.run_javascript(
            "JSON.stringify(Object.fromEntries("
            "new URLSearchParams(window.location.search)))"
        )
    except Exception:
        logger.debug("_url_get_params: JS read failed", exc_info=True)
        return {}
    try:
        return _json.loads(raw or "{}")
    except Exception:
        return {}

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_EMOTIONAL_TARGETS = [
    ("mystery", "Mystery"),
    ("dramatic_irony", "Dramatic Irony"),
    ("suspense", "Suspense"),
    ("surprise", "Surprise"),
    ("grief", "Grief"),
    ("rage", "Rage"),
    ("joy", "Joy"),
    ("fear", "Fear"),
    ("love", "Love"),
    ("regret", "Regret"),
]


def build_causality_tab(state: AppState) -> None:
    """Build the Causality tab with four sub-views."""

    with ui.column().classes("w-full h-full bg-slate-50"):
        with ui.tabs().props(
            "dense no-caps indicator-color=primary active-color=primary align=left"
        ).classes(
            "w-full bg-white border-b border-slate-200 px-4"
        ) as sub_tabs:
            ui.tab("topology", label="Causal Topology", icon="device_hub")
            ui.tab("evolution", label="Evolution", icon="timeline")
            ui.tab("whatif", label="What-If Workbench", icon="science")
            ui.tab("directive", label="Directive Builder", icon="theater_comedy")
            ui.tab("affective", label="Affective Dashboard", icon="favorite")

        # Track the active sub-tab so each panel can gate its refresh.
        # Default-load is "topology"; mirror that into AppState so panels
        # below first-paint without waiting for a tab change.
        if state.active_path in ("", "causality"):
            state.set_active_path("causality.topology")

        def _on_sub_tab(e):
            val = getattr(e, "args", None)
            if isinstance(val, str):
                state.set_active_path(f"causality.{val}")

        sub_tabs.on("update:model-value", _on_sub_tab)

        with ui.tab_panels(sub_tabs, value="topology").classes(
            "w-full flex-grow bg-slate-50"
        ):
            with ui.tab_panel("topology").classes("p-4"):
                _build_causal_topology(state)

            with ui.tab_panel("evolution").classes("p-4"):
                _build_evolution_panel(state)

            with ui.tab_panel("whatif").classes("p-4"):
                _build_whatif_workbench(state)

            with ui.tab_panel("directive").classes("p-4"):
                _build_directive_builder(state)

            with ui.tab_panel("affective").classes("p-4"):
                _build_affective_dashboard(state)


# =====================================================================
# 1. Causal Topology
# =====================================================================

def _build_causal_topology(state: AppState) -> None:
    """Sankey diagram + causal force graph with aspect/force/time filters."""

    from shadow_loom_ui.viz_helpers import (
        SANKEY_ASPECTS,
        _set_slider_bounds,
        fabula_time_bounds,
    )

    with ui.column().classes("w-full h-full gap-3"):
        # ── Top row: view & aspect ──────────────────────────────
        with ui.row().classes("w-full items-center gap-3 flex-wrap"):
            view_toggle = ui.toggle(
                {"sankey": "Sankey Flow", "force": "Force Graph"},
                value="sankey",
            ).props("dense no-caps color=primary")
            view_toggle.tooltip(
                "Sankey shows directed flow between events; "
                "Force lays nodes out by causal proximity"
            )

            ui.label("Aspect:").classes("text-sm text-slate-600 ml-4")
            aspect_select = ui.select(
                options={k: label for k, label in SANKEY_ASPECTS},
                value="causal_all",
            ).props("dense outlined options-dense").classes("min-w-56")
            aspect_select.tooltip(
                "Which slice of the causal/social graph to draw"
            )

            ui.label("Force layout:").classes("text-sm text-slate-600 ml-4")
            force_layout = ui.select(
                options={"force": "Force", "circular": "Circular", "cartesian": "Cartesian"},
                value="force",
            ).props("dense outlined options-dense").classes("min-w-32")
            force_layout.tooltip(
                "How to lay out the causal force graph "
                "(circular = ring; cartesian = time on x-axis)"
            )

        # ── Filter row (Sankey only) — sticky for tall diagrams ────
        filter_row = ui.row().classes(
            "w-full items-center gap-4 bg-white border border-slate-200 "
            "rounded-xl shadow-sm p-3 flex-wrap sticky"
        ).style("top: 0; z-index: 4;")
        with filter_row:
            ui.icon("filter_alt", color="primary")

            ui.label("Min force:").classes("text-sm text-slate-600")
            force_slider = ui.slider(min=0.0, max=10.0, step=0.5, value=0.0).props(
                "color=primary label-always dense"
            ).classes("w-48")
            force_slider.tooltip("Hide edges below this causal_force value")

            ui.label("Up to fabula t:").classes("text-sm text-slate-600 ml-4")
            time_slider = ui.slider(min=0, max=1, step=1, value=1).props(
                "color=secondary label-always dense"
            ).classes("flex-grow min-w-32")
            time_slider.tooltip("Only show edges with fabula_time ≤ this value")

            def _reset_time():
                # Drive through the global cursor so other tabs follow.
                _slider_state["local_origin_t"] = False
                state.set_fabula_cursor(None)

            ui.button("Full span", icon="restart_alt", on_click=_reset_time) \
                .props("flat dense no-caps color=secondary")

        graph_container = ui.column().classes(
            "w-full flex-grow bg-white border border-slate-200 "
            "rounded-xl shadow-sm p-4"
        )

        # Re-entrancy / coalescing guards. ``local_origin[key]`` is
        # set when this panel's slider initiated the change, so the
        # FABULA_CURSOR_CHANGED listener doesn't write the value back
        # at us (which used to freeze the slider mid-drag). ``rendering``
        # collapses overlapping renders into a single follow-up so a
        # rapid scrub doesn't queue up half-built Sankeys.
        _slider_state: dict = {
            "local_origin_t": False,
            "local_origin_f": False,
            "rendering": False,
            "pending": False,
        }

        def _sync_time_slider(ws) -> None:
            tmin, tmax = fabula_time_bounds(ws)
            if tmax <= tmin:
                return
            # Quasar's <q-slider> requires numeric min/max props. Passing
            # the props via the string parser (``time_slider.props(...)``)
            # stores them as *strings*, which the slider silently rejects
            # — the thumb appears to render but won't drag past the
            # original construction-time bounds. Write numeric values
            # straight into the props dict instead.
            _set_slider_bounds(time_slider, tmin, tmax)
            cur = state.fabula_cursor
            desired = tmax if cur is None else max(tmin, min(tmax, cur))
            if not _slider_state["local_origin_t"]:
                try:
                    cur_w = int(time_slider.value or 0)
                except (TypeError, ValueError):
                    cur_w = -1
                if cur_w != desired:
                    time_slider.value = desired
            _slider_state["local_origin_t"] = False

        def _refresh(**kw):
            if _slider_state["rendering"]:
                _slider_state["pending"] = True
                return
            _slider_state["rendering"] = True
            try:
                _do_refresh(**kw)
            finally:
                _slider_state["rendering"] = False
            if _slider_state["pending"]:
                _slider_state["pending"] = False
                ui.timer(0.01, lambda: _refresh(), once=True)

        def _do_refresh(**kw):
            graph_container.clear()
            ws = state.world_state
            if ws is None:
                filter_row.set_visibility(False)
                with graph_container:
                    # If a background task (typically ingestion) is
                    # populating the world, show a pulsing skeleton
                    # instead of the empty-state CTA so the user sees
                    # "loading" rather than "nothing here".
                    if state.running_task_count > 0:
                        render_chart_skeleton("360px")
                    else:
                        render_empty_state(
                            "No world model loaded.",
                            icon="hub",
                            hint="Open a project from the sidebar to see its causal topology.",
                        )
                return

            is_sankey = view_toggle.value == "sankey"
            filter_row.set_visibility(is_sankey)

            # Sync the time slider widget without bouncing user input back.
            _sync_time_slider(ws)
            tmin, tmax = fabula_time_bounds(ws)

            with graph_container:
                if is_sankey:
                    fmax = (
                        state.fabula_cursor
                        if state.fabula_cursor is not None
                        else tmax
                    )
                    aspect_label = dict(SANKEY_ASPECTS).get(
                        aspect_select.value, "Sankey"
                    )

                    def _on_sankey_click(ev):
                        # Wire cross-chart selection: store the clicked
                        # node id on AppState so other tabs can react.
                        try:
                            data = ev.args.get("data") or {}
                            nid = data.get("name") if isinstance(data, dict) else None
                            if nid:
                                # Route through the official setter so the
                                # inspector receives both the id and a
                                # resolved node_type. Sankey nodes are
                                # event-shaped in every aspect, so default
                                # to ``event`` and only widen if the id
                                # resolves to an entity/location/object/world_trait.
                                ws_now = state.world_state
                                node_type = "event"
                                if ws_now is not None:
                                    if nid in ws_now.entities:
                                        node_type = "entity"
                                    elif nid in ws_now.locations:
                                        node_type = "location"
                                    elif nid in ws_now.objects:
                                        node_type = "object"
                                    elif nid in getattr(ws_now, "world_traits", {}):
                                        node_type = "world_trait"
                                state.select_node(nid, node_type)
                                # Also pop the explain dialog so users
                                # see the causal structure of the click.
                                if ws_now is not None:
                                    open_explain_dialog(ws_now, nid)
                        except Exception:
                            logger.debug("sankey click: unparsable args", exc_info=True)

                    with_expand(
                        lambda h: render_causal_sankey(
                            ws,
                            height=h,
                            aspect=aspect_select.value,
                            min_force=float(force_slider.value or 0.0),
                            fabula_max=fmax if tmax > tmin else None,
                            on_click=_on_sankey_click,
                        ),
                        title=f"Sankey \u2014 {aspect_label}",
                    )
                else:
                    def _on_force_click(ev):
                        try:
                            data = ev.args.get("data") or {}
                            nid = data.get("name") if isinstance(data, dict) else None
                            if nid and state.world_state is not None:
                                open_explain_dialog(state.world_state, nid)
                        except Exception:
                            logger.debug("force click: unparsable args", exc_info=True)

                    with_expand(
                        lambda h: render_causal_force_graph(
                            ws,
                            height=h,
                            layout=force_layout.value or "force",
                            on_click=_on_force_click,
                        ),
                        title="Causal force graph",
                    )

        def _on_slider(key: str):
            if key == "t":
                # Push to global cursor; the FABULA_CURSOR_CHANGED
                # subscription below triggers _refresh(), so callers
                # see the same data as every other time-aware panel.
                try:
                    t = int(time_slider.value)
                except (TypeError, ValueError):
                    return
                if state.fabula_cursor == t:
                    return
                _slider_state["local_origin_t"] = True
                state.set_fabula_cursor(t)
                return
            _refresh()

        # Release-only sliders to avoid mid-drag freezes (see world_tab).
        view_toggle.on("update:model-value", lambda: _refresh())

        def _on_aspect():
            _url_set_param("aspect", aspect_select.value)
            _refresh()

        aspect_select.on("update:model-value", lambda: _on_aspect())
        force_layout.on("update:model-value", lambda: _refresh())
        force_slider.on("change", lambda: _on_slider("f"))
        time_slider.on("change", lambda: _on_slider("t"))

        # ── URL-sync hydration ─────────────────────────────────
        async def _hydrate_from_url():
            params = await _url_get_params()
            valid_aspects = {k for k, _ in SANKEY_ASPECTS}
            asp = params.get("aspect")
            changed = False
            if asp and asp in valid_aspects and asp != aspect_select.value:
                aspect_select.value = asp
                aspect_select.update()
                changed = True
            if changed:
                _refresh()

        ui.timer(0.1, lambda: state.spawn_task(_hydrate_from_url()), once=True)

        def _on_tasks(**kw):
            # Only re-render the empty/loading state; never rebuild a
            # populated chart on every task progress tick.
            if state.world_state is None:
                _refresh()

        # Gate by visibility so off-screen Topology doesn't rebuild
        # the Sankey/force graph on every cursor scrub from another tab.
        _TOPO_PATH = "causality.topology"
        _topo_dirty = {"on": True}
        _refresh_sync_topo = _refresh

        def _topo_gated(**kw):
            if not state.is_path_visible(_TOPO_PATH):
                _topo_dirty["on"] = True
                return
            _topo_dirty["on"] = False
            _refresh_sync_topo()

        _refresh = _topo_gated  # noqa: F811

        def _on_topo_path(**kw):
            if _topo_dirty["on"] and state.is_path_visible(_TOPO_PATH):
                _refresh()

        _refresh_sync_topo()
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)
        state.on(StateEvent.TASKS_CHANGED, _on_tasks)
        state.on(StateEvent.ACTIVE_PATH_CHANGED, _on_topo_path)
        # Cross-tab cursor sync: scrubbing the fabula cursor anywhere
        # (World tab, Causal Graph snapshot, Affective Dashboard) should
        # update the Sankey "Up to fabula t" filter so all panels share
        # one timeline.
        state.on(StateEvent.FABULA_CURSOR_CHANGED, lambda **_kw: _refresh())


# =====================================================================
# 1b. Evolution panel — character / relationship / world-trait timelines
# =====================================================================

def _build_evolution_panel(state: AppState) -> None:
    """How characters, relationships, and world traits change over the story."""

    from shadow_loom_ui.viz_helpers import (
        _set_slider_bounds,
        list_relationship_pairs,
        list_world_traits,
    )

    with ui.column().classes("w-full h-full gap-4"):
        with ui.tabs().props(
            "dense no-caps indicator-color=primary active-color=primary"
        ).classes("w-full bg-white border border-slate-200 rounded-xl") as evo_tabs:
            ui.tab("character", label="Character", icon="person")
            ui.tab("relationship", label="Relationship", icon="people")
            ui.tab("world", label="World Trait", icon="public")
            ui.tab("causal", label="Causal Graph", icon="account_tree")
            ui.tab("physics", label="Physics", icon="science")

        with ui.tab_panels(evo_tabs, value="character").classes("w-full flex-grow"):
            # ── Character ────────────────────────────────────────
            with ui.tab_panel("character").classes("p-3"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.icon("person", color="primary")
                    ui.label("Entity:").classes("text-sm text-slate-600")
                    char_select = ui.select(options={}, value=None).props(
                        "dense outlined options-dense"
                    ).classes("min-w-56")
                    char_select.tooltip(
                        "Pick an entity to see how its traits and status "
                        "evolve through fabula time"
                    )
                char_chart = ui.column().classes(
                    "w-full h-96 bg-white border border-slate-200 "
                    "rounded-xl shadow-sm p-3 mt-2"
                )

                def _refresh_char(**kw):
                    char_chart.clear()
                    ws = state.world_state
                    if ws is None or not ws.entities:
                        with char_chart:
                            render_empty_state(
                                "No entities loaded.",
                                icon="person_off",
                                hint="Load a project to see characters and their trait evolution.",
                            )
                        char_select.options = {}
                        char_select.value = None
                        char_select.update()
                        return
                    opts = {eid: ent.name for eid, ent in ws.entities.items()}
                    if char_select.options != opts:
                        char_select.options = opts
                        if char_select.value not in opts:
                            char_select.value = next(iter(opts))
                        char_select.update()
                    with char_chart:
                        eid = char_select.value
                        ent_name = ws.entities[eid].name if eid in ws.entities else eid
                        with_expand(
                            lambda h, eid=eid: render_entity_state_timeline(eid, ws, height=h),
                            title=f"Character evolution \u2014 {ent_name}",
                        )

                char_select.on("update:model-value", lambda: _refresh_char())

            # ── Relationship ─────────────────────────────────────
            with ui.tab_panel("relationship").classes("p-3"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.icon("people", color="primary")
                    ui.label("Pair:").classes("text-sm text-slate-600")
                    pair_select = ui.select(options={}, value=None).props(
                        "dense outlined options-dense"
                    ).classes("min-w-72")
                    pair_select.tooltip(
                        "Pick a relationship dyad to see affinity, fear, "
                        "and power-dynamic over time"
                    )
                rel_chart = ui.column().classes(
                    "w-full h-96 bg-white border border-slate-200 "
                    "rounded-xl shadow-sm p-3 mt-2"
                )

                def _refresh_rel(**kw):
                    rel_chart.clear()
                    ws = state.world_state
                    if ws is None:
                        with rel_chart:
                            ui.label("No world model loaded.").classes(
                                "text-sm text-slate-400 italic"
                            )
                        pair_select.options = {}
                        pair_select.value = None
                        pair_select.update()
                        return
                    pairs = list_relationship_pairs(ws)
                    if not pairs:
                        with rel_chart:
                            render_empty_state(
                                "No relationships found.",
                                icon="people_outline",
                                hint=(
                                    "Add `mutation_social` causal edges or "
                                    "RelationshipEdge entries to track how "
                                    "characters' bonds evolve."
                                ),
                            )
                        pair_select.options = {}
                        pair_select.value = None
                        pair_select.update()
                        return
                    opts = {f"{a}|{b}": f"{na} ⇄ {nb}" for a, b, na, nb in pairs}
                    if pair_select.options != opts:
                        pair_select.options = opts
                        if pair_select.value not in opts:
                            pair_select.value = next(iter(opts))
                        pair_select.update()
                    if not pair_select.value:
                        return
                    a, b = pair_select.value.split("|", 1)
                    na = ws.entities[a].name if a in ws.entities else a
                    nb = ws.entities[b].name if b in ws.entities else b
                    with rel_chart:
                        with_expand(
                            lambda h, a=a, b=b: render_relationship_timeline(ws, a, b, height=h),
                            title=f"Relationship \u2014 {na} \u21c4 {nb}",
                        )

                pair_select.on("update:model-value", lambda: _refresh_rel())

            # ── World trait ──────────────────────────────────────
            with ui.tab_panel("world").classes("p-3"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.icon("public", color="primary")
                    ui.label("Trait:").classes("text-sm text-slate-600")
                    world_select = ui.select(options={}, value=None).props(
                        "dense outlined options-dense"
                    ).classes("min-w-56")
                    world_select.tooltip(
                        "Pick a world-level trait (governance, magic, "
                        "environment\u2026) to see how it shifts over time"
                    )
                world_chart = ui.column().classes(
                    "w-full h-96 bg-white border border-slate-200 "
                    "rounded-xl shadow-sm p-3 mt-2"
                )

                def _refresh_world(**kw):
                    world_chart.clear()
                    ws = state.world_state
                    if ws is None or not ws.world_traits:
                        with world_chart:
                            render_empty_state(
                                "No world traits in this world.",
                                icon="public_off",
                                hint=(
                                    "World traits represent setting-level "
                                    "forces (e.g. surveillance state, magic "
                                    "system). Add `GlobalTrait` entries to "
                                    "see how they evolve."
                                ),
                            )
                        world_select.options = {}
                        world_select.value = None
                        world_select.update()
                        return
                    opts = {wid: name for wid, name in list_world_traits(ws)}
                    if world_select.options != opts:
                        world_select.options = opts
                        if world_select.value not in opts:
                            world_select.value = next(iter(opts))
                        world_select.update()
                    with world_chart:
                        wid = world_select.value
                        wname = ws.world_traits[wid].name if wid in ws.world_traits else wid
                        with_expand(
                            lambda h, wid=wid: render_world_trait_timeline(ws, wid, height=h),
                            title=f"World trait \u2014 {wname}",
                        )

                world_select.on("update:model-value", lambda: _refresh_world())

            # ── Causal graph snapshot ────────────────────────────
            with ui.tab_panel("causal").classes("p-3"):
                ui.label(
                    "Snapshot of the causal topology at a given fabula time. "
                    "Edges with fabula_time \u2264 t are included; scrub to "
                    "watch causality accrete."
                ).classes("text-xs text-slate-500 mb-2")

                causal_slider_row = ui.row().classes(
                    "w-full items-center gap-3 flex-wrap"
                )
                with causal_slider_row:
                    ui.icon("schedule", color="primary")
                    causal_time_label = ui.label("t=0").classes(
                        "text-sm font-mono text-slate-700 w-20"
                    )
                    causal_slider = ui.slider(
                        min=0, max=1, value=0, step=1
                    ).props("color=primary label-always dense").classes(
                        "flex-grow min-w-32"
                    )
                    causal_slider.tooltip(
                        "Filter causal edges to those with "
                        "fabula_time \u2264 t. Release-only to avoid lag. "
                        "Synced with the global fabula cursor."
                    )
                    causal_layout = ui.toggle(
                        {"force": "Force", "circular": "Circular",
                         "cartesian": "Cartesian"},
                        value="cartesian",
                    ).props("dense no-caps color=primary")
                    causal_show_deltas = ui.checkbox(
                        "Highlight \u0394", value=True,
                    ).tooltip(
                        "Pulse edges added at the current tick "
                        "(``fabula_time == t``) so causality firing is "
                        "visible rather than just the cumulative state."
                    )
                    causal_count = ui.label("").classes(
                        "text-xs text-slate-500"
                    )

                causal_chart = ui.column().classes(
                    "w-full bg-white border border-slate-200 "
                    "rounded-xl shadow-sm p-3 mt-2"
                ).style("height: 520px;")

                def _refresh_causal(**kw):
                    causal_chart.clear()
                    ws = state.world_state
                    if ws is None or not ws.causal_topology:
                        with causal_chart:
                            render_empty_state(
                                "No causal topology.",
                                icon="account_tree",
                                hint=(
                                    "Run causal extraction to populate "
                                    "CausalEdges and watch the graph "
                                    "evolve over fabula time."
                                ),
                            )
                        causal_slider_row.set_visibility(False)
                        return

                    edge_times = [
                        ce.fabula_time for ce in ws.causal_topology
                    ]
                    tmin = min(edge_times)
                    tmax = max(edge_times)
                    if tmax <= tmin:
                        causal_slider_row.set_visibility(False)
                        t = tmax
                    else:
                        causal_slider_row.set_visibility(True)
                        _set_slider_bounds(causal_slider, tmin, tmax)
                        # Source of truth: AppState.fabula_cursor
                        cur = state.fabula_cursor
                        if cur is None or cur < tmin or cur > tmax:
                            cur = tmax
                        if int(causal_slider.value or 0) != cur:
                            causal_slider.value = cur
                        t = cur
                        causal_time_label.text = f"t={t}"

                    cumulative = [
                        ce for ce in ws.causal_topology
                        if ce.fabula_time <= t
                    ]
                    delta_edges = (
                        [ce for ce in cumulative if ce.fabula_time == t]
                        if causal_show_deltas.value else []
                    )
                    causal_count.text = (
                        f"{len(cumulative)} / {len(ws.causal_topology)} edges"
                        + (f" (+{len(delta_edges)} new)"
                           if delta_edges else "")
                    )

                    # Use the canonical snapshot so all topologies, events,
                    # and replayed entity / world-trait state are consistent
                    # with what every other time-cursored view shows.
                    from shadow_loom_ui.viz_helpers import snapshot_world_at
                    try:
                        snap_ws = snapshot_world_at(ws, t)
                    except Exception:
                        logger.exception("Causal snapshot failed; manual fallback")
                        snap_ws = ws.model_copy(
                            update={"causal_topology": cumulative}
                        )
                    layout = causal_layout.value or "cartesian"
                    delta_ids = {
                        f"{ce.source_id}->{ce.target_id}"
                        for ce in delta_edges
                    }
                    with causal_chart:
                        with_expand(
                            lambda h, w=snap_ws, lay=layout, dids=delta_ids: (
                                render_causal_force_graph(
                                    w, height=h, layout=lay,
                                    highlight_edge_ids=dids,
                                )
                            ),
                            title=f"Causal graph @ t={t} ({layout})",
                        )

                def _on_causal_slider_change():
                    try:
                        t = int(causal_slider.value)
                    except (TypeError, ValueError):
                        return
                    # Push to global cursor; the listener triggers _refresh_causal.
                    state.set_fabula_cursor(t)

                causal_slider.on(
                    "change", lambda: _on_causal_slider_change()
                )
                causal_layout.on(
                    "update:model-value", lambda: _refresh_causal()
                )
                causal_show_deltas.on(
                    "update:model-value", lambda: _refresh_causal()
                )
                # Re-render whenever any other panel moves the cursor,
                # but only when the Evolution tab is actually visible.
                def _causal_cursor_listener(**_kw):
                    if state.is_path_visible("causality.evolution"):
                        _refresh_causal()

                state.on(
                    StateEvent.FABULA_CURSOR_CHANGED,
                    _causal_cursor_listener,
                )

            # ── Physics trajectory ──────────────────────────────
            with ui.tab_panel("physics").classes("p-3"):
                ui.label(
                    "Structural physics scalars sampled at evenly-spaced "
                    "fabula anchors. Pure graph math \u2014 no LLM calls "
                    "and no per-anchor pipeline runs."
                ).classes("text-xs text-slate-500 mb-2")

                with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                    ui.icon("science", color="primary")
                    ui.label("Focus:").classes("text-sm text-slate-600")
                    physics_focus = ui.select(
                        options={"__all__": "Omniscient (all entities)"},
                        value="__all__",
                        multiple=True,
                    ).props("dense outlined options-dense use-chips").classes(
                        "min-w-72"
                    )
                    physics_focus.tooltip(
                        "Pick one or more focus entities to scope the "
                        "ego-graph at every anchor; leave omniscient for "
                        "world-wide structural totals."
                    )
                    ui.label("Samples:").classes("text-sm text-slate-600")
                    physics_samples = ui.number(
                        value=12, min=2, max=40, step=1, format="%.0f",
                    ).props("dense outlined").classes("w-20")
                    physics_samples.tooltip(
                        "How many fabula anchors to sample."
                    )

                physics_chart = ui.column().classes(
                    "w-full bg-white border border-slate-200 "
                    "rounded-xl shadow-sm p-3 mt-2"
                ).style("height: 360px;")

                def _refresh_physics(**kw):
                    physics_chart.clear()
                    ws = state.world_state
                    if ws is None or not ws.events:
                        with physics_chart:
                            render_empty_state(
                                "No events to sample.",
                                icon="science",
                                hint=(
                                    "Load a project with events so the "
                                    "physics engine has anchors to "
                                    "sample."
                                ),
                            )
                        return

                    raw = physics_focus.value or []
                    if isinstance(raw, str):
                        raw = [raw]
                    focus_ids = [
                        eid for eid in raw if eid and eid != "__all__"
                    ]
                    try:
                        samples = int(physics_samples.value or 12)
                    except (TypeError, ValueError):
                        samples = 12

                    # Refresh selector options from the live world.
                    opts = {"__all__": "Omniscient (all entities)"}
                    opts.update(
                        {eid: ent.name for eid, ent in ws.entities.items()}
                    )
                    if physics_focus.options != opts:
                        physics_focus.options = opts
                        physics_focus.update()

                    fc = state.fabula_cursor
                    with physics_chart:
                        with_expand(
                            lambda h, w=ws, f=focus_ids, n=samples, fc=fc: (
                                render_physics_trajectory(
                                    w, f, samples=n, height=h,
                                    fabula_cursor=fc,
                                )
                            ),
                            title=(
                                "Physics trajectory"
                                + (f" \u2014 focus: {', '.join(focus_ids)}"
                                   if focus_ids else " \u2014 omniscient")
                            ),
                        )

                physics_focus.on(
                    "update:model-value", lambda: _refresh_physics()
                )
                physics_samples.on(
                    "change", lambda: _refresh_physics()
                )

        def _refresh_all(**kw):
            from shadow_loom_ui.viz_helpers import invalidate_snapshot_cache
            invalidate_snapshot_cache()
            _refresh_char()
            _refresh_rel()
            _refresh_world()
            _refresh_causal()
            _refresh_physics()

        # Visibility gating: skip the five evolution sub-refreshes when
        # the panel is off-screen. Cursor scrubs from other tabs no
        # longer trigger a full re-render of every timeline.
        _EVO_PATH = "causality.evolution"
        _evo_dirty = {"on": True}
        _refresh_all_sync = _refresh_all

        def _evo_gated(**kw):
            if not state.is_path_visible(_EVO_PATH):
                _evo_dirty["on"] = True
                return
            _evo_dirty["on"] = False
            _refresh_all_sync()

        def _evo_physics_gated(**kw):
            if not state.is_path_visible(_EVO_PATH):
                _evo_dirty["on"] = True
                return
            _refresh_physics()

        def _on_evo_path(**kw):
            if _evo_dirty["on"] and state.is_path_visible(_EVO_PATH):
                _evo_dirty["on"] = False
                _refresh_all_sync()

        _refresh_all_sync()
        state.on(StateEvent.WORLD_STATE_CHANGED, _evo_gated)
        state.on(StateEvent.ACTIVE_PATH_CHANGED, _on_evo_path)
        # Physics needle depends on the global fabula cursor; without
        # this listener, scrubbing the World tab or Affective Dashboard
        # left the trajectory cursor stale. (The causal-graph panel
        # already subscribes to FABULA_CURSOR_CHANGED separately.)
        state.on(
            StateEvent.FABULA_CURSOR_CHANGED,
            _evo_physics_gated,
        )


# =====================================================================
# 2. What-If Workbench (NL-driven with propagation viz)
# =====================================================================

def _build_whatif_workbench(state: AppState) -> None:
    """NL-driven intervention and counterfactual with propagation waterfall."""

    with ui.column().classes("w-full h-full gap-4"):
        # ── Single NL input with type toggle ──────────────────────
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("science", color="primary")
                ui.label("What-If Explorer").classes(
                    "text-lg font-semibold text-slate-800"
                )

            ui.label(
                "Describe what you want to change (intervention) or explore "
                "(counterfactual) in natural language."
            ).classes("text-sm text-slate-500 mb-2")

            whatif_type = ui.toggle(
                {"intervention": "Intervene (now)", "counterfactual": "What-If (past)"},
                value="intervention",
            ).props("dense no-caps color=primary")
            whatif_type.tooltip(
                "Intervene = mutate the world from the present forward. "
                "What-If = abduce the past, then re-propagate."
            )

            whatif_input = ui.textarea(
                placeholder="e.g., 'Kill Macbeth' or 'What if Romeo never met Juliet?'",
            ).classes("w-full q-mt-sm").props("rows=2 autogrow outlined dense")
            whatif_input.tooltip(
                "Free-form natural language. The parser maps this to a "
                "structured intervention or counterfactual query."
            )

            # NL suggestion chips
            with ui.row().classes("gap-1 flex-wrap q-mt-xs"):
                def _fill(text, wtype):
                    whatif_input.value = text
                    whatif_type.value = wtype

                ui.chip("Kill a character…", icon="flash_on",
                        on_click=lambda: _fill("Kill ", "intervention")).props("dense outline size=sm clickable")
                ui.chip("What if X never happened…", icon="alt_route",
                        on_click=lambda: _fill("What if ", "counterfactual")).props("dense outline size=sm clickable")
                ui.chip("Move character to…", icon="place",
                        on_click=lambda: _fill("Move ", "intervention")).props("dense outline size=sm clickable")
                ui.chip("Change relationship…", icon="people",
                        on_click=lambda: _fill("Make the relationship between ", "intervention")).props("dense outline size=sm clickable")

            result_container = ui.column().classes("w-full q-mt-md")

            async def _run_whatif():
                text = whatif_input.value.strip()
                if not text:
                    return
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return

                qtype = whatif_type.value
                try:
                    result, _task = await run_query_as_task(
                        state,
                        label=f"{qtype.title()}: {text[:40]}{'…' if len(text) > 40 else ''}",
                        kind=qtype,
                        runner=lambda: state.run_nl_query_async(
                            text, query_type=qtype,
                        ),
                        summary_fn=lambda r: (r.summary if r else "") or "Done",
                    )
                    _render_whatif_result(result_container, result)
                except Exception as e:
                    logger.exception("What-if query failed")

            ui.button(
                "Run", icon="play_arrow", on_click=_run_whatif
            ).props("unelevated color=primary no-caps").classes("rounded-lg shadow-sm")


def _render_whatif_result(container, result: NLQueryResult) -> None:
    """Render intervention/counterfactual result with propagation waterfall."""
    container.clear()

    if result.error:
        with container:
            ui.label(f"Error: {result.error}").classes("text-negative")
        return

    pr = result.pipeline_result
    if pr is None:
        with container:
            ui.label("No result.").classes("text-sm text-slate-400 italic")
        return

    with container:
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2 mb-3"):
                ui.badge(pr.query_type, color="primary").props("dense")
                if pr.world_model:
                    ui.badge(f"v{pr.world_model.version}", color="secondary").props("dense")
                if pr.converged is not None:
                    status = "converged" if pr.converged else "did not converge"
                    color = "positive" if pr.converged else "warning"
                    ui.badge(f"Audit: {status}", color=color).props("dense")

            if pr.prose:
                with ui.expansion("Generated Prose", icon="article", value=True).props("dense"):
                    ui.markdown(pr.prose[:2000])

            # Propagation waterfall (if mutations available in physics result)
            if pr.physics_result:
                mutations = pr.physics_result.get("mutations", [])
                blocked = pr.physics_result.get("blocked", [])
                if mutations or blocked:
                    with ui.expansion("Causal Propagation", icon="trending_up", value=True).props("dense"):
                        with ui.tabs().props("dense") as _prop_tabs:
                            _t_water = ui.tab("Waterfall", icon="bar_chart")
                            _t_graph = ui.tab("Graph", icon="hub")
                            _t_table = ui.tab("Table", icon="table_view")
                        with ui.tab_panels(_prop_tabs, value=_t_water).classes("w-full"):
                            with ui.tab_panel(_t_water):
                                with_expand(
                                    lambda h, m=mutations, b=blocked: (
                                        render_propagation_waterfall(
                                            m, b, height=h
                                        )
                                    ),
                                    title="Causal propagation waterfall",
                                    height="250px",
                                )
                            with ui.tab_panel(_t_graph):
                                with_expand(
                                    lambda h, m=mutations, b=blocked: (
                                        render_propagation_graph(
                                            m, b, height=h
                                        )
                                    ),
                                    title="Causal propagation graph",
                                    height="320px",
                                )
                            with ui.tab_panel(_t_table):
                                rows = mutations_to_propagation_rows(
                                    mutations, blocked
                                )
                                if rows:
                                    ui.table(
                                        columns=[
                                            {"name": "step", "label": "Step", "field": "step", "sortable": True},
                                            {"name": "entity", "label": "Entity", "field": "entity", "sortable": True},
                                            {"name": "trait", "label": "Trait", "field": "trait", "sortable": True},
                                            {"name": "delta", "label": "\u0394", "field": "delta", "sortable": True},
                                            {"name": "from_value", "label": "From", "field": "from_value"},
                                            {"name": "to_value", "label": "To", "field": "to_value"},
                                            {"name": "reason", "label": "Reason", "field": "reason"},
                                            {"name": "blocked", "label": "Blocked?", "field": "blocked", "sortable": True},
                                        ],
                                        rows=rows,
                                        pagination={"rowsPerPage": 15},
                                    ).props("dense flat bordered").classes("w-full")
                                else:
                                    ui.label("No propagation rows.").classes(
                                        "text-sm text-slate-500 italic"
                                    )

                # Hidden deltas (counterfactual)
                hidden = pr.physics_result.get("hidden_deltas", [])
                if hidden:
                    with ui.expansion("Hidden Deltas (abduction)", icon="visibility_off").props("dense"):
                        for hd in hidden:
                            def _fmt(v):
                                try:
                                    return f"{float(v):.2f}"
                                except (TypeError, ValueError):
                                    return "?"
                            ui.label(
                                f"{hd.get('entity', '?')}:{hd.get('trait', '?')} "
                                f"factual={_fmt(hd.get('factual'))} \u2192 "
                                f"counterfactual={_fmt(hd.get('counterfactual'))}"
                            ).classes("text-xs text-slate-500")

            if pr.physics_result and not pr.prose:
                with ui.expansion("Raw Physics", icon="data_object").props("dense"):
                    import json as _json
                    ui.code(_json.dumps(pr.physics_result, default=str)[:1500], language="json")


# =====================================================================
# 3. Directive Builder (visual + NL hybrid)
# =====================================================================

def _build_directive_builder(state: AppState) -> None:
    """Interactive emotional target builder with NL override."""

    with ui.column().classes("w-full h-full gap-4"):
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("theater_comedy", color="secondary")
                ui.label("Emotional Directive Builder").classes(
                    "text-lg font-semibold text-slate-800"
                )

            ui.label(
                "Choose an emotional effect to optimize the next scene for. "
                "Or describe what you want in natural language below."
            ).classes("text-sm text-slate-500 mb-2")

            # Entity picker
            entity_select = ui.select(
                options={},
                label="Target entities",
                multiple=True,
            ).classes("w-full").props("outlined dense use-chips")
            entity_select.tooltip(
                "Which characters the directive should focus on"
            )

            # Emotional target selector
            effect_select = ui.select(
                options={k: v for k, v in _EMOTIONAL_TARGETS},
                value="suspense",
                label="Target emotion",
            ).classes("w-64 q-mt-sm").props("outlined dense")
            effect_select.tooltip(
                "The dominant affect the next scene should evoke in \n"
                "the reader"
            )

            # Intensity slider
            with ui.row().classes("w-full items-center gap-2 q-mt-sm"):
                ui.label("Intensity:").classes("text-sm text-slate-600")
                intensity_slider = ui.slider(
                    min=0.1, max=1.0, value=0.7, step=0.1
                ).classes("flex-grow").props("label-always")
                intensity_slider.tooltip(
                    "How strongly to push the chosen emotion. "
                    "0.1 = subtle, 1.0 = full saturation."
                )

            # NL override
            nl_input = ui.textarea(
                placeholder="Or describe in natural language: 'Make the reader feel dread as the storm approaches…'",
            ).classes("w-full q-mt-sm").props("rows=2 autogrow outlined dense")

            result_container = ui.column().classes("w-full q-mt-md")

            async def _run_directive():
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return

                # If NL text is provided, use it directly
                nl_text = nl_input.value.strip()
                if nl_text:
                    query_text = nl_text
                else:
                    # Build from visual inputs
                    effect = effect_select.value
                    intensity = intensity_slider.value
                    entities = entity_select.value or []
                    entity_names = []
                    for eid in entities:
                        ent = state.world_state.entities.get(eid)
                        if ent:
                            entity_names.append(ent.name)

                    target_str = ", ".join(entity_names) if entity_names else "the characters"
                    query_text = (
                        f"Create a scene that makes {target_str} feel {effect.replace('_', ' ')} "
                        f"with intensity {intensity:.1f}"
                    )

                state.emit(StateEvent.QUERY_STARTED)
                try:
                    result, _task = await run_query_as_task(
                        state,
                        label=f"Directive: {effect_select.value}",
                        kind="directive",
                        runner=lambda: state.run_nl_query_async(
                            query_text, query_type="directive",
                        ),
                        summary_fn=lambda r: (r.summary if r else "") or "Done",
                    )
                    _render_directive_result(result_container, result)
                except Exception as e:
                    logger.exception("Directive failed")

            ui.button(
                "Generate Directive", icon="play_arrow", on_click=_run_directive
            ).props("unelevated color=secondary no-caps").classes("rounded-lg shadow-sm")

        # Update entity options on world state change
        def _update_entities(**kw):
            ws = state.world_state
            if ws is None:
                opts: dict = {}
            else:
                opts = {eid: ent.name for eid, ent in ws.entities.items()}
            if entity_select.options != opts:
                entity_select.options = opts
                # Drop any selected ids that no longer exist.
                if entity_select.value:
                    entity_select.value = [
                        v for v in entity_select.value if v in opts
                    ]
                entity_select.update()

        _update_entities()
        state.on(StateEvent.WORLD_STATE_CHANGED, _update_entities)


def _render_directive_result(container, result: NLQueryResult) -> None:
    """Render directive result."""
    container.clear()
    if result.error:
        with container:
            ui.label(f"Error: {result.error}").classes("text-negative")
        return

    pr = result.pipeline_result
    if pr is None:
        with container:
            ui.label("No result.").classes("text-sm text-slate-400 italic")
        return

    with container:
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2 mb-3"):
                ui.badge("directive", color="secondary").props("dense")
                if pr.converged is not None:
                    status = "converged" if pr.converged else "did not converge"
                    color = "positive" if pr.converged else "warning"
                    ui.badge(f"Audit: {status}", color=color).props("dense")

            if pr.prose:
                ui.markdown(pr.prose)


# =====================================================================
# 4. Affective Dashboard
# =====================================================================

def _build_affective_dashboard(state: AppState) -> None:
    """Emotional gauges + affective metrics over fabula time.

    Pared back from the previous collage of densities, polars, and
    calendar overlays to two focused panels:

      1. Gauges (current snapshot) \u2014 at-a-glance qualitative state.
      2. Multi-line time-series \u2014 how each affective metric evolves
         across fabula time, with the active cursor as a needle.

    Plus a compact event-timeline scatter for context.
    """

    from shadow_loom_ui.viz_helpers import (
        compute_affective_scores,
        fabula_time_bounds,
        invalidate_snapshot_cache,
        snapshot_world_at,
        snapshot_world_at_syuzhet,
        syuzhet_time_bounds,
        _set_slider_bounds,
        _top_entity_ids_by_event_degree,
    )

    # Re-entrancy guard: setting time_slider.value programmatically inside
    # _refresh used to re-fire ``update:model-value``, causing a feedback
    # loop that froze the UI on every drag.
    _suppress = {"slider": False}

    with ui.column().classes("w-full h-full gap-4"):
        # ── Timeline cursor row (Fabula | Syuzhet mode) ──────────
        slider_row = ui.row().classes(
            "w-full items-center gap-3 bg-white border border-slate-200 "
            "rounded-xl shadow-sm p-3 flex-wrap"
        )
        with slider_row:
            ui.icon("schedule", color="primary")
            mode_toggle = ui.toggle(
                {"fabula": "Fabula", "syuzhet": "Syuzhet"},
                value="fabula",
            ).props("dense no-caps color=primary")
            mode_toggle.tooltip(
                "Fabula = chronological story-world time. "
                "Syuzhet = the order the reader encounters events."
            )
            time_label = ui.label("live").classes(
                "text-sm font-mono text-slate-700 w-16"
            )
            time_slider = ui.slider(min=0, max=1, value=0, step=1).props(
                "color=primary label-always dense"
            ).classes("flex-grow min-w-32")
            time_slider.tooltip(
                "Scrub the timeline. \u2190/\u2192 step, space = live, "
                "f / s switch mode"
            )

            def _set_live():
                # Route through the setter so the FABULA/SYUZHET_CURSOR_CHANGED
                # subscription drives _refresh — no manual call here.
                if mode_toggle.value == "syuzhet":
                    state.set_syuzhet_cursor(None)
                    _url_set_param("syuzhet", None)
                else:
                    state.set_fabula_cursor(None)
                    _url_set_param("fabula", None)

            ui.button(
                "Live", icon="bolt", on_click=_set_live
            ).props("flat dense no-caps color=secondary")

            graded_gauges = ui.checkbox("Graded gauges", value=True).tooltip(
                "Show qualitative bands (low / moderate / high) on the affect gauges"
            )
            graded_gauges.on("update:model-value", lambda _e: _refresh())

            ui.label("Gauge:").classes("text-sm text-slate-600")
            gauge_select = ui.select(
                options={"__all__": "All metrics"},
                value="__all__",
            ).props("dense outlined options-dense").classes("min-w-40")
            gauge_select.tooltip(
                "Pick a single affective metric to focus the gauge on, "
                "or 'All metrics' to compare side-by-side."
            )
            gauge_select.on("update:model-value", lambda _e: _refresh())

        def _on_slider_change():
            if _suppress["slider"]:
                return
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            if mode_toggle.value == "syuzhet":
                if state.syuzhet_cursor == t:
                    return
                # Setter emits SYUZHET_CURSOR_CHANGED → ``_refresh`` runs
                # via the subscription registered below. Calling _refresh
                # here as well would double-render every slider release.
                state.set_syuzhet_cursor(t)
                _url_set_param("syuzhet", t)
            else:
                if state.fabula_cursor == t:
                    return
                state.set_fabula_cursor(t)
                _url_set_param("fabula", t)

        # Release-only: ``change`` fires once when the user releases the
        # thumb, avoiding the per-pixel rerender storm that froze the UI.
        time_slider.on("change", lambda: _on_slider_change())
        mode_toggle.on("update:model-value", lambda: _refresh())

        # Keyboard shortcuts on the slider element: \u2190/\u2192 step,
        # space toggles live, f/s switch mode.
        def _step(delta: int):
            try:
                cur = int(time_slider.value)
            except (TypeError, ValueError):
                return
            time_slider.value = cur + delta
            _on_slider_change()

        time_slider.on("keydown.left", lambda: _step(-1))
        time_slider.on("keydown.right", lambda: _step(1))
        time_slider.on("keydown.space", lambda: _set_live())
        time_slider.on(
            "keydown.f", lambda: (
                setattr(mode_toggle, "value", "fabula"), _refresh()
            )
        )
        time_slider.on(
            "keydown.s", lambda: (
                setattr(mode_toggle, "value", "syuzhet"), _refresh()
            )
        )

        gauge_container = ui.column().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        )
        timeline_container = ui.column().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        )

        # ── Stable timeline-chart slots (in-place updated) ─────────
        # Building two long-lived ``ui.echart`` handles here lets the
        # cursor-driven refresh patch ``chart.options`` in place via
        # :func:`update_chart_options` instead of clearing the column
        # and re-creating the DOM (which is what froze the UI under
        # rapid scrubbing).
        from shadow_loom_ui.viz import update_chart_options

        # Latest timeseries options snapshot — read by the
        # expand-to-dialog render_fn so the popout chart mirrors
        # whatever the inline chart currently shows.
        _ts_snapshot: dict = {"opts": None, "title": ""}
        # Same idea for the event-timeline scatter so its expand
        # button mirrors the live cursor + scatter data.
        _et_snapshot: dict = {"opts": None, "title": "Event Timeline"}

        def _render_ts_expanded(height: str) -> None:
            opts = _ts_snapshot["opts"]
            if opts is None:
                ui.label("No affective signal yet.").classes(
                    "text-sm text-slate-400 italic"
                )
                return
            ui.echart(opts).classes("w-full").style(f"height: {height};")

        def _render_et_expanded(height: str) -> None:
            opts = _et_snapshot["opts"]
            if opts is None:
                ui.label("No events.").classes(
                    "text-sm text-slate-400 italic"
                )
                return
            ui.echart(opts).classes("w-full").style(f"height: {height};")

        with timeline_container:
            with ui.row().classes("w-full items-center justify-between"):
                timeseries_label = ui.label("").classes(
                    "text-lg font-semibold text-slate-800 mb-2"
                )
                with ui.row().classes("items-center gap-2"):
                    normalize_toggle = ui.switch(
                        "Normalize", value=False,
                        on_change=lambda _: _refresh(),
                    ).props("dense").tooltip(
                        "Min-max scale each metric to [0,1] so trajectory "
                        "shapes are comparable across metrics with very "
                        "different magnitudes."
                    )
                    ui.button(
                        icon="open_in_full",
                        on_click=lambda: _open_ts_dialog(),
                    ).props("flat dense round size=sm color=grey-7").tooltip(
                        "Expand to full screen"
                    )
            timeseries_chart = ui.echart({}).classes("w-full").style(
                "height: 280px;"
            )
            timeseries_empty = ui.label("No affective signal yet.").classes(
                "text-sm text-slate-400 italic"
            )
            timeseries_empty.set_visibility(False)

            with ui.row().classes("w-full items-center justify-between mt-4"):
                event_timeline_label = ui.label("Event Timeline").classes(
                    "text-sm font-semibold text-slate-700 mb-1"
                )
                ui.button(
                    icon="open_in_full",
                    on_click=lambda: _open_et_dialog(),
                ).props("flat dense round size=sm color=grey-7").tooltip(
                    "Expand to full screen"
                )
            event_timeline_chart = ui.echart({}).classes("w-full").style(
                "height: 260px;"
            )
            event_timeline_empty = ui.label("No events.").classes(
                "text-sm text-slate-400 italic"
            )
            event_timeline_empty.set_visibility(False)

        def _open_ts_dialog() -> None:
            from shadow_loom_ui.viz import _open_expand_dialog
            _open_expand_dialog(
                _render_ts_expanded,
                _ts_snapshot["title"] or "Affective Metrics",
            )

        def _open_et_dialog() -> None:
            from shadow_loom_ui.viz import _open_expand_dialog
            _open_expand_dialog(
                _render_et_expanded,
                _et_snapshot["title"] or "Event Timeline",
            )

        # ── Data tables (events + affect only) ─────────────────────
        from shadow_loom_ui.viz_helpers import ws_to_event_rows

        data_expansion = ui.expansion("Raw Data", icon="table_chart").classes(
            "w-full bg-white border border-slate-200 rounded-xl"
        )
        with data_expansion:
            with ui.tabs().classes("w-full").props("dense") as data_tabs:
                ui.tab("events", label="Events", icon="bolt")
                ui.tab("affect", label="Affect", icon="favorite")

            _tprops = "dense flat bordered"
            with ui.tab_panels(data_tabs, value="events").classes("w-full"):
                with ui.tab_panel("events"):
                    event_table = ui.table(
                        columns=[
                            {"name": "id", "label": "ID", "field": "id", "sortable": True},
                            {"name": "fabula_time", "label": "Fabula t", "field": "fabula_time", "sortable": True},
                            {"name": "syuzhet_index", "label": "Syuzhet", "field": "syuzhet_index", "sortable": True},
                            {"name": "type", "label": "Type", "field": "type", "sortable": True},
                            {"name": "actors", "label": "Actors", "field": "actors"},
                            {"name": "targets", "label": "Targets", "field": "targets"},
                            {"name": "description", "label": "Description", "field": "description"},
                        ],
                        rows=[],
                        pagination={"rowsPerPage": 10},
                    ).props(_tprops).classes("w-full")
                with ui.tab_panel("affect"):
                    affect_table = ui.table(
                        columns=[
                            {"name": "metric", "label": "Metric", "field": "metric", "sortable": True},
                            {"name": "score", "label": "Score", "field": "score", "sortable": True},
                        ],
                        rows=[],
                    ).props(_tprops).classes("w-full")

        def _refresh(**kw):
            gauge_container.clear()
            ws = state.world_state
            if ws is None:
                slider_row.set_visibility(False)
                # Hide the stable timeline charts while there's no
                # world; the empty labels take their place.
                timeseries_chart.set_visibility(False)
                event_timeline_chart.set_visibility(False)
                timeseries_empty.set_visibility(True)
                event_timeline_empty.set_visibility(True)
                timeseries_label.text = ""
                with gauge_container:
                    if state.running_task_count > 0:
                        render_chart_skeleton("220px")
                    else:
                        ui.label("No world model loaded.").classes(
                            "text-sm text-slate-400 italic"
                        )
                return

            is_syuzhet = mode_toggle.value == "syuzhet"
            if is_syuzhet:
                tmin, tmax = syuzhet_time_bounds(ws)
                cursor = state.syuzhet_cursor
                cursor_prefix = "s"
            else:
                tmin, tmax = fabula_time_bounds(ws)
                cursor = state.fabula_cursor
                cursor_prefix = "t"

            if tmax > tmin:
                slider_row.set_visibility(True)
                # Only mutate slider state when something actually changes.
                # Setting ``time_slider.value`` always echoes back through
                # ``update:model-value`` (Quasar fires it asynchronously),
                # which used to re-enter ``_refresh`` *after* the in-flight
                # ``_suppress`` window had already closed — freezing the UI
                # mid-drag. We compare-then-set to break that loop.
                _set_slider_bounds(time_slider, tmin, tmax)
                if cursor is None:
                    desired = tmax
                    label_text = "live"
                else:
                    desired = max(tmin, min(tmax, cursor))
                    label_text = f"{cursor_prefix}={desired}"
                if int(time_slider.value or 0) != desired:
                    _suppress["slider"] = True
                    try:
                        time_slider.value = desired
                    finally:
                        _suppress["slider"] = False
                if time_label.text != label_text:
                    time_label.text = label_text
                if cursor is not None:
                    try:
                        if is_syuzhet:
                            ws = snapshot_world_at_syuzhet(ws, cursor)
                        else:
                            ws = snapshot_world_at(ws, cursor)
                    except Exception:
                        logger.exception(
                            "Snapshot failed; falling back to live"
                        )
            else:
                slider_row.set_visibility(False)

            # Engine-grade affects (suspense, surprise, dramatic_irony,
            # canonical mystery) need a focus entity set + syuzhet
            # anchor. Use the top-N entities by event-degree to bound
            # cost on large worlds; default the anchor to the snapshot's
            # max syuzhet so suspense/surprise reflect the unrevealed
            # tail rather than collapsing to zero.
            entity_ids = _top_entity_ids_by_event_degree(ws, limit=20)
            if is_syuzhet and state.syuzhet_cursor is not None:
                anchor = state.syuzhet_cursor
            else:
                anchor = max(
                    (e.syuzhet_index for e in ws.events), default=None
                )
            scores = compute_affective_scores(
                ws, entity_ids=entity_ids, syuzhet_anchor=anchor,
            )

            # Refresh gauge-select options so they mirror the currently
            # available metrics; preserve the user's selection if still valid.
            opts = {"__all__": "All metrics"}
            opts.update({k: k.replace("_", " ") for k in scores})
            if gauge_select.options != opts:
                gauge_select.options = opts
                if gauge_select.value not in opts:
                    gauge_select.value = "__all__"
                gauge_select.update()

            with gauge_container:
                if scores:
                    ui.label("Narrative Affect Scores").classes(
                        "text-lg font-semibold text-slate-800 mb-2"
                    )
                    sel = gauge_select.value
                    sel = None if sel in (None, "", "__all__") else sel
                    # Per-gauge height — the gauges are now laid out in
                    # a CSS grid (one ECharts per metric) so this is
                    # the height of each cell, not the whole container.
                    height_px = "200px" if sel else "180px"
                    with ui.element("div").classes("w-full"):
                        with_expand(
                            lambda h, s=scores, g=graded_gauges.value, sel=sel, hp=height_px: (
                                render_emotional_gauges_graded(
                                    s, height=hp, selected=sel
                                ) if g else render_emotional_gauges(
                                    s, height=hp, selected=sel
                                )
                            ),
                            title=(
                                f"Affect gauge \u2014 {sel.replace('_', ' ')}"
                                if sel else "Narrative affect scores"
                            ),
                        )
                else:
                    ui.label("No affective scores available.").classes(
                        "text-sm text-slate-400 italic"
                    )

            with timeline_container:
                axis_label = "Syuzhet Index" if is_syuzhet else "Fabula Time"
                # Stable timeline charts are built once at panel-build
                # time; here we just patch their options + label in
                # place. Avoids the DOM teardown that was the dominant
                # cost on every cursor scrub.
                fc = state.fabula_cursor if not is_syuzhet else None
                sc_for_chart = state.syuzhet_cursor if is_syuzhet else None
                axis = "syuzhet" if is_syuzhet else "fabula"
                eids = entity_ids
                from shadow_loom_ui.viz import (
                    affective_timeseries_options,
                    event_timeline_options,
                )

                ts_opts = affective_timeseries_options(
                    ws,
                    fabula_cursor=fc,
                    syuzhet_cursor=sc_for_chart,
                    axis=axis,
                    entity_ids=eids,
                    normalize=bool(normalize_toggle.value),
                )
                timeseries_label.text = (
                    f"Affective Metrics over {axis_label}"
                )
                _ts_snapshot["title"] = (
                    f"Affective Metrics over {axis_label}"
                )
                _ts_snapshot["opts"] = ts_opts
                if ts_opts is None:
                    timeseries_chart.set_visibility(False)
                    timeseries_empty.set_visibility(True)
                else:
                    timeseries_empty.set_visibility(False)
                    timeseries_chart.set_visibility(True)
                    update_chart_options(timeseries_chart, ts_opts)

                sc = state.syuzhet_cursor if is_syuzhet else None
                et_opts = event_timeline_options(
                    ws, fabula_cursor=fc, syuzhet_cursor=sc,
                )
                _et_snapshot["opts"] = et_opts
                _et_snapshot["title"] = f"Event Timeline ({axis_label})"
                if et_opts is None:
                    event_timeline_chart.set_visibility(False)
                    event_timeline_empty.set_visibility(True)
                else:
                    event_timeline_empty.set_visibility(False)
                    event_timeline_chart.set_visibility(True)
                    update_chart_options(event_timeline_chart, et_opts)

            # Refresh data tables to mirror the charts above.
            event_table.rows = ws_to_event_rows(ws)
            affect_table.rows = [
                {"metric": k, "score": round(v, 3)}
                for k, v in scores.items()
            ]

        def _on_world_changed(**kw):
            invalidate_snapshot_cache()
            _refresh()

        # ── URL-sync hydration ─────────────────────────────────
        async def _hydrate_cursor_from_url():
            params = await _url_get_params()
            if "syuzhet" in params:
                try:
                    mode_toggle.value = "syuzhet"
                    # Use the setter so every other time-aware panel
                    # (World tab, Causality Sankey, Physics) hydrates
                    # to the same cursor in lockstep instead of just
                    # the affective dashboard.
                    state.set_syuzhet_cursor(int(params["syuzhet"]))
                except (TypeError, ValueError):
                    pass
            if "fabula" in params:
                try:
                    state.set_fabula_cursor(int(params["fabula"]))
                except (TypeError, ValueError):
                    pass

        ui.timer(
            0.1,
            lambda: state.spawn_task(_hydrate_cursor_from_url()),
            once=True,
        )

        def _on_tasks_aff(**kw):
            if state.world_state is None:
                _refresh()

        # ── Visibility gating + per-panel async cancellation ──
        # The Affective Dashboard does the most work per cursor scrub
        # (snapshot + engine scoring + timeseries resampling). Skip
        # entirely when off-screen, and route refreshes through a
        # named task slot so rapid drags collapse to one render.
        _PANEL_PATH = "causality.affective"
        _PANEL_ID = "affective_dashboard"
        _dirty = {"on": True}

        async def _refresh_async():
            ws = state.world_state
            if ws is None:
                _refresh()
                return
            # Warm the heavy caches off the event loop. Both functions
            # are memoised on (id(ws), revision, ...), so the synchronous
            # _refresh() below hits the cache and returns instantly.
            try:
                cursor = (
                    state.syuzhet_cursor
                    if mode_toggle.value == "syuzhet"
                    else state.fabula_cursor
                )
                eids = await asyncio.to_thread(
                    _top_entity_ids_by_event_degree, ws, 20,
                )
                if mode_toggle.value == "syuzhet" and cursor is not None:
                    snap = await asyncio.to_thread(
                        snapshot_world_at_syuzhet, ws, cursor,
                    )
                elif cursor is not None:
                    snap = await asyncio.to_thread(
                        snapshot_world_at, ws, cursor,
                    )
                else:
                    snap = ws
                anchor = (
                    cursor if mode_toggle.value == "syuzhet"
                    else max(
                        (e.syuzhet_index for e in snap.events), default=None
                    )
                )
                await asyncio.to_thread(
                    compute_affective_scores,
                    snap, entity_ids=eids, syuzhet_anchor=anchor,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Affective async warm-up failed")
            _refresh_sync()

        _refresh_sync = _refresh

        def _gated_refresh(**kw):
            if not state.is_path_visible(_PANEL_PATH):
                _dirty["on"] = True
                return
            _dirty["on"] = False
            state.spawn_panel_task(_PANEL_ID, _refresh_async())

        # Re-bind the local name so all the existing slider / button
        # handlers (which captured ``_refresh``) now go through the
        # gated/async dispatcher without further changes.
        _refresh = _gated_refresh  # noqa: F811

        def _on_path_changed(**kw):
            if _dirty["on"] and state.is_path_visible(_PANEL_PATH):
                _refresh()

        # First paint runs synchronously so the panel isn't blank.
        _refresh_sync()
        state.on(StateEvent.WORLD_STATE_CHANGED, _on_world_changed)
        state.on(StateEvent.TASKS_CHANGED, _on_tasks_aff)
        state.on(StateEvent.ACTIVE_PATH_CHANGED, _on_path_changed)
        # Re-render whenever any other panel moves the global cursor
        # so the gauges, time-series, and event timeline track every
        # scrub action across the app.
        state.on(StateEvent.FABULA_CURSOR_CHANGED, lambda **_kw: _refresh())
        state.on(StateEvent.SYUZHET_CURSOR_CHANGED, lambda **_kw: _refresh())



