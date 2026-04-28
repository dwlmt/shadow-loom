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
from shadow_loom_ui.viz import (
    render_causal_force_graph,
    render_causal_sankey,
    render_chart_skeleton,
    render_displacement_chart,
    render_emotional_gauges,
    render_emotional_gauges_graded,
    render_empty_state,
    render_entity_state_timeline,
    render_event_calendar,
    render_event_polar,
    render_event_timeline,
    render_node_legend,
    render_propagation_graph,
    render_propagation_waterfall,
    render_calendar_graph_overlay,
    render_trait_radar_compare,
    render_relationship_timeline,
    render_world_trait_timeline,
    open_explain_dialog,
    with_expand,
)
from shadow_loom_ui.viz_helpers import (
    mutations_to_propagation_rows,
    ws_to_calendar_graph_rows,
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

    from shadow_loom_ui.viz_helpers import SANKEY_ASPECTS, fabula_time_bounds

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

        # Legend strip
        legend_row = ui.row().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm "
            "px-3 py-2"
        )
        with legend_row:
            render_node_legend()

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
                _suppress["t"] = True
                tmin, tmax = fabula_time_bounds(state.world_state) \
                    if state.world_state else (0, 1)
                time_slider.value = tmax
                _suppress["t"] = False
                _refresh()

            ui.button("Full span", icon="restart_alt", on_click=_reset_time) \
                .props("flat dense no-caps color=secondary")

        graph_container = ui.column().classes(
            "w-full flex-grow bg-white border border-slate-200 "
            "rounded-xl shadow-sm p-4"
        )

        # Re-entrancy guard: programmatic slider writes shouldn't re-trigger.
        _suppress: dict[str, bool] = {"t": False, "f": False}

        def _refresh(**kw):
            graph_container.clear()
            ws = state.world_state
            if ws is None:
                filter_row.set_visibility(False)
                legend_row.set_visibility(False)
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
            legend_row.set_visibility(True)

            # Sync the time slider bounds without re-triggering the handler.
            tmin, tmax = fabula_time_bounds(ws)
            if tmax > tmin:
                _suppress["t"] = True
                time_slider.props(f"min={tmin} max={tmax}")
                if time_slider.value is None or time_slider.value > tmax \
                        or time_slider.value < tmin:
                    time_slider.value = tmax
                _suppress["t"] = False

            with graph_container:
                if is_sankey:
                    fmax = int(time_slider.value) if time_slider.value else tmax
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
                                state.selected_node_id = nid
                                state.emit(StateEvent.NODE_SELECTED, node_id=nid)
                                # Also pop the explain dialog so users
                                # see the causal structure of the click.
                                if state.world_state is not None:
                                    open_explain_dialog(state.world_state, nid)
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
            if _suppress[key]:
                return
            _refresh()

        # Throttle slider events so dragging doesn't re-render per pixel.
        view_toggle.on("update:model-value", lambda: _refresh())

        def _on_aspect():
            _url_set_param("aspect", aspect_select.value)
            _refresh()

        aspect_select.on("update:model-value", lambda: _on_aspect())
        force_layout.on("update:model-value", lambda: _refresh())
        force_slider.on(
            "update:model-value",
            lambda: _on_slider("f"),
            throttle=0.25,
            leading_events=False,
        )
        time_slider.on(
            "update:model-value",
            lambda: _on_slider("t"),
            throttle=0.25,
            leading_events=False,
        )

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

        _refresh()
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)
        state.on(StateEvent.TASKS_CHANGED, _on_tasks)


# =====================================================================
# 1b. Evolution panel — character / relationship / world-trait timelines
# =====================================================================

def _build_evolution_panel(state: AppState) -> None:
    """How characters, relationships, and world traits change over the story."""

    from shadow_loom_ui.viz_helpers import (
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

        def _refresh_all(**kw):
            from shadow_loom_ui.viz_helpers import invalidate_snapshot_cache
            invalidate_snapshot_cache()
            _refresh_char()
            _refresh_rel()
            _refresh_world()

        _refresh_all()
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_all)


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

                state.emit(StateEvent.QUERY_STARTED)
                try:
                    result = await state.run_nl_query_async(
                        text, query_type=whatif_type.value
                    )
                    _render_whatif_result(result_container, result)
                except Exception as e:
                    logger.exception("What-if query failed")
                    ui.notify(f"Error: {e}", type="negative")

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
                                render_propagation_waterfall(mutations, blocked, height="250px")
                            with ui.tab_panel(_t_graph):
                                render_propagation_graph(
                                    mutations, blocked, height="320px"
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
                options=[],
                label="Target entities",
                multiple=True,
            ).classes("w-full").props("outlined dense")
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
                    result = await state.run_nl_query_async(
                        query_text, query_type="directive"
                    )
                    _render_directive_result(result_container, result)
                except Exception as e:
                    logger.exception("Directive failed")
                    ui.notify(f"Error: {e}", type="negative")

            ui.button(
                "Generate Directive", icon="play_arrow", on_click=_run_directive
            ).props("unelevated color=secondary no-caps").classes("rounded-lg shadow-sm")

        # Update entity options on world state change
        def _update_entities(**kw):
            ws = state.world_state
            if ws:
                entity_select.options = {eid: ent.name for eid, ent in ws.entities.items()}

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
    """Emotional gauges, narrative tension, and candidate events."""

    from shadow_loom_ui.viz_helpers import (
        fabula_time_bounds,
        invalidate_snapshot_cache,
        snapshot_world_at,
        snapshot_world_at_syuzhet,
        syuzhet_time_bounds,
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
                if mode_toggle.value == "syuzhet":
                    state.syuzhet_cursor = None
                    _url_set_param("syuzhet", None)
                else:
                    state.fabula_cursor = None
                    _url_set_param("fabula", None)
                time_label.text = "live"
                _refresh()

            ui.button(
                "Live", icon="bolt", on_click=_set_live
            ).props("flat dense no-caps color=secondary")

            graded_gauges = ui.checkbox("Graded gauges", value=True).tooltip(
                "Show qualitative bands (low / moderate / high) on the affect gauges"
            )
            graded_gauges.on("update:model-value", lambda _e: _refresh())

        def _on_slider_change():
            if _suppress["slider"]:
                return
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            if mode_toggle.value == "syuzhet":
                state.syuzhet_cursor = t
                time_label.text = f"s={t}"
                _url_set_param("syuzhet", t)
            else:
                state.fabula_cursor = t
                time_label.text = f"t={t}"
                _url_set_param("fabula", t)
            _refresh()

        # Throttle: Quasar fires ``update:model-value`` per pixel; without
        # this the snapshot+render work piles up and the slider freezes.
        time_slider.on(
            "update:model-value",
            lambda: _on_slider_change(),
            throttle=0.2,
            leading_events=False,
        )
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

        # ── Data tables for the timeline-tab charts ────────────────
        from shadow_loom_ui.viz_helpers import (
            ws_to_event_calendar_rows,
            ws_to_event_rows,
            ws_to_polar_event_rows,
        )

        data_expansion = ui.expansion("Raw Data", icon="table_chart").classes(
            "w-full bg-white border border-slate-200 rounded-xl"
        )
        with data_expansion:
            with ui.tabs().classes("w-full").props("dense") as data_tabs:
                ui.tab("events", label="Events", icon="bolt")
                ui.tab("affect", label="Affect", icon="favorite")
                ui.tab("buckets", label="Density", icon="view_module")
                ui.tab("polar", label="Actor × Type", icon="pie_chart")

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
                with ui.tab_panel("buckets"):
                    bucket_table = ui.table(
                        columns=[
                            {"name": "bucket", "label": "Bucket", "field": "bucket", "sortable": True},
                            {"name": "events", "label": "Events", "field": "events", "sortable": True},
                        ],
                        rows=[],
                        pagination={"rowsPerPage": 10},
                    ).props(_tprops).classes("w-full")
                with ui.tab_panel("polar"):
                    polar_table = ui.table(
                        columns=[
                            {"name": "actor", "label": "Actor", "field": "actor", "sortable": True},
                            {"name": "event_type", "label": "Event Type", "field": "event_type", "sortable": True},
                            {"name": "count", "label": "Count", "field": "count", "sortable": True},
                        ],
                        rows=[],
                        pagination={"rowsPerPage": 10},
                    ).props(_tprops).classes("w-full")

        def _refresh(**kw):
            gauge_container.clear()
            timeline_container.clear()
            ws = state.world_state
            if ws is None:
                slider_row.set_visibility(False)
                with gauge_container:
                    if state.running_task_count > 0:
                        render_chart_skeleton("220px")
                    else:
                        ui.label("No world model loaded.").classes(
                            "text-sm text-slate-400 italic"
                        )
                with timeline_container:
                    if state.running_task_count > 0:
                        render_chart_skeleton("260px")
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
                _suppress["slider"] = True
                time_slider.props(f"min={tmin} max={tmax}")
                if cursor is None:
                    time_slider.value = tmax
                    time_label.text = "live"
                else:
                    capped = max(tmin, min(tmax, cursor))
                    time_slider.value = capped
                    time_label.text = f"{cursor_prefix}={capped}"
                _suppress["slider"] = False
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

            scores = _compute_affective_scores(ws)

            with gauge_container:
                if scores:
                    ui.label("Narrative Affect Scores").classes(
                        "text-lg font-semibold text-slate-800 mb-2"
                    )
                    with ui.element("div").classes("w-full").style("height: 220px;"):
                        with_expand(
                            lambda h, s=scores, g=graded_gauges.value: (
                                render_emotional_gauges_graded(s, height=h)
                                if g else render_emotional_gauges(s, height=h)
                            ),
                            title="Narrative affect scores",
                        )
                else:
                    ui.label("No affective scores available.").classes(
                        "text-sm text-slate-400 italic"
                    )

            with timeline_container:
                ui.label("Event Timeline").classes(
                    "text-lg font-semibold text-slate-800 mb-2"
                )
                # Pass the active cursor so a needle appears on whichever
                # axis the user is scrubbing.
                fc = state.fabula_cursor if not is_syuzhet else None
                sc = state.syuzhet_cursor if is_syuzhet else None
                with ui.element("div").classes("w-full").style("height: 270px;"):
                    with_expand(
                        lambda h, w=ws, fc=fc, sc=sc: render_event_timeline(
                            w, height=h, fabula_cursor=fc, syuzhet_cursor=sc
                        ),
                        title="Event timeline (fabula \u00d7 syuzhet)",
                    )

                ui.label("Fabula \u2194 Syuzhet displacement").classes(
                    "text-sm font-semibold text-slate-700 mt-4 mb-1"
                )
                with ui.element("div").classes("w-full").style("height: 240px;"):
                    with_expand(
                        lambda h, w=ws: render_displacement_chart(w, height=h),
                        title="Fabula \u2194 syuzhet displacement",
                    )

                ui.label("Event Density (fabula time)").classes(
                    "text-sm font-semibold text-slate-700 mt-4 mb-1"
                )
                with ui.element("div").classes("w-full").style("height: 180px;"):
                    with_expand(
                        lambda h, w=ws: render_event_calendar(w, height=h),
                        title="Event density",
                    )

                ui.label("Density + chain reactions").classes(
                    "text-sm font-semibold text-slate-700 mt-4 mb-1"
                )
                with ui.element("div").classes("w-full").style("height: 320px;"):
                    with_expand(
                        lambda h, w=ws: render_calendar_graph_overlay(w, height=h),
                        title="Density + chain reactions",
                    )
                cal_rows = ws_to_calendar_graph_rows(ws)
                if cal_rows:
                    with ui.expansion(
                        "Per-event bucket assignment",
                        icon="table_view",
                    ).props("dense"):
                        ui.table(
                            columns=[
                                {"name": "event", "label": "Event", "field": "event", "sortable": True},
                                {"name": "bucket", "label": "Bucket", "field": "bucket", "sortable": True},
                                {"name": "fabula_time", "label": "Fabula t", "field": "fabula_time", "sortable": True},
                                {"name": "type", "label": "Type", "field": "type", "sortable": True},
                                {"name": "actor", "label": "Actor", "field": "actor"},
                                {"name": "stack", "label": "Stack-y", "field": "stack", "sortable": True},
                            ],
                            rows=cal_rows,
                            pagination={"rowsPerPage": 10},
                        ).props("dense flat bordered").classes("w-full")

                ui.label("Actor × Event-Type Distribution").classes(
                    "text-sm font-semibold text-slate-700 mt-4 mb-1"
                )
                with ui.element("div").classes("w-full").style("height: 340px;"):
                    with_expand(
                        lambda h, w=ws: render_event_polar(w, height=h),
                        title="Actor \u00d7 event-type",
                    )

            # Refresh data tables to mirror the charts above.
            event_table.rows = ws_to_event_rows(ws)
            affect_table.rows = [
                {"metric": k, "score": round(v, 3)}
                for k, v in scores.items()
            ]
            bucket_table.rows = ws_to_event_calendar_rows(ws)
            polar_table.rows = ws_to_polar_event_rows(ws)

        def _on_world_changed(**kw):
            invalidate_snapshot_cache()
            _refresh()

        # ── URL-sync hydration ─────────────────────────────────
        async def _hydrate_cursor_from_url():
            params = await _url_get_params()
            changed = False
            if "fabula" in params:
                try:
                    state.fabula_cursor = int(params["fabula"])
                    changed = True
                except (TypeError, ValueError):
                    pass
            if "syuzhet" in params:
                try:
                    state.syuzhet_cursor = int(params["syuzhet"])
                    mode_toggle.value = "syuzhet"
                    changed = True
                except (TypeError, ValueError):
                    pass
            if changed:
                _refresh()

        ui.timer(
            0.1,
            lambda: state.spawn_task(_hydrate_cursor_from_url()),
            once=True,
        )

        def _on_tasks_aff(**kw):
            if state.world_state is None:
                _refresh()

        _refresh()
        state.on(StateEvent.WORLD_STATE_CHANGED, _on_world_changed)
        state.on(StateEvent.TASKS_CHANGED, _on_tasks_aff)


def _compute_affective_scores(ws) -> dict[str, float]:
    """Compute basic affective scores from world state structure."""
    scores: dict[str, float] = {}

    if not ws.events:
        return scores

    # Mystery: proportion of information edges with late discovery
    if ws.information_topology:
        late = sum(
            1 for ie in ws.information_topology
            if ie.discovered_at_syuzhet and ie.discovered_at_syuzhet > 0
        )
        scores["mystery"] = min(1.0, late / max(1, len(ws.information_topology)))

    # Tension: fabula/syuzhet displacement
    if ws.events:
        displacements = []
        for evt in ws.events:
            if evt.fabula_time and evt.syuzhet_index:
                displacements.append(abs(evt.fabula_time - evt.syuzhet_index))
        if displacements:
            max_disp = max(displacements) or 1
            scores["narrative_tension"] = min(1.0, sum(displacements) / (len(displacements) * max_disp))

    # Conflict: proportion of negative affinity relationships
    if ws.social_topology:
        negative = sum(1 for r in ws.social_topology if r.affinity < 0)
        scores["conflict"] = min(1.0, negative / max(1, len(ws.social_topology)))

    # Danger: average fear across all relationships
    if ws.social_topology:
        avg_fear = sum(r.fear for r in ws.social_topology) / len(ws.social_topology)
        scores["danger"] = min(1.0, avg_fear)

    # Complexity: causal density (edges / events)
    if ws.events:
        scores["causal_density"] = min(1.0, len(ws.causal_topology) / max(1, len(ws.events) * 2))

    return scores
