"""World tab — multi-view ECharts exploration with click-to-inspect.

View modes: Overview | Social | Spatial | Ego | Temporal | Composition |
            Epistemic | Comparison
Clicking a node updates the left-panel inspector.
Bottom expansion shows filterable topology edge tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import (
    render_comparison_view,
    render_ego_graph,
    render_entity_lifelines,
    render_entity_state_timeline,
    render_epistemic_grid,
    render_event_gantt,
    render_event_timeline,
    render_relationship_heatmap,
    render_social_graph,
    render_spatial_map,
    render_sunburst,
    render_theme_river,
    render_world_graph,
    render_world_treemap,
    with_expand,
)
from shadow_loom_ui.viz_helpers import (
    _set_slider_bounds,
    fabula_time_bounds,
    snapshot_world_at,
    ws_to_causal_rows,
    ws_to_entity_rows,
    ws_to_event_rows,
    ws_to_info_rows,
    ws_to_object_rows,
    ws_to_social_rows,
    ws_to_spatial_rows,
    ws_to_trait_stats_rows,
    ws_to_world_trait_rows,
)

if TYPE_CHECKING:
    from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)

_VIEW_MODES = {
    "overview": "Overview",
    "social": "Social",
    "spatial": "Spatial",
    "ego": "Ego-Graph",
    "temporal": "Temporal",
    "composition": "Composition",
    "epistemic": "Epistemic",
    "comparison": "Comparison",
}


def build_world_tab(state: AppState) -> None:
    """Build the World tab layout."""

    with ui.column().classes("w-full h-full"):
        # ── Toolbar ───────────────────────────────────────────────
        with ui.row().classes("w-full items-center q-pa-sm gap-2"):
            view_mode = ui.toggle(
                _VIEW_MODES,
                value="overview",
            ).props("dense no-caps")

            # Ego-graph entity selector
            ego_select = ui.select(
                options=[],
                label="Focus entities",
                multiple=True,
            ).classes("w-64")
            ego_select.set_visibility(False)

            # Temporal entity selector
            temporal_select = ui.select(
                options=[],
                label="Entity",
            ).classes("w-48")
            temporal_select.set_visibility(False)

            # Epistemic believer filter (multi-select)
            epistemic_select = ui.select(
                options=[],
                label="Believers",
                multiple=True,
            ).classes("w-64").props("use-chips clearable")
            epistemic_select.set_visibility(False)
            epistemic_select.tooltip(
                "Filter which characters' belief panels are shown"
            )

            # Comparison entity multi-select
            compare_select = ui.select(
                options=[],
                label="Compare entities",
                multiple=True,
            ).classes("w-72").props("use-chips clearable")
            compare_select.set_visibility(False)
            compare_select.tooltip(
                "Pick 2–6 characters to compare side-by-side"
            )

            ui.button("Refresh", icon="refresh", on_click=lambda: _refresh()).props(
                "flat dense"
            )

            social_layout = ui.toggle(
                {"force": "Force", "circular": "Circular"},
                value="force",
            ).props("dense no-caps").tooltip("Social graph layout")
            social_layout.set_visibility(False)
            social_metric = ui.select(
                {
                    "affinity": "Affinity  (-1 hate ↔ +1 love)",
                    "fear": "Fear  (0 calm → 1 terrified)",
                    "power_dynamic": "Power dynamic  (-1 subservient ↔ +1 dominant)",
                },
                value="affinity",
                label="Heatmap metric",
            ).classes("w-64").props("dense outlined")
            social_metric.set_visibility(False)
            spatial_animated = ui.checkbox("Animated", value=True).tooltip(
                "Animate location nodes (rippleEffect)"
            )
            spatial_animated.set_visibility(False)

            def _on_mode_change():
                mode = view_mode.value
                ego_select.set_visibility(mode == "ego")
                temporal_select.set_visibility(mode == "temporal")
                epistemic_select.set_visibility(mode == "epistemic")
                compare_select.set_visibility(mode == "comparison")
                social_layout.set_visibility(mode == "social")
                social_metric.set_visibility(mode == "social")
                spatial_animated.set_visibility(mode == "spatial")
                _refresh()

            view_mode.on("update:model-value", _on_mode_change)
            social_layout.on("update:model-value", lambda _e: _refresh())
            social_metric.on("update:model-value", lambda _e: _refresh())
            spatial_animated.on("update:model-value", lambda _e: _refresh())

        # ── Fabula timeline slider ────────────────────────────────
        # Two independent guards (see _sync_slider_widget /
        # _on_slider_change below):
        #   ``local_origin`` — True iff the most recent cursor change
        #     was emitted by this slider, so the FABULA_CURSOR_CHANGED
        #     listener knows to skip writing the slider value back
        #     (which is what was causing the freeze: write-back fired a
        #     second async ``update:model-value``, which the heavy
        #     ``_refresh`` was racing with).
        #   ``rendering`` — True while a chart render is in flight; a
        #     second slider tick during that window queues a single
        #     follow-up render instead of stacking them.
        _slider_state = {
            "local_origin": False,
            "rendering": False,
            "pending": False,
        }

        slider_row = ui.row().classes(
            "w-full items-center q-px-md q-pb-sm gap-3 "
            "bg-slate-50 border-b border-slate-200"
        )
        with slider_row:
            ui.icon("schedule", color="primary")
            ui.label("Fabula time:").classes("text-sm text-slate-600")
            time_label = ui.label("live").classes(
                "text-sm font-mono text-slate-700 w-12"
            )
            time_slider = ui.slider(min=0, max=1, value=0, step=1).props(
                "color=primary label-always dense"
            ).classes("flex-grow")
            live_btn = ui.button(
                "Live", icon="bolt", on_click=lambda: _set_live()
            ).props("flat dense no-caps color=secondary")

        def _set_live():
            # Route through the official setter so every other panel
            # subscribed to FABULA_CURSOR_CHANGED re-renders in lockstep.
            _slider_state["local_origin"] = False
            state.set_fabula_cursor(None)

        def _on_slider_change():
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            if state.fabula_cursor == t:
                return
            # Mark this change as ours so the FABULA_CURSOR_CHANGED
            # listener doesn't bounce the slider value back at us.
            _slider_state["local_origin"] = True
            state.set_fabula_cursor(t)

        # Release-only: Quasar's ``change`` event fires once when the
        # user lets go of the thumb.
        time_slider.on("change", lambda: _on_slider_change())

        # ── Graph container ───────────────────────────────────────
        graph_container = ui.column().classes(
            "w-full flex-grow p-4"
        )

        # ── Raw data tables (expandable) ──────────────────────
        with ui.expansion("Raw Data", icon="table_chart").classes(
            "w-full mx-4 mb-4 bg-white border border-slate-200 rounded-xl"
        ):
            _build_data_tables(state)

        # ── Graph click → inspector ───────────────────────────────
        def _on_graph_click(e):
            data = e.args if isinstance(e.args, dict) else {}
            node_data = data.get("data", {})
            node_id = node_data.get("id") or data.get("name")
            node_type = node_data.get("_sl_node_type")
            if node_id and node_type:
                state.select_node(node_id, node_type)

        # ── Slider widget sync (kept distinct from chart refresh) ──
        def _sync_slider_widget(ws) -> None:
            """Push current world bounds + cursor into the slider widget.

            Only mutates the slider's ``value`` when (a) the change did
            *not* originate from this slider and (b) the value actually
            differs. This avoids the write-back echo loop that froze the
            UI on every drag.
            """
            tmin, tmax = fabula_time_bounds(ws)
            if tmax <= tmin:
                slider_row.set_visibility(False)
                return
            slider_row.set_visibility(True)
            _set_slider_bounds(time_slider, tmin, tmax)
            if state.fabula_cursor is None:
                desired = tmax
                label_text = "live"
            else:
                desired = max(tmin, min(tmax, state.fabula_cursor))
                label_text = f"t={desired}"
            if not _slider_state["local_origin"]:
                try:
                    cur = int(time_slider.value or 0)
                except (TypeError, ValueError):
                    cur = -1
                if cur != desired:
                    time_slider.value = desired
            if time_label.text != label_text:
                time_label.text = label_text
            # Reset for next tick — once we've consumed the local-origin
            # flag, subsequent external cursor changes should sync.
            _slider_state["local_origin"] = False

        # ── Render function ───────────────────────────────────────
        def _refresh(**kw):
            # Coalesce overlapping renders. If a refresh is already in
            # flight, mark "pending" and let it re-run once when done
            # — this prevents the dragged-slider freeze where every
            # tick clear()'d a half-rendered ECharts canvas.
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
                # Run again on next tick so the latest cursor lands.
                ui.timer(0.01, lambda: _refresh(), once=True)

        def _do_refresh(**kw):
            graph_container.clear()
            ws = state.world_state
            if ws is None:
                slider_row.set_visibility(False)
                with graph_container:
                    ui.label("No world model loaded.").classes(
                        "text-sm text-slate-400 italic q-pa-lg"
                    )
                return

            _sync_slider_widget(ws)
            _, tmax = fabula_time_bounds(ws)

            # Snapshot the world model if a cursor is active
            if state.fabula_cursor is not None and tmax > 0:
                try:
                    ws = snapshot_world_at(ws, state.fabula_cursor)
                except Exception:
                    logger.exception("Snapshot failed; falling back to live")

            # Update selectors
            entity_opts = {eid: ent.name for eid, ent in ws.entities.items()}
            ego_select.options = entity_opts
            temporal_select.options = entity_opts
            compare_select.options = entity_opts
            # Epistemic believer options: only entities that hold beliefs.
            believer_opts = {
                eid: ent.name
                for eid, ent in ws.entities.items()
                if ent.beliefs
            }
            epistemic_select.options = believer_opts

            mode = view_mode.value
            with graph_container:
                try:
                    if mode == "overview":
                        with_expand(
                            lambda h: render_world_graph(
                                ws, on_click=_on_graph_click, height=h
                            ),
                            title="World graph \u2014 overview",
                        )
                    elif mode == "social":
                        chosen_metric = social_metric.value or "affinity"
                        _metric_titles = {
                            "affinity": "Affinity heatmap  (–1 hate ↔ +1 love)",
                            "fear": "Fear heatmap  (0 calm → 1 terrified)",
                            "power_dynamic": "Power dynamic  (–1 subservient ↔ +1 dominant)",
                        }
                        with ui.expansion(
                            "Social Graph",
                            icon="hub",
                            value=True,
                        ).classes(
                            "w-full bg-white border border-slate-200 rounded-xl mb-2"
                        ):
                            with_expand(
                                lambda h, lay=social_layout.value or "force": (
                                    render_social_graph(
                                        ws,
                                        on_click=_on_graph_click,
                                        height=h,
                                        layout=lay,
                                    )
                                ),
                                title="Social graph (relationships)",
                                height="420px",
                            )
                        with ui.expansion(
                            _metric_titles.get(chosen_metric, "Relationship heatmap"),
                            icon="grid_on",
                            value=True,
                        ).classes(
                            "w-full bg-white border border-slate-200 rounded-xl mb-2"
                        ):
                            with_expand(
                                lambda h, m=chosen_metric: (
                                    render_relationship_heatmap(
                                        ws, metric=m, height=h,
                                    )
                                ),
                                title=_metric_titles.get(
                                    chosen_metric, "Relationship heatmap",
                                ),
                                height="420px",
                            )
                    elif mode == "spatial":
                        with_expand(
                            lambda h, an=bool(spatial_animated.value): (
                                render_spatial_map(
                                    ws,
                                    on_click=_on_graph_click,
                                    height=h,
                                    animated=an,
                                )
                            ),
                            title="Spatial map",
                        )
                    elif mode == "ego":
                        focus = ego_select.value
                        if focus:
                            ids = focus if isinstance(focus, list) else [focus]
                            with_expand(
                                lambda h, ids=ids: render_ego_graph(
                                    ws, ids,
                                    on_click=_on_graph_click,
                                    height=h,
                                ),
                                title=f"Ego graph \u2014 {', '.join(ids)}",
                            )
                        else:
                            ui.label("Select focus entities above.").classes(
                                "text-sm text-slate-500"
                            )
                    elif mode == "temporal":
                        # Top: Entity Lifelines — status/location/event
                        # ribbons for every character. Replaces the old
                        # single-entity trait line which read as noise
                        # without an entity selected.
                        with_expand(
                            lambda h: render_entity_lifelines(
                                ws, on_click=_on_graph_click, height=h,
                            ),
                            title="Entity lifelines (status, location, events)",
                            height="280px",
                        )
                        eid = temporal_select.value
                        if eid:
                            with_expand(
                                lambda h, eid=eid: (
                                    render_entity_state_timeline(
                                        eid, ws, height=h
                                    )
                                ),
                                title="Entity trait timeline",
                                height="250px",
                            )
                        # ThemeRiver for multi-entity trait flow
                        with_expand(
                            lambda h: render_theme_river(ws, height=h),
                            title="Trait theme river",
                            height="260px",
                        )
                        # Event swim lanes
                        with_expand(
                            lambda h: render_event_gantt(
                                ws, on_click=_on_graph_click, height=h
                            ),
                            title="Event swim-lanes (Gantt)",
                            height="250px",
                        )
                    elif mode == "composition":
                        with ui.row().classes("w-full gap-2 h-full"):
                            with ui.column().classes("flex-grow h-full"):
                                with_expand(
                                    lambda h: render_sunburst(
                                        ws, on_click=_on_graph_click, height=h
                                    ),
                                    title="World composition sunburst",
                                )
                            with ui.column().classes("w-1/2 h-full"):
                                with_expand(
                                    lambda h: render_world_treemap(
                                        ws, on_click=_on_graph_click, height=h
                                    ),
                                    title="World treemap",
                                )
                    elif mode == "epistemic":
                        # Per-character belief panels (one tile per
                        # believer) — reads more naturally than the old
                        # single who-knows-what heatmap because each
                        # character's worldview can be inspected on its
                        # own, with the actual ``perceived_state`` text.
                        sel = epistemic_select.value
                        sel_ids: list[str] | None
                        if isinstance(sel, list) and sel:
                            sel_ids = list(sel)
                        else:
                            sel_ids = None
                        render_epistemic_grid(ws, selected_ids=sel_ids)
                    elif mode == "comparison":
                        # Side-by-side multi-entity comparison: radar
                        # overlay + grouped trait bars + ranked table.
                        # Far clearer than the old parallel-coords +
                        # boxplot pair which just showed spaghetti.
                        sel = compare_select.value
                        if isinstance(sel, list) and sel:
                            chosen = list(sel)[:6]
                        else:
                            chosen = []
                        render_comparison_view(ws, entity_ids=chosen)
                except Exception as e:
                    logger.exception("Graph rendering failed")
                    ui.label(f"Render error: {e}").classes("text-negative")

        # Initial render + subscriptions
        _refresh()

        # Visibility gating so the World tab doesn't rebuild its graph
        # on every cursor scrub from the Causality / Affective tabs.
        _WORLD_PATH = "world"
        _world_dirty = {"on": False}
        _refresh_sync_world = _refresh

        def _world_gated(*args, **kw):
            if not state.is_path_visible(_WORLD_PATH):
                _world_dirty["on"] = True
                return
            _world_dirty["on"] = False
            _refresh_sync_world()

        def _on_world_path(**kw):
            if _world_dirty["on"] and state.is_path_visible(_WORLD_PATH):
                _world_dirty["on"] = False
                _refresh_sync_world()

        ego_select.on("update:model-value", lambda: _refresh_sync_world())
        temporal_select.on("update:model-value", lambda: _refresh_sync_world())
        epistemic_select.on("update:model-value", lambda: _refresh_sync_world())
        compare_select.on("update:model-value", lambda: _refresh_sync_world())
        state.on(StateEvent.WORLD_STATE_CHANGED, _world_gated)
        state.on(StateEvent.ACTIVE_PATH_CHANGED, _on_world_path)
        # Cross-tab cursor sync: when any other panel moves the global
        # fabula cursor (Causality Sankey, Affective Dashboard, Causal
        # Graph snapshot, URL hydration) the World view re-snapshots to
        # match. Without this the slider thumbs would appear stuck.
        state.on(StateEvent.FABULA_CURSOR_CHANGED, _world_gated)


# =====================================================================
# Data tables (nodes + edges + distributions)
# =====================================================================

def _build_data_tables(state: AppState) -> None:
    """Build node, edge, and distribution tables as sub-tabs."""

    with ui.tabs().classes("w-full").props("dense") as data_tabs:
        ui.tab("entities", label="Entities", icon="person")
        ui.tab("events", label="Events", icon="bolt")
        ui.tab("objects", label="Objects", icon="inventory_2")
        ui.tab("world_traits", label="World Traits", icon="public")
        ui.tab("trait_stats", label="Trait Stats", icon="bar_chart")
        ui.tab("causal", label="Causal Edges", icon="trending_up")
        ui.tab("spatial", label="Spatial Edges", icon="map")
        ui.tab("social", label="Social Edges", icon="people")
        ui.tab("info", label="Info Edges", icon="mail")

    _table_props = "dense flat bordered"

    with ui.tab_panels(data_tabs, value="entities").classes("w-full"):
        with ui.tab_panel("entities"):
            entity_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "status", "label": "Status", "field": "status", "sortable": True},
                    {"name": "location", "label": "Location", "field": "location", "sortable": True},
                    {"name": "traits", "label": "#Traits", "field": "traits", "sortable": True},
                    {"name": "beliefs", "label": "#Beliefs", "field": "beliefs", "sortable": True},
                    {"name": "constants", "label": "Constants", "field": "constants"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

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
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("objects"):
            object_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "location", "label": "Location", "field": "location", "sortable": True},
                    {"name": "owner", "label": "Owner", "field": "owner", "sortable": True},
                    {"name": "affordances", "label": "Affordances", "field": "affordances"},
                    {"name": "properties", "label": "Properties", "field": "properties"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("world_traits"):
            world_trait_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "magnitude", "label": "Magnitude", "field": "magnitude", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                    {"name": "description", "label": "Description", "field": "description"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("trait_stats"):
            trait_stats_table = ui.table(
                columns=[
                    {"name": "trait", "label": "Trait", "field": "trait", "sortable": True},
                    {"name": "min", "label": "Min", "field": "min", "sortable": True},
                    {"name": "q1", "label": "Q1", "field": "q1", "sortable": True},
                    {"name": "median", "label": "Median", "field": "median", "sortable": True},
                    {"name": "q3", "label": "Q3", "field": "q3", "sortable": True},
                    {"name": "max", "label": "Max", "field": "max", "sortable": True},
                    {"name": "outliers", "label": "Outliers", "field": "outliers", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("causal"):
            causal_table = ui.table(
                columns=[
                    {"name": "source", "label": "Source", "field": "source", "sortable": True},
                    {"name": "target", "label": "Target", "field": "target", "sortable": True},
                    {"name": "type", "label": "Type", "field": "type", "sortable": True},
                    {"name": "mechanism", "label": "Mechanism", "field": "mechanism"},
                    {"name": "force", "label": "Force", "field": "force", "sortable": True},
                    {"name": "evidence", "label": "Evidence", "field": "evidence", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("spatial"):
            spatial_table = ui.table(
                columns=[
                    {"name": "source", "label": "From", "field": "source", "sortable": True},
                    {"name": "target", "label": "To", "field": "target", "sortable": True},
                    {"name": "locked", "label": "Locked", "field": "locked", "sortable": True},
                    {"name": "barrier", "label": "Barrier", "field": "barrier"},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("social"):
            social_table = ui.table(
                columns=[
                    {"name": "source", "label": "From", "field": "source", "sortable": True},
                    {"name": "target", "label": "To", "field": "target", "sortable": True},
                    {"name": "affinity", "label": "Affinity", "field": "affinity", "sortable": True},
                    {"name": "fear", "label": "Fear", "field": "fear", "sortable": True},
                    {"name": "power", "label": "Power", "field": "power", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("info"):
            info_table = ui.table(
                columns=[
                    {"name": "kind", "label": "Kind", "field": "kind", "sortable": True},
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "participants", "label": "Participants", "field": "participants"},
                    {"name": "medium", "label": "Medium", "field": "medium", "sortable": True},
                    {"name": "directionality", "label": "Directionality", "field": "directionality", "sortable": True},
                    {
                        "name": "min_intelligibility",
                        "label": "Min intel.",
                        "field": "min_intelligibility",
                        "sortable": True,
                    },
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

    def _refresh_tables(**kw):
        ws = state.world_state
        if ws is None:
            return
        entity_table.rows = ws_to_entity_rows(ws)
        event_table.rows = ws_to_event_rows(ws)
        object_table.rows = ws_to_object_rows(ws)
        world_trait_table.rows = ws_to_world_trait_rows(ws)
        trait_stats_table.rows = ws_to_trait_stats_rows(ws)
        causal_table.rows = ws_to_causal_rows(ws)
        spatial_table.rows = ws_to_spatial_rows(ws)
        social_table.rows = ws_to_social_rows(ws)
        info_table.rows = ws_to_info_rows(ws)

    _refresh_tables()
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_tables)
