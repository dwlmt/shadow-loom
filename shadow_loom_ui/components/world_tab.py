# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""World tab — multi-view ECharts exploration with click-to-inspect.

View modes: Overview | Spatial | Information | Ego | Temporal |
            Composition | World State | Comparison
Clicking a node updates the left-panel inspector.
Bottom expansion shows filterable topology edge tables.

Note: the Beliefs / Concerns / Propositions / Relationships views
live on the dedicated **Social** tab.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import (
    render_causal_sankey,
    render_comparison_view,
    render_ego_graph,
    render_entity_lifelines,
    render_entity_state_timeline,
    render_event_gantt,
    render_event_type_bar,
    render_location_occupancy_bar,
    render_object_ownership_bar,
    render_population_summary,
    render_spatial_map,
    render_status_donut,
    render_sunburst,
    render_theme_river,
    render_world_graph,
    render_world_state_grid,
    render_world_trait_bars,
    render_world_treemap,
    with_expand,
)
from shadow_loom_ui.viz_helpers import (
    _set_slider_bounds,
    axis_bounds,
    event_axis_value,
    resolve_cursor,
    snapshot_world_at,
    ws_to_causal_rows,
    ws_to_entity_rows,
    ws_to_event_rows,
    ws_to_channel_rows,
    ws_to_utterance_rows,
    ws_to_object_rows,
    ws_to_social_rows,
    ws_to_spatial_rows,
    ws_to_trait_stats_rows,
    ws_to_world_trait_rows,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_VIEW_MODES = {
    "overview": "Overview",
    "spatial": "Spatial",
    "information": "Information",
    "ego": "Ego-Graph",
    "temporal": "Temporal",
    "composition": "Composition",
    "world_state": "World State",
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
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="World — state-at-time inspector",
                body_md=(
                    "The world model rendered at a chosen **fabula"
                    " time** (in-story chronology). Slide the time"
                    " cursor to watch the world evolve.\n\n"
                    "### View modes\n"
                    "- **Overview** — high-level summary: counts of"
                    " entities/locations/events, status distribution,"
                    " current-location heatmap.\n"
                    "- **Spatial** — location graph (rooms / regions)"
                    " with the doors/paths between them, plus current"
                    " entity positions.\n"
                    "- **Information** — standing communication"
                    " channels (telephones, mind-links, classified"
                    " pipelines) and which entities can transmit /"
                    " overhear / are deaf to them.\n"
                    "- **Ego-Graph** — the world filtered to one or"
                    " more focus entities: just what *they* can"
                    " plausibly perceive, hear, or remember at the"
                    " current anchor.\n"
                    "- **Temporal** — a single entity's full trajectory"
                    " (status, location, traits, beliefs) across"
                    " fabula time.\n"
                    "- **Composition** — trait-vector composition"
                    " breakdowns (what makes Macbeth *Macbeth*).\n"
                    "- **World State** — per-world-trait"
                    " **snapshot cards** at the current fabula"
                    " cursor (magnitude, inertia, affected"
                    " domains) with a small inline sparkline of the"
                    " trajectory. Drag the time cursor to watch each"
                    " card update in lockstep.\n"
                    "- **Comparison** — side-by-side trait /"
                    " relationship table for 2–6 picked entities.\n\n"
                    "### Looking for beliefs / concerns / propositions"
                    " / relationships?\n"
                    "They live on the **Social** tab now — unified"
                    " network graph + per-character cards + trait"
                    " trajectories, all time-sliced by the same"
                    " fabula cursor.\n\n"
                    "### Reading the diagrams\n"
                    "- Node colour usually encodes **type** (entity,"
                    " location, object) or **status** (alive, dead,"
                    " injured).\n"
                    "- Edge thickness usually encodes **strength** of"
                    " the underlying relationship / connection.\n"
                    "- Hover any node or edge for a tooltip with the"
                    " raw payload.\n\n"
                    "### Time cursor\n"
                    "Most views are **time-sliced**: events with"
                    " `fabula_time > cursor` are hidden, social /"
                    " spatial / channel edges that hadn't been"
                    " established yet are pruned, and entity"
                    " snapshots reflect the most recent change at or"
                    " before the cursor. Drag the cursor to scrub."
                ),
                tooltip="What is this tab?",
            )
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

            # World-state world-trait filter (multi-select).
            world_state_select = ui.select(
                options=[],
                label="World traits",
                multiple=True,
            ).classes("w-64").props("use-chips clearable")
            world_state_select.set_visibility(False)
            world_state_select.tooltip(
                "Filter which world-trait timeline cards are shown"
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

            spatial_animated = ui.checkbox("Animated", value=True).tooltip(
                "Animate location nodes (rippleEffect)"
            )
            spatial_animated.set_visibility(False)

            def _on_mode_change():
                mode = view_mode.value
                ego_select.set_visibility(mode == "ego")
                temporal_select.set_visibility(mode == "temporal")
                world_state_select.set_visibility(mode == "world_state")
                compare_select.set_visibility(mode == "comparison")
                spatial_animated.set_visibility(mode == "spatial")
                _refresh()

            view_mode.on("update:model-value", _on_mode_change)
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
            # Sorted list of (fabula_time, short_label) used by the
            # hover-tooltip lookup so the user knows where in the
            # plot the slider is pointing as they drag. Refreshed
            # by ``_sync_slider_widget`` whenever the world model
            # changes.
            "event_index": [],
        }

        slider_row = ui.row().classes(
            "w-full items-center q-px-md q-pb-sm gap-3 "
            "bg-slate-50 border-b border-slate-200"
        )
        with slider_row:
            ui.icon("schedule", color="primary")
            time_axis_label = ui.label(
                "Syuzhet idx:" if state.time_axis == "syuzhet"
                else "Fabula time:"
            ).classes("text-sm text-slate-600")
            time_label = ui.label("live").classes(
                "text-sm font-mono text-slate-700 w-12"
            )
            time_slider = ui.slider(min=0, max=1, value=0, step=1).props(
                "color=primary label-always dense"
            ).classes("flex-grow")
            time_slider.tooltip(
                "Drag to scrub through fabula time. The thumb tooltip "
                "shows the nearest plot event so you know where you "
                "are in the story."
            )
            live_btn = ui.button(
                "Live", icon="bolt", on_click=lambda: _set_live()
            ).props("flat dense no-caps color=secondary")

        def _set_live():
            # Route through the official setter so every other panel
            # subscribed to FABULA_CURSOR_CHANGED / SYUZHET_CURSOR_CHANGED
            # re-renders in lockstep. Dispatch by active axis so a syuzhet
            # axis flip back to live clears the syuzhet cursor (not the
            # fabula one), keeping Genette's two clocks decoupled.
            _slider_state["local_origin"] = False
            state.set_active_cursor(None)

        def _on_slider_change():
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            if state.active_cursor == t:
                return
            # Mark this change as ours so the cursor-change listener
            # doesn't bounce the slider value back at us.
            _slider_state["local_origin"] = True
            state.set_active_cursor(t)

        def _nearest_event_label(t: int) -> str:
            """Return ``"t=N \u2014 nearest event description"`` for the slider tooltip.

            Picks the event whose axis value is closest to ``t``;
            on ties, prefers the one at-or-before ``t`` so the label
            tells the user *what has just happened*, which matches how
            the rest of the world view interprets a cursor (everything
            up to and including ``t`` is considered current).
            """
            idx = _slider_state["event_index"]
            prefix = "s" if state.time_axis == "syuzhet" else "t"
            if not idx:
                return f"{prefix}={t}"
            # Min by (abs distance, prefer at-or-before via sign tiebreak).
            best = min(idx, key=lambda et: (abs(et[0] - t), 0 if et[0] <= t else 1))
            label = best[1]
            if len(label) > 80:
                label = label[:77] + "\u2026"
            return f"{prefix}={t} \u2014 {label}"

        def _push_label_value(text: str) -> None:
            """Update the slider's Quasar ``label-value`` prop in place."""
            # Escape only the characters Quasar's prop parser would
            # choke on; keep the visible text human-readable.
            safe = text.replace('"', "'").replace("\n", " ")
            time_slider.props(f'label-value="{safe}"')

        def _on_slider_input():
            """Live-drag handler: refresh the thumb tooltip without
            firing the heavy ``_refresh`` (which fires on release)."""
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            _push_label_value(_nearest_event_label(t))
            prefix = "s" if state.time_axis == "syuzhet" else "t"
            label_text = f"{prefix}={t}"
            if time_label.text != label_text:
                time_label.text = label_text

        # Live drag: ``update:model-value`` ticks for every pixel of
        # movement. Cheap (just rewrites the tooltip text) so we don't
        # need the rendering coalescer here.
        time_slider.on("update:model-value", lambda _e=None: _on_slider_input())

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
            axis = state.time_axis
            tmin, tmax = axis_bounds(ws, axis)
            if tmax <= tmin:
                slider_row.set_visibility(False)
                return
            slider_row.set_visibility(True)
            _set_slider_bounds(time_slider, tmin, tmax)
            # Refresh the (axis_value, label) lookup the live-drag
            # tooltip uses. Sorted by axis value so a future caller
            # that wants nearest-event-by-position can binary-search.
            event_index = []
            for evt in (getattr(ws, "events", None) or []):
                ft = event_axis_value(evt, axis)
                if ft is None:
                    continue
                desc = (
                    getattr(evt, "description", None)
                    or getattr(evt, "content", None)
                    or getattr(evt, "id", None)
                    or ""
                ).strip()
                if not desc:
                    continue
                event_index.append((int(ft), desc))
            event_index.sort(key=lambda x: x[0])
            _slider_state["event_index"] = event_index
            cur_prefix = "s" if axis == "syuzhet" else "t"
            cursor_value = state.active_cursor
            if cursor_value is None:
                desired = tmax
                label_text = "live"
            else:
                desired = max(tmin, min(tmax, cursor_value))
                label_text = f"{cur_prefix}={desired}"
            if not _slider_state["local_origin"]:
                try:
                    cur = int(time_slider.value or 0)
                except (TypeError, ValueError):
                    cur = -1
                if cur != desired:
                    time_slider.value = desired
            if time_label.text != label_text:
                time_label.text = label_text
            # Seed the thumb tooltip so the nearest-event hint is
            # visible the moment the slider appears (label-always).
            _push_label_value(_nearest_event_label(int(desired)))
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
            axis = state.time_axis
            _, tmax = axis_bounds(ws, axis)

            # Snapshot the world model if a cursor is active. The
            # cursor value is on the active axis; resolve it back to
            # a fabula time for the entity/state replay.
            cursor_value = state.active_cursor
            if cursor_value is not None and tmax > 0:
                try:
                    eff = resolve_cursor(ws, axis, cursor_value)
                    if eff is not None:
                        ws = snapshot_world_at(ws, eff)
                except Exception:
                    logger.exception("Snapshot failed; falling back to live")

            # Update selectors
            entity_opts = {eid: ent.name for eid, ent in ws.entities.items()}
            ego_select.options = entity_opts
            temporal_select.options = entity_opts
            compare_select.options = entity_opts
            # World-state world-trait options.
            world_trait_opts = {
                wid: wt.name for wid, wt in ws.world_traits.items()
            }
            world_state_select.options = world_trait_opts

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
                    elif mode == "information":
                        # Channels + utterance flow as a Sankey: who
                        # transmits what, through which channel, to
                        # whom. Complements the info topology table
                        # (added in F11) with a graph view.
                        with_expand(
                            lambda h: render_causal_sankey(
                                ws,
                                on_click=_on_graph_click,
                                height=h,
                                aspect="information",
                            ),
                            title="Information flow (channels & utterances)",
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
                        # Six focused mini-charts beat the old sunburst+
                        # treemap pair, which crammed 4 hierarchy levels
                        # into illegible rim labels and overflowing tiles.
                        # Each chart now answers ONE composition
                        # question ("who's where?", "what's the status
                        # mix?", "how strong are world traits?" …).
                        with ui.column().classes("w-full gap-3"):
                            # Row 0 — headline counts
                            render_population_summary(ws)
                            # Row 1 — status donut + location occupancy
                            with ui.row().classes("w-full gap-3 items-stretch"):
                                with ui.column().classes(
                                    "flex-grow basis-0 min-w-72 bg-white "
                                    "border border-slate-200 rounded-xl "
                                    "shadow-sm p-3 gap-1"
                                ):
                                    ui.label("Entity status mix").classes(
                                        "text-xs uppercase tracking-wide "
                                        "text-slate-500"
                                    )
                                    render_status_donut(ws)
                                with ui.column().classes(
                                    "flex-grow basis-0 min-w-96 bg-white "
                                    "border border-slate-200 rounded-xl "
                                    "shadow-sm p-3 gap-1"
                                ):
                                    ui.label("Location occupancy").classes(
                                        "text-xs uppercase tracking-wide "
                                        "text-slate-500"
                                    )
                                    render_location_occupancy_bar(
                                        ws, on_click=_on_graph_click,
                                    )
                            # Row 2 — event types + object ownership
                            with ui.row().classes("w-full gap-3 items-stretch"):
                                with ui.column().classes(
                                    "flex-grow basis-0 min-w-72 bg-white "
                                    "border border-slate-200 rounded-xl "
                                    "shadow-sm p-3 gap-1"
                                ):
                                    ui.label("Event types").classes(
                                        "text-xs uppercase tracking-wide "
                                        "text-slate-500"
                                    )
                                    render_event_type_bar(ws)
                                with ui.column().classes(
                                    "flex-grow basis-0 min-w-72 bg-white "
                                    "border border-slate-200 rounded-xl "
                                    "shadow-sm p-3 gap-1"
                                ):
                                    ui.label("Object ownership").classes(
                                        "text-xs uppercase tracking-wide "
                                        "text-slate-500"
                                    )
                                    render_object_ownership_bar(
                                        ws, on_click=_on_graph_click,
                                    )
                            # Row 3 — world-trait magnitude vs inertia
                            with ui.column().classes(
                                "w-full bg-white border border-slate-200 "
                                "rounded-xl shadow-sm p-3 gap-1"
                            ):
                                ui.label(
                                    "Global (world) traits — magnitude vs inertia"
                                ).classes(
                                    "text-xs uppercase tracking-wide "
                                    "text-slate-500"
                                )
                                render_world_trait_bars(ws)
                            # Row 4 — legacy hierarchy view (collapsed)
                            with ui.expansion(
                                "Hierarchy view (location → contents)",
                                icon="account_tree",
                            ).classes("w-full").props("dense"):
                                with ui.row().classes("w-full gap-2"):
                                    with ui.column().classes("flex-grow basis-0"):
                                        with_expand(
                                            lambda h: render_sunburst(
                                                ws, on_click=_on_graph_click,
                                                height=h,
                                            ),
                                            title="World composition sunburst",
                                        )
                                    with ui.column().classes("flex-grow basis-0"):
                                        with_expand(
                                            lambda h: render_world_treemap(
                                                ws, on_click=_on_graph_click,
                                                height=h,
                                            ),
                                            title="World treemap",
                                        )
                    elif mode == "epistemic":
                        # Migrated to Social tab. Kept as a no-op so
                        # any stale URL hash referring to this view
                        # mode lands on a friendly hint instead of a
                        # KeyError.
                        ui.label(
                            "Belief panels moved to the Social tab."
                        ).classes("text-sm text-slate-500 italic")
                    elif mode == "world_state":
                        # Per-world-trait timeline panels — mirrors the
                        # Character Beliefs grid but for global
                        # ``GlobalTrait`` magnitudes evolving across
                        # fabula time.
                        sel = world_state_select.value
                        wt_ids: list[str] | None
                        if isinstance(sel, list) and sel:
                            wt_ids = list(sel)
                        else:
                            wt_ids = None
                        render_world_state_grid(ws, selected_ids=wt_ids)
                    elif mode == "relationships":
                        ui.label(
                            "Relationship cards moved to the Social tab."
                        ).classes("text-sm text-slate-500 italic")
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
        compare_select.on("update:model-value", lambda: _refresh_sync_world())
        def _on_world_axis_change(**_kw):
            time_axis_label.text = (
                "Syuzhet idx:" if state.time_axis == "syuzhet"
                else "Fabula time:"
            )
            _world_gated()

        state.on(StateEvent.WORLD_STATE_CHANGED, _world_gated)
        state.on(StateEvent.ACTIVE_PATH_CHANGED, _on_world_path)
        # Cross-tab cursor sync: when any other panel moves the global
        # fabula cursor (Causality Sankey, Affective Dashboard, Causal
        # Graph snapshot, URL hydration) the World view re-snapshots to
        # match. Without this the slider thumbs would appear stuck.
        state.on(StateEvent.FABULA_CURSOR_CHANGED, _world_gated)
        state.on(StateEvent.SYUZHET_CURSOR_CHANGED, _world_gated)
        state.on(StateEvent.TIME_AXIS_CHANGED, _on_world_axis_change)


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
        from shadow_loom_ui.components._subtab_help import subtab_help
        with ui.tab_panel("entities"):
            subtab_help("world.entities")
            entity_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "status", "label": "Status", "field": "status", "sortable": True},
                    {"name": "location", "label": "Location", "field": "location", "sortable": True},
                    {"name": "traits", "label": "#Traits", "field": "traits", "sortable": True},
                    {"name": "beliefs", "label": "#Beliefs", "field": "beliefs", "sortable": True},
                    {"name": "snapshots", "label": "#Snaps", "field": "snapshots", "sortable": True},
                    {"name": "constants", "label": "Constants", "field": "constants"},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("events"):
            subtab_help("world.events")
            # Hide-superseded toggle: when True, the supersession-aware
            # transformer filters out events with a non-null
            # ``superseded_by_event_id``. Stored on the closure so the
            # refresh handler can re-apply it.
            hide_superseded_state = {"v": False}
            with ui.row().classes("w-full items-center gap-2 mb-1"):
                _hide_sw = ui.switch(
                    "Hide superseded events",
                    value=False,
                ).props("dense").classes("text-xs")
            event_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "fabula_time", "label": "Fabula t", "field": "fabula_time", "sortable": True},
                    {"name": "syuzhet_index", "label": "Syuzhet", "field": "syuzhet_index", "sortable": True},
                    {"name": "type", "label": "Type", "field": "type", "sortable": True},
                    {"name": "actors", "label": "Actors", "field": "actors"},
                    {"name": "targets", "label": "Targets", "field": "targets"},
                    {"name": "description", "label": "Description", "field": "description"},
                    {"name": "superseded_by_event_id", "label": "Superseded by", "field": "superseded_by_event_id", "sortable": True},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("objects"):
            subtab_help("world.objects")
            object_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "location", "label": "Location", "field": "location", "sortable": True},
                    {"name": "owner", "label": "Owner", "field": "owner", "sortable": True},
                    {"name": "affordances", "label": "Affordances", "field": "affordances"},
                    {"name": "properties", "label": "Properties", "field": "properties"},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("world_traits"):
            subtab_help("world.world_traits")
            world_trait_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "category", "label": "Category", "field": "category", "sortable": True},
                    {"name": "magnitude", "label": "Magnitude", "field": "magnitude", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                    {"name": "affected_domains", "label": "Domains", "field": "affected_domains"},
                    {"name": "snapshots", "label": "#Snaps", "field": "snapshots", "sortable": True},
                    {"name": "description", "label": "Description", "field": "description"},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("trait_stats"):
            subtab_help("world.trait_stats")
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
            subtab_help("world.causal")
            causal_table = ui.table(
                columns=[
                    {"name": "source", "label": "Source", "field": "source", "sortable": True},
                    {"name": "target", "label": "Target", "field": "target", "sortable": True},
                    {"name": "type", "label": "Type", "field": "type", "sortable": True},
                    {"name": "mechanism", "label": "Mechanism", "field": "mechanism"},
                    {"name": "force", "label": "Force", "field": "force", "sortable": True},
                    {"name": "evidence", "label": "Evidence", "field": "evidence", "sortable": True},
                    {"name": "delay", "label": "Delay", "field": "delay", "sortable": True},
                    {"name": "fabula_time", "label": "Fabula t", "field": "fabula_time", "sortable": True},
                    {"name": "trait_target", "label": "Trait", "field": "trait_target", "sortable": True},
                    {"name": "trait_delta", "label": "Δ", "field": "trait_delta", "sortable": True},
                    {"name": "rel_counterpart", "label": "Counterpart", "field": "rel_counterpart"},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("spatial"):
            subtab_help("world.spatial")
            spatial_table = ui.table(
                columns=[
                    {"name": "source", "label": "From", "field": "source", "sortable": True},
                    {"name": "target", "label": "To", "field": "target", "sortable": True},
                    {"name": "locked", "label": "Locked", "field": "locked", "sortable": True},
                    {"name": "barrier", "label": "Barrier", "field": "barrier"},
                    {"name": "established_at_fabula", "label": "Established", "field": "established_at_fabula", "sortable": True},
                    {"name": "destroyed_at_fabula", "label": "Destroyed", "field": "destroyed_at_fabula", "sortable": True},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("social"):
            subtab_help("world.social")
            social_table = ui.table(
                columns=[
                    {"name": "source", "label": "From", "field": "source", "sortable": True},
                    {"name": "target", "label": "To", "field": "target", "sortable": True},
                    {"name": "affinity", "label": "Affinity", "field": "affinity", "sortable": True},
                    {"name": "fear", "label": "Fear", "field": "fear", "sortable": True},
                    {"name": "power", "label": "Power", "field": "power", "sortable": True},
                    {"name": "inertia", "label": "Inertia", "field": "inertia", "sortable": True},
                    {"name": "evidence", "label": "Evidence", "field": "evidence", "sortable": True},
                    {"name": "axes_observed", "label": "#Axes", "field": "axes_observed", "sortable": True},
                    {"name": "last_updated_fabula", "label": "Updated t", "field": "last_updated_fabula", "sortable": True},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

        with ui.tab_panel("info"):
            subtab_help("world.info")
            ui.label("Channels (standing capabilities)").classes(
                "text-xs uppercase tracking-wide text-slate-500 mt-1"
            )
            channel_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "name", "label": "Name", "field": "name", "sortable": True},
                    {"name": "medium", "label": "Medium", "field": "medium", "sortable": True},
                    {"name": "directionality", "label": "Directionality", "field": "directionality", "sortable": True},
                    {"name": "participants", "label": "Participants", "field": "participants"},
                    {"name": "min_intelligibility", "label": "Min intel.", "field": "min_intelligibility", "sortable": True},
                    {"name": "established_at_fabula", "label": "Established", "field": "established_at_fabula", "sortable": True},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

            ui.label("Utterances (discrete messages)").classes(
                "text-xs uppercase tracking-wide text-slate-500 mt-3"
            )
            utterance_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "sortable": True},
                    {"name": "fabula_time", "label": "t", "field": "fabula_time", "sortable": True},
                    {"name": "syuzhet_index", "label": "syu", "field": "syuzhet_index", "sortable": True},
                    {"name": "speaker", "label": "Speaker", "field": "speaker", "sortable": True},
                    {"name": "addressees", "label": "Addressees", "field": "addressees"},
                    {"name": "via_channel_id", "label": "Channel", "field": "via_channel_id", "sortable": True},
                    {"name": "truth_value", "label": "Truth", "field": "truth_value", "sortable": True},
                    {"name": "content", "label": "Content", "field": "content"},
                    {"name": "world_id", "label": "Branch", "field": "world_id", "sortable": True},
                ],
                rows=[],
                pagination={"rowsPerPage": 10},
            ).props(_table_props).classes("w-full")

    def _refresh_tables(**kw):
        ws = state.world_state
        if ws is None:
            return
        entity_table.rows = ws_to_entity_rows(ws)
        evt_rows = ws_to_event_rows(ws)
        if hide_superseded_state.get("v"):
            evt_rows = [r for r in evt_rows if not r.get("superseded")]
        event_table.rows = evt_rows
        object_table.rows = ws_to_object_rows(ws)
        world_trait_table.rows = ws_to_world_trait_rows(ws)
        trait_stats_table.rows = ws_to_trait_stats_rows(ws)
        causal_table.rows = ws_to_causal_rows(ws)
        spatial_table.rows = ws_to_spatial_rows(ws)
        social_table.rows = ws_to_social_rows(ws)
        channel_table.rows = ws_to_channel_rows(ws)
        utterance_table.rows = ws_to_utterance_rows(ws)

    def _on_hide_superseded(e):
        hide_superseded_state["v"] = bool(e.value)
        _refresh_tables()

    _hide_sw.on_value_change(_on_hide_superseded)

    _refresh_tables()
    state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_tables)
