# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Causality panels — causal topology + ingestion warnings + affective dashboard.

The top-level Causality tab has been removed. The surviving public
entry points are now mounted by other tabs:

  * :func:`_build_causal_topology` → mounted from the Reasoning tab
    (after the events panel).
  * :func:`_build_ingestion_warnings_panel` → mounted from the Edit tab.
  * :func:`build_affective_dashboard` → mounted from the Affective tab.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.state import AppState, StateEvent
from shadow_loom_ui.viz import (
    render_causal_force_graph,
    render_causal_sankey,
    render_chart_skeleton,
    render_emotional_gauges,
    render_emotional_gauges_graded,
    render_empty_state,
    open_explain_dialog,
    with_expand,
)
from shadow_loom_ui.components._subtab_help import subtab_help

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


def _events_at(ws, t: int, *, axis: str = "fabula", limit: int = 3) -> str:
    """Return a short human-readable summary of events at ``t``.

    Used by the timeline sliders so dragging the cursor reveals
    *which event* is sitting at the current fabula time / syuzhet
    index. Picks events whose axis value equals ``t`` exactly; if
    none match, falls back to the closest event by absolute distance
    so the label is never empty mid-drag.

    Returns a string like ``"E_005: Mary confronts John (+1 more)"``
    or ``"\u2014"`` when there are no events at all.
    """
    if ws is None or not getattr(ws, "events", None):
        return "\u2014"

    def _axis_val(e):
        return e.fabula_time if axis == "fabula" else e.syuzhet_index

    candidates = [e for e in ws.events if _axis_val(e) is not None]
    if not candidates:
        return "\u2014"

    exact = [e for e in candidates if int(_axis_val(e)) == int(t)]
    if exact:
        # Stable order by id then syuzhet for deterministic display.
        exact.sort(key=lambda e: (e.syuzhet_index or 0, e.id))
        chosen = exact[:limit]
        extra = len(exact) - len(chosen)
    else:
        nearest = min(candidates, key=lambda e: abs(int(_axis_val(e)) - int(t)))
        chosen = [nearest]
        extra = 0

    parts = []
    for e in chosen:
        desc = (e.description or "").strip().replace("\n", " ")
        if len(desc) > 60:
            desc = desc[:57] + "\u2026"
        parts.append(f"{e.id}: {desc}" if desc else e.id)
    suffix = "" if exact else "  (nearest)"
    if extra > 0:
        suffix = f"  (+{extra} more)" + suffix
    return "  \u2022  ".join(parts) + suffix


# =====================================================================
# 1. Causal Topology
# =====================================================================

def _build_causal_topology(state: AppState) -> None:
    """Sankey diagram + causal force graph with aspect/force/time filters."""

    from shadow_loom_ui.viz_helpers import (
        SANKEY_ASPECTS,
        _set_slider_bounds,
        axis_bounds,
        fabula_time_bounds,
        resolve_cursor,
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
                options={
                    "force": "Force",
                    "timeline": "Timeline",
                    "circular": "Circular",
                    "cartesian": "Cartesian",
                },
                value="timeline",
            ).props("dense outlined options-dense").classes("min-w-32")
            force_layout.tooltip(
                "How to lay out the causal force graph. Timeline "
                "anchors event nodes to fabula_time on the x-axis "
                "while letting the force solver spread them "
                "vertically (best default for cascade reading); "
                "Cartesian additionally pins the y-axis to syuzhet "
                "index; Circular places nodes on a ring; Force is "
                "an unanchored physics layout."
            )

        # ── Focus / drill-down row ────────────────────────────────
        # Focus state lives on this closure (not AppState) so swapping
        # Causality \u2194 other tabs doesn't reset it; cleared via the
        # button or by clicking the same node twice.
        focus_state: dict = {"id": None, "hops": 2}
        focus_row = ui.row().classes(
            "w-full items-center gap-3 px-3 py-1 bg-amber-50 "
            "border border-amber-200 rounded-md flex-wrap"
        )
        with focus_row:
            ui.icon("center_focus_strong", color="amber-9")
            focus_label = ui.label(
                "No focus \u2014 click any node to drill into its "
                "ancestor / descendant chain."
            ).classes("text-xs text-amber-900")
            ui.space()
            ui.label("Hops:").classes("text-xs text-slate-600")
            hops_select = ui.select(
                options={1: "1", 2: "2", 3: "3", 4: "4"}, value=2,
            ).props("dense outlined options-dense").classes("w-16")
            hops_select.tooltip(
                "How many causal hops in each direction to keep "
                "around the focus node."
            )

            def _clear_focus():
                focus_state["id"] = None
                focus_label.text = (
                    "No focus \u2014 click any node to drill into "
                    "its ancestor / descendant chain."
                )
                _refresh()

            ui.button(
                "Clear", icon="close", on_click=_clear_focus,
            ).props("flat dense no-caps color=amber-9")
        focus_row.set_visibility(False)  # only shows once a graph is loaded

        # ── Modality legend (causal palette) ──────────────────────
        from shadow_loom_ui.viz import render_modality_legend
        render_modality_legend()

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

            time_slider_label = ui.label("Up to fabula t:").classes(
                "text-sm text-slate-600 ml-4"
            )
            time_slider = ui.slider(min=0, max=1, step=1, value=1).props(
                "color=secondary label-always dense"
            ).classes("flex-grow min-w-32")
            time_slider.tooltip(
                "Only show edges revealed at or before this cursor "
                "(syuzhet: events with index ≤ s; fabula: events with "
                "fabula_time ≤ t). Bound to the global time-axis picker."
            )

            def _reset_time():
                # Drive through the global cursor so other tabs follow.
                _slider_state["local_origin_t"] = False
                if state.time_axis == "syuzhet":
                    state.set_syuzhet_cursor(None)
                else:
                    state.set_fabula_cursor(None)

            ui.button("Full span", icon="restart_alt", on_click=_reset_time) \
                .props("flat dense no-caps color=secondary")

            # Full-width event-context line that mirrors what the
            # affective-tab slider shows: as the user drags the
            # ``Up to fabula t`` slider this label updates with the
            # event description sitting at that fabula time so the
            # scrub becomes a content-driven jump rather than a
            # blind number tweak.
            topo_event_at_cursor = ui.label("\u2014").classes(
                "text-xs text-slate-500 italic basis-full pl-1 pt-1 "
                "truncate"
            )
            topo_event_at_cursor.tooltip(
                "Event at the current fabula cursor (or nearest if "
                "none lands here)"
            )

        def _update_topo_event_label():
            ws = state.world_state
            if ws is None:
                topo_event_at_cursor.text = "\u2014"
                return
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            topo_event_at_cursor.text = _events_at(
                ws, t, axis=state.time_axis or "fabula",
            )

        time_slider.on(
            "update:model-value", lambda _e: _update_topo_event_label(),
        )

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
            axis = state.time_axis or "fabula"
            tmin, tmax = axis_bounds(ws, axis)
            time_slider_label.text = (
                "Up to syuzhet s:" if axis == "syuzhet" else "Up to fabula t:"
            )
            if tmax <= tmin:
                return
            # Quasar's <q-slider> requires numeric min/max props. Passing
            # the props via the string parser (``time_slider.props(...)``)
            # stores them as *strings*, which the slider silently rejects
            # — the thumb appears to render but won't drag past the
            # original construction-time bounds. Write numeric values
            # straight into the props dict instead.
            _set_slider_bounds(time_slider, tmin, tmax)
            cur = (
                state.syuzhet_cursor if axis == "syuzhet"
                else state.fabula_cursor
            )
            desired = tmax if cur is None else max(tmin, min(tmax, cur))
            if not _slider_state["local_origin_t"]:
                try:
                    cur_w = int(time_slider.value or 0)
                except (TypeError, ValueError):
                    cur_w = -1
                if cur_w != desired:
                    time_slider.value = desired
            _slider_state["local_origin_t"] = False
            # Mirror the cursor position into the human-readable
            # event-context line so it tracks programmatic changes
            # (e.g. a scrub from another tab).
            _update_topo_event_label()

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
            axis = state.time_axis or "fabula"
            tmin, tmax = axis_bounds(ws, axis)
            # The Sankey filters by fabula_time on edges; resolve the
            # syuzhet cursor through the fabula axis so the visible
            # span matches what the audience has been told by s=K.
            cursor_value = (
                state.syuzhet_cursor if axis == "syuzhet"
                else state.fabula_cursor
            )
            fabula_max = resolve_cursor(ws, axis, cursor_value)
            _, fabula_tmax = fabula_time_bounds(ws)

            with graph_container:
                # Show the focus row only once we know we have a graph.
                focus_row.set_visibility(True)

                # Helpers shared by Sankey + Force click handlers.
                def _resolve_node_type(nid: str, ws_now) -> str:
                    if ws_now is None:
                        return "event"
                    if nid in {e.id for e in ws_now.events}:
                        return "event"
                    if nid in ws_now.entities:
                        return "entity"
                    if nid in ws_now.locations:
                        return "location"
                    if nid in ws_now.objects:
                        return "object"
                    if nid in getattr(ws_now, "world_traits", {}):
                        return "world_trait"
                    return "event"

                def _set_focus(nid: str | None) -> None:
                    """Toggle focus on/off for ``nid`` and re-render."""
                    try:
                        focus_state["hops"] = int(hops_select.value or 2)
                    except (TypeError, ValueError):
                        focus_state["hops"] = 2
                    if focus_state["id"] == nid:
                        focus_state["id"] = None
                        focus_label.text = (
                            "No focus \u2014 click any node to drill "
                            "into its ancestor / descendant chain."
                        )
                    else:
                        focus_state["id"] = nid
                        focus_label.text = (
                            f"Focused on {nid} (\u00b1"
                            f"{focus_state['hops']} hops). Click "
                            "again to clear."
                        )
                    _refresh()

                if is_sankey:
                    fmax = (
                        fabula_max
                        if fabula_max is not None
                        else fabula_tmax
                    )
                    aspect_label = dict(SANKEY_ASPECTS).get(
                        aspect_select.value, "Sankey"
                    )

                    def _on_sankey_click(ev):
                        # Wire cross-chart selection: store the clicked
                        # node id on AppState so other tabs can react,
                        # and toggle the local focus drill-down so the
                        # Sankey re-renders restricted to that chain.
                        try:
                            data = ev.args.get("data") or {}
                            nid = (
                                data.get("name")
                                if isinstance(data, dict) else None
                            )
                            if not nid:
                                return
                            ws_now = state.world_state
                            state.select_node(
                                nid, _resolve_node_type(nid, ws_now),
                            )
                            _set_focus(nid)
                        except Exception:
                            logger.debug(
                                "sankey click: unparsable args",
                                exc_info=True,
                            )

                    with_expand(
                        lambda h: render_causal_sankey(
                            ws,
                            height=h,
                            aspect=aspect_select.value,
                            min_force=float(force_slider.value or 0.0),
                            fabula_max=fmax if fabula_tmax > 0 else None,
                            focus_id=focus_state["id"],
                            focus_max_hops=focus_state["hops"],
                            on_click=_on_sankey_click,
                        ),
                        title=f"Sankey \u2014 {aspect_label}",
                    )
                else:
                    def _on_force_click(ev):
                        try:
                            data = ev.args.get("data") or {}
                            nid = (
                                data.get("id") or data.get("name")
                                if isinstance(data, dict) else None
                            )
                            if not nid or state.world_state is None:
                                return
                            state.select_node(
                                nid,
                                _resolve_node_type(
                                    nid, state.world_state,
                                ),
                            )
                            _set_focus(nid)
                        except Exception:
                            logger.debug(
                                "force click: unparsable args",
                                exc_info=True,
                            )

                    with_expand(
                        lambda h: render_causal_force_graph(
                            ws,
                            height=h,
                            layout=force_layout.value or "timeline",
                            focus_id=focus_state["id"],
                            focus_max_hops=focus_state["hops"],
                            on_click=_on_force_click,
                        ),
                        title="Causal force graph",
                    )

        def _on_slider(key: str):
            if key == "t":
                # Push to global cursor; the cursor-changed subscription
                # below triggers _refresh(), so callers see the same data
                # as every other time-aware panel.
                try:
                    t = int(time_slider.value)
                except (TypeError, ValueError):
                    return
                axis = state.time_axis or "fabula"
                cur_global = (
                    state.syuzhet_cursor if axis == "syuzhet"
                    else state.fabula_cursor
                )
                if cur_global == t:
                    return
                _slider_state["local_origin_t"] = True
                if axis == "syuzhet":
                    state.set_syuzhet_cursor(t)
                else:
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
        hops_select.on("update:model-value", lambda: _refresh())
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
        # NOTE: the workspace top-tab listener sets ``active_path`` to the
        # bare top-level key (e.g. ``"reasoning"``); reasoning's sub-tabs
        # don't push a more specific path. Gating on ``"reasoning.topology"``
        # therefore matched *never* and silently swallowed every
        # slider/combo refresh. Gate on the top-level path instead so the
        # slider/aspect/layout/force-min controls actually take effect.
        _TOPO_PATH = "reasoning"
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
        # Cross-tab cursor sync: scrubbing the cursor anywhere (World
        # tab, Causal Graph snapshot, Affective Dashboard) should update
        # this panel's filter so all panels share one timeline.
        state.on(StateEvent.FABULA_CURSOR_CHANGED, lambda **_kw: _refresh())
        state.on(StateEvent.SYUZHET_CURSOR_CHANGED, lambda **_kw: _refresh())
        state.on(StateEvent.TIME_AXIS_CHANGED, lambda **_kw: _refresh())


# =====================================================================
# 5. Ingestion Warnings (Tier 4 #14)
# =====================================================================

def _build_ingestion_warnings_panel(state: AppState) -> None:
    """List validator / auto-fix records captured during the last ingest.

    Reads the per-project buffer maintained by
    :mod:`shadow_loom.ingestion_diagnostics`. Empty when no ingest has
    run in this process or all chunks passed cleanly.
    """
    from collections import Counter
    from shadow_loom.ingestion_diagnostics import get_ingestion_warnings

    project_id = getattr(state, "project_id", None)
    if project_id is None:
        ui.label("No project selected.").classes("text-slate-500")
        return

    records = get_ingestion_warnings(str(project_id))
    if not records:
        ui.label(
            "No ingestion warnings captured in this session. "
            "Re-ingest the project to populate this panel."
        ).classes("text-slate-500")
        return

    by_cat = Counter(r.category for r in records)
    with ui.row().classes("w-full items-center gap-3 mb-3"):
        ui.label(f"{len(records)} record(s)").classes(
            "text-sm font-semibold text-slate-700"
        )
        for cat, count in by_cat.most_common():
            ui.badge(f"{cat}: {count}").classes("text-xs")

    cat_filter = {"value": "(all)"}
    cat_options = ["(all)"] + sorted(by_cat.keys())

    list_container = ui.column().classes("w-full gap-1")

    def _render() -> None:
        list_container.clear()
        sel = cat_filter["value"]
        with list_container:
            for r in records:
                if sel != "(all)" and r.category != sel:
                    continue
                colour = (
                    "text-red-700"
                    if r.level in ("ERROR", "CRITICAL")
                    else "text-amber-700"
                    if r.level == "WARNING"
                    else "text-slate-700"
                )
                with ui.row().classes("w-full items-start gap-2"):
                    ui.label(r.level).classes(
                        f"text-xs font-mono w-16 {colour}"
                    )
                    ui.label(r.message).classes("text-xs font-mono flex-1")

    def _on_change(e):
        cat_filter["value"] = e.value
        _render()

    ui.select(
        options=cat_options,
        value="(all)",
        label="Filter by category",
        on_change=_on_change,
    ).classes("w-64 mb-2")

    _render()


# =====================================================================
# 4. Affective Dashboard
# =====================================================================

def build_affective_dashboard(state: AppState) -> None:
    """Emotional gauges + affective metrics over fabula time.

    Pared back from the previous collage of densities, polars, and
    calendar overlays to two focused panels:

      1. Gauges (current snapshot) \u2014 at-a-glance qualitative state.
      2. Multi-line time-series \u2014 how each affective metric evolves
         across fabula time, with the active cursor as a needle.

    Plus a compact event-timeline scatter for context.
    """

    from shadow_loom_ui.viz_helpers import (
        affective_timeseries,
        affective_timeseries_syuzhet,
        compute_affective_scores,
        compute_character_emotion_grid,
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
            # Time-axis is driven by the global picker in the workspace
            # toolbar (state.time_axis). Local label mirrors it so the
            # user can see at a glance which mode the slider is in.
            axis_label = ui.label(
                "Syuzhet" if state.time_axis == "syuzhet" else "Fabula"
            ).classes(
                "text-xs uppercase tracking-wide text-slate-500 "
                "px-2 py-0.5 rounded bg-slate-100 border border-slate-200"
            )
            axis_label.tooltip(
                "Active time axis (change via the global axis picker "
                "in the toolbar). Fabula = chronological story-world "
                "time. Syuzhet = the order the reader encounters "
                "events."
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
                if state.time_axis == "syuzhet":
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

            # Full-width event-context line: shows what event sits at
            # the cursor's current axis position so the slider becomes
            # navigable by *content* rather than just by t / s number.
            # Updated on every slider tick (cheap text update only —
            # no chart rerender) plus inside ``_refresh`` for state
            # changes from elsewhere.
            event_at_cursor = ui.label("\u2014").classes(
                "text-xs text-slate-500 italic basis-full pl-1 pt-1 "
                "truncate"
            )
            event_at_cursor.tooltip(
                "Event at the current cursor (or nearest if none lands here)"
            )

        def _update_event_label():
            ws = state.world_state
            if ws is None:
                event_at_cursor.text = "\u2014"
                return
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            axis = "syuzhet" if state.time_axis == "syuzhet" else "fabula"
            event_at_cursor.text = _events_at(ws, t, axis=axis)

        # ``input`` fires while the user drags so the label updates
        # live. ``_update_event_label`` only mutates a label, so it's
        # safe to wire to the high-frequency event without entering
        # the snapshot/scoring path.
        time_slider.on("update:model-value", lambda _e: _update_event_label())

        def _on_slider_change():
            if _suppress["slider"]:
                return
            try:
                t = int(time_slider.value)
            except (TypeError, ValueError):
                return
            if state.time_axis == "syuzhet":
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
            "keydown.f", lambda: state.set_time_axis("fabula")
        )
        time_slider.on(
            "keydown.s", lambda: state.set_time_axis("syuzhet")
        )

        gauge_container = ui.column().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        )
        char_emotion_container = ui.column().classes(
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
                    # R19-UI-(2): factual-vs-shadow overlay toggle.
                    # Only meaningful when the current VWM head is a
                    # shadow branch; harmless on factual (the overlay
                    # short-circuits when baseline is the same ws).
                    vs_factual_toggle = ui.switch(
                        "vs factual", value=False,
                        on_change=lambda _: _refresh(),
                    ).props("dense").tooltip(
                        "Overlay the factual mainline baseline as "
                        "dashed lines (affect) and hollow rings "
                        "(events) so shadow-branch divergences are "
                        "visible at a glance."
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
                _sth_aff = subtab_help
                with ui.tab_panel("events"):
                    _sth_aff("affective.events")
                    event_table = ui.table(
                        columns=[
                            {"name": "id", "label": "ID", "field": "id", "sortable": True},
                            {"name": "fabula_time", "label": "Fabula t", "field": "fabula_time", "sortable": True},
                            {"name": "syuzhet_index", "label": "Syuzhet", "field": "syuzhet_index", "sortable": True},
                            {"name": "type", "label": "Type", "field": "type", "sortable": True},
                            {"name": "actors", "label": "Actors", "field": "actors"},
                            {"name": "targets", "label": "Targets", "field": "targets"},
                            {"name": "at_location", "label": "At location", "field": "at_location", "sortable": True},
                            {"name": "description", "label": "Description", "field": "description"},
                        ],
                        rows=[],
                        pagination={"rowsPerPage": 10},
                    ).props(_tprops).classes("w-full")
                with ui.tab_panel("affect"):
                    _sth_aff("affective.affect")
                    affect_table = ui.table(
                        columns=[
                            {"name": "metric", "label": "Metric", "field": "metric", "sortable": True},
                            {"name": "score", "label": "Score (raw)", "field": "score", "sortable": True},
                            {"name": "normalized", "label": "Score (normalized)", "field": "normalized", "sortable": True},
                        ],
                        rows=[],
                    ).props(_tprops).classes("w-full")

        def _refresh(**kw):
            # Guard against the client being torn down between when the
            # timer/task was scheduled and when it actually runs (e.g.
            # rapid tab switches, page reload, browser nav). NiceGUI
            # raises RuntimeError("The client this element belongs to
            # has been deleted.") from any element method in that
            # window — including .clear(). Bail out silently; the next
            # mount will re-render from scratch.
            try:
                gauge_container.clear()
                char_emotion_container.clear()
            except RuntimeError:
                logger.debug(
                    "[causality_tab._refresh] client torn down before "
                    "refresh; skipping."
                )
                return
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

            is_syuzhet = state.time_axis == "syuzhet"
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
                # Refresh the event-context line to match the
                # programmatic cursor change (e.g. World tab scrub).
                _update_event_label()
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

            # The time-series and event-timeline charts must show the
            # *whole* story regardless of where the scrub cursor sits —
            # otherwise dragging the slider visually deletes events.
            # Keep a reference to the full, untrimmed world for those
            # views; ``ws`` (possibly snapshotted above) is still used
            # for the gauges and the raw-event table.
            full_ws = state.world_state

            # Engine-grade affects (suspense, surprise, dramatic_irony,
            # canonical mystery) need a focus entity set + reveal-set
            # anchor. Pick entities from the *full* world so the focus
            # set is stable across cursor scrubs (otherwise early-time
            # snapshots can drop the protagonist). The reveal-set
            # semantics differ per axis:
            #
            #   syuzhet mode: ``syuzhet_anchor`` = current reader
            #       position (or the snapshot's max syuzhet index when
            #       no cursor is set), ``fabula_anchor`` = None.
            #   fabula mode:  ``fabula_anchor`` = current fabula tick
            #       (driver of the snapshot), ``syuzhet_anchor`` =
            #       None.
            #
            # The previous fabula-mode form passed
            # ``syuzhet_anchor = max(e.syuzhet_index for e in ws.events)``
            # which read a syuzhet anchor off a fabula-snapshotted
            # world. On any non-linear plot (flashbacks → an early
            # fabula tick contains a high-syuzhet event) that pinned
            # the reveal-set near the end of the syuzhet axis,
            # zero-ing out suspense / surprise on the gauge while
            # the timeseries — which correctly uses ``fabula_anchor``
            # — still showed meaningful values. Per-axis split below
            # keeps the gauge and the timeseries in agreement.
            entity_ids = _top_entity_ids_by_event_degree(
                full_ws or ws, limit=20,
            )
            if is_syuzhet:
                syuzhet_anchor_val: int | None
                if state.syuzhet_cursor is not None:
                    syuzhet_anchor_val = int(state.syuzhet_cursor)
                else:
                    syuzhet_anchor_val = max(
                        (e.syuzhet_index for e in ws.events), default=None
                    )
                fabula_anchor_val: int | None = None
            else:
                syuzhet_anchor_val = None
                if state.fabula_cursor is not None:
                    fabula_anchor_val = int(state.fabula_cursor)
                else:
                    fabula_anchor_val = max(
                        (e.fabula_time for e in ws.events), default=None
                    )
            # Run the gauge scorer with ``ws_for_engine=full_ws`` so the
            # structural affects (suspense / mystery / dramatic irony /
            # surprise) can see the unrevealed tail — matching how the
            # time-series sampler scores each point.
            scores = compute_affective_scores(
                ws,
                entity_ids=entity_ids,
                syuzhet_anchor=syuzhet_anchor_val,
                ws_for_engine=full_ws,
                surprise_local=True,
                fabula_anchor=fabula_anchor_val,
            )

            # Per-metric normalisation for the gauges. Raw affect scores
            # all live in [0, 1] in principle, but each metric occupies
            # a very different empirical band on any given world (e.g.
            # ``mystery`` and ``causal_density`` routinely sit two
            # decades apart). That makes the gauges visually
            # incomparable — a "moderate" tension band looks identical
            # to a "very high" mystery band even though one occupies the
            # top of its own range and the other sits mid-range. Rescale
            # each metric to [0, 1] of *its own* observed range across
            # the full timeseries so the gauges show position-within-
            # range and become directly comparable. Cached samples are
            # cheap thanks to ``_AFFECT_TIMESERIES_CACHE``.
            try:
                if is_syuzhet:
                    _times, _series = affective_timeseries_syuzhet(
                        full_ws or ws, entity_ids=entity_ids,
                    )
                else:
                    _times, _series = affective_timeseries(
                        full_ws or ws, entity_ids=entity_ids,
                    )
            except Exception:
                logger.exception("Affective timeseries sampling failed")
                _series = {}
            gauge_scores: dict[str, float] = {}
            for k, v in scores.items():
                values = _series.get(k) or []
                if values:
                    lo = min(values + [v])
                    hi = max(values + [v])
                    span = hi - lo
                    gauge_scores[k] = (
                        (v - lo) / span if span > 1e-9 else 0.5
                    )
                else:
                    gauge_scores[k] = float(v)

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
                    ui.label("Narrative Affect Scores (normalized)").classes(
                        "text-lg font-semibold text-slate-800 mb-1"
                    )
                    ui.label(
                        "Each gauge shows the current value rescaled to "
                        "its own observed range across the timeline so "
                        "metrics with different intrinsic magnitudes are "
                        "directly comparable."
                    ).classes("text-xs text-slate-500 mb-2")
                    sel = gauge_select.value
                    sel = None if sel in (None, "", "__all__") else sel
                    # Per-gauge height — the gauges are now laid out in
                    # a CSS grid (one ECharts per metric) so this is
                    # the height of each cell, not the whole container.
                    height_px = "200px" if sel else "180px"
                    with ui.element("div").classes("w-full"):
                        with_expand(
                            lambda h, s=gauge_scores, g=graded_gauges.value, sel=sel, hp=height_px: (
                                render_emotional_gauges_graded(
                                    s, height=hp, selected=sel
                                ) if g else render_emotional_gauges(
                                    s, height=hp, selected=sel
                                )
                            ),
                            title=(
                                f"Affect gauge \u2014 {sel.replace('_', ' ')}"
                                if sel else "Narrative affect scores (normalized)"
                            ),
                        )
                else:
                    ui.label("No affective scores available.").classes(
                        "text-sm text-slate-400 italic"
                    )

            # ── OCC character-felt emotions (per-character heatmap) ──
            with char_emotion_container:
                from shadow_loom_ui.viz import (
                    render_character_emotion_heatmap,
                )
                ui.label("Character Emotions (OCC appraisals)").classes(
                    "text-lg font-semibold text-slate-800 mb-1"
                )
                ui.label(
                    "Per-character felt emotions \u2014 fear, joy, regret, "
                    "grief, rage, love \u2014 from the appraisal grid for "
                    "the top entities at the current cursor."
                ).classes("text-xs text-slate-500 mb-2")
                try:
                    char_grid = compute_character_emotion_grid(
                        full_ws or ws,
                        entity_ids=entity_ids,
                        syuzhet_anchor=syuzhet_anchor_val,
                    )
                except Exception:
                    logger.exception("Character emotion grid failed")
                    char_grid = {}
                if char_grid:
                    name_lookup = {
                        eid: ent.name
                        for eid, ent in (full_ws or ws).entities.items()
                    }
                    render_character_emotion_heatmap(
                        char_grid,
                        entity_names=name_lookup,
                        height="320px",
                    )
                else:
                    ui.label(
                        "No per-character emotion data \u2014 needs "
                        "propositions and concerns."
                    ).classes("text-sm text-slate-400 italic")

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

                # R19-UI-(2): resolve factual baseline ws for the
                # overlay. ``versioned_model.current`` is the raw
                # merged ``WorldStateV1`` whose ``entities`` dict is
                # the factual baseline (shadow clones live in the
                # ``shadow_*`` sidecars). Only attach when the toggle
                # is on AND the active head is shadow \u2014 otherwise
                # baseline equals the primary ws and the overlay
                # would just duplicate every series.
                _baseline_ws = None
                try:
                    if vs_factual_toggle.value:
                        _vwm = getattr(state, "versioned_model", None)
                        if _vwm is not None and getattr(_vwm, "history", None):
                            _head = _vwm.history[-1]
                            if getattr(_head, "world_id", "factual") == "shadow":
                                _baseline_ws = getattr(_vwm, "current", None)
                except Exception:
                    _baseline_ws = None

                ts_opts = affective_timeseries_options(
                    full_ws or ws,
                    fabula_cursor=fc,
                    syuzhet_cursor=sc_for_chart,
                    axis=axis,
                    entity_ids=eids,
                    normalize=bool(normalize_toggle.value),
                    ws_baseline=_baseline_ws,
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
                    full_ws or ws, fabula_cursor=fc, syuzhet_cursor=sc,
                    ws_baseline=_baseline_ws,
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

            # Refresh data tables to mirror the charts above. The
            # event table shows the full story (matches the Event
            # Timeline chart); the affect table mirrors the gauges.
            event_table.rows = ws_to_event_rows(full_ws or ws)
            affect_table.rows = [
                {
                    "metric": k,
                    "score": round(v, 3),
                    "normalized": round(gauge_scores.get(k, v), 3),
                }
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
                    state.set_time_axis("syuzhet")
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
        _PANEL_PATH = "affective"
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
                    if state.time_axis == "syuzhet"
                    else state.fabula_cursor
                )
                eids = await asyncio.to_thread(
                    _top_entity_ids_by_event_degree, ws, 20,
                )
                if state.time_axis == "syuzhet" and cursor is not None:
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
                    cursor if state.time_axis == "syuzhet"
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
            try:
                _refresh_sync()
            except RuntimeError:
                logger.debug(
                    "[causality_tab._refresh_async] client torn down "
                    "before sync refresh; skipping."
                )

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

        def _on_aff_axis_change(**_kw):
            axis_label.text = (
                "Syuzhet" if state.time_axis == "syuzhet" else "Fabula"
            )
            _refresh()
        state.on(StateEvent.TIME_AXIS_CHANGED, _on_aff_axis_change)



