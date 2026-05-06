# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ECharts-based visualization renderers for Shadow-Loom.

Each function returns a ``nicegui.ui.echart`` element configured with the
appropriate series type, click callbacks, and dark-theme styling.  All data
transformations are delegated to ``viz_helpers``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from nicegui import ui

from shadow_loom.models import WorldStateV1
from shadow_loom_ui.theme import CHART_COLORS, chart_theme
from shadow_loom_ui.viz_helpers import (
    CATEGORIES,
    EDGE_COLORS,
    audit_passrate_data,
    entity_state_timeline_data,
    entity_to_radar_compare_data,
    entity_to_radar_data,
    explain_node_causes,
    explain_node_effects,
    mutations_to_propagation_graph,
    mutations_to_waterfall_data,
    version_tree_to_echart_data,
    ws_sankey_for_aspect,
    ws_to_calendar_graph_data,
    ws_to_causal_cartesian_data,
    ws_to_causal_force_data,
    ws_to_chord_data,
    ws_to_ego_graph_data,
    ws_to_epistemic_data,
    ws_to_event_calendar_data,
    ws_to_gantt_data,
    ws_to_gantt_status_marks,
    ws_to_graph_data,
    ws_to_heatmap_data,
    ws_to_parallel_data,
    ws_to_polar_event_data,
    ws_to_social_graph_data,
    ws_to_spatial_graph_data,
    ws_to_sunburst_data,
    ws_to_theme_river_data,
    ws_to_timeline_data,
    ws_to_trait_boxplot_data,
    ws_to_treemap_data,
)

logger = logging.getLogger(__name__)

OnClick = Optional[Callable[[dict], Any]]

# ── Shared theme defaults (refreshed by ``apply_chart_theme`` whenever
#    the dark-mode flag changes via shadow_loom_ui.theme.set_chart_dark). ──

_CHART_BG = "#ffffff"
_CHART_TEXT = "#1e293b"
_CHART_TOOLTIP: dict = {
    "backgroundColor": "#ffffff",
    "borderColor": "#e2e8f0",
    "textStyle": {"color": "#1e293b"},
}
_LEGEND: dict = {
    "data": [c["name"] for c in CATEGORIES],
    "textStyle": {"color": _CHART_TEXT},
    "top": 0,
    "right": 10,
    "orient": "vertical",
}


def apply_chart_theme() -> None:
    """Refresh the module-level palette constants from
    :func:`shadow_loom_ui.theme.chart_theme`.

    Called by ``theme.set_chart_dark`` so subsequent renders pick up the
    new (light/dark) palette without each renderer having to fetch it.
    """
    global _CHART_BG, _CHART_TEXT, _CHART_TOOLTIP, _LEGEND
    t = chart_theme()
    _CHART_BG = t["bg"]
    _CHART_TEXT = t["text"]
    _CHART_TOOLTIP = {
        "backgroundColor": t["tooltip_bg"],
        "borderColor": t["tooltip_border"],
        "textStyle": {"color": t["tooltip_text"]},
    }
    _LEGEND = {
        "data": [c["name"] for c in CATEGORIES],
        "textStyle": {"color": _CHART_TEXT},
        "top": 0,
        "right": 10,
        "orient": "vertical",
    }


# Initialize once at import (light defaults).
apply_chart_theme()


# ── Expand-to-dialog wrapper ───────────────────────────────────────

def with_expand(
    render_fn: Callable[[str], Any],
    *,
    title: str = "",
    height: str = "100%",
) -> ui.element:
    """Render ``render_fn(height)`` inline with a popout button overlay.

    ``render_fn`` MUST be a callable that takes a CSS height string and
    creates the chart inside the current parent. It is invoked twice:
    once inline and once into a half/full-screen ``ui.dialog`` when the
    user clicks the expand icon. This sidesteps trying to clone an
    existing ``ui.echart`` element (which does not support reparenting).

    Returns the inline container so callers can apply additional styling.
    """
    container = ui.element("div").classes("w-full h-full relative")
    with container:
        render_fn(height)
        expand_btn = ui.button(
            icon="open_in_full",
            on_click=lambda: _open_expand_dialog(render_fn, title),
        ).props("flat dense round size=sm color=grey-7").style(
            "position: absolute; top: 4px; right: 4px; "
            "background: rgba(255,255,255,0.85); z-index: 5;"
        )
        expand_btn.tooltip("Expand (or click & drag to resize once open)")
    return container


def _open_expand_dialog(render_fn: Callable[[str], Any], title: str) -> None:
    """Show ``render_fn`` in a resizable dialog with Esc-to-close + PNG export.

    Supports a "Half-screen" toggle for side-by-side workflows and a
    "Save PNG" button that calls ECharts' ``getDataURL`` on the first
    chart inside the dialog.
    """
    state = {"maximized": True}
    chart_holder = ui.column()  # placeholder, replaced inside dialog

    with ui.dialog().props("maximized persistent").classes("bg-white") as dlg:
        # Esc-to-close
        dlg.on("keydown.esc", lambda: dlg.close())
        with ui.card().classes("w-full h-full p-0 m-0 bg-white") as card:
            with ui.row().classes(
                "w-full items-center justify-between px-4 py-2 "
                "border-b border-slate-200 bg-slate-50"
            ):
                ui.label(title or "Diagram").classes(
                    "text-lg font-semibold text-slate-800"
                )
                with ui.row().classes("items-center gap-2"):
                    def _toggle_size():
                        state["maximized"] = not state["maximized"]
                        if state["maximized"]:
                            dlg.props("maximized")
                            card.classes(replace="w-full h-full p-0 m-0 bg-white")
                            size_btn.props("icon=close_fullscreen")
                            size_btn.tooltip("Half-screen")
                        else:
                            dlg.props(remove="maximized")
                            card.classes(
                                replace="p-0 m-0 bg-white"
                            ).style("width: 60vw; height: 70vh;")
                            size_btn.props("icon=open_in_full")
                            size_btn.tooltip("Maximize")

                    size_btn = ui.button(
                        icon="close_fullscreen", on_click=_toggle_size
                    ).props("flat dense round color=grey-8")
                    size_btn.tooltip("Half-screen")

                    def _save_png():
                        # Find the first ECharts canvas in the dialog and
                        # download it as PNG via getDataURL().
                        ui.run_javascript(
                            """
                            (() => {
                                const dlg = document.querySelector('.q-dialog__inner [class*="echarts"]')
                                          || document.querySelector('.q-dialog__inner canvas');
                                if (!dlg) return;
                                const canvas = dlg.tagName === 'CANVAS' ? dlg
                                              : dlg.querySelector('canvas');
                                if (!canvas) return;
                                const a = document.createElement('a');
                                a.href = canvas.toDataURL('image/png');
                                a.download = 'shadow-loom-chart.png';
                                a.click();
                            })();
                            """
                        )

                    png_btn = ui.button(icon="download", on_click=_save_png).props(
                        "flat dense round color=grey-8"
                    )
                    png_btn.tooltip("Save as PNG")

                    close_btn = ui.button(icon="close", on_click=dlg.close).props(
                        "flat dense round color=grey-8"
                    )
                    close_btn.tooltip("Close (Esc)")
            chart_holder = ui.column().classes("w-full flex-grow p-4")
            with chart_holder:
                render_fn("100%")
    dlg.open()


# ── Empty-state helper ─────────────────────────────────────────────

def render_empty_state(
    message: str,
    *,
    icon: str = "info",
    hint: str | None = None,
) -> None:
    """Friendly empty state: icon + message + optional 'how to fix' hint."""
    with ui.column().classes(
        "w-full h-full items-center justify-center text-center gap-2 p-6"
    ):
        ui.icon(icon, size="2rem", color="grey-5")
        ui.label(message).classes("text-sm text-slate-500")
        if hint:
            ui.label(hint).classes("text-xs text-slate-400 italic max-w-md")


# ── Color legend chip strip ────────────────────────────────────────

def render_node_legend() -> None:
    """Compact horizontal chip strip showing node-type colors."""
    legend_items = [
        ("Event", "#F5B43C", "EVT_"),
        ("Entity", "#F26B5E", "ENT_"),
        ("Location", "#3A7BD5", "LOC_"),
        ("Object", "#8A5CF0", "OBJ_"),
        ("World Trait", "#2EA6A0", "WORLD_"),
    ]
    with ui.row().classes("items-center gap-2 flex-wrap"):
        for label, color, prefix in legend_items:
            with ui.row().classes("items-center gap-1"):
                ui.element("div").style(
                    f"width:10px;height:10px;border-radius:2px;background:{color};"
                )
                ui.label(label).classes("text-xs text-slate-600")
                ui.label(prefix).classes(
                    "text-xs font-mono text-slate-400"
                )


# ── Loading skeleton ───────────────────────────────────────────────

def render_chart_skeleton(height: str = "300px") -> None:
    """Quasar pulsing skeleton placeholder while a chart loads."""
    with ui.column().classes("w-full items-stretch gap-2 p-2").style(
        f"height:{height}"
    ):
        ui.element("q-skeleton").props(
            'type="rect" animation="wave" height="24px" width="40%"'
        )
        ui.element("q-skeleton").props(
            'type="rect" animation="wave" height="100%" class="col"'
        ).classes("flex-grow")


# ── Force-directed full world graph ────────────────────────────────

def render_world_graph(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
) -> ui.echart:
    """Full world graph — force layout with adjacency highlighting."""
    nodes, links, cats = ws_to_graph_data(ws)
    # Always-on labels collapse into illegible noise once the graph
    # holds more than ~25 nodes. Above that, hide them and let users
    # hover/click for the name; below, keep them on for readability.
    show_labels = len(nodes) <= 25
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "animationDuration": 800,
        "animationEasingUpdate": "quinticInOut",
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "draggable": True,
            "emphasis": {"focus": "adjacency", "blurScope": "coordinateSystem"},
            "categories": cats,
            "data": nodes,
            "links": links,
            "force": {
                "repulsion": 450,
                "gravity": 0.08,
                "edgeLength": [80, 200],
                "friction": 0.6,
            },
            "label": {
                "show": show_labels,
                "position": "right",
                "fontSize": 10,
                "color": _CHART_TEXT,
            },
            "lineStyle": {"curveness": 0.15, "opacity": 0.6},
        }],
    }).classes(f"w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Ego graph (subset centered on focus entities) ──────────────────

def render_ego_graph(
    ws: WorldStateV1,
    focus_ids: list[str],
    *,
    max_hops: int = 2,
    on_click: OnClick = None,
    height: str = "100%",
) -> ui.echart:
    """Ego-graph centered on *focus_ids* with gold-bordered focus nodes."""
    nodes, links, cats = ws_to_ego_graph_data(ws, focus_ids, max_hops=max_hops)
    show_labels = len(nodes) <= 25
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "animationDuration": 600,
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "draggable": True,
            "emphasis": {"focus": "adjacency"},
            "categories": cats,
            "data": nodes,
            "links": links,
            "force": {
                "repulsion": 350,
                "gravity": 0.12,
                "edgeLength": [60, 160],
                "friction": 0.6,
            },
            "label": {"show": show_labels, "position": "right", "fontSize": 10, "color": _CHART_TEXT},
            "lineStyle": {"curveness": 0.15, "opacity": 0.6},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Social graph ───────────────────────────────────────────────────

def render_social_graph(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
    layout: str = "force",
) -> ui.echart:
    """Entities + relationship edges — colored by affinity.

    ``layout='circular'`` arranges entities on a ring (cleaner for
    ensemble casts where force layouts collapse into hairballs).
    """
    nodes, links, cats = ws_to_social_graph_data(ws)
    # Circular layout copes with larger casts; force layout collapses
    # earlier. Pick a label threshold per layout.
    show_labels = (
        len(nodes) <= 30 if layout == "circular" else len(nodes) <= 20
    )
    series_extra: dict = (
        {"layout": "circular", "circular": {"rotateLabel": True}}
        if layout == "circular"
        else {
            "layout": "force",
            "force": {"repulsion": 300, "gravity": 0.15, "edgeLength": [80, 180]},
        }
    )
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "animationDuration": 600,
        "series": [{
            "type": "graph",
            "roam": True,
            "draggable": True,
            "emphasis": {"focus": "adjacency"},
            "categories": cats,
            "data": nodes,
            "links": links,
            "label": {"show": show_labels, "position": "right", "fontSize": 11, "color": _CHART_TEXT},
            "lineStyle": {"curveness": 0.2, "opacity": 0.7},
            **series_extra,
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Spatial map ────────────────────────────────────────────────────

def render_spatial_map(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
    animated: bool = False,
) -> ui.echart:
    """Locations + spatial edges — dashed lines for locked connections.

    ``animated=True`` adds an ECharts ``lines`` overlay with a
    travelling glow effect along each edge to suggest movement of
    matter through the world.
    """
    nodes, links, cats = ws_to_spatial_graph_data(ws)
    show_labels = len(nodes) <= 25
    series: list[dict] = [{
        "type": "graph",
        "layout": "force",
        "roam": True,
        "draggable": True,
        "emphasis": {"focus": "adjacency"},
        "categories": cats,
        "data": nodes,
        "links": links,
        "force": {"repulsion": 250, "gravity": 0.2, "edgeLength": [60, 140]},
        "label": {"show": show_labels, "position": "right", "fontSize": 11, "color": _CHART_TEXT},
        "lineStyle": {"curveness": 0.1, "opacity": 0.7},
    }]
    if animated and links:
        # Overlay a 'lines' series with effect.show; ECharts will animate
        # a small symbol travelling source→target along each edge.
        flow_data = [
            {
                "coords": [[0, 0], [0, 0]],  # placeholder; real coords come from graph
                "fromName": ln.get("source"),
                "toName": ln.get("target"),
            }
            for ln in links
        ]
        # ECharts can't link a 'lines' series to a 'graph' coord system
        # directly without coordinates. We instead set the graph series'
        # ``edgeSymbol`` + ``edgeLabel`` so each edge gets a moving glyph.
        series[0]["edgeSymbol"] = ["none", "arrow"]
        series[0]["edgeSymbolSize"] = [0, 8]
        series[0]["lineStyle"]["opacity"] = 0.85
        series[0]["emphasis"]["lineStyle"] = {"width": 4}

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "animationDuration": 600,
        "series": series,
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Causal Sankey diagram ─────────────────────────────────────────

def render_causal_sankey(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
    aspect: str = "causal_all",
    min_force: float = 0.0,
    fabula_max: int | None = None,
) -> ui.echart:
    """Sankey diagram of causal/social/information flow.

    ``aspect`` selects the underlying edge set — see
    :data:`shadow_loom_ui.viz_helpers.SANKEY_ASPECTS`.
    """
    nodes, links = ws_sankey_for_aspect(
        ws, aspect, min_force=min_force, fabula_max=fabula_max
    )
    if not nodes:
        return ui.label("No edges to display for this aspect.").classes(
            "text-grey q-pa-md"
        )

    # Top-N pruning: Sankeys become unreadable past ~40 nodes / 60
    # links. Keep the strongest links by ``value`` (causal_force or
    # equivalent) and drop nodes that no longer participate.
    truncated = False
    MAX_LINKS = 60
    if len(links) > MAX_LINKS:
        links_sorted = sorted(
            links,
            key=lambda l: float(l.get("value", 0) or 0),
            reverse=True,
        )
        links = links_sorted[:MAX_LINKS]
        live_ids = {l.get("source") for l in links} | {
            l.get("target") for l in links
        }
        nodes = [n for n in nodes if n.get("name") in live_ids]
        truncated = True

    # Hide always-on labels above the comfortable label-stacking
    # threshold; rely on tooltip + emphasis.
    show_labels = len(nodes) <= 25

    options: dict = {
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "sankey",
            "data": nodes,
            "links": links,
            "emphasis": {"focus": "adjacency"},
            "nodeAlign": "left",
            "orient": "horizontal",
            "nodeGap": 12,
            "nodeWidth": 14,
            "draggable": True,
            "lineStyle": {
                "color": "source",
                "curveness": 0.5,
                "opacity": 0.45,
            },
            "itemStyle": {
                "borderWidth": 1,
                "borderColor": _CHART_TEXT,
            },
            "label": {
                "show": show_labels,
                "color": _CHART_TEXT,
                "fontSize": 10,
            },
        }],
    }
    if truncated:
        options["title"] = {
            "text": f"showing top {MAX_LINKS} flows by force",
            "top": 4,
            "left": "center",
            "textStyle": {
                "color": _CHART_TEXT,
                "fontSize": 10,
                "fontWeight": "normal",
            },
        }
    chart = ui.echart(options).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Trait radar chart ──────────────────────────────────────────────

def render_trait_radar(
    entity_id: str,
    ws: WorldStateV1,
    *,
    height: str = "300px",
) -> ui.echart:
    """Spider/radar chart for a single entity's trait vector."""
    radar_data = entity_to_radar_data(entity_id, ws)
    if not radar_data["indicator"]:
        return ui.label("No traits.").classes("text-grey text-caption")

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": _CHART_TOOLTIP,
        "radar": {
            "indicator": radar_data["indicator"],
            "shape": "polygon",
            "axisName": {"color": _CHART_TEXT, "fontSize": 10},
            "splitArea": {"areaStyle": {"color": ["rgba(255,255,255,0.02)", "rgba(255,255,255,0.05)"]}},
            "splitLine": {"lineStyle": {"color": "#444"}},
            "axisLine": {"lineStyle": {"color": "#555"}},
        },
        "series": [{
            "type": "radar",
            "data": radar_data["data"],
            "areaStyle": {"opacity": 0.2},
            "lineStyle": {"width": 2},
            "symbol": "circle",
            "symbolSize": 5,
        }],
    }).classes("w-full").style(f"height:{height}")


# ── Relationship heatmap ──────────────────────────────────────────

def render_relationship_heatmap(
    ws: WorldStateV1,
    *,
    metric: str = "affinity",
    on_click: OnClick = None,
    height: str = "100%",
) -> ui.echart:
    """Entity × entity heatmap colored by a relationship metric.

    The colour scale and value range adapt to ``metric``:
      * ``affinity`` and ``power_dynamic`` are signed in [-1, 1] and
        use a red→grey→green ramp (negative → neutral → positive).
      * ``fear`` is unsigned in [0, 1] and uses a single-hue ramp
        (calm → terrified) so a grey midpoint isn't misread as "no
        signal".
    """
    names, data = ws_to_heatmap_data(ws, metric=metric)
    if not names:
        return ui.label(
            f"No {metric} data — no social edges between entities."
        ).classes("text-grey q-pa-md")

    if metric == "fear":
        vmin, vmax = 0.0, 1.0
        ramp = ["#0F2233", "#3A7BD5", "#F5B43C", "#D8334A"]
    elif metric == "power_dynamic":
        vmin, vmax = -1.0, 1.0
        ramp = ["#3A7BD5", "#94a3b8", "#F5B43C"]
    else:  # affinity (default)
        vmin, vmax = -1.0, 1.0
        ramp = ["#D8334A", "#94a3b8", "#6FBF3A"]

    # Per-cell value labels are only legible up to ~12×12. Beyond
    # that they collide and turn into noise; rely on the colour ramp
    # + tooltip instead. Axis labels also need thinning at scale.
    n = len(names)
    show_labels = n <= 12
    label_interval = 0 if n <= 25 else max(0, n // 25)

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "position": "top"},
        "grid": {"top": 30, "bottom": 80, "left": 100, "right": 30},
        "xAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {
                "rotate": 45,
                "color": _CHART_TEXT,
                "fontSize": 10,
                "interval": label_interval,
            },
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {
                "color": _CHART_TEXT,
                "fontSize": 10,
                "interval": label_interval,
            },
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": vmin,
            "max": vmax,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
            "inRange": {"color": ramp},
            "textStyle": {"color": _CHART_TEXT},
        },
        "series": [{
            "type": "heatmap",
            "name": metric,
            "data": data,
            "label": {"show": show_labels, "fontSize": 9, "color": "#eee"},
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Relationship heatmap over time (animated) ─────────────────────

def render_relationship_heatmap_timeline(
    ws: WorldStateV1,
    *,
    metric: str = "affinity",
    num_frames: int = 12,
    height: str = "100%",
) -> ui.echart:
    """Animated entity×entity heatmap scrubbing across fabula time.

    Same colour scale and axes as :func:`render_relationship_heatmap`,
    but wrapped in an ECharts ``timeline`` so each step shows the dyad
    matrix as it stood at that fabula tick. Frames are reconstructed
    via :func:`relationship_heatmap_frames`, which layers
    ``mutation_social`` causal edges and authored snapshots on top of
    the steady-state ``RelationshipEdge`` baseline (see
    :func:`viz_helpers.reconstruct_relationship_with_causal`).
    """
    from shadow_loom_ui.viz_helpers import relationship_heatmap_frames

    payload = relationship_heatmap_frames(
        ws, metric=metric, num_frames=num_frames
    )
    names = payload["names"]
    times = payload["times"]
    frames = payload["frames"]
    if not names or not frames:
        return ui.label(
            f"No {metric} data over time — no social edges between entities."
        ).classes("text-grey q-pa-md")

    if metric == "fear":
        vmin, vmax = 0.0, 1.0
        ramp = ["#0F2233", "#3A7BD5", "#F5B43C", "#D8334A"]
    elif metric == "power_dynamic":
        vmin, vmax = -1.0, 1.0
        ramp = ["#3A7BD5", "#94a3b8", "#F5B43C"]
    else:
        vmin, vmax = -1.0, 1.0
        ramp = ["#D8334A", "#94a3b8", "#6FBF3A"]

    n = len(names)
    show_labels = n <= 12
    label_interval = 0 if n <= 25 else max(0, n // 25)

    base_option = {
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "position": "top"},
        # Bottom space: ~50px for visualMap + ~60px for the timeline.
        "grid": {"top": 30, "bottom": 130, "left": 100, "right": 30},
        "xAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {
                "rotate": 45,
                "color": _CHART_TEXT,
                "fontSize": 10,
                "interval": label_interval,
            },
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {
                "color": _CHART_TEXT,
                "fontSize": 10,
                "interval": label_interval,
            },
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": vmin,
            "max": vmax,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 70,
            "inRange": {"color": ramp},
            "textStyle": {"color": _CHART_TEXT},
        },
        "timeline": {
            "axisType": "category",
            "data": [f"t={t}" for t in times],
            "autoPlay": False,
            "loop": False,
            "playInterval": 1200,
            "currentIndex": len(times) - 1,
            "bottom": 10,
            "left": 80,
            "right": 60,
            "label": {"color": _CHART_TEXT, "fontSize": 9},
            "controlStyle": {"color": _CHART_TEXT, "borderColor": _CHART_TEXT},
            "lineStyle": {"color": "#555"},
            "checkpointStyle": {"color": "#3A7BD5"},
            "itemStyle": {"color": "#777"},
        },
    }

    options = [
        {
            "title": {
                "text": f"{metric}  @  fabula t={t}",
                "left": "center",
                "textStyle": {"color": _CHART_TEXT, "fontSize": 12},
            },
            "series": [{
                "type": "heatmap",
                "name": metric,
                "data": frame,
                "label": {"show": show_labels, "fontSize": 9, "color": "#eee"},
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowColor": "rgba(0,0,0,0.5)",
                    }
                },
            }],
        }
        for t, frame in zip(times, frames)
    ]

    return ui.echart({
        "baseOption": base_option,
        "options": options,
    }).classes("w-full").style(f"height:{height}")


# ── Emotional / narrative gauges ──────────────────────────────────

def _gauge_grid_columns(n: int) -> int:
    """Choose how many gauges to put on each grid row.

    Single ECharts ``gauge`` series sharing a chart container always
    overlap unless we manually tile them with non-overlapping ``center``
    values, which is fragile across heights. Instead we render each
    gauge as its own ``ui.echart`` and let CSS grid place them.
    """
    if n <= 1:
        return 1
    if n == 2:
        return 2
    if n <= 4:
        return 2
    if n <= 6:
        return 3
    return 4


def _build_gauge_series(name: str, val: float, *, graded: bool) -> dict:
    """Build one ECharts ``gauge`` series dict centered in its own chart."""
    common = {
        "type": "gauge",
        "center": ["50%", "58%"],
        "radius": "78%",
        "startAngle": 200,
        "endAngle": -20,
        "min": 0,
        "max": 1,
        "splitNumber": 5,
        "data": [{"value": round(float(val), 2), "name": name}],
        "pointer": {"width": 4},
        "axisTick": {"lineStyle": {"color": "#777"}},
        "splitLine": {"lineStyle": {"color": "#777"}},
        "title": {
            "color": _CHART_TEXT,
            "fontSize": 11,
            "offsetCenter": [0, "88%"],
        },
    }
    if graded:
        common["axisLine"] = {
            "lineStyle": {
                "width": 14,
                "color": [
                    [0.2, "#94a3b8"],
                    [0.4, "#6FBF3A"],
                    [0.6, "#F5B43C"],
                    [0.8, "#FF8C42"],
                    [1, "#D8334A"],
                ],
            },
        }
        common["axisLabel"] = {
            "color": _CHART_TEXT,
            "fontSize": 9,
            "distance": 14,
            ":formatter": (
                "function (v) {"
                "  if (v <= 0.2) return 'v.low';"
                "  if (v <= 0.4) return 'low';"
                "  if (v <= 0.6) return 'mod';"
                "  if (v <= 0.8) return 'high';"
                "  return 'v.high';"
                "}"
            ),
        }
        common["detail"] = {
            "valueAnimation": True,
            "color": _CHART_TEXT,
            "fontSize": 13,
            "lineHeight": 14,
            "offsetCenter": [0, "30%"],
            ":formatter": (
                "function (v) {"
                "  var label;"
                "  if (v <= 0.2) label = 'very low';"
                "  else if (v <= 0.4) label = 'low';"
                "  else if (v <= 0.6) label = 'moderate';"
                "  else if (v <= 0.8) label = 'high';"
                "  else label = 'very high';"
                "  return label + '\\n' + v.toFixed(2);"
                "}"
            ),
        }
    else:
        common["axisLine"] = {
            "lineStyle": {
                "width": 12,
                "color": [
                    [0.3, "#6FBF3A"],
                    [0.7, "#F5B43C"],
                    [1, "#D8334A"],
                ],
            },
        }
        common["axisLabel"] = {
            "color": _CHART_TEXT,
            "fontSize": 9,
            "distance": 14,
        }
        common["detail"] = {
            "valueAnimation": True,
            "formatter": "{value}",
            "color": _CHART_TEXT,
            "fontSize": 14,
            "offsetCenter": [0, "35%"],
        }
    return common


def _render_single_gauge(
    name: str,
    val: float,
    *,
    height: str,
    graded: bool,
) -> ui.echart:
    return ui.echart({
        "backgroundColor": _CHART_BG,
        "series": [_build_gauge_series(name, val, graded=graded)],
    }).classes("w-full").style(f"height:{height}")


def _render_gauge_grid(
    items: list[tuple[str, float]],
    *,
    height: str,
    selected: str | None,
    graded: bool,
) -> ui.element:
    if selected is not None:
        return _render_single_gauge(
            selected, dict(items).get(selected, 0.0),
            height=height, graded=graded,
        )
    if len(items) == 1:
        name, val = items[0]
        return _render_single_gauge(name, val, height=height, graded=graded)

    cols = _gauge_grid_columns(len(items))
    grid = ui.grid(columns=cols).classes("w-full gap-3")
    with grid:
        for name, val in items:
            _render_single_gauge(name, val, height=height, graded=graded)
    return grid


def render_emotional_gauges(
    scores: dict[str, float],
    *,
    height: str = "200px",
    selected: str | None = None,
) -> ui.element:
    """Grid of gauge dials for emotional/narrative scores (0–1 scale).

    Each metric becomes its own ``ui.echart`` widget arranged in a
    CSS grid so titles / detail labels never overlap (which they did
    when packing multiple gauge series into a single chart). Pass
    ``selected`` to render only one gauge (used by the affective
    dashboard dropdown).
    """
    if not scores:
        return ui.label("No scores available.").classes("text-grey text-caption")
    items = list(scores.items())
    return _render_gauge_grid(
        items, height=height, selected=selected, graded=False,
    )


# ── Affective metrics over fabula time ────────────────────────────

_AFFECT_COLORS = {
    "mystery": "#8E44AD",
    "narrative_tension": "#D8334A",
    "conflict": "#E67E22",
    "danger": "#C0392B",
    "causal_density": "#3A7BD5",
    # Engine-grade structural affects (DirectiveAssembly).
    "suspense": "#4A148C",
    "surprise": "#FFB300",
    "dramatic_irony": "#00838F",
}


def _event_overlay_series(
    ws: WorldStateV1,
    *,
    axis: str = "fabula",
    y_value: float = 0.0,
) -> dict | None:
    """Build a hover-only scatter series of every event keyed to an x-axis.

    Used to overlay event markers on timeline charts that otherwise
    only plot scalar curves. Each marker shows the event's
    description on hover so the reader can locate where in the
    narrative a given peak/trough falls.

    ``axis="fabula"`` keys markers to ``EventNode.fabula_time``;
    ``axis="syuzhet"`` keys to ``EventNode.syuzhet_index``.
    ``y_value`` is the y-coordinate at which markers sit (use the
    chart's y-axis floor — e.g. ``0`` for 0–1 affect plots).
    """
    if not ws.events:
        return None
    type_color = {
        "choice": "#3A7BD5",
        "outcome": "#6FBF3A",
        "revelation": "#F5B43C",
        "utterance": "#8E44AD",
    }
    data: list[dict] = []
    for evt in sorted(ws.events, key=lambda e: e.fabula_time):
        x = evt.fabula_time if axis == "fabula" else evt.syuzhet_index
        if x is None:
            continue
        data.append({
            "name": evt.id,
            "value": [x, y_value, evt.description[:140] or evt.id, evt.event_type],
            "itemStyle": {
                "color": type_color.get(evt.event_type, "#607D8B"),
                "opacity": 0.55,
                "borderColor": "#ffffff",
                "borderWidth": 1,
            },
        })
    if not data:
        return None
    # Shrink markers as the event count grows so they don't smear
    # into a continuous strip on dense timelines (~150+ events). Keep
    # opacity-based decluttering so peaks remain visible.
    n = len(data)
    if n <= 30:
        sym_size = 9
    elif n <= 80:
        sym_size = 7
    elif n <= 200:
        sym_size = 5
    else:
        sym_size = 3
    return {
        "name": "events",
        "type": "scatter",
        "data": data,
        "symbol": "diamond",
        "symbolSize": sym_size,
        "z": 5,
        "tooltip": {
            "trigger": "item",
            ":formatter": (
                "function (p) {"
                "  var v = p.value;"
                "  return '<b>' + p.name + '</b> (' + v[3] + ')<br/>'"
                "       + 'x=' + v[0] + '<br/>' + v[2];"
                "}"
            ),
        },
    }


def render_affective_timeseries(
    ws: WorldStateV1,
    *,
    samples: int = 12,
    height: str = "260px",
    fabula_cursor: int | None = None,
    syuzhet_cursor: int | None = None,
    entity_ids: list[str] | None = None,
    axis: str = "fabula",
    normalize: bool = False,
) -> ui.echart:
    """Multi-line chart of affective scores over fabula or syuzhet time.

    Snapshots the world at evenly-spaced cursors (via
    :func:`viz_helpers.affective_timeseries` or its syuzhet variant)
    and plots each metric as a separate line band. When
    ``entity_ids`` is supplied the engine-grade suspense / surprise /
    dramatic-irony / canonical mystery curves are included.
    Pass ``axis="syuzhet"`` to sample along reading order; the
    ``syuzhet_cursor`` becomes the active needle.

    Note on the ``surprise`` series: the timeseries view passes
    ``surprise_local=True`` to the scorer, so the line plots the
    Itti & Baldi (2009) **Bayesian Surprise** form
    ``D_KL(q_s || q_{s-1})`` — a per-step belief-update magnitude
    that spikes at revelations and decays in quiet stretches. The
    standalone surprise *gauge* uses the cumulative form
    ``D_KL(p || q)`` (remaining gap to truth) which decays
    monotonically. The two forms are not on the same scale; see
    ``docs/academic-foundations.md`` §3.3.
    """
    opts = affective_timeseries_options(
        ws,
        samples=samples,
        fabula_cursor=fabula_cursor,
        syuzhet_cursor=syuzhet_cursor,
        entity_ids=entity_ids,
        axis=axis,
        normalize=normalize,
    )
    if opts is None:
        return ui.label("No affective signal yet.").classes(
            "text-grey text-caption"
        )
    return ui.echart(opts).classes("w-full").style(f"height:{height}")


def affective_timeseries_options(
    ws: WorldStateV1,
    *,
    samples: int = 12,
    fabula_cursor: int | None = None,
    syuzhet_cursor: int | None = None,
    entity_ids: list[str] | None = None,
    axis: str = "fabula",
    normalize: bool = False,
) -> dict | None:
    """Pure options builder for :func:`render_affective_timeseries`.

    Returns ``None`` when there's no signal to plot. Splitting the
    options dict from the ECharts element creation lets callers reuse
    a stable :class:`ui.echart` and patch ``chart.options`` in place
    instead of tearing down the DOM on every cursor scrub.

    With ``normalize=True`` each series is independently min-max
    scaled to ``[0, 1]`` so trajectory shapes can be compared even
    when raw magnitudes differ by an order of magnitude (a flat
    metric stays at 0.5). The tooltip continues to show raw values.
    """
    from shadow_loom_ui.viz_helpers import (
        affective_timeseries,
        affective_timeseries_syuzhet,
    )

    is_syuzhet = axis == "syuzhet"
    if is_syuzhet:
        times, series = affective_timeseries_syuzhet(
            ws, samples=samples, entity_ids=entity_ids,
        )
        x_name = "syuzhet index"
        cursor = syuzhet_cursor
        cursor_label = (
            f"s={syuzhet_cursor}" if syuzhet_cursor is not None else None
        )
    else:
        times, series = affective_timeseries(
            ws, samples=samples, entity_ids=entity_ids,
        )
        x_name = "fabula time"
        cursor = fabula_cursor
        cursor_label = (
            f"t={fabula_cursor}" if fabula_cursor is not None else None
        )
    if not times or not series:
        return None

    plot_series = []
    for name, values in series.items():
        if normalize and values:
            lo = min(values)
            hi = max(values)
            span = hi - lo
            if span > 1e-9:
                norm_values = [(v - lo) / span for v in values]
            else:
                norm_values = [0.5 for _ in values]
            data = [
                [t, round(nv, 4), round(rv, 4)]
                for t, nv, rv in zip(times, norm_values, values)
            ]
        else:
            data = [[t, round(v, 4), round(v, 4)] for t, v in zip(times, values)]
        plot_series.append({
            "name": name.replace("_", " "),
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "lineStyle": {"width": 2},
            "areaStyle": {"opacity": 0.10},
            "color": _AFFECT_COLORS.get(name, "#3A7BD5"),
            "itemStyle": {"color": _AFFECT_COLORS.get(name, "#3A7BD5")},
            "encode": {"x": 0, "y": 1, "tooltip": [0, 1, 2]},
            "data": data,
        })

    mark_lines = []
    if cursor is not None and cursor_label is not None:
        mark_lines.append({
            "xAxis": cursor,
            "label": {"formatter": cursor_label, "color": "#D8334A"},
            "lineStyle": {"color": "#D8334A", "type": "dashed", "width": 1},
        })
    if mark_lines:
        plot_series[0]["markLine"] = {"symbol": "none", "data": mark_lines}

    # Event-locator overlay: hover any marker to see which event
    # falls at that x. Anchored to y=0 (charts are 0-1 bounded).
    overlay = _event_overlay_series(ws, axis=axis, y_value=0.0)
    if overlay is not None:
        plot_series.append(overlay)

    return {
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
            ":formatter": (
                "function (params) {"
                "  if (!params || !params.length) return '';"
                "  var x = params[0].axisValueLabel || params[0].axisValue;"
                "  var rows = ['<b>' + x + '</b>'];"
                "  params.forEach(function(p) {"
                "    if (!p.value) return;"
                "    var marker = p.marker || '';"
                "    var name = p.seriesName;"
                "    var v = p.value;"
                "    if (Array.isArray(v) && v.length >= 3) {"
                "      var raw = (typeof v[2] === 'number') ? v[2].toFixed(3) : v[2];"
                "      var disp = (typeof v[1] === 'number') ? v[1].toFixed(3) : v[1];"
                "      if (v[1] === v[2]) {"
                "        rows.push(marker + name + ': ' + raw);"
                "      } else {"
                "        rows.push(marker + name + ': ' + disp + '  <span style=\"color:#94a3b8\">(raw ' + raw + ')</span>');"
                "      }"
                "    } else {"
                "      rows.push(marker + name + ': ' + (Array.isArray(v) ? v[1] : v));"
                "    }"
                "  });"
                "  return rows.join('<br/>');"
                "}"
            ),
        },
        "legend": {
            # Scroll mode keeps long series lists usable instead of
            # truncating them off-screen at high entity/trait counts.
            "type": "scroll",
            "data": [s["name"] for s in plot_series],
            "textStyle": {"color": _CHART_TEXT},
            "top": 0,
        },
        "grid": {"top": 40, "bottom": 40, "left": 50, "right": 20},
        "xAxis": {
            "type": "value",
            "name": x_name,
            "nameLocation": "middle",
            "nameGap": 25,
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "name": "score (normalized)" if normalize else "score",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "series": plot_series,
    }


def update_chart_options(chart: ui.echart, options: dict) -> None:
    """Patch ``chart.options`` in place and re-render.

    Avoids the full DOM teardown of ``container.clear()`` +
    rebuilding the chart element. Use for slider-driven panels where
    only the data + cursor markers change between renders.
    """
    chart.options.clear()
    chart.options.update(options)
    chart.update()


# ── Physics trajectory (structural scalars over fabula time) ──────

def render_physics_trajectory(
    ws: WorldStateV1,
    focus_entity_ids: list[str] | None = None,
    *,
    samples: int = 12,
    height: str = "300px",
    fabula_cursor: int | None = None,
) -> ui.echart:
    """Multi-line chart of structural physics scalars over fabula_time.

    Backed by :func:`viz_helpers.physics_trajectory`, which calls the
    structural ``calculate_narrative_physics`` engine at evenly-spaced
    anchors with an ``ObservationQuery``. Pure graph math; no LLM
    invocations.
    """
    from shadow_loom_ui.viz_helpers import (
        PHYSICS_METRIC_COLORS,
        physics_trajectory,
    )

    times, series = physics_trajectory(
        ws, focus_entity_ids, samples=samples,
    )
    if not times or not series:
        return ui.label("No physics signal yet.").classes(
            "text-grey text-caption"
        )

    plot_series = []
    for name, values in series.items():
        if not any(values):
            continue
        plot_series.append({
            "name": name.replace("_", " "),
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "lineStyle": {"width": 2},
            "color": PHYSICS_METRIC_COLORS.get(name, "#3A7BD5"),
            "itemStyle": {"color": PHYSICS_METRIC_COLORS.get(name, "#3A7BD5")},
            "data": [[t, v] for t, v in zip(times, values)],
        })

    if not plot_series:
        return ui.label("No physics signal yet.").classes(
            "text-grey text-caption"
        )

    mark_lines = []
    if fabula_cursor is not None:
        mark_lines.append({
            "xAxis": fabula_cursor,
            "label": {"formatter": f"t={fabula_cursor}", "color": "#D8334A"},
            "lineStyle": {"color": "#D8334A", "type": "dashed", "width": 1},
        })
    if mark_lines:
        plot_series[0]["markLine"] = {"symbol": "none", "data": mark_lines}

    # Event-locator overlay anchored to the lowest plotted value so
    # readers can hover any beat to see which event coincides with
    # the trajectory inflection.
    floor = min(
        (v for s in plot_series for v in (p[1] for p in s["data"]) if v is not None),
        default=0.0,
    )
    overlay = _event_overlay_series(ws, axis="fabula", y_value=floor)
    if overlay is not None:
        plot_series.append(overlay)

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
        },
        "legend": {
            "data": [s["name"] for s in plot_series],
            "textStyle": {"color": _CHART_TEXT},
            "top": 0,
            "type": "scroll",
        },
        "grid": {"top": 50, "bottom": 40, "left": 50, "right": 20},
        "xAxis": {
            "type": "value",
            "name": "fabula time",
            "nameLocation": "middle",
            "nameGap": 25,
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "yAxis": {
            "type": "value",
            "name": "value",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "series": plot_series,
    }).classes("w-full").style(f"height:{height}")


# ── Event timeline scatter ────────────────────────────────────────

def render_event_timeline(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "300px",
    fabula_cursor: int | None = None,
    syuzhet_cursor: int | None = None,
    enable_brush: bool = False,
    on_brush: OnClick = None,
    pulse_cursor: bool = True,
) -> ui.echart:
    """Scatter plot: fabula_time (x) vs syuzhet_index (y), colored by type.

    Optional ``fabula_cursor`` / ``syuzhet_cursor`` draw a vertical /
    horizontal needle line at that value to anchor the active cursor.
    With ``pulse_cursor=True`` (default) a pulsing ``effectScatter``
    point is overlaid at the cursor intersection so it draws the eye.

    With ``enable_brush=True`` the chart gains a horizontal ``dataZoom``
    brush so users can focus a fabula range. The selected window is
    forwarded to ``on_brush`` (if given) as ``{"start": x0, "end": x1}``.
    """
    opts = event_timeline_options(
        ws,
        fabula_cursor=fabula_cursor,
        syuzhet_cursor=syuzhet_cursor,
        enable_brush=enable_brush,
        pulse_cursor=pulse_cursor,
    )
    if opts is None:
        return ui.label("No events.").classes("text-grey text-caption")
    chart = ui.echart(opts).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    if enable_brush and on_brush:
        # ECharts emits ``datazoom`` with start/end as percentages of
        # the axis range. We forward the raw event so callers can
        # translate it back to fabula values themselves.
        chart.on("datazoom", on_brush)
    return chart


def event_timeline_options(
    ws: WorldStateV1,
    *,
    fabula_cursor: int | None = None,
    syuzhet_cursor: int | None = None,
    enable_brush: bool = False,
    pulse_cursor: bool = True,
) -> dict | None:
    """Pure options builder for :func:`render_event_timeline`.

    Returns ``None`` if there are no events. See
    :func:`update_chart_options` for in-place patching.
    """
    scatter_data = ws_to_timeline_data(ws)
    if not scatter_data:
        return None

    mark_lines: list[dict] = []
    if fabula_cursor is not None:
        mark_lines.append({
            "xAxis": fabula_cursor,
            "label": {"formatter": f"t={fabula_cursor}", "color": "#D8334A"},
            "lineStyle": {"color": "#D8334A", "width": 2, "type": "dashed"},
        })
    if syuzhet_cursor is not None:
        mark_lines.append({
            "yAxis": syuzhet_cursor,
            "label": {"formatter": f"s={syuzhet_cursor}", "color": "#3A7BD5"},
            "lineStyle": {"color": "#3A7BD5", "width": 2, "type": "dashed"},
        })

    series: list[dict] = [{
        "type": "scatter",
        "name": "events",
        "data": scatter_data,
        # Scale point size so dense plots (150+ events) don't turn
        # into a single overplotted blob, while sparse plots remain
        # easily clickable.
        "symbolSize": (
            14 if len(scatter_data) <= 40
            else 10 if len(scatter_data) <= 120
            else 6
        ),
        "emphasis": {"scale": 1.6},
        "label": {"show": False},
    }]
    if mark_lines:
        series[0]["markLine"] = {
            "symbol": "none",
            "data": mark_lines,
            "silent": True,
        }

    if pulse_cursor and (fabula_cursor is not None or syuzhet_cursor is not None):
        ys = sorted(d["value"][1] for d in scatter_data)
        xs = sorted(d["value"][0] for d in scatter_data)
        med_y = ys[len(ys) // 2]
        med_x = xs[len(xs) // 2]
        px = fabula_cursor if fabula_cursor is not None else med_x
        py = syuzhet_cursor if syuzhet_cursor is not None else med_y
        series.append({
            "type": "effectScatter",
            "name": "cursor",
            "data": [{"value": [px, py], "name": "cursor"}],
            "symbolSize": 14,
            "showEffectOn": "render",
            "rippleEffect": {"period": 3, "scale": 4, "brushType": "stroke"},
            "itemStyle": {"color": "#D8334A"},
            "silent": True,
            "z": 10,
        })

    extra: dict = {}
    if enable_brush:
        extra["dataZoom"] = [
            {
                "type": "slider",
                "xAxisIndex": 0,
                "height": 18,
                "bottom": 6,
                "brushSelect": True,
                "borderColor": "#cbd5e1",
                "backgroundColor": "#f1f5f9",
                "fillerColor": "rgba(58,123,213,0.18)",
                "handleStyle": {"color": "#3A7BD5"},
                "textStyle": {"color": _CHART_TEXT, "fontSize": 9},
            },
            {
                "type": "inside",
                "xAxisIndex": 0,
            },
        ]

    return {
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "item",
            # Each scatter point carries ``description`` and
            # ``event_type`` (see ``ws_to_timeline_data``); surface
            # them so the user knows *what* the hovered event was
            # without clicking through.
            ":formatter": (
                "function(p){"
                " var d = p.data || {};"
                " var lines = ["
                "  '<b>' + (d.name || p.name) + '</b>'"
                " ];"
                " if(d.event_type){"
                "  lines.push('<i>' + d.event_type + '</i>');"
                " }"
                " if(d.value){"
                "  lines.push('fabula t=' + d.value[0] +"
                "    ', syuzhet s=' + d.value[1]);"
                " }"
                " if(d.description){"
                "  lines.push(d.description);"
                " }"
                " return lines.join('<br/>');"
                "}"
            ),
        },
        "grid": {
            "top": 40,
            "bottom": 60 if enable_brush else 40,
            "left": 60,
            "right": 30,
        },
        "xAxis": {
            "type": "value",
            "name": "Fabula Time",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Syuzhet Index",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": series,
        **extra,
    }


# ── Fabula↔Syuzhet displacement bars ───────────────────────────────

def render_displacement_chart(
    ws: WorldStateV1,
    *,
    height: str = "260px",
) -> ui.echart:
    """Diverging bar of ``syuzhet_index - fabula_time`` per event.

    Negative bars = flashbacks (event told later than it happened
    relative to the linear baseline). Positive bars = foreshadowing
    or in-order pacing where reading order leads fabula order.

    Bars are sorted by syuzhet order so the chart reads left-to-right
    as the reader experiences the story.
    """
    if not ws.events:
        return ui.label("No events.").classes("text-grey text-caption")

    evts = sorted(ws.events, key=lambda e: e.syuzhet_index)
    # Normalize fabula and syuzhet to [0, 1] so the displacement is
    # comparable across stories with different time scales.
    f_vals = [e.fabula_time for e in evts]
    s_vals = [e.syuzhet_index for e in evts]
    f_min, f_max = min(f_vals), max(f_vals)
    s_min, s_max = min(s_vals), max(s_vals)
    f_span = max(1, f_max - f_min)
    s_span = max(1, s_max - s_min)

    rows: list[list] = []
    labels: list[str] = []
    for e in evts:
        f_norm = (e.fabula_time - f_min) / f_span
        s_norm = (e.syuzhet_index - s_min) / s_span
        # Positive = reader gets info before fabula "would" suggest = foreshadowing.
        # Negative = reader gets info after = flashback / withheld.
        disp = round(s_norm - f_norm, 3)
        labels.append(f"s{e.syuzhet_index}")
        rows.append([
            labels[-1],
            disp,
            e.id,
            e.description[:80] if e.description else e.id,
        ])

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "item",
            # Per-bar tooltip: event id, the displacement value, and
            # a short description of what happened. Uses the
            # ``:formatter`` (colon-prefixed) NiceGUI convention so
            # the JS body is evaluated rather than rendered as a
            # literal string.
            ":formatter": (
                "function(p){"
                " var v = p.value || [];"
                " var disp = v[1];"
                " var dir = disp > 0 ? 'foreshadowed' :"
                "   (disp < 0 ? 'flashback' : 'in order');"
                " return '<b>' + v[2] + '</b><br/>'"
                "  + 'displacement: ' + disp + ' (' + dir + ')<br/>'"
                "  + (v[3] || '');"
                "}"
            ),
        },
        "grid": {"top": 30, "bottom": 50, "left": 50, "right": 20},
        "xAxis": {
            "type": "category",
            "data": labels,
            "name": "Reading order →",
            "nameLocation": "middle",
            "nameGap": 28,
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9, "interval": "auto"},
        },
        "yAxis": {
            "type": "value",
            "min": -1,
            "max": 1,
            "name": "← Flashback   |   Foreshadow →",
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "series": [{
            "type": "bar",
            "data": rows,
            "encode": {"x": 0, "y": 1, "tooltip": [2, 1, 3]},
            "itemStyle": {
                # JS callback: bars above zero (foreshadow) blue,
                # below zero (flashback) red. Colon prefix tells
                # NiceGUI to evaluate the body as JS rather than
                # serialise the string.
                ":color": (
                    "function(p){return p.value[1] >= 0"
                    "  ? '#3A7BD5' : '#D8334A';}"
                ),
            },
            "barMaxWidth": 18,
        }],
    }).classes("w-full").style(f"height:{height}")


# ── Entity state timeline (stepped trait evolution) ───────────────

def render_entity_state_timeline(
    entity_id: str,
    ws: WorldStateV1,
    *,
    height: str = "300px",
) -> ui.echart:
    """Stepped line chart of an entity's trait values over fabula_time."""
    data = entity_state_timeline_data(entity_id, ws)
    if not data["times"]:
        return ui.label("No temporal data.").classes("text-grey text-caption")

    ent = ws.entities.get(entity_id)
    title = ent.name if ent else entity_id

    series = []
    colors = CHART_COLORS
    for i, (trait_name, values) in enumerate(data["series"].items()):
        series.append({
            "name": trait_name,
            "type": "line",
            "step": "middle",
            "data": values,
            "lineStyle": {"width": 2},
            "symbol": "circle",
            "symbolSize": 6,
            "itemStyle": {"color": colors[i % len(colors)]},
        })

    # Overlay event markers at each fabula tick that has events, so the
    # user can see *what happened* at a step change without leaving
    # the chart. Anchored at y=0 (the trait scale midpoint).
    from shadow_loom_ui.viz_helpers import event_overlay_series
    overlay = event_overlay_series(ws, data["times"], y_value=0.0)
    if overlay is not None:
        series.append(overlay)

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "title": {
            "text": title,
            "left": "center",
            "textStyle": {"color": _CHART_TEXT, "fontSize": 12},
        },
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis"},
        "legend": {
            "type": "scroll",
            "data": list(data["series"].keys()),
            "textStyle": {"color": _CHART_TEXT},
            "top": 24,
        },
        "grid": {"top": 64, "bottom": 30, "left": 50, "right": 20},
        "xAxis": {
            "type": "category",
            "data": [str(t) for t in data["times"]],
            "name": "Fabula Time",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Trait Value",
            "min": -1,
            "max": 1,
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": series,
    }).classes("w-full").style(f"height:{height}")


# ── Relationship timeline (dyad) ──────────────────────────────────

def render_relationship_timeline(
    ws: WorldStateV1,
    ent_a: str,
    ent_b: str,
    *,
    height: str = "300px",
) -> ui.echart:
    """Stepped line chart of affinity / fear / power between two entities."""
    from shadow_loom_ui.viz_helpers import relationship_timeline_data

    data = relationship_timeline_data(ws, ent_a, ent_b)
    if not data["times"]:
        return ui.label("No relationship data.").classes("text-grey text-caption")

    name_a = ws.entities[ent_a].name if ent_a in ws.entities else ent_a
    name_b = ws.entities[ent_b].name if ent_b in ws.entities else ent_b
    palette = {
        "affinity": "#3A7BD5",
        "fear": "#D8334A",
        "power_dynamic": "#F5B43C",
    }
    series = [
        {
            "name": metric,
            "type": "line",
            "step": "middle",
            "data": values,
            "lineStyle": {"width": 2},
            "symbol": "circle",
            "symbolSize": 6,
            "itemStyle": {"color": palette.get(metric, CHART_COLORS[i % len(CHART_COLORS)])},
        }
        for i, (metric, values) in enumerate(data["series"].items())
    ]

    # Overlay event markers (see ``render_entity_state_timeline``).
    from shadow_loom_ui.viz_helpers import event_overlay_series
    overlay = event_overlay_series(ws, data["times"], y_value=0.0)
    if overlay is not None:
        series.append(overlay)

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "title": {
            "text": f"{name_a} ⇄ {name_b}",
            "left": "center",
            "textStyle": {"color": _CHART_TEXT, "fontSize": 12},
        },
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis"},
        "legend": {
            "type": "scroll",
            "data": list(data["series"].keys()),
            "textStyle": {"color": _CHART_TEXT},
            "top": 24,
        },
        "grid": {"top": 64, "bottom": 30, "left": 50, "right": 20},
        "xAxis": {
            "type": "category",
            "data": [str(t) for t in data["times"]],
            "name": "Fabula Time",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Value",
            "min": -1,
            "max": 1,
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": series,
    }).classes("w-full").style(f"height:{height}")


# ── World-trait timeline ──────────────────────────────────────────

def render_world_trait_timeline(
    ws: WorldStateV1,
    world_id: str,
    *,
    height: str = "300px",
) -> ui.echart:
    """Stepped line chart of a world trait's magnitude over fabula_time."""
    from shadow_loom_ui.viz_helpers import world_trait_timeline_data

    data = world_trait_timeline_data(ws, world_id)
    if not data["times"]:
        return ui.label("No world-trait data.").classes("text-grey text-caption")

    wt = ws.world_traits.get(world_id)
    title = wt.name if wt else world_id

    series_list: list[dict] = [
        {
            "name": "intensity",
            "type": "line",
            "step": "middle",
            "data": data["value"],
            "itemStyle": {"color": "#2EA6A0"},
            "lineStyle": {"width": 2},
            "areaStyle": {"opacity": 0.15},
        },
        {
            "name": "inertia",
            "type": "line",
            "step": "middle",
            "data": data["inertia"],
            "itemStyle": {"color": "#8A5CF0"},
            "lineStyle": {"width": 2, "type": "dashed"},
        },
    ]
    # Overlay event markers (see ``render_entity_state_timeline``).
    # World traits live on a 0–1 magnitude scale so we anchor the
    # markers at 0.5 for visibility.
    from shadow_loom_ui.viz_helpers import event_overlay_series
    overlay = event_overlay_series(ws, data["times"], y_value=0.5)
    if overlay is not None:
        series_list.append(overlay)

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "title": {
            "text": title,
            "left": "center",
            "textStyle": {"color": _CHART_TEXT, "fontSize": 12},
        },
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis"},
        "legend": {
            "data": ["intensity", "inertia"],
            "textStyle": {"color": _CHART_TEXT},
            "top": 24,
        },
        "grid": {"top": 64, "bottom": 30, "left": 50, "right": 20},
        "xAxis": {
            "type": "category",
            "data": [str(t) for t in data["times"]],
            "name": "Fabula Time",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Magnitude",
            "min": 0,
            "max": 1,
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": series_list,
    }).classes("w-full").style(f"height:{height}")


# ── Version tree ──────────────────────────────────────────────────

def render_version_tree(
    tree_data: list[dict],
    current_version_id: int | None = None,
    *,
    on_click: OnClick = None,
    height: str | None = None,
    orient: str = "vertical",
) -> ui.echart:
    """Tree layout of project version history.

    ``orient='radial'`` switches to a radial layout (root in centre),
    helpful when many shadow branches diverge from a single factual
    spine.

    ``height`` defaults to a node-count driven minimum so deep or
    wide histories stay legible.
    """
    root = version_tree_to_echart_data(tree_data, current_version_id)
    if not root:
        return ui.label("No versions.").classes("text-grey text-caption")

    if height is None:
        # Tree depth/breadth correlates with len(tree_data); use that
        # as a proxy for required height.
        n = len(tree_data) if tree_data else 0
        height = f"{max(300, 18 * n + 80)}px"

    is_radial = orient == "radial"
    series_layout: dict = (
        {"layout": "radial", "symbol": "emptyCircle"}
        if is_radial
        else {"orient": "vertical"}
    )

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "tree",
            "data": [root],
            "top": 20,
            "bottom": 20,
            "left": 40,
            "right": 40,
            "symbolSize": 12,
            "edgeShape": "curve" if is_radial else "polyline",
            "edgeForkPosition": "50%",
            "label": {
                "position": "top" if not is_radial else "left",
                "verticalAlign": "middle",
                "align": "center",
                "fontSize": 10,
                "color": _CHART_TEXT,
            },
            "leaves": {
                "label": {"position": "bottom" if not is_radial else "right",
                          "verticalAlign": "middle", "align": "center"},
            },
            "lineStyle": {"color": "#555", "width": 1.5},
            "emphasis": {"focus": "descendant"},
            "expandAndCollapse": False,
            "animationDuration": 400,
            **series_layout,
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Causal force graph ────────────────────────────────────────────

def render_causal_force_graph(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
    layout: str = "force",
    highlight_edge_ids: set[str] | None = None,
) -> ui.echart:
    """Force-directed graph of causal edges, thickness = causal_force.

    ``layout`` options:
      - ``"force"`` (default): physics-based layout, good for medium graphs.
      - ``"circular"``: nodes on a ring, ideal for symmetry inspection.
      - ``"cartesian"``: anchors events on (fabula_time, syuzhet_index)
        axes so the temporal flow of causality is preserved.

    For >80 nodes the force layout auto-tunes its repulsion / friction
    (Webkit-dep style) so the graph doesn't explode into a hairball.
    """
    if layout == "cartesian":
        return _render_causal_cartesian(
            ws, on_click=on_click, height=height,
            highlight_edge_ids=highlight_edge_ids,
        )

    nodes, links, cats = ws_to_causal_force_data(ws)
    if not nodes:
        return ui.label("No causal topology.").classes("text-grey q-pa-md")

    if highlight_edge_ids:
        for lk in links:
            eid = f"{lk.get('source')}->{lk.get('target')}"
            if eid in highlight_edge_ids:
                ls = dict(lk.get("lineStyle", {}))
                ls["color"] = "#FF3D00"
                ls["width"] = max(float(ls.get("width", 2)) + 2, 4)
                ls["opacity"] = 1.0
                lk["lineStyle"] = ls

    n = len(nodes)
    if layout == "circular":
        series_extra: dict = {
            "layout": "circular",
            "circular": {"rotateLabel": True},
        }
    else:
        # Auto-tune for large graphs (Webkit-dep inspiration).
        if n > 200:
            force = {"repulsion": 900, "gravity": 0.05, "edgeLength": [40, 220], "friction": 0.4}
            curveness = 0.25
        elif n > 80:
            force = {"repulsion": 600, "gravity": 0.08, "edgeLength": [50, 200], "friction": 0.5}
            curveness = 0.2
        else:
            force = {"repulsion": 400, "gravity": 0.1, "edgeLength": [60, 180], "friction": 0.6}
            curveness = 0.15
        series_extra = {"layout": "force", "force": force}

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "legend": {
            "type": "scroll",
            "data": [c["name"] for c in cats],
            "textStyle": {"color": _CHART_TEXT},
            "top": 0,
            "right": 10,
            "orient": "vertical",
        },
        "series": [{
            "type": "graph",
            "roam": True,
            "draggable": True,
            "emphasis": {"focus": "adjacency"},
            "categories": cats,
            "data": nodes,
            "links": links,
            "label": {
                "show": n <= 60,  # hide labels on dense graphs
                "position": "right",
                "fontSize": 9,
                "color": _CHART_TEXT,
            },
            "lineStyle": {
                "curveness": curveness if layout != "circular" else 0.3,
                "opacity": 0.7 if n <= 80 else 0.45,
            },
            **series_extra,
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


def _render_causal_cartesian(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
    highlight_edge_ids: set[str] | None = None,
) -> ui.echart:
    """Causal graph anchored on (fabula_time, syuzhet_index) axes.

    Inspired by the ECharts ``graph-life-expectancy`` example: nodes
    keep their temporal coordinates while edges curve between them so
    the visual reading order matches the story order.
    """
    nodes, links = ws_to_causal_cartesian_data(ws)
    if not nodes:
        return ui.label("No causal topology.").classes("text-grey q-pa-md")

    if highlight_edge_ids:
        for lk in links:
            eid = f"{lk.get('source')}->{lk.get('target')}"
            if eid in highlight_edge_ids:
                ls = dict(lk.get("lineStyle", {}))
                ls["color"] = "#FF3D00"
                ls["width"] = max(float(ls.get("width", 2)) + 2, 4)
                ls["opacity"] = 1.0
                lk["lineStyle"] = ls

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "grid": {"top": 30, "bottom": 50, "left": 60, "right": 30},
        "xAxis": {
            "type": "value",
            "name": "Fabula Time →",
            "nameLocation": "middle",
            "nameGap": 28,
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 11},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "yAxis": {
            "type": "value",
            "name": "↑ Syuzhet (reading order)",
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 11},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "series": [{
            "type": "graph",
            "coordinateSystem": "cartesian2d",
            "data": nodes,
            "links": links,
            "roam": True,
            "draggable": False,
            "emphasis": {"focus": "adjacency"},
            "label": {"show": len(nodes) <= 40, "position": "top",
                       "fontSize": 9, "color": _CHART_TEXT},
            "lineStyle": {"opacity": 0.55, "curveness": 0.15},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Epistemic map (who-knows-what matrix) ─────────────────────────

def render_epistemic_map(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
) -> ui.echart:
    """Heatmap: entity (y) × belief target (x), colored by confidence."""
    ent_names, target_names, data = ws_to_epistemic_data(ws)
    if not data:
        return ui.label("No belief data.").classes("text-grey q-pa-md")

    # Drop per-cell text and thin axis ticks once the matrix grows
    # past what the eye can resolve. Tooltip still gives the exact
    # confidence on hover.
    n_cells = max(len(ent_names), len(target_names))
    show_labels = n_cells <= 12
    x_interval = 0 if len(target_names) <= 25 else max(0, len(target_names) // 25)
    y_interval = 0 if len(ent_names) <= 25 else max(0, len(ent_names) // 25)

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "position": "top"},
        "grid": {"top": 30, "bottom": 80, "left": 100, "right": 30},
        "xAxis": {
            "type": "category",
            "data": target_names,
            "name": "Belief About",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {
                "rotate": 45,
                "color": _CHART_TEXT,
                "fontSize": 10,
                "interval": x_interval,
            },
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": ent_names,
            "name": "Believer",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {
                "color": _CHART_TEXT,
                "fontSize": 10,
                "interval": y_interval,
            },
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": 0,
            "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
            "inRange": {"color": ["#1E2A3A", "#3A7BD5", "#6FBF3A"]},
            "textStyle": {"color": _CHART_TEXT},
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": show_labels, "fontSize": 9, "color": "#eee"},
            "emphasis": {"itemStyle": {"shadowBlur": 10}},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


def render_entity_belief_chart(
    ws: WorldStateV1,
    entity_id: str,
    *,
    height: str = "260px",
) -> ui.element:
    """One character's beliefs as a textual card list.

    A bar chart hid the most important part — the actual claim. This
    renders each belief as a row containing:

      • the **perceived_state** text (the *what*, prominent),
      • a small badge for the **target** (the *who/what about*),
      • a confidence bar coloured by conviction band (grey → blue →
        green) with the numeric value,
      • an inertia chip (how stubborn the belief is),
      • the fabula tick the belief was established at.

    Designed to be tiled in :func:`render_epistemic_grid` — each card
    becomes one believer's panel.
    """
    from shadow_loom_ui.viz_helpers import ws_to_entity_belief_rows

    rows = ws_to_entity_belief_rows(ws, entity_id)
    ent = ws.entities.get(entity_id)
    title = ent.name if ent else entity_id

    # Conviction colour bands.
    def _band(c: float) -> str:
        if c < 0.34:
            return "#94a3b8"   # grey — uncertain
        if c < 0.67:
            return "#3A7BD5"   # blue — moderate
        return "#6FBF3A"       # green — strong

    def _band_label(c: float) -> str:
        if c < 0.34:
            return "uncertain"
        if c < 0.67:
            return "moderate"
        return "convinced"

    container = ui.column().classes(
            "w-full bg-white"
        ).style(f"min-height:{height}")
    with container:
        # Header strip — name + belief count + "live" status hint.
        with ui.row().classes(
            "w-full items-baseline justify-between px-3 py-2 "
            "border-b border-slate-200 bg-slate-50"
        ):
            ui.label(title).classes("text-sm font-semibold text-slate-800")
            ui.label(
                f"{len(rows)} belief{'s' if len(rows) != 1 else ''}"
            ).classes("text-xs text-slate-500")

        if not rows:
            ui.label("No beliefs.").classes(
                "text-xs text-slate-400 italic px-3 py-3"
            )
            return container

        # Scrollable belief list — bounded so tall card grids don't
        # explode vertically.
        with ui.column().classes(
            "w-full gap-2 px-3 py-2 overflow-y-auto"
        ).style("max-height: 360px"):
            for r in rows:
                conf = float(r["confidence"])
                inertia = float(r["inertia"])
                color = _band(conf)
                with ui.column().classes(
                    "w-full gap-1 p-2 rounded-lg border border-slate-200 "
                    "bg-slate-50/60 hover:bg-slate-100/60 transition-colors"
                ):
                    # Top line: perceived_state — the actual claim.
                    ui.label(
                        f"\u201C{r['perceived_state']}\u201D"
                    ).classes(
                        "text-sm text-slate-800 leading-snug"
                    )
                    # Meta row: target + conviction + inertia + established
                    with ui.row().classes(
                        "w-full items-center gap-2 text-xs"
                    ):
                        ui.label(
                            f"about {r['target_name']}"
                        ).classes(
                            "text-slate-500 italic truncate flex-grow"
                        )
                        ui.label(_band_label(conf)).classes(
                            "px-2 py-0.5 rounded-full text-white font-mono"
                        ).style(f"background-color: {color}")
                        ui.label(
                            f"conf {conf:.2f}"
                        ).classes(
                            "px-2 py-0.5 rounded-full bg-slate-200 "
                            "text-slate-700 font-mono"
                        )
                        ui.label(
                            f"inertia {inertia:.2f}"
                        ).classes(
                            "px-2 py-0.5 rounded-full bg-slate-200 "
                            "text-slate-700 font-mono"
                        )
                        if r["established"]:
                            ui.label(
                                f"t={r['established']}"
                            ).classes(
                                "px-2 py-0.5 rounded-full bg-slate-200 "
                                "text-slate-600 font-mono"
                            )
                    # Provenance row \u2014 how the belief was acquired.
                    via_chn = r.get("acquired_via_channel_name") or r.get(
                        "acquired_via_channel_id"
                    )
                    via_evt = r.get("acquired_via_event_label") or r.get(
                        "acquired_via_event_id"
                    )
                    if via_chn or via_evt:
                        with ui.row().classes(
                            "w-full items-center gap-2 text-xs text-slate-500"
                        ):
                            if via_chn:
                                ui.label(f"via {via_chn}").classes(
                                    "px-2 py-0.5 rounded-full "
                                    "bg-violet-50 text-violet-700 "
                                    "border border-violet-200"
                                )
                            if via_evt:
                                ui.label(f"\u2190 {via_evt}").classes(
                                    "italic truncate"
                                )
                    # Confidence bar \u2014 visual reinforcement of the band.
                    with ui.element("div").classes(
                        "w-full h-1.5 rounded-full bg-slate-200 overflow-hidden"
                    ):
                        ui.element("div").classes("h-full rounded-full").style(
                            f"width: {int(conf * 100)}%; background-color: {color}"
                        )
    return container


def render_epistemic_grid(
    ws: WorldStateV1,
    *,
    selected_ids: list[str] | None = None,
    chart_height: str = "260px",
) -> ui.element:
    """Tiled grid of per-character belief charts.

    Renders one ``render_entity_belief_chart`` per believer, two-up on
    medium screens and three-up on wide screens. ``selected_ids``
    filters which believers to show (None = all).
    """
    from shadow_loom_ui.viz_helpers import list_believers

    believers = list_believers(ws)
    if selected_ids:
        sel = set(selected_ids)
        believers = [b for b in believers if b[0] in sel]

    container = ui.column().classes("w-full gap-3")
    with container:
        if not believers:
            ui.label(
                "No characters hold beliefs in this world model."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container
        # CSS grid: responsive 1/2/3 columns.
        grid = ui.element("div").classes(
            "w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"
        )
        with grid:
            for eid, _name, _count in believers:
                with ui.element("div").classes(
                    "border border-slate-200 rounded-xl bg-white shadow-sm "
                    "overflow-hidden"
                ):
                    render_entity_belief_chart(ws, eid, height=chart_height)
    return container


def render_world_trait_state_card(
    ws: WorldStateV1,
    world_id: str,
) -> ui.element:
    """Snapshot card for one global trait at the current ``ws`` cursor.

    Designed to mirror the per-believer belief card: shows the trait's
    *current* magnitude and inertia (i.e. the values on ``ws`` after
    ``snapshot_world_at``), the affected domains, the prose
    description, and a compact inline sparkline of the magnitude
    history so the reader can still see the trajectory without losing
    the headline state-at-cursor reading.
    """
    from shadow_loom_ui.viz_helpers import world_trait_timeline_data

    wt = ws.world_traits.get(world_id)
    name = wt.name if wt else world_id
    mag_val = float(wt.magnitude.value) if wt and wt.magnitude else 0.0
    inertia = float(wt.magnitude.inertia) if wt and wt.magnitude else 0.0
    description = (wt.description or "").strip() if wt else ""
    domains = ", ".join(getattr(wt, "affected_domains", []) or []) if wt else ""
    category = getattr(wt, "category", "") if wt else ""

    # Magnitude band colours match the belief-card conviction palette.
    if mag_val < 0.34:
        bar_color = "#94a3b8"
        band_label = "low"
    elif mag_val < 0.67:
        bar_color = "#3A7BD5"
        band_label = "moderate"
    else:
        bar_color = "#6FBF3A"
        band_label = "dominant"

    container = ui.column().classes("w-full bg-white")
    with container:
        # Header strip mirrors render_entity_belief_chart.
        with ui.row().classes(
            "w-full items-baseline justify-between px-3 py-2 "
            "border-b border-slate-200 bg-slate-50"
        ):
            ui.label(name).classes("text-sm font-semibold text-slate-800")
            ui.label(category).classes(
                "text-xs text-slate-500 italic"
            )

        with ui.column().classes("w-full gap-2 px-3 py-3"):
            if description:
                ui.label(description).classes(
                    "text-sm text-slate-700 leading-snug"
                )

            # Big magnitude readout + bar.
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(f"{mag_val:.2f}").classes(
                    "text-2xl font-mono font-semibold text-slate-800 w-16"
                )
                with ui.column().classes("flex-grow gap-1"):
                    with ui.row().classes(
                        "w-full items-center gap-2 text-xs"
                    ):
                        ui.label(band_label).classes(
                            "px-2 py-0.5 rounded-full text-white font-mono"
                        ).style(f"background-color: {bar_color}")
                        ui.label(f"inertia {inertia:.2f}").classes(
                            "px-2 py-0.5 rounded-full bg-slate-200 "
                            "text-slate-700 font-mono"
                        )
                        if domains:
                            ui.label(domains).classes(
                                "text-slate-500 truncate"
                            )
                    with ui.element("div").classes(
                        "w-full h-2 rounded-full bg-slate-200 overflow-hidden"
                    ):
                        ui.element("div").classes(
                            "h-full rounded-full"
                        ).style(
                            f"width: {int(max(0.0, min(1.0, mag_val)) * 100)}%; "
                            f"background-color: {bar_color}"
                        )

            # Inline sparkline of the magnitude trajectory — kept small
            # so the snapshot reading dominates.
            try:
                tl = world_trait_timeline_data(ws, world_id)
            except Exception:
                tl = {"times": [], "value": [], "inertia": []}
            if tl["times"] and len(tl["times"]) > 1:
                ui.echart({
                    "backgroundColor": "transparent",
                    "grid": {
                        "top": 6, "bottom": 18, "left": 28, "right": 8,
                    },
                    "xAxis": {
                        "type": "category",
                        "data": [str(t) for t in tl["times"]],
                        "axisLabel": {
                            "color": "#94a3b8", "fontSize": 8,
                            "interval": "auto",
                        },
                        "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
                        "axisTick": {"show": False},
                    },
                    "yAxis": {
                        "type": "value", "min": 0, "max": 1,
                        "axisLabel": {"color": "#94a3b8", "fontSize": 8},
                        "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
                    },
                    "tooltip": {**_CHART_TOOLTIP, "trigger": "axis"},
                    "series": [{
                        "type": "line",
                        "step": "middle",
                        "data": tl["value"],
                        "showSymbol": False,
                        "lineStyle": {"width": 1.5, "color": bar_color},
                        "areaStyle": {"opacity": 0.15, "color": bar_color},
                    }],
                }).classes("w-full").style("height:64px")
    return container


def render_world_state_grid(
    ws: WorldStateV1,
    *,
    selected_ids: list[str] | None = None,
    chart_height: str = "300px",
) -> ui.element:
    """Tiled grid of per-world-trait **snapshot** cards at the cursor.

    Mirrors :func:`render_epistemic_grid` but for
    ``WorldStateV1.world_traits``: one snapshot card per global trait
    rendered by :func:`render_world_trait_state_card`. ``ws`` is
    expected to already be snapshotted to the active fabula cursor by
    the caller (``snapshot_world_at``), so each card reads the
    magnitude / inertia *at that time* directly. ``selected_ids``
    filters which world traits to show (``None`` = all).

    ``chart_height`` is accepted for back-compat but ignored — snapshot
    cards size themselves to content.
    """
    from shadow_loom_ui.viz_helpers import list_world_traits

    traits = list_world_traits(ws)
    if selected_ids:
        sel = set(selected_ids)
        traits = [t for t in traits if t[0] in sel]

    container = ui.column().classes("w-full gap-3")
    with container:
        if not traits:
            ui.label(
                "No world traits in this world model."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container
        grid = ui.element("div").classes(
            "w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"
        )
        with grid:
            for wid, _wname in traits:
                with ui.element("div").classes(
                    "border border-slate-200 rounded-xl bg-white shadow-sm "
                    "overflow-hidden"
                ):
                    render_world_trait_state_card(ws, wid)
    return container


# ── Relationship snapshot cards (affinity / fear / power) ─────────

def render_relationship_state_card(
    ws: WorldStateV1,
    rel,  # RelationshipEdge — typed loosely to avoid an import cycle.
) -> ui.element:
    """Per-dyad card showing affinity / fear / power_dynamic at the cursor.

    Each axis is rendered as a labelled bar:
      • **affinity** in [-1, +1] (red ↔ green, centered),
      • **fear** in [0, 1] (cool grey → amber),
      • **power_dynamic** in [-1, +1] (purple ↔ teal, centered, where
        positive = source dominates target).

    Only axes that the model has actually observed (``rel.metrics``)
    are shown — an unobserved axis renders as a faint "—" so the
    reader can distinguish "deliberately zero" from "no data".
    """
    src = ws.entities.get(rel.source_entity_id)
    tgt = ws.entities.get(rel.target_entity_id)
    src_name = src.name if src else rel.source_entity_id
    tgt_name = tgt.name if tgt else rel.target_entity_id

    def _signed_bar(value: float, *, neg_color: str, pos_color: str) -> None:
        # Two-half bar centered on 0. ``value`` in [-1, 1].
        v = max(-1.0, min(1.0, float(value)))
        with ui.element("div").classes(
            "w-full h-2 rounded-full bg-slate-200 overflow-hidden flex"
        ):
            with ui.element("div").classes(
                "h-full flex items-center justify-end"
            ).style("width: 50%"):
                if v < 0:
                    ui.element("div").classes("h-full rounded-l-full").style(
                        f"width: {int(abs(v) * 100)}%; "
                        f"background-color: {neg_color}"
                    )
            with ui.element("div").classes(
                "h-full flex items-center"
            ).style("width: 50%"):
                if v > 0:
                    ui.element("div").classes("h-full rounded-r-full").style(
                        f"width: {int(v * 100)}%; "
                        f"background-color: {pos_color}"
                    )

    def _unsigned_bar(value: float, *, color: str) -> None:
        v = max(0.0, min(1.0, float(value)))
        with ui.element("div").classes(
            "w-full h-2 rounded-full bg-slate-200 overflow-hidden"
        ):
            ui.element("div").classes("h-full rounded-full").style(
                f"width: {int(v * 100)}%; background-color: {color}"
            )

    container = ui.column().classes("w-full bg-white")
    with container:
        with ui.row().classes(
            "w-full items-baseline justify-between px-3 py-2 "
            "border-b border-slate-200 bg-slate-50"
        ):
            ui.label(f"{src_name}  →  {tgt_name}").classes(
                "text-sm font-semibold text-slate-800 truncate"
            )
            ui.label(f"t={rel.last_updated_fabula}").classes(
                "text-xs font-mono text-slate-500"
            )

        with ui.column().classes("w-full gap-2 px-3 py-3"):
            for axis_name, label, signed, neg_color, pos_color in (
                ("affinity",      "Affinity",      True,  "#dc2626", "#16a34a"),
                ("fear",          "Fear",          False, "#f59e0b", "#f59e0b"),
                ("power_dynamic", "Power dynamic", True,  "#8a5cf0", "#0d9488"),
            ):
                m = rel.metrics.get(axis_name)
                with ui.row().classes(
                    "w-full items-center gap-2 text-xs"
                ):
                    ui.label(label).classes(
                        "text-slate-600 w-28 shrink-0"
                    )
                    if m is None or not getattr(m, "observed", True):
                        ui.label("—").classes(
                            "font-mono text-slate-400 w-12 text-right"
                        )
                        with ui.element("div").classes(
                            "flex-grow h-2 rounded-full bg-slate-100"
                        ):
                            pass
                    else:
                        ui.label(f"{float(m.value):+.2f}" if signed
                                 else f"{float(m.value):.2f}").classes(
                            "font-mono text-slate-800 w-12 text-right"
                        )
                        with ui.column().classes("flex-grow gap-0"):
                            if signed:
                                _signed_bar(
                                    m.value,
                                    neg_color=neg_color,
                                    pos_color=pos_color,
                                )
                            else:
                                _unsigned_bar(m.value, color=pos_color)
                        ui.label(
                            f"i {float(getattr(m, 'inertia', 0.0)):.2f}"
                        ).classes(
                            "font-mono text-slate-500 w-10 text-right"
                        )
    return container


def render_relationship_state_grid(
    ws: WorldStateV1,
    *,
    selected_pairs: list[tuple[str, str]] | None = None,
) -> ui.element:
    """Tiled grid of per-dyad relationship snapshot cards.

    ``ws`` is expected to be snapshotted to the active fabula cursor
    by the caller, so each card reflects the relationship state at
    that time. ``selected_pairs`` (optional) filters to specific
    ``(source_id, target_id)`` dyads.
    """
    rels = list(ws.social_topology)
    if selected_pairs:
        wanted = {tuple(p) for p in selected_pairs}
        rels = [
            r for r in rels
            if (r.source_entity_id, r.target_entity_id) in wanted
        ]
    # Sort by source then target name for stable grid ordering.
    def _sort_key(r):
        a = ws.entities.get(r.source_entity_id)
        b = ws.entities.get(r.target_entity_id)
        return (
            (a.name if a else r.source_entity_id).lower(),
            (b.name if b else r.target_entity_id).lower(),
        )
    rels.sort(key=_sort_key)

    container = ui.column().classes("w-full gap-3")
    with container:
        if not rels:
            ui.label(
                "No relationships in this world model."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container
        grid = ui.element("div").classes(
            "w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"
        )
        with grid:
            for rel in rels:
                with ui.element("div").classes(
                    "border border-slate-200 rounded-xl bg-white shadow-sm "
                    "overflow-hidden"
                ):
                    render_relationship_state_card(ws, rel)
    return container


# ── ThemeRiver (multi-entity trait evolution) ─────────────────────

def render_theme_river(
    ws: WorldStateV1,
    *,
    trait_names: list[str] | None = None,
    max_entities: int = 6,
    height: str = "400px",
) -> ui.echart:
    """ThemeRiver showing entity trait evolution as flowing bands."""
    data = ws_to_theme_river_data(ws, trait_names=trait_names, max_entities=max_entities)
    if not data:
        return ui.label("No temporal data for ThemeRiver.").classes("text-grey q-pa-md")

    # Extract legend entries
    legends = sorted({d[2] for d in data})

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "color": CHART_COLORS,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis"},
        "legend": {
            "data": legends,
            "textStyle": {"color": _CHART_TEXT},
            "top": 0,
            "type": "scroll",
        },
        "singleAxis": {
            "type": "category",
            "bottom": 50,
            "top": 50,
            "axisLabel": {"color": _CHART_TEXT},
            "axisLine": {"lineStyle": {"color": "#555"}},
        },
        "series": [{
            "type": "themeRiver",
            "data": data,
            "label": {"show": False},
            "emphasis": {"itemStyle": {"shadowBlur": 20, "shadowColor": "rgba(0,0,0,0.3)"}},
        }],
    }).classes("w-full").style(f"height:{height}")


# ── Chord diagram (relationship reciprocity) ─────────────────────

def render_chord_diagram(
    ws: WorldStateV1,
    *,
    metric: str = "affinity",
    height: str = "400px",
    use_v6_chord: bool = True,
    max_entities: int = 20,
) -> ui.echart:
    """Chord diagram showing bidirectional relationship strengths.

    With ``use_v6_chord=True`` (default) we emit ECharts v6's native
    ``chord`` series — gradient ribbons, ``minAngle`` filtering, and
    cleaner colouring. Older ECharts builds fall back to the circular
    ``graph`` layout automatically (the option spec is JSON; an unknown
    ``type`` is skipped silently).

    ``max_entities`` caps the diagram at the top-N entities by total
    relationship strength. Past ~20 the ribbons turn into an
    indecipherable ball; pass a larger value to override.
    """
    names, matrix = ws_to_chord_data(ws, metric=metric)
    if not names or all(all(v == 0 for v in row) for row in matrix):
        return ui.label("No relationship data for chord.").classes("text-grey q-pa-md")

    # Size each node by total outgoing strength so isolated entities are visible.
    out_strength: dict[str, float] = {n: 0.0 for n in names}
    for i, row in enumerate(matrix):
        out_strength[names[i]] = sum(row)
    max_out = max(out_strength.values()) or 1.0

    # Top-N filter by combined in/out strength so the busiest cast
    # members survive; rebuild ``names``/``matrix`` accordingly.
    truncated = False
    if len(names) > max_entities:
        in_strength = [sum(row[i] for row in matrix) for i in range(len(names))]
        combined = [
            (i, out_strength[names[i]] + in_strength[i]) for i in range(len(names))
        ]
        keep_idx = sorted(
            [i for i, _ in sorted(combined, key=lambda x: x[1], reverse=True)[:max_entities]]
        )
        names = [names[i] for i in keep_idx]
        matrix = [[matrix[i][j] for j in keep_idx] for i in keep_idx]
        out_strength = {n: 0.0 for n in names}
        for i, row in enumerate(matrix):
            out_strength[names[i]] = sum(row)
        max_out = max(out_strength.values()) or 1.0
        truncated = True

    show_labels = len(names) <= 24

    if use_v6_chord:
        # v6 chord series: nodes are named slices, links are matrix entries.
        chord_nodes = [
            {
                "name": n,
                "itemStyle": {"color": CHART_COLORS[i % len(CHART_COLORS)]},
            }
            for i, n in enumerate(names)
        ]
        chord_links: list[dict] = []
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val > 0 and i != j:
                    chord_links.append({
                        "source": names[i],
                        "target": names[j],
                        "value": float(val),
                    })
        return ui.echart({
            "backgroundColor": _CHART_BG,
            "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
            "title": (
                {
                    "text": f"top {max_entities} by relationship strength",
                    "top": 4,
                    "left": "center",
                    "textStyle": {
                        "color": _CHART_TEXT,
                        "fontSize": 10,
                        "fontWeight": "normal",
                    },
                }
                if truncated
                else {"show": False}
            ),
            "series": [{
                "type": "chord",
                "data": chord_nodes,
                "links": chord_links,
                "minAngle": 1,
                "padAngle": 1,
                "label": {"show": show_labels, "color": _CHART_TEXT, "fontSize": 11},
                "lineStyle": {"color": "gradient", "opacity": 0.55},
                "emphasis": {"focus": "adjacency",
                              "lineStyle": {"opacity": 0.85}},
            }],
        }).classes("w-full").style(f"height:{height}")

    # Fallback: graph with circular layout (works on ECharts < 6).
    nodes = [
        {
            "name": n,
            "symbolSize": 18 + 28 * (out_strength[n] / max_out),
            "itemStyle": {"color": CHART_COLORS[i % len(CHART_COLORS)]},
        }
        for i, n in enumerate(names)
    ]
    links = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val > 0 and i != j:
                links.append({
                    "source": names[i],
                    "target": names[j],
                    "value": val,
                    "lineStyle": {
                        "width": max(1, val * 5),
                        "opacity": 0.55,
                    },
                })

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "graph",
            "layout": "circular",
            "circular": {"rotateLabel": True},
            "data": nodes,
            "links": links,
            "roam": True,
            "label": {
                "show": True,
                "position": "right",
                "color": _CHART_TEXT,
                "fontSize": 11,
            },
            "lineStyle": {"curveness": 0.3, "color": "source"},
            "emphasis": {"focus": "adjacency"},
        }],
    }).classes("w-full").style(f"height:{height}")


# ── Parallel coordinates (entity trait comparison) ────────────────

def render_parallel_coords(
    ws: WorldStateV1,
    *,
    height: str = "400px",
    max_entities: int = 25,
) -> ui.echart:
    """Parallel coordinates comparing all entities' trait profiles.

    ``max_entities`` caps the line count to keep the chart from
    devolving into spaghetti; the highest-magnitude profiles win
    (sum of |trait values|).
    """
    dimensions, data_rows, entity_names = ws_to_parallel_data(ws)
    if not dimensions or not data_rows:
        return ui.label("No trait data for parallel view.").classes("text-grey q-pa-md")

    truncated = False
    if len(data_rows) > max_entities:
        # Rank entities by total magnitude across dimensions so the
        # most expressive profiles survive the prune.
        ranked = sorted(
            range(len(data_rows)),
            key=lambda i: sum(abs(v or 0) for v in data_rows[i]),
            reverse=True,
        )[:max_entities]
        keep = sorted(ranked)
        data_rows = [data_rows[i] for i in keep]
        entity_names = [entity_names[i] for i in keep]
        truncated = True

    colors = CHART_COLORS

    series = []
    for i, (row, name) in enumerate(zip(data_rows, entity_names)):
        series.append({
            "type": "parallel",
            "name": name,
            "data": [row],
            "lineStyle": {"width": 2, "color": colors[i % len(colors)], "opacity": 0.7},
        })

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": _CHART_TOOLTIP,
        "legend": {
            "data": entity_names,
            "textStyle": {"color": _CHART_TEXT},
            "top": 0,
            "type": "scroll",
        },
        "parallelAxis": [
            {
                "dim": i,
                "name": d["name"],
                "min": d["min"],
                "max": d["max"],
                "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 10},
                "axisLabel": {"color": _CHART_TEXT},
                "axisLine": {"lineStyle": {"color": "#555"}},
            }
            for i, d in enumerate(dimensions)
        ],
        "parallel": {"left": 60, "right": 40, "bottom": 30, "top": 50},
        "series": series,
    }).classes("w-full").style(f"height:{height}")


# ── Event swim lanes (Gantt-style) ───────────────────────────────

def render_event_gantt(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str | None = None,
    show_status_marks: bool = True,
) -> ui.echart:
    """Swim-lane Gantt chart: events grouped by actor, x = fabula_time.

    Each event becomes a horizontal bar from ``fabula_time`` to
    ``fabula_time + 1`` on its actor's lane, coloured by event type.
    Implemented with a ``custom`` series so we get true start/end bars
    instead of length-only stacked rectangles.

    With ``show_status_marks=True`` an extra scatter series annotates
    each tick where an entity's ``status`` flips (e.g. ❌ on the
    fabula tick they die).

    ``height`` defaults to a lane-driven minimum (≈ 22px per actor)
    so casts of 30+ don't get crushed into 400px. Pass an explicit
    height to override.
    """
    from shadow_loom_ui.viz_helpers import EVENT_TYPE_COLORS

    actor_names, items = ws_to_gantt_data(ws)
    if not items:
        return ui.label("No actor events for swim lanes.").classes("text-grey q-pa-md")

    if height is None:
        height = f"{max(400, 22 * len(actor_names) + 80)}px"

    # Each datum: [actor_idx, start, end, event_type, description, event_id]
    data = [
        [
            it["actor_idx"],
            it["start"],
            it["end"],
            it["event_type"],
            it["description"],
            it["event_id"],
        ]
        for it in items
    ]

    legend_types = sorted({it["event_type"] for it in items})

    # Pieces is needed so each bar is colored by its event_type without us
    # needing to write JS. We use visualMap.pieces over the 3rd dimension.
    pieces = [
        {"value": idx, "color": EVENT_TYPE_COLORS.get(et, "#94a3b8"), "label": et}
        for idx, et in enumerate(legend_types)
    ]
    type_to_idx = {et: idx for idx, et in enumerate(legend_types)}
    # Replace event_type strings with their pieces index
    for row in data:
        row[3] = type_to_idx[row[3]]

    chart_opts: dict = {
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "item",
            ":formatter": (
                "function (p) {"
                "  var v = p.value;"
                "  return '<b>' + v[5] + '</b><br/>'"
                "       + 'actor: ' + p.name + '<br/>'"
                "       + 't: ' + v[1] + ' \u2192 ' + v[2] + '<br/>'"
                "       + v[4];"
                "}"
            ),
        },
        "grid": {"top": 40, "bottom": 40, "left": 130, "right": 30},
        "xAxis": {
            "type": "value",
            "name": "Fabula Time",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
            "min": "dataMin",
            "max": "dataMax",
        },
        "yAxis": {
            "type": "category",
            "data": actor_names,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"show": True, "lineStyle": {"color": "#f1f5f9"}},
        },
        "visualMap": {
            "show": True,
            "type": "piecewise",
            "dimension": 3,
            "pieces": pieces,
            "orient": "horizontal",
            "top": 5,
            "left": "center",
            "textStyle": {"color": _CHART_TEXT},
        },
        "series": [{
            "type": "custom",
            "name": "events",
            "encode": {"x": [1, 2], "y": 0, "tooltip": [1, 2, 4]},
            "data": data,
            ":renderItem": (
                "function (params, api) {"
                "  var y = api.coord([0, api.value(0)])[1];"
                "  var x1 = api.coord([api.value(1), 0])[0];"
                "  var x2 = api.coord([api.value(2), 0])[0];"
                "  var height = api.size([0, 1])[1] * 0.55;"
                "  var width = Math.max(2, x2 - x1);"
                "  return {"
                "    type: 'rect',"
                "    shape: {x: x1, y: y - height/2, width: width, height: height},"
                "    style: api.style({stroke: '#1E2A3A', lineWidth: 0.5})"
                "  };"
                "}"
            ),
        }],
    }

    if show_status_marks:
        marks = ws_to_gantt_status_marks(ws)
        # Filter to actors that actually appear in the Gantt to avoid
        # mismatched y-axis indices.
        actor_set = set(actor_names)
        marks = [m for m in marks if m["actor"] in actor_set]
        if marks:
            chart_opts["series"].append({
                "type": "scatter",
                "name": "status",
                "symbol": "circle",
                "symbolSize": 14,
                "itemStyle": {"color": "#D8334A", "borderColor": "#1E2A3A", "borderWidth": 1},
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": "{@[2]}",
                    "fontSize": 12,
                    "color": _CHART_TEXT,
                },
                "data": [
                    {
                        "value": [m["fabula_time"], m["actor"], m["icon"]],
                        "name": f"{m['actor']} → {m['status']}",
                    }
                    for m in marks
                ],
                "tooltip": {
                    **_CHART_TOOLTIP,
                    "formatter": "{b}<br/>fabula_time = {@[0]}",
                },
                "z": 5,
            })

    chart = ui.echart(chart_opts).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Propagation waterfall ─────────────────────────────────────────

def render_propagation_waterfall(
    mutations: list[dict],
    blocked: list[dict] | None = None,
    *,
    height: str = "350px",
) -> ui.echart:
    """Waterfall chart showing causal propagation steps and blocks."""
    data = mutations_to_waterfall_data(mutations, blocked)
    if not data:
        return ui.label("No propagation data.").classes("text-grey q-pa-md")

    categories = [d["name"] for d in data]
    values = []
    for d in data:
        if d["type"] == "blocked":
            values.append({"value": 0, "itemStyle": {"color": "#94a3b8", "borderType": "dashed"}})
        elif d["type"] == "positive":
            values.append({"value": d["value"], "itemStyle": {"color": "#6FBF3A"}})
        else:
            values.append({"value": d["value"], "itemStyle": {"color": "#D8334A"}})

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis"},
        "grid": {"top": 30, "bottom": 80, "left": 50, "right": 30},
        "xAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {"rotate": 45, "color": _CHART_TEXT, "fontSize": 9},
            "axisLine": {"lineStyle": {"color": "#555"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Trait Delta",
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": [{
            "type": "bar",
            "data": values,
            "label": {"show": True, "position": "top", "fontSize": 9, "color": _CHART_TEXT},
        }],
    }).classes("w-full").style(f"height:{height}")


# ── Composition charts ────────────────────────────────────────────
# Small, focused diagrams that replace the old sunburst+treemap pair.
# Each answers ONE composition question rather than trying to show
# the whole hierarchy at once (which produced unreadable rims and
# text overlaps).

def render_population_summary(
    ws: WorldStateV1,
    *,
    height: str = "120px",
) -> ui.element:
    """Compact tile-row of headline counts (entities / locations / events / …)."""
    alive = sum(1 for e in ws.entities.values() if e.status == "healthy")
    injured = sum(1 for e in ws.entities.values() if e.status in ("injured", "ill", "unconscious"))
    dead = sum(1 for e in ws.entities.values() if e.status == "dead")
    utterances = sum(1 for ev in ws.events if ev.event_type == "utterance")
    tiles = [
        ("Entities", len(ws.entities), f"{alive} alive · {injured} hurt · {dead} dead", "person", "#3b82f6"),
        ("Locations", len(ws.locations), "spatial nodes", "place", "#10b981"),
        ("Objects", len(ws.objects), "narrative props", "inventory_2", "#a855f7"),
        ("World traits", len(ws.world_traits), "global forces", "public", "#f59e0b"),
        ("Events", len(ws.events), f"{utterances} utterances", "bolt", "#ef4444"),
        ("Causal", len(ws.causal_topology), "edges", "trending_up", "#0ea5e9"),
        ("Spatial", len(ws.spatial_topology), "edges", "map", "#22c55e"),
        ("Social", len(ws.social_topology), "edges", "people", "#ec4899"),
        ("Channels", len(ws.channels), "comm. capabilities", "mail", "#8b5cf6"),
    ]
    container = ui.row().classes("w-full gap-2 flex-wrap").style(f"min-height:{height}")
    with container:
        for label, value, sub, icon, color in tiles:
            with ui.card().classes(
                "p-3 gap-1 flex-grow bg-white border border-slate-200 rounded-xl shadow-sm"
            ).style("min-width:140px;"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon(icon).style(f"color:{color};font-size:20px;")
                    ui.label(str(value)).classes(
                        "text-2xl font-semibold text-slate-800"
                    )
                ui.label(label).classes("text-xs uppercase tracking-wide text-slate-500")
                ui.label(sub).classes("text-xs text-slate-400")
    return container


def render_status_donut(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "260px",
) -> ui.echart:
    """Donut of entity status distribution (healthy / injured / dead / …)."""
    counts: dict[str, int] = {}
    for e in ws.entities.values():
        counts[e.status] = counts.get(e.status, 0) + 1
    if not counts:
        return ui.label("No entities to summarise.").classes("text-grey q-pa-md")
    palette = {
        "healthy": "#10b981",
        "injured": "#f59e0b",
        "ill": "#eab308",
        "unconscious": "#94a3b8",
        "dead": "#ef4444",
    }
    data = [
        {"name": k, "value": v, "itemStyle": {"color": palette.get(k, "#64748b")}}
        for k, v in counts.items()
    ]
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item",
                    "formatter": "{b}: {c} ({d}%)"},
        "legend": {"bottom": 0, "textStyle": {"color": _CHART_TEXT, "fontSize": 10}},
        "series": [{
            "type": "pie",
            "radius": ["48%", "78%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderColor": _CHART_BG, "borderWidth": 2},
            "label": {"show": True, "color": _CHART_TEXT, "fontSize": 10,
                       "formatter": "{b}\n{c}"},
            "labelLine": {"show": True, "length": 6, "length2": 6},
            "data": data,
        }],
    }).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    return chart


def render_location_occupancy_bar(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "320px",
    top_n: int = 12,
) -> ui.echart:
    """Horizontal stacked bar: per location, count of entities + objects."""
    rows: list[tuple[str, str, int, int]] = []
    for lid, loc in ws.locations.items():
        n_ent = sum(1 for e in ws.entities.values() if e.location_id == lid)
        n_obj = sum(1 for o in ws.objects.values() if o.location_id == lid)
        if n_ent + n_obj == 0:
            continue
        rows.append((lid, loc.name, n_ent, n_obj))
    if not rows:
        return ui.label("No occupied locations.").classes("text-grey q-pa-md")
    rows.sort(key=lambda r: -(r[2] + r[3]))
    rows = rows[:top_n]
    rows.reverse()  # ECharts horizontal: largest at top
    names = [r[1] for r in rows]
    ents = [r[2] for r in rows]
    objs = [r[3] for r in rows]
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis",
                    "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "right": 10, "textStyle": {"color": _CHART_TEXT, "fontSize": 10}},
        "grid": {"left": 100, "right": 30, "top": 30, "bottom": 24},
        "xAxis": {"type": "value", "axisLabel": {"color": _CHART_TEXT, "fontSize": 10}},
        "yAxis": {
            "type": "category", "data": names,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "series": [
            {"name": "Entities", "type": "bar", "stack": "tot",
             "itemStyle": {"color": "#3b82f6"},
             "label": {"show": True, "color": "#fff", "fontSize": 10},
             "data": ents},
            {"name": "Objects", "type": "bar", "stack": "tot",
             "itemStyle": {"color": "#a855f7"},
             "label": {"show": True, "color": "#fff", "fontSize": 10},
             "data": objs},
        ],
    }).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    return chart


def render_world_trait_bars(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "320px",
) -> ui.echart:
    """Grouped horizontal bars per global trait: magnitude vs inertia."""
    if not ws.world_traits:
        return ui.label("No world traits.").classes("text-grey q-pa-md")
    traits = sorted(
        ws.world_traits.values(),
        key=lambda wt: -wt.magnitude.value,
    )
    names = [wt.name for wt in traits][::-1]
    mags = [round(wt.magnitude.value, 3) for wt in traits][::-1]
    inertias = [round(wt.magnitude.inertia, 3) for wt in traits][::-1]
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis",
                    "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "right": 10,
                    "textStyle": {"color": _CHART_TEXT, "fontSize": 10}},
        "grid": {"left": 130, "right": 30, "top": 30, "bottom": 24},
        "xAxis": {"type": "value", "max": 1.0,
                   "axisLabel": {"color": _CHART_TEXT, "fontSize": 10}},
        "yAxis": {"type": "category", "data": names,
                   "axisLabel": {"color": _CHART_TEXT, "fontSize": 10}},
        "series": [
            {"name": "Magnitude", "type": "bar",
             "itemStyle": {"color": "#f59e0b"},
             "label": {"show": True, "position": "right",
                        "color": _CHART_TEXT, "fontSize": 9},
             "data": mags},
            {"name": "Inertia", "type": "bar",
             "itemStyle": {"color": "#64748b"},
             "label": {"show": True, "position": "right",
                        "color": _CHART_TEXT, "fontSize": 9},
             "data": inertias},
        ],
    }).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    return chart


def render_event_type_bar(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "220px",
) -> ui.echart:
    """Vertical bar of event-type counts (choice / outcome / revelation / utterance)."""
    counts: dict[str, int] = {}
    for ev in ws.events:
        counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
    if not counts:
        return ui.label("No events.").classes("text-grey q-pa-md")
    palette = {
        "choice": "#3b82f6",
        "outcome": "#10b981",
        "revelation": "#a855f7",
        "utterance": "#ec4899",
    }
    types = sorted(counts.keys(), key=lambda t: -counts[t])
    data = [
        {"value": counts[t],
          "itemStyle": {"color": palette.get(t, "#64748b")}}
        for t in types
    ]
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis",
                    "axisPointer": {"type": "shadow"}},
        "grid": {"left": 40, "right": 20, "top": 24, "bottom": 30},
        "xAxis": {"type": "category", "data": types,
                   "axisLabel": {"color": _CHART_TEXT, "fontSize": 10}},
        "yAxis": {"type": "value",
                   "axisLabel": {"color": _CHART_TEXT, "fontSize": 10}},
        "series": [{
            "type": "bar", "data": data,
            "label": {"show": True, "position": "top",
                       "color": _CHART_TEXT, "fontSize": 10},
        }],
    }).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    return chart


def render_object_ownership_bar(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "260px",
    top_n: int = 10,
) -> ui.echart:
    """Horizontal bar of objects per owner; unowned bucketed as 'Unowned'."""
    if not ws.objects:
        return ui.label("No objects.").classes("text-grey q-pa-md")
    counts: dict[str, int] = {}
    for obj in ws.objects.values():
        if obj.owner_id and obj.owner_id in ws.entities:
            key = ws.entities[obj.owner_id].name
        elif obj.owner_id:
            key = obj.owner_id
        else:
            key = "Unowned"
        counts[key] = counts.get(key, 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n][::-1]
    names = [k for k, _ in items]
    values = [v for _, v in items]
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis",
                    "axisPointer": {"type": "shadow"}},
        "grid": {"left": 110, "right": 30, "top": 20, "bottom": 24},
        "xAxis": {"type": "value",
                   "axisLabel": {"color": _CHART_TEXT, "fontSize": 10}},
        "yAxis": {"type": "category", "data": names,
                   "axisLabel": {"color": _CHART_TEXT, "fontSize": 10}},
        "series": [{
            "type": "bar",
            "itemStyle": {"color": "#a855f7"},
            "label": {"show": True, "position": "right",
                       "color": _CHART_TEXT, "fontSize": 10},
            "data": values,
        }],
    }).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    return chart


# ── Legacy world sunburst ────────────────────────────────────────
# Retained for callers (e.g. example screenshots, tests) that still
# import it directly. The Composition view in the World tab now
# prefers the focused render_* charts above.

def render_sunburst(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "500px",
) -> ui.echart:
    """Sunburst showing world model composition: locations → entities → traits."""
    root = ws_to_sunburst_data(ws)
    if not root.get("children"):
        return ui.label("No world data for sunburst.").classes("text-grey q-pa-md")

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "sunburst",
            "data": root["children"],
            "radius": ["10%", "90%"],
            "sort": None,
            "emphasis": {"focus": "ancestor"},
            "levels": [
                {},
                # Outer rings: only show labels when the slice is wide
                # enough to fit text (``minAngle``) so we don't blast
                # tiny illegible characters around the rim.
                {"r0": "10%", "r": "35%", "label": {"rotate": "tangential", "color": _CHART_TEXT, "fontSize": 11, "minAngle": 4}},
                {"r0": "35%", "r": "65%", "label": {"rotate": "tangential", "color": _CHART_TEXT, "fontSize": 9, "minAngle": 6}},
                {"r0": "65%", "r": "90%", "label": {"rotate": "tangential", "color": _CHART_TEXT, "fontSize": 8, "minAngle": 8}},
            ],
            "label": {"color": _CHART_TEXT},
            "itemStyle": {"borderWidth": 1, "borderColor": "#ffffff"},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


def render_event_calendar(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "220px",
) -> ui.echart:
    """Linear-bucketed heatmap of event density across fabula time.

    A true Gregorian calendar layout doesn't apply to fabula time, so
    we render a single-row heatmap with one cell per fabula bucket.
    Useful for spotting bursts of narrative activity at a glance.
    """
    rows, _, max_count = ws_to_event_calendar_data(ws)
    if not rows:
        return ui.label("No events to summarise.").classes("text-grey q-pa-md")

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "position": "top",
            # Series data is ``[i, 0, count]`` so the count lives at
            # ``c[2]``; ``c[1]`` is the constant y-row index.
            "formatter": "Bucket {b}: {c[2]} event(s)",
        },
        "grid": {"left": 40, "right": 20, "top": 20, "bottom": 30},
        "xAxis": {
            "type": "category",
            "data": [str(r[0]) for r in rows],
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
            "axisLine": {"lineStyle": {"color": "#e2e8f0"}},
            "name": "Fabula bucket",
            "nameLocation": "middle",
            "nameGap": 22,
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "yAxis": {
            "type": "category",
            "data": ["events"],
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "visualMap": {
            "min": 0,
            "max": max(1, max_count),
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
            "inRange": {"color": ["#fef3c7", "#F5B43C", "#D8334A"]},
            "textStyle": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "series": [{
            "type": "heatmap",
            "data": [[i, 0, r[1]] for i, r in enumerate(rows)],
            "label": {"show": False},
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 8,
                    "shadowColor": "rgba(0,0,0,0.2)",
                }
            },
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


def render_world_treemap(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "500px",
) -> ui.echart:
    """Treemap of locations → entities/objects.

    Sized by child count rather than angle, so it stays legible when a
    handful of locations dominate the world (where the sunburst would
    cram everything into a thin slice).
    """
    roots = ws_to_treemap_data(ws)
    if not roots:
        return ui.label("No world data for treemap.").classes("text-grey q-pa-md")

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "treemap",
            "data": roots,
            "roam": False,
            "nodeClick": "zoomToNode",
            "breadcrumb": {
                "show": True,
                "bottom": 0,
                "itemStyle": {
                    "color": "#f8fafc",
                    "borderColor": "#e2e8f0",
                    "textStyle": {"color": _CHART_TEXT, "fontSize": 10},
                },
            },
            "label": {
                "show": True,
                "color": "#ffffff",
                "fontSize": 11,
                "formatter": "{b}",
            },
            "upperLabel": {"show": True, "color": "#ffffff", "fontSize": 11},
            "itemStyle": {"borderColor": "#ffffff", "borderWidth": 1, "gapWidth": 1},
            "levels": [
                {
                    "itemStyle": {
                        "borderWidth": 0,
                        "gapWidth": 4,
                    }
                },
                {
                    "itemStyle": {
                        "borderWidth": 2,
                        "gapWidth": 2,
                        "borderColorSaturation": 0.3,
                    },
                    "upperLabel": {"show": True, "height": 22},
                },
            ],
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


def render_event_polar(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "360px",
    top_n: int = 8,
) -> ui.echart:
    """Polar heatmap of event_type counts per top-N actor.

    Radial bands are actors (sorted by total event count, descending);
    angular slices are event types. Colour intensity encodes count.
    Better than a flat bar grid for visualising many actor/type pairs
    at once because it folds long actor lists into a radial layout.
    """
    actors, event_types, rows = ws_to_polar_event_data(ws, top_n=top_n)
    if not actors or not event_types:
        return ui.label("No actor/event data for polar chart.").classes(
            "text-grey q-pa-md"
        )

    max_c = max((r[2] for r in rows), default=1)
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "formatter": "{b}: count={c[2]}",
        },
        "polar": {"radius": ["18%", "85%"]},
        "angleAxis": {
            "type": "category",
            "data": event_types,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "radiusAxis": {
            "type": "category",
            "data": actors,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#e2e8f0"}},
            "z": 10,
        },
        "visualMap": {
            "min": 0,
            "max": max_c,
            "calculable": True,
            "orient": "vertical",
            "right": 5,
            "top": "middle",
            "inRange": {"color": ["#fef3c7", "#F5B43C", "#D8334A"]},
            "textStyle": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "series": [{
            "type": "heatmap",
            "coordinateSystem": "polar",
            "data": [[r[1], r[0], r[2]] for r in rows],
            "label": {"show": False},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


def render_trait_boxplot(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "320px",
) -> ui.echart:
    """Boxplot of trait-value distributions across all entities.

    Each box summarises one trait shared by ≥3 entities. Outliers
    are overlaid as scatter points so extreme characters stand out.
    """
    names, boxes, outliers = ws_to_trait_boxplot_data(ws)
    if not names:
        return ui.label(
            "Not enough shared traits for a distribution plot."
        ).classes("text-grey q-pa-md")

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "grid": {"left": 50, "right": 20, "top": 20, "bottom": 60},
        "xAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {
                "color": _CHART_TEXT,
                "fontSize": 10,
                "rotate": 30,
                "interval": 0,
            },
            "axisLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#e2e8f0"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "name": "trait value",
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "series": [
            {
                "name": "distribution",
                "type": "boxplot",
                "data": boxes,
                "itemStyle": {
                    "color": "#5C7C8A",
                    "borderColor": "#1E2A3A",
                    "borderWidth": 1,
                },
            },
            {
                "name": "outliers",
                "type": "scatter",
                "data": outliers,
                "symbolSize": 6,
                "itemStyle": {"color": "#D8334A"},
            },
        ],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Entity lifelines (status / location / event ribbons) ─────────

def render_entity_lifelines(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str | None = None,
) -> ui.echart:
    """Per-entity lifelines: status segments + location moves + events.

    Each character occupies one horizontal lane along fabula time.
    Coloured bars show ``status`` over time (green=healthy, amber=
    injured, blue=unconscious, near-black=dead). Diamond markers flag
    every location change with the new place name in the tooltip.
    Small dots render every event the character actor'd in, coloured
    by ``event_type``.

    This replaces the previous single-entity stepped trait line as the
    "Temporal" top diagram because the lifeline view answers "who is
    where, doing what, when" at a glance — the old chart only spoke
    when the user pre-selected an entity.

    ``height`` defaults to a lane-driven minimum (≈ 24px per entity)
    so 20-character casts don't overlap. Pass an explicit height to
    override.
    """
    from shadow_loom_ui.viz_helpers import ws_to_lifeline_data, _STATUS_COLORS

    data = ws_to_lifeline_data(ws)
    if not data["entities"]:
        return ui.label("No entities to show.").classes("text-grey q-pa-md")

    if height is None:
        height = f"{max(320, 24 * len(data['entities']) + 100)}px"

    names = [n for _eid, n in data["entities"]]
    tmin, tmax = data["tmin"], data["tmax"]

    # Status segments → custom series rendering [start, end, row].
    segment_data = [
        [
            seg["row"],
            seg["start"],
            seg["end"],
            seg["status"] or "unknown",
            seg["status_color"],
            seg["location_name"],
        ]
        for seg in data["segments"]
    ]

    # The custom renderer draws a rounded bar between the two x ticks
    # for the row's y position. We use a JS function string here
    # because ECharts custom series accept JS bodies via NiceGUI's
    # ``:fn`` magic on ``ui.echart`` option strings — but that
    # complicates serialisation. To keep this pure-Python we model the
    # segments as a stacked bar series instead, which renders the
    # same visual without needing a custom JS renderer.

    # Build per-row stacked bar lengths: each row gets its segments as
    # individual data points with explicit colour.
    bar_series: list[dict] = []
    # Collapse to one bar series per status so the legend reads cleanly.
    by_status: dict[str, list[list]] = {}
    for seg in data["segments"]:
        by_status.setdefault(seg["status"] or "unknown", []).append([
            seg["row"], seg["start"], seg["end"], seg["location_name"],
        ])
    # We render each segment as a horizontal bar via ``custom`` series
    # with a small JS-free trick: an inverted ``bar`` series with
    # ``data: [{value: [end-start], coord:[start,row]}]`` doesn't exist
    # in ECharts. Instead we use ``custom`` series with rectShape pieces
    # built server-side (no JS needed).
    pieces_data = []
    pieces_meta = []
    for seg in data["segments"]:
        pieces_data.append([seg["row"], seg["start"], seg["end"]])
        pieces_meta.append({
            "status": seg["status"] or "unknown",
            "color": seg["status_color"],
            "location": seg["location_name"],
        })

    # ECharts ``heatmap`` on a category-y, value-x grid with one cell
    # per integer (start..end-1) is the cleanest pure-JSON approach.
    heat_data: list[list] = []
    for seg in data["segments"]:
        for t in range(int(seg["start"]), max(int(seg["start"]) + 1, int(seg["end"]))):
            heat_data.append([t, seg["row"], 1, seg["status_color"]])

    # Pull color out into per-cell itemStyle via "value" tuple +
    # visualMap mapping by 4th dim — we instead provide direct itemStyle
    # by using ``data: [{value: [...], itemStyle: {color: ...}}]``.
    cells = [
        {
            "value": [t, row],
            "itemStyle": {"color": color},
        }
        for t, row, _, color in heat_data
    ]

    # Location-move markers
    move_points = [
        [m["time"], m["row"], m["location_name"]]
        for m in data["moves"]
    ]

    # Event markers
    event_points = [
        {
            "value": [e["time"], e["row"]],
            "itemStyle": {"color": e["color"]},
            "_event_id": e["event_id"],
            "_desc": e["description"],
            "_type": e["event_type"],
        }
        for e in data["events"]
    ]

    # Status legend
    status_legend = [
        {"name": s, "icon": "rect", "itemStyle": {"color": c}}
        for s, c in _STATUS_COLORS.items()
    ]

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "item",
        },
        "legend": [
            {
                "data": [s["name"] for s in status_legend],
                "top": 0,
                "left": "center",
                "textStyle": {"color": _CHART_TEXT, "fontSize": 10},
                "itemWidth": 14,
                "itemHeight": 8,
            },
        ],
        "grid": {"top": 36, "bottom": 30, "left": 110, "right": 20},
        "xAxis": {
            "type": "value",
            "min": tmin,
            "max": tmax,
            "name": "Fabula time",
            "nameGap": 18,
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "yAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            "axisTick": {"show": False},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        },
        "series": [
            # Status ribbon — one cell per integer fabula tick.
            {
                "name": "status",
                "type": "scatter",
                "symbol": "rect",
                "symbolSize": [10, 18],
                "data": cells,
                "z": 1,
                "tooltip": {"show": False},
            },
            # Location-change markers
            {
                "name": "location change",
                "type": "scatter",
                "symbol": "diamond",
                "symbolSize": 12,
                "data": [
                    # Embed the destination location in the value
                    # array (slot 2) so the tooltip formatter can
                    # read it natively — ECharts can't address
                    # arbitrary custom keys from a template string.
                    {
                        "value": [m[0], m[1], m[2]],
                        "name": str(m[2]),
                    }
                    for m in move_points
                ],
                "itemStyle": {
                    "color": "#ffffff",
                    "borderColor": "#1e2a3a",
                    "borderWidth": 1.5,
                },
                "z": 3,
                "tooltip": {
                    "formatter": "Moved → {@[2]}",
                },
            },
            # Event markers (small coloured dots)
            {
                "name": "events",
                "type": "scatter",
                "symbol": "circle",
                "symbolSize": 7,
                "data": event_points,
                "z": 2,
            },
        ]
        + [
            # Hidden series purely to populate the status legend with
            # the canonical colour swatch for each status.
            {
                "name": s["name"],
                "type": "scatter",
                "data": [],
                "itemStyle": s["itemStyle"],
                "symbol": "rect",
                "symbolSize": 10,
            }
            for s in status_legend
        ],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Multi-entity comparison view (radar overlay + ranking) ───────

def render_comparison_view(
    ws: WorldStateV1,
    *,
    entity_ids: list[str] | None = None,
    height: str = "440px",
) -> ui.element:
    """Side-by-side trait comparison for a small set of characters.

    Layout:
      • **Top:** radar chart overlaying each entity's trait profile on
        the same axes — quick visual gestalt of who is similar / who
        is opposite.
      • **Bottom:** per-trait ranking strip — for each trait, a small
        horizontal bar chart ranking the chosen entities along that
        dimension. Replaces the old parallel-coordinates + boxplot
        pair which read as spaghetti once more than two characters
        were present.
    """
    from shadow_loom_ui.viz_helpers import ws_to_comparison_data

    data = ws_to_comparison_data(ws, entity_ids=entity_ids)
    container = ui.column().classes("w-full gap-3")
    with container:
        if not data["entity_names"]:
            ui.label(
                "Pick entities to compare from the toolbar above."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container
        if not data["trait_names"]:
            ui.label(
                "Selected entities share no traits to compare."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container

        names = data["entity_names"]
        traits = data["trait_names"]
        matrix = data["matrix"]
        colors = CHART_COLORS

        # ── Radar overlay ─────────────────────────────────────────
        radar_indicators = [{"name": t, "max": 1.0} for t in traits]
        radar_series_data = [
            {
                "value": matrix[i],
                "name": names[i],
                "lineStyle": {"width": 2, "color": colors[i % len(colors)]},
                "areaStyle": {"opacity": 0.15, "color": colors[i % len(colors)]},
                "itemStyle": {"color": colors[i % len(colors)]},
            }
            for i in range(len(names))
        ]
        with ui.element("div").classes(
            "w-full border border-slate-200 rounded-xl bg-white shadow-sm "
            "overflow-hidden"
        ):
            ui.echart({
                "backgroundColor": _CHART_BG,
                "tooltip": _CHART_TOOLTIP,
                "legend": {
                    "data": names,
                    "top": 4,
                    "left": "center",
                    "textStyle": {"color": _CHART_TEXT, "fontSize": 11},
                },
                "radar": {
                    "indicator": radar_indicators,
                    "shape": "polygon",
                    "splitNumber": 4,
                    "axisName": {"color": _CHART_TEXT, "fontSize": 10},
                    "splitArea": {
                        "areaStyle": {
                            "color": ["rgba(241,245,249,0.4)", "rgba(255,255,255,0.4)"],
                        },
                    },
                    "splitLine": {"lineStyle": {"color": "#cbd5e1"}},
                    "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
                },
                "series": [{
                    "type": "radar",
                    "data": radar_series_data,
                    "symbol": "circle",
                    "symbolSize": 5,
                }],
            }).classes("w-full").style(f"height:{height}")

        # ── Per-trait ranking strips ─────────────────────────────
        with ui.element("div").classes(
            "w-full border border-slate-200 rounded-xl bg-white shadow-sm p-3"
        ):
            ui.label("Trait ranking (per dimension)").classes(
                "text-sm font-semibold text-slate-700 mb-2"
            )
            grid = ui.element("div").classes(
                "grid grid-cols-1 md:grid-cols-2 gap-2"
            )
            with grid:
                # Stable colour per entity name across all strips.
                color_for = {
                    n: colors[i % len(colors)] for i, n in enumerate(names)
                }
                for trait_name, ranked in data["ranking"]:
                    cats = [n for n, _v in ranked]
                    vals = [
                        {"value": v, "itemStyle": {"color": color_for[n]}}
                        for n, v in ranked
                    ]
                    ui.echart({
                        "backgroundColor": _CHART_BG,
                        "title": {
                            "text": trait_name,
                            "left": 8,
                            "top": 4,
                            "textStyle": {
                                "color": _CHART_TEXT,
                                "fontSize": 11,
                                "fontWeight": "600",
                            },
                        },
                        "tooltip": {
                            **_CHART_TOOLTIP,
                            "trigger": "axis",
                            "axisPointer": {"type": "shadow"},
                        },
                        "grid": {"top": 28, "bottom": 18, "left": 90, "right": 30},
                        "xAxis": {
                            "type": "value",
                            "min": 0, "max": 1,
                            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
                            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
                        },
                        "yAxis": {
                            "type": "category",
                            "data": cats,
                            "inverse": True,
                            "axisTick": {"show": False},
                            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
                            "axisLabel": {
                                "color": _CHART_TEXT,
                                "fontSize": 10,
                                "width": 80,
                                "overflow": "truncate",
                            },
                        },
                        "series": [{
                            "type": "bar",
                            "data": vals,
                            "barWidth": 12,
                            "label": {
                                "show": True,
                                "position": "right",
                                "formatter": "{c}",
                                "fontSize": 9,
                                "color": _CHART_TEXT,
                            },
                        }],
                    }).classes("w-full").style(
                        f"height:{max(80, 22 * len(cats) + 50)}px"
                    )
    return container


# =====================================================================
# NEW RENDERERS — added for the ECharts gallery-inspired charts.
# =====================================================================

# ── #3 Animated propagation graph ─────────────────────────────────

def render_propagation_graph(
    mutations: list[dict],
    blocked: list[dict] | None = None,
    *,
    height: str = "300px",
    on_click: OnClick = None,
) -> ui.echart:
    """Force-directed graph of a propagation cascade.

    Inspired by the ``graph-force-dynamic`` example. Nodes pop in
    via the default ECharts entry animation; positive deltas are
    green, negative red, blocked steps grey-dashed.
    """
    nodes, links = mutations_to_propagation_graph(mutations, blocked)
    if not nodes:
        return ui.label("No propagation data.").classes("text-grey q-pa-md")

    show_labels = len(nodes) <= 25
    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "animationDurationUpdate": 800,
        "animationEasingUpdate": "cubicInOut",
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "draggable": True,
            "data": nodes,
            "links": links,
            "force": {"repulsion": 280, "gravity": 0.12, "edgeLength": [60, 140]},
            "edgeSymbol": ["none", "arrow"],
            "edgeSymbolSize": [0, 8],
            "label": {"show": show_labels, "position": "right",
                       "fontSize": 10, "color": _CHART_TEXT},
            "emphasis": {"focus": "adjacency"},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── #9 Calendar overlay (event nodes on a density heat-row) ───────

def render_calendar_graph_overlay(
    ws: WorldStateV1,
    *,
    height: str = "320px",
    on_click: OnClick = None,
    bucket_count: int = 24,
) -> ui.echart:
    """Density heatmap of events per fabula bucket with the events
    themselves overlaid as a scatter graph and chain-reaction edges
    drawn between them.

    Inspired by the ``calendar-graph`` example: tells you *which*
    events drove a density spike, not just that one happened.
    """
    buckets, nodes, links, max_count = ws_to_calendar_graph_data(
        ws, bucket_count=bucket_count
    )
    if not buckets:
        return ui.label("No events to summarise.").classes("text-grey q-pa-md")

    max_stack = max((n["value"][1] for n in nodes), default=0) + 1

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "grid": {"top": 30, "bottom": 50, "left": 50, "right": 30},
        "xAxis": {
            "type": "category",
            "data": [str(i) for i in range(bucket_count)],
            "name": "Fabula bucket",
            "nameLocation": "middle",
            "nameGap": 28,
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
        },
        "yAxis": {
            "type": "value",
            "min": -0.5,
            "max": max(1, max_stack),
            "name": "events in bucket",
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
        },
        "visualMap": {
            "show": True,
            "min": 0,
            "max": max(1, max_count),
            "seriesIndex": 0,
            "calculable": True,
            "orient": "horizontal",
            "right": 10,
            "top": 0,
            "inRange": {"color": ["#fef3c7", "#F5B43C", "#D8334A"]},
            "textStyle": {"color": _CHART_TEXT, "fontSize": 9},
        },
        "series": [
            {
                "type": "bar",
                "name": "density",
                "data": [b[2] for b in buckets],
                "itemStyle": {"opacity": 0.5},
                "z": 1,
                "barCategoryGap": "5%",
            },
            {
                "type": "graph",
                "name": "events",
                "coordinateSystem": "cartesian2d",
                "data": nodes,
                "links": links,
                "symbolSize": 10,
                "label": {"show": False},
                "lineStyle": {
                    "color": EDGE_COLORS["causal"],
                    "opacity": 0.55,
                    "curveness": 0.25,
                    "width": 1.4,
                },
                "emphasis": {"focus": "adjacency", "label": {"show": True}},
                "z": 5,
            },
        ],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# Note: EDGE_COLORS is imported in the top-level import block.


# ── #12 Multi-snapshot trait radar overlay ────────────────────────

def render_trait_radar_compare(
    entity_id: str,
    ws: WorldStateV1,
    times: list[int],
    *,
    height: str = "320px",
) -> ui.echart:
    """Overlay an entity's trait radar at multiple snapshots.

    Useful for "Compare mode": e.g., Macbeth at t=0 / t=600 / t=1900
    on a single radar so the trajectory of guilt / ambition / fear is
    visible at a glance.
    """
    data = entity_to_radar_compare_data(entity_id, ws, times)
    if not data["indicator"]:
        return ui.label("No traits.").classes("text-grey text-caption")

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": _CHART_TOOLTIP,
        "legend": {
            "type": "scroll",
            "data": [s["name"] for s in data["series"]],
            "textStyle": {"color": _CHART_TEXT},
            "top": 4,
        },
        "radar": {
            "indicator": data["indicator"],
            "shape": "polygon",
            "axisName": {"color": _CHART_TEXT, "fontSize": 10},
            "splitArea": {"areaStyle": {"color": ["rgba(0,0,0,0.02)", "rgba(0,0,0,0.05)"]}},
            "splitLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisLine": {"lineStyle": {"color": "#94a3b8"}},
        },
        "series": [{
            "type": "radar",
            "data": data["series"],
            "areaStyle": {"opacity": 0.15},
            "lineStyle": {"width": 2},
            "symbol": "circle",
            "symbolSize": 4,
        }],
    }).classes("w-full").style(f"height:{height}")


# ── #17 Graded affective gauge ────────────────────────────────────

_GAUGE_GRADES = [
    (0.2, "very low"),
    (0.4, "low"),
    (0.6, "moderate"),
    (0.8, "high"),
    (1.01, "very high"),
]


def _grade_for(v: float) -> str:
    for thresh, label in _GAUGE_GRADES:
        if v < thresh:
            return label
    return "very high"


def render_emotional_gauges_graded(
    scores: dict[str, float],
    *,
    height: str = "220px",
    selected: str | None = None,
) -> ui.element:
    """Graded variant of ``render_emotional_gauges``.

    Replaces the bare 0–1 detail with a qualitative label
    (``very low`` → ``very high``) so non-numeric users get an
    immediate qualitative sense without losing the precise value.
    Each metric is rendered as its own gauge in a CSS grid so labels
    never overlap.
    """
    if not scores:
        return ui.label("No scores available.").classes("text-grey text-caption")
    items = list(scores.items())
    return _render_gauge_grid(
        items, height=height, selected=selected, graded=True,
    )


# ── #18 Audit pass-rate pictorial ─────────────────────────────────

def render_audit_passrate_pictorial(
    audit_history: list[dict],
    *,
    height: str = "240px",
) -> ui.echart:
    """Pass-rate per refinement loop iteration as a stacked pictorial.

    Each iteration becomes a vertical bar of human silhouettes whose
    fill height = pass rate, evoking the ``pictorialBar-body-fill``
    example. Useful at-a-glance diagnostic for the audit→refinement loop.
    """
    labels, rates, _rows = audit_passrate_data(audit_history)
    if not labels:
        return ui.label("No audit history.").classes("text-grey q-pa-md")

    # Pre-index by label so the JS tooltip formatter can look up the
    # full row (passed/total/converged) rather than just the label/value
    # the chart sees natively.
    rows_by_label = {r["iteration"]: r for r in _rows}

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "item",
            ":formatter": (
                "function(p){"
                f" var rows = {json.dumps(rows_by_label)};"
                " var r = rows[p.name] || {};"
                " var pct = (p.value * 100).toFixed(0);"
                " var line1 = '<b>' + p.name + '</b>: ' + pct + '%';"
                " var line2 = (r.passed != null && r.total != null) ?"
                "   ('passed ' + r.passed + ' of ' + r.total + ' checks') : '';"
                " var line3 = r.converged ? '\u2713 auditor passed this iteration'"
                "   : 'auditor still flagged issues';"
                " return [line1, line2, line3].filter(Boolean).join('<br/>');"
                "}"
            ),
        },
        "grid": {"top": 30, "bottom": 40, "left": 50, "right": 20},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9, "rotate": 30},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "name": "pass-rate",
            "nameTextStyle": {"color": _CHART_TEXT, "fontSize": 10},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 9},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
        },
        "series": [
            {
                "type": "pictorialBar",
                "name": "filled",
                "data": rates,
                "symbol": "path://M150 0 C200 0 230 50 230 110 C230 160 210 200 180 230 L180 380 C180 410 150 420 150 420 C150 420 120 410 120 380 L120 230 C90 200 70 160 70 110 C70 50 100 0 150 0 Z",
                "symbolBoundingData": 1,
                "symbolClip": True,
                "symbolSize": ["62%", "100%"],
                "itemStyle": {"color": "#6FBF3A", "opacity": 0.9},
                "label": {"show": True, "position": "top",
                           "color": _CHART_TEXT, "fontSize": 10,
                           ":formatter": "function (p) { return (p.value * 100).toFixed(0) + '%'; }"},
                "z": 10,
            },
            {
                "type": "pictorialBar",
                "name": "outline",
                "data": [1] * len(rates),
                "symbol": "path://M150 0 C200 0 230 50 230 110 C230 160 210 200 180 230 L180 380 C180 410 150 420 150 420 C150 420 120 410 120 380 L120 230 C90 200 70 160 70 110 C70 50 100 0 150 0 Z",
                "symbolBoundingData": 1,
                "symbolSize": ["62%", "100%"],
                "itemStyle": {"color": "transparent",
                              "borderColor": "#94a3b8", "borderWidth": 1.5},
                "z": 5,
            },
        ],
    }).classes("w-full").style(f"height:{height}")


# ── #15 Causal "explain this" inspector ───────────────────────────

def open_explain_dialog(
    ws: WorldStateV1,
    node_id: str,
    *,
    title_override: str | None = None,
) -> None:
    """Open a modal listing incoming + outgoing causal edges for a node.

    Demystifies why the engine produced an outcome by ranking edges
    by ``causal_force``. Used as a right-click handler on graph nodes.
    """
    incoming = explain_node_causes(ws, node_id)
    outgoing = explain_node_effects(ws, node_id)

    pretty = title_override or node_id
    with ui.dialog() as dlg, ui.card().classes("p-4 max-w-4xl"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("psychology", color="primary")
            ui.label(f"Why {pretty}?").classes(
                "text-lg font-semibold text-slate-800"
            )
            ui.space()
            ui.button(icon="close", on_click=dlg.close).props(
                "flat dense round color=grey-7"
            )

        if not incoming and not outgoing:
            ui.label("No causal edges touch this node.").classes(
                "text-sm text-slate-500 italic mt-2"
            )
        if incoming:
            ui.label(f"Incoming ({len(incoming)})").classes(
                "text-sm font-semibold text-slate-700 mt-3"
            )
            ui.table(
                columns=[
                    {"name": "source", "label": "Source", "field": "source", "sortable": True},
                    {"name": "type", "label": "Type", "field": "type", "sortable": True},
                    {"name": "mechanism", "label": "Mechanism", "field": "mechanism"},
                    {"name": "force", "label": "Force", "field": "force", "sortable": True},
                    {"name": "evidence", "label": "Evidence", "field": "evidence"},
                    {"name": "fabula_time", "label": "Fabula t", "field": "fabula_time", "sortable": True},
                    {"name": "trait_target", "label": "Trait", "field": "trait_target"},
                    {"name": "trait_delta", "label": "\u0394", "field": "trait_delta"},
                ],
                rows=incoming,
                pagination={"rowsPerPage": 10},
            ).props("dense flat bordered").classes("w-full")
        if outgoing:
            ui.label(f"Outgoing ({len(outgoing)})").classes(
                "text-sm font-semibold text-slate-700 mt-3"
            )
            ui.table(
                columns=[
                    {"name": "target", "label": "Target", "field": "target", "sortable": True},
                    {"name": "type", "label": "Type", "field": "type", "sortable": True},
                    {"name": "mechanism", "label": "Mechanism", "field": "mechanism"},
                    {"name": "force", "label": "Force", "field": "force", "sortable": True},
                    {"name": "evidence", "label": "Evidence", "field": "evidence"},
                    {"name": "fabula_time", "label": "Fabula t", "field": "fabula_time", "sortable": True},
                    {"name": "trait_target", "label": "Trait", "field": "trait_target"},
                    {"name": "trait_delta", "label": "\u0394", "field": "trait_delta"},
                ],
                rows=outgoing,
                pagination={"rowsPerPage": 10},
            ).props("dense flat bordered").classes("w-full")
    dlg.open()
