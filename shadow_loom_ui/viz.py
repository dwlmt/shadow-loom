"""ECharts-based visualization renderers for Shadow-Loom.

Each function returns a ``nicegui.ui.echart`` element configured with the
appropriate series type, click callbacks, and dark-theme styling.  All data
transformations are delegated to ``viz_helpers``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from nicegui import ui

from shadow_loom.models import WorldStateV1
from shadow_loom_ui.viz_helpers import (
    CATEGORIES,
    NODE_COLORS,
    entity_state_timeline_data,
    entity_to_radar_data,
    mutations_to_waterfall_data,
    version_tree_to_echart_data,
    ws_to_causal_force_data,
    ws_to_chord_data,
    ws_to_ego_graph_data,
    ws_to_epistemic_data,
    ws_to_gantt_data,
    ws_to_graph_data,
    ws_to_heatmap_data,
    ws_to_parallel_data,
    ws_to_sankey_data,
    ws_to_social_graph_data,
    ws_to_spatial_graph_data,
    ws_to_sunburst_data,
    ws_to_theme_river_data,
    ws_to_timeline_data,
)

logger = logging.getLogger(__name__)

OnClick = Optional[Callable[[dict], Any]]

# ── Shared dark theme defaults ──────────────────────────────────────

_DARK_BG = "#1e1e1e"
_DARK_TEXT = "#ccc"
_DARK_TOOLTIP = {
    "backgroundColor": "#333",
    "borderColor": "#555",
    "textStyle": {"color": "#eee"},
}

_LEGEND = {
    "data": [c["name"] for c in CATEGORIES],
    "textStyle": {"color": _DARK_TEXT},
    "top": 0,
    "right": 10,
    "orient": "vertical",
}


# ── Force-directed full world graph ────────────────────────────────

def render_world_graph(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
) -> ui.echart:
    """Full world graph — force layout with adjacency highlighting."""
    nodes, links, cats = ws_to_graph_data(ws)
    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
        "legend": [_LEGEND],
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
                "show": True,
                "position": "right",
                "fontSize": 10,
                "color": _DARK_TEXT,
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
    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
        "legend": [_LEGEND],
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
            "label": {"show": True, "position": "right", "fontSize": 10, "color": _DARK_TEXT},
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
) -> ui.echart:
    """Entities + relationship edges — colored by affinity."""
    nodes, links, cats = ws_to_social_graph_data(ws)
    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
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
            "force": {"repulsion": 300, "gravity": 0.15, "edgeLength": [80, 180]},
            "label": {"show": True, "position": "right", "fontSize": 11, "color": _DARK_TEXT},
            "lineStyle": {"curveness": 0.2, "opacity": 0.7},
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
) -> ui.echart:
    """Locations + spatial edges — dashed lines for locked connections."""
    nodes, links, cats = ws_to_spatial_graph_data(ws)
    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
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
            "force": {"repulsion": 250, "gravity": 0.2, "edgeLength": [60, 140]},
            "label": {"show": True, "position": "right", "fontSize": 11, "color": _DARK_TEXT},
            "lineStyle": {"curveness": 0.1, "opacity": 0.7},
        }],
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
) -> ui.echart:
    """Sankey diagram of causal topology — events flow left-to-right."""
    nodes, links = ws_to_sankey_data(ws)
    if not nodes:
        return ui.label("No causal edges to display.").classes("text-grey q-pa-md")

    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "sankey",
            "layout": "none",
            "data": nodes,
            "links": links,
            "emphasis": {"focus": "adjacency"},
            "nodeAlign": "left",
            "orient": "horizontal",
            "lineStyle": {"color": "gradient", "curveness": 0.5, "opacity": 0.4},
            "itemStyle": {"borderWidth": 1, "borderColor": "#555"},
            "label": {"color": _DARK_TEXT, "fontSize": 10},
        }],
    }).classes("w-full").style(f"height:{height}")

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
        "backgroundColor": _DARK_BG,
        "tooltip": _DARK_TOOLTIP,
        "radar": {
            "indicator": radar_data["indicator"],
            "shape": "polygon",
            "axisName": {"color": _DARK_TEXT, "fontSize": 10},
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
    """Entity × entity heatmap colored by a relationship metric."""
    names, data = ws_to_heatmap_data(ws, metric=metric)
    if not names:
        return ui.label("No relationship data.").classes("text-grey q-pa-md")

    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "position": "top"},
        "grid": {"top": 30, "bottom": 80, "left": 100, "right": 30},
        "xAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {"rotate": 45, "color": _DARK_TEXT, "fontSize": 10},
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {"color": _DARK_TEXT, "fontSize": 10},
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": -1,
            "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
            "inRange": {
                "color": ["#F44336", "#9E9E9E", "#4CAF50"],
            },
            "textStyle": {"color": _DARK_TEXT},
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": True, "fontSize": 9, "color": "#eee"},
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


# ── Emotional / narrative gauges ──────────────────────────────────

def render_emotional_gauges(
    scores: dict[str, float],
    *,
    height: str = "200px",
) -> ui.echart:
    """Row of gauge dials for emotional/narrative scores (0–1 scale)."""
    if not scores:
        return ui.label("No scores available.").classes("text-grey text-caption")

    n = len(scores)
    gauge_data = []
    for i, (name, val) in enumerate(scores.items()):
        gauge_data.append({
            "value": round(val, 2),
            "name": name,
        })

    return ui.echart({
        "backgroundColor": _DARK_BG,
        "series": [{
            "type": "gauge",
            "startAngle": 200,
            "endAngle": -20,
            "min": 0,
            "max": 1,
            "splitNumber": 5,
            "data": gauge_data,
            "axisLine": {
                "lineStyle": {
                    "width": 15,
                    "color": [
                        [0.3, "#4CAF50"],
                        [0.7, "#FF9800"],
                        [1, "#F44336"],
                    ],
                },
            },
            "pointer": {"width": 4},
            "axisTick": {"lineStyle": {"color": "#777"}},
            "splitLine": {"lineStyle": {"color": "#777"}},
            "axisLabel": {"color": _DARK_TEXT, "fontSize": 9},
            "detail": {
                "valueAnimation": True,
                "formatter": "{value}",
                "color": _DARK_TEXT,
                "fontSize": 14,
            },
            "title": {"color": _DARK_TEXT, "fontSize": 11},
        }],
    }).classes("w-full").style(f"height:{height}")


# ── Event timeline scatter ────────────────────────────────────────

def render_event_timeline(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "300px",
) -> ui.echart:
    """Scatter plot: fabula_time (x) vs syuzhet_index (y), colored by type."""
    scatter_data = ws_to_timeline_data(ws)
    if not scatter_data:
        return ui.label("No events.").classes("text-grey text-caption")

    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {
            **_DARK_TOOLTIP,
            "trigger": "item",
        },
        "grid": {"top": 40, "bottom": 40, "left": 60, "right": 30},
        "xAxis": {
            "type": "value",
            "name": "Fabula Time",
            "nameTextStyle": {"color": _DARK_TEXT},
            "axisLabel": {"color": _DARK_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Syuzhet Index",
            "nameTextStyle": {"color": _DARK_TEXT},
            "axisLabel": {"color": _DARK_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": [{
            "type": "scatter",
            "data": scatter_data,
            "symbolSize": 14,
            "emphasis": {"scale": 1.6},
            "label": {"show": False},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


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
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0",
              "#00BCD4", "#FFEB3B", "#FF5722", "#607D8B", "#795548"]
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

    return ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "axis"},
        "legend": {
            "data": list(data["series"].keys()),
            "textStyle": {"color": _DARK_TEXT},
            "top": 0,
        },
        "grid": {"top": 40, "bottom": 30, "left": 50, "right": 20},
        "xAxis": {
            "type": "category",
            "data": [str(t) for t in data["times"]],
            "name": "Fabula Time",
            "nameTextStyle": {"color": _DARK_TEXT},
            "axisLabel": {"color": _DARK_TEXT, "fontSize": 9},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Trait Value",
            "min": -1,
            "max": 1,
            "nameTextStyle": {"color": _DARK_TEXT},
            "axisLabel": {"color": _DARK_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": series,
    }).classes("w-full").style(f"height:{height}")


# ── Version tree ──────────────────────────────────────────────────

def render_version_tree(
    tree_data: list[dict],
    current_version_id: int | None = None,
    *,
    on_click: OnClick = None,
    height: str = "300px",
) -> ui.echart:
    """Tree layout of project version history."""
    root = version_tree_to_echart_data(tree_data, current_version_id)
    if not root:
        return ui.label("No versions.").classes("text-grey text-caption")

    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "tree",
            "data": [root],
            "orient": "vertical",
            "top": 20,
            "bottom": 20,
            "left": 40,
            "right": 40,
            "symbolSize": 12,
            "edgeShape": "polyline",
            "edgeForkPosition": "50%",
            "label": {
                "position": "top",
                "verticalAlign": "middle",
                "align": "center",
                "fontSize": 10,
                "color": _DARK_TEXT,
            },
            "leaves": {
                "label": {"position": "bottom", "verticalAlign": "middle", "align": "center"},
            },
            "lineStyle": {"color": "#555", "width": 1.5},
            "emphasis": {"focus": "descendant"},
            "expandAndCollapse": False,
            "animationDuration": 400,
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
) -> ui.echart:
    """Force-directed graph of causal edges, thickness = causal_force."""
    nodes, links, cats = ws_to_causal_force_data(ws)
    if not nodes:
        return ui.label("No causal topology.").classes("text-grey q-pa-md")

    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
        "legend": {
            "data": [c["name"] for c in cats],
            "textStyle": {"color": _DARK_TEXT},
            "top": 0,
            "right": 10,
            "orient": "vertical",
        },
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "draggable": True,
            "emphasis": {"focus": "adjacency"},
            "categories": cats,
            "data": nodes,
            "links": links,
            "force": {"repulsion": 400, "gravity": 0.1, "edgeLength": [60, 180], "friction": 0.6},
            "label": {"show": True, "position": "right", "fontSize": 9, "color": _DARK_TEXT},
            "lineStyle": {"curveness": 0.15, "opacity": 0.7},
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

    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "position": "top"},
        "grid": {"top": 30, "bottom": 80, "left": 100, "right": 30},
        "xAxis": {
            "type": "category",
            "data": target_names,
            "name": "Belief About",
            "nameTextStyle": {"color": _DARK_TEXT},
            "axisLabel": {"rotate": 45, "color": _DARK_TEXT, "fontSize": 10},
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": ent_names,
            "name": "Believer",
            "nameTextStyle": {"color": _DARK_TEXT},
            "axisLabel": {"color": _DARK_TEXT, "fontSize": 10},
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": 0,
            "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
            "inRange": {"color": ["#424242", "#2196F3", "#4CAF50"]},
            "textStyle": {"color": _DARK_TEXT},
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": True, "fontSize": 9, "color": "#eee"},
            "emphasis": {"itemStyle": {"shadowBlur": 10}},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart


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
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "axis"},
        "legend": {
            "data": legends,
            "textStyle": {"color": _DARK_TEXT},
            "top": 0,
            "type": "scroll",
        },
        "singleAxis": {
            "type": "category",
            "bottom": 50,
            "top": 50,
            "axisLabel": {"color": _DARK_TEXT},
            "axisLine": {"lineStyle": {"color": "#555"}},
        },
        "series": [{
            "type": "themeRiver",
            "data": data,
            "label": {"show": False},
            "emphasis": {"itemStyle": {"shadowBlur": 20, "shadowColor": "rgba(0,0,0,0.8)"}},
        }],
    }).classes("w-full").style(f"height:{height}")


# ── Chord diagram (relationship reciprocity) ─────────────────────

def render_chord_diagram(
    ws: WorldStateV1,
    *,
    metric: str = "affinity",
    height: str = "400px",
) -> ui.echart:
    """Chord diagram showing bidirectional relationship strengths."""
    names, matrix = ws_to_chord_data(ws, metric=metric)
    if not names or all(all(v == 0 for v in row) for row in matrix):
        return ui.label("No relationship data for chord.").classes("text-grey q-pa-md")

    # Build as a graph with circular layout (chord alternative for ECharts < 6)
    nodes = [{"name": n, "symbolSize": 30} for n in names]
    links = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val > 0 and i != j:
                links.append({
                    "source": names[i],
                    "target": names[j],
                    "value": val,
                    "lineStyle": {"width": max(1, val * 5), "opacity": 0.6},
                })

    return ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "graph",
            "layout": "circular",
            "circular": {"rotateLabel": True},
            "data": nodes,
            "links": links,
            "roam": True,
            "label": {"show": True, "position": "right", "color": _DARK_TEXT, "fontSize": 11},
            "lineStyle": {"curveness": 0.3, "color": "source"},
            "emphasis": {"focus": "adjacency"},
        }],
    }).classes("w-full").style(f"height:{height}")


# ── Parallel coordinates (entity trait comparison) ────────────────

def render_parallel_coords(
    ws: WorldStateV1,
    *,
    height: str = "400px",
) -> ui.echart:
    """Parallel coordinates comparing all entities' trait profiles."""
    dimensions, data_rows, entity_names = ws_to_parallel_data(ws)
    if not dimensions or not data_rows:
        return ui.label("No trait data for parallel view.").classes("text-grey q-pa-md")

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0",
              "#00BCD4", "#FFEB3B", "#FF5722", "#607D8B", "#795548"]

    series = []
    for i, (row, name) in enumerate(zip(data_rows, entity_names)):
        series.append({
            "type": "parallel",
            "name": name,
            "data": [row],
            "lineStyle": {"width": 2, "color": colors[i % len(colors)], "opacity": 0.7},
        })

    return ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": _DARK_TOOLTIP,
        "legend": {
            "data": entity_names,
            "textStyle": {"color": _DARK_TEXT},
            "top": 0,
            "type": "scroll",
        },
        "parallelAxis": [
            {
                "dim": i,
                "name": d["name"],
                "min": d["min"],
                "max": d["max"],
                "nameTextStyle": {"color": _DARK_TEXT, "fontSize": 10},
                "axisLabel": {"color": _DARK_TEXT},
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
    height: str = "400px",
) -> ui.echart:
    """Swim-lane Gantt chart: events grouped by actor, x = fabula_time."""
    from shadow_loom_ui.viz_helpers import EVENT_TYPE_COLORS

    actor_names, items = ws_to_gantt_data(ws)
    if not items:
        return ui.label("No actor events for swim lanes.").classes("text-grey q-pa-md")

    # Build horizontal bar series per event type (stacked = swim lanes)
    # Group items by event_type for separate series
    event_types: dict[str, list] = {}
    for item in items:
        et = item["event_type"]
        if et not in event_types:
            event_types[et] = []
        event_types[et].append(item)

    # For horizontal bars: each bar needs [start, end] on x-axis
    # Use multiple bar series with category y-axis
    series = []
    for et, et_items in event_types.items():
        color = EVENT_TYPE_COLORS.get(et, "#607D8B")
        # Build sparse data: one entry per actor row, None for others
        bar_data = [None] * len(actor_names)
        for item in et_items:
            idx = item["actor_idx"]
            duration = max(item["end"] - item["start"], 5)
            bar_data[idx] = {
                "value": duration,
                "name": item["description"],
                "itemStyle": {"color": color},
            }
        series.append({
            "type": "bar",
            "name": et,
            "stack": "gantt",
            "data": bar_data,
            "barWidth": "60%",
        })

    chart = ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {
            **_DARK_TOOLTIP,
            "trigger": "item",
            ":formatter": "params => params.name || params.seriesName",
        },
        "legend": {
            "data": list(event_types.keys()),
            "textStyle": {"color": _DARK_TEXT},
            "top": 5,
        },
        "grid": {"top": 35, "bottom": 40, "left": 120, "right": 30},
        "xAxis": {
            "type": "value",
            "name": "Duration",
            "nameTextStyle": {"color": _DARK_TEXT},
            "axisLabel": {"color": _DARK_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "yAxis": {
            "type": "category",
            "data": actor_names,
            "axisLabel": {"color": _DARK_TEXT, "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#555"}},
        },
        "series": series,
    }).classes("w-full").style(f"height:{height}")

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
            values.append({"value": 0, "itemStyle": {"color": "#616161", "borderType": "dashed"}})
        elif d["type"] == "positive":
            values.append({"value": d["value"], "itemStyle": {"color": "#4CAF50"}})
        else:
            values.append({"value": d["value"], "itemStyle": {"color": "#F44336"}})

    return ui.echart({
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "axis"},
        "grid": {"top": 30, "bottom": 80, "left": 50, "right": 30},
        "xAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {"rotate": 45, "color": _DARK_TEXT, "fontSize": 9},
            "axisLine": {"lineStyle": {"color": "#555"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Trait Delta",
            "nameTextStyle": {"color": _DARK_TEXT},
            "axisLabel": {"color": _DARK_TEXT},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": [{
            "type": "bar",
            "data": values,
            "label": {"show": True, "position": "top", "fontSize": 9, "color": _DARK_TEXT},
        }],
    }).classes("w-full").style(f"height:{height}")


# ── World sunburst ────────────────────────────────────────────────

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
        "backgroundColor": _DARK_BG,
        "tooltip": {**_DARK_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "sunburst",
            "data": root["children"],
            "radius": ["10%", "90%"],
            "sort": None,
            "emphasis": {"focus": "ancestor"},
            "levels": [
                {},
                {"r0": "10%", "r": "35%", "label": {"rotate": "tangential", "color": _DARK_TEXT, "fontSize": 11}},
                {"r0": "35%", "r": "65%", "label": {"rotate": "tangential", "color": _DARK_TEXT, "fontSize": 9}},
                {"r0": "65%", "r": "90%", "label": {"rotate": "tangential", "color": _DARK_TEXT, "fontSize": 8}},
            ],
            "label": {"color": _DARK_TEXT},
            "itemStyle": {"borderWidth": 1, "borderColor": "#1e1e1e"},
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click:
        chart.on("click", on_click)
    return chart
