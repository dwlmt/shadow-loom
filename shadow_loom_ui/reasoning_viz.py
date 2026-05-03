# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ECharts renderers for the new reasoning-trace surfaces.

Companion to :mod:`shadow_loom_ui.reasoning_helpers`. Each function
returns a NiceGUI ``ui.echart`` (or ``ui.label`` for empty states)
mirroring the style of :mod:`shadow_loom_ui.viz`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from nicegui import ui

from shadow_loom_ui.reasoning_helpers import (
    attribution_graph_data,
    belief_provenance_data,
    convergence_trajectory_data,
    foreshadowing_arcs_data,
)
from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)


# ── Theme constants — mirrored from viz.py at import time ────────────
# We import lazily inside each renderer so a future refresh of the
# palette (via theme.set_chart_dark) is picked up automatically.

def _theme() -> dict:
    from shadow_loom_ui.viz import (
        _CHART_BG, _CHART_TEXT, _CHART_TOOLTIP,
    )
    return {"bg": _CHART_BG, "text": _CHART_TEXT, "tooltip": _CHART_TOOLTIP}


# =====================================================================
# Belief provenance — chronological rows, rendered as a scatter timeline
# =====================================================================


def render_belief_provenance(
    ws: WorldStateV1,
    entity_id: str,
    *,
    height: str = "320px",
) -> ui.element:
    """Scatter timeline of how an entity's beliefs evolved.

    X-axis: fabula_time. Y-axis: belief target_id (categorical).
    Symbols: green ▲ for ``initial`` and ``added``, red ▼ for
    ``invalidated``. Tooltip names the trigger event.
    """
    rows = belief_provenance_data(ws, entity_id)
    if not rows:
        return ui.label(
            f"No belief provenance available for {entity_id}."
        ).classes("text-grey text-caption q-pa-md")

    th = _theme()
    targets = sorted({r["target_label"] for r in rows})
    series_added: List[List[Any]] = []
    series_invalid: List[List[Any]] = []
    for r in rows:
        point = [
            r["fabula_time"],
            r["target_label"],
            {
                "kind": r["kind"],
                "perceived_state": r["perceived_state"],
                "confidence": r["confidence"],
                "inertia": r["inertia"],
                "trigger": r["trigger_label"],
            },
        ]
        if r["kind"] == "invalidated":
            series_invalid.append(point)
        else:
            series_added.append(point)

    tooltip_fmt = (
        "function(p){"
        "var d=p.data[2]||{};"
        "return '<b>'+p.data[1]+'</b><br/>'"
        "+'fabula t='+p.data[0]+'<br/>'"
        "+'kind: '+(d.kind||'')+'<br/>'"
        "+(d.perceived_state?('belief: '+d.perceived_state+'<br/>'):'')"
        "+(d.confidence!=null?('confidence '+d.confidence.toFixed(2)+'<br/>'):'')"
        "+(d.inertia!=null?('inertia '+d.inertia.toFixed(2)+'<br/>'):'')"
        "+(d.trigger?('via '+d.trigger):'')"
        "}"
    )

    return ui.echart({
        "backgroundColor": th["bg"],
        "tooltip": {**th["tooltip"], "formatter": {":fn": tooltip_fmt}},
        "grid": {"left": 160, "right": 30, "top": 30, "bottom": 60},
        "legend": {
            "data": ["formed/added", "invalidated"],
            "textStyle": {"color": th["text"]},
            "top": 0,
        },
        "xAxis": {
            "type": "value",
            "name": "fabula time",
            "nameTextStyle": {"color": th["text"]},
            "axisLabel": {"color": th["text"]},
        },
        "yAxis": {
            "type": "category",
            "data": targets,
            "axisLabel": {"color": th["text"], "fontSize": 11},
        },
        "series": [
            {
                "name": "formed/added",
                "type": "scatter",
                "data": series_added,
                "symbol": "triangle",
                "symbolSize": 14,
                "itemStyle": {"color": "#2EA6A0"},
            },
            {
                "name": "invalidated",
                "type": "scatter",
                "data": series_invalid,
                "symbol": "triangle",
                "symbolRotate": 180,
                "symbolSize": 14,
                "itemStyle": {"color": "#D8334A"},
            },
        ],
    }).classes("w-full").style(f"height:{height}")


# =====================================================================
# Attribution graph — directed force layout, sized by depth
# =====================================================================


def render_attribution_graph(
    ws: WorldStateV1,
    target_id: str,
    *,
    max_depth: int = 4,
    min_force: float = 0.0,
    height: str = "420px",
    on_click: Optional[Any] = None,
) -> ui.element:
    """Reverse-walk causal graph from the outcome ``target_id``.

    The target is highlighted; ancestors fade with depth. Edges
    inherit width from causal_force and are dashed for weak evidence.
    """
    nodes, links, cats, _paths = attribution_graph_data(
        ws, target_id, max_depth=max_depth, min_force=min_force,
    )
    if not nodes:
        return ui.label(
            f"No causal ancestors found for {target_id}."
        ).classes("text-grey text-caption q-pa-md")

    th = _theme()
    chart = ui.echart({
        "backgroundColor": th["bg"],
        "tooltip": {**th["tooltip"], "trigger": "item"},
        "legend": {
            "data": [c["name"] for c in cats],
            "textStyle": {"color": th["text"]},
            "top": 0, "right": 10,
        },
        "series": [{
            "type": "graph",
            "layout": "force",
            "data": nodes,
            "links": links,
            "categories": cats,
            "roam": True,
            "label": {"show": True, "color": th["text"], "fontSize": 11},
            "edgeSymbol": ["none", "arrow"],
            "edgeSymbolSize": [0, 10],
            "force": {
                "repulsion": 220,
                "edgeLength": [60, 140],
                "gravity": 0.05,
            },
        }],
    }).classes("w-full").style(f"height:{height}")

    if on_click is not None:
        chart.on("chart:click", on_click)
    return chart


# =====================================================================
# Foreshadowing arcs — fabula timeline with curved setup→payoff lines
# =====================================================================


def render_foreshadowing_arcs(
    ws: WorldStateV1,
    *,
    height: str = "360px",
    show_loose_only: bool = False,
) -> ui.element:
    """Curved arcs from setup events to their payoff events.

    When ``show_loose_only`` is True, only arcs whose payoff target
    is NOT a registered event/entity are drawn — i.e. unpaid Chekhov's
    guns the writer should resolve.
    """
    rows = foreshadowing_arcs_data(ws)
    if show_loose_only:
        rows = [r for r in rows if r["is_loose"]]
    if not rows:
        msg = (
            "No unresolved setups found." if show_loose_only
            else "No foreshadowing arcs in this world."
        )
        return ui.label(msg).classes("text-grey text-caption q-pa-md")

    th = _theme()

    # ECharts custom-series "lines" with curveness
    lines_data: List[Dict[str, Any]] = []
    points: List[List[Any]] = []
    seen_points: set[tuple[int, str]] = set()

    for r in rows:
        color = "#FF8C42" if r["is_loose"] else "#3A7BD5"
        lines_data.append({
            "coords": [
                [r["setup_t"], r["setup_label"]],
                [r["payoff_t"], r["payoff_label"]],
            ],
            "lineStyle": {
                "color": color,
                "width": max(1.0, min(5.0, r["force"] / 2)),
                "curveness": 0.2,
                "type": "dashed" if r["evidence"] == "weak" else "solid",
                "opacity": 0.85,
            },
            "tooltip": {
                "formatter": (
                    f"<b>{r['setup_label']}</b> → <b>{r['payoff_label']}</b>"
                    f"<br/>span: {r['span']} ticks · force {r['force']:.1f}"
                    f"<br/>mechanism: {r['mechanism']}"
                    f"<br/>evidence: {r['evidence']}"
                    f"{'<br/><i>unresolved setup</i>' if r['is_loose'] else ''}"
                ),
            },
        })
        for t, lab in [(r["setup_t"], r["setup_label"]),
                       (r["payoff_t"], r["payoff_label"])]:
            key = (t, lab)
            if key in seen_points:
                continue
            seen_points.add(key)
            points.append([t, lab])

    y_categories = sorted({p[1] for p in points})

    return ui.echart({
        "backgroundColor": th["bg"],
        "tooltip": {**th["tooltip"], "trigger": "item"},
        "grid": {"left": 200, "right": 30, "top": 30, "bottom": 60},
        "xAxis": {
            "type": "value",
            "name": "fabula time",
            "nameTextStyle": {"color": th["text"]},
            "axisLabel": {"color": th["text"]},
        },
        "yAxis": {
            "type": "category",
            "data": y_categories,
            "axisLabel": {"color": th["text"], "fontSize": 10},
        },
        "series": [
            {
                "type": "scatter",
                "data": points,
                "symbolSize": 8,
                "itemStyle": {"color": "#9E9E9E"},
                "tooltip": {"show": False},
            },
            {
                "type": "lines",
                "data": lines_data,
                "polyline": False,
                "coordinateSystem": "cartesian2d",
                "effect": {"show": False},
            },
        ],
    }).classes("w-full").style(f"height:{height}")


# =====================================================================
# Convergence trajectory — line chart of audit metrics per iteration
# =====================================================================


def render_convergence_trajectory(
    feedback_result: Any,
    *,
    height: str = "240px",
) -> ui.element:
    """Multi-line chart: violations and critical violations per iteration."""
    rows = convergence_trajectory_data(feedback_result)
    if not rows:
        return ui.label("No audit history.").classes(
            "text-grey text-caption q-pa-md"
        )

    th = _theme()
    iterations = [r["iteration"] for r in rows]
    total = [r["violation_count"] for r in rows]
    crit = [r["critical_count"] for r in rows]
    passed_marks = [
        {"xAxis": r["iteration"], "label": {"formatter": "✓"}}
        for r in rows if r["passed"]
    ]

    return ui.echart({
        "backgroundColor": th["bg"],
        "tooltip": {**th["tooltip"], "trigger": "axis"},
        "legend": {
            "data": ["all violations", "critical"],
            "textStyle": {"color": th["text"]},
            "top": 0,
        },
        "grid": {"left": 50, "right": 30, "top": 36, "bottom": 36},
        "xAxis": {
            "type": "category",
            "data": iterations,
            "name": "iteration",
            "nameTextStyle": {"color": th["text"]},
            "axisLabel": {"color": th["text"]},
        },
        "yAxis": {
            "type": "value",
            "minInterval": 1,
            "axisLabel": {"color": th["text"]},
        },
        "series": [
            {
                "name": "all violations",
                "type": "line",
                "data": total,
                "smooth": True,
                "lineStyle": {"color": "#F5B43C", "width": 2.5},
                "itemStyle": {"color": "#F5B43C"},
                "symbolSize": 8,
                "markLine": {
                    "symbol": "none",
                    "data": passed_marks,
                    "lineStyle": {"color": "#2EA6A0", "type": "dashed"},
                },
            },
            {
                "name": "critical",
                "type": "line",
                "data": crit,
                "smooth": True,
                "lineStyle": {"color": "#D8334A", "width": 2.5},
                "itemStyle": {"color": "#D8334A"},
                "symbolSize": 8,
            },
        ],
    }).classes("w-full").style(f"height:{height}")


# =====================================================================
# World diff overlay — coloured world graph showing add/remove/change
# =====================================================================


def render_world_diff_overlay(
    diff: Dict[str, Any],
    *,
    height: str = "420px",
) -> ui.element:
    """Render a :func:`world_diff_data` payload as a chord-style summary.

    For now we show three bar columns (added / removed / changed) per
    entity-type bucket — a fast scan that beats reading raw JSON.
    Future work: overlay on the actual world graph.
    """
    if not diff:
        return ui.label("No diff to display.").classes(
            "text-grey text-caption q-pa-md"
        )

    th = _theme()
    buckets = ["entities", "events", "objects", "world_traits", "causal_edges"]
    labels = ["Entities", "Events", "Objects", "World traits", "Causal edges"]
    added = [len(diff.get("added", {}).get(b, [])) for b in buckets]
    removed = [len(diff.get("removed", {}).get(b, [])) for b in buckets]
    # 'changed' only tracked for entities/objects/world_traits
    changed = [
        len(diff.get("changed", {}).get(b, []))
        if b in {"entities", "objects", "world_traits"} else 0
        for b in buckets
    ]

    return ui.echart({
        "backgroundColor": th["bg"],
        "tooltip": {**th["tooltip"], "trigger": "axis"},
        "legend": {
            "data": ["added", "removed", "changed"],
            "textStyle": {"color": th["text"]},
            "top": 0,
        },
        "grid": {"left": 60, "right": 30, "top": 36, "bottom": 60},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"color": th["text"], "rotate": 20},
        },
        "yAxis": {
            "type": "value",
            "minInterval": 1,
            "axisLabel": {"color": th["text"]},
        },
        "series": [
            {
                "name": "added",
                "type": "bar",
                "data": added,
                "itemStyle": {"color": "#2EA6A0"},
            },
            {
                "name": "removed",
                "type": "bar",
                "data": removed,
                "itemStyle": {"color": "#D8334A"},
            },
            {
                "name": "changed",
                "type": "bar",
                "data": changed,
                "itemStyle": {"color": "#F5B43C"},
            },
        ],
    }).classes("w-full").style(f"height:{height}")
