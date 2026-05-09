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
    """Show ``render_fn`` in a resizable dialog with Esc-to-close + PNG/SVG export.

    Supports a "Half-screen" toggle for side-by-side workflows and
    "Save PNG" / "Save SVG" buttons. PNG goes via the chart canvas;
    SVG is sourced from the ECharts instance (``renderToSVGString``
    when available) or from any inline ``<svg>`` element inside the
    dialog as a fallback for non-ECharts diagrams.
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

                    def _save_svg():
                        # ECharts only exposes ``renderToSVGString`` /
                        # ``getDataURL({type:'svg'})`` when the chart
                        # was *initialised* with ``renderer: 'svg'``.
                        # All our charts use the default canvas
                        # renderer for performance, so the previous
                        # implementation silently failed for every
                        # ECharts diagram. Workaround: clone each
                        # chart's option into a temporary hidden DOM
                        # node, init a new ECharts instance there
                        # with the SVG renderer, serialise its root
                        # ``<svg>`` element, then dispose. Falls back
                        # to any inline ``<svg>`` (Mermaid, etc.) if
                        # no ECharts instance is found.
                        ui.run_javascript(
                            """
                            (() => {
                                const root = document.querySelector('.q-dialog__inner');
                                if (!root) return 'NOROOT';
                                let svgStr = null;

                                // 1) ECharts re-render in SVG mode.
                                if (window.echarts && window.echarts.getInstanceByDom) {
                                    const candidates = root.querySelectorAll('div');
                                    for (const d of candidates) {
                                        const inst = window.echarts.getInstanceByDom(d);
                                        if (!inst) continue;
                                        let opt;
                                        try { opt = inst.getOption(); }
                                        catch (e) { continue; }
                                        if (!opt) continue;
                                        const rect = d.getBoundingClientRect();
                                        const w = Math.max(400, Math.round(rect.width || 1000));
                                        const h = Math.max(300, Math.round(rect.height || 700));
                                        const tmp = document.createElement('div');
                                        tmp.style.position = 'fixed';
                                        tmp.style.left = '-10000px';
                                        tmp.style.top = '0';
                                        tmp.style.width = w + 'px';
                                        tmp.style.height = h + 'px';
                                        document.body.appendChild(tmp);
                                        try {
                                            const tmpChart = window.echarts.init(
                                                tmp, null, {renderer: 'svg', width: w, height: h}
                                            );
                                            // Disable animation so the SVG
                                            // is the final frame, not a
                                            // mid-tween snapshot.
                                            opt.animation = false;
                                            opt.animationDuration = 0;
                                            opt.animationDurationUpdate = 0;
                                            tmpChart.setOption(opt, true);
                                            // Force a synchronous layout pass.
                                            tmpChart.resize({width: w, height: h});
                                            const svgEl = tmp.querySelector('svg');
                                            if (svgEl) {
                                                svgStr = new XMLSerializer().serializeToString(svgEl);
                                            }
                                            tmpChart.dispose();
                                        } catch (e) {
                                            console.error('SVG export failed', e);
                                        } finally {
                                            tmp.remove();
                                        }
                                        if (svgStr) break;
                                    }
                                }

                                // 2) Inline <svg> fallback (Mermaid,
                                //    custom diagrams).
                                if (!svgStr) {
                                    const svgEl = root.querySelector('svg');
                                    if (svgEl) {
                                        svgStr = new XMLSerializer().serializeToString(svgEl);
                                    }
                                }
                                if (!svgStr) {
                                    return 'NOSVG';
                                }
                                if (!svgStr.includes('xmlns=')) {
                                    svgStr = svgStr.replace(
                                        '<svg',
                                        '<svg xmlns="http://www.w3.org/2000/svg"'
                                    );
                                }
                                const blob = new Blob(
                                    [svgStr],
                                    {type: 'image/svg+xml;charset=utf-8'}
                                );
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = 'shadow-loom-chart.svg';
                                document.body.appendChild(a);
                                a.click();
                                a.remove();
                                setTimeout(() => URL.revokeObjectURL(url), 1000);
                                return 'OK';
                            })();
                            """
                        )

                    svg_btn = ui.button(
                        icon="image", on_click=_save_svg,
                    ).props("flat dense round color=grey-8")
                    svg_btn.tooltip("Save as SVG (vector)")

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


def render_social_layer_legend() -> None:
    """Chip strip explaining the combined social-layer graph encoding.

    Covers nodes (character / proposition / desire / fear) *and* edges
    (relationship affinity, belief confidence, concern-of). Mirrors the
    encoding actually used by :func:`render_social_layer_graph`.
    """
    def _swatch(color: str, shape: str = "circle") -> None:
        radius = "999px" if shape == "circle" else "2px"
        if shape == "diamond":
            ui.element("div").style(
                f"width:12px;height:12px;background:{color};"
                "transform:rotate(45deg);"
            )
            return
        if shape == "tri":
            ui.element("div").style(
                "width:0;height:0;border-left:6px solid transparent;"
                "border-right:6px solid transparent;"
                f"border-bottom:11px solid {color};"
            )
            return
        if shape == "rect":
            ui.element("div").style(
                f"width:14px;height:10px;background:{color};"
                "border-radius:3px;"
            )
            return
        if shape == "line":
            ui.element("div").style(
                f"width:18px;height:3px;background:{color};border-radius:2px;"
            )
            return
        ui.element("div").style(
            f"width:11px;height:11px;border-radius:{radius};background:{color};"
        )

    def _chip(shape: str, color: str, text: str) -> None:
        with ui.row().classes("items-center gap-1"):
            _swatch(color, shape)
            ui.label(text).classes("text-xs text-slate-600")

    with ui.column().classes(
        "w-full gap-1 px-3 py-2 bg-slate-50 border-b border-slate-200"
    ):
        with ui.row().classes("items-center gap-3 flex-wrap"):
            ui.label("Nodes:").classes("text-xs font-semibold text-slate-700")
            _chip("circle", "#F26B5E", "Character")
            _chip(
                "diamond", "#8a5cf0",
                "Proposition (size = stakes, border thickness = audience surprise)",
            )
            _chip("rect", "#0ea5e9", "World trait (latent force)")
        with ui.row().classes("items-center gap-3 flex-wrap"):
            ui.label("Edges:").classes(
                "text-xs font-semibold text-slate-700"
            )
            _chip("line", "#16a34a", "Affinity + (curved)")
            _chip("line", "#dc2626", "Affinity \u2212 (curved)")
            _chip(
                "line", "#b45309",
                "Belief: entity \u2192 prop (amber arrow, thicker = surer)",
            )
            _chip("line", "#fde68a", "\u2026low confidence")
            _chip(
                "line", "#16a34a",
                "Desire: entity \u2192 prop (green dashed arrow)",
            )
            _chip(
                "line", "#dc2626",
                "Fear: entity \u2192 prop (red dashed arrow)",
            )
            _chip("line", "#0ea5e9", "World trait \u2192 prop (sky-blue dotted)")


# ── Knowledge-asymmetry heatmap ───────────────────────────────────

def render_knowledge_asymmetry_heatmap(
    ws: WorldStateV1,
    *,
    fabula_t: int | None = None,
    height: str = "100%",
    on_click: OnClick = None,
) -> ui.element:
    """Character × proposition asymmetry heatmap (Sternberg lens).

    Cell value:
      * **+conf (green)**  character believes \u2192 truth==True at @t
        (aligned knowledge).
      * **\u2212conf (red)** character believes \u2192 truth==False at @t
        (dramatic-irony / mistaken belief).
      * **blank**          no belief held (curiosity gap).

    A header strip above the heatmap shows the audience prior per
    proposition (white\u2192iris) so the viewer can also see the
    audience's expected stance. The combination encodes the gap /
    suspense / surprise triad in a single panel.
    """
    from shadow_loom_ui.viz_helpers import ws_to_knowledge_asymmetry_matrix

    ent_names, prop_labels, cells, meta = (
        ws_to_knowledge_asymmetry_matrix(ws, fabula_t=fabula_t)
    )
    if not ent_names or not prop_labels:
        return ui.label(
            "No characters and propositions to compare."
        ).classes("text-grey q-pa-md")

    # Tooltip enrichment: build a parallel index for nice hover.
    truth_str = ["true" if m["truth_at"] is True
                 else "false" if m["truth_at"] is False
                 else "—" for m in meta]
    prior_vals = [round(m["audience_prior"], 2) for m in meta]

    # Audience-prior strip is rendered as a 1-row heatmap above the
    # main matrix. We stack two ECharts grids in one chart.
    main_data = cells
    prior_data = [[c, 0, prior_vals[c]] for c in range(len(prop_labels))]

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "position": "top",
            "formatter": (
                "function(p){"
                "var rows=" + str(ent_names).replace("'", '"') + ";"
                "var cols=" + str(prop_labels).replace("'", '"') + ";"
                "var truth=" + str(truth_str).replace("'", '"') + ";"
                "var prior=" + str(prior_vals) + ";"
                "if(p.seriesIndex===0){"
                "  return '<b>Audience prior</b><br/>'+cols[p.data[0]]"
                "    +'<br/>prior: '+prior[p.data[0]];"
                "}"
                "var v=p.data[2];"
                "var stance=v>0?'aligned':(v<0?'mistaken':'unresolved');"
                "return rows[p.data[1]]+'<br/>'+cols[p.data[0]]"
                "  +'<br/>truth@t: '+truth[p.data[0]]"
                "  +'<br/>belief: '+stance+' (|'+Math.abs(v).toFixed(2)+'|)';"
                "}"
            ),
        },
        "grid": [
            {"top": 30, "height": 22, "left": 180, "right": 30},  # prior strip
            {"top": 70, "bottom": 130, "left": 180, "right": 30},  # matrix
        ],
        "xAxis": [
            {
                "gridIndex": 0,
                "type": "category",
                "data": prop_labels,
                "axisLabel": {"show": False},
                "axisTick": {"show": False},
            },
            {
                "gridIndex": 1,
                "type": "category",
                "data": prop_labels,
                "axisLabel": {
                    "rotate": 45,
                    "color": _CHART_TEXT,
                    "fontSize": 10,
                    "width": 110,
                    "overflow": "truncate",
                    "ellipsis": "…",
                },
                "splitArea": {"show": True},
            },
        ],
        "yAxis": [
            {
                "gridIndex": 0,
                "type": "category",
                "data": ["Audience"],
                "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            },
            {
                "gridIndex": 1,
                "type": "category",
                "data": ent_names,
                "axisLabel": {
                    "color": _CHART_TEXT,
                    "fontSize": 10,
                    "width": 160,
                    "overflow": "truncate",
                    "ellipsis": "…",
                },
                "splitArea": {"show": True},
            },
        ],
        "visualMap": [
            {
                "seriesIndex": 0,
                "min": 0.0, "max": 1.0,
                "show": False,
                "inRange": {"color": ["#ffffff", "#8a5cf0"]},
            },
            {
                "seriesIndex": 1,
                "min": -1.0, "max": 1.0,
                "calculable": True,
                "orient": "horizontal",
                "left": "center", "bottom": 10,
                "inRange": {"color": [
                    "#dc2626", "#fca5a5", "#f1f5f9", "#86efac", "#16a34a"
                ]},
                "textStyle": {"color": _CHART_TEXT},
            },
        ],
        "series": [
            {
                "name": "audience_prior",
                "type": "heatmap",
                "xAxisIndex": 0, "yAxisIndex": 0,
                "data": prior_data,
                "label": {"show": False},
            },
            {
                "name": "asymmetry",
                "type": "heatmap",
                "xAxisIndex": 1, "yAxisIndex": 1,
                "data": main_data,
                "label": {"show": len(prop_labels) <= 12, "fontSize": 9},
            },
        ],
    }).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    return chart


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
    fabula_t: int | None = None,
) -> ui.echart:
    """Full world graph — force layout with adjacency highlighting."""
    nodes, links, cats = ws_to_graph_data(ws, fabula_t=fabula_t)
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
    fabula_t: int | None = None,
) -> ui.echart:
    """Ego-graph centered on *focus_ids* with gold-bordered focus nodes."""
    nodes, links, cats = ws_to_ego_graph_data(ws, focus_ids, max_hops=max_hops, fabula_t=fabula_t)
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
    focus_id: str | None = None,
    focus_max_hops: int = 2,
) -> ui.echart:
    """Sankey diagram of causal/social/information flow.

    ``aspect`` selects the underlying edge set — see
    :data:`shadow_loom_ui.viz_helpers.SANKEY_ASPECTS`. ``focus_id``
    drills into a single node's ancestor/descendant chain (\u00b1
    ``focus_max_hops``). Edges are coloured per-link by causality
    modality (see :data:`shadow_loom_ui.viz_helpers.MODALITY_COLORS`).
    """
    nodes, links = ws_sankey_for_aspect(
        ws, aspect, min_force=min_force, fabula_max=fabula_max,
        focus_id=focus_id, focus_max_hops=focus_max_hops,
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
            "emphasis": {
                "focus": "adjacency",
                "lineStyle": {"opacity": 0.95},
            },
            # ``justify`` (vs ``left``) lets ECharts spread sink-only
            # nodes to the right edge so the temporal "rightward
            # flow" reading is honoured even when terminal events
            # have no outgoing edges.
            "nodeAlign": "justify",
            "orient": "horizontal",
            # More vertical breathing room between stacked nodes
            # and a slightly thicker node bar so the column reads
            # as a "lane" rather than a hairline.
            "nodeGap": 18,
            "nodeWidth": 16,
            "draggable": True,
            # Per-link colours (set in ws_to_sankey_data) carry the
            # modality encoding; series-level lineStyle.color must
            # NOT override them, so omit it here. Opacity defaults to
            # the per-link value (0.55) and bumps on hover via
            # ``emphasis``.
            "lineStyle": {
                "curveness": 0.5,
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
    title_lines: list[str] = []
    if truncated:
        title_lines.append(f"showing top {MAX_LINKS} flows by force")
    if focus_id:
        title_lines.append(
            f"focused on {focus_id} (\u00b1{focus_max_hops} hops)"
        )
    if title_lines:
        options["title"] = {
            "text": " \u2022 ".join(title_lines),
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


def render_modality_legend() -> ui.element:
    """Chip-strip legend for the causality-modality colour palette.

    Used above the causal Sankey + force graph so the per-edge colour
    encoding is self-documenting.
    """
    from shadow_loom_ui.viz_helpers import MODALITY_COLORS, MODALITY_LABELS

    container = ui.row().classes(
        "w-full items-center gap-3 px-3 py-2 bg-slate-50 "
        "border border-slate-200 rounded-md flex-wrap"
    )
    with container:
        ui.label("Causality modality:").classes(
            "text-xs font-semibold text-slate-700"
        )
        for key in (
            "chain_reaction", "mutation", "mutation_social",
            "affordance_gate", "ambient_propagation", "world_to_world",
        ):
            color = MODALITY_COLORS[key]
            label = MODALITY_LABELS[key]
            with ui.row().classes("items-center gap-1"):
                ui.html(
                    f'<span style="display:inline-block;width:18px;'
                    f'height:3px;background:{color};border-radius:2px;'
                    f'{"border-top:2px dashed " + color + ";background:transparent;height:0;" if key == "world_to_world" else ""}"></span>'
                )
                ui.label(label).classes("text-xs text-slate-600")
    return container


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
    # Thin axis labels well before they visually collide. fontSize=10 +
    # 45° rotation needs ~14px per label; below ~15 entities every
    # label fits, between 15 and 25 we drop every other, beyond that
    # we widen the stride. ECharts ``interval`` is "how many to skip"
    # so 1 = show every other, 2 = every third, etc.
    if n <= 15:
        label_interval = 0
    elif n <= 25:
        label_interval = 1
    else:
        label_interval = max(1, n // 20)

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "position": "top"},
        # Bottom: rotated 45° names need ~70px; visualMap bar takes ~45px;
        # 10px breathing room between them.
        "grid": {"top": 30, "bottom": 125, "left": 130, "right": 30},
        "xAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {
                "rotate": 45,
                "color": _CHART_TEXT,
                "fontSize": 10,
                "interval": label_interval,
                # Keep long character names from spilling into adjacent
                # cells / the visualMap bar by truncating with an
                # ellipsis. Tooltip still shows the full name on hover.
                "width": 110,
                "overflow": "truncate",
                "ellipsis": "…",
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
                "width": 120,
                "overflow": "truncate",
                "ellipsis": "…",
            },
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": vmin,
            "max": vmax,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 10,
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


# ── Affinity × power scatter (Greimas actantial quadrants) ────────

def render_relationship_quadrant(
    ws: WorldStateV1,
    *,
    fabula_t: int | None = None,
    height: str = "100%",
    on_click: OnClick = None,
) -> ui.element:
    """Per-dyad scatter on (affinity, power_dynamic), size = fear.

    Reads the four Greimas actantial roles directly off the plane:

      * top-right    \u2192 ally / helper (power\u2191, affinity+)
      * bottom-right \u2192 dependent / protege (power\u2193, affinity+)
      * top-left     \u2192 rival / threat (power\u2191, affinity\u2212)
      * bottom-left  \u2192 victim (power\u2193, affinity\u2212)

    A faint cross at (0,0) divides the quadrants. Each point is
    labelled ``source\u2192target`` and tooltipped with the underlying
    metrics.
    """
    points: list[dict] = []
    for rel in ws.social_topology:
        if fabula_t is not None and rel.last_updated_fabula > fabula_t:
            continue
        src = ws.entities.get(rel.source_entity_id)
        tgt = ws.entities.get(rel.target_entity_id)
        if not src or not tgt:
            continue
        aff = float(rel.affinity)
        power = float(rel.power_dynamic)
        fear = float(rel.fear)
        label = f"{src.name}\u2192{tgt.name}"
        points.append({
            "name": label,
            "value": [aff, power, max(8.0, 8.0 + fear * 30.0), fear],
            "itemStyle": {
                "color": (
                    "#16a34a" if aff > 0.1
                    else "#dc2626" if aff < -0.1
                    else "#94a3b8"
                ),
                "opacity": 0.75,
            },
            "_tip": (
                f"<b>{label}</b><br/>affinity: {aff:+.2f}<br/>"
                f"power: {power:+.2f}<br/>fear: {fear:.2f}"
            ),
        })
    if not points:
        return ui.label("No relationships to plot.").classes(
            "text-grey q-pa-md"
        )

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "item",
            "formatter": "function(p){return p.data._tip;}",
        },
        "grid": {"top": 30, "left": 60, "right": 30, "bottom": 60},
        "xAxis": {
            "name": "affinity \u2192",
            "min": -1, "max": 1,
            "axisLine": {"onZero": True},
            "splitLine": {"show": True},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "yAxis": {
            "name": "\u2191 power",
            "min": -1, "max": 1,
            "axisLine": {"onZero": True},
            "splitLine": {"show": True},
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "series": [{
            "type": "scatter",
            "data": points,
            "symbolSize": "function(d){return d[2];}",
            "label": {
                "show": len(points) <= 30,
                "formatter": "{b}",
                "position": "right",
                "fontSize": 9,
                "color": _CHART_TEXT,
            },
            "markArea": {
                "silent": True,
                "itemStyle": {"opacity": 0.04},
                "data": [
                    [{"name": "ally", "xAxis": 0, "yAxis": 0,
                      "itemStyle": {"color": "#16a34a"}},
                     {"xAxis": 1, "yAxis": 1}],
                    [{"name": "rival", "xAxis": -1, "yAxis": 0,
                      "itemStyle": {"color": "#dc2626"}},
                     {"xAxis": 0, "yAxis": 1}],
                    [{"name": "dependent", "xAxis": 0, "yAxis": -1,
                      "itemStyle": {"color": "#3A7BD5"}},
                     {"xAxis": 1, "yAxis": 0}],
                    [{"name": "victim", "xAxis": -1, "yAxis": -1,
                      "itemStyle": {"color": "#7c3aed"}},
                     {"xAxis": 0, "yAxis": 0}],
                ],
                "label": {"position": "insideTopLeft", "fontSize": 10,
                          "color": _CHART_TEXT, "opacity": 0.6},
            },
        }],
    }).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    return chart


# ── Concern activation timeline strip ─────────────────────────────

def render_concern_timeline(
    ws: WorldStateV1,
    *,
    height: str = "100%",
    on_click: OnClick = None,
    axis: str = "fabula",
) -> ui.element:
    """Per-concern horizon-style strip showing salience over fabula time.

    Each row is one ``(entity, concern)`` pair. The bar's left/right
    edges mark the activation window
    (``activation_window_start``\u2192``activation_window_end`` if set,
    else the concern's full lifespan); colour encodes polarity (green
    desire / red fear); opacity / width encodes salience replayed at
    each fabula time. A vertical dotted line marks the current cursor
    when present.

    ``axis`` selects the time axis for the x-axis label / sample
    points: ``"fabula"`` (default) plots over story-world time;
    ``"syuzhet"`` plots over reading order. Concern *replay* itself
    is fabula-native, so on the syuzhet axis we resolve each syuzhet
    sample point to its corresponding fabula time before reconstruction.
    """
    from shadow_loom.models import reconstruct_concern_at
    from shadow_loom_ui.viz_helpers import resolve_cursor

    use_syuzhet = (axis or "fabula").lower() == "syuzhet"
    if use_syuzhet:
        sample_axis_values: list[int] = sorted({
            int(evt.syuzhet_index) for evt in (ws.events or [])
        })
        # Map each syuzhet point back to the fabula time at-or-before
        # so concern replay still reflects story-world causality.
        time_pairs: list[tuple[int, int]] = [
            (s, int(resolve_cursor(ws, "syuzhet", s) or 0))
            for s in sample_axis_values
        ]
        axis_label = "syuzhet index \u2192"
    else:
        sample_axis_values = sorted({
            int(evt.fabula_time) for evt in (ws.events or [])
        })
        time_pairs = [(t, t) for t in sample_axis_values]
        axis_label = "fabula time \u2192"

    if not time_pairs:
        return ui.label(
            "No fabula events to plot concerns against."
        ).classes("text-grey q-pa-md")
    tmin, tmax = sample_axis_values[0], sample_axis_values[-1]
    if tmax == tmin:
        # Degenerate single-point axis would make every bar zero-width.
        # Pad the range so users see at least one cell per concern.
        tmax = tmin + 1

    rows: list[str] = []  # y-axis labels
    bars: list[dict] = []

    for ent in ws.entities.values():
        for c in ent.concerns:
            label = f"{ent.name} \u2014 {c.kind or c.concern_id}"
            rows.append(label)
            # Sample salience at each axis point and emit a bar
            # segment per consecutive identical-polarity active span.
            samples: list[tuple[int, str, float, bool]] = []
            for axis_t, fabula_t in time_pairs:
                snap = reconstruct_concern_at(c, fabula_t)
                samples.append((
                    axis_t, snap["polarity"],
                    float(snap["salience"]),
                    bool(snap["active"]),
                ))
            # Compress into runs.
            i = 0
            while i < len(samples):
                t0, pol0, sal0, act0 = samples[i]
                if not act0:
                    i += 1
                    continue
                j = i + 1
                while j < len(samples):
                    tj, polj, salj, actj = samples[j]
                    if not actj or polj != pol0:
                        break
                    j += 1
                t1 = samples[j - 1][0]
                # Mean salience across the run.
                run_sal = sum(s[2] for s in samples[i:j]) / max(1, j - i)
                color = "#16a34a" if pol0 == "desire" else "#dc2626"
                bars.append({
                    "name": f"{label} ({pol0})",
                    # value[1] MUST be the category label string, not
                    # an integer index — ECharts custom series + category
                    # yAxis coerces to string for the category lookup,
                    # and integer indices don't match the string
                    # ``rows`` data so every bar drops out silently.
                    "value": [t0, label, t1, run_sal],
                    "itemStyle": {
                        "color": color,
                        "opacity": 0.25 + 0.6 * run_sal,
                    },
                })
                i = j

    if not bars:
        return ui.label(
            "No active concerns in this fabula range."
        ).classes("text-grey q-pa-md")

    # Render as a custom series of rectangles.
    height_px_per_row = max(14, min(28, int(420 / max(1, len(rows)))))

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "item",
            "formatter": (
                "function(p){var d=p.data.value;"
                "return p.name+'<br/>t '+d[0]+'\u2192'+d[2]"
                "+'<br/>mean salience: '+d[3].toFixed(2);}"
            ),
        },
        "grid": {"top": 24, "left": 220, "right": 30, "bottom": 50},
        "xAxis": {
            "type": "value",
            "min": tmin, "max": tmax,
            "name": axis_label,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "yAxis": {
            "type": "category",
            "data": rows,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "series": [{
            "type": "custom",
            "renderItem": (
                "function(params, api){"
                "var t0 = api.value(0); var row = api.value(1);"
                "var t1 = api.value(2);"
                "var p0 = api.coord([t0, row]);"
                "var p1 = api.coord([t1, row]);"
                "var h = " + str(height_px_per_row) + ";"
                "return {type:'rect', shape:{"
                "  x: p0[0], y: p0[1] - h/2,"
                "  width: Math.max(2, p1[0]-p0[0]), height: h"
                "}, style: api.style()};"
                "}"
            ),
            "encode": {"x": [0, 2], "y": 1, "tooltip": [0, 2, 3]},
            "data": bars,
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
    axis: str = "fabula",
) -> ui.echart:
    """Animated entity×entity heatmap scrubbing across time.

    Same colour scale and axes as :func:`render_relationship_heatmap`,
    but wrapped in an ECharts ``timeline`` so each step shows the dyad
    matrix as it stood at that tick. ``axis`` selects whether the
    timeline scrubs fabula chronology or syuzhet reading order; in
    syuzhet mode each frame is reconstructed at the latest revealed
    fabula tick via :func:`viz_helpers.resolve_cursor`. Frames come
    from :func:`relationship_heatmap_frames`, which layers
    ``mutation_social`` causal edges and authored snapshots on top of
    the steady-state ``RelationshipEdge`` baseline (see
    :func:`viz_helpers.reconstruct_relationship_with_causal`).
    """
    from shadow_loom_ui.viz_helpers import relationship_heatmap_frames

    payload = relationship_heatmap_frames(
        ws, metric=metric, num_frames=num_frames, axis=axis,
    )
    names = payload["names"]
    times = payload["times"]
    fabula_times = payload.get("fabula_times") or times
    frames = payload["frames"]
    axis_used = payload.get("axis", axis)
    if not names or not frames:
        return ui.label(
            f"No {metric} data over time — no social edges between entities."
        ).classes("text-grey q-pa-md")

    cursor_glyph = "s" if axis_used == "syuzhet" else "t"

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
    if n <= 15:
        label_interval = 0
    elif n <= 25:
        label_interval = 1
    else:
        label_interval = max(1, n // 20)

    base_option = {
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "position": "top"},
        # Bottom: ~70px x-axis names + ~45px visualMap + ~60px timeline
        # + breathing room. Without this the rotated labels collide with
        # the visualMap, which in turn collides with the timeline strip.
        "grid": {"top": 30, "bottom": 185, "left": 130, "right": 30},
        "xAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {
                "rotate": 45,
                "color": _CHART_TEXT,
                "fontSize": 10,
                "interval": label_interval,
                "width": 110,
                "overflow": "truncate",
                "ellipsis": "…",
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
                "width": 120,
                "overflow": "truncate",
                "ellipsis": "…",
            },
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": vmin,
            "max": vmax,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 75,
            "inRange": {"color": ramp},
            "textStyle": {"color": _CHART_TEXT},
        },
        "timeline": {
            "axisType": "category",
            "data": [f"{cursor_glyph}={t}" for t in times],
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
                "text": (
                    f"{metric}  @  s={t} → t={ft}"
                    if axis_used == "syuzhet"
                    else f"{metric}  @  fabula t={ft}"
                ),
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
        for t, ft, frame in zip(times, fabula_times, frames)
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
                " if(d.superseded){"
                "  lines.push('<span style=\"color:#F59E0B\">"
                "\u2933 superseded by ' +"
                "    (d.superseded_by_event_id || '?') + '</span>');"
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
    fabula_t: int | None = None,
    compare_with: str | None = None,
) -> ui.echart:
    """Stepped trait-evolution chart for one entity (optionally overlaid
    with a second entity for comparison).

    Improvements over the legacy version:
      * x-axis is ``type: "value"`` so a 1-tick gap and a 1000-tick
        gap render at proportional widths (the previous category
        axis evenly distributed every event tick, hiding pacing).
      * Symbol markers are suppressed once the timeline has more than
        30 sample points to stop the chart turning into polka-dots.
      * A horizontal ``markLine`` at ``y=0`` anchors signed traits
        (good vs evil, hope vs fear); a vertical ``markLine`` at
        ``fabula_t`` is the shared "now" cursor used by every
        Temporal chart.
      * ``compare_with`` overlays a second entity's same-named
        traits as dashed lines on the same axis — much clearer than
        flipping to a separate Comparison tab for a one-off check.
    """
    from shadow_loom_ui.viz_helpers import (
        cursor_markline_series, temporal_xaxis_options,
    )

    data = entity_state_timeline_data(entity_id, ws)
    if not data["times"]:
        return ui.label("No temporal data.").classes("text-grey text-caption")

    ent = ws.entities.get(entity_id)
    title = ent.name if ent else entity_id
    n_pts = len(data["times"])
    show_symbols = n_pts <= 30

    series: list[dict] = []
    colors = CHART_COLORS
    for i, (trait_name, values) in enumerate(data["series"].items()):
        # Pair each value with its fabula time so type:value renders
        # the line with proportional spacing.
        paired = [[t, v] for t, v in zip(data["times"], values)]
        series.append({
            "name": trait_name,
            "type": "line",
            "step": "middle",
            "data": paired,
            "lineStyle": {"width": 2},
            "symbol": "circle" if show_symbols else "none",
            "symbolSize": 6,
            "showSymbol": show_symbols,
            "itemStyle": {"color": colors[i % len(colors)]},
        })

    # Optional second-entity overlay (dashed lines, same trait names).
    legend_names = list(data["series"].keys())
    if compare_with and compare_with != entity_id:
        cmp_data = entity_state_timeline_data(compare_with, ws)
        cmp_ent = ws.entities.get(compare_with)
        cmp_label = cmp_ent.name if cmp_ent else compare_with
        for i, (trait_name, values) in enumerate(cmp_data["series"].items()):
            if trait_name not in data["series"]:
                continue  # only overlay traits that exist on both
            paired = [[t, v] for t, v in zip(cmp_data["times"], values)]
            cmp_name = f"{trait_name} ({cmp_label})"
            series.append({
                "name": cmp_name,
                "type": "line",
                "step": "middle",
                "data": paired,
                "lineStyle": {
                    "width": 2, "type": "dashed", "opacity": 0.7,
                },
                "symbol": "circle" if show_symbols else "none",
                "symbolSize": 5,
                "itemStyle": {"color": colors[i % len(colors)]},
            })
            legend_names.append(cmp_name)

    cursor_s = cursor_markline_series(fabula_t)
    if cursor_s:
        series.append(cursor_s)

    xaxis = temporal_xaxis_options(ws)

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
            "data": legend_names,
            "textStyle": {"color": _CHART_TEXT},
            "top": 24,
        },
        "grid": {"top": 64, "bottom": 30, "left": 50, "right": 20},
        "xAxis": xaxis,
        "yAxis": {
            "type": "value",
            "name": "Trait Value",
            "min": -1,
            "max": 1,
            "nameTextStyle": {"color": _CHART_TEXT},
            "axisLabel": {"color": _CHART_TEXT},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
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
    focus_id: str | None = None,
    focus_max_hops: int = 2,
) -> ui.echart:
    """Force-directed graph of causal edges, thickness = causal_force.

    ``layout`` options:
      - ``"force"`` (default): physics-based layout, good for medium graphs.
      - ``"circular"``: nodes on a ring, ideal for symmetry inspection.
      - ``"timeline"``: physics layout but with event nodes pre-seeded
        at their ``fabula_time`` x-coordinate, so the cascade reads
        left-to-right while the force solver still spreads vertical
        clusters apart. Best default for "what caused what" reading.
      - ``"cartesian"``: anchors events on (fabula_time, syuzhet_index)
        axes so the temporal flow of causality is preserved exactly.

    ``focus_id`` restricts the rendered graph to the BFS neighbourhood
    of that node (\u00b1``focus_max_hops`` along the directed causal
    graph). The focus node gets a gold halo so the chain anchor is
    obvious. Edges are coloured by ``causality_type`` modality (see
    :data:`shadow_loom_ui.viz_helpers.MODALITY_COLORS`).

    For >80 nodes the force layout auto-tunes its repulsion / friction
    (Webkit-dep style) so the graph doesn't explode into a hairball.
    """
    if layout == "cartesian":
        return _render_causal_cartesian(
            ws, on_click=on_click, height=height,
            highlight_edge_ids=highlight_edge_ids,
            focus_id=focus_id, focus_max_hops=focus_max_hops,
        )

    nodes, links, cats = ws_to_causal_force_data(
        ws,
        focus_id=focus_id,
        focus_max_hops=focus_max_hops,
        layout_hint=layout,
    )
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
        if layout == "timeline":
            # Pre-seeded x positions on event nodes (set by
            # ws_to_causal_force_data when layout_hint='timeline')
            # act as a soft anchor: the force solver still moves
            # nodes but won't drag them across the temporal axis,
            # so the cascade reads left-to-right naturally without
            # collapsing into a single column the way Cartesian
            # mode can when many events share a fabula_time.
            force = {**force, "initLayout": "none", "gravity": 0.04}
        series_extra = {"layout": "force", "force": force}

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "series": [{
            "type": "graph",
            "roam": True,
            "draggable": True,
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 4}},
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
    focus_id: str | None = None,
    focus_max_hops: int = 2,
) -> ui.echart:
    """Causal graph anchored on (fabula_time, syuzhet_index) axes.

    Inspired by the ECharts ``graph-life-expectancy`` example: nodes
    keep their temporal coordinates while edges curve between them so
    the visual reading order matches the story order.
    """
    nodes, links = ws_to_causal_cartesian_data(
        ws, focus_id=focus_id, focus_max_hops=focus_max_hops,
    )
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
    x_interval = (
        0 if len(target_names) <= 15
        else 1 if len(target_names) <= 25
        else max(1, len(target_names) // 20)
    )
    y_interval = (
        0 if len(ent_names) <= 15
        else 1 if len(ent_names) <= 25
        else max(1, len(ent_names) // 20)
    )

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "position": "top"},
        # Bottom: rotated 45° names ~70px + visualMap ~45px + breathing.
        "grid": {"top": 30, "bottom": 130, "left": 130, "right": 30},
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
                "width": 110,
                "overflow": "truncate",
                "ellipsis": "…",
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
                "width": 120,
                "overflow": "truncate",
                "ellipsis": "…",
            },
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": 0,
            "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 10,
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
    domains_list: list[str] = list(getattr(wt, "affected_domains", []) or []) if wt else []
    category = getattr(wt, "category", "") if wt else ""
    proposition_id = getattr(wt, "proposition_id", None) if wt else None
    # Attenuation factor (1 - inertia) is the actual fraction of an
    # impulse that survives the per-chunk merge fold: see
    # ``_apply_world_trait_chunk_updates`` which folds new snapshots
    # via ``base + (1-inertia) * (target - base)``.
    attenuation_factor = max(0.0, min(1.0, 1.0 - inertia))

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
                        # Attenuation badge: shows the per-chunk merge-fold
                        # damping factor so the reader can see at a glance
                        # how much of an authored impulse will land.
                        ui.label(f"attn ×{attenuation_factor:.2f}").classes(
                            "px-2 py-0.5 rounded-full bg-amber-100 "
                            "text-amber-800 font-mono"
                        ).tooltip(
                            "Per-chunk merge-fold attenuation: "
                            f"value_after = base + {attenuation_factor:.2f} × (target − base). "
                            "High inertia ⇒ small attenuation factor ⇒ slow movement."
                        )
                        if proposition_id:
                            ui.label(f"⇄ {proposition_id}").classes(
                                "px-2 py-0.5 rounded-full bg-indigo-100 "
                                "text-indigo-800 font-mono"
                            ).tooltip(
                                f"Linked proposition: {proposition_id}. "
                                "Pearl-Rung-2 truth clamps on this proposition "
                                "also shift this world trait."
                            )
                    # Per-domain chips replace the flat csv text — one
                    # rounded pill per affected domain so the canonical
                    # 7-domain set is visually scannable.
                    if domains_list:
                        with ui.row().classes("w-full flex-wrap gap-1 text-xs"):
                            for d in domains_list:
                                ui.label(d).classes(
                                    "px-2 py-0.5 rounded-full bg-slate-100 "
                                    "text-slate-600 font-mono"
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

    # Detect a fully-mirrored prior (every observed metric is False).
    # The model synthesises reverse-direction edges as weak fallback
    # priors when only one direction was extracted; without a badge
    # the card looks like ground truth.
    metric_objs = [
        m for m in (rel.metrics.values() if rel.metrics else [])
    ]
    is_mirror = (
        bool(metric_objs)
        and all(not getattr(m, "observed", True) for m in metric_objs)
    )
    es_color = {
        "strong": "#16a34a", "moderate": "#f59e0b", "weak": "#94a3b8",
    }

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
            with ui.row().classes("items-baseline gap-2 min-w-0"):
                ui.label(f"{src_name}  →  {tgt_name}").classes(
                    "text-sm font-semibold text-slate-800 truncate"
                )
                if is_mirror:
                    ui.label("mirrored prior").classes(
                        "text-[10px] uppercase px-1.5 py-0.5 rounded "
                        "bg-amber-50 text-amber-700 "
                        "border border-amber-200"
                    ).tooltip(
                        "All metrics on this dyad were synthesised "
                        "from the reverse direction as a weak prior "
                        "(no direct extraction). Power_dynamic is "
                        "sign-flipped; affinity/fear copy the forward "
                        "value; inertia is halved."
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
                        # Per-axis evidence dot: green/amber/grey for
                        # strong/moderate/weak. Distinct from the
                        # edge-level rollup so the user sees the
                        # actual abduction variance per axis.
                        es = getattr(m, "evidence_strength", "moderate")
                        ui.label("●").classes(
                            "text-[10px] w-3 text-center"
                        ).style(
                            f"color: {es_color.get(es, '#94a3b8')}"
                        ).tooltip(
                            f"Extraction evidence: {es} "
                            f"— last_updated_fabula="
                            f"{getattr(m, 'last_updated_fabula', 0)}"
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
    group_by: str = "trait",
    top_n: int = 5,
    fabula_t: int | None = None,
) -> ui.element:
    """ThemeRiver showing trait *energy* (|value|) flowing across fabula time.

    Improvements over the legacy version:
      * Sparse-data guard: hides itself with a friendly hint when the
        story has fewer than 4 distinct event ticks (themeriver bands
        below that read as a rendering bug).
      * ``group_by="trait"`` (new default) collapses bands across the
        cast so each trait is one ribbon \u2014 readable at any cast size.
        Pass ``group_by="entity"`` for the legacy ``entity:trait``
        rivulets.
      * Trait selection ranks by variance \u00d7 occurrence so flat axes
        don't waste vertical space.
      * Bands encode |trait| as thickness so negative values
        contribute (the legacy code shifted [0,1] traits to [0.1,1.1]
        and silently dropped negative-range traits).
      * Shared "now" cursor as a vertical guide line synced with
        every other Temporal chart.
    """
    from shadow_loom_ui.viz_helpers import (
        ws_to_theme_river_data, axis_bounds,
    )

    tmin, tmax = axis_bounds(ws)
    if (tmax - tmin) < 4:
        return ui.label(
            "Story too short for ThemeRiver \u2014 add more events to "
            "see trait flow."
        ).classes("text-grey q-pa-md text-caption italic")

    data = ws_to_theme_river_data(
        ws,
        trait_names=trait_names,
        max_entities=max_entities,
        group_by=group_by,
        top_n=top_n,
    )
    if not data:
        return ui.label("No temporal data for ThemeRiver.").classes(
            "text-grey q-pa-md"
        )

    legends = sorted({d[2] for d in data})

    series: list[dict] = [{
        "type": "themeRiver",
        "data": data,
        "label": {"show": False},
        "emphasis": {
            "itemStyle": {
                "shadowBlur": 20,
                "shadowColor": "rgba(0,0,0,0.3)",
            },
        },
    }]
    # ThemeRiver uses ``singleAxis`` rather than xAxis, so we can't
    # piggy-back on cursor_markline_series (which targets xAxis). Add
    # a tiny ``markArea`` on the themeRiver series to highlight the
    # cursor tick instead \u2014 visually equivalent.
    if fabula_t is not None:
        cursor_str = str(int(fabula_t))
        series[0]["markLine"] = {
            "silent": True,
            "symbol": ["none", "none"],
            "lineStyle": {
                "color": "#FF6B35", "width": 2,
                "type": "dashed", "opacity": 0.85,
            },
            "label": {
                "show": True, "formatter": "now",
                "color": "#FF6B35", "fontSize": 10,
                "backgroundColor": "rgba(255,255,255,0.85)",
                "padding": [1, 4, 1, 4], "borderRadius": 3,
            },
            "data": [{"xAxis": cursor_str}],
        }

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
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        },
        "series": series,
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
    on_seek=None,
    height: str | None = None,
    show_status_marks: bool = True,
    fabula_t: int | None = None,
    event_types: set[str] | None = None,
) -> ui.echart:
    """Swim-lane Gantt chart: events grouped by actor, x = fabula_time.

    Single-tick events render as small coloured glyphs (rather than
    1-tick rectangles that pretend to be Gantt bars but read as
    scatter); events that genuinely span time render as proper bars.
    The x-axis is locked to the world's full fabula range via
    :func:`temporal_xaxis_options` so the cursor line lands at the
    same screen-x as the lifeline / themeriver above.

    ``event_types``: optional whitelist of ``event_type`` strings.
    ``fabula_t``: shared "now" cursor.
    ``on_seek``: when supplied, called with an integer fabula time on
    glyph click so the World tab moves the cursor.
    """
    from shadow_loom_ui.viz_helpers import (
        EVENT_TYPE_COLORS, temporal_xaxis_options, cursor_markline_series,
    )

    actor_names, items = ws_to_gantt_data(ws)
    if event_types is not None:
        items = [it for it in items if it["event_type"] in event_types]
    if not items:
        return ui.label("No actor events for swim lanes.").classes("text-grey q-pa-md")

    if height is None:
        height = f"{max(400, 22 * len(actor_names) + 80)}px"

    # Each datum: [actor_idx, start, end, type_idx, description, event_id, color]
    legend_types = sorted({it["event_type"] for it in items})
    type_to_idx = {et: idx for idx, et in enumerate(legend_types)}
    data = [
        [
            it["actor_idx"], it["start"], it["end"],
            type_to_idx[it["event_type"]],
            it["description"], it["event_id"],
            EVENT_TYPE_COLORS.get(it["event_type"], "#94a3b8"),
        ]
        for it in items
    ]

    # Per-piece colour swatches drive the legend so toggling a type
    # in the legend hides those events.
    pieces = [
        {"value": idx, "color": EVENT_TYPE_COLORS.get(et, "#94a3b8"), "label": et}
        for idx, et in enumerate(legend_types)
    ]

    chart_opts: dict = {
        "backgroundColor": _CHART_BG,
        "tooltip": {
            **_CHART_TOOLTIP,
            "trigger": "item",
            ":formatter": (
                "function (p) {"
                "  var v = p.value;"
                "  if (!Array.isArray(v)) return p.name || '';"
                "  return '<b>' + (v[5] || '') + '</b><br/>'"
                "       + 'actor: ' + (p.name || '') + '<br/>'"
                "       + 't: ' + v[1]"
                "       + (v[2] > v[1] + 1 ? ' \u2192 ' + v[2] : '') + '<br/>'"
                "       + (v[4] || '');"
                "}"
            ),
        },
        "grid": {"top": 40, "bottom": 40, "left": 130, "right": 30},
        "xAxis": temporal_xaxis_options(ws),
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
            # Encoding the colour at index 6 lets the renderItem read it
            # back via api.value(6); visualMap on dim 3 still drives
            # the legend-toggle behaviour on event types.
            "encode": {"x": [1, 2], "y": 0, "tooltip": [1, 2, 4]},
            "data": data,
            ":renderItem": (
                "function (params, api) {"
                "  var y    = api.coord([0, api.value(0)])[1];"
                "  var x1   = api.coord([api.value(1), 0])[0];"
                "  var x2   = api.coord([api.value(2), 0])[0];"
                "  var lane = api.size([0, 1])[1] * 0.55;"
                "  var w    = x2 - x1;"
                "  var fill = api.visual('color') || api.value(6);"
                "  // Single-tick events: render as a glyph (circle)"
                "  // instead of a 1-tick rectangle so they don't"
                "  // masquerade as duration."
                "  if (w < 6) {"
                "    var r = Math.min(7, lane / 2);"
                "    return {"
                "      type: 'circle',"
                "      shape: {cx: x1, cy: y, r: r},"
                "      style: api.style({fill: fill, stroke: '#1E2A3A', lineWidth: 0.5})"
                "    };"
                "  }"
                "  return {"
                "    type: 'rect',"
                "    shape: {x: x1, y: y - lane/2, width: Math.max(2, w), height: lane, r: 2},"
                "    style: api.style({fill: fill, stroke: '#1E2A3A', lineWidth: 0.5})"
                "  };"
                "}"
            ),
        }],
    }

    if show_status_marks:
        marks = ws_to_gantt_status_marks(ws)
        actor_set = set(actor_names)
        marks = [m for m in marks if m["actor"] in actor_set]
        if marks:
            chart_opts["series"].append({
                "type": "scatter",
                "name": "status",
                "symbol": "circle",
                "symbolSize": 14,
                "itemStyle": {
                    "color": "#D8334A",
                    "borderColor": "#1E2A3A",
                    "borderWidth": 1,
                },
                "label": {
                    "show": True, "position": "top",
                    "formatter": "{@[2]}", "fontSize": 12,
                    "color": _CHART_TEXT,
                },
                "data": [
                    {
                        "value": [m["fabula_time"], m["actor"], m["icon"]],
                        "name": f"{m['actor']} \u2192 {m['status']}",
                    }
                    for m in marks
                ],
                "tooltip": {
                    **_CHART_TOOLTIP,
                    "formatter": "{b}<br/>fabula_time = {@[0]}",
                },
                "z": 5,
            })

    cursor_s = cursor_markline_series(fabula_t)
    if cursor_s:
        chart_opts["series"].append(cursor_s)

    chart = ui.echart(chart_opts).classes("w-full").style(f"height:{height}")

    def _on_chart_click(e):
        if on_click:
            on_click(e)
        if on_seek:
            try:
                args = e.args if isinstance(e.args, dict) else {}
                v = args.get("value") or (args.get("data", {}) or {}).get("value")
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    # Custom series rows: [actor_idx, start, ...]
                    on_seek(int(v[1]))
            except Exception:
                pass

    chart.on("click", _on_chart_click)
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
    on_seek=None,
    height: str | None = None,
    fabula_t: int | None = None,
    sort_by: str = "first_appearance",
    event_types: set[str] | None = None,
) -> ui.echart:
    """Per-entity lifelines: status segments + location moves + events.

    Each character occupies one horizontal lane along fabula time.
    Coloured bars span continuous status periods (green=healthy,
    amber=injured, blue=unconscious, near-black=dead). Diamond
    markers flag every location change; small dots render every event
    the character actor'd in, coloured by ``event_type``.

    Implementation notes:
      * Status ribbons are drawn with a single ECharts ``custom``
        series that emits one rect per (entity, status-period). The
        previous implementation emitted one scatter cell per integer
        tick, which scaled as ``O(entities * tmax)`` and dropped the
        per-segment tooltip; the custom-series rendering is
        ``O(segments)`` and lights up tooltips for free.
      * Off-page periods (before the entity first appears, after a
        death) render as a thin grey hairline so the lane remains
        legible without lying about presence.
      * A red \u2716 glyph drops at the death tick and the lane stops
        there (no giant black "dead" ribbon stretching to tmax).
      * ``fabula_t`` (when supplied) draws a vertical "now" markline
        shared across every Temporal chart.
      * ``on_seek`` (when supplied) is called with an integer fabula
        time when the user clicks a status segment / event glyph /
        location diamond, so the World tab can move the time cursor
        from a chart click. ``on_click`` still fires for inspector
        binding.
    """
    from shadow_loom_ui.viz_helpers import (
        ws_to_lifeline_data, _STATUS_COLORS,
        temporal_xaxis_options, cursor_markline_series,
    )

    data = ws_to_lifeline_data(
        ws, sort_by=sort_by, event_types=event_types,
    )
    if not data["entities"]:
        return ui.label("No entities to show.").classes("text-grey q-pa-md")

    if height is None:
        height = f"{max(320, 24 * len(data['entities']) + 100)}px"

    names = [n for _eid, n in data["entities"]]

    # ---- Status ribbon: custom series, one rect per segment ----------
    # Encoded as [row, start, end, color]; the renderItem reads these
    # back via api.value(i). Storing colour in the data row keeps the
    # render JS small (no visualMap pieces required).
    ribbon_data = [
        [
            seg["row"], seg["start"], seg["end"],
            seg["status_color"], seg["status"], seg["location_name"],
            seg["duration"],
        ]
        for seg in data["segments"]
    ]
    ribbon_series = {
        "name": "status",
        "type": "custom",
        "data": ribbon_data,
        "encode": {"x": [1, 2], "y": 0, "tooltip": [1, 2, 4, 5, 6]},
        "z": 2,
        "tooltip": {
            ":formatter": (
                "function(p){"
                "  var v=p.value;"
                "  return '<b>'+p.name+' \u2014 '+v[4]+'</b><br/>'"
                "       + 't '+v[1]+' \u2192 '+v[2]+' ('+v[6]+' ticks)<br/>'"
                "       + 'at: '+(v[5] || '\u2014');"
                "}"
            ),
        },
        ":renderItem": (
            "function(params, api){"
            "  var y    = api.coord([0, api.value(0)])[1];"
            "  var x1   = api.coord([api.value(1), 0])[0];"
            "  var x2   = api.coord([api.value(2), 0])[0];"
            "  var lane = api.size([0, 1])[1] * 0.55;"
            "  var w    = Math.max(2, x2 - x1);"
            "  return {"
            "    type: 'rect',"
            "    shape: {x: x1, y: y - lane/2, width: w, height: lane, r: 3},"
            "    style: api.style({fill: api.value(3), stroke: '#1e2a3a', lineWidth: 0.5})"
            "  };"
            "}"
        ),
    }

    # ---- Off-page hairlines (before first / after life_end) ----------
    tmin, tmax = data["tmin"], data["tmax"]
    hairline_data: list[list] = []
    for ls in data["lifespans"]:
        if ls["first_t"] > tmin:
            hairline_data.append([ls["row"], tmin, ls["first_t"], "#cbd5e1"])
        if ls["last_t"] < tmax:
            hairline_data.append([ls["row"], ls["last_t"], tmax, "#e5e7eb"])
    hairline_series = {
        "name": "off-page",
        "type": "custom",
        "data": hairline_data,
        "encode": {"x": [1, 2], "y": 0},
        "z": 1,
        "silent": True,
        "tooltip": {"show": False},
        ":renderItem": (
            "function(params, api){"
            "  var y  = api.coord([0, api.value(0)])[1];"
            "  var x1 = api.coord([api.value(1), 0])[0];"
            "  var x2 = api.coord([api.value(2), 0])[0];"
            "  return {"
            "    type: 'rect',"
            "    shape: {x: x1, y: y - 1, width: Math.max(1, x2 - x1), height: 2},"
            "    style: api.style({fill: api.value(3), stroke: 'none'})"
            "  };"
            "}"
        ),
    } if hairline_data else None

    # ---- Death markers ----------------------------------------------
    death_points = [
        {
            "value": [ls["death_t"], ls["row"]],
            "name": names[ls["row"]],
        }
        for ls in data["lifespans"]
        if ls["death_t"] is not None
    ]
    death_series = {
        "name": "death",
        "type": "scatter",
        "symbol": "path://M2,2 L14,14 M14,2 L2,14",  # simple X glyph
        "symbolSize": 16,
        "data": death_points,
        "itemStyle": {"color": "#D8334A", "borderColor": "#1E2A3A", "borderWidth": 1},
        "z": 6,
        "tooltip": {"formatter": "{b} \u2014 died at t={@[0]}"},
    } if death_points else None

    # ---- Location-change diamonds -----------------------------------
    move_series = {
        "name": "location change",
        "type": "scatter",
        "symbol": "diamond",
        "symbolSize": 11,
        "data": [
            {"value": [m["time"], m["row"], m["location_name"]],
             "name": str(m["location_name"])}
            for m in data["moves"]
        ],
        "itemStyle": {"color": "#ffffff", "borderColor": "#1e2a3a", "borderWidth": 1.5},
        "z": 4,
        "tooltip": {"formatter": "Moved \u2192 {@[2]}<br/>t={@[0]}"},
    }

    # ---- Event dots --------------------------------------------------
    event_points = [
        {
            "value": [e["time"], e["row"]],
            "itemStyle": {"color": e["color"]},
            "_event_id": e["event_id"],
            "_desc": e["description"],
            "_type": e["event_type"],
            "name": e["description"],
        }
        for e in data["events"]
    ]
    event_series = {
        "name": "events",
        "type": "scatter",
        "symbol": "circle",
        "symbolSize": 7,
        "data": event_points,
        "z": 3,
        "tooltip": {
            ":formatter": (
                "function(p){"
                "  var v=p.value;"
                "  var d=p.data || {};"
                "  return '<b>'+(d._event_id || '')+'</b> ['+(d._type || '')+']<br/>'"
                "       + 't='+v[0]+'<br/>'+ (d._desc || '');"
                "}"
            ),
        },
    }

    # ---- Status legend (canonical swatches) -------------------------
    status_legend_series = [
        {
            "name": s,
            "type": "scatter",
            "data": [],
            "itemStyle": {"color": c},
            "symbol": "rect",
            "symbolSize": 10,
        }
        for s, c in _STATUS_COLORS.items()
    ]

    series: list[dict] = []
    if hairline_series:
        series.append(hairline_series)
    series.append(ribbon_series)
    series.append(move_series)
    series.append(event_series)
    if death_series:
        series.append(death_series)
    series.extend(status_legend_series)
    cursor_s = cursor_markline_series(fabula_t)
    if cursor_s:
        series.append(cursor_s)

    xaxis = temporal_xaxis_options(ws)
    # Override the data-derived axis range with the lifeline-data-
    # derived range only when the world has events outside the
    # lifeline cast (rare but possible if an entity_ids filter is
    # passed through from the toolbar).
    xaxis["min"] = min(xaxis["min"], tmin)
    xaxis["max"] = max(xaxis["max"], tmax)

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        "legend": [
            {
                "data": list(_STATUS_COLORS.keys()),
                "top": 0,
                "left": "center",
                "textStyle": {"color": _CHART_TEXT, "fontSize": 10},
                "itemWidth": 14,
                "itemHeight": 8,
            },
        ],
        "grid": {"top": 36, "bottom": 30, "left": 110, "right": 20},
        "xAxis": xaxis,
        "yAxis": {
            "type": "category",
            "data": names,
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            "axisTick": {"show": False},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        },
        "series": series,
    }).classes("w-full").style(f"height:{height}")

    def _on_chart_click(e):
        if on_click:
            on_click(e)
        if on_seek:
            try:
                args = e.args if isinstance(e.args, dict) else {}
                v = args.get("value") or (args.get("data", {}) or {}).get("value")
                if isinstance(v, (list, tuple)) and v:
                    # The status ribbon emits [row, start, end, ...];
                    # the others emit [time, row, ...]. Heuristic:
                    # whichever of the first two slots is the larger
                    # absolute integer is most likely "time".
                    candidate = v[1] if args.get("seriesName") == "status" else v[0]
                    if candidate is not None:
                        on_seek(int(candidate))
            except Exception:
                pass

    chart.on("click", _on_chart_click)
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


# ── Character emotion grid (OCC appraisals per character) ─────────

def render_character_emotion_heatmap(
    grid: dict[str, dict[str, float]],
    *,
    entity_names: dict[str, str] | None = None,
    height: str = "320px",
) -> ui.echart:
    """OCC character-felt emotion heatmap (entities × emotions).

    ``grid`` is the mapping returned by
    :func:`shadow_loom_ui.viz_helpers.compute_character_emotion_grid`:
    ``{entity_id: {emotion: scalar}}``. Cells are coloured by score
    (white \u2192 deep crimson) so a glance reveals which character is
    saturating which emotion at the current cursor.
    """
    if not grid:
        return ui.label(
            "No per-character emotion data."
        ).classes("text-grey q-pa-md")

    emotions = ["fear", "joy", "regret", "grief", "rage", "love"]
    eids = list(grid.keys())
    names = entity_names or {}
    y_labels = [names.get(eid, eid) for eid in eids]

    data = []
    for yi, eid in enumerate(eids):
        row = grid.get(eid, {})
        for xi, emo in enumerate(emotions):
            v = float(row.get(emo, 0.0) or 0.0)
            data.append([xi, yi, round(v, 3)])

    options = {
        "tooltip": {
            "position": "top",
            ":formatter": (
                "function(p){return p.marker + "
                "p.value[2].toFixed(3);}"
            ),
        },
        "grid": {
            "left": 140, "right": 30, "top": 30, "bottom": 60,
            "containLabel": True,
        },
        "xAxis": {
            "type": "category",
            "data": [e.title() for e in emotions],
            "splitArea": {"show": True},
            "axisLabel": {"fontSize": 11},
        },
        "yAxis": {
            "type": "category",
            "data": y_labels,
            "splitArea": {"show": True},
            "axisLabel": {
                "fontSize": 11,
                "width": 130,
                "overflow": "truncate",
                "ellipsis": "…",
            },
        },
        "visualMap": {
            "min": 0.0,
            "max": 1.0,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 5,
            "inRange": {
                "color": [
                    "#ffffff", "#fde0dd", "#fa9fb5",
                    "#dd3497", "#7a0177",
                ],
            },
            "textStyle": {"fontSize": 10},
        },
        "series": [
            {
                "name": "score",
                "type": "heatmap",
                "data": data,
                "label": {
                    "show": True,
                    "fontSize": 10,
                    ":formatter": (
                        "function(p){return p.value[2].toFixed(2);}"
                    ),
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 8,
                        "shadowColor": "rgba(0,0,0,0.3)",
                    },
                },
            }
        ],
    }
    return ui.echart(options).classes("w-full").style(f"height: {height};")


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


# =====================================================================
# Social-layer renderers
# (Beliefs / Concerns / Propositions / Relationships unified network +
#  per-character cards + per-character trait trajectories)
# =====================================================================

def render_social_layer_graph(
    ws: WorldStateV1,
    *,
    on_click: OnClick = None,
    height: str = "100%",
    layout: str = "force",
    include_relationships: bool = True,
    include_beliefs: bool = True,
    include_concerns: bool = True,
    include_propositions: bool = True,
    fabula_t: int | None = None,
    event_t: int | None = None,
    ego_id: str | None = None,
    ego_max_hops: int = 1,
    pov_id: str | None = None,
    intermental_ids: list[str] | None = None,
    intermental_threshold: float = 0.4,
) -> ui.echart:
    """Combined entity / proposition / concern / belief / relationship graph.

    Replaces the social-only graph (which only showed entity-entity
    affinity edges) with a richer network where:
      • Characters are circles (coral),
      • Propositions are diamonds (iris) sized by stakes,
      • Desires/fears are dashed green/red arrows from the holder
        to the proposition (sized by salience), and
      • Beliefs are solid amber arrows from holder to proposition
        (coloured by confidence).
    All time-sliced when ``fabula_t`` is provided.
    """
    from shadow_loom_ui.viz_helpers import ws_to_social_layer_graph

    nodes, links, cats = ws_to_social_layer_graph(
        ws,
        include_relationships=include_relationships,
        include_beliefs=include_beliefs,
        include_concerns=include_concerns,
        include_propositions=include_propositions,
        fabula_t=fabula_t,
        event_t=event_t,
        ego_id=ego_id,
        ego_max_hops=ego_max_hops,
        pov_id=pov_id,
        intermental_ids=intermental_ids,
        intermental_threshold=intermental_threshold,
    )
    if not nodes:
        return ui.label(
            "No social-layer data (no characters, propositions, "
            "concerns or beliefs in this world)."
        ).classes("text-grey q-pa-md")

    show_labels = len(nodes) <= 35
    series_extra: dict
    if layout == "circular":
        series_extra = {
            "layout": "circular",
            "circular": {"rotateLabel": True},
        }
    else:
        series_extra = {
            "layout": "force",
            "force": {
                "repulsion": 420,
                "gravity": 0.12,
                "edgeLength": [90, 220],
            },
            # ``autoCurveness`` makes ECharts spread overlapping
            # edges between the same node pair onto separate
            # curves. Without it the entity\u2192prop belief edge,
            # the entity\u2192prop direct desire/fear edge, and the
            # entity\u2192concern\u2192prop two-hop path collapse
            # visually onto each other and the desire/fear
            # signalling looks like there's only a belief edge.
            "autoCurveness": True,
        }

    chart = ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "item"},
        # Legend intentionally omitted: ECharts' category legend would
        # show only the three node categories and would mislead viewers
        # into thinking that's the full visual encoding (it isn't —
        # belief / relationship edges, polarity colours, salience and
        # stakes sizings are all extra channels). The chip-strip
        # legend rendered above the chart documents the real encoding.
        "animationDuration": 600,
        "series": [{
            "type": "graph",
            "roam": True,
            "draggable": True,
            "emphasis": {"focus": "adjacency"},
            "categories": cats,
            "data": nodes,
            "links": links,
            "label": {
                "show": show_labels,
                "position": "right",
                "fontSize": 11,
                "color": _CHART_TEXT,
            },
            # NOTE: do NOT set ``lineStyle`` at the series level here
            # \u2014 ECharts treats the series-level ``lineStyle`` as a
            # default that *overrides* per-link ``lineStyle`` for any
            # property the link doesn't explicitly set, and in
            # practice the wrapper merges them in a way that flattens
            # per-link ``type`` (dashed vs solid) and ``curveness``,
            # making belief edges, desire/fear edges, and concern
            # \u2192 prop edges all look identical. Per-link styling
            # (set in viz_helpers.ws_to_social_layer_graph) carries
            # the full encoding.
            **series_extra,
        }],
    }).classes("w-full").style(f"height:{height}")
    if on_click:
        chart.on("click", on_click)
    return chart


def render_entity_concern_card(
    ws: WorldStateV1,
    entity_id: str,
    *,
    fabula_t: int | None = None,
    height: str = "100%",
) -> ui.element:
    """One character's concerns as a textual card list.

    Mirrors :func:`render_entity_belief_chart`. Each row shows the
    referenced proposition, polarity (desire / fear) badge, salience
    bar, kind chip, and a faint indicator if currently inactive.
    """
    from shadow_loom_ui.viz_helpers import ws_to_entity_concern_rows

    rows = ws_to_entity_concern_rows(ws, entity_id, fabula_t=fabula_t)
    ent = ws.entities.get(entity_id)
    title = ent.name if ent else entity_id

    container = ui.column().classes("w-full bg-white").style(
        f"min-height:{height}"
    )
    with container:
        with ui.row().classes(
            "w-full items-baseline justify-between px-3 py-2 "
            "border-b border-slate-200 bg-slate-50"
        ):
            ui.label(title).classes("text-sm font-semibold text-slate-800")
            ui.label(
                f"{len(rows)} concern{'s' if len(rows) != 1 else ''}"
            ).classes("text-xs text-slate-500")
        if not rows:
            ui.label("No concerns.").classes(
                "text-xs text-slate-400 italic px-3 py-3"
            )
            return container
        with ui.column().classes(
            "w-full gap-2 px-3 py-2 overflow-y-auto"
        ).style("max-height: 360px"):
            for r in rows:
                pol = r["polarity"]
                pol_color = "#16a34a" if pol == "desire" else "#dc2626"
                pol_label = "desires" if pol == "desire" else "fears"
                sal = float(r["salience"])
                inactive = not r["active"]
                with ui.column().classes(
                    "w-full gap-1 p-2 rounded-lg border border-slate-200 "
                    "bg-slate-50/60 hover:bg-slate-100/60 transition-colors"
                    + (" opacity-60" if inactive else "")
                ):
                    ui.label(
                        f"\u201C{r['proposition_desc']}\u201D"
                    ).classes(
                        "text-sm text-slate-800 leading-snug"
                    )
                    with ui.row().classes(
                        "w-full items-center gap-2 text-xs"
                    ):
                        ui.label(pol_label).classes(
                            "px-2 py-0.5 rounded-full text-white font-mono"
                        ).style(f"background-color: {pol_color}")
                        ui.label(f"sal {sal:.2f}").classes(
                            "px-2 py-0.5 rounded-full bg-slate-200 "
                            "text-slate-700 font-mono"
                        )
                        if r.get("kind"):
                            ui.label(str(r["kind"])).classes(
                                "px-2 py-0.5 rounded-full bg-violet-50 "
                                "text-violet-700 border border-violet-200"
                            )
                        if inactive:
                            ui.label("inactive").classes(
                                "px-2 py-0.5 rounded-full bg-slate-300 "
                                "text-slate-700 font-mono"
                            )
                    # Salience bar.
                    with ui.element("div").classes(
                        "w-full h-1.5 rounded-full bg-slate-200 overflow-hidden"
                    ):
                        ui.element("div").classes("h-full rounded-full").style(
                            f"width: {int(sal * 100)}%; background-color: {pol_color}"
                        )
    return container


def render_concern_salience_heatmap(
    ws: WorldStateV1,
    *,
    fabula_t: int | None = None,
    selected_ids: list[str] | None = None,
    height: str = "320px",
) -> ui.element:
    """Entity × concern salience heatmap at the active fabula cursor.

    Implements the Sternberg-triad / Frijda concern-salience lens at
    a glance: rows are characters that hold concerns, columns are
    each unique concern (labelled by the underlying proposition or
    the concern's ``name``/``id``), and the cell colour encodes
    salience replayed at ``fabula_t`` via
    :func:`reconstruct_concern_at`. Cells render blank when a
    character does not hold the concern, distinguishing "absent
    concern" from "low salience" — the curiosity-vs-suspense
    distinction in the audit lens.

    Parameters mirror the surrounding concern panels so the social
    tab can wire the same selection filter through.
    """
    from shadow_loom.models import reconstruct_concern_at

    container = ui.column().classes("w-full gap-2")
    with container:
        # Collect (entity_name, concern_label, salience) triples.
        # Rows preserve entity order; columns are unique concern
        # labels in first-seen order.
        sel = set(selected_ids) if selected_ids else None
        rows: list[str] = []
        col_index: dict[str, int] = {}
        cells: list[tuple[int, int, float]] = []
        ent_lookup: dict[str, str] = {}

        for eid, ent in (ws.entities or {}).items():
            if sel is not None and eid not in sel:
                continue
            concerns = list(getattr(ent, "concerns", None) or [])
            if not concerns:
                continue
            row_idx = len(rows)
            rows.append(ent.name or eid)
            ent_lookup[ent.name or eid] = eid
            for c in concerns:
                # Column key: prefer the concern's display name,
                # fall back to proposition_id, then concern id.
                label = (
                    getattr(c, "name", None)
                    or getattr(c, "proposition_id", None)
                    or getattr(c, "id", None)
                    or "concern"
                )
                if label not in col_index:
                    col_index[label] = len(col_index)
                col_idx = col_index[label]
                if fabula_t is None:
                    sal = float(getattr(c, "salience", 0.0) or 0.0)
                else:
                    try:
                        snap = reconstruct_concern_at(c, fabula_t)
                        sal = float(snap.get("salience", 0.0) or 0.0)
                    except Exception:
                        sal = float(getattr(c, "salience", 0.0) or 0.0)
                cells.append((col_idx, row_idx, sal))

        if not rows or not col_index:
            ui.label(
                "No concerns to plot at this cursor."
            ).classes("text-sm text-slate-500 italic q-pa-md")
            return container

        cols = sorted(col_index, key=lambda k: col_index[k])
        ui.label(
            f"Concern salience \u2014 {len(rows)} characters \u00d7 "
            f"{len(cols)} concerns"
        ).classes("text-xs font-semibold text-slate-700")
        option = {
            "tooltip": {
                "position": "top",
                "formatter": (
                    "function(p){return p.value[2]==null?'':"
                    "p.name+'<br/>salience: '+Number(p.value[2]).toFixed(2);}"
                ),
            },
            # Top: visualMap (~30px) + breathing room. Bottom: rotated
            # 30° column labels need ~50px so they don't clip.
            "grid": {"left": 160, "right": 30, "top": 60, "bottom": 70},
            "xAxis": {
                "type": "category", "data": cols,
                "axisLabel": {
                    "interval": 0, "rotate": 30, "fontSize": 10,
                    "width": 100, "overflow": "truncate", "ellipsis": "…",
                },
                "splitArea": {"show": True},
            },
            "yAxis": {
                "type": "category", "data": rows,
                "axisLabel": {
                    "fontSize": 10,
                    "width": 140, "overflow": "truncate", "ellipsis": "…",
                },
                "splitArea": {"show": True},
            },
            "visualMap": {
                "min": 0.0, "max": 1.0,
                "calculable": True, "orient": "horizontal",
                "left": "center", "top": 5,
                "inRange": {"color": ["#f1f5f9", "#3b82f6", "#1e3a8a"]},
            },
            "series": [{
                "name": "salience",
                "type": "heatmap",
                "data": [[c, r, round(v, 3)] for (c, r, v) in cells],
                "label": {"show": False},
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 6,
                        "shadowColor": "rgba(0,0,0,0.4)",
                    },
                },
            }],
        }
        ui.echart(option).classes("w-full").style(f"height: {height}")
    return container


def render_proposition_truth_sparkline(
    ws: WorldStateV1,
    proposition_id: str,
    *,
    height: str = "60px",
) -> ui.element:
    """Compact truth_at_fabula sparkline for one proposition.

    Plots the discrete True/False commits in
    :attr:`Proposition.truth_at_fabula` against fabula time so a
    researcher can see at a glance how often (and when) the
    storyworld committed to a value for the proposition. Distinct
    from :func:`render_proposition_stake_timeline` which plots
    continuous stakes / audience prior — this is just the binary
    truth track.
    """
    container = ui.element("div").classes("w-full")
    with container:
        prop = next(
            (p for p in (ws.propositions or [])
             if getattr(p, "proposition_id", None) == proposition_id),
            None,
        )
        if prop is None:
            ui.label("(no proposition)").classes(
                "text-[10px] text-slate-400 italic"
            )
            return container
        commits = dict(getattr(prop, "truth_at_fabula", {}) or {})
        if not commits:
            ui.label("no truth commits").classes(
                "text-[10px] text-slate-400 italic"
            )
            return container
        items = sorted(commits.items(), key=lambda kv: int(kv[0]))
        xs = [int(k) for k, _ in items]
        ys = [1 if bool(v) else 0 for _, v in items]
        option = {
            "grid": {"left": 4, "right": 4, "top": 4, "bottom": 4},
            "xAxis": {"type": "value", "show": False, "min": min(xs) - 1, "max": max(xs) + 1},
            "yAxis": {"type": "value", "show": False, "min": -0.2, "max": 1.2},
            "tooltip": {
                "trigger": "axis",
                "formatter": (
                    "function(ps){var p=ps[0];"
                    "return 't='+p.value[0]+'<br/>'+(p.value[1]?'true':'false');}"
                ),
            },
            "series": [{
                "type": "scatter",
                "symbolSize": 8,
                "data": [[x, y] for x, y in zip(xs, ys)],
                "itemStyle": {
                    "color": "#16a34a",
                },
            }],
        }
        ui.echart(option).classes("w-full").style(f"height: {height}")
    return container


def render_entity_concerns_grid(
    ws: WorldStateV1,
    *,
    selected_ids: list[str] | None = None,
    fabula_t: int | None = None,
    chart_height: str = "260px",
) -> ui.element:
    """Tiled grid of per-character concern cards."""
    from shadow_loom_ui.viz_helpers import list_concern_holders

    holders = list_concern_holders(ws)
    if selected_ids:
        sel = set(selected_ids)
        holders = [h for h in holders if h[0] in sel]
    container = ui.column().classes("w-full gap-3")
    with container:
        if not holders:
            ui.label(
                "No characters carry concerns in this world model."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container
        grid = ui.element("div").classes(
            "w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"
        )
        with grid:
            for eid, _name, _count in holders:
                with ui.element("div").classes(
                    "border border-slate-200 rounded-xl bg-white shadow-sm "
                    "overflow-hidden"
                ):
                    render_entity_concern_card(
                        ws, eid, fabula_t=fabula_t,
                        height=chart_height,
                    )
    return container


def render_proposition_state_card(
    ws: WorldStateV1,
    proposition_id: str,
    *,
    fabula_t: int | None = None,
) -> ui.element:
    """Snapshot card for one proposition at the current cursor."""
    from shadow_loom.models import reconstruct_proposition_at

    prop = next(
        (p for p in (ws.propositions or [])
         if p.proposition_id == proposition_id),
        None,
    )
    if prop is None:
        return ui.label(f"Unknown proposition {proposition_id}").classes(
            "text-grey"
        )
    if fabula_t is not None:
        snap = reconstruct_proposition_at(prop, fabula_t)
        stakes = float(snap["stakes"])
        prior = float(snap["audience_default_prior"])
        desc = snap["description"]
        truth = snap["truth_at"]
    else:
        stakes = float(prop.stakes)
        prior = float(prop.audience_default_prior)
        desc = prop.description
        truth = None
        for t in sorted(prop.truth_at_fabula.keys()):
            truth = prop.truth_at_fabula[t]

    # Count related concerns + beliefs.
    concern_holders = [
        ent for ent in ws.entities.values()
        for c in ent.concerns if c.proposition_id == proposition_id
    ]
    believers = [
        (ent, b) for ent in ws.entities.values()
        for b in ent.beliefs
        if getattr(b, "proposition_id", None) == proposition_id
    ]

    if truth is True:
        truth_color = "#16a34a"
        truth_text = "TRUE"
    elif truth is False:
        truth_color = "#dc2626"
        truth_text = "FALSE"
    else:
        truth_color = "#94a3b8"
        truth_text = "—"

    container = ui.column().classes("w-full bg-white")
    with container:
        with ui.row().classes(
            "w-full items-center justify-between px-3 py-2 "
            "border-b border-slate-200 bg-slate-50"
        ):
            ui.label(prop.kind).classes(
                "px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 "
                "border border-violet-200 text-xs font-mono"
            )
            ui.label(truth_text).classes(
                "px-2 py-0.5 rounded-full text-white text-xs font-mono"
            ).style(f"background-color: {truth_color}")
        with ui.column().classes("w-full gap-2 px-3 py-3"):
            ui.label(f"\u201C{desc}\u201D").classes(
                "text-sm text-slate-800 leading-snug"
            )
            with ui.row().classes("w-full items-center gap-2 text-xs"):
                ui.label("stakes").classes("text-slate-500 w-16")
                with ui.element("div").classes(
                    "flex-grow h-1.5 rounded-full bg-slate-200 overflow-hidden"
                ):
                    ui.element("div").classes(
                        "h-full rounded-full"
                    ).style(
                        f"width: {int(stakes * 100)}%; "
                        f"background-color: #f59e0b"
                    )
                ui.label(f"{stakes:.2f}").classes(
                    "font-mono text-slate-700 w-10 text-right"
                )
            with ui.row().classes("w-full items-center gap-2 text-xs"):
                ui.label("aud prior").classes("text-slate-500 w-16")
                with ui.element("div").classes(
                    "flex-grow h-1.5 rounded-full bg-slate-200 overflow-hidden"
                ):
                    ui.element("div").classes(
                        "h-full rounded-full"
                    ).style(
                        f"width: {int(prior * 100)}%; "
                        f"background-color: #3A7BD5"
                    )
                ui.label(f"{prior:.2f}").classes(
                    "font-mono text-slate-700 w-10 text-right"
                )
            with ui.row().classes("w-full items-center gap-2 text-xs text-slate-500"):
                ui.label(f"{len(concern_holders)} concerned").classes(
                    "px-2 py-0.5 rounded-full bg-slate-100"
                )
                ui.label(f"{len(believers)} believers").classes(
                    "px-2 py-0.5 rounded-full bg-slate-100"
                )
                if prop.referent_ids:
                    ui.label(
                        f"refs: {', '.join(prop.referent_ids[:3])}"
                    ).classes("italic truncate")
    return container


def render_propositions_grid(
    ws: WorldStateV1,
    *,
    fabula_t: int | None = None,
    selected_kinds: list[str] | None = None,
) -> ui.element:
    """Tiled grid of per-proposition snapshot cards."""
    props = list(ws.propositions or [])
    if selected_kinds:
        wanted = set(selected_kinds)
        props = [p for p in props if p.kind in wanted]

    container = ui.column().classes("w-full gap-3")
    with container:
        if not props:
            ui.label(
                "No propositions in this world model."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container
        grid = ui.element("div").classes(
            "w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"
        )
        with grid:
            for p in props:
                with ui.element("div").classes(
                    "border border-slate-200 rounded-xl bg-white shadow-sm "
                    "overflow-hidden"
                ):
                    render_proposition_state_card(
                        ws, p.proposition_id, fabula_t=fabula_t,
                    )
    return container


def render_entity_trait_trajectory(
    ws: WorldStateV1,
    entity_id: str,
    *,
    height: str = "260px",
    trait_names: list[str] | None = None,
) -> ui.echart:
    """Multi-line trait trajectory for one character over fabula time.

    The same trace the Social tab uses to show *how the inner state
    of each character moves* alongside their concerns / beliefs.
    """
    from shadow_loom_ui.viz_helpers import entity_trait_trajectory

    ent = ws.entities.get(entity_id)
    times, series = entity_trait_trajectory(
        ws, entity_id, trait_names=trait_names,
    )
    if not series:
        return ui.label("No trait data.").classes("text-grey q-pa-md")

    legends = list(series.keys())
    # Event markers: vertical lines at each fabula_time tick whose
    # event has either (a) a CausalEdge into this entity, or (b) a
    # belief-mutation referencing this entity. This makes "which event
    # caused which inner-state change" legible (Bremond / Todorov).
    evt_lookup = {evt.id: evt for evt in (ws.events or [])}
    causally_relevant_times: dict[int, list[str]] = {}
    for ce in ws.causal_topology:
        if ce.target_id == entity_id and ce.source_id in evt_lookup:
            evt = evt_lookup[ce.source_id]
            causally_relevant_times.setdefault(evt.fabula_time, []).append(
                (evt.description or evt.id)[:48]
            )
    # Belief-acquisition markers from the entity itself.
    ent_obj = ws.entities.get(entity_id)
    if ent_obj is not None:
        for b in ent_obj.beliefs:
            via = getattr(b, "acquired_via_event_id", None)
            if via and via in evt_lookup:
                evt = evt_lookup[via]
                causally_relevant_times.setdefault(
                    evt.fabula_time, []
                ).append(f"belief: {(b.perceived_state or '')[:40]}")
    # Build markLine x-axis indices (the xAxis is categorical on times).
    time_to_idx = {t: i for i, t in enumerate(times)}
    mark_lines: list[dict] = []
    for ft, descs in sorted(causally_relevant_times.items()):
        if ft not in time_to_idx:
            continue
        mark_lines.append({
            "xAxis": time_to_idx[ft],
            "label": {
                "formatter": (descs[0] if descs else "")[:24],
                "fontSize": 9,
                "color": "#475569",
                "rotate": 90,
                "position": "insideEndTop",
            },
            "lineStyle": {
                "color": "#f59e0b",
                "type": "dashed",
                "opacity": 0.6,
                "width": 1,
            },
        })
    chart_series = [
        {
            "name": name,
            "type": "line",
            "smooth": True,
            "symbol": "circle",
            "symbolSize": 5,
            "lineStyle": {"width": 2},
            "data": values,
        }
        for name, values in series.items()
    ]
    if mark_lines and chart_series:
        chart_series[0]["markLine"] = {
            "silent": False,
            "symbol": ["none", "none"],
            "data": mark_lines,
        }
    title = ent.name if ent else entity_id
    return ui.echart({
        "backgroundColor": _CHART_BG,
        "color": CHART_COLORS,
        "title": {
            "text": f"{title} — trait trajectory",
            "left": "center",
            "top": 4,
            "textStyle": {
                "color": _CHART_TEXT,
                "fontSize": 12,
                "fontWeight": "normal",
            },
        },
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis"},
        "legend": {
            "data": legends,
            "top": 26,
            "type": "scroll",
            "textStyle": {"color": _CHART_TEXT, "fontSize": 10},
        },
        "grid": {"left": 40, "right": 20, "top": 70, "bottom": 30},
        "xAxis": {
            "type": "category",
            "data": [str(t) for t in times],
            "name": "fabula t",
            "axisLine": {"lineStyle": {"color": _CHART_TEXT}},
        },
        "yAxis": {
            "type": "value",
            "min": 0, "max": 1,
            "axisLine": {"lineStyle": {"color": _CHART_TEXT}},
            "splitLine": {"lineStyle": {"color": "#e5e7eb"}},
        },
        "series": chart_series,
    }).classes("w-full").style(f"height:{height}")


def render_trait_trajectories_grid(
    ws: WorldStateV1,
    *,
    selected_ids: list[str] | None = None,
    chart_height: str = "260px",
) -> ui.element:
    """Tiled grid of per-character trait trajectories.

    Each card is a compact inline chart with an expand button that
    opens a full-screen version of the same trajectory (same pattern
    used by every other chart in the UI via :func:`with_expand`).
    """
    if selected_ids:
        ent_ids = [e for e in selected_ids if e in ws.entities]
    else:
        ent_ids = [e for e, ent in ws.entities.items() if ent.traits]

    container = ui.column().classes("w-full gap-3")
    with container:
        if not ent_ids:
            ui.label(
                "No characters have traits in this world model."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container
        grid = ui.element("div").classes(
            "w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"
        )
        with grid:
            for eid in ent_ids:
                ent = ws.entities.get(eid)
                title = (ent.name if ent else eid) + " — trait trajectory"
                with ui.element("div").classes(
                    "border border-slate-200 rounded-xl bg-white shadow-sm "
                    "overflow-hidden p-2"
                ):
                    with_expand(
                        lambda h, e=eid: render_entity_trait_trajectory(
                            ws, e, height=h,
                        ),
                        title=title,
                        height=chart_height,
                    )
    return container


# =====================================================================
# P1 ADDITIONS — Channels overlay, Proposition stake/surprise, axis-lint
# =====================================================================

def render_channels_overview(
    ws: WorldStateV1,
    *,
    fabula_t: int | None = None,
    state: "Any" = None,
) -> ui.element:
    """Per-channel cards exposing intelligibility, directionality, lifespan.

    The :class:`Channel` model is first-class (medium, n-ary
    participants with per-participant decode probability,
    directionality, established/terminated ticks, evidence_strength)
    but currently has no UI surface. This card grid renders one card
    per channel, badging:

      * **directionality** — broadcast / duplex / simplex glyph + chip
      * **medium** — telephone / telepathy / classified pipeline / ...
      * **intelligibility** per-participant bar (opacity = decode prob)
      * **lifespan** — ``established_at`` ... ``terminated_at`` chip
        with a strike-through when the channel is severed at or
        before the active fabula cursor
      * **evidence_strength** — extraction confidence dot

    When ``state`` is provided each per-participant intelligibility
    bar becomes an inline editor (NiceGUI slider with a Save button)
    that calls :meth:`AppState.apply_world_state_patch` with an
    ``update_channel_intelligibility`` op so the edit persists as a
    new version. Click an entity chip to deep-link the inspector via
    :data:`StateEvent.NODE_SELECTED` (handled by the social tab).
    """
    container = ui.column().classes("w-full gap-3")
    with container:
        channels = list((ws.channels or {}).values())
        if not channels:
            ui.label(
                "No standing channels in this world model. "
                "Channels model who *can* communicate (telephone, "
                "telepathy, classified pipeline, …); discrete "
                "messages are utterance-typed events that may "
                "reference a channel via via_channel_id."
            ).classes("text-sm text-slate-500 italic q-pa-lg")
            return container
        # Stable name-sorted order.
        channels.sort(key=lambda c: c.name.lower())
        grid = ui.element("div").classes(
            "w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"
        )
        es_color = {"strong": "#16a34a", "moderate": "#f59e0b", "weak": "#94a3b8"}
        dir_glyph = {"broadcast": "📢", "duplex": "↔", "simplex": "→"}
        with grid:
            for ch in channels:
                terminated = (
                    ch.terminated_at_fabula is not None
                    and fabula_t is not None
                    and ch.terminated_at_fabula <= fabula_t
                )
                with ui.element("div").classes(
                    "border border-slate-200 rounded-xl bg-white shadow-sm "
                    "overflow-hidden"
                ):
                    with ui.row().classes(
                        "w-full items-baseline justify-between px-3 py-2 "
                        "border-b border-slate-200 bg-slate-50"
                    ):
                        with ui.row().classes("items-center gap-2 min-w-0"):
                            ui.label(
                                dir_glyph.get(ch.directionality, "•")
                            ).classes("text-base")
                            name_classes = (
                                "text-sm font-semibold truncate "
                                + ("line-through text-slate-400"
                                   if terminated else "text-slate-800")
                            )
                            ui.label(ch.name).classes(name_classes)
                        ui.label(
                            "●"
                        ).classes("text-xs").style(
                            f"color: {es_color.get(ch.evidence_strength, '#94a3b8')}"
                        ).tooltip(
                            f"Extraction evidence: {ch.evidence_strength}"
                        )
                    with ui.column().classes("w-full gap-2 px-3 py-3"):
                        with ui.row().classes("items-center gap-2 text-xs flex-wrap"):
                            ui.label(ch.medium).classes(
                                "px-2 py-0.5 rounded-full bg-violet-50 "
                                "text-violet-700 border border-violet-200 "
                                "font-mono"
                            )
                            ui.label(ch.directionality).classes(
                                "px-2 py-0.5 rounded-full bg-slate-100 "
                                "text-slate-600 font-mono"
                            )
                            life = (
                                f"t {ch.established_at_fabula}"
                                + (f"–{ch.terminated_at_fabula}"
                                   if ch.terminated_at_fabula is not None
                                   else "–…")
                            )
                            life_classes = (
                                "px-2 py-0.5 rounded-full font-mono "
                                + ("bg-amber-50 text-amber-700 line-through"
                                   if terminated
                                   else "bg-slate-100 text-slate-600")
                            )
                            ui.label(life).classes(life_classes)
                        # Per-participant intelligibility (opacity ∝ decode prob).
                        with ui.column().classes("w-full gap-1 mt-1"):
                            ui.label("Participants & intelligibility").classes(
                                "text-[10px] uppercase text-slate-500 tracking-wide"
                            )
                            for pid in ch.participant_ids:
                                ent = ws.entities.get(pid)
                                obj = (ws.objects or {}).get(pid)
                                pname = (
                                    ent.name if ent
                                    else (obj.name if obj else pid)
                                )
                                # 1.0 when key absent (fully intelligible).
                                decode = float(ch.intelligibility.get(pid, 1.0))
                                if state is None:
                                    with ui.row().classes(
                                        "w-full items-center gap-2 text-xs"
                                    ):
                                        ui.label(pname).classes(
                                            "text-slate-700 truncate w-32 shrink-0"
                                        )
                                        with ui.element("div").classes(
                                            "flex-grow h-2 rounded-full bg-slate-200 "
                                            "overflow-hidden"
                                        ):
                                            # Opacity ∝ decode probability so
                                            # opaque participants (decode=0)
                                            # render as faint bars — visually
                                            # encoding "can't read this channel".
                                            opacity = max(0.15, decode)
                                            ui.element("div").classes(
                                                "h-full rounded-full"
                                            ).style(
                                                f"width: {int(decode * 100)}%; "
                                                f"background-color: #3A7BD5; "
                                                f"opacity: {opacity:.2f}"
                                            )
                                        ui.label(f"{decode:.2f}").classes(
                                            "font-mono text-slate-600 w-10 text-right"
                                        )
                                else:
                                    # Editable slider: writes through
                                    # ``state.apply_world_state_patch`` with
                                    # an ``update_channel_intelligibility``
                                    # op so the edit persists as a new
                                    # version.
                                    with ui.row().classes(
                                        "w-full items-center gap-2 text-xs"
                                    ):
                                        ui.label(pname).classes(
                                            "text-slate-700 truncate w-28 shrink-0"
                                        )
                                        slider = ui.slider(
                                            min=0.0, max=1.0, step=0.05,
                                            value=decode,
                                        ).props(
                                            "label-always color=primary "
                                            "dense"
                                        ).classes("flex-grow")
                                        readout = ui.label(f"{decode:.2f}").classes(
                                            "font-mono text-slate-600 w-10 text-right"
                                        )

                                        def _on_change(
                                            _e=None,
                                            *,
                                            s=slider,
                                            r=readout,
                                            cid=ch.id,
                                            participant=pid,
                                            pretty=pname,
                                        ) -> None:
                                            try:
                                                v = float(s.value)
                                            except (TypeError, ValueError):
                                                return
                                            v = max(0.0, min(1.0, v))
                                            r.text = f"{v:.2f}"
                                            ok, log = state.apply_world_state_patch(
                                                {
                                                    "update_channel_intelligibility": {
                                                        cid: {participant: v},
                                                    }
                                                },
                                                description=(
                                                    f"Channel intelligibility: "
                                                    f"{cid}[{pretty}] = {v:.2f}"
                                                ),
                                            )
                                            if ok:
                                                ui.notify(
                                                    "Intelligibility updated",
                                                    type="positive",
                                                    position="top-right",
                                                    timeout=1500,
                                                )
                                            else:
                                                ui.notify(
                                                    "Update failed: "
                                                    + (log[0] if log else "unknown"),
                                                    type="negative",
                                                )

                                        slider.on(
                                            "change",
                                            lambda _e=None, h=_on_change: h(),
                                        )
                        # Utterance count via this channel.
                        utter_n = sum(
                            1 for e in ws.events
                            if getattr(e, "via_channel_id", None) == ch.id
                        )
                        if utter_n:
                            ui.label(
                                f"{utter_n} utterance"
                                + ("s" if utter_n != 1 else "")
                                + " carried"
                            ).classes(
                                "text-[10px] text-slate-500 italic"
                            )
    return container


def render_proposition_stake_timeline(
    ws: WorldStateV1,
    proposition_id: str,
    *,
    height: str = "240px",
) -> ui.element:
    """Stake / audience-prior curve + Bayesian-surprise spike for one proposition.

    Surfaces three signals on a shared fabula axis:

      * **stakes** — replayed via ``reconstruct_proposition_at`` so
        narrative escalation is visible.
      * **audience prior** — likewise replayed; reflects how the
        narrator has framed the question.
      * **truth-commit spikes** — at every fabula tick where
        ``Proposition.truth_at_fabula`` commits a value, an
        Itti-Baldi-style ``-log p(P)`` marker. ``p`` is the audience's
        prior at that tick, so unexpected reveals (low p) score a
        bigger spike than telegraphed inevitabilities. This is
        Brewer-Lichtenstein "surprise" reduced to a single number.

    Returns an ECharts line chart; falls back to a label when the
    proposition is unknown or the world has no fabula span.
    """
    import math

    prop = next(
        (p for p in (ws.propositions or [])
         if p.proposition_id == proposition_id),
        None,
    )
    if prop is None:
        return ui.label(
            f"Unknown proposition {proposition_id}"
        ).classes("text-grey q-pa-md")

    from shadow_loom.models import reconstruct_proposition_at
    from shadow_loom_ui.viz_helpers import fabula_time_bounds

    tmin, tmax = fabula_time_bounds(ws)
    if tmax <= tmin:
        return ui.label(
            "Single-tick world — stake timeline needs a fabula span."
        ).classes("text-grey q-pa-md")

    # Sample evenly across the fabula span (capped) plus union the
    # explicit truth-commit ticks so spikes always land on integer
    # samples rather than being interpolated away.
    n = 32
    step = max(1, (tmax - tmin) // (n - 1))
    sample_set = set(range(tmin, tmax + 1, step))
    sample_set.add(tmax)
    sample_set.update(int(t) for t in (prop.truth_at_fabula or {}).keys())
    samples = sorted(sample_set)

    stake_pts = []
    prior_pts = []
    surprise_pts = []
    last_truth: bool | None = None
    for t in samples:
        snap = reconstruct_proposition_at(prop, t)
        s = float(snap["stakes"])
        p = float(snap["audience_default_prior"])
        stake_pts.append([t, round(s, 3)])
        prior_pts.append([t, round(p, 3)])
        truth = snap["truth_at"]
        # Itti-Baldi surprise on the *transition* into a new committed truth.
        if truth is not None and truth != last_truth:
            # p_observed = audience prior toward the realised value.
            p_obs = p if truth else (1.0 - p)
            p_obs = max(min(p_obs, 1 - 1e-6), 1e-6)
            surprise = -math.log(p_obs)
            surprise_pts.append([t, round(surprise, 3)])
        last_truth = truth

    return ui.echart({
        "backgroundColor": _CHART_BG,
        "tooltip": {**_CHART_TOOLTIP, "trigger": "axis"},
        "legend": {
            "data": ["stakes", "aud. prior", "−log p (surprise)"],
            "textStyle": {"color": _CHART_TEXT}, "top": 0,
        },
        "grid": {"top": 30, "left": 50, "right": 50, "bottom": 40},
        "xAxis": {
            "type": "value",
            "name": "fabula t",
            "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
            "nameTextStyle": {"color": _CHART_TEXT},
        },
        "yAxis": [
            {
                "type": "value", "min": 0, "max": 1,
                "name": "stakes / prior",
                "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
                "nameTextStyle": {"color": _CHART_TEXT},
            },
            {
                "type": "value", "min": 0,
                "name": "−log p", "position": "right",
                "axisLabel": {"color": _CHART_TEXT, "fontSize": 10},
                "nameTextStyle": {"color": _CHART_TEXT},
                "splitLine": {"show": False},
            },
        ],
        "series": [
            {
                "name": "stakes", "type": "line",
                "data": stake_pts, "smooth": True,
                "lineStyle": {"color": "#f59e0b", "width": 2},
                "itemStyle": {"color": "#f59e0b"},
            },
            {
                "name": "aud. prior", "type": "line",
                "data": prior_pts, "smooth": True,
                "lineStyle": {"color": "#3A7BD5", "width": 2,
                              "type": "dashed"},
                "itemStyle": {"color": "#3A7BD5"},
            },
            {
                "name": "−log p (surprise)", "type": "scatter",
                "yAxisIndex": 1, "data": surprise_pts,
                "symbolSize": 14,
                "itemStyle": {"color": "#dc2626"},
            },
        ],
    }).classes("w-full").style(f"height: {height}")
