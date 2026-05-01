"""Reasoning Trace rail.

Renders the structured output of a rung-2 (intervention) or rung-3
(counterfactual) physics result so the user can see *why* the engine
produced its prose:

    1. Pinned do-set      — what was forcibly intervened on
    2. Evidence (rung-3)  — what present-day facts we conditioned on
    3. Abduced past       — hidden_deltas the engine inferred
    4. Forward cascade    — mutations that propagated downstream
    5. Social cascade     — relationship metric shifts
    6. Blocked            — propagations halted by inertia/affordance

Used both as an inline section inside the Reasoning tab and as a
standalone collapsible card the chat / story tabs can embed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from nicegui import ui

from shadow_loom_ui.reasoning_helpers import (
    extract_reasoning_trace,
    reasoning_trace_summary,
)
from shadow_loom_ui.state import AppState

logger = logging.getLogger(__name__)


def render_reasoning_trace(
    state: AppState,
    physics_result: Optional[Dict[str, Any]] = None,
    *,
    title: str = "Reasoning trace",
    expanded: bool = True,
) -> ui.element:
    """Render the structured rung-2/rung-3 explanation panel.

    If ``physics_result`` is omitted we read ``state.last_result``,
    which is the most recent pipeline result this session.
    """
    if physics_result is None:
        pr = state.last_result
        physics_result = pr.physics_result if pr else None

    trace = extract_reasoning_trace(physics_result, ws=state.world_state)

    container = ui.column().classes("w-full gap-2")
    with container:
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("psychology", color="primary")
            ui.label(title).classes(
                "text-base font-semibold text-slate-800"
            )
            if trace["rung"]:
                ui.badge(f"rung {trace['rung']}", color="primary").props("dense")
            ui.space()
            ui.label(reasoning_trace_summary(trace)).classes(
                "text-xs text-slate-500"
            )

        if not any(trace[k] for k in (
            "do_set", "evidence", "abduction",
            "cascade", "social_cascade", "blocked",
            "rule3_pruned_interventions",
            "rule2_redundant_evidence",
            "cyclic_propagation_clusters",
            "pruned_utterance_event_ids",
            "disabled_channel_ids",
        )) and not trace.get("pruned_beliefs_count"):
            ui.label(
                "No causal-physics trace available. "
                "Run an Intervene or What-If query to populate this panel."
            ).classes("text-sm text-slate-400 italic")
            return container

        # 1. Pinned (do-set)
        if trace["do_set"]:
            with ui.expansion(
                f"Pinned (do-set) — {len(trace['do_set'])}",
                icon="push_pin",
                value=expanded,
            ).props("dense").classes("w-full bg-slate-50 rounded-lg"):
                for item in trace["do_set"]:
                    with ui.row().classes("items-baseline gap-2"):
                        ui.badge(item["label"] or item["node_id"]).props("dense color=primary")
                        if item["value"]:
                            ui.label(f"= {item['value']}").classes("text-xs text-slate-600")
                        ui.label(item["node_id"]).classes(
                            "text-[10px] text-slate-400 font-mono"
                        )

        # 2. Evidence (rung-3 only)
        if trace["evidence"]:
            with ui.expansion(
                f"Evidence we conditioned on — {len(trace['evidence'])}",
                icon="visibility",
                value=expanded,
            ).props("dense").classes("w-full bg-slate-50 rounded-lg"):
                ui.label(
                    "Present-day facts the abduction step took as fixed."
                ).classes("text-xs text-slate-500")
                for item in trace["evidence"]:
                    with ui.row().classes("items-baseline gap-2"):
                        ui.badge(item["label"] or item["node_id"]).props("dense")
                        ui.label(item["node_id"]).classes(
                            "text-[10px] text-slate-400 font-mono"
                        )

        # 3. Abduced past (rung-3 only)
        if trace["abduction"]:
            with ui.expansion(
                f"Abduced past (hidden_deltas) — {len(trace['abduction'])}",
                icon="history_edu",
                value=expanded,
            ).props("dense").classes("w-full bg-amber-50 rounded-lg"):
                ui.label(
                    "Trait values the engine inferred *must* have differed in "
                    "the past for the present-day evidence to make sense."
                ).classes("text-xs text-slate-600")
                for r in trace["abduction"]:
                    sign = "+" if r["delta"] >= 0 else ""
                    color = "positive" if r["delta"] >= 0 else "negative"
                    with ui.row().classes("items-baseline gap-2"):
                        ui.badge(r["label"] or r["node_id"]).props("dense")
                        ui.label(f".{r['trait']}").classes(
                            "text-xs text-slate-700"
                        )
                        ui.badge(f"{sign}{r['delta']:.2f}", color=color).props("dense")

        # 4. Forward cascade
        if trace["cascade"]:
            with ui.expansion(
                f"Forward cascade — {len(trace['cascade'])} mutation(s)",
                icon="trending_flat",
                value=expanded,
            ).props("dense").classes("w-full bg-slate-50 rounded-lg"):
                _render_cascade_table(trace["cascade"])

        # 5. Social cascade
        if trace["social_cascade"]:
            with ui.expansion(
                f"Social cascade — {len(trace['social_cascade'])} shift(s)",
                icon="diversity_3",
                value=expanded,
            ).props("dense").classes("w-full bg-slate-50 rounded-lg"):
                _render_social_cascade_table(trace["social_cascade"])

        # 6. Blocked
        if trace["blocked"]:
            with ui.expansion(
                f"Blocked propagations — {len(trace['blocked'])}",
                icon="block",
                value=expanded,
            ).props("dense").classes("w-full bg-rose-50 rounded-lg"):
                ui.label(
                    "Propagations the engine attempted but absorbed by "
                    "inertia or barred by an affordance gate."
                ).classes("text-xs text-slate-600")
                _render_blocked_table(trace["blocked"])

        # 7. ctf-calculus pre-flight (Correa & Bareinboim 2025)
        if trace["rule3_pruned_interventions"]:
            with ui.expansion(
                f"Rule-3 pruned interventions — "
                f"{len(trace['rule3_pruned_interventions'])}",
                icon="filter_alt_off",
                value=False,
            ).props("dense").classes("w-full bg-violet-50 rounded-lg"):
                ui.label(
                    "These interventions were proven vacuous on the AMWN "
                    "before simulation: the surgery has no directed path "
                    "to any target/evidence variable."
                ).classes("text-xs text-slate-600")
                for item in trace["rule3_pruned_interventions"]:
                    with ui.row().classes("items-baseline gap-2"):
                        ui.badge(item["label"] or item["node_id"]).props("dense color=purple")
                        ui.label(item["path"]).classes(
                            "text-[10px] text-slate-500 font-mono"
                        )

        if trace["rule2_redundant_evidence"]:
            with ui.expansion(
                f"Rule-2 redundant evidence — "
                f"{len(trace['rule2_redundant_evidence'])}",
                icon="visibility_off",
                value=False,
            ).props("dense").classes("w-full bg-violet-50 rounded-lg"):
                ui.label(
                    "These evidence nodes are d-separated from the "
                    "intervened variables on the AMWN; abduction on them "
                    "cannot change the counterfactual distribution."
                ).classes("text-xs text-slate-600")
                for item in trace["rule2_redundant_evidence"]:
                    with ui.row().classes("items-baseline gap-2"):
                        ui.badge(item["label"] or item["node_id"]).props("dense color=purple")
                        ui.label(item["node_id"]).classes(
                            "text-[10px] text-slate-500 font-mono"
                        )

        if trace["cyclic_propagation_clusters"]:
            with ui.expansion(
                f"Cyclic propagation clusters — "
                f"{len(trace['cyclic_propagation_clusters'])}",
                icon="all_inclusive",
                value=False,
            ).props("dense").classes("w-full bg-orange-50 rounded-lg"):
                ui.label(
                    "Propagations refused because the source sits in a "
                    "strongly-connected component of the causal graph. "
                    "These are extraction problems, not narrative miracles."
                ).classes("text-xs text-slate-600")
                for item in trace["cyclic_propagation_clusters"]:
                    with ui.row().classes("items-baseline gap-2"):
                        ui.badge(item["label"] or item["node_id"]).props("dense color=orange")
                        if item.get("trait"):
                            ui.label(f".{item['trait']}").classes(
                                "text-xs text-slate-700"
                            )

        # 8. Counterfactual epistemic side-effects (channels & beliefs)
        epistemic_total = (
            int(trace.get("pruned_beliefs_count") or 0)
            + len(trace.get("pruned_utterance_event_ids") or [])
            + len(trace.get("disabled_channel_ids") or [])
        )
        if epistemic_total:
            with ui.expansion(
                f"Epistemic fallout — {epistemic_total}",
                icon="hub",
                value=expanded,
            ).props("dense").classes("w-full bg-sky-50 rounded-lg"):
                ui.label(
                    "Counterfactual surgery on utterances or channels "
                    "invalidated the beliefs whose provenance pointed at "
                    "those nodes. Downstream reasoning runs on a graph "
                    "with the listed beliefs removed."
                ).classes("text-xs text-slate-600")
                if trace.get("pruned_beliefs_count"):
                    ui.label(
                        f"Beliefs pruned: {trace['pruned_beliefs_count']}"
                    ).classes("text-sm text-slate-700")
                if trace.get("pruned_utterance_event_ids"):
                    ui.label("Utterances neutralised").classes(
                        "text-xs font-semibold text-slate-700 mt-1"
                    )
                    for item in trace["pruned_utterance_event_ids"]:
                        with ui.row().classes("items-baseline gap-2"):
                            ui.badge(item["label"] or item["node_id"]).props(
                                "dense color=blue"
                            )
                            ui.label(item["node_id"]).classes(
                                "text-[10px] text-slate-500 font-mono"
                            )
                if trace.get("disabled_channel_ids"):
                    ui.label("Channels severed").classes(
                        "text-xs font-semibold text-slate-700 mt-1"
                    )
                    for item in trace["disabled_channel_ids"]:
                        with ui.row().classes("items-baseline gap-2"):
                            ui.badge(item["label"] or item["node_id"]).props(
                                "dense color=blue"
                            )
                            ui.label(item["node_id"]).classes(
                                "text-[10px] text-slate-500 font-mono"
                            )

        # 9. Information provenance subgraph (utterance/channel flow
        # rooted at the focal node). Renders as a table for now; the
        # underlying ``information_flow`` dict is also force-graph
        # ready when a richer view is wanted.
        info_flow = trace.get("information_flow")
        if info_flow and info_flow.get("edges"):
            with ui.expansion(
                f"Information provenance \u2014 {len(info_flow['edges'])} edges",
                icon="forum",
                value=expanded,
            ).props("dense").classes("w-full bg-violet-50 rounded-lg"):
                ui.label(
                    f"Who-told-whom around {info_flow.get('focal_label') or info_flow.get('focal_id')}. "
                    "Walks utterance addressees, channel participants, and "
                    "speaker links so the epistemic path utterance \u2192 belief "
                    "is visible alongside the structural cascade."
                ).classes("text-xs text-slate-600")
                for e in info_flow["edges"][:30]:
                    with ui.row().classes("items-baseline gap-2"):
                        ui.badge(e.get("source_label") or e.get("source")).props(
                            "dense color=purple"
                        )
                        ui.label(e.get("kind") or "\u2192").classes(
                            "text-[10px] text-slate-500 font-mono"
                        )
                        ui.badge(e.get("target_label") or e.get("target")).props(
                            "dense color=purple-7"
                        )
                        if e.get("channel_id"):
                            ui.label(f"via {e['channel_id']}").classes(
                                "text-[10px] text-slate-500"
                            )
                        if e.get("truth_value"):
                            ui.label(e["truth_value"]).classes(
                                "text-[10px] text-slate-500 italic"
                            )
                        if e.get("intelligibility") is not None:
                            ui.label(
                                f"intel {float(e['intelligibility']):.2f}"
                            ).classes("text-[10px] text-slate-500 font-mono")

    return container


def _render_cascade_table(rows: list[dict]) -> None:
    """Plain-English row format: target.trait: old → new (impact / inertia)."""
    columns = [
        {"name": "target", "label": "Target", "field": "target", "align": "left"},
        {"name": "trait", "label": "Trait", "field": "trait", "align": "left"},
        {"name": "old", "label": "Old", "field": "old", "align": "right"},
        {"name": "new", "label": "New", "field": "new", "align": "right"},
        {"name": "impact", "label": "Impact", "field": "impact", "align": "right"},
        {"name": "inertia", "label": "Inertia", "field": "inertia", "align": "right"},
        {"name": "delay", "label": "Delay", "field": "delay", "align": "right"},
        {"name": "trigger", "label": "Triggered by", "field": "trigger", "align": "left"},
    ]
    table_rows = [
        {
            "target": r["label"] or r["node_id"],
            "trait": r["trait"],
            "old": f"{r['old']:.2f}",
            "new": f"{r['new']:.2f}",
            "impact": f"{r['impact']:.2f}",
            "inertia": f"{r['inertia']:.2f}",
            "delay": r["delay"],
            "trigger": r["trigger"] or "—",
        }
        for r in rows
    ]
    ui.table(
        columns=columns, rows=table_rows,
        pagination={"rowsPerPage": 12},
    ).props("dense flat bordered").classes("w-full")


def _render_social_cascade_table(rows: list[dict]) -> None:
    columns = [
        {"name": "source", "label": "From", "field": "source", "align": "left"},
        {"name": "target", "label": "To", "field": "target", "align": "left"},
        {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
        {"name": "old", "label": "Old", "field": "old", "align": "right"},
        {"name": "new", "label": "New", "field": "new", "align": "right"},
        {"name": "impact", "label": "Impact", "field": "impact", "align": "right"},
        {"name": "inertia", "label": "Inertia", "field": "inertia", "align": "right"},
        {"name": "trigger", "label": "Triggered by", "field": "trigger", "align": "left"},
    ]
    table_rows = [
        {
            "source": r["source_label"] or r["source"],
            "target": r["target_label"] or r["target"],
            "metric": r["metric"],
            "old": f"{r['old']:.2f}",
            "new": f"{r['new']:.2f}",
            "impact": f"{r['impact']:.2f}",
            "inertia": f"{r['inertia']:.2f}",
            "trigger": r["trigger"] or "—",
        }
        for r in rows
    ]
    ui.table(
        columns=columns, rows=table_rows,
        pagination={"rowsPerPage": 12},
    ).props("dense flat bordered").classes("w-full")


def _render_blocked_table(rows: list[dict]) -> None:
    columns = [
        {"name": "target", "label": "Target", "field": "target", "align": "left"},
        {"name": "trait", "label": "Trait", "field": "trait", "align": "left"},
        {"name": "impact", "label": "Impact", "field": "impact", "align": "right"},
        {"name": "inertia", "label": "Inertia", "field": "inertia", "align": "right"},
        {"name": "reason", "label": "Reason", "field": "reason", "align": "left"},
    ]
    table_rows = [
        {
            "target": r["label"] or r["node_id"],
            "trait": r["trait"],
            "impact": f"{r['impact']:.2f}",
            "inertia": f"{r['inertia']:.2f}",
            "reason": r["reason"],
        }
        for r in rows
    ]
    ui.table(
        columns=columns, rows=table_rows,
        pagination={"rowsPerPage": 12},
    ).props("dense flat bordered").classes("w-full")
