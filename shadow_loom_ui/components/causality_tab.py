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
    render_emotional_gauges,
    render_event_timeline,
    render_propagation_waterfall,
)

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

    with ui.column().classes("w-full h-full"):
        with ui.tabs().classes("w-full").props("dense no-caps") as sub_tabs:
            ui.tab("topology", label="Causal Topology", icon="device_hub")
            ui.tab("whatif", label="What-If Workbench", icon="science")
            ui.tab("directive", label="Directive Builder", icon="theater_comedy")
            ui.tab("affective", label="Affective Dashboard", icon="favorite")

        with ui.tab_panels(sub_tabs, value="topology").classes("w-full flex-grow"):
            with ui.tab_panel("topology").classes("q-pa-sm"):
                _build_causal_topology(state)

            with ui.tab_panel("whatif").classes("q-pa-sm"):
                _build_whatif_workbench(state)

            with ui.tab_panel("directive").classes("q-pa-sm"):
                _build_directive_builder(state)

            with ui.tab_panel("affective").classes("q-pa-sm"):
                _build_affective_dashboard(state)


# =====================================================================
# 1. Causal Topology
# =====================================================================

def _build_causal_topology(state: AppState) -> None:
    """Sankey diagram + causal force graph."""

    with ui.column().classes("w-full h-full gap-3"):
        view_toggle = ui.toggle(
            {"sankey": "Sankey Flow", "force": "Force Graph"},
            value="sankey",
        ).props("dense no-caps")

        graph_container = ui.column().classes("w-full flex-grow")

        def _refresh(**kw):
            graph_container.clear()
            ws = state.world_state
            if ws is None:
                with graph_container:
                    ui.label("No world model loaded.").classes("text-body2 text-grey")
                return

            with graph_container:
                if view_toggle.value == "sankey":
                    render_causal_sankey(ws, height="100%")
                else:
                    render_causal_force_graph(ws, height="100%")

        _refresh()
        view_toggle.on("update:model-value", lambda: _refresh())
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)


# =====================================================================
# 2. What-If Workbench (NL-driven with propagation viz)
# =====================================================================

def _build_whatif_workbench(state: AppState) -> None:
    """NL-driven intervention and counterfactual with propagation waterfall."""

    with ui.column().classes("w-full h-full gap-4"):
        # ── Single NL input with type toggle ──────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("science", color="primary")
                ui.label("What-If Explorer").classes("text-subtitle1")

            ui.label(
                "Describe what you want to change (intervention) or explore "
                "(counterfactual) in natural language."
            ).classes("text-caption text-grey q-mb-sm")

            whatif_type = ui.toggle(
                {"intervention": "Intervene (now)", "counterfactual": "What-If (past)"},
                value="intervention",
            ).props("dense no-caps")

            whatif_input = ui.textarea(
                placeholder="e.g., 'Kill Macbeth' or 'What if Romeo never met Juliet?'",
            ).classes("w-full q-mt-sm").props("rows=2 autogrow outlined dense")

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
            ).props("color=primary no-caps")


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
            ui.label("No result.").classes("text-grey")
        return

    with container:
        with ui.card().classes("w-full q-pa-md"):
            with ui.row().classes("items-center gap-2 q-mb-sm"):
                ui.badge(pr.query_type, color="primary").props("dense")
                if pr.world_model:
                    ui.badge(f"v{pr.world_model.version}", color="teal").props("dense")
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
                        render_propagation_waterfall(mutations, blocked, height="250px")

                # Hidden deltas (counterfactual)
                hidden = pr.physics_result.get("hidden_deltas", [])
                if hidden:
                    with ui.expansion("Hidden Deltas (abduction)", icon="visibility_off").props("dense"):
                        for hd in hidden:
                            ui.label(
                                f"{hd.get('entity', '?')}:{hd.get('trait', '?')} "
                                f"factual={hd.get('factual', '?'):.2f} → "
                                f"counterfactual={hd.get('counterfactual', '?'):.2f}"
                            ).classes("text-caption")

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
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("theater_comedy", color="purple")
                ui.label("Emotional Directive Builder").classes("text-subtitle1")

            ui.label(
                "Choose an emotional effect to optimize the next scene for. "
                "Or describe what you want in natural language below."
            ).classes("text-caption text-grey q-mb-sm")

            # Entity picker
            entity_select = ui.select(
                options=[],
                label="Target entities",
                multiple=True,
            ).classes("w-full").props("outlined dense")

            # Emotional target selector
            effect_select = ui.select(
                options={k: v for k, v in _EMOTIONAL_TARGETS},
                value="suspense",
                label="Target emotion",
            ).classes("w-64 q-mt-sm").props("outlined dense")

            # Intensity slider
            with ui.row().classes("w-full items-center gap-2 q-mt-sm"):
                ui.label("Intensity:").classes("text-body2")
                intensity_slider = ui.slider(
                    min=0.1, max=1.0, value=0.7, step=0.1
                ).classes("flex-grow").props("label-always")

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
            ).props("color=purple no-caps")

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
            ui.label("No result.").classes("text-grey")
        return

    with container:
        with ui.card().classes("w-full q-pa-md"):
            with ui.row().classes("items-center gap-2 q-mb-sm"):
                ui.badge("directive", color="purple").props("dense")
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

    with ui.column().classes("w-full h-full gap-4"):
        gauge_container = ui.column().classes("w-full")
        timeline_container = ui.column().classes("w-full")

        def _refresh(**kw):
            gauge_container.clear()
            timeline_container.clear()
            ws = state.world_state
            if ws is None:
                with gauge_container:
                    ui.label("No world model loaded.").classes("text-body2 text-grey")
                return

            scores = _compute_affective_scores(ws)

            with gauge_container:
                if scores:
                    ui.label("Narrative Affect Scores").classes("text-subtitle1 q-mb-sm")
                    render_emotional_gauges(scores, height="200px")
                else:
                    ui.label("No affective scores available.").classes("text-grey")

            with timeline_container:
                ui.label("Event Timeline").classes("text-subtitle1 q-mb-sm")
                render_event_timeline(ws, height="250px")

        _refresh()
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh)


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
