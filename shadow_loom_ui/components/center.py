"""Center panel: NL query input, prose output, graph view, and audit tabs."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from nicegui import ui

from shadow_loom_ui.graph_viz import render_world_graph, render_ego_graph, render_causal_subgraph

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState, NLQueryResult

logger = logging.getLogger(__name__)


def build_center_panel(state: AppState) -> None:
    """Build the center panel with query + output tabs."""

    graph_container = None
    prose_container = None
    audit_container = None
    parse_container = None
    query_input = None
    status_label = None

    async def _submit_query():
        """Handle NL query submission."""
        text = query_input.value.strip()
        if not text:
            ui.notify("Enter a query first", type="warning")
            return
        if state.world_state is None:
            ui.notify("Load a world model first", type="warning")
            return

        status_label.set_text("Parsing and executing...")
        spinner.set_visibility(True)

        try:
            # Run in background thread to not block UI
            result = await asyncio.get_event_loop().run_in_executor(
                None, state.run_nl_query, text,
            )
            _display_result(result)
        except Exception as e:
            logger.exception("Query execution failed")
            ui.notify(f"Error: {e}", type="negative")
        finally:
            spinner.set_visibility(False)
            status_label.set_text("")

    def _display_result(result: NLQueryResult):
        """Update all tabs with query result."""
        # Parse info tab
        if parse_container is not None:
            parse_container.clear()
            with parse_container:
                _render_parse_info(result)

        # Prose tab
        if prose_container is not None:
            prose_container.clear()
            with prose_container:
                _render_prose(result)

        # Graph tab
        if graph_container is not None:
            _refresh_graph()

        # Audit tab
        if audit_container is not None:
            audit_container.clear()
            with audit_container:
                _render_audit(result)

        if result.error:
            ui.notify(result.error, type="negative")
        else:
            ui.notify(result.summary, type="positive")

    def _render_parse_info(result: NLQueryResult):
        """Show parse diagnostics: query type, resolved IDs, fallback."""
        pr = result.parse_result
        if pr is None:
            ui.label("No parse result")
            return

        if pr.parsed:
            with ui.row().classes("items-center gap-2"):
                ui.badge(pr.parsed.query_type, color="primary").classes("text-body1")
                if pr.fallback:
                    ui.badge(f"Fallback: {pr.fallback.strategy}", color="orange")
                if pr.is_valid:
                    ui.icon("check_circle", color="green")
                else:
                    ui.icon("error", color="red")

            ui.label("Reasoning").classes("text-subtitle2 q-mt-sm")
            ui.label(pr.parsed.reasoning).classes("text-body2")

            if pr.parsed.resolved_ids:
                ui.label("Resolved IDs").classes("text-subtitle2 q-mt-sm")
                columns = [
                    {"name": "name", "label": "Name", "field": "name"},
                    {"name": "id", "label": "Graph ID", "field": "id"},
                    {"name": "conf", "label": "Confidence", "field": "conf"},
                ]
                rows = [
                    {"name": r.natural_name, "id": r.resolved_id,
                     "conf": f"{r.confidence:.0%}"}
                    for r in pr.parsed.resolved_ids
                ]
                ui.table(columns=columns, rows=rows).classes("w-full").props("dense")

        if pr.validation_errors:
            ui.label("Validation Issues").classes("text-subtitle2 q-mt-sm")
            for ve in pr.validation_errors:
                color = "red" if ve.severity == "error" else "orange"
                ui.label(f"[{ve.severity}] {ve.field}: {ve.message}").classes(
                    f"text-body2 text-{color}"
                )

        if pr.fallback and pr.fallback.id_remappings:
            ui.label("ID Remappings (fuzzy)").classes("text-subtitle2 q-mt-sm")
            for old, new in pr.fallback.id_remappings.items():
                ui.label(f"  {old} → {new}").classes("text-body2")

        # Show the structured query as JSON
        if pr.query:
            ui.label("Structured Query").classes("text-subtitle2 q-mt-sm")
            ui.code(pr.query.model_dump_json(indent=2), language="json").classes(
                "w-full max-h-48 overflow-auto"
            )

    def _render_prose(result: NLQueryResult):
        """Show generated prose."""
        pr = result.pipeline_result
        if pr is None:
            ui.label("No pipeline result")
            return

        if pr.prose:
            ui.markdown(pr.prose).classes("text-body1 q-pa-md")
        elif pr.query_type in ("interrogate", "general"):
            # Show physics state for non-prose queries
            physics = pr.physics_result
            if physics:
                # For interrogate/general, the answer is in physics_state
                answer = physics.get("answer") or physics.get("physics_state", {})
                if isinstance(answer, str):
                    ui.markdown(answer).classes("text-body1 q-pa-md")
                else:
                    ui.code(
                        __import__("json").dumps(answer, indent=2, default=str),
                        language="json",
                    ).classes("w-full max-h-96 overflow-auto")
        else:
            ui.label("No prose generated").classes("text-body2 text-grey")

    def _render_audit(result: NLQueryResult):
        """Show audit loop results and/or full-story evaluation scorecard."""
        import json as _json

        pr = result.pipeline_result
        if pr is None:
            ui.label("No audit data").classes("text-body2 text-grey")
            return

        # --- Evaluation result (from EvaluationQuery) ---
        if pr.evaluation_result is not None:
            _render_narrative_order_card(pr.evaluation_result.get("narrative_order") if isinstance(pr.evaluation_result, dict) else getattr(pr.evaluation_result, "narrative_order", None))
            return

        if pr.feedback_result is None:
            ui.label("No audit data").classes("text-body2 text-grey")
            return

        fb = pr.feedback_result

        # --- Convergence status ---
        with ui.row().classes("items-center gap-2"):
            icon = "check_circle" if fb.converged else "warning"
            color = "green" if fb.converged else "orange"
            ui.icon(icon, color=color)
            ui.label(
                f"{'Converged' if fb.converged else 'Did not converge'}"
                f" after {fb.iterations} iteration(s)"
            ).classes("text-body1")

        # --- Per-cycle change impact (delta metrics) ---
        if fb.change_impact is not None:
            _render_change_impact(fb.change_impact)

        # --- Audit cycle history ---
        if fb.history:
            ui.label("Audit Cycles").classes("text-subtitle2 q-mt-md")
            for i, cycle in enumerate(fb.history):
                with ui.expansion(f"Cycle {i + 1}", icon="loop").classes("w-full"):
                    ar = cycle.audit_result
                    with ui.row().classes("items-center gap-2"):
                        pass_icon = "check" if ar.passed else "close"
                        pass_color = "green" if ar.passed else "red"
                        ui.icon(pass_icon, color=pass_color)
                        ui.label(
                            f"{'PASSED' if ar.passed else 'FAILED'}"
                            + (f" — {ar.audit_summary}" if ar.audit_summary else "")
                        ).classes("text-body2")

                    if ar.violations:
                        for v in ar.violations:
                            sev_color = "red" if v.severity == "critical" else "orange" if v.severity == "major" else "grey"
                            with ui.row().classes("items-start gap-1 q-ml-md"):
                                ui.badge(v.severity, color=sev_color).classes("text-caption")
                                ui.label(f"{v.violation_type}: {v.description}").classes("text-body2")
                            if v.evidence_quote:
                                ui.label(f'  "{v.evidence_quote}"').classes(
                                    "text-caption text-grey q-ml-xl"
                                )

                    # Per-cycle change impact
                    if cycle.change_impact is not None:
                        with ui.expansion("Change Impact", icon="trending_up").classes("w-full q-mt-xs"):
                            _render_change_impact(cycle.change_impact)

    def _render_change_impact(change_impact):
        """Render per-cycle ChangeImpactMetrics (causal + affective delta)."""
        if change_impact is None:
            return

        # Handle both dict and Pydantic model
        if isinstance(change_impact, dict):
            causal = change_impact.get("causal_feedback", {})
            affective = change_impact.get("affective_feedback", {})
        else:
            causal = change_impact.causal_feedback
            affective = change_impact.affective_feedback
            causal = causal.model_dump() if hasattr(causal, "model_dump") else causal
            affective = affective.model_dump() if hasattr(affective, "model_dump") else affective

        # --- Causal delta ---
        miracles = causal.get("miracle_steps_detected", [])
        fs = causal.get("foreshadowing_payoff_score", 0.0)
        cp = causal.get("cognitive_plausibility_score", 1.0)

        with ui.row().classes("items-center gap-4 q-mt-xs"):
            ui.label(f"Foreshadowing: {fs:.0%}").classes("text-body2")
            ui.linear_progress(value=fs, color="blue").classes("w-24")
            ui.label(f"Plausibility: {cp:.0%}").classes("text-body2")
            ui.linear_progress(
                value=cp,
                color="green" if cp >= 0.7 else "orange" if cp >= 0.4 else "red",
            ).classes("w-24")

        if miracles:
            ui.label(f"Miracle steps: {len(miracles)}").classes("text-body2 text-red")
            for ms in miracles:
                ui.label(f"  • {ms}").classes("text-caption text-red")

        # --- Affective delta ---
        scores = affective.get("emotional_trajectory_scores", {})
        if scores:
            with ui.row().classes("items-center gap-4 q-mt-xs"):
                for effect, score in scores.items():
                    ui.label(f"{effect}: {score:.2f}").classes("text-body2")

        loss = affective.get("affective_loss_mse", 0.0)
        loss_color = "green" if loss <= 0.3 else "orange" if loss <= 0.6 else "red"
        ui.label(f"Affective loss: {loss:.4f}").classes(f"text-caption text-{loss_color}")

    def _render_narrative_order_card(narrative_order):
        """Render a NarrativeOrderObject as a visual scorecard."""
        if narrative_order is None:
            return

        # Handle both dict and Pydantic model
        if isinstance(narrative_order, dict):
            no = narrative_order
            causal = no.get("causal_feedback", {})
            affective = no.get("affective_feedback", {})
            quality = no.get("quality_synthesis", {})
            overall = no.get("overall_pass", False)
        else:
            causal = narrative_order.causal_feedback
            affective = narrative_order.affective_feedback
            quality = narrative_order.quality_synthesis
            overall = narrative_order.overall_pass
            # Convert to dicts for uniform access
            causal = causal.model_dump() if hasattr(causal, "model_dump") else causal
            affective = affective.model_dump() if hasattr(affective, "model_dump") else affective
            quality = quality.model_dump() if hasattr(quality, "model_dump") else quality

        # Overall pass/fail badge
        with ui.row().classes("items-center gap-2 q-mb-sm"):
            if overall:
                ui.badge("PASS", color="green").classes("text-body1")
                ui.icon("verified", color="green")
            else:
                ui.badge("FAIL", color="red").classes("text-body1")
                ui.icon("error_outline", color="red")

        # --- Causal Physics Feedback ---
        with ui.expansion("Causal Physics", icon="engineering").classes("w-full"):
            # Foreshadowing score
            fs = causal.get("foreshadowing_payoff_score", 0.0)
            ui.label(f"Foreshadowing Payoff: {fs:.0%}").classes("text-body2")
            ui.linear_progress(value=fs, color="blue").classes("w-full")

            # Cognitive plausibility
            cp = causal.get("cognitive_plausibility_score", 1.0)
            ui.label(f"Cognitive Plausibility: {cp:.0%}").classes("text-body2 q-mt-sm")
            ui.linear_progress(
                value=cp,
                color="green" if cp >= 0.7 else "orange" if cp >= 0.4 else "red",
            ).classes("w-full")
            details = causal.get("cognitive_plausibility_details", "")
            if details:
                ui.label(details).classes("text-caption text-grey")

            # Miracle steps
            miracles = causal.get("miracle_steps_detected", [])
            if miracles:
                ui.label(f"Miracle Steps Detected: {len(miracles)}").classes(
                    "text-body2 text-red q-mt-sm"
                )
                for ms in miracles:
                    ui.label(f"  • {ms}").classes("text-caption text-red")
            else:
                ui.label("No miracle steps detected").classes(
                    "text-body2 text-green q-mt-sm"
                )

        # --- Affective State Feedback ---
        with ui.expansion("Affective State", icon="psychology").classes("w-full"):
            scores = affective.get("emotional_trajectory_scores", {})
            if scores:
                ui.label("Emotional Trajectory Scores").classes("text-subtitle2")
                for effect, score in scores.items():
                    ui.label(f"{effect}: {score:.2f}").classes("text-body2")
                    ui.linear_progress(
                        value=min(score, 1.0),
                        color="purple",
                    ).classes("w-full")

            kl = affective.get("kl_divergence_prediction_error")
            if kl is not None:
                ui.label(f"KL Divergence (Surprise): {kl:.4f}").classes(
                    "text-body2 q-mt-sm"
                )

            loss = affective.get("affective_loss_mse", 0.0)
            loss_color = "green" if loss <= 0.3 else "orange" if loss <= 0.6 else "red"
            ui.label(f"Affective Loss MSE: {loss:.4f}").classes(
                f"text-body2 text-{loss_color} q-mt-sm"
            )

        # --- Story Quality Synthesis ---
        with ui.expansion("Quality Synthesis", icon="auto_stories").classes("w-full"):
            review = quality.get("coherence_and_consistency_review", "")
            if review:
                ui.label("Coherence & Consistency").classes("text-subtitle2")
                ui.markdown(review).classes("text-body2")

            hacking = quality.get("reward_hacking_diagnostics")
            if hacking:
                ui.label("Reward Hacking Detected").classes(
                    "text-subtitle2 text-orange q-mt-sm"
                )
                ui.markdown(hacking).classes("text-body2")

            directives = quality.get("actionable_rewrite_directives", [])
            if directives:
                ui.label("Rewrite Directives").classes("text-subtitle2 q-mt-sm")
                for i, d in enumerate(directives, 1):
                    ui.label(f"  {i}. {d}").classes("text-body2")

    def _refresh_graph():
        """Re-render the graph visualization."""
        if graph_container is None:
            return
        graph_container.clear()
        with graph_container:
            if state.world_state is None:
                ui.label("No world model loaded").classes("text-grey")
                return
            try:
                html = render_world_graph(state.world_state, height="500px")
                ui.html(html).classes("w-full")
            except Exception as e:
                logger.exception("Graph rendering failed")
                ui.label(f"Graph error: {e}").classes("text-red")

    async def _submit_manual_edit():
        """Handle manual edit submission."""
        text = edit_textarea.value.strip()
        if not text:
            ui.notify("Enter some prose first", type="warning")
            return
        if state.world_state is None:
            ui.notify("Load a world model first", type="warning")
            return

        desc = edit_description.value.strip()
        status_label.set_text("Re-extracting topology from edited prose...")
        spinner.set_visibility(True)

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: state.run_manual_edit(text, description=desc),
            )
            _display_result(result)
            if not result.error:
                edit_textarea.value = ""
                edit_description.value = ""
        except Exception as e:
            logger.exception("Manual edit failed")
            ui.notify(f"Error: {e}", type="negative")
        finally:
            spinner.set_visibility(False)
            status_label.set_text("")

    # ---- Build the UI ----

    edit_textarea = None
    edit_description = None

    with ui.column().classes("w-full h-full"):
        # Query input bar
        with ui.row().classes("w-full items-center q-pa-sm gap-2"):
            query_input = ui.input(
                placeholder="Ask anything about the story... (natural language)",
            ).classes("flex-grow").props('outlined dense')
            query_input.on("keydown.enter", _submit_query)

            ui.button(icon="send", on_click=_submit_query).props("flat dense color=primary")
            spinner = ui.spinner("dots", size="sm")
            spinner.set_visibility(False)

        status_label = ui.label("").classes("text-caption text-grey q-px-sm")

        # Tabs
        with ui.tabs().classes("w-full") as tabs:
            parse_tab = ui.tab("Parse", icon="psychology")
            prose_tab = ui.tab("Prose", icon="article")
            graph_tab = ui.tab("Graph", icon="hub")
            audit_tab = ui.tab("Audit", icon="fact_check")
            edit_tab = ui.tab("Manual Edit", icon="edit_note")

        with ui.tab_panels(tabs, value=parse_tab).classes("w-full flex-grow"):
            with ui.tab_panel(parse_tab):
                parse_container = ui.column().classes("w-full q-pa-sm overflow-auto")

            with ui.tab_panel(prose_tab):
                prose_container = ui.column().classes("w-full q-pa-sm overflow-auto")

            with ui.tab_panel(graph_tab):
                graph_container = ui.column().classes("w-full overflow-auto")

            with ui.tab_panel(audit_tab):
                with ui.column().classes("w-full q-pa-sm"):
                    with ui.row().classes("w-full justify-end q-mb-sm"):
                        async def _run_evaluation():
                            if state.world_state is None:
                                ui.notify("Load a world model first", type="warning")
                                return
                            spinner.set_visibility(True)
                            status_label.set_text("Running full-story evaluation...")
                            try:
                                from shadow_loom.query_models import EvaluationQuery
                                query = EvaluationQuery(include_full_prose=False)
                                eval_result = await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: state.run_structured_query(query),
                                )
                                _display_result(eval_result)
                            except Exception as e:
                                logger.exception("Evaluation failed")
                                ui.notify(f"Error: {e}", type="negative")
                            finally:
                                spinner.set_visibility(False)
                                status_label.set_text("")

                        ui.button(
                            "Evaluate Story",
                            icon="assessment",
                            on_click=_run_evaluation,
                        ).props("color=deep-purple outline")
                    audit_container = ui.column().classes("w-full overflow-auto")

            with ui.tab_panel(edit_tab):
                with ui.column().classes("w-full q-pa-sm"):
                    ui.label(
                        "Write or paste narrative prose directly. It will be "
                        "re-extracted into the world model's topology."
                    ).classes("text-caption text-grey")
                    edit_description = ui.input(
                        placeholder="Description of changes (optional)",
                    ).classes("w-full").props("outlined dense")
                    edit_textarea = ui.textarea(
                        placeholder="Write your narrative prose here...",
                    ).classes("w-full").props("outlined rows=10")
                    with ui.row().classes("w-full justify-end"):
                        ui.button(
                            "Submit Edit",
                            icon="edit_note",
                            on_click=_submit_manual_edit,
                        ).props("color=primary")

    # Wire graph refresh to world state changes
    prev_cb = state.on_world_state_changed

    def _on_ws_changed():
        if prev_cb:
            prev_cb()
        _refresh_graph()

    state.on_world_state_changed = _on_ws_changed
