# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audit tab — NarrativeOrderObject scorecard, evaluation, audit loop replay.

Provides:
- Full-story evaluation with structured scorecard visualization
- Per-query audit history with convergence tracking
- Audit loop iteration replay (prose + violations per cycle)
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
from typing import TYPE_CHECKING

from nicegui import ui

# ── UI tuning constants ─────────────────────────────────────────────
# Centralised so the auditor surfaces feel coherent and tuning is a
# one-line change. Previously each truncation site picked its own
# limit (500 / 300 / 20 / 5 / 5 / 3) which read as ad-hoc.
_PROSE_PREVIEW_CHARS = 400
_LIST_PREVIEW_ITEMS = 5
_DIAGNOSTIC_PREVIEW_ITEMS = 20
_CYCLE_VIOLATIONS_PREVIEW = 3

from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent
from shadow_loom_ui.task_helpers import run_query_as_task
from shadow_loom_ui.reasoning_helpers import (
    directive_for_violation,
    violation_explanation,
)
from shadow_loom_ui.reasoning_viz import render_convergence_trajectory
from shadow_loom_ui.viz import (
    render_audit_passrate_pictorial,
    render_emotional_gauges,
    with_expand,
)
from shadow_loom_ui.viz_helpers import audit_passrate_data

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_audit_tab(state: AppState) -> None:
    """Build the Audit tab layout."""

    with ui.column().classes("w-full h-full p-6 gap-4 bg-slate-50"):        # ── Header with help popover ────────────────────────────────────
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("fact_check", color="primary")
            ui.label("Audit").classes(
                "text-sm font-semibold text-slate-700"
            )
            ui.space()
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Audit — quality scorecard & activity log",
                body_md=(
                    "Two things in one tab:\n\n"
                    "### Full-story evaluation (top)\n"
                    "Run an LLM-backed quality audit over the entire"
                    " prose corpus on the active version's lineage."
                    " Click a chip to launch a focused evaluation:\n"
                    "- **Run full evaluation** — NarrativeOrder"
                    " composite scorecard (foreshadowing pay-off,"
                    " cognitive plausibility, affective fit).\n"
                    "- **Check for miracle steps** — detects"
                    " unexplained jumps in entity state (a character"
                    " teleporting, a death undone) the engine could"
                    " not justify from prior events.\n"
                    "- **Evaluate character consistency** — cognitive"
                    " plausibility per entity: do their choices match"
                    " their established traits and beliefs?\n\n"
                    "Scorecards include:\n"
                    "- **Foreshadowing pay-off score** — set-ups that"
                    " landed vs dropped threads.\n"
                    "- **Cognitive plausibility score** — weighted"
                    " average over per-entity belief consistency.\n"
                    "- **Affective loss MSE** — distance between the"
                    " requested emotional trajectory and what the"
                    " prose actually achieved (lower is better).\n"
                    "- **Miracle steps detected** — list of"
                    " unexplained state changes with offending event"
                    " ids.\n"
                    "- **Rewrite directives** — actionable suggestions"
                    " the auditor produced; clicking one populates the"
                    " command bar with a Direct query.\n\n"
                    "### Activity log (below)\n"
                    "Chronological feed of every action on this"
                    " project: ingestion, queries, manual edits,"
                    " version saves, deletes, branch promotions. Each"
                    " entry shows the user, timestamp, raw NL"
                    " question, parsed query, and a diff summary of"
                    " the world-state changeset.\n\n"
                    "### Tips\n"
                    "- Evaluation does **not** create a new version —"
                    " it only scores existing prose.\n"
                    "- Click any version row to load that version into"
                    " every panel.\n"
                    "- Use the *engine threshold failures* chips on"
                    " each entry to jump back to the offending audit"
                    " iteration."
                ),
                tooltip="What is this tab?",
            )
        # ── NL prompt for evaluation ──────────────────────────────
        with ui.row().classes("w-full gap-2 flex-wrap"):
            ui.chip(
                "Run full evaluation",
                icon="fact_check",
                on_click=lambda: state.emit(
                    StateEvent.QUERY_STARTED,
                    suggestion="Evaluate the story quality comprehensively",
                    query_type="evaluate",
                ),
            ).props("dense outline clickable color=primary")
            ui.chip(
                "Check for miracle steps",
                icon="warning",
                on_click=lambda: state.emit(
                    StateEvent.QUERY_STARTED,
                    suggestion="Are there any miracle steps or impossible state changes in the story?",
                    query_type="evaluate",
                ),
            ).props("dense outline clickable")
            ui.chip(
                "Evaluate character consistency",
                icon="psychology",
                on_click=lambda: state.emit(
                    StateEvent.QUERY_STARTED,
                    suggestion="Evaluate character consistency and cognitive plausibility",
                    query_type="evaluate",
                ),
            ).props("dense outline clickable")

        # ── Full Story Evaluation ─────────────────────────────────
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("fact_check", size="md", color="primary")
                ui.label("Full Story Evaluation").classes(
                    "text-lg font-semibold text-slate-800"
                )

            ui.label(
                "Run a comprehensive narrative quality evaluation across the entire world model."
            ).classes("text-sm text-slate-500")

            eval_container = ui.column().classes("w-full q-mt-sm")

            async def _run_evaluation():
                if state.world_state is None:
                    ui.notify("No world model loaded", type="warning")
                    return

                # Inline spinner inside the result container so progress
                # appears where the user is looking, not in a stray
                # label far above.
                eval_container.clear()
                with eval_container:
                    with ui.row().classes("items-center gap-2 text-slate-600"):
                        ui.spinner(size="sm")
                        ui.label("Running evaluation…").classes("text-sm")
                try:
                    from shadow_loom.query_models import EvaluationQuery
                    query = EvaluationQuery()
                    result, _task = await run_query_as_task(
                        state,
                        label="Full-story evaluation",
                        kind="evaluate",
                        runner=lambda: asyncio.to_thread(
                            state.run_structured_query, query,
                        ),
                        summary_fn=lambda r: (r.summary if r else "") or "Done",
                    )
                    _render_evaluation_result(eval_container, result)
                except Exception as e:
                    logger.exception("Evaluation failed")
                    eval_container.clear()
                    with eval_container:
                        ui.label(f"Error: {e}").classes(
                            "text-sm text-negative"
                        )

            ui.button(
                "Run Evaluation", icon="play_arrow", on_click=_run_evaluation
            ).props("unelevated color=primary no-caps").classes("rounded-lg shadow-sm")

        # ── Per-Query Audit History ────────────────────────
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            ui.label("Query Audit History").classes(
                "text-lg font-semibold text-slate-800"
            )
            history_container = ui.column().classes("w-full")

            def _refresh_history(**kw):
                history_container.clear()
                if not state.query_history:
                    with history_container:
                        ui.label("No queries yet. Use the command bar to ask questions.").classes(
                            "text-sm text-slate-400 italic"
                        )
                    return

                with history_container:
                    for i, result in enumerate(reversed(state.query_history)):
                        _render_query_audit_entry(i, result, state)

            _refresh_history()
            state.on(StateEvent.PIPELINE_RESULT, _refresh_history)
            state.on(StateEvent.PROJECT_LOADED, _refresh_history)
            # World-state mutations (manual edit, rollback, branch
            # switch) can invalidate the displayed audit entries —
            # e.g. version numbers referenced in headers / diffs no
            # longer match the live world. Re-render so the audit
            # panel stays in lockstep with the rest of the UI.
            state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_history)
            state.on(StateEvent.VERSION_CHANGED, _refresh_history)


# =====================================================================
# Evaluation result renderer (NarrativeOrderObject scorecard)
# =====================================================================

def _render_evaluation_result(container, result: NLQueryResult) -> None:
    """Render the full-story evaluation with structured scorecard."""
    container.clear()

    if result.error:
        with container:
            ui.label(f"Error: {result.error}").classes("text-negative")
        return

    pr = result.pipeline_result
    if pr is None:
        with container:
            ui.label("No evaluation result.").classes(
                "text-sm text-slate-400 italic"
            )
        return

    with container:
        # Convergence summary first — sets context for the scorecard
        # below. Iteration count is shown even when ``converged`` is
        # None (EvaluationQuery skips the rewrite loop, but the
        # auditor may still have iterated internally).
        iterations = getattr(pr, "audit_iterations", 0) or 0
        if pr.converged is not None or iterations:
            with ui.row().classes("items-center gap-2"):
                if pr.converged is not None:
                    status = "Converged" if pr.converged else "Did not converge"
                    color = "positive" if pr.converged else "warning"
                    icon = "check_circle" if pr.converged else "sync_problem"
                    ui.icon(icon, color=color)
                    ui.badge(status, color=color).props("dense")
                ui.label(f"Audit iterations: {iterations}").classes(
                    "text-xs text-slate-500"
                )

        # Evaluation scorecard (NarrativeOrderObject)
        eval_result = getattr(pr, "evaluation_result", None)

        if eval_result:
            _render_scorecard(container, eval_result)
        elif pr.prose:
            with ui.card().classes(
                "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
            ):
                ui.label("Evaluation Report").classes(
                    "text-sm font-semibold text-slate-700"
                )
                ui.markdown(pr.prose)


def _render_scorecard(container, eval_result) -> None:
    """Render NarrativeOrderObject scorecard with gauges and metrics."""
    with container:
        # Extract scores
        narrative_order = getattr(eval_result, "narrative_order", None)
        if narrative_order is None:
            ui.label("No scorecard data.").classes(
                "text-sm text-slate-400 italic"
            )
            return

        # Note: per UX decision we deliberately do NOT render an
        # overall PASS/FAIL badge here — a single binary verdict on a
        # multi-dimensional score is misleading. Users get the
        # per-section signals below (causal, affective, quality
        # synthesis) plus the convergence row above.

        # Causal feedback
        causal = getattr(narrative_order, "causal_feedback", None)
        if causal:
            _render_causal_section(causal)

        # Affective feedback
        affective = getattr(narrative_order, "affective_feedback", None)
        if affective:
            _render_affective_section(affective)

        # Quality synthesis
        quality = getattr(narrative_order, "quality_synthesis", None)
        if quality:
            _render_quality_synthesis(quality)


def _render_causal_section(causal) -> None:
    """Render the Causal Metrics card. Skips entirely if no signals."""
    scores: dict[str, float] = {}
    if hasattr(causal, "foreshadowing_payoff_score") \
            and causal.foreshadowing_payoff_score is not None:
        scores["Foreshadowing"] = causal.foreshadowing_payoff_score
    if hasattr(causal, "cognitive_plausibility_score") \
            and causal.cognitive_plausibility_score is not None:
        scores["Plausibility"] = causal.cognitive_plausibility_score

    miracles = getattr(causal, "miracle_steps_detected", []) or []
    details = getattr(causal, "cognitive_plausibility_details", "")
    diagnostic_fields = [
        ("cyclic_propagation_clusters",
         "Cyclic propagation clusters", "loop"),
        ("noisy_or_absorbed_propagations",
         "Noisy-OR absorbed propagations", "filter_alt"),
        ("rule3_pruned_interventions",
         "Rule-3 pruned interventions", "block"),
        ("rule2_redundant_evidence",
         "Rule-2 redundant evidence", "science"),
    ]
    diagnostic_items = [
        (label, icon, getattr(causal, attr, []) or [])
        for attr, label, icon in diagnostic_fields
    ]
    has_diagnostics = any(items for _, _, items in diagnostic_items)

    if not (scores or miracles or details or has_diagnostics):
        return

    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        ui.label("Causal Metrics").classes(
            "text-sm font-semibold text-slate-700 mb-2"
        )
        if scores:
            with_expand(
                lambda h, s=scores: render_emotional_gauges(s, height=h),
                title="Causal metrics",
                height="150px",
            )

        # Miracle steps — use icons (not emoji) to match the rest of
        # the UI's iconographic vocabulary.
        if miracles:
            with ui.row().classes("items-center gap-1 mt-2"):
                ui.icon("warning", color="negative")
                ui.label(
                    f"Miracle steps detected: {len(miracles)}"
                ).classes("text-negative text-sm")
            for m in miracles[:_LIST_PREVIEW_ITEMS]:
                ui.label(f"• {m}").classes("text-xs text-negative ml-4")
            if len(miracles) > _LIST_PREVIEW_ITEMS:
                ui.label(
                    f"… and {len(miracles) - _LIST_PREVIEW_ITEMS} more"
                ).classes("text-xs text-slate-400 italic ml-4")
        else:
            with ui.row().classes("items-center gap-1 mt-2"):
                ui.icon("check_circle", color="positive")
                ui.label("No miracle steps").classes(
                    "text-positive text-sm"
                )

        # Plausibility narrative
        if details:
            with ui.expansion(
                "Plausibility details", icon="psychology",
            ).props("dense").classes("mt-2"):
                ui.markdown(details)

        # ctf-calculus diagnostics — grouped under one parent
        # expansion so the four sub-sections don't stack as visual
        # noise when the user is unlikely to open them.
        if has_diagnostics:
            total = sum(len(items) for _, _, items in diagnostic_items)
            with ui.expansion(
                f"ctf-calculus diagnostics ({total})",
                icon="biotech",
            ).props("dense").classes("mt-2"):
                for label, icon, items in diagnostic_items:
                    if not items:
                        continue
                    with ui.expansion(
                        f"{label} ({len(items)})", icon=icon,
                    ).props("dense").classes("mt-1"):
                        for item in items[:_DIAGNOSTIC_PREVIEW_ITEMS]:
                            ui.label(f"• {item}").classes(
                                "text-xs text-slate-600"
                            )
                        if len(items) > _DIAGNOSTIC_PREVIEW_ITEMS:
                            ui.label(
                                f"… and {len(items) - _DIAGNOSTIC_PREVIEW_ITEMS} more"
                            ).classes("text-xs text-slate-400 italic")


def _render_affective_section(affective) -> None:
    """Render the Affective Metrics card. Skips entirely if no signals."""
    trajectory = getattr(
        affective, "emotional_trajectory_scores", None,
    ) or {}
    loss = getattr(affective, "affective_loss_mse", None)
    kl = getattr(affective, "kl_divergence_prediction_error", None)

    if not (trajectory or loss is not None or kl is not None):
        return

    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        ui.label("Affective Metrics").classes(
            "text-sm font-semibold text-slate-700 mb-2"
        )
        # Emotional trajectory scores are 0–1 "higher is better"
        # intensities — render as gauges.
        if trajectory:
            with_expand(
                lambda h, s=dict(trajectory): render_emotional_gauges(
                    s, height=h
                ),
                title="Emotional trajectory (achieved intensity)",
                height="150px",
            )
        else:
            ui.label("No targeted emotions to score.").classes(
                "text-xs text-slate-400 italic"
            )

        # Affective loss MSE is unbounded and "lower is better" —
        # must NOT be rendered on a 0–1 gauge.
        if loss is not None:
            if loss <= 0.05:
                loss_color, loss_word = "positive", "excellent fit"
            elif loss <= 0.15:
                loss_color, loss_word = "primary", "good fit"
            elif loss <= 0.30:
                loss_color, loss_word = "warning", "weak fit"
            else:
                loss_color, loss_word = "negative", "poor fit"
            with ui.row().classes("items-center gap-2 mt-2"):
                ui.label("Affective loss (MSE, lower=better):").classes(
                    "text-xs text-slate-600"
                )
                ui.badge(
                    f"{loss:.3f} — {loss_word}",
                    color=loss_color,
                ).props("dense")

        if kl is not None:
            ui.label(
                f"KL Divergence (surprise): {kl:.3f}"
            ).classes("text-xs text-slate-500 mt-2")


def _render_quality_synthesis(quality) -> None:
    """Render the Quality Synthesis card. Directives first (most
    actionable), opened by default; reviews and diagnostics follow."""
    directives = getattr(quality, "actionable_rewrite_directives", None)
    coherence = getattr(quality, "coherence_and_consistency_review", None)
    reward_hacking = getattr(quality, "reward_hacking_diagnostics", None)

    if not (directives or coherence or reward_hacking):
        return

    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        ui.label("Quality Synthesis").classes(
            "text-sm font-semibold text-slate-700 mb-2"
        )
        # Directives are the only actionable section — surface first
        # and open by default.
        if directives:
            with ui.expansion(
                "Rewrite Directives", icon="edit_note", value=True,
            ).props("dense"):
                if isinstance(directives, list):
                    ui.markdown("\n".join(f"- {d}" for d in directives))
                else:
                    ui.markdown(str(directives))
        if coherence:
            with ui.expansion(
                "Coherence Review", icon="check_circle",
            ).props("dense"):
                ui.markdown(coherence)
        if reward_hacking:
            with ui.expansion(
                "Reward-Hacking Diagnostics", icon="warning",
            ).props("dense"):
                ui.markdown(reward_hacking)


# =====================================================================
# Per-query audit entry with loop replay
# =====================================================================

def _render_query_audit_entry(index: int, result: NLQueryResult, state: AppState) -> None:
    """Render a single query's audit information with iteration replay."""
    pr = result.pipeline_result
    query_type = pr.query_type if pr else "unknown"
    converged = pr.converged if pr else None

    icon = "check_circle" if converged else "warning" if converged is False else "help"
    with ui.expansion(
        f"#{index + 1} — {query_type}",
        icon=icon,
    ).classes("w-full").props("dense"):
        if result.summary:
            # Summary is now multi-line lay-user text (humanize_pipeline_result).
            ui.markdown(result.summary).classes("text-sm text-slate-600")

        if result.error:
            ui.label(f"Error: {result.error}").classes("text-xs text-negative")

        if pr is not None:
            # Convergence
            if pr.converged is not None:
                status = "Converged" if pr.converged else "Did not converge"
                with ui.row().classes("items-center gap-2"):
                    color = "positive" if pr.converged else "warning"
                    badge_icon = (
                        "check_circle" if pr.converged else "sync_problem"
                    )
                    ui.icon(badge_icon, color=color)
                    ui.badge(status, color=color).props("dense")
                    ui.label(f"{pr.audit_iterations} iterations").classes("text-xs text-slate-500")

            # Engine threshold gate + achieved-vs-target affective intensity.
            # Sourced from the deterministic CausalPhysicsFeedback /
            # AffectiveStateFeedback the auditor attaches to every cycle
            # (independent of the LLM verdict).
            feedback = getattr(pr, "feedback_result", None)
            if feedback is not None:
                if feedback.engine_thresholds_passed is not None:
                    with ui.row().classes("items-center gap-2 mt-1"):
                        thresh_ok = feedback.engine_thresholds_passed
                        gate_color = "positive" if thresh_ok else "negative"
                        gate_icon = "verified" if thresh_ok else "gpp_bad"
                        ui.icon(gate_icon, color=gate_color)
                        ui.badge(
                            "Quality thresholds: passed" if thresh_ok
                            else "Quality thresholds: failed",
                            color=gate_color,
                        ).props("dense")
                    if (not feedback.engine_thresholds_passed
                            and feedback.engine_threshold_failures):
                        with ui.column().classes("gap-0 mt-1 ml-2"):
                            for f in feedback.engine_threshold_failures[:_LIST_PREVIEW_ITEMS]:
                                ui.label(f"• {f}").classes(
                                    "text-xs text-negative"
                                )
                ci = getattr(feedback, "change_impact", None)
                parsed = result.parse_result.parsed if result.parse_result else None
                req_effect = getattr(parsed, "target_effect", None) if parsed else None
                req_intensity = getattr(parsed, "intensity", None) if parsed else None
                if ci is not None and ci.affective_feedback is not None:
                    af = ci.affective_feedback
                    scores = af.emotional_trajectory_scores or {}
                    if req_effect and req_effect in scores:
                        achieved = scores[req_effect]
                        with ui.row().classes("items-center gap-2 mt-1"):
                            ui.label("Achieved intensity:").classes(
                                "text-xs text-slate-500"
                            )
                            if req_intensity is not None:
                                gap = achieved - req_intensity
                                gap_color = (
                                    "positive" if abs(gap) <= 0.15
                                    else "warning"
                                )
                                ui.badge(
                                    f"{req_effect}: {achieved:.2f} "
                                    f"(asked {req_intensity:.2f}, gap {gap:+.2f})",
                                    color=gap_color,
                                ).props("dense")
                            else:
                                ui.badge(
                                    f"{req_effect}: {achieved:.2f}",
                                    color="primary",
                                ).props("dense")
                    if af.affective_loss_mse is not None:
                        ui.label(
                            f"Affective loss (distance from target): "
                            f"{af.affective_loss_mse:.3f}"
                        ).classes("text-xs text-slate-500")

            # Prose excerpt
            if pr.prose:
                with ui.expansion("Prose", icon="article").props("dense"):
                    ui.markdown(textwrap.shorten(
                        pr.prose,
                        width=_PROSE_PREVIEW_CHARS,
                        placeholder="…",
                    ))

            # Evaluation scorecard — when this query was an
            # ``evaluate`` (e.g. triggered from the chat command bar
            # or an audit-tab chip), surface the same scorecard the
            # dedicated "Run Evaluation" button shows so users don't
            # have to re-run the audit just to see results.
            if getattr(pr, "evaluation_result", None) is not None:
                with ui.expansion(
                    "Evaluation Scorecard", icon="fact_check",
                ).props("dense"):
                    scorecard_container = ui.column().classes("w-full gap-2")
                    _render_scorecard(scorecard_container, pr.evaluation_result)

            # Audit loop replay (if feedback_result has history)
            feedback = getattr(pr, "feedback_result", None)
            if feedback and hasattr(feedback, "history") and feedback.history:
                # Collect all violations from audit cycles
                all_violations = []
                for cycle in feedback.history:
                    audit = getattr(cycle, "audit_result", None)
                    if audit:
                        all_violations.extend(getattr(audit, "violations", []))

                # Single-iteration loops don't benefit from a wrapper
                # expansion — render the body inline.
                n_iters = len(feedback.history)
                loop_container = (
                    ui.expansion(
                        f"Audit Loop ({n_iters} iterations)", icon="replay",
                    ).props("dense")
                    if n_iters > 1
                    else ui.column().classes("w-full")
                )
                with loop_container:
                    # Build per-iteration audit history for the
                    # pass-rate pictorial chart + table.
                    history_rows: list[dict] = []
                    for cycle in feedback.history:
                        audit = getattr(cycle, "audit_result", None)
                        if not audit:
                            continue
                        viols = getattr(audit, "violations", []) or []
                        total = len(viols)
                        # ``passed`` flag on the audit means the *iteration*
                        # passed; for per-violation pass we count any with
                        # severity != critical/major as "ok".
                        passed = sum(
                            1 for v in viols
                            if getattr(v, "severity", "") not in ("critical", "major")
                        )
                        history_rows.append({
                            "label": f"Iteration {getattr(cycle, 'iteration', '?')}",
                            "passed": passed,
                            "total": total,
                            "converged": bool(getattr(audit, "passed", False)),
                            "issues": [
                                {"ok": getattr(v, "severity", "") not in ("critical", "major")}
                                for v in viols
                            ],
                        })

                    if history_rows:
                        with_expand(
                            lambda h, hr=history_rows: (
                                render_audit_passrate_pictorial(
                                    hr, height=h
                                )
                            ),
                            title="Audit pass-rate per iteration",
                            height="220px",
                        )
                        _, _, table_rows = audit_passrate_data(history_rows)
                        with ui.expansion(
                            "Pass-rate table", icon="table_view",
                        ).props("dense"):
                            ui.table(
                                columns=[
                                    {"name": "iteration", "label": "Iteration", "field": "iteration", "sortable": True},
                                    {"name": "passed", "label": "Passed", "field": "passed", "sortable": True},
                                    {"name": "total", "label": "Total", "field": "total", "sortable": True},
                                    {"name": "pass_rate", "label": "Pass-rate", "field": "pass_rate", "sortable": True},
                                    {"name": "converged", "label": "Converged?", "field": "converged"},
                                ],
                                rows=table_rows,
                                pagination={"rowsPerPage": 10},
                            ).props("dense flat bordered").classes("w-full")

                    for cycle in feedback.history:
                        _render_audit_cycle(cycle)

                    # Convergence trajectory line chart
                    with_expand(
                        lambda h, fb=feedback: render_convergence_trajectory(
                            fb, height=h,
                        ),
                        title="Convergence trajectory",
                        height="240px",
                    )

                # Violations summary from audit cycles
                if all_violations:
                    with ui.expansion(f"Violations ({len(all_violations)})").props("dense"):
                        for v in all_violations:
                            _render_violation(v, state=state)

        # Parse info
        if result.parse_result and result.parse_result.parsed:
            with ui.expansion("Parse Details", icon="code").props("dense"):
                parsed = result.parse_result.parsed
                if hasattr(parsed, "reasoning") and parsed.reasoning:
                    ui.label(f"Reasoning: {parsed.reasoning}").classes(
                        "text-xs text-slate-500"
                    )


def _render_audit_cycle(cycle) -> None:
    """Render a single audit cycle (iteration) in the replay view."""
    iteration = getattr(cycle, "iteration", "?")
    with ui.card().classes(
        "w-full p-3 mb-1 bg-slate-100 border border-slate-200 rounded-lg"
    ):
        ui.label(f"Iteration {iteration}").classes(
            "text-sm font-semibold text-slate-700"
        )

        # Prose excerpt for this iteration
        prose = getattr(cycle, "prose", "")
        if prose:
            ui.markdown(textwrap.shorten(
                prose, width=_PROSE_PREVIEW_CHARS, placeholder="…",
            )).classes("text-xs text-slate-600")

        # Audit result
        audit = getattr(cycle, "audit_result", None)
        if audit:
            passed = getattr(audit, "passed", None)
            if passed is not None:
                color = "positive" if passed else "warning"
                icon = "check_circle" if passed else "error_outline"
                with ui.row().classes("items-center gap-1"):
                    ui.icon(icon, color=color)
                    ui.badge(
                        "passed" if passed else "failed", color=color,
                    ).props("dense")

            violations = getattr(audit, "violations", [])
            if violations:
                for v in violations[:_CYCLE_VIOLATIONS_PREVIEW]:
                    _render_violation(v, state=None, compact=True)


def _render_violation(v, *, state: "AppState | None" = None, compact: bool = False) -> None:
    """Render one :class:`AuditViolation` with type, plain-English explanation,
    and (when applicable) a one-click directive chip that pre-fills a
    corrective query in the command bar.

    When ``state`` is provided, the "Fix this" button emits
    ``StateEvent.QUERY_STARTED`` so the chat command bar pre-fills the
    suggested directive. When ``state`` is None (e.g. inside the audit
    cycle compact view) the chip is omitted.
    """
    severity = getattr(v, "severity", "")
    severity_color = {
        "critical": "negative",
        "major": "warning",
        "minor": "info",
    }.get(severity, "grey")
    severity_icon = {
        "critical": "error",
        "major": "warning",
        "minor": "info",
    }.get(severity, "help")
    vtype = getattr(v, "violation_type", "")
    description = getattr(v, "description", str(v))
    feedback = getattr(v, "feedback", "")
    explanation = violation_explanation(vtype) if vtype else ""

    card_classes = (
        "w-full p-2 mb-1 bg-white border border-slate-200 rounded-md"
        if not compact
        else "w-full p-1 mb-1 bg-slate-50 rounded"
    )
    with ui.card().classes(card_classes):
        with ui.row().classes("items-center gap-2 flex-wrap"):
            if severity:
                ui.icon(severity_icon, color=severity_color).classes(
                    "text-sm"
                )
                ui.badge(severity, color=severity_color).props("dense")
            if vtype:
                ui.badge(vtype, color="grey").props("dense outline")
            ui.label(
                textwrap.shorten(description, width=200, placeholder="…")
                if compact else description
            ).classes("text-xs text-slate-700")
        # Plain-English explanation is the highest-value text for
        # users trying to understand a failure — show in both modes.
        if explanation:
            ui.label(explanation).classes(
                "text-[11px] text-slate-500 italic mt-1"
            )
        if not compact:
            template = directive_for_violation(vtype) if vtype else None
            with ui.row().classes("items-center gap-2 mt-1"):
                if template is not None and state is not None:
                    prompt_text, qtype = template

                    def _on_fix(p=prompt_text, q=qtype, s=state):
                        s.emit(
                            StateEvent.QUERY_STARTED,
                            suggestion=p,
                            query_type=q,
                        )
                        ui.notify(
                            "Pre-filled corrective directive in command bar.",
                            type="positive",
                        )

                    ui.button(
                        "Fix this", icon="auto_fix_high", on_click=_on_fix,
                    ).props(
                        "dense outline color=primary size=sm no-caps"
                    ).tooltip(prompt_text)
                if feedback:
                    with ui.expansion(
                        "Auditor's rewrite instruction", icon="edit_note",
                    ).props("dense").classes("w-full"):
                        ui.markdown(feedback)
