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

# Affective loss above this counts as a complete miss when collapsing
# the loss into the 0-1 "emotional fit" score shown in the hero tile.
# Sourced from ``AuditorConfig.max_affective_loss`` so the tile and
# the pass/fail gate share a single definition of "too far".
try:
    from shadow_loom.auditor import AuditorConfig as _AC
    _AFFECTIVE_LOSS_FAIL: float = float(_AC.model_fields["max_affective_loss"].default)
except Exception:
    _AFFECTIVE_LOSS_FAIL = 0.3

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
    """Build the Audit tab — quality report + activity log.

    Author-first layout:
    1. **Quality Report** (hero) — verdict, score tiles, top fixes.
    2. **What the audit found** — plain-language findings, jargon
       hidden behind ``Advanced`` expansions.
    3. **Activity log** — every query / edit / version on this
       project, in reverse-chronological order.
    """

    with ui.column().classes("w-full h-full p-6 gap-4 bg-slate-50"):
        # ── Header with help popover ────────────────────────────────────
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("fact_check", color="primary")
            ui.label("Audit").classes(
                "text-sm font-semibold text-slate-700"
            )
            ui.space()
            from shadow_loom_ui.components.help_popover import help_popover
            help_popover(
                title="Audit — quality report & activity log",
                body_md=(
                    "Use this tab to ask: *is my story working?*\n\n"
                    "### Quality report (top)\n"
                    "Click **Get a quality report** to score the"
                    " current branch's prose. You'll see:\n"
                    "- a one-line verdict and three traffic-light"
                    " tiles (plausibility, foreshadowing, emotional"
                    " fit),\n"
                    "- a **Top fixes** list — actionable rewrite"
                    " ideas; click any one to pre-fill a Direct"
                    " query in the command bar,\n"
                    "- expandable **findings** with the auditor's"
                    " full notes, and an **Advanced** section with"
                    " the underlying scores and diagnostics.\n\n"
                    "Quality reports never create a new version —"
                    " they only score existing prose. Charts move"
                    " into a *Show evidence* expansion so the report"
                    " reads first.\n\n"
                    "### Activity log (below)\n"
                    "Every action on this project in reverse-chrono"
                    " order: ingestion, queries, manual edits, branch"
                    " promotions. Each entry shows whether the"
                    " auditor accepted the prose and lets you replay"
                    " its iterations."
                ),
                tooltip="What is this tab?",
            )

        # ── Quality Report (hero) ─────────────────────────────────
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("auto_awesome", size="md", color="primary")
                ui.label("Quality Report").classes(
                    "text-lg font-semibold text-slate-800"
                )
            ui.label(
                "How well does the current story hold together? "
                "Pick a focus or run the full report."
            ).classes("text-sm text-slate-500")

            # Result + spinner surface lives here.
            eval_container = ui.column().classes("w-full q-mt-sm")

            def _make_runner(label: str, suggestion: str):
                async def _run():
                    if state.world_state is None:
                        ui.notify("No world model loaded", type="warning")
                        return
                    eval_container.clear()
                    with eval_container:
                        with ui.row().classes(
                            "items-center gap-2 text-slate-600"
                        ):
                            ui.spinner(size="sm")
                            ui.label(f"Running {label.lower()}…").classes(
                                "text-sm"
                            )
                    try:
                        from shadow_loom.query_models import EvaluationQuery
                        # Keep using the structured EvaluationQuery so
                        # we get a populated scorecard back; the
                        # ``suggestion`` text is captured on the task
                        # label for the activity log.
                        query = EvaluationQuery()
                        result, _task = await run_query_as_task(
                            state,
                            label=label,
                            kind="evaluate",
                            runner=lambda: asyncio.to_thread(
                                state.run_structured_query, query,
                            ),
                            summary_fn=lambda r: (
                                r.summary if r else ""
                            ) or "Done",
                        )
                        _render_evaluation_result(eval_container, result, state)
                    except Exception as e:
                        logger.exception("Evaluation failed")
                        eval_container.clear()
                        with eval_container:
                            ui.label(f"Error: {e}").classes(
                                "text-sm text-negative"
                            )
                return _run

            with ui.row().classes("w-full gap-2 flex-wrap q-mt-sm"):
                ui.button(
                    "Get a quality report",
                    icon="play_arrow",
                    on_click=_make_runner(
                        "Full quality report",
                        "Evaluate the story quality comprehensively",
                    ),
                ).props(
                    "unelevated color=primary no-caps"
                ).classes("rounded-lg shadow-sm")
                ui.button(
                    "Check for impossible moments",
                    icon="warning",
                    on_click=_make_runner(
                        "Impossible-moments check",
                        "Are there any miracle steps or impossible "
                        "state changes in the story?",
                    ),
                ).props("flat color=primary no-caps")
                ui.button(
                    "Check character consistency",
                    icon="psychology",
                    on_click=_make_runner(
                        "Character consistency",
                        "Evaluate character consistency and cognitive "
                        "plausibility",
                    ),
                ).props("flat color=primary no-caps")

            # Empty placeholder before any run.
            with eval_container:
                ui.label(
                    "No report yet — pick an action above. The full "
                    "report scores plausibility, foreshadowing, and "
                    "emotional fit."
                ).classes("text-sm text-slate-400 italic")

        # ── Activity log ───────────────────────────────────────────
        with ui.card().classes(
            "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-6"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("history", color="primary")
                ui.label("Activity log").classes(
                    "text-lg font-semibold text-slate-800"
                )
            ui.label(
                "Every query and edit on this project, newest first. "
                "Expand a row to see what the auditor checked."
            ).classes("text-sm text-slate-500 mb-2")
            history_container = ui.column().classes("w-full")

            def _refresh_history(**kw):
                history_container.clear()
                if not state.query_history:
                    with history_container:
                        ui.label(
                            "Nothing yet — use the command bar at the "
                            "bottom to ask a question or write a scene."
                        ).classes("text-sm text-slate-400 italic")
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

def _render_evaluation_result(container, result: NLQueryResult, state: AppState) -> None:
    """Render the quality report — hero verdict, score tiles, top
    fixes, then findings, then evidence/advanced.

    Order matters: lay-author readers should see *what to do* before
    *why*. Charts and engine internals live in collapsed expansions
    so they don't dominate the page.
    """
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

    eval_result = getattr(pr, "evaluation_result", None)

    with container:
        # Pure-prose evaluations (rare — scorecard couldn't be parsed)
        # just dump the auditor's narrative; nothing else useful.
        if eval_result is None:
            if pr.prose:
                with ui.card().classes(
                    "w-full bg-white border border-slate-200 "
                    "rounded-xl shadow-sm p-4"
                ):
                    ui.label("Auditor's notes").classes(
                        "text-sm font-semibold text-slate-700"
                    )
                    ui.markdown(pr.prose)
            return

        narrative_order = getattr(eval_result, "narrative_order", None)
        causal = getattr(narrative_order, "causal_feedback", None) \
            if narrative_order else None
        affective = getattr(narrative_order, "affective_feedback", None) \
            if narrative_order else None
        quality = getattr(narrative_order, "quality_synthesis", None) \
            if narrative_order else None

        # ── Hero verdict + score tiles ─────────────────────────────
        _render_hero_verdict(causal, affective)

        # ── Top fixes (action-first) ──────────────────────────────
        if quality is not None:
            _render_top_fixes(quality, state)

        # ── Findings (plain language) ─────────────────────────────
        if causal is not None:
            _render_causal_text(causal)
        if affective is not None:
            _render_affective_text(affective)
        if quality is not None:
            _render_quality_extras(quality)

        # ── Convergence + evidence (charts + raw scores) ──────────
        _render_evidence_block(pr, causal, affective, narrative_order)


# ── Hero / verdict helpers ──────────────────────────────────────────

def _score_tone(score: float | None) -> tuple[str, str, str]:
    """Map a 0-1 score to (color, plain-word, icon).

    Used for traffic-light score tiles. None → grey/"unknown".
    """
    if score is None:
        return ("grey", "—", "help")
    if score >= 0.75:
        return ("positive", "strong", "check_circle")
    if score >= 0.5:
        return ("primary", "okay", "trending_flat")
    if score >= 0.25:
        return ("warning", "needs work", "warning")
    return ("negative", "weak", "error")


def _loss_tone(loss: float | None) -> tuple[str, str, str]:
    """Map a signed affective loss (lower=better, range [-1, +1]) to a tile tone.

    Negative values indicate a strong match (the structural-effect
    score is subtracted from zero); the bands sit on the positive side
    where the loss is genuinely measuring distance from the target.
    """
    if loss is None:
        return ("grey", "—", "help")
    if loss <= 0.05:
        return ("positive", "strong", "check_circle")
    if loss <= 0.15:
        return ("primary", "okay", "trending_flat")
    if loss <= _AFFECTIVE_LOSS_FAIL:
        return ("warning", "needs work", "warning")
    return ("negative", "weak", "error")


def _render_hero_verdict(causal, affective) -> None:
    """One-line plain-English verdict + three score tiles."""
    plausibility = (
        getattr(causal, "cognitive_plausibility_score", None)
        if causal else None
    )
    foreshadowing = (
        getattr(causal, "foreshadowing_payoff_score", None)
        if causal else None
    )
    affective_loss = (
        getattr(affective, "affective_loss_mse", None)
        if affective else None
    )

    # Convert loss to a 0-1 "fit" score for verdict aggregation.
    # ``_AFFECTIVE_LOSS_FAIL`` matches the auditor's
    # ``max_affective_loss`` threshold so the verdict average and the
    # pass/fail gate agree on what counts as a failure boundary.
    if affective_loss is None:
        emotional_fit_score: float | None = None
    else:
        emotional_fit_score = max(
            0.0,
            1.0 - min(1.0, max(0.0, affective_loss) / _AFFECTIVE_LOSS_FAIL),
        )

    scores = [s for s in (plausibility, foreshadowing, emotional_fit_score)
              if s is not None]
    if scores:
        avg = sum(scores) / len(scores)
        if avg >= 0.75:
            verdict = "Your story is holding together well."
            v_color, v_icon = "positive", "celebration"
        elif avg >= 0.5:
            verdict = "Mostly working — a couple of spots to tighten."
            v_color, v_icon = "primary", "thumb_up"
        elif avg >= 0.25:
            verdict = "Some real trouble spots to address."
            v_color, v_icon = "warning", "build"
        else:
            verdict = "Significant rework recommended."
            v_color, v_icon = "negative", "report_problem"
    else:
        verdict = "Quality report ready."
        v_color, v_icon = "primary", "info"

    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon(v_icon, color=v_color, size="md")
            ui.label(verdict).classes(
                "text-base font-semibold text-slate-800"
            )

        # Three score tiles, side by side. Each shows: title, big
        # plain-English word, color badge.
        with ui.row().classes("w-full gap-3 mt-3 flex-wrap"):
            _render_score_tile(
                "Plausibility",
                "Do events follow from what came before?",
                plausibility,
                kind="score",
            )
            _render_score_tile(
                "Foreshadowing",
                "Do set-ups pay off?",
                foreshadowing,
                kind="score",
            )
            _render_score_tile(
                "Emotional fit",
                "Did the prose hit the requested feeling?",
                affective_loss,
                kind="loss",
            )


def _render_score_tile(title: str, sub: str, value, *, kind: str) -> None:
    """Single tile in the hero score row.

    ``kind`` is ``"score"`` (0-1, higher better) or ``"loss"``
    (0-∞, lower better). Tiles flex-grow so three sit nicely on
    desktop and stack on narrow widths.
    """
    if kind == "loss":
        color, word, icon = _loss_tone(value)
    else:
        color, word, icon = _score_tone(value)
    with ui.card().classes(
        "flex-1 min-w-[180px] bg-slate-50 border border-slate-200 "
        "rounded-lg shadow-none p-3"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon, color=color)
            ui.label(title).classes(
                "text-sm font-semibold text-slate-700"
            )
        ui.label(sub).classes("text-[11px] text-slate-500")
        with ui.row().classes("items-center gap-2 mt-2"):
            ui.badge(word, color=color).props("dense")
            if value is not None:
                if kind == "loss":
                    ui.label(f"loss {value:.2f}").classes(
                        "text-[11px] text-slate-400"
                    )
                else:
                    ui.label(f"{value:.0%}").classes(
                        "text-[11px] text-slate-400"
                    )


def _render_top_fixes(quality, state: AppState) -> None:
    """Action-first card surfacing rewrite directives as primary CTAs.

    Directives are the single most useful auditor output for an
    author — pull them out of the old "Quality Synthesis" expansion
    so they sit right under the verdict.
    """
    directives = getattr(quality, "actionable_rewrite_directives", None)
    if not directives:
        return
    items: list[str]
    if isinstance(directives, list):
        items = [str(d).strip() for d in directives if str(d).strip()]
    else:
        items = [s.strip("-• \t") for s in str(directives).splitlines()
                 if s.strip("-• \t")]
    if not items:
        return

    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("auto_fix_high", color="primary")
            ui.label("Top fixes to try").classes(
                "text-sm font-semibold text-slate-700"
            )
        ui.label(
            "Click a fix to pre-fill it as a Direct query in the "
            "command bar — review, edit, then send."
        ).classes("text-[11px] text-slate-500 mb-2")

        for text in items[:_LIST_PREVIEW_ITEMS]:
            with ui.row().classes(
                "w-full items-start gap-2 p-2 rounded-md "
                "bg-slate-50 border border-slate-100"
            ):
                ui.icon("arrow_right", color="primary").classes("mt-1")
                ui.label(text).classes(
                    "text-sm text-slate-700 flex-1"
                )

                def _on_click(t=text, s=state):
                    s.emit(
                        StateEvent.QUERY_STARTED,
                        suggestion=t,
                        query_type="directive",
                    )
                    ui.notify(
                        "Pre-filled as a Direct query in the command bar.",
                        type="positive",
                    )

                ui.button("Use this fix", icon="auto_fix_high",
                          on_click=_on_click).props(
                    "dense outline color=primary size=sm no-caps"
                )
        if len(items) > _LIST_PREVIEW_ITEMS:
            ui.label(
                f"… plus {len(items) - _LIST_PREVIEW_ITEMS} more in "
                "the full findings below."
            ).classes("text-xs text-slate-400 italic")


def _render_quality_extras(quality) -> None:
    """Coherence review + reward-hacking notes (less actionable than
    directives — kept in expansions)."""
    coherence = getattr(quality, "coherence_and_consistency_review", None)
    reward_hacking = getattr(quality, "reward_hacking_diagnostics", None)
    if not (coherence or reward_hacking):
        return
    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        ui.label("Auditor's broader notes").classes(
            "text-sm font-semibold text-slate-700 mb-2"
        )
        if coherence:
            with ui.expansion(
                "Consistency review", icon="check_circle",
            ).props("dense"):
                ui.markdown(coherence)
        if reward_hacking:
            with ui.expansion(
                "Shortcuts the auditor caught", icon="warning",
            ).props("dense"):
                ui.markdown(reward_hacking)


def _render_evidence_block(pr, causal, affective, narrative_order) -> None:
    """Bottom card: convergence, charts, raw scores. Collapsed by
    default — authors don't need this to act on the report."""
    with ui.expansion(
        "Show evidence (charts & raw scores)",
        icon="bar_chart",
    ).props("dense").classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm"
    ):
        # Convergence row
        iterations = getattr(pr, "audit_iterations", 0) or 0
        if pr.converged is not None or iterations:
            with ui.row().classes("items-center gap-2 p-2"):
                if pr.converged is not None:
                    status = (
                        "Auditor agreed" if pr.converged
                        else "Auditor still flagging issues"
                    )
                    color = "positive" if pr.converged else "warning"
                    icon = (
                        "check_circle" if pr.converged
                        else "sync_problem"
                    )
                    ui.icon(icon, color=color)
                    ui.badge(status, color=color).props("dense")
                ui.label(
                    f"{iterations} audit iteration"
                    f"{'s' if iterations != 1 else ''}"
                ).classes("text-xs text-slate-500")

        # Charts
        if causal is not None and _causal_has_charts(causal):
            _render_causal_charts(causal)
        if affective is not None and _affective_has_charts(affective):
            _render_affective_charts(affective)

        # Raw numeric scores (for power users / debugging)
        with ui.expansion(
            "Raw scores", icon="numbers",
        ).props("dense"):
            rows = []
            if causal is not None:
                p = getattr(causal, "cognitive_plausibility_score", None)
                f = getattr(causal, "foreshadowing_payoff_score", None)
                if p is not None:
                    rows.append(("Cognitive plausibility", f"{p:.3f}"))
                if f is not None:
                    rows.append(("Foreshadowing pay-off", f"{f:.3f}"))
            if affective is not None:
                loss = getattr(affective, "affective_loss_mse", None)
                kl = getattr(
                    affective, "kl_divergence_prediction_error", None,
                )
                if loss is not None:
                    rows.append(("Affective loss (MSE, ↓)", f"{loss:.3f}"))
                if kl is not None:
                    rows.append(("KL divergence (surprise)", f"{kl:.3f}"))
            if rows:
                with ui.column().classes("gap-1 p-2"):
                    for k, v in rows:
                        with ui.row().classes("items-center gap-2"):
                            ui.label(k).classes(
                                "text-xs text-slate-600"
                            )
                            ui.badge(v, color="grey").props(
                                "dense outline"
                            )
            else:
                ui.label("No numeric scores available.").classes(
                    "text-xs text-slate-400 italic p-2"
                )


def _render_scorecard(container, eval_result, state: AppState | None = None) -> None:
    """Compatibility shim — the per-query audit entries call this to
    re-render the scorecard inside an expansion. Delegates to the
    same flow as the hero renderer (verdict tiles + findings) but
    skips the action-first directives card (which would be confusing
    when a per-query audit is expanded inside the activity log)."""
    with container:
        narrative_order = getattr(eval_result, "narrative_order", None)
        if narrative_order is None:
            ui.label("No scorecard data.").classes(
                "text-sm text-slate-400 italic"
            )
            return
        causal = getattr(narrative_order, "causal_feedback", None)
        affective = getattr(narrative_order, "affective_feedback", None)
        quality = getattr(narrative_order, "quality_synthesis", None)

        _render_hero_verdict(causal, affective)
        if causal is not None:
            _render_causal_text(causal)
        if affective is not None:
            _render_affective_text(affective)
        if quality is not None:
            # Inside an activity-log expansion, directives + extras
            # collapse together to save vertical space.
            if state is not None:
                _render_top_fixes(quality, state)
            _render_quality_extras(quality)


def _causal_has_charts(causal) -> bool:
    """True if the causal section has any chartable signals."""
    has_score = (
        getattr(causal, "foreshadowing_payoff_score", None) is not None
        or getattr(causal, "cognitive_plausibility_score", None) is not None
    )
    return bool(has_score)


def _render_causal_text(causal) -> None:
    """Render the textual half of the Causal Metrics card (counts,
    miracle steps, plausibility narrative, ctf-calculus diagnostics).
    The supporting gauges render separately under the charts block."""
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

    if not (miracles or details or has_diagnostics):
        return

    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        ui.label("Causal findings").classes(
            "text-sm font-semibold text-slate-700 mb-2"
        )
        # Miracle steps — use icons (not emoji) to match the rest of
        # the UI's iconographic vocabulary.
        if miracles:
            with ui.row().classes("items-center gap-1"):
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
            with ui.row().classes("items-center gap-1"):
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


def _render_causal_charts(causal) -> None:
    """Render the gauge half of the Causal Metrics card."""
    scores: dict[str, float] = {}
    if hasattr(causal, "foreshadowing_payoff_score") \
            and causal.foreshadowing_payoff_score is not None:
        scores["Foreshadowing"] = causal.foreshadowing_payoff_score
    if hasattr(causal, "cognitive_plausibility_score") \
            and causal.cognitive_plausibility_score is not None:
        scores["Plausibility"] = causal.cognitive_plausibility_score
    if not scores:
        return
    ui.label("Causal metrics").classes(
        "text-xs font-semibold text-slate-600 mt-2"
    )
    with_expand(
        lambda h, s=scores: render_emotional_gauges(s, height=h),
        title="Causal metrics",
        height="150px",
    )


def _affective_has_charts(affective) -> bool:
    """True if the affective section has any chartable trajectory."""
    trajectory = getattr(
        affective, "emotional_trajectory_scores", None,
    ) or {}
    return bool(trajectory)


def _render_affective_text(affective) -> None:
    """Render the textual half of the Affective Metrics card (loss
    badge + KL surprise). The trajectory gauges live under charts."""
    loss = getattr(affective, "affective_loss_mse", None)
    kl = getattr(affective, "kl_divergence_prediction_error", None)

    if loss is None and kl is None:
        return

    with ui.card().classes(
        "w-full bg-white border border-slate-200 rounded-xl shadow-sm p-4"
    ):
        ui.label("Affective findings").classes(
            "text-sm font-semibold text-slate-700 mb-2"
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
            with ui.row().classes("items-center gap-2"):
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
            ).classes("text-xs text-slate-500 mt-1")


def _render_affective_charts(affective) -> None:
    """Render the trajectory gauge half of the Affective card."""
    trajectory = getattr(
        affective, "emotional_trajectory_scores", None,
    ) or {}
    if not trajectory:
        return
    ui.label("Emotional trajectory").classes(
        "text-xs font-semibold text-slate-600 mt-2"
    )
    with_expand(
        lambda h, s=dict(trajectory): render_emotional_gauges(
            s, height=h
        ),
        title="Emotional trajectory (achieved intensity)",
        height="150px",
    )


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

            # Surface correction_error from the feedback loop. This is
            # populated when the loop bypass-passed (failed-open
            # auditor) or aborted on a generation/refinement LLM
            # failure. Without rendering it, those paths display a
            # green "Converged" badge with no explanation.
            fb_err = getattr(getattr(pr, "feedback_result", None), "correction_error", None)
            if fb_err:
                with ui.row().classes("items-center gap-2"):
                    ui.icon("warning", color="warning")
                    ui.label(f"Auditor diagnostic: {fb_err}").classes(
                        "text-xs text-warning"
                    )
            if getattr(pr, "reextraction_failed", False):
                rx_err = (
                    getattr(pr, "reextraction_error", None)
                    or "re-extraction skipped"
                )
                with ui.row().classes("items-center gap-2"):
                    ui.icon("warning", color="warning")
                    ui.label(
                        f"World model not updated: {rx_err}"
                    ).classes("text-xs text-warning")

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
                    _render_scorecard(scorecard_container, pr.evaluation_result, state)

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
                    # Build per-iteration audit history rows up-front;
                    # they back both the text summary and the
                    # pass-rate pictorial below.
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
                            "label": f"Iteration {int(getattr(cycle, 'iteration', 0)) + 1}",
                            "passed": passed,
                            "total": total,
                            "converged": bool(getattr(audit, "passed", False)),
                            "issues": [
                                {"ok": getattr(v, "severity", "") not in ("critical", "major")}
                                for v in viols
                            ],
                        })

                    # ── Text first: per-iteration cycle prose +
                    #    pass-rate table + violations summary ──────
                    for cycle in feedback.history:
                        _render_audit_cycle(cycle)

                    if history_rows:
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

                    if all_violations:
                        with ui.expansion(f"Violations ({len(all_violations)})").props("dense"):
                            for v in all_violations:
                                _render_violation(v, state=state)

                    # ── Supporting charts (visual evidence under
                    #    the textual findings above) ───────────────
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

                    # Convergence trajectory line chart
                    with_expand(
                        lambda h, fb=feedback: render_convergence_trajectory(
                            fb, height=h,
                        ),
                        title="Convergence trajectory",
                        height="240px",
                    )

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
    # ``cycle.iteration`` is 0-based in the data model; humans count
    # from 1 in the UI.
    raw_iter = getattr(cycle, "iteration", None)
    iteration = (int(raw_iter) + 1) if isinstance(raw_iter, int) else "?"
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
