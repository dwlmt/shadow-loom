# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command bar — persistent always-visible NL interface at the bottom.

Natural language is the PRIMARY interaction mode. The command bar is
always visible (not collapsed), with context-aware prompt suggestions,
smart type routing, and inline result previews.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List

from nicegui import ui

from shadow_loom_ui.state import AppState, NLQueryResult, StateEvent
from shadow_loom_ui.task_helpers import capture_logs_to_task, notify_task_complete
from shadow_loom_ui.components.help_popover import help_popover
from shadow_loom_ui.components._safe_md import safe_markdown

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_QUERY_TYPES = [
    ("general", "Ask", "chat"),
    ("observation", "Continue", "auto_stories"),
    ("intervention", "Intervene", "flash_on"),
    ("counterfactual", "What-If", "alt_route"),
    ("directive", "Direct", "theater_comedy"),
    ("interrogate", "Interrogation", "psychology"),
    ("evaluate", "Evaluate", "fact_check"),
]

# One-line hover hints, keyed by query type. Used both in the Mode
# selector tooltip and in the help-popover markdown table so the two
# surfaces stay in sync.
_QUERY_TYPE_HINTS: dict[str, str] = {
    "": (
        "Auto-detect — let the parser pick the right mode from your "
        "wording. Safe default."
    ),
    "general": (
        "Ask — read-only Q&A over the world graph. Returns an answer; "
        "does not advance the timeline or write prose."
    ),
    "observation": (
        "Continue — generate the next scene in chronological order. "
        "Advances time, writes prose, creates a new factual version."
    ),
    "intervention": (
        "Intervene — surgically force a state change "
        "(\"kill X\", \"move Y to Z\") and let the physics propagate "
        "the consequences forward. Writes a new factual version."
    ),
    "counterfactual": (
        "What-If — re-run history under a changed past event "
        "(\"what if Banquo had survived?\"). Forks a SHADOW branch so "
        "the factual mainline is preserved."
    ),
    "directive": (
        "Direct — optimise the next scene for a specific emotional "
        "effect (suspense, dread, dramatic irony, …). Searches over "
        "candidate beats and picks the highest-scoring one."
    ),
    "interrogate": (
        "Interrogate — graph pathfinding with proof "
        "(\"Who knows X?\", \"Is there a path from A to B?\"). "
        "Read-only; returns Causal Bridges as evidence."
    ),
    "evaluate": (
        "Evaluate — score the whole story so far against the "
        "narrative-quality scorecard (causal plausibility, affective "
        "trajectory, coherence). Read-only."
    ),
    "manual_edit": (
        "Write prose — bypass the engine and paste your own prose. "
        "Re-extracts topology and merges as a new factual version."
    ),
}

_MODE_HELP_BODY = (
    "Pick how the engine should interpret your input. "
    "**Auto-detect** is usually correct \u2014 these explicit modes are "
    "for cases where you want to override the parser.\n\n"
    "| Mode | What it does |\n"
    "|---|---|\n"
    "| **Auto-detect** | Let the parser pick the right mode from your wording. |\n"
    "| **Ask** | Read-only Q&A over the world graph. No prose, no version. |\n"
    "| **Continue** | Generate the next scene in chronological order. New factual version. |\n"
    "| **Intervene** | Force a state change and propagate consequences. New factual version. |\n"
    "| **What-If** | Re-run history under a changed past event. Forks a *shadow* branch. |\n"
    "| **Direct** | Optimise the next scene for a target emotional effect. |\n"
    "| **Interrogation** | Graph pathfinding with Causal-Bridge proof. Read-only. |\n"
    "| **Evaluate** | Score the whole story against the narrative-quality scorecard. |\n"
    "| **\u270f Write prose** | Bypass the engine and paste your own prose as canon. |\n\n"
    "Shadow branches stay browsable in the version sidebar and can be "
    "promoted onto the factual mainline later."
)

# Writer-friendly prompt starters mapped to query types.
# Every starter must be runnable as written — no unbound
# ``{entity}`` placeholders that would make it through to the parser
# untouched. Suggestions that need a specific entity are emitted by
# the Explorer / World tabs, where the selected node is in scope.
_PROMPT_STARTERS = [
    ("Continue the story…", "observation"),
    ("What would happen if…", "counterfactual"),
    ("Make this scene more suspenseful…", "directive"),
    ("Why does this character do that?", "interrogate"),
    ("Who knows what at this point in the story?", "interrogate"),
    ("What are the active relationships right now?", "general"),
]


def build_chat_drawer(state: AppState) -> None:
    """Build the persistent command bar — always visible at bottom."""
    _build_command_bar(state)


def build_chat_panel(state: AppState) -> None:
    """Legacy alias."""
    _build_command_bar(state)


def _build_command_bar(state: AppState) -> None:
    """Persistent command bar: input always visible, history expands upward."""

    messages: List[dict] = []
    selected_type = {"value": ""}  # Empty = auto-detect
    manual_mode = {"active": False}
    last_request = {"text": "", "qtype": "", "manual": False, "forced": False}

    with ui.column().classes("w-full gap-0").style(
        "border-top: 2px solid #444; background: #1a1a2e; flex: 0 0 auto;"
    ):
        # Messages are tracked but not displayed inline — the bar is a
        # permanent narrow strip. (Results still surface via toast
        # notifications + the per-tab result panels.)
        chat_container = ui.column()
        chat_container.set_visibility(False)

        # ── Typing / loading indicator (only visible while running) ─
        typing_row = ui.row().classes("w-full q-px-md items-center gap-2")
        typing_row.set_visibility(False)
        with typing_row:
            ui.spinner("dots", size="sm", color="primary")
            typing_label = ui.label("Processing\u2026").classes(
                "text-xs text-slate-500"
            )

        # ── Implausibility action row (shown after a flagged result) ──
        implausible_row = ui.row().classes("w-full q-px-md items-center gap-2")
        implausible_row.set_visibility(False)


        # ── Main input row (ALWAYS VISIBLE, fixed height) ─────────
        with ui.row().classes(
            "w-full items-center q-px-sm q-py-xs gap-2 no-wrap"
        ).style("min-height: 64px;"):
            # Query type selector — sized to fit the longest option label
            # ("✏ Write prose") so the combo never truncates. ``stack-label``
            # keeps the floating label compact and ``items-center`` (above)
            # vertically aligns the field with the textarea.
            type_select = ui.select(
                options={
                    "": "Auto-detect",
                    **{k: label for k, label, _ in _QUERY_TYPES},
                    "manual_edit": "✏ Write prose",
                },
                value="",
                label="Mode",
            ).props(
                "dense outlined options-dense stack-label "
                "bg-color=white behavior=menu"
            ).style("min-width: 156px; height: 48px;")
            # Hover tooltip on the field itself — updates as the user
            # changes mode so the hint always matches the current pick.
            with type_select:
                _mode_tip = ui.tooltip(_QUERY_TYPE_HINTS[""]).classes(
                    "text-xs max-w-xs leading-snug whitespace-normal"
                )

            # Click-to-open help card listing every mode side-by-side
            # for users who want to compare before choosing.
            help_popover(
                "Query modes",
                _MODE_HELP_BODY,
                tooltip="What does each mode do?",
            )

            def _on_type_change():
                val = type_select.value
                if val == "manual_edit":
                    manual_mode["active"] = True
                    selected_type["value"] = ""
                else:
                    manual_mode["active"] = False
                    selected_type["value"] = val
                # Keep the hover hint in sync with the active mode.
                _mode_tip.text = _QUERY_TYPE_HINTS.get(
                    val or "", _QUERY_TYPE_HINTS[""],
                )
                _mode_tip.update()
                _update_placeholder()
                _refresh_anchor_visibility()

            type_select.on("update:model-value", _on_type_change)

            # ── Manual-edit anchor selector (visible only in Write-prose mode)
            # Lets the user say *where* in the timeline the new prose
            # belongs ("End of story" by default; or after a specific
            # event). The selected event's fabula_time anchors the
            # re-extraction so events land in the right slot rather
            # than colliding with existing chronology.
            anchor_options: dict[str, str] = {"": "Append at end"}
            anchor_state = {"event_id": None}

            def _refresh_anchor_options() -> None:
                ws = state.world_state
                opts = {"": "Append at end"}
                if ws is not None:
                    sorted_evts = sorted(
                        ws.events, key=lambda e: e.fabula_time,
                    )[-50:]  # last 50 by fabula time keeps the list sane
                    for e in sorted_evts:
                        desc = (e.description or e.id)[:40]
                        opts[e.id] = f"After t={e.fabula_time}: {desc}"
                anchor_select.options = opts
                if anchor_select.value not in opts:
                    anchor_select.value = ""
                anchor_select.update()

            anchor_select = ui.select(
                options=anchor_options,
                value="",
                label="Insert",
            ).props(
                "dense outlined options-dense stack-label "
                "bg-color=white behavior=menu"
            ).style("min-width: 180px; height: 48px;")
            anchor_select.tooltip(
                "Where in the timeline the new prose lands. "
                "'Append at end' continues the story; pick an event "
                "to insert immediately after it."
            )

            def _on_anchor_change():
                anchor_state["event_id"] = anchor_select.value or None

            anchor_select.on("update:model-value", _on_anchor_change)

            def _refresh_anchor_visibility() -> None:
                if manual_mode["active"]:
                    _refresh_anchor_options()
                    anchor_select.set_visibility(True)
                else:
                    anchor_select.set_visibility(False)

            anchor_select.set_visibility(False)
            state.on(
                StateEvent.WORLD_STATE_CHANGED,
                lambda **_kw: _refresh_anchor_options() if manual_mode["active"] else None,
            )

            # Main text input — fixed two visible lines.
            text_input = ui.textarea(
                placeholder="Ask anything about your story, or describe what should happen next…",
            ).classes("flex-grow").props(
                "outlined dense bg-color=white "
                "input-style='height: 48px; min-height: 48px; "
                "max-height: 48px; overflow-y: auto;'"
            )

            def _update_placeholder():
                if manual_mode["active"]:
                    text_input.props('placeholder="Write your prose here — it will become canon…"')
                elif selected_type["value"] == "directive":
                    text_input.props('placeholder="Describe the emotional effect you want (e.g. make the reader feel dread)…"')
                elif selected_type["value"] == "counterfactual":
                    text_input.props('placeholder="What if [something had been different]…"')
                elif selected_type["value"] == "intervention":
                    text_input.props('placeholder="Force a change: kill [character], move [entity] to [location]…"')
                else:
                    text_input.props('placeholder="Ask anything about your story, or describe what should happen next…"')

            # Send button
            send_btn = ui.button(icon="send", on_click=lambda: _send()).props(
                "unelevated round dense color=primary"
            ).classes("shadow-sm").style("height: 40px; width: 40px;")

            # Help icon — click for full mode reference.
            help_popover(
                title="Channel — the natural-language command bar",
                body_md=(
                    "This bar is your conduit to the world model. Type a"
                    " question or instruction, optionally pick a **mode**,"
                    " and hit **Send** (or `Enter`; `Shift+Enter` for a"
                    " newline).\n\n"
                    "### Modes\n"
                    "| Mode | What it does | Saves a version? | Surfaces in |\n"
                    "|------|--------------|------------------|-------------|\n"
                    "| **Auto-detect** | Parses the text and routes to the best mode. | depends | depends |\n"
                    "| **Ask** | Read-only Q&A over the world graph. | no | Answer panel |\n"
                    "| **Continue** | Generate the next scene. | yes (factual) | Story tab |\n"
                    "| **Intervene** | Force a change *now* and continue. | yes (factual) | Story tab |\n"
                    "| **What-If** | Replay an alternate history from a past divergence. | yes (shadow) | Story tab |\n"
                    "| **Direct** | Target an emotional effect (dread, suspense, irony). | yes (factual) | Story tab |\n"
                    "| **Interrogation** | Diagnostic causal Q&A; can require explicit proof. | no | Answer panel |\n"
                    "| **Evaluate** | Audit the whole story for quality / plausibility. | no | Audit tab |\n"
                    "| **✏ Write prose** | Paste your own canon prose; re-extracted into the model. | yes (factual) | Story tab |\n\n"
                    "### Insert anchor (Write-prose only)\n"
                    "When you pick **✏ Write prose**, an *Insert*"
                    " selector appears beside the mode picker.\n"
                    "- **Append at end** *(default)* — the new prose"
                    " continues from the chronological end of the"
                    " story.\n"
                    "- **After t=… : <event>** — the new prose is"
                    " anchored immediately after that event, so its"
                    " re-extracted events get a `fabula_time` that"
                    " slots into the right place rather than"
                    " colliding with existing chronology.\n\n"
                    "### Implausibility gate\n"
                    "If the engine cannot resolve your request against the"
                    " current world state (unknown character, dead"
                    " already, etc.) it short-circuits with an"
                    " explanation and **does not change the world**. A"
                    " *Force generate anyway* button appears so you can"
                    " override.\n\n"
                    "### Tips\n"
                    "- Reference characters by name; the parser resolves"
                    " them to ids.\n"
                    "- Be concrete: `Kill Macbeth at the castle` beats"
                    " `something bad happens`.\n"
                    "- Background tasks survive tab navigation; the"
                    " tasks indicator (top bar) shows in-flight work.\n"
                    "- The Reasoning tab shows the engine's step-by-step"
                    " trace for the most recent rung-2/rung-3 query."
                ),
                tooltip="What can I type here?",
                icon_classes="text-slate-400 cursor-pointer text-base",
            )

        # ── Keyboard shortcut ─────────────────────────────────────
        text_input.on(
            "keydown.enter",
            lambda e: _send() if not getattr(e, "args", {}).get("shiftKey") else None,
        )

        # ── Send logic ────────────────────────────────────────────
        _is_running = {"v": False}

        def _refresh_implausible_row(result: NLQueryResult | None) -> None:
            implausible_row.clear()
            pr = result.pipeline_result if result else None
            if not pr or not pr.implausible or last_request["forced"] or last_request["manual"]:
                implausible_row.set_visibility(False)
                return
            with implausible_row:
                ui.icon("warning", color="warning")
                ui.label(
                    "Engine deemed this query implausible "
                    "\u2014 the world model was not changed."
                ).classes("text-xs text-slate-500")
                ui.button(
                    "Force generate anyway",
                    icon="bolt",
                    on_click=lambda: _send(force=True),
                ).props("dense outline color=warning size=sm no-caps")
            implausible_row.set_visibility(True)

        async def _send(force: bool = False):
            if _is_running["v"]:
                return
            if force:
                # Prefer the current textarea contents if the user has
                # edited them since the last submission; otherwise fall
                # back to replaying last_request verbatim.
                edited = text_input.value.strip() if text_input.value else ""
                if edited:
                    text = edited
                    qtype_for_run = selected_type["value"] or "general"
                    use_manual = manual_mode["active"]
                else:
                    text = last_request["text"]
                    qtype_for_run = last_request["qtype"]
                    use_manual = last_request["manual"]
                if not text:
                    return
            else:
                text = text_input.value.strip()
                if not text:
                    return
                qtype_for_run = selected_type["value"] or "general"
                use_manual = manual_mode["active"]
                last_request.update({
                    "text": text,
                    "qtype": qtype_for_run,
                    "manual": use_manual,
                    "forced": False,
                })

            _is_running["v"] = True
            if not force:
                text_input.value = ""

            # Add user message
            user_text = text + ("  *(forced)*" if force else "")
            messages.append({"role": "user", "text": user_text})
            _render_messages(chat_container, messages)

            if state.world_state is None:
                messages.append({
                    "role": "assistant",
                    "text": "\u26a0\ufe0f No world model loaded. Ingest a story first.",
                })
                _render_messages(chat_container, messages)
                _is_running["v"] = False
                return

            implausible_row.set_visibility(False)
            typing_row.set_visibility(True)
            send_btn.props("loading")
            state.emit(StateEvent.QUERY_STARTED)

            # Register this query as a tracked background task so it shows up
            # in the header tasks indicator and produces a sticky completion
            # notification (users may navigate to other tabs while it runs).
            task_label = (
                f"{'Manual edit' if use_manual else qtype_for_run.title()}: "
                f"{text[:40]}{'…' if len(text) > 40 else ''}"
            )
            task = state.start_task(
                label=task_label,
                kind="manual_edit" if use_manual else "query",
            )

            result: NLQueryResult | None = None
            try:
                with capture_logs_to_task(state, task):
                    if use_manual:
                        result = await asyncio.to_thread(
                            state.run_manual_edit, text,
                            insert_after_event_id=anchor_state["event_id"],
                        )
                    else:
                        result = await state.run_nl_query_async(
                            text, query_type=qtype_for_run,
                            force_implausible=force,
                        )
                if force:
                    last_request["forced"] = True
                _append_result(messages, result)

                # Build a short summary for the task + notification
                if result and result.error:
                    state.finish_task(task, error=result.error)
                else:
                    summary = (result.summary if result else "") or "Done"
                    state.finish_task(task, result_summary=summary[:120])
                notify_task_complete(task)
            except Exception as e:
                logger.exception("Command bar query failed")
                messages.append({"role": "assistant", "text": f"\u274c Error: {e}"})
                state.finish_task(task, error=str(e))
                notify_task_complete(task)
            finally:
                # All post-task UI mutations need guarding: the user
                # may have navigated away mid-query, in which case the
                # owning client is gone and any element access raises
                # ``RuntimeError("…has been deleted.")``.
                try:
                    typing_row.set_visibility(False)
                    send_btn.props(remove="loading")
                except RuntimeError as exc:
                    logger.debug(
                        "[chat] post-task UI cleanup skipped "
                        "(dead client): %s", exc,
                    )
                _is_running["v"] = False

            try:
                _render_messages(chat_container, messages)
                _refresh_implausible_row(result)
            except RuntimeError as exc:
                logger.debug(
                    "[chat] post-task render skipped "
                    "(dead client): %s", exc,
                )

        # ── Listen for prompt suggestions from other components ───
        def _on_suggestion(**kwargs):
            suggestion = kwargs.get("suggestion", "")
            qtype = kwargs.get("query_type", "")
            if suggestion:
                text_input.value = suggestion
                if qtype:
                    type_select.value = qtype
                    selected_type["value"] = qtype
                    manual_mode["active"] = False
                text_input.run_method("focus")

        state.on(StateEvent.QUERY_STARTED, _on_suggestion)

        # ── Drop chat history when the active branch changes ──────
        # Each chat message card represents the result of a query
        # that was run against a *specific* version's world model
        # (a What-If forked from C, an Interrogation answered from
        # P, an Intervene against the previous head, …). When the
        # user clicks a different version in the sidebar — including
        # walking back to a parent — those cards no longer apply to
        # the freshly loaded world: the answer is wrong, the What-If
        # was branched off a sibling, the intervention targeted an
        # event that may not even exist on this branch. Leaving them
        # rendered makes the bar look as if the engine is still
        # claiming those results for the current branch.
        #
        # ``load_db_version`` already clears ``state.query_history``
        # (per-branch derived state) and emits VERSION_CHANGED in
        # lockstep with WORLD_STATE_CHANGED. Mirror that here by
        # dropping our in-closure ``messages`` buffer so the chat
        # cards disappear at the same moment the world swaps.
        def _clear_chat_on_branch(**_kwargs):
            if not messages:
                return
            messages.clear()
            try:
                _render_messages(chat_container, messages)
                implausible_row.set_visibility(False)
            except RuntimeError as exc:
                logger.debug(
                    "[chat] clear-on-branch render skipped "
                    "(dead client): %s", exc,
                )

        state.on(StateEvent.VERSION_CHANGED, _clear_chat_on_branch)
        state.on(StateEvent.PROJECT_LOADED, _clear_chat_on_branch)


def _build_context_suggestions(state: AppState, container) -> None:  # pragma: no cover - removed
    """Deprecated: in-bar suggestion chips were removed for less clutter."""
    return


def _render_messages(container, messages: List[dict]) -> None:
    """Re-render chat messages.

    Background tasks routinely outlive the page they were launched
    from. If the user navigates away (or refreshes) mid-query, the
    chat container's owning NiceGUI client has been deleted by the
    time the task completes and tries to re-render. Swallow the
    resulting ``RuntimeError`` rather than letting it crash the
    background-task handler.
    """
    try:
        container.clear()
        with container:
            for msg in messages[-30:]:
                is_user = msg["role"] == "user"
                with ui.chat_message(
                    sent=is_user,
                    name="You" if is_user else "Shadow Loom",
                ).classes("w-full"):
                    safe_markdown(msg["text"])
    except RuntimeError as exc:
        msg = str(exc)
        if (
            "has been deleted" in msg
            or "no current slot" in msg.lower()
        ):
            logger.debug(
                "[chat] _render_messages skipped (dead client/slot): %s", exc,
            )
            return
        raise


def _append_result(messages: List[dict], result: NLQueryResult) -> None:
    """Format a pipeline result as a chat message.

    Interrogate/general queries (which don't produce prose) render as
    a *structured response card* (claim → evidence → confidence →
    caveats) rather than a raw JSON dump. Rung-2/rung-3 queries get a
    one-line reasoning-trace summary that hyperlinks the user to the
    Reasoning tab for the full breakdown.
    """
    parts = []

    if result.error:
        parts.append(f"**Error:** {result.error}")
    else:
        pr = result.pipeline_result
        if pr is not None:
            parts.append(f"**{pr.query_type}**")
            if pr.implausible:
                # Show prominent warning regardless of whether prose was generated.
                icon = "\u26a0\ufe0f"
                parts.append(
                    f"{icon} **Implausible request:** {pr.implausibility_reason or 'unspecified'}"
                )
                unresolved = (pr.implausibility_details or {}).get(
                    "unresolved_targets", []
                )
                if unresolved:
                    bullets = "\n".join(
                        f"- `{u.get('target','?')}`: {u.get('reason','?')}"
                        for u in unresolved
                    )
                    parts.append(bullets)
            if pr.prose:
                excerpt = pr.prose[:800]
                if len(pr.prose) > 800:
                    excerpt += "\n\n*\u2026see Story tab for full text*"
                parts.append(excerpt)
            if pr.physics_result and not pr.prose:
                # Structured response card replaces raw JSON dump.
                from shadow_loom_ui.reasoning_helpers import (
                    structured_response_data,
                )
                # We don't have direct access to ws here (only the
                # pipeline result) — labels will fall back to ids,
                # which is fine for the chat preview. Full label
                # resolution happens in the Reasoning tab.
                card = structured_response_data(pr.physics_result)
                claim_md = card["claim"] or "(no claim returned)"
                conf_pct = int(card["confidence"] * 100)
                parts.append(f"**Claim:** {claim_md}")
                parts.append(f"*Confidence: {conf_pct}%*")
                if card["evidence"]:
                    ev_lines = "\n".join(
                        f"- `{e['node_id']}` ({e['kind']})"
                        for e in card["evidence"][:8]
                    )
                    parts.append(f"**Evidence:**\n{ev_lines}")
                if card["caveats"]:
                    cav_lines = "\n".join(f"- {c}" for c in card["caveats"])
                    parts.append(f"**Caveats:**\n{cav_lines}")
                if card.get("rule2_redundant_evidence"):
                    r2_lines = "\n".join(
                        f"- `{e['node_id']}` \u2014 {e['label']}"
                        for e in card["rule2_redundant_evidence"][:5]
                    )
                    parts.append(
                        "**Rule-2 \u2014 redundant evidence pruned:**\n"
                        + r2_lines
                    )
                if card.get("rule3_pruned_interventions"):
                    r3_lines = "\n".join(
                        f"- {p['label']}"
                        + (f" \u2014 *{p['reason']}*" if p.get("reason") else "")
                        for p in card["rule3_pruned_interventions"][:5]
                    )
                    parts.append(
                        "**Rule-3 \u2014 interventions pruned (back-door blocked):**\n"
                        + r3_lines
                    )
            # Rung-2/rung-3 trace summary
            if pr.query_type in ("intervention", "counterfactual") and pr.physics_result:
                from shadow_loom_ui.reasoning_helpers import (
                    extract_reasoning_trace,
                    reasoning_trace_summary,
                )
                trace = extract_reasoning_trace(pr.physics_result)
                summary = reasoning_trace_summary(trace)
                if summary and summary != "no reasoning trace":
                    parts.append(
                        f"\U0001F9E0 *Reasoning trace: {summary}.*  "
                        "*(Open Reasoning tab for the full breakdown.)*"
                    )
            if pr.evaluation_result:
                noo = getattr(pr.evaluation_result, "narrative_order", None)
                if noo:
                    overall = getattr(noo, "overall_pass", None)
                    if overall is not None:
                        parts.append(f"{'✓' if overall else '✗'} **Overall: {'PASS' if overall else 'FAIL'}**")
                    parts.append("*See Audit tab for full scorecard.*")
            if pr.converged is not None:
                icon = "✓" if pr.converged else "⚠"
                status = "converged" if pr.converged else "did not converge"
                parts.append(f"{icon} *Audit: {status} ({pr.audit_iterations} iterations)*")

            # Surface correction_error so users see when the loop
            # bypass-passed (failed-open auditor) or aborted on a
            # generation/refinement failure. Without this, those paths
            # can show "converged" with no explanation of what happened.
            fb_err = getattr(getattr(pr, "feedback_result", None), "correction_error", None)
            if fb_err:
                parts.append(f"⚠ *Auditor diagnostic: {fb_err}*")
            if getattr(pr, "reextraction_failed", False):
                rx_err = getattr(pr, "reextraction_error", None) or "re-extraction skipped"
                parts.append(f"⚠ *World model not updated — {rx_err}*")

            # Engine threshold gate + achieved-vs-target intensity ─
            # surface deterministic affective metrics so the user sees
            # WHY the auditor said pass/fail without opening the
            # Audit tab.
            fb = getattr(pr, "feedback_result", None)
            if fb is not None:
                if fb.engine_thresholds_passed is False and fb.engine_threshold_failures:
                    failure_lines = "\n".join(
                        f"- {_friendly_chat_threshold(f)}"
                        for f in fb.engine_threshold_failures[:4]
                    )
                    parts.append(
                        "⚠ **Quality thresholds missed:**\n" + failure_lines
                    )
                ci = fb.change_impact
                parsed = result.parse_result.parsed if result.parse_result else None
                req_effect = getattr(parsed, "target_effect", None) if parsed else None
                req_intensity = getattr(parsed, "intensity", None) if parsed else None
                if ci is not None and ci.affective_feedback is not None:
                    af = ci.affective_feedback
                    scores = af.emotional_trajectory_scores or {}
                    if req_effect and req_effect in scores:
                        achieved = scores[req_effect]
                        if req_intensity is not None:
                            gap = achieved - req_intensity
                            arrow = "✓" if abs(gap) <= 0.15 else ("↑" if gap > 0 else "↓")
                            parts.append(
                                f"{arrow} *{req_effect.replace('_',' ').title()}: "
                                f"asked {req_intensity:.2f}, achieved {achieved:.2f} "
                                f"(gap {gap:+.2f}).*"
                            )
                        else:
                            parts.append(
                                f"*{req_effect.replace('_',' ').title()} "
                                f"intensity achieved: {achieved:.2f}.*"
                            )
                    if af.affective_loss_mse is not None and af.affective_loss_mse > 0:
                        parts.append(
                            f"*Distance from requested feeling: "
                            f"{af.affective_loss_mse:.2f} (lower is better).*"
                        )
            if pr.world_model:
                parts.append(f"📝 *World model updated → v{pr.world_model.version}*")

        if not parts:
            parse = result.parse_result
            if parse and parse.parsed:
                parts.append(f"**Reasoning:** {parse.parsed.reasoning}")

    messages.append({
        "role": "assistant",
        "text": "\n\n".join(parts) if parts else (result.summary or "Done."),
    })


_CHAT_FRIENDLY_THRESHOLDS = {
    "foreshadowing_payoff_score": "foreshadowing pay-off was too low",
    "cognitive_plausibility_score": "characters' beliefs weren't consistent enough",
    "affective_loss_mse": "the scene's emotional fit missed the target",
    "miracle_steps_detected": "an unexplained leap was detected",
}


def _friendly_chat_threshold(failure: str) -> str:
    """Translate an engine threshold failure into a chat-friendly string."""
    for key, label in _CHAT_FRIENDLY_THRESHOLDS.items():
        if failure.startswith(key):
            return f"{label} — `{failure}`"
    return failure
