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
from shadow_loom_ui.theme import feather

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_QUERY_TYPES = [
    ("general", "General", "chat"),
    ("observation", "Continue", "auto_stories"),
    ("intervention", "Intervene", "flash_on"),
    ("counterfactual", "What-If", "alt_route"),
    ("directive", "Direct", "theater_comedy"),
    ("interrogate", "Ask", "psychology"),
    ("evaluate", "Evaluate", "fact_check"),
]

# Writer-friendly prompt starters mapped to query types
_PROMPT_STARTERS = [
    ("Continue the story…", "observation"),
    ("What would happen if…", "counterfactual"),
    ("Make this scene more suspenseful…", "directive"),
    ("Why does this character…", "interrogate"),
    ("What does {entity} know?", "interrogate"),
    ("Show me {entity}'s relationships", "general"),
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

            def _on_type_change():
                val = type_select.value
                if val == "manual_edit":
                    manual_mode["active"] = True
                    selected_type["value"] = ""
                else:
                    manual_mode["active"] = False
                    selected_type["value"] = val
                _update_placeholder()

            type_select.on("update:model-value", _on_type_change)

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
                typing_row.set_visibility(False)
                send_btn.props(remove="loading")
                _is_running["v"] = False

            _render_messages(chat_container, messages)
            _refresh_implausible_row(result)

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


def _build_context_suggestions(state: AppState, container) -> None:  # pragma: no cover - removed
    """Deprecated: in-bar suggestion chips were removed for less clutter."""
    return


def _render_messages(container, messages: List[dict]) -> None:
    """Re-render chat messages."""
    container.clear()
    with container:
        for msg in messages[-30:]:
            is_user = msg["role"] == "user"
            with ui.chat_message(
                sent=is_user,
                name="You" if is_user else "Shadow Loom",
            ).classes("w-full"):
                ui.markdown(msg["text"])


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
