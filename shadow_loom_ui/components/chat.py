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

    with ui.column().classes("w-full").style(
        "border-top: 2px solid #444; background: #1a1a2e;"
    ):
        # ── Expandable history (scrolls up) ───────────────────────
        history_expansion = ui.expansion(
            "History", icon="history", value=False
        ).classes("w-full").props("dense")
        with history_expansion:
            scroll = ui.scroll_area().classes("w-full").style("max-height: 300px")
            with scroll:
                chat_container = ui.column().classes("w-full q-pa-xs gap-1")

        # ── Typing / loading indicator ────────────────────────────
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

        # ── Context suggestions row ──────────────────────────────
        suggestions_row = ui.row().classes("w-full q-px-md gap-1 flex-wrap")
        _build_context_suggestions(state, suggestions_row)

        # ── Main input row (ALWAYS VISIBLE) ──────────────────────
        with ui.row().classes("w-full items-end q-pa-sm gap-2").style(
            "min-height: 56px;"
        ):
            # Query type selector (compact)
            type_select = ui.select(
                options={
                    "": "Auto-detect",
                    **{k: label for k, label, _ in _QUERY_TYPES},
                    "manual_edit": "✏ Write prose",
                },
                value="",
                label="Mode",
            ).classes("w-32").props("dense outlined")

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

            # Main text input
            text_input = ui.textarea(
                placeholder="Ask anything about your story, or describe what should happen next…",
            ).classes("flex-grow").props(
                "rows=1 autogrow outlined dense"
            ).style("max-height: 120px;")

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
            ).classes("mb-1 shadow-sm")

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
            history_expansion.value = True  # Show history when sending

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
            # Update suggestions after result
            _build_context_suggestions(state, suggestions_row)

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


def _build_context_suggestions(state: AppState, container) -> None:
    """Rebuild context-aware NL prompt suggestions."""
    container.clear()
    ws = state.world_state
    if ws is None:
        with container:
            ui.chip(
                "Ingest a story to get started",
                icon="upload",
            ).props("dense outline size=sm color=grey")
        return

    with container:
        # Dynamic suggestions based on selected node
        if state.selected_node_id and state.selected_node_type == "Entity":
            ent = ws.entities.get(state.selected_node_id)
            if ent:
                name = ent.name
                ui.chip(
                    f"What does {name} believe?",
                    icon="psychology",
                    on_click=lambda n=name: state.emit(
                        StateEvent.QUERY_STARTED,
                        suggestion=f"What does {n} believe about the other characters?",
                        query_type="interrogate",
                    ),
                ).props("dense outline size=sm clickable")
                ui.chip(
                    f"Continue from {name}'s POV",
                    icon="auto_stories",
                    on_click=lambda n=name: state.emit(
                        StateEvent.QUERY_STARTED,
                        suggestion=f"Continue the story from {n}'s point of view",
                        query_type="observation",
                    ),
                ).props("dense outline size=sm clickable")
                ui.chip(
                    f"What if {name} died?",
                    icon="alt_route",
                    on_click=lambda n=name: state.emit(
                        StateEvent.QUERY_STARTED,
                        suggestion=f"What would happen if {n} died?",
                        query_type="counterfactual",
                    ),
                ).props("dense outline size=sm clickable")
                return

        # General suggestions
        entity_names = [e.name for e in list(ws.entities.values())[:3]]
        ui.chip(
            "Continue the story",
            icon="auto_stories",
            on_click=lambda: state.emit(
                StateEvent.QUERY_STARTED,
                suggestion="Continue the story naturally",
                query_type="observation",
            ),
        ).props("dense outline size=sm clickable")

        if entity_names:
            ui.chip(
                f"What if {entity_names[0]}…",
                icon="alt_route",
                on_click=lambda n=entity_names[0]: state.emit(
                    StateEvent.QUERY_STARTED,
                    suggestion=f"What would happen if {n} ",
                    query_type="counterfactual",
                ),
            ).props("dense outline size=sm clickable")

        ui.chip(
            "Make it more suspenseful",
            icon="theater_comedy",
            on_click=lambda: state.emit(
                StateEvent.QUERY_STARTED,
                suggestion="Make the next scene feel more suspenseful and tense",
                query_type="directive",
            ),
        ).props("dense outline size=sm clickable")

        ui.chip(
            "Evaluate quality",
            icon="fact_check",
            on_click=lambda: state.emit(
                StateEvent.QUERY_STARTED,
                suggestion="Evaluate the story quality",
                query_type="evaluate",
            ),
        ).props("dense outline size=sm clickable")


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
    """Format a pipeline result as a chat message."""
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
                # For interrogate/general — show structured answer
                import json as _json
                parts.append(f"```json\n{_json.dumps(pr.physics_result, default=str)[:600]}\n```")
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
