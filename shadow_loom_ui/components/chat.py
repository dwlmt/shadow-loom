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
            typing_label = ui.label("Processing…").classes("text-caption text-grey")

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
                "round dense color=primary"
            ).classes("q-mb-xs")

        # ── Keyboard shortcut ─────────────────────────────────────
        text_input.on(
            "keydown.enter",
            lambda e: _send() if not getattr(e, "args", {}).get("shiftKey") else None,
        )

        # ── Send logic ────────────────────────────────────────────
        async def _send():
            text = text_input.value.strip()
            if not text:
                return
            text_input.value = ""

            # Add user message
            messages.append({"role": "user", "text": text})
            _render_messages(chat_container, messages)
            history_expansion.value = True  # Show history when sending

            if state.world_state is None:
                messages.append({
                    "role": "assistant",
                    "text": "⚠️ No world model loaded. Ingest a story first.",
                })
                _render_messages(chat_container, messages)
                return

            typing_row.set_visibility(True)
            send_btn.props("loading")
            state.emit(StateEvent.QUERY_STARTED)

            try:
                if manual_mode["active"]:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: state.run_manual_edit(text),
                    )
                else:
                    qtype = selected_type["value"] or "general"
                    result = await state.run_nl_query_async(text, query_type=qtype)
                _append_result(messages, result)
            except Exception as e:
                logger.exception("Command bar query failed")
                messages.append({"role": "assistant", "text": f"❌ Error: {e}"})
            finally:
                typing_row.set_visibility(False)
                send_btn.props(remove="loading")

            _render_messages(chat_container, messages)
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
            if pr.prose:
                excerpt = pr.prose[:800]
                if len(pr.prose) > 800:
                    excerpt += "\n\n*…see Story tab for full text*"
                parts.append(excerpt)
            if pr.physics_state and not pr.prose:
                # For interrogate/general — show structured answer
                parts.append(f"```json\n{pr.physics_state[:600]}\n```")
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
