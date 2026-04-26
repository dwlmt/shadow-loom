"""Bottom chat drawer — persistent query bar across all workspace tabs.

Provides query type chips, chat history, typing indicator, and
routes results to appropriate pipeline paths.
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
    ("general", "General"),
    ("observation", "Observe"),
    ("intervention", "Intervene"),
    ("counterfactual", "Counterfactual"),
    ("directive", "Directive"),
    ("interrogate", "Interrogate"),
    ("evaluate", "Evaluate"),
]


def build_chat_drawer(state: AppState) -> None:
    """Build a collapsible bottom chat drawer."""

    with ui.expansion("Chat", icon="chat", value=False).classes(
        "w-full"
    ).props("dense").style("border-top: 1px solid #333"):
        _build_chat_content(state)


def build_chat_panel(state: AppState) -> None:
    """Legacy alias — builds chat content inline (no drawer wrapper)."""
    _build_chat_content(state)


def _build_chat_content(state: AppState) -> None:
    """Chat content: message history + input bar."""

    messages: List[dict] = []

    # Message display area
    scroll = ui.scroll_area().classes("w-full").style("max-height: 250px")
    with scroll:
        chat_container = ui.column().classes("w-full q-pa-xs gap-1")

    # Typing indicator
    typing_row = ui.row().classes("w-full q-px-sm items-center gap-2")
    typing_row.set_visibility(False)
    with typing_row:
        ui.spinner("dots", size="sm")
        ui.label("Processing...").classes("text-caption text-grey")

    # Input area
    selected_type = {"value": "general"}
    manual_mode = {"active": False}

    with ui.row().classes("w-full items-end q-pa-xs gap-1"):
        # Query type chips
        with ui.row().classes("gap-1 flex-wrap"):
            chip_refs = {}
            for key, label in _QUERY_TYPES:
                chip = ui.chip(
                    label,
                    selectable=True,
                    selected=(key == "general"),
                    on_click=lambda k=key: _select_type(k),
                ).props("dense outline size=sm")
                chip_refs[key] = chip

            edit_chip = ui.chip(
                "Edit",
                icon="edit",
                selectable=True,
                on_click=lambda: _toggle_manual(),
            ).props("dense outline size=sm")

        def _select_type(key: str):
            selected_type["value"] = key
            manual_mode["active"] = False
            for k, c in chip_refs.items():
                c.set_selected(k == key)
            edit_chip.set_selected(False)

        def _toggle_manual():
            manual_mode["active"] = not manual_mode["active"]
            if manual_mode["active"]:
                for c in chip_refs.values():
                    c.set_selected(False)
            else:
                _select_type(selected_type["value"])

        text_input = ui.textarea(
            placeholder="Ask about your story or write prose...",
        ).classes("flex-grow").props("rows=1 autogrow outlined dense")

        async def _send():
            text = text_input.value.strip()
            if not text:
                return
            text_input.value = ""

            # Add user message
            messages.append({"role": "user", "text": text})
            _render_messages(chat_container, messages)

            if state.world_state is None:
                messages.append({
                    "role": "assistant",
                    "text": "No world model loaded. Ingest a story first.",
                })
                _render_messages(chat_container, messages)
                return

            typing_row.set_visibility(True)
            state.emit(StateEvent.QUERY_STARTED)

            try:
                if manual_mode["active"]:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: state.run_manual_edit(text),
                    )
                else:
                    result = await state.run_nl_query_async(
                        text, query_type=selected_type["value"]
                    )
                _append_result(messages, result)
            except Exception as e:
                logger.exception("Chat query failed")
                messages.append({"role": "assistant", "text": f"Error: {e}"})
            finally:
                typing_row.set_visibility(False)

            _render_messages(chat_container, messages)

        ui.button(icon="send", on_click=_send).props("round dense color=primary")

    text_input.on(
        "keydown.enter",
        lambda e: _send() if not getattr(e, "args", {}).get("shiftKey") else None,
    )


def _render_messages(container, messages: List[dict]) -> None:
    """Re-render chat messages."""
    container.clear()
    with container:
        for msg in messages[-20:]:  # Keep last 20 visible
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
            parts.append(f"*{pr.query_type}*")
            if pr.prose:
                parts.append(pr.prose[:1000])
                if len(pr.prose) > 1000:
                    parts.append("*(truncated — see Story tab for full text)*")
            if pr.physics_state and not pr.prose:
                parts.append(f"```json\n{pr.physics_state[:800]}\n```")
            if pr.converged is not None:
                status = "converged" if pr.converged else "did not converge"
                parts.append(f"*Audit: {status} ({pr.audit_iterations} iters)*")
            if pr.world_model:
                parts.append(f"*World model v{pr.world_model.version}*")

        if not parts:
            parse = result.parse_result
            if parse and parse.parsed:
                parts.append(f"**Reasoning:** {parse.parsed.reasoning}")

    messages.append({
        "role": "assistant",
        "text": "\n\n".join(parts) if parts else (result.summary or "Done."),
    })
