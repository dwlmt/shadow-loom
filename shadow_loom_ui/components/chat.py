"""Right panel: chat interface for interrogation and general queries."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List

from nicegui import ui

if TYPE_CHECKING:
    from shadow_loom_ui.state import AppState, NLQueryResult

logger = logging.getLogger(__name__)


def build_chat_panel(state: AppState) -> None:
    """Build the right-panel chat interface."""

    messages: List[dict] = []
    chat_container = None
    input_field = None

    async def _send_message():
        text = input_field.value.strip()
        if not text:
            return
        input_field.value = ""

        # Add user message
        messages.append({"role": "user", "text": text})
        _render_messages()

        if state.world_state is None:
            messages.append({
                "role": "assistant",
                "text": "No world model loaded. Please ingest a story first.",
            })
            _render_messages()
            return

        # Show typing indicator
        messages.append({"role": "assistant", "text": "...", "loading": True})
        _render_messages()

        try:
            # Detect manual edit intent
            lower = text.lower()
            if lower.startswith("edit:") or lower.startswith("write:"):
                prose = text.split(":", 1)[1].strip()
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: state.run_manual_edit(prose, description="Chat edit"),
                )
            else:
                selected_type = chat_type_select.value
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: state.run_nl_query(text, query_type=selected_type),
                )
            # Remove typing indicator
            messages.pop()
            _format_chat_response(result)
        except Exception as e:
            messages.pop()
            messages.append({"role": "assistant", "text": f"Error: {e}"})
            logger.exception("Chat query failed")

        _render_messages()

    def _format_chat_response(result: NLQueryResult):
        """Format pipeline result as chat messages."""
        pr = result.parse_result
        parts = []

        # Show what was parsed
        if pr and pr.parsed:
            parts.append(f"**Query type:** {pr.parsed.query_type}")
            if pr.fallback:
                parts.append(f"*(Fallback: {pr.fallback.strategy})*")

        # Show the answer
        if result.error:
            parts.append(f"\n{result.error}")
        elif result.pipeline_result:
            pip = result.pipeline_result
            if pip.prose:
                parts.append(f"\n{pip.prose}")
            elif pip.query_type in ("interrogate", "general"):
                physics = pip.physics_result
                answer = physics.get("answer") or physics.get("physics_state", {})
                if isinstance(answer, str):
                    parts.append(f"\n{answer}")
                elif isinstance(answer, dict):
                    # Summarise key points
                    for k, v in list(answer.items())[:10]:
                        if isinstance(v, str):
                            parts.append(f"**{k}:** {v}")
                        elif isinstance(v, (int, float)):
                            parts.append(f"**{k}:** {v}")

            # World model version
            if pip.world_model:
                parts.append(f"\n*World model v{pip.world_model.version}*")

        messages.append({"role": "assistant", "text": "\n".join(parts) or "Done."})

    def _render_messages():
        """Re-render all chat messages."""
        if chat_container is None:
            return
        chat_container.clear()
        with chat_container:
            for msg in messages:
                is_user = msg["role"] == "user"
                loading = msg.get("loading", False)

                with ui.chat_message(
                    sent=is_user,
                    text_html=False,
                ).classes("w-full"):
                    if loading:
                        ui.spinner("dots", size="sm")
                    else:
                        ui.markdown(msg["text"])

            # Auto-scroll to bottom
            ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")

    # ---- Build the UI ----

    with ui.column().classes("w-full h-full"):
        ui.label("Chat").classes("text-h6 q-pa-sm")
        ui.label(
            "Ask questions in natural language. Interrogation and general "
            "queries work best here."
        ).classes("text-caption text-grey q-px-sm")

        ui.separator()

        chat_container = ui.column().classes(
            "w-full flex-grow overflow-auto q-pa-sm"
        ).style("max-height: calc(100vh - 250px)")

        # Input area
        with ui.row().classes("w-full items-center q-pa-sm gap-2"):
            chat_type_select = ui.select(
                options={
                    "observation": "Observation",
                    "intervention": "Intervention",
                    "counterfactual": "Counterfactual",
                    "directive": "Directive",
                    "interrogate": "Interrogation",
                    "general": "General",
                },
                value="general",
                label="Type",
            ).classes("w-36").props("outlined dense")

            input_field = ui.input(
                placeholder="Ask about the story world...",
            ).classes("flex-grow").props("outlined dense")
            input_field.on("keydown.enter", _send_message)

            ui.button(icon="send", on_click=_send_message).props("flat dense color=primary")
