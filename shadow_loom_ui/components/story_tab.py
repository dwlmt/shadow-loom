"""Story tab — unified chat interface, source text viewer, query type chips.

This is the primary writer-facing surface. All query types are accessible
through a single chat input with optional type override chips.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

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
]


def build_story_tab(state: AppState) -> None:
    """Build the Story tab with chat interface and source text."""

    with ui.splitter(value=70).classes("w-full h-full"):
        with ui.splitter.before:
            _build_chat_area(state)
        with ui.splitter.after:
            _build_context_sidebar(state)


def _build_chat_area(state: AppState) -> None:
    """Main chat area — messages + input."""

    # ---- Source text (collapsible) ----
    if state.raw_text:
        with ui.expansion("Source Text", icon="description").classes("w-full"):
            ui.markdown(state.raw_text[:5000] + ("..." if len(state.raw_text) > 5000 else ""))

    # ---- Chat messages ----
    messages_container = ui.column().classes("w-full flex-grow q-pa-sm gap-2")
    scroll = ui.scroll_area().classes("w-full flex-grow")

    with scroll:
        with messages_container:
            if not state.query_history:
                with ui.row().classes("w-full justify-center q-pa-lg"):
                    with ui.column().classes("items-center gap-2"):
                        ui.icon("chat_bubble_outline", size="xl", color="grey")
                        ui.label("Ask anything about your story").classes(
                            "text-body1 text-grey"
                        )
                        ui.label(
                            "Try: 'What happens if Romeo never meets Juliet?' or "
                            "'Describe the scene at the party'"
                        ).classes("text-caption text-grey")

    # ---- Typing indicator ----
    typing_row = ui.row().classes("w-full q-px-sm items-center gap-2")
    typing_row.set_visibility(False)
    with typing_row:
        ui.spinner("dots", size="sm")
        ui.label("Thinking...").classes("text-caption text-grey")

    # ---- Input area ----
    selected_type = {"value": "general"}
    manual_mode = {"active": False}

    with ui.row().classes("w-full items-end q-pa-sm gap-2"):
        # Query type chips
        with ui.row().classes("gap-1 flex-wrap"):
            chip_refs = {}
            for key, label in _QUERY_TYPES:
                chip = ui.chip(
                    label,
                    selectable=True,
                    selected=(key == "general"),
                    on_click=lambda k=key: _select_type(k),
                ).props("dense outline")
                chip_refs[key] = chip

            # Manual edit toggle
            edit_chip = ui.chip(
                "Manual Edit",
                icon="edit",
                selectable=True,
                on_click=lambda: _toggle_manual(),
            ).props("dense outline")

        def _select_type(key: str):
            selected_type["value"] = key
            manual_mode["active"] = False
            for k, c in chip_refs.items():
                c.set_selected(k == key)
            edit_chip.set_selected(False)
            text_input.props(
                'label="Ask about your story..."' if key != "directive"
                else 'label="Write a directive for the narrative..."'
            )

        def _toggle_manual():
            manual_mode["active"] = not manual_mode["active"]
            if manual_mode["active"]:
                for c in chip_refs.values():
                    c.set_selected(False)
                text_input.props('label="Write narrative prose..."')
            else:
                _select_type(selected_type["value"])

        # Text input
        text_input = ui.textarea(
            label="Ask about your story...",
            placeholder="Type your question or narrative here...",
        ).classes("flex-grow").props("rows=2 autogrow")

        async def _send():
            text = text_input.value.strip()
            if not text:
                return
            text_input.value = ""

            # Add user message
            with messages_container:
                ui.chat_message(
                    text=text,
                    name=state.display_name or state.username or "You",
                    sent=True,
                    avatar=state.avatar_url or None,
                ).classes("q-mb-sm")

            typing_row.set_visibility(True)

            try:
                if manual_mode["active"]:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: state.run_manual_edit(text),
                    )
                else:
                    result = await state.run_nl_query_async(
                        text, query_type=selected_type["value"]
                    )

                _render_result_message(messages_container, result)
            except Exception as e:
                logger.exception("Query failed")
                with messages_container:
                    ui.chat_message(
                        text=f"Error: {e}",
                        name="Shadow Loom",
                        sent=False,
                    ).classes("q-mb-sm text-negative")
            finally:
                typing_row.set_visibility(False)

        ui.button(icon="send", on_click=_send).props("round dense color=primary")

    # Allow Enter to send (Shift+Enter for newline)
    text_input.on(
        "keydown.enter",
        lambda e: _send() if not e.args.get("shiftKey") else None,
    )


def _render_result_message(container, result: NLQueryResult) -> None:
    """Render a pipeline result as a chat message."""
    with container:
        parts = []

        if result.error:
            parts.append(f"**Error:** {result.error}")
        else:
            pr = result.pipeline_result
            if pr is not None:
                # Query type badge
                parts.append(f"*{pr.query_type}*")

                # Prose output
                if pr.prose:
                    parts.append(pr.prose)

                # Physics answer (for interrogate/general)
                if pr.physics_state and not pr.prose:
                    parts.append(f"```json\n{pr.physics_state[:2000]}\n```")

                # Audit info
                if pr.converged is not None:
                    status = "converged" if pr.converged else "did not converge"
                    parts.append(
                        f"*Audit: {status} ({pr.audit_iterations} iterations)*"
                    )

                # World model version
                if pr.world_model:
                    parts.append(f"*World model updated to v{pr.world_model.version}*")

            # Parse info (if no pipeline result)
            if not parts or (len(parts) == 1 and parts[0].startswith("*")):
                parse = result.parse_result
                if parse and parse.parsed:
                    parts.append(f"**Reasoning:** {parse.parsed.reasoning}")

        if not parts:
            parts.append(result.summary or "No output.")

        ui.chat_message(
            name="Shadow Loom",
            sent=False,
            avatar="auto_stories",
        )
        with ui.column().classes("q-ml-lg q-mb-sm"):
            for part in parts:
                ui.markdown(part)


def _build_context_sidebar(state: AppState) -> None:
    """Right sidebar — quick entity reference for current context."""

    with ui.column().classes("w-full h-full q-pa-sm"):
        ui.label("Quick Reference").classes("text-subtitle1")
        ui.separator()

        entity_list = ui.column().classes("w-full gap-1")

        def _refresh_entities():
            entity_list.clear()
            if state.world_state is None:
                with entity_list:
                    ui.label("No world model loaded").classes("text-caption text-grey")
                return
            with entity_list:
                for ent in state.world_state.entities[:30]:
                    with ui.expansion(ent.name, icon="person").classes("w-full").props("dense"):
                        if ent.status:
                            ui.badge(ent.status, color="blue-grey").props("dense")
                        if ent.current_location_id:
                            ui.label(f"Location: {ent.current_location_id}").classes(
                                "text-caption"
                            )
                        # Top traits
                        for trait_name, trait_vec in list(ent.traits.items())[:5]:
                            with ui.row().classes("items-center gap-1"):
                                ui.label(trait_name).classes("text-caption")
                                ui.linear_progress(
                                    value=max(0, min(1, (trait_vec.value + 1) / 2)),
                                    show_value=False,
                                ).classes("w-20")

        _refresh_entities()

        # Re-render on world state changes
        state.on(StateEvent.WORLD_STATE_CHANGED, _refresh_entities)
