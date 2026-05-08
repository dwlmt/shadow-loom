# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reusable help popout — info icon that opens a markdown dialog.

Used by the main panels (command bar, version sidebar, answer panel,
etc.) to give users a discoverable, in-place explanation of what the
panel does, what each control means, and which query types it
produces.

Tooltips are good for one-line hints; this component is for the cases
where a tooltip would be too long to read on hover and where the user
benefits from a click-to-open card they can scan and dismiss.
"""

from __future__ import annotations

from nicegui import ui


def help_popover(
    title: str,
    body_md: str,
    *,
    icon: str = "help_outline",
    icon_classes: str = "text-slate-400 cursor-pointer text-base hover:text-primary",
    tooltip: str = "What does this panel do?",
) -> None:
    """Render an info icon that opens a markdown dialog on click.

    Parameters
    ----------
    title:
        Heading shown at the top of the popup card.
    body_md:
        Markdown body. Use bullet lists, **bold** terms, and short
        paragraphs — users will read this in a small dialog.
    icon, icon_classes, tooltip:
        Visual customisation. Defaults match the rest of the UI.
    """

    def _open():
        # Build the dialog once per click and tear it down on close
        # so repeated clicks do not leak elements onto the page.
        dialog = ui.dialog()
        with dialog, ui.card().classes("p-4 gap-2 max-w-md"):
            ui.label(title).classes(
                "text-base font-semibold text-slate-800"
            )
            ui.markdown(body_md).classes(
                "text-sm text-slate-600 leading-relaxed"
            )
            with ui.row().classes("w-full justify-end mt-1"):
                ui.button(
                    "Close",
                    on_click=lambda: (dialog.close(), dialog.delete()),
                ).props("flat dense no-caps color=primary")
        dialog.on("hide", lambda: dialog.delete())
        dialog.open()

    with ui.icon(icon).classes(icon_classes).on("click", _open).props(
        f'role="button" tabindex="0" aria-label="{tooltip}"'
    ):
        ui.tooltip(tooltip).classes("text-xs")
