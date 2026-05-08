# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared one-click *copy to clipboard* icon button.

Used by the Story tab prose cards and the Answer panel Q&A cards so a
researcher can pull a generated paragraph or claim straight into a
side document without re-typing or scrubbing the export tab. The
actual copy is delegated to ``navigator.clipboard.writeText`` via
``ui.run_javascript`` so the user's clipboard permissions apply
unchanged.
"""
from __future__ import annotations

import json

from nicegui import ui


def copy_button(text: str, *, tooltip: str = "Copy to clipboard") -> None:
    """Render a small flat copy-icon button that copies ``text``.

    The text is JSON-encoded into the JS snippet so newlines, quotes,
    and Unicode round-trip safely. A toast confirms the copy. No-op
    when ``text`` is empty so empty cards do not advertise a useless
    button.
    """
    if not text:
        return

    def _copy(_e=None) -> None:
        ui.run_javascript(
            f"navigator.clipboard.writeText({json.dumps(text)})"
        )
        ui.notify(
            "Copied to clipboard", type="positive",
            position="top-right", timeout=1500,
        )

    ui.button(
        icon="content_copy",
        on_click=_copy,
    ).props(
        'flat dense round size=sm color=secondary '
        f'aria-label="{tooltip}"'
    ).tooltip(tooltip)
