# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTML-safe markdown rendering helpers.

NiceGUI's :func:`nicegui.ui.markdown` passes raw HTML straight through
to the browser by default (markdown2 with no ``extras=["safe"]`` does
not strip embedded ``<script>`` / ``<iframe>`` / ``<img onerror=…>``
tags). Almost every markdown call site in the UI displays LLM-generated
or user-supplied content — so any text containing a raw HTML payload
would execute in the visitor's browser, a textbook stored-XSS vector.

:func:`safe_markdown` HTML-escapes the input *before* markdown
rendering. Markdown syntax (``**bold**``, ``# headings``, fenced code
blocks, tables, links written with ``[text](url)``) still works
because markdown2 operates on the escaped text. Inline raw HTML such
as ``<script>``, ``<iframe>``, or ``<img onerror>`` is rendered as
inert literal text.
"""
from __future__ import annotations

import html
from typing import Any

from nicegui import ui


def safe_markdown(text: str | None, **kwargs: Any):
    """Render *text* as markdown with raw HTML escaped.

    Returns the underlying ``ui.markdown`` element so callers can chain
    ``.classes(...)``, ``.style(...)``, etc., as with the unsafe variant.

    ``text`` is coerced to ``str`` and ``html.escape``-d before being
    handed to markdown2; ``None`` becomes an empty string.
    """
    safe = html.escape(str(text or ""), quote=False)
    return ui.markdown(safe, **kwargs)


__all__ = ["safe_markdown"]
