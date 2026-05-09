# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared renderer for Pearl-ladder rung chips on result cards.

The Story tab prose cards, Answer panel cards, and Audit-log query
rows all want to mark each result with the Pearl causal rung it
came from so a researcher can audit the engine's epistemic stance
at a glance:

* **Rung 1: Observation** — pure read-only Q&A or seeing-conditioning.
* **Rung 2: Intervention** — ``do(x)`` style write that mutates the
  factual world (or its shadow).
* **Rung 3: Counterfactual** — replay-with-a-changed-past that forks
  a shadow branch.

Other ``query_type`` values (``directive``, ``evaluate``,
``manual_edit``) are not Pearl-ladder operations and render no chip;
``ingestion`` is the bootstrap and likewise has no rung.

Centralising this here keeps the chip vocabulary and colours
consistent across surfaces and lets future query types map to a rung
in one place.
"""
from __future__ import annotations

from nicegui import ui

# query_type → (rung number, label, Quasar colour)
_RUNG_BY_QTYPE: dict[str, tuple[int, str, str]] = {
    # Rung 1 — Observation / seeing
    "observation": (1, "Rung 1: Observation", "blue"),
    "general": (1, "Rung 1: Observation", "blue"),
    "interrogate": (1, "Rung 1: Observation", "blue"),
    # Rung 2 — Intervention / do(x)
    "intervention": (2, "Rung 2: Intervention", "deep-orange"),
    # Rung 3 — Counterfactual / replay-with-altered-past
    "counterfactual": (3, "Rung 3: Counterfactual", "purple"),
}


def pearl_rung_for(qtype: str | None) -> tuple[int, str, str] | None:
    """Return the Pearl rung tuple ``(rung, label, colour)`` for a
    ``query_type``, or ``None`` if the qtype is not a Pearl-ladder
    operation."""
    if not qtype:
        return None
    return _RUNG_BY_QTYPE.get(str(qtype).lower())


def render_pearl_chip(qtype: str | None) -> None:
    """Render a Quasar badge tagging the Pearl rung of a result.

    No-op when ``qtype`` is not a Pearl-ladder query type
    (e.g. ``directive``, ``evaluate``, ``manual_edit``,
    ``ingestion``) so the surface stays uncluttered for ops that
    don't have a rung.
    """
    info = pearl_rung_for(qtype)
    if info is None:
        return
    _, label, colour = info
    ui.badge(label, color=colour).props("dense outline")
