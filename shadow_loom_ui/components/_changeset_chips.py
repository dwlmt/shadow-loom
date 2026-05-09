# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared renderer for ``MergeChangeset`` summaries.

The version sidebar, story-card badges, audit-log activity entries
and version dialogs all want to surface the same "what changed in
this version?" picture. Centralising the rendering here keeps the
chip vocabulary and grouping consistent across surfaces and means a
new counter only has to be added in one place.

Counter source: ``shadow_loom.extract_graph.MergeChangeset`` plus
``shadow_loom.db.get_version_tree``'s normalised summary dict.
"""
from __future__ import annotations

from typing import Optional, Iterable, Tuple

from nicegui import ui


# Group → ordered list of (counter_key, short_label, tooltip).
_CHIP_GROUPS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        "Added", "text-emerald-700 bg-emerald-50 border-emerald-200",
        [
            ("events_added", "evt", "Events added"),
            ("causal_edges_added", "causal", "Causal edges added"),
            ("social_edges_added", "social", "Social edges added"),
            ("spatial_edges_added", "spatial", "Spatial edges added"),
            ("entities_added", "ent", "Entities added"),
            ("objects_added", "obj", "Objects added"),
            ("locations_added", "loc", "Locations added"),
            ("world_traits_added", "world", "World traits added"),
            ("entity_updates_applied", "upd", "Entity updates applied"),
        ],
    ),
    (
        "Affect", "text-indigo-700 bg-indigo-50 border-indigo-200",
        [
            ("propositions_added", "prop", "Propositions added"),
            ("proposition_truths_committed", "truth", "Proposition truths committed"),
            ("proposition_snapshots_added", "p-snap", "Proposition snapshots added"),
            ("concerns_added", "ccn", "Concerns added"),
            ("concern_snapshots_added", "c-snap", "Concern snapshots added"),
            ("belief_confidence_updates_applied", "belief", "Belief confidence updates"),
        ],
    ),
    (
        "Removed", "text-rose-700 bg-rose-50 border-rose-200",
        [
            ("events_removed", "evt", "Events removed"),
            ("causal_edges_removed", "causal", "Causal edges removed"),
            ("social_edges_removed", "social", "Social edges removed"),
            ("spatial_edges_removed", "spatial", "Spatial edges removed"),
            ("channels_removed", "chn", "Channels removed"),
            ("entities_removed", "ent", "Entities removed"),
            ("objects_removed", "obj", "Objects removed"),
            ("locations_removed", "loc", "Locations removed"),
            ("world_traits_removed", "world", "World traits removed"),
            ("propositions_removed", "prop", "Propositions removed"),
            ("concerns_removed", "ccn", "Concerns removed"),
        ],
    ),
    (
        "Superseded", "text-amber-700 bg-amber-50 border-amber-200",
        [
            ("events_superseded", "evt", "Events superseded"),
        ],
    ),
]


def _iter_nonzero(
    summary: dict, items: Iterable[Tuple[str, str, str]],
) -> list[Tuple[str, str, str, int]]:
    out: list[Tuple[str, str, str, int]] = []
    for key, label, tip in items:
        v = int(summary.get(key, 0) or 0)
        if v:
            out.append((key, label, tip, v))
    return out


def changeset_summary_total(summary: Optional[dict]) -> int:
    """Sum of all counters; useful for "no changes" detection."""
    if not summary:
        return 0
    total = 0
    for _, _, items in _CHIP_GROUPS:
        for key, _, _ in items:
            total += int(summary.get(key, 0) or 0)
    return total


def render_changeset_chips(
    summary: Optional[dict],
    *,
    compact: bool = False,
    empty_label: Optional[str] = "No structural changes",
) -> None:
    """Render grouped chips for a normalised changeset summary dict.

    ``summary`` is the dict returned by
    ``shadow_loom.db.get_version_tree`` per version, or any equivalent
    flattened ``MergeChangeset`` payload. ``None`` / empty / all-zero
    summaries render the ``empty_label`` (or nothing if ``empty_label``
    is None).

    ``compact=True`` renders small inline chips suitable for per-card
    badges. ``compact=False`` renders full grouped cards.
    """
    if changeset_summary_total(summary) == 0:
        if empty_label:
            ui.label(empty_label).classes("text-xs italic text-slate-400")
        return

    assert summary is not None  # for type-checkers; total>0 implies non-None

    if compact:
        with ui.row().classes("flex-wrap items-center gap-1"):
            for _label, color_classes, items in _CHIP_GROUPS:
                rows = _iter_nonzero(summary, items)
                for key, short, tip, v in rows:
                    ui.label(f"+{v} {short}" if "removed" not in key
                             and "superseded" not in key
                             else (f"−{v} {short}" if "removed" in key
                                   else f"⤳{v} {short}")).classes(
                        f"text-[10px] px-1.5 py-0.5 rounded border {color_classes}"
                    ).tooltip(f"{tip}: {v}")
        return

    with ui.column().classes("w-full gap-1"):
        for label, color_classes, items in _CHIP_GROUPS:
            rows = _iter_nonzero(summary, items)
            if not rows:
                continue
            with ui.row().classes("w-full items-center gap-1 flex-wrap"):
                ui.label(label).classes(
                    "text-[11px] uppercase tracking-wide text-slate-500 w-20"
                )
                for key, short, tip, v in rows:
                    sign = (
                        "−" if "removed" in key
                        else ("⤳" if "superseded" in key else "+")
                    )
                    ui.label(f"{sign}{v} {short}").classes(
                        f"text-xs px-2 py-0.5 rounded border {color_classes}"
                    ).tooltip(f"{tip}: {v}")
