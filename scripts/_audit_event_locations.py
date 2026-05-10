# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage report for ``EventNode.at_location_id`` across all example worlds.

For each fixture, prints:
  * total event count
  * how many carry an explicit ``at_location_id``
  * how many would resolve via the primary-actor fallback
  * how many would remain unresolvable (no actor located at fabula_time)
  * a list of unresolvable event ids (truncated)

Used as the success metric for PRs 10-11 of the EventNode.at_location_id
backfill: the goal is 100% explicit coverage on the bundled fixtures so
the auditor's spatial co-presence rules and the Map sub-tab's event
glyphs have ground truth to compare against without relying on the
fallback.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_loom.models import event_location_at  # noqa: E402

WORLDS = [
    "a_court_of_thorn_and_roses", "a_fish_called_wanda", "apocalypse_now",
    "brief_encounter", "dads_army", "death_on_the_nile", "frankenstein",
    "gone_girl", "great_expectations", "great_gatsby", "macbeth",
    "nineteen_eighty_four", "once_upon_a_time_in_the_west", "persuasion",
    "reservoir_dogs", "romeo_and_juliet", "the_devil_wears_prada",
    "the_lion_the_witch_and_the_wardrobe", "tinker_tailor_soldier_spy",
    "wuthering_heights",
]


def audit(name: str) -> dict:
    mod = importlib.import_module(f"example_worlds.{name}")
    ws = mod.world_state
    events = list(ws.events or [])
    explicit = 0
    fallback = 0
    unresolved: list[str] = []
    for ev in events:
        if getattr(ev, "at_location_id", None):
            explicit += 1
            continue
        try:
            resolved = event_location_at(ev, ws, fallback="actor")
        except Exception:
            resolved = None
        if resolved:
            fallback += 1
        else:
            unresolved.append(ev.id)
    total = len(events)
    return {
        "name": name, "total": total, "explicit": explicit,
        "fallback": fallback, "unresolved": unresolved,
    }


def main():
    rows = [audit(w) for w in WORLDS]
    width = max(len(r["name"]) for r in rows)
    print(
        f"{'world'.ljust(width)}  {'total':>5}  {'explicit':>8}  "
        f"{'fallback':>8}  {'unresolved':>10}  pct_explicit"
    )
    print("-" * (width + 60))
    for r in rows:
        pct = (100.0 * r["explicit"] / r["total"]) if r["total"] else 0.0
        print(
            f"{r['name'].ljust(width)}  {r['total']:>5}  "
            f"{r['explicit']:>8}  {r['fallback']:>8}  "
            f"{len(r['unresolved']):>10}  {pct:>11.1f}%"
        )
    totals = {
        "total": sum(r["total"] for r in rows),
        "explicit": sum(r["explicit"] for r in rows),
        "fallback": sum(r["fallback"] for r in rows),
        "unresolved": sum(len(r["unresolved"]) for r in rows),
    }
    pct = (100.0 * totals["explicit"] / totals["total"]) if totals["total"] else 0.0
    print("-" * (width + 60))
    print(
        f"{'TOTAL'.ljust(width)}  {totals['total']:>5}  "
        f"{totals['explicit']:>8}  {totals['fallback']:>8}  "
        f"{totals['unresolved']:>10}  {pct:>11.1f}%"
    )
    # Detail block: any world with unresolved events.
    for r in rows:
        if r["unresolved"]:
            print(f"\n{r['name']}: unresolved event ids ({len(r['unresolved'])}):")
            for eid in r["unresolved"][:20]:
                print(f"  - {eid}")
            if len(r["unresolved"]) > 20:
                print(f"  ... ({len(r['unresolved']) - 20} more)")


if __name__ == "__main__":
    main()
