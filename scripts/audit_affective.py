#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit affective scores across all example_worlds fixtures.

Runs the full structural-affect + emotion scorer suite at 7 evenly-
spaced syuzhet anchors per fixture and prints a min/median/max
summary so we can see whether the four scorers (mystery, irony,
suspense, surprise) and the six emotion scorers (grief/rage/joy/
regret/love/fear) sit on a similar [0, 1] gauge across the corpus.

Usage:
    python scripts/audit_affective.py
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import example_worlds  # noqa: E402
from shadow_loom.directive_assembly import DirectiveAssembler  # noqa: E402
from shadow_loom_ui.viz_helpers import affective_timeseries_syuzhet  # noqa: E402


STRUCTURAL = ["mystery", "dramatic_irony", "suspense", "surprise"]
EMOTIONS = ["grief", "rage", "joy", "regret", "love", "fear"]


def _focal_entity_ids(ws) -> list[str]:
    # Top-N by participation in events (actor or target).
    counts: dict[str, int] = {}
    for ev in ws.events:
        for a in (ev.actor_ids or []):
            counts[a] = counts.get(a, 0) + 1
        for t in (ev.target_ids or []):
            counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [eid for eid, _ in ranked[:6]]


def audit_world(name: str, ws) -> dict[str, list[float]]:
    eids = _focal_entity_ids(ws)
    times, series = affective_timeseries_syuzhet(
        ws, samples=7, entity_ids=eids,
    )

    # Also score emotions across the same anchors using assembler.
    assembler = DirectiveAssembler(None, {}, ws)
    emo_series: dict[str, list[float]] = {emo: [] for emo in EMOTIONS}
    for s in times:
        for emo in EMOTIONS:
            try:
                # compute_affective_score returns *negative* of the
                # closeness for emotion targets; flip back to [0, 1].
                v = -assembler.compute_affective_score(emo, eids, s)
                emo_series[emo].append(max(0.0, min(1.0, v)))
            except Exception:
                emo_series[emo].append(0.0)

    out: dict[str, list[float]] = {}
    for k in STRUCTURAL:
        out[k] = [round(x, 3) for x in series.get(k, [])]
    for k in EMOTIONS:
        out[k] = [round(x, 3) for x in emo_series[k]]
    return out


def main() -> None:
    rows: list[tuple[str, dict[str, list[float]]]] = []
    for mod in sorted(pkgutil.iter_modules(example_worlds.__path__),
                      key=lambda m: m.name):
        if mod.name.startswith("_"):
            continue
        m = importlib.import_module(f"example_worlds.{mod.name}")
        ws = getattr(m, "world_state", None)
        if ws is None:
            continue
        try:
            scores = audit_world(mod.name, ws)
        except Exception as exc:  # noqa: BLE001
            print(f"!! {mod.name}: {exc}")
            continue
        rows.append((mod.name, scores))

    print(f"\n{'='*100}")
    print(f"PER-WORLD CURVES (7 syuzhet anchors)")
    print(f"{'='*100}")
    for name, scores in rows:
        print(f"\n{name}")
        for k in STRUCTURAL + EMOTIONS:
            vals = scores.get(k, [])
            if not vals:
                continue
            print(f"  {k:18s} {vals}")

    print(f"\n{'='*100}")
    print(f"SCALE SUMMARY (per scorer, across {len(rows)} worlds × 7 anchors)")
    print(f"{'='*100}")
    print(f"{'scorer':18s} {'min':>7s} {'median':>7s} {'mean':>7s} {'max':>7s} {'#nonzero/total':>17s}")
    for k in STRUCTURAL + EMOTIONS:
        all_vals = [v for _, s in rows for v in s.get(k, [])]
        if not all_vals:
            continue
        nz = sum(1 for v in all_vals if v > 1e-6)
        print(
            f"{k:18s} {min(all_vals):7.3f} {median(all_vals):7.3f} "
            f"{mean(all_vals):7.3f} {max(all_vals):7.3f} "
            f"{nz:>9d}/{len(all_vals):d}"
        )


if __name__ == "__main__":
    main()
