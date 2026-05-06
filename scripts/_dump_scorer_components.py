#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Print real per-component scorer values for the paper's worked examples."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from example_worlds import (  # noqa: E402
    macbeth, death_on_the_nile, romeo_and_juliet, reservoir_dogs,
    gone_girl, wuthering_heights, brief_encounter, tinker_tailor_soldier_spy,
)
from shadow_loom.directive_assembly import DirectiveAssembler  # noqa: E402
from shadow_loom_ui.viz_helpers import (  # noqa: E402
    affective_timeseries_syuzhet, syuzhet_time_bounds,
)


def _focal(ws, n=6):
    counts = {}
    for ev in ws.events:
        for a in (ev.actor_ids or []):
            counts[a] = counts.get(a, 0) + 1
        for t in (ev.target_ids or []):
            counts[t] = counts.get(t, 0) + 1
    return [eid for eid, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def _anchor_curve(ws, name):
    eids = _focal(ws)
    times, series = affective_timeseries_syuzhet(ws, samples=9, entity_ids=eids)
    print(f"\n=== {name} (anchors {times}) ===")
    for k in ["mystery", "dramatic_irony", "suspense", "surprise"]:
        v = series.get(k, [])
        print(f"  {k:16s}: {v}  peak={max(v) if v else 0:.3f} at idx {v.index(max(v)) if v else -1}")
    print(f"  focal entities: {eids}")
    return times, series, eids


def _emotion_per_entity(ws, name, anchor=None):
    eids = _focal(ws)
    smin, smax = syuzhet_time_bounds(ws)
    s = anchor if anchor is not None else smax
    a = DirectiveAssembler(None, {}, ws)
    print(f"\n--- {name} emotion targets at syuzhet={s} (per entity) ---")
    for eid in eids[:4]:
        row = []
        for emo in ["grief", "rage", "joy", "regret", "love", "fear"]:
            try:
                v = -a.compute_affective_score(emo, [eid], s)
            except Exception:
                v = 0.0
            row.append(f"{emo}={v:+.2f}")
        print(f"  {eid:30s} " + "  ".join(row))


for ws, name in [
    (macbeth.world_state, "Macbeth"),
    (death_on_the_nile.world_state, "Death on the Nile"),
    (romeo_and_juliet.world_state, "Romeo and Juliet"),
    (reservoir_dogs.world_state, "Reservoir Dogs"),
    (gone_girl.world_state, "Gone Girl"),
    (wuthering_heights.world_state, "Wuthering Heights"),
    (tinker_tailor_soldier_spy.world_state, "Tinker Tailor Soldier Spy"),
]:
    _anchor_curve(ws, name)

for ws, name in [
    (macbeth.world_state, "Macbeth"),
    (gone_girl.world_state, "Gone Girl"),
    (brief_encounter.world_state, "Brief Encounter"),
    (wuthering_heights.world_state, "Wuthering Heights"),
]:
    _emotion_per_entity(ws, name)
