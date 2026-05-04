# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit script: list every observed-but-static (axis, dyad) pair per
fixture, alongside the candidate events that involve both entities.

For each fixture in ``example_worlds/`` and each axis in
``("affinity", "fear", "power_dynamic")``, print the dyads that are
declared ``observed=True`` with non-zero magnitude but never mutated
by any ``mutation_social`` causal edge with that ``trait_target``.

Usage:
    PYTHONPATH=. python scripts/audit_static_axes.py [fixture_name]
"""
from __future__ import annotations
import importlib
import pkgutil
import sys
from collections import defaultdict

import example_worlds
from shadow_loom.models import WorldStateV1

AXES = ("affinity", "fear", "power_dynamic")


def _static(ws: WorldStateV1, axis: str):
    observed = []
    for r in ws.social_topology:
        m = r.metrics.get(axis)
        if m is None or not m.observed or abs(m.value) <= 1e-6:
            continue
        observed.append((r.source_entity_id, r.target_entity_id, m.value))
    mutated = set()
    for c in ws.causal_topology:
        if c.causality_type != "mutation_social" or c.trait_target != axis:
            continue
        if c.target_id and c.rel_counterpart_id:
            mutated.add((c.target_id, c.rel_counterpart_id))
    return [(s, t, v) for (s, t, v) in observed if (s, t) not in mutated]


def _events_involving(ws: WorldStateV1, a: str, b: str):
    out = []
    for ev in ws.events:
        actors = set(ev.actor_ids or [])
        targets = set(ev.target_ids or [])
        all_ids = actors | targets
        if a in all_ids and b in all_ids:
            out.append(ev)
        elif a in all_ids and getattr(ev, "addressee_ids", None) and b in (ev.addressee_ids or []):
            out.append(ev)
        elif b in all_ids and getattr(ev, "addressee_ids", None) and a in (ev.addressee_ids or []):
            out.append(ev)
    return out


def main(only: str | None = None) -> int:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"affinity": 0, "fear": 0, "power_dynamic": 0})
    for mod in sorted(pkgutil.iter_modules(example_worlds.__path__), key=lambda m: m.name):
        if mod.name.startswith("_"):
            continue
        if only and mod.name != only:
            continue
        m = importlib.import_module(f"example_worlds.{mod.name}")
        ws = getattr(m, "world_state", None)
        if not isinstance(ws, WorldStateV1):
            continue
        print(f"\n=== {mod.name} ({len(ws.events)} events, {len(ws.social_topology)} rels, {len([c for c in ws.causal_topology if c.causality_type=='mutation_social'])} social mutations) ===")
        for axis in AXES:
            static = _static(ws, axis)
            summary[mod.name][axis] = len(static)
            if not static:
                continue
            print(f"\n  -- {axis}: {len(static)} static dyad(s)")
            for src, tgt, val in static:
                print(f"    {src} → {tgt}  value={val:+.2f}")
                events = _events_involving(ws, src, tgt)
                if not events:
                    print(f"      (no events involve both entities)")
                    continue
                for ev in events[:6]:
                    desc = (ev.description or "")[:90]
                    print(f"      EVT t={ev.fabula_time}: {ev.id} — {desc}")

    if not only:
        print("\n\n=== SUMMARY ===")
        print(f"{'fixture':<35} {'aff':>4} {'fear':>4} {'pwr':>4}")
        for name in sorted(summary):
            s = summary[name]
            print(f"{name:<35} {s['affinity']:>4} {s['fear']:>4} {s['power_dynamic']:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
