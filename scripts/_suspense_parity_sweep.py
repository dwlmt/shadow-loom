"""Cross-validate unified suspense vs legacy EFK suspense across all 20 example worlds.

Affect Unification, Step 5 validation. For each world we sample suspense at five
syuzhet anchors (early/quartile/mid/three-quarters/late) under both
``mode='efk'`` (legacy aggregator) and ``mode='unified'`` (entropy-of-open-
outcomes over ``BeliefState[ENT_AUDIENCE]``). The contract is that the unified
substrate stays close to the legacy curve in shape (correlation) and absolute
level (mean abs delta) on heavily-foreshadowed plots — and is allowed to
diverge on plots that deliberately blindside (the unified scorer is the
better-grounded one of the two; the parity check is to confirm we have not
accidentally inverted the sign or destroyed dynamic range).

Usage::

    conda activate shadow-loom && python scripts/_suspense_parity_sweep.py
"""

from __future__ import annotations

import importlib
import math
import statistics
from pathlib import Path
from typing import List, Tuple

from shadow_loom.directive_assembly import DirectiveAssembler

WORLDS_DIR = Path(__file__).resolve().parent.parent / "example_worlds"
ANCHOR_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)


def _world_modules() -> List[str]:
    names = []
    for f in sorted(WORLDS_DIR.glob("*.py")):
        if f.stem.startswith("_"):
            continue
        names.append(f"example_worlds.{f.stem}")
    return names


def _focal(world) -> str:
    """Pick a representative focal entity (first POV-ish character).

    Prefer entities that appear in the most events; fall back to the first
    entity id if no events.
    """
    if not world.events:
        return next(iter(world.entities))
    counts: dict[str, int] = {}
    for evt in world.events:
        for eid in (evt.actor_ids or []) + (evt.target_ids or []):
            counts[eid] = counts.get(eid, 0) + 1
    if not counts:
        return next(iter(world.entities))
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _anchors(world) -> List[int]:
    n = len(world.events)
    if n == 0:
        return []
    out = []
    for f in ANCHOR_FRACTIONS:
        idx = max(0, min(n - 1, int(round(f * (n - 1)))))
        out.append(idx)
    # de-dup while preserving order
    seen = set()
    return [a for a in out if not (a in seen or seen.add(a))]


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main() -> int:
    rows: List[Tuple[str, int, float, float, float, float]] = []
    failures: List[str] = []

    for modname in _world_modules():
        try:
            mod = importlib.import_module(modname)
            world = mod.world_state
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{modname}: import failed: {exc}")
            continue

        focal = _focal(world)
        anchors = _anchors(world)
        if not anchors:
            failures.append(f"{modname}: no events, skipped")
            continue

        legacy_vals: List[float] = []
        unified_vals: List[float] = []

        da = DirectiveAssembler(None, {}, world)

        for a in anchors:
            try:
                legacy = da.compute_suspense_score([focal], syuzhet_anchor=a, mode="efk")
            except Exception as exc:  # noqa: BLE001
                legacy = float("nan")
                failures.append(f"{modname}@{a} legacy: {exc}")

            try:
                unified = da.compute_suspense_score([focal], syuzhet_anchor=a, mode="unified")
            except Exception as exc:  # noqa: BLE001
                unified = float("nan")
                failures.append(f"{modname}@{a} unified: {exc}")

            legacy_vals.append(legacy)
            unified_vals.append(unified)

        # Filter NaNs for stats.
        pairs = [(l, u) for l, u in zip(legacy_vals, unified_vals)
                 if not (math.isnan(l) or math.isnan(u))]
        if not pairs:
            failures.append(f"{modname}: all anchors NaN")
            continue
        ls, us = zip(*pairs)
        mean_abs_delta = statistics.mean(abs(l - u) for l, u in pairs)
        rho = _pearson(list(ls), list(us))
        rows.append((
            modname.split(".")[-1],
            len(world.events),
            statistics.mean(ls),
            statistics.mean(us),
            mean_abs_delta,
            rho,
        ))

    # Print table.
    print()
    print(f"{'world':<36} {'#evts':>5} {'⟨legacy⟩':>9} {'⟨unified⟩':>10} {'mean|Δ|':>8} {'ρ(L,U)':>7}")
    print("-" * 80)
    for name, n, ml, mu, dlt, rho in sorted(rows):
        rho_s = f"{rho:+.3f}" if not math.isnan(rho) else "  n/a"
        print(f"{name:<36} {n:>5} {ml:>9.4f} {mu:>10.4f} {dlt:>8.4f} {rho_s:>7}")
    print("-" * 80)
    print(f"worlds OK: {len(rows)}/{len(_world_modules())}")
    if rows:
        print(f"global mean |Δ|: {statistics.mean(r[4] for r in rows):.4f}")
        rhos = [r[5] for r in rows if not math.isnan(r[5])]
        if rhos:
            print(f"global mean ρ:   {statistics.mean(rhos):+.3f}")
            print(f"#worlds ρ ≥ 0:   {sum(1 for r in rhos if r >= 0)}/{len(rhos)}")

    if failures:
        print()
        print(f"failures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
