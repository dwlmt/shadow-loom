# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audit example_worlds/*.py for events whose ``chain_reaction``
in-degree > 1.

These are candidates for the precondition-mis-tagged-as-sufficient-
cause pattern that caused the Macbeth Duncan-murder bug (2026-05-29):
one parent is the *real* sufficient cause, sibling parents are
*preconditions* that prevent the descendant from being pruned under
the disjunctive Pearl closure (see
:func:`shadow_loom.causal_closure.expand_chain_reaction_closure`).

A multi-parent listing is NOT automatically a bug — many cases are
genuine Halpern-Pearl over-determination (e.g. Aslan's death is
caused by both the pact AND the Deep Magic). The script surfaces
candidates for *human review*, distinguishing:

  * Event sibling parents — most often genuine over-determination
    or chained cause; usually fine.
  * WORLD_* sibling parents — these are usually *standing
    conditions* / ambient pressures, not sufficient causes; if the
    author intended an enabling-precondition relationship, the
    edge should be removed or the world trait re-modelled.

Run:

    python -m scripts._audit_chain_reaction_in_degree
"""
from __future__ import annotations

import importlib
import pathlib
from collections import defaultdict


def main() -> int:
    root = pathlib.Path("example_worlds")
    suspects: list[tuple[str, dict[str, list[tuple[str, str]]]]] = []
    for f in sorted(root.glob("*.py")):
        if f.name.startswith("_"):
            continue
        mod_name = f"example_worlds.{f.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  SKIP {mod_name}: {e.__class__.__name__}: {e}")
            continue
        ws = getattr(mod, "world_state", None)
        if ws is None:
            continue
        parents: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for e in (ws.causal_topology or []):
            if getattr(e, "causality_type", None) != "chain_reaction":
                continue
            parents[e.target_id].append(
                (e.source_id, getattr(e, "mechanism", "") or "")
            )
        multi = {tgt: ps for tgt, ps in parents.items() if len(ps) > 1}
        if multi:
            suspects.append((mod_name, multi))

    if not suspects:
        print("No multi-parent chain_reaction events found.")
        return 0

    total_multi = 0
    total_world_mix = 0
    for mod_name, multi in suspects:
        print(f"\n=== {mod_name} — {len(multi)} multi-parent event(s) ===")
        for tgt, ps in sorted(multi.items()):
            world_count = sum(1 for src, _ in ps if src.startswith("WORLD_"))
            evt_count = sum(1 for src, _ in ps if src.startswith("EVT_"))
            tag = ""
            if world_count and evt_count:
                tag = "  [REVIEW: mixed EVT + WORLD parents]"
                total_world_mix += 1
            print(f"  {tgt}:{tag}")
            for src, mech in ps:
                kind = "WORLD" if src.startswith("WORLD_") else "EVT"
                print(f"    \u2190 [{kind}] {src}  [{mech}]")
            total_multi += 1

    print(
        f"\nSummary: {total_multi} multi-parent chain_reaction "
        f"event(s) across {len(suspects)} fixture(s); "
        f"{total_world_mix} have mixed EVT + WORLD parents (most "
        f"likely candidates for precondition mis-tagging)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
