# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic backfill of ``EventNode.at_location_id`` for an example world.

Loads the named ``example_worlds.<world>`` module, resolves each event's
effective location via ``event_location_at(evt, ws, fallback="actor")``,
and rewrites the source file in place to add an explicit
``at_location_id="LOC_X"`` keyword to every ``EventNode(...)`` constructor
that does not already have one. Uses libcst so formatting / comments / order
of other keywords survive untouched.

Skipped silently:
  * events whose primary actor cannot be located at ``fabula_time``;
  * ``EventNode(...)`` calls that already declare ``at_location_id``.

Usage::

    python scripts/_backfill_event_locations.py brief_encounter
    python scripts/_backfill_event_locations.py --all
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import libcst as cst

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_loom.models import event_location_at  # noqa: E402

WORLDS_DIR = ROOT / "example_worlds"


def resolve_locations(world_name: str) -> dict[str, str]:
    """Return ``{event_id: LOC_id}`` for every event we can deterministically
    place via the primary-actor fallback."""
    mod = importlib.import_module(f"example_worlds.{world_name}")
    ws = mod.world_state
    out: dict[str, str] = {}
    for ev in ws.events or []:
        if getattr(ev, "at_location_id", None):
            continue
        try:
            loc = event_location_at(ev, ws, fallback="actor")
        except Exception:
            loc = None
        if loc and loc in (ws.locations or {}):
            out[ev.id] = loc
    return out


class _EventNodeRewriter(cst.CSTTransformer):
    """Insert ``at_location_id="LOC_X"`` into ``EventNode(...)`` calls.

    Only touches calls whose ``id="EVT_..."`` keyword matches the
    ``locations`` map and which do not already carry ``at_location_id``.
    """

    def __init__(self, locations: dict[str, str]):
        self.locations = locations
        self.applied: set[str] = set()

    def leave_Call(self, original_node, updated_node):
        func = updated_node.func
        # Match ``EventNode(...)``.
        if not (isinstance(func, cst.Name) and func.value == "EventNode"):
            return updated_node
        evt_id: str | None = None
        has_at_loc = False
        insert_after_idx: int | None = None
        for i, arg in enumerate(updated_node.args):
            kw = arg.keyword
            if kw is None:
                continue
            if kw.value == "id":
                v = arg.value
                if isinstance(v, cst.SimpleString):
                    evt_id = v.evaluated_value
            elif kw.value == "at_location_id":
                has_at_loc = True
            elif kw.value == "description":
                insert_after_idx = i
        if has_at_loc or evt_id is None:
            return updated_node
        loc_id = self.locations.get(evt_id)
        if loc_id is None:
            return updated_node
        # Build the new keyword arg.
        new_arg = cst.Arg(
            keyword=cst.Name("at_location_id"),
            equal=cst.AssignEqual(
                whitespace_before=cst.SimpleWhitespace(""),
                whitespace_after=cst.SimpleWhitespace(""),
            ),
            value=cst.SimpleString(f'"{loc_id}"'),
        )
        # Append after description (or at the end). Make sure the
        # preceding arg has a trailing comma.
        new_args = list(updated_node.args)
        target_idx = insert_after_idx if insert_after_idx is not None else len(new_args) - 1
        # Ensure target arg has a comma.
        target = new_args[target_idx]
        if target.comma is cst.MaybeSentinel.DEFAULT or target.comma is None:
            target = target.with_changes(
                comma=cst.Comma(
                    whitespace_after=cst.SimpleWhitespace(" "),
                ),
            )
            new_args[target_idx] = target
        new_args.insert(target_idx + 1, new_arg)
        self.applied.add(evt_id)
        return updated_node.with_changes(args=new_args)


def backfill_world(world_name: str, *, dry_run: bool = False) -> tuple[int, int, set[str]]:
    """Returns ``(applied_count, candidate_count, applied_event_ids)``."""
    locations = resolve_locations(world_name)
    if not locations:
        return 0, 0, set()
    src_path = WORLDS_DIR / f"{world_name}.py"
    src = src_path.read_text(encoding="utf-8")
    module = cst.parse_module(src)
    rewriter = _EventNodeRewriter(locations)
    new_module = module.visit(rewriter)
    if not dry_run and rewriter.applied:
        src_path.write_text(new_module.code, encoding="utf-8")
    return len(rewriter.applied), len(locations), rewriter.applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worlds", nargs="*", help="World names to backfill.")
    ap.add_argument("--all", action="store_true",
                    help="Backfill every example world.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute patches without writing.")
    args = ap.parse_args()

    if args.all:
        worlds = sorted(
            p.stem for p in WORLDS_DIR.glob("*.py")
            if p.stem != "__init__"
        )
    else:
        worlds = list(args.worlds)
    if not worlds:
        ap.error("Specify at least one world or pass --all.")

    grand_applied = 0
    for w in worlds:
        try:
            applied, total, ids = backfill_world(w, dry_run=args.dry_run)
        except Exception as e:
            print(f"{w}: ERROR — {e}", file=sys.stderr)
            continue
        grand_applied += applied
        verb = "would patch" if args.dry_run else "patched"
        print(f"{w}: {verb} {applied}/{total} events")
        if applied and applied <= 6:
            for eid in sorted(ids):
                print(f"  - {eid}")
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"\nTotal events {('would be ' if args.dry_run else '')}backfilled: {grand_applied}{suffix}")


if __name__ == "__main__":
    main()
