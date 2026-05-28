"""Audit referent linkage on example worlds.

Walks every ``example_worlds/*.py`` module, loads its ``build_world()``
``WorldStateV1``, and reports propositions / beliefs that lack the
event-id linkage required for shadow-branch counterfactual surgery to
prune them correctly:

  1. Propositions whose ``truth_at_fabula`` has 2+ entries (i.e. the
     truth value flips during the fabula) but whose ``referent_ids``
     contains no ``EVT_`` id at all. Without an event referent the
     shadow-suppression sweep cannot decide which tick to roll back to
     when the originating event is intervened away.
  2. Propositions whose ``truth_at_fabula`` ticks are not within
     +/- delta of *any* of the propositions referenced events'
     ``fabula_time``. The mismatch usually indicates a stale tick or a
     missing referent.
  3. Beliefs whose ``proposition_id`` points at a Proposition that
     itself lacks event referents (cascading the above gap onto every
     belief that hangs off it).

Run::

    python -m scripts._audit_referent_linkage

Output is a single Markdown report on stdout. Exit code is 0 even when
findings exist so the script can be wired into CI as a non-blocking
quality gate.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from collections import defaultdict
from typing import Iterable

import example_worlds
from shadow_loom.models import EventNode, Proposition, WorldStateV1


# R19-M15: branch-aware audit hook. Set
# ``SHADOW_LOOM_AUDIT_BRANCH=<label>`` to audit a shadow branch.
def _branch_projected(ws: WorldStateV1) -> WorldStateV1:
    label = os.environ.get("SHADOW_LOOM_AUDIT_BRANCH")
    if not label:
        return ws
    try:
        return ws.projected_for_branch("shadow", label)
    except Exception:
        return ws


def _iter_world_modules() -> Iterable[tuple[str, object]]:
    """Yield ``(module_name, module)`` for every world file."""
    pkg = example_worlds
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        yield info.name, importlib.import_module(f"example_worlds.{info.name}")


def _build_world(module) -> WorldStateV1 | None:
    # Worlds typically expose a module-level ``world_state`` constant;
    # fall back to ``build_world()`` for any future world that uses a
    # builder function instead.
    ws = getattr(module, "world_state", None)
    if isinstance(ws, WorldStateV1):
        return ws
    fn = getattr(module, "build_world", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! build_world() raised {type(exc).__name__}: {exc}")
        return None


def _audit_world(name: str, ws: WorldStateV1) -> list[str]:
    ws = _branch_projected(ws)
    findings: list[str] = []
    event_index: dict[str, EventNode] = {e.id: e for e in ws.events}

    # 1 + 2: proposition-level linkage
    for prop in ws.propositions:
        ticks = sorted((prop.truth_at_fabula or {}).keys())
        event_refs = [r for r in (prop.referent_ids or []) if r.startswith("EVT_")]

        if len(ticks) >= 2 and not event_refs:
            findings.append(
                f"- **{prop.proposition_id}** has {len(ticks)} truth ticks "
                f"({ticks}) but no `EVT_` referent. Shadow-suppression "
                f"cannot prune this proposition when its causing event is "
                f"intervened away."
            )

        if event_refs:
            ref_times = [
                event_index[r].fabula_time
                for r in event_refs
                if r in event_index
            ]
            if ref_times:
                window = (min(ref_times) - 1000, max(ref_times) + 1000)
                for tick in ticks:
                    # ``fabula_time == 0`` (and small <=500 anchors) are
                    # conventional baseline priors, not event-driven
                    # ticks — skip them to keep the signal narrow.
                    if tick <= 500:
                        continue
                    if not (window[0] <= tick <= window[1]):
                        findings.append(
                            f"- **{prop.proposition_id}** truth tick "
                            f"`{tick}` is outside the +/- 1000 window of "
                            f"its event referents (events at {ref_times}). "
                            f"Likely a stale tick or missing referent."
                        )

    # 3: belief → proposition cascade
    bad_props = {
        p.proposition_id for p in ws.propositions
        if not [r for r in (p.referent_ids or []) if r.startswith("EVT_")]
        and len(p.truth_at_fabula or {}) >= 2
    }
    if bad_props:
        beliefs_hit: dict[str, list[str]] = defaultdict(list)
        for ent in ws.entities.values():
            for b in ent.beliefs:
                if b.proposition_id and b.proposition_id in bad_props:
                    beliefs_hit[b.proposition_id].append(ent.id)
        for pid, holders in beliefs_hit.items():
            findings.append(
                f"- **{pid}** is referenced by belief on "
                f"{sorted(set(holders))} but the proposition itself "
                f"lacks event referents (see above)."
            )

    return findings


def main() -> int:
    print("# Referent linkage audit\n")
    total = 0
    for name, mod in _iter_world_modules():
        ws = _build_world(mod)
        if ws is None:
            continue
        findings = _audit_world(name, ws)
        if findings:
            print(f"## `{name}` — {len(findings)} finding(s)\n")
            for line in findings:
                print(line)
            print()
            total += len(findings)
    print(f"---\n**Total findings: {total}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
