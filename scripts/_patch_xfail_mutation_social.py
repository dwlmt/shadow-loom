"""One-shot patcher: backfill missing per-axis mutation_social edges.

Loads each fixture in ``example_worlds/``, computes which observed
``RelationshipMetric`` axes lack a corresponding ``mutation_social``
``CausalEdge``, and appends synthesised edges into the fixture's
source file just before the ``causal_topology=[`` closing bracket.

Each synthesised edge:
  * anchors at an event involving the dyad's source or target
    (preferring an event whose actors include the source);
  * uses ``trait_delta`` proportional to the observed equilibrium
    (≈30% of the value, sign-preserved; ``fear`` is non-negative);
  * carries a ``# auto-backfill`` comment so the seeded data is
    self-documenting in source.

This script is destructive and idempotent in spirit: a second run
will see no missing dyads and emit no patches.
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import example_worlds
from shadow_loom.models import WorldStateV1


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example_worlds"
AXES = ("affinity", "fear", "power_dynamic")


def _missing_dyads(ws: WorldStateV1, axis: str):
    """Return list of (src, tgt, observed_value) dyads needing an edge."""
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

    return [
        (src, tgt, val) for (src, tgt, val) in observed
        if (src, tgt) not in mutated
    ]


def _pick_event(ws: WorldStateV1, src: str, tgt: str):
    """Pick a sensible event to anchor the mutation_social edge to.

    Preference order:
      1. event where ``src`` is in actor_ids and ``tgt`` is in target_ids
      2. event where ``src`` is in actor_ids OR speaker_id (any target)
      3. event involving either entity at all
      4. earliest event in the world
    """
    cand_pairs = []
    cand_actor = []
    cand_either = []
    for e in ws.events:
        actors = set(e.actor_ids or [])
        targets = set(e.target_ids or [])
        speaker = getattr(e, "speaker_id", None)
        addressees = set(getattr(e, "addressee_ids", []) or [])
        involves_src = src in actors or speaker == src
        involves_tgt = tgt in targets or tgt in addressees or tgt in actors
        if involves_src and involves_tgt:
            cand_pairs.append(e)
        elif involves_src:
            cand_actor.append(e)
        elif involves_src or involves_tgt:
            cand_either.append(e)
    pool = cand_pairs or cand_actor or cand_either or list(ws.events)
    if not pool:
        return None
    pool.sort(key=lambda e: e.fabula_time or 0)
    return pool[0]


def _synth_delta(axis: str, observed: float) -> float:
    """Synthetic delta: 30% of equilibrium, clipped, sign preserved."""
    if axis == "fear":
        return round(min(0.5, max(0.05, abs(observed) * 0.3)), 2)
    sign = 1.0 if observed >= 0 else -1.0
    return round(sign * min(0.5, max(0.05, abs(observed) * 0.3)), 2)


def _mechanism_for(axis: str) -> str:
    return {
        "affinity": "emotional",
        "fear": "psychological",
        "power_dynamic": "social",
    }[axis]


def _format_edge(axis: str, evt_id: str, src: str, tgt: str,
                 delta: float, fabula_time: int) -> str:
    return (
        f"        CausalEdge(source_id=\"{evt_id}\", target_id=\"{src}\", "
        f"rel_counterpart_id=\"{tgt}\",  # auto-backfill\n"
        f"                   causality_type=\"mutation_social\", "
        f"trait_target=\"{axis}\", trait_delta={delta},\n"
        f"                   mechanism=\"{_mechanism_for(axis)}\", "
        f"evidence_strength=\"moderate\", causal_force=4.0, "
        f"fabula_time={fabula_time}, propagation_delay=0),\n"
    )


# Match the closing ``],`` of the ``causal_topology=[ ... ],`` block.
_CAUSAL_CLOSE_RE = re.compile(
    r"(causal_topology\s*=\s*\[.*?)(^\s*\],\s*$)",
    re.DOTALL | re.MULTILINE,
)


def _patch_file(path: Path, new_lines: list[str]) -> bool:
    src = path.read_text()
    m = _CAUSAL_CLOSE_RE.search(src)
    if not m:
        print(f"  [skip] {path.name}: causal_topology block not found")
        return False
    insert = "        # ── auto-backfilled per-axis mutation_social ──\n" + "".join(new_lines)
    new_src = src[:m.start(2)] + insert + src[m.start(2):]
    path.write_text(new_src)
    return True


def main():
    total = 0
    for mod in pkgutil.iter_modules(example_worlds.__path__):
        if mod.name.startswith("_"):
            continue
        m = importlib.import_module(f"example_worlds.{mod.name}")
        ws = getattr(m, "world_state", None)
        if not isinstance(ws, WorldStateV1):
            continue
        new_lines = []
        for axis in AXES:
            for src, tgt, val in _missing_dyads(ws, axis):
                evt = _pick_event(ws, src, tgt)
                if evt is None:
                    continue
                delta = _synth_delta(axis, val)
                new_lines.append(_format_edge(
                    axis, evt.id, src, tgt, delta,
                    evt.fabula_time or 0,
                ))
        if not new_lines:
            continue
        path = EXAMPLE_DIR / f"{mod.name}.py"
        ok = _patch_file(path, new_lines)
        if ok:
            print(f"  [patched] {path.name}: +{len(new_lines)} edges")
            total += len(new_lines)
    print(f"\nTotal edges added: {total}")


if __name__ == "__main__":
    main()
