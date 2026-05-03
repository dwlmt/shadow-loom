# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply JSON patch plans to example_worlds/<name>.py fixtures.

Reads patch files from ``/tmp/fixture_patches/<name>.json`` produced
by per-fixture LLM planning passes (one Explore subagent per fixture).
Each patch has two lists:

* ``add_edges``: dicts of CausalEdge kwargs (mutation_social) to insert
  into ``ws.causal_topology``.
* ``flip_to_unobserved``: dicts ``{source_entity_id, target_entity_id, axis}``
  identifying ``RelationshipMetric`` instances to mark ``observed=False``.

Strategy: textual surgery on the fixture .py file, NOT load-and-rewrite
(which would lose comments, formatting, ordering, and human-curated
inertia bands). The fixtures follow a stable structure:

* ``causal_topology=[`` ... ``]`` — closing ``    ],`` line is the
  insertion point. New edges are appended before it as a clearly
  marked block of ``CausalEdge(...)`` lines.
* ``RelationshipEdge(source_entity_id='X', target_entity_id='Y',
  metrics={ "axis": RelationshipMetric(...), ... })`` — locate the
  axis line within the named edge and insert ``observed=False`` if not
  already present (or replace ``observed=True`` if explicit).

Idempotent: re-running the same patch leaves the file unchanged.

Usage:
    PYTHONPATH=. python scripts/apply_axis_patches.py [fixture_name ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORLDS = REPO / "example_worlds"
PATCHES = Path("/tmp/fixture_patches")

# Sentinel so we can detect already-applied patches and skip re-emit.
SENTINEL = "# ─── auto-patched mutation_social edges (per-axis coverage) ───"


def _format_edge(e: dict) -> str:
    """Render a single ``CausalEdge`` constructor call for a
    mutation_social patch. Conservative formatting: one line per call,
    matching the existing style in macbeth.py / wanda.py.
    """
    return (
        f'        CausalEdge(source_id="{e["source_id"]}", '
        f'target_id="{e["target_id"]}", '
        f'rel_counterpart_id="{e["rel_counterpart_id"]}", '
        f'causality_type="mutation_social", '
        f'trait_target="{e["trait_target"]}", '
        f'trait_delta={float(e["trait_delta"])}, '
        f'mechanism="{e["mechanism"]}", '
        f'evidence_strength="{e["evidence_strength"]}", '
        f'causal_force={float(e["causal_force"])}, '
        f'fabula_time={int(e["fabula_time"])}, '
        f'propagation_delay={int(e.get("propagation_delay", 0))}),'
    )


def _insert_edges(src: str, edges: list[dict]) -> tuple[str, int]:
    """Insert formatted ``CausalEdge`` lines at the closing ``],`` of
    the ``causal_topology=[`` block. Returns (new_src, n_inserted).

    If the SENTINEL is already present, the existing auto-patched
    block is removed before re-insertion (idempotent re-run).
    """
    if not edges:
        return src, 0

    # 1. Strip any prior auto-patched block so re-runs don't accumulate.
    stripped = re.sub(
        rf"\n        {re.escape(SENTINEL)}\n(?:        CausalEdge\(.*?\),\n)+",
        "\n",
        src,
        flags=re.DOTALL,
    )

    # 2. Find the closing ``    ],`` of causal_topology=[
    open_pat = re.compile(r"^    causal_topology=\[\s*$", re.MULTILINE)
    m_open = open_pat.search(stripped)
    if not m_open:
        raise RuntimeError("could not locate causal_topology=[ block")

    # Find the matching ``    ],`` after open. Walk forward looking
    # for the first line that is exactly ``    ],`` (4 spaces).
    after = stripped[m_open.end():]
    close_pat = re.compile(r"^    \],?\s*$", re.MULTILINE)
    m_close = close_pat.search(after)
    if not m_close:
        raise RuntimeError("could not locate closing ], of causal_topology")

    insert_pos = m_open.end() + m_close.start()

    block = (
        f"\n        {SENTINEL}\n"
        + "\n".join(_format_edge(e) for e in edges)
        + "\n"
    )
    new_src = stripped[:insert_pos] + block + stripped[insert_pos:]
    return new_src, len(edges)


def _flip_unobserved(src: str, flips: list[dict]) -> tuple[str, int]:
    """Mutate ``RelationshipMetric(...)`` constructor kwargs so the
    axis identified by ``(source_entity_id, target_entity_id, axis)``
    has ``observed=False``. Idempotent.

    Surgery is scoped to one ``RelationshipEdge(...)`` block at a time,
    located by the (source, target) pair, so we don't mis-target a
    metric on a different dyad.
    """
    if not flips:
        return src, 0

    n = 0
    for flip in flips:
        s, t, axis = (
            flip["source_entity_id"],
            flip["target_entity_id"],
            flip["axis"],
        )
        # Locate the RelationshipEdge block. The non-greedy ``.*?\)``
        # would match the first ``)`` (a RelationshipMetric's closing
        # paren), truncating the block before later axes. Anchor on
        # the conventional closing line ``\n        ),`` instead — all
        # 16 example_worlds emit RelationshipEdge with that format.
        pat = re.compile(
            rf"RelationshipEdge\(\s*"
            rf"source_entity_id=['\"]{re.escape(s)}['\"],\s*"
            rf"target_entity_id=['\"]{re.escape(t)}['\"],"
            rf".*?\n        \),",
            re.DOTALL,
        )
        m = pat.search(src)
        if not m:
            print(f"  WARN: could not locate {s} → {t} edge; skipping flip")
            continue
        block = m.group(0)

        # Find the metric line: `"axis": RelationshipMetric(...)`
        metric_pat = re.compile(
            rf'(["\']){re.escape(axis)}\1\s*:\s*RelationshipMetric\(([^)]*)\)',
        )
        mm = metric_pat.search(block)
        if not mm:
            print(f"  WARN: {s} → {t} has no {axis!r} metric; skipping flip")
            continue
        kwargs_str = mm.group(2)

        if re.search(r"observed\s*=\s*False", kwargs_str):
            # Already flipped.
            continue
        if re.search(r"observed\s*=\s*True", kwargs_str):
            new_kwargs = re.sub(r"observed\s*=\s*True", "observed=False", kwargs_str)
        else:
            # Default observed=True implicit; append observed=False.
            new_kwargs = kwargs_str.rstrip() + ", observed=False"

        # Also zero the value so the test's ``abs(value) <= 1e-6`` arm
        # treats this as a true unobserved baseline (defensive — the
        # ``observed=False`` arm alone is sufficient).
        new_kwargs = re.sub(
            r"value\s*=\s*-?\d+(\.\d+)?",
            "value=0.0",
            new_kwargs,
            count=1,
        )

        new_metric = (
            f'{mm.group(1)}{axis}{mm.group(1)}: '
            f'RelationshipMetric({new_kwargs})'
        )
        new_block = block[: mm.start()] + new_metric + block[mm.end():]
        src = src[: m.start()] + new_block + src[m.end():]
        n += 1
    return src, n


def apply_patch(name: str) -> tuple[int, int]:
    fixture_path = WORLDS / f"{name}.py"
    patch_path = PATCHES / f"{name}.json"
    if not fixture_path.exists():
        print(f"SKIP {name}: no fixture file")
        return 0, 0
    if not patch_path.exists():
        print(f"SKIP {name}: no patch file at {patch_path}")
        return 0, 0

    patch = json.loads(patch_path.read_text())
    src = fixture_path.read_text()

    src, n_edges = _insert_edges(src, patch.get("add_edges", []))
    src, n_flips = _flip_unobserved(src, patch.get("flip_to_unobserved", []))

    fixture_path.write_text(src)
    print(f"PATCH {name}: +{n_edges} mutation_social edges, ~{n_flips} observed=False flips")
    return n_edges, n_flips


def main(argv: list[str]) -> int:
    if argv:
        names = argv
    else:
        names = sorted(p.stem for p in PATCHES.glob("*.json"))
    total_e = total_f = 0
    for name in names:
        e, f = apply_patch(name)
        total_e += e
        total_f += f
    print(f"\nTotal: +{total_e} edges, ~{total_f} flips across {len(names)} fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
