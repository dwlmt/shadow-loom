# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ad-hoc validation of P0-5 minimal per-axis do-mutilation + inertia-gate
behaviour across the example worlds NOT covered by _verify_pearl_ctf.py.

For each world we:
  A. EFFECTIVE do() — clamp an existing trait axis to a value far from its
     current value (|shift| > inertia so the gate fires). Assert G_{\bar X}:
     after surgery no incoming causal edge targets that axis or is untyped,
     while edges that explicitly target a *different* axis survive.
  B. PHANTOM/absent-axis do(=1.0) — materialise an axis that does not exist
     and clamp it to 1.0 (shift 1.0 > default inertia → fires). Assert the
     axis is materialised and untyped in-edges are severed.
  C. NO-OP do(=0.0) on the same absent axis — shift 0.0 ≤ inertia, so the
     gate blocks it: assert NO causal in-edges are severed.

NOT a unit test — investigative validation printed as a report.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.disable(logging.CRITICAL)

from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import (
    AMWNInstantiator,
    _entity_trait_inertia_default,
    _inertia_epsilon,
)

import importlib

EXTRA_WORLDS = [
    "a_court_of_thorn_and_roses", "a_fish_called_wanda", "apocalypse_now",
    "brief_encounter", "dads_army", "death_on_the_nile", "great_expectations",
    "great_gatsby", "once_upon_a_time_in_the_west", "persuasion",
    "reservoir_dogs", "the_devil_wears_prada",
    "the_lion_the_witch_and_the_wardrobe", "wuthering_heights",
]

PHANTOM = "zzz_phantom_axis"
PASS = FAIL = 0


def chk(world, label, cond, detail=""):
    global PASS, FAIL
    PASS, FAIL = (PASS + (1 if cond else 0), FAIL + (0 if cond else 1))
    tag = "PASS" if cond else "FAIL"
    sfx = f"  [{detail}]" if detail else ""
    print(f"  [{tag}] {world:30s} {label}{sfx}")


def in_causal_axis_edges(g, nid, axis):
    return [
        (u, v) for u, v, d in g.in_edges(nid, data=True)
        if d.get("edge_type") == "causal" and d.get("trait_target") in (None, axis)
    ]


def all_in_causal(g, nid):
    return [
        (u, v) for u, v, d in g.in_edges(nid, data=True)
        if d.get("edge_type") == "causal"
    ]


def pick_subject(ws):
    """Pick an entity + an existing trait axis whose value gives the
    largest possible |shift| to a 0/1 clamp (maximising the chance the
    inertia gate fires), preferring entities that also have a second axis
    so the 'other-axis preserved' sub-assertion is exercised."""
    best = None
    for eid, ent in ws.entities.items():
        traits = ent.traits or {}
        if not traits:
            continue
        for axis, tv in traits.items():
            val = float(getattr(tv, "value", tv))
            inertia = float(getattr(tv, "inertia", 0.5))
            target = 1.0 if val < 0.5 else 0.0
            shift = abs(target - val)
            if shift <= inertia + _inertia_epsilon():
                continue  # would be blocked; not an effective clamp
            score = (shift, len(traits))
            if best is None or score > best[0]:
                best = (score, eid, axis, val, inertia, target)
    if best is None:
        return None
    _, eid, axis, val, inertia, target = best
    return eid, axis, val, inertia, target


def sandbox_for(ws, focus):
    ego = extract_ego_graph_from_memory(ws, focus_entity_ids=focus)
    return AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")


def run_world(name):
    mod = importlib.import_module(f"example_worlds.{name}")
    ws = mod.world_state
    subj = pick_subject(ws)
    if subj is None:
        chk(name, "no effective-clamp trait subject found (skipped)", True, "skip")
        return
    eid, axis, val, inertia, target = subj
    focus = [eid]

    # --- A: effective do() → minimal per-axis mutilation ---
    g = sandbox_for(ws, focus)
    other_axis_before = [
        (u, v) for u, v, d in g.in_edges(eid, data=True)
        if d.get("edge_type") == "causal"
        and d.get("trait_target") not in (None, axis)
    ]
    AMWNInstantiator.execute_interventions(g, {f"{eid}.traits.{axis}": target})
    surviving = in_causal_axis_edges(g, eid, axis)
    chk(name, f"A do({eid}.{axis}={target}) severs axis+untyped edges (G_xbar)",
        not surviving,
        detail=f"val={val:.2f} inertia={inertia:.2f} surviving={len(surviving)}")
    if other_axis_before:
        other_axis_after = [
            (u, v) for u, v, d in g.in_edges(eid, data=True)
            if d.get("edge_type") == "causal"
            and d.get("trait_target") not in (None, axis)
        ]
        chk(name, "A other-axis edges preserved (minimal, not whole-node)",
            len(other_axis_after) == len(other_axis_before),
            detail=f"before={len(other_axis_before)} after={len(other_axis_after)}")

    # --- B: phantom absent-axis do(=1.0) → materialise + sever untyped ---
    g2 = sandbox_for(ws, focus)
    untyped_before = [
        (u, v) for u, v, d in g2.in_edges(eid, data=True)
        if d.get("edge_type") == "causal" and d.get("trait_target") is None
    ]
    AMWNInstantiator.execute_interventions(g2, {f"{eid}.traits.{PHANTOM}": 1.0})
    mat = (g2.nodes.get(eid) or {}).get("traits", {}).get(PHANTOM)
    materialised = isinstance(mat, dict) and "value" in mat and "inertia" in mat
    chk(name, "B phantom axis materialised as TraitVector",
        materialised, detail=f"node={mat}")
    untyped_after = [
        (u, v) for u, v, d in g2.in_edges(eid, data=True)
        if d.get("edge_type") == "causal" and d.get("trait_target") is None
    ]
    chk(name, "B phantom do(=1.0) severs untyped in-edges",
        len(untyped_after) == 0 or len(untyped_before) == 0,
        detail=f"untyped before={len(untyped_before)} after={len(untyped_after)}")

    # --- C: no-op do(=0.0) on absent axis → inertia-blocked, no severance ---
    g3 = sandbox_for(ws, focus)
    before = len(all_in_causal(g3, eid))
    AMWNInstantiator.execute_interventions(g3, {f"{eid}.traits.{PHANTOM}": 0.0})
    after = len(all_in_causal(g3, eid))
    chk(name, "C no-op do(=0.0) on absent axis severs nothing (inertia gate)",
        before == after, detail=f"in-causal before={before} after={after}")


def main():
    print("=" * 78)
    print("  Minimal per-axis do-mutilation + inertia-gate validation")
    print("  (14 worlds outside the Pearl-CTF set)")
    print("=" * 78)
    for w in EXTRA_WORLDS:
        print(f"\n--- {w} ---")
        try:
            run_world(w)
        except Exception as e:  # noqa: BLE001
            chk(w, f"world raised {type(e).__name__}: {e}", False)
    print("\n" + "=" * 78)
    total = PASS + FAIL
    pct = 100.0 * PASS / total if total else 0.0
    print(f"RESULT: {PASS}/{total} checks PASSED ({pct:.1f}%), {FAIL} FAILED")
    print("=" * 78)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
