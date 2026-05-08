#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full-pipeline sweep across all example_worlds.

For each world, runs:
  * AMWN build (causal diagram)
  * narrative_physics for query types: observation, intervention,
    counterfactual, directive
  * Direct Pearl rungs (Rung 1, Rung 2, Rung 3) via CausalPhysicsEngine

Reports timings and headline metrics so we can sanity-check curves and
compare against narrative-theory expectations.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Silence the noisy CausalPhysics·Propagate cycle warnings — they're
# expected on counter-concern dyads and would drown the output.
logging.getLogger("shadow_loom.causal_physics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import example_worlds  # noqa: E402
from shadow_loom.amwn import build_causal_diagram  # noqa: E402
from shadow_loom.causal_physics import CausalPhysicsEngine  # noqa: E402
from shadow_loom.extract_graph import extract_ego_graph_from_memory  # noqa: E402
from shadow_loom.instantiator import AMWNInstantiator  # noqa: E402
from shadow_loom.narrative_physics import calculate_narrative_physics  # noqa: E402
from shadow_loom.query_models import (  # noqa: E402
    ObservationQuery, InterventionQuery, CounterfactualQuery, DirectiveQuery,
)


def _focal_entities(ws, n: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for ev in ws.events:
        for a in (ev.actor_ids or []):
            counts[a] = counts.get(a, 0) + 1
        for t in (ev.target_ids or []):
            counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [eid for eid, _ in ranked[:n] if eid.startswith("ENT_")]


def _pick_intervention(ws, focus: list[str]) -> tuple[str, str, float] | None:
    """Pick (entity_id, trait, value) for a do-operator on first focal entity
    that has a numeric trait."""
    for eid in focus:
        ent = ws.entities.get(eid)
        if ent is None or not ent.traits:
            continue
        for tname, tv in ent.traits.items():
            if hasattr(tv, "value") and isinstance(tv.value, (int, float)):
                return eid, tname, 0.0
    return None


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0


def sweep_world(name: str, ws) -> dict:
    focus = _focal_entities(ws, n=5)
    row: dict = {"world": name, "n_entities": len(ws.entities),
                 "n_events": len(ws.events), "focus": focus}

    # ── AMWN build ─────────────────────────────────────────
    diagram, t_amwn = _time(build_causal_diagram, ws)
    row["amwn_ms"] = t_amwn
    row["amwn_nodes"] = diagram.number_of_nodes()
    row["amwn_edges"] = diagram.number_of_edges()

    # ── Ego graph (used by causal physics) ─────────────────
    ego, t_ego = _time(extract_ego_graph_from_memory, ws,
                       focus_entity_ids=focus)
    row["ego_ms"] = t_ego
    row["ego_entities"] = len(getattr(ego, "entities", {}) or {})

    payload = ego.model_dump()

    # ── Pearl Rung 1 (observation) ─────────────────────────
    sb1 = AMWNInstantiator.create_sandbox(payload, "observation")
    eng1 = CausalPhysicsEngine(sb1, ws)
    r1, t_r1 = _time(eng1.execute, rung=2, interventions={},
                     target_node_ids=focus)
    row["r1_ms"] = t_r1
    row["r1_mutations"] = len(r1.mutations)
    row["r1_blocked"] = len(r1.blocked)

    # ── Pearl Rung 2 (do(...)) ─────────────────────────────
    pick = _pick_intervention(ws, focus)
    if pick is not None:
        eid, tname, val = pick
        sb2 = AMWNInstantiator.create_sandbox(payload, "intervention")
        eng2 = CausalPhysicsEngine(sb2, ws)
        r2, t_r2 = _time(eng2.execute, rung=2,
                         interventions={f"{eid}.traits.{tname}": val},
                         target_node_ids=[e for e in focus if e != eid])
        row["r2_ms"] = t_r2
        row["r2_intervened"] = r2.intervened_nodes
        row["r2_pruned"] = r2.rule3_pruned_interventions
        row["r2_mutations"] = len(r2.mutations)
        row["r2_target"] = f"{eid}.{tname}"
    else:
        row["r2_ms"] = None
        row["r2_target"] = None

    # ── Pearl Rung 3 (abduction + do) ──────────────────────
    if pick is not None:
        eid, tname, val = pick
        sb3 = AMWNInstantiator.create_sandbox(payload, "counterfactual")
        eng3 = CausalPhysicsEngine(sb3, ws)
        r3, t_r3 = _time(eng3.execute, rung=3,
                         interventions={f"{eid}.traits.{tname}": val},
                         evidence_node_ids=focus,
                         target_node_ids=[e for e in focus if e != eid])
        row["r3_ms"] = t_r3
        row["r3_hidden_vars"] = sum(len(v) for v in r3.hidden_deltas.values())
        row["r3_mutations"] = len(r3.mutations)
    else:
        row["r3_ms"] = None

    # ── narrative_physics: 4 query types ────────────────────
    timings: dict[str, float] = {}

    obs_q = ObservationQuery(focus_entity_ids=focus)
    _, timings["np_observation"] = _time(
        calculate_narrative_physics, obs_q, ws)

    if pick is not None:
        eid, tname, val = pick
        iq = InterventionQuery(
            interventions={f"{eid}.traits.{tname}": val},
            focus_entity_ids=focus,
            target_node_ids=[e for e in focus if e != eid],
        )
        _, timings["np_intervention"] = _time(
            calculate_narrative_physics, iq, ws)

        cq = CounterfactualQuery(
            historical_interventions={f"{eid}.traits.{tname}": val},
            evidence_node_ids=focus,
            focus_entity_ids=focus,
            target_node_ids=[e for e in focus if e != eid],
        )
        _, timings["np_counterfactual"] = _time(
            calculate_narrative_physics, cq, ws)

    if focus:
        dq = DirectiveQuery(
            target_entity_ids=focus[:2],
            target_effect="suspense",
            focus_entity_ids=focus,
        )
        _, timings["np_directive"] = _time(
            calculate_narrative_physics, dq, ws,
            use_causal_engine=True,
        )

    for k, v in timings.items():
        row[k + "_ms"] = v

    return row


def main() -> None:
    rows = []
    for mod in sorted(pkgutil.iter_modules(example_worlds.__path__),
                      key=lambda m: m.name):
        if mod.name.startswith("_"):
            continue
        try:
            m = importlib.import_module(f"example_worlds.{mod.name}")
            ws = getattr(m, "world_state", None)
            if ws is None:
                continue
            row = sweep_world(mod.name, ws)
            rows.append(row)
            print(f"OK {mod.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {mod.name}: {exc!r}")

    print("\n" + "=" * 110)
    print("PER-WORLD TIMINGS (ms)")
    print("=" * 110)
    cols = ["amwn_ms", "ego_ms", "r1_ms", "r2_ms", "r3_ms",
            "np_observation_ms", "np_intervention_ms",
            "np_counterfactual_ms", "np_directive_ms"]
    hdr = f"{'world':32s} {'|V|':>5s} {'|E|':>5s}  " + " ".join(
        f"{c.replace('_ms',''):>14s}" for c in cols)
    print(hdr)
    for r in rows:
        line = (f"{r['world']:32s} {r['amwn_nodes']:>5d} "
                f"{r['amwn_edges']:>5d}  ")
        for c in cols:
            v = r.get(c)
            line += f"{(f'{v:>10.1f}' if isinstance(v,(int,float)) else '-'):>14s} "
        print(line)

    print("\n" + "=" * 110)
    print("PEARL-RUNG STRUCTURAL METRICS")
    print("=" * 110)
    hdr = (f"{'world':32s} {'r1.mut':>7s} {'r1.blk':>7s} "
           f"{'r2.tgt':<28s} {'r2.mut':>7s} {'r2.pruned':>10s} "
           f"{'r3.hidden':>10s} {'r3.mut':>7s}")
    print(hdr)
    for r in rows:
        line = (f"{r['world']:32s} {r.get('r1_mutations','-'):>7} "
                f"{r.get('r1_blocked','-'):>7} "
                f"{str(r.get('r2_target','-') or '-'):<28s} "
                f"{r.get('r2_mutations','-'):>7} "
                f"{len(r.get('r2_pruned',[]) or []):>10} "
                f"{r.get('r3_hidden_vars','-'):>10} "
                f"{r.get('r3_mutations','-'):>7}")
        print(line)

    print("\n" + "=" * 110)
    print("AGGREGATE TIMINGS (ms) over", len(rows), "worlds")
    print("=" * 110)
    print(f"{'stage':22s} {'min':>9s} {'median':>9s} {'mean':>9s} {'max':>9s}")
    for c in cols:
        vals = [r.get(c) for r in rows if isinstance(r.get(c), (int, float))]
        if not vals:
            continue
        print(f"{c.replace('_ms',''):22s} {min(vals):9.1f} {median(vals):9.1f} "
              f"{mean(vals):9.1f} {max(vals):9.1f}")


if __name__ == "__main__":
    main()
