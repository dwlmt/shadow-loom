#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run real Pearl-rung queries on Macbeth and dump the engine's numeric output.

Used to thread real numbers into the paper and docs worked examples.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from example_worlds import macbeth, gone_girl, romeo_and_juliet  # noqa: E402
from shadow_loom.amwn import build_causal_diagram  # noqa: E402
from shadow_loom.causal_physics import CausalPhysicsEngine  # noqa: E402
from shadow_loom.extract_graph import extract_ego_graph_from_memory  # noqa: E402
from shadow_loom.instantiator import AMWNInstantiator  # noqa: E402


def _summarise(result, name):
    print(f"\n=== {name} ===")
    print(f"  intervened_nodes      : {result.intervened_nodes}")
    print(f"  rule3_pruned          : {result.rule3_pruned_interventions}")
    print(f"  rule2_redundant       : {result.rule2_redundant_evidence}")
    print(f"  hidden_deltas         : {dict(list(result.hidden_deltas.items())[:6])}")
    print(f"  mutations ({len(result.mutations)}):")
    for m in result.mutations[:10]:
        print(f"    {m.node_id:24s} {m.trait:18s} {m.old_value:+.3f} -> {m.new_value:+.3f}  (impact={m.impact:+.3f})")
    print(f"  social_mutations ({len(result.social_mutations)}):")
    for m in result.social_mutations[:6]:
        print(f"    {m.source_entity_id}->{m.target_entity_id} {m.metric:10s} "
              f"{m.old_value:+.3f} -> {m.new_value:+.3f}  via {m.triggered_by}")
    print(f"  blocked ({len(result.blocked)}):")
    for b in result.blocked[:5]:
        print(f"    {b.node_id} {b.trait:18s} impact={b.impact:+.3f} reason={b.reason}")


def run_macbeth():
    ws = macbeth.world_state
    focus = ["ENT_MACBETH", "ENT_LADY_MACBETH", "ENT_DUNCAN", "ENT_BANQUO", "ENT_MACDUFF"]
    ego = extract_ego_graph_from_memory(ws, focus_entity_ids=focus)
    payload = ego.model_dump()

    # ── Rung 1: Observation (no interventions, no abduction) ─────
    sb = AMWNInstantiator.create_sandbox(payload, "observation")
    eng = CausalPhysicsEngine(sb, ws)
    r1 = eng.execute(rung=2, interventions={}, target_node_ids=focus)
    _summarise(r1, "MACBETH · Rung 1 (Observation, do={})")

    # ── Rung 2: Intervention do(ambition=0) ──────────────────────
    sb = AMWNInstantiator.create_sandbox(payload, "intervention")
    eng = CausalPhysicsEngine(sb, ws)
    r2 = eng.execute(
        rung=2,
        interventions={"ENT_MACBETH.traits.ambition": 0.0},
        target_node_ids=["ENT_DUNCAN", "ENT_LADY_MACBETH"],
    )
    _summarise(r2, "MACBETH · Rung 2 (do(ENT_MACBETH.ambition=0))")

    # ── Rung 3: Counterfactual (abduction + do) ──────────────────
    sb = AMWNInstantiator.create_sandbox(payload, "counterfactual")
    eng = CausalPhysicsEngine(sb, ws)
    r3 = eng.execute(
        rung=3,
        interventions={"ENT_MACBETH.traits.ambition": 0.0},
        evidence_node_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"],
        target_node_ids=["ENT_DUNCAN", "ENT_LADY_MACBETH"],
    )
    _summarise(r3, "MACBETH · Rung 3 (abduction + do(ambition=0))")

    # ── Rung 2 with Rule-3 vacuous intervention to show pruning ──
    sb = AMWNInstantiator.create_sandbox(payload, "intervention")
    eng = CausalPhysicsEngine(sb, ws)
    r4 = eng.execute(
        rung=2,
        interventions={"ENT_DUNCAN.traits.kindness": 0.0},
        target_node_ids=["ENT_BANQUO"],   # no path -> Rule 3 should flag
    )
    _summarise(r4, "MACBETH · Rung 2 (do(ENT_DUNCAN.kindness=0), target=ENT_BANQUO)")


def run_romeo():
    ws = romeo_and_juliet.world_state
    focus = ["ENT_ROMEO", "ENT_JULIET", "ENT_TYBALT", "ENT_MERCUTIO", "ENT_FRIAR_LAURENCE"]
    ego = extract_ego_graph_from_memory(ws, focus_entity_ids=focus)
    payload = ego.model_dump()

    sb = AMWNInstantiator.create_sandbox(payload, "intervention")
    eng = CausalPhysicsEngine(sb, ws)
    r = eng.execute(
        rung=2,
        interventions={"ENT_FRIAR_LAURENCE.traits.diligence": 1.0},
        target_node_ids=["ENT_ROMEO", "ENT_JULIET"],
    )
    _summarise(r, "ROMEO+JULIET · Rung 2 (do(Friar.diligence=1.0))")


def run_gone_girl():
    ws = gone_girl.world_state
    focus = ["ENT_NICK", "ENT_AMY", "ENT_BONEY"]
    ego = extract_ego_graph_from_memory(ws, focus_entity_ids=focus)
    payload = ego.model_dump()

    sb = AMWNInstantiator.create_sandbox(payload, "counterfactual")
    eng = CausalPhysicsEngine(sb, ws)
    r = eng.execute(
        rung=3,
        interventions={"ENT_AMY.traits.deceit": 0.0},
        evidence_node_ids=["ENT_NICK", "ENT_AMY"],
        target_node_ids=["ENT_NICK"],
    )
    _summarise(r, "GONE GIRL · Rung 3 (abduction + do(Amy.deceit=0))")


run_macbeth()
run_romeo()
run_gone_girl()
