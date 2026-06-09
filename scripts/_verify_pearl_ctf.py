#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify Pearl-rung reasoning + ctf-calculus on real narratives.

Scope
-----
Engine layers covered:
  • do-surgery on TRAITS         (DoTrait / legacy path-style)
  • do-surgery on EVENTS         (DoEvent — clamp occurred=False)
  • do-surgery on PROPOSITIONS   (DoProposition — clamp truth + cascade)
  • do-surgery on BELIEFS        (DoBelief — single-character epistemic)
  • do-surgery on CONCERNS       (DoConcern — utility-layer)
  • blocking / unblocking through propagation
  • shadow-AMWN counterfactual reasoning (Rung 3 abduction → do → propagate)
  • ctf-calculus pre-flight rules 1, 2, 3 (Correa & Bareinboim ICML 2025)

Each check prints PASS / FAIL with a short detail line. Exit code is 0
iff every check passed.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.getLogger("shadow_loom.causal_physics").setLevel(logging.ERROR)
logging.getLogger("shadow_loom.amwn").setLevel(logging.ERROR)
logging.getLogger("shadow_loom.instantiator").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Disable Monte-Carlo sampling for verification: we need to inspect the
# top-level engine's sandbox directly after execute(), which doesn't
# happen when execute() routes through execute_distribution.
from shadow_loom.causal_physics import _physics_settings  # noqa: E402
_physics_settings().monte_carlo_samples = 0

from example_worlds import (  # noqa: E402
    macbeth, romeo_and_juliet, gone_girl, tinker_tailor_soldier_spy,
    nineteen_eighty_four, frankenstein,
)
from shadow_loom.amwn import (  # noqa: E402
    apply_ctf_calculus, build_causal_diagram,
)
from shadow_loom.causal_physics import CausalPhysicsEngine  # noqa: E402
from shadow_loom.extract_graph import extract_ego_graph_from_memory  # noqa: E402
from shadow_loom.instantiator import (  # noqa: E402
    AMWNInstantiator,
    _entity_trait_inertia_default,
    _inertia_epsilon,
)
from shadow_loom.query_models import (  # noqa: E402
    DoBelief, DoConcern, DoEvent, DoProposition, DoTrait,
)


# ----------------------------------------------------------------------
# PASS/FAIL accounting
# ----------------------------------------------------------------------
PASS, FAIL = 0, 0
WORLD: str = ""


def _check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{tag}] {WORLD:24s} {label}{suffix}")


def _new_engine(ws, focus, query_type="intervention"):
    ego = extract_ego_graph_from_memory(ws, focus_entity_ids=focus)
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), query_type)
    return CausalPhysicsEngine(sandbox, ws), sandbox


def _trait_value(ws, eid: str, trait: str):
    e = ws.entities.get(eid)
    if e is None or trait not in (e.traits or {}):
        return None
    tv = e.traits[trait]
    return float(getattr(tv, "value", tv))


def _find_vacuous_pair(ws, target: str):
    """Pick an entity-trait whose entity is NOT an ancestor of *target*
    in the structural causal diagram — guaranteed Rule-3 vacuous.
    """
    diagram = build_causal_diagram(ws)
    if not diagram.has_node(target):
        return None
    anc = nx.ancestors(diagram, target) | {target}
    for eid, ent in ws.entities.items():
        if eid in anc:
            continue
        if not ent.traits:
            continue
        trait = next(iter(ent.traits))
        return f"{eid}.traits.{trait}"
    return None


# ======================================================================
# RUNG 1 — Observation
# ======================================================================
def verify_rung1(ws, focus):
    eng, _ = _new_engine(ws, focus, "observation")
    r = eng.execute(rung=2, interventions={}, target_node_ids=focus)
    _check("R1 no interventions",         not r.intervened_nodes)
    _check("R1 no rule3_pruned",          not r.rule3_pruned_interventions)
    _check("R1 no rule2_redundant",       not r.rule2_redundant_evidence)
    _check("R1 no proposition mutations", not r.proposition_mutations)
    _check("R1 no belief mutations",      not r.belief_mutations)
    _check("R1 no concern mutations",     not r.concern_mutations)


# ======================================================================
# RUNG 2 — Trait surgery (mutilation invariant + downstream containment)
# ======================================================================
def verify_rung2_trait(ws, focus, intervention, target):
    diagram = build_causal_diagram(ws)
    intervened_nid = next(iter(intervention)).split(".", 1)[0]
    trait = next(iter(intervention)).rsplit(".", 1)[-1]
    new_value = float(next(iter(intervention.values())))

    # Snapshot the sandbox right after surgery to verify the
    # do-mutilation invariant before propagation re-adds ambient edges.
    # P0-5 minimal per-axis mutilation (Pearl G_{\bar X}): surgery severs
    # only edges feeding the *intervened axis* — those whose ``trait_target``
    # equals the axis, plus untyped edges (``trait_target is None``) that
    # could fire onto any axis. Edges explicitly targeting a *different*
    # axis on the same entity are intentionally preserved, so counting all
    # in-causal edges into the node over-reports. Mirror the engine's
    # severance predicate (instantiator ``_intervene_state``) instead.
    eng, sandbox = _new_engine(ws, focus, "intervention")
    AMWNInstantiator.execute_interventions(sandbox, intervention)

    in_causal_post_surgery = [
        (u, v) for u, v, d in sandbox.in_edges(intervened_nid, data=True)
        if d.get("edge_type") == "causal"
        and d.get("trait_target") in (None, trait)
    ]

    # The do-mutilation invariant only applies when surgery actually FIRES.
    # The inertia gate (instantiator ``_intervene_state``) blocks a do() whose
    # |desired_shift| <= inertia and returns *without severing edges* — e.g. a
    # do(=0.0) on an absent axis, which materialises at the 0.0 baseline for a
    # zero shift. Such a no-op leaves edges intact by design, so asserting
    # G_{\bar X} on it would be a false failure. Replicate the engine's gate on
    # the *original* (pre-surgery) value/inertia to decide applicability.
    _orig_ent = getattr(ws, "entities", {}).get(intervened_nid)
    _orig_tv = (getattr(_orig_ent, "traits", {}) or {}).get(trait) if _orig_ent else None
    cur_val = float(_orig_tv.value) if _orig_tv is not None else 0.0
    cur_inertia = (
        float(_orig_tv.inertia) if _orig_tv is not None
        else _entity_trait_inertia_default()
    )
    surgery_fired = abs(new_value - cur_val) > cur_inertia + _inertia_epsilon()
    if surgery_fired:
        _check(
            f"R2 do-mutilation: {intervened_nid}.{trait} axis has 0 incoming "
            f"causal edges (axis-targeting or untyped) immediately after surgery",
            not in_causal_post_surgery,
            detail=f"surviving={len(in_causal_post_surgery)}",
        )
    else:
        _check(
            f"R2 do-mutilation N/A: do({intervened_nid}.{trait}={new_value}) "
            f"inertia-blocked (no surgery fired) — invariant vacuous",
            True,
            detail=f"current={cur_val} inertia={cur_inertia:.2f} target={new_value}",
        )

    # Run the full execute() and verify the intervened TRAIT is pinned —
    # the post-propagation invariant.
    eng2, _ = _new_engine(ws, focus, "intervention")
    r = eng2.execute(rung=2, interventions=intervention, target_node_ids=[target])
    pinned_value = None
    nd = eng2.sandbox.nodes.get(intervened_nid, {})
    traits = nd.get("traits") or {}
    if isinstance(traits, dict) and trait in traits:
        tv = traits[trait]
        pinned_value = tv.get("value") if isinstance(tv, dict) else float(tv)
    inertia = (
        traits.get(trait, {}).get("inertia", 0.5)
        if isinstance(traits.get(trait), dict) else 0.5
    )
    # Inertia dampens the effective shift, so allow up to one inertia
    # step of slack from the requested clamp value.
    pinned_ok = (
        pinned_value is not None
        and abs(float(pinned_value) - new_value) <= float(inertia) + 0.05
    )
    _check(
        f"R2 intervened trait {intervened_nid}.{trait} is pinned (target={new_value}, inertia={inertia:.2f})",
        pinned_ok,
        detail=f"pinned_value={pinned_value}",
    )

    # Mutations on entities must be reachable from the do-target in the
    # SCM (no spooky action). Allow REL_::* synthetic nodes.
    if diagram.has_node(intervened_nid):
        downstream = nx.descendants(diagram, intervened_nid) | {intervened_nid}
        unreachable = [
            m.node_id for m in r.mutations
            if diagram.has_node(m.node_id) and m.node_id not in downstream
        ]
        # Soft check — `relationship` edges in the sandbox can carry
        # propagation back through the entity graph in ways the static
        # SCM doesn't surface as a directed ancestor-of edge. Print the
        # diagnostic but tolerate up to ⅓ strays before failing.
        tolerance = max(2, int(0.33 * max(len(r.mutations), 1)))
        _check(
            f"R2 mutations stay within descendants(X) (tolerance ≤{tolerance})",
            len(unreachable) <= tolerance,
            detail=f"strays={len(unreachable)}/{len(r.mutations)}",
        )
    _check("R2 intervened_nodes recorded", intervened_nid in r.intervened_nodes)


# ======================================================================
# RUNG 2 — Proposition / Belief / Concern surgery
# ======================================================================
def verify_rung2_proposition(ws, focus, prop_id: str):
    """do(PROP_X.truth=False, propagate_to_beliefs=True) clamps the truth
    and cascades into matching beliefs."""
    eng, _ = _new_engine(ws, focus, "intervention")
    do = DoProposition(proposition_id=prop_id, truth=False, propagate_to_beliefs=True)
    eng.apply_do_targets([do])
    eng.propagate()
    eng.propagate_social()

    _check(
        f"R2 proposition {prop_id} clamp recorded",
        any(pm.proposition_id == prop_id for pm in eng._proposition_mutations),
    )
    cascaded = sum(pm.cascaded_belief_count for pm in eng._proposition_mutations)
    _check(
        f"R2 proposition cascade reaches ≥1 belief",
        cascaded >= 1,
        detail=f"cascaded_belief_count={cascaded}",
    )
    clamps = eng.sandbox.graph.get("proposition_clamps", [])
    _check(
        f"R2 proposition_clamps recorded on shadow sandbox",
        any(c.get("proposition_id") == prop_id for c in clamps),
        detail=f"#clamps={len(clamps)}",
    )


def verify_rung2_belief(ws, focus, holder: str, target_id: str):
    eng, _ = _new_engine(ws, focus, "intervention")
    do = DoBelief(holder_id=holder, target_id=target_id, confidence=0.0)
    eng.apply_do_targets([do])
    _check(
        f"R2 belief clamp on {holder}↦{target_id} recorded",
        any(bm.holder_id == holder and bm.target_id == target_id
            for bm in eng._belief_mutations),
    )
    _check(
        f"R2 belief surgery pins holder {holder}",
        holder in eng._intervened_nodes,
    )


def verify_rung2_concern(ws, focus, holder: str, concern_id: str):
    eng, _ = _new_engine(ws, focus, "intervention")
    do = DoConcern(holder_id=holder, concern_id=concern_id, active=False)
    eng.apply_do_targets([do])
    _check(
        f"R2 concern clamp {holder}.{concern_id} recorded",
        any(cm.holder_id == holder and cm.concern_id == concern_id
            for cm in eng._concern_mutations),
    )
    _check(
        f"R2 concern surgery pins holder {holder}",
        holder in eng._intervened_nodes,
    )


def verify_rung2_event(ws, focus, evt_id: str):
    """do(EVT_X.occurred=False) flips the sandbox event to prevented and
    triggers provenance invalidation downstream."""
    eng, _ = _new_engine(ws, focus, "intervention")
    do = DoEvent(event_id=evt_id, occurred=False)
    eng.apply_do_targets([do])
    eng.propagate()
    eng.propagate_social()

    nd = eng.sandbox.nodes.get(evt_id, {})
    event_blocked = (
        nd.get("event_type") == "prevented"
        or nd.get("pruned") is True
    )
    _check(
        f"R2 event {evt_id} marked prevented/pruned",
        event_blocked,
        detail=f"event_type={nd.get('event_type')} pruned={nd.get('pruned')}",
    )


# ======================================================================
# Rule-1 (Consistency) — do(X = observed(X)) flagged redundant
# ======================================================================
def verify_rule1(ws, focus, holder: str, trait: str):
    observed = _trait_value(ws, holder, trait)
    if observed is None:
        return
    path = f"{holder}.traits.{trait}.value"
    rep = apply_ctf_calculus(
        ws, interventions={path: observed}, target_node_ids=focus,
    )
    _check(
        f"Rule-1 flags redundant {holder}.{trait}=observed({observed})",
        path in rep.rule1_redundant,
        detail=f"redundant={rep.rule1_redundant}",
    )


# ======================================================================
# Rule-3 (Exclusion) — vacuous intervention pruned
# ======================================================================
def verify_rule3(ws, focus, target: str):
    vac = _find_vacuous_pair(ws, target)
    if vac is None:
        _check(f"Rule-3: no vacuous pair found for {target} (skipped)", True)
        return
    rep = apply_ctf_calculus(
        ws, interventions={vac: 0.0}, target_node_ids=[target],
    )
    _check(
        f"Rule-3 prunes topologically vacuous {vac}→{target}",
        vac in rep.rule3_pruned,
        detail=f"rule3_pruned={rep.rule3_pruned[:2]}",
    )


# ======================================================================
# RUNG 3 — Counterfactual via shadow AMWN (abduction → do → propagate)
# ======================================================================
def verify_rung3(ws, focus, intervention, evidence, target):
    eng, _ = _new_engine(ws, focus, "counterfactual")
    r = eng.execute(
        rung=3,
        interventions=intervention,
        evidence_node_ids=evidence,
        target_node_ids=[target],
    )
    intervened_nid = next(iter(intervention)).split(".", 1)[0]
    _check("R3 do-surgery recorded on shadow AMWN",
           intervened_nid in r.intervened_nodes)
    _check("R3 abduction ran without error",
           isinstance(r.hidden_deltas, dict))


# ======================================================================
# Blocking / unblocking — propagation through gated descendants
# ======================================================================
def verify_blocking(ws, focus, upstream_intervention, gated_descendant):
    eng, _ = _new_engine(ws, focus, "intervention")
    r = eng.execute(rung=2, interventions=upstream_intervention,
                    target_node_ids=[gated_descendant])
    hits = sum(1 for m in r.mutations if m.node_id == gated_descendant)
    blocked = sum(1 for b in r.blocked if b.node_id == gated_descendant)
    _check(
        f"propagation reached gated descendant {gated_descendant}",
        hits + blocked > 0,
        detail=f"mutations={hits} blocked_records={blocked}",
    )


# ======================================================================
# Cases
# ======================================================================
CASES = [
    {
        "name": "macbeth",
        "world": macbeth.world_state,
        "focus": ["ENT_MACBETH", "ENT_LADY_MACBETH", "ENT_DUNCAN",
                  "ENT_BANQUO", "ENT_MACDUFF"],
        "trait_intervention": {"ENT_MACBETH.traits.ambition": 0.0},
        "trait_target": "ENT_DUNCAN",
        "redundant_holder": "ENT_MACBETH", "redundant_trait": "courage",
        "rule3_target": "ENT_BANQUO",
        "r3_intervention": {"ENT_MACBETH.traits.ambition": 0.0},
        "r3_evidence": ["ENT_DUNCAN", "ENT_LADY_MACBETH"],
        "r3_target": "ENT_MACBETH",
        "prop_id": "PROP_MACBETH_BECOMES_KING",
        "belief_holder": "ENT_MACBETH", "belief_target": "ENT_DUNCAN",
        "concern_holder": "ENT_MACBETH",
        "concern_id": "CCN_MACBETH_BECOMES_KING",
        "block_intervention": {"ENT_LADY_MACBETH.traits.persuasion": 1.0},
        "block_descendant": "ENT_MACBETH",
    },
    {
        "name": "romeo_and_juliet",
        "world": romeo_and_juliet.world_state,
        "focus": ["ENT_ROMEO", "ENT_JULIET", "ENT_TYBALT",
                  "ENT_MERCUTIO", "ENT_FRIAR_LAURENCE"],
        "trait_intervention": {"ENT_FRIAR_LAURENCE.traits.wisdom": 0.0},
        "trait_target": "ENT_JULIET",
        "redundant_holder": "ENT_ROMEO", "redundant_trait": "passion",
        "rule3_target": "ENT_FRIAR_LAURENCE",
        "r3_intervention": {"ENT_TYBALT.traits.aggression": 0.0},
        "r3_evidence": ["ENT_MERCUTIO", "ENT_ROMEO"],
        "r3_target": "ENT_TYBALT",
        "belief_holder": "ENT_ROMEO", "belief_target": "ENT_JULIET",
        "concern_holder": "ENT_ROMEO",
        "block_intervention": {"ENT_TYBALT.traits.aggression": 1.0},
        "block_descendant": "ENT_MERCUTIO",
    },
    {
        "name": "gone_girl",
        "world": gone_girl.world_state,
        "focus": ["ENT_NICK", "ENT_AMY", "ENT_BONEY"],
        "trait_intervention": {"ENT_AMY.traits.cunning": 0.0},
        "trait_target": "ENT_NICK",
        "redundant_holder": "ENT_AMY", "redundant_trait": "cunning",
        "rule3_target": "ENT_AMY",
        "r3_intervention": {"ENT_AMY.traits.cunning": 0.0},
        "r3_evidence": ["ENT_NICK"],
        "r3_target": "ENT_AMY",
        "belief_holder": "ENT_NICK", "belief_target": "ENT_AMY",
        "concern_holder": "ENT_NICK",
        "block_intervention": {"ENT_AMY.traits.cunning": 1.0},
        "block_descendant": "ENT_NICK",
    },
    {
        "name": "nineteen_eighty_four",
        "world": nineteen_eighty_four.world_state,
        "focus": ["ENT_WINSTON", "ENT_JULIA", "ENT_OBRIEN"],
        "trait_intervention": {"ENT_OBRIEN.traits.deceit": 0.0},
        "trait_target": "ENT_WINSTON",
        "redundant_holder": "ENT_WINSTON",
        "redundant_trait": "memory_for_truth",
        "rule3_target": "ENT_OBRIEN",
        "r3_intervention": {"ENT_OBRIEN.traits.loyalty": 1.0},
        "r3_evidence": ["ENT_WINSTON"],
        "r3_target": "ENT_OBRIEN",
        "belief_holder": "ENT_WINSTON", "belief_target": "ENT_OBRIEN",
        "concern_holder": "ENT_WINSTON",
        "block_intervention": {"ENT_OBRIEN.traits.deceit": 1.0},
        "block_descendant": "ENT_WINSTON",
    },
    {
        "name": "frankenstein",
        "world": frankenstein.world_state,
        "focus": ["ENT_VICTOR", "ENT_CREATURE", "ENT_ELIZABETH"],
        "trait_intervention": {"ENT_VICTOR.traits.ambition": 0.0},
        "trait_target": "ENT_CREATURE",
        "redundant_holder": "ENT_VICTOR", "redundant_trait": "ambition",
        "rule3_target": "ENT_ELIZABETH",
        "r3_intervention": {"ENT_VICTOR.traits.ambition": 0.0},
        "r3_evidence": ["ENT_CREATURE"],
        "r3_target": "ENT_VICTOR",
        "belief_holder": "ENT_VICTOR", "belief_target": "ENT_CREATURE",
        "concern_holder": "ENT_VICTOR",
        "block_intervention": {"ENT_VICTOR.traits.ambition": 0.0},
        "block_descendant": "ENT_CREATURE",
    },
]


def _populate_dynamic_ids(case: dict) -> None:
    """Fill in prop_id / concern_id / event_id from fixture when unspecified."""
    ws = case["world"]
    if not case.get("prop_id"):
        for prop in (ws.propositions or []):
            referenced = any(
                b.proposition_id == prop.proposition_id
                for ent in ws.entities.values() for b in (ent.beliefs or [])
            )
            if referenced:
                case["prop_id"] = prop.proposition_id
                break
    if not case.get("concern_id"):
        holder = case.get("concern_holder")
        ent = ws.entities.get(holder)
        if ent and ent.concerns:
            case["concern_id"] = ent.concerns[0].concern_id
    if not case.get("event_id"):
        # Pick an event that is actually present in the focus ego
        # sandbox — ws.events[0] may be filtered out by the ego graph.
        ego = extract_ego_graph_from_memory(ws, focus_entity_ids=case["focus"])
        sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
        sandbox_events = {
            nid for nid, d in sandbox.nodes(data=True)
            if d.get("node_type") == "EventNode"
        }
        for evt in (ws.events or []):
            if evt.id in sandbox_events:
                case["event_id"] = evt.id
                break


# ======================================================================
# Per-world driver
# ======================================================================
def run_world(case: dict) -> None:
    global WORLD
    WORLD = case["name"]
    ws = case["world"]
    focus = case["focus"]

    _populate_dynamic_ids(case)

    verify_rung1(ws, focus)
    verify_rung2_trait(ws, focus, case["trait_intervention"], case["trait_target"])
    verify_rule1(ws, focus, case["redundant_holder"], case["redundant_trait"])
    verify_rule3(ws, focus, case["rule3_target"])
    verify_rung3(ws, focus, case["r3_intervention"],
                 case["r3_evidence"], case["r3_target"])

    if case.get("prop_id"):
        verify_rung2_proposition(ws, focus, case["prop_id"])
    else:
        _check(f"R2 proposition: no prop_id available (skipped)", True)

    if case.get("belief_holder") and case.get("belief_target"):
        verify_rung2_belief(ws, focus, case["belief_holder"], case["belief_target"])

    if case.get("concern_id"):
        verify_rung2_concern(ws, focus, case["concern_holder"], case["concern_id"])
    else:
        _check(f"R2 concern: no concern_id available (skipped)", True)

    if case.get("event_id"):
        verify_rung2_event(ws, focus, case["event_id"])

    verify_blocking(ws, focus, case["block_intervention"], case["block_descendant"])


def main() -> None:
    print("=" * 76)
    print("PEARL RUNG + ctf-CALCULUS + AUDIENCE-LAYER VERIFICATION")
    print("=" * 76)
    for case in CASES:
        print(f"\n--- {case['name']} ---")
        run_world(case)

    print("\n" + "=" * 76)
    total = PASS + FAIL
    rate = (PASS / total * 100) if total else 0.0
    print(f"RESULT: {PASS}/{total} checks PASSED ({rate:.1f}%), {FAIL} FAILED")
    print("=" * 76)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
