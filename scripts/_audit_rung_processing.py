# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real-plot audit of Pearl-rung processing + affective scorers +
interrogation for the gold-standard example worlds.

Exercises realistic queries a user would actually run and prints a
diagnostic report. NOT a unit test — this is an investigative audit
intended to surface logical / semantic regressions the schema audit
cannot see.

Usage:
    python scripts/_audit_rung_processing.py
"""
from __future__ import annotations

import importlib
import json
from typing import Any, Dict, List

from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import (
    ObservationQuery,
    InterventionQuery,
    CounterfactualQuery,
    DoEvent,
    DoProposition,
    DoChannel,
    DoTrait,
)
from shadow_loom.affective_scorers import compute_affective_scorers
from shadow_loom.interrogate_posterior import (
    interrogate_posterior,
    surface_entity_constants,
    audit_posterior_consistency,
)


# ----------------------------------------------------------------- helpers

def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _sub(title: str) -> None:
    print(f"\n— {title} —")


def _fmt_keys(d: Dict[str, Any], skip: tuple = ()) -> str:
    return ", ".join(sorted(k for k in d.keys() if k not in skip))


def _ps_node_count(ps: Any) -> int:
    if isinstance(ps, dict):
        if "nodes" in ps:
            return len(ps["nodes"])
        if "entities" in ps:
            return len(ps.get("entities") or {})
    return 0


def _summarize_result(r: Dict[str, Any]) -> str:
    ps = r.get("physics_state")
    bits = [
        f"status={r.get('status')}",
        f"nodes={_ps_node_count(ps)}",
    ]
    for k in (
        "intervened_nodes", "do_targets", "historical_do_targets",
        "evidence_conditions", "hidden_deltas", "mutations",
        "social_mutations", "proposition_mutations", "belief_mutations",
        "concern_mutations", "past_anchor", "implausibility_reason",
        "intervention_inert", "skipped_interventions",
    ):
        if k in r and r[k]:
            v = r[k]
            if isinstance(v, (list, dict)):
                bits.append(f"{k}={len(v)}")
            else:
                bits.append(f"{k}={v}")
    return " | ".join(bits)


# ----------------------------------------------------------------- audit body

def audit_world(name: str, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    _hr(name.upper())
    mod = importlib.import_module(f"example_worlds.{name}")
    ws = mod.world_state
    print(f"entities={len(ws.entities)} events={len(ws.events)} "
          f"propositions={len(ws.propositions)} "
          f"social_edges={len(ws.social_topology)} "
          f"channels={len(ws.channels)}")

    # ---- Interrogation surfaces ----
    _sub("Interrogation: entity_constants")
    constants = surface_entity_constants(ws)
    print(f"  {len(constants)} constants surfaced")
    for c in constants[:8]:
        print(f"    {c['entity_id']:30s} {c['constant']}")
    if len(constants) > 8:
        print(f"    ... ({len(constants)-8} more)")

    # ---- Affective scorers across timeline ----
    _sub("Affective scorers at fabula ticks")
    if ws.events:
        ticks = sorted({e.fabula_time for e in ws.events})
        sample_ticks = [ticks[0], ticks[len(ticks)//2], ticks[-1]]
    else:
        sample_ticks = [None]
    sample_ticks.append(None)  # canonical
    for tick in sample_ticks:
        scores = compute_affective_scorers(ws, fabula_time=tick)
        label = f"tick={tick}" if tick is not None else "tick=canonical"
        print(f"  {label:20s} mystery={scores['mystery']:.3f}  "
              f"irony={scores['irony']:.3f}  "
              f"suspense={scores['suspense']:.3f}  "
              f"surprise={scores['surprise']:.3f}  "
              f"tension={scores['tension']:.3f}  "
              f"ambivalence={scores['ambivalence']:.3f}")

    # ---- Posterior at canonical ----
    _sub("Posterior (top-5 by stakes, canonical truth)")
    posterior = interrogate_posterior(ws)
    for row in posterior[:5]:
        t = row["truth_at_query"]
        print(f"  {row['proposition_id']:42s} truth={str(t):5s} "
              f"tick={row['canonical_tick']} stakes={row['stakes']:.2f} "
              f"sup={row['support_weight']:.2f}/con={row['contradict_weight']:.2f}")
    audit_warnings = audit_posterior_consistency(posterior)
    if audit_warnings:
        _sub(f"Posterior audit warnings ({len(audit_warnings)})")
        for w in audit_warnings[:10]:
            print(f"  {w}")
        if len(audit_warnings) > 10:
            print(f"  ... ({len(audit_warnings)-10} more)")
    else:
        print(f"  ✓ no posterior audit warnings")

    # ---- Run scripted queries ----
    findings: List[str] = []
    for q_spec in queries:
        _sub(f"{q_spec['label']}")
        q = q_spec["query"]
        try:
            r = calculate_narrative_physics(q, ws, use_causal_engine=True)
        except Exception as exc:
            print(f"  ✗ RAISED {type(exc).__name__}: {exc}")
            findings.append(f"{name}/{q_spec['label']}: raised {exc}")
            continue
        print(f"  {_summarize_result(r)}")
        for check_name, check_fn in q_spec.get("checks", []):
            try:
                ok, detail = check_fn(r, ws)
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
            mark = "✓" if ok else "✗"
            print(f"    {mark} {check_name}: {detail}")
            if not ok:
                findings.append(f"{name}/{q_spec['label']}/{check_name}: {detail}")

    return {"name": name, "findings": findings}


# ----------------------------------------------------------------- checks

def _check_intervened_node_changed(node_id: str):
    def _check(r, ws):
        ps = r.get("physics_state") or {}
        if not isinstance(ps, dict) or "nodes" not in ps:
            return False, "physics_state lacks 'nodes' (not engine graph)"
        for n in ps["nodes"]:
            if n.get("id") == node_id:
                et = n.get("event_type")
                return et == "prevented", f"node {node_id} event_type={et!r}"
        return False, f"node {node_id} not found in physics_state"
    return _check


def _check_proposition_mutation_contains(pid: str):
    def _check(r, ws):
        muts = r.get("proposition_mutations") or []
        for m in muts:
            if m.get("proposition_id") == pid:
                return True, f"found mutation {m}"
        return False, f"no proposition_mutation for {pid}; got {len(muts)} muts"
    return _check


def _check_past_anchor_resolved(r, ws):
    anchor = r.get("past_anchor")
    if anchor is None:
        return False, "past_anchor is None"
    if isinstance(anchor, dict):
        return True, f"fabula_time={anchor.get('fabula_time')} event={anchor.get('event_id')}"
    return True, str(anchor)


def _check_hidden_deltas_populated(r, ws):
    hd = r.get("hidden_deltas") or {}
    return bool(hd), f"hidden_deltas keys={list(hd.keys())[:5]}"


def _check_ego_includes(entity_id: str):
    def _check(r, ws):
        ps = r.get("physics_state") or {}
        if not isinstance(ps, dict):
            return False, f"physics_state not dict: {type(ps)}"
        # ObservationQuery shape: focus_entities is a list of entity dicts
        for ent in ps.get("focus_entities") or []:
            if isinstance(ent, dict) and ent.get("id") == entity_id:
                return True, f"focus_entities contains {entity_id}"
        # present_entities may be a list of entity dicts as well
        for ent in ps.get("present_entities") or []:
            if isinstance(ent, dict) and ent.get("id") == entity_id:
                return True, f"present_entities contains {entity_id}"
        # WorldStateV1-ish shape: entities is a dict id->entity
        ents = ps.get("entities")
        if isinstance(ents, dict) and entity_id in ents:
            return True, f"entities contains {entity_id}"
        # node-link form (rung-2/3 sandbox): nodes is a list of node dicts
        if "nodes" in ps:
            for n in ps["nodes"]:
                if n.get("id") == entity_id:
                    return True, f"node {entity_id} present"
        return False, f"{entity_id} missing"
    return _check


def _check_status_success(r, ws):
    return r.get("status") == "success", f"status={r.get('status')}"


# ----------------------------------------------------------------- query specs

MACBETH_QUERIES = [
    {
        "label": "Rung-1 ego: what does Macbeth see?",
        "query": ObservationQuery(focus_entity_ids=["ENT_MACBETH"]),
        "checks": [
            ("status", _check_status_success),
            ("ego includes Macbeth", _check_ego_includes("ENT_MACBETH")),
        ],
    },
    {
        "label": "Rung-2 do: prevent Duncan's murder",
        "query": InterventionQuery(
            interventions={},
            do_targets=[DoEvent(event_id="EVT_DUNCAN_MURDER", occurred=False)],
            force_implausible=True,
        ),
        "checks": [
            ("EVT_DUNCAN_MURDER prevented", _check_intervened_node_changed("EVT_DUNCAN_MURDER")),
        ],
    },
    {
        "label": "Rung-2 do: force Macbeth becomes king to False",
        "query": InterventionQuery(
            interventions={},
            do_targets=[DoProposition(proposition_id="PROP_MACBETH_BECOMES_KING", truth=False, fabula_time=10000)],
            force_implausible=True,
        ),
        "checks": [
            ("proposition_mutations PROP_MACBETH_BECOMES_KING",
             _check_proposition_mutation_contains("PROP_MACBETH_BECOMES_KING")),
        ],
    },
    {
        "label": "Rung-3 counterfactual: what if Macbeth refused the prophecy?",
        "query": CounterfactualQuery(
            historical_interventions={},
            historical_do_targets=[DoEvent(event_id="EVT_DUNCAN_MURDER", occurred=False)],
            evidence_node_ids=["EVT_MACBETH_CROWNED"],
            force_implausible=True,
        ),
        "checks": [
            ("past_anchor resolved", _check_past_anchor_resolved),
        ],
    },
]


ROMEO_QUERIES = [
    {
        "label": "Rung-1 ego: Romeo's view",
        "query": ObservationQuery(focus_entity_ids=["ENT_ROMEO"]),
        "checks": [("status", _check_status_success)],
    },
    {
        "label": "Rung-2 do: keep Friar's go-between channel open",
        "query": InterventionQuery(
            interventions={},
            do_targets=[DoChannel(channel_id="CHN_FRIAR_GO_BETWEEN", active=True)],
            force_implausible=True,
        ),
        "checks": [("status acceptable",
                    lambda r, ws: (r.get("status") in ("success", "implausible"),
                                   f"status={r.get('status')}"))],
    },
    {
        "label": "Rung-3 counterfactual: what if Juliet was not 'apparently dead'?",
        "query": CounterfactualQuery(
            historical_interventions={},
            historical_do_targets=[
                DoProposition(proposition_id="PROP_JULIET_APPEARS_DEAD",
                              truth=False, fabula_time=14000),
            ],
            evidence_node_ids=[],
            force_implausible=True,
        ),
        "checks": [("status acceptable",
                    lambda r, ws: (r.get("status") in ("success", "implausible"),
                                   f"status={r.get('status')}"))],
    },
]


GONE_GIRL_QUERIES = [
    {
        "label": "Rung-1 ego: Nick's view",
        "query": ObservationQuery(focus_entity_ids=["ENT_NICK"]),
        "checks": [("ego includes Nick", _check_ego_includes("ENT_NICK"))],
    },
    {
        "label": "Rung-2 do: Amy is exposed early",
        "query": InterventionQuery(
            interventions={},
            do_targets=[DoProposition(proposition_id="PROP_AMY_EXPOSED",
                                      truth=True, fabula_time=10000)],
            force_implausible=True,
        ),
        "checks": [("proposition_mutations PROP_AMY_EXPOSED",
                    _check_proposition_mutation_contains("PROP_AMY_EXPOSED"))],
    },
    {
        "label": "Rung-3 counterfactual: what if Amy never disappeared?",
        "query": CounterfactualQuery(
            historical_interventions={},
            historical_do_targets=[
                DoProposition(proposition_id="PROP_AMY_BETRAYS_NICK",
                              truth=False, fabula_time=2000),
            ],
            evidence_node_ids=[],
            force_implausible=True,
        ),
        "checks": [("past_anchor resolved", _check_past_anchor_resolved)],
    },
]


DOTN_QUERIES = [
    {
        "label": "Rung-1 omniscient observation",
        "query": ObservationQuery(focus_entity_ids=[]),
        "checks": [("status", _check_status_success)],
    },
    {
        "label": "Rung-2 do: prevent Linnet's murder",
        "query": InterventionQuery(
            interventions={},
            do_targets=[DoProposition(proposition_id="PROP_LINNET_DIES",
                                      truth=False, fabula_time=9000)],
            force_implausible=True,
        ),
        "checks": [("proposition_mutations PROP_LINNET_DIES",
                    _check_proposition_mutation_contains("PROP_LINNET_DIES"))],
    },
    {
        "label": "Rung-3 counterfactual: if Otterbourne had named the killer",
        "query": CounterfactualQuery(
            historical_interventions={},
            historical_do_targets=[
                DoProposition(proposition_id="PROP_OTTERBOURNE_NAMES_KILLER",
                              truth=True, fabula_time=10000),
            ],
            evidence_node_ids=["EVT_POIROT_SOLVES"],
            force_implausible=True,
        ),
        "checks": [("past_anchor resolved", _check_past_anchor_resolved)],
    },
]


FRANKENSTEIN_QUERIES = [
    {
        "label": "Rung-1 ego: Creature's view",
        "query": ObservationQuery(focus_entity_ids=["ENT_CREATURE"]),
        "checks": [("ego includes Creature", _check_ego_includes("ENT_CREATURE"))],
    },
    {
        "label": "Rung-2 do: Victor destroys the Creature's mate (already canonical) — push True",
        "query": InterventionQuery(
            interventions={},
            do_targets=[DoTrait(holder_id="ENT_VICTOR", trait_name="hubris", value=0.0)],
            force_implausible=True,
        ),
        "checks": [("status acceptable",
                    lambda r, ws: (r.get("status") in ("success", "implausible"),
                                   f"status={r.get('status')}"))],
    },
]


SUITES = [
    ("macbeth", MACBETH_QUERIES),
    ("romeo_and_juliet", ROMEO_QUERIES),
    ("gone_girl", GONE_GIRL_QUERIES),
    ("death_on_the_nile", DOTN_QUERIES),
    ("frankenstein", FRANKENSTEIN_QUERIES),
]


def main() -> None:
    all_findings: List[str] = []
    for name, queries in SUITES:
        try:
            res = audit_world(name, queries)
            all_findings.extend(res["findings"])
        except Exception as exc:  # noqa: BLE001
            print(f"\n!!! {name} AUDIT RAISED {type(exc).__name__}: {exc}")
            all_findings.append(f"{name}: top-level exception {exc}")

    _hr("AUDIT SUMMARY")
    if not all_findings:
        print("  ✓ all real-plot rung/affective/interrogation checks passed")
    else:
        print(f"  ✗ {len(all_findings)} findings:")
        for f in all_findings:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
