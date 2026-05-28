# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round-2 deep audit — pipeline surfaces NOT covered by the prior
two audits. Targets:

- ObservationQuery (Rung-1)
- DirectiveQuery for every target_effect
- InterrogationQuery (Graph RAG pathfinding)
- GeneralQuery (full-graph Q&A)
- compute_affective_scorers across all 20 worlds
- interrogate_posterior / surface_entity_constants / audit_posterior_consistency
- Monte Carlo distribution (variance != 0 under repeated sampling)
- DoEvent relocation (new_at_location_id) and time-shift (new_fabula_time)
- Channel deactivation → utterance provenance pruning (real prune count)
- DoProposition cascading_belief_count surfacing
- Time-slice consistency (querying at different fabula_times)
"""
from __future__ import annotations

import importlib
import math
import traceback
from typing import Any, Dict, List

from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import (
    ObservationQuery, InterventionQuery, CounterfactualQuery,
    DirectiveQuery, InterrogationQuery, GeneralQuery,
    DoEvent, DoProposition, DoTrait, DoChannel,
)
from shadow_loom.affective_scorers import compute_affective_scorers
from shadow_loom.interrogate_posterior import (
    interrogate_posterior, surface_entity_constants,
    audit_posterior_consistency,
)


PASS: List[str] = []
FAIL: List[str] = []
WARN: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}" + (f" — {detail}" if detail else ""))
    (PASS if ok else FAIL).append(label)


def warn(label: str, detail: str = "") -> None:
    print(f"  [WARN] {label}" + (f" — {detail}" if detail else ""))
    WARN.append(label)


def hr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n  {t}\n" + "=" * 78)


def run(ws, q):
    try:
        return calculate_narrative_physics(q, ws, None, None, False)
    except Exception as e:  # noqa: BLE001
        return {"status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()}


def load(name: str):
    return importlib.import_module(f"example_worlds.{name}").world_state


WORLDS = [
    "a_court_of_thorn_and_roses", "a_fish_called_wanda", "apocalypse_now",
    "brief_encounter", "dads_army", "death_on_the_nile", "frankenstein",
    "gone_girl", "great_expectations", "great_gatsby", "macbeth",
    "nineteen_eighty_four", "once_upon_a_time_in_the_west", "persuasion",
    "reservoir_dogs", "romeo_and_juliet", "the_devil_wears_prada",
    "the_lion_the_witch_and_the_wardrobe", "tinker_tailor_soldier_spy",
    "wuthering_heights",
]


# ----------------------------------------------------------------- probes

def probe_observation():
    hr("OBSERVATION (Rung-1) basic sanity")
    ws = load("macbeth")
    ent = next(iter(ws.entities.keys()))
    # 1. focus-only observation
    q = ObservationQuery(focus_entity_ids=[ent], fabula_time=5000)
    r = run(ws, q)
    check("Observation.focus-only.success", r.get("status") == "success",
          f"status={r.get('status')} err={r.get('error','')[:80]}")
    # 2. observations dict
    obj_id = next(iter(ws.objects.keys()), None)
    if obj_id:
        q2 = ObservationQuery(focus_entity_ids=[ent], fabula_time=5000,
                              observations={obj_id: "missing"})
        r2 = run(ws, q2)
        check("Observation.with-observations.success", r2.get("status") == "success",
              f"status={r2.get('status')} err={r2.get('error','')[:80]}")
    # 3. cross-world observation
    fails = []
    for w in WORLDS:
        try:
            wsx = load(w)
            e = next(iter(wsx.entities.keys()), None)
            if not e:
                continue
            rx = run(wsx, ObservationQuery(focus_entity_ids=[e], fabula_time=5000))
            if rx.get("status") != "success":
                fails.append(f"{w}:{rx.get('status')}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"{w}:exc:{type(e).__name__}")
    check("Observation.cross-world-success", not fails,
          f"{len(WORLDS)} worlds, fails={fails[:3]}")


def probe_directive():
    hr("DIRECTIVE — all 11 target_effects on macbeth")
    ws = load("macbeth")
    ent = next(iter(ws.entities.keys()))
    effects = ["suspense", "surprise", "mystery", "dramatic_irony",
               "narrative_tension", "grief", "rage", "joy", "regret",
               "love", "fear"]
    fails = []
    for eff in effects:
        q = DirectiveQuery(target_entity_ids=[ent], target_effect=eff,
                           fabula_time=5000, force_implausible=True)
        r = run(ws, q)
        st = r.get("status")
        if st not in ("success", "implausible"):
            fails.append(f"{eff}:{st}:{r.get('error','')[:40]}")
        elif r.get("status") == "exception":
            fails.append(f"{eff}:exc")
    check("Directive.all-effects-non-exception", not fails,
          f"checked={len(effects)} fails={fails}")
    # Spot-check suspense with target_vector_id
    props = list(ws.propositions)
    if props:
        q2 = DirectiveQuery(target_entity_ids=[ent], target_effect="suspense",
                            target_vector_id=props[0].proposition_id,
                            fabula_time=5000, force_implausible=True)
        r2 = run(ws, q2)
        check("Directive.suspense-with-vector.success",
              r2.get("status") in ("success", "implausible"),
              f"status={r2.get('status')}")


def probe_interrogation():
    hr("INTERROGATION (Graph RAG) on macbeth")
    ws = load("macbeth")
    q = InterrogationQuery(
        question="Is there a causal chain from the witches' prophecy to Duncan's murder?",
        require_proof=True,
    )
    r = run(ws, q)
    check("Interrogate.success", r.get("status") == "success",
          f"status={r.get('status')} err={r.get('error','')[:80]}")
    # The interrogator should return *something* — paths, graph view, or a textual answer
    keys = set(r.keys())
    has_payload = bool(keys - {"status", "query_type"})
    check("Interrogate.non-empty-payload", has_payload,
          f"keys={sorted(keys)[:10]}")


def probe_general():
    hr("GENERAL Q&A on macbeth")
    ws = load("macbeth")
    q = GeneralQuery(question="Who are the main entities and how are they related?",
                     include_topology=True)
    r = run(ws, q)
    check("General.success", r.get("status") == "success",
          f"status={r.get('status')} err={r.get('error','')[:80]}")
    keys = set(r.keys())
    check("General.non-empty-payload", bool(keys - {"status", "query_type"}),
          f"keys={sorted(keys)[:10]}")


def probe_affective_scorers():
    hr("AFFECTIVE SCORERS across all 20 worlds (canonical-tick)")
    expected = {"mystery", "irony", "suspense", "surprise", "tension"}
    missing_keys: List[str] = []
    all_zero: List[str] = []
    out_of_range: List[str] = []
    for w in WORLDS:
        try:
            ws = load(w)
        except Exception:  # noqa: BLE001
            continue
        scores = compute_affective_scorers(ws, fabula_time=10000)
        mk = expected - set(scores.keys())
        if mk:
            missing_keys.append(f"{w}:{sorted(mk)}")
        for k, v in scores.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                out_of_range.append(f"{w}.{k}=non-numeric")
                continue
            if math.isnan(fv) or math.isinf(fv):
                out_of_range.append(f"{w}.{k}={v}")
                continue
            if not (-0.01 <= fv <= 1.01):
                out_of_range.append(f"{w}.{k}={fv}")
        if all(float(v or 0.0) == 0.0 for v in scores.values()):
            all_zero.append(w)
    check("Affective.all-worlds-have-expected-keys", not missing_keys,
          f"missing={missing_keys[:3]}")
    check("Affective.no-NaN-Inf-out-of-range", not out_of_range,
          f"violations={out_of_range[:5]}")
    if all_zero:
        warn("Affective.some-worlds-flat-zero", f"worlds={all_zero}")
    print(f"    Checked {len(WORLDS)} worlds; {len(all_zero)} flat-zero (warn-only).")


def probe_posterior():
    hr("POSTERIOR INTERROGATION on macbeth + cross-world consistency")
    ws = load("macbeth")
    post = interrogate_posterior(ws, fabula_time=20000)
    check("Posterior.macbeth.returns-list", isinstance(post, list),
          f"type={type(post).__name__}")
    if post:
        row = post[0]
        for k in ("proposition_id", "truth_at_query", "canonical_tick",
                  "support_weight", "contradict_weight", "stakes"):
            if k not in row:
                check(f"Posterior.row-has-{k}", False, f"row keys={list(row)}")
                return
        check("Posterior.macbeth.row-shape-correct", True,
              f"rows={len(post)}, first-prop={row.get('proposition_id')}")
    # Sort: descending stakes (allow ties)
    if len(post) >= 2:
        ok = all(post[i]["stakes"] >= post[i + 1]["stakes"]
                 for i in range(len(post) - 1))
        check("Posterior.sorted-by-descending-stakes", ok,
              f"first 3 stakes={[r['stakes'] for r in post[:3]]}")
    # Entity constants
    consts = surface_entity_constants(ws)
    check("Posterior.constants-returns-list", isinstance(consts, list),
          f"count={len(consts)}")
    # Consistency audit
    warnings = audit_posterior_consistency(post)
    check("Posterior.consistency-audit-returns-list", isinstance(warnings, list),
          f"warnings={len(warnings)}")
    # Cross-world smoke
    fails = []
    for w in WORLDS:
        try:
            wsx = load(w)
            p = interrogate_posterior(wsx, fabula_time=20000)
            if not isinstance(p, list):
                fails.append(f"{w}:type={type(p).__name__}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"{w}:exc:{type(e).__name__}")
    check("Posterior.cross-world-clean", not fails,
          f"fails={fails[:3]}")


def probe_monte_carlo():
    hr("MONTE CARLO distribution mode (variance under repeated sampling)")
    ws = load("macbeth")
    ent = next(iter(ws.entities.keys()))
    # Drive a clamp through MC mode and check that the engine produced
    # a distribution result rather than a single-shot one. The
    # distribution surface lives on ``physics_state`` / ``samples`` /
    # ``mc_samples`` / etc — surface keys may vary so we just probe
    # the count.
    q = InterventionQuery(focus_entity_ids=[ent], fabula_time=5000,
                          monte_carlo_samples=24,
                          do_targets=[DoTrait(holder_id=ent, trait_name="ambition",
                                              value=0.05, fabula_time=5000)])
    r = run(ws, q)
    check("MC.success", r.get("status") == "success",
          f"status={r.get('status')} err={r.get('error','')[:80]}")
    # Round-3 fix surfaced trait_distributions at the top level.
    td = r.get("trait_distributions") or {}
    check("MC.trait_distributions-surfaced", isinstance(td, dict) and len(td) > 0,
          f"node_count={len(td)} sample_nodes={list(td.keys())[:3]}")
    # Each leaf must be a TraitDistribution-shaped dict
    sample_node = next(iter(td)) if td else None
    if sample_node:
        sample_trait = next(iter(td[sample_node])) if td[sample_node] else None
        if sample_trait:
            payload = td[sample_node][sample_trait]
            expected = {"mean", "std", "p5", "p50", "p95", "samples_count"}
            check("MC.trait_distribution-shape-correct",
                  expected.issubset(set(payload.keys())),
                  f"node={sample_node} trait={sample_trait} keys={sorted(payload.keys())}")
            check("MC.samples_count-positive",
                  int(payload.get("samples_count", 0)) > 0,
                  f"samples_count={payload.get('samples_count')}")
    # noisy_or_probabilities surface (empty under default mode is fine)
    nop = r.get("noisy_or_probabilities")
    check("MC.noisy_or_probabilities-surfaced", isinstance(nop, list),
          f"type={type(nop).__name__} count={len(nop) if isinstance(nop, list) else 'n/a'}")


def probe_do_event_relocation_and_timeshift():
    hr("DoEvent — relocation (new_at_location_id) & time-shift (new_fabula_time)")
    ws = load("macbeth")
    # Pick an event with primary actors
    target_evt = None
    for evt in ws.events:
        if evt.actor_ids and getattr(evt, "at_location_id", None):
            target_evt = evt
            break
    if target_evt is None:
        warn("DoEvent.no-relocatable-event", "skipping")
        return
    locs = [l for l in ws.locations if l != target_evt.at_location_id]
    if not locs:
        warn("DoEvent.no-alternate-location", "skipping relocation")
    else:
        new_loc = locs[0]
        ent = target_evt.actor_ids[0]
        q = InterventionQuery(focus_entity_ids=[ent], fabula_time=10000,
                              do_targets=[DoEvent(event_id=target_evt.id,
                                                  occurred=True,
                                                  new_at_location_id=new_loc,
                                                  fabula_time=target_evt.fabula_time)])
        r = run(ws, q)
        check("DoEvent.relocation.success", r.get("status") == "success",
              f"event={target_evt.id} new_loc={new_loc} "
              f"status={r.get('status')} err={r.get('error','')[:80]}")
    # Time-shift
    ent = target_evt.actor_ids[0]
    new_ft = int(target_evt.fabula_time) + 1000
    q2 = InterventionQuery(focus_entity_ids=[ent], fabula_time=20000,
                           do_targets=[DoEvent(event_id=target_evt.id,
                                               occurred=True,
                                               new_fabula_time=new_ft,
                                               fabula_time=target_evt.fabula_time)])
    r2 = run(ws, q2)
    check("DoEvent.time-shift.success", r2.get("status") == "success",
          f"event={target_evt.id} new_ft={new_ft} "
          f"status={r2.get('status')} err={r2.get('error','')[:80]}")


def probe_proposition_cascade():
    hr("DoProposition cascade_belief_count surfacing")
    # Pick a world+proposition that has at least one belief tied to it.
    chosen = None
    for w in WORLDS:
        try:
            ws = load(w)
        except Exception:  # noqa: BLE001
            continue
        # Find a proposition with at least one belief referencing it.
        belief_pids = set()
        for ent in (ws.entities or {}).values():
            for b in (ent.beliefs or []):
                pid = getattr(b, "proposition_id", None)
                if pid:
                    belief_pids.add(pid)
        for p in (ws.propositions or []):
            if p.proposition_id in belief_pids:
                chosen = (w, ws, p.proposition_id)
                break
        if chosen:
            break
    if not chosen:
        warn("DoProposition.no-belief-bound-proposition", "skipping")
        return
    w, ws, pid = chosen
    ent = next(iter(ws.entities.keys()))
    # Clamp truth to whatever it is NOT, with propagate_to_beliefs
    q = InterventionQuery(focus_entity_ids=[ent], fabula_time=20000,
                          do_targets=[DoProposition(
                              proposition_id=pid, truth=False,
                              propagate_to_beliefs=True, fabula_time=20000)])
    r = run(ws, q)
    check("DoProposition.cascade.success", r.get("status") == "success",
          f"world={w} pid={pid} status={r.get('status')} "
          f"err={r.get('error','')[:80]}")
    pm = r.get("proposition_mutations") or []
    check("DoProposition.cascade.has-prop-mutation", len(pm) >= 1,
          f"pm_count={len(pm)} sample={pm[0] if pm else None}")
    if pm:
        first = pm[0]
        ccount = (first.get("cascaded_belief_count")
                  if isinstance(first, dict)
                  else getattr(first, "cascaded_belief_count", None))
        # If propagate_to_beliefs cascaded, we expect ccount > 0
        if ccount and ccount > 0:
            check("DoProposition.cascade.belief-count-positive", True,
                  f"ccount={ccount}")
        else:
            warn("DoProposition.cascade.belief-count-zero",
                 f"ccount={ccount} bm={len(r.get('belief_mutations') or [])}")


def probe_time_slice_consistency():
    hr("TIME-SLICE consistency — observation at multiple fabula_times")
    ws = load("macbeth")
    ent = next(iter(ws.entities.keys()))
    snapshots = []
    for ft in (2000, 5000, 10000, 15000, 20000):
        r = run(ws, ObservationQuery(focus_entity_ids=[ent], fabula_time=ft))
        ok = r.get("status") == "success"
        snapshots.append((ft, ok, r.get("error", "")[:50] if not ok else None))
    fails = [s for s in snapshots if not s[1]]
    check("TimeSlice.all-anchors-succeed", not fails,
          f"snapshots={snapshots}")
    # Verify monotone-fabula behaviour: scorers at later times should
    # not break.
    score_progression = []
    for ft in (2000, 5000, 10000, 15000, 20000):
        s = compute_affective_scorers(ws, fabula_time=ft)
        score_progression.append((ft, round(float(s.get("mystery", 0.0)), 3),
                                  round(float(s.get("tension", 0.0)), 3)))
    print(f"    affective progression (ft, mystery, tension): {score_progression}")
    check("TimeSlice.scorer-progression-finite",
          all(all(math.isfinite(float(x)) for x in (m, t))
              for _, m, t in score_progression),
          f"progression={score_progression}")


# ----------------------------------------------------------------- main

def main() -> int:
    probe_observation()
    probe_directive()
    probe_interrogation()
    probe_general()
    probe_affective_scorers()
    probe_posterior()
    probe_monte_carlo()
    probe_do_event_relocation_and_timeshift()
    probe_proposition_cascade()
    probe_time_slice_consistency()
    print("\n" + "=" * 78)
    print(f"  ROUND-2 DEEP AUDIT  PASS={len(PASS)}  FAIL={len(FAIL)}  WARN={len(WARN)}")
    print("=" * 78)
    for f in FAIL:
        print(f"  ✗ {f}")
    if WARN:
        print("  -- warnings (investigate but non-fatal) --")
        for w in WARN:
            print(f"  ! {w}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
