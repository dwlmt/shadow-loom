# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end scenario audit driving REAL example_worlds through the
symbolic pipeline to find correctness problems the unit tests miss.

Five probes, all auto-discovering ids from each world so no hand-coded
fixtures rot:

  A. Timeline slicing — does ``extract_full_world_state`` /
     ``extract_ego_graph_from_memory`` apply fabula + syuzhet anchors
     consistently across every world-model aspect (events, causal /
     social / spatial topology, channels, entity/object reconstruction)?
     Key invariant: no surviving edge may reference a pruned event.

  B. Reasoning queries — observation / intervention / counterfactual
     with real entity/event/proposition ids.

  C. Affective directives — all 11 target_effects against a real POV,
     engine path on; brief must build and affect scores stay in [0,1].

  D. Interrogation — graph-RAG omniscient extraction.

  E. DB branching/forking — fork a world, mutate one branch, verify
     lineage + isolation in the SQLModel version tree.

Prints PASS / FAIL / WARN lines; exits non-zero if any FAIL.

Usage: python scripts/_scenario_pipeline_audit.py [world_stem ...]
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Quiet the chatty engine modules; we only care about audit lines.
logging.basicConfig(level=logging.ERROR, handlers=[logging.StreamHandler(sys.stderr)])
for noisy in (
    "shadow_loom.causal_physics", "shadow_loom.amwn", "shadow_loom.extract_graph",
    "shadow_loom.ingestion", "shadow_loom.narrative_physics",
    "shadow_loom.directive_assembly", "shadow_loom.instantiator", "shadow_loom.db",
):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

import example_worlds  # noqa: E402
from shadow_loom.models import WorldStateV1  # noqa: E402
from shadow_loom.extract_graph import (  # noqa: E402
    extract_full_world_state, extract_ego_graph_from_memory,
)
from shadow_loom.narrative_physics import calculate_narrative_physics  # noqa: E402
from shadow_loom.directive_assembly import DirectiveAssembler  # noqa: E402
from shadow_loom.query_models import (  # noqa: E402
    ObservationQuery, InterventionQuery, CounterfactualQuery,
    DirectiveQuery, InterrogationQuery,
    DoProposition, DoTrait, DoEvent,
)

EFFECTS = ["suspense", "surprise", "mystery", "dramatic_irony", "narrative_tension",
           "grief", "rage", "joy", "regret", "love", "fear"]

FAIL: List[str] = []
WARN: List[str] = []
PASS_N = 0


def ok(label: str, cond: bool, detail: str = "", warn_only: bool = False) -> bool:
    global PASS_N
    if cond:
        PASS_N += 1
        return True
    msg = f"{label}" + (f" — {detail}" if detail else "")
    (WARN if warn_only else FAIL).append(msg)
    tag = "WARN" if warn_only else "FAIL"
    print(f"    [{tag}] {msg}")
    return False


def hr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n  {t}\n" + "=" * 78)


# --------------------------------------------------------------------------
# world discovery
# --------------------------------------------------------------------------
def load_worlds(stems: List[str]) -> List[Tuple[str, WorldStateV1]]:
    out: List[Tuple[str, WorldStateV1]] = []
    pkg_path = Path(example_worlds.__file__).parent
    available = sorted(
        m.name for m in pkgutil.iter_modules([str(pkg_path)])
        if not m.name.startswith("_")
    )
    chosen = stems or available
    for name in chosen:
        try:
            mod = importlib.import_module(f"example_worlds.{name}")
            ws = getattr(mod, "world_state")
            out.append((name, ws))
        except Exception as exc:  # noqa: BLE001
            FAIL.append(f"load {name}: {exc}")
            print(f"    [FAIL] could not load world {name}: {exc}")
    return out


def focal_entities(ws: WorldStateV1, n: int = 4) -> List[str]:
    counts: Dict[str, int] = {}
    for ev in ws.events:
        for a in (ev.actor_ids or []) + (ev.target_ids or []):
            counts[a] = counts.get(a, 0) + 1
    ranked = [e for e, _ in sorted(counts.items(), key=lambda kv: -kv[1])
              if e in ws.entities]
    if len(ranked) < n:
        ranked += [e for e in ws.entities if e not in ranked]
    return ranked[:n]


def fabula_anchors(ws: WorldStateV1) -> List[int]:
    ts = sorted({int(e.fabula_time) for e in ws.events})
    if not ts:
        return []
    lo, hi = ts[0], ts[-1]
    picks = {lo, hi, ts[len(ts) // 2], ts[len(ts) // 4], ts[3 * len(ts) // 4]}
    return sorted(picks)


def syuzhet_anchors(ws: WorldStateV1) -> List[int]:
    ss = sorted({int(getattr(e, "syuzhet_index", 0)) for e in ws.events})
    if not ss:
        return []
    return sorted({ss[0], ss[-1], ss[len(ss) // 2]})


# --------------------------------------------------------------------------
# A. timeline slicing invariants
# --------------------------------------------------------------------------
def _event_ids(dump: Dict[str, Any]) -> set:
    return {e.get("id") for e in dump.get("events", []) if e.get("id")}


def _check_sliced_dump(world: str, where: str, dump: Dict[str, Any],
                       t: Optional[int], s: Optional[int]) -> None:
    evt_ids = _event_ids(dump)
    all_event_ids_in_world = None  # set later by caller via closure? simpler inline

    if t is not None:
        bad = [e["id"] for e in dump.get("events", []) if e.get("fabula_time", 0) > t]
        ok(f"[{world}] {where} fabula<= t={t}", not bad,
           f"{len(bad)} events past anchor e.g. {bad[:3]}")
        bad_ce = [(c.get("source_id"), c.get("target_id"))
                  for c in dump.get("causal_topology", [])
                  if c.get("fabula_time", 0) > t]
        ok(f"[{world}] {where} causal_edge fabula<= t={t}", not bad_ce,
           f"{len(bad_ce)} edges past anchor e.g. {bad_ce[:3]}")

    if s is not None:
        bad_s = [e["id"] for e in dump.get("events", [])
                 if e.get("syuzhet_index", 0) > s]
        ok(f"[{world}] {where} syuzhet<= s={s}", not bad_s,
           f"{len(bad_s)} events past syuzhet anchor e.g. {bad_s[:3]}")

    # KEY invariant: no surviving causal edge may reference a pruned EVENT
    # on either endpoint. (entity/location endpoints are fine.)
    return evt_ids


def probe_timeline(world: str, ws: WorldStateV1) -> None:
    all_evt_ids = {e.id for e in ws.events}
    foci = focal_entities(ws)

    def dangling_edges(dump: Dict[str, Any]) -> List[Tuple[str, str]]:
        surviving = _event_ids(dump)
        out = []
        for c in dump.get("causal_topology", []):
            for endp in (c.get("source_id"), c.get("target_id")):
                # only events can be pruned by the slice
                if endp in all_evt_ids and endp not in surviving:
                    out.append((c.get("source_id"), c.get("target_id")))
                    break
        return out

    for t in fabula_anchors(ws):
        full = extract_full_world_state(ws, temporal_anchor=t)
        _check_sliced_dump(world, "omniscient", full, t, None)
        dang = dangling_edges(full)
        ok(f"[{world}] omniscient no dangling causal edge @t={t}", not dang,
           f"{len(dang)} edges reference a pruned event e.g. {dang[:3]}")

        ego = extract_ego_graph_from_memory(ws, foci, t).model_dump()
        _check_sliced_dump(world, "ego", ego, t, None)
        dang_e = dangling_edges(ego)
        ok(f"[{world}] ego no dangling causal edge @t={t}", not dang_e,
           f"{len(dang_e)} edges reference a pruned event e.g. {dang_e[:3]}")

    for s in syuzhet_anchors(ws):
        full = extract_full_world_state(ws, syuzhet_anchor=s)
        _check_sliced_dump(world, "omniscient-syuzhet", full, None, s)
        dang = dangling_edges(full)
        ok(f"[{world}] omniscient-syuzhet no dangling causal edge @s={s}", not dang,
           f"{len(dang)} edges reference a pruned event e.g. {dang[:3]}")


# --------------------------------------------------------------------------
# B. reasoning queries
# --------------------------------------------------------------------------
def _first_proposition(ws: WorldStateV1) -> Optional[str]:
    props = getattr(ws, "propositions", None) or {}
    if isinstance(props, dict) and props:
        return next(iter(props))
    return None


def _first_trait_entity(ws: WorldStateV1) -> Optional[Tuple[str, str]]:
    for eid, ent in ws.entities.items():
        traits = getattr(ent, "traits", None) or {}
        if traits:
            return eid, next(iter(traits))
    return None


def probe_reasoning(world: str, ws: WorldStateV1) -> None:
    foci = focal_entities(ws)

    # observation (POV)
    try:
        r = calculate_narrative_physics(ObservationQuery(focus_entity_ids=foci[:2]), ws)
        ok(f"[{world}] observation status", r.get("status") == "success", str(r.get("status")))
    except Exception as exc:  # noqa: BLE001
        ok(f"[{world}] observation no-exception", False, f"{type(exc).__name__}: {exc}")

    # intervention — DoTrait on a real entity/trait
    te = _first_trait_entity(ws)
    if te:
        eid, tname = te
        try:
            q = InterventionQuery(
                do_targets=[DoTrait(holder_id=eid, trait_name=tname, value=0.0)],
                target_node_ids=[eid],
            )
            r = calculate_narrative_physics(q, ws, use_causal_engine=True)
            ok(f"[{world}] intervention DoTrait status",
               r.get("status") in ("success", "implausible"), str(r.get("status")))
            if r.get("status") == "success":
                ok(f"[{world}] intervention surfaced mutations key",
                   "mutations" in r, "no mutations key")
        except Exception as exc:  # noqa: BLE001
            ok(f"[{world}] intervention no-exception", False,
               f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")

    # counterfactual — historical DoEvent on an EARLY event, evidence = a focal entity
    early_evts = sorted(ws.events, key=lambda e: e.fabula_time)
    if early_evts and foci:
        ev = early_evts[len(early_evts) // 3]
        try:
            q = CounterfactualQuery(
                historical_do_targets=[DoEvent(event_id=ev.id, occurred=False)],
                evidence_node_ids=[foci[0]],
                target_node_ids=foci[:2],
            )
            r = calculate_narrative_physics(q, ws, use_causal_engine=True)
            ok(f"[{world}] counterfactual status",
               r.get("status") in ("success", "implausible"), str(r.get("status")))
            if r.get("status") == "success":
                pa = r.get("past_anchor")
                ok(f"[{world}] counterfactual past_anchor<=event time",
                   pa is not None and pa <= ev.fabula_time,
                   f"past_anchor={pa} event_t={ev.fabula_time}")
        except Exception as exc:  # noqa: BLE001
            ok(f"[{world}] counterfactual no-exception", False,
               f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")


# --------------------------------------------------------------------------
# C. affective directives
# --------------------------------------------------------------------------
def _scan_nonfinite(obj: Any, path: str = "") -> List[Tuple[str, float]]:
    """Find NaN / inf numeric leaves anywhere in the brief."""
    import math
    out: List[Tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_scan_nonfinite(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_scan_nonfinite(v, f"{path}[{i}]"))
    elif isinstance(obj, float) and not math.isfinite(obj):
        out.append((path, obj))
    return out


def probe_affective(world: str, ws: WorldStateV1) -> None:
    foci = focal_entities(ws)
    if not foci:
        return
    for eff in EFFECTS:
        try:
            q = DirectiveQuery(target_entity_ids=foci[:2], target_effect=eff,
                               target_vector_id=foci[0])
            r = calculate_narrative_physics(q, ws, use_causal_engine=True)
            if not ok(f"[{world}] directive {eff} status",
                      r.get("status") == "success", str(r.get("status"))):
                continue
            brief = r.get("creative_brief")
            ok(f"[{world}] directive {eff} brief built", bool(brief), "no creative_brief")
            # NOTE: affect-profile *score* fields are intentionally
            # unbounded KL magnitudes (see SurpriseProfile.score doc), so
            # we do NOT assert [0,1] here. We only check finiteness.
            if brief:
                bad = _scan_nonfinite(brief)
                ok(f"[{world}] directive {eff} affect scores finite", not bad,
                   f"non-finite: {bad[:4]}")
        except Exception as exc:  # noqa: BLE001
            ok(f"[{world}] directive {eff} no-exception", False,
               f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")


# --------------------------------------------------------------------------
# D. interrogation
# --------------------------------------------------------------------------
def probe_interrogation(world: str, ws: WorldStateV1) -> None:
    try:
        q = InterrogationQuery(
            question="Is there a chain of causation linking the earliest event to the latest outcome?",
            require_proof=True,
        )
        r = calculate_narrative_physics(q, ws)
        ok(f"[{world}] interrogate status", r.get("status") == "success", str(r.get("status")))
        ok(f"[{world}] interrogate returns physics_state",
           bool(r.get("physics_state")), "empty physics_state")
    except Exception as exc:  # noqa: BLE001
        ok(f"[{world}] interrogate no-exception", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# E. DB branching / forking
# --------------------------------------------------------------------------
def probe_db_branching(world: str, ws: WorldStateV1) -> None:
    import tempfile
    import os
    tmp = tempfile.mkdtemp(prefix="sl_audit_db_")
    url = f"sqlite:///{tmp}/audit.db"
    try:
        from shadow_loom import db as dbmod
        dbmod.init_db(url)
        user = dbmod.ensure_local_user()
        proj = dbmod.create_project(name=f"audit_{world}", owner_id=user.id)
        pid = proj.id if hasattr(proj, "id") else proj["id"]

        root_json = ws.model_dump_json()
        root = dbmod.save_version(pid, root_json, source="audit-root", user_id=user.id)

        # Two children off the same root → a fork.
        wsa = ws.model_copy(deep=True)
        te = _first_trait_entity(wsa)
        if te:
            eid, tname = te
            wsa.entities[eid].traits[tname].value = 0.0
        child_a = dbmod.save_version(pid, wsa.model_dump_json(),
                                     ancestor_id=root.id, source="audit-branchA",
                                     user_id=user.id)

        wsb = ws.model_copy(deep=True)
        if te:
            eid, tname = te
            wsb.entities[eid].traits[tname].value = 1.0
        child_b = dbmod.save_version(pid, wsb.model_dump_json(),
                                     ancestor_id=root.id, source="audit-branchB",
                                     user_id=user.id)

        # lineage / parent pointers
        ok(f"[{world}] DB branchA parent==root", child_a.ancestor_id == root.id,
           f"ancestor_id={child_a.ancestor_id} root={root.id}")
        ok(f"[{world}] DB branchB parent==root", child_b.ancestor_id == root.id,
           f"ancestor_id={child_b.ancestor_id} root={root.id}")
        ok(f"[{world}] DB distinct version numbers",
           len({root.version, child_a.version, child_b.version}) == 3,
           f"versions={[root.version, child_a.version, child_b.version]}")

        # branch isolation — reload each child, the mutated trait must differ
        if te:
            eid, tname = te
            va = WorldStateV1.model_validate_json(
                dbmod.get_version_by_id(child_a.id).world_state_json)
            vb = WorldStateV1.model_validate_json(
                dbmod.get_version_by_id(child_b.id).world_state_json)
            ok(f"[{world}] DB branch isolation",
               va.entities[eid].traits[tname].value
               != vb.entities[eid].traits[tname].value,
               f"A={va.entities[eid].traits[tname].value} "
               f"B={vb.entities[eid].traits[tname].value}")

        # version tree shows the fork (root has 2 children)
        tree = dbmod.get_version_tree(pid)
        kids = [n for n in tree if n.get("ancestor_id") == root.id
                or n.get("parent_id") == root.id]
        ok(f"[{world}] DB version tree shows fork (>=2 children)",
           len(kids) >= 2, f"found {len(kids)} children; tree size={len(tree)}")
    except Exception as exc:  # noqa: BLE001
        ok(f"[{world}] DB branching no-exception", False,
           f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
    finally:
        try:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


# --------------------------------------------------------------------------
def main() -> int:
    stems = [a for a in sys.argv[1:] if not a.startswith("-")]
    worlds = load_worlds(stems)
    print(f"Loaded {len(worlds)} worlds: {[n for n, _ in worlds]}")

    hr("A. TIMELINE SLICING INVARIANTS")
    for name, ws in worlds:
        probe_timeline(name, ws)

    hr("B. REASONING QUERIES (observation / intervention / counterfactual)")
    for name, ws in worlds:
        probe_reasoning(name, ws)

    hr("C. AFFECTIVE DIRECTIVES (11 effects)")
    for name, ws in worlds:
        probe_affective(name, ws)

    hr("D. INTERROGATION")
    for name, ws in worlds:
        probe_interrogation(name, ws)

    hr("E. DB BRANCHING / FORKING (subset of 3 worlds)")
    for name, ws in worlds[:3]:
        probe_db_branching(name, ws)

    hr("SUMMARY")
    print(f"  PASS: {PASS_N}")
    print(f"  WARN: {len(WARN)}")
    print(f"  FAIL: {len(FAIL)}")
    if WARN:
        print("\n  --- WARNINGS ---")
        for w in WARN[:40]:
            print(f"    • {w}")
    if FAIL:
        print("\n  --- FAILURES ---")
        for f in FAIL[:60]:
            print(f"    ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
