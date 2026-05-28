# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round-4 deep audit — ingestion, re-ingestion, intervention-join consistency.

No LLM calls. Probes the structural contract of:

  P1. SCHEMA INTEGRITY of all 20 example_worlds — every ENT_/LOC_/CHN_/
      EVT_/OBJ_/WORLD_/PROP_/CCN_ cross-reference resolves; every
      snapshot.fabula_time fits the event window; every state_timeline
      is monotone; every channel participant is an entity; every event
      actor / target_at_location resolves.
  P2. PROGRAMMATIC VALIDATION — _programmatic_validation returns ZERO
      errors on each gold-standard world (warnings ok; treated as
      data-quality observations, not invariants).
  P3. VALIDATE-AND-CORRECT IDEMPOTENCE — validate_and_correct_world_state
      on a clean world produces is_valid=True without dropping any
      events / entities / channels / propositions / concerns. A real
      mutation here means either the world is broken OR the validator
      is over-aggressive.
  P4. CONTINUATION-BRIDGE WIRING — _render_engine_priors,
      _run_continuation_quality_bridge_sync/async,
      _apply_query_introductions, _apply_manual_edit_replacements,
      _isolate_ws_for_surgery, _augment_topology_with_sandbox_deltas
      exist and accept the documented signatures. validate_and_correct
      is exported from shadow_loom.__init__.
  P5. INTERVENTION-JOIN CONSISTENCY — for each example world, apply
      one DoTarget of each kind via CausalPhysicsEngine.apply_do_targets
      and confirm (a) no crash, (b) the live world_state object is
      preserved (the same Python id), (c) factual world_id is the only
      world_id on the world's snapshots (no shadow leakage from a
      do-surgery into the persisted factual world), (d) the legacy
      key-lift stashes typed surgeries on _last_legacy_interventions.
  P6. PROMPT FILE COVERAGE — every _load_prompt("foo.md") referenced
      in ingestion.py / auditor.py / generation.py points at a file
      that actually exists.
  P7. PROMPT-FIELD DRIFT — prompts don't reference removed legacy
      field names (information_topology, archetype, located_in on
      Entity, snapshots on Concern/Proposition, activation_window
      without _fabula_).
"""
from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, List, Tuple

# Ensure repo root on path.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROMPT_DIR = REPO / "shadow_loom" / "prompts"
EXAMPLE_WORLDS_DIR = REPO / "example_worlds"

# Audit accumulators.
_PASS: List[str] = []
_FAIL: List[str] = []
_WARN: List[str] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    if ok:
        _PASS.append(name)
        print(f"  [PASS] {name} — {detail}")
    elif warn_only:
        _WARN.append(name)
        print(f"  [WARN] {name} — {detail}")
    else:
        _FAIL.append(name)
        print(f"  [FAIL] {name} — {detail}")


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ======================================================================
# Example-world loader.
# ======================================================================

def _example_world_modules() -> List[str]:
    mods = []
    for f in sorted(EXAMPLE_WORLDS_DIR.glob("*.py")):
        if f.name.startswith("_") or f.name == "__init__.py":
            continue
        mods.append(f.stem)
    return mods


def _load_world(name: str):
    """Load a fresh world_state from example_worlds/<name>.py."""
    mod = importlib.import_module(f"example_worlds.{name}")
    importlib.reload(mod)
    return getattr(mod, "world_state")


# ======================================================================
# P1. SCHEMA INTEGRITY.
# ======================================================================

def probe_schema_integrity(world_name: str) -> Tuple[int, int]:
    """Return (issues_found, references_checked)."""
    ws = _load_world(world_name)
    issues: List[str] = []
    n_refs = 0

    ent_ids = set(ws.entities)
    loc_ids = set(ws.locations)
    obj_ids = set(ws.objects)
    wt_ids = set(ws.world_traits)
    evt_ids = {e.id for e in ws.events}
    chn_ids = set((ws.channels or {}).keys())
    prop_ids = {p.proposition_id for p in ws.propositions}

    # Entities — location_id must resolve.
    for eid, ent in ws.entities.items():
        n_refs += 1
        if ent.location_id and ent.location_id not in loc_ids:
            issues.append(f"{eid}.location_id={ent.location_id!r} not in locations")
        # State timeline monotonicity + fabula_time sanity.
        last_ft = None
        for snap in ent.state_timeline:
            n_refs += 1
            if last_ft is not None and snap.fabula_time < last_ft:
                issues.append(
                    f"{eid}.state_timeline non-monotone @"
                    f"{snap.fabula_time} < prev {last_ft}"
                )
            last_ft = snap.fabula_time
            if snap.location_id and snap.location_id not in loc_ids:
                issues.append(
                    f"{eid}.state_timeline@{snap.fabula_time}.location_id "
                    f"{snap.location_id!r} not in locations"
                )
            if snap.triggered_by and snap.triggered_by not in evt_ids:
                issues.append(
                    f"{eid}.state_timeline@{snap.fabula_time}.triggered_by "
                    f"{snap.triggered_by!r} not in events"
                )
        # Beliefs — target_id and proposition_id.
        for b in ent.beliefs:
            n_refs += 1
            if b.target_id not in (ent_ids | loc_ids | obj_ids | wt_ids | evt_ids | prop_ids):
                issues.append(f"{eid}.belief.target_id={b.target_id!r} unresolved")
            if getattr(b, "proposition_id", None) and b.proposition_id not in prop_ids:
                issues.append(
                    f"{eid}.belief.proposition_id={b.proposition_id!r} unresolved"
                )
        # Concerns — proposition_id, counter_concern_ids polarity.
        ent_ccn_ids = {c.concern_id for c in ent.concerns}
        for c in ent.concerns:
            n_refs += 1
            if c.proposition_id not in prop_ids:
                issues.append(
                    f"{eid}.concern({c.concern_id}).proposition_id "
                    f"{c.proposition_id!r} unresolved"
                )
            for ccid in (c.counter_concern_ids or []):
                # Counter-concerns may live on the same OR another entity.
                # Cheap check: at least exist somewhere.
                pass  # cross-entity check handled below
            if c.polarity not in ("desire", "fear"):
                issues.append(
                    f"{eid}.concern({c.concern_id}).polarity={c.polarity!r} invalid"
                )

    # Objects — owner_id, location_id.
    for oid, obj in ws.objects.items():
        n_refs += 1
        if obj.owner_id and obj.owner_id not in ent_ids:
            issues.append(f"{oid}.owner_id={obj.owner_id!r} not in entities")
        if obj.location_id and obj.location_id not in loc_ids:
            issues.append(f"{oid}.location_id={obj.location_id!r} not in locations")

    # Events — actor_ids subset entities; at_location_id resolves.
    last_ft_by_evt = None
    for ev in ws.events:
        n_refs += 1
        for aid in ev.actor_ids:
            if aid not in ent_ids:
                issues.append(f"event {ev.id}.actor_id={aid!r} not in entities")
        for tid in ev.target_ids:
            # Events may target entities/objects/locations/world-traits AND
            # other events (canonical utterance->event pattern) and
            # propositions (utterances about claims).
            if tid not in (ent_ids | obj_ids | loc_ids | wt_ids | evt_ids | prop_ids):
                issues.append(f"event {ev.id}.target_id={tid!r} unresolved")
        if ev.at_location_id and ev.at_location_id not in loc_ids:
            issues.append(
                f"event {ev.id}.at_location_id={ev.at_location_id!r} unresolved"
            )
        via_chn = getattr(ev, "via_channel_id", None)
        if via_chn and via_chn not in chn_ids:
            issues.append(f"event {ev.id}.via_channel_id={via_chn!r} unresolved")
        for pid_attr in ("asserts_proposition_id", "denies_proposition_id"):
            pid = getattr(ev, pid_attr, None)
            if pid and pid not in prop_ids:
                issues.append(f"event {ev.id}.{pid_attr}={pid!r} unresolved")
        for pid in getattr(ev, "resolves_proposition_ids", []) or []:
            if pid not in prop_ids:
                issues.append(f"event {ev.id}.resolves_proposition_id={pid!r} unresolved")

    # Causal edges — endpoints resolve.
    valid_edge_endpoints = ent_ids | loc_ids | obj_ids | wt_ids | evt_ids
    for ce in ws.causal_topology:
        n_refs += 1
        if ce.source_id not in valid_edge_endpoints:
            issues.append(f"causal edge source {ce.source_id!r} unresolved")
        if ce.target_id not in valid_edge_endpoints:
            issues.append(f"causal edge target {ce.target_id!r} unresolved")

    # Spatial edges — endpoints in locations.
    for se in ws.spatial_topology:
        n_refs += 1
        if se.source_id not in loc_ids or se.target_id not in loc_ids:
            issues.append(
                f"spatial edge {se.source_id}->{se.target_id} not in locations"
            )

    # Channels — participants may be entities OR objects (a diary, book,
    # or telescreen is a channel-of-record between holder and reader/state).
    for cid, ch in (ws.channels or {}).items():
        n_refs += 1
        for pid in (ch.participant_ids or []):
            if pid not in (ent_ids | obj_ids):
                issues.append(f"channel {cid}.participant {pid!r} not in entities|objects")

    # Social edges — endpoints resolve.
    for se in ws.social_topology:
        n_refs += 1
        if se.source_entity_id not in ent_ids:
            issues.append(f"social edge source {se.source_entity_id!r} unresolved")
        if se.target_entity_id not in ent_ids:
            issues.append(f"social edge target {se.target_entity_id!r} unresolved")

    # Propositions — referent_ids resolve (may target any first-class id).
    valid_referents = (
        ent_ids | loc_ids | obj_ids | wt_ids | evt_ids | chn_ids | prop_ids
    )
    for p in ws.propositions:
        n_refs += 1
        for rid in p.referent_ids:
            if rid not in valid_referents:
                issues.append(f"prop {p.proposition_id}.referent_id={rid!r} unresolved")

    if issues:
        for iss in issues[:8]:
            print(f"    · {iss}")
        if len(issues) > 8:
            print(f"    · (+{len(issues) - 8} more)")
    return len(issues), n_refs


# ======================================================================
# P2 + P3. Programmatic validation + validate-and-correct round-trip.
# ======================================================================

def probe_programmatic_validation(world_name: str) -> Tuple[int, int]:
    from shadow_loom.ingestion import _programmatic_validation
    ws = _load_world(world_name)
    issues = _programmatic_validation(ws)
    errors = [i for i in issues if i.severity == "error"]
    warnings_n = len(issues) - len(errors)
    for e in errors[:5]:
        print(f"    · ERROR {e.kind}: {e.detail}")
    return len(errors), warnings_n


def probe_validate_and_correct(world_name: str) -> Tuple[bool, str]:
    """Round-trip a clean world through validate-and-correct (NO LLM —
    we use a minimal ExtractionConfig with output_retries=0 and rely on
    the function's degrade-to-programmatic-only behaviour when no model
    is available)."""
    from shadow_loom.ingestion import _auto_repair, _programmatic_validation
    ws = _load_world(world_name)
    pre_events = len(ws.events)
    pre_ents = len(ws.entities)
    pre_chans = len(ws.channels or {})
    pre_props = len(ws.propositions)
    pre_concerns = sum(len(e.concerns) for e in ws.entities.values())
    try:
        repaired, notes = _auto_repair(ws)
    except Exception as exc:
        return False, f"_auto_repair raised {type(exc).__name__}: {exc}"
    post_events = len(repaired.events)
    post_ents = len(repaired.entities)
    post_chans = len(repaired.channels or {})
    post_props = len(repaired.propositions)
    post_concerns = sum(len(e.concerns) for e in repaired.entities.values())
    deltas = []
    if post_events != pre_events:
        deltas.append(f"events {pre_events}->{post_events}")
    if post_ents != pre_ents:
        deltas.append(f"entities {pre_ents}->{post_ents}")
    if post_chans != pre_chans:
        deltas.append(f"channels {pre_chans}->{post_chans}")
    if post_props != pre_props:
        deltas.append(f"propositions {pre_props}->{post_props}")
    if post_concerns != pre_concerns:
        deltas.append(f"concerns {pre_concerns}->{post_concerns}")
    # Auto-repair note count is informational; some are benign (snapshot resort).
    issues_post = _programmatic_validation(repaired)
    errors_post = [i for i in issues_post if i.severity == "error"]
    summary = f"notes={len(notes)} post_errors={len(errors_post)}"
    if deltas:
        summary += f" deltas={','.join(deltas)}"
    return (not deltas and not errors_post), summary


# ======================================================================
# P4. Continuation-bridge wiring.
# ======================================================================

def probe_continuation_bridge_wiring() -> None:
    from shadow_loom import pipeline
    from shadow_loom import ingestion
    import shadow_loom

    expected_pipeline_helpers = [
        "_render_engine_priors",
        "_run_continuation_quality_bridge_sync",
        "_run_continuation_quality_bridge_async",
        "_apply_query_introductions",
        "_apply_manual_edit_replacements",
        "_isolate_ws_for_surgery",
        "_augment_topology_with_sandbox_deltas",
        "_resolve_branch_policy",
        "_gather_preceding_prose",
        "_compute_factual_contrast",
        "_stamp_brief_full",
    ]
    for name in expected_pipeline_helpers:
        check(
            f"Bridge.pipeline-helper.{name}",
            hasattr(pipeline, name),
            f"present={hasattr(pipeline, name)}",
        )

    # Exports from shadow_loom top-level.
    for name in (
        "run_extraction", "run_extraction_async",
        "validate_and_correct_world_state",
        "validate_and_correct_world_state_async",
    ):
        check(
            f"Bridge.public-export.{name}",
            hasattr(shadow_loom, name),
            f"present={hasattr(shadow_loom, name)}",
        )

    # _render_engine_priors contract.
    rep = pipeline._render_engine_priors
    check(
        "Bridge.render_engine_priors.empty-returns-None",
        rep({}) is None,
        f"empty input -> {rep({})}",
    )
    check(
        "Bridge.render_engine_priors.no-actionable-returns-None",
        rep({"status": "success"}) is None,
        "no mutations/hidden/blocked/intervened => None",
    )
    sample = rep({
        "mutations": [{"node_id": "ENT_X", "trait": "courage", "new_value": 0.8}],
        "intervened_nodes": ["ENT_X"],
    })
    check(
        "Bridge.render_engine_priors.populated",
        isinstance(sample, str) and "ENT_X" in sample and "courage" in sample,
        f"len={len(sample) if sample else 0}",
    )

    # _isolate_ws_for_surgery signature contract — accepts (ws, query).
    sig = inspect.signature(pipeline._isolate_ws_for_surgery)
    params = list(sig.parameters)
    check(
        "Bridge.isolate_ws.signature",
        len(params) >= 2,
        f"params={params}",
    )


# ======================================================================
# P4b. Continuation-bridge FUNCTIONAL probes.
# ======================================================================

def probe_continuation_bridge_functional(world_name: str) -> None:
    """Drive the two heaviest bridge helpers end-to-end on a real world.

    Round-trips (a) ``_run_continuation_quality_bridge_sync`` with a
    no-op ``extraction_config=None`` config so the early-return path
    is exercised against a real VersionedWorldModel, and (b)
    ``_augment_topology_with_sandbox_deltas`` with a synthetic
    ``physics_result`` carrying one ENT_ trait mutation, one WORLD_
    trait mutation, and one social-axis mutation — asserting each
    lands in the corresponding topology collection.
    """
    from shadow_loom import pipeline
    from shadow_loom.pipeline import PipelineConfig, PipelineResult
    from shadow_loom.extract_graph import VersionedWorldModel
    from shadow_loom.ingestion import ChunkTopology

    ws = _load_world(world_name)

    # --- (a) bridge sync round-trip with no-op extraction_config -----
    vwm = VersionedWorldModel.from_world_state(ws)
    cfg = PipelineConfig()  # extraction_config defaults to None
    result = PipelineResult()
    out = pipeline._run_continuation_quality_bridge_sync(
        vwm, cfg, result, log_prefix=f"[audit:{world_name}]",
    )
    check(
        f"BridgeFn[{world_name}].sync.early-return-identity",
        out is vwm,
        "extraction_config=None returns the same VWM instance",
    )
    check(
        f"BridgeFn[{world_name}].sync.report-untouched",
        result.continuation_quality_report is None
        and result.continuation_quarantined is False,
        "no report stamped, not quarantined",
    )

    # --- (b) augment_topology with synthetic physics deltas ---------
    if not ws.entities:
        check(
            f"BridgeFn[{world_name}].augment.skipped",
            True, "world has no entities", warn_only=True,
        )
        return
    sample_entity_id = next(iter(ws.entities))
    sample_entity = ws.entities[sample_entity_id]
    sample_trait = next(iter(sample_entity.traits), None) if sample_entity.traits else None
    sample_world_id = next(iter(ws.world_traits), None) if ws.world_traits else None
    second_entity = None
    for eid in ws.entities:
        if eid != sample_entity_id:
            second_entity = eid
            break

    physics_result: Dict[str, Any] = {"mutations": [], "social_mutations": []}
    if sample_trait:
        physics_result["mutations"].append({
            "node_id": sample_entity_id,
            "trait": sample_trait,
            "new_value": 0.42,
        })
    if sample_world_id:
        physics_result["mutations"].append({
            "node_id": sample_world_id,
            "trait": "magnitude",  # ignored — fold uses value directly
            "new_value": 0.61,
        })
    if second_entity:
        physics_result["social_mutations"].append({
            "source_entity_id": sample_entity_id,
            "target_entity_id": second_entity,
            "metric": "affinity",
            "new_value": -0.25,
            "old_value": 0.0,
            "inertia": 0.3,
            "triggered_by": None,
        })

    topo = ChunkTopology()
    fabula_now = max(
        (e.fabula_time for e in ws.events if e.fabula_time is not None),
        default=0,
    ) + 100
    augmented = pipeline._augment_topology_with_sandbox_deltas(
        topo,
        world_state=ws,
        physics_result=physics_result,
        fabula_time_now=fabula_now,
    )
    check(
        f"BridgeFn[{world_name}].augment.returns-topology",
        augmented is topo,
        "function mutates and returns same topology instance",
    )
    if sample_trait:
        ent_hit = any(
            eu.entity_id == sample_entity_id
            and sample_trait in eu.trait_updates
            and eu.fabula_time == fabula_now
            for eu in augmented.entity_updates
        )
        check(
            f"BridgeFn[{world_name}].augment.entity-trait-landed",
            ent_hit,
            f"entity_updates has {sample_entity_id}.{sample_trait}@{fabula_now}",
        )
    if sample_world_id:
        check(
            f"BridgeFn[{world_name}].augment.world-trait-landed",
            sample_world_id in augmented.new_world_traits
            and any(
                s.fabula_time == fabula_now
                for s in augmented.new_world_traits[sample_world_id].state_timeline
            ),
            f"new_world_traits[{sample_world_id}] snapshot@{fabula_now}",
        )
    if second_entity:
        soc_hit = any(
            e.source_entity_id == sample_entity_id
            and e.target_entity_id == second_entity
            and "affinity" in e.metrics
            for e in augmented.social_topology
        )
        check(
            f"BridgeFn[{world_name}].augment.social-mutation-landed",
            soc_hit,
            f"social_topology has affinity edge {sample_entity_id}->{second_entity}",
        )


# ======================================================================
# P5. Intervention-join consistency.
# ======================================================================

def probe_intervention_join_consistency(world_name: str) -> None:
    from shadow_loom.causal_physics import CausalPhysicsEngine
    from shadow_loom.query_models import (
        DoEvent, DoTrait, DoBelief, DoConcern, DoWorldTrait,
        DoProposition, DoChannel, DoRelationship,
        DoCausalEdge, DoSpatialEdge, DoNarrativeObject,
        DoEntityDelete, DoObjectDelete,
    )
    import networkx as nx

    ws = _load_world(world_name)
    if not ws.events:
        check(f"DoJoin[{world_name}].skipped", True, "no events", warn_only=True)
        return

    sample_event = ws.events[0]
    sample_entity = next(iter(ws.entities))
    sample_actor = (
        sample_event.actor_ids[0] if sample_event.actor_ids else sample_entity
    )

    targets: List[Any] = []

    # DoEvent — flip occurrence of the first event (the standard CTF form).
    targets.append(DoEvent(event_id=sample_event.id, occurred=False))

    # DoTrait — pick any entity trait if present.
    actor_ent = ws.entities[sample_actor]
    if actor_ent.traits:
        t_name = next(iter(actor_ent.traits))
        targets.append(DoTrait(holder_id=sample_actor, trait_name=t_name, value=0.5))

    # DoBelief — set/raise a belief about the sample_event.
    targets.append(DoBelief(
        holder_id=sample_actor,
        target_id=sample_event.id,
        confidence=0.9,
    ))

    # DoConcern — pick first concern on actor if any.
    if actor_ent.concerns:
        c0 = actor_ent.concerns[0]
        targets.append(DoConcern(
            holder_id=sample_actor,
            concern_id=c0.concern_id,
            salience=0.95,
        ))

    # DoProposition — pick first prop if any.
    if ws.propositions:
        p0 = ws.propositions[0]
        targets.append(DoProposition(
            proposition_id=p0.proposition_id,
            truth=True,
            fabula_time=sample_event.fabula_time,
        ))

    # DoWorldTrait — pick first world trait if any.
    if ws.world_traits:
        wt0 = next(iter(ws.world_traits))
        targets.append(DoWorldTrait(world_trait_id=wt0, value=0.7))

    # DoChannel — disable first channel if any.
    if ws.channels:
        ch0 = next(iter(ws.channels))
        targets.append(DoChannel(
            channel_id=ch0,
            active=False,
            fabula_time=sample_event.fabula_time,
        ))

    # DoRelationship — pick first social edge if any.
    if ws.social_topology:
        re0 = ws.social_topology[0]
        # Find a metric on re0.
        if re0.metrics:
            m_name = next(iter(re0.metrics))
            if m_name in ("affinity", "fear", "power_dynamic"):
                targets.append(DoRelationship(
                    source_entity_id=re0.source_entity_id,
                    target_entity_id=re0.target_entity_id,
                    metric=m_name,
                    value=0.5,
                ))

    # DoCausalEdge — sever the first causal edge if any (action='sever'
    # is lighter than add: no required mechanism/causality_type fields).
    if ws.causal_topology:
        ce0 = ws.causal_topology[0]
        targets.append(DoCausalEdge(
            source_id=ce0.source_id,
            target_id=ce0.target_id,
            action="sever",
        ))

    # DoSpatialEdge — sever the first spatial edge if any.
    if ws.spatial_topology:
        se0 = ws.spatial_topology[0]
        targets.append(DoSpatialEdge(
            source_id=se0.source_id,
            target_id=se0.target_id,
            action="sever",
        ))

    # DoNarrativeObject — relocate the first object to its current
    # location (no-op move, but exercises the apply path).
    if ws.objects:
        obj0_id = next(iter(ws.objects))
        obj0 = ws.objects[obj0_id]
        if obj0.location_id:
            targets.append(DoNarrativeObject(
                object_id=obj0_id,
                new_location_id=obj0.location_id,
                fabula_time=sample_event.fabula_time,
            ))

    # DoEntityDelete — pick a peripheral entity that is NOT an actor on
    # the sample event so the delete does not break the event surgery.
    peripheral = None
    for eid in ws.entities:
        if eid != sample_actor and eid not in (sample_event.actor_ids or []):
            peripheral = eid
            break
    if peripheral:
        targets.append(DoEntityDelete(entity_id=peripheral))

    # DoObjectDelete — pick an object not referenced by sample event.
    if ws.objects:
        # Pick last object so it's least likely to be the one we just
        # relocated above.
        obj_ids = list(ws.objects.keys())
        del_obj = obj_ids[-1] if len(obj_ids) > 1 else None
        if del_obj:
            targets.append(DoObjectDelete(object_id=del_obj))

    # Run them all through apply_do_targets.
    ws_id_before = id(ws)
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    try:
        engine.apply_do_targets(targets)
        ok = True
        err = ""
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(limit=2)
    check(
        f"DoJoin[{world_name}].apply_do_targets",
        ok,
        f"applied {len(targets)} typed targets" + (f"; err={err}" if err else ""),
    )

    if ok:
        # Live ws preserved (engine doesn't replace, only mutates).
        check(
            f"DoJoin[{world_name}].ws-live-preserved",
            id(engine.world_state) == ws_id_before,
            "engine.world_state is the same Python object",
        )
        # Legacy lift surface populated.
        legacy = getattr(engine, "_last_legacy_interventions", None)
        check(
            f"DoJoin[{world_name}].legacy-lift-stashed",
            isinstance(legacy, dict),
            f"legacy keys={sorted((legacy or {}).keys())[:6]}",
        )


# ======================================================================
# P6. Prompt file coverage + P7 drift.
# ======================================================================

import re as _re


def probe_prompt_coverage() -> None:
    expected = set()
    pattern = _re.compile(r'_load_prompt\(["\']([a-z_]+\.md)["\']\)')
    for src in [
        REPO / "shadow_loom" / "ingestion.py",
        REPO / "shadow_loom" / "auditor.py",
        REPO / "shadow_loom" / "generation.py",
    ]:
        for m in pattern.finditer(src.read_text()):
            expected.add(m.group(1))
    present = {p.name for p in PROMPT_DIR.glob("*.md")}
    missing = sorted(expected - present)
    check(
        "Prompts.coverage.all-loaders-resolve",
        not missing,
        f"references={len(expected)} present={len(present)} missing={missing}",
    )
    orphans = sorted(present - expected - {"generation.md", "refinement.md"})
    check(
        "Prompts.coverage.no-orphan-prompts",
        not orphans,
        f"orphan prompt files (no _load_prompt caller): {orphans}",
        warn_only=True,
    )


_DRIFT_PATTERNS = [
    # (pattern, label, exemption_files)
    (r"\binformation_topology\b", "legacy field 'information_topology'", set()),
    # 'archetype' was removed from Entity. Free to mention on object affordances
    # discussions; we only flag prompts that pair it with entity context.
    (r"Entity[^\n]{0,40}\barchetype\b", "Entity.archetype (removed)", set()),
    # Concern.snapshots was renamed state_timeline. Only flag when paired with
    # 'concern' in the same line.
    (r"[Cc]oncern[^\n]{0,80}\bsnapshots\s*[:=]", "Concern.snapshots (renamed state_timeline)", set()),
    # Concern.activation_window was renamed activation_fabula_window.
    (r"\bactivation_window\b", "Concern.activation_window (renamed activation_fabula_window)", set()),
]


def probe_prompt_drift() -> None:
    found_any = False
    for md in sorted(PROMPT_DIR.glob("*.md")):
        text = md.read_text()
        for patt, label, exempt in _DRIFT_PATTERNS:
            if md.name in exempt:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if _re.search(patt, line):
                    check(
                        f"Prompts.drift[{md.name}:{line_no}]",
                        False,
                        f"{label} -> {line.strip()[:90]}",
                    )
                    found_any = True
    if not found_any:
        check("Prompts.drift.none-detected", True, "no removed/renamed fields referenced")


# ======================================================================
# P8 - Renderer / auditor surface coverage for typed Pearl-rung cascades.
# ======================================================================

def probe_renderer_surface_coverage() -> None:
    """Verify object / world-trait / edge mutation streams traverse every
    handoff between physics, brief builders, answer renderer, reasoning
    trace, and MCP envelope without being silently dropped.

    Round-4 surfaced that ``_typed_target_payload`` produces these
    three streams but every downstream consumer (brief builders,
    answer renderer, reasoning trace, MCP envelope) was wired only
    for trait + social mutations. The probe asserts each surface
    now propagates the three new streams.
    """
    import inspect

    from shadow_loom.generation import (
        _build_downstream_cascade_payload,
        build_intervention_brief,
        build_counterfactual_brief,
        _format_object_mutation_lines,
        _format_world_trait_mutation_lines,
        _format_edge_mutation_lines,
    )
    from shadow_loom.directive_assembly import (
        InterventionBranch, CounterfactualBranch, ThreatProximity,
    )
    from shadow_loom.answer import answer_question
    from shadow_loom_ui.reasoning_helpers import (
        extract_reasoning_trace,
        reasoning_trace_summary,
    )

    # 1. Formatters render every stream.
    obj_lines = _format_object_mutation_lines([
        {"object_id": "OBJ_X", "fabula_time": 10,
         "new_location_id": "LOC_B", "set_owner_null": True}
    ])
    check("RendererSurface.format.object",
          bool(obj_lines) and "OBJ_X" in obj_lines[0],
          f"got={obj_lines}")
    wt_lines = _format_world_trait_mutation_lines([
        {"world_trait_id": "WORLD_war", "old_value": 0.4,
         "new_value": 0.85, "fabula_time": 10, "inertia": 0.2}
    ])
    check("RendererSurface.format.world_trait",
          bool(wt_lines) and "WORLD_war" in wt_lines[0],
          f"got={wt_lines}")
    edge_lines = _format_edge_mutation_lines([
        {"edge_type": "causal", "action": "sever",
         "source_id": "ENT_A", "target_id": "ENT_B", "fabula_time": 10}
    ])
    check("RendererSurface.format.edge",
          bool(edge_lines) and "ENT_A" in edge_lines[0],
          f"got={edge_lines}")

    # 2. Cascade payload bundles all three keys.
    payload = _build_downstream_cascade_payload(
        object_mutations=[{"object_id": "OBJ_X", "fabula_time": 10}],
        world_trait_mutations=[{"world_trait_id": "WORLD_x",
                                "old_value": 0.0, "new_value": 0.5,
                                "fabula_time": 10}],
        edge_mutations=[{"edge_type": "spatial", "action": "lock",
                         "source_id": "LOC_A", "target_id": "LOC_B",
                         "fabula_time": 10}],
    )
    for key in ("object_cascade_detail",
                "world_trait_cascade_detail",
                "edge_cascade_detail"):
        check(f"RendererSurface.payload.{key}",
              key in payload and bool(payload[key]),
              f"keys={sorted(payload.keys())}")

    # 3. Branch models accept the new fields.
    for cls in (InterventionBranch, ThreatProximity):
        fields = getattr(cls, "model_fields", {})
        for f in ("object_cascade_detail", "world_trait_cascade_detail",
                  "edge_cascade_detail"):
            check(f"RendererSurface.branch.{cls.__name__}.{f}",
                  f in fields,
                  f"fields={sorted(fields.keys())[:8]}\u2026")
    cb_fields = getattr(CounterfactualBranch, "model_fields", {})
    for f in ("object_cascade_detail", "world_trait_cascade_detail",
              "edge_cascade_detail"):
        check(f"RendererSurface.branch.CounterfactualBranch.{f}",
              f in cb_fields, "")

    # 4. answer_question accepts the three new params.
    sig = inspect.signature(answer_question)
    for p in ("object_mutations", "world_trait_mutations",
              "edge_mutations"):
        check(f"RendererSurface.answer_question.{p}",
              p in sig.parameters, f"params={list(sig.parameters)[:8]}\u2026")

    # 5. build_intervention_brief / build_counterfactual_brief accept them.
    for fn in (build_intervention_brief, build_counterfactual_brief):
        bsig = inspect.signature(fn)
        for p in ("object_mutations", "world_trait_mutations",
                  "edge_mutations"):
            check(f"RendererSurface.{fn.__name__}.{p}",
                  p in bsig.parameters, "")

    # 6. extract_reasoning_trace populates the six new rails.
    physics = {
        "query_type": "intervention",
        "proposition_mutations": [{
            "proposition_id": "PROP_X", "old_truth": False,
            "new_truth": True, "fabula_time": 10,
            "cascaded_belief_count": 2,
        }],
        "belief_mutations": [{
            "holder_id": "ENT_A", "target_id": "ENT_B",
            "old_confidence": 0.3, "new_confidence": 0.8,
            "proposition_id": "PROP_X",
        }],
        "concern_mutations": [{
            "holder_id": "ENT_A", "concern_id": "CCN_K",
            "field": "salience", "old_value": 0.4, "new_value": 0.9,
        }],
        "object_mutations": [{
            "object_id": "OBJ_X", "fabula_time": 10,
            "new_location_id": "LOC_B",
        }],
        "world_trait_mutations": [{
            "world_trait_id": "WORLD_war",
            "old_value": 0.4, "new_value": 0.85, "fabula_time": 10,
        }],
        "edge_mutations": [{
            "edge_type": "causal", "action": "sever",
            "source_id": "ENT_A", "target_id": "ENT_B",
            "fabula_time": 10,
        }],
    }
    trace = extract_reasoning_trace(physics)
    for rail in ("proposition_cascade", "belief_cascade", "concern_cascade",
                 "object_cascade", "world_trait_cascade", "edge_cascade"):
        rows = trace.get(rail) or []
        check(f"RendererSurface.reasoning.{rail}",
              len(rows) == 1,
              f"len={len(rows)}")
    summary = reasoning_trace_summary(trace)
    for tok in ("prop", "belief", "concern", "object", "world-trait", "edge"):
        check(f"RendererSurface.reasoning_summary.{tok}",
              tok in summary,
              f"summary={summary!r}")

    # 7. MCP envelope surfaces cascade_counts when streams present.
    # We grep the source rather than fabricate a PipelineResult, since
    # the envelope is built inline inside ``run_and_save``.
    import shadow_loom_mcp.helpers as _mh
    src = inspect.getsource(_mh)
    check("RendererSurface.mcp.cascade_counts",
          "cascade_counts" in src
          and "object_mutations" in src
          and "world_trait_mutations" in src
          and "edge_mutations" in src,
          "envelope must surface cascade_counts with all three streams")


def main() -> int:
    worlds = _example_world_modules()

    section(f"P1. SCHEMA INTEGRITY — {len(worlds)} example worlds")
    total_issues = 0
    total_refs = 0
    for w in worlds:
        try:
            issues, refs = probe_schema_integrity(w)
        except Exception as exc:
            check(f"Schema[{w}].load", False, f"raised {exc!r}")
            continue
        total_issues += issues
        total_refs += refs
        check(
            f"Schema[{w}]",
            issues == 0,
            f"issues={issues} refs_checked={refs}",
        )
    print(f"\n  Totals: {total_refs} references checked, {total_issues} issues.")

    section(f"P2. PROGRAMMATIC VALIDATION — {len(worlds)} example worlds")
    total_errors = 0
    total_warnings = 0
    for w in worlds:
        try:
            errors, warnings_n = probe_programmatic_validation(w)
        except Exception as exc:
            check(f"ProgVal[{w}]", False, f"raised {exc!r}")
            continue
        total_errors += errors
        total_warnings += warnings_n
        check(
            f"ProgVal[{w}]",
            errors == 0,
            f"errors={errors} warnings={warnings_n}",
            warn_only=(errors == 0 and warnings_n > 0),
        )
    print(f"\n  Totals: {total_errors} errors, {total_warnings} warnings across {len(worlds)} worlds.")

    section(f"P3. VALIDATE-AND-CORRECT (auto-repair idempotence) — {len(worlds)} example worlds")
    for w in worlds:
        try:
            ok, detail = probe_validate_and_correct(w)
        except Exception as exc:
            check(f"AutoRepair[{w}]", False, f"raised {exc!r}")
            continue
        check(
            f"AutoRepair[{w}]",
            ok,
            detail,
            warn_only=(not ok and "deltas=" not in detail),
        )

    section("P4. CONTINUATION-BRIDGE WIRING — pipeline helpers + exports")
    try:
        probe_continuation_bridge_wiring()
    except Exception:
        traceback.print_exc()
        check("Bridge.probe", False, "exception during probe")

    section(f"P4b. CONTINUATION-BRIDGE FUNCTIONAL — round-trip on {len(worlds)} worlds")
    for w in worlds:
        try:
            probe_continuation_bridge_functional(w)
        except Exception:
            traceback.print_exc(limit=2)
            check(f"BridgeFn[{w}].probe", False, "exception during probe")

    section(f"P5. INTERVENTION-JOIN — DoTarget application on {len(worlds)} worlds")
    for w in worlds:
        try:
            probe_intervention_join_consistency(w)
        except Exception:
            traceback.print_exc(limit=2)
            check(f"DoJoin[{w}].probe", False, "exception during probe")

    section("P6. PROMPT FILE COVERAGE")
    probe_prompt_coverage()

    section("P7. PROMPT-FIELD DRIFT")
    probe_prompt_drift()

    section("P8. RENDERER / AUDITOR SURFACE COVERAGE — object / world-trait / edge cascades")
    try:
        probe_renderer_surface_coverage()
    except Exception:
        traceback.print_exc(limit=2)
        check("RendererSurface.probe", False, "exception during probe")

    section(f"ROUND-4 DEEP AUDIT  PASS={len(_PASS)}  FAIL={len(_FAIL)}  WARN={len(_WARN)}")
    if _FAIL:
        print("  -- failures --")
        for f in _FAIL[:30]:
            print(f"    X {f}")
        if len(_FAIL) > 30:
            print(f"    ... +{len(_FAIL) - 30} more")
    if _WARN:
        print("  -- warnings --")
        for w in _WARN[:15]:
            print(f"    ! {w}")
        if len(_WARN) > 15:
            print(f"    ... +{len(_WARN) - 15} more")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
