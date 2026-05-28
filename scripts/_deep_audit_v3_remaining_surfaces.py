"""Round-3 deep audit — remaining unchecked surfaces.

Probes (real worlds, no mocks, no LLM):
  1. ManualEditQuery shape via narrative_physics handler.
  2. EvaluationQuery shape via narrative_physics handler.
  3. IntroducedChannelSpec → introduced_elements_to_spawns materialisation.
  4. Shadow world / AMWN branching (projected_for_branch + sidecar layering).
  5. Concern.activation_fabula_window gating via reconstruct_concern_at.
  6. DoEvent relocation cascades EntityStateSnapshot to primary actors.
  7. Pipeline entry-point smoke (signature, no execution).
"""
from __future__ import annotations

import importlib
import inspect
import sys
from typing import Optional

from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import (
    ManualEditQuery, EvaluationQuery, InterventionQuery, DoEvent,
)
from shadow_loom.introduced_elements import IntroducedElements, IntroducedChannelSpec
from shadow_loom.extract_graph import introduced_elements_to_spawns
from shadow_loom.models import (
    Concern, ConcernSnapshot, reconstruct_concern_at, WorldStateV1, Entity,
)


PASS = 0
FAIL = 0
WARN = 0
FAILS: list[str] = []
WARNS: list[str] = []


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def check(name: str, cond: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name} — {detail}")
        return True
    FAIL += 1
    FAILS.append(name)
    print(f"  [FAIL] {name} — {detail}")
    return False


def warn(name: str, detail: str = "") -> None:
    global WARN
    WARN += 1
    WARNS.append(name)
    print(f"  [WARN] {name} — {detail}")


def load(world_name: str) -> WorldStateV1:
    return importlib.import_module(f"example_worlds.{world_name}").world_state


def run(ws: WorldStateV1, query) -> dict:
    return calculate_narrative_physics(query, ws, None, None, False)


# ---------------------------------------------------------------------------
# 1. ManualEditQuery
# ---------------------------------------------------------------------------
def probe_manual_edit():
    hr("MANUAL EDIT (Rung-1 prose authoring) — narrative_physics handler")
    ws = load("macbeth")
    q = ManualEditQuery(
        edited_prose="Macbeth pauses on the heath, doubt lining his brow.",
        focus_entity_ids=[next(iter(ws.entities.keys()))],
    )
    r = run(ws, q)
    check("ManualEdit.success", r.get("status") == "success",
          f"status={r.get('status')} err={r.get('error','')[:80]}")
    check("ManualEdit.query_type-correct", r.get("query_type") == "manual_edit",
          f"qt={r.get('query_type')}")
    check("ManualEdit.edited_prose-preserved",
          r.get("edited_prose") == q.edited_prose,
          f"len={len(r.get('edited_prose',''))}")
    check("ManualEdit.physics_state-included",
          isinstance(r.get("physics_state"), dict) and bool(r.get("physics_state")),
          f"type={type(r.get('physics_state')).__name__}")


# ---------------------------------------------------------------------------
# 2. EvaluationQuery
# ---------------------------------------------------------------------------
def probe_evaluation():
    hr("EVALUATION (full-story audit) — narrative_physics handler")
    ws = load("macbeth")
    q = EvaluationQuery(focus_entity_ids=[], include_full_prose=True)
    r = run(ws, q)
    check("Evaluate.success", r.get("status") == "success",
          f"status={r.get('status')} err={r.get('error','')[:80]}")
    check("Evaluate.query_type-correct", r.get("query_type") == "evaluate",
          f"qt={r.get('query_type')}")
    check("Evaluate.physics_state-included",
          isinstance(r.get("physics_state"), dict) and bool(r.get("physics_state")),
          f"type={type(r.get('physics_state')).__name__}")


# ---------------------------------------------------------------------------
# 3. IntroducedChannelSpec → spawn materialisation
# ---------------------------------------------------------------------------
def probe_introduced_channel():
    hr("INTRODUCED CHANNEL SPEC — channel-genesis via introduced_elements_to_spawns")
    ws = load("macbeth")
    # Pick two existing entities as participants.
    ent_ids = list(ws.entities.keys())[:2]
    if len(ent_ids) < 2:
        warn("IntroducedChannel.skipped", "macbeth has <2 entities")
        return
    new_chn_id = "CHN_TEST_RAVEN_COURIER"
    spec = IntroducedChannelSpec(
        id=new_chn_id,
        name="Raven Courier",
        justification="Test probe — synthetic channel.",
        medium="raven_courier",
        participant_ids=ent_ids,
        directionality="duplex",
    )
    payload = IntroducedElements(channels=[spec])
    out = introduced_elements_to_spawns(payload, ws)
    check("IntroducedChannel.spawn-shape-correct",
          isinstance(out, dict) and "channels" in out,
          f"keys={sorted(out.keys())}")
    channels_out = out.get("channels", {})
    check("IntroducedChannel.materialised",
          new_chn_id in channels_out,
          f"channels_out_ids={list(channels_out.keys())}")
    if new_chn_id in channels_out:
        ch = channels_out[new_chn_id]
        check("IntroducedChannel.medium-preserved",
              getattr(ch, "medium", None) == "raven_courier",
              f"medium={getattr(ch, 'medium', None)}")
        check("IntroducedChannel.participants-preserved",
              sorted(getattr(ch, "participant_ids", []) or []) == sorted(ent_ids),
              f"participants={getattr(ch, 'participant_ids', None)}")
    # Idempotency: re-introducing a channel that already exists in the
    # world is a no-op (skipped).
    existing_chn_ids = list(ws.channels.keys()) if ws.channels else []
    if existing_chn_ids:
        existing_id = existing_chn_ids[0]
        existing = ws.channels[existing_id]
        dup_spec = IntroducedChannelSpec(
            id=existing_id,
            name="Dup",
            justification="Test idempotency.",
            medium=getattr(existing, "medium", "x") or "x",
            participant_ids=getattr(existing, "participant_ids", []) or ent_ids,
        )
        out2 = introduced_elements_to_spawns(IntroducedElements(channels=[dup_spec]), ws)
        check("IntroducedChannel.duplicate-skipped",
              existing_id not in out2.get("channels", {}),
              f"existing={existing_id} dup_out={list(out2.get('channels', {}).keys())}")


# ---------------------------------------------------------------------------
# 4. Shadow world / AMWN branching
# ---------------------------------------------------------------------------
def probe_shadow_world_branching():
    hr("SHADOW WORLD / AMWN BRANCHING — projected_for_branch")
    ws = load("macbeth")
    # Factual read returns self (identity check).
    proj_factual = ws.projected_for_branch("factual")
    check("Shadow.factual-returns-self", proj_factual is ws,
          f"id_match={proj_factual is ws}")
    # Shadow read with no sidecar data falls through to self.
    proj_empty_shadow = ws.projected_for_branch("shadow", "nonexistent_branch")
    check("Shadow.empty-shadow-falls-through", proj_empty_shadow is ws,
          f"id_match={proj_empty_shadow is ws}")
    # Strict mode: shadow read without branch_label raises.
    try:
        ws.projected_for_branch("shadow", None, strict=True)
        check("Shadow.strict-raises-on-missing-label", False, "no exception raised")
    except ValueError as e:
        check("Shadow.strict-raises-on-missing-label", True,
              f"raised ValueError: {str(e)[:60]}")
    # Inject a synthetic shadow entity and verify layered read.
    real_ent_id = next(iter(ws.entities.keys()))
    real_ent = ws.entities[real_ent_id]
    # Clone with mutated name as a shadow variant.
    shadow_clone = Entity(
        id=real_ent.id,
        name=f"{real_ent.name}_SHADOW",
        location_id=real_ent.location_id,
        status=real_ent.status,
        traits=getattr(real_ent, "traits", {}),
    )
    branch_label = "audit_test_branch"
    # Write into the sidecar.
    ws.shadow_entities.setdefault(branch_label, {})[real_ent_id] = shadow_clone
    try:
        proj_shadow = ws.projected_for_branch("shadow", branch_label)
        check("Shadow.layered-read-returns-new-ws",
              proj_shadow is not ws,
              f"id_match={proj_shadow is ws}")
        shadow_ent = proj_shadow.entities.get(real_ent_id)
        check("Shadow.layered-entity-name-overridden",
              shadow_ent is not None and shadow_ent.name == f"{real_ent.name}_SHADOW",
              f"name={getattr(shadow_ent, 'name', None)}")
        # Factual still untouched.
        check("Shadow.factual-entity-unmutated",
              ws.entities[real_ent_id].name == real_ent.name,
              f"factual_name={ws.entities[real_ent_id].name}")
    finally:
        # Cleanup so subsequent probes see clean state.
        ws.shadow_entities.pop(branch_label, None)


# ---------------------------------------------------------------------------
# 5. Concern.activation_fabula_window
# ---------------------------------------------------------------------------
def probe_concern_activation_window():
    hr("CONCERN activation_fabula_window — reconstruct_concern_at gating")
    # Synthetic concern with explicit window [1000, 5000].
    c = Concern(
        concern_id="CCN_AUDIT_TEST",
        proposition_id="PROP_TEST",
        polarity="desire",
        salience=0.7,
        activation_fabula_window=[1000, 5000],
    )
    inside_lo = reconstruct_concern_at(c, 1000)
    inside_mid = reconstruct_concern_at(c, 3000)
    inside_hi = reconstruct_concern_at(c, 5000)
    outside_before = reconstruct_concern_at(c, 999)
    outside_after = reconstruct_concern_at(c, 5001)
    check("Concern.window-boundary-lo-inclusive",
          inside_lo["active"] is True,
          f"active@1000={inside_lo['active']}")
    check("Concern.window-mid-active",
          inside_mid["active"] is True,
          f"active@3000={inside_mid['active']}")
    check("Concern.window-boundary-hi-inclusive",
          inside_hi["active"] is True,
          f"active@5000={inside_hi['active']}")
    check("Concern.window-outside-before-inactive",
          outside_before["active"] is False,
          f"active@999={outside_before['active']}")
    check("Concern.window-outside-after-inactive",
          outside_after["active"] is False,
          f"active@5001={outside_after['active']}")
    # No window → always active.
    c_no_window = Concern(
        concern_id="CCN_AUDIT_NO_WINDOW",
        proposition_id="PROP_TEST",
        polarity="desire",
        salience=0.5,
    )
    check("Concern.no-window-always-active",
          reconstruct_concern_at(c_no_window, 99999)["active"] is True,
          "active=True for any t when no window set")
    # Snapshot override: a later snapshot can supply a window.
    c_evolving = Concern(
        concern_id="CCN_AUDIT_EVOLVING",
        proposition_id="PROP_TEST",
        polarity="desire",
        salience=0.5,
        state_timeline=[
            ConcernSnapshot(
                fabula_time=2000,
                triggered_by="EVT_TEST",
                activation_fabula_window=[2000, 4000],
            ),
        ],
    )
    # Before snapshot tick: no window → active.
    pre = reconstruct_concern_at(c_evolving, 1500)
    # At/after snapshot: window applies.
    at_snap = reconstruct_concern_at(c_evolving, 3000)
    post_window = reconstruct_concern_at(c_evolving, 4500)
    check("Concern.snapshot-installs-window",
          pre["active"] is True and at_snap["active"] is True and post_window["active"] is False,
          f"pre@1500={pre['active']} mid@3000={at_snap['active']} after@4500={post_window['active']}")


# ---------------------------------------------------------------------------
# 6. DoEvent relocation cascades to primary actor state_timeline
# ---------------------------------------------------------------------------
def probe_do_event_relocation_cascade():
    hr("DoEvent RELOCATION — co-presence cascade to actor state_timeline")
    ws = load("macbeth")
    # Find an event with at least one living primary actor & a target loc.
    target_evt = None
    new_loc_id: Optional[str] = None
    for evt in ws.events:
        actor_ids = []
        if getattr(evt, "speaker_id", None):
            actor_ids.append(evt.speaker_id)
        for aid in getattr(evt, "actor_ids", None) or []:
            if aid not in actor_ids:
                actor_ids.append(aid)
        if not actor_ids or not getattr(evt, "at_location_id", None):
            continue
        # Find a living actor whose status is not 'dead'.
        living = None
        for aid in actor_ids:
            actor = ws.entities.get(aid)
            if actor is None:
                continue
            if str(getattr(actor, "status", "") or "").lower() != "dead":
                living = aid
                break
        if living is None:
            continue
        # Pick a different LOC_ id.
        for lid in (ws.locations or {}).keys():
            if lid != evt.at_location_id:
                new_loc_id = lid
                break
        if new_loc_id:
            target_evt = evt
            break
    if not target_evt or not new_loc_id:
        warn("DoEvent.cascade.no-suitable-fixture", "no event/loc pair found")
        return

    # Pick the first living primary actor we found above.
    primary_actor_id = None
    actor_id_list = []
    if getattr(target_evt, "speaker_id", None):
        actor_id_list.append(target_evt.speaker_id)
    for aid in getattr(target_evt, "actor_ids", None) or []:
        if aid not in actor_id_list:
            actor_id_list.append(aid)
    for aid in actor_id_list:
        a = ws.entities.get(aid)
        if a is not None and str(getattr(a, "status", "") or "").lower() != "dead":
            primary_actor_id = aid
            break
    assert primary_actor_id is not None

    actor = ws.entities[primary_actor_id]
    pre_timeline_len = len(getattr(actor, "state_timeline", None) or [])
    pre_event_loc = target_evt.at_location_id
    ft = int(target_evt.fabula_time)
    q = InterventionQuery(
        focus_entity_ids=[primary_actor_id],
        fabula_time=ft,
        do_targets=[DoEvent(
            event_id=target_evt.id,
            occurred=True,  # gate: relocation only fires when occurred=True
            new_at_location_id=new_loc_id,
        )],
    )
    r = run(ws, q)
    check("DoEvent.cascade.success", r.get("status") == "success",
          f"event={target_evt.id} new_loc={new_loc_id} status={r.get('status')}")
    # Event location rewritten on world_state.events (mutation persists).
    check("DoEvent.cascade.event-location-rewritten",
          target_evt.at_location_id == new_loc_id,
          f"old={pre_event_loc} new={target_evt.at_location_id} expected={new_loc_id}")
    # Actor timeline appended with a snapshot at ft with location_id=new_loc.
    post_timeline = getattr(actor, "state_timeline", None) or []
    matching_snaps = [s for s in post_timeline
                      if int(getattr(s, "fabula_time", -1)) == ft
                      and getattr(s, "location_id", None) == new_loc_id]
    check("DoEvent.cascade.actor-snapshot-appended",
          len(matching_snaps) >= 1,
          f"actor={primary_actor_id} pre_len={pre_timeline_len} post_len={len(post_timeline)} "
          f"matching={len(matching_snaps)}")
    if matching_snaps:
        check("DoEvent.cascade.snapshot-triggered-by-event",
              getattr(matching_snaps[0], "triggered_by", None) == target_evt.id,
              f"triggered_by={getattr(matching_snaps[0], 'triggered_by', None)}")

    # UX safety probe: when ``new_at_location_id`` is set, the
    # ``DoEvent`` model auto-implies ``occurred=True`` so the
    # relocation actually lands (callers routinely forget the flag —
    # the relocation reads as a positive surgery). Verify both the
    # model-level auto-flip and that the engine then mutates the live
    # world_state. Re-load a fresh world to avoid contaminating other
    # probes with the previous mutation.
    ws_fresh = importlib.reload(
        importlib.import_module("example_worlds.macbeth")
    ).world_state
    evt_fresh = next(e for e in ws_fresh.events if e.id == target_evt.id)
    pre_loc_fresh = evt_fresh.at_location_id
    de_auto = DoEvent(
        event_id=target_evt.id,
        # occurred omitted — defaults to False, should auto-flip to True
        new_at_location_id=new_loc_id,
    )
    check("DoEvent.auto_imply.occurred-flipped-to-true",
          de_auto.occurred is True,
          f"occurred={de_auto.occurred} (auto-implied from new_at_location_id)")
    q_auto = InterventionQuery(
        focus_entity_ids=[primary_actor_id],
        fabula_time=ft,
        do_targets=[de_auto],
    )
    r_auto = run(ws_fresh, q_auto)
    check("DoEvent.auto_imply.relocation-actually-landed",
          evt_fresh.at_location_id == new_loc_id,
          f"loc={evt_fresh.at_location_id} expected={new_loc_id} status={r_auto.get('status')}")


# ---------------------------------------------------------------------------
# 7. Pipeline entry-point smoke
# ---------------------------------------------------------------------------
def probe_pipeline_entry_point():
    hr("PIPELINE entry-point signature smoke")
    try:
        from shadow_loom.pipeline import run_pipeline, run_pipeline_async, PipelineResult
    except Exception as e:
        check("Pipeline.imports", False, f"import failed: {e}")
        return
    check("Pipeline.imports", True, "run_pipeline + run_pipeline_async + PipelineResult importable")
    sig = inspect.signature(run_pipeline)
    params = set(sig.parameters.keys())
    expected = {"query", "world_state", "raw_text"}
    check("Pipeline.signature-has-expected-params",
          expected.issubset(params),
          f"params={sorted(params)[:10]} missing={sorted(expected - params)}")


# ---------------------------------------------------------------------------
def main():
    probe_manual_edit()
    probe_evaluation()
    probe_introduced_channel()
    probe_shadow_world_branching()
    probe_concern_activation_window()
    probe_do_event_relocation_cascade()
    probe_pipeline_entry_point()
    print(f"\n{'=' * 78}\n  ROUND-3 DEEP AUDIT  PASS={PASS}  FAIL={FAIL}  WARN={WARN}\n{'=' * 78}")
    if FAILS:
        print("  -- failures --")
        for f in FAILS:
            print(f"  X {f}")
    if WARNS:
        print("  -- warnings --")
        for w in WARNS:
            print(f"  ! {w}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
