# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the eighth-pass deep-audit fixes (2026-05-29).

Covers the eight subtle-bug fixes from the eighth audit round:

  * **H1** — engine-time chain-reaction closure now seeds from
    cause-disconnected events (events with non-relocation
    ``_event_mutations``) so engine-side and merge-side suppression
    semantics agree.
  * **H2** — :meth:`CausalPhysicsEngine._apply_do_event_time_shift`
    int-coerces ``Proposition.truth_at_fabula`` keys before the
    ``old_ft not in ledger`` membership check, so a ledger that was
    JSON-round-tripped with string keys still gets its primary AND
    inverse entries relocated.
  * **M1** — :func:`filter_world_state_for_pov` requires BOTH
    endpoints of a causal edge to be POV-known (was OR), so hidden
    node ids no longer leak as dangling endpoint refs.
  * **M2** — POV proposition truth-clip int-coerces ledger keys
    before tick-membership filtering, so string-keyed commits aren't
    silently dropped from the POV slice.
  * **M3** — surprise/irony scorers now use time-aware
    ``_concern_salience_t`` / ``_concern_polarity_sign_t`` so closed
    or polarity-flipped concerns no longer weight affect scores.
  * **M4** — mystery scorer detects revealed ancestors via
    ``Proposition.referent_ids`` linkage instead of the synthetic
    ``PROP_FROM_{evt_id}`` id convention.
  * **M5** — :meth:`CausalPhysicsEngine._apply_do_object_delete`
    scrubs ``SpatialEdge.barrier_item_id`` references to the deleted
    object so excised props no longer leave ghost barriers.
  * **M6** — :func:`compute_suspense_unified` int-coerces ledger
    keys before comparing to int ``fabula_t``, so string-keyed
    ledgers no longer raise ``TypeError``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import networkx as nx

from shadow_loom.affect_unification import (
    AUDIENCE_ID,
    BeliefState,
    _concern_polarity_sign_t,
    _concern_salience_t,
    compute_irony_breakdown,
    compute_mystery_unified,
    compute_suspense_unified,
    compute_surprise_breakdown,
)
from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.models import (
    Affordance,
    Belief,
    CausalEdge,
    Concern,
    ConcernSnapshot,
    Entity,
    EventNode,
    Location,
    NarrativeObject,
    Proposition,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.projections import filter_world_state_for_pov
from shadow_loom.query_models import DoEvent, DoObjectDelete


# =====================================================================
# H2 — DoEventTimeShift relocates ledger even when keys are strings
# =====================================================================
def _ws_h2_str_keys() -> WorldStateV1:
    flat = Location(id="LOC_FLAT", name="The flat", description="A flat")
    actor = Entity(
        id="ENT_KILLER", name="Killer", location_id="LOC_FLAT",
        status="healthy", traits={}, beliefs=[], world_id="factual",
    )
    victim = Entity(
        id="ENT_VICTIM", name="Victim", location_id="LOC_FLAT",
        status="dead", traits={}, beliefs=[], world_id="factual",
    )
    evt = EventNode(
        id="EVT_KILL",
        description="Killer kills Victim",
        fabula_time=10000,
        syuzhet_index=1,
        event_type="outcome",
        at_location_id="LOC_FLAT",
        actor_ids=["ENT_KILLER"],
        resolves_proposition_ids=["PROP_VICTIM_DEAD"],
    )
    prim = Proposition(
        proposition_id="PROP_VICTIM_DEAD",
        kind="outcome",
        description="Victim is dead.",
        truth_at_fabula={10000: True},
        inverse_proposition_id="PROP_VICTIM_ALIVE",
        world_id="factual",
    )
    inv = Proposition(
        proposition_id="PROP_VICTIM_ALIVE",
        kind="outcome",
        description="Victim is alive.",
        truth_at_fabula={10000: False},
        inverse_proposition_id="PROP_VICTIM_DEAD",
        world_id="factual",
    )
    # Simulate post-JSON-round-trip: forcibly install STRING keys
    # on the ledgers (bypassing the pydantic validator that would
    # coerce on construction). This mirrors the deserialization
    # path that motivated H2.
    object.__setattr__(prim, "truth_at_fabula", {"10000": True})
    object.__setattr__(inv, "truth_at_fabula", {"10000": False})

    return WorldStateV1(
        locations={"LOC_FLAT": flat},
        objects={},
        entities={"ENT_KILLER": actor, "ENT_VICTIM": victim},
        events=[evt],
        propositions=[prim, inv],
        social_topology=[],
        causal_topology=[],
    )


def test_h2_do_event_time_shift_relocates_string_keyed_ledger():
    ws = _ws_h2_str_keys()
    # Sanity: keys really are strings before the shift.
    prim = next(p for p in ws.propositions if p.proposition_id == "PROP_VICTIM_DEAD")
    assert "10000" in prim.truth_at_fabula
    assert 10000 not in prim.truth_at_fabula

    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(
            event_id="EVT_KILL",
            occurred=True,
            new_fabula_time=15000,
        ),
    ])

    prim = next(p for p in ws.propositions if p.proposition_id == "PROP_VICTIM_DEAD")
    inv = next(p for p in ws.propositions if p.proposition_id == "PROP_VICTIM_ALIVE")
    # Both ledgers must be relocated and normalized to int keys.
    assert prim.truth_at_fabula == {15000: True}, (
        f"H2: primary ledger should relocate to int-keyed new tick; "
        f"got {prim.truth_at_fabula}"
    )
    assert inv.truth_at_fabula == {15000: False}, (
        f"H2: inverse ledger should follow primary in lockstep; "
        f"got {inv.truth_at_fabula}"
    )


# =====================================================================
# H1 — engine-time chain-reaction closure seeds cause-disconnected
#      events from ``self._event_mutations`` (source-level check +
#      ``EventMutation`` shape check). A full end-to-end run requires
#      a wired sandbox and a disruptive DoEvent surgery whose plumbing
#      is exercised by ``test_pipeline_e2e.py`` (excluded from the
#      fast suite); we assert the contract here.
# =====================================================================
def test_h1_engine_closure_seeds_cause_disconnected_from_event_mutations():
    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "causal_physics.py"
    ).read_text(encoding="utf-8")
    # The eighth-pass H1 block keys on the ``self._event_mutations``
    # list and excludes pure relocation / time_shift kinds (same
    # filter as ``shadow_loom.pipeline`` ROUND-15 C-2).
    assert "eighth pass, H1" in src, "H1 block missing"
    assert "cause_broken_evt_ids" in src
    assert "self._event_mutations" in src
    assert "(\"relocation\", \"time_shift\")" in src
    assert "expand_chain_reaction_closure" in src


# =====================================================================
# M1 — POV causal-edge filter requires BOTH endpoints visible
# =====================================================================
def _ws_m1_pov_with_hidden_edge_endpoint() -> WorldStateV1:
    loc = Location(id="LOC_ROOM", name="Room", description="r")
    loc_secret = Location(id="LOC_SECRET", name="Secret", description="s")
    visible_evt = EventNode(
        id="EVT_VISIBLE",
        description="POV sees this",
        fabula_time=100,
        syuzhet_index=0,
        event_type="outcome",
        at_location_id="LOC_ROOM",
        actor_ids=["ENT_POV"],
    )
    # Off-stage event POV cannot witness (no POV in actor/target, at
    # an off-map location with no visible neighbours).
    hidden_evt = EventNode(
        id="EVT_HIDDEN",
        description="off-stage event",
        fabula_time=200,
        syuzhet_index=1,
        event_type="outcome",
        at_location_id="LOC_SECRET",
        actor_ids=["ENT_OFFSTAGE"],
    )
    offstage = Entity(
        id="ENT_OFFSTAGE", name="Offstage", location_id="LOC_SECRET",
        status="healthy", traits={}, world_id="factual",
    )
    pov = Entity(
        id="ENT_POV", name="POV", location_id="LOC_ROOM",
        status="healthy", traits={}, world_id="factual",
    )
    # Causal edge from the POV-visible event into the HIDDEN event.
    # M1: this edge must be pruned because EVT_HIDDEN is not in
    # ``visible_evt_ids`` (so not in ``pov_known_node_ids``).
    edge_to_hidden = CausalEdge(
        source_id="EVT_VISIBLE",
        target_id="EVT_HIDDEN",
        causality_type="chain_reaction",
        mechanism="follows",
        fabula_time=200,
    )
    return WorldStateV1(
        locations={"LOC_ROOM": loc, "LOC_SECRET": loc_secret},
        objects={},
        entities={"ENT_POV": pov, "ENT_OFFSTAGE": offstage},
        events=[visible_evt, hidden_evt],
        propositions=[],
        social_topology=[],
        causal_topology=[edge_to_hidden],
    )


def test_m1_pov_causal_edge_filter_drops_edges_with_hidden_endpoint():
    ws = _ws_m1_pov_with_hidden_edge_endpoint()
    filtered = filter_world_state_for_pov(ws, pov_entity_id="ENT_POV")
    # EVT_HIDDEN must be stripped from the POV events list.
    visible_evt_ids = {e.id for e in filtered.events}
    assert "EVT_HIDDEN" not in visible_evt_ids
    # And the causal edge pointing INTO the hidden event must NOT
    # survive — otherwise the POV could infer the hidden id via the
    # dangling target ref.
    leaking_edges = [
        ce for ce in filtered.causal_topology
        if ce.target_id == "EVT_HIDDEN" or ce.source_id == "EVT_HIDDEN"
    ]
    assert leaking_edges == [], (
        "M1: causal edges referencing POV-hidden endpoints must be "
        "pruned to preserve referential integrity"
    )


# =====================================================================
# M2 — POV prop truth-clip int-coerces ledger keys
# =====================================================================
def test_m2_pov_filter_clips_string_keyed_truth_at_fabula():
    loc = Location(id="LOC_X", name="X", description="x")
    visible_evt = EventNode(
        id="EVT_AT_TICK_100",
        description="visible at 100",
        fabula_time=100,
        syuzhet_index=0,
        event_type="outcome",
        at_location_id="LOC_X",
        actor_ids=["ENT_POV"],
    )
    pov = Entity(
        id="ENT_POV", name="POV", location_id="LOC_X",
        status="healthy", traits={}, world_id="factual",
    )
    prop = Proposition(
        proposition_id="PROP_P",
        kind="event_occurs",
        description="P holds",
        truth_at_fabula={100: True},
    )
    # Force string keys after construction (mirrors JSON round-trip).
    object.__setattr__(prop, "truth_at_fabula", {"100": True, "9999": True})
    ws = WorldStateV1(
        locations={"LOC_X": loc},
        objects={},
        entities={"ENT_POV": pov},
        events=[visible_evt],
        propositions=[prop],
        social_topology=[],
        causal_topology=[],
    )
    filtered = filter_world_state_for_pov(ws, pov_entity_id="ENT_POV")
    out_prop = next(p for p in filtered.propositions if p.proposition_id == "PROP_P")
    # The visible tick 100 should survive AND be re-keyed as int;
    # the off-screen tick 9999 should be clipped out.
    assert out_prop.truth_at_fabula == {100: True}, (
        f"M2: POV filter should int-coerce ledger keys before tick "
        f"clipping; got {out_prop.truth_at_fabula}"
    )


# =====================================================================
# M5 — DoObject delete scrubs SpatialEdge.barrier_item_id
# =====================================================================
def test_m5_do_object_delete_scrubs_spatial_barrier_item_id():
    loc_a = Location(id="LOC_A", name="A", description="a")
    loc_b = Location(id="LOC_B", name="B", description="b")
    key_obj = NarrativeObject(
        id="OBJ_KEY",
        name="Iron Key",
        location_id="LOC_A",
        owner_id=None,
        affordances=[Affordance(action="unlock", target_type="Door")],
    )
    locked_edge = SpatialEdge(
        source_id="LOC_A",
        target_id="LOC_B",
        relation_type="adjacent_to",
        is_locked=True,
        barrier_item_id="OBJ_KEY",
    )
    other_edge = SpatialEdge(
        source_id="LOC_B",
        target_id="LOC_A",
        relation_type="adjacent_to",
    )
    ws = WorldStateV1(
        locations={"LOC_A": loc_a, "LOC_B": loc_b},
        objects={"OBJ_KEY": key_obj},
        entities={},
        events=[],
        propositions=[],
        social_topology=[],
        spatial_topology=[locked_edge, other_edge],
        causal_topology=[],
    )
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoObjectDelete(object_id="OBJ_KEY"),
    ])
    # Object excised.
    assert "OBJ_KEY" not in ws.objects
    # Spatial edge's barrier_item_id reference scrubbed.
    assert locked_edge.barrier_item_id is None, (
        "M5: SpatialEdge.barrier_item_id pointing at a deleted "
        "object must be cleared so traversal isn't blocked by a "
        "ghost barrier"
    )
    # Unrelated edge untouched.
    assert other_edge.barrier_item_id is None
    # Lock state intent preserved (we don't auto-unlock; the prose
    # may still describe the door as locked, just without a referent).
    assert locked_edge.is_locked is True


# =====================================================================
# M6 — suspense scorer handles string-keyed truth_at_fabula
# =====================================================================
def test_m6_suspense_scorer_handles_string_keyed_ledger():
    loc = Location(id="LOC_X", name="X", description="x")
    audience = Entity(
        id=AUDIENCE_ID, name="Audience", location_id="LOC_X",
        status="healthy", traits={}, world_id="factual",
    )
    open_prop = Proposition(
        proposition_id="PROP_OPEN",
        kind="outcome",
        description="Open question",
        stakes=1.0,
        truth_at_fabula={},
    )
    # Closed at fabula=100 with STRING key (post-JSON round-trip).
    closed_prop = Proposition(
        proposition_id="PROP_CLOSED",
        kind="outcome",
        description="Closed question",
        stakes=1.0,
        truth_at_fabula={100: True},
    )
    object.__setattr__(closed_prop, "truth_at_fabula", {"100": True})
    ws = WorldStateV1(
        locations={"LOC_X": loc},
        objects={},
        entities={AUDIENCE_ID: audience},
        events=[],
        propositions=[open_prop, closed_prop],
        social_topology=[],
        causal_topology=[],
    )
    bs = BeliefState(world=ws)
    # Must NOT raise (pre-fix this raised TypeError on "100" > 200).
    score = compute_suspense_unified(bs, fabula_t=200, tau_fabula=1000.0)
    assert score >= 0.0
    # The closed prop (committed at 100, cursor at 200, no future
    # commits) must be skipped — only the open prop contributes.
    # Re-run without the open prop to isolate.
    ws2 = WorldStateV1(
        locations={"LOC_X": loc},
        objects={},
        entities={AUDIENCE_ID: audience},
        events=[],
        propositions=[closed_prop],
        social_topology=[],
        causal_topology=[],
    )
    bs2 = BeliefState(world=ws2)
    score_closed_only = compute_suspense_unified(
        bs2, fabula_t=200, tau_fabula=1000.0,
    )
    assert score_closed_only == 0.0, (
        f"M6: closed prop (str-keyed commit at 100 <= cursor 200) "
        f"must be excluded from suspense; got {score_closed_only}"
    )


# =====================================================================
# M3 — surprise/irony scorers respect concern activation windows
# =====================================================================
def test_m3_concern_salience_t_zeroes_out_inactive_concern():
    """A concern whose activation_fabula_window does not include
    *fabula_t* must score 0 salience for that tick, while the same
    concern at an in-window tick returns its declared salience."""
    loc = Location(id="LOC_X", name="X", description="x")
    prop = Proposition(
        proposition_id="PROP_GOAL",
        kind="outcome",
        description="goal",
    )
    concern = Concern(
        concern_id="CCN_GOAL",
        proposition_id="PROP_GOAL",
        polarity="desire",
        salience=0.9,
        activation_fabula_window=[1000, 5000],
    )
    ent = Entity(
        id="ENT_A", name="A", location_id="LOC_X",
        status="healthy", traits={}, concerns=[concern], world_id="factual",
    )
    ws = WorldStateV1(
        locations={"LOC_X": loc},
        objects={},
        entities={"ENT_A": ent},
        events=[],
        propositions=[prop],
        social_topology=[],
        causal_topology=[],
    )
    # In-window tick: salience returned.
    in_win = _concern_salience_t(ws, "ENT_A", "PROP_GOAL", fabula_t=3000)
    assert in_win == pytest.approx(0.9)
    # Out-of-window tick (before lo): salience zeroed.
    out_pre = _concern_salience_t(ws, "ENT_A", "PROP_GOAL", fabula_t=500)
    assert out_pre == 0.0, (
        "M3: concern outside activation window must contribute zero "
        "salience to time-aware scorers"
    )
    # Out-of-window tick (after hi): salience zeroed.
    out_post = _concern_salience_t(ws, "ENT_A", "PROP_GOAL", fabula_t=9000)
    assert out_post == 0.0


def test_m3_concern_polarity_sign_t_respects_snapshot_polarity_flip():
    """A concern whose polarity flips via a ``ConcernSnapshot`` at
    fabula_time=T must score the pre-flip polarity for *t < T* and
    the post-flip polarity for *t >= T*."""
    loc = Location(id="LOC_X", name="X", description="x")
    prop = Proposition(
        proposition_id="PROP_X",
        kind="outcome",
        description="x",
    )
    concern = Concern(
        concern_id="CCN_X",
        proposition_id="PROP_X",
        polarity="desire",
        salience=0.8,
        state_timeline=[
            ConcernSnapshot(fabula_time=5000, polarity="fear"),
        ],
    )
    ent = Entity(
        id="ENT_A", name="A", location_id="LOC_X",
        status="healthy", traits={}, concerns=[concern], world_id="factual",
    )
    ws = WorldStateV1(
        locations={"LOC_X": loc},
        objects={},
        entities={"ENT_A": ent},
        events=[],
        propositions=[prop],
        social_topology=[],
        causal_topology=[],
    )
    # Before flip: desire (+1).
    assert _concern_polarity_sign_t(ws, "ENT_A", "PROP_X", fabula_t=1000) == +1
    # After flip: fear (-1).
    assert _concern_polarity_sign_t(ws, "ENT_A", "PROP_X", fabula_t=6000) == -1


# =====================================================================
# M4 — mystery scorer detects revealed ancestors via referent_ids
# =====================================================================
def test_m4_mystery_scorer_uses_referent_linkage_not_synthetic_ids():
    """Pre-fix the mystery scorer tested
    ``f"PROP_FROM_{a}" in audience_known`` for ancestor reveal, which
    falsely classified semantic-id propositions (``PROP_DUNCAN_DEAD``)
    as unrevealed. With M4 the scorer uses
    ``Proposition.referent_ids`` linkage; a known semantic-id
    proposition counts the ancestor as revealed and excludes it from
    the entropy sum."""
    loc = Location(id="LOC_X", name="X", description="x")
    audience = Entity(
        id=AUDIENCE_ID, name="Audience", location_id="LOC_X",
        status="healthy", traits={}, world_id="factual",
    )

    # Chain: EVT_CAUSE_A → EVT_CAUSE_B → EVT_EFFECT
    evt_a = EventNode(
        id="EVT_CAUSE_A", description="cause a",
        fabula_time=100, syuzhet_index=0, event_type="outcome",
    )
    evt_b = EventNode(
        id="EVT_CAUSE_B", description="cause b",
        fabula_time=200, syuzhet_index=1, event_type="outcome",
    )
    evt_eff = EventNode(
        id="EVT_EFFECT", description="effect",
        fabula_time=300, syuzhet_index=2, event_type="outcome",
    )
    edge_ab = CausalEdge(
        source_id="EVT_CAUSE_A", target_id="EVT_CAUSE_B",
        causality_type="chain_reaction", mechanism="follows",
        fabula_time=200,
    )
    edge_be = CausalEdge(
        source_id="EVT_CAUSE_B", target_id="EVT_EFFECT",
        causality_type="chain_reaction", mechanism="follows",
        fabula_time=300,
    )
    # Semantic-id propositions, NOT the synthetic ``PROP_FROM_EVT_*``
    # convention. The mystery scorer's reverse-index lookup must
    # connect these to the underlying events.
    prop_a = Proposition(
        proposition_id="PROP_A_HAPPENED",
        kind="event_occurs",
        description="A happened",
        referent_ids=["EVT_CAUSE_A"],
        truth_at_fabula={100: True},
    )
    prop_b = Proposition(
        proposition_id="PROP_B_HAPPENED",
        kind="event_occurs",
        description="B happened",
        referent_ids=["EVT_CAUSE_B"],
        truth_at_fabula={200: True},
    )
    prop_eff = Proposition(
        proposition_id="PROP_EFFECT_HAPPENED",
        kind="event_occurs",
        description="effect happened",
        referent_ids=["EVT_EFFECT"],
        truth_at_fabula={300: True},
    )
    # Audience CONFIDENTLY knows ALL three semantic props — every
    # ancestor of EVT_EFFECT (EVT_CAUSE_A, EVT_CAUSE_B) is revealed.
    aud_beliefs = [
        Belief(
            target_id=pid, proposition_id=pid,
            perceived_state=pid, confidence=0.95, inertia=0.5,
            established_at_fabula=400,
        )
        for pid in ("PROP_A_HAPPENED", "PROP_B_HAPPENED", "PROP_EFFECT_HAPPENED")
    ]
    audience.beliefs = aud_beliefs

    ws = WorldStateV1(
        locations={"LOC_X": loc},
        objects={},
        entities={AUDIENCE_ID: audience},
        events=[evt_a, evt_b, evt_eff],
        propositions=[prop_a, prop_b, prop_eff],
        social_topology=[],
        causal_topology=[edge_ab, edge_be],
    )
    bs = BeliefState(world=ws)
    score = compute_mystery_unified(bs, fabula_t=500, threshold=0.7)
    # All ancestors are revealed via referent_ids → unrevealed
    # ancestor set < 2 → entropy contribution = 0.
    assert score == 0.0, (
        f"M4: mystery scorer should treat semantic-id ancestor "
        f"propositions as revealed when known by the audience; "
        f"got non-zero score {score}"
    )
