# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the seventh-pass deep-audit fixes (2026-05-29).

Covers the ten subtle-bug fixes implemented in this round:

  * **B1/B2/B3** — ingestion ``_mirror_truth_commit_to_inverse``
    helper: mirrors truth commits onto the declared inverse
    proposition so PROP_X / PROP_NOT_X stay symmetric across Phase C
    chunk-truth writes, the post-pass synthesis, and the patch
    surgery.
  * **B4** — :meth:`CausalPhysicsEngine._apply_do_event_time_shift`
    relocates the inverse-proposition ledger entry in lockstep with
    the primary ledger when DoEventTimeShift moves a committing
    event.
  * **A6** — :meth:`WorldStateV1.projected_for_branch` nullifies
    dangling ``Belief.acquired_via_channel_id`` references (top-level
    and per-snapshot ``beliefs_added``) when the cited channel is
    tombstoned on the projected branch.
  * **A7** — :func:`_demote_social_metric_axes` rewinds
    ``RelationshipMetric.last_updated_fabula`` to the latest surviving
    ``mutation_social`` tick when the originating edge is removed.
  * **A2** — shadow-branch merge wires ``topology.removed_event_ids``
    into ``WorldStateV1.shadow_removed_event_ids[branch_label]`` so
    ``projected_for_branch`` can suppress them.
  * **C7** — POV filter prunes ``social_topology`` to dyads where the
    POV is at least one endpoint.
  * **C8** — POV filter prunes ``spatial_topology`` to edges where
    at least one endpoint is in the POV-known location set.
  * **C9** — POV filter scrubs ``beliefs_added`` /
    ``beliefs_invalidated`` on retained snapshots of non-POV entities
    so per-tick deltas don't leak alongside the cleared top-level
    belief list.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import networkx as nx

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.ingestion import _mirror_truth_commit_to_inverse
from shadow_loom.models import (
    Belief,
    Channel,
    Entity,
    EntityStateSnapshot,
    EventNode,
    Location,
    Proposition,
    RelationshipEdge,
    RelationshipMetric,
    SpatialEdge,
    WorldStateV1,
)
from shadow_loom.projections import filter_world_state_for_pov
from shadow_loom.query_models import DoEvent


# =====================================================================
# B1/B2/B3 — inverse-prop mirror helper
# =====================================================================
def _prop(pid: str, truth=None, inverse=None) -> Proposition:
    return Proposition(
        proposition_id=pid,
        kind="event_occurs",
        description=f"{pid} description",
        truth_at_fabula=truth or {},
        inverse_proposition_id=inverse,
    )


def test_mirror_writes_flipped_value_onto_inverse():
    prim = _prop("PROP_DUNCAN_ALIVE", inverse="PROP_DUNCAN_DEAD")
    inv = _prop("PROP_DUNCAN_DEAD", inverse="PROP_DUNCAN_ALIVE")
    index = {"PROP_DUNCAN_ALIVE": prim, "PROP_DUNCAN_DEAD": inv}

    out_id = _mirror_truth_commit_to_inverse(
        index, "PROP_DUNCAN_ALIVE", fab=5000, val=False,
    )
    assert out_id == "PROP_DUNCAN_DEAD"
    # Mirror flips False → True at the same fabula tick.
    assert index["PROP_DUNCAN_DEAD"].truth_at_fabula == {5000: True}
    # Primary is not mutated by the mirror helper itself.
    assert index["PROP_DUNCAN_ALIVE"].truth_at_fabula == {}


def test_mirror_noop_when_no_inverse_declared():
    prim = _prop("PROP_X")
    index = {"PROP_X": prim}
    out_id = _mirror_truth_commit_to_inverse(index, "PROP_X", fab=100, val=True)
    assert out_id is None


def test_mirror_noop_when_inverse_missing_from_index():
    prim = _prop("PROP_X", inverse="PROP_NOT_X")
    index = {"PROP_X": prim}  # PROP_NOT_X not loaded
    out_id = _mirror_truth_commit_to_inverse(index, "PROP_X", fab=100, val=True)
    assert out_id is None


def test_mirror_idempotent_when_inverse_already_carries_flipped_value():
    prim = _prop("PROP_X", inverse="PROP_NOT_X")
    inv = _prop("PROP_NOT_X", truth={100: False}, inverse="PROP_X")
    index = {"PROP_X": prim, "PROP_NOT_X": inv}
    # PROP_X = True at 100 → inverse must be False at 100. Already so.
    out_id = _mirror_truth_commit_to_inverse(index, "PROP_X", fab=100, val=True)
    assert out_id == "PROP_NOT_X"
    assert index["PROP_NOT_X"].truth_at_fabula == {100: False}


def test_mirror_overwrites_conflicting_existing_value(caplog):
    prim = _prop("PROP_X", inverse="PROP_NOT_X")
    # Inverse pre-pinned with the WRONG value at the same tick.
    inv = _prop("PROP_NOT_X", truth={100: True}, inverse="PROP_X")
    index = {"PROP_X": prim, "PROP_NOT_X": inv}
    with caplog.at_level("WARNING"):
        out_id = _mirror_truth_commit_to_inverse(
            index, "PROP_X", fab=100, val=True,
        )
    assert out_id == "PROP_NOT_X"
    # Last-write-wins: mirror overwrote the conflict.
    assert index["PROP_NOT_X"].truth_at_fabula == {100: False}
    assert any("inverse-mirror" in r.message for r in caplog.records)


def test_mirror_returns_none_for_unknown_primary_pid():
    index = {"PROP_X": _prop("PROP_X")}
    assert _mirror_truth_commit_to_inverse(
        index, "PROP_UNKNOWN", fab=1, val=True,
    ) is None


# =====================================================================
# B4 — DoEventTimeShift relocates the inverse-proposition ledger
# =====================================================================
def _ws_with_event_committing_pair() -> WorldStateV1:
    """Event commits PROP_DESI_DEAD@True at 13000; inverse
    PROP_DESI_ALIVE carries the mirror @False at the same tick."""
    flat = Location(id="LOC_FLAT", name="The flat", description="A flat")
    amy = Entity(
        id="ENT_AMY", name="Amy", location_id="LOC_FLAT", status="healthy",
        traits={}, beliefs=[], world_id="factual",
    )
    desi = Entity(
        id="ENT_DESI", name="Desi", location_id="LOC_FLAT", status="dead",
        traits={}, beliefs=[], world_id="factual",
    )
    evt = EventNode(
        id="EVT_AMY_KILLS_DESI",
        description="Amy kills Desi",
        fabula_time=13000,
        syuzhet_index=10,
        event_type="outcome",
        at_location_id="LOC_FLAT",
        actor_ids=["ENT_AMY"],
        resolves_proposition_ids=["PROP_DESI_DEAD"],
    )
    prim = Proposition(
        proposition_id="PROP_DESI_DEAD",
        kind="outcome",
        description="Desi is dead.",
        truth_at_fabula={13000: True},
        inverse_proposition_id="PROP_DESI_ALIVE",
        world_id="factual",
    )
    inv = Proposition(
        proposition_id="PROP_DESI_ALIVE",
        kind="outcome",
        description="Desi is alive.",
        truth_at_fabula={13000: False},
        inverse_proposition_id="PROP_DESI_DEAD",
        world_id="factual",
    )
    return WorldStateV1(
        locations={"LOC_FLAT": flat},
        objects={},
        entities={"ENT_AMY": amy, "ENT_DESI": desi},
        events=[evt],
        propositions=[prim, inv],
        social_topology=[],
        causal_topology=[],
    )


def test_do_event_time_shift_relocates_inverse_proposition_ledger():
    ws = _ws_with_event_committing_pair()
    engine = CausalPhysicsEngine(nx.MultiDiGraph(), ws)
    engine.apply_do_targets([
        DoEvent(
            event_id="EVT_AMY_KILLS_DESI",
            occurred=True,
            new_fabula_time=18000,
        ),
    ])
    prim = next(p for p in ws.propositions if p.proposition_id == "PROP_DESI_DEAD")
    inv = next(p for p in ws.propositions if p.proposition_id == "PROP_DESI_ALIVE")
    # Primary ledger moved 13000 → 18000.
    assert prim.truth_at_fabula == {18000: True}
    # Inverse ledger must move in lockstep with the flipped value.
    assert inv.truth_at_fabula == {18000: False}, (
        f"inverse ledger should follow the shifted committer; "
        f"got {inv.truth_at_fabula}"
    )


# =====================================================================
# A6 — projected_for_branch nullifies dangling acquired_via_channel_id
# =====================================================================
def _ws_with_belief_via_channel() -> WorldStateV1:
    """Entity carries a Belief whose ``acquired_via_channel_id``
    points at CHN_X (top level + inside a state_timeline snapshot)."""
    loc = Location(id="LOC_X", name="X", description="x")
    chn = Channel(
        id="CHN_X",
        name="The channel",
        description="A channel",
        medium="speech",
        participant_ids=["ENT_A", "ENT_B"],
        intelligibility={"ENT_A": 1.0, "ENT_B": 1.0},
    )
    prop = Proposition(
        proposition_id="PROP_P",
        kind="event_occurs",
        description="P holds",
    )
    blf = Belief(
        target_id="PROP_P",
        proposition_id="PROP_P",
        perceived_state="P holds",
        confidence=0.8,
        inertia=0.5,
        established_at_fabula=1000,
        acquired_via_channel_id="CHN_X",
    )
    snap_blf = Belief(
        target_id="PROP_P",
        proposition_id="PROP_P",
        perceived_state="P holds (snap)",
        confidence=0.7,
        inertia=0.5,
        established_at_fabula=1500,
        acquired_via_channel_id="CHN_X",
    )
    ent_a = Entity(
        id="ENT_A", name="A", location_id="LOC_X",
        status="healthy", traits={},
        beliefs=[blf],
        state_timeline=[
            EntityStateSnapshot(
                fabula_time=1500,
                beliefs_added=[snap_blf],
            ),
        ],
        world_id="factual",
    )
    ent_b = Entity(
        id="ENT_B", name="B", location_id="LOC_X",
        status="healthy", traits={}, world_id="factual",
    )
    return WorldStateV1(
        locations={"LOC_X": loc},
        objects={},
        entities={"ENT_A": ent_a, "ENT_B": ent_b},
        channels={"CHN_X": chn},
        events=[],
        propositions=[prop],
        social_topology=[],
        causal_topology=[],
    )


def test_projected_for_branch_scrubs_acquired_via_channel_id_for_tombstoned_channel():
    ws = _ws_with_belief_via_channel()
    # Tombstone CHN_X on the shadow branch.
    ws.shadow_removed_channel_ids = {"branch_a": ["CHN_X"]}

    proj = ws.projected_for_branch("shadow", "branch_a")

    # Channel is gone from the projected view.
    assert "CHN_X" not in proj.channels
    # Top-level belief's acquired_via_channel_id is nulled.
    ent_a = proj.entities["ENT_A"]
    assert ent_a.beliefs[0].acquired_via_channel_id is None
    # Snapshot-level beliefs_added also nulled.
    assert ent_a.state_timeline[0].beliefs_added[0].acquired_via_channel_id is None
    # Factual world is unmutated.
    assert ws.entities["ENT_A"].beliefs[0].acquired_via_channel_id == "CHN_X"
    assert (
        ws.entities["ENT_A"].state_timeline[0].beliefs_added[0].acquired_via_channel_id
        == "CHN_X"
    )


# =====================================================================
# A7 — _demote_social_metric_axes rewinds last_updated_fabula
# =====================================================================
def test_demote_social_metric_axes_rewinds_last_updated_to_surviving_tick():
    from shadow_loom.extract_graph import _demote_social_metric_axes
    from shadow_loom.models import CausalEdge

    # Two mutation_social edges on the same axis at ticks 1000 and 5000.
    # The 5000 edge is being removed; the 1000 edge survives.
    surviving = CausalEdge(
        source_id="EVT_EARLY",
        target_id="ENT_A",
        rel_counterpart_id="ENT_B",
        causality_type="mutation_social",
        trait_target="affinity",
        trait_delta=0.2,
        mechanism="early bump",
        fabula_time=1000,
    )
    rel = RelationshipEdge(
        source_entity_id="ENT_A",
        target_entity_id="ENT_B",
        metrics={
            "affinity": RelationshipMetric(
                value=0.6,
                evidence_strength="strong",
                last_updated_fabula=5000,  # stamp from the removed edge
            ),
        },
        world_id="factual",
    )
    n = _demote_social_metric_axes(
        social_topology=[rel],
        axes=[("ENT_A", "ENT_B", "affinity")],
        merge_world_id="factual",
        source="delete",
        remaining_causal_topology=[surviving],
    )
    assert n == 1
    metric = rel.metrics["affinity"]
    assert metric.evidence_strength == "weak"
    # last_updated rewound to the surviving edge's tick.
    assert metric.last_updated_fabula == 1000


def test_demote_social_metric_axes_rewinds_to_zero_when_no_survivor():
    from shadow_loom.extract_graph import _demote_social_metric_axes

    rel = RelationshipEdge(
        source_entity_id="ENT_A",
        target_entity_id="ENT_B",
        metrics={
            "affinity": RelationshipMetric(
                value=0.6,
                evidence_strength="moderate",
                last_updated_fabula=5000,
            ),
        },
        world_id="factual",
    )
    _demote_social_metric_axes(
        social_topology=[rel],
        axes=[("ENT_A", "ENT_B", "affinity")],
        merge_world_id="factual",
        source="suppress",
        remaining_causal_topology=[],  # no surviving justification
    )
    metric = rel.metrics["affinity"]
    assert metric.evidence_strength == "weak"
    assert metric.last_updated_fabula == 0


def test_demote_social_metric_axes_does_not_rewind_when_kwarg_absent():
    """Back-compat: pre-A7 callers that omit ``remaining_causal_topology``
    must keep getting the legacy behaviour (evidence demoted, timestamp
    untouched)."""
    from shadow_loom.extract_graph import _demote_social_metric_axes

    rel = RelationshipEdge(
        source_entity_id="ENT_A",
        target_entity_id="ENT_B",
        metrics={
            "affinity": RelationshipMetric(
                value=0.6,
                evidence_strength="strong",
                last_updated_fabula=5000,
            ),
        },
        world_id="factual",
    )
    _demote_social_metric_axes(
        social_topology=[rel],
        axes=[("ENT_A", "ENT_B", "affinity")],
        merge_world_id="factual",
        source="delete",
    )
    metric = rel.metrics["affinity"]
    assert metric.evidence_strength == "weak"
    # Timestamp unchanged when caller did not supply the topology kwarg.
    assert metric.last_updated_fabula == 5000


# =====================================================================
# A2 — shadow merge wires removed_event_ids into shadow_removed_event_ids
# =====================================================================
def test_extract_graph_merge_wires_removed_event_ids_into_shadow_sidecar():
    """Source-level check: the seventh-pass A2 fix added a block to
    mirror ``topology.removed_event_ids`` into
    ``merged.shadow_removed_event_ids[branch_label]`` alongside the
    existing channel-tombstone wiring. A regression that drops this
    block would silently leave operator-issued shadow deletions
    invisible to ``projected_for_branch`` replay.
    """
    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "extract_graph.py"
    ).read_text(encoding="utf-8")
    assert "shadow_removed_event_ids.setdefault(branch_label" in src, (
        "A2 wiring missing: removed_event_ids must be mirrored into "
        "shadow_removed_event_ids on shadow merge."
    )
    # And the surrounding loop must read from topology.removed_event_ids.
    assert "topology.removed_event_ids" in src


# =====================================================================
# C7 / C8 / C9 — POV filter coverage of social/spatial/snapshot leaks
# =====================================================================
def _ws_for_pov_filter() -> WorldStateV1:
    """World with a POV (ENT_POV), two non-POV entities (ENT_X, ENT_Y),
    multiple social edges (POV↔X, X↔Y), multiple locations and
    spatial edges (POV at LOC_HOME → LOC_HALL; LOC_FAR → LOC_NOWHERE
    completely off-map), and non-POV snapshots carrying belief
    deltas the C9 fix must scrub."""
    loc_home = Location(id="LOC_HOME", name="Home", description="home")
    loc_hall = Location(id="LOC_HALL", name="Hall", description="hall")
    loc_far = Location(id="LOC_FAR", name="Far", description="far")
    loc_nowhere = Location(id="LOC_NOWHERE", name="Nowhere", description="nowhere")

    prop = Proposition(
        proposition_id="PROP_OFFSCREEN",
        kind="event_occurs",
        description="something",
    )
    leaked_belief = Belief(
        target_id="PROP_OFFSCREEN",
        proposition_id="PROP_OFFSCREEN",
        perceived_state="off-page realisation",
        confidence=0.9,
        inertia=0.5,
        established_at_fabula=500,
    )

    pov = Entity(
        id="ENT_POV", name="POV", location_id="LOC_HOME",
        status="healthy", traits={}, world_id="factual",
    )
    ent_x = Entity(
        id="ENT_X", name="X", location_id="LOC_HOME",
        status="healthy", traits={},
        beliefs=[deepcopy(leaked_belief)],  # top-level (C-scrubbed already)
        state_timeline=[
            EntityStateSnapshot(
                fabula_time=500,
                # POV cannot witness this — the snapshot's belief
                # deltas must be scrubbed even if the snapshot itself
                # survives via fabula-tick gating.
                beliefs_added=[deepcopy(leaked_belief)],
                beliefs_invalidated=["PROP_PRIOR"],
            ),
        ],
        world_id="factual",
    )
    ent_y = Entity(
        id="ENT_Y", name="Y", location_id="LOC_FAR",
        status="healthy", traits={}, world_id="factual",
    )

    # Social edges:
    #   POV ↔ X  (one endpoint is POV — must SURVIVE)
    #   X ↔ Y    (neither endpoint is POV — must be PRUNED)
    pov_x = RelationshipEdge(
        source_entity_id="ENT_POV", target_entity_id="ENT_X",
        metrics={"affinity": RelationshipMetric(value=0.5)},
        world_id="factual",
    )
    x_y = RelationshipEdge(
        source_entity_id="ENT_X", target_entity_id="ENT_Y",
        metrics={"affinity": RelationshipMetric(value=-0.8)},
        world_id="factual",
    )

    # Spatial edges:
    #   HOME → HALL  (HOME is POV-known — must SURVIVE)
    #   FAR → NOWHERE  (neither endpoint in POV-known set — must be PRUNED)
    home_hall = SpatialEdge(
        source_id="LOC_HOME", target_id="LOC_HALL",
        relation_type="adjacent_to",
    )
    far_nowhere = SpatialEdge(
        source_id="LOC_FAR", target_id="LOC_NOWHERE",
        relation_type="adjacent_to",
    )

    # An event ENT_POV can witness so visible_evt_ids is non-empty and
    # the entity-loop in filter_world_state_for_pov runs.
    evt = EventNode(
        id="EVT_AT_HOME",
        description="something happens at home",
        fabula_time=500,
        syuzhet_index=0,
        event_type="outcome",
        at_location_id="LOC_HOME",
        actor_ids=["ENT_POV", "ENT_X"],
    )

    return WorldStateV1(
        locations={
            "LOC_HOME": loc_home, "LOC_HALL": loc_hall,
            "LOC_FAR": loc_far, "LOC_NOWHERE": loc_nowhere,
        },
        objects={},
        entities={"ENT_POV": pov, "ENT_X": ent_x, "ENT_Y": ent_y},
        events=[evt],
        propositions=[prop],
        social_topology=[pov_x, x_y],
        spatial_topology=[home_hall, far_nowhere],
        causal_topology=[],
    )


def test_c7_pov_filter_prunes_social_topology_between_non_pov_entities():
    ws = _ws_for_pov_filter()
    filtered = filter_world_state_for_pov(ws, pov_entity_id="ENT_POV")
    # POV↔X must remain.
    pairs = {
        (re.source_entity_id, re.target_entity_id)
        for re in filtered.social_topology
    }
    # X↔Y leaked private rivalry between non-POV characters → pruned.
    assert ("ENT_X", "ENT_Y") not in pairs
    # POV's own dyad survives.
    assert any(
        "ENT_POV" in (re.source_entity_id, re.target_entity_id)
        for re in filtered.social_topology
    )


def test_c8_pov_filter_prunes_spatial_topology_between_unknown_locations():
    ws = _ws_for_pov_filter()
    filtered = filter_world_state_for_pov(ws, pov_entity_id="ENT_POV")
    pairs = {(se.source_id, se.target_id) for se in filtered.spatial_topology}
    # FAR → NOWHERE: neither endpoint POV-known → pruned.
    assert ("LOC_FAR", "LOC_NOWHERE") not in pairs
    # HOME → HALL: HOME is POV-known (POV's own location) → retained.
    assert ("LOC_HOME", "LOC_HALL") in pairs


def test_c9_pov_filter_scrubs_snapshot_belief_deltas_on_non_pov_entities():
    ws = _ws_for_pov_filter()
    filtered = filter_world_state_for_pov(ws, pov_entity_id="ENT_POV")
    ent_x = filtered.entities["ENT_X"]
    # Top-level scrub still in force.
    assert ent_x.beliefs == []
    # And C9 specifically: snapshot-level deltas cleared on retained
    # snapshots.
    for snap in ent_x.state_timeline:
        assert snap.beliefs_added == [], (
            "non-POV snapshot beliefs_added must be scrubbed to avoid "
            "leaking other characters' belief formation"
        )
        assert snap.beliefs_invalidated == []


def test_c9_pov_keeps_their_own_snapshot_belief_deltas():
    """Sanity: the POV's OWN snapshots must NOT be scrubbed —
    only the non-POV entity loop runs the C9 cleaner."""
    ws = _ws_for_pov_filter()
    pov_belief = Belief(
        target_id="PROP_OFFSCREEN",
        proposition_id="PROP_OFFSCREEN",
        perceived_state="POV's own realisation",
        confidence=1.0,
        inertia=0.5,
        established_at_fabula=500,
    )
    ws.entities["ENT_POV"].state_timeline = [
        EntityStateSnapshot(
            fabula_time=500,
            beliefs_added=[pov_belief],
        ),
    ]
    filtered = filter_world_state_for_pov(ws, pov_entity_id="ENT_POV")
    pov_out = filtered.entities["ENT_POV"]
    assert pov_out.state_timeline
    assert pov_out.state_timeline[0].beliefs_added, (
        "POV's own snapshot beliefs_added must be preserved"
    )
