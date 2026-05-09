# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the generic ``_coalesce_timeline`` helper.

Covers the snapshot-coalescence semantics shared across
``Proposition.state_timeline`` and ``Concern.state_timeline``:

- empty / no-op snapshots are dropped;
- snapshots at the same ``(fabula_time, triggered_by)`` merge their
  first non-None per-field values;
- output is sorted by ``(fabula_time, triggered_by)``;
- diff-fields are configurable per timeline.
"""

from __future__ import annotations

from shadow_loom.affect_unification import _coalesce_timeline
from shadow_loom.models import ConcernSnapshot, PropositionSnapshot


_PROP_FIELDS = ("stakes", "audience_default_prior", "description")
_CCN_FIELDS = (
    "salience", "polarity", "activation_fabula_window",
    "counter_concern_ids", "kind",
)


def test_coalesce_drops_no_op_snapshot():
    snaps = [
        PropositionSnapshot(fabula_time=10, triggered_by="EVT_X"),  # all None
        PropositionSnapshot(fabula_time=20, triggered_by="EVT_Y", stakes=0.7),
    ]
    out = _coalesce_timeline(snaps, _PROP_FIELDS)
    assert len(out) == 1
    assert out[0].fabula_time == 20
    assert out[0].stakes == 0.7


def test_coalesce_merges_same_key_snapshots():
    snaps = [
        PropositionSnapshot(
            fabula_time=20, triggered_by="EVT_Y", stakes=0.5,
        ),
        PropositionSnapshot(
            fabula_time=20, triggered_by="EVT_Y", description="reframed",
        ),
    ]
    out = _coalesce_timeline(snaps, _PROP_FIELDS)
    assert len(out) == 1
    assert out[0].stakes == 0.5
    assert out[0].description == "reframed"


def test_coalesce_keeps_first_nonnone_on_conflict():
    snaps = [
        PropositionSnapshot(fabula_time=20, triggered_by="EVT_Y", stakes=0.5),
        PropositionSnapshot(fabula_time=20, triggered_by="EVT_Y", stakes=0.9),
    ]
    out = _coalesce_timeline(snaps, _PROP_FIELDS)
    assert len(out) == 1
    # First non-None wins (input order).
    assert out[0].stakes == 0.5


def test_coalesce_sorts_by_fabula_then_trigger():
    snaps = [
        PropositionSnapshot(fabula_time=30, triggered_by="EVT_Z", stakes=0.3),
        PropositionSnapshot(fabula_time=10, triggered_by="EVT_B", stakes=0.1),
        PropositionSnapshot(fabula_time=10, triggered_by="EVT_A", stakes=0.2),
    ]
    out = _coalesce_timeline(snaps, _PROP_FIELDS)
    keys = [(s.fabula_time, s.triggered_by) for s in out]
    assert keys == [(10, "EVT_A"), (10, "EVT_B"), (30, "EVT_Z")]


def test_coalesce_concern_snapshot_polarity_flip():
    snaps = [
        ConcernSnapshot(fabula_time=50, triggered_by="EVT_BETRAY", polarity="fear"),
        ConcernSnapshot(fabula_time=20, triggered_by="EVT_HOPE", polarity="desire"),
    ]
    out = _coalesce_timeline(snaps, _CCN_FIELDS)
    assert [s.polarity for s in out] == ["desire", "fear"]
    assert [s.fabula_time for s in out] == [20, 50]


def test_coalesce_empty_input_returns_empty():
    assert _coalesce_timeline([], _PROP_FIELDS) == []


def test_coalesce_distinct_triggers_at_same_fabula_kept_separate():
    snaps = [
        PropositionSnapshot(fabula_time=10, triggered_by="EVT_A", stakes=0.1),
        PropositionSnapshot(fabula_time=10, triggered_by="EVT_B", stakes=0.2),
    ]
    out = _coalesce_timeline(snaps, _PROP_FIELDS)
    assert len(out) == 2
    assert {s.stakes for s in out} == {0.1, 0.2}
