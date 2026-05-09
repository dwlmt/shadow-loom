# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the affect-merge pass in ``VersionedWorldModel.merge``.

Exercises new propositions, truth commits, framing snapshots, concern
snapshots, and idempotency on re-merge.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import (
    ChunkConcernSnapshot,
    ChunkPropositionSnapshot,
    ChunkTopology,
    PropositionTruthCommit,
)
from shadow_loom.models import Concern, Proposition

from example_worlds.macbeth import world_state as macbeth_ws


def _empty_topology() -> ChunkTopology:
    return ChunkTopology()


# =====================================================================
# new_propositions
# =====================================================================
class TestNewPropositions:
    def test_merge_adds_proposition_with_changeset_counter(self):
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = _empty_topology()
        topo.new_propositions["PROP_NEW_X"] = Proposition(
            proposition_id="PROP_NEW_X",
            kind="trait_holds",
            referent_ids=[next(iter(macbeth_ws.entities))],
            description="Newly introduced proposition.",
        )
        vwm_next = vwm.merge(topo, source="test", description="add prop")
        assert any(
            p.proposition_id == "PROP_NEW_X"
            for p in vwm_next.current.propositions
        )
        assert vwm_next.history[-1].changeset.propositions_added == 1

    def test_merge_idempotent_on_duplicate_proposition(self):
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = _empty_topology()
        prop = Proposition(
            proposition_id="PROP_DUP",
            kind="trait_holds",
            referent_ids=[next(iter(macbeth_ws.entities))],
            description="Will be re-merged.",
        )
        topo.new_propositions["PROP_DUP"] = prop
        vwm_next = vwm.merge(topo, source="test", description="add")
        # Re-apply
        vwm_again = vwm_next.merge(topo, source="test", description="re-add")
        # PROP_DUP appears exactly once.
        ids = [p.proposition_id for p in vwm_again.current.propositions]
        assert ids.count("PROP_DUP") == 1
        assert vwm_again.history[-1].changeset.propositions_added == 0


# =====================================================================
# Truth commits
# =====================================================================
class TestTruthCommits:
    def test_truth_commit_lands_on_proposition(self):
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = _empty_topology()
        topo.new_propositions["PROP_TC"] = Proposition(
            proposition_id="PROP_TC",
            kind="trait_holds",
            referent_ids=[next(iter(macbeth_ws.entities))],
            description="Will be committed true.",
        )
        topo.proposition_truth_commits.append(
            PropositionTruthCommit(
                proposition_id="PROP_TC",
                fabula_time=999,
                truth=True,
                triggered_by="EVT_FAKE",
            )
        )
        vwm_next = vwm.merge(topo, source="test", description="commit")
        prop = next(
            p for p in vwm_next.current.propositions if p.proposition_id == "PROP_TC"
        )
        assert prop.truth_at_fabula.get(999) is True
        assert vwm_next.history[-1].changeset.proposition_truths_committed == 1


# =====================================================================
# Snapshots are append-only and dedup on re-merge
# =====================================================================
class TestSnapshots:
    def test_proposition_snapshot_appended(self):
        vwm = VersionedWorldModel.from_world_state(deepcopy(macbeth_ws))
        topo = _empty_topology()
        topo.new_propositions["PROP_SNAP"] = Proposition(
            proposition_id="PROP_SNAP",
            kind="trait_holds",
            referent_ids=[next(iter(macbeth_ws.entities))],
            description="Snapshot target.",
        )
        topo.proposition_snapshots.append(
            ChunkPropositionSnapshot(
                proposition_id="PROP_SNAP",
                fabula_time=10,
                triggered_by="EVT_FAKE",
                stakes=0.7,
            )
        )
        vwm_next = vwm.merge(topo, source="test", description="snap")
        prop = next(
            p for p in vwm_next.current.propositions if p.proposition_id == "PROP_SNAP"
        )
        assert len(prop.state_timeline) == 1
        # Re-merge: still 1 (dedup on (fabula_time, triggered_by, world_id)).
        vwm_again = vwm_next.merge(topo, source="test", description="re-snap")
        prop2 = next(
            p for p in vwm_again.current.propositions if p.proposition_id == "PROP_SNAP"
        )
        assert len(prop2.state_timeline) == 1

    def test_concern_snapshot_appended_to_matching_concern(self):
        ws = deepcopy(macbeth_ws)
        first_eid = next(iter(ws.entities))
        ws.propositions = list(ws.propositions) + [
            Proposition(
                proposition_id="PROP_C",
                kind="trait_holds",
                referent_ids=[first_eid],
                description="Concern target.",
            )
        ]
        ws.entities[first_eid].concerns = list(ws.entities[first_eid].concerns) + [
            Concern(
                concern_id="CCN_C",
                proposition_id="PROP_C",
                polarity="fear",
                salience=0.4,
            )
        ]
        vwm = VersionedWorldModel.from_world_state(ws)
        topo = _empty_topology()
        topo.concern_snapshots.append(
            ChunkConcernSnapshot(
                concern_id="CCN_C",
                fabula_time=20,
                triggered_by="EVT_FAKE",
                salience=0.85,
            )
        )
        vwm_next = vwm.merge(topo, source="test", description="csnap")
        c = next(
            c for c in vwm_next.current.entities[first_eid].concerns
            if c.concern_id == "CCN_C"
        )
        assert len(c.state_timeline) == 1
        assert vwm_next.history[-1].changeset.concern_snapshots_added == 1
