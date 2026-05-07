# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the deletion pass in ``VersionedWorldModel.merge``.

Covers ChunkTopology ``removed_*`` field handling, cascades, and the
matching counters on ``MergeChangeset``.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    Concern,
    Entity,
    Proposition,
)

from example_worlds.macbeth import world_state as macbeth_ws


def _empty_topology() -> ChunkTopology:
    return ChunkTopology()


# =====================================================================
# Event removal cascades to dependent causal/social/spatial edges
# =====================================================================
class TestEventRemoval:
    def test_remove_event_drops_dependent_causal_edges(self):
        ws = deepcopy(macbeth_ws)
        # Pick an event that participates in at least one causal edge.
        edge = next(iter(ws.causal_topology), None)
        if edge is None:
            pytest.skip("Macbeth fixture has no causal edges to test against.")
        target_event_id = edge.source_id
        if not any(e.id == target_event_id for e in ws.events):
            pytest.skip("Causal edge source does not match any event id.")
        vwm = VersionedWorldModel.from_world_state(ws)
        topo = _empty_topology()
        topo.removed_event_ids.append(target_event_id)
        before_edges = len(vwm.current.causal_topology)
        vwm_next = vwm.merge(topo, source="test", description="remove event")
        cs = vwm_next.history[-1].changeset
        assert cs.events_removed >= 1
        assert cs.causal_edges_removed >= 1
        # Removed event is gone.
        assert not any(e.id == target_event_id for e in vwm_next.current.events)
        # No surviving causal edge references it.
        assert all(
            ce.source_id != target_event_id and ce.target_id != target_event_id
            for ce in vwm_next.current.causal_topology
        )
        assert len(vwm_next.current.causal_topology) < before_edges


# =====================================================================
# Entity removal cascades to social edges + cross-entity beliefs
# =====================================================================
class TestEntityRemoval:
    def test_remove_entity_drops_beliefs_targeting_it(self):
        ws = deepcopy(macbeth_ws)
        # Pick any entity referenced by at least one other entity's belief.
        target_id = None
        for eid, ent in ws.entities.items():
            for other_id, other in ws.entities.items():
                if other_id == eid:
                    continue
                if any(b.target_id == eid for b in other.beliefs):
                    target_id = eid
                    break
            if target_id:
                break
        if target_id is None:
            pytest.skip("No cross-entity belief in Macbeth fixture.")
        vwm = VersionedWorldModel.from_world_state(ws)
        topo = _empty_topology()
        topo.removed_entity_ids.append(target_id)
        vwm_next = vwm.merge(topo, source="test", description="remove entity")
        cs = vwm_next.history[-1].changeset
        assert cs.entities_removed == 1
        assert target_id not in vwm_next.current.entities
        # No surviving entity carries a belief pointing at the removed entity.
        for ent in vwm_next.current.entities.values():
            assert all(b.target_id != target_id for b in ent.beliefs)


# =====================================================================
# Proposition removal cascades to dependent concerns
# =====================================================================
class TestPropositionRemoval:
    def test_remove_proposition_drops_dependent_concerns(self):
        ws = deepcopy(macbeth_ws)
        # Inject a proposition + concern referencing it so we control the test.
        ws.propositions = list(ws.propositions) + [
            Proposition(
                proposition_id="PROP_TEST_DEL",
                kind="trait_holds",
                referent_ids=[next(iter(ws.entities))],
                description="Test prop to delete.",
            )
        ]
        first_eid = next(iter(ws.entities))
        ws.entities[first_eid].concerns = list(ws.entities[first_eid].concerns) + [
            Concern(
                concern_id="CCN_TEST_DEL",
                proposition_id="PROP_TEST_DEL",
                polarity="fear",
            )
        ]
        vwm = VersionedWorldModel.from_world_state(ws)
        topo = _empty_topology()
        topo.removed_proposition_ids.append("PROP_TEST_DEL")
        vwm_next = vwm.merge(topo, source="test", description="remove prop")
        cs = vwm_next.history[-1].changeset
        assert cs.propositions_removed == 1
        assert cs.concerns_removed >= 1
        assert all(
            p.proposition_id != "PROP_TEST_DEL"
            for p in vwm_next.current.propositions
        )
        assert all(
            c.proposition_id != "PROP_TEST_DEL"
            for c in vwm_next.current.entities[first_eid].concerns
        )


# =====================================================================
# Targeted concern removal (no cascade)
# =====================================================================
class TestConcernRemoval:
    def test_remove_concern_by_pair(self):
        ws = deepcopy(macbeth_ws)
        ws.propositions = list(ws.propositions) + [
            Proposition(
                proposition_id="PROP_KEEP",
                kind="trait_holds",
                referent_ids=[next(iter(ws.entities))],
                description="Test prop to keep.",
            )
        ]
        first_eid = next(iter(ws.entities))
        ws.entities[first_eid].concerns = list(ws.entities[first_eid].concerns) + [
            Concern(
                concern_id="CCN_DROP_ME",
                proposition_id="PROP_KEEP",
                polarity="desire",
            )
        ]
        vwm = VersionedWorldModel.from_world_state(ws)
        topo = _empty_topology()
        topo.removed_concern_ids.append((first_eid, "CCN_DROP_ME"))
        vwm_next = vwm.merge(topo, source="test", description="drop concern")
        cs = vwm_next.history[-1].changeset
        assert cs.concerns_removed == 1
        # PROP_KEEP must survive (only the concern was targeted).
        assert any(
            p.proposition_id == "PROP_KEEP"
            for p in vwm_next.current.propositions
        )
        assert all(
            c.concern_id != "CCN_DROP_ME"
            for c in vwm_next.current.entities[first_eid].concerns
        )
