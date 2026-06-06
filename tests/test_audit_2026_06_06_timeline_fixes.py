# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-06-06 scenario audit.

Finding: ``extract_full_world_state`` (the omniscient view feeding
interrogation / observation / renderer / auditor prompts) leaked
*dangling causal edges* — edges whose EVENT endpoint was pruned by the
fabula or syuzhet anchor still survived, exposing a not-yet-narrated
(future) event id to consumers that walk ``causal_topology``.

Two facets of one bug:

  1. The syuzhet-path cascade keyed on ``cause_id`` / ``effect_id`` —
     fields that do not exist on :class:`CausalEdge` (the real fields
     are ``source_id`` / ``target_id``) — so the prune was a silent
     no-op.
  2. The fabula path had no endpoint cascade at all; it filtered edges
     only by ``ce.fabula_time <= t``.

Fix: a single cascade after the fabula/syuzhet slice AND the
supersession prune drops any causal edge whose EVT_ endpoint is no
longer in the surviving event set.
"""
from __future__ import annotations

import importlib

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, CausalEdge, TraitVector,
)
from shadow_loom.extract_graph import extract_full_world_state


def _chain_world() -> WorldStateV1:
    """EVT_A(t=1,s=1) --chain--> EVT_B(t=5,s=5) --mutation--> ENT_T."""
    return WorldStateV1(
        locations={
            "LOC_A": Location(id="LOC_A", name="A", description="A", ambient_state={}),
        },
        objects={},
        entities={
            "ENT_T": Entity(
                id="ENT_T", name="Target", location_id="LOC_A", status="healthy",
                traits={"fear": TraitVector(value=0.3, inertia=0.1)},
            ),
        },
        events=[
            EventNode(id="EVT_A", fabula_time=1, syuzhet_index=1,
                      event_type="choice", actor_ids=[], target_ids=[],
                      description="A"),
            EventNode(id="EVT_B", fabula_time=5, syuzhet_index=5,
                      event_type="outcome", actor_ids=[], target_ids=["ENT_T"],
                      description="B"),
        ],
        causal_topology=[
            CausalEdge(source_id="EVT_A", target_id="EVT_B",
                       causality_type="chain_reaction", causal_force=5.0,
                       mechanism="physical", evidence_strength="strong",
                       fabula_time=1),
            CausalEdge(source_id="EVT_B", target_id="ENT_T",
                       causality_type="mutation", causal_force=5.0,
                       mechanism="physical", evidence_strength="strong",
                       fabula_time=5, trait_target="fear", trait_delta=0.5),
        ],
        spatial_topology=[],
    )


def _dangling(dump, all_event_ids):
    surviving = {e.get("id") for e in dump.get("events", []) if e.get("id")}
    out = []
    for c in dump.get("causal_topology", []):
        for endp in (c.get("source_id"), c.get("target_id")):
            if endp in all_event_ids and endp not in surviving:
                out.append((c.get("source_id"), c.get("target_id")))
                break
    return out


class TestOmniscientNoDanglingCausalEdges:

    def test_fabula_anchor_drops_edge_into_pruned_event(self):
        ws = _chain_world()
        # Anchor between A (t=1) and B (t=5): B is pruned.
        dump = extract_full_world_state(ws, temporal_anchor=3)
        ids = {"EVT_A", "EVT_B"}
        assert "EVT_B" not in {e["id"] for e in dump["events"]}
        # The EVT_A -> EVT_B chain edge must be cascaded away;
        # the EVT_B -> ENT_T edge too (source pruned).
        assert _dangling(dump, ids) == []
        srcs = {(c["source_id"], c["target_id"]) for c in dump["causal_topology"]}
        assert ("EVT_A", "EVT_B") not in srcs
        assert ("EVT_B", "ENT_T") not in srcs

    def test_syuzhet_anchor_drops_edge_into_pruned_event(self):
        ws = _chain_world()
        dump = extract_full_world_state(ws, syuzhet_anchor=3)
        ids = {"EVT_A", "EVT_B"}
        assert "EVT_B" not in {e["id"] for e in dump["events"]}
        assert _dangling(dump, ids) == []

    def test_no_anchor_keeps_all_edges(self):
        ws = _chain_world()
        dump = extract_full_world_state(ws)
        # Without any anchor nothing is event-pruned, so both edges stay.
        assert len(dump["causal_topology"]) == 2

    def test_edge_kept_when_both_endpoints_survive(self):
        ws = _chain_world()
        # Anchor at/after B: both events survive, both edges kept.
        dump = extract_full_world_state(ws, temporal_anchor=5)
        assert {"EVT_A", "EVT_B"} <= {e["id"] for e in dump["events"]}
        assert len(dump["causal_topology"]) == 2


class TestRealWorldCorpusInvariant:
    """Lock the invariant on a real example world so a future field
    rename can't silently re-break the cascade."""

    def test_macbeth_no_dangling_edges_at_mid_anchor(self):
        ws = importlib.import_module("example_worlds.macbeth").world_state
        all_evt = {e.id for e in ws.events}
        ts = sorted({e.fabula_time for e in ws.events})
        mid = ts[len(ts) // 2]
        dump = extract_full_world_state(ws, temporal_anchor=mid)
        assert _dangling(dump, all_evt) == []

        ss = sorted({e.syuzhet_index for e in ws.events})
        smid = ss[len(ss) // 2]
        dump_s = extract_full_world_state(ws, syuzhet_anchor=smid)
        assert _dangling(dump_s, all_evt) == []
