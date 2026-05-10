# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the sandbox-delta -> topology bridge.

Covers three behaviours added by the recent shadow-branch work that
were not previously exercised:

* :func:`shadow_loom.pipeline._augment_topology_with_sandbox_deltas`
  cumulatively stacks WorldTraitSnapshot entries when called with
  multiple deltas for the same WORLD_ trait in a single run.
* The same function composes Rung-3 hidden-delta abduction off the
  most-recent stacked snapshot rather than the canonical baseline,
  so mutations + hidden_deltas don't silently overwrite each other.
* :func:`shadow_loom.ingestion._deduplicate_social` resolves ties
  with ``>=`` so a later-appended bridged metric wins against a
  same-tick extractor metric (this is the contract the bridge
  relies on; without it bridged social mutations would be dropped).
"""

from __future__ import annotations

from shadow_loom.ingestion import ChunkTopology, _deduplicate_social
from shadow_loom.models import (
    Entity,
    GlobalTrait,
    RelationshipEdge,
    RelationshipMetric,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.pipeline import _augment_topology_with_sandbox_deltas


def _ws_with_world_trait(value: float = 0.5, inertia: float = 0.4) -> WorldStateV1:
    return WorldStateV1(
        locations={},
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_X", status="healthy",
                traits={},
            ),
            "ENT_B": Entity(
                id="ENT_B", name="B", location_id="LOC_X", status="healthy",
                traits={},
            ),
        },
        events=[],
        causal_topology=[],
        world_traits={
            "WORLD_FEAR": GlobalTrait(
                id="WORLD_FEAR",
                name="Ambient Fear",
                description="Background dread.",
                category="social_structure",
                magnitude=TraitVector(value=value, inertia=inertia),
                affected_domains=["psychological"],
            ),
        },
    )


# =====================================================================
# WORLD trait cumulative stacking
# =====================================================================


class TestWorldTraitCumulativeStacking:
    def test_two_world_mutations_stack_into_state_timeline(self):
        """Two TraitMutation entries for the same WORLD_ trait at
        different fabula times must both survive in the bridged
        ``new_world_traits`` entry's ``state_timeline``."""
        ws = _ws_with_world_trait(value=0.5)
        topo = ChunkTopology()
        physics_result = {
            "mutations": [
                {"node_id": "WORLD_FEAR", "trait": "intensity", "new_value": 0.7},
                {"node_id": "WORLD_FEAR", "trait": "intensity", "new_value": 0.85},
            ],
        }
        # Simulate two separate fabula ticks by calling the bridge
        # twice — the second call must NOT overwrite the first.
        _augment_topology_with_sandbox_deltas(
            topo, world_state=ws,
            physics_result={"mutations": [physics_result["mutations"][0]]},
            fabula_time_now=100,
        )
        _augment_topology_with_sandbox_deltas(
            topo, world_state=ws,
            physics_result={"mutations": [physics_result["mutations"][1]]},
            fabula_time_now=200,
        )
        bridged = topo.new_world_traits["WORLD_FEAR"]
        assert len(bridged.state_timeline) == 2
        assert bridged.state_timeline[0].fabula_time == 100
        assert bridged.state_timeline[0].magnitude.value == 0.7
        assert bridged.state_timeline[1].fabula_time == 200
        assert bridged.state_timeline[1].magnitude.value == 0.85

    def test_duplicate_snapshot_at_same_tick_skipped(self):
        """Re-running the bridge with the identical delta must not
        produce duplicate snapshots at the same (fabula_time, world_id)."""
        ws = _ws_with_world_trait(value=0.5)
        topo = ChunkTopology()
        delta = {"node_id": "WORLD_FEAR", "trait": "intensity", "new_value": 0.7}
        for _ in range(3):
            _augment_topology_with_sandbox_deltas(
                topo, world_state=ws,
                physics_result={"mutations": [delta]},
                fabula_time_now=100,
            )
        bridged = topo.new_world_traits["WORLD_FEAR"]
        assert len(bridged.state_timeline) == 1


# =====================================================================
# Hidden-delta WORLD baseline composition
# =====================================================================


class TestHiddenDeltaBaseline:
    def test_hidden_delta_composes_off_stacked_value(self):
        """A WORLD_ mutation followed by a hidden_delta on the same
        trait must add the delta to the *stacked* value (not rebase
        off the canonical magnitude). Otherwise the mutation is lost."""
        ws = _ws_with_world_trait(value=0.2)
        topo = ChunkTopology()
        # First: deterministic mutation pushes intensity to 0.7.
        _augment_topology_with_sandbox_deltas(
            topo, world_state=ws,
            physics_result={
                "mutations": [
                    {"node_id": "WORLD_FEAR", "trait": "intensity", "new_value": 0.7},
                ],
            },
            fabula_time_now=100,
        )
        # Second: Rung-3 abduction infers an additional +0.15 historical drift.
        _augment_topology_with_sandbox_deltas(
            topo, world_state=ws,
            physics_result={
                "hidden_deltas": {"WORLD_FEAR": {"intensity": 0.15}},
            },
            fabula_time_now=120,
            fabula_time_historical=50,
        )
        bridged = topo.new_world_traits["WORLD_FEAR"]
        # Two snapshots — mutation at t=100, hidden anchor at t=50.
        assert len(bridged.state_timeline) == 2
        # Hidden snapshot's value is 0.7 + 0.15 = 0.85 (composed off
        # the most-recent stacked snapshot, NOT 0.2 + 0.15 from the
        # canonical baseline).
        hidden_snap = next(
            s for s in bridged.state_timeline if s.fabula_time == 50
        )
        assert hidden_snap.magnitude.value == 0.85


# =====================================================================
# Social mutation tie-break ordering
# =====================================================================


class TestSocialDedupTieBreak:
    def test_later_appended_wins_on_tie(self):
        """``_deduplicate_social`` uses ``>=`` so a later-appended
        edge at the same fabula tick wins. This is the contract the
        sandbox bridge relies on (it appends after the prose extractor
        without inflating ``last_updated_fabula``)."""
        extractor_edge = RelationshipEdge(
            source_entity_id="ENT_A",
            target_entity_id="ENT_B",
            metrics={
                "affinity": RelationshipMetric(
                    value=0.2, inertia=0.3,
                    evidence_strength="weak", last_updated_fabula=100,
                    observed=True,
                ),
            },
        )
        bridged_edge = RelationshipEdge(
            source_entity_id="ENT_A",
            target_entity_id="ENT_B",
            metrics={
                "affinity": RelationshipMetric(
                    value=0.8, inertia=0.3,
                    evidence_strength="moderate", last_updated_fabula=100,
                    observed=True,
                ),
            },
        )
        # Order matters: bridge runs after extraction, so the bridged
        # edge appears later in social_topology.
        result = _deduplicate_social([extractor_edge, bridged_edge])
        assert len(result) == 1
        assert result[0].metrics["affinity"].value == 0.8
        # Reverse order: extractor "wins" because it would now be the later one.
        result_reversed = _deduplicate_social([bridged_edge, extractor_edge])
        assert result_reversed[0].metrics["affinity"].value == 0.2

    def test_strictly_newer_metric_still_wins(self):
        """Ordering only matters at ties — a strictly newer
        ``last_updated_fabula`` always wins regardless of position."""
        older = RelationshipEdge(
            source_entity_id="ENT_A",
            target_entity_id="ENT_B",
            metrics={
                "affinity": RelationshipMetric(
                    value=0.2, inertia=0.3,
                    evidence_strength="weak", last_updated_fabula=50,
                    observed=True,
                ),
            },
        )
        newer = RelationshipEdge(
            source_entity_id="ENT_A",
            target_entity_id="ENT_B",
            metrics={
                "affinity": RelationshipMetric(
                    value=0.9, inertia=0.3,
                    evidence_strength="strong", last_updated_fabula=200,
                    observed=True,
                ),
            },
        )
        result = _deduplicate_social([newer, older])
        assert result[0].metrics["affinity"].value == 0.9


# =====================================================================
# EVT_ at_location_id reveal (PR 6 of EventNode.at_location_id)
# =====================================================================


class TestEventLocationReveal:
    def _ws(self):
        from shadow_loom.models import EventNode, Location, WorldStateV1
        return WorldStateV1(
            locations={
                "LOC_HALL": Location(name="Hall", description="d", ambient_state={}),
                "LOC_GARDEN": Location(name="Garden", description="d", ambient_state={}),
            },
            objects={},
            entities={},
            events=[
                EventNode(
                    id="EVT_MEET", fabula_time=10, syuzhet_index=10,
                    event_type="outcome", actor_ids=[],
                    description="meet", at_location_id="LOC_HALL",
                ),
            ],
            causal_topology=[],
            world_traits={},
        )

    def test_bare_loc_value_rewrites_anchor(self):
        ws = self._ws()
        topo = ChunkTopology()
        _augment_topology_with_sandbox_deltas(
            topo, world_state=ws,
            physics_result={"observation_facts": {"EVT_MEET": "LOC_GARDEN"}},
            fabula_time_now=10,
        )
        assert ws.events[0].at_location_id == "LOC_GARDEN"

    def test_at_prefix_value_rewrites_anchor(self):
        ws = self._ws()
        topo = ChunkTopology()
        _augment_topology_with_sandbox_deltas(
            topo, world_state=ws,
            physics_result={"observation_facts": {"EVT_MEET": "at LOC_GARDEN"}},
            fabula_time_now=10,
        )
        assert ws.events[0].at_location_id == "LOC_GARDEN"

    def test_unknown_loc_skipped(self):
        ws = self._ws()
        topo = ChunkTopology()
        _augment_topology_with_sandbox_deltas(
            topo, world_state=ws,
            physics_result={"observation_facts": {"EVT_MEET": "LOC_NONEXISTENT"}},
            fabula_time_now=10,
        )
        assert ws.events[0].at_location_id == "LOC_HALL"

    def test_unknown_event_id_skipped(self):
        ws = self._ws()
        topo = ChunkTopology()
        _augment_topology_with_sandbox_deltas(
            topo, world_state=ws,
            physics_result={"observation_facts": {"EVT_GHOST": "LOC_GARDEN"}},
            fabula_time_now=10,
        )
        # The known event is untouched.
        assert ws.events[0].at_location_id == "LOC_HALL"
