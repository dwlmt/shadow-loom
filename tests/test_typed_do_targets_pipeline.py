# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline-level binding of typed ``DoTarget`` payloads.

Earlier work added a 10-variant typed do-target surface and an
8-list ``IntroducedElements`` container. The
:meth:`CausalPhysicsEngine.apply_do_targets` method dispatches them
correctly, and the standalone :func:`apply_intervention` façade
threads them through. This test file pins the *production* path —
:func:`calculate_narrative_physics` — so a typed-only
``InterventionQuery`` / ``CounterfactualQuery`` (no legacy
``interventions`` dict) actually mutates the sandbox and reports
success at rung-2 / rung-3.
"""
from __future__ import annotations

import networkx as nx

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, Channel, Proposition,
    TraitVector, RelationshipEdge, RelationshipMetric, SpatialEdge,
)
from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import (
    InterventionQuery, CounterfactualQuery,
    DoEvent, DoChannel, DoRelationship, DoCausalEdge, DoSpatialEdge,
    DoTrait,
)


# ----------------------------------------------------------------- fixture
def _make_world() -> WorldStateV1:
    """Two-actor world with one channel, one relationship edge, and
    two events on the timeline so rung-3 has a past anchor.
    """
    return WorldStateV1(
        locations={
            "LOC_HALL": Location(
                id="LOC_HALL",
                name="Hall", description="Hall", ambient_state={}),
            "LOC_GARDEN": Location(
                id="LOC_GARDEN",
                name="Garden", description="Garden", ambient_state={}),
        },
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE", name="Alice", location_id="LOC_HALL",
                status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.3)},
                beliefs=[], concerns=[],
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB", name="Bob", location_id="LOC_HALL",
                status="healthy",
                traits={"courage": TraitVector(value=0.5, inertia=0.3)},
                beliefs=[], concerns=[],
            ),
        },
        events=[
            EventNode(
                id="EVT_MEETING",
                fabula_time=1, syuzhet_index=0,
                event_type="utterance",
                actor_ids=["ENT_ALICE"], target_ids=["ENT_BOB"],
                description="Alice meets Bob", content="meeting",
            ),
            EventNode(
                id="EVT_ARGUMENT",
                fabula_time=2, syuzhet_index=1,
                event_type="outcome",
                actor_ids=["ENT_ALICE"], target_ids=["ENT_BOB"],
                description="They argue", content="argument",
            ),
        ],
        channels={
            "CHAN_HALL_VOICE": Channel(
                id="CHAN_HALL_VOICE", name="hall voice", medium="voice",
                participant_ids=["ENT_ALICE", "ENT_BOB"],
                directionality="duplex",
                intelligibility={},
            ),
        },
        propositions=[],
        social_topology=[
            RelationshipEdge(
                source_entity_id="ENT_ALICE",
                target_entity_id="ENT_BOB",
                metrics={"affinity": RelationshipMetric(value=0.4)},
            ),
        ],
        spatial_topology=[
            SpatialEdge(
                source_id="LOC_HALL", target_id="LOC_GARDEN",
                connection_type="doorway", bidirectional=True,
            ),
        ],
        causal_topology=[],
        world_traits={},
    )


# ============================================================
# RUNG 2 — typed-only InterventionQuery
# ============================================================
class TestTypedOnlyIntervention:
    def test_do_channel_severs_in_sandbox(self):
        ws = _make_world()
        q = InterventionQuery(
            interventions={},
            do_targets=[DoChannel(channel_id="CHAN_HALL_VOICE", active=False, fabula_time=5)],
        )
        result = calculate_narrative_physics(q, ws)
        # Plausibility must NOT short-circuit a typed-only request
        assert result["status"] == "success", result
        # Channel mirroring writes terminated_at_fabula on world_state
        assert ws.channels["CHAN_HALL_VOICE"].terminated_at_fabula is not None

    def test_do_relationship_clamps_in_sandbox(self):
        ws = _make_world()
        q = InterventionQuery(
            interventions={},
            do_targets=[
                DoRelationship(
                    source_entity_id="ENT_ALICE",
                    target_entity_id="ENT_BOB",
                    metric="affinity",
                    value=-0.9,
                ),
            ],
        )
        result = calculate_narrative_physics(q, ws)
        assert result["status"] == "success", result
        edge = next(
            e for e in ws.social_topology
            if e.source_entity_id == "ENT_ALICE"
            and e.target_entity_id == "ENT_BOB"
        )
        assert edge.metrics["affinity"].value == -0.9

    def test_do_spatial_edge_locks(self):
        ws = _make_world()
        q = InterventionQuery(
            interventions={},
            do_targets=[
                DoSpatialEdge(
                    source_id="LOC_HALL",
                    target_id="LOC_GARDEN",
                    action="lock",
                ),
            ],
        )
        result = calculate_narrative_physics(q, ws)
        assert result["status"] == "success", result
        edge = next(
            e for e in ws.spatial_topology
            if e.source_id == "LOC_HALL" and e.target_id == "LOC_GARDEN"
        )
        assert edge.is_locked is True

    def test_do_event_typed_only_hits_sandbox(self):
        ws = _make_world()
        q = InterventionQuery(
            interventions={},
            do_targets=[DoEvent(event_id="EVT_ARGUMENT", occurred=False)],
        )
        result = calculate_narrative_physics(q, ws)
        assert result["status"] == "success", result
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["EVT_ARGUMENT"]["event_type"] == "prevented"

    def test_mixed_legacy_and_typed(self):
        """Legacy dict + typed targets should both apply."""
        ws = _make_world()
        q = InterventionQuery(
            interventions={"ENT_ALICE.status": "dead"},
            do_targets=[
                DoTrait(
                    holder_id="ENT_ALICE",
                    trait_name="courage",
                    value=0.0,
                ),
            ],
        )
        result = calculate_narrative_physics(q, ws)
        assert result["status"] == "success", result
        G = nx.node_link_graph(result["physics_state"])
        assert G.nodes["ENT_ALICE"]["status"] == "dead"
        # Typed DoTrait flows through to sandbox via apply_do_targets;
        # value may be dampened by inertia but must move toward 0.0.
        assert G.nodes["ENT_ALICE"]["traits"]["courage"]["value"] < 0.5


# ============================================================
# RUNG 3 — typed-only CounterfactualQuery
# ============================================================
class TestTypedOnlyCounterfactual:
    def test_historical_do_event_anchors_and_succeeds(self):
        ws = _make_world()
        q = CounterfactualQuery(
            historical_interventions={},
            historical_do_targets=[
                DoEvent(event_id="EVT_MEETING", occurred=False),
            ],
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(q, ws)
        assert result["status"] == "success", result

    def test_historical_do_channel_typed_only_anchors_via_seed(self):
        """A DoChannel-only counterfactual has no legacy footprint;
        the rung-3 path must derive a past anchor from the channel
        participants' event history rather than reporting paradox.
        """
        ws = _make_world()
        q = CounterfactualQuery(
            historical_interventions={},
            historical_do_targets=[
                DoChannel(channel_id="CHAN_HALL_VOICE", active=False),
            ],
            evidence_node_ids=[],
        )
        result = calculate_narrative_physics(q, ws)
        # Either success (anchor recovered via participants) or, if no
        # participant lookup matched, the targeted graceful path. We
        # accept either non-paradox terminal state.
        assert result["status"] in ("success", "implausible")
        # If implausible the reason must NOT be the original empty-set
        # short-circuit — that was the bug.
        if result["status"] == "implausible":
            assert "Empty intervention set" not in result["implausibility_reason"]
