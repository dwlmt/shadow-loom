# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Phase A3 Proposition Catalogue + Phase C reconciler.

These tests deliberately bypass the LLM agent — we hand-build a
``PropositionCatalogue`` and a ``ChunkTopology`` to exercise the
reconciler's six-step fold without paying token cost or requiring
a model endpoint.
"""

from __future__ import annotations

from shadow_loom.ingestion import (
    ChunkConcernSnapshot,
    ChunkPropositionSnapshot,
    ChunkTopology,
    ConcernSeed,
    PropositionCatalogue,
    PropositionTruthCommit,
    reconcile_affect,
)
from shadow_loom.models import (
    Concern,
    Entity,
    EventNode,
    Location,
    Proposition,
    PropositionSnapshot,
    WorldStateV1,
)


def _world() -> WorldStateV1:
    duncan = Entity(
        id="ENT_DUNCAN", name="Duncan", location_id="LOC_CASTLE",
        status="healthy", traits={},
    )
    macbeth = Entity(
        id="ENT_MACBETH", name="Macbeth", location_id="LOC_CASTLE",
        status="healthy", traits={},
    )
    evt_murder = EventNode(
        id="EVT_MURDER", fabula_time=100, syuzhet_index=0,
        event_type="outcome", actor_ids=["ENT_MACBETH"],
        target_ids=["ENT_DUNCAN"],
        description="Macbeth murders Duncan",
        resolves_proposition_ids=["PROP_DUNCAN_DEAD"],
    )
    return WorldStateV1(
        locations={"LOC_CASTLE": Location(
            id="LOC_CASTLE", name="Castle", description="x")},
        objects={},
        entities={"ENT_DUNCAN": duncan, "ENT_MACBETH": macbeth},
        events=[evt_murder],
        causal_topology=[],
    )


def test_reconcile_lands_catalogue_propositions_and_seeds():
    ws = _world()
    catalogue = PropositionCatalogue(
        propositions=[Proposition(
            proposition_id="PROP_DUNCAN_DEAD", kind="event_occurs",
            referent_ids=["ENT_DUNCAN"], description="Duncan is dead",
            stakes=0.9, audience_default_prior=0.1,
        )],
        concern_seeds=[ConcernSeed(
            concern_id="CCN_MACBETH_DESIRES_CROWN",
            entity_id="ENT_MACBETH",
            proposition_id="PROP_DUNCAN_DEAD",
            polarity="desire",
            baseline_salience=0.85,
        )],
    )

    out = reconcile_affect(ws, catalogue, [])

    assert [p.proposition_id for p in out.propositions] == ["PROP_DUNCAN_DEAD"]
    p = out.propositions[0]
    assert p.stakes == 0.9
    assert p.audience_default_prior == 0.1
    # Inline EVT_MURDER.resolves_proposition_ids commits truth.
    assert p.truth_at_fabula == {100: True}
    macbeth = out.entities["ENT_MACBETH"]
    assert len(macbeth.concerns) == 1
    assert macbeth.concerns[0].concern_id == "CCN_MACBETH_DESIRES_CROWN"
    assert macbeth.concerns[0].polarity == "desire"


def test_reconcile_per_chunk_snapshots_attach_to_timelines():
    ws = _world()
    catalogue = PropositionCatalogue(
        propositions=[Proposition(
            proposition_id="PROP_DUNCAN_DEAD", kind="event_occurs",
            referent_ids=["ENT_DUNCAN"], description="Duncan is dead",
        )],
        concern_seeds=[ConcernSeed(
            concern_id="CCN_DUNCAN_FEARS_DEATH",
            entity_id="ENT_DUNCAN",
            proposition_id="PROP_DUNCAN_DEAD",
            polarity="fear",
            baseline_salience=0.5,
        )],
    )
    topo = ChunkTopology(
        proposition_snapshots=[ChunkPropositionSnapshot(
            proposition_id="PROP_DUNCAN_DEAD", fabula_time=100,
            triggered_by="EVT_MURDER", stakes=1.0,
        )],
        concern_snapshots=[ChunkConcernSnapshot(
            concern_id="CCN_DUNCAN_FEARS_DEATH", fabula_time=100,
            triggered_by="EVT_MURDER", salience=1.0, polarity="fear",
        )],
    )

    out = reconcile_affect(ws, catalogue, [topo])

    p = out.propositions[0]
    assert len(p.state_timeline) == 1
    assert p.state_timeline[0].stakes == 1.0
    assert p.state_timeline[0].triggered_by == "EVT_MURDER"

    duncan = out.entities["ENT_DUNCAN"]
    assert len(duncan.concerns) == 1
    ccn = duncan.concerns[0]
    assert len(ccn.state_timeline) == 1
    assert ccn.state_timeline[0].salience == 1.0


def test_reconcile_truth_commit_overrides_inline_event():
    """Per-chunk affect agent's explicit truth_commit beats inline default."""
    ws = _world()
    catalogue = PropositionCatalogue(
        propositions=[Proposition(
            proposition_id="PROP_DUNCAN_DEAD", kind="event_occurs",
            referent_ids=["ENT_DUNCAN"], description="Duncan is dead",
        )],
    )
    topo = ChunkTopology(
        proposition_truth_commits=[PropositionTruthCommit(
            proposition_id="PROP_DUNCAN_DEAD", fabula_time=100,
            triggered_by="EVT_MURDER", truth=False,
        )],
    )

    out = reconcile_affect(ws, catalogue, [topo])

    # Affect agent's explicit False overrides EVT_MURDER's inline True.
    assert out.propositions[0].truth_at_fabula[100] is False


def test_reconcile_seed_dedup_idempotent_on_re_run():
    ws = _world()
    catalogue = PropositionCatalogue(
        propositions=[Proposition(
            proposition_id="PROP_DUNCAN_DEAD", kind="event_occurs",
            referent_ids=["ENT_DUNCAN"], description="Duncan is dead",
        )],
        concern_seeds=[ConcernSeed(
            concern_id="CCN_DUNCAN_FEARS_DEATH",
            entity_id="ENT_DUNCAN",
            proposition_id="PROP_DUNCAN_DEAD",
            polarity="fear",
            baseline_salience=0.5,
        )],
    )

    out = reconcile_affect(ws, catalogue, [])
    assert len(out.entities["ENT_DUNCAN"].concerns) == 1
    out2 = reconcile_affect(out, catalogue, [])
    # No duplicate concern despite re-applying the catalogue.
    assert len(out2.entities["ENT_DUNCAN"].concerns) == 1


def test_reconcile_drops_snapshot_for_unknown_proposition():
    ws = _world()
    catalogue = PropositionCatalogue(
        propositions=[Proposition(
            proposition_id="PROP_DUNCAN_DEAD", kind="event_occurs",
            referent_ids=["ENT_DUNCAN"], description="Duncan is dead",
        )],
    )
    topo = ChunkTopology(
        proposition_snapshots=[ChunkPropositionSnapshot(
            proposition_id="PROP_GHOST", fabula_time=100,
            triggered_by="EVT_MURDER", stakes=1.0,
        )],
    )
    out = reconcile_affect(ws, catalogue, [topo])
    # No timeline added, no exceptions.
    assert out.propositions[0].state_timeline == []


def test_reconcile_no_op_for_empty_catalogue_and_topologies():
    ws = _world()
    out = reconcile_affect(ws, None, [])
    assert out.propositions == []
    assert all(not e.concerns for e in out.entities.values())
