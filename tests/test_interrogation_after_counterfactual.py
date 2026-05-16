# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end regression: counterfactual prune cascade is reflected in
the post-merge ``world_state`` so a subsequent Rung-1 interrogation
cannot contradict the rendered counterfactual prose.

Scenario (the "A Fish Called Wanda" bug): a counterfactual that
prevents ``EVT_KEN_KILLS_DOGS`` should also remove the downstream
``EVT_MRS_COADY_DIES_HEART_ATTACK`` from the merged shadow snapshot.
If it doesn't, an interrogation issued while sitting on the shadow
branch reads the persisted ``world_state.events``, sees Mrs Coady's
death event still present, and answers "yes she died" — directly
contradicting the prose the renderer just produced for the same
branch.

This guard runs the actual merge path (no mocks of the merge handler)
so any future regression in the suppressed-event cascade fails here.
"""

from __future__ import annotations

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    CausalEdge,
    EventNode,
    WorldStateV1,
)


def _evt(eid: str, *, desc: str = "") -> EventNode:
    return EventNode(
        id=eid, fabula_time=0, syuzhet_index=0,
        event_type="outcome", actor_ids=[], target_ids=[],
        description=desc or eid,
        world_id="factual",
    )


def _chain(src: str, tgt: str) -> CausalEdge:
    return CausalEdge(
        source_id=src, target_id=tgt,
        causality_type="chain_reaction",
        mechanism="physical", fabula_time=0,
        world_id="factual",
    )


def _factual_world_with_dog_chain() -> WorldStateV1:
    """Minimal world: KEN_KILLS_DOGS \u2192 MRS_COADY_DIES \u2192 TRIAL_COLLAPSES."""
    return WorldStateV1(
        entities={},
        events=[
            _evt("EVT_KEN_KILLS_DOGS", desc="Ken silences the dogs"),
            _evt("EVT_MRS_COADY_DIES", desc="Mrs Coady dies of heart attack"),
            _evt("EVT_TRIAL_COLLAPSES", desc="Trial collapses"),
        ],
        causal_topology=[
            _chain("EVT_KEN_KILLS_DOGS", "EVT_MRS_COADY_DIES"),
            _chain("EVT_MRS_COADY_DIES", "EVT_TRIAL_COLLAPSES"),
        ],
        spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[], channels={},
    )


def test_shadow_merge_with_suppressed_events_removes_them_from_world_state():
    """After merging a shadow topology that carries ``suppressed_event_ids``,
    the resulting ``vwm.current.events`` no longer contains the closure.

    This is the load-bearing guarantee for the interrogation-on-shadow
    Q&A consistency: ``narrative_physics.calculate_narrative_physics``
    reads ``global_world_state.events`` on the interrogate path, so
    pruning them at merge time is the only way to keep the Q&A LLM
    from re-asserting events the counterfactual prose just denied.
    """
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_dog_chain())

    # Build the topology a shadow counterfactual would produce: no new
    # content, just the suppression set the closure helper computed.
    topology = ChunkTopology(
        chunk_id="CHK_SHADOW_CF",
        suppressed_event_ids=[
            "EVT_KEN_KILLS_DOGS",
            "EVT_MRS_COADY_DIES",
        ],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")

    event_ids = {e.id for e in vwm2.current.events}
    assert "EVT_KEN_KILLS_DOGS" not in event_ids, (
        "Pruned root event must be removed from the shadow snapshot — "
        "otherwise an interrogation on this branch will assert it "
        "happened and contradict the counterfactual prose."
    )
    assert "EVT_MRS_COADY_DIES" not in event_ids, (
        "Pruned descendant event must be removed from the shadow "
        "snapshot — this is the Mrs Coady regression."
    )
    # TRIAL_COLLAPSES wasn't in the suppression set (the topology
    # author left it; the closure helper had no reason to expand to
    # it given only the first two were rooted). Asserting it survives
    # documents that the merge handler does NOT over-prune.
    assert "EVT_TRIAL_COLLAPSES" in event_ids


def test_shadow_merge_cascades_causal_edges_for_suppressed_events():
    """Causal edges touching a suppressed event must also be removed
    so a downstream consumer (Q&A, graph viz) can't reconstruct the
    severed chain by walking edges to surviving event ids."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_dog_chain())
    topology = ChunkTopology(
        chunk_id="CHK_SHADOW_CF",
        suppressed_event_ids=[
            "EVT_KEN_KILLS_DOGS",
            "EVT_MRS_COADY_DIES",
        ],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")
    for edge in vwm2.current.causal_topology:
        assert edge.source_id not in {"EVT_KEN_KILLS_DOGS", "EVT_MRS_COADY_DIES"}
        assert edge.target_id not in {"EVT_KEN_KILLS_DOGS", "EVT_MRS_COADY_DIES"}


def test_shadow_merge_with_empty_suppression_is_noop_on_events():
    """Defensive: an empty suppression set must NOT touch
    ``world_state.events`` even on a shadow merge."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_dog_chain())
    topology = ChunkTopology(
        chunk_id="CHK_SHADOW_EMPTY",
        suppressed_event_ids=[],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_empty")
    event_ids = {e.id for e in vwm2.current.events}
    assert event_ids == {
        "EVT_KEN_KILLS_DOGS",
        "EVT_MRS_COADY_DIES",
        "EVT_TRIAL_COLLAPSES",
    }
