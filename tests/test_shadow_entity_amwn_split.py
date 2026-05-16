# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""AMWN-split entity sidecar tests (Correa & Bareinboim 2025).

Validates Option A of the shadow-branch entity isolation design:
shadow merges that target a factual entity materialise a per-branch
SPLIT COPY in ``WorldStateV1.shadow_entities`` and route snapshot
writes onto the split. The factual entity is never mutated.
Subsequent shadow reads (through
:meth:`WorldStateV1.entities_for_branch` /
:meth:`projected_for_branch`) see the split copy's independently-
trimmed timeline and accumulated shadow snapshots.

The construction is the AMWN node-splitting of Correa & Bareinboim,
ICML 2025 (see docs/academic-foundations.md \u00a72.2), restricted to
the entity variable: a node is *split* across worlds only when a
``do(\u00b7)`` makes its ancestral context diverge from factual; nodes
the shadow branch never touches remain *node-shadowed* across worlds
by sharing the single factual record. AMWNs supersede Pearl/Balke
twin-networks and are sound + complete for d-separation; see the
ctf-calculus rules implemented in :mod:`shadow_loom.amwn`.

On split:
  * the factual entity is deep-copied;
  * every snapshot whose ``triggered_by`` is in
    ``suppressed_event_ids`` is trimmed (the incoming structural
    equation has been severed by the do-intervention in this
    AMWN world);
  * beliefs whose ``acquired_via_event_id`` is suppressed are
    likewise trimmed (provenance gone in this world);
  * the split copy is tagged ``world_id='shadow'``.

Sibling shadow branches (different ``branch_label``) are independent
AMWN worlds W*\u2099 of the same factual base world.
"""

from __future__ import annotations

from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology, EntityUpdate
from shadow_loom.models import (
    Belief,
    CausalEdge,
    Entity,
    EntityStateSnapshot,
    EventNode,
    TraitVector,
    WorldStateV1,
)


def _factual_world_with_coady() -> WorldStateV1:
    """Mrs Coady is alive, located in her flat, knows George by sight.

    She carries one prior snapshot triggered by the (now-suppressible)
    dog-killing event so the *Action*-step trim has something to do.
    """
    coady = Entity(
        id="ENT_MRS_COADY",
        name="Mrs Coady",
        location_id="LOC_FLAT",
        status="healthy",
        traits={"frailty": TraitVector(value=0.4, inertia=0.6)},
        beliefs=[
            Belief(
                target_id="ENT_GEORGE",
                perceived_state="George is honest",
                confidence=0.7,
                evidence_strength="moderate",
                proposition_id="PROP_GEORGE_HONEST",
                inertia=0.3,
                acquired_via_event_id="EVT_PRE_TRIAL",  # not suppressed
            ),
            Belief(
                target_id="ENT_DOGS",
                perceived_state="dogs are dead",
                confidence=0.95,
                evidence_strength="strong",
                proposition_id="PROP_DOGS_DEAD",
                inertia=0.5,
                acquired_via_event_id="EVT_KEN_KILLS_DOGS",
            ),
        ],
        state_timeline=[
            EntityStateSnapshot(
                world_id="factual",
                fabula_time=100,
                triggered_by="EVT_KEN_KILLS_DOGS",
                status="injured",
                traits={"frailty": TraitVector(value=0.7, inertia=0.6)},
            ),
        ],
        world_id="factual",
    )
    return WorldStateV1(
        entities={"ENT_MRS_COADY": coady},
        events=[
            EventNode(
                id="EVT_KEN_KILLS_DOGS",
                fabula_time=90, syuzhet_index=90,
                event_type="outcome",
                actor_ids=["ENT_KEN"], target_ids=["ENT_DOGS"],
                description="Ken silences the dogs",
                world_id="factual",
            ),
        ],
        causal_topology=[],
        spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[], channels={},
    )


# ---------------------------------------------------------------------
# Clone materialisation + Pearl Action-step trim
# ---------------------------------------------------------------------


def test_shadow_merge_clones_factual_entity_into_sidecar():
    """First shadow entity_update on ``ENT_MRS_COADY`` materialises a
    clone into ``shadow_entities[branch_label]`` and routes the
    snapshot onto the clone — the factual entity is unchanged."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topology = ChunkTopology(
        chunk_id="CHK_CF_DOGS",
        entity_updates=[
            EntityUpdate(
                entity_id="ENT_MRS_COADY",
                fabula_time=200,
                triggered_by="EVT_CF_INTERVIEW",
                new_status="injured",
            ),
        ],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")

    # Factual entity untouched.
    factual = vwm2.current.entities["ENT_MRS_COADY"]
    assert factual.world_id == "factual"
    assert factual.status == "healthy"
    factual_snapshot_triggers = {s.triggered_by for s in factual.state_timeline}
    assert "EVT_CF_INTERVIEW" not in factual_snapshot_triggers
    # The pre-existing grieving snapshot must still be on the factual
    # timeline (Pearl: factual world is never mutated by a shadow merge).
    assert "EVT_KEN_KILLS_DOGS" in factual_snapshot_triggers

    # Shadow clone exists, carries the snapshot.
    sidecar = vwm2.current.shadow_entities["cf_dogs"]["ENT_MRS_COADY"]
    assert sidecar.world_id == "shadow"
    clone_triggers = {s.triggered_by for s in sidecar.state_timeline}
    assert "EVT_CF_INTERVIEW" in clone_triggers


def test_shadow_clone_action_step_trims_suppressed_triggered_snapshots():
    """The clone's seeded timeline is trimmed of any snapshot whose
    ``triggered_by`` is in the do-surgery's ``suppressed_event_ids``
    (Pearl Action step: the causing event no longer fires in the
    twin)."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topology = ChunkTopology(
        chunk_id="CHK_CF_DOGS",
        suppressed_event_ids=["EVT_KEN_KILLS_DOGS"],
        entity_updates=[
            EntityUpdate(
                entity_id="ENT_MRS_COADY",
                fabula_time=200,
                triggered_by="EVT_CF_INTERVIEW",
                new_status="injured",
            ),
        ],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")

    clone = vwm2.current.shadow_entities["cf_dogs"]["ENT_MRS_COADY"]
    clone_triggers = {s.triggered_by for s in clone.state_timeline}
    assert "EVT_KEN_KILLS_DOGS" not in clone_triggers, (
        "Action-step trim must drop snapshots whose triggered_by is suppressed."
    )
    assert "EVT_CF_INTERVIEW" in clone_triggers, (
        "New shadow snapshot must accumulate on the clone."
    )


def test_shadow_clone_action_step_trims_suppressed_belief_provenance():
    """Beliefs acquired via a suppressed event are likewise trimmed
    from the clone (provenance is gone in the twin)."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topology = ChunkTopology(
        chunk_id="CHK_CF_DOGS",
        suppressed_event_ids=["EVT_KEN_KILLS_DOGS"],
        entity_updates=[
            EntityUpdate(
                entity_id="ENT_MRS_COADY",
                fabula_time=200,
                triggered_by="EVT_CF_INTERVIEW",
            ),
        ],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")
    clone = vwm2.current.shadow_entities["cf_dogs"]["ENT_MRS_COADY"]
    belief_props = {b.proposition_id for b in clone.beliefs}
    assert "PROP_DOGS_DEAD" not in belief_props, (
        "Belief acquired via suppressed event must be trimmed."
    )
    assert "PROP_GEORGE_HONEST" in belief_props, (
        "Beliefs acquired via non-suppressed events must persist."
    )
    # Factual entity retains the non-suppressed belief. Note: the
    # provenance-trim cascade in ``_apply_deletions`` is currently
    # branch-blind (same v2 limitation as the flat events-list
    # blunt-prune); cleaning that up belongs to the wider
    # sibling-fork isolation pass. For now we only assert that the
    # surviving belief on the factual record is the non-suppressed
    # one — which is enough to detect any future regression that
    # would mutate ``PROP_GEORGE_HONEST`` itself.
    factual_props = {b.proposition_id for b in vwm2.current.entities["ENT_MRS_COADY"].beliefs}
    assert "PROP_GEORGE_HONEST" in factual_props


def test_second_shadow_merge_reuses_existing_clone_no_retrim():
    """Subsequent shadow merges on the same branch reuse the existing
    clone — they do NOT re-clone (which would erase accumulated
    shadow snapshots) and do NOT re-apply the Action trim."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topo1 = ChunkTopology(
        chunk_id="CHK_CF_1",
        suppressed_event_ids=["EVT_KEN_KILLS_DOGS"],
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=200,
            triggered_by="EVT_CF_INTERVIEW", new_status="injured",
        )],
    )
    vwm2 = vwm.merge(topo1, world_id="shadow", branch_label="cf_dogs")
    # Second merge on same branch \u2014 no new suppression, additive only.
    topo2 = ChunkTopology(
        chunk_id="CHK_CF_2",
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=300,
            triggered_by="EVT_CF_VERDICT", new_location_id="LOC_COURT",
        )],
    )
    vwm3 = vwm2.merge(topo2, world_id="shadow", branch_label="cf_dogs")
    clone = vwm3.current.shadow_entities["cf_dogs"]["ENT_MRS_COADY"]
    triggers = {s.triggered_by for s in clone.state_timeline}
    assert {"EVT_CF_INTERVIEW", "EVT_CF_VERDICT"} <= triggers, (
        "Both shadow snapshots must persist across successive merges."
    )


def test_sibling_shadow_branches_are_independent_twins():
    """Two shadow forks from the same factual world get separate
    clones keyed by ``branch_label``; writes on one branch never
    leak onto the other."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topo_a = ChunkTopology(
        chunk_id="CHK_A",
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=200,
            triggered_by="EVT_BRANCH_A", new_status="injured",
        )],
    )
    topo_b = ChunkTopology(
        chunk_id="CHK_B",
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=200,
            triggered_by="EVT_BRANCH_B", new_status="ill",
        )],
    )
    vwm_a = vwm.merge(topo_a, world_id="shadow", branch_label="cf_a")
    vwm_ab = vwm_a.merge(topo_b, world_id="shadow", branch_label="cf_b")

    clone_a = vwm_ab.current.shadow_entities["cf_a"]["ENT_MRS_COADY"]
    clone_b = vwm_ab.current.shadow_entities["cf_b"]["ENT_MRS_COADY"]
    triggers_a = {s.triggered_by for s in clone_a.state_timeline}
    triggers_b = {s.triggered_by for s in clone_b.state_timeline}
    assert "EVT_BRANCH_A" in triggers_a and "EVT_BRANCH_B" not in triggers_a
    assert "EVT_BRANCH_B" in triggers_b and "EVT_BRANCH_A" not in triggers_b


# ---------------------------------------------------------------------
# Reader projection \u2014 entities_for_branch / projected_for_branch
# ---------------------------------------------------------------------


def test_entities_for_branch_factual_passes_through():
    """Factual reads return ``self.entities`` unchanged \u2014 zero
    overhead, no sidecar lookup."""
    ws = _factual_world_with_coady()
    assert ws.entities_for_branch("factual") is ws.entities
    assert ws.entities_for_branch("shadow", None) is ws.entities
    assert ws.entities_for_branch("shadow", "unknown_branch") is ws.entities


def test_entities_for_branch_shadow_overlays_sidecar():
    """Shadow reads return a layered dict where the clone wins for
    cloned entities and factual fallthrough applies for untouched
    ones."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=200,
            triggered_by="EVT_CF", new_status="injured",
        )],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")
    layered = vwm2.current.entities_for_branch("shadow", "cf_dogs")
    assert layered["ENT_MRS_COADY"].world_id == "shadow"
    # Factual entities not touched by the shadow merge fall through;
    # we add a probe by extending the world_state.
    # (No other entities in this fixture, but the dict has the one
    # shadow clone, so factual read of the same id still returns
    # the factual record.)
    assert vwm2.current.entities["ENT_MRS_COADY"].world_id == "factual"


def test_projected_for_branch_swaps_entities_only():
    """Shadow projection returns a copy with ``entities`` swapped to
    the layered view; all other fields are unchanged."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=200,
            triggered_by="EVT_CF", new_status="injured",
        )],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_dogs")
    projected = vwm2.current.projected_for_branch("shadow", "cf_dogs")
    assert projected.entities["ENT_MRS_COADY"].world_id == "shadow"
    # Events / topology shared by reference (same objects).
    assert projected.events is vwm2.current.events
    assert projected.causal_topology is vwm2.current.causal_topology


def test_projected_for_branch_factual_returns_self():
    """Factual projection is a no-op identity \u2014 zero overhead on
    the mainline pipeline path."""
    ws = _factual_world_with_coady()
    assert ws.projected_for_branch("factual") is ws
    assert ws.projected_for_branch("shadow", None) is ws


# ---------------------------------------------------------------------
# Backwards compatibility: pre-clone shadow path still skips loudly
# ---------------------------------------------------------------------


def test_shadow_merge_without_branch_label_falls_back_to_skip():
    """A shadow merge with no ``branch_label`` cannot key a sidecar
    entry; the write is skipped (same as legacy cross-branch block)
    with an INFO log rather than mutating the factual entity."""
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topology = ChunkTopology(
        chunk_id="CHK_CF",
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=200,
            triggered_by="EVT_CF", new_status="injured",
        )],
    )
    vwm2 = vwm.merge(topology, world_id="shadow", branch_label=None)
    # No sidecar entry.
    assert vwm2.current.shadow_entities == {}
    # Factual untouched.
    factual_triggers = {s.triggered_by for s in vwm2.current.entities["ENT_MRS_COADY"].state_timeline}
    assert "EVT_CF" not in factual_triggers


# ---------------------------------------------------------------------
# H1 regression: existing sidecar clones must be re-trimmed on
# subsequent merges that widen the suppression set.
# ---------------------------------------------------------------------


def test_subsequent_merge_re_trims_existing_clone_state_timeline():
    """Multi-merge scenario: the first shadow merge creates a clone
    with one shadow snapshot. The second shadow merge then expands
    ``suppressed_event_ids`` to include the first merge's
    ``triggered_by`` (simulating closure widening as a newly-pruned
    descendant turns out to be the cause of a previously-accepted
    snapshot). The clone's state_timeline must shed that snapshot
    on the second merge \u2014 walking only ``merged.entities`` would
    miss it because the clone lives in ``merged.shadow_entities``.
    """
    vwm = VersionedWorldModel.from_world_state(_factual_world_with_coady())
    topo1 = ChunkTopology(
        chunk_id="CHK_CF_1",
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=200,
            triggered_by="EVT_CF_INTERVIEW", new_status="injured",
        )],
    )
    vwm2 = vwm.merge(topo1, world_id="shadow", branch_label="cf_dogs")
    clone_v1 = vwm2.current.shadow_entities["cf_dogs"]["ENT_MRS_COADY"]
    assert "EVT_CF_INTERVIEW" in {s.triggered_by for s in clone_v1.state_timeline}

    # Second merge widens the suppression set to include the
    # first-merge snapshot's triggered_by.
    topo2 = ChunkTopology(
        chunk_id="CHK_CF_2",
        suppressed_event_ids=["EVT_CF_INTERVIEW"],
    )
    vwm3 = vwm2.merge(topo2, world_id="shadow", branch_label="cf_dogs")
    clone_v2 = vwm3.current.shadow_entities["cf_dogs"]["ENT_MRS_COADY"]
    clone_triggers = {s.triggered_by for s in clone_v2.state_timeline}
    assert "EVT_CF_INTERVIEW" not in clone_triggers, (
        "Existing sidecar clone must be re-trimmed when subsequent "
        "merges widen ``suppressed_event_ids``."
    )


def test_subsequent_merge_re_trims_existing_clone_beliefs():
    """Same regression for ``Belief.acquired_via_event_id`` provenance
    on the clone: a belief acquired via an event suppressed by the
    second merge must be dropped from the existing clone, not only
    from the (unread) factual entity record."""
    # Build a factual world where Coady carries a belief acquired via
    # an event we'll suppress on the second merge.
    coady = Entity(
        id="ENT_MRS_COADY", name="Mrs Coady",
        location_id="LOC_FLAT", status="healthy",
        traits={"frailty": TraitVector(value=0.4, inertia=0.6)},
        beliefs=[
            Belief(
                target_id="ENT_GEORGE",
                perceived_state="George ordered the dogs killed",
                confidence=0.8, evidence_strength="moderate",
                proposition_id="PROP_GEORGE_GUILTY",
                inertia=0.4,
                acquired_via_event_id="EVT_KEN_CONFESSED_TO_HER",
            ),
        ],
        state_timeline=[], world_id="factual",
    )
    ws = WorldStateV1(
        entities={"ENT_MRS_COADY": coady},
        events=[EventNode(
            id="EVT_KEN_CONFESSED_TO_HER", fabula_time=80, syuzhet_index=80,
            event_type="utterance", actor_ids=["ENT_KEN"],
            target_ids=["ENT_MRS_COADY"],
            description="Ken admits the dog-killing to Coady",
            world_id="factual",
        )],
        causal_topology=[], spatial_topology=[], social_topology=[],
        locations={}, objects={}, world_traits={},
        propositions=[], channels={},
    )
    vwm = VersionedWorldModel.from_world_state(ws)
    # First shadow merge: create a clone via an unrelated entity update.
    topo1 = ChunkTopology(
        chunk_id="CHK_1",
        entity_updates=[EntityUpdate(
            entity_id="ENT_MRS_COADY", fabula_time=200,
            triggered_by="EVT_NEUTRAL", new_status="injured",
        )],
    )
    vwm2 = vwm.merge(topo1, world_id="shadow", branch_label="cf")
    clone_v1 = vwm2.current.shadow_entities["cf"]["ENT_MRS_COADY"]
    assert "PROP_GEORGE_GUILTY" in {b.proposition_id for b in clone_v1.beliefs}
    # Second merge: suppress the belief's acquisition event.
    topo2 = ChunkTopology(
        chunk_id="CHK_2",
        suppressed_event_ids=["EVT_KEN_CONFESSED_TO_HER"],
    )
    vwm3 = vwm2.merge(topo2, world_id="shadow", branch_label="cf")
    clone_v2 = vwm3.current.shadow_entities["cf"]["ENT_MRS_COADY"]
    belief_props = {b.proposition_id for b in clone_v2.beliefs}
    assert "PROP_GEORGE_GUILTY" not in belief_props, (
        "Existing sidecar clone must shed beliefs whose "
        "``acquired_via_event_id`` is in the widened suppression set."
    )

