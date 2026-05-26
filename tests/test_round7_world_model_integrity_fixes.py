# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for three world-model integrity defects surfaced by
the round-7 audit (2026-05-26) AFTER the initial trait-clamp fixes:

1. ``shadow_loom.projections.reconstruct_world_trait_at_causal``
   silently ignored an authored ``WorldTraitSnapshot`` whenever its
   magnitude happened to equal the canonical baseline value, leaving
   the causal-accumulated drift in place. Mirror-bug of the entity-side
   silent-overwrite defect already fixed in the same audit.

2. ``shadow_loom.models.reconstruct_entity_at`` permitted a proposition-
   scoped ``BeliefConfidenceShift`` to land on a same-target belief
   whose ``proposition_id`` was ``None``. When an entity held two
   beliefs about the same target (one bare, one joined to a
   ``Proposition``), an Affect-side shift addressed to the PROP-joined
   belief silently moved confidence on the bare belief instead.

3. ``shadow_loom.extract_graph._merge_belief_confidence_updates`` (the
   merge-time twin of #2) had the same permissive proposition-id
   matching rule.
"""

from __future__ import annotations

from shadow_loom.models import (
    Belief,
    BeliefConfidenceShift,
    Entity,
    EntityStateSnapshot,
    GlobalTrait,
    Location,
    Proposition,
    TraitVector,
    WorldStateV1,
    WorldTraitSnapshot,
    reconstruct_entity_at,
)
from shadow_loom.projections import reconstruct_world_trait_at_causal


# =====================================================================
# Bug 1 — world-trait causal-replay stomp
# =====================================================================


class TestWorldTraitProjectionSnapshotAuthority:
    """``reconstruct_world_trait_at_causal`` must respect any authored
    snapshot, even one that intentionally resets the magnitude back to
    the baseline value. The discriminator must be 'did a snapshot
    actually write?' not 'does the result differ from baseline?'.
    """

    def _ws_with_world_trait_and_mutation(self) -> WorldStateV1:
        from shadow_loom.models import CausalEdge, EventNode
        return WorldStateV1(
            locations={
                "LOC_X": Location(
                    id="LOC_X", name="X", description="d", ambient_state={},
                ),
            },
            objects={},
            entities={},
            events=[
                EventNode(
                    id="EVT_SPIKE", fabula_time=3, syuzhet_index=3,
                    event_type="outcome", actor_ids=[],
                    description="spike", at_location_id="LOC_X",
                ),
            ],
            causal_topology=[
                CausalEdge(
                    source_id="EVT_SPIKE",
                    target_id="WORLD_DREAD",
                    causality_type="mutation",
                    fabula_time=3,
                    trait_target="intensity",
                    trait_delta=0.3,
                    causal_force=1.0,
                    evidence_strength="moderate",
                    mechanism="psychological",
                ),
            ],
            world_traits={
                "WORLD_DREAD": GlobalTrait(
                    id="WORLD_DREAD",
                    name="Ambient Dread",
                    description="d",
                    category="social_structure",
                    magnitude=TraitVector(value=0.5, inertia=0.3),
                    affected_domains=["psychological"],
                ),
            },
        )

    def test_snapshot_resetting_to_baseline_value_overrides_causal_drift(self):
        ws = self._ws_with_world_trait_and_mutation()
        # Snapshot at t=5 resets magnitude back to the baseline 0.5.
        # Pre-fix: the equality-vs-baseline guard treated this as a
        # no-op snapshot and left causal-drifted 0.8 in place.
        ws.world_traits["WORLD_DREAD"].state_timeline.append(
            WorldTraitSnapshot(
                fabula_time=5,
                triggered_by=None,
                magnitude=TraitVector(value=0.5, inertia=0.3),
            )
        )
        result = reconstruct_world_trait_at_causal(ws, "WORLD_DREAD", fabula_time=6)
        assert result["magnitude"]["value"] == 0.5

    def test_no_snapshot_keeps_causal_accumulation(self):
        ws = self._ws_with_world_trait_and_mutation()
        result = reconstruct_world_trait_at_causal(ws, "WORLD_DREAD", fabula_time=6)
        # 0.5 baseline + 0.3 mutation = 0.8
        assert result["magnitude"]["value"] == 0.8

    def test_snapshot_with_distinct_value_still_wins(self):
        ws = self._ws_with_world_trait_and_mutation()
        ws.world_traits["WORLD_DREAD"].state_timeline.append(
            WorldTraitSnapshot(
                fabula_time=5,
                triggered_by=None,
                magnitude=TraitVector(value=0.2, inertia=0.3),
            )
        )
        result = reconstruct_world_trait_at_causal(ws, "WORLD_DREAD", fabula_time=6)
        assert result["magnitude"]["value"] == 0.2


# =====================================================================
# Bug 2 + 3 — proposition-scoped belief-shift discriminator
# =====================================================================


def _entity_with_two_beliefs_about_same_target() -> Entity:
    """Holder has TWO beliefs about ENT_TARGET:
       * one bare (proposition_id=None) at confidence 0.4
       * one joined to PROP_X at confidence 0.6
    A proposition-scoped shift addressed to PROP_X must NOT land on
    the bare belief.
    """
    return Entity(
        id="ENT_HOLDER", name="Holder", location_id="LOC_X", status="healthy",
        traits={},
        beliefs=[
            Belief(
                target_id="ENT_TARGET",
                perceived_state="alive",
                confidence=0.4,
                inertia=0.3,
                established_at_fabula=0,
                proposition_id=None,
            ),
            Belief(
                target_id="ENT_TARGET",
                perceived_state="loyal",
                confidence=0.6,
                inertia=0.3,
                established_at_fabula=0,
                proposition_id="PROP_X",
            ),
        ],
    )


class TestBeliefShiftPropositionDiscriminator:
    """``reconstruct_entity_at`` must apply a proposition-scoped
    ``BeliefConfidenceShift`` only to the belief joined to that
    proposition, never to a same-target bare belief.
    """

    def test_proposition_scoped_shift_targets_propositioned_belief_only(self):
        ent = _entity_with_two_beliefs_about_same_target()
        ent.state_timeline.append(
            EntityStateSnapshot(
                fabula_time=10,
                triggered_by=None,
                belief_confidence_updates=[
                    BeliefConfidenceShift(
                        target_id="ENT_TARGET",
                        proposition_id="PROP_X",
                        new_confidence=0.95,
                    ),
                ],
            )
        )
        snap = reconstruct_entity_at(ent, fabula_time=20)
        by_prop = {b.get("proposition_id"): b["confidence"] for b in snap["beliefs"]}
        # PROP_X belief got the shift; bare belief untouched.
        assert by_prop[None] == 0.4
        assert by_prop["PROP_X"] == 0.95

    def test_bare_shift_still_matches_bare_belief(self):
        """Backwards-compat: a shift with ``proposition_id=None`` must
        still match the bare belief (legacy/wildcard behaviour)."""
        ent = _entity_with_two_beliefs_about_same_target()
        ent.state_timeline.append(
            EntityStateSnapshot(
                fabula_time=10,
                triggered_by=None,
                belief_confidence_updates=[
                    BeliefConfidenceShift(
                        target_id="ENT_TARGET",
                        proposition_id=None,
                        new_confidence=0.85,
                    ),
                ],
            )
        )
        snap = reconstruct_entity_at(ent, fabula_time=20)
        by_prop = {b.get("proposition_id"): b["confidence"] for b in snap["beliefs"]}
        # Bare shift lands on the first matching belief (the bare one,
        # since it's listed first); PROP_X belief is untouched.
        assert by_prop[None] == 0.85
        assert by_prop["PROP_X"] == 0.6


class TestMergeBeliefShiftPropositionDiscriminator:
    """``shadow_loom.extract_graph._apply_belief_confidence_updates``
    must apply the same strict proposition-id rule at merge time.
    """

    def test_proposition_scoped_merge_update_targets_propositioned_belief(self):
        from shadow_loom.extract_graph import (
            _apply_belief_confidence_updates,
            MergeChangeset,
        )
        from shadow_loom.ingestion import (
            BeliefConfidenceUpdate,
            ChunkTopology,
            EntityUpdate,
        )

        ent = _entity_with_two_beliefs_about_same_target()
        ws = WorldStateV1(
            locations={
                "LOC_X": Location(
                    id="LOC_X", name="X", description="d", ambient_state={},
                ),
            },
            objects={},
            entities={"ENT_HOLDER": ent},
            events=[],
            causal_topology=[],
            world_traits={},
        )
        topo = ChunkTopology()
        topo.entity_updates.append(EntityUpdate(
            entity_id="ENT_HOLDER",
            fabula_time=10,
            triggered_by=None,
            belief_confidence_updates=[
                BeliefConfidenceUpdate(
                    target_id="ENT_TARGET",
                    proposition_id="PROP_X",
                    new_confidence=0.95,
                ),
            ],
        ))

        _apply_belief_confidence_updates(
            ws, topo, changeset=MergeChangeset(), merge_world_id="factual",
        )

        by_prop = {b.proposition_id: b.confidence for b in ws.entities["ENT_HOLDER"].beliefs}
        assert by_prop[None] == 0.4
        assert by_prop["PROP_X"] == 0.95
