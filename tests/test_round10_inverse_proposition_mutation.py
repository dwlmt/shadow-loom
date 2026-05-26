# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round-10 audit — inverse-proposition mutation persistence.

* **R10-F1** (major) — ``_apply_do_proposition``'s round-8 inverse
  mirror block wrote the flipped truth onto
  ``world_state.propositions[idx]`` but did NOT append a paired
  :class:`PropositionMutation` row. The pipeline merge bridge at
  ``shadow_loom/pipeline.py`` only iterates ``proposition_mutations``
  to emit ``PropositionTruthCommit`` rows for the merge fold, so the
  inverse clamp lived only on the deep-cloned shadow world (via
  ``_isolate_ws_for_surgery``) and was discarded after physics. Every
  Do-surgery on a proposition with a declared
  ``inverse_proposition_id`` therefore silently desynced the inverse
  on the canonical timeline — Phase C ingestion mirrors both sides on
  the canonical write path, so the asymmetry was specific to the
  Pearl Rung-2 surgery surface.
"""
from __future__ import annotations

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import (
    Entity,
    EventNode,
    Location,
    Proposition,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.query_models import DoProposition


def _make_world_with_inverse_pair() -> WorldStateV1:
    return WorldStateV1(
        locations={
            "LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={}),
        },
        objects={},
        entities={
            "ENT_DUNCAN": Entity(
                id="ENT_DUNCAN", name="Duncan", location_id="LOC_A",
                status="healthy",
                traits={"vitality": TraitVector(value=0.7, inertia=0.3)},
                beliefs=[], concerns=[],
            ),
        },
        events=[
            EventNode(
                id="EVT_ANCHOR", fabula_time=10, syuzhet_index=10,
                event_type="outcome", actor_ids=["ENT_DUNCAN"], target_ids=[],
                description="Anchor.",
            ),
        ],
        causal_topology=[], spatial_topology=[], social_topology=[],
        channels={},
        propositions=[
            Proposition(
                world_id="factual",
                proposition_id="PROP_DUNCAN_ALIVE",
                kind="trait_holds",
                referent_ids=["ENT_DUNCAN"],
                description="Duncan is alive",
                audience_default_prior=0.9,
                stakes=0.9,
                truth_at_fabula={5: True},
                inverse_proposition_id="PROP_DUNCAN_DEAD",
            ),
            Proposition(
                world_id="factual",
                proposition_id="PROP_DUNCAN_DEAD",
                kind="trait_holds",
                referent_ids=["ENT_DUNCAN"],
                description="Duncan is dead",
                audience_default_prior=0.1,
                stakes=0.9,
                truth_at_fabula={5: False},
                inverse_proposition_id="PROP_DUNCAN_ALIVE",
            ),
        ],
        world_traits={},
    )


def _engine(ws: WorldStateV1) -> CausalPhysicsEngine:
    ego = extract_ego_graph_from_memory(ws, ["ENT_DUNCAN"])
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    return CausalPhysicsEngine(sandbox, ws)


class TestInversePropositionMutationEmitted:
    def test_inverse_mirror_emits_paired_mutation_row(self):
        """Flipping ``PROP_DUNCAN_ALIVE → False`` must also emit a
        ``PropositionMutation`` for ``PROP_DUNCAN_DEAD → True`` so the
        pipeline merge bridge promotes both sides of the pair onto the
        canonical timeline.
        """
        ws = _make_world_with_inverse_pair()
        eng = _engine(ws)
        eng.apply_do_targets([DoProposition(
            proposition_id="PROP_DUNCAN_ALIVE",
            truth=False,
            fabula_time=20,
            propagate_to_beliefs=False,
        )])

        # Primary mutation row must be present.
        primary = [
            m for m in eng._proposition_mutations
            if m.proposition_id == "PROP_DUNCAN_ALIVE"
        ]
        assert len(primary) == 1
        assert primary[0].new_truth is False
        assert primary[0].old_truth is True
        assert primary[0].fabula_time == 20

        # Inverse mutation row — the round-10 fix. Without this row the
        # pipeline never emits a PropositionTruthCommit for the inverse
        # and the merge silently desyncs the pair on the canonical world.
        inverse = [
            m for m in eng._proposition_mutations
            if m.proposition_id == "PROP_DUNCAN_DEAD"
        ]
        assert len(inverse) == 1, (
            "DoProposition on a paired proposition must emit a paired "
            "PropositionMutation for the inverse so the merge bridge "
            "promotes both halves of the inverse pair."
        )
        assert inverse[0].new_truth is True
        assert inverse[0].old_truth is False
        assert inverse[0].fabula_time == 20

    def test_inverse_mirror_on_world_state_still_intact(self):
        """The round-8 in-place mirror onto ``world_state.propositions``
        must still happen (round-10 only ADDS the mutation row; the
        local mirror is what the directive assembler / brief renderer
        reads off the engine-isolated world)."""
        ws = _make_world_with_inverse_pair()
        eng = _engine(ws)
        eng.apply_do_targets([DoProposition(
            proposition_id="PROP_DUNCAN_ALIVE",
            truth=False,
            fabula_time=20,
            propagate_to_beliefs=False,
        )])
        alive = next(p for p in ws.propositions if p.proposition_id == "PROP_DUNCAN_ALIVE")
        dead = next(p for p in ws.propositions if p.proposition_id == "PROP_DUNCAN_DEAD")
        assert alive.truth_at_fabula.get(20) is False
        assert dead.truth_at_fabula.get(20) is True

    def test_no_inverse_no_paired_mutation(self):
        """When the targeted proposition has no ``inverse_proposition_id``,
        only the single primary mutation row is emitted."""
        ws = _make_world_with_inverse_pair()
        # Drop the back-reference on PROP_DUNCAN_ALIVE so it no longer
        # declares an inverse.
        ws.propositions = [
            p.model_copy(update={"inverse_proposition_id": None})
            if p.proposition_id == "PROP_DUNCAN_ALIVE" else p
            for p in ws.propositions
        ]
        eng = _engine(ws)
        eng.apply_do_targets([DoProposition(
            proposition_id="PROP_DUNCAN_ALIVE",
            truth=False,
            fabula_time=20,
            propagate_to_beliefs=False,
        )])
        assert len(eng._proposition_mutations) == 1
        assert eng._proposition_mutations[0].proposition_id == "PROP_DUNCAN_ALIVE"

    def test_inverse_conflict_does_not_emit_paired_row(self):
        """If the inverse already carries a contradictory truth at the
        same fabula tick the round-8 code keeps the existing value (and
        warns); no mirror is applied, so no paired mutation row should
        be emitted either."""
        ws = _make_world_with_inverse_pair()
        # Pre-stamp PROP_DUNCAN_DEAD with the WRONG truth at the target
        # fabula time so the inverse-conflict branch fires.
        ws.propositions = [
            p.model_copy(
                update={"truth_at_fabula": {**p.truth_at_fabula, 20: False}}
            ) if p.proposition_id == "PROP_DUNCAN_DEAD" else p
            for p in ws.propositions
        ]
        eng = _engine(ws)
        eng.apply_do_targets([DoProposition(
            proposition_id="PROP_DUNCAN_ALIVE",
            truth=False,
            fabula_time=20,
            propagate_to_beliefs=False,
        )])
        # Primary still records, inverse must NOT — the mirror was
        # rejected by the conflict guard.
        ids = [m.proposition_id for m in eng._proposition_mutations]
        assert "PROP_DUNCAN_ALIVE" in ids
        assert "PROP_DUNCAN_DEAD" not in ids
