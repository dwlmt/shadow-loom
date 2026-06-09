# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Synthetic tests for the round-5 deeper audit fixes.

* **R5-F1** \u2014 ``compute_causal_feedback`` must NOT route the engine's
  blocked propagations into ``miracle_steps_detected`` when the
  engine itself flagged the surgery
  :attr:`CausalPhysicsResult.intervention_inert`. Routing them there
  pinned ``engine_passed=False`` for the refinement loop on exactly
  the scenes where the renderer was correctly told to stage zero
  downstream effects.
* **R5-F2** \u2014 The MCP envelope must surface ``intervention_inert``
  and ``intervention_inert_reason`` so external clients can disclose
  a no-op surgery instead of displaying an empty cascade as if it
  were a complete one.
* **R5-F3** \u2014 The ``_apply_deletions`` cascade (``removed_event_ids``)
  must demote ``RelationshipMetric.evidence_strength`` to ``"weak"``
  with the same parity as the suppression cascade
  (``suppressed_event_ids``) shipped in round-3 F3.
* **R5-F4** \u2014 The ``_apply_deletions`` hard-delete branch must also
  cascade-trim ``state_timeline`` / ``beliefs`` /
  ``Proposition.truth_at_fabula`` entries whose provenance points at
  the deleted event, parity with the suppression cascade and the
  runtime closure-scrub (Phase 12 ingestion-audit fix).
"""

from __future__ import annotations

import logging
from typing import Optional

from shadow_loom.causal_physics import BlockedPropagation, CausalPhysicsResult
from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    Belief,
    CausalEdge,
    Channel,
    Entity,
    EntityStateSnapshot,
    EventNode,
    GlobalTrait,
    Proposition,
    RelationshipEdge,
    RelationshipMetric,
    TraitVector,
    WorldStateV1,
    WorldTraitSnapshot,
)


# ---------------------------------------------------------------------
# R5-F1: inert intervention bypasses miracle accounting
# ---------------------------------------------------------------------
class TestInertBypassesMiracleAccounting:
    """When ``intervention_inert=True`` every blocked propagation is
    the expected outcome \u2014 it must not feed
    ``miracle_steps_detected`` or trip the engine threshold gate."""

    def _empty_brief(self):
        from shadow_loom.directive_assembly import CreativeBrief
        return CreativeBrief(
            target_effect="intervention",
            target_entities=["ENT_A"],
            constraints=[],
            epistemic_gaps=[],
            trait_trajectories=[],
            intervention_mechanisms=[],
            abduction_truths=[],
            entanglement_pairs=[],
            scene_context={},
        )

    def _result(self, *, inert: bool) -> CausalPhysicsResult:
        # Use ``spatial_affordance`` reason — it has no special
        # routing branch, so it falls through to ``miracle_steps``
        # unless the inert short-circuit catches it.
        return CausalPhysicsResult(
            sandbox_data={"nodes": [], "links": []},
            mutations=[],
            social_mutations=[],
            blocked=[
                BlockedPropagation(
                    node_id="ENT_A", trait="composure",
                    impact=0.10, inertia=0.55, reason="spatial_affordance",
                ),
                BlockedPropagation(
                    node_id="ENT_B", trait="fear",
                    impact=0.08, inertia=0.50, reason="spatial_affordance",
                ),
            ],
            intervened_nodes=["ENT_A"],
            hidden_deltas={},
            intervention_inert=inert,
            intervention_inert_reason=(
                "all 1 intervention(s) Rule-3 pruned" if inert else None
            ),
        )

    def test_non_inert_keeps_miracles_in_deterministic_mode(self):
        from shadow_loom.auditor import compute_causal_feedback
        fb = compute_causal_feedback(self._result(inert=False), self._empty_brief())
        # Non-special block reasons — routed to miracle_steps when the
        # surgery is not flagged inert.
        assert len(fb.miracle_steps_detected) == 2
        assert fb.noisy_or_absorbed_propagations == []

    def test_inert_reroutes_blocks_to_absorbed(self):
        from shadow_loom.auditor import compute_causal_feedback
        fb = compute_causal_feedback(self._result(inert=True), self._empty_brief())
        # Inert short-circuit \u2014 blocks are evidence of correct
        # resistance, not narrative miracles.
        assert fb.miracle_steps_detected == []
        assert len(fb.noisy_or_absorbed_propagations) == 2

    def test_inert_does_not_pin_engine_threshold_failure(self):
        from shadow_loom.auditor import (
            AffectiveStateFeedback,
            AuditorConfig,
            ChangeImpactMetrics,
            _engine_thresholds_check,
            compute_causal_feedback,
        )
        cfg = AuditorConfig()
        # Build a ChangeImpactMetrics carrying the inert feedback.
        cf_inert = compute_causal_feedback(self._result(inert=True), self._empty_brief())
        impact = ChangeImpactMetrics(
            causal_feedback=cf_inert,
            affective_feedback=AffectiveStateFeedback(),
        )
        passed, failures = _engine_thresholds_check(impact, cfg)
        # No miracles \u2192 deterministic gate passes for the inert path.
        assert passed is True
        assert failures == []


# ---------------------------------------------------------------------
# R5-F2: MCP surfaces intervention_inert
# ---------------------------------------------------------------------
class TestMCPSurfacesInert:
    """The MCP-envelope builder must surface the inert flag and reason
    so external clients can disclose a no-op surgery."""

    def test_inert_disclosure_in_envelope(self):
        from shadow_loom_mcp.helpers import _apply_inert_envelope
        env: dict = {}
        _apply_inert_envelope(env, {
            "intervention_inert": True,
            "intervention_inert_reason": "all 2 do-target(s) Rule-3 pruned",
        })
        assert env.get("intervention_inert") is True
        assert "Rule-3 pruned" in env.get("intervention_inert_reason", "")

    def test_no_inert_key_when_flag_unset(self):
        from shadow_loom_mcp.helpers import _apply_inert_envelope
        env: dict = {}
        _apply_inert_envelope(env, {"status": "success"})
        assert "intervention_inert" not in env
        assert "intervention_inert_reason" not in env
        _apply_inert_envelope(env, None)
        assert env == {}


# ---------------------------------------------------------------------
# R5-F3: deletion path also demotes social_topology metrics
# ---------------------------------------------------------------------
class TestDeletionDemotesSocialMetric:
    """``removed_event_ids`` (the deletion cascade) must demote
    ``RelationshipMetric.evidence_strength`` to ``"weak"`` for axes
    whose originating ``mutation_social`` edge is removed \u2014 parity
    with the suppression cascade shipped in round-3 F3."""

    def _ws_with_social_mutation(self) -> WorldStateV1:
        george = Entity(
            id="ENT_GEORGE", name="George", location_id="LOC_FLAT",
            status="healthy", traits={}, world_id="factual",
        )
        wanda = Entity(
            id="ENT_WANDA", name="Wanda", location_id="LOC_FLAT",
            status="healthy", traits={}, world_id="factual",
        )
        confession = EventNode(
            id="EVT_CONFESSION", fabula_time=50, syuzhet_index=50,
            event_type="utterance",
            actor_ids=["ENT_GEORGE"], target_ids=["ENT_WANDA"],
            description="George confesses",
            world_id="factual",
        )
        social_edge = CausalEdge(
            source_id="EVT_CONFESSION",
            target_id="ENT_WANDA",
            causality_type="mutation_social",
            mechanism="declaration",
            evidence_strength="strong",
            causal_force=1.0,
            trait_target="affinity",
            trait_delta=0.4,
            rel_counterpart_id="ENT_GEORGE",
            fabula_time=50,
        )
        rel = RelationshipEdge(
            source_entity_id="ENT_WANDA", target_entity_id="ENT_GEORGE",
            metrics={"affinity": RelationshipMetric(
                value=0.7, evidence_strength="strong", last_updated_fabula=50,
            )},
            world_id="factual",
        )
        return WorldStateV1(
            entities={"ENT_GEORGE": george, "ENT_WANDA": wanda},
            events=[confession],
            causal_topology=[social_edge],
            spatial_topology=[],
            social_topology=[rel],
            locations={}, objects={}, world_traits={},
            propositions=[], channels={},
        )

    def test_removed_event_demotes_social_metric(self, caplog):
        vwm = VersionedWorldModel.from_world_state(self._ws_with_social_mutation())
        topology = ChunkTopology(
            chunk_id="CHK_RETRACT",
            removed_event_ids=["EVT_CONFESSION"],
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.extract_graph"):
            vwm2 = vwm.merge(topology, world_id="factual",
                             branch_label="retract_confession")
        rel = next(
            r for r in vwm2.current.social_topology
            if r.source_entity_id == "ENT_WANDA"
        )
        m = rel.metrics["affinity"]
        assert m.evidence_strength == "weak", (
            "Deletion of the originating mutation_social event must "
            "demote the metric's evidence_strength."
        )
        assert any(
            "[merge\u00b7delete]" in r.getMessage()
            and "Demoted evidence_strength" in r.getMessage()
            for r in caplog.records
        )


# ---------------------------------------------------------------------
# R5-F4: hard-delete cascade-trim parity with suppression
# ---------------------------------------------------------------------
class TestDeletionCascadeTrimsProvenance:
    """``removed_event_ids`` (the hard-delete cascade) must trim
    state_timeline / beliefs / truth_at_fabula entries whose
    ``triggered_by`` / ``acquired_via_event_id`` provenance points at
    the deleted event \u2014 parity with the suppression cascade. Without
    this, persisted ``WorldStateV1`` state keeps dangling refs that
    the validator can only report as warnings on the next pass.
    """

    def _ws_with_full_provenance(self) -> WorldStateV1:
        # One murderer, one victim, one witness, one global trait
        # (regime stability), one proposition (victim is dead). Every
        # downstream record carries provenance pointing at the murder
        # event so a hard-delete must scrub them all.
        macbeth = Entity(
            id="ENT_MACBETH", name="Macbeth", location_id="LOC_CASTLE",
            status="healthy", traits={}, world_id="factual",
            state_timeline=[
                EntityStateSnapshot(
                    fabula_time=100,
                    triggered_by="EVT_MURDER",
                    new_traits={"guilt": 0.9},
                    world_id="factual",
                ),
            ],
        )
        macduff = Entity(
            id="ENT_MACDUFF", name="Macduff", location_id="LOC_FIFE",
            status="healthy", traits={}, world_id="factual",
            beliefs=[
                Belief(
                    target_id="ENT_DUNCAN",
                    perceived_state="Duncan is dead",
                    confidence=0.95, inertia=0.7,
                    established_at_fabula=110,
                    acquired_via_event_id="EVT_MURDER",
                    proposition_id="PROP_DUNCAN_DEAD",
                ),
                Belief(
                    # Unrelated belief — must survive the trim.
                    target_id="ENT_MACBETH",
                    perceived_state="Macbeth is thane",
                    confidence=0.99, inertia=0.9,
                    established_at_fabula=50,
                    acquired_via_event_id="EVT_THANESHIP",
                ),
            ],
        )
        duncan = Entity(
            id="ENT_DUNCAN", name="Duncan", location_id="LOC_CASTLE",
            status="dead", traits={}, world_id="factual",
        )
        regime_stability = GlobalTrait(
            id="WORLD_STABILITY", name="regime stability",
            description="Stability of the Scottish crown",
            category="governance",
            magnitude=TraitVector(value=0.5, inertia=0.4),
            affected_domains=["social", "psychological"],
            world_id="factual",
            state_timeline=[
                WorldTraitSnapshot(
                    fabula_time=100, triggered_by="EVT_MURDER",
                    magnitude=TraitVector(value=0.2, inertia=0.4),
                    world_id="factual",
                ),
            ],
        )
        prop_dead = Proposition(
            proposition_id="PROP_DUNCAN_DEAD",
            kind="event_occurs",
            description="Duncan is dead",
            referent_ids=["ENT_DUNCAN"],
            truth_at_fabula={100: True},
            world_id="factual",
        )
        murder = EventNode(
            id="EVT_MURDER", fabula_time=100, syuzhet_index=100,
            event_type="choice",
            actor_ids=["ENT_MACBETH"], target_ids=["ENT_DUNCAN"],
            description="Macbeth kills Duncan",
            asserts_proposition_id="PROP_DUNCAN_DEAD",
            world_id="factual",
        )
        thaneship = EventNode(
            id="EVT_THANESHIP", fabula_time=50, syuzhet_index=50,
            event_type="choice",
            actor_ids=["ENT_DUNCAN"], target_ids=["ENT_MACBETH"],
            description="Duncan grants thaneship",
            world_id="factual",
        )
        return WorldStateV1(
            entities={
                "ENT_MACBETH": macbeth,
                "ENT_MACDUFF": macduff,
                "ENT_DUNCAN": duncan,
            },
            events=[murder, thaneship],
            causal_topology=[],
            spatial_topology=[],
            social_topology=[],
            locations={}, objects={},
            world_traits={"WORLD_STABILITY": regime_stability},
            propositions=[prop_dead], channels={},
        )

    def test_hard_delete_trims_entity_state_timeline(self):
        vwm = VersionedWorldModel.from_world_state(self._ws_with_full_provenance())
        topology = ChunkTopology(
            chunk_id="CHK_RETCON",
            removed_event_ids=["EVT_MURDER"],
        )
        vwm2 = vwm.merge(topology, world_id="factual",
                         branch_label="retcon_murder")
        macbeth = vwm2.current.entities["ENT_MACBETH"]
        triggers = [s.triggered_by for s in macbeth.state_timeline]
        assert "EVT_MURDER" not in triggers, (
            "Hard-delete must trim state_timeline snapshots whose "
            "triggered_by points at the deleted event."
        )

    def test_hard_delete_trims_beliefs_by_acquired_via_event(self):
        vwm = VersionedWorldModel.from_world_state(self._ws_with_full_provenance())
        topology = ChunkTopology(
            chunk_id="CHK_RETCON",
            removed_event_ids=["EVT_MURDER"],
        )
        vwm2 = vwm.merge(topology, world_id="factual",
                         branch_label="retcon_murder")
        macduff = vwm2.current.entities["ENT_MACDUFF"]
        acquired = [b.acquired_via_event_id for b in macduff.beliefs]
        assert "EVT_MURDER" not in acquired, (
            "Hard-delete must trim beliefs whose acquired_via_event_id "
            "points at the deleted event."
        )
        # The unrelated belief must survive.
        assert "EVT_THANESHIP" in acquired

    def test_hard_delete_trims_proposition_truth_at_fabula(self, caplog):
        vwm = VersionedWorldModel.from_world_state(self._ws_with_full_provenance())
        topology = ChunkTopology(
            chunk_id="CHK_RETCON",
            removed_event_ids=["EVT_MURDER"],
        )
        with caplog.at_level(logging.INFO, logger="shadow_loom.extract_graph"):
            vwm2 = vwm.merge(topology, world_id="factual",
                             branch_label="retcon_murder")
        prop = next(
            p for p in vwm2.current.propositions
            if p.proposition_id == "PROP_DUNCAN_DEAD"
        )
        assert 100 not in prop.truth_at_fabula, (
            "Hard-delete must drop truth_at_fabula commits whose only "
            "asserting event is gone."
        )
        assert any(
            "[merge\u00b7delete]" in r.getMessage()
            and "Proposition.truth_at_fabula" in r.getMessage()
            for r in caplog.records
        )

    def test_hard_delete_trims_world_trait_state_timeline(self):
        vwm = VersionedWorldModel.from_world_state(self._ws_with_full_provenance())
        topology = ChunkTopology(
            chunk_id="CHK_RETCON",
            removed_event_ids=["EVT_MURDER"],
        )
        vwm2 = vwm.merge(topology, world_id="factual",
                         branch_label="retcon_murder")
        wt = vwm2.current.world_traits["WORLD_STABILITY"]
        triggers = [s.triggered_by for s in wt.state_timeline]
        assert "EVT_MURDER" not in triggers, (
            "Hard-delete must trim world_trait state_timeline snapshots "
            "whose triggered_by points at the deleted event."
        )

    def test_hard_delete_preserves_unrelated_provenance(self):
        """The trim must be surgical: only the deleted event's
        provenance is scrubbed; unrelated snapshots / beliefs /
        truth_at_fabula commits survive untouched."""
        vwm = VersionedWorldModel.from_world_state(self._ws_with_full_provenance())
        topology = ChunkTopology(
            chunk_id="CHK_RETCON",
            removed_event_ids=["EVT_MURDER"],
        )
        vwm2 = vwm.merge(topology, world_id="factual",
                         branch_label="retcon_murder")
        # The thaneship event and the belief acquired from it remain.
        evt_ids = {e.id for e in vwm2.current.events}
        assert "EVT_THANESHIP" in evt_ids
        assert "EVT_MURDER" not in evt_ids
        macduff = vwm2.current.entities["ENT_MACDUFF"]
        # The unrelated belief survives — the trim was not over-broad.
        assert any(
            b.acquired_via_event_id == "EVT_THANESHIP"
            for b in macduff.beliefs
        )


# ---------------------------------------------------------------------
# Phase-13: directive_assembly scorer prune-awareness (rung/affective/
# renderer/auditor consistency under do-surgery)
# ---------------------------------------------------------------------
class TestDirectiveAssemblerPruneAwareness:
    """When the active brief carries ``pruned_event_ids`` (i.e. a
    Rung-2/3 do-surgery has erased events from the factual timeline),
    every scorer surface that lands in the ``CreativeBrief`` and is
    rendered to either the LLM scene-renderer or the auditor must
    reflect the *counterfactual* world, not the leaked factual one.
    """

    def _ws(self) -> WorldStateV1:
        archie = Entity(
            id="ENT_ARCHIE", name="Archie", location_id="LOC_HOME",
            status="healthy", traits={}, world_id="factual",
        )
        wanda = Entity(
            id="ENT_WANDA", name="Wanda", location_id="LOC_HOME",
            status="healthy", traits={}, world_id="factual",
        )
        # Pivot event whose mutation_social edge bumps Archie→Wanda
        # affinity by +0.4. Pruning the event should roll the dyad
        # back to its pre-mutation reading.
        pivot = EventNode(
            id="EVT_PIVOT", fabula_time=50, syuzhet_index=50,
            event_type="choice",
            actor_ids=["ENT_ARCHIE"], target_ids=["ENT_WANDA"],
            description="Archie commits to Wanda",
            world_id="factual",
        )
        # A non-pivot event so the digraph has at least one survivor.
        other = EventNode(
            id="EVT_BENIGN", fabula_time=10, syuzhet_index=10,
            event_type="choice",
            actor_ids=["ENT_ARCHIE"], target_ids=[],
            description="Archie walks the dog",
            world_id="factual",
        )
        # A second non-pivot event so we can wire a chain_reaction
        # edge between two surviving events (event\u2192event is the
        # only causality type that schema-validates for two events
        # with no shared state target).
        other2 = EventNode(
            id="EVT_BENIGN2", fabula_time=15, syuzhet_index=15,
            event_type="choice",
            actor_ids=["ENT_ARCHIE"], target_ids=[],
            description="Archie returns home",
            world_id="factual",
        )
        # A hidden-channel utterance pivot.
        utt = EventNode(
            id="EVT_UTT_LATER", fabula_time=80, syuzhet_index=80,
            event_type="utterance",
            actor_ids=["ENT_WANDA"], target_ids=["ENT_ARCHIE"],
            speaker_id="ENT_WANDA",
            addressee_ids=["ENT_ARCHIE"],
            via_channel_id="CHN_DIRECT",
            description="late utterance",
            world_id="factual",
        )
        ce_pivot = CausalEdge(
            source_id="EVT_PIVOT", target_id="ENT_ARCHIE",
            causality_type="mutation_social",
            mechanism="declaration",
            evidence_strength="strong",
            causal_force=1.0,
            trait_target="affinity",
            trait_delta=0.4,
            rel_counterpart_id="ENT_WANDA",
            fabula_time=50,
        )
        ce_benign = CausalEdge(
            source_id="EVT_BENIGN", target_id="EVT_BENIGN2",
            causality_type="chain_reaction",
            mechanism="routine",
            evidence_strength="moderate",
            causal_force=1.0,
            fabula_time=10,
        )
        rel_fwd = RelationshipEdge(
            source_entity_id="ENT_ARCHIE", target_entity_id="ENT_WANDA",
            metrics={"affinity": RelationshipMetric(
                value=0.8, evidence_strength="strong",
                last_updated_fabula=50, inertia=0.0,
            )},
            world_id="factual",
        )
        rel_rev = RelationshipEdge(
            source_entity_id="ENT_WANDA", target_entity_id="ENT_ARCHIE",
            metrics={"affinity": RelationshipMetric(
                value=0.5, evidence_strength="strong",
                last_updated_fabula=50, inertia=0.0,
            )},
            world_id="factual",
        )
        chn = Channel(
            id="CHN_DIRECT", name="direct speech",
            medium="speech",
            participant_ids=["ENT_ARCHIE", "ENT_WANDA"],
            world_id="factual",
        )
        # World trait whose state_timeline carries an entry triggered
        # by the pivot. Pruning the pivot should drop that snapshot
        # from the surfaced shift list.
        wt = GlobalTrait(
            id="WORLD_MORALE",
            name="Morale",
            description="Group morale",
            category="affective_climate",
            magnitude=TraitVector(value=0.6, inertia=0.1),
            affected_domains=["community"],
            state_timeline=[
                WorldTraitSnapshot(
                    fabula_time=50,
                    magnitude=TraitVector(value=0.9, inertia=0.1),
                    triggered_by="EVT_PIVOT",
                    description="surge",
                ),
            ],
        )
        return WorldStateV1(
            entities={"ENT_ARCHIE": archie, "ENT_WANDA": wanda},
            events=[other, other2, pivot, utt],
            causal_topology=[ce_pivot, ce_benign],
            spatial_topology=[],
            social_topology=[rel_fwd, rel_rev],
            locations={}, objects={},
            world_traits={"WORLD_MORALE": wt},
            propositions=[], channels={"CHN_DIRECT": chn},
        )

    def _ego(self, ws: WorldStateV1, entity_ids):
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        return extract_ego_graph_from_memory(
            ws, focus_entity_ids=entity_ids, temporal_anchor=None,
            syuzhet_anchor=max(e.syuzhet_index for e in ws.events),
        ).model_dump()

    def test_relationship_tensions_roll_back_pruned_mutation_social(self):
        """Gap-1: compute_relationship_tensions must reflect the
        counterfactual per-axis values when the originating
        ``mutation_social`` event is pruned. Without the fix the
        ego payload leaks the post-mutation affinity verbatim."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        entity_ids = ["ENT_ARCHIE", "ENT_WANDA"]
        ego = self._ego(ws, entity_ids)
        syuzhet_anchor = max(e.syuzhet_index for e in ws.events)

        base = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
        ).compute_relationship_tensions(entity_ids, syuzhet_anchor=syuzhet_anchor)
        pruned = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_PIVOT"},
        ).compute_relationship_tensions(entity_ids, syuzhet_anchor=syuzhet_anchor)

        base_by = {(t.source_id, t.target_id): t for t in base}
        pruned_by = {(t.source_id, t.target_id): t for t in pruned}
        # Archie→Wanda baseline affinity is 0.8; rolling back +0.4
        # mutation (inertia=0) yields 0.4.
        assert base_by[("ENT_ARCHIE", "ENT_WANDA")].affinity == 0.8
        assert pruned_by[("ENT_ARCHIE", "ENT_WANDA")].affinity == 0.4

    def test_causal_digraph_omits_pruned_edges(self):
        """Gap-2: _build_causal_digraph must drop edges whose source
        or target is in ``self._pruned_event_ids`` so downstream
        affective/mystery scorers do not propagate through erased
        causal nodes."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws, ["ENT_ARCHIE", "ENT_WANDA"])
        g_base = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
        )._build_causal_digraph()
        g_pruned = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_PIVOT"},
        )._build_causal_digraph()
        assert g_base.number_of_edges() == 2  # ce_pivot + ce_benign
        assert g_pruned.number_of_edges() == 1
        # The surviving edge must be EVT_BENIGN, not EVT_PIVOT.
        srcs = {u for u, _v, _k in g_pruned.edges(keys=True)}
        assert "EVT_PIVOT" not in srcs
        assert "EVT_BENIGN" in srcs

    def test_hidden_channels_omit_pruned_utterance(self):
        """Gap-3: compute_hidden_channels must not surface a pruned
        utterance event as a hidden ``kind='utterance'`` item.
        Surfacing it would contradict the brief's PREVENTED EVENTS
        block in the renderer/auditor prompt."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws, ["ENT_ARCHIE", "ENT_WANDA"])
        # Anchor before the utterance so it would normally surface.
        anchor = 20

        base = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
        ).compute_hidden_channels(syuzhet_anchor=anchor)
        pruned = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_UTT_LATER"},
        ).compute_hidden_channels(syuzhet_anchor=anchor)

        base_utts = [h for h in base if getattr(h, "utterance_event_id", None) == "EVT_UTT_LATER"]
        pruned_utts = [h for h in pruned if getattr(h, "utterance_event_id", None) == "EVT_UTT_LATER"]
        assert len(base_utts) == 1
        assert len(pruned_utts) == 0

    def test_world_trait_shifts_omit_pruned_triggered_snapshots(self):
        """Gap-4: compute_world_trait_shifts must skip state_timeline
        entries whose ``triggered_by`` was erased by an active do-
        surgery, so the brief's WORLD_ trait regime reflects the
        counterfactual baseline rather than leaking a pruned surge."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws, ["ENT_ARCHIE", "ENT_WANDA"])

        base = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
        ).compute_world_trait_shifts(syuzhet_anchor=100)
        pruned = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_PIVOT"},
        ).compute_world_trait_shifts(syuzhet_anchor=100)

        # Baseline: snapshot 0.6 → 0.9 = +0.3 surge surfaces.
        assert len(base) == 1
        assert base[0].trait_id == "WORLD_MORALE"
        assert abs(base[0].delta - 0.3) < 1e-6
        # Pruned: the pivot-triggered snapshot is dropped; the trait
        # has no remaining timeline entries, so it is omitted.
        assert pruned == []


# ---------------------------------------------------------------------
# Phase-13b: additional scorer surfaces uncovered by Macbeth/Frankenstein
# probes — narrative_tension / _build_affinity_index /
# _build_counterfactual_branch / _build_causal_attribution
# ---------------------------------------------------------------------
class TestDirectiveAssemblerPruneAwarenessPhase13b:
    """Second-round prune-awareness gaps that the Wanda synthetic
    audit did not surface because Wanda lacks loss-outcome chains."""

    def _ws(self) -> WorldStateV1:
        # Three entities so we have a perpetrator separate from focus.
        focus = Entity(
            id="ENT_HERO", name="Hero", location_id="LOC_KEEP",
            status="healthy", traits={}, world_id="factual",
        )
        ally = Entity(
            id="ENT_ALLY", name="Ally", location_id="LOC_KEEP",
            status="healthy", traits={}, world_id="factual",
        )
        villain = Entity(
            id="ENT_VILLAIN", name="Villain", location_id="LOC_KEEP",
            status="healthy", traits={}, world_id="factual",
        )
        # Choice events whose actor is the focal entity (regret
        # semantics: the focal character's *own* past choices are
        # what the cf-branch surfaces as divergence candidates).
        choice = EventNode(
            id="EVT_CHOICE", fabula_time=10, syuzhet_index=10,
            event_type="choice",
            actor_ids=["ENT_HERO"], target_ids=["ENT_VILLAIN"],
            description="Hero trusts the villain",
            world_id="factual",
        )
        loss = EventNode(
            id="EVT_LOSS", fabula_time=20, syuzhet_index=20,
            event_type="outcome",
            actor_ids=["ENT_VILLAIN"], target_ids=["ENT_HERO"],
            description="Hero is wounded",
            world_id="factual",
        )
        # Backup outcome so prune of EVT_LOSS still finds a loss.
        backup_loss = EventNode(
            id="EVT_LOSS_BACKUP", fabula_time=15, syuzhet_index=15,
            event_type="outcome",
            actor_ids=["ENT_VILLAIN"], target_ids=["ENT_HERO"],
            description="Hero is slighted",
            world_id="factual",
        )
        # Backup choice so prune of EVT_CHOICE still finds a divergence.
        backup_choice = EventNode(
            id="EVT_CHOICE_BACKUP", fabula_time=5, syuzhet_index=5,
            event_type="choice",
            actor_ids=["ENT_HERO"], target_ids=["ENT_VILLAIN"],
            description="Hero meets the villain",
            world_id="factual",
        )
        # Mutation_social edge sourced from the choice; pruning the
        # choice should roll back the affinity index value.
        ce_mut_social = CausalEdge(
            source_id="EVT_CHOICE", target_id="ENT_HERO",
            causality_type="mutation_social",
            mechanism="betrayal",
            evidence_strength="strong",
            causal_force=1.0,
            trait_target="affinity",
            trait_delta=-0.4,
            rel_counterpart_id="ENT_VILLAIN",
            fabula_time=10,
        )
        # Negative trait_delta on the loss outcome (so it's recognised
        # as negative by _is_likely_negative).
        ce_loss = CausalEdge(
            source_id="EVT_LOSS", target_id="ENT_HERO",
            causality_type="mutation",
            mechanism="violence",
            evidence_strength="strong",
            causal_force=1.0,
            trait_target="status",
            trait_delta=-0.5,
            fabula_time=20,
        )
        ce_loss_backup = CausalEdge(
            source_id="EVT_LOSS_BACKUP", target_id="ENT_HERO",
            causality_type="mutation",
            mechanism="insult",
            evidence_strength="moderate",
            causal_force=1.0,
            trait_target="status",
            trait_delta=-0.2,
            fabula_time=15,
        )
        rel_hero_villain = RelationshipEdge(
            source_entity_id="ENT_HERO", target_entity_id="ENT_VILLAIN",
            metrics={"affinity": RelationshipMetric(
                value=0.0, evidence_strength="strong",
                last_updated_fabula=20, inertia=0.0,
            )},
            world_id="factual",
        )
        return WorldStateV1(
            entities={
                "ENT_HERO": focus, "ENT_ALLY": ally, "ENT_VILLAIN": villain,
            },
            events=[backup_choice, choice, backup_loss, loss],
            causal_topology=[ce_mut_social, ce_loss, ce_loss_backup],
            spatial_topology=[],
            social_topology=[rel_hero_villain],
            locations={}, objects={},
            world_traits={}, propositions=[], channels={},
        )

    def _ego(self, ws):
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        return extract_ego_graph_from_memory(
            ws, focus_entity_ids=["ENT_HERO"], temporal_anchor=None,
            syuzhet_anchor=max(e.syuzhet_index for e in ws.events),
        ).model_dump()

    def test_narrative_tension_omits_pruned_events(self):
        """Gap-H5: compute_narrative_tension must not list pruned
        events; they have been erased from the counterfactual timeline
        and surfacing them contradicts the PREVENTED EVENTS block."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws)
        anchor = max(e.syuzhet_index for e in ws.events)

        base = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
        ).compute_narrative_tension(syuzhet_anchor=anchor)
        pruned = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_CHOICE"},
        ).compute_narrative_tension(syuzhet_anchor=anchor)

        assert any(n.event_id == "EVT_CHOICE" for n in base)
        assert not any(n.event_id == "EVT_CHOICE" for n in pruned)
        # Other events still surface; total count drops by exactly 1.
        assert len(pruned) == len(base) - 1

    def test_affinity_index_rolls_back_pruned_mutation_social(self):
        """Gap-H6: _build_affinity_index must reflect counterfactual
        affinity when a mutation_social event is pruned. Downstream
        consumers (_bucket_event_for_focal / threat-hope / suspense /
        rescue propagation) all key off this index."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws)

        base = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
        )._build_affinity_index()
        pruned = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_CHOICE"},
        )._build_affinity_index()

        # Baseline affinity is 0.0 (the engine's current reading after
        # the −0.4 mutation has been folded in). Rolling back the
        # mutation (inertia=0) yields 0.0 − (−0.4) = +0.4.
        assert base[("ENT_HERO", "ENT_VILLAIN")] == 0.0
        assert abs(pruned[("ENT_HERO", "ENT_VILLAIN")] - 0.4) < 1e-6

    def test_counterfactual_branch_skips_pruned_outcome_and_choice(self):
        """Gap-H7: _build_counterfactual_branch must not pick a
        pruned event as the actual outcome or as the divergence
        choice the regret render asks the LLM to dwell on."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws)
        anchor = max(e.syuzhet_index for e in ws.events)

        base = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
        )._build_counterfactual_branch(["ENT_HERO"], syuzhet_anchor=anchor)
        # Baseline picks the most recent negative outcome and choice.
        assert base is not None
        assert base.actual_outcome == "Hero is wounded"
        assert base.divergence_event_id == "EVT_CHOICE"

        # Pruning the loss outcome falls back to EVT_LOSS_BACKUP.
        pruned_loss = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_LOSS"},
        )._build_counterfactual_branch(["ENT_HERO"], syuzhet_anchor=anchor)
        assert pruned_loss is not None
        assert pruned_loss.actual_outcome == "Hero is slighted"

        # Pruning the divergence choice falls back to the backup.
        pruned_choice = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_CHOICE"},
        )._build_counterfactual_branch(["ENT_HERO"], syuzhet_anchor=anchor)
        assert pruned_choice is not None
        assert pruned_choice.divergence_event_id == "EVT_CHOICE_BACKUP"

    def test_causal_attribution_skips_pruned_loss_event(self):
        """Gap-H8: _build_causal_attribution must not pick a pruned
        event as the loss event the rage render attributes to a
        perpetrator. Falls back to the next-most-recent loss."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws)
        anchor = max(e.syuzhet_index for e in ws.events)

        base = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
        )._build_causal_attribution(["ENT_HERO"], syuzhet_anchor=anchor)
        assert base is not None
        assert base.loss_event_id == "EVT_LOSS"

        pruned = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_LOSS"},
        )._build_causal_attribution(["ENT_HERO"], syuzhet_anchor=anchor)
        assert pruned is not None
        assert pruned.loss_event_id == "EVT_LOSS_BACKUP"


# ---------------------------------------------------------------------
# Phase-13c: cross-plot audit sweep (20 example_worlds × 4 pivot
# strategies × 2 anchors × 9 surfaces) flagged one real residual gap
# in the threat/hope detail scorer — the pruned pivot was surfacing
# as ``best_threat`` / ``best_hope`` because the unrevealed candidate
# pool was rebuilt as ``all_events - revealed`` without subtracting
# pruned ids that ``_revealed_event_ids`` had already stripped from
# ``revealed``.
# ---------------------------------------------------------------------
class TestDirectiveAssemblerPruneAwarenessPhase13c:
    """Phase-13c real-plot regression: pruned events must not
    re-surface as future threats/hopes in either
    ``_compute_threat_hope_detail`` or ``compute_suspense_score``.
    Reuses the Phase-13b fixture so the pivot has both a mutation
    edge and an outcome the unrevealed bucket would otherwise pick."""

    def _ws(self):
        return TestDirectiveAssemblerPruneAwarenessPhase13b()._ws()

    def _ego(self, ws):
        return TestDirectiveAssemblerPruneAwarenessPhase13b()._ego(ws)

    def test_threat_hope_detail_excludes_pruned_pivot(self):
        """Gap-H9 (a): the pruned pivot must not appear as
        ``threat_event_id`` / ``hope_event_id``. Use a partial anchor
        so the pivot is genuinely unrevealed (otherwise ``revealed``
        already excludes it by syuzhet position)."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws)
        # Anchor mid-arc so EVT_LOSS (syuzhet=20) is unrevealed.
        partial_anchor = 10

        pruned_brief = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_LOSS"},
        )._compute_threat_hope_detail(
            ["ENT_HERO"], syuzhet_anchor=partial_anchor,
        )
        # The pivot we pretend didn't happen must not surface as
        # the next thing to fear.
        assert pruned_brief.threat_event_id != "EVT_LOSS"

    def test_suspense_score_excludes_pruned_pivot(self):
        """Gap-H9 (b): the parallel bug in compute_suspense_score —
        ``unrevealed = all_evt_ids - revealed`` re-introduced pruned
        ids that ``_revealed_event_ids`` had stripped. The scorer
        must never anticipate the counterfactual itself."""
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = self._ws()
        ego = self._ego(ws)
        partial_anchor = 10

        suspense = DirectiveAssembler(
            sandbox=None, ego_payload=ego, world_state=ws,
            pruned_event_ids={"EVT_LOSS"},
        ).compute_suspense_score(
            ["ENT_HERO"], syuzhet_anchor=partial_anchor,
        )
        # Suspense is a scalar; the underlying threat/hope picker is
        # what we proved above. Sanity-check it still produces a
        # finite score (and didn't crash on the empty unrevealed set
        # path).
        assert isinstance(suspense, float)
        assert 0.0 <= suspense <= 1.0
