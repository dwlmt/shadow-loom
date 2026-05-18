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
"""

from __future__ import annotations

import logging
from typing import Optional

from shadow_loom.causal_physics import BlockedPropagation, CausalPhysicsResult
from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ChunkTopology
from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    RelationshipEdge,
    RelationshipMetric,
    WorldStateV1,
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
