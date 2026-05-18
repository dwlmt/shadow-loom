# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Synthetic tests covering the round-3 audit fixes (F1-F5).

These tests exercise the deeper structural fixes shipped after the
*Fish Called Wanda* counterfactual debug session:

* **F1** \u2014 ``intervention_inert`` propagation from
  :class:`CausalPhysicsResult` through ``narrative_physics`` dict,
  ``_render_engine_priors``, both ``build_intervention_brief`` /
  ``build_counterfactual_brief``, and the new
  ``_build_inert_intervention_constraints`` HARD constraint block.
* **F2** \u2014 ``_log_feedback_outcome`` warnings on refinement
  non-convergence / engine-threshold failure / correction_error.
* **F3** \u2014 Suppression-cascade demotion of
  :class:`RelationshipMetric` evidence_strength when the originating
  ``mutation_social`` causal edge is pruned on the shadow branch.
* **F4** \u2014 ``inert_intervention_aftermath`` violation type in the
  auditor prompt.
* **F5** \u2014 ``_render_engine_priors`` surfacing
  ``pruned_utterance_event_ids`` / ``disabled_channel_ids`` /
  ``skipped_interventions`` / ``INERT INTERVENTION``.
"""

from __future__ import annotations

import logging
from typing import Optional

import pytest

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
# F1 + F5: engine-priors rendering
# ---------------------------------------------------------------------
class TestEnginePriorsInertAndPruned:
    """The pipeline's prior-string sent to the re-extraction agents
    must surface every negative-physics signal."""

    def _priors(self, **kwargs) -> Optional[str]:
        from shadow_loom.pipeline import _render_engine_priors
        return _render_engine_priors(kwargs)

    def test_inert_intervention_surfaces_in_priors(self):
        out = self._priors(
            mutations=[],
            intervention_inert=True,
            intervention_inert_reason="all 2 do-target(s) Rule-3 pruned",
        )
        assert out is not None
        assert "INERT INTERVENTION" in out
        assert "Rule-3 pruned" in out
        assert "ZERO downstream" in out

    def test_pruned_utterances_surface_in_priors(self):
        out = self._priors(
            pruned_utterance_event_ids=["EVT_UTT_1", "EVT_UTT_2"],
        )
        assert out is not None
        assert "Pruned utterance events" in out
        assert "EVT_UTT_1" in out and "EVT_UTT_2" in out

    def test_disabled_channels_surface_in_priors(self):
        out = self._priors(
            disabled_channel_ids=["CH_PHONE", "CH_LETTER"],
        )
        assert out is not None
        assert "Disabled channels" in out
        assert "CH_PHONE" in out

    def test_skipped_interventions_surface_in_priors(self):
        out = self._priors(
            skipped_interventions=[
                {"target_path": "ENT_GHOST.status", "reason": "unknown_node"},
            ],
        )
        assert out is not None
        assert "Skipped interventions" in out
        assert "ENT_GHOST" in out

    def test_empty_priors_returns_none(self):
        # No actionable engine output \u2014 caller should omit the block.
        assert self._priors() is None
        assert self._priors(mutations=[], blocked=[]) is None

    def test_priors_renders_no_inert_block_when_flag_false(self):
        out = self._priors(
            intervened_nodes=["ENT_X"],
            intervention_inert=False,
        )
        assert out is not None
        assert "INERT INTERVENTION" not in out


# ---------------------------------------------------------------------
# F1: inert constraint block emission
# ---------------------------------------------------------------------
class TestInertInterventionConstraintBlock:
    """The brief builders must emit a HARD constraint block whenever
    the engine flagged ``intervention_inert=True``."""

    def test_block_emitted_when_inert(self):
        from shadow_loom.generation import _build_inert_intervention_constraints
        blocks = _build_inert_intervention_constraints(
            intervention_inert=True,
            intervention_inert_reason="cycle-absorbed",
            rung_label="Rung-2 intervention",
            world_label="intervened",
        )
        assert len(blocks) == 1
        b = blocks[0]
        assert b.constraint_type == "mathematical"
        assert b.priority == "hard"
        assert "INERT INTERVENTION" in b.instruction
        assert "Rung-2 intervention" in b.instruction
        assert "cycle-absorbed" in b.instruction
        assert "Stage the *attempt*" in b.instruction
        assert b.evidence["intervention_inert"] is True

    def test_no_block_when_not_inert(self):
        from shadow_loom.generation import _build_inert_intervention_constraints
        assert _build_inert_intervention_constraints(
            intervention_inert=False,
            intervention_inert_reason=None,
            rung_label="Rung-2 intervention",
            world_label="intervened",
        ) == []

    def test_block_handles_missing_reason(self):
        from shadow_loom.generation import _build_inert_intervention_constraints
        blocks = _build_inert_intervention_constraints(
            intervention_inert=True,
            intervention_inert_reason=None,
            rung_label="Rung-3 counterfactual",
            world_label="counterfactual",
        )
        assert len(blocks) == 1
        # Falls back to a generic reason rather than crashing on None.
        assert "no actionable downstream propagation" in blocks[0].instruction


# ---------------------------------------------------------------------
# F2: refinement-loop loud failure logging
# ---------------------------------------------------------------------
class TestLogFeedbackOutcome:
    """Each of converged=False, engine_thresholds_passed=False, and
    correction_error must emit its own WARNING."""

    def _scene(self):
        from shadow_loom.generation import GeneratedScene
        return GeneratedScene(prose="stub")

    def _make_feedback(self, **overrides):
        from shadow_loom.auditor import FeedbackLoopResult
        kwargs = dict(
            final_scene=self._scene(),
            converged=True,
            iterations=1,
            history=[],
            final_graph_version=0,
        )
        kwargs.update(overrides)
        return FeedbackLoopResult(**kwargs)

    def test_no_warnings_on_clean_pass(self, caplog):
        from shadow_loom.pipeline import _log_feedback_outcome
        fb = self._make_feedback(converged=True)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.pipeline"):
            _log_feedback_outcome(fb, async_path=False)
        assert not any(
            "did NOT converge" in r.getMessage()
            or "Engine-threshold check FAILED" in r.getMessage()
            or "correction_error" in r.getMessage()
            for r in caplog.records
        )

    def test_warns_on_non_convergence(self, caplog):
        from shadow_loom.pipeline import _log_feedback_outcome
        fb = self._make_feedback(converged=False, iterations=4)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.pipeline"):
            _log_feedback_outcome(fb, async_path=False)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("did NOT converge after 4" in m for m in msgs)
        # Tag selection on the sync path.
        assert any("[Pipeline]" in m for m in msgs)

    def test_warns_on_engine_threshold_failure(self, caplog):
        from shadow_loom.pipeline import _log_feedback_outcome
        fb = self._make_feedback(
            converged=True,
            engine_thresholds_passed=False,
            engine_threshold_failures=["affective_loss_mse>0.5", "kl<min"],
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.pipeline"):
            _log_feedback_outcome(fb, async_path=False)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("Engine-threshold check FAILED" in m for m in msgs)
        assert any("affective_loss_mse>0.5" in m for m in msgs)

    def test_warns_on_correction_error(self, caplog):
        from shadow_loom.pipeline import _log_feedback_outcome
        fb = self._make_feedback(
            converged=False,
            correction_error="OpenRouter timeout after 60s",
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.pipeline"):
            _log_feedback_outcome(fb, async_path=False)
        msgs = [r.getMessage() for r in caplog.records]
        # Both non-convergence AND correction_error warnings fire.
        assert any("did NOT converge" in m for m in msgs)
        assert any("correction_error=" in m for m in msgs)
        assert any("OpenRouter timeout" in m for m in msgs)

    def test_async_tag_used_on_async_path(self, caplog):
        from shadow_loom.pipeline import _log_feedback_outcome
        fb = self._make_feedback(converged=False)
        with caplog.at_level(logging.WARNING, logger="shadow_loom.pipeline"):
            _log_feedback_outcome(fb, async_path=True)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[Pipeline\u00b7Async]" in m for m in msgs)


# ---------------------------------------------------------------------
# F3: suppression demotes RelationshipMetric.evidence_strength
# ---------------------------------------------------------------------
class TestSuppressionDemotesSocialMetric:
    """Suppressing a ``mutation_social`` event must demote
    ``RelationshipMetric.evidence_strength`` to 'weak' on the
    affected axis because the metric's last justifying event is now
    gone."""

    def _ws_with_social_mutation(self) -> WorldStateV1:
        george = Entity(
            id="ENT_GEORGE", name="George", location_id="LOC_FLAT",
            status="healthy", traits={}, world_id="factual",
        )
        wanda = Entity(
            id="ENT_WANDA", name="Wanda", location_id="LOC_FLAT",
            status="healthy", traits={}, world_id="factual",
        )
        confession_event = EventNode(
            id="EVT_CONFESSION", fabula_time=50, syuzhet_index=50,
            event_type="utterance",
            actor_ids=["ENT_GEORGE"], target_ids=["ENT_WANDA"],
            description="George confesses love to Wanda",
            world_id="factual",
        )
        # mutation_social edge: confession bumps Wanda's affinity to George
        # source = EVT_, target = perspective entity (Wanda),
        # rel_counterpart_id = other (George).
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
                value=0.7,
                evidence_strength="strong",
                last_updated_fabula=50,
            )},
            # Round-9 audit fix \u2014 demotion is now branch-scoped, so for
            # this shadow-branch merge the rel must be tagged shadow to
            # exercise the demotion path. Cross-branch demotion (factual
            # rel during shadow merge) is now blocked as a leak.
            world_id="shadow",
        )
        return WorldStateV1(
            entities={"ENT_GEORGE": george, "ENT_WANDA": wanda},
            events=[confession_event],
            causal_topology=[social_edge],
            spatial_topology=[],
            social_topology=[rel],
            locations={}, objects={}, world_traits={},
            propositions=[], channels={},
        )

    def test_suppression_demotes_strong_to_weak(self, caplog):
        vwm = VersionedWorldModel.from_world_state(self._ws_with_social_mutation())
        topology = ChunkTopology(
            chunk_id="CHK_CF",
            suppressed_event_ids=["EVT_CONFESSION"],
        )
        with caplog.at_level(logging.WARNING, logger="shadow_loom.extract_graph"):
            vwm2 = vwm.merge(topology, world_id="shadow", branch_label="cf_no_confession")
        rel = next(
            r for r in vwm2.current.social_topology
            if r.source_entity_id == "ENT_WANDA"
            and r.target_entity_id == "ENT_GEORGE"
        )
        m = rel.metrics["affinity"]
        assert m.evidence_strength == "weak", (
            "RelationshipMetric.evidence_strength must demote to 'weak' "
            "when its originating mutation_social event is suppressed."
        )
        # Warning fired naming the affected axis.
        assert any(
            "Demoted evidence_strength" in r.getMessage()
            and "ENT_WANDA" in r.getMessage()
            and "affinity" in r.getMessage()
            for r in caplog.records
        )

    def test_no_demotion_when_other_event_suppressed(self):
        # Suppressing an unrelated event must not touch the metric.
        ws = self._ws_with_social_mutation()
        # Add an unrelated event we can suppress.
        ws.events.append(EventNode(
            id="EVT_UNRELATED", fabula_time=60, syuzhet_index=60,
            event_type="outcome",
            actor_ids=["ENT_GEORGE"], target_ids=[],
            description="unrelated",
            world_id="factual",
        ))
        vwm = VersionedWorldModel.from_world_state(ws)
        vwm2 = vwm.merge(
            ChunkTopology(
                chunk_id="CHK", suppressed_event_ids=["EVT_UNRELATED"],
            ),
            world_id="shadow", branch_label="cf_unrelated",
        )
        rel = next(
            r for r in vwm2.current.social_topology
            if r.source_entity_id == "ENT_WANDA"
        )
        assert rel.metrics["affinity"].evidence_strength == "strong"


# ---------------------------------------------------------------------
# F4: auditor prompt carries the inert_intervention_aftermath rule
# ---------------------------------------------------------------------
class TestAuditorPromptInertRule:
    """The shipped auditor prompt must enumerate the new violation
    type and rule so audits see them in the system prompt."""

    def _prompt(self) -> str:
        from pathlib import Path
        import shadow_loom
        root = Path(shadow_loom.__file__).parent
        return (root / "prompts" / "auditor.md").read_text(encoding="utf-8")

    def test_violation_type_enumerated(self):
        assert "inert_intervention_aftermath" in self._prompt()

    def test_precedence_rule_present(self):
        text = self._prompt()
        assert "Inert-intervention precedence" in text
        assert "INERT INTERVENTION" in text

    def test_inert_rule_shields_omission_and_flags_aftermath(self):
        text = self._prompt()
        # Defensive policy: do NOT fire spurious violations for correct
        # omissions, but DO flag aftermath leaks.
        assert "Do NOT fire `miracle_step`" in text
        assert "inert_intervention_aftermath:" in text
