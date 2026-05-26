# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Round-9 audit \u2014 cross-branch leaks, shadow-branch problems, and
renderer/auditor inconsistencies.

* **R9-F1** (critical) \u2014 ``_demote_social_metric_axes`` did not gate
  on ``rel.world_id`` matching the merge branch. A shadow merge whose
  removed ``mutation_social`` edge shared ``(source, target, axis)``
  with a factual ``RelationshipMetric`` would demote the FACTUAL
  metric's ``evidence_strength`` to ``"weak"`` \u2014 a cross-branch leak
  silently corrupting canonical state.
* **R9-F5** (critical) \u2014 ``AuditViolation.violation_type`` Literal
  was missing ``inert_intervention_aftermath`` despite the auditor
  prompt and the UI ``VIOLATION_EXPLANATIONS`` both referencing it.
  The LLM emission of that violation would fail Pydantic validation
  and the violation would be silently dropped \u2014 exactly the path
  meant to police prose that contradicts inert verdicts.
* **R9-F6** (minor) \u2014 ``build_intervention_brief`` accepted
  ``rule2_redundant_evidence`` but never emitted the corresponding
  soft-constraint block (only ``build_counterfactual_brief`` did),
  leaving Rule-2 evidence-gating opaque to the Rung-2 renderer.
"""
from __future__ import annotations

from shadow_loom.auditor import AuditViolation
from shadow_loom.extract_graph import _demote_social_metric_axes
from shadow_loom.generation import build_intervention_brief
from shadow_loom.models import (
    Entity,
    Location,
    RelationshipEdge,
    RelationshipMetric,
    TraitVector,
    WorldStateV1,
)


# ---------------------------------------------------------------------
# R9-F1 \u2014 social-metric demotion is branch-scoped
# ---------------------------------------------------------------------
class TestSocialMetricDemotionBranchScoped:
    def _social_topology_two_branches(self) -> list[RelationshipEdge]:
        # Factual + shadow record share the SAME (source, target,
        # axis) triple. A shadow merge that removes a mutation_social
        # edge MUST NOT demote the factual metric.
        return [
            RelationshipEdge(
                source_entity_id="ENT_A", target_entity_id="ENT_B",
                metrics={"affinity": RelationshipMetric(
                    value=0.6, evidence_strength="strong",
                    last_updated_fabula=10,
                )},
                world_id="factual",
            ),
            RelationshipEdge(
                source_entity_id="ENT_A", target_entity_id="ENT_B",
                metrics={"affinity": RelationshipMetric(
                    value=-0.4, evidence_strength="strong",
                    last_updated_fabula=20,
                )},
                world_id="shadow",
            ),
        ]

    def test_shadow_merge_does_not_demote_factual_metric(self):
        social = self._social_topology_two_branches()
        n = _demote_social_metric_axes(
            social,
            [("ENT_A", "ENT_B", "affinity")],
            merge_world_id="shadow",
            source="suppress",
        )
        # Exactly one demotion \u2014 on the shadow record.
        assert n == 1
        factual = next(r for r in social if r.world_id == "factual")
        shadow = next(r for r in social if r.world_id == "shadow")
        assert factual.metrics["affinity"].evidence_strength == "strong", (
            "factual RelationshipMetric must NOT be demoted by a "
            "shadow-branch merge"
        )
        assert shadow.metrics["affinity"].evidence_strength == "weak"

    def test_factual_merge_does_not_demote_shadow_metric(self):
        social = self._social_topology_two_branches()
        n = _demote_social_metric_axes(
            social,
            [("ENT_A", "ENT_B", "affinity")],
            merge_world_id="factual",
            source="delete",
        )
        assert n == 1
        factual = next(r for r in social if r.world_id == "factual")
        shadow = next(r for r in social if r.world_id == "shadow")
        assert factual.metrics["affinity"].evidence_strength == "weak"
        assert shadow.metrics["affinity"].evidence_strength == "strong"

    def test_legacy_missing_world_id_treated_as_factual(self):
        # Defensive: a RelationshipEdge whose world_id defaults must
        # still be eligible for demotion on a factual merge (back-compat).
        rel = RelationshipEdge(
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            metrics={"affinity": RelationshipMetric(
                value=0.2, evidence_strength="strong",
            )},
        )
        # Strip world_id to simulate a legacy row (in practice the
        # default is already "factual"; this still verifies the guard
        # does not over-reject).
        n = _demote_social_metric_axes(
            [rel],
            [("ENT_A", "ENT_B", "affinity")],
            merge_world_id="factual",
            source="delete",
        )
        assert n == 1
        assert rel.metrics["affinity"].evidence_strength == "weak"


# ---------------------------------------------------------------------
# R9-F5 \u2014 inert_intervention_aftermath is in the Literal enum
# ---------------------------------------------------------------------
class TestInertAftermathInViolationLiteral:
    def test_literal_admits_inert_intervention_aftermath(self):
        field = AuditViolation.model_fields["violation_type"]
        args = getattr(field.annotation, "__args__", ())
        assert "inert_intervention_aftermath" in args, (
            "the Literal must list inert_intervention_aftermath so LLM "
            "emissions of this violation type survive Pydantic validation"
        )

    def test_can_instantiate_inert_aftermath_violation(self):
        v = AuditViolation(
            violation_type="inert_intervention_aftermath",
            rationale=("inert_intervention_aftermath: prose says "
                       "'still steady' but the engine flagged inert"),
            severity="critical",
            description="Prose depicts the change taking hold.",
            feedback="Remove aftermath beats; only stage attempted surgery.",
        )
        assert v.violation_type == "inert_intervention_aftermath"


# ---------------------------------------------------------------------
# R9-F6 \u2014 intervention brief surfaces Rule-2 redundant-evidence block
# ---------------------------------------------------------------------
class TestInterventionBriefRule2Block:
    def _ws(self) -> WorldStateV1:
        return WorldStateV1(
            locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={})},
            objects={},
            entities={
                "ENT_ALICE": Entity(
                    id="ENT_ALICE", name="Alice", location_id="LOC_A",
                    status="healthy",
                    traits={"courage": TraitVector(value=0.5, inertia=0.2)},
                ),
            },
            events=[],
            causal_topology=[], spatial_topology=[], social_topology=[],
            channels={}, propositions=[], world_traits={},
        )

    def test_rule2_emits_soft_constraint_when_redundant_present(self):
        from shadow_loom.query_models import InterventionQuery
        ws = self._ws()
        query = InterventionQuery(interventions={"ENT_ALICE.courage": 0.9})
        brief = build_intervention_brief(
            query=query,
            world_state=ws,
            physics_state={"nodes": [], "links": []},
            rule3_pruned_interventions=None,
            rule2_redundant_evidence=["EVT_OFFSCREEN", "ENT_BOB.fear"],
        )
        rule2_blocks = [
            c for c in brief.constraints
            if "REDUNDANT EVIDENCE (Rule-2)" in (c.instruction or "")
        ]
        assert len(rule2_blocks) == 1, (
            "intervention brief must surface a Rule-2 redundant-evidence "
            "constraint (parity with build_counterfactual_brief)"
        )
        block = rule2_blocks[0]
        assert block.priority == "soft"
        assert "EVT_OFFSCREEN" in block.instruction
        assert block.evidence.get("rule2_redundant") == [
            "EVT_OFFSCREEN", "ENT_BOB.fear",
        ]

    def test_no_rule2_block_when_no_redundant_evidence(self):
        from shadow_loom.query_models import InterventionQuery
        ws = self._ws()
        query = InterventionQuery(interventions={"ENT_ALICE.courage": 0.9})
        brief = build_intervention_brief(
            query=query,
            world_state=ws,
            physics_state={"nodes": [], "links": []},
            rule3_pruned_interventions=None,
            rule2_redundant_evidence=None,
        )
        rule2_blocks = [
            c for c in brief.constraints
            if "REDUNDANT EVIDENCE (Rule-2)" in (c.instruction or "")
        ]
        assert rule2_blocks == []
