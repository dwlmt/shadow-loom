"""
Tests for the Auditor module (Steps 11–12).

Exercises:
  • Audit prompt assembly
  • Graph versioning (deep-copy isolation)
  • AuditResult / AuditViolation model construction
  • Feedback loop orchestration (mocked LLM)
  • Effect → audit category mapping
"""
import copy
import pytest
from unittest.mock import patch, MagicMock

import networkx as nx

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, NarrativeObject,
    CausalEdge, SpatialEdge, RelationshipEdge, InformationEdge,
    TraitVector, Affordance, Belief,
)
from shadow_loom.directive_assembly import (
    DirectiveAssembler, CreativeBrief, ConstraintBlock,
    RenderingDirective, ThreatProximity, CounterfactualBranch,
    CausalAttribution, EntanglementPair, InterventionMechanism,
    AbductionTruth, EpistemicGap, TraitTrajectory,
)
from shadow_loom.generation import GeneratedScene, GenerationConfig
from shadow_loom.auditor import (
    AuditResult, AuditViolation, AuditCycleSnapshot,
    AuditorConfig, FeedbackLoopResult, VersionedGraph,
    EFFECT_AUDIT_CATEGORIES,
    assemble_audit_prompt, _build_refinement_prompt,
    run_feedback_loop,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.query_models import DirectiveQuery

# ── Reuse plots ──
from tests.test_plot_models.macbeth import world_state as macbeth_ws
from tests.test_plot_models.gone_girl import world_state as gone_girl_ws


# =====================================================================
# Helper fixtures
# =====================================================================

def _ego_payload(ws, focus_ids):
    return extract_ego_graph_from_memory(ws, focus_ids).model_dump()


def _make_brief(
    effect="mystery",
    entities=None,
    constraints=None,
    epistemic_gaps=None,
    trait_trajectories=None,
    intervention_mechanisms=None,
    abduction_truths=None,
    threat_proximity=None,
    counterfactual_branch=None,
    causal_attribution=None,
    entanglement_pairs=None,
    rendering=None,
) -> CreativeBrief:
    """Build a minimal CreativeBrief for testing."""
    return CreativeBrief(
        target_effect=effect,
        target_entities=entities or ["ENT_TEST"],
        constraints=constraints or [
            ConstraintBlock(
                constraint_type="epistemic",
                priority="hard",
                instruction="Do NOT reveal the hidden cause.",
                evidence={"hidden_cause_ids": ["EVT_SECRET"]},
            ),
        ],
        epistemic_gaps=epistemic_gaps or [],
        trait_trajectories=trait_trajectories or [],
        intervention_mechanisms=intervention_mechanisms or [],
        abduction_truths=abduction_truths or [],
        threat_proximity=threat_proximity,
        counterfactual_branch=counterfactual_branch,
        causal_attribution=causal_attribution,
        entanglement_pairs=entanglement_pairs or [],
        rendering=rendering,
        scene_context={},
    )


def _make_scene(prose="The room was dark. She could not see.") -> GeneratedScene:
    return GeneratedScene(
        prose=prose,
        pov_entity="ENT_TEST",
        rendering_mode="mystery",
        constraints_honoured=["Hidden cause constraint"],
        constraints_violated=[],
    )


def _make_passing_audit() -> AuditResult:
    return AuditResult(
        passed=True,
        violations=[],
        audit_summary="All constraints honoured.",
    )


def _make_failing_audit() -> AuditResult:
    return AuditResult(
        passed=False,
        violations=[
            AuditViolation(
                violation_type="epistemic_leakage",
                severity="critical",
                description="The prose hints at the hidden cause.",
                evidence_quote="a shadow moved behind the curtain",
                feedback=(
                    "Epistemic Leakage Detected. Remove the shadow "
                    "reference — it points to the hidden ancestor."
                ),
            ),
        ],
        audit_summary="One critical violation found.",
    )


# =====================================================================
# EFFECT → AUDIT CATEGORY MAPPING
# =====================================================================

class TestEffectAuditCategories:
    """EFFECT_AUDIT_CATEGORIES must cover all directive effects."""

    def test_all_effects_mapped(self):
        expected = {
            "mystery", "dramatic_irony", "surprise",
            "suspense", "fear", "joy",
            "regret", "grief", "rage", "love",
            "observation", "intervention", "counterfactual",
        }
        assert expected <= set(EFFECT_AUDIT_CATEGORIES.keys())

    def test_epistemic_effects_have_epistemic_category(self):
        for eff in ("mystery", "dramatic_irony", "surprise"):
            assert "epistemic" in EFFECT_AUDIT_CATEGORIES[eff]

    def test_probabilistic_effects_have_probabilistic_category(self):
        for eff in ("suspense", "fear", "joy"):
            assert "probabilistic" in EFFECT_AUDIT_CATEGORIES[eff]

    def test_counterfactual_effects_have_counterfactual_category(self):
        for eff in ("regret", "grief", "rage", "love"):
            assert "counterfactual" in EFFECT_AUDIT_CATEGORIES[eff]

    def test_all_effects_include_physics(self):
        for cats in EFFECT_AUDIT_CATEGORIES.values():
            assert "physics" in cats


# =====================================================================
# AUDIT MODELS
# =====================================================================

class TestAuditModels:
    """AuditResult and AuditViolation models must validate correctly."""

    def test_passing_audit_result(self):
        ar = _make_passing_audit()
        assert ar.passed is True
        assert ar.violations == []

    def test_failing_audit_result(self):
        ar = _make_failing_audit()
        assert ar.passed is False
        assert len(ar.violations) == 1
        assert ar.violations[0].violation_type == "epistemic_leakage"
        assert ar.violations[0].severity == "critical"

    def test_all_violation_types_accepted(self):
        types = [
            "epistemic_leakage", "knowledge_contamination",
            "low_kl_divergence", "suspense_threshold",
            "tonal_mismatch", "magnitude_too_low",
            "reasoning_failure", "affective_failure",
            "attribution_failure", "empathy_weight",
            "miracle_step", "abduction_failure",
        ]
        for vt in types:
            v = AuditViolation(
                violation_type=vt,
                severity="major",
                description="test",
                feedback="fix it",
            )
            assert v.violation_type == vt

    def test_cycle_snapshot_structure(self):
        snap = AuditCycleSnapshot(
            iteration=0,
            prose="test prose",
            audit_result=_make_passing_audit(),
            graph_version=0,
            graph_data={},
        )
        assert snap.iteration == 0
        assert snap.graph_version == 0

    def test_feedback_loop_result_converged(self):
        result = FeedbackLoopResult(
            final_scene=_make_scene(),
            converged=True,
            iterations=1,
            history=[],
            final_graph_version=0,
        )
        assert result.converged is True
        assert result.iterations == 1


# =====================================================================
# GRAPH VERSIONING
# =====================================================================

class TestVersionedGraph:
    """VersionedGraph must deep-copy and isolate mutations."""

    def _make_sandbox(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        G.add_node("ENT_A", node_type="Entity", traits={"courage": {"value": 0.5, "inertia": 0.3}})
        G.add_node("ENT_B", node_type="Entity", traits={"fear": {"value": 0.2, "inertia": 0.4}})
        G.add_edge("ENT_A", "ENT_B", edge_type="relationship", affinity=0.8)
        return G

    def test_initial_version_is_zero(self):
        vg = VersionedGraph(self._make_sandbox())
        assert vg.version == 0

    def test_fork_increments_version(self):
        vg = VersionedGraph(self._make_sandbox())
        vg.fork()
        assert vg.version == 1
        vg.fork()
        assert vg.version == 2

    def test_original_not_mutated_after_fork(self):
        sandbox = self._make_sandbox()
        vg = VersionedGraph(sandbox)

        # Mutate the current copy
        vg.current.nodes["ENT_A"]["traits"]["courage"]["value"] = 0.99

        # Original should be unaffected
        assert vg.original.nodes["ENT_A"]["traits"]["courage"]["value"] == 0.5

    def test_fork_resets_to_original(self):
        sandbox = self._make_sandbox()
        vg = VersionedGraph(sandbox)

        # Mutate current
        vg.current.nodes["ENT_A"]["traits"]["courage"]["value"] = 0.99
        assert vg.current.nodes["ENT_A"]["traits"]["courage"]["value"] == 0.99

        # Fork resets
        forked = vg.fork()
        assert forked.nodes["ENT_A"]["traits"]["courage"]["value"] == 0.5

    def test_original_sandbox_not_mutated(self):
        """The original sandbox passed to __init__ must not be mutated."""
        sandbox = self._make_sandbox()
        original_val = sandbox.nodes["ENT_A"]["traits"]["courage"]["value"]

        vg = VersionedGraph(sandbox)
        vg.current.nodes["ENT_A"]["traits"]["courage"]["value"] = 0.99
        vg.fork()

        # The input sandbox should be untouched
        assert sandbox.nodes["ENT_A"]["traits"]["courage"]["value"] == original_val

    def test_snapshot_data_returns_dict(self):
        vg = VersionedGraph(self._make_sandbox())
        data = vg.snapshot_data()
        assert isinstance(data, dict)
        assert "nodes" in data or "directed" in data  # nx.node_link_data format


# =====================================================================
# AUDIT PROMPT ASSEMBLY
# =====================================================================

class TestAuditPromptAssembly:
    """assemble_audit_prompt must produce structured prompt text."""

    def test_basic_prompt_includes_prose(self):
        prose = "The room was dark."
        brief = _make_brief()
        prompt = assemble_audit_prompt(prose, brief, ["epistemic"])
        assert "The room was dark." in prompt

    def test_prompt_includes_target_effect(self):
        brief = _make_brief(effect="suspense")
        prompt = assemble_audit_prompt("test", brief, ["probabilistic"])
        assert "SUSPENSE" in prompt

    def test_prompt_includes_constraints(self):
        brief = _make_brief()
        prompt = assemble_audit_prompt("test", brief, ["epistemic"])
        assert "Do NOT reveal the hidden cause" in prompt

    def test_prompt_includes_audit_categories(self):
        prompt = assemble_audit_prompt(
            "test", _make_brief(), ["epistemic", "physics"],
        )
        assert "epistemic" in prompt
        assert "physics" in prompt

    def test_prompt_includes_prior_feedback(self):
        prompt = assemble_audit_prompt(
            "test",
            _make_brief(),
            ["epistemic"],
            prior_feedback=["Fix the shadow reference."],
        )
        assert "Fix the shadow reference." in prompt

    def test_prompt_includes_epistemic_gaps(self):
        brief = _make_brief(
            epistemic_gaps=[
                EpistemicGap(
                    entity_id="ENT_TEST",
                    belief_target_id="ENT_VILLAIN",
                    believed_state="friendly",
                    actual_state="hostile",
                    gap_type="contradicted",
                    gap_magnitude=0.9,
                ),
            ],
        )
        prompt = assemble_audit_prompt("test", brief, ["epistemic"])
        assert "ENT_VILLAIN" in prompt
        assert "friendly" in prompt

    def test_prompt_includes_intervention_mechanisms(self):
        brief = _make_brief(
            intervention_mechanisms=[
                InterventionMechanism(
                    node_id="OBJ_VAULT",
                    old_state="locked",
                    new_state="open",
                    mechanism_hint="physical force",
                    inertia=0.8,
                ),
            ],
        )
        prompt = assemble_audit_prompt("test", brief, ["physics"])
        assert "OBJ_VAULT" in prompt
        assert "locked" in prompt

    def test_prompt_includes_abduction_truths(self):
        brief = _make_brief(
            abduction_truths=[
                AbductionTruth(
                    entity_id="ENT_THIEF",
                    hidden_variable="stole the key yesterday",
                    weave_hint="reach into pocket",
                ),
            ],
        )
        prompt = assemble_audit_prompt("test", brief, ["physics"])
        assert "stole the key yesterday" in prompt

    def test_prompt_includes_threat_proximity(self):
        brief = _make_brief(
            effect="suspense",
            threat_proximity=ThreatProximity(
                threat_description="The assassin approaches.",
                threat_probability=0.8,
                hope_probability=0.3,
                spatial_distance=2,
            ),
        )
        prompt = assemble_audit_prompt("test", brief, ["probabilistic"])
        assert "assassin" in prompt
        assert "0.80" in prompt

    def test_prompt_includes_counterfactual_branch(self):
        brief = _make_brief(
            effect="regret",
            counterfactual_branch=CounterfactualBranch(
                actual_outcome="The child died.",
                simulated_outcome="If she had acted, the child would have lived.",
            ),
        )
        prompt = assemble_audit_prompt("test", brief, ["counterfactual"])
        assert "The child died." in prompt

    def test_prompt_includes_causal_attribution(self):
        brief = _make_brief(
            effect="rage",
            causal_attribution=CausalAttribution(
                perpetrator_id="ENT_VILLAIN",
                perpetrator_name="Lord Dark",
                loss_event_id="EVT_MURDER",
                loss_description="The murder of the king.",
            ),
        )
        prompt = assemble_audit_prompt("test", brief, ["counterfactual"])
        assert "Lord Dark" in prompt

    def test_prompt_includes_entanglement_pairs(self):
        brief = _make_brief(
            effect="love",
            entanglement_pairs=[
                EntanglementPair(
                    entity_a="ENT_ROMEO",
                    entity_b="ENT_JULIET",
                    coupling_strength=0.95,
                ),
            ],
        )
        prompt = assemble_audit_prompt("test", brief, ["counterfactual"])
        assert "ENT_ROMEO" in prompt
        assert "ENT_JULIET" in prompt


# =====================================================================
# REFINEMENT PROMPT
# =====================================================================

class TestRefinementPrompt:
    """_build_refinement_prompt must inject feedback into the rendering prompt."""

    def test_refinement_prompt_includes_original(self):
        violations = [_make_failing_audit().violations[0]]
        prompt = _build_refinement_prompt("ORIGINAL PROMPT", violations, 1)
        assert "ORIGINAL PROMPT" in prompt

    def test_refinement_prompt_includes_violation_feedback(self):
        violations = [_make_failing_audit().violations[0]]
        prompt = _build_refinement_prompt("ORIG", violations, 1)
        assert "Epistemic Leakage Detected" in prompt

    def test_refinement_prompt_includes_evidence_quote(self):
        violations = [_make_failing_audit().violations[0]]
        prompt = _build_refinement_prompt("ORIG", violations, 1)
        assert "a shadow moved behind the curtain" in prompt

    def test_refinement_prompt_includes_iteration_number(self):
        violations = [_make_failing_audit().violations[0]]
        prompt = _build_refinement_prompt("ORIG", violations, 3)
        assert "Iteration 3" in prompt

    def test_multiple_violations_all_included(self):
        v1 = AuditViolation(
            violation_type="epistemic_leakage",
            severity="critical",
            description="test1",
            feedback="Fix issue 1.",
        )
        v2 = AuditViolation(
            violation_type="miracle_step",
            severity="critical",
            description="test2",
            feedback="Fix issue 2.",
        )
        prompt = _build_refinement_prompt("ORIG", [v1, v2], 1)
        assert "Fix issue 1." in prompt
        assert "Fix issue 2." in prompt


# =====================================================================
# FEEDBACK LOOP (mocked LLM)
# =====================================================================

class TestFeedbackLoop:
    """run_feedback_loop must iterate audit → rewrite until convergence."""

    def _mock_run_sync(self, output):
        """Create a mock that returns a pydantic_ai-style result."""
        mock_result = MagicMock()
        mock_result.output = output
        return mock_result

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_immediate_pass_returns_iteration_1(self, mock_gen_agent, mock_run_audit):
        """If the first audit passes, loop returns after 1 iteration."""
        mock_run_audit.return_value = _make_passing_audit()

        result = run_feedback_loop(
            initial_scene=_make_scene(),
            brief=_make_brief(),
            auditor_config=AuditorConfig(max_iterations=3),
        )

        assert result.converged is True
        assert result.iterations == 1
        assert len(result.history) == 1
        assert result.history[0].audit_result.passed is True
        # Generation agent should NOT have been called (no rewrite needed)
        mock_gen_agent.assert_not_called()

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_fail_then_pass_returns_iteration_2(self, mock_gen_agent, mock_run_audit):
        """Fail on first audit, pass on second → 2 iterations."""
        mock_run_audit.side_effect = [
            _make_failing_audit(),
            _make_passing_audit(),
        ]
        # Mock the generation agent for the rewrite
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.return_value = self._mock_run_sync(
            _make_scene("Rewritten prose that passes.")
        )
        mock_gen_agent.return_value = mock_agent_instance

        result = run_feedback_loop(
            initial_scene=_make_scene(),
            brief=_make_brief(),
            auditor_config=AuditorConfig(max_iterations=3),
        )

        assert result.converged is True
        assert result.iterations == 2
        assert len(result.history) == 2
        assert result.history[0].audit_result.passed is False
        assert result.history[1].audit_result.passed is True

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_exhaust_iterations(self, mock_gen_agent, mock_run_audit):
        """If all audits fail, loop exhausts max_iterations."""
        mock_run_audit.return_value = _make_failing_audit()
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.return_value = self._mock_run_sync(
            _make_scene("Still failing prose.")
        )
        mock_gen_agent.return_value = mock_agent_instance

        result = run_feedback_loop(
            initial_scene=_make_scene(),
            brief=_make_brief(),
            auditor_config=AuditorConfig(max_iterations=2),
        )

        assert result.converged is False
        assert result.iterations == 2

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_graph_versioning_in_loop(self, mock_gen_agent, mock_run_audit):
        """Graph must be versioned and forked across iterations."""
        mock_run_audit.side_effect = [
            _make_failing_audit(),
            _make_passing_audit(),
        ]
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.return_value = self._mock_run_sync(
            _make_scene("Rewritten.")
        )
        mock_gen_agent.return_value = mock_agent_instance

        sandbox = nx.MultiDiGraph()
        sandbox.add_node("ENT_A", node_type="Entity", traits={"fear": {"value": 0.5}})

        result = run_feedback_loop(
            initial_scene=_make_scene(),
            brief=_make_brief(),
            sandbox=sandbox,
            auditor_config=AuditorConfig(max_iterations=3),
        )

        assert result.converged is True
        # Graph version should be at least 1 after one fork
        assert result.final_graph_version >= 1
        # History should have graph snapshots
        assert result.history[0].graph_version == 0

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_original_sandbox_not_mutated(self, mock_gen_agent, mock_run_audit):
        """The original sandbox passed to run_feedback_loop must not be mutated."""
        mock_run_audit.side_effect = [
            _make_failing_audit(),
            _make_passing_audit(),
        ]
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.return_value = self._mock_run_sync(
            _make_scene("Rewritten.")
        )
        mock_gen_agent.return_value = mock_agent_instance

        sandbox = nx.MultiDiGraph()
        sandbox.add_node("ENT_A", node_type="Entity", traits={"fear": {"value": 0.5}})
        original_val = sandbox.nodes["ENT_A"]["traits"]["fear"]["value"]

        run_feedback_loop(
            initial_scene=_make_scene(),
            brief=_make_brief(),
            sandbox=sandbox,
            auditor_config=AuditorConfig(max_iterations=3),
        )

        # Original must be untouched
        assert sandbox.nodes["ENT_A"]["traits"]["fear"]["value"] == original_val

    @patch("shadow_loom.auditor.run_audit")
    @patch("shadow_loom.auditor._build_generation_agent")
    def test_accumulated_feedback_passed_to_auditor(self, mock_gen_agent, mock_run_audit):
        """Prior feedback must accumulate across iterations."""
        call_args = []

        def capture_audit(prose, brief, config=None, prior_feedback=None):
            call_args.append(prior_feedback)
            if len(call_args) < 3:
                return _make_failing_audit()
            return _make_passing_audit()

        mock_run_audit.side_effect = capture_audit
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.return_value = self._mock_run_sync(
            _make_scene("Rewritten.")
        )
        mock_gen_agent.return_value = mock_agent_instance

        run_feedback_loop(
            initial_scene=_make_scene(),
            brief=_make_brief(),
            auditor_config=AuditorConfig(max_iterations=5),
        )

        # First call: no prior feedback
        assert call_args[0] is None
        # Second call: has feedback from first failure
        assert call_args[1] is not None
        assert len(call_args[1]) >= 1
        # Third call: accumulated feedback from both failures
        assert call_args[2] is not None
        assert len(call_args[2]) >= 2


# =====================================================================
# INTEGRATION: Audit prompt from real world state
# =====================================================================

class TestAuditPromptFromWorldState:
    """Assemble audit prompts from real plot model data."""

    def test_macbeth_mystery_audit_prompt(self):
        """Mystery audit prompt for Macbeth must reference hidden causes."""
        ego = _ego_payload(macbeth_ws, ["ENT_MACBETH"])
        assembler = DirectiveAssembler(None, ego, macbeth_ws)
        directive = DirectiveQuery(
            target_entity_ids=["ENT_MACBETH"],
            target_effect="mystery",
            intensity=0.8,
        )
        brief = assembler.assemble(directive, syuzhet_anchor=3)

        prompt = assemble_audit_prompt(
            "Macbeth stood before the body. Blood pooled beneath the king.",
            brief,
            ["epistemic", "physics"],
        )

        assert "MYSTERY" in prompt
        assert "ENT_MACBETH" in prompt
        assert "CONSTRAINTS" in prompt

    def test_gone_girl_dramatic_irony_audit_prompt(self):
        """Dramatic irony audit for Gone Girl must reference reader/character asymmetry."""
        ego = _ego_payload(gone_girl_ws, ["ENT_NICK"])
        assembler = DirectiveAssembler(None, ego, gone_girl_ws)
        directive = DirectiveQuery(
            target_entity_ids=["ENT_NICK"],
            target_effect="dramatic_irony",
            intensity=0.9,
        )
        brief = assembler.assemble(directive, syuzhet_anchor=5)

        prompt = assemble_audit_prompt(
            "Nick smiled at the cameras. Everything was going to be fine.",
            brief,
            ["epistemic", "physics"],
        )

        assert "DRAMATIC_IRONY" in prompt
        assert "ENT_NICK" in prompt


# =====================================================================
# AuditResult.failed_open flag
# =====================================================================

class TestAuditResultFailedOpen:
    """Tests for the fail-open behaviour marker."""

    def test_default_is_false(self):
        """Normal audit results have failed_open=False."""
        result = AuditResult(
            passed=True,
            violations=[],
            audit_summary="All good.",
        )
        assert result.failed_open is False

    def test_explicit_failed_open(self):
        """Error-fallback results have failed_open=True."""
        result = AuditResult(
            passed=True,
            violations=[],
            audit_summary="Audit skipped due to LLM error: timeout",
            failed_open=True,
        )
        assert result.failed_open is True
        assert result.passed is True  # fail-open = pass through

    def test_failed_open_distinguishable_from_real_pass(self):
        """Callers can detect that a pass-through is not a genuine pass."""
        real = AuditResult(passed=True, violations=[], audit_summary="Clean.")
        fallback = AuditResult(
            passed=True, violations=[],
            audit_summary="Audit skipped due to LLM error: conn refused",
            failed_open=True,
        )
        assert real.passed == fallback.passed
        assert real.failed_open != fallback.failed_open
