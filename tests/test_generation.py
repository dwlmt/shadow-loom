# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shadow_loom.generation helper functions.

Covers:
  - _format_scene_context (current_locations key fix)
  - _resolve_model branching
  - Brief builders for observation/intervention/counterfactual
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from shadow_loom.generation import (
    _format_scene_context,
    _resolve_model,
    build_observation_brief,
    build_intervention_brief,
    build_counterfactual_brief,
    assemble_rendering_prompt,
)
from shadow_loom.directive_assembly import CreativeBrief
from shadow_loom.query_models import (
    ObservationQuery,
    InterventionQuery,
    CounterfactualQuery,
)
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from example_worlds.macbeth import world_state as macbeth_ws


# =====================================================================
# _format_scene_context
# =====================================================================

class TestFormatSceneContext:
    """Test the ego-graph → text formatter."""

    def test_empty_context(self):
        result = _format_scene_context({})
        assert "No scene context" in result

    def test_focus_entities_rendered(self):
        ctx = {
            "focus_entities": [
                {
                    "id": "ENT_ALICE",
                    "name": "Alice",
                    "location_id": "LOC_ROOM",
                    "status": "healthy",
                    "traits": {"courage": {"value": 0.9, "inertia": 0.5}},
                },
            ],
        }
        result = _format_scene_context(ctx)
        assert "Alice" in result
        assert "ENT_ALICE" in result
        assert "courage=0.90" in result

    def test_present_entities_rendered(self):
        ctx = {
            "present_entities": [
                {"id": "ENT_BOB", "name": "Bob", "location_id": "LOC_ROOM"},
            ],
        }
        result = _format_scene_context(ctx)
        assert "Bob" in result
        assert "present at" in result

    def test_current_locations_key(self):
        """The ego-graph payload uses 'current_locations', not 'relevant_locations'."""
        ctx = {
            "current_locations": [
                {"id": "LOC_CASTLE", "name": "Dunsinane Castle"},
            ],
        }
        result = _format_scene_context(ctx)
        assert "Dunsinane Castle" in result
        assert "LOC_CASTLE" in result

    def test_relevant_locations_legacy_fallback(self):
        """Legacy 'relevant_locations' key still works as fallback."""
        ctx = {
            "relevant_locations": [
                {"id": "LOC_OLD", "name": "Old Key"},
            ],
        }
        result = _format_scene_context(ctx)
        assert "Old Key" in result

    def test_recent_memory_rendered(self):
        ctx = {
            "recent_memory": [
                {"id": "EVT_FIGHT", "description": "A fight broke out"},
            ],
        }
        result = _format_scene_context(ctx)
        assert "EVT_FIGHT" in result
        assert "fight broke out" in result

    def test_traits_near_baseline_omitted(self):
        """Traits close to 0.5 (within 0.15) should not appear."""
        ctx = {
            "focus_entities": [
                {
                    "id": "ENT_X",
                    "name": "X",
                    "location_id": "LOC_A",
                    "status": "healthy",
                    "traits": {"courage": {"value": 0.55, "inertia": 0.5}},
                },
            ],
        }
        result = _format_scene_context(ctx)
        assert "courage" not in result
        assert "baseline" in result

    def test_real_ego_graph_payload(self):
        """Full integration: real ego graph payload → format."""
        ego = extract_ego_graph_from_memory(macbeth_ws, ["ENT_MACBETH"])
        ctx = ego.model_dump()
        result = _format_scene_context(ctx)
        assert "Macbeth" in result
        # current_locations should be picked up
        assert "LOC_" in result or "Location:" in result


# =====================================================================
# _resolve_model
# =====================================================================

class TestResolveModel:
    """Test model string resolution."""

    def test_ollama_prefix(self):
        """ollama: prefix should return an OllamaModel instance."""
        model = _resolve_model("ollama:qwen3:8b")
        assert model is not None
        assert not isinstance(model, str)

    def test_non_ollama_passthrough(self):
        """Non-ollama strings without recognized prefix are returned as-is."""
        model = _resolve_model("anthropic:claude-3-sonnet")
        assert model == "anthropic:claude-3-sonnet"

    def test_non_ollama_plain_string(self):
        model = _resolve_model("test-model")
        assert model == "test-model"

    def test_openai_prefix_requires_key(self):
        """openai: prefix should raise ValueError without an API key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _resolve_model("openai:gpt-4")

    def test_openrouter_prefix_requires_key(self):
        """openrouter: prefix should raise ValueError without an API key."""
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            _resolve_model("openrouter:google/gemini-2.0-flash")


# =====================================================================
# Brief Builders
# =====================================================================

class TestBuildObservationBrief:
    """Observation brief builder."""

    def test_basic_observation_brief(self):
        query = ObservationQuery(
            observations={},
            focus_entity_ids=["ENT_MACBETH"],
        )
        ego = extract_ego_graph_from_memory(macbeth_ws, ["ENT_MACBETH"])
        brief = build_observation_brief(query, ego.model_dump(), macbeth_ws)
        assert isinstance(brief, CreativeBrief)
        assert brief.target_effect == "observation"
        assert "ENT_MACBETH" in brief.target_entities

    def test_observation_with_observations(self):
        query = ObservationQuery(
            observations={"ENT_MACBETH": "looks nervous"},
            focus_entity_ids=["ENT_MACBETH"],
        )
        ego = extract_ego_graph_from_memory(macbeth_ws, ["ENT_MACBETH"])
        brief = build_observation_brief(query, ego.model_dump(), macbeth_ws)
        # Observation briefs are lightweight — no constraints added
        assert brief.target_effect == "observation"
        assert "ENT_MACBETH" in brief.target_entities


class TestBuildInterventionBrief:
    """Intervention brief builder."""

    def test_basic_intervention_brief(self):
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "dead"},
        )
        brief = build_intervention_brief(query, {}, macbeth_ws)
        assert isinstance(brief, CreativeBrief)
        assert brief.target_effect == "intervention"
        assert brief.intervention_mechanisms
        assert brief.intervention_mechanisms[0].node_id == "ENT_MACBETH"

    def test_intervention_trait_mechanism(self):
        query = InterventionQuery(
            interventions={"ENT_MACBETH.traits.guilt": 1.0},
        )
        brief = build_intervention_brief(query, {}, macbeth_ws)
        mech = brief.intervention_mechanisms[0]
        assert "psychological" in mech.mechanism_hint

    def test_intervention_with_blocked(self):
        query = InterventionQuery(
            interventions={"ENT_MACBETH.status": "dead"},
        )
        blocked = [{"node_id": "ENT_MACBETH", "trait": "status", "reason": "inertia"}]
        brief = build_intervention_brief(query, {}, macbeth_ws, blocked=blocked)
        block_constraints = [c for c in brief.constraints if "BLOCKED" in c.instruction]
        assert block_constraints


class TestBuildCounterfactualBrief:
    """Counterfactual brief builder."""

    def test_basic_counterfactual_brief(self):
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=["ENT_MACBETH"],
        )
        brief = build_counterfactual_brief(query, {}, macbeth_ws)
        assert isinstance(brief, CreativeBrief)
        assert brief.target_effect == "counterfactual"
        assert brief.counterfactual_branch is not None

    def test_counterfactual_with_hidden_deltas(self):
        query = CounterfactualQuery(
            historical_interventions={"EVT_DUNCAN_MURDER.event_type": "prevented"},
            evidence_node_ids=["ENT_MACBETH"],
        )
        deltas = {"ENT_MACBETH": {"guilt": -0.3}}
        brief = build_counterfactual_brief(query, {}, macbeth_ws, hidden_deltas=deltas)
        assert brief.abduction_truths
        assert brief.abduction_truths[0].entity_id == "ENT_MACBETH"


# =====================================================================
# assemble_rendering_prompt
# =====================================================================

class TestAssembleRenderingPrompt:
    """Test prompt assembly from a brief."""

    def test_basic_assembly(self):
        brief = CreativeBrief(
            target_effect="suspense",
            target_entities=["ENT_MACBETH"],
            scene_context={"focus_entities": []},
        )
        prompt = assemble_rendering_prompt(brief, "directive")
        assert isinstance(prompt, str)
        assert "suspense" in prompt.lower() or "SUSPENSE" in prompt

    def test_includes_constraints(self):
        from shadow_loom.directive_assembly import ConstraintBlock
        brief = CreativeBrief(
            target_effect="fear",
            target_entities=["ENT_MACBETH"],
            constraints=[
                ConstraintBlock(
                    constraint_type="mathematical",
                    priority="hard",
                    instruction="Increase fear to 0.9",
                    evidence={},
                ),
            ],
            scene_context={},
        )
        prompt = assemble_rendering_prompt(brief, "directive")
        assert "fear" in prompt.lower()
