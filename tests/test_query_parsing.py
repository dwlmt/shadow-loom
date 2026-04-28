"""Unit tests for shadow_loom.query_parsing.

Tests cover:
  - Graph summary building
  - ID collection
  - Validation logic for each query type
  - Query construction from ParsedQuery
  - Integration with mocked LLM agent
  - Fallback logic (fuzzy ID resolution + general-query fallback)
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from shadow_loom.query_parsing import (
    FallbackInfo,
    ParsedQuery,
    QueryParseResult,
    QueryParsingConfig,
    QUERY_TYPES,
    ResolvedID,
    ValidationError,
    _apply_fallback,
    _build_general_fallback,
    _build_graph_summary,
    _build_name_index,
    _build_query,
    _build_typed_system_prompt,
    _collect_all_ids,
    _fuzzy_resolve_id,
    _normalise_id_name,
    _try_fuzzy_repair,
    _validate_parsed_query,
    parse_query,
)
from shadow_loom.query_models import (
    CounterfactualQuery,
    DirectiveQuery,
    GeneralQuery,
    InterrogationQuery,
    InterventionQuery,
    ObservationQuery,
)
from example_worlds.macbeth import world_state as macbeth_ws


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def macbeth():
    return macbeth_ws


# =====================================================================
# _build_graph_summary
# =====================================================================

class TestBuildGraphSummary:
    def test_contains_entity_ids(self, macbeth):
        summary = _build_graph_summary(macbeth)
        assert "ENT_MACBETH" in summary
        assert "ENT_LADY_MACBETH" in summary

    def test_contains_location_ids(self, macbeth):
        summary = _build_graph_summary(macbeth)
        assert "LOC_" in summary

    def test_contains_event_ids(self, macbeth):
        summary = _build_graph_summary(macbeth)
        assert "EVT_" in summary

    def test_contains_object_ids(self, macbeth):
        summary = _build_graph_summary(macbeth)
        assert "OBJ_" in summary

    def test_contains_relationships(self, macbeth):
        summary = _build_graph_summary(macbeth)
        if macbeth.social_topology:
            assert "RELATIONSHIPS:" in summary

    def test_events_sorted_chronologically(self, macbeth):
        summary = _build_graph_summary(macbeth)
        lines = [l for l in summary.split("\n") if l.strip().startswith("EVT_")]
        # Extract fabula_times from summary lines
        times = []
        for line in lines:
            if "t=" in line:
                t_str = line.split("t=")[1].split()[0]
                times.append(int(t_str))
        assert times == sorted(times), "Events should be in chronological order"


# =====================================================================
# _collect_all_ids
# =====================================================================

class TestCollectAllIds:
    def test_entities_included(self, macbeth):
        ids = _collect_all_ids(macbeth)
        assert "ENT_MACBETH" in ids

    def test_locations_included(self, macbeth):
        ids = _collect_all_ids(macbeth)
        for lid in macbeth.locations:
            assert lid in ids

    def test_objects_included(self, macbeth):
        ids = _collect_all_ids(macbeth)
        for oid in macbeth.objects:
            assert oid in ids

    def test_events_included(self, macbeth):
        ids = _collect_all_ids(macbeth)
        for evt in macbeth.events:
            assert evt.id in ids


# =====================================================================
# _validate_parsed_query
# =====================================================================

class TestValidation:
    """Validation logic for each query type."""

    def test_observation_valid(self, macbeth):
        parsed = ParsedQuery(
            query_type="observation",
            reasoning="observe",
            focus_entity_ids=["ENT_MACBETH"],
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert not errors

    def test_observation_invalid_entity(self, macbeth):
        parsed = ParsedQuery(
            query_type="observation",
            reasoning="observe",
            focus_entity_ids=["ENT_NONEXISTENT"],
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("ENT_NONEXISTENT" in e.message for e in errors)

    def test_observation_no_world_state_skips_id_check(self):
        parsed = ParsedQuery(
            query_type="observation",
            reasoning="observe",
            focus_entity_ids=["ENT_ANYTHING"],
        )
        errors = _validate_parsed_query(parsed, None)
        assert not errors

    def test_intervention_valid(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="intervene",
            interventions={"ENT_MACBETH": "dead"},
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert not errors

    def test_intervention_empty(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="intervene",
            interventions={},
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("at least one" in e.message for e in errors)

    def test_intervention_missing(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="intervene",
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("at least one" in e.message for e in errors)

    def test_intervention_bad_id(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="intervene",
            interventions={"ENT_NOBODY": "dead"},
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("ENT_NOBODY" in e.message for e in errors)

    def test_counterfactual_valid(self, macbeth):
        evt_id = macbeth.events[0].id
        ent_id = next(iter(macbeth.entities))
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="what if",
            historical_interventions={evt_id: "different outcome"},
            evidence_node_ids=[ent_id],
        )
        errors = _validate_parsed_query(parsed, macbeth)
        hard_errors = [e for e in errors if e.severity == "error"]
        assert not hard_errors

    def test_counterfactual_missing_interventions(self, macbeth):
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="what if",
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("historical_interventions" in e.field for e in errors)

    def test_counterfactual_no_evidence_is_warning(self, macbeth):
        evt_id = macbeth.events[0].id
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="what if",
            historical_interventions={evt_id: "changed"},
        )
        errors = _validate_parsed_query(parsed, macbeth)
        warnings = [e for e in errors if e.severity == "warning"]
        assert any("evidence" in e.message.lower() for e in warnings)

    def test_directive_valid(self, macbeth):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="maximise suspense",
            target_entity_ids=["ENT_MACBETH"],
            target_effect="suspense",
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert not errors

    def test_directive_missing_entities(self, macbeth):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="grief",
            target_effect="grief",
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("target_entity_ids" in e.field for e in errors)

    def test_directive_missing_effect(self, macbeth):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="directive",
            target_entity_ids=["ENT_MACBETH"],
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("target_effect" in e.field for e in errors)

    def test_directive_bad_intensity(self, macbeth):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="directive",
            target_entity_ids=["ENT_MACBETH"],
            target_effect="fear",
            intensity=1.5,
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("intensity" in e.field.lower() or "intensity" in e.message.lower() for e in errors)

    def test_interrogate_valid(self):
        parsed = ParsedQuery(
            query_type="interrogate",
            reasoning="pathfinding",
            question="Can Macbeth reach the courtyard?",
        )
        errors = _validate_parsed_query(parsed, None)
        assert not errors

    def test_interrogate_missing_question(self):
        parsed = ParsedQuery(
            query_type="interrogate",
            reasoning="pathfinding",
        )
        errors = _validate_parsed_query(parsed, None)
        assert any("question" in e.field for e in errors)

    def test_general_valid(self):
        parsed = ParsedQuery(
            query_type="general",
            reasoning="broad question",
            question="What are all the relationships?",
        )
        errors = _validate_parsed_query(parsed, None)
        assert not errors

    def test_general_missing_question(self):
        parsed = ParsedQuery(
            query_type="general",
            reasoning="broad",
        )
        errors = _validate_parsed_query(parsed, None)
        assert any("question" in e.field for e in errors)

    def test_resolved_ids_validated(self, macbeth):
        parsed = ParsedQuery(
            query_type="general",
            reasoning="test",
            question="Who is Macbeth?",
            resolved_ids=[
                ResolvedID(natural_name="Macbeth", resolved_id="ENT_MACBETH"),
                ResolvedID(natural_name="Ghost", resolved_id="ENT_GHOST_FAKE"),
            ],
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("ENT_GHOST_FAKE" in e.message for e in errors)


# =====================================================================
# _build_query
# =====================================================================

class TestBuildQuery:
    def test_observation(self):
        parsed = ParsedQuery(
            query_type="observation",
            reasoning="test",
            observations={"OBJ_CUP": "empty"},
            focus_entity_ids=["ENT_MACBETH"],
        )
        q = _build_query(parsed)
        assert isinstance(q, ObservationQuery)
        assert q.observations == {"OBJ_CUP": "empty"}
        assert q.focus_entity_ids == ["ENT_MACBETH"]

    def test_intervention(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_MACBETH": "dead"},
        )
        q = _build_query(parsed)
        assert isinstance(q, InterventionQuery)
        # Bare ENT_ + str value is normalised to .status
        assert q.interventions == {"ENT_MACBETH.status": "dead"}

    def test_intervention_preserves_dotted_keys(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_MACBETH.location_id": "LOC_HEATH"},
        )
        q = _build_query(parsed)
        assert q.interventions == {"ENT_MACBETH.location_id": "LOC_HEATH"}

    def test_counterfactual(self):
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="test",
            historical_interventions={"EVT_MURDER": "prevented"},
            evidence_node_ids=["ENT_DUNCAN"],
        )
        q = _build_query(parsed)
        assert isinstance(q, CounterfactualQuery)
        # Bare EVT_ key is normalised to .event_type
        assert q.historical_interventions == {"EVT_MURDER.event_type": "prevented"}
        assert q.evidence_node_ids == ["ENT_DUNCAN"]

    def test_counterfactual_dict_valued_bare_evt_left_unnormalised(self):
        # A dict value on a bare EVT_* key in historical_interventions is
        # ambiguous (alter vs. spawn) and previously got normalised to
        # ``.event_type`` with a dict payload, which would silently
        # corrupt event_type. The normaliser now leaves such keys bare
        # so the downstream layer can reject them cleanly.
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="test",
            historical_interventions={
                "EVT_MURDER": {"new_outcome": "Duncan survives"}
            },
            evidence_node_ids=["ENT_DUNCAN"],
        )
        q = _build_query(parsed)
        # Key is preserved verbatim — no dotted suffix added.
        assert "EVT_MURDER" in q.historical_interventions
        assert "EVT_MURDER.event_type" not in q.historical_interventions

    def test_directive(self):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="test",
            target_entity_ids=["ENT_MACBETH"],
            target_effect="grief",
            target_vector_id="EVT_SOMETHING",
            intensity=0.8,
        )
        q = _build_query(parsed)
        assert isinstance(q, DirectiveQuery)
        assert q.target_effect == "grief"
        assert q.intensity == 0.8

    def test_directive_defaults(self):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="test",
            target_entity_ids=["ENT_MACBETH"],
            target_effect="suspense",
        )
        q = _build_query(parsed)
        assert isinstance(q, DirectiveQuery)
        assert q.intensity == 1.0
        assert q.target_vector_id is None

    def test_interrogate(self):
        parsed = ParsedQuery(
            query_type="interrogate",
            reasoning="test",
            question="Who killed Duncan?",
            require_proof=False,
        )
        q = _build_query(parsed)
        assert isinstance(q, InterrogationQuery)
        assert q.question == "Who killed Duncan?"
        assert q.require_proof is False

    def test_interrogate_defaults(self):
        parsed = ParsedQuery(
            query_type="interrogate",
            reasoning="test",
            question="path?",
        )
        q = _build_query(parsed)
        assert isinstance(q, InterrogationQuery)
        assert q.require_proof is True

    def test_general(self):
        parsed = ParsedQuery(
            query_type="general",
            reasoning="test",
            question="Tell me about the world",
            include_topology=False,
        )
        q = _build_query(parsed)
        assert isinstance(q, GeneralQuery)
        assert q.include_topology is False

    def test_general_defaults(self):
        parsed = ParsedQuery(
            query_type="general",
            reasoning="test",
            question="What is going on?",
        )
        q = _build_query(parsed)
        assert isinstance(q, GeneralQuery)
        assert q.include_topology is True

    def test_intervention_numeric_value(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_MACBETH": 0.99},
        )
        q = _build_query(parsed)
        assert isinstance(q, InterventionQuery)
        # Bare ENT_ + numeric value can't be safely auto-routed; left as-is.
        assert q.interventions["ENT_MACBETH"] == 0.99

    def test_intervention_dict_value(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_NEW": {"name": "Ghost", "type": "entity"}},
        )
        q = _build_query(parsed)
        assert isinstance(q, InterventionQuery)
        # Bare ENT_ + dict value is normalised to .spawn (genesis).
        assert isinstance(q.interventions["ENT_NEW.spawn"], dict)


# =====================================================================
# parse_query with mocked LLM
# =====================================================================

def _make_mock_result(parsed: ParsedQuery):
    """Build a mock agent result wrapping a ParsedQuery."""
    mock = MagicMock()
    mock.output = parsed
    return mock


class TestParseQueryMocked:
    """Test parse_query with a mocked LLM agent."""

    def _run(self, nl: str, parsed: ParsedQuery, world_state=None):
        """Helper: mock the agent and call parse_query."""
        mock_result = _make_mock_result(parsed)
        with patch("shadow_loom.query_parsing.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_sync.return_value = mock_result
            return parse_query(nl, world_state=world_state)

    def test_observation_from_natural_language(self, macbeth):
        parsed = ParsedQuery(
            query_type="observation",
            reasoning="User wants to see what happens next from Macbeth's POV.",
            focus_entity_ids=["ENT_MACBETH"],
            resolved_ids=[ResolvedID(natural_name="Macbeth", resolved_id="ENT_MACBETH")],
        )
        result = self._run(
            "Show me the scene from Macbeth's perspective",
            parsed,
            world_state=macbeth,
        )
        assert result.is_valid
        assert result.query is not None
        assert isinstance(result.query, ObservationQuery)
        assert result.query.focus_entity_ids == ["ENT_MACBETH"]

    def test_intervention_from_natural_language(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="User wants to kill Macbeth.",
            interventions={"ENT_MACBETH": "dead"},
            resolved_ids=[ResolvedID(natural_name="Macbeth", resolved_id="ENT_MACBETH")],
        )
        result = self._run("Kill Macbeth", parsed, world_state=macbeth)
        assert result.is_valid
        assert isinstance(result.query, InterventionQuery)

    def test_counterfactual_from_natural_language(self, macbeth):
        # Find an actual event and entity ID from the fixture
        evt_id = macbeth.events[0].id
        ent_id = next(iter(macbeth.entities))
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="What-if about past event.",
            historical_interventions={evt_id: "never happened"},
            evidence_node_ids=[ent_id],
        )
        result = self._run(
            "What if Duncan was never murdered?",
            parsed,
            world_state=macbeth,
        )
        assert result.is_valid
        assert isinstance(result.query, CounterfactualQuery)

    def test_directive_from_natural_language(self, macbeth):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="Maximise suspense for Macbeth.",
            target_entity_ids=["ENT_MACBETH"],
            target_effect="suspense",
            intensity=0.9,
        )
        result = self._run(
            "Maximise the suspense for Macbeth",
            parsed,
            world_state=macbeth,
        )
        assert result.is_valid
        assert isinstance(result.query, DirectiveQuery)
        assert result.query.target_effect == "suspense"

    def test_interrogate_from_natural_language(self):
        parsed = ParsedQuery(
            query_type="interrogate",
            reasoning="Pathfinding question.",
            question="Is there a path from Inverness to Dunsinane?",
        )
        result = self._run(
            "Can someone travel from Inverness Castle to Dunsinane?",
            parsed,
        )
        assert result.is_valid
        assert isinstance(result.query, InterrogationQuery)

    def test_general_from_natural_language(self):
        parsed = ParsedQuery(
            query_type="general",
            reasoning="Broad question.",
            question="What are all the relationships in the story?",
        )
        result = self._run("Tell me about all the relationships", parsed)
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)

    def test_invalid_id_triggers_fallback(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="Bad ID test.",
            interventions={"ENT_DOES_NOT_EXIST": "dead"},
        )
        result = self._run("Kill the ghost", parsed, world_state=macbeth)
        # Fallback kicks in — the result is valid but downgraded to general
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)
        assert result.fallback is not None
        assert result.fallback.strategy == "general_fallback"
        assert result.fallback.original_query_type == "intervention"

    def test_no_world_state_skips_id_validation(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="No world state.",
            interventions={"ENT_ANYONE": "alive"},
        )
        result = self._run("Revive someone", parsed, world_state=None)
        assert result.is_valid
        assert isinstance(result.query, InterventionQuery)

    def test_world_model_summary_passed_to_agent(self, macbeth):
        parsed = ParsedQuery(
            query_type="general",
            reasoning="test",
            question="Who is Macbeth?",
        )
        mock_result = _make_mock_result(parsed)
        with patch("shadow_loom.query_parsing.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_sync.return_value = mock_result
            parse_query("Who is Macbeth?", world_state=macbeth)
            # Check that the user message contained the world model summary
            call_args = instance.run_sync.call_args
            user_msg = call_args[0][0]
            assert "WORLD MODEL" in user_msg
            assert "ENT_MACBETH" in user_msg

    def test_warnings_still_produce_valid_query(self, macbeth):
        """Counterfactual with missing evidence is a warning, not an error."""
        evt_id = macbeth.events[0].id
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="what if",
            historical_interventions={evt_id: "changed"},
            # No evidence_node_ids — this is a warning
        )
        result = self._run("What if things were different?", parsed, world_state=macbeth)
        assert result.is_valid  # warnings don't block
        assert isinstance(result.query, CounterfactualQuery)
        assert len(result.validation_errors) > 0  # but warning is recorded
        assert all(e.severity == "warning" for e in result.validation_errors)


# =====================================================================
# QueryParseResult model
# =====================================================================

class TestQueryParseResult:
    def test_valid_result(self):
        parsed = ParsedQuery(query_type="general", reasoning="test", question="hello")
        result = QueryParseResult(
            query=GeneralQuery(question="hello"),
            parsed=parsed,
            is_valid=True,
        )
        assert result.is_valid
        assert result.query is not None

    def test_invalid_result(self):
        parsed = ParsedQuery(query_type="intervention", reasoning="test")
        result = QueryParseResult(
            query=None,
            parsed=parsed,
            validation_errors=[
                ValidationError(field="interventions", message="missing")
            ],
            is_valid=False,
        )
        assert not result.is_valid
        assert result.query is None


# =====================================================================
# Fuzzy ID helpers
# =====================================================================

class TestNormaliseIdName:
    def test_strips_ent_prefix(self):
        assert _normalise_id_name("ENT_MACBETH") == "MACBETH"

    def test_strips_evt_prefix(self):
        assert _normalise_id_name("EVT_DUNCAN_MURDER") == "DUNCAN_MURDER"

    def test_strips_obj_prefix(self):
        assert _normalise_id_name("OBJ_DAGGER") == "DAGGER"

    def test_strips_loc_prefix(self):
        assert _normalise_id_name("LOC_CASTLE") == "CASTLE"

    def test_collapses_non_alphanum(self):
        assert _normalise_id_name("Lady Macbeth") == "LADY_MACBETH"

    def test_strips_leading_trailing_underscores(self):
        assert _normalise_id_name(" _macbeth_ ") == "MACBETH"


class TestBuildNameIndex:
    def test_entity_ids_indexed(self, macbeth):
        index = _build_name_index(macbeth)
        assert "MACBETH" in index
        assert index["MACBETH"] == "ENT_MACBETH"

    def test_entity_names_indexed(self, macbeth):
        index = _build_name_index(macbeth)
        ent = macbeth.entities["ENT_MACBETH"]
        normalised = _normalise_id_name(ent.name)
        assert normalised in index

    def test_location_ids_indexed(self, macbeth):
        index = _build_name_index(macbeth)
        for lid in macbeth.locations:
            assert _normalise_id_name(lid) in index

    def test_object_ids_indexed(self, macbeth):
        index = _build_name_index(macbeth)
        for oid in macbeth.objects:
            assert _normalise_id_name(oid) in index


class TestFuzzyResolveId:
    def test_exact_match(self, macbeth):
        index = _build_name_index(macbeth)
        assert _fuzzy_resolve_id("ENT_MACBETH", index) == "ENT_MACBETH"

    def test_name_match(self, macbeth):
        index = _build_name_index(macbeth)
        result = _fuzzy_resolve_id("Macbeth", index)
        assert result == "ENT_MACBETH"

    def test_close_typo_resolves(self, macbeth):
        index = _build_name_index(macbeth)
        result = _fuzzy_resolve_id("ENT_MACBTH", index, threshold=0.6)
        assert result == "ENT_MACBETH"

    def test_completely_wrong_returns_none(self, macbeth):
        index = _build_name_index(macbeth)
        result = _fuzzy_resolve_id("ENT_ZZZZZZZZZZZ", index, threshold=0.6)
        assert result is None


# =====================================================================
# Fuzzy repair
# =====================================================================

class TestTryFuzzyRepair:
    def test_repairs_close_typo(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_MACBTH": "dead"},
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert any("ENT_MACBTH" in e.message for e in errors)

        patched, remappings, remaining = _try_fuzzy_repair(parsed, errors, macbeth)
        assert "ENT_MACBTH" in remappings
        assert remappings["ENT_MACBTH"] == "ENT_MACBETH"
        assert patched.interventions == {"ENT_MACBETH": "dead"}
        hard_remaining = [e for e in remaining if e.severity == "error"]
        assert not hard_remaining

    def test_repairs_focus_entity_ids(self, macbeth):
        parsed = ParsedQuery(
            query_type="observation",
            reasoning="test",
            focus_entity_ids=["ENT_MACBTH"],
        )
        errors = _validate_parsed_query(parsed, macbeth)
        patched, remappings, remaining = _try_fuzzy_repair(parsed, errors, macbeth)
        assert patched.focus_entity_ids == ["ENT_MACBETH"]

    def test_no_op_when_no_id_errors(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_MACBETH": "dead"},
        )
        errors = _validate_parsed_query(parsed, macbeth)
        assert not errors
        patched, remappings, remaining = _try_fuzzy_repair(parsed, errors, macbeth)
        assert not remappings
        assert not remaining

    def test_unrepairable_returns_original_errors(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
        )
        errors = _validate_parsed_query(parsed, macbeth)
        # "at least one intervention" — not an ID error
        patched, remappings, remaining = _try_fuzzy_repair(parsed, errors, macbeth)
        assert not remappings
        assert remaining == errors


# =====================================================================
# General fallback
# =====================================================================

class TestBuildGeneralFallback:
    def test_produces_valid_general_query(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
        )
        errors = [ValidationError(field="interventions", message="missing")]
        result = _build_general_fallback("Kill the ghost", parsed, errors)
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)
        assert "Kill the ghost" in result.query.question

    def test_fallback_info_populated(self):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="test",
            target_effect="suspense",
        )
        errors = [ValidationError(field="target_entity_ids", message="missing")]
        result = _build_general_fallback("Maximise suspense", parsed, errors)
        assert result.fallback is not None
        assert result.fallback.strategy == "general_fallback"
        assert result.fallback.original_query_type == "directive"
        assert len(result.fallback.original_errors) == 1


# =====================================================================
# _apply_fallback (integration of fuzzy + general)
# =====================================================================

class TestApplyFallback:
    def test_fuzzy_repair_path(self, macbeth):
        """Fuzzy repair succeeds for a close typo."""
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_MACBTH": "dead"},
        )
        errors = _validate_parsed_query(parsed, macbeth)
        result = _apply_fallback("Kill Macbeth", parsed, errors, macbeth)
        assert result.is_valid
        assert isinstance(result.query, InterventionQuery)
        assert result.fallback is not None
        assert result.fallback.strategy == "fuzzy_id_resolution"
        assert "ENT_MACBTH" in result.fallback.id_remappings

    def test_general_fallback_path_no_world_state(self):
        """Without a world state, fuzzy repair is impossible — fall back to general."""
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
        )
        errors = [ValidationError(
            field="interventions",
            message="Intervention query requires at least one intervention.",
        )]
        result = _apply_fallback("Do something", parsed, errors, None)
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)
        assert result.fallback.strategy == "general_fallback"

    def test_general_fallback_when_fuzzy_fails(self, macbeth):
        """Completely bogus IDs can't be fuzzy-resolved — falls back to general."""
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_XYZZY_NOBODY_999": "dead"},
        )
        errors = _validate_parsed_query(parsed, macbeth)
        result = _apply_fallback("Kill nobody", parsed, errors, macbeth)
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)
        assert result.fallback.strategy == "general_fallback"

    def test_structural_errors_fall_to_general(self, macbeth):
        """Missing required fields (not ID errors) go straight to general fallback."""
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="test",
        )
        errors = _validate_parsed_query(parsed, macbeth)
        result = _apply_fallback("Do directive", parsed, errors, macbeth)
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)
        assert result.fallback.strategy == "general_fallback"


# =====================================================================
# parse_query with mocked LLM — fallback integration
# =====================================================================

class TestParseQueryFallback:
    """Test that parse_query triggers fallback instead of returning invalid."""

    def _run(self, nl, parsed, world_state=None):
        mock_result = _make_mock_result(parsed)
        with patch("shadow_loom.query_parsing.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_sync.return_value = mock_result
            return parse_query(nl, world_state=world_state)

    def test_typo_id_fuzzy_repaired(self, macbeth):
        """A close-enough typo in an ID gets fuzzy-resolved."""
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="Kill Macbeth.",
            interventions={"ENT_MACBTH": "dead"},
        )
        result = self._run("Kill Macbeth", parsed, world_state=macbeth)
        assert result.is_valid
        assert isinstance(result.query, InterventionQuery)
        assert result.fallback is not None
        assert result.fallback.strategy == "fuzzy_id_resolution"

    def test_completely_bad_id_falls_to_general(self, macbeth):
        """A totally unresolvable ID falls back to general query."""
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_DOES_NOT_EXIST_AT_ALL_999": "dead"},
        )
        result = self._run("Kill the ghost", parsed, world_state=macbeth)
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)
        assert result.fallback is not None
        assert result.fallback.strategy == "general_fallback"
        assert result.fallback.original_query_type == "intervention"

    def test_missing_required_fields_falls_to_general(self, macbeth):
        """Missing structural fields also trigger general fallback."""
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="test",
        )
        result = self._run("Do something dramatic", parsed, world_state=macbeth)
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)
        assert result.fallback.strategy == "general_fallback"

    def test_no_fallback_on_valid_query(self, macbeth):
        """A perfectly valid query has no fallback info."""
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="test",
            interventions={"ENT_MACBETH": "dead"},
        )
        result = self._run("Kill Macbeth", parsed, world_state=macbeth)
        assert result.is_valid
        assert result.fallback is None


# =====================================================================
# _resolve_model
# =====================================================================

class TestResolveModel:
    """Test model string resolution in query_parsing."""

    def test_ollama_prefix_returns_model_object(self):
        from shadow_loom.query_parsing import _resolve_model
        model = _resolve_model("ollama:qwen3:8b")
        assert model is not None
        assert not isinstance(model, str)

    def test_non_ollama_passthrough(self):
        from shadow_loom.query_parsing import _resolve_model
        model = _resolve_model("anthropic:claude-3-sonnet")
        assert model == "anthropic:claude-3-sonnet"


# =====================================================================
# parse_query_async (mocked LLM)
# =====================================================================

class TestParseQueryAsync:
    """Test the async variant of parse_query."""

    @pytest.fixture
    def macbeth(self):
        return macbeth_ws

    async def _run_async(self, nl, parsed, world_state=None):
        mock_result = _make_mock_result(parsed)
        with patch("shadow_loom.query_parsing.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run = MagicMock()
            # Make run() return an awaitable
            import asyncio
            future = asyncio.Future()
            future.set_result(mock_result)
            instance.run.return_value = future
            from shadow_loom.query_parsing import parse_query_async
            return await parse_query_async(nl, world_state=world_state)

    @pytest.mark.asyncio
    async def test_async_observation(self, macbeth):
        parsed = ParsedQuery(
            query_type="observation",
            reasoning="Async test.",
            focus_entity_ids=["ENT_MACBETH"],
        )
        result = await self._run_async(
            "Show me Macbeth", parsed, world_state=macbeth,
        )
        assert result.is_valid
        assert isinstance(result.query, ObservationQuery)

    @pytest.mark.asyncio
    async def test_async_fallback_on_bad_id(self, macbeth):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="Async bad ID.",
            interventions={"ENT_DOES_NOT_EXIST_AT_ALL_999": "dead"},
        )
        result = await self._run_async(
            "Kill nobody", parsed, world_state=macbeth,
        )
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)
        assert result.fallback is not None

    @pytest.mark.asyncio
    async def test_async_no_world_state(self):
        parsed = ParsedQuery(
            query_type="general",
            reasoning="Async general.",
            question="What is happening?",
        )
        result = await self._run_async("What is happening?", parsed)
        assert result.is_valid
        assert isinstance(result.query, GeneralQuery)


# =====================================================================
# Typed query parsing (pre-specified query_type)
# =====================================================================

class TestTypedQueryParsing:
    """Tests for parse_query with explicit query_type parameter."""

    def _run(self, nl, parsed, query_type, world_state=None):
        mock_result = _make_mock_result(parsed)
        with patch("shadow_loom.query_parsing.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_sync.return_value = mock_result
            return parse_query(nl, query_type=query_type, world_state=world_state)

    def test_typed_prompt_used_for_intervention(self, macbeth):
        """When query_type is specified, the typed prompt is used."""
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="Force Macbeth dead.",
            interventions={"ENT_MACBETH": "dead"},
        )
        mock_result = _make_mock_result(parsed)
        with patch("shadow_loom.query_parsing.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_sync.return_value = mock_result
            parse_query("Kill Macbeth", query_type="intervention", world_state=macbeth)
            # Verify the typed prompt was used (not the legacy one)
            agent_call_kwargs = MockAgent.call_args
            prompt = agent_call_kwargs.kwargs.get("system_prompt", "")
            assert "INTERVENTION QUERY" in prompt
            assert "classify it into exactly one" not in prompt

    def test_typed_prompt_used_for_observation(self, macbeth):
        parsed = ParsedQuery(
            query_type="observation",
            reasoning="Show the scene.",
            focus_entity_ids=["ENT_MACBETH"],
        )
        mock_result = _make_mock_result(parsed)
        with patch("shadow_loom.query_parsing.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_sync.return_value = mock_result
            parse_query("Show Macbeth's scene", query_type="observation", world_state=macbeth)
            prompt = MockAgent.call_args.kwargs.get("system_prompt", "")
            assert "OBSERVATION QUERY" in prompt

    def test_type_override_on_misclassification(self, macbeth):
        """If the LLM returns the wrong query_type, it gets overridden."""
        parsed = ParsedQuery(
            query_type="general",  # LLM returns wrong type
            reasoning="Misclassified",
            interventions={"ENT_MACBETH": "dead"},
        )
        result = self._run("Kill Macbeth", parsed, "intervention", macbeth)
        assert result.is_valid
        assert isinstance(result.query, InterventionQuery)

    def test_invalid_query_type_raises(self):
        with pytest.raises(ValueError, match="Invalid query_type"):
            parse_query("test", query_type="invalid_type")

    def test_all_query_types_are_valid(self):
        """Every type in QUERY_TYPES has a corresponding prompt."""
        for qt in QUERY_TYPES:
            prompt = _build_typed_system_prompt(qt)
            assert qt in prompt
            assert "ID RESOLUTION RULES" in prompt

    def test_none_query_type_uses_legacy_prompt(self, macbeth):
        """When query_type is None, the legacy classification prompt is used."""
        parsed = ParsedQuery(
            query_type="general",
            reasoning="test",
            question="What is happening?",
        )
        mock_result = _make_mock_result(parsed)
        with patch("shadow_loom.query_parsing.Agent") as MockAgent:
            instance = MockAgent.return_value
            instance.run_sync.return_value = mock_result
            parse_query("What is happening?", query_type=None, world_state=macbeth)
            prompt = MockAgent.call_args.kwargs.get("system_prompt", "")
            assert "classify it into exactly one" in prompt

    def test_typed_counterfactual(self, macbeth):
        evt_id = macbeth.events[0].id
        ent_id = next(iter(macbeth.entities))
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="What-if about past.",
            historical_interventions={evt_id: "never happened"},
            evidence_node_ids=[ent_id],
        )
        result = self._run(
            "What if Duncan was never murdered?",
            parsed, "counterfactual", macbeth,
        )
        assert result.is_valid
        assert isinstance(result.query, CounterfactualQuery)

    def test_typed_directive(self, macbeth):
        parsed = ParsedQuery(
            query_type="directive",
            reasoning="Maximise suspense.",
            target_entity_ids=["ENT_MACBETH"],
            target_effect="suspense",
        )
        result = self._run(
            "Maximise suspense for Macbeth",
            parsed, "directive", macbeth,
        )
        assert result.is_valid
        assert isinstance(result.query, DirectiveQuery)

    def test_typed_interrogate(self):
        parsed = ParsedQuery(
            query_type="interrogate",
            reasoning="Pathfinding.",
            question="Who caused X?",
        )
        result = self._run("Who caused X?", parsed, "interrogate")
        assert result.is_valid
        assert isinstance(result.query, InterrogationQuery)
