# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for shadow_loom_ui.reasoning_helpers."""
from __future__ import annotations

import pytest

from example_worlds.macbeth import world_state as macbeth_ws

from shadow_loom_ui.reasoning_helpers import (
    VIOLATION_DIRECTIVE_TEMPLATES,
    VIOLATION_EXPLANATIONS,
    attribution_graph_data,
    belief_provenance_data,
    convergence_trajectory_data,
    directive_for_violation,
    event_context_data,
    extract_reasoning_trace,
    foreshadowing_arcs_data,
    reasoning_trace_summary,
    structured_response_data,
    syuzhet_event_index,
    violation_explanation,
    world_diff_data,
)


# =====================================================================
# Violation explanations + directives
# =====================================================================


def test_violation_explanation_covers_all_known_types():
    """Every violation_type in the auditor's Literal must have an entry."""
    from shadow_loom.auditor import AuditViolation
    # Pull the Literal values out of the field annotation
    field_info = AuditViolation.model_fields["violation_type"]
    # pydantic v2: annotation is a typing.Literal — extract its args
    args = getattr(field_info.annotation, "__args__", ())
    assert args, "expected violation_type to be a Literal with members"
    for v in args:
        assert v in VIOLATION_EXPLANATIONS, f"missing explanation for {v}"


def test_violation_explanation_unknown_type_falls_back():
    msg = violation_explanation("not_a_real_type")
    assert "unrecognised" in msg.lower()


def test_directive_for_violation_returns_template_when_known():
    template = directive_for_violation("miracle_step")
    assert template is not None
    prompt, qtype = template
    assert prompt
    assert qtype in ("directive", "intervention", "counterfactual")


def test_directive_for_violation_unknown_returns_none():
    assert directive_for_violation("not_a_real_type") is None


def test_all_violation_directives_have_valid_query_types():
    valid = {"directive", "intervention", "counterfactual", "general"}
    for vtype, (prompt, qtype) in VIOLATION_DIRECTIVE_TEMPLATES.items():
        assert prompt, f"empty prompt for {vtype}"
        assert qtype in valid, f"invalid query_type {qtype} for {vtype}"


# =====================================================================
# Reasoning trace
# =====================================================================


def test_extract_reasoning_trace_empty_inputs():
    trace = extract_reasoning_trace(None)
    assert trace["rung"] is None
    for k in ("do_set", "evidence", "abduction", "cascade", "social_cascade", "blocked"):
        assert trace[k] == []


def test_extract_reasoning_trace_intervention():
    physics = {
        "query_type": "intervention",
        "intervened_nodes": ["ENT_MACBETH"],
        "mutations": [
            {
                "node_id": "ENT_MACBETH",
                "trait": "guilt",
                "old_value": 0.2,
                "new_value": 0.8,
                "impact": 0.6,
                "inertia": 0.3,
                "edge_kind": "mutation",
                "propagation_delay": 0,
                "triggered_by": "EVT_DUNCAN_MURDER",
            },
        ],
        "social_mutations": [
            {
                "source_entity_id": "ENT_MACBETH",
                "target_entity_id": "ENT_LADY_MACBETH",
                "metric": "affinity",
                "old_value": 0.5, "new_value": 0.3,
                "impact": -0.2, "inertia": 0.4,
                "triggered_by": "EVT_DUNCAN_MURDER",
            },
        ],
        "blocked": [
            {"node_id": "ENT_BANQUO", "trait": "loyalty",
             "impact": 0.1, "inertia": 0.9, "reason": "inertia"},
        ],
    }
    trace = extract_reasoning_trace(physics, ws=macbeth_ws)
    assert trace["rung"] == 2
    assert len(trace["do_set"]) == 1
    assert trace["do_set"][0]["node_id"] == "ENT_MACBETH"
    # macbeth_ws maps the id to a name
    assert trace["do_set"][0]["label"] == macbeth_ws.entities["ENT_MACBETH"].name
    assert len(trace["cascade"]) == 1
    assert trace["cascade"][0]["trait"] == "guilt"
    assert len(trace["social_cascade"]) == 1
    assert len(trace["blocked"]) == 1
    assert trace["evidence"] == []  # rung-2 has no evidence
    assert trace["abduction"] == []


def test_extract_reasoning_trace_counterfactual_includes_abduction():
    physics = {
        "query_type": "counterfactual",
        "evidence_node_ids": ["ENT_MACBETH"],
        "hidden_deltas": {
            "ENT_MACBETH": {"ambition": 0.3, "fear": -0.1},
        },
        "intervened_nodes": [],
        "mutations": [],
    }
    trace = extract_reasoning_trace(physics, ws=macbeth_ws)
    assert trace["rung"] == 3
    assert len(trace["evidence"]) == 1
    assert len(trace["abduction"]) == 2
    abduced_traits = {a["trait"] for a in trace["abduction"]}
    assert abduced_traits == {"ambition", "fear"}


def test_reasoning_trace_summary_describes_counts():
    trace = {
        "rung": 3,
        "do_set": [{}],
        "evidence": [{}],
        "abduction": [{}, {}],
        "cascade": [{}, {}, {}],
        "social_cascade": [],
        "blocked": [{}],
    }
    s = reasoning_trace_summary(trace)
    assert "rung-3" in s
    assert "1 pinned" in s
    assert "2 abduced" in s
    assert "3 mutations" in s
    assert "1 blocked" in s


def test_reasoning_trace_summary_empty():
    trace = extract_reasoning_trace(None)
    assert reasoning_trace_summary(trace) == "no reasoning trace"


# =====================================================================
# Belief provenance
# =====================================================================


def test_belief_provenance_unknown_entity_returns_empty():
    assert belief_provenance_data(macbeth_ws, "ENT_DOES_NOT_EXIST") == []


def test_belief_provenance_returns_initial_beliefs_when_present():
    """Pick any entity that has initial beliefs in the seed."""
    target_id = None
    for eid, ent in macbeth_ws.entities.items():
        if ent.beliefs:
            target_id = eid
            break
    if target_id is None:
        pytest.skip("Macbeth seed has no initial beliefs to test against")
    rows = belief_provenance_data(macbeth_ws, target_id)
    assert rows, "expected at least the initial beliefs"
    # Initial rows must come first chronologically
    initial_kinds = [r["kind"] for r in rows if r["kind"] == "initial"]
    assert initial_kinds, "expected at least one 'initial' kind row"
    # Rows are sorted by fabula_time ascending
    times = [r["fabula_time"] for r in rows]
    assert times == sorted(times)


# =====================================================================
# Attribution graph
# =====================================================================


def test_attribution_graph_unknown_target_returns_empty():
    nodes, links, cats, paths = attribution_graph_data(macbeth_ws, "EVT_DOES_NOT_EXIST")
    assert nodes == []
    assert links == []
    assert cats == []
    assert paths == []


def test_attribution_graph_target_is_in_nodes():
    """For any event, target is included and highlighted."""
    if not macbeth_ws.events:
        pytest.skip("no events in macbeth seed")
    target_id = macbeth_ws.events[-1].id  # last event (most causes upstream)
    nodes, links, _cats, _paths = attribution_graph_data(macbeth_ws, target_id, max_depth=3)
    node_ids = {n["id"] for n in nodes}
    assert target_id in node_ids
    target_node = next(n for n in nodes if n["id"] == target_id)
    # Highlighted with gold border
    assert target_node["itemStyle"]["borderColor"] == "#FFD700"


def test_attribution_graph_paths_sorted_by_force_desc():
    if not macbeth_ws.events:
        pytest.skip("no events in macbeth seed")
    # Find an event with at least one incoming causal edge
    target_id = None
    for evt in macbeth_ws.events:
        if any(ce.target_id == evt.id for ce in macbeth_ws.causal_topology):
            target_id = evt.id
            break
    if target_id is None:
        pytest.skip("no event has incoming causal edges in seed")
    _, _, _, paths = attribution_graph_data(macbeth_ws, target_id, max_depth=4)
    if len(paths) >= 2:
        forces = [p["force"] for p in paths]
        assert forces == sorted(forces, reverse=True)


def test_attribution_graph_respects_min_force():
    if not macbeth_ws.events:
        pytest.skip("no events in macbeth seed")
    target_id = macbeth_ws.events[-1].id
    _, links_low, _, _ = attribution_graph_data(
        macbeth_ws, target_id, max_depth=4, min_force=0.0,
    )
    _, links_high, _, _ = attribution_graph_data(
        macbeth_ws, target_id, max_depth=4, min_force=99.0,
    )
    assert len(links_high) <= len(links_low)


# =====================================================================
# Foreshadowing arcs
# =====================================================================


def test_foreshadowing_arcs_returns_only_delayed_edges():
    arcs = foreshadowing_arcs_data(macbeth_ws)
    # Every arc must correspond to an edge with propagation_delay > 0
    for arc in arcs:
        assert arc["span"] >= 1
        assert arc["force"] >= 0
        assert arc["evidence"] in ("weak", "moderate", "strong")


def test_foreshadowing_arcs_loose_flag_consistent():
    arcs = foreshadowing_arcs_data(macbeth_ws)
    event_ids = {evt.id for evt in macbeth_ws.events}
    entity_ids = set(macbeth_ws.entities)
    for arc in arcs:
        expected_loose = (
            arc["payoff_id"] not in event_ids
            and arc["payoff_id"] not in entity_ids
        )
        assert arc["is_loose"] == expected_loose


# =====================================================================
# Convergence trajectory
# =====================================================================


def test_convergence_trajectory_handles_none():
    assert convergence_trajectory_data(None) == []


class _FakeAudit:
    def __init__(self, violations, passed):
        self.violations = violations
        self.passed = passed


class _FakeViolation:
    def __init__(self, severity):
        self.severity = severity


class _FakeCycle:
    def __init__(self, iteration, audit):
        self.iteration = iteration
        self.audit_result = audit


class _FakeFeedback:
    def __init__(self, history):
        self.history = history


def test_convergence_trajectory_counts_critical_separately():
    fb = _FakeFeedback([
        _FakeCycle(1, _FakeAudit(
            [_FakeViolation("critical"), _FakeViolation("minor")],
            passed=False,
        )),
        _FakeCycle(2, _FakeAudit(
            [_FakeViolation("minor")], passed=True,
        )),
    ])
    rows = convergence_trajectory_data(fb)
    assert len(rows) == 2
    assert rows[0]["iteration"] == 1
    assert rows[0]["violation_count"] == 2
    assert rows[0]["critical_count"] == 1
    assert rows[0]["passed"] is False
    assert rows[1]["violation_count"] == 1
    assert rows[1]["critical_count"] == 0
    assert rows[1]["passed"] is True


# =====================================================================
# World diff
# =====================================================================


def test_world_diff_identical_world_yields_no_changes():
    diff = world_diff_data(macbeth_ws, macbeth_ws)
    assert diff["totals"]["added"] == 0
    assert diff["totals"]["removed"] == 0
    assert diff["totals"]["changed"] == 0


def test_world_diff_detects_status_change():
    import copy
    other = copy.deepcopy(macbeth_ws)
    eid = next(iter(other.entities))
    original_status = other.entities[eid].status
    new_status = "dead" if original_status != "dead" else "injured"
    other.entities[eid].status = new_status
    diff = world_diff_data(macbeth_ws, other)
    assert diff["totals"]["changed"] >= 1
    changed_fields = {(c["id"], c["field"]) for c in diff["changed"]["entities"]}
    assert (eid, "status") in changed_fields


def test_world_diff_detects_added_event():
    import copy
    from shadow_loom.models import EventNode
    other = copy.deepcopy(macbeth_ws)
    new_evt = EventNode(
        id="EVT_TEST_ADDED",
        fabula_time=99999,
        syuzhet_index=99999,
        event_type="outcome",
        actor_ids=[],
        target_ids=[],
        description="test event",
    )
    other.events.append(new_evt)
    diff = world_diff_data(macbeth_ws, other)
    assert "EVT_TEST_ADDED" in diff["added"]["events"]


# =====================================================================
# Structured response data
# =====================================================================


def test_structured_response_handles_empty_input():
    card = structured_response_data(None)
    assert card["claim"]
    assert card["evidence"] == []
    assert card["confidence"] == 0.0


def test_structured_response_implausible_yields_low_confidence():
    physics = {
        "query_type": "intervention",
        "status": "implausible",
        "implausibility_reason": "no resolvable target",
    }
    card = structured_response_data(physics)
    assert card["confidence"] <= 0.2
    assert any("implausible" in c.lower() for c in card["caveats"])


def test_structured_response_extracts_proof_evidence():
    physics = {
        "query_type": "interrogate",
        "answer": "Yes, there is a path.",
        "proof": [
            {"id": "ENT_MACBETH", "kind": "entity"},
            {"id": "LOC_CASTLE", "kind": "location"},
        ],
    }
    card = structured_response_data(physics, ws=macbeth_ws)
    assert "path" in card["claim"].lower()
    assert len(card["evidence"]) == 2
    assert card["evidence"][0]["node_id"] == "ENT_MACBETH"


def test_structured_response_uses_focus_entities_when_no_proof():
    physics = {
        "query_type": "general",
        "focus_entity_ids": ["ENT_MACBETH"],
    }
    card = structured_response_data(physics, ws=macbeth_ws)
    assert len(card["evidence"]) == 1
    assert card["evidence"][0]["kind"] == "focus"


# =====================================================================
# Event navigator helpers
# =====================================================================


def test_syuzhet_event_index_sorted_by_syuzhet():
    rows = syuzhet_event_index(macbeth_ws)
    assert rows, "macbeth seed should have events"
    indices = [r["syuzhet_index"] for r in rows]
    assert indices == sorted(indices)
    # Each row must carry the structural fields the UI consumes
    for r in rows:
        assert {"id", "syuzhet_index", "fabula_time", "event_type",
                "description", "actor_count", "target_count"} <= set(r)


def test_event_context_unknown_id_returns_empty():
    assert event_context_data(macbeth_ws, "EVT_DOES_NOT_EXIST") == {}


def test_event_context_includes_actors_targets_and_neighbours():
    # Pick the first event with at least one actor
    target_evt = next(
        (e for e in macbeth_ws.events if e.actor_ids), None,
    )
    if target_evt is None:
        pytest.skip("no event with actors in seed")
    ctx = event_context_data(macbeth_ws, target_evt.id)
    assert ctx["event"]["id"] == target_evt.id
    assert ctx["event"]["fabula_time"] == target_evt.fabula_time
    assert ctx["event"]["syuzhet_index"] == target_evt.syuzhet_index
    # Actors must be reconstructed dossiers
    assert len(ctx["actors"]) == len(target_evt.actor_ids)
    for a in ctx["actors"]:
        assert "id" in a and "name" in a
        assert "traits" in a and isinstance(a["traits"], dict)
        assert "beliefs" in a and isinstance(a["beliefs"], list)
    # Neighbours present
    assert "syuzhet_neighbours" in ctx
    assert "prev" in ctx["syuzhet_neighbours"]
    assert "next" in ctx["syuzhet_neighbours"]


def test_event_context_world_traits_active_sorted_by_magnitude():
    target_evt = macbeth_ws.events[0]
    ctx = event_context_data(macbeth_ws, target_evt.id)
    wts = ctx["world_traits_active"]
    if len(wts) >= 2:
        magnitudes = [w["magnitude"]["value"] for w in wts]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_event_context_incoming_outgoing_sorted_by_force():
    # Find an event that has both incoming and outgoing causal edges
    target_evt = None
    for e in macbeth_ws.events:
        has_in = any(ce.target_id == e.id for ce in macbeth_ws.causal_topology)
        has_out = any(ce.source_id == e.id for ce in macbeth_ws.causal_topology)
        if has_in and has_out:
            target_evt = e
            break
    if target_evt is None:
        pytest.skip("no event has both in and out causal edges")
    ctx = event_context_data(macbeth_ws, target_evt.id)
    if len(ctx["incoming"]) >= 2:
        forces = [r["force"] for r in ctx["incoming"]]
        assert forces == sorted(forces, reverse=True)
    if len(ctx["outgoing"]) >= 2:
        forces = [r["force"] for r in ctx["outgoing"]]
        assert forces == sorted(forces, reverse=True)


def test_event_context_actor_beliefs_respect_fabula_time():
    """Beliefs returned for an actor must be those valid AT the event's fabula_time."""
    # Find any actor with at least one initial belief
    target_evt = None
    for e in macbeth_ws.events:
        for aid in e.actor_ids:
            ent = macbeth_ws.entities.get(aid)
            if ent and ent.beliefs:
                target_evt = e
                break
        if target_evt is not None:
            break
    if target_evt is None:
        pytest.skip("no actor has initial beliefs in seed")
    ctx = event_context_data(macbeth_ws, target_evt.id)
    for actor in ctx["actors"]:
        for belief in actor["beliefs"]:
            # Every belief must have been established no later than this event
            est = belief.get("established_at_fabula", 0)
            assert est <= target_evt.fabula_time, (
                f"actor {actor['id']} sees belief established at {est} "
                f"during event at fabula t={target_evt.fabula_time}"
            )
