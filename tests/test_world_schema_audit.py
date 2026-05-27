# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``audit_world_schema`` (AUDIT P1-7)."""
from __future__ import annotations

import pytest

from shadow_loom.models import (
    Belief,
    CausalEdge,
    Entity,
    EventNode,
    Location,
    Proposition,
    RelationshipEdge,
    RelationshipMetric,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.world_schema_audit import audit_world_schema


def _base_ws(**overrides) -> WorldStateV1:
    defaults = dict(
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A",
                location_id="LOC_X", status="healthy",
                traits={"loyalty": TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")},
            ),
        },
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        objects={},
        events=[],
        causal_topology=[],
        propositions=[],
    )
    defaults.update(overrides)
    return WorldStateV1(**defaults)


def test_clean_world_has_no_issues():
    issues = audit_world_schema(_base_ws())
    assert issues == []


def test_detects_unknown_event_location():
    ws = _base_ws(events=[
        EventNode(id="EVT_X", fabula_time=1000, syuzhet_index=1,
                  event_type="outcome", actor_ids=["ENT_A"], target_ids=[],
                  description="x", at_location_id="LOC_NOPE"),
    ])
    issues = audit_world_schema(ws)
    assert any("at_location_id ='LOC_NOPE'" in i for i in issues)


def test_detects_dangling_actor():
    ws = _base_ws(events=[
        EventNode(id="EVT_X", fabula_time=1000, syuzhet_index=1,
                  event_type="outcome", actor_ids=["ENT_NOPE"], target_ids=[],
                  description="x", at_location_id="LOC_X"),
    ])
    issues = audit_world_schema(ws)
    assert any("actor_id ='ENT_NOPE'" in i for i in issues)


def test_detects_dangling_belief_proposition():
    ent = Entity(
        id="ENT_A", name="A",
        location_id="LOC_X", status="healthy",
        traits={"loyalty": TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")},
        beliefs=[
            Belief(target_id="ENT_A", perceived_state="self",
                   proposition_id="PROP_NOPE",
                   confidence=0.5, inertia=0.5, evidence_strength="moderate"),
        ],
    )
    ws = _base_ws(entities={"ENT_A": ent})
    issues = audit_world_schema(ws)
    assert any("proposition_id='PROP_NOPE'" in i for i in issues)


def test_detects_truth_tick_before_event():
    evt = EventNode(id="EVT_X", fabula_time=5000, syuzhet_index=1,
                    event_type="outcome", actor_ids=["ENT_A"], target_ids=[],
                    description="x", at_location_id="LOC_X")
    prop = Proposition(
        proposition_id="PROP_Y", kind="event_occurs",
        referent_ids=["EVT_X"],
        description="x", audience_default_prior=0.5, stakes=0.5,
        truth_at_fabula={1000: True},  # before EVT_X's fabula_time
    )
    ws = _base_ws(events=[evt], propositions=[prop])
    issues = audit_world_schema(ws)
    assert any("PROP_Y truth_at_fabula tick=1000 predates" in i for i in issues)


def test_detects_relationship_metric_before_established():
    re = RelationshipEdge(
        source_entity_id="ENT_A", target_entity_id="ENT_A",
        established_at_fabula=5000,
        metrics={
            "affinity": RelationshipMetric(
                value=0.5, inertia=0.5, evidence_strength="moderate",
                last_updated_fabula=1000,  # before established
            ),
        },
    )
    ws = _base_ws(social_topology=[re])
    issues = audit_world_schema(ws)
    assert any("last_updated=1000 predates established=5000" in i for i in issues)
