# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``surface_entity_constants`` (AUDIT round-2 P0)."""
from __future__ import annotations

from shadow_loom.interrogate_posterior import surface_entity_constants
from shadow_loom.models import (
    Entity,
    Location,
    TraitVector,
    WorldStateV1,
)


def _trait() -> TraitVector:
    return TraitVector(value=0.5, inertia=0.5, evidence_strength="weak")


def _ws(entities):
    return WorldStateV1(
        entities=entities,
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        objects={},
        events=[],
        causal_topology=[],
        propositions=[],
    )


def test_surface_no_constants_returns_empty():
    ent = Entity(id="ENT_A", name="A", location_id="LOC_X",
                 status="healthy", traits={"loyalty": _trait()})
    assert surface_entity_constants(_ws({"ENT_A": ent})) == []


def test_surface_emits_one_row_per_constant():
    ent = Entity(id="ENT_FEYRE", name="Feyre", location_id="LOC_X",
                 status="healthy", traits={"loyalty": _trait()},
                 constants=["mortal_origin", "huntress"])
    rows = surface_entity_constants(_ws({"ENT_FEYRE": ent}))
    assert len(rows) == 2
    tags = {r["constant"] for r in rows}
    assert tags == {"mortal_origin", "huntress"}
    assert all(r["entity_id"] == "ENT_FEYRE" for r in rows)
    assert all(r["kind"] == "ConstantPrerequisite" for r in rows)


def test_subject_filter_restricts_scan():
    a = Entity(id="ENT_A", name="A", location_id="LOC_X",
               status="healthy", traits={"loyalty": _trait()},
               constants=["tag_a"])
    b = Entity(id="ENT_B", name="B", location_id="LOC_X",
               status="healthy", traits={"loyalty": _trait()},
               constants=["tag_b"])
    rows = surface_entity_constants(_ws({"ENT_A": a, "ENT_B": b}),
                                    subject_ids=["ENT_A"])
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "ENT_A"
