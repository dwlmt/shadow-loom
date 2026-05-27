# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``WorldStateV1.projected_for_branch(strict=...)`` (AUDIT P1-6)."""
from __future__ import annotations

import pytest

from shadow_loom.models import (
    Entity,
    Location,
    TraitVector,
    WorldStateV1,
)


def _ws() -> WorldStateV1:
    return WorldStateV1(
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


def test_default_falls_back_silently_on_missing_label():
    ws = _ws()
    result = ws.projected_for_branch(branch_world_id="shadow", branch_label=None)
    assert result is ws


def test_strict_raises_on_missing_label():
    ws = _ws()
    with pytest.raises(ValueError, match="Shadow read without branch_label"):
        ws.projected_for_branch(
            branch_world_id="shadow", branch_label=None, strict=True,
        )


def test_strict_does_not_raise_for_factual_reads():
    ws = _ws()
    result = ws.projected_for_branch(
        branch_world_id="factual", branch_label=None, strict=True,
    )
    assert result is ws


def test_strict_does_not_raise_when_label_present_but_no_sidecar():
    # No sidecar entries for this label -> projected_for_branch
    # returns self (no surgery to apply). This is *not* the buggy
    # condition the strict flag guards against; it should be silent.
    ws = _ws()
    result = ws.projected_for_branch(
        branch_world_id="shadow", branch_label="branch_x", strict=True,
    )
    assert result is ws
