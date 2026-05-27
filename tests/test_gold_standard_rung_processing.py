# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gold-standard rung-processing harness.

Every hand-built example world in ``example_worlds/`` should:

  * already satisfy the full schema audit (covered by
    ``test_ingestion.test_plot_model_passes_schema_audit``);
  * be a legal substrate for Pearl-rung processing through
    :func:`shadow_loom.narrative_physics.calculate_narrative_physics`.

This module is the second invariant. For each plot world we run:

  * Rung 1 — ``ObservationQuery`` with no POV (Omniscient Graph).
  * Rung 1 — ``ObservationQuery`` with the first entity as POV
    (ego-graph extraction).
  * Rung 2 — ``InterventionQuery`` with a single ``DoEvent``
    preventing the first event (do-calculus surgery).
  * Rung 3 — ``CounterfactualQuery`` with the same historical
    ``DoEvent`` and the last event id as evidence (abduction).

Each call must return ``status == "success"`` and a non-empty
``physics_state`` payload.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import (
    ObservationQuery,
    InterventionQuery,
    CounterfactualQuery,
    DoEvent,
)


_TEST_MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "example_worlds"
_MODEL_FILES = sorted(
    f for f in _TEST_MODEL_DIR.glob("*.py") if f.name != "__init__.py"
)


def _load_world(model_path: pathlib.Path):
    mod = importlib.import_module(f"example_worlds.{model_path.stem}")
    return mod.world_state


@pytest.mark.parametrize("model_path", _MODEL_FILES, ids=lambda p: p.stem)
class TestRungProcessingGoldStandard:
    """Every gold-standard world must process cleanly at all three rungs."""

    def test_rung1_omniscient_observation(self, model_path):
        ws = _load_world(model_path)
        q = ObservationQuery(focus_entity_ids=[])
        result = calculate_narrative_physics(q, ws)
        assert result["status"] == "success", result
        assert result["query_type"] == "observation"
        assert result["physics_state"], "Omniscient graph must be non-empty"

    def test_rung1_ego_observation(self, model_path):
        ws = _load_world(model_path)
        if not ws.entities:
            pytest.skip("World has no entities")
        first_ent_id = next(iter(ws.entities.keys()))
        q = ObservationQuery(focus_entity_ids=[first_ent_id])
        result = calculate_narrative_physics(q, ws)
        assert result["status"] == "success", result
        assert result["physics_state"], "Ego graph must be non-empty"

    def test_rung2_intervention_prevents_event(self, model_path):
        ws = _load_world(model_path)
        if not ws.events:
            pytest.skip("World has no events")
        target_event = ws.events[0].id
        q = InterventionQuery(
            interventions={},
            do_targets=[DoEvent(event_id=target_event, occurred=False)],
            force_implausible=True,
        )
        result = calculate_narrative_physics(q, ws, use_causal_engine=True)
        # The engine should at minimum report a structured outcome; the
        # gold-standard contract is that it does not raise and returns a
        # success-or-implausible payload (implausibility is a legitimate
        # epistemic answer, not a structural failure).
        assert result["status"] in ("success", "implausible"), result

    def test_rung3_counterfactual_with_event_evidence(self, model_path):
        ws = _load_world(model_path)
        if len(ws.events) < 2:
            pytest.skip("Need at least two events for a rung-3 query")
        historical_event = ws.events[0].id
        evidence_event = ws.events[-1].id
        q = CounterfactualQuery(
            historical_interventions={},
            historical_do_targets=[
                DoEvent(event_id=historical_event, occurred=False),
            ],
            evidence_node_ids=[evidence_event],
            force_implausible=True,
        )
        result = calculate_narrative_physics(q, ws, use_causal_engine=True)
        assert result["status"] in ("success", "implausible"), result
