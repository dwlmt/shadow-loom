# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression: do() on a trait axis the entity does not yet have.

Previously such an intervention wrote a bare float into the sandbox node
(bypassing the inertia gate) and emitted a ``TraitMutation`` with
``old_value=NaN``, while *also* failing to record the target in
``skipped_interventions``. Under the create-if-absent do-clamp semantics
(matching ``_intervene_relationship``), an absent axis is materialised as a
well-formed ``TraitVector`` seeded at the 0.0 baseline, so the clamp applies
cleanly: no NaN, a finite ``old_value`` measured from 0.0, and the
intervention is (correctly) *not* skipped because it did take effect.
"""
import math

import importlib

from shadow_loom.causal_physics import CausalPhysicsEngine
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator


def _run(modname, focus, interventions):
    ws = importlib.import_module(f"example_worlds.{modname}").world_state
    ego = extract_ego_graph_from_memory(ws, focus_entity_ids=focus)
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "intervention")
    engine = CausalPhysicsEngine(sandbox, ws)
    result = engine.execute(
        rung=2, interventions=interventions, target_node_ids=focus,
    )
    return ws, sandbox, result


def test_absent_trait_clamp_emits_no_nan_old_value():
    # ENT_FRIAR_LAURENCE has no ``diligence`` axis in the fixture.
    ws, sandbox, result = _run(
        "romeo_and_juliet",
        ["ENT_ROMEO", "ENT_JULIET", "ENT_FRIAR_LAURENCE"],
        {"ENT_FRIAR_LAURENCE.traits.diligence": 1.0},
    )
    assert "diligence" not in (ws.entities["ENT_FRIAR_LAURENCE"].traits or {})

    nan_rows = [
        m for m in result.mutations
        if isinstance(m.old_value, float) and math.isnan(m.old_value)
    ]
    assert nan_rows == [], f"NaN old_value leaked: {nan_rows}"

    diligence_rows = [
        m for m in result.mutations
        if m.node_id == "ENT_FRIAR_LAURENCE" and m.trait == "diligence"
    ]
    assert diligence_rows, "absent-axis clamp produced no mutation row"
    row = diligence_rows[0]
    assert row.old_value == 0.0  # measured from the materialised baseline
    assert row.new_value == 1.0
    assert math.isfinite(row.old_value) and math.isfinite(row.new_value)


def test_absent_trait_clamp_is_not_falsely_skipped():
    # The intervention applies (axis is materialised + clamped), so it must
    # NOT appear in the skipped_interventions ledger reserved for targets
    # that genuinely cannot be resolved (malformed key / unknown node).
    _ws, sandbox, _result = _run(
        "romeo_and_juliet",
        ["ENT_ROMEO", "ENT_JULIET", "ENT_FRIAR_LAURENCE"],
        {"ENT_FRIAR_LAURENCE.traits.diligence": 1.0},
    )
    skipped = sandbox.graph.get("skipped_interventions") or []
    diligence_skips = [
        s for s in skipped
        if "diligence" in str(s.get("target_path", ""))
    ]
    assert diligence_skips == [], f"intervention falsely skipped: {diligence_skips}"


def test_absent_trait_noop_clamp_emits_no_nan():
    # ENT_DUNCAN has no ``kindness`` axis; clamping it to its 0.0 baseline is
    # a no-op, which must be handled without a NaN row.
    _ws, _sandbox, result = _run(
        "macbeth",
        ["ENT_MACBETH", "ENT_DUNCAN", "ENT_BANQUO"],
        {"ENT_DUNCAN.traits.kindness": 0.0},
    )
    nan_rows = [
        m for m in result.mutations
        if isinstance(m.old_value, float) and math.isnan(m.old_value)
    ]
    assert nan_rows == [], f"NaN old_value leaked on no-op clamp: {nan_rows}"


def test_existing_trait_clamp_unchanged():
    # Regression guard: clamping a trait the entity *does* have still works.
    ws, _sandbox, result = _run(
        "macbeth",
        ["ENT_MACBETH", "ENT_DUNCAN"],
        {"ENT_MACBETH.traits.ambition": 0.0},
    )
    assert "ambition" in (ws.entities["ENT_MACBETH"].traits or {})
    rows = [
        m for m in result.mutations
        if m.node_id == "ENT_MACBETH" and m.trait == "ambition"
    ]
    assert rows, "existing-trait clamp produced no mutation row"
    assert all(math.isfinite(m.old_value) for m in rows)
