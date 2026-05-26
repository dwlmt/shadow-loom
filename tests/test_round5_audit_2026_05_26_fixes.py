# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the round-5 audit remediation batch dated
2026-05-26.

Each section pins one finding from the round-5 audit so a regression
in the async defer-reextraction contract, the evaluation branch
isolation (prose-scope + focus-id cap), the MCP evaluate prose
lineage scoping, the DirectiveQuery bounds, or the save_version
IntegrityError retry specificity will fail loudly.
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from shadow_loom import pipeline as _pipeline
from shadow_loom.db import get_lineage_to_root, save_version
from shadow_loom.models import (
    Entity,
    EventNode,
    Location,
    TraitVector,
    WorldStateV1,
)
from shadow_loom.query_models import (
    DirectiveQuery,
    ObservationQuery,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seven_entity_world() -> WorldStateV1:
    locs = {
        "LOC_HALL": Location(
            id="LOC_HALL", name="Hall", description="Hall", ambient_state={},
        ),
    }
    ents = {}
    for i in range(7):
        eid = f"ENT_{i:02d}"
        ents[eid] = Entity(
            id=eid, name=f"E{i}", location_id="LOC_HALL",
            status="healthy",
            traits={"courage": TraitVector(value=0.5, inertia=0.2)},
        )
    return WorldStateV1(
        locations=locs,
        objects={},
        entities=ents,
        events=[
            EventNode(
                id="EVT_0", fabula_time=1, syuzhet_index=1,
                event_type="choice", actor_ids=["ENT_00"],
                target_ids=["ENT_01"], description="They meet",
            ),
        ],
        social_topology=[],
        spatial_topology=[],
        causal_topology=[],
    )


# ===========================================================================
# #1 + #2 — run_pipeline_async rejects cfg.defer_reextraction with ValueError
# ===========================================================================

class TestAsyncDeferReextractionRejected:
    """The async pipeline has no ``finish_reextraction_async`` parallel.

    Silently ignoring ``cfg.defer_reextraction`` (the prior behaviour)
    made the flag a lie on the async path. The contract is now: reject
    at the boundary so callers get a loud ``ValueError`` rather than a
    blocking inline run.
    """

    def test_async_pipeline_raises_value_error_on_defer(self):
        ws = _seven_entity_world()
        cfg = _pipeline.PipelineConfig(defer_reextraction=True)
        query = ObservationQuery(reasoning="r", question="What happens?")
        with pytest.raises(ValueError, match="defer_reextraction"):
            asyncio.run(
                _pipeline.run_pipeline_async(
                    query, world_state=ws, config=cfg,
                )
            )

    def test_async_pipeline_accepts_skip_with_defer(self):
        """When ``skip_reextraction`` wins, defer is moot \u2014 no raise."""
        cfg = _pipeline.PipelineConfig(
            defer_reextraction=True, skip_reextraction=True,
        )
        # Construct cfg-only; we only assert the guard does not fire
        # for the skip-wins combination. (Driving a full async pipeline
        # here would require a heavy LLM stub harness.)
        # Source-inspect that the guard is conditioned on
        # ``not cfg.skip_reextraction``.
        src = inspect.getsource(_pipeline.run_pipeline_async)
        assert "if cfg.defer_reextraction and not cfg.skip_reextraction" in src

    def test_defer_reextraction_docstring_flags_sync_only(self):
        field = _pipeline.PipelineConfig.model_fields["defer_reextraction"]
        assert "Sync-only" in (field.description or "")


# ===========================================================================
# #3 — _run_evaluation_branch uses ALL entities when focus_entity_ids is empty
# ===========================================================================

class TestEvaluationFocusIdsNoCap:
    """Empty ``focus_entity_ids`` must mean *whole cast*, not first five.

    The prior ``[:5]`` cap silently truncated the ego graph for any
    story with more than five entities, biasing the evaluator toward
    whatever five dictionary keys happened to come first.
    """

    def test_no_five_cap_in_source(self):
        src = inspect.getsource(_pipeline._run_evaluation_branch)
        # The cap line is gone; no ``[:5]`` slicing on
        # ``ws.entities.keys()``.
        assert "list(ws.entities.keys())[:5]" not in src
        # The replacement uses ``or ws.entities.keys()``.
        assert "query.focus_entity_ids or ws.entities.keys()" in src


# ===========================================================================
# #4 — _run_evaluation_branch filters vwm.history by branch world_id
# ===========================================================================

class TestEvaluationProseBranchScoped:
    """Shadow evaluation must not score factual prose, and vice versa.

    Before the round-5 fix, the prose collection loop walked
    ``vwm.history`` unfiltered, so a shadow-branch evaluation graded
    factual prose alongside its own \u2014 contaminating the scorecard
    with prose the branch never produced.
    """

    def test_source_resolves_branch_policy_before_prose_loop(self):
        src = inspect.getsource(_pipeline._run_evaluation_branch)
        policy_idx = src.index("_resolve_branch_policy")
        loop_idx = src.index("for entry in vwm.history")
        # Branch policy must be resolved BEFORE the prose collection
        # loop so the loop can filter on ``_eval_branch_id``.
        assert policy_idx < loop_idx, (
            "branch policy must be resolved before prose loop"
        )

    def test_source_filters_loop_by_world_id(self):
        src = inspect.getsource(_pipeline._run_evaluation_branch)
        # The loop body must skip entries whose world_id does not
        # match the resolved branch.
        assert "entry_world_id" in src
        assert "_eval_branch_id" in src
        assert "continue" in src


# ===========================================================================
# #5 + #6 — MCP evaluate scopes get_all_prose to branch lineage
# ===========================================================================

class TestMcpEvaluateBranchScoped:
    """The MCP ``evaluate`` tool loaded a branch-projected world state
    but then pulled the *entire project's* prose via ``get_all_prose``
    with no branch_path. A shadow evaluation was therefore scored
    against the union of every sibling branch's prose.
    """

    def test_evaluate_source_passes_branch_path(self):
        import shadow_loom_mcp.server as srv
        src = inspect.getsource(srv.evaluate)
        assert "get_lineage_to_root" in src
        assert "branch_path=branch_path" in src

    def test_evaluate_captures_ver_row_id(self):
        import shadow_loom_mcp.server as srv
        src = inspect.getsource(srv.evaluate)
        # The previous code discarded the version row id with
        # ``_``; the fix must bind it.
        assert "ws, ver_row_id = load_world_state_projected" in src


# ===========================================================================
# #5 — get_lineage_to_root helper exists and walks ancestor_id
# ===========================================================================

class TestGetLineageToRootHelper:
    def test_missing_version_returns_empty(self):
        # An id that almost certainly does not exist returns [].
        result = get_lineage_to_root(2**31 - 1)
        assert result == []

    def test_source_walks_ancestor_id(self):
        src = inspect.getsource(get_lineage_to_root)
        # The walk must climb ``ancestor_id`` and reverse so the
        # output is root \u2192 leaf (the order ``get_all_prose``
        # preserves when given a branch_path).
        assert "ancestor_id" in src
        assert "chain.reverse()" in src
        # Cycle guard.
        assert "seen" in src


# ===========================================================================
# #7 — DirectiveQuery.intensity is bounded to [0.0, 1.0]
# ===========================================================================

class TestDirectiveIntensityBounds:
    def test_intensity_above_one_rejected(self):
        with pytest.raises(ValidationError):
            DirectiveQuery(
                reasoning="r",
                target_entity_ids=["ENT_X"],
                target_effect="suspense",
                intensity=1.5,
            )

    def test_intensity_negative_rejected(self):
        with pytest.raises(ValidationError):
            DirectiveQuery(
                reasoning="r",
                target_entity_ids=["ENT_X"],
                target_effect="suspense",
                intensity=-0.1,
            )

    def test_intensity_in_bounds_accepted(self):
        q = DirectiveQuery(
            reasoning="r",
            target_entity_ids=["ENT_X"],
            target_effect="suspense",
            intensity=0.5,
        )
        assert q.intensity == 0.5


# ===========================================================================
# #8 — DirectiveQuery.target_entity_ids requires at least one id
# ===========================================================================

class TestDirectiveTargetEntityIdsRequired:
    def test_empty_target_ids_rejected(self):
        with pytest.raises(ValidationError):
            DirectiveQuery(
                reasoning="r",
                target_entity_ids=[],
                target_effect="suspense",
            )

    def test_single_target_id_accepted(self):
        q = DirectiveQuery(
            reasoning="r",
            target_entity_ids=["ENT_X"],
            target_effect="suspense",
        )
        assert q.target_entity_ids == ["ENT_X"]


# ===========================================================================
# #9 — save_version only retries unique-version IntegrityErrors
# ===========================================================================

class TestSaveVersionRetrySpecificity:
    """The retry loop must not mask non-collision IntegrityErrors
    (FK violations, NOT NULL breaches, the cross-project ancestry
    guard) behind N retries that finally raise a generic RuntimeError
    with the original cause buried.
    """

    def test_source_inspects_error_message(self):
        src = inspect.getsource(save_version)
        assert "_is_version_collision" in src
        # The check must look at both the wrapped DBAPI orig and the
        # SQLAlchemy message.
        assert "getattr(e, \"orig\", e)" in src
        # Non-collision path re-raises immediately rather than
        # continuing.
        assert "raise" in src

    def test_non_collision_integrity_error_reraised_immediately(self):
        """Simulate a non-version-uniqueness IntegrityError and
        confirm the function raises it on the first attempt instead
        of looping ``max_attempts`` times and producing a
        ``RuntimeError``.
        """
        # Build a fake session whose commit raises a non-collision
        # IntegrityError. The message intentionally lacks the
        # "version" + "unique"/"duplicate" combo that triggers retry.
        fake_session = MagicMock()
        fake_session.__enter__ = MagicMock(return_value=fake_session)
        fake_session.__exit__ = MagicMock(return_value=False)
        # NOT NULL violations don't mention "version" \u2014 must re-raise.
        fk_err = IntegrityError(
            "INSERT INTO versions ...",
            params=None,
            orig=Exception("NOT NULL constraint failed: versions.source"),
        )
        fake_session.commit.side_effect = fk_err
        # The session also needs to provide ``_next_version_number``
        # support via ``exec``; we monkeypatch the helper to short out.
        with patch("shadow_loom.db.get_session", return_value=fake_session), \
             patch("shadow_loom.db._next_version_number", return_value=1), \
             patch.object(fake_session, "get", return_value=None):
            with pytest.raises(IntegrityError):
                save_version(
                    project_id=999_999,  # bogus pid; we never hit DB
                    source="test",
                    description="d",
                    world_state_json="{}",
                    changeset_json=None,
                    prose=None,
                )
        # Crucially: commit was called *once*, not max_attempts times.
        assert fake_session.commit.call_count == 1
