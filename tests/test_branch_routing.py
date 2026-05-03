# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for AMWN branch routing — Story-integration plan, Step 8.

Covers the three sliceable branch behaviours that don't require running
the full LLM pipeline:

* :func:`_resolve_branch_policy` — counterfactuals fork into shadow,
  interventions/manual edits stay on the factual mainline, and
  ``branch_policy`` overrides take precedence.
* :func:`db.list_branches` — walks the version DAG and returns one
  summary per branch (factual mainline + each shadow fork).
* :func:`db.promote_branch` — copies a shadow version onto a new
  factual VersionRow whose ``ancestor_id`` is the current factual head.
"""

from __future__ import annotations

import pytest

from shadow_loom.db import (
    create_project,
    init_db,
    list_branches,
    promote_branch,
    save_version,
    upsert_user,
    VersionMutationError,
)
from shadow_loom.pipeline import PipelineConfig, _resolve_branch_policy
from shadow_loom.query_models import (
    CounterfactualQuery,
    InterventionQuery,
    ManualEditQuery,
)


# =====================================================================
# Pipeline branch-policy resolution
# =====================================================================


class TestResolveBranchPolicy:
    """``_resolve_branch_policy`` is the single source of truth for
    where a pipeline run lands in the AMWN DAG."""

    def test_counterfactual_auto_routes_to_shadow(self):
        q = CounterfactualQuery(
            original_query="What if Macbeth refused to kill Duncan?",
            historical_interventions={"EVT_KILL_DUNCAN": "refused"},
            evidence_node_ids=["macbeth"],
        )
        cfg = PipelineConfig()  # branch_policy="auto" default
        world_id, label = _resolve_branch_policy(q, cfg)
        assert world_id == "shadow"
        assert label and "Macbeth" in label

    def test_intervention_auto_stays_factual(self):
        q = InterventionQuery(
            original_query="Have Macbeth confess to Banquo.",
            interventions={"macbeth.confessed": True},
        )
        cfg = PipelineConfig()
        world_id, _ = _resolve_branch_policy(q, cfg)
        assert world_id == "factual"

    def test_manual_edit_auto_stays_factual(self):
        q = ManualEditQuery(
            original_query="edit",
            description="Tweak Banquo's age.",
            edited_prose="Banquo, now sixty, paced the hall.",
        )
        cfg = PipelineConfig()
        world_id, _ = _resolve_branch_policy(q, cfg)
        assert world_id == "factual"

    def test_mainline_override_forces_factual(self):
        """Even a counterfactual lands on factual when policy demands."""
        q = CounterfactualQuery(
            original_query="What if Duncan survived?",
            historical_interventions={"EVT_KILL_DUNCAN": "failed"},
            evidence_node_ids=["duncan"],
        )
        cfg = PipelineConfig(branch_policy="mainline")
        world_id, _ = _resolve_branch_policy(q, cfg)
        assert world_id == "factual"

    def test_shadow_override_forks_intervention(self):
        q = InterventionQuery(
            original_query="Have Banquo flee earlier.",
            interventions={"banquo.location": "forest"},
        )
        cfg = PipelineConfig(branch_policy="shadow")
        world_id, label = _resolve_branch_policy(q, cfg)
        assert world_id == "shadow"
        assert label  # non-empty


# =====================================================================
# DB-level branch DAG queries
# =====================================================================


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db("sqlite://")
    yield


def _seed_project() -> tuple[int, int]:
    user = upsert_user(
        "local", "branch-test", "branchuser", email="b@example.com",
    )
    proj = create_project(
        name="BranchTest", owner_id=user.id, description="",
    )
    return user.id, proj.id


def _save(project_id, user_id, version, *, ancestor_id=None,
          world_id="factual", branch_label=None, source="test"):
    return save_version(
        project_id=project_id,
        world_state_json="{}",
        version=version,
        source=source,
        description=f"v{version}",
        user_id=user_id,
        ancestor_id=ancestor_id,
        world_id=world_id,
        branch_label=branch_label,
    )


class TestListBranches:
    def test_factual_only_yields_one_branch(self):
        uid, pid = _seed_project()
        v0 = _save(pid, uid, 0)
        v1 = _save(pid, uid, 1, ancestor_id=v0.id)
        v2 = _save(pid, uid, 2, ancestor_id=v1.id)

        branches = list_branches(pid)
        assert len(branches) == 1
        b = branches[0]
        assert b["world_id"] == "factual"
        assert b["root_version_row_id"] == v0.id
        assert b["head_version_row_id"] == v2.id
        assert b["version_count"] == 3

    def test_shadow_fork_appears_as_separate_branch(self):
        uid, pid = _seed_project()
        v0 = _save(pid, uid, 0)
        v1 = _save(pid, uid, 1, ancestor_id=v0.id)
        # Shadow fork off v0
        s1 = _save(
            pid, uid, 2, ancestor_id=v0.id,
            world_id="shadow", branch_label="What-if Duncan lived",
            source="counterfactual",
        )
        s2 = _save(
            pid, uid, 3, ancestor_id=s1.id,
            world_id="shadow", branch_label="What-if Duncan lived",
            source="counterfactual",
        )

        branches = list_branches(pid)
        assert len(branches) == 2

        factual = next(b for b in branches if b["world_id"] == "factual")
        shadow = next(b for b in branches if b["world_id"] == "shadow")

        assert factual["root_version_row_id"] == v0.id
        assert factual["head_version_row_id"] == v1.id

        assert shadow["root_version_row_id"] == s1.id
        assert shadow["root_ancestor_id"] == v0.id  # fork point
        assert shadow["head_version_row_id"] == s2.id
        assert shadow["branch_label"] == "What-if Duncan lived"
        assert shadow["version_count"] == 2


class TestPromoteBranch:
    def test_promote_appends_factual_version_from_shadow(self):
        uid, pid = _seed_project()
        v0 = _save(pid, uid, 0)
        v1 = _save(pid, uid, 1, ancestor_id=v0.id)
        s1 = _save(
            pid, uid, 2, ancestor_id=v0.id,
            world_id="shadow", branch_label="alt",
            source="counterfactual",
        )

        promoted = promote_branch(s1.id, user_id=uid, description="canon!")

        assert promoted.world_id == "factual"
        assert promoted.ancestor_id == v1.id  # off current factual head
        assert promoted.source == "promote_branch"
        assert "canon" in (promoted.description or "")

        branches = list_branches(pid)
        factual = next(b for b in branches if b["world_id"] == "factual")
        assert factual["head_version_row_id"] == promoted.id
        assert factual["version_count"] == 3  # v0, v1, promoted

    def test_promote_factual_version_rejected(self):
        uid, pid = _seed_project()
        v0 = _save(pid, uid, 0)
        with pytest.raises(VersionMutationError):
            promote_branch(v0.id, user_id=uid)

    def test_promote_unknown_version_rejected(self):
        uid, _ = _seed_project()
        with pytest.raises(VersionMutationError):
            promote_branch(99999, user_id=uid)
