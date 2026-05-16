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


def _empty_world_state():
    """Build a minimal valid ``WorldStateV1`` for resolver tests."""
    from tests.conftest import make_empty_world_state

    return make_empty_world_state()


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

    def test_auto_inherits_active_shadow_branch(self):
        """Non-counterfactual queries on an active shadow branch stay on that
        branch under auto policy and inherit its label."""
        from shadow_loom.extract_graph import (
            VersionedWorldModel,
            WorldModelVersion,
        )

        vwm = VersionedWorldModel.from_world_state(_empty_world_state())
        # Append a shadow head onto the DAG.
        vwm.history.append(WorldModelVersion(
            version=1,
            timestamp="2026-05-05T00:00:00Z",
            source="pipeline",
            description="shadow fork",
            world_id="shadow",
            branch_label="What if Macbeth refused",
        ))

        q = InterventionQuery(
            original_query="Have Banquo confess.",
            interventions={"banquo.confessed": True},
        )
        cfg = PipelineConfig()  # auto
        world_id, label = _resolve_branch_policy(q, cfg, vwm)
        assert world_id == "shadow"
        assert label == "What if Macbeth refused"

    def test_auto_counterfactual_forks_off_shadow_with_new_label(self):
        """A counterfactual launched from an active shadow branch creates a
        new fork — its label derives from the new query, NOT the parent."""
        from shadow_loom.extract_graph import (
            VersionedWorldModel,
            WorldModelVersion,
        )

        vwm = VersionedWorldModel.from_world_state(_empty_world_state())
        vwm.history.append(WorldModelVersion(
            version=1,
            timestamp="2026-05-05T00:00:00Z",
            source="pipeline",
            description="parent shadow fork",
            world_id="shadow",
            branch_label="Parent fork label",
        ))

        q = CounterfactualQuery(
            original_query="What if Banquo had escaped earlier?",
            historical_interventions={"EVT_BANQUO_DEATH": "escaped"},
            evidence_node_ids=["banquo"],
        )
        cfg = PipelineConfig()
        world_id, label = _resolve_branch_policy(q, cfg, vwm)
        assert world_id == "shadow"
        assert label and "Banquo" in label
        assert label != "Parent fork label"

    def test_auto_inherits_factual_when_active_branch_is_factual(self):
        from shadow_loom.extract_graph import VersionedWorldModel

        vwm = VersionedWorldModel.from_world_state(_empty_world_state())
        # default v0 entry is factual
        q = InterventionQuery(
            original_query="x",
            interventions={"a.b": 1},
        )
        cfg = PipelineConfig()
        world_id, _ = _resolve_branch_policy(q, cfg, vwm)
        assert world_id == "factual"

    def test_shadow_fork_synthesizes_label_when_query_lacks_nl(self):
        """Regression: a programmatically constructed shadow query with
        no ``original_query`` / ``description`` must still receive a
        non-empty ``branch_label``. An empty label silently drops
        shadow proposition truth commits at the merge boundary and
        makes ``projected_for_branch`` a no-op, so downstream
        interrogation reads factual baseline and contradicts the
        rendered counterfactual prose (the Mrs Coady / dogs regression
        in *A Fish Called Wanda*)."""
        q = CounterfactualQuery(
            original_query=None,
            historical_interventions={"EVT_DOG_DEATH": "averted"},
            evidence_node_ids=["mrs_coady"],
        )
        cfg = PipelineConfig()
        world_id, label = _resolve_branch_policy(q, cfg)
        assert world_id == "shadow"
        assert label, "shadow fork must never have empty branch_label"
        assert "counterfactual" in label

    def test_shadow_override_synthesizes_label_when_query_lacks_nl(self):
        """Same invariant under explicit policy='shadow' on a non-CF query."""
        q = InterventionQuery(
            original_query=None,
            interventions={"x.y": 1},
        )
        cfg = PipelineConfig(branch_policy="shadow")
        world_id, label = _resolve_branch_policy(q, cfg)
        assert world_id == "shadow"
        assert label, "shadow fork must never have empty branch_label"

    def test_merge_synthesizes_label_for_orphan_shadow_call(self, caplog):
        """Defence-in-depth: a direct ``merge(world_id='shadow', branch_label=None)``
        call (UI save_version on a legacy label-less head, MCP, seeder)
        must synthesize a fallback label and warn — never propagate the
        corruption that silently demotes shadow writes."""
        import logging

        from shadow_loom.extract_graph import VersionedWorldModel
        from shadow_loom.ingestion import ChunkTopology

        vwm = VersionedWorldModel.from_world_state(_empty_world_state())
        topology = ChunkTopology()  # empty merge is a valid no-op
        with caplog.at_level(logging.WARNING, logger="shadow_loom.extract_graph"):
            vwm_next = vwm.merge(
                topology,
                source="test_orphan",
                world_id="shadow",
                branch_label=None,
            )
        head = vwm_next.history[-1]
        assert head.world_id == "shadow"
        assert head.branch_label, "merge must synthesize a label"
        assert "branch_label" in caplog.text


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

        # The factual mainline has advanced (v0 → v1) past the
        # shadow's fork point (v0), so a vanilla promote must reject
        # to protect those advances; force=True acknowledges the
        # overwrite. See ``test_promote_diverged_rejected_without_force``.
        promoted = promote_branch(
            s1.id, user_id=uid, description="canon!", force=True,
        )

        assert promoted.world_id == "factual"
        assert promoted.ancestor_id == v1.id  # off current factual head
        assert promoted.source == "promote_branch"
        assert "canon" in (promoted.description or "")

        branches = list_branches(pid)
        factual = next(b for b in branches if b["world_id"] == "factual")
        assert factual["head_version_row_id"] == promoted.id
        assert factual["version_count"] == 3  # v0, v1, promoted

    def test_promote_non_diverged_succeeds_without_force(self):
        # Shadow forks off the current factual head with no
        # subsequent factual advances; promote is safe and must not
        # require ``force``.
        uid, pid = _seed_project()
        v0 = _save(pid, uid, 0)
        s1 = _save(
            pid, uid, 1, ancestor_id=v0.id,
            world_id="shadow", branch_label="alt",
            source="counterfactual",
        )

        promoted = promote_branch(s1.id, user_id=uid)

        assert promoted.world_id == "factual"
        assert promoted.ancestor_id == v0.id

    def test_promote_diverged_rejected_without_force(self):
        # Factual mainline advanced past the fork point: refuse to
        # overwrite without an explicit force.
        uid, pid = _seed_project()
        v0 = _save(pid, uid, 0)
        _save(pid, uid, 1, ancestor_id=v0.id)  # factual advance
        s1 = _save(
            pid, uid, 2, ancestor_id=v0.id,
            world_id="shadow", branch_label="alt",
            source="counterfactual",
        )
        with pytest.raises(VersionMutationError, match="force=True"):
            promote_branch(s1.id, user_id=uid)

    def test_promote_factual_version_rejected(self):
        uid, pid = _seed_project()
        v0 = _save(pid, uid, 0)
        with pytest.raises(VersionMutationError):
            promote_branch(v0.id, user_id=uid)

    def test_promote_unknown_version_rejected(self):
        uid, _ = _seed_project()
        with pytest.raises(VersionMutationError):
            promote_branch(99999, user_id=uid)


# =====================================================================
# AppState-level: DB-load must carry branch identity into the in-memory
# VersionedWorldModel so the next pipeline run inherits the right
# world_id / branch_label and prose-continuity walks the real lineage.
# =====================================================================


class TestAppStateBranchRehydration:
    def _ws_json(self) -> str:
        from tests.conftest import make_empty_world_state
        return make_empty_world_state().model_dump_json()

    def _seed_factual_then_shadow(self):
        uid, pid = _seed_project()
        ws_json = self._ws_json()
        v0 = save_version(
            project_id=pid, world_state_json=ws_json,
            version=0, source="ingestion", description="root",
            user_id=uid, world_id="factual",
        )
        s1 = save_version(
            project_id=pid, world_state_json=ws_json,
            version=1, source="counterfactual",
            description="alt fork",
            user_id=uid, ancestor_id=v0.id,
            world_id="shadow", branch_label="What-if Duncan lived",
            prose="Shadow prose chunk.",
        )
        return uid, pid, v0, s1

    def test_load_db_version_rehydrates_shadow_identity(self):
        from shadow_loom.models import WorldStateV1
        from shadow_loom_ui.state import AppState

        uid, pid, v0, s1 = self._seed_factual_then_shadow()
        state = AppState()
        state.user_id = uid
        state.project_id = pid

        ws = WorldStateV1.model_validate_json(s1.world_state_json)
        state.load_db_version(ws, s1.id, version_number=s1.version)

        assert state.versioned_model is not None
        latest = state.versioned_model.history[-1]
        # Without the rehydration fix this would be ("factual", None)
        # — the synthetic v0 from ``from_world_state``.
        assert latest.world_id == "shadow"
        assert latest.branch_label == "What-if Duncan lived"
        # Lineage carries both rows root → head.
        assert len(state.versioned_model.history) == 2
        assert state.versioned_model.history[0].world_id == "factual"

    def test_load_project_rehydrates_shadow_identity(self):
        from shadow_loom.models import WorldStateV1
        from shadow_loom_ui.state import AppState

        uid, pid, _, s1 = self._seed_factual_then_shadow()
        state = AppState()
        state.user_id = uid

        ws = WorldStateV1.model_validate_json(s1.world_state_json)
        state.load_project(
            project_id=pid,
            project_name="BranchTest",
            world_state=ws,
            version_row_id=s1.id,
        )

        latest = state.versioned_model.history[-1]
        assert latest.world_id == "shadow"
        assert latest.branch_label == "What-if Duncan lived"

    def test_rehydrated_history_drives_branch_policy(self):
        # End-to-end of the bug: after loading a shadow head from DB,
        # ``_resolve_branch_policy`` under "auto" must inherit shadow
        # rather than silently re-tagging the next write to factual.
        from shadow_loom.models import WorldStateV1
        from shadow_loom_ui.state import AppState

        uid, pid, _, s1 = self._seed_factual_then_shadow()
        state = AppState()
        state.user_id = uid
        state.project_id = pid
        ws = WorldStateV1.model_validate_json(s1.world_state_json)
        state.load_db_version(ws, s1.id, version_number=s1.version)

        q = ManualEditQuery(
            original_query="Tweak prose.",
            description="manual",
            edited_prose="The dagger sat on the table.",
        )
        cfg = PipelineConfig()  # auto
        world_id, label = _resolve_branch_policy(
            q, cfg, state.versioned_model,
        )
        assert world_id == "shadow"
        assert label == "What-if Duncan lived"

    def test_rehydrated_history_supports_prose_continuity(self):
        # ``_gather_preceding_prose`` consults vwm.history; without
        # rehydration the loaded shadow's prose (and any shadow
        # ancestors) would be invisible to continuity prompts.
        from shadow_loom.models import WorldStateV1
        from shadow_loom.pipeline import _gather_preceding_prose
        from shadow_loom_ui.state import AppState

        uid, pid, _, s1 = self._seed_factual_then_shadow()
        state = AppState()
        state.user_id = uid
        state.project_id = pid
        ws = WorldStateV1.model_validate_json(s1.world_state_json)
        state.load_db_version(ws, s1.id, version_number=s1.version)

        prose = _gather_preceding_prose(
            state.versioned_model,
            branch_world_id="shadow",
            branch_label="What-if Duncan lived",
        )
        assert prose is not None
        assert "Shadow prose chunk." in prose


# =====================================================================
# UI-side raw vs projected world-state plumbing (Phase 5 — May 2026).
#
# After ``load_db_version`` / ``load_project`` ``state.world_state``
# is the AMWN-projected view (shadow clones layered into ``entities``
# so panel reads see the do(\u00b7)-modified world). That view is
# UNSAFE to serialize: dumping it conflates the factual baseline with
# the shadow clones, and a reload overwrites the factual entry for
# every cloned id with the clone (factual baseline LOST). Every save
# / export / JSON-editor path must therefore route through
# ``state.raw_world_state`` and pass the loaded row's branch identity
# (``state.head_branch()``) to ``save_version`` so the new row stays
# on the same shadow branch.
# =====================================================================


class TestAppStateRawWorldStateAccessor:
    def _seed_factual_then_shadow(self):
        from copy import deepcopy
        from shadow_loom.models import WorldStateV1
        from tests.conftest import make_empty_world_state
        uid, pid = _seed_project()
        base = make_empty_world_state()
        # Build a shadow row whose JSON has a populated
        # ``shadow_entities`` sidecar entry that diverges from the
        # factual baseline. We bypass the merge engine and write the
        # JSON directly so the test isolates the raw-vs-projected
        # plumbing from the merge logic.
        from shadow_loom.models import Entity
        ent = Entity(
            id="ENT_X", name="X",
            location_id="LOC_VOID", status="healthy",
            traits={}, beliefs=[],
        )
        base.entities["ENT_X"] = ent
        clone = deepcopy(ent)
        clone.world_id = "shadow"
        clone.status = "injured"
        base.shadow_entities = {"cf": {"ENT_X": clone}}
        v0 = save_version(
            project_id=pid, world_state_json=base.model_dump_json(),
            version=0, source="ingestion", description="root",
            user_id=uid, world_id="factual",
        )
        s1 = save_version(
            project_id=pid, world_state_json=base.model_dump_json(),
            version=1, source="counterfactual",
            description="alt fork",
            user_id=uid, ancestor_id=v0.id,
            world_id="shadow", branch_label="cf",
            prose="shadow",
        )
        return uid, pid, base, s1

    def test_raw_world_state_returns_unprojected_snapshot(self):
        from shadow_loom.models import WorldStateV1
        from shadow_loom_ui.state import AppState

        uid, pid, base, s1 = self._seed_factual_then_shadow()
        state = AppState()
        state.user_id = uid
        state.project_id = pid
        ws = WorldStateV1.model_validate_json(s1.world_state_json)
        state.load_db_version(ws, s1.id, version_number=s1.version)

        # state.world_state is the PROJECTED view — entities[ENT_X]
        # is the shadow clone.
        assert state.world_state is not None
        assert (
            state.world_state.entities["ENT_X"].status
            == "injured"
        )
        # state.raw_world_state is the un-projected snapshot —
        # entities[ENT_X] is the factual baseline, the sidecar still
        # holds the shadow clone.
        raw = state.raw_world_state
        assert raw is not None
        assert raw.entities["ENT_X"].status == "healthy"
        assert (
            raw.shadow_entities["cf"]["ENT_X"].status
            == "injured"
        )

    def test_to_json_round_trips_factual_baseline(self):
        """Dumping ``to_json`` on a loaded shadow row must preserve
        the factual baseline for cloned ids. Before the fix this
        dumped the projected view and the factual baseline for any
        cloned entity was lost on the next reload.
        """
        from shadow_loom.models import WorldStateV1
        from shadow_loom_ui.state import AppState

        uid, pid, _, s1 = self._seed_factual_then_shadow()
        state = AppState()
        state.user_id = uid
        state.project_id = pid
        ws = WorldStateV1.model_validate_json(s1.world_state_json)
        state.load_db_version(ws, s1.id, version_number=s1.version)

        round_tripped = WorldStateV1.model_validate_json(state.to_json())
        assert round_tripped.entities["ENT_X"].status == "healthy"
        assert (
            round_tripped.shadow_entities["cf"]["ENT_X"].status
            == "injured"
        )

    def test_head_branch_reports_loaded_row_identity(self):
        from shadow_loom.models import WorldStateV1
        from shadow_loom_ui.state import AppState

        uid, pid, _, s1 = self._seed_factual_then_shadow()
        state = AppState()
        state.user_id = uid
        state.project_id = pid
        ws = WorldStateV1.model_validate_json(s1.world_state_json)
        state.load_db_version(ws, s1.id, version_number=s1.version)

        world_id, label = state.head_branch()
        assert world_id == "shadow"
        assert label == "cf"

    def test_patch_world_state_preserves_branch_and_baseline(self):
        """End-to-end regression for the UI patch flow: a no-op
        patch on a shadow row must (1) persist as shadow with the
        same branch_label and (2) round-trip the factual baseline +
        sidecar intact.
        """
        from shadow_loom.models import WorldStateV1
        from shadow_loom.ingestion import WorldStatePatch
        from shadow_loom_ui.db import get_version_by_id
        from shadow_loom_ui.state import AppState

        uid, pid, _, s1 = self._seed_factual_then_shadow()
        state = AppState()
        state.user_id = uid
        state.project_id = pid
        state.current_version_row_id = s1.id
        ws = WorldStateV1.model_validate_json(s1.world_state_json)
        state.load_db_version(ws, s1.id, version_number=s1.version)

        ok, _changes = state.apply_world_state_patch(
            WorldStatePatch(notes="no-op"),
            description="regression no-op",
        )
        assert ok

        new_row = get_version_by_id(state.current_version_row_id)
        assert new_row is not None
        # (1) Branch identity preserved.
        assert new_row.world_id == "shadow"
        assert new_row.branch_label == "cf"
        # (2) Factual baseline + sidecar both intact on the new row.
        new_ws = WorldStateV1.model_validate_json(new_row.world_state_json)
        assert new_ws.entities["ENT_X"].status == "healthy"
        assert (
            new_ws.shadow_entities["cf"]["ENT_X"].status
            == "injured"
        )
