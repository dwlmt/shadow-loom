"""Unit tests for delete_version + reparent_version DB functions."""
from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "sqlite://"

import pytest

from shadow_loom.db import (
    ProjectMemberRow,
    VersionMutationError,
    create_project,
    delete_project,
    delete_version,
    get_active_version,
    get_project,
    get_session,
    get_version_by_id,
    get_version_tree,
    init_db,
    log_activity,
    reparent_version,
    save_version,
    set_active_version,
    upsert_user,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db("sqlite://")
    yield


def _seed_user_and_project():
    user = upsert_user("local", "vt-1", "vt-user", display_name="VT User")
    proj = create_project(name="VTProj", owner_id=user.id)
    return user, proj


def _seed_chain(proj_id: int, user_id: int, n: int) -> list[int]:
    """Save a linear chain of n versions; return version row ids in order."""
    ids: list[int] = []
    parent: int | None = None
    for i in range(n):
        v = save_version(
            project_id=proj_id,
            world_state_json="{}",
            ancestor_id=parent,
            source="ingestion" if i == 0 else "pipeline",
            description=f"v{i}",
            user_id=user_id,
            version=i,
        )
        ids.append(v.id)
        parent = v.id
    return ids


# =====================================================================
# delete_version
# =====================================================================


class TestDeleteVersion:
    def test_delete_middle_rejoins_children(self):
        user, proj = _seed_user_and_project()
        v0, v1, v2 = _seed_chain(proj.id, user.id, 3)

        result = delete_version(v1, user.id)
        assert v1 in result["deleted"]
        # v2 should be re-parented onto v0
        assert result["reparented"] == {v2: v0}

        v2_row = get_version_by_id(v2)
        assert v2_row is not None
        assert v2_row.ancestor_id == v0
        # v1 is gone
        assert get_version_by_id(v1) is None

    def test_delete_leaf_succeeds_no_reparent(self):
        user, proj = _seed_user_and_project()
        v0, v1 = _seed_chain(proj.id, user.id, 2)

        result = delete_version(v1, user.id)
        assert result["deleted"] == [v1]
        assert result["reparented"] == {}
        assert get_version_by_id(v1) is None

    def test_delete_root_rejected(self):
        user, proj = _seed_user_and_project()
        v0, _ = _seed_chain(proj.id, user.id, 2)
        with pytest.raises(VersionMutationError, match="root"):
            delete_version(v0, user.id)

    def test_delete_missing_rejected(self):
        user, proj = _seed_user_and_project()
        _seed_chain(proj.id, user.id, 1)
        with pytest.raises(VersionMutationError, match="not found"):
            delete_version(999_999, user.id)

    def test_delete_non_owner_rejected(self):
        owner, proj = _seed_user_and_project()
        _v0, v1 = _seed_chain(proj.id, owner.id, 2)
        intruder = upsert_user("local", "vt-2", "intruder", display_name="X")
        with pytest.raises(PermissionError):
            delete_version(v1, intruder.id)

    def test_delete_editor_member_allowed(self):
        owner, proj = _seed_user_and_project()
        _v0, v1 = _seed_chain(proj.id, owner.id, 2)
        editor = upsert_user("local", "vt-3", "editor", display_name="E")
        with get_session() as s:
            s.add(ProjectMemberRow(
                project_id=proj.id, user_id=editor.id, role="editor",
            ))
            s.commit()
        result = delete_version(v1, editor.id)
        assert v1 in result["deleted"]

    def test_cascade_deletes_subtree(self):
        user, proj = _seed_user_and_project()
        v0, v1, v2, v3 = _seed_chain(proj.id, user.id, 4)
        # Add a sibling branch off v1 → v1b
        v1b = save_version(
            project_id=proj.id,
            world_state_json="{}",
            ancestor_id=v1,
            source="branch",
            description="v1b",
            user_id=user.id,
            version=4,
        ).id

        result = delete_version(v1, user.id, cascade=True)
        assert set(result["deleted"]) == {v1, v2, v3, v1b}
        assert result["reparented"] == {}
        # All deleted
        for vid in (v1, v2, v3, v1b):
            assert get_version_by_id(vid) is None
        # Root preserved
        assert get_version_by_id(v0) is not None


# =====================================================================
# reparent_version
# =====================================================================


class TestReparentVersion:
    def test_reparent_under_sibling(self):
        user, proj = _seed_user_and_project()
        v0, v1, v2 = _seed_chain(proj.id, user.id, 3)
        # Branch off v0
        v1b = save_version(
            project_id=proj.id,
            world_state_json="{}",
            ancestor_id=v0,
            source="branch",
            description="v1b",
            user_id=user.id,
            version=3,
        ).id

        # Move v2 under v1b instead of v1.
        assert reparent_version(v2, v1b, user.id) is True
        v2_row = get_version_by_id(v2)
        assert v2_row.ancestor_id == v1b

    def test_reparent_root_rejected(self):
        user, proj = _seed_user_and_project()
        v0, v1 = _seed_chain(proj.id, user.id, 2)
        with pytest.raises(VersionMutationError, match="root"):
            reparent_version(v0, v1, user.id)

    def test_reparent_self_rejected(self):
        user, proj = _seed_user_and_project()
        _v0, v1 = _seed_chain(proj.id, user.id, 2)
        with pytest.raises(VersionMutationError, match="own ancestor"):
            reparent_version(v1, v1, user.id)

    def test_reparent_under_descendant_rejects_cycle(self):
        user, proj = _seed_user_and_project()
        v0, v1, v2 = _seed_chain(proj.id, user.id, 3)
        # Try to move v1 under v2 (its own descendant) — would create a cycle.
        with pytest.raises(VersionMutationError, match="descendant"):
            reparent_version(v1, v2, user.id)

    def test_reparent_cross_project_rejected(self):
        user, proj = _seed_user_and_project()
        _v0, v1 = _seed_chain(proj.id, user.id, 2)
        proj2 = create_project(name="Other", owner_id=user.id)
        other_v0 = save_version(
            project_id=proj2.id,
            world_state_json="{}",
            source="ingestion",
            user_id=user.id,
            version=0,
        ).id
        with pytest.raises(VersionMutationError, match="same project"):
            reparent_version(v1, other_v0, user.id)

    def test_reparent_non_owner_rejected(self):
        owner, proj = _seed_user_and_project()
        _v0, v1 = _seed_chain(proj.id, owner.id, 2)
        intruder = upsert_user("local", "vt-4", "intruder2", display_name="X2")
        with pytest.raises(PermissionError):
            reparent_version(v1, None, intruder.id)


# =====================================================================
# Tree consistency after mutations
# =====================================================================


class TestTreeConsistency:
    def test_tree_remains_traversable_after_rejoin(self):
        user, proj = _seed_user_and_project()
        v0, v1, v2, v3 = _seed_chain(proj.id, user.id, 4)
        delete_version(v1, user.id)
        delete_version(v2, user.id)

        tree = get_version_tree(proj.id)
        # All remaining nodes reachable from root by ancestor chain
        by_id = {n["id"]: n for n in tree}
        assert v0 in by_id and v3 in by_id
        # v3's ancestor must transitively reach v0
        cur = by_id[v3]
        depth = 0
        while cur["ancestor_id"] is not None and depth < 10:
            cur = by_id[cur["ancestor_id"]]
            depth += 1
        assert cur["id"] == v0


# =====================================================================
# Active-version pointer retargeting on delete
# =====================================================================


class TestActivePointerOnDelete:
    def test_rejoin_delete_retargets_active_pointer_to_parent(self):
        user, proj = _seed_user_and_project()
        v0, v1, v2 = _seed_chain(proj.id, user.id, 3)
        set_active_version(proj.id, user.id, v1)

        delete_version(v1, user.id)

        # Pointer should now reference v0 (the parent of the deleted row),
        # not be cleared and not be left dangling.
        active = get_active_version(proj.id, user.id)
        assert active is not None
        assert active.id == v0

    def test_cascade_delete_retargets_active_pointer_to_subtree_parent(self):
        user, proj = _seed_user_and_project()
        v0, v1, v2, v3 = _seed_chain(proj.id, user.id, 4)
        # Active points at the deepest leaf inside the subtree being removed.
        set_active_version(proj.id, user.id, v3)

        delete_version(v1, user.id, cascade=True)

        active = get_active_version(proj.id, user.id)
        assert active is not None
        # v1's parent is v0, so the pointer must land on v0.
        assert active.id == v0

    def test_get_active_version_is_read_only(self):
        # With FK enforcement enabled, the pointer cannot reference a
        # missing version. Confirm get_active_version simply round-trips
        # the active row without mutating the DB on a normal read.
        user, proj = _seed_user_and_project()
        v0, v1 = _seed_chain(proj.id, user.id, 2)
        set_active_version(proj.id, user.id, v1)

        active = get_active_version(proj.id, user.id)
        assert active is not None and active.id == v1

        # Pointer row should still be present and unchanged after the read.
        from shadow_loom.db import ActiveVersionRow
        from sqlmodel import select
        with get_session() as s:
            row = s.exec(
                select(ActiveVersionRow).where(
                    ActiveVersionRow.project_id == proj.id,
                    ActiveVersionRow.user_id == user.id,
                )
            ).first()
            assert row is not None
            assert row.version_row_id == v1


# =====================================================================
# delete_project FK ordering (regression: SQLite FKs are now enforced)
# =====================================================================


class TestDeleteProjectFK:
    def test_delete_project_with_versions_activity_and_active_pointer(self):
        # Reproduces the IntegrityError seen when deleting a project
        # whose versions had ActivityRow.version_id and
        # ActiveVersionRow.version_row_id references; FK enforcement
        # rejected the version DELETE because dependent rows still
        # pointed at it.
        user, proj = _seed_user_and_project()
        v_ids = _seed_chain(proj.id, user.id, 3)

        # Active-version pointer at the deepest leaf
        set_active_version(proj.id, user.id, v_ids[-1])
        # Activity row referencing the middle version
        log_activity(
            project_id=proj.id,
            action="pipeline",
            user_id=user.id,
            summary="test activity",
            version_id=v_ids[1],
        )

        # Should not raise IntegrityError.
        assert delete_project(proj.id, user.id) is True
        assert get_project(proj.id) is None
        # All versions cascaded.
        for vid in v_ids:
            assert get_version_by_id(vid) is None
