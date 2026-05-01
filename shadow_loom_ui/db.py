# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Re-export shared DB layer from shadow_loom.db.

This shim exists so that existing ``from shadow_loom_ui.db import ...``
imports continue to work without modification.  All logic lives in
:mod:`shadow_loom.db`.
"""

from shadow_loom.db import (  # noqa: F401 — re-exports
    Base,
    UserRow,
    ProjectRow,
    VersionRow,
    ProjectMemberRow,
    ProjectStarRow,
    ApiKeyRow,
    ActivityRow,
    get_engine,
    get_session,
    init_db,
    ensure_example_user,
    ensure_local_user,
    get_example_user_id,
    upsert_user,
    get_user,
    update_user_profile,
    create_project,
    get_project,
    update_project,
    fork_project,
    delete_project,
    ProjectDeleteError,
    find_project_by_name,
    list_projects,
    list_example_projects,
    add_project_member,
    remove_project_member,
    list_project_members,
    get_user_project_role,
    toggle_star,
    is_starred,
    list_starred_projects,
    create_api_key,
    validate_api_key,
    list_api_keys,
    revoke_api_key,
    log_activity,
    get_project_activity,
    get_user_activity,
    save_version,
    get_version,
    get_version_by_id,
    get_latest_version,
    list_versions,
    get_version_tree,
    get_version_lineage,
    get_version_children,
    delete_version,
    reparent_version,
    VersionMutationError,
    list_branches,
    promote_branch,
    bookmark_version,
    label_version,
    set_active_version,
    get_active_version,
    clear_active_version,
    get_all_prose,
    search_users,
    # Backward-compatible aliases
    save_snapshot,
    load_latest_snapshot,
    load_snapshot,
)

# Alias for backward compat — old code references SnapshotRow
SnapshotRow = VersionRow
