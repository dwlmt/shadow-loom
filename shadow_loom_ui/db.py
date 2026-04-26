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
    get_engine,
    get_session,
    init_db,
    ensure_example_user,
    get_example_user_id,
    upsert_user,
    create_project,
    get_project,
    find_project_by_name,
    list_projects,
    save_version,
    get_version,
    get_version_by_id,
    get_latest_version,
    list_versions,
    get_version_tree,
    get_version_lineage,
    get_version_children,
    # Backward-compatible aliases
    save_snapshot,
    load_latest_snapshot,
    load_snapshot,
)

# Alias for backward compat — old code references SnapshotRow
SnapshotRow = VersionRow
