"""NiceGUI UI component modules."""

# Legacy components (kept for backward compatibility)
from shadow_loom_ui.components.explorer import build_explorer
from shadow_loom_ui.components.center import build_center_panel
from shadow_loom_ui.components.chat import build_chat_panel
from shadow_loom_ui.components.topology import build_topology_drawer
from shadow_loom_ui.components.dialogs import (
    build_ingest_dialog,
    build_project_dialog,
    build_version_dialog,
)

# New multi-page components
from shadow_loom_ui.components.login import build_login_page
from shadow_loom_ui.components.dashboard import build_dashboard
from shadow_loom_ui.components.workspace import build_workspace
from shadow_loom_ui.components.story_tab import build_story_tab
from shadow_loom_ui.components.world_tab import build_world_tab
from shadow_loom_ui.components.timeline_tab import build_timeline_tab
from shadow_loom_ui.components.audit_tab import build_audit_tab
from shadow_loom_ui.components.export_tab import build_export_tab
from shadow_loom_ui.components.settings import build_settings

__all__ = [
    # Legacy
    "build_explorer",
    "build_center_panel",
    "build_chat_panel",
    "build_topology_drawer",
    "build_ingest_dialog",
    "build_project_dialog",
    "build_version_dialog",
    # New
    "build_login_page",
    "build_dashboard",
    "build_workspace",
    "build_story_tab",
    "build_world_tab",
    "build_timeline_tab",
    "build_audit_tab",
    "build_export_tab",
    "build_settings",
]
