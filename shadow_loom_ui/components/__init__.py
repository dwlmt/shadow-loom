# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""NiceGUI UI component modules."""

from shadow_loom_ui.components.login import build_login_page
from shadow_loom_ui.components.dashboard import build_dashboard
from shadow_loom_ui.components.workspace import build_workspace
from shadow_loom_ui.components.story_tab import build_story_tab
from shadow_loom_ui.components.world_tab import build_world_tab
from shadow_loom_ui.components.causality_tab import build_causality_tab
from shadow_loom_ui.components.audit_tab import build_audit_tab
from shadow_loom_ui.components.export_tab import build_export_tab
from shadow_loom_ui.components.explorer_tab import build_explorer_tab
from shadow_loom_ui.components.version_sidebar import build_version_sidebar
from shadow_loom_ui.components.chat import build_chat_drawer
from shadow_loom_ui.components.settings import build_settings
from shadow_loom_ui.components.dialogs import (
    build_ingest_dialog,
    build_project_dialog,
    build_version_dialog,
)

__all__ = [
    "build_login_page",
    "build_dashboard",
    "build_workspace",
    "build_story_tab",
    "build_world_tab",
    "build_causality_tab",
    "build_audit_tab",
    "build_export_tab",
    "build_explorer_tab",
    "build_version_sidebar",
    "build_chat_drawer",
    "build_settings",
    "build_ingest_dialog",
    "build_project_dialog",
    "build_version_dialog",
]
