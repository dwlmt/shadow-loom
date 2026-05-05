# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Click-resolution tests for the version sidebar.

Covers the contract between
:func:`shadow_loom_ui.viz_helpers.version_tree_to_echart_data` (which
decorates shadow nodes with a ``" — {branch_label}"`` suffix and bakes
``_vid`` into each node payload) and
:func:`shadow_loom_ui.components.version_sidebar._resolve_clicked_version`
(which reads that payload back out of the ECharts click event).

Regressions guarded:

* Reading ``_vid`` from the top-level ``args`` dict instead of
  ``args["data"]`` made shadow-branch clicks silently no-op (the legacy
  name-fallback couldn't match the decorated label), so Story / Audit
  / Source kept rendering the previous branch.
* Synthetic multi-root wrapper nodes have no ``_vid`` and no matching
  row in ``tree_data`` and must resolve to ``None`` (don't load
  anything).
"""

from __future__ import annotations

from shadow_loom_ui.components.version_sidebar import (
    _resolve_clicked_version,
)
from shadow_loom_ui.viz_helpers import version_tree_to_echart_data


def _tree_data() -> list[dict]:
    """Tree: v0 (factual root) → v1 (factual) → v2 (shadow what-if)."""
    return [
        {"id": 100, "version": 0, "ancestor_id": None,
         "world_id": "factual", "branch_label": None, "source": "ingest"},
        {"id": 101, "version": 1, "ancestor_id": 100,
         "world_id": "factual", "branch_label": None, "source": "continue"},
        {"id": 102, "version": 2, "ancestor_id": 101,
         "world_id": "shadow", "branch_label": "What if Duncan lived",
         "source": "what_if"},
    ]


def _node_for_vid(root: dict, vid: int) -> dict:
    """Walk the ECharts tree and return the node with ``_vid == vid``."""
    if root.get("_vid") == vid:
        return root
    for child in root.get("children") or []:
        found = _node_for_vid(child, vid)
        if found is not None:
            return found
    return None  # type: ignore[return-value]


def test_factual_root_click_resolves_to_v0_row():
    tree_data = _tree_data()
    root_node = version_tree_to_echart_data(tree_data, current_version_id=101)
    # ECharts payload: top-level event with the clicked node under "data".
    args = {"name": root_node["name"], "data": root_node}
    v = _resolve_clicked_version(args, tree_data)
    assert v is not None and v["id"] == 100 and v["version"] == 0


def test_factual_mid_branch_click_resolves():
    tree_data = _tree_data()
    root = version_tree_to_echart_data(tree_data, current_version_id=100)
    v1_node = _node_for_vid(root, 101)
    args = {"name": v1_node["name"], "data": v1_node}
    v = _resolve_clicked_version(args, tree_data)
    assert v is not None and v["id"] == 101


def test_shadow_branch_click_resolves_despite_label_suffix():
    """The shadow node's ``name`` is ``"v2 — What if Duncan lived"`` —
    label-only matching would fail. ``_vid`` must be read from
    ``args["data"]`` for the click to land."""
    tree_data = _tree_data()
    root = version_tree_to_echart_data(tree_data, current_version_id=100)
    shadow_node = _node_for_vid(root, 102)
    assert "—" in shadow_node["name"], (
        "test setup: shadow node should have a decorated label"
    )
    args = {"name": shadow_node["name"], "data": shadow_node}
    v = _resolve_clicked_version(args, tree_data)
    assert v is not None and v["id"] == 102 and v["world_id"] == "shadow"


def test_synthetic_multi_root_wrapper_resolves_to_none():
    """When two root versions exist, ``version_tree_to_echart_data``
    wraps them in a synthetic ``{"name": "root", ...}`` with no ``_vid``.
    Clicking it should be a no-op, not a crash and not a stray load."""
    tree_data = [
        {"id": 200, "version": 0, "ancestor_id": None,
         "world_id": "factual", "branch_label": None, "source": "ingest"},
        {"id": 201, "version": 0, "ancestor_id": None,
         "world_id": "factual", "branch_label": None, "source": "ingest"},
    ]
    root = version_tree_to_echart_data(tree_data)
    assert root.get("name") == "root" and "_vid" not in root
    args = {"name": "root", "data": root}
    assert _resolve_clicked_version(args, tree_data) is None


def test_empty_args_returns_none():
    assert _resolve_clicked_version({}, _tree_data()) is None


def test_legacy_label_fallback_for_factual_rows():
    """If ``data`` is missing but ``name`` matches an undecorated
    factual label, the fallback path should still resolve. Shadow rows
    intentionally do NOT match this fallback (their label is decorated)."""
    tree_data = _tree_data()
    assert _resolve_clicked_version(
        {"name": "v1"}, tree_data,
    )["id"] == 101
    # Decorated shadow label is not matched by the fallback:
    assert _resolve_clicked_version(
        {"name": "v2 \u2014 What if Duncan lived"}, tree_data,
    ) is None
