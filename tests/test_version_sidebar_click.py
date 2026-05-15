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
    """If ``data`` is missing but ``name`` matches the version label,
    the fallback path resolves both undecorated factual labels
    (``"v1"``) and decorated shadow labels (``"v2 — branch"``).

    The decorated-shadow case matters because some ECharts payload
    shapes (synthetic events, future versions, custom themes) may
    deliver only the ``name`` field at the top level, with no nested
    ``data`` dict carrying ``_vid``.
    """
    tree_data = _tree_data()
    assert _resolve_clicked_version(
        {"name": "v1"}, tree_data,
    )["id"] == 101
    # Decorated shadow label IS now matched by the fallback so a
    # name-only payload still lands on the right row.
    assert _resolve_clicked_version(
        {"name": "v2 \u2014 What if Duncan lived"}, tree_data,
    )["id"] == 102


def test_resolves_when_args_is_list_of_payload():
    """NiceGUI delivers unfiltered chart events as a *list* of Vue
    ``$emit`` arguments. ECharts' ``componentClick`` carries a single
    payload, so ``e.args`` arrives as ``[payload_dict]``. The previous
    handler did ``args = e.args if isinstance(e.args, dict) else {}``
    and silently no-op'd, which is exactly the regression that kept
    Story / World / Audit pinned to the previous version after a
    branch click."""
    tree_data = _tree_data()
    root = version_tree_to_echart_data(tree_data, current_version_id=100)
    shadow_node = _node_for_vid(root, 102)
    payload = {"name": shadow_node["name"], "data": shadow_node}
    v = _resolve_clicked_version([payload], tree_data)
    assert v is not None and v["id"] == 102


def test_resolves_when_args_is_json_string():
    """Defensive: if NiceGUI delivers the ``stringifyEventArgs``
    payload un-decoded (a JSON string), we should still resolve."""
    import json as _json
    tree_data = _tree_data()
    root = version_tree_to_echart_data(tree_data, current_version_id=100)
    v1_node = _node_for_vid(root, 101)
    payload = {"name": v1_node["name"], "data": v1_node}
    v = _resolve_clicked_version(_json.dumps(payload), tree_data)
    assert v is not None and v["id"] == 101


def test_resolves_when_data_is_json_string():
    """``args["data"]`` may itself be a JSON-string when an args
    filter is in effect. Tolerate it."""
    import json as _json
    tree_data = _tree_data()
    root = version_tree_to_echart_data(tree_data, current_version_id=100)
    v1_node = _node_for_vid(root, 101)
    payload = {"name": v1_node["name"], "data": _json.dumps(v1_node)}
    v = _resolve_clicked_version(payload, tree_data)
    assert v is not None and v["id"] == 101


def test_render_version_tree_subscribes_to_componentClick():
    """``render_version_tree`` must listen for ``componentClick``,
    not bare ``click``.

    NiceGUI's ECharts Vue wrapper re-emits ECharts' internal ``"click"``
    as a Vue ``componentClick`` event (see
    ``nicegui/elements/echart/echart.js``). A Python listener for
    ``"click"`` registers as Vue ``onClick``, which the wrapper never
    fires for chart-node interactions — so before this fix every
    branch click in the version sidebar was a silent no-op and Story
    / World / Audit / Reasoning all stayed pinned to the previously
    loaded version. Guard the wiring so a future "simplification" back
    to ``"click"`` regresses loudly.
    """
    from shadow_loom_ui.viz import render_version_tree

    tree_data = _tree_data()
    chart = render_version_tree(
        tree_data,
        current_version_id=101,
        on_click=lambda e: None,
    )
    listener_types = {
        lst.type for lst in chart._event_listeners.values()
    }
    assert "componentClick" in listener_types, (
        f"render_version_tree must subscribe to ECharts' "
        f"componentClick (got {listener_types!r})"
    )
    assert "click" not in listener_types, (
        "render_version_tree must NOT use bare 'click' — that Vue "
        "event never fires for ECharts node interactions"
    )
