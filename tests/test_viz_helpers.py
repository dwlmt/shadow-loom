# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for new viz_helpers (Sankey DAG safety + fabula snapshot)."""
from __future__ import annotations

from example_worlds.macbeth import world_state as macbeth_ws

from shadow_loom_ui.viz_helpers import (
    fabula_time_bounds,
    snapshot_world_at,
    ws_to_entity_rows,
    ws_to_event_calendar_data,
    ws_to_event_calendar_rows,
    ws_to_event_rows,
    ws_to_object_rows,
    ws_to_polar_event_data,
    ws_to_polar_event_rows,
    ws_to_sankey_data,
    ws_to_sunburst_data,
    ws_to_trait_boxplot_data,
    ws_to_trait_stats_rows,
    ws_to_treemap_data,
    ws_to_world_trait_rows,
)


def test_fabula_time_bounds_macbeth():
    tmin, tmax = fabula_time_bounds(macbeth_ws)
    assert tmin <= tmax
    # Macbeth seed has events
    assert tmax > tmin


def test_sankey_is_acyclic():
    nodes, links = ws_to_sankey_data(macbeth_ws)
    assert nodes, "expected non-empty sankey nodes"
    assert links, "expected non-empty sankey links"

    # Every node must have a unique name (sankey matches by name).
    names = [n["name"] for n in nodes]
    assert len(names) == len(set(names))

    # Build adjacency and assert no cycles via DFS.
    adj: dict[str, list[str]] = {}
    for l in links:
        adj.setdefault(l["source"], []).append(l["target"])

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in names}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for nxt in adj.get(node, ()):
            if color.get(nxt) == GRAY:
                return False  # back-edge → cycle
            if color.get(nxt) == WHITE and not visit(nxt):
                return False
        color[node] = BLACK
        return True

    for n in names:
        if color[n] == WHITE:
            assert visit(n), f"sankey graph has a cycle starting at {n}"


def test_snapshot_world_at_filters_events_and_replays_state():
    tmin, tmax = fabula_time_bounds(macbeth_ws)
    mid = (tmin + tmax) // 2

    snap = snapshot_world_at(macbeth_ws, mid)

    # Events past the cursor are dropped
    assert all(evt.fabula_time <= mid for evt in snap.events)
    assert len(snap.events) <= len(macbeth_ws.events)

    # Entity set is preserved (structural)
    assert set(snap.entities.keys()) == set(macbeth_ws.entities.keys())

    # Original world state is unchanged (deep copy semantics)
    orig_evt_count = len(macbeth_ws.events)
    snap2 = snapshot_world_at(macbeth_ws, tmin)
    assert len(macbeth_ws.events) == orig_evt_count
    assert len(snap2.events) <= len(snap.events)


def test_snapshot_world_at_max_returns_all_events():
    _, tmax = fabula_time_bounds(macbeth_ws)
    snap = snapshot_world_at(macbeth_ws, tmax)
    assert len(snap.events) == len(macbeth_ws.events)


# =====================================================================
# Sunburst hierarchy
# =====================================================================


def test_sunburst_has_locations_and_traits():
    root = ws_to_sunburst_data(macbeth_ws)
    assert root["name"] == "World"
    assert root["children"], "expected non-empty sunburst children"

    # At least one location should contain at least one entity child.
    loc_with_entity = False
    for loc in root["children"]:
        if loc.get("children"):
            for child in loc["children"]:
                # Trait children are leaves under entity nodes.
                if child.get("children"):
                    loc_with_entity = True
                    break
        if loc_with_entity:
            break
    assert loc_with_entity, "expected at least one location → entity → trait nesting"

    # Every node must carry a numeric ``value`` so ECharts can size it.
    def _walk(node: dict) -> None:
        assert isinstance(node.get("value"), (int, float))
        assert node.get("value", 0) >= 1
        for c in node.get("children", []) or []:
            _walk(c)

    for loc in root["children"]:
        _walk(loc)


def test_sunburst_color_rotation_is_per_location():
    root = ws_to_sunburst_data(macbeth_ws)
    locs = [c for c in root["children"] if c.get("children")]
    if len(locs) >= 2:
        # Adjacent locations must use distinct colours (palette rotates).
        c0 = locs[0]["itemStyle"]["color"]
        c1 = locs[1]["itemStyle"]["color"]
        assert c0 != c1


def test_sunburst_empty_world_returns_root_with_no_children():
    from shadow_loom.models import WorldStateV1
    empty = WorldStateV1(
        locations={}, objects={}, entities={}, events=[],
        world_traits={}, causal_topology=[],
    )
    root = ws_to_sunburst_data(empty)
    assert root["name"] == "World"
    assert root["children"] == []


# =====================================================================
# New chart helpers: calendar / treemap / polar / boxplot
# =====================================================================


def test_event_calendar_buckets_are_non_decreasing_indices():
    rows, _, max_count = ws_to_event_calendar_data(macbeth_ws)
    assert rows, "macbeth has events"
    indices = [r[0] for r in rows]
    assert indices == sorted(indices)
    assert all(r[1] >= 1 for r in rows)
    assert max_count >= max(r[1] for r in rows)


def test_event_calendar_empty_world():
    from shadow_loom.models import WorldStateV1
    empty = WorldStateV1(
        locations={}, objects={}, entities={}, events=[],
        world_traits={}, causal_topology=[],
    )
    rows, lo, hi = ws_to_event_calendar_data(empty)
    assert rows == []
    assert lo == 0 and hi == 0


def test_treemap_locations_have_children():
    roots = ws_to_treemap_data(macbeth_ws)
    assert roots, "expected at least one location with contents"
    # All root values must equal sum of child values.
    for r in roots:
        assert r["value"] == sum(c["value"] for c in r["children"])
        assert r["value"] >= 1


def test_polar_event_data_respects_top_n():
    actors, types, rows = ws_to_polar_event_data(macbeth_ws, top_n=3)
    assert len(actors) <= 3
    assert types, "expected at least one event type"
    # Every row index must be within bounds of returned axes.
    for actor_idx, type_idx, count in rows:
        assert 0 <= actor_idx < len(actors)
        assert 0 <= type_idx < len(types)
        assert count >= 1


def test_trait_boxplot_quartiles_are_monotonic():
    names, boxes, _outliers = ws_to_trait_boxplot_data(
        macbeth_ws, min_samples=2,
    )
    if not names:
        # Macbeth seed may not share enough traits — accept and skip rest.
        return
    for low, q1, med, q3, high in boxes:
        assert low <= q1 <= med <= q3 <= high
        # Trait values are 0..1.
        assert 0.0 <= low and high <= 1.0


def test_trait_boxplot_min_samples_filters_singletons():
    names_loose, _, _ = ws_to_trait_boxplot_data(macbeth_ws, min_samples=1)
    names_strict, _, _ = ws_to_trait_boxplot_data(macbeth_ws, min_samples=10)
    # Stricter sample threshold can never include more traits.
    assert len(names_strict) <= len(names_loose)


# =====================================================================
# Tabular data helpers (data panels backing the charts)
# =====================================================================


def test_entity_rows_have_required_keys():
    rows = ws_to_entity_rows(macbeth_ws)
    assert rows
    required = {"id", "name", "status", "location", "traits", "beliefs"}
    for r in rows:
        assert required.issubset(r.keys())


def test_event_rows_are_sorted_by_fabula_time():
    rows = ws_to_event_rows(macbeth_ws)
    assert rows
    times = [r["fabula_time"] for r in rows]
    assert times == sorted(times)
    # Each row should resolve actor IDs into names where possible.
    for r in rows:
        assert "actors" in r
        assert "targets" in r


def test_object_rows_resolve_owner_and_location():
    rows = ws_to_object_rows(macbeth_ws)
    # Macbeth has at least the dagger; just verify schema.
    for r in rows:
        assert {"id", "name", "location", "owner"}.issubset(r.keys())


def test_world_trait_rows_clamp_floats():
    rows = ws_to_world_trait_rows(macbeth_ws)
    for r in rows:
        assert isinstance(r["magnitude"], float)
        assert isinstance(r["inertia"], float)
        assert 0.0 <= r["magnitude"] <= 1.0
        assert 0.0 <= r["inertia"] <= 1.0


def test_calendar_rows_match_chart_data():
    chart_rows, _, _ = ws_to_event_calendar_data(macbeth_ws)
    table_rows = ws_to_event_calendar_rows(macbeth_ws)
    assert len(chart_rows) == len(table_rows)
    for (b, c), tr in zip(chart_rows, table_rows):
        assert tr["bucket"] == b
        assert tr["events"] == c


def test_polar_rows_sorted_by_count_desc():
    rows = ws_to_polar_event_rows(macbeth_ws, top_n=5)
    counts = [r["count"] for r in rows]
    assert counts == sorted(counts, reverse=True)


def test_trait_stats_rows_quartile_monotonic():
    rows = ws_to_trait_stats_rows(macbeth_ws, min_samples=2)
    for r in rows:
        assert r["min"] <= r["q1"] <= r["median"] <= r["q3"] <= r["max"]
        assert r["outliers"] >= 0


# ── Engine-grade affective scores (suspense / surprise / dramatic_irony) ──

def test_compute_affective_scores_no_entities_keeps_heuristic_only():
    """Backward-compat: omitting entity_ids must not call DirectiveAssembly."""
    from shadow_loom_ui.viz_helpers import compute_affective_scores

    scores = compute_affective_scores(macbeth_ws)
    # Engine-grade structural keys should NOT appear without entity_ids.
    assert "suspense" not in scores
    assert "surprise" not in scores
    assert "dramatic_irony" not in scores
    # Heuristic keys still present.
    assert "narrative_tension" in scores or "causal_density" in scores


def test_compute_affective_scores_with_entities_adds_engine_metrics():
    """When entity_ids are provided, the four engine metrics appear."""
    from shadow_loom_ui.viz_helpers import (
        _top_entity_ids_by_event_degree,
        compute_affective_scores,
    )

    eids = _top_entity_ids_by_event_degree(macbeth_ws, limit=10)
    assert eids, "macbeth fixture should have entities"
    scores = compute_affective_scores(macbeth_ws, entity_ids=eids)
    for key in ("mystery", "dramatic_irony", "suspense", "surprise"):
        assert key in scores, f"missing engine metric: {key}"
        assert 0.0 <= scores[key] <= 1.0


def test_affective_timeseries_syuzhet_returns_engine_curves():
    from shadow_loom_ui.viz_helpers import (
        _top_entity_ids_by_event_degree,
        affective_timeseries_syuzhet,
    )

    eids = _top_entity_ids_by_event_degree(macbeth_ws, limit=10)
    indices, series = affective_timeseries_syuzhet(
        macbeth_ws, samples=4, entity_ids=eids,
    )
    assert len(indices) >= 2
    # Suspense / surprise should be sampled and aligned to indices.
    for key in ("suspense", "surprise"):
        assert key in series
        assert len(series[key]) == len(indices)


# ── Snapshot cache hygiene (revision-stamped keys) ───────────────────

def test_invalidate_snapshot_cache_bumps_revision_and_invalidates():
    """A cache invalidation must drop entries *and* bump the revision so a
    stale read against a recycled ``id(ws)`` cannot succeed."""
    from shadow_loom_ui import viz_helpers as vh

    snap = vh.snapshot_world_at(macbeth_ws, fabula_time_bounds(macbeth_ws)[0])
    # Cache was populated by the call above.
    assert vh._SNAPSHOT_CACHE, "snapshot_world_at should populate the cache"
    pre_rev = vh._SNAPSHOT_REVISION
    vh.invalidate_snapshot_cache()
    assert not vh._SNAPSHOT_CACHE
    assert vh._SNAPSHOT_REVISION == pre_rev + 1
    # Direct lookup with the old key shape (no revision) cannot succeed.
    assert vh._snapshot_cache_get(macbeth_ws, 0) is None
    # And the snapshot still works post-invalidation.
    snap2 = vh.snapshot_world_at(
        macbeth_ws, fabula_time_bounds(macbeth_ws)[0],
    )
    assert snap2 is not None
    # Belt-and-braces: keep a reference so ``snap`` isn't optimised out.
    assert snap is not None


# ── Cursor plumbing regression guards ───────────────────────────────

def test_appstate_set_fabula_cursor_emits_event_once():
    """Slider plumbing relies on ``set_fabula_cursor`` emitting exactly one
    ``FABULA_CURSOR_CHANGED`` per distinct value and zero for repeats."""
    from shadow_loom_ui.state import AppState, StateEvent

    state = AppState()
    received: list[int | None] = []
    state.on(
        StateEvent.FABULA_CURSOR_CHANGED,
        lambda **kw: received.append(kw.get("cursor")),
    )

    state.set_fabula_cursor(5)
    state.set_fabula_cursor(5)  # de-duplicated
    state.set_fabula_cursor(None)
    state.set_syuzhet_cursor(3)  # different event, must not appear
    assert received == [5, None]


def test_world_tab_routes_slider_through_setter():
    """Regression guard: ``world_tab._on_slider_change`` and ``_set_live``
    must call the official ``state.set_fabula_cursor`` API instead of
    writing ``state.fabula_cursor`` directly. Direct writes bypass the
    event bus and desync every other time-aware panel."""
    from pathlib import Path

    src = Path(
        "shadow_loom_ui/components/world_tab.py"
    ).read_text(encoding="utf-8")
    assert "state.set_fabula_cursor(" in src, (
        "world_tab must route slider changes through state.set_fabula_cursor"
    )
    # No bare attribute write to ``state.fabula_cursor`` (would bypass the bus).
    import re

    assign_pattern = re.compile(r"\bstate\.fabula_cursor\s*=(?!=)")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not assign_pattern.search(stripped), (
            f"world_tab must not assign state.fabula_cursor directly: {line!r}"
        )
