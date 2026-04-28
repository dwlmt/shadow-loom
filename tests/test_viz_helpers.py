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
