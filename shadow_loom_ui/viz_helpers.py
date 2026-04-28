"""Data transformation helpers: WorldStateV1 → ECharts option dicts.

Pure functions that convert Shadow-Loom world model objects into the
node/link/category structures consumed by Apache ECharts series configs.
No NiceGUI imports — this module is purely data-oriented.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from shadow_loom.models import (
    CausalEdge,
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
)

# ── Visual constants ────────────────────────────────────────────────

# Vibrant palette — kept in sync with theme.CHART_COLORS by hex value.
# (We avoid importing theme.py here to keep this module a pure-data
# helper that doesn't pull in NiceGUI at import time.)
NODE_COLORS: dict[str, str] = {
    "Entity": "#F26B5E",          # Coral
    "Location": "#3A7BD5",        # Sapphire
    "EventNode": "#F5B43C",       # Amber
    "NarrativeObject": "#8A5CF0", # Iris
    "WorldTrait": "#2EA6A0",      # Teal
}

EDGE_COLORS: dict[str, str] = {
    "causal": "#D8334A",            # Crimson
    "relationship": "#E36BB8",      # Magenta Rose
    "located_in": "#FF8C42",        # Tangerine
    "owned_by": "#FF8C42",          # Tangerine
    "connected_to": "#3A7BD5",      # Sapphire
    "communicating_with": "#F5B43C",# Amber
    "eavesdropped_by": "#D8334A",   # Crimson
}

NODE_SYMBOLS: dict[str, str] = {
    "Entity": "circle",
    "Location": "rect",
    "EventNode": "triangle",
    "NarrativeObject": "diamond",
    "WorldTrait": "pin",
}

NODE_SIZES: dict[str, int] = {
    "Entity": 30,
    "Location": 25,
    "EventNode": 20,
    "NarrativeObject": 18,
    "WorldTrait": 22,
}

CATEGORIES: list[dict[str, str]] = [
    {"name": "Entity"},
    {"name": "Location"},
    {"name": "EventNode"},
    {"name": "NarrativeObject"},
    {"name": "WorldTrait"},
]

_CATEGORY_INDEX = {c["name"]: i for i, c in enumerate(CATEGORIES)}

EVENT_TYPE_COLORS: dict[str, str] = {
    "choice": "#3A7BD5",      # Sapphire
    "outcome": "#F5B43C",     # Amber
    "revelation": "#8A5CF0",  # Iris
    "action": "#6FBF3A",      # Spring Green
    "dialogue": "#F26B5E",    # Coral
    "transition": "#FF8C42",  # Tangerine
}


# ── World graph (force-directed) ───────────────────────────────────

def ws_to_graph_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Convert a WorldStateV1 into ECharts graph ``(nodes, links, categories)``."""
    nodes: list[dict] = []
    links: list[dict] = []
    all_ids: set[str] = set()

    def _node(nid: str, name: str, ntype: str, **extra: Any) -> None:
        all_ids.add(nid)
        tooltip = f"<b>{name}</b><br/>Type: {ntype}"
        for k, v in extra.items():
            if v:
                tooltip += f"<br/>{k}: {v}"
        nodes.append({
            "id": nid,
            "name": name,
            "category": _CATEGORY_INDEX.get(ntype, 0),
            "symbolSize": NODE_SIZES.get(ntype, 20),
            "symbol": NODE_SYMBOLS.get(ntype, "circle"),
            "itemStyle": {"color": NODE_COLORS.get(ntype, "#9E9E9E")},
            "tooltip": {"formatter": tooltip},
            **{k: v for k, v in extra.items() if k.startswith("_sl_")},
        })

    # Entities
    for eid, ent in ws.entities.items():
        top_traits = ", ".join(
            f"{k}={v.value:.1f}" for k, v in list(ent.traits.items())[:3]
        )
        _node(eid, ent.name, "Entity", status=ent.status, traits=top_traits,
              _sl_node_type="Entity")

    # Locations
    for lid, loc in ws.locations.items():
        _node(lid, loc.name, "Location", description=loc.description[:80] if loc.description else "",
              _sl_node_type="Location")

    # Events
    for evt in ws.events:
        _node(evt.id, evt.id, "EventNode", description=evt.description[:80],
              fabula_time=str(evt.fabula_time), _sl_node_type="EventNode")

    # Objects
    for oid, obj in ws.objects.items():
        _node(oid, obj.name, "NarrativeObject", _sl_node_type="NarrativeObject")

    # World Traits
    for wid, wt in ws.world_traits.items():
        _node(wid, wt.name, "WorldTrait", magnitude=f"{wt.magnitude.value:.2f}",
              _sl_node_type="WorldTrait")

    # ── Links ──

    def _link(src: str, tgt: str, etype: str, **extra: Any) -> None:
        if src in all_ids and tgt in all_ids:
            links.append({
                "source": src,
                "target": tgt,
                "lineStyle": {
                    "color": EDGE_COLORS.get(etype, "#BDBDBD"),
                    "width": extra.pop("width", 1.5),
                    "type": extra.pop("dash", "solid"),
                },
                "value": etype,
                **extra,
            })

    # Entity → Location
    for eid, ent in ws.entities.items():
        if ent.location_id in ws.locations:
            _link(eid, ent.location_id, "located_in")

    # Object edges
    for oid, obj in ws.objects.items():
        if obj.owner_id and obj.owner_id in ws.entities:
            _link(oid, obj.owner_id, "owned_by")
        elif obj.location_id and obj.location_id in ws.locations:
            _link(oid, obj.location_id, "located_in")

    # Causal topology
    for ce in ws.causal_topology:
        w = min(5, max(1, ce.causal_force / 2))
        _link(ce.source_id, ce.target_id, "causal", width=w)

    # Spatial topology — single edge per pair (visualised as undirected).
    for se in ws.spatial_topology:
        _link(se.source_id, se.target_id, "connected_to")

    # Social topology
    for rel in ws.social_topology:
        _link(rel.source_entity_id, rel.target_entity_id, "relationship",
              width=max(1, abs(rel.affinity) * 3))

    # Information topology
    for ie in ws.information_topology:
        for tid in ie.target_ids:
            _link(ie.source_id, tid, "communicating_with", dash="dashed")

    return nodes, links, list(CATEGORIES)


# ── Ego graph ──────────────────────────────────────────────────────

def ws_to_ego_graph_data(
    ws: WorldStateV1,
    focus_ids: list[str],
    max_hops: int = 2,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build graph data centered on *focus_ids* up to *max_hops* away.

    Uses a simple BFS over all topology edges to collect the subgraph.
    """
    # Build adjacency from all topologies
    adj: dict[str, set[str]] = {}
    for eid, ent in ws.entities.items():
        if ent.location_id in ws.locations:
            adj.setdefault(eid, set()).add(ent.location_id)
            adj.setdefault(ent.location_id, set()).add(eid)
    for ce in ws.causal_topology:
        adj.setdefault(ce.source_id, set()).add(ce.target_id)
        adj.setdefault(ce.target_id, set()).add(ce.source_id)
    for se in ws.spatial_topology:
        adj.setdefault(se.source_id, set()).add(se.target_id)
        adj.setdefault(se.target_id, set()).add(se.source_id)
    for rel in ws.social_topology:
        adj.setdefault(rel.source_entity_id, set()).add(rel.target_entity_id)
        adj.setdefault(rel.target_entity_id, set()).add(rel.source_entity_id)
    for ie in ws.information_topology:
        for tid in ie.target_ids:
            adj.setdefault(ie.source_id, set()).add(tid)
            adj.setdefault(tid, set()).add(ie.source_id)

    # BFS
    visited: set[str] = set()
    frontier = set(focus_ids)
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            next_frontier |= adj.get(nid, set())
        frontier = next_frontier - visited
    visited |= frontier  # include the final ring

    # Filter full graph data to visited IDs
    full_nodes, full_links, cats = ws_to_graph_data(ws)
    node_set = visited
    nodes = [n for n in full_nodes if n["id"] in node_set]
    links = [l for l in full_links if l["source"] in node_set and l["target"] in node_set]

    # Highlight focus nodes
    for n in nodes:
        if n["id"] in set(focus_ids):
            n["itemStyle"] = {**n.get("itemStyle", {}), "borderColor": "#FFD700", "borderWidth": 3}
            n["symbolSize"] = n.get("symbolSize", 20) * 1.4

    return nodes, links, cats


# ── Sankey: shared label/cycle-skip helpers ────────────────────────

# Sankey aspect identifiers driving the selectable diagram.
SANKEY_ASPECTS: list[tuple[str, str]] = [
    ("causal_all", "All causal flow"),
    ("chain_reaction", "Event chain reactions"),
    ("mutation", "Mutations on characters"),
    ("mutation_social", "Social/relationship mutations"),
    ("affordance_gate", "State-gated events"),
    ("ambient_propagation", "Ambient state propagation"),
    ("information", "Information flow"),
    ("world_influence", "World-trait influence"),
]


def _sankey_label_fn(ws: WorldStateV1) -> Callable[[str], str]:
    event_map = {evt.id: evt for evt in ws.events}
    entity_map = {eid: ent.name for eid, ent in ws.entities.items()}
    loc_map = {lid: loc.name for lid, loc in ws.locations.items()}
    wt_map = {wid: wt.name for wid, wt in ws.world_traits.items()}
    obj_map = {oid: o.name for oid, o in ws.objects.items()}

    def _label(nid: str) -> str:
        if nid in event_map:
            evt = event_map[nid]
            desc = (evt.description or evt.id)[:32]
            return f"t{evt.fabula_time}: {desc}"
        if nid in entity_map:
            return entity_map[nid]
        if nid in loc_map:
            return loc_map[nid]
        if nid in wt_map:
            return wt_map[nid]
        if nid in obj_map:
            return obj_map[nid]
        return nid
    return _label


def _sankey_rank_fn(ws: WorldStateV1) -> Callable[[str], int]:
    event_map = {evt.id: evt for evt in ws.events}

    def _rank(nid: str) -> int:
        if nid in event_map:
            return event_map[nid].fabula_time
        return -10_000
    return _rank


def _build_sankey(
    ws: WorldStateV1,
    raw_edges: Iterable[tuple[str, str, float, str]],
) -> tuple[list[dict], list[dict]]:
    """Generic DAG-safe Sankey builder.

    ``raw_edges`` yields ``(source_id, target_id, value, tooltip_html)``.
    """
    label = _sankey_label_fn(ws)
    rank = _sankey_rank_fn(ws)

    seen_ids: set[str] = set()
    nodes: list[dict] = []
    links: list[dict] = []
    adj: dict[str, set[str]] = {}

    def _ensure(nid: str) -> None:
        if nid in seen_ids:
            return
        seen_ids.add(nid)
        nodes.append({
            "name": nid,
            "label": {"formatter": label(nid)},
            "tooltip": {"formatter": label(nid)},
        })

    def _would_cycle(src: str, tgt: str) -> bool:
        if src == tgt:
            return True
        stack = list(adj.get(tgt, ()))
        visited = {tgt}
        while stack:
            n = stack.pop()
            if n == src:
                return True
            if n in visited:
                continue
            visited.add(n)
            stack.extend(adj.get(n, ()))
        return False

    sorted_edges = sorted(raw_edges, key=lambda e: (rank(e[0]), rank(e[1])))

    for src, tgt, val, tip in sorted_edges:
        if _would_cycle(src, tgt):
            continue
        _ensure(src)
        _ensure(tgt)
        adj.setdefault(src, set()).add(tgt)
        link: dict = {
            "source": src,
            "target": tgt,
            "value": max(0.5, float(val)),
        }
        if tip:
            link["tooltip"] = {"formatter": tip}
        links.append(link)

    return nodes, links


# ── Causal Sankey ──────────────────────────────────────────────────

def ws_to_sankey_data(
    ws: WorldStateV1,
    *,
    edge_filter: Optional[Callable[[CausalEdge], bool]] = None,
) -> tuple[list[dict], list[dict]]:
    """Build Sankey ``(nodes, links)`` from causal topology.

    ``edge_filter`` optionally restricts which ``CausalEdge`` rows are
    included (e.g. by ``causality_type``, ``mechanism``, force range).
    """
    def _iter():
        for ce in ws.causal_topology:
            if edge_filter is not None and not edge_filter(ce):
                continue
            tip = (
                f"{ce.causality_type} · {ce.mechanism}<br/>"
                f"force {ce.causal_force:.1f} · evidence {ce.evidence_strength}"
                f"<br/>fabula t={ce.fabula_time}"
            )
            yield (ce.source_id, ce.target_id, ce.causal_force, tip)
    return _build_sankey(ws, _iter())


def ws_to_information_sankey_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[dict]]:
    """Sankey of communication: source → each target per InformationEdge."""
    def _iter():
        for ie in ws.information_topology:
            tip = (
                f"medium: {ie.medium}<br/>"
                f"encrypted: {ie.is_encrypted}<br/>"
                f"established t={ie.established_at_fabula}"
            )
            for tgt in ie.target_ids:
                yield (ie.source_id, tgt, 1.0, tip)
    return _build_sankey(ws, _iter())


def ws_to_world_influence_sankey_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[dict]]:
    """Sankey from WORLD_ traits outward through their causal edges."""
    def _iter():
        for ce in ws.causal_topology:
            if not ce.source_id.startswith("WORLD_"):
                continue
            tip = (
                f"{ce.causality_type} · {ce.mechanism}<br/>"
                f"force {ce.causal_force:.1f}"
            )
            yield (ce.source_id, ce.target_id, ce.causal_force, tip)
    return _build_sankey(ws, _iter())


def ws_sankey_for_aspect(
    ws: WorldStateV1,
    aspect: str,
    *,
    min_force: float = 0.0,
    fabula_max: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """Dispatch ``aspect`` (key from ``SANKEY_ASPECTS``) to the right builder.

    ``min_force`` and ``fabula_max`` apply to causal-edge aspects only.
    """
    if aspect == "information":
        return ws_to_information_sankey_data(ws)
    if aspect == "world_influence":
        return ws_to_world_influence_sankey_data(ws)

    def _flt(ce: CausalEdge) -> bool:
        if ce.causal_force < min_force:
            return False
        if fabula_max is not None and ce.fabula_time > fabula_max:
            return False
        if aspect == "causal_all":
            return True
        return ce.causality_type == aspect

    return ws_to_sankey_data(ws, edge_filter=_flt)


# ── Social graph (entities + relationships only) ───────────────────

def ws_to_social_graph_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Entities-only graph with relationship edges."""
    nodes: list[dict] = []
    links: list[dict] = []
    cats = [{"name": "Entity"}]

    for eid, ent in ws.entities.items():
        top_traits = ", ".join(
            f"{k}={v.value:.1f}" for k, v in list(ent.traits.items())[:3]
        )
        tooltip = f"<b>{ent.name}</b><br/>Status: {ent.status}<br/>{top_traits}"
        nodes.append({
            "id": eid,
            "name": ent.name,
            "category": 0,
            "symbolSize": 35,
            "symbol": "circle",
            "itemStyle": {"color": NODE_COLORS["Entity"]},
            "tooltip": {"formatter": tooltip},
            "_sl_node_type": "Entity",
        })

    for rel in ws.social_topology:
        aff = rel.affinity
        color = "#4CAF50" if aff > 0 else "#F44336" if aff < 0 else "#9E9E9E"
        tooltip = (
            f"affinity={aff:.2f}<br/>"
            f"fear={rel.fear:.2f}<br/>"
            f"power={rel.power_dynamic:.2f}"
        )
        links.append({
            "source": rel.source_entity_id,
            "target": rel.target_entity_id,
            "lineStyle": {
                "color": color,
                "width": max(1, abs(aff) * 4),
            },
            "tooltip": {"formatter": tooltip},
        })

    return nodes, links, cats


# ── Spatial map (locations + spatial edges) ────────────────────────

def ws_to_spatial_graph_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Locations-only graph with spatial edges."""
    nodes: list[dict] = []
    links: list[dict] = []
    cats = [{"name": "Location"}]

    for lid, loc in ws.locations.items():
        tooltip = f"<b>{loc.name}</b>"
        if loc.description:
            tooltip += f"<br/>{loc.description[:80]}"
        nodes.append({
            "id": lid,
            "name": loc.name,
            "category": 0,
            "symbolSize": 30,
            "symbol": "rect",
            "itemStyle": {"color": NODE_COLORS["Location"]},
            "tooltip": {"formatter": tooltip},
            "_sl_node_type": "Location",
        })

    for se in ws.spatial_topology:
        style = "dashed" if se.is_locked else "solid"
        links.append({
            "source": se.source_id,
            "target": se.target_id,
            "lineStyle": {"color": EDGE_COLORS["connected_to"], "width": 2, "type": style},
        })

    return nodes, links, cats


# ── Trait radar chart ──────────────────────────────────────────────

def entity_to_radar_data(
    entity_id: str,
    ws: WorldStateV1,
) -> dict:
    """Build a radar chart config for a single entity's traits.

    Returns ``{"indicator": [...], "data": [{"name": ..., "value": [...]}]}``.
    """
    ent = ws.entities.get(entity_id)
    if ent is None:
        return {"indicator": [], "data": []}

    indicator: list[dict] = []
    values: list[float] = []
    for tname, tv in ent.traits.items():
        indicator.append({"name": tname, "max": 1.0})
        values.append(max(0.0, min(1.0, tv.value)))

    return {
        "indicator": indicator,
        "data": [{"name": ent.name, "value": values}],
    }


# ── Relationship heatmap ──────────────────────────────────────────

def ws_to_heatmap_data(
    ws: WorldStateV1,
    metric: str = "affinity",
) -> tuple[list[str], list[list]]:
    """Build entity×entity heatmap data.

    Returns ``(entity_names, data)`` where each data item is
    ``[x_index, y_index, value]``.
    """
    ent_ids = list(ws.entities.keys())
    ent_names = [ws.entities[eid].name for eid in ent_ids]
    idx = {eid: i for i, eid in enumerate(ent_ids)}

    data: list[list] = []
    for rel in ws.social_topology:
        si = idx.get(rel.source_entity_id)
        ti = idx.get(rel.target_entity_id)
        if si is not None and ti is not None:
            val = getattr(rel, metric, 0.0)
            data.append([si, ti, round(val, 2)])
            data.append([ti, si, round(val, 2)])

    return ent_names, data


# ── Event timeline scatter ─────────────────────────────────────────

def ws_to_timeline_data(
    ws: WorldStateV1,
) -> list[dict]:
    """Events as scatter data: ``[{name, fabula_time, syuzhet_index, event_type, description}]``."""
    return [
        {
            "name": evt.id,
            "value": [evt.fabula_time, evt.syuzhet_index],
            "itemStyle": {
                "color": EVENT_TYPE_COLORS.get(evt.event_type, "#607D8B"),
            },
            "description": evt.description[:80],
            "event_type": evt.event_type,
        }
        for evt in sorted(ws.events, key=lambda e: e.fabula_time)
    ]


# ── World stats summary ───────────────────────────────────────────

def ws_stats(ws: WorldStateV1) -> dict[str, int]:
    """Quick counts of every major object type in the world model."""
    return {
        "entities": len(ws.entities),
        "locations": len(ws.locations),
        "events": len(ws.events),
        "objects": len(ws.objects),
        "world_traits": len(ws.world_traits),
        "causal_edges": len(ws.causal_topology),
        "spatial_edges": len(ws.spatial_topology),
        "social_edges": len(ws.social_topology),
        "info_edges": len(ws.information_topology),
    }


# ── Entity state timeline (stepped trait evolution) ───────────────

def entity_state_timeline_data(
    entity_id: str,
    ws: WorldStateV1,
) -> dict:
    """Build line-chart data for an entity's trait evolution over fabula_time.

    Returns ``{"times": [...], "series": {trait_name: [values]}}``.
    """
    ent = ws.entities.get(entity_id)
    if ent is None:
        return {"times": [], "series": {}}

    # Collect all fabula_times from events + entity's own state_timeline
    times: list[int] = sorted({evt.fabula_time for evt in ws.events})
    if not times:
        return {"times": [], "series": {}}

    trait_names = list(ent.traits.keys())
    series: dict[str, list[float]] = {t: [] for t in trait_names}

    for t in times:
        snapshot = reconstruct_entity_at(ent, t)
        for tn in trait_names:
            tv = snapshot.get("traits", {}).get(tn)
            if tv is not None:
                series[tn].append(round(tv["value"] if isinstance(tv, dict) else tv, 3))
            else:
                series[tn].append(round(ent.traits[tn].value, 3))

    return {"times": times, "series": series}


# ── Causal force graph (force-directed with edge weights) ─────────

def ws_to_causal_force_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build a force-directed graph of causal edges with thickness = causal_force.

    Returns ``(nodes, links, categories)`` for ECharts graph series.
    """
    nodes: list[dict] = []
    links: list[dict] = []
    cats = [
        {"name": "Event"},
        {"name": "Entity"},
        {"name": "Location"},
        {"name": "Object"},
        {"name": "WorldTrait"},
    ]
    seen: set[str] = set()

    def _ensure(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        if nid in {evt.id for evt in ws.events}:
            evt = next(e for e in ws.events if e.id == nid)
            nodes.append({
                "id": nid, "name": nid,
                "category": 0,
                "symbolSize": 20, "symbol": "triangle",
                "itemStyle": {"color": NODE_COLORS["EventNode"]},
                "tooltip": {"formatter": f"<b>{nid}</b><br/>{evt.description[:60]}"},
            })
        elif nid in ws.entities:
            nodes.append({
                "id": nid, "name": ws.entities[nid].name,
                "category": 1,
                "symbolSize": 25, "symbol": "circle",
                "itemStyle": {"color": NODE_COLORS["Entity"]},
            })
        elif nid in ws.locations:
            nodes.append({
                "id": nid, "name": ws.locations[nid].name,
                "category": 2,
                "symbolSize": 20, "symbol": "rect",
                "itemStyle": {"color": NODE_COLORS["Location"]},
            })
        elif nid in ws.objects:
            nodes.append({
                "id": nid, "name": ws.objects[nid].name,
                "category": 3,
                "symbolSize": 16, "symbol": "diamond",
                "itemStyle": {"color": NODE_COLORS["NarrativeObject"]},
            })
        elif nid in ws.world_traits:
            nodes.append({
                "id": nid, "name": ws.world_traits[nid].name,
                "category": 4,
                "symbolSize": 20, "symbol": "pin",
                "itemStyle": {"color": NODE_COLORS["WorldTrait"]},
            })
        else:
            nodes.append({
                "id": nid, "name": nid,
                "category": 0,
                "symbolSize": 14,
                "itemStyle": {"color": "#9E9E9E"},
            })

    for ce in ws.causal_topology:
        _ensure(ce.source_id)
        _ensure(ce.target_id)
        w = min(6, max(1, ce.causal_force / 1.5))
        tooltip = (
            f"{ce.causality_type}<br/>"
            f"mechanism: {ce.mechanism}<br/>"
            f"force: {ce.causal_force}<br/>"
            f"evidence: {ce.evidence_strength}"
        )
        links.append({
            "source": ce.source_id,
            "target": ce.target_id,
            "lineStyle": {"width": w, "color": EDGE_COLORS["causal"]},
            "tooltip": {"formatter": tooltip},
        })

    return nodes, links, cats


# ── Version tree data ─────────────────────────────────────────────

def version_tree_to_echart_data(
    tree_data: list[dict],
    current_version_id: int | None = None,
) -> dict:
    """Convert DB version tree list into ECharts tree layout data.

    Each version becomes a node; parent→child edges follow ancestor_id.
    Returns a single root dict suitable for ECharts ``series[0].data``.
    """
    if not tree_data:
        return {}

    by_id: dict[int, dict] = {}
    for v in tree_data:
        vid = v["id"]
        is_current = vid == current_version_id
        by_id[vid] = {
            "name": f"v{v['version']}",
            "value": v.get("source", ""),
            "children": [],
            "itemStyle": {
                "color": "#FFD700" if is_current else "#4CAF50",
                "borderWidth": 3 if is_current else 1,
            },
            "label": {"fontWeight": "bold" if is_current else "normal"},
            "_vid": vid,
            "_version": v["version"],
        }

    # Wire parent→child
    roots: list[dict] = []
    for v in tree_data:
        node = by_id[v["id"]]
        ancestor = v.get("ancestor_id")
        if ancestor and ancestor in by_id:
            by_id[ancestor]["children"].append(node)
        else:
            roots.append(node)

    if len(roots) == 1:
        return roots[0]
    # Multiple roots — wrap in synthetic root
    return {"name": "root", "children": roots, "itemStyle": {"color": "#666"}}


# ── Epistemic map (who knows what) ───────────────────────────────

def ws_to_epistemic_data(
    ws: WorldStateV1,
) -> tuple[list[str], list[str], list[list]]:
    """Build a matrix of entity beliefs about other entities.

    Returns ``(entity_names, target_names, data)`` where data items
    are ``[x_idx, y_idx, confidence]``.
    """
    ent_ids = list(ws.entities.keys())
    ent_names = [ws.entities[eid].name for eid in ent_ids]
    idx = {eid: i for i, eid in enumerate(ent_ids)}

    # Collect belief targets (could be entities or other IDs)
    target_set: set[str] = set()
    for eid, ent in ws.entities.items():
        for b in ent.beliefs:
            target_set.add(b.target_id)
    # Filter to known entities
    target_ids = [t for t in ent_ids if t in target_set]
    target_names = [ws.entities[t].name for t in target_ids]
    tidx = {t: i for i, t in enumerate(target_ids)}

    data: list[list] = []
    for eid, ent in ws.entities.items():
        si = idx.get(eid)
        if si is None:
            continue
        for b in ent.beliefs:
            ti = tidx.get(b.target_id)
            if ti is not None:
                data.append([si, ti, round(b.confidence, 2)])

    return ent_names, target_names, data


# ── Topology table rows (correct field names) ────────────────────

def ws_to_causal_rows(ws: WorldStateV1) -> list[dict]:
    """Causal topology as table rows."""
    return [
        {
            "source": ce.source_id,
            "target": ce.target_id,
            "type": ce.causality_type,
            "mechanism": ce.mechanism,
            "force": round(ce.causal_force, 2),
            "evidence": ce.evidence_strength,
            "delay": ce.propagation_delay,
        }
        for ce in ws.causal_topology
    ]


def ws_to_spatial_rows(ws: WorldStateV1) -> list[dict]:
    """Spatial topology as table rows."""
    return [
        {
            "source": se.source_id,
            "target": se.target_id,
            "locked": se.is_locked,
            "barrier": se.barrier_item_id or "",
        }
        for se in ws.spatial_topology
    ]


def ws_to_social_rows(ws: WorldStateV1) -> list[dict]:
    """Social topology as table rows."""
    return [
        {
            "source": rel.source_entity_id,
            "target": rel.target_entity_id,
            "affinity": round(rel.affinity, 2),
            "fear": round(rel.fear, 2),
            "power": round(rel.power_dynamic, 2),
            "inertia": round(rel.inertia, 2),
        }
        for rel in ws.social_topology
    ]


def ws_to_info_rows(ws: WorldStateV1) -> list[dict]:
    """Information topology as table rows."""
    return [
        {
            "source": ie.source_id,
            "targets": ", ".join(ie.target_ids),
            "medium": ie.medium,
            "encrypted": ie.is_encrypted,
        }
        for ie in ws.information_topology
    ]


# ── ThemeRiver (multi-entity trait evolution over time) ───────────

def ws_to_theme_river_data(
    ws: WorldStateV1,
    trait_names: list[str] | None = None,
    max_entities: int = 6,
) -> list[list]:
    """Build ThemeRiver series data: [[time, value, "entity:trait"], ...].

    Each river band is an entity-trait pair showing evolution over fabula time.
    """
    if not ws.events:
        return []

    times = sorted({evt.fabula_time for evt in ws.events})
    if not times:
        return []

    ent_ids = list(ws.entities.keys())[:max_entities]

    if trait_names is None:
        trait_counts: dict[str, int] = {}
        for eid in ent_ids:
            for t in ws.entities[eid].traits:
                trait_counts[t] = trait_counts.get(t, 0) + 1
        trait_names = sorted(trait_counts, key=trait_counts.get, reverse=True)[:4]

    data: list[list] = []
    for t in times:
        for eid in ent_ids:
            ent = ws.entities[eid]
            snapshot = reconstruct_entity_at(ent, t)
            for tn in trait_names:
                tv = snapshot.get("traits", {}).get(tn)
                if tv is not None:
                    val = tv["value"] if isinstance(tv, dict) else tv
                else:
                    val = ent.traits[tn].value if tn in ent.traits else 0.5
                # ThemeRiver needs positive values; shift from [0,1] to [0.1, 1.1]
                data.append([str(t), round(max(0.01, val + 0.1), 3), f"{ent.name}:{tn}"])

    return data


# ── Chord diagram (relationship reciprocity) ─────────────────────

def ws_to_chord_data(
    ws: WorldStateV1,
    metric: str = "affinity",
) -> tuple[list[str], list[list[float]]]:
    """Build chord diagram matrix: entity names + NxN adjacency matrix.

    Returns ``(entity_names, matrix)`` where matrix[i][j] is the
    absolute metric value from entity i to entity j.
    """
    ent_ids = list(ws.entities.keys())
    ent_names = [ws.entities[eid].name for eid in ent_ids]
    n = len(ent_ids)
    idx = {eid: i for i, eid in enumerate(ent_ids)}

    matrix = [[0.0] * n for _ in range(n)]
    for rel in ws.social_topology:
        si = idx.get(rel.source_entity_id)
        ti = idx.get(rel.target_entity_id)
        if si is not None and ti is not None:
            val = abs(getattr(rel, metric, 0.0))
            matrix[si][ti] = round(val, 2)

    return ent_names, matrix


# ── Parallel coordinates (trait comparison across entities) ───────

def ws_to_parallel_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[list[float]], list[str]]:
    """Build parallel coordinate axes + data.

    Returns ``(dimensions, data_rows, entity_names)``.
    """
    trait_set: set[str] = set()
    for ent in ws.entities.values():
        trait_set |= set(ent.traits.keys())
    trait_names = sorted(trait_set)

    dimensions = [{"name": t, "min": 0, "max": 1} for t in trait_names]
    data_rows: list[list[float]] = []
    entity_names: list[str] = []

    for eid, ent in ws.entities.items():
        row = []
        for t in trait_names:
            tv = ent.traits.get(t)
            row.append(round(tv.value, 3) if tv else 0.5)
        data_rows.append(row)
        entity_names.append(ent.name)

    return dimensions, data_rows, entity_names


# ── Event Gantt / swim lanes ─────────────────────────────────────

def ws_to_gantt_data(
    ws: WorldStateV1,
) -> tuple[list[str], list[dict]]:
    """Build Gantt/swim-lane data: actor lanes x event time spans.

    Returns ``(actor_names, event_items)``.
    """
    actor_ids: list[str] = []
    actor_names: list[str] = []
    for eid, ent in ws.entities.items():
        actor_ids.append(eid)
        actor_names.append(ent.name)
    actor_idx = {aid: i for i, aid in enumerate(actor_ids)}

    items: list[dict] = []
    for evt in sorted(ws.events, key=lambda e: e.fabula_time):
        actors = evt.actor_ids if evt.actor_ids else []
        for aid in actors:
            idx_val = actor_idx.get(aid)
            if idx_val is not None:
                items.append({
                    "actor_idx": idx_val,
                    "start": evt.fabula_time,
                    "end": evt.fabula_time + 1,
                    "event_id": evt.id,
                    "event_type": evt.event_type,
                    "description": evt.description[:50] if evt.description else evt.id,
                })

    return actor_names, items


# ── Propagation waterfall (causal chain impact) ──────────────────

def mutations_to_waterfall_data(
    mutations: list[dict],
    blocked: list[dict] | None = None,
) -> list[dict]:
    """Convert causal physics mutations into waterfall chart data.

    mutations: [{"entity_id", "trait", "old", "new", "impact"}]
    blocked: [{"source", "target", "reason", "trait"}]
    """
    data: list[dict] = []

    for m in mutations:
        shift = m.get("new", 0) - m.get("old", 0)
        data.append({
            "name": f"{m.get('entity_id', '?')}:{m.get('trait', '?')}",
            "value": round(shift, 3),
            "type": "positive" if shift > 0 else "negative",
            "detail": f"impact={m.get('impact', 0):.2f}",
        })

    if blocked:
        for b in blocked:
            data.append({
                "name": f"{b.get('target', '?')}:{b.get('trait', '?')}",
                "value": 0,
                "type": "blocked",
                "detail": f"blocked by {b.get('reason', 'unknown')}",
            })

    return data


# ── Sunburst (world model composition hierarchy) ─────────────────

# Local palette to colour-rotate locations. Mirrors theme.CHART_COLORS but
# kept here to avoid a UI-side import in this pure-data module.
_SUNBURST_RING_COLORS = [
    "#F26B5E", "#2EA6A0", "#F5B43C", "#3A7BD5", "#E36BB8",
    "#6FBF3A", "#8A5CF0", "#FF8C42", "#D8334A", "#1E2A3A",
]


def ws_to_sunburst_data(ws: WorldStateV1) -> dict:
    """Build sunburst hierarchy: World → Locations → Entities → Traits."""
    children: list[dict] = []

    for li, (lid, loc) in enumerate(ws.locations.items()):
        loc_color = _SUNBURST_RING_COLORS[li % len(_SUNBURST_RING_COLORS)]
        loc_children: list[dict] = []
        for eid, ent in ws.entities.items():
            if ent.location_id == lid:
                trait_children = [
                    {
                        "name": tname,
                        "value": max(1, int(tv.value * 10)),
                        "itemStyle": {"color": loc_color, "opacity": 0.55},
                    }
                    for tname, tv in list(ent.traits.items())[:6]
                ]
                loc_children.append({
                    "name": ent.name,
                    "value": max(1, len(ent.traits)),
                    "children": trait_children,
                    "itemStyle": {"color": loc_color, "opacity": 0.8},
                })
        for oid, obj in ws.objects.items():
            if obj.location_id == lid:
                loc_children.append({
                    "name": obj.name,
                    "value": 1,
                    "itemStyle": {"color": loc_color, "opacity": 0.7},
                })
        children.append({
            "name": loc.name,
            "value": max(1, len(loc_children)),
            "children": loc_children,
            "itemStyle": {"color": loc_color},
        })

    # Entities without a location
    for eid, ent in ws.entities.items():
        if ent.location_id not in ws.locations:
            children.append({
                "name": ent.name,
                "value": max(1, len(ent.traits)),
                "itemStyle": {"color": NODE_COLORS["Entity"]},
            })

    if ws.world_traits:
        wt_children = [
            {
                "name": wt.name,
                "value": max(1, int(wt.magnitude.value * 10)),
                "itemStyle": {"color": NODE_COLORS["WorldTrait"], "opacity": 0.7},
            }
            for wt in ws.world_traits.values()
        ]
        children.append({
            "name": "World Traits",
            "value": len(ws.world_traits),
            "children": wt_children,
            "itemStyle": {"color": NODE_COLORS["WorldTrait"]},
        })

    return {"name": "World", "children": children}


# ── Fabula timeline helpers ───────────────────────────────────────

# Snapshot cache: keyed by (id(ws), t). Bounded to keep memory predictable;
# call ``invalidate_snapshot_cache()`` whenever the WorldState changes.
_SNAPSHOT_CACHE: "dict[tuple[int, int], WorldStateV1]" = {}
_SNAPSHOT_CACHE_MAX = 64


def _snapshot_cache_get(ws: WorldStateV1, t: int) -> Optional[WorldStateV1]:
    return _SNAPSHOT_CACHE.get((id(ws), t))


def _snapshot_cache_put(ws: WorldStateV1, t: int, snap: WorldStateV1) -> None:
    if len(_SNAPSHOT_CACHE) >= _SNAPSHOT_CACHE_MAX:
        # Drop an arbitrary entry — slider scrubbing is sequential so the
        # working set is small and FIFO eviction is fine.
        _SNAPSHOT_CACHE.pop(next(iter(_SNAPSHOT_CACHE)))
    _SNAPSHOT_CACHE[(id(ws), t)] = snap


def invalidate_snapshot_cache() -> None:
    """Drop all cached fabula-time snapshots (call on WORLD_STATE_CHANGED)."""
    _SNAPSHOT_CACHE.clear()


def fabula_time_bounds(ws: WorldStateV1) -> tuple[int, int]:
    """Return ``(min, max)`` fabula_time across all events.

    Returns ``(0, 0)`` for empty event lists so callers can disable
    the slider safely.
    """
    if not ws.events:
        return (0, 0)
    times = [evt.fabula_time for evt in ws.events]
    return (min(times), max(times))


def syuzhet_time_bounds(ws: WorldStateV1) -> tuple[int, int]:
    """Return ``(min, max)`` syuzhet_index across all events."""
    if not ws.events:
        return (0, 0)
    idxs = [evt.syuzhet_index for evt in ws.events]
    return (min(idxs), max(idxs))


def snapshot_world_at_syuzhet(ws: WorldStateV1, s: int) -> WorldStateV1:
    """Return ``ws`` filtered to events with ``syuzhet_index <= s``.

    Unlike :func:`snapshot_world_at`, this does NOT replay entity or
    world-trait state — those evolve in fabula time, not reading order.
    Only the ``events`` list is trimmed so views ordered by reader
    knowledge (suspense, reveals, dramatic irony) get a reading-time
    cursor.
    """
    cache_key = (id(ws), -1 - s)  # negative key namespace for syuzhet snapshots
    cached = _SNAPSHOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    new = ws.model_copy(deep=False)
    new.events = [evt for evt in ws.events if evt.syuzhet_index <= s]
    if len(_SNAPSHOT_CACHE) >= _SNAPSHOT_CACHE_MAX:
        _SNAPSHOT_CACHE.pop(next(iter(_SNAPSHOT_CACHE)))
    _SNAPSHOT_CACHE[cache_key] = new
    return new


# ── Relationship timeline (dyad evolution over fabula time) ────────

_REL_METRICS: tuple[str, ...] = ("affinity", "fear", "power_dynamic")


def relationship_timeline_data(
    ws: WorldStateV1,
    ent_a: str,
    ent_b: str,
) -> dict:
    """Build line data for affinity / fear / power between two entities.

    Replays ``mutation_social`` :class:`CausalEdge` entries in fabula
    order, accumulating ``trait_delta`` per metric. The starting values
    come from the matching :class:`RelationshipEdge` if present (so the
    final cumulative values are consistent with the steady-state edge),
    otherwise from 0.

    Edges are matched on the unordered pair ``{target_id, rel_counterpart_id}``
    so direction-agnostic metrics (affinity, fear) work in either ordering.
    For ``power_dynamic`` the sign is flipped when the perspective entity
    is reversed relative to the requested ``(ent_a, ent_b)`` pair.

    Returns ``{"times": [...], "series": {metric: [values]}}``.
    """
    pair = {ent_a, ent_b}

    # Initial steady-state values from RelationshipEdge (a -> b orientation).
    init = {m: 0.0 for m in _REL_METRICS}
    for re in ws.social_topology:
        if {re.source_entity_id, re.target_entity_id} != pair:
            continue
        flip = (re.source_entity_id == ent_b)
        init["affinity"] = re.affinity
        init["fear"] = re.fear
        init["power_dynamic"] = -re.power_dynamic if flip else re.power_dynamic
        break

    # Collect mutation_social edges for this dyad.
    edges = []
    for ce in ws.causal_topology:
        if ce.causality_type != "mutation_social":
            continue
        if ce.trait_target not in _REL_METRICS:
            continue
        if {ce.target_id, ce.rel_counterpart_id} != pair:
            continue
        edges.append(ce)
    edges.sort(key=lambda c: c.fabula_time)

    if not edges:
        # No timeline — show a single point at fabula_time 0 with init values.
        tmin, _ = fabula_time_bounds(ws)
        return {
            "times": [tmin],
            "series": {m: [init[m]] for m in _REL_METRICS},
        }

    # Working backwards: the cumulative series should END at ``init`` so the
    # steady-state RelationshipEdge value is the final value. So we compute
    # forward deltas and then offset.
    cum = {m: 0.0 for m in _REL_METRICS}
    times: list[int] = []
    rows: list[dict[str, float]] = []
    for ce in edges:
        delta = ce.trait_delta or 0.0
        # Flip sign for power_dynamic when perspective is reversed.
        if ce.trait_target == "power_dynamic" and ce.target_id == ent_b:
            delta = -delta
        cum[ce.trait_target] = cum[ce.trait_target] + delta
        times.append(ce.fabula_time)
        rows.append(dict(cum))

    # Offset so the final cumulative value matches the RelationshipEdge.
    final = rows[-1]
    offset = {m: init[m] - final[m] for m in _REL_METRICS}
    series = {m: [round(r[m] + offset[m], 3) for r in rows] for m in _REL_METRICS}

    # Prepend an initial baseline point for clarity.
    tmin, _ = fabula_time_bounds(ws)
    if times[0] > tmin:
        times.insert(0, tmin)
        for m in _REL_METRICS:
            # Baseline = init - (sum of all signed deltas for this metric).
            total = sum(
                ((c.trait_delta or 0.0) * (-1 if (c.trait_target == "power_dynamic" and c.target_id == ent_b) else 1))
                for c in edges if c.trait_target == m
            )
            series[m].insert(0, round(init[m] - total, 3))

    return {"times": times, "series": series}


def list_relationship_pairs(ws: WorldStateV1) -> list[tuple[str, str, str, str]]:
    """List entity pairs that have *any* relationship signal.

    Returns ``[(id_a, id_b, name_a, name_b), ...]``, one entry per
    unordered pair found in either ``social_topology`` or
    ``mutation_social`` causal edges.
    """
    pairs: dict[tuple[str, str], None] = {}
    for re in ws.social_topology:
        a, b = sorted([re.source_entity_id, re.target_entity_id])
        pairs[(a, b)] = None
    for ce in ws.causal_topology:
        if ce.causality_type != "mutation_social":
            continue
        if not ce.rel_counterpart_id:
            continue
        a, b = sorted([ce.target_id, ce.rel_counterpart_id])
        pairs[(a, b)] = None

    out: list[tuple[str, str, str, str]] = []
    for (a, b) in pairs:
        ent_a = ws.entities.get(a)
        ent_b = ws.entities.get(b)
        if ent_a is None or ent_b is None:
            continue
        out.append((a, b, ent_a.name, ent_b.name))
    out.sort(key=lambda r: (r[2].lower(), r[3].lower()))
    return out


# ── World-trait timeline (magnitude evolution over fabula time) ───

def world_trait_timeline_data(
    ws: WorldStateV1,
    world_id: str,
) -> dict:
    """Build line data for a world trait's magnitude over fabula_time.

    Returns ``{"times": [...], "value": [...], "inertia": [...]}``.
    """
    wt = ws.world_traits.get(world_id)
    if wt is None:
        return {"times": [], "value": [], "inertia": []}

    tmin, tmax = fabula_time_bounds(ws)
    # Sample at every snapshot fabula_time plus the bounds.
    sample_times: set[int] = {tmin, tmax}
    for snap in wt.state_timeline:
        sample_times.add(snap.fabula_time)
    times = sorted(t for t in sample_times if tmin <= t <= tmax) or [0]

    values: list[float] = []
    inertias: list[float] = []
    for t in times:
        snap = reconstruct_world_trait_at(wt, t)
        mag = snap["magnitude"]
        values.append(round(mag["value"], 3))
        inertias.append(round(mag["inertia"], 3))

    return {"times": times, "value": values, "inertia": inertias}


def list_world_traits(ws: WorldStateV1) -> list[tuple[str, str]]:
    """``[(world_id, display_name), ...]`` sorted by name."""
    out = [(wid, wt.name) for wid, wt in ws.world_traits.items()]
    out.sort(key=lambda r: r[1].lower())
    return out


def snapshot_world_at(ws: WorldStateV1, t: int) -> WorldStateV1:
    """Return a shallow copy of ``ws`` reconstructed to fabula_time ``t``.

    Each :class:`Entity` has its mutable fields (traits, beliefs,
    status, location_id) replayed via :func:`reconstruct_entity_at`,
    each :class:`GlobalTrait` via :func:`reconstruct_world_trait_at`,
    and the events list is filtered to those with
    ``fabula_time <= t``.

    Topology edges are kept intact since they encode structural
    relationships, not state. ``NarrativeObject`` instances are also
    left untouched: the model has no per-object state timeline, so
    object ``location_id`` / ``owner_id`` always reflect the latest
    snapshot. Renderers that show objects on a historical cursor
    should treat object placement as approximate.

    The returned model is suitable to re-feed into existing renderers
    without further changes.
    """
    cached = _snapshot_cache_get(ws, t)
    if cached is not None:
        return cached

    new = ws.model_copy(deep=True)

    for eid, ent in new.entities.items():
        snap = reconstruct_entity_at(ent, t)
        # Replay traits
        from shadow_loom.models import TraitVector  # local import avoids cycles
        ent.traits = {
            k: TraitVector(value=v["value"], inertia=v["inertia"])
            for k, v in snap["traits"].items()
        }
        # Status / location
        ent.status = snap["status"]
        ent.location_id = snap["location_id"]
        # Beliefs: snap returns dicts -> reuse pydantic validators
        from shadow_loom.models import Belief
        ent.beliefs = [Belief(**b) for b in snap["beliefs"]]

    for wid, wt in new.world_traits.items():
        snap = reconstruct_world_trait_at(wt, t)
        from shadow_loom.models import TraitVector
        mag = snap["magnitude"]
        wt.magnitude = TraitVector(value=mag["value"], inertia=mag["inertia"])
        if snap.get("description") is not None:
            wt.description = snap["description"]

    new.events = [evt for evt in new.events if evt.fabula_time <= t]
    _snapshot_cache_put(ws, t, new)
    return new


# ── Calendar heatmap (event density over fabula time) ──────────────

def ws_to_event_calendar_data(
    ws: WorldStateV1,
    *,
    bucket_size: int | None = None,
) -> tuple[list[list[int]], int, int]:
    """Bucket events by fabula_time for an ECharts calendar/heatmap view.

    Returns ``(rows, min_bucket, max_bucket)`` where each row is
    ``[bucket_index, count]``. ``bucket_size`` defaults to a value that
    produces ~30 buckets across the fabula span (at least 1).

    Designed to feed an ECharts ``heatmap`` series on a single-axis
    layout — true calendar views require Gregorian dates which the
    fabula timeline doesn't have, so we use a linear bucketed heatmap
    that conveys the same density information.
    """
    if not ws.events:
        return [], 0, 0
    times = sorted(evt.fabula_time for evt in ws.events)
    tmin, tmax = times[0], times[-1]
    span = max(1, tmax - tmin)
    if bucket_size is None or bucket_size <= 0:
        bucket_size = max(1, span // 30)
    buckets: dict[int, int] = {}
    for t in times:
        b = (t - tmin) // bucket_size
        buckets[b] = buckets.get(b, 0) + 1
    rows = [[b, c] for b, c in sorted(buckets.items())]
    return rows, 0, max(buckets) if buckets else 0


# ── Treemap (locations → entities/objects) ─────────────────────────

def ws_to_treemap_data(ws: WorldStateV1) -> list[dict]:
    """Build a treemap hierarchy of locations → contents.

    Unlike the sunburst, the treemap scales children by area so it
    reads well even when one location holds many entities/objects.
    Returns a list of root nodes suitable for an ECharts ``treemap``
    series ``data`` field.
    """
    roots: list[dict] = []
    palette = _SUNBURST_RING_COLORS
    for li, (lid, loc) in enumerate(ws.locations.items()):
        color = palette[li % len(palette)]
        children: list[dict] = []
        for ent in ws.entities.values():
            if ent.location_id == lid:
                children.append({
                    "name": ent.name,
                    "value": max(1, len(ent.traits)),
                    "itemStyle": {"color": color},
                })
        for obj in ws.objects.values():
            if obj.location_id == lid:
                children.append({
                    "name": obj.name,
                    "value": 1,
                    "itemStyle": {"color": color, "opacity": 0.6},
                })
        if children:
            roots.append({
                "name": loc.name,
                "value": sum(c["value"] for c in children),
                "children": children,
                "itemStyle": {"color": color},
            })

    # Orphan entities (no resolved location)
    orphans = [
        {"name": ent.name, "value": max(1, len(ent.traits))}
        for ent in ws.entities.values()
        if ent.location_id not in ws.locations
    ]
    if orphans:
        roots.append({
            "name": "Unplaced",
            "value": sum(o["value"] for o in orphans),
            "children": orphans,
            "itemStyle": {"color": NODE_COLORS["Entity"]},
        })
    return roots


# ── Polar bar (event-type breakdown per actor) ─────────────────────

def ws_to_polar_event_data(
    ws: WorldStateV1,
    *,
    top_n: int = 8,
) -> tuple[list[str], list[str], list[list[Any]]]:
    """Aggregate events into actor × event_type counts.

    Returns ``(actors, event_types, rows)`` where each row is
    ``[actor_index, event_type_index, count]``, suitable for a polar
    heatmap or stacked bar series. Actors are limited to the top
    ``top_n`` by total event count to keep the chart legible.
    """
    if not ws.events:
        return [], [], []

    counts: dict[tuple[str, str], int] = {}
    actor_totals: dict[str, int] = {}
    type_set: set[str] = set()

    for evt in ws.events:
        et = evt.event_type or "unknown"
        type_set.add(et)
        actors = evt.actor_ids or ["(none)"]
        for aid in actors:
            counts[(aid, et)] = counts.get((aid, et), 0) + 1
            actor_totals[aid] = actor_totals.get(aid, 0) + 1

    top_actors = [
        a for a, _ in sorted(
            actor_totals.items(), key=lambda kv: kv[1], reverse=True,
        )[:top_n]
    ]
    actor_index = {a: i for i, a in enumerate(top_actors)}

    event_types = sorted(type_set)
    type_index = {t: i for i, t in enumerate(event_types)}

    rows: list[list[Any]] = []
    for (aid, et), c in counts.items():
        if aid not in actor_index:
            continue
        rows.append([actor_index[aid], type_index[et], c])

    actor_labels = [
        ws.entities[a].name if a in ws.entities else a
        for a in top_actors
    ]
    return actor_labels, event_types, rows


# ── Boxplot (trait distributions across entities) ──────────────────

def ws_to_trait_boxplot_data(
    ws: WorldStateV1,
    *,
    min_samples: int = 3,
) -> tuple[list[str], list[list[float]], list[list[float]]]:
    """Compute boxplot summaries for each trait shared across entities.

    Returns ``(trait_names, boxplot_rows, outlier_points)`` where:
      - ``trait_names`` is the x-axis label list,
      - each row in ``boxplot_rows`` is ``[low, q1, median, q3, high]``
        in the order ECharts' ``boxplot`` series expects,
      - ``outlier_points`` is ``[trait_index, value]`` pairs for
        scatter overlay.

    Traits with fewer than ``min_samples`` entities recorded are
    excluded so we never display a degenerate box.
    """
    by_trait: dict[str, list[float]] = {}
    for ent in ws.entities.values():
        for tname, tv in ent.traits.items():
            by_trait.setdefault(tname, []).append(float(tv.value))

    trait_names: list[str] = []
    rows: list[list[float]] = []
    outliers: list[list[float]] = []

    for tname in sorted(by_trait):
        values = sorted(by_trait[tname])
        if len(values) < min_samples:
            continue
        n = len(values)

        def _pct(p: float) -> float:
            # Linear-interpolated quantile (matches numpy default 'linear').
            if n == 1:
                return values[0]
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            return values[f] + (values[c] - values[f]) * (k - f)

        q1 = _pct(0.25)
        med = _pct(0.50)
        q3 = _pct(0.75)
        iqr = q3 - q1
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        non_out = [v for v in values if lo_fence <= v <= hi_fence]
        low = min(non_out) if non_out else values[0]
        high = max(non_out) if non_out else values[-1]
        idx = len(trait_names)
        trait_names.append(tname)
        rows.append([low, q1, med, q3, high])
        for v in values:
            if v < lo_fence or v > hi_fence:
                outliers.append([idx, v])

    return trait_names, rows, outliers


# =====================================================================
# Tabular data helpers — back the data panels next to each chart
# =====================================================================


def ws_to_entity_rows(ws: WorldStateV1) -> list[dict]:
    """One row per entity with key state fields."""
    rows: list[dict] = []
    for eid, ent in ws.entities.items():
        loc = ws.locations.get(ent.location_id) if ent.location_id else None
        rows.append({
            "id": eid,
            "name": ent.name,
            "status": ent.status,
            "location": loc.name if loc else (ent.location_id or ""),
            "traits": len(ent.traits),
            "beliefs": len(ent.beliefs),
            "constants": ", ".join(ent.constants) if ent.constants else "",
        })
    return rows


def ws_to_event_rows(ws: WorldStateV1) -> list[dict]:
    """One row per event in fabula-time order."""

    def _name_of(nid: str) -> str:
        if nid in ws.entities:
            return ws.entities[nid].name
        if nid in ws.objects:
            return ws.objects[nid].name
        if nid in ws.locations:
            return ws.locations[nid].name
        return nid

    rows: list[dict] = []
    for evt in sorted(ws.events, key=lambda e: e.fabula_time):
        rows.append({
            "id": evt.id,
            "fabula_time": evt.fabula_time,
            "syuzhet_index": evt.syuzhet_index,
            "type": evt.event_type,
            "actors": ", ".join(_name_of(a) for a in evt.actor_ids) or "—",
            "targets": ", ".join(_name_of(t) for t in evt.target_ids) or "—",
            "description": evt.description,
        })
    return rows


def ws_to_object_rows(ws: WorldStateV1) -> list[dict]:
    """One row per narrative object."""
    rows: list[dict] = []
    for oid, obj in ws.objects.items():
        loc = ws.locations.get(obj.location_id) if obj.location_id else None
        owner = ws.entities.get(obj.owner_id) if obj.owner_id else None
        rows.append({
            "id": oid,
            "name": obj.name,
            "location": loc.name if loc else (obj.location_id or "—"),
            "owner": owner.name if owner else (obj.owner_id or "—"),
            "affordances": ", ".join(a.action for a in obj.affordances) or "—",
            "properties": ", ".join(
                f"{k}={v}" for k, v in obj.properties.items()
            ) or "—",
        })
    return rows


def ws_to_world_trait_rows(ws: WorldStateV1) -> list[dict]:
    """One row per global/world-level trait."""
    rows: list[dict] = []
    for wid, wt in ws.world_traits.items():
        rows.append({
            "id": wid,
            "name": wt.name,
            "magnitude": round(wt.magnitude.value, 3),
            "inertia": round(wt.magnitude.inertia, 3),
            "description": getattr(wt, "description", "") or "",
        })
    return rows


def ws_to_event_calendar_rows(ws: WorldStateV1) -> list[dict]:
    """Tabular form of the event-density calendar buckets."""
    rows, _, _ = ws_to_event_calendar_data(ws)
    return [{"bucket": b, "events": c} for b, c in rows]


def ws_to_polar_event_rows(ws: WorldStateV1, *, top_n: int = 8) -> list[dict]:
    """Flat ``(actor, event_type, count)`` rows for the polar pivot."""
    actors, types, raw = ws_to_polar_event_data(ws, top_n=top_n)
    out: list[dict] = []
    for actor_idx, type_idx, count in raw:
        out.append({
            "actor": actors[actor_idx],
            "event_type": types[type_idx],
            "count": count,
        })
    return sorted(out, key=lambda r: (-r["count"], r["actor"], r["event_type"]))


def ws_to_trait_stats_rows(
    ws: WorldStateV1,
    *,
    min_samples: int = 3,
) -> list[dict]:
    """Trait-distribution summaries (the boxplot data, in tabular form)."""
    names, boxes, outliers = ws_to_trait_boxplot_data(
        ws, min_samples=min_samples,
    )
    out_count: dict[int, int] = {}
    for idx, _v in outliers:
        out_count[idx] = out_count.get(idx, 0) + 1
    rows: list[dict] = []
    for i, name in enumerate(names):
        low, q1, med, q3, high = boxes[i]
        rows.append({
            "trait": name,
            "min": round(low, 3),
            "q1": round(q1, 3),
            "median": round(med, 3),
            "q3": round(q3, 3),
            "max": round(high, 3),
            "outliers": out_count.get(i, 0),
        })
    return rows


# =====================================================================
# NEW HELPERS — added for the ECharts gallery-inspired charts.
#
# All helpers are pure-data: no NiceGUI imports.
# =====================================================================

# ── #1 Cartesian-anchored causal force graph ──────────────────────

def ws_to_causal_cartesian_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[dict]]:
    """Causal nodes + links anchored on (fabula_time, syuzhet_index).

    Only EventNodes have natural temporal coordinates. Non-event nodes
    referenced by causal edges are projected onto the mean fabula/syuzhet
    of the events that touch them so the layout still has them somewhere
    sensible (rather than being scattered randomly).

    Returns ``(nodes, links)`` ready for an ECharts ``graph`` series with
    ``coordinateSystem: 'cartesian2d'``.
    """
    if not ws.events or not ws.causal_topology:
        return [], []

    # Index events for quick lookup
    evt_by_id = {e.id: e for e in ws.events}
    evt_ids = set(evt_by_id.keys())

    # Compute mean coords for every non-event referenced by causal edges
    coord_acc: dict[str, list[tuple[int, int]]] = {}
    for ce in ws.causal_topology:
        for nid in (ce.source_id, ce.target_id):
            if nid in evt_ids:
                continue
            # Find the events this node participates in (as actor/target)
            for evt in ws.events:
                if nid in evt.actor_ids or nid in evt.target_ids:
                    coord_acc.setdefault(nid, []).append(
                        (evt.fabula_time, evt.syuzhet_index)
                    )

    def _mean_xy(nid: str) -> tuple[float, float]:
        pts = coord_acc.get(nid)
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        # Fallback: place at the midpoint of the fabula range so the
        # node is at least visible.
        xs = [e.fabula_time for e in ws.events]
        ys = [e.syuzhet_index for e in ws.events]
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    nodes: list[dict] = []
    seen: set[str] = set()

    def _ensure(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        if nid in evt_ids:
            evt = evt_by_id[nid]
            nodes.append({
                "id": nid,
                "name": nid,
                "value": [evt.fabula_time, evt.syuzhet_index],
                "symbol": "triangle",
                "symbolSize": 16,
                "itemStyle": {"color": NODE_COLORS["EventNode"]},
                "tooltip": {
                    "formatter": (
                        f"<b>{nid}</b><br/>"
                        f"t={evt.fabula_time}, s={evt.syuzhet_index}<br/>"
                        f"{evt.description[:80]}"
                    )
                },
            })
            return
        x, y = _mean_xy(nid)
        if nid in ws.entities:
            color = NODE_COLORS["Entity"]; sym = "circle"; size = 18
            label = ws.entities[nid].name
        elif nid in ws.locations:
            color = NODE_COLORS["Location"]; sym = "rect"; size = 16
            label = ws.locations[nid].name
        elif nid in ws.objects:
            color = NODE_COLORS["NarrativeObject"]; sym = "diamond"; size = 14
            label = ws.objects[nid].name
        elif nid in ws.world_traits:
            color = NODE_COLORS["WorldTrait"]; sym = "pin"; size = 16
            label = ws.world_traits[nid].name
        else:
            color = "#9E9E9E"; sym = "circle"; size = 12
            label = nid
        nodes.append({
            "id": nid,
            "name": label,
            "value": [round(x, 2), round(y, 2)],
            "symbol": sym,
            "symbolSize": size,
            "itemStyle": {"color": color},
        })

    links: list[dict] = []
    for ce in ws.causal_topology:
        _ensure(ce.source_id)
        _ensure(ce.target_id)
        w = min(6, max(1, ce.causal_force / 1.5))
        links.append({
            "source": ce.source_id,
            "target": ce.target_id,
            "lineStyle": {"width": w, "color": EDGE_COLORS["causal"], "opacity": 0.6, "curveness": 0.15},
            "tooltip": {"formatter": f"{ce.causality_type}<br/>force={ce.causal_force}"},
        })
    return nodes, links


# ── #3 Animated propagation graph ─────────────────────────────────

def mutations_to_propagation_graph(
    mutations: list[dict],
    blocked: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build a node/link graph of a propagation cascade.

    Each mutation is expected to have ``source``, ``target`` (or ``entity``)
    and a numeric ``delta`` field (signed). The first node in any chain
    becomes the root; descendants fan out by call order.

    Used to visualise an intervention/counterfactual cascade as a live
    force-graph instead of a static waterfall.
    """
    nodes: dict[str, dict] = {}
    links: list[dict] = []

    def _label(m: dict) -> str:
        return (
            m.get("entity")
            or m.get("target")
            or m.get("source")
            or m.get("name")
            or "?"
        )

    def _delta(m: dict) -> float:
        try:
            return float(m.get("delta") or m.get("trait_delta") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    for i, m in enumerate(mutations or []):
        src = m.get("source")
        tgt = _label(m)
        delta = _delta(m)
        for nid in filter(None, (src, tgt)):
            if nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "name": nid,
                    "symbolSize": 18,
                    "itemStyle": {
                        "color": "#6FBF3A" if delta >= 0 else "#D8334A",
                    },
                }
        if src and tgt and src != tgt:
            links.append({
                "source": src,
                "target": tgt,
                "value": round(delta, 3),
                "lineStyle": {
                    "width": max(1.5, min(6, abs(delta) * 6)),
                    "color": "#6FBF3A" if delta >= 0 else "#D8334A",
                    "opacity": 0.7,
                    "curveness": 0.15,
                },
            })

    for b in blocked or []:
        src = b.get("source") or b.get("blocked_by")
        tgt = _label(b)
        for nid in filter(None, (src, tgt)):
            if nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "name": nid,
                    "symbolSize": 14,
                    "itemStyle": {"color": "#94a3b8"},
                }
        if src and tgt and src != tgt:
            links.append({
                "source": src,
                "target": tgt,
                "lineStyle": {
                    "type": "dashed",
                    "color": "#94a3b8",
                    "width": 1.5,
                    "opacity": 0.6,
                },
            })

    return list(nodes.values()), links


def mutations_to_propagation_rows(
    mutations: list[dict],
    blocked: list[dict] | None = None,
) -> list[dict]:
    """Tabular companion for ``mutations_to_propagation_graph``."""
    out: list[dict] = []
    for i, m in enumerate(mutations or []):
        out.append({
            "step": i + 1,
            "kind": "mutation",
            "source": m.get("source", ""),
            "target": (
                m.get("entity") or m.get("target") or m.get("name") or ""
            ),
            "trait": m.get("trait") or m.get("trait_target") or "",
            "delta": round(float(
                m.get("delta") or m.get("trait_delta") or 0.0
            ), 3),
            "mechanism": m.get("mechanism", ""),
        })
    base = len(out)
    for j, b in enumerate(blocked or []):
        out.append({
            "step": base + j + 1,
            "kind": "blocked",
            "source": b.get("source") or b.get("blocked_by", ""),
            "target": (
                b.get("entity") or b.get("target") or b.get("name") or ""
            ),
            "trait": b.get("trait") or b.get("trait_target") or "",
            "delta": 0,
            "mechanism": b.get("reason") or b.get("mechanism", ""),
        })
    return out


# ── #9 Calendar-graph overlay ─────────────────────────────────────

def ws_to_calendar_graph_data(
    ws: WorldStateV1,
    *,
    bucket_count: int = 24,
) -> tuple[list[list], list[dict], list[dict], int]:
    """Bucketed event-density bars *plus* node positions for an overlay.

    Returns ``(buckets, nodes, links, max_count)`` where:
      - ``buckets``: ``[[bucket_idx, 0, count], …]`` for the heatmap.
      - ``nodes``: scatter-style nodes positioned at
        ``[bucket_idx, jitter_y]`` for each event.
      - ``links``: chain_reaction links between events that share the
        ``[bucket_idx, jitter_y]`` coordinate space, so causal chains
        are visible across the calendar.
      - ``max_count``: max events in any bucket (for visualMap scaling).
    """
    if not ws.events:
        return [], [], [], 0

    f_vals = [e.fabula_time for e in ws.events]
    f_min, f_max = min(f_vals), max(f_vals)
    span = max(1, f_max - f_min)
    width = max(1, span / bucket_count)

    def _bucket(t: int) -> int:
        return min(bucket_count - 1, int((t - f_min) / width))

    counts: dict[int, int] = {}
    bucket_for_event: dict[str, int] = {}
    for evt in ws.events:
        b = _bucket(evt.fabula_time)
        counts[b] = counts.get(b, 0) + 1
        bucket_for_event[evt.id] = b

    buckets = [[i, 0, counts.get(i, 0)] for i in range(bucket_count)]
    max_count = max((c for _, _, c in buckets), default=0)

    # Stack events vertically inside their bucket so they don't overlap
    bucket_stack: dict[int, int] = {}
    nodes: list[dict] = []
    for evt in sorted(ws.events, key=lambda e: e.fabula_time):
        b = bucket_for_event[evt.id]
        slot = bucket_stack.get(b, 0)
        bucket_stack[b] = slot + 1
        nodes.append({
            "id": evt.id,
            "name": evt.id,
            "value": [b, slot],
            "symbolSize": 10,
            "itemStyle": {
                "color": EVENT_TYPE_COLORS.get(evt.event_type, "#94a3b8"),
            },
            "tooltip": {
                "formatter": (
                    f"<b>{evt.id}</b><br/>"
                    f"t={evt.fabula_time}<br/>"
                    f"{evt.description[:60]}"
                ),
            },
        })

    evt_ids = {e.id for e in ws.events}
    links: list[dict] = []
    for ce in ws.causal_topology:
        if (
            ce.causality_type == "chain_reaction"
            and ce.source_id in evt_ids
            and ce.target_id in evt_ids
        ):
            links.append({
                "source": ce.source_id,
                "target": ce.target_id,
                "lineStyle": {
                    "width": 1.4, "color": EDGE_COLORS["causal"], "opacity": 0.55,
                    "curveness": 0.25,
                },
            })
    return buckets, nodes, links, max_count


def ws_to_calendar_graph_rows(
    ws: WorldStateV1,
    *,
    bucket_count: int = 24,
) -> list[dict]:
    """Tabular companion: events listed by bucket index."""
    if not ws.events:
        return []
    f_vals = [e.fabula_time for e in ws.events]
    f_min, f_max = min(f_vals), max(f_vals)
    span = max(1, f_max - f_min)
    width = max(1, span / bucket_count)

    def _bucket(t: int) -> int:
        return min(bucket_count - 1, int((t - f_min) / width))

    rows = []
    for evt in sorted(ws.events, key=lambda e: (e.fabula_time, e.syuzhet_index)):
        rows.append({
            "bucket": _bucket(evt.fabula_time),
            "fabula_time": evt.fabula_time,
            "id": evt.id,
            "type": evt.event_type,
            "description": evt.description[:80],
        })
    return rows


# ── #12 Multi-snapshot trait radar overlay ────────────────────────

def entity_to_radar_compare_data(
    entity_id: str,
    ws: WorldStateV1,
    times: list[int],
) -> dict:
    """Build a multi-series radar dataset overlaying one entity at N times.

    Returns ``{"indicator": [...], "series": [{"name": "t=…", "value": [...]}, …]}``.
    Used for the "Compare mode" timeline-overlay radar.
    """
    ent = ws.entities.get(entity_id)
    if ent is None:
        return {"indicator": [], "series": []}
    # Stable trait order = union of trait names across all snapshots
    trait_names: list[str] = list(ent.traits.keys())
    snapshots: list[tuple[int, dict]] = []
    for t in sorted(set(times)):
        try:
            snap = reconstruct_entity_at(ent, t)
            snapshots.append((t, snap))
            for k in snap.get("traits", {}).keys():
                if k not in trait_names:
                    trait_names.append(k)
        except Exception:
            continue
    if not trait_names:
        return {"indicator": [], "series": []}
    indicator = [{"name": k, "max": 1.0} for k in trait_names]
    series = []
    for t, snap in snapshots:
        traits = snap.get("traits", {})
        series.append({
            "name": f"t={t}",
            "value": [
                round(float(traits.get(k, {}).get("value", 0.0)), 3)
                for k in trait_names
            ],
        })
    return {"indicator": indicator, "series": series}


def entity_radar_compare_rows(
    entity_id: str,
    ws: WorldStateV1,
    times: list[int],
) -> list[dict]:
    """Tabular companion for ``entity_to_radar_compare_data``."""
    data = entity_to_radar_compare_data(entity_id, ws, times)
    if not data["indicator"]:
        return []
    rows: list[dict] = []
    for ind, *_ in zip(data["indicator"]):
        pass
    trait_names = [d["name"] for d in data["indicator"]]
    for ti, s in enumerate(data["series"]):
        for i, k in enumerate(trait_names):
            rows.append({
                "snapshot": s["name"],
                "trait": k,
                "value": s["value"][i],
            })
    return rows


# ── #15 Causal "explain this" inspector ───────────────────────────

def explain_node_causes(
    ws: WorldStateV1,
    node_id: str,
    *,
    limit: int = 30,
) -> list[dict]:
    """Return the incoming causal edges for ``node_id``, ranked by force.

    Each row is shaped for a NiceGUI table.
    """
    rows: list[dict] = []
    for ce in ws.causal_topology:
        if ce.target_id != node_id:
            continue
        rows.append({
            "source": ce.source_id,
            "type": ce.causality_type,
            "mechanism": ce.mechanism,
            "force": round(ce.causal_force, 2),
            "evidence": ce.evidence_strength,
            "fabula_time": ce.fabula_time,
            "trait_target": ce.trait_target or "",
            "trait_delta": (
                round(ce.trait_delta, 3) if ce.trait_delta is not None else ""
            ),
        })
    rows.sort(key=lambda r: (-r["force"], r["fabula_time"]))
    return rows[:limit]


def explain_node_effects(
    ws: WorldStateV1,
    node_id: str,
    *,
    limit: int = 30,
) -> list[dict]:
    """Outgoing causal edges from ``node_id``, ranked by force."""
    rows: list[dict] = []
    for ce in ws.causal_topology:
        if ce.source_id != node_id:
            continue
        rows.append({
            "target": ce.target_id,
            "type": ce.causality_type,
            "mechanism": ce.mechanism,
            "force": round(ce.causal_force, 2),
            "evidence": ce.evidence_strength,
            "fabula_time": ce.fabula_time,
            "trait_target": ce.trait_target or "",
            "trait_delta": (
                round(ce.trait_delta, 3) if ce.trait_delta is not None else ""
            ),
        })
    rows.sort(key=lambda r: (-r["force"], r["fabula_time"]))
    return rows[:limit]


# ── #18 Audit pass-rate pictorial / data ──────────────────────────

def audit_passrate_data(
    audit_history: list[dict],
) -> tuple[list[str], list[float], list[dict]]:
    """Compute pass-rate per refinement loop iteration.

    ``audit_history`` is a list of per-iteration dicts; each entry is
    expected to have either ``passed``/``total`` ints or a list of issue
    dicts under ``issues``. Returns ``(labels, passrates, rows)``.
    """
    labels: list[str] = []
    passrates: list[float] = []
    rows: list[dict] = []
    for i, it in enumerate(audit_history or []):
        passed = it.get("passed")
        total = it.get("total")
        if passed is None or total is None:
            issues = it.get("issues") or []
            total = len(issues) if issues else int(it.get("checks_total", 0) or 0)
            passed = total - sum(1 for x in issues if not x.get("ok", True))
        try:
            rate = float(passed) / float(total) if total else 1.0
        except (TypeError, ValueError):
            rate = 0.0
        rate = max(0.0, min(1.0, rate))
        label = it.get("label") or f"iter {i + 1}"
        labels.append(label)
        passrates.append(round(rate, 3))
        rows.append({
            "iteration": label,
            "passed": int(passed or 0),
            "total": int(total or 0),
            "pass_rate": round(rate, 3),
            "converged": bool(it.get("converged", False)),
        })
    return labels, passrates, rows


# ── #19 Status-flip markers for the Gantt chart ───────────────────

def ws_to_gantt_status_marks(
    ws: WorldStateV1,
) -> list[dict]:
    """For each entity, the (fabula_time, status) point where status flips.

    Returns rows ``{"actor": entity_name, "fabula_time": int,
    "status": str, "icon": str}`` so the Gantt renderer can decorate
    bars with markPoints (e.g. 💀 on the tick where status="dead").
    """
    icon_for = {
        "dead": "❌",
        "injured": "⚕",
        "ill": "✦",
        "unconscious": "✧",
        "healthy": "✓",
    }
    out: list[dict] = []
    for ent in ws.entities.values():
        last_status = ent.status
        for snap in sorted(ent.state_timeline, key=lambda s: s.fabula_time):
            if snap.status and snap.status != last_status:
                out.append({
                    "actor": ent.name,
                    "actor_id": ent.id,
                    "fabula_time": snap.fabula_time,
                    "status": snap.status,
                    "icon": icon_for.get(snap.status, "•"),
                })
                last_status = snap.status
    return out
