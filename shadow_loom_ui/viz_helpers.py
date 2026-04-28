"""Data transformation helpers: WorldStateV1 → ECharts option dicts.

Pure functions that convert Shadow-Loom world model objects into the
node/link/category structures consumed by Apache ECharts series configs.
No NiceGUI imports — this module is purely data-oriented.
"""

from __future__ import annotations

from typing import Any

from shadow_loom.models import WorldStateV1, reconstruct_entity_at

# ── Visual constants ────────────────────────────────────────────────

# Mort-artistic palette (warm, muted)
NODE_COLORS: dict[str, str] = {
    "Entity": "#C68661",          # Copper
    "Location": "#5C7C8A",        # Slate Blue
    "EventNode": "#D4A35B",       # Aged Gold
    "NarrativeObject": "#856B7D", # Faded Plum
    "WorldTrait": "#456A6B",      # Deep Spruce
}

EDGE_COLORS: dict[str, str] = {
    "causal": "#9E4D4D",            # Dusty Brick
    "relationship": "#B58988",      # Dusty Rose
    "located_in": "#8C7A6B",        # Warm Clay
    "owned_by": "#8C7A6B",          # Warm Clay
    "connected_to": "#5C7C8A",      # Slate Blue
    "communicating_with": "#D4A35B",# Aged Gold
    "eavesdropped_by": "#9E4D4D",   # Dusty Brick
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
    "choice": "#5C7C8A",      # Slate Blue
    "outcome": "#D4A35B",     # Aged Gold
    "revelation": "#856B7D",  # Faded Plum
    "action": "#95A577",      # Olive
    "dialogue": "#C68661",    # Copper
    "transition": "#8C7A6B",  # Warm Clay
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

    # Spatial topology
    for se in ws.spatial_topology:
        _link(se.source_id, se.target_id, "connected_to")
        _link(se.target_id, se.source_id, "connected_to")

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


# ── Causal Sankey ──────────────────────────────────────────────────

def ws_to_sankey_data(
    ws: WorldStateV1,
) -> tuple[list[dict], list[dict]]:
    """Build Sankey ``(nodes, links)`` from causal topology.

    Nodes are events ordered by fabula_time; links are causal edges with
    ``value`` equal to ``causal_force``.
    """
    event_map = {evt.id: evt for evt in ws.events}
    entity_map = {eid: ent.name for eid, ent in ws.entities.items()}
    loc_map = {lid: loc.name for lid, loc in ws.locations.items()}
    wt_map = {wid: wt.name for wid, wt in ws.world_traits.items()}

    seen_ids: set[str] = set()
    nodes: list[dict] = []
    links: list[dict] = []

    def _ensure_node(nid: str) -> None:
        if nid in seen_ids:
            return
        seen_ids.add(nid)
        if nid in event_map:
            label = f"t{event_map[nid].fabula_time}: {event_map[nid].description[:30]}"
        elif nid in entity_map:
            label = entity_map[nid]
        elif nid in loc_map:
            label = loc_map[nid]
        elif nid in wt_map:
            label = wt_map[nid]
        else:
            label = nid
        nodes.append({"name": nid, "label": label})

    for ce in ws.causal_topology:
        _ensure_node(ce.source_id)
        _ensure_node(ce.target_id)
        links.append({
            "source": ce.source_id,
            "target": ce.target_id,
            "value": max(0.5, ce.causal_force),
        })

    return nodes, links


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

def ws_to_sunburst_data(ws: WorldStateV1) -> dict:
    """Build sunburst hierarchy: World → Locations → Entities → Traits."""
    children: list[dict] = []

    for lid, loc in ws.locations.items():
        loc_children: list[dict] = []
        for eid, ent in ws.entities.items():
            if ent.location_id == lid:
                trait_children = [
                    {
                        "name": tname,
                        "value": max(1, int(tv.value * 10)),
                        "itemStyle": {"color": NODE_COLORS.get("Entity", "#4CAF50")},
                    }
                    for tname, tv in list(ent.traits.items())[:6]
                ]
                loc_children.append({
                    "name": ent.name,
                    "value": max(1, len(ent.traits)),
                    "children": trait_children,
                    "itemStyle": {"color": NODE_COLORS["Entity"]},
                })
        for oid, obj in ws.objects.items():
            if obj.location_id == lid:
                loc_children.append({
                    "name": obj.name,
                    "value": 1,
                    "itemStyle": {"color": NODE_COLORS["NarrativeObject"]},
                })
        children.append({
            "name": loc.name,
            "value": max(1, len(loc_children)),
            "children": loc_children,
            "itemStyle": {"color": NODE_COLORS["Location"]},
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
                "itemStyle": {"color": NODE_COLORS["WorldTrait"]},
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
