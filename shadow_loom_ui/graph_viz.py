"""pyvis graph rendering utilities for Shadow-Loom.

Provides functions that take a WorldStateV1 (or subsets) and return
self-contained HTML strings suitable for embedding in NiceGUI via ui.html().
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

import networkx as nx
from pyvis.network import Network

from shadow_loom.extract_graph import EgoGraphPayload, extract_ego_graph_from_memory
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)

# Node colours by type
_NODE_COLORS = {
    "Entity": "#4CAF50",
    "Location": "#2196F3",
    "EventNode": "#FF9800",
    "NarrativeObject": "#9C27B0",
    "WorldTrait": "#00BFA5",
}

# Edge colours by type
_EDGE_COLORS = {
    "causal": "#F44336",
    "relationship": "#E91E63",
    "located_in": "#607D8B",
    "owned_by": "#795548",
    "connected_to": "#00BCD4",
    "communicating_with": "#FFEB3B",
    "eavesdropped_by": "#FF5722",
}

_EDGE_DASH = {
    "communicating_with": [5, 5],
    "eavesdropped_by": [2, 4],
}


def _build_pyvis_from_nx(G: nx.MultiDiGraph, height: str = "600px") -> str:
    """Convert a NetworkX MultiDiGraph to a pyvis HTML string."""
    net = Network(height=height, width="100%", directed=True, notebook=False,
                  bgcolor="#1e1e1e", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    for node_id, data in G.nodes(data=True):
        node_type = data.get("node_type", "Unknown")
        label = data.get("name", str(node_id))
        color = _NODE_COLORS.get(node_type, "#9E9E9E")
        title_parts = [f"<b>{label}</b> ({node_id})", f"Type: {node_type}"]
        if node_type == "Entity":
            status = data.get("status", "unknown")
            title_parts.append(f"Status: {status}")
            traits = data.get("traits", {})
            if traits:
                for tn, td in traits.items():
                    val = td.get("value", 0.5) if isinstance(td, dict) else td
                    title_parts.append(f"  {tn}: {val:.2f}")
        elif node_type == "Location":
            desc = data.get("description", "")
            if desc:
                title_parts.append(desc[:120])
        elif node_type == "EventNode":
            desc = data.get("description", "")
            ft = data.get("fabula_time", "?")
            title_parts.append(f"T={ft}: {desc[:100]}")
        elif node_type == "NarrativeObject":
            props = data.get("properties", {})
            if props:
                title_parts.append(str(props))
        elif node_type == "WorldTrait":
            desc = data.get("description", "")
            mag = data.get("magnitude", {})
            mag_val = mag.get("value", "?") if isinstance(mag, dict) else "?"
            domains = ", ".join(data.get("affected_domains", []))
            title_parts.append(f"Magnitude: {mag_val}")
            if domains:
                title_parts.append(f"Domains: {domains}")
            if desc:
                title_parts.append(desc[:120])

        shape = {
            "Entity": "dot",
            "Location": "square",
            "EventNode": "triangle",
            "NarrativeObject": "diamond",
            "WorldTrait": "star",
        }.get(node_type, "dot")

        net.add_node(
            str(node_id), label=label, color=color, shape=shape,
            title="<br>".join(title_parts),
            size=25 if node_type == "Entity" else 20,
        )

    for u, v, data in G.edges(data=True):
        edge_type = data.get("edge_type", "unknown")
        color = _EDGE_COLORS.get(edge_type, "#BDBDBD")
        width = 2
        dashes = _EDGE_DASH.get(edge_type, False)

        title_parts = [edge_type]
        if edge_type == "causal":
            mech = data.get("mechanism", "")
            force = data.get("causal_force", "")
            ctype = data.get("causality_type", "")
            title_parts.append(f"{ctype}: {mech} (force={force})")
            width = min(5, max(1, float(force) / 2)) if force else 2
        elif edge_type == "relationship":
            aff = data.get("affinity", 0)
            fear = data.get("fear", 0)
            pwr = data.get("power_dynamic", 0)
            title_parts.append(f"affinity={aff:.2f} fear={fear:.2f} power={pwr:.2f}")

        net.add_edge(
            str(u), str(v), color=color, width=width,
            title="<br>".join(title_parts),
            dashes=dashes,
        )

    return net.generate_html()


def render_world_graph(world_state: WorldStateV1, height: str = "600px") -> str:
    """Render the full world state as an interactive pyvis graph.

    Returns an HTML string.
    """
    G = nx.MultiDiGraph()

    # Nodes
    for eid, ent in world_state.entities.items():
        G.add_node(eid, node_type="Entity", name=ent.name,
                   status=ent.status, location_id=ent.location_id,
                   traits={k: {"value": v.value, "inertia": v.inertia}
                           for k, v in ent.traits.items()})

    for lid, loc in world_state.locations.items():
        G.add_node(lid, node_type="Location", name=loc.name,
                   description=loc.description)

    for oid, obj in world_state.objects.items():
        G.add_node(oid, node_type="NarrativeObject", name=obj.name,
                   location_id=obj.location_id, owner_id=obj.owner_id,
                   properties=obj.properties)

    for evt in world_state.events:
        G.add_node(evt.id, node_type="EventNode", name=evt.id,
                   description=evt.description, fabula_time=evt.fabula_time)

    # World Traits
    for wid, wt in world_state.world_traits.items():
        G.add_node(wid, node_type="WorldTrait", name=wt.name,
                   description=wt.description,
                   magnitude={"value": wt.magnitude.value,
                              "inertia": wt.magnitude.inertia},
                   affected_domains=wt.affected_domains)

    # Edges: entity → location
    for eid, ent in world_state.entities.items():
        if ent.location_id in world_state.locations:
            G.add_edge(eid, ent.location_id, edge_type="located_in")

    # Object edges
    for oid, obj in world_state.objects.items():
        if obj.owner_id and obj.owner_id in world_state.entities:
            G.add_edge(oid, obj.owner_id, edge_type="owned_by")
        elif obj.location_id and obj.location_id in world_state.locations:
            G.add_edge(oid, obj.location_id, edge_type="located_in")

    # Causal topology
    all_ids = set(G.nodes)
    for ce in world_state.causal_topology:
        if ce.source_id in all_ids and ce.target_id in all_ids:
            G.add_edge(ce.source_id, ce.target_id, edge_type="causal",
                       mechanism=ce.mechanism, causal_force=ce.causal_force,
                       causality_type=ce.causality_type)

    # Spatial topology
    for se in world_state.spatial_topology:
        if se.source_id in all_ids and se.target_id in all_ids:
            G.add_edge(se.source_id, se.target_id, edge_type="connected_to",
                       is_locked=se.is_locked)
            G.add_edge(se.target_id, se.source_id, edge_type="connected_to",
                       is_locked=se.is_locked)

    # Social topology
    for rel in world_state.social_topology:
        if rel.source_entity_id in all_ids and rel.target_entity_id in all_ids:
            G.add_edge(rel.source_entity_id, rel.target_entity_id,
                       edge_type="relationship",
                       affinity=rel.affinity, fear=rel.fear,
                       power_dynamic=rel.power_dynamic)

    # Information topology
    for ie in world_state.information_topology:
        if ie.source_id in all_ids:
            for tid in ie.target_ids:
                if tid in all_ids:
                    G.add_edge(ie.source_id, tid, edge_type="communicating_with",
                               medium=ie.medium, is_encrypted=ie.is_encrypted)

    return _build_pyvis_from_nx(G, height=height)


def render_ego_graph(
    world_state: WorldStateV1,
    focus_entity_ids: List[str],
    temporal_anchor: Optional[int] = None,
    height: str = "600px",
) -> str:
    """Render an ego-graph centered on specific entities.

    Uses the existing extract_ego_graph_from_memory + AMWNInstantiator
    pipeline, then converts to pyvis HTML.
    """
    ego = extract_ego_graph_from_memory(
        world_state, focus_entity_ids,
        temporal_anchor=temporal_anchor,
    )
    sandbox = AMWNInstantiator.create_sandbox(ego.model_dump(), "observation")
    return _build_pyvis_from_nx(sandbox, height=height)


def render_causal_subgraph(
    world_state: WorldStateV1,
    event_ids: Optional[Set[str]] = None,
    height: str = "600px",
) -> str:
    """Render just the causal topology (events + causal edges).

    If *event_ids* is provided, only shows those events and their
    direct causal neighbours.
    """
    G = nx.MultiDiGraph()

    target_ids: Set[str] = set()
    if event_ids:
        target_ids = set(event_ids)
        # Add 1-hop causal neighbours
        for ce in world_state.causal_topology:
            if ce.source_id in target_ids:
                target_ids.add(ce.target_id)
            if ce.target_id in target_ids:
                target_ids.add(ce.source_id)

    for evt in world_state.events:
        if event_ids and evt.id not in target_ids:
            continue
        G.add_node(evt.id, node_type="EventNode", name=evt.id,
                   description=evt.description, fabula_time=evt.fabula_time)

    # Also add entity/state nodes referenced by causal edges
    graph_ids = set(G.nodes)
    for ce in world_state.causal_topology:
        src_ok = ce.source_id in graph_ids or not event_ids
        tgt_ok = ce.target_id in graph_ids or not event_ids
        if event_ids and not (ce.source_id in target_ids or ce.target_id in target_ids):
            continue
        # Add entity/location nodes as needed
        for nid in (ce.source_id, ce.target_id):
            if nid not in G.nodes:
                if nid in world_state.entities:
                    ent = world_state.entities[nid]
                    G.add_node(nid, node_type="Entity", name=ent.name)
                elif nid in world_state.locations:
                    loc = world_state.locations[nid]
                    G.add_node(nid, node_type="Location", name=loc.name)
                elif nid in world_state.objects:
                    obj = world_state.objects[nid]
                    G.add_node(nid, node_type="NarrativeObject", name=obj.name)
                elif nid in world_state.world_traits:
                    wt = world_state.world_traits[nid]
                    G.add_node(nid, node_type="WorldTrait", name=wt.name)
        if ce.source_id in G.nodes and ce.target_id in G.nodes:
            G.add_edge(ce.source_id, ce.target_id, edge_type="causal",
                       mechanism=ce.mechanism, causal_force=ce.causal_force,
                       causality_type=ce.causality_type)

    return _build_pyvis_from_nx(G, height=height)
