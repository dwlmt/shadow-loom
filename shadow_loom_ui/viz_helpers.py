# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data transformation helpers: WorldStateV1 → ECharts option dicts.

Pure functions that convert Shadow-Loom world model objects into the
node/link/category structures consumed by Apache ECharts series configs.
No NiceGUI imports — this module is purely data-oriented.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

from shadow_loom.models import (
    CausalEdge,
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
)

logger = logging.getLogger(__name__)


# ── Causal-physics-aware state reconstruction ─────────────────────
#
# The model-level ``reconstruct_*_at`` functions in ``shadow_loom.models``
# only replay authored ``state_timeline`` snapshots. The helpers below
# layer ``CausalEdge`` mutations on top so every time-evolution view in
# the UI reflects how the causal graph actually moves traits and world
# magnitudes — using snapshots as authoritative overrides when present.

def reconstruct_entity_with_causal(
    ws: WorldStateV1,
    entity_id: str,
    fabula_time: int,
) -> dict:
    """Reconstruct an entity at ``fabula_time`` using both causal physics
    and authored snapshots.

    1. Seed traits from ``Entity.traits`` (pre-story baseline).
    2. Replay every ``mutation`` :class:`CausalEdge` whose ``target_id``
       is this entity and whose ``fabula_time <= fabula_time``,
       accumulating signed ``trait_delta`` per ``trait_target`` (clamped
       to ``[-1, 1]``).
    3. Apply the model-level ``reconstruct_entity_at`` snapshot replay
       on top — explicit snapshots override running causal values at
       their tick (status, location, beliefs, and any trait values).

    Returns the same shape as :func:`reconstruct_entity_at`.
    """
    ent = ws.entities.get(entity_id)
    if ent is None:
        return {"traits": {}, "beliefs": [], "status": "healthy", "location_id": ""}

    # Causal replay
    running: dict[str, dict] = {
        k: {
            "value": v.value,
            "inertia": v.inertia,
            "evidence_strength": v.evidence_strength,
        }
        for k, v in ent.traits.items()
    }
    mutations = sorted(
        (
            ce for ce in ws.causal_topology
            if ce.causality_type == "mutation"
            and ce.target_id == entity_id
            and ce.trait_target
            and ce.trait_delta is not None
            and ce.fabula_time <= fabula_time
        ),
        key=lambda c: c.fabula_time,
    )
    for ce in mutations:
        cur = running.get(
            ce.trait_target,
            {"value": 0.0, "inertia": 0.5, "evidence_strength": "moderate"},
        )
        new_val = max(-1.0, min(1.0, cur["value"] + (ce.trait_delta or 0.0)))
        running[ce.trait_target] = {
            "value": new_val,
            "inertia": cur["inertia"],
            "evidence_strength": cur.get("evidence_strength", "moderate"),
        }

    # Snapshot overlay (authoritative)
    snap = reconstruct_entity_at(ent, fabula_time)
    for tn, tv in snap.get("traits", {}).items():
        if tn in ent.traits or tn in running:
            # Snapshots take precedence at their tick — replace running
            # value with the authored one so any later causal edges
            # accumulate from the corrected base on subsequent ticks.
            running[tn] = tv if isinstance(tv, dict) else {"value": float(tv), "inertia": 0.5}

    return {
        "traits": running,
        "beliefs": snap["beliefs"],
        "status": snap["status"],
        "location_id": snap["location_id"],
    }


def reconstruct_world_trait_with_causal(
    ws: WorldStateV1,
    world_id: str,
    fabula_time: int,
) -> dict:
    """Reconstruct a :class:`GlobalTrait` at ``fabula_time`` using both
    causal physics and authored snapshots.

    World traits can be moved by ``mutation`` causal edges (Event →
    WORLD_*, allowed by :class:`CausalEdge`'s validator). We replay
    those signed ``trait_delta`` values onto the baseline ``magnitude``,
    then let any :class:`WorldTraitSnapshot` override at its tick.

    Returns the same shape as :func:`reconstruct_world_trait_at`.
    """
    wt = ws.world_traits.get(world_id)
    if wt is None:
        return {"magnitude": {"value": 0.0, "inertia": 0.0}, "description": ""}

    value = wt.magnitude.value
    inertia = wt.magnitude.inertia
    evidence_strength = wt.magnitude.evidence_strength
    mutations = sorted(
        (
            ce for ce in ws.causal_topology
            if ce.causality_type == "mutation"
            and ce.target_id == world_id
            and ce.trait_delta is not None
            and ce.fabula_time <= fabula_time
        ),
        key=lambda c: c.fabula_time,
    )
    for ce in mutations:
        value = max(0.0, min(1.0, value + (ce.trait_delta or 0.0)))

    snap = reconstruct_world_trait_at(wt, fabula_time)
    # Snapshot magnitude wins when explicitly set.
    snap_mag = snap.get("magnitude") or {}
    if snap_mag and (
        snap_mag.get("value") != wt.magnitude.value
        or snap_mag.get("inertia") != wt.magnitude.inertia
    ):
        value = snap_mag.get("value", value)
        inertia = snap_mag.get("inertia", inertia)

    return {
        "magnitude": {
            "value": value,
            "inertia": inertia,
            "evidence_strength": evidence_strength,
        },
        "description": snap.get("description", wt.description),
    }


def reconstruct_relationship_with_causal(
    ws: WorldStateV1,
    source_entity_id: str,
    target_entity_id: str,
    fabula_time: int,
) -> dict | None:
    """Reconstruct relationship metrics for ``(source, target)`` at ``t``.

    Convention (matches the causal_physics engine, see
    ``CausalPhysics._seed_active_sources``): each
    :class:`RelationshipMetric.value` is the **post-canonical** state
    at exactly ``last_updated_fabula`` — every ``mutation_social``
    edge with ``fabula_time <= last_updated_fabula`` has *already*
    contributed to that value. We therefore reconstruct the value at
    an arbitrary ``t`` by **walking back** from ``last_updated_fabula``,
    subtracting the cumulative delta of mutations whose
    ``fabula_time`` lies in ``(t, last_updated_fabula]``.

    Adding mutations on top of ``value`` (the previous behaviour)
    double-counted every canonical effect and silently drove every
    timeline trace into clamp saturation — the visual symptom was
    "the chart is flat at ±1.0 from the moment of last_updated".

    Pair selection prefers the **directed** match
    (``source→target``) so asymmetric relationships (where the dyad
    is authored in both directions with different values, e.g. Luke
    →Obi-Wan affinity=0.9 vs Obi-Wan→Luke affinity=0.7) reconstruct
    against the correct half.

    Returns ``None`` if no :class:`RelationshipEdge` exists for the
    pair, otherwise a dict matching the shape of a serialised edge
    with reconstructed ``affinity``, ``fear``, ``power_dynamic`` AND
    a ``per_axis`` sub-dict carrying value + evidence + observed +
    last_updated_fabula for each axis.

    Clamping is per-metric — affinity/power_dynamic in [-1, 1],
    fear in [0, 1] — matching the model documentation.
    """
    # Prefer the exact directed match. Fall back to the reversed dyad
    # only when no directed edge exists (legacy fixtures sometimes
    # record only one half).
    base = next(
        (
            r for r in ws.social_topology
            if r.source_entity_id == source_entity_id
            and r.target_entity_id == target_entity_id
        ),
        None,
    )
    if base is None:
        base = next(
            (
                r for r in ws.social_topology
                if r.source_entity_id == target_entity_id
                and r.target_entity_id == source_entity_id
            ),
            None,
        )
    if base is None:
        return None

    per_axis: dict[str, dict] = {}
    for axis_name in ("affinity", "fear", "power_dynamic"):
        m = base.metrics.get(axis_name)
        if m is not None:
            per_axis[axis_name] = {
                "value": m.value,
                "evidence_strength": m.evidence_strength,
                "observed": m.observed,
                "last_updated_fabula": m.last_updated_fabula or 0,
                "inertia": m.inertia,
            }
        else:
            per_axis[axis_name] = {
                "value": 0.0,
                "evidence_strength": "weak",
                "observed": False,
                "last_updated_fabula": 0,
                "inertia": 0.3,
            }

    pair = {source_entity_id, target_entity_id}
    mutations = sorted(
        (
            ce for ce in ws.causal_topology
            if ce.causality_type == "mutation_social"
            and ce.target_id in pair
            and ce.trait_target
            and ce.trait_delta is not None
        ),
        key=lambda c: c.fabula_time,
    )
    # Walk back from each axis' authored ``value`` (state at
    # ``last_updated_fabula``) by undoing mutations that occurred in
    # ``(t, last_updated_fabula]``. For ``t >= last_updated_fabula``
    # the value is simply the authored one — no extrapolation.
    def _clamp(metric: str, v: float) -> float:
        if metric == "fear":
            return max(0.0, min(1.0, v))
        return max(-1.0, min(1.0, v))

    for axis_name, info in per_axis.items():
        last_upd = info["last_updated_fabula"]
        if fabula_time >= last_upd:
            continue  # authored value already valid at t
        rollback = 0.0
        for ce in mutations:
            metric = (ce.trait_target or "").lower()
            if metric == "power":
                metric = "power_dynamic"
            if metric != axis_name:
                continue
            # Only undo mutations strictly after t and at or before
            # the axis' last_updated_fabula. Mutations whose
            # fabula_time exceeds last_updated_fabula are ignored —
            # they cannot legitimately have contributed to ``value``.
            if ce.fabula_time > fabula_time and ce.fabula_time <= last_upd:
                rollback += float(ce.trait_delta or 0.0)
        info["value"] = _clamp(axis_name, info["value"] - rollback)

    out = base.model_dump()
    for axis_name, info in per_axis.items():
        out[axis_name] = info["value"]
    out["per_axis"] = {
        name: {
            "value": info["value"],
            "evidence_strength": info["evidence_strength"],
            "observed": info["observed"],
            "last_updated_fabula": info["last_updated_fabula"],
        }
        for name, info in per_axis.items()
    }
    return out


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
    "Channel": "#C46BD9",         # Orchid
}

EDGE_COLORS: dict[str, str] = {
    "causal": "#D8334A",            # Crimson
    "relationship": "#E36BB8",      # Magenta Rose
    "located_in": "#FF8C42",        # Tangerine
    "owned_by": "#FF8C42",          # Tangerine
    "connected_to": "#3A7BD5",      # Sapphire
    "communicating_with": "#F5B43C",# Amber
    "eavesdropped_by": "#D8334A",   # Crimson
    "actor_of": "#6FBF3A",          # Spring Green — entity acted in event
    "target_of": "#FF6B6B",         # Soft Red    — entity/object was acted upon
    "used_in": "#FFB84D",           # Light Amber — object used in event
    "governs": "#8A5CF0",           # Iris        — world trait constrains event
}

# World-trait domains that match causal-edge mechanism strings.
# A WorldTrait whose ``affected_domains`` contains any of these keys
# is considered to govern an event whose incoming CausalEdges carry a
# matching mechanism substring.
_DOMAIN_MECHANISM_HINTS: dict[str, tuple[str, ...]] = {
    "physical":      ("physical", "kinetic", "chemical", "force"),
    "psychological": ("psychological", "trauma", "cognitive"),
    "epistemic":     ("epistemic", "revelation", "deception", "discovery"),
    "social":        ("social", "coercion", "alliance", "shaming"),
    "emotional":     ("emotional", "fear", "love", "anger"),
    "informational": ("informational", "broadcast", "leak", "communication"),
    "betrayal":      ("betrayal", "treachery"),
}

NODE_SYMBOLS: dict[str, str] = {
    "Entity": "circle",
    "Location": "rect",
    "EventNode": "triangle",
    "NarrativeObject": "diamond",
    "WorldTrait": "pin",
    "Channel": "roundRect",
}

NODE_SIZES: dict[str, int] = {
    "Entity": 30,
    "Location": 25,
    "EventNode": 20,
    "NarrativeObject": 18,
    "WorldTrait": 22,
    "Channel": 22,
}

CATEGORIES: list[dict[str, str]] = [
    {"name": "Entity"},
    {"name": "Location"},
    {"name": "EventNode"},
    {"name": "NarrativeObject"},
    {"name": "WorldTrait"},
    {"name": "Channel"},
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

    # Social topology — pick the per-axis metric with the largest
    # absolute value among observed axes for edge width, so a
    # high-fear / zero-affinity dyad isn't rendered as a flat baseline
    # line just because the legacy aggregate keyed off ``affinity``.
    for rel in ws.social_topology:
        observed_mags = [
            abs(m.value) for name, m in rel.metrics.items()
            if m.observed
        ]
        magnitude = max(observed_mags) if observed_mags else 0.0
        _link(rel.source_entity_id, rel.target_entity_id, "relationship",
              width=max(1, magnitude * 3))

    # Channels (standing comms capabilities) — render as first-class
    # nodes so a click can select the channel itself; participants are
    # linked to the channel node with dashed amber lines.
    for cid, ch in ws.channels.items():
        intel_summary = ", ".join(
            f"{pid}={p:.2f}" for pid, p in list(ch.intelligibility.items())[:3]
        ) if ch.intelligibility else "all 1.0"
        _node(
            cid, ch.name, "Channel",
            medium=ch.medium,
            directionality=ch.directionality,
            participants=str(len(ch.participant_ids)),
            intelligibility=intel_summary,
            _sl_node_type="Channel",
        )
        for pid in ch.participant_ids:
            _link(pid, cid, "communicating_with", dash="dashed")
    # Discrete utterance events: actor (or speaker) → each addressee.
    for evt in ws.events:
        if evt.event_type != "utterance":
            continue
        sender = evt.speaker_id or (evt.actor_ids[0] if evt.actor_ids else None)
        if not sender:
            continue
        for aid in evt.addressee_ids:
            _link(sender, aid, "utters_to")

    # ── Event participation edges ──
    # These were missing from the overview, so events appeared
    # disconnected from the very characters and objects that acted in
    # them (the only path between an entity and an event was via an
    # explicit ``CausalEdge``, which most authored worlds don't carry
    # for every actor/target pair). Without them the overview can't
    # answer "who is involved in what?" at a glance.
    for evt in ws.events:
        for aid in evt.actor_ids:
            _link(aid, evt.id, "actor_of", width=1.5)
        for tid in evt.target_ids:
            # Distinguish entity-target vs object/location-target so
            # the legend reads cleanly.
            etype = "target_of" if tid in ws.entities else "used_in"
            _link(evt.id, tid, etype, width=1.5)

    # ── World-trait → Event "governs" edges ──
    # If an event has any incoming CausalEdge whose mechanism falls
    # in a WorldTrait's ``affected_domains``, draw a soft governance
    # edge from the trait to the event. This makes the "constraint
    # field" of the world legible without needing the author to wire
    # an explicit WORLD_→EVT_ causal edge for every interaction.
    if ws.world_traits and ws.events:
        # Per-event mechanism set
        evt_mechs: dict[str, set[str]] = {evt.id: set() for evt in ws.events}
        for ce in ws.causal_topology:
            if ce.target_id in evt_mechs:
                evt_mechs[ce.target_id].add((ce.mechanism or "").lower())

        for wid, wt in ws.world_traits.items():
            domain_hints: set[str] = set()
            for d in (wt.affected_domains or []):
                domain_hints.update(_DOMAIN_MECHANISM_HINTS.get(d, (d,)))
            if not domain_hints:
                continue
            for evt in ws.events:
                mechs = evt_mechs.get(evt.id, set())
                if any(
                    any(h in m for h in domain_hints) for m in mechs if m
                ):
                    # Avoid double-linking if an explicit WORLD→EVT
                    # CausalEdge already exists.
                    already = any(
                        ce.source_id == wid and ce.target_id == evt.id
                        for ce in ws.causal_topology
                    )
                    if not already:
                        _link(wid, evt.id, "governs", dash="dotted",
                              width=1.0)

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
    for ch in ws.channels.values():
        pids = list(ch.participant_ids)
        for i, src in enumerate(pids):
            for tgt in pids[i + 1:]:
                adj.setdefault(src, set()).add(tgt)
                adj.setdefault(tgt, set()).add(src)
    for evt in ws.events:
        if evt.event_type != "utterance":
            continue
        sender = evt.speaker_id or (evt.actor_ids[0] if evt.actor_ids else None)
        if not sender:
            continue
        for aid in evt.addressee_ids:
            adj.setdefault(sender, set()).add(aid)
            adj.setdefault(aid, set()).add(sender)

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
    """Sankey of communication: speaker → each addressee per utterance event,
    plus standing channel links between every participant pair."""
    def _iter():
        for evt in ws.events:
            if evt.event_type != "utterance":
                continue
            ch = ws.channels.get(evt.via_channel_id) if evt.via_channel_id else None
            tip = (
                f"medium: {ch.medium if ch else 'unmediated'}<br/>"
                f"truth: {evt.truth_value or 'unspecified'}<br/>"
                f"syuzhet={evt.syuzhet_index} fabula={evt.fabula_time}"
            )
            sender = evt.speaker_id or (evt.actor_ids[0] if evt.actor_ids else None)
            if not sender:
                continue
            for aid in evt.addressee_ids:
                yield (sender, aid, 1.0, tip)
        for ch in ws.channels.values():
            tip = (
                f"medium: {ch.medium}<br/>"
                f"directionality: {ch.directionality}<br/>"
                f"established t={ch.established_at_fabula}"
            )
            pids = list(ch.participant_ids)
            for i, src in enumerate(pids):
                for tgt in pids[i + 1:]:
                    yield (src, tgt, 0.5, tip)
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


def events_at_times(
    ws: WorldStateV1,
    times: list[int],
) -> dict[int, list[dict]]:
    """Group events by fabula_time, restricted to ``times``.

    Returned dict maps each time present in ``times`` to a list of
    event dicts ``{id, event_type, description}``. Used by the
    stepped-line timeline charts (entity / relationship / world-trait)
    to overlay markers explaining *why* a value changed at a tick.
    """
    if not times:
        return {}
    wanted = set(times)
    out: dict[int, list[dict]] = {}
    for evt in ws.events:
        if evt.fabula_time in wanted:
            out.setdefault(evt.fabula_time, []).append({
                "id": evt.id,
                "event_type": evt.event_type,
                "description": (evt.description or "")[:120],
            })
    return out


def event_overlay_series(
    ws: WorldStateV1,
    times: list[int],
    *,
    name: str = "events",
    y_value: float = 0.0,
) -> dict | None:
    """Build a scatter series overlaying event markers at ``times``.

    For each fabula_time in ``times`` that has one or more events, emit
    a single scatter point at ``(time_string, y_value)`` whose data
    payload carries the event ids/types/descriptions. The
    accompanying chart's ``tooltip.formatter`` (added by callers) then
    surfaces the per-event detail on hover.

    Returns ``None`` when no events fall on any of the chart's times,
    so callers can ``if series: series.append(overlay)``.

    The series is keyed off the chart's category x-axis (``time_str``)
    so it lines up with stepped-line charts that also use the same
    ``[str(t) for t in times]`` axis labels.
    """
    grouped = events_at_times(ws, times)
    if not grouped:
        return None
    data: list[dict] = []
    for t in times:
        evs = grouped.get(t)
        if not evs:
            continue
        # Compose a multi-line label; rendered via ``:formatter`` on
        # the chart's tooltip (axis trigger picks this series up at the
        # hovered category).
        lines = "<br/>".join(
            f"<b>{e['id']}</b> [{e['event_type']}]: {e['description']}"
            for e in evs[:5]
        )
        if len(evs) > 5:
            lines += f"<br/>… (+{len(evs) - 5} more)"
        data.append({
            "name": f"events@{t}",
            "value": [str(t), y_value],
            "events": evs,
            "events_html": lines,
            "symbolSize": 10 if len(evs) == 1 else 14,
            "itemStyle": {
                "color": EVENT_TYPE_COLORS.get(evs[0]["event_type"], "#607D8B"),
                "borderColor": "#1e293b",
                "borderWidth": 1,
                "opacity": 0.85,
            },
        })
    if not data:
        return None
    return {
        "name": name,
        "type": "scatter",
        "data": data,
        "symbol": "diamond",
        "z": 20,
        "tooltip": {
            ":formatter": (
                "function(p){"
                " if(p.data && p.data.events_html){"
                "  return '<b>Events at t=' + p.data.value[0] +"
                "    '</b><br/>' + p.data.events_html;"
                " }"
                " return p.name;"
                "}"
            ),
        },
        "legendHoverLink": True,
    }


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
        "channels": len(ws.channels),
        "utterance_events": sum(1 for e in ws.events if e.event_type == "utterance"),
    }


# ── Entity state timeline (stepped trait evolution) ───────────────

def entity_state_timeline_data(
    entity_id: str,
    ws: WorldStateV1,
) -> dict:
    """Build line-chart data for an entity's trait evolution over fabula_time.

    Trait values come from causal physics in conjunction with the
    authored world model: at each sample point we call
    :func:`reconstruct_entity_with_causal`, which seeds from
    ``Entity.traits``, replays every ``mutation`` :class:`CausalEdge`
    targeting the entity in fabula order, and lets any explicit
    :class:`EntityStateSnapshot` override the running value.

    Sample points are the union of all event fabula_times,
    mutation-edge fabula_times, and snapshot fabula_times.

    Returns ``{"times": [...], "series": {trait_name: [values]}}``.
    """
    ent = ws.entities.get(entity_id)
    if ent is None:
        return {"times": [], "series": {}}

    # ── Collect mutation edges targeting this entity ──────────────
    mutations = sorted(
        (
            ce for ce in ws.causal_topology
            if ce.causality_type == "mutation"
            and ce.target_id == entity_id
            and ce.trait_target
            and ce.trait_delta is not None
        ),
        key=lambda c: c.fabula_time,
    )

    # ── Build the universe of trait names (initial + ever-mutated) ─
    trait_names: list[str] = list(ent.traits.keys())
    for ce in mutations:
        if ce.trait_target and ce.trait_target not in trait_names:
            trait_names.append(ce.trait_target)

    # ── Sample points ─────────────────────────────────────────────
    sample_set: set[int] = {evt.fabula_time for evt in ws.events}
    sample_set.update(ce.fabula_time for ce in mutations)
    sample_set.update(snap.fabula_time for snap in ent.state_timeline)
    if not sample_set:
        return {"times": [], "series": {}}
    times = sorted(sample_set)

    series: dict[str, list[float]] = {tn: [] for tn in trait_names}
    for t in times:
        snap_state = reconstruct_entity_with_causal(ws, entity_id, t)
        snap_traits = snap_state.get("traits", {}) or {}
        for tn in trait_names:
            tv = snap_traits.get(tn)
            if tv is None:
                # Trait not yet introduced — fall back to baseline.
                base = ent.traits.get(tn)
                series[tn].append(round(base.value if base else 0.0, 3))
            else:
                val = tv["value"] if isinstance(tv, dict) else tv
                series[tn].append(round(float(val), 3))

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
        is_shadow = (v.get("world_id") or "factual") == "shadow"
        # Factual mainline = green, current selection = gold; shadow
        # forks render in violet so the AMWN branch is visually
        # distinguishable from the canonical timeline at a glance
        # (Story-integration plan, Step 3).
        if is_current:
            base_color = "#FFD700"
        elif is_shadow:
            base_color = "#8B5CF6"
        else:
            base_color = "#4CAF50"
        label_suffix = ""
        if is_shadow and v.get("branch_label"):
            label_suffix = f" \u2014 {v['branch_label']}"
        by_id[vid] = {
            "name": f"v{v['version']}{label_suffix}",
            "value": v.get("source", ""),
            "children": [],
            "itemStyle": {
                "color": base_color,
                "borderColor": "#7C3AED" if is_shadow else None,
                "borderWidth": 3 if is_current else (2 if is_shadow else 1),
            },
            "label": {"fontWeight": "bold" if is_current else "normal"},
            "_vid": vid,
            "_version": v["version"],
            "_world_id": v.get("world_id", "factual"),
            "_branch_label": v.get("branch_label"),
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


def list_believers(ws: WorldStateV1) -> list[tuple[str, str, int]]:
    """Return ``(entity_id, name, belief_count)`` for entities with beliefs.

    Sorted by belief_count descending so the most epistemically active
    characters surface first.
    """
    rows: list[tuple[str, str, int]] = []
    for eid, ent in ws.entities.items():
        if ent.beliefs:
            rows.append((eid, ent.name, len(ent.beliefs)))
    rows.sort(key=lambda r: (-r[2], r[1]))
    return rows


def ws_to_entity_belief_rows(
    ws: WorldStateV1,
    entity_id: str,
) -> list[dict]:
    """Build per-belief rows for one believer's belief panel.

    Each row has ``target_name``, ``perceived_state``, ``confidence``,
    ``inertia``, ``established`` (fabula time), and ``correct``
    (heuristic: ``True`` iff the perceived state is consistent with the
    target's actual current state). Used by ``render_entity_belief_chart``
    to render a horizontal bar per belief.
    """
    ent = ws.entities.get(entity_id)
    if ent is None or not ent.beliefs:
        return []

    rows: list[dict] = []
    for b in ent.beliefs:
        target = ws.entities.get(b.target_id)
        if target is not None:
            target_name = target.name
        elif b.target_id in ws.objects:
            target_name = ws.objects[b.target_id].name
        elif b.target_id in ws.locations:
            target_name = ws.locations[b.target_id].name
        else:
            target_name = b.target_id

        # Provenance: how did this entity acquire the belief?
        via_evt_id = getattr(b, "acquired_via_event_id", None)
        via_chn_id = getattr(b, "acquired_via_channel_id", None)
        via_channel_name = (
            ws.channels[via_chn_id].name
            if via_chn_id and via_chn_id in ws.channels
            else None
        )
        via_event_label = None
        if via_evt_id:
            evt = next((e for e in ws.events if e.id == via_evt_id), None)
            if evt is not None:
                via_event_label = (
                    (evt.content or evt.description or evt.id)[:60]
                )

        rows.append({
            "target_id": b.target_id,
            "target_name": target_name,
            "perceived_state": b.perceived_state,
            "confidence": round(float(b.confidence), 2),
            "inertia": round(float(b.inertia), 2),
            "established": int(getattr(b, "established_at_fabula", 0) or 0),
            "acquired_via_event_id": via_evt_id,
            "acquired_via_channel_id": via_chn_id,
            "acquired_via_channel_name": via_channel_name,
            "acquired_via_event_label": via_event_label,
        })
    # Sort by confidence desc so high-conviction beliefs are visually salient.
    rows.sort(key=lambda r: (-r["confidence"], r["target_name"]))
    return rows


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
            "fabula_time": ce.fabula_time,
            "trait_target": ce.trait_target or "",
            "trait_delta": (
                round(ce.trait_delta, 2) if ce.trait_delta is not None else None
            ),
            "rel_counterpart": ce.rel_counterpart_id or "",
            "world_id": ce.world_id,
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
            "established_at_fabula": se.established_at_fabula,
            "destroyed_at_fabula": (
                se.destroyed_at_fabula
                if se.destroyed_at_fabula is not None
                else "—"
            ),
            "world_id": se.world_id,
        }
        for se in ws.spatial_topology
    ]


def ws_to_social_rows(ws: WorldStateV1) -> list[dict]:
    """Social topology as table rows.

    Per-axis ``observed`` markers come from the new metric-keyed
    ``RelationshipEdge.metrics`` dict so callers can distinguish a
    deliberately-zero metric (the entities were measured to be
    indifferent) from an unobserved one (no data).
    """
    rows: list[dict] = []
    for rel in ws.social_topology:
        m_aff = rel.metrics.get("affinity")  # type: ignore[arg-type]
        m_fear = rel.metrics.get("fear")  # type: ignore[arg-type]
        m_pow = rel.metrics.get("power_dynamic")  # type: ignore[arg-type]
        rows.append({
            "source": rel.source_entity_id,
            "target": rel.target_entity_id,
            "affinity": round(m_aff.value, 2) if m_aff else "—",
            "fear": round(m_fear.value, 2) if m_fear else "—",
            "power": round(m_pow.value, 2) if m_pow else "—",
            "inertia": round(rel.inertia, 2),
            "evidence": rel.evidence_strength,
            "axes_observed": len(rel.metrics),
            "last_updated_fabula": rel.last_updated_fabula,
            "world_id": rel.world_id,
        })
    return rows


def ws_to_info_rows(ws: WorldStateV1) -> list[dict]:
    """Communication signals as table rows: standing channels and utterance events.

    Combined view kept for back-compat (single-table consumers). Channel
    rows fill the channel-shaped columns; utterance rows additionally
    fill ``content``, ``syuzhet_index``, ``truth_value``, and
    ``acquired_via_channel_id`` while leaving channel-only columns
    (``directionality``, ``min_intelligibility``) blank to avoid the
    earlier mismatch where ``truth_value`` was stuffed into the
    ``directionality`` column.

    Prefer :func:`ws_to_channel_rows` and :func:`ws_to_utterance_rows`
    for split tables.
    """
    rows: list[dict] = []
    for ch in ws.channels.values():
        rows.append({
            "kind": "channel",
            "id": ch.id,
            "participants": ", ".join(ch.participant_ids),
            "medium": ch.medium,
            "directionality": ch.directionality,
            "min_intelligibility": (
                round(min(ch.intelligibility.values()), 2)
                if ch.intelligibility else 1.0
            ),
            "content": "",
            "syuzhet_index": None,
            "truth_value": "",
            "via_channel_id": "",
        })
    for evt in ws.events:
        if evt.event_type != "utterance":
            continue
        rows.append({
            "kind": "utterance",
            "id": evt.id,
            "participants": (
                f"{evt.speaker_id or ''} \u2192 " + ", ".join(evt.addressee_ids)
            ),
            "medium": (
                ws.channels[evt.via_channel_id].medium
                if evt.via_channel_id and evt.via_channel_id in ws.channels
                else "unmediated"
            ),
            "directionality": "",
            "min_intelligibility": None,
            "content": (evt.content or "")[:120],
            "syuzhet_index": evt.syuzhet_index,
            "truth_value": evt.truth_value or "",
            "via_channel_id": evt.via_channel_id or "",
        })
    return rows


def ws_to_channel_rows(ws: WorldStateV1) -> list[dict]:
    """Standing communication channels as table rows."""
    return [
        {
            "id": ch.id,
            "name": ch.name,
            "medium": ch.medium,
            "directionality": ch.directionality,
            "participants": ", ".join(ch.participant_ids),
            "min_intelligibility": (
                round(min(ch.intelligibility.values()), 2)
                if ch.intelligibility else 1.0
            ),
            "established_at_fabula": ch.established_at_fabula,
            "world_id": ch.world_id,
        }
        for ch in ws.channels.values()
    ]


def ws_to_utterance_rows(ws: WorldStateV1) -> list[dict]:
    """Discrete utterance events as table rows."""
    rows: list[dict] = []
    for evt in ws.events:
        if evt.event_type != "utterance":
            continue
        rows.append({
            "id": evt.id,
            "fabula_time": evt.fabula_time,
            "syuzhet_index": evt.syuzhet_index,
            "speaker": evt.speaker_id or "",
            "addressees": ", ".join(evt.addressee_ids),
            "via_channel_id": evt.via_channel_id or "",
            "truth_value": evt.truth_value or "",
            "content": (evt.content or "")[:120],
            "world_id": evt.world_id,
        })
    rows.sort(key=lambda r: (r["fabula_time"], r["syuzhet_index"]))
    return rows


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
            snapshot = reconstruct_entity_with_causal(ws, eid, t)
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
    *,
    entity_ids: list[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """Build Gantt/swim-lane data: actor lanes x event time spans.

    ``entity_ids``: if provided, restrict the lanes to this subset of
    entity IDs (in declaration order). Useful when a multi-select
    upstream wants to focus on a few characters out of a large cast.

    Returns ``(actor_names, event_items)``.
    """
    keep: set[str] | None = set(entity_ids) if entity_ids else None
    actor_ids: list[str] = []
    actor_names: list[str] = []
    for eid, ent in ws.entities.items():
        if keep is not None and eid not in keep:
            continue
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


# ── Entity lifelines (status / location / event ribbons) ─────────

# Status colours mirror narrative weight: alive states are green-ish,
# distress states warm, terminal states near-black.
_STATUS_COLORS: dict[str, str] = {
    "healthy": "#6FBF3A",
    "injured": "#F5B43C",
    "ill": "#E0A030",
    "unconscious": "#3A7BD5",
    "dead": "#1e2a3a",
}


def ws_to_lifeline_data(
    ws: WorldStateV1,
    *,
    entity_ids: list[str] | None = None,
) -> dict:
    """Build per-entity lifeline segments for ``render_entity_lifelines``.

    Each entity gets a chronological list of ``(start, end, status,
    location_id)`` segments derived from its ``state_timeline`` and
    initial state. Status changes drive segment colour; location
    changes are emitted separately as point markers so the lifeline
    "kinks" visibly at every move.

    ``entity_ids``: if provided, restrict the lanes to this subset.
    Useful when an upstream multi-select wants to focus on a few
    characters out of a large cast.

    Returns ``{"entities": [(eid, name)],
              "segments": [{"row", "start", "end", "status",
                            "status_color", "location_name"}],
              "moves":    [{"row", "time", "location_name"}],
              "events":   [{"row", "time", "event_type", "description",
                            "event_id", "color"}],
              "tmin": int, "tmax": int}``.
    """
    if not ws.entities:
        return {
            "entities": [], "segments": [], "moves": [],
            "events": [], "tmin": 0, "tmax": 0,
        }

    keep: set[str] | None = set(entity_ids) if entity_ids else None
    sel_entities = {
        eid: ent for eid, ent in ws.entities.items()
        if keep is None or eid in keep
    }
    if not sel_entities:
        return {
            "entities": [], "segments": [], "moves": [],
            "events": [], "tmin": 0, "tmax": 0,
        }

    # Time bounds from snapshots + events; fall back to a unit range.
    times: set[int] = set()
    for ent in sel_entities.values():
        for snap in ent.state_timeline:
            times.add(int(snap.fabula_time))
    for evt in ws.events:
        times.add(int(evt.fabula_time))
    if not times:
        tmin, tmax = 0, 1
    else:
        tmin, tmax = min(times), max(times)
        if tmax == tmin:
            tmax = tmin + 1

    entities: list[tuple[str, str]] = [
        (eid, ent.name) for eid, ent in sel_entities.items()
    ]
    row_for = {eid: i for i, (eid, _) in enumerate(entities)}

    segments: list[dict] = []
    moves: list[dict] = []

    def _loc_name(lid: str | None) -> str:
        if not lid:
            return ""
        loc = ws.locations.get(lid)
        return loc.name if loc else lid

    for eid, ent in sel_entities.items():
        row = row_for[eid]
        snaps = sorted(ent.state_timeline, key=lambda s: s.fabula_time)
        cur_status = ent.status
        cur_loc = ent.location_id
        seg_start = tmin
        # Walk snapshots, emitting a segment whenever status changes.
        for snap in snaps:
            t = int(snap.fabula_time)
            new_status = snap.status if snap.status is not None else cur_status
            new_loc = snap.location_id if snap.location_id is not None else cur_loc
            if new_status != cur_status and t > seg_start:
                segments.append({
                    "row": row,
                    "start": seg_start,
                    "end": t,
                    "status": cur_status,
                    "status_color": _STATUS_COLORS.get(cur_status, "#94a3b8"),
                    "location_name": _loc_name(cur_loc),
                })
                seg_start = t
                cur_status = new_status
            else:
                cur_status = new_status
            if new_loc != cur_loc:
                moves.append({
                    "row": row,
                    "time": t,
                    "location_name": _loc_name(new_loc),
                })
                cur_loc = new_loc
        # Final segment to tmax.
        if seg_start <= tmax:
            segments.append({
                "row": row,
                "start": seg_start,
                "end": tmax,
                "status": cur_status,
                "status_color": _STATUS_COLORS.get(cur_status, "#94a3b8"),
                "location_name": _loc_name(cur_loc),
            })

    # Event markers per actor row.
    events: list[dict] = []
    for evt in sorted(ws.events, key=lambda e: e.fabula_time):
        for aid in (evt.actor_ids or []):
            row = row_for.get(aid)
            if row is None:
                continue
            events.append({
                "row": row,
                "time": int(evt.fabula_time),
                "event_type": evt.event_type,
                "description": (evt.description or evt.id)[:80],
                "event_id": evt.id,
                "color": EVENT_TYPE_COLORS.get(evt.event_type, "#94a3b8"),
            })

    return {
        "entities": entities,
        "segments": segments,
        "moves": moves,
        "events": events,
        "tmin": tmin,
        "tmax": tmax,
    }


# ── Multi-entity comparison data (radar + grouped bars + ranking) ──

def ws_to_comparison_data(
    ws: WorldStateV1,
    entity_ids: list[str] | None = None,
    *,
    max_traits: int = 8,
) -> dict:
    """Pick the most informative shared traits across selected entities
    and return a structure that drives radar + grouped bars + ranking.

    ``entity_ids=None`` (or empty) auto-picks the top entities by
    number of traits. Trait selection prefers traits where the chosen
    entities differ the most (max - min spread), so the comparison
    actually highlights distinguishing axes rather than ones where
    everyone scores the same.

    Returns::

        {
          "entity_names": [str, ...],
          "trait_names":  [str, ...],
          "matrix":       [[float per trait, ...], per entity, ...],
          "ranking":      [(trait_name, [(entity_name, value), ...
                            sorted desc])]
        }
    """
    all_ents = list(ws.entities.items())
    if not all_ents:
        return {"entity_names": [], "trait_names": [], "matrix": [], "ranking": []}

    if entity_ids:
        chosen = [(eid, ws.entities[eid]) for eid in entity_ids if eid in ws.entities]
    else:
        chosen = sorted(all_ents, key=lambda kv: -len(kv[1].traits))[:4]
    if not chosen:
        return {"entity_names": [], "trait_names": [], "matrix": [], "ranking": []}

    # Universe of traits any chosen entity has.
    trait_universe: set[str] = set()
    for _eid, ent in chosen:
        trait_universe |= set(ent.traits.keys())
    if not trait_universe:
        return {
            "entity_names": [ent.name for _eid, ent in chosen],
            "trait_names": [], "matrix": [], "ranking": [],
        }

    # Score each trait by (a) coverage across chosen entities and
    # (b) value spread, so we rank "differentiating" traits highest.
    def _score(tn: str) -> tuple[float, float]:
        vals = []
        for _eid, ent in chosen:
            if tn in ent.traits:
                vals.append(ent.traits[tn].value)
        if not vals:
            return (-1.0, 0.0)
        coverage = len(vals) / len(chosen)
        spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
        return (coverage, spread)

    trait_names = sorted(
        trait_universe,
        key=lambda tn: (_score(tn)[0] + _score(tn)[1] * 1.5),
        reverse=True,
    )[:max_traits]
    trait_names.sort()  # stable display order alphabetically

    entity_names = [ent.name for _eid, ent in chosen]
    matrix: list[list[float]] = []
    for _eid, ent in chosen:
        row = []
        for tn in trait_names:
            tv = ent.traits.get(tn)
            row.append(round(float(tv.value), 3) if tv else 0.0)
        matrix.append(row)

    # Per-trait ranking across the chosen entities.
    ranking: list[tuple[str, list[tuple[str, float]]]] = []
    for j, tn in enumerate(trait_names):
        col = [(entity_names[i], matrix[i][j]) for i in range(len(entity_names))]
        col.sort(key=lambda kv: -kv[1])
        ranking.append((tn, col))

    return {
        "entity_names": entity_names,
        "trait_names": trait_names,
        "matrix": matrix,
        "ranking": ranking,
    }


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

# Snapshot cache: keyed by (id(ws), revision, t). Bounded to keep memory
# predictable; ``invalidate_snapshot_cache()`` clears the cache *and*
# bumps the monotonic revision so any cache key built from a stale
# revision can never collide with a fresh one even if the same
# ``id(ws)`` happens to be reused (Python may recycle ids of GC'd
# WorldStateV1 instances). This is defence-in-depth: in-place
# mutations that forget to emit ``WORLD_STATE_CHANGED`` will still
# return stale data, but at least re-loading a different project into
# the same memory slot can't.
_SNAPSHOT_CACHE: "dict[tuple[int, int, int], WorldStateV1]" = {}
_SNAPSHOT_CACHE_MAX = 64
_SNAPSHOT_REVISION: int = 0

# Affect caches — keyed on (id(ws), revision, ...). Cursor scrubbing
# repeatedly calls compute_affective_scores and affective_timeseries
# with the same world model; without these caches the engine scorers
# (suspense / surprise / dramatic_irony / mystery — each O(events ·
# entities · traits)) ran on every slider release and froze the UI.
# Bounded to keep memory predictable.
_AFFECT_SCORE_CACHE: "dict[tuple, dict[str, float]]" = {}
_AFFECT_TIMESERIES_CACHE: "dict[tuple, tuple[list[int], dict[str, list[float]]]]" = {}
_AFFECT_CACHE_MAX = 256


def _snapshot_cache_get(ws: WorldStateV1, t: int) -> Optional[WorldStateV1]:
    return _SNAPSHOT_CACHE.get((id(ws), _SNAPSHOT_REVISION, t))


def _snapshot_cache_put(ws: WorldStateV1, t: int, snap: WorldStateV1) -> None:
    if len(_SNAPSHOT_CACHE) >= _SNAPSHOT_CACHE_MAX:
        # Drop an arbitrary entry — slider scrubbing is sequential so the
        # working set is small and FIFO eviction is fine.
        _SNAPSHOT_CACHE.pop(next(iter(_SNAPSHOT_CACHE)))
    _SNAPSHOT_CACHE[(id(ws), _SNAPSHOT_REVISION, t)] = snap


def invalidate_snapshot_cache() -> None:
    """Drop all cached fabula-time snapshots and bump the cache revision.

    Called from :meth:`AppState.emit` for ``WORLD_STATE_CHANGED`` so
    every panel's snapshot cache is reset whenever the world model is
    replaced or mutated. The revision bump means that any code path
    still holding a stale cache key (e.g. a stack frame mid-render
    when the world flipped underneath it) is guaranteed to miss the
    cache and recompute against the live data.
    """
    global _SNAPSHOT_REVISION
    _SNAPSHOT_CACHE.clear()
    _SNAPSHOT_REVISION += 1
    # The affect caches are keyed on the same revision, so bumping the
    # revision logically invalidates them. We also clear them to keep
    # memory predictable when projects are swapped frequently.
    _AFFECT_SCORE_CACHE.clear()
    _AFFECT_TIMESERIES_CACHE.clear()


def fabula_time_bounds(ws: WorldStateV1) -> tuple[int, int]:
    """Return ``(min, max)`` fabula_time across all events.

    Returns ``(0, 0)`` for empty event lists so callers can disable
    the slider safely.
    """
    if not ws.events:
        return (0, 0)
    times = [evt.fabula_time for evt in ws.events]
    return (min(times), max(times))


def _set_slider_bounds(slider, tmin: int, tmax: int) -> None:
    """Update a NiceGUI slider's ``min``/``max`` props with numeric values.

    Calling ``slider.props(f"min={tmin} max={tmax}")`` routes the values
    through Quasar's whitespace-prop parser, which stores them as
    *strings* (``"0"``, ``"5000"``). Quasar's ``<q-slider>`` requires
    numeric ``min``/``max`` \u2014 with strings the thumb renders but
    drag/keyboard input is silently clamped to the original construction-
    time range, so the slider appears frozen. Writing numeric values
    directly into the props dict triggers a single websocket update with
    the correct types.
    """
    tmin_i = int(tmin)
    tmax_i = int(tmax)
    props = slider.props
    changed = False
    if props.get("min") != tmin_i:
        with props.suspend_updates():
            props["min"] = tmin_i
        changed = True
    if props.get("max") != tmax_i:
        with props.suspend_updates():
            props["max"] = tmax_i
        changed = True
    if changed:
        # The slider may have been removed from the DOM between when a
        # debounced refresh fired and now (e.g. user switched tabs or
        # the panel was rebuilt). ``element.update()`` walks up to the
        # parent slot and raises ``RuntimeError`` if that slot was
        # cleared. Swallow that single, expected race.
        try:
            slider.update()
        except RuntimeError:
            return


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


def relationship_heatmap_frames(
    ws: WorldStateV1,
    *,
    metric: str = "affinity",
    num_frames: int = 12,
) -> dict:
    """Time-sliced entity×entity matrices for an animated heatmap.

    Builds one matrix per fabula tick (capped at ``num_frames``,
    distributed evenly between the world's earliest and latest
    ``fabula_time``) using
    :func:`reconstruct_relationship_with_causal` so each frame
    reflects authored snapshots **and** ``mutation_social`` causal
    edges accumulated through that tick. Two-cell symmetry mirrors
    :func:`ws_to_heatmap_data` so both heatmaps render the same way.

    Returns ``{"names": [...], "times": [...], "frames": [[[x,y,v]...], ...]}``
    where ``frames[i]`` is the matrix at ``times[i]``. Returns empty
    lists if the world has no entities or no events.
    """
    ent_ids = list(ws.entities.keys())
    if not ent_ids:
        return {"names": [], "times": [], "frames": []}
    ent_names = [ws.entities[eid].name for eid in ent_ids]
    idx = {eid: i for i, eid in enumerate(ent_ids)}

    tmin, tmax = fabula_time_bounds(ws)
    if tmax <= tmin:
        # No span — emit a single frame so the chart still renders.
        times = [tmax]
    else:
        n = max(2, min(num_frames, tmax - tmin + 1))
        step = (tmax - tmin) / (n - 1)
        times = [int(round(tmin + step * i)) for i in range(n)]
        # Deduplicate while preserving order (small spans collapse).
        seen: set[int] = set()
        times = [t for t in times if not (t in seen or seen.add(t))]

    # Walk every dyad once per frame using the causal-aware
    # reconstructor. We deliberately iterate ``social_topology`` (not
    # the cartesian product of entities) — characters with no edge
    # have no signal to display and would clutter the matrix.
    frames: list[list[list]] = []
    for t in times:
        frame_data: list[list] = []
        seen_pairs: set[tuple[str, str]] = set()
        for rel in ws.social_topology:
            si = idx.get(rel.source_entity_id)
            ti = idx.get(rel.target_entity_id)
            if si is None or ti is None:
                continue
            pair_key = tuple(sorted([rel.source_entity_id, rel.target_entity_id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            recon = reconstruct_relationship_with_causal(
                ws, rel.source_entity_id, rel.target_entity_id, t
            )
            if recon is None:
                continue
            val = recon.get(metric, 0.0)
            try:
                v = round(float(val), 2)
            except (TypeError, ValueError):
                v = 0.0
            frame_data.append([si, ti, v])
            frame_data.append([ti, si, v])
        frames.append(frame_data)

    return {"names": ent_names, "times": times, "frames": frames}


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

    Uses :func:`reconstruct_world_trait_with_causal` so the curve
    reflects both authored :class:`WorldTraitSnapshot` entries *and*
    any ``mutation`` :class:`CausalEdge` whose ``target_id`` is this
    world trait — they accumulate signed ``trait_delta`` onto the
    baseline magnitude.

    Returns ``{"times": [...], "value": [...], "inertia": [...]}``.
    """
    wt = ws.world_traits.get(world_id)
    if wt is None:
        return {"times": [], "value": [], "inertia": []}

    tmin, tmax = fabula_time_bounds(ws)
    # Sample at every snapshot fabula_time, every mutation-edge tick
    # targeting this trait, plus the bounds.
    sample_times: set[int] = {tmin, tmax}
    for snap in wt.state_timeline:
        sample_times.add(snap.fabula_time)
    for ce in ws.causal_topology:
        if (
            ce.causality_type == "mutation"
            and ce.target_id == world_id
            and ce.trait_delta is not None
        ):
            sample_times.add(ce.fabula_time)
    times = sorted(t for t in sample_times if tmin <= t <= tmax) or [0]

    values: list[float] = []
    inertias: list[float] = []
    for t in times:
        snap = reconstruct_world_trait_with_causal(ws, world_id, t)
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
    status, location_id) replayed via
    :func:`reconstruct_entity_with_causal` — i.e. baseline traits +
    every ``mutation`` :class:`CausalEdge` up to ``t`` + authored
    :class:`EntityStateSnapshot` overrides. Each :class:`GlobalTrait`
    is replayed via :func:`reconstruct_world_trait_with_causal` so
    causal edges targeting ``WORLD_*`` ids also move world magnitudes.
    The events list is filtered to ``fabula_time <= t``.

    Topology edges are also time-sliced to mirror the canonical
    :func:`shadow_loom.extract_graph.extract_ego_graph_from_memory`
    rules so ego-graph / evolution views show the network *as it was*
    at ``t`` rather than the final-frame topology:

      * ``causal_topology``: ``fabula_time <= t``
      * ``social_topology``: ``last_updated_fabula <= t`` (relationship
        metric values reflect their last update; later-updated edges
        are dropped, matching the ego-graph extractor)
      * ``spatial_topology``: established by ``t`` and not yet
        destroyed at ``t``
      * ``channels``: established by ``t`` and not yet
        terminated at ``t``

    ``NarrativeObject`` instances are left untouched: the model has no
    per-object state timeline, so object ``location_id`` / ``owner_id``
    always reflect the latest snapshot.

    The returned model is suitable to re-feed into existing renderers
    without further changes.
    """
    cached = _snapshot_cache_get(ws, t)
    if cached is not None:
        return cached

    new = ws.model_copy(deep=True)

    # Local imports avoid a top-level cycle with shadow_loom.models.
    from shadow_loom.models import Belief, TraitVector

    for eid, ent in new.entities.items():
        # NB: pass the *original* ws — its causal_topology is the
        # ground truth and identical to ``new.causal_topology`` since
        # we just deep-copied; reading from ws keeps the helper API
        # (which expects a ``WorldStateV1``) consistent with callers.
        snap = reconstruct_entity_with_causal(ws, eid, t)
        ent.traits = {
            k: TraitVector(
                value=v["value"],
                inertia=v["inertia"],
                evidence_strength=v.get("evidence_strength", "moderate"),
            )
            for k, v in snap["traits"].items()
        }
        ent.status = snap["status"]
        ent.location_id = snap["location_id"]
        ent.beliefs = [Belief(**b) for b in snap["beliefs"]]

    for wid, wt in new.world_traits.items():
        snap = reconstruct_world_trait_with_causal(ws, wid, t)
        mag = snap["magnitude"]
        wt.magnitude = TraitVector(
            value=mag["value"],
            inertia=mag["inertia"],
            evidence_strength=mag.get("evidence_strength", "moderate"),
        )
        if snap.get("description") is not None:
            wt.description = snap["description"]

    new.events = [evt for evt in new.events if evt.fabula_time <= t]

    # Topology time-slice (canonical rules from extract_ego_graph_from_memory).
    new.causal_topology = [
        ce for ce in new.causal_topology if ce.fabula_time <= t
    ]
    # Per-axis time-slice for relationships: keep an edge if any axis was
    # observed at or before ``t``; drop axes whose own
    # ``last_updated_fabula > t``. Recreating the cross-axis discard bug
    # (whole-edge in/out off ``max(per-axis last_updated_fabula)``) would
    # lose unrelated axes the user explicitly wanted to inspect.
    sliced_social: list = []
    for rel in new.social_topology:
        survivors = {
            name: m for name, m in rel.metrics.items()
            if m.last_updated_fabula <= t
        }
        if not survivors:
            continue
        rel.metrics = survivors  # type: ignore[assignment]
        sliced_social.append(rel)
    new.social_topology = sliced_social
    # Replay mutation_social causal edges so affinity / fear / power
    # values reflect their state at ``t`` rather than the final-frame
    # numbers stored on the edge.
    for rel in new.social_topology:
        snap = reconstruct_relationship_with_causal(
            ws, rel.source_entity_id, rel.target_entity_id, t,
        )
        if snap is None:
            continue
        # The new RelationshipEdge stores per-axis state under
        # ``metrics``; mutate in place so existing edge identity is
        # preserved. Falls back to the current value for any axis
        # the reconstructor didn't touch.
        #
        # IMPORTANT: ``reconstruct_relationship_with_causal`` always
        # emits all three axes in ``snap`` (filling unobserved ones
        # with value=0.0). Naively writing those into ``rel.metrics``
        # — or worse, creating a fresh ``RelationshipMetric`` (which
        # defaults ``observed=True``) — silently flips never-observed
        # axes into "observed" with value 0.0. The affective scorers
        # (``conflict``, ``danger``, ``narrative_tension``) gate on
        # ``observed=True``, so the spurious zeros pin those gauges
        # flat across the entire fabula axis. Honour the per-axis
        # ``observed`` flag carried in ``snap["per_axis"]`` instead.
        per_axis = snap.get("per_axis") or {}
        for axis in ("affinity", "fear", "power_dynamic"):
            info = per_axis.get(axis)
            if info is None:
                continue
            existed = axis in rel.metrics
            if not info.get("observed") and not existed:
                # Never observed and no replay touched it → don't
                # fabricate a 0.0 metric just to mirror the API shape.
                continue
            value = float(snap.get(axis, info.get("value", 0.0)))
            if existed:
                rel.metrics[axis].value = value
            else:
                # Axis got a real value via mutation_social replay even
                # though the source edge had no baseline. Preserve the
                # observed=False flag so it stays out of the affective
                # aggregates until the LLM actually authors a reading.
                from shadow_loom.models import RelationshipMetric
                rel.metrics[axis] = RelationshipMetric(
                    value=value, observed=bool(info.get("observed", False)),
                )
    new.spatial_topology = [
        se for se in new.spatial_topology
        if se.established_at_fabula <= t
        and (se.destroyed_at_fabula is None or se.destroyed_at_fabula > t)
    ]
    new.channels = {
        cid: ch for cid, ch in new.channels.items()
        if ch.established_at_fabula <= t
        and (ch.terminated_at_fabula is None or ch.terminated_at_fabula > t)
    }

    _snapshot_cache_put(ws, t, new)
    return new


# ── Affective scores (single snapshot + over-time series) ─────────

def _top_entity_ids_by_event_degree(
    ws: WorldStateV1, limit: int = 20,
) -> list[str]:
    """Return up to ``limit`` entity IDs ranked by event participation.

    Used to bound the cost of engine affective scorers (notably
    ``compute_surprise_score`` which is O(traits × incoming edges)) on
    large worlds. Falls back to insertion order when an entity has no
    event participation so we still surface *something*.
    """
    if not ws.entities:
        return []
    degree: dict[str, int] = {eid: 0 for eid in ws.entities}
    for evt in ws.events:
        for eid in (*evt.actor_ids, *evt.target_ids):
            if eid in degree:
                degree[eid] += 1
    ranked = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))
    return [eid for eid, _ in ranked[:max(1, limit)]]


def _engine_structural_scores(
    ws: WorldStateV1,
    entity_ids: list[str],
    syuzhet_anchor: int | None,
    *,
    surprise_local: bool = False,
) -> dict[str, float]:
    """Run :class:`DirectiveAssembly`'s four structural affect scorers.

    Returns a dict keyed by ``mystery``, ``dramatic_irony``,
    ``suspense``, ``surprise``. Failures are swallowed (logged at
    debug) so a single broken metric never wipes the whole gauge row.

    ``surprise_local=True`` switches the surprise scorer to
    Itti & Baldi Bayesian Surprise (per-step belief-update
    magnitude) — used by the timeseries view so the chart shows
    spike-and-decay around revelations rather than the
    monotonically declining cumulative form.
    """
    out: dict[str, float] = {}
    if not entity_ids or not ws.events:
        return out
    try:
        # Local import — avoids importing the full directive-assembly
        # graph stack at module load (it depends on networkx + the
        # whole shadow_loom package which is heavy for unit tests).
        from shadow_loom.directive_assembly import DirectiveAssembler
    except Exception:  # pragma: no cover - defensive
        logger.debug("DirectiveAssembler unavailable", exc_info=True)
        return out
    try:
        assembler = DirectiveAssembler(
            sandbox=None, ego_payload={}, world_state=ws,
        )
    except Exception:
        logger.debug("DirectiveAssembler init failed", exc_info=True)
        return out
    metric_calls = (
        ("mystery", lambda eids, sa: assembler.compute_mystery_score(eids, sa)),
        ("dramatic_irony", lambda eids, sa: assembler.compute_dramatic_irony_score(eids, sa)),
        ("suspense", lambda eids, sa: assembler.compute_suspense_score(eids, sa)),
        ("surprise", lambda eids, sa: assembler.compute_surprise_score(
            eids, sa, local=surprise_local,
        )),
    )
    for name, fn in metric_calls:
        try:
            out[name] = float(fn(entity_ids, syuzhet_anchor))
        except Exception:
            logger.debug("affective metric %s failed", name, exc_info=True)
    return out


def compute_affective_scores(
    ws: WorldStateV1,
    *,
    entity_ids: list[str] | None = None,
    syuzhet_anchor: int | None = None,
    ws_for_engine: WorldStateV1 | None = None,
    surprise_local: bool = False,
) -> dict[str, float]:
    """Cached wrapper around :func:`_compute_affective_scores_uncached`.

    Slider scrubbing repeatedly asks for scores on the same cached
    snapshot; recomputing the engine scorers every time froze the UI.
    Cached on ``(id(ws), revision, entity_ids, syuzhet_anchor)`` so a
    return visit to a previously-seen snapshot is O(1).

    ``ws_for_engine`` is an optional override for the structural-affect
    scorers (mystery, dramatic_irony, suspense, surprise). The fabula
    timeseries builder passes a fabula-trimmed snapshot as ``ws`` (so
    the heuristic ``conflict``/``danger``/``narrative_tension`` layers
    see the right time-sliced relationship state) but needs the engine
    layer to operate on the *full* world so "unrevealed", "hidden
    causal ancestors", and "future trait state" sets stay non-empty.
    """
    eids_key = tuple(entity_ids) if entity_ids else ()
    engine_id = id(ws_for_engine) if ws_for_engine is not None else id(ws)
    cache_key = (id(ws), engine_id, _SNAPSHOT_REVISION, eids_key,
                 syuzhet_anchor, surprise_local)
    cached = _AFFECT_SCORE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    result = _compute_affective_scores_uncached(
        ws,
        entity_ids=entity_ids,
        syuzhet_anchor=syuzhet_anchor,
        ws_for_engine=ws_for_engine,
        surprise_local=surprise_local,
    )
    if len(_AFFECT_SCORE_CACHE) >= _AFFECT_CACHE_MAX:
        _AFFECT_SCORE_CACHE.pop(next(iter(_AFFECT_SCORE_CACHE)))
    _AFFECT_SCORE_CACHE[cache_key] = dict(result)
    return result


def _compute_affective_scores_uncached(
    ws: WorldStateV1,
    *,
    entity_ids: list[str] | None = None,
    syuzhet_anchor: int | None = None,
    ws_for_engine: WorldStateV1 | None = None,
    surprise_local: bool = False,
) -> dict[str, float]:
    """Compute basic affective/narrative scores from a world snapshot.

    All scores are normalised to ``[0, 1]``. Missing source data
    (e.g. no information topology) simply omits that key.

    When ``entity_ids`` is supplied, the four engine-grade structural
    affects from :class:`shadow_loom.directive_assembly.DirectiveAssembly`
    are also computed and *override* the heuristic ``mystery`` key with
    the canonical graph-walk version. ``syuzhet_anchor`` defaults to
    "reader has seen everything" (i.e. uses the snapshot's max syuzhet
    index) so suspense / surprise / dramatic-irony are non-zero even
    when the caller doesn't track a separate reader cursor.

    Definitions:
      * **mystery** — (heuristic) fraction of utterance events whose
        ``syuzhet_index >= 1`` (i.e. revealed mid-narrative rather
        than at the very start). Replaced with the engine's ratio of
        hidden causal ancestors when entity_ids are given.
      * **dramatic_irony** — (engine, requires entity_ids) information
        asymmetry where the reader knows more than the character.
      * **suspense** — (engine, requires entity_ids) probabilistic
        valence between unrevealed threat vs hope outcomes.
      * **surprise** — (engine, requires entity_ids) KL divergence
        between the reader's prior and the actual trait state.
      * **narrative_tension** — composite of (a) mean negative
        affinity magnitude, (b) mean fear, (c) fraction of recent
        events with destructive types.
      * **conflict** — fraction of relationships with affinity < 0.
      * **danger** — mean fear across active relationships.
      * **causal_density** — observed edges per event, mapped through
        a soft saturation ``d / (d + K)`` with ``K = 3``. Replaces an
        earlier ``min(1, d / 3)`` clamp that pinned every dense world
        (ACOTAR runs at ~5 edges/event) flat at 1.0 across the entire
        timeline. The saturation form keeps the metric in ``[0, 1]``
        while preserving variation above the K-threshold.
    """
    scores: dict[str, float] = {}
    if not ws.events:
        return scores

    utterances = [e for e in ws.events if e.event_type == "utterance"]
    if utterances:
        late = sum(
            1 for e in utterances
            if e.syuzhet_index and e.syuzhet_index >= 1
        )
        scores["mystery"] = min(
            1.0, late / max(1, len(utterances))
        )

    rels = ws.social_topology
    if rels:
        # Per-axis observed-aware aggregates: an axis the LLM never
        # measured contributes nothing (issue #8). Aggregating over
        # unobserved-zero axes silently deflates conflict / danger /
        # tension scores.
        observed_aff = [
            r.metrics["affinity"].value for r in rels
            if "affinity" in r.metrics and r.metrics["affinity"].observed
        ]
        observed_fear = [
            r.metrics["fear"].value for r in rels
            if "fear" in r.metrics and r.metrics["fear"].observed
        ]
        if observed_aff:
            negative = sum(1 for v in observed_aff if v < 0)
            scores["conflict"] = min(1.0, negative / len(observed_aff))
        if observed_fear:
            avg_fear = sum(observed_fear) / len(observed_fear)
            scores["danger"] = min(1.0, max(0.0, avg_fear))

    # Composite narrative tension.
    if rels or ws.events:
        observed_aff = [
            r.metrics["affinity"].value for r in rels
            if "affinity" in r.metrics and r.metrics["affinity"].observed
        ]
        observed_fear = [
            r.metrics["fear"].value for r in rels
            if "fear" in r.metrics and r.metrics["fear"].observed
        ]
        avg_neg_affinity = (
            sum(max(0.0, -v) for v in observed_aff) / len(observed_aff)
            if observed_aff else 0.0
        )
        avg_fear_t = (
            sum(observed_fear) / len(observed_fear) if observed_fear else 0.0
        )
        # High-force causal edges are a proxy for high-stakes mutations.
        # ``causal_force`` is on a 0-10 scale; treat >= 7 as "high stakes".
        if ws.causal_topology:
            high_force = sum(
                1 for ce in ws.causal_topology if ce.causal_force >= 7
            )
            high_force_share = high_force / len(ws.causal_topology)
        else:
            high_force_share = 0.0
        # Weighted sum with weights summing to 1.0 — keeps result in [0, 1].
        tension = (
            0.40 * avg_neg_affinity
            + 0.35 * avg_fear_t
            + 0.25 * high_force_share
        )
        scores["narrative_tension"] = min(1.0, max(0.0, tension))

    # Causal density: edges per event, soft-saturating via
    # ``d / (d + K)`` rather than a hard ``min(1, d/3)`` clamp. The
    # clamp pinned every dense world (ACOTAR sits at 4-6 edges/event
    # across all snapshots) flat at 1.0, hiding genuine fabula-time
    # variation in the chart. ``K = 1.5`` widens the dynamic range
    # (sparse passages drop to ~0.25, mid worlds sit at 0.55-0.75,
    # dense ones reach ~0.85+) so the chart shows graded contrast
    # rather than the previous compressed 0.4-0.9 band that K=3 gave.
    if ws.events:
        density = len(ws.causal_topology) / max(1, len(ws.events))
        scores["causal_density"] = density / (density + 1.5)

    # Engine-grade structural affects — only computed when the caller
    # supplies a focus set (typically the world's top-N entities). The
    # ``mystery`` key from DirectiveAssembly takes precedence over the
    # heuristic above because it is the one the directive-assembly
    # optimiser actually targets.
    #
    # Theory note: all four structural affects (mystery / dramatic_irony
    # / suspense / surprise) are defined as set operations between the
    # *full* event graph and a syuzhet anchor (Sternberg 1978; Brewer &
    # Lichtenstein 1982; Cheong & Young 2015; Itti & Baldi 2009). When
    # the caller hands us a fabula-trimmed snapshot (``ws``), the
    # "unrevealed", "hidden ancestors", and "future trait" sets the
    # scorers depend on are all empty by construction — collapsing
    # suspense/surprise to 0 and pinning mystery near 1.0 across the
    # whole timeline. ``ws_for_engine`` lets the timeseries builder
    # hand us the full ws for the engine layer while the heuristics
    # above still see the time-sliced relationship state.
    if entity_ids:
        engine_ws = ws_for_engine if ws_for_engine is not None else ws
        if syuzhet_anchor is None:
            # Default: anchor *before* the first reveal so the entire
            # event list counts as the unrevealed tail. Anchoring at
            # ``max(syuzhet_index)`` (a previous version of this branch)
            # marked every event as already-revealed, which collapsed
            # suspense's ``unrevealed = all - revealed`` set to ∅ and
            # pulled surprise's prior all the way onto the posterior —
            # zeroing both scores on every unanchored snapshot. Using
            # ``min - 1`` keeps the reader at the narrative threshold
            # so the structural affects retain their full contrast.
            min_s = min(
                (e.syuzhet_index for e in engine_ws.events), default=None
            )
            syuzhet_anchor = (min_s - 1) if min_s is not None else None
        engine = _engine_structural_scores(
            engine_ws, entity_ids, syuzhet_anchor,
            surprise_local=surprise_local,
        )
        scores.update(engine)
    return scores


def affective_timeseries(
    ws: WorldStateV1,
    *,
    samples: int = 12,
    entity_ids: list[str] | None = None,
) -> tuple[list[int], dict[str, list[float]]]:
    """Sample :func:`compute_affective_scores` at ``samples`` evenly-spaced
    fabula_time cursors across the story.

    Returns ``(times, series)`` where ``series`` maps each affective
    metric name to a list of values aligned to ``times``. Snapshots
    are taken via :func:`snapshot_world_at` so the cache is shared
    with the slider-driven views. When ``entity_ids`` is supplied the
    engine-grade structural affects (suspense, surprise, dramatic_irony,
    canonical mystery) are sampled too.

    The result is cached on ``(id(ws), revision, samples, entity_ids)``
    so that scrubbing the timeline cursor — which only moves the chart's
    needle, not the underlying data — doesn't trigger a full 12-sample
    re-snapshot + engine rescore on every release.
    """
    eids_key = tuple(entity_ids) if entity_ids else ()
    cache_key = ("fabula", id(ws), _SNAPSHOT_REVISION, int(samples), eids_key)
    cached = _AFFECT_TIMESERIES_CACHE.get(cache_key)
    if cached is not None:
        times_c, series_c = cached
        return list(times_c), {k: list(v) for k, v in series_c.items()}

    tmin, tmax = fabula_time_bounds(ws)
    if tmax <= tmin:
        scores = compute_affective_scores(ws, entity_ids=entity_ids)
        return [tmin], {k: [v] for k, v in scores.items()}

    samples = max(2, int(samples))
    step = max(1, (tmax - tmin) // (samples - 1))
    times = list(range(tmin, tmax + 1, step))
    if times[-1] != tmax:
        times.append(tmax)

    series: dict[str, list[float]] = {}
    for i, t in enumerate(times):
        snap = snapshot_world_at(ws, t)
        # Reader's syuzhet position at fabula time t = the latest syuzhet
        # index among events that have already happened in story-time.
        # The engine layer (suspense / mystery / dramatic_irony /
        # surprise) is run against the full ``ws`` rather than ``snap``
        # so its set-theoretic operands (``unrevealed``, ``hidden
        # ancestors``, ``future trait state``) are non-empty — see the
        # block in ``_compute_affective_scores_uncached`` that consumes
        # ``ws_for_engine`` for the theory rationale.
        anchor = max(
            (e.syuzhet_index for e in snap.events), default=None,
        )
        scores = compute_affective_scores(
            snap,
            entity_ids=entity_ids,
            syuzhet_anchor=anchor,
            ws_for_engine=ws,
            surprise_local=True,
        )
        # Back-pad any newly-discovered metric so its column lines up
        # with previous time samples (missing = 0.0).
        for k in scores:
            if k not in series:
                series[k] = [0.0] * i
        # Append this sample's value (or 0.0) to every active series so
        # all lists stay length i+1.
        for k in series:
            series[k].append(round(float(scores.get(k, 0.0)), 3))
    if len(_AFFECT_TIMESERIES_CACHE) >= _AFFECT_CACHE_MAX:
        _AFFECT_TIMESERIES_CACHE.pop(next(iter(_AFFECT_TIMESERIES_CACHE)))
    _AFFECT_TIMESERIES_CACHE[cache_key] = (
        list(times), {k: list(v) for k, v in series.items()},
    )
    return times, series


def affective_timeseries_syuzhet(
    ws: WorldStateV1,
    *,
    samples: int = 12,
    entity_ids: list[str] | None = None,
) -> tuple[list[int], dict[str, list[float]]]:
    """Sample affective scores along the **syuzhet** (reader) axis.

    Mirrors :func:`affective_timeseries` but anchors against
    ``syuzhet_index`` so the resulting curves show how mystery,
    suspense, dramatic irony and surprise rise and fall *as the reader
    progresses through the text*.

    Each sample drives two views of the world simultaneously:

    * **Heuristic layer** (``conflict``, ``danger``,
      ``narrative_tension``, ``causal_density``) sees a
      syuzhet-revealed snapshot — events with ``syuzhet_index <= s``
      and the social/causal topology trimmed to the corresponding
      fabula horizon. Without this trim every heuristic was flat
      across the entire reader axis (the scorers don't consult
      ``syuzhet_anchor`` themselves; they read final-frame state).
    * **Engine layer** (``mystery``, ``dramatic_irony``, ``suspense``,
      ``surprise``) sees the full ``ws`` via ``ws_for_engine`` and
      partitions revealed/unrevealed itself through ``syuzhet_anchor``.
      Pre-trimming would hide the unrevealed tail suspense depends on.

    Cached on the same key shape as :func:`affective_timeseries`.
    """
    eids_key = tuple(entity_ids) if entity_ids else ()
    cache_key = ("syuzhet", id(ws), _SNAPSHOT_REVISION, int(samples), eids_key)
    cached = _AFFECT_TIMESERIES_CACHE.get(cache_key)
    if cached is not None:
        times_c, series_c = cached
        return list(times_c), {k: list(v) for k, v in series_c.items()}

    smin, smax = syuzhet_time_bounds(ws)
    if smax <= smin:
        scores = compute_affective_scores(
            ws, entity_ids=entity_ids, syuzhet_anchor=smin,
        )
        return [smin], {k: [v] for k, v in scores.items()}

    samples = max(2, int(samples))
    step = max(1, (smax - smin) // (samples - 1))
    indices = list(range(smin, smax + 1, step))
    if indices[-1] != smax:
        indices.append(smax)

    series: dict[str, list[float]] = {}
    for i, s in enumerate(indices):
        # Map the syuzhet anchor onto its corresponding fabula horizon:
        # the latest fabula_time among events the reader has now seen.
        # Used to time-slice social/causal topology so the heuristic
        # layer reflects only revealed-by-now state.
        revealed_events = [
            e for e in ws.events if e.syuzhet_index <= s
        ]
        if revealed_events:
            fabula_horizon = max(
                e.fabula_time for e in revealed_events
                if e.fabula_time is not None
            ) if any(e.fabula_time is not None for e in revealed_events) else None
            if fabula_horizon is not None:
                heuristic_ws = snapshot_world_at(ws, fabula_horizon)
            else:
                heuristic_ws = snapshot_world_at_syuzhet(ws, s)
        else:
            heuristic_ws = snapshot_world_at_syuzhet(ws, s)
        scores = compute_affective_scores(
            heuristic_ws,
            entity_ids=entity_ids,
            syuzhet_anchor=s,
            ws_for_engine=ws,
            surprise_local=True,
        )
        for k in scores:
            if k not in series:
                series[k] = [0.0] * i
        for k in series:
            series[k].append(round(float(scores.get(k, 0.0)), 3))
    if len(_AFFECT_TIMESERIES_CACHE) >= _AFFECT_CACHE_MAX:
        _AFFECT_TIMESERIES_CACHE.pop(next(iter(_AFFECT_TIMESERIES_CACHE)))
    _AFFECT_TIMESERIES_CACHE[cache_key] = (
        list(indices), {k: list(v) for k, v in series.items()},
    )
    return indices, series


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
            "snapshots": len(ent.state_timeline),
            "constants": ", ".join(ent.constants) if ent.constants else "",
            "world_id": ent.world_id,
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
            "world_id": evt.world_id,
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
            "world_id": obj.world_id,
        })
    return rows


def ws_to_world_trait_rows(ws: WorldStateV1) -> list[dict]:
    """One row per global/world-level trait."""
    rows: list[dict] = []
    for wid, wt in ws.world_traits.items():
        rows.append({
            "id": wid,
            "name": wt.name,
            "category": getattr(wt, "category", "") or "",
            "magnitude": round(wt.magnitude.value, 3),
            "inertia": round(wt.magnitude.inertia, 3),
            "affected_domains": ", ".join(wt.affected_domains) or "—",
            "snapshots": len(wt.state_timeline),
            "description": getattr(wt, "description", "") or "",
            "world_id": wt.world_id,
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
    """Tabular companion for ``mutations_to_propagation_graph``.

    Row schema (matches the table wired in causality_tab.py):
        step, entity, trait, delta, from_value, to_value, reason, blocked
    """
    out: list[dict] = []
    for i, m in enumerate(mutations or []):
        delta = 0.0
        try:
            delta = float(m.get("delta") or m.get("trait_delta") or 0.0)
        except (TypeError, ValueError):
            pass
        from_v = m.get("from_value")
        to_v = m.get("to_value")
        if from_v is None and to_v is None and "old" in m:
            from_v, to_v = m.get("old"), m.get("new")
        out.append({
            "step": i + 1,
            "entity": (
                m.get("entity") or m.get("target") or m.get("name") or ""
            ),
            "trait": m.get("trait") or m.get("trait_target") or "",
            "delta": round(delta, 3),
            "from_value": (
                round(float(from_v), 3)
                if isinstance(from_v, (int, float))
                else (from_v if from_v is not None else "")
            ),
            "to_value": (
                round(float(to_v), 3)
                if isinstance(to_v, (int, float))
                else (to_v if to_v is not None else "")
            ),
            "reason": m.get("reason") or m.get("mechanism", ""),
            "blocked": False,
        })
    base = len(out)
    for j, b in enumerate(blocked or []):
        out.append({
            "step": base + j + 1,
            "entity": (
                b.get("entity") or b.get("target") or b.get("name") or ""
            ),
            "trait": b.get("trait") or b.get("trait_target") or "",
            "delta": 0,
            "from_value": "",
            "to_value": "",
            "reason": b.get("reason") or b.get("mechanism", ""),
            "blocked": True,
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
    """Tabular companion for ``ws_to_calendar_graph_data``.

    Row schema (matches the table wired in causality_tab.py):
        event, bucket, fabula_time, type, actor, stack
    """
    if not ws.events:
        return []
    f_vals = [e.fabula_time for e in ws.events]
    f_min, f_max = min(f_vals), max(f_vals)
    span = max(1, f_max - f_min)
    width = max(1, span / bucket_count)

    def _bucket(t: int) -> int:
        return min(bucket_count - 1, int((t - f_min) / width))

    bucket_stack: dict[int, int] = {}
    rows: list[dict] = []
    for evt in sorted(ws.events, key=lambda e: (e.fabula_time, e.syuzhet_index)):
        b = _bucket(evt.fabula_time)
        slot = bucket_stack.get(b, 0)
        bucket_stack[b] = slot + 1
        actor = ""
        if evt.actor_ids:
            first = evt.actor_ids[0]
            actor = ws.entities[first].name if first in ws.entities else first
        rows.append({
            "event": evt.id,
            "bucket": b,
            "fabula_time": evt.fabula_time,
            "type": evt.event_type,
            "actor": actor,
            "stack": slot,
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
            snap = reconstruct_entity_with_causal(ws, entity_id, t)
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


# ── Physics trajectory (scalar metrics over fabula time) ──────────

# Metric keys plotted in the Physics Trajectory panel. Order is the
# rendering legend order; values are ECharts colours.
PHYSICS_METRIC_COLORS: dict[str, str] = {
    "present_entities": "#3A7BD5",
    "active_relationships": "#8E44AD",
    "avg_affinity": "#16A085",
    "avg_fear": "#C0392B",
    "causal_edges_in_scope": "#E67E22",
    "channel_edges_in_scope": "#2C7BB6",
    "spatial_edges_in_scope": "#7F8C8D",
}


def _physics_metrics_from_payload(payload: dict) -> dict[str, float]:
    """Reduce a single physics payload to scalar metrics.

    Handles both shapes returned by ``calculate_narrative_physics`` for
    observation queries:

      * Ego-graph payload (focus entities supplied) — uses
        ``present_entities`` / ``relevant_*`` lists.
      * Omniscient world dump (no focus entities) — uses
        ``entities`` and the full topology lists.
    """
    if "present_entities" in payload or "relevant_relationships" in payload:
        rels = payload.get("relevant_relationships", []) or []
        present = payload.get("present_entities", []) or []
        causal = payload.get("relevant_causal_edges", []) or []
        info = payload.get("relevant_channels", []) or []
        info_extra = payload.get("relevant_utterance_events", []) or []
        info = list(info) + list(info_extra)
        spatial = payload.get("relevant_spatial_edges", []) or []
    else:
        rels = list((payload.get("social_topology") or []))
        present = list((payload.get("entities") or {}).values()) \
            if isinstance(payload.get("entities"), dict) \
            else list(payload.get("entities") or [])
        causal = payload.get("causal_topology", []) or []
        channels_dump = payload.get("channels") or {}
        if isinstance(channels_dump, dict):
            info = list(channels_dump.values())
        else:
            info = list(channels_dump)
        info += [
            e for e in (payload.get("events") or [])
            if isinstance(e, dict) and e.get("event_type") == "utterance"
        ]
        spatial = payload.get("spatial_topology", []) or []

    affinities: list[float] = []
    fears: list[float] = []
    for r in rels:
        # Ego-payload edges go through ``RelationshipEdge.to_legacy_dict``
        # so the flat ``affinity``/``fear`` keys are present and authoritative.
        # Omniscient payload edges, however, come from ``model_dump()`` of the
        # validated WorldModel which only emits the per-axis ``metrics`` dict,
        # so we must fall back to ``metrics[axis].value`` (and skip the axis
        # entirely when it was never observed, instead of polluting the average
        # with a hallucinated 0.0).
        metrics = r.get("metrics") if isinstance(r.get("metrics"), dict) else None
        if "affinity" in r and isinstance(r.get("affinity"), (int, float)):
            affinities.append(float(r["affinity"]))
        elif metrics and isinstance(metrics.get("affinity"), dict):
            ax = metrics["affinity"]
            if ax.get("observed", True):
                affinities.append(float(ax.get("value", 0.0)))
        if "fear" in r and isinstance(r.get("fear"), (int, float)):
            fears.append(float(r["fear"]))
        elif metrics and isinstance(metrics.get("fear"), dict):
            ax = metrics["fear"]
            if ax.get("observed", True):
                fears.append(float(ax.get("value", 0.0)))
    return {
        "present_entities": float(len(present)),
        "active_relationships": float(len(rels)),
        "avg_affinity": (
            float(sum(affinities)) / len(affinities) if affinities else 0.0
        ),
        "avg_fear": (
            float(sum(fears)) / len(fears) if fears else 0.0
        ),
        "causal_edges_in_scope": float(len(causal)),
        "channel_edges_in_scope": float(len(info)),
        "spatial_edges_in_scope": float(len(spatial)),
    }


def physics_trajectory(
    ws: WorldStateV1,
    focus_entity_ids: list[str] | None = None,
    *,
    samples: int = 12,
) -> tuple[list[int], dict[str, list[float]]]:
    """Sample structural physics scalars at ``samples`` fabula anchors.

    For each anchor the helper invokes
    :func:`shadow_loom.narrative_physics.calculate_narrative_physics`
    with an :class:`ObservationQuery` (no LLM, pure structural) and
    extracts a fixed set of scalars from the resulting ego-graph
    payload. Use this to plot a "physics over time" view in the UI
    without re-running the full NL pipeline at each anchor.

    Results are cached per ``(id(ws), focus, samples)`` and the cache
    is cleared by :func:`invalidate_physics_trajectory_cache` (called
    automatically by ``AppState.emit(WORLD_STATE_CHANGED)``).

    Returns ``(times, series)`` where ``series`` keys are the metric
    names listed in :data:`PHYSICS_METRIC_COLORS`.
    """
    samples = max(2, int(samples))
    cache_key = (
        id(ws), tuple(sorted(focus_entity_ids or [])), samples,
    )
    cached = _PHYSICS_TRAJECTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Local imports to avoid pulling the physics engine into module
    # import time (it transitively imports the LLM client).
    from shadow_loom.narrative_physics import calculate_narrative_physics
    from shadow_loom.query_models import ObservationQuery

    tmin, tmax = fabula_time_bounds(ws)
    request = ObservationQuery(focus_entity_ids=focus_entity_ids or [])

    if tmax <= tmin:
        result = calculate_narrative_physics(
            request, ws, temporal_anchor=tmin,
        )
        metrics = _physics_metrics_from_payload(
            result.get("physics_state", {}) or {}
        )
        out = ([tmin], {k: [v] for k, v in metrics.items()})
        if len(_PHYSICS_TRAJECTORY_CACHE) >= _PHYSICS_TRAJECTORY_CACHE_MAX:
            _PHYSICS_TRAJECTORY_CACHE.pop(next(iter(_PHYSICS_TRAJECTORY_CACHE)))
        _PHYSICS_TRAJECTORY_CACHE[cache_key] = out
        return out

    step = max(1, (tmax - tmin) // (samples - 1))
    times = list(range(tmin, tmax + 1, step))
    if times[-1] != tmax:
        times.append(tmax)

    series: dict[str, list[float]] = {k: [] for k in PHYSICS_METRIC_COLORS}
    for t in times:
        result = calculate_narrative_physics(
            request, ws, temporal_anchor=t,
        )
        metrics = _physics_metrics_from_payload(
            result.get("physics_state", {}) or {}
        )
        for k in series:
            series[k].append(round(float(metrics.get(k, 0.0)), 3))
    out = (times, series)
    if len(_PHYSICS_TRAJECTORY_CACHE) >= _PHYSICS_TRAJECTORY_CACHE_MAX:
        _PHYSICS_TRAJECTORY_CACHE.pop(next(iter(_PHYSICS_TRAJECTORY_CACHE)))
    _PHYSICS_TRAJECTORY_CACHE[cache_key] = out
    return out


# Module-level physics trajectory cache. Keyed by
# ``(id(ws), tuple(sorted(focus)), samples)``; cleared on
# ``WORLD_STATE_CHANGED`` via ``invalidate_physics_trajectory_cache``.
_PHYSICS_TRAJECTORY_CACHE: "dict[tuple, tuple[list[int], dict[str, list[float]]]]" = {}
_PHYSICS_TRAJECTORY_CACHE_MAX = 16


def invalidate_physics_trajectory_cache() -> None:
    """Drop all cached physics trajectories."""
    _PHYSICS_TRAJECTORY_CACHE.clear()
