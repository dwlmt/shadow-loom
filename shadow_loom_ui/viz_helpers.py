# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data transformation helpers: WorldStateV1 → ECharts option dicts.

Pure functions that convert Shadow-Loom world model objects into the
node/link/category structures consumed by Apache ECharts series configs.
No NiceGUI imports — this module is purely data-oriented.
"""

from __future__ import annotations

import html
import logging
import threading
from typing import Any, Callable, Iterable, Optional

from shadow_loom.models import (
    CausalEdge,
    WorldStateV1,
    event_location_at,
    reconstruct_concern_at,
    reconstruct_entity_at,
    reconstruct_location_at,
    reconstruct_object_at,
    reconstruct_proposition_at,
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

# Per-``causality_type`` palette so the Sankey + force graph encode
# the *modality* of causation directly in colour rather than collapsing
# every kind of causal coupling onto a single crimson line. Picked to
# stay distinguishable from EDGE_COLORS above (which colour topology
# edges) and from NODE_COLORS (which colour the node categories the
# edges land on). Order matches the SANKEY_ASPECTS taxonomy.
MODALITY_COLORS: dict[str, str] = {
    "chain_reaction":      "#D8334A",  # crimson — direct event \u2192 event
    "mutation":            "#F5B43C",  # amber   — entity-state mutation
    "mutation_social":     "#E36BB8",  # rose    — social/relationship
    "affordance_gate":     "#3A7BD5",  # sapphire — gating preconditions
    "ambient_propagation": "#2EA6A0",  # teal    — environmental/world
    "world_to_world":      "#8A5CF0",  # iris    — WORLD_ \u2192 WORLD_ coupling
}

MODALITY_LABELS: dict[str, str] = {
    "chain_reaction":      "Event \u2192 Event",
    "mutation":            "Entity mutation",
    "mutation_social":     "Social mutation",
    "affordance_gate":     "Affordance gate",
    "ambient_propagation": "Ambient propagation",
    "world_to_world":      "World \u2192 World",
}


def _modality_key(ce: "CausalEdge") -> str:
    """Return the palette key for a causal edge.

    Treats WORLD_\u2192WORLD_ couplings as their own modality so the
    cross-trait latent network reads as distinct from the dominant
    event mesh, regardless of the edge's declared ``causality_type``.
    """
    if (
        ce.source_id.startswith("WORLD_")
        and ce.target_id.startswith("WORLD_")
    ):
        return "world_to_world"
    return ce.causality_type

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
    *,
    fabula_t: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Convert a WorldStateV1 into ECharts graph ``(nodes, links, categories)``."""
    nodes: list[dict] = []
    links: list[dict] = []
    all_ids: set[str] = set()

    def _node(nid: str, name: str, ntype: str, **extra: Any) -> None:
        all_ids.add(nid)
        # C1 (twelfth-pass audit): escape all dynamic fields before
        # composing the HTML tooltip; ECharts renders the formatter
        # string as HTML, so an LLM-extracted description containing
        # ``<img onerror=...>`` would otherwise execute on hover.
        tooltip = f"<b>{html.escape(str(name))}</b><br/>Type: {html.escape(str(ntype))}"
        for k, v in extra.items():
            if v:
                tooltip += f"<br/>{html.escape(str(k))}: {html.escape(str(v))}"
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
        superseded = bool(getattr(evt, "superseded_by_event_id", None))
        node_extras: dict[str, Any] = {
            "description": evt.description[:80],
            "fabula_time": str(evt.fabula_time),
            "_sl_node_type": "EventNode",
            "_sl_superseded": superseded,
        }
        if superseded:
            node_extras["superseded_by"] = evt.superseded_by_event_id
        _node(evt.id, evt.id, "EventNode", **node_extras)
        if superseded:
            # Mute the just-appended node so the constraint field
            # reads at-a-glance: superseded events sit ghost-grey with
            # a dashed amber border and 0.45 opacity. Downstream
            # filters can hide entirely via the ``_sl_superseded`` flag.
            n = nodes[-1]
            n["itemStyle"] = {
                **n.get("itemStyle", {}),
                "color": "#9CA3AF",
                "opacity": 0.45,
                "borderColor": "#F59E0B",
                "borderType": "dashed",
                "borderWidth": 1.5,
            }

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
        edge_dash = "dashed" if getattr(evt, "superseded_by_event_id", None) else "solid"
        for aid in evt.actor_ids:
            _link(aid, evt.id, "actor_of", width=1.5, dash=edge_dash)
        for tid in evt.target_ids:
            # Distinguish entity-target vs object/location-target so
            # the legend reads cleanly.
            etype = "target_of" if tid in ws.entities else "used_in"
            _link(evt.id, tid, etype, width=1.5, dash=edge_dash)

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

    # --- Current-event filter -----------------------------------------
    # When fabula_t is set (pinned cursor), restrict the graph to event
    # nodes at that exact fabula time plus their 1-hop neighbours.  This
    # shows "what is happening RIGHT NOW" rather than the accumulated
    # history of everything that has occurred up to the cursor.
    if fabula_t is not None and ws.events:
        evts_now = [e for e in ws.events if e.fabula_time == fabula_t]
        if not evts_now:
            all_times = sorted({e.fabula_time for e in ws.events})
            if all_times:
                nearest = min(all_times, key=lambda t: abs(t - fabula_t))
                evts_now = [e for e in ws.events if e.fabula_time == nearest]
        if evts_now:
            # Seed: the event nodes themselves.
            focus: set[str] = {e.id for e in evts_now}
            # 1-hop expansion over all graph links.
            adj_: dict[str, set[str]] = {}
            for lnk in links:
                adj_.setdefault(lnk["source"], set()).add(lnk["target"])
                adj_.setdefault(lnk["target"], set()).add(lnk["source"])
            reachable: set[str] = set(focus)
            for fid in focus:
                reachable |= adj_.get(fid, set())
            nodes = [n for n in nodes if n["id"] in reachable]
            links = [
                l for l in links
                if l["source"] in reachable and l["target"] in reachable
            ]

    return nodes, links, list(CATEGORIES)


# ── Ego graph ──────────────────────────────────────────────────────

def ws_to_ego_graph_data(
    ws: WorldStateV1,
    focus_ids: list[str],
    max_hops: int = 2,
    *,
    fabula_t: int | None = None,
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
    full_nodes, full_links, cats = ws_to_graph_data(ws, fabula_t=fabula_t)
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
    Each link gets a per-link ``lineStyle.color`` keyed off the
    edge's modality (see :data:`MODALITY_COLORS`) so the Sankey
    encodes *what kind* of causation the flow represents, not just
    its volume.
    """
    edges_with_mod: list[tuple[CausalEdge, str]] = []
    for ce in ws.causal_topology:
        if edge_filter is not None and not edge_filter(ce):
            continue
        edges_with_mod.append((ce, _modality_key(ce)))

    def _iter():
        for ce, modality in edges_with_mod:
            tip = (
                f"<b>{html.escape(str(MODALITY_LABELS.get(modality, ce.causality_type)))}"
                f"</b> \u00b7 {html.escape(str(ce.mechanism))}<br/>"
                f"force {ce.causal_force:.1f} \u00b7 evidence "
                f"{html.escape(str(ce.evidence_strength))}<br/>fabula t={ce.fabula_time}"
            )
            yield (ce.source_id, ce.target_id, ce.causal_force, tip)

    nodes, links = _build_sankey(ws, _iter())
    # _build_sankey iterates raw_edges in fabula order and may drop
    # cycle-closing edges; re-walk the *kept* edges so per-link
    # colours line up with what's actually drawn.
    kept = {(s, t) for s, t in {
        (l["source"], l["target"]) for l in links
    }}
    mod_by_pair: dict[tuple[str, str], str] = {}
    force_by_pair: dict[tuple[str, str], float] = {}
    for ce, modality in edges_with_mod:
        pair = (ce.source_id, ce.target_id)
        if pair not in kept:
            continue
        # Pick the strongest edge's modality if multiple share a pair.
        prev_force = force_by_pair.get(pair, -1.0)
        if float(ce.causal_force) > prev_force:
            mod_by_pair[pair] = modality
            force_by_pair[pair] = float(ce.causal_force)
    for link in links:
        pair = (link["source"], link["target"])
        modality = mod_by_pair.get(pair)
        if modality is None:
            continue
        link["_sl_modality"] = modality
        ls = dict(link.get("lineStyle") or {})
        ls["color"] = MODALITY_COLORS.get(
            modality, EDGE_COLORS["causal"]
        )
        ls.setdefault("opacity", 0.55)
        link["lineStyle"] = ls
    return nodes, links


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
                f"medium: {html.escape(str(ch.medium if ch else 'unmediated'))}<br/>"
                f"truth: {html.escape(str(evt.truth_value or 'unspecified'))}<br/>"
                f"syuzhet={evt.syuzhet_index} fabula={evt.fabula_time}"
            )
            sender = evt.speaker_id or (evt.actor_ids[0] if evt.actor_ids else None)
            if not sender:
                continue
            for aid in evt.addressee_ids:
                yield (sender, aid, 1.0, tip)
        for ch in ws.channels.values():
            tip = (
                f"medium: {html.escape(str(ch.medium))}<br/>"
                f"directionality: {html.escape(str(ch.directionality))}<br/>"
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
                f"{html.escape(str(ce.causality_type))} · {html.escape(str(ce.mechanism))}<br/>"
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
    focus_id: str | None = None,
    focus_max_hops: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Dispatch ``aspect`` (key from ``SANKEY_ASPECTS``) to the right builder.

    ``min_force`` and ``fabula_max`` apply to causal-edge aspects only.
    ``focus_id`` (when set) applies a post-build BFS \u00b1
    ``focus_max_hops`` filter on the resulting nodes/links so the
    Sankey can drill into a single chain. Works for every aspect
    (causal, information, world_influence) by operating on the
    built node/link lists.
    """
    if aspect == "information":
        nodes, links = ws_to_information_sankey_data(ws)
    elif aspect == "world_influence":
        nodes, links = ws_to_world_influence_sankey_data(ws)
    else:
        def _flt(ce: CausalEdge) -> bool:
            if ce.causal_force < min_force:
                return False
            if fabula_max is not None and ce.fabula_time > fabula_max:
                return False
            if aspect == "causal_all":
                return True
            return ce.causality_type == aspect
        nodes, links = ws_to_sankey_data(ws, edge_filter=_flt)

    if focus_id is None:
        return nodes, links

    # BFS \u00b1 max_hops over the directed link graph.
    out_adj: dict[str, set[str]] = {}
    in_adj: dict[str, set[str]] = {}
    for lk in links:
        out_adj.setdefault(lk["source"], set()).add(lk["target"])
        in_adj.setdefault(lk["target"], set()).add(lk["source"])
    keep: set[str] = {focus_id}
    frontier: set[str] = {focus_id}
    for _ in range(max(1, int(focus_max_hops))):
        nxt: set[str] = set()
        for nid in frontier:
            nxt.update(out_adj.get(nid, ()))
            nxt.update(in_adj.get(nid, ()))
        nxt -= keep
        if not nxt:
            break
        keep.update(nxt)
        frontier = nxt
    nodes = [n for n in nodes if n.get("name") in keep]
    links = [
        l for l in links
        if l["source"] in keep and l["target"] in keep
    ]
    # Highlight focus node so the chain anchor is visible.
    for n in nodes:
        if n.get("name") == focus_id:
            style = dict(n.get("itemStyle") or {})
            style["borderColor"] = "#FFD700"
            style["borderWidth"] = 3
            n["itemStyle"] = style
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
        tooltip = f"<b>{html.escape(str(ent.name))}</b><br/>Status: {html.escape(str(ent.status))}<br/>{html.escape(top_traits)}"
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
        tooltip = f"<b>{html.escape(str(loc.name))}</b>"
        if loc.description:
            tooltip += f"<br/>{html.escape(loc.description[:80])}"
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


# ── Map view (spatial + entities + objects + active channels) ──────

# Status-coloured entity markers used by the Map view. Kept small and
# explicit so an unfamiliar status string is rendered grey rather than
# silently dropped.
_MAP_STATUS_COLORS: dict[str, str] = {
    "healthy":     "#22c55e",  # emerald
    "injured":     "#f59e0b",  # amber
    "ill":         "#a855f7",  # violet
    "unconscious": "#64748b",  # slate
    "dead":        "#9ca3af",  # grey
}

_MAP_OFFSTAGE_LOC_ID = "__sl_offstage__"
_MAP_UNKNOWN_LOC_ID = "__sl_unknown__"


def ws_to_map_graph_data(
    ws: WorldStateV1,
    *,
    fabula_anchor: int,
    show_entities: bool = True,
    show_objects: bool = True,
    show_channels: bool = True,
    show_locked: bool = True,
    show_events: bool = True,
    channel_window: int = 0,
    event_window: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Map-view graph: locations, entities-in-location, objects-on-entity-or-floor.

    The single canvas behind the World tab's *Map* sub-tab. Resolves
    each entity / object's position at ``fabula_anchor`` via the
    snapshot-replay helpers in :mod:`shadow_loom.models` so movement
    and ownership transfers track the cursor.

    Parameters
    ----------
    fabula_anchor:
        Cursor on the fabula axis; the snapshot helpers replay every
        ``state_timeline`` entry up to and including this tick.
    show_entities, show_objects, show_channels, show_locked:
        Layer toggles. ``show_locked=False`` hides spatial edges
        flagged ``is_locked``.
    channel_window:
        Show channel arcs for utterances whose ``fabula_time`` lies
        within ``\u00b1channel_window`` of the cursor. ``0`` (the default
        and the user-spec rule) renders an arc only when an utterance
        fires *exactly* on the current tick.

    Returns ``(nodes, links, categories)`` shaped for an ECharts
    ``graph`` series. Categories are ``Location`` (0), ``Entity`` (1),
    ``NarrativeObject`` (2). Off-stage / unknown-location entities and
    objects are bucketed into pseudo-locations (rendered with a
    distinct grey fill) so users can see them rather than have them
    silently dropped from the canvas.
    """
    nodes: list[dict] = []
    links: list[dict] = []
    cats = [
        {"name": "Location", "itemStyle": {"color": NODE_COLORS["Location"]}},
        {"name": "Entity", "itemStyle": {"color": NODE_COLORS["Entity"]}},
        {"name": "Object", "itemStyle": {"color": NODE_COLORS["NarrativeObject"]}},
        {"name": "Event (★)", "itemStyle": {"color": "#facc15"}},
    ]

    fabula_anchor = int(fabula_anchor)

    # ── Inferred entity locations from nearby event anchors ──────
    # When an entity has no resolved snapshot location at the
    # cursor, fall back to the at_location_id of any event the
    # entity participates in within ±event_window of the cursor.
    # Captures the implicit invariant: "if something happens at a
    # location the characters and objects involved are present
    # together". Channel-mediated addressees are exempt from this
    # inference.
    inferred_loc: dict[str, str] = {}
    inferred_obj_loc: dict[str, str] = {}
    if ws.events:
        ev_window_inf = max(int(event_window), int(channel_window), 0)
        # Iterate nearest-cursor first so the closest event wins.
        nearby = [
            evt for evt in ws.events
            if evt.fabula_time is not None
            and abs(int(evt.fabula_time) - fabula_anchor) <= ev_window_inf
        ]
        nearby.sort(key=lambda e: abs(int(e.fabula_time) - fabula_anchor))
        for evt in nearby:
            try:
                evt_loc = event_location_at(evt, ws, fallback="actor")
            except Exception:
                evt_loc = getattr(evt, "at_location_id", None)
            if not evt_loc or evt_loc not in ws.locations:
                continue
            via = getattr(evt, "via_channel_id", None)
            channel_addressees = (
                set(evt.addressee_ids or []) if via else set()
            )
            participants = (
                set(evt.actor_ids or [])
                | (set(evt.target_ids or []) - channel_addressees)
            )
            sp = getattr(evt, "speaker_id", None)
            if sp:
                participants.add(sp)
            for pid in participants:
                if pid in ws.entities and pid not in inferred_loc:
                    inferred_loc[pid] = evt_loc
                elif pid in ws.objects and pid not in inferred_obj_loc:
                    inferred_obj_loc[pid] = evt_loc

    # ── Locations (real + pseudo) ────────────────────────────────
    used_pseudo_offstage = False
    used_pseudo_unknown = False
    real_loc_ids: set[str] = set()
    for lid, loc in ws.locations.items():
        real_loc_ids.add(lid)
        tooltip = f"<b>{html.escape(str(loc.name))}</b>"
        if loc.description:
            tooltip += f"<br/>{html.escape(loc.description[:120])}"
        nodes.append({
            "id": lid,
            "name": loc.name,
            "category": 0,
            "symbolSize": [70, 28],
            "symbol": "roundRect",
            "itemStyle": {
                "color": NODE_COLORS["Location"],
                "opacity": 0.85,
                "borderColor": "#1e293b",
                "borderWidth": 1,
            },
            "label": {
                "show": True,
                "position": "inside",
                "color": "#ffffff",
                "fontSize": 10,
                "overflow": "truncate",
                "width": 64,
                "ellipsis": "\u2026",
            },
            "tooltip": {"formatter": tooltip},
            "_sl_node_type": "Location",
        })

    def _ensure_offstage() -> str:
        nonlocal used_pseudo_offstage
        if not used_pseudo_offstage:
            nodes.append({
                "id": _MAP_OFFSTAGE_LOC_ID,
                "name": "(off-stage)",
                "category": 0,
                "symbolSize": [60, 24],
                "symbol": "roundRect",
                "itemStyle": {"color": "#475569", "opacity": 0.5},
                "label": {
                    "show": True, "position": "inside",
                    "color": "#e2e8f0", "fontSize": 10,
                    "overflow": "truncate", "width": 54,
                },
                "tooltip": {"formatter": "Entities/objects with no resolved location at this tick."},
                "_sl_node_type": "Location",
            })
            used_pseudo_offstage = True
        return _MAP_OFFSTAGE_LOC_ID

    def _ensure_unknown() -> str:
        nonlocal used_pseudo_unknown
        if not used_pseudo_unknown:
            nodes.append({
                "id": _MAP_UNKNOWN_LOC_ID,
                "name": "(unknown loc)",
                "category": 0,
                "symbolSize": [60, 24],
                "symbol": "roundRect",
                "itemStyle": {"color": "#7c2d12", "opacity": 0.5},
                "label": {
                    "show": True, "position": "inside",
                    "color": "#fed7aa", "fontSize": 10,
                    "overflow": "truncate", "width": 54,
                },
                "tooltip": {"formatter": "References to LOC_ ids not present in this world."},
                "_sl_node_type": "Location",
            })
            used_pseudo_unknown = True
        return _MAP_UNKNOWN_LOC_ID

    # ── Spatial edges between Locations ──────────────────────────
    for se in ws.spatial_topology:
        if se.destroyed_at_fabula is not None and int(se.destroyed_at_fabula) <= fabula_anchor:
            continue
        if se.established_at_fabula and int(se.established_at_fabula) > fabula_anchor:
            continue
        if se.is_locked and not show_locked:
            continue
        style = "dashed" if se.is_locked else "solid"
        links.append({
            "source": se.source_id,
            "target": se.target_id,
            "lineStyle": {
                "color": EDGE_COLORS["connected_to"],
                "width": 2,
                "type": style,
                "opacity": 0.6,
            },
            "symbol": ["none", "arrow" if not se.bidirectional else "none"],
        })

    # ── Entities resolved at the cursor ──────────────────────────
    entity_loc: dict[str, str] = {}
    if show_entities:
        for ent_id, ent in ws.entities.items():
            try:
                snap = reconstruct_entity_at(ent, fabula_anchor)
                resolved_loc = snap.get("location_id") or ent.location_id
                resolved_status = snap.get("status") or ent.status
            except Exception:
                resolved_loc = ent.location_id
                resolved_status = ent.status
            if resolved_loc and resolved_loc in real_loc_ids:
                anchor_loc = resolved_loc
            elif resolved_loc:
                anchor_loc = _ensure_unknown()
            elif ent_id in inferred_loc:
                # Implicit co-presence inference from a nearby event
                # anchor (see ``inferred_loc`` build-up above).
                anchor_loc = inferred_loc[ent_id]
            else:
                anchor_loc = _ensure_offstage()
            entity_loc[ent_id] = anchor_loc

            colour = _MAP_STATUS_COLORS.get(resolved_status or "", "#94a3b8")
            border = "#1e293b"
            if resolved_status == "dead":
                border = "#7f1d1d"
            tooltip = (
                f"<b>{html.escape(str(ent.name))}</b><br/>"
                f"status: {html.escape(str(resolved_status))}<br/>"
                f"loc: {html.escape(str(resolved_loc or '(off-stage)'))}"
            )
            nodes.append({
                "id": ent_id,
                "name": ent.name,
                "category": 1,
                "symbolSize": 14,
                "symbol": "circle",
                "itemStyle": {
                    "color": colour,
                    "borderColor": border,
                    "borderWidth": 1.5,
                    "opacity": 0.4 if resolved_status == "dead" else 1.0,
                },
                "label": {"show": False},
                "tooltip": {"formatter": tooltip},
                "_sl_node_type": "Entity",
            })
            # Visible "located_in" edge so the spatial pairing is
            # legible without having to read the force layout. Kept
            # thin and translucent so multiple entities sharing one
            # location don't drown out the rest of the graph.
            links.append({
                "source": anchor_loc,
                "target": ent_id,
                "lineStyle": {
                    "color": EDGE_COLORS.get("located_in", "#FF8C42"),
                    "width": 1.0,
                    "opacity": 0.55,
                    "type": "solid",
                },
                "symbol": ["none", "none"],
                "_sl_attractor": True,
            })

    # ── Objects resolved at the cursor ───────────────────────────
    if show_objects:
        for obj_id, obj in ws.objects.items():
            try:
                snap = reconstruct_object_at(obj, fabula_anchor)
            except Exception:
                snap = {
                    "location_id": obj.location_id,
                    "owner_id": obj.owner_id,
                    "properties": dict(obj.properties),
                }
            resolved_owner = snap.get("owner_id")
            resolved_loc = snap.get("location_id")

            if resolved_owner and resolved_owner in entity_loc:
                anchor_id = resolved_owner
                placement = f"held by {resolved_owner}"
            elif resolved_loc and resolved_loc in real_loc_ids:
                anchor_id = resolved_loc
                placement = f"in {resolved_loc}"
            elif resolved_owner:
                # owner exists in the world but show_entities was off;
                # fall back to the owner's resolved location if we can.
                fallback = ws.entities.get(resolved_owner)
                if fallback is not None and fallback.location_id in real_loc_ids:
                    anchor_id = fallback.location_id
                else:
                    anchor_id = _ensure_offstage()
                placement = f"held by {resolved_owner}"
            elif resolved_loc:
                anchor_id = _ensure_unknown()
                placement = f"in {resolved_loc} (unknown)"
            elif obj_id in inferred_obj_loc:
                # Implicit inference from a nearby event anchor.
                anchor_id = inferred_obj_loc[obj_id]
                placement = f"in {anchor_id} (via event)"
            else:
                anchor_id = _ensure_offstage()
                placement = "off-stage"

            props = snap.get("properties") or {}
            prop_str = ", ".join(f"{k}={v}" for k, v in list(props.items())[:4])
            tooltip = f"<b>{html.escape(str(obj.name))}</b><br/>{html.escape(str(placement))}"
            if prop_str:
                tooltip += f"<br/>{html.escape(prop_str)}"

            nodes.append({
                "id": obj_id,
                "name": obj.name,
                "category": 2,
                "symbolSize": 9,
                "symbol": "diamond",
                "itemStyle": {
                    "color": NODE_COLORS["NarrativeObject"],
                    "borderColor": "#1e293b",
                    "borderWidth": 1,
                },
                "label": {"show": False},
                "tooltip": {"formatter": tooltip},
                "_sl_node_type": "NarrativeObject",
            })
            # Visible placement edge: orange tangerine for in-location
            # objects, dashed for held-by-entity.
            held = anchor_id in ws.entities
            links.append({
                "source": anchor_id,
                "target": obj_id,
                "lineStyle": {
                    "color": EDGE_COLORS.get(
                        "owned_by" if held else "located_in",
                        "#FF8C42",
                    ),
                    "width": 1.0,
                    "opacity": 0.5,
                    "type": "dashed" if held else "solid",
                },
                "symbol": ["none", "none"],
                "_sl_attractor": True,
            })

    # ── Active channel arcs ──────────────────────────────────────
    # Two passes: (1) for each Channel that is *active* at the
    # cursor, draw a faint dotted arc between every pair of its
    # ``participant_ids`` so the communication topology is visible
    # at a glance even between utterances; (2) overlay a brighter
    # dashed arc per utterance fired within ±channel_window of the
    # cursor.
    if show_channels and show_entities:
        ent_ids_present = {n["id"] for n in nodes if n.get("category") == 1}
        for ch in (ws.channels or {}).values():
            est = int(ch.established_at_fabula or 0)
            term = ch.terminated_at_fabula
            if est > fabula_anchor:
                continue
            if term is not None and int(term) <= fabula_anchor:
                continue
            parts = [p for p in (ch.participant_ids or []) if p in ent_ids_present]
            if len(parts) < 2:
                continue
            arc_colour = NODE_COLORS.get("Channel", "#C46BD9")
            tip = (
                f"<b>{html.escape(str(ch.name))}</b><br/>"
                f"medium: {html.escape(str(ch.medium))}<br/>"
                f"directionality: {html.escape(str(ch.directionality))}"
            )
            if ch.directionality in ("broadcast", "simplex"):
                src = parts[0]
                pairs = [(src, tgt) for tgt in parts[1:]]
                arrow = ["none", "arrow"]
            else:
                pairs = [
                    (parts[i], parts[j])
                    for i in range(len(parts))
                    for j in range(i + 1, len(parts))
                ]
                arrow = ["none", "none"]
            for src, tgt in pairs:
                links.append({
                    "source": src,
                    "target": tgt,
                    "lineStyle": {
                        "color": arc_colour,
                        "width": 1.0,
                        "type": "dotted",
                        "curveness": 0.2,
                        "opacity": 0.45,
                    },
                    "symbol": arrow,
                    "symbolSize": 4,
                    "tooltip": {"formatter": tip},
                    "_sl_channel_dormant": True,
                })

        window = max(0, int(channel_window))
        for evt in ws.events or []:
            if evt.event_type != "utterance":
                continue
            try:
                evt_t = int(evt.fabula_time)
            except Exception:
                continue
            if abs(evt_t - fabula_anchor) > window:
                continue
            speaker = evt.speaker_id
            if not speaker:
                # Fall back to first actor when speaker is unset.
                speaker = next(iter(evt.actor_ids or []), None)
            if not speaker or speaker not in ws.entities:
                continue
            addressees = [a for a in (evt.addressee_ids or []) if a in ws.entities]
            if not addressees:
                continue
            # Channel lifecycle gating.
            ch = ws.channels.get(evt.via_channel_id) if evt.via_channel_id else None
            if ch is not None:
                if ch.established_at_fabula and int(ch.established_at_fabula) > fabula_anchor:
                    continue
                if ch.terminated_at_fabula is not None and int(ch.terminated_at_fabula) <= fabula_anchor:
                    continue
            medium = ch.medium if ch is not None else "direct"
            ch_label = ch.name if ch is not None else "(no channel)"
            arc_colour = EDGE_COLORS.get("communicating_with", "#F5B43C")
            content = (evt.content or evt.description or "").strip()
            if len(content) > 120:
                content = content[:117] + "\u2026"
            for addr in addressees:
                tip = (
                    f"<b>utterance @ t={evt_t}</b><br/>"
                    f"{html.escape(str(speaker))} \u2192 {html.escape(str(addr))}<br/>"
                    f"channel: {html.escape(str(ch_label))}<br/>"
                    f"medium: {html.escape(str(medium))}"
                )
                if content:
                    tip += f"<br/><i>{html.escape(content)}</i>"
                links.append({
                    "source": speaker,
                    "target": addr,
                    "lineStyle": {
                        "color": arc_colour,
                        "width": 2.5,
                        "type": "dashed",
                        "curveness": 0.25,
                        "opacity": 0.9,
                    },
                    "symbol": ["none", "arrow"],
                    "symbolSize": 6,
                    "tooltip": {"formatter": tip},
                    "_sl_channel_arc": True,
                })

    # ── Event glyphs + co-presence highlights ─────────────────────
    # PR 7 of EventNode.at_location_id. Render a star (★) glyph at
    # every windowed event's resolved location; outline each bound
    # participant in yellow when their reconstructed location matches
    # the anchor, and in red dashed when it does NOT (a co-presence
    # violation visible at a glance). Channel-mediated addressees are
    # NOT highlighted as physical co-present — the channel arc above
    # already encodes their virtual presence.
    if show_events:
        ev_window = max(0, int(event_window))
        # Index entity nodes by id so we can mutate their itemStyle.
        ent_nodes_by_id = {
            n["id"]: n for n in nodes if n.get("category") == 1
        }
        for evt in ws.events or []:
            try:
                evt_t = int(evt.fabula_time)
            except Exception:
                continue
            if abs(evt_t - fabula_anchor) > ev_window:
                continue
            try:
                evt_loc = event_location_at(evt, ws, fallback="actor")
            except Exception:
                evt_loc = getattr(evt, "at_location_id", None)
            if not evt_loc or evt_loc not in real_loc_ids:
                continue

            # Channel-mediated addressees are exempt from physical
            # co-presence; everyone else bound by the event must be
            # at ``evt_loc``.
            via = getattr(evt, "via_channel_id", None)
            ch = ws.channels.get(via) if via else None
            channel_addressees: set[str] = set()
            if ch is not None:
                channel_addressees = set(evt.addressee_ids or [])
            bound = set(evt.actor_ids or []) | (
                set(evt.target_ids or []) - channel_addressees
            )
            speaker = getattr(evt, "speaker_id", None)
            if speaker:
                bound.add(speaker)

            for pid in bound:
                node = ent_nodes_by_id.get(pid)
                if node is None:
                    continue
                # Where did the snapshot replay put them?
                resolved = entity_loc.get(pid)
                if resolved == evt_loc:
                    # Correct co-presence — yellow border emphasis.
                    node.setdefault("itemStyle", {})["borderColor"] = "#facc15"
                    node["itemStyle"]["borderWidth"] = 2.5
                else:
                    # Phantom / displaced bound participant.
                    node.setdefault("itemStyle", {})["borderColor"] = "#dc2626"
                    node["itemStyle"]["borderWidth"] = 2.5
                    node["itemStyle"]["borderType"] = "dashed"

            # Star glyph at the event's anchor location.
            desc = (evt.description or "").strip()
            if len(desc) > 120:
                desc = desc[:117] + "\u2026"
            tip = (
                f"<b>{html.escape(str(evt.id))}</b> ({html.escape(str(evt.event_type))}) @ t={evt_t}<br/>"
                f"at {html.escape(str(evt_loc))}"
            )
            if desc:
                tip += f"<br/>{html.escape(desc)}"
            if channel_addressees:
                tip += (
                    "<br/><i>channel-mediated: "
                    f"{html.escape(', '.join(sorted(channel_addressees)))}</i>"
                )
            event_node_id = f"__event_glyph__{evt.id}"
            nodes.append({
                "id": event_node_id,
                "name": "\u2605",
                "category": 3,
                "symbol": "diamond",
                "symbolSize": 12,
                "itemStyle": {
                    "color": "#facc15",
                    "borderColor": "#a16207",
                    "borderWidth": 1.0,
                    "opacity": 0.95,
                },
                "label": {
                    "show": True,
                    "position": "inside",
                    "color": "#1c1917",
                    "fontSize": 12,
                },
                "tooltip": {"formatter": tip},
                "_sl_node_type": "Event",
            })
            links.append({
                "source": evt_loc,
                "target": event_node_id,
                "lineStyle": {
                    "color": "#a16207",
                    "width": 1.0,
                    "opacity": 0.55,
                    "type": "dotted",
                },
                "symbol": ["none", "none"],
                "_sl_attractor": True,
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
    """Events as scatter data: ``[{name, fabula_time, syuzhet_index, event_type, description}]``.

    Superseded events are tagged with ``superseded=True`` and rendered
    with a muted grey colour so the supersession-aware UI surfaces can
    style or hide them.
    """
    out: list[dict] = []
    for evt in sorted(ws.events, key=lambda e: e.fabula_time):
        superseded = bool(getattr(evt, "superseded_by_event_id", None))
        color = (
            "#9CA3AF" if superseded
            else EVENT_TYPE_COLORS.get(evt.event_type, "#607D8B")
        )
        out.append({
            "name": evt.id,
            "value": [evt.fabula_time, evt.syuzhet_index],
            "itemStyle": {
                "color": color,
                "opacity": 0.45 if superseded else 1.0,
            },
            "description": (
                ("[superseded] " if superseded else "")
                + evt.description[:80]
            ),
            "event_type": evt.event_type,
            "superseded": superseded,
            "superseded_by_event_id": getattr(evt, "superseded_by_event_id", None),
        })
    return out


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
        #
        # C1 (2026-05-29 twelfth-pass audit): HTML-escape every dynamic
        # field before interpolating into the tooltip markup. Event
        # ``description`` (and, defensively, ``id`` / ``event_type``)
        # originates from LLM-extracted source text and could contain
        # ``<img onerror=...>`` or ``<script>`` payloads; ECharts
        # tooltips render the resulting string as HTML, so unescaped
        # interpolation was an XSS sink.
        lines = "<br/>".join(
            "<b>{}</b> [{}]: {}".format(
                html.escape(str(e["id"])),
                html.escape(str(e["event_type"])),
                html.escape(str(e["description"])),
            )
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
    *,
    focus_id: str | None = None,
    focus_max_hops: int = 2,
    layout_hint: str = "force",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build a force-directed graph of causal edges with thickness = causal_force.

    ``focus_id``: when set, restrict the graph to the BFS neighbourhood
    of that node (both ancestors and descendants up to
    ``focus_max_hops`` along the directed causal graph). The focus
    node itself gets a gold border so the user can see what's being
    drilled into.

    ``layout_hint``: when ``"timeline"`` the function attaches an
    ``x`` coordinate (= ``fabula_time``) to every event node so the
    caller can render with ``coordinateSystem='cartesian2d'`` *or*
    pass through to the standard ``force`` layout where the
    pre-seeded x positions act as a soft temporal anchor (events
    drift left-to-right by fabula time, vertical position relaxed by
    the force solver). Non-event nodes get their x averaged from the
    events that touch them.

    Returns ``(nodes, links, categories)`` for ECharts graph series.
    """
    edges = list(ws.causal_topology)
    if focus_id is not None:
        # Build adjacency in both directions for BFS.
        out_adj: dict[str, set[str]] = {}
        in_adj: dict[str, set[str]] = {}
        for ce in edges:
            out_adj.setdefault(ce.source_id, set()).add(ce.target_id)
            in_adj.setdefault(ce.target_id, set()).add(ce.source_id)
        keep: set[str] = {focus_id}
        frontier: set[str] = {focus_id}
        for _ in range(max(1, int(focus_max_hops))):
            nxt: set[str] = set()
            for nid in frontier:
                nxt.update(out_adj.get(nid, ()))
                nxt.update(in_adj.get(nid, ()))
            nxt -= keep
            if not nxt:
                break
            keep.update(nxt)
            frontier = nxt
        edges = [
            ce for ce in edges
            if ce.source_id in keep and ce.target_id in keep
        ]

    # Pre-compute timeline x-anchors when requested.
    x_by_id: dict[str, float] = {}
    if layout_hint == "timeline":
        evt_by_id = {e.id: e for e in ws.events}
        for nid in {ce.source_id for ce in edges} | {
            ce.target_id for ce in edges
        }:
            if nid in evt_by_id:
                x_by_id[nid] = float(evt_by_id[nid].fabula_time)
        # Average non-event nodes from neighbouring events.
        non_evt = {
            nid for ce in edges for nid in (ce.source_id, ce.target_id)
            if nid not in evt_by_id
        }
        for nid in non_evt:
            xs: list[float] = []
            for ce in edges:
                if ce.source_id == nid and ce.target_id in evt_by_id:
                    xs.append(float(evt_by_id[ce.target_id].fabula_time))
                elif ce.target_id == nid and ce.source_id in evt_by_id:
                    xs.append(float(evt_by_id[ce.source_id].fabula_time))
            if xs:
                x_by_id[nid] = sum(xs) / len(xs)

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
    evt_by_id = {evt.id: evt for evt in ws.events}

    def _ensure(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        if nid in evt_by_id:
            evt = evt_by_id[nid]
            node = {
                "id": nid, "name": nid,
                "category": 0,
                "symbolSize": 20, "symbol": "triangle",
                "itemStyle": {"color": NODE_COLORS["EventNode"]},
                "tooltip": {"formatter": (
                    f"<b>{nid}</b> [{evt.event_type}]<br/>"
                    f"t={evt.fabula_time}, s={evt.syuzhet_index}<br/>"
                    f"{(evt.description or '')[:80]}"
                )},
            }
        elif nid in ws.entities:
            node = {
                "id": nid, "name": ws.entities[nid].name,
                "category": 1,
                "symbolSize": 25, "symbol": "circle",
                "itemStyle": {"color": NODE_COLORS["Entity"]},
            }
        elif nid in ws.locations:
            node = {
                "id": nid, "name": ws.locations[nid].name,
                "category": 2,
                "symbolSize": 20, "symbol": "rect",
                "itemStyle": {"color": NODE_COLORS["Location"]},
            }
        elif nid in ws.objects:
            node = {
                "id": nid, "name": ws.objects[nid].name,
                "category": 3,
                "symbolSize": 16, "symbol": "diamond",
                "itemStyle": {"color": NODE_COLORS["NarrativeObject"]},
            }
        elif nid in ws.world_traits:
            node = {
                "id": nid, "name": ws.world_traits[nid].name,
                "category": 4,
                "symbolSize": 20, "symbol": "pin",
                "itemStyle": {"color": NODE_COLORS["WorldTrait"]},
            }
        else:
            node = {
                "id": nid, "name": nid,
                "category": 0,
                "symbolSize": 14,
                "itemStyle": {"color": "#9E9E9E"},
            }
        # Highlight the focus node so it's visually anchored when
        # drilling into a chain.
        if focus_id is not None and nid == focus_id:
            style = dict(node.get("itemStyle") or {})
            style["borderColor"] = "#FFD700"
            style["borderWidth"] = 4
            node["itemStyle"] = style
            node["symbolSize"] = max(node.get("symbolSize", 16) + 6, 28)
        # Attach timeline x-anchor (caller decides whether to use it).
        if nid in x_by_id:
            node["x"] = x_by_id[nid]
        nodes.append(node)

    for ce in edges:
        _ensure(ce.source_id)
        _ensure(ce.target_id)
        w = min(6, max(1, ce.causal_force / 1.5))
        modality = _modality_key(ce)
        color = MODALITY_COLORS.get(modality, EDGE_COLORS["causal"])
        tooltip = (
            f"<b>{html.escape(str(MODALITY_LABELS.get(modality, ce.causality_type)))}</b><br/>"
            f"mechanism: {html.escape(str(ce.mechanism))}<br/>"
            f"force: {ce.causal_force}<br/>"
            f"evidence: {html.escape(str(ce.evidence_strength))}"
        )
        # WORLD_\u2192WORLD_ kept dashed so the latent-coupling lattice
        # is readable even where the iris colour overlaps with other
        # high-saturation hues at a distance.
        is_world_to_world = modality == "world_to_world"
        line_style = {
            "width": max(w, 2.0) if is_world_to_world else w,
            "color": color,
            "opacity": 0.85,
            "curveness": 0.2 if is_world_to_world else 0.15,
        }
        if is_world_to_world:
            line_style["type"] = "dashed"
        links.append({
            "source": ce.source_id,
            "target": ce.target_id,
            "_sl_modality": modality,
            "_sl_force": float(ce.causal_force),
            "lineStyle": line_style,
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


def ws_to_belief_rows(
    ws: WorldStateV1,
    *,
    fabula_t: int | None = None,
) -> list[dict]:
    """All beliefs across all entities as flat table rows.

    Each row carries the believer + target metadata used by the Social
    tab's raw-data table. ``fabula_t`` filters out beliefs whose
    ``established_at_fabula`` is later than the cursor.
    """
    rows: list[dict] = []
    for ent in ws.entities.values():
        for b in ent.beliefs:
            est = int(getattr(b, "established_at_fabula", 0) or 0)
            if fabula_t is not None and est > fabula_t:
                continue
            target = ws.entities.get(b.target_id)
            if target is not None:
                target_name = target.name
                target_kind = "entity"
            elif b.target_id in ws.objects:
                target_name = ws.objects[b.target_id].name
                target_kind = "object"
            elif b.target_id in ws.locations:
                target_name = ws.locations[b.target_id].name
                target_kind = "location"
            else:
                target_name = b.target_id
                target_kind = "—"
            via_chn_id = getattr(b, "acquired_via_channel_id", None)
            via_chn_name = (
                ws.channels[via_chn_id].name
                if via_chn_id and via_chn_id in ws.channels
                else None
            )
            rows.append({
                "believer": ent.name,
                "believer_id": ent.id,
                "target": target_name,
                "target_id": b.target_id,
                "target_kind": target_kind,
                "perceived_state": b.perceived_state,
                "confidence": round(float(b.confidence), 2),
                "inertia": round(float(b.inertia), 2),
                "established_at_fabula": est,
                "acquired_via_event_id": getattr(
                    b, "acquired_via_event_id", None
                ),
                "acquired_via_channel": via_chn_name,
                "proposition_id": getattr(b, "proposition_id", None),
            })
    rows.sort(key=lambda r: (-r["confidence"], r["believer"], r["target"]))
    return rows


def ws_to_knowledge_asymmetry_matrix(
    ws: WorldStateV1,
    *,
    fabula_t: int | None = None,
) -> tuple[list[str], list[str], list[list], list[dict]]:
    """Character × proposition asymmetry matrix.

    Returns ``(entity_names, prop_labels, cells, meta)`` where each
    cell is ``[col_idx, row_idx, value]`` with value in ``[-1, +1]``:

      * ``+conf``   character believes (any state) and their belief is
                    consistent with the proposition being **true** at
                    the cursor.
      * ``-conf``   character believes but truth is **false** at the
                    cursor (dramatic-irony / mistaken belief).
      *  ``0``      no belief held (curiosity gap; rendered blank).

    Meta includes the audience prior (so callers can mark a "what the
    audience thinks" row at the top) and the truth value at cursor.

    The truth-stance heuristic assumes ``Belief.perceived_state`` is
    an *assertion* of the proposition; we compare it to
    ``truth_at_fabula`` ≤ cursor. This gives Sternberg's gaps /
    suspense / surprise triad as a single visual sweep.
    """
    from shadow_loom.models import reconstruct_proposition_at

    props = list(ws.propositions or [])
    ents = list(ws.entities.values())
    if not props or not ents:
        return [], [], [], []

    # Resolve truth + meta per prop.
    prop_meta: list[dict] = []
    truth_by_pid: dict[str, bool | None] = {}
    prior_by_pid: dict[str, float] = {}
    for prop in props:
        if fabula_t is not None:
            snap = reconstruct_proposition_at(prop, int(fabula_t))
            truth = snap.get("truth_at")
            prior = float(snap.get(
                "audience_default_prior",
                getattr(prop, "audience_default_prior", 0.5),
            ))
        else:
            truth = None
            for t in sorted((prop.truth_at_fabula or {}).keys()):
                truth = prop.truth_at_fabula[t]
            prior = float(getattr(prop, "audience_default_prior", 0.5))
        truth_by_pid[prop.proposition_id] = truth
        prior_by_pid[prop.proposition_id] = prior
        prop_meta.append({
            "id": prop.proposition_id,
            "label": (prop.description or prop.proposition_id)[:32],
            "truth_at": truth,
            "audience_prior": prior,
            "stakes": float(getattr(prop, "stakes", 0.0)),
        })

    # Entity rows (sort by name for stability).
    ent_sorted = sorted(ents, key=lambda e: (e.name or e.id))
    ent_names = [e.name or e.id for e in ent_sorted]
    prop_labels = [m["label"] for m in prop_meta]

    cells: list[list] = []
    for r, ent in enumerate(ent_sorted):
        # Index this entity's beliefs by proposition_id.
        bel_by_pid: dict[str, float] = {}
        for b in ent.beliefs:
            pid = getattr(b, "proposition_id", None)
            if not pid:
                continue
            if fabula_t is not None and getattr(
                b, "established_at_fabula", 0
            ) > fabula_t:
                continue
            bel_by_pid[pid] = max(
                bel_by_pid.get(pid, 0.0), float(b.confidence)
            )
        for c, m in enumerate(prop_meta):
            conf = bel_by_pid.get(m["id"])
            truth = m["truth_at"]
            if conf is None:
                # blank cell — curiosity gap
                continue
            if truth is None:
                value = 0.05  # belief held but truth unresolved
            elif truth:
                value = round(float(conf), 3)
            else:
                value = round(-float(conf), 3)
            cells.append([c, r, value])

    return ent_names, prop_labels, cells, prop_meta


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
            "terminated_at_fabula": ch.terminated_at_fabula,
            "evidence_strength": ch.evidence_strength,
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
    *,
    group_by: str = "entity",
    top_n: int = 5,
) -> list[list]:
    """Build ThemeRiver series data: ``[[time, value, label], ...]``.

    ``group_by="entity"`` (default): one band per ``entity:trait``
    pair, matching the legacy behaviour.
    ``group_by="trait"``: one band per trait, summed across the cast
    \u2014 cleaner story-level view that doesn't explode with cast size.

    ``trait_names=None``: auto-pick the top ``top_n`` traits by
    *variance \u00d7 occurrence* across the cast (was: top-4 by raw
    occurrence). Variance ranking surfaces the traits that actually
    move during the story, so the river bands carry information
    instead of being flat ribbons of stable axes.

    ThemeRiver requires non-negative magnitudes; we encode trait
    *energy* as ``|value - 0.5|`` for [0,1]-scale traits and
    ``|value|`` for [-1,1]-scale traits, then add a small floor so
    near-zero traits remain visible as a hairline rather than
    disappearing entirely.
    """
    if not ws.events or not ws.entities:
        return []

    times = sorted({int(evt.fabula_time) for evt in ws.events})
    if not times:
        return []

    # Cap the cast.
    ent_ids = list(ws.entities.keys())[:max_entities]
    if not ent_ids:
        return []

    # ---- Trait selection: variance \u00d7 occurrence ----------------------
    if trait_names is None:
        trait_stats: dict[str, list[float]] = {}
        for eid in ent_ids:
            ent = ws.entities[eid]
            for tname, tv in ent.traits.items():
                trait_stats.setdefault(tname, []).append(float(tv.value))
        scored: list[tuple[str, float]] = []
        for tname, vals in trait_stats.items():
            n = len(vals)
            if n < 1:
                continue
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            # variance \u00d7 occurrence so traits that are both volatile
            # and widely-shared rank highest.
            scored.append((tname, var * n))
        scored.sort(key=lambda x: -x[1])
        trait_names = [t for t, _ in scored[:top_n]]
    if not trait_names:
        return []

    data: list[list] = []
    for t in times:
        if group_by == "trait":
            # One band per trait, summing energy across the cast.
            for tn in trait_names:
                total = 0.0
                for eid in ent_ids:
                    ent = ws.entities[eid]
                    snapshot = reconstruct_entity_with_causal(ws, eid, t)
                    tv = snapshot.get("traits", {}).get(tn)
                    if tv is not None:
                        val = tv["value"] if isinstance(tv, dict) else tv
                    elif tn in ent.traits:
                        val = ent.traits[tn].value
                    else:
                        continue
                    total += abs(float(val))
                # Floor so a flat zero series shows as a hairline.
                data.append([str(t), round(max(0.05, total), 3), tn])
        else:
            # One band per entity:trait pair.
            for eid in ent_ids:
                ent = ws.entities[eid]
                snapshot = reconstruct_entity_with_causal(ws, eid, t)
                for tn in trait_names:
                    tv = snapshot.get("traits", {}).get(tn)
                    if tv is not None:
                        val = tv["value"] if isinstance(tv, dict) else tv
                    elif tn in ent.traits:
                        val = ent.traits[tn].value
                    else:
                        continue
                    # Encode magnitude (energy) so negative traits
                    # contribute thickness rather than disappearing.
                    energy = abs(float(val))
                    data.append([
                        str(t), round(max(0.05, energy), 3),
                        f"{ent.name}:{tn}",
                    ])

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
    sort_by: str = "first_appearance",
    event_types: set[str] | None = None,
) -> dict:
    """Build per-entity lifeline segments for ``render_entity_lifelines``.

    Each entity gets a chronological list of ``(start, end, status,
    location_id)`` segments derived from its ``state_timeline`` and
    initial state. Status changes drive segment colour; location
    changes are emitted separately as point markers so the lifeline
    "kinks" visibly at every move.

    ``entity_ids``: if provided, restrict the lanes to this subset.
    ``sort_by``: one of ``"first_appearance"`` (default — matches
    reading order), ``"alphabetical"``, ``"event_count"``,
    ``"last_appearance"``, ``"death_order"``.
    ``event_types``: optional whitelist; events with
    ``event_type not in event_types`` are dropped from the markers.

    Each entity also reports its ``first_t`` / ``last_t`` (from
    snapshots ∪ events they actor'd in) so the renderer can trim
    "not-yet-introduced" / "off-page" portions of the lane to a thin
    grey hairline rather than a misleading full-width healthy ribbon.
    For dead entities, ``death_t`` carries the tick the ``dead``
    status first appeared so the renderer can stop the ribbon and
    drop a ``\u2716`` marker there.

    Returns ``{"entities": [(eid, name)],
              "segments": [{"row", "start", "end", "status",
                            "status_color", "location_name",
                            "duration"}],
              "moves":    [{"row", "time", "location_name"}],
              "events":   [{"row", "time", "event_type", "description",
                            "event_id", "color"}],
              "lifespans":[{"row", "first_t", "last_t",
                             "death_t": int|None,
                             "current_status_at_tmax": str|None,
                             "current_color": str}],
              "tmin": int, "tmax": int}``.
    """
    if not ws.entities:
        return {
            "entities": [], "segments": [], "moves": [],
            "events": [], "lifespans": [], "tmin": 0, "tmax": 0,
        }

    keep: set[str] | None = set(entity_ids) if entity_ids else None
    sel_entities = {
        eid: ent for eid, ent in ws.entities.items()
        if keep is None or eid in keep
    }
    if not sel_entities:
        return {
            "entities": [], "segments": [], "moves": [],
            "events": [], "lifespans": [], "tmin": 0, "tmax": 0,
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

    # ---- Per-entity first/last appearance, death tick, event count -----
    per_ent: dict[str, dict] = {}
    for eid, ent in sel_entities.items():
        snap_ts = [int(s.fabula_time) for s in ent.state_timeline]
        evt_ts = [int(e.fabula_time) for e in ws.events if eid in (e.actor_ids or [])]
        appearances = snap_ts + evt_ts
        first_t = min(appearances) if appearances else tmin
        last_t = max(appearances) if appearances else tmax
        death_t: int | None = None
        for snap in sorted(ent.state_timeline, key=lambda s: s.fabula_time):
            if snap.status == "dead":
                death_t = int(snap.fabula_time)
                break
        per_ent[eid] = {
            "first_t": first_t,
            "last_t": last_t,
            "death_t": death_t,
            "event_count": len(evt_ts),
        }

    # ---- Lane order ----------------------------------------------------
    eid_list = list(sel_entities.keys())
    if sort_by == "alphabetical":
        eid_list.sort(key=lambda e: sel_entities[e].name.lower())
    elif sort_by == "event_count":
        eid_list.sort(key=lambda e: -per_ent[e]["event_count"])
    elif sort_by == "last_appearance":
        eid_list.sort(key=lambda e: per_ent[e]["last_t"])
    elif sort_by == "death_order":
        eid_list.sort(
            key=lambda e: (
                per_ent[e]["death_t"] if per_ent[e]["death_t"] is not None
                else float("inf"),
                per_ent[e]["first_t"],
            )
        )
    else:  # first_appearance (default)
        eid_list.sort(key=lambda e: per_ent[e]["first_t"])

    entities: list[tuple[str, str]] = [
        (eid, sel_entities[eid].name) for eid in eid_list
    ]
    row_for = {eid: i for i, eid in enumerate(eid_list)}

    segments: list[dict] = []
    moves: list[dict] = []
    lifespans: list[dict] = []

    def _loc_name(lid: str | None) -> str:
        if not lid:
            return ""
        loc = ws.locations.get(lid)
        return loc.name if loc else lid

    for eid in eid_list:
        ent = sel_entities[eid]
        row = row_for[eid]
        info = per_ent[eid]
        first_t = info["first_t"]
        death_t = info["death_t"]
        # The visible-life span ends at the death tick (so we don't
        # paint a giant black "dead" ribbon spanning to tmax) or at
        # tmax for entities that survive.
        life_end = death_t if death_t is not None else tmax

        snaps = sorted(ent.state_timeline, key=lambda s: s.fabula_time)
        cur_status = ent.status
        cur_loc = ent.location_id
        seg_start = first_t
        cur_color = _STATUS_COLORS.get(cur_status, "#94a3b8")

        # Walk snapshots, emitting a segment whenever status changes.
        for snap in snaps:
            t = int(snap.fabula_time)
            if t > life_end:
                break
            new_status = snap.status if snap.status is not None else cur_status
            new_loc = snap.location_id if snap.location_id is not None else cur_loc
            if new_status != cur_status and t > seg_start:
                segments.append({
                    "row": row,
                    "start": seg_start,
                    "end": t,
                    "duration": t - seg_start,
                    "status": cur_status or "unknown",
                    "status_color": _STATUS_COLORS.get(cur_status, "#94a3b8"),
                    "location_name": _loc_name(cur_loc),
                })
                seg_start = t
                cur_status = new_status
            else:
                cur_status = new_status
            if new_loc != cur_loc and t >= first_t:
                moves.append({
                    "row": row,
                    "time": t,
                    "location_name": _loc_name(new_loc),
                })
                cur_loc = new_loc
        # Final visible segment runs to ``life_end`` (death or tmax).
        if seg_start < life_end and cur_status != "dead":
            segments.append({
                "row": row,
                "start": seg_start,
                "end": life_end,
                "duration": life_end - seg_start,
                "status": cur_status or "unknown",
                "status_color": _STATUS_COLORS.get(cur_status, "#94a3b8"),
                "location_name": _loc_name(cur_loc),
            })

        # Lifespan summary so the renderer can paint the off-page
        # hairline (before first_t / after life_end) and decorate the
        # y-axis tick label with a current-status swatch.
        lifespans.append({
            "row": row,
            "first_t": first_t,
            "last_t": life_end,
            "death_t": death_t,
            "current_status_at_tmax": cur_status,
            "current_color": _STATUS_COLORS.get(cur_status, "#94a3b8"),
        })

    # Event markers per actor row, optionally chip-filtered.
    events: list[dict] = []
    type_filter = event_type_chip_filter(ws, event_types)
    for evt in sorted(ws.events, key=lambda e: e.fabula_time):
        if evt.event_type not in type_filter:
            continue
        for aid in (evt.actor_ids or []):
            row = row_for.get(aid)
            if row is None:
                continue
            # Skip events that fall after the actor's death — those
            # are typically posthumous references in extracted prose
            # and shouldn't appear as the actor doing something.
            dt = per_ent[aid]["death_t"]
            if dt is not None and int(evt.fabula_time) > dt:
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
        "lifespans": lifespans,
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
# Guards FIFO eviction + clear across every module cache below. The
# snapshot/affect/physics compute functions run inside ``asyncio.to_thread``
# worker threads, so two threads can race on ``pop(next(iter(d)))`` →
# "dictionary changed size during iteration"/KeyError. ``.get`` reads stay
# unlocked (atomic in CPython); only the mutating sections are guarded.
_CACHE_LOCK = threading.Lock()

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
    with _CACHE_LOCK:
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
    with _CACHE_LOCK:
        _SNAPSHOT_CACHE.clear()
        _SNAPSHOT_REVISION += 1
        # The affect caches are keyed on the same revision, so bumping the
        # revision logically invalidates them. We also clear them to keep
        # memory predictable when projects are swapped frequently.
        _AFFECT_SCORE_CACHE.clear()
        _AFFECT_TIMESERIES_CACHE.clear()
        try:
            _CHAR_EMOTION_CACHE.clear()
        except NameError:
            pass


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


# ── Time-axis abstraction (fabula vs syuzhet) ─────────────────────
#
# The UI exposes a single "time axis" toggle that flips every chart
# between Genette's *order of the story* (fabula) and *order of the
# telling* (syuzhet). Charts shouldn't read the cursor directly;
# they should ask for ``event_axis_value(evt, axis)`` and
# ``axis_bounds(ws, axis)`` so the same code drives both modes.

def event_axis_value(evt, axis: str = "fabula") -> int:
    """Return ``evt.fabula_time`` or ``evt.syuzhet_index``.

    Defensively coerces to ``int`` and falls back to fabula if an
    unknown axis is supplied.
    """
    if (axis or "fabula").lower() == "syuzhet":
        return int(getattr(evt, "syuzhet_index", 0) or 0)
    return int(getattr(evt, "fabula_time", 0) or 0)


def axis_bounds(ws: WorldStateV1, axis: str = "fabula") -> tuple[int, int]:
    """Min/max of the chosen time axis across ``ws.events``."""
    if (axis or "fabula").lower() == "syuzhet":
        return syuzhet_time_bounds(ws)
    return fabula_time_bounds(ws)


def resolve_cursor(
    ws: WorldStateV1, axis: str, value: int | None,
) -> int | None:
    """Resolve a cursor value on ``axis`` to an effective fabula time.

    * fabula axis: returns ``value`` unchanged.
    * syuzhet axis: returns the *latest* ``fabula_time`` among events
      whose ``syuzhet_index`` is \u2264 ``value``. This is the "what
      does the audience know by the time syuzhet=k is told?" rule
      that makes anachrony / dramatic-irony analysis legible.
    """
    if value is None:
        return None
    if (axis or "fabula").lower() != "syuzhet":
        return int(value)
    revealed = [
        int(evt.fabula_time) for evt in (ws.events or [])
        if int(getattr(evt, "syuzhet_index", 0) or 0) <= int(value)
    ]
    return max(revealed) if revealed else 0


# ── Shared temporal-chart helpers ─────────────────────────────────
#
# Used by every chart on the Temporal sub-tab so they share a single
# time-axis range and a single cursor "now" line. Without these the
# four charts each computed their own bounds (lifelines used data-
# derived tmin/tmax, gantt used "dataMin"/"dataMax", trait timeline
# used a category axis with one tick per event) so the cursor line
# landed at a different screen-x on each chart and the visual stack
# was incoherent. Centralising the axis options also lets us pass
# the same dict into ECharts ``markLine`` / ``markArea`` consistently.

def temporal_xaxis_options(
    ws: WorldStateV1,
    axis: str = "fabula",
    *,
    name: str | None = None,
) -> dict:
    """ECharts xAxis spec locked to the world's full time range.

    Use ``type: "value"`` on every Temporal chart so a 1-tick gap and
    a 1000-tick gap render with proportional widths (a plain
    ``type: "category"`` axis evenly distributes ticks regardless of
    their numeric distance, which made the trait timeline and theme
    river misleading).

    ``name`` defaults to "Fabula time" or "Syuzhet index" depending
    on the active axis so chart axis labels stay consistent without
    each call site re-deriving the label.
    """
    tmin, tmax = axis_bounds(ws, axis)
    if tmax <= tmin:
        tmax = tmin + 1
    if name is None:
        name = "Syuzhet index" if (axis or "").lower() == "syuzhet" else "Fabula time"
    return {
        "type": "value",
        "min": tmin,
        "max": tmax,
        "name": name,
        "nameGap": 18,
        "nameTextStyle": {"color": "#475569", "fontSize": 10},
        "axisLabel": {"color": "#475569", "fontSize": 9},
        "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
    }


def cursor_markline_series(
    fabula_t: int | None,
    *,
    color: str = "#FF6B35",
    label: str = "now",
) -> dict | None:
    """A near-zero-cost ECharts series whose only job is to draw a
    vertical "now" line at ``fabula_t``.

    Returns ``None`` when ``fabula_t`` is ``None`` so callers can
    ``if s := cursor_markline_series(...): series.append(s)``.

    Implemented as a tiny invisible scatter series carrying a
    ``markLine``; we can't put ``markLine`` on the chart root because
    ECharts only honours it on a series. Using its own series keeps
    the existing chart series untouched (so per-series tooltips,
    legend interactions, and visualMap mappings keep working).
    """
    if fabula_t is None:
        return None
    return {
        "name": "__cursor__",
        "type": "scatter",
        "data": [],
        "silent": True,
        "z": 50,
        "tooltip": {"show": False},
        "legendHoverLink": False,
        "markLine": {
            "silent": True,
            "symbol": ["none", "none"],
            "label": {
                "show": True,
                "position": "insideEndTop",
                "formatter": label,
                "color": color,
                "fontSize": 10,
                "backgroundColor": "rgba(255,255,255,0.85)",
                "padding": [1, 4, 1, 4],
                "borderRadius": 3,
            },
            "lineStyle": {
                "color": color,
                "width": 2,
                "type": "dashed",
                "opacity": 0.85,
            },
            "data": [{"xAxis": int(fabula_t)}],
        },
    }


def event_type_chip_filter(
    ws: WorldStateV1, allowed: set[str] | None,
) -> set[str]:
    """Resolve a chip-filter selection to a concrete set of event types.

    ``allowed=None`` means "no filter applied" (show all). An empty
    set means the user explicitly hid every type — return an empty
    set so callers render an empty chart rather than silently falling
    back to "all".
    """
    if allowed is None:
        return {evt.event_type for evt in (ws.events or [])}
    return set(allowed)


def snapshot_world_at_syuzhet(ws: WorldStateV1, s: int) -> WorldStateV1:
    """Return ``ws`` filtered to events with ``syuzhet_index <= s``.

    Unlike :func:`snapshot_world_at`, this does NOT replay entity or
    world-trait state — those evolve in fabula time, not reading order.
    Only the ``events`` list is trimmed so views ordered by reader
    knowledge (suspense, reveals, dramatic irony) get a reading-time
    cursor.
    """
    cache_key = (id(ws), _SNAPSHOT_REVISION, -1 - s)  # R19-L3: revision-keyed; negative t namespace = syuzhet
    cached = _SNAPSHOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    new = ws.model_copy(deep=False)
    new.events = [evt for evt in ws.events if evt.syuzhet_index <= s]
    with _CACHE_LOCK:
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
    axis: str = "fabula",
) -> dict:
    """Time-sliced entity×entity matrices for an animated heatmap.

    Builds one matrix per cursor tick (capped at ``num_frames``,
    distributed evenly between the world's earliest and latest
    cursor on the chosen ``axis``) using
    :func:`reconstruct_relationship_with_causal` so each frame
    reflects authored snapshots **and** ``mutation_social`` causal
    edges accumulated through that tick. Two-cell symmetry mirrors
    :func:`ws_to_heatmap_data` so both heatmaps render the same way.

    When ``axis='syuzhet'`` the slider scrubs reading-order indices
    and each sample is resolved to the latest revealed fabula tick
    via :func:`resolve_cursor`, so the heatmap shows what dyad state
    the audience has been told by then. ``times`` carries the cursor
    values along the chosen axis (used for tick labels), and
    ``fabula_times`` carries the resolved fabula ticks each frame
    was reconstructed at.

    Returns ``{"names": [...], "times": [...], "fabula_times": [...],
    "frames": [[[x,y,v]...], ...], "axis": "fabula"|"syuzhet"}``
    where ``frames[i]`` is the matrix at ``times[i]``. Returns empty
    lists if the world has no entities or no events.
    """
    ent_ids = list(ws.entities.keys())
    if not ent_ids:
        return {
            "names": [], "times": [], "fabula_times": [],
            "frames": [], "axis": axis,
        }
    ent_names = [ws.entities[eid].name for eid in ent_ids]
    idx = {eid: i for i, eid in enumerate(ent_ids)}

    tmin, tmax = axis_bounds(ws, axis)
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

    # Resolve each cursor sample to a fabula tick. For the fabula
    # axis this is the identity; for syuzhet we walk back to the
    # latest revealed fabula time via resolve_cursor.
    fabula_times = [resolve_cursor(ws, axis, t) or 0 for t in times]

    # Walk every dyad once per frame using the causal-aware
    # reconstructor. We deliberately iterate ``social_topology`` (not
    # the cartesian product of entities) — characters with no edge
    # have no signal to display and would clutter the matrix.
    frames: list[list[list]] = []
    for ft in fabula_times:
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
                ws, rel.source_entity_id, rel.target_entity_id, ft
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

    return {
        "names": ent_names,
        "times": times,
        "fabula_times": fabula_times,
        "frames": frames,
        "axis": axis,
    }


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

    :class:`NarrativeObject` instances are replayed via
    :func:`reconstruct_object_at` so ``location_id`` / ``owner_id`` /
    ``properties`` reflect the state at ``t`` rather than the final
    snapshot.

    Per-entity ``concerns`` are replayed via
    :func:`reconstruct_concern_at` so ``salience`` / ``polarity`` /
    ``activation_fabula_window`` / ``counter_concern_ids`` / ``kind``
    reflect the cursor.

    :class:`Proposition` entries are replayed via
    :func:`reconstruct_proposition_at` so ``stakes`` /
    ``audience_default_prior`` / ``description`` reflect the cursor,
    and ``truth_at_fabula`` is filtered to commits at or before ``t``.

    :class:`Location.ambient_state` is normalised via
    :func:`reconstruct_location_at`. The canonical helper currently
    returns the static ambient_state (no per-key timeline yet) but
    routing through it locks the contract in place for the future
    timeline addition.

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
        # Replay per-entity Concerns so salience / polarity / activation
        # window / counter_concern_ids / kind reflect the cursor. Without
        # this the dashboards downstream of snap_ws (concern table,
        # affective scorers) read final-frame salience even when the user
        # scrubs backwards.
        for concern in (ent.concerns or []):
            csnap = reconstruct_concern_at(concern, t)
            if csnap is None:
                continue
            if csnap.get("salience") is not None:
                concern.salience = csnap["salience"]
            if csnap.get("polarity") is not None:
                concern.polarity = csnap["polarity"]
            if csnap.get("activation_fabula_window") is not None:
                concern.activation_fabula_window = csnap[
                    "activation_fabula_window"
                ]
            if csnap.get("counter_concern_ids") is not None:
                concern.counter_concern_ids = list(
                    csnap["counter_concern_ids"]
                )
            if csnap.get("kind") is not None:
                concern.kind = csnap["kind"]

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

    # Replay NarrativeObject state timelines so location_id / owner_id /
    # properties reflect the state AT t rather than the latest snapshot.
    # ``reconstruct_object_at`` mirrors ``reconstruct_entity_at`` exactly:
    # it starts from initial fields and replays ObjectStateSnapshot entries
    # up to *t* inclusive, honouring set_location_null / set_owner_null and
    # per-key property mutations.
    for oid, obj in new.objects.items():
        obj_snap = reconstruct_object_at(obj, t)
        obj.location_id = obj_snap["location_id"]
        obj.owner_id = obj_snap["owner_id"]
        obj.properties = obj_snap["properties"]

    # Replay Proposition mutable framing (stakes / audience_default_prior /
    # description) and filter ``truth_at_fabula`` to commits at or before
    # ``t`` so dashboards that read prop.truth_at_fabula directly (NLQ
    # answers, prose ledgers) don't leak future truth commits. Mirrors
    # the gates in MCP ``_inspect_proposition`` (R16-9 / R16-10).
    for prop in (new.propositions or []):
        psnap = reconstruct_proposition_at(prop, t)
        if not isinstance(psnap, dict):
            continue
        if psnap.get("stakes") is not None:
            prop.stakes = psnap["stakes"]
        if psnap.get("audience_default_prior") is not None:
            prop.audience_default_prior = psnap["audience_default_prior"]
        if psnap.get("description") is not None:
            prop.description = psnap["description"]
        # Filter truth_at_fabula to <= t. Keys may be int or str post
        # serialization round-trips (see R1-3 in
        # ``reconstruct_proposition_at``); normalise before filtering.
        try:
            tmap = {
                int(k): v for k, v in (prop.truth_at_fabula or {}).items()
            }
        except (TypeError, ValueError):
            tmap = {}
        prop.truth_at_fabula = {
            k: v for k, v in tmap.items() if k <= t
        }

    # Normalise Location.ambient_state through reconstruct_location_at.
    # Today this is a static pass-through (no per-key ambient timeline),
    # but routing every snapshot through the canonical helper locks in
    # the contract so the future LocationStateSnapshot timeline will
    # propagate to every UI surface without further wiring.
    for lid, loc in new.locations.items():
        try:
            _ = reconstruct_location_at(loc, t)
        except Exception:
            # Defensive: future per-key timeline failures must not break
            # snapshot rendering. Live ambient_state remains visible.
            logger.debug(
                "snapshot_world_at: location replay failed for %s",
                lid, exc_info=True,
            )

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
    fabula_anchor: int | None = None,
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
    # Fabula-anchored reveal-set override for the fabula timeseries
    # sweep \u2014 see ``DirectiveAssembler._fabula_anchor_override``. When
    # active we also force ``surprise_local=False`` because the local
    # (Itti-Baldi per-step) form takes ``syuzhet_anchor - 1`` as the
    # prior anchor, which has no meaning under a fabula reveal-set.
    if fabula_anchor is not None:
        assembler._fabula_anchor_override = int(fabula_anchor)
        surprise_local = False
    metric_calls = (
        ("mystery", lambda eids, sa: assembler.compute_mystery_score(eids, sa)),
        ("dramatic_irony", lambda eids, sa: assembler.compute_dramatic_irony_score(eids, sa)),
        ("suspense", lambda eids, sa: assembler.compute_suspense_score(eids, sa)),
        ("surprise", lambda eids, sa: assembler.compute_surprise_score(
            eids, sa, local=surprise_local,
        )),
        ("narrative_tension", lambda eids, sa: assembler.compute_tension_score(eids, sa)),
    )
    # Weber-Fechner perceptual saturation for the engine-grade
    # structural affects. The raw scorers are mathematically
    # well-behaved in [0, 1] but their *typical* corpus peak sits
    # in the 0.1–0.3 band — readers expect a needle near the right
    # of the gauge to mean "very tense", not "0.15 out of 1.0".
    # ``1 - exp(-τ · s)`` maps the perceptually-meaningful raw
    # band onto the visible gauge range while preserving monotone
    # ordering and never exceeding 1.0. Per-metric τ tuned so a
    # canonical mid-arc peak lands near 0.6 of the gauge:
    #   suspense   τ=4 → 0.15 raw → 0.45 gauge
    #   surprise   τ=3 → 0.27 raw → 0.55 gauge (already blended)
    #   irony      τ=3 → matches surprise band
    #   mystery    τ=2 → already a fraction-of-events ratio
    _PERCEPTUAL_TAU = {
        "suspense": 4.0,
        "surprise": 3.0,
        "dramatic_irony": 3.0,
        "mystery": 2.0,
        "narrative_tension": 2.5,
    }
    import math as _math
    for name, fn in metric_calls:
        try:
            raw = float(fn(entity_ids, syuzhet_anchor))
            tau = _PERCEPTUAL_TAU.get(name)
            if tau and 0.0 <= raw <= 1.0:
                out[name] = 1.0 - _math.exp(-tau * raw)
            else:
                out[name] = raw
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
    fabula_anchor: int | None = None,
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
                 syuzhet_anchor, surprise_local, fabula_anchor)
    cached = _AFFECT_SCORE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    result = _compute_affective_scores_uncached(
        ws,
        entity_ids=entity_ids,
        syuzhet_anchor=syuzhet_anchor,
        ws_for_engine=ws_for_engine,
        surprise_local=surprise_local,
        fabula_anchor=fabula_anchor,
    )
    with _CACHE_LOCK:
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
    fabula_anchor: int | None = None,
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
        a soft saturation ``d / (d + K)`` with ``K = 1.5`` (matches
        ``docs/academic-foundations.md`` §3.5). Replaces an earlier
        ``min(1, d / 3)`` clamp that pinned every dense world (ACOTAR
        runs at ~5 edges/event) flat at 1.0 across the entire
        timeline. The saturation form keeps the metric in ``[0, 1]``
        while preserving variation above the K-threshold.
    """
    scores: dict[str, float] = {}
    if not ws.events:
        return scores

    # Heuristic mystery fallback (Sternberg-style "share of the story
    # the reader has not yet been shown"). The engine version below
    # overrides this key when ``entity_ids`` is supplied; this branch
    # exists so the chart still has a mystery line in cast-less views.
    #
    # Previously: ``late_utterances / total_utterances`` where "late"
    # was ``syuzhet_index >= 1``. That collapsed to ~1.0 on every plot
    # whose utterances weren't all crammed at anchor 0 and had no
    # connection to revealed/hidden ancestor structure. The current
    # form is the share of *events* that sit beyond the reader's
    # current syuzhet anchor — a cheap proxy for "how much causal
    # material is still to come" — and degenerates to 0 when the
    # reader has seen everything (anchor is None or at the maximum).
    if syuzhet_anchor is not None:
        unrevealed = sum(
            1 for e in ws.events
            if e.syuzhet_index is not None and e.syuzhet_index > syuzhet_anchor
        )
        scores["mystery"] = min(1.0, unrevealed / max(1, len(ws.events)))
    else:
        scores["mystery"] = 0.0

    rels = ws.social_topology
    # ── Recent-activity pulse from the snapshot's tail ───────────
    # ``conflict`` and ``danger`` computed purely from
    # ``social_topology`` go flat across the timeline whenever the
    # authored fixture pins most ``last_updated_fabula`` to t=0
    # (a_fish_called_wanda is a pathological case: only 5 of 16
    # social edges have updates after t=0, and ``fear`` is observed
    # on just 3 edges world-wide). Fold in a recency pulse derived
    # from the snapshot's own causal/event tail so the chart tracks
    # the actual narrative beat at the cursor rather than the
    # final-frame relationship summary.
    recency_conflict = 0.0
    recency_danger = 0.0
    if ws.events:
        ev_times = [e.fabula_time for e in ws.events if e.fabula_time is not None]
        if ev_times:
            t_max = max(ev_times)
            t_min = min(ev_times)
            span = max(1, t_max - t_min)
            window_lo = t_max - max(1, span // 4)  # last quartile of the snapshot
            recent_events = [
                e for e in ws.events
                if e.fabula_time is not None and e.fabula_time >= window_lo
            ]
            recent_causal = [
                ce for ce in (ws.causal_topology or [])
                if ce.fabula_time is not None and ce.fabula_time >= window_lo
            ]
            # Conflict pulse: share of recent mutation_social edges
            # that pulled affinity down (negative force_signed proxy:
            # we don't store sign so use any high-force social edge).
            if recent_causal:
                social_pulse = sum(
                    1 for ce in recent_causal
                    if ce.causality_type == "mutation_social"
                    and ce.causal_force >= 5.0
                ) / len(recent_causal)
                recency_conflict = min(1.0, social_pulse)
            # Danger pulse: share of recent high-force causal edges
            # (any modality) — high causal_force events are typically
            # threats / violence / consequence beats.
            if recent_causal:
                high_force_recent = sum(
                    1 for ce in recent_causal if ce.causal_force >= 7.0
                ) / len(recent_causal)
                recency_danger = min(1.0, high_force_recent)
            elif recent_events:
                # No causal coverage — fall back to event density.
                recency_danger = min(
                    1.0, len(recent_events) / max(1, len(ws.events))
                )

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
            base_conflict = negative / len(observed_aff)
            # 60% baseline (final-frame topology) + 40% recency pulse.
            scores["conflict"] = min(
                1.0, 0.6 * base_conflict + 0.4 * recency_conflict
            )
        if observed_fear:
            avg_fear = sum(observed_fear) / len(observed_fear)
            base_danger = max(0.0, avg_fear)
            scores["danger"] = min(
                1.0, 0.6 * base_danger + 0.4 * recency_danger
            )
        elif recency_danger > 0.0:
            # No observed fear axis at all — surface the pulse so the
            # gauge isn't dead silent on event-driven worlds.
            scores["danger"] = recency_danger
    elif recency_conflict or recency_danger:
        scores["conflict"] = recency_conflict
        scores["danger"] = recency_danger

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
        if syuzhet_anchor is None and fabula_anchor is None:
            # Default: anchor *before* the first reveal so the entire
            # event list counts as the unrevealed tail. Anchoring at
            # ``max(syuzhet_index)`` (a previous version of this branch)
            # marked every event as already-revealed, which collapsed
            # suspense's ``unrevealed = all - revealed`` set to ∅ and
            # pulled surprise's prior all the way onto the posterior —
            # zeroing both scores on every unanchored snapshot. Using
            # ``min - 1`` keeps the reader at the narrative threshold
            # so the structural affects retain their full contrast.
            #
            # Skipped when ``fabula_anchor`` is supplied — in that case
            # the scorers consult ``_fabula_anchor_override`` for their
            # reveal-set and ``syuzhet_anchor`` must stay ``None`` so
            # each scorer's internal ``is None`` branches route through
            # the fabula-aware path instead of treating syuzhet=0 as a
            # literal reader cursor at the start of the discourse.
            min_s = min(
                (e.syuzhet_index for e in engine_ws.events), default=None
            )
            syuzhet_anchor = (min_s - 1) if min_s is not None else None
        engine = _engine_structural_scores(
            engine_ws, entity_ids, syuzhet_anchor,
            surprise_local=surprise_local,
            fabula_anchor=fabula_anchor,
        )
        scores.update(engine)
    return scores


_CHAR_EMOTION_CACHE: "dict[tuple, dict[str, dict[str, float]]]" = {}


def compute_character_emotion_grid(
    ws: WorldStateV1,
    *,
    entity_ids: list[str] | None = None,
    syuzhet_anchor: int | None = None,
) -> dict[str, dict[str, float]]:
    """OCC character-felt emotion appraisals per (entity, emotion).

    Runs the six appraisal computations from
    :mod:`shadow_loom.affect_unification` (fear, joy, regret, grief,
    rage, love) for each focal entity in ``entity_ids`` and returns a
    nested mapping ``{entity_id: {emotion: scalar}}`` where each
    scalar is a single salience number in roughly ``[0, 1]`` chosen
    as the headline field of the corresponding profile (e.g.
    ``object_fear_score`` for fear, ``own_joy_score`` for joy).

    Cached on ``(id(ws), revision, entity_ids, syuzhet_anchor)`` so
    rapid cursor scrubbing collapses to a single recompute per unique
    snapshot.
    """
    eids = tuple(entity_ids) if entity_ids else ()
    key = (id(ws), _SNAPSHOT_REVISION, eids, syuzhet_anchor)
    cached = _CHAR_EMOTION_CACHE.get(key)
    if cached is not None:
        return {e: dict(d) for e, d in cached.items()}

    try:
        from shadow_loom.affect_unification import (
            AUDIENCE_ID,
            BeliefState,
            backfill_character_belief_propositions,
            compute_fear_appraisal,
            compute_grief_appraisal,
            compute_joy_appraisal,
            compute_love_appraisal,
            compute_rage_appraisal,
            compute_regret_appraisal,
            synthesise_audience_entity,
            synthesise_propositions,
        )
    except Exception:
        return {}

    if not entity_ids or not ws.events:
        return {}

    # Make sure the proposition / audience plumbing the appraisal
    # functions depend on exists. ``synthesise_*`` are idempotent.
    if not ws.propositions:
        try:
            synthesise_propositions(ws)
            backfill_character_belief_propositions(ws)
        except Exception:
            return {}
    if AUDIENCE_ID not in ws.entities:
        try:
            synthesise_audience_entity(ws)
        except Exception:
            return {}

    bs = BeliefState(world=ws)

    # Resolve fabula-time anchor: prefer the latest revealed event at
    # the syuzhet cursor; fall back to the global max so all
    # appraisals see a fully-played world.
    if syuzhet_anchor is not None:
        revealed_ft = [
            e.fabula_time for e in ws.events
            if e.syuzhet_index is not None
            and e.syuzhet_index <= syuzhet_anchor
        ]
        ft_now = max(revealed_ft) if revealed_ft else None
    else:
        ft_now = max(
            (e.fabula_time for e in ws.events), default=None,
        )
    if ft_now is None:
        return {}

    grid: dict[str, dict[str, float]] = {}
    for eid in entity_ids:
        if eid == AUDIENCE_ID or eid not in ws.entities:
            continue
        row: dict[str, float] = {}
        try:
            ap = compute_fear_appraisal(bs, eid, ft_now)
            row["fear"] = float(ap.object_fear_score or 0.0)
        except Exception:
            row["fear"] = 0.0
        try:
            ap = compute_joy_appraisal(bs, eid, ft_now)
            row["joy"] = float(ap.own_joy_score or 0.0)
        except Exception:
            row["joy"] = 0.0
        try:
            ap = compute_regret_appraisal(bs, eid, ft_now)
            row["regret"] = float(ap.agentive_regret_score or 0.0)
        except Exception:
            row["regret"] = 0.0
        try:
            ap = compute_grief_appraisal(bs, eid, ft_now)
            row["grief"] = float(ap.coupling_strength or 0.0)
        except Exception:
            row["grief"] = 0.0
        try:
            ap = compute_rage_appraisal(bs, eid, ft_now)
            row["rage"] = float(ap.blocked_concern_score or 0.0)
        except Exception:
            row["rage"] = 0.0
        try:
            ap = compute_love_appraisal(bs, eid, ft_now)
            # Sternberg triangular: take the strongest of the three
            # legs as the headline so a partner-less character with
            # zero scores stays at zero, and a full triad reads as
            # one strong love signal rather than three weak ones.
            row["love"] = float(max(
                ap.intimacy_score or 0.0,
                ap.passion_score or 0.0,
                ap.commitment_score or 0.0,
            ))
        except Exception:
            row["love"] = 0.0
        grid[eid] = row

    with _CACHE_LOCK:
        if len(_CHAR_EMOTION_CACHE) >= _AFFECT_CACHE_MAX:
            _CHAR_EMOTION_CACHE.pop(next(iter(_CHAR_EMOTION_CACHE)))
        _CHAR_EMOTION_CACHE[key] = {e: dict(d) for e, d in grid.items()}
    return grid


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
        scores = compute_affective_scores(
            ws, entity_ids=entity_ids, surprise_local=True,
        )
        return [tmin], {k: [v] for k, v in scores.items()}

    samples = max(2, int(samples))
    step = max(1, (tmax - tmin) // (samples - 1))
    times = list(range(tmin, tmax + 1, step))
    if times[-1] != tmax:
        times.append(tmax)

    series: dict[str, list[float]] = {}
    for i, t in enumerate(times):
        snap = snapshot_world_at(ws, t)
        # The engine layer (suspense / mystery / dramatic_irony /
        # surprise) is run against the full ``ws`` rather than ``snap``
        # so its set-theoretic operands (``unrevealed``, ``hidden
        # ancestors``, ``future trait state``) are non-empty \u2014 see the
        # block in ``_compute_affective_scores_uncached`` that consumes
        # ``ws_for_engine`` for the theory rationale.
        #
        # We pass ``fabula_anchor=t`` (and *not* a derived
        # ``syuzhet_anchor``) so the engine's reveal-set is anchored on
        # the fabula sweep's physical resolution semantics. Deriving a
        # syuzhet anchor as ``max(syuzhet_index)`` over snap.events
        # is incorrect when the author placed flashbacks (early-fabula
        # events at late-syuzhet indices) \u2014 a single flashback in the
        # opening fabula window would mark every later-syuzhet event
        # as "revealed" and zero out suspense / surprise across the
        # rest of the timeline.
        scores = compute_affective_scores(
            snap,
            entity_ids=entity_ids,
            syuzhet_anchor=None,
            ws_for_engine=ws,
            surprise_local=True,
            fabula_anchor=t,
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
    with _CACHE_LOCK:
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
            surprise_local=True,
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
    with _CACHE_LOCK:
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
        at_loc_id = getattr(evt, "at_location_id", None)
        at_loc_name = "—"
        if at_loc_id:
            at_loc_name = (
                ws.locations[at_loc_id].name
                if at_loc_id in ws.locations
                else at_loc_id
            )
        rows.append({
            "id": evt.id,
            "fabula_time": evt.fabula_time,
            "syuzhet_index": evt.syuzhet_index,
            "type": evt.event_type,
            "actors": ", ".join(_name_of(a) for a in evt.actor_ids) or "—",
            "targets": ", ".join(_name_of(t) for t in evt.target_ids) or "—",
            "at_location": at_loc_name,
            "at_location_id": at_loc_id,
            "description": evt.description,
            "world_id": evt.world_id,
            "superseded_by_event_id": getattr(evt, "superseded_by_event_id", None),
            "superseded": bool(getattr(evt, "superseded_by_event_id", None)),
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
    *,
    focus_id: str | None = None,
    focus_max_hops: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Causal nodes + links anchored on (fabula_time, syuzhet_index).

    Only EventNodes have natural temporal coordinates. Non-event nodes
    referenced by causal edges are projected onto the mean fabula/syuzhet
    of the events that touch them so the layout still has them somewhere
    sensible (rather than being scattered randomly).

    ``focus_id`` (when provided) restricts the rendered graph to the
    BFS neighbourhood of that node along the directed causal graph.
    Edges are coloured by ``causality_type`` modality so the
    Cartesian view shares the Sankey/force palette.

    Returns ``(nodes, links)`` ready for an ECharts ``graph`` series with
    ``coordinateSystem: 'cartesian2d'``.
    """
    if not ws.events or not ws.causal_topology:
        return [], []

    edges = list(ws.causal_topology)
    if focus_id is not None:
        out_adj: dict[str, set[str]] = {}
        in_adj: dict[str, set[str]] = {}
        for ce in edges:
            out_adj.setdefault(ce.source_id, set()).add(ce.target_id)
            in_adj.setdefault(ce.target_id, set()).add(ce.source_id)
        keep: set[str] = {focus_id}
        frontier: set[str] = {focus_id}
        for _ in range(max(1, int(focus_max_hops))):
            nxt: set[str] = set()
            for nid in frontier:
                nxt.update(out_adj.get(nid, ()))
                nxt.update(in_adj.get(nid, ()))
            nxt -= keep
            if not nxt:
                break
            keep.update(nxt)
            frontier = nxt
        edges = [
            ce for ce in edges
            if ce.source_id in keep and ce.target_id in keep
        ]
        if not edges:
            return [], []

    # Index events for quick lookup
    evt_by_id = {e.id: e for e in ws.events}
    evt_ids = set(evt_by_id.keys())

    # Compute mean coords for every non-event referenced by causal edges
    coord_acc: dict[str, list[tuple[int, int]]] = {}
    for ce in edges:
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
            node = {
                "id": nid,
                "name": nid,
                "value": [evt.fabula_time, evt.syuzhet_index],
                "symbol": "triangle",
                "symbolSize": 16,
                "itemStyle": {"color": NODE_COLORS["EventNode"]},
                "tooltip": {
                    "formatter": (
                        f"<b>{nid}</b> [{evt.event_type}]<br/>"
                        f"t={evt.fabula_time}, s={evt.syuzhet_index}<br/>"
                        f"{(evt.description or '')[:80]}"
                    )
                },
            }
            if focus_id is not None and nid == focus_id:
                node["itemStyle"] = {
                    **node["itemStyle"],
                    "borderColor": "#FFD700",
                    "borderWidth": 4,
                }
                node["symbolSize"] = 24
            nodes.append(node)
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
        node = {
            "id": nid,
            "name": label,
            "value": [round(x, 2), round(y, 2)],
            "symbol": sym,
            "symbolSize": size,
            "itemStyle": {"color": color},
        }
        if focus_id is not None and nid == focus_id:
            node["itemStyle"] = {
                **node["itemStyle"],
                "borderColor": "#FFD700",
                "borderWidth": 4,
            }
            node["symbolSize"] = max(size + 6, 24)
        nodes.append(node)

    links: list[dict] = []
    for ce in edges:
        _ensure(ce.source_id)
        _ensure(ce.target_id)
        w = min(6, max(1, ce.causal_force / 1.5))
        modality = _modality_key(ce)
        color = MODALITY_COLORS.get(modality, EDGE_COLORS["causal"])
        is_world_to_world = modality == "world_to_world"
        ls = {
            "width": max(w, 2.0) if is_world_to_world else w,
            "color": color,
            "opacity": 0.7,
            "curveness": 0.15,
        }
        if is_world_to_world:
            ls["type"] = "dashed"
        links.append({
            "source": ce.source_id,
            "target": ce.target_id,
            "_sl_modality": modality,
            "lineStyle": ls,
            "tooltip": {"formatter": (
                f"<b>{MODALITY_LABELS.get(modality, ce.causality_type)}</b>"
                f"<br/>force={ce.causal_force}"
            )},
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
                    f"<b>{html.escape(str(evt.id))}</b><br/>"
                    f"t={evt.fabula_time}<br/>"
                    f"{html.escape((evt.description or '')[:60])}"
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
        with _CACHE_LOCK:
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
    with _CACHE_LOCK:
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
    with _CACHE_LOCK:
        _PHYSICS_TRAJECTORY_CACHE.clear()


# =====================================================================
# Social-layer helpers (Beliefs / Concerns / Propositions / Relationships)
# =====================================================================

def list_concern_holders(ws: WorldStateV1) -> list[tuple[str, str, int]]:
    """Return ``(entity_id, name, concern_count)`` for entities with concerns.

    Sorted by concern_count desc so the most loaded characters come first.
    """
    rows: list[tuple[str, str, int]] = []
    for eid, ent in ws.entities.items():
        if ent.concerns:
            rows.append((eid, ent.name, len(ent.concerns)))
    rows.sort(key=lambda r: (-r[2], r[1]))
    return rows


def ws_to_proposition_rows(
    ws: WorldStateV1,
    fabula_t: int | None = None,
) -> list[dict]:
    """Per-proposition rows, optionally replayed at ``fabula_t``.

    When ``fabula_t`` is None the row carries the proposition's
    initial framing fields. Otherwise
    :func:`shadow_loom.models.reconstruct_proposition_at` is used so
    ``stakes`` / ``audience_default_prior`` / ``description`` /
    ``truth_at`` reflect the requested moment.
    """
    from shadow_loom.models import reconstruct_proposition_at

    rows: list[dict] = []
    for prop in (ws.propositions or []):
        if fabula_t is not None:
            snap = reconstruct_proposition_at(prop, fabula_t)
            stakes = snap["stakes"]
            prior = snap["audience_default_prior"]
            desc = snap["description"]
            truth = snap["truth_at"]
        else:
            stakes = prop.stakes
            prior = prop.audience_default_prior
            desc = prop.description
            truth = None
            for t in sorted(prop.truth_at_fabula.keys()):
                truth = prop.truth_at_fabula[t]
        # How many entities hold a concern about this proposition?
        concern_count = sum(
            1 for ent in ws.entities.values()
            for c in ent.concerns if c.proposition_id == prop.proposition_id
        )
        # How many beliefs reference the proposition?
        belief_count = sum(
            1 for ent in ws.entities.values()
            for b in ent.beliefs
            if getattr(b, "proposition_id", None) == prop.proposition_id
        )
        rows.append({
            "id": prop.proposition_id,
            "kind": prop.kind,
            "description": desc,
            "referent_ids": ", ".join(prop.referent_ids or []),
            "stakes": round(float(stakes), 3),
            "audience_default_prior": round(float(prior), 3),
            "truth_at": (
                "true" if truth is True
                else "false" if truth is False
                else "—"
            ),
            "concerns_referencing": concern_count,
            "beliefs_referencing": belief_count,
            "world_id": prop.world_id,
        })
    return rows


def ws_to_concern_rows(
    ws: WorldStateV1,
    fabula_t: int | None = None,
    *,
    only_active: bool = False,
) -> list[dict]:
    """Per-(entity, concern) rows, optionally replayed at ``fabula_t``."""
    from shadow_loom.models import reconstruct_concern_at

    rows: list[dict] = []
    for ent in ws.entities.values():
        for concern in ent.concerns:
            if fabula_t is not None:
                snap = reconstruct_concern_at(concern, fabula_t)
                salience = snap["salience"]
                polarity = snap["polarity"]
                kind = snap.get("kind") or "—"
                active = snap["active"]
            else:
                salience = concern.salience
                polarity = concern.polarity
                kind = concern.kind or "—"
                active = True
                if concern.activation_fabula_window:
                    active = False  # unknown without a cursor
            if only_active and not active:
                continue
            # Look up the proposition description for context.
            prop = next(
                (p for p in (ws.propositions or [])
                 if p.proposition_id == concern.proposition_id),
                None,
            )
            prop_desc = prop.description if prop else concern.proposition_id
            ccids = list(getattr(concern, "counter_concern_ids", []) or [])
            triggered_by = (
                snap.get("triggered_by") if fabula_t is not None else None
            )
            window = getattr(concern, "activation_fabula_window", None)
            rows.append({
                "entity_id": ent.id,
                "entity": ent.name,
                "concern_id": concern.concern_id,
                "proposition_id": concern.proposition_id,
                "proposition_desc": prop_desc,
                "polarity": polarity,
                "salience": round(float(salience), 3),
                "kind": kind,
                "active": "✓" if active else "—",
                "_active_bool": active,
                "counter_concern_ids": ", ".join(ccids),
                "activation_window": (
                    f"{window[0]}–{window[1]}" if window else ""
                ),
                "triggered_by": triggered_by or "",
                "world_id": concern.world_id,
            })
    rows.sort(key=lambda r: (-r["salience"], r["entity"]))
    return rows


def ws_to_entity_concern_rows(
    ws: WorldStateV1,
    entity_id: str,
    fabula_t: int | None = None,
) -> list[dict]:
    """Per-concern rows for one entity (for the per-character card)."""
    from shadow_loom.models import reconstruct_concern_at

    ent = ws.entities.get(entity_id)
    if ent is None or not ent.concerns:
        return []
    rows: list[dict] = []
    for concern in ent.concerns:
        if fabula_t is not None:
            snap = reconstruct_concern_at(concern, fabula_t)
            salience = snap["salience"]
            polarity = snap["polarity"]
            kind = snap.get("kind") or None
            active = snap["active"]
        else:
            salience = concern.salience
            polarity = concern.polarity
            kind = concern.kind
            active = True
        prop = next(
            (p for p in (ws.propositions or [])
             if p.proposition_id == concern.proposition_id),
            None,
        )
        rows.append({
            "concern_id": concern.concern_id,
            "proposition_id": concern.proposition_id,
            "proposition_desc": prop.description if prop else concern.proposition_id,
            "polarity": polarity,
            "salience": round(float(salience), 3),
            "kind": kind,
            "active": active,
        })
    rows.sort(key=lambda r: (-r["salience"], r["concern_id"]))
    return rows


def ws_to_social_layer_graph(
    ws: WorldStateV1,
    *,
    include_relationships: bool = True,
    include_beliefs: bool = True,
    include_concerns: bool = True,
    include_propositions: bool = True,
    fabula_t: int | None = None,
    event_t: int | None = None,
    ego_id: str | None = None,
    ego_max_hops: int = 1,
    pov_id: str | None = None,
    intermental_ids: list[str] | None = None,
    intermental_threshold: float = 0.4,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Combined social-layer graph: entities, propositions, concerns,
    beliefs (as edges), relationships (as edges).

    Node categories:
      0 = Entity (character)
      1 = Proposition
      2 = Concern (one per ``(entity, concern_id)`` pair)

    Edges:
      • Entity → Entity   (relationship; coloured by affinity)
      • Entity → Concern  (holds; weighted by salience, coloured by polarity)
      • Concern → Proposition (about)
      • Entity → Proposition  (belief; coloured by confidence)

    All series respect ``fabula_t``: relationships are time-sliced via
    ``last_updated_fabula <= t``; concerns honour their activation
    window and replayed salience/polarity; beliefs honour
    ``established_at_fabula <= t``.
    """
    from shadow_loom.models import reconstruct_concern_at

    nodes: list[dict] = []
    links: list[dict] = []
    # ECharts categories interact unreliably with per-node ``symbol``
    # and ``itemStyle`` overrides through NiceGUI's echart wrapper:
    # any colour set on a category silently masks per-node colour for
    # nodes assigned to that category, and any *gap* in a category's
    # ``itemStyle`` causes adjacent categories' nodes to drop from
    # the render. We don't use ECharts' built-in legend here (a
    # chip-strip legend above the chart documents the encoding), so
    # categories only matter as a fallback. Collapse to a single
    # neutral category and let per-node ``symbol`` + ``itemStyle``
    # carry every visual distinction (Character circle, Proposition
    # diamond, Desire green-triangle, Fear red-triangle, World Trait
    # roundRect) -- that path renders reliably.
    cats = [{"name": "Social"}]

    # --- Entity nodes -------------------------------------------------
    for eid, ent in ws.entities.items():
        nodes.append({
            "id": eid,
            "name": ent.name,
            "category": 0,
            "symbol": "circle",
            "symbolSize": 36,
            "itemStyle": {"color": NODE_COLORS["Entity"]},
            "tooltip": {"formatter": (
                f"<b>{html.escape(str(ent.name))}</b><br/>Status: {html.escape(str(ent.status))}<br/>"
                f"{len(ent.beliefs)} beliefs, {len(ent.concerns)} concerns"
            )},
            "_sl_node_type": "Entity",
        })

    # --- Proposition nodes -------------------------------------------
    if include_propositions:
        from shadow_loom.models import reconstruct_proposition_at
        for prop in (ws.propositions or []):
            stakes = prop.stakes
            desc = prop.description
            prior = float(getattr(prop, "audience_default_prior", 0.5))
            if fabula_t is not None:
                snap = reconstruct_proposition_at(prop, fabula_t)
                stakes = snap["stakes"]
                desc = snap["description"]
                prior = float(snap.get("audience_default_prior", prior))
            # Diamond, sized by stakes.
            size = 18 + 22 * max(0.0, min(1.0, float(stakes)))
            # Audience-prior "surprise potential" is encoded as
            # *border thickness* on a neutral grey border. Using
            # colour here (red/amber/green) would collide with three
            # other red/green encodings already on the chart
            # (affinity, concern polarity, belief confidence). Width
            # alone keeps the channel readable: a thin border = the
            # audience expects this proposition (prior ≈ 0.5);
            # a thick border = the audience holds a strong prior
            # (either way) so the proposition will land as either
            # confirmation or surprise.
            border_w = 1.5 + 4.0 * abs(prior - 0.5)
            nodes.append({
                "id": prop.proposition_id,
                "name": desc[:40],
                "category": 0,
                "symbol": "diamond",
                "symbolSize": size,
                "itemStyle": {
                    "color": "#8a5cf0",  # iris
                    "borderColor": "#475569",  # slate-600 — neutral
                    "borderWidth": border_w,
                },
                "tooltip": {"formatter": (
                    f"<b>{html.escape(str(desc))}</b><br/>kind: {html.escape(str(prop.kind))}<br/>"
                    f"stakes: {float(stakes):.2f}<br/>"
                    f"audience prior: {prior:.2f}"
                )},
                "_sl_node_type": "Proposition",
            })

    # --- Concern nodes -----------------------------------------------
    # Concerns are dual-encoded so the desire/fear-of-proposition
    # relationship reads at the same visual weight as belief:
    #   1. A small triangle node anchored to the entity (green =
    #      desire, red = fear), sized by salience. This gives the
    #      "fan of concerns radiating from a character" reading and
    #      groups concerns visually by holder.
    #   2. A *direct* entity \u2192 proposition edge (dashed,
    #      arrowed, green/red, width by salience). This is the
    #      counterpart to the belief edge and makes the
    #      \"who wants/fears what\" reading immediate \u2014 without
    #      it the desire/fear relationship was two hops away
    #      (entity \u2192 triangle \u2192 prop) while beliefs were one,
    #      so the chart looked like \"only beliefs touch propositions\".
    # Concerns are rendered as **direct entity \u2192 proposition
    # edges** rather than as a separate triangle node + two edges.
    # The earlier triangle-node design produced visually busy
    # graphs where the desire/fear relationship was two hops away
    # from the proposition (entity \u2192 triangle \u2192 prop) while
    # belief was one hop (entity \u2192 prop), so concerns read as
    # secondary structure. Direct dashed green/red arrows put
    # desires/fears on the same visual plane as beliefs (solid
    # amber arrow) and make "who wants/fears what" immediate.
    if include_concerns and include_propositions:
        for ent in ws.entities.values():
            for concern in ent.concerns:
                if fabula_t is not None:
                    snap = reconstruct_concern_at(concern, fabula_t)
                    salience = snap["salience"]
                    polarity = snap["polarity"]
                    if not snap["active"]:
                        continue  # skip inactive concerns at this t
                else:
                    salience = concern.salience
                    polarity = concern.polarity
                if not any(
                    p.proposition_id == concern.proposition_id
                    for p in (ws.propositions or [])
                ):
                    continue
                color = "#16a34a" if polarity == "desire" else "#dc2626"
                links.append({
                    "source": ent.id,
                    "target": concern.proposition_id,
                    "_sl_kind": (
                        "desire" if polarity == "desire" else "fear"
                    ),
                    "_sl_holder": ent.id,
                    "_sl_pid": concern.proposition_id,
                    "_sl_concern_id": concern.concern_id,
                    "_sl_salience": float(salience),
                    "symbol": ["none", "arrow"],
                    "symbolSize": [4, 9],
                    "lineStyle": {
                        "color": color,
                        "width": max(1.5, float(salience) * 4.0),
                        "type": "dashed",
                        "opacity": 0.9,
                        "curveness": -0.18,
                    },
                    "tooltip": {"formatter": (
                        f"<b>{html.escape(str(ent.name))}</b> "
                        f"{'desires' if polarity == 'desire' else 'fears'}"
                        f"<br/>\u2192 {html.escape(str(concern.proposition_id))}"
                        f"<br/>salience: {float(salience):.2f}"
                        f"<br/>kind: {html.escape(str(concern.kind or '\u2014'))}"
                    )},
                })

    # --- Belief edges (Entity → Proposition) -------------------------
    if include_beliefs and include_propositions:
        for ent in ws.entities.values():
            for b in ent.beliefs:
                pid = getattr(b, "proposition_id", None)
                if not pid:
                    continue
                if fabula_t is not None and getattr(
                    b, "established_at_fabula", 0
                ) > fabula_t:
                    continue
                if not any(
                    p.proposition_id == pid
                    for p in (ws.propositions or [])
                ):
                    continue
                conf = float(b.confidence)
                # Belief confidence uses an *amber* ramp (low = pale,
                # high = deep) so it stays distinguishable from
                # affinity edges (green/red) and concern edges
                # (green/red, dashed). Amber is also colour-blind
                # safe against red/green. Arrow points from holder
                # to proposition so direction is unambiguous (this
                # is an epistemic edge, not a symmetric tie).
                if conf < 0.34:
                    bcolor = "#fde68a"   # amber-200 — weak/uncertain
                elif conf < 0.67:
                    bcolor = "#f59e0b"   # amber-500 — moderate
                else:
                    bcolor = "#b45309"   # amber-700 — strong/sure
                links.append({
                    "source": ent.id,
                    "target": pid,
                    "_sl_kind": "belief",
                    "_sl_holder": ent.id,
                    "_sl_pid": pid,
                    "_sl_state": b.perceived_state,
                    "_sl_conf": conf,
                    "symbol": ["none", "arrow"],
                    "symbolSize": [4, 7],
                    "lineStyle": {
                        "color": bcolor,
                        "width": max(1.0, conf * 3.5),
                        "type": "solid",
                        "opacity": 0.85,
                        # Explicit opposite curveness from desire/fear
                        # edges (-0.18) so the entity \u2192 prop belief
                        # arrow and the entity \u2192 prop desire/fear
                        # arrow sit on visibly separate arcs even when
                        # both exist. Relying on ECharts'
                        # ``autoCurveness`` here is unreliable: it only
                        # auto-assigns to links *without* explicit
                        # curveness, so the desire/fear edge (which
                        # sets curveness) keeps its arc but belief is
                        # left flat at curveness=0 and one edge type
                        # ends up visually drawn on top of the other,
                        # making desire/fear arrows disappear.
                        "curveness": 0.18,
                    },
                    "tooltip": {"formatter": (
                        f"{html.escape(str(ent.name))} believes "
                        f"({conf:.2f}): {html.escape(str(b.perceived_state))}"
                    )},
                })

    # --- Relationship edges (Entity ↔ Entity) ------------------------
    if include_relationships:
        for rel in ws.social_topology:
            if fabula_t is not None and rel.last_updated_fabula > fabula_t:
                continue
            aff = float(rel.affinity)
            color = (
                "#16a34a" if aff > 0
                else "#dc2626" if aff < 0
                else "#94a3b8"
            )
            # Curved + no arrow keeps relationship edges visually
            # distinct from belief edges (straight + arrow) and
            # concern edges (dashed + arrow), so the three edge
            # types are unambiguous even when they overlap on the
            # same entity.
            links.append({
                "source": rel.source_entity_id,
                "target": rel.target_entity_id,
                "lineStyle": {
                    "color": color,
                    "width": max(1.0, abs(aff) * 4.0),
                    "type": "solid",
                    "opacity": 0.7,
                    "curveness": 0.25,
                },
                "tooltip": {"formatter": (
                    f"affinity={aff:+.2f}<br/>fear={float(rel.fear):.2f}"
                    f"<br/>power={float(rel.power_dynamic):+.2f}"
                )},
            })

    # --- POV filter (focalisation) -----------------------------------
    # Restrict the world to *one* character's epistemic horizon: only
    # entities they hold relationships toward (or vice-versa) plus
    # propositions they have beliefs/concerns about. Other entities
    # are dimmed but kept so the structure of "what they don't see"
    # is still visible (greyed out).
    if pov_id is not None and pov_id in ws.entities:
        pov_ent = ws.entities[pov_id]
        known_entities: set[str] = {pov_id}
        for rel in ws.social_topology:
            if pov_id in (rel.source_entity_id, rel.target_entity_id):
                known_entities.add(rel.source_entity_id)
                known_entities.add(rel.target_entity_id)
        known_props: set[str] = set()
        for b in pov_ent.beliefs:
            pid = getattr(b, "proposition_id", None)
            if pid:
                known_props.add(pid)
            if b.target_id in ws.entities:
                known_entities.add(b.target_id)
        for c in pov_ent.concerns:
            known_props.add(c.proposition_id)
        # Dim unknown nodes; keep them in-graph so layout is stable.
        for n in nodes:
            nt = n.get("_sl_node_type")
            nid = n["id"]
            visible = (
                (nt == "Entity" and nid in known_entities) or
                (nt == "Proposition" and nid in known_props)
            )
            if not visible:
                style = dict(n.get("itemStyle") or {})
                style["opacity"] = 0.15
                n["itemStyle"] = style
                n["label"] = {"show": False}
        # Drop edges touching unknown propositions; dim edges between
        # entities outside POV.
        kept_links: list[dict] = []
        for lnk in links:
            kind = lnk.get("_sl_kind")
            s, t = lnk["source"], lnk["target"]
            # Belief edges: keep only those held by POV.
            if kind == "belief":
                if lnk.get("_sl_holder") != pov_id:
                    continue
            # Desire/fear edges: keep only those held by POV.
            if kind in ("desire", "fear"):
                if lnk.get("_sl_holder") != pov_id:
                    continue
            kept_links.append(lnk)
        links = kept_links
        # Highlight POV node.
        for n in nodes:
            if n["id"] == pov_id:
                style = dict(n.get("itemStyle") or {})
                style["borderColor"] = "#0ea5e9"
                style["borderWidth"] = 4
                n["itemStyle"] = style

    # --- Intermental overlay (Palmer two-mind) -----------------------
    # When two or more egos are selected, recompute belief edges as
    # *shared* (thick teal, both-believe-same-state above threshold)
    # vs *divergent* (red, both-believe-different-state above
    # threshold). Single-ego beliefs are dimmed.
    if intermental_ids and len(intermental_ids) >= 2:
        ids = [i for i in intermental_ids if i in ws.entities]
        if len(ids) >= 2:
            # Build per-(holder, pid) state index.
            state_idx: dict[tuple[str, str], tuple[str, float]] = {}
            for ent in ws.entities.values():
                if ent.id not in ids:
                    continue
                for b in ent.beliefs:
                    pid = getattr(b, "proposition_id", None)
                    if not pid:
                        continue
                    if fabula_t is not None and getattr(
                        b, "established_at_fabula", 0
                    ) > fabula_t:
                        continue
                    if float(b.confidence) < intermental_threshold:
                        continue
                    state_idx[(ent.id, pid)] = (
                        b.perceived_state, float(b.confidence)
                    )
            # Group by pid.
            by_pid: dict[str, list[tuple[str, str, float]]] = {}
            for (eid, pid), (state, conf) in state_idx.items():
                by_pid.setdefault(pid, []).append((eid, state, conf))
            # Drop original belief edges (we'll re-add a styled overlay).
            kept = []
            for lnk in links:
                if lnk.get("_sl_kind") == "belief":
                    # dim it
                    ls = dict(lnk.get("lineStyle") or {})
                    ls["opacity"] = 0.15
                    lnk["lineStyle"] = ls
                kept.append(lnk)
            links = kept
            # Add overlay edges between each pair of selected egos.
            for pid, holders in by_pid.items():
                if len(holders) < 2:
                    continue
                for i in range(len(holders)):
                    for j in range(i + 1, len(holders)):
                        e1, s1, c1 = holders[i]
                        e2, s2, c2 = holders[j]
                        same = (s1 or "").strip() == (s2 or "").strip()
                        color = "#0d9488" if same else "#dc2626"
                        label = ("shared belief" if same
                                 else "divergent belief")
                        w = 2.5 + 2.0 * min(c1, c2)
                        # Triangle e1 – pid – e2 reads as a "two-mind
                        # arc" through the proposition.
                        for src in (e1, e2):
                            links.append({
                                "source": src,
                                "target": pid,
                                "_sl_kind": "intermental",
                                "lineStyle": {
                                    "color": color,
                                    "width": w,
                                    "type": "solid",
                                    "opacity": 0.85,
                                    "curveness": 0.25,
                                },
                                "tooltip": {"formatter": (
                                    f"{label}<br/>"
                                    f"{ws.entities[e1].name}: "
                                    f"{s1} ({c1:.2f})<br/>"
                                    f"{ws.entities[e2].name}: "
                                    f"{s2} ({c2:.2f})"
                                )},
                            })
            # Highlight the selected egos.
            for n in nodes:
                if n["id"] in ids:
                    style = dict(n.get("itemStyle") or {})
                    style["borderColor"] = "#0d9488"
                    style["borderWidth"] = 3
                    n["itemStyle"] = style

    # --- World-trait nodes ------------------------------------------------
    # For every GlobalTrait whose proposition_id resolves to a proposition
    # already in the graph, add a WORLD_ node and an edge from it to that
    # proposition.  This lets Network / Ego / Intermental views show the
    # named-latent forces that characters' beliefs and concerns are grounded
    # in, rather than leaving propositions visually floating without context.
    if include_propositions and (ws.world_traits or {}) and nodes:
        existing_prop_ids = {n["id"] for n in nodes if n.get("_sl_node_type") == "Proposition"}
        for wid, trait in (ws.world_traits or {}).items():
            pid = getattr(trait, "proposition_id", None)
            if not pid or pid not in existing_prop_ids:
                continue
            # Only add the WORLD_ node once.
            if any(n["id"] == wid for n in nodes):
                continue
            nodes.append({
                "id": wid,
                "name": trait.name,
                "category": 0,
                "symbol": "roundRect",
                "symbolSize": 26,
                "itemStyle": {"color": "#0ea5e9"},
                "tooltip": {"formatter": (
                    f"<b>{trait.name}</b><br/>World trait: {wid}<br/>"
                    f"{(trait.description or '')[:80]}"
                )},
                "_sl_node_type": "WorldTrait",
            })
            links.append({
                "source": wid,
                "target": pid,
                "symbol": ["none", "arrow"],
                "symbolSize": [4, 7],
                "lineStyle": {
                    "color": "#0ea5e9",
                    "width": 2.0,
                    "type": "dotted",
                    "opacity": 0.75,
                },
                "tooltip": {"formatter": f"World trait → {trait.name}"},
            })
        # (Categories collapsed to a single neutral entry above; per-node
        # symbol/itemStyle carries the World Trait styling.)

    # --- Current-event entity filter -------------------------------------
    # When event_t is set (pinned cursor, not live) restrict entity nodes
    # to those directly involved in events at that exact fabula time, plus
    # their immediate social-relationship neighbours.  Live mode
    # (event_t is None) leaves all entities visible.
    if event_t is not None and ws.events:
        _evts_now = [e for e in ws.events if e.fabula_time == event_t]
        if not _evts_now:
            _all_times = sorted({e.fabula_time for e in ws.events})
            if _all_times:
                _nearest = min(_all_times, key=lambda _t: abs(_t - event_t))
                _evts_now = [e for e in ws.events if e.fabula_time == _nearest]
        if _evts_now:
            _active: set[str] = set()
            for _e in _evts_now:
                _active.update(_e.actor_ids or [])
                _active.update(_e.target_ids or [])
                if getattr(_e, "speaker_id", None):
                    _active.add(_e.speaker_id)
                _active.update(_e.addressee_ids or [])
            _active &= ws.entities.keys()
            # Always retain focus entities so per-mode filters still
            # have a valid anchor node.
            if ego_id:
                _active.add(ego_id)
            if pov_id:
                _active.add(pov_id)
            if intermental_ids:
                _active.update(intermental_ids)
            _active &= ws.entities.keys()
            if _active:
                # Pull in direct social-relationship neighbours.
                for _rel in ws.social_topology:
                    if _rel.source_entity_id in _active:
                        _active.add(_rel.target_entity_id)
                    if _rel.target_entity_id in _active:
                        _active.add(_rel.source_entity_id)
                _active &= ws.entities.keys()
                nodes = [
                    n for n in nodes
                    if n.get("_sl_node_type") != "Entity"
                    or n["id"] in _active
                ]
                _ce_node_ids = {n["id"] for n in nodes}
                links = [
                    l for l in links
                    if l["source"] in _ce_node_ids
                    and l["target"] in _ce_node_ids
                ]

    # --- Remove isolated nodes -------------------------------------------
    # Nodes with no edges are visual noise: drop them *before* the ego /
    # POV / intermental filters (which do their own structural pruning).
    # Note: ego_id / pov_id / intermental filters run after this block
    # so they see only the connected subgraph.
    if not (ego_id is not None or pov_id is not None or intermental_ids):
        connected_ids: set[str] = set()
        for lnk in links:
            connected_ids.add(lnk["source"])
            connected_ids.add(lnk["target"])
        nodes = [n for n in nodes if n["id"] in connected_ids]

    # --- Ego filter ----------------------------------------------------
    # Restrict to the BFS neighbourhood of ``ego_id`` (over the
    # social-layer link graph just built). Concern nodes are pulled in
    # via their owning entity, and proposition nodes via beliefs /
    # concerns that touch the ego.
    if ego_id is not None and any(n["id"] == ego_id for n in nodes):
        adj: dict[str, set[str]] = {}
        for lnk in links:
            s, t = lnk["source"], lnk["target"]
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)
        visited: set[str] = {ego_id}
        frontier: set[str] = {ego_id}
        for _ in range(max(1, int(ego_max_hops))):
            nxt: set[str] = set()
            for nid in frontier:
                nxt |= adj.get(nid, set())
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        nodes = [n for n in nodes if n["id"] in visited]
        links = [
            l for l in links
            if l["source"] in visited and l["target"] in visited
        ]
        # Highlight the ego node.
        for n in nodes:
            if n["id"] == ego_id:
                n["itemStyle"] = {
                    **n.get("itemStyle", {}),
                    "borderColor": "#FFD700",
                    "borderWidth": 3,
                }
                n["symbolSize"] = n.get("symbolSize", 36) * 1.3

    return nodes, links, cats


def entity_trait_trajectory(
    ws: WorldStateV1,
    entity_id: str,
    *,
    trait_names: list[str] | None = None,
    max_traits: int = 6,
) -> tuple[list[int], dict[str, list[float]]]:
    """Return ``(times, {trait_name: [values…]})`` for one entity.

    Replays ``reconstruct_entity_with_causal`` at every distinct
    ``fabula_time`` in the world's event list. ``trait_names`` filters
    which traits to plot; defaults to the top ``max_traits`` by
    presence in the entity's baseline.
    """
    ent = ws.entities.get(entity_id)
    if ent is None:
        return [], {}
    times = sorted({evt.fabula_time for evt in ws.events})
    if not times:
        # Fall back to baseline-only.
        baseline = list(ent.traits.keys())[:max_traits]
        return [0], {tn: [float(ent.traits[tn].value)] for tn in baseline}

    if trait_names is None:
        trait_names = list(ent.traits.keys())[:max_traits]

    series: dict[str, list[float]] = {tn: [] for tn in trait_names}
    for t in times:
        snap = reconstruct_entity_with_causal(ws, entity_id, t)
        traits = snap.get("traits", {}) or {}
        for tn in trait_names:
            tv = traits.get(tn)
            if tv is None:
                series[tn].append(
                    float(ent.traits[tn].value) if tn in ent.traits else 0.5
                )
            else:
                val = tv["value"] if isinstance(tv, dict) else float(tv)
                series[tn].append(round(float(val), 3))
    return times, series
