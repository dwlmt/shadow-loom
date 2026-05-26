# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared event / channel projections consumed by both the UI and MCP layers.

The MCP server's ``inspect()`` tool and the UI's Explorer inspector both need to
serialise an :class:`EventNode` (including the post-refactor utterance payload)
and a :class:`Channel` into a UI-agnostic dict. Historically each layer rolled
its own projection, which caused the inspectors to drift after the channels &
beliefs refactor (utterance fields disappeared from MCP; the UI had no channel
inspector at all).

These helpers are pure functions with no UI / MCP imports so they can be used
on either side of the boundary and unit-tested in isolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shadow_loom.models import (
    Channel,
    EventNode,
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
)


def reconstruct_entity_at_causal(
    ws: WorldStateV1,
    entity_id: str,
    fabula_time: int,
) -> Dict[str, Any]:
    """Reconstruct an entity at ``fabula_time`` overlaying causal mutations.

    1. Seed traits from ``Entity.traits`` (pre-story baseline).
    2. Replay every ``mutation`` / ``mutation_social`` :class:`CausalEdge`
       whose ``target_id`` is this entity, ``trait_target`` is set, and
       ``fabula_time <= fabula_time`` — accumulating signed
       ``trait_delta`` per axis (clamped to ``[0, 1]`` because
       ``TraitVector.value`` is non-negative on the schema).
    3. Apply the snapshot replay on top — authored
       :class:`EntityStateSnapshot` values override the running causal
       values at their tick.

    Mirrors ``reconstruct_entity_at`` shape so callers can swap.
    """
    ent = ws.entities.get(entity_id)
    if ent is None:
        return {"traits": {}, "beliefs": [], "status": "healthy", "location_id": ""}

    running: Dict[str, Dict[str, Any]] = {
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
            if ce.causality_type in ("mutation", "mutation_social")
            and ce.target_id == entity_id
            and ce.trait_target
            and ce.trait_delta is not None
            and ce.fabula_time <= fabula_time
            # Branch safety: only apply mutations whose ``world_id``
            # matches the holder entity's branch. A shadow-tagged
            # mutation edge accidentally targeting a factual entity
            # (or vice versa) would otherwise drift the canonical
            # trait value silently.
            and (getattr(ce, "world_id", "factual") or "factual")
                == (getattr(ent, "world_id", "factual") or "factual")
        ),
        key=lambda c: c.fabula_time,
    )
    for ce in mutations:
        cur = running.get(
            ce.trait_target,
            {"value": 0.0, "inertia": 0.5, "evidence_strength": "moderate"},
        )
        # Entity-trait values are non-negative on the schema
        # (``TraitVector.value`` is ``ge=0``); the prior ``[-1, 1]``
        # clamp produced incoherent negative readings in the MCP
        # ``inspect()`` payload (audit 2026-05-26).
        new_val = max(0.0, min(1.0, cur["value"] + (ce.trait_delta or 0.0)))
        running[ce.trait_target] = {
            "value": new_val,
            "inertia": cur["inertia"],
            "evidence_strength": cur.get("evidence_strength", "moderate"),
        }

    snap = reconstruct_entity_at(ent, fabula_time)
    # Authored ``EntityStateSnapshot`` entries override the running
    # causal accumulation at their tick — but only for trait names a
    # snapshot actually wrote. ``reconstruct_entity_at`` seeds its
    # ``traits`` dict from ``ent.traits`` so unchanged baseline traits
    # appear in ``snap['traits']`` too; the previous unconditional
    # copy back into ``running`` silently discarded every causal
    # mutation accumulated above for any pre-seeded trait (audit
    # 2026-05-26). The world-trait variant below already had this
    # discrimination — this brings the entity variant into line.
    holder_world = getattr(ent, "world_id", "factual") or "factual"
    snapshot_touched_traits: set[str] = set()
    for s in ent.state_timeline:
        if s.fabula_time > fabula_time:
            continue
        snap_world = getattr(s, "world_id", holder_world) or holder_world
        if snap_world != holder_world:
            continue
        snapshot_touched_traits.update(s.traits.keys())
    for tn, tv in snap.get("traits", {}).items():
        if tn not in snapshot_touched_traits:
            continue
        running[tn] = (
            tv if isinstance(tv, dict)
            else {"value": float(tv), "inertia": 0.5, "evidence_strength": "moderate"}
        )

    return {
        "traits": running,
        "beliefs": snap.get("beliefs", []),
        "status": snap.get("status", ent.status),
        "location_id": snap.get("location_id", ent.location_id),
    }


def reconstruct_world_trait_at_causal(
    ws: WorldStateV1,
    world_id: str,
    fabula_time: int,
) -> Dict[str, Any]:
    """Reconstruct a :class:`GlobalTrait` at ``fabula_time`` overlaying
    causal mutations on top of the snapshot replay.

    Mirrors ``reconstruct_world_trait_at`` shape.
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
            if ce.causality_type in ("mutation", "mutation_social")
            and ce.target_id == world_id
            and ce.trait_delta is not None
            and ce.fabula_time <= fabula_time
            # Branch safety: mirror of the entity-side filter \u2014
            # a shadow-tagged mutation must not drift the canonical
            # global trait, and vice versa.
            and (getattr(ce, "world_id", "factual") or "factual")
                == (getattr(wt, "world_id", "factual") or "factual")
        ),
        key=lambda c: c.fabula_time,
    )
    for ce in mutations:
        value = max(0.0, min(1.0, value + (ce.trait_delta or 0.0)))

    snap = reconstruct_world_trait_at(wt, fabula_time)
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


def _resolve_names(ws: WorldStateV1, ids: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "id": i,
            "name": ws.entities[i].name if i in ws.entities else i,
        }
        for i in ids
    ]


def project_event(ws: WorldStateV1, evt: EventNode) -> Dict[str, Any]:
    """Project an :class:`EventNode` into a serialisable dict.

    Always emits the structural fields. When ``evt.event_type == "utterance"``
    (or when any of the utterance payload fields are populated for a
    revelation) also emits ``content``, ``via_channel_id``, ``speaker_id``,
    ``addressee_ids`` and ``truth_value`` so callers can render the
    epistemic payload without re-checking the model.
    """
    causes = [
        {
            "source": ce.source_id,
            "mechanism": ce.mechanism,
            "force": ce.causal_force,
            "type": ce.causality_type,
        }
        for ce in ws.causal_topology
        if ce.target_id == evt.id
    ]
    effects = [
        {
            "target": ce.target_id,
            "mechanism": ce.mechanism,
            "force": ce.causal_force,
            "type": ce.causality_type,
        }
        for ce in ws.causal_topology
        if ce.source_id == evt.id
    ]

    out: Dict[str, Any] = {
        "id": evt.id,
        "type": "EventNode",
        "event_type": evt.event_type,
        "fabula_time": evt.fabula_time,
        "syuzhet_index": evt.syuzhet_index,
        "description": evt.description,
        "actors": _resolve_names(ws, evt.actor_ids),
        "targets": _resolve_names(ws, evt.target_ids),
        "caused_by": causes,
        "causes": effects,
        "superseded_by_event_id": getattr(evt, "superseded_by_event_id", None),
    }

    has_utterance_payload = (
        evt.event_type == "utterance"
        or evt.content is not None
        or evt.via_channel_id is not None
        or evt.speaker_id is not None
        or bool(evt.addressee_ids)
        or evt.truth_value is not None
    )
    if has_utterance_payload:
        ch = (
            ws.channels.get(evt.via_channel_id)
            if evt.via_channel_id else None
        )
        out["utterance"] = {
            "content": evt.content,
            "via_channel_id": evt.via_channel_id,
            "channel_medium": ch.medium if ch else None,
            "speaker": (
                {
                    "id": evt.speaker_id,
                    "name": ws.entities[evt.speaker_id].name
                    if evt.speaker_id and evt.speaker_id in ws.entities
                    else evt.speaker_id,
                }
                if evt.speaker_id else None
            ),
            "addressees": _resolve_names(ws, evt.addressee_ids),
            "truth_value": evt.truth_value,
        }
    return out


def project_channel(ws: WorldStateV1, ch: Channel) -> Dict[str, Any]:
    """Project a :class:`Channel` into a serialisable dict.

    Includes participants (with names), directionality, intelligibility map,
    establishment / termination ticks, and the IDs of any utterance events
    that have already travelled over this channel.
    """
    utterances = [
        {
            "id": e.id,
            "fabula_time": e.fabula_time,
            "syuzhet_index": e.syuzhet_index,
            "speaker_id": e.speaker_id,
            "addressee_ids": list(e.addressee_ids),
            "truth_value": e.truth_value,
            "content_preview": (e.content or "")[:80],
        }
        for e in ws.events
        if e.event_type == "utterance" and e.via_channel_id == ch.id
    ]
    return {
        "id": ch.id,
        "type": "Channel",
        "name": ch.name,
        "medium": ch.medium,
        "participants": _resolve_names(ws, ch.participant_ids),
        "directionality": ch.directionality,
        "intelligibility": dict(ch.intelligibility),
        "established_at_fabula": ch.established_at_fabula,
        "terminated_at_fabula": ch.terminated_at_fabula,
        "evidence_strength": ch.evidence_strength,
        "utterances": utterances,
    }


def trace_information_flow(
    ws: WorldStateV1,
    node_id: str,
    *,
    direction: str = "both",
    depth: int = 3,
) -> Dict[str, Any]:
    """Trace epistemic information flow through utterance events and channels.

    Companion to :func:`shadow_loom_mcp.server.trace_causality` (which only
    walks ``ws.causal_topology``). Returns nodes and edges along the
    utterance/channel provenance graph so callers can answer "how did this
    belief reach character X?" or "who else heard this?".
    """
    visited: set[str] = set()
    edges: List[Dict[str, Any]] = []

    # Index utterances by speaker / addressee / channel for cheap lookups.
    utterances = [e for e in ws.events if e.event_type == "utterance"]
    by_speaker: Dict[str, List[EventNode]] = {}
    by_addressee: Dict[str, List[EventNode]] = {}
    by_channel: Dict[str, List[EventNode]] = {}
    for e in utterances:
        if e.speaker_id:
            by_speaker.setdefault(e.speaker_id, []).append(e)
        for a in e.addressee_ids:
            by_addressee.setdefault(a, []).append(e)
        if e.via_channel_id:
            by_channel.setdefault(e.via_channel_id, []).append(e)

    def _walk(nid: str, d: int, downstream: bool) -> None:
        if d <= 0 or nid in visited:
            return
        visited.add(nid)

        # An entity/object node: follow its utterances forward (as speaker)
        # or backward (as addressee).
        if nid.startswith(("ENT_", "OBJ_")):
            forward = by_speaker.get(nid, []) if downstream else by_addressee.get(nid, [])
            backward = by_addressee.get(nid, []) if downstream else by_speaker.get(nid, [])
            for e in forward:
                edges.append({
                    "source": nid, "target": e.id,
                    "kind": "speaks" if downstream else "addressed_by",
                    "channel_id": e.via_channel_id,
                    "truth_value": e.truth_value,
                })
                _walk(e.id, d - 1, downstream)
            for e in backward:
                # Only surface secondary edges, do not recurse — they are
                # context for the caller, not part of the trace.
                edges.append({
                    "source": e.id, "target": nid,
                    "kind": "addresses" if downstream else "spoken_by",
                    "channel_id": e.via_channel_id,
                    "truth_value": e.truth_value,
                })

        # An utterance event: branch to its addressees and (if present) its
        # channel; the channel in turn branches to its participants.
        elif nid.startswith("EVT_"):
            evt = next((e for e in utterances if e.id == nid), None)
            if evt is None:
                return
            for a in evt.addressee_ids:
                edges.append({
                    "source": nid, "target": a,
                    "kind": "delivers_to",
                    "channel_id": evt.via_channel_id,
                })
                _walk(a, d - 1, downstream)
            if evt.via_channel_id and evt.via_channel_id in ws.channels:
                edges.append({
                    "source": nid, "target": evt.via_channel_id,
                    "kind": "via_channel",
                })
                _walk(evt.via_channel_id, d - 1, downstream)

        # A channel: branch to every participant (subject to directionality).
        elif nid.startswith("CHN_"):
            ch = ws.channels.get(nid)
            if ch is None:
                return
            for pid in ch.participant_ids:
                edges.append({
                    "source": nid, "target": pid,
                    "kind": "channel_participant",
                    "intelligibility": float(ch.intelligibility.get(pid, 1.0)),
                })
            for e in by_channel.get(nid, []):
                edges.append({
                    "source": e.id, "target": nid,
                    "kind": "carried_utterance",
                })

    if direction in ("upstream", "both"):
        _walk(node_id, depth, downstream=False)
    if direction in ("downstream", "both"):
        # Reset visited so the downstream walk can re-enter the root.
        for v in list(visited):
            visited.discard(v)
        visited.add(node_id)
        _walk(node_id, depth, downstream=True)

    return {
        "root": node_id,
        "direction": direction,
        "depth": depth,
        "nodes": sorted(visited | {node_id}),
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# POV intelligibility filter (scaffold)
# ---------------------------------------------------------------------------

def pov_visible_event_ids(
    ws: WorldStateV1,
    pov_entity_id: str,
    *,
    intelligibility_threshold: float = 0.3,
) -> set[str]:
    """Return the EVT_ ids the POV character could plausibly know about.

    Scaffold for per-POV "limited omniscience" filtering. An utterance
    is visible to ``pov_entity_id`` when:
      * the POV is the speaker, **or**
      * the POV is in ``addressee_ids``, **or**
      * the POV is a participant in ``via_channel_id`` whose
        ``intelligibility[pov_entity_id]`` >= ``intelligibility_threshold``
        (missing entry defaults to 1.0 = fully intelligible).

    Non-utterance events are visible when the POV is in ``actor_ids`` or
    ``target_ids``. This is intentionally conservative: a richer
    implementation would also consult spatial co-location, but this
    minimum primitive unblocks the ``pov_entity_id`` API surface.
    Belief-provenance reasoning lives downstream in
    ``DirectiveAssembler.compute_dramatic_irony_score``, where the
    ``Belief.acquired_via_event_id`` / ``acquired_via_channel_id`` gates
    govern whether a belief still counts as knowledge in the current
    branch.
    """
    visible: set[str] = set()
    if not pov_entity_id:
        return visible
    for evt in ws.events:
        if evt.event_type == "utterance":
            if evt.speaker_id == pov_entity_id:
                visible.add(evt.id)
                continue
            if pov_entity_id in evt.addressee_ids:
                visible.add(evt.id)
                continue
            ch = ws.channels.get(evt.via_channel_id) if evt.via_channel_id else None
            if ch and pov_entity_id in ch.participant_ids:
                intel = float(ch.intelligibility.get(pov_entity_id, 1.0))
                if intel >= intelligibility_threshold:
                    visible.add(evt.id)
        else:
            if pov_entity_id in (evt.actor_ids or []) or pov_entity_id in (
                evt.target_ids or []
            ):
                visible.add(evt.id)
    return visible


def filter_world_state_for_pov(
    ws: WorldStateV1,
    pov_entity_id: Optional[str],
    *,
    intelligibility_threshold: float = 0.3,
) -> WorldStateV1:
    """Return ``ws`` filtered through ``pov_entity_id``'s epistemic lens.

    Restricts the world to what the POV character could plausibly know:

      * ``events`` — only those visible via :func:`pov_visible_event_ids`.
      * ``channels`` — only those the POV participates in; the
        ``intelligibility`` map is also stripped to the POV's own entry
        so other participants' decode probabilities don't leak.
      * ``causal_topology`` — only edges with at least one endpoint the
        POV could plausibly know (a visible event, the POV themself, an
        object held / co-located with the POV, or a world trait the POV
        already has a belief about).
      * ``entities`` — the POV keeps their full record. Every other
        entity is reduced: ``beliefs`` are stripped (POV can't read
        minds), ``concerns`` are stripped (private interior state),
        and ``state_timeline`` snapshots are kept only at fabula ticks
        the POV could witness via a visible event.
      * ``propositions`` — kept (they are audience-level claims), but
        ``truth_at_fabula`` is filtered to ticks the POV could witness.

    Callers that pass ``pov_entity_id=None`` get the world unchanged.
    This is intentionally conservative: a richer implementation would
    also gate on spatial co-location and existing belief provenance.
    """
    if not pov_entity_id or pov_entity_id not in ws.entities:
        return ws
    visible_evt_ids = pov_visible_event_ids(
        ws, pov_entity_id,
        intelligibility_threshold=intelligibility_threshold,
    )
    visible_chn_ids = {
        cid for cid, ch in ws.channels.items()
        if pov_entity_id in ch.participant_ids
    }
    # Fabula ticks the POV could plausibly witness — used to gate
    # off-screen state-timeline drift on other entities/world traits.
    visible_fabula_ticks: set[int] = {
        e.fabula_time for e in ws.events if e.id in visible_evt_ids
    }

    filtered = ws.model_copy(deep=True)
    filtered.events = [e for e in filtered.events if e.id in visible_evt_ids]

    # Channels: prune non-participant channels; on surviving channels
    # strip the intelligibility map to the POV's own entry so other
    # participants' decode probabilities aren't readable.
    pruned_channels: Dict[str, Channel] = {}
    for cid, ch in filtered.channels.items():
        if cid not in visible_chn_ids:
            continue
        own = ch.intelligibility.get(pov_entity_id)
        ch.intelligibility = {pov_entity_id: own} if own is not None else {}
        pruned_channels[cid] = ch
    filtered.channels = pruned_channels

    # Causal topology: keep an edge if either endpoint is something
    # the POV could plausibly observe — a visible event, the POV
    # themself, an object the POV holds or shares a location with, or
    # a world trait. World traits are global and audience-readable so
    # we keep all WORLD_-keyed edges.
    pov_ent = filtered.entities.get(pov_entity_id)
    pov_loc = pov_ent.location_id if pov_ent else None
    pov_known_objects = {
        oid for oid, obj in filtered.objects.items()
        if obj.owner_id == pov_entity_id or (pov_loc and obj.location_id == pov_loc)
    }
    pov_known_node_ids = (
        visible_evt_ids
        | {pov_entity_id}
        | pov_known_objects
        | set(filtered.world_traits.keys())
    )
    filtered.causal_topology = [
        ce for ce in filtered.causal_topology
        if ce.source_id in pov_known_node_ids or ce.target_id in pov_known_node_ids
    ]

    # Entities: POV keeps full record; everyone else has interior
    # state stripped (beliefs, concerns) and timelines clipped to
    # fabula ticks the POV could witness.
    for eid, ent in filtered.entities.items():
        if eid == pov_entity_id:
            continue
        ent.beliefs = []
        ent.concerns = []
        if ent.state_timeline:
            ent.state_timeline = [
                snap for snap in ent.state_timeline
                if snap.fabula_time in visible_fabula_ticks
            ]

    # World traits: drift snapshots gated to visible ticks — POV can't
    # know about silent off-screen world drift. The trait itself stays
    # (it's a global background fact).
    for wt in filtered.world_traits.values():
        if wt.state_timeline:
            wt.state_timeline = [
                snap for snap in wt.state_timeline
                if snap.fabula_time in visible_fabula_ticks
            ]

    # Propositions: clip truth_at_fabula commits to ticks the POV
    # witnessed; clip state_timeline likewise.
    for prop in filtered.propositions:
        if prop.truth_at_fabula:
            prop.truth_at_fabula = {
                t: v for t, v in prop.truth_at_fabula.items()
                if t in visible_fabula_ticks
            }
        if prop.state_timeline:
            prop.state_timeline = [
                snap for snap in prop.state_timeline
                if snap.fabula_time in visible_fabula_ticks
            ]

    return filtered
