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

from shadow_loom.models import Channel, EventNode, WorldStateV1


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
    implementation would also consult ``Belief.acquired_via_event_id`` and
    spatial co-location, but this minimum primitive unblocks the
    ``pov_entity_id`` API surface.
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

    Scaffold: currently restricts ``ws.events`` to those visible to the POV
    via :func:`pov_visible_event_ids` and prunes channels the POV does not
    participate in. Callers that pass ``pov_entity_id=None`` get the world
    unchanged.

    TODO: also filter
      * other entities' ``beliefs`` (the POV cannot read minds);
      * ``causal_topology`` edges where both endpoints are POV-invisible;
      * ``Channel.intelligibility`` map entries for non-POV participants.
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
    filtered = ws.model_copy(deep=True)
    filtered.events = [e for e in filtered.events if e.id in visible_evt_ids]
    filtered.channels = {
        cid: ch for cid, ch in filtered.channels.items() if cid in visible_chn_ids
    }
    return filtered
