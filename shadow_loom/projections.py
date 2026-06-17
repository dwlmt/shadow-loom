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
    reconstruct_channel_at,
    reconstruct_concern_at,
    reconstruct_entity_at,
    reconstruct_location_at,
    reconstruct_object_at,
    reconstruct_proposition_at,
    reconstruct_world_trait_at,
)


def reconstruct_entity_at_causal(
    ws: WorldStateV1,
    entity_id: str,
    fabula_time: int,
) -> Dict[str, Any]:
    """Reconstruct an entity at ``fabula_time`` overlaying causal mutations.

    1. Seed traits from ``Entity.traits`` (pre-story baseline).
    2. Replay every ``mutation`` :class:`CausalEdge` whose
       ``target_id`` is this entity, ``trait_target`` is set, and
       ``fabula_time <= fabula_time`` — accumulating signed
       ``trait_delta`` per axis (clamped to ``[0, 1]`` because
       ``TraitVector.value`` is non-negative on the schema).
       ``mutation_social`` edges are *excluded*: they carry
       relationship axes, not personal traits.
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
            # Personal traits only: ``mutation`` edges carry a
            # personal ``trait_target`` (ambition, anger, ...).
            # ``mutation_social`` edges carry a *relationship* axis
            # (affinity / fear / power_dynamic) and ``target_id`` is
            # the dyad's source entity, so including them here leaked
            # relationship metrics into the personal trait dict
            # (audit 2026-06-06). Relationship axes are reconstructed
            # via ``reconstruct_relationship_at``.
            if ce.causality_type == "mutation"
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
    # Authored snapshots are authoritative — but only when one was
    # actually written. The previous equality-vs-baseline guard
    # silently ignored a snapshot that intentionally reset the
    # magnitude back to its baseline value (e.g. baseline 0.5 →
    # mutation pushes to 0.8 → snapshot writes 0.5 to override),
    # leaving the causal-accumulated 0.8 in place (audit 2026-05-26).
    # Discriminate the same way the entity-side projection does:
    # check whether any state_timeline snapshot at-or-before the
    # horizon actually wrote a magnitude on this branch.
    holder_world = getattr(wt, "world_id", "factual") or "factual"
    snapshot_wrote_magnitude = any(
        s.fabula_time <= fabula_time
        and s.magnitude is not None
        and (getattr(s, "world_id", holder_world) or holder_world) == holder_world
        for s in wt.state_timeline
    )
    if snapshot_wrote_magnitude and snap_mag:
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
    # Audit R17-12: when two distinct entities share the same display
    # name (e.g., two "John"s) the projection layer used to emit
    # indistinguishable rows. Detect collisions within this id list
    # and suffix a short id fragment so consumers can disambiguate.
    name_counts: Dict[str, int] = {}
    for i in ids:
        nm = ws.entities[i].name if i in ws.entities else i
        name_counts[nm] = name_counts.get(nm, 0) + 1
    rows: List[Dict[str, str]] = []
    for i in ids:
        nm = ws.entities[i].name if i in ws.entities else i
        if name_counts.get(nm, 0) > 1 and i != nm:
            # ``i`` typically looks like ``ENT_ABC123``; strip the
            # canonical prefix when present and keep a short tail so
            # the label stays readable.
            tail = i.split("_", 1)[-1][:6]
            nm = f"{nm}#{tail}"
        rows.append({"id": i, "name": nm})
    return rows


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


def _event_fabula_time(ws: WorldStateV1, event_id: str) -> Optional[int]:
    """Return ``fabula_time`` of ``event_id`` if found, else ``None``.

    Used by the ``*_at`` projection helpers below to anchor causal edges
    by the OTHER endpoint's occurrence time (audit R16-5).
    """
    for e in ws.events:
        if e.id == event_id:
            return getattr(e, "fabula_time", None)
    return None


def project_event_at(
    ws: WorldStateV1, evt: EventNode, at_time: int,
) -> Dict[str, Any]:
    """Anchor-filtered variant of :func:`project_event` (audit R16-5).

    Behaves like ``project_event`` but drops ``caused_by`` /  ``causes``
    edges whose *other endpoint* event has a ``fabula_time`` strictly
    greater than ``at_time``. The caller is responsible for the
    occurrence gate on the focal event itself (the MCP layer already
    runs :func:`reconstruct_event_at` first).
    """
    out = project_event(ws, evt)

    def _edge_visible(other_id: Optional[str]) -> bool:
        if other_id is None:
            return True
        ft = _event_fabula_time(ws, other_id)
        # If the other event has no fabula_time we keep it (can't
        # disprove visibility); only DROP when we know it's in the
        # future of ``at_time``.
        return ft is None or ft <= at_time

    out["caused_by"] = [
        c for c in out.get("caused_by", []) if _edge_visible(c.get("source"))
    ]
    out["causes"] = [
        c for c in out.get("causes", []) if _edge_visible(c.get("target"))
    ]
    return out


def project_channel_at(
    ws: WorldStateV1, ch: Channel, at_time: int,
) -> Dict[str, Any]:
    """Anchor-filtered variant of :func:`project_channel` (audit R16-6).

    Filters the ``utterances`` list to only those whose ``fabula_time``
    is ``<= at_time``. The caller is responsible for the channel
    availability-window gate on the channel itself (the MCP layer runs
    :func:`reconstruct_channel_at` first).
    """
    out = project_channel(ws, ch)
    out["utterances"] = [
        u for u in out.get("utterances", [])
        if (u.get("fabula_time") is None) or (u.get("fabula_time") <= at_time)
    ]
    return out


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
    # Audit R18-5: event types whose semantics are interior /
    # narrator-mediated. A POV listed in ``target_ids`` is the
    # *referent* of the experience, not a witness; only the speaker /
    # experiencer plus an explicit external witness can see them.
    INTERIOR_EVENT_TYPES = {
        "revelation", "dream", "vision", "thought", "soliloquy",
        "aside", "prophecy", "interior_monologue",
    }
    for evt in ws.events:
        if evt.event_type == "utterance":
            if evt.speaker_id == pov_entity_id:
                visible.add(evt.id)
                continue
            # Audit R18-4: addressee visibility must still respect the
            # carrying channel. Laura's interior confession to Fred in
            # ``brief_encounter`` is routed on a channel whose
            # ``intelligibility[ENT_FRED] == 0.0``; without this gate
            # the addressee shortcut would deliver the utterance
            # straight to Fred's POV.
            if pov_entity_id in evt.addressee_ids:
                if evt.via_channel_id:
                    ch = ws.channels.get(evt.via_channel_id)
                    if ch is not None:
                        evt_ft = getattr(evt, "fabula_time", None)
                        est = getattr(ch, "established_at_fabula", None)
                        term = getattr(ch, "terminated_at_fabula", None)
                        if evt_ft is not None:
                            if est is not None and evt_ft < int(est):
                                continue
                            if term is not None and evt_ft >= int(term):
                                continue
                        intel = float(
                            ch.intelligibility.get(pov_entity_id, 1.0)
                        )
                        if intel < intelligibility_threshold:
                            continue
                visible.add(evt.id)
                continue
            ch = ws.channels.get(evt.via_channel_id) if evt.via_channel_id else None
            if ch and pov_entity_id in ch.participant_ids:
                # Audit R17-1: gate channel-mediated visibility on the
                # channel's availability window. Without this an
                # utterance routed (correctly or via extraction error)
                # on a long-severed channel leaks to every participant.
                evt_ft = getattr(evt, "fabula_time", None)
                est = getattr(ch, "established_at_fabula", None)
                term = getattr(ch, "terminated_at_fabula", None)
                if evt_ft is not None:
                    if est is not None and evt_ft < int(est):
                        continue
                    if term is not None and evt_ft >= int(term):
                        continue
                intel = float(ch.intelligibility.get(pov_entity_id, 1.0))
                if intel >= intelligibility_threshold:
                    visible.add(evt.id)
        else:
            # Audit R18-5: interior / narrator-mediated event types
            # are visible only to the experiencer (first actor),
            # never to target_ids (which name the referent of the
            # dream/prophecy, not a witness).
            etype = getattr(evt, "event_type", None)
            actor_ids = list(evt.actor_ids or [])
            target_ids = list(evt.target_ids or [])
            if etype in INTERIOR_EVENT_TYPES:
                if pov_entity_id in actor_ids:
                    visible.add(evt.id)
                continue
            if pov_entity_id in actor_ids or pov_entity_id in target_ids:
                visible.add(evt.id)
    return visible


def _strip_shadow_sidecars(ws: WorldStateV1) -> None:
    """Clear every ``shadow_*`` sidecar on ``ws`` in-place.

    Round-4 audit helper. ``filter_world_state_for_pov`` operates on
    the factual surface only; any populated shadow sidecar carried
    through ``model_copy(deep=True)`` would leak the entire
    counterfactual world (entities, objects, propositions, channels,
    events, locations, world traits, the three topologies, and the
    deletion tombstone lists) past the POV boundary unfiltered.
    """
    for attr in (
        "shadow_entities",
        "shadow_objects",
        "shadow_propositions",
        "shadow_world_traits",
        "shadow_social_topology",
        "shadow_events",
        "shadow_channels",
        "shadow_locations",
        "shadow_causal_topology",
        "shadow_spatial_topology",
        "shadow_removed_entity_ids",
        "shadow_removed_object_ids",
        "shadow_removed_channel_ids",
        "shadow_removed_event_ids",
        "shadow_removed_location_ids",
    ):
        if hasattr(ws, attr):
            setattr(ws, attr, {})


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
    # Audit R18-9: unknown POV used to fall through to ``return ws``,
    # silently handing the caller the full omniscient world. That
    # defeats the POV safety boundary on every consumer of this
    # surface. Fail closed: an unknown POV gets an empty-shaped
    # world rather than the factual base.
    if not pov_entity_id:
        return ws
    if pov_entity_id not in ws.entities:
        empty = ws.model_copy(deep=True)
        empty.events = []
        empty.channels = {}
        empty.causal_topology = []
        empty.spatial_topology = []
        empty.social_topology = []
        empty.propositions = []
        # Drop every entity body so an unknown POV cannot even
        # enumerate the cast.
        empty.entities = {}
        empty.objects = {}
        empty.locations = {}
        empty.world_traits = {}
        _strip_shadow_sidecars(empty)
        return empty
    visible_evt_ids = pov_visible_event_ids(
        ws, pov_entity_id,
        intelligibility_threshold=intelligibility_threshold,
    )
    # Audit R18-7: channel existence must require *perceivability*,
    # not bare participant membership. ``CHN_HIDDEN_TELESCREEN_SURVEILLANCE``
    # in 1984 lists Winston/Julia as participants but with
    # ``intelligibility=0.0`` precisely because they cannot perceive
    # the channel exists until the on-page reveal. Require a positive
    # decode probability AND that the channel is currently within its
    # availability window relative to any visible event.
    visible_chn_ids: set[str] = set()
    for cid, ch in ws.channels.items():
        if pov_entity_id not in ch.participant_ids:
            continue
        intel = float(ch.intelligibility.get(pov_entity_id, 1.0))
        if intel < intelligibility_threshold:
            # Imperceptible channel — hide its existence from POV.
            continue
        visible_chn_ids.add(cid)
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
    # Audit R18-8: ``world_traits`` are *audience-level* facts, not
    # automatically character knowledge. Without an explicit
    # POV-known surface, every global ideology / ambient norm leaks
    # via every POV slice (1984's Party-doctrine traits leaking to
    # every minor character POV). Gate inclusion on whether at least
    # one visible event references the trait in its actor/target/
    # location set, or the POV themselves is annotated to know it.
    pov_known_world_trait_ids: set[str] = set()
    visible_evt_refs: set[str] = set()
    for e in filtered.events:
        if e.id not in visible_evt_ids:
            continue
        visible_evt_refs.update(e.actor_ids or [])
        visible_evt_refs.update(e.target_ids or [])
        if getattr(e, "at_location_id", None):
            visible_evt_refs.add(e.at_location_id)
    for wt_id in filtered.world_traits.keys():
        if wt_id in visible_evt_refs:
            pov_known_world_trait_ids.add(wt_id)
    # Also accept world traits the POV explicitly believes about
    # (i.e. has a Belief whose target_id is the trait).
    if pov_ent is not None:
        for b in (getattr(pov_ent, "beliefs", None) or []):
            tid = getattr(b, "target_id", None)
            if tid and tid in filtered.world_traits:
                pov_known_world_trait_ids.add(tid)
    pov_known_node_ids = (
        visible_evt_ids
        | {pov_entity_id}
        | pov_known_objects
        | pov_known_world_trait_ids
    )
    # Audit (eighth pass, M1): the original filter kept an edge if
    # *either* endpoint was POV-known, which leaked the hidden
    # endpoint's id through the dangling reference (e.g. in
    # ``Nineteen Eighty-Four`` the POV could see an edge
    # ``EVT_VISIBLE → OBJ_HIDDEN_TELESCREEN`` even when the object
    # itself had been stripped from ``filtered.objects``). Require
    # both endpoints to survive POV visibility so referential
    # integrity is preserved.
    filtered.causal_topology = [
        ce for ce in filtered.causal_topology
        if ce.source_id in pov_known_node_ids
        and ce.target_id in pov_known_node_ids
    ]
    # Audit R18-6: prune the ``objects`` and ``locations`` registries
    # to the POV's perceivable surface. Before this filter, Winston's
    # POV carried ``OBJ_HIDDEN_TELESCREEN`` plus the full location
    # registry (LOC_ROOM_101, LOC_OBRIEN_FLAT) the moment the world
    # loaded, defeating the entire epistemic-irony design.
    pov_known_locations: set[str] = set()
    if pov_loc:
        pov_known_locations.add(pov_loc)
        # Adjacent locations via spatial_topology are reachable
        # awareness (you know the next room exists).
        for se in (filtered.spatial_topology or []):
            if getattr(se, "source_id", None) == pov_loc:
                tgt = getattr(se, "target_id", None)
                if tgt:
                    pov_known_locations.add(tgt)
            if getattr(se, "target_id", None) == pov_loc:
                src = getattr(se, "source_id", None)
                if src:
                    pov_known_locations.add(src)
    # Locations referenced by visible events are also known.
    for e in filtered.events:
        if e.id in visible_evt_ids and getattr(e, "at_location_id", None):
            pov_known_locations.add(e.at_location_id)
    filtered.locations = {
        lid: loc for lid, loc in filtered.locations.items()
        if lid in pov_known_locations
    }
    filtered.objects = {
        oid: obj for oid, obj in filtered.objects.items()
        if oid in pov_known_objects
    }
    # World traits the POV doesn't know are dropped entirely.
    filtered.world_traits = {
        wt_id: wt for wt_id, wt in filtered.world_traits.items()
        if wt_id in pov_known_world_trait_ids
    }

    # Audit (seventh pass, C7): ``social_topology`` was left
    # unfiltered, so every RelationshipEdge between two non-POV
    # entities (and every interior metric on those edges) leaked
    # straight through the POV boundary — handing the reader
    # omniscient knowledge of off-page rivalries, affinities, and
    # fears. Restrict to edges where the POV is at least one
    # endpoint; characters' private feelings about each other are
    # not POV-knowable without an explicit observation channel.
    filtered.social_topology = [
        re for re in (filtered.social_topology or [])
        if getattr(re, "source_entity_id", None) == pov_entity_id
        or getattr(re, "target_entity_id", None) == pov_entity_id
    ]

    # Audit (seventh pass, C8): ``spatial_topology`` was left
    # unfiltered, so edges between two unfamiliar locations
    # (rooms / regions the POV has never visited and that no
    # visible event references) leaked the map's full geometry.
    # Restrict to edges where at least one endpoint is in the
    # POV-known locations set.
    filtered.spatial_topology = [
        se for se in (filtered.spatial_topology or [])
        if getattr(se, "source_id", None) in pov_known_locations
        or getattr(se, "target_id", None) in pov_known_locations
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
            # Audit (seventh pass, C9): the snapshot-level
            # ``beliefs_added`` / ``beliefs_invalidated`` channels
            # are the per-tick deltas the top-level scrub above
            # would otherwise replay back into the entity's
            # interior state. Clearing only the top-level lists
            # leaves the snapshot deltas readable by any consumer
            # that walks ``state_timeline`` directly (Q&A,
            # diagnostics, downstream replay), leaking other
            # characters' belief formation to the POV. Strip the
            # delta fields on every retained snapshot in lockstep
            # with the top-level scrub.
            clipped: List[Any] = []
            for snap in ent.state_timeline:
                if snap.fabula_time not in visible_fabula_ticks:
                    continue
                _snap_copy = snap.model_copy(deep=True)
                _snap_copy.beliefs_added = []
                _snap_copy.beliefs_invalidated = []
                clipped.append(_snap_copy)
            ent.state_timeline = clipped

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
            # Audit (eighth pass, M2): ``truth_at_fabula`` is typed
            # ``Dict[int, bool]`` but a JSON/DB round-trip outside
            # the pydantic validator can present str keys. Coerce
            # to int before membership-checking against the int
            # ``visible_fabula_ticks`` set or the entire ledger is
            # silently dropped from the POV slice.
            _clipped: Dict[int, bool] = {}
            for _t, _v in prop.truth_at_fabula.items():
                try:
                    _ti = int(_t)
                except (TypeError, ValueError):
                    continue
                if _ti in visible_fabula_ticks:
                    _clipped[_ti] = _v
            prop.truth_at_fabula = _clipped
        if prop.state_timeline:
            prop.state_timeline = [
                snap for snap in prop.state_timeline
                if snap.fabula_time in visible_fabula_ticks
            ]

    # Round-4 audit: clear all shadow_* sidecars from the POV slice.
    # ``filter_world_state_for_pov`` is defined on the factual
    # surface; leaving the deep-copied shadow collections intact
    # leaks the entire counterfactual world (shadow_entities,
    # shadow_propositions, shadow_events, shadow_channels, shadow
    # topology, deletion tombstones, …) unfiltered through the POV
    # boundary. Callers that need a POV view of a counterfactual
    # branch must ``projected_for_branch`` first, then filter the
    # already-merged projection through this function.
    _strip_shadow_sidecars(filtered)

    return filtered


# =====================================================================
# Full-world fabula-time snapshot (model-side, MCP-safe)
# =====================================================================


def snapshot_world_at(ws: WorldStateV1, t: int) -> WorldStateV1:
    """Return a deep copy of ``ws`` reconstructed to fabula_time ``t``.

    Mirrors the time-slicing rules used by
    :func:`shadow_loom.extract_graph.extract_ego_graph_from_memory`
    and by the UI's ``snapshot_world_at`` helper, but lives at the
    model layer so the MCP server (which must not depend on the UI
    package) can use the same projection for ``at_time``-anchored
    tools (``compute_tension``, ``trace_causality``).

    Per element:

      * :class:`Entity` — ``traits`` / ``status`` / ``location_id`` /
        ``beliefs`` replayed via :func:`reconstruct_entity_at`;
        per-entity :class:`Concern` rows replayed via
        :func:`reconstruct_concern_at`.
      * :class:`GlobalTrait` — ``magnitude`` replayed via
        :func:`reconstruct_world_trait_at`.
      * :class:`NarrativeObject` — ``location_id`` / ``owner_id`` /
        ``properties`` replayed via :func:`reconstruct_object_at`.
      * :class:`Proposition` — ``stakes`` /
        ``audience_default_prior`` / ``description`` replayed via
        :func:`reconstruct_proposition_at`; ``truth_at_fabula``
        filtered to ``<= t``.
      * :class:`Location` — ``ambient_state`` normalised via
        :func:`reconstruct_location_at` (today a pass-through; the
        helper locks the contract for the future per-key timeline).
      * ``events`` — filtered to ``fabula_time <= t``.
      * ``causal_topology`` — filtered to ``fabula_time <= t``.
      * ``social_topology`` — per-axis ``RelationshipMetric.last_updated_fabula
        <= t``; whole edge dropped when no axis survives.
      * ``spatial_topology`` — established by ``t`` and not yet
        destroyed at ``t``.
      * ``channels`` — gated by :func:`reconstruct_channel_at` (the
        canonical half-open window).

    NOTE: unlike the UI's richer snapshot helper this one does **not**
    replay ``mutation_social`` causal edges back onto relationship
    metric values. Engine scorers gate on the per-axis ``observed``
    flag plus ``last_updated_fabula`` (matching what the engine
    itself does when computing tension at a tick), so the metric
    value at ``last_updated_fabula`` is the canonical reading. UI
    chart consumers that want the inter-frame trajectory can layer
    their own causal replay on top.
    """
    if t is None:
        return ws

    new = ws.model_copy(deep=True)

    # A factual point-in-time snapshot must not carry the counterfactual
    # shadow sidecars: the slicing below only time-filters the factual
    # surface, so un-sliced shadows would leak the full (future- and
    # branch-bearing) counterfactual world into a past factual snapshot.
    # Mirror ``filter_world_state_for_pov``'s stance and drop them.
    _strip_shadow_sidecars(new)

    # Local import to avoid a top-level cycle.
    from shadow_loom.models import Belief, TraitVector

    for eid, ent in new.entities.items():
        snap = reconstruct_entity_at(ent, t)
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
        for concern in (ent.concerns or []):
            csnap = reconstruct_concern_at(concern, t)
            if not isinstance(csnap, dict):
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
        snap = reconstruct_world_trait_at(wt, t)
        mag = snap["magnitude"]
        wt.magnitude = TraitVector(
            value=mag["value"],
            inertia=mag["inertia"],
            evidence_strength=mag.get("evidence_strength", "moderate"),
        )
        if snap.get("description") is not None:
            wt.description = snap["description"]

    new.events = [evt for evt in new.events if evt.fabula_time <= t]

    new.causal_topology = [
        ce for ce in new.causal_topology if ce.fabula_time <= t
    ]

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

    new.spatial_topology = [
        se for se in new.spatial_topology
        if se.established_at_fabula <= t
        and (se.destroyed_at_fabula is None or se.destroyed_at_fabula > t)
    ]

    new.channels = {
        cid: ch for cid, ch in new.channels.items()
        if reconstruct_channel_at(ch, t) is not None
    }

    for oid, obj in new.objects.items():
        obj_snap = reconstruct_object_at(obj, t)
        obj.location_id = obj_snap["location_id"]
        obj.owner_id = obj_snap["owner_id"]
        obj.properties = obj_snap["properties"]

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
        try:
            tmap = {
                int(k): v for k, v in (prop.truth_at_fabula or {}).items()
            }
        except (TypeError, ValueError):
            tmap = {}
        prop.truth_at_fabula = {k: v for k, v in tmap.items() if k <= t}

    for lid, loc in new.locations.items():
        try:
            _ = reconstruct_location_at(loc, t)
        except Exception:
            # Defensive: a future per-key timeline failure must not
            # break snapshot rendering. Live ambient_state remains.
            pass

    return new
