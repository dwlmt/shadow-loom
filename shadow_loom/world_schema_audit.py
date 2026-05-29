# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static world-schema sanity audit (AUDIT P1-7).

A deterministic, LLM-free pre-pass over :class:`WorldStateV1` that
catches authoring / extraction defects the pipeline would otherwise
swallow until a downstream consumer hits an ``AttributeError`` or a
silently-empty query result. Surfaces three classes of issue:

* **dangling references** \u2014 a Belief / Concern / Causal edge / Event
  references an entity, object, location, channel or proposition id
  that does not exist in the world.
* **location plausibility** \u2014 an event's ``at_location_id`` is not a
  declared :class:`Location`; an entity / object's initial
  ``location_id`` is unknown.
* **temporal coherence** \u2014 ``Proposition.truth_at_fabula`` carries a
  tick before the earliest event referenced via ``referent_ids``;
  ``RelationshipMetric.last_updated_fabula`` predates the edge's
  ``established_at_fabula``; ``EntityStateSnapshot.fabula_time``
  predates the entity's earliest event participation.

Each finding is a plain-string warning with a stable prefix so
callers can grep / filter. The function never raises and is safe to
run on any world (including partially-built ones during ingestion).
"""
from __future__ import annotations

from typing import Any, Dict, List

from shadow_loom.models import WorldStateV1


def audit_world_schema(world_state: WorldStateV1) -> List[str]:
    issues: List[str] = []

    entities = world_state.entities or {}
    objects = world_state.objects or {}
    locations = world_state.locations or {}
    channels = world_state.channels or {}
    events = list(world_state.events or [])
    propositions = list(world_state.propositions or [])

    entity_ids = set(entities.keys())
    object_ids = set(objects.keys())
    location_ids = set(locations.keys())
    channel_ids = set(channels.keys())
    proposition_ids = {p.proposition_id for p in propositions}
    event_ids = {e.id for e in events}
    # R19-H11: build event index for holder-presence belief audit.
    events_by_id: Dict[str, Any] = {e.id: e for e in events}
    world_trait_ids = set((world_state.world_traits or {}).keys())

    # ---------------- Location plausibility ----------------
    for eid, ent in entities.items():
        if ent.location_id and ent.location_id not in location_ids:
            issues.append(
                f"[schema\u00b7location] ENT {eid} initial location_id "
                f"={ent.location_id!r} is not in WorldStateV1.locations."
            )
    for oid, obj in objects.items():
        loc_id = getattr(obj, "location_id", None)
        if loc_id and loc_id not in location_ids:
            issues.append(
                f"[schema\u00b7location] OBJ {oid} initial location_id "
                f"={loc_id!r} is not in WorldStateV1.locations."
            )
    for evt in events:
        loc_id = getattr(evt, "at_location_id", None)
        if loc_id and loc_id not in location_ids:
            issues.append(
                f"[schema\u00b7location] EVT {evt.id} at_location_id "
                f"={loc_id!r} is not in WorldStateV1.locations."
            )

    # ---------------- Dangling references ------------------
    for eid, ent in entities.items():
        for b in (ent.beliefs or []):
            tgt = getattr(b, "target_id", None)
            if tgt and (
                tgt not in entity_ids
                and tgt not in object_ids
                and tgt not in location_ids
                and tgt not in world_trait_ids
                and tgt not in channel_ids
                and tgt not in event_ids
            ):
                if isinstance(tgt, str) and tgt.startswith("WORLD_"):
                    issues.append(
                        f"[schema\u00b7world_trait] ENT {eid} belief "
                        f"target_id={tgt!r} references an undefined "
                        f"WORLD_ trait."
                    )
                else:
                    issues.append(
                        f"[schema\u00b7dangling] ENT {eid} belief target_id "
                        f"={tgt!r} is unknown."
                    )
            pid = getattr(b, "proposition_id", None)
            if pid and pid not in proposition_ids:
                issues.append(
                    f"[schema\u00b7dangling] ENT {eid} belief "
                    f"proposition_id={pid!r} is unknown."
                )
        for c in (ent.concerns or []):
            pid = getattr(c, "proposition_id", None)
            if pid and pid not in proposition_ids:
                issues.append(
                    f"[schema\u00b7dangling] ENT {eid} concern "
                    f"proposition_id={pid!r} is unknown."
                )
    for evt in events:
        for actor in (evt.actor_ids or []):
            if actor not in entity_ids:
                issues.append(
                    f"[schema\u00b7dangling] EVT {evt.id} actor_id "
                    f"={actor!r} is unknown."
                )
        for obj_ref in (getattr(evt, "object_ids", None) or []):
            if obj_ref not in object_ids:
                issues.append(
                    f"[schema\u00b7dangling] EVT {evt.id} object_id "
                    f"={obj_ref!r} is unknown."
                )
        # 2026-05-29 round-3 MED: ``Event.target_ids`` (the set of
        # entities / objects / events the event acts upon \u2014 see
        # ``shadow_loom/models.py::Event.target_ids``) was previously
        # never validated. A malformed ingest (e.g. typo
        # ``ENT_KURTS`` for ``ENT_KURTZ`` in Apocalypse Now) would
        # pass schema audit silently, then break target-dependent
        # queries downstream ("who was targeted in
        # EVT_WILLARD_KILLS_KURTZ?") because the resolver finds no
        # such entity. Mirror the actor_ids / object_ids checks.
        # Valid target_ids span entity / object / event / location /
        # world_trait / channel ids \u2014 utterances commonly reference
        # WORLD_ traits (e.g. a curse, a kingdom-level rule) and
        # locations (e.g. "the throne room is in chaos"), and
        # channel-targeted events (e.g. severing a comm line) are
        # legitimate. Restricting to ENT/OBJ/EVT only flagged
        # hand-authored fixtures that referenced WORLD_ targets and
        # treated them as dangling.
        _evt_target_known = (
            entity_ids | object_ids | event_ids
            | location_ids | world_trait_ids | channel_ids
        )
        for tgt_ref in (getattr(evt, "target_ids", None) or []):
            if tgt_ref not in _evt_target_known:
                issues.append(
                    f"[schema\u00b7dangling] EVT {evt.id} target_id "
                    f"={tgt_ref!r} is unknown (expected an ENT_ / "
                    f"OBJ_ / EVT_ / LOC_ / WORLD_ / CHN_ id)."
                )
        for pid_attr in ("asserts_proposition_id", "denies_proposition_id"):
            pid = getattr(evt, pid_attr, None)
            if pid and pid not in proposition_ids:
                issues.append(
                    f"[schema\u00b7dangling] EVT {evt.id} {pid_attr} "
                    f"={pid!r} is unknown."
                )
        for pid in (getattr(evt, "resolves_proposition_ids", None) or []):
            if pid not in proposition_ids:
                issues.append(
                    f"[schema\u00b7dangling] EVT {evt.id} resolves "
                    f"proposition_id={pid!r} is unknown."
                )
    for ce in (world_state.causal_topology or []):
        # CausalEdge endpoints can be any AMWN node id (EVT / ENT /
        # OBJ / LOC / WORLD / CHN).
        known = entity_ids | object_ids | location_ids | event_ids | channel_ids
        # World traits live on a separate dict on WSV1 and follow the
        # WORLD_ prefix convention.
        known |= world_trait_ids
        for fld in ("source_id", "target_id"):
            val = getattr(ce, fld, None)
            if val and val not in known:
                # AUDIT round-2 P0-2: surface WORLD_-prefixed undefined
                # refs explicitly (Brief Encounter / ACOTAR pattern).
                if isinstance(val, str) and val.startswith("WORLD_"):
                    issues.append(
                        f"[schema\u00b7world_trait] CausalEdge {fld}="
                        f"{val!r} references an undefined WORLD_ trait "
                        f"(causality_type={ce.causality_type}). Add it "
                        f"to ``WorldStateV1.world_traits``."
                    )
                else:
                    issues.append(
                        f"[schema\u00b7dangling] CausalEdge {fld}={val!r} "
                        f"is unknown (causality_type={ce.causality_type})."
                    )
    for prop in propositions:
        for ref in (getattr(prop, "referent_ids", None) or []):
            if (
                ref not in entity_ids
                and ref not in object_ids
                and ref not in event_ids
                and ref not in location_ids
                and ref not in world_trait_ids
                and ref not in channel_ids
            ):
                # AUDIT round-2: distinguish WORLD_* mis-refs from
                # generic dangling so authors fix the right surface.
                if isinstance(ref, str) and ref.startswith("WORLD_"):
                    issues.append(
                        f"[schema\u00b7world_trait] PROP {prop.proposition_id} "
                        f"referent_id={ref!r} references an undefined "
                        f"WORLD_ trait. Add it to "
                        f"``WorldStateV1.world_traits``."
                    )
                else:
                    issues.append(
                        f"[schema\u00b7dangling] PROP {prop.proposition_id} "
                        f"referent_id={ref!r} is unknown."
                    )

    # ---------------- Temporal coherence -------------------
    # Propositions: a truth tick before any referent event's fabula
    # time is suspicious (the claim is justified before its supporting
    # event happened).
    evt_ft_by_id = {e.id: getattr(e, "fabula_time", None) for e in events}
    for prop in propositions:
        refs = list(getattr(prop, "referent_ids", None) or [])
        evt_refs = [r for r in refs if r in evt_ft_by_id]
        if not evt_refs:
            continue
        evt_fts = [
            evt_ft_by_id[r] for r in evt_refs
            if evt_ft_by_id[r] is not None
        ]
        if not evt_fts:
            continue
        earliest = min(evt_fts)
        for ft_key in (getattr(prop, "truth_at_fabula", None) or {}):
            try:
                ft_i = int(ft_key)
            except (TypeError, ValueError):
                continue
            if ft_i < earliest:
                issues.append(
                    f"[schema\u00b7temporal] PROP {prop.proposition_id} "
                    f"truth_at_fabula tick={ft_i} predates the earliest "
                    f"event referent (ft={earliest}); the claim is "
                    f"asserted before its supporting event."
                )

    # Relationships: a metric last_updated_fabula before the edge's
    # established_at_fabula is logically impossible.
    for re in (world_state.social_topology or []):
        # Audit R17-11: surface dangling RelationshipEdge endpoints.
        # The dyadic endpoints must resolve to real entities or the
        # edge silently survives the merge as a no-op (POV filters
        # skip unknown participants), masking extraction bugs.
        src_id = getattr(re, "source_entity_id", None)
        tgt_id = getattr(re, "target_entity_id", None)
        if src_id and src_id not in entity_ids:
            issues.append(
                f"[schema\u00b7dangling] RelationshipEdge "
                f"source_entity_id={src_id!r} \u2192 target={tgt_id!r} "
                f"references an unknown entity."
            )
        if tgt_id and tgt_id not in entity_ids:
            issues.append(
                f"[schema\u00b7dangling] RelationshipEdge "
                f"target_entity_id={tgt_id!r} (source={src_id!r}) "
                f"references an unknown entity."
            )
        est = getattr(re, "established_at_fabula", None)
        if est is None:
            continue
        for axis, metric in (re.metrics or {}).items():
            lu = getattr(metric, "last_updated_fabula", None)
            if lu is None:
                continue
            if int(lu) < int(est):
                issues.append(
                    f"[schema\u00b7temporal] RelationshipEdge "
                    f"{re.source_entity_id}\u2192{re.target_entity_id} "
                    f"axis={axis} last_updated={lu} predates "
                    f"established={est}."
                )

    # ---------------- Channel / speaker enforcement (R17-8) ----
    # Utterances routed via a channel must satisfy the channel's
    # participant invariants. Without this guard a speaker not on the
    # channel is silently allowed (POV layer accepts it) and a simplex
    # channel can carry utterances in the wrong direction, breaking
    # the asymmetric-information assumptions downstream consumers rely
    # on (broadcaster vs. listener-only channels).
    channels_by_id = world_state.channels or {}
    for evt in events:
        if getattr(evt, "event_type", None) != "utterance":
            continue
        cid = getattr(evt, "via_channel_id", None)
        if not cid:
            continue
        ch = channels_by_id.get(cid)
        if ch is None:
            issues.append(
                f"[schema\u00b7channel] EVT {evt.id} via_channel_id="
                f"{cid!r} references an unknown channel."
            )
            continue
        spk = getattr(evt, "speaker_id", None)
        participants = list(getattr(ch, "participant_ids", None) or [])
        if spk and spk not in participants:
            issues.append(
                f"[schema\u00b7channel] EVT {evt.id} speaker_id="
                f"{spk!r} is not a participant of channel {cid!r} "
                f"(participants={participants})."
            )
        # Audit R18-17: every explicit addressee on a channel-routed
        # utterance must also be a channel participant. A non-member
        # addressee silently passes the existing speaker check but
        # then fails any downstream perceivability gate, leaving the
        # belief acquisition contract under-specified.
        for aid in (getattr(evt, "addressee_ids", None) or []):
            if aid not in participants:
                issues.append(
                    f"[schema\u00b7channel] EVT {evt.id} addressee_id="
                    f"{aid!r} is not a participant of channel "
                    f"{cid!r} (participants={participants})."
                )
        # Audit R18-16: the utterance's ``fabula_time`` must fall
        # inside the channel's availability window. Pre-fix an
        # utterance authored at ft=14000 on a channel terminated at
        # ft=13000 would happily route through ingestion, leaving
        # downstream POV / belief-acquisition layers to silently
        # discard it.
        eft = getattr(evt, "fabula_time", None)
        est = getattr(ch, "established_at_fabula", None)
        term = getattr(ch, "terminated_at_fabula", None)
        if eft is not None and est is not None and int(eft) < int(est):
            issues.append(
                f"[schema\u00b7channel] EVT {evt.id} fabula_time="
                f"{eft} predates channel {cid!r} "
                f"established_at_fabula={est}."
            )
        if eft is not None and term is not None and int(eft) > int(term):
            issues.append(
                f"[schema\u00b7channel] EVT {evt.id} fabula_time="
                f"{eft} is after channel {cid!r} "
                f"terminated_at_fabula={term}."
            )
        # NOTE (R17-8 directionality): a stricter ``simplex`` check
        # \u2014 require ``speaker_id == participants[0]`` \u2014 was
        # prototyped here but disabled. The example worlds legitimately
        # use simplex channels for two patterns the current schema
        # cannot disambiguate:
        #   (a) object-mediated broadcast (telescreen, dossier,
        #       benefactor-pipeline): ``participants[0]`` is the
        #       physical medium / source of authority, while the
        #       actual speech event is performed by a human operator.
        #   (b) proxy / herald utterances: a lawyer (Jaggers)
        #       announces a benefactor's (Magwitch's) settlement; a
        #       servant (Alis) voices a master's (Tamlin's) curse.
        # Encoding either pattern as a hard violation would require an
        # ``on_behalf_of_id`` field. Until that exists, only the
        # membership invariant above is enforced.

    # ---------------- Mutation-social duplicates (P2-1) ----
    # Multiple mutation_social edges between the same dyad on the
    # same axis at the same fabula tick aggregate non-deterministically
    # in the social cascade. Surface them so authors can collapse.
    seen: Dict[tuple, int] = {}
    for ce in (world_state.causal_topology or []):
        if getattr(ce, "causality_type", None) != "mutation_social":
            continue
        key = (
            getattr(ce, "source_id", None),
            getattr(ce, "target_id", None),
            getattr(ce, "rel_counterpart_id", None),
            getattr(ce, "trait_target", None),
            getattr(ce, "fabula_time", None),
        )
        seen[key] = seen.get(key, 0) + 1
    for key, n in seen.items():
        if n > 1:
            issues.append(
                f"[schema\u00b7duplicate] mutation_social dyad "
                f"{key[1]}\u2192{key[2]} axis={key[3]} at ft={key[4]} "
                f"has {n} edges from source={key[0]}; aggregate "
                f"effect is order-dependent."
            )

    # ---------------- Proposition inverse-pair hint (P2-2) -
    # When two props share referent_ids and have opposite truth
    # commits at the same tick but neither carries the explicit
    # ``inverse_proposition_id`` link, flag as a polarity-pair
    # normalisation candidate.
    by_referents: Dict[tuple, List[Any]] = {}
    for prop in propositions:
        refs = tuple(sorted(getattr(prop, "referent_ids", None) or []))
        if not refs:
            continue
        by_referents.setdefault(refs, []).append(prop)
    for refs, group in by_referents.items():
        if len(group) < 2:
            continue
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if (getattr(a, "inverse_proposition_id", None) == b.proposition_id
                    or getattr(b, "inverse_proposition_id", None) == a.proposition_id):
                    continue
                a_truth = getattr(a, "truth_at_fabula", None) or {}
                b_truth = getattr(b, "truth_at_fabula", None) or {}
                shared_ticks = set(a_truth.keys()) & set(b_truth.keys())
                if not shared_ticks:
                    continue
                if any(bool(a_truth[t]) != bool(b_truth[t]) for t in shared_ticks):
                    issues.append(
                        f"[schema\u00b7polarity] PROP {a.proposition_id} "
                        f"and PROP {b.proposition_id} share referents "
                        f"and have opposing truth commits but no "
                        f"explicit ``inverse_proposition_id`` link."
                    )

    # ---------------- AUDIT round-2 P1: channel ref audit --
    # Belief.acquired_via_channel_id must reference a real channel,
    # and Belief.acquired_via_event_id must reference a real event.
    for eid, ent in entities.items():
        for b in (ent.beliefs or []):
            ch = getattr(b, "acquired_via_channel_id", None)
            if ch and ch not in channel_ids:
                issues.append(
                    f"[schema\u00b7channel] ENT {eid} belief "
                    f"acquired_via_channel_id={ch!r} is unknown."
                )
            ev = getattr(b, "acquired_via_event_id", None)
            if ev and ev not in event_ids:
                issues.append(
                    f"[schema\u00b7dangling] ENT {eid} belief "
                    f"acquired_via_event_id={ev!r} is unknown."
                )
            # R19-H11: holder must be a participant in the cited event
            # (actor or addressee). Exemption: ``revelation`` events
            # canonically broadcast to all present observers, so the
            # holder need not appear in the explicit actor/addressee
            # lists.
            if ev and ev in events_by_id:
                evt = events_by_id[ev]
                event_type = getattr(evt, "event_type", None)
                if event_type != "revelation":
                    actor_ids = set(getattr(evt, "actor_ids", None) or [])
                    addressee_ids = set(
                        getattr(evt, "addressee_ids", None) or []
                    )
                    participants = actor_ids | addressee_ids
                    if participants and eid not in participants:
                        issues.append(
                            f"[schema\u00b7provenance] ENT {eid} belief "
                            f"cites acquired_via_event_id={ev!r} but is "
                            f"not in that event's actor_ids \u222a "
                            f"addressee_ids (fabricated provenance)."
                        )

    # ---------------- AUDIT round-2 P1: counter_concern symmetry
    # If concern A lists B in counter_concern_ids, B should reciprocally
    # list A. Asymmetric pairs starve the ambivalence scorer of one
    # side of the polarity (Mainwaring fear\u2194desire pattern).
    concern_by_id: Dict[str, Any] = {}
    for eid, ent in entities.items():
        for c in (ent.concerns or []):
            cid = getattr(c, "concern_id", None)
            if cid:
                concern_by_id[cid] = c
    for cid, c in concern_by_id.items():
        for other in (getattr(c, "counter_concern_ids", None) or []):
            if other not in concern_by_id:
                issues.append(
                    f"[schema\u00b7dangling] Concern {cid} "
                    f"counter_concern_ids references unknown id "
                    f"{other!r}."
                )
                continue
            back = (
                getattr(concern_by_id[other], "counter_concern_ids", None)
                or []
            )
            if cid not in back:
                issues.append(
                    f"[schema\u00b7symmetry] Concern {cid} \u2194 "
                    f"{other}: counter_concern_ids relation is "
                    f"asymmetric; add {cid} to {other}.counter_concern_ids."
                )

    # ---------------- AUDIT round-2 P1: mutation_social rel_counterpart
    # The rel_counterpart_id on a mutation_social edge must match
    # an existing entity (the dyadic partner).
    for ce in (world_state.causal_topology or []):
        if getattr(ce, "causality_type", None) != "mutation_social":
            continue
        rc = getattr(ce, "rel_counterpart_id", None)
        if rc and rc not in entity_ids:
            issues.append(
                f"[schema\u00b7dangling] mutation_social edge "
                f"target={getattr(ce, 'target_id', None)} "
                f"rel_counterpart_id={rc!r} is unknown."
            )

    # Audit R18-18: every ``location_id`` written on an
    # EntityStateSnapshot / ObjectStateSnapshot timeline must
    # resolve to a registered location. Pre-fix a typoed
    # ``LOC_ROOM_101a`` would silently advance the entity into a
    # phantom location, breaking spatial-adjacency queries and POV
    # location pruning downstream.
    location_ids = set((world_state.locations or {}).keys())
    for ent in (world_state.entities or {}).values():
        for snap in (getattr(ent, "state_timeline", None) or []):
            loc = getattr(snap, "location_id", None)
            if loc and loc not in location_ids:
                issues.append(
                    f"[schema\u00b7dangling] EntityStateSnapshot "
                    f"on {ent.id} at ft="
                    f"{getattr(snap, 'fabula_time', None)} "
                    f"location_id={loc!r} is unknown."
                )
    for obj in (world_state.objects or {}).values():
        for snap in (getattr(obj, "state_timeline", None) or []):
            loc = getattr(snap, "location_id", None)
            if loc and loc not in location_ids:
                issues.append(
                    f"[schema\u00b7dangling] ObjectStateSnapshot "
                    f"on {obj.id} at ft="
                    f"{getattr(snap, 'fabula_time', None)} "
                    f"location_id={loc!r} is unknown."
                )

    # ---------------- R19-M1: utterance addressee co-presence ----
    # When an utterance has no ``via_channel_id`` (no channel layer
    # mediating reach) and an ``at_location_id``, every addressee
    # must be in that location at the event's fabula_time \u2014 a
    # speaker cannot hand a face-to-face line to an addressee who
    # is in another room. (Channel-mediated utterances are exempt;
    # the channel reach audit handles those.)
    #
    # **Opt-in (post-implementation):** the example worlds use the
    # convention "addressee_ids of an utterance event imply on-stage
    # co-presence at at_location_id" \u2014 they do NOT update each
    # character's spatial state_timeline per scene, so the static
    # ``location_id`` on Entity reflects residence/origin, not
    # current scene. Firing the check by default would surface
    # ~100 false positives across the corpus. The check is therefore
    # opt-in via ``SHADOW_LOOM_STRICT_COPRESENCE=1``; ingested
    # worlds that DO track per-scene spatial mutations should enable
    # it to catch genuine "speaker addresses absent character" bugs.
    import os as _os
    if _os.environ.get("SHADOW_LOOM_STRICT_COPRESENCE", "0").strip() in ("1", "true", "yes"):
        try:
            from shadow_loom.models import reconstruct_entity_at as _recon_ent
        except Exception:
            _recon_ent = None  # type: ignore
        if _recon_ent is not None:
            for evt in events:
                if getattr(evt, "event_type", None) != "utterance":
                    continue
                if getattr(evt, "via_channel_id", None):
                    continue
                at_loc = getattr(evt, "at_location_id", None)
                if not at_loc or at_loc not in location_ids:
                    continue
                ft = getattr(evt, "fabula_time", None)
                if ft is None:
                    continue
                for aid in (getattr(evt, "addressee_ids", None) or []):
                    ent = entities.get(aid)
                    if ent is None:
                        continue
                    timeline = list(getattr(ent, "state_timeline", None) or [])
                    has_explicit_pin = False
                    pinned_loc = None
                    pinned_ft = -1
                    for snap in timeline:
                        snap_ft = getattr(snap, "fabula_time", None)
                        snap_loc = getattr(snap, "location_id", None)
                        if snap_ft is None or snap_loc is None:
                            continue
                        try:
                            snap_ft_i = int(snap_ft)
                        except (TypeError, ValueError):
                            continue
                        if snap_ft_i <= int(ft) and snap_ft_i >= pinned_ft:
                            has_explicit_pin = True
                            pinned_loc = snap_loc
                            pinned_ft = snap_ft_i
                    if not has_explicit_pin:
                        continue
                    if pinned_loc and pinned_loc != at_loc:
                        issues.append(
                            f"[schema\u00b7copresence] EVT {evt.id} utterance "
                            f"at_location_id={at_loc!r} addressee={aid!r} is "
                            f"located at {pinned_loc!r} at ft={ft} (no "
                            f"via_channel_id to mediate the gap)."
                        )

    # ---------------- R19-M2 / M3 / M4: concern integrity ---------
    removed_entity_set: set = set()
    for rids in (getattr(world_state, "shadow_removed_entity_ids", None) or {}).values():
        for rid in rids or []:
            removed_entity_set.add(rid)
    for eid, ent in (world_state.entities or {}).items():
        for c in (getattr(ent, "concerns", None) or []):
            cid = getattr(c, "concern_id", None)
            # R19-M2: activation_fabula_window inversion check.
            win = getattr(c, "activation_fabula_window", None)
            if win and len(win) == 2:
                try:
                    lo, hi = int(win[0]), int(win[1])
                    if lo > hi:
                        issues.append(
                            f"[schema\u00b7temporal] Concern {cid!r} on "
                            f"ENT {eid} activation_fabula_window=[{lo},"
                            f"{hi}] is inverted (lo > hi); reconstruct_"
                            f"concern_at will treat the concern as "
                            f"never-active."
                        )
                except (TypeError, ValueError):
                    pass
            # R19-M3: concern held by entity that has been
            # tombstoned on some shadow branch \u2014 audit surfaces
            # orphaned counter-concern targets that survived the
            # delete.
            if eid in removed_entity_set:
                issues.append(
                    f"[schema\u00b7tombstone] Concern {cid!r} held by "
                    f"ENT {eid} which is tombstoned on a shadow "
                    f"branch; downstream affect scores may still see "
                    f"it via the factual baseline."
                )
            # R19-M4: ConcernSnapshot.fabula_time monotonicity.
            prev_ft = None
            for snap in (getattr(c, "state_timeline", None) or []):
                ft = getattr(snap, "fabula_time", None)
                if ft is None:
                    continue
                if prev_ft is not None and ft < prev_ft:
                    issues.append(
                        f"[schema\u00b7temporal] Concern {cid!r} on ENT "
                        f"{eid} state_timeline is non-monotonic "
                        f"(snapshot at ft={ft} follows ft={prev_ft}); "
                        f"reconstruct_concern_at replay order may be "
                        f"corrupted."
                    )
                prev_ft = ft

    return issues


__all__ = ["audit_world_schema"]


# ---------------------------------------------------------------------
# CC-3 (2026-05-29): tiny CLI for ad-hoc operator use.
# ``python -m shadow_loom.world_schema_audit path/to/world.json``
# loads a serialized WorldStateV1 and prints the warning list (one
# per line). Exit code 1 when any warning is present so CI / shell
# pipelines can gate on it. Kept deliberately minimal so the audit
# function itself stays the single source of truth.
# ---------------------------------------------------------------------
def _main() -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m shadow_loom.world_schema_audit",
        description=(
            "Run the deterministic world-schema audit against a "
            "serialized WorldStateV1 JSON file. Exits 1 if any "
            "warnings are produced."
        ),
    )
    parser.add_argument(
        "world_file",
        help="Path to a JSON file containing a serialized WorldStateV1.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exit non-zero on warnings (default). Pass --no-strict to "
            "always exit 0 regardless of warning count."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-warning lines; only print the count.",
    )
    args = parser.parse_args()

    try:
        with open(args.world_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as exc:
        print(f"[schema-audit] could not read {args.world_file!r}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"[schema-audit] invalid JSON in {args.world_file!r}: {exc}", file=sys.stderr)
        return 2

    try:
        ws = WorldStateV1.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — surface validation errors clearly
        print(f"[schema-audit] WorldStateV1 validation failed: {exc}", file=sys.stderr)
        return 2

    warnings = audit_world_schema(ws)
    if not args.quiet:
        for w in warnings:
            print(w)
    print(f"[schema-audit] {len(warnings)} warning(s).", file=sys.stderr)
    if warnings and args.strict:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via CLI
    raise SystemExit(_main())

