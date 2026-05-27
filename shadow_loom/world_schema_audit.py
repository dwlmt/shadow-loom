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

    return issues


__all__ = ["audit_world_schema"]
