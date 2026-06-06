# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Affect unification layer — one belief substrate, four affects.

Implements the design captured in
``/memories/repo/affect-unification-plan.md``: every
narrative-affect score (suspense, surprise, dramatic irony, mystery)
is a query over a single time-indexed belief tensor

    BeliefState[agent, proposition, t] -> confidence in [0, 1]

This module provides:

  * :func:`synthesise_propositions` — derive a ``Proposition`` per
    ``EventNode`` (and implicit outcomes), populating
    ``WorldStateV1.propositions``.
  * :func:`synthesise_audience_entity` — build the reserved
    ``ENT_AUDIENCE`` entity whose ``state_timeline`` records the
    audience's belief deltas as the syuzhet stream is consumed.
  * :class:`BeliefState` — read-side wrapper over agents' beliefs
    with a single ``confidence(agent_id, proposition_id, fabula_t)``
    accessor, used by the four scorers.
  * :func:`compute_unified_affects` — Brewer-Lichtenstein/Zillmann
    suspense, Itti-Baldi surprise, Pfister/Sternberg per-character
    irony, and Carroll erotetic mystery, all derived from the same
    ``BeliefState`` so they cannot drift.

The legacy scorers in ``directive_assembly.py`` remain the primary
path; this module runs in parallel so the unified scores can be
cross-validated against them on the example_worlds before we
deprecate the legacy paths (Step 5 in the plan).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from shadow_loom.models import (
    Belief,
    Concern,
    Entity,
    EntityStateSnapshot,
    EventNode,
    Proposition,
    TraitVector,
    WorldStateV1,
    reconstruct_concern_at,
    reconstruct_entity_at,
    reconstruct_proposition_at,
)

_logger = logging.getLogger(__name__)

#: Reserved entity id for the synthesised audience agent.
AUDIENCE_ID = "ENT_AUDIENCE"

#: Audience prior confidence in an unrevealed event proposition.
_AUDIENCE_PRIOR_DEFAULT: float = 0.5

#: Confidence applied to the audience belief once an event is
#: directly narrated (revelation / outcome).
_AUDIENCE_REVEALED_CONFIDENCE: float = 1.0

#: Numerical clamp for log/KL terms — keeps surprise and irony finite
#: when a belief crosses 0.0 or 1.0 exactly.
_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Fabula-time-aware field resolvers
#
# Propositions and Concerns now carry a ``state_timeline`` so their
# mutable framing fields (``stakes`` / ``audience_default_prior`` /
# ``salience`` / ``polarity`` / ``activation_fabula_window``) can
# evolve over the story the same way ``Entity.state_timeline`` and
# ``GlobalTrait.state_timeline`` already do. These thin helpers wrap
# :func:`reconstruct_proposition_at` / :func:`reconstruct_concern_at`
# with a fast static-fallback so callsites stay one-line and cheap
# when no snapshots are present (the overwhelmingly common case for
# pre-existing example_worlds).
# ---------------------------------------------------------------------------

def _prop_stakes_at(prop: Proposition, fabula_t: int) -> float:
    """Return ``prop.stakes`` resolved at *fabula_t*.

    Falls back to the static ``prop.stakes`` when no
    ``PropositionSnapshot`` entries exist (cheap path).
    """
    if not prop.state_timeline:
        return float(prop.stakes)
    return float(reconstruct_proposition_at(prop, fabula_t)["stakes"])


def _prop_audience_prior_at(prop: Proposition, fabula_t: int) -> float:
    """Return ``prop.audience_default_prior`` resolved at *fabula_t*."""
    if not prop.state_timeline:
        return float(prop.audience_default_prior)
    return float(reconstruct_proposition_at(prop, fabula_t)["audience_default_prior"])


def _proposition_truth_at(prop: Proposition, fabula_t: int) -> Optional[bool]:
    """Return the proposition's committed ground truth effective at *fabula_t*.

    Walks ``truth_at_fabula`` and returns the value of the latest commit
    at or before *fabula_t*. Returns ``None`` when no commit has landed
    yet (the truth is still undetermined for the audience). Keys may
    arrive str-typed after a JSON / DB round-trip (see the M6 audit
    note), so they are coerced defensively.
    """
    best_t: Optional[int] = None
    best_v: Optional[bool] = None
    for _t, _v in prop.truth_at_fabula.items():
        try:
            _ti = int(_t)
        except (TypeError, ValueError):
            continue
        if _ti <= fabula_t and (best_t is None or _ti > best_t):
            best_t = _ti
            best_v = bool(_v)
    return best_v


def _concern_salience_at(c: Concern, fabula_t: int) -> float:
    """Return ``c.salience`` resolved at *fabula_t*."""
    if not c.state_timeline:
        return float(c.salience)
    return float(reconstruct_concern_at(c, fabula_t)["salience"])


def _concern_polarity_at(c: Concern, fabula_t: int) -> str:
    """Return ``c.polarity`` resolved at *fabula_t*."""
    if not c.state_timeline:
        return c.polarity
    return reconstruct_concern_at(c, fabula_t)["polarity"]


def _concern_window_at(
    c: Concern, fabula_t: int,
) -> Optional[Tuple[int, int]]:
    """Return ``c.activation_fabula_window`` resolved at *fabula_t*."""
    if not c.state_timeline:
        return c.activation_fabula_window
    return reconstruct_concern_at(c, fabula_t)["activation_fabula_window"]


# ---------------------------------------------------------------------------
# Step 1/3: proposition synthesis
# ---------------------------------------------------------------------------

def _stakes_for_event(
    evt: EventNode, force_by_target: Dict[str, float],
) -> float:
    """Per-event stakes proxy: max incoming causal_force, clamped to [0, 1]."""
    f = force_by_target.get(evt.id, 0.0)
    return max(0.0, min(1.0, f))


def _kind_for_event(evt: EventNode) -> str:
    """Map an EventNode onto a Proposition.kind.

    ``choice``-typed events become ``outcome`` (Brewer-Lichtenstein
    open question whose resolution drives suspense). Everything else
    is an ``event_occurs`` proposition whose audience confidence
    transitions from prior to ~1.0 at the syuzhet reveal step.
    """
    if evt.event_type == "choice":
        return "outcome"
    return "event_occurs"


# ---------------------------------------------------------------------------
# Generic timeline coalescence
#
# Phase C reconciler helper. Folds a list of snapshot Pydantic models
# (``PropositionSnapshot`` / ``ConcernSnapshot`` / any future BaseModel
# whose mutable fields are all ``Optional`` diffs) into a deterministic
# fabula-time-sorted list with two cleanups:
#
#   1. Drops entries whose every diff field is ``None`` (no-op
#      snapshots wasting tokens or replay cycles).
#   2. Coalesces same-(fabula_time, triggered_by) entries into a single
#      snapshot whose fields take the *first non-None* value across the
#      group — the catalogue agent and per-chunk affect agent should
#      not produce duplicates, but parallel chunks with overlapping
#      events occasionally both observe the same beat; merging keeps
#      replay deterministic.
#
# Callers pass ``diff_fields`` so the helper does not need to introspect
# the schema. Same shape as the existing :func:`_coalesce_snapshots` in
# ingestion.py (which is specialised to ``EntityStateSnapshot``); the
# two implementations stay separate because EntityStateSnapshot's merge
# rules (beliefs union, traits-by-key, etc.) are richer than the simple
# first-non-None rule that suffices for proposition / concern framing.
# ---------------------------------------------------------------------------


def _coalesce_timeline(
    snaps: list,
    diff_fields: Tuple[str, ...],
) -> list:
    """Deterministically coalesce a list of diff-only snapshot models.

    *snaps*: a list of Pydantic models all of the same class. Each
    must expose ``fabula_time: int``, ``triggered_by: Optional[str]``,
    and the names in *diff_fields* as ``Optional[...]`` attributes.

    Returns a new list, sorted by fabula_time (then by ``triggered_by``
    for determinism). No-op snapshots (every diff field None) are
    dropped. Same-(fabula_time, triggered_by) entries collapse into
    a single snapshot whose diff fields take the first non-None value
    in input order.
    """
    if not snaps:
        return []

    def _is_noop(s) -> bool:
        return all(getattr(s, f, None) is None for f in diff_fields)

    cleaned = [s for s in snaps if not _is_noop(s)]
    if not cleaned:
        return []

    grouped: Dict[Tuple[int, Optional[str]], list] = {}
    order: List[Tuple[int, Optional[str]]] = []
    for s in cleaned:
        key = (s.fabula_time, getattr(s, "triggered_by", None))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(s)

    out: list = []
    for key in order:
        group = grouped[key]
        if len(group) == 1:
            out.append(group[0])
            continue
        merged: Dict[str, object] = {}
        for f in diff_fields:
            for s in group:
                v = getattr(s, f, None)
                if v is not None:
                    merged[f] = v
                    break
        out.append(group[0].model_copy(update=merged))

    out.sort(key=lambda s: (s.fabula_time, getattr(s, "triggered_by", "") or ""))
    return out


def synthesise_propositions(world: WorldStateV1) -> List[Proposition]:
    """Populate ``world.propositions`` with one ``Proposition`` per event.

    Idempotent: if propositions already exist with PROP_FROM_EVT_*
    ids they are not duplicated. Returns the full proposition list
    on the world after synthesis.
    """
    existing_ids = {p.proposition_id for p in world.propositions}

    # Pre-compute max incoming causal_force per target for stakes.
    force_by_target: Dict[str, float] = {}
    for ce in world.causal_topology:
        cur = force_by_target.get(ce.target_id, 0.0)
        if ce.causal_force > cur:
            force_by_target[ce.target_id] = ce.causal_force

    new_props: List[Proposition] = []
    for evt in world.events:
        prop_id = f"PROP_FROM_{evt.id}"
        if prop_id in existing_ids:
            continue
        prop = Proposition(
            proposition_id=prop_id,
            kind=_kind_for_event(evt),  # type: ignore[arg-type]
            referent_ids=[evt.id],
            description=evt.description,
            audience_default_prior=_AUDIENCE_PRIOR_DEFAULT,
            stakes=max(0.1, _stakes_for_event(evt, force_by_target)),
            truth_at_fabula={evt.fabula_time: True},
        )
        new_props.append(prop)
        existing_ids.add(prop_id)

    if new_props:
        world.propositions = list(world.propositions) + new_props

    return world.propositions


# ---------------------------------------------------------------------------
# Step 4: ENT_AUDIENCE synthesis
# ---------------------------------------------------------------------------

def _audience_seed_entity() -> Entity:
    """Return a freshly-seeded ENT_AUDIENCE entity with an empty timeline.

    ``location_id`` is set to the sentinel ``"LOC_NONE"`` because
    :class:`Entity` requires a non-null ``location_id``. The
    :func:`synthesise_audience_entity` caller registers a matching
    sentinel ``Location`` in ``world.locations`` so the broken-link
    validator stays satisfied. The audience-agent is an abstract
    reader-perspective and is not anywhere in the spatial graph;
    consumers should treat ``LOC_NONE`` as "outside the diegesis"
    and skip it when computing spatial proximity.
    """
    return Entity(
        node_type="Entity",
        id=AUDIENCE_ID,
        name="Audience",
        location_id="LOC_NONE",
        status="healthy",
        traits={},
        beliefs=[],
        constants=["audience_agent"],
        state_timeline=[],
    )


def _audience_seed_location() -> "Location":
    """Return the sentinel ``LOC_NONE`` location used by ENT_AUDIENCE."""
    from shadow_loom.models import Location  # local to avoid cycles
    return Location(
        node_type="Location",
        id="LOC_NONE",
        name="(no location)",
        description=(
            "Sentinel non-place where the synthesised ENT_AUDIENCE "
            "agent lives. Outside the diegesis; not a real setting."
        ),
        ambient_state={},
    )


def backfill_character_belief_propositions(world: WorldStateV1) -> int:
    """Deterministic Step-3 back-fill: link character beliefs to propositions.

    For every ``Belief`` on every (non-audience) entity whose
    ``target_id`` is an event id we synthesised a proposition for,
    set ``proposition_id = f"PROP_FROM_{target_id}"`` if it's not
    already set. Walks both initial ``entity.beliefs`` and every
    ``EntityStateSnapshot.beliefs_added`` on the timeline.

    Returns the number of beliefs updated. Idempotent; safe to call
    multiple times. Does NOT touch beliefs whose target is an
    entity / object / location — those need the (optional, LLM-driven)
    string-clustering pass.
    """
    event_ids = {e.id for e in world.events}
    n = 0

    def _backfill(beliefs: List[Belief]) -> None:
        nonlocal n
        for b in beliefs:
            if b.proposition_id is not None:
                continue
            if b.target_id in event_ids:
                b.proposition_id = f"PROP_FROM_{b.target_id}"
                n += 1

    for eid, ent in world.entities.items():
        if eid == AUDIENCE_ID:
            continue
        _backfill(ent.beliefs)
        for snap in ent.state_timeline:
            _backfill(snap.beliefs_added)

    return n


def backfill_belief_propositions_by_referent(world: WorldStateV1) -> int:
    """Deterministic non-event belief→proposition backfill.

    For every (non-audience) ``Belief`` whose ``proposition_id`` is
    not yet set and whose ``target_id`` appears in **exactly one**
    proposition's ``referent_ids``, bind the belief to that
    proposition. Walks both initial ``entity.beliefs`` and every
    ``EntityStateSnapshot.beliefs_added`` on the timeline.

    Complements :func:`backfill_character_belief_propositions` (which
    only handles event-target beliefs). Together they spare the
    expensive LLM clustering pass for the unambiguous referent-match
    case (a belief about ENT_DUNCAN with a single PROP that lists
    ENT_DUNCAN as a referent — almost always the right link).

    The single-referent guard is deliberate: when two propositions
    share a referent (e.g. a relation_holds and an identity_is both
    naming the same character), we defer to the LLM clustering pass
    rather than guess. Returns the number of beliefs updated.
    Idempotent.
    """
    # Build target_id → [proposition_ids] index.
    by_referent: Dict[str, List[str]] = {}
    for prop in world.propositions:
        for rid in prop.referent_ids:
            by_referent.setdefault(rid, []).append(prop.proposition_id)

    n = 0

    def _backfill(beliefs: List[Belief]) -> None:
        nonlocal n
        for b in beliefs:
            if b.proposition_id is not None:
                continue
            candidates = by_referent.get(b.target_id, [])
            if len(candidates) == 1:
                b.proposition_id = candidates[0]
                n += 1

    for eid, ent in world.entities.items():
        if eid == AUDIENCE_ID:
            continue
        _backfill(ent.beliefs)
        for snap in ent.state_timeline:
            _backfill(snap.beliefs_added)

    return n


def synthesise_audience_entity(
    world: WorldStateV1,
    *,
    branch_world_id: str = "factual",
    branch_label: Optional[str] = None,
) -> Entity:
    """Synthesise (or refresh) ``ENT_AUDIENCE`` from the syuzhet stream.

    Walks the events in syuzhet order. For each event whose
    proposition becomes audience-known at that step, emits a
    ``Belief(proposition_id=..., confidence=1.0, inertia=1.0)`` in an
    ``EntityStateSnapshot`` keyed by the event's fabula_time. The
    audience entity is added to ``world.entities`` if absent and
    overwritten if present (idempotent rebuild).

    Mapping of event_type → audience-belief update:

      * ``revelation`` / ``outcome`` → confidence 1.0, inertia 1.0
        (narrator/world commits the fact).
      * ``utterance`` → confidence by ``truth_value``: ``true`` → 1.0,
        ``false`` → 0.1, ``unknown`` → 0.5, ``performative`` → no
        belief update.
      * ``choice`` → confidence 1.0 on the choice happening (the
        outcome proposition) but the *result* of the choice resolves
        only when its caused outcome event fires.
    """
    # Ensure propositions exist before synthesising audience beliefs.
    if not world.propositions:
        synthesise_propositions(world)

    # R19-H13: project to the requested branch so the audience entity
    # is rebuilt from the events/propositions visible on that branch
    # (not the canonical union). Today ``projected_for_branch`` forks
    # ``entities`` only; once R19-H14 lands and events/channels also
    # fork, this same call will start delivering branch-scoped events
    # without any further change here.
    try:
        scoped = world.projected_for_branch(
            branch_world_id=branch_world_id or "factual",
            branch_label=branch_label,
        )
    except Exception:
        scoped = world

    # Register the sentinel "no location" entry the audience entity
    # points at — keeps the broken-link validator satisfied without
    # forcing plot authors to declare it themselves.
    if "LOC_NONE" not in world.locations:
        world.locations["LOC_NONE"] = _audience_seed_location()

    audience = _audience_seed_entity()

    # Reverse index: event_id -> authored semantic propositions that
    # reference it. Authored / ingested worlds attach semantic
    # proposition ids (``PROP_DUNCAN_DEAD``) to events via
    # ``referent_ids`` rather than the synthetic ``PROP_FROM_{evt}``
    # convention. The unified surprise / mystery scorers read those
    # authored proposition_ids; keying the audience's beliefs onto the
    # *synthetic* ids (the legacy behaviour) left the audience's
    # knowledge invisible to those scorers, so every authored world
    # scored 0 mystery and 0 surprise even though the audience plainly
    # learns the facts as the syuzhet unfolds. Key onto the semantic
    # proposition when one is committed at the event's tick; fall back
    # to the synthetic id only when no authored proposition covers the
    # event (auto-synthesised worlds, or commits not yet landed).
    scoped_event_ids = {e.id for e in scoped.events}
    evt_to_semantic_props: Dict[str, List[Proposition]] = {}
    for _prop in scoped.propositions:
        if _prop.proposition_id.startswith("PROP_FROM_"):
            continue
        for _ref in (_prop.referent_ids or []):
            if _ref in scoped_event_ids:
                evt_to_semantic_props.setdefault(_ref, []).append(_prop)

    def _make_belief(prop_id: str, conf: float) -> Belief:
        return Belief(
            target_id=evt.id,
            perceived_state=evt.description,
            confidence=conf,
            inertia=1.0,
            established_at_fabula=evt.fabula_time,
            acquired_via_event_id=evt.id,
            # R19-H12: audience learns utterance content *through* the
            # channel the utterance routed over. Recording the channel
            # provenance lets downstream auditors flag audience
            # beliefs whose channel was later severed (and lets the
            # affective scorers reason about channel reach).
            acquired_via_channel_id=(
                getattr(evt, "via_channel_id", None)
                if evt.event_type == "utterance"
                else None
            ),
            evidence_strength="strong",
            proposition_id=prop_id,
        )

    # Emit beliefs in syuzhet order; the snapshot fabula_time is the
    # event's fabula_time (audience learns it at the moment of
    # narration, but the belief is *about* a fabula-time-stamped
    # proposition).
    by_syuzhet = sorted(scoped.events, key=lambda e: e.syuzhet_index)
    for evt in by_syuzhet:
        confidence: Optional[float]
        if evt.event_type in ("revelation", "outcome", "choice"):
            confidence = _AUDIENCE_REVEALED_CONFIDENCE
        elif evt.event_type == "utterance":
            tv = evt.truth_value
            if tv == "true":
                confidence = 1.0
            elif tv == "false":
                confidence = 0.1
            elif tv == "unknown":
                confidence = 0.5
            else:  # performative or None — no belief update
                confidence = None
        else:
            confidence = _AUDIENCE_REVEALED_CONFIDENCE

        if confidence is None:
            continue

        beliefs_added: List[Belief] = []
        for sp in evt_to_semantic_props.get(evt.id, ()):
            truth = _proposition_truth_at(sp, evt.fabula_time)
            if truth is None:
                # The event references the proposition but its ground
                # truth has not committed by this tick — don't fabricate
                # a direction for the audience.
                continue
            # Audience witnesses the event and learns the proposition's
            # committed value: high confidence when true, low (knows-it-
            # is-false) when false. ``confidence`` carries the reveal
            # strength (utterance reliability for spoken events).
            directed = confidence if truth else max(_EPS, 1.0 - confidence)
            beliefs_added.append(_make_belief(sp.proposition_id, directed))

        if not beliefs_added:
            # No authored proposition committed here — preserve the
            # legacy synthetic-id belief so auto-synthesised worlds and
            # the ``PROP_FROM_*`` reveal fallback keep working.
            beliefs_added.append(
                _make_belief(f"PROP_FROM_{evt.id}", confidence)
            )

        snap = EntityStateSnapshot(
            world_id=getattr(evt, "world_id", "factual") or "factual",
            fabula_time=evt.fabula_time,
            triggered_by=evt.id,
            beliefs_added=beliefs_added,
        )
        audience.state_timeline.append(snap)

    # ------------------------------------------------------------------
    # Commit-driven audience reveals for entity-anchored propositions.
    #
    # A proposition may attach to the timeline through its
    # ``truth_at_fabula`` ground-truth commits rather than through an
    # event in ``referent_ids`` — a common authoring style where the
    # proposition is *about* entities ("Heathcliff loves Catherine",
    # referent_ids=[ENT_HEATHCLIFF, ENT_CATHERINE]) and no single event
    # carries it. The syuzhet-ordered loop above only emits beliefs for
    # propositions reachable from a revealed event, so these
    # entity-anchored propositions would never get an audience belief.
    # Their confidence would then sit frozen at the static
    # ``audience_default_prior`` (``_prop_audience_prior_at`` does NOT
    # consult ``truth_at_fabula``), zeroing every surprise / mystery
    # score that reads audience-belief movement — exactly the symptom
    # seen on brief_encounter and wuthering_heights.
    #
    # Emit one belief snapshot per truth commit, gated at the commit's
    # own fabula tick so the audience "learns" the committed value when
    # the narrative reaches that fabula moment (``reconstruct_entity_at``
    # replays snapshots with ``fabula_time <= cursor``). Provenance is
    # tagged to the earliest event that carries the audience's
    # fabula-now to the commit, so counterfactual surgery can still
    # prune these beliefs if that event no longer fires.
    # ------------------------------------------------------------------
    events_by_fabula = sorted(
        (e for e in scoped.events if e.fabula_time is not None),
        key=lambda e: e.fabula_time,
    )

    def _provenance_event(commit_t: int):
        after = [e for e in events_by_fabula if e.fabula_time >= commit_t]
        if after:
            return min(after, key=lambda e: e.syuzhet_index)
        return events_by_fabula[-1] if events_by_fabula else None

    for prop in scoped.propositions:
        if prop.proposition_id.startswith("PROP_FROM_"):
            continue
        # Skip propositions the event-referent path already handled.
        if any(r in scoped_event_ids for r in (prop.referent_ids or [])):
            continue
        if not prop.truth_at_fabula:
            continue
        target = next(iter(prop.referent_ids or []), prop.proposition_id)
        for ct_raw, val in prop.truth_at_fabula.items():
            try:
                ct = int(ct_raw)
            except (TypeError, ValueError):
                continue
            prov = _provenance_event(ct)
            directed = (
                _AUDIENCE_REVEALED_CONFIDENCE
                if bool(val)
                else max(_EPS, 1.0 - _AUDIENCE_REVEALED_CONFIDENCE)
            )
            belief = Belief(
                target_id=target,
                perceived_state=prop.description,
                confidence=directed,
                inertia=1.0,
                established_at_fabula=ct,
                acquired_via_event_id=prov.id if prov else None,
                evidence_strength="strong",
                proposition_id=prop.proposition_id,
            )
            audience.state_timeline.append(
                EntityStateSnapshot(
                    world_id=(
                        (getattr(prov, "world_id", "factual") or "factual")
                        if prov
                        else "factual"
                    ),
                    fabula_time=ct,
                    triggered_by=prov.id if prov else None,
                    beliefs_added=[belief],
                )
            )

    # ------------------------------------------------------------------
    # Audience concern seeding.
    #
    # The OSS-extraction audit (2026-05-15) found that ENT_AUDIENCE
    # had zero concerns across every ingested plot, which silently
    # zeroed out every dramatic-irony / suspense / surprise score
    # that multiplies through audience concern salience. We seed one
    # concern per ``kind='outcome'`` proposition (the Brewer-
    # Lichtenstein open-question class that drives suspense) using
    # the proposition's own ``audience_default_prior`` and ``stakes``
    # to weight the per-entity ``salience``. Polarity is ``desire``
    # by default (the audience wants the open question resolved
    # *positively*); the affect agent is free to overlay a
    # ``ConcernSnapshot`` that flips polarity on a given fabula tick.
    # ------------------------------------------------------------------
    for prop in scoped.propositions:
        if prop.kind != "outcome":
            continue
        ccn_id = f"CCN_AUDIENCE_{prop.proposition_id[len('PROP_'):]}"
        # Salience = average of audience's prior interest and the
        # proposition's narrative stakes. Both are in [0, 1] so the
        # mean stays in range.
        salience = max(
            0.05,
            min(1.0, 0.5 * (prop.audience_default_prior + prop.stakes)),
        )
        audience.concerns.append(
            Concern(
                concern_id=ccn_id,
                proposition_id=prop.proposition_id,
                polarity="desire",
                salience=salience,
                kind="outcome",
            )
        )

    world.entities[AUDIENCE_ID] = audience
    return audience


# ---------------------------------------------------------------------------
# BeliefState reader
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """Time-indexed belief reader over a ``WorldStateV1``.

    One instance is built per scoring call. ``confidence(agent_id,
    proposition_id, fabula_t)`` returns the agent's belief in the
    proposition reconstructed at ``fabula_t``, falling back to the
    proposition's ``audience_default_prior`` when the agent holds no
    belief about it.
    """

    world: WorldStateV1
    _prop_index: Dict[str, Proposition] = field(default_factory=dict)
    _cache: Dict[Tuple[str, int], Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._prop_index = {p.proposition_id: p for p in self.world.propositions}

    def _agent_beliefs_at(
        self, agent_id: str, fabula_t: int,
    ) -> Dict[str, float]:
        """Return ``{proposition_id: confidence}`` for one agent at ``fabula_t``.

        Cached per (agent, fabula_t).
        """
        key = (agent_id, fabula_t)
        if key in self._cache:
            return self._cache[key]
        ent = self.world.entities.get(agent_id)
        if ent is None:
            self._cache[key] = {}
            return self._cache[key]
        recon = reconstruct_entity_at(ent, fabula_t)
        out: Dict[str, float] = {}
        for b in recon["beliefs"]:
            pid = b.get("proposition_id")
            if not pid:
                continue
            # If multiple beliefs exist for the same proposition the
            # most recent (last in list, since reconstruct appends in
            # fabula order) wins.
            out[pid] = float(b.get("confidence", 0.5))
        self._cache[key] = out
        return out

    def confidence(
        self, agent_id: str, proposition_id: str, fabula_t: int,
    ) -> float:
        """Return ``p_agent(P, t)`` clamped into ``[_EPS, 1 - _EPS]``."""
        beliefs = self._agent_beliefs_at(agent_id, fabula_t)
        if proposition_id in beliefs:
            p = beliefs[proposition_id]
        else:
            prop = self._prop_index.get(proposition_id)
            p = (
                _prop_audience_prior_at(prop, fabula_t)
                if prop
                else _AUDIENCE_PRIOR_DEFAULT
            )
        return max(_EPS, min(1.0 - _EPS, float(p)))

    def known_propositions(
        self, agent_id: str, fabula_t: int, threshold: float = 0.7,
    ) -> Set[str]:
        """Propositions the agent confidently believes at ``fabula_t``."""
        return {
            pid for pid, p in self._agent_beliefs_at(agent_id, fabula_t).items()
            if p >= threshold
        }


# ---------------------------------------------------------------------------
# Affect queries
# ---------------------------------------------------------------------------

def _binary_entropy(p: float) -> float:
    """Shannon entropy in nats for a Bernoulli variable with prob p."""
    p = max(_EPS, min(1.0 - _EPS, p))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _binary_kl(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)) in nats."""
    p = max(_EPS, min(1.0 - _EPS, p))
    q = max(_EPS, min(1.0 - _EPS, q))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def _auto_tau_fabula(world: WorldStateV1) -> float:
    """Median inter-event fabula-time gap, used as the imminence kernel scale.

    Mirrors the scaling logic used by the legacy
    ``compute_suspense_score`` so the unified suspense lives on the
    same time-units as the per-world event spacing rather than a
    hardcoded constant. Floored at 1.0.
    """
    fts = sorted({e.fabula_time for e in world.events})
    if len(fts) < 2:
        return 1.0
    gaps = [b - a for a, b in zip(fts, fts[1:]) if b > a]
    if not gaps:
        return 1.0
    gaps.sort()
    return max(1.0, float(gaps[len(gaps) // 2]))


def compute_suspense_unified(
    bs: BeliefState, fabula_t: int,
    *, tau_fabula: Optional[float] = None,
) -> float:
    """Brewer-Lichtenstein suspense: outcome-set entropy × stakes × imminence.

    Sums over open ``outcome`` propositions whose truth has not yet
    committed at ``fabula_t``: ``H(p_aud(P)) · stakes · exp(-dt/tau)``.
    ``tau_fabula`` defaults to the world's median inter-event gap.
    """
    if tau_fabula is None:
        tau_fabula = _auto_tau_fabula(bs.world)
    total = 0.0
    for prop in bs.world.propositions:
        # Originally restricted to ``kind == "outcome"`` (Brewer-
        # Lichtenstein open questions). In practice the auto-
        # synthesiser only tags ``event_type == "choice"`` events
        # as outcomes, leaving the bulk of dramatically uncertain
        # propositions (event_occurs, trait_holds, identity_is)
        # off the ledger and crushing the score on most worlds.
        # Treat any *uncommitted* proposition the audience is
        # uncertain about as an open outcome — the entropy term
        # naturally collapses to zero for propositions whose
        # audience confidence has already saturated, so this
        # widening costs nothing on already-resolved props but
        # rescues every unrevealed event/trait/identity beat.
        if prop.kind not in ("outcome", "event_occurs", "trait_holds", "identity_is"):
            continue
        # A proposition is "open" when there is at least one truth
        # commit STRICTLY in the future. Multi-commit propositions
        # (e.g. ``{18000: True, 20000: False}``) flip back and forth
        # so the prior commit at 18000 must NOT close the question
        # at 19000 — the impending reversal at 20000 keeps it open.
        # Only skip when every commit is at or before the cursor AND
        # there are no future commits left.
        # Audit (eighth pass, M6): ``truth_at_fabula`` keys are
        # typed ``int`` but JSON/DB round-trip outside the validator
        # can leave str keys. ``"19000" > 19000`` raises TypeError
        # in Python 3; coerce defensively before comparing.
        _ledger_ticks: List[int] = []
        for _t in prop.truth_at_fabula:
            try:
                _ledger_ticks.append(int(_t))
            except (TypeError, ValueError):
                continue
        future_commits = [t for t in _ledger_ticks if t > fabula_t]
        if not future_commits and any(
            t <= fabula_t for t in _ledger_ticks
        ):
            continue
        p_aud = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        # Imminence kernel — pull next future commitment time if any.
        if future_commits:
            dt = min(future_commits) - fabula_t
            imminence = math.exp(-dt / max(1.0, tau_fabula))
        else:
            imminence = 1.0
        total += _binary_entropy(p_aud) * _prop_stakes_at(prop, fabula_t) * imminence
    return total


def compute_surprise_unified(
    bs: BeliefState, fabula_t: int, prior_fabula_t: int,
) -> float:
    """Itti-Baldi Bayesian surprise on audience belief revision.

    Sums KL(p_aud(P, t) || p_aud(P, t-1)) × stakes across all
    propositions whose audience confidence moved.
    """
    total = 0.0
    for prop in bs.world.propositions:
        p_now = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        p_prev = bs.confidence(
            AUDIENCE_ID, prop.proposition_id, prior_fabula_t,
        )
        if abs(p_now - p_prev) < _EPS:
            continue
        total += _binary_kl(p_now, p_prev) * _prop_stakes_at(prop, fabula_t)
    return total


def compute_irony_unified(
    bs: BeliefState, focal_id: str, fabula_t: int,
) -> float:
    """Pfister / Sternberg dramatic irony: audience-vs-focal KL × stakes.

    Sums KL(p_aud(P, t) || p_focal(P, t)) × stakes across propositions
    where audience and focal disagree. KL is asymmetric, so audience-
    knows-more and focal-knows-more produce distinguishable signals.
    """
    if focal_id == AUDIENCE_ID:
        return 0.0
    total = 0.0
    for prop in bs.world.propositions:
        p_aud = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        p_focal = bs.confidence(focal_id, prop.proposition_id, fabula_t)
        if abs(p_aud - p_focal) < _EPS:
            continue
        total += _binary_kl(p_aud, p_focal) * _prop_stakes_at(prop, fabula_t)
    return total


def compute_mystery_unified(
    bs: BeliefState, fabula_t: int, threshold: float = 0.7,
) -> float:
    """Carroll erotetic mystery: entropy over hidden causes of known effects.

    For each effect proposition the audience confidently knows
    happened, sums Shannon entropy of the softmax-normalised
    causal_force distribution over its *unrevealed* ancestors in the
    causal graph.
    """
    # Build causal digraph (proposition-level by referent EventNode id).
    g = nx.DiGraph()
    for evt in bs.world.events:
        g.add_node(evt.id)
    for ce in bs.world.causal_topology:
        g.add_edge(ce.source_id, ce.target_id, weight=ce.causal_force)

    audience_known = bs.known_propositions(AUDIENCE_ID, fabula_t, threshold)
    # Map proposition -> referent event id (we synth one prop per evt,
    # so the mapping is direct).
    known_evt_ids = {
        bs._prop_index[pid].referent_ids[0]
        for pid in audience_known
        if pid in bs._prop_index and bs._prop_index[pid].referent_ids
    }
    if not known_evt_ids:
        return 0.0

    # Audit (eighth pass, M4): the original "unrevealed" check tested
    # ``f"PROP_FROM_{a}" not in audience_known`` — a synthetic id
    # convention used only by the auto-synthesiser. Real example
    # worlds (``macbeth``, ``gone_girl``, …) attach semantic
    # proposition ids (``PROP_DUNCAN_DEAD``) that reference event
    # ids via ``Proposition.referent_ids``, so every authored
    # ancestor was mis-classified as unrevealed → systematically
    # inflated mystery entropy. Build the reverse index
    # ``event_id -> {proposition_id}`` once and treat an ancestor
    # as revealed iff the audience confidently knows at least one
    # proposition referencing it.
    evt_to_prop_ids: Dict[str, Set[str]] = {}
    for _prop in bs.world.propositions:
        for _ref in (_prop.referent_ids or []):
            if isinstance(_ref, str) and _ref.startswith("EVT_"):
                evt_to_prop_ids.setdefault(_ref, set()).add(_prop.proposition_id)

    def _ancestor_revealed(_evt_id: str) -> bool:
        _props = evt_to_prop_ids.get(_evt_id) or set()
        # Synthetic-id fallback for legacy / auto-synthesised worlds.
        if f"PROP_FROM_{_evt_id}" in audience_known:
            return True
        return bool(_props & audience_known)

    total = 0.0
    for evt_id in known_evt_ids:
        if evt_id not in g:
            continue
        ancestors = list(nx.ancestors(g, evt_id))
        # Restrict to ancestors NOT yet known to the audience.
        unrevealed = [
            a for a in ancestors
            if not _ancestor_revealed(a)
        ]
        if len(unrevealed) < 2:
            continue
        # Edge weight from each unrevealed ancestor along its highest-
        # weight path *into* evt_id (reverse BFS over predecessors).
        weights: List[float] = []
        for a in unrevealed:
            try:
                path = nx.shortest_path(
                    g, source=a, target=evt_id, weight=lambda u, v, d: 1.0 / max(_EPS, d.get("weight", 0.5)),
                )
                # Path strength = product of edge weights.
                strength = 1.0
                for u, v in zip(path, path[1:]):
                    strength *= g.edges[u, v].get("weight", 0.5)
                weights.append(strength)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if len(weights) < 2:
            continue
        # Softmax-normalise then entropy.
        s = sum(weights)
        if s <= 0:
            continue
        probs = [w / s for w in weights]
        h = -sum(p * math.log(max(_EPS, p)) for p in probs)
        total += h
    return total


# ---------------------------------------------------------------------------
# Concern-conditioned helpers (Tan/Ortony, Pfister felicity, Ryan tellability)
# ---------------------------------------------------------------------------

def _concerns_for(
    world: WorldStateV1, entity_id: str, prop_id: str,
) -> List[Concern]:
    """All concerns of ``entity_id`` referencing ``prop_id``."""
    ent = world.entities.get(entity_id)
    if ent is None:
        return []
    return [c for c in ent.concerns if c.proposition_id == prop_id]


def _concern_salience(
    world: WorldStateV1, entity_id: str, prop_id: str,
) -> float:
    """Max salience over the entity's concerns about ``prop_id`` (else 0)."""
    cs = _concerns_for(world, entity_id, prop_id)
    return max((c.salience for c in cs), default=0.0)


def _concern_active_at(c: Concern, fabula_t: int) -> bool:
    """True iff ``c.activation_fabula_window`` (resolved at *fabula_t*)
    is None (always active) or contains *fabula_t* inclusive.
    """
    win = _concern_window_at(c, fabula_t)
    if not win:
        return True
    try:
        lo, hi = int(win[0]), int(win[1])
    except (TypeError, ValueError, IndexError):
        return True
    return lo <= fabula_t <= hi


def _concern_salience_t(
    world: WorldStateV1, entity_id: str, prop_id: str, fabula_t: int,
) -> float:
    """Audit (eighth pass, M3): time-aware variant of
    :func:`_concern_salience`. Reconstructs each concern at
    *fabula_t* (so snapshot-driven salience drift counts) and zeroes
    out concerns whose activation window does not include
    *fabula_t* (so closed / not-yet-open concerns can't weight
    affect scores).
    """
    cs = _concerns_for(world, entity_id, prop_id)
    if not cs:
        return 0.0
    return max(
        (_concern_salience_at(c, fabula_t) for c in cs
         if _concern_active_at(c, fabula_t)),
        default=0.0,
    )


def _concern_polarity_sign(
    world: WorldStateV1, entity_id: str, prop_id: str,
) -> int:
    """+1 if entity desires the proposition true, -1 if fears it, 0 if no concern.

    When both desire and fear concerns exist (ambivalence) the
    higher-salience polarity wins; ties resolve to 0.
    
    P1-FIX (P1-23): When ambivalent (both polarities present), the
    effective salience is min(desire, fear) to capture the conflict,
    not the winner. This prevents unilateral dominance when both
    emotions are strong.
    """
    cs = _concerns_for(world, entity_id, prop_id)
    if not cs:
        return 0
    desire = max((c.salience for c in cs if c.polarity == "desire"), default=0.0)
    fear = max((c.salience for c in cs if c.polarity == "fear"), default=0.0)
    
    # P1-23: Ambivalence uses min() not max() to capture the conflict strength
    if desire > 0.0 and fear > 0.0:
        # Ambivalent: both present, winner determined by larger but reduced by conflict
        if desire > fear:
            return +1
        elif fear > desire:
            return -1
        else:
            return 0
    elif desire > fear:
        return +1
    elif fear > desire:
        return -1
    return 0


def _concern_polarity_sign_t(
    world: WorldStateV1, entity_id: str, prop_id: str, fabula_t: int,
) -> int:
    """Audit (eighth pass, M3): time-aware variant of
    :func:`_concern_polarity_sign`. Snapshot-reconstructs polarity
    and salience at *fabula_t* and skips concerns whose activation
    window does not include *fabula_t*.
    """
    cs = [
        c for c in _concerns_for(world, entity_id, prop_id)
        if _concern_active_at(c, fabula_t)
    ]
    if not cs:
        return 0
    desire = max(
        (_concern_salience_at(c, fabula_t) for c in cs
         if _concern_polarity_at(c, fabula_t) == "desire"),
        default=0.0,
    )
    fear = max(
        (_concern_salience_at(c, fabula_t) for c in cs
         if _concern_polarity_at(c, fabula_t) == "fear"),
        default=0.0,
    )
    if desire > 0.0 and fear > 0.0:
        if desire > fear:
            return +1
        elif fear > desire:
            return -1
        return 0
    elif desire > fear:
        return +1
    elif fear > desire:
        return -1
    return 0


def classify_counterfactual_form(
    actual_satisfactions: Dict[str, int],
    counterfactual_satisfactions: Dict[str, int],
    *,
    salience_weights: Optional[Dict[str, float]] = None,
) -> str:
    """Phase-7 helper: Frye/Aristotle narrative-form classification of a
    Rung-3 counterfactual.

    Inputs are ``{concern_id: signed_satisfaction}`` maps where the
    sign convention is +1 if the concern is satisfied (desire ∧ true,
    or fear ∧ false), -1 if unsatisfied, 0 if undecidable.
    ``salience_weights`` is an optional ``{concern_id: salience}`` map
    that weights each concern's contribution; when omitted every
    concern weighs 1.0.

    Returns one of:
      * ``"tragic"`` — actual is concern-load-worse than counterfactual
        (Σ_actual − Σ_counterfactual < 0). Roese & Olson's
        upward-counterfactual ⇒ regret frame.
      * ``"comic"`` — actual is concern-load-better than counterfactual
        (Σ_actual − Σ_counterfactual > 0). Gilovich-Medvec downward
        relief frame.
      * ``"ironic"`` — sign is mixed across concerns (some flip +, some
        flip −) but the net is ≈ 0. The counterfactual rearranges the
        utility landscape rather than improving or worsening it.
      * ``"neutral"`` — no concern moved between the two worlds.
    """
    sw = salience_weights or {}
    moved = {
        cid for cid in (set(actual_satisfactions) | set(counterfactual_satisfactions))
        if actual_satisfactions.get(cid, 0) != counterfactual_satisfactions.get(cid, 0)
    }
    if not moved:
        return "neutral"

    deltas: List[float] = []
    for cid in moved:
        actual = actual_satisfactions.get(cid, 0)
        cf = counterfactual_satisfactions.get(cid, 0)
        w = float(sw.get(cid, 1.0))
        deltas.append(w * (actual - cf))

    net = sum(deltas)
    pos = any(d > 0 for d in deltas)
    neg = any(d < 0 for d in deltas)

    # Mixed-sign with near-zero net = ironic rearrangement.
    if pos and neg and abs(net) < 1e-6:
        return "ironic"
    if net < 0:
        return "tragic"
    if net > 0:
        return "comic"
    # Single-sign deltas summing to 0 (cancellation across same-direction
    # concerns of equal weight, e.g. both worlds satisfy one concern and
    # leave another unchanged) — treat as neutral.
    return "neutral"


def _tellability(world: WorldStateV1, prop_id: str) -> float:
    """Ryan tellability: sum of concern salience across ALL entities for ``prop_id``.

    A proposition many characters care about is more 'tellable' and
    should dominate the mystery accumulator.
    """
    total = 0.0
    for ent in world.entities.values():
        if ent.id == AUDIENCE_ID:
            continue
        for c in ent.concerns:
            if c.proposition_id == prop_id:
                total += c.salience
    return total


# ---------------------------------------------------------------------------
# Breakdown scorers — theory-grounded decompositions of the four affects
# ---------------------------------------------------------------------------

@dataclass
class SurpriseBreakdown:
    """Tan/Ortony valence-decomposed Bayesian surprise.

    ``pleasant_score`` and ``unpleasant_score`` partition the total
    KL-driven surprise by alignment with the focal entity's
    :class:`Concern` polarities — concern-aligned reveals (a desired
    proposition flipping toward true, or a feared one toward false)
    accumulate as 'pleasant' surprise; the opposite as 'unpleasant'.
    """
    total: float
    pleasant_score: float
    unpleasant_score: float
    per_focal_score: Dict[str, float]


def compute_surprise_breakdown(
    bs: BeliefState, fabula_t: int, prior_fabula_t: int,
    *, focal_id: Optional[str] = None,
    other_focal_ids: Optional[List[str]] = None,
) -> SurpriseBreakdown:
    """Decompose surprise into pleasant/unpleasant + per-focal slices.

    Theory: Tan (1996) / Ortony, Clore & Collins (1988) — surprise
    valence is a function of alignment between the belief revision
    and the experiencer's concerns. We treat the focal entity's
    concerns as the valence anchor; per-focal slices weight every
    KL contribution by the entity's concern salience for that
    proposition (so high-stakes-to-this-character revelations
    dominate their personal surprise score).
    """
    total = 0.0
    pleasant = 0.0
    unpleasant = 0.0
    per_focal: Dict[str, float] = {}
    focals = list(other_focal_ids or [])
    if focal_id and focal_id not in focals:
        focals.insert(0, focal_id)
    for fid in focals:
        per_focal.setdefault(fid, 0.0)

    for prop in bs.world.propositions:
        p_now = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        p_prev = bs.confidence(
            AUDIENCE_ID, prop.proposition_id, prior_fabula_t,
        )
        if abs(p_now - p_prev) < _EPS:
            continue
        kl = _binary_kl(p_now, p_prev) * _prop_stakes_at(prop, fabula_t)
        total += kl

        # Polarity (relative to focal). +1 = belief moved toward true.
        movement = +1 if p_now > p_prev else -1
        if focal_id is not None:
            # Audit (eighth pass, M3): time-aware polarity respects
            # snapshot drift and activation windows so a concern
            # closed before ``fabula_t`` (or one whose polarity
            # flipped via a snapshot) is scored against the right
            # state instead of the static initial field.
            sign = _concern_polarity_sign_t(
                bs.world, focal_id, prop.proposition_id, fabula_t,
            )
            # Aligned: focal desires true & moved toward true; or
            # focal fears true & moved toward false.
            if sign != 0:
                if sign * movement > 0:
                    pleasant += kl
                else:
                    unpleasant += kl

        for fid in focals:
            # Audit (eighth pass, M3): time-aware salience.
            sal = _concern_salience_t(
                bs.world, fid, prop.proposition_id, fabula_t,
            )
            per_focal[fid] = per_focal.get(fid, 0.0) + kl * sal

    return SurpriseBreakdown(
        total=total,
        pleasant_score=pleasant,
        unpleasant_score=unpleasant,
        per_focal_score=per_focal,
    )


# Sternberg three-mode irony: prop.kind buckets
_IRONY_SUSPENSE_KINDS = {"outcome"}
_IRONY_CURIOSITY_KINDS = {"identity_is", "relation_holds"}
_IRONY_SURPRISE_KINDS = {"event_occurs", "trait_holds"}


@dataclass
class IronyBreakdown:
    """Sternberg three-mode + Pfister felicity + Wall gradient decomposition.

    Theory:
      * Sternberg (1978) — dramatic irony partitions by temporal
        relation of the audience's privileged knowledge to the
        focal's: ``suspense_irony`` (audience knows the *outcome*),
        ``curiosity_irony`` (audience knows a *cause/identity*),
        ``surprise_irony`` (audience knows a fact the focal will
        soon discover).
      * Pfister (1977/1988) "felicity conditions" — irony only lands
        if the proposition is relevant to the focal's concerns;
        ``concern_weighted_score`` multiplies each per-prop KL by the
        focal's concern salience for that proposition.
      * Wall (1983) discrepant-awareness gradient — across all
        non-audience entities, identify the *most ironised*
        character (highest audience-advantage KL) so the renderer
        can spotlight them automatically.
    """
    audience_advantage: float
    focal_advantage: float
    suspense_irony_score: float
    curiosity_irony_score: float
    surprise_irony_score: float
    concern_weighted_score: float
    most_ironised_entity_id: Optional[str]
    most_ironised_score: float


def compute_irony_breakdown(
    bs: BeliefState, focal_id: str, fabula_t: int,
) -> IronyBreakdown:
    """See :class:`IronyBreakdown` for theory references."""
    if focal_id == AUDIENCE_ID:
        return IronyBreakdown(
            audience_advantage=0.0, focal_advantage=0.0,
            suspense_irony_score=0.0, curiosity_irony_score=0.0,
            surprise_irony_score=0.0, concern_weighted_score=0.0,
            most_ironised_entity_id=None, most_ironised_score=0.0,
        )

    aud_adv = 0.0
    foc_adv = 0.0
    sus_irony = 0.0
    cur_irony = 0.0
    sur_irony = 0.0
    concern_weighted = 0.0

    for prop in bs.world.propositions:
        p_aud = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        p_focal = bs.confidence(focal_id, prop.proposition_id, fabula_t)
        if abs(p_aud - p_focal) < _EPS:
            continue
        kl_a_f = _binary_kl(p_aud, p_focal) * _prop_stakes_at(prop, fabula_t)
        kl_f_a = _binary_kl(p_focal, p_aud) * _prop_stakes_at(prop, fabula_t)
        aud_adv += kl_a_f
        foc_adv += kl_f_a

        if prop.kind in _IRONY_SUSPENSE_KINDS:
            sus_irony += kl_a_f
        elif prop.kind in _IRONY_CURIOSITY_KINDS:
            cur_irony += kl_a_f
        elif prop.kind in _IRONY_SURPRISE_KINDS:
            sur_irony += kl_a_f

        # Audit (eighth pass, M3): time-aware salience so closed /
        # not-yet-open concerns can't weight irony scores.
        sal = _concern_salience_t(
            bs.world, focal_id, prop.proposition_id, fabula_t,
        )
        concern_weighted += kl_a_f * sal

    # Wall gradient: which character is most ironised (relative to audience)?
    most_id: Optional[str] = None
    most_score = 0.0
    for eid in bs.world.entities:
        if eid == AUDIENCE_ID:
            continue
        s = compute_irony_unified(bs, eid, fabula_t)
        if s > most_score:
            most_score = s
            most_id = eid

    return IronyBreakdown(
        audience_advantage=aud_adv,
        focal_advantage=foc_adv,
        suspense_irony_score=sus_irony,
        curiosity_irony_score=cur_irony,
        surprise_irony_score=sur_irony,
        concern_weighted_score=concern_weighted,
        most_ironised_entity_id=most_id,
        most_ironised_score=most_score,
    )


# Iser characterisation-gap kinds: 'who is X really?' / 'is trait T true of X?'
_MYSTERY_CHARACTER_KINDS = {"identity_is", "trait_holds"}


@dataclass
class MysteryBreakdown:
    """Carroll macro/micro + Sternberg curiosity + Iser character-gap + Ryan tellability.

    Theory:
      * Carroll (1990) erotetic — the existing causal-ancestor
        entropy is the *plot-gap* score (effects whose causes are
        hidden). The ``governing_question`` surfaces the open
        proposition whose resolution would commit the most other
        propositions (highest stakes × causal in-degree) — Carroll's
        macro-question.
      * Iser (1978) "Leerstellen" — narratives generate mystery from
        characterisation gaps too, not only causal gaps.
        ``character_gap_score`` sums entropy over kind ∈
        {identity_is, trait_holds} propositions whose audience
        confidence is below the revelation threshold.
      * Ryan (1991) tellability — ``tellability_weighted_score``
        weights each open question by the count and salience of
        :class:`Concern` records referencing it across all entities.
        Mysteries many characters care about dominate.
    """
    plot_gap_score: float
    character_gap_score: float
    tellability_weighted_score: float
    governing_question_id: Optional[str]
    governing_question_description: Optional[str]
    character_gap_descriptions: List[str]


def compute_mystery_breakdown(
    bs: BeliefState, fabula_t: int, *, threshold: float = 0.7,
    top_k: int = 5,
) -> MysteryBreakdown:
    """See :class:`MysteryBreakdown` for theory references."""
    # Plot-gap (Carroll erotetic) — reuse the canonical scorer.
    plot_gap = compute_mystery_unified(bs, fabula_t, threshold=threshold)

    # Character-gap (Iser): unresolved identity/trait propositions.
    character_gap = 0.0
    char_gap_props: List[Tuple[float, Proposition]] = []
    for prop in bs.world.propositions:
        if prop.kind not in _MYSTERY_CHARACTER_KINDS:
            continue
        p_aud = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        # Open = not yet near 0 or 1.
        if abs(p_aud - 0.5) > 0.4:
            continue
        h = _binary_entropy(p_aud) * _prop_stakes_at(prop, fabula_t)
        character_gap += h
        char_gap_props.append((h, prop))
    char_gap_props.sort(key=lambda kv: -kv[0])
    char_gap_descs = [
        prop.description or prop.proposition_id
        for _, prop in char_gap_props[:top_k]
    ]

    # Tellability (Ryan) weight on plot-gap propositions.
    tellability_weighted = 0.0
    for prop in bs.world.propositions:
        p_aud = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        if p_aud < threshold:
            continue
        tw = _tellability(bs.world, prop.proposition_id)
        tellability_weighted += _binary_entropy(p_aud) * _prop_stakes_at(prop, fabula_t) * tw

    # Governing question (Carroll macro): open prop with max
    # stakes × causal in-degree.
    g = nx.DiGraph()
    for evt in bs.world.events:
        g.add_node(evt.id)
    for ce in bs.world.causal_topology:
        g.add_edge(ce.source_id, ce.target_id, weight=ce.causal_force)

    governing_id: Optional[str] = None
    governing_desc: Optional[str] = None
    governing_score = 0.0
    for prop in bs.world.propositions:
        if not prop.referent_ids:
            continue
        p_aud = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        # Open = audience uncertain.
        if abs(p_aud - 0.5) > 0.45:
            continue
        evt_id = prop.referent_ids[0]
        in_deg = g.in_degree(evt_id) if evt_id in g else 0
        score = _prop_stakes_at(prop, fabula_t) * (1 + in_deg)
        if score > governing_score:
            governing_score = score
            governing_id = prop.proposition_id
            governing_desc = prop.description

    return MysteryBreakdown(
        plot_gap_score=plot_gap,
        character_gap_score=character_gap,
        tellability_weighted_score=tellability_weighted,
        governing_question_id=governing_id,
        governing_question_description=governing_desc,
        character_gap_descriptions=char_gap_descs,
    )


@dataclass
class UnifiedAffectScores:
    suspense: float
    surprise: float
    irony: Dict[str, float]
    mystery: float


def compute_unified_affects(
    world: WorldStateV1,
    fabula_t: int,
    *,
    prior_fabula_t: Optional[int] = None,
    focal_ids: Optional[List[str]] = None,
    ensure_synthesis: bool = True,
) -> UnifiedAffectScores:
    """One-call entry point. Synthesises propositions / audience if needed,
    then runs all four affect queries and returns a typed bundle.

    ``focal_ids`` defaults to all non-audience entities. ``prior_fabula_t``
    defaults to ``fabula_t - 1`` (so surprise is the single-step belief
    revision into ``fabula_t``).
    """
    if ensure_synthesis:
        if not world.propositions:
            synthesise_propositions(world)
        backfill_character_belief_propositions(world)
        if AUDIENCE_ID not in world.entities:
            synthesise_audience_entity(world)

    bs = BeliefState(world=world)
    if focal_ids is None:
        focal_ids = [eid for eid in world.entities if eid != AUDIENCE_ID]
    if prior_fabula_t is None:
        prior_fabula_t = fabula_t - 1

    suspense = compute_suspense_unified(bs, fabula_t)
    surprise = compute_surprise_unified(bs, fabula_t, prior_fabula_t)
    irony = {
        fid: compute_irony_unified(bs, fid, fabula_t) for fid in focal_ids
    }
    mystery = compute_mystery_unified(bs, fabula_t)

    return UnifiedAffectScores(
        suspense=suspense,
        surprise=surprise,
        irony=irony,
        mystery=mystery,
    )


# ===========================================================================
# Character-felt emotion appraisals
# ===========================================================================
#
# Theory grounding: Ortony, Clore & Collins (1988) "OCC" appraisal grid;
# Lazarus (1991) cognitive appraisal; Frijda (1986) action readiness;
# Roseman (1996) appraisal-emotion mapping. Every score below grounds out
# in the same substrate (BeliefState + Concern + Relationship +
# causal_topology) so emotions cannot drift from each other or from the
# four reader-affects above.

def _focal_concerns(
    world: WorldStateV1, focal_id: str,
    *, polarity: Optional[str] = None,
    fabula_t: Optional[int] = None,
) -> List[Concern]:
    """All concerns held by ``focal_id``, optionally filtered by polarity
    and active at ``fabula_t`` (respects ``activation_fabula_window``)."""
    ent = world.entities.get(focal_id)
    if ent is None:
        return []
    out = []
    for c in ent.concerns:
        # Resolve concern state at fabula_t so polarity / window
        # snapshot updates are honoured (mirrors Entity / GlobalTrait
        # snapshot semantics).
        _pol = (
            _concern_polarity_at(c, fabula_t)
            if fabula_t is not None
            else c.polarity
        )
        if polarity is not None and _pol != polarity:
            continue
        if fabula_t is not None:
            _win = _concern_window_at(c, fabula_t)
            if _win:
                lo, hi = _win
                if not (lo <= fabula_t <= hi):
                    continue
        out.append(c)
    return out


def _affinity_between(
    world: WorldStateV1, src_id: str, tgt_id: str,
) -> float:
    """Directed affinity from ``src_id`` to ``tgt_id`` (0.0 if no edge)."""
    for rel in world.social_topology:
        if rel.source_entity_id == src_id and rel.target_entity_id == tgt_id:
            m = rel.metrics.get("affinity")
            if m is not None and m.observed:
                return float(m.value)
    return 0.0


def _focal_belief_about_event(
    bs: BeliefState, focal_id: str, event_id: str, fabula_t: int,
) -> float:
    """Convenience: focal's confidence about ``PROP_FROM_<event_id>``."""
    return bs.confidence(focal_id, f"PROP_FROM_{event_id}", fabula_t)


# ---------------------------------------------------------------------------
# Fear (Lazarus + Öhman/LeDoux + Frijda)
# ---------------------------------------------------------------------------

@dataclass
class FearAppraisal:
    """Lazarus appraisal × Öhman/LeDoux modal split × Frijda action-readiness.

    Theory:
      * Lazarus (1991) — fear = ``p(harm) × magnitude × low(coping)``.
        ``object_fear_score`` aggregates the focal's high-confidence
        fear-polarity concerns weighted by stakes × salience.
      * Öhman & Mineka (2001) / LeDoux (1996) — distinguish *anxiety*
        (diffuse, no clear object: high entropy across fear concerns)
        from *fear* (object-specific). ``anxiety_score`` is the
        Shannon entropy over the focal's fear-concern beliefs;
        ``object_fear_score`` is the max-salience peak.
      * Frijda (1986) action-readiness — when no flight path exists
        (focal's location has no spatial exits OR all exits are
        blocked) the affect escalates to **dread**.
    """
    object_fear_score: float
    anxiety_score: float
    coping_score: float
    flight_available: bool
    dread: bool
    primary_concern_id: Optional[str]
    primary_concern_description: Optional[str]


def _focal_coping(world: WorldStateV1, focal_id: str) -> float:
    """Crude Lazarus coping proxy in [0, 1].

    Mean of the focal's traits that map onto resilience-axes
    (``resilience``, ``confidence``, ``physical_strength``,
    ``social_support``). Falls back to 0.5 when no such traits
    exist on the entity.
    """
    ent = world.entities.get(focal_id)
    if ent is None or not ent.traits:
        return 0.5
    keys = ("resilience", "confidence", "physical_strength", "social_support")
    vals = []
    for k in keys:
        v = ent.traits.get(k)
        if v is None:
            continue
        if hasattr(v, "value"):
            vals.append(float(v.value))
        else:
            vals.append(float(v))
    if not vals:
        return 0.5
    return max(0.0, min(1.0, sum(vals) / len(vals)))


def _flight_available(world: WorldStateV1, focal_id: str) -> bool:
    """True if focal's location has at least one connected exit."""
    ent = world.entities.get(focal_id)
    if ent is None:
        return True  # unknowable → don't escalate to dread
    loc = world.locations.get(ent.location_id)
    if loc is None:
        return True
    # Locations may carry connections in ``ambient_state`` or via a
    # separate connectivity edge; we use the conservative heuristic
    # that a non-sentinel location with at least one connection
    # mentioned in ambient_state['connected_to'] permits flight.
    connected = loc.ambient_state.get("connected_to") if loc.ambient_state else None
    if connected:
        return True
    # Fallback: count other locations sharing any common entity-link.
    return len(world.locations) > 1 and ent.location_id != "LOC_NONE"


def compute_fear_appraisal(
    bs: BeliefState, focal_id: str, fabula_t: int,
) -> FearAppraisal:
    """See :class:`FearAppraisal` for theory references."""
    concerns = _focal_concerns(
        bs.world, focal_id, polarity="fear", fabula_t=fabula_t,
    )
    if not concerns:
        return FearAppraisal(
            object_fear_score=0.0,
            anxiety_score=0.0,
            coping_score=_focal_coping(bs.world, focal_id),
            flight_available=_flight_available(bs.world, focal_id),
            dread=False,
            primary_concern_id=None,
            primary_concern_description=None,
        )

    coping = _focal_coping(bs.world, focal_id)
    coping_factor = max(0.1, 1.0 - coping)

    obj_score = 0.0
    anxiety = 0.0
    anxiety_weight_sum = 0.0  # P1-25: Track total weight for normalization
    primary: Optional[Tuple[float, Concern, Optional[Proposition]]] = None
    prop_idx = {p.proposition_id: p for p in bs.world.propositions}
    for c in concerns:
        prop = prop_idx.get(c.proposition_id)
        prob = bs.confidence(focal_id, c.proposition_id, fabula_t)
        stakes = _prop_stakes_at(prop, fabula_t) if prop else 0.5
        sal = _concern_salience_at(c, fabula_t)
        s = prob * stakes * sal * coping_factor
        if s > obj_score:
            obj_score = s
            primary = (s, c, prop)
        # P1-25: Accumulate anxiety as weighted entropy sum
        anxiety += _binary_entropy(prob) * sal
        anxiety_weight_sum += sal

    flight = _flight_available(bs.world, focal_id)
    dread = obj_score > 0.4 and not flight and coping < 0.4

    # P1-25: Normalize anxiety by total salience weight to prevent unbounded sum
    # Shannon entropy is [0, ln(2)] per term; normalizing keeps anxiety in [0,1]
    if anxiety_weight_sum > 0.0:
        anxiety = anxiety / anxiety_weight_sum

    return FearAppraisal(
        object_fear_score=obj_score,
        anxiety_score=anxiety,
        coping_score=coping,
        flight_available=flight,
        dread=dread,
        primary_concern_id=primary[1].concern_id if primary else None,
        primary_concern_description=(
            primary[2].description if primary and primary[2] else None
        ),
    )


# ---------------------------------------------------------------------------
# Joy (Fredrickson + OCC happy-for / gloating)
# ---------------------------------------------------------------------------

@dataclass
class JoyAppraisal:
    """Fredrickson broaden-and-build × OCC desirable-event branches.

    Theory:
      * Fredrickson (2001) broaden-and-build — joy widens cognitive
        scope. ``own_joy_score`` aggregates realised desires of the
        focal (desire-polarity concerns whose belief approaches 1.0).
      * Ortony, Clore & Collins (1988) — joy decomposes into:
          * ``own_joy_score`` — focal's desirable event realised
          * ``happy_for_score`` — liked-other's desirable event
            realised (affinity[focal→other] > 0.3)
          * ``gloating_score`` — disliked-other's desirable event
            *not* realised, or feared event realised
            (affinity[focal→other] < -0.3)
      * Lazarus relief — ``relief_score`` aggregates the focal's
        feared concerns whose audience-confirmed belief just dropped
        toward false (requires a ``prior_fabula_t``; zero otherwise).
    """
    own_joy_score: float
    happy_for_score: float
    gloating_score: float
    relief_score: float
    primary_concern_id: Optional[str]
    primary_concern_description: Optional[str]


def compute_joy_appraisal(
    bs: BeliefState, focal_id: str, fabula_t: int,
    *, prior_fabula_t: Optional[int] = None,
) -> JoyAppraisal:
    """See :class:`JoyAppraisal` for theory references."""
    prop_idx = {p.proposition_id: p for p in bs.world.propositions}

    own = 0.0
    primary: Optional[Tuple[float, Concern, Optional[Proposition]]] = None
    for c in _focal_concerns(
        bs.world, focal_id, polarity="desire", fabula_t=fabula_t,
    ):
        prop = prop_idx.get(c.proposition_id)
        prob = bs.confidence(focal_id, c.proposition_id, fabula_t)
        stakes = _prop_stakes_at(prop, fabula_t) if prop else 0.5
        s = prob * stakes * _concern_salience_at(c, fabula_t)
        own += s
        if s > (primary[0] if primary else 0.0):
            primary = (s, c, prop)

    happy_for = 0.0
    gloating = 0.0
    for other_id, other in bs.world.entities.items():
        if other_id in (focal_id, AUDIENCE_ID):
            continue
        aff = _affinity_between(bs.world, focal_id, other_id)
        if abs(aff) < 0.3:
            continue
        for c in other.concerns:
            prop = prop_idx.get(c.proposition_id)
            # P1-26: For gloating, use FOCAL's belief about whether the OTHER's
            # feared event happened, not the other's own belief. Schadenfreude
            # is joy at the enemy's misfortune as WE perceive it.
            prob_focal = bs.confidence(focal_id, c.proposition_id, fabula_t)
            prob_other = bs.confidence(other_id, c.proposition_id, fabula_t)
            stakes = _prop_stakes_at(prop, fabula_t) if prop else 0.5
            _pol = _concern_polarity_at(c, fabula_t)
            if aff > 0.3 and _pol == "desire":
                # Happy-for uses other's belief (empathy)
                base = prob_other * stakes * _concern_salience_at(c, fabula_t)
                happy_for += base
            elif aff < -0.3 and _pol == "fear":
                # P1-26: Gloating uses focal's belief that the other's fear realized
                base = prob_focal * stakes * _concern_salience_at(c, fabula_t)
                gloating += base

    relief = 0.0
    if prior_fabula_t is not None:
        for c in _focal_concerns(
            bs.world, focal_id, polarity="fear", fabula_t=fabula_t,
        ):
            p_now = bs.confidence(focal_id, c.proposition_id, fabula_t)
            p_prev = bs.confidence(
                focal_id, c.proposition_id, prior_fabula_t,
            )
            if p_prev - p_now > 0.2:
                prop = prop_idx.get(c.proposition_id)
                stakes = _prop_stakes_at(prop, fabula_t) if prop else 0.5
                relief += (p_prev - p_now) * stakes * _concern_salience_at(c, fabula_t)

    return JoyAppraisal(
        own_joy_score=own,
        happy_for_score=happy_for,
        gloating_score=gloating,
        relief_score=relief,
        primary_concern_id=primary[1].concern_id if primary else None,
        primary_concern_description=(
            primary[2].description if primary and primary[2] else None
        ),
    )


# ---------------------------------------------------------------------------
# Regret (Kahneman-Miller norm theory + Roese commission/omission)
# ---------------------------------------------------------------------------

@dataclass
class RegretAppraisal:
    """Kahneman-Miller closeness/controllability × Roese commission/omission.

    Theory:
      * Kahneman & Miller (1986) norm theory — regret intensity
        scales with the *closeness* of the unchosen counterfactual
        and the *controllability* of the divergence event.
        ``agentive_regret_score`` is populated when the divergence
        event was a ``choice`` event by the focal (high
        controllability); ``disappointment_score`` when no such
        choice exists (negative outcome from external causes).
      * Roese (1997) action/inaction asymmetry — commission generates
        hot short-term regret; omission generates cold long-term
        regret. ``mode`` records which.
      * Gilovich & Medvec (1995) downward/upward counterfactual —
        ``downward_relief_score`` flags 'could have been worse' when
        a sibling negative outcome was averted by the focal's choice.
    """
    agentive_regret_score: float
    disappointment_score: float
    commission_score: float
    omission_score: float
    downward_relief_score: float
    divergence_event_id: Optional[str]
    loss_event_id: Optional[str]
    mode: str  # "commission" | "omission" | "disappointment" | "none"


def _focal_negative_event(
    world: WorldStateV1, focal_id: str, fabula_t: int,
) -> Optional[EventNode]:
    """Most recent event ≤ ``fabula_t`` that hurt the focal.

    Two flavours of harm are recognised:

    1. **Trait-level harm** — a ``mutation`` causal edge with
       ``trait_delta < 0`` whose target is the focal.
    2. **Status-level harm to a cared-about other** — a ``mutation``
       edge whose target is some other entity *with positive
       affinity from the focal* (>0.2) and whose status flips to a
       loss state (``dead``/``lost``/``destroyed``).

    Without (2) the regret/rage paths only fire on focal-trait
    decrements and silently miss canonical narrative losses (death
    of an ally, captured loved one) which carry no trait_delta.
    Theory: Lazarus (1991) primary-appraisal *goal congruence* is
    violated by harm to cared-about goals just as much as by harm
    to the self.
    """
    candidates: List[Tuple[int, EventNode]] = []
    for ce in world.causal_topology:
        if ce.causality_type != "mutation":
            continue
        if ce.fabula_time > fabula_t:
            continue
        # Direct trait-decrement on focal.
        if ce.target_id == focal_id and ce.trait_delta is not None and ce.trait_delta < 0:
            evt = next((e for e in world.events if e.id == ce.source_id), None)
            if evt is not None:
                candidates.append((evt.fabula_time, evt))
            continue
        # Status-loss on a cared-about other.
        tgt = ce.target_id
        if tgt == focal_id or not isinstance(tgt, str) or not tgt.startswith("ENT_"):
            continue
        status = _status_at_fabula(world, tgt, fabula_t)
        if status not in ("dead", "lost", "destroyed"):
            continue
        if _affinity_between(world, focal_id, tgt) <= 0.2:
            continue
        evt = next((e for e in world.events if e.id == ce.source_id), None)
        if evt is not None:
            candidates.append((evt.fabula_time, evt))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    return candidates[0][1]


def _focal_recent_choice(
    world: WorldStateV1, focal_id: str, before_fabula_t: int,
) -> Optional[EventNode]:
    """Most recent ``choice`` event by ``focal_id`` strictly before ``before_fabula_t``."""
    cs = [
        e for e in world.events
        if e.event_type == "choice"
        and focal_id in e.actor_ids
        and e.fabula_time < before_fabula_t
    ]
    if not cs:
        return None
    cs.sort(key=lambda e: -e.fabula_time)
    return cs[0]


def compute_regret_appraisal(
    bs: BeliefState, focal_id: str, fabula_t: int,
) -> RegretAppraisal:
    """See :class:`RegretAppraisal` for theory references."""
    loss = _focal_negative_event(bs.world, focal_id, fabula_t)
    if loss is None:
        return RegretAppraisal(
            agentive_regret_score=0.0, disappointment_score=0.0,
            commission_score=0.0, omission_score=0.0,
            downward_relief_score=0.0,
            divergence_event_id=None, loss_event_id=None, mode="none",
        )

    choice = _focal_recent_choice(bs.world, focal_id, loss.fabula_time)
    # Magnitude proxy: |trait_delta| summed over loss's mutation edges.
    magnitude = 0.0
    for ce in bs.world.causal_topology:
        if ce.source_id == loss.id and ce.causality_type == "mutation" and ce.trait_delta is not None:
            magnitude += abs(ce.trait_delta)
    magnitude = magnitude or 1.0

    if choice is not None:
        # Closeness proxy: smaller fabula gap → closer counterfactual.
        gap = max(1, loss.fabula_time - choice.fabula_time)
        closeness = math.exp(-gap / max(1.0, _auto_tau_fabula(bs.world)))
        agentive = magnitude * closeness
        disappointment = 0.0
        # Commission vs omission heuristic: choice description containing
        # negation/refusal markers ⇒ omission.
        desc = (choice.description or "").lower()
        omission_markers = (
            "decline", "refuse", "fail to", "did not", "didn't",
            "ignore", "withhold", "neglect", "abstain",
        )
        if any(m in desc for m in omission_markers):
            mode = "omission"
            commission_score = 0.0
            omission_score = agentive
        else:
            mode = "commission"
            commission_score = agentive
            omission_score = 0.0
    else:
        agentive = 0.0
        disappointment = magnitude
        commission_score = 0.0
        omission_score = 0.0
        mode = "disappointment"

    # Downward relief: any sibling event off the same choice with
    # *worse* magnitude that the focal averted.
    downward = 0.0
    if choice is not None:
        for ce in bs.world.causal_topology:
            if ce.source_id != choice.id or ce.causality_type != "chain_reaction":
                continue
            sibling = next(
                (e for e in bs.world.events if e.id == ce.target_id),
                None,
            )
            if sibling is None or sibling.id == loss.id:
                continue
            sib_mag = sum(
                abs(c.trait_delta or 0.0) for c in bs.world.causal_topology
                if c.source_id == sibling.id and c.causality_type == "mutation"
            )
            if sib_mag > magnitude:
                downward += sib_mag - magnitude

    return RegretAppraisal(
        agentive_regret_score=agentive,
        disappointment_score=disappointment,
        commission_score=commission_score,
        omission_score=omission_score,
        downward_relief_score=downward,
        divergence_event_id=choice.id if choice else None,
        loss_event_id=loss.id,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Grief (Bowlby coupling + Kübler-Ross stage detection)
# ---------------------------------------------------------------------------

@dataclass
class GriefAppraisal:
    """Bowlby attachment × Kübler-Ross stage detection × Worden tasks.

    Theory:
      * Bowlby (1969/1980) attachment — grief intensity scales with
        the strength of the bond to the lost figure.
        ``coupling_strength`` is the geometric mean of mutual
        affinity between focal and the deceased.
      * Kübler-Ross (1969) stage model — ``stage`` ∈
        {denial, anger, bargaining, depression, acceptance, none}
        derived from the focal's belief & concern dynamics post-loss
        (denial = focal under-confident about loss despite audience
        certainty; anger = new fear concerns toward perpetrator;
        bargaining = new outcome concerns whose realisation would
        undo loss; depression = drop in concern salience;
        acceptance = stabilisation).
      * Worden (1991) tasks of mourning — ``unfinished_concern_count``
        is the count of pre-loss focal concerns referencing the lost
        entity that remain unrevised post-loss.
    """
    coupling_strength: float
    loss_event_id: Optional[str]
    lost_entity_id: Optional[str]
    stage: str  # "denial"|"anger"|"bargaining"|"depression"|"acceptance"|"none"
    unfinished_concern_count: int


def _status_at_fabula(
    world: WorldStateV1, entity_id: str, fabula_t: int,
) -> Optional[str]:
    """Effective status of ``entity_id`` at ``fabula_t``.

    Walks ``Entity.state_timeline`` in fabula order and returns the
    most recent status set on or before ``fabula_t``; falls back to
    the entity's top-level (initial) status. Returns None when the
    entity is unknown.
    """
    ent = world.entities.get(entity_id)
    if ent is None:
        return None
    current = ent.status
    if not ent.state_timeline:
        return current
    for snap in sorted(ent.state_timeline, key=lambda s: s.fabula_time):
        if snap.fabula_time > fabula_t:
            break
        if snap.status is not None:
            current = snap.status
    return current


def _detect_lost_entity(
    world: WorldStateV1, focal_id: str, fabula_t: int,
) -> Tuple[Optional[EventNode], Optional[str]]:
    """Find the most recent death/loss event affecting an entity the focal cared about."""
    candidates: List[Tuple[int, EventNode, str]] = []
    for evt in world.events:
        if evt.fabula_time > fabula_t:
            continue
        # Mutation edges turning a target entity's status to dead/lost.
        for ce in world.causal_topology:
            if ce.source_id != evt.id or ce.causality_type != "mutation":
                continue
            tgt = ce.target_id
            if tgt == focal_id or not tgt.startswith("ENT_"):
                continue
            # Heuristic: target entity's status at ``fabula_t`` (read from
            # state_timeline, falling back to initial status) is in the
            # loss vocabulary, AND focal had a positive bond to them.
            status = _status_at_fabula(world, tgt, fabula_t)
            if status in ("dead", "lost", "destroyed"):
                if _affinity_between(world, focal_id, tgt) > 0.2:
                    candidates.append((evt.fabula_time, evt, tgt))
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: -t[0])
    return candidates[0][1], candidates[0][2]


def compute_grief_appraisal(
    bs: BeliefState, focal_id: str, fabula_t: int,
) -> GriefAppraisal:
    """See :class:`GriefAppraisal` for theory references."""
    loss_event, lost_id = _detect_lost_entity(bs.world, focal_id, fabula_t)
    if loss_event is None or lost_id is None:
        return GriefAppraisal(
            coupling_strength=0.0,
            loss_event_id=None, lost_entity_id=None,
            stage="none", unfinished_concern_count=0,
        )

    aff_focal = _affinity_between(bs.world, focal_id, lost_id)
    aff_back = _affinity_between(bs.world, lost_id, focal_id)
    coupling = (max(0.0, aff_focal) * max(0.0, aff_back)) ** 0.5
    if coupling == 0.0:
        coupling = max(0.0, aff_focal) * 0.5

    # Stage detection.
    p_focal_loss = _focal_belief_about_event(
        bs, focal_id, loss_event.id, fabula_t,
    )
    p_aud_loss = bs.confidence(
        AUDIENCE_ID, f"PROP_FROM_{loss_event.id}", fabula_t,
    )

    new_fear_re_perpetrator = False
    perp_ids = set(loss_event.actor_ids)
    for c in _focal_concerns(bs.world, focal_id, polarity="fear", fabula_t=fabula_t):
        prop = next(
            (p for p in bs.world.propositions if p.proposition_id == c.proposition_id),
            None,
        )
        if prop and any(r in perp_ids for r in prop.referent_ids):
            new_fear_re_perpetrator = True
            break

    bargaining = False
    for c in _focal_concerns(bs.world, focal_id, polarity="desire", fabula_t=fabula_t):
        prop = next(
            (p for p in bs.world.propositions if p.proposition_id == c.proposition_id),
            None,
        )
        if prop and lost_id in prop.referent_ids and prop.kind == "outcome":
            bargaining = True
            break

    if p_aud_loss > 0.7 and p_focal_loss < 0.5:
        stage = "denial"
    elif new_fear_re_perpetrator and perp_ids and focal_id not in perp_ids:
        stage = "anger"
    elif bargaining:
        stage = "bargaining"
    else:
        # Depression vs acceptance: depression = focal's max concern
        # salience dropped; acceptance = stable. Without temporal
        # comparison we conservatively pick depression while focal
        # still has active fear concerns referencing the lost entity,
        # else acceptance.
        active_fear = any(
            (lost_id in (next(
                (p for p in bs.world.propositions if p.proposition_id == c.proposition_id),
                None,
            ).referent_ids if next(
                (p for p in bs.world.propositions if p.proposition_id == c.proposition_id),
                None,
            ) else []))
            for c in _focal_concerns(bs.world, focal_id, polarity="fear", fabula_t=fabula_t)
        )
        stage = "depression" if active_fear else "acceptance"

    # Worden unfinished tasks: focal concerns referencing lost entity.
    unfinished = 0
    for c in _focal_concerns(bs.world, focal_id, fabula_t=fabula_t):
        prop = next(
            (p for p in bs.world.propositions if p.proposition_id == c.proposition_id),
            None,
        )
        if prop and lost_id in prop.referent_ids:
            unfinished += 1

    return GriefAppraisal(
        coupling_strength=round(coupling, 3),
        loss_event_id=loss_event.id,
        lost_entity_id=lost_id,
        stage=stage,
        unfinished_concern_count=unfinished,
    )


# ---------------------------------------------------------------------------
# Rage (Berkowitz + Averill + Tedeschi-Felson)
# ---------------------------------------------------------------------------

_NORMATIVE_HARM_KINDS = {"betrayal", "abandonment", "humiliation", "injustice"}


@dataclass
class RageAppraisal:
    """Berkowitz frustration-aggression × Averill normative violation × Tedeschi-Felson coercive action.

    Theory:
      * Berkowitz (1989) — rage = ``blocked_concern_salience ×
        perpetrator_proximity × attribution_clarity``. We approximate
        proximity as inverse spatial gap (1 hop ≈ co-located) and
        attribution_clarity as the focal's confidence in the
        perpetrator-identity proposition.
      * Averill (1982) — rage requires *perceived violation of norm*.
        ``normative_violation`` is True when the loss event's
        triggering concern has a ``kind`` in {betrayal, abandonment,
        humiliation, injustice}.
      * Tedeschi & Felson (1994) — distinguish *retributive* rage
        (target = perpetrator) from *displaced* rage (target ≠
        perpetrator). ``mode`` records which.
    """
    blocked_concern_score: float
    perpetrator_id: Optional[str]
    attribution_clarity: float
    perpetrator_proximity: int  # 0 = co-located, larger = further; -1 = unknown
    normative_violation: bool
    mode: str  # "frustration"|"directed_rage"|"retributive_rage"|"displaced_rage"|"none"


def compute_rage_appraisal(
    bs: BeliefState, focal_id: str, fabula_t: int,
    *, intended_target_id: Optional[str] = None,
) -> RageAppraisal:
    """See :class:`RageAppraisal` for theory references.

    ``intended_target_id`` lets the caller name the entity the focal
    is currently directing aggression at; if it differs from the
    causally-attributed perpetrator the mode is *displaced*.
    """
    loss = _focal_negative_event(bs.world, focal_id, fabula_t)
    if loss is None:
        return RageAppraisal(
            blocked_concern_score=0.0, perpetrator_id=None,
            attribution_clarity=0.0, perpetrator_proximity=-1,
            normative_violation=False, mode="none",
        )

    # Blocked-concern salience: focal's fear concerns whose proposition
    # the loss event committed (audience belief now near 1.0).
    prop_idx = {p.proposition_id: p for p in bs.world.propositions}
    blocked = 0.0
    normative = False
    for c in _focal_concerns(bs.world, focal_id, polarity="fear", fabula_t=fabula_t):
        prop = prop_idx.get(c.proposition_id)
        if prop is None:
            continue
        prob = bs.confidence(focal_id, c.proposition_id, fabula_t)
        if prob > 0.5:
            blocked += prob * _prop_stakes_at(prop, fabula_t) * _concern_salience_at(c, fabula_t)
            if c.kind in _NORMATIVE_HARM_KINDS:
                normative = True

    # Perpetrator: most-recent actor on the loss event.
    perpetrator_id = loss.actor_ids[0] if loss.actor_ids else None
    attribution = 0.0
    if perpetrator_id is not None:
        # Confidence proxy: any belief/proposition naming perpetrator as
        # actor on this event ⇒ high; absence ⇒ low.
        attribution = _focal_belief_about_event(
            bs, focal_id, loss.id, fabula_t,
        )

    # Proximity: same location ⇒ 0; different ⇒ 1; unknown ⇒ -1.
    proximity = -1
    focal_ent = bs.world.entities.get(focal_id)
    perp_ent = bs.world.entities.get(perpetrator_id) if perpetrator_id else None
    if focal_ent is not None and perp_ent is not None:
        proximity = 0 if focal_ent.location_id == perp_ent.location_id else 1

    # Mode classification.
    if perpetrator_id is None or attribution < 0.3:
        mode = "frustration"
    elif intended_target_id is not None and intended_target_id != perpetrator_id:
        mode = "displaced_rage"
    elif normative and attribution > 0.6:
        mode = "retributive_rage"
    elif blocked > 0:
        mode = "directed_rage"
    else:
        mode = "frustration"

    return RageAppraisal(
        blocked_concern_score=blocked,
        perpetrator_id=perpetrator_id,
        attribution_clarity=attribution,
        perpetrator_proximity=proximity,
        normative_violation=normative,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Love (Sternberg triangular + Berscheid-Hatfield + Bowlby attachment style)
# ---------------------------------------------------------------------------

@dataclass
class LoveAppraisal:
    """Sternberg triangular × Berscheid-Hatfield passionate/companionate × Bowlby attachment style.

    Theory:
      * Sternberg (1986) triangular theory — love decomposes into
        ``intimacy_score`` (mutual high-confidence beliefs about each
        other), ``passion_score`` (high-salience concerns referring
        to the partner), and ``commitment_score`` (relationship
        durability — affinity stability over fabula time, weighted
        by inertia of the affinity metric).
      * Berscheid & Hatfield (1969/1974) — passionate love
        (high passion, lower intimacy) vs. companionate love
        (high intimacy, lower passion). ``style`` records which
        register dominates, or 'balanced' when both are high.
      * Bowlby (1969) attachment styles — diagnosed via the
        relationship's `fear` metric: high fear toward partner with
        high affinity ⇒ anxious; low affinity-back combined with
        high focal-affinity ⇒ avoidant; balanced mutual affinity ⇒
        secure.
    """
    primary_partner_id: Optional[str]
    intimacy_score: float
    passion_score: float
    commitment_score: float
    style: str  # "passionate"|"companionate"|"balanced"|"none"
    attachment_style: str  # "secure"|"anxious"|"avoidant"|"unknown"


def _belief_count_about(
    world: WorldStateV1, holder_id: str, target_id: str,
) -> int:
    """Count of ``holder``'s beliefs whose target_id == ``target_id``."""
    ent = world.entities.get(holder_id)
    if ent is None:
        return 0
    n = sum(1 for b in ent.beliefs if b.target_id == target_id)
    for snap in ent.state_timeline:
        n += sum(1 for b in snap.beliefs_added if b.target_id == target_id)
    return n


def compute_love_appraisal(
    bs: BeliefState, focal_id: str, fabula_t: int,
    *, partner_id: Optional[str] = None,
) -> LoveAppraisal:
    """See :class:`LoveAppraisal` for theory references."""
    # Find primary partner: highest mutual-affinity sibling.
    if partner_id is None:
        best: Optional[Tuple[float, str]] = None
        for other_id in bs.world.entities:
            if other_id in (focal_id, AUDIENCE_ID):
                continue
            a1 = _affinity_between(bs.world, focal_id, other_id)
            a2 = _affinity_between(bs.world, other_id, focal_id)
            if a1 < 0.3:
                continue
            mutual = (a1 * max(a2, 0.0)) ** 0.5 if a2 > 0 else a1 * 0.5
            if not best or mutual > best[0]:
                best = (mutual, other_id)
        if best is None:
            return LoveAppraisal(
                primary_partner_id=None, intimacy_score=0.0,
                passion_score=0.0, commitment_score=0.0,
                style="none", attachment_style="unknown",
            )
        partner_id = best[1]

    # Intimacy: mutual belief-count about each other (normalised by max).
    f_about_p = _belief_count_about(bs.world, focal_id, partner_id)
    p_about_f = _belief_count_about(bs.world, partner_id, focal_id)
    intimacy = float(min(f_about_p, p_about_f))

    # Passion: focal concerns whose proposition references the partner.
    passion = 0.0
    prop_idx = {p.proposition_id: p for p in bs.world.propositions}
    for c in _focal_concerns(bs.world, focal_id, fabula_t=fabula_t):
        prop = prop_idx.get(c.proposition_id)
        if prop and partner_id in prop.referent_ids:
            passion += _concern_salience_at(c, fabula_t) * (
                _prop_stakes_at(prop, fabula_t) if prop else 0.5
            )

    # Commitment: affinity inertia × normalised relationship age.
    # Sternberg (1986) commitment is *durability over time*, so the
    # right age proxy is the fabula span across which this dyad has
    # been on-stage — not a single ``last_updated_fabula`` instant
    # (which collapsed age to 1 and pinned commitment at
    # ``inertia × ε`` on every world). Approximate the dyad's
    # first-seen fabula tick from the earliest event in which both
    # participants appear (actor / target / addressee / participant).
    aff_inertia = 0.3
    aff_last_ft = fabula_t
    for rel in bs.world.social_topology:
        if rel.source_entity_id == focal_id and rel.target_entity_id == partner_id:
            m = rel.metrics.get("affinity")
            if m is not None:
                aff_inertia = m.inertia
                aff_last_ft = m.last_updated_fabula
                break
    aff_first_ft = fabula_t
    for evt in bs.world.events:
        if evt.fabula_time > fabula_t:
            continue
        ids = (
            set(getattr(evt, "actor_ids", []) or [])
            | set(getattr(evt, "target_ids", []) or [])
            | set(getattr(evt, "addressee_ids", []) or [])
            | set(getattr(evt, "participant_ids", []) or [])
        )
        if focal_id in ids and partner_id in ids:
            if evt.fabula_time < aff_first_ft:
                aff_first_ft = evt.fabula_time
    age = max(0, aff_last_ft - aff_first_ft) + 1
    tau = _auto_tau_fabula(bs.world)
    commitment = aff_inertia * min(1.0, age / max(1.0, tau * 4))

    # Style.
    if passion > intimacy * 1.5 and passion > 0.5:
        style = "passionate"
    elif intimacy > passion * 1.5 and intimacy > 1:
        style = "companionate"
    elif passion > 0 or intimacy > 0:
        style = "balanced"
    else:
        style = "none"

    # Attachment style via fear metric on focal→partner edge.
    attachment = "unknown"
    aff_focal = _affinity_between(bs.world, focal_id, partner_id)
    aff_back = _affinity_between(bs.world, partner_id, focal_id)
    fear_val = 0.0
    for rel in bs.world.social_topology:
        if rel.source_entity_id == focal_id and rel.target_entity_id == partner_id:
            m = rel.metrics.get("fear")
            if m is not None and m.observed:
                fear_val = m.value
            break
    if aff_focal > 0.3 and fear_val > 0.4:
        attachment = "anxious"
    elif aff_focal > 0.3 and aff_back < 0.2:
        attachment = "avoidant"
    elif aff_focal > 0.3 and aff_back > 0.3 and fear_val < 0.2:
        attachment = "secure"

    return LoveAppraisal(
        primary_partner_id=partner_id,
        intimacy_score=intimacy,
        passion_score=passion,
        commitment_score=commitment,
        style=style,
        attachment_style=attachment,
    )
