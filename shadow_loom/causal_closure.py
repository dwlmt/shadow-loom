"""Shared disjunctive-Pearl closure helpers for ``chain_reaction``
Event→Event edges.

This module unifies the three previously-duplicated implementations
of the do-surgery descendant-closure rule that previously lived in:

  * :mod:`shadow_loom.pipeline`              (merge-time closure)
  * :mod:`shadow_loom.directive_assembly`    (brief constraints)
  * :mod:`shadow_loom.causal_physics`        (intervention-time closure)

Semantics — see :func:`expand_chain_reaction_closure` for the full
Pearl / Halpern-Pearl rationale. The short version, with the
post-2026-05-29 INUS ``necessity`` flag taken into account:

    An event Y joins the closure iff Y has at least one
    ``chain_reaction`` incoming edge AND either:

      (a) **some** ``necessary`` parent of Y is in the closure OR in
          the cause-disconnected set — pruning a necessary
          precondition forces Y to fall, regardless of how many
          sufficient causes survive; OR

      (b) Y has at least one ``sufficient`` parent and **every**
          ``sufficient`` parent of Y is in the closure OR in the
          cause-disconnected set — the classical disjunctive rule.

    ``contributory`` parents are ignored for the closure: they shape
    the effect but neither suffice for it nor are required by it.

``enables`` / ``affordance_gate`` / ``ambient_propagation`` /
``mutation`` edges are modifiers, not sufficient causes, so they do
not participate (regardless of necessity).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

    from .models import WorldStateV1


# A parent map maps child_event_id -> list of (parent_event_id,
# mechanism, necessity). ``necessity`` is one of
# {"sufficient", "necessary", "contributory"}; see
# :class:`shadow_loom.models.CausalEdge.necessity`. Default
# ``"sufficient"`` for legacy edges without the flag.
Necessity = Literal["sufficient", "necessary", "contributory"]
ChainReactionParents = Dict[str, List[Tuple[str, str, Necessity]]]


def chain_reaction_parents_from_world_state(
    world_state: "WorldStateV1",
) -> ChainReactionParents:
    """Build the chain_reaction parent map from a ``WorldStateV1``.

    Reads ``world_state.causal_topology`` and keeps only edges whose
    ``causality_type == "chain_reaction"``. Each retained edge is
    bucketed under its ``target_id`` as a
    ``(source_id, mechanism, necessity)`` triple.
    """
    parents_of: ChainReactionParents = {}
    for edge in getattr(world_state, "causal_topology", None) or []:
        if getattr(edge, "causality_type", None) != "chain_reaction":
            continue
        mech = getattr(edge, "mechanism", "") or ""
        nec = getattr(edge, "necessity", "sufficient") or "sufficient"
        parents_of.setdefault(edge.target_id, []).append(
            (edge.source_id, mech, nec)
        )
    return parents_of


def chain_reaction_parents_from_sandbox(
    sandbox: "nx.MultiDiGraph",
) -> ChainReactionParents:
    """Build the chain_reaction parent map from an AMWN sandbox graph.

    Walks edges whose ``edge_type == "causal"`` and
    ``causality_type == "chain_reaction"``.
    """
    parents_of: ChainReactionParents = {}
    for u, v, edata in sandbox.edges(data=True):
        if edata.get("edge_type") != "causal":
            continue
        if edata.get("causality_type") != "chain_reaction":
            continue
        mech = edata.get("mechanism", "") or ""
        nec = edata.get("necessity", "sufficient") or "sufficient"
        parents_of.setdefault(v, []).append((u, mech, nec))
    return parents_of


def expand_chain_reaction_closure(
    parents_of: ChainReactionParents,
    seed_ids: Iterable[str],
    *,
    cause_disconnected_ids: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Expand ``seed_ids`` to its chain_reaction descendant closure,
    iterated to fixpoint, honouring per-edge ``necessity``.

    Closure rule (Pearl + INUS / Halpern-Pearl):

      Y joins the closure iff Y has at least one chain_reaction
      incoming edge AND either:

        (a) **some** ``necessary`` parent of Y is in the closure or
            in ``cause_disconnected_ids`` — necessary preconditions
            cannot be substituted for, so erasing one forces Y to
            fall regardless of sufficient-cause survivors; OR

        (b) Y has at least one ``sufficient`` parent and **every**
            ``sufficient`` parent of Y is in the closure or in
            ``cause_disconnected_ids`` — the classical disjunctive
            sufficient-cause rule (preserves Halpern-Pearl
            over-determination: a surviving sufficient cause keeps
            the descendant alive).

      ``contributory`` parents are ignored for closure: they shape
      the effect (raise probability, add force) but do not
      participate in suppression decisions.

    Cause-disconnected events act as broken parents for the
    descendant rule but are NOT added to the returned closure — the
    do-flipped event itself remains in the persisted world; only its
    now-unsupported descendants are suppressed.

    Events with no participating chain_reaction parents (only
    ``contributory`` edges, or none at all) are exogenous and never
    enter the closure unless seeded.

    The returned set always includes the seeds (even if they have no
    incoming chain_reaction edges).
    """
    seeds = set(seed_ids)
    cause_broken = set(cause_disconnected_ids or [])
    if not seeds and not cause_broken:
        return set()

    closure: Set[str] = set(seeds)
    changed = True
    while changed:
        changed = False
        for eid, parents in parents_of.items():
            if eid in closure or eid in cause_broken:
                continue

            # Rule (a): any necessary parent gone → Y falls.
            necessary_pruned = any(
                nec == "necessary" and (
                    (p in closure) or (p in cause_broken)
                )
                for p, _mech, nec in parents
            )
            if necessary_pruned:
                closure.add(eid)
                changed = True
                continue

            # Rule (b): all sufficient parents gone → Y falls.
            sufficient = [
                p for p, _mech, nec in parents if nec == "sufficient"
            ]
            if sufficient and all(
                (p in closure) or (p in cause_broken) for p in sufficient
            ):
                closure.add(eid)
                changed = True
    return closure


def edges_within_closure(
    parents_of: ChainReactionParents,
    closure: Set[str],
) -> List[Tuple[str, str, str]]:
    """Return ``(parent_id, child_id, mechanism)`` triples for every
    chain_reaction edge whose endpoints both lie inside ``closure``.

    Used by directive_assembly to surface the original causal tree
    in brief constraints. ``necessity`` is intentionally not
    propagated here — the brief block is for renderer guidance, not
    closure replay.
    """
    edges: List[Tuple[str, str, str]] = []
    for child, parents in parents_of.items():
        if child not in closure:
            continue
        for parent, mech, _nec in parents:
            if parent in closure:
                edges.append((parent, child, mech))
    return edges
