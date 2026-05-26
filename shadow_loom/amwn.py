# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ancestral Multi-World Network (AMWN) and Counterfactual (ctf-) Calculus.

Implements the graphical machinery from Correa & Bareinboim, "Counterfactual
Graphical Models: Constraints and Inference" (ICML 2025), specialised for the
shadow-loom causal diagram (built from ``WorldStateV1.causal_topology``).

This module provides:

  • :func:`build_causal_diagram` — strips ``WorldStateV1`` to a directed
    structural diagram ``G`` over node-IDs (no temporal metadata).
  • :func:`build_amwn` — constructs the AMWN ``G^A(G, W*)`` for a set of
    counterfactual query variables ``W* = {V_t, ...}``, performing
    *node shadowing*: copies of the same variable across two intervention
    contexts are merged when the projection of those interventions onto
    the variable's ancestors agree.
  • :func:`check_ctf_independence` — Rule 2 (Independence): tests
    ``(Y_r ⊥ X_t | W*)`` via d-separation on the AMWN.
  • :func:`check_exclusion` — Rule 3 (Exclusion): tests
    ``X ∩ An(Y) = ∅`` in ``G_{Z̄}`` (the diagram with edges *into* Z cut).
  • :func:`check_consistency` — Rule 1 (Consistency): trivially holds for a
    pair ``(Y_{T*x}, X_{T*}=x)`` versus ``(Y_{T*}, X_{T*}=x)``.

The construction matches the paper's Definition A.1 (ancestral components)
in spirit but operates on a *latent-free* SCM (no bidirected ``U`` arcs), so
soundness for d-separation holds; completeness across worlds with shared
unobserved confounders would require explicit bidirected edges that the
shadow-loom data model does not currently encode.

**Closed-world assumption.** Identifiability via Rule 2 here is
*sound but incomplete with respect to unobserved confounders*. The AMWN is
built directly from ``WorldStateV1.causal_topology``, which is the engine's
ground truth; any latent common cause that the LLM extraction missed is
silently treated as absent. Two nodes that the AMWN flags as d-separated
are therefore independent **only relative to the extracted graph** — not
necessarily independent in the underlying narrative. Callers using the
ctf-calculus report to short-circuit simulation should treat Rule 2 / 3
flags as advisory, not authoritative.

Relationship metrics (affinity / fear / power_dynamic on social edges) are
lifted into synthetic ``REL::<src>::<tgt>::<metric>`` nodes by
:func:`build_causal_diagram` so that ``mutation_social`` causal edges can
contribute to d-separation reasoning. These are graph-only artefacts — they
never enter the simulation sandbox.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

import networkx as nx

from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counterfactual variable representation
# ---------------------------------------------------------------------------

InterventionItem = Tuple[str, str]  # (target_path, hashable repr of value)
InterventionContext = FrozenSet[InterventionItem]


# Sentinel returned by ``_resolve_observed_value`` when an intervention
# path cannot be resolved against the current world state (the node is
# absent, the attribute path doesn't exist, etc.). Rule 1 (Consistency)
# only applies when there is a concrete observed value to compare the
# do-value against, so unresolvable paths are skipped.
_SENTINEL_UNRESOLVED = object()


def _resolve_observed_value(world_state: WorldStateV1, path: str) -> Any:
    """Resolve a dotted intervention path to its current observed value.

    Supports the same path shapes that ``AMWNInstantiator.execute_interventions``
    accepts: bare ``ENT_X`` (returns the entity dump), ``ENT_X.spawn``
    (always unresolvable — spawn creates a new node), ``ENT_X.traits.<name>``,
    ``ENT_X.traits.<name>.value``, ``ENT_X.location_id``, ``ENT_X.status``,
    and ``WORLD_X.magnitude.value``. Any unsupported attribute returns
    the ``_SENTINEL_UNRESOLVED`` sentinel so the caller skips Rule 1
    rather than emitting a spurious "redundant" verdict.
    """
    if "." not in path:
        return _SENTINEL_UNRESOLVED  # Bare-node interventions have no scalar to compare.

    node_id, sub_path = path.split(".", 1)
    sub = sub_path.strip()
    if sub == "spawn":
        return _SENTINEL_UNRESOLVED

    # Resolve the host node from the world state.
    host: Any = None
    if node_id in world_state.entities:
        host = world_state.entities[node_id]
    elif node_id in (getattr(world_state, "objects", {}) or {}):
        host = world_state.objects[node_id]
    elif node_id in (getattr(world_state, "locations", {}) or {}):
        host = world_state.locations[node_id]
    elif node_id in (getattr(world_state, "world_traits", {}) or {}):
        host = world_state.world_traits[node_id]
    if host is None:
        return _SENTINEL_UNRESOLVED

    parts = sub.split(".")
    cursor: Any = host
    for part in parts:
        if isinstance(cursor, dict):
            if part not in cursor:
                return _SENTINEL_UNRESOLVED
            cursor = cursor[part]
            continue
        # Pydantic model or arbitrary object — try attribute access first,
        # then fall back to model_dump for dict-style access on traits.
        if hasattr(cursor, part):
            cursor = getattr(cursor, part)
            continue
        if hasattr(cursor, "model_dump"):
            dumped = cursor.model_dump()
            if isinstance(dumped, dict) and part in dumped:
                cursor = dumped[part]
                continue
        return _SENTINEL_UNRESOLVED
    return cursor


def _hashable_value(value: Any) -> str:
    """Coerce an intervention value to a stable hashable string.

    AMWN node identity depends on the *projection* of an intervention
    context onto a variable's ancestors; we therefore need a canonical
    string form that two equivalent contexts will share.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    try:
        if isinstance(value, dict):
            return repr(sorted(value.items()))
        if isinstance(value, (list, tuple, set)):
            return repr(sorted(value, key=repr))
    except TypeError:
        pass
    return repr(value)


def _to_context(interventions: Mapping[str, Any]) -> InterventionContext:
    """Normalise an intervention dict to a frozenset of ``(node_id, "")``.

    AMWN node identity for d-separation depends only on *which* variables
    are surgically cut, not the values they were set to: two interventions
    that hit the same variable produce identical do-surgery (incoming
    edges removed) regardless of value. We therefore use the empty string
    as a canonical value so floating-point drift in values cannot fragment
    the AMWN's node-shadowing behaviour. Value-sensitive reasoning lives
    in :func:`check_consistency`, which compares raw values directly.

    AUDIT (post-2026-05-26): preserve dotted-path granularity. The SCM
    treats ``ENT_alice.traits.anger`` and ``ENT_alice.traits.fear`` as
    *different* variables (per-axis trait nodes); collapsing both onto
    the entity id falsely identified non-overlapping do-surgeries as
    the same intervention and over-shadowed AMWN nodes. We now strip
    only the value-suffix (everything after the second dot for
    ``traits.``/``properties.``/``beliefs.``-style scoped axes is kept
    on the variable id). For non-scoped paths the full path is used.
    """
    items: Set[InterventionItem] = set()
    for path, _value in interventions.items():
        items.add((path, ""))
    return frozenset(items)


@dataclass(frozen=True)
class CounterfactualVar:
    """A counterfactual variable ``V_t`` — variable ``var_id`` evaluated under
    the intervention context ``context``.

    Two ``CounterfactualVar`` instances are equal iff they share both the
    variable id and the (frozen) intervention context — this is the key that
    the AMWN uses for node shadowing.
    """
    var_id: str
    context: InterventionContext = field(default_factory=frozenset)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if not self.context:
            return self.var_id
        ctx = ",".join(f"{k}={v}" for k, v in sorted(self.context))
        return f"{self.var_id}[{ctx}]"


# ---------------------------------------------------------------------------
# Causal diagram construction
# ---------------------------------------------------------------------------

def _rel_node_id(source_id: str, target_id: str, metric: str) -> str:
    """Synthetic AMWN node id for a relationship-metric variable.

    Relationship metrics live on edges in ``WorldStateV1.social_topology``
    rather than as first-class nodes. To make ``mutation_social`` causal
    edges visible to d-separation reasoning, :func:`build_causal_diagram`
    promotes each ``(source, target, metric)`` triple into a node with the
    canonical id produced by this helper.

    The ``REL::`` prefix guarantees the synthetic id can never collide
    with a real graph node (every real id uses a single-prefix convention
    like ``ENT_``, ``EVT_``, etc.).
    """
    return f"REL::{source_id}::{target_id}::{metric}"


def _prop_node_id(proposition_id: str) -> str:
    """Synthetic AMWN node id for a :class:`Proposition`.

    Propositions are audience-side utility-layer nodes (the things
    characters hold beliefs *about*). They are not part of the
    structural causal substrate, but Phase 4 lifts them into the AMWN
    so Pearl Rung-2 / Rung-3 surgeries via :class:`DoProposition` (and
    cascades into :class:`DoBelief`) become first-class queryable
    do-targets in d-separation reasoning.
    """
    return f"PROP::{proposition_id}"


def _ccn_node_id(concern_id: str) -> str:
    """Synthetic AMWN node id for a :class:`Concern`.

    Concerns are per-entity utility weights over propositions. Lifting
    them into the AMWN lets the renderer / directive assembler reason
    about commission-vs-omission counterfactuals at the utility layer
    (Roese), and lets ``find_pod`` query their activation windows.
    """
    return f"CCN::{concern_id}"


def build_causal_diagram(
    world_state: WorldStateV1,
    *,
    allow_unobserved_confounders: Optional[bool] = None,
) -> nx.DiGraph:
    """Strip ``world_state`` to a structural directed diagram ``G``.

    Nodes are entity / event / object / world-trait IDs; edges are the
    directed causal links from ``world_state.causal_topology``. Temporal
    metadata, mechanism, evidence_strength, etc. are *not* carried over —
    the AMWN reasons over structure only.

    Self-loops (a node listed as its own cause, which can happen with
    misformed extraction) are dropped to keep ``nx.ancestors`` well-defined.

    Relationship-metric variables are promoted into synthetic nodes
    ``REL::<src>::<tgt>::<metric>`` so ``mutation_social`` causal edges
    contribute to d-separation reasoning. Each is wired downstream from
    its triggering event (the ``CausalEdge.source_id``) and from its two
    endpoint entities, so the AMWN sees them as caused both by the event
    and by the relationship's participants.

    Parameters
    ----------
    allow_unobserved_confounders
        When ``True`` (or, by default, when
        ``CausalPhysicsSettings.allow_unobserved_confounders`` is set in
        ``config.env``), every pair of distinct nodes that share an
        observed parent is also given a *latent shared parent*
        ``U_<a>__<b>`` — modelling a potentially-unobserved confounder.
        This makes the diagram *sound-but-incomplete* explicit:
        d-separation will refuse to mark the two siblings independent
        because the latent ``U`` opens an active path between them.
        Default ``None`` defers to the settings value.
    """
    if allow_unobserved_confounders is None:
        try:
            from shadow_loom.settings import get_settings  # local to avoid cycles
            allow_unobserved_confounders = bool(
                get_settings().physics.allow_unobserved_confounders
            )
        except Exception:  # pragma: no cover - defensive: settings always loadable
            allow_unobserved_confounders = False

    g = nx.DiGraph()
    # Seed nodes so isolates are still queryable
    for nid in world_state.entities:
        g.add_node(nid)
    for evt in world_state.events:
        g.add_node(evt.id)
    for nid in world_state.objects:
        g.add_node(nid)
    for nid in world_state.locations:
        g.add_node(nid)
    for nid in getattr(world_state, "world_traits", {}) or {}:
        g.add_node(nid)
    # Channels are first-class diagram nodes so d-separation reasoning
    # over channel surgery (Rule 3) and channel-mediated evidence
    # relevance (Rule 2) flows through them. Each utterance event
    # gets a directed edge speaker → utterance → channel → addressee
    # so removing the channel cuts every downstream addressee belief
    # path the channel was carrying. Channels with no participants
    # are seeded as isolates (still queryable, never carry flow).
    for cid, ch in (getattr(world_state, "channels", {}) or {}).items():
        g.add_node(cid)
        for pid in getattr(ch, "participant_ids", []) or []:
            # Bidirectional standing capability: any participant can be
            # source or sink. Exact directionality of an utterance is
            # captured per-utterance below.
            g.add_edge(cid, pid)
    for ce in world_state.causal_topology:
        if ce.source_id == ce.target_id:
            continue
        g.add_edge(ce.source_id, ce.target_id)
        # Promote mutation_social edges into a synthetic relationship-metric
        # node so d-separation reasoning can flow through social state.
        if getattr(ce, "causality_type", None) == "mutation_social":
            counterpart = getattr(ce, "rel_counterpart_id", None)
            metric = getattr(ce, "trait_target", None)
            if counterpart and metric:
                rel_node = _rel_node_id(ce.target_id, counterpart, metric)
                g.add_node(rel_node)
                g.add_edge(ce.source_id, rel_node)
                # The two endpoint entities are also (weak) parents — their
                # current state co-determines the relationship metric.
                g.add_edge(ce.target_id, rel_node)
                g.add_edge(counterpart, rel_node)
    # Also seed REL nodes from the static social topology so observation-
    # only queries still see relationship metrics in the diagram. Skip
    # axes that were never observed (`metrics[axis].observed is False` or
    # the axis is absent from ``metrics``) — materialising a `REL_*`
    # node for an unmeasured axis injects a phantom variable into
    # latent / d-separation queries that never appeared in the source
    # text.
    for rel in getattr(world_state, "social_topology", []) or []:
        rel_metrics = getattr(rel, "metrics", {}) or {}
        for metric in ("affinity", "fear", "power_dynamic"):
            m = rel_metrics.get(metric)
            if m is None:
                continue
            if hasattr(m, "observed"):
                if not m.observed:
                    continue
            elif isinstance(m, dict) and not m.get("observed", True):
                continue
            rel_node = _rel_node_id(rel.source_entity_id, rel.target_entity_id, metric)
            if not g.has_node(rel_node):
                g.add_node(rel_node)
                g.add_edge(rel.source_entity_id, rel_node)
                g.add_edge(rel.target_entity_id, rel_node)

    # Per-utterance routing: speaker → utterance → channel → addressees.
    # An utterance event is the *vehicle* for information flow; without
    # this wiring a Rule 3 surgery on the channel would only break the
    # standing-capability edges (``channel → participant``) and miss
    # the per-utterance content path. With both layers present the
    # AMWN can answer "if we sever this channel, which addressee
    # beliefs become unreachable?" by checking d-separation against
    # the channel node.
    for evt in world_state.events:
        if getattr(evt, "event_type", None) != "utterance":
            continue
        ch_id = getattr(evt, "via_channel_id", None)
        sp_id = getattr(evt, "speaker_id", None)
        addressees = getattr(evt, "addressee_ids", []) or []
        if sp_id and g.has_node(sp_id):
            g.add_edge(sp_id, evt.id)
        if ch_id:
            if not g.has_node(ch_id):
                g.add_node(ch_id)
            g.add_edge(evt.id, ch_id)
            for aid in addressees:
                if g.has_node(aid):
                    g.add_edge(ch_id, aid)
        else:
            for aid in addressees:
                if g.has_node(aid):
                    g.add_edge(evt.id, aid)

    # ------------------------------------------------------------------
    # Phase-4 utility-layer lift: PROP::, CCN::, and belief→PROP edges.
    #
    # Propositions and concerns are audience-side / utility-layer nodes
    # (they don't directly cause events) but Pearl Rung-2 / Rung-3
    # surgeries via :class:`DoProposition`, :class:`DoBelief` and
    # :class:`DoConcern` need them as first-class do-targets. Mirroring
    # the existing ``REL::`` synthetic-node pattern, we promote each
    # proposition and concern into a synthetic AMWN node and wire:
    #
    #   * ``referent → PROP::<prop_id>``    (the proposition is *about*
    #     its referents — referent state co-determines proposition truth)
    #   * ``CCN::<ccn_id> → holder``        (a concern is held *by* an
    #     entity and shapes its disposition)
    #   * ``CCN::<ccn_id> ↔ PROP::<prop_id>`` (utility-over edge: the
    #     concern is a desire/fear *about* this proposition)
    #   * ``CCN::<a> ↔ CCN::<b>`` for each ``counter_concern_ids`` pair
    #     (ambivalence: undirected via two directed edges since DiGraph)
    #   * ``entity → PROP::<prop_id>`` for each :class:`Belief` carrying
    #     a ``proposition_id`` (epistemic-about edge)
    #
    # ``check_ctf_independence`` already filters via ``amwn.has_node`` so
    # synthetic nodes never absent from a query are safely ignored;
    # ``apply_ctf_calculus`` filters by ``diagram.has_node(node_id)`` so
    # legacy event/trait queries are unaffected.
    for prop in (getattr(world_state, "propositions", []) or []):
        prop_node = _prop_node_id(prop.proposition_id)
        if not g.has_node(prop_node):
            g.add_node(prop_node, node_kind="proposition")
        for ref in (prop.referent_ids or []):
            if g.has_node(ref):
                g.add_edge(ref, prop_node)

    for eid, ent in (world_state.entities or {}).items():
        for c in (getattr(ent, "concerns", None) or []):
            ccn_node = _ccn_node_id(c.concern_id)
            if not g.has_node(ccn_node):
                g.add_node(ccn_node, node_kind="concern", holder=eid)
            # Concern shapes the holder's disposition.
            if g.has_node(eid):
                g.add_edge(ccn_node, eid)
            # Utility-over: concern is *about* its proposition. Bidirectional
            # edges so d-separation reasoning sees the connection from
            # either end (DoConcern surgery should reach the proposition,
            # and DoProposition cascades should reach concerns).
            if c.proposition_id:
                prop_node = _prop_node_id(c.proposition_id)
                if not g.has_node(prop_node):
                    g.add_node(prop_node, node_kind="proposition")
                g.add_edge(ccn_node, prop_node)
                g.add_edge(prop_node, ccn_node)
            # Counter-concern (ambivalence) pairs.
            for cc_id in (c.counter_concern_ids or []):
                cc_node = _ccn_node_id(cc_id)
                if not g.has_node(cc_node):
                    g.add_node(cc_node, node_kind="concern")
                g.add_edge(ccn_node, cc_node)
                g.add_edge(cc_node, ccn_node)
        # Belief → Proposition (epistemic-about) edges.
        for b in (getattr(ent, "beliefs", None) or []):
            pid = getattr(b, "proposition_id", None)
            if not pid:
                continue
            prop_node = _prop_node_id(pid)
            if not g.has_node(prop_node):
                g.add_node(prop_node, node_kind="proposition")
            if g.has_node(eid):
                g.add_edge(eid, prop_node)

    # Optional: inject explicit ``U_*`` latent confounders. For every pair
    # of distinct nodes that share at least one *observed* parent in the
    # current diagram, materialise a single shared latent ``U_<a>__<b>``
    # parent. d-separation queries on the resulting diagram will then
    # refuse to mark the two nodes independent solely because their
    # observed parents are conditioned on — opening an active path
    # through the latent and yielding the sound-but-incomplete
    # behaviour callers ask for via the settings flag.
    if allow_unobserved_confounders:
        # Find sibling pairs (sharing a common parent) on the *original*
        # observed-only diagram; mutating ``g`` while iterating its
        # successors would otherwise create runaway latent injections.
        sibling_pairs: Set[Tuple[str, str]] = set()
        for parent in list(g.nodes()):
            children = sorted(c for c in g.successors(parent)
                              if not str(c).startswith(("U_", "REL::", "PROP::", "CCN::")))
            for i, a in enumerate(children):
                for b in children[i + 1:]:
                    sibling_pairs.add((a, b))
        for a, b in sibling_pairs:
            latent = f"U_{a}__{b}"
            if g.has_node(latent):
                continue
            g.add_node(latent, latent=True)
            g.add_edge(latent, a)
            g.add_edge(latent, b)
        logger.debug(
            "[AMWN] Injected %d latent U_* confounders for sibling pairs.",
            len(sibling_pairs),
        )

    return g


# ---------------------------------------------------------------------------
# AMWN construction
# ---------------------------------------------------------------------------

def _mutilate_into(diagram: nx.DiGraph, intervened: Iterable[str]) -> nx.DiGraph:
    """Return ``G_{T̄}`` — a copy of ``diagram`` with edges *into* every
    node in ``intervened`` removed. This is Pearl's do-surgery."""
    g = diagram.copy()
    for tnode in intervened:
        if g.has_node(tnode):
            g.remove_edges_from([(p, tnode) for p in list(g.predecessors(tnode))])
    return g


def _project_context(
    context: InterventionContext,
    variable: str,
    mutilated: nx.DiGraph,
) -> InterventionContext:
    """Project ``context`` onto ``variable``'s ancestors in ``G_{T̄}``.

    Per Definition A.1, two ancestor sets ``An(W_{t1})`` and ``An(W_{t2})``
    are merged when their interventions ``T1`` and ``T2`` agree on the
    relevant ancestors; the projection ``T ∩ An(V)_{G_{T̄}}`` is the
    minimal intervention context that affects ``V``.
    """
    if variable in mutilated:
        v_anc = nx.ancestors(mutilated, variable) | {variable}
    else:
        v_anc = {variable}
    return frozenset((nid, val) for (nid, val) in context if nid in v_anc)


def build_amwn(
    diagram: nx.DiGraph,
    query_vars: List[Tuple[str, Mapping[str, Any]]],
) -> nx.DiGraph:
    """Build the Ancestral Multi-World Network ``G^A(G, W*)``.

    Parameters
    ----------
    diagram :
        Structural causal diagram (output of :func:`build_causal_diagram`).
    query_vars :
        List of ``(variable_id, intervention_dict)`` pairs — each entry is a
        counterfactual ``W_t`` to include in the AMWN.

    Returns
    -------
    nx.DiGraph
        A directed graph whose nodes are :class:`CounterfactualVar`
        instances. Two ``(V, projection)`` copies coming from different
        worlds are *the same node* (shadowing).

        Each node carries:
          • ``var_id``      — the underlying structural variable
          • ``context``     — the projected intervention context
          • ``intervened``  — bool, True iff this node was directly cut
                              (incoming edges removed by do-surgery)
    """
    amwn = nx.DiGraph()

    for var_id, interventions in query_vars:
        full_context = _to_context(interventions)
        intervened_nodes = {nid for nid, _ in full_context}
        mutilated = _mutilate_into(diagram, intervened_nodes)

        if not mutilated.has_node(var_id):
            cf = CounterfactualVar(var_id, _project_context(full_context, var_id, mutilated))
            amwn.add_node(cf, var_id=var_id, context=cf.context, intervened=(var_id in intervened_nodes))
            continue

        # Ancestors of var_id in the mutilated diagram, including var_id itself.
        anc_set = nx.ancestors(mutilated, var_id) | {var_id}

        for v in anc_set:
            v_proj = _project_context(full_context, v, mutilated)
            v_node = CounterfactualVar(v, v_proj)
            amwn.add_node(
                v_node,
                var_id=v,
                context=v_proj,
                intervened=(v in intervened_nodes),
            )

            # If v is a do-surgery target, drop all incoming edges.
            if v in intervened_nodes:
                continue

            # Wire edges from the (also-projected) parents that are in
            # the ancestral set of var_id under T̄.
            for p in diagram.predecessors(v):
                if p not in anc_set:
                    continue
                p_proj = _project_context(full_context, p, mutilated)
                p_node = CounterfactualVar(p, p_proj)
                amwn.add_node(
                    p_node,
                    var_id=p,
                    context=p_proj,
                    intervened=(p in intervened_nodes),
                )
                amwn.add_edge(p_node, v_node)

    return amwn


# ---------------------------------------------------------------------------
# Rule 1 — Consistency
# ---------------------------------------------------------------------------

def check_consistency(
    target_var: str,
    intervention_value: Any,
    observed_value: Any,
) -> bool:
    """Rule 1 — ``P(Y_{T*x}, X_{T*}=x) = P(Y_{T*}, X_{T*}=x)``.

    Trivially: if a variable is *observed* at value ``x``, then additionally
    intervening on it to ``x`` is a no-op. This helper returns ``True`` when
    the observed and intervened values agree (so the intervention is
    redundant under Rule 1) and ``False`` otherwise.

    The shadow-loom engine uses this guard to suppress vacuous
    ``do(X = observed(X))`` interventions before graph surgery runs.
    """
    return _hashable_value(intervention_value) == _hashable_value(observed_value)


# ---------------------------------------------------------------------------
# Rule 2 — Independence (counterfactual d-separation on the AMWN)
# ---------------------------------------------------------------------------

def check_ctf_independence(
    diagram: nx.DiGraph,
    x_vars: List[Tuple[str, Mapping[str, Any]]],
    y_vars: List[Tuple[str, Mapping[str, Any]]],
    z_vars: Optional[List[Tuple[str, Mapping[str, Any]]]] = None,
) -> bool:
    """Rule 2 — ``(Y_r ⊥ X_t | W*)`` in ``G^A``.

    Builds the AMWN over ``X ∪ Y ∪ Z`` and asks NetworkX whether the
    counterfactual nodes ``Y_r`` are d-separated from ``X_t`` given ``W*``.

    Returns ``True`` when the independence holds (so by Rule 2 the
    conditioning on ``X_t`` may be dropped); ``False`` otherwise.
    """
    z_vars = list(z_vars or [])
    all_vars = list(x_vars) + list(y_vars) + z_vars
    amwn = build_amwn(diagram, all_vars)

    def _resolve(query: List[Tuple[str, Mapping[str, Any]]]) -> Set[CounterfactualVar]:
        out: Set[CounterfactualVar] = set()
        for var_id, interventions in query:
            ctx = _to_context(interventions)
            mutilated = _mutilate_into(diagram, {nid for nid, _ in ctx})
            proj = _project_context(ctx, var_id, mutilated)
            out.add(CounterfactualVar(var_id, proj))
        return out

    x_set = _resolve(x_vars)
    y_set = _resolve(y_vars)
    z_set = _resolve(z_vars)

    # Drop any resolved nodes that didn't make it into the AMWN
    # (variable absent from the diagram entirely).
    x_set = {v for v in x_set if amwn.has_node(v)}
    y_set = {v for v in y_set if amwn.has_node(v)}
    z_set = {v for v in z_set if amwn.has_node(v)}

    if not x_set or not y_set:
        # If either side is empty after filtering, treat as independent —
        # there is nothing to condition on.
        return True

    if x_set & y_set:
        # A variable can never be independent of itself.
        return False

    # NetworkX 3.x renamed ``d_separated`` → ``is_d_separator``.
    # Prefer the new spelling, fall back to the legacy name for older pins.
    _is_dsep = getattr(nx, "is_d_separator", None) or getattr(nx, "d_separated", None)
    if _is_dsep is None:  # pragma: no cover - networkx is a hard dep
        raise RuntimeError("networkx is missing a d-separation routine")
    try:
        return bool(_is_dsep(amwn, x_set, y_set, z_set))
    except nx.NetworkXError:
        # NetworkX raises if Z overlaps X∪Y. Treat as not-independent
        # (conservative: fall through to simulation).
        return False


# ---------------------------------------------------------------------------
# Rule 3 — Exclusion
# ---------------------------------------------------------------------------

def check_exclusion(
    diagram: nx.DiGraph,
    x_set: Iterable[str],
    y_set: Iterable[str],
    z_set: Optional[Iterable[str]] = None,
) -> bool:
    """Rule 3 — ``P(y_{xz}) = P(y_z)`` if ``X ∩ An(Y) = ∅`` in ``G_{Z̄}``.

    Returns ``True`` when the intervention on ``X`` is excluded (vacuous)
    given ``Z`` — i.e. no element of ``X`` is an ancestor of any element of
    ``Y`` once edges into ``Z`` are removed.
    """
    x_set = set(x_set)
    y_set = set(y_set)
    z_set = set(z_set or [])
    if not x_set:
        return True

    mutilated = _mutilate_into(diagram, z_set)

    ancestors_of_y: Set[str] = set()
    for y in y_set:
        if mutilated.has_node(y):
            ancestors_of_y |= nx.ancestors(mutilated, y) | {y}

    return x_set.isdisjoint(ancestors_of_y)


# ---------------------------------------------------------------------------
# Convenience: vacuity report for a Rung-2/3 query
# ---------------------------------------------------------------------------

@dataclass
class CtfCalculusReport:
    """Pre-flight report from applying ctf-calculus to a query."""
    rule3_pruned: List[str] = field(default_factory=list)
    rule2_redundant_evidence: List[str] = field(default_factory=list)
    rule1_redundant: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.rule3_pruned or self.rule2_redundant_evidence or self.rule1_redundant)


def apply_ctf_calculus(
    world_state: WorldStateV1,
    interventions: Mapping[str, Any],
    evidence_node_ids: Optional[Iterable[str]] = None,
    target_node_ids: Optional[Iterable[str]] = None,
    *,
    diagram: Optional[nx.DiGraph] = None,
) -> CtfCalculusReport:
    """Apply the three ctf-calculus rules as a pre-flight check.

    Rule 3 (Exclusion) prunes intervention keys whose target node has no
    directed path to any *query target* in the mutilated diagram, where
    edges into evidence and other-intervened nodes are cut.

    Rule 2 (Independence) flags evidence nodes that are d-separated from
    every intervened variable on the AMWN, given the full counterfactual
    context ``W*`` (all interventions) as the conditioning set.

    Spawn-style intervention paths (``X.spawn``) and intervention targets
    that don't yet exist in the diagram are skipped — they create new
    nodes that the static causal topology cannot reason about.

    The caller may pass a pre-built ``diagram`` to avoid rebuilding it
    when a single ``world_state`` is reused across many candidate
    interventions (e.g. inside ``evaluate_candidate_events``).
    """
    if diagram is None:
        diagram = build_causal_diagram(world_state)
    evidence_node_ids = list(evidence_node_ids or [])
    target_node_ids = list(target_node_ids or [])

    report = CtfCalculusReport()

    # Partition intervention paths into "real" (resolvable to a node we
    # can reason about) and "spawn" (new node — skip pre-flight).
    intervened_node_ids: Set[str] = set()
    spawn_paths: Set[str] = set()
    for path in interventions:
        node_id = path.split(".", 1)[0] if "." in path else path
        sub = path.split(".", 1)[1] if "." in path else ""
        if sub == "spawn" or path.endswith(".spawn"):
            spawn_paths.add(path)
            continue
        if not diagram.has_node(node_id):
            # Node not in static topology — can't analyse; preserve.
            continue
        intervened_node_ids.add(node_id)

    # ---- Rule 1 (Consistency) ----
    # ``do(X = observed(X))`` is a redundant no-op: surgery cannot change
    # what is already observed. We resolve each intervention path back to
    # the live world-state value and flag the path when ``check_consistency``
    # confirms the do-value matches. Spawn paths and unresolvable
    # attributes are skipped (they have no observed value to compare).
    for path, intervention_value in interventions.items():
        if path in spawn_paths:
            continue
        observed = _resolve_observed_value(world_state, path)
        if observed is _SENTINEL_UNRESOLVED:
            continue
        if check_consistency(path, intervention_value, observed):
            report.rule1_redundant.append(path)
            logger.info(
                "[ctf-calculus\u00b7Rule1] Intervention %s is redundant \u2014 "
                "observed value already equals %r.",
                path, intervention_value,
            )

    # ---- Rule 3 (Exclusion) ----
    # Y = explicit query targets only. Evidence is conditioning, not
    # query, so it goes into Z together with the *other* interventions.
    if target_node_ids:
        y_set = set(target_node_ids)
        for path in interventions:
            if path in spawn_paths:
                continue
            node_id = path.split(".", 1)[0] if "." in path else path
            if node_id not in intervened_node_ids:
                continue
            other_interventions = intervened_node_ids - {node_id}
            z_for_rule3 = other_interventions | set(evidence_node_ids)
            if check_exclusion(diagram, {node_id}, y_set, z_for_rule3):
                report.rule3_pruned.append(path)
                logger.info(
                    "[ctf-calculus·Rule3] Pruning intervention %s — no path "
                    "to query targets %s in mutilated diagram.",
                    path, sorted(y_set),
                )

    # ---- Rule 2 (Independence) ----
    # Test Y_r ⊥ X_t | W* for each evidence node, with the *other*
    # evidence as the conditioning set W*. Conditioning on the
    # interventions themselves would overlap X (NetworkXError); the
    # interventions appear only on the X side via their projection.
    if evidence_node_ids and intervened_node_ids:
        x_query = [(nid, dict(interventions)) for nid in intervened_node_ids]
        for ev in evidence_node_ids:
            if not diagram.has_node(ev):
                continue
            y_query = [(ev, {})]
            # W*: every *other* evidence node, with no intervention context
            # (we are asking about pre-surgery observed values).
            z_query = [(o, {}) for o in evidence_node_ids if o != ev]
            if check_ctf_independence(diagram, x_query, y_query, z_query):
                report.rule2_redundant_evidence.append(ev)
                logger.info(
                    "[ctf-calculus·Rule2] Evidence %s d-separated from "
                    "interventions on AMWN given other evidence — "
                    "abduction is redundant.",
                    ev,
                )

    return report


__all__ = [
    "CounterfactualVar",
    "CtfCalculusReport",
    "build_causal_diagram",
    "build_amwn",
    "check_consistency",
    "check_ctf_independence",
    "check_exclusion",
    "apply_ctf_calculus",
    "_rel_node_id",
]
