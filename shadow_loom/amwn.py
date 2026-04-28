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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

import networkx as nx

from shadow_loom.models import WorldStateV1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counterfactual variable representation
# ---------------------------------------------------------------------------

InterventionItem = Tuple[str, str]  # (target_path, hashable repr of value)
InterventionContext = FrozenSet[InterventionItem]


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
    """Normalise an intervention dict to a frozenset of ``(node_id, value)``."""
    items: Set[InterventionItem] = set()
    for path, value in interventions.items():
        # We only care about the intervened *node* for graph surgery —
        # property paths inside the node (``ENT_X.traits.fear``) collapse
        # to the same surgical cut on incoming edges.
        node_id = path.split(".", 1)[0] if "." in path else path
        items.add((node_id, _hashable_value(value)))
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

def build_causal_diagram(world_state: WorldStateV1) -> nx.DiGraph:
    """Strip ``world_state`` to a structural directed diagram ``G``.

    Nodes are entity / event / object / world-trait IDs; edges are the
    directed causal links from ``world_state.causal_topology``. Temporal
    metadata, mechanism, evidence_strength, etc. are *not* carried over —
    the AMWN reasons over structure only.

    Self-loops (a node listed as its own cause, which can happen with
    misformed extraction) are dropped to keep ``nx.ancestors`` well-defined.
    """
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
    for ce in world_state.causal_topology:
        if ce.source_id == ce.target_id:
            continue
        g.add_edge(ce.source_id, ce.target_id)
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
) -> CtfCalculusReport:
    """Apply the three ctf-calculus rules as a pre-flight check.

    For each intervention key, test whether the surgery is excluded by
    Rule 3 (no path to any evidence/target node). For each evidence node,
    test whether it is independent of the interventions by Rule 2 (and
    therefore redundant). The returned report lists pruning candidates;
    the caller decides whether to skip them.

    This is *static graphical reasoning* — no simulation or trait math.
    """
    diagram = build_causal_diagram(world_state)
    evidence_node_ids = list(evidence_node_ids or [])
    target_node_ids = list(target_node_ids or [])
    # The "downstream interest" set Y for Rule 3 is the union of evidence
    # and any explicit query targets. If neither is provided we cannot
    # prune via Rule 3 (every node is potentially relevant).
    y_universe = set(evidence_node_ids) | set(target_node_ids)

    report = CtfCalculusReport()

    intervened_node_ids = {
        path.split(".", 1)[0] if "." in path else path
        for path in interventions
    }

    if y_universe:
        for path in interventions:
            node_id = path.split(".", 1)[0] if "." in path else path
            if node_id.endswith(".spawn") or path.endswith(".spawn"):
                continue
            # Z = the *other* interventions (we mutilate edges into them too)
            other_interventions = intervened_node_ids - {node_id}
            if check_exclusion(diagram, {node_id}, y_universe, other_interventions):
                report.rule3_pruned.append(path)
                logger.info(
                    "[ctf-calculus·Rule3] Pruning intervention %s — disconnected "
                    "from {evidence ∪ targets} in mutilated diagram.", path,
                )

    if evidence_node_ids and intervened_node_ids:
        # Test Y_r ⊥ X_t (no extra conditioning) for each evidence node.
        x_query = [(nid, dict(interventions)) for nid in intervened_node_ids]
        for ev in evidence_node_ids:
            y_query = [(ev, {})]
            if check_ctf_independence(diagram, x_query, y_query):
                report.rule2_redundant_evidence.append(ev)
                logger.info(
                    "[ctf-calculus·Rule2] Evidence %s d-separated from "
                    "interventions on AMWN — abduction is redundant.", ev,
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
]
