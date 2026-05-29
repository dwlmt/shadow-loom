# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the disjunctive chain_reaction descendant
closure introduced after the 2026-05-29 counterfactual-prevention
audit.

Three regimes are exercised:

  * **expand_chain_reaction_closure** — the unified Pearl closure
    helper in :mod:`shadow_loom.causal_closure` (single parent
    cascades, multi-parent over-determination preserved, exogenous
    events never pruned, cause_disconnected acts as broken parent).
  * **build_causal_diagram Event→target wiring** — Rule 3 of the
    ctf-calculus must not declare ``do(EVT)/query(ENT)`` vacuous
    when ``EVT.target_ids`` names ENT (regression-locks the bug
    where ``EVT_DUNCAN_MURDER`` was wrongly Rule-3 pruned for
    ``query(ENT_DUNCAN)``).
  * **execute_interventions Step B.6** — the intervention-time
    closure must mark descendants ``pruned=True`` and surface them
    in ``CausalPhysicsResult.pruned_utterance_event_ids`` so the
    brief's SEVERED CAUSAL CHAINS block carries the whole cascade.
"""
from __future__ import annotations

import pytest

from shadow_loom.amwn import (
    apply_ctf_calculus,
    build_causal_diagram,
    check_exclusion,
)
from shadow_loom.causal_closure import (
    chain_reaction_parents_from_world_state,
    edges_within_closure,
    expand_chain_reaction_closure,
)
from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    Location,
    TraitVector,
    WorldStateV1,
)


# =====================================================================
# Fixtures
# =====================================================================


def _make_world(events, causal_topology, entities=None):
    """Tiny WorldStateV1 with a single location and the requested
    events + causal topology."""
    ents = entities or {
        "ENT_VICTIM": Entity(
            id="ENT_VICTIM", name="Victim", location_id="LOC_A",
            status="healthy",
            traits={"fear": TraitVector(value=0.3, inertia=0.1)},
        ),
    }
    return WorldStateV1(
        locations={
            "LOC_A": Location(
                id="LOC_A", name="A", description="A", ambient_state={},
            ),
        },
        objects={},
        entities=ents,
        events=events,
        causal_topology=causal_topology,
        spatial_topology=[],
    )


def _evt(eid: str, ft: int, *, targets=None, etype="outcome") -> EventNode:
    return EventNode(
        id=eid, fabula_time=ft, syuzhet_index=ft, event_type=etype,
        actor_ids=[], target_ids=targets or [],
        description=f"{eid} occurs",
    )


def _chain(src: str, tgt: str, *, ft: int = 1,
           necessity: str = "sufficient") -> CausalEdge:
    return CausalEdge(
        source_id=src, target_id=tgt,
        causality_type="chain_reaction", causal_force=5.0,
        mechanism="physical", evidence_strength="strong",
        necessity=necessity,
        fabula_time=ft,
    )


# =====================================================================
# expand_chain_reaction_closure (unit tests on the shared helper)
# =====================================================================


class TestExpandClosure:

    def test_empty_seeds_returns_empty(self):
        assert expand_chain_reaction_closure({}, set()) == set()

    def test_single_chain_parent_cascades(self):
        # A → B → C; do(A=prevented) must add {A, B, C} to closure.
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_B", 2), _evt("EVT_C", 3)],
            causal_topology=[
                _chain("EVT_A", "EVT_B"),
                _chain("EVT_B", "EVT_C"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_A"})
        assert closure == {"EVT_A", "EVT_B", "EVT_C"}

    def test_overdetermined_event_preserved(self):
        # A → C and B → C (both chain_reaction); do(A=prevented) must
        # NOT add C — Pearl over-determination keeps C alive because B
        # is still a surviving sufficient cause.
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_B", 1), _evt("EVT_C", 2)],
            causal_topology=[
                _chain("EVT_A", "EVT_C"),
                _chain("EVT_B", "EVT_C"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_A"})
        assert closure == {"EVT_A"}
        assert "EVT_C" not in closure

    def test_overdetermined_collapses_when_all_parents_pruned(self):
        # A → C and B → C; do({A, B}=prevented) collapses C too.
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_B", 1), _evt("EVT_C", 2)],
            causal_topology=[
                _chain("EVT_A", "EVT_C"),
                _chain("EVT_B", "EVT_C"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_A", "EVT_B"})
        assert closure == {"EVT_A", "EVT_B", "EVT_C"}

    def test_exogenous_event_never_pruned(self):
        # B has no chain_reaction parent — even if a sibling is pruned,
        # B stays alive.
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_B", 1)],
            causal_topology=[],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_A"})
        assert closure == {"EVT_A"}
        assert "EVT_B" not in closure

    def test_cause_disconnected_acts_as_broken_parent(self):
        # A → B, A cause-disconnected (not in closure itself), B
        # should still cascade because its only sufficient cause was
        # broken.
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_B", 2)],
            causal_topology=[_chain("EVT_A", "EVT_B")],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(
            parents, set(), cause_disconnected_ids={"EVT_A"},
        )
        assert "EVT_A" not in closure  # cause-disconnected ≠ pruned
        assert "EVT_B" in closure

    def test_non_chain_reaction_edges_ignored(self):
        # Affordance_gate State→Event is a modifier, not a sufficient
        # cause; pruning the modifier source must NOT propagate to
        # the descendant event in the disjunctive closure.
        # (Schema forbids ``mutation`` on Event→Event, which is the
        # other natural way an author might mis-tag a precondition;
        # the rule still applies for State→Event modifiers.)
        ws = _make_world(
            events=[_evt("EVT_B", 2)],
            causal_topology=[
                CausalEdge(
                    source_id="ENT_VICTIM", target_id="EVT_B",
                    causality_type="affordance_gate", causal_force=5.0,
                    mechanism="physical", evidence_strength="strong",
                    fabula_time=1,
                ),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        # Seed a non-chain-reaction-parented entity id; the helper
        # must NOT expand to EVT_B because the only edge is a
        # modifier (affordance_gate), not chain_reaction.
        closure = expand_chain_reaction_closure(parents, {"ENT_VICTIM"})
        assert closure == {"ENT_VICTIM"}
        assert "EVT_B" not in closure

    def test_edges_within_closure_returns_internal_edges(self):
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_B", 2), _evt("EVT_C", 3)],
            causal_topology=[
                _chain("EVT_A", "EVT_B"),
                _chain("EVT_B", "EVT_C"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_A"})
        edges = edges_within_closure(parents, closure)
        triples = {(p, c) for p, c, _ in edges}
        assert ("EVT_A", "EVT_B") in triples
        assert ("EVT_B", "EVT_C") in triples


# =====================================================================
# build_causal_diagram Event→target wiring (Rule 3 regression)
# =====================================================================


class TestEventTargetWiring:

    def test_event_to_target_edge_auto_wired(self):
        # EVT_MURDER targets ENT_VICTIM — the diagram must contain
        # the direct effect edge even without an explicit
        # causal_topology row.
        ws = _make_world(
            events=[_evt("EVT_MURDER", 1, targets=["ENT_VICTIM"])],
            causal_topology=[],
        )
        g = build_causal_diagram(ws)
        assert g.has_edge("EVT_MURDER", "ENT_VICTIM"), (
            "auto-wired Event→target edge missing"
        )

    def test_actor_edge_not_auto_wired(self):
        # An actor performing an event is not necessarily affected by
        # it; only target_ids should be auto-wired.
        ws = _make_world(
            events=[EventNode(
                id="EVT_GREET", fabula_time=1, syuzhet_index=1,
                event_type="utterance",
                actor_ids=["ENT_VICTIM"], target_ids=[],
                description="Alice greets Bob",
                content="Hello", speaker_id="ENT_VICTIM",
                truth_value="performative",
            )],
            causal_topology=[],
        )
        g = build_causal_diagram(ws)
        # Utterance events skip the auto-wire block entirely, and
        # actor edges are no longer wired regardless — so no
        # EVT_GREET → ENT_VICTIM edge from the auto-wire pass.
        assert not g.has_edge("EVT_GREET", "ENT_VICTIM")

    def test_rule3_not_vacuous_when_target_edge_present(self):
        # Regression: do(EVT_MURDER=prevented) / query(ENT_VICTIM)
        # must NOT be Rule-3 vacuous — the auto-wired Event→target
        # edge makes the event a structural ancestor of the entity.
        ws = _make_world(
            events=[_evt("EVT_MURDER", 1, targets=["ENT_VICTIM"])],
            causal_topology=[],
        )
        g = build_causal_diagram(ws)
        # check_exclusion returns True ⇔ intervention IS excluded
        # (vacuous) — we want it to be False here.
        excluded = check_exclusion(
            g,
            x_set={"EVT_MURDER"},
            y_set={"ENT_VICTIM"},
        )
        assert excluded is False, (
            "Rule 3 wrongly declared do(EVT_MURDER) vacuous for "
            "query(ENT_VICTIM) — the Event→target edge should keep "
            "the murder structurally upstream of its victim."
        )


# =====================================================================
# CausalPhysicsEngine Step B.6 — intervention-time closure
# =====================================================================


class TestInterventionTimeClosure:

    def test_pruned_set_expands_via_chain_reaction_closure(self):
        # Direct unit test of the shared helper as it is invoked at
        # intervention time: a do-surgery seeds {EVT_A}; the helper
        # must expand to the full cascade {EVT_A, EVT_B, EVT_C}.
        # This locks the contract that
        # ``execute_interventions`` Step B.6 relies on.
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_B", 2), _evt("EVT_C", 3)],
            causal_topology=[
                _chain("EVT_A", "EVT_B"),
                _chain("EVT_B", "EVT_C"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_A"})
        # The cascade must include both descendants — the brief's
        # SEVERED CAUSAL CHAINS block depends on this.
        assert "EVT_B" in closure
        assert "EVT_C" in closure


# =====================================================================
# necessity flag (rec #5 — INUS-aware closure)
# =====================================================================


class TestNecessityFlag:

    def test_default_necessity_is_sufficient(self):
        # Bare _chain() must default to "sufficient", preserving the
        # pre-flag disjunctive semantics so legacy fixtures keep
        # working unchanged.
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_B", 2)],
            causal_topology=[_chain("EVT_A", "EVT_B")],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        # parents map values are 3-tuples; the necessity slot must
        # default to "sufficient".
        nec_slots = {nec for ps in parents.values() for _, _, nec in ps}
        assert nec_slots == {"sufficient"}

    def test_necessary_parent_cascades_even_with_surviving_sufficient(self):
        # EVT_C has TWO sufficient parents (A, B) AND one necessary
        # parent (P, e.g. a precondition like "victim is present").
        # Pruning P must force C to fall even though A and B survive
        # — necessary preconditions cannot be substituted for.
        ws = _make_world(
            events=[
                _evt("EVT_A", 1), _evt("EVT_B", 1),
                _evt("EVT_P", 1), _evt("EVT_C", 2),
            ],
            causal_topology=[
                _chain("EVT_A", "EVT_C", necessity="sufficient"),
                _chain("EVT_B", "EVT_C", necessity="sufficient"),
                _chain("EVT_P", "EVT_C", necessity="necessary"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_P"})
        assert "EVT_C" in closure, (
            "pruning a necessary parent must cascade even when "
            "sufficient parents survive"
        )

    def test_surviving_necessary_parent_does_not_prevent_closure(self):
        # EVT_C has ONE sufficient parent (A) and one necessary
        # parent (P). do(A=prevented) leaves P intact, but the
        # sole sufficient parent is gone → rule (b) fires and C
        # joins the closure. (Necessary parents are not themselves
        # sufficient causes.)
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_P", 1), _evt("EVT_C", 2)],
            causal_topology=[
                _chain("EVT_A", "EVT_C", necessity="sufficient"),
                _chain("EVT_P", "EVT_C", necessity="necessary"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_A"})
        assert "EVT_C" in closure

    def test_contributory_parents_ignored(self):
        # A pure-contributory parent does not participate in
        # closure: pruning it leaves the effect alive (it only
        # raises probability / adds force).
        ws = _make_world(
            events=[_evt("EVT_A", 1), _evt("EVT_C", 2)],
            causal_topology=[
                _chain("EVT_A", "EVT_C", necessity="contributory"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(parents, {"EVT_A"})
        assert "EVT_C" not in closure, (
            "contributory parents must not trigger closure"
        )

    def test_contributory_only_parent_is_exogenous_for_closure(self):
        # Mix: EVT_C has one sufficient parent (A) and one
        # contributory (M). Pruning the contributory M alone must
        # NOT cascade. Pruning the sufficient A alone MUST cascade.
        ws = _make_world(
            events=[
                _evt("EVT_A", 1), _evt("EVT_M", 1), _evt("EVT_C", 2),
            ],
            causal_topology=[
                _chain("EVT_A", "EVT_C", necessity="sufficient"),
                _chain("EVT_M", "EVT_C", necessity="contributory"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)

        closure_m = expand_chain_reaction_closure(parents, {"EVT_M"})
        assert "EVT_C" not in closure_m

        closure_a = expand_chain_reaction_closure(parents, {"EVT_A"})
        assert "EVT_C" in closure_a

    def test_cause_disconnected_necessary_parent_triggers_closure(self):
        # Necessary-parent rule must also fire for parents in the
        # cause_disconnected_ids set (do-flipped events whose
        # outcome was inverted rather than erased).
        ws = _make_world(
            events=[
                _evt("EVT_A", 1), _evt("EVT_P", 1), _evt("EVT_C", 2),
            ],
            causal_topology=[
                _chain("EVT_A", "EVT_C", necessity="sufficient"),
                _chain("EVT_P", "EVT_C", necessity="necessary"),
            ],
        )
        parents = chain_reaction_parents_from_world_state(ws)
        closure = expand_chain_reaction_closure(
            parents, set(), cause_disconnected_ids={"EVT_P"},
        )
        assert "EVT_C" in closure
        # cause_disconnected itself does not enter the closure
        assert "EVT_P" not in closure
