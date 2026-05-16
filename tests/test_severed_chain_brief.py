# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the severed-chain CONTEXT + positive-substitution brief
blocks.

Both builders close the gap that lets the prose renderer confabulate
a substitute failure mode for any character whose plan depended on a
do-pruned event. The "A Fish Called Wanda" counterfactual that
prevents Ken from killing Mrs Coady's dogs is the canonical example:

* Without these blocks, the renderer saw only ``=== PREVENTED EVENTS
  (HARD) ===`` (a bare list of pruned event ids) and confabulated a
  novel "missing canine evidence" failure mode for the prosecution.
* With them, the renderer additionally sees the severed
  parent\u2192child chain_reaction links and the CURRENT post-prune
  status of every implicated entity, so the prose can be rendered
  with Mrs Coady alive and the prosecution proceeding from her
  actual testimony.

These tests verify the deterministic constraint emitters in
isolation; their wiring into the intervention / counterfactual
brief paths is covered by the brief-build tests.
"""

from __future__ import annotations

from types import SimpleNamespace

from shadow_loom.directive_assembly import (
    _expand_chain_reaction_closure,
    build_dependent_state_substitution_constraints,
    build_prune_cascade_context_constraints,
)


def _evt(eid: str, *, actor_ids=(), target_ids=(), desc=""):
    return SimpleNamespace(
        id=eid, event_type="outcome", fabula_time=0,
        actor_ids=list(actor_ids), target_ids=list(target_ids),
        description=desc,
    )


def _edge(src: str, tgt: str, *, mechanism="physical"):
    return SimpleNamespace(
        source_id=src, target_id=tgt,
        causality_type="chain_reaction",
        mechanism=mechanism, fabula_time=0,
    )


def _ws(events=(), edges=(), entities=None):
    return SimpleNamespace(
        events=list(events),
        causal_topology=list(edges),
        entities=entities or {},
    )


# ---------------------------------------------------------------------
# _expand_chain_reaction_closure — closure + edge tracking
# ---------------------------------------------------------------------


def test_expand_closure_returns_roots_when_no_edges():
    ws = _ws(events=[_evt("EVT_A")], edges=[])
    closure, edges = _expand_chain_reaction_closure(ws, ["EVT_A"])
    assert closure == {"EVT_A"}
    assert edges == []


def test_expand_closure_walks_chain_and_returns_edges():
    ws = _ws(
        events=[_evt("EVT_A"), _evt("EVT_B"), _evt("EVT_C")],
        edges=[
            _edge("EVT_A", "EVT_B", mechanism="physical"),
            _edge("EVT_B", "EVT_C", mechanism="legal"),
        ],
    )
    closure, edges = _expand_chain_reaction_closure(ws, ["EVT_A"])
    assert closure == {"EVT_A", "EVT_B", "EVT_C"}
    assert ("EVT_A", "EVT_B", "physical") in edges
    assert ("EVT_B", "EVT_C", "legal") in edges


def test_expand_closure_disjunctive_survivor_blocks():
    """A->C, B->C; prune only A => C survives, no edge for B->C
    reported (C not in closure)."""
    ws = _ws(
        events=[_evt(x) for x in ("EVT_A", "EVT_B", "EVT_C")],
        edges=[_edge("EVT_A", "EVT_C"), _edge("EVT_B", "EVT_C")],
    )
    closure, edges = _expand_chain_reaction_closure(ws, ["EVT_A"])
    assert closure == {"EVT_A"}
    assert edges == []


# ---------------------------------------------------------------------
# build_prune_cascade_context_constraints
# ---------------------------------------------------------------------


def test_prune_cascade_empty_roots_emits_nothing():
    ws = _ws()
    assert build_prune_cascade_context_constraints(ws, None) == []
    assert build_prune_cascade_context_constraints(ws, []) == []


def test_prune_cascade_emits_severed_chains_block():
    ws = _ws(
        events=[
            _evt("EVT_KEN_KILLS_DOGS", desc="Ken kills Mrs Coady's dogs"),
            _evt("EVT_MRS_COADY_DIES", desc="Mrs Coady dies of heart attack"),
            _evt("EVT_TRIAL_COLLAPSES", desc="Trial collapses without witness"),
        ],
        edges=[
            _edge("EVT_KEN_KILLS_DOGS", "EVT_MRS_COADY_DIES",
                  mechanism="physical"),
            _edge("EVT_MRS_COADY_DIES", "EVT_TRIAL_COLLAPSES",
                  mechanism="legal"),
        ],
    )
    blocks = build_prune_cascade_context_constraints(
        ws, ["EVT_KEN_KILLS_DOGS"], world_label="counterfactual",
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b.priority == "hard"
    text = b.instruction
    assert "SEVERED CAUSAL CHAINS (HARD" in text
    assert "EVT_KEN_KILLS_DOGS" in text
    assert "EVT_MRS_COADY_DIES" in text
    assert "EVT_TRIAL_COLLAPSES" in text
    assert "[physical]" in text
    assert "[legal]" in text
    assert "counterfactual" in text
    # Evidence must include the full closure for downstream tooling.
    closure = set(b.evidence["pruned_closure_event_ids"])
    assert closure == {
        "EVT_KEN_KILLS_DOGS", "EVT_MRS_COADY_DIES", "EVT_TRIAL_COLLAPSES",
    }
    # And the edges in machine-readable form.
    edges_payload = b.evidence["severed_chain_reaction_edges"]
    assert any(
        e["source"] == "EVT_KEN_KILLS_DOGS"
        and e["target"] == "EVT_MRS_COADY_DIES"
        for e in edges_payload
    )


def test_prune_cascade_omits_anti_confabulation_phrases_correctly():
    """The instruction explicitly forbids "missing evidence" /
    "absent witness" reframings — the canonical confabulation modes."""
    ws = _ws(
        events=[_evt("EVT_A", desc="root")],
        edges=[],
    )
    blocks = build_prune_cascade_context_constraints(
        ws, ["EVT_A"], world_label="counterfactual",
    )
    assert len(blocks) == 1
    text = blocks[0].instruction
    assert "missing evidence" in text
    assert "absent witness" in text


# ---------------------------------------------------------------------
# build_dependent_state_substitution_constraints
# ---------------------------------------------------------------------


def _ent(eid: str, *, name: str, status: str = "alive", location: str = "LOC_X"):
    return SimpleNamespace(
        id=eid, name=name, status=status, location_id=location,
    )


def test_dependent_state_empty_inputs_emit_nothing():
    ws = _ws(entities={"ENT_X": _ent("ENT_X", name="X")})
    assert build_dependent_state_substitution_constraints(ws, None) == []
    assert build_dependent_state_substitution_constraints(ws, []) == []


def test_dependent_state_surfaces_event_participants():
    """Actors / targets of any event in the prune closure must appear
    in the substitution block with their CURRENT world_state status."""
    ws = _ws(
        events=[
            _evt(
                "EVT_KILL_DOGS",
                actor_ids=["ENT_KEN"],
                target_ids=["ENT_MRS_COADY"],
                desc="dogs killed",
            ),
        ],
        entities={
            "ENT_KEN": _ent("ENT_KEN", name="Ken"),
            "ENT_MRS_COADY": _ent(
                "ENT_MRS_COADY", name="Mrs Coady",
                status="alive", location="LOC_FLAT",
            ),
        },
    )
    blocks = build_dependent_state_substitution_constraints(
        ws, ["EVT_KILL_DOGS"], world_label="counterfactual",
    )
    assert len(blocks) == 1
    text = blocks[0].instruction
    assert "DEPENDENT-STATE SUBSTITUTIONS (HARD)" in text
    assert "ENT_MRS_COADY" in text
    assert "Mrs Coady" in text
    assert "'alive'" in text
    assert "LOC_FLAT" in text
    assert "ENT_KEN" in text
    # And the explicit anti-confabulation directive.
    assert "substitute failure mode" in text
    assert sorted(blocks[0].evidence["implicated_entity_ids"]) == [
        "ENT_KEN", "ENT_MRS_COADY",
    ]


def test_dependent_state_parses_belief_holder_arrow_syntax():
    """``affected_beliefs`` entries are formatted ``holder\u2192target``
    by narrative_physics. The builder must parse the holder out."""
    ws = _ws(entities={"ENT_GEORGE": _ent("ENT_GEORGE", name="George")})
    blocks = build_dependent_state_substitution_constraints(
        ws, [],
        affected_beliefs=["ENT_GEORGE\u2192ENT_MRS_COADY"],
        world_label="counterfactual",
    )
    assert len(blocks) == 1
    text = blocks[0].instruction
    assert "ENT_GEORGE" in text
    assert "George" in text


def test_dependent_state_parses_concern_holder_dot_syntax():
    """``affected_concerns`` entries are formatted ``holder.concern_id``."""
    ws = _ws(entities={"ENT_WANDA": _ent("ENT_WANDA", name="Wanda")})
    blocks = build_dependent_state_substitution_constraints(
        ws, [],
        affected_concerns=["ENT_WANDA.CCN_DIAMONDS"],
        world_label="counterfactual",
    )
    assert len(blocks) == 1
    assert "ENT_WANDA" in blocks[0].instruction


def test_dependent_state_skips_non_entity_ids():
    """OBJ_ / LOC_ / WORLD_ ids appearing in target_ids must not
    populate the entity list."""
    ws = _ws(
        events=[
            _evt(
                "EVT_X",
                actor_ids=["ENT_REAL"],
                target_ids=["OBJ_KEY", "LOC_SAFE", "WORLD_WEATHER"],
            ),
        ],
        entities={"ENT_REAL": _ent("ENT_REAL", name="Real")},
    )
    blocks = build_dependent_state_substitution_constraints(
        ws, ["EVT_X"],
    )
    assert len(blocks) == 1
    text = blocks[0].instruction
    assert "ENT_REAL" in text
    assert "OBJ_KEY" not in text
    assert "LOC_SAFE" not in text
    assert "WORLD_WEATHER" not in text
