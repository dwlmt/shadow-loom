# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Brief-vocabulary hygiene regression tests.

The CreativeBrief is the renderer LLM's prompt; everything inside it
has a tendency to be echoed verbatim in the generated prose.  When
the brief uses meta-narrative or pipeline vocabulary (``the reader``,
``the audience``, ``alternate timeline``, ``ego-graph``, ``trait
vectors``, ``structural entanglement``…), the LLM mirrors it and the
NarrativeAuditor then flags the prose under the ``meta_narration``
violation type.

These tests lock the brief layer down: every ``ConstraintBlock``
``instruction`` and every ``RenderingDirective.stylistic_instructions``
entry MUST be phrased in scene-internal directive voice and MUST NOT
contain any of the banned brief-vocabulary tokens enumerated in
``shadow_loom/prompts/generation.md`` Rule 10.

If a future edit reintroduces ``"the reader feels…"`` or
``"alternate timeline"`` into the brief, this test will fail at
brief-assembly time --- long before a generation roundtrip --- and
catch the regression deterministically without an LLM call.
"""
from __future__ import annotations

import pytest

from shadow_loom.directive_assembly import DirectiveAssembler
from shadow_loom.extract_graph import extract_ego_graph_from_memory
from shadow_loom.query_models import DirectiveQuery

from example_worlds.macbeth import world_state as macbeth_ws
from example_worlds.nineteen_eighty_four import world_state as orwell_ws
from example_worlds.gone_girl import world_state as gone_girl_ws
from example_worlds.brief_encounter import world_state as brief_encounter_ws
from example_worlds.romeo_and_juliet import world_state as romeo_ws


# ---------------------------------------------------------------------
# Banned tokens — case-insensitive substring match against every
# instruction string the renderer LLM will see for an effect.
# Each token is tagged with the rationale so a failure message points
# the maintainer at the relevant subsystem.
# ---------------------------------------------------------------------
META_RECEIVER_TOKENS = [
    # Reader/audience-talk → leaks as "the reader felt the suspense build…"
    "the reader",
    "reader's",
    "the audience",
    "audience's",
]

PIPELINE_TOKENS = [
    # Pipeline / system vocabulary → leaks as "the simulation revealed…"
    "ego-graph",
    "ego graph",
    "trait vector",
    "trait vectors",
    "damage_potential",
    "structural entanglement",
    "structural pillar",
    "central node",
    "kl divergence",
    "prediction error",
    "the simulation",
    "the directive",
    "the brief",
]

COUNTERFACTUAL_TOKENS = [
    # Counterfactual-structure vocabulary → leaks as "in the alternate
    # timeline, he…" which is exactly the meta-narration pattern the
    # auditor flags.
    "alternate timeline",
    "alternative timeline",
    "alternate path",
    "alternative path",
    "the timeline",
    "the divergence",
    "the branch",
    "the fracture",
    "the possible world",
    "this reality",
    "another reality",
]

ALL_BANNED = META_RECEIVER_TOKENS + PIPELINE_TOKENS + COUNTERFACTUAL_TOKENS


ALL_EFFECTS = [
    "mystery",
    "dramatic_irony",
    "suspense",
    "surprise",
    "fear",
    "joy",
    "regret",
    "grief",
    "rage",
    "love",
]


# ---------------------------------------------------------------------
# Fixture matrix: pick a representative POV entity per world that
# carries enough beliefs / traits / causal edges to populate every
# branch of the assembler.
# ---------------------------------------------------------------------
WORLD_POV = [
    pytest.param(macbeth_ws, "ENT_MACBETH", id="macbeth"),
    pytest.param(orwell_ws, "ENT_WINSTON", id="1984"),
    pytest.param(gone_girl_ws, "ENT_NICK", id="gone_girl"),
    pytest.param(brief_encounter_ws, "ENT_LAURA", id="brief_encounter"),
    pytest.param(romeo_ws, "ENT_ROMEO", id="romeo_juliet"),
]


def _scan(text: str) -> list[str]:
    """Return banned tokens that appear in ``text`` (case-insensitive)."""
    haystack = text.lower()
    return [tok for tok in ALL_BANNED if tok.lower() in haystack]


@pytest.mark.parametrize("ws,pov", WORLD_POV)
@pytest.mark.parametrize("effect", ALL_EFFECTS)
def test_brief_has_no_meta_or_pipeline_vocabulary(ws, pov, effect):
    """No ConstraintBlock / stylistic_instruction may carry banned vocab.

    The brief is the renderer's prompt; banned tokens here become
    ``meta_narration`` violations downstream because the LLM mirrors
    them into prose.  Assert at brief-assembly time so regressions are
    caught without a generation roundtrip.
    """
    ego = extract_ego_graph_from_memory(ws, [pov]).model_dump()
    assembler = DirectiveAssembler(None, ego, ws)
    directive = DirectiveQuery(
        target_entity_ids=[pov],
        target_effect=effect,
        intensity=0.6,
    )
    brief = assembler.assemble(directive)

    offences: list[tuple[str, str, list[str]]] = []
    for c in brief.constraints:
        hits = _scan(c.instruction)
        if hits:
            offences.append(("ConstraintBlock", c.instruction, hits))
    if brief.rendering is not None:
        for si in brief.rendering.stylistic_instructions:
            hits = _scan(si)
            if hits:
                offences.append(("stylistic_instruction", si, hits))

    if offences:
        report = "\n".join(
            f"  [{kind}] tokens={hits}\n    text={text!r}"
            for kind, text, hits in offences
        )
        pytest.fail(
            f"Brief for effect={effect!r}, world POV={pov!r} contains "
            f"banned brief-vocabulary that the renderer LLM will mirror "
            f"into prose and the auditor will flag as meta_narration:\n"
            f"{report}"
        )


def test_banned_lists_are_mutually_exclusive_and_lowercase():
    """Sanity guard on the banned-token list itself."""
    seen: set[str] = set()
    for tok in ALL_BANNED:
        assert tok == tok.lower(), f"banned token {tok!r} must be lowercase"
        assert tok not in seen, f"duplicate banned token {tok!r}"
        seen.add(tok)


# ---------------------------------------------------------------------
# Counterfactual-brief hygiene — `build_counterfactual_brief` lives in
# the generation layer (not the directive assembler) and was a known
# leak point: its constraints, stylistic instructions, and
# `CounterfactualBranch.{actual,simulated}_outcome` strings used to
# carry "alternate timeline / divergence / branch / what-if /
# conditional or subjunctive mood" — every one of which is banned by
# generation Rule 10 / Category-4b meta-narration audit. Lock that
# down here so the renderer never sees forbidden vocabulary in a
# Rung-3 brief.
# ---------------------------------------------------------------------
@pytest.mark.parametrize("ws,pov", WORLD_POV)
def test_counterfactual_brief_has_no_meta_or_pipeline_vocabulary(ws, pov):
    from shadow_loom.generation import build_counterfactual_brief
    from shadow_loom.query_models import CounterfactualQuery

    # Pick any event id from the world so the intervention resolves to
    # at least one entity (otherwise target_entities is empty and the
    # rendering directive section short-circuits).
    if not ws.events:
        pytest.skip("fixture has no events to intervene on")
    evt_id = ws.events[0].id

    query = CounterfactualQuery(
        original_query=f"What if {evt_id} had not happened?",
        historical_interventions={f"{evt_id}.outcome": "did_not_occur"},
        evidence_node_ids=[],
        target_node_ids=[],
    )

    brief = build_counterfactual_brief(
        query=query,
        physics_state={},
        world_state=ws,
        hidden_deltas={pov: {"guilt": 0.3}},
        rule3_pruned_interventions=[f"{evt_id}.outcome"],
        rule2_redundant_evidence=[],
        rule3_pruning_mode="advisory",
    )

    offences: list[tuple[str, str, list[str]]] = []
    for c in brief.constraints:
        hits = _scan(c.instruction)
        if hits:
            offences.append(("ConstraintBlock", c.instruction, hits))
    if brief.rendering is not None:
        for si in brief.rendering.stylistic_instructions:
            hits = _scan(si)
            if hits:
                offences.append(("stylistic_instruction", si, hits))
    if brief.counterfactual_branch is not None:
        cf = brief.counterfactual_branch
        for label, text in (
            ("cf_branch.actual_outcome", cf.actual_outcome),
            ("cf_branch.simulated_outcome", cf.simulated_outcome),
        ):
            hits = _scan(text)
            if hits:
                offences.append((label, text, hits))

    if offences:
        report = "\n".join(
            f"  [{kind}] tokens={hits}\n    text={text!r}"
            for kind, text, hits in offences
        )
        pytest.fail(
            f"Counterfactual brief for world POV={pov!r} contains banned "
            f"brief-vocabulary that the renderer LLM will mirror into "
            f"prose and the auditor will flag as meta_narration:\n"
            f"{report}"
        )
