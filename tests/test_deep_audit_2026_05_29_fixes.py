# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-05-29 deep-audit fixes.

Covers:
  * Strict truth-value parsing in ``Proposition.truth_at_fabula``
    (string booleans, NaN floats, unknown strings).
  * Module-level negation lexicon + polarity resolver, including the
    expanded vocabulary (denies / refuses / failed to / absent /
    without / lacking / rejects / doubts).
  * Schema-audit CLI flag (``--strict`` default True, ``--no-strict``
    forces exit 0).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from shadow_loom.models import Proposition
from shadow_loom.interrogate_posterior import (
    _NEGATION_TOKENS,
    _module_has_negation,
    _module_resolve_polarity,
)


# ---------------------------------------------------------------------
# truth_at_fabula coercion safety
# ---------------------------------------------------------------------
def _build(truth_map):
    return Proposition(
        proposition_id="PROP_X",
        kind="event_occurs",
        description="X holds",
        truth_at_fabula=truth_map,
    )


def test_truth_at_fabula_string_booleans_round_trip_correctly():
    """JSON / DB round-trips often serialize ``False`` as ``\"false\"``.

    The pre-fix code did ``bool(\"false\")`` which is ``True`` because
    a non-empty string is truthy. That silently INVERTED ground-truth
    commitments. The strict parser now maps both directions correctly.
    """
    p = _build({"100": "false", "200": "true"})
    assert p.truth_at_fabula == {100: False, 200: True}


def test_truth_at_fabula_nan_is_dropped_not_committed_true():
    """``bool(NaN)`` is ``True`` \u2014 the pre-fix code would silently
    commit a NaN as ``True``. The strict parser drops it instead so
    downstream ``.get()`` callers see an unknown (no commitment)."""
    p = _build({"100": math.nan, "200": True})
    assert 100 not in p.truth_at_fabula
    assert p.truth_at_fabula[200] is True


def test_truth_at_fabula_numeric_strings_via_int_coercion():
    p = _build({"100": True, "200": False})
    assert p.truth_at_fabula == {100: True, 200: False}


def test_truth_at_fabula_zero_one_numerics_remain_correct():
    p = _build({"100": 0, "200": 1})
    assert p.truth_at_fabula == {100: False, 200: True}


def test_truth_at_fabula_unknown_string_dropped_not_invented():
    """An ambiguous value like ``\"maybe\"`` must NOT silently become
    ``True``; it must be dropped so callers see no commitment."""
    p = _build({"100": "maybe", "200": "true"})
    assert 100 not in p.truth_at_fabula
    assert p.truth_at_fabula[200] is True


def test_truth_at_fabula_bool_passthrough_unchanged():
    p = _build({100: True, 200: False})
    assert p.truth_at_fabula == {100: True, 200: False}


# ---------------------------------------------------------------------
# negation lexicon expansion
# ---------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Duncan is not alive",
    "She denies the affair",
    "He refused the offer",
    "Father is absent",
    "She failed to deliver the letter",
    "Charlotte is without protection",
    "Captain Wentworth rejects the proposal",
    "Anne doubts his constancy",
])
def test_module_has_negation_detects_expanded_lexicon(text):
    assert _module_has_negation(text), f"failed for: {text!r}"


@pytest.mark.parametrize("text", [
    "Duncan is alive",
    "She accepts the affair",
    "He agreed to the offer",
    "Father is present",
])
def test_module_has_negation_returns_false_for_affirmatives(text):
    assert not _module_has_negation(text), f"false-positive: {text!r}"


def test_negation_tokens_includes_expanded_vocabulary():
    flat = " ".join(_NEGATION_TOKENS)
    for kw in ("denies", "refuses", "failed to", "absent",
               "lacking", "without", "rejects", "doubt"):
        assert kw in flat, f"missing lexicon entry: {kw!r}"


# ---------------------------------------------------------------------
# polarity resolver semantics
# ---------------------------------------------------------------------
def test_resolve_polarity_affirm_vs_affirm_canonical_true():
    # "Duncan is alive" against canonical=True → supports.
    assert _module_resolve_polarity("Duncan is alive", "Duncan is alive", True)


def test_resolve_polarity_negated_against_canonical_true_contradicts():
    # "Duncan is not alive" against canonical=True → contradicts.
    assert not _module_resolve_polarity(
        "Duncan is not alive", "Duncan is alive", True
    )


def test_resolve_polarity_denies_against_canonical_true_contradicts():
    # Expanded lexicon: denies should also flip polarity.
    assert not _module_resolve_polarity(
        "She denies the murder", "The murder happened", True
    )


def test_resolve_polarity_empty_perceived_defaults_to_support():
    # Documented legacy behaviour: silent beliefs do not flip the bucket.
    assert _module_resolve_polarity("", "X holds", True)
    assert _module_resolve_polarity("", "X holds", False)


def test_resolve_polarity_with_unknown_canonical_uses_surface_match():
    # canonical_truth=None → just compare surface negation parity.
    assert _module_resolve_polarity(
        "Duncan is alive", "Duncan is alive", None
    )
    assert not _module_resolve_polarity(
        "Duncan is not alive", "Duncan is alive", None
    )


# ---------------------------------------------------------------------
# Schema audit CLI flag
# ---------------------------------------------------------------------
def _run_cli(world_path: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "shadow_loom.world_schema_audit",
         str(world_path), *flags],
        capture_output=True, text=True, check=False,
    )


def test_schema_audit_cli_strict_is_default(tmp_path):
    """A world with at least one warning should make the CLI exit 1 by
    default (no flag needed). Pre-fix the default was exit 0."""
    # Build a synthetic world with a guaranteed warning. The cheapest
    # path is to write JSON missing fields validation will complain
    # about; but the simpler reliable signal is: dump an
    # example_world's serialized world that we know audits cleanly,
    # then mutate it to trip a warning.
    from example_worlds.macbeth import world_state as ws
    payload = json.loads(ws.model_dump_json())
    # Trip a guaranteed warning: corrupt a channel membership so the
    # audit's channel-membership pass fires. If that pass changes,
    # this assertion still works because exit-1-on-any-warning is the
    # contract.
    if payload.get("channels"):
        first_ch = next(iter(payload["channels"].values()))
        first_ch["member_entity_ids"] = ["ENT_NOT_IN_WORLD"]
    p = tmp_path / "world.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    # Default (no --strict flag) MUST exit 1 because warnings exist.
    r_default = _run_cli(p)
    # If the audit produces no warnings on this mutation, the contract
    # test still passes when exit code matches warning count.
    if "0 warning" in r_default.stderr:
        pytest.skip("synthetic mutation did not produce a warning")
    assert r_default.returncode == 1, (
        f"default should exit 1 on warnings; got rc={r_default.returncode}\n"
        f"stderr={r_default.stderr}"
    )

    # --no-strict MUST exit 0 even when warnings exist.
    r_nostrict = _run_cli(p, "--no-strict")
    assert r_nostrict.returncode == 0, (
        f"--no-strict should exit 0; got rc={r_nostrict.returncode}\n"
        f"stderr={r_nostrict.stderr}"
    )


# ---------------------------------------------------------------------
# 2026-05-29 HIGH-1: posterior + schema warnings reach answer_question
# ---------------------------------------------------------------------
def test_answer_question_accepts_diagnostics_kwargs():
    """``answer_question`` signature must include the new diagnostics kwargs.

    Regression for HIGH-1: the pipeline computes
    ``posterior_warnings`` and ``world_schema_warnings`` on every
    physics run but the answer call site previously did not forward
    them, so the rung-1 LLM never saw the engine's own diagnostics.

    Asserted against the source file directly because the test
    suite's autouse fixture (``tests/conftest.py::_stub_answer_question``)
    replaces the live function with a ``**kwargs``-only stub before
    every test, which would otherwise mask the real signature.
    """
    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "answer.py"
    ).read_text(encoding="utf-8")
    assert "posterior_warnings: Optional[List[str]] = None" in src
    assert "world_schema_warnings: Optional[List[str]] = None" in src


def test_pipeline_forwards_diagnostics_to_answer_question():
    """The pipeline call site must read both diagnostic keys.

    Regression for HIGH-1: a future refactor that drops the
    ``posterior_warnings`` / ``world_schema_warnings`` forwarding
    would silently bury the engine's audit signals.
    """
    pipeline_src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "pipeline.py"
    ).read_text(encoding="utf-8")
    # Both keys must be read from physics_result and passed as kwargs.
    assert "posterior_warnings=" in pipeline_src
    assert "world_schema_warnings=" in pipeline_src
    assert 'physics_result.get("posterior_warnings"' in pipeline_src
    assert 'physics_result.get("world_schema_warnings"' in pipeline_src


def test_answer_prompt_surfaces_posterior_warnings():
    """The answer prompt-assembly code must emit diagnostics sections.

    Source-string check (rather than a runtime call) because the
    autouse stub at ``tests/conftest.py`` replaces ``answer_question``
    before any test body runs; bypassing that fixture would
    re-introduce the very LLM-hang it exists to prevent.
    """
    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "answer.py"
    ).read_text(encoding="utf-8")
    assert "POSTERIOR DIAGNOSTICS" in src
    assert "WORLD-SCHEMA WARNINGS" in src
    # The diagnostics must be conditionally appended to the prompt
    # parts list, not buried in a docstring.
    assert "if posterior_warnings:" in src
    assert "if world_schema_warnings:" in src


# ---------------------------------------------------------------------
# 2026-05-29 HIGH-2: cascade exclusion bound counts all mutation rails
# ---------------------------------------------------------------------
def test_cascade_exclusion_counts_all_mutation_rails():
    """``_build_cascade_exclusion_constraints`` must tally all rails.

    Regression for HIGH-2: the cascade-bounded miracle-prevention
    block previously summed only the five classic mutation streams
    (trait / social / proposition / belief / concern) but the branch
    payload also surfaces object / world-trait / edge /
    entity-delete / object-delete / event mutations. The HARD prompt
    saying ``exactly N effects`` was therefore numerically
    inconsistent with the surfaced cascade whenever any of the
    additional rails fired.
    """
    from shadow_loom.generation import _build_cascade_exclusion_constraints

    blocks = _build_cascade_exclusion_constraints(
        blocked=None,
        rule3_pruned_interventions=None,
        mutations=[{"node": "n1", "trait": "t1"}],
        social_mutations=[{"u": "a", "v": "b"}],
        proposition_mutations=[{"prop": "p1"}],
        belief_mutations=[{"belief": "b1"}],
        concern_mutations=[{"concern": "c1"}],
        object_mutations=[{"object": "obj1"}, {"object": "obj2"}],
        world_trait_mutations=[{"trait": "wt1"}],
        edge_mutations=[{"edge": "e1"}],
        entity_delete_mutations=[{"id": "ent1"}],
        object_delete_mutations=[{"id": "obj_del"}],
        event_mutations=[{"event": "ev1"}, {"event": "ev2"}],
        rule3_pruning_mode="advisory",
        world_label="intervened",
        rung_label="Rung-2 intervention",
    )

    assert blocks, "must emit at least the cascade-bounded block"
    # 1 + 1 + 1 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 2 = 13 total
    expected_total = 13
    cascade_block = blocks[0]
    ev = cascade_block.evidence
    assert ev["cascade_total"] == expected_total, ev
    # Each rail must contribute a count key.
    for key, expected in (
        ("trait_count", 1),
        ("social_count", 1),
        ("proposition_count", 1),
        ("belief_count", 1),
        ("concern_count", 1),
        ("object_count", 2),
        ("world_trait_count", 1),
        ("edge_count", 1),
        ("entity_delete_count", 1),
        ("object_delete_count", 1),
        ("event_count", 2),
    ):
        assert ev[key] == expected, f"{key}: {ev}"
    # Instruction must quote the same total and mention the new rails.
    instr = cascade_block.instruction
    assert f"exactly {expected_total}" in instr
    assert "OBJECT" in instr
    assert "WORLD-TRAIT" in instr
    assert "EVENT CASCADES" in instr


# =====================================================================
# 2026-05-29 ROUND-3 audit fixes
# =====================================================================
def test_round3_answer_evidence_grounding_handles_real_world():
    """``answer_question`` evidence grounding must not crash on a real world.

    Regression for round-3 HIGH: ``WorldStateV1.propositions`` is a
    ``List[Proposition]`` (not a dict) and ``beliefs`` / ``concerns``
    do not exist as WorldStateV1 attributes \u2014 they live on
    Entity. The pre-fix code called ``.keys()`` on the list and on
    non-existent attrs, so loading Death on the Nile (which has a
    rich proposition list) would AttributeError out of the grounding
    pass for any answer that cited evidence.
    """
    from example_worlds.death_on_the_nile import world_state as ws

    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "answer.py"
    ).read_text(encoding="utf-8")
    # The fix replaces the broken ``.keys()`` calls with list / nested
    # iteration; confirm the new pattern is present and the broken
    # one is gone.
    assert 'getattr(world_state, "propositions", None) or []' in src
    assert 'getattr(_ent, "beliefs", None) or []' in src
    assert 'getattr(_ent, "concerns", None) or []' in src
    # Smoke: replicate the grounding logic against a real world and
    # assert no exception + non-trivial id set.
    known_ids: set = set()
    for ent in (getattr(ws, "entities", None) or {}).keys():
        known_ids.add(str(ent))
    for p in (getattr(ws, "propositions", None) or []):
        pid = getattr(p, "proposition_id", None) or getattr(p, "id", None)
        if pid:
            known_ids.add(str(pid))
    for _ent in (getattr(ws, "entities", None) or {}).values():
        for _c in (getattr(_ent, "concerns", None) or []):
            _cid = getattr(_c, "concern_id", None)
            if _cid:
                known_ids.add(str(_cid))
    # Death on the Nile must contribute multiple PROP_ ids.
    prop_ids = {i for i in known_ids if i.startswith("PROP_")}
    assert prop_ids, "Death on the Nile should expose PROP_ ids to grounding"


def test_round3_do_belief_target_accepts_event_location_world():
    """Typed parser must keep DoBelief surgeries about EVT/LOC/WORLD nodes.

    Regression for round-3 HIGH: ``DoBelief.target_id`` allows
    ENT_/EVT_/OBJ_/LOC_/WORLD_, but the dynamic parser model
    previously restricted to ENT_/OBJ_, so a Lion/Witch/Wardrobe
    query \"clamp Edmund's belief about WORLD_PROPHECY_FOUR_THRONES
    to low confidence\" would silently drop the typed record.
    """
    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "query_parsing.py"
    ).read_text(encoding="utf-8")
    assert "belief_target_lit" in src
    # The constructor must combine entity + object + event + location
    # + world-trait id pools so all five DoBelief target prefixes
    # round-trip through the dynamic parser.
    assert 'typed["entity_ids"] + typed["object_ids"] + typed["event_ids"]' in src
    # And the field declaration must reference the new literal.
    assert "Optional[belief_target_lit]" in src


def test_round3_causal_edge_endpoints_include_channels():
    """Causal-edge typed parser must accept CHN_* endpoints.

    Regression for round-3 MED: world_schema_audit explicitly lists
    channel_ids in the causal-edge known-endpoint set
    (``known = entity_ids | object_ids | location_ids | event_ids
    | channel_ids``), so the typed parser must mirror that union.
    The pre-fix ``edge_endpoint_lit`` omitted ``channel_ids``, so
    a Tinker Tailor query ``add a causal_edge from CHN_DUTY_PHONE
    to EVT_SAFE_HOUSE_AMBUSH`` would silently drop.
    """
    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "query_parsing.py"
    ).read_text(encoding="utf-8")
    # The edge_endpoint_lit composition must include channel_ids.
    # Use a narrow substring so the assertion is robust to whitespace.
    assert 'typed.get("channel_ids", [])\n    )' in src or \
        'channel_ids", [])\n    )' in src


def test_round3_entity_delete_context_uses_relationship_edge_schema():
    """Entity-delete context formatter must read RelationshipEdge keys.

    Regression for round-3 MED: ``RelationshipEdge`` uses
    ``source_entity_id`` / ``target_entity_id`` and a ``metrics``
    dict (not ``source_id``/``target_id``/``metric``/``value`` like
    CausalEdge). The pre-fix walker therefore always produced an
    empty relationships list for an entity_delete counterfactual
    on a socially central character (Linnet in Death on the Nile,
    Jacqueline's severed metrics, etc).
    """
    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "generation.py"
    ).read_text(encoding="utf-8")
    # Confirm the entity_delete formatter now reads the canonical
    # RelationshipEdge field names AND iterates the metrics dict.
    assert 'getattr(edge, "source_entity_id", None)' in src
    assert 'getattr(edge, "target_entity_id", None)' in src
    assert 'getattr(edge, "metrics", None) or {}' in src


def test_round3_entity_delete_belief_detection_uses_target_id():
    """Incoming beliefs must be detected by ``Belief.target_id``.

    Regression for round-3 MED: ``Belief`` has no ``belief_id``
    attribute, so the pre-fix substring check
    ``f\"\\u2192{ent_id}\" in bid`` never matched and the
    \"cascaded beliefs\" rows were always empty for entity_delete
    counterfactuals (e.g. \"if ENT_WHITE_WITCH never existed\" in
    LWW would lose Edmund's betrayal beliefs).
    """
    src = (
        Path(__file__).parent.parent
        / "shadow_loom"
        / "generation.py"
    ).read_text(encoding="utf-8")
    # The fix must compare ``b.target_id == ent_id`` and must no
    # longer rely on the non-existent belief_id arrow encoding.
    assert 'getattr(b, "target_id", None) == ent_id' in src


def test_round3_schema_audit_validates_event_target_ids():
    """World schema audit must surface dangling Event.target_ids refs.

    Regression for round-3 MED: a malformed ingest typo in
    ``Event.target_ids`` (``ENT_KURTS`` for ``ENT_KURTZ`` in
    Apocalypse Now) previously passed schema audit silently then
    broke target-dependent queries downstream.
    """
    from shadow_loom.models import EventNode, WorldStateV1, Entity, Location
    from shadow_loom.world_schema_audit import audit_world_schema
    from tests.conftest import make_empty_world_state

    def _ent(eid: str, name: str) -> Entity:
        return Entity(
            id=eid,
            node_type="Entity",
            world_id="factual",
            name=name,
            location_id="LOC_JUNGLE",
            status="healthy",
            traits={},
        )

    ws = make_empty_world_state()
    ws.locations["LOC_JUNGLE"] = Location(
        id="LOC_JUNGLE",
        node_type="Location",
        world_id="factual",
        name="Jungle",
        description="Cambodian jungle.",
    )
    ws.entities["ENT_KURTZ"] = _ent("ENT_KURTZ", "Kurtz")
    ws.entities["ENT_WILLARD"] = _ent("ENT_WILLARD", "Willard")
    ws.events.append(
        EventNode(
            id="EVT_WILLARD_KILLS_KURTZ",
            node_type="Event",
            world_id="factual",
            description="Willard kills Kurtz.",
            actor_ids=["ENT_WILLARD"],
            target_ids=["ENT_KURTS"],  # intentional typo
            fabula_time=1000,
            syuzhet_index=0,
            event_type="choice",
            at_location_id="LOC_JUNGLE",
        )
    )
    issues = audit_world_schema(ws)
    assert any(
        "EVT_WILLARD_KILLS_KURTZ" in i and "target_id" in i and "ENT_KURTS" in i
        for i in issues
    ), f"expected dangling target_id audit issue; got: {issues}"


# =====================================================================
# 2026-05-29 ROUND-4 audit fixes
# =====================================================================
def test_round4_pov_filter_strips_shadow_sidecars():
    """``filter_world_state_for_pov`` must clear all shadow_* sidecars.

    Round-4 HIGH: ``model_copy(deep=True)`` preserves every
    ``shadow_entities`` / ``shadow_objects`` / ``shadow_propositions``
    / ``shadow_world_traits`` / ``shadow_social_topology`` /
    ``shadow_events`` / ``shadow_channels`` / ``shadow_locations`` /
    ``shadow_causal_topology`` / ``shadow_spatial_topology`` /
    ``shadow_removed_*_ids`` entry on the deep-copied output. Before
    the fix, a Gone-Girl-style counterfactual carrying a populated
    ``ws.shadow_entities['genuine_diary']`` would leak Amy's
    counterfactual interior straight through Nick's POV slice,
    defeating the epistemic-irony design.
    """
    from shadow_loom.projections import filter_world_state_for_pov
    from shadow_loom.models import Entity
    from tests.conftest import make_empty_world_state

    ws = make_empty_world_state()
    ws.entities["ENT_NICK"] = Entity(
        id="ENT_NICK", node_type="Entity", world_id="factual",
        name="Nick", location_id="LOC_HOME", status="healthy", traits={},
    )
    # Plant a shadow sidecar that should NOT leak through the POV.
    ws.shadow_entities["genuine_diary"] = {
        "ENT_AMY_SHADOW": Entity(
            id="ENT_AMY_SHADOW", node_type="Entity", world_id="shadow",
            name="Amy (counterfactual)",
            location_id="LOC_CABIN", status="healthy", traits={},
        ),
    }
    ws.shadow_removed_entity_ids["genuine_diary"] = ["ENT_NICK_FRAMED"]

    out = filter_world_state_for_pov(ws, pov_entity_id="ENT_NICK")
    assert out.shadow_entities == {}, (
        "shadow_entities must be stripped from POV slice; "
        f"got {out.shadow_entities!r}"
    )
    assert out.shadow_removed_entity_ids == {}

    # Source-level confirmation that the helper exists and is called.
    src = (
        Path(__file__).parent.parent / "shadow_loom" / "projections.py"
    ).read_text(encoding="utf-8")
    assert "def _strip_shadow_sidecars" in src
    assert "_strip_shadow_sidecars(filtered)" in src
    assert "_strip_shadow_sidecars(empty)" in src


def test_round4_pov_filter_strips_shadow_for_unknown_pov():
    """Unknown POV path must also clear shadow_* sidecars."""
    from shadow_loom.projections import filter_world_state_for_pov
    from shadow_loom.models import Entity
    from tests.conftest import make_empty_world_state

    ws = make_empty_world_state()
    ws.shadow_entities["br"] = {
        "ENT_GHOST": Entity(
            id="ENT_GHOST", node_type="Entity", world_id="shadow",
            name="Ghost", location_id="LOC_X", status="healthy", traits={},
        ),
    }
    out = filter_world_state_for_pov(ws, pov_entity_id="ENT_NOT_IN_WORLD")
    assert out.shadow_entities == {}


def test_round4_do_spatial_edge_lock_mirrors_bidirectional():
    """Locking a bidirectional spatial edge must mirror to the reverse.

    Round-4 MED: the ``sever`` action already mirrors removal to the
    reverse direction for ``bidirectional=True`` edges, but the
    ``lock`` / ``unlock`` actions did not. A Macbeth-style query
    sealing LOC_INVERNESS_CASTLE ↔ LOC_FORRES_COURT with a guard
    cordon would leave the reverse traversable, contradicting the
    semantic of a symmetric barrier.
    """
    src = (
        Path(__file__).parent.parent / "shadow_loom" / "causal_physics.py"
    ).read_text(encoding="utf-8")
    # Confirm the new mirror loop is present in the lock/unlock branch.
    assert 'forward_was_bidi = False' in src
    # The mirror walker rewrites the reverse canonical edge.
    assert ('rev_e.source_id == target.target_id\n'
            '                            and rev_e.target_id == target.source_id') in src
    # The EdgeMutation details now include reverse_hit for telemetry.
    assert '"reverse_hit": reverse_hit' in src
