# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime smoke harness for recent world-model changes.

Exercises against a broad range of real example worlds:
  * world_schema_audit pre-pass + warning surfacing
  * Proposition.truth_at_fabula coercion (incl. JSON round-trip with
    string keys and string boolean values — possible silent inversion).
  * interrogate_posterior polarity resolver with negated and
    affirmative perceived_state strings.
  * DoTarget union covers entity_delete / object_delete and survives a
    CounterfactualBranch serialize/parse round-trip.
  * generation._format_intervention_branch / _format_counterfactual /
    _format_do_target_causal_context handle delete kinds.
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from typing import Any, List

WORLDS = [
    "macbeth",
    "gone_girl",
    "reservoir_dogs",
    "persuasion",
    "the_devil_wears_prada",
    "nineteen_eighty_four",
    "romeo_and_juliet",
    "great_gatsby",
]

REPORT: List[str] = []

def log(msg: str) -> None:
    print(msg)
    REPORT.append(msg)

def header(title: str) -> None:
    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)


def load_world(name: str):
    return importlib.import_module(f"example_worlds.{name}").world_state


def smoke_schema_audit() -> None:
    header("[1] world_schema_audit on every world")
    from shadow_loom.world_schema_audit import audit_world_schema
    for name in WORLDS:
        try:
            ws = load_world(name)
            warnings = audit_world_schema(ws)
            log(f"  {name:30s} warnings={len(warnings)}")
            for w in warnings[:3]:
                log(f"      - {w}")
            if len(warnings) > 3:
                log(f"      ... (+{len(warnings) - 3} more)")
        except Exception as exc:
            log(f"  {name:30s} ERROR: {exc}")
            traceback.print_exc()


def smoke_truth_at_fabula_roundtrip() -> None:
    header("[2] truth_at_fabula JSON round-trip (key/value coercion)")
    from shadow_loom.models import Proposition

    # Case A: int keys, bool values — baseline.
    p_a = Proposition(
        proposition_id="PROP_A",
        kind="event_occurs",
        description="A holds",
        truth_at_fabula={100: True, 200: False},
    )
    log(f"  A baseline: {p_a.truth_at_fabula}")
    assert p_a.truth_at_fabula == {100: True, 200: False}

    # Case B: simulate JSON round-trip — string keys.
    js = p_a.model_dump_json()
    raw = json.loads(js)
    raw["truth_at_fabula"] = {"100": True, "200": False}
    p_b = Proposition.model_validate(raw)
    log(f"  B str-keys roundtrip: {p_b.truth_at_fabula}")
    assert p_b.truth_at_fabula == {100: True, 200: False}, "string-key int coercion failed"

    # Case C (BUG CHECK): string boolean values via bool() silently invert.
    raw_c = dict(raw)
    raw_c["truth_at_fabula"] = {"100": "false", "200": "true"}
    p_c = Proposition.model_validate(raw_c)
    log(f"  C str-bool 'false'/'true' → {p_c.truth_at_fabula}")
    # If bool('false') == True, this assertion fails — that's the bug.
    if p_c.truth_at_fabula == {100: False, 200: True}:
        log("    OK (strict string-bool parsing)")
    elif p_c.truth_at_fabula == {100: True, 200: True}:
        log("    *** BUG CONFIRMED: 'false' → True (bool() truthiness) ***")
    else:
        log(f"    unexpected: {p_c.truth_at_fabula}")

    # Case D: numeric 0/1 keys/values
    raw_d = dict(raw)
    raw_d["truth_at_fabula"] = {"100": 0, "200": 1}
    p_d = Proposition.model_validate(raw_d)
    log(f"  D numeric 0/1: {p_d.truth_at_fabula}")
    assert p_d.truth_at_fabula == {100: False, 200: True}

    # Case E: NaN — does float NaN coerce to True?
    raw_e = dict(raw)
    raw_e["truth_at_fabula"] = {"100": float("nan")}
    p_e = Proposition.model_validate(raw_e)
    log(f"  E NaN value → {p_e.truth_at_fabula} (bool(NaN)={bool(float('nan'))})")

    # Case F: real-plot serialization parity
    ws_macbeth = load_world("macbeth")
    if ws_macbeth.propositions:
        sample = next(iter(ws_macbeth.propositions.values()))
        orig = dict(sample.truth_at_fabula)
        dumped = sample.model_dump()
        # Simulate DB round-trip to JSON-string keys
        dumped["truth_at_fabula"] = {str(k): v for k, v in orig.items()}
        roundtripped = Proposition.model_validate(dumped)
        ok = roundtripped.truth_at_fabula == orig
        log(f"  F macbeth {sample.proposition_id} roundtrip OK={ok}")


def smoke_reconstruct_proposition_at() -> None:
    header("[3] reconstruct_proposition_at against real worlds")
    from shadow_loom.models import reconstruct_proposition_at
    for name in ("macbeth", "nineteen_eighty_four"):
        ws = load_world(name)
        # propositions can be either a Dict or List depending on model evolution
        props = ws.propositions
        if isinstance(props, dict):
            iter_props = list(props.items())[:3]
        else:
            iter_props = [(p.proposition_id, p) for p in (props or [])[:3]]
        if not iter_props:
            log(f"  {name}: no propositions, skip")
            continue
        for pid, prop in iter_props:
            if not prop.truth_at_fabula:
                continue
            max_t = max(prop.truth_at_fabula.keys())
            try:
                val = reconstruct_proposition_at(prop, max_t)
                log(f"  {name}/{pid}@{max_t}: {val}")
                # Also test below earliest tick
                min_t = min(prop.truth_at_fabula.keys())
                val_pre = reconstruct_proposition_at(prop, min_t - 1)
                log(f"  {name}/{pid}@{min_t - 1} (pre-earliest): {val_pre}")
            except Exception as exc:
                log(f"  {name}/{pid} ERROR: {exc}")


def smoke_posterior_polarity() -> None:
    header("[4] interrogate_posterior polarity on real worlds")
    from shadow_loom.interrogate_posterior import interrogate_posterior
    for name in ("macbeth", "gone_girl", "reservoir_dogs", "the_devil_wears_prada"):
        try:
            ws = load_world(name)
            rows = interrogate_posterior(ws, fabula_time=None)
            n_with_contradiction = sum(
                1 for r in rows
                if r.contradict_weight > 0 and r.support_weight > 0
            )
            n_all_support = sum(
                1 for r in rows
                if r.support_weight > 0 and r.contradict_weight == 0
            )
            log(f"  {name}: rows={len(rows)} "
                f"contradicted={n_with_contradiction} "
                f"all-supporting={n_all_support}")
            # Show one row where we expect potential negation handling
            for r in rows[:5]:
                log(f"    {r.proposition_id}: "
                    f"support={r.support_weight:.2f} "
                    f"contradict={r.contradict_weight:.2f} "
                    f"truth={r.truth_at_query}")
        except Exception as exc:
            log(f"  {name} ERROR: {exc}")
            traceback.print_exc()


def smoke_posterior_negation_unit() -> None:
    header("[5] _resolve_polarity unit on synthetic negations")
    # The helpers are defined inside interrogate_posterior() function
    # scope, so reach into the module-level lexicon directly if exposed,
    # else exercise via interrogate_posterior on a synthetic world.
    import shadow_loom.interrogate_posterior as ipm
    has_module_level = all(
        hasattr(ipm, n) for n in ("_resolve_polarity", "_NEGATION_TOKENS")
    )
    log(f"  module-level _resolve_polarity exposed: {has_module_level}")
    if not has_module_level:
        log("  *** FINDING: polarity helpers are nested inside "
            "interrogate_posterior() — cannot be unit tested independently. ***")
        return
    _resolve_polarity = ipm._resolve_polarity
    _NEGATION_TOKENS = ipm._NEGATION_TOKENS
    samples = [
        ("Duncan is alive", "Duncan is alive", True, True, "affirm-true"),
        ("Duncan is NOT alive", "Duncan is alive", True, False, "negated against True"),
        ("Duncan is not alive", "Duncan is alive", False, True, "negated, canonical false"),
        ("She denies the affair", "She had the affair", True, False, "DENIES (missing token)"),
        ("He refuses to confess", "He confesses", True, False, "REFUSES (missing token)"),
        ("She failed to deliver", "She delivers", True, False, "FAILED TO (missing)"),
        ("Father is absent", "Father is present", True, False, "ABSENT (missing)"),
        ("", "X holds", True, True, "empty perceived → unconditional support"),
        (None, "X holds", True, True, "None perceived → empty path"),
    ]
    for perceived, desc, canonical, expected, label in samples:
        try:
            result = _resolve_polarity(perceived or "", desc, canonical)
            mark = "OK" if result == expected else "*** MISMATCH ***"
            log(f"  {mark}  [{label}] "
                f"perceived={perceived!r} desc={desc!r} "
                f"canonical={canonical} → support={result} (want {expected})")
        except Exception as exc:
            log(f"  ERR [{label}]: {exc}")
    log(f"  _NEGATION_TOKENS contains denies? {' denies ' in _NEGATION_TOKENS or any('den' in t for t in _NEGATION_TOKENS)}")
    log(f"  _NEGATION_TOKENS contains absent? {any('absent' in t for t in _NEGATION_TOKENS)}")


def smoke_do_target_delete_parsing() -> None:
    header("[6] DoEntityDelete / DoObjectDelete via parser converter")
    from shadow_loom.query_parsing import _do_target_items_to_typed
    from shadow_loom.query_models import DoEntityDelete, DoObjectDelete

    items = [
        {"target_kind": "entity_delete", "entity_id": "ENT_DUNCAN"},
        {"target_kind": "object_delete", "object_id": "OBJ_DAGGER"},
        # fallback to target_id naming
        {"target_kind": "entity_delete", "target_id": "ENT_BANQUO"},
        {"target_kind": "object_delete", "target_id": "OBJ_CROWN"},
        # missing id — should be dropped silently
        {"target_kind": "entity_delete"},
    ]
    try:
        # Suppress the converter's dropped-record warnings during smoke
        # so they don't bury the log signal.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            out = _do_target_items_to_typed(items)
        log(f"  lifted {len(out)} of {len(items)} items")
        for t in out:
            log(f"    {type(t).__name__}: {t.model_dump(exclude_none=True)}")
        kinds = [type(t).__name__ for t in out]
        assert "DoEntityDelete" in kinds, "DoEntityDelete missing"
        assert "DoObjectDelete" in kinds, "DoObjectDelete missing"
        log("  OK")
    except Exception as exc:
        log(f"  ERR: {exc}")
        traceback.print_exc()


def smoke_counterfactual_branch_do_targets() -> None:
    header("[7] CounterfactualBranch.do_targets serialization round-trip")
    from shadow_loom.directive_assembly import CounterfactualBranch
    from shadow_loom.query_models import DoEntityDelete, DoObjectDelete, DoTrait

    targets = [
        DoEntityDelete(entity_id="ENT_DUNCAN"),
        DoObjectDelete(object_id="OBJ_DAGGER"),
        DoTrait(entity_id="ENT_MACBETH", trait_name="guilt", trait_value=0.0),
    ]
    try:
        # CounterfactualBranch has several required fields; build a minimal valid instance
        sig = CounterfactualBranch.model_fields
        log(f"  CounterfactualBranch fields: {list(sig.keys())[:10]} ...")
        # Try with just do_targets + any required defaults
        try:
            cb = CounterfactualBranch(do_targets=targets)
        except Exception as e:
            log(f"  CounterfactualBranch requires more fields: {e}")
            # Inspect required fields
            required = [k for k, f in sig.items() if f.is_required()]
            log(f"  required: {required}")
            return
        log(f"  built branch with {len(cb.do_targets)} do_targets")
        # Round-trip via JSON
        js = cb.model_dump_json()
        re_cb = CounterfactualBranch.model_validate_json(js)
        log(f"  round-tripped do_targets count: {len(re_cb.do_targets)}")
        for t in re_cb.do_targets:
            log(f"    {type(t).__name__}: {t.model_dump(exclude_none=True)}")
        kinds = [type(t).__name__ for t in re_cb.do_targets]
        assert kinds == ["DoEntityDelete", "DoObjectDelete", "DoTrait"], f"kinds={kinds}"
        log("  OK")
    except Exception as exc:
        log(f"  ERR: {exc}")
        traceback.print_exc()


def smoke_format_intervention_delete() -> None:
    header("[8] _format_intervention_branch + _format_do_target_causal_context (delete kinds)")
    from shadow_loom.generation import (
        _format_do_target_causal_context,
    )
    from shadow_loom.query_models import DoEntityDelete, DoObjectDelete
    ws = load_world("macbeth")
    # Pick real ids
    ent_id = next(iter(ws.entities.keys()))
    obj_id = next(iter(ws.objects.keys())) if ws.objects else None

    # _format_do_target_causal_context expects (target, world_state)
    try:
        s1 = _format_do_target_causal_context(DoEntityDelete(entity_id=ent_id), ws)
        log(f"  entity_delete context ({len(s1)} chars):")
        log("    " + (s1[:240].replace("\n", "\n    ") if s1 else "<empty>"))
        assert s1, "entity_delete context returned empty"
    except TypeError as te:
        log(f"  signature mismatch: {te}")
        # Try other signature shapes
        import inspect
        sig = inspect.signature(_format_do_target_causal_context)
        log(f"  signature: {sig}")
    except Exception as exc:
        log(f"  ERR entity_delete: {exc}")
        traceback.print_exc()

    if obj_id:
        try:
            s2 = _format_do_target_causal_context(DoObjectDelete(object_id=obj_id), ws)
            log(f"  object_delete context ({len(s2)} chars):")
            log("    " + (s2[:240].replace("\n", "\n    ") if s2 else "<empty>"))
        except Exception as exc:
            log(f"  ERR object_delete: {exc}")


def main() -> int:
    funcs = [
        smoke_schema_audit,
        smoke_truth_at_fabula_roundtrip,
        smoke_reconstruct_proposition_at,
        smoke_posterior_polarity,
        smoke_posterior_negation_unit,
        smoke_do_target_delete_parsing,
        smoke_counterfactual_branch_do_targets,
        smoke_format_intervention_delete,
    ]
    for fn in funcs:
        try:
            fn()
        except Exception as exc:
            log(f"FATAL in {fn.__name__}: {exc}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
