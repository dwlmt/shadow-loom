#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoke-test the 6 query types not exercised by the recent chain.

Runs each untested query type through ``run_pipeline`` against the
``a_fish_called_wanda`` example world (same world as the audited
chain). Audit + re-extraction are skipped to keep wall-clock low — the
goal is to confirm that the brief builds, physics state assembles,
and (where applicable) prose generation returns non-empty output.

Untested types (from chain audit):
    intervention, directive, interrogate, general, manual_edit, evaluate

Output is a one-line PASS/FAIL summary per type plus a short reason on
failure. Exits non-zero if any type fails.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
# Quiet the very chatty modules to keep the smoke summary readable.
for noisy in (
    "shadow_loom.causal_physics",
    "shadow_loom.amwn",
    "shadow_loom.extract_graph",
    "shadow_loom.ingestion",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from example_worlds.a_fish_called_wanda import world_state as wanda_ws  # noqa: E402
from shadow_loom.pipeline import (  # noqa: E402
    PipelineConfig,
    run_pipeline,
)
from shadow_loom.auditor import AuditorConfig  # noqa: E402
from shadow_loom.generation import GenerationConfig  # noqa: E402
from shadow_loom.ingestion import ExtractionConfig  # noqa: E402
from shadow_loom.query_models import (  # noqa: E402
    DirectiveQuery,
    DoEvent,
    DoProposition,
    EvaluationQuery,
    GeneralQuery,
    InterrogationQuery,
    InterventionQuery,
    ManualEditQuery,
)


# A lightweight default config — skip audit + reextraction so the
# smoke run finishes in minutes rather than tens of minutes. The goal
# is to surface dispatch / brief / physics failures, not full fidelity.
def _cfg() -> PipelineConfig:
    return PipelineConfig(
        use_causal_engine=True,
        generation_config=GenerationConfig(max_tokens=1024),
        auditor_config=AuditorConfig(
            max_iterations=1,
            max_tokens_audit=1024,
            max_tokens_generation=1024,
        ),
        skip_audit=True,
        skip_reextraction=True,
        extraction_config=ExtractionConfig(),
    )


def _fresh_ws():
    return wanda_ws.model_copy(deep=True)


# ---------------------------------------------------------------------
# Per-type test bodies — each returns a short status string on success
# and raises on failure.
# ---------------------------------------------------------------------

def test_intervention() -> str:
    """Rung-2 do-surgery: force Ken's loyalty trait via a typed do_target."""
    q = InterventionQuery(
        do_targets=[
            DoProposition(
                proposition_id="PROP_KEN_LOYAL_TO_GEORGE",
                truth=False,
            ),
        ],
        target_node_ids=["ENT_KEN", "ENT_GEORGE"],
    )
    result = run_pipeline(q, world_state=_fresh_ws(), config=_cfg())
    assert result.prose, "no prose returned"
    assert result.query_type == "intervention"
    return f"prose={len(result.prose)}ch"


def test_directive() -> str:
    """Affective optimisation: maximise suspense around Ken."""
    q = DirectiveQuery(
        target_entity_ids=["ENT_KEN"],
        target_effect="suspense",
        target_vector_id="ENT_KEN",
    )
    result = run_pipeline(q, world_state=_fresh_ws(), config=_cfg())
    assert result.prose, "no prose returned"
    assert result.query_type == "directive"
    return f"prose={len(result.prose)}ch"


def test_interrogate() -> str:
    """Graph-RAG Q&A — no prose, returns a proof / answer."""
    q = InterrogationQuery(
        question="Is there a chain of causation linking the heist to Mrs Coady's death?",
        require_proof=True,
    )
    result = run_pipeline(q, world_state=_fresh_ws(), config=_cfg())
    # No prose expected; answer surfaces on physics_result or result.answer.
    answer = getattr(result, "answer", None) or result.physics_result.get("answer")
    assert answer, f"no answer surfaced (physics keys: {list(result.physics_result.keys())})"
    return f"answer={len(str(answer))}ch"


def test_general() -> str:
    """Open-ended Q&A against the full graph."""
    q = GeneralQuery(
        question="Summarise the betrayal arcs between Wanda, Otto, George and Ken.",
        include_topology=True,
    )
    result = run_pipeline(q, world_state=_fresh_ws(), config=_cfg())
    answer = getattr(result, "answer", None) or result.physics_result.get("answer")
    assert answer, f"no answer (physics keys: {list(result.physics_result.keys())})"
    return f"answer={len(str(answer))}ch"


def test_manual_edit() -> str:
    """User-authored prose bypasses generation; re-extraction is skipped
    here too (the config flag) so we only verify dispatch+merge plumbing."""
    q = ManualEditQuery(
        edited_prose=(
            "Ken returned to the hideout garage carrying a single goldfish "
            "in a plastic bag. He set the bag beside the safe and watched "
            "the fish circle in silence."
        ),
        edit_description="add a contemplative beat for Ken at the garage",
    )
    result = run_pipeline(q, world_state=_fresh_ws(), config=_cfg())
    # Manual-edit returns the user's prose verbatim.
    assert result.prose, "manual_edit dropped user prose"
    assert "goldfish" in result.prose, "user prose not preserved"
    return f"prose={len(result.prose)}ch"


def test_evaluate() -> str:
    """Scorecard query — no prose, no merge."""
    q = EvaluationQuery()
    result = run_pipeline(q, world_state=_fresh_ws(), config=_cfg())
    # Evaluate should produce some kind of scorecard surface.
    score = (
        getattr(result, "evaluation", None)
        or result.physics_result.get("evaluation")
        or result.physics_result.get("scorecard")
    )
    assert score is not None, (
        f"no scorecard surfaced (physics keys: {list(result.physics_result.keys())})"
    )
    return f"score_keys={list(score.keys()) if isinstance(score, dict) else type(score).__name__}"


TESTS: dict[str, Callable[[], str]] = {
    "intervention": test_intervention,
    "directive": test_directive,
    "interrogate": test_interrogate,
    "general": test_general,
    "manual_edit": test_manual_edit,
    "evaluate": test_evaluate,
}


def main() -> int:
    print("=" * 70)
    print(f"Smoke-testing {len(TESTS)} untested query types against wanda world")
    print("=" * 70)

    results: list[tuple[str, bool, str, float]] = []
    for name, fn in TESTS.items():
        print(f"\n--- {name} ---", flush=True)
        t0 = time.time()
        try:
            detail = fn()
            elapsed = time.time() - t0
            results.append((name, True, detail, elapsed))
            print(f"  PASS ({elapsed:.1f}s) — {detail}", flush=True)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            tb = traceback.format_exc(limit=4)
            results.append((name, False, f"{type(exc).__name__}: {exc}", elapsed))
            print(f"  FAIL ({elapsed:.1f}s) — {type(exc).__name__}: {exc}", flush=True)
            print(tb, flush=True)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    for name, ok, detail, elapsed in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name:<14} {elapsed:6.1f}s  {detail}")
    print(f"\n{n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
