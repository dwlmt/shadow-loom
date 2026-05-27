"""Test-wide fixtures.

The shipped default for ``CausalPhysicsSettings.monte_carlo_samples``
is non-zero so that production callers get a Bayesian Monte-Carlo
posterior rather than a single point estimate. The unit-test suite,
however, is full of assertions that pin *exact* deterministic
post-propagation values (e.g. ``hidden_deltas[trait] == 0.4``); routing
those calls through the perturbative MC orchestrator would inject Beta /
Normal sampling noise and break dozens of legacy assertions that have
nothing to do with sampling.

The autouse fixture below zeros ``monte_carlo_samples`` for every test
so the default ``execute()`` stays deterministic. Tests that *do* want
to exercise the Monte-Carlo path (``TestExecuteDistribution`` etc.)
explicitly set the value back to a positive number on the settings
singleton — those assignments win because pytest applies fixtures
inside-out and the test body executes after this fixture's setup phase.
"""

from __future__ import annotations

import os

import pytest


# Round-11 R11-09: several test modules set ``DATABASE_URL`` and
# ``MCP_ALLOW_OPEN_MODE`` at module scope (before imports that consume
# the values lazily). Those writes are persistent and leak into any
# subsequent test process / interactive shell that inherits the env.
# This session-scoped autouse fixture snapshots and restores the two
# variables at session start and teardown so test runs cannot
# permanently mutate the developer's environment, and so a missing
# module-scope write in one file cannot accidentally pick up a
# stale value from another file's import order.
@pytest.fixture(scope="session", autouse=True)
def _isolate_test_env_vars():
    sentinels = ("DATABASE_URL", "MCP_ALLOW_OPEN_MODE", "SHADOW_LOOM_SECRET_KEY")
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in sentinels}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _disable_monte_carlo_for_tests():
    """Force deterministic ``execute()`` semantics in the test suite."""
    try:
        from shadow_loom.settings import get_settings as _get_settings
    except Exception:
        # Settings module unavailable (e.g. during very early import-time
        # collection failures). Skip silently — the failing test will
        # surface the underlying problem on its own.
        yield
        return

    physics = _get_settings().physics
    saved = physics.monte_carlo_samples
    physics.monte_carlo_samples = 0
    try:
        yield
    finally:
        physics.monte_carlo_samples = saved


@pytest.fixture(autouse=True)
def _stub_answer_question(monkeypatch):
    """Stub the Q&A LLM call so non-prose pipeline tests don't hit the network.

    ``shadow_loom.pipeline._run_answer_step`` calls
    :func:`shadow_loom.answer.answer_question`, which in turn invokes
    a pydantic-ai agent against the configured LLM endpoint. In a
    test environment (no API keys / unreachable endpoint) the
    underlying ``run_sync`` call hangs rather than returning a
    network error, locking the suite at the first non-prose
    parametrized case (``InterrogationQuery`` / ``GeneralQuery``).

    We replace ``answer_question`` with a deterministic stub that
    returns a neutral :class:`AnswerCard`. Tests that genuinely
    exercise the live Q&A agent live in ``test_live_e2e.py`` (already
    excluded from the default run) and can override this fixture.
    """
    try:
        import shadow_loom.answer as _answer_module
        from shadow_loom.answer import AnswerCard
    except Exception:
        yield
        return

    def _stub(question, physics_state, **kwargs):  # noqa: ARG001
        return AnswerCard(
            answer="(stubbed answer for test)",
            evidence_node_ids=[],
            confidence=0.5,
            caveats=["test stub — LLM not invoked"],
        )

    monkeypatch.setattr(_answer_module, "answer_question", _stub)
    # Also patch the symbol imported into pipeline.py if it has been
    # bound at module load time (currently it's a deferred import
    # inside ``_run_answer_step``, but guard against future hoisting).
    try:
        import shadow_loom.pipeline as _pipeline_module
        if hasattr(_pipeline_module, "answer_question"):
            monkeypatch.setattr(_pipeline_module, "answer_question", _stub)
    except Exception:
        pass
    yield


def make_empty_world_state():
    """Build a minimal valid ``WorldStateV1`` for unit tests.

    Centralised here so per-file ``_empty_world_state`` helpers can
    import a single canonical version instead of duplicating the
    required-field list.
    """
    from shadow_loom.models import WorldStateV1

    return WorldStateV1(
        locations={},
        objects={},
        entities={},
        events=[],
        causal_topology=[],
    )
