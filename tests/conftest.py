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

import pytest


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
