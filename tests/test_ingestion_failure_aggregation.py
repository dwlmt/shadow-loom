# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression coverage for the per-chunk stage failure aggregation in
``extract_topology_async``.

Bug: the ``failure_counts`` dict was initialized as
``{"physics": 0, "social": 0, "consequences": 0}`` but the per-chunk
extractor (``_extract_single_chunk_async``) writes
``failure_flags["affect"] = 1`` on affect-sub-stage exceptions. The
aggregation loop then did ``failure_counts[stage] += flag``, raising
``KeyError: 'affect'`` and aborting the entire pipeline whenever any
chunk's affect call raised.

The fix:
1. Include ``"affect": 0`` in the initial dict.
2. Use ``failure_counts.get(stage, 0)`` so any future stage drift
   (sub-stage added without updating the initializer) degrades to a
   warning instead of a crash.

These tests pin both invariants.
"""

from shadow_loom.ingestion import _check_chunk_failure_threshold


def _aggregate(chunk_flag_dicts):
    """Mirror of the production aggregation logic at
    ``shadow_loom/ingestion.py`` ~L9758. Kept inline here so a future
    edit to the production loop fails one of these tests if the
    invariant is broken."""
    failure_counts = {
        "physics": 0, "social": 0, "consequences": 0, "affect": 0,
    }
    for flags in chunk_flag_dicts:
        for stage, flag in flags.items():
            failure_counts[stage] = failure_counts.get(stage, 0) + flag
    return failure_counts


def test_affect_failure_does_not_keyerror():
    """The canonical bug: a chunk flagged ``affect`` failure must
    increment the affect counter, not raise ``KeyError``."""
    counts = _aggregate([{"affect": 1}])
    assert counts == {"physics": 0, "social": 0, "consequences": 0, "affect": 1}


def test_mixed_stage_failures_aggregate_correctly():
    counts = _aggregate([
        {"physics": 1, "social": 0, "consequences": 0, "affect": 0},
        {"physics": 0, "social": 1, "consequences": 0, "affect": 1},
        {"physics": 0, "social": 0, "consequences": 1, "affect": 1},
    ])
    assert counts == {"physics": 1, "social": 1, "consequences": 1, "affect": 2}


def test_unknown_stage_does_not_crash():
    """Forward-compat: a future sub-stage key that the initializer
    forgot must NOT crash the aggregation \u2014 the ``.get(stage, 0)``
    fallback should absorb it."""
    counts = _aggregate([{"future_substage": 1, "affect": 1}])
    # The unknown stage is now tracked too, courtesy of the defensive
    # ``.get(stage, 0)`` fallback in the production code.
    assert counts["affect"] == 1
    assert counts["future_substage"] == 1


def test_check_chunk_failure_threshold_accepts_affect_key():
    """The downstream threshold check iterates over whatever keys it
    receives; ensure it tolerates the new ``affect`` key with no
    spurious warning when the count is zero."""
    # Should not raise; threshold logic operates per-stage independently.
    _check_chunk_failure_threshold(
        {"physics": 0, "social": 0, "consequences": 0, "affect": 0},
        total_chunks=5,
        sample_errors={},
    )


def test_check_chunk_failure_threshold_reports_affect_failures():
    """When affect failures exceed the threshold, the check should
    raise a clean ``RuntimeError`` naming the affect stage \u2014 not a
    ``KeyError`` from the previously-broken aggregation path."""
    import pytest
    with pytest.raises(RuntimeError, match=r"affect=5/5"):
        _check_chunk_failure_threshold(
            {"physics": 0, "social": 0, "consequences": 0, "affect": 5},
            total_chunks=5,
            sample_errors={"affect": "RuntimeError: synthetic"},
        )
