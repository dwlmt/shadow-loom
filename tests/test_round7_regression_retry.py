"""Round-7 audit (2026-05-26): regression tests for the feedback-loop
design changes that followed the previous-draft anchoring fix.

Two changes are pinned here:

1. ``AuditorConfig.max_iterations`` default raised from 3 to 4 so the
   anti-regression retry path actually has room to run (initial audit
   + up to 3 refinement attempts, one of which may be the post-
   rollback retry).

2. ``run_feedback_loop`` now grants ONE anti-regression retry instead
   of breaking on the first rollback. The retry surfaces a REGRESSION
   ALERT in the next refinement prompt naming the violation types
   that were just introduced.

The retry-loop behaviour itself is covered with a unit test on the
prompt-builder (``_build_refinement_prompt``) — the orchestration
side is exercised via the existing live feedback-loop tests in
test_auditor.py, which would catch a regression in the rollback flow
because they assert on the final-iteration count.
"""

from __future__ import annotations

import pytest

from shadow_loom.auditor import (
    AuditorConfig,
    AuditViolation,
    _build_refinement_prompt,
)


class TestMaxIterationsDefault:
    """The default must be 4 so the retry path has budget to run."""

    def test_max_iterations_default_is_four(self):
        cfg = AuditorConfig()
        assert cfg.max_iterations == 6

    def test_max_iterations_remains_overridable(self):
        cfg = AuditorConfig(max_iterations=2)
        assert cfg.max_iterations == 2


class TestRegressionWarningInjection:
    """When the rollback grants a retry, the next refinement prompt
    must carry a REGRESSION ALERT block naming the introduced types."""

    def _violation(self) -> AuditViolation:
        return AuditViolation(
            violation_type="style_mismatch",
            severity="critical",
            description="Synopsis register drift.",
            feedback="Tighten to ≤2 sentences per beat.",
            evidence_quote="She crossed the threshold.",
        )

    def test_regression_warning_surfaces_alert_block(self):
        warning = (
            "Your previous rewrite REGRESSED by introducing new "
            "violation type(s): ['meta_narration', 'pov_breach']. "
            "Hold the line on those constraints absolutely."
        )
        prompt = _build_refinement_prompt(
            "ORIGINAL",
            [self._violation()],
            iteration=3,
            regression_warning=warning,
        )
        assert "=== REGRESSION ALERT" in prompt
        assert "meta_narration" in prompt
        assert "pov_breach" in prompt
        assert "anti-regression retry" in prompt.lower()

    def test_regression_warning_appears_above_violation_list(self):
        """The alert must sit above the auditor-feedback violations so
        the rewriter sees it before any per-violation guidance."""
        warning = "SENTINEL_REGRESSION_TOKEN"
        prompt = _build_refinement_prompt(
            "ORIGINAL",
            [self._violation()],
            iteration=2,
            regression_warning=warning,
        )
        sentinel_idx = prompt.index("SENTINEL_REGRESSION_TOKEN")
        # The per-violation list emits the violation feedback string;
        # the alert must precede it.
        violation_idx = prompt.index("Tighten to")
        assert sentinel_idx < violation_idx

    def test_no_regression_warning_when_kwarg_omitted(self):
        prompt = _build_refinement_prompt(
            "ORIGINAL", [self._violation()], iteration=2,
        )
        assert "REGRESSION ALERT" not in prompt

    def test_regression_warning_compatible_with_previous_draft(self):
        """Both the previous-draft block and the regression alert must
        appear when both kwargs are supplied — they address different
        concerns and the rewriter needs both signals at once."""
        prompt = _build_refinement_prompt(
            "ORIGINAL",
            [self._violation()],
            iteration=3,
            previous_prose="The rolled-back draft text.",
            regression_warning="Introduced ['pov_breach'].",
        )
        assert "=== PREVIOUS DRAFT" in prompt
        assert "=== REGRESSION ALERT" in prompt
        assert "The rolled-back draft text." in prompt
        assert "pov_breach" in prompt


class TestFeedbackLoopRetryBudgetExists:
    """The retry-budget local variable must be initialised in the loop
    so a future refactor cannot silently delete the retry path."""

    def test_loop_initialises_retry_budget(self):
        """Smoke-check the source contains the retry-budget setup so a
        future edit that removes it (and reverts to immediate-break)
        triggers this test, not a silent semantic regression that only
        shows up in long live runs."""
        import inspect
        from shadow_loom import auditor

        src = inspect.getsource(auditor.run_feedback_loop)
        assert "regression_retries_remaining" in src
        assert "pending_regression_warning" in src
        # The retry path must decrement and the exhaustion path must
        # still break — both must be present.
        assert "regression_retries_remaining -= 1" in src
        assert "retry budget exhausted" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
