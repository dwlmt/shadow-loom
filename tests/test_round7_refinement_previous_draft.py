"""Round-7 audit (2026-05-26): regression tests for previous-draft anchoring.

The feedback-loop audit found that the refinement prompt was instructing
the rewriter to "rewrite from scratch" **without** showing it the
previous draft prose. That allowed iteration N+1 to re-roll surface
choices the iteration-N draft had already gotten right (POV lock,
rendering mode, anti-meta framing), causing new violation types to
appear and the regression detector to roll back & exit the loop
early.

The fix:
  1. ``_build_refinement_prompt`` now accepts a ``previous_prose``
     kwarg and, when supplied, injects a ``=== PREVIOUS DRAFT ===``
     block before the auditor feedback.
  2. The rewrite-task footer switches to a *minimal surgical edit*
     directive when ``previous_prose`` is present.
  3. The feedback-loop call site passes ``current_scene.prose`` so
     every refinement after the first audit anchors to the most
     recent draft.
  4. The ``refinement.md`` system prompt rule was updated from
     "Rewrite from scratch" to a conditional minimal-edit rule.

These tests pin the new behaviour so the loop never silently drifts
back to from-scratch rewrites.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shadow_loom.auditor import AuditViolation, _build_refinement_prompt


def _make_violation() -> AuditViolation:
    return AuditViolation(
        violation_type="style_mismatch",
        severity="critical",
        description="Synopsis register drift.",
        feedback="Tighten to ≤2 sentences per beat.",
        evidence_quote="She crossed the threshold and felt the room contract.",
    )


class TestPreviousDraftInjection:
    """previous_prose must surface verbatim and reshape the rewrite task."""

    def test_previous_draft_injected_when_supplied(self):
        prev = "She crossed the threshold and felt the room contract."
        prompt = _build_refinement_prompt(
            "ORIGINAL", [_make_violation()], 2, previous_prose=prev,
        )
        assert "=== PREVIOUS DRAFT" in prompt
        assert prev in prompt

    def test_previous_draft_absent_when_not_supplied(self):
        prompt = _build_refinement_prompt(
            "ORIGINAL", [_make_violation()], 2,
        )
        assert "=== PREVIOUS DRAFT" not in prompt

    def test_rewrite_task_switches_to_minimal_edit_when_draft_present(self):
        prompt = _build_refinement_prompt(
            "ORIGINAL", [_make_violation()], 2,
            previous_prose="prior draft text",
        )
        # Minimal-edit directive must surface so the model does not
        # re-roll the surface choices that already passed audit.
        assert "MINIMAL EDIT" in prompt or "minimal edit" in prompt.lower()
        assert "from scratch" not in prompt.lower() or "Do NOT re-render" in prompt
        # And the previous-draft footer must explicitly forbid full
        # re-renders.
        assert (
            "surgical" in prompt.lower()
            or "Preserve every surface choice" in prompt
        )

    def test_rewrite_task_falls_back_to_from_scratch_without_draft(self):
        prompt = _build_refinement_prompt(
            "ORIGINAL", [_make_violation()], 2,
        )
        assert "from scratch" in prompt.lower()

    def test_previous_draft_truncated_when_oversized(self):
        # Build a 12,000-char draft and confirm the prompt caps it.
        prev = "A" * 12000
        prompt = _build_refinement_prompt(
            "ORIGINAL", [_make_violation()], 2, previous_prose=prev,
        )
        assert "truncated" in prompt
        # The full 12k payload must not all be present.
        assert prompt.count("A") < 12000

    def test_minimal_edit_directive_preserves_pov_and_mode_language(self):
        """The new directive must name the constraints most commonly lost
        in from-scratch rewrites (POV, mode, anti-meta, blocked traits),
        because the round-7 log showed iter-2 dropping POV and emitting
        meta-narration after a from-scratch rewrite."""
        prompt = _build_refinement_prompt(
            "ORIGINAL", [_make_violation()], 2,
            previous_prose="x",
        )
        # POV / mode / anti-meta / blocked-trait reminders must all
        # appear in the rewrite-task footer so the rewriter holds the
        # line on each one.
        assert "POV" in prompt
        assert "rendering mode" in prompt
        assert "anti-meta" in prompt.lower() or "meta-narration" in prompt.lower()
        assert "blocked" in prompt.lower()


class TestRefinementSystemPromptAligned:
    """The static refinement.md prompt must agree with the runtime change.

    If the system prompt still said "Rewrite from scratch" while the
    user prompt now says "minimal edit," the model would receive
    contradictory instructions and likely default to the louder
    (system-prompt) directive.
    """

    def test_refinement_md_no_longer_demands_from_scratch_rewrite(self):
        prompt_path = (
            Path(__file__).resolve().parent.parent
            / "shadow_loom" / "prompts" / "refinement.md"
        )
        text = prompt_path.read_text(encoding="utf-8")
        # The old rule was: "3. **Rewrite from scratch.** Do not try to
        # patch the previous draft ..." The new rule must NOT contain
        # that imperative form.
        assert "**Rewrite from scratch.**" not in text
        assert "Do not try to patch the previous draft" not in text

    def test_refinement_md_has_minimal_edit_rule(self):
        prompt_path = (
            Path(__file__).resolve().parent.parent
            / "shadow_loom" / "prompts" / "refinement.md"
        )
        text = prompt_path.read_text(encoding="utf-8")
        assert "Minimal surgical edit" in text or "minimal surgical edit" in text.lower()
        assert "PREVIOUS DRAFT" in text

    def test_refinement_md_pins_pov_entity_preservation(self):
        """The round-7 audit specifically saw iter-2 drop ``pov_entity``
        to ``None``. Refinement.md must now warn against this explicitly,
        not just rely on the brief upstream."""
        prompt_path = (
            Path(__file__).resolve().parent.parent
            / "shadow_loom" / "prompts" / "refinement.md"
        )
        text = prompt_path.read_text(encoding="utf-8")
        assert "pov_entity" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
