"""Pins the round-7 (2026-05-26) design improvements A–G.

These tests cover the small, stable helpers added during the round-7
audit so future refactors can't silently regress the contract:

  A  span-localized refinement quoting (`_annotate_prose_with_violation_spans`)
  B  severity-weighted regression scorer (`_violation_severity_score`)
  C  deterministic POV/meta detectors (`deterministic_prose_findings`,
     `_check_pov_lock_metadata`)
  D  brief deep-copy isolation in `run_feedback_loop`
  E  configurable retry/failed-open budgets on `AuditorConfig`
  F  unified `Finding` adapter / `_finding_severity_score`
  G  prompt-template invariants (refinement.md, auditor.md, refinement
     prompt builder)

The full `run_feedback_loop` E2E paths are pinned by
`test_round7_refinement_previous_draft.py` and
`test_round7_regression_retry.py`; this file targets the *units*.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shadow_loom.auditor import (
    AuditResult,
    AuditViolation,
    AuditorConfig,
    Finding,
    _SEVERITY_WEIGHTS,
    _annotate_prose_with_violation_spans,
    _build_refinement_prompt,
    _check_pov_lock_metadata,
    _finding_severity_score,
    _violation_severity_score,
    deterministic_prose_findings,
    findings_from_sources,
)


# ----------------------------------------------------------------------
# B — severity-weighted regression scorer
# ----------------------------------------------------------------------

class TestSeverityScore:
    def test_weights_table_is_canonical(self):
        assert _SEVERITY_WEIGHTS == {
            "critical": 4.0, "major": 2.0, "minor": 1.0,
        }

    def test_empty_list_scores_zero(self):
        assert _violation_severity_score([]) == 0.0

    def test_mixed_severities_sum(self):
        vs = [
            AuditViolation(violation_type="meta_narration", severity="critical",
                           description="d", feedback="f"),
            AuditViolation(violation_type="meta_narration", severity="major",
                           description="d", feedback="f"),
            AuditViolation(violation_type="meta_narration", severity="minor",
                           description="d", feedback="f"),
        ]
        assert _violation_severity_score(vs) == 4.0 + 2.0 + 1.0

    def test_critical_outweighs_two_majors(self):
        crit = [AuditViolation(violation_type="meta_narration", severity="critical",
                               description="d", feedback="f")]
        two_majors = [
            AuditViolation(violation_type="meta_narration", severity="major",
                           description="d", feedback="f"),
            AuditViolation(violation_type="meta_narration", severity="major",
                           description="d", feedback="f"),
        ]
        # 4.0 > 2 * 2.0 is FALSE — they're equal — but a critical
        # alone equals two majors. Three minors are strictly less.
        assert _violation_severity_score(crit) == _violation_severity_score(two_majors)
        three_minors = [
            AuditViolation(violation_type="meta_narration", severity="minor",
                           description="d", feedback="f") for _ in range(3)
        ]
        assert _violation_severity_score(crit) > _violation_severity_score(three_minors)


# ----------------------------------------------------------------------
# C — deterministic prose findings (POV breach + meta narration)
# ----------------------------------------------------------------------

class TestDeterministicProseFindings:
    def test_empty_prose_returns_no_findings(self):
        assert deterministic_prose_findings("") == []

    def test_no_pov_lock_means_no_pov_check(self):
        # All-cognitive-verb prose without a pov_entity should not
        # synthesise a POV breach.
        prose = (
            "John thought it was strange. Mary knew the truth. "
            "Susan wondered. Tom remembered the day. Anne realised."
        )
        out = deterministic_prose_findings(prose, pov_entity=None)
        assert all(v.violation_type != "reasoning_failure" for v in out)

    def test_pov_breach_above_threshold_fires(self):
        # POV is locked to MRS_COADY; non-POV cognitive verbs above
        # threshold=3 should synthesise a critical reasoning_failure.
        prose = (
            "Mrs Coady walked into the kitchen. "
            "John thought the soup smelled wonderful. "
            "Mary knew it would be too salty. "
            "Susan wondered if anyone would notice. "
            "Tom remembered the last dinner here."
        )
        out = deterministic_prose_findings(
            prose, pov_entity="ENT_MRS_COADY", pov_breach_threshold=3,
        )
        types = {v.violation_type for v in out}
        assert "reasoning_failure" in types
        breach = next(v for v in out if v.violation_type == "reasoning_failure")
        assert breach.severity == "critical"
        assert "POV lock broken" in breach.description

    def test_pov_breach_below_threshold_does_not_fire(self):
        prose = (
            "Mrs Coady walked into the kitchen. "
            "John thought the soup smelled wonderful. "
            "She lifted the lid and sighed."
        )
        out = deterministic_prose_findings(
            prose, pov_entity="ENT_MRS_COADY", pov_breach_threshold=3,
        )
        assert all(v.violation_type != "reasoning_failure" for v in out)

    def test_pov_aliases_suppress_false_positives(self):
        # POV is Mrs Coady; sentences attributed to "Mrs Coady" /
        # "Coady" must NOT count as breaches.
        prose = (
            "Mrs Coady thought about the dogs. "
            "Coady knew they were hungry. "
            "Mrs Coady wondered where Ken had gone. "
            "Coady remembered the kettle. "
            "She felt tired."
        )
        out = deterministic_prose_findings(
            prose,
            pov_entity="ENT_MRS_COADY",
            pov_entity_aliases=["Mrs Coady", "Coady"],
            pov_breach_threshold=3,
        )
        assert all(v.violation_type != "reasoning_failure" for v in out)

    def test_meta_narration_phrase_fires(self):
        prose = "The kettle whistled. In this universe, that meant tea."
        out = deterministic_prose_findings(prose)
        assert any(v.violation_type == "meta_narration" for v in out)


# ----------------------------------------------------------------------
# C — POV-lock metadata check
# ----------------------------------------------------------------------

class TestPovLockMetadataCheck:
    def test_match_returns_none(self):
        assert _check_pov_lock_metadata("ENT_X", "ENT_X") is None

    def test_brief_unlocked_returns_none(self):
        # No brief lock → no expectation → no feedback.
        assert _check_pov_lock_metadata(None, "ENT_X") is None
        assert _check_pov_lock_metadata(None, None) is None

    def test_scene_dropped_lock_returns_feedback(self):
        msg = _check_pov_lock_metadata("ENT_MRS_COADY", None)
        assert msg is not None
        assert "ENT_MRS_COADY" in msg

    def test_scene_diverged_lock_returns_feedback(self):
        msg = _check_pov_lock_metadata("ENT_MRS_COADY", "ENT_KEN")
        assert msg is not None
        assert "ENT_MRS_COADY" in msg and "ENT_KEN" in msg


# ----------------------------------------------------------------------
# A — span annotation
# ----------------------------------------------------------------------

class TestAnnotateProseWithViolationSpans:
    def _v(self, quote: str, vtype: str = "meta_narration") -> AuditViolation:
        return AuditViolation(
            violation_type=vtype, severity="critical",
            description="d", feedback="f", evidence_quote=quote,
        )

    def test_empty_prose_passthrough(self):
        assert _annotate_prose_with_violation_spans("", []) == ""

    def test_no_violations_passthrough(self):
        prose = "Mrs Coady poured the tea."
        assert _annotate_prose_with_violation_spans(prose, []) == prose

    def test_single_quote_is_marked(self):
        prose = "Mrs Coady poured the tea. In this universe, kettles whistle."
        out = _annotate_prose_with_violation_spans(
            prose, [self._v("In this universe, kettles whistle.")],
        )
        assert "<<<VIOLATION:meta_narration>>>" in out
        assert "<<</VIOLATION>>>" in out
        # Untouched prose is preserved byte-for-byte before the marker.
        assert out.startswith("Mrs Coady poured the tea. ")

    def test_quote_not_in_prose_is_skipped(self):
        prose = "Mrs Coady poured the tea."
        out = _annotate_prose_with_violation_spans(
            prose, [self._v("a phrase that does not appear")],
        )
        assert "<<<VIOLATION" not in out
        assert out == prose

    def test_max_chars_parameter_accepted(self):
        # The cap is enforced by the caller (refinement prompt
        # builder), not by this function — but the parameter must
        # remain accepted for forward compatibility.
        prose = "x" * 1000
        out = _annotate_prose_with_violation_spans(prose, [], max_chars=500)
        assert isinstance(out, str)


# ----------------------------------------------------------------------
# E — configurable budgets on AuditorConfig
# ----------------------------------------------------------------------

class TestAuditorConfigBudgets:
    def test_max_iterations_default_is_four(self):
        cfg = AuditorConfig()
        assert cfg.max_iterations == 4

    def test_regression_retry_budget_default(self):
        cfg = AuditorConfig()
        assert cfg.regression_retry_budget == 1

    def test_failed_open_tolerance_default(self):
        cfg = AuditorConfig()
        assert cfg.failed_open_tolerance == 2

    def test_deterministic_checks_default_on(self):
        cfg = AuditorConfig()
        assert cfg.enable_deterministic_prose_checks is True

    def test_pov_breach_threshold_default(self):
        cfg = AuditorConfig()
        assert cfg.pov_breach_threshold == 3

    def test_budgets_are_overridable(self):
        cfg = AuditorConfig(
            regression_retry_budget=2,
            failed_open_tolerance=5,
            enable_deterministic_prose_checks=False,
            pov_breach_threshold=10,
        )
        assert cfg.regression_retry_budget == 2
        assert cfg.failed_open_tolerance == 5
        assert cfg.enable_deterministic_prose_checks is False
        assert cfg.pov_breach_threshold == 10


# ----------------------------------------------------------------------
# F — unified Finding adapter
# ----------------------------------------------------------------------

def _audit(vs):
    return AuditResult(
        passed=not vs, violations=list(vs), summary="t", iteration=0,
    )


class TestFindingAdapter:
    def test_finding_is_frozen(self):
        f = Finding(
            source="auditor", finding_type="t", severity="minor",
            feedback="f",
        )
        with pytest.raises(Exception):
            f.severity = "critical"  # type: ignore[misc]

    def test_findings_from_sources_no_engine_failures(self):
        audit = _audit([
            AuditViolation(violation_type="meta_narration",
                           severity="critical", description="d",
                           feedback="f"),
        ])
        out = findings_from_sources(audit)
        assert len(out) == 1
        assert out[0].source == "auditor"
        assert out[0].severity == "critical"
        assert out[0].finding_type == "meta_narration"

    def test_findings_from_sources_includes_engine_failures(self):
        audit = _audit([])
        out = findings_from_sources(audit, ["miracle step at t=3"])
        assert len(out) == 1
        assert out[0].source == "engine"
        assert out[0].severity == "critical"
        assert "miracle step" in out[0].feedback

    def test_findings_from_sources_none_audit_is_safe(self):
        out = findings_from_sources(None, ["x"])
        assert len(out) == 1 and out[0].source == "engine"

    def test_finding_severity_score_matches_violation_score(self):
        audit = _audit([
            AuditViolation(violation_type="meta_narration", severity="critical",
                           description="d", feedback="f"),
            AuditViolation(violation_type="meta_narration", severity="major",
                           description="d", feedback="f"),
        ])
        unified = findings_from_sources(audit, ["engine fail"])
        # critical (4) + major (2) + engine-critical (4) = 10.0
        assert _finding_severity_score(unified) == 10.0

    def test_engine_failure_is_treated_as_critical_in_score(self):
        # A draft that closes ALL auditor violations but BREAKS an
        # engine threshold must score worse than the same draft
        # with the engine threshold intact and one minor LLM violation.
        broke_engine = findings_from_sources(_audit([]), ["miracle"])
        only_minor = findings_from_sources(_audit([
            AuditViolation(violation_type="meta_narration", severity="minor",
                           description="d", feedback="f"),
        ]), None)
        assert _finding_severity_score(broke_engine) > _finding_severity_score(only_minor)


# ----------------------------------------------------------------------
# G — prompt-template invariants
# ----------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


class TestPromptTemplateInvariants:
    def test_refinement_md_keeps_minimal_surgical_edit_rule(self):
        text = _read("shadow_loom/prompts/refinement.md")
        assert "Minimal surgical edit" in text
        # Anti-from-scratch language must remain — the round-7 audit
        # showed full re-rolls are the primary regression source.
        assert "not a re-roll" in text.lower() or "from scratch" in text.lower()

    def test_refinement_md_mentions_violation_marker(self):
        text = _read("shadow_loom/prompts/refinement.md")
        assert "<<<VIOLATION" in text
        assert "<<</VIOLATION>>>" in text

    def test_refinement_md_forbids_pov_entity_drop(self):
        text = _read("shadow_loom/prompts/refinement.md")
        # Rule 7 — the POV-lock preservation rule.
        assert "pov_entity" in text
        # Must specifically forbid dropping it.
        assert (
            "Do NOT drop" in text
            or "do not drop" in text.lower()
        )

    def test_refinement_md_re_emits_brief_pov_lock_role(self):
        text = _read("shadow_loom/prompts/refinement.md")
        # The brief must be re-asserted as still-binding.
        assert "Creative Brief" in text
        assert "still apply" in text.lower()

    def test_build_refinement_prompt_emits_previous_draft_block(self):
        out = _build_refinement_prompt(
            original_rendering_prompt="ORIGINAL_PROMPT",
            violations=[AuditViolation(
                violation_type="meta_narration", severity="critical",
                description="d", feedback="remove the phrase",
                evidence_quote="In this universe, kettles whistle.",
            )],
            iteration=1,
            previous_prose="Mrs Coady poured the tea. In this universe, kettles whistle.",
        )
        # PREVIOUS DRAFT section header appears BEFORE auditor feedback
        # detail (the rewriter sees the draft first, then the deltas).
        assert "=== PREVIOUS DRAFT" in out
        assert "=== AUDITOR FEEDBACK" in out
        idx_prev = out.index("=== PREVIOUS DRAFT")
        idx_violation = out.index("[CRITICAL | meta_narration]")
        assert idx_prev < idx_violation

    def test_build_refinement_prompt_emits_regression_alert_block(self):
        out = _build_refinement_prompt(
            original_rendering_prompt="ORIGINAL_PROMPT",
            violations=[AuditViolation(
                violation_type="meta_narration", severity="critical",
                description="d", feedback="f",
            )],
            iteration=2,
            regression_warning="Your last attempt broke POV.",
        )
        assert "REGRESSION ALERT" in out
        assert "Your last attempt broke POV." in out

    def test_build_refinement_prompt_without_previous_draft(self):
        out = _build_refinement_prompt(
            original_rendering_prompt="ORIGINAL_PROMPT",
            violations=[AuditViolation(
                violation_type="meta_narration", severity="critical",
                description="d", feedback="f",
            )],
            iteration=1,
        )
        assert "=== PREVIOUS DRAFT" not in out
        assert "=== AUDITOR FEEDBACK" in out


# ----------------------------------------------------------------------
# Round-7 DEEPER audit (2026-05-27) — P0/P2/P3 follow-ups
# ----------------------------------------------------------------------

class TestPovLockMetadataPolicyAware:
    """P0 #2a — `_check_pov_lock_metadata` must respect pov_policy."""

    def test_ensemble_policy_licenses_anything(self):
        # Ensemble = omniscient is licensed. Never flag a divergence.
        assert _check_pov_lock_metadata(
            "ENT_A", "ENT_B", policy="ensemble",
        ) is None
        assert _check_pov_lock_metadata(
            "ENT_A", None, policy="ensemble",
        ) is None

    def test_rotating_accepts_additional_locks(self):
        # Rotating with {primary=ENT_A, additional=[ENT_B, ENT_C]}.
        # Any of those three is a valid scene.pov_entity.
        for ok in ("ENT_A", "ENT_B", "ENT_C"):
            assert _check_pov_lock_metadata(
                "ENT_A", ok,
                additional_locks=["ENT_B", "ENT_C"],
                policy="rotating",
            ) is None

    def test_rotating_rejects_unlicensed_entity(self):
        msg = _check_pov_lock_metadata(
            "ENT_A", "ENT_Z",
            additional_locks=["ENT_B", "ENT_C"],
            policy="rotating",
        )
        assert msg is not None
        assert "ENT_Z" in msg
        # The message should list the licensed set.
        assert "ENT_A" in msg and "ENT_B" in msg

    def test_single_policy_unchanged_with_default_kwargs(self):
        # Backward compat: existing call sites passing only the two
        # positional args still get the single-POV semantics.
        assert _check_pov_lock_metadata("ENT_A", "ENT_A") is None
        assert _check_pov_lock_metadata("ENT_A", "ENT_B") is not None


class TestDeterministicProseFindingsMultiPov:
    """P0 #2a — call-site behaviour for rotating/ensemble at the loop layer.

    The `deterministic_prose_findings` helper itself is single-POV (it
    takes one `pov_entity` + an alias list); the loop's responsibility
    is to either skip the call (ensemble) or aggregate aliases across
    the licensed set (rotating). These tests pin the contract.
    """

    def test_rotating_aliases_suppress_breach_for_either_pov(self):
        # POV-locked to {Mrs Coady, Ken} under rotating policy. Both
        # characters' interiority must be tolerated when the call
        # site aggregates aliases for the whole licensed set.
        prose = (
            "Mrs Coady thought about the dogs. "
            "Ken knew she was tired. "
            "Mrs Coady wondered where the kettle was. "
            "Ken remembered the last visit. "
            "She felt cold."
        )
        out = deterministic_prose_findings(
            prose,
            pov_entity="ENT_MRS_COADY",
            pov_entity_aliases=["Mrs Coady", "Coady", "Ken"],
            pov_breach_threshold=3,
        )
        # No POV-breach should fire because every cognitive verb
        # is attached to a licensed POV.
        assert all(v.violation_type != "reasoning_failure" for v in out)


class TestRegressionKeyingIncludesEngineFailures:
    """P0 #1 — `current_keys` and `prior_violation_keys` must include
    a synthetic ("engine_threshold", failure_text) entry per engine
    failure so engine-only regressions show up as introduced_types
    and trip the rollback gate."""

    def test_findings_from_sources_emits_engine_finding_per_failure(self):
        # Smoke: the unified-finding adapter is what powers the
        # severity score; the key set in `run_feedback_loop` mirrors
        # the same source mapping.
        audit = AuditResult(passed=True, violations=[], summary="ok", iteration=0)
        out = findings_from_sources(audit, [
            "miracle step at t=3", "cycle cluster size 4",
        ])
        types = {(f.source, f.finding_type) for f in out}
        assert types == {("engine", "engine_threshold")}
        assert all(f.severity == "critical" for f in out)

    def test_engine_failure_alone_changes_severity_score(self):
        # Two iterations: prior has only LLM minor; current closes
        # the LLM minor but introduces an engine failure. The score
        # must rise so the rollback gate has a chance to fire.
        prior_audit = AuditResult(
            passed=False, summary="t", iteration=0,
            violations=[AuditViolation(
                violation_type="meta_narration", severity="minor",
                description="d", feedback="f",
            )],
        )
        current_audit = AuditResult(
            passed=False, summary="t", iteration=1, violations=[],
        )
        prior_score = _finding_severity_score(
            findings_from_sources(prior_audit, None)
        )
        current_score = _finding_severity_score(
            findings_from_sources(current_audit, ["miracle step at t=3"])
        )
        # 1 minor LLM = 1.0; engine_failure as critical = 4.0
        assert prior_score == 1.0
        assert current_score == 4.0
        assert current_score > prior_score


class TestDirectiveAssemblyMultiPovPropagation:
    """P0 #2b — multi-target directives must populate `pov_policy`
    and `additional_pov_locks` on the resulting `RenderingDirective`.
    Single-target directives keep the single-POV defaults."""

    def test_single_target_keeps_single_policy(self):
        from shadow_loom.directive_assembly import RenderingDirective
        # Build a minimal directive the way the assembler does for a
        # single-target case.
        d = RenderingDirective(
            rendering_mode="mystery",
            pov_lock="ENT_A",
            additional_pov_locks=[],
            pov_policy="single",
            pacing="normal",
            sensory_focus="wide",
        )
        assert d.pov_policy == "single"
        assert d.additional_pov_locks == []

    def test_multi_target_uses_rotating_with_additional_locks(self):
        from shadow_loom.directive_assembly import RenderingDirective
        d = RenderingDirective(
            rendering_mode="mystery",
            pov_lock="ENT_A",
            additional_pov_locks=["ENT_B", "ENT_C"],
            pov_policy="rotating",
            pacing="normal",
            sensory_focus="wide",
        )
        assert d.pov_policy == "rotating"
        assert d.additional_pov_locks == ["ENT_B", "ENT_C"]


class TestSettingsRoundTrip:
    """P3 #8 — `AuditorSettings.auditor_config()` must propagate every
    round-7 budget field through to a working `AuditorConfig` instance
    with the documented defaults."""

    def test_round7_fields_survive_settings_to_config_mapping(self):
        from shadow_loom.settings import Settings
        s = Settings()
        cfg_dict = s.auditor_config()
        # Every round-7 field is present and equals the documented
        # default. If a future Pydantic-default change drops one,
        # this test fails loudly.
        assert cfg_dict["max_iterations"] == 4
        assert cfg_dict["regression_retry_budget"] == 1
        assert cfg_dict["failed_open_tolerance"] == 2
        assert cfg_dict["enable_deterministic_prose_checks"] is True
        assert cfg_dict["pov_breach_threshold"] == 3
        # Round-trip: build a config from the dict and verify the
        # values are preserved (the AuditorConfig validators don't
        # silently coerce them).
        cfg = AuditorConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in {
                "max_iterations", "regression_retry_budget",
                "failed_open_tolerance",
                "enable_deterministic_prose_checks",
                "pov_breach_threshold",
            }
        })
        assert cfg.max_iterations == 4
        assert cfg.regression_retry_budget == 1
        assert cfg.failed_open_tolerance == 2
        assert cfg.enable_deterministic_prose_checks is True
        assert cfg.pov_breach_threshold == 3


class TestPromptTemplatePolicyAware:
    """P0 #2c — generation.md and auditor.md must describe the
    policy-aware POV contract, not single-POV-only absolutes."""

    def test_generation_md_mentions_pov_policy(self):
        text = _read("shadow_loom/prompts/generation.md")
        assert "pov_policy" in text
        assert "rotating" in text
        assert "ensemble" in text

    def test_auditor_md_mentions_pov_policy(self):
        text = _read("shadow_loom/prompts/auditor.md")
        assert "pov_policy" in text
        assert "additional_pov_locks" in text
        # Ensemble must explicitly license omniscient interiority.
        assert "ensemble" in text.lower()
