# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Round-8 design audit (UI + MCP surfaces).

Covers:

* MCP-P1-02: unified ``findings[]`` in the MCP response envelope.
* MCP-P1-01: scene POV/policy metadata in the MCP response envelope.
* MCP-P2-03: per-iteration ``audit_trace[]`` in the MCP response envelope.
* MCP-P1-03: per-call ``auditor_overrides`` allowlist for the run tools.
* UI-P2-01: severity score / failed_open / engine_failure_count on the
  convergence trajectory rows.
* UI-P2-02: query-length preflight surfaces on UI settings.
* MCP-P2-02: ``set_active_version`` docs/runtime alignment (write+editor).
"""

from __future__ import annotations

import pytest

from shadow_loom.auditor import (
    AuditCycleSnapshot,
    AuditorConfig,
    AuditResult,
    AuditViolation,
    FeedbackLoopResult,
)
from shadow_loom.generation import GeneratedScene
from shadow_loom.pipeline import PipelineResult


# ----------------------------------------------------------------------
# Helpers — build minimal PipelineResult fixtures without invoking the
# full pipeline (the envelope helpers only read typed attributes).
# ----------------------------------------------------------------------

def _make_scene(pov_entity="ENT_A", mode="dramatic_irony") -> GeneratedScene:
    return GeneratedScene(
        prose="x",
        pov_entity=pov_entity,
        rendering_mode=mode,
    )


def _make_audit(violations=None, passed=True, failed_open=False):
    return AuditResult(
        passed=passed,
        violations=list(violations or []),
        audit_summary="ok" if passed else "still flagged",
        failed_open=failed_open,
    )


def _make_cycle(it, audit) -> AuditCycleSnapshot:
    return AuditCycleSnapshot(
        iteration=it,
        prose="x",
        audit_result=audit,
        graph_version=0,
    )


def _make_feedback(cycles, engine_failures=None) -> FeedbackLoopResult:
    return FeedbackLoopResult(
        final_scene=_make_scene(),
        converged=True,
        iterations=len(cycles),
        history=list(cycles),
        final_graph_version=0,
        engine_thresholds_passed=not bool(engine_failures),
        engine_threshold_failures=list(engine_failures or []),
    )


def _make_result(
    *,
    scene=None,
    feedback=None,
    history_steps=None,
) -> PipelineResult:
    r = PipelineResult(
        scene=scene if scene is not None else _make_scene(),
        feedback_result=feedback,
        query_type="directive",
    )
    if history_steps:
        for step in history_steps:
            r.history.steps.append(step)
    return r


# ----------------------------------------------------------------------
# MCP-P1-02: unified findings[] envelope
# ----------------------------------------------------------------------

class TestFindingsEnvelope:
    def test_no_feedback_returns_none(self):
        from shadow_loom_mcp.helpers import _findings_envelope
        out = _findings_envelope(_make_result(feedback=None))
        assert out is None

    def test_emits_auditor_and_engine_findings(self):
        from shadow_loom_mcp.helpers import _findings_envelope
        v = AuditViolation(
            violation_type="meta_narration", severity="major",
            description="d", feedback="f",
        )
        fb = _make_feedback(
            [_make_cycle(0, _make_audit([v], passed=False))],
            engine_failures=["miracle step at t=3"],
        )
        out = _findings_envelope(_make_result(feedback=fb))
        sources = {(f["source"], f["finding_type"], f["severity"]) for f in out}
        assert ("auditor", "meta_narration", "major") in sources
        assert ("engine", "engine_threshold", "critical") in sources


# ----------------------------------------------------------------------
# MCP-P2-03: per-iteration audit_trace[] envelope
# ----------------------------------------------------------------------

class TestAuditTraceEnvelope:
    def test_engine_failures_attach_to_last_iteration_only(self):
        from shadow_loom_mcp.helpers import _audit_trace_envelope
        v = AuditViolation(
            violation_type="meta_narration", severity="minor",
            description="d", feedback="f",
        )
        cycles = [
            _make_cycle(0, _make_audit([v, v], passed=False)),
            _make_cycle(1, _make_audit([], passed=True)),
        ]
        fb = _make_feedback(cycles, engine_failures=["cycle cluster size 4"])
        trace = _audit_trace_envelope(_make_result(feedback=fb))
        assert len(trace) == 2
        assert trace[0]["engine_failure_count"] == 0
        assert trace[0]["violation_count"] == 2
        assert trace[1]["engine_failure_count"] == 1
        # final iteration severity must be >= prior even though violation
        # count dropped — engine failure (critical=4.0) dominates.
        assert trace[1]["severity_score"] >= trace[0]["severity_score"]
        assert trace[1]["passed"] is True
        assert trace[0]["failed_open"] is False


# ----------------------------------------------------------------------
# MCP-P1-01: scene metadata envelope
# ----------------------------------------------------------------------

class TestSceneMetadataEnvelope:
    def test_single_pov_defaults_when_no_brief_in_history(self):
        from shadow_loom_mcp.helpers import _scene_metadata_envelope
        meta = _scene_metadata_envelope(_make_result())
        assert meta["pov_entity"] == "ENT_A"
        assert meta["additional_pov_locks"] == []
        assert meta["pov_policy"] == "single"

    def test_extracts_rotating_policy_from_generation_history(self):
        from shadow_loom_mcp.helpers import _scene_metadata_envelope
        history_step = {
            "step": "generation",
            "scene": {"prose": "x", "pov_entity": "ENT_A"},
            "brief": {
                "rendering": {
                    "rendering_mode": "dramatic_irony",
                    "pov_lock": "ENT_A",
                    "additional_pov_locks": ["ENT_B", "ENT_C"],
                    "pov_policy": "rotating",
                },
            },
        }
        r = _make_result(history_steps=[history_step])
        meta = _scene_metadata_envelope(r)
        assert meta["pov_entity"] == "ENT_A"
        assert meta["additional_pov_locks"] == ["ENT_B", "ENT_C"]
        assert meta["pov_policy"] == "rotating"

    def test_no_scene_returns_none(self):
        from shadow_loom_mcp.helpers import _scene_metadata_envelope
        r = PipelineResult(query_type="ask")
        assert _scene_metadata_envelope(r) is None


# ----------------------------------------------------------------------
# MCP-P1-03: per-call auditor_overrides allowlist
# ----------------------------------------------------------------------

class TestAuditorOverridesAllowlist:
    def test_none_or_empty_returns_none(self):
        from shadow_loom_mcp.helpers import _build_auditor_config_override
        # Round-9 C5: now returns (config_or_None, rejected_list).
        cfg, rejected = _build_auditor_config_override(None)
        assert cfg is None and rejected == []
        cfg, rejected = _build_auditor_config_override({})
        assert cfg is None and rejected == []

    def test_disallowed_keys_dropped(self):
        from shadow_loom_mcp.helpers import _build_auditor_config_override
        # ``auditor_model`` is intentionally NOT in the allowlist so a
        # remote client can't downgrade engine rigour by swapping the
        # model out under the operator's nose.
        cfg, rejected = _build_auditor_config_override(
            {"auditor_model": "bad/model"},
        )
        assert cfg is None
        assert {r["key"] for r in rejected} == {"auditor_model"}
        assert rejected[0]["reason"] == "disallowed_key"

    def test_allowed_keys_round_trip_into_config(self):
        from shadow_loom_mcp.helpers import _build_auditor_config_override
        cfg, rejected = _build_auditor_config_override({
            "max_iterations": 7,
            "regression_retry_budget": 2,
            "failed_open_tolerance": 3,
            "enable_deterministic_prose_checks": False,
            "pov_breach_threshold": 5,
            "auditor_model": "rejected",  # dropped
        })
        assert isinstance(cfg, AuditorConfig)
        assert cfg.max_iterations == 7
        assert cfg.regression_retry_budget == 2
        assert cfg.failed_open_tolerance == 3
        assert cfg.enable_deterministic_prose_checks is False
        assert cfg.pov_breach_threshold == 5
        # Round-9 C5: disallowed key is now surfaced as a structured
        # rejection record (was previously only logged).
        assert {r["key"] for r in rejected} == {"auditor_model"}


# ----------------------------------------------------------------------
# UI-P2-01: convergence trajectory carries severity + failed_open + engine
# ----------------------------------------------------------------------

class TestConvergenceTrajectoryNewFields:
    def test_rows_include_round8_fields(self):
        from shadow_loom_ui.reasoning_helpers import convergence_trajectory_data
        v = AuditViolation(
            violation_type="meta_narration", severity="minor",
            description="d", feedback="f",
        )
        fb = _make_feedback(
            [
                _make_cycle(0, _make_audit([v, v], passed=False, failed_open=False)),
                _make_cycle(1, _make_audit([], passed=True, failed_open=False)),
            ],
            engine_failures=["miracle step at t=3"],
        )
        rows = convergence_trajectory_data(fb)
        assert len(rows) == 2
        assert "severity_score" in rows[0]
        assert "failed_open" in rows[0]
        assert "engine_failure_count" in rows[0]
        assert rows[0]["engine_failure_count"] == 0
        assert rows[1]["engine_failure_count"] == 1
        # Final cycle's severity score must reflect the engine failure
        # even though its LLM violation count is zero.
        assert rows[1]["severity_score"] > 0.0


# ----------------------------------------------------------------------
# UI-P2-02: max_query_chars setting exists with safe default
# ----------------------------------------------------------------------

class TestUIMaxQueryChars:
    def test_default_cap_is_sane(self):
        from shadow_loom.settings import Settings
        s = Settings()
        assert s.ui.max_query_chars >= 500
        assert s.ui.max_query_chars <= 10_000

    def test_zero_disables_the_guard(self):
        # The chat preflight treats 0 as "disabled" — make sure the
        # field validator accepts 0.
        from shadow_loom.settings import UISettings
        s = UISettings(max_query_chars=0)
        assert s.max_query_chars == 0
