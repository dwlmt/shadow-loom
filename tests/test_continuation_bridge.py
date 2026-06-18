"""Round-14 audit (ING-4): regression tests for the continuation quality
bridge.

These tests stub ``validate_and_correct_world_state`` /
``validate_and_correct_world_state_async`` so the bridge logic itself —
populating ``PipelineResult.continuation_quality_report`` and toggling
``PipelineResult.continuation_quarantined`` — is exercised without
spinning up a live LLM.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from shadow_loom import pipeline as pipeline_mod
from shadow_loom.ingestion import ExtractionConfig
from shadow_loom.pipeline import (
    PipelineConfig,
    PipelineResult,
    _run_continuation_quality_bridge_sync,
    _run_continuation_quality_bridge_async,
)


class _FakeReport(SimpleNamespace):
    """Minimal stand-in for ``ValidationReport``.

    Only ``is_valid`` is consulted by the bridge under test.
    """

    def __init__(self, is_valid: bool, **kw: Any) -> None:
        super().__init__(is_valid=is_valid, **kw)


class _FakeSnap:
    """Duck-typed ``WorldSnapshot`` (``version`` + ``world_state``)."""

    def __init__(self, version: int, world_state: Any) -> None:
        self.version = version
        self.world_state = world_state

    def model_copy(self, *, update: dict[str, Any]) -> "_FakeSnap":
        snap = _FakeSnap(self.version, self.world_state)
        for k, v in (update or {}).items():
            setattr(snap, k, v)
        return snap


class _FakeVWM:
    """Duck-typed ``VersionedWorldModel`` for bridge tests.

    Models the fields the bridge touches: ``current``, ``version``,
    ``snapshots`` (with a head snapshot mirroring ``current``), and
    ``model_copy``.
    """

    def __init__(self, current: Any, version: int = 1) -> None:
        self.current = current
        self.version = version
        self.snapshots = [_FakeSnap(version, current)]

    def model_copy(self, *, update: dict[str, Any]) -> "_FakeVWM":
        copy = _FakeVWM(self.current, self.version)
        for k, v in (update or {}).items():
            setattr(copy, k, v)
        return copy


# --- helpers -----------------------------------------------------------------


def _make_cfg() -> PipelineConfig:
    """Build a PipelineConfig with ``extraction_config`` set.

    The bridge short-circuits when ``extraction_config is None``, so we
    only need a non-None config; the stub patch never reads its fields.
    """
    return PipelineConfig(extraction_config=ExtractionConfig())


# --- sync path ---------------------------------------------------------------


def test_sync_bridge_records_valid_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean report leaves quarantine flag False and stores the report."""

    sentinel_ws = SimpleNamespace(tag="clean")
    report = _FakeReport(is_valid=True)

    def _stub(ws, _cfg, *, log_prefix):  # type: ignore[no-untyped-def]
        return sentinel_ws, report

    monkeypatch.setattr(pipeline_mod, "validate_and_correct_world_state", _stub)

    vwm = _FakeVWM(SimpleNamespace(tag="pre"))
    result = PipelineResult()

    out = _run_continuation_quality_bridge_sync(
        vwm, _make_cfg(), result, log_prefix="[test]",
    )

    assert result.continuation_quality_report is report
    assert result.continuation_quarantined is False
    assert out.current is sentinel_ws
    # Head snapshot must mirror the corrected current, not the pre-merge world.
    head = next(s for s in out.snapshots if s.version == out.version)
    assert head.world_state.tag == "clean"


def test_sync_bridge_quarantines_on_invalid_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid report flips the quarantine flag but keeps the world."""

    sentinel_ws = SimpleNamespace(tag="dirty")
    report = _FakeReport(is_valid=False)

    def _stub(ws, _cfg, *, log_prefix):  # type: ignore[no-untyped-def]
        return sentinel_ws, report

    monkeypatch.setattr(pipeline_mod, "validate_and_correct_world_state", _stub)

    vwm = _FakeVWM(SimpleNamespace(tag="pre"))
    result = PipelineResult()

    out = _run_continuation_quality_bridge_sync(
        vwm, _make_cfg(), result, log_prefix="[test]",
    )

    assert result.continuation_quality_report is report
    assert result.continuation_quarantined is True
    assert out.current is sentinel_ws


def test_sync_bridge_short_circuits_when_extraction_config_none() -> None:
    """No extraction_config → bridge returns the input vwm untouched."""

    vwm = _FakeVWM(SimpleNamespace(tag="pre"))
    result = PipelineResult()
    cfg = PipelineConfig()  # extraction_config defaults to None

    out = _run_continuation_quality_bridge_sync(
        vwm, cfg, result, log_prefix="[test]",
    )
    assert out is vwm
    assert result.continuation_quality_report is None
    assert result.continuation_quarantined is False


def test_sync_bridge_swallows_validator_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising validator must not corrupt the pipeline result."""

    def _boom(ws, _cfg, *, log_prefix):  # type: ignore[no-untyped-def]
        raise RuntimeError("validator down")

    monkeypatch.setattr(pipeline_mod, "validate_and_correct_world_state", _boom)

    vwm = _FakeVWM(SimpleNamespace(tag="pre"))
    result = PipelineResult()

    out = _run_continuation_quality_bridge_sync(
        vwm, _make_cfg(), result, log_prefix="[test]",
    )

    assert out is vwm
    assert result.continuation_quality_report is None
    assert result.continuation_quarantined is False


# --- async path --------------------------------------------------------------


def test_async_bridge_records_valid_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_ws = SimpleNamespace(tag="clean-async")
    report = _FakeReport(is_valid=True)

    async def _stub(ws, _cfg, *, log_prefix):  # type: ignore[no-untyped-def]
        return sentinel_ws, report

    monkeypatch.setattr(
        pipeline_mod, "validate_and_correct_world_state_async", _stub,
    )

    vwm = _FakeVWM(SimpleNamespace(tag="pre"))
    result = PipelineResult()

    out = asyncio.run(
        _run_continuation_quality_bridge_async(
            vwm, _make_cfg(), result, log_prefix="[test]",
        )
    )

    assert result.continuation_quality_report is report
    assert result.continuation_quarantined is False
    assert out.current is sentinel_ws
    # Head snapshot must mirror the corrected current, not the pre-merge world.
    head = next(s for s in out.snapshots if s.version == out.version)
    assert head.world_state.tag == "clean-async"


def test_async_bridge_quarantines_on_invalid_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_ws = SimpleNamespace(tag="dirty-async")
    report = _FakeReport(is_valid=False)

    async def _stub(ws, _cfg, *, log_prefix):  # type: ignore[no-untyped-def]
        return sentinel_ws, report

    monkeypatch.setattr(
        pipeline_mod, "validate_and_correct_world_state_async", _stub,
    )

    vwm = _FakeVWM(SimpleNamespace(tag="pre"))
    result = PipelineResult()

    out = asyncio.run(
        _run_continuation_quality_bridge_async(
            vwm, _make_cfg(), result, log_prefix="[test]",
        )
    )

    assert result.continuation_quality_report is report
    assert result.continuation_quarantined is True
    assert out.current is sentinel_ws
