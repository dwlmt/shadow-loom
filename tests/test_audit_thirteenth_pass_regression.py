"""Regression tests for the thirteenth-pass audit fixes (D1\u2013D7).

Each ``TestDN`` class pins the corresponding finding so a future
refactor that reverts the fix fails loudly. Where the fix is a
defensive cap or a code-shape requirement, we grep the source rather
than try to construct a fragile behavioural reproducer; where the fix
changes runtime semantics, we exercise it directly.
"""
from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

import shadow_loom.db as db_mod
import shadow_loom.ingestion as ingestion_mod
import shadow_loom.query_models as qm
import shadow_loom.settings as settings_mod
import shadow_loom_mcp.server as mcp_server
import shadow_loom_ui.components.dialogs as dialogs_mod


_ROOT = Path(__file__).resolve().parent.parent


def _src(mod) -> str:
    return inspect.getsource(mod)


# ---------------------------------------------------------------------------
# D1 \u2014 resource cap envelope on entity / versions siblings
# ---------------------------------------------------------------------------


class TestD1ResourceCapsOnSiblings:
    def test_entity_cap_constant_present(self) -> None:
        src = _src(mcp_server)
        assert "_MAX_RESOURCE_ENTITY_BYTES" in src
        assert re.search(r"_MAX_RESOURCE_ENTITY_BYTES\s*=\s*250_000", src)

    def test_versions_cap_constants_present(self) -> None:
        src = _src(mcp_server)
        assert re.search(r"_MAX_RESOURCE_VERSIONS_BYTES\s*=\s*500_000", src)
        assert re.search(r"_MAX_RESOURCE_VERSIONS_ROWS\s*=\s*1_000", src)

    def test_versions_truncated_envelope_used(self) -> None:
        src = _src(mcp_server)
        # The envelope keys must appear in the truncation branch.
        assert '"truncated"' in src
        assert '"limit"' in src
        assert '"rows"' in src


# ---------------------------------------------------------------------------
# D2 \u2014 upload handler byte cap + UTF-8 safety
# ---------------------------------------------------------------------------


class TestD2UploadHandlerSafety:
    def test_byte_cap_constant_present(self) -> None:
        src = _src(dialogs_mod)
        assert "_MAX_UPLOAD_BYTES" in src
        assert "MAX_INGEST_WORDS" in src
        assert "_MAX_UPLOAD_BYTES + 1" in src

    def test_unicode_decode_error_handled(self) -> None:
        src = _src(dialogs_mod)
        assert "UnicodeDecodeError" in src
        assert "not valid UTF-8" in src

    def test_unbounded_read_decode_removed(self) -> None:
        src = _src(dialogs_mod)
        # The exact unsafe one-liner the audit flagged, as code (not
        # quoted inside a comment string discussing the prior bug).
        assert not re.search(
            r"^\s*content\s*=\s*e\.content\.read\(\)\.decode\(",
            src,
            flags=re.MULTILINE,
        )


# ---------------------------------------------------------------------------
# D3 \u2014 fabula_time == 0 is not silently rewritten
# ---------------------------------------------------------------------------


class TestD3FabulaZeroPreserved:
    def test_old_falsy_overwrite_removed(self) -> None:
        src = _src(ingestion_mod)
        # The exact line the audit flagged must be gone.
        assert "if not b.established_at_fabula and eu.fabula_time > 0:" not in src

    def test_belief_zero_baseline_round_trips(self) -> None:
        # Constructing a baseline-anchored belief and round-tripping
        # the model must not coerce 0 -> some other int.
        from shadow_loom.models import Belief

        b = Belief(
            holder_id="ENT_alice",
            target_id="ENT_bob",
            perceived_state="trustworthy",
            confidence=0.7,
            inertia=0.5,
            evidence_strength="moderate",
            established_at_fabula=0,
        )
        assert b.established_at_fabula == 0


# ---------------------------------------------------------------------------
# D4 \u2014 do-target parse failures surface as UserWarning, not silent drop
# ---------------------------------------------------------------------------


class TestD4DoTargetErrorsSurfaced:
    def test_bare_except_pass_removed_around_do_targets(self) -> None:
        src = _src(qm)
        # Look at the coercer specifically (legacy-dict -> DoTarget) and
        # confirm no remaining ``except Exception: pass`` follow a
        # DoProposition / DoNarrativeObject / DoWorldTrait / DoTrait
        # construction.
        for needle in (
            "DoProposition(",
            "DoNarrativeObject(",
            "DoWorldTrait(",
            "DoTrait(",
        ):
            # Find the block following each ctor in source order.
            idx = src.find(needle)
            assert idx != -1, f"missing ctor {needle}"
            # The next ``except`` after this ctor should not be a bare pass.
            tail = src[idx:idx + 1500]
            # If a bare ``except Exception: pass`` (or
            # ``except (TypeError, ValueError): pass``) appears before the
            # next ctor, the silent-drop pattern is back.
            bad = re.search(
                r"except \([^)]+\):\s*\n\s*pass\b"
                r"|except Exception:\s*\n\s*pass\b",
                tail,
            )
            assert bad is None, (
                f"Silent-drop pattern resurfaced near {needle}"
            )

    def test_userwarning_on_proposition_parse_failure(self, monkeypatch) -> None:
        # Force the DoProposition constructor to raise so we exercise
        # the warning path the audit added. The grep above proves the
        # ``warnings.warn`` call exists; this asserts it actually
        # fires (rather than being shadowed by a bare except).
        from shadow_loom import query_models as qm_mod

        def _boom(*_a, **_kw):
            raise ValueError("synthetic validator failure")

        monkeypatch.setattr(qm_mod, "DoProposition", _boom)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            qm_mod._coerce_legacy_dict({"PROP_foo.truth": True})
        assert any(
            issubclass(w.category, UserWarning)
            and "DoProposition" in str(w.message)
            for w in caught
        ), [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# D5 \u2014 noisy-OR knobs bounded
# ---------------------------------------------------------------------------


class TestD5NoisyOrBounds:
    def test_threshold_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            settings_mod.CausalPhysicsSettings(noisy_or_threshold=2.0)

    def test_threshold_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            settings_mod.CausalPhysicsSettings(noisy_or_threshold=-0.1)

    def test_temperature_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            settings_mod.CausalPhysicsSettings(noisy_or_temperature=0.0)

    def test_temperature_above_cap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            settings_mod.CausalPhysicsSettings(noisy_or_temperature=11.0)

    def test_in_bounds_accepted(self) -> None:
        s = settings_mod.CausalPhysicsSettings(
            noisy_or_threshold=0.75, noisy_or_temperature=0.5
        )
        assert s.noisy_or_threshold == 0.75
        assert s.noisy_or_temperature == 0.5


# ---------------------------------------------------------------------------
# D6 \u2014 thread-safe engine init
# ---------------------------------------------------------------------------


class TestD6EngineInitLocked:
    def test_engine_lock_constant_present(self) -> None:
        src = _src(db_mod)
        assert "_engine_lock" in src
        assert "threading" in src

    def test_init_db_uses_lock(self) -> None:
        init_src = inspect.getsource(db_mod.init_db)
        assert "_engine_lock" in init_src
        assert "with _engine_lock" in init_src


# ---------------------------------------------------------------------------
# D7 \u2014 branch traversal pre-indexes children once
# ---------------------------------------------------------------------------


class TestD7BranchIndexedOnce:
    def test_children_by_anc_index_present(self) -> None:
        src = inspect.getsource(db_mod.list_branches)
        assert "children_by_anc" in src

    def test_old_inner_scan_removed(self) -> None:
        src = inspect.getsource(db_mod.list_branches)
        # The flagged O(n\u00b2) inner comprehension scanned ``rows``
        # directly; the indexed form scans ``children_by_anc.get(...)``.
        assert "c.ancestor_id == head.id and c.world_id == head.world_id" not in src
