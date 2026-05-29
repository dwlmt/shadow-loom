"""Twelfth-pass audit (C1-C7) regression tests.

Each ``TestCx*`` class corresponds to one finding from the twelfth
audit pass and locks in the fix so a future regression flags the
specific surface that was tightened. Tests are deliberately a mix of
source-level greps (cheap structural guards) and behavioural checks
(verify the fix actually changes runtime output) per finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# =====================================================================
# C1 — HTML-escape tooltip dynamic fields
# =====================================================================
class TestC1TooltipFieldsEscaped:
    """All ECharts tooltip composition in ``viz_helpers.py`` must run
    user-supplied / LLM-extracted text through ``html.escape`` before
    interpolation. ECharts renders the formatter string as HTML, so an
    unescaped ``loc.description = '<img src=x onerror=alert(1)>'`` is a
    DOM-level XSS sink in the rendered chart."""

    VIZ = REPO / "shadow_loom_ui" / "viz_helpers.py"

    def test_html_module_imported(self):
        src = self.VIZ.read_text()
        assert "import html" in src

    def test_html_escape_widely_applied(self):
        src = self.VIZ.read_text()
        # The fix should have introduced many escape sites — a low
        # bar (>= 20) catches a wholesale revert without false-
        # positiving on a single legitimate refactor.
        assert src.count("html.escape(") >= 20, (
            "Expected widespread html.escape usage in tooltip "
            "composition; counted "
            f"{src.count('html.escape(')} occurrences."
        )

    def test_no_unescaped_loc_description_in_tooltips(self):
        src = self.VIZ.read_text()
        # ``{loc.description[:N]}`` directly interpolated into an
        # f-string is the canonical XSS sink from the audit; assert
        # no such bare interpolation remains.
        assert "{loc.description[:" not in src or all(
            "html.escape" in line
            for line in src.splitlines()
            if "{loc.description[:" in line
        )


# =====================================================================
# C2 — ContextVar-based ingestion-diagnostics scoping
# =====================================================================
class TestC2DiagnosticsContextScoping:
    """``capture_ingestion_warnings`` must route records by per-context
    project_id (ContextVar) instead of a shared global LIFO stack so
    concurrent ingestions in different threads / tasks don't cross-
    attribute log lines."""

    def test_contextvar_present_and_no_global_stack(self):
        src = (REPO / "shadow_loom" / "ingestion_diagnostics.py").read_text()
        assert "contextvars" in src
        assert "ContextVar" in src
        # The previous LIFO stack should be gone.
        assert "_project_stack" not in src

    def test_two_threads_do_not_cross_attribute(self):
        import logging
        import threading

        from shadow_loom.ingestion_diagnostics import (
            capture_ingestion_warnings,
            get_ingestion_warnings,
        )

        log = logging.getLogger("shadow_loom.ingestion")
        ready = threading.Event()
        proceed = threading.Event()

        def worker(pid: str, marker: str) -> None:
            with capture_ingestion_warnings(project_id=pid):
                ready.set()
                proceed.wait(timeout=2.0)
                log.warning(f"[Auto-Fix] {marker}")

        t1 = threading.Thread(target=worker, args=("proj-A", "from-A"))
        t2 = threading.Thread(target=worker, args=("proj-B", "from-B"))
        t1.start(); t2.start()
        proceed.set()
        t1.join(timeout=5.0); t2.join(timeout=5.0)

        a = [w.message for w in get_ingestion_warnings("proj-A")]
        b = [w.message for w in get_ingestion_warnings("proj-B")]
        # Each project's buffer must only contain its own marker.
        assert all("from-A" in m for m in a), a
        assert all("from-B" in m for m in b), b
        assert not any("from-B" in m for m in a)
        assert not any("from-A" in m for m in b)


# =====================================================================
# C3 — syuzhet_anchor threaded into ego-graph extraction
# =====================================================================
class TestC3TensionAnchorThreadedThroughEgoExtract:
    """``compute_tension`` must forward ``syuzhet_anchor`` to
    ``extract_ego_graph_from_memory`` so the assembled DirectiveAssembler
    sees a reader-anchored sandbox rather than a future-leaking one."""

    def test_call_site_passes_syuzhet_anchor(self):
        src = (REPO / "shadow_loom_mcp" / "server.py").read_text()
        # Find compute_tension and check the call site nearby.
        assert (
            "extract_ego_graph_from_memory(\n"
            "            ws, entity_ids, syuzhet_anchor=syuzhet_anchor,\n"
            "        )"
        ) in src


# =====================================================================
# C4 — Transitive inverse-truth closure
# =====================================================================
class TestC4InverseChainPropagation:
    """A 3-prop chain A↔B↔C with truth committed on A must propagate
    to B (flipped) and C (flipped twice = same polarity as A)."""

    def test_three_link_chain_propagates_with_alternating_polarity(self):
        from shadow_loom.ingestion import _mirror_truth_commit_to_inverse
        from shadow_loom.models import Proposition

        a = Proposition(
            proposition_id="PROP_A", kind="outcome",
            description="A holds", inverse_proposition_id="PROP_B",
            truth_at_fabula={100: True},
        )
        b = Proposition(
            proposition_id="PROP_B", kind="outcome",
            description="not A", inverse_proposition_id="PROP_C",
        )
        c = Proposition(
            proposition_id="PROP_C", kind="outcome",
            description="A again",
        )
        idx = {p.proposition_id: p for p in (a, b, c)}

        _mirror_truth_commit_to_inverse(idx, "PROP_A", 100, True)

        assert idx["PROP_B"].truth_at_fabula.get(100) is False
        assert idx["PROP_C"].truth_at_fabula.get(100) is True

    def test_cycle_in_inverse_graph_terminates(self):
        """A↔B↔A cycle must not infinite-loop; the BFS visited set
        guards the walk."""
        from shadow_loom.ingestion import _mirror_truth_commit_to_inverse
        from shadow_loom.models import Proposition

        a = Proposition(
            proposition_id="PROP_A", kind="outcome",
            description="A", inverse_proposition_id="PROP_B",
            truth_at_fabula={50: True},
        )
        b = Proposition(
            proposition_id="PROP_B", kind="outcome",
            description="B", inverse_proposition_id="PROP_A",
        )
        idx = {"PROP_A": a, "PROP_B": b}
        _mirror_truth_commit_to_inverse(idx, "PROP_A", 50, True)
        assert idx["PROP_B"].truth_at_fabula.get(50) is False


# =====================================================================
# C5 — entity_ids boundary validation
# =====================================================================
class TestC5EntityIdsRejectMalformed:
    """``compute_tension`` must reject malformed ``entity_ids`` at the
    MCP boundary with a clear error envelope, not a downstream
    AttributeError/KeyError."""

    def test_server_has_boundary_validation_branch(self):
        src = (REPO / "shadow_loom_mcp" / "server.py").read_text()
        assert "entity_ids must be a list" in src
        assert "_allowed_prefixes" in src
        assert "entity_ids contains malformed IDs" in src


# =====================================================================
# C6 — Strict-persist test shard exists
# =====================================================================
class TestC6StrictPersistShardExists:
    """A dedicated test file must exist that re-enables
    ``SHADOW_LOOM_STRICT_PERSIST=1`` (overriding the session-wide soft
    default) and verifies the strict-validation branch."""

    def test_shard_file_present_and_sets_env(self):
        shard = REPO / "tests" / "test_strict_persist_smoke.py"
        assert shard.exists()
        src = shard.read_text()
        assert "SHADOW_LOOM_STRICT_PERSIST" in src
        assert '"1"' in src or "'1'" in src


# =====================================================================
# C7 — resource_world payload cap
# =====================================================================
class TestC7ResourceWorldPayloadCapped:
    """The unauthenticated ``world://`` MCP resource must cap its
    payload size; over the cap it degrades to a summary projection
    rather than serving an unbounded blob."""

    def test_cap_constant_and_summary_helper_defined(self):
        src = (REPO / "shadow_loom_mcp" / "server.py").read_text()
        assert "_MAX_RESOURCE_WORLD_BYTES" in src
        assert "def _summary_projection" in src
        assert "world payload exceeds" in src
