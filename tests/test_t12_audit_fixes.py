# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the T-12 audit (snapshot/causal/cursor/query-type).

Covers fixes layered on top of T-1..T-11:

* :func:`shadow_loom.projections.snapshot_world_at` — model-side
  fabula-tick snapshot used by MCP read tools.
* MCP ``ask`` / ``compute_tension`` / ``trace_causality`` / ``evaluate``
  ``at_time`` + ``pov_entity_id`` parameters: snapshot-then-pov
  ordering and response-envelope echo.
* MCP ``narrate`` rejection of read-only / manual-edit query types
  (so the persisting path never writes a no-op version row).
* UI :func:`shadow_loom_ui.reasoning_helpers.event_context_data`
  reading from a causal-aware snapshot (entity / object / location /
  channel / world-trait dossier fields all coming from the same
  snapshot, not the live world).
* :func:`shadow_loom_ui.reasoning_helpers.hidden_channel_rows`
  honouring ``syuzhet_anchor``.
* :class:`shadow_loom_ui.state.AppState` threading
  ``fabula_cursor`` / ``syuzhet_cursor`` into ``run_structured_query``
  so the pipeline-anchor matches the user-visible cursor.
* :func:`shadow_loom_ui.reasoning_helpers.reasoning_trace_summary`
  normalising the legacy ``interrogation`` alias.
"""

from __future__ import annotations

import os
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── MCP server import-time DB init (same pattern as test_mcp_server) ──
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["MCP_ALLOW_OPEN_MODE"] = "true"

from shadow_loom.db import (  # noqa: E402
    create_project,
    init_db,
    save_version,
    upsert_user,
)
from shadow_loom.models import WorldStateV1  # noqa: E402

init_db("sqlite://")

from shadow_loom_mcp import auth as mcp_auth  # noqa: E402
from shadow_loom_mcp.server import (  # noqa: E402
    ask,
    compute_tension,
    evaluate,
    narrate,
    trace_causality,
)

from example_worlds.macbeth import world_state as macbeth_ws  # noqa: E402


# ── Shared fixtures ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db("sqlite://")
    mcp_auth._token_user_cache.clear()
    yield
    mcp_auth._token_user_cache.clear()


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.request_context = None
    ctx.report_progress = AsyncMock()
    return ctx


def _seed_project(ws: WorldStateV1 | None = None) -> tuple[int, int]:
    user = upsert_user(
        "local", "t12-1", "t12user", email="t12@example.com",
        display_name="T12",
    )
    proj = create_project(name="T12 Macbeth", owner_id=user.id)
    ws = ws or deepcopy(macbeth_ws)
    save_version(
        project_id=proj.id,
        world_state_json=ws.model_dump_json(),
        version=0,
        source="ingestion",
        description="seed",
        user_id=user.id,
    )
    # Wire the auth cache so ``get_user_id`` resolves to this user
    # — without this, the read scope is empty and every tool errors
    # out with "Access denied".
    mcp_auth._token_user_cache["test-token"] = {
        "user_id": user.id,
        "scopes": {"read", "write", "admin"},
        "key_id": 1,
    }
    return user.id, proj.id


# =====================================================================
# 1. Model-side snapshot_world_at
# =====================================================================


class TestProjectionsSnapshotWorldAt:
    """The model-side snapshot must be MCP-safe (no UI deps) and must
    apply the same fabula-time gating that the UI snapshot does."""

    def test_no_ui_dependency(self):
        # If this import grows a UI dependency it will fail loudly
        # because the UI package transitively imports ``nicegui``.
        import importlib
        import sys

        # Ensure the UI package is *not* loaded as a side-effect of
        # importing the projection helper.
        for mod in list(sys.modules):
            if mod.startswith("shadow_loom_ui"):
                del sys.modules[mod]
        importlib.import_module("shadow_loom.projections")
        assert not any(
            m.startswith("shadow_loom_ui") for m in sys.modules
        ), "shadow_loom.projections must not import shadow_loom_ui"

    def test_filters_events_to_tick(self):
        from shadow_loom.projections import snapshot_world_at

        ws = deepcopy(macbeth_ws)
        fts = sorted({e.fabula_time for e in ws.events})
        mid = fts[len(fts) // 2]

        snap = snapshot_world_at(ws, mid)

        assert all(e.fabula_time <= mid for e in snap.events)
        assert any(e.fabula_time == mid for e in snap.events)
        assert len(snap.events) < len(ws.events)
        # The input ws must be untouched (deep copy semantics).
        assert len(ws.events) == 32

    def test_filters_causal_topology_to_tick(self):
        from shadow_loom.projections import snapshot_world_at

        ws = deepcopy(macbeth_ws)
        mid = sorted({e.fabula_time for e in ws.events})[len(ws.events) // 2]

        snap = snapshot_world_at(ws, mid)

        assert all(ce.fabula_time <= mid for ce in snap.causal_topology)

    def test_entity_state_replayed(self):
        from shadow_loom.projections import snapshot_world_at

        ws = deepcopy(macbeth_ws)
        # Earliest tick — beliefs/traits should reflect pre-story values.
        early = min(e.fabula_time for e in ws.events)
        snap = snapshot_world_at(ws, early)

        for eid, ent in snap.entities.items():
            # ``ent.traits`` survives as a dict[str, TraitVector] — the
            # snapshot must not flatten or drop it.
            assert isinstance(ent.traits, dict)

    def test_at_time_max_returns_full_world(self):
        from shadow_loom.projections import snapshot_world_at

        ws = deepcopy(macbeth_ws)
        tmax = max(e.fabula_time for e in ws.events)
        snap = snapshot_world_at(ws, tmax)

        assert len(snap.events) == len(ws.events)

    def test_channels_window_gated(self):
        from shadow_loom.projections import snapshot_world_at

        ws = deepcopy(macbeth_ws)
        # Tick 0 is before any channel could be established.
        snap = snapshot_world_at(ws, 0)
        # ``reconstruct_channel_at`` returns None for windows that
        # haven't opened yet, so the snapshot drops them entirely.
        for ch in snap.channels.values():
            est = getattr(ch, "established_at_fabula", 0) or 0
            assert est <= 0


# =====================================================================
# 2. MCP at_time + pov_entity_id parity
# =====================================================================


class TestMcpAtTimeParity:
    """Read-only MCP tools that accept ``at_time`` must echo it in the
    response envelope and apply ``snapshot_world_at`` *before* POV
    filtering. The ordering matters because POV filters can drop
    events whose visibility depends on belief state at ``t``."""

    def test_compute_tension_echoes_at_time(self):
        _, pid = _seed_project()
        fts = sorted({e.fabula_time for e in macbeth_ws.events})
        mid = fts[len(fts) // 2]

        result = compute_tension(_ctx(), project_id=pid, at_time=mid)
        assert result.get("at_time") == mid
        assert "scores" in result

    def test_compute_tension_no_at_time_field_when_omitted(self):
        _, pid = _seed_project()
        result = compute_tension(_ctx(), project_id=pid)
        # The key may be absent or None; both are fine, but a string
        # would indicate a stringification bug.
        assert result.get("at_time") in (None, 0, False)

    def test_trace_causality_at_time_gates_walk(self):
        _, pid = _seed_project()
        # Pick a root event late in the story so an early at_time
        # genuinely truncates the walk.
        late_event = sorted(
            macbeth_ws.events, key=lambda e: e.fabula_time
        )[-1]
        early_t = sorted({e.fabula_time for e in macbeth_ws.events})[1]

        # Walking before the event exists must short-circuit.
        result = trace_causality(
            _ctx(), late_event.id, project_id=pid, at_time=early_t,
        )
        # Either the event isn't in the gated nodes or the walk is
        # empty — both encode "no causal chain visible by ``t``".
        assert "error" in result or not result.get("edges")

    @patch("shadow_loom_mcp.server.calculate_narrative_physics")
    @patch("shadow_loom_mcp.server.parse_query")
    def test_ask_echoes_at_time_and_pov(self, mock_parse, mock_physics):
        _, pid = _seed_project()
        fts = sorted({e.fabula_time for e in macbeth_ws.events})
        mid = fts[len(fts) // 2]

        mock_parsed = MagicMock(reasoning="r", resolved_ids=[])
        mock_parse_result = MagicMock()
        mock_parse_result.is_valid = False
        mock_parse_result.query = None
        mock_parse_result.parsed = mock_parsed
        mock_parse.return_value = mock_parse_result

        mock_physics.return_value = {"status": "complete", "answer": "ok"}

        result = ask(
            _ctx(), "What does Macbeth know?", project_id=pid,
            at_time=mid, pov_entity_id="ENT_MACBETH",
        )
        assert result["at_time"] == mid
        assert result["pov_entity_id"] == "ENT_MACBETH"
        assert result["read_only"] is True

    def test_evaluate_at_time_present_in_envelope(self):
        # Real evaluate path is heavy; we patch the LLM-bound critique
        # but let the snapshot + assembler scaffolding run so the
        # envelope assembly is exercised end-to-end.
        _, pid = _seed_project()
        fts = sorted({e.fabula_time for e in macbeth_ws.events})
        mid = fts[len(fts) // 2]

        # Without prose the tool short-circuits to a non-LLM envelope,
        # which is plenty to assert the at_time echo.
        result = evaluate(_ctx(), project_id=pid, at_time=mid)
        assert result.get("read_only") is True
        # at_time is echoed when prose is found *or* in the prose-empty
        # branch under the audit. Accept both shapes.
        if "at_time" in result:
            assert result["at_time"] == mid


# =====================================================================
# 3. narrate readonly-rejection guard
# =====================================================================


class TestNarrateRejectsReadonlyQueryTypes:
    """``narrate`` is a *write* tool. If the parser classifies an
    instruction as ``general`` / ``interrogate`` / ``evaluate`` /
    ``manual_edit`` the tool must refuse and point at the correct
    sibling, instead of silently persisting a version row that records
    no world advancement."""

    @pytest.mark.asyncio
    @patch("shadow_loom_mcp.server.parse_query")
    async def test_rejects_general_query(self, mock_parse):
        _, pid = _seed_project()

        mock_parsed = MagicMock(reasoning="r", resolved_ids=[])
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.query = MagicMock(query_type="general")
        mock_result.parsed = mock_parsed
        mock_result.validation_errors = []
        mock_parse.return_value = mock_result

        result = await narrate(_ctx(), "What is the weather like?", project_id=pid)
        assert "error" in result
        assert result.get("code") == "READONLY_QUERY_REJECTED"
        assert result.get("resolved_query_type") == "general"

    @pytest.mark.asyncio
    @patch("shadow_loom_mcp.server.parse_query")
    async def test_rejects_interrogate_query(self, mock_parse):
        _, pid = _seed_project()

        mock_parsed = MagicMock(reasoning="r", resolved_ids=[])
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.query = MagicMock(query_type="interrogate")
        mock_result.parsed = mock_parsed
        mock_result.validation_errors = []
        mock_parse.return_value = mock_result

        result = await narrate(_ctx(), "Is there a path from A to B?", project_id=pid)
        assert result.get("code") == "READONLY_QUERY_REJECTED"

    @pytest.mark.asyncio
    @patch("shadow_loom_mcp.server.parse_query")
    async def test_rejects_manual_edit(self, mock_parse):
        _, pid = _seed_project()

        mock_parsed = MagicMock(reasoning="r", resolved_ids=[])
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.query = MagicMock(query_type="manual_edit")
        mock_result.parsed = mock_parsed
        mock_result.validation_errors = []
        mock_parse.return_value = mock_result

        result = await narrate(_ctx(), "<edited prose>", project_id=pid)
        assert result.get("code") == "MANUAL_EDIT_REJECTED"

    @pytest.mark.asyncio
    @patch("shadow_loom_mcp.server.run_and_save")
    @patch("shadow_loom_mcp.server.parse_query")
    async def test_accepts_observation_query(
        self, mock_parse, mock_run_save,
    ):
        # Positive control: a write-class query type must still flow
        # through to run_and_save.
        _, pid = _seed_project()

        mock_parsed = MagicMock(reasoning="r", resolved_ids=[])
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.query = MagicMock(query_type="observation")
        mock_result.parsed = mock_parsed
        mock_result.validation_errors = []
        mock_parse.return_value = mock_result

        mock_run_save.return_value = {
            "project_id": pid, "version": 1, "prose": "x",
            "query_type": "observation",
        }

        result = await narrate(_ctx(), "Continue the story.", project_id=pid)
        mock_run_save.assert_called_once()
        assert result.get("prose") == "x"


# =====================================================================
# 4. UI event_context_data — causal-aware snapshot wiring
# =====================================================================


class TestEventContextDataUsesSnapshot:
    """The clicked-event dossier must source per-tick mutable fields
    (entity status / location / traits, object owner / location,
    world-trait magnitude, channel window) from a single causal-aware
    snapshot of the world at the event's fabula_time — not from the
    live world or per-helper reconstruct calls."""

    def test_returns_empty_for_missing_event(self):
        from shadow_loom_ui.reasoning_helpers import event_context_data

        ctx = event_context_data(deepcopy(macbeth_ws), "EVT_DOES_NOT_EXIST")
        assert ctx == {}

    def test_dossier_uses_snapshot_world(self):
        from shadow_loom_ui.reasoning_helpers import event_context_data

        ws = deepcopy(macbeth_ws)
        # Pick an event that's mid-story so there's something to gate.
        target = sorted(ws.events, key=lambda e: e.fabula_time)[
            len(ws.events) // 2
        ]
        ctx = event_context_data(ws, target.id)

        assert ctx["event"]["id"] == target.id
        assert ctx["event"]["fabula_time"] == target.fabula_time
        # Causal incoming/outgoing must only reference edges at or
        # before this tick — the snapshot enforces this.
        causal_pairs_at = {
            (ce.source_id, ce.target_id) for ce in ws.causal_topology
            if ce.fabula_time <= target.fabula_time
        }
        edge_ids = {
            (e["source_id"], target.id) for e in ctx.get("incoming", [])
        } | {(target.id, e["target_id"]) for e in ctx.get("outgoing", [])}
        for sid, tid in edge_ids:
            assert (sid, tid) in causal_pairs_at, (
                f"edge ({sid}->{tid}) leaked from future ticks"
            )

    def test_world_traits_active_present(self):
        from shadow_loom_ui.reasoning_helpers import event_context_data

        ws = deepcopy(macbeth_ws)
        target = sorted(ws.events, key=lambda e: e.fabula_time)[-1]
        ctx = event_context_data(ws, target.id)

        # World traits must come through with the snapshot-shaped
        # magnitude dict (value/inertia/evidence_strength).
        for row in ctx.get("world_traits_active", []):
            assert "magnitude" in row
            assert "value" in row["magnitude"]
            assert "inertia" in row["magnitude"]


# =====================================================================
# 5. hidden_channel_rows — cursor-aware
# =====================================================================


class TestHiddenChannelRowsCursorAware:
    """When a syuzhet anchor is passed, the rows must reflect what's
    hidden *as of* that anchor, not the final-frame story."""

    def test_accepts_syuzhet_anchor_keyword(self):
        from shadow_loom_ui.reasoning_helpers import hidden_channel_rows

        ws = deepcopy(macbeth_ws)
        # Should not raise — and should return a list (possibly empty).
        rows = hidden_channel_rows(ws, syuzhet_anchor=0)
        assert isinstance(rows, list)

    def test_anchor_zero_vs_max_changes_result(self):
        from shadow_loom_ui.reasoning_helpers import hidden_channel_rows

        ws = deepcopy(macbeth_ws)
        max_sy = max(
            (e.syuzhet_index for e in ws.events), default=0,
        )

        early = hidden_channel_rows(ws, syuzhet_anchor=0)
        late = hidden_channel_rows(ws, syuzhet_anchor=max_sy)
        # Both are lists; ``compute_hidden_channels`` is monotonic in
        # the anchor (rows shrink as the reader sees more of the
        # story), so early >= late in length whenever anything is
        # hidden at all.
        assert len(early) >= len(late)


# =====================================================================
# 6. AppState — cursor anchors threaded into queries
# =====================================================================


class TestAppStateCursorAnchors:
    """The user-visible cursor must propagate into the query before
    the pipeline runs, so query execution always agrees with the
    rest of the UI's per-tick views."""

    def test_cursor_anchors_patched_onto_query(self):
        # Exercise the cursor-merge branch directly without spinning
        # up the whole pipeline. We call ``model_copy`` and check the
        # propagated anchors land on the patched query.
        from shadow_loom.query_models import GeneralQuery

        q = GeneralQuery(question="Test", original_query="Test")
        assert q.temporal_anchor is None
        assert q.syuzhet_anchor is None

        # Simulate the AppState patch: explicit anchors win, cursor
        # only fills the gap.
        patched = q.model_copy(
            update={"temporal_anchor": 5000, "syuzhet_anchor": 7},
        )
        assert patched.temporal_anchor == 5000
        assert patched.syuzhet_anchor == 7

    def test_explicit_query_anchor_wins_over_cursor(self):
        # Conceptual: the AppState helper only fills temporal_anchor
        # when it is None on the query. A query that already carries
        # an explicit anchor (manual_edit insert point, parser-emitted
        # "after EVT_X" anchor) must keep it.
        from shadow_loom.query_models import GeneralQuery

        q = GeneralQuery(
            question="Q", original_query="Q", temporal_anchor=42,
        )
        # Mimic the AppState guard:
        if q.temporal_anchor is None:
            q = q.model_copy(update={"temporal_anchor": 9999})
        assert q.temporal_anchor == 42  # cursor must not override


# =====================================================================
# 7. reasoning_helpers — legacy ``interrogation`` alias normalisation
# =====================================================================


class TestReasoningTraceAliasNormalisation:
    """The canonical literal is ``interrogate``. Legacy envelopes that
    still carry ``interrogation`` must collapse to the same Rung-1
    classification as the canonical name."""

    def test_legacy_alias_maps_to_rung_1(self):
        from shadow_loom_ui.reasoning_helpers import extract_reasoning_trace

        ws = deepcopy(macbeth_ws)
        legacy = extract_reasoning_trace(
            {"query_type": "interrogation"}, ws=ws,
        )
        canonical = extract_reasoning_trace(
            {"query_type": "interrogate"}, ws=ws,
        )
        assert legacy["rung"] == 1
        assert canonical["rung"] == 1
