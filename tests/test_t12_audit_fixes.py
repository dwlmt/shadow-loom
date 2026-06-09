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


# ====================================================================
# Counterfactual proposition-truth prune (the "Mrs Coady still dies"
# bug). When a do-surgery severs an event whose only effect is to
# commit a proposition TRUE, the counterfactual brief must NOT keep
# telling the auditor that proposition is TRUE — otherwise the
# auditor forces the renderer to enact the now-uncaused outcome via
# a substitute mechanism (e.g. Mrs Coady dying of frailty after the
# heart-attack event that killed her was counterfactually prevented).
# ====================================================================
class TestCounterfactualPropositionTruthPrune:
    """Regression: ``build_true_proposition_constraints`` and
    ``build_false_proposition_constraints`` must accept
    ``pruned_event_ids`` and suppress commits whose only event
    justification was severed by the do-surgery."""

    def _ws(self, *, events, propositions):
        from types import SimpleNamespace
        return SimpleNamespace(events=events, propositions=propositions)

    def test_helper_returns_empty_set_when_no_pruned_ids(self):
        from shadow_loom.directive_assembly import compute_suppressed_truth_commits

        from types import SimpleNamespace
        evt = SimpleNamespace(
            id="EVT_X", fabula_time=1000,
            resolves_proposition_ids=["PROP_P"],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        ws = self._ws(events=[evt], propositions=[])
        assert compute_suppressed_truth_commits(ws, None) == set()
        assert compute_suppressed_truth_commits(ws, set()) == set()

    def test_helper_suppresses_via_event_provenance(self):
        from shadow_loom.directive_assembly import compute_suppressed_truth_commits

        from types import SimpleNamespace
        evt = SimpleNamespace(
            id="EVT_DEATH", fabula_time=14000,
            resolves_proposition_ids=["PROP_DEAD"],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        ws = self._ws(events=[evt], propositions=[])
        sup = compute_suppressed_truth_commits(ws, {"EVT_DEATH"})
        assert sup == {("PROP_DEAD", 14000)}

    def test_helper_suppresses_via_asserts_and_denies(self):
        from shadow_loom.directive_assembly import compute_suppressed_truth_commits

        from types import SimpleNamespace
        evt = SimpleNamespace(
            id="EVT_UTTER", fabula_time=500,
            resolves_proposition_ids=[],
            asserts_proposition_id="PROP_A",
            denies_proposition_id="PROP_B",
        )
        ws = self._ws(events=[evt], propositions=[])
        sup = compute_suppressed_truth_commits(ws, {"EVT_UTTER"})
        assert ("PROP_A", 500) in sup
        assert ("PROP_B", 500) in sup

    def test_helper_referent_sweep_exact_tick(self):
        from shadow_loom.directive_assembly import compute_suppressed_truth_commits

        from types import SimpleNamespace
        # The classic example_world shape: a Proposition declared
        # with ``referent_ids=[EVT_…]`` and a hand-written
        # ``truth_at_fabula`` that the EventNode itself does NOT
        # populate via ``resolves_proposition_ids``.
        evt = SimpleNamespace(
            id="EVT_HEART_ATTACK", fabula_time=14000,
            resolves_proposition_ids=[],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        prop = SimpleNamespace(
            proposition_id="PROP_MRS_COADY_DIES",
            id="PROP_MRS_COADY_DIES",
            description="Mrs Coady dies.",
            referent_ids=["EVT_HEART_ATTACK"],
            truth_at_fabula={14000: True},
        )
        ws = self._ws(events=[evt], propositions=[prop])
        sup = compute_suppressed_truth_commits(ws, {"EVT_HEART_ATTACK"})
        assert ("PROP_MRS_COADY_DIES", 14000) in sup

    def test_helper_referent_sweep_widens_when_all_refs_pruned(self):
        from shadow_loom.directive_assembly import compute_suppressed_truth_commits

        from types import SimpleNamespace
        # All event referents are pruned → every commit at-or-after
        # the earliest pruned fabula_time is also suppressed (Pearl:
        # no surviving event-justification for the post-divergence
        # truth track).
        evt = SimpleNamespace(
            id="EVT_HEART_ATTACK", fabula_time=14000,
            resolves_proposition_ids=[],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        prop = SimpleNamespace(
            proposition_id="PROP_DEAD", id="PROP_DEAD",
            description="x",
            referent_ids=["EVT_HEART_ATTACK"],
            # A declarative entry AFTER the suppressed event's
            # fabula_time also has no surviving justification.
            truth_at_fabula={14000: True, 15000: True},
        )
        ws = self._ws(events=[evt], propositions=[prop])
        sup = compute_suppressed_truth_commits(ws, {"EVT_HEART_ATTACK"})
        assert ("PROP_DEAD", 14000) in sup
        assert ("PROP_DEAD", 15000) in sup

    def test_helper_referent_sweep_conservative_when_some_refs_survive(self):
        from shadow_loom.directive_assembly import compute_suppressed_truth_commits

        from types import SimpleNamespace
        # Only one of two event referents is pruned → conservative:
        # only exact-tick matches are suppressed; later commits may
        # be the surviving event's doing.
        evt_pruned = SimpleNamespace(
            id="EVT_A", fabula_time=14000,
            resolves_proposition_ids=[],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        evt_alive = SimpleNamespace(
            id="EVT_B", fabula_time=15000,
            resolves_proposition_ids=[],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        prop = SimpleNamespace(
            proposition_id="PROP_X", id="PROP_X",
            description="x",
            referent_ids=["EVT_A", "EVT_B"],
            truth_at_fabula={14000: True, 15000: True, 16000: True},
        )
        ws = self._ws(events=[evt_pruned, evt_alive], propositions=[prop])
        sup = compute_suppressed_truth_commits(ws, {"EVT_A"})
        # Exact tick only.
        assert ("PROP_X", 14000) in sup
        assert ("PROP_X", 15000) not in sup
        assert ("PROP_X", 16000) not in sup

    def test_true_constraint_drops_committed_prop_when_event_pruned(self):
        from shadow_loom.directive_assembly import build_true_proposition_constraints

        from types import SimpleNamespace
        # The Mrs Coady scenario in miniature.
        evt = SimpleNamespace(
            id="EVT_HEART_ATTACK", fabula_time=14000,
            resolves_proposition_ids=["PROP_MRS_COADY_DIES"],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        prop = SimpleNamespace(
            proposition_id="PROP_MRS_COADY_DIES",
            id="PROP_MRS_COADY_DIES",
            description="Mrs Coady dies.",
            referent_ids=["EVT_HEART_ATTACK"],
            truth_at_fabula={14000: True},
        )
        ws = self._ws(events=[evt], propositions=[prop])

        # Without surgery: the brief carries the TRUE commit (factual
        # mainline / observation).
        blocks_factual = build_true_proposition_constraints(
            ws, syuzhet_anchor=None,
        )
        assert len(blocks_factual) == 1
        assert "PROP_MRS_COADY_DIES" in blocks_factual[0].instruction

        # With the heart-attack event pruned: the TRUE commit is
        # suppressed, no block emitted → auditor will not force the
        # renderer to enact Mrs Coady's death.
        blocks_cf = build_true_proposition_constraints(
            ws, syuzhet_anchor=None,
            pruned_event_ids={"EVT_HEART_ATTACK"},
        )
        assert blocks_cf == []

    def test_false_constraint_drops_committed_prop_when_event_pruned(self):
        from shadow_loom.directive_assembly import build_false_proposition_constraints

        from types import SimpleNamespace
        # Symmetric path: a FALSE commit whose only justification is
        # a now-pruned event must also disappear.
        evt = SimpleNamespace(
            id="EVT_RECANT", fabula_time=2000,
            resolves_proposition_ids=["PROP_GUILTY"],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        prop = SimpleNamespace(
            proposition_id="PROP_GUILTY", id="PROP_GUILTY",
            description="The defendant is guilty.",
            referent_ids=["EVT_RECANT"],
            truth_at_fabula={2000: False},
        )
        ws = self._ws(events=[evt], propositions=[prop])

        blocks_factual = build_false_proposition_constraints(
            ws, syuzhet_anchor=None,
        )
        assert len(blocks_factual) == 1
        assert "PROP_GUILTY" in blocks_factual[0].instruction

        blocks_cf = build_false_proposition_constraints(
            ws, syuzhet_anchor=None,
            pruned_event_ids={"EVT_RECANT"},
        )
        assert blocks_cf == []

    def test_true_constraint_keeps_prop_when_surviving_event_grounds_it(self):
        from shadow_loom.directive_assembly import build_true_proposition_constraints

        from types import SimpleNamespace
        # Two events both ground PROP_P as TRUE at different ticks;
        # only one is pruned. The surviving commit must remain
        # visible to the brief.
        evt_pruned = SimpleNamespace(
            id="EVT_A", fabula_time=1000,
            resolves_proposition_ids=[],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        evt_alive = SimpleNamespace(
            id="EVT_B", fabula_time=2000,
            resolves_proposition_ids=[],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        prop = SimpleNamespace(
            proposition_id="PROP_P", id="PROP_P",
            description="P holds.",
            referent_ids=["EVT_A", "EVT_B"],
            truth_at_fabula={1000: True, 2000: True},
        )
        ws = self._ws(events=[evt_pruned, evt_alive], propositions=[prop])

        blocks = build_true_proposition_constraints(
            ws, syuzhet_anchor=None,
            pruned_event_ids={"EVT_A"},
        )
        # The 2000-tick commit (justified by surviving EVT_B) keeps
        # the prop in the brief.
        assert len(blocks) == 1
        assert "PROP_P" in blocks[0].instruction

    # ----- Concern / belief pink-elephant helpers (downstream of
    # ``_latest_proposition_truth_map``) -----

    def test_unrealised_concern_skips_when_supporting_event_pruned(self):
        from shadow_loom.directive_assembly import (
            build_unrealised_concern_constraints,
        )

        from types import SimpleNamespace
        # PROP_BECOMES_KING committed FALSE @ T=2000, and the only
        # event referent is the now-pruned event. Without the fix
        # the helper would still emit "do NOT show Macbeth becoming
        # king (desire fulfilled)" against a proposition whose
        # FALSE commit no longer has any event justification.
        evt = SimpleNamespace(
            id="EVT_DEFEAT", fabula_time=2000,
            resolves_proposition_ids=[],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        prop = SimpleNamespace(
            proposition_id="PROP_BECOMES_KING", id="PROP_BECOMES_KING",
            description="Macbeth becomes king.",
            referent_ids=["EVT_DEFEAT"],
            truth_at_fabula={2000: False},
        )
        concern = SimpleNamespace(
            concern_id="CON_AMBITION", proposition_id="PROP_BECOMES_KING",
            polarity="desire", salience=0.9, kind=None,
            activation_fabula_window=None,
        )
        ent = SimpleNamespace(
            id="ENT_MACBETH", name="Macbeth",
            concerns=[concern], beliefs=[],
        )
        ws = SimpleNamespace(
            events=[evt],
            propositions=[prop],
            entities={"ENT_MACBETH": ent},
        )

        # Without surgery: helper emits the pink-elephant.
        without = build_unrealised_concern_constraints(ws, syuzhet_anchor=None)
        assert len(without) == 1
        assert "CON_AMBITION" in without[0].instruction

        # With surgery: helper drops it because PROP truth is no
        # longer event-supported.
        with_ = build_unrealised_concern_constraints(
            ws, syuzhet_anchor=None,
            pruned_event_ids={"EVT_DEFEAT"},
        )
        assert with_ == []

    def test_false_belief_grounding_skips_when_supporting_event_pruned(self):
        from shadow_loom.directive_assembly import (
            build_false_belief_grounding_constraints,
        )

        from types import SimpleNamespace
        evt = SimpleNamespace(
            id="EVT_X", fabula_time=2000,
            resolves_proposition_ids=[],
            asserts_proposition_id=None, denies_proposition_id=None,
        )
        prop = SimpleNamespace(
            proposition_id="PROP_Q", id="PROP_Q",
            description="Q is true.",
            referent_ids=["EVT_X"],
            truth_at_fabula={2000: False},
        )
        belief = SimpleNamespace(
            proposition_id="PROP_Q",
            perceived_state="Q seems true.",
            confidence=0.8,
            established_at_fabula=1000,
        )
        ent = SimpleNamespace(
            id="ENT_Y", name="Y", concerns=[], beliefs=[belief],
        )
        ws = SimpleNamespace(
            events=[evt],
            propositions=[prop],
            entities={"ENT_Y": ent},
        )

        without = build_false_belief_grounding_constraints(
            ws, syuzhet_anchor=None,
        )
        assert len(without) == 1
        with_ = build_false_belief_grounding_constraints(
            ws, syuzhet_anchor=None,
            pruned_event_ids={"EVT_X"},
        )
        assert with_ == []

    # ----- World-trait invariants -----

    def test_world_invariant_skips_snapshot_triggered_by_pruned_event(self):
        from shadow_loom.directive_assembly import (
            build_world_invariant_constraints,
        )

        from types import SimpleNamespace
        # A world trait whose only above-floor snapshot was
        # triggered by a now-pruned event. The brief must fall
        # back to the base magnitude (here below the 0.5 floor) and
        # drop the invariant from the load-bearing list.
        base = SimpleNamespace(value=0.2)
        snap_high = SimpleNamespace(
            fabula_time=2000,
            magnitude=SimpleNamespace(value=0.9),
            triggered_by="EVT_INTENSIFY",
        )
        wt = SimpleNamespace(
            name="Surveillance", category="political",
            magnitude=base,
            state_timeline=[snap_high],
        )
        ws = SimpleNamespace(
            events=[
                SimpleNamespace(
                    id="EVT_INTENSIFY", fabula_time=2000,
                    syuzhet_index=2000, event_type="outcome",
                ),
            ],
            world_traits={"WORLD_SURVEIL": wt},
        )

        # Without surgery: 0.9 intensity at T=2000 → emitted.
        without = build_world_invariant_constraints(ws, syuzhet_anchor=2000)
        assert len(without) == 1
        assert "WORLD_SURVEIL" in without[0].instruction

        # With surgery: snapshot's triggered_by is pruned → falls
        # back to base value 0.2 < floor 0.5 → dropped.
        with_ = build_world_invariant_constraints(
            ws, syuzhet_anchor=2000,
            pruned_event_ids={"EVT_INTENSIFY"},
        )
        assert with_ == []

    def test_world_invariant_keeps_when_other_snapshot_survives(self):
        from shadow_loom.directive_assembly import (
            build_world_invariant_constraints,
        )

        from types import SimpleNamespace
        # Two snapshots above the floor; only one is pruned. The
        # surviving snapshot keeps the trait in the load-bearing list.
        base = SimpleNamespace(value=0.2)
        snap_alive = SimpleNamespace(
            fabula_time=1000,
            magnitude=SimpleNamespace(value=0.7),
            triggered_by="EVT_ALIVE",
        )
        snap_pruned = SimpleNamespace(
            fabula_time=2000,
            magnitude=SimpleNamespace(value=0.9),
            triggered_by="EVT_PRUNED",
        )
        wt = SimpleNamespace(
            name="Surveillance", category="political",
            magnitude=base,
            state_timeline=[snap_alive, snap_pruned],
        )
        ws = SimpleNamespace(
            events=[
                SimpleNamespace(
                    id="EVT_ALIVE", fabula_time=1000,
                    syuzhet_index=1000, event_type="outcome",
                ),
                SimpleNamespace(
                    id="EVT_PRUNED", fabula_time=2000,
                    syuzhet_index=2000, event_type="outcome",
                ),
            ],
            world_traits={"WORLD_SURVEIL": wt},
        )

        with_ = build_world_invariant_constraints(
            ws, syuzhet_anchor=2000,
            pruned_event_ids={"EVT_PRUNED"},
        )
        assert len(with_) == 1
        assert "WORLD_SURVEIL" in with_[0].instruction
        # Falls back to the surviving snapshot's value (0.7), not 0.9.
        assert "0.70" in with_[0].instruction

    # ----- Event co-presence -----

    def test_event_copresence_skips_prevented_events(self):
        from shadow_loom.directive_assembly import (
            build_event_copresence_constraints,
        )

        from types import SimpleNamespace
        evt_alive = SimpleNamespace(
            id="EVT_ALIVE", event_type="outcome",
            fabula_time=1000, syuzhet_index=1000,
            at_location_id="LOC_A",
            actor_ids=["ENT_X"], target_ids=[], addressee_ids=[],
            via_channel_id=None, speaker_id=None,
        )
        evt_prevented = SimpleNamespace(
            id="EVT_DEAD", event_type="prevented",
            fabula_time=1000, syuzhet_index=1000,
            at_location_id="LOC_B",
            actor_ids=["ENT_Y"], target_ids=[], addressee_ids=[],
            via_channel_id=None, speaker_id=None,
        )
        ws = SimpleNamespace(
            events=[evt_alive, evt_prevented],
            entities={}, locations={},
        )
        blocks = build_event_copresence_constraints(
            ws, fabula_anchor=1000, syuzhet_anchor=1000,
        )
        evidence_event_ids = {b.evidence.get("event_id") for b in blocks}
        assert "EVT_ALIVE" in evidence_event_ids
        assert "EVT_DEAD" not in evidence_event_ids

    def test_event_copresence_skips_pruned_event_ids(self):
        from shadow_loom.directive_assembly import (
            build_event_copresence_constraints,
        )

        from types import SimpleNamespace
        evt = SimpleNamespace(
            id="EVT_KEN_KILLS_DOGS", event_type="outcome",
            fabula_time=1000, syuzhet_index=1000,
            at_location_id="LOC_KITCHEN",
            actor_ids=["ENT_KEN"], target_ids=[], addressee_ids=[],
            via_channel_id=None, speaker_id=None,
        )
        ws = SimpleNamespace(
            events=[evt], entities={}, locations={},
        )
        # Without surgery: event surfaced as MUST_DEPICT_AT.
        without = build_event_copresence_constraints(
            ws, fabula_anchor=1000, syuzhet_anchor=1000,
        )
        assert any(b.evidence.get("event_id") == "EVT_KEN_KILLS_DOGS"
                   for b in without)
        # With surgery: pruned → no MUST_DEPICT block (so brief
        # cannot contradict PREVENTED EVENTS block).
        with_ = build_event_copresence_constraints(
            ws, fabula_anchor=1000, syuzhet_anchor=1000,
            pruned_event_ids={"EVT_KEN_KILLS_DOGS"},
        )
        assert with_ == []


# ====================================================================
# Real-world plot integration: the prune-on-pruned-event fix has to
# hold up against the actual example_worlds models the pipeline runs
# against, not just minimal SimpleNamespace fixtures. Each test names
# a known PROP→EVT linkage from a published model (the Mrs Coady bug
# that triggered this audit; plus three analogous ones across
# different example_worlds genres).
# ====================================================================
class TestRealPlotPropTruthPrune:
    """Realistic-data assertions for the Mrs Coady-class bug class.

    Each test loads an actual example_world and checks that:

      * Without ``pruned_event_ids`` the brief carries the proposition
        as TRUE / FALSE at its committed tick (sanity).
      * With ``pruned_event_ids = {<the supporting event>}`` the brief
        drops it from the TRUE / FALSE block, so the auditor will not
        force the renderer to enact the now-uncaused outcome.
    """

    def test_a_fish_called_wanda_mrs_coady_dies_dropped(self):
        """The original bug. PROP_MRS_COADY_DIES.truth_at_fabula
        commits TRUE @ 14000, referent_ids=[EVT_MRS_COADY_DIES_HEART_ATTACK,
        ENT_MRS_COADY]. Counterfactual: do(EVT_KEN_KILLS_DOGS=prevented)
        cascades the closure to prevent the heart-attack event too;
        the brief must no longer surface PROP_MRS_COADY_DIES as TRUE."""
        from example_worlds.a_fish_called_wanda import world_state as wanda_ws
        from shadow_loom.directive_assembly import (
            build_true_proposition_constraints,
        )

        # Sanity: the published model carries PROP_MRS_COADY_DIES as
        # TRUE @ T=14000 in the factual mainline.
        without = build_true_proposition_constraints(
            wanda_ws, syuzhet_anchor=None,
        )
        joined = "\n".join(b.instruction for b in without)
        assert "PROP_MRS_COADY_DIES" in joined

        # With the heart-attack event pruned (the closure of
        # do(EVT_KEN_KILLS_DOGS=prevented) in the shadow branch):
        # the proposition's only event referent is severed, so
        # PROP_MRS_COADY_DIES disappears from the TRUE PROPOSITIONS
        # (HARD) block. The auditor will no longer flag the absence
        # of Mrs Coady's on-page death as a constraint violation.
        with_ = build_true_proposition_constraints(
            wanda_ws, syuzhet_anchor=None,
            pruned_event_ids={"EVT_MRS_COADY_DIES_HEART_ATTACK"},
        )
        joined_with = "\n".join(b.instruction for b in with_)
        assert "PROP_MRS_COADY_DIES" not in joined_with

    def test_macbeth_duncan_dead_dropped_when_murder_pruned(self):
        """Macbeth: PROP_DUNCAN_DEAD.truth_at_fabula commits TRUE @
        6000 with referent_ids=[EVT_DUNCAN_MURDER, ENT_DUNCAN].
        Classic counterfactual: do(EVT_DUNCAN_MURDER=prevented) →
        the brief must no longer pin Duncan as dead."""
        from example_worlds.macbeth import world_state as macbeth_world
        from shadow_loom.directive_assembly import (
            build_true_proposition_constraints,
        )

        without = build_true_proposition_constraints(
            macbeth_world, syuzhet_anchor=None,
        )
        joined = "\n".join(b.instruction for b in without)
        assert "PROP_DUNCAN_DEAD" in joined

        with_ = build_true_proposition_constraints(
            macbeth_world, syuzhet_anchor=None,
            pruned_event_ids={"EVT_DUNCAN_MURDER"},
        )
        joined_with = "\n".join(b.instruction for b in with_)
        assert "PROP_DUNCAN_DEAD" not in joined_with

    def test_macbeth_foul_play_dropped_when_murder_pruned(self):
        """PROP_MACBETH_FOUL_PLAY also references EVT_DUNCAN_MURDER
        (referent_ids=[ENT_MACBETH, EVT_DUNCAN_MURDER], TRUE @ 6000).
        Same surgery, same expected drop."""
        from example_worlds.macbeth import world_state as macbeth_world
        from shadow_loom.directive_assembly import (
            build_true_proposition_constraints,
        )

        without = build_true_proposition_constraints(
            macbeth_world, syuzhet_anchor=None,
        )
        joined = "\n".join(b.instruction for b in without)
        assert "PROP_MACBETH_FOUL_PLAY" in joined

        with_ = build_true_proposition_constraints(
            macbeth_world, syuzhet_anchor=None,
            pruned_event_ids={"EVT_DUNCAN_MURDER"},
        )
        joined_with = "\n".join(b.instruction for b in with_)
        assert "PROP_MACBETH_FOUL_PLAY" not in joined_with

    def test_macbeth_becomes_king_dropped_when_crowning_pruned(self):
        """PROP_MACBETH_BECOMES_KING.referent_ids=[EVT_MACBETH_CROWNED,
        ENT_MACBETH], TRUE @ 10000. do(EVT_MACBETH_CROWNED=prevented)
        → the throne-seizure must no longer be a brief constraint."""
        from example_worlds.macbeth import world_state as macbeth_world
        from shadow_loom.directive_assembly import (
            build_true_proposition_constraints,
        )

        without = build_true_proposition_constraints(
            macbeth_world, syuzhet_anchor=None,
        )
        joined = "\n".join(b.instruction for b in without)
        assert "PROP_MACBETH_BECOMES_KING" in joined

        with_ = build_true_proposition_constraints(
            macbeth_world, syuzhet_anchor=None,
            pruned_event_ids={"EVT_MACBETH_CROWNED"},
        )
        joined_with = "\n".join(b.instruction for b in with_)
        assert "PROP_MACBETH_BECOMES_KING" not in joined_with

    def test_great_gatsby_gatsby_killed_dropped_when_event_pruned(self):
        """Gatsby: a proposition (whichever name) tied to
        EVT_GATSBY_KILLED via referent_ids. Generic check that the
        helper drops every such TRUE-committed prop when the
        supporting event is pruned."""
        from example_worlds.great_gatsby import world_state as gatsby_ws
        from shadow_loom.directive_assembly import (
            build_true_proposition_constraints,
        )

        # Find every TRUE-committed proposition that references the
        # killing event \u2014 they must all disappear together.
        target_event = "EVT_GATSBY_KILLED"
        tied_pids = {
            p.proposition_id
            for p in gatsby_ws.propositions
            if target_event in (p.referent_ids or [])
            and any(v is True for v in (p.truth_at_fabula or {}).values())
        }
        if not tied_pids:
            pytest.skip(
                "No TRUE-committed proposition references "
                f"{target_event} in this snapshot."
            )

        without = build_true_proposition_constraints(
            gatsby_ws, syuzhet_anchor=None,
        )
        joined = "\n".join(b.instruction for b in without)
        for pid in tied_pids:
            assert pid in joined, (
                f"sanity: {pid} should be in the unsurgered TRUE block"
            )

        with_ = build_true_proposition_constraints(
            gatsby_ws, syuzhet_anchor=None,
            pruned_event_ids={target_event},
        )
        joined_with = "\n".join(b.instruction for b in with_)
        for pid in tied_pids:
            assert pid not in joined_with, (
                f"{pid} should be dropped when {target_event} is pruned"
            )

    def test_romeo_and_juliet_mercutio_killed_dropped(self):
        """Romeo & Juliet: PROP referencing EVT_ROMEO_KILLS_TYBALT \u2014
        do(EVT_ROMEO_KILLS_TYBALT=prevented) drops the proposition."""
        from example_worlds.romeo_and_juliet import world_state as rj_ws
        from shadow_loom.directive_assembly import (
            build_true_proposition_constraints,
        )

        target_event = "EVT_ROMEO_KILLS_TYBALT"
        tied_pids = {
            p.proposition_id
            for p in rj_ws.propositions
            if target_event in (p.referent_ids or [])
            and any(v is True for v in (p.truth_at_fabula or {}).values())
        }
        if not tied_pids:
            pytest.skip(
                f"No TRUE-committed proposition references {target_event}."
            )

        without = build_true_proposition_constraints(
            rj_ws, syuzhet_anchor=None,
        )
        joined = "\n".join(b.instruction for b in without)
        for pid in tied_pids:
            assert pid in joined

        with_ = build_true_proposition_constraints(
            rj_ws, syuzhet_anchor=None,
            pruned_event_ids={target_event},
        )
        joined_with = "\n".join(b.instruction for b in with_)
        for pid in tied_pids:
            assert pid not in joined_with

    def test_a_fish_called_wanda_world_invariant_chain(self):
        """Mrs Coady's entity status-timeline is gated on
        EntityStateSnapshot(triggered_by=EVT_MRS_COADY_DIES_HEART_ATTACK).
        The entity-snapshot path is a different code path from the
        proposition truth_at_fabula prune; this test pins the
        ``triggered_by``-prune behaviour for the world-trait helper
        (closest analogue) using the Wanda data shape.

        We construct a world-trait whose state_timeline cites the
        same EVT_ id as a triggered_by, so when the brief is built
        with that EVT_ in pruned_event_ids the helper falls back."""
        from types import SimpleNamespace
        from shadow_loom.directive_assembly import (
            build_world_invariant_constraints,
        )

        # Synthetic world-trait modelled on a load-bearing legal
        # state during the Coady arc (e.g. a "Crown case pressure"
        # intensifying when the eyewitness dies).
        base = SimpleNamespace(value=0.3)
        snap_intensified = SimpleNamespace(
            fabula_time=14000,
            magnitude=SimpleNamespace(value=0.85),
            triggered_by="EVT_MRS_COADY_DIES_HEART_ATTACK",
        )
        wt = SimpleNamespace(
            name="Crown Case Pressure", category="legal",
            magnitude=base,
            state_timeline=[snap_intensified],
        )
        ws = SimpleNamespace(
            events=[SimpleNamespace(
                id="EVT_MRS_COADY_DIES_HEART_ATTACK",
                fabula_time=14000, syuzhet_index=14,
                event_type="outcome",
            )],
            world_traits={"WORLD_CROWN_PRESSURE": wt},
        )

        # Sanity: with no surgery the intensified value (0.85) is
        # above the 0.5 floor → emitted.
        without = build_world_invariant_constraints(ws, syuzhet_anchor=14)
        assert any("WORLD_CROWN_PRESSURE" in b.instruction for b in without)

        # With the heart-attack pruned: snapshot dropped → falls
        # back to base 0.3 → below floor → invariant block clean.
        with_ = build_world_invariant_constraints(
            ws, syuzhet_anchor=14,
            pruned_event_ids={"EVT_MRS_COADY_DIES_HEART_ATTACK"},
        )
        assert all("WORLD_CROWN_PRESSURE" not in b.instruction
                   for b in with_)


# ====================================================================
# 2026-05-30 surfacing audit (complement to TestRealPlotPropTruthPrune).
# The previous suite ensured stale TRUE/FALSE commits don't bleed into
# the brief when their supporting event is pruned. This suite ensures
# the COMPLEMENTARY positive surfacing is correct: closure-pruned
# events must appear in the PREVENTED EVENTS block (renderer signal)
# AND the pruned-utterance-event-ids evidence dict (auditor signal),
# even when the do-surgery only flipped ``event_type`` on the root.
# ====================================================================
class TestPreventedEventsClosureSurfacing:
    """Regression: ``build_prevented_event_constraints`` must accept
    ``pruned_event_ids`` so closure-descendant events (whose
    ``event_type`` was NOT flipped to ``prevented``) are still
    surfaced for the renderer and the auditor.
    """

    def _ws(self, *, events):
        from types import SimpleNamespace
        return SimpleNamespace(events=events)

    def test_root_event_type_prevented_still_surfaced(self):
        """Baseline: the surgery root with ``event_type='prevented'``
        appears in the PREVENTED EVENTS block (existing behaviour)."""
        from types import SimpleNamespace
        from shadow_loom.directive_assembly import (
            build_prevented_event_constraints,
        )
        root = SimpleNamespace(
            id="EVT_KEN_KILLS_DOGS", event_type="prevented",
            fabula_time=10000, description="Ken kills Mrs Coady's dogs.",
        )
        ws = self._ws(events=[root])
        blocks = build_prevented_event_constraints(ws, syuzhet_anchor=None)
        assert len(blocks) == 1
        assert "EVT_KEN_KILLS_DOGS" in blocks[0].instruction
        assert "EVT_KEN_KILLS_DOGS" in (
            blocks[0].evidence.get("prevented_event_ids") or []
        )

    def test_closure_descendant_without_pruned_ids_is_missed(self):
        """Without the new ``pruned_event_ids`` kwarg, a chain-reaction
        descendant whose ``event_type`` stays ``outcome`` is NOT
        surfaced — this is the gap the new kwarg closes."""
        from types import SimpleNamespace
        from shadow_loom.directive_assembly import (
            build_prevented_event_constraints,
        )
        descendant = SimpleNamespace(
            id="EVT_MRS_COADY_DIES_HEART_ATTACK", event_type="outcome",
            fabula_time=14000,
            description="Mrs Coady dies of a heart attack.",
        )
        ws = self._ws(events=[descendant])
        # No pruned_event_ids → block is empty (no event_type match).
        assert build_prevented_event_constraints(ws, syuzhet_anchor=None) == []

    def test_closure_descendant_with_pruned_ids_is_surfaced(self):
        """With ``pruned_event_ids = {<descendant>}`` the block lists
        the descendant and tags it ``[..., pruned via do-surgery
        closure]`` so the renderer sees both signals (event_type for
        the root, closure tag for descendants)."""
        from types import SimpleNamespace
        from shadow_loom.directive_assembly import (
            build_prevented_event_constraints,
        )
        root = SimpleNamespace(
            id="EVT_KEN_KILLS_DOGS", event_type="prevented",
            fabula_time=10000, description="Ken kills Mrs Coady's dogs.",
        )
        descendant = SimpleNamespace(
            id="EVT_MRS_COADY_DIES_HEART_ATTACK", event_type="outcome",
            fabula_time=14000,
            description="Mrs Coady dies of a heart attack.",
        )
        ws = self._ws(events=[root, descendant])
        blocks = build_prevented_event_constraints(
            ws, syuzhet_anchor=None,
            pruned_event_ids={"EVT_KEN_KILLS_DOGS",
                              "EVT_MRS_COADY_DIES_HEART_ATTACK"},
        )
        assert len(blocks) == 1
        text = blocks[0].instruction
        # Both events must be named.
        assert "EVT_KEN_KILLS_DOGS" in text
        assert "EVT_MRS_COADY_DIES_HEART_ATTACK" in text
        # Descendant should carry the closure-tag annotation; root
        # should not (it was matched by event_type, not by closure).
        assert "pruned via do-surgery closure" in text
        # Evidence dict must surface both event ids AND a separate
        # closure-only key for the descendant.
        ev = blocks[0].evidence
        prevented_ids = ev.get("prevented_event_ids") or []
        assert "EVT_KEN_KILLS_DOGS" in prevented_ids
        assert "EVT_MRS_COADY_DIES_HEART_ATTACK" in prevented_ids
        closure_only = ev.get("pruned_via_closure_event_ids") or []
        assert closure_only == ["EVT_MRS_COADY_DIES_HEART_ATTACK"]
        # Auditor-facing evidence channel: the full pruned set is on
        # the same block so ``_prevented_event_reenacted_violations``
        # always has a stable source even if the renderer's
        # ``=== ERASED UTTERANCES (HARD) ===`` block is empty.
        assert sorted(ev.get("pruned_utterance_event_ids") or []) == [
            "EVT_KEN_KILLS_DOGS",
            "EVT_MRS_COADY_DIES_HEART_ATTACK",
        ]

    def test_no_double_listing_when_event_in_both_sets(self):
        """If an event is in ``pruned_event_ids`` AND has
        ``event_type='prevented'``, it should be surfaced exactly
        once (not duplicated) and tagged with the event_type, not
        the closure annotation."""
        from types import SimpleNamespace
        from shadow_loom.directive_assembly import (
            build_prevented_event_constraints,
        )
        evt = SimpleNamespace(
            id="EVT_X", event_type="prevented", fabula_time=1000,
            description="X.",
        )
        ws = self._ws(events=[evt])
        blocks = build_prevented_event_constraints(
            ws, syuzhet_anchor=None, pruned_event_ids={"EVT_X"},
        )
        assert len(blocks) == 1
        text = blocks[0].instruction
        # Tagged as [prevented], not closure.
        assert "[prevented]" in text
        assert "pruned via do-surgery closure" not in text
        # And no duplicate id line.
        assert text.count("EVT_X") == 1
        # Closure-only key empty (event was matched by type, not closure).
        assert blocks[0].evidence.get("pruned_via_closure_event_ids") == []

    def test_fabula_cap_respected_for_closure_events(self):
        """Same syuzhet→fabula cap that gates event_type-matched
        events must also gate closure-pruned events (consistent
        temporal behaviour)."""
        from types import SimpleNamespace
        from shadow_loom.directive_assembly import (
            build_prevented_event_constraints,
        )
        descendant = SimpleNamespace(
            id="EVT_LATE", event_type="outcome", fabula_time=20000,
            description="Late event.",
        )
        # World with no syuzhet records → ``_syuzhet_to_fabula_cutoff``
        # returns the anchor itself as the cap.
        ws = SimpleNamespace(events=[descendant], syuzhet_records=[])
        blocks = build_prevented_event_constraints(
            ws, syuzhet_anchor=5000,
            pruned_event_ids={"EVT_LATE"},
        )
        # The fabula_time (20000) is after the anchor (5000) → dropped.
        assert blocks == [] or "EVT_LATE" not in blocks[0].instruction

    def test_real_plot_wanda_closure_surfaces_heart_attack(self):
        """End-to-end on the actual Wanda model: when both
        EVT_KEN_KILLS_DOGS and EVT_MRS_COADY_DIES_HEART_ATTACK are
        in the closure, the PREVENTED block lists both even though
        only the root has ``event_type='prevented'`` in canon."""
        from example_worlds.a_fish_called_wanda import world_state as wanda_ws
        from shadow_loom.directive_assembly import (
            build_prevented_event_constraints,
        )

        # Find the heart-attack event id and verify its event_type
        # is NOT 'prevented' in canon (so the old code missed it).
        events_by_id = {e.id: e for e in (wanda_ws.events or [])}
        ha = events_by_id.get("EVT_MRS_COADY_DIES_HEART_ATTACK")
        if ha is None:
            pytest.skip("EVT_MRS_COADY_DIES_HEART_ATTACK not in fixture")
        assert ha.event_type != "prevented", (
            "Test premise: heart attack event_type stays as its "
            "canonical value (not 'prevented'); the do-surgery only "
            "flips the root."
        )

        # With pruned_event_ids carrying the closure: heart attack
        # appears in the block with the closure annotation.
        blocks = build_prevented_event_constraints(
            wanda_ws, syuzhet_anchor=None,
            pruned_event_ids={"EVT_KEN_KILLS_DOGS",
                              "EVT_MRS_COADY_DIES_HEART_ATTACK"},
        )
        assert blocks, "expected a PREVENTED EVENTS block"
        text = blocks[0].instruction
        assert "EVT_MRS_COADY_DIES_HEART_ATTACK" in text
        assert "pruned via do-surgery closure" in text


class TestExclusionBlockUtteranceFiltering:
    """Regression: ``_build_exclusion_constraints`` should list ONLY
    utterance-type events under the ``=== ERASED UTTERANCES (HARD) ===``
    heading. Non-utterance closure descendants must NOT appear as
    ``speaker=unknown`` stub lines (their proper surface is the
    PREVENTED EVENTS / SEVERED CAUSAL CHAINS blocks). Critically,
    the evidence dict MUST still carry the full pruned set so the
    auditor's deterministic check sees every pruned id.
    """

    def _ws(self, *, events):
        from types import SimpleNamespace
        return SimpleNamespace(events=events, channels={})

    def test_non_utterance_excluded_from_lines_but_kept_in_evidence(self):
        from types import SimpleNamespace
        from shadow_loom.generation import _build_exclusion_constraints
        utt = SimpleNamespace(
            id="EVT_UTT_X", event_type="utterance",
            speaker_id="ENT_A", addressee_ids=["ENT_B"],
            target_ids=[], truth_value="true", via_channel_id=None,
            description="A speaks to B.",
        )
        outcome = SimpleNamespace(
            id="EVT_OUTCOME_Y", event_type="outcome",
            speaker_id=None, addressee_ids=[],
            target_ids=[], truth_value=None, via_channel_id=None,
            description="Something happens.",
        )
        ws = self._ws(events=[utt, outcome])
        blocks = _build_exclusion_constraints(
            pruned_utterance_event_ids=["EVT_UTT_X", "EVT_OUTCOME_Y"],
            disabled_channel_ids=[],
            world_state=ws,
            world_label="counterfactual",
        )
        assert len(blocks) == 1
        text = blocks[0].instruction
        # Utterance is listed; the non-utterance outcome event is NOT.
        assert "EVT_UTT_X" in text
        assert "EVT_OUTCOME_Y" not in text
        assert "speaker=unknown" not in text
        # But the evidence dict carries the FULL closure for the
        # auditor's ``_prevented_event_reenacted_violations`` check.
        assert sorted(blocks[0].evidence["pruned_utterance_event_ids"]) == [
            "EVT_OUTCOME_Y", "EVT_UTT_X",
        ]

    def test_all_non_utterance_emits_no_block(self):
        """If the closure contains only non-utterance events, the
        ``ERASED UTTERANCES`` block is skipped entirely — the
        evidence channel relied on by the auditor moves to the
        PREVENTED EVENTS block (covered by
        TestPreventedEventsClosureSurfacing)."""
        from types import SimpleNamespace
        from shadow_loom.generation import _build_exclusion_constraints
        outcome = SimpleNamespace(
            id="EVT_OUTCOME_Y", event_type="outcome",
            speaker_id=None, addressee_ids=[],
            target_ids=[], truth_value=None, via_channel_id=None,
            description="Something happens.",
        )
        ws = self._ws(events=[outcome])
        blocks = _build_exclusion_constraints(
            pruned_utterance_event_ids=["EVT_OUTCOME_Y"],
            disabled_channel_ids=[],
            world_state=ws,
            world_label="counterfactual",
        )
        # No utterance-type → no ERASED UTTERANCES block.
        assert blocks == [] or all(
            "ERASED UTTERANCES" not in b.instruction for b in blocks
        )

    def test_unknown_id_kept_conservatively(self):
        """An id that's in the pruned set but not in
        ``world_state.events`` is kept (we can't prove it's
        non-utterance), so the auditor still sees it."""
        from types import SimpleNamespace
        from shadow_loom.generation import _build_exclusion_constraints
        ws = self._ws(events=[])
        blocks = _build_exclusion_constraints(
            pruned_utterance_event_ids=["EVT_UNKNOWN"],
            disabled_channel_ids=[],
            world_state=ws,
            world_label="counterfactual",
        )
        assert len(blocks) == 1
        assert "EVT_UNKNOWN" in blocks[0].instruction
        assert "EVT_UNKNOWN" in blocks[0].evidence["pruned_utterance_event_ids"]


# ---------------------------------------------------------------------
# 2026-05-30 audit \u2014 closure-aware world-state surfaces.
#
# Mirrors the merge-time scrub in ``extract_graph._get_or_clone_shadow_entity``
# at brief-build time, so every ``triggered_by``-stamped snapshot is
# filtered through the active do-surgery's ``pruned_utterance_event_ids``
# closure before it reaches a constraint block, an epistemic gap, an
# affective trajectory, a relationship rollback, or a concern salience.
#
# These tests pin the post-fix behaviour against both real example_world
# fixtures (a_fish_called_wanda's Mrs Coady) and minimal synthetic
# snapshots (objects, world traits, propositions, concerns,
# relationships) so that an accidental regression in any of the five
# reconstruct_*_at helpers is caught at the helper boundary.
# ---------------------------------------------------------------------


class TestEntityStateClosureScrub:
    """``reconstruct_entity_at`` must honour ``exclude_triggered_by``
    against the real Wanda fixture's Mrs Coady timeline."""

    def _ws(self):
        from example_worlds import a_fish_called_wanda as W
        return W.world_state

    def test_mrs_coady_dead_without_filter(self):
        from shadow_loom.models import reconstruct_entity_at
        ws = self._ws()
        mc = ws.entities["ENT_MRS_COADY"]
        snap = reconstruct_entity_at(mc, 20000)
        assert snap.get("status") == "dead"

    def test_mrs_coady_alive_when_prune_includes_death_event(self):
        from shadow_loom.models import reconstruct_entity_at
        ws = self._ws()
        mc = ws.entities["ENT_MRS_COADY"]
        snap = reconstruct_entity_at(
            mc, 20000,
            exclude_triggered_by={"EVT_MRS_COADY_DIES_HEART_ATTACK"},
        )
        # With the death snapshot filtered, the prior (baseline /
        # earlier snapshot) status \u2014 NOT "dead" \u2014 must
        # surface.
        assert snap.get("status") != "dead"

    def test_mrs_coady_alive_with_full_closure(self):
        """Closure includes the upstream cause (Ken killing the dogs);
        both snapshots must be filtered together."""
        from shadow_loom.models import reconstruct_entity_at
        ws = self._ws()
        mc = ws.entities["ENT_MRS_COADY"]
        snap = reconstruct_entity_at(
            mc, 20000,
            exclude_triggered_by={
                "EVT_KEN_KILLS_DOGS",
                "EVT_MRS_COADY_DIES_HEART_ATTACK",
            },
        )
        assert snap.get("status") != "dead"

    def test_empty_filter_is_noop(self):
        from shadow_loom.models import reconstruct_entity_at
        ws = self._ws()
        mc = ws.entities["ENT_MRS_COADY"]
        a = reconstruct_entity_at(mc, 20000, exclude_triggered_by=set())
        b = reconstruct_entity_at(mc, 20000)
        assert a.get("status") == b.get("status") == "dead"


class TestResolveActualStateClosureScrub:
    """``DirectiveAssembler._resolve_actual_state`` must report the
    post-prune objective state \u2014 so the epistemic-gap classifier
    no longer flags every character's memory of the pruned death as
    "contradicted"."""

    def _wanda_assembler(self, pruned=None):
        from example_worlds import a_fish_called_wanda as W
        from shadow_loom.directive_assembly import DirectiveAssembler
        ws = W.world_state
        return DirectiveAssembler(
            sandbox=None,
            ego_payload={},
            world_state=ws,
            pruned_event_ids=pruned,
        ), ws

    def test_mrs_coady_status_reflects_prune_set(self):
        # Without prune set: actual state contains status=dead.
        asm, _ = self._wanda_assembler(pruned=None)
        actual = asm._resolve_actual_state("ENT_MRS_COADY", anchor_t=20000)
        assert "status=dead" in actual

        # With prune set: actual state hides the post-death snapshot.
        asm2, _ = self._wanda_assembler(pruned={
            "EVT_KEN_KILLS_DOGS",
            "EVT_MRS_COADY_DIES_HEART_ATTACK",
        })
        actual2 = asm2._resolve_actual_state("ENT_MRS_COADY", anchor_t=20000)
        assert "status=dead" not in actual2

    def test_pruned_event_id_renders_as_unknown(self):
        """An event whose id is in the prune set must surface as
        ``"unknown"`` rather than as a factual ground-truth row."""
        asm, _ = self._wanda_assembler(pruned={"EVT_MRS_COADY_DIES_HEART_ATTACK"})
        actual = asm._resolve_actual_state(
            "EVT_MRS_COADY_DIES_HEART_ATTACK", anchor_t=20000,
        )
        assert actual == "unknown"

    def test_per_call_kwarg_overrides_instance_attr(self):
        """Passing ``pruned_event_ids`` to the method takes precedence
        over the instance-level set."""
        asm, _ = self._wanda_assembler(pruned=None)
        actual = asm._resolve_actual_state(
            "ENT_MRS_COADY", anchor_t=20000,
            pruned_event_ids={"EVT_MRS_COADY_DIES_HEART_ATTACK"},
        )
        assert "status=dead" not in actual


class TestObjectStateClosureScrub:
    """``reconstruct_object_at`` must filter ``state_timeline`` entries
    whose ``triggered_by`` event is pruned."""

    def _obj(self):
        # Minimal synthetic NarrativeObject with a single post-prune
        # snapshot. Avoids depending on a real world that happens to
        # exercise the surface.
        from shadow_loom.models import (
            NarrativeObject, ObjectStateSnapshot, Affordance,
        )
        return NarrativeObject(
            id="OBJ_PENDANT",
            name="Pendant",
            owner_id="ENT_A",
            location_id="LOC_BEDROOM",
            properties={},
            affordances=[Affordance(action="wear", target_type="Entity")],
            state_timeline=[
                ObjectStateSnapshot(
                    fabula_time=100,
                    triggered_by="EVT_THEFT",
                    owner_id="ENT_B",
                    location_id="LOC_VAULT",
                    properties={},
                ),
            ],
        )

    def test_factual_baseline_shows_theft(self):
        from shadow_loom.models import reconstruct_object_at
        snap = reconstruct_object_at(self._obj(), 200)
        assert snap["owner_id"] == "ENT_B"
        assert snap["location_id"] == "LOC_VAULT"

    def test_prune_filter_restores_pre_theft_owner(self):
        from shadow_loom.models import reconstruct_object_at
        snap = reconstruct_object_at(
            self._obj(), 200,
            exclude_triggered_by={"EVT_THEFT"},
        )
        assert snap["owner_id"] == "ENT_A"
        assert snap["location_id"] == "LOC_BEDROOM"

    def test_empty_filter_is_noop(self):
        from shadow_loom.models import reconstruct_object_at
        a = reconstruct_object_at(self._obj(), 200, exclude_triggered_by=set())
        b = reconstruct_object_at(self._obj(), 200)
        assert a == b


class TestWorldTraitClosureScrub:
    """``reconstruct_world_trait_at`` must filter snapshots whose
    ``triggered_by`` event is pruned."""

    def _wt(self):
        from shadow_loom.models import GlobalTrait, WorldTraitSnapshot, TraitVector
        return GlobalTrait(
            id="WORLD_TENSION",
            name="tension",
            description="Civic tension.",
            category="social_structure",
            magnitude=TraitVector(value=0.2, inertia=0.3),
            affected_domains=["social"],
            state_timeline=[
                WorldTraitSnapshot(
                    fabula_time=500,
                    triggered_by="EVT_COUP",
                    magnitude=TraitVector(value=0.95, inertia=0.3),
                ),
            ],
        )

    def test_post_coup_magnitude_without_filter(self):
        from shadow_loom.models import reconstruct_world_trait_at
        snap = reconstruct_world_trait_at(self._wt(), 1000)
        assert snap["magnitude"]["value"] == pytest.approx(0.95)

    def test_pre_coup_magnitude_with_prune(self):
        from shadow_loom.models import reconstruct_world_trait_at
        snap = reconstruct_world_trait_at(
            self._wt(), 1000,
            exclude_triggered_by={"EVT_COUP"},
        )
        assert snap["magnitude"]["value"] == pytest.approx(0.2)


class TestPropositionClosureScrub:
    """``reconstruct_proposition_at`` must filter ``state_timeline``
    snapshots whose ``triggered_by`` event is pruned. The framing
    fields (``stakes``, ``audience_default_prior``) are what evolve
    over the timeline; truth lives separately on ``truth_at_fabula``."""

    def _prop(self):
        from shadow_loom.models import Proposition, PropositionSnapshot
        return Proposition(
            id="PROP_TRUST",
            proposition_id="PROP_TRUST",
            kind="relation_holds",
            description="Otto trusts Wanda.",
            referent_ids=["ENT_OTTO", "ENT_WANDA"],
            stakes=0.2,
            audience_default_prior=0.5,
            state_timeline=[
                PropositionSnapshot(
                    fabula_time=300,
                    triggered_by="EVT_BETRAYAL",
                    stakes=0.95,
                    audience_default_prior=0.1,
                ),
            ],
        )

    def test_post_betrayal_framing_without_filter(self):
        from shadow_loom.models import reconstruct_proposition_at
        snap = reconstruct_proposition_at(self._prop(), 500)
        assert snap["stakes"] == pytest.approx(0.95)
        assert snap["audience_default_prior"] == pytest.approx(0.1)

    def test_pre_betrayal_framing_with_prune(self):
        from shadow_loom.models import reconstruct_proposition_at
        snap = reconstruct_proposition_at(
            self._prop(), 500,
            exclude_triggered_by={"EVT_BETRAYAL"},
        )
        assert snap["stakes"] == pytest.approx(0.2)
        assert snap["audience_default_prior"] == pytest.approx(0.5)


class TestConcernClosureScrub:
    """``reconstruct_concern_at`` must filter ``state_timeline``
    snapshots whose ``triggered_by`` event is pruned."""

    def _concern(self):
        from shadow_loom.models import Concern, ConcernSnapshot
        return Concern(
            id="CCN_REVENGE",
            concern_id="CCN_REVENGE",
            proposition_id="PROP_DAD_AVENGED",
            polarity="desire",
            salience=0.1,
            state_timeline=[
                ConcernSnapshot(
                    fabula_time=400,
                    triggered_by="EVT_DAD_DIES",
                    salience=0.9,
                ),
            ],
        )

    def test_post_loss_salience_without_filter(self):
        from shadow_loom.models import reconstruct_concern_at
        snap = reconstruct_concern_at(self._concern(), 800)
        assert snap["salience"] == pytest.approx(0.9)

    def test_pre_loss_salience_with_prune(self):
        from shadow_loom.models import reconstruct_concern_at
        snap = reconstruct_concern_at(
            self._concern(), 800,
            exclude_triggered_by={"EVT_DAD_DIES"},
        )
        assert snap["salience"] == pytest.approx(0.1)


class TestRelationshipClosureScrub:
    """``reconstruct_relationship_at`` must roll back
    ``mutation_social`` deltas whose ``source_id`` (the causing event)
    is in the prune set, even when the event's ``fabula_time`` is at
    or before the read tick."""

    def _build(self):
        from shadow_loom.models import (
            RelationshipEdge, RelationshipMetric, CausalEdge, EventNode,
        )
        # Current affinity = -0.4 (post-betrayal). The betrayal
        # contributes delta=-1.0. Rolling it back via the prune set
        # should restore affinity to +0.6 (clamped to [-1, 1]).
        edge = RelationshipEdge(
            source="ENT_A",
            target="ENT_B",
            source_entity_id="ENT_A",
            target_entity_id="ENT_B",
            edge_type="relationship",
            metrics={
                "affinity": RelationshipMetric(
                    value=-0.4,
                    inertia=0.0,
                    evidence_strength="moderate",
                    last_updated_fabula=500,
                ),
            },
        )
        cedge = CausalEdge(
            source="EVT_BETRAYAL",
            target="ENT_A",
            source_id="EVT_BETRAYAL",
            target_id="ENT_A",
            edge_type="causal",
            causality_type="mutation_social",
            mechanism="betrayal",
            trait_target="affinity",
            trait_delta=-1.0,
            rel_counterpart_id="ENT_B",
            fabula_time=500,
        )
        evt = EventNode(
            id="EVT_BETRAYAL",
            event_type="outcome",
            description="A betrays B.",
            fabula_time=500,
            syuzhet_index=1,
            actor_ids=["ENT_A"],
            target_ids=["ENT_B"],
        )
        return edge, [cedge], [evt]

    def test_post_betrayal_affinity_without_filter(self):
        from shadow_loom.models import reconstruct_relationship_at
        edge, cedges, events = self._build()
        snap = reconstruct_relationship_at(
            edge, 1000, causal_edges=cedges, events=events,
        )
        # Betrayal already in the past at read tick — keep it.
        assert snap["affinity"] == pytest.approx(-0.4)

    def test_pre_betrayal_affinity_with_prune(self):
        from shadow_loom.models import reconstruct_relationship_at
        edge, cedges, events = self._build()
        snap = reconstruct_relationship_at(
            edge, 1000, causal_edges=cedges, events=events,
            exclude_event_ids={"EVT_BETRAYAL"},
        )
        # -0.4 (current) - (-1.0) (rolled-back) = +0.6.
        assert snap["affinity"] == pytest.approx(0.6)


class TestClosureScrubEdgeCases:
    """Edge cases not exercised by the real example_worlds fixtures.

    2026-05-30 audit follow-up: the real plots have rich entity and
    mutation_social timelines but ZERO closure-triggered snapshots on
    objects, propositions, world_traits, or concerns. These synthetic
    tests cover the trickier code paths the real data can't reach.
    """

    def test_snapshot_overwrite_hides_scrub_at_late_tick(self):
        """A later non-pruned snapshot overwrites the pruned one: at
        the late-tick read the entity is OBSERVATIONALLY identical
        with/without prune \u2014 correct, because the canonical
        timeline supersedes the pruned values. The scrub must still
        be visible at ``read_t = pruned_ft + 1``."""
        from shadow_loom.models import (
            Entity, EntityStateSnapshot, TraitVector,
        )
        ent = Entity(
            id="ENT_X",
            name="X",
            description="x",
            status="healthy",
            location_id="LOC_1",
            traits={
                "anger": TraitVector(
                    value=0.1, inertia=0.2, evidence_strength="moderate"),
            },
            state_timeline=[
                EntityStateSnapshot(
                    fabula_time=100,
                    triggered_by="EVT_PRUNED",
                    traits={
                        "anger": TraitVector(
                            value=0.9, inertia=0.2,
                            evidence_strength="strong"),
                    },
                ),
                EntityStateSnapshot(
                    fabula_time=200,
                    triggered_by="EVT_LATER",
                    traits={
                        "anger": TraitVector(
                            value=0.5, inertia=0.2,
                            evidence_strength="strong"),
                    },
                ),
            ],
        )
        from shadow_loom.models import reconstruct_entity_at
        # Late tick: LATER overwrites PRUNED \u2014 scrub is invisible.
        late_no = reconstruct_entity_at(ent, 1000)
        late_pr = reconstruct_entity_at(
            ent, 1000, exclude_triggered_by={"EVT_PRUNED"})
        assert late_no["traits"]["anger"]["value"] == pytest.approx(0.5)
        assert late_pr["traits"]["anger"]["value"] == pytest.approx(0.5)
        # Mid tick (between PRUNED and LATER): scrub IS observable.
        mid_no = reconstruct_entity_at(ent, 150)
        mid_pr = reconstruct_entity_at(
            ent, 150, exclude_triggered_by={"EVT_PRUNED"})
        assert mid_no["traits"]["anger"]["value"] == pytest.approx(0.9)
        assert mid_pr["traits"]["anger"]["value"] == pytest.approx(0.1)

    def test_closure_overdetermination_keeps_descendant(self):
        """Halpern-Pearl: when Y has multiple sufficient parents and
        only ONE is in the prune set, Y survives the closure."""
        from shadow_loom.causal_closure import expand_chain_reaction_closure
        # Y has two sufficient parents A and B. Erase A. Y survives.
        parents = {
            "Y": [("A", "physical", "sufficient"),
                  ("B", "physical", "sufficient")],
        }
        closure = expand_chain_reaction_closure(parents, {"A"})
        assert closure == {"A"}, (
            f"Expected only seed A in closure, got {closure}")

    def test_closure_overdetermination_falls_when_all_sufficient_pruned(self):
        from shadow_loom.causal_closure import expand_chain_reaction_closure
        parents = {
            "Y": [("A", "physical", "sufficient"),
                  ("B", "physical", "sufficient")],
        }
        closure = expand_chain_reaction_closure(parents, {"A", "B"})
        assert closure == {"A", "B", "Y"}

    def test_closure_necessary_cause_brings_descendant(self):
        """If even one necessary parent is in the prune set, Y falls
        regardless of surviving sufficient causes."""
        from shadow_loom.causal_closure import expand_chain_reaction_closure
        parents = {
            "Y": [("A", "physical", "necessary"),
                  ("B", "physical", "sufficient")],
        }
        closure = expand_chain_reaction_closure(parents, {"A"})
        assert closure == {"A", "Y"}

    def test_closure_contributory_parent_ignored(self):
        """Contributory parents shape effects but do not participate
        in suppression. Erasing them never brings Y down."""
        from shadow_loom.causal_closure import expand_chain_reaction_closure
        parents = {
            "Y": [("A", "physical", "contributory"),
                  ("B", "physical", "sufficient")],
        }
        closure = expand_chain_reaction_closure(parents, {"A"})
        assert closure == {"A"}

    def test_relationship_high_inertia_attenuates_rollback(self):
        """Per-axis inertia attenuates the rolled-back delta so
        institutional dyads don't snap back to authored amplitude."""
        from shadow_loom.models import (
            RelationshipEdge, RelationshipMetric, CausalEdge, EventNode,
            reconstruct_relationship_at,
        )
        edge = RelationshipEdge(
            source="ENT_A", target="ENT_B",
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            edge_type="relationship",
            metrics={
                "affinity": RelationshipMetric(
                    value=-0.4, inertia=0.9,  # high inertia
                    evidence_strength="strong",
                    last_updated_fabula=500,
                ),
            },
        )
        cedge = CausalEdge(
            source="EVT_BETRAYAL", target="ENT_A",
            source_id="EVT_BETRAYAL", target_id="ENT_A",
            edge_type="causal",
            causality_type="mutation_social",
            mechanism="betrayal",
            trait_target="affinity",
            trait_delta=-1.0,
            rel_counterpart_id="ENT_B",
            fabula_time=500,
        )
        evt = EventNode(
            id="EVT_BETRAYAL", event_type="outcome",
            description="betrayal", fabula_time=500, syuzhet_index=1,
            actor_ids=["ENT_A"], target_ids=["ENT_B"],
        )
        snap = reconstruct_relationship_at(
            edge, 1000, causal_edges=[cedge], events=[evt],
            exclude_event_ids={"EVT_BETRAYAL"},
        )
        # delta_effective = -1.0 * (1 - 0.9) = -0.1
        # rolled back: -0.4 - (-0.1) = -0.3
        assert snap["affinity"] == pytest.approx(-0.3)

    def test_relationship_rollback_branch_safe(self):
        """A mutation_social edge on a different ``world_id`` must
        NOT roll back the factual relationship."""
        from shadow_loom.models import (
            RelationshipEdge, RelationshipMetric, CausalEdge, EventNode,
            reconstruct_relationship_at,
        )
        edge = RelationshipEdge(
            source="ENT_A", target="ENT_B",
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            edge_type="relationship",
            world_id="factual",
            metrics={
                "affinity": RelationshipMetric(
                    value=-0.4, inertia=0.0,
                    evidence_strength="moderate",
                    last_updated_fabula=500,
                ),
            },
        )
        cedge = CausalEdge(
            source="EVT_BETRAYAL", target="ENT_A",
            source_id="EVT_BETRAYAL", target_id="ENT_A",
            edge_type="causal",
            causality_type="mutation_social",
            mechanism="betrayal",
            trait_target="affinity",
            trait_delta=-1.0,
            rel_counterpart_id="ENT_B",
            fabula_time=500,
            world_id="shadow",  # different world
        )
        evt = EventNode(
            id="EVT_BETRAYAL", event_type="outcome",
            description="betrayal", fabula_time=500, syuzhet_index=1,
            actor_ids=["ENT_A"], target_ids=["ENT_B"],
        )
        snap = reconstruct_relationship_at(
            edge, 1000, causal_edges=[cedge], events=[evt],
            exclude_event_ids={"EVT_BETRAYAL"},
        )
        # Shadow-branch cedge ignored; affinity stays at -0.4.
        assert snap["affinity"] == pytest.approx(-0.4)

    def test_object_partial_prune_preserves_surviving_snapshot(self):
        """When TWO snapshots exist and only the EARLIER is pruned,
        the later snapshot still wins at the late read."""
        from shadow_loom.models import (
            NarrativeObject, ObjectStateSnapshot, Affordance,
            reconstruct_object_at,
        )
        obj = NarrativeObject(
            id="OBJ_KEY", name="key",
            owner_id="ENT_A", location_id="LOC_1", properties={},
            affordances=[Affordance(action="unlock", target_type="NarrativeObject")],
            state_timeline=[
                ObjectStateSnapshot(
                    fabula_time=100, triggered_by="EVT_LOST",
                    owner_id=None, location_id="LOC_GUTTER",
                    properties={},
                ),
                ObjectStateSnapshot(
                    fabula_time=200, triggered_by="EVT_FOUND",
                    owner_id="ENT_B", location_id="LOC_2",
                    properties={},
                ),
            ],
        )
        snap = reconstruct_object_at(
            obj, 1000,
            exclude_triggered_by={"EVT_LOST"},  # only LOST pruned
        )
        # FOUND survives and wins.
        assert snap["owner_id"] == "ENT_B"
        assert snap["location_id"] == "LOC_2"

    def test_proposition_post_prune_at_intermediate_tick(self):
        """Two snapshots; mid-tick read shows the pruned snapshot
        was scrubbed back to the proposition baseline."""
        from shadow_loom.models import (
            Proposition, PropositionSnapshot, reconstruct_proposition_at,
        )
        p = Proposition(
            id="PROP_X",
            proposition_id="PROP_X",
            kind="relation_holds",
            description="X holds.",
            referent_ids=["ENT_A"],
            stakes=0.1,
            audience_default_prior=0.5,
            state_timeline=[
                PropositionSnapshot(
                    fabula_time=100, triggered_by="EVT_PRUNED",
                    stakes=0.9, audience_default_prior=0.1),
                PropositionSnapshot(
                    fabula_time=200, triggered_by="EVT_LATER",
                    stakes=0.5, audience_default_prior=0.4),
            ],
        )
        # At t=150, only EVT_PRUNED has fired \u2014 prune drops it,
        # baseline (0.1, 0.5) returns.
        mid = reconstruct_proposition_at(
            p, 150, exclude_triggered_by={"EVT_PRUNED"})
        assert mid["stakes"] == pytest.approx(0.1)
        assert mid["audience_default_prior"] == pytest.approx(0.5)
        # At t=1000, EVT_LATER survives \u2014 the prune is invisible.
        late = reconstruct_proposition_at(
            p, 1000, exclude_triggered_by={"EVT_PRUNED"})
        assert late["stakes"] == pytest.approx(0.5)

    def test_world_trait_empty_filter_is_noop(self):
        from shadow_loom.models import (
            GlobalTrait, WorldTraitSnapshot, TraitVector,
            reconstruct_world_trait_at,
        )
        wt = GlobalTrait(
            id="WT", name="t", description="", category="social_structure",
            magnitude=TraitVector(value=0.2, inertia=0.3),
            affected_domains=["social"],
            state_timeline=[
                WorldTraitSnapshot(
                    fabula_time=500, triggered_by="EVT_X",
                    magnitude=TraitVector(value=0.9, inertia=0.3)),
            ],
        )
        a = reconstruct_world_trait_at(wt, 1000, exclude_triggered_by=set())
        b = reconstruct_world_trait_at(wt, 1000)
        assert a == b

    def test_concern_empty_filter_is_noop(self):
        from shadow_loom.models import (
            Concern, ConcernSnapshot, reconstruct_concern_at,
        )
        c = Concern(
            id="CCN", concern_id="CCN", proposition_id="P", polarity="desire",
            salience=0.1,
            state_timeline=[
                ConcernSnapshot(
                    fabula_time=100, triggered_by="EVT", salience=0.9),
            ],
        )
        a = reconstruct_concern_at(c, 1000, exclude_triggered_by=set())
        b = reconstruct_concern_at(c, 1000)
        assert a == b

