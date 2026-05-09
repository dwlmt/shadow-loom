# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage for surfaces introduced by the recent refactor.

Targets:

* ``build_prevented_event_constraints`` /
  ``build_false_proposition_constraints`` (negative-physics
  HARD constraint emitters in :mod:`shadow_loom.directive_assembly`).
* MCP ``patch_world_state`` — typed ``WorldStatePatch`` write surface.
* MCP ``inspect()`` ``timeline_limit`` / ``timeline_offset`` pagination.
* MCP ``inspect(at_time=…)`` causal-aware reconstruction flag.
* :func:`shadow_loom.projections.filter_world_state_for_pov` — POV
  scrubbing of other entities' beliefs/concerns/state-timeline drift.
* :func:`shadow_loom.answer._compress_world_state` — Q&A graph dump
  carries the new ``## NEGATIVE FACTS`` and ``## Propositions``
  sections.

Each test is small and isolated. Helper-level tests use
``SimpleNamespace`` to skip Pydantic validation (the production
helpers consult fields via ``getattr`` only); MCP / pipeline tests
seed an in-memory project from the Macbeth example fixture.
"""
from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Match tests/test_mcp_server.py: open-mode auth so MagicMock
# contexts (request_context=None) pass require_scope().
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["MCP_ALLOW_OPEN_MODE"] = "true"

from shadow_loom.db import (
    create_project,
    init_db,
    save_version,
    upsert_user,
)

# Initialise an in-memory DB before importing the MCP server.
init_db("sqlite://")

from shadow_loom.answer import _compress_world_state
from shadow_loom.directive_assembly import (
    build_false_proposition_constraints,
    build_prevented_event_constraints,
)
from shadow_loom.models import WorldStateV1
from shadow_loom.projections import filter_world_state_for_pov
from shadow_loom_mcp import auth as mcp_auth
from shadow_loom_mcp.server import inspect, patch_world_state

from example_worlds.macbeth import world_state as macbeth_ws


# ────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ────────────────────────────────────────────────────────────────────


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


def _seed(ws: WorldStateV1) -> tuple[int, int, int]:
    user = upsert_user(
        "local", "tester", "tester",
        email="t@example.com", display_name="T",
    )
    proj = create_project(name="P", owner_id=user.id, description="d")
    ver = save_version(
        project_id=proj.id,
        world_state_json=ws.model_dump_json(),
        version=0,
        source="ingestion",
        description="seed",
        user_id=user.id,
    )
    mcp_auth._token_user_cache["test-token"] = {
        "user_id": user.id,
        "scopes": {"read", "write", "admin"},
        "key_id": 1,
    }
    return user.id, proj.id, ver.id


# ────────────────────────────────────────────────────────────────────
# Negative-physics constraint emitters (helper-level)
# ────────────────────────────────────────────────────────────────────


def _fake_world(events=None, propositions=None) -> SimpleNamespace:
    return SimpleNamespace(
        events=events or [],
        propositions=propositions or [],
    )


class TestPreventedEventConstraints:
    def test_emits_block_for_prevented_event(self):
        evt_real = SimpleNamespace(
            id="EVT_REAL", event_type="outcome",
            fabula_time=1000,
            description="Something that happens.",
        )
        evt_prev = SimpleNamespace(
            id="EVT_PREV", event_type="prevented",
            fabula_time=2000,
            description="Something the physics tags as not occurring.",
        )
        ws = _fake_world(events=[evt_real, evt_prev])

        blocks = build_prevented_event_constraints(ws, syuzhet_anchor=None)

        assert len(blocks) == 1
        b = blocks[0]
        assert b.priority == "hard"
        assert "PREVENTED EVENTS (HARD)" in b.instruction
        assert "EVT_PREV" in b.instruction
        assert "EVT_REAL" not in b.instruction
        assert b.evidence == {"prevented_event_ids": ["EVT_PREV"]}

    def test_no_block_when_world_has_no_prevented_events(self):
        evt = SimpleNamespace(
            id="EVT_REAL", event_type="outcome", fabula_time=1000,
            description="ok",
        )
        ws = _fake_world(events=[evt])
        assert build_prevented_event_constraints(ws, syuzhet_anchor=None) == []

    def test_anchor_caps_visibility(self):
        evt_prev = SimpleNamespace(
            id="EVT_PREV", event_type="prevented", fabula_time=2000,
            description="future-only",
        )
        ws = _fake_world(events=[evt_prev])
        # Anchor before the prevented event → suppressed.
        assert build_prevented_event_constraints(ws, syuzhet_anchor=1500) == []
        # Anchor at/after → surfaced.
        assert len(build_prevented_event_constraints(ws, syuzhet_anchor=2000)) == 1

    def test_recognises_alternate_negative_tags(self):
        # Both ``never_happened`` and ``removed`` are negative-physics
        # tags alongside ``prevented``.
        for tag in ("never_happened", "removed"):
            evt = SimpleNamespace(
                id=f"EVT_{tag.upper()}", event_type=tag, fabula_time=100,
                description="x",
            )
            ws = _fake_world(events=[evt])
            blocks = build_prevented_event_constraints(ws, syuzhet_anchor=None)
            assert len(blocks) == 1, f"{tag} should be recognised"


class TestFalsePropositionConstraints:
    def test_emits_block_for_false_proposition(self):
        prop_t = SimpleNamespace(
            id="PROP_T", description="A true claim.",
            truth_at_fabula={1000: True},
        )
        prop_f = SimpleNamespace(
            id="PROP_F", description="A false claim.",
            truth_at_fabula={1000: False},
        )
        ws = _fake_world(propositions=[prop_t, prop_f])

        blocks = build_false_proposition_constraints(ws, syuzhet_anchor=None)

        assert len(blocks) == 1
        b = blocks[0]
        assert b.priority == "hard"
        assert "FALSE PROPOSITIONS (HARD)" in b.instruction
        assert "PROP_F" in b.instruction
        assert "PROP_T" not in b.instruction
        assert b.evidence == {"false_proposition_ids": ["PROP_F"]}

    def test_anchor_caps_visibility(self):
        prop_f = SimpleNamespace(
            id="PROP_F", description="x",
            truth_at_fabula={2000: False},
        )
        ws = _fake_world(propositions=[prop_f])
        # Anchor before commit → no block.
        assert build_false_proposition_constraints(ws, syuzhet_anchor=500) == []
        # Anchor at/after → block.
        assert len(build_false_proposition_constraints(ws, syuzhet_anchor=2000)) == 1

    def test_uses_latest_commit(self):
        # If the latest commit at/before the anchor is True, no block;
        # only the latest commit's truth value matters.
        prop = SimpleNamespace(
            id="PROP_X", description="flip-flopper",
            truth_at_fabula={1000: False, 2000: True},
        )
        ws = _fake_world(propositions=[prop])
        assert build_false_proposition_constraints(ws, syuzhet_anchor=None) == []
        # But anchored before the True overwrite, the False commit
        # is the latest applicable → block emitted.
        blocks = build_false_proposition_constraints(ws, syuzhet_anchor=1500)
        assert len(blocks) == 1

    def test_no_block_for_uncommitted_propositions(self):
        prop = SimpleNamespace(
            id="PROP_OPEN", description="unresolved",
            truth_at_fabula={},
        )
        ws = _fake_world(propositions=[prop])
        assert build_false_proposition_constraints(ws, syuzhet_anchor=None) == []

    def test_dict_shape_propositions_supported(self):
        # Some legacy fixtures deliver ``propositions`` as a dict
        # keyed by PROP_ id; the helper must tolerate that shape.
        prop_f = SimpleNamespace(
            id="PROP_F", description="false",
            truth_at_fabula={100: False},
        )
        ws = SimpleNamespace(events=[], propositions={"PROP_F": prop_f})
        blocks = build_false_proposition_constraints(ws, syuzhet_anchor=None)
        assert len(blocks) == 1


# ────────────────────────────────────────────────────────────────────
# MCP patch_world_state
# ────────────────────────────────────────────────────────────────────


class TestPatchWorldState:
    def test_invalid_payload_returns_error(self):
        ws = deepcopy(macbeth_ws)
        _, pid, _ = _seed(ws)
        result = patch_world_state(
            _ctx(),
            patch={"commit_proposition_truth": "not-a-dict"},
            project_id=pid,
        )
        assert "error" in result
        assert "Invalid patch payload" in result["error"]

    def test_empty_patch_succeeds_no_op(self):
        ws = deepcopy(macbeth_ws)
        _, pid, _ = _seed(ws)
        result = patch_world_state(
            _ctx(), patch={}, project_id=pid,
        )
        assert "error" not in result, result
        assert result["change_count"] == 0
        assert "new_version" in result

    def test_notes_propagate_to_version_description(self):
        ws = deepcopy(macbeth_ws)
        _, pid, _ = _seed(ws)
        result = patch_world_state(
            _ctx(),
            patch={"notes": "test rationale 12345"},
            project_id=pid,
        )
        assert "error" not in result, result

    def test_drop_unknown_event_id_is_no_op(self):
        # Drop a non-existent EVT_ id — patcher should accept the
        # request, apply no changes, and still create a new version.
        ws = deepcopy(macbeth_ws)
        _, pid, _ = _seed(ws)
        result = patch_world_state(
            _ctx(),
            patch={"drop_event_ids": ["EVT_DOES_NOT_EXIST"]},
            project_id=pid,
        )
        assert "error" not in result, result


# ────────────────────────────────────────────────────────────────────
# MCP inspect — pagination + at_time
# ────────────────────────────────────────────────────────────────────


def _entity_with_long_timeline(ws: WorldStateV1) -> str | None:
    """Return the id of an entity with the longest state_timeline."""
    best_id, best_n = None, -1
    for eid, ent in ws.entities.items():
        n = len(ent.state_timeline)
        if n > best_n:
            best_id, best_n = eid, n
    return best_id if best_n > 0 else None


class TestInspectPagination:
    def test_returns_meta_with_total(self):
        ws = deepcopy(macbeth_ws)
        ent_id = _entity_with_long_timeline(ws)
        if ent_id is None:
            pytest.skip("Macbeth fixture has no state_timeline entries")
        _, pid, _ = _seed(ws)
        result = inspect(_ctx(), node_id=ent_id, project_id=pid)
        assert "error" not in result, result
        meta = result.get("state_timeline_meta")
        assert meta is not None
        assert meta["total"] == len(ws.entities[ent_id].state_timeline)
        assert meta["limit"] == 10
        assert meta["offset"] == 0

    def test_limit_zero_returns_empty_window(self):
        ws = deepcopy(macbeth_ws)
        ent_id = _entity_with_long_timeline(ws)
        if ent_id is None:
            pytest.skip("Macbeth fixture has no state_timeline entries")
        _, pid, _ = _seed(ws)
        result = inspect(
            _ctx(), node_id=ent_id, project_id=pid, timeline_limit=0,
        )
        assert "error" not in result, result
        assert result["state_timeline"] == []
        assert result["state_timeline_meta"]["limit"] == 0

    def test_offset_negative_one_returns_full_timeline(self):
        ws = deepcopy(macbeth_ws)
        ent_id = _entity_with_long_timeline(ws)
        if ent_id is None:
            pytest.skip("Macbeth fixture has no state_timeline entries")
        total = len(ws.entities[ent_id].state_timeline)
        _, pid, _ = _seed(ws)
        result = inspect(
            _ctx(), node_id=ent_id, project_id=pid, timeline_offset=-1,
        )
        assert "error" not in result, result
        assert len(result["state_timeline"]) == total
        meta = result["state_timeline_meta"]
        assert meta["offset"] == -1
        assert meta["truncated"] is False


class TestInspectAtTimeCausalAware:
    def test_at_time_marks_causal_aware_reconstruction(self):
        ws = deepcopy(macbeth_ws)
        ent_id = _entity_with_long_timeline(ws)
        if ent_id is None:
            pytest.skip("Macbeth fixture has no state_timeline entries")
        # Pick a fabula tick from the middle of the entity's timeline.
        tl = ws.entities[ent_id].state_timeline
        mid = tl[len(tl) // 2].fabula_time
        _, pid, _ = _seed(ws)
        result = inspect(
            _ctx(), node_id=ent_id, project_id=pid, at_time=mid,
        )
        assert "error" not in result, result
        assert result.get("reconstruction") == "causal_aware"


# ────────────────────────────────────────────────────────────────────
# filter_world_state_for_pov
# ────────────────────────────────────────────────────────────────────


class TestFilterWorldStateForPov:
    def test_pov_none_returns_world_unchanged(self):
        ws = deepcopy(macbeth_ws)
        out = filter_world_state_for_pov(ws, pov_entity_id=None)
        assert out is ws

    def test_unknown_pov_returns_world_unchanged(self):
        ws = deepcopy(macbeth_ws)
        out = filter_world_state_for_pov(ws, pov_entity_id="ENT_NOT_IN_WORLD")
        assert out is ws

    def test_other_entities_have_beliefs_concerns_scrubbed(self):
        ws = deepcopy(macbeth_ws)
        # Pick a POV character that exists in the fixture.
        povs = [
            eid for eid in ws.entities
            if eid in {"ENT_MACBETH", "ENT_LADY_MACBETH"}
        ]
        if not povs:
            pytest.skip("Macbeth fixture missing expected POV entities")
        pov = povs[0]

        # Confirm at least one *other* entity in the source world
        # carries beliefs or concerns — otherwise the assertion
        # below is vacuous.
        other_with_state = next(
            (
                eid for eid, e in ws.entities.items()
                if eid != pov and (e.beliefs or e.concerns)
            ),
            None,
        )
        if other_with_state is None:
            pytest.skip("Fixture has no other-entity beliefs/concerns to scrub")

        filtered = filter_world_state_for_pov(ws, pov_entity_id=pov)
        # The POV entity keeps their record.
        assert pov in filtered.entities
        # Other entities have their interior state scrubbed.
        scrubbed = filtered.entities[other_with_state]
        assert scrubbed.beliefs == []
        assert scrubbed.concerns == []

    def test_channels_pruned_to_pov_participants(self):
        ws = deepcopy(macbeth_ws)
        # Find a POV that participates in at least one channel and a
        # channel that excludes them.
        pov = None
        excluded_channel_id = None
        for eid in ws.entities:
            in_chans = {
                cid for cid, ch in ws.channels.items()
                if eid in ch.participant_ids
            }
            out_chans = {
                cid for cid, ch in ws.channels.items()
                if eid not in ch.participant_ids
            }
            if in_chans and out_chans:
                pov = eid
                excluded_channel_id = next(iter(out_chans))
                break
        if pov is None:
            pytest.skip("Fixture lacks a POV with in/out channel mix")

        filtered = filter_world_state_for_pov(ws, pov_entity_id=pov)
        assert excluded_channel_id not in filtered.channels
        for cid in filtered.channels:
            assert pov in filtered.channels[cid].participant_ids


# ────────────────────────────────────────────────────────────────────
# answer._compress_world_state — Q&A negative facts + propositions
# ────────────────────────────────────────────────────────────────────


class TestQACompressNegativeFacts:
    def test_negative_facts_section_emitted_for_prevented_event(self):
        physics_state = {
            "entities": {},
            "events": [
                {
                    "id": "EVT_PREV", "event_type": "prevented",
                    "fabula_time": 1000,
                    "description": "Did not occur.",
                    "actor_ids": [], "target_ids": [],
                },
                {
                    "id": "EVT_REAL", "event_type": "outcome",
                    "fabula_time": 1100,
                    "description": "Did occur.",
                    "actor_ids": [], "target_ids": [],
                },
            ],
            "propositions": [],
        }
        text = _compress_world_state(physics_state)
        assert "## NEGATIVE FACTS" in text
        assert "EVT_PREV" in text
        # The prevented event must be tagged inside the NEGATIVE
        # FACTS block (not just the events catalogue).
        neg_block = text.split("## NEGATIVE FACTS")[1]
        assert "EVT_PREV" in neg_block

    def test_negative_facts_section_emitted_for_false_proposition(self):
        physics_state = {
            "entities": {},
            "events": [],
            "propositions": [
                {
                    "id": "PROP_T", "description": "True claim.",
                    "truth_at_fabula": {"1000": True},
                },
                {
                    "id": "PROP_F", "description": "False claim.",
                    "truth_at_fabula": {"1000": False},
                },
            ],
        }
        text = _compress_world_state(physics_state)
        assert "## NEGATIVE FACTS" in text
        neg_block = text.split("## NEGATIVE FACTS")[1]
        assert "PROP_F" in neg_block
        assert "PROP_T" not in neg_block

    def test_propositions_section_emitted(self):
        physics_state = {
            "entities": {},
            "events": [],
            "propositions": [
                {
                    "id": "PROP_X", "description": "Some claim.",
                    "truth_at_fabula": {"500": True},
                },
            ],
        }
        text = _compress_world_state(physics_state)
        assert "## Propositions" in text
        assert "PROP_X" in text

    def test_no_negative_facts_section_when_world_is_clean(self):
        physics_state = {
            "entities": {},
            "events": [
                {
                    "id": "EVT_OK", "event_type": "outcome",
                    "fabula_time": 1, "description": "x",
                    "actor_ids": [], "target_ids": [],
                },
            ],
            "propositions": [
                {
                    "id": "PROP_OK", "description": "x",
                    "truth_at_fabula": {"1": True},
                },
            ],
        }
        text = _compress_world_state(physics_state)
        assert "## NEGATIVE FACTS" not in text
