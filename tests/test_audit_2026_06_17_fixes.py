# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Behavioural coverage for the 2026-06-17 audit round.

Covers two newly-fixed engine bugs and four previously source-only
verified areas (toggle_star, MCP idempotency TTL, SSRF base_url guard,
fork_project orphan rollback) with real behavioural assertions rather
than ``inspect.getsource`` substring checks.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from shadow_loom.models import (
    Channel,
    Entity,
    EventNode,
    Location,
    Proposition,
    WorldStateV1,
)


# ---------------------------------------------------------------------------
# Shared fixtures / builders
# ---------------------------------------------------------------------------

@pytest.fixture
def _isolated_db(monkeypatch):
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    import shadow_loom.db as _db
    monkeypatch.setattr(_db, "_engine", None)
    _db.init_db(f"sqlite:///{db_file}")
    yield _db
    try:
        os.unlink(db_file)
    except FileNotFoundError:
        pass


def _empty_world_state() -> WorldStateV1:
    return WorldStateV1(
        locations={"LOC_X": Location(id="LOC_X", name="X", description="x")},
        objects={},
        entities={
            "ENT_A": Entity(
                id="ENT_A", name="A", location_id="LOC_X",
                status="healthy", traits={},
            ),
        },
        events=[
            EventNode(
                id="EVT_1", fabula_time=0, syuzhet_index=0,
                event_type="outcome", actor_ids=["ENT_A"], target_ids=[],
                description="x",
            ),
        ],
        causal_topology=[],
    )


# ---------------------------------------------------------------------------
# 1. projections.trace_information_flow — "both" must not drop upstream nodes
# ---------------------------------------------------------------------------

def _chain_world() -> WorldStateV1:
    """ALICE --utterance--> BOB --utterance--> CAROL, all on one channel."""
    base = "Home"
    return WorldStateV1(
        locations={"LOC_H": Location(id="LOC_H", name=base, description=base)},
        objects={},
        entities={
            e: Entity(id=e, name=e, location_id="LOC_H", status="healthy", traits={})
            for e in ("ENT_ALICE", "ENT_BOB", "ENT_CAROL")
        },
        events=[
            EventNode(
                id="EVT_AB", fabula_time=1, syuzhet_index=1,
                event_type="utterance", actor_ids=["ENT_ALICE"], target_ids=[],
                speaker_id="ENT_ALICE", addressee_ids=["ENT_BOB"],
                via_channel_id="CHN", content="secret", truth_value="true",
                description="Alice tells Bob",
            ),
            EventNode(
                id="EVT_BC", fabula_time=2, syuzhet_index=2,
                event_type="utterance", actor_ids=["ENT_BOB"], target_ids=[],
                speaker_id="ENT_BOB", addressee_ids=["ENT_CAROL"],
                via_channel_id="CHN", content="secret", truth_value="true",
                description="Bob tells Carol",
            ),
        ],
        causal_topology=[],
        channels={
            "CHN": Channel(
                id="CHN", name="line", medium="telephone",
                participant_ids=["ENT_ALICE", "ENT_BOB", "ENT_CAROL"],
                directionality="duplex",
                intelligibility={"ENT_ALICE": 1.0, "ENT_BOB": 1.0, "ENT_CAROL": 1.0},
                established_at_fabula=0, evidence_strength="strong",
            ),
        },
    )


def test_trace_information_flow_downstream_actually_traverses():
    # Regression: the downstream pass used to be a silent no-op (the root
    # was pre-added before the recursive walk, tripping its visited
    # guard), so a downstream trace returned only the root node.
    from shadow_loom.projections import trace_information_flow

    result = trace_information_flow(_chain_world(), "ENT_BOB", direction="downstream", depth=4)
    node_set = set(result["nodes"])
    assert node_set != {"ENT_BOB"}, "downstream walk produced only the root"
    # Bob's outgoing utterance and its recipient must be reached.
    assert "EVT_BC" in node_set
    assert "ENT_CAROL" in node_set


def test_trace_information_flow_both_unions_upstream_and_downstream():
    # "both" must contain an upstream-only node (Bob's *incoming*
    # utterance EVT_AB) AND a downstream-only node (Carol) — the previous
    # code dropped the upstream side entirely.
    from shadow_loom.projections import trace_information_flow

    result = trace_information_flow(_chain_world(), "ENT_BOB", direction="both", depth=4)
    node_set = set(result["nodes"])
    assert "EVT_AB" in node_set, "upstream-only node lost in 'both'"
    assert "ENT_CAROL" in node_set, "downstream-only node lost in 'both'"


# ---------------------------------------------------------------------------
# 2. narrative_physics.find_pod — non-integer truth key must not crash
# ---------------------------------------------------------------------------

def test_find_pod_tolerates_noninteger_truth_key():
    from shadow_loom.narrative_physics import find_pod
    from shadow_loom.query_models import DoProposition

    ws = _empty_world_state()
    prop = Proposition(
        proposition_id="PROP_X", kind="outcome", description="x is so",
        referent_ids=["ENT_A"], truth_at_fabula={0: True},
    )
    # Inject a malformed key directly into the dict, bypassing the
    # field validator that would normally coerce string keys at the
    # construction boundary (simulating legacy/corrupted persisted data).
    prop.truth_at_fabula["bogus"] = True
    ws.propositions = [prop]

    pod = find_pod(DoProposition(proposition_id="PROP_X", truth=False), ws)
    assert pod.target_id == "PROP_X"
    assert pod.target_kind == "proposition"


# ---------------------------------------------------------------------------
# 3. db.toggle_star — behavioural roundtrip + counter consistency
# ---------------------------------------------------------------------------

def test_toggle_star_roundtrip_and_count(_isolated_db):
    db = _isolated_db
    u1 = db.upsert_user("local", "star:1", "u1").id
    u2 = db.upsert_user("local", "star:2", "u2").id
    proj = db.create_project("starred", owner_id=u1)

    assert db.toggle_star(proj.id, u1) is True
    assert db.is_starred(proj.id, u1) is True
    assert db.get_project(proj.id).star_count == 1

    # Second distinct user stars -> count 2.
    assert db.toggle_star(proj.id, u2) is True
    assert db.get_project(proj.id).star_count == 2

    # Unstar u1 -> count back to 1, is_starred False.
    assert db.toggle_star(proj.id, u1) is False
    assert db.is_starred(proj.id, u1) is False
    assert db.get_project(proj.id).star_count == 1


# ---------------------------------------------------------------------------
# 4. db MCP idempotency — save/get roundtrip, TTL expiry, purge
# ---------------------------------------------------------------------------

def test_mcp_idempotency_save_get_roundtrip(_isolated_db):
    db = _isolated_db
    u = db.upsert_user("local", "idem:1", "u").id
    proj = db.create_project("p", owner_id=u)
    payload = {"ok": True, "value": 42}

    db.save_mcp_idempotent_response(proj.id, None, "key-1", payload)
    got = db.get_mcp_idempotent_response(proj.id, None, "key-1")
    assert got == payload
    # Unknown key is a miss.
    assert db.get_mcp_idempotent_response(proj.id, None, "key-unknown") is None


def _age_idempotency_row(db, days: int):
    from sqlmodel import select
    from shadow_loom.db import McpIdempotencyRow, get_session
    with get_session() as s:
        row = s.exec(select(McpIdempotencyRow)).first()
        row.created_at = datetime.now(timezone.utc) - timedelta(days=days)
        s.add(row)
        s.commit()


def test_mcp_idempotency_ttl_expiry_is_a_miss(_isolated_db, monkeypatch):
    db = _isolated_db
    monkeypatch.setenv("SHADOW_LOOM_MCP_IDEMPOTENCY_TTL_DAYS", "30")
    u = db.upsert_user("local", "idem:2", "u").id
    proj = db.create_project("p", owner_id=u)
    db.save_mcp_idempotent_response(proj.id, None, "key-2", {"x": 1})

    _age_idempotency_row(db, days=31)
    # Older than the 30-day TTL -> read as a miss.
    assert db.get_mcp_idempotent_response(proj.id, None, "key-2") is None


def test_purge_mcp_idempotency_removes_stale(_isolated_db, monkeypatch):
    db = _isolated_db
    monkeypatch.setenv("SHADOW_LOOM_MCP_IDEMPOTENCY_TTL_DAYS", "30")
    u = db.upsert_user("local", "idem:3", "u").id
    proj = db.create_project("p", owner_id=u)
    db.save_mcp_idempotent_response(proj.id, None, "key-3", {"x": 1})

    assert db.purge_mcp_idempotency() == 0  # fresh row survives
    _age_idempotency_row(db, days=99)
    assert db.purge_mcp_idempotency() == 1  # stale row removed
    assert db.purge_mcp_idempotency() == 0  # nothing left


# ---------------------------------------------------------------------------
# 5. settings._assert_safe_base_url — SSRF guard (hosted vs single-user)
# ---------------------------------------------------------------------------

def _force_hosted(monkeypatch, hosted: bool):
    import shadow_loom.settings as s
    fake = SimpleNamespace(oauth=SimpleNamespace(auth_required=hosted))
    monkeypatch.setattr(s, "get_settings", lambda: fake)
    return s


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://127.0.0.1:11434/v1",                  # loopback
    "http://10.0.0.5/v1",                          # private 10/8
    "http://192.168.1.10/v1",                      # private 192.168
])
def test_assert_safe_base_url_blocks_internal_in_hosted(monkeypatch, url):
    s = _force_hosted(monkeypatch, True)
    with pytest.raises(ValueError):
        s._assert_safe_base_url(url)


def test_assert_safe_base_url_allows_public_in_hosted(monkeypatch):
    s = _force_hosted(monkeypatch, True)
    s._assert_safe_base_url("https://openrouter.ai/api/v1")  # no raise


def test_assert_safe_base_url_allows_internal_in_single_user(monkeypatch):
    s = _force_hosted(monkeypatch, False)
    # Single-user mode trusts local endpoints (llama.cpp / Ollama).
    s._assert_safe_base_url("http://127.0.0.1:11434/v1")  # no raise


# ---------------------------------------------------------------------------
# 6. db.fork_project — invalid source payload leaves no orphan project
# ---------------------------------------------------------------------------

def test_fork_project_valid(_isolated_db):
    db = _isolated_db
    u = db.upsert_user("local", "fork:1", "u").id
    src = db.create_project("src", owner_id=u)
    db.save_version(src.id, _empty_world_state().model_dump_json(),
                    version=0, source="pipeline", user_id=u)
    forked = db.fork_project(src.id, u, new_name="dst", actor_id=u)
    assert forked is not None
    assert forked.name == "dst"


def test_fork_project_invalid_payload_leaves_no_orphan(_isolated_db, monkeypatch):
    db = _isolated_db
    # The suite-wide conftest softens persistence (STRICT_PERSIST=0); the
    # orphan-prevention guard only raises in strict mode, so re-enable it.
    monkeypatch.setenv("SHADOW_LOOM_STRICT_PERSIST", "1")
    u = db.upsert_user("local", "fork:2", "u").id
    bad = db.create_project("badsrc", owner_id=u)
    # Persist a deliberately invalid payload (accept_partial bypasses the
    # save-time guard); the fork must re-validate and refuse.
    db.save_version(bad.id, "{not valid json", version=0, source="pipeline",
                    user_id=u, accept_partial=True)

    before = {p["name"] for p in db.list_projects()}
    with pytest.raises(ValueError):
        db.fork_project(bad.id, u, new_name="should_not_persist", actor_id=u)
    after = {p["name"] for p in db.list_projects()}
    assert "should_not_persist" not in after
    assert after == before
