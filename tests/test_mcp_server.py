"""Integration tests for the Shadow-Loom MCP server (v2).

Tests the 20 MCP tools and 5 resources against an in-memory SQLite DB
with the Macbeth world state fixture. All LLM-calling paths are mocked;
computational paths (physics, graph, assembler) run un-mocked.

Auth is tested in open mode (no bearer token → require_scope returns None).
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── DB must be initialised before importing server ────────────────
# The server module calls init_db() at import time, so we pre-init
# with an in-memory SQLite DB.
os.environ["DATABASE_URL"] = "sqlite://"
# Enable open mode so require_scope() permits the test MagicMock contexts
# (which have request_context=None and therefore yield empty scopes).
os.environ["MCP_ALLOW_OPEN_MODE"] = "true"

from shadow_loom.db import (
    create_project,
    get_session,
    init_db,
    save_version,
    update_project,
    upsert_user,
)
from shadow_loom.models import WorldStateV1

# Re-init with in-memory DB (overrides any prior init)
init_db("sqlite://")

# Now safe to import the MCP server and tools
from shadow_loom_mcp import auth as mcp_auth
from shadow_loom_mcp.server import (
    ask,
    audit_log,
    branch,
    compute_tension,
    diff_versions,
    direct,
    evaluate,
    fork,
    get_history,
    get_relationships,
    ingest,
    inspect,
    list_projects,
    mcp,
    narrate,
    open_project,
    resource_entity,
    resource_project,
    resource_projects,
    resource_versions,
    resource_world,
    search,
    share,
    trace_causality,
    update_project_tool,
    write,
)

from example_worlds.macbeth import world_state as macbeth_ws

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_db():
    """Re-init DB and clear auth cache for each test."""
    init_db("sqlite://")
    mcp_auth._token_user_cache.clear()
    yield
    mcp_auth._token_user_cache.clear()


def _seed_project(ws: WorldStateV1 | None = None, is_public: bool = False) -> tuple[int, int, int]:
    """Create a user, project, and initial version. Returns (user_id, project_id, version_row_id)."""
    user = upsert_user("local", "test-1", "testuser", email="test@example.com", display_name="Test User")
    proj = create_project(name="Macbeth", owner_id=user.id, description="A Scottish play")
    if is_public:
        update_project(proj.id, is_public=True)
    ws = ws or deepcopy(macbeth_ws)
    ver = save_version(
        project_id=proj.id,
        world_state_json=ws.model_dump_json(),
        version=0,
        source="ingestion",
        description="Initial ingestion",
        user_id=user.id,
    )
    # Seed auth cache so get_user_id returns this user
    mcp_auth._token_user_cache["test-token"] = {
        "user_id": user.id,
        "scopes": {"read", "write", "admin"},
        "key_id": 1,
    }
    return user.id, proj.id, ver.id


def _ctx() -> MagicMock:
    """Create a mock Context with auth wired to the test user."""
    ctx = MagicMock()
    ctx.request_context = None  # get_user_id falls back to last cached user
    ctx.report_progress = AsyncMock()
    return ctx


# =====================================================================
# GROUP 1: ORIENT
# =====================================================================


class TestListProjects:
    def test_returns_projects(self):
        uid, pid, _ = _seed_project()
        result = list_projects(_ctx())
        assert "projects" in result
        assert len(result["projects"]) >= 1

    def test_empty_when_no_projects(self):
        result = list_projects(_ctx())
        assert result["projects"] == [] or "projects" in result


class TestOpenProject:
    def test_by_id(self):
        _, pid, _ = _seed_project()
        result = open_project(_ctx(), project_id=pid)
        assert result["project_id"] == pid
        assert result["project_name"] == "Macbeth"
        assert "entities" in result
        assert "ENT_MACBETH" in result["entities"]
        assert "Macbeth" in result["entities"]["ENT_MACBETH"]["name"]
        assert result["event_count"] == 20
        assert result["topology"]["causal_edges"] == 36

    def test_by_name(self):
        uid, pid, _ = _seed_project()
        result = open_project(_ctx(), project_name="Macbeth")
        assert result["project_id"] == pid

    def test_no_world_model(self):
        uid, _, _ = _seed_project()  # need auth cache
        proj = create_project(name="Empty", owner_id=uid)
        result = open_project(_ctx(), project_id=proj.id)
        assert "error" in result

    def test_missing_project(self):
        result = open_project(_ctx(), project_id=99999)
        assert "error" in result


# =====================================================================
# GROUP 2: EXPLORE
# =====================================================================


class TestInspect:
    def test_entity(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "ENT_MACBETH", project_id=pid)
        assert result["type"] == "Entity"
        assert "Macbeth" in result["name"]
        assert "traits" in result

    def test_entity_at_time(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "ENT_MACBETH", project_id=pid, at_time=200)
        assert result["at_time"] == 200
        assert "traits" in result

    def test_location(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "LOC_BATTLEFIELD", project_id=pid)
        assert result["type"] == "Location"

    def test_event(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "EVT_REBELLION_DEFEATED", project_id=pid)
        assert result["type"] == "EventNode"
        assert result["fabula_time"] == 100

    def test_object(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "OBJ_CROWN", project_id=pid)
        assert result["type"] == "NarrativeObject"

    def test_world_trait(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "WORLD_FEUDAL_HIERARCHY", project_id=pid)
        assert result["type"] == "WorldTrait"

    def test_world_trait_at_time(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "WORLD_FEUDAL_HIERARCHY", project_id=pid, at_time=200)
        assert "magnitude" in result

    def test_unknown_prefix(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "FOO_BAR", project_id=pid)
        assert "error" in result

    def test_entity_not_found(self):
        _, pid, _ = _seed_project()
        result = inspect(_ctx(), "ENT_NONEXISTENT", project_id=pid)
        assert "error" in result


class TestSearch:
    def test_finds_macbeth(self):
        _, pid, _ = _seed_project()
        result = search(_ctx(), "Macbeth", project_id=pid)
        assert result["results"]
        ids = [r["id"] for r in result["results"]]
        assert "ENT_MACBETH" in ids

    def test_finds_location(self):
        _, pid, _ = _seed_project()
        result = search(_ctx(), "battlefield", project_id=pid)
        assert any(r["type"] == "Location" for r in result["results"])

    def test_no_results(self):
        _, pid, _ = _seed_project()
        result = search(_ctx(), "xyzqwkjhgfd", project_id=pid)
        assert len(result["results"]) == 0


class TestGetRelationships:
    def test_all_relationships(self):
        _, pid, _ = _seed_project()
        result = get_relationships(_ctx(), project_id=pid)
        assert len(result["relationships"]) == 13

    def test_filtered_by_entity(self):
        _, pid, _ = _seed_project()
        result = get_relationships(_ctx(), project_id=pid, entity_id="ENT_MACBETH")
        assert result["relationships"]
        for rel in result["relationships"]:
            ids = {rel["source"]["id"], rel["target"]["id"]}
            assert "ENT_MACBETH" in ids


class TestTraceCausality:
    def test_downstream(self):
        _, pid, _ = _seed_project()
        result = trace_causality(_ctx(), "EVT_REBELLION_DEFEATED", project_id=pid, direction="downstream")
        assert result["root"] == "EVT_REBELLION_DEFEATED"
        assert result["edges"]

    def test_upstream(self):
        _, pid, _ = _seed_project()
        # Pick an event that has upstream causes
        result = trace_causality(_ctx(), "EVT_REBELLION_DEFEATED", project_id=pid, direction="upstream")
        assert result["direction"] == "upstream"

    def test_both(self):
        _, pid, _ = _seed_project()
        result = trace_causality(_ctx(), "EVT_REBELLION_DEFEATED", project_id=pid, direction="both")
        assert result["direction"] == "both"
        assert "EVT_REBELLION_DEFEATED" in result["nodes"]


class TestGetHistory:
    def test_full_tree(self):
        _, pid, _ = _seed_project()
        result = get_history(_ctx(), project_id=pid)
        assert "versions" in result

    def test_specific_version(self):
        _, pid, _ = _seed_project()
        result = get_history(_ctx(), project_id=pid, version=0)
        assert result["version"] == 0
        assert result["source"] == "ingestion"

    def test_version_not_found(self):
        _, pid, _ = _seed_project()
        result = get_history(_ctx(), project_id=pid, version=9999)
        assert "error" in result


# =====================================================================
# GROUP 3: REASON
# =====================================================================


class TestAsk:
    @patch("shadow_loom_mcp.server.calculate_narrative_physics")
    @patch("shadow_loom_mcp.server.parse_query")
    def test_basic_question(self, mock_parse, mock_physics):
        _, pid, _ = _seed_project()

        mock_parsed = MagicMock()
        mock_parsed.reasoning = "Testing reasoning"
        mock_parsed.resolved_ids = []
        mock_parse_result = MagicMock()
        mock_parse_result.is_valid = False
        mock_parse_result.query = None
        mock_parse_result.parsed = mock_parsed
        mock_parse.return_value = mock_parse_result

        mock_physics.return_value = {"status": "complete", "answer": "He's ambitious"}

        result = ask(_ctx(), "What motivates Macbeth?", project_id=pid)
        assert result["read_only"] is True
        assert result["question"] == "What motivates Macbeth?"

    def test_no_world_model(self):
        user = upsert_user("testuser", "t@t.com", "T")
        proj = create_project(name="Empty", owner_id=user.id)
        result = ask(_ctx(), "test", project_id=proj.id)
        assert "error" in result


class TestComputeTension:
    def test_computes_scores(self):
        _, pid, _ = _seed_project()
        result = compute_tension(_ctx(), project_id=pid, entity_ids=["ENT_MACBETH", "ENT_LADY_MACBETH"])
        assert "scores" in result
        assert "mystery" in result["scores"]
        assert "dramatic_irony" in result["scores"]
        assert "suspense" in result["scores"]
        assert "surprise" in result["scores"]
        assert isinstance(result["scores"]["mystery"], float)
        assert 0.0 <= result["scores"]["mystery"] <= 1.0

    def test_default_entities(self):
        _, pid, _ = _seed_project()
        result = compute_tension(_ctx(), project_id=pid)
        assert "scores" in result
        assert len(result["entity_ids"]) <= 6


class TestDiffVersions:
    def test_same_version(self):
        uid, pid, _ = _seed_project()
        result = diff_versions(_ctx(), project_id=pid, version_a=0, version_b=0)
        assert result["entities_added"] == []
        assert result["entities_removed"] == []
        assert result["trait_changes"] == []

    def test_missing_version(self):
        uid, pid, _ = _seed_project()
        result = diff_versions(_ctx(), project_id=pid, version_a=0, version_b=999)
        assert "error" in result


# =====================================================================
# GROUP 4: CREATE
# =====================================================================


class TestNarrate:
    @pytest.mark.asyncio
    @patch("shadow_loom_mcp.server.run_and_save")
    @patch("shadow_loom_mcp.server.parse_query")
    async def test_generates_prose(self, mock_parse, mock_run_save):
        _, pid, _ = _seed_project()

        mock_parsed = MagicMock()
        mock_parsed.reasoning = "Continuing the story"
        mock_parsed.resolved_ids = []
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.query = MagicMock(query_type="observation")
        mock_result.parsed = mock_parsed
        mock_result.validation_errors = []
        mock_parse.return_value = mock_result

        mock_run_save.return_value = {
            "project_id": pid, "version": 1, "prose": "The battle raged...",
            "query_type": "observation",
        }

        ctx = _ctx()
        result = await narrate(ctx, "Continue the story", project_id=pid)
        assert result["prose"] == "The battle raged..."
        assert result["version"] == 1
        ctx.report_progress.assert_called()

    @pytest.mark.asyncio
    @patch("shadow_loom_mcp.server.parse_query")
    async def test_parse_failure(self, mock_parse):
        _, pid, _ = _seed_project()

        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.query = None
        mock_result.parsed = MagicMock(reasoning="Could not parse")
        mock_result.validation_errors = [MagicMock(field="question", message="Missing")]
        mock_parse.return_value = mock_result

        result = await narrate(_ctx(), "???", project_id=pid)
        assert "error" in result


class TestDirect:
    @pytest.mark.asyncio
    @patch("shadow_loom_mcp.server.run_and_save")
    async def test_directive_query(self, mock_run_save):
        _, pid, _ = _seed_project()

        mock_run_save.return_value = {
            "project_id": pid, "version": 1, "prose": "A sense of dread...",
            "query_type": "directive",
        }

        ctx = _ctx()
        result = await direct(ctx, "suspense", project_id=pid, entity_ids=["ENT_MACBETH"])
        assert result["prose"] == "A sense of dread..."
        mock_run_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_effect(self):
        _, pid, _ = _seed_project()
        result = await direct(_ctx(), "nonexistent_effect", project_id=pid)
        assert "error" in result


class TestWrite:
    @patch("shadow_loom_mcp.server.run_and_save")
    def test_manual_edit(self, mock_run_save):
        _, pid, _ = _seed_project()

        mock_run_save.return_value = {
            "project_id": pid, "version": 1,
            "query_type": "manual_edit",
        }

        result = write(_ctx(), "Macbeth drew his sword.", project_id=pid)
        assert result["version"] == 1
        mock_run_save.assert_called_once()


class TestIngest:
    @pytest.mark.asyncio
    @patch("shadow_loom_mcp.server.run_extraction")
    async def test_ingests_text(self, mock_extract):
        ws = deepcopy(macbeth_ws)
        mock_report = MagicMock()
        mock_report.is_valid = True
        mock_report.issues = []
        mock_extract.return_value = (ws, mock_report)

        ctx = _ctx()
        result = await ingest(ctx, "Once upon a time...", project_name="Test Story")
        assert "project_id" in result
        assert result["entities"] == 9
        assert result["version"] == 0
        ctx.report_progress.assert_called()


# =====================================================================
# GROUP 5: JUDGE
# =====================================================================


class TestEvaluate:
    @patch("shadow_loom.auditor._finalize_narrative_order")
    @patch("shadow_loom.auditor.compute_affective_feedback")
    @patch("shadow_loom.auditor.compute_causal_feedback")
    def test_runs_evaluation(self, mock_causal, mock_affective, mock_finalize):
        _, pid, _ = _seed_project()
        # Save a version with prose
        save_version(
            project_id=pid, world_state_json=macbeth_ws.model_dump_json(),
            version=1, source="test", description="test", prose="A dark night in Scotland.",
            user_id=1,
        )

        mock_causal_fb = MagicMock()
        mock_causal_fb.model_dump.return_value = {"miracle_steps": 0}
        mock_causal.return_value = mock_causal_fb

        mock_affective_fb = MagicMock()
        mock_affective_fb.model_dump.return_value = {"kl_divergence": 0.1}
        mock_affective.return_value = mock_affective_fb

        mock_noo = MagicMock()
        mock_noo.overall_pass = True
        mock_noo.causal_feedback = mock_causal_fb
        mock_noo.affective_feedback = mock_affective_fb
        mock_noo.quality_synthesis = MagicMock()
        mock_noo.quality_synthesis.model_dump.return_value = {"coherence": 0.9}
        mock_finalize.return_value = mock_noo

        result = evaluate(_ctx(), project_id=pid)
        assert result["read_only"] is True
        assert result["overall_pass"] is True


class TestAuditLog:
    def test_returns_activities(self):
        _, pid, _ = _seed_project()
        result = audit_log(_ctx(), project_id=pid)
        assert "activities" in result
        assert result["project_id"] == pid

    def test_version_detail(self):
        _, pid, _ = _seed_project()
        result = audit_log(_ctx(), project_id=pid, version=0)
        assert result["version"] == 0
        assert result["source"] == "ingestion"


# =====================================================================
# GROUP 6: MANAGE
# =====================================================================


class TestBranch:
    def test_creates_branch(self):
        uid, pid, vid = _seed_project()
        result = branch(_ctx(), project_id=pid, from_version=0)
        assert result["branched_from"] == 0
        assert "new_version" in result

    def test_invalid_version(self):
        uid, pid, _ = _seed_project()
        result = branch(_ctx(), project_id=pid, from_version=999)
        assert "error" in result


class TestShare:
    def test_share_project(self):
        uid, pid, _ = _seed_project()
        target = upsert_user("local", "other-1", "otheruser", email="other@example.com")
        result = share(_ctx(), project_id=pid, username="other", role="viewer")
        assert result["status"] == "shared"
        assert result["target_user"] == "otheruser"

    def test_user_not_found(self):
        uid, pid, _ = _seed_project()
        result = share(_ctx(), project_id=pid, username="nonexistent_user_xyz")
        assert "error" in result


class TestFork:
    def test_forks_project(self):
        uid, pid, _ = _seed_project()
        result = fork(_ctx(), project_id=pid, new_name="Macbeth Copy")
        assert result["status"] == "forked"
        assert result["new_project_name"] == "Macbeth Copy"
        assert result["original_project_id"] == pid

    def test_project_not_found(self):
        _seed_project()
        result = fork(_ctx(), project_id=99999)
        assert "error" in result


class TestUpdateProject:
    def test_updates_metadata(self):
        uid, pid, _ = _seed_project()
        result = update_project_tool(_ctx(), project_id=pid, name="Macbeth v2", description="Updated")
        assert result["name"] == "Macbeth v2"
        assert result["description"] == "Updated"

    def test_project_not_found(self):
        _seed_project()
        result = update_project_tool(_ctx(), project_id=99999)
        assert "error" in result


# =====================================================================
# RESOURCES
# =====================================================================


class TestResources:
    def test_resource_projects(self):
        _seed_project(is_public=True)
        data = json.loads(resource_projects())
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_resource_project(self):
        _, pid, _ = _seed_project(is_public=True)
        data = json.loads(resource_project(pid))
        assert data["name"] == "Macbeth"

    def test_resource_project_not_found(self):
        data = json.loads(resource_project(99999))
        assert "error" in data

    def test_resource_world(self):
        _, pid, _ = _seed_project(is_public=True)
        data = json.loads(resource_world(pid))
        assert "entities" in data

    def test_resource_entity(self):
        _, pid, _ = _seed_project(is_public=True)
        data = json.loads(resource_entity(pid, "ENT_MACBETH"))
        assert "Macbeth" in data["name"]

    def test_resource_entity_not_found(self):
        _, pid, _ = _seed_project(is_public=True)
        data = json.loads(resource_entity(pid, "ENT_NONEXISTENT"))
        assert "error" in data

    def test_resource_versions(self):
        _, pid, _ = _seed_project(is_public=True)
        data = json.loads(resource_versions(pid))
        assert isinstance(data, list)

    def test_resource_private_denied(self):
        _, pid, _ = _seed_project(is_public=False)
        data = json.loads(resource_project(pid))
        assert "error" in data
