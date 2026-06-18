# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Behavioural coverage for the REST adapter (``shadow_loom_rest``).

The REST layer reuses the MCP tool bodies verbatim by passing a
:class:`shadow_loom_mcp.auth.Principal` into the ``ctx`` slot. These
tests assert the transport plumbing (catalogue parity, auth, error→HTTP
mapping, argument validation) and the ``Principal`` seam in
``shadow_loom_mcp.auth`` — not the pipeline tools, which need an LLM.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", url)
    # Production fail-closed: no anonymous open-mode fallback.
    monkeypatch.delenv("MCP_ALLOW_OPEN_MODE", raising=False)
    import shadow_loom.db as _db
    monkeypatch.setattr(_db, "_engine", None)
    _db.init_db(url)

    from shadow_loom_rest.app import create_app
    with TestClient(create_app()) as c:
        c._db = _db  # type: ignore[attr-defined]
        yield c
    try:
        os.unlink(db_file)
    except FileNotFoundError:
        pass


def _user_with_key(db, scopes="read,write,admin", suffix="1"):
    u = db.upsert_user("local", f"rest:{suffix}", f"u{suffix}")
    _row, raw = db.create_api_key(u.id, f"key-{suffix}", scopes=scopes)
    return u, {"Authorization": f"Bearer {raw}"}


# ── Catalogue / meta ────────────────────────────────────────────────

def test_healthz_reports_tool_count(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["tools"] == 42


def test_tools_catalogue_mirrors_mcp(client):
    import asyncio
    from shadow_loom_mcp import server

    rest_names = {t["name"] for t in client.get("/tools").json()["tools"]}
    mcp_names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert rest_names == mcp_names
    assert len(rest_names) == 42


def test_tool_schema_endpoint(client):
    schema = client.get("/tools/open_project").json()
    assert schema["name"] == "open_project"
    props = schema["parameters"]["properties"]
    # ctx is injected by the transport, never exposed as an argument.
    assert "ctx" not in props
    assert {"project_id", "project_name", "version"} <= set(props)


def test_unknown_tool_schema_404(client):
    assert client.get("/tools/does_not_exist").status_code == 404


# ── Auth ─────────────────────────────────────────────────────────────

def test_missing_token_is_401(client):
    r = client.post("/tools/list_projects", json={})
    assert r.status_code == 401


def test_invalid_token_is_401(client):
    r = client.post(
        "/tools/list_projects", json={}, headers={"Authorization": "Bearer sl_nope"}
    )
    assert r.status_code == 401


def test_valid_key_dispatches_to_tool_body(client):
    db = client._db
    u, headers = _user_with_key(db)
    db.create_project("RoundTrip", owner_id=u.id)
    r = client.post("/tools/list_projects", json={}, headers=headers)
    assert r.status_code == 200
    assert "RoundTrip" in {p["name"] for p in r.json()["projects"]}


def test_open_mode_allows_anonymous(client, monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_OPEN_MODE", "true")
    r = client.post("/tools/list_projects", json={})
    assert r.status_code == 200
    assert "projects" in r.json()


# ── Scope / access mapping ───────────────────────────────────────────

def test_scopeless_key_is_403(client):
    db = client._db
    _u, headers = _user_with_key(db, scopes="", suffix="noscope")
    r = client.post("/tools/list_projects", json={}, headers=headers)
    assert r.status_code == 403
    assert "error" in r.json()


def test_missing_world_model_maps_to_404(client):
    db = client._db
    u, headers = _user_with_key(db, suffix="empty")
    proj = db.create_project("NoVersions", owner_id=u.id)
    r = client.post("/tools/open_project", json={"project_id": proj.id}, headers=headers)
    assert r.status_code == 404
    assert "error" in r.json()


# ── Argument validation ──────────────────────────────────────────────

def test_unknown_argument_is_422(client):
    db = client._db
    _u, headers = _user_with_key(db, suffix="badarg")
    r = client.post("/tools/list_projects", json={"bogus": 1}, headers=headers)
    assert r.status_code == 422


def test_missing_required_argument_is_422(client):
    db = client._db
    _u, headers = _user_with_key(db, suffix="reqarg")
    # diff_versions requires project_id, version_a, version_b.
    r = client.post("/tools/diff_versions", json={"project_id": 1}, headers=headers)
    assert r.status_code == 422


def test_unknown_tool_invoke_404(client):
    db = client._db
    _u, headers = _user_with_key(db, suffix="404")
    r = client.post("/tools/no_such_tool", json={}, headers=headers)
    assert r.status_code == 404


# ── Principal seam (unit) ────────────────────────────────────────────

def test_resolve_principal_roundtrip(client):
    db = client._db
    from shadow_loom_mcp.auth import Principal, get_scopes, get_user_id, resolve_principal

    u = db.upsert_user("local", "rest:principal", "pp")
    _row, raw = db.create_api_key(u.id, "pk", scopes="read,write")
    p = resolve_principal(raw)
    assert isinstance(p, Principal)
    assert p.user_id == u.id
    assert p.scopes == {"read", "write"}
    # The auth chokepoints short-circuit on a Principal.
    assert get_user_id(p) == u.id
    assert get_scopes(p) == {"read", "write"}


def test_resolve_principal_rejects_bad_token(client):
    from shadow_loom_mcp.auth import resolve_principal

    assert resolve_principal("sl_not_a_real_key") is None
    assert resolve_principal(None) is None
    assert resolve_principal("") is None


def test_same_api_key_authenticates_mcp_and_rest(client):
    # One key per user, one ApiKeyRow table: the key minted for MCP use
    # resolves identically over REST. This is the shared-credential
    # guarantee — no separate REST key issuance.
    db = client._db
    from shadow_loom_mcp.auth import _validate_bearer_token, resolve_principal

    u = db.upsert_user("local", "rest:shared", "shared")
    db.create_project("SharedKeyProj", owner_id=u.id)
    _row, raw = db.create_api_key(u.id, "shared-key", scopes="read,write")

    # MCP transport validates the bearer token via this same function.
    assert _validate_bearer_token(raw) is True
    # REST transport resolves the very same token to the same identity.
    principal = resolve_principal(raw)
    assert principal is not None and principal.user_id == u.id

    # And it actually drives a REST call end-to-end.
    r = client.post(
        "/tools/list_projects", json={}, headers={"Authorization": f"Bearer {raw}"}
    )
    assert r.status_code == 200
    assert "SharedKeyProj" in {p["name"] for p in r.json()["projects"]}
