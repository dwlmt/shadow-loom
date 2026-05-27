# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for round-13 design fixes.

See ``/memories/repo/round-13-implementation.md`` for the full plan.
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite://")

import json
import threading
from pathlib import Path

import pytest


# ---------------------------------------------------------------------
# R13-01: session registry lock prevents duplicate AppState creation
# under concurrent ``_get_session_state`` calls. We exercise the lock
# at the helper level since spinning up a real NiceGUI client is
# heavy — the contract is that the registry-mutating code path is
# serialised by ``_SESSION_REGISTRY_LOCK``.
# ---------------------------------------------------------------------

def test_r13_01_session_registry_lock_exists():
    import shadow_loom_ui.app as app_mod
    lock = getattr(app_mod, "_SESSION_REGISTRY_LOCK")
    # Re-entrant so a single thread can call eviction from inside the
    # create path without deadlocking.
    assert lock is not None
    # threading.RLock is a factory function; the returned object
    # exposes acquire/release.
    assert hasattr(lock, "acquire") and hasattr(lock, "release")


def test_r13_01_eviction_under_lock():
    """Under the lock, concurrent evictors must not double-teardown
    the same AppState. We simulate by injecting two stale entries and
    spinning two evictor threads; teardown must be called exactly
    once per session id."""
    import shadow_loom_ui.app as app_mod

    teardown_calls: dict[str, int] = {}

    class _FakeState:
        def __init__(self, sid):
            self.sid = sid
        def teardown(self, *a, **kw):
            teardown_calls[self.sid] = teardown_calls.get(self.sid, 0) + 1

    app_mod._SESSION_STATES.clear()
    app_mod._SESSION_LAST_TOUCH.clear()
    for i in range(20):
        sid = f"stale-{i}"
        app_mod._SESSION_STATES[sid] = _FakeState(sid)
        # Force "ages ago" so the eviction path picks them up.
        app_mod._SESSION_LAST_TOUCH[sid] = 0.0

    barrier = threading.Barrier(4)

    def _evict():
        barrier.wait()
        # ``now`` far in the future so the TTL check fires.
        app_mod._evict_idle_sessions(1e12)

    threads = [threading.Thread(target=_evict) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every fake state torn down exactly once.
    for i in range(20):
        sid = f"stale-{i}"
        assert teardown_calls.get(sid, 0) == 1, (sid, teardown_calls.get(sid))
    assert app_mod._SESSION_STATES == {}


# ---------------------------------------------------------------------
# R13-02 / R13-03: hot-path indexes declared on the SQLModel tables.
# ---------------------------------------------------------------------

def test_r13_02_03_user_lookup_indexes_declared():
    from shadow_loom.db import (
        ProjectRow, ProjectMemberRow, ProjectStarRow, ApiKeyRow,
    )

    def _index_columns(model_cls):
        cols: set[str] = set()
        for idx in model_cls.__table__.indexes:
            cols.update(c.name for c in idx.columns)
        # ``index=True`` on a column also produces a single-col index
        # that lives on the column itself.
        for col in model_cls.__table__.columns:
            if col.index:
                cols.add(col.name)
        return cols

    assert "owner_id" in _index_columns(ProjectRow)
    assert "user_id" in _index_columns(ProjectMemberRow)
    assert "user_id" in _index_columns(ProjectStarRow)
    assert "user_id" in _index_columns(ApiKeyRow)


# ---------------------------------------------------------------------
# R13-04: Fernet fail-closed when SHADOW_LOOM_SECRET_KEY is invalid.
# ---------------------------------------------------------------------

def test_r13_04_invalid_secret_key_raises():
    import shadow_loom.db as dbmod

    dbmod.reset_fernet_cache()
    os.environ["SHADOW_LOOM_SECRET_KEY"] = "this-is-not-a-valid-fernet-key"
    try:
        with pytest.raises(dbmod.InvalidSecretKeyError):
            dbmod._get_fernet()
        # Encryption path must also fail closed rather than returning
        # plaintext.
        with pytest.raises(dbmod.InvalidSecretKeyError):
            dbmod._encrypt_api_key("sk-secret")
    finally:
        os.environ.pop("SHADOW_LOOM_SECRET_KEY", None)
        dbmod.reset_fernet_cache()


def test_r13_04_unset_secret_still_returns_none():
    """Unset env is the documented "plaintext dev" mode; must keep
    working (warning only)."""
    import shadow_loom.db as dbmod

    dbmod.reset_fernet_cache()
    os.environ.pop("SHADOW_LOOM_SECRET_KEY", None)
    assert dbmod._get_fernet() is None
    # And encryption short-circuits to plaintext rather than raising.
    assert dbmod._encrypt_api_key("sk-dev") == "sk-dev"


# ---------------------------------------------------------------------
# R13-05: rotating SHADOW_LOOM_SECRET_KEY at runtime is picked up
# by the next ``_get_fernet`` without manual cache reset.
# ---------------------------------------------------------------------

def test_r13_05_fernet_cache_invalidates_on_env_change():
    import shadow_loom.db as dbmod
    from cryptography.fernet import Fernet

    dbmod.reset_fernet_cache()
    key_a = Fernet.generate_key().decode("ascii")
    key_b = Fernet.generate_key().decode("ascii")
    assert key_a != key_b

    os.environ["SHADOW_LOOM_SECRET_KEY"] = key_a
    f_a = dbmod._get_fernet()
    assert f_a is not None

    os.environ["SHADOW_LOOM_SECRET_KEY"] = key_b
    f_b = dbmod._get_fernet()
    assert f_b is not None
    assert f_a is not f_b  # new instance built from new key

    os.environ.pop("SHADOW_LOOM_SECRET_KEY", None)
    dbmod.reset_fernet_cache()


def test_r13_05_reset_fernet_cache_helper_exists():
    import shadow_loom.db as dbmod
    assert callable(dbmod.reset_fernet_cache)


# ---------------------------------------------------------------------
# R13-06: settings cache reset hook
# ---------------------------------------------------------------------

def test_r13_06_reset_settings_cache_clears_lru():
    from shadow_loom.settings import get_settings, reset_settings_cache

    a = get_settings()
    b = get_settings()
    assert a is b  # cached

    reset_settings_cache()
    c = get_settings()
    # Fresh object after reset.
    assert c is not a


# ---------------------------------------------------------------------
# R13-07: atomic checkpoint writes
# ---------------------------------------------------------------------

def test_r13_07_atomic_write_replaces_existing(tmp_path: Path):
    from shadow_loom.ingestion import _atomic_write_json

    target = tmp_path / "ckpt.json"
    target.write_text("OLD", encoding="utf-8")

    _atomic_write_json(target, json.dumps({"x": 1}))
    assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1}


def test_r13_07_atomic_write_no_partial_on_crash(tmp_path: Path, monkeypatch):
    """If the write step raises after the temp file is created, the
    target path must NOT contain a half-written body. We force a
    crash by monkeypatching os.replace and confirm the original
    contents survive intact."""
    import shadow_loom.ingestion as ingmod

    target = tmp_path / "ckpt.json"
    target.write_text('{"orig": true}', encoding="utf-8")

    def _boom(*a, **kw):
        raise OSError("simulated rename failure")

    monkeypatch.setattr("os.replace", _boom)

    with pytest.raises(OSError):
        ingmod._atomic_write_json(target, '{"new": true}')

    # Original file untouched; no half-written content.
    assert target.read_text(encoding="utf-8") == '{"orig": true}'
    # And the temp sibling was cleaned up.
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_r13_07_atomic_write_concurrent_no_corruption(tmp_path: Path):
    """N threads writing distinct payloads to the same target — the
    final file must be one of the payloads, never a truncated mix."""
    from shadow_loom.ingestion import _atomic_write_json

    target = tmp_path / "ckpt.json"
    payloads = [json.dumps({"writer": i, "data": "x" * 1024}) for i in range(16)]
    barrier = threading.Barrier(len(payloads))

    def _writer(body: str):
        barrier.wait()
        _atomic_write_json(target, body)

    threads = [threading.Thread(target=_writer, args=(p,)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = json.loads(target.read_text(encoding="utf-8"))
    assert final in [json.loads(p) for p in payloads]
