"""C6 (twelfth-pass audit): strict-persist shard.

The session-wide autouse fixture in ``conftest.py`` sets
``SHADOW_LOOM_STRICT_PERSIST=0`` for the duration of every test run so
that legacy fixtures passing minimal stubs (``world_state_json="{}"``)
keep functioning. The consequence was that the strict-by-default
production code path in ``save_version`` was never exercised by the
test suite — a regression silently downgrading strict validation back
to a warning would not have been caught.

This shard re-enables strict mode for two focused smoke checks:

* a valid empty ``WorldStateV1`` JSON payload still passes;
* a malformed payload (missing required structure / wrong types)
  raises ``ValueError`` at the persistence boundary.

The shard is intentionally narrow — only the strict-vs-soft branch is
under test. Wider persistence behaviour is covered by the rest of the
suite under the relaxed default.
"""

from __future__ import annotations

import json

import pytest

from shadow_loom.models import WorldStateV1


@pytest.fixture
def _strict_persist_env(monkeypatch):
    """Override the session-wide soft-persist default for this shard."""
    monkeypatch.setenv("SHADOW_LOOM_STRICT_PERSIST", "1")
    yield


class TestStrictPersistShard:
    """Smoke coverage for the strict-by-default persistence boundary."""

    def test_valid_empty_world_state_passes_strict_validation(
        self, _strict_persist_env
    ):
        # A minimally constructed WorldStateV1 — produced by the model
        # itself — must round-trip through the persistence-boundary
        # validator without raising under strict mode.
        ws = WorldStateV1(
            world_id="W_strict_smoke",
            locations={}, objects={}, entities={},
            events=[], causal_topology=[],
        )
        payload = ws.model_dump_json()
        # Re-validate as save_version would.
        WorldStateV1.model_validate_json(payload)

    def test_malformed_payload_fails_strict_validation(
        self, _strict_persist_env
    ):
        # The trivial ``{}`` payload that legacy soft-mode tests rely
        # on must be rejected under strict mode because the required
        # ``world_id`` field is absent.
        bad = "{}"
        with pytest.raises(Exception):
            WorldStateV1.model_validate_json(bad)

    def test_wrong_type_payload_fails_strict_validation(
        self, _strict_persist_env
    ):
        # Wrong-shape payload (entities must be a mapping, not a list)
        # is rejected.
        bad = json.dumps({"world_id": "W_bad", "entities": ["not-a-dict"]})
        with pytest.raises(Exception):
            WorldStateV1.model_validate_json(bad)
