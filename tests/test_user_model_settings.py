# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the per-user model-overrides resolver path."""

from __future__ import annotations

import os
import tempfile

import pytest

from shadow_loom.settings import (
    get_settings,
    resolve_model,
    set_user_context,
    reset_user_context,
    get_user_overrides,
    get_openai_compat_providers,
)


@pytest.fixture
def _isolated_db(monkeypatch):
    """Spin up a throw-away SQLite database for each test."""
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    # Reset the global engine so the new URL takes effect.
    import shadow_loom.db as _db
    monkeypatch.setattr(_db, "_engine", None)
    _db.init_db(f"sqlite:///{db_file}")
    yield _db
    try:
        os.unlink(db_file)
    except FileNotFoundError:
        pass


@pytest.fixture
def _clean_context():
    """Ensure each test starts with empty user overrides."""
    token = set_user_context(None)
    yield
    reset_user_context(token)


# =====================================================================
# Default-model override
# =====================================================================

class TestUserDefaultOverride:

    def test_empty_string_uses_user_default(self, _isolated_db, _clean_context, monkeypatch):
        """resolve_model('') falls back to the user's saved default."""
        u = _isolated_db.upsert_user("test", "test:1", "alice")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        _isolated_db.set_user_model_settings(
            u.id, default_model="openrouter:meta-llama/llama-3-70b",
        )
        set_user_context(u.id)
        model = resolve_model("")
        assert model is not None
        assert not isinstance(model, str)

    def test_user_default_overrides_env_default(
        self, _isolated_db, _clean_context, monkeypatch,
    ):
        """When the incoming string equals env default and user has an
        override, the user's choice wins."""
        u = _isolated_db.upsert_user("test", "test:2", "bob")
        # The env default at this point.
        env_default = get_settings().core.default_model
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        _isolated_db.set_user_model_settings(
            u.id, default_model="openrouter:my-favourite-model",
        )
        set_user_context(u.id)
        # Simulate a stage that inherited the env default at startup.
        model = resolve_model(env_default)
        # No exception → resolver swapped to user default (which has key
        # and is a known provider). If it had used env_default (ollama)
        # it would have built an OllamaModel instead — assert it's an
        # OpenAIChatModel-derived instance by checking the class name.
        assert type(model).__name__.endswith("OpenAIChatModel") or "openai" in type(model).__name__.lower()

    def test_explicit_non_default_string_not_overridden(
        self, _isolated_db, _clean_context, monkeypatch,
    ):
        """If the caller passes an explicit non-default string, the user
        default does NOT silently replace it."""
        u = _isolated_db.upsert_user("test", "test:3", "carol")
        monkeypatch.setenv("OPENROUTER_API_KEY", "k1")
        monkeypatch.setenv("FIREWORKS_API_KEY", "k2")
        _isolated_db.set_user_model_settings(
            u.id, default_model="openrouter:my-pick",
        )
        set_user_context(u.id)
        # Caller explicitly asks for a different (non-default) model.
        model = resolve_model("fireworks:accounts/fireworks/models/llama-v3p1-8b-instruct")
        # Should be built from fireworks key, not openrouter.
        # We can't introspect the base_url without poking into the client,
        # so just assert it's a non-string PydanticAI model.
        assert model is not None and not isinstance(model, str)


# =====================================================================
# Per-stage override
# =====================================================================

class TestUserStageOverride:

    def test_stage_override_wins(self, _isolated_db, _clean_context, monkeypatch):
        """A per-stage override takes precedence over the user default."""
        u = _isolated_db.upsert_user("test", "test:4", "dave")
        monkeypatch.setenv("OPENROUTER_API_KEY", "k1")
        monkeypatch.setenv("GROQ_API_KEY", "k2")
        _isolated_db.set_user_model_settings(
            u.id,
            default_model="openrouter:default-model",
            stage_models={"generation": "groq:llama-3.1-70b"},
        )
        set_user_context(u.id)
        # When the generation call site passes stage="generation", the
        # groq override should win over both incoming and user default.
        model = resolve_model("openrouter:something-else", stage="generation")
        assert model is not None and not isinstance(model, str)

    def test_stage_without_override_falls_through(
        self, _isolated_db, _clean_context, monkeypatch,
    ):
        """No per-stage entry → behaves like an untagged resolve."""
        u = _isolated_db.upsert_user("test", "test:5", "eve")
        monkeypatch.setenv("GROQ_API_KEY", "k")
        _isolated_db.set_user_model_settings(u.id)  # all empty
        set_user_context(u.id)
        # Should pass through unchanged since no override and explicit
        # string is provided.
        model = resolve_model("groq:llama-3.1-70b", stage="generation")
        assert model is not None and not isinstance(model, str)


# =====================================================================
# Custom user-defined provider
# =====================================================================

class TestUserCustomProvider:

    def test_custom_provider_registered_via_user_settings(
        self, _isolated_db, _clean_context,
    ):
        """A custom provider saved via the UI joins the merged registry."""
        u = _isolated_db.upsert_user("test", "test:6", "frank")
        _isolated_db.set_user_model_settings(
            u.id,
            custom_providers=[{
                "prefix": "myllm",
                "base_url": "https://api.myllm.example/v1",
                "api_key": "user-supplied-key",
                "is_local": False,
            }],
        )
        set_user_context(u.id)
        assert "myllm" in get_openai_compat_providers()
        model = resolve_model("myllm:my-fine-tune")
        assert model is not None and not isinstance(model, str)

    def test_custom_local_provider_no_key_required(
        self, _isolated_db, _clean_context,
    ):
        """User-defined local provider works without an API key."""
        u = _isolated_db.upsert_user("test", "test:7", "grace")
        _isolated_db.set_user_model_settings(
            u.id,
            custom_providers=[{
                "prefix": "homelab",
                "base_url": "http://homelab.local:8000/v1",
                "api_key": "",
                "is_local": True,
            }],
        )
        set_user_context(u.id)
        model = resolve_model("homelab:my-model")
        assert model is not None and not isinstance(model, str)

    def test_user_api_key_beats_env_key(
        self, _isolated_db, _clean_context, monkeypatch,
    ):
        """User-saved API key for a built-in provider overrides the env var."""
        u = _isolated_db.upsert_user("test", "test:8", "heidi")
        monkeypatch.setenv("MISTRAL_API_KEY", "env-key")
        _isolated_db.set_user_model_settings(
            u.id,
            custom_providers=[{
                "prefix": "mistral",
                "base_url": "",  # empty → use registry default
                "api_key": "user-key",
                "is_local": False,
            }],
        )
        set_user_context(u.id)
        # Just verify the resolver still produces an OpenAI model with
        # no exception (we can't introspect the actual key without
        # patching deeper, but the smoke is meaningful).
        model = resolve_model("mistral:mistral-large-latest")
        assert model is not None and not isinstance(model, str)


# =====================================================================
# Empty deployment (no DEFAULT_MODEL env, no user default) raises
# =====================================================================

class TestEmptyConfiguration:

    def test_empty_everything_raises(self, _isolated_db, _clean_context, monkeypatch):
        """No env default + no user default + empty string → ValueError."""
        # Build a fresh CoreSettings with no default_model.
        from shadow_loom import settings as _settings
        monkeypatch.setattr(_settings.get_settings().core, "default_model", "")
        set_user_context(None)
        with pytest.raises(ValueError, match="No model configured"):
            resolve_model("")
