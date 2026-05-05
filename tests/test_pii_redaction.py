# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for PII redaction in observability logging."""
from __future__ import annotations

import os

import pytest

from shadow_loom._agent_logging import redact_pii


class TestRedactPII:
    def test_email_redacted(self):
        assert "[REDACTED_EMAIL]" in redact_pii("Reach me at alice@example.com please")
        assert "alice@example.com" not in redact_pii("Reach me at alice@example.com please")

    def test_sl_token_redacted(self):
        token = "sl_" + "a" * 32
        out = redact_pii(f"token={token} ok")
        assert token not in out
        assert "[REDACTED_SL_TOKEN]" in out

    def test_openai_style_key_redacted(self):
        key = "sk-proj-" + "x" * 24
        out = redact_pii(f"OPENAI_API_KEY={key}")
        assert key not in out
        assert "[REDACTED_API_KEY]" in out

    def test_bearer_token_redacted(self):
        out = redact_pii("Authorization: Bearer abc.def-XYZ_123")
        assert "abc.def-XYZ_123" not in out
        assert "[REDACTED]" in out

    def test_credit_card_redacted(self):
        out = redact_pii("paid with 4111 1111 1111 1111 today")
        assert "4111 1111 1111 1111" not in out
        assert "[REDACTED_PAN]" in out

    def test_narrative_text_untouched(self):
        narrative = (
            "Macbeth, his sword still wet, paused at the threshold and "
            "glanced back at the sleeping king."
        )
        assert redact_pii(narrative) == narrative

    def test_empty_string(self):
        assert redact_pii("") == ""

    def test_disable_via_env(self, monkeypatch):
        monkeypatch.setenv("SHADOW_LOOM_DISABLE_LOG_REDACTION", "1")
        assert "alice@example.com" in redact_pii("hi alice@example.com")

    def test_multiple_patterns_in_one_string(self):
        out = redact_pii("Email alice@example.com Authorization: Bearer xyz.abc-123")
        assert "alice@example.com" not in out
        assert "xyz.abc-123" not in out
        assert "[REDACTED_EMAIL]" in out
