# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the OpenRouter provider-routing fallback inside
``_MergedSystemPromptsModel.request``.

When OpenRouter routes a request through a single high-throughput
provider via the ``:nitro`` (or ``:floor`` / ``:online``) suffix and
that provider returns null-body completions, every retry hits the
same dead provider. The retry loop drops the routing suffix once half
the attempt budget has been burned so the bare model id can re-shop
across all of OpenRouter's healthy providers; the original suffix is
restored on exit so subsequent agent calls start fresh on the user's
chosen tier.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from shadow_loom.settings import (
    _MergedSystemPromptsModel,
    _PROVIDER_RETRY_ATTEMPTS,
)


class _FakeBaseModel:
    """Minimal stand-in for ``OpenAIChatModel`` exposing ``_model_name``
    and a ``request`` coroutine driven by a script of side effects.
    """

    def __init__(self, model_name: str, script):
        self._model_name = model_name
        self._calls: list[str] = []
        # Each entry is either an exception to raise or the sentinel
        # ``"OK"`` to return successfully.
        self._script = list(script)

    @property
    def model_name(self):
        return self._model_name

    async def _real_request(self, *args, **kwargs):
        # Records the model_name in effect at the moment of the call.
        self._calls.append(self._model_name)
        if not self._script:
            return "OK"
        nxt = self._script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


class _Harness(_MergedSystemPromptsModel, _FakeBaseModel):
    """Combines the retry mixin with the fake base model so we can
    exercise ``_MergedSystemPromptsModel.request`` end-to-end without
    talking to a real provider."""

    pass


def _malformed_completion_error() -> UnexpectedModelBehavior:
    return UnexpectedModelBehavior(
        "Invalid response from openai chat completions endpoint: "
        "id None, choices None"
    )


@pytest.fixture(autouse=True)
def _patch_openai_chat_model_request():
    """Redirect ``OpenAIChatModel.request`` (called inside the retry
    loop via ``OpenAIChatModel.request(self, ...)``) to our fake's
    ``_real_request`` so the harness doesn't need a live provider."""
    from pydantic_ai.models.openai import OpenAIChatModel

    async def _stub(self, *args, **kwargs):
        return await self._real_request(*args, **kwargs)

    with patch.object(OpenAIChatModel, "request", _stub):
        # Also collapse the retry backoff so the test is fast.
        async def _noop(*_a, **_kw):
            return None
        with patch("asyncio.sleep", new=_noop):
            yield


class TestProviderRoutingFallback:
    def test_drops_nitro_suffix_eagerly_after_first_failure(self):
        """A persistent malformed-completion stream falls back to the
        bare model id on the second attempt onward \u2014 a stuck
        provider rarely recovers within one backoff window, so we
        re-shop across all healthy providers as soon as possible."""
        # Script: every attempt fails. The retry loop should still
        # exhaust _PROVIDER_RETRY_ATTEMPTS and then re-raise.
        script = [_malformed_completion_error()] * _PROVIDER_RETRY_ATTEMPTS
        h = _Harness("qwen/qwen3-30b-a3b:nitro", script)
        with pytest.raises(UnexpectedModelBehavior):
            asyncio.run(h.request())
        # All attempts ran.
        assert len(h._calls) == _PROVIDER_RETRY_ATTEMPTS
        # First attempt stayed on the suffixed model id.
        assert h._calls[0] == "qwen/qwen3-30b-a3b:nitro"
        # Every subsequent attempt hit the bare model id.
        assert all(
            name == "qwen/qwen3-30b-a3b" for name in h._calls[1:]
        )

    def test_restores_original_model_name_on_exit(self):
        script = [_malformed_completion_error()] * _PROVIDER_RETRY_ATTEMPTS
        h = _Harness("qwen/qwen3-30b-a3b:nitro", script)
        with pytest.raises(UnexpectedModelBehavior):
            asyncio.run(h.request())
        # After the call returns, the user's chosen routing tier is back.
        assert h._model_name == "qwen/qwen3-30b-a3b:nitro"

    def test_restores_original_model_name_on_success(self):
        # Fail once on the suffixed id, then succeed on the next attempt.
        script = [_malformed_completion_error(), "OK"]
        h = _Harness("qwen/qwen3-30b-a3b:nitro", script)
        result = asyncio.run(h.request())
        assert result == "OK"
        assert h._model_name == "qwen/qwen3-30b-a3b:nitro"

    def test_floor_suffix_also_dropped(self):
        script = [_malformed_completion_error()] * _PROVIDER_RETRY_ATTEMPTS
        h = _Harness("anthropic/claude-3.5-sonnet:floor", script)
        with pytest.raises(UnexpectedModelBehavior):
            asyncio.run(h.request())
        # Every attempt after the first hit the bare model id.
        assert h._calls[1] == "anthropic/claude-3.5-sonnet"

    def test_no_suffix_means_no_fallback_swap(self):
        """Bare model ids retry as-is — no rewrite on later attempts."""
        script = [_malformed_completion_error()] * _PROVIDER_RETRY_ATTEMPTS
        h = _Harness("qwen/qwen3-30b-a3b", script)
        with pytest.raises(UnexpectedModelBehavior):
            asyncio.run(h.request())
        # Every attempt used the original (already bare) model id.
        assert all(name == "qwen/qwen3-30b-a3b" for name in h._calls)
