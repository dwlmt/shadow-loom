# SPDX-FileCopyrightText: 2025-2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared helper for logging LLM-Agent outputs at INFO level.

Every ``Agent.run_sync(...)`` call site in shadow_loom should call
:func:`log_agent_output` immediately after with the calling module's
logger so that the structured output of every agent is captured in
the standard log stream — useful for debugging the pipeline,
reproducing audit/refinement loops, and offline analysis.

The helper handles three formatting cases:
  * pydantic ``BaseModel`` → ``.model_dump_json()``
  * plain ``dict``/``list``/scalars → ``json.dumps`` with a fallback
  * everything else → ``repr()``

Output is truncated to ``max_chars`` (default 4000) to keep log
files manageable when an agent returns a very long structured
payload.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from pydantic_ai import Agent
from pydantic import BaseModel

from shadow_loom.settings import get_settings

_DEFAULT_MAX_CHARS = 4000
_LANGFUSE_LOGGER = logging.getLogger(__name__)
_LANGFUSE_CLIENT: Any | None = None
_LANGFUSE_DISABLED = False
_INSTRUMENTED = False


def _serialise(output: Any) -> str:
    if isinstance(output, BaseModel):
        try:
            return output.model_dump_json()
        except Exception:  # pragma: no cover — defensive
            return repr(output)
    try:
        return json.dumps(output, default=str, ensure_ascii=False)
    except Exception:
        return repr(output)


def _safe_prompt_preview(prompt: Any, max_chars: int = 1000) -> str:
    try:
        text = str(prompt)
    except Exception:
        return "<unserializable prompt>"
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + f"... [+{len(text) - max_chars} chars]"
    return text


def _get_langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT, _LANGFUSE_DISABLED
    if _LANGFUSE_DISABLED:
        return None
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    core = get_settings().core
    if not core.langfuse_public_key or not core.langfuse_secret_key:
        _LANGFUSE_DISABLED = True
        return None

    try:
        from langfuse import Langfuse

        kwargs: dict[str, Any] = {
            "public_key": core.langfuse_public_key,
            "secret_key": core.langfuse_secret_key,
        }
        if core.langfuse_base_url:
            kwargs["host"] = core.langfuse_base_url
        _LANGFUSE_CLIENT = Langfuse(**kwargs)
        return _LANGFUSE_CLIENT
    except Exception as exc:
        _LANGFUSE_DISABLED = True
        _LANGFUSE_LOGGER.warning(
            "[Langfuse] Failed to initialize client: %s", exc,
        )
        return None


def _log_to_langfuse(
    agent_name: str,
    *,
    prompt: Any = None,
    output: Any = None,
    error: Exception | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    client = _get_langfuse_client()
    if client is None:
        return

    payload: dict[str, Any] = {
        "name": f"shadow-loom.{agent_name}",
        "input": _safe_prompt_preview(prompt) if prompt is not None else None,
        "output": _serialise(output) if output is not None else None,
        "metadata": metadata or {},
    }
    if error is not None:
        payload["level"] = "ERROR"
        payload["status_message"] = repr(error)

    try:
        trace_fn = getattr(client, "trace", None)
        if callable(trace_fn):
            trace = trace_fn(**payload)
            flush_fn = getattr(client, "flush", None)
            if callable(flush_fn):
                flush_fn()
            return

        event_fn = getattr(client, "event", None)
        if callable(event_fn):
            event_fn(**payload)
            flush_fn = getattr(client, "flush", None)
            if callable(flush_fn):
                flush_fn()
    except Exception as exc:
        _LANGFUSE_LOGGER.warning("[Langfuse] Failed to emit trace: %s", exc)


def _agent_name(agent: Any) -> str:
    name = getattr(agent, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    output_type = getattr(agent, "output_type", None)
    if output_type is not None:
        type_name = getattr(output_type, "__name__", None)
        if isinstance(type_name, str) and type_name.strip():
            return type_name
    return agent.__class__.__name__


def configure_agent_instrumentation() -> None:
    """Patch PydanticAI Agent methods once to emit Langfuse traces."""
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return

    original_run_sync = Agent.run_sync
    original_run = Agent.run

    def instrumented_run_sync(self: Agent, *args: Any, **kwargs: Any):
        agent_name = _agent_name(self)
        prompt = args[0] if args else kwargs.get("user_prompt")
        try:
            result = original_run_sync(self, *args, **kwargs)
            _log_to_langfuse(
                agent_name,
                prompt=prompt,
                output=getattr(result, "output", None),
                metadata={"call": "run_sync"},
            )
            return result
        except Exception as exc:
            _log_to_langfuse(
                agent_name,
                prompt=prompt,
                error=exc,
                metadata={"call": "run_sync"},
            )
            raise

    async def instrumented_run(self: Agent, *args: Any, **kwargs: Any):
        agent_name = _agent_name(self)
        prompt = args[0] if args else kwargs.get("user_prompt")
        try:
            result = await original_run(self, *args, **kwargs)
            _log_to_langfuse(
                agent_name,
                prompt=prompt,
                output=getattr(result, "output", None),
                metadata={"call": "run"},
            )
            return result
        except Exception as exc:
            _log_to_langfuse(
                agent_name,
                prompt=prompt,
                error=exc,
                metadata={"call": "run"},
            )
            raise

    if inspect.iscoroutinefunction(original_run):
        Agent.run = instrumented_run  # type: ignore[method-assign]
    Agent.run_sync = instrumented_run_sync  # type: ignore[method-assign]
    _INSTRUMENTED = True


def log_agent_output(
    logger: logging.Logger,
    agent_name: str,
    output: Any,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> None:
    """Log an agent's structured output at INFO level.

    Parameters
    ----------
    logger:
        The caller's module logger (so the log line is attributed to
        the right module).
    agent_name:
        Short label identifying the agent (e.g. ``"Auditor"``,
        ``"Generation"``, ``"PhysicsExtraction"``).
    output:
        Whatever the agent returned (typically a pydantic model from
        ``result.output``).
    max_chars:
        Truncate the serialised payload to this many characters. Set
        to a non-positive number to disable truncation.
    """
    payload = _serialise(output)
    if max_chars > 0 and len(payload) > max_chars:
        payload = payload[:max_chars] + f"... [+{len(payload) - max_chars} chars]"
    logger.info("[%s] Agent output: %s", agent_name, payload)
    _log_to_langfuse(agent_name, output=output, metadata={"source": "log_agent_output"})


configure_agent_instrumentation()
