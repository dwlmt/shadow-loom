# SPDX-FileCopyrightText: 2026 David Rae Wilmot
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
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from pydantic_ai import Agent
from pydantic import BaseModel

from shadow_loom.settings import get_settings

_DEFAULT_MAX_CHARS = 4000
_LANGFUSE_LOGGER = logging.getLogger(__name__)
_LANGFUSE_CLIENT: Any | None = None
_LANGFUSE_DISABLED = False
_LANGFUSE_INIT_LOCK = threading.Lock()
_INSTRUMENTED = False

# ---------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------
# Narrative content is intentionally verbose, so we don't try to scrub
# every name — that would defeat the point of the system. We DO scrub
# operationally-leaked credentials and contact identifiers that might
# end up in a prompt or agent output by mistake (a directive that
# echoes the operator's email back, a stack trace embedding a bearer
# token, an envvar dump, etc.). Disable per-call by setting
# ``SHADOW_LOOM_DISABLE_LOG_REDACTION=1``.
_REDACTORS: tuple[tuple[re.Pattern[str], str], ...] = (
    # E-mail addresses
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    # Shadow-Loom bearer tokens (prefixed sl_)
    (re.compile(r"\bsl_[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_SL_TOKEN]"),
    # Generic API keys (sk-…, pk-…, key-…) — body may contain dashes/underscores
    (re.compile(r"\b(?:sk|pk|key)[-_][A-Za-z0-9_\-]{16,}\b"), "[REDACTED_API_KEY]"),
    # Bearer tokens in Authorization headers
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"), r"\1[REDACTED]"),
    # Credit-card-like 13-19 digit runs (with optional spaces / dashes)
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_PAN]"),
)


def redact_pii(text: str) -> str:
    """Scrub well-known credential / contact patterns from *text*.

    Conservative on purpose: it removes things that are *almost
    certainly* sensitive (emails, our own bearer-token format, common
    third-party API-key prefixes, Authorization-header bearer tokens,
    credit-card-like digit runs) and leaves narrative content alone.
    """
    if not text or os.environ.get("SHADOW_LOOM_DISABLE_LOG_REDACTION", "").lower() in {"1", "true", "yes", "on"}:
        return text
    redacted = text
    for pattern, replacement in _REDACTORS:
        redacted = pattern.sub(replacement, redacted)
    return redacted

# Database logging (imported lazily to avoid circular imports)
_DB_LOGGING_AVAILABLE = True


def _get_db_models():
    """Lazy import of database models to avoid circular imports."""
    global _DB_LOGGING_AVAILABLE
    if not _DB_LOGGING_AVAILABLE:
        return None, None, None
        
    try:
        from shadow_loom.db import get_session, AgentCallLogRow, ApiCallLogRow
        return get_session, AgentCallLogRow, ApiCallLogRow
    except ImportError:
        _DB_LOGGING_AVAILABLE = False
        return None, None, None


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
    text = redact_pii(text)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + f"... [+{len(text) - max_chars} chars]"
    return text


def _get_langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT, _LANGFUSE_DISABLED
    # Lock-free fast path for the common already-resolved case.
    if _LANGFUSE_DISABLED:
        return None
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    # Lazy init runs from both async and sync agent wrappers; serialize so
    # we don't construct the client twice or race the disabled flag.
    with _LANGFUSE_INIT_LOCK:
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
        "output": redact_pii(_serialise(output)) if output is not None else None,
        "metadata": metadata or {},
    }
    if error is not None:
        payload["level"] = "ERROR"
        payload["status_message"] = repr(error)

    # Method name varies by langfuse major version: v3/v4 expose
    # ``create_event`` (``trace``/``event`` were removed), while v2 has
    # ``trace`` and ``event``. Try them in modern-first order so tracing
    # works on whatever version is installed rather than silently no-opping.
    try:
        for method_name in ("create_event", "trace", "event"):
            emit_fn = getattr(client, method_name, None)
            if callable(emit_fn):
                emit_fn(**payload)
                flush_fn = getattr(client, "flush", None)
                if callable(flush_fn):
                    flush_fn()
                return
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


def _get_agent_type(agent_name: str) -> str:
    """Map agent names to standardized types for cost tracking."""
    agent_name_lower = agent_name.lower()
    
    if "physics" in agent_name_lower:
        return "Physics"
    elif "social" in agent_name_lower:
        return "Social" 
    elif "audit" in agent_name_lower:
        return "Auditor"
    elif "evaluation" in agent_name_lower or "quality" in agent_name_lower:
        return "Evaluation"
    elif "generation" in agent_name_lower:
        return "Generation"
    elif "query" in agent_name_lower or "parsing" in agent_name_lower:
        return "QueryParsing"
    elif "research" in agent_name_lower:
        return "Research"
    else:
        return "Other"


def _extract_context_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """Extract user_id, project_id, version_id from agent kwargs/dependencies."""
    context = {
        'user_id': None,
        'project_id': None, 
        'version_id': None
    }
    
    # Try to extract from dependencies object if present
    deps = kwargs.get('deps') or kwargs.get('dependencies')
    if deps:
        context['user_id'] = getattr(deps, 'user_id', None)
        context['project_id'] = getattr(deps, 'project_id', None)
        context['version_id'] = getattr(deps, 'version_id', None)
    
    # Also check direct kwargs. Pop them so they are not forwarded to
    # PydanticAI's Agent.run/run_sync, which would raise TypeError on
    # these unexpected keyword arguments.
    context['user_id'] = context['user_id'] or kwargs.pop('user_id', None)
    context['project_id'] = context['project_id'] or kwargs.pop('project_id', None)
    context['version_id'] = context['version_id'] or kwargs.pop('version_id', None)

    return context


def _extract_model_info(agent: Agent) -> tuple[Optional[str], Optional[str]]:
    """Extract model provider and name from agent configuration."""
    try:
        model = getattr(agent, 'model', None)
        if model:
            # Legacy combined "provider:model" string (older pydantic-ai).
            combined = getattr(model, 'name', None)
            if isinstance(combined, str) and ':' in combined:
                provider, name = combined.split(':', 1)
                return provider, name
            # pydantic-ai 1.x: provider lives on `.system` (e.g. "openai"),
            # model on `.model_name` (e.g. "gpt-4o"). There is no `.name`.
            name = getattr(model, 'model_name', None) or combined
            if name:
                return getattr(model, 'system', None) or "unknown", name
    except Exception:
        pass
    return None, None


def _extract_token_usage(result: Any) -> Optional[Dict[str, int]]:
    """Extract token usage information from agent result."""
    def _read(obj: Any) -> Dict[str, Optional[int]]:
        # pydantic-ai 1.x RunUsage exposes input_tokens / output_tokens
        # (older versions used prompt_tokens / completion_tokens). Read the
        # new names first, falling back to the legacy ones, so cost tracking
        # works across versions instead of silently recording None/0.
        prompt = getattr(obj, 'input_tokens', None)
        if prompt is None:
            prompt = getattr(obj, 'prompt_tokens', None)
        completion = getattr(obj, 'output_tokens', None)
        if completion is None:
            completion = getattr(obj, 'completion_tokens', None)
        return {
            'prompt_tokens': prompt,
            'completion_tokens': completion,
            'total_tokens': getattr(obj, 'total_tokens', None),
        }

    try:
        if hasattr(result, 'usage'):
            return _read(result.usage)
        elif hasattr(result, 'cost'):
            return _read(result.cost)
    except Exception:
        pass
    return None


@contextmanager
def track_agent_call(
    user_id: Optional[int],
    agent_type: str, 
    agent_name: str,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """Context manager to track agent execution with automatic database logging.
    
    Parameters
    ---------- 
    user_id:
        ID of the user executing the agent
    agent_type:
        Standardized agent type (Physics, Auditor, etc.)
    agent_name:
        Full agent name/class name
    project_id:
        Optional project context
    version_id:
        Optional version context
    model_provider:
        LLM provider name (openai, anthropic, etc.)
    model_name:
        Specific model name (gpt-4o, claude-3, etc.)
    """
    get_session, AgentCallLogRow, _ = _get_db_models()
    
    # Skip database logging if not available or no user context
    if not get_session or not user_id:
        yield None
        return
        
    start_time = time.time()
    log_entry = AgentCallLogRow(
        user_id=user_id,
        project_id=project_id,
        version_id=version_id,
        agent_type=agent_type,
        agent_name=agent_name,
        model_provider=model_provider,
        model_name=model_name,
        status="running"
    )
    
    session = get_session()
    try:
        session.add(log_entry)
        session.commit()
        session.refresh(log_entry)
        
        yield log_entry
        
        # Success - update metrics
        execution_time = int((time.time() - start_time) * 1000)
        log_entry.execution_time_ms = execution_time
        log_entry.status = "success"
        session.commit()
        
    except Exception as e:
        # Error - log failure
        execution_time = int((time.time() - start_time) * 1000)
        log_entry.execution_time_ms = execution_time
        log_entry.status = "error" 
        log_entry.error_message = str(e)[:512]
        try:
            session.commit()
        except Exception:
            session.rollback()
        raise
    finally:
        session.close()


def log_api_call(
    user_id: Optional[int],
    provider: str,
    service_type: str,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None,
    response_time_ms: int = 0,
    status_code: Optional[int] = None,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
    agent_call_log_id: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Log an external API call for cost tracking.
    
    Returns the created ApiCallLogRow or None if logging is unavailable.
    """
    get_session, _, ApiCallLogRow = _get_db_models()
    
    # Skip if database logging not available or no user context
    if not get_session or not user_id:
        return None
        
    log_entry = ApiCallLogRow(
        user_id=user_id,
        project_id=project_id,
        version_id=version_id,
        agent_call_log_id=agent_call_log_id,
        provider=provider,
        service_type=service_type,
        request_size=request_size,
        response_size=response_size,
        response_time_ms=response_time_ms,
        status_code=status_code,
        status="success" if not status_code or 200 <= status_code < 300 else "error",
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    
    session = get_session()
    try:
        session.add(log_entry)
        session.commit()
        session.refresh(log_entry)
        return log_entry
    except Exception as e:
        session.rollback()
        _LANGFUSE_LOGGER.warning(f"Failed to log API call: {e}")
        return None
    finally:
        session.close()


def configure_agent_instrumentation() -> None:
    """Patch PydanticAI Agent methods once to emit Langfuse traces and cost tracking."""
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return

    original_run_sync = Agent.run_sync
    original_run = Agent.run

    def instrumented_run_sync(self: Agent, *args: Any, **kwargs: Any):
        agent_name = _agent_name(self)
        prompt = args[0] if args else kwargs.get("user_prompt")
        
        # Extract context for database logging
        context = _extract_context_from_kwargs(kwargs)
        model_provider, model_name = _extract_model_info(self)
        agent_type = _get_agent_type(agent_name)
        
        with track_agent_call(
            user_id=context['user_id'],
            agent_type=agent_type,
            agent_name=agent_name,
            project_id=context['project_id'],
            version_id=context['version_id'],
            model_provider=model_provider,
            model_name=model_name
        ) as log_entry:
            try:
                result = original_run_sync(self, *args, **kwargs)
                
                # Extract token usage from result if available
                usage = _extract_token_usage(result)
                if usage and log_entry:
                    # Update the log entry with token usage
                    get_session, _, _ = _get_db_models()
                    if get_session:
                        session = get_session()
                        try:
                            # log_entry was created in track_agent_call's
                            # (now-closed) session, so it is detached.
                            # ``merge`` re-attaches it to the new session
                            # so subsequent attribute writes are tracked
                            # and the commit actually persists them.
                            tracked = session.merge(log_entry)
                            tracked.prompt_tokens = usage.get('prompt_tokens')
                            tracked.completion_tokens = usage.get('completion_tokens') 
                            tracked.total_tokens = usage.get('total_tokens')
                            session.commit()
                            
                            # Trigger immediate cost calculation for this entry
                            try:
                                from shadow_loom.cost_calculation import (
                                    CostCalculator,
                                    increment_user_lifetime_usage,
                                )
                                calculator = CostCalculator(session)
                                tracked.estimated_cost_usd = calculator.calculate_agent_call_cost(tracked)
                                session.commit()
                                # Bump the user's lifetime rollup so
                                # the UI cost panel reflects the call
                                # immediately, without waiting for the
                                # nightly summary batch.
                                try:
                                    increment_user_lifetime_usage(
                                        session,
                                        tracked.user_id,
                                        agent_tokens=int(tracked.total_tokens or 0),
                                        agent_cost_usd=float(tracked.estimated_cost_usd or 0.0),
                                        agent_calls=1,
                                    )
                                except Exception as rollup_err:
                                    _LANGFUSE_LOGGER.warning(
                                        f"User-lifetime rollup failed: {rollup_err}"
                                    )
                            except Exception as cost_err:
                                # Don't fail the agent call if cost calculation fails
                                _LANGFUSE_LOGGER.warning(f"Cost calculation failed: {cost_err}")
                        except Exception:
                            session.rollback()
                        finally:
                            session.close()
                
                # Continue with existing Langfuse logging
                _log_to_langfuse(
                    agent_name,
                    prompt=prompt,
                    output=getattr(result, "output", None),
                    metadata={"call": "run_sync", "log_id": log_entry.id if log_entry else None},
                )
                return result
                
            except Exception as exc:
                _log_to_langfuse(
                    agent_name,
                    prompt=prompt,
                    error=exc,
                    metadata={"call": "run_sync", "log_id": log_entry.id if log_entry else None},
                )
                raise

    async def instrumented_run(self: Agent, *args: Any, **kwargs: Any):
        agent_name = _agent_name(self)
        prompt = args[0] if args else kwargs.get("user_prompt")
        
        # Extract context for database logging
        context = _extract_context_from_kwargs(kwargs)
        model_provider, model_name = _extract_model_info(self)
        agent_type = _get_agent_type(agent_name)
        
        with track_agent_call(
            user_id=context['user_id'],
            agent_type=agent_type,
            agent_name=agent_name,
            project_id=context['project_id'],
            version_id=context['version_id'],
            model_provider=model_provider,
            model_name=model_name
        ) as log_entry:
            try:
                result = await original_run(self, *args, **kwargs)
                
                # Extract token usage from result if available
                usage = _extract_token_usage(result)
                if usage and log_entry:
                    # Update the log entry with token usage
                    get_session, _, _ = _get_db_models()
                    if get_session:
                        session = get_session()
                        try:
                            # See instrumented_run_sync above for why
                            # the merge is required.
                            tracked = session.merge(log_entry)
                            tracked.prompt_tokens = usage.get('prompt_tokens')
                            tracked.completion_tokens = usage.get('completion_tokens') 
                            tracked.total_tokens = usage.get('total_tokens')
                            session.commit()
                            
                            # Trigger immediate cost calculation for this entry
                            try:
                                from shadow_loom.cost_calculation import (
                                    CostCalculator,
                                    increment_user_lifetime_usage,
                                )
                                calculator = CostCalculator(session)
                                tracked.estimated_cost_usd = calculator.calculate_agent_call_cost(tracked)
                                session.commit()
                                # Per-call lifetime rollup; see
                                # ``instrumented_run_sync`` for rationale.
                                try:
                                    increment_user_lifetime_usage(
                                        session,
                                        tracked.user_id,
                                        agent_tokens=int(tracked.total_tokens or 0),
                                        agent_cost_usd=float(tracked.estimated_cost_usd or 0.0),
                                        agent_calls=1,
                                    )
                                except Exception as rollup_err:
                                    _LANGFUSE_LOGGER.warning(
                                        f"User-lifetime rollup failed: {rollup_err}"
                                    )
                            except Exception as cost_err:
                                # Don't fail the agent call if cost calculation fails
                                _LANGFUSE_LOGGER.warning(f"Cost calculation failed: {cost_err}")
                        except Exception:
                            session.rollback()
                        finally:
                            session.close()
                
                # Continue with existing Langfuse logging
                _log_to_langfuse(
                    agent_name,
                    prompt=prompt,
                    output=getattr(result, "output", None),
                    metadata={"call": "run", "log_id": log_entry.id if log_entry else None},
                )
                return result
                
            except Exception as exc:
                _log_to_langfuse(
                    agent_name,
                    prompt=prompt,
                    error=exc,
                    metadata={"call": "run", "log_id": log_entry.id if log_entry else None},
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
    payload = redact_pii(payload)
    if max_chars > 0 and len(payload) > max_chars:
        payload = payload[:max_chars] + f"... [+{len(payload) - max_chars} chars]"
    logger.info("[%s] Agent output: %s", agent_name, payload)
    _log_to_langfuse(agent_name, output=output, metadata={"source": "log_agent_output"})


configure_agent_instrumentation()
