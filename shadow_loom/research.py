# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optional web-research grounding for Shadow Loom.

This module is **off by default**. When enabled it lets the user
attach external research topics (e.g. "1972 Lockheed L-1011 cockpit
details") to a project, which are looked up via a pluggable
``ResearchProvider`` (Tavily is the first backend) and distilled by
the ``ResearchExtraction`` agent into segregated ``WorldFact`` records.

Crucially ``WorldFact`` is *separate* from ``Entity`` traits, beliefs,
``RelationshipEdge``, ``EventNode``, and ``GlobalTrait``. It carries
its own ``source_url`` provenance and never auto-modifies the
closed-world model extracted from user prose. See
``docs/research-extraction-plan.md`` for the full design rationale.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import List, Literal, Optional, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =====================================================================
# 1. SNIPPET MODEL — raw provider payload
# =====================================================================

class ResearchSnippet(BaseModel):
    """A single ranked snippet returned by a ``ResearchProvider``.

    Stored verbatim on every ``WorldFact`` so the user can inspect the
    raw source the LLM distilled from. Never displayed to the
    generation LLM directly; only the LLM-distilled ``WorldFact.summary``
    is surfaced to the directive assembler.
    """

    url: str = Field(description="Canonical source URL.")
    title: str = Field(default="", description="Page / article title.")
    content: str = Field(
        description=(
            "Cleaned snippet text returned by the provider. May be a "
            "short excerpt; not the full page."
        ),
    )
    score: float = Field(
        default=0.0,
        description="Provider-supplied relevance score (0.0–1.0).",
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of retrieval.",
    )
    provider: str = Field(
        default="",
        description="Provider name, e.g. 'tavily'. Empty for fixtures.",
    )
    provider_model: str = Field(
        default="",
        description=(
            "Provider model / search-depth identifier when the provider "
            "exposes one (e.g. tavily 'basic' / 'advanced')."
        ),
    )


# =====================================================================
# 2. WORLD FACT — segregated namespace on WorldStateV1
# =====================================================================

class WorldFact(BaseModel):
    """A piece of LLM-distilled external research attached to the project.

    World facts live in their own namespace on ``WorldStateV1.world_facts``
    and never leak into entity traits / beliefs / events. The directive
    assembler may surface a fact under ``CreativeBrief.external_research``
    when one of its ``related_node_ids`` overlaps the scene focus, but
    the auditor never promotes a fact into a trait or belief.
    """

    id: str = Field(description="Unique ID with FACT_ prefix, e.g. FACT_L1011_COCKPIT_1972")
    topic: str = Field(
        description=(
            "The user-supplied research goal that triggered the lookup, "
            "e.g. '1972 Lockheed L-1011 cockpit details'."
        ),
    )
    summary: str = Field(
        description=(
            "One-paragraph LLM-distilled fact derived strictly from the "
            "raw snippets. The agent is instructed never to invent facts "
            "not present in the snippets."
        ),
    )
    raw_snippets: List[ResearchSnippet] = Field(
        default_factory=list,
        description="Verbatim provider payload that the summary was distilled from.",
    )
    related_node_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Optional list of ENT_/LOC_/OBJ_/EVT_ IDs this fact is "
            "relevant to. Used by directive assembly to surface the fact "
            "only when the scene focus overlaps."
        ),
    )
    confidence: Literal["low", "moderate", "high"] = Field(
        default="moderate",
        description=(
            "Agent's self-assessed confidence in the distilled summary. "
            "'low' = sparse / contradictory snippets, 'high' = "
            "consistent across multiple authoritative sources."
        ),
    )
    source_url_primary: str = Field(
        default="",
        description="Single most-authoritative source URL (also present in raw_snippets).",
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp the fact was first established.",
    )
    provider: str = Field(default="", description="Provider used (e.g. 'tavily').")


# =====================================================================
# 3. PROVIDER PROTOCOL + IMPLEMENTATIONS
# =====================================================================

class ResearchProvider(Protocol):
    """Pluggable backend that turns a query into ranked snippets."""

    name: str

    def search(self, query: str, max_results: int = 5, user_id: Optional[int] = None, 
               project_id: Optional[int] = None, agent_call_log_id: Optional[int] = None) -> List[ResearchSnippet]:
        ...


class NullProvider:
    """Placeholder used whenever research is disabled.

    Calling ``search`` raises so callers cannot accidentally hit the
    network when the feature is off.
    """

    name = "none"

    def search(self, query: str, max_results: int = 5, user_id: Optional[int] = None, 
               project_id: Optional[int] = None, agent_call_log_id: Optional[int] = None) -> List[ResearchSnippet]:
        raise RuntimeError(
            "Research is disabled. Set extraction.enable_research_agent=True "
            "and a non-'none' research provider before calling search()."
        )


class TavilyProvider:
    """Tavily-backed research provider.

    Reads ``TAVILY_API_KEY`` from settings. The ``tavily-python``
    package is an *optional* extra (``pip install shadow-loom[research]``)
    — the import is deferred so the core install stays lean.
    """

    name = "tavily"

    def __init__(
        self,
        api_key: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "TavilyProvider requires a non-empty api_key. Set "
                "TAVILY_API_KEY in config.env or .env."
            )
        self.api_key = api_key
        self.search_depth = search_depth
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            try:
                from tavily import TavilyClient  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover — exercised only when extra missing
                raise RuntimeError(
                    "The 'tavily-python' package is required to use "
                    "TavilyProvider. Install with: pip install "
                    "shadow-loom[research]  (or: pip install tavily-python)."
                ) from e
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def search(self, query: str, max_results: int = 5, user_id: Optional[int] = None, 
               project_id: Optional[int] = None, agent_call_log_id: Optional[int] = None) -> List[ResearchSnippet]:
        """Execute a search query via Tavily API with cost tracking.
        
        Parameters
        ----------
        query : str
            The search query
        max_results : int
            Maximum number of results to return
        user_id : int, optional
            User ID for cost tracking
        project_id : int, optional  
            Project ID for cost tracking
        agent_call_log_id : int, optional
            Associated agent call ID for cost tracking
        """
        import time
        start_time = time.time()

        # Round-10 R10-06: research queries can contain narrative
        # spoilers or, for some operators, mildly sensitive entity
        # names. Log a short hash by default and only embed the raw
        # text when SHADOW_LOOM_LOG_RESEARCH_QUERIES is explicitly
        # opted in. The hash is stable so audit + cost-tracking
        # workflows can still correlate retries.
        _log_raw = os.environ.get("SHADOW_LOOM_LOG_RESEARCH_QUERIES", "").strip().lower() in {"1", "true", "yes"}
        query_hash = "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        log_query_value = query if _log_raw else query_hash

        client = self._get_client()
        # Round-12 R12-06: clamp ``max_results`` to a hard ceiling so
        # a caller (or a forwarded MCP param) cannot demand 1000
        # snippets and blow the downstream prompt size / cost budget.
        # 10 is enough for distillation; deeper research should be
        # multiple targeted queries, not one fan-out.
        _MAX_RESULTS_CEILING = 10
        try:
            max_results = max(1, min(int(max_results), _MAX_RESULTS_CEILING))
        except (TypeError, ValueError):
            max_results = 5
        kwargs: dict = {
            "query": query,
            "max_results": max_results,
            "search_depth": self.search_depth,
        }
        if self.include_domains:
            kwargs["include_domains"] = self.include_domains
        if self.exclude_domains:
            kwargs["exclude_domains"] = self.exclude_domains

        # Round-12 R12-05: wrap the Tavily call with bounded retries
        # and a wall-clock timeout per attempt. ``TavilyClient`` uses
        # httpx under the hood with no caller-visible timeout, so a
        # slow upstream could otherwise pin the worker indefinitely.
        # Three attempts with exponential backoff (1s, 2s, 4s)
        # absorbs transient 5xx / connection-reset blips without
        # masking persistent provider outages — the final exception
        # re-raises so the caller's existing failure-logging branch
        # runs.
        import concurrent.futures as _cf

        _PER_ATTEMPT_TIMEOUT_S_RAW = os.environ.get(
            "SHADOW_LOOM_RESEARCH_TIMEOUT_S", "15"
        )
        try:
            _PER_ATTEMPT_TIMEOUT_S = float(_PER_ATTEMPT_TIMEOUT_S_RAW)
            if _PER_ATTEMPT_TIMEOUT_S <= 0 or _PER_ATTEMPT_TIMEOUT_S > 600:
                raise ValueError("out of range")
        except (TypeError, ValueError):
            # R20-L24: tolerate bad env vars so a typo does not crash
            # every research call — fall back to the documented default.
            logger.warning(
                "[Research] Invalid SHADOW_LOOM_RESEARCH_TIMEOUT_S=%r; "
                "falling back to 15s",
                _PER_ATTEMPT_TIMEOUT_S_RAW,
            )
            _PER_ATTEMPT_TIMEOUT_S = 15.0
        _MAX_ATTEMPTS = 3

        def _do_search():
            return client.search(**kwargs)

        try:
            last_exc: Optional[Exception] = None
            raw = None
            for _attempt in range(1, _MAX_ATTEMPTS + 1):
                with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                    _fut = _pool.submit(_do_search)
                    try:
                        raw = _fut.result(timeout=_PER_ATTEMPT_TIMEOUT_S)
                        last_exc = None
                        break
                    except _cf.TimeoutError as exc:
                        last_exc = TimeoutError(
                            f"Tavily search exceeded {_PER_ATTEMPT_TIMEOUT_S:.0f}s "
                            f"on attempt {_attempt}/{_MAX_ATTEMPTS}"
                        )
                    except Exception as exc:
                        last_exc = exc
                if _attempt < _MAX_ATTEMPTS:
                    time.sleep(2 ** (_attempt - 1))
            if raw is None:
                assert last_exc is not None
                raise last_exc
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Log API call for cost tracking
            if user_id:
                try:
                    from shadow_loom._agent_logging import log_api_call
                    results_count = len(raw.get("results", [])) if isinstance(raw, dict) else 0
                    request_size = len(query.encode('utf-8'))  # Query size in bytes
                    
                    log_api_call(
                        user_id=user_id,
                        provider="tavily",
                        service_type="web_search", 
                        request_size=request_size,
                        response_time_ms=response_time_ms,
                        status_code=200,  # Assume success if no exception
                        project_id=project_id,
                        agent_call_log_id=agent_call_log_id,
                        metadata={
                            "query": log_query_value,
                            "max_results": max_results, 
                            "search_depth": self.search_depth,
                            "results_count": results_count
                        }
                    )
                except Exception as log_err:
                    logger.warning(f"Failed to log Tavily API call: {log_err}")
                    
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Log failed API call
            if user_id:
                try:
                    from shadow_loom._agent_logging import log_api_call
                    log_api_call(
                        user_id=user_id,
                        provider="tavily",
                        service_type="web_search",
                        request_size=len(query.encode('utf-8')),
                        response_time_ms=response_time_ms,
                        status_code=500,  # Error status
                        project_id=project_id,
                        agent_call_log_id=agent_call_log_id,
                        metadata={
                            "query": log_query_value,
                            "error": str(e),
                            "search_depth": self.search_depth
                        }
                    )
                except Exception as log_err:
                    logger.warning(f"Failed to log failed Tavily API call: {log_err}")
                    
            logger.exception("[TavilyProvider] search FAILED for query=%s", log_query_value)
            raise

        results = raw.get("results", []) if isinstance(raw, dict) else []
        # Round-12 R12-06: cap each snippet's content length so a
        # provider returning very long page extracts cannot blow the
        # downstream prompt size / cost budget. 1500 chars is enough
        # context for the distillation agent to anchor a WorldFact
        # while keeping the assembled prompt bounded even at the
        # ceiling of 10 snippets.
        _SNIPPET_CHAR_CAP = 1500
        snippets: List[ResearchSnippet] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or ""
            if not url:
                continue
            content = (item.get("content", "") or "")[:_SNIPPET_CHAR_CAP]
            snippets.append(ResearchSnippet(
                url=url,
                title=item.get("title", "") or "",
                content=content,
                score=float(item.get("score", 0.0) or 0.0),
                provider=self.name,
                provider_model=self.search_depth,
            ))
        return snippets


# =====================================================================
# 4. CACHE KEY HELPER
# =====================================================================

def cache_key(provider: str, provider_model: str, query: str) -> str:
    """Stable SHA-256 cache key for (provider, model/depth, query).

    The DB cache layer (``shadow_loom.db.research_cache``) keys rows
    on this digest so identical re-runs hit the cache and produce
    deterministic ``WorldFact`` output across pipeline runs.
    """
    h = hashlib.sha256()
    h.update(provider.encode("utf-8"))
    h.update(b"\x00")
    h.update(provider_model.encode("utf-8"))
    h.update(b"\x00")
    h.update(query.encode("utf-8"))
    return h.hexdigest()


# =====================================================================
# 5. PROVIDER FACTORY
# =====================================================================

def build_provider(name: str, *, api_key: str = "", search_depth: str = "basic", **kwargs) -> ResearchProvider:
    """Construct a provider by name. Used by ingestion + MCP."""
    name = (name or "none").lower()
    if name in {"none", "null", ""}:
        return NullProvider()
    if name == "tavily":
        return TavilyProvider(
            api_key=api_key or os.environ.get("TAVILY_API_KEY", ""),
            search_depth=search_depth,  # type: ignore
            **kwargs,
        )
    raise ValueError(f"Unknown research provider: {name!r}")


# =====================================================================
# 6. HIGH-LEVEL HELPER — provider call + agent distil + persistence
# =====================================================================

def lookup_and_persist_topic(
    *,
    project_id: int,
    user_id: int,
    topic: str,
    provider_override: Optional[str] = None,
    max_results_override: Optional[int] = None,
) -> dict:
    """Look up *topic*, distil to ``WorldFact``, persist under *project_id*.

    Shared by the MCP ``research_topic`` tool and the UI Research tab so
    both surfaces hit the same cache + persistence layer with identical
    per-account isolation.

    Returns a dict with either ``{"error": str}`` or the persisted fact
    summary fields. Never raises on provider / agent failure — always
    folds the error into the return dict.
    """
    import json as _json

    # Local imports to avoid top-level cycles (db ↔ models ↔ research).
    from shadow_loom.db import (
        allocate_world_fact_id,
        get_cached_research,
        save_cached_research,
        upsert_world_fact,
    )
    from shadow_loom.ingestion import _build_research_agent, ExtractionConfig
    from shadow_loom.settings import get_settings

    topic = (topic or "").strip()
    if not topic:
        return {"error": "topic must be a non-empty string"}

    settings = get_settings()
    ext_kwargs = settings.extraction_config()
    ext_kwargs["enable_research_agent"] = True
    if provider_override is not None:
        ext_kwargs["research_provider"] = provider_override
    if max_results_override is not None:
        # Round-12 R12-06: clamp at the entry point as well so the
        # config-derived value carried into the cache key / agent
        # prompt is already bounded. Defence-in-depth with the
        # provider-level clamp.
        try:
            _mr = max(1, min(int(max_results_override), 10))
        except (TypeError, ValueError):
            _mr = 5
        ext_kwargs["research_max_results_per_query"] = _mr
    ext_kwargs["research_topics"] = [topic]
    config = ExtractionConfig(**ext_kwargs)

    if config.research_provider == "none":
        return {"error": (
            "No research provider configured. Set "
            "EXTRACTION_RESEARCH_PROVIDER=tavily and TAVILY_API_KEY in "
            "your environment, or pass provider_override='tavily'."
        )}

    key = cache_key(config.research_provider, config.research_provider_model, topic)
    cached = get_cached_research(user_id=user_id, key=key)
    if cached is not None:
        try:
            snippets = [ResearchSnippet(**s) for s in _json.loads(cached.snippets_json)]
        except Exception:
            snippets = []
    else:
        try:
            prov = build_provider(
                config.research_provider,
                api_key=settings.core.tavily_api_key,
                search_depth=config.research_provider_model or "basic",
            )
            snippets = list(prov.search(topic, max_results=config.research_max_results_per_query,
                                      user_id=user_id, project_id=project_id))
        except Exception as e:
            # R20-H8: do not leak ``repr(e)`` to clients — it can
            # contain API key fragments / URL params. Log the
            # exception (with traceback) server-side and return a
            # sanitized type name.
            logger.exception(
                "[Research] Provider call failed for topic=%r", topic
            )
            return {"error": f"Provider call failed: {type(e).__name__}"}
        try:
            save_cached_research(
                user_id=user_id,
                key=key,
                provider=config.research_provider,
                provider_model=config.research_provider_model,
                query=topic,
                snippets_json=_json.dumps([s.model_dump(mode="json") for s in snippets]),
            )
        except Exception:
            logger.exception("[Research] cache write failed for topic=%r", topic)

    if not snippets:
        return {"error": f"Provider returned no results for topic={topic!r}."}

    snippet_block = "\n\n".join(
        f"[{i+1}] {s.title}\nURL: {s.url}\n{s.content}"
        for i, s in enumerate(snippets)
    )
    user_msg = (
        f"Topic: {topic}\n\n"
        f"Snippets ({len(snippets)}):\n\n{snippet_block}\n\n"
        "Distil the above into a single WorldFact per the system prompt."
    )

    try:
        agent = _build_research_agent(config)
        result = agent.run_sync(
            user_msg,
            model_settings={"max_tokens": 4096},
            user_id=user_id,
            project_id=project_id,
        )
        fact = result.output
    except Exception as e:
        # R20-H9: sanitize — see R20-H8 above.
        logger.exception(
            "[Research] Research agent failed for topic=%r", topic
        )
        return {"error": f"Research agent failed: {type(e).__name__}"}

    # Round-10 R10-04: allocate the FACT id via the helper and retry
    # on IntegrityError so concurrent MCP research_topic callers cannot
    # both claim the same id and crash the second one. We bound the
    # retry loop so a stuck constraint never spins forever.
    from sqlalchemy.exc import IntegrityError as _IntegrityError

    last_err: Optional[Exception] = None
    fact_id = ""
    for _attempt in range(5):
        fact_id = allocate_world_fact_id(project_id)
        try:
            upsert_world_fact(
                project_id=project_id,
                fact_id=fact_id,
                topic=topic,
                summary=fact.summary,
                confidence=fact.confidence,
                source_url_primary=fact.source_url_primary or "",
                provider=config.research_provider,
                related_node_ids_json=_json.dumps(list(fact.related_node_ids)),
                raw_snippets_json=_json.dumps([s.model_dump(mode="json") for s in snippets]),
            )
            break
        except _IntegrityError as e:
            last_err = e
            continue
    else:
        return {"error": f"Could not allocate WorldFact id after 5 attempts: {last_err!r}"}

    return {
        "project_id": project_id,
        "fact_id": fact_id,
        "topic": topic,
        "summary": fact.summary,
        "confidence": fact.confidence,
        "source_url_primary": fact.source_url_primary,
        "related_node_ids": list(fact.related_node_ids),
        "snippet_count": len(snippets),
        "cached": cached is not None,
    }

