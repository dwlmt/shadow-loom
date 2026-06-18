# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FastAPI app mirroring the Shadow-Loom MCP service over HTTP/JSON.

Design — *one* implementation, two transports
---------------------------------------------
The MCP tools (``shadow_loom_mcp.server``) are plain callables once
decorated (fastmcp returns the original function), and they take their
identity through a ``ctx`` parameter that funnels into
``shadow_loom_mcp.auth``. This adapter therefore reuses them directly:

  * ``GET  /tools``            — list every tool with its JSON-schema
    (the same catalogue MCP clients discover).
  * ``GET  /tools/{name}``     — one tool's schema.
  * ``POST /tools/{name}``     — invoke a tool; the JSON body is the
    tool's keyword arguments. Auth comes from the ``Authorization:
    Bearer <api-key>`` header, resolved to a
    :class:`~shadow_loom_mcp.auth.Principal` and passed in the ``ctx``
    slot.

No tool logic, validation, scope check, or response shape is duplicated
here — this file is purely transport plumbing.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from shadow_loom_mcp import server as _server
from shadow_loom_mcp.auth import Principal, _open_mode_enabled, resolve_principal

logger = logging.getLogger(__name__)

API_VERSION = "1"


# ── Tool registry (built once from the MCP server) ───────────────────


def _collect_tools() -> list[Any]:
    """Return the MCP server's ``FunctionTool`` list.

    ``mcp.list_tools`` is async; call it on a private loop so this works
    whether or not an event loop is already running at import/startup.
    """
    coro_factory = _server.mcp.list_tools
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro_factory())).result()


def _build_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for tool in _collect_tools():
        name = tool.name
        fn = getattr(_server, name, None)
        if fn is None or not callable(fn):
            # A tool registered under a name we can't resolve back to a
            # module-level callable — skip rather than guess.
            logger.warning("REST: no module callable for MCP tool %r; skipping.", name)
            continue
        sig = inspect.signature(fn)
        param_names = [p for p in sig.parameters if p != "ctx"]
        required = [
            p
            for p, spec in sig.parameters.items()
            if p != "ctx" and spec.default is inspect.Parameter.empty
        ]
        registry[name] = {
            "fn": fn,
            "is_async": inspect.iscoroutinefunction(fn),
            "description": (tool.description or "").strip(),
            "parameters": tool.parameters,
            "param_names": set(param_names),
            "required": required,
        }
    return registry


# ── Auth dependency ──────────────────────────────────────────────────


def get_principal(authorization: Optional[str] = Header(default=None)) -> Principal:
    """Resolve the bearer token to a :class:`Principal`.

    Mirrors the MCP HTTP transport: a valid API key is required, except
    in open mode (``MCP_ALLOW_OPEN_MODE=true``, dev/test only) where an
    anonymous principal is allowed and the tools' own open-mode handling
    takes over.
    """
    token: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[len("bearer ") :].strip()

    if token:
        principal = resolve_principal(token)
        if principal is None:
            raise HTTPException(status_code=401, detail="Invalid or expired API key.")
        return principal

    if _open_mode_enabled():
        return Principal(user_id=None, scopes=set())
    raise HTTPException(
        status_code=401,
        detail="Missing bearer token. Send 'Authorization: Bearer <api-key>'.",
    )


# ── Error envelope → HTTP status mapping ─────────────────────────────


def _status_for_error(message: str) -> int:
    """Map a tool's ``{"error": ...}`` envelope to an HTTP status code.

    The tools return human-readable error strings (never structured
    codes for most paths), so this is a deliberately small heuristic:
    everything still travels in the JSON body unchanged — only the HTTP
    status is inferred so REST clients can branch on it.
    """
    low = message.lower()
    if "rate limit" in low:
        return 429
    if "access denied" in low or "permission" in low or "scope" in low:
        return 403
    if "not found" in low or "no world model" in low:
        return 404
    return 400


# ── App factory ──────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="Shadow Loom REST API",
        version=API_VERSION,
        description=(
            "HTTP/JSON mirror of the Shadow-Loom MCP service. Every tool "
            "is invoked via POST /tools/{name} with the tool's arguments "
            "as a JSON object. Authenticate with 'Authorization: Bearer "
            "<api-key>'. See docs/rest-api.md."
        ),
    )

    registry = _build_registry()
    app.state.tool_registry = registry

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict:
        return {"status": "ok", "tools": len(registry), "api_version": API_VERSION}

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "service": "Shadow Loom REST API",
            "api_version": API_VERSION,
            "tool_count": len(registry),
            "endpoints": {
                "list_tools": "GET /tools",
                "tool_schema": "GET /tools/{name}",
                "invoke": "POST /tools/{name}",
                "openapi": "GET /openapi.json",
                "docs": "GET /docs",
            },
        }

    @app.get("/tools", tags=["tools"])
    def list_tools() -> dict:
        return {
            "tools": [
                {
                    "name": name,
                    "description": meta["description"],
                    "parameters": meta["parameters"],
                }
                for name, meta in sorted(registry.items())
            ]
        }

    @app.get("/tools/{tool_name}", tags=["tools"])
    def tool_schema(tool_name: str) -> dict:
        meta = registry.get(tool_name)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Unknown tool '{tool_name}'.")
        return {
            "name": tool_name,
            "description": meta["description"],
            "parameters": meta["parameters"],
        }

    @app.post("/tools/{tool_name}", tags=["tools"])
    async def invoke_tool(
        tool_name: str,
        principal: Principal = Depends(get_principal),
        body: Optional[dict[str, Any]] = Body(default=None),
    ):
        meta = registry.get(tool_name)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Unknown tool '{tool_name}'.")

        args = body or {}
        if not isinstance(args, dict):
            raise HTTPException(
                status_code=422,
                detail="Request body must be a JSON object of tool arguments.",
            )

        unknown = sorted(set(args) - meta["param_names"])
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown argument(s) for '{tool_name}': {unknown}.",
            )
        missing = [r for r in meta["required"] if r not in args]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required argument(s) for '{tool_name}': {missing}.",
            )

        fn = meta["fn"]
        # The tool bodies are wrapped by @_safe_tool, so unhandled
        # exceptions already come back as a sanitised error dict rather
        # than propagating — we don't re-wrap them here.
        if meta["is_async"]:
            result = await fn(principal, **args)
        else:
            result = fn(principal, **args)

        # Some successful tool results carry an ``error`` string purely as
        # human-readable context alongside a real payload — e.g. the
        # causal-/window-gated ``inspect`` reconstructions ("not_yet_occurred",
        # "outside_availability_window") return a ``reconstruction`` key plus an
        # explanatory ``error``. Those are 200s, not failures; only a dict whose
        # sole signal is ``error`` maps to an HTTP error status.
        if (
            isinstance(result, dict)
            and isinstance(result.get("error"), str)
            and "reconstruction" not in result
        ):
            return JSONResponse(
                status_code=_status_for_error(result["error"]),
                content=result,
            )
        return result

    return app


app = create_app()
