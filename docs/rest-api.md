# REST API Guide — `shadow_loom_rest`

Shadow-Loom ships a [FastAPI](https://fastapi.tiangolo.com) adapter that
exposes the **same 42 operations** as the [MCP server](mcp-guide.md) over plain
HTTP/JSON, for clients that don't speak the Model Context Protocol (web apps,
`curl`, CI scripts, language runtimes without an MCP SDK).

It lives in [`shadow_loom_rest/`](../shadow_loom_rest/) and is deliberately thin:
**it reimplements nothing.** Every endpoint dispatches to the exact same tool
function registered on the MCP server, so authentication, scope/role checks,
rate limits, world-state loading, the pipeline, and the response envelopes are
shared verbatim with MCP. See [§6 Shared-code architecture](#6-shared-code-architecture).

---

## 1. Quick start

```bash
conda activate shadow-loom
pip install -e .

# Run the REST server (defaults to 127.0.0.1:8100)
python -m shadow_loom_rest

# Bind elsewhere
REST_HOST=0.0.0.0 REST_PORT=8100 python -m shadow_loom_rest
```

Interactive docs are auto-generated at `GET /docs` (Swagger UI) and
`GET /openapi.json`.

| Env var | Purpose |
|---|---|
| `DATABASE_URL` | SQLite / Postgres URL backing the version store (same store as MCP and the UI). |
| `REST_HOST` | Bind address (default `127.0.0.1`). |
| `REST_PORT` / `PORT` | Bind port (default `8100`). |
| `AUTH_REQUIRED` | Set `true` in production so the open-mode fallback (below) stays disabled. |
| `MCP_ALLOW_OPEN_MODE` | If `true`, requests **without** a bearer token are allowed as an anonymous principal (dev/test only — fail-open). Default `false`. Ignored when `AUTH_REQUIRED=true`. |

All other behaviour-shaping settings (`MCP_SKIP_AUDIT`, `MCP_INGEST_*`,
`SHADOW_LOOM_MCP_RATE_PER_MIN`, the per-project research toggles, …) are read
from the same [`settings.py`](../shadow_loom/settings.py) the MCP server uses
and apply identically here.

---

## 2. Authentication & scopes

Send the user's API key as a bearer token:

```
Authorization: Bearer sl_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**The same API keys work for MCP and REST.** A key is an `ApiKeyRow` minted
once per user via `db.create_api_key(user_id, name, scopes="read,write")`
(returns the raw `sl_…` token exactly once). There is **no separate REST key
issuance** — the REST adapter resolves the token through the very same
`validate_api_key` / token-cache path as the MCP `verifier`
(see [`auth.resolve_principal`](../shadow_loom_mcp/auth.py)), so revoking a key
(`db.revoke_api_key`) disables it on both transports at once.

Scopes and per-project role enforcement are identical to MCP — see
[mcp-guide §2](mcp-guide.md#2-authentication--scopes). In brief:

| Scope | Tool groups |
|---|---|
| `read` | ORIENT + EXPLORE + REASON + JUDGE + `get_active_version` |
| `write` | CREATE + most MANAGE (branch, fork, delete, …) |
| `admin` | `share`, `update_project_tool` |

**Open mode.** Without `Authorization`, the server returns `401` unless
`MCP_ALLOW_OPEN_MODE=true`, in which case an anonymous principal is used and the
tools' own open-mode handling applies (trusted local dev only).

---

## 3. Endpoints

The REST surface is a faithful RPC-style mirror of the MCP tool catalogue:
three meta routes plus one invocation route that covers all 42 tools.

| Method & path | Purpose |
|---|---|
| `GET /healthz` | Liveness + tool count. |
| `GET /` | Service banner + endpoint map. |
| `GET /tools` | List every tool: `name`, `description`, and JSON-schema `parameters` (the same catalogue MCP clients discover). |
| `GET /tools/{name}` | One tool's schema. |
| `POST /tools/{name}` | Invoke a tool. The JSON request body is the tool's keyword arguments (`ctx` is supplied by the transport and never appears in the schema). |

### Invoking a tool

```bash
# List the caller's projects
curl -s -X POST http://127.0.0.1:8100/tools/list_projects \
  -H "Authorization: Bearer $SL_KEY" \
  -H "Content-Type: application/json" -d '{}'

# Open a project's manifest
curl -s -X POST http://127.0.0.1:8100/tools/open_project \
  -H "Authorization: Bearer $SL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"project_id": 42}'

# Inspect a node at a fabula time, through a character's POV
curl -s -X POST http://127.0.0.1:8100/tools/inspect \
  -H "Authorization: Bearer $SL_KEY" -H "Content-Type: application/json" \
  -d '{"node_id": "ENT_MACBETH", "project_id": 42, "at_time": 12000, "pov_entity_id": "ENT_BANQUO"}'

# Generate prose (async tool — same body, same response envelope as MCP)
curl -s -X POST http://127.0.0.1:8100/tools/narrate \
  -H "Authorization: Bearer $SL_KEY" -H "Content-Type: application/json" \
  -d '{"instruction": "Macbeth hesitates at the chamber door", "project_id": 42, "idempotency_key": "draft-1"}'
```

The full tool catalogue, parameters, and semantics — including the coarse-grained
dispatchers (`discover`, `trace`, `author`, `manage`) recommended for new
integrations — are documented once in
[mcp-guide §3](mcp-guide.md#3-the-42-tools-by-cognitive-task). The `parameters`
schema returned by `GET /tools/{name}` is authoritative and always in sync,
since it is generated from the same tool definitions.

---

## 4. Responses & error mapping

Successful calls return the tool's response dict as the JSON body with `200`.
The auto-versioning contract, response envelope fields (`version`,
`version_row_id`, `changeset`, `findings`, `audit_trace`, `idempotent_replay`,
…), and idempotency semantics are exactly those of the MCP tools — see
[mcp-guide §5](mcp-guide.md#5-the-auto-versioning-contract) and
[§7](mcp-guide.md#7-error-surface).

When a tool returns an `{"error": "..."}` envelope, the adapter keeps the dict
as the body and infers an HTTP status so REST clients can branch on it:

| Condition (substring in the error) | HTTP status |
|---|---|
| `rate limit` | `429` |
| `access denied` / `permission` / `scope` | `403` |
| `not found` / `no world model` | `404` |
| anything else | `400` |

Transport-level failures are distinct from tool error envelopes:

| Situation | Status |
|---|---|
| Missing bearer token (and not open mode) | `401` |
| Invalid / expired / revoked key | `401` |
| Unknown tool name | `404` |
| Unknown or missing required argument | `422` |

Because the tool bodies are wrapped by the MCP server's `@_safe_tool`,
unhandled exceptions surface as a sanitised `{"error": "<tool> failed",
"error_type": "..."}` envelope (mapped to `400`) rather than leaking internals.

---

## 5. Rate limiting

The expensive generation tools (`narrate`, `direct`) enforce the same
per-(user, kind) in-process token bucket as MCP, governed by
`SHADOW_LOOM_MCP_RATE_PER_MIN` (default 10/min). Because the limiter keys off the
resolved user id, a user's MCP and REST calls **share one budget** — the limit
is per identity, not per transport.

---

## 6. Shared-code architecture

The guiding principle is *one implementation, two transports*:

```
            Authorization: Bearer <api-key>
                       │
   MCP (FastMCP) ──┐   │   ┌── REST (FastAPI)
   ctx: Context    │   │   │  Principal(user_id, scopes)
                   ▼   ▼   ▼
        shadow_loom_mcp.auth   ← resolve identity (validate_api_key, token cache)
        shadow_loom_mcp.server ← the 42 tool bodies (require_scope, resolve_project, …)
        shadow_loom_mcp.helpers← load_world_state*, run_and_save (pipeline)
                   │
                   ▼
        shadow_loom.* (engine, db, models)
```

* The MCP tools take their identity through a `ctx` parameter. The REST adapter
  has no FastMCP context, so it builds a
  [`Principal`](../shadow_loom_mcp/auth.py) from the bearer token and passes it
  into that **same `ctx` slot**.
* `auth._resolve_cache_entry`, `get_user_id`, and `get_scopes` short-circuit on
  a `Principal`, so every downstream check (`require_scope`,
  `check_project_access`, `check_rate_limit`) and helper (`resolve_project`,
  `load_world_state_projected`, `run_and_save`) is reused without modification.
* `Principal.report_progress` is an async no-op so the streaming generation
  tools (`narrate` / `direct` / `ingest`) run unchanged over REST.
* `GET /tools` reflects FastMCP's own per-tool JSON schema, so the REST
  catalogue can never drift from the MCP one.

Net result: adding or changing a tool in `shadow_loom_mcp/server.py` updates both
transports at once. The REST package contains no business logic.

---

## 7. Tests

[`tests/test_rest_api.py`](../tests/test_rest_api.py) covers catalogue parity
(REST tool set == MCP tool set == 42), auth (`401` missing/invalid, valid-key
dispatch, open-mode anonymous), error→HTTP mapping (`403` scope, `404` missing
world model / unknown tool, `422` bad args), the `Principal`/`resolve_principal`
seam, and an explicit assertion that **one API key authenticates on both MCP and
REST**.

---

## See also

* [mcp-guide.md](mcp-guide.md) — the canonical tool catalogue and semantics this
  REST surface mirrors.
* [settings.md](settings.md) — every environment variable.
* [architecture.md](architecture.md) — the 12-step pipeline behind the tools.
