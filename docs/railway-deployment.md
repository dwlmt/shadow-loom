# Railway Deployment Guide

End-to-end recipe for running Shadow-Loom in production on
[Railway](https://railway.app) with managed Postgres, OAuth login, and
LLM calls routed through OpenRouter.

The same image works on Fly.io, Cloud Run, and any other PaaS that
injects `$PORT` and a Postgres `$DATABASE_URL` — the Railway-specific
parts are flagged inline.

---

## 0. Prerequisites

* A Railway account ([railway.app](https://railway.app)).
* An OpenRouter API key ([openrouter.ai/keys](https://openrouter.ai/keys)).
* OAuth credentials from at least one provider (GitHub is the fastest):
  [github.com/settings/developers](https://github.com/settings/developers).
* This repository pushed to a Git provider Railway can read.

---

## 1. Create the Railway project

1. **New Project → Deploy from GitHub Repo** and pick this repo.
2. Railway will detect the [`Dockerfile`](../Dockerfile) and the
   [`railway.toml`](../railway.toml) at the project root and use them
   for build + deploy. No buildpack guesswork.
3. After the first build finishes, **add a Postgres service** to the
   same project: **+ New → Database → Add PostgreSQL**.
4. Open the Postgres service → **Variables** → copy the
   `DATABASE_URL` reference.

   In the **web service**'s Variables tab, add:

   ```
   DATABASE_URL = ${{Postgres.DATABASE_URL}}
   ```

   The `${{...}}` syntax tells Railway to inject the live value at
   deploy time. Shadow-Loom rewrites the legacy `postgres://` prefix
   to `postgresql+psycopg://` automatically — see
   [`shadow_loom/db.py`](../shadow_loom/db.py).

5. Generate a public domain for the web service: **Settings →
   Networking → Generate Domain**. Note the URL — you'll need it for
   `OAUTH_REDIRECT_BASE`.

> Railway injects `PORT` for you. The image binds to whatever value
> arrives there (see [`shadow_loom/settings.py`](../shadow_loom/settings.py)
> `UISettings._honour_platform_port`).

---

## 2. Set environment variables

Open the web service's **Variables** tab and paste in the values from
[`.env.production.example`](../.env.production.example). The
**minimum** required for a working production deploy is below.

### Required

| Variable | How to get it |
|---|---|
| `STORAGE_SECRET` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `OAUTH_REDIRECT_BASE` | The Railway public domain, e.g. `https://shadow-loom-production.up.railway.app` (no trailing slash) |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |

### At least one OAuth provider

Set both halves of one provider's credentials. The login page hides any
provider whose client id is empty.

| Provider | Variables | Callback URL to register |
|---|---|---|
| GitHub | `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | `${OAUTH_REDIRECT_BASE}/auth/github/callback` |
| Google | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | `${OAUTH_REDIRECT_BASE}/auth/google/callback` |
| Discord | `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET` | `${OAUTH_REDIRECT_BASE}/auth/discord/callback` |
| Microsoft | `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET` | `${OAUTH_REDIRECT_BASE}/auth/microsoft/callback` |
| Apple | `APPLE_CLIENT_ID` *plus* either `APPLE_CLIENT_SECRET` (pre-minted JWT) or the trio `APPLE_TEAM_ID` + `APPLE_KEY_ID` + `APPLE_PRIVATE_KEY` | `${OAUTH_REDIRECT_BASE}/auth/apple/callback` (must be HTTPS — Apple rejects `http://`) |

For GitHub specifically:

1. <https://github.com/settings/developers> → **New OAuth App**
2. Homepage URL: your Railway domain
3. Authorization callback URL: `https://YOUR-DOMAIN/auth/github/callback`
4. Generate a client secret, paste both into Railway

For Apple specifically:

1. Create an **App ID** with Sign in with Apple enabled at
   <https://developer.apple.com/account/resources/identifiers/list>.
2. Create a **Services ID** under the same identifier, add your
   Railway domain to *Domains and Subdomains*, and add the full
   callback URL above to *Return URLs*. Put the Services ID in
   `APPLE_CLIENT_ID` (e.g. `com.example.shadow-loom.web`).
3. Create a **Key** with Sign in with Apple enabled, download the
   `.p8` file, and copy its full PEM contents (including the
   BEGIN/END lines) into `APPLE_PRIVATE_KEY`. Set `APPLE_KEY_ID` to
   the 10-character key id and `APPLE_TEAM_ID` to your developer
   team id (top-right of the Apple Developer portal). Shadow-Loom
   will mint and refresh the ES256 client_secret JWT for you.

### Recommended LLM routing

```
DEFAULT_MODEL=openrouter:qwen/qwen3.6-35b-a3b
GENERATION_MODEL=openrouter:qwen/qwen3.6-35b-a3b
QUERY_PARSING_MODEL=openrouter:qwen/qwen3.6-35b-a3b
AUDITOR_MODEL=openrouter:qwen/qwen3.6-35b-a3b
AUDITOR_GENERATION_MODEL=openrouter:qwen/qwen3.6-35b-a3b
EXTRACTION_MODEL=openrouter:qwen/qwen3.6-35b-a3b

# Output-token caps tuned for Qwen3.6-35B-A3B (262K native context).
GENERATION_MAX_TOKENS=64000
QUERY_PARSING_MAX_TOKENS=16000
AUDITOR_MAX_TOKENS=16000
AUDITOR_MAX_TOKENS_GENERATION=64000
```

The defaults match local development against
`ollama:qwen3.6:35b`, so prompts are tuned for it. See
[`qwen/qwen3.6-35b-a3b` on OpenRouter](https://openrouter.ai/qwen/qwen3.6-35b-a3b)
for live pricing (~$0.15/M in, $1/M out at the time of writing) and
provider routing. Pick any other model from
<https://openrouter.ai/models>; the `openrouter:` prefix is parsed
by PydanticAI and everything after the colon is the OpenRouter model
id verbatim. **If you swap to a short-output model** (Claude 3.5
Sonnet caps at ~8K, GPT-4o at ~16K) drop the `*_MAX_TOKENS` knobs to
match — the defaults assume Qwen3.6's full 262K window. See
[settings.md §3](settings.md#3-generation-step-10--prose-rendering)
for the full list of stage-specific model knobs.

### Production safety

Make sure these stay set:

```
PIPELINE_SKIP_AUDIT=false
MCP_ALLOW_OPEN_MODE=false      # critical — open mode disables auth checks
UI_RELOAD=false
```

`MCP_ALLOW_OPEN_MODE=true` is **only** for local development.

### Optional: external research provider (off by default)

The pipeline can call an external web-research provider (currently
Tavily) and distil results into a segregated `WorldStateV1.world_facts`
collection that the renderer treats as background-only context. The
feature is **off by default** and requires three things:

1. **Re-build the image with the `[research]` extra.** Add a
   `INSTALL_EXTRAS` build arg in your Railway service settings under
   *Build → Build Command* (or set it via `railway.toml`):

   ```
   --build-arg INSTALL_EXTRAS=research
   ```

   The bundled [Dockerfile](../Dockerfile) reads this arg and runs
   `pip install ".[research]"` instead of plain `pip install .`.

2. **Add the API key + toggles in the Variables tab**:

   ```
   TAVILY_API_KEY=tvly-...
   EXTRACTION_ENABLE_RESEARCH_AGENT=true
   EXTRACTION_RESEARCH_PROVIDER=tavily
   EXTRACTION_RESEARCH_PROVIDER_MODEL=basic   # or "advanced"
   EXTRACTION_RESEARCH_MAX_RESULTS_PER_QUERY=5
   EXTRACTION_RESEARCH_TOPICS=[]              # JSON array; empty = lookup live via MCP only
   ```

3. **Drive lookups from your client.** With the toggles on, agents can
   call the `research_topic` MCP tool (write scope) to add a
   `WorldFact` to the active project; `list_world_facts` and
   `delete_world_fact` round out the management surface. See
   [mcp-guide.md §3 RESEARCH](mcp-guide.md#research--look-up-real-world-background-on-a-topic-optional)
   and the research-related sections of
   [docs/architecture.md](architecture.md#step-3d--optional-external-research-segregated-off-by-default)
   and [CONTENT-POLICY.md](../CONTENT-POLICY.md#64-external-research-data) for the
   per-account isolation contract.

Provider calls are cached per-account and never reused across users
(see [CONTENT-POLICY.md §6.4 / §6.4a](../CONTENT-POLICY.md)). Leaving
`EXTRACTION_ENABLE_RESEARCH_AGENT=false` (the default) is enough to
guarantee no provider call is ever made — the `TAVILY_API_KEY`
variable can stay blank.

---

## 3. Deploy

Once the variables are saved, hit **Deploy** (or push to the tracked
branch). Railway will:

1. Build the Docker image (the layered `pip install .` step is
   ~3–5 minutes on a cold cache, ~30s on a warm cache).
2. Run the `HEALTHCHECK` defined in the [Dockerfile](../Dockerfile)
   against `/`.
3. Cut traffic over once healthy.

On first boot Shadow-Loom will:

* Create all SQLModel tables on the attached Postgres
  (`init_db` in [`shadow_loom/db.py`](../shadow_loom/db.py)).
* Seed the example user and example projects (the worlds defined in
  [`example_worlds/`](../example_worlds/)).
* Mount the OAuth `/auth/*` routes and the auth middleware.

Hit your Railway domain, click **Sign in with GitHub** (or your
chosen provider), and you should land on the dashboard.

---

## 4. Wiring an MCP client

Shadow-Loom's MCP server runs as a separate entry point
(`python -m shadow_loom_mcp`). You have two deployment options:

* **Same image, second service.** Add another Railway service from the
  same repo with `CMD ["python", "-m", "shadow_loom_mcp"]` (override the
  start command in **Settings → Deploy**). Share the same Postgres.
* **Local only.** Most users will run the MCP server on their own
  machine pointing at the production database — set `DATABASE_URL` in
  `.env` to the Railway Postgres connection string and run
  `python -m shadow_loom_mcp`.

API keys for MCP/Bearer auth are issued from **Account → API Keys** in
the deployed UI. See [mcp-guide.md](mcp-guide.md) for the agent-facing
contract.

---

## 5. Operational notes

| Topic | Notes |
|---|---|
| **Persistent storage** | Postgres holds all state. The container filesystem is ephemeral on Railway — never write user data to local disk. |
| **Logs** | `railway logs` or the Railway dashboard. Shadow-Loom logs at INFO by default; raise to DEBUG via standard `LOGGING_LEVEL` if you wire it through. |
| **Migrations** | Tables are created with `SQLModel.metadata.create_all` on startup — additive only. Schema changes that drop or rename columns need an Alembic migration; one is not yet shipped. |
| **Backups** | Railway Postgres has automated daily backups on the paid tier. Verify in the Postgres service settings. |
| **Cost** | The biggest variable is OpenRouter usage. Set per-key spend limits in the OpenRouter dashboard. The audit loop dominates — see [settings.md §5](settings.md#5-auditor-step-11--llm-as-judge-auditrefine-loop) for the cost knobs. |
| **Scaling** | The default is `numReplicas = 1`. The session state cache (`_SESSION_STATES` in [`shadow_loom_ui/app.py`](../shadow_loom_ui/app.py)) is in-process — running multiple replicas needs a sticky-session ingress or a shared session store. Single replica is the supported topology for now. |

---

## 6. Local production-parity smoke test

Before pushing, you can run the production image against a local
Postgres:

```bash
docker build -t shadow-loom .

docker run --rm -p 8080:8080 \
    -e PORT=8080 \
    -e DATABASE_URL='postgresql://user:pass@host.docker.internal:5432/shadow' \
    -e STORAGE_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
    -e OAUTH_REDIRECT_BASE='http://localhost:8080' \
    -e OPENROUTER_API_KEY=sk-or-... \
    -e GITHUB_CLIENT_ID=... -e GITHUB_CLIENT_SECRET=... \
    -e DEFAULT_MODEL='openrouter:qwen/qwen3.6-35b-a3b' \
    shadow-loom
```

If the container comes up healthy and you can sign in with GitHub on
`http://localhost:8080`, the production deploy will work.

---

## See also

* [`Dockerfile`](../Dockerfile) — the production image.
* [`railway.toml`](../railway.toml) — Railway config-as-code.
* [`.env.production.example`](../.env.production.example) — copy-paste env template.
* [settings.md](settings.md) — every runtime knob and tuning recipe.
* [mcp-guide.md](mcp-guide.md) — agent-facing contract for the MCP service.
* [ui-guide.md](ui-guide.md) — what users see once they're signed in.
