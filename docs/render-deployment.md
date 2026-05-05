# Render Deployment Guide

Render's [Blueprint][blueprint] provisioning reads a single
[`render.yaml`](../render.yaml) at the repo root and provisions both
the web service and the managed Postgres database in one click.

This guide covers a from-scratch deploy. For Railway, see
[railway-deployment.md](railway-deployment.md). The two PaaS targets
share the same Dockerfile and `.env.production.example` template.

[blueprint]: https://render.com/docs/blueprint-spec

---

## 1. Prerequisites

* A GitHub (or GitLab) account hosting your fork of this repo.
* A [Render](https://render.com) account (the free tier covers
  evaluation; the Postgres add-on requires the paid `basic-256mb`
  plan or higher).
* An [OpenRouter](https://openrouter.ai/keys) API key (recommended —
  Render web services have no GPU, so local Ollama isn't an option).
* OAuth credentials for at least one provider — see
  [§ 4](#4-oauth-callback-urls) for the callback URL pattern.

---

## 2. Deploy via Blueprint

1. Push your fork to GitHub (Render polls Git for blueprint changes).
2. In the Render dashboard, click **New** → **Blueprint** and select
   your repo. Render reads
   [`render.yaml`](../render.yaml) and shows a preview of the
   resources it will create:
     * `shadow-loom-pg` — managed Postgres 16, plan `basic-256mb`.
     * `shadow-loom` — Docker web service, plan `starter`.
3. Click **Apply**. Render builds the Docker image (~3–5 minutes
   cold) and provisions Postgres in parallel.
4. The first build will fail health checks until you fill in the
   secrets in the next section — that's expected.

---

## 3. Required environment variables

Open the **shadow-loom** service → **Environment** tab in the Render
dashboard and fill in the values flagged as `sync: false` in
[`render.yaml`](../render.yaml). The blueprint pre-populates safe
defaults for everything else (model strings, token caps,
`UI_SOURCE_URL`, etc.).

| Variable | Purpose |
| --- | --- |
| `OAUTH_REDIRECT_BASE` | Public URL of the deploy *without trailing slash* — e.g. `https://shadow-loom.onrender.com` or your custom domain. Used to build OAuth callback URLs. |
| `OPENROUTER_API_KEY` | Get from <https://openrouter.ai/keys>. Required for the default `openrouter:` model strings. |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | Or any other provider — at least one OAuth provider must be configured. |
| `APPLE_CLIENT_ID` (+ JWT trio) | *Optional.* Sign in with Apple. Set `APPLE_CLIENT_ID` to your Services ID and either supply a pre-minted ES256 JWT in `APPLE_CLIENT_SECRET`, or supply `APPLE_TEAM_ID`, `APPLE_KEY_ID`, and `APPLE_PRIVATE_KEY` (the .p8 contents) and Shadow-Loom will mint and refresh the JWT for you. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | *Optional.* Trace LLM calls through [Langfuse](https://langfuse.com). |

Already set automatically by the blueprint:

* `DATABASE_URL` — wired from `shadow-loom-pg.connectionString`.
* `STORAGE_SECRET` — generated once on first deploy; Render keeps it
  stable across redeploys.
* `PORT` — Render injects this; the UI honours it via
  `UISettings._honour_platform_port` in
  [`shadow_loom/settings.py`](../shadow_loom/settings.py).
* `UI_SOURCE_URL` — required by AGPL § 13 to expose the source link in
  the footer; set to the canonical upstream by default. Override if
  you publish a modified fork.

---

## 4. OAuth callback URLs

Every provider's redirect URI follows the same pattern:

    ${OAUTH_REDIRECT_BASE}/auth/<provider>/callback

| Provider | Console | Callback URL |
| --- | --- | --- |
| GitHub | <https://github.com/settings/developers> | `${OAUTH_REDIRECT_BASE}/auth/github/callback` |
| Google | <https://console.cloud.google.com/apis/credentials> | `${OAUTH_REDIRECT_BASE}/auth/google/callback` |
| Discord | <https://discord.com/developers/applications> | `${OAUTH_REDIRECT_BASE}/auth/discord/callback` |
| Microsoft | <https://portal.azure.com> → App registrations | `${OAUTH_REDIRECT_BASE}/auth/microsoft/callback` |
| Apple | <https://developer.apple.com/account/resources/identifiers/list/serviceId> | `${OAUTH_REDIRECT_BASE}/auth/apple/callback` (must be `https://` — Apple rejects `http://`, even on `localhost`) |

Add the exact URL above to each provider's **Authorized redirect
URIs** field. Only providers with both `_CLIENT_ID` and
`_CLIENT_SECRET` set will appear on the sign-in page; missing
providers are silently hidden.

---

## 5. First boot

Once the build is healthy, Render routes traffic. On first request:

* SQLModel creates all tables on the attached Postgres
  (`init_db` in [`shadow_loom/db.py`](../shadow_loom/db.py)).
* The example seeder
  ([`shadow_loom_ui/example_seeder.py`](../shadow_loom_ui/example_seeder.py))
  imports every fixture in [`example_worlds/`](../example_worlds/) and
  attaches the matching prose from [`sample_plots/`](../sample_plots/),
  creating one project per fixture under the built-in *example* user.
  This is **idempotent** — restart-safe — and runs to completion
  before any user request is served.
* OAuth `/auth/*` routes mount and the auth middleware activates.

Hit your Render domain, click **Sign in** with your configured
provider, and the dashboard should list every example world ready to
fork into your own account.

---

## 6. Picking models for production

The blueprint defaults every pipeline stage to OpenRouter
[`qwen/qwen3.6-35b-a3b`](https://openrouter.ai/qwen/qwen3.6-35b-a3b) —
the sparse-MoE 35B/3B-active model that local development also targets,
so the prompts are tuned for it. It accepts a 262K-token context,
streams structured output, costs ~$0.15/M in / $1/M out, and ships
under Apache 2.0. Setting `DEFAULT_MODEL` alone is enough — every
pipeline stage inherits it. Add per-stage overrides only when you want
them to differ:

```
DEFAULT_MODEL=openrouter:google/gemini-2.5-flash
GENERATION_MODEL=openrouter:anthropic/claude-3.5-sonnet
AUDITOR_MODEL=openrouter:anthropic/claude-3.5-sonnet
EXTRACTION_MODEL=openrouter:google/gemini-2.5-pro
QUERY_PARSING_MODEL=openrouter:anthropic/claude-3.5-haiku
```

If you swap to a short-output model (Claude 3.5 Sonnet caps at ~8K
output, GPT-4o at ~16K) **also lower the `*_MAX_TOKENS` env vars** —
the blueprint defaults (64000 / 16000 / 64000 / 16000) assume
Qwen3.6's full 262K window. See
[settings.md § 3](settings.md#3-generation-step-10--prose-rendering)
for the full per-stage knob reference. Any model on
<https://openrouter.ai/models> is accepted; the prefix `openrouter:`
is parsed by PydanticAI and the rest is the model id verbatim.

You can also point at OpenAI directly:

```
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=openai:gpt-4o
AUDITOR_MAX_TOKENS=8000
GENERATION_MAX_TOKENS=16000
```

Local Ollama is **not** reachable from a Render web service — there's
no GPU and no host bridge. Use OpenRouter or OpenAI for any deploy.

---

## 7. Custom domain

In Render's web service → **Settings** → **Custom Domains**, add your
domain and follow the DNS instructions. Once the domain is verified:

1. Update `OAUTH_REDIRECT_BASE` to the new URL.
2. Update every OAuth provider's redirect URI to match.
3. Trigger a manual redeploy so the change takes effect.

---

## 8. Operational notes

| Topic | Notes |
|---|---|
| **Persistent storage** | Postgres holds all state. The container filesystem is ephemeral on Render — never write user data to local disk. |
| **Logs** | Render dashboard → service → **Logs**. INFO level by default. |
| **Migrations** | Tables are created via `SQLModel.metadata.create_all` on startup — additive only. Schema changes that drop columns need an Alembic migration; one is not yet shipped. |
| **Backups** | Render Postgres has automated daily backups on every paid tier; restore from the database service's **Backups** tab. |
| **Cost** | OpenRouter usage dominates. Set a per-key spend cap in the OpenRouter dashboard. The audit loop is the biggest variable — see [settings.md § 5](settings.md#5-auditor-step-11--llm-as-judge-auditrefine-loop). |
| **Scaling** | The default is `numInstances: 1`. Session state is in-process (`_SESSION_STATES` in [`shadow_loom_ui/app.py`](../shadow_loom_ui/app.py)) — running multiple replicas needs sticky sessions or a shared session store. |

---

## 9. MCP server

Shadow-Loom's MCP server is a separate entry point
(`python -m shadow_loom_mcp`). Two deployment options:

* **Add a second Render web service** pointing at the same repo.
  Override the start command in **Settings → Docker** to
  `python -m shadow_loom_mcp`. Wire it to the same Postgres via
  `DATABASE_URL` from `shadow-loom-pg.connectionString`.
* **Run locally** pointing at the production database — set
  `DATABASE_URL` in your local `.env` to the Render Postgres external
  URL and run `python -m shadow_loom_mcp`.

API keys are issued from **Account → API Keys** in the deployed UI.
See [mcp-guide.md](mcp-guide.md) for the agent-facing contract.

---

## 10. Local production-parity smoke test

Before pushing, run the production image against a local Postgres:

```bash
docker build -t shadow-loom .

docker run --rm -p 7860:7860 \
    -e DATABASE_URL='postgresql://user:pass@host.docker.internal:5432/shadow' \
    -e STORAGE_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
    -e OAUTH_REDIRECT_BASE='http://localhost:7860' \
    -e OPENROUTER_API_KEY=sk-or-... \
    -e GITHUB_CLIENT_ID=... -e GITHUB_CLIENT_SECRET=... \
    -e DEFAULT_MODEL='openrouter:qwen/qwen3.6-35b-a3b' \
    shadow-loom
```

If the container comes up healthy and you can sign in on
`http://localhost:7860`, the Render deploy will work.

---

## See also

* [`Dockerfile`](../Dockerfile) — production image (shared with Railway).
* [`render.yaml`](../render.yaml) — Render Blueprint.
* [`.env.production.example`](../.env.production.example) — full env template.
* [railway-deployment.md](railway-deployment.md) — equivalent Railway guide.
* [settings.md](settings.md) — every runtime knob and tuning recipe.
* [mcp-guide.md](mcp-guide.md) — agent-facing MCP contract.
