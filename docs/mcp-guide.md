# MCP Guide — `shadow_loom_mcp`

Shadow-Loom ships a [Model Context Protocol](https://modelcontextprotocol.io)
server that exposes the entire causal narrative engine — ingestion, simulation,
generation, audit, version management — as **37 tools and 5 resources** that any
MCP-aware agent (Claude Desktop, Cursor, Continue, custom clients) can drive
directly.

The server is implemented on top of [FastMCP](https://github.com/jlowin/fastmcp)
and lives in [`shadow_loom_mcp/server.py`](../shadow_loom_mcp/server.py).

---

## 1. Quick start

```bash
conda activate shadow-loom
pip install -e .

# Run the server (stdio transport, FastMCP default)
python -m shadow_loom_mcp
```

The server reads its configuration from `config.env` /
[`shadow_loom/settings.py`](../shadow_loom/settings.py). Key settings:

| Env var | Purpose |
|---|---|
| `SHADOW_LOOM_DATABASE_URL` | SQLite / Postgres URL backing the version store. |
| `MCP_ALLOW_OPEN_MODE` | If `true`, falls back to the most-recently-cached token when a request arrives without an active context (dev/test only — fail-open). Default `false`. |
| `MCP_SKIP_AUDIT` | Default value of the `skip_audit` flag on `narrate` / `direct`. |
| `MCP_INGEST_FABULA_TIME_SPACING` | Spacing for fabula ticks during `ingest` (default 1000). |
| `MCP_INGEST_MAX_CORRECTION_RETRIES` | Programmatic-validation correction retries during `ingest`. |

### Connecting from Claude Desktop / Cursor

```json
{
  "mcpServers": {
    "shadow-loom": {
      "command": "python",
      "args": ["-m", "shadow_loom_mcp"],
      "env": {
        "SHADOW_LOOM_DATABASE_URL": "sqlite:///shadow_loom.db"
      }
    }
  }
}
```

---

## 2. Authentication & scopes

All requests carry a **bearer token** that maps to an `ApiKeyRow` in the
database. See [`shadow_loom_mcp/auth.py`](../shadow_loom_mcp/auth.py).

* Bearer token → cached `{user_id, scopes, key_id}` on first validation.
* Tools enforce one of three scopes:

| Scope | Tool groups |
|---|---|
| `read` | ORIENT + EXPLORE + REASON + JUDGE + `set_active_version` / `get_active_version` |
| `write` | CREATE + most MANAGE (branch, fork, delete, promote_branch, …) |
| `admin` | `share`, `update_project_tool` |

Resources (`world://…`) skip scope checks — they are intended for read-only
context priming. Every tool also checks **per-project membership** via
`check_project_access(...)` so a `read`-scoped key cannot read another user's
private project.

The fail-closed default means a tool returns `{"error": "missing scope: write"}`
rather than executing if the bearer token is unknown or under-scoped.

---

## 3. The 37 tools, by cognitive task

### ORIENT — "What stories exist? What's in this one?"

| Tool | Scope | Purpose |
|---|---|---|
| `list_projects` | read | List every project the user can access. |
| `open_project` | read | Returns a full **manifest**: entity/location/object/event/world-trait IDs with names, topology counts (including `channels` and `utterance_events`), `current_version` (the version actually loaded — honours the active-version pointer or an explicit `version=`) and `latest_version` (the project tip). **Always call this first.** |

### EXPLORE — "Tell me about this character / event / place / channel."

| Tool | Scope | Purpose |
|---|---|---|
| `inspect(node_id, at_time=None)` | read | Auto-routes by ID prefix (`ENT_`, `LOC_`, `EVT_`, `OBJ_`, `CHN_`, `WORLD_`). Pass `at_time` to time-slice an entity / world trait via `reconstruct_entity_at`. |
| `search(query, kinds=…)` | read | Fuzzy + substring search across nodes. |
| `get_relationships(entity_id)` | read | Affinity / fear / power / inertia table for an entity. |
| `list_channels(entity_id=None, medium=None)` | read | Enumerate `Channel` nodes with participants, intelligibility map, medium, and per-channel utterance count. |
| `get_channel_history(channel_id)` | read | Ordered list of `utterance` events on a channel, with speaker / addressees / `truth_value` / `content`. |
| `who_can_hear(speaker_id, at_time=None)` | read | Resolves the audience reachable from a speaker via current channels, weighted by per-recipient intelligibility. |
| `trace_causality(node_id, …)` | read | Walks `causal_topology` ancestors and descendants up to a depth. |
| `get_history(node_id)` | read | Timeline of state changes for an entity or world trait. |
| `list_branches()` | read | Walks the version DAG and returns one summary per branch (`world_id` `factual` / `shadow`, `branch_label`, root + head version, fork-point ancestor, version count). |
| `export_prose(branch_path=None)` | read | Returns prose across versions — either implicit linear order, or a specific lineage when `branch_path` (a list of `version_row_id`s) is supplied. |

### REASON — "Answer questions, run what-ifs without writing prose."

| Tool | Scope | Purpose |
|---|---|---|
| `ask(question)` | read | Routes through `parse_query` → `GeneralQuery` / `InterrogationQuery` (no prose, no version write). |
| `compute_tension(vector_id, …)` | read | Runs the affective scorers (mystery / irony / suspense / surprise) over the current graph for a given POV. |
| `diff_versions(v_a, v_b)` | read | Structured changeset between two versions. |

### CREATE — "Advance the story."

| Tool | Scope | Notes |
|---|---|---|
| `narrate(instruction, mode=None, skip_audit, force_implausible, speaker_id=None, addressee_ids=None, via_channel_id=None)` | write | The main creation entry point. NL → `parse_query` → `run_pipeline` → version write → active pointer advance. `mode` can pin the query type to `observe` / `intervene` / `counterfactual`. The `speaker_id` / `addressee_ids` / `via_channel_id` hints constrain the parser to emit a properly-typed `utterance` event with channel provenance. The parser also extracts optional **story-point anchors** (`temporal_anchor` / `syuzhet_anchor` / `anchor_after_event_id`) from the user's NL request so phrases like "after EVT_BANQUO_DEATH" or "in act 3" pin the query to the right slice without the caller looking the time up. |
| `direct(target_effect, entity_ids=…, intensity=0.8, temporal_anchor=None, syuzhet_anchor=None, anchor_after_event_id=None, …)` | write | Builds a `DirectiveQuery` directly (no NL parse) and runs the affective optimisation pipeline. The optional anchor arguments override `PipelineConfig.temporal_anchor` / `syuzhet_anchor` for a single call. |
| `write(prose, description="")` | write | Manual edit — supplies user prose, runs prose → topology re-extraction, merges into a new version (skips physics + LLM rendering). |
| `ingest(text, project_name, label=None)` | write | Creates a brand new project and runs the 5-step ingestion to produce v0. |

`narrate`, `direct`, and `ingest` emit FastMCP `progress_notifications` so an
MCP client can show a progress bar.

### JUDGE — "How good is this story?"

| Tool | Scope | Purpose |
|---|---|---|
| `evaluate(focus_entity_ids=…)` | read | Runs the full `EvaluationQuery` — collects all prose across versions, recomputes engine metrics, returns the `NarrativeOrderObject` scorecard. |
| `audit_log(version=None)` | read | Returns the auditor's structured loss for a given version. |

### RESEARCH — "Look up real-world background on a topic." (optional)

These tools are no-ops unless the operator has installed the optional
`[research]` extra (`pip install -e ".[research]"`) and configured a
provider. They never mutate `Entity` / `EventNode` / `RelationshipEdge`
/ `GlobalTrait` namespaces — results land only in
`WorldStateV1.world_facts`. Provider calls are cached **per-account**
so one user's lookups are never reused for another.

| Tool | Scope | Purpose |
|---|---|---|
| `research_topic(topic, provider=None, max_results=None)` | write | Calls the configured provider (default Tavily) for `topic`, distils the snippets through the `research_extraction` agent and persists a `WorldFact` against the project. Returns `{fact_id, summary, confidence, source_url_primary, related_node_ids, snippet_count, cached}`. |
| `list_world_facts()` | read | Enumerate every `WorldFact` attached to the project. |
| `delete_world_fact(fact_id)` | write | Remove a fact. The next version saved will exclude it from `WorldStateV1.world_facts`. |
| `get_research_status()` | read | Process-wide status: provider, enabled flag, API-key presence (boolean — never the key), default topics. Used by the UI Research tab to gate the run-now buttons. |
| `get_project_settings()` | read | Per-project settings — currently `research_topics: list[str]`. Settings live outside `WorldStateV1` so they never fork with shadow branches and never bloat version snapshots. |
| `set_project_settings(research_topics)` | write | Replace the project's `research_topics` list. Topics are stripped + de-duplicated. Pass `[]` to clear. Editing settings does not mutate any version. |

See [docs/research-extraction-plan.md](research-extraction-plan.md) for
the segregation contract and [CONTENT-POLICY.md §6.4a](../CONTENT-POLICY.md)
for the user-facing guarantees.

### MANAGE — "Edit the metadata, branch, fork, share, delete."

| Tool | Scope |
|---|---|
| `branch`, `fork`, `share` | write / write / admin |
| `update_project_tool` | admin |
| `delete_project`, `delete_version`, `reparent_version` | write |
| `promote_branch(version_row_id)` | write |
| `set_active_version`, `get_active_version` | read / read |

`set_active_version` updates the per-user `ActiveVersionRow` pointer, so every
subsequent `narrate` / `inspect` / etc. resolves against that version unless
overridden by an explicit `version=` argument. The same pointer is read by
the NiceGUI workspace, so an agent and a human author always converge on the
same branch tip.

`promote_branch` copies a shadow-branch version onto a new factual `VersionRow`
so the chosen counterfactual becomes canon while the shadow source stays
browsable for diffing.

---

## 4. The 5 resources

```
world://projects                                 # list of projects + summaries
world://project/{project_id}                     # project metadata + version list
world://project/{project_id}/world               # full WorldStateV1 JSON
world://project/{project_id}/entity/{entity_id}  # one entity (with timeline)
world://project/{project_id}/versions            # version tree (ancestry)
```

Resources are designed for **context priming** — point your agent at
`world://project/{id}/world` once at the start of a conversation and it has
the full graph in scope without burning a tool call per inspection.

---

## 5. The auto-versioning contract

Every write tool funnels through
[`shadow_loom_mcp/helpers.py::run_and_save`](../shadow_loom_mcp/helpers.py):

1. Resolve the **ancestor** (`ancestor_row_id`) — either the user's active
   version or an explicit `version=` argument.
2. Run `run_pipeline(query, versioned_model=...)`.
3. If `result.implausible` → return the explanation, **do not write a version**.
4. If `result.reextraction_failed` → return the prose with a warning, **do not
   write a version** (prose and world model are out of sync).
5. If the merged world model is byte-identical to the ancestor (for example a
   `direct` call that produced no graph change) → return the prose with a
   `world_model_unchanged` / `version_skipped` flag, **do not write a version**.
6. Otherwise `save_version(...)` with `ancestor_id = ancestor_row_id` and call
   `set_active_version(...)` so the user's pointer follows the new tip.

The response always includes `version`, `ancestor_id`, `prose`,
`physics_state`, the auditor verdict, **and a `branch` envelope**
(`{world_id, branch_label, ancestor_id}`) so the agent can tell whether the
new version landed on factual mainline or on a shadow fork — enough for the
agent to decide what to do next without re-querying.

---

## 6. Typical agent workflow

```
list_projects
  └── open_project(project_id=…)            ← manifest, IDs, current version
        └── inspect(node_id="ENT_MACBETH")  ← deepen context
        └── ask("Who knows about the dagger?")
        └── narrate("Macbeth hesitates outside Duncan's chamber")
              └── (server: parse → physics → render → audit → save → set_active)
        └── direct("dramatic_irony", entity_ids=["ENT_LADY_MACBETH"], intensity=0.9)
        └── evaluate(focus_entity_ids=["ENT_MACBETH"])
        └── branch(version=4, label="bloodier-ending")
```

---

## 7. Error surface

Every tool is wrapped in `_safe_tool`, which catches unhandled exceptions and
returns `{"error": "<tool_name> failed: <message>"}` instead of bubbling a
stack trace through the transport. Common error shapes:

```json
{ "error": "missing scope: write" }
{ "error": "Project 7 not accessible by user 3" }
{ "error": "Query parsing failed",
  "reasoning": "...", "validation_errors": [{"field":"interventions","message":"..."}] }
{ "implausible": true,
  "implausibility_reason": "Rung-2 intervention is impossible: ...",
  "implausibility_details": {"unresolved_targets": [...]} }
```

The `implausible` envelope is **not** an error — the request was understood but
the engine refused to mutate the world. Callers should surface the reason and
optionally retry with `force_implausible=true`.

---

## 8. Tests

`tests/test_mcp_server.py` exercises every tool against an in-memory SQLite DB
with a mocked Context. The fail-closed auth path, the per-project membership
check, the `run_and_save` versioning contract, and the implausibility envelope
all have dedicated test cases.

```bash
python -m pytest tests/test_mcp_server.py -q
```

---

## See also

* [pipeline-walkthrough.md](pipeline-walkthrough.md) — what happens **inside** a `narrate` / `direct` / `write` / `ingest` call once `run_and_save` invokes the pipeline.
* [query-and-cycles.md](query-and-cycles.md) — the eight typed queries `narrate` parses NL into, and the per-type cycle each runs.
* [architecture.md §10](architecture.md) — the conceptual summary of the MCP surface plus links into the rest of the engine.
* [ui-guide.md](ui-guide.md) — the human-facing equivalent of the same tools, useful when debugging an agent's session against your own eyes.
* [design-decisions.md](design-decisions.md) — *why* every write goes through the implausibility / re-extraction gate before persisting a version.
