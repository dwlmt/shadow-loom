# UI Guide

The Shadow-Loom UI is a NiceGUI workspace
([`shadow_loom_ui/app.py`](../shadow_loom_ui/app.py)) reachable on
`http://localhost:7860` (override with `UI_PORT`):

```bash
python -m shadow_loom_ui
```

Query results, intervention runs, and Editor saves are **autosaved** as new
version rows; there is no longer an explicit workspace-level Save button.
Manual saves still exist on the Story tab (Save & Re-ingest) and the Editor
tab (Save / Save Anyway), as documented below.

The workspace is composed of a left-hand **version sidebar** and eight tabs
defined in [`components/workspace.py`](../shadow_loom_ui/components/workspace.py):

```
story · explorer · world · causality · reasoning · audit · editor · export
```

State is centralised in [`state.py::AppState`](../shadow_loom_ui/state.py),
a pub/sub event bus. Every tab subscribes to the events it cares about
(`PROJECT_LOADED`, `WORLD_STATE_CHANGED`, `FABULA_CURSOR_CHANGED`,
`SYUZHET_CURSOR_CHANGED`, `VERSION_CHANGED`, `ACTIVE_PATH_CHANGED`).

---

## Version sidebar (always visible)

[`components/version_sidebar.py`](../shadow_loom_ui/components/version_sidebar.py)

* Tree of every `VersionRow` in the project, rooted at the original
  ingestion. Branches reflect manual edits, pipeline runs, and counterfactual
  branches. **Factual** rows render in green; **shadow** branches (the
  persisted forks produced by counterfactual queries under
  `branch_policy=auto`) render in violet.
* Click a row to swap the active version (`AppState.load_db_version`).
  Cursors are reset; every tab re-renders.
* Toolbar:
  * **Delete** — rejoin-aware (`db.delete_version` re-parents children).
  * **Reparent** — change a version's `ancestor_id` (rejected if it would
    multi-root the tree).
  * **Promote to canon** (shadow rows only) — calls `db.promote_branch` /
    the matching MCP tool to copy the shadow version onto a new factual
    `VersionRow`.
  * **Diff against factual** (shadow rows only) — opens a structured diff
    against the current factual head.

The active pointer is mirrored to `ActiveVersionRow` so the MCP server
shows the same version as the UI for that user.

---

## 1. Story tab

[`components/story_tab.py`](../shadow_loom_ui/components/story_tab.py)

* Shows the raw narrative text in a textarea + a "Save & Re-ingest" button.
* Clicking re-ingest opens a confirmation dialog, runs the full
  [`pipeline.run_pipeline`](../shadow_loom/pipeline.py), and persists the
  result as a new version with `source="ingestion"` and the previous
  version as ancestor.
* This tab is **not** autosaved — the textarea is plain prose, not graph.

## 2. Explorer tab

[`components/explorer_tab.py`](../shadow_loom_ui/components/explorer_tab.py)

* Tree view of every node in the world model grouped by category
  (entities, locations, objects, events, world traits) plus an inspector
  panel for the selected node.
* Read-only — for write access use the **Editor** tab.

## 3. World tab

[`components/world_tab.py`](../shadow_loom_ui/components/world_tab.py)

* Cytoscape graph visualisation of the active world (entities + locations +
  objects + edges).
* Fabula-time slider; node colours / sizes update via
  `state.set_fabula_cursor` (debounced 120 ms).
* Gated by `is_path_visible("world")` so it only re-renders when visible.

## 4. Causality tab

[`components/causality_tab.py`](../shadow_loom_ui/components/causality_tab.py)

Three sub-tabs:

* **Topology** — Sankey of causal flow up to the fabula cursor.
* **Evolution** — trait trajectory plots per entity, with cursor needle.
* **Affective Dashboard** — suspense / mystery / irony / surprise / emotion
  gauges and time-series, computed by `viz_helpers.compute_affective_scores`
  (which routes through `DirectiveAssembler.compute_*_score`). Top-20
  entities by event degree are shown by default.

Heavy panels go through `state.spawn_panel_task` — rapid scrubs collapse to
the most recent render.

## 5. Reasoning tab

[`components/reasoning_tab.py`](../shadow_loom_ui/components/reasoning_tab.py)

* Visualises the most recent intervention / counterfactual reasoning trace
  (mutations, blocked propagations, abduction back-fills, hidden deltas).
* Uses [`reasoning_helpers.py`](../shadow_loom_ui/reasoning_helpers.py) to
  format the `CausalPhysicsResult` payload.

## 6. Audit tab

[`components/audit_tab.py`](../shadow_loom_ui/components/audit_tab.py)

* Displays the structured `AuditReport` for the latest generated scene.
* Causal / abductive / affective sections each show their loss + offending
  prose spans.

## 7. Editor tab — **manual world-model editing**

[`components/editor_tab.py`](../shadow_loom_ui/components/editor_tab.py)

The editor has three integrated surfaces, all backed by a single JSON
textarea (the source of truth).

### 7a. Validate button (live feedback)

Runs the same `_programmatic_validation` as the ingestion pipeline against
the current textarea contents and renders results inline:

* **Green panel** — counts (entities / events / locations / causal edges).
* **Red panel** — errors with `[category] detail` for each issue. **Errors
  block saving.**
* **Amber panel** — warnings (orphans, missing mutation coverage, low
  info-edge density, dead-actor references). Allowed but flagged.

The same validation is auto-run after every structural add / remove.

### 7b. Structural editor (collapsible "Structural editor — add / remove items")

One panel per collection, with item rows showing summaries:

* **Locations** — `LOC_*`
* **Entities** — `ENT_*`
* **Objects** — `OBJ_*`
* **Events** — `EVT_*` (event_type `choice` / `outcome` / `revelation` /
  `utterance`; utterance events expose `speaker_id`, `addressee_ids`,
  `via_channel_id`, `truth_value`, and `content` fields in the add
  dialog and validate that the speaker and addressees resolve)
* **World Traits** — `WORLD_*`
* **Causal Edges** — `source_id → target_id [causality_type]`
* **Relationship Edges** — `source ↔ target  affinity / fear / power`
* **Spatial Edges** — `source → target  locked=…`
* **Channels** — `participants  medium=…  intelligibility=…` (the
  speech-act layer that replaces the legacy `InformationEdge` collection)

Each row has a **delete** button → confirmation dialog warning about
broken references → removal → re-validate.

Each panel has an **add** button → small dialog collecting the minimum
required fields:

* IDs auto-prefixed and uppercased (e.g. typing `"foo bar"` in an event ID
  field becomes `EVT_FOO_BAR`).
* Events: `fabula_time` and `syuzhet_index` auto-suggested from the next
  free slot. Choosing `event_type=utterance` makes `speaker_id` and at
  least one `addressee_id` mandatory.
* Causal edges: enum-validated `causality_type`; both endpoints required.
* Channels: comma-separated `participant_ids` parsed into a list; the
  `intelligibility` map is editable as JSON.

All builders raise `ValueError` on missing or invalid fields, surfaced as
toasts. Newly-added items pass `WorldStateV1.model_validate` round-trip.

### 7c. JSON textarea (full-fidelity manual edit)

For fields the structural editor doesn't expose (snapshot trait values,
beliefs, ambient state, evidence_strength, etc.) the user edits the JSON
directly. A live character counter and live validation feedback keep the
loop tight.

### Save flow

Clicking **Save** runs:

1. Pydantic schema validation (`WorldStateV1.model_validate_json`). Errors
   → toast with first three issues, save aborted.
2. `_programmatic_validation` consistency check.
3. If errors → blocking dialog listing every issue, no save button.
4. If warnings only → confirmation dialog summarising counts + warnings,
   button reads **"Save Anyway"**.
5. On confirm: `db.save_version(... source="manual_edit", ancestor_id=
   state.current_version_row_id, user_id=state.user_id)`. The new version
   is a child of the current one; the previous version is preserved.
6. `state.load_db_version(new_ws, new_ver.id, version_number=new_ver.version)`
   atomically swaps the active world; every tab re-renders.
7. `db.set_active_version(...)` mirrors the new version into the
   active-version pointer so the MCP server's read tools default to it.

Permission rules match the Story tab: project owner OR project role
`editor` / `admin`. Users without write access see a read-only textarea
and no Save button.

## 8. Export tab

[`components/export_tab.py`](../shadow_loom_ui/components/export_tab.py)

* Download the current world as `world_state.json`.
* Download the version tree as a structured JSONL file.
* Copy the active version row ID for use in MCP / scripts.

---

## Cross-tab events

The events most useful when extending a tab:

| Event | Emitted by | Listened to by |
|---|---|---|
| `PROJECT_LOADED` | project picker, MCP `open_project` | every tab — full re-render |
| `WORLD_STATE_CHANGED` | `load_db_version`, manual save, pipeline | every tab + cache invalidation in `viz_helpers` |
| `VERSION_CHANGED` | `load_db_version` | version sidebar highlight, story tab raw-text reload |
| `FABULA_CURSOR_CHANGED` | sliders in world / causality tabs | world tab, causality sub-tabs |
| `SYUZHET_CURSOR_CHANGED` | reasoning tab, causality affective dashboard | causality affective dashboard |
| `ACTIVE_PATH_CHANGED` | top tabs, causality sub-tabs | gated panels for "render only when visible" |

---

## Performance discipline

* **Debounce cursor writes.** `state.set_fabula_cursor` debounces emit by
  120 ms via `_schedule_cursor_emit` (probes for a running loop first to
  avoid coroutine-never-awaited warnings in unit tests).
* **Gate by `active_path`.** Heavy panels keep a `_dirty` flag and only
  rebuild on `ACTIVE_PATH_CHANGED` if dirty + visible.
* **Cancel previous panel rebuilds.** `state.spawn_panel_task(panel_id,
  coro)` cancels any prior task with the same panel id, collapsing rapid
  scrubs to the latest render.
* **Reuse chart handles.** ECharts panels build the chart once and patch
  options in place via `viz.update_chart_options(chart, opts)` — no
  `container.clear()` on each scrub.

These are documented in `/memories/repo/architecture.md` "Timeline-control
hang fixes".

---

## See also

* [architecture.md §9](architecture.md) — the conceptual map of the UI's eight tabs and the `AppState` event bus.
* [pipeline-walkthrough.md](pipeline-walkthrough.md) — what happens behind the scenes when the **Story** tab issues a query.
* [query-and-cycles.md](query-and-cycles.md) — the eight query types that the chat box and the **Reasoning** tab build.
* [mcp-guide.md](mcp-guide.md) — the agent-facing equivalent of the workspace; the active version pointer is shared so the UI and an MCP client always see the same tip.
* [use-cases.md](use-cases.md) §5 — the **Editor** tab and the manual-editing workflow.
* [paper/shadow_loom.pdf](../paper/shadow_loom.pdf) **Appendix D** (`app:ui`) — the same eight tabs described tab-by-tab with example sessions on bundled fixtures (Macbeth, Death on the Nile, Reservoir Dogs, Romeo and Juliet, Gone Girl).
* [paper/shadow_loom.pdf](../paper/shadow_loom.pdf) **Appendix B** (`app:walkthrough`) — an end-to-end narrative walkthrough of the pipeline that the UI exposes, using the *Macbeth* fixture from Step 0 (the user typing a query) through Step 8 (audit, re-extraction, and merge).
