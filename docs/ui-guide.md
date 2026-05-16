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

The workspace is composed of a left-hand **version sidebar**, a top-level
**chat / command bar**, a dedicated **Answer panel** (above the chat bar,
for read-only Q&A results), and ten cross-linked tabs defined in
[`components/workspace.py`](../shadow_loom_ui/components/workspace.py):

```
story · explorer · world · social · affective · reasoning · audit · research · editor · export
```

State is centralised in [`state.py::AppState`](../shadow_loom_ui/state.py),
a pub/sub event bus. Every tab subscribes to the events it cares about
(`PROJECT_LOADED`, `WORLD_STATE_CHANGED`, `FABULA_CURSOR_CHANGED`,
`SYUZHET_CURSOR_CHANGED`, `VERSION_CHANGED`, `ACTIVE_PATH_CHANGED`,
`PIPELINE_RESULT`, `QUERY_STARTED`).

Every panel header carries a clickable info-icon **help popover**
([`components/help_popover.py`](../shadow_loom_ui/components/help_popover.py))
that opens a Markdown reference for that surface — what the panel does,
how to read its diagrams, what each control means, and what does *not*
appear there. The popovers are intended as in-product documentation that
tracks the code; the equivalent material is reproduced here for
reference.

---

## Version sidebar (always visible)

[`components/version_sidebar.py`](../shadow_loom_ui/components/version_sidebar.py)

* Vertical tree of every `VersionRow` in the project, rooted at the
  original ingestion. Branches reflect manual edits, pipeline runs, and
  counterfactual branches. **Factual** rows render in green; **shadow**
  branches (the persisted forks produced by counterfactual queries under
  `branch_policy=auto`) render in violet. The legacy radial layout has
  been removed; the tree is the only view.
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

## Chat / command bar and Answer panel

The chat bar at the bottom of the workspace dispatches one of the typed
query objects (see [query-and-cycles.md](query-and-cycles.md)). The
**Interrogation** mode is read-only:

* `interrogate` — targeted questioning of an entity, event, or belief
  ("who knows what at this point?", "why does Y act?").

Read-only queries do **not** create a new `VersionRow`. Their result is
rendered in the dedicated **Answer panel**
([`components/answer_panel.py`](../shadow_loom_ui/components/answer_panel.py))
that sits directly above the chat bar: a card with the model's claim, a
confidence badge (🟢 ≥ 70 / 🟡 40–69 / 🔴 < 40), an evidence-id list
(linking back to the graph nodes consulted), and any caveats. The panel
clears on `VERSION_CHANGED` and `PROJECT_LOADED` so a stale answer never
lingers across versions.

Write-mode queries (`observation` / `intervention` / `counterfactual` /
`directive` / `evaluate` / `manual_edit`) take their normal pipeline
route and surface in whichever tab consumes their result
(Story / Reasoning / Audit / Affective).

> *Developer note*: the underlying `GeneralQuery` model still exists in
> [`shadow_loom/query_models.py`](../shadow_loom/query_models.py) and is
> retained as a last-resort fallback inside the parser, but the chat bar
> no longer offers an explicit **Ask** mode \u2014 free-form questions
> should be issued in **Interrogation** mode instead.

Read-only queries do **not** create a new `VersionRow`. Their result is
rendered in the dedicated **Answer panel**
([`components/answer_panel.py`](../shadow_loom_ui/components/answer_panel.py))
that sits directly above the chat bar: a card with the model's claim, a
confidence badge (🟢 ≥ 70 / 🟡 40–69 / 🔴 < 40), an evidence-id list
(linking back to the graph nodes consulted), and any caveats. The panel
clears on `VERSION_CHANGED` and `PROJECT_LOADED` so a stale answer never
lingers across versions.

Write-mode queries (`observation` / `intervention` / `counterfactual` /
`directive` / `evaluate` / `manual_edit`) take their normal pipeline
route and surface in whichever tab consumes their result
(Story / Reasoning / Audit / Affective).

## 1. Story tab

[`components/story_tab.py`](../shadow_loom_ui/components/story_tab.py)

* Shows the raw narrative text in a textarea + a "Save & Re-ingest" button.
* Clicking re-ingest opens a confirmation dialog, runs the full
  [`pipeline.run_pipeline`](../shadow_loom/pipeline.py), and persists the
  result as a new version with `source="ingestion"` and the previous
  version as ancestor.
* This tab is **not** autosaved — the textarea is plain prose, not graph.
* Read-only Q&A queries (`interrogate`) and full-story
  `evaluate` runs are deliberately filtered out of this tab's
  `PIPELINE_RESULT` listener: the prose feed only re-renders when the
  result actually carries new *story* prose, so a question or an audit
  report never perturbs the lineage view. (Evaluation reports surface
  on the Audit tab.)

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

### View modes

The toolbar `view_mode` toggle switches the main panel between several
lenses on the same time-sliced world state. The mode keys are stable
(used in deep links / state) even when labels change.

| Mode key | Toolbar label | What it shows |
| -------- | ------------- | ------------- |
| `overview` | Overview | High-level world graph + status / population summaries. |
| `social` | Social | Relationship graph between entities. Heatmap metric (`affinity` / `fear` / `power_dynamic`) and layout (`force` / `circular`) selectors appear in the toolbar. Tick **Animate over fabula time** to swap the static entity×entity heatmap for a timeline-scrubber heatmap that replays the matrix tick-by-tick (causal-aware: layers `mutation_social` causal edges and authored snapshots on top of the steady-state `RelationshipEdge` baseline). |
| `spatial` | Spatial | Location graph with paths between rooms / regions and current entity positions. Animated location nodes (rippleEffect) toggleable. The **Map** sub-tab additionally renders an ECharts force graph of locations, fabula-anchored entities, narrative objects (held or on the floor), active channel arcs, and **event glyphs** (★) at every windowed event's `at_location_id`. Bound participants on the page get a yellow border (correct co-presence); displaced bound actors get a red dashed border (`event_copresence_violation` candidate); channel-mediated addressees stay normal-bordered (the dashed amber arc encodes their virtual presence). Toolbar switches: *Entities*, *Objects*, *Channels*, *Locked edges*, *Events ★*, plus *Channel window ±* and *Event window ±* sliders that widen the on-tick rule to surface utterances / events near the cursor. See [design-decisions.md §D22](design-decisions.md#d22-events-have-an-explicit-spatial-anchor-eventnodeat_location_id). |
| `information` | Information | Standing communication channels (telephones, mind-links, classified pipelines) and which entities can transmit / overhear / are deaf to them. |
| `ego` | Ego-Graph | World filtered to one or more focus entities — just what they can plausibly perceive, hear, or remember at the cursor. |
| `temporal` | Temporal | A single entity's full trajectory (status, location, traits, beliefs) across fabula time. |
| `composition` | Composition | Trait-vector composition breakdowns + global `WorldTrait` magnitude/inertia bars + sunburst / treemap composition views. |
| `epistemic` | **Character Beliefs** | Tiled grid of per-character belief panels — one card per believer showing perceived state, target, confidence, inertia, provenance (utterance / event / channel), and the fabula tick the belief was established at. The toolbar `Believers` multi-select filters which characters are shown. |
| `world_state` | **World State** | Tiled grid of per-`GlobalTrait` **snapshot cards** at the current fabula cursor — one card per world trait. Each card shows the trait's magnitude (with a 0–1 bar coloured by band: low / moderate / dominant), inertia, category, affected domains, prose description, and a compact inline sparkline of the magnitude trajectory. Drag the fabula slider to update every card in lockstep — the snapshot reading dominates, the sparkline gives trajectory context. The toolbar `World traits` multi-select filters which traits are shown. |
| `relationships` | **Relationships** | Surfaces the `RelationshipEdge` axes (`affinity`, `fear`, `power_dynamic`) that were previously buried inside the Social view. Two stacked panels: (1) the entity×entity heatmap of the chosen metric, static or animated over fabula time (same controls as Social — `Heatmap metric` selector and `Animate over fabula time` checkbox); (2) per-dyad **snapshot cards** showing all three axes side-by-side as labelled bars (signed bars for affinity / power, unsigned for fear), with each axis's value and per-axis inertia. Unobserved axes render as a faint "—" so deliberately-zero is distinguishable from no-data. |
| `comparison` | Comparison | Side-by-side trait / relationship table for 2–6 picked entities (radar overlay + grouped trait bars + ranked table). |

Heavy panels are always rendered against the snapshot returned by
`snapshot_world_at(ws, fabula_cursor)` so dragging the cursor scrubs
beliefs, world-trait magnitudes, and topology consistently with the
rest of the UI.

## 4. Causality tab

Removed. The top-level Causality tab was retired in May 2026; its
sub-views were promoted to first-class tabs:

* **Topology / Evolution** — the causal Sankey, trait trajectories, and
  per-event causal-graph snapshot now live inside the **Social** and
  **World** tabs (animated relationships, snapshot cards).
* **Affective Dashboard** — promoted to its own top-level **Affective**
  tab (see below).

[`components/causality_tab.py`](../shadow_loom_ui/components/causality_tab.py)
is kept only as an internal builder library that exposes the affective
rendering helpers; it is not mounted as a workspace tab.

## 5. Reasoning tab

[`components/reasoning_tab.py`](../shadow_loom_ui/components/reasoning_tab.py)

* Visualises the most recent intervention / counterfactual reasoning trace
  (mutations, blocked propagations, abduction back-fills, hidden deltas).
* Uses [`reasoning_helpers.py`](../shadow_loom_ui/reasoning_helpers.py) to
  format the `CausalPhysicsResult` payload.

## 6. Audit tab

[`components/audit_tab.py`](../shadow_loom_ui/components/audit_tab.py)

* Displays the structured `AuditResult` (or, for full feedback-loop runs, the per-iteration `AuditCycleSnapshot.audit_result` inside the `FeedbackLoopResult`) for the latest generated scene.
* Causal / abductive / affective sections each show their loss + offending
  prose spans.
* **Layout: text first, charts underneath.** Each scorecard renders
  textual findings (Quality Synthesis directives → Causal findings →
  Affective findings) in a stack of cards, then a single "Supporting
  charts" card holds the gauges (foreshadowing / plausibility /
  emotional trajectory) below. The per-query Audit Loop block follows
  the same convention: per-iteration cycle prose, pass-rate table and
  violations list render before the pass-rate pictorial and
  convergence trajectory chart.
* **Per-tile help popovers.** Every hero score tile (Cognitive
  plausibility, Foreshadowing, Emotional fit) and every diagnostic row
  (Quality thresholds, Achieved intensity, Audit pass-rate per
  iteration, Convergence trajectory) carries an inline ⓘ popover that
  explains what the metric measures, the score bands (`strong` /
  `needs work` / `weak`), and how to read the chart. Click the icon
  to read the in-product documentation; click anywhere else to
  dismiss.
* **Failure diagnostics surfaced inline.** When the feedback loop
  bypass-passes (failed-open auditor) or aborts on a generation /
  refinement LLM failure, the `correction_error` is rendered as an
  amber "Auditor diagnostic" row right under the Converged badge.
  When `result.reextraction_failed` is true (Step 6 skipped because
  the renderer produced a `[Generation failed: ...]` placeholder
  scene) a "World model not updated" amber row appears with the
  `reextraction_error`. The chat card shows the same diagnostics so
  users see them without opening the Audit tab.
* Iteration numbers in cycle headers and chart x-axes are 1-indexed
  (the data model stores them 0-indexed; the UI converts).
* Note on cognitive plausibility: a character holding a belief that is
  contradicted by reality is treated as *dramatic irony*, not as a
  plausibility violation. The deterministic
  `cognitive_plausibility_score` only drops when an entity's *actions*
  are inconsistent with their *own* established beliefs (the LLM-side
  `NarrativeOrderObject` check); contradicted-belief counts are
  reported as informational context.

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
   state.current_version_row_id, user_id=state.user_id, world_id=…,
   branch_label=…)` — the branch identity is read from
   `state.head_branch()` so an edit on a shadow row stays on that
   shadow branch instead of silently demoting to factual mainline.
   The new version is a child of the current one; the previous
   version is preserved.
6. `state.load_db_version(new_ws, new_ver.id, version_number=new_ver.version)`
   atomically swaps the active world; every tab re-renders.
7. `db.set_active_version(...)` mirrors the new version into the
   active-version pointer so the MCP server's read tools default to it.

> **Raw vs projected.** The Editor renders and saves the **raw**
> (un-projected) world state — i.e. the persisted JSON with its
> factual `entities` baseline + `shadow_entities` sidecar visible as
> distinct top-level fields. Other tabs (Explorer, World, Story)
> read the **projected** view, which layers the active shadow
> branch's AMWN-split clones over the factual baseline so an
> Inspector lookup on a shadow row returns the do(·)-modified
> entity. Dumping the projected view to JSON would overwrite the
> factual baseline for every cloned id; the Editor / Export paths
> route through `AppState.raw_world_state` to keep the persisted
> snapshot round-trippable.

Permission rules match the Story tab: project owner OR project role
`editor` / `admin`. Users without write access see a read-only textarea
and no Save button.

## 8. Research tab

[`components/research_tab.py`](../shadow_loom_ui/components/research_tab.py)

* Manage research **topics**, run background lookup tasks, and browse the
  resulting **facts** (each carrying a source URL and a confidence score).
* **Segregation policy**: research is a separate store. Facts are not
  written into the world model automatically; they are reference material
  available to the author and (optionally) to the renderer's prompt
  scaffolding. Promoting a research finding into canon is a deliberate,
  manual step taken in the Editor tab.
* A status strip at the top of the tab shows in-flight lookups, the
  number of stored facts per topic, and the last refresh time. The Run
  buttons are gated on `settings.core.tavily_api_key` being present
  *and* `settings.extraction.enable_research_agent` being true; if
  either is missing the buttons stay disabled with an explanatory
  tooltip.
* Snippets surfaced in the topic preview render the provider's
  `content` field (the long-form excerpt). The cached
  `raw_snippets_json` blob serialises each snippet via
  `model_dump(mode="json")` so `datetime` fields round-trip safely
  through `json.dumps`.

## 9. Export tab

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
| `FABULA_CURSOR_CHANGED` | sliders in world / social tabs | world tab, social tab, affective dashboard |
| `SYUZHET_CURSOR_CHANGED` | reasoning tab, affective dashboard | world tab, social tab, affective dashboard |
| `ACTIVE_PATH_CHANGED` | top tabs, sub-tab controllers | gated panels for "render only when visible" |

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

* [architecture.md §9](architecture.md) — the conceptual map of the UI's nine tabs and the `AppState` event bus.
* [pipeline-walkthrough.md](pipeline-walkthrough.md) — what happens behind the scenes when the **Story** tab issues a query.
* [query-and-cycles.md](query-and-cycles.md) — the typed query taxonomy that the chat box and the **Reasoning** tab build, including the read-only `interrogate` cycle served by the **Answer panel**.
* [mcp-guide.md](mcp-guide.md) — the agent-facing equivalent of the workspace; the active version pointer is shared so the UI and an MCP client always see the same tip.
* [use-cases.md](use-cases.md) §5 — the **Editor** tab and the manual-editing workflow.
* [paper/shadow_loom.pdf](../paper/shadow_loom.pdf) **Appendix D** (`app:ui`) — the same surfaces described tab-by-tab with example sessions on bundled fixtures (Macbeth, Death on the Nile, Reservoir Dogs, Romeo and Juliet, Gone Girl).
* [paper/shadow_loom.pdf](../paper/shadow_loom.pdf) **Appendix B** (`app:walkthrough`) — an end-to-end narrative walkthrough of the pipeline that the UI exposes, using the *Macbeth* fixture from Step 0 (the user typing a query) through Step 8 (audit, re-extraction, and merge).
