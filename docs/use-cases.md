# Use Cases

Shadow-Loom is built around a single capability: **treat a story as a graph
with physics**, then use that graph to answer questions, generate scenes, or
audit existing prose. The capabilities below all fall out of that core idea.

---

## 1. AI-assisted long-form fiction

**Audience:** novelists, screenwriters, game writers, IF authors.

The author writes a chapter; Shadow-Loom ingests it into a `WorldStateV1`.
From there:

* They ask the system *"what could plausibly happen next that maximises
  dramatic irony for the reader?"* — Shadow-Loom enumerates candidate
  interventions, rejects the ones that violate causal physics or affordance
  rules, ranks the survivors by the affective scorer, and returns a
  `CreativeBrief` plus a constrained-LLM rendered scene.
* They edit a scene by hand → re-ingest → diff against the previous version
  in the version tree.
* They protect against continuity errors: the auditor flags miracle steps
  ("character X teleported"), broken belief states ("Y knew this in chapter
  3 but acts ignorant in chapter 5"), and dead-actor violations.

This is the primary use case driving the architecture.

## 2. Story-aware QA / continuity checking

**Audience:** editors, writing rooms, game-narrative engineers shipping a
patch.

Even without using the generative pipeline, the ingestion + validation layer
on its own is useful:

* `_programmatic_validation` (re-exposed in the **Editor** tab) catches
  broken IDs, dangling edge endpoints, snapshot `triggered_by` pointing at
  events that no longer exist, syuzhet/fabula contradictions, and dead-actor
  references.
* The LLM auditor catches semantic contradictions humans miss in long texts
  (a character changing eye colour, a contradictory backstory mentioned 200
  pages apart, etc.).
* Versioning + diff lets editors verify which scene caused which trait shift.

## 3. Counterfactual exploration

**Audience:** literary analysts, screenwriting students, game designers
balancing branching narratives.

Pearl rung-3 counterfactuals in a literary setting:

* *"What if Macbeth had refused to kill Duncan?"* — `do(EVT_DUNCAN_MURDER =
  ⊘)` and replay forward physics. The system reports which downstream events
  collapse, which character traits diverge, and which beliefs never form.
* The AMWN sandbox tags everything `world_id="shadow"` so the canonical
  graph is never polluted.
* For game designers this means programmatically estimating which player
  choices have meaningful narrative consequences vs which are cosmetic.

## 4. Suspense / tension shaping

**Audience:** anyone who needs to dial the emotional intensity of a scene.

Because suspense, mystery, dramatic irony and surprise are computable
quantities (see [academic-foundations.md §3.1 (Wilmot & Keller suspense)](academic-foundations.md#31-suspense-as-uncertainty-reduction--wilmot--keller-acl-2020),
[§3.3 (KL surprise)](academic-foundations.md#33-surprise-as-kl-divergence),
[§3.4 (dramatic irony)](academic-foundations.md#34-dramatic-irony-as-epistemic-asymmetry)),
the user can:

* Ask *"raise the suspense in chapter 7 to 0.8 without revealing the
  betrayal"* and the directive assembler will choose interventions that move
  the threat-vs-hope ratio while protecting the relevant
  `InformationEdge.discovered_at_syuzhet` constraints.
* Compare two manuscripts on the same affective axes.
* Build heatmaps of suspense, irony and surprise across an existing book —
  see the **Causality → Affective Dashboard** in the UI.

## 5. World-model authoring (manual structural editing)

**Audience:** worldbuilders, tabletop GMs, transmedia franchise editors who
want a typed canon they can query.

The **Editor** tab in the UI lets a user fix or extend a world model
directly:

* Add / remove entities, locations, objects, events, world traits, and
  every edge type via small dialogs.
* Edit JSON directly for fields the structural editor doesn't expose
  (snapshot trait values, beliefs, ambient state).
* The **Validate** button runs the same consistency checks as ingestion
  before saving; errors block, warnings can be acknowledged.
* Saves are versioned with the previous version as ancestor — full
  history preserved.

This is also the recovery path when the LLM ingestion misses something
subtle — the human can correct one entity and the rest of the model stays
intact.

## 6. Story dataset generation

**Audience:** ML researchers training narrative-aware language models.

Because every generated scene comes with:

* the typed graph it was generated from,
* the `CreativeBrief` it satisfied,
* the audit report,

Shadow-Loom can produce **(prompt, brief, prose, audit-loss)** quadruples
that are useful as supervision for training models that learn to respect
hard narrative constraints. See [academic-foundations.md §4 (Constrained / neuro-symbolic generation)](academic-foundations.md#4-constrained--neuro-symbolic-generation)
for related lines of work.

## 7. Programmable narrative analytics

**Audience:** digital-humanities researchers and computational narratology.

Every metric is a Python function on a Pydantic model — no LLM in the loop,
deterministic, batch-friendly. Researchers can:

* Compute affective trajectories across an entire corpus.
* Compare structural-irony density between authors.
* Study the relationship between `RelationshipEdge` polarity and reader-rated
  emotional impact (using an annotated corpus or similar).
* Replicate experiments from prior literature (Reagan et al.'s "emotional
  arcs", Sternberg's curiosity/suspense/surprise) on a typed substrate.

## 8. MCP-driven agentic editing

**Audience:** developers building writing assistants on top of Claude /
GPT / local models.

The MCP server exposes the world model as a tool surface so an LLM agent
can:

* `open_project`, browse entities, fetch beliefs at a fabula time
* propose interventions, read back the auditor's loss
* commit a new version when the audit passes

This is how a chat-style writing assistant becomes a true **co-author with
memory** rather than a stateless completion box.

---

## What Shadow-Loom is *not* for

* Real-time per-token streaming generation in latency-critical loops.
  The simulation is heavy enough that one scene can take many seconds.
* Replacing human authorial judgement. The audit catches inconsistencies
  but the affective scorer is a *ranking* tool, not a literary critic.
* Generating short-form text without context — the value comes from the
  graph; if you don't have a story yet, use a normal LLM.

---

## See also

* [architecture.md](architecture.md) — the engine that makes every use case above possible.
* [query-and-cycles.md](query-and-cycles.md) — maps each use case to the query type that powers it (counterfactual exploration → `CounterfactualQuery`, suspense shaping → `DirectiveQuery`, manual editing → `ManualEditQuery`, QA → `EvaluationQuery`/`InterrogationQuery`).
* [ui-guide.md](ui-guide.md) — the workspace tabs that surface each capability to a human author.
* [mcp-guide.md](mcp-guide.md) — the agent-facing surface for use cases that integrate with Claude Desktop, Cursor, or custom MCP clients.
* [design-decisions.md](design-decisions.md) — the trade-offs that constrain what the system is and isn't well-suited for.
