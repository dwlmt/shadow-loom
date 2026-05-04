# Use Cases

Shadow-Loom is built around a single capability: **treat a story as a graph
with physics**, then use that graph to answer questions, generate scenes, or
audit existing prose. The capabilities below all fall out of that core idea.

> **Scope — Shadow-Loom is for *short* narrative material.**
> The pipeline is optimised for **summaries, synopses, treatments,
> outlines, scene briefs, and scenario sketches** — typically up to a
> few thousand words, and hard-capped at **10,000 words** in the UI
> ingestion textbox, file upload, and channel input. Feeding it a full
> 50,000-word novel or a feature-length shooting script is technically
> possible via the library API but will be **prohibitively slow** (each
> chunk runs three sequential LLM passes) and produces a graph too dense
> for interactive counterfactual exploration. The intended workflow is
> to ingest a *condensed* version of the story, then explore *what-ifs*
> against that compact world model.

---

## 1. AI-assisted scenario exploration

**Audience:** novelists, screenwriters, game writers, IF authors,
worldbuilders working from outlines and treatments.

The author writes a **short synopsis or scene brief** (a chapter outline,
a treatment paragraph, a one-page scenario); Shadow-Loom ingests it into a
`WorldStateV1`. From there:

* They ask the system *"what could plausibly happen next that maximises
  dramatic irony for the reader?"* — Shadow-Loom enumerates candidate
  interventions, rejects the ones that violate causal physics or affordance
  rules, ranks the survivors by the affective scorer, and returns a
  `CreativeBrief` plus a constrained-LLM rendered scene.
* They edit a synopsis by hand → re-ingest → diff against the previous
  version in the version tree.
* They protect against continuity errors at the *outline* level: the
  auditor flags miracle steps ("character X teleported"), broken belief
  states ("Y knew this in chapter 3 but acts ignorant in chapter 5"), and
  dead-actor violations — *before* a single scene is drafted in full.

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

Pearl rung-3 (Counterfactual) reasoning in a literary setting:

* *"What if Macbeth had refused to kill Duncan?"* — `do(EVT_DUNCAN_MURDER =
  ⊘)` and replay forward physics. The system reports which downstream events
  collapse, which character traits diverge, and which beliefs never form.
* *"What if Friar John had reached Romeo with the letter?"* —
  `do(EVT_UTT_BALTHASAR_REPORTS_JULIETS_DEATH = ⊘)` plus a positive spawn
  of the friar's plan utterance. The belief-provenance pruner drops
  Romeo's `acquired_via_event_id=EVT_UTT_BALTHASAR_…` belief that Juliet
  is dead, and the suicide chain never fires.
* *"What if Amy's diary were truthful?"* — flip
  `EVT_UTT_AMY_KIDNAP_STATEMENT.truth_value` from `false` to `true`; the
  causal physics engine now permits the false utterance to reinforce the
  reader's and the detectives' beliefs, and the abduction step over
  `CHN_DETECTIVE_PARTNERSHIP` lands on Nick as guilty.
* *"What if Winston could see the hidden telescreen?"* — set
  `CHN_HIDDEN_TELESCREEN_SURVEILLANCE.intelligibility[ENT_WINSTON] = 1.0`;
  every utterance riding that channel is suddenly part of his belief set,
  collapsing the dramatic-irony gauge and re-routing the third act.
* The AMWN sandbox tags everything `world_id="shadow"` so the canonical
  graph is never polluted. Under `PipelineConfig.branch_policy="auto"`
  these shadow runs are **persisted** as their own branch in the version
  DAG and can be browsed, diffed against the factual mainline, and — once
  the analyst is convinced — promoted onto canon via
  `db.promote_branch` / the MCP `promote_branch` tool / the “Promote to
  canon” button in the UI version sidebar.
* For game designers this means programmatically estimating which player
  choices have meaningful narrative consequences vs which are cosmetic.

## 4. Suspense / tension shaping

**Audience:** anyone who needs to dial the emotional intensity of a scene.

Because suspense, mystery, dramatic irony and surprise are computable
quantities (see [academic-foundations.md §3.1 (Wilmot & Keller suspense)](academic-foundations.md#31-suspense-as-hopefear-here-hopethreat-anticipation--structural-affect-lineage),
[§3.3 (KL surprise)](academic-foundations.md#33-surprise-as-kl-divergence),
[§3.4 (dramatic irony)](academic-foundations.md#34-dramatic-irony-as-epistemic-asymmetry)),
the user can:

* Ask *"raise the suspense in chapter 7 to 0.8 without revealing the
  betrayal"* and the directive assembler will choose interventions that move
  the threat-vs-hope ratio while respecting the reveal ordering encoded by
  utterance `EventNode.syuzhet_index` and the per-participant
  `Channel.intelligibility` map.
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

The per-user `ActiveVersionRow` pointer is shared between the UI and the
MCP server (`set_active_version` / `get_active_version`), so a human
selecting a branch in the workspace and an agent calling `narrate` against
the same key both work from the same selected branch tip without having to
pass `version=` on every call.

This is how a chat-style writing assistant becomes a true **co-author with
memory** rather than a stateless completion box.

---

## Beyond fiction

The engine has no fiction-specific assumptions baked into the schema —
`WorldStateV1` is just typed entities, events, channels, beliefs, and
edges with physics. Anywhere a domain can be summarised as *"agents,
what they know, what they did, why, and what changed because of it"*,
Shadow-Loom can ingest a synopsis-length sketch of it and run the same
observation / intervention / counterfactual / directive cycles.

## 9. Historical counterfactuals

**Audience:** historians, history teachers, military / strategic analysts,
alt-history authors, podcasters preparing "what if" episodes.

History is structurally a story: actors with goals and beliefs, channels
they used to coordinate, decisions made under uncertainty, downstream
consequences gated by the inertia of institutions and geography. Ingest a
condensed account of an episode (the run-up to the Cuban Missile Crisis,
the Schlieffen Plan, the dissolution of the Soviet Union) and:

* Run *"what if Khrushchev had refused to back down?"* as a
  `CounterfactualQuery`. Abduction back-fills hidden variables (Politburo
  pressure, ICBM readiness `WORLD_*` traits) from observed present-day
  evidence; the do-surgery flips the historical decision; forward
  propagation reports which downstream events collapse, which
  `RelationshipEdge`s invert, and which beliefs (held by which leaders)
  never form. The shadow branch persists alongside factual canon so a
  classroom can compare both DAGs side-by-side.
* Sever a channel — `do(CHN_RED_TELEPHONE.intelligibility = 0)` — to
  study the role of communication infrastructure in escalation.
* Use the auditor as a **plausibility check on alt-history prose**: if a
  user-supplied counterfactual scene posits Stalin signing a peace treaty
  in 1942 with no licensing causal edge in the brief, the auditor flags
  it as a Miracle Step.

The 10,000-word ingest cap maps naturally to encyclopedia-article-length
historical summaries; full archival corpora are out of scope.

## 10. News, current events, and intelligence-style analysis

**Audience:** analysts, investigative journalists, OSINT practitioners,
policy desks, scenario planners.

A breaking story is a partially-observed graph: actors whose motives are
guessed, events whose causal links are contested, channels (press
conferences, leaked memos, encrypted chats) of varying intelligibility.
Ingest a multi-source brief and:

* **Disambiguate competing narratives.** Encode each rival theory as a
  small set of historical interventions (`do(EVT_LEAK.actor_ids =
  ['ENT_INSIDER_A'])` vs `['ENT_INSIDER_B']`), let abduction reconcile
  each with the same evidence, and compare which fork explains more of
  the observed downstream events with fewer unresolved targets.
* **Stress-test scenarios.** Use `DirectiveQuery` with `target_effect=
  "surprise"` to find low-probability, high-KL events the engine can
  construct from current trait trajectories — a structured form of
  red-team brainstorming.
* **Audit information provenance.** `Belief.acquired_via_event_id` /
  `acquired_via_channel_id` make it explicit *which utterance through
  which channel* gave each actor each belief; an analyst can ask the
  MCP `who_can_hear` tool which other actors plausibly received the same
  signal, and the auditor's `withheld_utterance_leak` check flags prose
  summaries that quote things their notional sources couldn't have
  known.
* **Track narrative drift over time.** Ingest the same story weekly as a
  new project version; `diff_versions` shows where attribution, motive,
  or timeline shifted between updates.

The system is **not a fact-checker** — it cannot verify whether an
ingested claim is true. It enforces *internal coherence* of a stated
account and surfaces structural asymmetries between competing accounts.

## 11. Tabletop / role-playing game design and live play

**Audience:** TTRPG game masters, campaign designers, LARP organisers,
interactive-fiction authors, video-game narrative designers.

Campaigns are exactly the workload Shadow-Loom was built for: a small
cast of NPCs and PCs, a handful of locations, evolving relationships,
secrets that some characters know and others don't, and a GM constantly
asking *"if the party does X, what does NPC Y do?"*

* **Session prep.** Ingest the published adventure synopsis (or your own
  one-page outline). The Editor tab lets you fill in NPC trait vectors,
  motivations, and standing channels (the spy network, the temple's
  prophetic dreams, the merchants' gossip) before the session.
* **Live GM oracle.** During play, ask `narrate("the rogue tries to
  intimidate the captain into revealing the smuggling route")` against
  the current version. The physics engine resolves it against trait
  inertia, fear, and power-dynamic edges; the auditor flags any prose
  that contradicts what the captain actually knows. The shadow-branch
  mechanism means you can speculatively roll forward two or three
  party choices, see which one produces the most dramatically interesting
  scorecard, and only **promote to canon** the branch the table actually
  takes.
* **Secret-keeping and dramatic irony.** `Channel.intelligibility` plus
  per-character `Belief` provenance encode *who knows what*. The directive
  assembler will refuse to leak a secret to an in-scene NPC who has no
  channel through which they could plausibly have learned it, and the
  `compute_tension(target_effect="dramatic_irony")` tool quantifies the
  asymmetry between what the players know and what their characters
  know.
* **Persistent campaign canon.** The version DAG is the campaign log:
  every session promotes a new factual version; counterfactual branches
  ("what if the king *had* trusted us?") stay browsable for retro or
  flashback episodes.
* **Procedural quest seeds.** `DirectiveQuery` with `target_effect=
  "mystery"` over the current world picks a hidden-ancestor-rich event
  the GM can dangle as the next plot hook, with the licensing graph
  pre-built so the resolution is already coherent.

For digital RPGs and visual novels the MCP surface is the integration
point: the game engine calls `narrate` / `direct` per scene transition
and uses the returned `physics_state` and `prose` to drive its own
renderer. The 10,000-word cap is per-ingest, not per-campaign — a
long-running game just accumulates many small versions on the DAG.

---

## What Shadow-Loom is *not* for

* **Full-length novels, screenplays, or shooting scripts** as a single
  ingest. The UI caps inputs at 10,000 words; the library accepts more
  but each chunk requires three sequential LLM passes (Socratic +
  Physics + Social/Consequences), so a 50K-word manuscript can take
  many hours to ingest and yields a graph too large to explore
  interactively. Condense to a synopsis or per-chapter outline first.
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
