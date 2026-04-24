# Shadow Loom

A neuro-symbolic causal narrative AI framework. Integrates classical narratology, Pearl's causal inference, information theory, and modern LLM orchestration to move from static world-building to mathematical simulation, prose generation, and rigorous self-auditing.

Built on hybrid neuro-symbolic reasoning with open-source LLMs, CTF-calculus, and Ancestral Multi-World Networks.

---

## Architecture Overview

The pipeline is divided into five phases and twelve steps:

1. **World State & Initialization** — establishing the physical and epistemic baseline
2. **Mathematical Simulation** — calculating what is physically and emotionally possible
3. **The Generative Constraint** — bridging hard math and natural language
4. **Prose Generation** — rendering constrained mathematics into literature
5. **The Nested Learning Audit** — self-evaluation and correction

---

### Phase 1: World State & Initialization

Before any new text is generated, the AI establishes the physical and epistemic baseline of the story.

#### Step 1: Entity & Ontology Ingestion

- **Function:** Parse story elements (Characters, Objects, Locations) into graph nodes.
- **Mechanism:** Assigns permanent physical trait vectors to nodes — `inertia` (resistance to change), `damage_potential`, and `spatial_affordances` (e.g., "inside", "locked"). Each `TraitVector` carries a `value` in [0, 1] and an `inertia` threshold that must be overcome for mutation.
- **Implementation:** `WorldStateV1` schema in `shadow_loom/models.py` — `Entity`, `NarrativeObject`, `Location` nodes with typed `TraitVector`, `Affordance`, and `Belief` sub-models.

#### Step 2: Canonical Graph Maintenance

- **Function:** Maintain the master database of the story.
- **Mechanism:** Stores every verified event on a strict `fabula_time` axis (objective physical chronology) to maintain cause-and-effect continuity. Causal edges (`CausalEdge`) carry `mechanism` (e.g., `physical_force`, `psychological`, `epistemic_revelation`) and `evidence_strength` (`weak`/`moderate`/`strong`) for statistical variance.
- **Implementation:** `WorldStateV1` holds `events`, `causal_topology`, `social_topology`, `spatial_topology`, and `information_topology` as typed Pydantic collections.

#### Step 3: Epistemic Synchronization

- **Function:** Separate what is physically true from what the reader knows.
- **Mechanism:** Maps the `syuzhet_index` (the order information is revealed in the text) alongside `fabula_time`. This dual-index system tracks who knows what at any exact moment. `InformationEdge` tracks communication channels with `established_at_fabula`, `terminated_at_fabula`, and `discovered_at_syuzhet` timestamps. `Belief` nodes on entities record `perceived_state` with `confidence` and `established_at_fabula`.
- **Implementation:** `EventNode.fabula_time` vs `EventNode.syuzhet_index`; `InformationEdge.discovered_at_syuzhet`; `Entity.beliefs`.

#### Step 4: AI Director Intent Formulation

- **Function:** A high-level system prompt dictates the narrative goal for the next generation cycle.
- **Mechanism:** The Director sets a specific emotional target and intensity (e.g., `target_effect: "suspense"`, `intensity: 0.8`) and selects the focal characters. Supported effects:
  - **Structural:** `mystery`, `dramatic_irony`, `suspense`, `surprise`
  - **Emotional:** `grief`, `rage`, `joy`, `regret`, `love`, `fear`
- **Implementation:** `DirectiveQuery` in `shadow_loom/query_models.py`. Other query types: `ObservationQuery` (Rung 1), `InterventionQuery` (Rung 2), `CounterfactualQuery` (Rung 3), `InterrogationQuery` (Graph RAG).

#### Step 5: Localized Ego-Graph Extraction

- **Function:** Save compute and avoid context-window overload by pulling only relevant data.
- **Mechanism:** Extracts a sub-graph containing only the nodes and causal edges relevant to the current scene, target characters, and requested emotion. Includes 1-hop spatial neighbors, co-located entities, temporal filtering (beliefs and relationships time-sliced to `temporal_anchor`), and a capped `memory_limit` of recent events.
- **Implementation:** `extract_ego_graph_from_memory()` in `shadow_loom/extract_graph.py`. Returns an `EgoGraphPayload` with `focus_entities`, `current_locations`, `present_entities`, `present_objects`, `relevant_relationships`, `relevant_causal_edges`, `relevant_spatial_edges`, `relevant_information_edges`, and `recent_memory`.

---

### Phase 2: Mathematical Simulation

The AI halts language generation to calculate what is physically and emotionally possible.

#### Step 6: AMWN Shadow Instantiation

- **Function:** Protect the canonical database from hallucinated or rejected timelines.
- **Mechanism:** Creates a volatile, in-memory NetworkX `MultiDiGraph` shadow graph (an Ancestral Multi-World Network). All simulation math is performed safely within this sandbox. Nodes are tagged `world_id: "shadow"` for volatile branches. The instantiator also wires:
  - `located_in` / `owned_by` edges for spatial and inventory topology
  - `relationship` edges with `affinity`, `fear`, `power_dynamic`, and `inertia`
  - `causal` edges with `mechanism` and `evidence_strength`
  - `connected_to` edges with `is_locked` and `barrier_item_id`
  - `communicating_with` edges with `medium` and `is_encrypted`
  - `eavesdropped_by` edges for epistemic leakage on unencrypted channels
- **Implementation:** `AMWNInstantiator.create_sandbox()` in `shadow_loom/instantiator.py`.

#### Step 7: Causal Physics & $do$-Calculus

- **Function:** Deterministic physics engine that calculates the boundaries of reality.
- **Mechanism:**
  - **Abduction (Rung 3):** Infers unobserved background events. Back-propagates present-day evidence into the historical sandbox — entity traits are blended 50% toward factual values, beliefs are back-propagated, and event evidence propagates through causal edges weighted by `evidence_strength` with mechanism-targeted gating via `MECHANISM_TRAIT_MAP`.
  - **Action (Rung 2):** Applies the $do$-operator ($do(X=x)$) to simulate theoretical choices, severing incoming causal edges. Six surgery types: spatial (with affordance path-checking), inventory, relationship (Impact > Inertia dampening), state mutation, genesis (spawn new nodes), and comms (establish/sever channels).
  - **Propagation:** Topological-sort-based forward propagation through the causal sub-graph. Per-trait signed delta impact: $(V_{\text{source}} - V_{\text{current}}) \times w$. Only mutations where $|Impact| > Inertia$ pass. Spatial affordance checks gate cross-location influence. Dampened shift: $\Delta_{\text{effective}} = Impact - \text{sign} \times Inertia$.
- **Implementation:** `CausalPhysicsEngine` in `shadow_loom/causal_physics.py`. Returns `CausalPhysicsResult` with `mutations`, `blocked`, `hidden_deltas`, `intervened_nodes`.

#### Step 8: Affective Calculus Engine

- **Function:** Grade the surviving physical branches for emotional and psychological resonance.
- **Mechanism:** Runs mathematical queries on the graph geometry to calculate four structural effects:

  | Effect | Calculation |
  |---|---|
  | **Mystery** | Walks backward from known effect nodes, counts causal ancestors hidden from the reader's syuzhet graph. $\text{score} = \frac{\text{hidden ancestors}}{\text{total ancestors}}$ |
  | **Dramatic Irony** | Finds revealed causal edges targeting an entity where the source event is NOT in the character's belief set. $\text{score} = \frac{\text{irony gaps}}{\text{total connections}}$ |
  | **Suspense** | Forward along causal edges: classifies unrevealed events as threat (entity = victim) vs hope (entity = actor). $\text{score} = P(\text{threat}) - P(\text{hope})$ using `evidence_strength` as probability proxy. Returns 0 when hope is extinguished (despair, not suspense). |
  | **Surprise** | Binary KL divergence per trait. Prior starts at maximum entropy (0.5), adjusted toward truth for each revealed causal edge. $D_{KL}(p \| q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$. Normalised to [0, 1]. |

  Emotion effects (`grief`, `rage`, `joy`, `fear`, `love`, `regret`) use trait-trajectory headroom analysis with direction-aware scoring (traits that should increase vs decrease for each effect).

- **Implementation:** `DirectiveAssembler` in `shadow_loom/directive_assembly.py`. Dedicated methods: `compute_mystery_score()`, `compute_dramatic_irony_score()`, `compute_suspense_score()`, `compute_surprise_score()`, `compute_affective_score()`.

---

### Phase 3: The Generative Constraint

The system bridges the gap between hard math and natural language.

#### Step 9: Directive Assembly & The Envelope of Possibilities

- **Function:** Prune mathematical failures and package the optimal success into a Creative Brief.
- **Mechanism:** The engine deletes any narrative branch that violated the physical constraints of Step 7. It ranks remaining valid branches by how perfectly they match the Director's emotional target from Step 8. The winning branch is compiled into a **Semantic Prompt Injection** — a rigid, inescapable set of natural language guardrails.
  - **Mystery:** Hidden predecessor counts + "do not reveal" constraints
  - **Dramatic Irony:** Reader/character asymmetry + "character MUST NOT learn" constraints
  - **Suspense:** Threat-vs-hope tension + withheld event protection + hidden channel constraints
  - **Surprise:** KL divergence magnitude + belief-shattering revelation constraints + flashback reveals
  - **Emotions:** Per-trait mathematical shift constraints with headroom and inertia evidence
- **Implementation:** `DirectiveAssembler.assemble()` returns a `CreativeBrief` with typed `ConstraintBlock` entries. `evaluate_candidate_events()` forks reality for each candidate intervention, prunes the impossible, and ranks survivors by affective score.

---

### Phase 4: Prose Generation

The Large Language Model is engaged strictly as a creative renderer, not an unrestricted author.

#### Step 10: LLM Rendering

- **Function:** Translate the mathematical guardrails into literature.
- **Mechanism:** A highly creative LLM receives the Semantic Prompt Injection. It weaves the mandatory physical state changes, hidden background truths, and epistemic information restrictions into flowing, natural dialogue and prose.
- **Output:** Scene-length narrative text that satisfies every constraint in the Creative Brief.

---

### Phase 5: The Nested Learning Audit

The system evaluates the LLM's output to catch and correct "Reward Hacking" or logical leaps.

#### Step 11: The Recursive Narrative Auditor (LLM-as-a-Judge)

- **Function:** A "Small Core" orchestrator model acts as literary critic and physics inspector.
- **Mechanism:** Reverse-engineers the prose from Step 10 back into causal claims.
  - **Causal Audit:** Checks for "Miracle Steps" — did the text bypass the physics graph?
  - **Abduction Audit:** Runs executable counterfactual questions against the text to ensure implicit events logically hold.
  - **Affective Audit:** Measures the actual epistemic gap in the prose to ensure the LLM didn't accidentally spoil a twist or ruin the suspense.

#### Step 12: Inner-Loop Refinement

- **Function:** Self-correction and finalization.
- **Mechanism:** If the Auditor detects an error (a non-zero loss), it generates explicit feedback and forces the LLM in Step 10 to rewrite the scene. This nested loop repeats until the text converges perfectly with the mathematical constraints. Once verified, the output is displayed to the user, and the new world state is permanently committed back to the Canonical Graph in Step 2.

---

## Project Structure

```
shadow_loom/
├── models.py              # WorldStateV1 schema (Pydantic v2)
├── query_models.py         # Query types: Observation, Intervention, Counterfactual, Directive, Interrogation
├── extract_graph.py        # Step 5: Ego-graph extraction with temporal filtering
├── instantiator.py         # Step 6: AMWN shadow sandbox (NetworkX MultiDiGraph)
├── causal_physics.py       # Step 7: CausalPhysicsEngine (3-rung CTF simulation)
├── directive_assembly.py   # Steps 8-9: Affective calculus + CreativeBrief assembly
├── narrative_physics.py    # Legacy pipeline orchestrator (routes all query types)
└── __init__.py             # Public API exports
```

## Running Tests

```bash
conda run -n shadow-loom python -m pytest tests/ -v
```
