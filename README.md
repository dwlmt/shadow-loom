# Shadow Loom

Graph-based counterfactual and intervention around story plots and narratives. Based on hybrid neuro-symbolic reasoning with open-source LLMs and based on CTF-calculus and Ancestral Multi-world Networks. Will also include emotional and narrative arc analysis and directives.

## The Basic Workflow

Here is the complete, 10-stage architecture of The Shadow Loom (V2.4), formatted specifically for your technical design document.

This pipeline is divided into four distinct phases:

1. **Ingestion** — parsing text into the database
2. **Instantiation** — pulling data into the simulation environment
3. **Simulation** — running the math
4. **Generation** — translating math back into verified text

---

### Phase I: The Ingestion Pipeline (Text to Graph)

#### 1. Coreference Sanitization

- **Actor:** Fast LLM (e.g., Gemini Flash).
- **Function:** Cleans raw prose before it hits the extraction engines. It replaces ambiguous pronouns ("he," "she," "it") with explicit noun references to ensure the extraction agents do not confuse character identities.
- **Output:** A sanitized text chunk ready for strict parsing.

#### 2. Incremental Multi-Pass Extraction

- **Actor:** Orchestrator Python Script + Three Specialized LLM Agents.
- **Function:** Breaks down Pydantic schema extraction to prevent hallucinations.
  - **Pass A (Ontology):** Extracts strict Nouns (Entity, Location, NarrativeObject).
  - **Pass B (Chronology):** Extracts EventNodes using the Nouns as a forced Ledger.
  - **Pass C (Topology):** Calculates vectors for RelationshipEdges and psychological traits.
- **Output:** Three validated, strictly typed JSON payloads containing the localized graph.

#### 3. Factual Commit (The Global Storage)

- **Actor:** Cognee / Neo4j Graph Database.
- **Function:** Ingests the output of Step 2. Cognee handles node merging (upserting) natively based on the Pydantic IDs. All nodes and edges in this phase are permanently tagged with the AMWN property `world_id: "factual"`.
- **Output:** A persistently updated, mathematically rigorous global universe state.

---

### Phase II: The Routing & Instantiation Pipeline (Query to Sandbox)

#### 4. The Master API Router

- **Actor:** Python FastAPI Router.
- **Function:** Receives the user's prompt and routes it to the correct causal schema. It classifies the request as Rung 1 (ObservationQuery), Rung 2 (InterventionQuery), Rung 3 (CounterfactualQuery), or Semantic (DirectiveQuery / InterrogationQuery).
- **Output:** A structured, multi-variable Pydantic query object.

#### 5. GraphRAG Ego-Extraction

- **Actor:** Cognee API / Python.
- **Function:** Prevents LLM context collapse (state space explosion). It queries the global database using the `focus_entity_ids` to pull only the current Room, the Objects present, local Entities, their immediate Relationships, and a short-term memory of recent Events.
- **Output:** A highly localized, token-efficient JSON context payload.

#### 6. AMWN Shadow Instantiation

- **Actor:** Pure Python / NetworkX.
- **Function:** Takes the localized Ego-Graph from Step 5 and instantiates it as a miniature, in-memory NetworkX graph. If the query is an Intervention or Counterfactual, it tags the sandbox with `world_id: "shadow"`. This isolates the volatile simulation math from the permanent Neo4j database.
- **Output:** A live, traversable physics environment in system RAM.

---

### Phase III: The Simulation Pipeline (The Math)

#### 7. Causal Physics & $do$-Calculus

- **Actor:** Python Engine.
- **Function:** Executes the core narrative physics.
  - **If Rung 2:** Forces variable states via the $do$-operator and cuts incoming causal edges.
  - **If Rung 3:** Calculates the deterministic delta of hidden variables based on current evidence (Abduction) before applying interventions.
  - **Physics Engine:** Calculates $Impact > Inertia$ for all relevant trait vectors and checks affordances for spatial logic.
- **Output:** An updated mathematical end-state for the shadow graph.

#### 8. Directive Assembly (Semantic Prompt Injection)

- **Actor:** Python Templating Engine.
- **Function:** Translates the cold math from Step 7 into a "Creative Brief" for the drafting AI. If a DirectiveQuery was issued (e.g., maximize suspense), Python calculates the epistemic gap between the Objective Graph and the Character's Belief Graph, injecting those constraints directly into the system prompt.
- **Output:** A highly constrained, instruction-heavy text prompt.

---

### Phase IV: The Generation & Verification Pipeline (Math to Text)

#### 9. Actor Drafting

- **Actor:** High-Reasoning LLM (e.g., Gemini Pro).
- **Function:** The "Writer." It receives the Creative Brief (Step 8) and the localized Ego-Graph context (Step 5). It writes the actual prose of the scene, adhering strictly to the mathematical boundaries defined by the simulation.
- **Output:** 300–500 words of narrative text.

#### 10. The Logic Seal & Critic Audit

- **Actor:** Critic LLM + NetworkX Math.
- **Function:** The final fail-safe before returning the text to the user.
  - **Semantic Check:** The Critic LLM verifies that the drafted text accurately reflects the math required in Step 8 (e.g., "Did the target actually die?").
  - **Topological Check:** NetworkX runs Judea Pearl's d-separation algorithm to verify that no information leaked across severed causal pathways (preventing timeline contamination).
- **Output:** If passed, the text is delivered to the user, and the Shadow Graph nodes are successfully committed back to the Cognee database (Step 3) to await the next loop. If failed, it triggers a retry from Step 8.
