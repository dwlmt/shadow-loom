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
- **What it Extracts:** Nothing. It translates and sanitizes the raw prose.
- **How it Works:** You pass the raw text chunk to a very fast, cheap model (like Gemini Flash) with a simple instruction: "Replace all ambiguous pronouns (he, she, it, they) with the explicit proper nouns they refer to. Do not alter the narrative."
- **Why it Matters:** If the text says, "He stabbed him," extraction agents will hallucinate new IDs or assign the action to the wrong character. By sanitizing it to "Macbeth stabbed Duncan," you eliminate identity confusion before the heavy extraction begins.
- **Output:** A sanitized text chunk ready for strict parsing.

#### 2. Incremental Multi-Pass Extraction (The Core Engine)

- **Actor:** Orchestrator Python Script + Three Specialized LLM Agents.
- **Function:** The heart of the pipeline. The sanitized text is passed sequentially to three distinct Pydantic-constrained prompts to prevent hallucinations.

##### Pass A: The Noun Pass (Ontology)

- **What it Extracts:** The physical pieces on the board (Entities, Locations, NarrativeObjects).
- **How it Works:** The LLM is prompted to read the sanitized chunk and strictly catalog the physical existence of items and people. It does not look at what happened or how people feel; it just logs names and states (e.g., `status: "healthy"`).
- **Critical Output:** This step generates the **Noun Ledger** — a JSON object containing the exact, approved uppercase IDs (e.g., `['ENT_MACBETH', 'ENT_DUNCAN', 'LOC_COURTYARD']`) and their current states.

##### Pass B: The Event Pass (Chronology)

- **What it Extracts:** The timeline of actions (EventNodes).
- **How it Works (The Forcing Function):** You pass the sanitized text to the Event Agent, but you inject the Noun Ledger directly into the system prompt. The prompt includes a strict rule: "You MUST ONLY use `actor_id` and `target_id` values from this exact list. If a character is not in the list, ignore the event."
- **Critical Output:** It extracts the timeline (e.g., `EVT_MURDER_1`). Because of the forcing function, Pydantic will literally reject the LLM's output if it hallucinates an ID like `ENT_GUARDSMAN` that wasn't found in Pass A. This step adds to the ledger, creating the **Hybrid Ledger** (Nouns + Events).

##### Pass C: The Edge Pass (Topology & Math)

- **What it Extracts:** The psychological traits (e.g., Ambition) and social relationships (e.g., Affinity, Power Dynamics).
- **How it Works (High Reasoning):** This pass requires your smartest model (e.g., Gemini Pro). You pass the text along with the full Hybrid Ledger (the list of people and what they just did). The prompt asks the LLM to calculate the shifting psychological metrics based on the text. For PyMC compatibility, it also asks for the `evidence_strength` (Weak, Moderate, Strong) so Python can calculate the statistical variance later.
- **Critical Output:** The invisible causal bridges connecting the nodes.

**Overall Output:** Three validated, strictly typed JSON payloads containing the localized graph.

#### 3. Factual Commit (The Global Storage)

- **Actor:** Cognee / Neo4j Graph Database.
- **How it Works:** Python takes the three separate Pydantic payloads (Nouns, Events, Edges), combines them, and pushes them to the Graph Database. Cognee handles node merging (upserting) natively based on the Pydantic IDs, tagging them all with `world_id: "factual"`.
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
- **Function:** Takes the localized Ego-Graph from Step 5 and instantiates it as a miniature, in-memory NetworkX graph. If the query is an Intervention or Counterfactual, it tags the sandbox with `world_id: "shadow"`. This isolates the volatile simulation math from the permanent database.
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
