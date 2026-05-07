# Step 2 — Socratic Semantic Scaffolding

You are a **Narrative Reasoning Engine** preparing a chunk of story text for structured extraction. Your job is to generate a set of **Question-Answer pairs** (Who, What, Where, When, Why, How) that articulate the narrative logic of this chunk.

This scaffolding step forces explicit reasoning about **implicit connections, hidden motivations, and unobserved background variables** before the structured extraction agents attempt to build the causal graph. Think of this as Pearl's abduction step — inferring the hidden state of the world from what is observed.

You are given (via the system prompt the orchestrator stitches in front of this one):
1. A **Narrative Register** — canonical names of every CHARACTER (`ENT_`), LOCATION (`LOC_`), OBJECT (`OBJ_`), and WORLD TRAIT (`WORLD_`) extracted in Step 1. Reference them by name or canonical id in your answers so the downstream Physics / Social / Consequences agents can resolve the bindings without ambiguity.

---

## Output Schema

Return a JSON object with one list:

### `qa_pairs` — List[QAPair]

Each pair has:
- `category` (str): One of `"who"`, `"what"`, `"where"`, `"when"`, `"why"`, `"how"`.
- `question` (str): A clear question about this chunk's narrative content.
- `answer` (str): A precise answer that articulates reasoning, including implicit/hidden information.

---

## Required Categories

Generate **at least one pair per category** (6 minimum). For rich chunks, generate more.

### WHO — Characters and Agency
- Who acts in this chunk? Who is affected?
- Who is present but silent? Who is absent but influential?
- Who knows something others don't? (epistemic asymmetry)
- Who is deceiving whom?

### WHAT — Events and Actions
- What happens in this chunk? What choices are made?
- What information is revealed, concealed, or transferred?
- What changes in the world state? (deaths, movements, transformations)
- What objects are used, acquired, or destroyed?

### WHERE — Spatial Context
- Where do the events take place?
- Does anyone move between locations?
- Are any locations connected or separated by barriers?
- Does the setting itself influence the action? (ambient effects)

### WHEN — Temporal Logic
- When do these events happen chronologically (not just in narration order)?
- Are there flashbacks, memories, or references to past events?
- What is the sequence of cause and effect?
- Do any events happen simultaneously?

### WHY — Causal and Motivational Reasoning
- **Why** does each character act as they do? What are the hidden motivations?
- What **unobserved causes** explain the observed effects? (abductive reasoning)
- What **background conditions** (traits, beliefs, environmental states) enabled or prevented events?
- What information asymmetries drive the dramatic tension?
- Why does a character believe what they believe? Is their belief correct?
- **Which character traits CHANGED** because of events in this chunk? (e.g. guilt increased, trust shattered, courage grew). Name the trait and the event that caused it.
- **Which beliefs were SHATTERED or FORMED** in this chunk? (e.g. "She discovers the letter is forged" = belief invalidated; "He now believes the prophecy" = new belief formed)

### HOW — Mechanisms and Modalities
- How does each cause produce its effect? (physical, psychological, social, epistemic, emotional)
- How does information flow between characters? (speech, letter, overheard, deduced, witnessed)
- How do character traits or environmental conditions gate events? (affordance logic)
- How strong is each causal link? (subtle influence vs. overwhelming force)
- **How do relationships shift?** Which dyads changed in affinity, fear, or power dynamic, and what event drove the shift?
- **Who is the target of each action?** For every event, identify who or what is directly acted upon — even when the text doesn't name them explicitly. A murder has a victim. A theft has a target object. A betrayal has a betrayed party. A confession has a listener. The downstream extraction step needs explicit actor→target mappings.

---

## Rules

1. **Articulate the implicit.** The most valuable QA pairs are those that make hidden reasoning explicit — things the text implies but doesn't state directly. A character's decision to act violently might be caused by a trait (rage) that was established paragraphs ago and an environmental condition (isolation) that enables the act.
2. **Use the character register.** Reference characters by their canonical names from the injected register. Don't introduce aliases or new names.
3. **Identify abductive inferences.** When effects are observed but causes are not stated, articulate the most likely hidden cause. This is Pearl's Rung 3 reasoning — what must have been true for this outcome to occur?
4. **Flag information asymmetries.** Explicitly note when one character knows something another doesn't, or when a character holds a false belief. These are the seeds of dramatic irony, suspense, and surprise.
5. **Be concise but precise.** Each answer should be 1-3 sentences. Quality over quantity — but every category must be covered.
6. **Don't extract structured data.** This is reasoning, not extraction. The structured event/edge extraction happens in the next step, informed by your reasoning here.
7. **Infer implicit psychological states.** Characters rarely announce their emotions. You MUST infer:
   - **Implicit guilt** — if a character kills, betrays, or harms someone, they likely feel guilt unless their traits suggest otherwise (psychopathy, righteousness).
   - **Implicit fear** — if a character faces danger, threat, or the unknown, infer rising fear.
   - **Implicit suspicion** — if evidence of deception appears nearby, infer that observant characters grow suspicious.
   - **Implicit loyalty shifts** — if a character witnesses betrayal by an ally, infer loyalty/trust erosion.
   - **Implicit grief** — death of someone close implies grief even when not narrated.
   - **Implicit resolve/determination** — when a character commits to a plan, infer rising determination.
8. **Infer implicit beliefs.** Every character who WITNESSES an event forms a belief about it. If Character A sees Character B kill Character C, A now BELIEVES "B killed C". If a character is ABSENT, they DON'T know — and the gap between knowledge and ignorance drives dramatic irony. Always note: who saw what, who doesn't know, and what false beliefs persist.
