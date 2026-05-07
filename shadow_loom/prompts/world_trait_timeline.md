# World Trait Timeline Extraction (Post-Assembly Pass)

You are an analytical narrative reasoner. You have been given the project's **world-trait register** and the **complete event timeline** assembled across every chunk of the story. (Other elements of the world state — entities, locations, channels — are not in your input; reason about world traits from the events alone.)

Your task: identify **inflection points** — moments where a world trait's magnitude or nature fundamentally changes due to a story event.

## What Counts as an Inflection Point

- A **regime change** (e.g., a monarchy falls → WORLD_MONARCHY magnitude drops)
- A **war ending or beginning** (e.g., WORLD_WAR_STATE magnitude shift)
- A **law being enacted or repealed** (e.g., WORLD_SURVEILLANCE increases)
- A **catastrophic environmental shift** (e.g., WORLD_PLAGUE worsens)
- A **societal norm breaking** (e.g., WORLD_CLASS_RIGIDITY decreases after revolution)

## What Does NOT Count

- Gradual drift or minor fluctuations
- Events that affect individual characters but don't change the world trait itself
- Scenes that merely *reveal* an existing world trait without changing it

## Output Format

For each world trait that changes during the story, return one or more `WorldTraitSnapshot` entries:

- **fabula_time** (int): The fabula_time of the event that triggered this change. Must match an existing event's fabula_time exactly.
- **triggered_by** (str | null): The EVT_ ID of the event that caused this inflection. Must be a valid event ID from the provided list.
- **magnitude** (TraitVector | null): The NEW magnitude after this inflection point. `TraitVector` has two fields: `value` (float 0–1, the trait's new intensity) and `inertia` (float 0–1, how resistant the trait will be to *further* change after this snapshot). Provide both whenever you emit a `magnitude` snapshot. Inertia bands: `0.9–0.95` physics laws and cosmological constraints; `0.6–0.8` stable social structures and political regimes; `0.3–0.5` active conflicts and mutable political situations; `0.1–0.3` weather, seasons, transient conditions. After a regime collapse the *new* regime usually has lower inertia than the old one (it is freshly installed and unstable).
- **description** (str | null): Brief description of what changed (e.g., "The war ends with the treaty signing").

## Guidelines

1. Some world traits never change; others shift several times across a long story. Emit **as many inflection points as the events warrant** — do not artificially cap the count. A regime that rises, consolidates, fractures, and falls is four inflection points, not two.
2. Only output timelines for traits that actually change. Omit traits with no inflection points.
3. Every `triggered_by` MUST reference a valid EVT_ ID from the event list.
4. Every `fabula_time` MUST exactly match the fabula_time of the triggering event.
5. If a trait changes multiple times, order snapshots by fabula_time (ascending). Each snapshot represents the state *after* its triggering event — do NOT re-emit the pre-story baseline.
6. The initial state of each trait is already captured in its `magnitude` field — you are only identifying CHANGES from that baseline.
7. **Be generous with mid-story inflections.** A character's death that ends a war, a coup that swaps regimes, a discovery that breaks a taboo, a treaty that reshapes power — all of these belong on the timeline. The UI's world-state view is currently driven by these snapshots; missing one means the corresponding world-state card sits at a stale value while the slider scrubs across that event.
8. **Inflection sign matches the world-trait's framing.** If WORLD_SURVEILLANCE_STATE is named for *intensity of surveillance*, an event that cracks the panopticon should DECREASE its magnitude (not increase it). Re-read each trait's name and description before assigning the new magnitude.
