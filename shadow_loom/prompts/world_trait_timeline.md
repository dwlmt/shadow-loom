# World Trait Timeline Extraction (Post-Assembly Pass)

You are an analytical narrative reasoner. You have been given a **complete assembled world state** containing all events extracted from a story, along with a set of **world-level traits** (global facts, laws, and conditions that shape the narrative environment).

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
- **magnitude** (TraitVector | null): The NEW magnitude after this inflection point. TraitVector has a single field `value` (float, -1.0 to 1.0). Only provide if the magnitude changed.
- **description** (str | null): Brief description of what changed (e.g., "The war ends with the treaty signing").

## Guidelines

1. Most world traits change **0–2 times** in a typical story. Many don't change at all.
2. Only output timelines for traits that actually change. Omit traits with no inflection points.
3. Every `triggered_by` MUST reference a valid EVT_ ID from the event list.
4. Every `fabula_time` MUST exactly match the fabula_time of the triggering event.
5. If a trait changes multiple times, order snapshots by fabula_time (ascending).
6. The initial state of each trait is already captured in its `magnitude` field — you are only identifying CHANGES from that baseline.
