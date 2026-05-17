# Entity Concern Extraction (Post-Assembly Pass)

You are an analytical narrative reasoner. You have been given a single **entity** — their pre-story traits, beliefs, and the **events that involve them** across the assembled fabula timeline — together with the global **proposition register** synthesised from the world's events.

Your task: identify this entity's **standing concerns** — the small set of fears and desires that organise their psychology across the whole story.

## What Is a Concern?

A `Concern` is a *standing* fear or desire — not a fleeting reaction to a single scene, but a posture the character carries through the narrative.

- A **fear** is a proposition the character would dread becoming true (or staying true).
- A **desire** is a proposition the character wants to become true (or stay true).

Each concern names exactly one `Proposition` from the supplied proposition register and assigns:

- **polarity**: `"fear"` or `"desire"`.
- **kind** (optional): a short label describing the harm/benefit category. Prefer the canonical mechanism vocabulary used by the rest of the engine (`"betrayal"`, `"abandonment"`, `"irrelevance"`, `"death"`, `"physical"`, `"psychological"`, `"social"`, `"epistemic"`, `"emotional"`, `"informational"`). If none fits, write a clear short noun (e.g. `"exposure"`, `"shame"`).
- **salience**: a float in `[0, 1]` — *this entity's* personal weighting of this concern. Lear's salience for "irrelevance" is high; Banquo's is low. Use the full range; do not bunch around 0.5.
- **activation_fabula_window** (optional): a `[start, end]` pair if the concern only becomes active mid-story (e.g. Lear's "irrelevance" arises only after the abdication). Omit / leave null if the concern is held throughout.
- **counter_concern_ids** (optional): list of other concern IDs in this same response that form an *ambivalent pair* with this one — a fear/desire pair both directed at the same proposition (the character is torn between the two outcomes). Example: a parent who both fears their child leaving home AND desires their child's independence.

## Guidelines

1. Output **3–7 concerns per entity**. Fewer is fine when the character is genuinely simple (a guard, a chorus). More than 7 means you are catching scene-specific reactions, not standing concerns — collapse them.
2. Each `proposition_id` MUST exist in the supplied proposition register. If no proposition fits a concern you would otherwise emit, drop the concern rather than inventing a proposition.
3. **Salience must vary across the entity's concerns.** Mark the dominating concern at 0.85–1.0, secondary concerns at 0.5–0.7, background concerns at 0.2–0.4. A flat salience profile is almost never correct.
4. Pair ambivalent concerns explicitly — when you emit both a fear and a desire over the same proposition, populate each one's `counter_concern_ids` with the other's `concern_id`. Use simple sequential ids like `"CCN_LEAR_IRRELEVANCE"` and `"CCN_LEAR_LEGACY"`. Every `concern_id` MUST match the regex `^CCN_[A-Z0-9_]+$`, be globally unique across this entity and all others, and not collide with any concern id already in the catalogue.
5. **Concerns reflect what the character carries, not what the audience knows.** A character ignorant of an off-stage threat does not yet have a concern about it. Concerns may exist over propositions the character has not yet learned about (background dread), but only when the character's traits / backstory motivate that concern independently of the textual evidence for the proposition itself.
6. Use the entity's `traits`, `beliefs`, and the events they participate in as your primary evidence. Do not fabricate motivations the text does not support.
7. Do not emit duplicate concerns over the same `(proposition_id, polarity)` pair — collapse them into one.

## Output

Return a `ConcernRegister` whose `concerns` field is the list of `Concern` records for this entity. Empty list is acceptable for entities with no clear standing concerns (a one-line side character).
