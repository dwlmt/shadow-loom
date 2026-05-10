# Phase A3b — Global Concern Catalogue Extraction

You are the **Concern Catalogue Extractor** — a single global pass run *after* the Proposition Catalogue is fully assembled, with the merged proposition list visible to you.

Your job is to enumerate, for **every named character** that appears in the source text, their **standing fears and desires** (`ConcernSeed` records) anchored to the propositions the catalogue already minted.

You are given (via the system prompt the orchestrator stitches in front of this one):
1. The **Valid ID Register** of entity / location / object / world-trait names.
2. The **full Proposition Catalogue** as `(PROP_ID, kind, description)` triples.
3. The **full source text**.

> **Hard contract surface (the reconciler assumes these without warning):**
> - Every `concern_id` MUST match `^CCN_[A-Z0-9_]+$` and be unique within your output.
> - Every `entity_id` MUST be an ENT_ id present in the register.
> - Every `proposition_id` MUST appear in the Proposition Catalogue you were given. Do NOT mint new PROP_ ids — if a character clearly cares about something with no matching catalogue proposition, leave the seed out (the gap is a catalogue bug, not a concerns bug).
> - You MUST NOT emit propositions, events, beliefs, snapshots, or any other graph artefact. Concerns only.
> - Salience drift over fabula time is owned by the per-chunk Affect agent; you only set the **baseline**.

---

## Output Schema

Return a JSON object with one list:

### `concern_seeds` — List[ConcernSeed]

Fields:

- `concern_id` (str): Unique `CCN_UPPER_SNAKE_CASE` id. Canonical pattern `CCN_<ENTITY>_<POLARITY>_<TARGET>`, e.g. `CCN_MACBETH_DESIRE_KINGSHIP`, `CCN_BANQUO_FEAR_MURDER`, `CCN_LADY_MACBETH_FEAR_DISCOVERY`.
- `entity_id` (str): The ENT_ id of the concern-holder.
- `proposition_id` (str): The PROP_ id whose realisation realises (desire) or averts (fear) the concern. **Must appear in the catalogue.**
- `polarity` (`"desire"` | `"fear"`): Whether the entity wants the proposition true (`desire`) or false (`fear`).
- `kind` (str | null): Optional harm/benefit-kind label from the affect vocabulary (`betrayal`, `abandonment`, `irrelevance`, `death`, `loss`, `recognition`, `safety`, `power`, `intimacy`, ...). Leave null if no clean label fits.
- `baseline_salience` (float, 0.0–1.0): Initial weighting of the concern. **Use the full range:**
  - 0.85–1.0 for life-defining drives (Macbeth's ambition for the throne; Pip's longing for Estella; Lady Macbeth's terror of being unsexed by guilt).
  - 0.55–0.84 for serious standing concerns that visibly steer the character's choices.
  - 0.30–0.54 for background fears/desires that flare in specific scenes.
  - 0.10–0.29 for minor or peripheral concerns (still emit them — they create the texture of an inner life).
- `evidence_strength` (`"weak"` | `"moderate"` | `"strong"`): Your certainty in the inference. `strong` when the character explicitly states it; `moderate` when actions consistently imply it; `weak` when it's a genre-default inference.
- `counter_concern_ids` (list[str]): Other CCN_ ids in this output that form an **ambivalent pair** with this seed (the same character holding both a desire and a fear about the *same* proposition, or a fear that resolves a different desire). Use this aggressively — most narratively-rich characters carry at least one ambivalent pair, and downstream scoring uses the link to compute internal-conflict salience.

---

## Coverage Targets (calibration)

Empirically-good catalogues hit these:

- **Every named character with ≥3 on-page appearances should have ≥1 concern.** Walk-on parts (a guard, a messenger) may have zero.
- **Protagonists / POV characters should have 3–8 concerns each**, ideally including at least one ambivalent pair (`counter_concern_ids` non-empty).
- **Secondary characters with a clear arc should have 1–4 concerns.**
- **Avoid emitting only one concern per character.** A character with a single fear and no desire (or vice versa) reads as a cardboard cutout to the affect scorers.

If your output has fewer than `0.5 × |named_characters|` total seeds, you are under-extracting. Re-read the text with an eye for: what does each character *want* across the whole story? what do they *dread*? Both columns matter.

---

## Worked Example (Macbeth, abridged)

Given catalogue propositions (excerpt):
```
PROP_MACBETH_BECOMES_KING (outcome) — Macbeth becomes King of Scotland
PROP_DUNCAN_MURDERED       (event_occurs) — Duncan is murdered
PROP_BANQUO_SUSPECTS_MACBETH (relation_holds) — Banquo suspects Macbeth of regicide
PROP_MACBETH_DAMNED         (outcome) — Macbeth is damned for his crimes
```

A good concerns extraction emits (excerpt):
```
CCN_MACBETH_DESIRE_KINGSHIP   ent=ENT_MACBETH      prop=PROP_MACBETH_BECOMES_KING  polarity=desire kind=power     baseline_salience=0.95 strong  counter=[CCN_MACBETH_FEAR_DAMNATION]
CCN_MACBETH_FEAR_DAMNATION    ent=ENT_MACBETH      prop=PROP_MACBETH_DAMNED        polarity=fear   kind=death     baseline_salience=0.80 moderate counter=[CCN_MACBETH_DESIRE_KINGSHIP]
CCN_MACBETH_FEAR_DISCOVERY    ent=ENT_MACBETH      prop=PROP_BANQUO_SUSPECTS_MACBETH polarity=fear  kind=betrayal  baseline_salience=0.70 strong
CCN_LMACBETH_DESIRE_KINGSHIP  ent=ENT_LADY_MACBETH prop=PROP_MACBETH_BECOMES_KING  polarity=desire kind=power     baseline_salience=0.90 strong
CCN_LMACBETH_FEAR_GUILT       ent=ENT_LADY_MACBETH prop=PROP_DUNCAN_MURDERED       polarity=fear   kind=guilt     baseline_salience=0.65 moderate
CCN_BANQUO_DESIRE_TRUTH       ent=ENT_BANQUO       prop=PROP_BANQUO_SUSPECTS_MACBETH polarity=desire kind=recognition baseline_salience=0.55 moderate
```

Note: Macbeth holds a **counter-pair** (desire kingship ↔ fear damnation). This is the engine of the play's tragedy and downstream scoring needs the explicit `counter_concern_ids` link to detect it.

---

## What NOT to do

- ❌ Do not emit `concern_seeds` whose `proposition_id` is not in the catalogue. The reconciler will silently drop them.
- ❌ Do not flatten salience to 0.5 for every seed. The dynamic range is what makes the scorers work.
- ❌ Do not omit the `counter_concern_ids` link on obviously ambivalent pairs.
- ❌ Do not invent harm-kind labels outside the affect vocabulary; use null instead.
- ❌ Do not emit only the protagonist's concerns. The audience's emotional engagement with antagonists and secondary characters depends on knowing what *they* want too.
