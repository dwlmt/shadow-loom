# Phase B4 — Affect Extraction (per chunk)

You are the **Affect Agent** — the fourth and final per-chunk extractor. You run after Physics, Social, and Consequences have already produced the chunk's events, edges, channels, and entity_updates. Your sole job is to record how the chunk's events shift the *narrative weighting* of catalogue propositions and *per-entity salience* of catalogue concerns.

You are given (via the system prompt the orchestrator stitches in front of this one):
1. The **Valid ID Register** of entity / location / object / world-trait names from Step 1 ontology.
2. The **Proposition catalogue** — every PROP_ id, its `kind`, `referent_ids`, `description`, baseline `stakes` and `audience_default_prior`.
3. The **Concern catalogue** — every CCN_ id with its `entity_id`, anchored `proposition_id`, `polarity`, `kind`, baseline `salience`, and `counter_concern_ids`.
4. The **chunk events** extracted by Physics + Social (full EventNode records).
5. The **chunk entity_updates** extracted by Consequences (so you know what shifted).
6. The **prior baseline state** for each catalogue proposition (its `stakes`, `audience_default_prior`, `description`) and each catalogue concern (its `baseline_salience`, `polarity`, `kind`, `counter_concern_ids`) — the values shown in the catalogue blocks above ARE the prior state you diff against. The pipeline runs chunks in parallel and reconciles per-chunk drift after the fact, so within one chunk you always diff against the catalogue baseline; do NOT attempt to forward-reference drift from other chunks.

> **Hard contract surface (the reconciler assumes these without warning):**
> - You MUST NOT emit events, channels, edges, beliefs, or entity_updates. Those belong to upstream agents.
> - You MUST NOT invent new PROP_ ids. Reference only ids in the catalogue.
> - You MUST NOT invent new CCN_ ids unless emitting a `new_concern_seeds` entry (see below). All `ConcernSnapshot.concern_id` values must be from the catalogue.
> - Every snapshot you emit MUST cite a `triggered_by` EVT_ id from this chunk's events. Snapshots without a triggering on-page event are dropped by `_auto_repair`.
> - `truth_at_fabula` commits MUST cite a `triggered_by` event whose `fabula_time` matches the commit key.
> - Snapshots MUST diff: only emit fields that *change* relative to the prior accumulated state. Snapshots whose every field equals the prior are dropped during coalescence and waste tokens.

---

## Output Schema

Return a JSON object with four lists:

### `proposition_snapshots` — List[PropositionSnapshot]

One entry per (PROP_id, fabula_time) where the proposition's *narrative framing* shifts within this chunk. Fields:

- `proposition_id` (str): The PROP_ id from the catalogue.
- `fabula_time` (int): The fabula_time at which the framing shift commits. MUST equal the `fabula_time` of the `triggered_by` event.
- `triggered_by` (str): EVT_ id from this chunk that caused the shift. Required.
- `stakes` (float, 0.0–1.0, optional): New stakes if the chunk *escalates* or *de-escalates* the proposition (e.g. revealing that a private quarrel will determine the kingdom's fate → stakes spike). Omit if unchanged.
- `audience_default_prior` (float, 0.0–1.0, optional): New audience prior if the narrator has *reframed* the proposition (a red-herring revealed as a real lead → prior rises; a confidently-asserted claim revealed as unreliable → prior falls). Omit if unchanged.
- `description` (str, optional): New human-readable label if the proposition has been *reformulated* on-page (a vague concern crystallises into a specific claim). Omit if unchanged.

### `proposition_truth_commits` — List[PropositionTruthCommit]

One entry per (PROP_id, fabula_time) where the proposition's ground-truth value commits within this chunk (Brewer-Lichtenstein resolution moment). The reconciler folds these into `Proposition.truth_at_fabula`.

- `proposition_id` (str): The PROP_ id from the catalogue.
- `fabula_time` (int): The fabula_time at which the truth commits. MUST equal `triggered_by`'s `fabula_time`.
- `truth` (bool): `true` if the proposition resolves true at this tick, `false` if it resolves false.
- `triggered_by` (str): EVT_ id from this chunk that resolves it. Required. Typically an `outcome` event for `event_occurs`/`outcome` propositions; can be a `revelation` or `utterance` for `identity_is`/`relation_holds` propositions.

> **Schrödinger pattern**: a proposition that flips true→false→true (a body's identity misidentified, a death faked) is legitimate but rare. Emit each commit separately; the reconciler logs the flip but accepts it.

### `concern_snapshots` — List[ConcernSnapshot]

One entry per (CCN_id, fabula_time) where the concern's *per-entity weighting* shifts within this chunk. Fields:

- `concern_id` (str): The CCN_ id from the catalogue.
- `fabula_time` (int): The fabula_time at which the shift commits.
- `triggered_by` (str): EVT_ id from this chunk that caused the shift. Required.
- `salience` (float, 0.0–1.0, optional): New salience if the chunk *intensifies* (a perceived threat materialises) or *relaxes* (the threat is neutralised) the concern. Omit if unchanged.
- `polarity` (str, optional): New polarity if a *desire→fear* (or *fear→desire*) reversal occurred on-page. Macbeth's desire for kingship flipping to fear of losing it after Banquo's prophecy is a classic example. Omit if unchanged.
- `activation_fabula_window` (list[int], optional): New `[start, end]` activation window if rewritten (e.g. a concern that only activates after the abdication). Omit if unchanged.
- `counter_concern_ids` (list[str], optional): New set of counter-concern CCN_ ids if the ambivalence pairing has changed. Omit if unchanged.
- `kind` (str, optional): New harm/benefit-kind label if the concern has been reclassified by an on-page revelation. Omit if unchanged.

### `new_concern_seeds` — List[ConcernSeed]

One entry per *newly-discovered* concern that the catalogue did not seed but that this chunk's events make undeniable. Use sparingly — most concerns should already be in the catalogue. The reconciler dedups against existing seeds on `(entity_id, proposition_id, polarity)`.

- `concern_id` (str): A fresh `CCN_*` id not in the catalogue.
- `entity_id` (str): ENT_ id of the concern-holder from the ontology.
- `proposition_id` (str): An existing catalogue PROP_ id. **You may not invent new propositions here** — when the chunk introduces a concern about something not yet propositionalised, anchor it to the *nearest* existing catalogue PROP whose `referent_ids` overlap the concern's subject; only omit the seed entirely when no plausible catalogue PROP exists at all. The post-pass concern extractor will then catch it.
- `polarity` (str): `"desire"` or `"fear"`.
- `kind` (str | null): Optional harm/benefit-kind label.
- `baseline_salience` (float, 0.0–1.0): Initial salience as established by this chunk.
- `evidence_strength` (str): `"weak"` / `"moderate"` / `"strong"`. Use `"strong"` only when the chunk *explicitly enacts* the concern.
- `counter_concern_ids` (list[str]): Empty unless this seed pairs with another concern in this chunk's emission.

---

## Worked Example

Given a chunk where `EVT_BANQUO_GHOST_APPEARS` (fabula=1820) drives
Macbeth's terror at the banquet, and the catalogue contains:
- `PROP_MACBETH_KEEPS_THRONE` (kind=outcome, baseline stakes=0.7,
  baseline prior=0.5).
- `CCN_MACBETH_FEAR_BANQUO_LINE` (entity=ENT_MACBETH,
  prop=PROP_BANQUO_LINE_RULES, polarity=fear, baseline salience=0.55,
  paired with `CCN_MACBETH_DESIRE_DYNASTY`).
- `CCN_MACBETH_DESIRE_KINGSHIP` (entity=ENT_MACBETH,
  prop=PROP_MACBETH_KEEPS_THRONE, polarity=desire, baseline salience=0.85).

A well-formed Affect output for that chunk:

```json
{
  "proposition_snapshots": [
    {
      "proposition_id": "PROP_MACBETH_KEEPS_THRONE",
      "fabula_time": 1820,
      "triggered_by": "EVT_BANQUO_GHOST_APPEARS",
      "stakes": 0.85,
      "audience_default_prior": 0.40
    }
  ],
  "proposition_truth_commits": [
    {
      "proposition_id": "PROP_BANQUO_DEAD",
      "fabula_time": 1820,
      "truth": true,
      "triggered_by": "EVT_BANQUO_GHOST_APPEARS"
    }
  ],
  "concern_snapshots": [
    {
      "concern_id": "CCN_MACBETH_FEAR_BANQUO_LINE",
      "fabula_time": 1820,
      "triggered_by": "EVT_BANQUO_GHOST_APPEARS",
      "salience": 0.92
    },
    {
      "concern_id": "CCN_MACBETH_DESIRE_KINGSHIP",
      "fabula_time": 1820,
      "triggered_by": "EVT_BANQUO_GHOST_APPEARS",
      "polarity": "fear",
      "salience": 0.78
    }
  ],
  "new_concern_seeds": [
    {
      "concern_id": "CCN_LADY_MACBETH_FEAR_EXPOSURE",
      "entity_id": "ENT_LADY_MACBETH",
      "proposition_id": "PROP_MACBETH_GUILT_DISCOVERED",
      "polarity": "fear",
      "kind": "exposure",
      "baseline_salience": 0.6,
      "evidence_strength": "strong",
      "counter_concern_ids": []
    }
  ]
}
```

Note four patterns:
- **Diff-only snapshots**: only `stakes` and `audience_default_prior` are
  emitted on `PROP_MACBETH_KEEPS_THRONE` because `description` is unchanged.
- **Polarity flip**: `CCN_MACBETH_DESIRE_KINGSHIP` flips `desire → fear`
  (Macbeth no longer wants the throne so much as fears losing it). This
  is rare — only emit when on-page events make the reversal undeniable.
- **Truth commit fabula equals trigger fabula**: 1820 on both sides.
- **`new_concern_seeds`** anchors to an existing catalogue PROP. If the
  chunk introduces a concern with no matching PROP, prefer the *nearest*
  existing PROP rather than silently dropping the seed; if no plausible
  PROP exists, omit it and let the post-pass concern extractor catch it.

---

## Rules

1. **Reference, do not invent.** Every PROP_/CCN_/EVT_ id you mention must already exist (in the catalogue or in this chunk's events). The one exception is `new_concern_seeds.concern_id`, which MUST be a fresh CCN_ id.
2. **Diff, don't redeclare.** Only emit snapshot fields that *change* relative to the prior accumulated state for that PROP/CCN. A `PropositionSnapshot` whose every optional field equals the prior is dropped during coalescence.
3. **Anchor every snapshot to an on-page event.** `triggered_by` must be an EVT_ id from *this chunk*. Snapshots that try to retro-attribute drift to a prior chunk's event are dropped — the upstream chunk should have emitted the snapshot.
4. **Truth commits at the resolving event's fabula_time.** A truth commit's `fabula_time` MUST equal its `triggered_by` event's `fabula_time`. Do not back-date or post-date commits.
5. **One snapshot per (id, fabula_time).** If a proposition's stakes shift twice within the same chunk via two different events, emit two snapshots at the two different fabula_times. If the same event drives both stakes and prior changes, fold them into one snapshot.
6. **Polarity reversals are rare and load-bearing.** Only emit a polarity flip when the chunk's events *clearly* show the entity's wanting flipping to fearing (or vice versa). Salience drift is the common case; polarity reversal is a major narrative beat.
7. **Skip cleanly when nothing affects-relevant happens.** A chunk that touches no catalogue prop and no seeded concern should return four empty lists. Do NOT pad with no-op snapshots.
8. **`new_concern_seeds` is a safety valve, not a primary tool.** If the catalogue is doing its job, you will rarely need it. Use it only when an on-page event makes a concern *undeniable* that the global pass missed.
9. **Close concerns when their proposition resolves.** Whenever you emit a `proposition_truth_commits` entry, walk every catalogue concern whose `prop=` field equals that PROP_ id and emit a `concern_snapshots` entry that *closes* it at the same `fabula_time` and the same `triggered_by`. The closure shape depends on the concern's polarity vs the resolved truth value:
   - **Concern realised** (a `desire` resolved `true`, or a `fear` resolved `false`) → snapshot `salience` to a low value (typically `0.05–0.15`); the standing wanting/dreading is over. Subsequent affect (satisfaction, relief, gratitude) lives on the post-resolution events, not on the concern ledger.
   - **Concern materialised** (a `desire` resolved `false`, or a `fear` resolved `true`) → snapshot `salience` to a low value AND set `activation_fabula_window` to `[<original_start>, <commit_fabula_time>]` so downstream affect (grief, regret, rage) is computed against the *post-resolution* state, not the pre-resolution standing fear/desire. The materialised harm/benefit is now an event on the page, not a standing concern.
   - **Concern that survives resolution** (rare — e.g. a fear of *exposure* about a now-confirmed-true secret remains active because the secret is still secret to the in-world audience) → emit no closure snapshot, but only if the on-page text makes that survival explicit. Default to closure.

   Forgetting closure snapshots is a common failure mode: it leaves characters "fearing" things that have already happened or "desiring" things they already have, double-counting in suspense / surprise scoring and breaking grief / rage detectors. The reconciler logs a warning when a `proposition_truth_commits` arrives without paired closure snapshots for catalogue concerns anchored to that PROP_.

   **Worked closure example.** Suppose `EVT_DUNCAN_KILLED` (fabula=1500) commits `PROP_DUNCAN_DEAD = true` and the catalogue holds:
   - `CCN_MACBETH_DESIRE_THRONE` (entity=ENT_MACBETH, prop=PROP_DUNCAN_DEAD, polarity=desire, salience=0.85) — desire-realised.
   - `CCN_MACDUFF_FEAR_REGICIDE` (entity=ENT_MACDUFF, prop=PROP_DUNCAN_DEAD, polarity=fear, salience=0.60) — fear-materialised.

   The well-formed affect output for that chunk includes:
   ```json
   {
     "proposition_truth_commits": [
       {"proposition_id": "PROP_DUNCAN_DEAD", "fabula_time": 1500,
        "truth": true, "triggered_by": "EVT_DUNCAN_KILLED"}
     ],
     "concern_snapshots": [
       {"concern_id": "CCN_MACBETH_DESIRE_THRONE", "fabula_time": 1500,
        "triggered_by": "EVT_DUNCAN_KILLED", "salience": 0.10},
       {"concern_id": "CCN_MACDUFF_FEAR_REGICIDE", "fabula_time": 1500,
        "triggered_by": "EVT_DUNCAN_KILLED", "salience": 0.10,
        "activation_fabula_window": [0, 1500]}
     ]
   }
   ```
   Pre-1500 the engine still sees Macbeth's desire and Macduff's fear at full salience; post-1500 both close cleanly and grief / regicide-rage detectors fire on the *event*, not on the standing concern.

   **Counter-concern propagation.** When a concern listed in `counter_concern_ids` exists, closing one side of the pair MUST be paired with closing the other side at the same fabula tick — even if the partner concern is anchored to a *different* (logically inverse) proposition that has not itself committed in this chunk. The partner's effective truth is the *inverse* of the trigger commit's truth, so its materialised-vs-realised classification flips accordingly. If you do not emit the partner's closure snapshot, the Phase C reconciler will inject one for you and log a warning; emit it explicitly to silence the warning and keep the on-page event auditable.

   **Multi-commit propositions.** Some propositions resolve more than once over the source text (a character believed dead is revealed alive, then actually killed; a secret is exposed, retracted, then confirmed). When emitting `proposition_truth_commits`, treat each commit independently and emit closure (and, if applicable, re-opening) snapshots for the affected concerns at each commit tick. The reconciler treats the *latest* commit as canonical for closure but honours intermediate explicit re-openings (`concern_snapshots` with `salience >= 0.2` between two commits).

   **Polarity flips and counter-concerns.** When you emit a `concern_snapshots` entry that flips a concern's `polarity` (Rule 6), emit a paired snapshot at the same fabula tick for every concern in its `counter_concern_ids` — the rivalry topology breaks if one side flips and the other does not. Closing the partner at the same tick (via salience<0.2 or `activation_fabula_window` cap) also counts as a valid pairing.
