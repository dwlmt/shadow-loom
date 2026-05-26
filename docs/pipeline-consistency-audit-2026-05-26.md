# Shadow Loom Pipeline Consistency Audit — 2026-05-26

Read-only audit across **ingestion → creative brief → renderer → auditor → refinement**,
focused on (a) per-query-type correctness, (b) renderer ↔ auditor block parity,
(c) feedback-loop wiring back into the brief, and (d) over-strictness risk that
could prevent the inner loop from converging.

---

## Executive Summary (severity-ranked)

1. **Major — `narrative_tension` has no dedicated brief-builder branch, no
   rendering-mode section, and no refinement recipe**, despite being a
   first-class `target_effect` with an audit-category mapping.
   It can pass query validation and reach the renderer/auditor on fallback
   guidance, weakening convergence relative to other effects.
2. **Major — Consequences parity retry under-checks `mutation_social` edges.**
   The async parity helper only inspects `mutation` edges, so social-trait
   mutation gaps do not trigger the same Consequences retry path that
   plain-trait mutations do — pipeline can "converge" while social
   relationship-dynamics parity is still partially unresolved.
3. **Medium — Minor `style_mismatch` can still block convergence**, despite
   the auditor prompt explicitly saying "do NOT spend a regeneration cycle
   on this alone". The minor-severity bypass allowlist in the loop is keyed
   by `violation_type` and does not include `style_mismatch`.

The rest of the pipeline shows strong parity and well-placed convergence
safeguards (failed-open handled, regression rollback, non-converged merge
blocking). The three items above are the only issues worth addressing as
"prompt/contract correctness"; nothing else surfaced as a real bug.

---

## 1. Per-Query-Type Coverage Matrix

| target_effect      | Ingestion | Brief builder branch | Renderer mode section | Audit categories                                  | Refinement recipe | Gap |
|--------------------|-----------|----------------------|------------------------|---------------------------------------------------|-------------------|-----|
| general            | n/a       | n/a (no prose path)  | n/a                    | n/a (short-circuit before auditor)                | n/a               | —   |
| interrogate        | n/a       | n/a (no prose path)  | n/a                    | n/a (short-circuit before auditor)                | n/a               | —   |
| observation        | ✓         | ✓                    | ✓                      | physics + universal                               | generic           | —   |
| intervention       | ✓         | ✓                    | ✓                      | physics + universal                               | generic           | —   |
| counterfactual     | ✓         | ✓                    | ✓                      | physics + universal                               | generic           | —   |
| mystery            | ✓         | explicit             | explicit               | epistemic + physics + universal                   | explicit          | —   |
| dramatic_irony     | ✓         | explicit             | explicit               | epistemic + physics + universal                   | explicit          | —   |
| surprise           | ✓         | explicit             | explicit               | epistemic + physics + universal                   | explicit          | —   |
| suspense           | ✓         | explicit             | explicit               | probabilistic + physics + universal               | explicit          | —   |
| fear               | ✓         | explicit             | explicit               | probabilistic + physics + universal               | explicit          | —   |
| joy                | ✓         | explicit             | explicit               | probabilistic + physics + universal               | explicit          | —   |
| regret             | ✓         | explicit             | explicit               | counterfactual + physics + universal              | explicit          | —   |
| grief              | ✓         | explicit             | explicit               | counterfactual + physics + universal              | explicit          | —   |
| rage               | ✓         | explicit             | explicit               | counterfactual + physics + universal              | explicit          | —   |
| love               | ✓         | explicit             | explicit               | counterfactual + physics + universal              | explicit          | —   |
| **narrative_tension** | ✓      | **MISSING**          | **MISSING**            | epistemic + probabilistic + physics + universal   | **MISSING**       | **Major** |

Anchor references:

- Query-type short-circuit (general/interrogate skip the auditor as
  documented): [shadow_loom/pipeline.py](shadow_loom/pipeline.py#L2650).
- `target_effect` enum (includes `narrative_tension`):
  [shadow_loom/query_models.py](shadow_loom/query_models.py#L701).
- `EFFECT_AUDIT_CATEGORIES` table (includes `narrative_tension`):
  [shadow_loom/auditor.py](shadow_loom/auditor.py#L80).
- Directive-assembly per-effect branches (no `narrative_tension` branch):
  [shadow_loom/directive_assembly.py](shadow_loom/directive_assembly.py#L6340),
  [shadow_loom/directive_assembly.py](shadow_loom/directive_assembly.py#L7142).
- Renderer rendering-mode sections (no `narrative_tension` mode):
  [shadow_loom/prompts/generation.md](shadow_loom/prompts/generation.md#L50).
- Refinement recipes (no `narrative_tension` recipe):
  [shadow_loom/prompts/refinement.md](shadow_loom/prompts/refinement.md#L42),
  [shadow_loom/prompts/refinement.md](shadow_loom/prompts/refinement.md#L87).

---

## 2. Auditor Strictness / Convergence Risk

| # | Risk | Where | Verdict |
|---|------|-------|---------|
| 2.1 | `compute_overall_pass` defaults: `min_foreshadowing_score=0.6`, `min_cognitive_plausibility=0.7`, `max_affective_loss=0.3`, `max_miracle_steps=0` | [shadow_loom/auditor.py](shadow_loom/auditor.py#L611-L640) | Reasonable. Miracle-step accounting correctly routes `cycle`, `noisy_or_absorbed`, and (under noisy-OR mode) `inertia` blocks to the `noisy_or_absorbed` bucket instead of `miracle_steps`, so `max=0` does not over-fire. |
| 2.2 | `_engine_thresholds_check` parity with `compute_overall_pass` | [shadow_loom/auditor.py](shadow_loom/auditor.py#L1100-L1175) | Mirrored: same fields, same comparisons, same `ignore_spatial_blocks` filter. |
| 2.3 | Severity routing: minor violations forcing regen | [shadow_loom/auditor.py](shadow_loom/auditor.py#L4214-L4232) | **Risk.** `style_mismatch` is omitted from the minor-only bypass allowlist even though `auditor.md` and the audit-prompt assembler both classify it as the canonical "do not spend a regen cycle" case. |
| 2.4 | Paraphrase Jaccard / containment thresholds (0.45 / 0.6 with ≥4 token overlap, ≥6 token min) | [shadow_loom/auditor.py](shadow_loom/auditor.py#L2807-L2832) | Conservative enough: short fragments are skipped, only content-words count, and stop-words stripped. Low false-positive risk. |
| 2.5 | `max_iterations` default | [shadow_loom/auditor.py](shadow_loom/auditor.py#L585) | Default 3 with regression rollback. Combined with prior-feedback non-regression block, convergence behaviour is sound. |
| 2.6 | Hard violations without a refinement recipe | [shadow_loom/prompts/refinement.md](shadow_loom/prompts/refinement.md) | See §4 — three deterministic violation types are unrepresented; none are loop-blockers (their `feedback` field already carries the per-instance fix) but adding them would tighten convergence. |

---

## 3. Renderer ↔ Auditor Block Parity

Both prompts are explicitly engineered to mirror each other. Verified:

| Block | Renderer (generation.py / generation.md) | Auditor (auditor.py / auditor.md) | Match |
|-------|------------------------------------------|------------------------------------|-------|
| STORY SO FAR (preceding prose) | [generation.py:3599](shadow_loom/generation.py#L3599) | [auditor.py:1479](shadow_loom/auditor.py#L1479) | ✓ (same `preceding_prose_max_chars` cap, same shadow-branch precedence rule) |
| BRANCH CONTEXT (shadow vs factual) | [generation.py:3679](shadow_loom/generation.py#L3679) | [auditor.py:1526](shadow_loom/auditor.py#L1526) | ✓ |
| STYLE FIDELITY (band + density + form-class) | generation.md L23–48, [generation.py:3720](shadow_loom/generation.py#L3720) | [auditor.py:1567–1610](shadow_loom/auditor.py#L1567) | ✓ (adjusted_word_band shared via `narrative_style.py`) |
| USER'S ORIGINAL REQUEST | renderer prompt header | [auditor.py:1556](shadow_loom/auditor.py#L1556) | ✓ |
| CONSTRAINTS (ConstraintBlock list) | [generation.py:3774](shadow_loom/generation.py#L3774) | auditor.py `_format_constraints_for_audit` | ✓ |
| RENDERING DIRECTIVE (mode/pacing/POV/tone) | [generation.py:3805](shadow_loom/generation.py#L3805) | [auditor.py:1610](shadow_loom/auditor.py#L1610) | ✓ (POV-temporal rule explicitly mirrored) |
| PHYSICS OVERRIDE | generation.md §"PHYSICS OVERRIDE" | [auditor.py:1680](shadow_loom/auditor.py#L1680) | ✓ |
| HIDDEN CHANNELS / UTTERANCES | generation rule block | [auditor.py:1697](shadow_loom/auditor.py#L1697) | ✓ |
| ERASED UTTERANCES / DISABLED CHANNELS / SEVERED CAUSAL CHAINS / DEPENDENT-STATE SUBSTITUTIONS / FALSE PROPOSITIONS / PREVENTED EVENTS | inline HARD constraints in brief (built by directive_assembly) | [auditor.py:1731–2005](shadow_loom/auditor.py#L1731) | ✓ |
| BLOCKED PROPAGATIONS (HARD) | generation rule §Intervention | auditor.md §"Blocked-propagation precedence" | ✓ (no-trait-name-recitation rule mirrored on both sides) |
| INERT INTERVENTION | inline HARD via `_build_inert_intervention_constraints` | auditor.md §"Inert-intervention precedence" | ✓ |
| SCENE CONTEXT (`format_scene_context_for_prompt`) | [generation.py:3830](shadow_loom/generation.py#L3830) | auditor.py uses same formatter | ✓ |
| THREAT PROXIMITY / INTERVENTION SANDBOX / COUNTERFACTUAL BRANCH / CAUSAL ATTRIBUTION / ENTANGLEMENT / ABDUCTION TRUTHS / INTERVENTION MECHANISMS | renderer formatters (`_format_*`) reused | reused via direct import (auditor.py:43–53) | ✓ (single source of truth — drift impossible) |
| Profile blocks: SURPRISE / IRONY / MYSTERY / FEAR / JOY / REGRET / GRIEF / RAGE / LOVE | [generation.py:3907](shadow_loom/generation.py#L3907) | `_format_propositional_context` in auditor.py | ✓ |
| EXTERNAL RESEARCH (WorldFact) | renderer prompt block | [auditor.py:1875](shadow_loom/auditor.py#L1875) (`reasoning_failure` + `world_fact_fidelity:` prefix) | ✓ |
| `introduced_elements` / reuse-first rule | generation.md schema + Rule | auditor.md categories 4c/4d | ✓ |
| Rule 10 (no meta-narration) — universal | generation.md Rule 10 | auditor.md Category 4b | ✓ |
| Rule 11 (event spatial anchoring / implicit co-location / phantom-witness ban) | generation.md Rule 11 | auditor.md Literal entries `event_location_mismatch`, `event_copresence_violation`, `event_copresence_omission` | ✓ |

No drift detected between the two prompts on any structural block.

---

## 4. Feedback-Loop Wiring

### 4.1 violation_type → refinement-recipe coverage

`refinement.md` documents correction patterns for the original 12
violation families. The auditor `Literal` has since grown to ~30 entries.
Unrepresented entries (each carries its own per-instance `feedback`
field, so the rewriter is not blind — but a recipe would tighten loops):

- `withheld_utterance_leak`, `pruned_utterance_leak`, `disabled_channel_leak`,
  `blocked_propagation_leak`
- `belief_provenance_contradiction`, `utterance_truth_contradiction`,
  `channel_intelligibility_violation`
- `inert_intervention_aftermath`
- `style_mismatch`, `meta_narration`
- `undeclared_element`, `unjustified_introduction`
- `object_misuse`, `object_position_mismatch`, `entity_position_mismatch`
- `event_location_mismatch`, `event_copresence_violation`,
  `event_copresence_omission`
- `spurious_abduction`, `premature_payoff`

Most are deterministic-precheck violations whose `feedback` is templated
to be self-contained. Adding short recipes for the high-frequency ones
(`style_mismatch`, `meta_narration`, `undeclared_element`,
`event_copresence_*`) would help the rewriter when several fire at once.

### 4.2 Brief mutation hooks across iterations

- `_inject_miracle_step_mechanisms` (auditor.py) appends
  `InterventionMechanism` entries to `brief.intervention_mechanisms`
  for every blocked propagation that the engine or auditor flagged.
  The next render pass picks them up because the renderer prompt
  rebuilds from the (now-mutated) brief.
- Style re-anchor: when a style violation has fired in this or any
  prior iteration, `_build_refinement_prompt` re-emits the
  `STYLE FIDELITY` block immediately before the rewrite footer so the
  contract sits in the same attention window as the violation list.
  [auditor.py:2358–2402](shadow_loom/auditor.py#L2358).
- Non-regression block: prior-iteration violations whose
  `violation_type` is *not* in the current set are surfaced as
  "must remain fixed" constraints to prevent ping-ponging.
- Token-budget guard: feedback truncated at 8 000 chars to avoid
  blowing the rewriter's context.

### 4.3 Engine-threshold surfacing

`_build_refinement_prompt` accepts an `engine_failures: list[str]`
parameter and emits an `=== ENGINE-THRESHOLD FAILURES ===` block above
the violation list, with the same numeric labels the scorecard uses.
`refinement.md` does not yet mention this block specifically — the
rewriter has to infer that the listed thresholds must move; an
explicit recipe ("when `cognitive_plausibility_score < min`, add
mechanism beats for the listed miracle-step ids") would tighten the
loop.

---

## 5. Pipeline Dispatch

- `general` and `interrogate` short-circuit before render/audit:
  [pipeline.py:2650](shadow_loom/pipeline.py#L2650).
- All other query types render → audit → (refine if needed) → merge.
- `resolve_audit_categories` falls back to `["physics"] + universal`
  for any unmapped effect, preserving safety.
- Non-converged audits block merge unless explicitly overridden:
  [pipeline.py:2962](shadow_loom/pipeline.py#L2962),
  [pipeline.py:2973](shadow_loom/pipeline.py#L2973).
- Failed-open audits (LLM error fallback) are explicitly marked
  non-converged so a green pass cannot be faked by a transient
  network failure: [auditor.py:4100](shadow_loom/auditor.py#L4100).
- Regression rollback restores the prior better draft when violation
  count increases across iterations: [auditor.py:4137–4182](shadow_loom/auditor.py#L4137).

---

## 6. Inert-Intervention Path Consistency

All three layers agree:

- Brief: `_build_inert_intervention_constraints` emits a HARD block
  telling the renderer to depict the *attempted* surgery and its
  *resistance only* — no downstream consequence.
- Renderer: generation.md INTERVENTION mode says "render the physical
  struggle that forced the state change" and the BLOCKED PROPAGATIONS
  rules say "NEVER stage a listed trait as a near-change or as a
  subtle cue of a shift".
- Auditor: auditor.md "Inert-intervention precedence" rule (Category 4)
  forbids firing `miracle_step` / `magnitude_too_low` /
  `abduction_failure` / `affective_failure` / `reasoning_failure` on
  inert scenes, and adds `inert_intervention_aftermath` for the
  inverse failure mode (prose depicting "the shift held").
- compute_causal_feedback explicitly routes blocks under
  `intervention_inert=True` into `noisy_or_absorbed_propagations`
  instead of `miracle_steps` so engine-threshold checks do not pin
  `engine_passed=False`.

No inconsistencies.

---

## 7. Negative-Physics Block Coverage

`auditor.md` references the `PREVENTED EVENTS (HARD)`,
`FALSE PROPOSITIONS (HARD)`, `SEVERED CAUSAL CHAINS (HARD)` and
`DEPENDENT-STATE SUBSTITUTIONS (HARD)` constraint blocks as expected
inputs and emits a typed `reasoning_failure` violation with a
specific rationale prefix for each (`prevented_event:`,
`false_proposition:`, `severed_causal_chain:`,
`dependent_state_substitution:`). The auditor-prompt assembler scans
`brief.constraints` for these strings and emits the matching
NEGATIVE PHYSICS reminder block. Verified in
[auditor.py:1731–1816](shadow_loom/auditor.py#L1731). Directive
assembly emits these blocks consistently across observation,
intervention and counterfactual branches.

---

## 8. Ingestion → Downstream Coverage

Verified — ingestion produces everything the renderer/auditor depend on:

| Required by downstream | Ingestion source |
|------------------------|------------------|
| `event.at_location_id` (for spatial / co-presence audits) | physics_extraction.md + ingestion repair pass |
| object `affordances`, `state_timeline` (for `reconstruct_object_at`) | ontology_objects.md + consequences_extraction.md |
| entity `state_timeline` (for `reconstruct_entity_at`) | consequences_extraction.md |
| channel `intelligibility` (per-listener) | social_extraction.md |
| `utterance.content` + `truth_value` + speaker/addressee | social_extraction.md |
| `narrative_style` (`format`, `target_word_min/max`, `prose_density`, `voice`, `style_exemplar`) | inferred at ingestion tail: [ingestion.py:19546](shadow_loom/ingestion.py#L19546) |
| Propositions with `activation_fabula_window` + `polarity` (for `premature_payoff`) | proposition_catalogue.md + proposition_catalogue_concerns.md |
| Belief provenance (`acquired_via_event_id` / `acquired_via_channel_id`) | concern_extraction.md + belief_proposition_clustering.md |
| `WorldFact` external research records | research_extraction.md |

**Gap: mutation_social parity retry** — see Executive Summary §2:

- Consequences prompt requires both `mutation` and `mutation_social`
  parity in entity updates:
  [prompts/consequences_extraction.md:11,62,68](shadow_loom/prompts/consequences_extraction.md#L11).
- The async parity helper only checks `mutation` edges, not
  `mutation_social`: [ingestion.py:6854](shadow_loom/ingestion.py#L6854).
- The async retry trigger uses that helper:
  [ingestion.py:8642](shadow_loom/ingestion.py#L8642).
- Downstream validator surfaces the gap as a warning, not a hard
  failure: [ingestion.py:15014](shadow_loom/ingestion.py#L15014).

Pipeline can therefore "converge" with social-mutation parity still
partially unresolved.

---

## 9. Style / Form-Class Budget Triplication Check

The per-format quantitative thresholds (`synopsis` ≤2 sentences /
0% dialogue; `scene` 3–8 sentences; `short_story` 4–12;
`novel_excerpt` 6–20; `screenplay` action+speaker-tagged dialogue;
`verse` line-broken; `news_article` ≤4 sentences + inverted pyramid;
`transcript` speaker turns only; `[Composition rule | HARD]`
compressed-POV override applies `synopsis` thresholds + POV lock +
≤15% interior monologue) appear identically in:

- `prompts/generation.md` (renderer)
- `prompts/auditor.md` (auditor)
- `prompts/refinement.md` (rewriter)

No drift. The `adjusted_word_band` ±50% loosening and the
`narrative_style.py` formatter are imported into both the renderer
and the auditor (single source of truth for the band).

---

## 10. `violation_type` Literal vs `auditor.md` Parity

Compared the `auditor.md` enumeration in the §Output Schema block
against the `Literal[...]` in `AuditViolation.violation_type`
([auditor.py:340–500](shadow_loom/auditor.py#L340)). All entries
match character-for-character. No risk of Pydantic validation failures
on a typed string the LLM might emit. (The Literal additionally
documents — via comments — the rationale prefixes the LLM may apply,
which is not part of the type system but is mirrored in `auditor.md`.)

---

## 11. Recommended Concrete Changes

Numbered by severity. Each one is small and surgical.

### R1 (Major) — Wire `narrative_tension` end-to-end

- **shadow_loom/directive_assembly.py** — add an explicit
  `narrative_tension` branch in the per-effect dispatch that builds
  the `NARRATIVE TENSION` payload (displacements, withheld_causes,
  upcoming_revelations) as a first-class brief field. Today the
  composite is *computed* (the `compute_tension_score` path) but
  no constraint block / payload is assembled for the renderer.
- **shadow_loom/prompts/generation.md** — add a dedicated
  `NARRATIVE TENSION` rendering-mode subsection under §Rendering
  Modes, alongside the per-effect modes already documented (the
  current single paragraph under Category 1 is descriptive, not a
  directive-mode contract).
- **shadow_loom/prompts/refinement.md** — add a "### Narrative
  Tension Threshold Not Met" recipe paragraph paralleling
  "Suspense Threshold Not Met".

### R2 (Major) — Extend Consequences mutation parity to `mutation_social`

- **shadow_loom/ingestion.py:6854** — extend the async parity helper
  to also walk `mutation_social` edges (with `rel_counterpart_id`
  cross-check) and signal a retry when the dyad-update is missing.
- Cross-check [ingestion.py:8642](shadow_loom/ingestion.py#L8642)
  and [ingestion.py:15014](shadow_loom/ingestion.py#L15014) to ensure
  the same predicate is used in retry-trigger and validator-warning
  paths.

### R3 (Medium) — Add `style_mismatch` to the minor-severity bypass

- **shadow_loom/auditor.py:4214–4232** — extend the minor-only
  bypass allowlist to include `style_mismatch` so the loop can
  converge on prose whose only outstanding issue is a `minor`-tier
  style drift (the auditor prompt already documents this intent at
  [auditor.py:1593](shadow_loom/auditor.py#L1593) but the
  feedback-loop code does not honour it).

### R4 (Minor) — Add explicit refinement recipes for high-frequency new violation types

- **shadow_loom/prompts/refinement.md** — append short recipes for
  `meta_narration`, `undeclared_element`,
  `event_copresence_violation`, `inert_intervention_aftermath`, and
  `unjustified_introduction`. Each rewriter today receives the
  templated `feedback` field; a short recipe (≤3 bullets per
  violation) would reduce loop iterations when several fire at once.

### R5 (Minor) — Document the engine-threshold block in refinement.md

- **shadow_loom/prompts/refinement.md** — add a brief paragraph
  describing the `=== ENGINE-THRESHOLD FAILURES ===` block surfaced
  by `_build_refinement_prompt` and how each threshold should
  influence the rewrite (e.g. `foreshadowing_payoff_score < min` →
  pay off an unresolved chain; `cognitive_plausibility_score < min` →
  add mechanism beats for the listed miracle-step ids).

---

## Final Assessment

The pipeline architecture is largely coherent. Renderer ↔ auditor
mirroring is unusually strong (helper formatters are shared by direct
import, so prompt drift is structurally impossible for the major
payload blocks). Convergence safeguards (failed-open handling,
regression rollback, non-converged-merge blocking) are well-placed
and prevent the most common failure modes.

The only material correctness gap is `narrative_tension` (R1). The
only material ingestion-consistency gap is `mutation_social` parity
(R2). The only material over-strictness risk is the
`style_mismatch` minor-bypass omission (R3). The remaining items
(R4, R5) are quality-of-life refinements that would shave iterations
off the typical convergence path without changing any structural
contract.
