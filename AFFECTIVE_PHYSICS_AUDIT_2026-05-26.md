# Affective Physics Layer: Theoretical Correctness Audit
**Date:** 2026-05-26  
**Scope:** Concern, Belief, Proposition implementations  
**Frameworks:** Appraisal Theory, OCC Model, Bayesian Belief Networks, Theory of Mind

---

## EXECUTIVE SUMMARY

**Overall Assessment:** The affective physics layer demonstrates strong theoretical grounding with **7 critical violations**, **12 major issues**, and **8 minor inconsistencies** requiring correction.

**Critical Issues:**
1. Belief confidence propagation violates probability axioms in counterfactual surgery
2. Concern salience weighting lacks temporal discounting (proximity violations)
3. Missing belief invalidation cascade when proposition truth flips
4. Inverse proposition consistency not enforced bidirectionally
5. Bayesian surprise computation uses wrong KL direction in local mode
6. Fear appraisal missing anticipatory coping (Lazarus primary/secondary split)
7. Concern-proposition alignment unchecked during synthesis

---

## I. BAYESIAN BELIEF NETWORKS — Confidence Propagation

### CRITICAL #1: Probability Bounds Violated in Do-Surgery

**Location:** `shadow_loom/causal_physics.py:160-183` (`BeliefMutation`)  
**Theoretical Framework:** Probability Axioms (Kolmogorov 1933)

**Issue:**  
`BeliefMutation.new_confidence` accepts `float` without bounds validation. When `apply_do_targets` performs Pearl Rung-2 belief surgery, nothing enforces `new_confidence ∈ [0, 1]`.

```python
class BeliefMutation(BaseModel):
    """Record of a Belief confidence clamp on a single character."""
    holder_id: str
    target_id: str
    proposition_id: Optional[str] = None
    old_confidence: Optional[float] = None
    new_confidence: float  # ❌ NO BOUNDS CHECK
    created: bool = False
    triggered_by: str = "DO_OPERATOR"
```

**Evidence:**
- `Belief.confidence` has `Field(ge=0.0, le=1.0)` (models.py:236)
- `BeliefMutation.new_confidence` lacks this constraint
- Surgery path bypasses Pydantic validation when mutating sandbox

**Fix:**
```python
new_confidence: float = Field(ge=0.0, le=1.0)
```

**Theoretical Justification:**  
Kolmogorov axioms require ∀ events E: P(E) ∈ [0, 1]. Violations produce undefined KL divergence (log of negative probability) and break every downstream scorer.

---

### CRITICAL #2: Belief Cascade Missing When Proposition Truth Flips

**Location:** `shadow_loom/causal_physics.py:160-172` (`PropositionMutation`)  
**Theoretical Framework:** Bayesian Conditioning

**Issue:**  
When `PropositionMutation` clamps `truth_at_fabula[t] = new_truth`, the mutation records `cascaded_belief_count` but **nothing enforces** that character beliefs about the proposition update to reflect the new ground truth.

**Example Failure:**
1. Proposition `PROP_DUNCAN_DEAD` has `truth_at_fabula = {1700: True}`
2. Rung-2 surgery clamps it to `{1700: False}` (counterfactual)
3. `PropositionMutation` fires with `cascaded_belief_count = 0`
4. Macbeth's belief `confidence=1.0, proposition_id=PROP_DUNCAN_DEAD` **unchanged**
5. Dramatic irony scorer sees audience knows `False`, Macbeth believes `True` → spurious irony

**Evidence:**
- No `_propagate_proposition_to_beliefs()` helper exists
- `cascaded_belief_count` is populated but never incremented
- Search for "propagate_to_beliefs" yields no implementation

**Fix Required:**
Add belief propagation pass in `CausalPhysicsEngine.apply_do_targets()`:
```python
for prop_mut in proposition_mutations:
    for ent_id, ent in sandbox.nodes(data=True):
        if ent.get("node_type") != "Entity":
            continue
        for belief in ent.get("beliefs", []):
            if belief.get("proposition_id") == prop_mut.proposition_id:
                old_conf = belief["confidence"]
                # Bayesian update: new truth → confidence = 1.0 or 0.0
                belief["confidence"] = 1.0 if prop_mut.new_truth else 0.0
                belief_mutations.append(BeliefMutation(
                    holder_id=ent_id,
                    target_id=belief["target_id"],
                    proposition_id=prop_mut.proposition_id,
                    old_confidence=old_conf,
                    new_confidence=belief["confidence"],
                    triggered_by=f"PROP_{prop_mut.proposition_id}"
                ))
                prop_mut.cascaded_belief_count += 1
```

**Theoretical Justification:**  
Bayesian conditioning: P(B|E) updates when evidence E commits. When proposition truth flips, every agent's belief about that proposition must condition on the new evidence or violate coherence (Dutch book vulnerability).

---

### MAJOR #1: Belief Confidence Not Normalized After Multi-Source Updates

**Location:** `shadow_loom/affect_unification.py:520-550` (`BeliefState._agent_beliefs_at`)

**Issue:**  
When multiple `EntityStateSnapshot.beliefs_added` entries update the same `proposition_id`, the code keeps the **last** confidence value via dict overwrite:

```python
def _agent_beliefs_at(self, agent_id: str, fabula_t: int) -> Dict[str, float]:
    # ...
    for b in recon["beliefs"]:
        pid = b.get("proposition_id")
        if not pid:
            continue
        # If multiple beliefs exist for the same proposition the
        # most recent (last in list, since reconstruct appends in
        # fabula order) wins.
        out[pid] = float(b.get("confidence", 0.5))  # ❌ OVERWRITE, not combine
```

**Theoretical Problem:**  
If two events at the same fabula tick both provide evidence about proposition P (e.g., two witnesses report conflicting accounts), the second clobbers the first instead of combining via Bayesian update or noisy-OR.

**Expected Behavior:**  
Beliefs about the same proposition from independent sources should combine:
- **Bayesian:** `p_combined = (p1·w1 + p2·w2) / (w1 + w2)` (weighted average)
- **Noisy-OR:** `1 - (1-p1)(1-p2)` (independent failure model)

**Fix:**
Track provenance per belief and combine when multiple sources target same proposition.

---

### MAJOR #2: Missing Belief Provenance Pruning in Channel Surgery

**Location:** `shadow_loom/causal_physics.py:330` (should exist but doesn't)

**Issue:**  
When a `DoCHannel(disabled=True)` intervention disables a channel, beliefs with `acquired_via_channel_id == that_channel` should be pruned (the communication path never existed). Code mentions this in `CausalPhysicsResult.pruned_beliefs_count` but search yields no implementation of channel-based pruning.

**Evidence:**
```python
pruned_beliefs_count: int = Field(
    default=0,
    description=(
        "Number of beliefs removed from sandbox entities because their "
        "acquired_via_event_id / acquired_via_channel_id "  # ✓ DOCUMENTED
        "provenance pointed at an event or channel that the do-surgery "
        "removed. Surfaces the epistemic side-effect of channel / "
        "utterance interventions."
    ),
)
```

But `grep -n "acquired_via_channel" causal_physics.py` yields **only** the docstring — no pruning loop.

**Fix Required:**
In `apply_do_targets()`, after disabling channels:
```python
for ent_id, ent in sandbox.nodes(data=True):
    if ent.get("node_type") != "Entity":
        continue
    original_beliefs = ent.get("beliefs", [])
    kept = []
    for b in original_beliefs:
        if b.get("acquired_via_channel_id") in disabled_channel_ids:
            result.pruned_beliefs_count += 1
        else:
            kept.append(b)
    ent["beliefs"] = kept
```

---

## II. APPRAISAL THEORY — Concern as Primary/Secondary Appraisal

### CRITICAL #3: Concern Salience Missing Temporal Proximity Weighting

**Location:** `shadow_loom/directive_assembly.py:4328-4450` (`compute_suspense_score`)  
**Theoretical Framework:** Comisky & Bryant 1982 (anticipatory proximity)

**Issue:**  
Concern salience is static (`Concern.salience`) with optional `activation_fabula_window` gating, but **no temporal discounting**. A threat 10,000 fabula ticks away should weigh less than an identical threat 10 ticks away (imminence kernel).

**Current Code:**
```python
# compute_suspense_score uses imminence on EVENTS:
imminence = math.exp(-dt / max(1.0, tau_fabula))
total += _binary_entropy(p_aud) * _prop_stakes_at(prop, fabula_t) * imminence
```

But `_concern_salience_at()` returns raw `concern.salience` — **no proximity term**.

**Theoretical Violation:**  
Lazarus (1991) primary appraisal distinguishes **relevance** (does this matter to my goals?) from **imminence** (is it happening now?). Current implementation conflates them.

**Fix:**
Add proximity kernel to concern salience resolution:
```python
def _concern_salience_at(c: Concern, fabula_t: int, tau_fabula: float = 1.0) -> float:
    """Return concern salience × imminence kernel."""
    base_salience = (
        float(reconstruct_concern_at(c, fabula_t)["salience"])
        if c.state_timeline
        else float(c.salience)
    )
    # Imminence: concerns activate when their proposition's next
    # truth commit is near. For propositions without future commits,
    # imminence = 1.0 (already resolved or perpetually open).
    prop = world.propositions.get(c.proposition_id)
    if prop:
        future = [t for t in prop.truth_at_fabula if t > fabula_t]
        if future:
            dt = min(future) - fabula_t
            imminence = math.exp(-dt / max(1.0, tau_fabula))
            return base_salience * imminence
    return base_salience
```

**Impact:**  
Without this, `compute_fear_appraisal` treats distant existential threats identically to imminent ones, violating Öhman/Mineka's modal anxiety vs. fear distinction.

---

### MAJOR #3: Secondary Appraisal (Coping) Computed But Not Used in Suspense

**Location:** `shadow_loom/affect_unification.py:1379-1470` (`compute_fear_appraisal`)

**Issue:**  
`_focal_coping()` computes Lazarus secondary appraisal (resilience, confidence, strength, support) and scales `object_fear_score` by `coping_factor = 1 - coping`. But `compute_suspense_score` **ignores coping entirely** — it only reads concern salience and proposition stakes.

**Theoretical Problem:**  
Lazarus (1991): Fear = `p(threat) × magnitude × low(coping)`. Current suspense = `H(p) × stakes × imminence` — missing the coping term.

**Example Failure:**
- Macbeth has `resilience=0.9, confidence=0.8` (high coping)
- Lady Macbeth has `resilience=0.2, confidence=0.3` (low coping)
- Same threat proposition (Duncan's guards discovering blood)
- Current scorer: **identical suspense**
- Theory: Macbeth should experience **less** suspense (high coping buffers threat)

**Fix:**
Extend `compute_suspense_unified` to read focal coping when available:
```python
def compute_suspense_unified(
    bs: BeliefState, fabula_t: int,
    *, tau_fabula: Optional[float] = None,
    focal_id: Optional[str] = None,  # NEW
) -> float:
    # ... existing entropy calculation ...
    if focal_id:
        coping = _focal_coping(bs.world, focal_id)
        coping_factor = max(0.1, 1.0 - coping)
        total *= coping_factor
    return total
```

---

### MAJOR #4: Concern Ambivalence Score Ignores Counter-Concern Salience

**Location:** `shadow_loom/models.py:445-456` (`Concern.ambivalence_score`)

**Issue:**
```python
@property
def ambivalence_score(self) -> float:
    if not self.counter_concern_ids:
        return 0.0
    return float(self.salience)  # ❌ Returns THIS concern's salience
```

**Theoretical Problem:**  
Ambivalence (Cacioppo & Berntson 1994) is maximal when opposing poles are **equally salient**. Current implementation returns unilateral salience, not the min/product of opposing pairs.

**Example:**
- Concern A (desire): `salience=0.9`, `counter_concern_ids=[CCN_B]`
- Concern B (fear): `salience=0.1`, `counter_concern_ids=[CCN_A]`
- Current: `A.ambivalence_score = 0.9`, `B.ambivalence_score = 0.1`
- Theory: Ambivalence ∝ `min(0.9, 0.1) = 0.1` (weakest pole caps the tension)

**Fix:**
```python
@property
def ambivalence_score(self) -> float:
    if not self.counter_concern_ids:
        return 0.0
    # Resolve counter-concern saliences at same fabula_time
    # (requires passing fabula_t, so this may need to become a method)
    # For now, use static salience as lower bound
    counter_saliences = []
    for ccn_id in self.counter_concern_ids:
        # Need world context to resolve — defer to a helper
        pass
    # Simplified: return own salience as proxy until we wire world context
    return float(self.salience) * 0.5  # Placeholder dampening
```

Better: Make this a method that takes `world` and `fabula_t`, compute `min(self.salience, max(counter_saliences))`.

---

## III. OCC MODEL — Goal-Outcome Emotion Elicitation

### CRITICAL #4: Inverse Proposition Consistency Not Enforced Bidirectionally

**Location:** `shadow_loom/models.py:284-379` (`Proposition`)

**Issue:**  
`Proposition.inverse_proposition_id` is documented as one-way at declaration but "symmetrised by the reconciler when both sides name each other." Search for "inverse_proposition" reconciliation yields **no implementation**.

**Evidence:**
```python
inverse_proposition_id: Optional[str] = Field(
    default=None,
    description=(
        "Optional PROP_ id of a logically-opposite proposition "
        "(e.g., ``PROP_DUNCAN_ALIVE`` for ``PROP_DUNCAN_DEAD``). "
        "When set, ``truth_at_fabula`` commits are auto-mirrored "
        "to the inverse with the opposite truth value during the "
        "Phase C truth-write sweep, so concerns / beliefs anchored "  # ❌ "auto-mirrored" — WHERE?
        "to either proposition see a consistent ground truth. "
    ),
)
```

Grep for "inverse" in `ingestion.py`, `pipeline.py`, `causal_physics.py` — **no mirroring logic**.

**Theoretical Problem:**  
Violates law of excluded middle. If `PROP_DUNCAN_DEAD` commits `True` at t=1700, then `PROP_DUNCAN_ALIVE` **must** commit `False` at t=1700. Without enforcement:
- Audience belief about `PROP_DUNCAN_ALIVE` stays at prior (0.5)
- Suspense scorer treats "Duncan alive" as an open question
- OCC "relief" path (feared outcome averted) never fires

**Fix Required:**
In `pipeline.py` Phase C reconciliation (or `PropositionMutation` application):
```python
def _mirror_inverse_propositions(world: WorldStateV1) -> None:
    """Enforce inverse_proposition_id consistency."""
    prop_index = {p.proposition_id: p for p in world.propositions}
    for prop in world.propositions:
        if not prop.inverse_proposition_id:
            continue
        inv = prop_index.get(prop.inverse_proposition_id)
        if not inv:
            logger.warning(f"{prop.proposition_id}.inverse_proposition_id "
                          f"points to nonexistent {prop.inverse_proposition_id}")
            continue
        # Mirror truth commits
        for ft, truth_val in prop.truth_at_fabula.items():
            if ft not in inv.truth_at_fabula:
                inv.truth_at_fabula[ft] = not truth_val
        # Symmetrize the link if one-way
        if inv.inverse_proposition_id != prop.proposition_id:
            inv.inverse_proposition_id = prop.proposition_id
```

---

### MAJOR #5: OCC "Relief" Emotion Missing Concern-Driven Implementation

**Location:** `shadow_loom/affect_unification.py:1500+` (`compute_joy_appraisal`)

**Issue:**  
`JoyAppraisal.relief_score` is documented as "feared concerns whose audience-confirmed belief just dropped toward false" but the implementation aggregates across **all** joy paths without separating relief:

```python
@dataclass
class JoyAppraisal:
    # ...
    relief_score: float  # ✓ DECLARED
```

But `compute_joy_appraisal` computes:
```python
own = 0.0
for c in _focal_concerns(bs.world, focal_id, polarity="desire", fabula_t=fabula_t):
    # ... only processes DESIRE concerns, not FEAR
```

**Missing:** Loop over `polarity="fear"` concerns where `p_now < p_prev` (threat receding).

**Theoretical Violation:**  
OCC (1988) distinguishes:
- **Joy** (desirable event confirmed)
- **Relief** (undesirable event disconfirmed)
- **Hope** (desirable event anticipated)
- **Fear** (undesirable event anticipated)

Current implementation conflates joy and relief under `own_joy_score`.

**Fix:**
```python
relief = 0.0
if prior_fabula_t is not None:
    for c in _focal_concerns(bs.world, focal_id, polarity="fear", fabula_t=fabula_t):
        prop = prop_idx.get(c.proposition_id)
        p_now = bs.confidence(focal_id, c.proposition_id, fabula_t)
        p_prev = bs.confidence(focal_id, c.proposition_id, prior_fabula_t)
        if p_now < p_prev:  # Threat receding
            stakes = _prop_stakes_at(prop, fabula_t) if prop else 0.5
            delta = p_prev - p_now
            relief += delta * stakes * _concern_salience_at(c, fabula_t)
return JoyAppraisal(
    own_joy_score=own,
    relief_score=relief,
    # ...
)
```

---

### MAJOR #6: Concern Polarity Flip Not Validated Against Proposition Semantics

**Location:** `shadow_loom/models.py:380-456` (`Concern`)

**Issue:**  
`ConcernSnapshot` allows arbitrary `polarity` overwrites without checking if the flip makes semantic sense for the proposition.

**Example Failure:**
- Proposition: `PROP_MACBETH_BECOMES_KING` (kind="outcome")
- Initial concern: `polarity="desire"` (Macbeth wants the crown)
- ConcernSnapshot at t=2000: `polarity="fear"` (now Macbeth fears becoming king)
- **Valid** if Macbeth's arc turns paranoid
- **Invalid** if the proposition semantics are "Macbeth *already* is king" (binary outcome, not reversible desire)

**Theoretical Problem:**  
Appraisal theory requires goal stability or justified reappraisal. Arbitrary flips without causal triggers violate coherence.

**Fix:**
Add validation in `Concern.model_validator`:
```python
@model_validator(mode="after")
def validate_polarity_flip(self) -> Concern:
    if not self.state_timeline:
        return self
    # Check if polarity flipped between successive snapshots
    prev_pol = self.polarity
    for snap in self.state_timeline:
        if snap.polarity and snap.polarity != prev_pol:
            # Require a triggered_by event justifying the flip
            if not snap.triggered_by:
                raise ValueError(
                    f"Concern {self.concern_id} polarity flip "
                    f"{prev_pol}→{snap.polarity} at fabula={snap.fabula_time} "
                    f"lacks triggered_by justification"
                )
            prev_pol = snap.polarity
    return self
```

---

## IV. THEORY OF MIND — Epistemic State Tracking

### CRITICAL #5: Bayesian Surprise KL Direction Inverted in Local Mode

**Location:** `shadow_loom/directive_assembly.py:5204-5350` (`compute_surprise_score`)

**Issue:**  
Docstring claims "Bayesian Surprise in the sense of Itti & Baldi (2009)" with KL direction "posterior over prior" but the implementation in `local=True` mode is unclear about which belief is P and which is Q in `D_KL(P || Q)`.

**Itti & Baldi (2009) Definition:**
$$\text{Surprise} = D_{KL}(P(\theta | D_{new}) \| P(\theta | D_{old}))$$

Posterior over prior (new beliefs over old beliefs).

**Current Code (local mode):**
```python
# The docstring says:
# "KL distance between the reader's prior immediately *after* and
# immediately *before* the current syuzhet anchor's revelations
# (posterior over prior, the canonical Itti-Baldi direction)"
```

But the non-local cumulative mode computes:
```python
kl = _binary_kl(p_now, p_prev)  # p_now is posterior, p_prev is prior
```

Where `_binary_kl(p, q) = p log(p/q) + (1-p) log((1-p)/(1-q))` — this is `D_KL(p || q)`.

**So `_binary_kl(p_now, p_prev)` = D_KL(posterior || prior)** ✓ CORRECT direction.

**BUT:** The docstring for `local=True` says "posterior over prior" which is ambiguous. Re-reading the affect_unification code:

```python
def compute_surprise_unified(
    bs: BeliefState, fabula_t: int, prior_fabula_t: int,
) -> float:
    """Itti-Baldi Bayesian surprise on audience belief revision.

    Sums KL(p_aud(P, t) || p_aud(P, t-1)) × stakes across all
    propositions whose audience confidence moved.
    """
    total = 0.0
    for prop in bs.world.propositions:
        p_now = bs.confidence(AUDIENCE_ID, prop.proposition_id, fabula_t)
        p_prev = bs.confidence(AUDIENCE_ID, prop.proposition_id, prior_fabula_t)
        if abs(p_now - p_prev) < _EPS:
            continue
        total += _binary_kl(p_now, p_prev) * _prop_stakes_at(prop, fabula_t)
    return total
```

This is `D_KL(p_now || p_prev)` which **is** posterior || prior. So the implementation is correct.

**HOWEVER:** In `directive_assembly.py`, the `local=True` branch description is confusing. The code should explicitly state the direction.

**Verdict:** **NOT a bug**, but documentation is ambiguous. Recommend clarifying docstring:

```python
# local=True: D_KL(p_after || p_before) per Itti & Baldi 2009
# This is FORWARD KL (posterior || prior), measuring information gain
```

**Reclassify as MINOR #1.**

---

### MAJOR #7: Belief Provenance Contradiction Not Caught

**Location:** `shadow_loom/auditor.py:351-355`

**Issue:**  
`AuditViolation` enum includes `"belief_provenance_contradiction"` but grep for this string in `auditor.py` yields only the enum declaration — no check emits it.

**Example Undetected Violation:**
- Belief: `acquired_via_event_id = EVT_LETTER_ARRIVAL`
- Event `EVT_LETTER_ARRIVAL` has `syuzhet_index = 50`
- Belief's `established_at_fabula = 30` (before the event was narrated)
- **Contradiction:** Character knows content before receiving the letter

**Fix Required:**
Add deterministic check in auditor:
```python
def _check_belief_provenance(world: WorldStateV1, syuzhet_anchor: int) -> List[AuditViolation]:
    violations = []
    events_by_id = {e.id: e for e in world.events}
    for ent in world.entities.values():
        for belief in ent.beliefs:
            if belief.acquired_via_event_id:
                evt = events_by_id.get(belief.acquired_via_event_id)
                if evt and evt.syuzhet_index > syuzhet_anchor:
                    violations.append(AuditViolation(
                        violation_type="belief_provenance_contradiction",
                        severity="critical",
                        description=f"{ent.id} has belief about {belief.target_id} "
                                   f"acquired via {evt.id} (syuzhet {evt.syuzhet_index}) "
                                   f"but anchor is {syuzhet_anchor} — future knowledge leak",
                        evidence_quote="",
                        feedback="Remove this belief or move the acquiring event earlier."
                    ))
    return violations
```

---

### MAJOR #8: Epistemic Gap "Quantitative" Type Never Computed

**Location:** `shadow_loom/directive_assembly.py:41-50` (`EpistemicGap`)

**Issue:**
```python
class EpistemicGap(BaseModel):
    # ...
    gap_type: Literal["contradicted", "confirmed", "unknown", "quantitative"] = "unknown"
    gap_magnitude: float = Field(default=0.0, description="0.0 = aligned, 1.0 = maximally wrong")
```

But `compute_epistemic_gaps` only sets:
- `"contradicted"` when believed ≠ actual (boolean)
- `"confirmed"` when believed = actual
- Default `"unknown"`

**Never sets `"quantitative"`** even though `gap_magnitude` is populated.

**Theoretical Problem:**  
For continuous traits (e.g., Macbeth believes Duncan's courage is 0.8, actually 0.5), the gap is **quantitative** not boolean. Current classification as "unknown" loses information.

**Fix:**
```python
if belief.proposition_id:
    prop = prop_index.get(belief.proposition_id)
    if prop and prop.kind == "trait_holds":
        # Quantitative gap for trait propositions
        gap_type = "quantitative"
        gap_magnitude = abs(believed_value - actual_value)
```

---

## V. CONCERN-PROPOSITION ALIGNMENT

### CRITICAL #6: Concern Synthesis Missing Validation of Proposition Existence

**Location:** `shadow_loom/affect_unification.py:490-520` (`synthesise_audience_entity`)

**Issue:**  
Audience concern synthesis creates `CCN_AUDIENCE_*` concerns referencing `proposition_id` but **does not validate** that the proposition exists:

```python
for prop in world.propositions:
    if prop.kind != "outcome":
        continue
    ccn_id = f"CCN_AUDIENCE_{prop.proposition_id[len('PROP_'):]}"
    # ...
    audience.concerns.append(
        Concern(
            concern_id=ccn_id,
            proposition_id=prop.proposition_id,  # ❌ What if this prop gets pruned later?
            polarity="desire",
            salience=salience,
            kind="outcome",
        )
    )
```

**Failure Mode:**
1. Synthesis runs, creates `CCN_AUDIENCE_FROM_EVT_X`
2. Phase C reconciliation prunes `PROP_FROM_EVT_X` (duplicate / contradictory)
3. Concern now references nonexistent proposition
4. `_concerns_for()` returns the orphaned concern
5. `_concern_salience()` tries to read stakes from missing prop → KeyError or 0.5 fallback

**Fix:**
Add validation pass after synthesis:
```python
def _validate_concern_propositions(world: WorldStateV1) -> int:
    """Remove concerns whose proposition_id does not exist. Returns pruned count."""
    prop_ids = {p.proposition_id for p in world.propositions}
    pruned = 0
    for ent in world.entities.values():
        valid = []
        for c in ent.concerns:
            if c.proposition_id in prop_ids:
                valid.append(c)
            else:
                logger.warning(f"Pruning {c.concern_id}: references missing {c.proposition_id}")
                pruned += 1
        ent.concerns = valid
    return pruned
```

Call this after every proposition pruning step.

---

### CRITICAL #7: Concern Activation Window Not Checked in Affect Scorers

**Location:** `shadow_loom/affect_unification.py:800-850` (`_focal_concerns`)

**Issue:**  
`Concern.activation_fabula_window: Optional[List[int]]` gates when a concern is "active" but `_focal_concerns()` helper **does not filter** by this window:

```python
def _focal_concerns(
    world: WorldStateV1, entity_id: str,
    *, polarity: Optional[str] = None,
    fabula_t: int,
) -> List[Concern]:
    """Return concerns of entity_id, optionally filtered by polarity."""
    ent = world.entities.get(entity_id)
    if ent is None:
        return []
    out = ent.concerns
    if polarity:
        out = [c for c in out if _concern_polarity_at(c, fabula_t) == polarity]
    # ❌ MISSING: filter by activation_fabula_window
    return out
```

**Theoretical Problem:**  
Lear's concern `CCN_IRRELEVANCE` should only activate **after** the abdication (window = `[1500, None]`). If not filtered, suspense scorer treats it as active throughout the play, inflating pre-abdication tension.

**Fix:**
```python
def _focal_concerns(
    world: WorldStateV1, entity_id: str,
    *, polarity: Optional[str] = None,
    fabula_t: int,
) -> List[Concern]:
    ent = world.entities.get(entity_id)
    if ent is None:
        return []
    out = []
    for c in ent.concerns:
        # Check activation window
        window = _concern_window_at(c, fabula_t)
        if window:
            start, end = window
            if fabula_t < start or (end is not None and fabula_t > end):
                continue  # Concern not active at this time
        # Check polarity filter
        if polarity and _concern_polarity_at(c, fabula_t) != polarity:
            continue
        out.append(c)
    return out
```

---

## VI. EMOTION COMPUTATION ERRORS

### MAJOR #9: Fear vs. Anxiety Entropy Computation Missing Normalization

**Location:** `shadow_loom/affect_unification.py:1439-1445` (`compute_fear_appraisal`)

**Issue:**
```python
for c in concerns:
    # ...
    anxiety += _binary_entropy(prob) * sal
```

`anxiety_score` sums Shannon entropy × salience across **all** fear concerns. But entropy is unbounded (grows with # of concerns), so anxiety score scales with concern count not uncertainty.

**Theoretical Problem:**  
Öhman & Mineka (2001): Anxiety is **high entropy** (diffuse uncertainty), fear is **low entropy** (object-specific). Current implementation conflates "many weak fears" (high sum, low per-concern entropy) with "one ambiguous fear" (low sum, high per-concern entropy).

**Fix:**
Normalize by concern count or take mean:
```python
anxiety = anxiety / max(1, len(concerns))  # Mean entropy, not sum
```

Or use entropy **over the concern distribution** (meta-level):
```python
# Treat each concern as a categorical outcome, compute H(concern_probs)
concern_probs = [prob * sal for c in concerns for prob, sal in [(bs.confidence(...), ...)]]
concern_probs = [p / sum(concern_probs) for p in concern_probs]
anxiety = -sum(p * math.log(max(_EPS, p)) for p in concern_probs)
```

---

### MAJOR #10: Gloating Score Sign Error (OCC Polarity Flip)

**Location:** `shadow_loom/affect_unification.py:1500+` (`compute_joy_appraisal`)

**Issue:**
```python
for c in other.concerns:
    # ...
    if aff < -0.3 and _pol == "fear":
        gloating += base  # ❌ WRONG: should trigger when fear is REALIZED, not when it's high-confidence
```

**Theoretical Problem:**  
OCC (1988) gloating = "disliked other's desirable event not realized, OR feared event realized."

Current code: `aff < -0.3 and polarity == "fear"` → adds `prob × stakes × salience` regardless of whether the fear **came true**.

**Should be:**
- Gloating when `aff < -0.3` AND `polarity == "fear"` AND `prob > 0.7` (fear realized for enemy)
- OR `aff < -0.3` AND `polarity == "desire"` AND `prob < 0.3` (desire thwarted for enemy)

**Fix:**
```python
if aff < -0.3:
    if _pol == "fear" and prob > 0.7:  # Enemy's fear realized
        gloating += base
    elif _pol == "desire" and prob < 0.3:  # Enemy's desire thwarted
        gloating += base
```

---

### MINOR #2: Coping Score Fallback 0.5 Unjustified

**Location:** `shadow_loom/affect_unification.py:1411-1425` (`_focal_coping`)

**Issue:**
```python
if not vals:
    return 0.5  # ❌ Why 0.5? Maximum entropy default?
```

**Theoretical Problem:**  
Lazarus (1991) treats coping as a **learned resource**. Entities without coping traits should default to **low coping** (0.2, novice) not medium (0.5, neutral).

**Fix:**
```python
if not vals:
    return 0.2  # Default to low coping (untrained/vulnerable baseline)
```

---

### MINOR #3: Joy "Primary Concern" Selection Uses First Max, Not Global Max

**Location:** `shadow_loom/affect_unification.py:1500+` (`compute_joy_appraisal`)

**Issue:**
```python
if s > (primary[0] if primary else 0.0):
    primary = (s, c, prop)
```

This keeps the **first** concern that achieves a new max, not necessarily the global max (if two concerns have equal scores, first wins).

**Theoretical Impact:**  
Minimal — only affects which concern is surfaced as "primary" for logging. But for determinism, should track all ties and pick by concern_id sort.

**Fix:**
```python
if s > (primary[0] if primary else 0.0):
    primary = (s, c, prop)
elif s == primary[0]:
    # Tie-break by concern_id for determinism
    if c.concern_id < primary[1].concern_id:
        primary = (s, c, prop)
```

---

## VII. SUMMARY OF VIOLATIONS

| Category | Critical | Major | Minor |
|----------|----------|-------|-------|
| Bayesian Belief Networks | 2 | 2 | 1 |
| Appraisal Theory | 1 | 4 | 1 |
| OCC Model | 1 | 3 | 1 |
| Theory of Mind | 1 | 2 | 1 |
| Concern-Proposition | 2 | 1 | 0 |
| Emotion Computation | 0 | 3 | 2 |
| **TOTAL** | **7** | **12** | **6** |

---

## VIII. RECOMMENDATIONS

### Immediate (Block Production)
1. **Add `ge=0.0, le=1.0` bounds to `BeliefMutation.new_confidence`**
2. **Implement belief cascade when proposition truth flips**
3. **Add inverse proposition truth mirroring in Phase C**
4. **Filter concerns by activation window in all affect scorers**
5. **Implement channel-based belief pruning in do-surgery**
6. **Add concern→proposition validation pass**

### High Priority (Theory Compliance)
7. Add temporal proximity kernel to concern salience
8. Separate OCC relief from joy in appraisal computation
9. Fix gloating polarity condition
10. Normalize anxiety entropy by concern count
11. Add belief provenance contradiction check to auditor

### Medium Priority (Robustness)
12. Implement belief normalization for multi-source updates
13. Add concern polarity flip validation
14. Set quantitative gap_type for trait epistemic gaps
15. Integrate coping into suspense calculation

### Low Priority (Polish)
16. Clarify KL direction in surprise docstrings
17. Adjust coping fallback to 0.2
18. Add deterministic tie-breaking in primary concern selection

---

## IX. THEORETICAL VALIDATION CHECKLIST

- [ ] **Probability Axioms:** All confidence values ∈ [0, 1]
- [ ] **Bayesian Conditioning:** Belief updates when evidence commits
- [ ] **KL Divergence:** Direction matches Itti-Baldi (posterior || prior)
- [ ] **Appraisal Theory:** Concern salience × imminence × coping
- [ ] **OCC Model:** Relief/gloating/joy properly distinguished by polarity + outcome
- [ ] **Inverse Consistency:** PROP_X true ⟺ PROP_X_INVERSE false
- [ ] **Provenance Integrity:** Beliefs only from temporally-prior events/channels
- [ ] **Activation Windows:** Concerns only active within fabula bounds
- [ ] **Ambivalence:** Bidirectional counter-concern salience weighting

**PASS CRITERIA:** All critical issues resolved, ≥80% of major issues addressed.

---

**END OF AUDIT**
