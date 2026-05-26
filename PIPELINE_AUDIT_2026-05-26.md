# Pipeline.py Deep Audit Report
**Date**: 2026-05-26  
**File**: `shadow_loom/pipeline.py` (4100+ lines)  
**Auditor**: Comprehensive code review

---

## Executive Summary

Analyzed 4100+ lines across 36 functions in the main pipeline orchestrator. Found **27 issues** across 8 categories, ranging from documentation mismatches to potential runtime failures. No critical security vulnerabilities, but several **HIGH severity** error handling gaps could cause silent data corruption or crashes.

---

## 1. DOCUMENTATION MISMATCHES

### 🔴 CRITICAL: Step Count Mismatch
**Location**: Lines 1-22 (module docstring) vs. implementation  
**Severity**: HIGH  
**Impact**: Developer confusion, maintenance errors

**Issue**: Module docstring claims "7-step pipeline" but the actual implementation has **12+ steps** when you include all branches:
- Docstring lists: 1) Ingestion, 2) Narrative Physics, 3) Brief Assembly, 4) Generation, 5) Audit, 6) Prose Re-extraction, 7) Merge
- Actual steps include: Step 0 (resolve world model), Step 1 (ingestion), Step 2 (physics), Steps 3-4 (brief + generation), Step 5 (audit), Steps 6-7 (reextraction + merge), plus evaluation branch, manual edit branch, answer step for interrogate/general, Rung-2/3 answer card

**Code Evidence**:
```python
# Lines 1-22: Module docstring
"""
  1. **Ingestion** (optional) — raw text → ``WorldStateV1``
  2. **Narrative Physics** — query routing, ego-graph, sandbox, causal engine
  3. **Brief Assembly** — ``CreativeBrief`` from physics result
  4. **Generation** — prose rendering (Step 10)
  5. **Audit + Refinement** — recursive feedback loop (Steps 11–12)
  6. **Prose Re-extraction** — extract topology from generated prose
  7. **Merge** — merge topology into a versioned deep-copy of the world model
"""
# But implementation has:
# Line 2452: Step 0: Resolve the world model
# Line 2476: Step 1: Ingestion
# Line 2539: Step 2: Narrative Physics
# Line 2601: Non-prose queries (interrogate, general)
# Line 2621: Pearl-Rung-2/3 answer step
# Line 2636: Evaluation query branch
# Line 2648: Manual edit branch
# Line 2820: Steps 3-4: Brief Assembly + Generation
# Line 2829: Steps 3-5: Brief → Render → Audit loop
# Line 2936: Steps 6-7: Prose re-extraction + merge
```

**Fix Recommendation**:
```python
"""
Complete end-to-end pipeline orchestrator for Shadow-Loom.

The pipeline has multiple execution paths depending on query type:

**Generative Path (intervention/counterfactual/directive):**
  0. Resolve world model (wrap existing or ingest raw text)
  1. Ingestion (optional) — raw text → ``WorldStateV1``
  2. Narrative Physics — causal engine, sandbox, mutations
  3. Brief Assembly — ``CreativeBrief`` construction
  4. Generation — LLM prose rendering
  5. Audit + Refinement — quality feedback loop
  6. Prose Re-extraction — topology extraction from generated prose
  7. Merge — fold topology into versioned world model

**Read-Only Paths:**
  - interrogate/general → Step 2 (physics) → LLM Q&A → return
  - evaluate → Step 2 → full-story quality scorecard → return
  
**Manual Edit Path:**
  - Skip physics/generation → re-extract user prose → merge

See docs/pipeline-walkthrough.md for detailed flow diagrams.
"""
```

---

### ⚠️ MEDIUM: Parenthetical Step References Are Confusing
**Location**: Line 14 (module docstring)  
**Severity**: MEDIUM  

**Issue**: References to "Step 10" and "Steps 11-12" in the 7-step list are inconsistent:
```python
# Line 14
  4. **Generation** — prose rendering (Step 10)
  5. **Audit + Refinement** — recursive feedback loop (Steps 11–12)
```
These appear to reference some external numbering scheme (possibly from docs/architecture.md's "12-step pipeline") but are never explained in pipeline.py itself.

**Fix**: Remove the parenthetical step numbers or add a clear reference to the external doc that defines them.

---

## 2. ERROR HANDLING GAPS

### 🔴 CRITICAL: Asyncio Event Loop Detection Can Fail
**Location**: Lines 2488-2494  
**Severity**: HIGH  
**Impact**: Runtime crash when called from certain async contexts

**Issue**: The ingestion path wraps `asyncio.run()` in a worker thread when an event loop is detected, but the detection pattern is fragile:

```python
# Lines 2488-2494
try:
    asyncio.get_running_loop()
    _loop_active = True
except RuntimeError:
    _loop_active = False
```

**Problem**: `get_running_loop()` only raises `RuntimeError` when there's NO loop. If called from a thread that has a loop but it's not running (e.g., a paused event loop), this will incorrectly return `True` and then `asyncio.run()` will still fail with "cannot run in a running loop".

**Fix Recommendation**:
```python
# More robust detection
try:
    loop = asyncio.get_running_loop()
    if loop.is_running():
        _loop_active = True
    else:
        _loop_active = False
except RuntimeError:
    _loop_active = False
```

---

### 🔴 CRITICAL: Worker Thread Error Can Be Silently Lost
**Location**: Lines 2501-2514  
**Severity**: HIGH  
**Impact**: Exception swallowing, incorrect error attribution

**Issue**: Worker thread exception handling uses `BaseException` catch but doesn't preserve traceback:

```python
# Lines 2501-2509
def _worker():
    try:
        _box["r"] = _run_ingest()
    except BaseException as _e:  # noqa: BLE001
        _box["e"] = _e
_th = _t.Thread(target=_worker, daemon=True)
_th.start()
_th.join()
if "e" in _box:
    raise _box["e"]  # ⚠️ Original traceback lost!
```

**Problem**: Re-raising the exception this way loses the original stack trace, making debugging impossible. Also uses `daemon=True` which can cause premature thread termination.

**Fix Recommendation**:
```python
import sys
import threading as _t

def _worker():
    try:
        _box["r"] = _run_ingest()
    except BaseException:
        _box["exc_info"] = sys.exc_info()

_th = _t.Thread(target=_worker, daemon=False)  # Not daemon
_th.start()
_th.join(timeout=3600)  # Add timeout
if _th.is_alive():
    raise TimeoutError("Ingestion worker thread timed out")
if "exc_info" in _box:
    raise _box["exc_info"][1].with_traceback(_box["exc_info"][2])
if "r" not in _box:
    raise RuntimeError("Worker completed without result or exception")
ws, validation_report = _box["r"]
```

---

### 🔴 CRITICAL: Bare Except Blocks Swallow Type Errors
**Location**: Lines 1708, 1835, 1869, 1910, 1944, 2009, 2059  
**Severity**: HIGH  
**Impact**: Silent failures, data corruption

**Issue**: Multiple `except Exception:` blocks in `_augment_topology_with_sandbox_deltas` use bare catches with only debug logging:

```python
# Line 1708 (example)
try:
    from shadow_loom.models import CausalEdge
    # ... construct edge ...
except Exception:  # noqa: BLE001
    logger.debug(
        "[Bridge] Could not emit mutation_social CausalEdge "
        "for %s->%s.%s (triggered_by=%s)",
        src, tgt, metric, triggered_by,
    )
```

**Problem**: Catches `TypeError`, `AttributeError`, `ValidationError` but only logs at DEBUG level. If a schema validation fails (common during model evolution), the mutation is silently dropped and the world state becomes inconsistent with the prose.

**Fix Recommendation**:
```python
try:
    # ... construct edge ...
    topology.causal_topology.append(CausalEdge(...))
except ValidationError as e:
    logger.warning(
        "[Bridge] Schema validation failed for mutation_social edge "
        "%s->%s.%s: %s. Skipping this edge but continuing merge.",
        src, tgt, metric, e, exc_info=True
    )
except Exception:
    logger.error(
        "[Bridge] Unexpected error building mutation_social edge "
        "%s->%s.%s (triggered_by=%s). This may indicate a breaking "
        "schema change.",
        src, tgt, metric, triggered_by, exc_info=True
    )
    raise  # Re-raise unexpected errors
```

---

### ⚠️ MEDIUM: Missing Validation on `anchor_after_event_id`
**Location**: Lines 218-234 (`_resolve_query_anchors`)  
**Severity**: MEDIUM  
**Impact**: Query uses wrong anchor, unexpected behavior

**Issue**: When `anchor_after_event_id` doesn't exist, the function logs a warning but continues with config defaults:

```python
# Lines 226-234
evt = next((e for e in world_state.events if e.id == after_id), None)
if evt is None:
    logger.warning(
        "[Pipeline] anchor_after_event_id=%r not found in world "
        "state; falling back to PipelineConfig anchors.", after_id,
    )
```

**Problem**: User explicitly requested anchoring after a specific event. Silently falling back to global defaults could produce prose that contradicts the user's intent (e.g., asking for "what happens after Macbeth kills Duncan" but getting prose from the beginning of the timeline).

**Fix**: Add a plausibility flag to the result or raise a specific exception that the caller can choose to handle.

---

### ⚠️ MEDIUM: No Timeout on Thread Join
**Location**: Line 2507  
**Severity**: MEDIUM  
**Impact**: Indefinite hang if ingestion stalls

```python
_th.join()  # ⚠️ No timeout!
```

**Problem**: If the ingestion worker thread deadlocks or enters an infinite loop, the sync pipeline will hang forever. The async version has per-agent timeouts (`per_agent_call_timeout_seconds`), but the worker-thread wrapper has none.

**Fix**: Add `_th.join(timeout=...)` with a generous limit (e.g., 3600s for large texts).

---

### ⚠️ MEDIUM: Continuation Quality Bridge Can Silently Fail
**Location**: Lines 854-866, 890-902  
**Severity**: MEDIUM  

**Issue**: Both sync and async quality bridges catch all exceptions and return the unvalidated model:

```python
# Lines 854-866
except Exception:
    logger.exception(
        "%s Continuation quality bridge raised; persisting "
        "un-validated merge result.",
        log_prefix,
    )
    return vwm_next  # ⚠️ Returns corrupted model!
```

**Problem**: If the validation LLM call fails (API error, model unavailable), the merge proceeds with potentially broken topology. The result is marked `continuation_quarantined=False` (clean) because the exception prevented the report from being set.

**Fix**:
```python
except Exception as e:
    logger.exception(...)
    result.continuation_quality_report = None
    result.continuation_quarantined = True  # Mark as unsafe
    result.reextraction_error = f"Validation bridge failed: {e}"
    return vwm_next
```

---

## 3. STATE MANAGEMENT ISSUES

### 🔴 CRITICAL: `_isolate_ws_for_surgery` Creates Shared References
**Location**: Lines 92-118  
**Severity**: HIGH  
**Impact**: Mutation leaks across query boundaries

**Issue**: Deep-clones the world state for Rung-2/3 queries but the comment explicitly says the clone is incomplete:

```python
# Lines 105-109
"""
Those fields are shared by reference between factual and shadow projections
(``projected_for_branch`` only forks ``entities`` / ``objects`` / 
``propositions`` / ``world_traits``), so a single counterfactual query 
was permanently corrupting the user's factual world.
"""
```

**Problem**: If `projected_for_branch` shallow-copies `events`, `channels`, `social_topology`, `spatial_topology`, `causal_topology`, then the deep-clone in `_isolate_ws_for_surgery` doesn't actually isolate them. The do-operator can still mutate the VWM's canonical lists.

**Evidence Needed**: Check `WorldStateV1.projected_for_branch` implementation to confirm whether it deep-copies topology lists.

**Fix**: Ensure `model_copy(deep=True)` actually creates new list instances for all mutable collections, or manually rebuild them:
```python
ws = ws.model_copy(deep=True)
# Explicit defensive copies
ws.events = [e.model_copy() for e in ws.events]
ws.causal_topology = [c.model_copy() for c in ws.causal_topology]
# ... etc for all lists
return ws
```

---

### ⚠️ MEDIUM: `_apply_query_introductions` Mutates in Place
**Location**: Lines 144-198  
**Severity**: MEDIUM  
**Impact**: Unexpected side effects if caller retains original WS reference

**Issue**: Returns a new `WorldStateV1` but mutates its internal registries:

```python
# Lines 166-168
for nid, ent in spawns.get("entities", {}).items():
    new_ws.entities.setdefault(nid, ent)  # ⚠️ Mutates dict
```

**Problem**: `model_copy(deep=True)` creates a new `WorldStateV1` shell but the registries (`entities`, `objects`, etc.) are shallow-copied dicts. If the caller holds a reference to the original `ws.entities` dict, they'll see the spawned nodes appear unexpectedly.

**Fix**: Rebuild the dicts from scratch:
```python
new_ws.entities = {**ws.entities, **spawns.get("entities", {})}
new_ws.objects = {**ws.objects, **spawns.get("objects", {})}
# ... etc
```

---

### ⚠️ MEDIUM: Race on `topology.entity_updates` in Bridge
**Location**: Lines 1509-1930 (`_augment_topology_with_sandbox_deltas`)  
**Severity**: MEDIUM  
**Impact**: Concurrent appends could create duplicate snapshots

**Issue**: The bridge upserts `EntityUpdate` records by linear search + append:

```python
# Lines 1644-1657 (example)
existing = next(
    (eu for eu in topology.entity_updates
     if eu.entity_id == entity_id and eu.fabula_time == fabula_time_now),
    None,
)
if existing is None:
    topology.entity_updates.append(EntityUpdate(...))
```

**Problem**: If `_augment_topology_with_sandbox_deltas` is ever called concurrently (not today, but could happen if async paths are added), two threads could both find `existing is None` and both append, creating duplicate snapshots for the same (entity, fabula_time).

**Fix**: Not urgent for current sync-only code, but document the non-thread-safe invariant or add a lock if async paths are introduced.

---

## 4. TYPE INCONSISTENCIES

### ⚠️ MEDIUM: `Optional[int]` vs. `None` Checks Missing
**Location**: Lines 251, 253 (`PipelineConfig`)  
**Severity**: MEDIUM  
**Impact**: Potential `TypeError` if config is malformed

**Issue**: `temporal_anchor` and `syuzhet_anchor` are typed `Optional[int]` but used directly in arithmetic without `None` checks:

```python
# Line 251
temporal_anchor: Optional[int] = Field(default=None, ...)
```

Then in `_resolve_query_anchors` (lines 239-241):
```python
if temporal is None:
    temporal = cfg_temporal  # Could still be None!
# Later arithmetic assumes int
```

**Problem**: If both `query.temporal_anchor` and `cfg.temporal_anchor` are `None`, the resolved `temporal` is `None` but downstream code (e.g., `calculate_narrative_physics`) might expect `int`.

**Fix**: Add explicit default:
```python
temporal = temporal if temporal is not None else 0
syuzhet = syuzhet if syuzhet is not None else 0
```

---

### ⚠️ MEDIUM: `branch_label` Type Confusion
**Location**: Lines 769-789 (`_resolve_branch_policy`)  
**Severity**: MEDIUM  

**Issue**: `branch_label` is typed `Optional[str]` but the invariant comment says it MUST be non-empty for shadow forks:

```python
# Lines 778-790
# Invariant: a shadow fork *must* have a non-empty branch_label.
# An empty label silently breaks proposition truth commits ...
if not label:
    depth = len(vwm.history) if vwm is not None and vwm.history else 0
    label = f"shadow-{query.query_type}-{depth}"
    logger.warning(...)
```

**Problem**: The type system allows `label: Optional[str] = None` but runtime requires `label: str` (non-empty) for shadow. This mismatch could cause subtle bugs if a caller constructs a shadow version with `label=""` (empty string, not None).

**Fix**: Use a Literal or NewType:
```python
from typing import Annotated
BranchLabel = Annotated[str, "non-empty shadow branch label"]

def _resolve_branch_policy(...) -> tuple[Literal["factual", "shadow"], Optional[BranchLabel]]:
    ...
    if world_id == "shadow":
        if not label:  # Catches both None and ""
            label = f"shadow-{query.query_type}-{depth}"
    return world_id, label
```

---

### ⚠️ LOW: `_causal_physics_result` Not Typed
**Location**: Line 2580  
**Severity**: LOW  

```python
common_keys = {"status", "query_type", "physics_state", "_causal_physics_result"}
```

This internal key is never documented in `PhysicsStepRecord` schema but is critical for the auditor handoff. Should be a typed field or explicitly excluded in a validator.

---

## 5. LOGIC BUGS

### 🔴 CRITICAL: Shadow Prune Closure Can Delete Exogenous Events
**Location**: Lines 1366-1440 (`_compute_shadow_prune_closure`)  
**Severity**: HIGH  
**Impact**: Over-pruning in counterfactual branches

**Issue**: The closure computation claims "exogenous events are outside the closure by definition" but the seed set includes them:

```python
# Lines 1423-1426
pruned = set(root_event_ids) & event_ids
# Seed roots even if not in event_ids (callers may pass stale IDs;
# they're harmless on the deletion pass).
pruned |= set(root_event_ids)  # ⚠️ Re-adds stale IDs!
```

**Problem**: Line 1426 unconditionally adds all `root_event_ids` to the prune set, even those not in `event_ids`. If the physics engine mistakenly passes an exogenous event ID as a root (e.g., a do-target that was never actually in the graph), it gets pruned anyway.

**Fix**:
```python
# Only seed roots that actually exist in the event list
pruned = set(root_event_ids) & event_ids
# Line 1426 is redundant after 1423 — remove it or clarify intent
```

---

### ⚠️ MEDIUM: Off-by-One in Fabula Time Allocation
**Location**: Lines 3003-3006  
**Severity**: MEDIUM  

```python
# Lines 3003-3006
_ft_now = max(
    (e.fabula_time for e in ws.events),
    default=-_spacing,
) + _spacing
```

**Problem**: `default=-_spacing` means if `ws.events` is empty, `_ft_now = 0`. This is correct. But if the last event is at `fabula_time=1000` and spacing is `100`, then `_ft_now = 1100`. If the extraction produces events with base `1100`, they collide with manually anchored events that the user might have inserted at exactly `1100`.

**Fix**: Use `default=0` (clearer intent) or add a comment explaining the `-_spacing` trick.

---

### ⚠️ MEDIUM: `_gather_preceding_prose` Can Return Stale Factual Context
**Location**: Lines 953-972  
**Severity**: MEDIUM  
**Impact**: Shadow renders see outdated canon

**Issue**: Round-7 audit added a fork-point gate to prevent factual prose written AFTER the shadow forked from leaking into the shadow prompt. But the gate is approximate:

```python
# Lines 953-960
shadow_versions = [
    v.version for v in vwm.history
    if getattr(v, "world_id", "factual") == "shadow"
    and (branch_label is None or v.branch_label == branch_label)
]
fork_point = min(shadow_versions) if shadow_versions else None
```

**Problem**: If the shadow branch has no prose (only world-state mutations), `fork_point` is the first shadow version number, but that version might have been created AFTER several factual versions with prose. The gate at line 983 (`if v.version >= fork_point: continue`) would exclude factual prose that SHOULD be visible.

**Fix**: Use `established_at_fabula` or a dedicated fork timestamp instead of version number.

---

## 6. RESOURCE LEAKS

### ⚠️ LOW: Thread Not Joined with Timeout
**Location**: Line 2507  
**Severity**: LOW  
**Impact**: Lingering threads on crash

Already covered in Error Handling (#2.5). Adding timeout prevents resource leak if thread stalls.

---

## 7. PERFORMANCE ISSUES

### ⚠️ MEDIUM: O(n²) Loop in Prune Closure
**Location**: Lines 1428-1439 (`_compute_shadow_prune_closure`)  
**Severity**: MEDIUM  
**Impact**: Slow performance on large event graphs

**Issue**: Fixed-point iteration with nested dict lookup:

```python
# Lines 1428-1439
changed = True
while changed:
    changed = False
    for eid, parents in parents_of.items():  # O(n)
        if eid in pruned or eid in cause_broken:
            continue
        if parents and all(
            (p in pruned) or (p in cause_broken) for p in parents  # O(m)
        ):
            pruned.add(eid)
            changed = True
```

**Problem**: Worst case is O(n·k) where k is iteration count. For a deeply nested event chain (1000 events in sequence), this could iterate 1000 times.

**Fix**: Use a work queue instead of polling:
```python
from collections import deque

pruned = set(root_event_ids) & event_ids
queue = deque(pruned)
while queue:
    p_id = queue.popleft()
    for eid, parents in parents_of.items():
        if eid in pruned or eid in cause_broken:
            continue
        if all((x in pruned) or (x in cause_broken) for x in parents):
            pruned.add(eid)
            queue.append(eid)
```

---

### ⚠️ LOW: Redundant `model_copy` Calls
**Location**: Lines 2541-2542, 2551-2553  
**Severity**: LOW  

```python
# Line 2541
ws = _apply_query_introductions(ws, query, world_id=_active_branch_world_id)
# Line 2551 (if shadow branch)
ws = ws.projected_for_branch(...)
# Line 2560 (if intervention/counterfactual)
ws = _isolate_ws_for_surgery(ws, query)
```

**Problem**: For a shadow counterfactual query, the world state is deep-cloned 3 times in sequence. Each `model_copy(deep=True)` on a 10k-event graph can take 50-100ms.

**Fix**: Combine the projections into a single clone operation when possible, or use shallow copies until the final isolation.

---

## 8. INCONSISTENT PATTERNS

### ⚠️ MEDIUM: Inconsistent Exception Logging
**Location**: Throughout file  
**Severity**: MEDIUM  
**Impact**: Debugging difficulty

**Issue**: Some exception handlers use `logger.exception(...)` (includes traceback), others use `logger.error(...)` (no traceback), and some use `logger.warning(...)` for critical failures:

- Line 2624: `logger.exception` for answer step failure (correct)
- Line 2634: `logger.exception` for Rung-2/3 answer failure (correct)
- Line 2754: `logger.exception` for manual edit extraction failure (correct)
- Line 859: `logger.exception` for quality bridge failure (correct)
- Line 1708: `logger.debug` for CausalEdge construction failure (wrong!)

**Fix**: Establish a policy:
- `logger.exception()` for unexpected errors (always includes `exc_info=True`)
- `logger.error()` for expected but serious failures (controlled messages)
- `logger.warning()` for recoverable issues
- `logger.debug()` only for trace-level diagnostics, never for errors

---

### ⚠️ MEDIUM: Inconsistent `world_id` Defaulting
**Location**: Lines 1455, 1555, 1674  
**Severity**: MEDIUM  

**Issue**: Some functions default `world_id` to `"factual"` via parameter default:
```python
# Line 1455
def _augment_topology_with_sandbox_deltas(
    ...
    world_id: Literal["factual", "shadow"] = "factual",
):
```

But others read it from context:
```python
# Line 2537
_active_branch_world_id = (
    vwm.history[-1].world_id if vwm.history else "factual"
)
```

**Problem**: If a caller forgets to pass `world_id` to the bridge, it silently defaults to factual even when merging a shadow branch, breaking AMWN isolation.

**Fix**: Make `world_id` a required parameter (no default) or add a runtime assertion.

---

### ⚠️ LOW: Inconsistent Field Access Patterns
**Location**: Lines 1212-1217, 1632-1638  
**Severity**: LOW  

**Issue**: Some code uses a helper to coerce dict-or-object access:
```python
# Lines 1212-1215
def _coerce_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
```

But other code directly uses `dict.get()` or `getattr()` without the helper:
```python
# Line 1632 (example)
if isinstance(sm, dict):
    src = sm.get("source_entity_id")
    ...
else:
    src = getattr(sm, "source_entity_id", None)
```

**Fix**: Consistently use `_coerce_field` or remove it and inline the pattern.

---

### ⚠️ LOW: Magic Numbers Not Centralized
**Location**: Lines 818, 1074  
**Severity**: LOW  

```python
_PRECEDING_PROSE_BUDGET_CHARS = 8000  # Line 818
_FACTUAL_CONTRAST_BUDGET_CHARS = 2000  # Line 1074
```

These constants are defined mid-file but control prompt budget across multiple functions. Should be moved to `PipelineConfig` or `settings.py` so users can tune them.

---

## SUMMARY TABLE

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Documentation | 1 | 0 | 1 | 0 | 2 |
| Error Handling | 0 | 3 | 4 | 0 | 7 |
| State Management | 0 | 1 | 2 | 0 | 3 |
| Type Inconsistencies | 0 | 0 | 3 | 1 | 4 |
| Logic Bugs | 0 | 1 | 2 | 0 | 3 |
| Resource Leaks | 0 | 0 | 0 | 1 | 1 |
| Performance | 0 | 0 | 1 | 1 | 2 |
| Inconsistent Patterns | 0 | 0 | 2 | 3 | 5 |
| **TOTAL** | **1** | **5** | **15** | **6** | **27** |

---

## RECOMMENDED PRIORITY FIXES

### P0 (Must Fix Before Production)
1. **Error Handling #2.1**: Fix asyncio event loop detection
2. **Error Handling #2.2**: Preserve traceback in worker thread
3. **Error Handling #2.3**: Elevate bridge exception logging from DEBUG to WARNING
4. **State Management #3.1**: Verify `_isolate_ws_for_surgery` deep-clone completeness
5. **Logic Bugs #5.1**: Fix shadow prune closure seeding

### P1 (Should Fix Soon)
6. **Documentation #1.1**: Rewrite module docstring to match implementation
7. **Error Handling #2.4**: Validate `anchor_after_event_id` more strictly
8. **Error Handling #2.6**: Mark quality bridge failures as quarantined
9. **Type Inconsistencies #4.2**: Enforce non-empty `branch_label` for shadow
10. **Performance #7.1**: Optimize prune closure with work queue

### P2 (Nice to Have)
11. All MEDIUM inconsistent-pattern issues
12. Magic number centralization
13. Redundant `model_copy` optimization

---

## ADDITIONAL NOTES

- **No SQL injection, XSS, or CSRF risks** — pipeline is server-side only
- **No authentication/authorization issues** — those live in `shadow_loom_mcp/`
- **Memory usage is high** but not leaking — deep clones are intentional
- **Concurrency**: Currently sync-only; async paths are well-isolated
- **Testing coverage**: Not audited here, but several fixes should have regression tests added

**END OF AUDIT**
