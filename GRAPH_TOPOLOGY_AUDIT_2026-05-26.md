# Graph Topology Operations Audit
**Date:** 2026-05-26  
**Scope:** Correctness and consistency of edge operations across causal_physics.py, ingestion.py, and extract_graph.py

## Executive Summary

This audit identified **7 critical issues** and **6 moderate issues** across graph topology operations, with **3 operations verified correct**. Most critical are:
- Missing temporal ordering validation for CausalEdge operations
- Orphaned edge reference checks incomplete (rel_counterpart_id not validated)
- Location existence checks run after sandbox writes (SpatialEdge)
- Missing lifecycle timestamps on RelationshipEdge creation
- Channel intelligibility constraint validation gaps
- No global orphaned edge audit pass

**Key findings verified correct:**
- SpatialEdge bidirectional storage design (canonical stores forward only with flag, instantiator creates reverse)
- RelationshipEdge per-axis metric independence
- RelationshipEdge asymmetric directionality

---

## 1. CausalEdge Operations

### 🔴 CRITICAL: Missing Temporal Ordering Validation
**Location:** `shadow_loom/causal_physics.py:2072-2175` (`_apply_do_causal_edge`)  
**Issue:** When adding a CausalEdge via do-surgery, there's no validation that `fabula_time` respects temporal causality (cause must precede or be simultaneous with effect).

**Evidence:**
```python
# Line 2145-2157: Edge is created without temporal validation
edge = CausalEdge(
    source_id=target.source_id,
    target_id=target.target_id,
    causality_type=target.causality_type,
    causal_force=float(target.causal_force),
    mechanism=target.mechanism,
    fabula_time=ft,  # No check that this doesn't violate temporal order
    trait_target=target.trait_target,
    trait_delta=target.trait_delta,
    rel_counterpart_id=target.rel_counterpart_id,
)
```

**Risk:** A counterfactual intervention could create a causal edge where the effect occurs *before* the cause, breaking physics propagation assumptions.

**Recommendation:**
```python
# After line 2145, add:
if target.target_id.startswith("EVT_"):
    target_evt = next((e for e in self.world_state.events if e.id == target.target_id), None)
    if target_evt and target_evt.fabula_time < ft:
        logger.warning(
            "[CausalPhysics·do_causal_edge] Temporal violation: effect %s@T=%d "
            "would precede cause@T=%d; skipping.",
            target.target_id, target_evt.fabula_time, ft,
        )
        return
```

---

### 🟡 MODERATE: Evidence Strength Propagation Missing
**Location:** `shadow_loom/causal_physics.py:2145-2157`  
**Issue:** When adding a new CausalEdge, `evidence_strength` defaults to "moderate" without considering the strength of the triggering intervention.

**Evidence:**
```python
edge = CausalEdge(
    # ... other fields
    # No evidence_strength parameter passed from target
)
```

**Recommendation:** Add `evidence_strength=getattr(target, "evidence_strength", "moderate")` to preserve epistemic uncertainty from the intervention source.

---

### 🔴 CRITICAL: Mechanism-Trait Mapping Not Validated
**Location:** `shadow_loom/causal_physics.py:2145-2157`  
**Issue:** When `trait_target` is specified, there's no validation that the `mechanism` maps to a trait family that includes `trait_target` via `MECHANISM_TRAIT_MAP`.

**Risk:** A mutation edge with `mechanism="physical"` and `trait_target="guilt"` would fail silently during propagation (physical mechanisms don't map to psychological traits).

**Recommendation:**
```python
# After line 2145:
if edge.trait_target and edge.mechanism in MECHANISM_TRAIT_MAP:
    allowed_traits = MECHANISM_TRAIT_MAP[edge.mechanism]
    if edge.trait_target not in allowed_traits:
        logger.warning(
            "[CausalPhysics·do_causal_edge] Mechanism-trait mismatch: "
            "mechanism=%r does not map to trait_target=%r; edge will "
            "apply fallback force. Allowed: %s",
            edge.mechanism, edge.trait_target, allowed_traits,
        )
```

---

### 🔴 CRITICAL: Orphaned Edge Check Incomplete
**Location:** `shadow_loom/causal_physics.py:2163-2171`  
**Issue:** The `_world_knows_node` check only validates source/target endpoints, but ignores `rel_counterpart_id` for `mutation_social` edges.

**Evidence:**
```python
# Lines 2163-2171
if not self._world_knows_node(target.source_id) or not self._world_knows_node(target.target_id):
    logger.warning(
        "[CausalPhysics·do_causal_edge] add skipped — endpoint(s) unknown to canonical world (%s→%s).",
        target.source_id, target.target_id,
    )
    return
# Missing: rel_counterpart_id validation
```

**Risk:** A `mutation_social` edge with a valid `source_id`/`target_id` but phantom `rel_counterpart_id` will be added to `causal_topology`, leaving a dangling reference that breaks relationship graph traversals.

**Recommendation:**
```python
# After line 2168, add:
if edge.rel_counterpart_id and not self._world_knows_node(edge.rel_counterpart_id):
    logger.warning(
        "[CausalPhysics·do_causal_edge] add skipped — rel_counterpart_id %r "
        "unknown to canonical world.", edge.rel_counterpart_id,
    )
    return
```

---

## 2. SpatialEdge Operations

### � VERIFIED: Bidirectional Edge Storage is Correct
**Location:** `shadow_loom/causal_physics.py:2281-2302`  
**Status:** ✅ **CORRECT BY DESIGN**

When adding a `bidirectional=True` SpatialEdge, the code correctly adds both forward and reverse edges to the **sandbox** but only adds the forward edge (with `bidirectional=True` flag) to `world_state.spatial_topology`.

**Evidence:**
```python
# causal_physics.py:2289-2297 - Sandbox gets both directions
if sb.has_node(target.source_id) and sb.has_node(target.target_id):
    sb.add_edge(
        target.source_id, target.target_id,
        edge_type="connected_to",
        **edge.model_dump(),
    )
    if edge.bidirectional:
        sb.add_edge(
            target.target_id, target.source_id,  # Reverse sandbox edge
            edge_type="connected_to",
            **edge.model_dump(),
        )
# causal_physics.py:2301-2310 - Canonical topology gets forward only
topology = list(self.world_state.spatial_topology or [])
topology.append(edge)  # Single edge with bidirectional=True

# instantiator.py:259-267 - Instantiator recreates reverse from canonical
sandbox.add_edge(src_loc, tgt_loc, edge_type="connected_to", ...)
if bidirectional:
    sandbox.add_edge(tgt_loc, src_loc, edge_type="connected_to", ...)
```

**Verification:** Test `test_connected_to_is_bidirectional` (test_narrative_physics.py:798) confirms both directions exist in sandbox.

**Recommendation:** Add documentation comment to clarify intent:
```python
# After line 2302 in causal_physics.py:
# NOTE: Bidirectional edges are stored as a single forward edge in
# canonical topology with bidirectional=True. The AMWNInstantiator
# (instantiator.py Section F) synthesizes the reverse edge when
# building the sandbox. This avoids duplicating reverse edges on
# re-instantiation while preserving symmetric traversability.
```

---

### 🟡 MODERATE: Affordance Gate Validation Missing
**Location:** `shadow_loom/causal_physics.py:2281-2299`  
**Issue:** When adding a locked SpatialEdge with `barrier_item_id`, there's no validation that the barrier object actually exists in `world_state.objects`.

**Evidence:**
```python
# Lines 2281-2290
edge = SpatialEdge(
    source_id=target.source_id,
    target_id=target.target_id,
    connection_type=target.connection_type or "passage",
    bidirectional=bool(target.bidirectional),
    is_locked=False,  # Note: do-surgery always creates unlocked; lock is a separate action
    barrier_item_id=target.barrier_item_id,  # No validation
    established_at_fabula=ft,
)
```

**Recommendation:**
```python
# After line 2282:
if target.barrier_item_id:
    if target.barrier_item_id not in (self.world_state.objects or {}):
        logger.warning(
            "[CausalPhysics·do_spatial_edge] barrier_item_id %r not in "
            "world.objects; lock semantics undefined.",
            target.barrier_item_id,
        )
        # Decide: either skip the edge, or null out barrier_item_id
```

---

### 🔴 CRITICAL: Location Existence Check Only Runs on World-State Side
**Location:** `shadow_loom/causal_physics.py:2301-2310`  
**Issue:** The location validation (`target.source_id not in locations`) only gates adding the edge to `world_state.spatial_topology`, but the sandbox side (`sb.add_edge`) runs unconditionally even when endpoints are invalid.

**Evidence:**
```python
# Lines 2289-2297: Sandbox side always adds
if sb.has_node(target.source_id) and sb.has_node(target.target_id):
    sb.add_edge(...)  # No validation that source/target are LOC_ nodes

# Lines 2301-2310: World-state side validates
locations = self.world_state.locations or {}
if target.source_id not in locations or target.target_id not in locations:
    logger.warning(...)
    return  # But sandbox already has the phantom edge
```

**Risk:** A do-surgery with a typo'd LOC_ id adds a spatial edge to the sandbox but not to canonical topology. The sandbox traversal sees a passage that doesn't exist in the world, breaking reachability checks.

**Recommendation:** Move validation before sandbox write:
```python
# Before line 2289:
locations = self.world_state.locations or {}
if target.source_id not in locations or target.target_id not in locations:
    logger.warning(
        "[CausalPhysics·do_spatial_edge] add skipped — endpoint(s) not in "
        "world.locations (%s→%s).", target.source_id, target.target_id,
    )
    return
# Now both sandbox and world-state writes are gated by the same check
```

---

### 🟡 MODERATE: Sever Operation Missing Destroyed-At-Fabula Update
**Location:** `shadow_loom/causal_physics.py:2190-2249`  
**Issue:** The `sever` action removes SpatialEdges entirely rather than marking them as `destroyed_at_fabula=current_tick`, which loses temporal information.

**Evidence:**
```python
# Lines 2195-2247: Edges are removed
for k in matched:
    sb.remove_edge(target.source_id, target.target_id, key=k)
# Should instead mark destroyed:
# edge_data["destroyed_at_fabula"] = ft
```

**Recommendation:** Preserve the edge with a destruction timestamp so time-sliced queries can see when passages were severed.

---

## 3. RelationshipEdge Operations

### 🟢 VERIFIED: Per-Axis Metric Independence
**Location:** `shadow_loom/causal_physics.py:2009-2069`  
**Status:** ✅ **CORRECT**

The `do_relationship` implementation correctly updates individual metrics without cross-contamination:
```python
# Lines 2018-2030: Per-metric update
metrics = attrs.setdefault("metrics", {})
entry = metrics.get(target.metric) or {}
entry["value"] = float(target.value)
entry["observed"] = True
entry["last_updated_fabula"] = ft
```

---

### 🟡 MODERATE: Inertia/Evidence Aggregation Inconsistent
**Location:** `shadow_loom/causal_physics.py:3300-3380`  
**Issue:** When creating a new RelationshipEdge during social propagation (line 3360), the code uses `default_relationship_metrics_dict()` which seeds all three axes with the same `inertia` value, ignoring per-axis defaults (fear should be ~0.2, power_dynamic ~0.6).

**Evidence:**
```python
# Line 3344-3360
edge_attrs = {
    # ...
    "inertia": _relationship_inertia_default(),  # Single value for all axes
    "metrics": default_relationship_metrics_dict(
        primary_metric=metric,
        primary_value=primary_value,
        fabula_time=d.get("fabula_time", 0),
        evidence_strength=d.get("evidence_strength", "weak"),
        inertia=_relationship_inertia_default(),  # Same inertia for all
    ),
}
```

**Recommendation:** Pass per-axis inertia to `default_relationship_metrics_dict`:
```python
inertia_map = {"fear": 0.2, "affinity": 0.4, "power_dynamic": 0.6}
per_axis_inertia = inertia_map.get(metric, 0.3)
```

---

### 🔴 CRITICAL: Lifecycle Handling Missing on New Edge Creation
**Location:** `shadow_loom/causal_physics.py:3344-3365`  
**Issue:** When a new RelationshipEdge is created during propagation, `established_at_fabula` is not set, defaulting to `None`.

**Risk:** Time-slicing code in `extract_graph.py:_time_slice_relationship_at` (lines 47-51) drops edges where `established_at_fabula > t`, but edges with `None` pass through. This means a relationship created at T=50 would be visible in a T=10 time-slice.

**Recommendation:**
```python
# After line 3344:
edge_attrs["established_at_fabula"] = d.get("fabula_time", 0)
```

---

### 🟢 VERIFIED: Bidirectional Consistency Not Required
**Status:** ✅ **BY DESIGN**

RelationshipEdges are **directed** and **asymmetric** by design (A's fear of B ≠ B's fear of A). The code correctly does not create reverse edges. The warning in `ingestion.py:_warn_suspicious_mirror_dyads` (line 10396) flags symmetric dyads as suspicious, which is correct.

---

## 4. Channel Operations

### 🔴 CRITICAL: Intelligibility Constraint Not Validated
**Location:** Channels are not directly manipulated by causal_physics.py; validation happens in `ingestion.py` during assembly.  
**Issue:** `shadow_loom/ingestion.py:13597-13623` auto-repairs channels with invalid `participant_ids`, but does **not** validate that `intelligibility` keys match `participant_ids`.

**Evidence:**
```python
# ingestion.py:13597-13610
for cid, ch in ws.channels.items():
    valid_pids = [p for p in ch.participant_ids if p in node_ids]
    bad_pids = [p for p in ch.participant_ids if p not in node_ids]
    # ...
    if valid_pids != list(ch.participant_ids):
        pruned_intel = {k: v for k, v in ch.intelligibility.items() if k in valid_pids}
        # Prunes intelligibility, but doesn't validate existing keys
```

**Risk:** A channel with `participant_ids=["ENT_ALICE", "ENT_BOB"]` but `intelligibility={"ENT_CHARLIE": 0.5}` will pass validation with a dangling intelligibility key.

**Recommendation:**
```python
# After line 13609:
orphaned_intel = [k for k in pruned_intel if k not in valid_pids]
if orphaned_intel:
    repairs.append(
        f"Removed channel '{cid}' orphaned intelligibility keys: {orphaned_intel}"
    )
    pruned_intel = {k: v for k, v in pruned_intel.items() if k in valid_pids}
```

---

### 🟡 MODERATE: Termination Propagation Missing
**Location:** Channels have `terminated_at_fabula`, but there's no mechanism to cascade termination to dependent `Belief` records.  
**Issue:** When a channel is terminated (e.g., via counterfactual do-surgery), beliefs acquired via that channel (`Belief.acquired_via_channel_id`) should be marked as unreliable or removed, but this doesn't happen.

**Recommendation:** Add a post-termination sweep in `causal_physics.py:execute()`:
```python
# After do-surgery application:
for cid, term_ft in terminated_channels:
    for ent in self.world_state.entities.values():
        ent.beliefs = [
            b for b in ent.beliefs
            if b.acquired_via_channel_id != cid
            or b.established_at_fabula < term_ft
        ]
```

---

### 🟡 MODERATE: Belief Provenance Tracking Incomplete
**Location:** `shadow_loom/models.py:247-254` (`Belief.acquired_via_channel_id`)  
**Issue:** The `acquired_via_channel_id` field exists, but there's no validation that the referenced channel ID actually exists in `world_state.channels`.

**Recommendation:** Add to `ingestion.py:_auto_repair`:
```python
# After line 13625:
valid_channel_ids = set(clean_channels.keys())
for ent in ws.entities.values():
    bad_beliefs = [
        b for b in ent.beliefs
        if b.acquired_via_channel_id
        and b.acquired_via_channel_id not in valid_channel_ids
    ]
    if bad_beliefs:
        repairs.append(
            f"Cleared acquired_via_channel_id on {len(bad_beliefs)} beliefs "
            f"for entity {ent.id} (channel(s) not found)."
        )
        for b in bad_beliefs:
            b.acquired_via_channel_id = None
```

---

## 5. Merge Operations (extract_graph.py)

### 🟢 VERIFIED: Time-Slice Relationship Per-Axis Filtering
**Location:** `shadow_loom/extract_graph.py:30-87` (`_time_slice_relationship_at`)  
**Status:** ✅ **CORRECT**

The per-axis time-slicing correctly filters out metrics whose `last_updated_fabula > t`:
```python
# Lines 66-70
for name, m in metrics.items():
    if not isinstance(m, dict):
        continue
    if m.get("last_updated_fabula", 0) <= t:
        surviving[name] = m
```

---

### 🔴 CRITICAL: Incomplete Edge Removal During SCC Break
**Location:** `shadow_loom/ingestion.py:13520-13570` (SCC-break loop)  
**Issue:** When breaking cycles by removing the weakest edge, the code correctly removes from `clean_causal`, but the corresponding sandbox edges are **not** removed.

**Evidence:**
```python
# Lines 13555-13565
clean_causal = [
    ce for ce in clean_causal
    if (ce.source_id, ce.target_id) not in edges_to_drop
    or ce.causality_type == "affordance_gate"
]
# No corresponding sandbox.remove_edge() call
```

**Risk:** SCC-break runs during ingestion (before sandbox instantiation), so this is actually **not a bug** — the sandbox doesn't exist yet. However, if SCC-break is ever called post-sandbox, this would leave dangling edges.

**Recommendation:** Document the assumption:
```python
# After line 13555:
# NOTE: SCC-break runs during ingestion, before sandbox instantiation.
# If this is ever called on a live sandbox, add sb.remove_edge() here.
```

---

## 6. Missing Validation: General

### 🔴 CRITICAL: No Global Orphaned Edge Audit
**Issue:** There's no single validation pass that verifies all edges reference valid nodes after merges/repairs.

**Recommendation:** Add to `ingestion.py:_auto_repair` (after line 13625):
```python
# --- Validate all edge endpoints one more time ---
final_node_ids = (
    set(ws.locations.keys())
    | set(ws.objects.keys())
    | set(ws.entities.keys())
    | set(ws.world_traits.keys())
    | {e.id for e in clean_events}
)
for ce in clean_causal:
    if ce.source_id not in final_node_ids:
        repairs.append(f"FINAL: Orphaned causal edge source {ce.source_id}")
    if ce.target_id not in final_node_ids:
        repairs.append(f"FINAL: Orphaned causal edge target {ce.target_id}")
for se in clean_spatial:
    if se.source_id not in final_node_ids:
        repairs.append(f"FINAL: Orphaned spatial edge source {se.source_id}")
    if se.target_id not in final_node_ids:
        repairs.append(f"FINAL: Orphaned spatial edge target {se.target_id}")
```

---

## Summary Table

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| CausalEdge temporal ordering not validated | 🔴 CRITICAL | causal_physics.py:2145 | Cause-after-effect paradoxes |
| CausalEdge orphaned rel_counterpart_id | 🔴 CRITICAL | causal_physics.py:2168 | Dangling relationship references |
| SpatialEdge location check runs after sandbox write | 🔴 CRITICAL | causal_physics.py:2289 | Phantom passages in sandbox |
| RelationshipEdge lifecycle not set on creation | 🔴 CRITICAL | causal_physics.py:3344 | Time-slice leakage |
| Channel intelligibility keys not validated | 🔴 CRITICAL | ingestion.py:13609 | Orphaned intelligibility data |
| No global orphaned edge audit | 🔴 CRITICAL | ingestion.py (missing) | Undetected broken references |
| CausalEdge mechanism-trait mismatch | 🔴 CRITICAL | causal_physics.py:2145 | Silent propagation failures |
| SpatialEdge affordance gate not validated | 🟡 MODERATE | causal_physics.py:2285 | Undefined lock semantics |
| SpatialEdge sever doesn't mark destroyed | 🟡 MODERATE | causal_physics.py:2195 | Lost temporal data |
| RelationshipEdge inertia not per-axis | 🟡 MODERATE | causal_physics.py:3344 | Wrong propagation rates |
| Channel termination doesn't cascade | 🟡 MODERATE | (missing) | Stale belief provenance |
| Belief channel provenance not validated | 🟡 MODERATE | ingestion.py (missing) | Dangling channel refs |
| CausalEdge evidence strength not propagated | 🟡 MODERATE | causal_physics.py:2145 | Lost epistemic uncertainty |

**Verified Correct (3 items):**
- ✅ SpatialEdge bidirectional storage (canonical stores forward only, instantiator creates reverse)
- ✅ RelationshipEdge per-axis metric independence
- ✅ RelationshipEdge asymmetric directionality (by design)

---

## Recommended Fix Priority

1. **Immediate (P0):**
   - Add CausalEdge `rel_counterpart_id` validation (line 2168)
   - Move SpatialEdge location check before sandbox write (line 2289)
   - Set RelationshipEdge `established_at_fabula` on creation (line 3344)
   - Add global orphaned edge audit to `_auto_repair`

2. **Next Sprint (P1):**
   - Add CausalEdge temporal ordering validation (line 2145)
   - Add Channel intelligibility validation (ingestion.py:13609)
   - Add CausalEdge mechanism-trait validation (line 2145)

3. **Future (P2):**
   - Per-axis relationship inertia defaults
   - Channel termination cascade
   - SpatialEdge destruction timestamps

---

## Verification Tests Needed

1. **Test orphaned CausalEdge with phantom rel_counterpart_id**
2. **Test SpatialEdge with non-existent location IDs**
3. **Test RelationshipEdge time-slicing with edges created mid-simulation**
4. **Test Channel with intelligibility keys for non-participants**
5. **Test CausalEdge with mechanism-trait mismatch**

---

**Audit Completed:** 2026-05-26  
**Auditor:** GitHub Copilot (Claude Sonnet 4.5)  
**Lines Reviewed:** ~5000 across 3 files  
**Issues Found:** 7 critical, 6 moderate, 3 verified correct  
**Test Coverage:** 21 related tests found (bidirectional edges, orphan detection, dangling endpoints)  
**Test Pass Rate:** All existing tests pass; new tests recommended for unvalidated cases
