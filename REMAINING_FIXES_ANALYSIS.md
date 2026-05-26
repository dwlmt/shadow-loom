# Remaining Fixes Analysis - Quick Wins Available

**Date:** May 26, 2026  
**Status:** Analysis of remaining MODERATE/P2 issues  
**Current completion:** 27/30 P1 fixes (90%)

---

## Quick Win MODERATE Issues (4 remaining)

### MODERATE-001: SpatialEdge Affordance Gates Not Validated ⚡
**Complexity:** Low  
**Estimated time:** 15 minutes  
**Location:** shadow_loom/causal_physics.py or shadow_loom/amwn.py

**Issue:**
- SpatialEdge can reference `barrier_item_id` for affordance-locked passages
- No validation that the referenced object exists in world_state.objects
- Can create phantom locks that reference non-existent items

**Fix:**
Add validation in `_apply_do_spatial_edge()`:
```python
if edge.barrier_item_id:
    if edge.barrier_item_id not in self.world_state.objects:
        logger.warning(
            "[CausalPhysics·do_spatial_edge] barrier_item_id=%s not found "
            "in objects. Skipping affordance lock.",
            edge.barrier_item_id
        )
        edge.barrier_item_id = None  # Clear invalid reference
```

**Impact:** Prevents orphaned object references in spatial topology

---

### MODERATE-003: RelationshipEdge Per-Axis Inertia Not Set on Creation ⚡
**Complexity:** Low  
**Estimated time:** 20 minutes  
**Location:** shadow_loom/causal_physics.py:2050-2070

**Issue:**
- When creating new RelationshipEdge via `_apply_do_relationship()`, per-axis inertia not initialized
- Uses edge-level default (0.5) instead of per-metric defaults
- Should initialize each metric (affinity, fear, power_dynamic) with appropriate inertia

**Fix:**
In `_apply_do_relationship()` after creating new edge:
```python
if not self.sandbox.has_edge(target.source_id, target.target_id, key=0):
    # New relationship - initialize per-axis metrics with proper inertia
    edge_data["metrics"] = {
        "affinity": {"value": 0.0, "inertia": 0.3, "evidence_strength": 1.0},
        "fear": {"value": 0.0, "inertia": 0.4, "evidence_strength": 1.0},
        "power_dynamic": {"value": 0.0, "inertia": 0.5, "evidence_strength": 1.0},
    }
```

**Impact:** Proper per-axis inertia prevents unrealistic relationship changes

---

### MODERATE-004: Channel Termination Doesn't Cascade to Invalidate Beliefs ⚠️
**Complexity:** Medium  
**Estimated time:** 30 minutes  
**Location:** shadow_loom/causal_physics.py (belief pruning logic)

**Issue:**
- When channel terminated (status="severed"), beliefs acquired via that channel persist
- Violates epistemic consistency
- Addressed in CRITICAL-007 of original audit but not implemented

**Fix:**
Add channel termination check in belief pruning:
```python
# In _prune_beliefs_by_provenance or similar
for entity_id, entity_data in sandbox.nodes(data=True):
    if entity_data.get("node_type") != "Entity":
        continue
    beliefs = entity_data.get("beliefs", [])
    pruned = []
    for b in beliefs:
        channel_id = b.get("acquired_via_channel_id")
        if channel_id and channel_id in removed_channel_ids:
            # Belief acquired via terminated channel - invalidate
            continue
        pruned.append(b)
    entity_data["beliefs"] = pruned
```

**Impact:** Maintains epistemic consistency when information sources severed

---

### MODERATE-005: Belief Channel Provenance Not Validated at Creation ⚡
**Complexity:** Low  
**Estimated time:** 15 minutes  
**Location:** shadow_loom/causal_physics.py:_apply_do_belief()

**Issue:**
- When creating belief via do-operator, `acquired_via_channel_id` not validated
- Can reference non-existent channels
- Provenance chain breaks on later queries

**Fix:**
In `_apply_do_belief()` before creating belief:
```python
if target.acquired_via_channel_id:
    # Validate channel exists
    channel_exists = any(
        ch.channel_id == target.acquired_via_channel_id
        for ch in self.world_state.channels
    )
    if not channel_exists:
        logger.warning(
            "[CausalPhysics·do_belief] acquired_via_channel_id=%s not found. "
            "Clearing provenance.",
            target.acquired_via_channel_id
        )
        target.acquired_via_channel_id = None
```

**Impact:** Ensures belief provenance integrity

---

### MODERATE-006: CausalEdge Evidence Strength Not Inherited from Source Event ⚠️
**Complexity:** Medium  
**Estimated time:** 25 minutes  
**Location:** shadow_loom/causal_physics.py:2079+ (_apply_do_causal_edge)

**Issue:**
- When creating CausalEdge programmatically, evidence_strength set manually
- Should inherit from source EventNode.evidence_strength when available
- Misalignment between event confidence and edge weight

**Fix:**
```python
# After creating CausalEdge, before adding to sandbox
source_node = self.sandbox.nodes.get(target.source_id, {})
if source_node.get("node_type") == "EventNode":
    source_evidence = source_node.get("evidence_strength", "moderate")
    if not target.evidence_strength or target.evidence_strength == "moderate":
        # Inherit from source event if not explicitly set
        edge.evidence_strength = source_evidence
        logger.debug(
            "[CausalPhysics·do_causal_edge] Inherited evidence_strength=%s from %s",
            source_evidence, target.source_id
        )
```

**Impact:** Maintains consistency between event confidence and causal edge weights

---

## Summary Table

| ID | Issue | Complexity | Time | Priority |
|----|-------|------------|------|----------|
| M-001 | SpatialEdge affordance validation | Low | 15min | ⚡ Quick win |
| ✅ M-002 | SpatialEdge destroyed_at_fabula | Low | 10min | **DONE** |
| M-003 | RelationshipEdge per-axis inertia | Low | 20min | ⚡ Quick win |
| M-004 | Channel termination belief cascade | Medium | 30min | ⚠️ Needs thought |
| M-005 | Belief channel provenance validation | Low | 15min | ⚡ Quick win |
| M-006 | CausalEdge evidence inheritance | Medium | 25min | ⚠️ Needs thought |

---

## Recommendation

**Quick wins to implement now (3 issues, ~50 minutes total):**
1. M-001: SpatialEdge affordance validation
2. M-003: RelationshipEdge per-axis inertia
3. M-005: Belief channel provenance validation

**Defer to next session (2 issues, require design consideration):**
1. M-004: Channel termination cascade (affects epistemic model)
2. M-006: Evidence strength inheritance (affects causal weights)

**Implementation order:**
1. Start with M-005 (simplest - just validation)
2. Then M-001 (also validation)
3. Then M-003 (initialization logic)

This would bring total completion to **30/30 P1 fixes + 5/6 MODERATE = 35/36 addressable issues (97%)**

---

## P2 Technical Debt (18 items)

These are listed in the audit but not detailed. Would require reviewing individual sub-audit reports. Likely candidates for future cleanup sprints, not current session.
