# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data transforms for Shadow-Loom's *reasoning-trace* views.

These helpers convert the structured outputs of the causal physics
engine, the auditor, and the world model into shapes that the new
"Reasoning" tab and chat structured-response card consume.

Pure functions only — no NiceGUI or ECharts imports here so they can
be unit-tested without a UI runtime. The companion module
:mod:`shadow_loom_ui.reasoning_viz` contains the matching renderers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Violation type → plain-English explanation
# =====================================================================
#
# Sourced from the ``Literal`` on
# :class:`shadow_loom.auditor.AuditViolation.violation_type` —
# every value has an entry. Used by the audit tab to render a
# "why this matters" caption next to each violation.

VIOLATION_EXPLANATIONS: Dict[str, str] = {
    "epistemic_leakage": (
        "A character acted on information they could not yet have known. "
        "The prose leaked the narrator's omniscient view into a "
        "non-omniscient point of view."
    ),
    "knowledge_contamination": (
        "Two characters' belief states were conflated. Information "
        "passed between minds without a supporting causal channel "
        "(dialogue, observation, telepathy)."
    ),
    "low_kl_divergence": (
        "The revelation was insufficiently surprising. The reader's "
        "prior over outcomes barely shifted, so the moment lacks impact."
    ),
    "suspense_threshold": (
        "Tension dropped below the directive's target. Either the "
        "stakes weren't visible or the threat resolved too quickly."
    ),
    "tonal_mismatch": (
        "The prose register doesn't match the requested affect — "
        "comic phrasing in a tragic scene, or vice versa."
    ),
    "magnitude_too_low": (
        "The targeted emotion was hit, but at insufficient intensity "
        "to satisfy the directive's intensity parameter."
    ),
    "reasoning_failure": (
        "A character's chain of inference is non sequitur — they "
        "reached a conclusion their priors couldn't support."
    ),
    "affective_failure": (
        "The achieved emotional vector diverges from the requested one "
        "by more than the auditor's MSE tolerance."
    ),
    "attribution_failure": (
        "The prose obscures *who caused what* — actor IDs in the "
        "causal graph aren't traceable from the narrative."
    ),
    "empathy_weight": (
        "The reader has insufficient grounding in the affected "
        "character to feel the targeted emotion. Their stakes need "
        "earlier setup."
    ),
    "miracle_step": (
        "A state change occurred without a sufficient cause in the "
        "causal graph — the impact didn't exceed the target's inertia."
    ),
    "abduction_failure": (
        "A counterfactual claim could not be reconciled with present-day "
        "evidence. The implied past contradicts what we know is true now."
    ),
    "utterance_truth_contradiction": (
        "An on-page utterance was rendered as sincere when the world "
        "graph marks its content as a lie or as performative. The "
        "narrator should foreground the gap between what is said and "
        "what is true."
    ),
    "channel_intelligibility_violation": (
        "A character acquired a high-confidence belief through a "
        "channel where their intelligibility is low (encrypted line, "
        "foreign tongue, partial overhearing). Either lower the "
        "confidence or surface the partial decoding in the prose."
    ),
    "withheld_utterance_leak": (
        "An utterance whose syuzhet_index is still in the future was "
        "quoted, paraphrased, or implied in the current scene. The "
        "narrator pre-emptively revealed information the reader has "
        "not yet been given."
    ),
    "belief_provenance_contradiction": (
        "A character cited a belief whose provenance (acquired_via_event "
        "or acquired_via_channel) does not match how the prose said "
        "they came to know it. Reconcile the source of the knowledge."
    ),
    "style_mismatch": (
        "The prose register drifted from the source-style profile "
        "captured at ingestion (sentence length, vocabulary tier, "
        "voice). Tighten the wording to match the original author's "
        "fingerprint."
    ),
    "meta_narration": (
        "The prose stepped outside the diegesis to comment on its own "
        "structure — timelines, branches, counterfactual machinery — "
        "instead of rendering the world as a lived scene. Rewrite as "
        "embodied action and observation."
    ),
}


def violation_explanation(violation_type: str) -> str:
    """Return the plain-English explanation for ``violation_type``.

    Falls back to a generic note if the type isn't known so the UI
    never crashes on a future auditor change.
    """
    return VIOLATION_EXPLANATIONS.get(
        violation_type,
        "An unrecognised audit category fired. See the auditor logs.",
    )


# =====================================================================
# Audit violation → suggested directive (for actionable chips)
# =====================================================================
#
# Each entry maps a violation_type to a *prefilled prompt* and the
# query_type that the chat command bar should run. The audit tab
# renders these as one-click chips so a fan-analyst or writer can go
# from "audit found a problem" to "engine running a corrective query"
# without retyping anything.

VIOLATION_DIRECTIVE_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "miracle_step": (
        "Re-derive the flagged state change with sufficient causal force. "
        "Ensure impact > inertia for every mutated trait.",
        "directive",
    ),
    "epistemic_leakage": (
        "Rewrite the scene so the POV character only acts on facts they "
        "could plausibly know at this fabula time.",
        "directive",
    ),
    "knowledge_contamination": (
        "Add an explicit information channel (dialogue, observation, or "
        "message) before any character acts on knowledge originally held "
        "by another.",
        "directive",
    ),
    "low_kl_divergence": (
        "Increase the surprise of the revelation: add misdirection or "
        "raise the prior probability of the alternative outcome.",
        "directive",
    ),
    "suspense_threshold": (
        "Raise tension: prolong the threat, make stakes more visible, "
        "or delay the resolution.",
        "directive",
    ),
    "tonal_mismatch": (
        "Rewrite the scene in a register that matches the target effect.",
        "directive",
    ),
    "magnitude_too_low": (
        "Intensify the targeted emotion. Push intensity to 1.0.",
        "directive",
    ),
    "reasoning_failure": (
        "Rewrite the character's inference chain so each step is "
        "supported by their established beliefs.",
        "directive",
    ),
    "affective_failure": (
        "Re-target the affective vector to better match the directive's "
        "requested intensities.",
        "directive",
    ),
    "attribution_failure": (
        "Make causal attribution explicit in the prose: name the actor "
        "for every consequential action.",
        "directive",
    ),
    "empathy_weight": (
        "Add a brief grounding moment for the affected character before "
        "the consequential event.",
        "directive",
    ),
    "abduction_failure": (
        "Reconcile the counterfactual past with present-day evidence — "
        "either revise the abducted hidden_deltas or weaken the claim.",
        "counterfactual",
    ),
    "utterance_truth_contradiction": (
        "Rewrite the flagged utterance so the prose treatment matches its "
        "declared truth_value: foreground the lie / mark the line as "
        "performative / acknowledge the speaker's uncertainty.",
        "directive",
    ),
    "channel_intelligibility_violation": (
        "Lower the listener's confidence in the resulting belief, or stage "
        "the partial decoding in the prose so the channel's intelligibility "
        "is honoured.",
        "directive",
    ),
    "withheld_utterance_leak": (
        "Defer or rephrase the leaked content so its on-page disclosure no "
        "longer precedes the source utterance's syuzhet_index.",
        "directive",
    ),
    "belief_provenance_contradiction": (
        "Reconcile the prose with the belief's recorded provenance: either "
        "name the same source event/channel, or add the missing acquisition "
        "step before the character acts on the knowledge.",
        "directive",
    ),
}


def directive_for_violation(violation_type: str) -> Optional[Tuple[str, str]]:
    """Return ``(prompt_text, query_type)`` for one-click corrective action.

    Returns ``None`` when no template is registered for the type so
    the UI can skip rendering an action chip.
    """
    return VIOLATION_DIRECTIVE_TEMPLATES.get(violation_type)


# =====================================================================
# Reasoning trace — the headline rung-2 / rung-3 explanation surface
# =====================================================================


def extract_reasoning_trace(
    physics_result: Dict[str, Any] | None,
    *,
    ws: Optional[WorldStateV1] = None,
) -> Dict[str, Any]:
    """Pull every causal-physics signal needed by the Reasoning Trace rail.

    The pipeline stores a denormalised view of the physics engine's
    output on :attr:`PipelineResult.physics_result`. This helper folds
    that dict into a single structured object the UI can render
    without further inspection:

    ``{
        "rung": 2 | 3 | None,
        "do_set": [{"node_id", "label", "value"}],
        "evidence": [{"node_id", "label"}],          # rung-3 only
        "abduction": [{"node_id", "trait", "delta"}],# rung-3 only
        "cascade": [{"node_id", "trait", "old", "new", "impact", "inertia",
                     "edge_kind", "delay", "trigger"}],
        "social_cascade": [{"source", "target", "metric", "old", "new",
                            "impact", "inertia", "trigger"}],
        "blocked": [{"node_id", "trait", "impact", "inertia", "reason"}],
    }``

    All fields are optional in the source dict — missing pieces become
    empty lists, which the renderer handles cleanly.
    """
    out: Dict[str, Any] = {
        "rung": None,
        "do_set": [],
        "evidence": [],
        "abduction": [],
        "cascade": [],
        "social_cascade": [],
        "blocked": [],
        # ctf-calculus pre-flight surface (Correa & Bareinboim 2025)
        "rule3_pruned_interventions": [],
        "rule2_redundant_evidence": [],
        "cyclic_propagation_clusters": [],
        # Channels & beliefs subsystem — counterfactual epistemic fallout
        "pruned_beliefs_count": 0,
        "pruned_utterance_event_ids": [],
        "disabled_channel_ids": [],
        # Information provenance: who-told-whom subgraph rooted at focal node
        "information_flow": None,
    }
    if not physics_result:
        return out

    qtype = physics_result.get("query_type") or ""
    if qtype == "intervention":
        out["rung"] = 2
    elif qtype == "counterfactual":
        out["rung"] = 3

    label = _label_fn_for(ws)

    # ── do-set ────────────────────────────────────────────────────
    intervened = (
        physics_result.get("intervened_nodes")
        or physics_result.get("interventions")
        or []
    )
    if isinstance(intervened, dict):
        for nid, val in intervened.items():
            out["do_set"].append({
                "node_id": nid, "label": label(nid), "value": str(val),
            })
    elif isinstance(intervened, list):
        for item in intervened:
            if isinstance(item, dict):
                nid = item.get("node_id") or item.get("id") or ""
                val = item.get("value", "")
                out["do_set"].append({
                    "node_id": nid,
                    "label": label(nid),
                    "value": str(val),
                })
            else:
                out["do_set"].append({
                    "node_id": str(item),
                    "label": label(str(item)),
                    "value": "",
                })

    # ── evidence + abduction (rung-3 only) ────────────────────────
    for nid in physics_result.get("evidence_node_ids", []) or []:
        out["evidence"].append({"node_id": nid, "label": label(nid)})

    hidden = physics_result.get("hidden_deltas") or {}
    if isinstance(hidden, dict):
        for nid, traits in hidden.items():
            if not isinstance(traits, dict):
                continue
            for trait, delta in traits.items():
                try:
                    delta_f = float(delta)
                except (TypeError, ValueError):
                    continue
                out["abduction"].append({
                    "node_id": nid,
                    "label": label(nid),
                    "trait": trait,
                    "delta": delta_f,
                })

    # ── cascade ───────────────────────────────────────────────────
    for m in physics_result.get("mutations") or []:
        if not isinstance(m, dict):
            continue
        out["cascade"].append({
            "node_id": m.get("node_id", ""),
            "label": label(m.get("node_id", "")),
            "trait": m.get("trait", ""),
            "old": float(m.get("old_value", 0.0) or 0.0),
            "new": float(m.get("new_value", 0.0) or 0.0),
            "impact": float(m.get("impact", 0.0) or 0.0),
            "inertia": float(m.get("inertia", 0.0) or 0.0),
            "edge_kind": m.get("edge_kind", "mutation"),
            "delay": int(m.get("propagation_delay", 0) or 0),
            "trigger": m.get("triggered_by", ""),
        })

    for sm in physics_result.get("social_mutations") or []:
        if not isinstance(sm, dict):
            continue
        out["social_cascade"].append({
            "source": sm.get("source_entity_id", ""),
            "source_label": label(sm.get("source_entity_id", "")),
            "target": sm.get("target_entity_id", ""),
            "target_label": label(sm.get("target_entity_id", "")),
            "metric": sm.get("metric", "affinity"),
            "old": float(sm.get("old_value", 0.0) or 0.0),
            "new": float(sm.get("new_value", 0.0) or 0.0),
            "impact": float(sm.get("impact", 0.0) or 0.0),
            "inertia": float(sm.get("inertia", 0.0) or 0.0),
            "trigger": sm.get("triggered_by", ""),
        })

    for b in physics_result.get("blocked") or []:
        if not isinstance(b, dict):
            continue
        out["blocked"].append({
            "node_id": b.get("node_id", ""),
            "label": label(b.get("node_id", "")),
            "trait": b.get("trait", ""),
            "impact": float(b.get("impact", 0.0) or 0.0),
            "inertia": float(b.get("inertia", 0.0) or 0.0),
            "reason": b.get("reason", "inertia"),
        })

    # ── ctf-calculus pre-flight (Rules 2 & 3) ─────────────────────
    # The engine reports interventions and evidence the AMWN d-separation
    # check proved vacuous before simulation. Cycle-blocked propagations
    # are the third bucket that shouldn't show up as narrative miracles.
    for path in physics_result.get("rule3_pruned_interventions") or []:
        node_id = str(path).split(".", 1)[0] if "." in str(path) else str(path)
        out["rule3_pruned_interventions"].append({
            "path": str(path),
            "node_id": node_id,
            "label": label(node_id),
        })
    for nid in physics_result.get("rule2_redundant_evidence") or []:
        out["rule2_redundant_evidence"].append({
            "node_id": str(nid),
            "label": label(str(nid)),
        })
    # Cyclic clusters are also surfaced from blocked entries (reason="cycle")
    # so the UI can collapse them into a separate panel from miracle blocks.
    for b in out["blocked"]:
        if b.get("reason") == "cycle":
            out["cyclic_propagation_clusters"].append({
                "node_id": b.get("node_id", ""),
                "label": b.get("label", ""),
                "trait": b.get("trait", ""),
            })

    # ── Channels & beliefs counterfactual side-effects ───────────
    # When a counterfactual surgery removes utterance events or
    # disables channels, every belief whose ``acquired_via_*`` provenance
    # pointed at those IDs is no longer justifiable. The engine reports
    # the resulting prune so the UI can show the epistemic fallout next
    # to the structural cascade.
    try:
        out["pruned_beliefs_count"] = int(
            physics_result.get("pruned_beliefs_count", 0) or 0
        )
    except (TypeError, ValueError):
        out["pruned_beliefs_count"] = 0
    for eid in physics_result.get("pruned_utterance_event_ids") or []:
        out["pruned_utterance_event_ids"].append({
            "node_id": str(eid),
            "label": label(str(eid)),
        })
    for cid in physics_result.get("disabled_channel_ids") or []:
        out["disabled_channel_ids"].append({
            "node_id": str(cid),
            "label": label(str(cid)),
        })

    # ── Information provenance subgraph ───────────────────────────
    # Walk the utterance/channel graph rooted at the most salient
    # focal node (a do-set target, an evidence node, or a pruned
    # utterance). Renders alongside the structural cascade so the
    # epistemic path utterance \u2192 belief is legible.
    if ws is not None:
        focal_id: Optional[str] = None
        if out["do_set"]:
            focal_id = out["do_set"][0].get("node_id")
        elif out["evidence"]:
            focal_id = out["evidence"][0].get("node_id")
        elif out["pruned_utterance_event_ids"]:
            focal_id = out["pruned_utterance_event_ids"][0].get("node_id")
        elif out["disabled_channel_ids"]:
            focal_id = out["disabled_channel_ids"][0].get("node_id")
        if focal_id:
            try:
                from shadow_loom.projections import trace_information_flow

                flow = trace_information_flow(ws, focal_id, depth=2)
                # Decorate edges with human labels for the renderer.
                edge_rows = []
                node_ids: set[str] = set()
                for e in flow.get("edges", []):
                    s, t = e.get("source"), e.get("target")
                    if not (s and t):
                        continue
                    node_ids.add(s)
                    node_ids.add(t)
                    edge_rows.append({
                        "source": s,
                        "target": t,
                        "source_label": label(s),
                        "target_label": label(t),
                        "kind": e.get("kind"),
                        "channel_id": e.get("channel_id"),
                        "truth_value": e.get("truth_value"),
                        "intelligibility": e.get("intelligibility"),
                    })
                out["information_flow"] = {
                    "focal_id": focal_id,
                    "focal_label": label(focal_id),
                    "edges": edge_rows,
                    "node_count": len(node_ids),
                }
            except Exception:  # pragma: no cover - defensive UI path
                out["information_flow"] = None

    return out


def reasoning_trace_summary(trace: Dict[str, Any]) -> str:
    """One-line summary suitable for chat or task notification."""
    parts: List[str] = []
    if trace.get("rung"):
        parts.append(f"rung-{trace['rung']}")
    if trace.get("do_set"):
        parts.append(f"{len(trace['do_set'])} pinned")
    if trace.get("abduction"):
        parts.append(f"{len(trace['abduction'])} abduced")
    if trace.get("cascade"):
        parts.append(f"{len(trace['cascade'])} mutations")
    if trace.get("social_cascade"):
        parts.append(f"{len(trace['social_cascade'])} social")
    if trace.get("blocked"):
        parts.append(f"{len(trace['blocked'])} blocked")
    if trace.get("rule3_pruned_interventions"):
        parts.append(f"{len(trace['rule3_pruned_interventions'])} rule3-pruned")
    if trace.get("rule2_redundant_evidence"):
        parts.append(f"{len(trace['rule2_redundant_evidence'])} rule2-redundant")
    if trace.get("cyclic_propagation_clusters"):
        parts.append(f"{len(trace['cyclic_propagation_clusters'])} cycle-blocked")
    if trace.get("pruned_beliefs_count"):
        parts.append(f"{trace['pruned_beliefs_count']} beliefs pruned")
    if trace.get("pruned_utterance_event_ids"):
        parts.append(f"{len(trace['pruned_utterance_event_ids'])} utterances neutralised")
    if trace.get("disabled_channel_ids"):
        parts.append(f"{len(trace['disabled_channel_ids'])} channels severed")
    return " · ".join(parts) if parts else "no reasoning trace"


# =====================================================================
# Belief provenance — for the writer's "why does this character think X?"
# =====================================================================


def belief_provenance_data(
    ws: WorldStateV1,
    entity_id: str,
) -> List[Dict[str, Any]]:
    """Return a chronological provenance trail for one entity's beliefs.

    Combines:

    * Initial beliefs declared on :class:`Entity.beliefs` (origin = pre-story).
    * Each :class:`EntityStateSnapshot.beliefs_added` (origin = the
      ``triggered_by`` event description).
    * Each :class:`EntityStateSnapshot.beliefs_invalidated` entry
      (rendered as an "invalidated" row at the snapshot's fabula_time).

    Each row::

        {"fabula_time", "kind", "target_id", "target_label",
         "perceived_state", "confidence", "inertia",
         "trigger_id", "trigger_label"}

    ``kind`` is one of ``"initial"``, ``"added"``, ``"invalidated"``.
    """
    ent = ws.entities.get(entity_id)
    if ent is None:
        return []

    label = _label_fn_for(ws)
    rows: List[Dict[str, Any]] = []

    # ── Initial beliefs ───────────────────────────────────────────
    for b in ent.beliefs:
        rows.append({
            "fabula_time": int(b.established_at_fabula or 0),
            "kind": "initial",
            "target_id": b.target_id,
            "target_label": label(b.target_id),
            "perceived_state": b.perceived_state,
            "confidence": float(b.confidence),
            "inertia": float(b.inertia),
            "trigger_id": "",
            "trigger_label": "(pre-story)",
        })

    # ── Snapshot deltas ───────────────────────────────────────────
    for snap in sorted(ent.state_timeline, key=lambda s: s.fabula_time):
        trig_id = snap.triggered_by or ""
        trig_label = label(trig_id) if trig_id else ""
        for b in snap.beliefs_added:
            rows.append({
                "fabula_time": int(snap.fabula_time),
                "kind": "added",
                "target_id": b.target_id,
                "target_label": label(b.target_id),
                "perceived_state": b.perceived_state,
                "confidence": float(b.confidence),
                "inertia": float(b.inertia),
                "trigger_id": trig_id,
                "trigger_label": trig_label,
                "acquired_via_event_id": b.acquired_via_event_id or "",
                "acquired_via_event_label": (
                    label(b.acquired_via_event_id) if b.acquired_via_event_id else ""
                ),
                "acquired_via_channel_id": b.acquired_via_channel_id or "",
                "acquired_via_channel_label": (
                    label(b.acquired_via_channel_id) if b.acquired_via_channel_id else ""
                ),
            })
        for tid in snap.beliefs_invalidated:
            rows.append({
                "fabula_time": int(snap.fabula_time),
                "kind": "invalidated",
                "target_id": tid,
                "target_label": label(tid),
                "perceived_state": "",
                "confidence": 0.0,
                "inertia": 0.0,
                "trigger_id": trig_id,
                "trigger_label": trig_label,
                "acquired_via_event_id": "",
                "acquired_via_event_label": "",
                "acquired_via_channel_id": "",
                "acquired_via_channel_label": "",
            })

    rows.sort(key=lambda r: (r["fabula_time"], 0 if r["kind"] == "initial" else 1))
    return rows


# =====================================================================
# Attribution graph — reverse-walk causal edges from a target outcome
# =====================================================================


def attribution_graph_data(
    ws: WorldStateV1,
    target_id: str,
    *,
    max_depth: int = 4,
    min_force: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """BFS backwards from ``target_id`` along causal edges.

    Returns ``(nodes, links, categories, ranked_paths)`` where:

    * ``nodes`` and ``links`` are ECharts graph payloads.
    * ``categories`` is the per-node-type legend list.
    * ``ranked_paths`` is a list of dicts ``{"path": [labels],
      "force": product of causal_force, "depth": int}`` sorted by
      contribution strength — answers "what most caused X?" directly.

    Edges below ``min_force`` are ignored. Cycles are broken by the
    visited set.
    """
    if target_id not in _all_node_ids(ws):
        return [], [], [], []

    label = _label_fn_for(ws)

    # Build reverse adjacency: target → list[(source, edge)]
    rev: Dict[str, List[Tuple[str, CausalEdge]]] = {}
    for ce in ws.causal_topology:
        if ce.causal_force < min_force:
            continue
        rev.setdefault(ce.target_id, []).append((ce.source_id, ce))

    visited: set[str] = {target_id}
    nodes: Dict[str, Dict[str, Any]] = {}
    links: List[Dict[str, Any]] = []

    def _ensure_node(nid: str, *, depth: int) -> None:
        if nid in nodes:
            # Keep shallowest depth
            nodes[nid]["_depth"] = min(nodes[nid].get("_depth", depth), depth)
            return
        ntype = _node_type(ws, nid)
        nodes[nid] = {
            "id": nid,
            "name": label(nid),
            "_node_type": ntype,
            "_depth": depth,
        }

    _ensure_node(target_id, depth=0)
    nodes[target_id]["itemStyle"] = {"color": "#D8334A", "borderColor": "#FFD700", "borderWidth": 3}

    frontier: List[Tuple[str, int]] = [(target_id, 0)]
    paths: List[Dict[str, Any]] = []

    # Track per-node best path (highest force) by remembering parent
    parent_path: Dict[str, Dict[str, Any]] = {
        target_id: {"path": [label(target_id)], "force": 1.0, "depth": 0},
    }

    while frontier:
        next_frontier: List[Tuple[str, int]] = []
        for nid, depth in frontier:
            if depth >= max_depth:
                continue
            for src, edge in rev.get(nid, []):
                _ensure_node(src, depth=depth + 1)
                links.append({
                    "source": src,
                    "target": nid,
                    "value": float(edge.causal_force),
                    "lineStyle": {
                        "color": "#D8334A",
                        "width": max(1.0, min(6.0, edge.causal_force / 1.5)),
                        "type": "dashed" if edge.evidence_strength == "weak" else "solid",
                    },
                    "label": {"show": False},
                    "_mechanism": edge.mechanism,
                    "_evidence": edge.evidence_strength,
                    "_kind": edge.causality_type,
                })

                # Path extension via best-force ancestor
                base = parent_path.get(nid)
                if base is not None:
                    new_force = base["force"] * max(0.1, edge.causal_force)
                    new_depth = base["depth"] + 1
                    new_path = [label(src)] + base["path"]
                    prev = parent_path.get(src)
                    if prev is None or new_force > prev["force"]:
                        parent_path[src] = {
                            "path": new_path,
                            "force": new_force,
                            "depth": new_depth,
                        }

                if src not in visited:
                    visited.add(src)
                    next_frontier.append((src, depth + 1))
        frontier = next_frontier

    # ── Build ranked paths from leaves (nodes with no incoming) ───
    for nid, info in parent_path.items():
        if nid == target_id:
            continue
        if not rev.get(nid):  # leaf in the reverse graph
            paths.append({
                "path": info["path"],
                "force": info["force"],
                "depth": info["depth"],
                "root_id": nid,
            })
    paths.sort(key=lambda p: p["force"], reverse=True)

    # ── Style nodes by depth (fade further-away contributors) ─────
    type_colors = {
        "Entity": "#F26B5E",
        "Location": "#3A7BD5",
        "EventNode": "#F5B43C",
        "NarrativeObject": "#8A5CF0",
        "WorldTrait": "#2EA6A0",
        "unknown": "#9E9E9E",
    }
    type_index = {
        "Entity": 0, "Location": 1, "EventNode": 2,
        "NarrativeObject": 3, "WorldTrait": 4, "unknown": 5,
    }
    cats = [{"name": k} for k in type_index]

    out_nodes: List[Dict[str, Any]] = []
    for nid, n in nodes.items():
        depth = n.get("_depth", 0)
        ntype = n.get("_node_type", "unknown")
        opacity = max(0.35, 1.0 - depth * 0.18)
        n.setdefault("itemStyle", {"color": type_colors.get(ntype, "#9E9E9E")})
        n["itemStyle"].setdefault("opacity", opacity)
        n["category"] = type_index.get(ntype, type_index["unknown"])
        n["symbolSize"] = 32 if depth == 0 else max(14, 28 - depth * 4)
        n["tooltip"] = {
            "formatter": (
                f"<b>{n['name']}</b><br/>type: {ntype}<br/>"
                f"hops to outcome: {depth}"
            ),
        }
        # Strip private keys
        clean = {k: v for k, v in n.items() if not k.startswith("_")}
        out_nodes.append(clean)

    return out_nodes, links, cats, paths


# =====================================================================
# Foreshadowing → payoff arcs
# =====================================================================


def foreshadowing_arcs_data(
    ws: WorldStateV1,
    *,
    max_arc_span: int = 100,
) -> List[Dict[str, Any]]:
    """Find setup→payoff arcs in the causal topology.

    Heuristic: for every causal edge whose ``propagation_delay`` is
    non-zero (i.e. the cause and effect are separated in fabula time),
    treat the source as a *setup* and the target as a *payoff*. The
    UI renders each arc as a curved line connecting the two events on
    a fabula-time axis.

    Each row::

        {"setup_id", "setup_label", "setup_t",
         "payoff_id", "payoff_label", "payoff_t",
         "span", "force", "mechanism", "evidence", "is_loose"}

    ``is_loose`` is True for setups whose payoff target_id does NOT
    appear in :attr:`WorldStateV1.events` — i.e. unpaid Chekhov's-gun
    setups the writer should resolve.
    """
    label = _label_fn_for(ws)
    event_ids = {evt.id for evt in ws.events}
    fabula_for: Dict[str, int] = {evt.id: int(evt.fabula_time) for evt in ws.events}

    rows: List[Dict[str, Any]] = []
    for ce in ws.causal_topology:
        if ce.propagation_delay <= 0:
            continue
        setup_t = fabula_for.get(ce.source_id, ce.fabula_time)
        payoff_t = fabula_for.get(ce.target_id, ce.fabula_time + ce.propagation_delay)
        span = max(1, payoff_t - setup_t)
        if span > max_arc_span:
            continue
        rows.append({
            "setup_id": ce.source_id,
            "setup_label": label(ce.source_id),
            "setup_t": setup_t,
            "payoff_id": ce.target_id,
            "payoff_label": label(ce.target_id),
            "payoff_t": payoff_t,
            "span": span,
            "force": float(ce.causal_force),
            "mechanism": ce.mechanism,
            "evidence": ce.evidence_strength,
            "is_loose": ce.target_id not in event_ids and ce.target_id not in ws.entities,
        })
    rows.sort(key=lambda r: (r["setup_t"], r["payoff_t"]))
    return rows


# =====================================================================
# Convergence trajectory — refinement-loop quality over iterations
# =====================================================================


def convergence_trajectory_data(feedback_result: Any) -> List[Dict[str, Any]]:
    """Per-iteration metric trajectory from a :class:`FeedbackLoopResult`.

    Returns a list of ``{"iteration", "violation_count",
    "critical_count", "passed"}`` — one row per cycle in
    ``feedback_result.history``. Empty list if feedback_result is
    falsy or has no history.
    """
    if not feedback_result:
        return []
    history = getattr(feedback_result, "history", None) or []
    rows: List[Dict[str, Any]] = []
    for cycle in history:
        audit = getattr(cycle, "audit_result", None)
        violations = getattr(audit, "violations", []) if audit else []
        crit = sum(
            1 for v in violations if getattr(v, "severity", "") == "critical"
        )
        rows.append({
            # ``cycle.iteration`` is 0-based in the data model; humans
            # count from 1, and the trajectory chart x-axis labels
            # this value directly.
            "iteration": int(getattr(cycle, "iteration", len(rows))) + 1,
            "violation_count": len(violations),
            "critical_count": crit,
            "passed": bool(getattr(audit, "passed", False)) if audit else False,
        })
    return rows


# =====================================================================
# World diff — A vs B for the version-compare overlay
# =====================================================================


def world_diff_data(
    ws_a: WorldStateV1,
    ws_b: WorldStateV1,
) -> Dict[str, Any]:
    """Shallow diff between two world states.

    Returns::

        {
            "added":   {"entities": [id...], "events": [id...], ...},
            "removed": {...},
            "changed": {"entities": [{"id", "field", "from", "to"}], ...},
            "totals":  {"added": int, "removed": int, "changed": int},
        }

    Only the most user-meaningful fields are diffed:

    * Entity: ``status``, ``location_id``, top-level traits (value).
    * Event: presence (added/removed only — events are immutable in
      practice).
    * Object: ``location_id``, ``owner_id``.
    * GlobalTrait: ``magnitude.value``.
    * Causal edges: presence by ``(source_id, target_id, causality_type)``.
    """

    def _ent_traits(ent: Entity) -> Dict[str, float]:
        return {k: float(v.value) for k, v in ent.traits.items()}

    diff: Dict[str, Any] = {
        "added": {
            "entities": [], "events": [], "objects": [],
            "world_traits": [], "causal_edges": [],
            "channels": [], "utterance_events": [],
        },
        "removed": {
            "entities": [], "events": [], "objects": [],
            "world_traits": [], "causal_edges": [],
            "channels": [], "utterance_events": [],
        },
        "changed": {"entities": [], "objects": [], "world_traits": []},
    }

    a_ent = set(ws_a.entities)
    b_ent = set(ws_b.entities)
    diff["added"]["entities"] = sorted(b_ent - a_ent)
    diff["removed"]["entities"] = sorted(a_ent - b_ent)
    for eid in a_ent & b_ent:
        a, b = ws_a.entities[eid], ws_b.entities[eid]
        if a.status != b.status:
            diff["changed"]["entities"].append({
                "id": eid, "field": "status", "from": a.status, "to": b.status,
            })
        if a.location_id != b.location_id:
            diff["changed"]["entities"].append({
                "id": eid, "field": "location_id",
                "from": a.location_id, "to": b.location_id,
            })
        a_traits, b_traits = _ent_traits(a), _ent_traits(b)
        for tname in set(a_traits) | set(b_traits):
            av = a_traits.get(tname)
            bv = b_traits.get(tname)
            if av != bv:
                diff["changed"]["entities"].append({
                    "id": eid, "field": f"trait:{tname}",
                    "from": av, "to": bv,
                })

    a_evt = {evt.id for evt in ws_a.events}
    b_evt = {evt.id for evt in ws_b.events}
    diff["added"]["events"] = sorted(b_evt - a_evt)
    diff["removed"]["events"] = sorted(a_evt - b_evt)

    a_obj = set(ws_a.objects)
    b_obj = set(ws_b.objects)
    diff["added"]["objects"] = sorted(b_obj - a_obj)
    diff["removed"]["objects"] = sorted(a_obj - b_obj)
    for oid in a_obj & b_obj:
        a, b = ws_a.objects[oid], ws_b.objects[oid]
        if a.location_id != b.location_id:
            diff["changed"]["objects"].append({
                "id": oid, "field": "location_id",
                "from": a.location_id, "to": b.location_id,
            })
        if a.owner_id != b.owner_id:
            diff["changed"]["objects"].append({
                "id": oid, "field": "owner_id",
                "from": a.owner_id, "to": b.owner_id,
            })

    a_wt = set(ws_a.world_traits)
    b_wt = set(ws_b.world_traits)
    diff["added"]["world_traits"] = sorted(b_wt - a_wt)
    diff["removed"]["world_traits"] = sorted(a_wt - b_wt)
    for wid in a_wt & b_wt:
        a, b = ws_a.world_traits[wid], ws_b.world_traits[wid]
        if a.magnitude.value != b.magnitude.value:
            diff["changed"]["world_traits"].append({
                "id": wid, "field": "magnitude",
                "from": float(a.magnitude.value), "to": float(b.magnitude.value),
            })

    def _edge_key(ce: CausalEdge) -> Tuple[str, str, str]:
        return (ce.source_id, ce.target_id, ce.causality_type)

    a_edges = {_edge_key(ce) for ce in ws_a.causal_topology}
    b_edges = {_edge_key(ce) for ce in ws_b.causal_topology}
    diff["added"]["causal_edges"] = sorted(
        f"{s} → {t} ({k})" for (s, t, k) in (b_edges - a_edges)
    )
    diff["removed"]["causal_edges"] = sorted(
        f"{s} → {t} ({k})" for (s, t, k) in (a_edges - b_edges)
    )

    # Channels & beliefs subsystem — surface presence-level deltas so
    # the diff overlay matches the MCP ``diff_versions`` counters.
    a_chans = set(ws_a.channels)
    b_chans = set(ws_b.channels)
    diff["added"]["channels"] = sorted(b_chans - a_chans)
    diff["removed"]["channels"] = sorted(a_chans - b_chans)
    a_utts = {e.id for e in ws_a.events if e.event_type == "utterance"}
    b_utts = {e.id for e in ws_b.events if e.event_type == "utterance"}
    diff["added"]["utterance_events"] = sorted(b_utts - a_utts)
    diff["removed"]["utterance_events"] = sorted(a_utts - b_utts)

    diff["totals"] = {
        "added": sum(len(v) for v in diff["added"].values()),
        "removed": sum(len(v) for v in diff["removed"].values()),
        "changed": sum(len(v) for v in diff["changed"].values()),
    }
    return diff


# =====================================================================
# Hidden channels — channels/utterances not yet on-page for the reader,
# plus per-recipient intelligibility asymmetries
# =====================================================================


def hidden_channel_rows(
    ws: WorldStateV1,
    *,
    syuzhet_anchor: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Project ``DirectiveAssembler.compute_hidden_channels`` into UI rows.

    One row per HiddenChannel. Includes ``unintelligible_for`` so the
    UI can surface recipient-specific intelligibility asymmetry, which
    was previously only visible to MCP ``compute_tension`` callers.
    """
    try:
        from shadow_loom.directive_assembly import DirectiveAssembler
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        from shadow_loom.instantiator import AMWNInstantiator
    except Exception:  # pragma: no cover — import-time breakage only
        logger.exception("Could not import DirectiveAssembler")
        return []

    entity_ids = list(ws.entities.keys())[:6]
    if not entity_ids:
        return []
    try:
        ego = extract_ego_graph_from_memory(ws, entity_ids)
        ego_dict = ego.model_dump()
        sandbox = AMWNInstantiator.create_sandbox(ego_dict, "interrogate")
        assembler = DirectiveAssembler(sandbox, ego_dict, ws)
        anchor = syuzhet_anchor
        if anchor is None:
            anchor = max(
                (e.syuzhet_index for e in ws.events), default=0
            )
        hcs = assembler.compute_hidden_channels(anchor)
    except Exception:
        logger.exception("hidden_channel_rows: tension computation failed")
        return []

    label = _label_fn_for(ws)
    rows: List[Dict[str, Any]] = []
    for hc in hcs:
        rows.append({
            "kind": hc.kind,
            "channel_id": hc.channel_id or "",
            "utterance_event_id": hc.utterance_event_id or "",
            "medium": hc.medium,
            "participants": [
                {"id": p, "label": label(p)} for p in hc.participant_ids
            ],
            "addressees": [
                {"id": p, "label": label(p)} for p in hc.addressee_ids
            ],
            "speaker_id": hc.speaker_id or "",
            "speaker_label": label(hc.speaker_id) if hc.speaker_id else "",
            "discovered_at_syuzhet": hc.discovered_at_syuzhet,
            "unintelligible_for": [
                {"id": p, "label": label(p)} for p in hc.unintelligible_for
            ],
        })
    return rows


# =====================================================================
# Structured response card — for chat's interrogate / general results
# =====================================================================


def structured_response_data(
    physics_result: Dict[str, Any] | None,
    *,
    ws: Optional[WorldStateV1] = None,
) -> Dict[str, Any]:
    """Distill an interrogate/general physics_result into a card.

    Returns::

        {
            "claim":     str,
            "evidence":  [{"node_id", "label", "kind"}],
            "confidence": float (0..1),
            "caveats":   [str],
            "raw":       physics_result (for the Show JSON expansion)
        }
    """
    out: Dict[str, Any] = {
        "claim": "",
        "evidence": [],
        "confidence": 0.0,
        "caveats": [],
        "rule2_redundant_evidence": [],
        "rule3_pruned_interventions": [],
        "raw": physics_result or {},
    }
    if not physics_result:
        out["claim"] = "No reasoning result available."
        return out

    label = _label_fn_for(ws)

    # Claim — auditor/interrogator answer field, falling back to status
    claim = (
        physics_result.get("answer")
        or physics_result.get("claim")
        or physics_result.get("summary")
        or ""
    )
    if not claim:
        # Fallback: synthesise from status + query_type
        qtype = physics_result.get("query_type", "query")
        status = physics_result.get("status", "")
        if status == "implausible":
            claim = (
                f"The {qtype} could not be resolved — "
                f"{physics_result.get('implausibility_reason', 'unspecified')}."
            )
        else:
            claim = f"{qtype.capitalize()} completed."
    out["claim"] = claim

    # Evidence — pathfinding proofs, ego-graph nodes, or focus IDs
    proof = physics_result.get("proof") or physics_result.get("causal_bridges") or []
    if isinstance(proof, list):
        for item in proof[:20]:
            if isinstance(item, dict):
                nid = item.get("id") or item.get("node_id") or ""
                if not nid:
                    continue
                out["evidence"].append({
                    "node_id": nid,
                    "label": label(nid),
                    "kind": item.get("kind", "node"),
                })
            elif isinstance(item, str):
                out["evidence"].append({
                    "node_id": item, "label": label(item), "kind": "node",
                })

    if not out["evidence"]:
        for nid in (physics_result.get("focus_entity_ids") or [])[:10]:
            out["evidence"].append({
                "node_id": nid, "label": label(nid), "kind": "focus",
            })

    # Confidence — derive from evidence_strength roll-up if present
    raw_conf = physics_result.get("confidence")
    if raw_conf is not None:
        try:
            out["confidence"] = max(0.0, min(1.0, float(raw_conf)))
        except (TypeError, ValueError):
            pass
    else:
        # Inferred confidence: more evidence + non-implausible = higher
        if physics_result.get("status") == "implausible":
            out["confidence"] = 0.1
        else:
            out["confidence"] = min(1.0, 0.5 + 0.05 * len(out["evidence"]))

    # Caveats
    if physics_result.get("status") == "implausible":
        reason = physics_result.get("implausibility_reason") or "unspecified"
        out["caveats"].append(f"Engine flagged this as implausible: {reason}.")
    if physics_result.get("status") == "partial":
        out["caveats"].append("Engine returned a partial result — some queries unresolved.")
    if physics_result.get("blocked"):
        out["caveats"].append(
            f"{len(physics_result['blocked'])} propagation(s) were blocked by inertia or affordance gates."
        )

    # Rule-2 (redundant evidence) and Rule-3 (pruned interventions) —
    # surfaced on the chat card so the user understands why some
    # nodes/paths were dropped before audit (Story-integration plan
    # Step 7).
    for nid in (physics_result.get("rule2_redundant_evidence") or [])[:10]:
        out["rule2_redundant_evidence"].append({
            "node_id": nid, "label": label(nid),
        })
    for path in (physics_result.get("rule3_pruned_interventions") or [])[:10]:
        if isinstance(path, dict):
            out["rule3_pruned_interventions"].append({
                "label": path.get("label")
                or path.get("description")
                or " \u2192 ".join(label(n) for n in path.get("nodes", [])[:4])
                or "(pruned path)",
                "reason": path.get("reason") or "",
            })
        elif isinstance(path, str):
            out["rule3_pruned_interventions"].append({
                "label": label(path), "reason": "",
            })

    return out


# =====================================================================
# Event context — everything you need to know about a moment in syuzhet
# =====================================================================


def syuzhet_event_index(ws: WorldStateV1) -> List[Dict[str, Any]]:
    """Return events sorted by syuzhet_index for the narrative-order panel.

    Each row::

        {"id", "syuzhet_index", "fabula_time", "event_type",
         "description", "actor_count", "target_count"}
    """
    rows: List[Dict[str, Any]] = []
    for evt in sorted(ws.events, key=lambda e: (e.syuzhet_index, e.fabula_time)):
        rows.append({
            "id": evt.id,
            "syuzhet_index": int(evt.syuzhet_index),
            "fabula_time": int(evt.fabula_time),
            "event_type": evt.event_type,
            "description": evt.description or "",
            "actor_count": len(evt.actor_ids),
            "target_count": len(evt.target_ids),
        })
    return rows


def event_context_data(
    ws: WorldStateV1,
    event_id: str,
) -> Dict[str, Any]:
    """Build the full clicked-event context dossier.

    Returns a dict with every cross-referenced surface a writer or
    fan-analyst might want when inspecting a moment::

        {
            "event":        EventNode-as-dict (id, fabula_time, syuzhet_index, type, description),
            "actors":       [{"id", "name", "status", "location_id",
                              "traits": {name: {"value", "inertia"}},
                              "beliefs": [Belief-as-dict]}],
            "targets":      [same shape as actors],
            "incoming":     [{"source_id", "source_label", "type",
                              "mechanism", "force", "evidence",
                              "trait_target", "trait_delta"}],
            "outgoing":     [same shape, target side],
            "locations":    [{"id", "name", "description"}],
            "objects":      [{"id", "name", "owner_id", "location_id"}],
            "world_traits_active": [
                {"id", "name", "category", "magnitude": {value, inertia}, "description"}
            ],
            "syuzhet_neighbours": {
                "prev": EventNode-as-dict-or-None,
                "next": EventNode-as-dict-or-None,
            }
        }

    All entity beliefs/traits are reconstructed *at this event's
    fabula_time* using :func:`reconstruct_entity_at` — i.e. you see
    the character as they were *in that moment*, not pre-story.
    """
    evt = next((e for e in ws.events if e.id == event_id), None)
    if evt is None:
        return {}

    label = _label_fn_for(ws)
    ft = int(evt.fabula_time)

    # ── Actors and targets reconstructed at this moment ───────────
    def _entity_dossier(eid: str) -> Optional[Dict[str, Any]]:
        ent = ws.entities.get(eid)
        if ent is None:
            return None
        snap = reconstruct_entity_at(ent, ft)
        return {
            "id": eid,
            "name": ent.name,
            "status": snap["status"],
            "location_id": snap["location_id"],
            "traits": snap["traits"],
            "beliefs": snap["beliefs"],
        }

    actors = [_entity_dossier(eid) for eid in evt.actor_ids]
    actors = [a for a in actors if a is not None]

    targets: List[Dict[str, Any]] = []
    for tid in evt.target_ids:
        if tid in ws.entities:
            d = _entity_dossier(tid)
            if d is not None:
                targets.append({**d, "_kind": "Entity"})
        elif tid in ws.objects:
            obj = ws.objects[tid]
            targets.append({
                "id": tid, "name": obj.name,
                "owner_id": obj.owner_id, "location_id": obj.location_id,
                "_kind": "NarrativeObject",
            })
        elif tid in ws.locations:
            loc = ws.locations[tid]
            targets.append({
                "id": tid, "name": loc.name,
                "description": loc.description,
                "_kind": "Location",
            })

    # ── Causal edges in/out ───────────────────────────────────────
    incoming: List[Dict[str, Any]] = []
    outgoing: List[Dict[str, Any]] = []
    for ce in ws.causal_topology:
        if ce.target_id == event_id:
            incoming.append({
                "source_id": ce.source_id,
                "source_label": label(ce.source_id),
                "type": ce.causality_type,
                "mechanism": ce.mechanism,
                "force": float(ce.causal_force),
                "evidence": ce.evidence_strength,
                "delay": int(ce.propagation_delay),
                "trait_target": ce.trait_target,
                "trait_delta": ce.trait_delta,
            })
        if ce.source_id == event_id:
            outgoing.append({
                "target_id": ce.target_id,
                "target_label": label(ce.target_id),
                "type": ce.causality_type,
                "mechanism": ce.mechanism,
                "force": float(ce.causal_force),
                "evidence": ce.evidence_strength,
                "delay": int(ce.propagation_delay),
                "trait_target": ce.trait_target,
                "trait_delta": ce.trait_delta,
            })
    incoming.sort(key=lambda r: r["force"], reverse=True)
    outgoing.sort(key=lambda r: r["force"], reverse=True)

    # ── Locations involved (actor/target locations at this time) ──
    loc_ids: set[str] = set()
    for a in actors:
        if a.get("location_id"):
            loc_ids.add(a["location_id"])
    for t in targets:
        lid = t.get("location_id") if isinstance(t, dict) else None
        if lid:
            loc_ids.add(lid)
    locations = [
        {
            "id": lid,
            "name": ws.locations[lid].name,
            "description": ws.locations[lid].description,
        }
        for lid in loc_ids if lid in ws.locations
    ]

    # ── Objects: any object whose owner_id is an actor, or whose
    #    location_id matches an involved location, or which appears as
    #    a target_id ──
    objects: List[Dict[str, Any]] = []
    actor_ids = {a["id"] for a in actors}
    object_ids_seen: set[str] = set()
    for tid in evt.target_ids:
        if tid in ws.objects and tid not in object_ids_seen:
            obj = ws.objects[tid]
            objects.append({
                "id": tid, "name": obj.name,
                "owner_id": obj.owner_id, "location_id": obj.location_id,
            })
            object_ids_seen.add(tid)
    for oid, obj in ws.objects.items():
        if oid in object_ids_seen:
            continue
        if (obj.owner_id and obj.owner_id in actor_ids) or (
            obj.location_id and obj.location_id in loc_ids
        ):
            objects.append({
                "id": oid, "name": obj.name,
                "owner_id": obj.owner_id, "location_id": obj.location_id,
            })
            object_ids_seen.add(oid)

    # ── World traits — magnitude reconstructed at this moment ─────
    world_traits_active: List[Dict[str, Any]] = []
    for wid, wt in ws.world_traits.items():
        snap = reconstruct_world_trait_at(wt, ft)
        world_traits_active.append({
            "id": wid,
            "name": wt.name,
            "category": wt.category,
            "magnitude": snap["magnitude"],
            "description": snap.get("description") or wt.description,
        })
    world_traits_active.sort(
        key=lambda r: r["magnitude"]["value"], reverse=True,
    )

    # ── Syuzhet neighbours ────────────────────────────────────────
    ordered = sorted(
        ws.events, key=lambda e: (e.syuzhet_index, e.fabula_time),
    )
    idx = next((i for i, e in enumerate(ordered) if e.id == event_id), None)
    prev_evt = ordered[idx - 1] if (idx is not None and idx > 0) else None
    next_evt = (
        ordered[idx + 1] if (idx is not None and idx + 1 < len(ordered)) else None
    )

    def _evt_min(e: Optional[EventNode]) -> Optional[Dict[str, Any]]:
        if e is None:
            return None
        return {
            "id": e.id,
            "syuzhet_index": int(e.syuzhet_index),
            "fabula_time": int(e.fabula_time),
            "description": (e.description or "")[:80],
        }

    return {
        "event": {
            "id": evt.id,
            "fabula_time": int(evt.fabula_time),
            "syuzhet_index": int(evt.syuzhet_index),
            "event_type": evt.event_type,
            "description": evt.description or "",
            "actor_ids": list(evt.actor_ids),
            "target_ids": list(evt.target_ids),
        },
        "actors": actors,
        "targets": targets,
        "incoming": incoming,
        "outgoing": outgoing,
        "locations": locations,
        "objects": objects,
        "world_traits_active": world_traits_active,
        "syuzhet_neighbours": {
            "prev": _evt_min(prev_evt),
            "next": _evt_min(next_evt),
        },
    }


# =====================================================================
# Internal helpers
# =====================================================================


def _all_node_ids(ws: WorldStateV1) -> set[str]:
    ids: set[str] = set()
    ids.update(ws.entities)
    ids.update(ws.locations)
    ids.update(ws.objects)
    ids.update(ws.world_traits)
    ids.update(evt.id for evt in ws.events)
    return ids


def _node_type(ws: WorldStateV1, nid: str) -> str:
    if nid in ws.entities:
        return "Entity"
    if nid in ws.locations:
        return "Location"
    if nid in ws.objects:
        return "NarrativeObject"
    if nid in ws.world_traits:
        return "WorldTrait"
    if nid in ws.channels:
        return "Channel"
    if any(evt.id == nid for evt in ws.events):
        return "EventNode"
    return "unknown"


def _label_fn_for(ws: Optional[WorldStateV1]):
    """Return a fn that turns a node_id into a human-readable label.

    Falls back to the raw id when ``ws`` is None or the id is unknown.
    """
    if ws is None:
        return lambda nid: str(nid)

    event_map = {evt.id: evt for evt in ws.events}

    def _label(nid: str) -> str:
        if not nid:
            return ""
        if nid in ws.entities:
            return ws.entities[nid].name
        if nid in ws.locations:
            return ws.locations[nid].name
        if nid in ws.objects:
            return ws.objects[nid].name
        if nid in ws.world_traits:
            return ws.world_traits[nid].name
        if nid in ws.channels:
            return ws.channels[nid].name
        if nid in event_map:
            evt = event_map[nid]
            desc = (evt.description or evt.id)[:40]
            return f"t{evt.fabula_time}: {desc}"
        return str(nid)

    return _label
