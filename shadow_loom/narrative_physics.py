# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Dict, Any, List, Optional
import logging

import networkx as nx

from shadow_loom.models import (
    WorldStateV1,
    default_relationship_metrics_dict,
    reconstruct_concern_at,
    reconstruct_entity_at,
)
from shadow_loom.query_models import UserRequest
from shadow_loom.extract_graph import extract_ego_graph_from_memory, extract_full_world_state
from shadow_loom.instantiator import AMWNInstantiator
from shadow_loom.causal_physics import CausalPhysicsEngine, CausalPhysicsResult
from shadow_loom.directive_assembly import DirectiveAssembler
from shadow_loom.settings import get_settings

logger = logging.getLogger(__name__)


def _get_delay_target_ft(
    sandbox: nx.MultiDiGraph,
    target_id: str,
    world_state: WorldStateV1,
) -> float:
    """Resolve the effective fabula_time for a causal edge target.

    Entity targets have no ``fabula_time`` — fall back to the simulation
    horizon (max fabula_time across all events in the world state).
    """
    node_data = sandbox.nodes.get(target_id)
    if node_data:
        ft = node_data.get("fabula_time")
        if ft is not None:
            return ft
    # Fallback: simulation horizon
    if world_state.events:
        return max(e.fabula_time for e in world_state.events)
    return 0


def _check_intervention_plausibility(
    interventions: Dict[str, Any],
    ws: WorldStateV1,
) -> Optional[Dict[str, Any]]:
    """Decide whether a Rung-2/3 intervention set can be applied at all.

    Returns ``None`` when at least one target resolves to a known node
    (or is a ``.spawn`` genesis), meaning the engine has *something* to
    operate on.  Otherwise returns a diagnostics dict describing why the
    request is implausible — used by the pipeline to short-circuit
    generation and avoid mutating the world model.
    """
    if not interventions:
        return {
            "reason": "Empty intervention set — nothing to apply.",
            "unresolved_targets": [],
        }

    valid_ids = (
        set(ws.entities.keys())
        | {e.id for e in ws.events}
        | set(ws.objects.keys())
        | set(ws.locations.keys())
        | set(getattr(ws, "world_traits", {}).keys())
        | set(getattr(ws, "channels", {}).keys())
    )

    unresolved: List[Dict[str, str]] = []
    non_spawn_total = 0
    for key in interventions:
        if "." not in key:
            unresolved.append({"target": key, "reason": "malformed key (missing '.')"})
            non_spawn_total += 1
            continue
        node_id, prop = key.split(".", 1)
        if prop == "spawn":
            continue  # genesis: node doesn't need to exist yet
        non_spawn_total += 1
        if node_id not in valid_ids:
            unresolved.append({"target": key, "reason": f"unknown node id '{node_id}'"})

    # If every non-spawn target failed to resolve, the request is implausible.
    if unresolved and len(unresolved) >= non_spawn_total:
        return {
            "reason": (
                f"None of the {len(interventions)} intervention target(s) "
                "could be resolved against the current world state."
            ),
            "unresolved_targets": unresolved,
        }
    return None


def _check_engine_vacuity(
    physics_result: CausalPhysicsResult,
    *,
    rung: int,
    interventions: Dict[str, Any],
    evidence_node_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Tier-2 implausibility: the engine ran but produced no causal effect.

    A counterfactual or intervention is engine-implausible when *all* of:

      • the do-operator could not bind to any sandbox node (``intervened_nodes`` empty);
      • forward propagation produced no trait mutations;
      • social propagation produced no relationship mutations;
      • (rung 3 only) abduction produced no hidden_deltas.

    This means the requested change has no representable consequences in
    the AMWN — the shadow physics engine can simulate the surgery but the
    world does not actually move.  We surface that as implausible so the
    pipeline can either explain (default) or force generation anyway.
    """
    if physics_result.intervened_nodes:
        return None
    if physics_result.mutations or physics_result.social_mutations:
        return None
    # Typed-surgery mutation lists count as engine effect too — an
    # ``OBJ_X.owner_id`` clamp that produces an :class:`ObjectMutation`
    # has demonstrably moved the world even if no trait propagated
    # (e.g. cyclic SCC absorbed downstream traits).
    if (physics_result.object_mutations
            or physics_result.proposition_mutations
            or physics_result.belief_mutations
            or physics_result.concern_mutations
            or physics_result.world_trait_mutations):
        return None
    if rung == 3 and physics_result.hidden_deltas:
        return None

    blocked_count = len(physics_result.blocked)
    if rung == 3:
        reason = (
            "Rung-3 abduction is impossible: the requested historical "
            "interventions could not bind to any node in the shadow "
            "sandbox, and the present-day evidence produced no latent "
            "deltas, so the counterfactual has no representable effect "
            "on the world."
        )
    else:
        reason = (
            "Rung-2 intervention is impossible: the do-operator could "
            "not bind to any node in the shadow sandbox and propagation "
            "produced no downstream trait or relationship change."
        )
    return {
        "reason": reason,
        "tier": 2,
        "rung": rung,
        "unresolved_targets": [
            {"target": k, "reason": "did not bind to any sandbox node"}
            for k in interventions
        ],
        "evidence_node_ids": list(evidence_node_ids or []),
        "blocked_propagations": blocked_count,
    }


# =====================================================================
# Phase 2 — Sandbox utility-layer preservation
# =====================================================================
def _stamp_utility_layer(
    sandbox: nx.MultiDiGraph,
    global_world_state: WorldStateV1,
    *,
    fabula_anchor: Optional[int] = None,
) -> None:
    """Annotate ``sandbox.graph`` with the audience-side utility layer.

    Phase-2 contract: every Rung-2 / Rung-3 sandbox carries the
    proposition list and a back-pointer to its parent factual world so
    downstream consumers (directive assembler, affect unification,
    renderer) can reconstruct the satisfied/unsatisfied utility delta
    without re-walking the global ``WorldStateV1``.

    Concerns are already serialised on entity nodes by the instantiator
    (they are plain attributes on :class:`Entity`); we only need to
    surface the proposition layer plus the parent back-pointer plus the
    fabula anchor so consumers can resolve ``truth_at_fabula`` lookups
    against the right time slice.
    """
    sandbox.graph["parent_world_id"] = id(global_world_state)
    sandbox.graph["propositions"] = [
        p.model_dump() for p in (global_world_state.propositions or [])
    ]
    if fabula_anchor is not None:
        sandbox.graph["fabula_anchor"] = int(fabula_anchor)


def _load_propositions_from_sandbox(sandbox_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recover the proposition list from a serialised sandbox payload.

    Returns the raw dicts (not :class:`Proposition` instances) so
    consumers that only need ``proposition_id`` / ``truth_at_fabula`` /
    ``referent_ids`` lookups don't pay the model-construction cost.
    Callers that need typed objects can ``Proposition(**d)`` themselves.
    """
    if not isinstance(sandbox_data, dict):
        return []
    graph_attrs = sandbox_data.get("graph") or {}
    if isinstance(graph_attrs, dict):
        props = graph_attrs.get("propositions")
        if isinstance(props, list):
            return list(props)
    # NetworkX serialises graph attrs differently across versions; check
    # the top-level key as a fallback.
    props = sandbox_data.get("propositions")
    return list(props) if isinstance(props, list) else []


def _typed_target_payload(
    request: UserRequest,
    physics_result: Any = None,
) -> Dict[str, Any]:
    """Phase-7 helper: pack the typed Pearl-rung surgery metadata into a
    dict slice safe to merge into the rung-2/rung-3 result.

    Surfaces:
      * ``do_targets`` (rung-2) / ``historical_do_targets`` (rung-3) —
        verbatim model-dumps of the typed :class:`DoTarget` payload
        the parser produced (empty list when the caller used the
        legacy dict surface only).
      * ``proposition_mutations`` / ``belief_mutations`` /
        ``concern_mutations`` — engine-collected per-target diffs
        (Phase-1 collectors; empty when the legacy non-engine path
        ran or no typed surgery touched the utility layer).
      * ``affected_propositions`` / ``affected_beliefs`` /
        ``affected_concerns`` — flat ID lists derived from the
        mutations above so the renderer / answer surface can
        ask "did the surgery touch X?" without iterating the
        structured mutation lists.
    """
    out: Dict[str, Any] = {}

    qt = getattr(request, "query_type", None)
    if qt == "intervention":
        do_targets = list(getattr(request, "do_targets", None) or [])
        out["do_targets"] = [t.model_dump() for t in do_targets]
    elif qt == "counterfactual":
        do_targets = list(getattr(request, "historical_do_targets", None) or [])
        out["historical_do_targets"] = [t.model_dump() for t in do_targets]
    else:
        return out

    pm = list(getattr(physics_result, "proposition_mutations", None) or [])
    bm = list(getattr(physics_result, "belief_mutations", None) or [])
    cm = list(getattr(physics_result, "concern_mutations", None) or [])
    out["proposition_mutations"] = [m.model_dump() for m in pm]
    out["belief_mutations"] = [m.model_dump() for m in bm]
    out["concern_mutations"] = [m.model_dump() for m in cm]

    out["affected_propositions"] = sorted({
        getattr(m, "proposition_id", None) for m in pm
        if getattr(m, "proposition_id", None)
    })
    # Belief mutations key off (holder_id, target_id) — surface as
    # "ENT_X→ENT_Y" so the renderer can phrase epistemic shifts.
    out["affected_beliefs"] = sorted({
        f"{getattr(m, 'holder_id', '?')}\u2192{getattr(m, 'target_id', '?')}"
        for m in bm
    })
    out["affected_concerns"] = sorted({
        getattr(m, "concern_id", None) for m in cm
        if getattr(m, "concern_id", None)
    })
    return out


def calculate_narrative_physics(
    request: UserRequest,
    global_world_state: WorldStateV1,
    temporal_anchor: Optional[int] = None,
    syuzhet_anchor: Optional[int] = None,
    use_causal_engine: bool = False,
) -> Dict[str, Any]:
    """
    Executes structural graph math and topological surgeries.
    Returns the serialized graph state purely in Python dictionaries.

    Parameters
    ----------
    temporal_anchor : int or None
        Fabula-time cutoff for time-slicing (causal physics layer).
    syuzhet_anchor : int or None
        Narrative-position cutoff for reader epistemic state
        (suspense / surprise layer).  Only used when
        ``use_causal_engine=True`` for directive queries.
    """
    # ==========================================
    # RUNG 1: OBSERVATION
    # ==========================================
    if request.query_type == "observation":
        # If no POV specified, fall back to the Omniscient Graph
        if not request.focus_entity_ids:
            logger.info("[Observation] No POV specified — extracting Omniscient Graph")
            full_state = extract_full_world_state(
                global_world_state, temporal_anchor,
                syuzhet_anchor=syuzhet_anchor,
            )
            return {
                "status": "success",
                "query_type": "observation",
                "physics_state": full_state,
                "directives": request.observations,
                # Surface the structured ``observations`` mapping so the
                # pipeline merge bridge can materialise reveals as
                # deterministic EntityUpdate / WorldTraitSnapshot rows
                # instead of relying on prose extraction to recover them.
                "observation_facts": dict(request.observations or {}),
            }

        logger.info("[Observation] Multi-Ego extraction for POV: %s", request.focus_entity_ids)
        ego_graph = extract_ego_graph_from_memory(global_world_state, request.focus_entity_ids, temporal_anchor, syuzhet_anchor=syuzhet_anchor)

        return {
            "status": "success",
            "query_type": "observation",
            "physics_state": ego_graph.model_dump(),
            "directives": request.observations,
            # See above — same bridge contract for the POV path.
            "observation_facts": dict(request.observations or {}),
        }

    # ==========================================
    # RUNG 2: INTERVENTION (do-calculus)
    # ==========================================
    elif request.query_type == "intervention":
        # Typed do-target surface (DoChannel / DoRelationship /
        # DoCausalEdge / DoSpatialEdge / DoBelief / DoConcern /
        # DoProposition / DoWorldTrait / DoEvent / DoTrait). Coexists
        # with the legacy ``request.interventions`` dict; both
        # contribute to focus / shadow-path seeds so the ego graph
        # admits the full causal lineage of every surgery regardless
        # of which surface the user used.
        typed_do_targets = list(getattr(request, "do_targets", None) or [])

        # --- Plausibility gate ---
        # Plausibility is checked against a union of the legacy dict
        # and the legacy-equivalent lift of the typed do-targets so
        # typed-only requests (no ``request.interventions`` dict at
        # all) are not falsely flagged as empty. The four edge-typed
        # surgeries plus the affect-layer surgeries have no legacy
        # representation; their feasibility is enforced inside
        # :meth:`CausalPhysicsEngine.apply_do_targets` (which logs and
        # skips on missing endpoints) so plausibility short-circuiting
        # them here would forbid valid surgeries.
        plausibility_view = dict(request.interventions or {})
        plausibility_view.update(_lift_do_targets_to_legacy_dict(typed_do_targets))
        # When typed-only and no DoEvent/DoTrait lifts produced any
        # legacy keys, synthesise a sentinel ``.spawn`` key per typed
        # target so the empty-dict short-circuit doesn't fire.
        if not plausibility_view and typed_do_targets:
            plausibility_view = {f"TYPED_{i}.spawn": True for i in range(len(typed_do_targets))}
        bad = _check_intervention_plausibility(plausibility_view, global_world_state)
        if bad is not None:
            if not getattr(request, "force_implausible", False):
                logger.warning("[Intervention] Implausible request: %s", bad["reason"])
                return {
                    "status": "implausible",
                    "query_type": "intervention",
                    "physics_state": {},
                    "implausibility_reason": bad["reason"],
                    "implausibility_details": bad,
                }
            logger.warning(
                "[Intervention] Forced past implausibility gate: %s", bad["reason"]
            )
            _forced_warning = bad
        else:
            _forced_warning = None

        # Collect ALL affected entities from the intervention keys
        focus_ids = _resolve_focus_entities(request.interventions, global_world_state)
        if typed_do_targets:
            for eid in _resolve_focus_from_do_targets(typed_do_targets, global_world_state):
                if eid not in focus_ids:
                    focus_ids.append(eid)
        logger.info(
            "[Intervention] Resolved focus entities: %s from %d legacy "
            "interventions + %d typed do_targets",
            focus_ids, len(request.interventions or {}), len(typed_do_targets),
        )

        # Shadow-path seeds: every node id named on the LHS of an
        # intervention key (entity, event, world-trait, etc.) plus
        # every node referenced by a typed do-target. Lets the
        # ego-graph extractor walk the reverse causal graph and admit
        # upstream lineage that lives outside the recency window.
        intervention_seeds = {
            k.split(".", 1)[0] for k in (request.interventions or {}) if k
        }
        intervention_seeds |= _do_target_seed_ids(typed_do_targets)

        # 1. Time-Slice
        ego_graph = extract_ego_graph_from_memory(
            global_world_state, focus_ids, temporal_anchor,
            syuzhet_anchor=syuzhet_anchor,
            shadow_path_seed_ids=intervention_seeds,
        )

        # 2. Build Sandbox & Apply Math
        shadow_graph = AMWNInstantiator.create_sandbox(ego_graph.model_dump(), "intervention")
        # Phase-2: surface the audience-side utility layer (propositions +
        # parent-world back-pointer + fabula anchor) so the engine, the
        # directive assembler and the renderer can read it back without
        # re-walking the global world state.
        _stamp_utility_layer(
            shadow_graph, global_world_state, fabula_anchor=temporal_anchor,
        )
        logger.info("[Intervention] Sandbox built — %d nodes, %d edges. Applying surgeries.",
                     shadow_graph.number_of_nodes(), shadow_graph.number_of_edges())

        # Typed do-targets only land via the engine path (legacy
        # ``execute_interventions`` consumes only the string-keyed
        # dict). Force the engine on when typed surgeries are present
        # so callers using the default legacy path still see the
        # typed surface bind through.
        if typed_do_targets:
            use_causal_engine = True

        if use_causal_engine:
            engine = CausalPhysicsEngine(shadow_graph, global_world_state)
            # Apply typed do-targets first so the four edge surgeries
            # (Channel/Relationship/CausalEdge/SpatialEdge) and the
            # affect-layer surgeries (Belief/Concern/Proposition/
            # WorldTrait) reach the sandbox. ``apply_do_targets``
            # lifts DoEvent/DoTrait into the legacy keyspace and
            # stashes them on ``_last_legacy_interventions`` so we
            # can union them into the dict ``execute`` consumes for
            # CTF preflight, plausibility, and provenance pruning.
            interventions_for_execute = dict(request.interventions or {})
            if typed_do_targets:
                engine.apply_do_targets(typed_do_targets)
                interventions_for_execute = {
                    **interventions_for_execute,
                    **(getattr(engine, "_last_legacy_interventions", {}) or {}),
                }
            physics_result = engine.execute(
                rung=2,
                interventions=interventions_for_execute,
                target_node_ids=getattr(request, "target_node_ids", None) or [],
            )

            # Tier-2 vacuity check (after engine ran)
            # Skipped only when typed do-targets include edge surgeries
            # (DoChannel/DoRelationship/DoCausalEdge/DoSpatialEdge) or
            # affect-layer surgeries (DoBelief/DoConcern) — those
            # land via ``apply_do_targets`` without producing a
            # node-level mutation the vacuity check would detect.
            # Typed targets that round-trip via
            # :func:`_lift_do_targets_to_legacy_dict` (DoEvent / DoTrait
            # / DoProposition / DoWorldTrait / DoNarrativeObject) are
            # indistinguishable from legacy dict surgeries; vacuity
            # MUST still fire on those so the engine cannot silently
            # swallow a no-op.
            #
            # Additional guard: even when an edge-surgery target is
            # present, only skip vacuity if at least one edge surgery
            # actually mutated the sandbox (counter on the engine).
            # Otherwise an edge surgery that early-returned (missing
            # endpoints, invalid payload) would silently disable
            # vacuity for the whole query.
            if (
                _typed_targets_require_vacuity_skip(typed_do_targets)
                and getattr(engine, "_edge_do_targets_applied", 0) > 0
            ):
                vacuous = None
            else:
                vacuous = _check_engine_vacuity(
                    physics_result, rung=2,
                    interventions=interventions_for_execute,
                )
            if vacuous is not None and not getattr(request, "force_implausible", False):
                logger.warning("[Intervention] Engine vacuity: %s", vacuous["reason"])
                return {
                    "status": "implausible",
                    "query_type": "intervention",
                    "physics_state": physics_result.sandbox_data,
                    "implausibility_reason": vacuous["reason"],
                    "implausibility_details": vacuous,
                }
            if vacuous is not None:
                logger.warning(
                    "[Intervention] Forced past engine vacuity: %s",
                    vacuous["reason"],
                )
                _forced_warning = vacuous

            result = {
                "status": "success",
                "query_type": "intervention",
                "physics_state": physics_result.sandbox_data,
                "math_changes": request.interventions,
                "mutations": [m.model_dump() for m in physics_result.mutations],
                "social_mutations": [m.model_dump() for m in physics_result.social_mutations],
                "blocked": [b.model_dump() for b in physics_result.blocked],
                "intervened_nodes": physics_result.intervened_nodes,
                # ctf-calculus pre-flight (Correa & Bareinboim 2025).
                # Surfaced in JSON so UI / MCP / auditor can render the
                # interventions Rule 3 dropped and the evidence Rule 2
                # flagged as d-separated.
                "rule3_pruned_interventions": list(
                    physics_result.rule3_pruned_interventions
                ),
                "rule3_pruning_mode": physics_result.rule3_pruning_mode,
                "rule2_redundant_evidence": list(
                    physics_result.rule2_redundant_evidence
                ),
                # Channels & beliefs subsystem: surface what the
                # do-surgery epistemically removed so the auditor and UI
                # can reason about belief / utterance side-effects.
                "pruned_beliefs_count": physics_result.pruned_beliefs_count,
                "pruned_utterance_event_ids": list(
                    physics_result.pruned_utterance_event_ids
                ),
                "disabled_channel_ids": list(
                    physics_result.disabled_channel_ids
                ),
                # Skipped-intervention ledger: targets the engine
                # could not apply because the named node was absent
                # from the sandbox. Surfaced into the brief so the
                # renderer and the auditor know the intervention did
                # NOT land.
                "skipped_interventions": list(
                    physics_result.skipped_interventions
                ),
                # Inert-intervention disclosure: when every requested
                # do-target was Rule-3 pruned / cycle-absorbed the
                # engine sets these so the brief, auditor, and UI can
                # avoid fabricating consequences for a no-op surgery.
                "intervention_inert": physics_result.intervention_inert,
                "intervention_inert_reason": physics_result.intervention_inert_reason,
                # Typed object stashed under a private key so the pipeline
                # can forward it to the auditor (which needs the full
                # CausalPhysicsResult, not the JSON-serialised slices).
                # Filtered out of PipelineHistory extras at the call site.
                "_causal_physics_result": physics_result,
            }
        else:
            AMWNInstantiator.execute_interventions(shadow_graph, request.interventions)
            # Provenance-prune parity with the engine path: drop beliefs
            # whose acquired_via_* points at events/channels the surgery
            # epistemically invalidated, and tag the affected utterance
            # nodes ``pruned=True`` so the legacy cascades skip their
            # downstream causal contributions.
            _legacy_evt_pruned, _legacy_chn_pruned = (
                _collect_legacy_provenance_invalidations(request.interventions)
            )
            # Also pick up events that ``_enforce_affordance_gates``
            # marked ``pruned=True`` during the intervention sweep so
            # beliefs acquired from gate-blocked events get pruned too.
            for _nid, _ndata in shadow_graph.nodes(data=True):
                if _ndata.get("node_type") == "EventNode" and _ndata.get("pruned") is True:
                    _legacy_evt_pruned.add(_nid)
            if _legacy_evt_pruned or _legacy_chn_pruned:
                AMWNInstantiator._prune_beliefs_by_provenance(
                    shadow_graph,
                    removed_event_ids=_legacy_evt_pruned,
                    removed_channel_ids=_legacy_chn_pruned,
                )
                for _nid, _ndata in shadow_graph.nodes(data=True):
                    if _nid in _legacy_evt_pruned:
                        _ndata["pruned"] = True
                    elif (
                        _ndata.get("event_type") == "utterance"
                        and _ndata.get("via_channel_id") in _legacy_chn_pruned
                    ):
                        _ndata["pruned"] = True
                logger.info(
                    "[Intervention·Legacy] Provenance prune: %d event(s), "
                    "%d channel(s) invalidated.",
                    len(_legacy_evt_pruned), len(_legacy_chn_pruned),
                )
            logger.info("[Intervention] Surgeries complete — %d nodes, %d edges.",
                         shadow_graph.number_of_nodes(), shadow_graph.number_of_edges())
            result = {
                "status": "success",
                "query_type": "intervention",
                "physics_state": nx.node_link_data(shadow_graph),
                "math_changes": request.interventions
            }

        # Semantic Physics Override for split-screen (multi-room) scenarios
        physics_override = _generate_physics_override(shadow_graph)
        if physics_override:
            result["physics_override"] = physics_override

        if _forced_warning is not None:
            result["implausibility_warning"] = _forced_warning["reason"]
            result["implausibility_details"] = _forced_warning

        # Phase-7: surface typed Pearl-rung surgery metadata.
        result.update(_typed_target_payload(
            request,
            physics_result=result.get("_causal_physics_result"),
        ))

        return result

    # ==========================================
    # RUNG 3: COUNTERFACTUAL
    # ==========================================
    elif request.query_type == "counterfactual":
        # Typed historical do-target surface — see the rung-2 branch
        # for the rationale; same union semantics apply at rung 3.
        typed_historical_do_targets = list(
            getattr(request, "historical_do_targets", None) or []
        )

        # --- Plausibility gate ---
        plausibility_view = dict(request.historical_interventions or {})
        plausibility_view.update(
            _lift_do_targets_to_legacy_dict(typed_historical_do_targets)
        )
        if not plausibility_view and typed_historical_do_targets:
            plausibility_view = {
                f"TYPED_{i}.spawn": True
                for i in range(len(typed_historical_do_targets))
            }
        bad = _check_intervention_plausibility(plausibility_view, global_world_state)
        forced = getattr(request, "force_implausible", False)
        if bad is not None and not forced:
            logger.warning("[Counterfactual] Implausible request: %s", bad["reason"])
            return {
                "status": "implausible",
                "query_type": "counterfactual",
                "physics_state": {},
                "implausibility_reason": bad["reason"],
                "implausibility_details": bad,
            }
        _forced_warning: Optional[Dict[str, Any]] = bad if (bad is not None and forced) else None
        if _forced_warning is not None:
            logger.warning(
                "[Counterfactual] Forced past implausibility gate: %s",
                _forced_warning["reason"],
            )

        # Determine the historical anchor point from the requested interventions
        try:
            past_anchor = _calculate_past_anchor(plausibility_view, global_world_state)
        except ValueError as exc:
            # Typed-only request whose targets have no legacy
            # representation (e.g. DoChannel, DoRelationship,
            # DoCausalEdge, DoSpatialEdge): use the typed surface
            # to derive an anchor before reporting paradox.
            anchor_seeds = _do_target_seed_ids(typed_historical_do_targets)
            anchor_times: List[int] = []
            for seed in anchor_seeds:
                evt = next((e for e in global_world_state.events if e.id == seed), None)
                if evt:
                    anchor_times.append(evt.fabula_time)
                    continue
                rel = [
                    e.fabula_time for e in global_world_state.events
                    if seed in (e.actor_ids or []) or seed in (e.target_ids or [])
                ]
                if rel:
                    anchor_times.append(max(rel))
            if anchor_times:
                past_anchor = int(min(anchor_times))
            elif not forced:
                logger.warning("[Counterfactual] Temporal paradox: %s", exc)
                return {
                    "status": "implausible",
                    "query_type": "counterfactual",
                    "physics_state": {},
                    "implausibility_reason": str(exc),
                    "implausibility_details": {
                        "reason": str(exc),
                        "unresolved_targets": [
                            {"target": k, "reason": "no historical anchor (target absent from event timeline)"}
                            for k in request.historical_interventions
                        ],
                    },
                }
            else:
                # Forced: fall back to the simulation horizon (current 'now')
                past_anchor = int(
                    max((e.fabula_time for e in global_world_state.events), default=0)
                )
                logger.warning(
                    "[Counterfactual] Forced past temporal paradox — anchoring at horizon %d",
                    past_anchor,
                )
                _forced_warning = _forced_warning or {
                    "reason": str(exc),
                    "unresolved_targets": [
                        {"target": k, "reason": "no historical anchor"}
                        for k in request.historical_interventions
                    ],
                }

        focus_ids = _resolve_focus_entities(request.historical_interventions, global_world_state)
        if typed_historical_do_targets:
            for eid in _resolve_focus_from_do_targets(
                typed_historical_do_targets, global_world_state
            ):
                if eid not in focus_ids:
                    focus_ids.append(eid)
        logger.info("[Counterfactual] Point of Divergence: T=%d | Focus: %s", past_anchor, focus_ids)
        # Shadow-path seeds: counterfactual surgery LHS ids plus the
        # rung-3 evidence conditions. Both must be reachable from the
        # sandbox so abduction and twin-network surgery have the full
        # causal lineage to reason over.
        ctf_seeds: set = {
            k.split(".", 1)[0]
            for k in (request.historical_interventions or {}) if k
        }
        ctf_seeds.update(eid for eid in (request.evidence_node_ids or []) if eid)
        ctf_seeds |= _do_target_seed_ids(typed_historical_do_targets)
        ego_graph = extract_ego_graph_from_memory(
            global_world_state, focus_ids, past_anchor,
            syuzhet_anchor=syuzhet_anchor,
            shadow_path_seed_ids=ctf_seeds,
        )
        shadow_graph = AMWNInstantiator.create_sandbox(ego_graph.model_dump(), "counterfactual")
        # Phase-2: surface the audience-side utility layer onto the
        # historical sandbox; ``past_anchor`` is the Point of Divergence
        # so consumers reading ``truth_at_fabula`` resolve against the
        # right slice.
        _stamp_utility_layer(
            shadow_graph, global_world_state, fabula_anchor=past_anchor,
        )
        logger.info("[Counterfactual] Historical sandbox built — %d nodes. Applying surgeries.",
                     shadow_graph.number_of_nodes())

        # Force engine path when typed historical surgeries are
        # supplied; see the rung-2 branch for rationale.
        if typed_historical_do_targets:
            use_causal_engine = True

        if use_causal_engine:
            engine = CausalPhysicsEngine(shadow_graph, global_world_state)
            historical_for_execute = dict(request.historical_interventions or {})
            if typed_historical_do_targets:
                engine.apply_do_targets(typed_historical_do_targets)
                historical_for_execute = {
                    **historical_for_execute,
                    **(getattr(engine, "_last_legacy_interventions", {}) or {}),
                }
            physics_result = engine.execute(
                rung=3,
                interventions=historical_for_execute,
                evidence_node_ids=request.evidence_node_ids,
                target_node_ids=getattr(request, "target_node_ids", None) or [],
            )

            # Tier-2 vacuity check: did the abductive simulation produce anything?
            # Skipped when typed historical surgeries ran; see rung-2 rationale.
            # Additionally require that at least one edge surgery actually
            # applied so an early-returning edge target doesn't disable
            # vacuity for a complete no-op counterfactual.
            if (
                _typed_targets_require_vacuity_skip(typed_historical_do_targets)
                and getattr(engine, "_edge_do_targets_applied", 0) > 0
            ):
                vacuous = None
            else:
                vacuous = _check_engine_vacuity(
                    physics_result, rung=3,
                    interventions=historical_for_execute,
                    evidence_node_ids=request.evidence_node_ids,
                )
            if vacuous is not None and not forced:
                logger.warning("[Counterfactual] Engine vacuity: %s", vacuous["reason"])
                return {
                    "status": "implausible",
                    "query_type": "counterfactual",
                    "physics_state": physics_result.sandbox_data,
                    "implausibility_reason": vacuous["reason"],
                    "implausibility_details": vacuous,
                }
            if vacuous is not None:
                logger.warning(
                    "[Counterfactual] Forced past engine vacuity: %s",
                    vacuous["reason"],
                )
                _forced_warning = _forced_warning or vacuous

            result = {
                "status": "success",
                "query_type": "counterfactual",
                "physics_state": physics_result.sandbox_data,
                "math_changes": request.historical_interventions,
                "evidence_conditions": request.evidence_node_ids,
                "mutations": [m.model_dump() for m in physics_result.mutations],
                "social_mutations": [m.model_dump() for m in physics_result.social_mutations],
                "blocked": [b.model_dump() for b in physics_result.blocked],
                "hidden_deltas": physics_result.hidden_deltas,
                # ctf-calculus pre-flight (Correa & Bareinboim 2025) — see
                # intervention branch for rationale.
                "rule3_pruned_interventions": list(
                    physics_result.rule3_pruned_interventions
                ),
                "rule3_pruning_mode": physics_result.rule3_pruning_mode,
                "rule2_redundant_evidence": list(
                    physics_result.rule2_redundant_evidence
                ),
                # Channels & beliefs subsystem: counterfactual surgeries
                # on past utterances / channels propagate as belief
                # provenance pruning. Surface the totals so downstream
                # consumers can render "by removing this utterance N
                # downstream beliefs evaporate" diagnostics.
                "pruned_beliefs_count": physics_result.pruned_beliefs_count,
                "pruned_utterance_event_ids": list(
                    physics_result.pruned_utterance_event_ids
                ),
                "disabled_channel_ids": list(
                    physics_result.disabled_channel_ids
                ),
                "skipped_interventions": list(
                    physics_result.skipped_interventions
                ),
                # Inert-intervention disclosure (parity with the
                # intervention branch above) — surfaces a no-op CTF
                # so the brief / auditor / UI do not fabricate
                # downstream consequences.
                "intervention_inert": physics_result.intervention_inert,
                "intervention_inert_reason": physics_result.intervention_inert_reason,
                # Typed object stashed for the pipeline → auditor handoff;
                # see the intervention branch for rationale.
                "_causal_physics_result": physics_result,
            }
        else:
            # --- ABDUCTION STEP: Update hidden variables from present evidence ---
            _apply_abduction(shadow_graph, request.evidence_node_ids, global_world_state)

            AMWNInstantiator.execute_interventions(shadow_graph, request.historical_interventions)

            # Provenance-prune parity with the engine path — see the
            # intervention branch comment above for rationale.
            _ctf_evt_pruned, _ctf_chn_pruned = (
                _collect_legacy_provenance_invalidations(request.historical_interventions)
            )
            # Pick up events ``_enforce_affordance_gates`` blocked
            # during the historical-intervention sweep so beliefs
            # acquired from gate-blocked events are pruned too.
            for _nid, _ndata in shadow_graph.nodes(data=True):
                if _ndata.get("node_type") == "EventNode" and _ndata.get("pruned") is True:
                    _ctf_evt_pruned.add(_nid)
            if _ctf_evt_pruned or _ctf_chn_pruned:
                AMWNInstantiator._prune_beliefs_by_provenance(
                    shadow_graph,
                    removed_event_ids=_ctf_evt_pruned,
                    removed_channel_ids=_ctf_chn_pruned,
                )
                for _nid, _ndata in shadow_graph.nodes(data=True):
                    if _nid in _ctf_evt_pruned:
                        _ndata["pruned"] = True
                    elif (
                        _ndata.get("event_type") == "utterance"
                        and _ndata.get("via_channel_id") in _ctf_chn_pruned
                    ):
                        _ndata["pruned"] = True
                logger.info(
                    "[Counterfactual·Legacy] Provenance prune: %d event(s), "
                    "%d channel(s) invalidated.",
                    len(_ctf_evt_pruned), len(_ctf_chn_pruned),
                )

            # --- PREDICTION STEP: Forward cascade through causal topology ---
            _apply_forward_cascade(shadow_graph, global_world_state)

            # --- SOCIAL PREDICTION: Propagate mutation_social edges ---
            _apply_social_cascade(shadow_graph, global_world_state)

            result = {
                "status": "success",
                "query_type": "counterfactual",
                "physics_state": nx.node_link_data(shadow_graph),
                "math_changes": request.historical_interventions,
                "evidence_conditions": request.evidence_node_ids
            }

        physics_override = _generate_physics_override(shadow_graph)
        if physics_override:
            result["physics_override"] = physics_override

        # Surface the Point-of-Divergence so the pipeline can stamp
        # rung-3 hidden_deltas snapshots at the actual historical
        # anchor instead of falling back to the global event-timeline
        # minimum (which would write abducted state at the start of
        # the story).
        result["past_anchor"] = int(past_anchor)

        if _forced_warning is not None:
            result["implausibility_warning"] = _forced_warning["reason"]
            result["implausibility_details"] = _forced_warning

        # Phase-7: surface typed Pearl-rung surgery metadata.
        result.update(_typed_target_payload(
            request,
            physics_result=result.get("_causal_physics_result"),
        ))

        return result

    # ==========================================
    # SEMANTIC: DIRECTIVE
    # ==========================================
    elif request.query_type == "directive":
        logger.info("[Directive] Target entities: %s | Effect: %s | Intensity: %.2f",
                     request.target_entity_ids, request.target_effect, request.intensity)

        # --- Plausibility gate ---
        unknown = [eid for eid in request.target_entity_ids if eid not in global_world_state.entities]
        directive_warning: Optional[Dict[str, Any]] = None
        target_entity_ids = list(request.target_entity_ids)
        if request.target_entity_ids and len(unknown) == len(request.target_entity_ids):
            if not getattr(request, "force_implausible", False):
                logger.warning("[Directive] Implausible: no target entities exist: %s", unknown)
                return {
                    "status": "implausible",
                    "query_type": "directive",
                    "physics_state": {},
                    "implausibility_reason": (
                        "None of the directive target entities exist in the current world state."
                    ),
                    "implausibility_details": {
                        "reason": "unknown target entities",
                        "unresolved_targets": [
                            {"target": eid, "reason": "unknown entity id"} for eid in unknown
                        ],
                    },
                }
            # Forced: fall back to the first known entity (if any) as POV.
            fallback = next(iter(global_world_state.entities), None)
            directive_warning = {
                "reason": (
                    "None of the directive target entities exist in the current "
                    "world state."
                ),
                "unresolved_targets": [
                    {"target": eid, "reason": "unknown entity id"} for eid in unknown
                ],
                "fallback_pov": fallback,
            }
            target_entity_ids = [fallback] if fallback else []
            logger.warning(
                "[Directive] Forced past implausibility gate \u2014 fallback POV %s",
                fallback,
            )

        # Directive briefs need the same shadow-path coverage as Pearl-
        # rung sandboxes so DirectiveAssembler's epistemic-gap /
        # tension / trajectory computations see the upstream lineage of
        # the target entities, not just the most recent N events.
        ego_graph = extract_ego_graph_from_memory(
            global_world_state, target_entity_ids, temporal_anchor,
            syuzhet_anchor=syuzhet_anchor,
            shadow_path_seed_ids=set(target_entity_ids),
        )
        ego_dump = ego_graph.model_dump()

        if use_causal_engine:
            assembler = DirectiveAssembler(
                sandbox=None, ego_payload=ego_dump, world_state=global_world_state,
            )
            brief = assembler.assemble(request, syuzhet_anchor=syuzhet_anchor)
            result = {
                "status": "success",
                "query_type": "directive",
                "physics_state": ego_dump,
                "creative_brief": brief.model_dump(),
                "target_effect": request.target_effect,
            }
            if directive_warning is not None:
                result["implausibility_warning"] = directive_warning["reason"]
                result["implausibility_details"] = directive_warning
            return result

        injection_rules = _generate_directive_rules(
            ego_dump,
            request.target_vector_id,
            request.intensity
        )

        result = {
            "status": "success",
            "query_type": "directive",
            "physics_state": ego_dump,
            "directives": injection_rules,
            "target_effect": request.target_effect
        }
        if directive_warning is not None:
            result["implausibility_warning"] = directive_warning["reason"]
            result["implausibility_details"] = directive_warning
        return result

    # ==========================================
    # GRAPH RAG: INTERROGATION
    # ==========================================
    elif request.query_type == "interrogate":
        logger.info("[Interrogation] Extracting Omniscient Graph for question: %s", request.question[:80])
        full_state = extract_full_world_state(
            global_world_state, temporal_anchor,
            syuzhet_anchor=syuzhet_anchor,
        )

        return {
            "status": "success",
            "query_type": "interrogate",
            "physics_state": full_state,
            "question": request.question,
            "require_proof": request.require_proof
        }

    # ==========================================
    # FULL-GRAPH Q&A: GENERAL QUESTION
    # ==========================================
    elif request.query_type == "general":
        logger.info("[General] Full-graph Q&A for question: %s", request.question[:80])
        full_state = extract_full_world_state(
            global_world_state, temporal_anchor,
            syuzhet_anchor=syuzhet_anchor,
        )

        return {
            "status": "success",
            "query_type": "general",
            "physics_state": full_state,
            "question": request.question,
            "include_topology": request.include_topology,
        }

    # ==========================================
    # MANUAL EDIT (user-authored prose — no simulation)
    # ==========================================
    elif request.query_type == "manual_edit":
        logger.info("[ManualEdit] User-authored prose (%d chars)", len(request.edited_prose))
        full_state = extract_full_world_state(
            global_world_state, temporal_anchor,
            syuzhet_anchor=syuzhet_anchor,
        )

        return {
            "status": "manual_edit",
            "query_type": "manual_edit",
            "physics_state": full_state,
            "edited_prose": request.edited_prose,
        }

    # ==========================================
    # EVALUATION (full-story quality audit)
    # ==========================================
    elif request.query_type == "evaluate":
        logger.info("[Evaluate] Full-story evaluation requested")
        full_state = extract_full_world_state(
            global_world_state, temporal_anchor,
            syuzhet_anchor=syuzhet_anchor,
        )

        return {
            "status": "success",
            "query_type": "evaluate",
            "physics_state": full_state,
        }

    # Fallback
    raise ValueError(f"Unknown Query Type: {request.query_type}")

# ==========================================
# HELPER 0: RESOLVE ALL FOCUS ENTITIES FROM INTERVENTIONS
# ==========================================
def _resolve_focus_entities(interventions: Dict[str, Any], global_world_state: WorldStateV1) -> List[str]:
    """
    Given intervention keys like 'EVT_DUNCAN_MURDER.event_type' or 'ENT_MACBETH.status',
    collect ALL affected entities so the ego-graph is the union of their rooms.
    Genesis (.spawn) keys are skipped since their nodes don't exist yet.

    For event references we expand to *every* actor AND every target on the
    event (not just the first), so multi-actor events (e.g. a duel, a joint
    decision) and victim-bearing events (e.g. a murder where actor !=
    victim) all surface in the ego-graph. Without this an event like
    ``EVT_KEN_KILLS_DOGS`` would only seed ``ENT_KEN`` even though the
    dogs and their owner are equally part of the scene.
    """
    seen: set = set()
    focus_ids: List[str] = []

    def _add(eid: Optional[str]) -> None:
        if not eid:
            return
        if eid in global_world_state.entities and eid not in seen:
            seen.add(eid)
            focus_ids.append(eid)

    for target_path in interventions:
        if '.' not in target_path:
            continue
        node_id, prop = target_path.split('.', 1)
        if prop == "spawn":
            continue

        # Direct entity reference
        if node_id in global_world_state.entities:
            _add(node_id)
        else:
            # Event reference — pull in every actor and target so the
            # ego-graph isn't truncated to a single participant.
            event = next((e for e in global_world_state.events if e.id == node_id), None)
            if event is not None:
                for _aid in event.actor_ids or []:
                    _add(_aid)
                for _tid in event.target_ids or []:
                    _add(_tid)
                # Utterance speaker / addressees carry the same kind of
                # multi-participant footprint.
                _add(getattr(event, "speaker_id", None))
                for _adid in getattr(event, "addressee_ids", []) or []:
                    _add(_adid)
            else:
                # Object reference — use the object's owner
                obj = global_world_state.objects.get(node_id)
                if obj and obj.owner_id and obj.owner_id in global_world_state.entities:
                    _add(obj.owner_id)

        # Also resolve comms targets so their rooms are in the ego-graph
        if prop == "communicating_with":
            value = interventions[target_path]
            if isinstance(value, list):
                for tgt_id in value:
                    _add(tgt_id)

    # Last resort: first entity in the world state
    if not focus_ids and global_world_state.entities:
        focus_ids.append(next(iter(global_world_state.entities)))

    return focus_ids

# ==========================================
# HELPER 1: THE TIME MACHINE
# ==========================================
def _calculate_past_anchor(interventions: Dict[str, Any], global_world_state: WorldStateV1) -> int:
    """
    Finds the exact 'Point of Divergence' (fabula_time) for a Rung 3 Counterfactual.
    If multiple historical interventions are provided, it finds the oldest one,
    ensuring the simulation rolls back far enough to capture all cascading effects.
    """
    earliest_time = float('inf')
    
    for target_path in interventions.keys():
        # e.g., "EVT_MURDER_1.event_type" -> node_id = "EVT_MURDER_1"
        if '.' not in target_path:
            continue
        node_id, prop = target_path.split('.', 1)

        # Genesis spawns have no historical footprint — skip them
        if prop == "spawn":
            continue
        
        # Scenario A: The intervention explicitly targets a past EventNode
        event = next((evt for evt in global_world_state.events if evt.id == node_id), None)
        if event:
            earliest_time = min(earliest_time, event.fabula_time)
            continue
            
        # Scenario B: The intervention targets a Noun (Entity/Object) in the past
        # e.g., "OBJ_DAGGER.owner_id": "ENT_MACBETH"
        # We must find the last time this object was interacted with to anchor the timeline.
        relevant_events = [
            evt for evt in global_world_state.events 
            if node_id in evt.actor_ids or node_id in evt.target_ids or (evt.description and node_id in evt.description)
        ]
        if relevant_events:
            # Find the most recent event involving this noun
            latest_relevant_time = max(evt.fabula_time for evt in relevant_events)
            earliest_time = min(earliest_time, latest_relevant_time)

    if earliest_time == float('inf'):
        raise ValueError(
            f"Temporal Paradox: Could not find a historical anchor for interventions: {interventions}"
        )

    logger.debug("[PastAnchor] earliest_time=%d from %d intervention keys", earliest_time, len(interventions))
    return int(earliest_time)

# ==========================================
# HELPER 2: THE SEMANTIC COMPILER
# ==========================================
def _generate_directive_rules(ego_graph: Dict[str, Any], vector_target_id: str, shift: float) -> str:
    """
    Translates cold graph math into a "Creative Brief" for the drafting LLM.
    Calculates the current mathematical state of a trait or relationship and 
    instructs the LLM exactly how hard to push the narrative to achieve the target shift.
    """
    if not vector_target_id:
        # Fallback if no specific vector is targeted, just a general concept
        return f"NARRATIVE DIRECTIVE: Shift the emotional polarity of the scene by {shift:+.2f}."

    # Parse the target: e.g., "ENT_MACBETH.traits.ambition" -> "ENT_MACBETH", "traits.ambition"
    if '.' not in vector_target_id:
        return f"Alter the state of {vector_target_id} by a trajectory of {shift:+.2f}."
    node_id, vector_path = vector_target_id.split('.', 1)
    
    # 1. SOCIAL DIRECTIVE (Relationships)
    if vector_path.startswith("relationships."):
        # e.g., "relationships.ENT_DUNCAN.affinity"
        parts = vector_path.split('.')
        if len(parts) != 3:
            return f"Alter the state of {vector_target_id} by a trajectory of {shift:+.2f}."
        _, target_entity, metric = parts
        
        # Locate the current metric in the localized graph. Read from the
        # per-axis ``metrics`` dict when present so the directive can
        # convey the per-axis ``evidence_strength`` to the rendering LLM
        # ("affinity currently sits at 0.2 — strong evidence" vs "0.2 —
        # weakly inferred"). Falls back to the legacy flat key only when
        # the per-axis state is unavailable.
        current_val: Any = "unknown"
        evidence_qualifier = ""
        for rel in ego_graph.get("relevant_relationships", []):
            if rel.get("source_entity_id") == node_id and rel.get("target_entity_id") == target_entity:
                rel_metrics = rel.get("metrics") or {}
                axis_state = rel_metrics.get(metric) if isinstance(rel_metrics, dict) else None
                if isinstance(axis_state, dict) and "value" in axis_state:
                    current_val = axis_state["value"]
                    es = axis_state.get("evidence_strength", "moderate")
                    obs = axis_state.get("observed", True)
                    if not obs:
                        evidence_qualifier = " (axis previously unobserved)"
                    else:
                        evidence_qualifier = f" ({es} evidence)"
                else:
                    current_val = rel.get(metric, 0.0)
                break

        return (
            f"[MATHEMATICAL CONSTRAINT]: The {metric.upper()} between {node_id} and {target_entity} "
            f"currently sits at {current_val}{evidence_qualifier}. You MUST write the prose such that this metric is "
            f"forcefully shifted by {shift:+.2f}. Provide clear, physical narrative evidence of this change "
            f"in their dialogue or body language."
        )

    # 2. PSYCHOLOGICAL DIRECTIVE (Traits)
    elif vector_path.startswith("traits."):
        # e.g., "traits.ambition" or "traits.ambition.value"
        trait_parts = vector_path.split('.')
        trait_name = trait_parts[1] if len(trait_parts) >= 2 else vector_path
        
        # Look up the trait on the POV character or present entities
        current_val = "unknown"
        for ent in ego_graph.get("focus_entities", []):
            if ent.get("id") == node_id:
                current_val = ent.get("traits", {}).get(trait_name, {}).get("value", "unknown")
                break
        if current_val == "unknown":
            for ent in ego_graph.get("present_entities", []):
                if ent.get("id") == node_id:
                    current_val = ent.get("traits", {}).get(trait_name, {}).get("value", "unknown")
                    break

        return (
            f"[MATHEMATICAL CONSTRAINT]: {node_id}'s internal '{trait_name}' trait "
            f"currently sits at {current_val}. The events of this scene MUST shatter their inertia "
            f"and shift this trait by {shift:+.2f}. Focus the internal monologue on this psychological pivot."
        )

    # 3. EPISTEMIC DIRECTIVE (Beliefs / Dramatic Irony)
    elif vector_path.startswith("beliefs."):
        return (
            f"[EPISTEMIC CONSTRAINT]: Force a realization. {node_id}'s confidence in their belief "
            f"about {vector_path.split('.')[1]} must shift by {shift:+.2f}. Shatter their current worldview."
        )

    return f"Alter the state of {vector_target_id} by a trajectory of {shift:+.2f}."


# ==========================================
# HELPER 3: THE PHYSICS OVERRIDE DETECTOR
# ==========================================
def _generate_physics_override(sandbox: nx.MultiDiGraph) -> Optional[str]:
    """
    Detects split-screen scenarios (entities in separate rooms communicating remotely).
    Returns a strict [PHYSICS OVERRIDE] string for the LLM, or None if single-room.
    """
    # Collect unique locations that Entity nodes are located_in
    entity_locations: set = set()
    for node_id, data in sandbox.nodes(data=True):
        if data.get("node_type") == "Entity":
            loc = data.get("location_id")
            if loc:
                entity_locations.add(loc)

    if len(entity_locations) <= 1:
        return None

    # Check if any communicating_with edges exist
    has_comms = any(
        d.get("edge_type") == "communicating_with"
        for _, _, d in sandbox.edges(data=True)
    )

    if has_comms:
        return (
            "[PHYSICS OVERRIDE]: Characters are in SEPARATE locations communicating remotely. "
            "You MUST NOT describe physical touching, exchanging of items, or any direct physical interaction. "
            "All interaction must be limited to the communication medium (speech, telepathy, etc.)."
        )

    return None


# ==========================================
# HELPER 4: ABDUCTION (Evidence → Latent Variable Update)
# ==========================================

# Mechanism → trait affinity mapping.  When an event's causal mechanism
# is known, only traits in the corresponding list are updated.  Traits
# not in any list receive a reduced fallback impulse.
_MECHANISM_TRAIT_MAP: Dict[str, List[str]] = {
    "physical": ["courage", "fear", "anger", "pain", "strength"],
    "physical_force": ["courage", "fear", "anger", "pain", "strength"],
    "psychological": ["guilt", "paranoia", "despair", "hope", "anxiety", "fear", "grief", "remorse"],
    "epistemic_revelation": ["suspicion", "curiosity", "paranoia", "guilt"],
    "epistemic": ["suspicion", "curiosity", "paranoia", "guilt"],
    "social_coercion": ["ambition", "fear", "rebelliousness", "loyalty", "obedience"],
    "social": ["ambition", "fear", "rebelliousness", "loyalty", "obedience"],
    "emotional": ["love", "affection", "grief", "despair", "hope", "anger", "fear"],
    "informational": ["suspicion", "curiosity", "paranoia"],
    "betrayal": ["anger", "grief", "fear", "loyalty", "affinity"],
}

# Fallback multiplier for traits not matching the mechanism.
_MECHANISM_FALLBACK_FACTOR = 0.2


# ==========================================
# Legacy provenance-prune parity with CausalPhysicsEngine
# ==========================================
# Mirrors ``CausalPhysicsEngine._collect_provenance_invalidations``
# so the legacy (use_causal_engine=False) intervention and
# counterfactual paths drop beliefs whose ``acquired_via_*``
# provenance has been epistemically invalidated by the surgery.
# Without this parity the legacy path could leave stale beliefs in
# the sandbox while the engine path correctly pruned them.
_LEGACY_DESTRUCTIVE_EVENT_TYPES = {"prevented", "never_happened", "removed"}
_LEGACY_DESTRUCTIVE_TRUTH = {"false", "performative"}
_LEGACY_DESTRUCTIVE_STATUS = {"severed", "disabled", "down"}


def _collect_legacy_provenance_invalidations(
    interventions: Dict[str, Any] | None,
) -> tuple[set[str], set[str]]:
    """Return (removed_event_ids, removed_channel_ids) for legacy paths.

    Kept conservative — only unambiguously destructive surgeries
    invalidate provenance. See the engine version for full rationale.
    """
    removed_events: set[str] = set()
    removed_channels: set[str] = set()
    for path, value in (interventions or {}).items():
        if "." not in path:
            continue
        node_id, prop = path.split(".", 1)
        if node_id.startswith("EVT_"):
            if prop == "event_type" and isinstance(value, str) and value in _LEGACY_DESTRUCTIVE_EVENT_TYPES:
                removed_events.add(node_id)
            elif prop == "truth_value" and isinstance(value, str) and value in _LEGACY_DESTRUCTIVE_TRUTH:
                removed_events.add(node_id)
        elif node_id.startswith(("CHN_", "CHAN_")):
            if prop == "status" and isinstance(value, str) and value in _LEGACY_DESTRUCTIVE_STATUS:
                removed_channels.add(node_id)
            elif prop == "participant_ids" and isinstance(value, list) and len(value) == 0:
                removed_channels.add(node_id)
    return removed_events, removed_channels


def _apply_abduction(
    sandbox: nx.MultiDiGraph,
    evidence_node_ids: List[str],
    global_world_state: WorldStateV1,
) -> None:
    """
    Rung 3 Abduction: uses present-day evidence nodes to back-propagate
    latent trait/belief updates into the historical shadow graph.

    For each evidence node that exists in the CURRENT world state, we update
    the sandbox's entity traits and beliefs to reflect what MUST have been
    true at the point of divergence given the observed evidence.
    """
    if not evidence_node_ids:
        return

    physics_settings = get_settings().physics
    evidence_strength_multiplier = physics_settings.strength_multiplier

    for eid in evidence_node_ids:
        # Case 1: Evidence is an Entity — condition on its current factual state
        if eid in global_world_state.entities and sandbox.has_node(eid):
            factual_entity = global_world_state.entities[eid]
            node_data = sandbox.nodes[eid]

            # Determine the temporal horizon of the sandbox for reconstruction
            max_ft = max(
                (d.get("fabula_time", 0) for _, d in sandbox.nodes(data=True) if d.get("fabula_time")),
                default=None,
            )

            # Reconstruct entity state at the sandbox's temporal horizon
            if max_ft is not None and factual_entity.state_timeline:
                reconstructed = reconstruct_entity_at(factual_entity, max_ft)
                target_traits = reconstructed["traits"]
                target_beliefs = reconstructed["beliefs"]
            else:
                target_traits = {k: {"value": v.value, "inertia": v.inertia} for k, v in factual_entity.traits.items()}
                target_beliefs = [b.model_dump() for b in factual_entity.beliefs]

            # Back-propagate traits toward reconstructed evidence values (50% blend)
            for trait_name, tv in target_traits.items():
                tv_value = tv["value"] if isinstance(tv, dict) else tv.value
                sandbox_traits = node_data.get("traits", {})
                if trait_name in sandbox_traits and isinstance(sandbox_traits[trait_name], dict):
                    old_val = sandbox_traits[trait_name].get("value", 0.5)
                    shift = (tv_value - old_val) * 0.5
                    sandbox_traits[trait_name]["value"] = max(0.0, min(1.0, old_val + shift))
            # Back-propagate beliefs
            if "beliefs" not in node_data:
                node_data["beliefs"] = []
            existing_beliefs = node_data["beliefs"]
            for belief in target_beliefs:
                b_target_id = belief.get("target_id") if isinstance(belief, dict) else belief.target_id
                b_state = belief.get("perceived_state") if isinstance(belief, dict) else belief.perceived_state
                if not any(b.get("target_id") == b_target_id and
                          b.get("perceived_state") == b_state
                          for b in existing_beliefs):
                    existing_beliefs.append(belief if isinstance(belief, dict) else belief.model_dump())
            logger.info("[Abduction] Conditioned entity %s on present-day evidence.", eid)

        # Case 2: Evidence is an Event — propagate through causal edges
        elif sandbox.has_node(eid):
            node_data = sandbox.nodes[eid]
            if node_data.get("node_type") == "EventNode":
                for ce in global_world_state.causal_topology:
                    if ce.source_id == eid:
                        # Respect propagation_delay
                        if ce.propagation_delay > 0:
                            target_ft = _get_delay_target_ft(sandbox, ce.target_id, global_world_state)
                            if target_ft < ce.fabula_time + ce.propagation_delay:
                                continue
                        mult = evidence_strength_multiplier.get(
                            ce.evidence_strength, physics_settings.strength_moderate
                        )
                        force_scale = ce.causal_force / physics_settings.causal_force_scaling
                        target_node = sandbox.nodes.get(ce.target_id)
                        if target_node and target_node.get("node_type") == "Entity":
                            traits = target_node.get("traits", {})

                            # Precise mutation: use trait_target/trait_delta
                            if ce.causality_type == "mutation" and ce.trait_target is not None:
                                td = traits.get(ce.trait_target)
                                if isinstance(td, dict) and "value" in td:
                                    delta = (ce.trait_delta if ce.trait_delta is not None else 1.0) * mult * force_scale
                                    td["value"] = max(0.0, min(1.0, td["value"] + delta))
                                continue

                            relevant = _MECHANISM_TRAIT_MAP.get(ce.mechanism, None)
                            for trait_name, trait_data in traits.items():
                                if not isinstance(trait_data, dict) or "value" not in trait_data:
                                    continue
                                old_val = trait_data["value"]
                                if relevant is None or trait_name in relevant:
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * force_scale))
                                else:
                                    trait_data["value"] = max(0.0, min(1.0, old_val + mult * force_scale * _MECHANISM_FALLBACK_FACTOR))
                logger.info("[Abduction] Propagated evidence from event %s.", eid)
        else:
            logger.warning("[Abduction] Evidence node %s not in sandbox. Skipping.", eid)


# ==========================================
# HELPER 5: FORWARD CASCADE (CTF Step 3 — Prediction)
# ==========================================
def _apply_forward_cascade(
    sandbox: nx.MultiDiGraph,
    global_world_state: WorldStateV1,
) -> None:
    """
    CTF Step 3 — Prediction: After abduction and the do-operator, walk
    the causal topology forward, adjusting downstream entity traits
    proportionally to the edge's evidence_strength.

    Aligned with CausalPhysicsEngine.propagate():
      • Topological-sort ordering (falls back to fabula_time if cycles).
      • Impact > Inertia gating — small impulses are absorbed.
      • Bidirectional shifts — traits can decrease as well as increase.
      • Mechanism-targeted updates via _MECHANISM_TRAIT_MAP.
      • Spatial affordance checks (unlocked paths only).
    """
    strength_mult = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}

    # Event ids that the AMWN surgery marked as ``pruned`` — either an
    # utterance whose bridging channel was severed, or an event that
    # provenance invalidation flagged (event_type='prevented',
    # truth_value='false', etc.). Causal edges originating from any of
    # these must NOT propagate forward — the legacy cascade previously
    # ignored the prune flag, so a counterfactual that severed a channel
    # or prevented an event still saw its downstream traits drift.
    pruned_event_ids: set[str] = {
        nid for nid, ndata in sandbox.nodes(data=True)
        if ndata.get("pruned") is True
    }

    # 1. Build a causal DiGraph from the global causal_topology,
    #    restricted to nodes present in the sandbox.
    causal_graph = nx.DiGraph()
    edge_meta: Dict[tuple, Dict[str, Any]] = {}  # (src, tgt) → {weight, mechanism}

    for ce in global_world_state.causal_topology:
        src = ce.source_id
        tgt = ce.target_id
        if src in pruned_event_ids or tgt in pruned_event_ids:
            continue
        if not sandbox.has_node(src) and src not in {
            nid for nid, _ in sandbox.nodes(data=True)
        }:
            continue
        # Respect propagation_delay
        if ce.propagation_delay > 0:
            target_ft = _get_delay_target_ft(sandbox, tgt, global_world_state)
            if target_ft < ce.fabula_time + ce.propagation_delay:
                continue
        evidence_w = strength_mult.get(ce.evidence_strength, 0.5)
        force_scale = ce.causal_force / get_settings().physics.causal_force_scaling
        weight = evidence_w * force_scale
        key = (src, tgt)
        if causal_graph.has_edge(src, tgt):
            existing_w = causal_graph[src][tgt].get("weight", 0.0)
            weight = max(existing_w, weight)
        causal_graph.add_edge(src, tgt, weight=weight)
        # Only overwrite edge_meta if this edge actually won the weight contest
        if key not in edge_meta or weight >= edge_meta[key]["weight"]:
            edge_meta[key] = {
                "weight": weight,
                "mechanism": ce.mechanism,
                "causality_type": ce.causality_type,
                "trait_target": ce.trait_target,
                "trait_delta": ce.trait_delta,
            }

    if causal_graph.number_of_edges() == 0:
        return

    # 2. Topological sort (fall back to fabula_time order if cycles)
    try:
        execution_order = list(nx.topological_sort(causal_graph))
    except nx.NetworkXUnfeasible:
        logger.warning("[Forward Cascade] Cyclic causal graph — falling back to fabula_time order.")
        execution_order = sorted(
            causal_graph.nodes(),
            key=lambda nid: next(
                (e.fabula_time for e in global_world_state.events if e.id == nid),
                float("inf"),
            ),
        )

    # 3. Build a spatial traversability sub-graph for affordance checks.
    # Mirrors `CausalPhysicsEngine._build_spatial_traversable`: locked
    # passages are admitted when ANY entity in the sandbox owns an item
    # whose ``unlock`` affordance targets the barrier (by node-type or
    # name). Without this parity the legacy cascade silently dropped
    # locked-but-unlockable edges that the engine path accepted, making
    # query outcomes path-dependent on `use_causal_engine`.
    traversable = nx.DiGraph()
    for u, v, d in sandbox.edges(data=True):
        if d.get("edge_type") != "connected_to":
            continue
        if not d.get("is_locked", False):
            traversable.add_edge(u, v)
            continue
        barrier_id = d.get("barrier_item_id")
        if not barrier_id:
            continue
        barrier_node = sandbox.nodes.get(barrier_id, {})
        barrier_name = barrier_node.get("name", "")
        barrier_node_type = barrier_node.get("node_type", "NarrativeObject")
        for nid, ndata in sandbox.nodes(data=True):
            if ndata.get("node_type") != "NarrativeObject":
                continue
            if ndata.get("owner_id") is None:
                continue
            for aff in ndata.get("affordances", []) or []:
                if not isinstance(aff, dict):
                    continue
                if aff.get("action") != "unlock":
                    continue
                aff_target = aff.get("target_type", "")
                if aff_target == barrier_node_type or aff_target == barrier_name:
                    traversable.add_edge(u, v)
                    break
            else:
                continue
            break

    # 4. Propagate
    for node_id in execution_order:
        target = sandbox.nodes.get(node_id)
        if not target or target.get("node_type") != "Entity":
            continue

        incoming = list(causal_graph.in_edges(node_id, data=True))
        if not incoming:
            continue

        traits = target.get("traits", {})
        if not traits:
            continue

        tgt_loc = target.get("location_id")

        for trait_name, trait_data in traits.items():
            if not isinstance(trait_data, dict) or "value" not in trait_data:
                continue

            current_val = trait_data["value"]
            trait_inertia = trait_data.get("inertia", 0.5)
            total_impact = 0.0
            spatial_ok = True

            for src, _, edata in incoming:
                w = edata.get("weight", 0.5)
                meta = edge_meta.get((src, node_id), {})
                mechanism = meta.get("mechanism", "physical")
                edge_ctype = meta.get("causality_type", "chain_reaction")
                edge_trait_target = meta.get("trait_target")
                edge_trait_delta = meta.get("trait_delta")

                # Precise mutation: if the edge specifies a trait_target,
                # only affect that specific trait (skip all others).
                if edge_ctype == "mutation" and edge_trait_target is not None:
                    if trait_name != edge_trait_target:
                        continue  # this edge doesn't affect this trait
                    if edge_trait_delta is not None:
                        total_impact += edge_trait_delta * w
                        continue

                relevant_traits = _MECHANISM_TRAIT_MAP.get(mechanism)

                # Mechanism-targeted gating
                if relevant_traits is not None and trait_name not in relevant_traits:
                    logger.debug("[ForwardCascade] %s→%s trait=%s: mechanism=%s fallback, w %.3f→%.3f",
                                 src, node_id, trait_name, mechanism, w, w * _MECHANISM_FALLBACK_FACTOR)
                    w *= _MECHANISM_FALLBACK_FACTOR

                src_data = sandbox.nodes.get(src)
                if not src_data:
                    continue

                # WorldTrait domain filtering (mirrors CausalPhysicsEngine.propagate)
                if src_data.get("node_type") == "WorldTrait":
                    affected = src_data.get("affected_domains", [])
                    if affected and mechanism not in affected:
                        logger.debug(
                            "[ForwardCascade] %s→%s: mechanism=%s not in affected_domains %s, applying fallback",
                            src, node_id, mechanism, affected,
                        )
                        w *= _MECHANISM_FALLBACK_FACTOR

                if src_data.get("node_type") == "Entity":
                    src_trait = src_data.get("traits", {}).get(trait_name)
                    if isinstance(src_trait, dict) and "value" in src_trait:
                        # Signed delta: shift toward source trait value
                        total_impact += (src_trait["value"] - current_val) * w
                    else:
                        total_impact += w
                    # Spatial affordance
                    src_loc = src_data.get("location_id")
                    if (src_loc and tgt_loc and src_loc != tgt_loc
                            and traversable.has_node(src_loc)
                            and traversable.has_node(tgt_loc)):
                        if not nx.has_path(traversable, src_loc, tgt_loc):
                            spatial_ok = False
                elif src_data.get("node_type") == "WorldTrait":
                    # Magnitude-scaled impulse (mirrors CausalPhysicsEngine.propagate)
                    mag = src_data.get("magnitude", {})
                    mag_value = mag.get("value", 0.5) if isinstance(mag, dict) else 0.5
                    total_impact += mag_value * w
                else:
                    total_impact += w

            if not spatial_ok:
                logger.debug("[ForwardCascade] BLOCKED spatial: %s.%s", node_id, trait_name)
                continue

            if abs(total_impact) <= trait_inertia:
                logger.debug("[ForwardCascade] BLOCKED inertia: %s.%s |impact|=%.3f <= inertia=%.3f",
                             node_id, trait_name, abs(total_impact), trait_inertia)
                continue

            # Dampened shift: effective = impact - sign × inertia
            sign = 1 if total_impact > 0 else -1
            effective_shift = total_impact - sign * trait_inertia
            new_val = max(0.0, min(1.0, current_val + effective_shift))
            trait_data["value"] = new_val

    logger.info("[Forward Cascade] Propagation complete.")


# ==========================================
# HELPER 6: SOCIAL CASCADE (mutation_social propagation)
# ==========================================
def _apply_social_cascade(
    sandbox: nx.MultiDiGraph,
    global_world_state: WorldStateV1,
) -> None:
    """
    Walk ``mutation_social`` causal edges from the global causal_topology
    and apply relationship metric deltas to the sandbox's relationship edges.

    Mirrors CausalPhysicsEngine.propagate_social() for the legacy
    (use_causal_engine=False) counterfactual path.
    """
    strength_mult = {"weak": 0.25, "moderate": 0.5, "strong": 0.75}

    # Same prune gate as the forward cascade — a mutation_social edge
    # whose source is a pruned event (severed-channel utterance or
    # provenance-invalidated event) must not propagate.
    pruned_event_ids: set[str] = {
        nid for nid, ndata in sandbox.nodes(data=True)
        if ndata.get("pruned") is True
    }

    for ce in global_world_state.causal_topology:
        if ce.causality_type != "mutation_social":
            continue
        if ce.source_id in pruned_event_ids:
            continue

        target_id = ce.target_id          # perspective entity
        counterpart_id = ce.rel_counterpart_id  # other entity in dyad
        metric = ce.trait_target
        raw_delta = ce.trait_delta or 0.0

        if not counterpart_id or not metric:
            continue
        if not sandbox.has_node(target_id) or not sandbox.has_node(counterpart_id):
            continue

        # Respect propagation_delay
        if ce.propagation_delay > 0:
            target_ft = _get_delay_target_ft(sandbox, ce.target_id, global_world_state)
            if target_ft < ce.fabula_time + ce.propagation_delay:
                continue

        # Scale delta
        evidence_w = strength_mult.get(ce.evidence_strength, 0.5)
        force_scale = ce.causal_force / get_settings().physics.causal_force_scaling
        scaled_delta = raw_delta * evidence_w * force_scale

        # Find the relationship edge target_id → counterpart_id
        rel_found = False
        for ru, rv, rkey, rdata in sandbox.out_edges(target_id, data=True, keys=True):
            if rv == counterpart_id and rdata.get("edge_type") == "relationship":
                rel_found = True
                # Prefer per-axis ``metrics[metric].value`` when present.
                per_metric = (rdata.get("metrics") or {}).get(metric) or {}
                if "value" in per_metric and isinstance(per_metric["value"], (int, float)):
                    current_val = float(per_metric["value"])
                else:
                    current_val = rdata.get(metric, 0.0)
                    if not isinstance(current_val, (int, float)):
                        current_val = 0.0
                rel_inertia = per_metric.get("inertia", rdata.get("inertia", 0.3))

                # Impact > Inertia gating
                if abs(scaled_delta) <= rel_inertia:
                    logger.debug("[SocialCascade] Inertia blocked: %s→%s %s |delta|=%.3f <= inertia=%.3f",
                                 target_id, counterpart_id, metric, abs(scaled_delta), rel_inertia)
                    break

                # Dampened shift
                sign = 1 if scaled_delta > 0 else -1
                effective_shift = scaled_delta - sign * rel_inertia
                new_val = current_val + effective_shift

                # Clamp
                if metric == "fear":
                    new_val = max(0.0, min(1.0, new_val))
                else:
                    new_val = max(-1.0, min(1.0, new_val))

                sandbox[ru][rv][rkey][metric] = new_val
                # Mirror into the per-axis ``metrics`` dict so
                # downstream readers using the new shape stay in sync.
                if isinstance(rdata.get("metrics"), dict):
                    axis_state = rdata["metrics"].setdefault(metric, {
                        "value": current_val,
                        "inertia": rel_inertia,
                        "evidence_strength": ce.evidence_strength,
                        "last_updated_fabula": ce.fabula_time,
                        "observed": True,
                    })
                    axis_state["value"] = new_val
                    axis_state["last_updated_fabula"] = max(
                        axis_state.get("last_updated_fabula", 0), ce.fabula_time,
                    )
                    # Propagated mutation = a measurement; flip observed.
                    axis_state["observed"] = True
                    # Carry triggering edge's evidence_strength forward.
                    if ce.evidence_strength:
                        axis_state["evidence_strength"] = ce.evidence_strength
                logger.info("[SocialCascade] %s→%s %s: %.3f→%.3f (trigger=%s)",
                            target_id, counterpart_id, metric, current_val, new_val, ce.source_id)
                break

        # No existing relationship edge — create one with full per-axis metrics
        if not rel_found:
            if metric == "fear":
                primary_value = max(0.0, min(1.0, scaled_delta))
            else:
                primary_value = max(-1.0, min(1.0, scaled_delta))
            edge_attrs = {
                "edge_type": "relationship",
                "affinity": 0.0,
                "fear": 0.0,
                "power_dynamic": 0.0,
                "inertia": 0.3,
                "evidence_strength": "weak",
                "last_updated_fabula": ce.fabula_time,
                "world_id": "shadow",
                "metrics": default_relationship_metrics_dict(
                    primary_metric=metric,
                    primary_value=primary_value,
                    fabula_time=ce.fabula_time,
                    evidence_strength=ce.evidence_strength,
                    inertia=0.3,
                ),
            }
            edge_attrs[metric] = primary_value
            sandbox.add_edge(target_id, counterpart_id, **edge_attrs)
            logger.info("[SocialCascade] Created relationship %s→%s with %s=%.3f (trigger=%s)",
                        target_id, counterpart_id, metric, edge_attrs[metric], ce.source_id)


# =====================================================================
# Phase 3 — Public typed-target Pearl-Rung facades
#
# ``apply_intervention`` and ``find_pod`` are thin wrappers over the
# existing rung-2 / rung-3 logic. They accept the typed
# :class:`DoTarget` discriminated union (Phase 0) and delegate into the
# typed-dispatch layer on the causal engine (Phase 1) so callers can
# request proposition / belief / concern surgeries without round-
# tripping through the legacy ``Dict[str, Any]`` shape.
# =====================================================================

from pydantic import BaseModel as _PdBaseModel  # local alias to avoid header pollution


class PointOfDivergence(_PdBaseModel):
    """The historical anchor a Rung-3 counterfactual rolls back to.

    ``fabula_time`` is the earliest fabula time that would need to be
    re-simulated to alter ``target_outcome``. ``candidate_event_ids`` is
    the ranked list of events that could plausibly serve as the divergent
    pivot (Phase 5 will re-rank by the Kahneman-Miller × concern-load
    mutability prior; Phase 3 emits them in temporal order, earliest
    first).
    """
    fabula_time: int
    candidate_event_ids: List[str] = []
    target_kind: str = "event"
    target_id: Optional[str] = None
    mutability_prior_used: bool = False


def _resolve_focus_from_do_targets(do_targets: List[Any], world_state: WorldStateV1) -> List[str]:
    """Best-effort focus-entity resolution for a typed-target list.

    DoBelief / DoConcern / DoTrait carry an explicit ``holder_id``;
    DoEvent's actor / target ids are looked up on the event;
    DoProposition contributes the proposition's ``referent_ids``;
    DoChannel / DoRelationship / DoCausalEdge / DoSpatialEdge
    contribute their endpoint ids when those resolve to entities.
    Falls back to a single arbitrary entity when nothing matches so the
    ego-graph extractor always has *something* to anchor on.
    """
    from shadow_loom.query_models import (
        DoEvent, DoTrait, DoBelief, DoConcern, DoProposition,
        DoChannel, DoRelationship, DoCausalEdge, DoSpatialEdge,
    )
    focus: List[str] = []
    seen: set[str] = set()

    def _add(eid: Optional[str]) -> None:
        if eid and eid in world_state.entities and eid not in seen:
            focus.append(eid)
            seen.add(eid)

    for t in do_targets:
        if isinstance(t, (DoTrait, DoBelief, DoConcern)):
            _add(t.holder_id)
        elif isinstance(t, DoEvent):
            evt = next((e for e in world_state.events if e.id == t.event_id), None)
            if evt:
                for eid in (evt.actor_ids or []):
                    _add(eid)
                for eid in (evt.target_ids or []):
                    _add(eid)
        elif isinstance(t, DoProposition):
            prop = next(
                (p for p in (world_state.propositions or [])
                 if p.proposition_id == t.proposition_id),
                None,
            )
            if prop:
                for eid in (prop.referent_ids or []):
                    _add(eid)
        elif isinstance(t, DoRelationship):
            _add(t.source_entity_id)
            _add(t.target_entity_id)
        elif isinstance(t, DoChannel):
            ch = (world_state.channels or {}).get(t.channel_id)
            if ch:
                for eid in (ch.participant_ids or []):
                    _add(eid)
        elif isinstance(t, DoCausalEdge):
            _add(t.source_id)
            _add(t.target_id)
            _add(getattr(t, "rel_counterpart_id", None))
        elif isinstance(t, DoSpatialEdge):
            # Spatial endpoints are LOC_ ids; admit any entity at those
            # locations as plausible focus.
            for ent_id, ent in world_state.entities.items():
                if ent.location_id in (t.source_id, t.target_id):
                    _add(ent_id)

    if not focus and world_state.entities:
        focus.append(next(iter(world_state.entities)))
    return focus


def _do_target_seed_ids(do_targets: List[Any]) -> set:
    """Collect every node id mentioned on the LHS of a typed
    :class:`DoTarget` for shadow-path expansion in the ego-graph
    extractor. Mirrors the legacy ``intervention_seeds`` collection
    from the dict-shaped surgery path so rung-2/3 prose sees the
    upstream causal lineage of every typed surgery.
    """
    from shadow_loom.query_models import (
        DoEvent, DoTrait, DoBelief, DoConcern, DoProposition, DoWorldTrait,
        DoChannel, DoRelationship, DoCausalEdge, DoSpatialEdge,
    )
    seeds: set = set()
    for t in do_targets:
        if isinstance(t, DoEvent):
            seeds.add(t.event_id)
        elif isinstance(t, DoTrait):
            seeds.add(t.holder_id)
        elif isinstance(t, DoBelief):
            seeds.add(t.holder_id)
            if t.target_id:
                seeds.add(t.target_id)
        elif isinstance(t, DoConcern):
            seeds.add(t.holder_id)
        elif isinstance(t, DoProposition):
            seeds.add(t.proposition_id)
        elif isinstance(t, DoWorldTrait):
            seeds.add(t.world_trait_id)
        elif isinstance(t, DoChannel):
            seeds.add(t.channel_id)
        elif isinstance(t, DoRelationship):
            seeds.add(t.source_entity_id)
            seeds.add(t.target_entity_id)
        elif isinstance(t, (DoCausalEdge, DoSpatialEdge)):
            seeds.add(t.source_id)
            seeds.add(t.target_id)
    return {s for s in seeds if s}


def _typed_targets_require_vacuity_skip(do_targets: List[Any]) -> bool:
    """Return True iff any typed target carries a surgery shape that
    legitimately bypasses the vacuity check.

    Only edge-typed surgeries (DoChannel / DoRelationship / DoCausalEdge
    / DoSpatialEdge) qualify: those rewrite the sandbox topology
    directly and do not produce a node-level mutation
    :func:`_check_engine_vacuity` would observe.

    DoBelief and DoConcern are intentionally NOT in this set, even
    though they reach the sandbox via :meth:`apply_do_targets` rather
    than the legacy dict path: their handlers
    (:meth:`CausalPhysicsEngine._apply_do_belief` /
    :meth:`_apply_do_concern`) can silently early-return when the
    holder / concern is missing, and we want vacuity to surface that
    as implausibility rather than mask it as success. The recorded
    BeliefMutation / ConcernMutation lists feed into vacuity already
    (see :func:`_check_engine_vacuity`), so a successful Belief /
    Concern surgery WILL bypass vacuity through the mutation-list
    short-circuit rather than the typed-target short-circuit.

    The five legacy-round-trippable variants (DoEvent / DoTrait /
    DoProposition / DoWorldTrait / DoNarrativeObject) are also
    excluded: vacuity MUST still fire on those (otherwise the engine
    could silently swallow a no-op without the implausibility
    disclosure)."""
    from shadow_loom.query_models import (
        DoCausalEdge,
        DoChannel,
        DoRelationship,
        DoSpatialEdge,
    )
    edge_only = (DoCausalEdge, DoChannel, DoRelationship, DoSpatialEdge)
    return any(isinstance(t, edge_only) for t in (do_targets or []))


def _lift_do_targets_to_legacy_dict(do_targets: List[Any]) -> Dict[str, Any]:
    """Lift the legacy-compatible subset of typed ``DoTarget``s into the
    ``Dict[str, Any]`` shape consumed by ``engine.execute(...)``'s CTF
    preflight, plausibility checks, and provenance pruner.

    Five DoTarget variants round-trip cleanly into the legacy keyspace:
    ``DoEvent``, ``DoTrait``, ``DoProposition``, ``DoWorldTrait``, and
    ``DoNarrativeObject`` (one key per ``location_id`` / ``owner_id``
    mutation). The remaining edge / pair surgeries (DoBelief /
    DoConcern / DoChannel / DoRelationship / DoCausalEdge /
    DoSpatialEdge) cannot be encoded as a single dotted key (they
    require a holder/target pair or edge endpoint information the
    legacy shape never carried) \u2014 they reach the sandbox via
    ``engine.apply_do_targets`` and bypass the legacy dict path.
    """
    from shadow_loom.query_models import (
        DoEvent,
        DoNarrativeObject,
        DoProposition,
        DoTrait,
        DoWorldTrait,
    )
    out: Dict[str, Any] = {}
    for t in do_targets:
        if isinstance(t, DoEvent):
            out[f"{t.event_id}.event_type"] = (
                "prevented" if t.occurred is False else "occurred"
            )
        elif isinstance(t, DoTrait):
            out[f"{t.holder_id}.traits.{t.trait_name}"] = float(t.value)
        elif isinstance(t, DoProposition):
            out[f"{t.proposition_id}.truth"] = bool(t.truth)
        elif isinstance(t, DoWorldTrait):
            out[f"{t.world_trait_id}.value"] = float(t.value)
        elif isinstance(t, DoNarrativeObject):
            # Emit one dotted key per requested mutation. ``set_*_null``
            # round-trips as a ``None`` value the coercer reverses back
            # into the same null-mutation kwargs.
            if t.new_location_id is not None:
                out[f"{t.object_id}.location_id"] = t.new_location_id
            elif getattr(t, "set_location_null", False):
                out[f"{t.object_id}.location_id"] = None
            if t.new_owner_id is not None:
                out[f"{t.object_id}.owner_id"] = t.new_owner_id
            elif getattr(t, "set_owner_null", False):
                out[f"{t.object_id}.owner_id"] = None
    return out


def apply_intervention(
    do_targets: List[Any],
    global_world_state: WorldStateV1,
    *,
    fabula_anchor: Optional[int] = None,
    syuzhet_anchor: Optional[int] = None,
    use_causal_engine: bool = True,
    target_node_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Public typed-target facade over the Rung-2 (do-operator) pipeline.

    Builds the ego graph, instantiates the sandbox (with the Phase-2
    utility-layer stamp), and dispatches every typed
    :class:`DoTarget` through :meth:`CausalPhysicsEngine.apply_do_targets`.

    Returns the same dict shape as the rung-2 branch of
    :func:`calculate_narrative_physics` so existing downstream consumers
    (directive assembler, generator, MCP) can opt in incrementally.
    Adds ``do_targets`` (model-dumped) and the new typed mutation lists
    (belief / concern / proposition) for callers that need them.
    """
    if not do_targets:
        return {
            "status": "implausible",
            "query_type": "intervention",
            "physics_state": {},
            "implausibility_reason": "Empty do_targets list \u2014 nothing to apply.",
        }

    focus_ids = _resolve_focus_from_do_targets(do_targets, global_world_state)
    ego_graph = extract_ego_graph_from_memory(
        global_world_state, focus_ids, fabula_anchor,
        syuzhet_anchor=syuzhet_anchor,
    )
    shadow_graph = AMWNInstantiator.create_sandbox(ego_graph.model_dump(), "intervention")
    _stamp_utility_layer(shadow_graph, global_world_state, fabula_anchor=fabula_anchor)

    if use_causal_engine:
        engine = CausalPhysicsEngine(shadow_graph, global_world_state)
        engine.apply_do_targets(do_targets)
        # ``apply_do_targets`` stashed the legacy event/trait dict it
        # constructed; pass it (possibly empty) through ``execute`` so
        # provenance invalidation, propagation, and Rule-3 pre-flight
        # all see the surgeries the typed dispatch performed.
        legacy = getattr(engine, "_last_legacy_interventions", {}) or {}
        physics_result = engine.execute(
            rung=2,
            interventions=legacy,
            target_node_ids=target_node_ids or [],
        )
        return {
            "status": "success",
            "query_type": "intervention",
            "physics_state": physics_result.sandbox_data,
            "do_targets": [t.model_dump() for t in do_targets],
            "math_changes": legacy,
            "mutations": [m.model_dump() for m in physics_result.mutations],
            "social_mutations": [m.model_dump() for m in physics_result.social_mutations],
            "blocked": [b.model_dump() for b in physics_result.blocked],
            "intervened_nodes": physics_result.intervened_nodes,
            "belief_mutations": [m.model_dump() for m in physics_result.belief_mutations],
            "concern_mutations": [m.model_dump() for m in physics_result.concern_mutations],
            "proposition_mutations": [m.model_dump() for m in physics_result.proposition_mutations],
            "_causal_physics_result": physics_result,
        }

    # Legacy non-engine path: only event / trait targets are supported.
    from shadow_loom.query_models import DoEvent, DoTrait
    legacy: Dict[str, Any] = {}
    for t in do_targets:
        if isinstance(t, DoEvent):
            legacy[f"{t.event_id}.event_type"] = (
                "prevented" if t.occurred is False else "occurred"
            )
        elif isinstance(t, DoTrait):
            legacy[f"{t.holder_id}.traits.{t.trait_name}"] = float(t.value)
    if legacy:
        AMWNInstantiator.execute_interventions(shadow_graph, legacy)
    return {
        "status": "success",
        "query_type": "intervention",
        "physics_state": nx.node_link_data(shadow_graph),
        "do_targets": [t.model_dump() for t in do_targets],
        "math_changes": legacy,
    }


def _mutability_score(
    event: Any,
    global_world_state: WorldStateV1,
) -> float:
    """Phase-5 mutability prior: Kahneman-Miller atypicality × Roese
    concern-load.

    Sums, over every concern in the world that targets a proposition
    whose ``referent_ids`` intersect ``event.actor_ids ∪ event.target_ids``:

        salience(concern) × (1 - typicality(event, concern))

    where ``typicality`` is 0.5 when the event falls inside the concern's
    activation window (so the concern is "expecting" something like this
    to happen — Kahneman-Miller's *normal* line) and 1.0 when the event
    is outside the window or the concern has no window (so an event
    inside an "always-on" concern is *typical*; events outside any
    relevant window are maximally atypical and therefore most mutable).

    Returns 0.0 when no concern in the world is utility-coupled to
    ``event``'s referents — a temporally-earliest fallback ordering then
    takes precedence.
    """
    referents = set(event.actor_ids or []) | set(event.target_ids or [])
    if not referents:
        return 0.0
    score = 0.0
    propositions = global_world_state.propositions or []
    prop_by_id = {p.proposition_id: p for p in propositions}
    for ent in (global_world_state.entities or {}).values():
        for c in (getattr(ent, "concerns", None) or []):
            prop = prop_by_id.get(c.proposition_id)
            if prop is None:
                continue
            if not (set(prop.referent_ids or []) & referents):
                continue
            # Resolve concern state at the event's fabula_time so the
            # mutability scorer respects ``ConcernSnapshot`` updates
            # (salience / window can evolve over the story the same
            # way Entity / GlobalTrait state does).
            if c.state_timeline:
                _resolved = reconstruct_concern_at(c, event.fabula_time)
                window = _resolved["activation_fabula_window"]
                _salience = float(_resolved["salience"])
            else:
                window = c.activation_fabula_window
                _salience = float(c.salience)
            if window and len(window) == 2:
                lo, hi = window
                inside = (lo <= event.fabula_time <= hi)
                # Inside the activation window means the concern is
                # *live* at this fabula_time, so the event genuinely
                # bears on it — contribute ``salience * 0.5``. Outside
                # the window the concern is dormant (not yet acquired
                # or already closed), so the event is irrelevant to
                # mutability re-ranking and contributes 0.
                typicality = 0.5 if inside else 1.0
            else:
                # Always-on concern (or cleared-window sentinel): the
                # concern is live at every tick → always in scope.
                typicality = 0.5
            score += _salience * (1.0 - typicality)
    return score


def _rerank_candidates_by_mutability(
    candidate_event_ids: List[str],
    global_world_state: WorldStateV1,
) -> List[str]:
    """Stable-sort candidates by descending mutability score.

    Ties (and zero-score candidates, which is the common case when no
    concern is utility-coupled to the event's referents) preserve the
    upstream temporal ordering produced by :func:`find_pod`.
    """
    by_id = {e.id: e for e in (global_world_state.events or [])}
    indexed = list(enumerate(candidate_event_ids))
    def _key(pair):
        i, eid = pair
        evt = by_id.get(eid)
        s = _mutability_score(evt, global_world_state) if evt is not None else 0.0
        # Negative score so descending; index keeps ties stable.
        return (-s, i)
    return [eid for _, eid in sorted(indexed, key=_key)]


def find_pod(
    target_outcome: Any,
    global_world_state: WorldStateV1,
    *,
    fabula_anchor: Optional[int] = None,
    mutability_prior: bool = False,
) -> PointOfDivergence:
    """Locate the historical fabula_time a Rung-3 counterfactual must
    roll back to in order to alter ``target_outcome``.

    ``target_outcome`` is a typed :class:`DoTarget` describing the
    outcome the caller wants to perturb (an event being prevented, a
    proposition's truth flipped, a belief clamped, etc).

    Ranking of ``candidate_event_ids``:
      * ``mutability_prior=False`` (default): temporal — earliest
        relevant event first.
      * ``mutability_prior=True``: re-rank by the Kahneman-Miller ×
        Roese concern-load score (see :func:`_mutability_score`), with
        temporal order as tie-break. Sets
        ``PointOfDivergence.mutability_prior_used=True``.
    """
    from shadow_loom.query_models import (
        DoEvent, DoProposition, DoBelief, DoConcern, DoTrait,
    )

    candidate_event_ids: List[str] = []
    target_kind = getattr(target_outcome, "target_kind", "event")
    target_id: Optional[str] = None

    if isinstance(target_outcome, DoEvent):
        target_id = target_outcome.event_id
        evt = next((e for e in global_world_state.events if e.id == target_id), None)
        if evt is None:
            raise ValueError(
                f"find_pod: event {target_id} not in world state."
            )
        # Anchor at the event itself; any earlier event in its causal
        # ancestry is a candidate divergence pivot.
        anchor_ft = int(evt.fabula_time)
        ancestor_ids = {
            ce.source_id for ce in (global_world_state.causal_topology or [])
            if ce.target_id == target_id
        }
        ancestor_events = [
            e for e in global_world_state.events
            if e.id in ancestor_ids and e.fabula_time < anchor_ft
        ]
        ancestor_events.sort(key=lambda e: e.fabula_time)
        candidate_event_ids = [e.id for e in ancestor_events] + [target_id]
        if mutability_prior:
            candidate_event_ids = _rerank_candidates_by_mutability(
                candidate_event_ids, global_world_state,
            )
        return PointOfDivergence(
            fabula_time=anchor_ft,
            candidate_event_ids=candidate_event_ids,
            target_kind="event", target_id=target_id,
            mutability_prior_used=mutability_prior,
        )

    if isinstance(target_outcome, DoProposition):
        target_id = target_outcome.proposition_id
        prop = next(
            (p for p in (global_world_state.propositions or [])
             if p.proposition_id == target_id),
            None,
        )
        if prop is None:
            raise ValueError(f"find_pod: proposition {target_id} not in world state.")
        # Anchor at the latest fabula_time the proposition's truth was
        # asserted before the requested anchor (or before the simulation
        # horizon).
        horizon = (
            int(fabula_anchor) if fabula_anchor is not None
            else int(max((e.fabula_time for e in global_world_state.events), default=0))
        )
        truth_keys = sorted(int(k) for k in (prop.truth_at_fabula or {}).keys() if int(k) <= horizon)
        anchor_ft = truth_keys[-1] if truth_keys else 0
        # Candidate events: any event touching one of the proposition's referents
        # before the anchor.
        ref_set = set(prop.referent_ids or [])
        cand = [
            e for e in global_world_state.events
            if (set(e.actor_ids or []) | set(e.target_ids or [])) & ref_set
            and e.fabula_time <= horizon
        ]
        cand.sort(key=lambda e: e.fabula_time)
        candidate_event_ids = [e.id for e in cand]
        if mutability_prior:
            candidate_event_ids = _rerank_candidates_by_mutability(
                candidate_event_ids, global_world_state,
            )
        return PointOfDivergence(
            fabula_time=anchor_ft,
            candidate_event_ids=candidate_event_ids,
            target_kind="proposition", target_id=target_id,
            mutability_prior_used=mutability_prior,
        )

    if isinstance(target_outcome, DoBelief):
        # Anchor: latest event involving the belief's target before fabula_anchor.
        target_id = f"{target_outcome.holder_id}\u2192{target_outcome.target_id}"
        horizon = (
            int(fabula_anchor) if fabula_anchor is not None
            else int(max((e.fabula_time for e in global_world_state.events), default=0))
        )
        cand = [
            e for e in global_world_state.events
            if (target_outcome.target_id in (e.actor_ids or [])
                or target_outcome.target_id in (e.target_ids or []))
            and e.fabula_time <= horizon
        ]
        cand.sort(key=lambda e: e.fabula_time)
        anchor_ft = cand[-1].fabula_time if cand else 0
        candidate_event_ids = [e.id for e in cand]
        if mutability_prior:
            candidate_event_ids = _rerank_candidates_by_mutability(
                candidate_event_ids, global_world_state,
            )
        return PointOfDivergence(
            fabula_time=int(anchor_ft),
            candidate_event_ids=candidate_event_ids,
            target_kind="belief", target_id=target_id,
            mutability_prior_used=mutability_prior,
        )

    if isinstance(target_outcome, DoConcern):
        # Anchor: start of the concern's activation window, or 0 if always-on.
        target_id = target_outcome.concern_id
        ent = global_world_state.entities.get(target_outcome.holder_id)
        if ent is None:
            raise ValueError(f"find_pod: holder {target_outcome.holder_id} not in world state.")
        concern = next(
            (c for c in (ent.concerns or []) if c.concern_id == target_id),
            None,
        )
        if concern is None:
            raise ValueError(
                f"find_pod: concern {target_id} not on {target_outcome.holder_id}."
            )
        window = concern.activation_fabula_window
        anchor_ft = int(window[0]) if window else 0
        return PointOfDivergence(
            fabula_time=anchor_ft,
            candidate_event_ids=[],
            target_kind="concern", target_id=target_id,
            mutability_prior_used=False,
        )

    if isinstance(target_outcome, DoTrait):
        target_id = f"{target_outcome.holder_id}.{target_outcome.trait_name}"
        horizon = (
            int(fabula_anchor) if fabula_anchor is not None
            else int(max((e.fabula_time for e in global_world_state.events), default=0))
        )
        cand = [
            e for e in global_world_state.events
            if target_outcome.holder_id in ((e.actor_ids or []) + (e.target_ids or []))
            and e.fabula_time <= horizon
        ]
        cand.sort(key=lambda e: e.fabula_time)
        anchor_ft = cand[0].fabula_time if cand else 0
        candidate_event_ids = [e.id for e in cand]
        if mutability_prior:
            candidate_event_ids = _rerank_candidates_by_mutability(
                candidate_event_ids, global_world_state,
            )
        return PointOfDivergence(
            fabula_time=int(anchor_ft),
            candidate_event_ids=candidate_event_ids,
            target_kind="trait", target_id=target_id,
            mutability_prior_used=mutability_prior,
        )

    raise TypeError(
        f"find_pod: unsupported target type {type(target_outcome).__name__}"
    )


    logger.info("[Social Cascade] Propagation complete.")