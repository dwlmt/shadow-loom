"""Shared helpers for Shadow-Loom MCP tools — project resolution, world loading."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastmcp import Context

from shadow_loom.db import (
    find_project_by_name,
    get_active_version,
    get_latest_version,
    get_project,
    get_version,
    save_version,
    set_active_version,
)
from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.models import WorldStateV1
from shadow_loom.pipeline import (
    PipelineConfig,
    PipelineResult,
    humanize_pipeline_result,
    run_pipeline,
)
from shadow_loom.query_models import UserRequest

from shadow_loom_mcp.auth import check_project_access, get_user_id

logger = logging.getLogger(__name__)


# ── Project resolution ────────────────────────────────────────────

def resolve_project(
    project_id: int | None,
    project_name: str | None,
    ctx: Context,
) -> tuple[int | None, str | None]:
    """Resolve a project by ID or name. Returns (project_id, error)."""
    if project_id is not None:
        err = check_project_access(project_id, ctx)
        if err:
            return None, err
        return project_id, None

    if project_name:
        user_id = get_user_id(ctx)
        proj = find_project_by_name(project_name, owner_id=user_id)
        if proj is None:
            proj = find_project_by_name(project_name)
        if proj is None:
            return None, f"Project '{project_name}' not found."
        err = check_project_access(proj.id, ctx)
        if err:
            return None, err
        return proj.id, None

    return None, "Specify project_id or project_name."


# ── World state loading ──────────────────────────────────────────

def load_world_state(
    project_id: int,
    version: int | None = None,
    *,
    ctx: Context | None = None,
) -> tuple[WorldStateV1 | None, int | None]:
    """Load a world state from DB. Returns (ws, version_row_id) or (None, None).

    Resolution order when ``version`` is omitted:
      1. The authenticated user's active-version pointer (per-project),
         when ``ctx`` is provided and a pointer exists.
      2. The project's latest version.
    """
    if version is not None:
        ver = get_version(project_id, version)
    else:
        ver = None
        if ctx is not None:
            user_id = get_user_id(ctx)
            if user_id is not None:
                ver = get_active_version(project_id, user_id)
        if ver is None:
            ver = get_latest_version(project_id)
    if ver is None:
        return None, None
    ws = WorldStateV1.model_validate_json(ver.world_state_json)
    return ws, ver.id


# ── Pipeline run + save ──────────────────────────────────────────

def run_and_save(
    query: UserRequest,
    project_id: int,
    world_state: WorldStateV1,
    ancestor_row_id: int | None,
    user_row_id: int | None,
    raw_query: str | None,
    *,
    skip_audit: bool = True,
    skip_reextraction: bool = True,
) -> dict[str, Any]:
    """Run pipeline, save new version, return response dict."""
    vwm = VersionedWorldModel.from_world_state(world_state)
    cfg = PipelineConfig(
        skip_audit=skip_audit,
        skip_reextraction=skip_reextraction,
    )

    try:
        result: PipelineResult = run_pipeline(query, versioned_model=vwm, config=cfg)
    except Exception as e:
        logger.exception("Pipeline failed")
        return {"error": f"Pipeline failed: {e}"}

    # If the engine deemed the request implausible AND the caller did NOT
    # ask to force generation, we explicitly skip persisting a new version
    # \u2014 the world state was not advanced, only an explanation was produced.
    short_circuited = bool(
        result.implausible
        and (result.world_model is None or result.world_model.version == vwm.version)
        and not result.feedback_result
    )

    new_ws = result.world_model.current if result.world_model else world_state

    # Detect whether the world model actually advanced. When the caller
    # opted into ``skip_reextraction`` (the default for narrate/direct)
    # the new version row stores prose against the *unchanged* world
    # graph \u2014 callers must know this to avoid building further turns
    # on the assumption that prose became canon.
    world_model_unchanged = bool(
        result.world_model is None
        or result.world_model.version == vwm.version
    )

    response: dict[str, Any] = {
        "project_id": project_id,
        "query_type": query.query_type,
    }

    if not short_circuited:
        changeset_json = None
        branch_world_id = "factual"
        branch_label = None
        if result.world_model and result.world_model.history:
            last = result.world_model.history[-1]
            if last.changeset:
                changeset_json = last.changeset.model_dump_json()
            branch_world_id = last.world_id
            branch_label = last.branch_label

        if world_model_unchanged and result.prose:
            description = (
                f"{query.query_type} query (prose only \u2014 world model not re-extracted)"
            )
        else:
            description = f"{query.query_type} query"

        ver = save_version(
            project_id=project_id,
            world_state_json=new_ws.model_dump_json(),
            ancestor_id=ancestor_row_id,
            source=query.query_type,
            description=description,
            changeset_json=changeset_json,
            raw_query=raw_query,
            parsed_query_json=query.model_dump_json(),
            prose=result.prose,
            user_id=user_row_id,
            world_id=branch_world_id,
            branch_label=branch_label,
        )
        response["version"] = ver.version
        response["version_row_id"] = ver.id
        # AMWN branch envelope (Story-integration plan, Step 6): tells
        # the MCP client which branch the new version landed on so it
        # can render the version DAG correctly without a follow-up call.
        response["branch"] = {
            "world_id": ver.world_id,
            "branch_label": ver.branch_label,
            "ancestor_id": ver.ancestor_id,
        }
        # Auto-advance the user's active-version pointer to the row we
        # just created so subsequent tool calls default to it.
        if user_row_id is not None:
            try:
                set_active_version(project_id, user_row_id, ver.id)
            except Exception:
                logger.exception("Failed to update active-version pointer")
        if world_model_unchanged and result.prose:
            response["world_model_unchanged"] = True
    else:
        # Surface that no version was created.
        response["version_skipped"] = True

    if result.reextraction_failed:
        # Prose was generated but Step 6\u20137 raised \u2014 caller must
        # not treat the new version as a canonical advancement.
        response["reextraction_failed"] = True
        if result.reextraction_error:
            response["reextraction_error"] = result.reextraction_error

    if result.implausible:
        response["implausible"] = True
        response["implausibility_reason"] = result.implausibility_reason
        if result.implausibility_details:
            response["implausibility_details"] = result.implausibility_details
    if result.prose:
        response["prose"] = result.prose
    if result.physics_result:
        response["physics_status"] = result.physics_result.get("status", "unknown")
        answer = result.physics_result.get("answer")
        if answer:
            response["answer"] = answer
        # ctf-calculus pre-flight surface (Correa & Bareinboim 2025).
        # Surfaced on the MCP envelope so external clients can see which
        # interventions/evidence the engine dropped before simulation
        # and explain to the user when their request was provably vacuous.
        rule3 = result.physics_result.get("rule3_pruned_interventions") or []
        rule2 = result.physics_result.get("rule2_redundant_evidence") or []
        if rule3:
            response["rule3_pruned_interventions"] = list(rule3)
        if rule2:
            response["rule2_redundant_evidence"] = list(rule2)
        # Channels & beliefs subsystem: surface counterfactual side-effects
        # on the epistemic layer (utterances neutralised, channels severed,
        # downstream beliefs invalidated by provenance pruning).
        pruned_utts = result.physics_result.get("pruned_utterance_event_ids") or []
        disabled_chans = result.physics_result.get("disabled_channel_ids") or []
        pruned_beliefs = result.physics_result.get("pruned_beliefs_count") or 0
        if pruned_utts:
            response["pruned_utterance_event_ids"] = list(pruned_utts)
        if disabled_chans:
            response["disabled_channel_ids"] = list(disabled_chans)
        if pruned_beliefs:
            response["pruned_beliefs_count"] = int(pruned_beliefs)
    if result.converged is not None:
        response["audit_converged"] = result.converged
        response["audit_iterations"] = result.audit_iterations
    if result.feedback_result and result.feedback_result.change_impact:
        response["change_impact"] = result.feedback_result.change_impact.model_dump()
    # Surface deterministic engine threshold gate + flat affective metrics so
    # external clients don't have to dig into the nested change_impact blob.
    fb = result.feedback_result
    if fb is not None:
        if fb.engine_thresholds_passed is not None:
            response["engine_thresholds_passed"] = fb.engine_thresholds_passed
        if fb.engine_threshold_failures:
            response["engine_threshold_failures"] = list(fb.engine_threshold_failures)
        ci = fb.change_impact
        if ci is not None:
            af = ci.affective_feedback
            cf = ci.causal_feedback
            if af is not None:
                if af.emotional_trajectory_scores:
                    response["achieved_intensity"] = dict(af.emotional_trajectory_scores)
                response["affective_loss"] = af.affective_loss_mse
            if cf is not None:
                response["foreshadowing_score"] = cf.foreshadowing_payoff_score
                response["cognitive_plausibility_score"] = cf.cognitive_plausibility_score
                response["miracle_step_count"] = len(cf.miracle_steps_detected or [])
    # Echo the directive's request so the achieved-vs-target gap is interpretable.
    requested_effect = getattr(query, "target_effect", None)
    requested_intensity = getattr(query, "intensity", None)
    if requested_effect is not None:
        response["requested_target_effect"] = requested_effect
    if requested_intensity is not None:
        response["requested_intensity"] = requested_intensity
    if result.evaluation_result:
        response["evaluation"] = result.evaluation_result.model_dump()
    if result.world_model:
        response["world_model_version"] = result.world_model.version

    # Plain-English summary for the user (LLMs / chat surfaces can read this
    # directly instead of trying to assemble one from the structured fields).
    response["lay_summary"] = humanize_pipeline_result(
        result,
        requested_effect=requested_effect,
        requested_intensity=requested_intensity,
    )

    return response
