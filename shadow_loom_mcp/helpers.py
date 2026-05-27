# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared helpers for Shadow-Loom MCP tools — project resolution, world loading."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import Context

from shadow_loom.db import (
    find_project_by_name,
    get_active_version,
    get_latest_version,
    get_mcp_idempotent_response,
    get_version,
    get_version_by_id,
    save_mcp_idempotent_response,
    save_version,
    set_active_version,
)
from shadow_loom.auditor import (
    AuditorConfig,
    Finding,
    _finding_severity_score,
    findings_from_sources,
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


def _apply_inert_envelope(response: dict, physics_result: dict | None) -> None:
    """Surface ``intervention_inert`` / ``intervention_inert_reason`` on
    the MCP response envelope when the engine flagged the surgery as a
    no-op (round-3 audit fix; extracted for testability in round 5)."""
    if not physics_result:
        return
    if physics_result.get("intervention_inert"):
        response["intervention_inert"] = True
        reason = physics_result.get("intervention_inert_reason")
        if reason:
            response["intervention_inert_reason"] = reason


# Round-8 audit (MCP-P1-02 / P1-01 / P2-03): contract-complete
# envelope helpers. The MCP surface historically only exposed a
# fragmented view of the audit + engine streams (`change_impact`,
# `engine_threshold_failures`, `audit_converged`, `audit_iterations`)
# and dropped scene-level POV metadata entirely, so external clients
# couldn't tell *which* POV a draft locked to, *why* a draft was
# accepted/retried/rolled back, or *which* engine threshold tripped
# without re-running the math themselves. These helpers project the
# already-typed core models (``Finding``, ``GeneratedScene``,
# ``FeedbackLoopResult.history``) into compact dicts and are wired
# in below at the end of ``_build_pipeline_envelope_extras``.

# Round-8 audit (MCP-P1-03): allowlist of per-run auditor overrides
# external clients are permitted to set. We intentionally restrict
# overrides to the budget knobs added in round-7 (plus ``max_iterations``,
# which was always env-tunable) — model names, temperatures, and
# thresholds for engine scoring are deliberately *not* exposed via the
# per-call surface so a misconfigured client can't downgrade engine
# rigour silently. Unknown / disallowed keys are dropped with a
# DEBUG log rather than raised so the call still proceeds.
_ALLOWED_AUDITOR_OVERRIDES: frozenset[str] = frozenset({
    "max_iterations",
    "regression_retry_budget",
    "failed_open_tolerance",
    "enable_deterministic_prose_checks",
    "pov_breach_threshold",
})


def _build_auditor_config_override(
    overrides: dict[str, Any] | None,
) -> tuple[AuditorConfig | None, list[dict[str, Any]]]:
    """Coerce a per-call ``auditor_overrides`` dict into an
    ``AuditorConfig`` instance for ``PipelineConfig.auditor_config``,
    or return ``None`` when the caller didn't supply any overrides.

    Returns ``(config_or_None, rejected_overrides)``. Round-9 C5 fix:
    validate **key-by-key** so a single invalid value no longer drops
    the entire override set. The ``rejected_overrides`` list contains
    one ``{key, reason}`` entry per dropped override and is wired into
    the response envelope so clients can surface the partial-apply
    explicitly.

    Round-9 C6: dropped overrides log **keys only** (plus the value's
    type/length metadata). Client-supplied values can contain
    arbitrary strings — never log them verbatim.
    """
    if not overrides:
        return None, []
    clean: dict[str, Any] = {}
    rejected: list[dict[str, Any]] = []

    def _value_meta(value: Any) -> str:
        if value is None:
            return "type=NoneType"
        try:
            length = len(value)  # type: ignore[arg-type]
            return f"type={type(value).__name__},len={length}"
        except TypeError:
            return f"type={type(value).__name__}"

    for k, v in overrides.items():
        if k not in _ALLOWED_AUDITOR_OVERRIDES:
            logger.debug(
                "[MCP] Dropping disallowed auditor override key %r (%s)",
                k, _value_meta(v),
            )
            rejected.append({"key": k, "reason": "disallowed_key"})
            continue
        if v is None:
            rejected.append({"key": k, "reason": "null_value"})
            continue
        # Round-9 C5: validate this single field by constructing a
        # one-key AuditorConfig. Pydantic surfaces a precise error for
        # the offending field; we keep every other valid override.
        try:
            AuditorConfig(**{k: v})
        except Exception as exc:  # noqa: BLE001 — bad client input
            logger.debug(
                "[MCP] Dropping invalid auditor override key %r (%s): %s",
                k, _value_meta(v), type(exc).__name__,
            )
            rejected.append({"key": k, "reason": "invalid_value"})
            continue
        clean[k] = v
    if not clean:
        return None, rejected
    try:
        return AuditorConfig(**clean), rejected
    except Exception as exc:  # noqa: BLE001 — should be unreachable
        logger.exception(
            "[MCP] Failed to assemble AuditorConfig from per-key-validated "
            "overrides (keys=%s); falling back to defaults",
            sorted(clean.keys()),
        )
        for k in clean:
            rejected.append({"key": k, "reason": "assembly_error"})
        return None, rejected


def _findings_envelope(result: "PipelineResult") -> list[dict[str, Any]] | None:
    """Project the final-cycle ``AuditResult`` + engine failures into a
    flat ``findings[]`` list using the core ``findings_from_sources``
    adapter. Returns ``None`` when no audit was run (skip_audit path).
    """
    fb = result.feedback_result
    if fb is None:
        return None
    final_audit = fb.history[-1].audit_result if fb.history else None
    findings = findings_from_sources(
        final_audit, list(fb.engine_threshold_failures or []),
    )
    return [f.model_dump() for f in findings]


def _audit_trace_envelope(
    result: "PipelineResult",
) -> list[dict[str, Any]] | None:
    """Per-iteration compact trace so external clients can explain
    *why* a draft was accepted/retried. Round-9 F11: engine failures
    are now sourced **per cycle** from ``snap.engine_threshold_failures``
    (populated by the feedback loop) instead of being attached only
    to the terminal row. Falls back to the result-level final-cycle
    failures when an older snapshot is missing the per-cycle list.
    """
    fb = result.feedback_result
    if fb is None or not fb.history:
        return None
    trace: list[dict[str, Any]] = []
    last_idx = len(fb.history) - 1
    fallback_final = list(fb.engine_threshold_failures or [])
    for idx, snap in enumerate(fb.history):
        per_cycle = list(getattr(snap, "engine_threshold_failures", []) or [])
        if not per_cycle and idx == last_idx:
            # Belt-and-braces: snapshots produced before the F11
            # patch don't carry per-cycle engine failures. For the
            # final row, fall back to the result-level field so the
            # trace stays consistent with ``engine_threshold_failures``
            # surfaced at the response root.
            per_cycle = fallback_final
        findings = findings_from_sources(snap.audit_result, per_cycle)
        trace.append({
            "iteration": snap.iteration,
            "passed": snap.audit_result.passed,
            "failed_open": snap.audit_result.failed_open,
            "violation_count": len(snap.audit_result.violations),
            "engine_failure_count": len(per_cycle),
            "severity_score": _finding_severity_score(findings),
            "summary": snap.audit_result.audit_summary or "",
        })
    return trace


def _scene_metadata_envelope(
    result: "PipelineResult",
) -> dict[str, Any] | None:
    """Surface scene-level POV + rendering metadata so callers can see
    which POV the renderer locked to (mirror of the brief's pov_lock
    for single-POV; final-beat POV for rotating; possibly None for
    ensemble) and which rendering mode produced the prose.

    For multi-POV directives the brief's ``additional_pov_locks`` /
    ``pov_policy`` are dug out of the recorded generation step in
    ``result.history`` when available; otherwise we report the
    single-POV defaults so clients can write uniform code.
    """
    scene = result.scene
    if scene is None:
        return None
    meta: dict[str, Any] = {
        "pov_entity": scene.pov_entity,
        "rendering_mode": scene.rendering_mode,
        "additional_pov_locks": [],
        "pov_policy": "single",
    }
    for step in result.history.steps:
        if step.get("step") != "generation":
            continue
        brief = step.get("brief") or {}
        rendering = brief.get("rendering") if isinstance(brief, dict) else None
        if not isinstance(rendering, dict):
            continue
        addl = rendering.get("additional_pov_locks") or []
        policy = rendering.get("pov_policy") or "single"
        if addl:
            meta["additional_pov_locks"] = list(addl)
        meta["pov_policy"] = policy
        break
    return meta


# ── Project resolution ────────────────────────────────────────────

def resolve_project(
    project_id: int | None,
    project_name: str | None,
    ctx: Context,
    *,
    min_role: str = "viewer",
) -> tuple[int | None, str | None]:
    """Resolve a project by ID or name. Returns (project_id, error).

    ``min_role`` defaults to ``"viewer"`` (read access). Mutation tools
    MUST pass ``min_role="editor"`` (or ``"admin"``) so viewer-level
    project members cannot mutate project data via an MCP write tool
    that only checks the global ``write`` scope.
    """
    if project_id is not None:
        err = check_project_access(project_id, ctx, min_role=min_role)
        if err:
            return None, err
        return project_id, None

    if project_name:
        user_id = get_user_id(ctx)
        proj = find_project_by_name(project_name, owner_id=user_id)
        if proj is None:
            # AUDIT (post-2026-05-26): the global fallback used to
            # resolve to *any* project with this display name,
            # which leaked the existence (and access-control verdict)
            # of strangers' private projects through the
            # ``check_project_access`` error path.
            #
            # Round-4 audit refinement: the previous fix pre-filtered
            # to ``proj.is_public or proj.owner_id is None``, which
            # also blocked legitimate collaborators (project members
            # with editor/admin grants) and shared-admin users from
            # resolving private projects by name. Delegate the
            # authorisation decision to ``check_project_access`` (which
            # already honours owner / public-at-viewer / member-by-role)
            # and, if access is denied, return the same generic
            # "not found" error so the original existence-leak is
            # still closed.
            proj = find_project_by_name(project_name)
            if proj is not None:
                if check_project_access(proj.id, ctx, min_role=min_role) is not None:
                    proj = None
        if proj is None:
            return None, f"Project '{project_name}' not found."
        err = check_project_access(proj.id, ctx, min_role=min_role)
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

    **Note**: returns the *raw* (un-projected) world state. Read-side
    tools should prefer :func:`load_world_state_projected` so shadow
    rows surface the per-branch AMWN-split layered view; only
    write-side tools that need to round-trip the JSON unchanged or
    that must persist into the same branch should use this raw path
    (see :func:`load_world_state_with_branch`).
    """
    ws, vid, _, _ = load_world_state_with_branch(
        project_id, version, ctx=ctx,
    )
    return ws, vid


def load_world_state_with_branch(
    project_id: int,
    version: int | None = None,
    *,
    ctx: Context | None = None,
) -> tuple[WorldStateV1 | None, int | None, str, str | None]:
    """Load a world state and also return its branch identity.

    Returns ``(ws, version_row_id, branch_world_id, branch_label)``.
    Write-side tools (``patch_world_state``, ``branch``) call this so
    the persisted child row keeps the same branch identity as its
    ancestor — the previous ``(ws, vid)`` load lost ``world_id`` /
    ``branch_label`` and silently demoted shadow-branch writes onto
    the factual mainline at ``save_version`` (which defaults
    ``world_id='factual'``).
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
        return None, None, "factual", None
    ws = WorldStateV1.model_validate_json(ver.world_state_json)
    return (
        ws,
        ver.id,
        getattr(ver, "world_id", "factual") or "factual",
        getattr(ver, "branch_label", None),
    )


def load_world_state_projected(
    project_id: int,
    version: int | None = None,
    *,
    ctx: Context | None = None,
) -> tuple[WorldStateV1 | None, int | None]:
    """Branch-aware load for read tools.

    Equivalent to :func:`load_world_state` followed by
    ``ws.projected_for_branch(branch_world_id, branch_label)`` so
    every downstream reader of ``ws.entities`` / ``ws.objects`` /
    ``ws.propositions`` / ``ws.world_traits`` sees the AMWN-split
    layered view appropriate to the loaded row's branch. Factual
    rows are returned unchanged (``projected_for_branch`` is a no-op
    fast path).

    Without this, every MCP read tool that consumes the loaded ws
    silently returns the factual baseline even when the request
    targets a shadow row — the same drift class fixed in the UI.
    """
    ws, vid, bw, bl = load_world_state_with_branch(
        project_id, version, ctx=ctx,
    )
    if ws is None:
        return None, None
    return ws.projected_for_branch(branch_world_id=bw, branch_label=bl), vid


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
    skip_reextraction: bool | None = None,
    auditor_overrides: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Run pipeline, save new version, return response dict.

    ``skip_reextraction=None`` (the default) routes by ``query_type``:
    prose-rendering query types (``observation``, ``intervention``,
    ``counterfactual``, ``directive``, ``manual_edit``) get
    re-extraction enabled so the saved world graph reflects the prose
    just generated; other types keep the old fast-path. The previous
    unconditional default of ``True`` meant every MCP-driven Continue
    / What-If / Intervene persisted prose against an *unchanged*
    world graph, so chained MCP calls built on stale topology and the
    saved version's AMWN nodes never inherited the new
    entities/events the prose introduced. Round-6 audit: ``observation``
    advances the clock and produces narration that can introduce new
    facts about the world; omitting it from the prose-rendering set
    left MCP ``narrate`` (observation mode) saving prose against a
    stale topology too.
    """
    # Round-9 C7: short-circuit on idempotency_key. Client-supplied
    # retries (transport blip, MCP reconnect, user double-clicks the
    # "narrate" button) replay the previously cached envelope instead
    # of producing a duplicate version row and burning another
    # generation budget. Misses fall through to the normal pipeline
    # path and the response is cached at the end of the function.
    if idempotency_key:
        try:
            cached = get_mcp_idempotent_response(
                project_id=project_id,
                ancestor_id=ancestor_row_id,
                idempotency_key=idempotency_key,
            )
        except Exception:  # noqa: BLE001 — cache lookup must never block
            logger.exception(
                "[MCP] Idempotency cache lookup failed for "
                "project=%s ancestor=%s; treating as a miss",
                project_id, ancestor_row_id,
            )
            cached = None
        if cached is not None:
            cached.setdefault("idempotent_replay", True)
            return cached

    # Derive re-extraction policy from query type when caller didn't
    # explicitly set it. Keep the explicit override path so callers
    # that need the old fast behaviour can still opt in.
    if skip_reextraction is None:
        prose_rendering_types = {
            "observation",
            "intervention",
            "counterfactual",
            "directive",
            "manual_edit",
        }
        skip_reextraction = (
            getattr(query, "query_type", None) not in prose_rendering_types
        )
    # Seed the synthetic v0 in the VersionedWorldModel with the
    # ancestor row's branch identity. ``from_world_state`` defaults
    # to ``world_id="factual"``; if we accept that default for a
    # ``world_state`` that was actually loaded from a shadow row, the
    # pipeline's branch routing (which keys on
    # ``vwm.history[-1].world_id``) silently demotes a shadow
    # continuation back onto the factual mainline. Look up the
    # ancestor's tag and propagate it.
    seed_world_id: str = "factual"
    seed_branch_label: str | None = None
    if ancestor_row_id is not None:
        ancestor_row = get_version_by_id(ancestor_row_id)
        if ancestor_row is not None:
            seed_world_id = ancestor_row.world_id or "factual"
            seed_branch_label = ancestor_row.branch_label
    vwm = VersionedWorldModel.from_world_state(
        world_state,
        world_id=seed_world_id,  # type: ignore[arg-type]
        branch_label=seed_branch_label,
    )
    auditor_cfg_override, rejected_overrides = _build_auditor_config_override(
        auditor_overrides,
    )
    cfg = PipelineConfig(
        skip_audit=skip_audit,
        skip_reextraction=skip_reextraction,
        auditor_config=auditor_cfg_override,
    )

    # Activate per-user model overrides (default model, per-stage
    # models, custom OpenAI-compat providers) for this MCP request.
    # The token MUST be reset in ``finally`` so that worker threads
    # (this function is invoked from ``asyncio.to_thread`` by MCP
    # tools) don't leak this user's overrides into the NEXT request
    # served by the same reused worker.
    _user_context_token: object | None = None
    try:
        from shadow_loom.settings import set_user_context as _set_user_context
        _user_context_token = _set_user_context(user_row_id)
    except Exception:  # noqa: BLE001 — never block the pipeline call
        logger.debug("[MCP] Failed to activate user model overrides", exc_info=True)

    try:
        try:
            result: PipelineResult = run_pipeline(query, versioned_model=vwm, config=cfg)
        except Exception as e:
            logger.exception("Pipeline failed")
            # Sanitised public error \u2014 raw ``str(e)`` could leak
            # internal paths / SQL / provider error bodies. Full
            # exception detail is captured by ``logger.exception``.
            return {
                "error": "Pipeline failed",
                "error_type": type(e).__name__,
            }
    finally:
        if _user_context_token is not None:
            try:
                from shadow_loom.settings import (
                    reset_user_context as _reset_user_context,
                )
                _reset_user_context(_user_context_token)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[MCP] Failed to reset user model overrides",
                    exc_info=True,
                )

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

    # Match the UI's persistence policy: do not write a new version row
    # when the engine produced prose against an unchanged world graph
    # because re-extraction raised. Otherwise the new row would store
    # divergent prose/world state under one version, and chained MCP
    # tools would build on a graph that never really advanced.
    skip_for_reextraction = bool(result.reextraction_failed)

    if not short_circuited and not skip_for_reextraction:
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
        # Normalised changeset block — surfaces every MergeChangeset
        # counter (additive + affect + deletion + supersession) on the
        # action response so external clients don't need a follow-up
        # ``get_history`` call to know what structurally changed.
        if (
            result.world_model
            and result.world_model.history
            and result.world_model.history[-1].changeset
        ):
            try:
                response["changeset"] = (
                    result.world_model.history[-1].changeset.model_dump()
                )
            except Exception:
                logger.exception("Failed to serialise changeset for MCP response")
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
            # Optimistic concurrency: detect concurrent writers by
            # comparing the pointer's pre-write value to the ancestor
            # this call expected. A mismatch means another tool /
            # session advanced the pointer while we were running the
            # pipeline; we still publish our new version (callers can
            # always rebase manually) but flag the response so clients
            # can warn and refresh their cached view (round-3 audit).
            prior_active_id: int | None = None
            try:
                prior_active = get_active_version(project_id, user_row_id)
                if prior_active is not None:
                    prior_active_id = prior_active.id
            except Exception:
                logger.exception(
                    "Failed to read prior active-version pointer for "
                    "concurrency check"
                )
            if (
                prior_active_id is not None
                and ancestor_row_id is not None
                and prior_active_id != ancestor_row_id
            ):
                response["active_version_conflict"] = {
                    "expected_ancestor_id": ancestor_row_id,
                    "actual_active_id": prior_active_id,
                }
                logger.warning(
                    "Active-version pointer drifted during MCP call "
                    "(expected ancestor %s, found %s); flagging response.",
                    ancestor_row_id, prior_active_id,
                )
            # Round-7 audit: only advance the active pointer when the
            # caller's view of the head still matches — otherwise the
            # mutation silently overrides whatever the concurrent
            # writer just committed, even though we just told the
            # client there was a conflict. Surface a refusal instead.
            if (
                "active_version_conflict" in response
            ):
                response["active_version_pointer_updated"] = False
            else:
                try:
                    set_active_version(project_id, user_row_id, ver.id)
                    response["active_version_pointer_updated"] = True
                except Exception:
                    logger.exception("Failed to update active-version pointer")
                    response["active_version_pointer_updated"] = False
        if world_model_unchanged and result.prose:
            response["world_model_unchanged"] = True
    elif skip_for_reextraction:
        # Surface that no version was created — the prose is still
        # returned to the caller below via ``result.reextraction_failed``
        # / ``result.prose`` so they can decide whether to retry.
        response["version_skipped"] = True
        response["version_skipped_reason"] = "reextraction_failed"
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
        # Round-5 audit: surface answer-step failures so MCP clients
        # can distinguish a real answer from the placeholder string
        # the pipeline returns when ``_run_answer_step`` raised.
        if result.physics_result.get("answer_failed"):
            response["answer_failed"] = True
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
        # Inert-intervention disclosure (round-3 audit fix). Surfaces a
        # no-op Rung-2/3 surgery so external clients can tell the user
        # "the requested change has no representable consequences" rather
        # than displaying an empty cascade as if it were a complete one.
        _apply_inert_envelope(response, result.physics_result)
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
                if af.affective_loss_mse is not None:
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

    # Round-8 audit (MCP-P1-02 / P1-01 / P2-03): unified findings,
    # per-iteration audit_trace, and scene POV/rendering metadata.
    # See ``_findings_envelope`` / ``_audit_trace_envelope`` /
    # ``_scene_metadata_envelope`` for rationale.
    findings_block = _findings_envelope(result)
    if findings_block is not None:
        response["findings"] = findings_block
    audit_trace = _audit_trace_envelope(result)
    if audit_trace is not None:
        response["audit_trace"] = audit_trace
    scene_meta = _scene_metadata_envelope(result)
    if scene_meta is not None:
        response["scene_metadata"] = scene_meta

    # Round-9 C5: per-key auditor-override validation telemetry. When
    # any override key was dropped (disallowed key, null value, or
    # invalid per-field value) surface a structured list so the
    # caller can see *which* keys didn't take effect and why, instead
    # of silently falling back to defaults.
    if rejected_overrides:
        response["rejected_auditor_overrides"] = rejected_overrides

    # Plain-English summary for the user (LLMs / chat surfaces can read this
    # directly instead of trying to assemble one from the structured fields).
    response["lay_summary"] = humanize_pipeline_result(
        result,
        requested_effect=requested_effect,
        requested_intensity=requested_intensity,
    )

    # Round-9 C7: persist this response under the caller's
    # idempotency_key so a client-side retry replays the same envelope
    # without re-running the pipeline (or producing a duplicate
    # version row). Cache write is best-effort — failures only
    # disable future replay, not the current response.
    if idempotency_key:
        try:
            save_mcp_idempotent_response(
                project_id=project_id,
                ancestor_id=ancestor_row_id,
                idempotency_key=idempotency_key,
                response=response,
                version_row_id=response.get("version_row_id"),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[MCP] Failed to persist idempotency cache for "
                "project=%s ancestor=%s (response still returned)",
                project_id, ancestor_row_id,
            )

    return response
