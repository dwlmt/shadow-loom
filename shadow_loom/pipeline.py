"""
Complete end-to-end pipeline orchestrator for Shadow-Loom.

Connects all pipeline stages into a single ``run_pipeline()`` entry point:

  1. **Ingestion** (optional) — raw text → ``WorldStateV1``
  2. **Narrative Physics** — query routing, ego-graph, sandbox, causal engine
  3. **Brief Assembly** — ``CreativeBrief`` from physics result
  4. **Generation** — prose rendering (Step 10)
  5. **Audit + Refinement** — recursive feedback loop (Steps 11–12)
  6. **Prose Re-extraction** — extract topology from generated prose
  7. **Merge** — merge topology into a versioned deep-copy of the world model

The pipeline is flexible:
  • An existing ``WorldStateV1`` can be passed in (skips ingestion).
  • A ``VersionedWorldModel`` can be passed in to continue an existing history.
  • Non-prose query types (interrogate, general) skip steps 3–7.
  • Every intermediate result is recorded in ``PipelineHistory``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from shadow_loom.settings import get_settings as _get_settings

from shadow_loom.auditor import (
    AuditorConfig,
    FeedbackLoopResult,
    compute_causal_feedback,
    compute_affective_feedback,
    _finalize_narrative_order,
    render_and_audit,
)
from shadow_loom.directive_assembly import CreativeBrief, DirectiveAssembler
from shadow_loom.extract_graph import (
    VersionedWorldModel,
    WorldModelVersion,
    extract_topology_from_prose,
)
from shadow_loom.generation import (
    GeneratedScene,
    GenerationConfig,
    render_from_query,
)
from shadow_loom.ingestion import ExtractionConfig, ValidationReport, run_extraction, run_extraction_async
from shadow_loom.models import WorldStateV1
from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.query_models import UserRequest, EvaluationQuery, EvaluationResult

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

class PipelineConfig(BaseModel):
    """Unified configuration for all pipeline stages."""

    # --- Physics ---
    use_causal_engine: bool = Field(
        default=True,
        description="Enable the Rung 2/3 causal physics engine.",
    )
    temporal_anchor: Optional[int] = Field(
        default=None,
        description="Fabula-time horizon for the physics engine.",
    )
    syuzhet_anchor: Optional[int] = Field(
        default=None,
        description="Syuzhet-index horizon for affective calculus.",
    )

    # --- Generation ---
    generation_config: Optional[GenerationConfig] = Field(
        default=None,
        description="LLM configuration for prose generation.",
    )

    # --- Audit ---
    auditor_config: Optional[AuditorConfig] = Field(
        default=None,
        description="LLM configuration for the audit/refinement loop.",
    )
    skip_audit: bool = Field(
        default=False,
        description="Skip the audit loop (Step 11–12). Useful for fast iteration.",
    )

    # --- Re-extraction + Merge ---
    extraction_config: Optional[ExtractionConfig] = Field(
        default=None,
        description="LLM configuration for prose → topology re-extraction.",
    )
    skip_reextraction: bool = Field(
        default=False,
        description="Skip prose re-extraction and world-model merge (Step 6–7).",
    )

    # --- Ingestion ---
    ingestion_config: Optional[ExtractionConfig] = Field(
        default=None,
        description="LLM configuration for raw-text ingestion (Step 1). "
        "Only used when ``raw_text`` is provided instead of a WorldStateV1.",
    )

    # --- World-model versioning ---
    max_snapshots: int = Field(
        default=10,
        description="Maximum number of full world-state snapshots to retain "
        "in the VersionedWorldModel history (last-K).",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_from_settings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k, v in _get_settings().pipeline_config_kwargs().items():
                data.setdefault(k, v)
        return data


# =====================================================================
# Pipeline step records
# =====================================================================

class PhysicsStepRecord(BaseModel):
    """Snapshot of the narrative-physics result."""
    query_type: str
    status: str
    physics_state: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query-type-specific extras (mutations, hidden_deltas, etc.).",
    )


class GenerationStepRecord(BaseModel):
    """Snapshot of a single generation pass."""
    scene: GeneratedScene
    brief: Optional[CreativeBrief] = None


class AuditStepRecord(BaseModel):
    """Snapshot of the full audit/refinement loop."""
    feedback_result: FeedbackLoopResult


class ReextractionStepRecord(BaseModel):
    """Snapshot of the prose → topology extraction + merge step."""
    events_added: int = 0
    causal_edges_added: int = 0
    entity_updates_applied: int = 0
    entity_updates_skipped: List[str] = Field(default_factory=list)
    new_version: int = 0


class IngestionStepRecord(BaseModel):
    """Snapshot of the raw-text ingestion step."""
    is_valid: bool = True
    num_events: int = 0
    num_entities: int = 0
    num_locations: int = 0


class PipelineHistory(BaseModel):
    """Ordered record of every step executed in a pipeline run."""
    steps: List[Dict[str, Any]] = Field(default_factory=list)

    def record(self, step_name: str, data: BaseModel | Dict[str, Any]) -> None:
        payload = data.model_dump() if isinstance(data, BaseModel) else data
        self.steps.append({"step": step_name, **payload})


# =====================================================================
# Pipeline result
# =====================================================================

class PipelineResult(BaseModel):
    """Everything the pipeline produces in a single run."""

    # --- The answer ---
    prose: Optional[str] = Field(
        default=None,
        description="The final generated prose (None for non-prose queries).",
    )
    scene: Optional[GeneratedScene] = Field(
        default=None,
        description="The full GeneratedScene object (None for non-prose queries).",
    )

    # --- Physics output ---
    physics_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw output of calculate_narrative_physics.",
    )

    # --- Audit ---
    converged: Optional[bool] = Field(
        default=None,
        description="Whether the audit loop converged (None if skipped).",
    )
    audit_iterations: Optional[int] = Field(
        default=None,
        description="Number of audit iterations run (None if skipped).",
    )
    feedback_result: Optional[FeedbackLoopResult] = Field(
        default=None,
        description="Full audit loop result (None if skipped/not applicable).",
    )

    # --- Evaluation ---
    evaluation_result: Optional[EvaluationResult] = Field(
        default=None,
        description="Full-story evaluation result (None if not an evaluate query).",
    )

    # --- World model ---
    world_model: Optional[VersionedWorldModel] = Field(
        default=None,
        description="The versioned world model after merge (or initial if no merge).",
    )

    # --- History ---
    history: PipelineHistory = Field(
        default_factory=PipelineHistory,
        description="Ordered record of every pipeline step executed.",
    )

    # --- Query metadata ---
    query_type: str = ""


# =====================================================================
# The pipeline
# =====================================================================

def run_pipeline(
    query: UserRequest,
    *,
    world_state: WorldStateV1 | None = None,
    versioned_model: VersionedWorldModel | None = None,
    raw_text: str | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the full Shadow-Loom pipeline end-to-end.

    Supply exactly one of:
      • ``world_state`` — an existing ``WorldStateV1`` (wrapped in a new
        ``VersionedWorldModel`` automatically).
      • ``versioned_model`` — an existing ``VersionedWorldModel`` to
        continue building on (preserves history).
      • ``raw_text`` — raw narrative text to ingest first (Step 1).

    Parameters
    ----------
    query : UserRequest
        The user query (observation, intervention, counterfactual,
        directive, interrogate, or general).
    world_state : WorldStateV1 or None
        Pre-built world state. Mutually exclusive with the others.
    versioned_model : VersionedWorldModel or None
        Pre-built versioned world model with history.
    raw_text : str or None
        Raw narrative text to ingest into a WorldStateV1 first.
    config : PipelineConfig or None
        Unified configuration for all stages.

    Returns
    -------
    PipelineResult
        Contains the final prose, scene, physics result, audit result,
        updated versioned world model, and full step history.
    """
    cfg = config or PipelineConfig()
    history = PipelineHistory()
    result = PipelineResult(query_type=query.query_type, history=history)

    # =================================================================
    # Step 0: Resolve the world model
    # =================================================================
    sources = sum([world_state is not None, versioned_model is not None, raw_text is not None])
    if sources == 0:
        raise ValueError(
            "Supply one of: world_state, versioned_model, or raw_text."
        )
    if sources > 1:
        raise ValueError(
            "Supply only one of: world_state, versioned_model, or raw_text."
        )

    if raw_text is not None:
        # --- Step 1: Ingestion ---
        logger.info("[Pipeline] Step 1: Ingesting raw text (%d chars).", len(raw_text))
        ing_cfg = cfg.ingestion_config or ExtractionConfig()
        ws, validation_report = run_extraction(raw_text, config=ing_cfg)
        vwm = VersionedWorldModel.from_world_state(ws, max_snapshots=cfg.max_snapshots)
        history.record("ingestion", IngestionStepRecord(
            is_valid=validation_report.is_valid,
            num_events=len(ws.events),
            num_entities=len(ws.entities),
            num_locations=len(ws.locations),
        ))
        logger.info(
            "[Pipeline] Ingestion complete — %d events, %d entities, valid=%s.",
            len(ws.events), len(ws.entities), validation_report.is_valid,
        )
    elif versioned_model is not None:
        vwm = versioned_model
        ws = vwm.current
        logger.info(
            "[Pipeline] Using existing VersionedWorldModel at v%d.",
            vwm.version,
        )
    else:
        # world_state is not None
        vwm = VersionedWorldModel.from_world_state(world_state, max_snapshots=cfg.max_snapshots)
        ws = vwm.current
        logger.info("[Pipeline] Wrapped WorldStateV1 in VersionedWorldModel v0.")

    result.world_model = vwm

    # =================================================================
    # Step 2: Narrative Physics
    # =================================================================
    logger.info(
        "[Pipeline] Step 2: Narrative physics — query_type=%s, causal_engine=%s.",
        query.query_type, cfg.use_causal_engine,
    )
    physics_result = calculate_narrative_physics(
        request=query,
        global_world_state=ws,
        temporal_anchor=cfg.temporal_anchor,
        syuzhet_anchor=cfg.syuzhet_anchor,
        use_causal_engine=cfg.use_causal_engine,
    )
    result.physics_result = physics_result

    # Extract the keys that aren't common metadata
    common_keys = {"status", "query_type", "physics_state"}
    extras = {k: v for k, v in physics_result.items() if k not in common_keys}

    history.record("narrative_physics", PhysicsStepRecord(
        query_type=physics_result.get("query_type", query.query_type),
        status=physics_result.get("status", "unknown"),
        physics_state=physics_result.get("physics_state", {}),
        extra=extras,
    ))

    # =================================================================
    # Non-prose queries stop here
    # =================================================================
    if query.query_type in ("interrogate", "general"):
        logger.info(
            "[Pipeline] Query type '%s' — returning physics state (no prose).",
            query.query_type,
        )
        return result

    # =================================================================
    # Evaluation query — full-story quality audit
    # =================================================================
    if query.query_type == "evaluate":
        logger.info("[Pipeline] Evaluation query — running full-story quality audit.")
        # Collect all prose from versioned model history
        all_prose_parts: list[str] = []
        if vwm.history:
            for entry in vwm.history:
                if hasattr(entry, 'prose') and entry.prose:
                    all_prose_parts.append(entry.prose)
        # If no history prose, try to generate a summary from events
        if not all_prose_parts:
            events_summary = "; ".join(
                f"[{e.id}] {e.description}" for e in ws.events[:50]
            )
            all_prose_parts.append(
                f"Story events summary: {events_summary}"
            )
        full_prose = "\n\n---\n\n".join(all_prose_parts)

        # Build an assembler for affective scoring
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        focus_ids = (
            query.focus_entity_ids
            if query.focus_entity_ids
            else list(ws.entities.keys())[:5]
        )
        ego_graph = extract_ego_graph_from_memory(ws, focus_ids)
        eval_assembler = DirectiveAssembler(
            sandbox=None, ego_payload=ego_graph.model_dump(), world_state=ws,
        )

        # Build a minimal brief for the evaluation
        eval_brief = CreativeBrief(
            target_effect="observation",
            target_entities=focus_ids,
            scene_context=physics_result.get("physics_state", {}),
        )
        # Populate analytics on the brief
        eval_brief.epistemic_gaps = eval_assembler.compute_epistemic_gaps(focus_ids)
        eval_brief.narrative_tensions = eval_assembler.compute_narrative_tension()
        eval_brief.trait_trajectories = eval_assembler.compute_trait_trajectories(focus_ids)

        # Compute engine metrics
        causal_fb = compute_causal_feedback(None, eval_brief, ws)
        affective_fb = compute_affective_feedback(eval_brief, eval_assembler, focus_ids)

        # Run LLM evaluation
        from shadow_loom.auditor import AuditorConfig as _AC
        eval_cfg = cfg.auditor_config or _AC()
        narrative_order = _finalize_narrative_order(
            prose=full_prose,
            brief=eval_brief,
            causal_feedback=causal_fb,
            affective_feedback=affective_fb,
            config=eval_cfg,
        )

        eval_result = EvaluationResult(
            narrative_order=narrative_order.model_dump(),
            story_prose_evaluated=full_prose if query.include_full_prose else "",
            version_count=len(all_prose_parts),
        )
        result.evaluation_result = eval_result
        result.prose = full_prose

        history.record("evaluation", {
            "overall_pass": narrative_order.overall_pass,
            "foreshadowing_score": narrative_order.causal_feedback.foreshadowing_payoff_score,
            "affective_loss": narrative_order.affective_feedback.affective_loss_mse,
            "miracle_steps": len(narrative_order.causal_feedback.miracle_steps_detected),
        })

        return result

    # =================================================================
    # Manual edit: skip physics/generation/audit — prose is user-supplied
    # =================================================================
    if query.query_type == "manual_edit":
        logger.info("[Pipeline] Manual edit — skipping generation, running re-extraction.")
        result.prose = query.edited_prose

        # Create a minimal GeneratedScene wrapper for consistency
        from shadow_loom.generation import GeneratedScene
        result.scene = GeneratedScene(
            prose=query.edited_prose,
            scene_summary=query.description or "Manual edit",
        )
        history.record("generation", GenerationStepRecord(
            scene=result.scene,
            brief=None,
        ))

        # Always run re-extraction for manual edits (the whole point)
        logger.info("[Pipeline] Extracting topology from manually edited prose.")
        try:
            topology = extract_topology_from_prose(
                prose=result.prose,
                world_state=ws,
                config=cfg.extraction_config,
            )
            description = (
                f"Manual edit: {query.description}"
                if query.description
                else "Manual edit"
            )
            vwm_next = vwm.merge(
                topology,
                source="manual_edit",
                description=description,
            )
            changeset = vwm_next.history[-1].changeset
            history.record("reextraction_merge", ReextractionStepRecord(
                events_added=changeset.events_added if changeset else 0,
                causal_edges_added=changeset.causal_edges_added if changeset else 0,
                entity_updates_applied=changeset.entity_updates_applied if changeset else 0,
                entity_updates_skipped=changeset.entity_updates_skipped if changeset else [],
                new_version=vwm_next.version,
            ))
            result.world_model = vwm_next
            logger.info(
                "[Pipeline] Manual edit merge complete — v%d → v%d.",
                vwm.version, vwm_next.version,
            )
        except Exception:
            logger.exception("[Pipeline] Manual edit re-extraction/merge failed.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})

        return result

    # =================================================================
    # Step 3–4: Brief Assembly + Generation
    # =================================================================
    physics_state = physics_result.get("physics_state", {})

    # For directive queries with causal engine, the brief is already built
    brief: CreativeBrief | None = None
    if query.query_type == "directive" and "creative_brief" in physics_result:
        brief_data = physics_result["creative_brief"]
        brief = CreativeBrief(**brief_data) if isinstance(brief_data, dict) else brief_data

    if cfg.skip_audit:
        # Generate once, no audit loop
        logger.info("[Pipeline] Steps 3–4: Generating prose (audit skipped).")
        gen_cfg = cfg.generation_config or GenerationConfig()
        scene = render_from_query(query, physics_result, ws, gen_cfg)

        history.record("generation", GenerationStepRecord(
            scene=scene,
            brief=brief,
        ))
        result.scene = scene
        result.prose = scene.prose

    else:
        # =============================================================
        # Steps 3–5: Brief → Render → Audit loop
        # =============================================================
        logger.info("[Pipeline] Steps 3–5: Generation + audit loop.")

        if brief is not None:
            # Directive with pre-built brief — use render_and_audit
            # Try to reconstruct the assembler for engine metrics
            _assembler = None
            if query.query_type == "directive":
                try:
                    from shadow_loom.extract_graph import extract_ego_graph_from_memory
                    _ego = extract_ego_graph_from_memory(ws, brief.target_entities)
                    _assembler = DirectiveAssembler(
                        sandbox=None, ego_payload=_ego.model_dump(), world_state=ws,
                    )
                except Exception:
                    logger.debug("[Pipeline] Could not build assembler for engine metrics.")

            feedback = render_and_audit(
                brief=brief,
                world_state=ws,
                auditor_config=cfg.auditor_config,
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
                assembler=_assembler,
            )
        else:
            # Non-directive: render first, then audit
            gen_cfg = cfg.generation_config or GenerationConfig()
            initial_scene = render_from_query(query, physics_result, ws, gen_cfg)

            # Build a brief for the auditor from the query
            brief = _build_brief_for_query(query, physics_result, ws)

            from shadow_loom.auditor import run_feedback_loop
            feedback = run_feedback_loop(
                initial_scene=initial_scene,
                brief=brief,
                world_state=ws,
                auditor_config=cfg.auditor_config,
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
            )

        history.record("generation", GenerationStepRecord(
            scene=feedback.final_scene,
            brief=brief,
        ))
        history.record("audit", AuditStepRecord(feedback_result=feedback))

        result.scene = feedback.final_scene
        result.prose = feedback.final_scene.prose
        result.converged = feedback.converged
        result.audit_iterations = feedback.iterations
        result.feedback_result = feedback

    # =================================================================
    # Steps 6–7: Prose re-extraction + merge
    # =================================================================
    if cfg.skip_reextraction:
        logger.info("[Pipeline] Steps 6–7: Re-extraction skipped.")
    else:
        logger.info("[Pipeline] Steps 6–7: Extracting topology from prose and merging.")
        try:
            topology = extract_topology_from_prose(
                prose=result.prose,
                world_state=ws,
                config=cfg.extraction_config,
            )
            description = (
                f"Pipeline merge after {query.query_type} query"
                f" (audit={'converged' if result.converged else 'skipped/failed'})"
            )
            vwm_next = vwm.merge(
                topology,
                source="pipeline",
                description=description,
            )
            # Record the changeset
            changeset = vwm_next.history[-1].changeset
            history.record("reextraction_merge", ReextractionStepRecord(
                events_added=changeset.events_added if changeset else 0,
                causal_edges_added=changeset.causal_edges_added if changeset else 0,
                entity_updates_applied=changeset.entity_updates_applied if changeset else 0,
                entity_updates_skipped=changeset.entity_updates_skipped if changeset else [],
                new_version=vwm_next.version,
            ))
            result.world_model = vwm_next
            logger.info(
                "[Pipeline] Merge complete — v%d → v%d (+%d events).",
                vwm.version, vwm_next.version,
                changeset.events_added if changeset else 0,
            )
        except Exception:
            logger.exception("[Pipeline] Re-extraction/merge failed — returning unmerged model.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})

    logger.info("[Pipeline] Complete — query_type=%s, prose=%s.",
                query.query_type, "yes" if result.prose else "no")
    return result


async def run_pipeline_async(
    query: UserRequest,
    *,
    world_state: WorldStateV1 | None = None,
    versioned_model: VersionedWorldModel | None = None,
    raw_text: str | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Async variant of :func:`run_pipeline`.

    Uses :func:`run_extraction_async` for parallel ingestion when
    ``raw_text`` is supplied.  All other stages are identical to the
    synchronous pipeline.
    """
    cfg = config or PipelineConfig()
    history = PipelineHistory()
    result = PipelineResult(query_type=query.query_type, history=history)

    # =================================================================
    # Step 0: Resolve the world model (async ingestion when raw_text)
    # =================================================================
    sources = sum([world_state is not None, versioned_model is not None, raw_text is not None])
    if sources == 0:
        raise ValueError("Supply one of: world_state, versioned_model, or raw_text.")
    if sources > 1:
        raise ValueError("Supply only one of: world_state, versioned_model, or raw_text.")

    if raw_text is not None:
        logger.info("[Pipeline·Async] Step 1: Ingesting raw text (%d chars).", len(raw_text))
        ing_cfg = cfg.ingestion_config or ExtractionConfig()
        ws, validation_report = await run_extraction_async(raw_text, config=ing_cfg)
        vwm = VersionedWorldModel.from_world_state(ws, max_snapshots=cfg.max_snapshots)
        history.record("ingestion", IngestionStepRecord(
            is_valid=validation_report.is_valid,
            num_events=len(ws.events),
            num_entities=len(ws.entities),
            num_locations=len(ws.locations),
        ))
        logger.info(
            "[Pipeline·Async] Ingestion complete — %d events, %d entities, valid=%s.",
            len(ws.events), len(ws.entities), validation_report.is_valid,
        )
    elif versioned_model is not None:
        vwm = versioned_model
        ws = vwm.current
    else:
        vwm = VersionedWorldModel.from_world_state(world_state, max_snapshots=cfg.max_snapshots)
        ws = vwm.current

    result.world_model = vwm

    # Remaining steps are identical to sync pipeline — delegate
    # (physics, generation, audit, re-extraction are all sync)
    logger.info(
        "[Pipeline·Async] Step 2: Narrative physics — query_type=%s, causal_engine=%s.",
        query.query_type, cfg.use_causal_engine,
    )
    physics_result = calculate_narrative_physics(
        request=query,
        global_world_state=ws,
        temporal_anchor=cfg.temporal_anchor,
        syuzhet_anchor=cfg.syuzhet_anchor,
        use_causal_engine=cfg.use_causal_engine,
    )
    result.physics_result = physics_result

    common_keys = {"status", "query_type", "physics_state"}
    extras = {k: v for k, v in physics_result.items() if k not in common_keys}
    history.record("narrative_physics", PhysicsStepRecord(
        query_type=physics_result.get("query_type", query.query_type),
        status=physics_result.get("status", "unknown"),
        physics_state=physics_result.get("physics_state", {}),
        extra=extras,
    ))

    if query.query_type in ("interrogate", "general"):
        return result

    # Manual edit: same as sync path
    if query.query_type == "manual_edit":
        logger.info("[Pipeline·Async] Manual edit — skipping generation, running re-extraction.")
        result.prose = query.edited_prose
        from shadow_loom.generation import GeneratedScene
        result.scene = GeneratedScene(
            prose=query.edited_prose,
            scene_summary=query.description or "Manual edit",
        )
        history.record("generation", GenerationStepRecord(scene=result.scene, brief=None))
        try:
            topology = extract_topology_from_prose(
                prose=result.prose, world_state=ws, config=cfg.extraction_config,
            )
            description = f"Manual edit: {query.description}" if query.description else "Manual edit"
            vwm_next = vwm.merge(topology, source="manual_edit", description=description)
            changeset = vwm_next.history[-1].changeset
            history.record("reextraction_merge", ReextractionStepRecord(
                events_added=changeset.events_added if changeset else 0,
                causal_edges_added=changeset.causal_edges_added if changeset else 0,
                entity_updates_applied=changeset.entity_updates_applied if changeset else 0,
                entity_updates_skipped=changeset.entity_updates_skipped if changeset else [],
                new_version=vwm_next.version,
            ))
            result.world_model = vwm_next
        except Exception:
            logger.exception("[Pipeline·Async] Manual edit re-extraction/merge failed.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})
        return result

    # Steps 3–4: Brief + Generation (same as sync)
    physics_state = physics_result.get("physics_state", {})
    brief: CreativeBrief | None = None
    if query.query_type == "directive" and "creative_brief" in physics_result:
        brief_data = physics_result["creative_brief"]
        brief = CreativeBrief(**brief_data) if isinstance(brief_data, dict) else brief_data

    if cfg.skip_audit:
        gen_cfg = cfg.generation_config or GenerationConfig()
        scene = render_from_query(query, physics_result, ws, gen_cfg)
        history.record("generation", GenerationStepRecord(scene=scene, brief=brief))
        result.scene = scene
        result.prose = scene.prose
    else:
        if brief is not None:
            feedback = render_and_audit(
                brief=brief, world_state=ws,
                auditor_config=cfg.auditor_config,
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
            )
        else:
            gen_cfg = cfg.generation_config or GenerationConfig()
            initial_scene = render_from_query(query, physics_result, ws, gen_cfg)
            brief = _build_brief_for_query(query, physics_result, ws)
            from shadow_loom.auditor import run_feedback_loop
            feedback = run_feedback_loop(
                initial_scene=initial_scene, brief=brief, world_state=ws,
                auditor_config=cfg.auditor_config,
                generation_config=cfg.generation_config,
                query_type=query.query_type,
                physics_state=physics_state,
            )
        history.record("generation", GenerationStepRecord(scene=feedback.final_scene, brief=brief))
        history.record("audit", AuditStepRecord(feedback_result=feedback))
        result.scene = feedback.final_scene
        result.prose = feedback.final_scene.prose
        result.converged = feedback.converged
        result.audit_iterations = feedback.iterations
        result.feedback_result = feedback

    # Steps 6–7: Re-extraction + merge (same as sync)
    if cfg.skip_reextraction:
        logger.info("[Pipeline·Async] Steps 6–7: Re-extraction skipped.")
    else:
        logger.info("[Pipeline·Async] Steps 6–7: Extracting topology from prose and merging.")
        try:
            topology = extract_topology_from_prose(
                prose=result.prose, world_state=ws, config=cfg.extraction_config,
            )
            description = (
                f"Pipeline merge after {query.query_type} query"
                f" (audit={'converged' if result.converged else 'skipped/failed'})"
            )
            vwm_next = vwm.merge(topology, source="pipeline", description=description)
            changeset = vwm_next.history[-1].changeset
            history.record("reextraction_merge", ReextractionStepRecord(
                events_added=changeset.events_added if changeset else 0,
                causal_edges_added=changeset.causal_edges_added if changeset else 0,
                entity_updates_applied=changeset.entity_updates_applied if changeset else 0,
                entity_updates_skipped=changeset.entity_updates_skipped if changeset else [],
                new_version=vwm_next.version,
            ))
            result.world_model = vwm_next
        except Exception:
            logger.exception("[Pipeline·Async] Re-extraction/merge failed.")
            history.record("reextraction_merge", {"error": "extraction_or_merge_failed"})

    logger.info("[Pipeline·Async] Complete — query_type=%s, prose=%s.",
                query.query_type, "yes" if result.prose else "no")
    return result


# =====================================================================
# Internal: build a brief for non-directive queries
# =====================================================================

def _build_brief_for_query(
    query: UserRequest,
    physics_result: Dict[str, Any],
    world_state: WorldStateV1,
) -> CreativeBrief:
    """Build a CreativeBrief from a non-directive query's physics result."""
    from shadow_loom.generation import (
        build_counterfactual_brief,
        build_intervention_brief,
        build_observation_brief,
    )

    physics_state = physics_result.get("physics_state", {})

    if query.query_type == "observation":
        return build_observation_brief(query, physics_state, world_state)
    elif query.query_type == "intervention":
        return build_intervention_brief(
            query, physics_state, world_state,
            mutations=physics_result.get("mutations"),
            blocked=physics_result.get("blocked"),
        )
    elif query.query_type == "counterfactual":
        return build_counterfactual_brief(
            query, physics_state, world_state,
            hidden_deltas=physics_result.get("hidden_deltas"),
        )
    else:
        # Fallback minimal brief
        return CreativeBrief(
            target_effect="observation",
            target_entities=[],
            scene_context=physics_state,
        )
