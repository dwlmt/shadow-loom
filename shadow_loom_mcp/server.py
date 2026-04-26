"""FastMCP server for Shadow-Loom.

Exposes the full pipeline as MCP tools, including a natural-language
query interface that external LLMs can use to interact with the
narrative world model.

All state is persisted in the shared ``shadow_loom.db`` database.
Every tool accepts ``user_id`` and ``project_id`` / ``project_name``
to identify the context.

Run standalone:  python -m shadow_loom_mcp.server
Or as a library:  from shadow_loom_mcp.server import mcp; mcp.run()
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from shadow_loom.db import (
    create_project,
    find_project_by_name,
    get_latest_version,
    get_project,
    get_version,
    get_version_children,
    get_version_lineage,
    get_version_tree,
    init_db,
    list_projects as db_list_projects,
    list_versions,
    save_version,
    upsert_user,
)
from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ExtractionConfig, run_extraction
from shadow_loom.models import WorldStateV1
from shadow_loom.pipeline import PipelineConfig, PipelineResult, run_pipeline
from shadow_loom.query_models import (
    CounterfactualQuery,
    DirectiveQuery,
    EvaluationQuery,
    GeneralQuery,
    InterrogationQuery,
    InterventionQuery,
    ManualEditQuery,
    ObservationQuery,
    UserRequest,
)
from shadow_loom.query_parsing import (
    QueryParseResult,
    QueryParsingConfig,
    parse_query,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Shadow Loom",
    description=(
        "Causal narrative engine — interact with story world models "
        "using natural language or structured queries.  All operations "
        "are project-scoped and version-tracked."
    ),
)

# =====================================================================
# DB initialisation (runs once on import / startup)
# =====================================================================

_DB_URL = os.environ.get("DATABASE_URL", "sqlite:///shadow_loom.db")
init_db(_DB_URL)


# =====================================================================
# Internal helpers
# =====================================================================


def _resolve_project(
    project_id: Optional[int],
    project_name: Optional[str],
    user_id: Optional[str],
) -> Optional[int]:
    """Resolve a project by id or name.  Returns project_id or None."""
    if project_id is not None:
        return project_id
    if project_name:
        owner_id = _resolve_user_row_id(user_id) if user_id else None
        proj = find_project_by_name(project_name, owner_id=owner_id)
        if proj is None:
            proj = find_project_by_name(project_name)
        return proj.id if proj else None
    return None


def _resolve_user_row_id(user_id: str) -> Optional[int]:
    """Ensure the user exists in the DB and return their row id."""
    if not user_id or user_id == "anonymous":
        return None
    row = upsert_user(
        provider="mcp",
        provider_id=f"mcp:{user_id}",
        username=user_id,
    )
    return row.id


def _load_world_state(
    project_id: int,
    version: Optional[int] = None,
) -> tuple[Optional[WorldStateV1], Optional[int]]:
    """Load a world state from the DB.

    Returns (world_state, version_row_id) or (None, None).
    """
    if version is not None:
        ver = get_version(project_id, version)
    else:
        ver = get_latest_version(project_id)
    if ver is None:
        return None, None
    ws = WorldStateV1.model_validate_json(ver.world_state_json)
    return ws, ver.id


def _run_and_save(
    query: UserRequest,
    project_id: int,
    world_state: WorldStateV1,
    ancestor_row_id: int | None,
    user_row_id: int | None,
    raw_query: str | None,
    skip_audit: bool = True,
    skip_reextraction: bool = True,
) -> dict:
    """Run pipeline on *world_state*, save new version, return response."""
    ws = world_state

    vwm = VersionedWorldModel.from_world_state(ws)
    cfg = PipelineConfig(
        skip_audit=skip_audit,
        skip_reextraction=skip_reextraction,
    )

    try:
        result: PipelineResult = run_pipeline(query, versioned_model=vwm, config=cfg)
    except Exception as e:
        return {"error": f"Pipeline failed: {e}"}

    new_ws = result.world_model.current if result.world_model else ws

    changeset_json = None
    if result.world_model and result.world_model.history:
        last = result.world_model.history[-1]
        if last.changeset:
            changeset_json = last.changeset.model_dump_json()

    ver = save_version(
        project_id=project_id,
        world_state_json=new_ws.model_dump_json(),
        ancestor_id=ancestor_row_id,
        source=query.query_type,
        description=f"{query.query_type} query",
        changeset_json=changeset_json,
        raw_query=raw_query,
        parsed_query_json=query.model_dump_json(),
        prose=result.prose,
        user_id=user_row_id,
    )

    response: dict[str, Any] = {
        "project_id": project_id,
        "version": ver.version,
        "version_row_id": ver.id,
        "query_type": query.query_type,
    }
    if result.prose:
        response["prose"] = result.prose
    if result.physics_result:
        response["physics_status"] = result.physics_result.get("status", "unknown")
        answer = result.physics_result.get("answer")
        if answer:
            response["answer"] = answer
    if result.converged is not None:
        response["audit_converged"] = result.converged
        response["audit_iterations"] = result.audit_iterations
    if result.feedback_result and result.feedback_result.change_impact:
        response["change_impact"] = result.feedback_result.change_impact.model_dump()
    if result.evaluation_result:
        response["evaluation"] = result.evaluation_result.model_dump()
    if result.world_model:
        response["world_model_version"] = result.world_model.version

    return response


# =====================================================================
# Tool: Natural Language Query
# =====================================================================


@mcp.tool()
def query_natural_language(
    question: str,
    query_type: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    user_id: str = "anonymous",
    version: Optional[int] = None,
    skip_audit: bool = True,
    skip_reextraction: bool = True,
) -> str:
    """Ask anything about the story world in natural language.

    This is the primary interface.  Your question is automatically
    parsed and executed through the full pipeline.

    Args:
        question: Your natural language question or command.
        query_type: The type of query — one of: observation, intervention,
                    counterfactual, directive, interrogate, general, manual_edit.
        project_id: Project to query (by ID).
        project_name: Project to query (by name).
        user_id: Identifier for the user.
        version: Specific version to load (default: latest).
        skip_audit: Skip the audit loop (default True).
        skip_reextraction: Skip re-extracting topology (default True).

    Returns:
        JSON with parse diagnostics and pipeline results.
    """
    pid = _resolve_project(project_id, project_name, user_id)
    if pid is None:
        return json.dumps({"error": "Project not found. Specify project_id or project_name."})

    ws, ancestor_row_id = _load_world_state(pid, version)
    if ws is None:
        return json.dumps({"error": "No world model found. Use 'ingest_text' first."})

    parse_result: QueryParseResult = parse_query(
        question, query_type=query_type, world_state=ws, config=QueryParsingConfig(),
    )

    if not parse_result.is_valid or parse_result.query is None:
        return json.dumps({
            "error": "Query validation failed",
            "query_type": parse_result.parsed.query_type if parse_result.parsed else None,
            "reasoning": parse_result.parsed.reasoning if parse_result.parsed else None,
            "validation_errors": [
                {"field": e.field, "message": e.message, "severity": e.severity}
                for e in parse_result.validation_errors
            ],
            "fallback": parse_result.fallback.model_dump() if parse_result.fallback else None,
        })

    user_row_id = _resolve_user_row_id(user_id)
    response = _run_and_save(
        query=parse_result.query,
        project_id=pid,
        world_state=ws,
        ancestor_row_id=ancestor_row_id,
        user_row_id=user_row_id,
        raw_query=question,
        skip_audit=skip_audit,
        skip_reextraction=skip_reextraction,
    )

    if parse_result.parsed:
        response["reasoning"] = parse_result.parsed.reasoning
        response["resolved_ids"] = [
            {"name": r.natural_name, "id": r.resolved_id, "confidence": r.confidence}
            for r in parse_result.parsed.resolved_ids
        ]
    if parse_result.fallback:
        response["fallback"] = {
            "strategy": parse_result.fallback.strategy,
            "reason": parse_result.fallback.reason,
        }

    return json.dumps(response, default=str)


# =====================================================================
# Tool: Ingest raw text
# =====================================================================


@mcp.tool()
def ingest_text(
    text: str,
    project_name: str = "MCP Project",
    user_id: str = "anonymous",
    label: Optional[str] = None,
) -> str:
    """Ingest raw narrative text to create a new project and world model.

    Args:
        text: The full narrative text to ingest.
        project_name: Name for this project/story.
        user_id: Identifier for the user.
        label: Optional label/tag for the project.

    Returns:
        JSON summary of the extracted world model.
    """
    config = ExtractionConfig(
        chunk_strategy="act_headings",
        fabula_time_spacing=100,
        output_retries=5,
        max_correction_retries=1,
    )

    try:
        ws, report = run_extraction(text, config)
    except Exception as e:
        return json.dumps({"error": f"Ingestion failed: {e}"})

    user_row_id = _resolve_user_row_id(user_id)
    proj = create_project(
        name=project_name, owner_id=user_row_id, label=label, raw_text=text,
    )

    save_version(
        project_id=proj.id,
        world_state_json=ws.model_dump_json(),
        version=0,
        source="ingestion",
        description="Initial ingestion",
        user_id=user_row_id,
    )

    return json.dumps({
        "project_id": proj.id,
        "project_name": project_name,
        "version": 0,
        "entities": len(ws.entities),
        "locations": len(ws.locations),
        "objects": len(ws.objects),
        "world_traits": len(ws.world_traits),
        "events": len(ws.events),
        "causal_edges": len(ws.causal_topology),
        "spatial_edges": len(ws.spatial_topology),
        "social_edges": len(ws.social_topology),
        "information_edges": len(ws.information_topology),
        "is_valid": report.is_valid,
        "issues": len(report.issues),
    })


# =====================================================================
# Tool: Manual Edit
# =====================================================================


@mcp.tool()
def manual_edit(
    edited_prose: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    user_id: str = "anonymous",
    version: Optional[int] = None,
    description: str = "",
) -> str:
    """Apply user-written prose as a manual edit to the world model.

    The prose is re-extracted into topology and merged.  Skips physics,
    LLM generation, and audit.

    Args:
        edited_prose: The user's narrative prose text.
        project_id: Project to edit (by ID).
        project_name: Project to edit (by name).
        user_id: Identifier for the user.
        version: Version to branch from (default: latest).
        description: Description of the edit.

    Returns:
        JSON with the new version info.
    """
    pid = _resolve_project(project_id, project_name, user_id)
    if pid is None:
        return json.dumps({"error": "Project not found."})

    ws, ancestor_row_id = _load_world_state(pid, version)
    if ws is None:
        return json.dumps({"error": "No world model found for this project."})

    user_row_id = _resolve_user_row_id(user_id)
    query = ManualEditQuery(edited_prose=edited_prose, description=description)

    response = _run_and_save(
        query=query, project_id=pid, world_state=ws,
        ancestor_row_id=ancestor_row_id,
        user_row_id=user_row_id, raw_query=edited_prose,
        skip_audit=True, skip_reextraction=False,
    )

    return json.dumps(response, default=str)


# =====================================================================
# Tool: Run structured query
# =====================================================================


@mcp.tool()
def run_structured_query(
    query_type: str,
    query_params: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    user_id: str = "anonymous",
    version: Optional[int] = None,
    skip_audit: bool = True,
    skip_reextraction: bool = True,
) -> str:
    """Run a pre-built structured query through the pipeline.

    Args:
        query_type: One of: observation, intervention, counterfactual,
                    directive, interrogate, general, manual_edit.
        query_params: JSON string with the query parameters.
        project_id: Project (by ID).
        project_name: Project (by name).
        user_id: Identifier for the user.
        version: Version to load (default: latest).
        skip_audit: Skip audit loop.
        skip_reextraction: Skip re-extraction.

    Returns:
        JSON pipeline result.
    """
    pid = _resolve_project(project_id, project_name, user_id)
    if pid is None:
        return json.dumps({"error": "Project not found."})

    ws, ancestor_row_id = _load_world_state(pid, version)
    if ws is None:
        return json.dumps({"error": "No world model found."})

    params = json.loads(query_params)
    query_map: dict[str, type] = {
        "observation": ObservationQuery,
        "intervention": InterventionQuery,
        "counterfactual": CounterfactualQuery,
        "directive": DirectiveQuery,
        "interrogate": InterrogationQuery,
        "general": GeneralQuery,
        "manual_edit": ManualEditQuery,
        "evaluate": EvaluationQuery,
    }

    cls = query_map.get(query_type)
    if cls is None:
        return json.dumps({"error": f"Unknown query type: {query_type}"})

    query = cls(**params)
    user_row_id = _resolve_user_row_id(user_id)

    response = _run_and_save(
        query=query, project_id=pid, world_state=ws,
        ancestor_row_id=ancestor_row_id,
        user_row_id=user_row_id, raw_query=json.dumps(params),
        skip_audit=skip_audit, skip_reextraction=skip_reextraction,
    )

    return json.dumps(response, default=str)


# =====================================================================
# Tool: List projects (metadata)
# =====================================================================


@mcp.tool()
def list_projects(user_id: str = "anonymous") -> str:
    """List projects available to a user (own + example projects).

    Args:
        user_id: Identifier for the user.

    Returns:
        JSON array of project summaries.
    """
    user_row_id = _resolve_user_row_id(user_id)
    projects = db_list_projects(user_id=user_row_id)
    return json.dumps(projects, default=str)


# =====================================================================
# Tool: Get project info
# =====================================================================


@mcp.tool()
def get_project_info(project_id: int) -> str:
    """Get detailed info about a specific project.

    Args:
        project_id: The project ID.

    Returns:
        JSON with project details and version count.
    """
    proj = get_project(project_id)
    if proj is None:
        return json.dumps({"error": f"Project {project_id} not found."})

    ver = get_latest_version(project_id)
    versions = list_versions(project_id)

    return json.dumps({
        "id": proj.id,
        "name": proj.name,
        "label": proj.label,
        "owner_id": proj.owner_id,
        "created_at": str(proj.created_at),
        "updated_at": str(proj.updated_at),
        "version_count": len(versions),
        "latest_version": ver.version if ver else None,
        "has_raw_text": proj.raw_text is not None,
    })


# =====================================================================
# Tool: Get world state summary
# =====================================================================


@mcp.tool()
def get_world_summary(
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    user_id: str = "anonymous",
    version: Optional[int] = None,
) -> str:
    """Get a summary of the world model for a project.

    Args:
        project_id: Project (by ID).
        project_name: Project (by name).
        user_id: User identifier.
        version: Specific version (default: latest).

    Returns:
        JSON with entity/location/event counts and topology stats.
    """
    pid = _resolve_project(project_id, project_name, user_id)
    if pid is None:
        return json.dumps({"error": "Project not found."})

    ws, _ = _load_world_state(pid, version)
    if ws is None:
        return json.dumps({"error": "No world model found."})

    proj = get_project(pid)
    return json.dumps({
        "project_id": pid,
        "project_name": proj.name if proj else "",
        "entities": {eid: ent.name for eid, ent in ws.entities.items()},
        "locations": {lid: loc.name for lid, loc in ws.locations.items()},
        "objects": {oid: obj.name for oid, obj in ws.objects.items()},
        "world_traits": {wid: wt.name for wid, wt in ws.world_traits.items()},
        "event_count": len(ws.events),
        "causal_edges": len(ws.causal_topology),
        "spatial_edges": len(ws.spatial_topology),
        "social_edges": len(ws.social_topology),
        "information_edges": len(ws.information_topology),
    })


# =====================================================================
# Tool: Get entity details
# =====================================================================


@mcp.tool()
def get_entity(
    entity_id: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    user_id: str = "anonymous",
    version: Optional[int] = None,
) -> str:
    """Get detailed information about a specific entity.

    Args:
        entity_id: The entity ID (e.g., ENT_MACBETH).
        project_id: Project (by ID).
        project_name: Project (by name).
        user_id: User identifier.
        version: Specific version (default: latest).

    Returns:
        JSON with entity details including traits, beliefs, timeline.
    """
    pid = _resolve_project(project_id, project_name, user_id)
    if pid is None:
        return json.dumps({"error": "Project not found."})

    ws, _ = _load_world_state(pid, version)
    if ws is None:
        return json.dumps({"error": "No world model found."})

    ent = ws.entities.get(entity_id)
    if ent is None:
        return json.dumps({"error": f"Entity '{entity_id}' not found."})

    return ent.model_dump_json(indent=2)


# =====================================================================
# Tool: Get world trait details
# =====================================================================


@mcp.tool()
def get_world_trait(
    world_trait_id: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    user_id: str = "anonymous",
    version: Optional[int] = None,
) -> str:
    """Get detailed information about a specific world trait.

    Args:
        world_trait_id: The world trait ID (e.g., WORLD_SURVEILLANCE_STATE).
        project_id: Project (by ID).
        project_name: Project (by name).
        user_id: User identifier.
        version: Specific version (default: latest).

    Returns:
        JSON with world trait details including magnitude, domains, timeline.
    """
    pid = _resolve_project(project_id, project_name, user_id)
    if pid is None:
        return json.dumps({"error": "Project not found."})

    ws, _ = _load_world_state(pid, version)
    if ws is None:
        return json.dumps({"error": "No world model found."})

    wt = ws.world_traits.get(world_trait_id)
    if wt is None:
        return json.dumps({"error": f"World trait '{world_trait_id}' not found."})

    return wt.model_dump_json(indent=2)


# =====================================================================
# Tool: Get relationships
# =====================================================================


@mcp.tool()
def get_relationships(
    entity_id: Optional[str] = None,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    user_id: str = "anonymous",
    version: Optional[int] = None,
) -> str:
    """Get social relationships, optionally filtered by entity.

    Args:
        entity_id: Optional entity ID to filter by.
        project_id: Project (by ID).
        project_name: Project (by name).
        user_id: User identifier.
        version: Specific version (default: latest).

    Returns:
        JSON array of relationship edges.
    """
    pid = _resolve_project(project_id, project_name, user_id)
    if pid is None:
        return json.dumps({"error": "Project not found."})

    ws, _ = _load_world_state(pid, version)
    if ws is None:
        return json.dumps({"error": "No world model found."})

    edges = ws.social_topology
    if entity_id:
        edges = [
            e for e in edges
            if e.source_entity_id == entity_id or e.target_entity_id == entity_id
        ]

    return json.dumps([e.model_dump() for e in edges], default=str)


# =====================================================================
# Tool: Version tree
# =====================================================================


@mcp.tool()
def query_version_tree(project_id: int) -> str:
    """Get the full version tree for a project.

    Args:
        project_id: The project ID.

    Returns:
        JSON array of version nodes with tree structure.
    """
    tree = get_version_tree(project_id)
    if not tree:
        return json.dumps({"error": "No versions found for this project."})
    return json.dumps(tree, default=str)


# =====================================================================
# Tool: Version detail
# =====================================================================


@mcp.tool()
def query_version_detail(project_id: int, version: int) -> str:
    """Get full details for a specific version including changeset, query, and prose.

    Args:
        project_id: The project ID.
        version: The version number.

    Returns:
        JSON with full version record.
    """
    ver = get_version(project_id, version)
    if ver is None:
        return json.dumps({"error": f"Version {version} not found."})

    changeset = None
    if ver.changeset_json:
        try:
            changeset = json.loads(ver.changeset_json)
        except (json.JSONDecodeError, TypeError):
            pass

    parsed_query = None
    if ver.parsed_query_json:
        try:
            parsed_query = json.loads(ver.parsed_query_json)
        except (json.JSONDecodeError, TypeError):
            pass

    return json.dumps({
        "id": ver.id,
        "project_id": ver.project_id,
        "version": ver.version,
        "ancestor_id": ver.ancestor_id,
        "source": ver.source,
        "description": ver.description,
        "changeset": changeset,
        "raw_query": ver.raw_query,
        "parsed_query": parsed_query,
        "prose": ver.prose,
        "user_id": ver.user_id,
        "created_at": str(ver.created_at),
    }, default=str)


# =====================================================================
# Tool: Version lineage
# =====================================================================


@mcp.tool()
def query_version_lineage(project_id: int, version: int) -> str:
    """Get the ancestor chain from root to a specific version.

    Args:
        project_id: The project ID.
        version: The version number to trace back from.

    Returns:
        JSON array ordered root → target version.
    """
    lineage = get_version_lineage(project_id, version)
    if not lineage:
        return json.dumps({"error": f"Version {version} not found."})
    return json.dumps(lineage, default=str)


# =====================================================================
# Tool: Rollback (branch from previous version)
# =====================================================================


@mcp.tool()
def rollback_world_model(
    project_id: int,
    target_version: int,
    user_id: str = "anonymous",
) -> str:
    """Create a new version that branches from a previous version.

    Does not delete history — creates a new branch point.

    Args:
        project_id: The project ID.
        target_version: The version number to branch from.
        user_id: User identifier.

    Returns:
        JSON with the new version info.
    """
    ws, ancestor_row_id = _load_world_state(project_id, target_version)
    if ws is None:
        return json.dumps({"error": f"Version {target_version} not found."})

    user_row_id = _resolve_user_row_id(user_id)

    ver = save_version(
        project_id=project_id,
        world_state_json=ws.model_dump_json(),
        ancestor_id=ancestor_row_id,
        source="rollback",
        description=f"Branched from v{target_version}",
        user_id=user_row_id,
    )

    return json.dumps({
        "project_id": project_id,
        "new_version": ver.version,
        "branched_from": target_version,
        "version_row_id": ver.id,
    })


# =====================================================================
# Tool: Load world state from JSON
# =====================================================================


@mcp.tool()
def load_world_state_json(
    world_state_json: str,
    project_id: Optional[int] = None,
    project_name: str = "Imported",
    user_id: str = "anonymous",
) -> str:
    """Load a world model from a JSON string.

    If project_id is given, adds a new version.  Otherwise creates a
    new project.

    Args:
        world_state_json: JSON representation of a WorldStateV1.
        project_id: Existing project to add version to (optional).
        project_name: Name for new project (if project_id not given).
        user_id: User identifier.

    Returns:
        JSON summary of the loaded model.
    """
    try:
        ws = WorldStateV1.model_validate_json(world_state_json)
    except Exception as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    user_row_id = _resolve_user_row_id(user_id)

    if project_id is None:
        proj = create_project(name=project_name, owner_id=user_row_id)
        project_id = proj.id

    latest = get_latest_version(project_id)
    ancestor_id = latest.id if latest else None

    ver = save_version(
        project_id=project_id,
        world_state_json=ws.model_dump_json(),
        ancestor_id=ancestor_id,
        source="import",
        description="Loaded from JSON",
        user_id=user_row_id,
    )

    return json.dumps({
        "project_id": project_id,
        "version": ver.version,
        "entities": len(ws.entities),
        "locations": len(ws.locations),
        "world_traits": len(ws.world_traits),
        "events": len(ws.events),
    })


# =====================================================================
# Resources
# =====================================================================


@mcp.resource("world://current/{project_id}")
def get_current_world(project_id: int) -> str:
    """The current world state for a project as JSON."""
    ws, _ = _load_world_state(project_id)
    if ws is None:
        return json.dumps({"error": "No world model loaded."})
    return ws.model_dump_json(indent=2)


@mcp.resource("world://versions/{project_id}")
def get_version_resource(project_id: int) -> str:
    """Version tree for a project."""
    tree = get_version_tree(project_id)
    return json.dumps(tree, default=str)


# =====================================================================
# Entry point
# =====================================================================


@mcp.tool()
def evaluate_story(
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    user_id: str = "anonymous",
    version: Optional[int] = None,
    focus_entity_ids: Optional[List[str]] = None,
) -> str:
    """Run a full-story quality evaluation using the NarrativeOrderObject scorecard.

    Computes engine-derived causal physics and affective metrics, plus an
    LLM-generated literary critique.  Returns a structured scorecard with
    overall pass/fail.

    Args:
        project_id: Project (by ID).
        project_name: Project (by name).
        user_id: Identifier for the user.
        version: Specific version to evaluate (default: latest).
        focus_entity_ids: Entities to focus on (default: all).

    Returns:
        JSON with NarrativeOrderObject scorecard and evaluation details.
    """
    pid = _resolve_project(project_id, project_name, user_id)
    if pid is None:
        return json.dumps({"error": "Project not found."})

    ws, ancestor_row_id = _load_world_state(pid, version)
    if ws is None:
        return json.dumps({"error": "No world model found."})

    query = EvaluationQuery(
        focus_entity_ids=focus_entity_ids or [],
        include_full_prose=True,
    )
    user_row_id = _resolve_user_row_id(user_id)

    response = _run_and_save(
        query=query,
        project_id=pid,
        world_state=ws,
        ancestor_row_id=ancestor_row_id,
        user_row_id=user_row_id,
        raw_query="evaluate_story",
        skip_audit=True,
        skip_reextraction=True,
    )

    return json.dumps(response, default=str)


if __name__ == "__main__":
    mcp.run()
