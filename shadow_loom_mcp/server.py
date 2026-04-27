"""Shadow-Loom MCP server — agent-first narrative intelligence.

20 tools grouped by cognitive task:
  ORIENT  (2) — list_projects, open_project
  EXPLORE (5) — inspect, search, get_relationships, trace_causality, get_history
  REASON  (3) — ask, compute_tension, diff_versions
  CREATE  (4) — narrate, direct, write, ingest
  JUDGE   (2) — evaluate, audit_log
  MANAGE  (4) — branch, share, fork, update_project

5 resources:
  world://projects
  world://project/{id}
  world://project/{id}/world
  world://project/{id}/entity/{eid}
  world://project/{id}/versions

Auth: bearer token (API key from DB). Scopes: read, write, admin.

Run:  python -m shadow_loom_mcp
"""

from __future__ import annotations

import json
import functools
import inspect as _inspect
import logging
import os
from difflib import SequenceMatcher
from typing import Any, List, Optional

from fastmcp import Context, FastMCP

from shadow_loom.db import (
    add_project_member,
    create_project,
    fork_project,
    get_all_prose,
    get_latest_version,
    get_project,
    get_project_activity,
    get_user_project_role,
    get_version,
    get_version_tree,
    init_db,
    list_projects as db_list_projects,
    list_versions,
    save_version,
    search_users,
    update_project,
)
from shadow_loom.extract_graph import VersionedWorldModel
from shadow_loom.ingestion import ExtractionConfig, run_extraction
from shadow_loom.models import (
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
)
from shadow_loom.narrative_physics import calculate_narrative_physics
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
)
from shadow_loom.query_parsing import QueryParsingConfig, parse_query

from shadow_loom_mcp.auth import (
    check_project_access,
    get_user_id,
    require_scope,
    verifier,
)
from shadow_loom_mcp.helpers import load_world_state, resolve_project, run_and_save

logger = logging.getLogger(__name__)

# ── DB init ───────────────────────────────────────────────────────

_DB_URL = os.environ.get("DATABASE_URL", "sqlite:///shadow_loom.db")
init_db(_DB_URL)

# ── Server ────────────────────────────────────────────────────────

mcp = FastMCP(
    "Shadow Loom",
    instructions=(
        "Causal narrative engine for interactive fiction. Interact with "
        "story world models via natural language or structured queries. "
        "All operations are project-scoped and version-tracked. "
        "Authenticate with a Bearer token (API key).\n\n"
        "Workflow: list_projects → open_project → inspect/ask/compute_tension "
        "→ narrate/direct/write → evaluate."
    ),
    auth=verifier,
)


# ── Error-handling decorator ──────────────────────────────────────

def _safe_tool(fn):
    """Wrap an MCP tool function so unhandled exceptions return an error dict."""
    if _inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Tool %s failed", fn.__name__)
                return {"error": f"{fn.__name__} failed: {e}"}
    else:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Tool %s failed", fn.__name__)
                return {"error": f"{fn.__name__} failed: {e}"}
    return wrapper


# =====================================================================
# GROUP 1: ORIENT — "What stories exist? What's in this one?"
# =====================================================================


@mcp.tool()
@_safe_tool
def list_projects(ctx: Context) -> dict:
    """List all projects accessible to the authenticated user.

    Returns project summaries including name, description, version count,
    and whether the project is public.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}
    user_id = get_user_id(ctx)
    projects = db_list_projects(user_id=user_id)
    return {"projects": projects}


@mcp.tool()
@_safe_tool
def open_project(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
) -> dict:
    """Open a project and get a complete world model manifest.

    Returns all entity, location, object, event, and world trait IDs
    with names, plus topology edge counts and current version info.
    Use this as the first step when working with a project.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, ver_row_id = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found. Use 'ingest' first."}

    proj = get_project(pid)
    versions = list_versions(pid)

    return {
        "project_id": pid,
        "project_name": proj.name if proj else "",
        "description": proj.description if proj else None,
        "current_version": versions[-1]["version"] if versions else 0,
        "version_count": len(versions),
        "entities": {eid: {"name": e.name, "status": e.status, "location": e.location_id}
                     for eid, e in ws.entities.items()},
        "locations": {lid: {"name": l.name} for lid, l in ws.locations.items()},
        "objects": {oid: {"name": o.name, "owner": o.owner_id, "location": o.location_id}
                    for oid, o in ws.objects.items()},
        "world_traits": {wid: {"name": t.name, "magnitude": t.magnitude.value}
                         for wid, t in ws.world_traits.items()},
        "event_count": len(ws.events),
        "topology": {
            "causal_edges": len(ws.causal_topology),
            "spatial_edges": len(ws.spatial_topology),
            "social_edges": len(ws.social_topology),
            "information_edges": len(ws.information_topology),
        },
    }


# =====================================================================
# GROUP 2: EXPLORE — "Tell me about this character / event / place."
# =====================================================================


@mcp.tool()
@_safe_tool
def inspect(
    ctx: Context,
    node_id: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
    at_time: Optional[int] = None,
) -> dict:
    """Inspect any node in the world model by ID.

    Auto-detects the node type from the ID prefix:
      ENT_  → entity (traits, beliefs, constants, state timeline)
      LOC_  → location (ambient state, connections, occupants)
      EVT_  → event (actors, targets, causal causes/effects)
      OBJ_  → object (owner, location, affordances, properties)
      WORLD_ → world trait (magnitude, domains, timeline)

    Optionally provide at_time (fabula_time) to see the reconstructed
    state at a specific point in the story's timeline.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    if node_id.startswith("ENT_"):
        return _inspect_entity(ws, node_id, at_time)
    elif node_id.startswith("LOC_"):
        return _inspect_location(ws, node_id)
    elif node_id.startswith("EVT_"):
        return _inspect_event(ws, node_id)
    elif node_id.startswith("OBJ_"):
        return _inspect_object(ws, node_id)
    elif node_id.startswith("WORLD_"):
        return _inspect_world_trait(ws, node_id, at_time)
    else:
        return {"error": f"Unknown node ID prefix: {node_id}. Expected ENT_, LOC_, EVT_, OBJ_, or WORLD_."}


def _inspect_entity(ws: WorldStateV1, eid: str, at_time: int | None) -> dict:
    ent = ws.entities.get(eid)
    if ent is None:
        return {"error": f"Entity '{eid}' not found."}

    if at_time is not None:
        snapshot = reconstruct_entity_at(ent, at_time)
        return {
            "id": eid, "name": ent.name, "type": "Entity",
            "at_time": at_time,
            "status": snapshot["status"],
            "location_id": snapshot["location_id"],
            "traits": snapshot["traits"],
            "beliefs": snapshot["beliefs"],
            "constants": ent.constants,
        }

    return {
        "id": eid, "name": ent.name, "type": "Entity",
        "status": ent.status,
        "location_id": ent.location_id,
        "traits": {k: {"value": v.value, "inertia": v.inertia} for k, v in ent.traits.items()},
        "beliefs": [b.model_dump() for b in ent.beliefs],
        "constants": ent.constants,
        "state_timeline": [s.model_dump() for s in ent.state_timeline[-10:]],
    }


def _inspect_location(ws: WorldStateV1, lid: str) -> dict:
    loc = ws.locations.get(lid)
    if loc is None:
        return {"error": f"Location '{lid}' not found."}

    occupants = [{"id": eid, "name": e.name} for eid, e in ws.entities.items()
                 if e.location_id == lid]
    connections = []
    for se in ws.spatial_topology:
        if se.source_id == lid or se.target_id == lid:
            other = se.target_id if se.source_id == lid else se.source_id
            other_name = ws.locations[other].name if other in ws.locations else other
            connections.append({"target": other, "name": other_name, "locked": se.is_locked})

    return {
        "id": lid, "name": loc.name, "type": "Location",
        "description": loc.description,
        "ambient_state": {k: {"value": v.value, "volatility": v.volatility}
                          for k, v in loc.ambient_state.items()},
        "occupants": occupants,
        "connections": connections,
    }


def _inspect_event(ws: WorldStateV1, evt_id: str) -> dict:
    evt = next((e for e in ws.events if e.id == evt_id), None)
    if evt is None:
        return {"error": f"Event '{evt_id}' not found."}

    def _resolve_names(ids):
        return [{"id": i, "name": ws.entities[i].name if i in ws.entities else i} for i in ids]

    causes = [{"source": ce.source_id, "mechanism": ce.mechanism,
               "force": ce.causal_force, "type": ce.causality_type}
              for ce in ws.causal_topology if ce.target_id == evt_id]
    effects = [{"target": ce.target_id, "mechanism": ce.mechanism,
                "force": ce.causal_force, "type": ce.causality_type}
               for ce in ws.causal_topology if ce.source_id == evt_id]

    return {
        "id": evt.id, "type": "EventNode",
        "event_type": evt.event_type,
        "fabula_time": evt.fabula_time,
        "syuzhet_index": evt.syuzhet_index,
        "description": evt.description,
        "actors": _resolve_names(evt.actor_ids),
        "targets": _resolve_names(evt.target_ids),
        "caused_by": causes,
        "causes": effects,
    }


def _inspect_object(ws: WorldStateV1, oid: str) -> dict:
    obj = ws.objects.get(oid)
    if obj is None:
        return {"error": f"Object '{oid}' not found."}

    return {
        "id": oid, "name": obj.name, "type": "NarrativeObject",
        "location_id": obj.location_id,
        "owner_id": obj.owner_id,
        "properties": obj.properties,
        "affordances": [{"action": a.action, "target_type": a.target_type} for a in obj.affordances],
    }


def _inspect_world_trait(ws: WorldStateV1, wid: str, at_time: int | None) -> dict:
    wt = ws.world_traits.get(wid)
    if wt is None:
        return {"error": f"World trait '{wid}' not found."}

    if at_time is not None:
        snapshot = reconstruct_world_trait_at(wt, at_time)
        return {
            "id": wid, "name": wt.name, "type": "WorldTrait",
            "at_time": at_time,
            "magnitude": snapshot.get("magnitude"),
            "description": snapshot.get("description"),
        }

    return {
        "id": wid, "name": wt.name, "type": "WorldTrait",
        "description": wt.description,
        "category": wt.category,
        "magnitude": {"value": wt.magnitude.value, "inertia": wt.magnitude.inertia},
        "affected_domains": wt.affected_domains,
        "state_timeline": [s.model_dump() for s in wt.state_timeline[-10:]],
    }


@mcp.tool()
@_safe_tool
def search(
    ctx: Context,
    query: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
) -> dict:
    """Fuzzy search across all nodes in the world model.

    Searches entity names, location names, event descriptions, object names,
    and world trait names. Returns ranked results with relevance scores.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    q = query.lower()
    results = []

    for eid, ent in ws.entities.items():
        score = SequenceMatcher(None, q, ent.name.lower()).ratio()
        # Also check traits and constants for keyword matches
        for c in ent.constants:
            s2 = SequenceMatcher(None, q, c.lower()).ratio()
            score = max(score, s2 * 0.8)
        if score > 0.3:
            results.append({"id": eid, "name": ent.name, "type": "Entity",
                            "relevance": round(score, 3), "snippet": f"Status: {ent.status}"})

    for lid, loc in ws.locations.items():
        score = SequenceMatcher(None, q, loc.name.lower()).ratio()
        if loc.description:
            s2 = SequenceMatcher(None, q, loc.description.lower()).ratio()
            score = max(score, s2 * 0.7)
        if score > 0.3:
            results.append({"id": lid, "name": loc.name, "type": "Location",
                            "relevance": round(score, 3),
                            "snippet": (loc.description or "")[:80]})

    for evt in ws.events:
        desc = evt.description or ""
        score = SequenceMatcher(None, q, desc.lower()).ratio()
        if score > 0.3:
            results.append({"id": evt.id, "name": desc[:50], "type": "EventNode",
                            "relevance": round(score, 3),
                            "snippet": f"t={evt.fabula_time} {evt.event_type}"})

    for oid, obj in ws.objects.items():
        score = SequenceMatcher(None, q, obj.name.lower()).ratio()
        if score > 0.3:
            results.append({"id": oid, "name": obj.name, "type": "NarrativeObject",
                            "relevance": round(score, 3), "snippet": ""})

    for wid, wt in ws.world_traits.items():
        score = SequenceMatcher(None, q, wt.name.lower()).ratio()
        if wt.description:
            s2 = SequenceMatcher(None, q, wt.description.lower()).ratio()
            score = max(score, s2 * 0.7)
        if score > 0.3:
            results.append({"id": wid, "name": wt.name, "type": "WorldTrait",
                            "relevance": round(score, 3),
                            "snippet": f"magnitude={wt.magnitude.value:.2f}"})

    results.sort(key=lambda r: r["relevance"], reverse=True)
    return {"results": results[:20], "total": len(results)}


@mcp.tool()
@_safe_tool
def get_relationships(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    entity_id: Optional[str] = None,
    version: Optional[int] = None,
) -> dict:
    """Get social relationships with names resolved.

    Optionally filtered to relationships involving a specific entity.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    edges = ws.social_topology
    if entity_id:
        edges = [e for e in edges
                 if e.source_entity_id == entity_id or e.target_entity_id == entity_id]

    def _name(eid):
        return ws.entities[eid].name if eid in ws.entities else eid

    return {
        "relationships": [
            {
                "source": {"id": e.source_entity_id, "name": _name(e.source_entity_id)},
                "target": {"id": e.target_entity_id, "name": _name(e.target_entity_id)},
                "affinity": e.affinity,
                "fear": e.fear,
                "power_dynamic": e.power_dynamic,
                "inertia": e.inertia,
            }
            for e in edges
        ],
    }


@mcp.tool()
@_safe_tool
def trace_causality(
    ctx: Context,
    node_id: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
    direction: str = "both",
    depth: int = 3,
) -> dict:
    """Trace causal chains upstream and/or downstream from a node.

    Args:
        node_id: The event or entity ID to trace from.
        direction: 'upstream' (causes), 'downstream' (effects), or 'both'.
        depth: Maximum traversal depth (default 3).

    Returns a subgraph of causal edges with mechanism and force details.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    visited: set[str] = set()
    edges_out: list[dict] = []

    def _walk_upstream(nid: str, d: int):
        if d <= 0 or nid in visited:
            return
        visited.add(nid)
        for ce in ws.causal_topology:
            if ce.target_id == nid:
                edges_out.append({
                    "source": ce.source_id, "target": ce.target_id,
                    "type": ce.causality_type, "mechanism": ce.mechanism,
                    "force": ce.causal_force, "trait_target": ce.trait_target,
                    "trait_delta": ce.trait_delta, "delay": ce.propagation_delay,
                })
                _walk_upstream(ce.source_id, d - 1)

    def _walk_downstream(nid: str, d: int):
        if d <= 0 or nid in visited:
            return
        visited.add(nid)
        for ce in ws.causal_topology:
            if ce.source_id == nid:
                edges_out.append({
                    "source": ce.source_id, "target": ce.target_id,
                    "type": ce.causality_type, "mechanism": ce.mechanism,
                    "force": ce.causal_force, "trait_target": ce.trait_target,
                    "trait_delta": ce.trait_delta, "delay": ce.propagation_delay,
                })
                _walk_downstream(ce.target_id, d - 1)

    if direction in ("upstream", "both"):
        _walk_upstream(node_id, depth)
    visited_up = visited.copy()
    if direction in ("downstream", "both"):
        visited = set()  # reset for downstream walk
        _walk_downstream(node_id, depth)

    all_nodes = visited_up | visited | {node_id}
    return {
        "root": node_id,
        "direction": direction,
        "depth": depth,
        "nodes": sorted(all_nodes),
        "edges": edges_out,
    }


@mcp.tool()
@_safe_tool
def get_history(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
) -> dict:
    """Get version history — full tree or single version detail.

    Without version: returns the full version tree.
    With version: returns that version's detail including prose, changeset, and query.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    if version is not None:
        ver = get_version(pid, version)
        if ver is None:
            return {"error": f"Version {version} not found."}

        changeset = None
        if ver.changeset_json:
            try:
                changeset = json.loads(ver.changeset_json)
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "version": ver.version,
            "source": ver.source,
            "description": ver.description,
            "prose": ver.prose,
            "raw_query": ver.raw_query,
            "changeset": changeset,
            "ancestor_id": ver.ancestor_id,
            "created_at": str(ver.created_at),
        }

    tree = get_version_tree(pid)
    return {"project_id": pid, "versions": tree or []}


# =====================================================================
# GROUP 3: REASON — "Why did X happen? What tensions exist?"
# =====================================================================


@mcp.tool()
@_safe_tool
def ask(
    ctx: Context,
    question: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
) -> dict:
    """Ask a read-only question about the story world.

    Runs the physics engine to analyze the world model without generating
    prose or creating a new version. Use this for analysis — use 'narrate'
    when you want to advance the story.

    Examples:
      "What does Macbeth believe about Lady Macbeth?"
      "Who is at the castle right now?"
      "What are the causal consequences of the murder?"
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    # Parse the question to resolve IDs
    parse_result = parse_query(
        question, query_type="interrogate", world_state=ws,
        config=QueryParsingConfig(),
    )

    if not parse_result.is_valid or parse_result.query is None:
        # Fallback: run as general physics query
        query = InterrogationQuery(question=question, require_proof=True)
    else:
        query = parse_result.query

    # Run physics only — no generation, no version creation
    try:
        physics = calculate_narrative_physics(
            request=query,
            global_world_state=ws,
        )
    except Exception as e:
        logger.exception("Physics calculation failed")
        return {"error": f"Analysis failed: {e}"}

    result: dict[str, Any] = {
        "question": question,
        "query_type": query.query_type,
        "read_only": True,
    }

    if parse_result.parsed:
        result["reasoning"] = parse_result.parsed.reasoning

    # Extract structured answer from physics result
    if isinstance(physics, dict):
        if "answer" in physics:
            result["answer"] = physics["answer"]
        if "ego_graph" in physics:
            ego = physics["ego_graph"]
            if isinstance(ego, dict):
                result["focus_entities"] = ego.get("focus_entities", [])
                result["recent_events"] = [
                    {"id": e.get("id"), "description": e.get("description"),
                     "fabula_time": e.get("fabula_time")}
                    for e in ego.get("recent_memory", [])
                ][:5]
        result["physics_status"] = physics.get("status", "complete")

    return result


@mcp.tool()
@_safe_tool
def compute_tension(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    entity_ids: Optional[List[str]] = None,
    version: Optional[int] = None,
) -> dict:
    """Compute narrative tension scores for the story or specific entities.

    Returns 8 structural and emotional scores:
      mystery, dramatic_irony, suspense, surprise (0–1 each)
      epistemic_gaps, hidden_channels, narrative_tensions, relationship_tensions

    These scores measure the story's potential for each effect at the
    current point in the narrative.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    if not entity_ids:
        entity_ids = list(ws.entities.keys())[:6]

    # Build ego-graph and DirectiveAssembler for score computation
    try:
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        from shadow_loom.instantiator import AMWNInstantiator
        from shadow_loom.directive_assembly import DirectiveAssembler

        ego_payload = extract_ego_graph_from_memory(ws, entity_ids)
        ego_dict = ego_payload.model_dump()
        sandbox = AMWNInstantiator.create_sandbox(ego_dict, "interrogate")
        assembler = DirectiveAssembler(sandbox, ego_dict, ws)

        scores: dict[str, Any] = {
            "entity_ids": entity_ids,
            "scores": {
                "mystery": round(assembler.compute_mystery_score(entity_ids), 3),
                "dramatic_irony": round(assembler.compute_dramatic_irony_score(entity_ids), 3),
                "suspense": round(assembler.compute_suspense_score(entity_ids), 3),
                "surprise": round(assembler.compute_surprise_score(entity_ids), 3),
            },
            "epistemic_gaps": [g.model_dump() for g in assembler.compute_epistemic_gaps(entity_ids)],
            "narrative_tensions": [t.model_dump() for t in assembler.compute_narrative_tension()],
            "hidden_channels": [c.model_dump() for c in assembler.compute_hidden_channels()],
            "trait_trajectories": [t.model_dump() for t in assembler.compute_trait_trajectories(entity_ids)],
            "relationship_tensions": [t.model_dump() for t in assembler.compute_relationship_tensions(entity_ids)],
        }
        return scores

    except Exception as e:
        logger.exception("Tension computation failed")
        return {"error": f"Tension computation failed: {e}"}


@mcp.tool()
@_safe_tool
def diff_versions(
    ctx: Context,
    project_id: int,
    version_a: int,
    version_b: int,
) -> dict:
    """Compare two versions of a world model structurally.

    Returns entities, events, edges, and traits that were
    added, removed, or changed between the two versions.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    err = check_project_access(project_id, ctx)
    if err:
        return {"error": err}

    ws_a, _ = load_world_state(project_id, version_a)
    ws_b, _ = load_world_state(project_id, version_b)
    if ws_a is None:
        return {"error": f"Version {version_a} not found."}
    if ws_b is None:
        return {"error": f"Version {version_b} not found."}

    diff: dict[str, Any] = {
        "version_a": version_a,
        "version_b": version_b,
    }

    # Entity diff
    ents_a, ents_b = set(ws_a.entities.keys()), set(ws_b.entities.keys())
    diff["entities_added"] = sorted(ents_b - ents_a)
    diff["entities_removed"] = sorted(ents_a - ents_b)
    trait_changes = []
    entity_changes = []
    for eid in ents_a & ents_b:
        ea, eb = ws_a.entities[eid], ws_b.entities[eid]
        if ea.status != eb.status:
            entity_changes.append({"entity": eid, "field": "status", "before": ea.status, "after": eb.status})
        if ea.location_id != eb.location_id:
            entity_changes.append({"entity": eid, "field": "location_id", "before": ea.location_id, "after": eb.location_id})
        for trait_name in set(ea.traits) | set(eb.traits):
            va = ea.traits.get(trait_name)
            vb = eb.traits.get(trait_name)
            val_a = va.value if va else None
            val_b = vb.value if vb else None
            if val_a != val_b:
                trait_changes.append({
                    "entity": eid, "trait": trait_name,
                    "before": val_a, "after": val_b,
                })
    diff["trait_changes"] = trait_changes
    diff["entity_changes"] = entity_changes

    # Location diff
    locs_a, locs_b = set(ws_a.locations.keys()), set(ws_b.locations.keys())
    diff["locations_added"] = sorted(locs_b - locs_a)
    diff["locations_removed"] = sorted(locs_a - locs_b)

    # Object diff
    objs_a, objs_b = set(ws_a.objects.keys()), set(ws_b.objects.keys())
    diff["objects_added"] = sorted(objs_b - objs_a)
    diff["objects_removed"] = sorted(objs_a - objs_b)

    # World trait diff
    wts_a, wts_b = set(ws_a.world_traits.keys()), set(ws_b.world_traits.keys())
    diff["world_traits_added"] = sorted(wts_b - wts_a)
    diff["world_traits_removed"] = sorted(wts_a - wts_b)
    wt_changes = []
    for wid in wts_a & wts_b:
        wa, wb = ws_a.world_traits[wid], ws_b.world_traits[wid]
        if wa.magnitude.value != wb.magnitude.value:
            wt_changes.append({
                "world_trait": wid, "before": wa.magnitude.value, "after": wb.magnitude.value,
            })
    diff["world_trait_changes"] = wt_changes

    # Event diff
    evts_a = {e.id for e in ws_a.events}
    evts_b = {e.id for e in ws_b.events}
    diff["events_added"] = sorted(evts_b - evts_a)
    diff["events_removed"] = sorted(evts_a - evts_b)

    # Topology diffs
    def _causal_key(e):
        return f"{e.source_id}->{e.target_id}"

    causal_a = {_causal_key(e) for e in ws_a.causal_topology}
    causal_b = {_causal_key(e) for e in ws_b.causal_topology}
    diff["causal_edges_added"] = len(causal_b - causal_a)
    diff["causal_edges_removed"] = len(causal_a - causal_b)

    social_a = {f"{e.source_entity_id}->{e.target_entity_id}" for e in ws_a.social_topology}
    social_b = {f"{e.source_entity_id}->{e.target_entity_id}" for e in ws_b.social_topology}
    diff["social_edges_added"] = len(social_b - social_a)
    diff["social_edges_removed"] = len(social_a - social_b)

    spatial_a = {f"{e.source_id}->{e.target_id}" for e in ws_a.spatial_topology}
    spatial_b = {f"{e.source_id}->{e.target_id}" for e in ws_b.spatial_topology}
    diff["spatial_edges_added"] = len(spatial_b - spatial_a)
    diff["spatial_edges_removed"] = len(spatial_a - spatial_b)

    info_a = {f"{e.source_id}->{','.join(sorted(e.target_ids))}" for e in ws_a.information_topology}
    info_b = {f"{e.source_id}->{','.join(sorted(e.target_ids))}" for e in ws_b.information_topology}
    diff["information_edges_added"] = len(info_b - info_a)
    diff["information_edges_removed"] = len(info_a - info_b)

    return diff


# =====================================================================
# GROUP 4: CREATE — "Generate prose / advance the story / import text."
# =====================================================================


@mcp.tool()
@_safe_tool
async def narrate(
    ctx: Context,
    instruction: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    mode: Optional[str] = None,
    version: Optional[int] = None,
    skip_audit: bool = True,
) -> dict:
    """Generate prose and advance the story using natural language.

    This is the primary creation tool. Your instruction is parsed,
    run through physics simulation, and rendered as prose. A new version
    is created.

    Args:
        instruction: Natural language instruction (e.g., "Continue the
            story from Macbeth's perspective" or "Kill Duncan").
        mode: Force a query type — 'observe', 'intervene', 'counterfactual',
              or None for auto-detect.
        skip_audit: Skip the audit loop for faster results (default True).

    Reports progress via MCP progress notifications.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, ancestor_row_id = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found. Use 'ingest' first."}

    user_row_id = get_user_id(ctx)

    # Map mode to query_type
    mode_map = {
        "observe": "observation",
        "intervene": "intervention",
        "counterfactual": "counterfactual",
    }
    query_type = mode_map.get(mode) if mode else None

    # Stage 1: Parse
    await ctx.report_progress(1, 4, "Parsing instruction...")
    parse_result = parse_query(
        instruction, query_type=query_type, world_state=ws,
        config=QueryParsingConfig(),
    )

    if not parse_result.is_valid or parse_result.query is None:
        return {
            "error": "Query parsing failed",
            "reasoning": parse_result.parsed.reasoning if parse_result.parsed else None,
            "validation_errors": [
                {"field": e.field, "message": e.message}
                for e in parse_result.validation_errors
            ],
        }

    # Stage 2–4: Pipeline
    await ctx.report_progress(2, 4, "Running physics simulation...")

    response = run_and_save(
        query=parse_result.query,
        project_id=pid,
        world_state=ws,
        ancestor_row_id=ancestor_row_id,
        user_row_id=user_row_id,
        raw_query=instruction,
        skip_audit=skip_audit,
    )

    await ctx.report_progress(4, 4, "Complete")

    if parse_result.parsed:
        response["reasoning"] = parse_result.parsed.reasoning
        response["resolved_ids"] = [
            {"name": r.natural_name, "id": r.resolved_id, "confidence": r.confidence}
            for r in parse_result.parsed.resolved_ids
        ]

    return response


@mcp.tool()
@_safe_tool
async def direct(
    ctx: Context,
    target_effect: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    entity_ids: Optional[List[str]] = None,
    intensity: float = 0.8,
    version: Optional[int] = None,
    skip_audit: bool = True,
) -> dict:
    """Generate a scene optimized for a specific emotional effect.

    Available effects: mystery, dramatic_irony, suspense, surprise,
    grief, rage, joy, fear, love, regret.

    Args:
        target_effect: The emotional effect to optimize for.
        entity_ids: Entities to focus on (default: all).
        intensity: Effect intensity 0.0–1.0 (default 0.8).
        skip_audit: Skip the audit loop (default True).
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, ancestor_row_id = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    valid_effects = {
        "mystery", "dramatic_irony", "suspense", "surprise",
        "grief", "rage", "joy", "fear", "love", "regret",
    }
    if target_effect not in valid_effects:
        return {"error": f"Invalid effect '{target_effect}'. Choose from: {', '.join(sorted(valid_effects))}"}

    user_row_id = get_user_id(ctx)

    query = DirectiveQuery(
        target_entity_ids=entity_ids or [],
        target_effect=target_effect,
        intensity=intensity,
    )

    await ctx.report_progress(1, 3, f"Assembling {target_effect} directive...")

    response = run_and_save(
        query=query,
        project_id=pid,
        world_state=ws,
        ancestor_row_id=ancestor_row_id,
        user_row_id=user_row_id,
        raw_query=f"Directive: {target_effect} (intensity={intensity})",
        skip_audit=skip_audit,
    )

    await ctx.report_progress(3, 3, "Complete")
    return response


@mcp.tool()
@_safe_tool
def write(
    ctx: Context,
    prose: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    description: str = "",
    version: Optional[int] = None,
) -> dict:
    """Apply user-written prose as a manual edit to the world model.

    The prose is re-extracted into topology and merged into the world
    model. Creates a new version. Use this when you want to write
    narrative directly rather than having the engine generate it.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, ancestor_row_id = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    user_row_id = get_user_id(ctx)
    query = ManualEditQuery(edited_prose=prose, description=description)

    response = run_and_save(
        query=query,
        project_id=pid,
        world_state=ws,
        ancestor_row_id=ancestor_row_id,
        user_row_id=user_row_id,
        raw_query=prose,
        skip_audit=True,
        skip_reextraction=False,
    )

    return response


@mcp.tool()
@_safe_tool
async def ingest(
    ctx: Context,
    text: str,
    project_name: str = "MCP Project",
    label: Optional[str] = None,
) -> dict:
    """Ingest raw narrative text to create a new project and world model.

    Extracts entities, locations, events, objects, world traits, and all
    topology edges from the text. Creates a new project with version 0.

    Reports progress via MCP progress notifications.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)

    config = ExtractionConfig(
        chunk_strategy="act_headings",
        fabula_time_spacing=100,
        output_retries=5,
        max_correction_retries=1,
    )

    await ctx.report_progress(1, 3, "Extracting ontology and topology...")

    try:
        ws, report = run_extraction(text, config)
    except Exception as e:
        logger.exception("Ingestion failed")
        return {"error": f"Ingestion failed: {e}"}

    await ctx.report_progress(2, 3, "Saving project...")

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

    await ctx.report_progress(3, 3, "Complete")

    return {
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
        "validation": {
            "is_valid": report.is_valid,
            "issue_count": len(report.issues),
        },
    }


# =====================================================================
# GROUP 5: JUDGE — "Is this story good? What's broken?"
# =====================================================================


@mcp.tool()
@_safe_tool
def evaluate(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    focus_entity_ids: Optional[List[str]] = None,
    version: Optional[int] = None,
    target_effect: str = "suspense",
) -> dict:
    """Run a full-story quality evaluation (read-only).

    Computes the NarrativeOrderObject scorecard:
      - Causal metrics: miracle steps, foreshadowing payoff, cognitive plausibility
      - Affective metrics: emotional trajectory scores, KL divergence, affective loss
      - Quality synthesis: coherence review, reward-hacking diagnostics, rewrite directives
      - Overall pass/fail

    Does NOT create a new version.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state(pid, version)
    if ws is None:
        return {"error": "No world model found."}

    # Build machinery for evaluation
    try:
        from shadow_loom.extract_graph import extract_ego_graph_from_memory
        from shadow_loom.instantiator import AMWNInstantiator
        from shadow_loom.directive_assembly import DirectiveAssembler, CreativeBrief
        from shadow_loom.auditor import (
            compute_causal_feedback,
            compute_affective_feedback,
            _finalize_narrative_order,
            AuditorConfig,
        )

        eids = focus_entity_ids or list(ws.entities.keys())

        # Build assembler for affective scoring
        ego_payload = extract_ego_graph_from_memory(ws, eids[:6])
        ego_dict = ego_payload.model_dump()
        sandbox = AMWNInstantiator.create_sandbox(ego_dict, "evaluate")
        assembler = DirectiveAssembler(sandbox, ego_dict, ws)

        # Compute engine metrics
        causal_fb = compute_causal_feedback(None, CreativeBrief(
            target_effect=target_effect, target_entities=eids[:6],
            constraints=[], scene_context={},
        ), ws)
        affective_fb = compute_affective_feedback(
            CreativeBrief(
                target_effect=target_effect, target_entities=eids[:6],
                constraints=[], scene_context={},
            ),
            assembler, eids[:6],
        )

        # Collect all prose for evaluation
        prose_list = get_all_prose(pid)
        all_prose = "\n\n".join(entry["prose"] for entry in prose_list if entry.get("prose"))

        if all_prose:
            # Run full evaluation with LLM literary critique
            noo = _finalize_narrative_order(
                all_prose,
                CreativeBrief(
                    target_effect=target_effect, target_entities=eids[:6],
                    constraints=[], scene_context={},
                ),
                causal_fb, affective_fb,
                AuditorConfig(),
            )
            return {
                "read_only": True,
                "overall_pass": noo.overall_pass,
                "causal_feedback": noo.causal_feedback.model_dump(),
                "affective_feedback": noo.affective_feedback.model_dump(),
                "quality_synthesis": noo.quality_synthesis.model_dump(),
            }
        else:
            # No prose — return engine metrics only
            return {
                "read_only": True,
                "note": "No prose found — returning engine metrics only.",
                "causal_feedback": causal_fb.model_dump(),
                "affective_feedback": affective_fb.model_dump(),
            }

    except Exception as e:
        logger.exception("Evaluation failed")
        return {"error": f"Evaluation failed: {e}"}


@mcp.tool()
@_safe_tool
def audit_log(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
) -> dict:
    """Get audit history and activity log for a project.

    Without version: returns the project activity log (queries, edits, etc.).
    With version: returns that version's audit/convergence details.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    if version is not None:
        ver = get_version(pid, version)
        if ver is None:
            return {"error": f"Version {version} not found."}
        return {
            "version": ver.version,
            "source": ver.source,
            "prose_length": len(ver.prose) if ver.prose else 0,
            "raw_query": ver.raw_query,
            "created_at": str(ver.created_at),
        }

    activities = get_project_activity(pid, limit=50)
    return {
        "project_id": pid,
        "activities": activities,
    }


# =====================================================================
# GROUP 6: MANAGE — "Branch, share, fork, configure."
# =====================================================================


@mcp.tool()
@_safe_tool
def branch(
    ctx: Context,
    project_id: int,
    from_version: int,
) -> dict:
    """Create a new version branching from a previous version.

    Does not delete history — creates a new branch point.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    err = check_project_access(project_id, ctx)
    if err:
        return {"error": err}

    ws, ancestor_row_id = load_world_state(project_id, from_version)
    if ws is None:
        return {"error": f"Version {from_version} not found."}

    user_row_id = get_user_id(ctx)

    ver = save_version(
        project_id=project_id,
        world_state_json=ws.model_dump_json(),
        ancestor_id=ancestor_row_id,
        source="branch",
        description=f"Branched from v{from_version}",
        user_id=user_row_id,
    )

    return {
        "project_id": project_id,
        "new_version": ver.version,
        "branched_from": from_version,
    }


@mcp.tool()
@_safe_tool
def share(
    ctx: Context,
    project_id: int,
    username: str,
    role: str = "viewer",
) -> dict:
    """Share a project with another user.

    Only project owners and admins can share.

    Args:
        username: Username of the person to share with.
        role: 'viewer', 'editor', or 'admin'.
    """
    err = require_scope(ctx, "admin")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)
    err = check_project_access(project_id, ctx)
    if err:
        return {"error": err}

    proj = get_project(project_id)
    if proj is None:
        return {"error": "Project not found."}

    # Only owner or admin can share
    is_owner = proj.owner_id == user_row_id
    if not is_owner:
        user_role = get_user_project_role(project_id, user_row_id) if user_row_id else None
        if user_role != "admin":
            return {"error": "Only project owners and admins can share."}

    targets = search_users(username)
    if not targets:
        return {"error": f"User '{username}' not found."}

    target = targets[0]
    add_project_member(project_id, target["id"], role)

    return {
        "status": "shared",
        "project_id": project_id,
        "target_user": target["username"],
        "role": role,
    }


@mcp.tool()
@_safe_tool
def fork(
    ctx: Context,
    project_id: int,
    new_name: Optional[str] = None,
) -> dict:
    """Fork (copy) a project to your own account."""
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)
    if user_row_id is None:
        return {"error": "Authentication required to fork."}

    err = check_project_access(project_id, ctx)
    if err:
        return {"error": err}

    proj = get_project(project_id)
    if proj is None:
        return {"error": "Project not found."}

    fork_name = new_name or f"{proj.name} (fork)"
    new_proj = fork_project(project_id, user_row_id, fork_name)
    if new_proj is None:
        return {"error": "Fork failed — source project or version not found."}

    return {
        "status": "forked",
        "original_project_id": project_id,
        "new_project_id": new_proj.id,
        "new_project_name": new_proj.name,
    }


@mcp.tool()
@_safe_tool
def update_project_tool(
    ctx: Context,
    project_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_public: Optional[bool] = None,
) -> dict:
    """Update project metadata (name, description, visibility).

    Requires admin scope. Only owners can update.
    """
    err = require_scope(ctx, "admin")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)
    proj = get_project(project_id)
    if proj is None:
        return {"error": "Project not found."}
    if proj.owner_id != user_row_id:
        return {"error": "Only the project owner can update settings."}

    updated = update_project(
        project_id,
        name=name,
        description=description,
        is_public=is_public,
    )
    if updated is None:
        return {"error": "Update failed."}

    return {
        "project_id": project_id,
        "name": updated.name,
        "description": updated.description,
        "is_public": updated.is_public,
    }


# =====================================================================
# RESOURCES (unauthenticated — only expose public data)
# =====================================================================


@mcp.resource("world://projects")
def resource_projects() -> str:
    """List of all public projects (unauthenticated).

    For user-scoped project lists, use the list_projects tool with auth.
    """
    all_projects = db_list_projects()
    public = [p for p in all_projects if p.get("is_public")]
    return json.dumps(public, default=str)


@mcp.resource("world://project/{project_id}")
def resource_project(project_id: int) -> str:
    """Public project metadata (unauthenticated)."""
    proj = get_project(project_id)
    if proj is None:
        return json.dumps({"error": "Not found"})
    if not proj.is_public:
        return json.dumps({"error": "Access denied — use open_project tool with auth"})
    versions = list_versions(project_id)
    return json.dumps({
        "id": proj.id, "name": proj.name,
        "description": proj.description,
        "owner_id": proj.owner_id,
        "is_public": proj.is_public,
        "version_count": len(versions),
    }, default=str)


@mcp.resource("world://project/{project_id}/world")
def resource_world(project_id: int) -> str:
    """Current world state of a public project (unauthenticated)."""
    proj = get_project(project_id)
    if proj is None or not proj.is_public:
        return json.dumps({"error": "Not found or access denied"})
    ws, _ = load_world_state(project_id)
    if ws is None:
        return json.dumps({"error": "No world model"})
    return ws.model_dump_json(indent=2)


@mcp.resource("world://project/{project_id}/entity/{entity_id}")
def resource_entity(project_id: int, entity_id: str) -> str:
    """Entity detail from a public project (unauthenticated)."""
    proj = get_project(project_id)
    if proj is None or not proj.is_public:
        return json.dumps({"error": "Not found or access denied"})
    ws, _ = load_world_state(project_id)
    if ws is None:
        return json.dumps({"error": "No world model"})
    ent = ws.entities.get(entity_id)
    if ent is None:
        return json.dumps({"error": f"Entity {entity_id} not found"})
    return ent.model_dump_json(indent=2)


@mcp.resource("world://project/{project_id}/versions")
def resource_versions(project_id: int) -> str:
    """Version tree of a public project (unauthenticated)."""
    proj = get_project(project_id)
    if proj is None or not proj.is_public:
        return json.dumps({"error": "Not found or access denied"})
    tree = get_version_tree(project_id)
    return json.dumps(tree or [], default=str)


# =====================================================================
# Entry point
# =====================================================================


if __name__ == "__main__":
    mcp.run()
