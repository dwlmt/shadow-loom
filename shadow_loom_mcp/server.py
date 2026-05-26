# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shadow-Loom MCP server — agent-first narrative intelligence.

42 tools grouped by cognitive task. Four coarse-grained dispatchers wrap the
granular surface so new integrations can reach most functionality through
one well-known entry point; the granular tools remain registered for
backward compatibility.

  DISPATCHERS (4) — discover, trace, author, manage
  ORIENT      (2) — list_projects, open_project
  EXPLORE    (10) — inspect, search, get_relationships,
                    list_channels, get_channel_history, who_can_hear,
                    trace_causality, get_history, list_branches,
                    export_prose
  REASON      (3) — ask, compute_tension, diff_versions
  CREATE      (4) — narrate, direct, write, ingest
  JUDGE       (2) — evaluate, audit_log
  RESEARCH    (6) — research_topic, list_world_facts, delete_world_fact,
                    get_research_status, get_project_settings,
                    set_project_settings
  MANAGE     (10) — branch, fork, share, promote_branch,
                    update_project_tool, delete_project, delete_version,
                    reparent_version, set_active_version, get_active_version

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

import asyncio
import json
import functools
import inspect as _inspect
import logging
from difflib import SequenceMatcher
from typing import Any, List, Optional

from fastmcp import Context, FastMCP

from shadow_loom.db import (
    add_project_member,
    clear_active_version as db_clear_active_version,
    create_project,
    delete_project as db_delete_project,
    delete_version as db_delete_version,
    delete_world_fact as db_delete_world_fact,
    fork_project,
    get_active_version as db_get_active_version,
    get_all_prose,
    get_lineage_to_root,
    get_project,
    get_project_activity,
    get_project_settings as db_get_project_settings,
    get_user_project_role,
    get_version,
    get_version_by_id,
    get_version_tree,
    init_db,
    list_projects as db_list_projects,
    list_versions,
    list_world_facts as db_list_world_facts,
    list_branches as db_list_branches,
    promote_branch as db_promote_branch,
    ProjectDeleteError,
    reparent_version as db_reparent_version,
    save_version,
    search_users,
    set_active_version as db_set_active_version,
    set_project_settings as db_set_project_settings,
    update_project,
    VersionMutationError,
)
from shadow_loom.ingestion import (
    ExtractionConfig,
    WorldStatePatch,
    _apply_world_state_patch,
    run_extraction_async,
)
from shadow_loom.models import (
    WorldStateV1,
    reconstruct_entity_at,
    reconstruct_world_trait_at,
    reconstruct_proposition_at,
    reconstruct_concern_at,
)
from shadow_loom.narrative_physics import calculate_narrative_physics
from shadow_loom.projections import (
    filter_world_state_for_pov,
    project_channel,
    project_event,
    reconstruct_entity_at_causal,
    reconstruct_world_trait_at_causal,
    trace_information_flow,
)
from shadow_loom.query_models import (
    DirectiveQuery,
    GeneralQuery,
    InterrogationQuery,
    ManualEditQuery,
)
from shadow_loom.query_parsing import QueryParsingConfig, parse_query

from shadow_loom_mcp.auth import (
    check_project_access,
    get_user_id,
    require_scope,
    verifier,
)
from shadow_loom_mcp.helpers import (
    load_world_state,
    load_world_state_projected,
    load_world_state_with_branch,
    resolve_project,
    run_and_save,
)

from shadow_loom.settings import get_settings as _get_settings

logger = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────

_settings = _get_settings()


def _enforce_word_cap(text: str | None, *, field: str) -> dict | None:
    """Reject text inputs that exceed ``physics.max_ingest_words``.

    Mirrors the UI's ingestion gate so the same workload is not
    accepted via API and rejected in the browser. Returns an error
    dict on overflow or ``None`` when the input is acceptable.
    """
    if not text:
        return None
    cap = int(_settings.physics.max_ingest_words)
    n = len(text.split())
    if n > cap:
        return {
            "error": (
                f"{field} has {n:,} words, exceeding the {cap:,}-word "
                f"ingest limit. Trim the text or split into smaller "
                f"calls."
            ),
            "field": field,
            "word_count": n,
            "limit": cap,
        }
    return None

# ── DB init ───────────────────────────────────────────────────────

init_db(_settings.core.database_url)

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
    """Wrap an MCP tool function so unhandled exceptions return an error dict.

    Client-facing error text is sanitised to ``"<tool> failed"`` plus the
    exception class name (no message body, no stack). Full exception
    detail is logged server-side via ``logger.exception`` so operators
    can still triage. Without this the raw ``str(exc)`` was returned to
    the caller, which can leak internal paths, SQL fragments, provider
    error bodies, and stack-derived identifiers (round-3 audit).
    """
    if _inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Tool %s failed", fn.__name__)
                return {
                    "error": f"{fn.__name__} failed",
                    "error_type": type(e).__name__,
                }
    else:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Tool %s failed", fn.__name__)
                return {
                    "error": f"{fn.__name__} failed",
                    "error_type": type(e).__name__,
                }
    return wrapper


def _sanitised_error(op: str, exc: BaseException) -> dict:
    """Return a client-safe error envelope for caught exceptions.

    AUDIT (post-2026-05-26): inline ``return {"error": str(e)}`` /
    ``f"... {e}"`` patterns inside ``@_safe_tool`` tools still leaked
    provider error bodies, SQL fragments, and stack-derived identifiers
    because they ran *before* the wrapper's blanket handler. Use this
    helper at every caught-exception return site so the client only
    sees ``{op} failed`` plus the exception class name; full detail is
    logged server-side via ``logger.exception`` for operator triage.
    """
    logger.exception("%s failed", op)
    return {"error": f"{op} failed", "error_type": type(exc).__name__}


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

    ws, ver_row_id = load_world_state_projected(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found. Use 'ingest' first."}

    proj = get_project(pid)
    versions = list_versions(pid)

    # Report the version we actually loaded (which may differ from the
    # latest version when the caller passed an explicit ``version`` or
    # the active-version pointer points elsewhere).
    loaded_version = None
    if ver_row_id is not None:
        loaded_row = get_version_by_id(ver_row_id)
        if loaded_row is not None:
            loaded_version = loaded_row.version
    if loaded_version is None:
        loaded_version = versions[-1]["version"] if versions else 0

    return {
        "project_id": pid,
        "project_name": proj.name if proj else "",
        "description": proj.description if proj else None,
        "current_version": loaded_version,
        "latest_version": versions[-1]["version"] if versions else 0,
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
            "channels": len(ws.channels),
            "utterance_events": sum(1 for e in ws.events if e.event_type == "utterance"),
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
    pov_entity_id: Optional[str] = None,
    timeline_limit: int = 10,
    timeline_offset: int = 0,
) -> dict:
    """Inspect any node in the world model by ID.

    Auto-detects the node type from the ID prefix:
      ENT_  \u2192 entity (traits, beliefs, constants, state timeline)
      LOC_  \u2192 location (ambient state, connections, occupants)
      EVT_  \u2192 event (actors, targets, causal causes/effects, supersession)
      OBJ_  \u2192 object (owner, location, affordances, properties)
      CHN_  \u2192 channel (participants, intelligibility, utterances)
      WORLD_ \u2192 world trait (magnitude, domains, timeline)
      PROP_ \u2192 proposition (kind, referents, truth_at_fabula, framing timeline)
      CCN_  \u2192 concern (entity, polarity, salience, activation window, timeline)

    Optionally provide ``at_time`` (fabula_time) to see the reconstructed
    state at a specific point in the story's timeline. Time-slice
    reconstruction is *causal-aware* for entities and world traits:
    mutation causal edges are replayed on top of the state timeline.

    ``pov_entity_id`` (scaffold): if set, the world is filtered through that
    character's epistemic lens before inspection \u2014 utterances they could
    not plausibly hear and channels they don't participate in are pruned.
    Use this for limited-omniscience views.

    ``timeline_limit`` / ``timeline_offset`` paginate ``state_timeline``
    on entity, world-trait, proposition, and concern responses. The
    response carries ``state_timeline_total`` and the applied window so
    callers can detect truncation. Default returns the most recent 10
    entries (offset 0 = newest end). Pass ``timeline_limit=0`` to
    suppress the timeline entirely; pass a large limit + ``offset=-1``
    to retrieve the full history.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    if pov_entity_id:
        ws = filter_world_state_for_pov(
            ws, pov_entity_id,
            intelligibility_threshold=_settings.physics.intelligibility_threshold,
        )

    if node_id.startswith("ENT_"):
        return _inspect_entity(ws, node_id, at_time, timeline_limit, timeline_offset)
    elif node_id.startswith("LOC_"):
        return _inspect_location(ws, node_id)
    elif node_id.startswith("EVT_"):
        return _inspect_event(ws, node_id)
    elif node_id.startswith("OBJ_"):
        return _inspect_object(ws, node_id)
    elif node_id.startswith("CHN_"):
        return _inspect_channel(ws, node_id)
    elif node_id.startswith("WORLD_"):
        return _inspect_world_trait(ws, node_id, at_time, timeline_limit, timeline_offset)
    elif node_id.startswith("PROP_"):
        return _inspect_proposition(ws, node_id, at_time, timeline_limit, timeline_offset)
    elif node_id.startswith("CCN_"):
        return _inspect_concern(ws, node_id, at_time, timeline_limit, timeline_offset)
    else:
        return {"error": f"Unknown node ID prefix: {node_id}. Expected ENT_, LOC_, EVT_, OBJ_, CHN_, WORLD_, PROP_, or CCN_."}


def _paginate_timeline(
    items: list,
    limit: int,
    offset: int,
) -> tuple[list, dict]:
    """Window a state_timeline list and return ``(slice, metadata)``.

    Convention:
      * ``limit == 0`` → empty slice; metadata still carries total.
      * ``offset == -1`` → return the entire list (callers asking for
        full history without paging).
      * ``offset == 0`` (default) → return the *most recent* ``limit``
        entries, mirroring legacy ``[-10:]`` behaviour.
      * ``offset > 0`` → skip ``offset`` newest entries before taking
        ``limit`` (e.g. offset=10, limit=10 → entries 11–20 from the end).
    """
    total = len(items)
    if limit == 0:
        return [], {
            "total": total,
            "offset": offset,
            "limit": 0,
            "truncated": total > 0,
        }
    if offset == -1:
        return list(items), {
            "total": total,
            "offset": -1,
            "limit": total,
            "truncated": False,
        }
    end = total - offset if offset > 0 else total
    start = max(0, end - max(0, limit))
    window = items[start:end]
    return window, {
        "total": total,
        "offset": offset,
        "limit": limit,
        "truncated": (start > 0) or (end < total),
        "window": [start, end],
    }


def _inspect_entity(
    ws: WorldStateV1,
    eid: str,
    at_time: int | None,
    timeline_limit: int = 10,
    timeline_offset: int = 0,
) -> dict:
    ent = ws.entities.get(eid)
    if ent is None:
        return {"error": f"Entity '{eid}' not found."}

    if at_time is not None:
        snapshot = reconstruct_entity_at_causal(ws, eid, at_time)
        return {
            "id": eid, "name": ent.name, "type": "Entity",
            "at_time": at_time,
            "reconstruction": "causal_aware",
            "status": snapshot["status"],
            "location_id": snapshot["location_id"],
            "traits": snapshot["traits"],
            "beliefs": snapshot["beliefs"],
            "constants": ent.constants,
        }

    timeline_window, timeline_meta = _paginate_timeline(
        ent.state_timeline, timeline_limit, timeline_offset,
    )
    return {
        "id": eid, "name": ent.name, "type": "Entity",
        "status": ent.status,
        "location_id": ent.location_id,
        "traits": {
            k: {
                "value": v.value,
                "inertia": v.inertia,
                "evidence_strength": v.evidence_strength,
            }
            for k, v in ent.traits.items()
        },
        "beliefs": [b.model_dump() for b in ent.beliefs],
        "constants": ent.constants,
        "state_timeline": [s.model_dump() for s in timeline_window],
        "state_timeline_meta": timeline_meta,
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
        "ambient_state": {
            k: {
                "value": v.value,
                "volatility": v.volatility,
                "evidence_strength": v.evidence_strength,
            }
            for k, v in loc.ambient_state.items()
        },
        "occupants": occupants,
        "connections": connections,
    }


def _inspect_event(ws: WorldStateV1, evt_id: str) -> dict:
    evt = next((e for e in ws.events if e.id == evt_id), None)
    if evt is None:
        return {"error": f"Event '{evt_id}' not found."}
    return project_event(ws, evt)


def _inspect_channel(ws: WorldStateV1, cid: str) -> dict:
    ch = ws.channels.get(cid)
    if ch is None:
        return {"error": f"Channel '{cid}' not found."}
    return project_channel(ws, ch)


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


def _inspect_world_trait(
    ws: WorldStateV1,
    wid: str,
    at_time: int | None,
    timeline_limit: int = 10,
    timeline_offset: int = 0,
) -> dict:
    wt = ws.world_traits.get(wid)
    if wt is None:
        return {"error": f"World trait '{wid}' not found."}

    if at_time is not None:
        snapshot = reconstruct_world_trait_at_causal(ws, wid, at_time)
        return {
            "id": wid, "name": wt.name, "type": "WorldTrait",
            "at_time": at_time,
            "reconstruction": "causal_aware",
            "magnitude": snapshot.get("magnitude"),
            "description": snapshot.get("description"),
        }

    timeline_window, timeline_meta = _paginate_timeline(
        wt.state_timeline, timeline_limit, timeline_offset,
    )
    return {
        "id": wid, "name": wt.name, "type": "WorldTrait",
        "description": wt.description,
        "category": wt.category,
        "magnitude": {
            "value": wt.magnitude.value,
            "inertia": wt.magnitude.inertia,
            "evidence_strength": wt.magnitude.evidence_strength,
        },
        "affected_domains": wt.affected_domains,
        "state_timeline": [s.model_dump() for s in timeline_window],
        "state_timeline_meta": timeline_meta,
    }


def _inspect_proposition(
    ws: WorldStateV1,
    pid: str,
    at_time: int | None,
    timeline_limit: int = 10,
    timeline_offset: int = 0,
) -> dict:
    prop = next((p for p in ws.propositions if p.proposition_id == pid), None)
    if prop is None:
        return {"error": f"Proposition '{pid}' not found."}
    timeline_window, timeline_meta = _paginate_timeline(
        prop.state_timeline, timeline_limit, timeline_offset,
    )
    out: dict = {
        "id": pid,
        "type": "Proposition",
        "kind": prop.kind,
        "description": prop.description,
        "referent_ids": list(prop.referent_ids),
        "world_id": prop.world_id,
        "audience_default_prior": prop.audience_default_prior,
        "stakes": prop.stakes,
        "truth_at_fabula": dict(prop.truth_at_fabula),
        "state_timeline": [s.model_dump() for s in timeline_window],
        "state_timeline_meta": timeline_meta,
    }
    if at_time is not None:
        out["at_time"] = at_time
        out["snapshot"] = reconstruct_proposition_at(prop, at_time)
    return out


def _inspect_concern(
    ws: WorldStateV1,
    cid: str,
    at_time: int | None,
    timeline_limit: int = 10,
    timeline_offset: int = 0,
) -> dict:
    holder_id: str | None = None
    concern = None
    for ent_id, ent in ws.entities.items():
        for c in ent.concerns:
            if c.concern_id == cid:
                concern = c
                holder_id = ent_id
                break
        if concern is not None:
            break
    if concern is None:
        return {"error": f"Concern '{cid}' not found."}
    timeline_window, timeline_meta = _paginate_timeline(
        concern.state_timeline, timeline_limit, timeline_offset,
    )
    out: dict = {
        "id": cid,
        "type": "Concern",
        "entity_id": holder_id,
        "proposition_id": concern.proposition_id,
        "polarity": concern.polarity,
        "kind": concern.kind,
        "salience": concern.salience,
        "activation_fabula_window": concern.activation_fabula_window,
        "counter_concern_ids": list(concern.counter_concern_ids),
        "world_id": concern.world_id,
        "state_timeline": [s.model_dump() for s in timeline_window],
        "state_timeline_meta": timeline_meta,
    }
    if at_time is not None:
        out["at_time"] = at_time
        out["snapshot"] = reconstruct_concern_at(concern, at_time)
    return out


@mcp.tool()
@_safe_tool
def search(
    ctx: Context,
    query: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
    node_type: Optional[str] = None,
) -> dict:
    """Fuzzy search across all nodes in the world model.

    Searches entity names, location names, event descriptions and utterance
    content, object names, world trait names, and channel names/media.
    Returns ranked results with relevance scores.

    Optional ``node_type`` filter restricts results to one of:
    ``entity``, ``location``, ``event``, ``utterance``, ``object``,
    ``world_trait``, ``channel``.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    q = query.lower()
    nt = (node_type or "").lower().strip() or None
    results = []

    def _accept(kind: str) -> bool:
        return nt is None or nt == kind

    if _accept("entity"):
        for eid, ent in ws.entities.items():
            score = SequenceMatcher(None, q, ent.name.lower()).ratio()
            # Also check traits and constants for keyword matches
            for c in ent.constants:
                s2 = SequenceMatcher(None, q, c.lower()).ratio()
                score = max(score, s2 * 0.8)
            if score > 0.3:
                results.append({"id": eid, "name": ent.name, "type": "Entity",
                                "relevance": round(score, 3), "snippet": f"Status: {ent.status}"})

    if _accept("location"):
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
        is_utt = getattr(evt, "event_type", None) == "utterance"
        kind = "utterance" if is_utt else "event"
        if not _accept(kind):
            continue
        desc = evt.description or ""
        content = getattr(evt, "content", None) or ""
        score = SequenceMatcher(None, q, desc.lower()).ratio()
        if content:
            s2 = SequenceMatcher(None, q, content.lower()).ratio()
            score = max(score, s2)
        if score > 0.3:
            label = (content or desc)[:50] if is_utt else desc[:50]
            snippet_bits = [f"t={evt.fabula_time}", evt.event_type]
            if is_utt and getattr(evt, "via_channel_id", None):
                snippet_bits.append(f"via {evt.via_channel_id}")
            results.append({"id": evt.id, "name": label, "type": "EventNode",
                            "relevance": round(score, 3),
                            "snippet": " ".join(snippet_bits)})

    if _accept("object"):
        for oid, obj in ws.objects.items():
            score = SequenceMatcher(None, q, obj.name.lower()).ratio()
            if score > 0.3:
                results.append({"id": oid, "name": obj.name, "type": "NarrativeObject",
                                "relevance": round(score, 3), "snippet": ""})

    if _accept("world_trait"):
        for wid, wt in ws.world_traits.items():
            score = SequenceMatcher(None, q, wt.name.lower()).ratio()
            if wt.description:
                s2 = SequenceMatcher(None, q, wt.description.lower()).ratio()
                score = max(score, s2 * 0.7)
            if score > 0.3:
                results.append({"id": wid, "name": wt.name, "type": "WorldTrait",
                                "relevance": round(score, 3),
                                "snippet": f"magnitude={wt.magnitude.value:.2f}"})

    if _accept("channel"):
        for cid, ch in ws.channels.items():
            score = SequenceMatcher(None, q, ch.name.lower()).ratio()
            if ch.medium:
                s2 = SequenceMatcher(None, q, ch.medium.lower()).ratio()
                score = max(score, s2 * 0.8)
            if score > 0.3:
                results.append({"id": cid, "name": ch.name, "type": "Channel",
                                "relevance": round(score, 3),
                                "snippet": f"medium={ch.medium} participants={len(ch.participant_ids)}"})

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

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
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
                # Aggregate edge-level views (back-compat): minimum
                # inertia / strongest evidence / most-recent timestamp
                # across observed metrics.
                "affinity": e.affinity,
                "fear": e.fear,
                "power_dynamic": e.power_dynamic,
                "inertia": e.inertia,
                "evidence_strength": e.evidence_strength,
                "last_updated_fabula": e.last_updated_fabula,
                # Per-axis detail — clients that care about per-metric
                # uncertainty / staleness should read here. Axes the
                # extractor never observed are absent from the dict
                # (distinct from a meaningful 0.0).
                "metrics": {
                    name: {
                        "value": m.value,
                        "inertia": m.inertia,
                        "evidence_strength": m.evidence_strength,
                        "last_updated_fabula": m.last_updated_fabula,
                        "observed": m.observed,
                    }
                    for name, m in e.metrics.items()
                },
            }
            for e in edges
        ],
    }


@mcp.tool()
@_safe_tool
def list_channels(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    entity_id: Optional[str] = None,
    medium: Optional[str] = None,
    version: Optional[int] = None,
) -> dict:
    """Enumerate communication channels with participant names + utterance counts.

    Optional filters:
      * ``entity_id`` \u2014 only channels in which the entity participates.
      * ``medium`` \u2014 substring match on channel medium (e.g. ``"telephone"``).
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    def _name(nid: str) -> str:
        if nid in ws.entities:
            return ws.entities[nid].name
        if nid in ws.objects:
            return ws.objects[nid].name
        return nid

    # Pre-count utterances per channel.
    utt_counts: dict[str, int] = {}
    for evt in ws.events:
        if evt.event_type == "utterance" and evt.via_channel_id:
            utt_counts[evt.via_channel_id] = utt_counts.get(evt.via_channel_id, 0) + 1

    medium_q = (medium or "").lower().strip()
    out = []
    for cid, ch in ws.channels.items():
        if entity_id and entity_id not in ch.participant_ids:
            continue
        if medium_q and medium_q not in (ch.medium or "").lower():
            continue
        out.append({
            "id": cid,
            "name": ch.name,
            "medium": ch.medium,
            "directionality": ch.directionality,
            "participants": [
                {"id": pid_, "name": _name(pid_)} for pid_ in ch.participant_ids
            ],
            "intelligibility": dict(ch.intelligibility),
            "established_at_fabula": ch.established_at_fabula,
            "terminated_at_fabula": ch.terminated_at_fabula,
            "evidence_strength": ch.evidence_strength,
            "utterance_count": utt_counts.get(cid, 0),
        })
    out.sort(key=lambda r: (-r["utterance_count"], r["id"]))
    return {"channels": out, "total": len(out)}


@mcp.tool()
@_safe_tool
def get_channel_history(
    ctx: Context,
    channel_id: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
    limit: int = 100,
) -> dict:
    """All utterances carried by ``channel_id``, in chronological order.

    Each row carries ``content``, ``truth_value``, ``speaker``, ``addressees``,
    ``fabula_time``, and ``syuzhet_index``.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    if channel_id not in ws.channels:
        return {"error": f"Channel {channel_id} not found."}

    ch = ws.channels[channel_id]

    def _name(nid: str) -> str:
        if nid in ws.entities:
            return ws.entities[nid].name
        if nid in ws.objects:
            return ws.objects[nid].name
        return nid

    utts = [
        evt for evt in ws.events
        if evt.event_type == "utterance" and evt.via_channel_id == channel_id
    ]
    utts.sort(key=lambda e: (e.fabula_time, e.syuzhet_index))

    rows = []
    for evt in utts[:limit]:
        speaker = evt.speaker_id or (evt.actor_ids[0] if evt.actor_ids else None)
        rows.append({
            "id": evt.id,
            "fabula_time": evt.fabula_time,
            "syuzhet_index": evt.syuzhet_index,
            "speaker": {"id": speaker, "name": _name(speaker)} if speaker else None,
            "addressees": [
                {"id": aid, "name": _name(aid)} for aid in evt.addressee_ids
            ],
            "content": evt.content,
            "truth_value": evt.truth_value,
            "description": evt.description,
        })
    return {
        "channel": {
            "id": channel_id,
            "name": ch.name,
            "medium": ch.medium,
            "directionality": ch.directionality,
        },
        "utterances": rows,
        "total": len(utts),
        "returned": len(rows),
    }


@mcp.tool()
@_safe_tool
def who_can_hear(
    ctx: Context,
    channel_id: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
    threshold: Optional[float] = None,
) -> dict:
    """Participants whose intelligibility on ``channel_id`` meets ``threshold``.

    Default threshold is ``physics.intelligibility_threshold`` from settings.
    Missing intelligibility entries default to 1.0 (fully intelligible).
    Result rows are split into ``addressable`` (intelligibility \u2265 threshold)
    and ``opaque`` (below threshold) for symmetry.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    if channel_id not in ws.channels:
        return {"error": f"Channel {channel_id} not found."}

    thr = float(threshold) if threshold is not None else float(
        _settings.physics.intelligibility_threshold
    )
    ch = ws.channels[channel_id]

    def _name(nid: str) -> str:
        if nid in ws.entities:
            return ws.entities[nid].name
        if nid in ws.objects:
            return ws.objects[nid].name
        return nid

    addressable: list[dict] = []
    opaque: list[dict] = []
    for pid_ in ch.participant_ids:
        intel = float(ch.intelligibility.get(pid_, 1.0))
        row = {"id": pid_, "name": _name(pid_), "intelligibility": intel}
        (addressable if intel >= thr else opaque).append(row)
    addressable.sort(key=lambda r: -r["intelligibility"])
    opaque.sort(key=lambda r: -r["intelligibility"])
    return {
        "channel_id": channel_id,
        "channel_name": ch.name,
        "threshold": thr,
        "addressable": addressable,
        "opaque": opaque,
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
    include_information_flow: bool = True,
) -> dict:
    """Trace causal chains upstream and/or downstream from a node.

    Args:
        node_id: The event or entity ID to trace from.
        direction: 'upstream' (causes), 'downstream' (effects), or 'both'.
        depth: Maximum traversal depth (default 3).
        include_information_flow: When True (default) also walks the
            epistemic provenance graph — utterance events and channels
            — in addition to ``ws.causal_topology``. Disable to recover
            the legacy causal-only behaviour.

    Returns a subgraph of causal edges with mechanism and force details,
    plus (when ``include_information_flow``) an ``information_flow``
    block listing utterance / channel edges traversed.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
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
    response: dict = {
        "root": node_id,
        "direction": direction,
        "depth": depth,
        "nodes": sorted(all_nodes),
        "edges": edges_out,
    }
    if include_information_flow:
        response["information_flow"] = trace_information_flow(
            ws, node_id, direction=direction, depth=depth,
        )
    return response


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
# AMWN BRANCHES — list / promote shadow forks (Story-integration plan, Step 6)
# =====================================================================


@mcp.tool()
@_safe_tool
def list_branches(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
) -> dict:
    """List all AMWN branches in the project's version DAG.

    Each branch entry includes ``world_id`` ('factual' or 'shadow'),
    ``branch_label``, the root version (and its fork point ancestor),
    the head version, and total version count on the branch.

    Use this before navigating shadow forks so the client knows which
    branch a version belongs to without walking the DAG itself.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}
    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}
    branches = db_list_branches(pid)
    return {"project_id": pid, "branches": branches}


@mcp.tool()
@_safe_tool
def promote_branch(
    ctx: Context,
    version_row_id: int,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Promote a shadow-branch version onto the factual mainline.

    Creates a new factual VersionRow whose world_state and prose are
    copied from the shadow source, with ``ancestor_id`` pointing at the
    current factual head. The shadow source remains untouched so the
    fork stays browsable.

    Returns the new factual version's row id and version number, plus
    its branch envelope.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}
    pid, err = resolve_project(
        project_id, project_name, ctx, min_role="editor",
    )
    if err:
        return {"error": err}
    # IDOR guard: promote_branch takes a row id but db_promote_branch
    # writes against the row's own project_id. Without this check a
    # caller could promote a shadow row from a project they have no
    # access to into that project's factual mainline.
    src_row = get_version_by_id(version_row_id)
    if src_row is None:
        return {"error": f"Version row {version_row_id} not found."}
    if src_row.project_id != pid:
        return {"error": (
            f"Version row {version_row_id} does not belong to project {pid}."
        )}
    user_row_id = get_user_id(ctx)
    try:
        promoted = db_promote_branch(
            version_row_id, user_id=user_row_id, description=description,
        )
    except VersionMutationError as e:
        return _sanitised_error("promote_branch", e)
    return {
        "project_id": pid,
        "version_row_id": promoted.id,
        "version": promoted.version,
        "ancestor_id": promoted.ancestor_id,
        "branch": {
            "world_id": promoted.world_id,
            "branch_label": promoted.branch_label,
            "ancestor_id": promoted.ancestor_id,
        },
    }


@mcp.tool()
@_safe_tool
def export_prose(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    branch_path: Optional[List[int]] = None,
) -> dict:
    """Return all prose for a project as an ordered list of versions.

    Without ``branch_path`` the result is the implicit linear history
    ordered by ``version`` — convenient for a quick read-through but
    mixes branches together.

    With ``branch_path`` (a list of ``version_row_id`` values) only
    those versions are returned, in the supplied order. This lets a
    client walk a specific lineage through the AMWN DAG — e.g. the
    factual prefix concatenated with a shadow fork's prose — to render
    one branch's narrative cleanly (Story-integration plan, Step 6).
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}
    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}
    entries = get_all_prose(pid, branch_path=branch_path)
    return {"project_id": pid, "entries": entries}


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
    pov_entity_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict:
    """Ask a read-only question about the story world.

    Runs the physics engine to analyze the world model without generating
    prose or creating a new version. Use this for analysis — use 'narrate'
    when you want to advance the story.

    Examples:
      "What does Macbeth believe about Lady Macbeth?"
      "Who is at the castle right now?"
      "What are the causal consequences of the murder?"

    ``mode`` selects the read-only query family:
      - ``"general"`` *(default)* — broad Q&A over the world graph;
        the engine answers in natural language without requiring
        explicit causal-bridge proof. This matches the UI's **Ask**
        mode.
      - ``"interrogate"`` — graph pathfinding with proof; returns
        Causal Bridges as evidence for "who knows X?" / "is there a
        path from A to B?" questions. This matches the UI's
        **Interrogation** mode.

    ``pov_entity_id`` (scaffold): if set, the world is filtered through that
    character's epistemic lens before analysis (utterances they could not
    plausibly hear and channels they don't participate in are pruned).
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    if pov_entity_id:
        ws = filter_world_state_for_pov(
            ws, pov_entity_id,
            intelligibility_threshold=_settings.physics.intelligibility_threshold,
        )

    qmode = (mode or "general").lower()
    if qmode not in ("general", "interrogate"):
        return {
            "error": (
                f"Unknown mode {mode!r}; expected 'general' or "
                f"'interrogate'."
            )
        }

    # Parse the question to resolve IDs
    parse_result = parse_query(
        question, query_type=qmode, world_state=ws,
        config=QueryParsingConfig(),
    )

    if not parse_result.is_valid or parse_result.query is None:
        # Fallback: build a typed query matching the caller's requested
        # mode. The previous fallback always constructed an
        # InterrogationQuery, which forced general-mode questions down
        # the pathfinding-with-proof path and produced wrong-shaped
        # answers for open-ended questions. Honour ``qmode`` so a
        # parse-failure in general mode still runs as a general query.
        if qmode == "general":
            query = GeneralQuery(
                question=question,
                original_query=question,
            )
        else:
            query = InterrogationQuery(
                question=question,
                require_proof=True,
                original_query=question,
            )
    else:
        query = parse_result.query

    # Run physics only — no generation, no version creation
    try:
        physics = calculate_narrative_physics(
            request=query,
            global_world_state=ws,
        )
    except Exception as e:
        return _sanitised_error("analyze", e)

    # Run the LLM Q&A step over the physics-state slice so the MCP
    # response carries a real natural-language answer (matches what
    # the UI Answer panel surfaces). Without this the tool would
    # bottom out at the physics envelope alone, which is exactly the
    # behaviour that made the UI's Ask/Interrogation feel broken.
    if isinstance(physics, dict) and physics.get("query_type") in (
        "interrogate", "general",
    ):
        try:
            from shadow_loom.answer import answer_question

            card = answer_question(
                question=question,
                physics_state=physics.get("physics_state"),
                query_type=physics.get("query_type", "interrogate"),
                require_proof=bool(getattr(query, "require_proof", False)),
            )
            physics["answer"] = card.answer
            physics["confidence"] = card.confidence
            physics["caveats"] = list(card.caveats)
            physics["evidence_node_ids"] = list(card.evidence_node_ids)
        except Exception:
            logger.exception("[MCP·ask] answer_question failed")

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
        if "confidence" in physics:
            result["confidence"] = physics["confidence"]
        if physics.get("caveats"):
            result["caveats"] = list(physics["caveats"])
        if physics.get("evidence_node_ids"):
            result["evidence_node_ids"] = list(physics["evidence_node_ids"])
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
    syuzhet_anchor: Optional[int] = None,
    target_vector_id: Optional[str] = None,
    pov_entity_id: Optional[str] = None,
) -> dict:
    """Compute narrative tension scores for the story or specific entities.

    Returns 4 structural and emotional scores (mystery, dramatic_irony,
    suspense, surprise) plus structural diagnostics (epistemic gaps,
    hidden information channels, narrative tensions, trait trajectories,
    relationship tensions).

    Args:
        entity_ids: Optional list of entity IDs to focus the analysis on.
            Defaults to the first six entities in the world.
        syuzhet_anchor: The reader's position in the syuzhet (reading
            order). When supplied, mystery / dramatic-irony / surprise
            are evaluated against what the reader has *seen so far*
            instead of the full story; suspense is computed against the
            unrevealed future.
        target_vector_id: Optional ``node.path.metric`` selector
            (e.g. ``ENT_001.traits.fear``). When set, the response
            includes a ``vector_state`` block with the current value
            for that vector so callers can plan a directive against it.

    The ``suspense_breakdown`` block exposes the per-entity threat /
    hope decomposition that drives the suspense score, including the
    most threatening unrevealed event, its probability, and the spatial
    distance to the focal entity.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    ws, _ = load_world_state_projected(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    if pov_entity_id:
        ws = filter_world_state_for_pov(
            ws, pov_entity_id,
            intelligibility_threshold=_settings.physics.intelligibility_threshold,
        )

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
            "syuzhet_anchor": syuzhet_anchor,
            "scores": {
                "mystery": round(
                    assembler.compute_mystery_score(entity_ids, syuzhet_anchor), 3
                ),
                "dramatic_irony": round(
                    assembler.compute_dramatic_irony_score(entity_ids, syuzhet_anchor), 3
                ),
                "suspense": round(
                    assembler.compute_suspense_score(entity_ids, syuzhet_anchor), 3
                ),
                "surprise": round(
                    assembler.compute_surprise_score(entity_ids, syuzhet_anchor), 3
                ),
                "narrative_tension": round(
                    assembler.compute_tension_score(entity_ids, syuzhet_anchor), 3
                ),
            },
            "epistemic_gaps": [g.model_dump() for g in assembler.compute_epistemic_gaps(entity_ids)],
            "narrative_tensions": [t.model_dump() for t in assembler.compute_narrative_tension(syuzhet_anchor)],
            "hidden_channels": [c.model_dump() for c in assembler.compute_hidden_channels(syuzhet_anchor)],
            "trait_trajectories": [t.model_dump() for t in assembler.compute_trait_trajectories(entity_ids)],
            "relationship_tensions": [t.model_dump() for t in assembler.compute_relationship_tensions(entity_ids)],
        }

        # Suspense breakdown — threat vs hope decomposition powering the
        # aggregate suspense score. Useful for directive callers that
        # want to pick the highest-leverage threat to escalate.
        try:
            tp = assembler._compute_threat_hope_detail(entity_ids, syuzhet_anchor)
            scores["suspense_breakdown"] = tp.model_dump()
        except Exception:
            logger.debug("threat/hope detail unavailable", exc_info=True)

        # Vector state lookup — when the caller pre-identified a target
        # node.metric, echo its current value so they can size a delta.
        if target_vector_id:
            scores["vector_state"] = _resolve_vector_state(ws, target_vector_id)

        return scores

    except Exception as e:
        return _sanitised_error("tension", e)


def _resolve_vector_state(ws: WorldStateV1, vector_id: str) -> dict:
    """Resolve a ``node.path.metric`` selector to its current value.

    Mirrors ``DirectiveAssembler._build_vector_constraint`` parsing so a
    caller can preview the state targeted by ``directive.target_vector_id``
    without invoking the full directive pipeline.
    """
    if "." not in vector_id:
        return {"target_vector_id": vector_id, "error": "expected 'node_id.path[.metric]'"}
    node_id, path = vector_id.split(".", 1)
    parts = path.split(".")

    if parts[0] == "traits" and len(parts) >= 2:
        ent = ws.entities.get(node_id)
        if ent is None:
            return {"target_vector_id": vector_id, "error": f"entity '{node_id}' not found"}
        tv = ent.traits.get(parts[1])
        if tv is None:
            return {"target_vector_id": vector_id, "error": f"trait '{parts[1]}' not found"}
        return {
            "target_vector_id": vector_id,
            "kind": "trait",
            "node_id": node_id,
            "trait": parts[1],
            "value": tv.value,
            "inertia": tv.inertia,
        }

    if parts[0] == "relationships" and len(parts) == 3:
        target_entity, metric = parts[1], parts[2]
        for e in ws.social_topology:
            if e.source_entity_id == node_id and e.target_entity_id == target_entity:
                # Pull per-axis state (value / inertia / evidence /
                # staleness) when the requested metric is one of the
                # closed RelationshipMetric axes; fall back to the
                # edge-level aggregate property for any unknown axis.
                m = e.metrics.get(metric) if metric in ("affinity", "fear", "power_dynamic") else None
                return {
                    "target_vector_id": vector_id,
                    "kind": "relationship",
                    "source_id": node_id,
                    "target_id": target_entity,
                    "metric": metric,
                    "value": (m.value if m is not None else getattr(e, metric, None)),
                    "inertia": (m.inertia if m is not None else e.inertia),
                    "evidence_strength": (m.evidence_strength if m is not None else e.evidence_strength),
                    "last_updated_fabula": (m.last_updated_fabula if m is not None else e.last_updated_fabula),
                    "observed": (m.observed if m is not None else False),
                }
        return {"target_vector_id": vector_id, "error": "relationship not found"}

    if parts[0] == "beliefs" and len(parts) >= 2:
        ent = ws.entities.get(node_id)
        if ent is None:
            return {"target_vector_id": vector_id, "error": f"entity '{node_id}' not found"}
        target = parts[1]
        belief = next((b for b in ent.beliefs if b.target_id == target), None)
        if belief is None:
            return {
                "target_vector_id": vector_id,
                "kind": "belief",
                "node_id": node_id,
                "belief_target_id": target,
                "value": None,
                "note": "no current belief on this target",
            }
        return {
            "target_vector_id": vector_id,
            "kind": "belief",
            "node_id": node_id,
            "belief_target_id": target,
            "perceived_state": belief.perceived_state,
            "confidence": belief.confidence,
        }

    return {"target_vector_id": vector_id, "error": f"unsupported path '{path}'"}


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

    ws_a, _ = load_world_state_projected(project_id, version_a, ctx=ctx)
    ws_b, _ = load_world_state_projected(project_id, version_b, ctx=ctx)
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

    # Per-axis relationship metric deltas — the headline mutation surface
    # for ``mutation_social`` and surgical ``do(rel.X.affinity=...)``
    # interventions. Without this view a user comparing two snapshots
    # sees zero deltas whenever the dyads' endpoint pairs are identical
    # but the per-axis values shifted significantly.
    rel_index_a = {
        f"{e.source_entity_id}->{e.target_entity_id}": e
        for e in ws_a.social_topology
    }
    rel_index_b = {
        f"{e.source_entity_id}->{e.target_entity_id}": e
        for e in ws_b.social_topology
    }
    rel_metric_changes: list[dict] = []
    for key in sorted(set(rel_index_a) & set(rel_index_b)):
        ra, rb = rel_index_a[key], rel_index_b[key]
        for axis in ("affinity", "fear", "power_dynamic"):
            ma = ra.metrics.get(axis)
            mb = rb.metrics.get(axis)
            if ma is None and mb is None:
                continue
            entry: dict = {"dyad": key, "axis": axis}
            if ma is None:
                entry["before"] = None
                entry["after"] = {
                    "value": mb.value, "evidence": mb.evidence_strength,
                    "observed": mb.observed,
                }
            elif mb is None:
                entry["before"] = {
                    "value": ma.value, "evidence": ma.evidence_strength,
                    "observed": ma.observed,
                }
                entry["after"] = None
            else:
                if (
                    abs(ma.value - mb.value) < 1e-9
                    and ma.evidence_strength == mb.evidence_strength
                    and ma.observed == mb.observed
                    and ma.last_updated_fabula == mb.last_updated_fabula
                ):
                    continue
                entry["before"] = {
                    "value": ma.value, "evidence": ma.evidence_strength,
                    "observed": ma.observed,
                    "last_updated_fabula": ma.last_updated_fabula,
                }
                entry["after"] = {
                    "value": mb.value, "evidence": mb.evidence_strength,
                    "observed": mb.observed,
                    "last_updated_fabula": mb.last_updated_fabula,
                }
                entry["value_delta"] = round(mb.value - ma.value, 4)
            rel_metric_changes.append(entry)
    diff["relationship_metric_changes"] = rel_metric_changes

    spatial_a = {f"{e.source_id}->{e.target_id}" for e in ws_a.spatial_topology}
    spatial_b = {f"{e.source_id}->{e.target_id}" for e in ws_b.spatial_topology}
    diff["spatial_edges_added"] = len(spatial_b - spatial_a)
    diff["spatial_edges_removed"] = len(spatial_a - spatial_b)

    channels_a = {
        f"{c.medium}|{','.join(sorted(c.participant_ids))}|{c.established_at_fabula}"
        for c in ws_a.channels.values()
    }
    channels_b = {
        f"{c.medium}|{','.join(sorted(c.participant_ids))}|{c.established_at_fabula}"
        for c in ws_b.channels.values()
    }
    diff["channels_added"] = len(channels_b - channels_a)
    diff["channels_removed"] = len(channels_a - channels_b)
    utt_a = {e.id for e in ws_a.events if e.event_type == "utterance"}
    utt_b = {e.id for e in ws_b.events if e.event_type == "utterance"}
    diff["utterance_events_added"] = len(utt_b - utt_a)
    diff["utterance_events_removed"] = len(utt_a - utt_b)

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
    skip_audit: bool = _settings.mcp.skip_audit,
    force_implausible: bool = False,
    speaker_id: Optional[str] = None,
    addressee_ids: Optional[List[str]] = None,
    via_channel_id: Optional[str] = None,
) -> dict:
    """Generate prose and advance the story using natural language.

    This is the primary creation tool. Your instruction is parsed,
    run through physics simulation, and rendered as prose. A new version
    is created.

    Args:
        instruction: Natural language instruction (e.g., "Continue the
            story from Macbeth's perspective" or "Kill Duncan").
        mode: Force a query type \u2014 'observe', 'intervene', 'counterfactual',
              or None for auto-detect.
        skip_audit: Skip the audit loop for faster results (default True).
        force_implausible: For Rung-2/3 queries, generate prose even when the
            engine cannot resolve the requested targets against the world
            (the implausibility reason is still reported on the response).
        speaker_id: Optional ENT_ id constraining the utterance event the
            generator may produce. Useful for "X says Y to Z via the coded
            telegram" prompts where the parser would otherwise have to
            infer the channel from prose.
        addressee_ids: Optional list of ENT_ ids the speaker intends to
            reach (distinct from overhearers, which are derived from
            channel intelligibility).
        via_channel_id: Optional CHN_ id pinning the utterance to a
            specific standing channel.

    Reports progress via MCP progress notifications.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    pid, err = resolve_project(
        project_id, project_name, ctx, min_role="editor",
    )
    if err:
        return {"error": err}

    ws, ancestor_row_id = load_world_state(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found. Use 'ingest' first."}

    user_row_id = get_user_id(ctx)

    # Append structured channel/speaker hints to the instruction so the
    # parser produces an utterance-typed event with the requested
    # provenance. Validated against the loaded ws so callers get fast
    # feedback on stale ids.
    hints: list[str] = []
    if speaker_id:
        if speaker_id not in ws.entities:
            return {"error": f"speaker_id {speaker_id} not in world entities."}
        hints.append(f"speaker_id={speaker_id}")
    if addressee_ids:
        bad = [a for a in addressee_ids if a not in ws.entities]
        if bad:
            return {"error": f"addressee_ids not found: {bad}"}
        hints.append("addressee_ids=" + ",".join(addressee_ids))
    if via_channel_id:
        if via_channel_id not in ws.channels:
            return {"error": f"via_channel_id {via_channel_id} not in world channels."}
        hints.append(f"via_channel_id={via_channel_id}")
    if hints:
        instruction = (
            f"{instruction}\n\n"
            f"[utterance constraints: {'; '.join(hints)}]"
        )

    # Map mode to query_type.
    # AUDIT (post-2026-05-26): contract drift fix — ``narrate`` previously
    # accepted any string and silently fell back to ``None`` (then to the
    # parser default ``directive``). Now we validate strictly, mirroring
    # the ``ask`` tool: an unknown mode is rejected with a typed error.
    mode_map = {
        "observe": "observation",
        "intervene": "intervention",
        "counterfactual": "counterfactual",
        "directive": "directive",
    }
    if mode is not None and mode not in mode_map:
        return {
            "error": (
                f"unknown narrate mode {mode!r}; expected one of "
                f"{sorted(mode_map)}"
            ),
            "code": "INVALID_MODE",
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

    # Apply force_implausible override on supported query types.
    if force_implausible and hasattr(parse_result.query, "force_implausible"):
        parse_result.query = parse_result.query.model_copy(
            update={"force_implausible": True}
        )

    # Stage 2–4: Pipeline
    await ctx.report_progress(2, 4, "Running physics simulation...")

    # ``run_and_save`` is synchronous and runs an LLM + physics pass
    # that can take several seconds; offload it so the MCP event loop
    # stays responsive to other concurrent requests.
    response = await asyncio.to_thread(
        run_and_save,
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
    skip_audit: bool = _settings.mcp.skip_audit,
    force_implausible: bool = False,
) -> dict:
    """Generate a scene optimized for a specific emotional effect.

    Available effects: mystery, dramatic_irony, suspense, surprise,
    grief, rage, joy, fear, love, regret.

    Args:
        target_effect: The emotional effect to optimize for.
        entity_ids: Entities to focus on (default: all).
        intensity: Effect intensity 0.0–1.0 (default 0.8).
        skip_audit: Skip the audit loop (default True).
        force_implausible: Generate prose even when none of ``entity_ids``
            exist in the world (a fallback POV is used).
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    pid, err = resolve_project(
        project_id, project_name, ctx, min_role="editor",
    )
    if err:
        return {"error": err}

    ws, ancestor_row_id = load_world_state(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    valid_effects = {
        "mystery", "dramatic_irony", "suspense", "surprise", "narrative_tension",
        "grief", "rage", "joy", "fear", "love", "regret",
    }
    if target_effect not in valid_effects:
        return {"error": f"Invalid effect '{target_effect}'. Choose from: {', '.join(sorted(valid_effects))}"}

    user_row_id = get_user_id(ctx)

    query = DirectiveQuery(
        target_entity_ids=entity_ids or [],
        target_effect=target_effect,
        intensity=intensity,
        force_implausible=force_implausible,
        original_query=f"Directive: {target_effect} (intensity={intensity})",
    )

    await ctx.report_progress(1, 3, f"Assembling {target_effect} directive...")

    # Offload synchronous pipeline work so the event loop can keep
    # serving other requests during the multi-second LLM + physics run.
    response = await asyncio.to_thread(
        run_and_save,
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
    insert_after_event_id: Optional[str] = None,
    insert_at_fabula_time: Optional[int] = None,
    replace_event_ids: Optional[List[str]] = None,
    replace_entity_ids: Optional[List[str]] = None,
    replace_object_ids: Optional[List[str]] = None,
    replace_location_ids: Optional[List[str]] = None,
    replace_world_trait_ids: Optional[List[str]] = None,
    replace_channel_ids: Optional[List[str]] = None,
    replace_proposition_ids: Optional[List[str]] = None,
    replace_concern_ids: Optional[List[List[str]]] = None,
    focus_entity_ids: Optional[List[str]] = None,
) -> dict:
    """Apply user-written prose as a manual edit to the world model.

    The prose is re-extracted into topology and merged into the world
    model. Creates a new version. Use this when you want to write
    narrative directly rather than having the engine generate it.

    Timeline anchoring (optional — choose at most one of the first two):

    - ``insert_at_fabula_time``: explicit fabula_time anchor. New events
      extracted from ``prose`` are placed at this time and onwards.
      Use for backfilling history or inserting between known beats.
    - ``insert_after_event_id``: the new events anchor at
      ``event.fabula_time + extraction.fabula_time_spacing``. Convenient
      when you know which existing event the edit should follow.
    - When both are omitted, the edit appends after the current
      chronological end (``max(events.fabula_time) + spacing``).

    Replace semantics (true replacement of existing graph nodes — every
    list cascades dependent edges/snapshots/concerns through the merge
    deletion pass; leave empty for purely additive edits):

    - ``replace_event_ids``: EVT_ ids to drop.
    - ``replace_entity_ids``: ENT_ ids to drop.
    - ``replace_object_ids``: OBJ_ ids to drop.
    - ``replace_location_ids``: LOC_ ids to drop.
    - ``replace_world_trait_ids``: WORLD_ ids to drop.
    - ``replace_channel_ids``: CHN_ ids to drop.
    - ``replace_proposition_ids``: PROP_ ids to drop.
    - ``replace_concern_ids``: list of ``[entity_id, concern_id]`` pairs
      to drop (concerns are scoped to a holding entity).

    ``focus_entity_ids`` records the entities most affected by the edit
    so downstream views (ego-graph scoping, version diffs) can highlight
    them; it does not change the merge itself.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    over = _enforce_word_cap(prose, field="prose")
    if over:
        return over

    pid, err = resolve_project(
        project_id, project_name, ctx, min_role="editor",
    )
    if err:
        return {"error": err}

    ws, ancestor_row_id = load_world_state(pid, version, ctx=ctx)
    if ws is None:
        return {"error": "No world model found."}

    # Validate timeline anchoring inputs against the loaded world so
    # the caller gets an actionable error instead of a silent fallback
    # to "append at chronological end".
    if insert_after_event_id is not None:
        if not any(e.id == insert_after_event_id for e in ws.events):
            return {
                "error": (
                    f"insert_after_event_id '{insert_after_event_id}' "
                    f"not found in world model v{version if version is not None else 'latest'}."
                )
            }
    if replace_event_ids:
        known_ids = {e.id for e in ws.events}
        missing = [eid for eid in replace_event_ids if eid not in known_ids]
        if missing:
            return {
                "error": (
                    f"replace_event_ids not found in world model: "
                    f"{', '.join(missing)}"
                )
            }

    # Validate the broader replace_* surfaces against the loaded world
    # so the caller gets an actionable error rather than silently
    # noop-ing a typo'd id at merge time.
    def _check(name: str, ids: list[str] | None, known: set[str]) -> dict | None:
        if not ids:
            return None
        missing = [i for i in ids if i not in known]
        if missing:
            return {"error": f"{name} not found in world model: {', '.join(missing)}"}
        return None

    for name, ids, known in [
        ("replace_entity_ids", replace_entity_ids, set(ws.entities.keys())),
        ("replace_object_ids", replace_object_ids, set(ws.objects.keys())),
        ("replace_location_ids", replace_location_ids, set(ws.locations.keys())),
        ("replace_world_trait_ids", replace_world_trait_ids, set(ws.world_traits.keys())),
        ("replace_channel_ids", replace_channel_ids, set(ws.channels.keys())),
        ("replace_proposition_ids", replace_proposition_ids, {p.proposition_id for p in ws.propositions}),
    ]:
        err = _check(name, ids, known)
        if err:
            return err
    if replace_concern_ids:
        known_concerns = {
            (eid, c.concern_id)
            for eid, ent in ws.entities.items()
            for c in ent.concerns
        }
        missing_pairs = [
            (eid, cid) for eid, cid in (tuple(p) for p in replace_concern_ids)
            if (eid, cid) not in known_concerns
        ]
        if missing_pairs:
            return {
                "error": (
                    "replace_concern_ids pairs not found in world model: "
                    + ", ".join(f"({eid}, {cid})" for eid, cid in missing_pairs)
                )
            }

    user_row_id = get_user_id(ctx)
    query = ManualEditQuery(
        edited_prose=prose,
        description=description,
        focus_entity_ids=focus_entity_ids or [],
        insert_after_event_id=insert_after_event_id,
        insert_at_fabula_time=insert_at_fabula_time,
        replace_event_ids=replace_event_ids or [],
        replace_entity_ids=replace_entity_ids or [],
        replace_object_ids=replace_object_ids or [],
        replace_location_ids=replace_location_ids or [],
        replace_world_trait_ids=replace_world_trait_ids or [],
        replace_channel_ids=replace_channel_ids or [],
        replace_proposition_ids=replace_proposition_ids or [],
        replace_concern_ids=[tuple(p) for p in (replace_concern_ids or [])],
        original_query=description or prose,
    )

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

    over = _enforce_word_cap(text, field="text")
    if over:
        return over

    user_row_id = get_user_id(ctx)

    config = ExtractionConfig(
        fabula_time_spacing=_settings.mcp.ingest_fabula_time_spacing,
        max_correction_retries=_settings.mcp.ingest_max_correction_retries,
    )

    await ctx.report_progress(1, 3, "Extracting ontology and topology...")

    try:
        ws, report = await run_extraction_async(text, config)
    except Exception as e:
        return _sanitised_error("ingest", e)

    await ctx.report_progress(2, 3, "Saving project...")

    proj = create_project(
        name=project_name, owner_id=user_row_id, label=label, raw_text=text,
    )

    try:
        save_version(
            project_id=proj.id,
            world_state_json=ws.model_dump_json(),
            version=0,
            source="ingestion",
            description="Initial ingestion",
            user_id=user_row_id,
        )
    except Exception as e:
        # Don't leave an orphan project with zero versions if the
        # initial save fails — the row would be permanently broken.
        logger.exception(
            "Initial save_version failed for project %s; rolling back project row",
            proj.id,
        )
        try:
            db_delete_project(proj.id)
        except Exception:
            logger.exception("Failed to roll back orphan project %s", proj.id)
        return _sanitised_error("ingest", e)

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
        "channels": len(ws.channels),
        "utterance_events": sum(1 for e in ws.events if e.event_type == "utterance"),
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

    ws, ver_row_id = load_world_state_projected(pid, version, ctx=ctx)
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

        # Collect all prose for evaluation — scoped to the loaded
        # version's lineage so a shadow-branch evaluation grades the
        # shadow's own prose (plus the factual prefix it diverged
        # from), not an arbitrary linear union of every branch's prose
        # for the project. Without this scoping the scorecard is
        # contaminated by sibling branches the user is not asking
        # about.
        branch_path: Optional[List[int]] = None
        if ver_row_id is not None:
            try:
                branch_path = get_lineage_to_root(ver_row_id) or None
            except Exception:
                logger.exception(
                    "[evaluate] Failed to build branch_path for ver_row_id=%s; "
                    "falling back to project-wide prose.",
                    ver_row_id,
                )
                branch_path = None
        prose_list = get_all_prose(pid, branch_path=branch_path)
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
        return _sanitised_error("evaluate", e)


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
# GROUP 5b: RESEARCH — "Look up real-world background on a topic."
# =====================================================================
#
# Research tools query an external provider (Tavily by default) and
# distil the results into ``WorldFact`` records that live in a
# segregated ``WorldStateV1.world_facts`` collection. They never mutate
# Entities, Events, RelationshipEdges or world traits. See
# ``docs/research-extraction-plan.md`` for the rationale and
# ``shadow_loom/research.py`` for the provider layer.


@mcp.tool()
@_safe_tool
def research_topic(
    ctx: Context,
    topic: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    provider: Optional[str] = None,
    max_results: Optional[int] = None,
) -> dict:
    """Look up *topic* via the configured research provider and persist a WorldFact.

    Calls ``provider.search(topic)``, runs the research-extraction agent
    to distil the snippets into a single ``WorldFact``, caches the raw
    provider call (per-user) and persists the resulting fact under the
    project. Requires ``write`` scope and at least the ``editor``
    project role.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    pid, err = resolve_project(
        project_id, project_name, ctx, min_role="editor",
    )
    if err:
        return {"error": err}

    user_id = get_user_id(ctx)
    from shadow_loom.research import lookup_and_persist_topic

    return lookup_and_persist_topic(
        project_id=pid,
        user_id=user_id,
        topic=topic,
        provider_override=provider,
        max_results_override=max_results,
    )


@mcp.tool()
@_safe_tool
def list_world_facts(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
) -> dict:
    """List all ``WorldFact`` records for a project."""
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}
    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    import json as _json
    rows = db_list_world_facts(pid)
    facts = []
    for r in rows:
        try:
            related = _json.loads(r.related_node_ids_json) if r.related_node_ids_json else []
        except (ValueError, TypeError) as e:
            logger.warning(
                "Corrupt related_node_ids_json on world fact %s: %s",
                r.fact_id, e,
            )
            related = []
        facts.append({
            "fact_id": r.fact_id,
            "topic": r.topic,
            "summary": r.summary,
            "confidence": r.confidence,
            "source_url_primary": r.source_url_primary,
            "provider": r.provider,
            "related_node_ids": related,
            "retrieved_at": str(r.retrieved_at) if r.retrieved_at else None,
        })
    return {"project_id": pid, "facts": facts, "count": len(facts)}


@mcp.tool()
@_safe_tool
def delete_world_fact(
    ctx: Context,
    fact_id: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
) -> dict:
    """Delete a ``WorldFact`` by id from a project."""
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}
    pid, err = resolve_project(
        project_id, project_name, ctx, min_role="editor",
    )
    if err:
        return {"error": err}
    removed = db_delete_world_fact(project_id=pid, fact_id=fact_id)
    return {"project_id": pid, "fact_id": fact_id, "deleted": removed}


@mcp.tool()
@_safe_tool
def get_project_settings(
    ctx: Context,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
) -> dict:
    """Return per-project settings (currently: research_topics).

    Project settings live outside ``WorldStateV1`` so they do not fork
    with shadow branches and do not bloat version snapshots.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}
    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}
    settings = db_get_project_settings(pid)
    return {"project_id": pid, **settings}


@mcp.tool()
@_safe_tool
def set_project_settings(
    ctx: Context,
    research_topics: List[str],
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
) -> dict:
    """Replace the project's ``research_topics`` list.

    Topics are stripped + de-duplicated. Pass an empty list to clear.
    Editing settings does not mutate any version row.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}
    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}
    err = check_project_access(pid, ctx, min_role="editor")
    if err:
        return {"error": err}
    try:
        db_set_project_settings(pid, research_topics=list(research_topics))
    except ValueError as e:
        return _sanitised_error("set_research_topics", e)
    return {"project_id": pid, **db_get_project_settings(pid)}


@mcp.tool()
@_safe_tool
def get_research_status(ctx: Context) -> dict:
    """Return process-wide research-extraction status.

    Reports whether the research agent is enabled, which provider is
    configured, and whether the provider API key is present. Never
    returns the API key itself.
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}
    settings = _get_settings()
    extraction = settings.extraction
    provider = extraction.research_provider
    api_key_present = False
    if provider == "tavily":
        # ``tavily_api_key`` lives on the ``core`` sub-settings group
        # (CoreSettings), not on the top-level Settings facade. The old
        # ``getattr(settings, "tavily_api_key", "")`` silently returned ""
        # for every caller, so api_key_present was always False even
        # when correctly configured.
        api_key_present = bool(getattr(settings.core, "tavily_api_key", "") or "")
    return {
        "enabled": bool(extraction.enable_research_agent),
        "provider": provider,
        "provider_model": extraction.research_provider_model,
        "max_results_per_query": extraction.research_max_results_per_query,
        "api_key_present": api_key_present,
        "default_topics": list(extraction.research_topics),
    }


# =====================================================================
# GROUP 6: MANAGE — "Branch, share, fork, configure."
# =====================================================================


@mcp.tool()
@_safe_tool
def patch_world_state(
    ctx: Context,
    patch: dict,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    version: Optional[int] = None,
    description: str = "",
) -> dict:
    """Apply a structured ``WorldStatePatch`` directly to the world model.

    Bypasses the prose re-extraction round-trip when the change is
    already known structurally (e.g. backfilling ``Belief.proposition_id``,
    committing a proposition truth, renaming a channel, fixing a
    miswired event field). Creates a new version on success.

    The ``patch`` argument is a dict mirroring :class:`WorldStatePatch`.
    Supported ops (all optional, all default to no-op):

      Structural fixes:
        * ``event_renames``: ``{old_evt_id: new_evt_id}``
        * ``drop_event_ids``: ``[evt_id, ...]``
        * ``update_event_fields``: ``{evt_id: {field: value, ...}}``
        * ``update_entity_location``: ``{entity_id: location_id}``
        * ``add_state_timeline_entries``: ``{entity_id: [snapshot_dict, ...]}``

      Edge / channel surgery:
        * ``drop_causal_edges`` / ``add_causal_edges``
        * ``drop_social_edges`` / ``add_social_edges``
        * ``drop_spatial_edges`` / ``add_spatial_edges``
        * ``drop_channel_ids`` / ``add_channels`` / ``channel_renames``

      Proposition / concern / belief layer (post-affect-unification):
        * ``add_propositions``: ``{prop_id: proposition_dict}``
        * ``update_proposition_snapshots``: ``{prop_id: [snapshot_dict, ...]}``
        * ``commit_proposition_truth``: ``{prop_id: {fabula_time: bool}}``
        * ``add_concerns``: ``{entity_id: [concern_dict, ...]}``
        * ``update_concern_snapshots``: ``{entity_id: {concern_id: [snapshot_dict, ...]}}``
        * ``set_belief_proposition_ids``: ``[{entity_id, target_id, proposition_id}, ...]``

      Free-text:
        * ``notes``: rationale stored in the version description.

    Returns ``{new_version, changes: [...]}`` on success or
    ``{error}`` if the patch fails to validate or apply.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    pid, err = resolve_project(project_id, project_name, ctx)
    if err:
        return {"error": err}

    err = check_project_access(pid, ctx, min_role="editor")
    if err:
        return {"error": err}

    ws, ancestor_row_id, ancestor_world_id, ancestor_branch_label = (
        load_world_state_with_branch(pid, version, ctx=ctx)
    )
    if ws is None:
        return {"error": "No world model found."}

    try:
        typed_patch = WorldStatePatch.model_validate(patch)
    except Exception as exc:
        return _sanitised_error("patch_world_state", exc)

    try:
        new_ws, changes = _apply_world_state_patch(ws, typed_patch)
    except Exception as exc:
        return _sanitised_error("patch_world_state", exc)

    user_row_id = get_user_id(ctx)
    desc = description or typed_patch.notes or (
        f"Structured patch ({len(changes)} change(s))"
    )
    try:
        ver = save_version(
            project_id=pid,
            world_state_json=new_ws.model_dump_json(),
            ancestor_id=ancestor_row_id,
            source="patch_world_state",
            description=desc,
            user_id=user_row_id,
            # Carry the ancestor row's branch identity so a patch on
            # a shadow row stays on that shadow branch instead of
            # silently demoting onto the factual mainline via the
            # ``save_version`` default ``world_id='factual'``.
            world_id=ancestor_world_id,
            branch_label=ancestor_branch_label,
        )
    except Exception as exc:
        return _sanitised_error("patch_world_state", exc)

    return {
        "project_id": pid,
        "new_version": ver.version,
        "changes": changes,
        "change_count": len(changes),
    }


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

    err = check_project_access(project_id, ctx, min_role="editor")
    if err:
        return {"error": err}

    ws, ancestor_row_id, ancestor_world_id, ancestor_branch_label = (
        load_world_state_with_branch(project_id, from_version, ctx=ctx)
    )
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
        # Carry the ancestor's branch identity onto the child node.
        # Without this, branching from a shadow row would silently
        # demote the child onto the factual mainline at the
        # ``save_version`` default ``world_id='factual'`` and the
        # next pipeline run on that child would author against the
        # wrong AMWN world.
        world_id=ancestor_world_id,
        branch_label=ancestor_branch_label,
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
    err = check_project_access(project_id, ctx, min_role="admin")
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
    # search_users uses substring match on username/email, so the first
    # hit may be a different user whose name *contains* the requested
    # one (e.g. 'malice' for 'alice'). Require an exact-username match
    # before sharing.
    exact = [t for t in targets if t.get("username") == username]
    if not exact:
        return {"error": f"User '{username}' not found."}

    # Validate role against the documented enum so unknown values are
    # rejected before reaching the DB layer.
    if role not in {"viewer", "editor", "admin"}:
        return {"error": (
            f"Invalid role '{role}'. Must be one of viewer, editor, admin."
        )}

    target = exact[0]
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


@mcp.tool()
@_safe_tool
def delete_project(
    ctx: Context,
    project_id: int,
) -> dict:
    """Permanently delete a project and all of its versions.

    Owner-only and irreversible. Cascades the project's versions,
    activity log, stars, and member access. Refuses to delete the
    seeded example projects.

    Returns ``{"status": "deleted", "project_id": <int>}`` on success
    or ``{"error": "..."}`` on failure.
    """
    err = require_scope(ctx, "admin")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)
    if user_row_id is None:
        return {"error": "Authentication required to delete projects."}

    err = check_project_access(project_id, ctx, min_role="admin")
    if err:
        return {"error": err}

    try:
        db_delete_project(project_id, user_row_id)
    except ProjectDeleteError as exc:
        return {"error": str(exc)}
    except PermissionError as exc:
        return {"error": str(exc)}

    return {"status": "deleted", "project_id": project_id}


@mcp.tool()
@_safe_tool
def delete_version(
    ctx: Context,
    version_row_id: int,
    cascade: bool = False,
) -> dict:
    """Delete a single version from a project's version tree.

    By default the version's children are *re-parented* onto the deleted
    version's parent so the tree remains connected (a "rejoin"). Set
    ``cascade=True`` to delete every descendant as well.

    The root version (v0) cannot be deleted. Owner-or-editor only.

    Returns ``{"deleted": [ids], "reparented": {child_id: new_parent_id}}``
    on success, or ``{"error": "..."}`` on failure.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)
    if user_row_id is None:
        return {"error": "Authentication required to delete versions."}

    # Locate the owning project for the access check.
    row = get_version_by_id(version_row_id)
    if row is None:
        return {"error": f"Version {version_row_id} not found."}
    err = check_project_access(row.project_id, ctx, min_role="editor")
    if err:
        return {"error": err}

    try:
        result = db_delete_version(
            version_row_id, user_row_id, cascade=cascade,
        )
    except VersionMutationError as exc:
        return {"error": str(exc)}
    except PermissionError as exc:
        return {"error": str(exc)}

    return {
        "deleted": result["deleted"],
        "reparented": result["reparented"],
        "project_id": row.project_id,
    }


@mcp.tool()
@_safe_tool
def reparent_version(
    ctx: Context,
    version_row_id: int,
    new_ancestor_id: Optional[int],
) -> dict:
    """Move a version under a new ancestor (rejoin / branch graft).

    Both versions must belong to the same project. The operation is
    rejected if it would create a cycle (i.e. ``new_ancestor_id`` is a
    descendant of ``version_row_id``) or if the version is the root.

    ``new_ancestor_id`` is required and must be a non-null version
    row ID belonging to the same project. Passing ``null`` is
    rejected because every non-root version must have an ancestor.

    Owner-or-editor only.
    """
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)
    if user_row_id is None:
        return {"error": "Authentication required to reparent versions."}

    row = get_version_by_id(version_row_id)
    if row is None:
        return {"error": f"Version {version_row_id} not found."}
    err = check_project_access(row.project_id, ctx, min_role="editor")
    if err:
        return {"error": err}

    try:
        db_reparent_version(version_row_id, new_ancestor_id, user_row_id)
    except VersionMutationError as exc:
        return {"error": str(exc)}
    except PermissionError as exc:
        return {"error": str(exc)}

    return {
        "status": "reparented",
        "version_row_id": version_row_id,
        "new_ancestor_id": new_ancestor_id,
        "project_id": row.project_id,
    }


@mcp.tool()
@_safe_tool
def set_active_version(
    ctx: Context,
    project_id: int,
    version: Optional[int] = None,
    version_row_id: Optional[int] = None,
) -> dict:
    """Set the caller's active version pointer for a project.

    All subsequent read tools (``open_project``, ``inspect``, ``ask``,
    ``compute_tension``, etc.) called *without* an explicit ``version``
    will default to this row instead of the project's latest version.

    Specify *either* ``version`` (sequential project-scoped number) or
    ``version_row_id`` (the underlying DB row id). The pointer is
    per-user-per-project and survives across tool calls.

    Pass nothing besides ``project_id`` to *clear* the pointer (so
    subsequent reads fall back to the latest version).
    """
    # Round-7 audit: this endpoint mutates per-user state
    # (``active_versions``) and is therefore a write operation, not a
    # read. Requiring the ``write`` scope keeps read-only tokens from
    # silently redirecting another caller's default version.
    err = require_scope(ctx, "write")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)
    if user_row_id is None:
        return {"error": "Authentication required to set active version."}

    err = check_project_access(project_id, ctx, min_role="editor")
    if err:
        return {"error": err}

    if version is None and version_row_id is None:
        cleared = db_clear_active_version(project_id, user_row_id)
        return {
            "status": "cleared" if cleared else "noop",
            "project_id": project_id,
        }

    if version_row_id is None:
        ver = get_version(project_id, version)
        if ver is None:
            return {"error": f"Version {version} not found in project {project_id}."}
        version_row_id = ver.id

    try:
        row = db_set_active_version(project_id, user_row_id, version_row_id)
    except ValueError as exc:
        return {"error": str(exc)}

    return {
        "status": "set",
        "project_id": project_id,
        "version_row_id": row.version_row_id,
    }


@mcp.tool()
@_safe_tool
def get_active_version(ctx: Context, project_id: int) -> dict:
    """Return the caller's active-version pointer for a project, if any.

    Returns ``{"active": false, ...}`` when no pointer is set (read tools
    will fall back to the project's latest version).
    """
    err = require_scope(ctx, "read")
    if err:
        return {"error": err}

    user_row_id = get_user_id(ctx)
    if user_row_id is None:
        return {"error": "Authentication required."}

    err = check_project_access(project_id, ctx)
    if err:
        return {"error": err}

    ver = db_get_active_version(project_id, user_row_id)
    if ver is None:
        return {"active": False, "project_id": project_id}
    return {
        "active": True,
        "project_id": project_id,
        "version": ver.version,
        "version_row_id": ver.id,
    }


# =====================================================================
# CONSOLIDATED DISPATCHERS — coarse-grained tools for callers
# =====================================================================
# These wrap the granular tools above behind action-discriminated
# entry points. Behavior is identical — they delegate to the existing
# functions. The granular tools remain registered for backward
# compatibility but are marked DEPRECATED in their docstrings; prefer
# the dispatchers below for new integrations.


@mcp.tool()
@_safe_tool
def discover(
    ctx: Context,
    scope: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    payload: Optional[dict] = None,
) -> dict:
    """Coarse-grained discovery — "what can I work with?".

    Args:
        scope: One of:
            - ``"projects"`` — list every project the caller can read.
            - ``"branches"`` — list AMWN branches in the project.
            - ``"channels"`` — list standing channels in the project.
            - ``"world_facts"`` — list segregated background facts.
            - ``"propositions"`` — list shared propositions; pass
              ``payload={"at_time": <int>}`` to fold each one through
              :func:`reconstruct_proposition_at` for that fabula tick.
            - ``"concerns"`` — list per-entity concerns; ``payload`` may
              contain ``at_time`` (replay each concern), ``entity_id``
              (filter to one holder), and ``only_active`` (drop concerns
              whose activation window excludes ``at_time``).
            - ``"superseded_events"`` — list every event with a
              ``superseded_by_event_id`` and the chain that supersedes
              it. Useful after a counterfactual was promoted onto the
              factual mainline.
        project_id / project_name: Required for every scope except
            ``"projects"``.
        payload: Reserved for scope-specific filters. See per-scope
            descriptions above.

    Returns the same envelope as the underlying granular tool.
    """
    payload = payload or {}
    if scope == "projects":
        return list_projects(ctx)
    if scope == "branches":
        return list_branches(ctx, project_id=project_id, project_name=project_name)
    if scope == "channels":
        return list_channels(
            ctx,
            project_id=project_id,
            project_name=project_name,
            version=payload.get("version"),
        )
    if scope == "world_facts":
        return list_world_facts(
            ctx, project_id=project_id, project_name=project_name,
        )
    if scope in ("propositions", "concerns", "superseded_events"):
        err = require_scope(ctx, "read")
        if err:
            return {"error": err}
        pid, err = resolve_project(project_id, project_name, ctx)
        if err:
            return {"error": err}
        ws, _ = load_world_state_projected(pid, payload.get("version"), ctx=ctx)
        if ws is None:
            return {"error": "No world model found."}
        at_time = payload.get("at_time")
        if scope == "propositions":
            items = []
            for prop in ws.propositions:
                row = {
                    "id": prop.proposition_id,
                    "kind": prop.kind,
                    "description": prop.description,
                    "world_id": prop.world_id,
                    "stakes": prop.stakes,
                    "audience_default_prior": prop.audience_default_prior,
                    "referent_ids": list(prop.referent_ids),
                    "truth_at_fabula": dict(prop.truth_at_fabula),
                    "state_timeline_count": len(prop.state_timeline),
                }
                if at_time is not None:
                    row["snapshot"] = reconstruct_proposition_at(prop, int(at_time))
                items.append(row)
            return {"propositions": items, "count": len(items)}
        if scope == "concerns":
            entity_filter = payload.get("entity_id")
            only_active = bool(payload.get("only_active"))
            items = []
            for ent_id, ent in ws.entities.items():
                if entity_filter and ent_id != entity_filter:
                    continue
                for c in ent.concerns:
                    row = {
                        "id": c.concern_id,
                        "entity_id": ent_id,
                        "entity_name": ent.name,
                        "proposition_id": c.proposition_id,
                        "polarity": c.polarity,
                        "salience": c.salience,
                        "kind": c.kind,
                        "world_id": c.world_id,
                        "counter_concern_ids": list(
                            getattr(c, "counter_concern_ids", []) or []
                        ),
                        "activation_fabula_window": (
                            list(c.activation_fabula_window)
                            if getattr(c, "activation_fabula_window", None)
                            else None
                        ),
                        "state_timeline_count": len(c.state_timeline),
                    }
                    if at_time is not None:
                        snap = reconstruct_concern_at(c, int(at_time))
                        if only_active and not snap["active"]:
                            continue
                        row["snapshot"] = snap
                    items.append(row)
            return {"concerns": items, "count": len(items)}
        # superseded_events
        chains: list[dict] = []
        successors_by_id = {e.id: e for e in ws.events}
        for evt in ws.events:
            sup = getattr(evt, "superseded_by_event_id", None)
            if not sup:
                continue
            chain = [evt.id]
            cursor = sup
            seen = {evt.id}
            while cursor and cursor not in seen:
                seen.add(cursor)
                chain.append(cursor)
                nxt = successors_by_id.get(cursor)
                cursor = (
                    getattr(nxt, "superseded_by_event_id", None)
                    if nxt is not None else None
                )
            chains.append({
                "event_id": evt.id,
                "fabula_time": evt.fabula_time,
                "syuzhet_index": evt.syuzhet_index,
                "description": evt.description,
                "superseded_by_event_id": sup,
                "successor_chain": chain[1:],
            })
        return {"superseded_events": chains, "count": len(chains)}
    return {"error": (
        f"Unknown discover scope {scope!r}. Expected one of: "
        "projects, branches, channels, world_facts, propositions, "
        "concerns, superseded_events."
    )}


@mcp.tool()
@_safe_tool
def trace(
    ctx: Context,
    kind: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    payload: Optional[dict] = None,
) -> dict:
    """Follow a chain through the project — causal, history, or channel.

    Args:
        kind: One of:
            - ``"causal"`` — wraps :func:`trace_causality`. Payload keys:
              ``node_id`` (required), ``version``, ``direction``
              (``upstream``/``downstream``/``both``), ``depth``,
              ``include_information_flow``.
            - ``"history"`` — wraps :func:`get_history`. Payload keys:
              ``version`` (omit for the full version tree).
            - ``"channel"`` — wraps :func:`get_channel_history`. Payload
              keys: ``channel_id`` (required), ``version``.
        payload: kind-specific arguments (see above).
    """
    payload = payload or {}
    if kind == "causal":
        node_id = payload.get("node_id")
        if not node_id:
            return {"error": "trace(kind='causal') requires payload.node_id"}
        try:
            depth = int(payload.get("depth", 3))
        except (TypeError, ValueError):
            return {"error": (
                f"trace(kind='causal') payload.depth must be an integer "
                f"(got {payload.get('depth')!r})."
            )}
        return trace_causality(
            ctx,
            node_id=node_id,
            project_id=project_id,
            project_name=project_name,
            version=payload.get("version"),
            direction=payload.get("direction", "both"),
            depth=depth,
            include_information_flow=bool(payload.get("include_information_flow", True)),
        )
    if kind == "history":
        return get_history(
            ctx,
            project_id=project_id,
            project_name=project_name,
            version=payload.get("version"),
        )
    if kind == "channel":
        channel_id = payload.get("channel_id")
        if not channel_id:
            return {"error": "trace(kind='channel') requires payload.channel_id"}
        return get_channel_history(
            ctx,
            channel_id=channel_id,
            project_id=project_id,
            project_name=project_name,
            version=payload.get("version"),
        )
    return {"error": (
        f"Unknown trace kind {kind!r}. Expected one of: causal, history, channel."
    )}


@mcp.tool()
@_safe_tool
async def author(
    ctx: Context,
    action: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    payload: Optional[dict] = None,
) -> dict:
    """Mutate world data — ingest text, edit prose, research, or forget.

    Args:
        action: One of:
            - ``"ingest"`` — wraps :func:`ingest`. Payload: ``text``
              (required), ``label``. ``project_name`` becomes the new
              project's name (default ``"MCP Project"``). Async — emits
              progress notifications.
            - ``"edit"`` — wraps :func:`write`. Payload: ``prose``
              (required), ``description``, ``version``.
            - ``"research"`` — wraps :func:`research_topic`. Payload:
              ``topic`` (required), ``provider``, ``max_results``.
            - ``"forget_fact"`` — wraps :func:`delete_world_fact`.
              Payload: ``fact_id`` (required).
    """
    payload = payload or {}
    if action == "ingest":
        text = payload.get("text")
        if not text:
            return {"error": "author(action='ingest') requires payload.text"}
        return await ingest(
            ctx,
            text=text,
            project_name=project_name or "MCP Project",
            label=payload.get("label"),
        )
    if action == "edit":
        prose = payload.get("prose")
        if not prose:
            return {"error": "author(action='edit') requires payload.prose"}
        return write(
            ctx,
            prose=prose,
            project_id=project_id,
            project_name=project_name,
            description=payload.get("description", ""),
            version=payload.get("version"),
        )
    if action == "research":
        topic = payload.get("topic")
        if not topic:
            return {"error": "author(action='research') requires payload.topic"}
        return research_topic(
            ctx,
            topic=topic,
            project_id=project_id,
            project_name=project_name,
            provider=payload.get("provider"),
            max_results=payload.get("max_results"),
        )
    if action == "forget_fact":
        fact_id = payload.get("fact_id")
        if not fact_id:
            return {"error": "author(action='forget_fact') requires payload.fact_id"}
        return delete_world_fact(
            ctx,
            fact_id=fact_id,
            project_id=project_id,
            project_name=project_name,
        )
    return {"error": (
        f"Unknown author action {action!r}. Expected one of: "
        "ingest, edit, research, forget_fact."
    )}


@mcp.tool()
@_safe_tool
def manage(
    ctx: Context,
    action: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    payload: Optional[dict] = None,
) -> dict:
    """Project / version administration — branch, fork, share, configure.

    Args:
        action: One of:
            - ``"branch"`` — :func:`branch`. Payload: ``from_version``.
            - ``"fork"`` — :func:`fork`. Payload: ``new_name``.
            - ``"share"`` — :func:`share`. Payload: ``username``,
              ``role`` (``viewer``/``editor``/``admin``).
            - ``"promote_branch"`` — :func:`promote_branch`. Payload:
              ``version_row_id``, ``description``.
            - ``"update_project"`` — :func:`update_project_tool`.
              Payload: any of ``name``, ``description``, ``is_public``.
            - ``"delete_project"`` — :func:`delete_project`.
            - ``"delete_version"`` — :func:`delete_version`. Payload:
              ``version_row_id``, ``cascade``.
            - ``"reparent_version"`` — :func:`reparent_version`. Payload:
              ``version_row_id``, ``new_ancestor_id``.
            - ``"set_active_version"`` — :func:`set_active_version`.
              Payload: ``version_row_id`` *or* ``version`` (sequential
              project-scoped number). Pass neither to clear the pointer.
            - ``"get_active_version"`` — :func:`get_active_version`.
            - ``"get_settings"`` — :func:`get_project_settings`.
            - ``"set_settings"`` — :func:`set_project_settings`. Payload:
              ``research_topics``.
            - ``"research_status"`` — :func:`get_research_status` (no
              project required).
            - ``"export_prose"`` — :func:`export_prose`. Payload:
              ``branch_path``.
    """
    payload = payload or {}
    p = lambda k, default=None: payload.get(k, default)  # noqa: E731

    def _resolve_pid(min_role: str = "viewer") -> tuple[Optional[int], Optional[dict]]:
        """Resolve project_id from id or name with the given min_role.

        Round-5 audit: several manage sub-actions previously required
        ``project_id`` directly and silently ignored the top-level
        ``project_name`` argument the docstring advertised. Routing
        through ``resolve_project`` honours the documented name path
        and enforces per-action role minima at the helper boundary.
        """
        nonlocal project_id
        if project_id is not None:
            pid, err = resolve_project(project_id, None, ctx, min_role=min_role)
            return pid, ({"error": err} if err else None)
        if project_name:
            pid, err = resolve_project(None, project_name, ctx, min_role=min_role)
            if err:
                return None, {"error": err}
            project_id = pid
            return pid, None
        return None, None

    def _coerce_int(value, *, field: str) -> tuple[Optional[int], Optional[dict]]:
        """Return (int, None) or (None, error_dict) for a payload value."""
        if value is None:
            return None, None
        try:
            return int(value), None
        except (TypeError, ValueError):
            return None, {"error": (
                f"manage(action={action!r}) payload.{field} must be an integer "
                f"(got {value!r})."
            )}

    # Actions with no project resolution required.
    if action == "research_status":
        return get_research_status(ctx)

    if action == "branch":
        from_v, err = _coerce_int(p("from_version"), field="from_version")
        if err:
            return err
        pid, perr = _resolve_pid(min_role="editor")
        if perr:
            return perr
        if from_v is None or pid is None:
            return {"error": "manage(action='branch') requires project_id or project_name and payload.from_version"}
        return branch(ctx, project_id=pid, from_version=from_v)
    if action == "fork":
        new_name = p("new_name")
        pid, perr = _resolve_pid(min_role="editor")
        if perr:
            return perr
        # AUDIT (post-2026-05-26): the granular ``fork`` tool defaults
        # ``new_name`` to ``"<project> (fork)"`` when omitted; manage()
        # now mirrors that contract so both surfaces accept the same
        # payload (previously manage required ``new_name`` and rejected
        # callers who relied on the default).
        if pid is None:
            return {"error": "manage(action='fork') requires project_id or project_name"}
        return fork(ctx, project_id=pid, new_name=new_name)
    if action == "share":
        pid, perr = _resolve_pid(min_role="admin")
        if perr:
            return perr
        if pid is None:
            return {"error": "manage(action='share') requires project_id or project_name"}
        username = p("username")
        if not username:
            return {"error": "manage(action='share') requires payload.username"}
        return share(
            ctx=ctx,
            project_id=pid,
            username=username,
            role=p("role", "viewer"),
        )
    if action == "promote_branch":
        vrid, err = _coerce_int(p("version_row_id"), field="version_row_id")
        if err:
            return err
        if vrid is None:
            return {"error": "manage(action='promote_branch') requires payload.version_row_id"}
        return promote_branch(
            ctx=ctx,
            version_row_id=vrid,
            project_id=project_id,
            project_name=project_name,
            description=p("description", ""),
        )
    if action == "update_project":
        pid, perr = _resolve_pid(min_role="admin")
        if perr:
            return perr
        if pid is None:
            return {"error": "manage(action='update_project') requires project_id or project_name"}
        return update_project_tool(
            ctx=ctx,
            project_id=pid,
            name=p("name"),
            description=p("description"),
            is_public=p("is_public"),
        )
    if action == "delete_project":
        pid, perr = _resolve_pid(min_role="admin")
        if perr:
            return perr
        if pid is None:
            return {"error": "manage(action='delete_project') requires project_id or project_name"}
        return delete_project(ctx, project_id=pid)
    if action == "delete_version":
        vrid, err = _coerce_int(p("version_row_id"), field="version_row_id")
        if err:
            return err
        return delete_version(
            ctx=ctx,
            version_row_id=vrid,
            cascade=bool(p("cascade", False)),
        )
    if action == "reparent_version":
        vrid, err = _coerce_int(p("version_row_id"), field="version_row_id")
        if err:
            return err
        if vrid is None:
            return {"error": "manage(action='reparent_version') requires payload.version_row_id"}
        raw_anc = p("new_ancestor_id")
        if raw_anc is None:
            new_anc: Optional[int] = None
        else:
            new_anc, err = _coerce_int(raw_anc, field="new_ancestor_id")
            if err:
                return err
        return reparent_version(
            ctx=ctx,
            version_row_id=vrid,
            new_ancestor_id=new_anc,
        )
    if action == "set_active_version":
        pid, perr = _resolve_pid(min_role="editor")
        if perr:
            return perr
        if pid is None:
            return {"error": "manage(action='set_active_version') requires project_id or project_name"}
        vrid, err = _coerce_int(p("version_row_id"), field="version_row_id")
        if err:
            return err
        version_num, err = _coerce_int(p("version"), field="version")
        if err:
            return err
        return set_active_version(
            ctx,
            project_id=pid,
            version=version_num,
            version_row_id=vrid,
        )
    if action == "get_active_version":
        pid, perr = _resolve_pid(min_role="viewer")
        if perr:
            return perr
        if pid is None:
            return {"error": "manage(action='get_active_version') requires project_id or project_name"}
        return get_active_version(ctx, project_id=pid)
    if action == "get_settings":
        return get_project_settings(
            ctx, project_id=project_id, project_name=project_name,
        )
    if action == "set_settings":
        topics = p("research_topics")
        if topics is None:
            return {"error": "manage(action='set_settings') requires payload.research_topics"}
        # ``list(topics)`` happily iterates strings character-by-character,
        # silently turning "foreshadowing" into
        # ["f","o","r","e","s","h","a","d","o","w","i","n","g"]. Require
        # an actual sequence (list/tuple) and reject str/bytes outright
        # so the caller gets an explicit error instead of garbage data.
        if isinstance(topics, (str, bytes)):
            return {"error": (
                "manage(action='set_settings'): payload.research_topics "
                "must be a list of strings, not a single string."
            )}
        try:
            topics_list = [str(t) for t in topics]
        except TypeError:
            return {"error": (
                "manage(action='set_settings'): payload.research_topics "
                "must be an iterable of strings."
            )}
        return set_project_settings(
            ctx,
            research_topics=topics_list,
            project_id=project_id,
            project_name=project_name,
        )
    if action == "export_prose":
        return export_prose(
            ctx,
            project_id=project_id,
            project_name=project_name,
            branch_path=p("branch_path"),
        )
    return {"error": (
        f"Unknown manage action {action!r}. See docstring for the action enum."
    )}


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
    ws, _ = load_world_state_projected(project_id)
    if ws is None:
        return json.dumps({"error": "No world model"})
    return ws.model_dump_json(indent=2)


@mcp.resource("world://project/{project_id}/entity/{entity_id}")
def resource_entity(project_id: int, entity_id: str) -> str:
    """Entity detail from a public project (unauthenticated)."""
    proj = get_project(project_id)
    if proj is None or not proj.is_public:
        return json.dumps({"error": "Not found or access denied"})
    ws, _ = load_world_state_projected(project_id)
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
