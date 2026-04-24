"""
Text-to-WorldState Extraction Pipeline.

Three-step LLM extraction using PydanticAI + Ollama:
  Step 1 — Global Ontology Extraction (entities, locations, objects)
  Step 2 — Chunk-by-chunk Topology Extraction (events, edges)
  Step 3 — Assembly + Validation (merge, sort, audit)

All LLM system prompts are loaded from external markdown files
in the ``prompts/`` directory at the project root.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.providers.ollama import OllamaProvider

from shadow_loom.models import (
    CausalEdge,
    Entity,
    EventNode,
    InformationEdge,
    Location,
    NarrativeObject,
    RelationshipEdge,
    SpatialEdge,
    WorldStateV1,
)

logger = logging.getLogger(__name__)

# =====================================================================
# Prompts directory — lives at the project root, not inside the package
# =====================================================================
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Default Ollama base URL for local inference
_OLLAMA_BASE_URL = "http://localhost:11434/v1/"


def _load_prompt(filename: str) -> str:
    """Read a markdown prompt file from the ``prompts/`` directory."""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _resolve_model(model_str: str):
    """
    Resolve a PydanticAI model string. For Ollama models, construct
    the provider with the local base URL so ``OLLAMA_BASE_URL`` doesn't
    need to be set as an environment variable.
    """
    import os
    if model_str.startswith("ollama:"):
        base_url = os.environ.get("OLLAMA_BASE_URL", _OLLAMA_BASE_URL)
        model_name = model_str.split(":", 1)[1]
        from pydantic_ai.models.ollama import OllamaModel
        return OllamaModel(model_name, provider=OllamaProvider(base_url=base_url))
    return model_str


# =====================================================================
# Extraction-Specific Pydantic Models
# =====================================================================


class GlobalRegister(BaseModel):
    """Step 1 output: the static ontology (nouns) of the narrative."""
    locations: Dict[str, Location] = Field(
        description="All unique locations keyed by LOC_ IDs (e.g. LOC_INVERNESS_CASTLE).",
    )
    objects: Dict[str, NarrativeObject] = Field(
        description="All unique narrative objects keyed by OBJ_ IDs (e.g. OBJ_DAGGER).",
    )
    entities: Dict[str, Entity] = Field(
        description="All unique entities (characters, groups) keyed by ENT_ IDs (e.g. ENT_MACBETH).",
    )


class ChunkTopology(BaseModel):
    """Step 2 output: localised events and edges from a single text chunk."""
    events: List[EventNode] = Field(default_factory=list)
    causal_topology: List[CausalEdge] = Field(default_factory=list)
    information_topology: List[InformationEdge] = Field(default_factory=list)
    social_topology: List[RelationshipEdge] = Field(default_factory=list)
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)


class ChunkChronology(BaseModel):
    """Pass B output: events only from a single text chunk."""
    events: List[EventNode] = Field(default_factory=list)


class ChunkEdges(BaseModel):
    """Pass C output: edges only from a single text chunk."""
    causal_topology: List[CausalEdge] = Field(default_factory=list)
    information_topology: List[InformationEdge] = Field(default_factory=list)
    social_topology: List[RelationshipEdge] = Field(default_factory=list)
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    """A single problem found during validation."""
    severity: Literal["error", "warning"] = Field(
        description="'error' = must fix, 'warning' = informational",
    )
    category: str = Field(
        description="e.g. 'hallucinated_id', 'broken_link', 'contradiction', 'duplicate', 'orphan'",
    )
    detail: str = Field(description="Human-readable description of the issue.")


class ValidationReport(BaseModel):
    """Step 3 output: structured audit of an assembled WorldStateV1."""
    is_valid: bool = Field(description="True if no errors were found (warnings are OK).")
    issues: List[ValidationIssue] = Field(
        default_factory=list,
        description="All issues found, both errors and warnings.",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Recommended fixes or improvements.",
    )


class ExtractionConfig(BaseModel):
    """Runtime configuration for the extraction pipeline."""
    model: str = Field(
        default="ollama:qwen3.6:27b",
        description="PydanticAI model string (e.g. 'ollama:qwen3.6:27b', 'openai:gpt-4o').",
    )
    chunk_strategy: Literal["act_headings", "paragraph"] = Field(
        default="act_headings",
        description="How to split the input text into chunks for Step 2.",
    )
    output_retries: int = Field(
        default=3,
        description="Max retries for PydanticAI output validation.",
    )
    fabula_time_spacing: int = Field(
        default=100,
        description="Base spacing between fabula_time values (e.g. 100 → 100, 200, 300). "
        "Gaps allow flashbacks and interstitial events to be inserted later.",
    )
    min_chunk_chars: int = Field(
        default=1500,
        description="Minimum chunk size in characters. Adjacent small paragraphs are "
        "merged until they reach this threshold.",
    )


# =====================================================================
# Text Chunking
# =====================================================================

# Heading patterns that signal a new section (case-insensitive)
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"Act\s+[IVXLCDM\d]+"            # Act I, Act 2, etc.
    r"|Part\s+[IVXLCDM\d]+"          # Part I, Part 2
    r"|Chapter\s+[IVXLCDM\d]+"       # Chapter I, Chapter 2
    r"|Book\s+[IVXLCDM\d]+"          # Book I, Book 2
    r"|Section\s+[IVXLCDM\d]+"       # Section I, Section 2
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)


def chunk_text(text: str, strategy: str = "act_headings", min_chunk_chars: int = 1500) -> List[str]:
    """
    Split narrative text into chunks for Step 2 extraction.

    Parameters
    ----------
    text : str
        Full prose text.
    strategy : str
        ``"act_headings"`` — split on act/chapter/part headings, fall
        back to paragraphs if no headings are found.
        ``"paragraph"`` — split on double newlines.
    min_chunk_chars : int
        Minimum chunk size. Adjacent small paragraphs are merged
        until they reach this threshold (paragraph strategy only).

    Returns
    -------
    list[str]
        Non-empty text chunks in document order.
    """
    if strategy == "act_headings":
        splits = _HEADING_RE.split(text)
        # Re-attach heading lines to their bodies
        matches = list(_HEADING_RE.finditer(text))
        if matches:
            chunks: List[str] = []
            # Text before the first heading (if any)
            preamble = text[: matches[0].start()].strip()
            if preamble:
                chunks.append(preamble)
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
            if chunks:
                return chunks
        # Fallback: no headings found — use paragraph strategy
        logger.info("[Chunking] No act/section headings found — falling back to paragraph split.")

    # Paragraph split
    paragraphs = re.split(r"\n\s*\n", text)
    raw = [p.strip() for p in paragraphs if p.strip()]

    # Merge adjacent small paragraphs up to min_chunk_chars
    if min_chunk_chars > 0 and raw:
        merged: List[str] = [raw[0]]
        for para in raw[1:]:
            if len(merged[-1]) < min_chunk_chars:
                merged[-1] += "\n\n" + para
            else:
                merged.append(para)
        # Final merge: if the last chunk is tiny, append to previous
        if len(merged) >= 2 and len(merged[-1]) < min_chunk_chars // 2:
            merged[-2] += "\n\n" + merged[-1]
            merged.pop()
        return merged
    return raw


# =====================================================================
# Step 1 — Global Ontology Extraction
# =====================================================================

def _build_ontology_agent(config: ExtractionConfig) -> Agent[None, GlobalRegister]:
    """Construct the Step 1 PydanticAI agent."""
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(GlobalRegister),
        system_prompt=_load_prompt("ontology_extraction.md"),
        retries=config.output_retries,
    )


def extract_ontology(text: str, config: ExtractionConfig | None = None) -> GlobalRegister:
    """
    Step 1: Extract the global ontology (locations, objects, entities)
    from the full manuscript text.
    """
    config = config or ExtractionConfig()
    agent = _build_ontology_agent(config)
    logger.info("[Step 1] Extracting global ontology with %s …", config.model)
    result = agent.run_sync(text)
    register = result.output
    logger.info(
        "[Step 1] Ontology extracted — %d locations, %d objects, %d entities.",
        len(register.locations), len(register.objects), len(register.entities),
    )
    return register


# =====================================================================
# Step 2 — Chunk-by-Chunk Extraction (Pass B: Events, Pass C: Edges)
# =====================================================================

class _ChronologyDeps(BaseModel):
    """Dependencies for Pass B (event extraction)."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    previous_event_ids: List[str] = Field(default_factory=list)


class _EdgesDeps(BaseModel):
    """Dependencies for Pass C (edge extraction)."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    chunk_event_ids: List[str] = Field(default_factory=list)
    previous_event_ids: List[str] = Field(default_factory=list)


def _build_chronology_agent(config: ExtractionConfig) -> Agent[_ChronologyDeps, ChunkChronology]:
    """Construct the Pass B agent — event extraction only."""
    agent: Agent[_ChronologyDeps, ChunkChronology] = Agent(
        _resolve_model(config.model),
        deps_type=_ChronologyDeps,
        output_type=NativeOutput(ChunkChronology),
        system_prompt=_load_prompt("chunk_chronology.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_chronology(ctx: RunContext[_ChronologyDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        return (
            "=== VALID ID REGISTER (from Step 1) ===\n"
            f"ENTITY IDs: {entity_ids}\n"
            f"ENTITY NAMES: {entity_names}\n"
            f"LOCATION IDs: {location_ids}\n"
            f"OBJECT IDs: {object_ids}\n"
            f"PREVIOUSLY EXTRACTED EVENT IDs: {ctx.deps.previous_event_ids}\n"
            "\n"
            "You MUST ONLY use ENT_, LOC_, OBJ_ IDs from the lists above.\n"
            "You MAY create new EVT_ IDs for events discovered in this chunk.\n"
            "Do NOT invent new ENT_, LOC_, or OBJ_ IDs."
        )

    return agent


def _build_edges_agent(config: ExtractionConfig) -> Agent[_EdgesDeps, ChunkEdges]:
    """Construct the Pass C agent — edge extraction given known events."""
    agent: Agent[_EdgesDeps, ChunkEdges] = Agent(
        _resolve_model(config.model),
        deps_type=_EdgesDeps,
        output_type=NativeOutput(ChunkEdges),
        system_prompt=_load_prompt("chunk_edges.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_edges(ctx: RunContext[_EdgesDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        all_evt_ids = ctx.deps.previous_event_ids + ctx.deps.chunk_event_ids
        return (
            "=== VALID ID REGISTER ===\n"
            f"ENTITY IDs: {entity_ids}\n"
            f"ENTITY NAMES: {entity_names}\n"
            f"LOCATION IDs: {location_ids}\n"
            f"OBJECT IDs: {object_ids}\n"
            f"THIS CHUNK'S EVENT IDs: {ctx.deps.chunk_event_ids}\n"
            f"PREVIOUS CHUNKS' EVENT IDs: {ctx.deps.previous_event_ids}\n"
            f"ALL VALID EVENT IDs: {all_evt_ids}\n"
            "\n"
            "You MUST ONLY use IDs from the lists above.\n"
            "Do NOT invent new IDs of any kind.\n"
            "Causal edges MAY reference PREVIOUS CHUNKS' EVENT IDs as "
            "source_event_id for cross-chunk causation."
        )

    return agent


def extract_topology(
    chunks: List[str],
    register: GlobalRegister,
    config: ExtractionConfig | None = None,
) -> List[ChunkTopology]:
    """
    Step 2: Two-pass extraction for each chunk.

    **Pass B** extracts events (chronology) with a focused, simple schema.
    **Pass C** extracts edges (topology) given the concrete events from Pass B.

    This split reduces schema complexity per LLM call and lets the edge
    pass reference real event IDs rather than hallucinating them.
    """
    config = config or ExtractionConfig()
    chrono_agent = _build_chronology_agent(config)
    edges_agent = _build_edges_agent(config)
    topologies: List[ChunkTopology] = []
    syuzhet_counter = 0
    fabula_time_base = config.fabula_time_spacing
    all_event_ids: List[str] = []

    for i, chunk in enumerate(chunks):
        logger.info("[Step 2B] Processing chunk %d/%d (%d chars) — events …", i + 1, len(chunks), len(chunk))

        # --- Pass B: Event extraction ---
        chrono_msg = (
            f"Chunk {i + 1} of {len(chunks)} "
            f"(syuzhet_index offset: {syuzhet_counter}, "
            f"fabula_time_base: {fabula_time_base}):\n\n{chunk}"
        )
        chrono_deps = _ChronologyDeps(
            global_register=register,
            previous_event_ids=all_event_ids.copy(),
        )
        try:
            chrono_result = chrono_agent.run_sync(chrono_msg, deps=chrono_deps)
            chrono = chrono_result.output
        except Exception:
            logger.exception("[Step 2B] Chunk %d FAILED — returning empty events.", i + 1)
            chrono = ChunkChronology()

        logger.info("[Step 2B] Chunk %d: %d events extracted.", i + 1, len(chrono.events))

        # Build event summary for Pass C
        chunk_evt_ids = [e.id for e in chrono.events]
        event_summary_lines = []
        for e in chrono.events:
            event_summary_lines.append(
                f"  - {e.id} (fabula={e.fabula_time}, syuzhet={e.syuzhet_index}, "
                f"type={e.event_type}, actor={e.actor_id}, target={e.target_id}): "
                f"{e.description}"
            )
        event_summary = "\n".join(event_summary_lines)

        # --- Pass C: Edge extraction ---
        edges = ChunkEdges()
        if chrono.events:
            logger.info("[Step 2C] Processing chunk %d/%d — edges …", i + 1, len(chunks))
            edges_msg = (
                f"Chunk {i + 1} of {len(chunks)}.\n\n"
                f"EVENTS EXTRACTED FROM THIS CHUNK:\n{event_summary}\n\n"
                f"ORIGINAL TEXT:\n{chunk}"
            )
            edges_deps = _EdgesDeps(
                global_register=register,
                chunk_event_ids=chunk_evt_ids,
                previous_event_ids=all_event_ids.copy(),
            )
            try:
                edges_result = edges_agent.run_sync(edges_msg, deps=edges_deps)
                edges = edges_result.output
            except Exception:
                logger.exception("[Step 2C] Chunk %d FAILED — returning empty edges.", i + 1)
        else:
            logger.info("[Step 2C] Chunk %d: skipping edge pass (no events).", i + 1)

        # Merge into ChunkTopology
        topo = ChunkTopology(
            events=chrono.events,
            causal_topology=edges.causal_topology,
            information_topology=edges.information_topology,
            social_topology=edges.social_topology,
            spatial_topology=edges.spatial_topology,
        )
        topologies.append(topo)

        # Accumulate counters
        syuzhet_counter += len(chrono.events)
        if chrono.events:
            max_fabula = max(e.fabula_time for e in chrono.events)
            fabula_time_base = (
                (max_fabula // config.fabula_time_spacing + 1)
                * config.fabula_time_spacing
            )
        all_event_ids.extend(chunk_evt_ids)

        logger.info(
            "[Step 2] Chunk %d: %d events, %d causal, %d social, %d spatial, %d info edges.",
            i + 1, len(topo.events), len(topo.causal_topology),
            len(topo.social_topology), len(topo.spatial_topology),
            len(topo.information_topology),
        )

    return topologies


# =====================================================================
# Step 3 — Assembly + Validation
# =====================================================================

def _deduplicate_social(edges: List[RelationshipEdge]) -> List[RelationshipEdge]:
    """Keep only the most recent edge per (source, target) pair."""
    best: dict[tuple[str, str], RelationshipEdge] = {}
    for e in edges:
        key = (e.source_entity_id, e.target_entity_id)
        if key not in best or e.last_updated_fabula > best[key].last_updated_fabula:
            best[key] = e
    return list(best.values())


def _deduplicate_spatial(edges: List[SpatialEdge]) -> List[SpatialEdge]:
    """Keep one edge per (source, target) pair, preferring the latest."""
    best: dict[tuple[str, str], SpatialEdge] = {}
    for e in edges:
        key = (e.source_id, e.target_id)
        if key not in best or e.established_at_fabula > best[key].established_at_fabula:
            best[key] = e
    return list(best.values())


def assemble_world_state(
    register: GlobalRegister,
    topologies: List[ChunkTopology],
) -> WorldStateV1:
    """
    Merge the Step 1 register and Step 2 chunk topologies into a
    single ``WorldStateV1``.

    Social and spatial edges are deduplicated per (source, target) pair,
    keeping the most recently updated edge.
    """
    events: List[EventNode] = []
    causal_topology: List[CausalEdge] = []
    information_topology: List[InformationEdge] = []
    social_topology: List[RelationshipEdge] = []
    spatial_topology: List[SpatialEdge] = []

    for topo in topologies:
        events.extend(topo.events)
        causal_topology.extend(topo.causal_topology)
        information_topology.extend(topo.information_topology)
        social_topology.extend(topo.social_topology)
        spatial_topology.extend(topo.spatial_topology)

    # Sort events chronologically
    events.sort(key=lambda e: e.fabula_time)
    causal_topology.sort(key=lambda c: c.fabula_time)

    # Deduplicate relationship and spatial edges across chunks
    social_before = len(social_topology)
    social_topology = _deduplicate_social(social_topology)
    spatial_before = len(spatial_topology)
    spatial_topology = _deduplicate_spatial(spatial_topology)
    if social_before != len(social_topology) or spatial_before != len(spatial_topology):
        logger.info(
            "[Step 3] Deduplicated edges: social %d→%d, spatial %d→%d.",
            social_before, len(social_topology),
            spatial_before, len(spatial_topology),
        )

    ws = WorldStateV1(
        locations=register.locations,
        objects=register.objects,
        entities=register.entities,
        events=events,
        causal_topology=causal_topology,
        spatial_topology=spatial_topology,
        information_topology=information_topology,
        social_topology=social_topology,
    )
    logger.info(
        "[Step 3] Assembled WorldStateV1 — %d events, %d causal, %d social, "
        "%d spatial, %d info edges.",
        len(ws.events), len(ws.causal_topology), len(ws.social_topology),
        len(ws.spatial_topology), len(ws.information_topology),
    )
    return ws


def _programmatic_validation(ws: WorldStateV1) -> List[ValidationIssue]:
    """Fast structural checks that don't require an LLM."""
    issues: List[ValidationIssue] = []

    # Build the valid ID set
    valid_ids = (
        set(ws.locations.keys())
        | set(ws.objects.keys())
        | set(ws.entities.keys())
        | {e.id for e in ws.events}
    )

    # Check causal edges
    for ce in ws.causal_topology:
        if ce.source_event_id not in valid_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"CausalEdge.source_event_id '{ce.source_event_id}' not in node set.",
            ))
        if ce.target_node_id not in valid_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"CausalEdge.target_node_id '{ce.target_node_id}' not in node set.",
            ))

    # Check relationship edges
    entity_ids = set(ws.entities.keys())
    for re_edge in ws.social_topology:
        if re_edge.source_entity_id not in entity_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"RelationshipEdge.source_entity_id '{re_edge.source_entity_id}' not in entities.",
            ))
        if re_edge.target_entity_id not in entity_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"RelationshipEdge.target_entity_id '{re_edge.target_entity_id}' not in entities.",
            ))

    # Check spatial edges
    location_ids = set(ws.locations.keys())
    for se in ws.spatial_topology:
        if se.source_id not in location_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"SpatialEdge.source_id '{se.source_id}' not in locations.",
            ))
        if se.target_id not in location_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"SpatialEdge.target_id '{se.target_id}' not in locations.",
            ))

    # Check information edges
    node_ids = set(ws.entities.keys()) | set(ws.objects.keys())
    for ie in ws.information_topology:
        if ie.source_id not in node_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"InformationEdge.source_id '{ie.source_id}' not in entities/objects.",
            ))
        for tid in ie.target_ids:
            if tid not in node_ids:
                issues.append(ValidationIssue(
                    severity="error", category="broken_link",
                    detail=f"InformationEdge.target_id '{tid}' not in entities/objects.",
                ))

    # Check event actor_id and target_id references
    all_node_ids = valid_ids
    for evt in ws.events:
        if evt.actor_id and evt.actor_id not in all_node_ids:
            issues.append(ValidationIssue(
                severity="error", category="hallucinated_id",
                detail=f"EventNode '{evt.id}' actor_id '{evt.actor_id}' not in node set.",
            ))
        if evt.target_id and evt.target_id not in all_node_ids:
            issues.append(ValidationIssue(
                severity="error", category="hallucinated_id",
                detail=f"EventNode '{evt.id}' target_id '{evt.target_id}' not in node set.",
            ))

    # Check entity location_id references
    for eid, ent in ws.entities.items():
        if ent.location_id not in location_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"Entity '{eid}' location_id '{ent.location_id}' not in locations.",
            ))

    # Check object location_id and owner_id references
    for oid, obj in ws.objects.items():
        if obj.location_id and obj.location_id not in location_ids:
            issues.append(ValidationIssue(
                severity="warning", category="broken_link",
                detail=f"Object '{oid}' location_id '{obj.location_id}' not in locations.",
            ))
        if obj.owner_id and obj.owner_id not in entity_ids:
            issues.append(ValidationIssue(
                severity="warning", category="broken_link",
                detail=f"Object '{oid}' owner_id '{obj.owner_id}' not in entities.",
            ))

    # Check for duplicate event IDs
    seen_evt_ids: set[str] = set()
    for evt in ws.events:
        if evt.id in seen_evt_ids:
            issues.append(ValidationIssue(
                severity="error", category="duplicate",
                detail=f"Duplicate event ID: '{evt.id}'.",
            ))
        seen_evt_ids.add(evt.id)

    # --- Time validation ---
    issues.extend(_validate_time_ordering(ws))

    # --- Dead-actor validation ---
    issues.extend(_validate_dead_actors(ws))

    return issues


def _validate_dead_actors(ws: WorldStateV1) -> List[ValidationIssue]:
    """Check that entities marked dead do not act after their death event."""
    issues: List[ValidationIssue] = []

    # Find entities with status == "dead"
    dead_entities = {eid for eid, ent in ws.entities.items() if ent.status == "dead"}
    if not dead_entities:
        return issues

    # For each dead entity, find the earliest outcome event targeting them
    # as the probable death event
    death_times: dict[str, int] = {}
    death_events: dict[str, str] = {}
    for evt in ws.events:
        if (
            evt.target_id in dead_entities
            and evt.event_type == "outcome"
        ):
            if evt.target_id not in death_times or evt.fabula_time < death_times[evt.target_id]:
                death_times[evt.target_id] = evt.fabula_time
                death_events[evt.target_id] = evt.id

    # Check for actor references after death
    for evt in ws.events:
        if evt.actor_id and evt.actor_id in death_times:
            death_t = death_times[evt.actor_id]
            if evt.fabula_time > death_t:
                issues.append(ValidationIssue(
                    severity="error",
                    category="contradiction",
                    detail=(
                        f"Dead entity '{evt.actor_id}' acts in '{evt.id}' "
                        f"(fabula={evt.fabula_time}) after death in "
                        f"'{death_events[evt.actor_id]}' (fabula={death_t})."
                    ),
                ))

    return issues


def _validate_time_ordering(ws: WorldStateV1) -> List[ValidationIssue]:
    """Validate fabula_time and syuzhet_index correctness."""
    issues: List[ValidationIssue] = []

    if not ws.events:
        return issues

    # 1. Check syuzhet_index is contiguous (no gaps, no duplicates)
    syuzhet_indices = sorted(e.syuzhet_index for e in ws.events)
    syuzhet_set = set(syuzhet_indices)
    if len(syuzhet_set) != len(ws.events):
        dupes = [s for s in syuzhet_set if syuzhet_indices.count(s) > 1]
        issues.append(ValidationIssue(
            severity="error", category="temporal",
            detail=f"Duplicate syuzhet_index values: {dupes}.",
        ))
    if syuzhet_indices:
        expected = list(range(syuzhet_indices[0], syuzhet_indices[0] + len(syuzhet_indices)))
        if syuzhet_indices != expected:
            gaps = set(expected) - syuzhet_set
            if gaps:
                issues.append(ValidationIssue(
                    severity="warning", category="temporal",
                    detail=f"Non-contiguous syuzhet_index — gaps at: {sorted(gaps)[:10]}.",
                ))

    # 2. Check fabula_time has reasonable spacing (warn if contiguous 1,2,3…)
    fabula_times = sorted(set(e.fabula_time for e in ws.events))
    if len(fabula_times) >= 3:
        diffs = [fabula_times[i + 1] - fabula_times[i] for i in range(len(fabula_times) - 1)]
        median_diff = sorted(diffs)[len(diffs) // 2]
        if median_diff <= 1:
            issues.append(ValidationIssue(
                severity="warning", category="temporal",
                detail=(
                    f"fabula_time values use contiguous small integers "
                    f"(median gap={median_diff}). Expected spacing ~100 to leave "
                    f"room for flashbacks and interstitial events."
                ),
            ))

    # 3. Check causal edges: cause must precede or coincide with effect in fabula_time
    evt_fabula = {e.id: e.fabula_time for e in ws.events}
    for ce in ws.causal_topology:
        src_t = evt_fabula.get(ce.source_event_id)
        # target can be an event or another node type
        tgt_t = evt_fabula.get(ce.target_node_id)
        if src_t is not None and tgt_t is not None and src_t > tgt_t:
            issues.append(ValidationIssue(
                severity="error", category="temporal",
                detail=(
                    f"Causal edge '{ce.source_event_id}' (fabula={src_t}) → "
                    f"'{ce.target_node_id}' (fabula={tgt_t}): cause after effect."
                ),
            ))

    # 4. Check information edges: terminated cannot precede established
    for ie in ws.information_topology:
        if ie.terminated_at_fabula is not None and ie.terminated_at_fabula < ie.established_at_fabula:
            issues.append(ValidationIssue(
                severity="error", category="temporal",
                detail=(
                    f"InformationEdge '{ie.source_id}'→{ie.target_ids}: "
                    f"terminated_at_fabula ({ie.terminated_at_fabula}) < "
                    f"established_at_fabula ({ie.established_at_fabula})."
                ),
            ))

    return issues


def _build_validation_agent(config: ExtractionConfig) -> Agent[None, ValidationReport]:
    """Construct the Step 3 LLM validation agent."""
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(ValidationReport),
        system_prompt=_load_prompt("validation.md"),
        retries=config.output_retries,
    )


def validate_world_state(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> ValidationReport:
    """
    Step 3: Run programmatic and LLM-based validation on the
    assembled world state.
    """
    config = config or ExtractionConfig()

    # Phase A: fast programmatic checks
    prog_issues = _programmatic_validation(ws)
    prog_errors = [i for i in prog_issues if i.severity == "error"]
    logger.info(
        "[Step 3·Programmatic] %d issues (%d errors, %d warnings).",
        len(prog_issues), len(prog_errors), len(prog_issues) - len(prog_errors),
    )

    # Phase B: LLM audit for semantic contradictions
    agent = _build_validation_agent(config)
    ws_json = ws.model_dump_json(indent=2)
    # Truncate if extremely long to fit context window
    max_chars = 80_000
    if len(ws_json) > max_chars:
        ws_json = ws_json[:max_chars] + "\n... [TRUNCATED]"

    logger.info("[Step 3·LLM] Running validation agent (%d chars) …", len(ws_json))
    result = agent.run_sync(
        f"Validate the following WorldStateV1 JSON:\n\n{ws_json}"
    )
    llm_report = result.output

    # Merge programmatic + LLM issues
    all_issues = prog_issues + llm_report.issues
    has_errors = any(i.severity == "error" for i in all_issues)

    merged = ValidationReport(
        is_valid=not has_errors,
        issues=all_issues,
        suggestions=llm_report.suggestions,
    )
    logger.info(
        "[Step 3] Validation complete — is_valid=%s, %d total issues.",
        merged.is_valid, len(merged.issues),
    )
    return merged


# =====================================================================
# Top-Level Orchestrator
# =====================================================================

def run_extraction(
    text: str,
    config: ExtractionConfig | None = None,
) -> Tuple[WorldStateV1, ValidationReport]:
    """
    Run the full 3-step extraction pipeline.

    Parameters
    ----------
    text : str
        Full narrative prose text.
    config : ExtractionConfig or None
        Pipeline configuration. Uses defaults if None.

    Returns
    -------
    (WorldStateV1, ValidationReport)
        The assembled world state and its validation report.
    """
    config = config or ExtractionConfig()
    logger.info("[Pipeline] Starting extraction with model=%s, strategy=%s", config.model, config.chunk_strategy)

    # Step 1: Global Ontology
    register = extract_ontology(text, config)

    # Step 2: Chunk Topology
    chunks = chunk_text(text, strategy=config.chunk_strategy, min_chunk_chars=config.min_chunk_chars)
    logger.info("[Pipeline] Text split into %d chunks.", len(chunks))
    topologies = extract_topology(chunks, register, config)

    # Step 3: Assembly + Validation
    world_state = assemble_world_state(register, topologies)
    report = validate_world_state(world_state, config)

    return world_state, report
