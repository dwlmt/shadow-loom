"""
Text-to-WorldState Register-Hybrid Extraction Pipeline.

Five-step LLM extraction using PydanticAI + Ollama:
  Step 1 — Global Coreference Pre-Pass (three separate passes → GlobalRegister)
    Step 1a — Location extraction (LOC_ nodes)
    Step 1b — Object extraction (OBJ_ nodes, with location context)
    Step 1c — Entity extraction (ENT_ nodes, with location + object context)
  Step 2 — Semantic Scaffolding (Socratic QA per chunk)
  Step 3 — Decomposed Topology Extraction (Physics Agent + Social Agent per chunk)
  Step 4 — Pydantic Propose-Critique-Repair (per-chunk result validation)
  Step 5 — Global Assembly + Mathematical Sorting + Validation + Correction

All LLM system prompts are loaded from external markdown files
in the ``prompts/`` directory inside the package.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.providers.ollama import OllamaProvider

from shadow_loom.models import (
    Belief,
    CausalEdge,
    Entity,
    EntityStateSnapshot,
    EventNode,
    InformationEdge,
    Location,
    NarrativeObject,
    RelationshipEdge,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
)

logger = logging.getLogger(__name__)

# =====================================================================
# Prompts directory — lives inside the package
# =====================================================================
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

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


class LocationRegister(BaseModel):
    """Step 1a output: all unique locations extracted from the narrative."""
    locations: Dict[str, Location] = Field(
        description="All unique locations keyed by LOC_ IDs (e.g. LOC_INVERNESS_CASTLE).",
    )


class ObjectRegister(BaseModel):
    """Step 1b output: all unique narrative objects extracted from the narrative."""
    objects: Dict[str, NarrativeObject] = Field(
        description="All unique narrative objects keyed by OBJ_ IDs (e.g. OBJ_DAGGER).",
    )


class EntityRegister(BaseModel):
    """Step 1c output: all unique entities (characters, groups) extracted from the narrative."""
    entities: Dict[str, Entity] = Field(
        description="All unique entities (characters, groups) keyed by ENT_ IDs (e.g. ENT_MACBETH).",
    )


class ChunkTopology(BaseModel):
    """Merged output per chunk: events + all edge types. Used by assembly."""
    events: List[EventNode] = Field(default_factory=list)
    causal_topology: List[CausalEdge] = Field(default_factory=list)
    information_topology: List[InformationEdge] = Field(default_factory=list)
    social_topology: List[RelationshipEdge] = Field(default_factory=list)
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)
    entity_updates: List["EntityUpdate"] = Field(default_factory=list)


class QAPair(BaseModel):
    """A single Socratic question-answer pair from semantic scaffolding."""
    category: Literal["who", "what", "where", "when", "why", "how"] = Field(
        description="The interrogative category of this QA pair.",
    )
    question: str = Field(description="The question about this chunk of text.")
    answer: str = Field(description="The answer, articulating implicit reasoning and hidden variables.")


class SocraticScaffold(BaseModel):
    """Step 2 output: semantic scaffolding QA pairs for a single chunk."""
    qa_pairs: List[QAPair] = Field(
        default_factory=list,
        description="Who/What/Where/When/Why/How pairs articulating the "
        "narrative logic of this chunk before structured extraction.",
    )


class PhysicsExtraction(BaseModel):
    """Step 3a output: events + causal/spatial edges from the Physics Agent."""
    events: List[EventNode] = Field(default_factory=list)
    causal_topology: List[CausalEdge] = Field(default_factory=list)
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)
    entity_updates: List["EntityUpdate"] = Field(default_factory=list)


class SocialExtraction(BaseModel):
    """Step 3b output: information + relationship edges from the Social Agent."""
    information_topology: List[InformationEdge] = Field(default_factory=list)
    social_topology: List[RelationshipEdge] = Field(default_factory=list)


class EntityUpdate(BaseModel):
    """Per-chunk delta: how an entity's state changed during this chunk."""
    entity_id: str = Field(description="ENT_ ID of the entity that changed.")
    fabula_time: int = Field(description="fabula_time when this change occurred.")
    triggered_by: Optional[str] = Field(default=None, description="EVT_ ID that caused this change.")
    trait_updates: Dict[str, TraitVector] = Field(
        default_factory=dict,
        description="Updated trait values. Only include traits that changed.",
    )
    new_beliefs: List[Belief] = Field(default_factory=list, description="New beliefs formed.")
    invalidated_belief_targets: List[str] = Field(
        default_factory=list,
        description="target_ids of beliefs shattered by this event.",
    )
    new_status: Optional[Literal["healthy", "injured", "ill", "dead", "unconscious"]] = Field(
        default=None, description="New status if changed.",
    )
    new_location_id: Optional[str] = Field(default=None, description="New location if entity moved.")


# Resolve forward references now that EntityUpdate is defined
ChunkTopology.model_rebuild()
PhysicsExtraction.model_rebuild()


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
        default=5,
        description="Max retries for PydanticAI output validation.",
    )
    fabula_time_spacing: int = Field(
        default=1000,
        description="Base spacing between fabula_time values (e.g. 1000 → 1000, 2000, 3000). "
        "Gaps allow flashbacks and interstitial events to be inserted later.",
    )
    min_chunk_chars: int = Field(
        default=1500,
        description="Minimum chunk size in characters. Adjacent small paragraphs are "
        "merged until they reach this threshold.",
    )
    chunk_overlap_chars: int = Field(
        default=300,
        description="Number of trailing characters from the previous chunk to prepend "
        "as context for the next chunk. Helps maintain coreference across "
        "chunk boundaries.",
    )
    max_correction_retries: int = Field(
        default=3,
        description="Maximum correction passes after validation. Each pass feeds "
        "programmatic errors back to the LLM for targeted repair.",
    )
    max_concurrent_chunks: int = Field(
        default=4,
        description="Maximum number of chunks to extract in parallel during "
        "async topology extraction. Controls LLM request concurrency.",
    )
    estimated_events_per_chunk: int = Field(
        default=10,
        description="Estimated events per chunk — used to pre-allocate syuzhet "
        "and fabula_time ranges for parallel extraction.",
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
# Step 1 — Global Ontology Extraction (Three Separate Passes)
# =====================================================================

# --- Step 1a: Location extraction (no dependencies) ---

def _build_location_agent(config: ExtractionConfig) -> Agent[None, LocationRegister]:
    """Construct the Step 1a Location Agent."""
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(LocationRegister),
        system_prompt=_load_prompt("ontology_locations.md"),
        retries=config.output_retries,
    )


# --- Step 1b: Object extraction (depends on locations) ---

class _ObjectDeps(BaseModel):
    """Dependencies for Step 1b — objects need location IDs for location_id."""
    model_config = {"protected_namespaces": ()}
    location_register: LocationRegister


def _build_object_agent(config: ExtractionConfig) -> Agent[_ObjectDeps, ObjectRegister]:
    """Construct the Step 1b Object Agent."""
    agent: Agent[_ObjectDeps, ObjectRegister] = Agent(
        _resolve_model(config.model),
        deps_type=_ObjectDeps,
        output_type=NativeOutput(ObjectRegister),
        system_prompt=_load_prompt("ontology_objects.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_locations_for_objects(ctx: RunContext[_ObjectDeps]) -> str:
        loc_ids = sorted(ctx.deps.location_register.locations.keys())
        loc_names = {
            lid: ctx.deps.location_register.locations[lid].name
            for lid in loc_ids
        }
        return (
            "=== LOCATION REGISTER (from Step 1a) ===\n"
            f"LOCATION IDs: {loc_ids}\n"
            f"LOCATION NAMES: {loc_names}\n"
            "\n"
            "Use ONLY these LOC_ IDs when assigning location_id to objects.\n"
            "Set location_id to null if the object is held by someone."
        )

    return agent


# --- Step 1c: Entity extraction (depends on locations + objects) ---

class _EntityDeps(BaseModel):
    """Dependencies for Step 1c — entities need location + object IDs."""
    model_config = {"protected_namespaces": ()}
    location_register: LocationRegister
    object_register: ObjectRegister


def _build_entity_agent(config: ExtractionConfig) -> Agent[_EntityDeps, EntityRegister]:
    """Construct the Step 1c Entity Agent."""
    agent: Agent[_EntityDeps, EntityRegister] = Agent(
        _resolve_model(config.model),
        deps_type=_EntityDeps,
        output_type=NativeOutput(EntityRegister),
        system_prompt=_load_prompt("ontology_entities.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_registers_for_entities(ctx: RunContext[_EntityDeps]) -> str:
        loc_ids = sorted(ctx.deps.location_register.locations.keys())
        loc_names = {
            lid: ctx.deps.location_register.locations[lid].name
            for lid in loc_ids
        }
        obj_ids = sorted(ctx.deps.object_register.objects.keys())
        obj_names = {
            oid: ctx.deps.object_register.objects[oid].name
            for oid in obj_ids
        }
        return (
            "=== LOCATION REGISTER (from Step 1a) ===\n"
            f"LOCATION IDs: {loc_ids}\n"
            f"LOCATION NAMES: {loc_names}\n"
            "\n"
            "=== OBJECT REGISTER (from Step 1b) ===\n"
            f"OBJECT IDs: {obj_ids}\n"
            f"OBJECT NAMES: {obj_names}\n"
            "\n"
            "Use ONLY these LOC_ IDs when assigning location_id.\n"
            "You may reference LOC_ and OBJ_ IDs in belief target_id fields.\n"
            "You may also reference ENT_ IDs you are creating in this pass."
        )

    return agent


# --- Legacy single-pass agent (kept for backward compatibility) ---

def _build_ontology_agent(config: ExtractionConfig) -> Agent[None, GlobalRegister]:
    """Construct the legacy single-pass Step 1 PydanticAI agent."""
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(GlobalRegister),
        system_prompt=_load_prompt("ontology_extraction.md"),
        retries=config.output_retries,
    )


def _resolve_object_owner_ids(
    objects: Dict[str, NarrativeObject],
    entities: Dict[str, Entity],
) -> Dict[str, NarrativeObject]:
    """
    Resolve owner_id strings from Step 1b (which may be names/descriptions)
    to canonical ENT_ IDs now that entities are available from Step 1c.
    """
    entity_name_map: Dict[str, str] = {}
    for eid, ent in entities.items():
        entity_name_map[ent.name.lower()] = eid
        entity_name_map[eid.lower()] = eid
        # Also index by individual name parts and common abbreviations
        # e.g. "King Duncan" → match "duncan", "king duncan"
        for part in ent.name.lower().split():
            if len(part) > 2 and part not in entity_name_map:
                entity_name_map[part] = eid
        # Strip parenthetical suffixes: "Macduff (Thane of Fife)" → "macduff"
        base = ent.name.split("(")[0].strip().lower()
        if base and base not in entity_name_map:
            entity_name_map[base] = eid
        # Map ID-like forms: "ENT_THREE_WITCHES" → also match "three witches"
        readable = eid.replace("ENT_", "").replace("_", " ").lower()
        if readable not in entity_name_map:
            entity_name_map[readable] = eid

    resolved: Dict[str, NarrativeObject] = {}
    for oid, obj in objects.items():
        if obj.owner_id and not obj.owner_id.startswith("ENT_"):
            owner_lower = obj.owner_id.lower()
            # Try exact match first
            matched = entity_name_map.get(owner_lower)
            # Try substring match if exact fails
            if not matched:
                for name_key, eid in entity_name_map.items():
                    if name_key in owner_lower or owner_lower in name_key:
                        matched = eid
                        break
            if matched:
                resolved[oid] = obj.model_copy(update={"owner_id": matched})
                logger.debug("[Step 1·Resolve] Object %s owner_id '%s' → '%s'", oid, obj.owner_id, matched)
            else:
                logger.warning("[Step 1·Resolve] Object %s owner_id '%s' could not be resolved to an entity.", oid, obj.owner_id)
                resolved[oid] = obj
        else:
            resolved[oid] = obj
    return resolved


def extract_ontology(text: str, config: ExtractionConfig | None = None) -> GlobalRegister:
    """
    Step 1: Extract the global ontology (locations, objects, entities)
    from the full manuscript text via three separate passes.

    Pass order:
      1a. Locations — no dependencies, extracts all LOC_ nodes.
      1b. Objects — receives location register, extracts all OBJ_ nodes.
      1c. Entities — receives location + object registers, extracts all ENT_ nodes.

    The three passes are then merged into a single GlobalRegister.
    """
    config = config or ExtractionConfig()

    # --- Step 1a: Locations ---
    location_agent = _build_location_agent(config)
    logger.info("[Step 1a] Extracting locations with %s …", config.model)
    try:
        loc_result = location_agent.run_sync(text)
        loc_register = loc_result.output
    except Exception:
        logger.exception("[Step 1a] Location extraction failed — retrying once …")
        loc_result = location_agent.run_sync(text)
        loc_register = loc_result.output
    logger.info("[Step 1a] Extracted %d locations.", len(loc_register.locations))

    # --- Step 1b: Objects (with location context) ---
    object_agent = _build_object_agent(config)
    obj_deps = _ObjectDeps(location_register=loc_register)
    logger.info("[Step 1b] Extracting objects with %s …", config.model)
    try:
        obj_result = object_agent.run_sync(text, deps=obj_deps)
        obj_register = obj_result.output
    except Exception:
        logger.exception("[Step 1b] Object extraction failed — retrying once …")
        obj_result = object_agent.run_sync(text, deps=obj_deps)
        obj_register = obj_result.output
    logger.info("[Step 1b] Extracted %d objects.", len(obj_register.objects))

    # --- Step 1c: Entities (with location + object context) ---
    entity_agent = _build_entity_agent(config)
    ent_deps = _EntityDeps(location_register=loc_register, object_register=obj_register)
    logger.info("[Step 1c] Extracting entities with %s …", config.model)
    try:
        ent_result = entity_agent.run_sync(text, deps=ent_deps)
        ent_register = ent_result.output
    except Exception:
        logger.exception("[Step 1c] Entity extraction failed — retrying once …")
        ent_result = entity_agent.run_sync(text, deps=ent_deps)
        ent_register = ent_result.output
    logger.info("[Step 1c] Extracted %d entities.", len(ent_register.entities))

    # --- Resolve object owner_ids to ENT_ IDs ---
    resolved_objects = _resolve_object_owner_ids(obj_register.objects, ent_register.entities)

    # --- Merge into GlobalRegister ---
    register = GlobalRegister(
        locations=loc_register.locations,
        objects=resolved_objects,
        entities=ent_register.entities,
    )
    logger.info(
        "[Step 1] Ontology extracted — %d locations, %d objects, %d entities.",
        len(register.locations), len(register.objects), len(register.entities),
    )
    return register


async def extract_ontology_async(
    text: str,
    config: ExtractionConfig | None = None,
) -> GlobalRegister:
    """Async variant of :func:`extract_ontology`.

    Runs Step 1a (locations) first, then Steps 1b (objects) and
    1c (entities) in parallel via ``asyncio.gather``.  Entity
    extraction does not structurally depend on object IDs — objects
    reference entities via ``owner_id``, not the reverse — so
    running them concurrently is safe.  ``_resolve_object_owner_ids``
    is called after both complete.
    """
    config = config or ExtractionConfig()

    # --- Step 1a: Locations (must complete first — both 1b and 1c need it) ---
    location_agent = _build_location_agent(config)
    logger.info("[Step 1a] Extracting locations with %s …", config.model)
    try:
        loc_result = await location_agent.run(text)
        loc_register = loc_result.output
    except Exception:
        logger.exception("[Step 1a] Location extraction failed — retrying once …")
        loc_result = await location_agent.run(text)
        loc_register = loc_result.output
    logger.info("[Step 1a] Extracted %d locations.", len(loc_register.locations))

    # --- Steps 1b + 1c in parallel ---
    async def _extract_objects() -> ObjectRegister:
        object_agent = _build_object_agent(config)
        obj_deps = _ObjectDeps(location_register=loc_register)
        logger.info("[Step 1b] Extracting objects with %s …", config.model)
        try:
            obj_result = await object_agent.run(text, deps=obj_deps)
            return obj_result.output
        except Exception:
            logger.exception("[Step 1b] Object extraction failed — retrying once …")
            obj_result = await object_agent.run(text, deps=obj_deps)
            return obj_result.output

    async def _extract_entities() -> EntityRegister:
        entity_agent = _build_entity_agent(config)
        # Entity extraction does NOT structurally need object IDs.
        # We pass an empty ObjectRegister so the agent still gets location context.
        ent_deps = _EntityDeps(
            location_register=loc_register,
            object_register=ObjectRegister(objects={}),
        )
        logger.info("[Step 1c] Extracting entities with %s …", config.model)
        try:
            ent_result = await entity_agent.run(text, deps=ent_deps)
            return ent_result.output
        except Exception:
            logger.exception("[Step 1c] Entity extraction failed — retrying once …")
            ent_result = await entity_agent.run(text, deps=ent_deps)
            return ent_result.output

    obj_register, ent_register = await asyncio.gather(
        _extract_objects(), _extract_entities()
    )
    logger.info("[Step 1b] Extracted %d objects.", len(obj_register.objects))
    logger.info("[Step 1c] Extracted %d entities.", len(ent_register.entities))

    # --- Resolve object owner_ids to ENT_ IDs (needs both registers) ---
    resolved_objects = _resolve_object_owner_ids(obj_register.objects, ent_register.entities)

    register = GlobalRegister(
        locations=loc_register.locations,
        objects=resolved_objects,
        entities=ent_register.entities,
    )
    logger.info(
        "[Step 1] Ontology extracted — %d locations, %d objects, %d entities.",
        len(register.locations), len(register.objects), len(register.entities),
    )
    return register


# =====================================================================
# Step 2 — Semantic Scaffolding (Socratic QA)
# =====================================================================

class _SocraticDeps(BaseModel):
    """Dependencies for Step 2 (Socratic QA scaffolding)."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister


def _build_socratic_agent(config: ExtractionConfig) -> Agent[_SocraticDeps, SocraticScaffold]:
    """Construct the Step 2 Socratic QA agent."""
    agent: Agent[_SocraticDeps, SocraticScaffold] = Agent(
        _resolve_model(config.model),
        deps_type=_SocraticDeps,
        output_type=NativeOutput(SocraticScaffold),
        system_prompt=_load_prompt("socratic_scaffolding.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_socratic(ctx: RunContext[_SocraticDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        return (
            "=== NARRATIVE REGISTER (from Step 1) ===\n"
            f"CHARACTERS: {entity_names}\n"
            f"LOCATIONS: {location_ids}\n"
            f"OBJECTS: {object_ids}\n"
            "\n"
            "Reference these characters, locations, and objects by their "
            "canonical names or IDs in your answers."
        )

    return agent


# =====================================================================
# Step 3 — Decomposed Topology Extraction (Physics + Social Agents)
# =====================================================================

class _PhysicsDeps(BaseModel):
    """Dependencies for Step 3a (Physics Agent: events + causal + spatial)."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    scaffold: SocraticScaffold
    previous_event_ids: List[str] = Field(default_factory=list)


class _SocialDeps(BaseModel):
    """Dependencies for Step 3b (Social Agent: info + relationship)."""
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    scaffold: SocraticScaffold
    chunk_event_ids: List[str] = Field(default_factory=list)
    previous_event_ids: List[str] = Field(default_factory=list)


def _format_scaffold(scaffold: SocraticScaffold) -> str:
    """Format a SocraticScaffold as a readable text block for agent injection."""
    if not scaffold.qa_pairs:
        return "(No scaffolding QA available for this chunk.)"
    lines = []
    for qa in scaffold.qa_pairs:
        lines.append(f"  [{qa.category.upper()}] Q: {qa.question}")
        lines.append(f"           A: {qa.answer}")
    return "\n".join(lines)


def _build_valid_id_set(reg: GlobalRegister, event_ids: List[str] | None = None) -> set[str]:
    """Build the complete set of valid IDs from the register + optional event IDs."""
    valid = (
        set(reg.entities.keys())
        | set(reg.locations.keys())
        | set(reg.objects.keys())
    )
    if event_ids:
        valid |= set(event_ids)
    return valid


# =====================================================================
# Fuzzy ID Resolution — fix LLM typos without expensive retries
# =====================================================================


def _fuzzy_resolve_id(candidate: str, valid_ids: set[str]) -> Optional[str]:
    """Attempt to resolve *candidate* to a valid ID via fuzzy matching.

    Matching strategy (in priority order):
    1. Exact match.
    2. Case-insensitive exact match (same prefix).
    3. Substring match on the base portion (``ENT_KING_DUNCAN`` ↔ ``ENT_DUNCAN``).

    Returns the matched valid ID or *None* if no match is found.
    Only considers IDs sharing the same prefix (``EVT_``, ``ENT_``, etc.)
    so prefix semantics are preserved and ``model_validator`` stays happy.
    """
    if candidate in valid_ids:
        return candidate

    # Determine prefix
    prefix = ""
    for p in ("EVT_", "ENT_", "LOC_", "OBJ_"):
        if candidate.startswith(p):
            prefix = p
            break
    if not prefix:
        return None  # can't fuzzy-match unprefixed IDs safely

    same_prefix = {vid for vid in valid_ids if vid.startswith(prefix)}
    if not same_prefix:
        return None

    # Case-insensitive exact match
    lower_map = {vid.lower(): vid for vid in same_prefix}
    if candidate.lower() in lower_map:
        return lower_map[candidate.lower()]

    # Substring match on the base (without prefix, underscores → spaces)
    cand_base = candidate[len(prefix):].replace("_", " ").lower().strip()
    if not cand_base:
        return None

    best: Optional[str] = None
    best_len = 0
    for vid in same_prefix:
        vid_base = vid[len(prefix):].replace("_", " ").lower().strip()
        if cand_base in vid_base or vid_base in cand_base:
            match_len = min(len(cand_base), len(vid_base))
            if match_len > best_len:
                best = vid
                best_len = match_len

    return best


def _fix_id(candidate: str, valid_ids: set[str], field_label: str, fixes: List[str]) -> Tuple[str, bool]:
    """Try to fuzzy-fix *candidate*. Returns (resolved_id, was_fixed).

    Appends a human-readable note to *fixes* when a correction is made.
    """
    if candidate in valid_ids:
        return candidate, False
    resolved = _fuzzy_resolve_id(candidate, valid_ids)
    if resolved:
        fixes.append(f"[Auto-Fix] {field_label} '{candidate}' → '{resolved}'")
        return resolved, True
    return candidate, False


def _build_physics_agent(config: ExtractionConfig) -> Agent[_PhysicsDeps, PhysicsExtraction]:
    """Construct the Step 3a Physics Agent — events + causal + spatial edges."""
    agent: Agent[_PhysicsDeps, PhysicsExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_PhysicsDeps,
        output_type=NativeOutput(PhysicsExtraction),
        system_prompt=_load_prompt("physics_extraction.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_physics(ctx: RunContext[_PhysicsDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        scaffold_text = _format_scaffold(ctx.deps.scaffold)

        # Build compact entity baseline so the LLM knows starting trait values
        entity_baselines: List[str] = []
        for eid in entity_ids:
            ent = reg.entities[eid]
            traits_str = ", ".join(
                f"{k}={v.value:.1f}" for k, v in ent.traits.items()
            )
            entity_baselines.append(
                f"  {eid} ({ent.name}): status={ent.status}, "
                f"loc={ent.location_id}, traits=[{traits_str}]"
            )
        baselines_block = "\n".join(entity_baselines)

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
            "Do NOT invent new ENT_, LOC_, or OBJ_ IDs.\n"
            "\n"
            "=== ENTITY BASELINES (initial trait values — use for entity_updates) ===\n"
            f"{baselines_block}\n"
            "\n"
            "When emitting entity_updates, use these baselines as reference.\n"
            "The trait_updates values should be the NEW absolute value after the event, not the delta.\n"
            "\n"
            "=== SOCRATIC SCAFFOLD (semantic pre-analysis) ===\n"
            f"{scaffold_text}\n"
            "\n"
            "Use the scaffold above to inform your extraction — it identifies "
            "hidden motivations, implicit causal chains, and unobserved variables "
            "that you should capture as events and edges."
        )

    @agent.output_validator
    def validate_physics_ids(ctx: RunContext[_PhysicsDeps], result: PhysicsExtraction) -> PhysicsExtraction:
        """Per-chunk validation — fix typos via fuzzy match, retry only for unfixable IDs."""
        reg = ctx.deps.global_register
        new_evt_ids = [e.id for e in result.events]
        valid = _build_valid_id_set(reg, ctx.deps.previous_event_ids + new_evt_ids)
        entity_ids = set(reg.entities.keys())
        location_ids = set(reg.locations.keys())
        fixes: List[str] = []
        bad: List[str] = []

        # --- Fix causal edge IDs ---
        fixed_causal: List[CausalEdge] = []
        for ce in result.causal_topology:
            updates: dict = {}
            src, src_fixed = _fix_id(ce.source_id, valid, "CausalEdge.source_id", fixes)
            tgt, tgt_fixed = _fix_id(ce.target_id, valid, "CausalEdge.target_id", fixes)
            if src != ce.source_id:
                updates["source_id"] = src
            if tgt != ce.target_id:
                updates["target_id"] = tgt
            if ce.rel_counterpart_id and ce.rel_counterpart_id not in valid:
                rc, rc_fixed = _fix_id(ce.rel_counterpart_id, valid, "CausalEdge.rel_counterpart_id", fixes)
                if rc != ce.rel_counterpart_id:
                    updates["rel_counterpart_id"] = rc
                if not rc_fixed and rc not in valid:
                    bad.append(f"CausalEdge rel_counterpart_id '{ce.rel_counterpart_id}' is not a valid ID.")
            if src not in valid and not updates.get("source_id"):
                bad.append(f"CausalEdge source_id '{ce.source_id}' is not a valid ID.")
            if tgt not in valid and not updates.get("target_id"):
                bad.append(f"CausalEdge target_id '{ce.target_id}' is not a valid ID.")
            try:
                fixed_causal.append(ce.model_copy(update=updates) if updates else ce)
            except Exception:
                # model_validator rejected the fix (prefix mismatch) — drop edge
                fixes.append(f"[Auto-Fix] Dropped causal edge {ce.source_id}→{ce.target_id} (validation error after fix)")

        # --- Fix spatial edge IDs ---
        fixed_spatial: List[SpatialEdge] = []
        for se in result.spatial_topology:
            updates = {}
            src, _ = _fix_id(se.source_id, location_ids, "SpatialEdge.source_id", fixes)
            tgt, _ = _fix_id(se.target_id, location_ids, "SpatialEdge.target_id", fixes)
            if src != se.source_id:
                updates["source_id"] = src
            if tgt != se.target_id:
                updates["target_id"] = tgt
            if src not in location_ids:
                bad.append(f"SpatialEdge source_id '{se.source_id}' is not a valid location.")
            elif tgt not in location_ids:
                bad.append(f"SpatialEdge target_id '{se.target_id}' is not a valid location.")
            else:
                fixed_spatial.append(se.model_copy(update=updates) if updates else se)

        # --- Fix entity_update IDs ---
        fixed_updates: List[EntityUpdate] = []
        for eu in result.entity_updates:
            updates = {}
            eid, _ = _fix_id(eu.entity_id, entity_ids, "EntityUpdate.entity_id", fixes)
            if eid != eu.entity_id:
                updates["entity_id"] = eid
            if eu.triggered_by and eu.triggered_by not in valid:
                trig, _ = _fix_id(eu.triggered_by, valid, "EntityUpdate.triggered_by", fixes)
                if trig != eu.triggered_by:
                    updates["triggered_by"] = trig
                if trig not in valid:
                    bad.append(f"EntityUpdate triggered_by '{eu.triggered_by}' is not a valid event.")
            if eu.new_location_id and eu.new_location_id not in location_ids:
                loc, _ = _fix_id(eu.new_location_id, location_ids, "EntityUpdate.new_location_id", fixes)
                if loc != eu.new_location_id:
                    updates["new_location_id"] = loc
                if loc not in location_ids:
                    bad.append(f"EntityUpdate new_location_id '{eu.new_location_id}' is not a valid location.")
            if eid not in entity_ids:
                bad.append(f"EntityUpdate entity_id '{eu.entity_id}' is not a valid entity.")
            else:
                fixed_updates.append(eu.model_copy(update=updates) if updates else eu)

        if fixes:
            logger.info("[Validator·Physics] Auto-fixed %d ID(s): %s", len(fixes), "; ".join(fixes))

        if bad:
            raise ModelRetry(
                "The following IDs could not be auto-resolved. "
                "Fix them using ONLY IDs from the register:\n" + "\n".join(bad)
            )

        return PhysicsExtraction(
            events=result.events,
            causal_topology=fixed_causal,
            spatial_topology=fixed_spatial,
            entity_updates=fixed_updates,
        )

    return agent


def _build_social_agent(config: ExtractionConfig) -> Agent[_SocialDeps, SocialExtraction]:
    """Construct the Step 3b Social Agent — information + relationship edges."""
    agent: Agent[_SocialDeps, SocialExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_SocialDeps,
        output_type=NativeOutput(SocialExtraction),
        system_prompt=_load_prompt("social_extraction.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_social(ctx: RunContext[_SocialDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        all_evt_ids = ctx.deps.previous_event_ids + ctx.deps.chunk_event_ids
        scaffold_text = _format_scaffold(ctx.deps.scaffold)
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
            "\n"
            "=== SOCRATIC SCAFFOLD (semantic pre-analysis) ===\n"
            f"{scaffold_text}\n"
            "\n"
            "Use the scaffold above to identify implicit social dynamics, "
            "hidden information flows, and unspoken relationship shifts."
        )

    @agent.output_validator
    def validate_social_ids(ctx: RunContext[_SocialDeps], result: SocialExtraction) -> SocialExtraction:
        """Per-chunk validation — fix typos via fuzzy match, retry only for unfixable IDs."""
        reg = ctx.deps.global_register
        entity_ids = set(reg.entities.keys())
        node_ids = entity_ids | set(reg.objects.keys())
        fixes: List[str] = []
        bad: List[str] = []

        # --- Fix information edge IDs ---
        fixed_info: List[InformationEdge] = []
        for ie in result.information_topology:
            updates: dict = {}
            src, _ = _fix_id(ie.source_id, node_ids, "InformationEdge.source_id", fixes)
            if src != ie.source_id:
                updates["source_id"] = src
            if src not in node_ids:
                bad.append(f"InformationEdge source_id '{ie.source_id}' is not a valid entity/object.")
                continue
            fixed_targets = []
            for tid in ie.target_ids:
                t, _ = _fix_id(tid, node_ids, "InformationEdge.target_id", fixes)
                if t in node_ids:
                    fixed_targets.append(t)
                else:
                    bad.append(f"InformationEdge target_id '{tid}' is not a valid entity/object.")
            if fixed_targets:
                if fixed_targets != list(ie.target_ids):
                    updates["target_ids"] = fixed_targets
                fixed_info.append(ie.model_copy(update=updates) if updates else ie)

        # --- Fix relationship edge IDs ---
        fixed_social: List[RelationshipEdge] = []
        for re_edge in result.social_topology:
            updates = {}
            src, _ = _fix_id(re_edge.source_entity_id, entity_ids, "RelationshipEdge.source_entity_id", fixes)
            tgt, _ = _fix_id(re_edge.target_entity_id, entity_ids, "RelationshipEdge.target_entity_id", fixes)
            if src != re_edge.source_entity_id:
                updates["source_entity_id"] = src
            if tgt != re_edge.target_entity_id:
                updates["target_entity_id"] = tgt
            if src not in entity_ids:
                bad.append(f"RelationshipEdge source_entity_id '{re_edge.source_entity_id}' is not a valid entity.")
            elif tgt not in entity_ids:
                bad.append(f"RelationshipEdge target_entity_id '{re_edge.target_entity_id}' is not a valid entity.")
            elif src == tgt:
                fixes.append(f"[Auto-Fix] Dropped self-referencing RelationshipEdge '{src}'→'{tgt}'")
            else:
                fixed_social.append(re_edge.model_copy(update=updates) if updates else re_edge)

        if fixes:
            logger.info("[Validator·Social] Auto-fixed %d issue(s): %s", len(fixes), "; ".join(fixes))

        if bad:
            raise ModelRetry(
                "The following IDs could not be auto-resolved. "
                "Fix them using ONLY IDs from the register:\n" + "\n".join(bad)
            )

        return SocialExtraction(
            information_topology=fixed_info,
            social_topology=fixed_social,
        )

    return agent


def extract_topology(
    chunks: List[str],
    register: GlobalRegister,
    config: ExtractionConfig | None = None,
) -> List[ChunkTopology]:
    """
    Steps 2–4: Three-agent extraction per chunk with Socratic scaffolding.

    **Step 2** — Socratic QA scaffolding: lightweight agent generates
    Who/What/Where/When/Why/How pairs to articulate hidden reasoning.

    **Step 3a** — Physics Agent: extracts events + causal + spatial edges,
    informed by the scaffold. Result validator catches hallucinated IDs.

    **Step 3b** — Social Agent: extracts information + relationship edges,
    informed by the scaffold + concrete events from 3a. Result validator
    catches hallucinated IDs.

    The GlobalRegister (from Step 1) is injected into every agent via
    dependency injection, preventing hallucination of new entity/location/
    object IDs.
    """
    config = config or ExtractionConfig()
    socratic_agent = _build_socratic_agent(config)
    physics_agent = _build_physics_agent(config)
    social_agent = _build_social_agent(config)
    topologies: List[ChunkTopology] = []
    syuzhet_counter = 0
    fabula_time_base = config.fabula_time_spacing
    all_event_ids: List[str] = []
    prev_chunk_tail = ""  # trailing context for coreference continuity

    for i, chunk in enumerate(chunks):
        logger.info("[Step 2] Processing chunk %d/%d (%d chars) — scaffolding …", i + 1, len(chunks), len(chunk))

        # Prepend trailing context from previous chunk for coreference
        overlap_ctx = ""
        if prev_chunk_tail and config.chunk_overlap_chars > 0:
            overlap_ctx = (
                f"[CONTEXT FROM PREVIOUS CHUNK — do NOT re-extract events from this]\n"
                f"{prev_chunk_tail}\n"
                f"[END CONTEXT]\n\n"
            )

        chunk_with_ctx = f"{overlap_ctx}{chunk}"

        # --- Step 2: Socratic QA Scaffolding ---
        socratic_msg = (
            f"Chunk {i + 1} of {len(chunks)}:\n\n"
            f"{chunk_with_ctx}"
        )
        socratic_deps = _SocraticDeps(global_register=register)
        try:
            scaffold_result = socratic_agent.run_sync(socratic_msg, deps=socratic_deps)
            scaffold = scaffold_result.output
        except Exception:
            logger.exception("[Step 2] Chunk %d scaffolding FAILED — using empty scaffold.", i + 1)
            scaffold = SocraticScaffold()

        logger.info("[Step 2] Chunk %d: %d QA pairs generated.", i + 1, len(scaffold.qa_pairs))

        # --- Step 3a: Physics Agent (events + causal + spatial) ---
        logger.info("[Step 3a] Processing chunk %d/%d — physics …", i + 1, len(chunks))
        physics_msg = (
            f"Chunk {i + 1} of {len(chunks)} "
            f"(syuzhet_index offset: {syuzhet_counter}, "
            f"fabula_time_base: {fabula_time_base}, "
            f"fabula_time_spacing: {config.fabula_time_spacing}):\n\n"
            f"{chunk_with_ctx}"
        )
        physics_deps = _PhysicsDeps(
            global_register=register,
            scaffold=scaffold,
            previous_event_ids=all_event_ids.copy(),
        )
        try:
            physics_result = physics_agent.run_sync(physics_msg, deps=physics_deps)
            physics = physics_result.output
        except Exception:
            logger.exception("[Step 3a] Chunk %d FAILED — returning empty physics.", i + 1)
            physics = PhysicsExtraction()

        # Retry once if zero events from a substantive chunk
        if not physics.events and len(chunk) > 500:
            logger.info("[Step 3a] Chunk %d: 0 events from %d chars — retrying …", i + 1, len(chunk))
            retry_msg = (
                "IMPORTANT: The previous extraction returned zero events. "
                "Re-read the chunk carefully — every narrative chunk contains "
                "at least one event (choice, outcome, or revelation). "
                "Look for decisions, consequences, emotional shifts, and "
                "information reveals.\n\n" + physics_msg
            )
            try:
                physics_result = physics_agent.run_sync(retry_msg, deps=physics_deps)
                physics = physics_result.output
            except Exception:
                logger.exception("[Step 3a] Chunk %d retry FAILED.", i + 1)

        logger.info(
            "[Step 3a] Chunk %d: %d events, %d causal, %d spatial edges.",
            i + 1, len(physics.events), len(physics.causal_topology),
            len(physics.spatial_topology),
        )

        # Build event summary for Social Agent
        chunk_evt_ids = [e.id for e in physics.events]
        event_summary_lines = []
        for e in physics.events:
            event_summary_lines.append(
                f"  - {e.id} (fabula={e.fabula_time}, syuzhet={e.syuzhet_index}, "
                f"type={e.event_type}, actors={e.actor_ids}, targets={e.target_ids}): "
                f"{e.description}"
            )
        event_summary = "\n".join(event_summary_lines)

        # --- Step 3b: Social Agent (information + relationship) ---
        social = SocialExtraction()
        if physics.events:
            logger.info("[Step 3b] Processing chunk %d/%d — social …", i + 1, len(chunks))
            social_msg = (
                f"Chunk {i + 1} of {len(chunks)}.\n\n"
                f"EVENTS EXTRACTED FROM THIS CHUNK:\n{event_summary}\n\n"
                f"ORIGINAL TEXT:\n{chunk}"
            )
            social_deps = _SocialDeps(
                global_register=register,
                scaffold=scaffold,
                chunk_event_ids=chunk_evt_ids,
                previous_event_ids=all_event_ids.copy(),
            )
            try:
                social_result = social_agent.run_sync(social_msg, deps=social_deps)
                social = social_result.output
            except Exception:
                logger.exception("[Step 3b] Chunk %d FAILED — returning empty social.", i + 1)

            # Retry if zero info edges with multiple events (quality gate)
            if len(physics.events) >= 2 and not social.information_topology:
                logger.info("[Step 3b] Chunk %d: 0 info edges — retrying with emphasis …", i + 1)
                retry_social_msg = (
                    "IMPORTANT: The previous extraction returned zero InformationEdge "
                    "entries. Most narrative chunks contain conversations, prophecies, "
                    "letters, confessions, orders, announcements, or rumours — each "
                    "one MUST produce an InformationEdge. Re-read the text and extract "
                    "ALL information flows.\n\n" + social_msg
                )
                try:
                    retry_result = social_agent.run_sync(retry_social_msg, deps=social_deps)
                    retry_social = retry_result.output
                    if retry_social.information_topology:
                        # Merge: keep original social, take retry's info
                        social = SocialExtraction(
                            information_topology=retry_social.information_topology,
                            social_topology=social.social_topology,
                        )
                        logger.info(
                            "[Step 3b] Chunk %d: retry recovered %d info edges.",
                            i + 1, len(social.information_topology),
                        )
                except Exception:
                    logger.exception("[Step 3b] Chunk %d info retry FAILED.", i + 1)
        else:
            logger.info("[Step 3b] Chunk %d: skipping social pass (no events).", i + 1)

        # Merge into ChunkTopology
        topo = ChunkTopology(
            events=physics.events,
            causal_topology=physics.causal_topology,
            information_topology=social.information_topology,
            social_topology=social.social_topology,
            spatial_topology=physics.spatial_topology,
            entity_updates=physics.entity_updates,
        )
        topologies.append(topo)

        # Accumulate counters
        syuzhet_counter += len(physics.events)
        if physics.events:
            max_fabula = max(e.fabula_time for e in physics.events)
            # Ensure base always advances — even if LLM used small integers,
            # guarantee at least one spacing unit beyond the previous base.
            next_from_max = (
                (max_fabula // config.fabula_time_spacing + 1)
                * config.fabula_time_spacing
            )
            next_from_prev = fabula_time_base + config.fabula_time_spacing
            fabula_time_base = max(next_from_max, next_from_prev)
        all_event_ids.extend(chunk_evt_ids)

        # Save trailing context for next chunk's coreference overlap
        if config.chunk_overlap_chars > 0:
            prev_chunk_tail = chunk[-config.chunk_overlap_chars:]

        logger.info(
            "[Step 3] Chunk %d: %d events, %d causal, %d social, %d spatial, %d info edges.",
            i + 1, len(topo.events), len(topo.causal_topology),
            len(topo.social_topology), len(topo.spatial_topology),
            len(topo.information_topology),
        )

    return topologies


# =====================================================================
# Parallel Chunk Extraction (Async)
# =====================================================================


class _ChunkParams(BaseModel):
    """Pre-allocated parameters for a single chunk in parallel extraction."""
    chunk_index: int
    total_chunks: int
    syuzhet_offset: int
    fabula_time_base: int
    prev_chunk_tail: str


def _pre_allocate_chunk_params(
    chunks: List[str],
    config: ExtractionConfig,
) -> List[_ChunkParams]:
    """Pre-compute per-chunk extraction parameters for parallel dispatch.

    Each chunk gets a deterministic ``syuzhet_offset`` and
    ``fabula_time_base`` range so parallel extraction produces
    non-overlapping indices that can be reconciled afterwards.
    The ``prev_chunk_tail`` is pre-computed from raw chunk
    boundaries — no sequential dependency required.
    """
    params: List[_ChunkParams] = []
    est = config.estimated_events_per_chunk
    for i, chunk in enumerate(chunks):
        tail = ""
        if i > 0 and config.chunk_overlap_chars > 0:
            tail = chunks[i - 1][-config.chunk_overlap_chars:]
        params.append(_ChunkParams(
            chunk_index=i,
            total_chunks=len(chunks),
            syuzhet_offset=i * est,
            fabula_time_base=(i + 1) * est * config.fabula_time_spacing,
            prev_chunk_tail=tail,
        ))
    return params


async def _extract_single_chunk_async(
    chunk: str,
    params: _ChunkParams,
    register: GlobalRegister,
    config: ExtractionConfig,
    socratic_agent: Agent,
    physics_agent: Agent,
    social_agent: Agent,
) -> ChunkTopology:
    """Process one chunk through the three-agent pipeline (async).

    Runs Socratic scaffolding → Physics → Social for a single chunk.
    ``previous_event_ids`` is empty (advisory context only; the
    ``GlobalRegister`` provides structural ID validation).
    """
    i = params.chunk_index
    n = params.total_chunks

    # Prepend trailing context from previous chunk for coreference
    overlap_ctx = ""
    if params.prev_chunk_tail and config.chunk_overlap_chars > 0:
        overlap_ctx = (
            f"[CONTEXT FROM PREVIOUS CHUNK — do NOT re-extract events from this]\n"
            f"{params.prev_chunk_tail}\n"
            f"[END CONTEXT]\n\n"
        )
    chunk_with_ctx = f"{overlap_ctx}{chunk}"

    # --- Step 2: Socratic QA Scaffolding ---
    logger.info("[Step 2·Async] Processing chunk %d/%d (%d chars) — scaffolding …", i + 1, n, len(chunk))
    socratic_msg = f"Chunk {i + 1} of {n}:\n\n{chunk_with_ctx}"
    socratic_deps = _SocraticDeps(global_register=register)
    try:
        scaffold_result = await socratic_agent.run(socratic_msg, deps=socratic_deps)
        scaffold = scaffold_result.output
    except Exception:
        logger.exception("[Step 2·Async] Chunk %d scaffolding FAILED — using empty scaffold.", i + 1)
        scaffold = SocraticScaffold()

    logger.info("[Step 2·Async] Chunk %d: %d QA pairs generated.", i + 1, len(scaffold.qa_pairs))

    # --- Step 3a: Physics Agent ---
    logger.info("[Step 3a·Async] Processing chunk %d/%d — physics …", i + 1, n)
    physics_msg = (
        f"Chunk {i + 1} of {n} "
        f"(syuzhet_index offset: {params.syuzhet_offset}, "
        f"fabula_time_base: {params.fabula_time_base}, "
        f"fabula_time_spacing: {config.fabula_time_spacing}):\n\n"
        f"{chunk_with_ctx}"
    )
    physics_deps = _PhysicsDeps(
        global_register=register,
        scaffold=scaffold,
        previous_event_ids=[],  # no cross-chunk IDs in parallel mode
    )
    try:
        physics_result = await physics_agent.run(physics_msg, deps=physics_deps)
        physics = physics_result.output
    except Exception:
        logger.exception("[Step 3a·Async] Chunk %d FAILED — returning empty physics.", i + 1)
        physics = PhysicsExtraction()

    # Retry once if zero events from a substantive chunk
    if not physics.events and len(chunk) > 500:
        logger.info("[Step 3a·Async] Chunk %d: 0 events from %d chars — retrying …", i + 1, len(chunk))
        retry_msg = (
            "IMPORTANT: The previous extraction returned zero events. "
            "Re-read the chunk carefully — every narrative chunk contains "
            "at least one event (choice, outcome, or revelation). "
            "Look for decisions, consequences, emotional shifts, and "
            "information reveals.\n\n" + physics_msg
        )
        try:
            physics_result = await physics_agent.run(retry_msg, deps=physics_deps)
            physics = physics_result.output
        except Exception:
            logger.exception("[Step 3a·Async] Chunk %d retry FAILED.", i + 1)

    logger.info(
        "[Step 3a·Async] Chunk %d: %d events, %d causal, %d spatial edges.",
        i + 1, len(physics.events), len(physics.causal_topology),
        len(physics.spatial_topology),
    )

    # Build event summary for Social Agent
    chunk_evt_ids = [e.id for e in physics.events]
    event_summary_lines = []
    for e in physics.events:
        event_summary_lines.append(
            f"  - {e.id} (fabula={e.fabula_time}, syuzhet={e.syuzhet_index}, "
            f"type={e.event_type}, actors={e.actor_ids}, targets={e.target_ids}): "
            f"{e.description}"
        )
    event_summary = "\n".join(event_summary_lines)

    # --- Step 3b: Social Agent ---
    social = SocialExtraction()
    if physics.events:
        logger.info("[Step 3b·Async] Processing chunk %d/%d — social …", i + 1, n)
        social_msg = (
            f"Chunk {i + 1} of {n}.\n\n"
            f"EVENTS EXTRACTED FROM THIS CHUNK:\n{event_summary}\n\n"
            f"ORIGINAL TEXT:\n{chunk}"
        )
        social_deps = _SocialDeps(
            global_register=register,
            scaffold=scaffold,
            chunk_event_ids=chunk_evt_ids,
            previous_event_ids=[],  # no cross-chunk IDs in parallel mode
        )
        try:
            social_result = await social_agent.run(social_msg, deps=social_deps)
            social = social_result.output
        except Exception:
            logger.exception("[Step 3b·Async] Chunk %d FAILED — returning empty social.", i + 1)

        # Retry if zero info edges with multiple events (quality gate)
        if len(physics.events) >= 2 and not social.information_topology:
            logger.info("[Step 3b·Async] Chunk %d: 0 info edges — retrying with emphasis …", i + 1)
            retry_social_msg = (
                "IMPORTANT: The previous extraction returned zero InformationEdge "
                "entries. Most narrative chunks contain conversations, prophecies, "
                "letters, confessions, orders, announcements, or rumours — each "
                "one MUST produce an InformationEdge. Re-read the text and extract "
                "ALL information flows.\n\n" + social_msg
            )
            try:
                retry_result = await social_agent.run(retry_social_msg, deps=social_deps)
                retry_social = retry_result.output
                if retry_social.information_topology:
                    social = SocialExtraction(
                        information_topology=retry_social.information_topology,
                        social_topology=social.social_topology,
                    )
                    logger.info(
                        "[Step 3b·Async] Chunk %d: retry recovered %d info edges.",
                        i + 1, len(social.information_topology),
                    )
            except Exception:
                logger.exception("[Step 3b·Async] Chunk %d info retry FAILED.", i + 1)
    else:
        logger.info("[Step 3b·Async] Chunk %d: skipping social pass (no events).", i + 1)

    topo = ChunkTopology(
        events=physics.events,
        causal_topology=physics.causal_topology,
        information_topology=social.information_topology,
        social_topology=social.social_topology,
        spatial_topology=physics.spatial_topology,
        entity_updates=physics.entity_updates,
    )
    logger.info(
        "[Step 3·Async] Chunk %d: %d events, %d causal, %d social, %d spatial, %d info edges.",
        i + 1, len(topo.events), len(topo.causal_topology),
        len(topo.social_topology), len(topo.spatial_topology),
        len(topo.information_topology),
    )
    return topo


def _reconcile_chunk_topologies(
    topologies: List[ChunkTopology],
    config: ExtractionConfig,
) -> List[ChunkTopology]:
    """Post-merge reconciliation for parallel-extracted chunk topologies.

    1. Detects and renames duplicate ``EVT_`` IDs across chunks
       (appends ``_cN`` suffix where N is the chunk index).
    2. Re-numbers ``syuzhet_index`` globally in chunk order.
    3. Ensures ``fabula_time`` ordering across chunks: all events
       in chunk N have ``fabula_time`` < all events in chunk N+1.
    """
    # --- Pass 1: Detect and resolve duplicate event IDs across chunks ---
    global_evt_ids: Dict[str, int] = {}  # evt_id → first chunk index
    chunk_renames: List[Dict[str, str]] = [{} for _ in topologies]

    for ci, topo in enumerate(topologies):
        for evt in topo.events:
            if evt.id in global_evt_ids:
                # Collision — rename in the later chunk
                new_id = f"{evt.id}_c{ci}"
                # Ensure the rename itself doesn't collide
                suffix = ci
                while new_id in global_evt_ids:
                    suffix += len(topologies)
                    new_id = f"{evt.id}_c{suffix}"
                chunk_renames[ci][evt.id] = new_id
                global_evt_ids[new_id] = ci
                logger.info(
                    "[Reconcile] Duplicate EVT ID %s in chunk %d — renamed to %s.",
                    evt.id, ci, new_id,
                )
            else:
                global_evt_ids[evt.id] = ci

    # Apply renames to events and all edge references
    reconciled: List[ChunkTopology] = []
    for ci, topo in enumerate(topologies):
        rmap = chunk_renames[ci]
        if rmap:
            topo = _apply_event_renames(topo, rmap)
        reconciled.append(topo)

    # --- Pass 2: Re-number syuzhet_index globally in chunk order ---
    # Build per-chunk old → new syuzhet mappings for remapping discovered_at_syuzhet
    chunk_syuzhet_remaps: List[Dict[int, int]] = []
    syuzhet_counter = 0
    for topo in reconciled:
        remap: Dict[int, int] = {}
        sorted_events = sorted(topo.events, key=lambda e: e.syuzhet_index)
        for evt in sorted_events:
            remap[evt.syuzhet_index] = syuzhet_counter
            evt.syuzhet_index = syuzhet_counter
            syuzhet_counter += 1
        chunk_syuzhet_remaps.append(remap)

    # Remap discovered_at_syuzhet on InformationEdges using per-chunk maps
    for ci, topo in enumerate(reconciled):
        remap = chunk_syuzhet_remaps[ci]
        for ie in topo.information_topology:
            if ie.discovered_at_syuzhet in remap:
                ie.discovered_at_syuzhet = remap[ie.discovered_at_syuzhet]

    # --- Pass 3: Ensure fabula_time inter-chunk ordering ---
    # Within each chunk, preserve relative order. Between chunks,
    # shift later chunks so they don't overlap earlier ones.
    spacing = config.fabula_time_spacing
    running_max = 0
    for topo in reconciled:
        if not topo.events:
            continue
        chunk_min = min(e.fabula_time for e in topo.events)
        # Ensure this chunk starts above the running max
        needed_shift = 0
        if chunk_min <= running_max:
            needed_shift = running_max + spacing - chunk_min
        if needed_shift > 0:
            _shift_fabula_times(topo, needed_shift)
        if topo.events:
            running_max = max(e.fabula_time for e in topo.events)

    return reconciled


def _apply_event_renames(topo: ChunkTopology, rmap: Dict[str, str]) -> ChunkTopology:
    """Apply event ID renames to all fields in a ChunkTopology."""
    def _r(eid: str) -> str:
        return rmap.get(eid, eid)

    new_events = [
        e.model_copy(update={"id": _r(e.id)}) for e in topo.events
    ]
    new_causal = [
        ce.model_copy(update={
            "source_id": _r(ce.source_id),
            "target_id": _r(ce.target_id),
        }) for ce in topo.causal_topology
    ]
    new_info = [
        ie.model_copy(update={
            "source_id": _r(ie.source_id),
            "target_ids": [_r(tid) for tid in ie.target_ids],
        })
        for ie in topo.information_topology
    ]
    new_entity_updates = [
        eu.model_copy(update={
            "triggered_by": _r(eu.triggered_by) if eu.triggered_by else None,
        }) for eu in topo.entity_updates
    ]
    return ChunkTopology(
        events=new_events,
        causal_topology=new_causal,
        information_topology=new_info,
        social_topology=topo.social_topology,
        spatial_topology=topo.spatial_topology,
        entity_updates=new_entity_updates,
    )


def _shift_fabula_times(topo: ChunkTopology, shift: int) -> None:
    """Shift all fabula_time values in a ChunkTopology by *shift* (in-place)."""
    for evt in topo.events:
        evt.fabula_time += shift
    for ce in topo.causal_topology:
        ce.fabula_time += shift
    for ie in topo.information_topology:
        ie.established_at_fabula += shift
        if ie.terminated_at_fabula is not None:
            ie.terminated_at_fabula += shift
    for se in topo.social_topology:
        se.last_updated_fabula += shift
    for sp in topo.spatial_topology:
        sp.established_at_fabula += shift
        if sp.destroyed_at_fabula is not None:
            sp.destroyed_at_fabula += shift
    for eu in topo.entity_updates:
        eu.fabula_time += shift
        for belief in eu.new_beliefs:
            belief.established_at_fabula += shift


async def extract_topology_async(
    chunks: List[str],
    register: GlobalRegister,
    config: ExtractionConfig | None = None,
) -> List[ChunkTopology]:
    """Async variant of :func:`extract_topology` — extracts chunks in parallel.

    Chunks are dispatched concurrently (limited by
    ``config.max_concurrent_chunks``) with pre-allocated syuzhet and
    fabula_time ranges.  After all chunks complete, a reconciliation
    pass re-numbers ``syuzhet_index`` globally, ensures inter-chunk
    ``fabula_time`` ordering, and resolves any duplicate event IDs.
    """
    config = config or ExtractionConfig()
    socratic_agent = _build_socratic_agent(config)
    physics_agent = _build_physics_agent(config)
    social_agent = _build_social_agent(config)

    params_list = _pre_allocate_chunk_params(chunks, config)
    semaphore = asyncio.Semaphore(config.max_concurrent_chunks)

    async def _guarded_extract(chunk: str, params: _ChunkParams) -> ChunkTopology:
        async with semaphore:
            return await _extract_single_chunk_async(
                chunk, params, register, config,
                socratic_agent, physics_agent, social_agent,
            )

    topologies = await asyncio.gather(*[
        _guarded_extract(chunk, params)
        for chunk, params in zip(chunks, params_list)
    ])
    topologies_list = list(topologies)

    # Post-merge reconciliation
    topologies_list = _reconcile_chunk_topologies(topologies_list, config)

    logger.info(
        "[Step 3·Async] All %d chunks extracted and reconciled.", len(topologies_list),
    )
    return topologies_list


# =====================================================================
# Fabula-Time Normalization
# =====================================================================


def _normalize_fabula_times(ws: WorldStateV1, spacing: int = 1000) -> WorldStateV1:
    """
    Re-space ``fabula_time`` values using *spacing* when the LLM ignores
    the requested 100-scale and returns small sequential integers (1, 2, 3 …).

    Builds a monotonic mapping ``{old_time: new_time}`` from the sorted
    unique fabula_time values found on events, then applies it to every
    temporal field across events, edges, and entity beliefs.

    Returns the original world-state unchanged when times are already
    well-spaced (median gap ≥ spacing / 2).
    """
    if not ws.events:
        return ws

    unique_times = sorted({e.fabula_time for e in ws.events})
    if len(unique_times) < 2:
        return ws

    diffs = [unique_times[i + 1] - unique_times[i] for i in range(len(unique_times) - 1)]
    median_diff = sorted(diffs)[len(diffs) // 2]
    if median_diff >= spacing // 2:
        return ws  # already well-spaced

    # Build old → new mapping
    time_map: dict[int, int] = {t: (i + 1) * spacing for i, t in enumerate(unique_times)}

    def _map(t: int | None) -> int | None:
        if t is None:
            return None
        return time_map.get(t, t)

    new_events = [
        e.model_copy(update={"fabula_time": time_map[e.fabula_time]}) for e in ws.events
    ]
    new_causal = [
        ce.model_copy(update={"fabula_time": _map(ce.fabula_time) or ce.fabula_time})
        for ce in ws.causal_topology
    ]
    new_info = [
        ie.model_copy(update={
            "established_at_fabula": _map(ie.established_at_fabula) or ie.established_at_fabula,
            "terminated_at_fabula": _map(ie.terminated_at_fabula),
        })
        for ie in ws.information_topology
    ]
    new_social = [
        re_edge.model_copy(update={
            "last_updated_fabula": _map(re_edge.last_updated_fabula) or re_edge.last_updated_fabula,
        })
        for re_edge in ws.social_topology
    ]
    new_spatial = [
        se.model_copy(update={
            "established_at_fabula": _map(se.established_at_fabula) or se.established_at_fabula,
            "destroyed_at_fabula": _map(se.destroyed_at_fabula),
        })
        for se in ws.spatial_topology
    ]

    # Remap belief established_at_fabula (0 = pre-story, stays 0)
    new_entities: dict[str, Entity] = {}
    for eid, ent in ws.entities.items():
        new_beliefs = [
            b.model_copy(update={
                "established_at_fabula": _map(b.established_at_fabula) or b.established_at_fabula,
            })
            for b in ent.beliefs
        ]
        new_timeline = [
            snap.model_copy(update={"fabula_time": _map(snap.fabula_time) or snap.fabula_time})
            for snap in ent.state_timeline
        ]
        new_entities[eid] = ent.model_copy(update={
            "beliefs": new_beliefs,
            "state_timeline": new_timeline,
        })

    logger.info(
        "[Normalize] Rescaled %d unique fabula_time values (median gap %d → %d).",
        len(unique_times), median_diff, spacing,
    )

    return WorldStateV1(
        locations=ws.locations,
        objects=ws.objects,
        entities=new_entities,
        events=new_events,
        causal_topology=new_causal,
        spatial_topology=new_spatial,
        information_topology=new_info,
        social_topology=new_social,
    )


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


def _deduplicate_causal(edges: List[CausalEdge]) -> List[CausalEdge]:
    """Deduplicate causal edges by (source, target, causality_type, fabula_time).

    Keeps the edge with the highest ``causal_force`` when duplicates
    are found (the stronger signal wins).
    """
    best: dict[tuple, CausalEdge] = {}
    for e in edges:
        key = (e.source_id, e.target_id, e.causality_type, e.fabula_time)
        if key not in best or e.causal_force > best[key].causal_force:
            best[key] = e
    return list(best.values())


def _deduplicate_info(edges: List[InformationEdge]) -> List[InformationEdge]:
    """Deduplicate information edges by (source, targets, medium, established_at).

    Keeps the first occurrence when duplicates are found.
    """
    seen: set[tuple] = set()
    deduped: List[InformationEdge] = []
    for e in edges:
        key = (e.source_id, tuple(sorted(e.target_ids)), e.medium, e.established_at_fabula)
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


# Public aliases for reuse outside the ingestion pipeline
deduplicate_social = _deduplicate_social
deduplicate_spatial = _deduplicate_spatial
deduplicate_causal = _deduplicate_causal
deduplicate_info = _deduplicate_info


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

    # Collect entity updates from all chunks into state_timeline
    all_entity_updates: Dict[str, List[EntityStateSnapshot]] = {}
    for topo in topologies:
        for eu in topo.entity_updates:
            snap = EntityStateSnapshot(
                fabula_time=eu.fabula_time,
                triggered_by=eu.triggered_by,
                traits=eu.trait_updates,
                beliefs_added=eu.new_beliefs,
                beliefs_invalidated=eu.invalidated_belief_targets,
                status=eu.new_status,
                location_id=eu.new_location_id,
            )
            all_entity_updates.setdefault(eu.entity_id, []).append(snap)

    # Sort events chronologically
    events.sort(key=lambda e: e.fabula_time)
    causal_topology.sort(key=lambda c: c.fabula_time)

    # Deduplicate relationship, spatial, causal, and information edges across chunks
    social_before = len(social_topology)
    social_topology = _deduplicate_social(social_topology)
    spatial_before = len(spatial_topology)
    spatial_topology = _deduplicate_spatial(spatial_topology)
    causal_before = len(causal_topology)
    causal_topology = _deduplicate_causal(causal_topology)
    info_before = len(information_topology)
    information_topology = _deduplicate_info(information_topology)
    deduped_parts = []
    if social_before != len(social_topology):
        deduped_parts.append(f"social {social_before}→{len(social_topology)}")
    if spatial_before != len(spatial_topology):
        deduped_parts.append(f"spatial {spatial_before}→{len(spatial_topology)}")
    if causal_before != len(causal_topology):
        deduped_parts.append(f"causal {causal_before}→{len(causal_topology)}")
    if info_before != len(information_topology):
        deduped_parts.append(f"info {info_before}→{len(information_topology)}")
    if deduped_parts:
        logger.info("[Step 3] Deduplicated edges: %s.", ", ".join(deduped_parts))

    ws = WorldStateV1(
        locations=register.locations,
        objects=register.objects,
        entities={
            eid: (
                ent.model_copy(update={"state_timeline": sorted(all_entity_updates[eid], key=lambda s: s.fabula_time)})
                if eid in all_entity_updates
                else ent
            )
            for eid, ent in register.entities.items()
        },
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


# =====================================================================
# Auto-Repair — programmatically fix broken links
# =====================================================================

def _auto_repair(ws: WorldStateV1) -> Tuple[WorldStateV1, List[str]]:
    """
    Programmatically repair a WorldStateV1 by removing broken edges
    and duplicate events. Returns (repaired_ws, list_of_repairs).

    This is inspired by GraphRAG's entity-summarization merging step
    but applied at the validation layer — strip provably broken
    references rather than forcing LLM re-extraction.
    """
    repairs: List[str] = []

    valid_ids = (
        set(ws.locations.keys())
        | set(ws.objects.keys())
        | set(ws.entities.keys())
        | {e.id for e in ws.events}
    )
    event_ids = {e.id for e in ws.events}
    entity_ids = set(ws.entities.keys())
    location_ids = set(ws.locations.keys())
    node_ids = entity_ids | set(ws.objects.keys())

    # --- Deduplicate events (keep first occurrence) ---
    seen_evt: set[str] = set()
    deduped_events: List[EventNode] = []
    for evt in ws.events:
        if evt.id in seen_evt:
            repairs.append(f"Removed duplicate event '{evt.id}'.")
        else:
            seen_evt.add(evt.id)
            deduped_events.append(evt)

    # --- Fix broken event actor_ids / target_ids references ---
    object_ids = set(ws.objects.keys())
    clean_events: List[EventNode] = []
    for evt in deduped_events:
        updates: dict = {}
        bad_actors = [a for a in evt.actor_ids if a not in entity_ids]
        if bad_actors:
            repairs.append(f"Removed invalid actor_ids {bad_actors} from event '{evt.id}'.")
            updates["actor_ids"] = [a for a in evt.actor_ids if a in entity_ids]
        bad_targets = [t for t in evt.target_ids if t not in (entity_ids | object_ids)]
        if bad_targets:
            repairs.append(f"Removed invalid target_ids {bad_targets} from event '{evt.id}'.")
            updates["target_ids"] = [t for t in evt.target_ids if t in (entity_ids | object_ids)]
        clean_events.append(evt.model_copy(update=updates) if updates else evt)

    # --- Fix broken entity location_ids ---
    first_loc = next(iter(ws.locations.keys()), None)
    if first_loc:
        new_entities_map: dict[str, Entity] = {}
        for eid, ent in ws.entities.items():
            if ent.location_id not in location_ids:
                repairs.append(f"Fixed entity '{eid}' location_id '{ent.location_id}' → '{first_loc}'.")
                new_entities_map[eid] = ent.model_copy(update={"location_id": first_loc})
            else:
                new_entities_map[eid] = ent
        ws = ws.model_copy(update={"entities": new_entities_map})

    # --- Strip broken causal edges ---
    clean_causal: List[CausalEdge] = []
    for ce in ws.causal_topology:
        if ce.source_id not in valid_ids:
            repairs.append(f"Removed causal edge: source '{ce.source_id}' not in node set.")
        elif ce.target_id not in valid_ids:
            repairs.append(f"Removed causal edge: target '{ce.target_id}' not in node set.")
        elif ce.rel_counterpart_id and ce.rel_counterpart_id not in valid_ids:
            repairs.append(f"Removed causal edge: rel_counterpart_id '{ce.rel_counterpart_id}' not in node set.")
        else:
            clean_causal.append(ce)

    # --- Strip broken social edges ---
    clean_social: List[RelationshipEdge] = []
    for re_edge in ws.social_topology:
        if re_edge.source_entity_id not in entity_ids or re_edge.target_entity_id not in entity_ids:
            repairs.append(
                f"Removed relationship edge: '{re_edge.source_entity_id}' → '{re_edge.target_entity_id}'."
            )
        else:
            clean_social.append(re_edge)

    # --- Strip broken spatial edges ---
    clean_spatial: List[SpatialEdge] = []
    for se in ws.spatial_topology:
        if se.source_id not in location_ids or se.target_id not in location_ids:
            repairs.append(f"Removed spatial edge: '{se.source_id}' → '{se.target_id}'.")
        else:
            clean_spatial.append(se)

    # --- Strip broken information edges ---
    clean_info: List[InformationEdge] = []
    for ie in ws.information_topology:
        if ie.source_id not in node_ids:
            repairs.append(f"Removed info edge: source '{ie.source_id}' not in entities/objects.")
            continue
        clean_targets = [t for t in ie.target_ids if t in node_ids]
        bad_targets = [t for t in ie.target_ids if t not in node_ids]
        for bt in bad_targets:
            repairs.append(f"Removed info edge target '{bt}' (not in entities/objects).")
        if clean_targets:
            clean_info.append(ie.model_copy(update={"target_ids": clean_targets}))
        else:
            repairs.append(f"Removed info edge from '{ie.source_id}' (all targets invalid).")

    if repairs:
        logger.info("[Auto-Repair] Applied %d repairs.", len(repairs))
        ws = WorldStateV1(
            locations=ws.locations,
            objects=ws.objects,
            entities=ws.entities,
            events=clean_events,
            causal_topology=clean_causal,
            spatial_topology=clean_spatial,
            information_topology=clean_info,
            social_topology=clean_social,
        )

    return ws, repairs


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
    event_ids = {e.id for e in ws.events}
    for ce in ws.causal_topology:
        if ce.source_id not in valid_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"CausalEdge.source_id '{ce.source_id}' not in node set.",
            ))
        if ce.target_id not in valid_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"CausalEdge.target_id '{ce.target_id}' not in node set.",
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

    # Check event actor_ids (must be entities) and target_ids (must be entities/objects)
    object_ids = set(ws.objects.keys())
    for evt in ws.events:
        for aid in evt.actor_ids:
            if aid not in entity_ids:
                issues.append(ValidationIssue(
                    severity="error", category="hallucinated_id",
                    detail=f"EventNode '{evt.id}' actor_ids entry '{aid}' is not a valid entity.",
                ))
        for tid in evt.target_ids:
            if tid not in (entity_ids | object_ids):
                issues.append(ValidationIssue(
                    severity="error", category="hallucinated_id",
                    detail=f"EventNode '{evt.id}' target_ids entry '{tid}' is not a valid entity/object.",
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

    # Check entity belief target_id references
    all_valid_belief_targets = (
        set(ws.locations.keys()) | set(ws.objects.keys())
        | set(ws.entities.keys()) | {e.id for e in ws.events}
    )
    for eid, ent in ws.entities.items():
        for belief in ent.beliefs:
            if belief.target_id not in all_valid_belief_targets:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=f"Entity '{eid}' belief target_id '{belief.target_id}' not in locations/objects/entities/events.",
                ))

    # Check entity state_timeline references
    for eid, ent in ws.entities.items():
        prev_ft = -1
        for snap in ent.state_timeline:
            if snap.triggered_by and snap.triggered_by not in event_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=f"Entity '{eid}' state_timeline triggered_by '{snap.triggered_by}' not in events.",
                ))
            if snap.location_id and snap.location_id not in location_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=f"Entity '{eid}' state_timeline location_id '{snap.location_id}' not in locations.",
                ))
            if snap.fabula_time < prev_ft:
                issues.append(ValidationIssue(
                    severity="warning", category="temporal",
                    detail=f"Entity '{eid}' state_timeline not monotonic: fabula_time {snap.fabula_time} follows {prev_ft}.",
                ))
            prev_ft = snap.fabula_time

    # Check for duplicate event IDs
    seen_evt_ids: set[str] = set()
    for evt in ws.events:
        if evt.id in seen_evt_ids:
            issues.append(ValidationIssue(
                severity="error", category="duplicate",
                detail=f"Duplicate event ID: '{evt.id}'.",
            ))
        seen_evt_ids.add(evt.id)

    # --- Orphan event check ---
    if ws.events:
        referenced_events: set[str] = set()
        for ce in ws.causal_topology:
            if ce.source_id.startswith("EVT_"):
                referenced_events.add(ce.source_id)
            if ce.target_id.startswith("EVT_"):
                referenced_events.add(ce.target_id)
        orphan_events = [e for e in ws.events if e.id not in referenced_events]
        if orphan_events:
            orphan_ids = [e.id for e in orphan_events[:10]]
            issues.append(ValidationIssue(
                severity="warning", category="orphan",
                detail=(
                    f"{len(orphan_events)} event(s) not referenced by any causal edge: "
                    f"{orphan_ids}{'…' if len(orphan_events) > 10 else ''}. "
                    f"Consider adding causal connections."
                ),
            ))

    # --- Information edge density check ---
    if len(ws.events) >= 3 and len(ws.information_topology) == 0:
        issues.append(ValidationIssue(
            severity="warning", category="missing_information",
            detail="Zero information edges extracted. Most narratives contain conversations, "
            "letters, or revelations that should produce InformationEdge entries.",
        ))
    elif len(ws.events) >= 5 and len(ws.information_topology) < len(ws.events) // 5:
        issues.append(ValidationIssue(
            severity="warning", category="missing_information",
            detail=(
                f"Low information edge density: {len(ws.information_topology)} info edges "
                f"for {len(ws.events)} events (ratio {len(ws.information_topology)/len(ws.events):.2f}). "
                f"Expected at least 1 info edge per 5 events."
            ),
        ))

    # --- Self-referencing relationship edges ---
    for re_edge in ws.social_topology:
        if re_edge.source_entity_id == re_edge.target_entity_id:
            issues.append(ValidationIssue(
                severity="warning", category="contradiction",
                detail=(
                    f"Self-referencing RelationshipEdge: "
                    f"'{re_edge.source_entity_id}' → '{re_edge.target_entity_id}'. "
                    f"Relationships should be between different entities."
                ),
            ))

    # --- Mutation edge coverage check ---
    # Events that change entity state should have mutation causal edges
    if ws.events and ws.causal_topology:
        mutation_target_events: set[str] = set()
        for ce in ws.causal_topology:
            if ce.causality_type in ("mutation", "mutation_social"):
                mutation_target_events.add(ce.source_id)
        # Significant events: choices and outcomes typically cause state changes
        sig_events = [
            e for e in ws.events
            if e.event_type in ("choice", "outcome") and e.target_ids
        ]
        unmutated = [e for e in sig_events if e.id not in mutation_target_events]
        if len(unmutated) > len(sig_events) // 2 and len(unmutated) >= 3:
            sample_ids = [e.id for e in unmutated[:5]]
            issues.append(ValidationIssue(
                severity="warning", category="missing_mutation",
                detail=(
                    f"{len(unmutated)}/{len(sig_events)} significant events "
                    f"(choices/outcomes with targets) lack mutation causal edges: "
                    f"{sample_ids}{'…' if len(unmutated) > 5 else ''}. "
                    f"Events that affect characters should produce mutation edges."
                ),
            ))

    # --- State timeline coverage check ---
    # Entities involved in events (as targets) should have state_timeline entries
    if ws.events:
        targeted_entities: set[str] = set()
        for evt in ws.events:
            for tid in evt.target_ids:
                if tid in entity_ids:
                    targeted_entities.add(tid)
        entities_with_timeline = {
            eid for eid in entity_ids
            if ws.entities[eid].state_timeline
        }
        missing_timeline = targeted_entities - entities_with_timeline
        if missing_timeline and len(missing_timeline) >= 2:
            issues.append(ValidationIssue(
                severity="warning", category="missing_state_timeline",
                detail=(
                    f"{len(missing_timeline)} entities targeted by events lack "
                    f"state_timeline entries: {sorted(missing_timeline)[:5]}. "
                    f"Entity state changes should be tracked in state_timeline."
                ),
            ))

    # --- Time validation ---
    issues.extend(_validate_time_ordering(ws))

    # --- Dead-actor validation ---
    issues.extend(_validate_dead_actors(ws))

    return issues


def _validate_dead_actors(ws: WorldStateV1) -> List[ValidationIssue]:
    """Check that entities marked dead do not act after their death event.

    Emits **warnings** (not errors) because narratives commonly use fake
    deaths, ghost scenes, flashback POV, and murder-suicides. The LLM
    validator can promote these to errors when it knows the story context.
    """
    issues: List[ValidationIssue] = []

    # Find entities with status == "dead"
    dead_entities = {eid for eid, ent in ws.entities.items() if ent.status == "dead"}
    if not dead_entities:
        return issues

    # For each dead entity, find the earliest death-like event.
    # A death event can be ANY event_type — suicides are often "choice",
    # learning someone died is "revelation", and killings are "outcome".
    # We match:
    #   (a) entity is in target_ids, OR
    #   (b) entity is in actor_ids with no targets (self-caused death)
    death_times: dict[str, int] = {}
    death_events: dict[str, str] = {}
    for evt in ws.events:
        # Case (a): entity is a target
        for tid in evt.target_ids:
            if tid in dead_entities:
                if tid not in death_times or evt.fabula_time < death_times[tid]:
                    death_times[tid] = evt.fabula_time
                    death_events[tid] = evt.id
        # Case (b): entity is an actor with no targets (suicide / self-death)
        if not evt.target_ids:
            for aid in evt.actor_ids:
                if aid in dead_entities:
                    if aid not in death_times or evt.fabula_time < death_times[aid]:
                        death_times[aid] = evt.fabula_time
                        death_events[aid] = evt.id

    # Check for actor references after death
    for evt in ws.events:
        for aid in evt.actor_ids:
            if aid in death_times:
                death_t = death_times[aid]
                death_evt = death_events[aid]
                # Skip the death event itself (the entity is the actor of their own death)
                if evt.id == death_evt:
                    continue
                if evt.fabula_time > death_t:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="contradiction",
                        detail=(
                            f"Dead entity '{aid}' acts in '{evt.id}' "
                            f"(fabula={evt.fabula_time}) after death in "
                            f"'{death_evt}' (fabula={death_t}). "
                            f"Could be a fake death, ghost, or flashback."
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

    # 3. Check causal edges: for chain_reaction (event→event), cause must
    #    precede or coincide with effect in fabula_time.
    evt_fabula = {e.id: e.fabula_time for e in ws.events}
    for ce in ws.causal_topology:
        if ce.causality_type != "chain_reaction":
            continue  # temporal ordering only meaningful for event→event
        src_t = evt_fabula.get(ce.source_id)
        tgt_t = evt_fabula.get(ce.target_id)
        if src_t is not None and tgt_t is not None and src_t + ce.propagation_delay > tgt_t:
            issues.append(ValidationIssue(
                severity="error", category="temporal",
                detail=(
                    f"Causal edge '{ce.source_id}' (fabula={src_t}, delay={ce.propagation_delay}) → "
                    f"'{ce.target_id}' (fabula={tgt_t}): effect manifests before cause + delay."
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


def _build_correction_agent(config: ExtractionConfig) -> Agent[None, WorldStateV1]:
    """Construct the correction agent that repairs a WorldStateV1 given errors."""
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldStateV1),
        system_prompt=_load_prompt("correction.md"),
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

    # Summarise programmatic findings so the LLM doesn't duplicate them
    if prog_issues:
        prog_summary = "\n".join(
            f"  [{i.severity}/{i.category}] {i.detail}" for i in prog_issues
        )
        preamble = (
            "The following issues were ALREADY found by programmatic validation. "
            "Do NOT re-report them:\n" + prog_summary + "\n\n"
        )
    else:
        preamble = "Programmatic validation found 0 issues.\n\n"

    result = agent.run_sync(
        preamble + f"Validate the following WorldStateV1 JSON:\n\n{ws_json}"
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

    # Step 3: Assembly + Normalize + Auto-Repair + Validation
    world_state = assemble_world_state(register, topologies)

    # Normalize fabula_time if the LLM used small integers
    world_state = _normalize_fabula_times(world_state, config.fabula_time_spacing)

    # Auto-repair broken links and duplicates before validation
    world_state, repairs = _auto_repair(world_state)
    if repairs:
        logger.info("[Pipeline] Auto-repaired %d issues before validation.", len(repairs))

    report = validate_world_state(world_state, config)

    # --- Correction retry loop ---
    # If programmatic errors remain after auto-repair, attempt LLM correction
    for retry_num in range(config.max_correction_retries):
        prog_errors = [i for i in report.issues if i.severity == "error"]
        if not prog_errors:
            break

        logger.info(
            "[Pipeline·Correction %d/%d] %d errors remain — running correction agent.",
            retry_num + 1, config.max_correction_retries, len(prog_errors),
        )

        try:
            correction_agent = _build_correction_agent(config)
            ws_json = world_state.model_dump_json(indent=2)
            max_chars = 80_000
            if len(ws_json) > max_chars:
                ws_json = ws_json[:max_chars] + "\n... [TRUNCATED]"

            error_summary = "\n".join(
                f"  [{e.category}] {e.detail}" for e in prog_errors
            )
            correction_msg = (
                f"The following {len(prog_errors)} error(s) were found in this WorldStateV1. "
                f"Fix them and return the corrected WorldStateV1:\n\n"
                f"ERRORS:\n{error_summary}\n\n"
                f"WORLD STATE:\n{ws_json}"
            )

            correction_result = correction_agent.run_sync(correction_msg)
            world_state = correction_result.output
            logger.info("[Pipeline·Correction %d] Correction applied.", retry_num + 1)

            # Re-normalize, re-repair, and re-validate after correction
            world_state = _normalize_fabula_times(world_state, config.fabula_time_spacing)
            world_state, new_repairs = _auto_repair(world_state)
            if new_repairs:
                repairs.extend(new_repairs)
            report = validate_world_state(world_state, config)

        except Exception:
            logger.exception(
                "[Pipeline·Correction %d] Correction agent FAILED — keeping previous state.",
                retry_num + 1,
            )
            break

    return world_state, report


async def run_extraction_async(
    text: str,
    config: ExtractionConfig | None = None,
) -> Tuple[WorldStateV1, ValidationReport]:
    """Async variant of :func:`run_extraction`.

    Uses :func:`extract_ontology_async` (parallel 1b/1c) and
    :func:`extract_topology_async` (parallel chunk dispatch) for
    higher throughput.  Validation and correction remain synchronous
    (they are fast compared to extraction).
    """
    config = config or ExtractionConfig()
    logger.info("[Pipeline·Async] Starting extraction with model=%s, strategy=%s", config.model, config.chunk_strategy)

    # Step 1: Global Ontology (parallel 1b + 1c)
    register = await extract_ontology_async(text, config)

    # Step 2: Chunk Topology (parallel chunks)
    chunks = chunk_text(text, strategy=config.chunk_strategy, min_chunk_chars=config.min_chunk_chars)
    logger.info("[Pipeline·Async] Text split into %d chunks.", len(chunks))
    topologies = await extract_topology_async(chunks, register, config)

    # Step 3: Assembly + Normalize + Auto-Repair + Validation (same as sync)
    world_state = assemble_world_state(register, topologies)
    world_state = _normalize_fabula_times(world_state, config.fabula_time_spacing)
    world_state, repairs = _auto_repair(world_state)
    if repairs:
        logger.info("[Pipeline·Async] Auto-repaired %d issues before validation.", len(repairs))

    report = validate_world_state(world_state, config)

    # --- Correction retry loop (sync — fast relative to extraction) ---
    for retry_num in range(config.max_correction_retries):
        prog_errors = [i for i in report.issues if i.severity == "error"]
        if not prog_errors:
            break

        logger.info(
            "[Pipeline·Async·Correction %d/%d] %d errors remain — running correction agent.",
            retry_num + 1, config.max_correction_retries, len(prog_errors),
        )

        try:
            correction_agent = _build_correction_agent(config)
            ws_json = world_state.model_dump_json(indent=2)
            max_chars = 80_000
            if len(ws_json) > max_chars:
                ws_json = ws_json[:max_chars] + "\n... [TRUNCATED]"

            error_summary = "\n".join(
                f"  [{e.category}] {e.detail}" for e in prog_errors
            )
            correction_msg = (
                f"The following {len(prog_errors)} error(s) were found in this WorldStateV1. "
                f"Fix them and return the corrected WorldStateV1:\n\n"
                f"ERRORS:\n{error_summary}\n\n"
                f"WORLD STATE:\n{ws_json}"
            )

            correction_result = correction_agent.run_sync(correction_msg)
            world_state = correction_result.output
            logger.info("[Pipeline·Async·Correction %d] Correction applied.", retry_num + 1)

            world_state = _normalize_fabula_times(world_state, config.fabula_time_spacing)
            world_state, new_repairs = _auto_repair(world_state)
            if new_repairs:
                repairs.extend(new_repairs)
            report = validate_world_state(world_state, config)

        except Exception:
            logger.exception(
                "[Pipeline·Async·Correction %d] Correction agent FAILED — keeping previous state.",
                retry_num + 1,
            )
            break

    return world_state, report
