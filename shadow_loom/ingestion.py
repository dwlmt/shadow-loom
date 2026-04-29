"""
Text-to-WorldState Register-Hybrid Extraction Pipeline.

Five-step LLM extraction using PydanticAI + Ollama:
  Step 1 — Global Coreference Pre-Pass (three separate passes → GlobalRegister)
    Step 1a — Location extraction (LOC_ nodes)
    Step 1b — Object extraction (OBJ_ nodes, with location context)
    Step 1c — Entity extraction (ENT_ nodes, with location + object context)
  Step 2 — Semantic Scaffolding (Socratic QA per chunk)
  Step 3 — Decomposed Topology Extraction (Physics + Social + Consequences agents per chunk)
  Step 4 — Pydantic Propose-Critique-Repair (per-chunk result validation)
  Step 5 — Global Assembly + Mathematical Sorting + Validation + Correction
  Step 5b — Post-Assembly World Trait Timeline Extraction (single focused LLM pass)

All LLM system prompts are loaded from external markdown files
in the ``prompts/`` directory inside the package.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext

from shadow_loom.settings import get_settings as _get_settings, resolve_model as _resolve_model

from shadow_loom.models import (
    Belief,
    CausalEdge,
    Entity,
    EntityStateSnapshot,
    EventNode,
    GlobalTrait,
    InformationEdge,
    Location,
    NarrativeObject,
    RelationshipEdge,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
    WorldTraitSnapshot,
)
from shadow_loom._agent_logging import log_agent_output

logger = logging.getLogger(__name__)

# =====================================================================
# Prompts directory — lives inside the package
# =====================================================================
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Read a markdown prompt file from the ``prompts/`` directory."""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


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
    world_traits: Dict[str, GlobalTrait] = Field(
        default_factory=dict,
        description="World-level facts, laws, and conditions keyed by WORLD_ IDs.",
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


class WorldTraitsRegister(BaseModel):
    """Step 1d output: world-level facts, laws, and conditions."""
    world_traits: Dict[str, GlobalTrait] = Field(
        description="All world-level traits keyed by WORLD_ IDs (e.g. WORLD_SURVEILLANCE_STATE).",
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


class ConsequencesExtraction(BaseModel):
    """Step 3c output: per-entity state deltas from the Consequences Agent.

    Run *after* the Physics Agent so it can ground each EntityUpdate in the
    actual events and mutation edges produced by Physics. Decoupling this
    from the Physics pass lets the LLM concentrate fully on the
    trait/belief/status accounting that previously had to share attention
    with event/edge extraction.
    """
    entity_updates: List["EntityUpdate"] = Field(default_factory=list)


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
ConsequencesExtraction.model_rebuild()


class WorldTraitTimelineExtraction(BaseModel):
    """Post-assembly Step 5 output: inflection points for world traits."""
    timelines: Dict[str, List[WorldTraitSnapshot]] = Field(
        default_factory=dict,
        description=(
            "Mapping of WORLD_ ID → list of WorldTraitSnapshot entries. "
            "Each snapshot marks a moment where the world trait fundamentally "
            "changed (e.g., a war ends, a law is repealed, a regime falls)."
        ),
    )


# =====================================================================
# Canonical vocabularies — kept in sync with downstream physics engine
# (``shadow_loom.causal_physics.MECHANISM_TRAIT_MAP`` + the
# ``RelationshipEdge`` schema). Used by per-chunk output validators to
# auto-correct non-fatal LLM drift without paying a full ``ModelRetry``.
# =====================================================================
_CANONICAL_MECHANISMS: set[str] = {
    "physical", "physical_force",
    "psychological",
    "epistemic", "epistemic_revelation",
    "social", "social_coercion",
    "emotional",
    "informational",
    "betrayal",
}
# Custom labels we deliberately tolerate — they bypass the routing
# fallback (so they receive full impulse to all traits) and are common
# domain words the LLM legitimately reaches for.
_TOLERATED_MECHANISMS: set[str] = {
    "kinetic", "chemical", "seduction", "coercion", "deduction",
    "manipulation", "intimidation", "persuasion", "supernatural",
    "ritual", "biological", "environmental",
}
_RELATIONSHIP_METRICS: set[str] = {"affinity", "fear", "power_dynamic"}
_VALID_STATUSES: set[str] = {"healthy", "injured", "ill", "dead", "unconscious"}


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


def _ext_defaults() -> dict:
    return _get_settings().extraction_config()


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
        default=5,
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
    enable_consequences_agent: bool = Field(
        default=True,
        description="If true, run a third per-chunk agent (Step 3c) that focuses "
        "exclusively on producing EntityUpdate records (trait/belief/status/location "
        "deltas) from the events Physics produced. When enabled, the Physics agent's "
        "own entity_updates output is discarded in favour of the Consequences output. "
        "Disable to fall back to the legacy two-agent (Physics + Social) split.",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_from_settings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k, v in _ext_defaults().items():
                data.setdefault(k, v)
        return data


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


# --- Step 1d: World Traits extraction (no dependencies) ---

def _build_world_traits_agent(config: ExtractionConfig) -> Agent[None, WorldTraitsRegister]:
    """Construct the Step 1d World Traits Agent."""
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldTraitsRegister),
        system_prompt=_load_prompt("ontology_world_traits.md"),
        retries=config.output_retries,
    )


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
    log_agent_output(logger, "LocationOntology", loc_register)
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
    log_agent_output(logger, "ObjectOntology", obj_register)
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
    log_agent_output(logger, "EntityOntology", ent_register)
    logger.info("[Step 1c] Extracted %d entities.", len(ent_register.entities))

    # --- Step 1d: World Traits (no dependencies) ---
    world_traits_agent = _build_world_traits_agent(config)
    logger.info("[Step 1d] Extracting world traits with %s …", config.model)
    try:
        wt_result = world_traits_agent.run_sync(text)
        wt_register = wt_result.output
    except Exception:
        logger.exception("[Step 1d] World traits extraction failed — retrying once …")
        wt_result = world_traits_agent.run_sync(text)
        wt_register = wt_result.output
    log_agent_output(logger, "WorldTraitsOntology", wt_register)
    logger.info("[Step 1d] Extracted %d world traits.", len(wt_register.world_traits))

    # --- Resolve object owner_ids to ENT_ IDs ---
    resolved_objects = _resolve_object_owner_ids(obj_register.objects, ent_register.entities)

    # --- Merge into GlobalRegister ---
    register = GlobalRegister(
        locations=loc_register.locations,
        objects=resolved_objects,
        entities=ent_register.entities,
        world_traits=wt_register.world_traits,
    )
    logger.info(
        "[Step 1] Ontology extracted — %d locations, %d objects, %d entities, %d world traits.",
        len(register.locations), len(register.objects), len(register.entities),
        len(register.world_traits),
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

    async def _extract_world_traits() -> WorldTraitsRegister:
        world_traits_agent = _build_world_traits_agent(config)
        logger.info("[Step 1d] Extracting world traits with %s …", config.model)
        try:
            wt_result = await world_traits_agent.run(text)
            return wt_result.output
        except Exception:
            logger.exception("[Step 1d] World traits extraction failed — retrying once …")
            wt_result = await world_traits_agent.run(text)
            return wt_result.output

    obj_register, ent_register, wt_register = await asyncio.gather(
        _extract_objects(), _extract_entities(), _extract_world_traits()
    )
    logger.info("[Step 1b] Extracted %d objects.", len(obj_register.objects))
    logger.info("[Step 1c] Extracted %d entities.", len(ent_register.entities))
    logger.info("[Step 1d] Extracted %d world traits.", len(wt_register.world_traits))

    # --- Resolve object owner_ids to ENT_ IDs (needs both registers) ---
    resolved_objects = _resolve_object_owner_ids(obj_register.objects, ent_register.entities)

    register = GlobalRegister(
        locations=loc_register.locations,
        objects=resolved_objects,
        entities=ent_register.entities,
        world_traits=wt_register.world_traits,
    )
    logger.info(
        "[Step 1] Ontology extracted — %d locations, %d objects, %d entities, %d world traits.",
        len(register.locations), len(register.objects), len(register.entities),
        len(register.world_traits),
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


class _ConsequencesDeps(BaseModel):
    """Dependencies for Step 3c (Consequences Agent: entity_updates).

    Receives the events and mutation/mutation_social edges already produced
    by the Physics Agent so that every EntityUpdate it emits is anchored
    to a concrete event and aligned with the causal mutations.
    """
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    scaffold: SocraticScaffold
    chunk_events: List[EventNode] = Field(default_factory=list)
    chunk_causal: List[CausalEdge] = Field(default_factory=list)
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
        | set(reg.world_traits.keys())
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
    for p in ("EVT_", "ENT_", "LOC_", "OBJ_", "WORLD_"):
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


# =====================================================================
# Numeric / semantic sanitisation helpers — used by every per-chunk
# output validator. Each helper returns the corrected value plus,
# where relevant, an "issue" string appended to a shared *notes* list
# so the caller can decide whether to log, drop, or pass through.
# =====================================================================


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive range [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _sanitize_causal_edge(
    ce: CausalEdge, notes: List[str],
) -> Optional[CausalEdge]:
    """Apply numeric clamps and semantic fixes to a CausalEdge.

    Returns the (possibly mutated) edge or ``None`` if the edge is
    structurally incoherent enough that downstream physics would just
    silently drop it (in which case we drop it now and log).
    """
    updates: dict = {}

    # Drop self-loops — physics treats these as zero-effect cycles
    # and the AMWN structural diagram throws them away anyway.
    if ce.source_id == ce.target_id:
        notes.append(
            f"[Auto-Fix] Dropped self-loop CausalEdge '{ce.source_id}'→"
            f"'{ce.target_id}' ({ce.causality_type})"
        )
        return None

    # Clamp causal_force to [0, 10]
    if ce.causal_force is not None and not (0.0 <= ce.causal_force <= 10.0):
        clamped = _clamp(ce.causal_force, 0.0, 10.0)
        notes.append(
            f"[Auto-Fix] Clamped CausalEdge causal_force {ce.causal_force} → {clamped} "
            f"({ce.source_id}→{ce.target_id})"
        )
        updates["causal_force"] = clamped

    # Clamp trait_delta to [-1, 1]
    if ce.trait_delta is not None and not (-1.0 <= ce.trait_delta <= 1.0):
        clamped = _clamp(ce.trait_delta, -1.0, 1.0)
        notes.append(
            f"[Auto-Fix] Clamped CausalEdge trait_delta {ce.trait_delta} → {clamped} "
            f"({ce.source_id}→{ce.target_id})"
        )
        updates["trait_delta"] = clamped

    # Clamp propagation_delay to >= 0
    if ce.propagation_delay is not None and ce.propagation_delay < 0:
        notes.append(
            f"[Auto-Fix] Clamped negative propagation_delay {ce.propagation_delay} → 0 "
            f"({ce.source_id}→{ce.target_id})"
        )
        updates["propagation_delay"] = 0

    # Mutation edges should carry trait_target + trait_delta. The downstream
    # physics engine falls back to a generic trait-routing path when these
    # are missing, which silently degrades fidelity. Log so the issue is
    # visible without rejecting the edge entirely.
    if ce.causality_type in ("mutation", "mutation_social"):
        if not ce.trait_target:
            notes.append(
                f"[Quality] {ce.causality_type} edge {ce.source_id}→{ce.target_id} "
                "missing trait_target — physics will use generic routing."
            )
        if ce.trait_delta is None:
            notes.append(
                f"[Quality] {ce.causality_type} edge {ce.source_id}→{ce.target_id} "
                "missing trait_delta — physics will use a default (+1.0) magnitude."
            )

    # mutation_social: trait_target must be a relationship metric, and
    # rel_counterpart_id must differ from target_id. The schema-level
    # validator already requires rel_counterpart_id; we strengthen here.
    if ce.causality_type == "mutation_social":
        if ce.trait_target and ce.trait_target not in _RELATIONSHIP_METRICS:
            notes.append(
                f"[Auto-Fix] Dropped mutation_social edge {ce.source_id}→{ce.target_id}: "
                f"trait_target '{ce.trait_target}' is not one of "
                f"{sorted(_RELATIONSHIP_METRICS)}."
            )
            return None
        if ce.rel_counterpart_id and ce.rel_counterpart_id == ce.target_id:
            notes.append(
                f"[Auto-Fix] Dropped mutation_social edge {ce.source_id}→{ce.target_id}: "
                "rel_counterpart_id is the same as target_id (self-relationship)."
            )
            return None

    # Mechanism vocabulary check — fully informational. The physics engine
    # routes by mechanism via MECHANISM_TRAIT_MAP and falls back to a
    # penalty for unknown labels. We surface unknown-but-acceptable labels
    # so authors can spot persistent novel labels they may want canonised.
    if ce.mechanism:
        mech = ce.mechanism.strip()
        if mech != ce.mechanism:
            updates["mechanism"] = mech
        if (
            mech not in _CANONICAL_MECHANISMS
            and mech not in _TOLERATED_MECHANISMS
        ):
            notes.append(
                f"[Quality] CausalEdge mechanism '{mech}' is non-canonical "
                f"({ce.source_id}→{ce.target_id}) — physics will apply the "
                "mechanism-routing fallback (~20% impulse) for off-list traits."
            )

    if updates:
        try:
            return ce.model_copy(update=updates)
        except Exception:
            notes.append(
                f"[Auto-Fix] Dropped CausalEdge {ce.source_id}→{ce.target_id} "
                "after numeric clamp triggered a schema rejection."
            )
            return None
    return ce


def _sanitize_relationship_edge(
    re_edge: RelationshipEdge, notes: List[str],
) -> RelationshipEdge:
    """Clamp the metric and inertia ranges on a RelationshipEdge."""
    updates: dict = {}
    for field, lo, hi in (
        ("affinity", -1.0, 1.0),
        ("fear", 0.0, 1.0),
        ("power_dynamic", -1.0, 1.0),
        ("inertia", 0.0, 1.0),
    ):
        cur = getattr(re_edge, field)
        if cur is None:
            continue
        if not (lo <= cur <= hi):
            clamped = _clamp(cur, lo, hi)
            notes.append(
                f"[Auto-Fix] Clamped RelationshipEdge.{field} {cur} → {clamped} "
                f"({re_edge.source_entity_id}→{re_edge.target_entity_id})"
            )
            updates[field] = clamped
    return re_edge.model_copy(update=updates) if updates else re_edge


def _sanitize_entity_update(
    eu: "EntityUpdate", notes: List[str],
) -> Optional["EntityUpdate"]:
    """Clamp trait/inertia/confidence ranges and validate status enum.

    Drops the update entirely only when *every* field is no-op after
    sanitisation (the LLM produced a hollow record).
    """
    updates: dict = {}

    # status enum guard — schema enforces it, but LLMs sometimes return
    # title-case or synonyms. Try to coerce common aliases before giving up.
    if eu.new_status and eu.new_status not in _VALID_STATUSES:
        coerced = eu.new_status.lower().strip()
        alias = {
            "alive": "healthy", "well": "healthy",
            "wounded": "injured", "hurt": "injured",
            "sick": "ill", "diseased": "ill",
            "deceased": "dead", "killed": "dead",
            "ko": "unconscious", "knocked_out": "unconscious",
            "asleep": "unconscious",
        }.get(coerced)
        if alias:
            updates["new_status"] = alias
            notes.append(
                f"[Auto-Fix] EntityUpdate.new_status '{eu.new_status}' → '{alias}' "
                f"({eu.entity_id}@{eu.fabula_time})"
            )
        else:
            updates["new_status"] = None
            notes.append(
                f"[Auto-Fix] EntityUpdate.new_status '{eu.new_status}' is unknown — "
                f"dropped ({eu.entity_id}@{eu.fabula_time})."
            )

    # Clamp trait values + inertia to [0, 1]; physics asserts these.
    if eu.trait_updates:
        cleaned: Dict[str, TraitVector] = {}
        for tname, tv in eu.trait_updates.items():
            new_value = _clamp(tv.value, 0.0, 1.0)
            new_inertia = _clamp(tv.inertia, 0.0, 1.0)
            # Avoid 1.0 inertia — it makes the trait literally unmovable.
            if new_inertia >= 1.0:
                new_inertia = 0.99
                notes.append(
                    f"[Auto-Fix] Capped EntityUpdate trait '{tname}' inertia at 0.99 "
                    f"({eu.entity_id}@{eu.fabula_time}) — 1.0 would freeze it forever."
                )
            if new_value != tv.value or new_inertia != tv.inertia:
                notes.append(
                    f"[Auto-Fix] Clamped EntityUpdate trait '{tname}' "
                    f"value/inertia {tv.value:.2f}/{tv.inertia:.2f} → "
                    f"{new_value:.2f}/{new_inertia:.2f} "
                    f"({eu.entity_id}@{eu.fabula_time})"
                )
            cleaned[tname] = TraitVector(value=new_value, inertia=new_inertia)
        if cleaned != eu.trait_updates:
            updates["trait_updates"] = cleaned

    # Clamp belief confidence + inertia, ensure established_at_fabula is set.
    if eu.new_beliefs:
        cleaned_beliefs: List[Belief] = []
        for b in eu.new_beliefs:
            b_updates: dict = {}
            new_conf = _clamp(b.confidence, 0.0, 1.0)
            new_in = _clamp(b.inertia, 0.0, 1.0)
            if new_in >= 1.0:
                new_in = 0.99
            if new_conf != b.confidence:
                b_updates["confidence"] = new_conf
            if new_in != b.inertia:
                b_updates["inertia"] = new_in
            # Default missing established_at_fabula to the EntityUpdate's
            # own fabula_time so downstream time-slicing works.
            if not b.established_at_fabula and eu.fabula_time > 0:
                b_updates["established_at_fabula"] = eu.fabula_time
            cleaned_beliefs.append(b.model_copy(update=b_updates) if b_updates else b)
        updates["new_beliefs"] = cleaned_beliefs

    # Drop entirely-empty updates: no traits, no beliefs, no
    # invalidations, no status, no location.
    candidate = eu.model_copy(update=updates) if updates else eu
    if (
        not candidate.trait_updates
        and not candidate.new_beliefs
        and not candidate.invalidated_belief_targets
        and candidate.new_status is None
        and candidate.new_location_id is None
    ):
        notes.append(
            f"[Auto-Fix] Dropped empty EntityUpdate for {eu.entity_id}@{eu.fabula_time} "
            "(no trait/belief/status/location change)."
        )
        return None
    return candidate


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
            f"WORLD TRAIT IDs: {list(ctx.deps.global_register.world_traits.keys())}\n"
            f"PREVIOUSLY EXTRACTED EVENT IDs: {ctx.deps.previous_event_ids}\n"
            "\n"
            "You MUST ONLY use ENT_, LOC_, OBJ_, WORLD_ IDs from the lists above.\n"
            "You MAY create new EVT_ IDs for events discovered in this chunk.\n"
            "Do NOT invent new ENT_, LOC_, OBJ_, or WORLD_ IDs.\n"
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
                fixed = ce.model_copy(update=updates) if updates else ce
            except Exception:
                # model_validator rejected the fix (prefix mismatch) — drop edge
                fixes.append(f"[Auto-Fix] Dropped causal edge {ce.source_id}→{ce.target_id} (validation error after fix)")
                continue
            sanitised = _sanitize_causal_edge(fixed, fixes)
            if sanitised is not None:
                fixed_causal.append(sanitised)

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
            elif src == tgt:
                fixes.append(
                    f"[Auto-Fix] Dropped self-loop SpatialEdge '{src}'→'{tgt}'"
                )
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
                cleaned_eu = eu.model_copy(update=updates) if updates else eu
                sanitised_eu = _sanitize_entity_update(cleaned_eu, fixes)
                if sanitised_eu is not None:
                    fixed_updates.append(sanitised_eu)

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
                    if t == src:
                        # self-broadcast: information cannot flow to itself
                        fixes.append(
                            f"[Auto-Fix] Dropped self-targeting InformationEdge target "
                            f"'{src}' on edge from '{src}'."
                        )
                        continue
                    fixed_targets.append(t)
                else:
                    bad.append(f"InformationEdge target_id '{tid}' is not a valid entity/object.")
            if not fixed_targets:
                fixes.append(
                    f"[Auto-Fix] Dropped InformationEdge from '{src}' — no valid targets remain."
                )
                continue
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
                rebuilt = re_edge.model_copy(update=updates) if updates else re_edge
                fixed_social.append(_sanitize_relationship_edge(rebuilt, fixes))

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


def _build_consequences_agent(
    config: ExtractionConfig,
) -> Agent[_ConsequencesDeps, ConsequencesExtraction]:
    """Construct the Step 3c Consequences Agent — entity_updates only.

    The agent is given the events and mutation/mutation_social edges
    already produced by the Physics Agent, plus the entity baselines
    from the Global Register. Its sole job is to translate those into
    EntityUpdate records (trait deltas, new/invalidated beliefs, status
    changes, location changes).
    """
    agent: Agent[_ConsequencesDeps, ConsequencesExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_ConsequencesDeps,
        output_type=NativeOutput(ConsequencesExtraction),
        system_prompt=_load_prompt("consequences_extraction.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_register_for_consequences(ctx: RunContext[_ConsequencesDeps]) -> str:
        reg = ctx.deps.global_register
        entity_ids = sorted(reg.entities.keys())
        location_ids = sorted(reg.locations.keys())
        object_ids = sorted(reg.objects.keys())
        entity_names = {eid: reg.entities[eid].name for eid in entity_ids}
        scaffold_text = _format_scaffold(ctx.deps.scaffold)

        # Compact entity baselines so the LLM knows starting trait values.
        entity_baselines: List[str] = []
        for eid in entity_ids:
            ent = reg.entities[eid]
            traits_str = ", ".join(
                f"{k}={v.value:.1f}/i={v.inertia:.2f}" for k, v in ent.traits.items()
            )
            entity_baselines.append(
                f"  {eid} ({ent.name}): status={ent.status}, "
                f"loc={ent.location_id}, traits=[{traits_str}]"
            )
        baselines_block = "\n".join(entity_baselines)

        # Compact event summary (the only events whose consequences matter).
        evt_lines: List[str] = []
        for e in ctx.deps.chunk_events:
            evt_lines.append(
                f"  - {e.id} (fabula={e.fabula_time}, type={e.event_type}, "
                f"actors={e.actor_ids}, targets={e.target_ids}): {e.description}"
            )
        events_block = "\n".join(evt_lines) if evt_lines else "  (no events in this chunk)"

        # Compact mutation hints from Physics — every mutation/mutation_social
        # edge implies an EntityUpdate is needed.
        mut_lines: List[str] = []
        for ce in ctx.deps.chunk_causal:
            if ce.causality_type in ("mutation", "mutation_social"):
                mut_lines.append(
                    f"  - {ce.source_id} → {ce.target_id} "
                    f"[{ce.causality_type}] trait={ce.trait_target} "
                    f"delta={ce.trait_delta} (force={ce.causal_force}, "
                    f"evidence={ce.evidence_strength})"
                )
        mutations_block = (
            "\n".join(mut_lines)
            if mut_lines
            else "  (no mutation edges — infer trait/belief deltas from events directly)"
        )

        chunk_evt_ids = [e.id for e in ctx.deps.chunk_events]

        return (
            "=== VALID ID REGISTER (from Step 1) ===\n"
            f"ENTITY IDs: {entity_ids}\n"
            f"ENTITY NAMES: {entity_names}\n"
            f"LOCATION IDs: {location_ids}\n"
            f"OBJECT IDs: {object_ids}\n"
            f"WORLD TRAIT IDs: {list(reg.world_traits.keys())}\n"
            f"THIS CHUNK'S EVENT IDs: {chunk_evt_ids}\n"
            f"PREVIOUS CHUNKS' EVENT IDs: {ctx.deps.previous_event_ids}\n"
            "\n"
            "You MUST ONLY use IDs from the lists above. Do NOT invent any IDs.\n"
            "\n"
            "=== ENTITY BASELINES (initial trait values + inertia) ===\n"
            f"{baselines_block}\n"
            "\n"
            "Trait_updates values are the NEW absolute trait value after the "
            "event, NOT the delta. The engine computes deltas from baselines.\n"
            "\n"
            "=== EVENTS EXTRACTED FROM THIS CHUNK (by Physics Agent) ===\n"
            f"{events_block}\n"
            "\n"
            "=== MUTATION EDGES FROM PHYSICS (each implies an EntityUpdate) ===\n"
            f"{mutations_block}\n"
            "\n"
            "=== SOCRATIC SCAFFOLD (semantic pre-analysis) ===\n"
            f"{scaffold_text}\n"
            "\n"
            "Use the scaffold's WHY/HOW answers to surface IMPLICIT trait "
            "shifts (guilt after killing, fear after threat, grief after loss) "
            "even when the prose does not name them."
        )

    @agent.output_validator
    def validate_consequences_ids(
        ctx: RunContext[_ConsequencesDeps],
        result: ConsequencesExtraction,
    ) -> ConsequencesExtraction:
        """Validate entity_update IDs — fix typos, retry only for unfixable IDs."""
        reg = ctx.deps.global_register
        chunk_evt_ids = [e.id for e in ctx.deps.chunk_events]
        valid = _build_valid_id_set(
            reg, ctx.deps.previous_event_ids + chunk_evt_ids,
        )
        entity_ids = set(reg.entities.keys())
        location_ids = set(reg.locations.keys())
        fixes: List[str] = []
        bad: List[str] = []

        fixed_updates: List[EntityUpdate] = []
        for eu in result.entity_updates:
            updates: dict = {}
            eid, _ = _fix_id(eu.entity_id, entity_ids, "EntityUpdate.entity_id", fixes)
            if eid != eu.entity_id:
                updates["entity_id"] = eid
            if eu.triggered_by and eu.triggered_by not in valid:
                trig, _ = _fix_id(eu.triggered_by, valid, "EntityUpdate.triggered_by", fixes)
                if trig != eu.triggered_by:
                    updates["triggered_by"] = trig
                if trig not in valid:
                    bad.append(
                        f"EntityUpdate triggered_by '{eu.triggered_by}' is not a valid event."
                    )
            if eu.new_location_id and eu.new_location_id not in location_ids:
                loc, _ = _fix_id(eu.new_location_id, location_ids, "EntityUpdate.new_location_id", fixes)
                if loc != eu.new_location_id:
                    updates["new_location_id"] = loc
                if loc not in location_ids:
                    bad.append(
                        f"EntityUpdate new_location_id '{eu.new_location_id}' is not a valid location."
                    )
            if eid not in entity_ids:
                bad.append(f"EntityUpdate entity_id '{eu.entity_id}' is not a valid entity.")
            else:
                cleaned_eu = eu.model_copy(update=updates) if updates else eu
                sanitised_eu = _sanitize_entity_update(cleaned_eu, fixes)
                if sanitised_eu is not None:
                    fixed_updates.append(sanitised_eu)

        # --- Mutation⇄EntityUpdate parity audit ---
        # Every Physics mutation/mutation_social edge that targets an entity
        # SHOULD have a matching EntityUpdate. Surface gaps so they are
        # visible in the log; do not retry (the agent already had every
        # mutation listed in its system prompt).
        eu_keys: set[Tuple[str, int, Optional[str]]] = {
            (eu.entity_id, eu.fabula_time, eu.triggered_by) for eu in fixed_updates
        }
        eu_entity_set: set[str] = {eu.entity_id for eu in fixed_updates}
        missing: List[str] = []
        for ce in ctx.deps.chunk_causal:
            if ce.causality_type not in ("mutation", "mutation_social"):
                continue
            target_ent = (
                ce.target_id if ce.causality_type == "mutation"
                else ce.target_id  # mutation_social: target_id is the perspective entity
            )
            if not target_ent.startswith("ENT_"):
                continue
            # Loose match: same entity touched by something in this chunk is OK.
            # We only flag when the entity has no update at all.
            if target_ent not in eu_entity_set:
                missing.append(
                    f"{ce.source_id} → {target_ent} "
                    f"({ce.causality_type}, trait={ce.trait_target})"
                )
        if missing:
            logger.info(
                "[Validator·Consequences] %d mutation edge(s) lack a "
                "corresponding EntityUpdate: %s",
                len(missing), "; ".join(missing[:5]),
            )

        # --- Dead-actor warning ---
        # If an EntityUpdate marks an entity dead, warn when subsequent
        # events in this chunk still list that entity as an actor.
        deaths: Dict[str, int] = {
            eu.entity_id: eu.fabula_time
            for eu in fixed_updates
            if eu.new_status == "dead"
        }
        for ev in ctx.deps.chunk_events:
            for actor in ev.actor_ids:
                if actor in deaths and ev.fabula_time > deaths[actor]:
                    logger.info(
                        "[Validator·Consequences] %s is marked dead at "
                        "fabula=%d but still acts in event %s at fabula=%d.",
                        actor, deaths[actor], ev.id, ev.fabula_time,
                    )

        if fixes:
            logger.info(
                "[Validator·Consequences] Auto-fixed %d issue(s): %s",
                len(fixes), "; ".join(fixes[:8]),
            )
        if bad:
            raise ModelRetry(
                "The following IDs could not be auto-resolved. "
                "Fix them using ONLY IDs from the register:\n" + "\n".join(bad)
            )

        return ConsequencesExtraction(entity_updates=fixed_updates)

    return agent


def extract_topology(
    chunks: List[str],
    register: GlobalRegister,
    config: ExtractionConfig | None = None,
) -> List[ChunkTopology]:
    """
    Steps 2–4: Per-chunk agent pipeline with Socratic scaffolding.

    **Step 2** — Socratic QA scaffolding: lightweight agent generates
    Who/What/Where/When/Why/How pairs to articulate hidden reasoning.

    **Step 3a** — Physics Agent: extracts events + causal + spatial edges,
    informed by the scaffold. Result validator catches hallucinated IDs.

    **Step 3b** — Social Agent: extracts information + relationship edges,
    informed by the scaffold + concrete events from 3a. Result validator
    catches hallucinated IDs.

    **Step 3c** — Consequences Agent (enabled by default): translates
    events + mutation edges from 3a into per-entity ``EntityUpdate``
    records (trait deltas, new/invalidated beliefs, status, location).
    When enabled, replaces the ``entity_updates`` Physics produced.
    Toggle via ``ExtractionConfig.enable_consequences_agent``.

    The GlobalRegister (from Step 1) is injected into every agent via
    dependency injection, preventing hallucination of new entity/location/
    object IDs.
    """
    config = config or ExtractionConfig()
    socratic_agent = _build_socratic_agent(config)
    physics_agent = _build_physics_agent(config)
    social_agent = _build_social_agent(config)
    consequences_agent = (
        _build_consequences_agent(config)
        if config.enable_consequences_agent else None
    )
    topologies: List[ChunkTopology] = []
    syuzhet_counter = 0
    # Informational only — the LLM is told the highest fabula_time it has
    # produced so far so it can place continuation events after it. It is
    # NOT used to shift the LLM's output: chunks are free to use earlier
    # fabula_time values to encode flashbacks, prologues, or interstitial
    # events. (Syuzhet position ≠ fabula position by design.)
    prev_max_fabula = 0
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
            log_agent_output(logger, f"Socratic[chunk={i + 1}]", scaffold)
        except Exception:
            logger.exception("[Step 2] Chunk %d scaffolding FAILED — using empty scaffold.", i + 1)
            scaffold = SocraticScaffold()

        logger.info("[Step 2] Chunk %d: %d QA pairs generated.", i + 1, len(scaffold.qa_pairs))

        # --- Step 3a: Physics Agent (events + causal + spatial) ---
        logger.info("[Step 3a] Processing chunk %d/%d — physics …", i + 1, len(chunks))
        physics_msg = (
            f"Chunk {i + 1} of {len(chunks)} "
            f"(syuzhet_index offset: {syuzhet_counter}, "
            f"fabula_time_spacing: {config.fabula_time_spacing}, "
            f"max fabula_time so far: {prev_max_fabula}).\n\n"
            f"Use ABSOLUTE story-world chronology for fabula_time. Most "
            f"continuation events will follow the previous max, but the "
            f"narration MAY jump in either direction: flashbacks / "
            f"prologues / pre-story events use SMALLER fabula_time, and "
            f"flash-forwards / prophecies / glimpses of the future use "
            f"LARGER fabula_time than surrounding chunks. Chunk position "
            f"in the syuzhet does NOT determine fabula order.\n\n"
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
            log_agent_output(logger, f"PhysicsExtraction[chunk={i + 1}]", physics)
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
                log_agent_output(logger, f"PhysicsExtraction[chunk={i + 1},retry]", physics)
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
                log_agent_output(logger, f"SocialExtraction[chunk={i + 1}]", social)
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
                    log_agent_output(logger, f"SocialExtraction[chunk={i + 1},retry]", retry_social)
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

        # --- Step 3c: Consequences Agent (entity_updates) ---
        entity_updates_final = physics.entity_updates
        if consequences_agent is not None and physics.events:
            logger.info("[Step 3c] Processing chunk %d/%d — consequences …", i + 1, len(chunks))
            consequences_msg = (
                f"Chunk {i + 1} of {len(chunks)}.\n\n"
                f"EVENTS EXTRACTED FROM THIS CHUNK:\n{event_summary}\n\n"
                f"ORIGINAL TEXT:\n{chunk}"
            )
            consequences_deps = _ConsequencesDeps(
                global_register=register,
                scaffold=scaffold,
                chunk_events=physics.events,
                chunk_causal=physics.causal_topology,
                previous_event_ids=all_event_ids.copy(),
            )
            try:
                consequences_result = consequences_agent.run_sync(
                    consequences_msg, deps=consequences_deps,
                )
                consequences = consequences_result.output
                log_agent_output(
                    logger, f"ConsequencesExtraction[chunk={i + 1}]", consequences,
                )
                entity_updates_final = consequences.entity_updates
            except Exception:
                logger.exception(
                    "[Step 3c] Chunk %d FAILED — falling back to physics.entity_updates.",
                    i + 1,
                )

        # Merge into ChunkTopology
        topo = ChunkTopology(
            events=physics.events,
            causal_topology=physics.causal_topology,
            information_topology=social.information_topology,
            social_topology=social.social_topology,
            spatial_topology=physics.spatial_topology,
            entity_updates=entity_updates_final,
        )
        topologies.append(topo)

        # Accumulate counters. ``prev_max_fabula`` is purely informational —
        # it tells the next chunk's prompt what the high-water mark is so
        # the LLM can place forward-marching events sensibly. We do NOT
        # shift the LLM's output, so flashbacks remain expressible.
        syuzhet_counter += len(physics.events)
        if physics.events:
            chunk_max = max(e.fabula_time for e in physics.events)
            if chunk_max > prev_max_fabula:
                prev_max_fabula = chunk_max
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
    """Pre-allocated parameters for a single chunk in parallel extraction.

    ``syuzhet_offset`` is deterministic from chunk position because syuzhet
    *is* narration order. ``fabula_time`` is intentionally NOT pre-allocated:
    chunks must be free to encode flashbacks, prologues, and other
    non-monotone story-world chronologies (see
    `docs/academic-foundations.md` §1.1).
    """
    chunk_index: int
    total_chunks: int
    syuzhet_offset: int
    prev_chunk_tail: str


def _pre_allocate_chunk_params(
    chunks: List[str],
    config: ExtractionConfig,
) -> List[_ChunkParams]:
    """Pre-compute per-chunk extraction parameters for parallel dispatch.

    ``syuzhet_offset`` is allocated deterministically from chunk position
    because syuzhet position equals narration position. ``fabula_time``
    is intentionally NOT pre-allocated: forcing chunk order onto fabula
    order would make flashbacks structurally impossible. Chunks are
    expected to use absolute story-world chronology, and the global
    ``run_extraction_async`` pipeline relies on the validator to flag
    any temporal contradictions.
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
    consequences_agent: Optional[Agent] = None,
) -> ChunkTopology:
    """Process one chunk through the per-chunk agent pipeline (async).

    Runs Socratic scaffolding → Physics → (Social ∥ Consequences) for a
    single chunk. Social and Consequences are dispatched concurrently
    via ``asyncio.gather`` since both depend only on the Physics output.
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
        f"fabula_time_spacing: {config.fabula_time_spacing}).\n\n"
        f"Use ABSOLUTE story-world chronology for fabula_time. Earlier "
        f"story-time = smaller fabula_time, later story-time = larger. "
        f"The narration MAY jump in either direction: flashbacks / "
        f"prologues use SMALLER fabula_time than surrounding chunks; "
        f"flash-forwards / prophecies / glimpses of the future use "
        f"LARGER fabula_time. Chunk position in the syuzhet does NOT "
        f"determine fabula order.\n\n"
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
    # --- Step 3c: Consequences Agent (optional, parallel with Social) ---
    # Both agents depend only on the Physics output, so we dispatch them
    # concurrently to halve the wall-clock cost of the consequences pass.
    social = SocialExtraction()
    entity_updates_final = physics.entity_updates  # legacy fallback

    async def _run_social() -> SocialExtraction:
        if not physics.events:
            logger.info(
                "[Step 3b·Async] Chunk %d: skipping social pass (no events).", i + 1,
            )
            return SocialExtraction()
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
            previous_event_ids=[],
        )
        try:
            social_result = await social_agent.run(social_msg, deps=social_deps)
            local_social = social_result.output
        except Exception:
            logger.exception(
                "[Step 3b·Async] Chunk %d FAILED — returning empty social.", i + 1,
            )
            return SocialExtraction()

        # Retry if zero info edges with multiple events (quality gate)
        if len(physics.events) >= 2 and not local_social.information_topology:
            logger.info(
                "[Step 3b·Async] Chunk %d: 0 info edges — retrying with emphasis …", i + 1,
            )
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
                    local_social = SocialExtraction(
                        information_topology=retry_social.information_topology,
                        social_topology=local_social.social_topology,
                    )
                    logger.info(
                        "[Step 3b·Async] Chunk %d: retry recovered %d info edges.",
                        i + 1, len(local_social.information_topology),
                    )
            except Exception:
                logger.exception("[Step 3b·Async] Chunk %d info retry FAILED.", i + 1)
        return local_social

    async def _run_consequences() -> Optional[ConsequencesExtraction]:
        if consequences_agent is None or not physics.events:
            return None
        logger.info("[Step 3c·Async] Processing chunk %d/%d — consequences …", i + 1, n)
        consequences_msg = (
            f"Chunk {i + 1} of {n}.\n\n"
            f"EVENTS EXTRACTED FROM THIS CHUNK:\n{event_summary}\n\n"
            f"ORIGINAL TEXT:\n{chunk}"
        )
        consequences_deps = _ConsequencesDeps(
            global_register=register,
            scaffold=scaffold,
            chunk_events=physics.events,
            chunk_causal=physics.causal_topology,
            previous_event_ids=[],
        )
        try:
            consequences_result = await consequences_agent.run(
                consequences_msg, deps=consequences_deps,
            )
            return consequences_result.output
        except Exception:
            logger.exception(
                "[Step 3c·Async] Chunk %d FAILED — falling back to physics.entity_updates.",
                i + 1,
            )
            return None

    social, consequences_out = await asyncio.gather(
        _run_social(), _run_consequences(),
    )
    if consequences_out is not None:
        entity_updates_final = consequences_out.entity_updates

    topo = ChunkTopology(
        events=physics.events,
        causal_topology=physics.causal_topology,
        information_topology=social.information_topology,
        social_topology=social.social_topology,
        spatial_topology=physics.spatial_topology,
        entity_updates=entity_updates_final,
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
    2. Re-numbers ``syuzhet_index`` globally in chunk order — syuzhet IS
       narration order, so the chunk-position assignment is canonical.
       Per-chunk and global fallback maps are built so that any
       ``InformationEdge.discovered_at_syuzhet`` reference resolves
       correctly regardless of whether the LLM used a chunk-local or
       global value.

    Note: there is intentionally NO inter-chunk ``fabula_time`` shift.
    Forcing chunk order onto fabula order would erase flashbacks (per
    the fabula/syuzhet design — see `docs/academic-foundations.md`
    §1.1). Temporal contradictions inside the merged graph are caught
    later by ``_validate_time_ordering`` and the auditor.
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
    # Build a per-chunk old → new syuzhet map AND a flat fallback map keyed
    # by old syuzhet only. The flat map is used as a fallback when an
    # ``InformationEdge.discovered_at_syuzhet`` references a value that
    # isn't in its own chunk's local remap (e.g. the LLM cited a global
    # syuzhet index from another chunk).
    chunk_syuzhet_remaps: List[Dict[int, int]] = []
    flat_syuzhet_remap: Dict[int, int] = {}
    flat_syuzhet_ambiguous: set[int] = set()
    syuzhet_counter = 0
    for topo in reconciled:
        remap: Dict[int, int] = {}
        sorted_events = sorted(topo.events, key=lambda e: e.syuzhet_index)
        for evt in sorted_events:
            old = evt.syuzhet_index
            remap[old] = syuzhet_counter
            if old in flat_syuzhet_remap:
                # Same chunk-local syuzhet appeared before — ambiguous as a
                # flat fallback. Drop it.
                if flat_syuzhet_remap[old] != syuzhet_counter:
                    flat_syuzhet_ambiguous.add(old)
            else:
                flat_syuzhet_remap[old] = syuzhet_counter
            evt.syuzhet_index = syuzhet_counter
            syuzhet_counter += 1
        chunk_syuzhet_remaps.append(remap)

    # Drop ambiguous values from the flat fallback map.
    for amb in flat_syuzhet_ambiguous:
        flat_syuzhet_remap.pop(amb, None)

    # Remap discovered_at_syuzhet — chunk-local first, then unambiguous flat.
    for ci, topo in enumerate(reconciled):
        local = chunk_syuzhet_remaps[ci]
        for ie in topo.information_topology:
            old = ie.discovered_at_syuzhet
            if old in local:
                ie.discovered_at_syuzhet = local[old]
            elif old in flat_syuzhet_remap:
                ie.discovered_at_syuzhet = flat_syuzhet_remap[old]
            # else: out-of-range value — leave for the validator to flag.

    # No Pass 3: fabula_time order is intentionally free across chunks so
    # flashbacks remain expressible. Cross-chunk temporal contradictions
    # are surfaced by ``_validate_time_ordering`` downstream.

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
    """Shift all fabula_time values in a ChunkTopology by *shift* (in-place).

    Values <= 0 are treated as the "pre-story baseline" sentinel and
    left untouched, so beliefs and edges that were established before
    the narrative begins are not pushed into story-time.
    """
    for evt in topo.events:
        if evt.fabula_time > 0:
            evt.fabula_time += shift
    for ce in topo.causal_topology:
        if ce.fabula_time > 0:
            ce.fabula_time += shift
    for ie in topo.information_topology:
        if ie.established_at_fabula > 0:
            ie.established_at_fabula += shift
        if ie.terminated_at_fabula is not None and ie.terminated_at_fabula > 0:
            ie.terminated_at_fabula += shift
    for se in topo.social_topology:
        if se.last_updated_fabula > 0:
            se.last_updated_fabula += shift
    for sp in topo.spatial_topology:
        if sp.established_at_fabula > 0:
            sp.established_at_fabula += shift
        if sp.destroyed_at_fabula is not None and sp.destroyed_at_fabula > 0:
            sp.destroyed_at_fabula += shift
    for eu in topo.entity_updates:
        if eu.fabula_time > 0:
            eu.fabula_time += shift
        for belief in eu.new_beliefs:
            if belief.established_at_fabula > 0:
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
    consequences_agent = (
        _build_consequences_agent(config)
        if config.enable_consequences_agent else None
    )

    params_list = _pre_allocate_chunk_params(chunks, config)
    semaphore = asyncio.Semaphore(config.max_concurrent_chunks)

    async def _guarded_extract(chunk: str, params: _ChunkParams) -> ChunkTopology:
        async with semaphore:
            return await _extract_single_chunk_async(
                chunk, params, register, config,
                socratic_agent, physics_agent, social_agent,
                consequences_agent,
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

    Builds a monotonic mapping over the union of every fabula_time value
    found anywhere in the world-state — events, every edge type, beliefs,
    and timeline snapshots — then applies it uniformly. This guarantees
    edges and beliefs stay synchronised with their referenced events
    after rescaling.

    The 0 value is preserved as the "pre-story baseline" sentinel: it
    is never remapped, and any belief / edge with fabula_time == 0 stays
    at 0 to keep its pre-story semantics.

    Returns the original world-state unchanged when event times are
    already well-spaced (median gap ≥ spacing / 2).
    """
    if not ws.events:
        return ws

    # --- Decide whether normalisation is needed (event spacing only) ---
    event_times = sorted({e.fabula_time for e in ws.events if e.fabula_time > 0})
    if len(event_times) < 2:
        return ws
    diffs = [event_times[i + 1] - event_times[i] for i in range(len(event_times) - 1)]
    median_diff = sorted(diffs)[len(diffs) // 2]
    if median_diff >= spacing // 2:
        return ws  # already well-spaced

    # --- Collect ALL fabula_time values across the world-state ---
    all_times: set[int] = set()
    for e in ws.events:
        all_times.add(e.fabula_time)
    for ce in ws.causal_topology:
        all_times.add(ce.fabula_time)
    for ie in ws.information_topology:
        all_times.add(ie.established_at_fabula)
        if ie.terminated_at_fabula is not None:
            all_times.add(ie.terminated_at_fabula)
    for se in ws.spatial_topology:
        all_times.add(se.established_at_fabula)
        if se.destroyed_at_fabula is not None:
            all_times.add(se.destroyed_at_fabula)
    for re_edge in ws.social_topology:
        all_times.add(re_edge.last_updated_fabula)
    for ent in ws.entities.values():
        for b in ent.beliefs:
            all_times.add(b.established_at_fabula)
        for snap in ent.state_timeline:
            all_times.add(snap.fabula_time)
    for wt in ws.world_traits.values():
        for snap in wt.state_timeline:
            all_times.add(snap.fabula_time)

    # 0 is the pre-story sentinel — keep it pinned at 0.
    nonzero_sorted = sorted(t for t in all_times if t > 0)
    if not nonzero_sorted:
        return ws

    # Build old → new mapping. 0 always stays 0.
    time_map: dict[int, int] = {0: 0}
    for i, t in enumerate(nonzero_sorted):
        time_map[t] = (i + 1) * spacing

    def _map(t: int | None) -> int | None:
        if t is None:
            return None
        return time_map.get(t, t)

    new_events = [
        e.model_copy(update={"fabula_time": _map(e.fabula_time) or e.fabula_time}) for e in ws.events
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
        "[Normalize] Rescaled %d unique fabula_time values (median event gap %d → %d).",
        len(nonzero_sorted), median_diff, spacing,
    )

    # Remap world trait snapshot fabula_times
    new_world_traits = {}
    for wid, wt in ws.world_traits.items():
        new_wt_timeline = [
            snap.model_copy(update={"fabula_time": _map(snap.fabula_time) or snap.fabula_time})
            for snap in wt.state_timeline
        ]
        new_world_traits[wid] = wt.model_copy(update={"state_timeline": new_wt_timeline})

    return WorldStateV1(
        locations=ws.locations,
        objects=ws.objects,
        entities=new_entities,
        events=new_events,
        world_traits=new_world_traits,
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

    When two edges share the same key, keeps the one with the later
    ``discovered_at_syuzhet`` so that subsequent re-extractions which
    update ``terminated_at_fabula`` (or other late-discovered fields)
    overwrite earlier records of the same channel rather than being
    silently dropped.
    """
    best: dict[tuple, InformationEdge] = {}
    for e in edges:
        key = (e.source_id, tuple(sorted(e.target_ids)), e.medium, e.established_at_fabula)
        existing = best.get(key)
        if existing is None or e.discovered_at_syuzhet >= existing.discovered_at_syuzhet:
            best[key] = e
    return list(best.values())


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
        world_traits=register.world_traits,
        events=events,
        causal_topology=causal_topology,
        spatial_topology=spatial_topology,
        information_topology=information_topology,
        social_topology=social_topology,
    )
    logger.info(
        "[Step 3] Assembled WorldStateV1 — %d events, %d causal, %d social, "
        "%d spatial, %d info edges, %d world traits.",
        len(ws.events), len(ws.causal_topology), len(ws.social_topology),
        len(ws.spatial_topology), len(ws.information_topology),
        len(ws.world_traits),
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
        | set(ws.world_traits.keys())
        | {e.id for e in ws.events}
    )
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
            world_traits=ws.world_traits,
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
        | set(ws.world_traits.keys())
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
        | set(ws.entities.keys()) | set(ws.world_traits.keys())
        | {e.id for e in ws.events}
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


# =====================================================================
# Step 5 — Post-Assembly World Trait Timeline Extraction
# =====================================================================


class _WorldTraitTimelineDeps(BaseModel):
    """Dependencies for Step 5 — world trait timeline extraction."""
    model_config = {"protected_namespaces": ()}
    world_traits: Dict[str, GlobalTrait]
    events: List[EventNode]


def _build_world_trait_timeline_agent(
    config: ExtractionConfig,
) -> Agent[_WorldTraitTimelineDeps, WorldTraitTimelineExtraction]:
    """Construct the Step 5 World Trait Timeline agent."""
    agent: Agent[_WorldTraitTimelineDeps, WorldTraitTimelineExtraction] = Agent(
        _resolve_model(config.model),
        deps_type=_WorldTraitTimelineDeps,
        output_type=NativeOutput(WorldTraitTimelineExtraction),
        system_prompt=_load_prompt("world_trait_timeline.md"),
        retries=config.output_retries,
    )

    @agent.system_prompt
    def inject_world_traits_and_events(ctx: RunContext[_WorldTraitTimelineDeps]) -> str:
        # Format world traits
        trait_lines: List[str] = []
        for wid, wt in ctx.deps.world_traits.items():
            trait_lines.append(
                f"  {wid} ({wt.name}): category={wt.category}, "
                f"magnitude={wt.magnitude.value:.2f}, inertia={wt.magnitude.inertia:.2f}\n"
                f"    Description: {wt.description}"
            )
        traits_block = "\n".join(trait_lines) if trait_lines else "(No world traits.)"

        # Format event timeline (compact)
        event_lines: List[str] = []
        for evt in ctx.deps.events:
            actors = ", ".join(evt.actor_ids) if evt.actor_ids else "none"
            event_lines.append(
                f"  {evt.id} (fabula={evt.fabula_time}, type={evt.event_type}, "
                f"actors=[{actors}]): {evt.description}"
            )
        events_block = "\n".join(event_lines) if event_lines else "(No events.)"

        return (
            "=== WORLD TRAITS (from Step 1d) ===\n"
            f"WORLD TRAIT IDs: {sorted(ctx.deps.world_traits.keys())}\n\n"
            f"{traits_block}\n\n"
            "=== COMPLETE EVENT TIMELINE (from Steps 2-4) ===\n"
            f"{events_block}\n\n"
            "Identify inflection points where the above world traits change "
            "due to specific events. Only include traits that actually change."
        )

    @agent.output_validator
    def validate_world_trait_timeline(
        ctx: RunContext[_WorldTraitTimelineDeps],
        result: WorldTraitTimelineExtraction,
    ) -> WorldTraitTimelineExtraction:
        """Validate that all referenced IDs are valid."""
        valid_world_ids = set(ctx.deps.world_traits.keys())
        valid_event_ids = {e.id for e in ctx.deps.events}
        event_fabula_map = {e.id: e.fabula_time for e in ctx.deps.events}
        bad: List[str] = []

        for wid, snapshots in result.timelines.items():
            if wid not in valid_world_ids:
                bad.append(f"World trait ID '{wid}' is not in the register.")
                continue
            for snap in snapshots:
                if snap.triggered_by and snap.triggered_by not in valid_event_ids:
                    bad.append(
                        f"WorldTraitSnapshot for '{wid}' references "
                        f"unknown event '{snap.triggered_by}'."
                    )
                if snap.triggered_by and snap.triggered_by in event_fabula_map:
                    expected_ft = event_fabula_map[snap.triggered_by]
                    if snap.fabula_time != expected_ft:
                        bad.append(
                            f"WorldTraitSnapshot for '{wid}' has fabula_time={snap.fabula_time} "
                            f"but triggered_by event '{snap.triggered_by}' has "
                            f"fabula_time={expected_ft}. They must match."
                        )

        if bad:
            raise ModelRetry(
                "The following issues were found in the world trait timeline. "
                "Fix them:\n" + "\n".join(bad)
            )
        return result

    return agent


def extract_world_trait_timelines(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Step 5: Post-assembly world trait timeline extraction.

    Given a fully assembled ``WorldStateV1`` with all events resolved,
    runs a single focused LLM call to identify inflection points where
    world traits changed due to specific events.

    Returns the world state with ``state_timeline`` populated on each
    ``GlobalTrait`` that experienced changes.
    """
    config = config or ExtractionConfig()

    if not ws.world_traits:
        logger.info("[Step 5] No world traits — skipping timeline extraction.")
        return ws

    agent = _build_world_trait_timeline_agent(config)
    deps = _WorldTraitTimelineDeps(
        world_traits=ws.world_traits,
        events=ws.events,
    )

    logger.info(
        "[Step 5] Extracting world trait timelines (%d traits, %d events) …",
        len(ws.world_traits), len(ws.events),
    )

    try:
        result = agent.run_sync(
            f"Analyze the following {len(ws.world_traits)} world trait(s) against "
            f"{len(ws.events)} events and identify any inflection points.",
            deps=deps,
        )
        extraction = result.output
        log_agent_output(logger, "WorldTraitTimeline", extraction)
    except Exception:
        logger.exception("[Step 5] World trait timeline extraction FAILED — skipping.")
        return ws

    # Apply timelines to the world state
    updated_traits: Dict[str, GlobalTrait] = {}
    changes_applied = 0
    for wid, wt in ws.world_traits.items():
        if wid in extraction.timelines and extraction.timelines[wid]:
            sorted_timeline = sorted(extraction.timelines[wid], key=lambda s: s.fabula_time)
            updated_traits[wid] = wt.model_copy(update={"state_timeline": sorted_timeline})
            changes_applied += 1
            logger.info(
                "[Step 5] %s: %d inflection point(s) identified.",
                wid, len(sorted_timeline),
            )
        else:
            updated_traits[wid] = wt

    ws = ws.model_copy(update={"world_traits": updated_traits})
    logger.info(
        "[Step 5] World trait timeline extraction complete — %d/%d traits changed.",
        changes_applied, len(ws.world_traits),
    )
    return ws


async def extract_world_trait_timelines_async(
    ws: WorldStateV1,
    config: ExtractionConfig | None = None,
) -> WorldStateV1:
    """Async variant of :func:`extract_world_trait_timelines`."""
    config = config or ExtractionConfig()

    if not ws.world_traits:
        logger.info("[Step 5·Async] No world traits — skipping timeline extraction.")
        return ws

    agent = _build_world_trait_timeline_agent(config)
    deps = _WorldTraitTimelineDeps(
        world_traits=ws.world_traits,
        events=ws.events,
    )

    logger.info(
        "[Step 5·Async] Extracting world trait timelines (%d traits, %d events) …",
        len(ws.world_traits), len(ws.events),
    )

    try:
        result = await agent.run(
            f"Analyze the following {len(ws.world_traits)} world trait(s) against "
            f"{len(ws.events)} events and identify any inflection points.",
            deps=deps,
        )
        extraction = result.output
    except Exception:
        logger.exception("[Step 5·Async] World trait timeline extraction FAILED — skipping.")
        return ws

    updated_traits: Dict[str, GlobalTrait] = {}
    changes_applied = 0
    for wid, wt in ws.world_traits.items():
        if wid in extraction.timelines and extraction.timelines[wid]:
            sorted_timeline = sorted(extraction.timelines[wid], key=lambda s: s.fabula_time)
            updated_traits[wid] = wt.model_copy(update={"state_timeline": sorted_timeline})
            changes_applied += 1
            logger.info(
                "[Step 5·Async] %s: %d inflection point(s) identified.",
                wid, len(sorted_timeline),
            )
        else:
            updated_traits[wid] = wt

    ws = ws.model_copy(update={"world_traits": updated_traits})
    logger.info(
        "[Step 5·Async] World trait timeline extraction complete — %d/%d traits changed.",
        changes_applied, len(ws.world_traits),
    )
    return ws


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

    # Step 5: Post-assembly world trait timeline extraction
    world_state = extract_world_trait_timelines(world_state, config)

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

    # Step 5: Post-assembly world trait timeline extraction
    world_state = await extract_world_trait_timelines_async(world_state, config)

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
