# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

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
import contextlib
import logging
import re
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from shadow_loom.research import WorldFact

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, RunContext

from shadow_loom.settings import get_settings as _get_settings, resolve_model as _resolve_model

from shadow_loom.models import (
    AmbientVector,
    Belief,
    CausalEdge,
    Channel,
    Entity,
    EntityStateSnapshot,
    EventNode,
    GlobalTrait,
    Location,
    NarrativeObject,
    RelationshipEdge,
    SpatialEdge,
    TraitVector,
    WorldStateV1,
    WorldTraitSnapshot,
)
from shadow_loom._agent_logging import log_agent_output
from shadow_loom.narrative_style import infer_narrative_style

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
    """Merged output per chunk: events + all edge types. Used by assembly.

    ``events`` includes both Physics-extracted events AND any
    ``event_type='utterance'`` events emitted by the Social Agent
    (utterance EventNodes are merged in by ``_extract_single_chunk*``).
    ``channels`` carries standing :class:`Channel` capabilities
    extracted by the Social Agent, replacing the legacy
    ``information_topology`` field.

    The ``new_*`` collections carry brand-new ontology nodes promoted
    from a Rung-2/3 sandbox after a ``.spawn`` (genesis) intervention.
    They are empty for plain ingestion chunks; the pipeline populates
    them when a query introduces characters / objects / locations /
    world-traits that did not exist in the canonical world state.
    """
    events: List[EventNode] = Field(default_factory=list)
    causal_topology: List[CausalEdge] = Field(default_factory=list)
    channels: Dict[str, Channel] = Field(default_factory=dict)
    social_topology: List[RelationshipEdge] = Field(default_factory=list)
    spatial_topology: List[SpatialEdge] = Field(default_factory=list)
    entity_updates: List["EntityUpdate"] = Field(default_factory=list)
    # Genesis spawns (post-physics promotion). Keyed by canonical id.
    new_entities: Dict[str, Entity] = Field(default_factory=dict)
    new_objects: Dict[str, NarrativeObject] = Field(default_factory=dict)
    new_locations: Dict[str, Location] = Field(default_factory=dict)
    new_world_traits: Dict[str, GlobalTrait] = Field(default_factory=dict)


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
    """Step 3b output from the Social Agent.

    Replaces the legacy ``information_topology`` field with a two-part
    split that mirrors the model:
      * ``channels`` — standing :class:`Channel` capabilities
        (telephone, mind-link, classified pipeline, ongoing
        correspondence). Keyed by CHN_ id.
      * ``utterance_events`` — discrete
        :class:`EventNode` records with ``event_type='utterance'``
        modelling on-page speech-acts. These are merged into the
        global ``events`` list during chunk-topology assembly so they
        participate in normal causal/temporal physics.
    """
    channels: Dict[str, Channel] = Field(default_factory=dict)
    utterance_events: List[EventNode] = Field(default_factory=list)
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
# Common LLM synonyms that should map to a canonical relationship axis
# rather than triggering ``mutation_social`` edge rejection. The
# Pydantic ``RelationshipMetric.metrics`` keys are a closed Literal so
# the LLM can't inject a synonym there directly, but ``CausalEdge.
# trait_target`` is a free string and the ``mutation_social`` validator
# only accepts the canonical names. Without this map a perfectly valid
# ``mutation_social`` edge whose trait_target reads "trust" or "power"
# is silently dropped instead of repaired.
_RELATIONSHIP_METRIC_ALIASES: dict[str, str] = {
    # Direct truncations / informal forms.
    "power": "power_dynamic",
    "dominance": "power_dynamic",
    "authority": "power_dynamic",
    "control": "power_dynamic",
    "trust": "affinity",
    "love": "affinity",
    "friendship": "affinity",
    "rapport": "affinity",
    "intimacy": "affinity",
    # Negative-direction synonyms map to the same axis; the sign is
    # carried by ``trait_delta`` (the LLM is instructed to use a
    # negative delta for hostility / contempt, etc.).
    "hostility": "affinity",
    "contempt": "affinity",
    "resentment": "affinity",
    "fearfulness": "fear",
    "dread": "fear",
    "terror": "fear",
    "anxiety": "fear",
}
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
        default="ollama:qwen3.6:35b",
        description="PydanticAI model string (e.g. 'ollama:qwen3.6:35b', 'openai:gpt-4o').",
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
    validation_payload_max_chars: int = Field(
        default=600_000,
        description=(
            "Hard char cap on the WorldStateV1 JSON sent to the LLM "
            "validator (Step 3 Phase B). When the serialised state "
            "exceeds this, the validator switches to a compact "
            "projection (or, as a last resort, truncates). The default "
            "(~600K chars \u2248 ~150K tokens) is sized for a 256K-token "
            "context window with comfortable headroom for the system "
            "prompt and the validator's structured-output response. "
            "Lower this when targeting a smaller-context model."
        ),
    )
    correction_subgraph_threshold_chars: int = Field(
        default=400_000,
        description=(
            "When the WorldStateV1 JSON sent to the correction-patch "
            "agent exceeds this size, fall back to an error-relevant "
            "subgraph (events named in the errors + their immediate "
            "causal neighbours + ontology header) instead of the full "
            "state. The default (~400K chars \u2248 ~100K tokens) keeps "
            "the patch contract rich enough for the LLM to reason "
            "across the whole topology on a 256K-token model while "
            "still cutting over before the prompt would crowd out the "
            "patch response."
        ),
    )
    max_concurrent_chunks: int = Field(
        default=4,
        ge=1,
        description="Maximum number of chunks to extract in parallel during "
        "async topology extraction. Controls LLM request concurrency. Must be >= 1; "
        "a value of 0 would create ``asyncio.Semaphore(0)`` and hang every chunk.",
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

    # ------------------------------------------------------------------
    # Step 3d — optional external research (off by default)
    # ------------------------------------------------------------------
    enable_research_agent: bool = Field(
        default=False,
        description=(
            "If true, after world-state assembly the pipeline calls the "
            "configured ``research_provider`` once per topic in "
            "``research_topics`` and a research-extraction agent distils "
            "each result into a ``WorldFact``. Facts are appended to "
            "``WorldStateV1.world_facts`` only — the agent is forbidden "
            "from mutating Entities, Events, RelationshipEdges or world "
            "traits. Off by default."
        ),
    )
    research_provider: Literal["none", "tavily"] = Field(
        default="none",
        description="Which ResearchProvider to use. 'none' = NullProvider (no-op).",
    )
    research_provider_model: str = Field(
        default="",
        description="Provider-specific search depth/model identifier (hashed into cache key).",
    )
    research_max_results_per_query: int = Field(
        default=5,
        description="Cap on snippets returned per provider call.",
    )
    research_topics: List[str] = Field(
        default_factory=list,
        description="Topics to look up at extraction time. May be empty.",
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
    """Construct the legacy single-pass Step 1 PydanticAI agent.

    .. deprecated::
        The active pipeline uses the four-pass split
        (`_build_location_agent`, `_build_object_agent`,
        `_build_entity_agent`, `_build_world_traits_agent`). This
        single-pass builder is retained only for backward-compat
        callers and emits a `DeprecationWarning` when invoked.
    """
    import warnings
    warnings.warn(
        "ontology_extraction.md / _build_ontology_agent is deprecated; "
        "use the split ontology_locations / ontology_objects / "
        "ontology_entities / ontology_world_traits agents instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "[ingestion] Legacy _build_ontology_agent invoked \u2014 the "
        "active pipeline uses the four-pass split; this prompt is "
        "stale relative to the current channel/world-trait schema."
    )
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


# ---------------------------------------------------------------------
# User-context plumbing (item #9 of the audit).
#
# Cost-tracking metadata (user_id / project_id / version_id) needs to
# reach every ``agent.run[_sync]`` call so per-call cost is attributed
# to the right user / project / version row in the DB. Threading kwargs
# through every helper would be invasive, so we stash the dict in a
# ``ContextVar`` for the duration of an ``run_extraction[_async]`` call
# and read it via ``_user_kwargs()`` immediately before each agent run.
# ---------------------------------------------------------------------

_user_context_var: ContextVar[Optional[Dict[str, Optional[int]]]] = ContextVar(
    "shadow_loom_user_context", default=None,
)


def _user_kwargs() -> Dict[str, Optional[int]]:
    """Return the active user_context as kwargs for ``agent.run[_sync]``.

    Returns an empty dict when no context is set so callers can splat it
    unconditionally: ``agent.run_sync(msg, deps=..., **_user_kwargs())``.
    """
    ctx = _user_context_var.get()
    return dict(ctx) if ctx else {}


@contextlib.contextmanager
def _user_context_scope(user_context: Dict[str, Optional[int]]):
    """Set ``_user_context_var`` for the duration of the with-block.

    Captures the token returned by ``ContextVar.set`` and resets it on
    exit (including exceptions). Without this scope, back-to-back
    extractions on the same thread / async task would inherit stale
    user attribution metadata from the previous run.
    """
    token = _user_context_var.set(user_context)
    try:
        yield
    finally:
        _user_context_var.reset(token)


def extract_ontology(text: str, config: ExtractionConfig | None = None, 
                    user_context: Optional[Dict[str, Optional[int]]] = None) -> GlobalRegister:
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
    user_context = user_context or {}

    # --- Step 1a: Locations ---
    location_agent = _build_location_agent(config)
    logger.info("[Step 1a] Extracting locations with %s …", config.model)
    try:
        loc_result = location_agent.run_sync(text, **user_context)
        loc_register = loc_result.output
    except Exception:
        logger.exception("[Step 1a] Location extraction failed — retrying once …")
        loc_result = location_agent.run_sync(text, **user_context)
        loc_register = loc_result.output
    log_agent_output(logger, "LocationOntology", loc_register)
    logger.info("[Step 1a] Extracted %d locations.", len(loc_register.locations))

    # --- Step 1b: Objects (with location context) ---
    object_agent = _build_object_agent(config)
    obj_deps = _ObjectDeps(location_register=loc_register)
    logger.info("[Step 1b] Extracting objects with %s …", config.model)
    try:
        obj_result = object_agent.run_sync(text, deps=obj_deps, **user_context)
        obj_register = obj_result.output
    except Exception:
        logger.exception("[Step 1b] Object extraction failed — retrying once …")
        obj_result = object_agent.run_sync(text, deps=obj_deps, **user_context)
        obj_register = obj_result.output
    log_agent_output(logger, "ObjectOntology", obj_register)
    logger.info("[Step 1b] Extracted %d objects.", len(obj_register.objects))

    # --- Step 1c: Entities (with location + object context) ---
    entity_agent = _build_entity_agent(config)
    ent_deps = _EntityDeps(location_register=loc_register, object_register=obj_register)
    logger.info("[Step 1c] Extracting entities with %s …", config.model)
    try:
        ent_result = entity_agent.run_sync(text, deps=ent_deps, **user_context)
        ent_register = ent_result.output
    except Exception:
        logger.exception("[Step 1c] Entity extraction failed — retrying once …")
        ent_result = entity_agent.run_sync(text, deps=ent_deps, **user_context)
        ent_register = ent_result.output
    log_agent_output(logger, "EntityOntology", ent_register)
    logger.info("[Step 1c] Extracted %d entities.", len(ent_register.entities))

    # --- Step 1d: World Traits (no dependencies) ---
    world_traits_agent = _build_world_traits_agent(config)
    logger.info("[Step 1d] Extracting world traits with %s …", config.model)
    try:
        wt_result = world_traits_agent.run_sync(text, **user_context)
        wt_register = wt_result.output
    except Exception:
        logger.exception("[Step 1d] World traits extraction failed — retrying once …")
        wt_result = world_traits_agent.run_sync(text, **user_context)
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
    sanitisation_notes: List[str] = []
    register = _sanitize_register(register, sanitisation_notes)
    for note in sanitisation_notes:
        logger.info(note)
    logger.info(
        "[Step 1] Ontology extracted — %d locations, %d objects, %d entities, %d world traits.",
        len(register.locations), len(register.objects), len(register.entities),
        len(register.world_traits),
    )
    return register


async def extract_ontology_async(
    text: str,
    config: ExtractionConfig | None = None,
    user_context: Optional[Dict[str, Optional[int]]] = None,
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
    user_context = user_context or {}

    # --- Step 1a: Locations (must complete first — both 1b and 1c need it) ---
    location_agent = _build_location_agent(config)
    logger.info("[Step 1a] Extracting locations with %s …", config.model)
    try:
        loc_result = await location_agent.run(text, **user_context)
        loc_register = loc_result.output
    except Exception:
        logger.exception("[Step 1a] Location extraction failed — retrying once …")
        loc_result = await location_agent.run(text, **user_context)
        loc_register = loc_result.output
    logger.info("[Step 1a] Extracted %d locations.", len(loc_register.locations))

    # --- Steps 1b + 1c in parallel ---
    async def _extract_objects() -> ObjectRegister:
        object_agent = _build_object_agent(config)
        obj_deps = _ObjectDeps(location_register=loc_register)
        logger.info("[Step 1b] Extracting objects with %s …", config.model)
        try:
            obj_result = await object_agent.run(text, deps=obj_deps, **user_context)
            return obj_result.output
        except Exception:
            logger.exception("[Step 1b] Object extraction failed — retrying once …")
            obj_result = await object_agent.run(text, deps=obj_deps, **user_context)
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
            ent_result = await entity_agent.run(text, deps=ent_deps, **user_context)
            return ent_result.output
        except Exception:
            logger.exception("[Step 1c] Entity extraction failed — retrying once …")
            ent_result = await entity_agent.run(text, deps=ent_deps, **user_context)
            return ent_result.output

    async def _extract_world_traits() -> WorldTraitsRegister:
        world_traits_agent = _build_world_traits_agent(config)
        logger.info("[Step 1d] Extracting world traits with %s …", config.model)
        try:
            wt_result = await world_traits_agent.run(text, **user_context)
            return wt_result.output
        except Exception:
            logger.exception("[Step 1d] World traits extraction failed — retrying once …")
            wt_result = await world_traits_agent.run(text, **user_context)
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
    sanitisation_notes: List[str] = []
    register = _sanitize_register(register, sanitisation_notes)
    for note in sanitisation_notes:
        logger.info(note)
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
    # Standing channels already extracted from prior chunks. Threaded
    # through the sequential ``extract_topology`` loop so the LLM can
    # reuse a CHN_ id by reference instead of inventing a near-duplicate
    # for the same standing capability (a long-running letter
    # correspondence, an ongoing telepathic bond, a spy-master pipeline
    # that spans the whole novel). Empty in the parallel-async path
    # where chunks have no causal ordering — dedup at assembly handles
    # that case via shape-key collapse.
    previous_chunk_channels: Dict[str, "Channel"] = Field(default_factory=dict)


class _ConsequencesDeps(BaseModel):
    """Dependencies for Step 3c (Consequences Agent: entity_updates).

    Receives the events and mutation/mutation_social edges already produced
    by the Physics Agent so that every EntityUpdate it emits is anchored
    to a concrete event and aligned with the causal mutations.

    Also receives the Social Agent's ``chunk_channels`` and
    ``chunk_utterance_events`` (when Step 3b has already run) so that
    any beliefs the agent emits can correctly populate
    ``acquired_via_event_id`` (the utterance) and
    ``acquired_via_channel_id`` (the standing capability the utterance
    rode over). Without this injection the prompt's belief-provenance
    fields are unfillable.
    """
    model_config = {"protected_namespaces": ()}
    global_register: GlobalRegister
    scaffold: SocraticScaffold
    chunk_events: List[EventNode] = Field(default_factory=list)
    chunk_causal: List[CausalEdge] = Field(default_factory=list)
    chunk_channels: Dict[str, "Channel"] = Field(default_factory=dict)
    chunk_utterance_events: List[EventNode] = Field(default_factory=list)
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


# Verbs / cue-tokens that strongly imply on-page speech-acts. The Social
# retry quality gate uses this to avoid retrying on legitimate pure-action
# chunks (chases, silent set-pieces, scenic description) where zero
# channels and zero utterances is the *correct* answer.
#
# Quote-mark coverage (used for paired-quote detection only — a *single*
# stray quote mark is no longer enough to trigger the retry, since
# narrators routinely use quoted single words for titles, scare-quotes,
# and proper-noun glosses):
#   "  — straight double  (ASCII 0x22)
#   \u201c \u201d         — curly double (English)
#   \u2018 \u2019         — curly single (English)
#   \u201a \u201e \u201f  — German low / high
#   \u00ab \u00bb         — French / Russian guillemets
#   \u2039 \u203a         — single guillemets
#   \u300c \u300d         — Japanese corner brackets
_QUOTE_CHARS = '"\u201c\u201d\u2018\u2019\u201a\u201e\u201f\u00ab\u00bb\u2039\u203a\u300c\u300d'
# Open/close pairs we accept as "balanced dialogue":
_QUOTE_PAIRS = (
    ("\u201c", "\u201d"),  # English curly double
    ("\u2018", "\u2019"),  # English curly single
    ("\u00ab", "\u00bb"),  # French guillemets
    ("\u2039", "\u203a"),  # single guillemets
    ("\u300c", "\u300d"),  # Japanese corner brackets
    ("\u201e", "\u201c"),  # German low-high
)
_SPEECH_VERB_RE = re.compile(
    r'\b('
    r'said|says|told|tells|asked|asks|replied|replies|whispered|whispers|'
    r'shouted|shouts|cried|cries|murmured|murmurs|muttered|mutters|'
    r'declared|declares|announced|announces|warned|warns|promised|promises|'
    r'confessed|confesses|admitted|admits|wrote|writes|read|reads|'
    r'letter|letters|note|notes|message|messages|prophecy|prophesied|'
    r'order|orders|command|commands|rumour|rumor|gossip|spoke|speaks|'
    r'answered|answers|interrupted|interrupts|exclaimed|exclaims'
    r')\b',
    re.IGNORECASE,
)

# Em-dash dialogue convention (French, Russian, Spanish, James Joyce):
#   — Have you no shame? he asked.
# Detected as a line starting with em-dash or en-dash followed by a
# capitalised letter (Latin or Cyrillic). Restricted to start-of-line
# anchoring to avoid false positives on parenthetical em-dashes mid
# sentence.
_EMDASH_DIALOGUE_RE = re.compile(
    r'(?:^|\n)\s*[\u2014\u2013]\s+[A-Z\u00C0-\u024f\u0400-\u04ff]'
)


def _chunk_likely_contains_speech(chunk_text: str) -> bool:
    """Return True if the chunk shows linguistic evidence of dialogue or
    written/transmitted communication. Cheap heuristic used by the
    Social Agent's empty-result retry gate.

    Decision tree (any one is sufficient):
      1. Em-dash / en-dash line opener (Joyce / French / Russian dialogue style).
      2. A balanced pair of dialogue-class quote marks (≥2 straight
         double-quotes, OR matching curly / guillemet / corner-bracket
         pair). A single stray quote on its own no longer counts —
         narrators routinely quote single words for titles, scare-quotes,
         and proper-noun glosses, and that was generating spurious
         social-extraction retries on action chunks.
      3. A recognised speech / writing / transmission verb (covers
         epistolary, reported speech, and channel-prose without quote
         marks).
    """
    if _EMDASH_DIALOGUE_RE.search(chunk_text):
        return True
    # Balanced quotes
    if chunk_text.count('"') >= 2:
        return True
    for opener, closer in _QUOTE_PAIRS:
        if opener in chunk_text and closer in chunk_text:
            return True
    if _SPEECH_VERB_RE.search(chunk_text):
        return True
    return False


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


def _normalize_id_candidate(candidate: str) -> str:
    """Strip whitespace and uppercase the prefix segment of an ID.

    LLMs occasionally emit IDs with stray whitespace (``"  ENT_X  "``)
    or mixed-case prefixes (``"Ent_x"``) — both forms are obviously
    broken but slip past exact-match before fuzzy lookup. Normalising
    here lets the simple ``candidate in valid_ids`` test recover most
    of these without burning a fuzzy-match round-trip.

    The body after the prefix is left untouched: entity / event id
    bodies are deliberately case-sensitive (``ENT_PETER`` vs
    ``ENT_PETER_PAN`` etc.) and re-casing them would produce more
    noise than signal.
    """
    if not isinstance(candidate, str):
        return candidate
    s = candidate.strip()
    for p in ("EVT_", "ENT_", "LOC_", "OBJ_", "WORLD_", "CHN_"):
        if s.upper().startswith(p):
            return p + s[len(p):]
    return s


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
    candidate = _normalize_id_candidate(candidate)
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
            # Try to repair via the alias map before dropping.
            alias = _RELATIONSHIP_METRIC_ALIASES.get(
                str(ce.trait_target).lower().strip()
            )
            if alias:
                notes.append(
                    f"[Auto-Fix] mutation_social trait_target "
                    f"'{ce.trait_target}' → '{alias}' "
                    f"({ce.source_id}→{ce.target_id})"
                )
                updates["trait_target"] = alias
            else:
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
    """Clamp per-metric value/inertia ranges and coerce evidence_strength.

    Operates on the new ``metrics`` dict structure. Each axis has its
    own range:
      * affinity \u2208 [-1, 1]
      * fear \u2208 [0, 1]
      * power_dynamic \u2208 [-1, 1]
      * inertia \u2208 [0, 0.99] (1.0 would freeze the metric forever)
    """
    metric_ranges = {
        "affinity": (-1.0, 1.0),
        "fear": (0.0, 1.0),
        "power_dynamic": (-1.0, 1.0),
    }
    es_aliases = {
        "high": "strong", "low": "weak", "medium": "moderate",
        "med": "moderate", "uncertain": "weak", "certain": "strong",
    }

    new_metrics: dict = {}
    changed = False
    for name, m in re_edge.metrics.items():
        m_updates: dict = {}
        # Clamp the metric value to its native range.
        lo, hi = metric_ranges.get(name, (-1.0, 1.0))
        if not (lo <= m.value <= hi):
            clamped = _clamp(m.value, lo, hi)
            notes.append(
                f"[Auto-Fix] Clamped RelationshipEdge.metrics['{name}'].value "
                f"{m.value} \u2192 {clamped} "
                f"({re_edge.source_entity_id}\u2192{re_edge.target_entity_id})"
            )
            m_updates["value"] = clamped
        # Per-metric inertia clamp; cap at 0.99 so propagation never freezes.
        if not (0.0 <= m.inertia <= 1.0):
            clamped_in = _clamp(m.inertia, 0.0, 0.99)
            notes.append(
                f"[Auto-Fix] Clamped RelationshipEdge.metrics['{name}'].inertia "
                f"{m.inertia} \u2192 {clamped_in} "
                f"({re_edge.source_entity_id}\u2192{re_edge.target_entity_id})"
            )
            m_updates["inertia"] = clamped_in
        elif m.inertia >= 1.0:
            notes.append(
                f"[Auto-Fix] Capped RelationshipEdge.metrics['{name}'].inertia "
                f"at 0.99 ({re_edge.source_entity_id}\u2192"
                f"{re_edge.target_entity_id}) \u2014 1.0 would freeze the axis."
            )
            m_updates["inertia"] = 0.99
        # Coerce evidence_strength aliases.
        if m.evidence_strength not in ("weak", "moderate", "strong"):
            alias = es_aliases.get(str(m.evidence_strength).lower().strip())
            if alias:
                notes.append(
                    f"[Auto-Fix] RelationshipEdge.metrics['{name}']."
                    f"evidence_strength '{m.evidence_strength}' \u2192 "
                    f"'{alias}' ({re_edge.source_entity_id}\u2192"
                    f"{re_edge.target_entity_id})"
                )
                m_updates["evidence_strength"] = alias
            else:
                m_updates["evidence_strength"] = "moderate"
        if m_updates:
            new_metrics[name] = m.model_copy(update=m_updates)
            changed = True
        else:
            new_metrics[name] = m

    if changed:
        return re_edge.model_copy(update={"metrics": new_metrics})
    return re_edge


def _sanitize_register(register: "GlobalRegister", notes: List[str]) -> "GlobalRegister":
    """Clamp baseline numeric ranges on Step-1 ontology records.

    Pydantic ``field_validator``s already coerce ``evidence_strength``
    aliases on :class:`TraitVector`, :class:`AmbientVector`,
    :class:`Belief`, and :class:`InformationEdge` at construction time,
    so this pass focuses on numeric ranges that the LLM occasionally
    overshoots (``value``, ``inertia``, ``volatility``, ``confidence``)
    and caps inertia / volatility at ``0.99`` so propagation is never
    literally frozen by a bad ``1.0`` extraction.
    """
    # Entities: traits + beliefs
    for ent in register.entities.values():
        new_traits: Dict[str, TraitVector] = {}
        for tname, tv in ent.traits.items():
            new_value = _clamp(tv.value, 0.0, 1.0)
            new_inertia = _clamp(tv.inertia, 0.0, 1.0)
            if new_inertia >= 1.0:
                new_inertia = 0.99
                notes.append(
                    f"[Auto-Fix] Capped baseline trait '{ent.id}.{tname}' inertia at 0.99 "
                    f"\u2014 1.0 would freeze the trait for the whole story."
                )
            if new_value != tv.value or new_inertia != tv.inertia:
                if new_value != tv.value or new_inertia != tv.inertia:
                    notes.append(
                        f"[Auto-Fix] Clamped baseline trait '{ent.id}.{tname}' "
                        f"value/inertia {tv.value:.2f}/{tv.inertia:.2f} \u2192 "
                        f"{new_value:.2f}/{new_inertia:.2f}"
                    )
            new_traits[tname] = TraitVector(
                value=new_value,
                inertia=new_inertia,
                evidence_strength=tv.evidence_strength,
            )
        ent.traits = new_traits

        new_beliefs: List[Belief] = []
        for b in ent.beliefs:
            updates: dict = {}
            new_conf = _clamp(b.confidence, 0.0, 1.0)
            new_in = _clamp(b.inertia, 0.0, 1.0)
            if new_in >= 1.0:
                new_in = 0.99
            if new_conf != b.confidence:
                updates["confidence"] = new_conf
            if new_in != b.inertia:
                updates["inertia"] = new_in
            new_beliefs.append(b.model_copy(update=updates) if updates else b)
        ent.beliefs = new_beliefs

    # Locations: ambient_state
    for loc in register.locations.values():
        new_ambient = {}
        for aname, av in loc.ambient_state.items():
            new_value = _clamp(av.value, 0.0, 1.0)
            new_volatility = _clamp(av.volatility, 0.0, 1.0)
            if new_value != av.value or new_volatility != av.volatility:
                notes.append(
                    f"[Auto-Fix] Clamped ambient '{loc.id}.{aname}' "
                    f"value/volatility {av.value:.2f}/{av.volatility:.2f} \u2192 "
                    f"{new_value:.2f}/{new_volatility:.2f}"
                )
            new_ambient[aname] = AmbientVector(
                value=new_value,
                volatility=new_volatility,
                evidence_strength=av.evidence_strength,
            )
        loc.ambient_state = new_ambient

    # World traits: magnitude (TraitVector)
    for wt in register.world_traits.values():
        mag = wt.magnitude
        new_value = _clamp(mag.value, 0.0, 1.0)
        new_inertia = _clamp(mag.inertia, 0.0, 1.0)
        if new_inertia >= 1.0:
            new_inertia = 0.99
        if new_value != mag.value or new_inertia != mag.inertia:
            notes.append(
                f"[Auto-Fix] Clamped world-trait '{wt.id}.magnitude' "
                f"value/inertia {mag.value:.2f}/{mag.inertia:.2f} \u2192 "
                f"{new_value:.2f}/{new_inertia:.2f}"
            )
            wt.magnitude = TraitVector(
                value=new_value,
                inertia=new_inertia,
                evidence_strength=mag.evidence_strength,
            )

    return register


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
        es_aliases = {
            "high": "strong", "low": "weak", "medium": "moderate",
            "med": "moderate", "uncertain": "weak", "certain": "strong",
        }
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
            new_es = tv.evidence_strength
            if new_es not in ("weak", "moderate", "strong"):
                new_es = es_aliases.get(str(new_es).lower().strip(), "moderate")
                notes.append(
                    f"[Auto-Fix] EntityUpdate trait '{tname}' evidence_strength "
                    f"'{tv.evidence_strength}' → '{new_es}' "
                    f"({eu.entity_id}@{eu.fabula_time})"
                )
            cleaned[tname] = TraitVector(
                value=new_value, inertia=new_inertia, evidence_strength=new_es,
            )
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
            # Coerce evidence_strength aliases (high/low/medium → strong/weak/moderate).
            if b.evidence_strength not in ("weak", "moderate", "strong"):
                alias = {
                    "high": "strong", "low": "weak", "medium": "moderate",
                    "med": "moderate", "uncertain": "weak", "certain": "strong",
                }.get(str(b.evidence_strength).lower().strip(), "moderate")
                b_updates["evidence_strength"] = alias
                notes.append(
                    f"[Auto-Fix] Belief.evidence_strength '{b.evidence_strength}' → "
                    f"'{alias}' ({eu.entity_id} re: {b.target_id})"
                )
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
        object_ids_set = set(reg.objects.keys())
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
                # barrier_item_id must reference a real OBJ_ id when set;
                # fuzzy-fix typos, drop dangling pointers (the lock fact
                # survives — the engine treats unowned barriers as
                # "no key exists in the world", which is structurally
                # weaker than a barrier whose unlock affordance no
                # entity carries, but at least it's not pointing at a
                # non-existent object).
                if se.barrier_item_id:
                    bid, _ = _fix_id(
                        se.barrier_item_id, object_ids_set,
                        "SpatialEdge.barrier_item_id", fixes,
                    )
                    if bid in object_ids_set:
                        if bid != se.barrier_item_id:
                            updates["barrier_item_id"] = bid
                    else:
                        fixes.append(
                            f"[Auto-Fix] SpatialEdge {src}→{tgt} "
                            f"barrier_item_id '{se.barrier_item_id}' is not "
                            f"a valid OBJ_ id; clearing reference."
                        )
                        updates["barrier_item_id"] = None
                # Coherence: a passage flagged is_locked=True with no
                # barrier_item_id is structurally inert — nothing in the
                # world can ever unlock it, so the affordance check
                # silently falls through. Downgrade to is_locked=False
                # so the renderer doesn't describe the door as locked
                # while the simulator treats it as freely traversable.
                resolved_barrier = updates.get("barrier_item_id", se.barrier_item_id)
                if se.is_locked and not resolved_barrier:
                    fixes.append(
                        f"[Auto-Fix] SpatialEdge {src}→{tgt} is_locked=True "
                        f"with no barrier_item_id — clearing the lock "
                        f"(an unblockable barrier breaks affordance gating)."
                    )
                    updates["is_locked"] = False
                # Lifecycle sanity: destroyed_at_fabula must be strictly
                # after established_at_fabula, otherwise the passage is
                # destroyed before (or at the same tick as) it was built
                # — which makes the edge a no-op for the simulator and
                # is almost certainly an LLM transcription slip. Drop
                # the destruction tick rather than the whole edge so
                # the connectivity fact survives.
                est = se.established_at_fabula or 0
                dest = se.destroyed_at_fabula
                if dest is not None and dest <= est:
                    fixes.append(
                        f"[Auto-Fix] SpatialEdge {src}→{tgt} destroyed_at_fabula"
                        f"={dest} <= established_at_fabula={est}; clearing "
                        f"destruction tick (passage stays traversable)."
                    )
                    updates["destroyed_at_fabula"] = None
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

    @agent.output_validator
    def reject_physics_utterances(
        ctx: RunContext[_PhysicsDeps], result: PhysicsExtraction,
    ) -> PhysicsExtraction:
        """Strip ``event_type='utterance'`` events from physics output.

        Per the physics_extraction.md prompt, utterances belong to the
        Social Agent. The schema permits any ``EventLiteral`` so smaller
        models routinely violate this and the social pass then re-emits
        the same utterance, leaving duplicates that dedup may or may
        not catch by id coincidence. Drop them here so the social pass
        is the single canonical source of utterances.
        """
        utterance_events = [e for e in result.events if e.event_type == "utterance"]
        if not utterance_events:
            return result
        kept = [e for e in result.events if e.event_type != "utterance"]
        dropped_ids = [e.id for e in utterance_events]
        dropped_set = set(dropped_ids)
        # Drop any causal/entity-update edges referencing the dropped ids
        # so downstream programmatic validation does not surface broken
        # links for things we just removed.
        cleaned_causal = [
            ce for ce in result.causal_topology
            if ce.source_id not in dropped_set and ce.target_id not in dropped_set
        ]
        cleaned_updates = [
            eu for eu in result.entity_updates
            if eu.triggered_by not in dropped_set
        ]
        logger.info(
            "[Validator·Physics] Dropped %d utterance event(s) from physics "
            "output (utterances belong to the Social Agent): %s.",
            len(utterance_events), dropped_ids,
        )
        return PhysicsExtraction(
            events=kept,
            causal_topology=cleaned_causal,
            spatial_topology=result.spatial_topology,
            entity_updates=cleaned_updates,
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

        # Prior-chunk channel summary so the LLM can reuse standing
        # capabilities by id rather than reinventing them under a new
        # name. Each line carries everything needed to identify the
        # right channel: id, medium, participants, directionality, and
        # the fabula tick it was established at.
        prior_chn_lines: List[str] = []
        for cid, ch in ctx.deps.previous_chunk_channels.items():
            prior_chn_lines.append(
                f"  - {cid} (medium={ch.medium}, "
                f"participants={ch.participant_ids}, "
                f"directionality={ch.directionality}, "
                f"established_at_fabula={ch.established_at_fabula})"
            )
        prior_channels_block = (
            "\n".join(prior_chn_lines)
            if prior_chn_lines
            else "  (none — this is the first chunk to extract channels)"
        )

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
            "=== STANDING CHANNELS ALREADY ESTABLISHED IN PRIOR CHUNKS ===\n"
            f"{prior_channels_block}\n"
            "\n"
            "When an utterance in THIS chunk travels over a standing "
            "capability that already appears above (e.g. an ongoing "
            "letter correspondence, a telephone line, a mind-bond, a "
            "spy pipeline), set the utterance's `via_channel_id` to "
            "the existing CHN_ id and DO NOT re-emit the channel in "
            "your `channels` dict. Only add a new entry to `channels` "
            "for genuinely-new standing capabilities established (or "
            "first observed) in this chunk.\n"
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

        # --- Fix Channel participants ---
        fixed_channels: Dict[str, Channel] = {}
        for cid, ch in result.channels.items():
            updates: dict = {}
            new_pids: List[str] = []
            for pid in ch.participant_ids:
                p, _ = _fix_id(pid, node_ids, "Channel.participant_ids", fixes)
                if p in node_ids:
                    if p not in new_pids:
                        new_pids.append(p)
                else:
                    bad.append(f"Channel '{cid}' participant_id '{pid}' is not a valid entity/object.")
            if len(new_pids) < 2:
                fixes.append(
                    f"[Auto-Fix] Dropped Channel '{cid}' — fewer than 2 valid participants."
                )
                continue
            if new_pids != list(ch.participant_ids):
                updates["participant_ids"] = new_pids
            # Drop intelligibility entries for participants we removed
            if ch.intelligibility:
                pruned_intel = {k: v for k, v in ch.intelligibility.items() if k in new_pids}
                if pruned_intel != dict(ch.intelligibility):
                    updates["intelligibility"] = pruned_intel
            if ch.evidence_strength not in ("weak", "moderate", "strong"):
                alias = {
                    "high": "strong", "low": "weak", "medium": "moderate",
                    "med": "moderate", "uncertain": "weak", "certain": "strong",
                }.get(str(ch.evidence_strength).lower().strip(), "moderate")
                updates["evidence_strength"] = alias
                fixes.append(
                    f"[Auto-Fix] Channel.evidence_strength "
                    f"'{ch.evidence_strength}' → '{alias}' (on {cid})"
                )
            fixed_channels[cid] = ch.model_copy(update=updates) if updates else ch

        # --- Fix utterance EventNode ids (speaker, addressees, channel) ---
        valid_channel_ids = set(fixed_channels.keys()) | set(
            ctx.deps.previous_chunk_channels.keys()
        )
        all_evt_ids = set(ctx.deps.previous_event_ids) | set(ctx.deps.chunk_event_ids)
        fixed_utterances: List[EventNode] = []
        seen_utt_ids: set[str] = set()
        for ev in result.utterance_events:
            if ev.event_type != "utterance":
                fixes.append(
                    f"[Auto-Fix] Coerced utterance_events entry '{ev.id}' "
                    f"event_type from '{ev.event_type}' to 'utterance'."
                )
                ev = ev.model_copy(update={"event_type": "utterance"})
            updates: dict = {}
            # --- Enforce EVT_UTT_ prefix to avoid collisions with Physics ids ---
            new_id = ev.id
            if not new_id.startswith("EVT_UTT_"):
                if new_id.startswith("EVT_"):
                    new_id = "EVT_UTT_" + new_id[len("EVT_"):]
                else:
                    new_id = "EVT_UTT_" + re.sub(r"[^A-Z0-9_]+", "_", new_id.upper()).strip("_")
                fixes.append(
                    f"[Auto-Fix] Renamed utterance '{ev.id}' → '{new_id}' "
                    f"(EVT_UTT_ prefix is required)."
                )
            # Resolve collisions with Physics-extracted ids OR previous-chunk ids
            # OR an earlier utterance in this same batch.
            if new_id in all_evt_ids or new_id in seen_utt_ids:
                base = new_id
                suffix = 2
                while f"{base}_{suffix}" in all_evt_ids or f"{base}_{suffix}" in seen_utt_ids:
                    suffix += 1
                renamed = f"{base}_{suffix}"
                fixes.append(
                    f"[Auto-Fix] Renamed utterance '{new_id}' → '{renamed}' "
                    f"(id collided with an existing event)."
                )
                new_id = renamed
            seen_utt_ids.add(new_id)
            if new_id != ev.id:
                updates["id"] = new_id
            # speaker_id
            if ev.speaker_id:
                sp, _ = _fix_id(ev.speaker_id, node_ids, "EventNode.speaker_id", fixes)
                if sp not in node_ids:
                    bad.append(f"Utterance '{ev.id}' speaker_id '{ev.speaker_id}' is not a valid entity/object.")
                elif sp != ev.speaker_id:
                    updates["speaker_id"] = sp
            # addressee_ids
            new_addrs: List[str] = []
            for aid in ev.addressee_ids:
                a, _ = _fix_id(aid, node_ids, "EventNode.addressee_ids", fixes)
                if a in node_ids and a not in new_addrs:
                    new_addrs.append(a)
                elif a not in node_ids:
                    bad.append(f"Utterance '{ev.id}' addressee_ids entry '{aid}' is not a valid entity/object.")
            if new_addrs != list(ev.addressee_ids):
                updates["addressee_ids"] = new_addrs
            # via_channel_id
            if ev.via_channel_id and ev.via_channel_id not in valid_channel_ids:
                # Drop the broken channel reference rather than fail — the
                # utterance is still valid as an unmediated speech-act.
                fixes.append(
                    f"[Auto-Fix] Utterance '{ev.id}' via_channel_id "
                    f"'{ev.via_channel_id}' is not in the chunk's channels; "
                    f"clearing reference."
                )
                updates["via_channel_id"] = None
            elif ev.via_channel_id and ev.via_channel_id in fixed_channels:
                # Membership check: when an utterance rides a channel
                # extracted in THIS chunk, every speaker/addressee MUST
                # appear in that channel's participant_ids — otherwise
                # the channel mediation is structurally incoherent
                # (you can't broadcast over a phone line you're not on).
                # We can only enforce this for in-chunk channels because
                # prior-chunk channels' participant lists may have been
                # widened by intervening extractions.
                ch_pids = set(fixed_channels[ev.via_channel_id].participant_ids)
                # Resolve the post-fix speaker / addressees so the
                # check sees the same ids the validator just rewrote.
                resolved_speaker = updates.get("speaker_id", ev.speaker_id)
                resolved_addrs = updates.get("addressee_ids", list(ev.addressee_ids))
                participants_needed = {resolved_speaker} | set(resolved_addrs)
                participants_needed.discard(None)
                missing = sorted(participants_needed - ch_pids)
                if missing:
                    fixes.append(
                        f"[Auto-Fix] Utterance '{ev.id}' via_channel_id "
                        f"'{ev.via_channel_id}' missing participant(s) "
                        f"{missing}; clearing channel reference (the "
                        f"utterance survives as unmediated speech)."
                    )
                    updates["via_channel_id"] = None
            # actor_ids consistency: if speaker present, ensure it appears in actor_ids
            if ev.speaker_id and ev.speaker_id not in ev.actor_ids:
                merged_actors = list(ev.actor_ids)
                merged_actors.append(updates.get("speaker_id", ev.speaker_id))
                updates["actor_ids"] = merged_actors
            fixed_utterances.append(ev.model_copy(update=updates) if updates else ev)

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
                sanitized = _sanitize_relationship_edge(rebuilt, fixes)
                # Drop dyads with no observed metrics — an edge whose
                # ``metrics`` dict is empty (or contains only
                # ``observed=False`` axes) is structurally valid but
                # contributes no signal: every aggregator returns 0.0
                # indistinguishable from a measured neutral, polluting
                # asymmetry / tension / propagation pipelines downstream.
                observed_axes = [
                    name for name, m in sanitized.metrics.items() if m.observed
                ]
                if not observed_axes:
                    fixes.append(
                        f"[Auto-Fix] Dropped RelationshipEdge "
                        f"{src}→{tgt} with no observed metrics "
                        f"(metrics={list(sanitized.metrics.keys())})."
                    )
                    continue
                fixed_social.append(sanitized)

        if fixes:
            logger.info("[Validator·Social] Auto-fixed %d issue(s): %s", len(fixes), "; ".join(fixes))

        if bad:
            raise ModelRetry(
                "The following IDs could not be auto-resolved. "
                "Fix them using ONLY IDs from the register:\n" + "\n".join(bad)
            )

        return SocialExtraction(
            channels=fixed_channels,
            utterance_events=fixed_utterances,
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

        # Compact channels + utterances summary so the agent can wire
        # belief provenance through ``acquired_via_event_id`` and
        # ``acquired_via_channel_id``.
        chn_lines: List[str] = []
        for cid, ch in ctx.deps.chunk_channels.items():
            chn_lines.append(
                f"  - {cid} (medium={ch.medium}, "
                f"participants={ch.participant_ids}, "
                f"directionality={ch.directionality})"
            )
        channels_block = "\n".join(chn_lines) if chn_lines else "  (no channels in this chunk)"

        utt_lines: List[str] = []
        for u in ctx.deps.chunk_utterance_events:
            utt_lines.append(
                f"  - {u.id} (fabula={u.fabula_time}, "
                f"speaker={u.speaker_id}, addressees={u.addressee_ids}, "
                f"via={u.via_channel_id}, truth={u.truth_value}): {u.description}"
            )
        utterances_block = (
            "\n".join(utt_lines) if utt_lines else "  (no utterance events in this chunk)"
        )

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
            "=== CHANNELS EXTRACTED FROM THIS CHUNK (by Social Agent) ===\n"
            f"{channels_block}\n"
            "\n"
            "=== UTTERANCE EVENTS EXTRACTED FROM THIS CHUNK (by Social Agent) ===\n"
            f"{utterances_block}\n"
            "\n"
            "When you emit a `new_beliefs` entry whose source is one of the "
            "utterance events above, set `acquired_via_event_id` to that "
            "utterance's id, and (when the utterance has a `via_channel_id`) "
            "set `acquired_via_channel_id` to the channel id. This lets "
            "counterfactual surgery prune downstream beliefs cleanly when "
            "the channel is severed or the utterance is rewritten.\n"
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


# =====================================================================
# Per-chunk quality gates — soft retries that fire when the LLM
# under-extracts on a structural axis the prompts already require but
# the schema can't enforce. Each helper returns True iff a retry is
# warranted; the caller is responsible for actually running it.
# =====================================================================


def _physics_causal_density_low(physics: "PhysicsExtraction") -> bool:
    """True iff the chunk has events but suspiciously few causal edges.

    Rule of thumb: every non-trivial event should participate in at
    least one causal edge (rule #12 of the physics prompt). If a chunk
    yields ``N >= 2`` events but ``len(causal_topology) < N``, the
    extractor very likely missed mutation / chain_reaction /
    affordance_gate edges. We retry with an explicit nudge.
    """
    n_evt = len(physics.events)
    if n_evt < 2:
        return False
    return len(physics.causal_topology) < n_evt


def _physics_missing_mutation_social(
    physics: "PhysicsExtraction",
    social: "SocialExtraction",
) -> List[str]:
    """Return a list of axes for which Social observed a non-zero value
    but Physics emitted no matching ``mutation_social`` edge.

    Per the per-axis coverage rule in ``physics_extraction.md``: every
    ``RelationshipEdge`` axis with ``observed=True`` and ``value != 0``
    commits Physics to at least one ``mutation_social`` edge for that
    axis. Without it the affective gauges (danger, conflict,
    power-dynamic) read flat. We collect violations as ``"axis"``
    strings (deduplicated) so the retry message can be specific.
    """
    if not social.social_topology or not physics.events:
        return []
    observed_axes: set[str] = set()
    for rel in social.social_topology:
        for axis_name, m in rel.metrics.items():
            if getattr(m, "observed", True) and float(m.value) != 0.0:
                observed_axes.add(axis_name)
    if not observed_axes:
        return []
    covered: set[str] = set()
    for ce in physics.causal_topology:
        if ce.causality_type == "mutation_social" and ce.trait_target:
            covered.add(str(ce.trait_target))
    missing = sorted(observed_axes - covered)
    return missing


def _physics_missing_mutation_social_per_dyad(
    physics: "PhysicsExtraction",
    social: "SocialExtraction",
) -> List[Tuple[str, str, str]]:
    """Return per-dyad-per-axis gaps in ``mutation_social`` coverage.

    Stronger than :func:`_physics_missing_mutation_social`: instead of
    only checking that *some* mutation_social edge exists per axis
    globally, this checks that **each individual dyad with an
    ``observed=True`` non-zero axis** has at least one mutation_social
    edge wired to *that exact (dyad, axis)* triple. Without this
    finer-grained check, a fixture can satisfy the global axis count
    while leaving individual relationship traces flat — the symptom
    we observed on the bundled Star Wars project, where six of nine
    dyads carried observed axes that no mutation_social edge ever
    touched. The downstream timeline reconstructor cannot move a
    metric without a mutation, so the affective gauges' sub-curves
    sit at the authored value across the entire fabula axis.

    Returns a list of ``(target_id, counterpart_id, axis)`` triples
    in deterministic order, one per missing dyad-axis combination.
    """
    if not social.social_topology or not physics.events:
        return []

    # Build the set of (target, counterpart, axis) triples Physics
    # actually covered. ``mutation_social`` semantics: target_id is
    # the perspective entity (whose view of the relationship mutates),
    # rel_counterpart_id is the other half of the dyad.
    #
    # ``RelationshipEdge`` is *directed* — affinity, fear and
    # power_dynamic on edge ``A→B`` describe how A feels/stands toward
    # B, and may differ from the reverse ``B→A`` edge. Indexing both
    # directions as covered (an earlier version of this function did
    # so on the assumption that affinity is undirected) silently
    # accepted single-direction mutation_social coverage as covering
    # both halves of the dyad — exactly the bug this finer-grained
    # check is meant to surface. We now key strictly on the directed
    # (target, counterpart) pair so a missing reverse-direction
    # mutation is still flagged when only the forward direction was
    # extracted.
    covered: set[Tuple[str, str, str]] = set()
    for ce in physics.causal_topology:
        if ce.causality_type != "mutation_social":
            continue
        if not ce.trait_target or not ce.target_id:
            continue
        counterpart = getattr(ce, "rel_counterpart_id", None) or ""
        axis = str(ce.trait_target).lower()
        if axis == "power":
            axis = "power_dynamic"
        covered.add((ce.target_id, counterpart, axis))

    missing: list[Tuple[str, str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for rel in social.social_topology:
        src = rel.source_entity_id
        tgt = rel.target_entity_id
        for axis_name, m in rel.metrics.items():
            if not getattr(m, "observed", True):
                continue
            if float(m.value) == 0.0:
                continue
            key = (src, tgt, axis_name)
            if key in seen:
                continue
            if key in covered:
                continue
            seen.add(key)
            missing.append(key)
    missing.sort()
    return missing


    missing.sort()
    return missing


def _consequences_mutation_parity_broken(
    physics: "PhysicsExtraction",
    consequences: "ConsequencesExtraction",
) -> List[str]:
    """Return entity ids that have an inbound mutation edge from Physics
    but no matching ``EntityUpdate`` in Consequences.

    Each ``mutation`` edge with an ``ENT_`` target should produce at
    least one ``EntityUpdate`` for that entity (the parity contract
    spelt out at the top of ``consequences_extraction.md``). When the
    parity is broken, the snapshot the UI shows for that entity will
    sit at the pre-story baseline and the trait shift the edge declared
    is never anchored on the timeline.
    """
    targets: set[str] = set()
    for ce in physics.causal_topology:
        if ce.causality_type == "mutation" and ce.target_id.startswith("ENT_"):
            targets.add(ce.target_id)
    if not targets:
        return []
    covered: set[str] = {eu.entity_id for eu in consequences.entity_updates}
    return sorted(targets - covered)


def _social_channel_underextracted(
    social: "SocialExtraction",
    *,
    min_repeated_dyad: int = 2,
) -> bool:
    """True when the chunk's utterance pattern strongly implies a
    standing channel that the LLM failed to extract.

    Heuristic: at least one ``(speaker_id, frozenset(addressee_ids))``
    pair appears in ``min_repeated_dyad`` or more utterance events
    AND none of those utterances carries a ``via_channel_id`` AND
    no Channel was emitted that already covers that pair.

    The cue is weakly-but-consistently informative: two letters from
    the same hand to the same recipient, two telephone calls between
    the same two people, two wireless broadcasts to the same
    audience — each is much more naturally modelled as a single
    persistent capability than as N independent unmediated
    speech-acts. Without a Channel, downstream cycle-detection,
    mediation tracking, and counterfactual surgery (severing a line
    of communication) all silently lose teeth.

    Returns True only when ALL three conditions hold simultaneously,
    so a chunk legitimately full of face-to-face conversation
    (different participants each time, or already-mediated
    correspondence) does not trigger a false-positive retry.
    """
    if not social.utterance_events:
        return False
    # Skip if every relevant utterance is already wired to some channel
    # (either this chunk's channels or a prior chunk's channel that
    # the validator preserved on ``via_channel_id``).
    pair_counts: Dict[Tuple[str, frozenset], int] = {}
    pair_unmediated: Dict[Tuple[str, frozenset], int] = {}
    for u in social.utterance_events:
        if not u.speaker_id or not u.addressee_ids:
            continue
        key = (u.speaker_id, frozenset(u.addressee_ids))
        pair_counts[key] = pair_counts.get(key, 0) + 1
        if not u.via_channel_id:
            pair_unmediated[key] = pair_unmediated.get(key, 0) + 1
    if not pair_counts:
        return False
    # A pair is already 'covered' if any channel in this chunk has a
    # superset of its participants — in that case we don't need to
    # retry just because via_channel_id wasn't filled in.
    covered_pairs: set[Tuple[str, frozenset]] = set()
    for ch in social.channels.values():
        ch_pids = set(ch.participant_ids)
        for key in pair_counts:
            speaker, addrs = key
            if {speaker} <= ch_pids and set(addrs) <= ch_pids:
                covered_pairs.add(key)
    for key, count in pair_counts.items():
        if (
            count >= min_repeated_dyad
            and pair_unmediated.get(key, 0) >= min_repeated_dyad
            and key not in covered_pairs
        ):
            return True
    return False


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
    # Standing channels accumulated across chunks. Threaded into the
    # Social Agent so chunk N can reuse a CHN_ id established in chunk
    # N-k instead of inventing a near-duplicate.
    accumulated_channels: Dict[str, "Channel"] = {}
    prev_chunk_tail = ""  # trailing context for coreference continuity
    # Per-chunk sub-stage failure tracking (item #8). When any single
    # sub-stage fails on >50% of chunks we escalate to RuntimeError
    # rather than silently persisting an empty graph.
    failure_counts: Dict[str, int] = {"physics": 0, "social": 0, "consequences": 0}

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
            scaffold_result = socratic_agent.run_sync(socratic_msg, deps=socratic_deps, **_user_kwargs())
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
            physics_result = physics_agent.run_sync(physics_msg, deps=physics_deps, **_user_kwargs())
            physics = physics_result.output
            log_agent_output(logger, f"PhysicsExtraction[chunk={i + 1}]", physics)
        except Exception:
            logger.exception("[Step 3a] Chunk %d FAILED — returning empty physics.", i + 1)
            physics = PhysicsExtraction()
            failure_counts["physics"] += 1

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
                physics_result = physics_agent.run_sync(retry_msg, deps=physics_deps, **_user_kwargs())
                physics = physics_result.output
                log_agent_output(logger, f"PhysicsExtraction[chunk={i + 1},retry]", physics)
            except Exception:
                logger.exception("[Step 3a] Chunk %d retry FAILED.", i + 1)

        # Retry if events were extracted but causal density is too low
        # (rule #12 — every event should participate in at least one
        # causal edge). Without this, events sit as orphan nodes with
        # no propagation effect on the world state.
        if _physics_causal_density_low(physics):
            n_evt = len(physics.events)
            n_causal = len(physics.causal_topology)
            logger.info(
                "[Step 3a] Chunk %d: low causal density (%d edges across "
                "%d events) — retrying with emphasis …",
                i + 1, n_causal, n_evt,
            )
            density_msg = (
                "IMPORTANT: The previous extraction produced "
                f"{n_evt} events but only {n_causal} causal edges. "
                "Every event MUST participate in at least one causal "
                "edge — re-extract with explicit attention to:\n"
                "  - chain_reaction edges between consecutive events,\n"
                "  - mutation edges for every event that changes a "
                "character's traits / status / location,\n"
                "  - mutation_social edges for every event that "
                "shifts a relationship axis (affinity / fear / "
                "power_dynamic), with the rel_counterpart_id and "
                "trait_target both set,\n"
                "  - affordance_gate edges for state-prerequisites,\n"
                "  - ambient_propagation for background drift.\n"
                "Aim for AT LEAST one outgoing causal edge per event "
                "and emit ALL implied mutations.\n\n" + physics_msg
            )
            try:
                density_result = physics_agent.run_sync(
                    density_msg, deps=physics_deps, **_user_kwargs(),
                )
                density_physics = density_result.output
                log_agent_output(
                    logger, f"PhysicsExtraction[chunk={i + 1},density_retry]",
                    density_physics,
                )
                # Only adopt the retry if it actually improved density
                # AND preserved the events list (we don't want a retry
                # that drops events to silently win).
                if (
                    len(density_physics.events) >= n_evt
                    and len(density_physics.causal_topology) > n_causal
                ):
                    physics = density_physics
                    logger.info(
                        "[Step 3a] Chunk %d: density retry recovered "
                        "%d→%d causal edges.",
                        i + 1, n_causal, len(physics.causal_topology),
                    )
            except Exception:
                logger.exception(
                    "[Step 3a] Chunk %d causal density retry FAILED.", i + 1,
                )

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
        # The social pass also runs when physics yielded zero events,
        # provided the chunk shows linguistic evidence of dialogue or
        # written communication — a pure-dialogue chunk (Mr Darcy's
        # letter, the radio announcement in 1984, the witches' first
        # scene) legitimately has no choices/outcomes but is exactly
        # where the channels and utterances live.
        social = SocialExtraction()
        run_social = bool(physics.events) or _chunk_likely_contains_speech(chunk)
        if run_social:
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
                previous_chunk_channels=dict(accumulated_channels),
            )
            try:
                social_result = social_agent.run_sync(social_msg, deps=social_deps, **_user_kwargs())
                social = social_result.output
                log_agent_output(logger, f"SocialExtraction[chunk={i + 1}]", social)
            except Exception:
                logger.exception("[Step 3b] Chunk %d FAILED — returning empty social.", i + 1)
                failure_counts["social"] += 1

            # Retry if zero channels AND zero utterance events with multiple events (quality gate).
            # Most narrative chunks contain at least one piece of communication, but
            # pure-action chunks (chases, silent set-pieces) do not — only retry
            # when the chunk text shows linguistic evidence of speech / writing.
            if (
                len(physics.events) >= 2
                and not social.channels
                and not social.utterance_events
                and _chunk_likely_contains_speech(chunk)
            ):
                logger.info(
                    "[Step 3b] Chunk %d: 0 channels and 0 utterance events — retrying with emphasis …",
                    i + 1,
                )
                retry_social_msg = (
                    "IMPORTANT: The previous extraction returned zero Channel "
                    "entries AND zero utterance events. Most narrative chunks "
                    "contain conversations, prophecies, letters, confessions, "
                    "orders, announcements, or rumours — each one MUST produce "
                    "either an EventNode(event_type='utterance') (for a "
                    "discrete on-page message) or a Channel (for a standing "
                    "capability such as a telephone link, mind-bond, or "
                    "classified pipeline). Re-read the text and extract ALL "
                    "information flows.\n\n" + social_msg
                )
                try:
                    retry_result = social_agent.run_sync(retry_social_msg, deps=social_deps, **_user_kwargs())
                    retry_social = retry_result.output
                    log_agent_output(logger, f"SocialExtraction[chunk={i + 1},retry]", retry_social)
                    if retry_social.channels or retry_social.utterance_events:
                        social = SocialExtraction(
                            channels=retry_social.channels or social.channels,
                            utterance_events=retry_social.utterance_events or social.utterance_events,
                            social_topology=social.social_topology,
                        )
                        logger.info(
                            "[Step 3b] Chunk %d: retry recovered %d channels, %d utterances.",
                            i + 1,
                            len(social.channels),
                            len(social.utterance_events),
                        )
                except Exception:
                    logger.exception("[Step 3b] Chunk %d info retry FAILED.", i + 1)

            # Channel under-extraction quality gate — when utterances
            # repeatedly traverse the same speaker→addressee dyad with
            # no via_channel_id and no Channel covers the pair, ask
            # the LLM to look again. This catches epistolary
            # exchanges, repeated phone calls, and standing
            # broadcasts that the agent rendered as N independent
            # speech-acts instead of a single persistent capability.
            if _social_channel_underextracted(social):
                logger.info(
                    "[Step 3b] Chunk %d: utterances cluster on a "
                    "speaker→addressee dyad with no Channel — "
                    "retrying with channel-inference emphasis …",
                    i + 1,
                )
                chn_retry_msg = (
                    "IMPORTANT: Your previous extraction emitted "
                    "MULTIPLE utterance events between the SAME speaker "
                    "and addressee(s) but no Channel that covers them, "
                    "and none of those utterances had a `via_channel_id`. "
                    "Repeated communication between the same parties is "
                    "almost always carried by a STANDING capability "
                    "(letter correspondence, telephone line, telepathic "
                    "bond, courier route, broadcast frequency). Re-emit "
                    "the extraction with: (a) at least one Channel for "
                    "each repeated dyad whose medium the text supports, "
                    "and (b) `via_channel_id` set on every utterance "
                    "that rides over one of those channels. Keep all "
                    "previously-extracted utterances and relationship "
                    "edges.\n\n" + social_msg
                )
                try:
                    chn_retry_result = social_agent.run_sync(
                        chn_retry_msg, deps=social_deps, **_user_kwargs()
                    )
                    chn_retry = chn_retry_result.output
                    log_agent_output(
                        logger,
                        f"SocialExtraction[chunk={i + 1},chn_retry]",
                        chn_retry,
                    )
                    # Only adopt if the retry strictly improves the
                    # gap: it must add at least one channel AND no
                    # longer trip the underextracted detector.
                    if (
                        len(chn_retry.channels) > len(social.channels)
                        and not _social_channel_underextracted(chn_retry)
                    ):
                        social = chn_retry
                        logger.info(
                            "[Step 3b] Chunk %d: channel-inference "
                            "retry recovered %d channels (was %d).",
                            i + 1,
                            len(social.channels),
                            len(social.channels) - 1,
                        )
                except Exception:
                    logger.exception(
                        "[Step 3b] Chunk %d channel-inference retry FAILED.",
                        i + 1,
                    )

            # Symmetric retry on empty social_topology when the chunk's
            # events involve multiple distinct entities — the per-axis
            # observation requirement makes per-edge omissions much
            # more likely under the new schema, and the social
            # propagator is a no-op without at least one edge.
            multi_entity_events = [
                e for e in physics.events
                if len(set(e.actor_ids) | set(e.target_ids)) >= 2
            ]
            if multi_entity_events and not social.social_topology:
                logger.info(
                    "[Step 3b] Chunk %d: 0 social edges across %d "
                    "multi-entity event(s) — retrying with emphasis …",
                    i + 1, len(multi_entity_events),
                )
                retry_rel_msg = (
                    "IMPORTANT: The previous extraction returned zero "
                    "RelationshipEdge entries despite the chunk containing "
                    "events with multiple distinct participants. For each "
                    "such event, infer the *minimum* relationship axes the "
                    "text supports — even one observed axis per dyad is "
                    "valuable. Use ``observed=True`` for axes the text "
                    "speaks to, and omit unobserved axes entirely (do not "
                    "fabricate neutral zeros).\n\n" + social_msg
                )
                try:
                    rel_retry_result = social_agent.run_sync(retry_rel_msg, deps=social_deps, **_user_kwargs())
                    rel_retry = rel_retry_result.output
                    log_agent_output(logger, f"SocialExtraction[chunk={i + 1},rel_retry]", rel_retry)
                    if rel_retry.social_topology:
                        social = SocialExtraction(
                            channels=social.channels,
                            utterance_events=social.utterance_events,
                            social_topology=rel_retry.social_topology,
                        )
                        logger.info(
                            "[Step 3b] Chunk %d: rel retry recovered %d social edges.",
                            i + 1, len(social.social_topology),
                        )
                except Exception:
                    logger.exception("[Step 3b] Chunk %d social retry FAILED.", i + 1)

            # Per-axis mutation_social coverage — if Social observed
            # any non-zero relationship axis but Physics never emitted
            # a matching ``mutation_social`` edge, retry physics with
            # the missing axes called out by name. Without this the
            # affective gauges (danger / conflict / power-dynamic) sit
            # at the baseline for the whole story.
            missing_axes = _physics_missing_mutation_social(physics, social)
            if missing_axes:
                logger.info(
                    "[Step 3a] Chunk %d: missing mutation_social axes %s "
                    "— retrying physics with axis-specific emphasis …",
                    i + 1, missing_axes,
                )
                axis_msg = (
                    "IMPORTANT: The Social Agent observed non-zero "
                    f"relationship reading(s) on the following axes "
                    f"but the Physics Agent emitted NO matching "
                    f"mutation_social causal edge for them: "
                    f"{missing_axes}. For each axis, find the on-page "
                    "event that produced the reading and emit a "
                    "mutation_social edge with source_id=<that event>, "
                    "target_id=<perspective entity>, "
                    "rel_counterpart_id=<other entity>, "
                    "trait_target=<axis>, and a signed trait_delta. "
                    "Keep all events and causal edges from your "
                    "previous extraction.\n\n" + physics_msg
                )
                try:
                    axis_result = physics_agent.run_sync(
                        axis_msg, deps=physics_deps, **_user_kwargs(),
                    )
                    axis_physics = axis_result.output
                    log_agent_output(
                        logger,
                        f"PhysicsExtraction[chunk={i + 1},axis_retry]",
                        axis_physics,
                    )
                    new_missing = _physics_missing_mutation_social(
                        axis_physics, social,
                    )
                    if (
                        len(axis_physics.events) >= len(physics.events)
                        and len(new_missing) < len(missing_axes)
                    ):
                        physics = axis_physics
                        logger.info(
                            "[Step 3a] Chunk %d: axis retry covered "
                            "%d/%d missing axes.",
                            i + 1,
                            len(missing_axes) - len(new_missing),
                            len(missing_axes),
                        )
                except Exception:
                    logger.exception(
                        "[Step 3a] Chunk %d axis retry FAILED.", i + 1,
                    )

            # Per-DYAD-per-axis mutation_social coverage — stronger
            # than the per-axis check above. The per-axis check only
            # asks "does *some* mutation_social edge cover this axis
            # *anywhere*?". A fixture can pass that and still leave
            # individual relationship traces flat (the bundled Star
            # Wars project: 3 axes covered globally but 6 of 9 dyads
            # had no mutation_social edge at all). The downstream
            # timeline reconstructor cannot move a metric without a
            # mutation, so the affective sub-curves for those dyads
            # render as flat lines from t=0 to t=∞. Retry once more
            # with the missing dyad×axis triples spelled out.
            missing_dyads = _physics_missing_mutation_social_per_dyad(
                physics, social,
            )
            if missing_dyads:
                # Compact summary cap to avoid prompt bloat on highly
                # social chunks; the LLM only needs a few examples to
                # generalise the pattern.
                summary_lines = [
                    f"  - dyad ({tgt} ↔ {cp}) is missing axis '{ax}'"
                    for tgt, cp, ax in missing_dyads[:25]
                ]
                more = (
                    f"\n  …and {len(missing_dyads) - 25} more"
                    if len(missing_dyads) > 25 else ""
                )
                logger.info(
                    "[Step 3a] Chunk %d: %d dyad×axis pair(s) lack a "
                    "mutation_social edge — retrying physics with "
                    "per-dyad emphasis …",
                    i + 1, len(missing_dyads),
                )
                dyad_msg = (
                    "IMPORTANT: For each of the following observed "
                    "relationship axes, the Physics Agent emitted no "
                    "mutation_social edge wired to that *specific* "
                    "dyad-axis combination. Without one, the timeline "
                    "reconstructor cannot evolve the metric and the "
                    "corresponding sub-curve in the affective dashboard "
                    "renders as a flat line.\n\n"
                    "Missing dyad×axis triples:\n"
                    + "\n".join(summary_lines) + more + "\n\n"
                    "For each missing triple, find (or invent if the "
                    "narrative implies one) the on-page event that "
                    "produced or shifted the reading, and emit a "
                    "mutation_social CausalEdge:\n"
                    "  source_id=<EVT_ id>, "
                    "causality_type='mutation_social', "
                    "target_id=<perspective entity>, "
                    "rel_counterpart_id=<other entity in the dyad>, "
                    "trait_target=<axis>, "
                    "trait_delta=<signed magnitude>.\n"
                    "Keep all events and causal edges from your "
                    "previous extraction.\n\n" + physics_msg
                )
                try:
                    dyad_result = physics_agent.run_sync(
                        dyad_msg, deps=physics_deps, **_user_kwargs(),
                    )
                    dyad_physics = dyad_result.output
                    log_agent_output(
                        logger,
                        f"PhysicsExtraction[chunk={i + 1},dyad_retry]",
                        dyad_physics,
                    )
                    new_missing_dyads = (
                        _physics_missing_mutation_social_per_dyad(
                            dyad_physics, social,
                        )
                    )
                    if (
                        len(dyad_physics.events) >= len(physics.events)
                        and len(new_missing_dyads) < len(missing_dyads)
                    ):
                        physics = dyad_physics
                        logger.info(
                            "[Step 3a] Chunk %d: dyad retry covered "
                            "%d/%d missing dyad×axis triples.",
                            i + 1,
                            len(missing_dyads) - len(new_missing_dyads),
                            len(missing_dyads),
                        )
                except Exception:
                    logger.exception(
                        "[Step 3a] Chunk %d dyad retry FAILED.", i + 1,
                    )
        else:
            logger.info("[Step 3b] Chunk %d: skipping social pass (no events, no speech cues).", i + 1)

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
                # Pass Physics events + Social utterance events — the
                # Consequences agent needs both so beliefs can reference
                # utterance ids in ``acquired_via_event_id``.
                chunk_events=list(physics.events) + list(social.utterance_events),
                chunk_causal=physics.causal_topology,
                chunk_channels=social.channels,
                chunk_utterance_events=social.utterance_events,
                previous_event_ids=all_event_ids.copy(),
            )
            try:
                consequences_result = consequences_agent.run_sync(
                    consequences_msg, deps=consequences_deps, **_user_kwargs(),
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
                failure_counts["consequences"] += 1
                consequences = ConsequencesExtraction()

            # Mutation-parity retry — Physics declared mutation edges
            # against entities, but Consequences emitted no
            # EntityUpdates for them, so the snapshot timeline will be
            # blank for those entities. Retry with the missing entity
            # ids called out by name.
            missing_entities = _consequences_mutation_parity_broken(
                physics, consequences,
            )
            if missing_entities:
                logger.info(
                    "[Step 3c] Chunk %d: mutation parity broken for "
                    "%d entit(y/ies) %s — retrying consequences …",
                    i + 1, len(missing_entities), missing_entities,
                )
                parity_msg = (
                    "IMPORTANT: The Physics Agent emitted mutation "
                    "causal edges that target the following entities, "
                    "but the previous extraction returned NO "
                    f"EntityUpdate for them: {missing_entities}. For "
                    "each one, emit at least one EntityUpdate "
                    "anchored on the triggering event's fabula_time, "
                    "with the new absolute trait values implied by "
                    "the mutation edge's trait_target / trait_delta "
                    "(and any implicit belief / status / location "
                    "changes the event causes). Keep all "
                    "EntityUpdates from your previous extraction.\n\n"
                    + consequences_msg
                )
                try:
                    parity_result = consequences_agent.run_sync(
                        parity_msg, deps=consequences_deps, **_user_kwargs(),
                    )
                    parity_consequences = parity_result.output
                    log_agent_output(
                        logger,
                        f"ConsequencesExtraction[chunk={i + 1},parity_retry]",
                        parity_consequences,
                    )
                    new_missing = _consequences_mutation_parity_broken(
                        physics, parity_consequences,
                    )
                    if len(new_missing) < len(missing_entities):
                        consequences = parity_consequences
                        entity_updates_final = consequences.entity_updates
                        logger.info(
                            "[Step 3c] Chunk %d: parity retry covered "
                            "%d/%d missing entities.",
                            i + 1,
                            len(missing_entities) - len(new_missing),
                            len(missing_entities),
                        )
                except Exception:
                    logger.exception(
                        "[Step 3c] Chunk %d parity retry FAILED.", i + 1,
                    )

        # Merge into ChunkTopology. Utterance events emitted by the
        # Social Agent are appended to the chunk's event list so they
        # participate in normal causal/temporal physics downstream.
        merged_events = _merge_utterances_into_events(
            physics.events, social.utterance_events,
            chunk_label=f"Step 3b chunk {i + 1}",
        )

        topo = ChunkTopology(
            events=merged_events,
            causal_topology=physics.causal_topology,
            channels=social.channels,
            social_topology=social.social_topology,
            spatial_topology=physics.spatial_topology,
            entity_updates=entity_updates_final,
        )
        topologies.append(topo)

        # Accumulate counters. ``prev_max_fabula`` is purely informational —
        # it tells the next chunk's prompt what the high-water mark is so
        # the LLM can place forward-marching events sensibly. We do NOT
        # shift the LLM's output, so flashbacks remain expressible.
        syuzhet_counter += len(merged_events)
        if merged_events:
            chunk_max = max(e.fabula_time for e in merged_events)
            if chunk_max > prev_max_fabula:
                prev_max_fabula = chunk_max
        all_event_ids.extend([e.id for e in merged_events])
        # Accumulate channels so subsequent chunks can reuse a CHN_ id
        # rather than reinventing the same standing capability. Newer
        # extractions overwrite older ones on id collision (the
        # cross-chunk dedup at assembly handles deeper merging).
        accumulated_channels.update(social.channels)

        # Save trailing context for next chunk's coreference overlap
        if config.chunk_overlap_chars > 0:
            prev_chunk_tail = chunk[-config.chunk_overlap_chars:]

        logger.info(
            "[Step 3] Chunk %d: %d events (%d utterances), %d causal, %d social, %d spatial, %d channels.",
            i + 1,
            len(topo.events),
            len(social.utterance_events),
            len(topo.causal_topology),
            len(topo.social_topology),
            len(topo.spatial_topology),
            len(topo.channels),
        )

    _check_chunk_failure_threshold(failure_counts, len(chunks))

    # Post-merge reconciliation: rename duplicate ``EVT_`` IDs across
    # chunks and renumber ``syuzhet_index`` globally in chunk order.
    # Without this the sync path silently kept chunk-local syuzhet
    # numbering — every chunk after the first restarted at 0 (or
    # whatever the LLM picked), so the syuzhet-axis affective views
    # showed wraparounds (e.g. Star Wars chunk 2 jumped from 33 back
    # to 1) and any duplicate ``EVT_FOO`` IDs across chunks silently
    # collided when the assembler dict-merged them. Mirrors the
    # existing async-path call at the end of ``extract_topology_async``.
    topologies = _reconcile_chunk_topologies(topologies, config)

    return topologies


def _check_chunk_failure_threshold(
    failure_counts: Dict[str, int],
    total_chunks: int,
) -> None:
    """Escalate to RuntimeError when any sub-stage failed on >50% of chunks.

    A handful of failed chunks is acceptable noise (one bad LLM round-trip,
    a transient connection drop), but if more than half the chunks failed
    a given sub-stage the resulting graph is unreliable and we'd rather
    raise loudly than silently persist a half-extracted world.
    """
    if total_chunks <= 0:
        return
    if total_chunks == 1:
        # Single-chunk runs cannot have ">half" failures by ratio, but
        # if the only chunk failed any sub-stage the resulting topology
        # is silently empty. Escalate any failure on the only chunk.
        breached = {s: c for s, c in failure_counts.items() if c >= 1}
        if breached:
            details = ", ".join(f"{s}={c}/1" for s, c in breached.items())
            raise RuntimeError(
                f"Single-chunk extraction failed sub-stages: {details}. "
                f"Refusing to return an empty topology."
            )
        return
    threshold = total_chunks / 2
    breached = {
        stage: count for stage, count in failure_counts.items()
        if count > threshold
    }
    if breached:
        details = ", ".join(
            f"{stage}={count}/{total_chunks}" for stage, count in breached.items()
        )
        raise RuntimeError(
            f"Chunk extraction failure threshold breached (>50%): {details}. "
            f"Refusing to return a partial topology — inspect upstream "
            f"agent / model errors."
        )
    if any(failure_counts.values()):
        details = ", ".join(
            f"{stage}={count}" for stage, count in failure_counts.items() if count
        )
        logger.warning(
            "[Pipeline] Chunk sub-stage failures (under threshold): %s of %d chunks.",
            details, total_chunks,
        )


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
) -> Tuple[ChunkTopology, Dict[str, int]]:
    """Process one chunk through the per-chunk agent pipeline (async).

    Runs Socratic scaffolding → Physics → Social → Consequences for a
    single chunk. Social and Consequences are sequential because
    Consequences depends on Social's ``utterance_events`` and
    ``channels`` to wire ``Belief.acquired_via_event_id`` /
    ``acquired_via_channel_id`` provenance correctly.
    ``previous_event_ids`` is empty (advisory context only; the
    ``GlobalRegister`` provides structural ID validation).

    Returns ``(topology, failure_flags)`` where ``failure_flags`` is a
    ``{stage: 0|1}`` dict so the caller can apply the >50%-of-chunks
    escalation rule (item #8 of the audit).
    """
    i = params.chunk_index
    n = params.total_chunks
    failure_flags: Dict[str, int] = {"physics": 0, "social": 0, "consequences": 0}

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
        scaffold_result = await socratic_agent.run(socratic_msg, deps=socratic_deps, **_user_kwargs())
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
        physics_result = await physics_agent.run(physics_msg, deps=physics_deps, **_user_kwargs())
        physics = physics_result.output
    except Exception:
        logger.exception("[Step 3a·Async] Chunk %d FAILED — returning empty physics.", i + 1)
        physics = PhysicsExtraction()
        failure_flags["physics"] = 1

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
            physics_result = await physics_agent.run(retry_msg, deps=physics_deps, **_user_kwargs())
            physics = physics_result.output
        except Exception:
            logger.exception("[Step 3a·Async] Chunk %d retry FAILED.", i + 1)

    # Causal-density retry — see sync path for rationale.
    if _physics_causal_density_low(physics):
        n_evt = len(physics.events)
        n_causal = len(physics.causal_topology)
        logger.info(
            "[Step 3a·Async] Chunk %d: low causal density (%d edges across "
            "%d events) — retrying with emphasis …",
            i + 1, n_causal, n_evt,
        )
        density_msg = (
            "IMPORTANT: The previous extraction produced "
            f"{n_evt} events but only {n_causal} causal edges. "
            "Every event MUST participate in at least one causal "
            "edge — re-extract with explicit attention to:\n"
            "  - chain_reaction edges between consecutive events,\n"
            "  - mutation edges for every event that changes a "
            "character's traits / status / location,\n"
            "  - mutation_social edges for every event that "
            "shifts a relationship axis (affinity / fear / "
            "power_dynamic), with the rel_counterpart_id and "
            "trait_target both set,\n"
            "  - affordance_gate edges for state-prerequisites,\n"
            "  - ambient_propagation for background drift.\n"
            "Aim for AT LEAST one outgoing causal edge per event "
            "and emit ALL implied mutations.\n\n" + physics_msg
        )
        try:
            density_result = await physics_agent.run(
                density_msg, deps=physics_deps, **_user_kwargs(),
            )
            density_physics = density_result.output
            if (
                len(density_physics.events) >= n_evt
                and len(density_physics.causal_topology) > n_causal
            ):
                physics = density_physics
                logger.info(
                    "[Step 3a·Async] Chunk %d: density retry recovered "
                    "%d→%d causal edges.",
                    i + 1, n_causal, len(physics.causal_topology),
                )
        except Exception:
            logger.exception(
                "[Step 3a·Async] Chunk %d causal density retry FAILED.", i + 1,
            )

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

    # --- Step 3b: Social Agent (must complete before Consequences) ---
    # --- Step 3c: Consequences Agent (optional, runs after Social so it
    # can wire belief provenance through utterance / channel ids) ---
    social = SocialExtraction()
    entity_updates_final = physics.entity_updates  # legacy fallback

    async def _run_social() -> SocialExtraction:
        # The social pass runs even when physics yielded zero events,
        # provided the chunk shows linguistic evidence of dialogue or
        # written communication — see comment in the sync pipeline.
        if not physics.events and not _chunk_likely_contains_speech(chunk):
            logger.info(
                "[Step 3b·Async] Chunk %d: skipping social pass (no events, no speech cues).", i + 1,
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
            social_result = await social_agent.run(social_msg, deps=social_deps, **_user_kwargs())
            local_social = social_result.output
        except Exception:
            logger.exception(
                "[Step 3b·Async] Chunk %d FAILED — returning empty social.", i + 1,
            )
            failure_flags["social"] = 1
            return SocialExtraction()

        # Retry if zero channels AND zero utterance events (quality gate)
        # AND the chunk shows linguistic evidence of dialogue. Pure-action
        # chunks (chases, silent set-pieces) legitimately produce neither.
        if (
            len(physics.events) >= 2
            and not local_social.channels
            and not local_social.utterance_events
            and _chunk_likely_contains_speech(chunk)
        ):
            logger.info(
                "[Step 3b·Async] Chunk %d: 0 channels and 0 utterance events — retrying with emphasis …",
                i + 1,
            )
            retry_social_msg = (
                "IMPORTANT: The previous extraction returned zero Channel "
                "entries AND zero utterance events. Most narrative chunks "
                "contain conversations, prophecies, letters, confessions, "
                "orders, announcements, or rumours — each one MUST produce "
                "either an EventNode(event_type='utterance') or a Channel. "
                "Re-read the text and extract ALL information flows.\n\n"
                + social_msg
            )
            try:
                retry_result = await social_agent.run(retry_social_msg, deps=social_deps, **_user_kwargs())
                retry_social = retry_result.output
                if retry_social.channels or retry_social.utterance_events:
                    local_social = SocialExtraction(
                        channels=retry_social.channels or local_social.channels,
                        utterance_events=retry_social.utterance_events or local_social.utterance_events,
                        social_topology=local_social.social_topology,
                    )
                    logger.info(
                        "[Step 3b·Async] Chunk %d: retry recovered %d channels, %d utterances.",
                        i + 1,
                        len(local_social.channels),
                        len(local_social.utterance_events),
                    )
            except Exception:
                logger.exception("[Step 3b·Async] Chunk %d info retry FAILED.", i + 1)

        # Channel under-extraction quality gate (async port of the
        # sync pipeline's check). Triggers when ≥2 utterances share a
        # speaker→addressee dyad with no `via_channel_id` and no
        # Channel covers the pair.
        if _social_channel_underextracted(local_social):
            logger.info(
                "[Step 3b·Async] Chunk %d: utterances cluster on a "
                "speaker→addressee dyad with no Channel — retrying "
                "with channel-inference emphasis …",
                i + 1,
            )
            chn_retry_msg = (
                "IMPORTANT: Your previous extraction emitted "
                "MULTIPLE utterance events between the SAME speaker "
                "and addressee(s) but no Channel that covers them, "
                "and none of those utterances had a `via_channel_id`. "
                "Repeated communication between the same parties is "
                "almost always carried by a STANDING capability "
                "(letter correspondence, telephone line, telepathic "
                "bond, courier route, broadcast frequency). Re-emit "
                "the extraction with: (a) at least one Channel for "
                "each repeated dyad whose medium the text supports, "
                "and (b) `via_channel_id` set on every utterance "
                "that rides over one of those channels. Keep all "
                "previously-extracted utterances and relationship "
                "edges.\n\n" + social_msg
            )
            try:
                chn_retry_result = await social_agent.run(
                    chn_retry_msg, deps=social_deps, **_user_kwargs(),
                )
                chn_retry = chn_retry_result.output
                if (
                    len(chn_retry.channels) > len(local_social.channels)
                    and not _social_channel_underextracted(chn_retry)
                ):
                    local_social = chn_retry
                    logger.info(
                        "[Step 3b·Async] Chunk %d: channel-inference "
                        "retry recovered %d channels.",
                        i + 1, len(local_social.channels),
                    )
            except Exception:
                logger.exception(
                    "[Step 3b·Async] Chunk %d channel-inference retry FAILED.",
                    i + 1,
                )

        # Symmetric retry on empty social_topology when the chunk's
        # events involve multiple distinct entities — ports the sync
        # extract_topology behaviour so async runs don't quietly drop
        # social-edge recall (audit item #6).
        multi_entity_events = [
            e for e in physics.events
            if len(set(e.actor_ids) | set(e.target_ids)) >= 2
        ]
        if multi_entity_events and not local_social.social_topology:
            logger.info(
                "[Step 3b·Async] Chunk %d: 0 social edges across %d "
                "multi-entity event(s) — retrying with emphasis …",
                i + 1, len(multi_entity_events),
            )
            retry_rel_msg = (
                "IMPORTANT: The previous extraction returned zero "
                "RelationshipEdge entries despite the chunk containing "
                "events with multiple distinct participants. For each "
                "such event, infer the *minimum* relationship axes the "
                "text supports — even one observed axis per dyad is "
                "valuable. Use ``observed=True`` for axes the text "
                "speaks to, and omit unobserved axes entirely (do not "
                "fabricate neutral zeros).\n\n" + social_msg
            )
            try:
                rel_retry_result = await social_agent.run(
                    retry_rel_msg, deps=social_deps, **_user_kwargs(),
                )
                rel_retry = rel_retry_result.output
                if rel_retry.social_topology:
                    local_social = SocialExtraction(
                        channels=local_social.channels,
                        utterance_events=local_social.utterance_events,
                        social_topology=rel_retry.social_topology,
                    )
                    logger.info(
                        "[Step 3b·Async] Chunk %d: rel retry recovered %d social edges.",
                        i + 1, len(local_social.social_topology),
                    )
            except Exception:
                logger.exception("[Step 3b·Async] Chunk %d social retry FAILED.", i + 1)
        return local_social

    async def _run_consequences(local_social: SocialExtraction) -> Optional[ConsequencesExtraction]:
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
            chunk_events=list(physics.events) + list(local_social.utterance_events),
            chunk_causal=physics.causal_topology,
            chunk_channels=local_social.channels,
            chunk_utterance_events=local_social.utterance_events,
            previous_event_ids=[],
        )
        try:
            consequences_result = await consequences_agent.run(
                consequences_msg, deps=consequences_deps, **_user_kwargs(),
            )
            consequences = consequences_result.output
        except Exception:
            logger.exception(
                "[Step 3c·Async] Chunk %d FAILED — falling back to physics.entity_updates.",
                i + 1,
            )
            failure_flags["consequences"] = 1
            return None

        # Mutation-parity retry — see sync path for rationale.
        missing_entities = _consequences_mutation_parity_broken(
            physics, consequences,
        )
        if missing_entities:
            logger.info(
                "[Step 3c·Async] Chunk %d: mutation parity broken for "
                "%d entit(y/ies) %s — retrying consequences …",
                i + 1, len(missing_entities), missing_entities,
            )
            parity_msg = (
                "IMPORTANT: The Physics Agent emitted mutation "
                "causal edges that target the following entities, "
                "but the previous extraction returned NO "
                f"EntityUpdate for them: {missing_entities}. For "
                "each one, emit at least one EntityUpdate "
                "anchored on the triggering event's fabula_time, "
                "with the new absolute trait values implied by "
                "the mutation edge's trait_target / trait_delta "
                "(and any implicit belief / status / location "
                "changes the event causes). Keep all "
                "EntityUpdates from your previous extraction.\n\n"
                + consequences_msg
            )
            try:
                parity_result = await consequences_agent.run(
                    parity_msg, deps=consequences_deps, **_user_kwargs(),
                )
                parity_consequences = parity_result.output
                new_missing = _consequences_mutation_parity_broken(
                    physics, parity_consequences,
                )
                if len(new_missing) < len(missing_entities):
                    consequences = parity_consequences
                    logger.info(
                        "[Step 3c·Async] Chunk %d: parity retry covered "
                        "%d/%d missing entities.",
                        i + 1,
                        len(missing_entities) - len(new_missing),
                        len(missing_entities),
                    )
            except Exception:
                logger.exception(
                    "[Step 3c·Async] Chunk %d parity retry FAILED.", i + 1,
                )
        return consequences

    social = await _run_social()

    # Per-axis mutation_social coverage — async equivalent of the sync
    # path's axis retry. Refines ``physics`` in-place before the
    # consequences pass so any newly-added mutation_social edges are
    # visible to consequences (and to the assembler downstream).
    missing_axes = _physics_missing_mutation_social(physics, social)
    if missing_axes:
        logger.info(
            "[Step 3a·Async] Chunk %d: missing mutation_social axes %s "
            "— retrying physics with axis-specific emphasis …",
            i + 1, missing_axes,
        )
        axis_msg = (
            "IMPORTANT: The Social Agent observed non-zero "
            f"relationship reading(s) on the following axes "
            f"but the Physics Agent emitted NO matching "
            f"mutation_social causal edge for them: "
            f"{missing_axes}. For each axis, find the on-page "
            "event that produced the reading and emit a "
            "mutation_social edge with source_id=<that event>, "
            "target_id=<perspective entity>, "
            "rel_counterpart_id=<other entity>, "
            "trait_target=<axis>, and a signed trait_delta. "
            "Keep all events and causal edges from your "
            "previous extraction.\n\n" + physics_msg
        )
        try:
            axis_result = await physics_agent.run(
                axis_msg, deps=physics_deps, **_user_kwargs(),
            )
            axis_physics = axis_result.output
            new_missing = _physics_missing_mutation_social(
                axis_physics, social,
            )
            if (
                len(axis_physics.events) >= len(physics.events)
                and len(new_missing) < len(missing_axes)
            ):
                physics = axis_physics
                logger.info(
                    "[Step 3a·Async] Chunk %d: axis retry covered "
                    "%d/%d missing axes.",
                    i + 1,
                    len(missing_axes) - len(new_missing),
                    len(missing_axes),
                )
        except Exception:
            logger.exception(
                "[Step 3a·Async] Chunk %d axis retry FAILED.", i + 1,
            )

    # Per-DYAD-per-axis coverage (async). Mirrors the sync path; see
    # the longer rationale in ``extract_topology``. The retry is the
    # last opportunity to anchor a specific dyad-axis combination on
    # an on-page event before consequences runs.
    missing_dyads = _physics_missing_mutation_social_per_dyad(
        physics, social,
    )
    if missing_dyads:
        summary_lines = [
            f"  - dyad ({tgt} ↔ {cp}) is missing axis '{ax}'"
            for tgt, cp, ax in missing_dyads[:25]
        ]
        more = (
            f"\n  …and {len(missing_dyads) - 25} more"
            if len(missing_dyads) > 25 else ""
        )
        logger.info(
            "[Step 3a·Async] Chunk %d: %d dyad×axis pair(s) lack a "
            "mutation_social edge — retrying physics with "
            "per-dyad emphasis …",
            i + 1, len(missing_dyads),
        )
        dyad_msg = (
            "IMPORTANT: For each of the following observed "
            "relationship axes, the Physics Agent emitted no "
            "mutation_social edge wired to that *specific* "
            "dyad-axis combination. Without one, the timeline "
            "reconstructor cannot evolve the metric and the "
            "corresponding sub-curve in the affective dashboard "
            "renders as a flat line.\n\n"
            "Missing dyad×axis triples:\n"
            + "\n".join(summary_lines) + more + "\n\n"
            "For each missing triple, find (or invent if the "
            "narrative implies one) the on-page event that "
            "produced or shifted the reading, and emit a "
            "mutation_social CausalEdge:\n"
            "  source_id=<EVT_ id>, "
            "causality_type='mutation_social', "
            "target_id=<perspective entity>, "
            "rel_counterpart_id=<other entity in the dyad>, "
            "trait_target=<axis>, "
            "trait_delta=<signed magnitude>.\n"
            "Keep all events and causal edges from your "
            "previous extraction.\n\n" + physics_msg
        )
        try:
            dyad_result = await physics_agent.run(
                dyad_msg, deps=physics_deps, **_user_kwargs(),
            )
            dyad_physics = dyad_result.output
            new_missing_dyads = (
                _physics_missing_mutation_social_per_dyad(
                    dyad_physics, social,
                )
            )
            if (
                len(dyad_physics.events) >= len(physics.events)
                and len(new_missing_dyads) < len(missing_dyads)
            ):
                physics = dyad_physics
                logger.info(
                    "[Step 3a·Async] Chunk %d: dyad retry covered "
                    "%d/%d missing dyad×axis triples.",
                    i + 1,
                    len(missing_dyads) - len(new_missing_dyads),
                    len(missing_dyads),
                )
        except Exception:
            logger.exception(
                "[Step 3a·Async] Chunk %d dyad retry FAILED.", i + 1,
            )

    consequences_out = await _run_consequences(social)
    if consequences_out is not None:
        entity_updates_final = consequences_out.entity_updates

    # Merge utterance events from the Social Agent into the chunk's event list.
    merged_events = _merge_utterances_into_events(
        physics.events, social.utterance_events,
        chunk_label=f"Step 3b·Async chunk {i + 1}",
    )

    topo = ChunkTopology(
        events=merged_events,
        causal_topology=physics.causal_topology,
        channels=social.channels,
        social_topology=social.social_topology,
        spatial_topology=physics.spatial_topology,
        entity_updates=entity_updates_final,
    )
    logger.info(
        "[Step 3·Async] Chunk %d: %d events (%d utterances), %d causal, %d social, %d spatial, %d channels.",
        i + 1,
        len(topo.events),
        len(social.utterance_events),
        len(topo.causal_topology),
        len(topo.social_topology),
        len(topo.spatial_topology),
        len(topo.channels),
    )
    return topo, failure_flags


def _reconcile_chunk_topologies(
    topologies: List[ChunkTopology],
    config: ExtractionConfig,
) -> List[ChunkTopology]:
    """Post-merge reconciliation for parallel-extracted chunk topologies.

    1. Detects and renames duplicate ``EVT_`` IDs across chunks
       (appends ``_cN`` suffix where N is the chunk index).
    2. Re-numbers ``syuzhet_index`` globally in chunk order — syuzhet IS
       narration order, so the chunk-position assignment is canonical.
       Utterance events carry their own ``syuzhet_index`` and are
       re-numbered through the same standard event-remap pass; there
       is no longer a separate ``discovered_at_syuzhet`` field on
       Channel that needs special handling.

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
    syuzhet_counter = 0
    for topo in reconciled:
        sorted_events = sorted(topo.events, key=lambda e: e.syuzhet_index)
        for evt in sorted_events:
            evt.syuzhet_index = syuzhet_counter
            syuzhet_counter += 1

    # --- Pass 3: Heal "LLM restarted fabula numbering" pathology ---
    #
    # Fabula order is intentionally free across chunks so that flashbacks
    # (chunk reaches into the past) and flash-forwards (chunk jumps to
    # the future) remain expressible. We MUST NOT blindly stack chunk N
    # after chunk N-1 — that would erase those structures.
    #
    # However, when a chunk's per-chunk extractor ignores the
    # "continue from prev_max" hint and emits small sequential integers
    # (1, 2, 3 …) instead of values aligned to ``fabula_time_spacing``,
    # those values silently collide with prior chunks once they reach
    # ``_normalize_fabula_times`` — distinct events collapse onto the
    # same normalised tick. We detect that pathology *only* when every
    # signal points to it and never when a flashback (deliberate use of
    # an earlier absolute fabula_time) is plausible:
    #
    #   1. The chunk has at least two events and its full positive
    #      fabula_time range fits inside a single ``fabula_time_spacing``
    #      slot (i.e. ``chunk_max < spacing``). A deliberate flashback
    #      uses *absolute* story-time values aligned to the global
    #      spacing — even a tightly-clustered flashback scene at
    #      fabula 100, 200, 300 with spacing 1000 keeps each event in
    #      its own slot when it eventually normalises, so this gate
    #      stays closed.
    #   2. The chunk's events are clustered as small consecutive
    #      integers (max ≤ events × 4) — the canonical "LLM gave up on
    #      the spacing hint and just counted" signature.
    #   3. Every positive fabula_time in chunk N is strictly below
    #      ``prev_max``. A flash-forward (chunk N already past
    #      prior chunks) trips this and is left alone, as is any
    #      mixed chunk with even a single event ≥ prev_max.
    #   4. The chunk has zero causal_topology edges referencing any
    #      event in a prior chunk. A deliberate flashback that
    #      revisits or causally connects to an earlier event almost
    #      always carries a ``chain_reaction`` / ``mutation`` edge
    #      linking back to the event being remembered — its absence
    #      reinforces the diagnosis of structural disconnection.
    #
    # When all four signals fire the chunk is shifted forward by
    # ``prior_max + spacing - chunk_min`` — preserving the chunk's
    # *internal* spacing (and therefore any intra-chunk ordering)
    # while placing the whole block after prior content with a
    # one-spacing gap.
    if config.fabula_time_spacing > 0 and len(reconciled) > 1:
        prior_max = 0
        prior_event_ids: set[str] = set()
        spacing = config.fabula_time_spacing
        for ci, topo in enumerate(reconciled):
            chunk_event_ids = {e.id for e in topo.events}
            chunk_fabs = [e.fabula_time for e in topo.events if e.fabula_time > 0]
            if ci == 0 or not chunk_fabs:
                if chunk_fabs:
                    prior_max = max(prior_max, max(chunk_fabs))
                prior_event_ids |= chunk_event_ids
                continue
            chunk_max = max(chunk_fabs)
            chunk_min = min(chunk_fabs)
            cross_chunk_edges = any(
                (ce.source_id in prior_event_ids) or (ce.target_id in prior_event_ids)
                for ce in topo.causal_topology
            )
            looks_restarted = (
                len(chunk_fabs) >= 2
                and chunk_max < spacing
                and chunk_max <= len(chunk_fabs) * 4
            )
            if (
                looks_restarted
                and chunk_max < prior_max
                and not cross_chunk_edges
            ):
                # Shift so chunk N starts at prior_max + spacing —
                # leaving a one-spacing gap to keep events distinguishable
                # without overstating a temporal jump.
                shift = prior_max + spacing - chunk_min
                _shift_fabula_times(topo, shift)
                logger.warning(
                    "[Reconcile] Chunk %d looked restarted (range %d-%d, "
                    "%d events, prev_max=%d, no causal links into prior "
                    "chunks) — shifted forward by %d to avoid fabula "
                    "collision.",
                    ci, chunk_min, chunk_max, len(chunk_fabs), prior_max, shift,
                )
                chunk_fabs = [e.fabula_time for e in topo.events if e.fabula_time > 0]
            if chunk_fabs:
                prior_max = max(prior_max, max(chunk_fabs))
            prior_event_ids |= chunk_event_ids

    return reconciled


def _apply_event_renames(topo: ChunkTopology, rmap: Dict[str, str]) -> ChunkTopology:
    """Apply event ID renames to all fields in a ChunkTopology.

    Channel objects do not carry event-id references and are therefore
    untouched. Utterance events live in ``topo.events`` and are renamed
    through the standard event loop.
    """
    def _r(eid: str) -> str:
        return rmap.get(eid, eid)

    def _r_list(ids: List[str]) -> List[str]:
        # ``target_ids`` may legitimately contain a renamed EVT_ ref
        # (e.g. an utterance whose target is the prior choice it
        # responds to). Preserve list order and identity for non-EVT ids.
        return [rmap.get(i, i) for i in ids]

    new_events = [
        e.model_copy(update={
            "id": _r(e.id),
            "target_ids": _r_list(e.target_ids),
            "actor_ids": _r_list(e.actor_ids),
        })
        for e in topo.events
    ]
    new_causal = [
        ce.model_copy(update={
            "source_id": _r(ce.source_id),
            "target_id": _r(ce.target_id),
        }) for ce in topo.causal_topology
    ]
    new_entity_updates = [
        eu.model_copy(update={
            "triggered_by": _r(eu.triggered_by) if eu.triggered_by else None,
            "new_beliefs": [
                b.model_copy(update={
                    "acquired_via_event_id": _r(b.acquired_via_event_id)
                    if b.acquired_via_event_id else None,
                })
                for b in eu.new_beliefs
            ],
        }) for eu in topo.entity_updates
    ]
    return ChunkTopology(
        events=new_events,
        causal_topology=new_causal,
        channels=topo.channels,
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
    for ch in topo.channels.values():
        if ch.established_at_fabula > 0:
            ch.established_at_fabula += shift
        if ch.terminated_at_fabula is not None and ch.terminated_at_fabula > 0:
            ch.terminated_at_fabula += shift
    for se in topo.social_topology:
        # last_updated_fabula is now per-axis; shift each observed
        # metric independently to preserve relative ordering.
        for m in se.metrics.values():
            if m.last_updated_fabula > 0:
                m.last_updated_fabula += shift
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

    async def _guarded_extract(chunk: str, params: _ChunkParams) -> Tuple[ChunkTopology, Dict[str, int]]:
        async with semaphore:
            return await _extract_single_chunk_async(
                chunk, params, register, config,
                socratic_agent, physics_agent, social_agent,
                consequences_agent,
            )

    chunk_results = await asyncio.gather(*[
        _guarded_extract(chunk, params)
        for chunk, params in zip(chunks, params_list)
    ])
    topologies_list = [r[0] for r in chunk_results]
    failure_counts: Dict[str, int] = {"physics": 0, "social": 0, "consequences": 0}
    for _, flags in chunk_results:
        for stage, flag in flags.items():
            failure_counts[stage] += flag
    _check_chunk_failure_threshold(failure_counts, len(chunks))

    # Post-merge reconciliation
    topologies_list = _reconcile_chunk_topologies(topologies_list, config)

    logger.info(
        "[Step 3·Async] All %d chunks extracted and reconciled.", len(topologies_list),
    )
    return topologies_list


# =====================================================================
# Step 3d — Optional Research Extraction (segregated)
# =====================================================================

def _build_research_agent(
    config: ExtractionConfig,
) -> Agent[None, "WorldFact"]:
    """Construct the Step 3d Research-Extraction Agent.

    The agent's output is a single ``WorldFact``. It is forbidden from
    emitting Entities, Events, or any topology — Pydantic's NativeOutput
    on ``WorldFact`` enforces that structurally; the prompt reinforces it.
    """
    from shadow_loom.research import WorldFact  # local to avoid top-level cycle risk

    agent: Agent[None, WorldFact] = Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldFact),
        system_prompt=_load_prompt("research_extraction.md"),
        retries=config.output_retries,
    )
    return agent


_FACT_ID_RE = re.compile(r"^FACT_(\d+)$")


def _next_fact_index(facts: List[Any]) -> int:
    """Return the next collision-free numeric suffix for a FACT_ id.

    Uses ``max(existing numeric suffix) + 1`` rather than ``len + 1``
    so sparse / hand-edited fact lists don't generate ids that collide
    with surviving entries (e.g. FACT_001, FACT_003 \u2192 next-by-length
    would re-emit FACT_003).
    """
    highest = 0
    for f in facts:
        fid = getattr(f, "id", None)
        if not isinstance(fid, str):
            continue
        m = _FACT_ID_RE.match(fid)
        if m:
            try:
                highest = max(highest, int(m.group(1)))
            except ValueError:
                continue
    return highest + 1


def _run_research_step(
    world_state: WorldStateV1,
    config: ExtractionConfig,
) -> WorldStateV1:
    """Step 3d (sync): for each configured topic, query provider + distil to WorldFact.

    Pure additive: only ``world_state.world_facts`` is mutated; entities,
    events and edges are untouched. Failures (provider error, agent
    refusal, empty results) are logged and skipped — research is best-
    effort and never fails extraction.
    """
    if not config.enable_research_agent:
        return world_state
    if not config.research_topics:
        logger.info("[Pipeline·Research] enabled but research_topics is empty — skipping.")
        return world_state

    from shadow_loom.research import (
        ResearchSnippet,
        build_provider,
    )

    try:
        # ``build_provider`` accepts a small fixed kwarg set
        # (``api_key``, ``search_depth``) plus provider-specific extras
        # forwarded into the provider constructor. Historically we
        # passed ``provider_model`` / ``max_results`` here, but the
        # Tavily constructor accepts neither and the call would raise
        # silently into the broad ``except`` below — making research
        # "work" only on the no-op NullProvider. ``provider_model`` is
        # used as the Tavily search-depth ("basic" or "advanced");
        # ``max_results`` is per-search and is forwarded into the
        # ``provider.search`` call below instead of construction.
        provider_kwargs: dict = {}
        depth = (config.research_provider_model or "").strip().lower()
        if depth in {"basic", "advanced"}:
            provider_kwargs["search_depth"] = depth
        provider = build_provider(
            config.research_provider,
            **provider_kwargs,
        )
    except Exception:
        logger.exception("[Pipeline·Research] failed to build provider — skipping.")
        return world_state

    agent = _build_research_agent(config)
    # Compute the next FACT id from the highest existing numeric suffix
    # rather than ``len(world_facts) + 1``. The latter collides when
    # facts have been hand-edited / partially deleted upstream and the
    # surviving id sequence is sparse (e.g. FACT_001, FACT_003 → next
    # by length is FACT_003 again).
    next_idx = _next_fact_index(world_state.world_facts)

    for topic in config.research_topics:
        try:
            snippets: List[ResearchSnippet] = list(
                provider.search(
                    topic,
                    max_results=config.research_max_results_per_query,
                )
            )
        except Exception:
            logger.exception("[Pipeline·Research] provider.search failed for topic=%r", topic)
            continue

        if not snippets:
            logger.info("[Pipeline·Research] no snippets for topic=%r — skipping.", topic)
            continue

        # Build the user message — minimal, structured.
        snippet_block = "\n\n".join(
            f"[{i+1}] {s.title}\nURL: {s.url}\n{s.content}"
            for i, s in enumerate(snippets)
        )
        user_msg = (
            f"Topic: {topic}\n\n"
            f"Snippets ({len(snippets)}):\n\n{snippet_block}\n\n"
            "Distil the above into a single WorldFact per the system prompt."
        )

        try:
            result = agent.run_sync(user_msg, **_user_kwargs())
            fact: WorldFact = result.output
        except Exception:
            logger.exception("[Pipeline·Research] agent failed for topic=%r — skipping.", topic)
            continue

        # Stamp pipeline-controlled fields the agent doesn't get to choose.
        fact_id = f"FACT_{next_idx:03d}"
        next_idx += 1
        stamped = fact.model_copy(update={
            "id": fact_id,
            "topic": topic,
            "provider": config.research_provider,
            "raw_snippets": snippets,
        })
        world_state.world_facts.append(stamped)
        logger.info(
            "[Pipeline·Research] +WorldFact %s topic=%r confidence=%s",
            fact_id, topic, stamped.confidence,
        )

    return world_state


async def _run_research_step_async(
    world_state: WorldStateV1,
    config: ExtractionConfig,
) -> WorldStateV1:
    """Async variant of ``_run_research_step``.

    Topics are processed sequentially (research is naturally low-volume
    and provider rate limits are typically the bottleneck, not local
    concurrency).
    """
    if not config.enable_research_agent:
        return world_state
    if not config.research_topics:
        logger.info("[Pipeline·Research·Async] enabled but no topics — skipping.")
        return world_state

    from shadow_loom.research import (
        ResearchSnippet,
        build_provider,
    )

    try:
        # See ``_run_research_step`` for why we don't pass
        # provider_model / max_results into ``build_provider``.
        provider_kwargs: dict = {}
        depth = (config.research_provider_model or "").strip().lower()
        if depth in {"basic", "advanced"}:
            provider_kwargs["search_depth"] = depth
        provider = build_provider(
            config.research_provider,
            **provider_kwargs,
        )
    except Exception:
        logger.exception("[Pipeline·Research·Async] failed to build provider — skipping.")
        return world_state

    agent = _build_research_agent(config)
    next_idx = _next_fact_index(world_state.world_facts)

    for topic in config.research_topics:
        try:
            # Provider.search is sync (Tavily client is sync); run in thread.
            snippets: List[ResearchSnippet] = list(
                await asyncio.to_thread(
                    provider.search,
                    topic,
                    config.research_max_results_per_query,
                )
            )
        except Exception:
            logger.exception("[Pipeline·Research·Async] provider.search failed for topic=%r", topic)
            continue

        if not snippets:
            continue

        snippet_block = "\n\n".join(
            f"[{i+1}] {s.title}\nURL: {s.url}\n{s.content}"
            for i, s in enumerate(snippets)
        )
        user_msg = (
            f"Topic: {topic}\n\n"
            f"Snippets ({len(snippets)}):\n\n{snippet_block}\n\n"
            "Distil the above into a single WorldFact per the system prompt."
        )

        try:
            result = await agent.run(user_msg, **_user_kwargs())
            fact: WorldFact = result.output
        except Exception:
            logger.exception("[Pipeline·Research·Async] agent failed for topic=%r — skipping.", topic)
            continue

        fact_id = f"FACT_{next_idx:03d}"
        next_idx += 1
        stamped = fact.model_copy(update={
            "id": fact_id,
            "topic": topic,
            "provider": config.research_provider,
            "raw_snippets": snippets,
        })
        world_state.world_facts.append(stamped)
        logger.info(
            "[Pipeline·Research·Async] +WorldFact %s topic=%r confidence=%s",
            fact_id, topic, stamped.confidence,
        )

    return world_state


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

    KNOWN LIMITATION (cross-chunk fabula collision): the mapping is
    keyed on raw fabula_time alone. When chunk 2's LLM ignores the
    "max so far" hint and restarts numbering at 1, chunk 1's
    ``fabula_time=1`` and chunk 2's ``fabula_time=1`` would
    deduplicate to the same key and collapse onto the same normalised
    tick. The chunk-restart healer in
    :func:`_reconcile_chunk_topologies` (Pass 3, added May 2026)
    detects this pattern *before* normalisation runs and shifts the
    affected chunk forward — preserving its internal spacing and
    leaving genuine flashbacks (chunks linked causally to prior
    events) and flash-forwards (chunks already past prior_max)
    untouched. A stray restart that slips past the heuristic (e.g.
    a multi-restart cascade across many chunks, or a chunk that
    coincidentally meets all flashback signals) can still merge
    unrelated events; re-ingesting under the strengthened prompts
    remains the recommended remedy for fixtures predating that
    healer.
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
    for ch in ws.channels.values():
        all_times.add(ch.established_at_fabula)
        if ch.terminated_at_fabula is not None:
            all_times.add(ch.terminated_at_fabula)
    for se in ws.spatial_topology:
        all_times.add(se.established_at_fabula)
        if se.destroyed_at_fabula is not None:
            all_times.add(se.destroyed_at_fabula)
    for re_edge in ws.social_topology:
        for m in re_edge.metrics.values():
            all_times.add(m.last_updated_fabula)
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
    new_channels = {
        cid: ch.model_copy(update={
            "established_at_fabula": _map(ch.established_at_fabula) or ch.established_at_fabula,
            "terminated_at_fabula": _map(ch.terminated_at_fabula),
        })
        for cid, ch in ws.channels.items()
    }
    new_social = []
    for re_edge in ws.social_topology:
        # Remap each per-axis last_updated_fabula independently.
        new_metrics = {
            name: m.model_copy(update={
                "last_updated_fabula": _map(m.last_updated_fabula) or m.last_updated_fabula,
            })
            for name, m in re_edge.metrics.items()
        }
        new_social.append(re_edge.model_copy(update={"metrics": new_metrics}))
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
        channels=new_channels,
        social_topology=new_social,
    )


# =====================================================================
# Step 3 — Assembly + Validation
# =====================================================================


def _merge_utterances_into_events(
    physics_events: List[EventNode],
    utterance_events: List[EventNode],
    *,
    chunk_label: str,
) -> List[EventNode]:
    """Append Social-agent utterance events onto the Physics event list.

    Drops any utterance whose id collides with a Physics event id (the
    validator ought to have renamed it via the EVT_UTT_ prefix rule, but
    we belt-and-brace here so that downstream code never sees a
    duplicate id). Returns a new list; ``physics_events`` is not
    mutated.
    """
    merged: List[EventNode] = list(physics_events)
    if not utterance_events:
        return merged
    existing_ids = {e.id for e in merged}
    for uev in utterance_events:
        if uev.id in existing_ids:
            logger.info(
                "[%s] dropping utterance '%s' — id collides with a Physics event.",
                chunk_label, uev.id,
            )
            continue
        merged.append(uev)
        existing_ids.add(uev.id)
    return merged


def _deduplicate_social(edges: List[RelationshipEdge]) -> List[RelationshipEdge]:
    """Merge multiple edges for the same (source, target) pair into one.

    Under the per-metric ``RelationshipEdge`` schema, two extractions of
    the same dyad may carry *different* observed axes (e.g. one chunk
    only mutated ``fear``, another only ``power_dynamic``). Naively
    keeping the most recent whole edge would discard the older axis.
    Instead we merge per-axis, picking the metric with the larger
    ``last_updated_fabula`` for each axis independently.
    """
    best: dict[tuple[str, str], RelationshipEdge] = {}
    for e in edges:
        key = (e.source_entity_id, e.target_entity_id)
        if key not in best:
            best[key] = e
            continue
        merged_metrics = dict(best[key].metrics)
        for name, m in e.metrics.items():
            existing = merged_metrics.get(name)
            # ``>=`` (rather than ``>``) so a later-appended metric at
            # the same fabula tick wins. Pipeline appends the sandbox
            # bridge's ``RelationshipEdge``s *after* the prose
            # extractor's edges, so this lets the deterministic
            # physics value override the LLM extractor's reading on
            # ties without needing to fudge ``last_updated_fabula``
            # into the future (which would break time-slicing).
            if existing is None or m.last_updated_fabula >= existing.last_updated_fabula:
                merged_metrics[name] = m
        best[key] = best[key].model_copy(update={"metrics": merged_metrics})
    return list(best.values())


def _mirror_missing_relationship_directions(
    edges: List[RelationshipEdge],
) -> List[RelationshipEdge]:
    """Deprecated shim — mirroring now happens automatically inside
    ``WorldStateV1``'s post-init validator
    (:meth:`WorldStateV1._mirror_missing_relationship_directions`).

    Kept here as a thin wrapper because a handful of internal callers
    (and possibly downstream tests) imported the function directly.
    Constructs a transient ``WorldStateV1``-shaped pass through the
    standalone mirror logic so the behaviour stays identical.
    """
    if not edges:
        return edges
    indexed: dict[tuple[str, str], RelationshipEdge] = {
        (e.source_entity_id, e.target_entity_id): e for e in edges
    }
    mirrored: list[RelationshipEdge] = []
    for (src, tgt), edge in list(indexed.items()):
        if (tgt, src) in indexed:
            continue
        new_metrics: dict[str, dict] = {}
        for name, m in edge.metrics.items():
            value = float(m.value)
            if name == "power_dynamic":
                value = -value
            new_metrics[name] = {
                "value": value,
                "inertia": float(m.inertia),
                "evidence_strength": "weak",
                "last_updated_fabula": int(m.last_updated_fabula),
                "observed": False,
            }
        if not new_metrics:
            continue
        try:
            mirror = RelationshipEdge(
                world_id=edge.world_id,
                source_entity_id=tgt,
                target_entity_id=src,
                metrics=new_metrics,  # type: ignore[arg-type]
            )
        except Exception:
            logger.warning(
                "Failed to mirror RelationshipEdge %s→%s; leaving "
                "reverse direction missing.", src, tgt, exc_info=True,
            )
            continue
        mirrored.append(mirror)
        indexed[(tgt, src)] = mirror
    return list(edges) + mirrored


def _deduplicate_spatial(edges: List[SpatialEdge]) -> List[SpatialEdge]:
    """Merge spatial edges per (source, target), preserving lifecycle state.

    Two extractor passes can describe the same passage with different
    lifecycle facts: one chunk may report the door as initially
    traversable (``established_at_fabula=0``), a later chunk may report
    it as locked from a particular tick (``is_locked=True``,
    ``barrier_item_id``), and a still-later chunk may report it as
    destroyed (``destroyed_at_fabula``). The previous implementation
    keyed only on ``(source, target)`` and let the latest-established
    edge win, silently dropping the lock and destruction facts.

    Strategy: keep the *earliest-established* edge per pair (so the
    passage's birth tick is preserved) and merge in any subsequent
    edge's lock and destruction facts.
    """
    by_pair: dict[tuple[str, str], List[SpatialEdge]] = {}
    for e in edges:
        by_pair.setdefault((e.source_id, e.target_id), []).append(e)
    merged: List[SpatialEdge] = []
    for pair, group in by_pair.items():
        # Sort by established_at_fabula so the earliest is canonical.
        group.sort(key=lambda e: e.established_at_fabula)
        canonical = group[0]
        update: dict = {}
        # Lock semantics: any pass reporting locked=True wins (a
        # passage explicitly described as locked at any point should
        # not be silently treated as freely traversable).
        for e in group[1:]:
            if e.is_locked and not canonical.is_locked:
                update["is_locked"] = True
                if e.barrier_item_id and not canonical.barrier_item_id:
                    update["barrier_item_id"] = e.barrier_item_id
            elif e.barrier_item_id and not canonical.barrier_item_id and not update.get("barrier_item_id"):
                update["barrier_item_id"] = e.barrier_item_id
        # Destruction: take the earliest non-null destroyed_at_fabula
        # across the group (if multiple chunks report destruction the
        # earliest tick is the one that fires).
        destroyed_ticks = [
            e.destroyed_at_fabula for e in group
            if e.destroyed_at_fabula is not None
        ]
        if destroyed_ticks:
            earliest = min(destroyed_ticks)
            if canonical.destroyed_at_fabula is None or earliest < canonical.destroyed_at_fabula:
                update["destroyed_at_fabula"] = earliest
        merged.append(canonical.model_copy(update=update) if update else canonical)
    return merged


def _deduplicate_causal(
    edges: List[CausalEdge],
    *,
    fabula_tolerance: int = 1,
) -> List[CausalEdge]:
    """Deduplicate causal edges by full semantic identity.

    Two ``mutation`` edges from the same event onto the same target
    entity but with *different ``trait_target``* (e.g. EVT_MURDER
    → ENT_MACBETH on ``guilt`` vs on ``ambition``) are NOT
    duplicates — they describe different state changes and must both
    survive. The same applies to ``mutation_social`` edges that
    differ on ``rel_counterpart_id`` (Macbeth's bond toward Banquo
    vs toward Lady Macbeth) and to edges that differ on
    ``mechanism`` (a kinetic vs psychological consequence of the
    same trigger).

    Keeps the edge with the highest ``causal_force`` when duplicates
    are found (the stronger signal wins).

    A second pass collapses neighbouring duplicates whose
    ``fabula_time`` differs by at most ``fabula_tolerance`` ticks
    (default ``1``). This absorbs LLM tick-jitter on re-extraction
    of the same causal arc, which previously survived as separate
    edges because the exact ``fabula_time`` differed by a single
    tick. Set ``fabula_tolerance=0`` to restore strict dedup.
    """
    def _key(e: CausalEdge) -> tuple:
        return (
            e.source_id,
            e.target_id,
            e.causality_type,
            e.trait_target,
            e.rel_counterpart_id,
            e.mechanism,
            e.fabula_time,
        )

    best: dict[tuple, CausalEdge] = {}
    for e in edges:
        key = _key(e)
        if key not in best or e.causal_force > best[key].causal_force:
            best[key] = e
    deduped = list(best.values())
    if fabula_tolerance <= 0:
        return deduped

    # Second pass: collapse neighbouring (source, target, type,
    # trait_target, rel_counterpart_id, mechanism) edges whose
    # fabula_time is within tolerance, keeping the higher force.
    deduped.sort(key=lambda e: (
        e.source_id, e.target_id, e.causality_type,
        e.trait_target or "", e.rel_counterpart_id or "", e.mechanism,
        e.fabula_time,
    ))
    collapsed: List[CausalEdge] = []
    for e in deduped:
        if collapsed:
            prev = collapsed[-1]
            if (
                prev.source_id == e.source_id
                and prev.target_id == e.target_id
                and prev.causality_type == e.causality_type
                and prev.trait_target == e.trait_target
                and prev.rel_counterpart_id == e.rel_counterpart_id
                and prev.mechanism == e.mechanism
                and abs(e.fabula_time - prev.fabula_time) <= fabula_tolerance
            ):
                if e.causal_force > prev.causal_force:
                    collapsed[-1] = e
                continue
        collapsed.append(e)
    return collapsed


def _deduplicate_channels_with_map(
    channel_dicts: List[Dict[str, Channel]],
) -> Tuple[Dict[str, Channel], Dict[str, str]]:
    """Merge per-chunk Channel dicts and also return an old→canonical id map.

    Keyed on ``(medium, sorted(participant_ids), directionality,
    established_at_fabula)`` rather than the LLM-generated ``CHN_``
    id, because two chunks may each invent their own id for the
    same standing capability. Directionality is part of the key so
    a duplex channel and a broadcast channel (e.g. a public
    proclamation vs a private chat) over the same participants are
    NOT collapsed.

    When two chunks describe the same channel, ``intelligibility``
    maps are *merged* per-participant (later wins on collisions);
    if both are non-empty the merged result preserves keys from
    both chunks. ``terminated_at_fabula`` collapses to the earliest
    non-null tick (the channel actually goes dead at the first
    reported termination).

    The returned ``forwarding_map`` lets callers rewrite every
    ``EventNode.via_channel_id`` and ``Belief.acquired_via_channel_id``
    that pointed at a now-collapsed id, so dedup never silently orphans
    those references (which used to be nulled by ``_auto_repair``).
    """
    best: dict[tuple, Channel] = {}
    # Track every id ever seen for each shape-key so the forwarding map
    # covers every collapsed alias, not just the most recent.
    aliases: dict[tuple, list[str]] = {}
    for chunk_channels in channel_dicts:
        for ch in chunk_channels.values():
            key = (
                ch.medium,
                tuple(sorted(ch.participant_ids)),
                ch.directionality,
                ch.established_at_fabula,
            )
            aliases.setdefault(key, []).append(ch.id)
            existing = best.get(key)
            if existing is None:
                best[key] = ch
                continue
            # Merge intelligibility maps (union of keys; later value
            # wins on key collision so the more recent extraction's
            # decode probability survives).
            merged_intel = dict(existing.intelligibility)
            merged_intel.update(ch.intelligibility)
            # Earliest non-null termination wins.
            term_candidates = [
                t for t in (existing.terminated_at_fabula, ch.terminated_at_fabula)
                if t is not None
            ]
            merged_term: Optional[int] = min(term_candidates) if term_candidates else None
            best[key] = ch.model_copy(update={
                "intelligibility": merged_intel,
                "terminated_at_fabula": merged_term,
            })
    deduped = {ch.id: ch for ch in best.values()}
    forwarding: Dict[str, str] = {}
    for key, ids in aliases.items():
        canonical = best[key].id
        for old in ids:
            if old != canonical:
                forwarding[old] = canonical
    return deduped, forwarding


def _deduplicate_channels(channel_dicts: List[Dict[str, Channel]]) -> Dict[str, Channel]:
    """Backwards-compatible shim: returns just the deduped dict.

    Prefer :func:`_deduplicate_channels_with_map` at call sites that
    can apply the forwarding map to ``via_channel_id`` /
    ``acquired_via_channel_id`` references.
    """
    deduped, _ = _deduplicate_channels_with_map(channel_dicts)
    return deduped


def _apply_channel_forwarding(
    forwarding: Dict[str, str],
    *,
    events: List[EventNode],
    entity_updates: Optional[List["EntityUpdate"]] = None,
) -> None:
    """Rewrite ``via_channel_id`` and ``Belief.acquired_via_channel_id`` in place.

    No-op when ``forwarding`` is empty. ``events`` and any
    ``entity_updates[*].new_beliefs`` lists are mutated; their
    container objects are replaced via ``model_copy`` so we don't rely
    on Pydantic's mutability semantics for nested models.
    """
    if not forwarding:
        return
    for i, evt in enumerate(events):
        if evt.via_channel_id and evt.via_channel_id in forwarding:
            events[i] = evt.model_copy(update={
                "via_channel_id": forwarding[evt.via_channel_id],
            })
    if entity_updates:
        for j, eu in enumerate(entity_updates):
            new_beliefs = eu.new_beliefs
            replaced_any = False
            rebuilt: List[Belief] = []
            for b in new_beliefs:
                if (
                    b.acquired_via_channel_id
                    and b.acquired_via_channel_id in forwarding
                ):
                    rebuilt.append(b.model_copy(update={
                        "acquired_via_channel_id": forwarding[b.acquired_via_channel_id],
                    }))
                    replaced_any = True
                else:
                    rebuilt.append(b)
            if replaced_any:
                entity_updates[j] = eu.model_copy(update={"new_beliefs": rebuilt})


# Public aliases for reuse outside the ingestion pipeline
deduplicate_social = _deduplicate_social
deduplicate_spatial = _deduplicate_spatial
deduplicate_causal = _deduplicate_causal
deduplicate_channels = _deduplicate_channels


def _snapshot_sort_key(s) -> tuple:
    """Stable, deterministic sort key for snapshot lists.

    Primary: ``fabula_time``. Secondary keys break ties when two extraction
    runs (or correction patches) produce snapshots with identical fabula
    times — without them, sort order depends on insertion order, which is
    non-deterministic under async chunk processing. Works for both
    :class:`EntityStateSnapshot` and :class:`WorldTraitSnapshot`.
    """
    return (
        getattr(s, "fabula_time", 0),
        getattr(s, "triggered_by", None) or "",
        getattr(s, "status", None) or "",
        getattr(s, "location_id", None) or "",
        len(getattr(s, "traits", None) or {}),
        len(getattr(s, "beliefs_added", None) or []),
    )


def _coalesce_snapshots(
    snaps: List[EntityStateSnapshot],
) -> List[EntityStateSnapshot]:
    """Merge same-fabula_time snapshots into a single deterministic snap.

    When two chunks emit an EntityUpdate for the same (entity,
    fabula_time) pair, the resulting EntityStateSnapshot list contains
    both records and replay-order becomes extraction-order dependent
    (audit item #10). This coalesces them per-tick using a stable rule
    set and returns the list time-sorted. Order within the same tick
    is preserved: the *first* snapshot at a tick keeps its position
    after merging.
    """
    if not snaps:
        return snaps
    by_tick: Dict[int, List[EntityStateSnapshot]] = {}
    order: List[int] = []
    for s in snaps:
        if s.fabula_time not in by_tick:
            order.append(s.fabula_time)
            by_tick[s.fabula_time] = []
        by_tick[s.fabula_time].append(s)

    coalesced: List[EntityStateSnapshot] = []
    for tick in sorted(set(order)):
        group = by_tick[tick]
        if len(group) == 1:
            coalesced.append(group[0])
            continue
        # All snapshots in a group MUST share a world_id; if a chunk
        # mixed factual + shadow snapshots at the same tick that is
        # itself a bug we want surfaced loudly rather than silently
        # retagged. Default to the group's first world_id.
        group_world_ids = {getattr(s, "world_id", "factual") for s in group}
        if len(group_world_ids) > 1:
            logger.warning(
                "_coalesce_snapshots: tick %d has mixed world_ids %s; "
                "keeping snapshots un-merged to preserve branch tagging.",
                tick, group_world_ids,
            )
            coalesced.extend(group)
            continue
        merged_world_id = next(iter(group_world_ids))
        # Merge fields:
        merged_traits: dict = {}
        merged_beliefs_added: List[Belief] = []
        seen_belief_keys: set = set()
        merged_invalidated: List[str] = []
        seen_invalid: set = set()
        first_triggered_by: Optional[str] = None
        first_status: Optional[str] = None
        first_location_id: Optional[str] = None
        for s in group:
            # later wins per key — chunks ordered by extraction so
            # later chunks describe later narration of the same tick
            for k, v in (s.traits or {}).items():
                merged_traits[k] = v
            for b in s.beliefs_added or []:
                # Dedup beliefs by (target_id, perceived_state,
                # acquired_via_event_id, acquired_via_channel_id) so
                # the *same* belief emitted twice is collapsed but two
                # acquisitions of the same proposition through
                # different provenance (e.g. directly witnessed AND
                # later told) are both kept.
                key = (
                    b.target_id,
                    b.perceived_state,
                    getattr(b, "acquired_via_event_id", None),
                    getattr(b, "acquired_via_channel_id", None),
                )
                if key in seen_belief_keys:
                    continue
                seen_belief_keys.add(key)
                merged_beliefs_added.append(b)
            for tgt in s.beliefs_invalidated or []:
                if tgt in seen_invalid:
                    continue
                seen_invalid.add(tgt)
                merged_invalidated.append(tgt)
            if first_triggered_by is None and s.triggered_by:
                first_triggered_by = s.triggered_by
            elif (
                s.triggered_by
                and first_triggered_by
                and s.triggered_by != first_triggered_by
            ):
                logger.warning(
                    "_coalesce_snapshots: tick %d has conflicting "
                    "triggered_by values (%s vs %s); keeping first.",
                    tick, first_triggered_by, s.triggered_by,
                )
            if first_status is None and s.status:
                first_status = s.status
            if first_location_id is None and s.location_id:
                first_location_id = s.location_id

        coalesced.append(EntityStateSnapshot(
            world_id=merged_world_id,
            fabula_time=tick,
            triggered_by=first_triggered_by,
            traits=merged_traits,
            beliefs_added=merged_beliefs_added,
            beliefs_invalidated=merged_invalidated,
            status=first_status,
            location_id=first_location_id,
        ))
    return coalesced


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
    channel_dicts: List[Dict[str, Channel]] = []
    social_topology: List[RelationshipEdge] = []
    spatial_topology: List[SpatialEdge] = []

    for topo in topologies:
        events.extend(topo.events)
        causal_topology.extend(topo.causal_topology)
        channel_dicts.append(topo.channels)
        social_topology.extend(topo.social_topology)
        spatial_topology.extend(topo.spatial_topology)

    # --- Channel dedup with forwarding map ---
    # Done up-front so we can rewrite stale via_channel_id /
    # acquired_via_channel_id references on events and entity_updates
    # before they get baked into the world state. Otherwise dedup would
    # silently orphan those references and ``_auto_repair`` would null
    # them out (lossy).
    raw_channel_count = sum(len(c) for c in channel_dicts)
    channels, channel_forwarding = _deduplicate_channels_with_map(channel_dicts)
    if channel_forwarding:
        # Rewrite events first (utterance.via_channel_id), then
        # mutate each topology's entity_updates so their beliefs pick
        # up the new channel ids before being folded into snapshots.
        _apply_channel_forwarding(channel_forwarding, events=events)
        for topo in topologies:
            _apply_channel_forwarding(
                channel_forwarding,
                events=[],  # events already covered globally
                entity_updates=topo.entity_updates,
            )
        logger.info(
            "[Step 3] Channel dedup forwarding: %d alias(es) rewritten.",
            len(channel_forwarding),
        )

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

    # Coalesce same-(entity, fabula_time) snapshots into a single
    # deterministic snapshot. Without this, two updates emitted by
    # different chunks for the same tick get appended verbatim and
    # replay order becomes extraction-order dependent \u2014 producing
    # unstable reconstructed state and double-applied belief mutations
    # (audit item #10). Merge rules per field:
    #   - traits: later (later in input order) wins per key
    #   - beliefs_added: union, deduped by target_id
    #   - beliefs_invalidated: union
    #   - triggered_by / status / location_id: first non-null wins
    for eid, snaps in all_entity_updates.items():
        all_entity_updates[eid] = _coalesce_snapshots(snaps)

    # Sort events chronologically. Add stable secondary keys so two
    # extraction runs over the same input produce byte-identical AMWN
    # ordering even when several events share a fabula tick (a common
    # case at chapter boundaries where multiple things happen "now").
    # Without these tie-breakers ordering depends on chunk-extraction
    # insertion order, which under async parallelism is itself
    # non-deterministic.
    events.sort(key=lambda e: (e.fabula_time, e.id))
    causal_topology.sort(
        key=lambda c: (c.fabula_time, c.source_id, c.target_id, c.causality_type)
    )

    # Deduplicate relationship, spatial, causal across chunks (channels
    # were deduped earlier so the forwarding map could rewrite events).
    # Reverse-direction mirroring of one-sided dyads now happens in
    # ``WorldStateV1``'s post-init validator so every consumer (ingestion,
    # example_worlds fixtures, snapshot reconstructions, test fixtures)
    # sees the same mirrored shape — no explicit call here.
    social_before = len(social_topology)
    social_topology = _deduplicate_social(social_topology)
    spatial_before = len(spatial_topology)
    spatial_topology = _deduplicate_spatial(spatial_topology)
    causal_before = len(causal_topology)
    causal_topology = _deduplicate_causal(causal_topology)
    deduped_parts = []
    if social_before != len(social_topology):
        deduped_parts.append(f"social {social_before}→{len(social_topology)}")
    if spatial_before != len(spatial_topology):
        deduped_parts.append(f"spatial {spatial_before}→{len(spatial_topology)}")
    if causal_before != len(causal_topology):
        deduped_parts.append(f"causal {causal_before}→{len(causal_topology)}")
    if raw_channel_count != len(channels):
        deduped_parts.append(f"channels {raw_channel_count}→{len(channels)}")
    if deduped_parts:
        logger.info("[Step 3] Deduplicated edges: %s.", ", ".join(deduped_parts))

    ws = WorldStateV1(
        locations=register.locations,
        objects=register.objects,
        entities={
            eid: (
                ent.model_copy(update={"state_timeline": sorted(all_entity_updates[eid], key=_snapshot_sort_key)})
                if eid in all_entity_updates
                else ent
            )
            for eid, ent in register.entities.items()
        },
        world_traits=register.world_traits,
        events=events,
        causal_topology=causal_topology,
        spatial_topology=spatial_topology,
        channels=channels,
        social_topology=social_topology,
    )
    utterance_count = sum(1 for e in events if e.event_type == "utterance")
    logger.info(
        "[Step 3] Assembled WorldStateV1 — %d events (%d utterances), %d causal, %d social, "
        "%d spatial, %d channels, %d world traits.",
        len(ws.events), utterance_count, len(ws.causal_topology), len(ws.social_topology),
        len(ws.spatial_topology), len(ws.channels),
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
    world_trait_ids = set(ws.world_traits.keys())
    event_id_set = {e.id for e in deduped_events}
    clean_events: List[EventNode] = []
    for evt in deduped_events:
        updates: dict = {}
        # Match the validator's per-event-type allowlist (see
        # _programmatic_validation): utterance actors may include OBJ_
        # (a dossier, a telescreen broadcast etc.), and utterance
        # targets additionally include EVT_/WORLD_/LOC_. Stripping all
        # non-entity actors here would silently destroy valid
        # speech-act provenance.
        is_utterance = evt.event_type == "utterance"
        actor_allowed = (entity_ids | object_ids) if is_utterance else entity_ids
        target_allowed = (
            entity_ids | object_ids | event_id_set | world_trait_ids | location_ids
            if is_utterance
            else entity_ids | object_ids
        )
        bad_actors = [a for a in evt.actor_ids if a not in actor_allowed]
        if bad_actors:
            repairs.append(f"Removed invalid actor_ids {bad_actors} from event '{evt.id}'.")
            updates["actor_ids"] = [a for a in evt.actor_ids if a in actor_allowed]
        bad_targets = [t for t in evt.target_ids if t not in target_allowed]
        if bad_targets:
            repairs.append(f"Removed invalid target_ids {bad_targets} from event '{evt.id}'.")
            updates["target_ids"] = [t for t in evt.target_ids if t in target_allowed]
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

    # --- Strip broken channels ---
    clean_channels: Dict[str, Channel] = {}
    for cid, ch in ws.channels.items():
        valid_pids = [p for p in ch.participant_ids if p in node_ids]
        bad_pids = [p for p in ch.participant_ids if p not in node_ids]
        for bp in bad_pids:
            repairs.append(f"Removed channel '{cid}' participant '{bp}' (not in entities/objects).")
        if len(valid_pids) < 2:
            repairs.append(f"Removed channel '{cid}' (fewer than 2 valid participants).")
            continue
        if valid_pids != list(ch.participant_ids):
            pruned_intel = {k: v for k, v in ch.intelligibility.items() if k in valid_pids}
            clean_channels[cid] = ch.model_copy(update={
                "participant_ids": valid_pids,
                "intelligibility": pruned_intel,
            })
        else:
            clean_channels[cid] = ch

    # --- Strip utterance events whose via_channel_id no longer resolves ---
    valid_channel_ids = set(clean_channels.keys())
    repaired_events: List[EventNode] = []
    for evt in clean_events:
        if (
            evt.event_type == "utterance"
            and evt.via_channel_id
            and evt.via_channel_id not in valid_channel_ids
        ):
            repairs.append(
                f"Cleared dangling via_channel_id '{evt.via_channel_id}' on utterance '{evt.id}'."
            )
            repaired_events.append(evt.model_copy(update={"via_channel_id": None}))
        else:
            repaired_events.append(evt)
    clean_events = repaired_events

    # --- Fuzzy-fix entity state_timeline.triggered_by references ---
    # Common failure mode: a consequences-extractor pass coined an EVT_ ID
    # spelt slightly differently from the physics-extractor pass (e.g.
    # ``EVT_LARS_MASSACRE`` vs ``EVT_LAR_MASSACRE``). Without this pass
    # the dangling reference would be flagged as a hard error and the
    # whole world-state would be sent through the LLM correction loop —
    # historically a much more destructive operation than just renaming
    # one ID. We try a fuzzy resolve first; if that fails we null out
    # the reference (a warning, not an error).
    event_ids_now = {e.id for e in clean_events}
    new_entities_map: Dict[str, Entity] = {}
    entities_changed = False
    for eid, ent in ws.entities.items():
        new_timeline: List[EntityStateSnapshot] = []
        timeline_changed = False
        for snap in ent.state_timeline:
            if snap.triggered_by and snap.triggered_by not in event_ids_now:
                resolved = _fuzzy_resolve_id(snap.triggered_by, event_ids_now)
                if resolved:
                    repairs.append(
                        f"Repaired entity '{eid}' state_timeline triggered_by "
                        f"'{snap.triggered_by}' → '{resolved}'."
                    )
                    new_timeline.append(snap.model_copy(update={"triggered_by": resolved}))
                    timeline_changed = True
                    continue
                else:
                    repairs.append(
                        f"Cleared dangling triggered_by '{snap.triggered_by}' on "
                        f"entity '{eid}' state_timeline (no fuzzy match)."
                    )
                    new_timeline.append(snap.model_copy(update={"triggered_by": None}))
                    timeline_changed = True
                    continue
            new_timeline.append(snap)
        if timeline_changed:
            new_entities_map[eid] = ent.model_copy(update={"state_timeline": new_timeline})
            entities_changed = True
        else:
            new_entities_map[eid] = ent
    if entities_changed:
        ws = ws.model_copy(update={"entities": new_entities_map})

    # --- Fuzzy-fix world_trait state_timeline.triggered_by references ---
    new_world_traits_map: Dict[str, GlobalTrait] = {}
    world_traits_changed = False
    for wid, wt in ws.world_traits.items():
        new_wt_timeline: List[WorldTraitSnapshot] = []
        wt_timeline_changed = False
        for snap in wt.state_timeline:
            trig = getattr(snap, "triggered_by", None)
            if trig and trig not in event_ids_now:
                resolved = _fuzzy_resolve_id(trig, event_ids_now)
                if resolved:
                    repairs.append(
                        f"Repaired world_trait '{wid}' state_timeline triggered_by "
                        f"'{trig}' → '{resolved}'."
                    )
                    new_wt_timeline.append(snap.model_copy(update={"triggered_by": resolved}))
                    wt_timeline_changed = True
                    continue
                else:
                    repairs.append(
                        f"Cleared dangling triggered_by '{trig}' on world_trait "
                        f"'{wid}' state_timeline (no fuzzy match)."
                    )
                    new_wt_timeline.append(snap.model_copy(update={"triggered_by": None}))
                    wt_timeline_changed = True
                    continue
            new_wt_timeline.append(snap)
        if wt_timeline_changed:
            new_world_traits_map[wid] = wt.model_copy(update={"state_timeline": new_wt_timeline})
            world_traits_changed = True
        else:
            new_world_traits_map[wid] = wt
    if world_traits_changed:
        ws = ws.model_copy(update={"world_traits": new_world_traits_map})

    # --- Self-referencing social edges ---
    # The validator flags these as "contradiction" warnings; under any
    # reasonable reading they're extraction noise. Drop them here so
    # the LLM correction loop is never invoked for self-loops.
    cleaned_social: List[RelationshipEdge] = []
    for re_edge in clean_social:
        if re_edge.source_entity_id == re_edge.target_entity_id:
            repairs.append(
                f"Removed self-referencing social edge: "
                f"'{re_edge.source_entity_id}' \u2192 '{re_edge.target_entity_id}'."
            )
        else:
            cleaned_social.append(re_edge)
    clean_social = cleaned_social

    # --- Channel terminate-before-establish ---
    # Drop the impossible termination tick rather than the whole channel:
    # the channel itself is usually correctly extracted, only the
    # terminated_at_fabula is a hallucinated date.
    fixed_channels: Dict[str, Channel] = {}
    for cid, ch in clean_channels.items():
        if (
            ch.terminated_at_fabula is not None
            and ch.terminated_at_fabula < ch.established_at_fabula
        ):
            repairs.append(
                f"Cleared invalid terminated_at_fabula={ch.terminated_at_fabula} "
                f"on channel '{cid}' (predates established_at_fabula="
                f"{ch.established_at_fabula})."
            )
            fixed_channels[cid] = ch.model_copy(update={"terminated_at_fabula": None})
        else:
            fixed_channels[cid] = ch
    clean_channels = fixed_channels

    # --- Belief provenance: rewrite/null dangling acquired_via_event_id /
    # --- acquired_via_channel_id refs on entity beliefs and snapshots.
    valid_event_ids = {e.id for e in clean_events}
    valid_channel_ids_set = set(clean_channels.keys())

    def _repair_belief(b: Belief, owner: str, *, in_snapshot_at: Optional[int] = None) -> Belief:
        update: dict = {}
        if b.acquired_via_event_id and b.acquired_via_event_id not in valid_event_ids:
            resolved = _fuzzy_resolve_id(b.acquired_via_event_id, valid_event_ids)
            label = f"belief about '{b.target_id}' on '{owner}'"
            if in_snapshot_at is not None:
                label += f" (snapshot at fabula={in_snapshot_at})"
            if resolved:
                repairs.append(
                    f"Repaired {label} acquired_via_event_id "
                    f"'{b.acquired_via_event_id}' \u2192 '{resolved}'."
                )
                update["acquired_via_event_id"] = resolved
            else:
                repairs.append(
                    f"Cleared dangling acquired_via_event_id "
                    f"'{b.acquired_via_event_id}' on {label} (no fuzzy match)."
                )
                update["acquired_via_event_id"] = None
        if (
            b.acquired_via_channel_id
            and b.acquired_via_channel_id not in valid_channel_ids_set
        ):
            repairs.append(
                f"Cleared dangling acquired_via_channel_id "
                f"'{b.acquired_via_channel_id}' on belief about '{b.target_id}' "
                f"on '{owner}'."
            )
            update["acquired_via_channel_id"] = None
        return b.model_copy(update=update) if update else b

    repaired_entities: Dict[str, Entity] = {}
    entities_belief_changed = False
    for eid, ent in ws.entities.items():
        ent_update: dict = {}
        # Standing beliefs
        if ent.beliefs:
            new_beliefs = [_repair_belief(b, eid) for b in ent.beliefs]
            if new_beliefs != list(ent.beliefs):
                ent_update["beliefs"] = new_beliefs
        # Snapshot-embedded beliefs
        if ent.state_timeline:
            new_tl: List[EntityStateSnapshot] = []
            tl_changed = False
            for snap in ent.state_timeline:
                if snap.beliefs_added:
                    repaired = [
                        _repair_belief(b, eid, in_snapshot_at=snap.fabula_time)
                        for b in snap.beliefs_added
                    ]
                    if repaired != list(snap.beliefs_added):
                        new_tl.append(snap.model_copy(update={"beliefs_added": repaired}))
                        tl_changed = True
                        continue
                new_tl.append(snap)
            if tl_changed:
                ent_update["state_timeline"] = new_tl
        if ent_update:
            entities_belief_changed = True
            repaired_entities[eid] = ent.model_copy(update=ent_update)
        else:
            repaired_entities[eid] = ent
    if entities_belief_changed:
        ws = ws.model_copy(update={"entities": repaired_entities})

    # --- Utterance missing-field auto-fixes ---
    # When an utterance lacks ``speaker_id`` but has a single ``actor_ids``
    # entry, take the actor as the speaker (the physics extractor often
    # writes the speaker into actor_ids). When ``addressee_ids`` is empty
    # but ``target_ids`` contains entity/object ids, take those as
    # addressees. These are deterministic shifts that the LLM correction
    # loop was previously being invoked for.
    repaired_utterances: List[EventNode] = []
    valid_addressees = entity_ids | set(ws.objects.keys())
    for evt in clean_events:
        if evt.event_type != "utterance":
            repaired_utterances.append(evt)
            continue
        update: dict = {}
        if not evt.speaker_id and len(evt.actor_ids) == 1 and evt.actor_ids[0] in valid_addressees:
            update["speaker_id"] = evt.actor_ids[0]
            repairs.append(
                f"Promoted single actor '{evt.actor_ids[0]}' to speaker_id "
                f"on utterance '{evt.id}'."
            )
        if not evt.addressee_ids and evt.target_ids:
            cand = [t for t in evt.target_ids if t in valid_addressees]
            if cand:
                update["addressee_ids"] = cand
                repairs.append(
                    f"Promoted target_ids \u2192 addressee_ids on utterance "
                    f"'{evt.id}': {cand}."
                )
        repaired_utterances.append(evt.model_copy(update=update) if update else evt)
    clean_events = repaired_utterances

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
            channels=clean_channels,
            social_topology=clean_social,
        )

    return ws, repairs


def _programmatic_validation(ws: WorldStateV1) -> List[ValidationIssue]:
    """Fast structural checks that don't require an LLM."""
    issues: List[ValidationIssue] = []

    # Cross-register ID-namespace collision check. The valid_ids set
    # below is built by union, which silently absorbs collisions; e.g.
    # a stray ``OBJ_DAGGER`` mistakenly registered under
    # ``ws.entities`` and a real ``OBJ_DAGGER`` in ``ws.objects`` both
    # collapse to a single membership token. Downstream lookups would
    # then resolve the ID to whichever register the consumer happened
    # to query first \u2014 a classic source of \"phantom entity\" bugs
    # during counterfactual surgery.
    register_views: List[Tuple[str, set]] = [
        ("locations", set(ws.locations.keys())),
        ("objects", set(ws.objects.keys())),
        ("entities", set(ws.entities.keys())),
        ("world_traits", set(ws.world_traits.keys())),
        ("channels", set(ws.channels.keys())),
    ]
    for i in range(len(register_views)):
        for j in range(i + 1, len(register_views)):
            name_a, ids_a = register_views[i]
            name_b, ids_b = register_views[j]
            overlap = ids_a & ids_b
            for dup in sorted(overlap):
                issues.append(ValidationIssue(
                    severity="error", category="duplicate",
                    detail=(
                        f"ID '{dup}' is registered in both "
                        f"ws.{name_a} and ws.{name_b}; downstream "
                        f"lookups will resolve ambiguously."
                    ),
                ))

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

    # Check channels
    node_ids = set(ws.entities.keys()) | set(ws.objects.keys())
    for cid, ch in ws.channels.items():
        for pid in ch.participant_ids:
            if pid not in node_ids:
                issues.append(ValidationIssue(
                    severity="error", category="broken_link",
                    detail=f"Channel '{cid}' participant_id '{pid}' not in entities/objects.",
                ))
        if len(ch.participant_ids) < 2:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"Channel '{cid}' has fewer than 2 participants.",
            ))
        if len(set(ch.participant_ids)) < len(ch.participant_ids):
            dupes = [p for p in ch.participant_ids if ch.participant_ids.count(p) > 1]
            issues.append(ValidationIssue(
                severity="warning", category="duplicate",
                detail=(
                    f"Channel '{cid}' has duplicate participant_ids "
                    f"{sorted(set(dupes))}; downstream consumers will see a "
                    f"phantom n-way channel."
                ),
            ))
        for k in ch.intelligibility.keys():
            if k not in ch.participant_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"Channel '{cid}' intelligibility entry for '{k}' is not "
                        f"a participant of the channel."
                    ),
                ))

    # Check utterance event references
    valid_channel_ids = set(ws.channels.keys())
    for evt in ws.events:
        if evt.event_type != "utterance":
            continue
        # Required fields for utterances. The Pydantic model marks
        # these Optional so that non-utterance events can omit them,
        # but for ``event_type='utterance'`` an absent speaker or
        # empty addressees breaks downstream consumers (belief
        # propagation, channel intelligibility routing, the social
        # propagator that mirrors speaker_id into actor_ids).
        if not evt.speaker_id:
            issues.append(ValidationIssue(
                severity="error", category="missing_field",
                detail=(
                    f"Utterance '{evt.id}' is missing required field "
                    f"'speaker_id'."
                ),
            ))
        if not evt.addressee_ids:
            issues.append(ValidationIssue(
                severity="error", category="missing_field",
                detail=(
                    f"Utterance '{evt.id}' is missing required field "
                    f"'addressee_ids' (must contain at least one ENT_ id)."
                ),
            ))
        if evt.via_channel_id and evt.via_channel_id not in valid_channel_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=(
                    f"Utterance '{evt.id}' via_channel_id '{evt.via_channel_id}' "
                    f"not in channels."
                ),
            ))
        # If utterance routes through a channel, both speaker and every
        # addressee must actually be participants in that channel —
        # otherwise the propagator will silently drop the message at
        # intelligibility-check time.
        if evt.via_channel_id and evt.via_channel_id in ws.channels:
            chan_participants = set(ws.channels[evt.via_channel_id].participant_ids)
            if evt.speaker_id and evt.speaker_id not in chan_participants:
                issues.append(ValidationIssue(
                    severity="error", category="broken_link",
                    detail=(
                        f"Utterance '{evt.id}' speaker_id '{evt.speaker_id}' "
                        f"is not a participant of via_channel_id '{evt.via_channel_id}'."
                    ),
                ))
            for aid in evt.addressee_ids:
                if aid not in chan_participants:
                    issues.append(ValidationIssue(
                        severity="error", category="broken_link",
                        detail=(
                            f"Utterance '{evt.id}' addressee '{aid}' is not a "
                            f"participant of via_channel_id '{evt.via_channel_id}'."
                        ),
                    ))
        if evt.speaker_id and evt.speaker_id not in node_ids:
            issues.append(ValidationIssue(
                severity="error", category="broken_link",
                detail=f"Utterance '{evt.id}' speaker_id '{evt.speaker_id}' not in entities/objects.",
            ))
        for aid in evt.addressee_ids:
            if aid not in node_ids:
                issues.append(ValidationIssue(
                    severity="error", category="broken_link",
                    detail=f"Utterance '{evt.id}' addressee_ids entry '{aid}' not in entities/objects.",
                ))

    # Check belief provenance: acquired_via_event_id must point at a real
    # event, acquired_via_channel_id at a real channel. Counterfactual
    # surgery uses these to prune downstream beliefs when an event /
    # channel is removed; dangling refs would silently break that.
    valid_event_ids = {e.id for e in ws.events}
    for eid, ent in ws.entities.items():
        for b in ent.beliefs:
            if (
                b.acquired_via_event_id
                and b.acquired_via_event_id not in valid_event_ids
            ):
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"Entity '{eid}' belief about '{b.target_id}' has "
                        f"acquired_via_event_id='{b.acquired_via_event_id}' "
                        f"that is not in events."
                    ),
                ))
            if (
                b.acquired_via_channel_id
                and b.acquired_via_channel_id not in valid_channel_ids
            ):
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"Entity '{eid}' belief about '{b.target_id}' has "
                        f"acquired_via_channel_id='{b.acquired_via_channel_id}' "
                        f"that is not in channels."
                    ),
                ))
        # State-timeline snapshots carry the same belief shape.
        for snap in ent.state_timeline:
            for b in snap.beliefs_added:
                if (
                    b.acquired_via_event_id
                    and b.acquired_via_event_id not in valid_event_ids
                ):
                    issues.append(ValidationIssue(
                        severity="warning", category="broken_link",
                        detail=(
                            f"Entity '{eid}' snapshot belief at "
                            f"fabula={snap.fabula_time} has dangling "
                            f"acquired_via_event_id='{b.acquired_via_event_id}'."
                        ),
                    ))
                if (
                    b.acquired_via_channel_id
                    and b.acquired_via_channel_id not in valid_channel_ids
                ):
                    issues.append(ValidationIssue(
                        severity="warning", category="broken_link",
                        detail=(
                            f"Entity '{eid}' snapshot belief at "
                            f"fabula={snap.fabula_time} has dangling "
                            f"acquired_via_channel_id='{b.acquired_via_channel_id}'."
                        ),
                    ))

    # Belief provenance — semantic coherence checks (warnings).
    #
    # The structural checks above only verify the referenced IDs exist.
    # These checks verify the *meaning* of the provenance edge:
    #   * The referenced event should normally be an utterance or a
    #     revelation-class event — beliefs acquired via random
    #     unrelated events are usually extraction noise.
    #   * When both channel and event provenance are set, they must be
    #     consistent: the utterance event's via_channel_id should match
    #     the belief's acquired_via_channel_id.
    #   * High-confidence beliefs acquired through low-intelligibility
    #     channels are epistemically suspect — flag for human review.
    event_index = {e.id: e for e in ws.events}
    revelation_event_types = {"utterance", "revelation", "discovery", "observation"}
    intel_warn_threshold = 0.3
    for eid, ent in ws.entities.items():
        for b in ent.beliefs:
            ev_id = b.acquired_via_event_id
            ch_id = b.acquired_via_channel_id
            if ev_id and ev_id in event_index:
                src_evt = event_index[ev_id]
                if src_evt.event_type not in revelation_event_types:
                    issues.append(ValidationIssue(
                        severity="warning", category="semantic_provenance",
                        detail=(
                            f"Entity '{eid}' belief about '{b.target_id}' "
                            f"is acquired_via_event_id='{ev_id}' whose "
                            f"event_type='{src_evt.event_type}' is not a "
                            f"revelation-class event "
                            f"({sorted(revelation_event_types)}). Likely "
                            f"extraction noise."
                        ),
                    ))
                # Cross-field coherence: if both channel and utterance
                # are set, the utterance must travel via that channel.
                if (
                    ch_id
                    and src_evt.event_type == "utterance"
                    and src_evt.via_channel_id
                    and src_evt.via_channel_id != ch_id
                ):
                    issues.append(ValidationIssue(
                        severity="warning", category="semantic_provenance",
                        detail=(
                            f"Entity '{eid}' belief about '{b.target_id}' "
                            f"declares acquired_via_channel_id='{ch_id}' "
                            f"but the source utterance '{ev_id}' was "
                            f"transmitted via_channel_id="
                            f"'{src_evt.via_channel_id}'. Provenance is "
                            f"internally inconsistent."
                        ),
                    ))
            # High-confidence belief through low-intelligibility channel.
            if ch_id and ch_id in ws.channels:
                ch = ws.channels[ch_id]
                intel = float(ch.intelligibility.get(eid, 1.0))
                conf = float(getattr(b, "confidence", 1.0) or 1.0)
                if intel < intel_warn_threshold and conf >= 0.8:
                    issues.append(ValidationIssue(
                        severity="warning", category="semantic_provenance",
                        detail=(
                            f"Entity '{eid}' holds confident belief "
                            f"(conf={conf:.2f}) about '{b.target_id}' "
                            f"acquired through channel '{ch_id}' where "
                            f"its intelligibility is only {intel:.2f}. "
                            f"Low-intelligibility channels should not "
                            f"yield high-confidence beliefs."
                        ),
                    ))

    # Check event actor_ids and target_ids.
    #
    # For non-utterance events: actor_ids must be ENT_, target_ids must
    # be ENT_/OBJ_ (mirrors the physics_extraction prompt).
    #
    # For utterance events: per social_extraction.md, the speaker (and
    # therefore actor_ids[0]) MAY be ENT_ or OBJ_ (e.g. a dossier, a
    # telescreen broadcast); target_ids MAY additionally include EVT_,
    # WORLD_, and LOC_ ids — utterances are *about* topics, and topics
    # are commonly past events, world facts, or places. Restricting
    # utterance target_ids to ENT_/OBJ_ would block the prompt's own
    # documented "X tells Y about EVT_Z" pattern.
    object_ids = set(ws.objects.keys())
    world_trait_ids = set(ws.world_traits.keys())
    event_id_set = {e.id for e in ws.events}
    for evt in ws.events:
        is_utterance = evt.event_type == "utterance"
        actor_allowed = (entity_ids | object_ids) if is_utterance else entity_ids
        target_allowed = (
            entity_ids | object_ids | event_id_set | world_trait_ids | location_ids
            if is_utterance
            else entity_ids | object_ids
        )
        actor_label = "entity/object" if is_utterance else "entity"
        target_label = (
            "entity/object/event/world_trait/location"
            if is_utterance
            else "entity/object"
        )
        for aid in evt.actor_ids:
            if aid not in actor_allowed:
                issues.append(ValidationIssue(
                    severity="error", category="hallucinated_id",
                    detail=f"EventNode '{evt.id}' actor_ids entry '{aid}' is not a valid {actor_label}.",
                ))
        for tid in evt.target_ids:
            if tid not in target_allowed:
                issues.append(ValidationIssue(
                    severity="error", category="hallucinated_id",
                    detail=f"EventNode '{evt.id}' target_ids entry '{tid}' is not a valid {target_label}.",
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
        # Same shape applies to beliefs sitting on snapshots; they
        # become an entity's live ``beliefs`` after world-state replay,
        # so a dangling target_id here is just a delayed broken_link.
        for snap in ent.state_timeline:
            for belief in snap.beliefs_added:
                if belief.target_id not in all_valid_belief_targets:
                    issues.append(ValidationIssue(
                        severity="warning", category="broken_link",
                        detail=(
                            f"Entity '{eid}' snapshot belief at "
                            f"fabula={snap.fabula_time} target_id "
                            f"'{belief.target_id}' not in "
                            f"locations/objects/entities/events/world_traits."
                        ),
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

    # Check world_trait state_timeline references (mirror of the entity
    # block above). Without this, EVT renames / drops in the patch path
    # silently leave dangling triggered_by ids on world traits, which
    # the LLM auditor is unlikely to surface and downstream surgery
    # cannot undo.
    for wid, wt in ws.world_traits.items():
        prev_ft = -1
        for snap in wt.state_timeline:
            if snap.triggered_by and snap.triggered_by not in event_ids:
                issues.append(ValidationIssue(
                    severity="warning", category="broken_link",
                    detail=(
                        f"World trait '{wid}' state_timeline triggered_by "
                        f"'{snap.triggered_by}' not in events."
                    ),
                ))
            if snap.fabula_time < prev_ft:
                issues.append(ValidationIssue(
                    severity="warning", category="temporal",
                    detail=(
                        f"World trait '{wid}' state_timeline not monotonic: "
                        f"fabula_time {snap.fabula_time} follows {prev_ft}."
                    ),
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

    # --- Information density check ---
    # Scale the expected information signal by the *narratively
    # information-bearing* event types only. Action-heavy chunks
    # (heists, battles, chases — Reservoir Dogs, Apocalypse Now) can
    # legitimately be all action/outcome events with no utterances or
    # channels, and the previous unconditional rule was flagging those
    # as "missing information" and feeding them into the LLM correction
    # loop, which then invented spurious channels.
    utterance_count = sum(1 for e in ws.events if e.event_type == "utterance")
    info_signal = len(ws.channels) + utterance_count
    info_bearing = sum(
        1 for e in ws.events
        if e.event_type in ("choice", "revelation", "utterance")
    )
    if info_bearing >= 3 and info_signal == 0:
        issues.append(ValidationIssue(
            severity="warning", category="missing_information",
            detail=(
                f"Zero channels and zero utterance events extracted across "
                f"{info_bearing} information-bearing event(s) "
                f"(choice/revelation/utterance). Most narratives with that "
                f"many decisions or revelations contain conversations, "
                f"letters, or proclamations that should produce a Channel "
                f"or an utterance EventNode."
            ),
        ))
    elif info_bearing >= 5 and info_signal < info_bearing // 5:
        issues.append(ValidationIssue(
            severity="warning", category="missing_information",
            detail=(
                f"Low information density: {len(ws.channels)} channels + "
                f"{utterance_count} utterances for {info_bearing} "
                f"information-bearing events (ratio "
                f"{info_signal / info_bearing:.2f}). Expected at least 1 "
                f"information signal per 5 information-bearing events."
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

    # --- Mutation-parity check ---
    #
    # Surfaces the single largest gap the 2026-05-01 plot-models audit found:
    # 75 mutation/mutation_social edges declared a (trait_target, trait_delta)
    # but the target entity had no corresponding state_timeline snapshot at
    # the edge's fabula_time. Without the snapshot the physics engine has
    # nothing to anchor downstream propagation and abduction reads to, and
    # the mutation effectively vanishes after one tick.
    #
    # We allow ±1 fabula tick of slack so authors can co-locate a snapshot
    # at a near-by event boundary (the engine's interpolation handles tiny
    # offsets cleanly).
    if ws.events and ws.causal_topology:
        snapshots_by_entity: Dict[str, set[int]] = {}
        for eid, ent in ws.entities.items():
            snapshots_by_entity[eid] = {snap.fabula_time for snap in ent.state_timeline}
        unmatched: List[str] = []
        for ce in ws.causal_topology:
            if ce.causality_type not in ("mutation", "mutation_social"):
                continue
            if ce.trait_target is None or ce.trait_delta is None:
                continue
            if ce.target_id not in entity_ids:
                continue
            snaps = snapshots_by_entity.get(ce.target_id, set())
            # ±1 tick slack to tolerate authoring co-location.
            if not any(abs(t - ce.fabula_time) <= 1 for t in snaps):
                unmatched.append(
                    f"{ce.source_id}\u2192{ce.target_id} "
                    f"({ce.trait_target}, fabula={ce.fabula_time})"
                )
        if unmatched:
            sample = unmatched[:5]
            issues.append(ValidationIssue(
                severity="warning", category="mutation_parity",
                detail=(
                    f"{len(unmatched)} mutation edge(s) declare a "
                    f"trait_target+trait_delta but the target entity has no "
                    f"state_timeline snapshot at the edge's fabula_time "
                    f"(\u00b11 tick slack): {sample}"
                    f"{'\u2026' if len(unmatched) > 5 else ''}. The mutation "
                    f"is recorded on the edge but never anchored on the "
                    f"entity, so propagation and abduction will under-read it."
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

    # 4. Check channels: terminated cannot precede established
    for cid, ch in ws.channels.items():
        if ch.terminated_at_fabula is not None and ch.terminated_at_fabula < ch.established_at_fabula:
            issues.append(ValidationIssue(
                severity="error", category="temporal",
                detail=(
                    f"Channel '{cid}' (participants={ch.participant_ids}): "
                    f"terminated_at_fabula ({ch.terminated_at_fabula}) < "
                    f"established_at_fabula ({ch.established_at_fabula})."
                ),
            ))

    return issues


def _build_validation_agent(config: ExtractionConfig) -> Agent[None, ValidationReport]:
    """Construct the Step 3 LLM validation agent.

    Uses :class:`PromptedOutput` rather than :class:`NativeOutput` because
    Ollama's OpenAI-compat ``response_format=json_schema`` path can return
    ``400 invalid message content type: <nil>`` for some models (e.g.
    ``qwen3.6:35b``). Prompted output injects the schema into the system
    prompt and parses JSON from plain text, which works reliably across
    both the local Ollama backend and OpenAI-compat providers like
    OpenRouter.
    """
    return Agent(
        _resolve_model(config.model),
        output_type=PromptedOutput(ValidationReport),
        system_prompt=_load_prompt("validation.md"),
        retries=config.output_retries,
    )


def _build_correction_agent(config: ExtractionConfig) -> Agent[None, WorldStateV1]:
    """Construct the legacy whole-WorldState correction agent.

    Retained for backwards compatibility but no longer used by the
    pipeline; ``_build_correction_patch_agent`` is now the preferred
    entry point because it asks the LLM for a *diff* instead of a full
    re-emission, eliminating the catastrophic-shrinkage failure mode
    where the model returned a JSON document missing whole topologies.
    """
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldStateV1),
        system_prompt=_load_prompt("correction.md"),
        retries=config.output_retries,
    )


# =====================================================================
# Patch-based correction (preferred path) — see correction.md
# =====================================================================


class _EdgeRef(BaseModel):
    """Identifies a single causal/social/spatial edge by its endpoints."""
    source_id: str
    target_id: str


class WorldStatePatch(BaseModel):
    """A *diff* to apply to an existing WorldStateV1.

    The correction agent emits one of these instead of a full
    WorldStateV1 so it can only describe *changes*, never accidentally
    drop unrelated parts of the world (the failure mode that destroyed
    every causal/social/spatial edge in the Star Wars fixture when the
    LLM was asked to re-emit a complete WorldStateV1 within a token
    budget).

    All fields default to "no change". The patch is applied in this
    order:

      1. ``event_renames``        — rewrite EVT_ IDs everywhere they appear
      2. ``drop_event_ids``       — remove events and dangling references
      3. ``update_event_fields``  — partial field updates on surviving events
      4. ``update_entity_location`` — fix dangling entity.location_id
      5. ``add_state_timeline_entries`` — extend entity.state_timeline
      6. ``drop_*`` for edges/channels — remove specific edges by endpoint
      7. ``add_*`` for edges/channels — append new edges/channels
      8. Final pass: ``_auto_repair`` prunes any newly-dangling refs
    """
    model_config = {"protected_namespaces": ()}

    event_renames: Dict[str, str] = Field(
        default_factory=dict,
        description="EVT_ ID renames: {old_id: new_id}. Applied to every reference in the world state.",
    )
    drop_event_ids: List[str] = Field(
        default_factory=list,
        description="Event IDs to remove entirely. Edges referencing these will be auto-pruned.",
    )
    update_event_fields: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-event field overrides: {event_id: {field: new_value}}. Use to add a missing speaker_id, addressee_ids, via_channel_id, etc.",
    )
    update_entity_location: Dict[str, str] = Field(
        default_factory=dict,
        description="Entity location_id overrides: {entity_id: new_location_id}.",
    )
    add_state_timeline_entries: Dict[str, List[EntityStateSnapshot]] = Field(
        default_factory=dict,
        description="Append snapshots to entity.state_timeline: {entity_id: [snapshot, ...]}.",
    )
    drop_causal_edges: List[_EdgeRef] = Field(
        default_factory=list,
        description="Causal edges to drop, identified by (source_id, target_id).",
    )
    add_causal_edges: List[CausalEdge] = Field(
        default_factory=list,
        description="New causal edges to append.",
    )
    drop_social_edges: List[_EdgeRef] = Field(
        default_factory=list,
        description="Social edges to drop.",
    )
    add_social_edges: List[RelationshipEdge] = Field(
        default_factory=list,
        description="New social edges to append.",
    )
    drop_spatial_edges: List[_EdgeRef] = Field(
        default_factory=list,
        description="Spatial edges to drop.",
    )
    add_spatial_edges: List[SpatialEdge] = Field(
        default_factory=list,
        description="New spatial edges to append.",
    )
    drop_channel_ids: List[str] = Field(
        default_factory=list,
        description="Channel IDs to drop entirely.",
    )
    add_channels: Dict[str, Channel] = Field(
        default_factory=dict,
        description="New channels keyed by channel_id.",
    )
    channel_renames: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Channel CHN_ ID renames: {old_id: new_id}. Forwards every "
            "via_channel_id / acquired_via_channel_id reference to the "
            "new id, so a simple typo fix does not have to be expressed "
            "as drop+add (which would null the provenance of every "
            "belief / utterance pointing at the old channel)."
        ),
    )
    notes: str = Field(
        default="",
        description="Free-text rationale for the maintainer log; not applied to the world state.",
    )


def _apply_world_state_patch(
    ws: WorldStateV1, patch: WorldStatePatch,
) -> Tuple[WorldStateV1, List[str]]:
    """Apply *patch* to *ws* and return the new world state + a change log."""
    changes: List[str] = []
    renames = dict(patch.event_renames)
    drop_evts = set(patch.drop_event_ids)
    chan_renames = dict(patch.channel_renames)

    def _r(evt_id: Optional[str]) -> Optional[str]:
        if evt_id is None:
            return None
        return renames.get(evt_id, evt_id)

    def _rc(chan_id: Optional[str]) -> Optional[str]:
        if chan_id is None:
            return None
        return chan_renames.get(chan_id, chan_id)

    # 1. Apply renames + drops to events (with field updates).
    field_updates = patch.update_event_fields
    new_events: List[EventNode] = []
    for evt in ws.events:
        new_id = renames.get(evt.id, evt.id)
        if new_id in drop_evts or evt.id in drop_evts:
            changes.append(f"Dropped event '{evt.id}'.")
            continue
        update: dict = {}
        if new_id != evt.id:
            update["id"] = new_id
            changes.append(f"Renamed event '{evt.id}' → '{new_id}'.")
        # Carry over field-level overrides keyed by either old or new id.
        overrides = field_updates.get(evt.id) or field_updates.get(new_id)
        if overrides:
            update.update(overrides)
            changes.append(f"Updated event '{new_id}' fields: {sorted(overrides.keys())}.")
        new_events.append(evt.model_copy(update=update) if update else evt)

    surviving_event_ids = {e.id for e in new_events}

    # 2. Rewrite event references in causal edges; drop those targeting removed events.
    drop_causal_pairs = {(e.source_id, e.target_id) for e in patch.drop_causal_edges}
    new_causal: List[CausalEdge] = []
    for ce in ws.causal_topology:
        new_src = _r(ce.source_id) if ce.source_id.startswith("EVT_") else ce.source_id
        new_tgt = _r(ce.target_id) if ce.target_id.startswith("EVT_") else ce.target_id
        if (ce.source_id, ce.target_id) in drop_causal_pairs or (new_src, new_tgt) in drop_causal_pairs:
            changes.append(f"Dropped causal edge {ce.source_id}→{ce.target_id}.")
            continue
        update: dict = {}
        if new_src != ce.source_id:
            update["source_id"] = new_src
        if new_tgt != ce.target_id:
            update["target_id"] = new_tgt
        if ce.rel_counterpart_id and ce.rel_counterpart_id.startswith("EVT_"):
            new_rc = _r(ce.rel_counterpart_id)
            if new_rc != ce.rel_counterpart_id:
                update["rel_counterpart_id"] = new_rc
        new_causal.append(ce.model_copy(update=update) if update else ce)

    # 3. Append new causal edges (with renames pre-applied).
    for ce in patch.add_causal_edges:
        update: dict = {}
        if ce.source_id.startswith("EVT_") and ce.source_id in renames:
            update["source_id"] = renames[ce.source_id]
        if ce.target_id.startswith("EVT_") and ce.target_id in renames:
            update["target_id"] = renames[ce.target_id]
        new_causal.append(ce.model_copy(update=update) if update else ce)
        changes.append(f"Added causal edge {ce.source_id}→{ce.target_id}.")

    # 4. Social edges — drop / add (no event renames apply).
    drop_social_pairs = {(e.source_id, e.target_id) for e in patch.drop_social_edges}
    new_social: List[RelationshipEdge] = []
    for re_edge in ws.social_topology:
        if (re_edge.source_entity_id, re_edge.target_entity_id) in drop_social_pairs:
            changes.append(f"Dropped social edge {re_edge.source_entity_id}→{re_edge.target_entity_id}.")
            continue
        new_social.append(re_edge)
    for re_edge in patch.add_social_edges:
        new_social.append(re_edge)
        changes.append(f"Added social edge {re_edge.source_entity_id}→{re_edge.target_entity_id}.")

    # 5. Spatial edges — drop / add.
    drop_spatial_pairs = {(e.source_id, e.target_id) for e in patch.drop_spatial_edges}
    new_spatial: List[SpatialEdge] = []
    for se in ws.spatial_topology:
        if (se.source_id, se.target_id) in drop_spatial_pairs:
            changes.append(f"Dropped spatial edge {se.source_id}→{se.target_id}.")
            continue
        new_spatial.append(se)
    for se in patch.add_spatial_edges:
        new_spatial.append(se)
        changes.append(f"Added spatial edge {se.source_id}→{se.target_id}.")

    # 6. Channels — rename, then drop, then add. Renames forward
    #    references from old → new id; drops still null references.
    drop_chan_ids = set(patch.drop_channel_ids)
    new_channels: Dict[str, Channel] = {}
    for cid, ch in ws.channels.items():
        new_cid = chan_renames.get(cid, cid)
        if new_cid in drop_chan_ids or cid in drop_chan_ids:
            changes.append(f"Dropped channel '{cid}'.")
            continue
        if new_cid != cid:
            changes.append(f"Renamed channel '{cid}' → '{new_cid}'.")
            new_channels[new_cid] = ch.model_copy(update={"id": new_cid})
        else:
            new_channels[cid] = ch
    for cid, ch in patch.add_channels.items():
        new_channels[cid] = ch
        changes.append(f"Added channel '{cid}'.")

    # 7. Entity location overrides + state_timeline appends.
    new_entities: Dict[str, Entity] = {}
    for eid, ent in ws.entities.items():
        update: dict = {}
        new_loc = patch.update_entity_location.get(eid)
        if new_loc and new_loc != ent.location_id:
            update["location_id"] = new_loc
            changes.append(f"Updated entity '{eid}' location_id → '{new_loc}'.")
        extras = patch.add_state_timeline_entries.get(eid)
        if extras:
            # Rewrite triggered_by through renames before appending.
            normalised_extras: List[EntityStateSnapshot] = []
            for snap in extras:
                snap_update: dict = {}
                if snap.triggered_by and snap.triggered_by in renames:
                    snap_update["triggered_by"] = renames[snap.triggered_by]
                normalised_extras.append(
                    snap.model_copy(update=snap_update) if snap_update else snap
                )
            merged = list(ent.state_timeline) + normalised_extras
            merged.sort(key=_snapshot_sort_key)
            update["state_timeline"] = merged
            changes.append(
                f"Appended {len(normalised_extras)} state_timeline snapshot(s) to entity '{eid}'."
            )
        new_entities[eid] = ent.model_copy(update=update) if update else ent

    # 8. Rewrite event-id and channel-id references throughout entity
    #    timelines and beliefs so renames/drops propagate transparently.
    #    Without this step the patch path silently leaves dangling
    #    Belief.acquired_via_event_id / acquired_via_channel_id refs,
    #    which the validator only flags at warning severity \u2014 and
    #    counterfactual surgery relies on these refs to roll back
    #    beliefs when their source event/channel is removed (so dangling
    #    provenance silently leaks invalidated beliefs into do-surgery).
    needs_rewrite = (
        bool(renames) or bool(drop_evts)
        or bool(drop_chan_ids) or bool(chan_renames)
    )
    if needs_rewrite:
        def _rewrite_belief(b: Belief) -> Tuple[Belief, bool]:
            update: dict = {}
            if b.acquired_via_event_id:
                new_evt = _r(b.acquired_via_event_id)
                if new_evt in drop_evts:
                    update["acquired_via_event_id"] = None
                elif new_evt != b.acquired_via_event_id:
                    update["acquired_via_event_id"] = new_evt
            if b.acquired_via_channel_id:
                new_chan = _rc(b.acquired_via_channel_id)
                if new_chan in drop_chan_ids:
                    update["acquired_via_channel_id"] = None
                elif new_chan != b.acquired_via_channel_id:
                    update["acquired_via_channel_id"] = new_chan
            return (b.model_copy(update=update), True) if update else (b, False)

        rewritten: Dict[str, Entity] = {}
        for eid, ent in new_entities.items():
            ent_update: dict = {}
            # --- Standing beliefs on the entity itself ---
            new_beliefs: List[Belief] = []
            beliefs_changed = False
            for b in ent.beliefs:
                rb, changed = _rewrite_belief(b)
                beliefs_changed = beliefs_changed or changed
                new_beliefs.append(rb)
            if beliefs_changed:
                ent_update["beliefs"] = new_beliefs

            # --- state_timeline snapshots: triggered_by + nested beliefs ---
            tl_changed = False
            new_tl: List[EntityStateSnapshot] = []
            for snap in ent.state_timeline:
                snap_update: dict = {}
                if snap.triggered_by:
                    new_trig = _r(snap.triggered_by)
                    if new_trig in drop_evts:
                        snap_update["triggered_by"] = None
                        tl_changed = True
                    elif new_trig != snap.triggered_by:
                        snap_update["triggered_by"] = new_trig
                        tl_changed = True
                # beliefs_added inside the snapshot
                if snap.beliefs_added:
                    snap_beliefs: List[Belief] = []
                    snap_beliefs_changed = False
                    for b in snap.beliefs_added:
                        rb, changed = _rewrite_belief(b)
                        snap_beliefs_changed = snap_beliefs_changed or changed
                        snap_beliefs.append(rb)
                    if snap_beliefs_changed:
                        snap_update["beliefs_added"] = snap_beliefs
                        tl_changed = True
                new_tl.append(snap.model_copy(update=snap_update) if snap_update else snap)
            if tl_changed:
                ent_update["state_timeline"] = new_tl

            rewritten[eid] = ent.model_copy(update=ent_update) if ent_update else ent
        new_entities = rewritten

        # --- Utterance via_channel_id (forward renames, null drops) ---
        if drop_chan_ids or chan_renames:
            updated_events: List[EventNode] = []
            for evt in new_events:
                if evt.event_type == "utterance" and evt.via_channel_id:
                    new_chan = _rc(evt.via_channel_id)
                    if new_chan in drop_chan_ids:
                        updated_events.append(evt.model_copy(update={"via_channel_id": None}))
                        changes.append(
                            f"Cleared dangling via_channel_id on utterance '{evt.id}' "
                            f"(channel was dropped by patch)."
                        )
                        continue
                    if new_chan != evt.via_channel_id:
                        updated_events.append(evt.model_copy(update={"via_channel_id": new_chan}))
                        continue
                updated_events.append(evt)
            new_events = updated_events

        # --- World-trait state_timeline triggered_by ---
        # The patch contract advertises that EVT renames/drops cascade
        # everywhere. Without this block the world-trait timeline
        # silently retains stale ids; the validator only catches it
        # at warning severity so a renamed event can leak unfixed
        # provenance into counterfactual surgery.
        if renames or drop_evts:
            new_world_traits: Dict[str, GlobalTrait] = {}
            wt_changed_any = False
            for wid, wt in ws.world_traits.items():
                wt_tl_changed = False
                new_wt_tl: List[Any] = []
                for snap in wt.state_timeline:
                    if snap.triggered_by:
                        new_trig = _r(snap.triggered_by)
                        if new_trig in drop_evts:
                            new_wt_tl.append(snap.model_copy(update={"triggered_by": None}))
                            wt_tl_changed = True
                            continue
                        if new_trig != snap.triggered_by:
                            new_wt_tl.append(snap.model_copy(update={"triggered_by": new_trig}))
                            wt_tl_changed = True
                            continue
                    new_wt_tl.append(snap)
                if wt_tl_changed:
                    new_world_traits[wid] = wt.model_copy(update={"state_timeline": new_wt_tl})
                    wt_changed_any = True
                    changes.append(
                        f"Forwarded EVT renames/drops on world_trait '{wid}' state_timeline."
                    )
                else:
                    new_world_traits[wid] = wt
            patched_world_traits = new_world_traits if wt_changed_any else ws.world_traits
        else:
            patched_world_traits = ws.world_traits
    else:
        patched_world_traits = ws.world_traits

    new_ws = WorldStateV1(
        locations=ws.locations,
        objects=ws.objects,
        entities=new_entities,
        events=new_events,
        world_traits=patched_world_traits,
        causal_topology=new_causal,
        spatial_topology=new_spatial,
        channels=new_channels,
        social_topology=new_social,
    )
    return new_ws, changes


def _build_correction_patch_agent(
    config: ExtractionConfig,
) -> Agent[None, WorldStatePatch]:
    """Construct the patch-based correction agent (preferred path).

    The agent receives the current WorldStateV1 + a list of programmatic
    errors and is asked to emit a *small* :class:`WorldStatePatch` that
    fixes only what the errors named. This is dramatically more robust
    than asking it to re-emit the entire WorldStateV1, which had the
    failure mode of silently dropping whole topology fields when the
    response token budget ran out.
    """
    return Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(WorldStatePatch),
        system_prompt=_load_prompt("correction.md"),
        retries=config.output_retries,
    )


def _is_correction_regression(
    before: WorldStateV1, after: WorldStateV1,
) -> Optional[str]:
    """Return a human-readable reason if *after* has lost too much vs *before*.

    Used as a circuit-breaker on patch application: if the LLM somehow
    drops more than half of any topology or any entities, we reject the
    patch and keep the previous state. Returns ``None`` when the post-
    correction state is acceptable.
    """
    def _ratio(a: int, b: int) -> float:
        return (a / b) if b > 0 else 1.0

    if len(after.entities) < len(before.entities):
        return (
            f"entities shrank from {len(before.entities)} to "
            f"{len(after.entities)} (correction is not allowed to drop "
            f"entities)"
        )
    if _ratio(len(after.events), len(before.events)) < 0.8:
        return (
            f"events shrank from {len(before.events)} to "
            f"{len(after.events)} (>20% loss)"
        )
    if before.causal_topology and _ratio(len(after.causal_topology), len(before.causal_topology)) < 0.5:
        return (
            f"causal_topology shrank from {len(before.causal_topology)} "
            f"to {len(after.causal_topology)} (>50% loss)"
        )
    if before.social_topology and _ratio(len(after.social_topology), len(before.social_topology)) < 0.5:
        return (
            f"social_topology shrank from {len(before.social_topology)} "
            f"to {len(after.social_topology)} (>50% loss)"
        )
    if before.spatial_topology and _ratio(len(after.spatial_topology), len(before.spatial_topology)) < 0.5:
        return (
            f"spatial_topology shrank from {len(before.spatial_topology)} "
            f"to {len(after.spatial_topology)} (>50% loss)"
        )
    # Channels and world-traits are *narrative ontology* — a correction
    # patch dropping more than half of either is almost certainly a
    # destructive hallucination. Channel loss in particular silently
    # severs every belief / utterance provenance edge that pointed at
    # the dropped CHN_, which the existing belief-provenance warnings
    # only surface *after* corruption has been persisted.
    if before.channels and _ratio(len(after.channels), len(before.channels)) < 0.5:
        return (
            f"channels shrank from {len(before.channels)} to "
            f"{len(after.channels)} (>50% loss)"
        )
    if before.world_traits and _ratio(len(after.world_traits), len(before.world_traits)) < 0.5:
        return (
            f"world_traits shrank from {len(before.world_traits)} to "
            f"{len(after.world_traits)} (>50% loss)"
        )
    return None


_EVT_ID_RE = re.compile(r"\bEVT_[A-Za-z0-9_]+")


def _build_correction_subgraph(
    world_state: WorldStateV1,
    prog_errors: List["ValidationIssue"],
) -> Optional[str]:
    """Return a JSON subgraph that focuses on error-relevant events.

    Used by ``_run_correction_patch`` when the full WorldState exceeds
    ~60 KB and would otherwise crowd out the LLM's reasoning budget.
    The subgraph contains:

    * ontology header (locations, objects, entities — keys + names only);
    * every event whose id appears in any error detail;
    * every event reachable in one causal hop from a seed event;
    * the causal/spatial edges between any two seed/neighbour events.

    Returns ``None`` if no event ids could be extracted from the errors
    (the caller should then fall back to the full state).
    """
    seed_ids: set[str] = set()
    for err in prog_errors:
        for match in _EVT_ID_RE.findall(err.detail):
            seed_ids.add(match)
    if not seed_ids:
        return None

    valid_event_ids = {e.id for e in world_state.events}
    seed_ids &= valid_event_ids

    # One-hop causal expansion
    expanded = set(seed_ids)
    for ce in world_state.causal_topology:
        if ce.source_id in seed_ids and ce.target_id in valid_event_ids:
            expanded.add(ce.target_id)
        if ce.target_id in seed_ids and ce.source_id in valid_event_ids:
            expanded.add(ce.source_id)

    relevant_events = [e for e in world_state.events if e.id in expanded]
    relevant_causal = [
        ce for ce in world_state.causal_topology
        if ce.source_id in expanded and ce.target_id in expanded
    ]

    # Ontology header — names only, no nested timelines / beliefs.
    ontology = {
        "locations": {lid: loc.name for lid, loc in world_state.locations.items()},
        "objects": {oid: obj.name for oid, obj in world_state.objects.items()},
        "entities": {
            eid: {"name": ent.name, "location_id": ent.location_id, "status": ent.status}
            for eid, ent in world_state.entities.items()
        },
        "world_traits": {
            wid: wt.name for wid, wt in world_state.world_traits.items()
        },
    }

    import json as _json
    payload = {
        "_subgraph_note": (
            f"Error-relevant subgraph: {len(relevant_events)} of "
            f"{len(world_state.events)} events shown. Patch ids must "
            f"target the full WorldState."
        ),
        "ontology_header": ontology,
        "events": [e.model_dump(mode="json") for e in relevant_events],
        "causal_topology": [ce.model_dump(mode="json") for ce in relevant_causal],
    }
    return _json.dumps(payload, indent=2)


def _run_correction_patch(
    world_state: WorldStateV1,
    prog_errors: List["ValidationIssue"],
    config: ExtractionConfig,
    log_prefix: str,
) -> Tuple[WorldStateV1, List[str]]:
    """Run one correction-agent iteration and apply the resulting patch.

    Returns ``(new_world_state, repairs_applied)``. On any failure
    (LLM error, regression guard tripped, empty patch) the original
    *world_state* is returned unchanged and the failure is logged.
    """
    try:
        agent = _build_correction_patch_agent(config)
    except Exception:
        logger.exception("%s Failed to build correction patch agent.", log_prefix)
        return world_state, []

    ws_json = world_state.model_dump_json(indent=2)
    error_summary = "\n".join(
        f"  [{e.category}] {e.detail}" for e in prog_errors
    )

    # Item #15: when the serialized state is too large to fit comfortably
    # in a single LLM context window, send only the error-relevant
    # subgraph (events mentioned in the error details + their immediate
    # causal neighbours + ontology header) instead of the full state.
    # The patch contract still applies to the full state on the way out.
    SUBGRAPH_THRESHOLD = config.correction_subgraph_threshold_chars
    state_payload = ws_json
    payload_note = ""
    if len(ws_json) > SUBGRAPH_THRESHOLD:
        subgraph_json = _build_correction_subgraph(world_state, prog_errors)
        if subgraph_json is not None and len(subgraph_json) < len(ws_json):
            state_payload = subgraph_json
            payload_note = (
                "\n\nNOTE: The full WorldState is too large to fit in a "
                "single prompt. Only an error-relevant SUBGRAPH is shown "
                "below (events referenced by the errors + their immediate "
                "causal neighbours + the ontology header). Your patch "
                "MUST still target ids that exist in the full WorldState; "
                "do NOT add edges that depend on context you cannot see "
                "here.\n"
            )
            logger.info(
                "%s Using subgraph payload (%d \u2192 %d chars) for correction.",
                log_prefix, len(ws_json), len(subgraph_json),
            )

    correction_msg = (
        f"The following {len(prog_errors)} programmatic error(s) were found "
        f"in the WorldStateV1 below. Emit a WorldStatePatch that fixes ONLY "
        f"these errors. Do not re-emit the entire world state. Do not drop "
        f"unrelated edges, events, or entities. If you cannot determine a "
        f"safe fix, leave the patch empty and explain why in `notes`.\n\n"
        f"ERRORS:\n{error_summary}{payload_note}\n\n"
        f"WORLD STATE:\n{state_payload}"
    )

    try:
        result = agent.run_sync(correction_msg, **_user_kwargs())
    except Exception:
        logger.exception("%s Correction agent FAILED — keeping previous state.", log_prefix)
        return world_state, []

    patch: WorldStatePatch = result.output
    if (
        not patch.event_renames
        and not patch.channel_renames
        and not patch.drop_event_ids
        and not patch.update_event_fields
        and not patch.update_entity_location
        and not patch.add_state_timeline_entries
        and not patch.drop_causal_edges
        and not patch.add_causal_edges
        and not patch.drop_social_edges
        and not patch.add_social_edges
        and not patch.drop_spatial_edges
        and not patch.add_spatial_edges
        and not patch.drop_channel_ids
        and not patch.add_channels
    ):
        logger.info(
            "%s Correction agent returned an empty patch (notes: %r).",
            log_prefix, patch.notes,
        )
        return world_state, []

    try:
        new_ws, changes = _apply_world_state_patch(world_state, patch)
    except Exception:
        logger.exception(
            "%s Failed to apply correction patch — keeping previous state.",
            log_prefix,
        )
        return world_state, []

    regression = _is_correction_regression(world_state, new_ws)
    if regression:
        logger.warning(
            "%s Rejected correction patch: %s. Keeping previous state. "
            "Patch notes: %r",
            log_prefix, regression, patch.notes,
        )
        return world_state, []

    logger.info(
        "%s Applied correction patch: %d change(s). Notes: %r",
        log_prefix, len(changes), patch.notes,
    )
    return new_ws, changes


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
            **_user_kwargs(),
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
            sorted_timeline = sorted(extraction.timelines[wid], key=_snapshot_sort_key)
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
            **_user_kwargs(),
        )
        extraction = result.output
    except Exception:
        logger.exception("[Step 5·Async] World trait timeline extraction FAILED — skipping.")
        return ws

    updated_traits: Dict[str, GlobalTrait] = {}
    changes_applied = 0
    for wid, wt in ws.world_traits.items():
        if wid in extraction.timelines and extraction.timelines[wid]:
            sorted_timeline = sorted(extraction.timelines[wid], key=_snapshot_sort_key)
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


def _build_compact_validation_view(ws: WorldStateV1) -> str:
    """Return a topology-preserving compact JSON view for LLM validation.

    Keeps every node/edge id, name, type and reference-bearing field so
    the auditor can still spot cross-element contradictions, but strips
    the deep nested arrays (entity beliefs / state_timelines / world
    trait timelines) that dominate serialised size on long manuscripts.
    Used by ``validate_world_state`` when the full ``model_dump_json``
    exceeds the LLM context budget.
    """
    import json as _json

    payload = {
        "_compact_note": (
            "Compact projection: nested entity beliefs / state_timelines "
            "and world-trait state_timelines are omitted. Counts are "
            "given so the auditor can still flag missing-data anomalies."
        ),
        "locations": {
            lid: {"name": loc.name} for lid, loc in ws.locations.items()
        },
        "objects": {
            oid: {"name": obj.name, "owner_id": obj.owner_id}
            for oid, obj in ws.objects.items()
        },
        "entities": {
            eid: {
                "name": ent.name,
                "status": ent.status,
                "location_id": ent.location_id,
                "n_beliefs": len(ent.beliefs),
                "n_state_timeline": len(ent.state_timeline),
                "trait_keys": sorted(ent.traits.keys()),
                # Belief summaries: keep the (target_id, perceived_state,
                # confidence) triple so the LLM auditor can still spot
                # internal contradictions like "two confident beliefs
                # about the same target with opposite perceived_state".
                # Provenance + inertia + evidence_strength are dropped.
                "beliefs_summary": [
                    {
                        "target_id": b.target_id,
                        "perceived_state": b.perceived_state,
                        "confidence": round(float(getattr(b, "confidence", 1.0) or 1.0), 2),
                    }
                    for b in ent.beliefs
                ],
                # State-timeline summary: just the per-tick triggered_by
                # + status / location transitions and the *count* of new
                # beliefs / trait deltas. Lets the auditor catch missing
                # status transitions ("alive entity referenced after
                # EVT_X_KILLS_Y") and movement / location inconsistencies.
                "timeline_summary": [
                    {
                        "fabula_time": s.fabula_time,
                        "triggered_by": s.triggered_by,
                        "status": s.status,
                        "location_id": s.location_id,
                        "n_traits_changed": len(s.traits or {}),
                        "n_beliefs_added": len(s.beliefs_added or []),
                        "n_beliefs_invalidated": len(s.beliefs_invalidated or []),
                    }
                    for s in ent.state_timeline
                ],
            }
            for eid, ent in ws.entities.items()
        },
        "world_traits": {
            wid: {
                "name": wt.name,
                "n_state_timeline": len(wt.state_timeline),
                "timeline_summary": [
                    {
                        "fabula_time": s.fabula_time,
                        "triggered_by": s.triggered_by,
                    }
                    for s in wt.state_timeline
                ],
            }
            for wid, wt in ws.world_traits.items()
        },
        "events": [e.model_dump(mode="json") for e in ws.events],
        "causal_topology": [ce.model_dump(mode="json") for ce in ws.causal_topology],
        "spatial_topology": [se.model_dump(mode="json") for se in ws.spatial_topology],
        "channels": {
            cid: ch.model_dump(mode="json") for cid, ch in ws.channels.items()
        },
        "social_topology": [
            edge.model_dump(mode="json") for edge in ws.social_topology
        ],
    }
    return _json.dumps(payload, indent=2)


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
    # When the full state is too large, send a *compact* projection that
    # preserves topology (events + edges + ontology header) but drops the
    # verbose nested state_timeline and beliefs payloads, which dominate
    # serialized size and rarely host the kind of cross-element semantic
    # contradictions the LLM auditor catches. Prior behaviour silently
    # truncated the JSON tail \u2014 invisible to the auditor and skewed
    # corrections toward front-loaded sections (audit item #9).
    max_chars = config.validation_payload_max_chars
    if len(ws_json) > max_chars:
        compact_payload = _build_compact_validation_view(ws)
        if len(compact_payload) < len(ws_json):
            logger.info(
                "[Step 3\u00b7LLM] WorldState too large (%d chars); using "
                "compact projection (%d chars) for LLM audit.",
                len(ws_json), len(compact_payload),
            )
            ws_json = compact_payload
        else:
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

    try:
        result = agent.run_sync(
            preamble + f"Validate the following WorldStateV1 JSON:\n\n{ws_json}",
            **_user_kwargs(),
        )
        llm_report = result.output
        llm_issues = llm_report.issues
        llm_suggestions = llm_report.suggestions
    except Exception as e:  # noqa: BLE001 — never crash import on validator failure
        logger.warning(
            "[Step 3·LLM] Validation agent failed (%s: %s) — "
            "falling back to programmatic-only validation.",
            type(e).__name__, e,
        )
        llm_issues = []
        llm_suggestions = []

    # Merge programmatic + LLM issues
    all_issues = prog_issues + llm_issues
    has_errors = any(i.severity == "error" for i in all_issues)

    merged = ValidationReport(
        is_valid=not has_errors,
        issues=all_issues,
        suggestions=llm_suggestions,
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
    *,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
) -> Tuple[WorldStateV1, ValidationReport]:
    """
    Run the full 3-step extraction pipeline.

    Parameters
    ----------
    text : str
        Full narrative prose text.
    config : ExtractionConfig or None
        Pipeline configuration. Uses defaults if None.
    user_id : int, optional
        User ID for cost tracking and audit logging.
    project_id : int, optional
        Project ID for cost tracking context.
    version_id : int, optional 
        Version ID for cost tracking context.

    Returns
    -------
    (WorldStateV1, ValidationReport)
        The assembled world state and its validation report.
    """
    config = config or ExtractionConfig()
    logger.info("[Pipeline] Starting extraction with model=%s, strategy=%s", config.model, config.chunk_strategy)
    
    # Prepare user context for cost tracking
    user_context = {
        'user_id': user_id,
        'project_id': project_id,
        'version_id': version_id
    }
    # Set ContextVar so internal agent calls (extract_topology,
    # _run_correction_patch, extract_world_trait_timelines, validation,
    # research) can splat the same kwargs into ``agent.run_sync`` for
    # cost attribution — see ``_user_kwargs()``. The ``with`` block
    # captures the set token and resets it on exit (including
    # exceptions) so back-to-back extractions on the same thread don't
    # inherit stale attribution metadata.
    with _user_context_scope(user_context):
        # Step 1: Global Ontology
        # Step 1: Extract ontology
        register = extract_ontology(text, config, user_context)

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
        # If programmatic errors remain after auto-repair, attempt LLM correction.
        # Snapshot event identity so we can re-run trait timelines afterwards if
        # any patch renamed / dropped / time-shifted events (which would have
        # left the pre-correction world-trait snapshots anchored to stale events).
        pre_correction_event_signature = tuple(
            (e.id, e.fabula_time) for e in world_state.events
        )
        for retry_num in range(config.max_correction_retries):
            prog_errors = [i for i in report.issues if i.severity == "error"]
            if not prog_errors:
                break

            log_prefix = f"[Pipeline·Correction {retry_num + 1}/{config.max_correction_retries}]"
            logger.info(
                "%s %d errors remain — running patch-based correction agent.",
                log_prefix, len(prog_errors),
            )

            new_world_state, change_log = _run_correction_patch(
                world_state, prog_errors, config, log_prefix,
            )
            if not change_log:
                # No-op or rejected patch — retrying would just burn more tokens.
                break
            world_state = new_world_state
            repairs.extend(change_log)

            # Re-normalise, re-repair, and re-validate after the patch.
            world_state = _normalize_fabula_times(world_state, config.fabula_time_spacing)
            world_state, new_repairs = _auto_repair(world_state)
            if new_repairs:
                repairs.extend(new_repairs)
            report = validate_world_state(world_state, config)

        # If correction renamed, dropped, or time-shifted events the
        # pre-correction world-trait timelines may now reference stale ids
        # or wrong fabula ticks. Re-run timeline extraction once and re-
        # validate. Skipped when nothing relevant changed.
        post_correction_event_signature = tuple(
            (e.id, e.fabula_time) for e in world_state.events
        )
        if (
            post_correction_event_signature != pre_correction_event_signature
            and world_state.world_traits
        ):
            logger.info(
                "[Pipeline] Re-running world-trait timeline extraction after "
                "correction touched events.",
            )
            try:
                world_state = extract_world_trait_timelines(world_state, config)
                world_state, post_repairs = _auto_repair(world_state)
                if post_repairs:
                    repairs.extend(post_repairs)
                report = validate_world_state(world_state, config)
            except Exception:
                logger.exception(
                    "[Pipeline] Post-correction timeline re-extraction failed — "
                    "keeping pre-correction timelines.",
                )

        # ------------------------------------------------------------------
        # Step 3d — optional, segregated external research (post-assembly)
        # ------------------------------------------------------------------
        try:
            world_state = _run_research_step(world_state, config)
        except Exception:
            logger.exception("[Pipeline·Research] unexpected failure — continuing without research.")

        # ------------------------------------------------------------------
        # Step 3e — capture source narrative style for downstream fidelity
        # ------------------------------------------------------------------
        try:
            world_state.narrative_style = infer_narrative_style(text)
            logger.info(
                "[Pipeline] Narrative style inferred: format=%s, target=%d–%d words, density=%s.",
                world_state.narrative_style.format,
                world_state.narrative_style.target_word_min,
                world_state.narrative_style.target_word_max,
                world_state.narrative_style.prose_density,
            )
        except Exception:
            logger.exception("[Pipeline] Narrative-style inference failed — continuing without it.")

        return world_state, report

async def run_extraction_async(
    text: str,
    config: ExtractionConfig | None = None,
    *,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
) -> Tuple[WorldStateV1, ValidationReport]:
    """Async variant of :func:`run_extraction`.
    
    Parameters
    ----------
    text : str
        Full narrative prose text.
    config : ExtractionConfig or None
        Pipeline configuration. Uses defaults if None.
    user_id : int, optional
        User ID for cost tracking and audit logging.
    project_id : int, optional
        Project ID for cost tracking context.
    version_id : int, optional
        Version ID for cost tracking context.
        
    Returns
    -------
    (WorldStateV1, ValidationReport)
        The assembled world state and its validation report.
    """

    config = config or ExtractionConfig()
    logger.info("[Pipeline·Async] Starting extraction with model=%s, strategy=%s", config.model, config.chunk_strategy)

    # Prepare user context for cost tracking
    user_context = {
        'user_id': user_id,
        'project_id': project_id,
        'version_id': version_id,
    }
    # See ``run_extraction`` for rationale; ContextVar is async-task
    # local under ``asyncio``. Note: ``asyncio.to_thread`` calls below
    # automatically copy the current Context (via
    # ``contextvars.copy_context``), so the worker thread sees the same
    # ``_user_context_var`` as the orchestrator task.
    with _user_context_scope(user_context):
        # Step 1: Global Ontology (parallel 1b + 1c)
        # Step 1: Extract ontology
        register = await extract_ontology_async(text, config, user_context)

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

        # Validation calls a sync LLM agent internally; offload it to a
        # worker thread so we don't block the event loop while it runs
        # (audit item #7 — same rationale for ``_run_correction_patch``
        # below).
        report = await asyncio.to_thread(validate_world_state, world_state, config)

        # Snapshot pre-correction event identity so we can detect rename /
        # drop / time-shift and re-run trait timelines once afterwards.
        pre_correction_event_signature = tuple(
            (e.id, e.fabula_time) for e in world_state.events
        )

        # --- Correction retry loop ---
        for retry_num in range(config.max_correction_retries):
            prog_errors = [i for i in report.issues if i.severity == "error"]
            if not prog_errors:
                break

            log_prefix = f"[Pipeline·Async·Correction {retry_num + 1}/{config.max_correction_retries}]"
            logger.info(
                "%s %d errors remain — running patch-based correction agent.",
                log_prefix, len(prog_errors),
            )

            # ``_run_correction_patch`` calls ``agent.run_sync`` internally;
            # wrap it in ``to_thread`` so concurrent extraction tasks under
            # the same event loop are not stalled by the LLM round-trip.
            new_world_state, change_log = await asyncio.to_thread(
                _run_correction_patch,
                world_state, prog_errors, config, log_prefix,
            )
            if not change_log:
                break
            world_state = new_world_state
            repairs.extend(change_log)

            world_state = _normalize_fabula_times(world_state, config.fabula_time_spacing)
            world_state, new_repairs = _auto_repair(world_state)
            if new_repairs:
                repairs.extend(new_repairs)
            report = await asyncio.to_thread(validate_world_state, world_state, config)

        post_correction_event_signature = tuple(
            (e.id, e.fabula_time) for e in world_state.events
        )
        if (
            post_correction_event_signature != pre_correction_event_signature
            and world_state.world_traits
        ):
            logger.info(
                "[Pipeline·Async] Re-running world-trait timeline extraction "
                "after correction touched events.",
            )
            try:
                world_state = await extract_world_trait_timelines_async(world_state, config)
                world_state, post_repairs = _auto_repair(world_state)
                if post_repairs:
                    repairs.extend(post_repairs)
                report = await asyncio.to_thread(validate_world_state, world_state, config)
            except Exception:
                logger.exception(
                    "[Pipeline·Async] Post-correction timeline re-extraction failed "
                    "— keeping pre-correction timelines.",
                )

        # ------------------------------------------------------------------
        # Step 3d — optional, segregated external research (post-assembly)
        # ------------------------------------------------------------------
        try:
            world_state = await _run_research_step_async(world_state, config)
        except Exception:
            logger.exception("[Pipeline·Research·Async] unexpected failure — continuing without research.")

        try:
            world_state.narrative_style = infer_narrative_style(text)
            logger.info(
                "[Pipeline·Async] Narrative style inferred: format=%s, target=%d–%d words, density=%s.",
                world_state.narrative_style.format,
                world_state.narrative_style.target_word_min,
                world_state.narrative_style.target_word_max,
                world_state.narrative_style.prose_density,
            )
        except Exception:
            logger.exception("[Pipeline·Async] Narrative-style inference failed — continuing without it.")

        return world_state, report
