"""
Natural-language → structured query parsing agent.

Takes a free-form user request and an optional ``WorldStateV1``, then:
  1. Classifies the query into one of the six query types.
  2. For graph-intervention types (intervention, counterfactual, directive),
     resolves entity / event / object IDs from the world model.
  3. Validates that referenced IDs exist and the query is well-formed.
  4. Returns a fully populated ``UserRequest`` ready for ``run_pipeline``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput

from shadow_loom.models import WorldStateV1
from shadow_loom.query_models import (
    CounterfactualQuery,
    DirectiveQuery,
    GeneralQuery,
    InterrogationQuery,
    InterventionQuery,
    ObservationQuery,
    UserRequest,
)

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434/v1/"


# =====================================================================
# Configuration
# =====================================================================

class QueryParsingConfig(BaseModel):
    """Runtime configuration for the query parsing agent."""
    model: str = Field(
        default="ollama:qwen3.6:27b",
        description="PydanticAI model string.",
    )
    output_retries: int = Field(
        default=3,
        description="Max retries for output validation.",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for the classification response.",
    )
    temperature: float = Field(
        default=0.1,
        description="Low temperature for deterministic classification.",
    )


# =====================================================================
# LLM output schema — intermediate structured classification
# =====================================================================

class ResolvedID(BaseModel):
    """A single ID resolved from natural language."""
    natural_name: str = Field(description="The name as mentioned in the user query.")
    resolved_id: str = Field(description="The matching graph ID (ENT_, EVT_, OBJ_, LOC_).")
    confidence: float = Field(default=1.0, description="0.0-1.0 match confidence.")


class ParsedQuery(BaseModel):
    """Structured LLM output: classified query type + resolved parameters."""

    query_type: Literal[
        "observation", "intervention", "counterfactual",
        "directive", "interrogate", "general",
    ] = Field(description="The best-matching query type.")

    reasoning: str = Field(
        description="Brief explanation of why this query type was chosen.",
    )

    # --- Observation ---
    observations: Optional[Dict[str, str]] = Field(
        default=None,
        description="For observation: {node_id: observed_state} pairs.",
    )
    focus_entity_ids: Optional[List[str]] = Field(
        default=None,
        description="For observation/directive: entity IDs to focus on.",
    )

    # --- Intervention ---
    interventions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="For intervention: {node_id: new_state_or_spawn_dict} pairs.",
    )

    # --- Counterfactual ---
    historical_interventions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="For counterfactual: {event_id: altered_outcome} pairs.",
    )
    evidence_node_ids: Optional[List[str]] = Field(
        default=None,
        description="For counterfactual: present-tense facts to condition on.",
    )

    # --- Directive ---
    target_entity_ids: Optional[List[str]] = Field(
        default=None,
        description="For directive: entities experiencing the effect.",
    )
    target_effect: Optional[Literal[
        "suspense", "surprise", "mystery", "dramatic_irony",
        "grief", "rage", "joy", "regret", "love", "fear",
    ]] = Field(default=None, description="For directive: the narrative effect.")
    target_vector_id: Optional[str] = Field(
        default=None,
        description="For directive: the trait/edge/objective node to target.",
    )
    intensity: Optional[float] = Field(
        default=None,
        description="For directive: 0.0-1.0 intensity multiplier.",
    )

    # --- Interrogation / General ---
    question: Optional[str] = Field(
        default=None,
        description="For interrogate/general: the question to answer.",
    )
    require_proof: Optional[bool] = Field(
        default=None,
        description="For interrogate: whether to require causal proof.",
    )
    include_topology: Optional[bool] = Field(
        default=None,
        description="For general: whether to include full topology.",
    )

    # --- Resolved IDs ---
    resolved_ids: List[ResolvedID] = Field(
        default_factory=list,
        description="All entity/event/object/location IDs resolved from natural language.",
    )


# =====================================================================
# Validation result
# =====================================================================

class ValidationError(BaseModel):
    """A single validation issue."""
    field: str
    message: str
    severity: Literal["error", "warning"] = "error"


class FallbackInfo(BaseModel):
    """Records what fallback strategy was applied and why."""
    strategy: Literal[
        "fuzzy_id_resolution", "general_fallback", "none",
    ] = Field(description="Which fallback strategy was used.")
    reason: str = Field(description="Human-readable explanation of the fallback.")
    original_query_type: Optional[str] = Field(
        default=None,
        description="The query type the LLM originally chose (before fallback).",
    )
    id_remappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of bad_id → corrected_id for fuzzy resolutions.",
    )
    original_errors: List[ValidationError] = Field(
        default_factory=list,
        description="The validation errors that triggered the fallback.",
    )


class QueryParseResult(BaseModel):
    """Complete result of query parsing: the built query + diagnostics."""
    query: Optional[UserRequest] = Field(
        default=None,
        description="The fully constructed query, or None if validation failed.",
    )
    parsed: ParsedQuery = Field(
        description="The raw LLM classification output.",
    )
    validation_errors: List[ValidationError] = Field(
        default_factory=list,
        description="Any validation issues found.",
    )
    is_valid: bool = Field(
        default=True,
        description="Whether the query passed all validation checks.",
    )
    fallback: Optional[FallbackInfo] = Field(
        default=None,
        description="Non-None when a fallback strategy was applied.",
    )


# =====================================================================
# World-model graph summary (compact representation for the LLM)
# =====================================================================

def _build_graph_summary(world_state: WorldStateV1) -> str:
    """Build a compact text summary of all IDs in the world model.

    Keeps the token count manageable while giving the LLM enough
    context to resolve natural-language references to graph IDs.
    """
    sections: list[str] = []

    # Entities
    if world_state.entities:
        sections.append("ENTITIES:")
        for eid, ent in world_state.entities.items():
            traits = ", ".join(
                f"{k}={v.value:.2f}" for k, v in ent.traits.items()
            )
            sections.append(
                f"  {eid}: {ent.name} | status={ent.status} | "
                f"location={ent.location_id} | traits=[{traits}]"
            )

    # Locations
    if world_state.locations:
        sections.append("LOCATIONS:")
        for lid, loc in world_state.locations.items():
            sections.append(f"  {lid}: {loc.name} — {loc.description[:80]}")

    # Objects
    if world_state.objects:
        sections.append("OBJECTS:")
        for oid, obj in world_state.objects.items():
            loc_or_owner = (
                f"owner={obj.owner_id}" if obj.owner_id
                else f"at {obj.location_id}" if obj.location_id
                else "unplaced"
            )
            props = ", ".join(f"{k}={v}" for k, v in obj.properties.items())
            affordances = ", ".join(a.action for a in obj.affordances)
            sections.append(
                f"  {oid}: {obj.name} | {loc_or_owner} | "
                f"props=[{props}] | can=[{affordances}]"
            )

    # Events (sorted by fabula_time)
    if world_state.events:
        sections.append("EVENTS (chronological):")
        for evt in sorted(world_state.events, key=lambda e: e.fabula_time):
            actors = ", ".join(evt.actor_ids) if evt.actor_ids else "none"
            targets = ", ".join(evt.target_ids) if evt.target_ids else "none"
            sections.append(
                f"  {evt.id}: t={evt.fabula_time} [{evt.event_type}] "
                f"actors=[{actors}] targets=[{targets}] — {evt.description[:80]}"
            )

    # Relationships
    if world_state.social_topology:
        sections.append("RELATIONSHIPS:")
        for rel in world_state.social_topology:
            sections.append(
                f"  {rel.source_entity_id}→{rel.target_entity_id}: "
                f"affinity={rel.affinity:.2f}, fear={rel.fear:.2f}, "
                f"power={rel.power_dynamic:.2f}"
            )

    return "\n".join(sections)


def _collect_all_ids(world_state: WorldStateV1) -> set[str]:
    """Collect every valid ID in the world model for validation."""
    ids: set[str] = set()
    ids.update(world_state.entities.keys())
    ids.update(world_state.locations.keys())
    ids.update(world_state.objects.keys())
    ids.update(e.id for e in world_state.events)
    return ids


# =====================================================================
# System prompt
# =====================================================================

_SYSTEM_PROMPT = """\
You are a query parsing agent for a narrative simulation engine called Shadow Loom.

Your job is to take a natural-language user request about a story world and \
classify it into exactly one of six query types, then extract the structured \
parameters needed to execute that query.

## THE SIX QUERY TYPES

1. **observation** — "What happens next?" / "Show me the scene from X's perspective."
   Advances time naturally. May condition on observed facts. May lock POV to specific entities.
   Use when: the user wants to see what happens, observe a scene, or get a character's POV.

2. **intervention** — "Make X do Y" / "Set trait Z to 0.9" / "Spawn a new object."
   Forces variables to specific states (do-operator). Cuts incoming causal edges.
   Use when: the user wants to forcibly change something in the present moment.
   Values can be strings (state changes) OR dicts (genesis spawns) OR numbers (trait values).

3. **counterfactual** — "What if X had never happened?" / "What if Y chose differently?"
   Travels back in time, changes past events, conditions on present evidence, re-simulates.
   Use when: the user asks "what if" about PAST events.

4. **directive** — "Maximise suspense" / "Make the reader feel grief" / "Create dramatic irony."
   Optimises the next event for a specific narrative/emotional effect.
   Effects: suspense, surprise, mystery, dramatic_irony, grief, rage, joy, regret, love, fear.
   Use when: the user wants to control the FEELING or EFFECT of the next scene.

5. **interrogate** — "Is there a path from A to B?" / "Who caused X?"
   Graph pathfinding / RAG query. Does NOT advance time or generate prose.
   Use when: the user wants factual answers about the graph structure.

6. **general** — Any open-ended question about the world state.
   Full-graph Q&A without advancing time.
   Use when: the question doesn't fit the other types, or asks broad analytical questions.

## ID RESOLUTION RULES

- Entity IDs start with ENT_ (e.g., ENT_MACBETH)
- Event IDs start with EVT_ (e.g., EVT_DUNCAN_MURDER)
- Object IDs start with OBJ_ (e.g., OBJ_DAGGER)
- Location IDs start with LOC_ (e.g., LOC_CASTLE)

When the user mentions a character, place, object, or event by name, resolve it \
to the correct graph ID from the provided world model summary. If no world model \
is provided, use reasonable ID conventions (ENT_CHARACTERNAME).

## OUTPUT RULES

- Set ONLY the fields relevant to the chosen query_type. Leave others null.
- For intervention: each key in `interventions` must be a valid node ID.
  String values = state changes. Dict values = genesis spawns. Number values = trait overrides.
- For counterfactual: `historical_interventions` keys should be EVT_ IDs.
  `evidence_node_ids` should be present-tense node IDs.
- For directive: `target_entity_ids` is required. `target_effect` is required.
  `target_vector_id` is the specific trait/edge/event to target (optional).
- For observation: `focus_entity_ids` is optional (omit for omniscient view).
- Always populate `resolved_ids` with every entity/event/object/location you resolved.
- Always provide `reasoning` explaining your classification.
"""


# =====================================================================
# Model resolution (shared with generation.py)
# =====================================================================

def _resolve_model(model_str: str):
    """Resolve a model string to a PydanticAI model instance."""
    if model_str.startswith("ollama:"):
        base_url = os.environ.get("OLLAMA_BASE_URL", _OLLAMA_BASE_URL)
        model_name = model_str.split(":", 1)[1]
        from pydantic_ai.models.ollama import OllamaModel
        from pydantic_ai.providers.ollama import OllamaProvider
        return OllamaModel(model_name, provider=OllamaProvider(base_url=base_url))
    return model_str


# =====================================================================
# Validation
# =====================================================================

def _validate_parsed_query(
    parsed: ParsedQuery,
    world_state: Optional[WorldStateV1],
) -> list[ValidationError]:
    """Validate the parsed query against the world model."""
    errors: list[ValidationError] = []
    valid_ids = _collect_all_ids(world_state) if world_state else None

    def _check_id(node_id: str, field_name: str) -> None:
        if valid_ids is not None and node_id not in valid_ids:
            errors.append(ValidationError(
                field=field_name,
                message=f"ID '{node_id}' not found in world model.",
            ))

    def _check_ids(node_ids: list[str], field_name: str) -> None:
        for nid in node_ids:
            _check_id(nid, field_name)

    qt = parsed.query_type

    if qt == "observation":
        if parsed.focus_entity_ids:
            _check_ids(parsed.focus_entity_ids, "focus_entity_ids")
        if parsed.observations:
            for nid in parsed.observations:
                _check_id(nid, "observations")

    elif qt == "intervention":
        if not parsed.interventions:
            errors.append(ValidationError(
                field="interventions",
                message="Intervention query requires at least one intervention.",
            ))
        else:
            for nid in parsed.interventions:
                _check_id(nid, "interventions")

    elif qt == "counterfactual":
        if not parsed.historical_interventions:
            errors.append(ValidationError(
                field="historical_interventions",
                message="Counterfactual query requires at least one historical intervention.",
            ))
        else:
            for nid in parsed.historical_interventions:
                _check_id(nid, "historical_interventions")
        if not parsed.evidence_node_ids:
            errors.append(ValidationError(
                field="evidence_node_ids",
                message="Counterfactual query requires at least one evidence node.",
                severity="warning",
            ))
        elif valid_ids:
            _check_ids(parsed.evidence_node_ids, "evidence_node_ids")

    elif qt == "directive":
        if not parsed.target_entity_ids:
            errors.append(ValidationError(
                field="target_entity_ids",
                message="Directive query requires at least one target entity.",
            ))
        else:
            _check_ids(parsed.target_entity_ids, "target_entity_ids")
        if not parsed.target_effect:
            errors.append(ValidationError(
                field="target_effect",
                message="Directive query requires a target_effect.",
            ))
        if parsed.target_vector_id:
            _check_id(parsed.target_vector_id, "target_vector_id")
        if parsed.intensity is not None and not (0.0 <= parsed.intensity <= 1.0):
            errors.append(ValidationError(
                field="intensity",
                message=f"Intensity must be 0.0-1.0, got {parsed.intensity}.",
            ))

    elif qt == "interrogate":
        if not parsed.question:
            errors.append(ValidationError(
                field="question",
                message="Interrogation query requires a question.",
            ))

    elif qt == "general":
        if not parsed.question:
            errors.append(ValidationError(
                field="question",
                message="General query requires a question.",
            ))

    # Check resolved IDs
    for rid in parsed.resolved_ids:
        _check_id(rid.resolved_id, f"resolved_ids[{rid.natural_name}]")

    return errors


# =====================================================================
# Fuzzy ID resolution
# =====================================================================

def _normalise_id_name(raw: str) -> str:
    """Lowercase, strip prefixes, collapse non-alphanumeric to underscores."""
    s = raw.strip().upper()
    for prefix in ("ENT_", "EVT_", "OBJ_", "LOC_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return re.sub(r"[^A-Z0-9]+", "_", s).strip("_")


def _build_name_index(world_state: WorldStateV1) -> Dict[str, str]:
    """Build a normalised-name → graph-ID lookup from the world model.

    Also indexes entity/object/location display names so the fuzzy
    matcher can resolve "Macbeth" → "ENT_MACBETH".
    """
    index: Dict[str, str] = {}

    for eid, ent in world_state.entities.items():
        index[_normalise_id_name(eid)] = eid
        index[_normalise_id_name(ent.name)] = eid

    for lid, loc in world_state.locations.items():
        index[_normalise_id_name(lid)] = lid
        index[_normalise_id_name(loc.name)] = lid

    for oid, obj in world_state.objects.items():
        index[_normalise_id_name(oid)] = oid
        index[_normalise_id_name(obj.name)] = oid

    for evt in world_state.events:
        index[_normalise_id_name(evt.id)] = evt.id
        index[_normalise_id_name(evt.description[:60])] = evt.id

    return index


def _fuzzy_resolve_id(
    bad_id: str,
    name_index: Dict[str, str],
    threshold: float = 0.6,
) -> Optional[str]:
    """Find the best fuzzy match for *bad_id* in the name index.

    Returns the real graph ID if the best match scores >= *threshold*,
    otherwise ``None``.
    """
    normalised = _normalise_id_name(bad_id)
    # Exact hit?
    if normalised in name_index:
        return name_index[normalised]

    best_score = 0.0
    best_id: Optional[str] = None
    for key, real_id in name_index.items():
        score = SequenceMatcher(None, normalised, key).ratio()
        if score > best_score:
            best_score = score
            best_id = real_id
    return best_id if best_score >= threshold else None


def _try_fuzzy_repair(
    parsed: ParsedQuery,
    errors: List[ValidationError],
    world_state: WorldStateV1,
) -> Tuple[ParsedQuery, Dict[str, str], List[ValidationError]]:
    """Attempt to repair unknown-ID validation errors via fuzzy matching.

    Returns (repaired_parsed, id_remappings, remaining_errors).
    ``id_remappings`` maps original-bad-id → corrected-id for every
    successful resolution.
    """
    name_index = _build_name_index(world_state)
    valid_ids = _collect_all_ids(world_state)
    id_errors = [
        e for e in errors
        if e.severity == "error" and "not found in world model" in e.message
    ]
    if not id_errors:
        return parsed, {}, errors

    remappings: Dict[str, str] = {}
    for err in id_errors:
        # Extract the bad ID from the message: "ID 'XXX' not found ..."
        match = re.search(r"ID '([^']+)'", err.message)
        if not match:
            continue
        bad_id = match.group(1)
        resolved = _fuzzy_resolve_id(bad_id, name_index)
        if resolved and resolved in valid_ids:
            remappings[bad_id] = resolved

    if not remappings:
        return parsed, {}, errors

    # Deep-copy the parsed output and apply remappings
    patched = parsed.model_copy(deep=True)

    def _remap_dict_keys(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if d is None:
            return None
        return {remappings.get(k, k): v for k, v in d.items()}

    def _remap_list(lst: Optional[List[str]]) -> Optional[List[str]]:
        if lst is None:
            return None
        return [remappings.get(x, x) for x in lst]

    patched.observations = _remap_dict_keys(patched.observations)
    patched.interventions = _remap_dict_keys(patched.interventions)
    patched.historical_interventions = _remap_dict_keys(patched.historical_interventions)
    patched.focus_entity_ids = _remap_list(patched.focus_entity_ids)
    patched.evidence_node_ids = _remap_list(patched.evidence_node_ids)
    patched.target_entity_ids = _remap_list(patched.target_entity_ids)
    if patched.target_vector_id and patched.target_vector_id in remappings:
        patched.target_vector_id = remappings[patched.target_vector_id]
    for rid in patched.resolved_ids:
        if rid.resolved_id in remappings:
            rid.resolved_id = remappings[rid.resolved_id]

    # Re-validate the patched query
    remaining = _validate_parsed_query(patched, world_state)
    return patched, remappings, remaining


def _build_general_fallback(
    natural_language: str,
    parsed: ParsedQuery,
    errors: List[ValidationError],
) -> QueryParseResult:
    """Last-resort fallback: wrap the original request as a general query."""
    fallback_parsed = ParsedQuery(
        query_type="general",
        reasoning=(
            f"Fallback: original {parsed.query_type!r} query could not be validated. "
            f"Errors: {'; '.join(e.message for e in errors)}"
        ),
        question=natural_language,
        include_topology=True,
    )
    query = _build_query(fallback_parsed)
    return QueryParseResult(
        query=query,
        parsed=fallback_parsed,
        validation_errors=[],
        is_valid=True,
        fallback=FallbackInfo(
            strategy="general_fallback",
            reason=(
                f"Could not validate {parsed.query_type!r} query — "
                f"falling back to general Q&A."
            ),
            original_query_type=parsed.query_type,
            original_errors=errors,
        ),
    )


def _apply_fallback(
    natural_language: str,
    parsed: ParsedQuery,
    errors: List[ValidationError],
    world_state: Optional[WorldStateV1],
) -> QueryParseResult:
    """Attempt fuzzy repair, then general-query fallback.

    Called when initial validation produces hard errors.
    """
    # --- Stage 1: fuzzy ID repair (only possible with a world state) ---
    if world_state is not None:
        id_errors = [
            e for e in errors
            if e.severity == "error" and "not found in world model" in e.message
        ]
        if id_errors:
            patched, remappings, remaining = _try_fuzzy_repair(
                parsed, errors, world_state,
            )
            has_remaining_hard = any(e.severity == "error" for e in remaining)
            if not has_remaining_hard and remappings:
                query = _build_query(patched)
                logger.info(
                    "[QueryParser] Fuzzy repair succeeded — remapped %s.",
                    remappings,
                )
                return QueryParseResult(
                    query=query,
                    parsed=patched,
                    validation_errors=remaining,
                    is_valid=True,
                    fallback=FallbackInfo(
                        strategy="fuzzy_id_resolution",
                        reason=(
                            f"Resolved {len(remappings)} unknown ID(s) via "
                            f"fuzzy matching."
                        ),
                        original_query_type=parsed.query_type,
                        id_remappings=remappings,
                        original_errors=errors,
                    ),
                )

    # --- Stage 2: fall back to general query ---
    logger.info(
        "[QueryParser] Fuzzy repair insufficient — falling back to general query.",
    )
    return _build_general_fallback(natural_language, parsed, errors)


# =====================================================================
# Query construction
# =====================================================================

def _build_query(parsed: ParsedQuery) -> UserRequest:
    """Construct a concrete query model from the parsed classification."""
    qt = parsed.query_type

    if qt == "observation":
        return ObservationQuery(
            observations=parsed.observations or {},
            focus_entity_ids=parsed.focus_entity_ids or [],
        )

    if qt == "intervention":
        return InterventionQuery(
            interventions=parsed.interventions or {},
        )

    if qt == "counterfactual":
        return CounterfactualQuery(
            historical_interventions=parsed.historical_interventions or {},
            evidence_node_ids=parsed.evidence_node_ids or [],
        )

    if qt == "directive":
        return DirectiveQuery(
            target_entity_ids=parsed.target_entity_ids or [],
            target_effect=parsed.target_effect or "suspense",
            target_vector_id=parsed.target_vector_id,
            intensity=parsed.intensity if parsed.intensity is not None else 1.0,
        )

    if qt == "interrogate":
        return InterrogationQuery(
            question=parsed.question or "",
            require_proof=parsed.require_proof if parsed.require_proof is not None else True,
        )

    # general
    return GeneralQuery(
        question=parsed.question or "",
        include_topology=parsed.include_topology if parsed.include_topology is not None else True,
    )


# =====================================================================
# Main entry point
# =====================================================================

def parse_query(
    natural_language: str,
    *,
    world_state: Optional[WorldStateV1] = None,
    config: Optional[QueryParsingConfig] = None,
) -> QueryParseResult:
    """Parse a natural-language request into a structured pipeline query.

    Parameters
    ----------
    natural_language : str
        The user's free-form request.
    world_state : WorldStateV1 or None
        The current world model. When provided, the agent resolves
        entity/event/object names to graph IDs and validates them.
    config : QueryParsingConfig or None
        LLM configuration. Uses defaults if not provided.

    Returns
    -------
    QueryParseResult
        Contains the constructed query (if valid), the raw parsed output,
        and any validation errors.
    """
    cfg = config or QueryParsingConfig()
    model = _resolve_model(cfg.model)

    # Build user message
    user_parts: list[str] = []
    if world_state:
        user_parts.append("## WORLD MODEL\n")
        user_parts.append(_build_graph_summary(world_state))
        user_parts.append("\n")
    user_parts.append("## USER REQUEST\n")
    user_parts.append(natural_language)

    user_message = "\n".join(user_parts)

    agent: Agent[None, ParsedQuery] = Agent(
        model,
        system_prompt=_SYSTEM_PROMPT,
        output_type=NativeOutput(ParsedQuery),
        retries=cfg.output_retries,
    )

    logger.info("[QueryParser] Classifying: %s", natural_language[:120])

    result = agent.run_sync(
        user_message,
        model_settings={
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        },
    )
    parsed: ParsedQuery = result.output

    # Validate
    errors = _validate_parsed_query(parsed, world_state)
    has_hard_errors = any(e.severity == "error" for e in errors)

    if has_hard_errors:
        logger.warning(
            "[QueryParser] Validation failed with %d error(s) — attempting fallback: %s",
            len(errors),
            "; ".join(e.message for e in errors),
        )
        return _apply_fallback(natural_language, parsed, errors, world_state)

    # Build the concrete query
    query = _build_query(parsed)
    logger.info("[QueryParser] Resolved to %s query", parsed.query_type)

    return QueryParseResult(
        query=query,
        parsed=parsed,
        validation_errors=errors,  # may contain warnings
        is_valid=True,
    )


async def parse_query_async(
    natural_language: str,
    *,
    world_state: Optional[WorldStateV1] = None,
    config: Optional[QueryParsingConfig] = None,
) -> QueryParseResult:
    """Async version of :func:`parse_query`."""
    cfg = config or QueryParsingConfig()
    model = _resolve_model(cfg.model)

    user_parts: list[str] = []
    if world_state:
        user_parts.append("## WORLD MODEL\n")
        user_parts.append(_build_graph_summary(world_state))
        user_parts.append("\n")
    user_parts.append("## USER REQUEST\n")
    user_parts.append(natural_language)

    user_message = "\n".join(user_parts)

    agent: Agent[None, ParsedQuery] = Agent(
        model,
        system_prompt=_SYSTEM_PROMPT,
        output_type=NativeOutput(ParsedQuery),
        retries=cfg.output_retries,
    )

    logger.info("[QueryParser] Classifying (async): %s", natural_language[:120])

    result = await agent.run(
        user_message,
        model_settings={
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        },
    )
    parsed: ParsedQuery = result.output

    errors = _validate_parsed_query(parsed, world_state)
    has_hard_errors = any(e.severity == "error" for e in errors)

    if has_hard_errors:
        logger.warning(
            "[QueryParser] Validation failed with %d error(s) — attempting fallback: %s",
            len(errors),
            "; ".join(e.message for e in errors),
        )
        return _apply_fallback(natural_language, parsed, errors, world_state)

    query = _build_query(parsed)
    logger.info("[QueryParser] Resolved to %s query", parsed.query_type)

    return QueryParseResult(
        query=query,
        parsed=parsed,
        validation_errors=errors,
        is_valid=True,
    )
