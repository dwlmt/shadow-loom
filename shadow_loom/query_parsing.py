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

from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_ai import Agent, PromptedOutput, ModelRetry

from shadow_loom.models import WorldStateV1
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

from shadow_loom.settings import get_settings as _get_settings, resolve_model as _resolve_model
from shadow_loom._agent_logging import log_agent_output

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

def _qp_defaults() -> dict:
    return _get_settings().query_parsing_config()


class QueryParsingConfig(BaseModel):
    """Runtime configuration for the query parsing agent."""
    model: str = Field(
        default="ollama:qwen3.6:27b",
        description="PydanticAI model string.",
    )
    output_retries: int = Field(
        default=5,
        description="Max retries for output validation.",
    )
    max_tokens: int = Field(
        default=64000,
        description="Maximum tokens for the classification response.",
    )
    temperature: float = Field(
        default=0.1,
        description="Low temperature for deterministic classification.",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_from_settings(cls, data):
        if isinstance(data, dict):
            for k, v in _qp_defaults().items():
                data.setdefault(k, v)
        return data


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
        "directive", "interrogate", "general", "manual_edit", "evaluate",
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

    # --- Shared by intervention + counterfactual ---
    target_node_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "For intervention/counterfactual: downstream nodes the user "
            "cares about. Used as the Y-set for the ctf-calculus Rule 3 "
            "(Exclusion) pre-flight to prove that an intervention is "
            "vacuous when no path exists to any of these nodes."
        ),
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

    # --- Manual Edit ---
    edited_prose: Optional[str] = Field(
        default=None,
        description="For manual_edit: the user's written/edited narrative prose.",
    )
    edit_description: Optional[str] = Field(
        default=None,
        description="For manual_edit: optional description of the changes.",
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
    parsed: Optional[ParsedQuery] = Field(
        default=None,
        description="The raw LLM classification output, or None if parsing failed before completion.",
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

    # Relationships — emit per-axis evidence + observed flag so the
    # planner LLM can distinguish ``affinity=0 (observed, strong)`` from
    # ``affinity=0 (unobserved)`` when reasoning about the world. The
    # legacy flat property accessors collapse both cases to the same
    # 0.00 string and silently lose the new signal.
    if world_state.social_topology:
        sections.append("RELATIONSHIPS:")
        es_short = {"weak": "w", "moderate": "m", "strong": "s"}
        for rel in world_state.social_topology:
            parts = []
            for axis_short, axis_name in (
                ("aff", "affinity"), ("fear", "fear"), ("pow", "power_dynamic"),
            ):
                m = rel.metrics.get(axis_name)
                if m is None:
                    parts.append(f"{axis_short}=–")
                elif not m.observed:
                    parts.append(f"{axis_short}=? (unobs)")
                else:
                    parts.append(
                        f"{axis_short}={m.value:+.2f}[{es_short.get(m.evidence_strength, 'm')}]"
                    )
            sections.append(
                f"  {rel.source_entity_id}→{rel.target_entity_id}: "
                + ", ".join(parts)
            )

    # World Traits
    if world_state.world_traits:
        sections.append("WORLD TRAITS:")
        for wid, wt in world_state.world_traits.items():
            domains = ", ".join(wt.affected_domains)
            sections.append(
                f"  {wid}: {wt.name} | mag={wt.magnitude.value:.2f} | "
                f"domains=[{domains}] — {wt.description[:80]}"
            )

    return "\n".join(sections)


def _collect_all_ids(world_state: WorldStateV1) -> set[str]:
    """Collect every valid ID in the world model for validation."""
    ids: set[str] = set()
    ids.update(world_state.entities.keys())
    ids.update(world_state.locations.keys())
    ids.update(world_state.objects.keys())
    ids.update(world_state.world_traits.keys())
    ids.update(e.id for e in world_state.events)
    return ids


def _collect_typed_ids(world_state: WorldStateV1) -> Dict[str, list[str]]:
    """Collect all IDs from the world model bucketed by node type.

    Used to drive both the categorised "VALID GRAPH IDS" prompt block
    and the dynamic ``Literal[...]`` constraints applied to structured
    output schemas.
    """
    return {
        "entity_ids": list(world_state.entities.keys()),
        "object_ids": list(world_state.objects.keys()),
        "location_ids": list(world_state.locations.keys()),
        "event_ids": [e.id for e in world_state.events],
        "world_trait_ids": list(world_state.world_traits.keys()),
        "channel_ids": list(getattr(world_state, "channels", {}).keys()),
        "utterance_event_ids": [
            e.id for e in world_state.events
            if getattr(e, "event_type", None) == "utterance"
        ],
    }


def _format_valid_ids_section(world_state: WorldStateV1) -> str:
    """Render a compact "VALID GRAPH IDS" prompt block grouped by node type.

    Listed alongside the full graph summary so the LLM has an explicit
    enumeration of every ID it is allowed to emit. Matches the
    ``Literal[...]`` enum the dynamic structured-output schema enforces.
    """
    lines = [
        "## VALID GRAPH IDS (use these EXACTLY — structured output is constrained to these enums)",
    ]

    if world_state.entities:
        lines.append(f"Entities ({len(world_state.entities)}):")
        for eid, ent in world_state.entities.items():
            lines.append(f"  - {eid}  ({ent.name})")

    if world_state.objects:
        lines.append(f"Objects ({len(world_state.objects)}):")
        for oid, obj in world_state.objects.items():
            lines.append(f"  - {oid}  ({obj.name})")

    if world_state.locations:
        lines.append(f"Locations ({len(world_state.locations)}):")
        for lid, loc in world_state.locations.items():
            lines.append(f"  - {lid}  ({loc.name})")

    if world_state.events:
        lines.append(f"Events ({len(world_state.events)}):")
        for evt in sorted(world_state.events, key=lambda e: e.fabula_time):
            lines.append(
                f"  - {evt.id}  (t={evt.fabula_time}, {evt.event_type}: "
                f"{evt.description[:60]})"
            )

    if world_state.world_traits:
        lines.append(f"World Traits ({len(world_state.world_traits)}):")
        for wid, wt in world_state.world_traits.items():
            lines.append(f"  - {wid}  ({wt.name})")

    return "\n".join(lines)


# =====================================================================
# Dynamic, ID-constrained structured output schemas
# =====================================================================
#
# For the four query types whose semantics require *exact* matches
# against the world model — intervention (rung 2), counterfactual
# (rung 3), directive, and interrogation — we generate a fresh
# Pydantic model per call whose ID-bearing fields are typed as
# ``Literal[<valid IDs>]``. NativeOutput then forces the LLM to emit
# only IDs that exist in the graph.
#
# The dynamic output is normalised back into the static
# :class:`ParsedQuery` shape so all downstream validation, fallback,
# and query-construction code keeps working unchanged.

#: Query types that benefit from constrained structured output.
_CONSTRAINED_QUERY_TYPES: set[str] = {
    "intervention", "counterfactual", "directive", "interrogate",
}

#: Allowed property roots per node prefix, advertised to the LLM via the
#: prompt and (loosely) validated by ``_validate_property_path``. Keep
#: in sync with ``_VALID_PROPERTY_ROOTS`` further down.
_PROPERTIES_BY_PREFIX: Dict[str, list[str]] = {
    "ENT": [
        "status", "location_id", "traits", "beliefs", "constants",
        "communicating_with", "spawn",
    ],
    "OBJ": ["owner_id", "location_id", "properties", "affordances", "spawn"],
    "LOC": ["ambient_state", "description", "spawn"],
    "EVT": [
        "event_type", "description", "actor_ids", "target_ids", "outcome",
        # Utterance-specific fields (event_type == 'utterance').
        "via_channel_id", "speaker_id", "addressee_ids", "truth_value",
        "content", "spawn",
    ],
    "WORLD": ["magnitude", "description", "affected_domains", "spawn"],
    # Communication channels are first-class graph nodes; they support
    # surgery on participant set, intelligibility map, status, and the
    # standing capability itself.
    "CHAN": [
        "medium", "participant_ids", "intelligibility", "directionality",
        "status", "spawn",
    ],
}


def _make_id_literal(ids: list[str]):
    """Build a ``Literal[id1, id2, ...]`` type from a list of IDs.

    Returns plain ``str`` when the list is empty so that
    ``create_model`` doesn't choke on an empty enum (the surrounding
    ``parse_query`` will simply skip the constrained code path when no
    IDs of the relevant type exist in the world).
    """
    if not ids:
        return str
    # ``Literal[tuple(ids)]`` works because PEP 604 / typing subscript
    # accepts a tuple in the same way as ``Literal[id1, id2, ...]``.
    return Literal[tuple(ids)]  # type: ignore[valid-type]


def _properties_help_block() -> str:
    """Plain-text reference of allowed `<id>.<property>` roots per prefix."""
    lines = ["Allowed `property` values per node-ID prefix:"]
    for prefix, props in _PROPERTIES_BY_PREFIX.items():
        lines.append(f"  - {prefix}_*: {', '.join(props)}")
    lines.append(
        "Use sub-paths after a dot for nested mutations, e.g. "
        "`traits.guilt`, `beliefs.B_FOO`, `properties.locked`."
    )
    return "\n".join(lines)


def _build_intervention_dynamic_model(world_state: WorldStateV1):
    """Create a per-call Pydantic model for the *intervention* query type.

    The output is a list of ``InterventionItem`` records. Each item's
    ``target_id`` is constrained to a ``Literal`` of all valid graph
    IDs in the world; ``property`` and ``value`` are free-form strings
    / Any so the LLM can express the full mutation surface.
    """
    typed = _collect_typed_ids(world_state)
    all_ids = (
        typed["entity_ids"] + typed["object_ids"] + typed["location_ids"]
        + typed["event_ids"] + typed["world_trait_ids"]
    )
    target_lit = _make_id_literal(all_ids)

    Item = create_model(
        "InterventionItem",
        target_id=(
            target_lit,
            Field(..., description="Graph node ID to mutate. MUST be an exact match."),
        ),
        property=(
            str,
            Field(
                ...,
                description=(
                    "Property path on the target node, e.g. 'status', "
                    "'location_id', 'traits.guilt', 'event_type', 'spawn'. "
                    + _properties_help_block()
                ),
            ),
        ),
        value=(
            Any,
            Field(
                ...,
                description="New value: string for state, number for trait, dict for spawn.",
            ),
        ),
        __base__=BaseModel,
    )

    return create_model(
        "DynamicInterventionOutput",
        reasoning=(str, Field(..., description="Why this interpretation.")),
        interventions=(
            List[Item],
            Field(
                ...,
                min_length=1,
                description="One or more constrained intervention items.",
            ),
        ),
        target_node_ids=(
            List[target_lit],
            Field(
                default_factory=list,
                description=(
                    "Downstream graph node IDs the user cares about. Used "
                    "by the ctf-calculus pre-flight (Rule 3) to prove an "
                    "intervention vacuous when it has no directed path to "
                    "any of these nodes. Include any entity, event, object "
                    "or world-trait the user explicitly mentions as the "
                    "thing they want to affect or observe the effect on. "
                    "Leave empty only when the user gives no downstream "
                    "reference at all."
                ),
            ),
        ),
        resolved_ids=(
            List[ResolvedID],
            Field(default_factory=list, description="Auxiliary name→ID resolutions."),
        ),
        __base__=BaseModel,
    )


def _build_counterfactual_dynamic_model(world_state: WorldStateV1):
    """Per-call model for *counterfactual*: historical_interventions
    are constrained to events; evidence_node_ids to any valid ID."""
    typed = _collect_typed_ids(world_state)
    # Historical interventions may target events (including utterance
    # events) OR communication channels. The latter unlocks queries
    # like "what if the ravens never carried Macbeth's letter" where
    # the surgery is on the standing capability, not on a single
    # discrete event.
    historical_target_ids = (
        typed["event_ids"] + typed.get("channel_ids", [])
    )
    historical_lit = _make_id_literal(historical_target_ids)
    all_ids = (
        typed["entity_ids"] + typed["object_ids"] + typed["location_ids"]
        + typed["event_ids"] + typed["world_trait_ids"]
        + typed.get("channel_ids", [])
    )
    all_lit = _make_id_literal(all_ids)

    HistoricalItem = create_model(
        "HistoricalInterventionItem",
        target_id=(
            historical_lit,
            Field(
                ...,
                description=(
                    "Event ID (EVT_*) or Channel ID (CHAN_*) to alter. "
                    "MUST be exact. Utterance events (event_type='utterance') "
                    "are addressed by their EVT_* id; channel-level "
                    "surgery (sever/establish/change intelligibility) uses "
                    "the CHAN_* id."
                ),
            ),
        ),
        property=(
            str,
            Field(
                ...,
                description=(
                    "Property to mutate. For events: 'event_type', "
                    "'description', 'outcome', 'actor_ids', 'target_ids', "
                    "and (utterances) 'truth_value', 'addressee_ids', "
                    "'via_channel_id', 'content'. For channels: 'status', "
                    "'participant_ids', 'intelligibility', 'medium'."
                ),
            ),
        ),
        value=(Any, Field(..., description="New value.")),
        __base__=BaseModel,
    )

    return create_model(
        "DynamicCounterfactualOutput",
        reasoning=(str, Field(..., description="Why this interpretation.")),
        historical_interventions=(
            List[HistoricalItem],
            Field(
                ...,
                min_length=1,
                description="Past events to alter, with constrained event IDs.",
            ),
        ),
        evidence_node_ids=(
            List[all_lit],
            Field(
                default_factory=list,
                description="Present-tense node IDs to condition on.",
            ),
        ),
        target_node_ids=(
            List[all_lit],
            Field(
                default_factory=list,
                description=(
                    "Downstream graph node IDs the user cares about. Used "
                    "by the ctf-calculus pre-flight (Rule 3) to prove a "
                    "historical intervention vacuous when no path exists "
                    "to any of these nodes in the mutilated diagram. "
                    "Include any present-tense entity, event, or world-trait "
                    "the user mentions as the thing they want changed."
                ),
            ),
        ),
        resolved_ids=(
            List[ResolvedID],
            Field(default_factory=list),
        ),
        __base__=BaseModel,
    )


def _build_directive_dynamic_model(world_state: WorldStateV1):
    """Per-call model for *directive*: target_entity_ids constrained to
    entities; target_vector_id constrained to any valid graph ID
    (sub-paths like ``ENT_X.traits.guilt`` use the optional
    ``target_vector_subpath`` field)."""
    typed = _collect_typed_ids(world_state)
    ent_lit = _make_id_literal(typed["entity_ids"])
    all_ids = (
        typed["entity_ids"] + typed["object_ids"] + typed["location_ids"]
        + typed["event_ids"] + typed["world_trait_ids"]
    )
    all_lit_optional = _make_id_literal(all_ids)

    EffectLit = Literal[
        "suspense", "surprise", "mystery", "dramatic_irony",
        "grief", "rage", "joy", "regret", "love", "fear",
    ]

    return create_model(
        "DynamicDirectiveOutput",
        reasoning=(str, Field(..., description="Why this interpretation.")),
        target_entity_ids=(
            List[ent_lit],
            Field(
                ...,
                min_length=1,
                description="Entities experiencing the effect. Must be exact ENT_ IDs.",
            ),
        ),
        target_effect=(
            EffectLit,
            Field(..., description="The narrative effect to maximise."),
        ),
        target_vector_id=(
            Optional[all_lit_optional],
            Field(
                default=None,
                description=(
                    "Optional base node ID this directive targets "
                    "(trait/edge/event). Must be an exact graph ID."
                ),
            ),
        ),
        target_vector_subpath=(
            Optional[str],
            Field(
                default=None,
                description=(
                    "Optional dotted sub-path appended to target_vector_id "
                    "(e.g. 'traits.guilt' or 'relationships.ENT_X.affinity')."
                ),
            ),
        ),
        intensity=(
            Optional[float],
            Field(default=1.0, ge=0.0, le=1.0, description="0.0–1.0 multiplier."),
        ),
        resolved_ids=(
            List[ResolvedID],
            Field(default_factory=list),
        ),
        __base__=BaseModel,
    )


def _build_interrogation_dynamic_model(world_state: WorldStateV1):
    """Per-call model for *interrogate*: free-form question plus a
    constrained ``referenced_node_ids`` enum so any IDs the LLM
    surfaces from the question are guaranteed to exist."""
    typed = _collect_typed_ids(world_state)
    all_ids = (
        typed["entity_ids"] + typed["object_ids"] + typed["location_ids"]
        + typed["event_ids"] + typed["world_trait_ids"]
    )
    all_lit = _make_id_literal(all_ids)

    return create_model(
        "DynamicInterrogationOutput",
        reasoning=(str, Field(..., description="Why this interpretation.")),
        question=(str, Field(..., description="The question to answer.")),
        require_proof=(bool, Field(default=True)),
        referenced_node_ids=(
            List[all_lit],
            Field(
                default_factory=list,
                description=(
                    "Every graph node ID the question refers to. Must be "
                    "exact — used by the pathfinder to scope the search."
                ),
            ),
        ),
        resolved_ids=(
            List[ResolvedID],
            Field(default_factory=list),
        ),
        __base__=BaseModel,
    )


_DYNAMIC_MODEL_BUILDERS = {
    "intervention": _build_intervention_dynamic_model,
    "counterfactual": _build_counterfactual_dynamic_model,
    "directive": _build_directive_dynamic_model,
    "interrogate": _build_interrogation_dynamic_model,
}


def _items_to_dotted_dict(items: list[Any]) -> Dict[str, Any]:
    """Collapse a list of {target_id, property, value} dicts/objects
    back into the dotted-key dict shape ``ParsedQuery`` expects."""
    out: Dict[str, Any] = {}
    for item in items:
        data = item if isinstance(item, dict) else item.model_dump()
        target = data.get("target_id")
        prop = (data.get("property") or "").strip()
        if not target:
            continue
        key = f"{target}.{prop}" if prop else target
        out[key] = data.get("value")
    return out


def _normalise_dynamic_to_parsed(dynamic_output: Any, query_type: str) -> ParsedQuery:
    """Convert a constrained dynamic-model instance into a ``ParsedQuery``."""
    data = dynamic_output.model_dump()
    base = dict(
        query_type=query_type,
        reasoning=data.get("reasoning", ""),
        resolved_ids=data.get("resolved_ids", []),
    )

    if query_type == "intervention":
        return ParsedQuery(
            **base,
            interventions=_items_to_dotted_dict(data.get("interventions") or []),
            target_node_ids=list(data.get("target_node_ids") or []),
        )

    if query_type == "counterfactual":
        return ParsedQuery(
            **base,
            historical_interventions=_items_to_dotted_dict(
                data.get("historical_interventions") or []
            ),
            evidence_node_ids=list(data.get("evidence_node_ids") or []),
            target_node_ids=list(data.get("target_node_ids") or []),
        )

    if query_type == "directive":
        target_vector_id = data.get("target_vector_id")
        subpath = (data.get("target_vector_subpath") or "").strip()
        if target_vector_id and subpath:
            target_vector_id = f"{target_vector_id}.{subpath}"
        return ParsedQuery(
            **base,
            target_entity_ids=list(data.get("target_entity_ids") or []),
            target_effect=data.get("target_effect"),
            target_vector_id=target_vector_id,
            intensity=data.get("intensity"),
        )

    if query_type == "interrogate":
        # ``referenced_node_ids`` isn't part of ParsedQuery; fold it
        # into ``resolved_ids`` so downstream validators still see it.
        ref_ids = list(data.get("referenced_node_ids") or [])
        existing = list(base["resolved_ids"])
        existing_ids = {r.resolved_id if isinstance(r, ResolvedID) else r.get("resolved_id") for r in existing}
        for rid in ref_ids:
            if rid not in existing_ids:
                existing.append(ResolvedID(natural_name=rid, resolved_id=rid))
        base["resolved_ids"] = existing
        return ParsedQuery(
            **base,
            question=data.get("question"),
            require_proof=data.get("require_proof", True),
        )

    # Should never happen given the dispatch table, but stay safe.
    return ParsedQuery(**base)


# =====================================================================
# Valid query type literals
# =====================================================================

QUERY_TYPES = (
    "observation", "intervention", "counterfactual",
    "directive", "interrogate", "general", "manual_edit", "evaluate",
)

QueryTypeLiteral = Literal[
    "observation", "intervention", "counterfactual",
    "directive", "interrogate", "general", "manual_edit", "evaluate",
]


# =====================================================================
# System prompts — shared preamble + per-type instructions
# =====================================================================

_PROMPT_PREAMBLE = """\
You are a query parsing agent for a narrative simulation engine called Shadow Loom.

The user's query type has already been identified as **{query_type}**. \
Your job is to extract the structured parameters needed to execute that \
query type from the user's natural-language request.

## ID RESOLUTION RULES

- Entity IDs start with ENT_ (e.g., ENT_MACBETH)
- Event IDs start with EVT_ (e.g., EVT_DUNCAN_MURDER)
- Object IDs start with OBJ_ (e.g., OBJ_DAGGER)
- Location IDs start with LOC_ (e.g., LOC_CASTLE)
- World Trait IDs start with WORLD_ (e.g., WORLD_SURVEILLANCE_STATE)

When the user mentions a character, place, object, or event by name, resolve it \
to the correct graph ID from the provided world model summary. If no world model \
is provided, use reasonable ID conventions (ENT_CHARACTERNAME).

Always populate `resolved_ids` with every entity/event/object/location you resolved.
Always provide `reasoning` explaining your interpretation of the user's intent.
Set `query_type` to "{query_type}".
Only populate the fields relevant to the {query_type} query type. Leave all other fields null.
"""

_TYPE_INSTRUCTIONS: Dict[str, str] = {
    "observation": """\
## OBSERVATION QUERY

Advances time naturally. May condition on observed facts. May lock POV to specific entities.
The user wants to see what happens, observe a scene, or get a character's POV.

**Fields to populate:**
- `observations`: Optional dict of {{node_id: observed_state}} pairs — facts to condition on.
- `focus_entity_ids`: Optional list of entity IDs to lock POV onto (omit for omniscient view).
""",
    "intervention": """\
## INTERVENTION QUERY

Forces variables to specific states (do-operator). Cuts incoming causal edges.
The user wants to forcibly change something in the present moment.

**Fields to populate:**
- `interventions`: Required dict of {{"node_id.property": new_value}} pairs.
- `target_node_ids`: Optional list of downstream graph node IDs the user explicitly
  cares about (the thing they want affected). Populate this whenever the user
  mentions a target — e.g. "make Macbeth kill Duncan" → target_node_ids=["ENT_DUNCAN"];
  "force the war to end" → target_node_ids=["WORLD_WAR"]. Used by the engine's
  ctf-calculus pre-flight to detect provably-vacuous interventions.

**KEY FORMAT — EVERY KEY MUST CONTAIN A DOT.**
The key is `<node_id>.<property_path>` (the property after the first dot is
the attribute being mutated). Bare node IDs without a `.property` suffix
are INVALID and will be rejected. Examples:

  - `"ENT_MACBETH.status": "dead"`              — set entity status
  - `"ENT_MACBETH.location_id": "LOC_HEATH"`     — move an entity
  - `"ENT_MACBETH.traits.guilt": 0.9`            — trait override (0.0–1.0)
  - `"OBJ_DAGGER.owner_id": "ENT_MACBETH"`       — transfer an object
  - `"EVT_DUNCAN_MURDER.event_type": "prevented"`— alter an event
  - `"ENT_GHOST.spawn": {{"name": "Banquo's ghost", "type": "entity"}}` — genesis

Value semantics:
  - String values = state / categorical changes (e.g. "dead", "healthy").
  - Number values = trait overrides on a `.traits.<name>` path (0.0–1.0).
  - Dict values on a `.spawn` path = genesis spawns.

At least one intervention is required.
""",
    "counterfactual": """\
## COUNTERFACTUAL QUERY

Goes back in time, changes past events, conditions on present evidence, re-simulates.
The user asks "what if" about PAST events.

**Fields to populate:**
- `historical_interventions`: Required dict of {{"event_id.property": altered_outcome}}
  — the past events to change.
- `evidence_node_ids`: List of present-tense node IDs to condition on (recommended but optional).
- `target_node_ids`: Optional list of present-tense graph node IDs the user wants
  changed by the counterfactual (the things they expect to look different in the
  re-simulated world). Used by the engine's ctf-calculus pre-flight.

**KEY FORMAT — EVERY KEY MUST CONTAIN A DOT.**
The key is `<event_id>.<property_path>`. Bare event IDs without a `.property`
suffix are INVALID and will be rejected. Common forms:

  - `"EVT_DUNCAN_MURDER.event_type": "prevented"`
  - `"EVT_DUNCAN_MURDER.outcome": "Duncan survives the night"`
  - `"EVT_GUARD_DUTY.event_type": "slept"`

If you only know that an event should be "changed" or "prevented" without a
specific attribute in mind, default to `.event_type`.

**UTTERANCE & CHANNEL counterfactuals.**
Past communications are first-class targets:

  - "What if Macbeth never told Lady M about the prophecy" →
    `"EVT_MACBETH_TELLS_LADY.truth_value": "performative"` or
    `"EVT_MACBETH_TELLS_LADY.event_type": "prevented"`.
  - "What if the message had been a lie" →
    `"EVT_LETTER_DELIVERED.truth_value": "false"`.
  - "What if the ravens couldn't carry messages" →
    `"CHAN_RAVENS.status": "severed"` or
    `"CHAN_RAVENS.intelligibility": {{"ENT_LADY_M": 0.0}}`.
  - "What if Banquo had eavesdropped" →
    `"CHAN_PROPHECY.participant_ids": ["ENT_MACBETH","ENT_BANQUO"]`.
""",
    "directive": """\
## DIRECTIVE QUERY

Optimises the next event to maximize a specific psychological or epistemic effect.
The user wants to control the FEELING or EFFECT of the next scene.

**Fields to populate:**
- `target_entity_ids`: Required list of entities experiencing the effect.
- `target_effect`: Required — one of: suspense, surprise, mystery, dramatic_irony, grief, rage, joy, regret, love, fear.
- `target_vector_id`: Optional — the specific trait/edge/event to target.
- `intensity`: Optional — 0.0 to 1.0 multiplier (default 1.0).
""",
    "interrogate": """\
## INTERROGATION QUERY

Graph pathfinding / RAG query. Does NOT advance time or generate prose.
The user wants factual answers about the graph structure.

**Fields to populate:**
- `question`: Required — the question to answer.
- `require_proof`: Optional bool — whether to require causal proof (default true).
""",
    "general": """\
## GENERAL QUERY

Full-graph Q&A without advancing time. Open-ended question about the world state.
Use for analytical questions that don't fit other types.

**Fields to populate:**
- `question`: Required — the question to answer.
- `include_topology`: Optional bool — include full topology in the response (default true).
""",
    "manual_edit": """\
## MANUAL EDIT QUERY

User-authored prose that bypasses generation. The text is re-extracted into topology.
The user is providing actual narrative prose to inject into the story.

**Fields to populate:**
- `edited_prose`: Required — the user's narrative prose text.
- `edit_description`: Optional description of the changes.
- `focus_entity_ids`: Optional list of entities most affected by the edit.
""",
    "evaluate": """\
## EVALUATION QUERY

Runs a full-story narrative quality audit. Does NOT advance time or generate new prose.
The user wants a comprehensive scorecard of the existing story (causal soundness,
affective trajectory, miracle-step detection, etc.).

**Fields to populate:**
- `focus_entity_ids`: Optional list of entities to focus the evaluation on (omit to use all).
""",
}


def _build_typed_system_prompt(query_type: str) -> str:
    """Build a focused system prompt for a known query type."""
    instructions = _TYPE_INSTRUCTIONS.get(query_type, _TYPE_INSTRUCTIONS["general"])
    return _PROMPT_PREAMBLE.format(query_type=query_type) + "\n" + instructions


# Legacy full prompt (used only when no query_type is specified)
_SYSTEM_PROMPT = """\
You are a query parsing agent for a narrative simulation engine called Shadow Loom.

Your job is to take a natural-language user request about a story world and \
classify it into exactly one of the supported query types, then extract the structured \
parameters needed to execute that query.

## SUPPORTED QUERY TYPES

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

7. **manual_edit** — "I want to write that X happens" / "Edit: Macbeth draws his dagger..."
   User-authored prose that bypasses generation. The text is re-extracted into topology.
   Use when: the user provides actual narrative prose they want to inject into the story,
   or explicitly says they want to write/edit the text themselves.
   The `edited_prose` field must contain the user's prose text.

## ID RESOLUTION RULES

- Entity IDs start with ENT_ (e.g., ENT_MACBETH)
- Event IDs start with EVT_ (e.g., EVT_DUNCAN_MURDER)
- Object IDs start with OBJ_ (e.g., OBJ_DAGGER)
- Location IDs start with LOC_ (e.g., LOC_CASTLE)
- World Trait IDs start with WORLD_ (e.g., WORLD_SURVEILLANCE_STATE)

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
            for key in parsed.interventions:
                # Keys are dotted paths like 'ENT_MACBETH.status'; validate
                # the base node ID, not the full key.
                base_id = key.split(".", 1)[0] if "." in key else key
                _check_id(base_id, "interventions")
                # Catch property/type mismatches like ``EVT_X.traits.guilt``
                # — events don't have traits, so the engine would silently
                # no-op or crash. Surface it here for a clean error.
                prop_err = _validate_property_path(key, "interventions")
                if prop_err is not None:
                    errors.append(prop_err)

    elif qt == "counterfactual":
        if not parsed.historical_interventions:
            errors.append(ValidationError(
                field="historical_interventions",
                message="Counterfactual query requires at least one historical intervention.",
            ))
        else:
            for key in parsed.historical_interventions:
                base_id = key.split(".", 1)[0] if "." in key else key
                _check_id(base_id, "historical_interventions")
                prop_err = _validate_property_path(
                    key, "historical_interventions",
                )
                if prop_err is not None:
                    errors.append(prop_err)
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

    elif qt == "manual_edit":
        if not parsed.edited_prose:
            errors.append(ValidationError(
                field="edited_prose",
                message="Manual edit query requires edited_prose.",
            ))
        if parsed.focus_entity_ids:
            _check_ids(parsed.focus_entity_ids, "focus_entity_ids")

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
    for prefix in ("ENT_", "EVT_", "OBJ_", "LOC_", "WORLD_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return re.sub(r"[^A-Z0-9]+", "_", s).strip("_")


# Stop-words that should never be treated as standalone aliases — they
# appear inside many entity/location names but matching them would
# wrongly resolve generic prose like "the heath" or "of England".
_ALIAS_STOPWORDS: set[str] = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at",
    "for", "with", "by", "from", "lady", "lord", "sir", "king",
    "queen", "thane", "duke", "earl", "his", "her", "their",
}


def _name_aliases(display_name: str) -> list[str]:
    """Return alternative phrasings of ``display_name`` for indexing.

    A character entry like ``"Macbeth (Thane of Glamis)"`` should resolve
    from any of: full string, ``"Macbeth"``, ``"Thane of Glamis"``. A
    name like ``"Lady Macbeth"`` should also resolve from the bare
    ``"Lady Macbeth"`` token even when the user writes ``"lady-macbeth"``.

    Aliases are de-duplicated, stripped of empty/short entries, and
    filtered against a stop-word set so common words like ``"the"`` /
    ``"of"`` don't end up pointing at random entities and triggering
    false-positive pre-resolutions.
    """
    out: set[str] = set()
    base = display_name.strip()
    if not base:
        return []
    out.add(base)
    # Pull "Macbeth" out of "Macbeth (Thane of Glamis)" and the parenthetical.
    paren = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", base)
    if paren:
        head, tail = paren.group(1).strip(), paren.group(2).strip()
        if head:
            out.add(head)
        if tail:
            out.add(tail)
    # First name / last name when the display is multi-word. Strip
    # surrounding punctuation from each token so a name like
    # "Macbeth (Thane of Glamis)" doesn't yield "Glamis)".
    parts = [p.strip("()[]{}.,;:!?\"'") for p in re.split(r"[\s,;:/]+", base) if p]
    if len(parts) > 1:
        out.update(p for p in parts if p)
    # Drop very short aliases (≤ 2 chars) and stop-words to keep the
    # mention scanner from latching onto common English words.
    return [
        a for a in out
        if len(a) > 2 and a.lower() not in _ALIAS_STOPWORDS
    ]


def _build_name_index(world_state: WorldStateV1) -> Dict[str, str]:
    """Build a normalised-name → graph-ID lookup from the world model.

    Also indexes entity/object/location display names so the fuzzy
    matcher can resolve "Macbeth" → "ENT_MACBETH". Honours
    :func:`_name_aliases` so parenthetical titles, first / last names,
    and slash-separated alternates all resolve to the same ID.
    """
    index: Dict[str, str] = {}

    def _add(key: str, real_id: str) -> None:
        norm = _normalise_id_name(key)
        if not norm:
            return
        # First-write-wins: if two different IDs claim the same alias,
        # keep the first registration so order-of-iteration controls
        # tiebreaks rather than silent overwrites.
        index.setdefault(norm, real_id)

    for eid, ent in world_state.entities.items():
        _add(eid, eid)
        for alias in _name_aliases(ent.name):
            _add(alias, eid)

    for lid, loc in world_state.locations.items():
        _add(lid, lid)
        for alias in _name_aliases(loc.name):
            _add(alias, lid)

    for oid, obj in world_state.objects.items():
        _add(oid, oid)
        for alias in _name_aliases(obj.name):
            _add(alias, oid)

    for evt in world_state.events:
        _add(evt.id, evt.id)
        # Index the event description as an alias so "the murder of
        # Duncan" can fuzzy-match EVT_DUNCAN_MURDER even when the
        # user never says the ID.
        if evt.description:
            _add(evt.description[:60], evt.id)

    for wid, wt in world_state.world_traits.items():
        _add(wid, wid)
        for alias in _name_aliases(wt.name):
            _add(alias, wid)

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


# =====================================================================
# Pre-resolution: scan user text for mentions before calling the LLM
# =====================================================================

# Tokens that look like ID prefixes — used to skip them during alias
# scanning so we don't double-add them.
_ID_PREFIXES = ("ENT_", "EVT_", "OBJ_", "LOC_", "WORLD_")


def _extract_mentioned_ids(
    natural_language: str,
    world_state: WorldStateV1,
    *,
    max_mentions: int = 12,
) -> List[Tuple[str, str]]:
    """Find graph IDs whose name/aliases appear (case-insensitive) in the user text.

    Returns up to ``max_mentions`` ``(mention_text, resolved_id)``
    tuples in the order they first occur in ``natural_language``. Used
    to inject a "MENTIONED IN QUERY" hints block into the LLM prompt
    so the agent doesn't have to invent IDs from prose alone.

    Strategy:
      • Build an alias → ID map from entities, locations, objects,
        events, world traits.
      • For each alias of length ≥ 3, look for a whole-word match in
        the user text (case-insensitive).
      • Deduplicate by resolved ID, keeping the longest matched alias
        (so ``"Lady Macbeth"`` wins over ``"Macbeth"`` when both fit).
      • Also catch literal ID mentions (``ENT_FOO``) the user typed
        directly.
    """
    if not natural_language or world_state is None:
        return []

    text = natural_language
    text_lower = text.lower()

    # Build (alias, resolved_id) pairs for every aliasable node.
    alias_pairs: list[tuple[str, str]] = []
    for eid, ent in world_state.entities.items():
        for alias in _name_aliases(ent.name):
            alias_pairs.append((alias, eid))
    for lid, loc in world_state.locations.items():
        for alias in _name_aliases(loc.name):
            alias_pairs.append((alias, lid))
    for oid, obj in world_state.objects.items():
        for alias in _name_aliases(obj.name):
            alias_pairs.append((alias, oid))
    for wid, wt in world_state.world_traits.items():
        for alias in _name_aliases(wt.name):
            alias_pairs.append((alias, wid))

    # Sort by alias length DESC so "Lady Macbeth" is checked before "Macbeth"
    # — guarantees the longer match wins for overlapping aliases.
    alias_pairs.sort(key=lambda kv: -len(kv[0]))

    # Track which IDs have already been claimed and which character
    # spans have been consumed (to avoid double-counting overlaps).
    claimed_ids: set[str] = set()
    consumed_spans: list[tuple[int, int]] = []
    hits: list[tuple[int, str, str]] = []  # (position, mention, id)

    def _span_overlaps(start: int, end: int) -> bool:
        for s, e in consumed_spans:
            if start < e and end > s:
                return True
        return False

    # Whole-word matching via regex word boundaries; lowercased on both sides.
    for alias, real_id in alias_pairs:
        if real_id in claimed_ids:
            continue
        if len(alias) < 3:
            continue
        pattern = r"\b" + re.escape(alias.lower()) + r"\b"
        m = re.search(pattern, text_lower)
        if m and not _span_overlaps(m.start(), m.end()):
            hits.append((m.start(), text[m.start():m.end()], real_id))
            claimed_ids.add(real_id)
            consumed_spans.append((m.start(), m.end()))

    # Also pick up literal ID mentions the user typed (``ENT_FOO``) so
    # the prompt confirms them as valid.
    valid_ids = _collect_all_ids(world_state)
    for prefix in _ID_PREFIXES:
        for m in re.finditer(rf"\b{prefix}[A-Za-z0-9_]+\b", text):
            tok = m.group(0)
            if tok in valid_ids and tok not in claimed_ids:
                hits.append((m.start(), tok, tok))
                claimed_ids.add(tok)

    hits.sort(key=lambda h: h[0])
    return [(mention, real_id) for _pos, mention, real_id in hits[:max_mentions]]


def _format_mentions_hint(
    mentions: List[Tuple[str, str]],
    world_state: WorldStateV1,
) -> str:
    """Render a compact "MENTIONED IN QUERY" block for the LLM prompt.

    Includes the resolved ID, node type, and a short descriptor so the
    LLM can confirm rather than reinvent. Returns an empty string when
    no mentions were extracted.
    """
    if not mentions:
        return ""
    lines = ["## MENTIONED IN QUERY (auto-resolved hints)"]
    for mention, real_id in mentions:
        descriptor: str
        if real_id in world_state.entities:
            ent = world_state.entities[real_id]
            descriptor = f"entity, name={ent.name!r}, status={ent.status}"
        elif real_id in world_state.locations:
            loc = world_state.locations[real_id]
            descriptor = f"location, name={loc.name!r}"
        elif real_id in world_state.objects:
            obj = world_state.objects[real_id]
            descriptor = f"object, name={obj.name!r}"
        elif real_id in world_state.world_traits:
            wt = world_state.world_traits[real_id]
            descriptor = f"world trait, name={wt.name!r}"
        else:
            # Event lookup
            evt = next((e for e in world_state.events if e.id == real_id), None)
            if evt is not None:
                descriptor = (
                    f"event, t={evt.fabula_time}, type={evt.event_type}, "
                    f"desc={evt.description[:60]!r}"
                )
            else:
                descriptor = "unknown node"
        lines.append(f"  {mention!r} → {real_id} ({descriptor})")
    lines.append(
        "Use these IDs verbatim in resolved_ids and any ID-bearing fields."
    )
    return "\n".join(lines)


# =====================================================================
# Property-path validation for intervention / counterfactual keys
# =====================================================================

# Per-node-prefix whitelist of property roots that the engine will
# accept on the LHS of a ``<id>.<property>`` intervention key.
# ``spawn`` is always permitted because genesis spawns are valid for
# every node type.
_VALID_PROPERTY_ROOTS: Dict[str, set[str]] = {
    "ENT": {
        "status", "location_id", "traits", "beliefs", "constants",
        "communicating_with", "spawn",
    },
    "OBJ": {
        "owner_id", "location_id", "properties", "affordances", "spawn",
    },
    "LOC": {"ambient_state", "description", "spawn"},
    "EVT": {
        "event_type", "description", "actor_ids", "target_ids",
        "outcome", "via_channel_id", "speaker_id", "addressee_ids",
        "truth_value", "content", "spawn",
    },
    "WORLD": {"magnitude", "description", "affected_domains", "spawn"},
    "CHAN": {
        "medium", "participant_ids", "intelligibility", "directionality",
        "status", "spawn",
    },
}


def _validate_property_path(
    key: str,
    field_name: str,
) -> Optional[ValidationError]:
    """Return a ValidationError if ``key`` targets an invalid property.

    Accepts dotted keys of the form ``<ID>.<root>[.<sub>...]``. Only
    the *root* property is checked — sub-paths (e.g. ``traits.guilt``)
    are passed through to the engine, which knows how to interpret
    them. Bare keys without a dot are caught earlier by
    :func:`_normalise_intervention_keys`; this validator focuses on
    "wrong property for this node type" mistakes.
    """
    if "." not in key:
        return None  # Handled by other validation.
    base_id, rest = key.split(".", 1)
    if "_" not in base_id:
        return None  # Unknown prefix — let _check_id handle it.
    prefix = base_id.split("_", 1)[0]
    allowed = _VALID_PROPERTY_ROOTS.get(prefix)
    if allowed is None:
        return None  # Unrecognised prefix; out of our jurisdiction.
    root = rest.split(".", 1)[0]
    if root not in allowed:
        return ValidationError(
            field=field_name,
            message=(
                f"Property {root!r} is not valid for {prefix}_ nodes. "
                f"Allowed: {sorted(allowed)}. "
                f"(Full key was {key!r}.)"
            ),
        )
    return None


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
    query = _build_query(fallback_parsed, natural_language=natural_language)
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
                query = _build_query(patched, natural_language=natural_language)
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

def _normalise_intervention_keys(
    interventions: Dict[str, Any],
    *,
    field_name: str,
) -> Dict[str, Any]:
    """Repair bare-ID keys in an intervention dict to the dotted form.

    The narrative-physics engine requires keys of the form
    ``<node_id>.<property_path>``. LLM outputs sometimes drop the
    property suffix and emit a bare ``EVT_X`` / ``ENT_X`` / etc. This
    helper infers a sensible default property based on the ID prefix
    and the value type, so the engine has *something* to operate on
    rather than rejecting the whole request as malformed.

    Heuristics (only applied when the key contains no ``.``):

      - ``EVT_*`` \u2192 ``.event_type`` (the standard "alter this event" axis)
      - ``ENT_*`` with str value \u2192 ``.status``
      - ``ENT_*`` with dict value \u2192 ``.spawn``
      - ``OBJ_*`` with dict value \u2192 ``.spawn``
      - ``LOC_*`` with dict value \u2192 ``.spawn``
      - ``WORLD_*`` with number value \u2192 ``.magnitude``

    Keys that don't match any heuristic are left unchanged so that
    downstream plausibility checks can report them clearly.
    """
    if not interventions:
        return interventions

    repaired: Dict[str, Any] = {}
    for key, value in interventions.items():
        if "." in key:
            repaired[key] = value
            continue

        prefix = key.split("_", 1)[0] if "_" in key else ""
        suffix: Optional[str] = None

        # Field-aware: in counterfactuals, a bare EVT_* key with a dict
        # value is ambiguous — it could mean "alter this event" or
        # "spawn a new past event". Forcibly normalising to .event_type
        # would silently corrupt event_type with a dict payload, so we
        # leave it bare and let the downstream layer reject it cleanly.
        if prefix == "EVT" and isinstance(value, dict):
            if field_name == "historical_interventions":
                # Leave malformed; downstream surfaces a clear error.
                repaired[key] = value
                continue
            suffix = "spawn"
        elif isinstance(value, dict):
            suffix = "spawn"
        elif prefix == "EVT":
            suffix = "event_type"
        elif prefix == "ENT" and isinstance(value, str):
            suffix = "status"
        elif prefix == "WORLD" and isinstance(value, (int, float)):
            suffix = "magnitude"

        if suffix is None:
            # Leave malformed; downstream layer will surface a clear error.
            repaired[key] = value
            continue

        new_key = f"{key}.{suffix}"
        logger.info(
            "[QueryParser] Normalised %s key %r \u2192 %r (inferred property).",
            field_name, key, new_key,
        )
        repaired[new_key] = value

    return repaired


def _build_query(
    parsed: ParsedQuery,
    natural_language: Optional[str] = None,
) -> UserRequest:
    """Construct a concrete query model from the parsed classification.

    ``natural_language`` is the user's verbatim request; it is stored
    on the resulting query as ``original_query`` so downstream stages
    (directive assembler, generator, auditor, persistence) can see
    exactly what the user asked for, not just the LLM's classification.
    """
    qt = parsed.query_type
    nl = natural_language

    if qt == "observation":
        return ObservationQuery(
            observations=parsed.observations or {},
            focus_entity_ids=parsed.focus_entity_ids or [],
            original_query=nl,
        )

    if qt == "intervention":
        return InterventionQuery(
            interventions=_normalise_intervention_keys(
                parsed.interventions or {}, field_name="interventions",
            ),
            target_node_ids=parsed.target_node_ids or [],
            original_query=nl,
        )

    if qt == "counterfactual":
        return CounterfactualQuery(
            historical_interventions=_normalise_intervention_keys(
                parsed.historical_interventions or {},
                field_name="historical_interventions",
            ),
            evidence_node_ids=parsed.evidence_node_ids or [],
            target_node_ids=parsed.target_node_ids or [],
            original_query=nl,
        )

    if qt == "directive":
        return DirectiveQuery(
            target_entity_ids=parsed.target_entity_ids or [],
            target_effect=parsed.target_effect or "suspense",
            target_vector_id=parsed.target_vector_id,
            intensity=parsed.intensity if parsed.intensity is not None else 1.0,
            original_query=nl,
        )

    if qt == "interrogate":
        return InterrogationQuery(
            question=parsed.question or nl or "",
            require_proof=parsed.require_proof if parsed.require_proof is not None else True,
            original_query=nl,
        )

    if qt == "manual_edit":
        return ManualEditQuery(
            edited_prose=parsed.edited_prose or nl or "",
            description=parsed.edit_description or "",
            focus_entity_ids=parsed.focus_entity_ids or [],
            original_query=nl,
        )

    if qt == "evaluate":
        return EvaluationQuery(
            focus_entity_ids=parsed.focus_entity_ids or [],
            original_query=nl,
        )

    # general
    return GeneralQuery(
        question=parsed.question or nl or "",
        include_topology=parsed.include_topology if parsed.include_topology is not None else True,
        original_query=nl,
    )


# =====================================================================
# Main entry point
# =====================================================================

def _resolve_system_prompt(query_type: Optional[str]) -> str:
    if query_type is not None:
        if query_type not in QUERY_TYPES:
            raise ValueError(
                f"Invalid query_type {query_type!r}. "
                f"Must be one of {QUERY_TYPES}."
            )
        return _build_typed_system_prompt(query_type)
    return _SYSTEM_PROMPT


def _build_user_message(
    natural_language: str,
    world_state: Optional[WorldStateV1],
    *,
    constrained: bool,
) -> str:
    """Compose the user-message body (graph dump + mention hints + request).

    When ``constrained=True`` we additionally emit the categorised
    "VALID GRAPH IDS" block so the LLM sees the exact enum its
    structured output will be validated against.
    """
    user_parts: list[str] = []
    if world_state:
        user_parts.append("## WORLD MODEL\n")
        user_parts.append(_build_graph_summary(world_state))
        user_parts.append("\n")
        if constrained:
            user_parts.append(_format_valid_ids_section(world_state))
            user_parts.append("\n")
        mentions = _extract_mentioned_ids(natural_language, world_state)
        hint = _format_mentions_hint(mentions, world_state)
        if hint:
            user_parts.append(hint)
            user_parts.append("\n")
    user_parts.append("## USER REQUEST\n")
    user_parts.append(natural_language)
    return "\n".join(user_parts)


def _select_output_model(
    query_type: Optional[str],
    world_state: Optional[WorldStateV1],
) -> Tuple[type[BaseModel], bool]:
    """Pick the structured-output schema.

    Returns ``(model_cls, is_constrained)``. Constrained dynamic
    schemas are only used when *both* the query type belongs to the
    constrained set *and* a non-empty world state is available — the
    ``Literal`` enums need at least some IDs to enumerate over.
    """
    if (
        query_type in _CONSTRAINED_QUERY_TYPES
        and world_state is not None
        and _collect_all_ids(world_state)
    ):
        builder = _DYNAMIC_MODEL_BUILDERS[query_type]
        return builder(world_state), True
    return ParsedQuery, False


def _attach_output_validator(
    agent: Agent,
    *,
    query_type: Optional[str],
    constrained: bool,
    world_state: Optional[WorldStateV1],
) -> None:
    """Wire an ``output_validator`` onto the agent that normalises the
    LLM output into a ``ParsedQuery`` and re-runs structural validation
    against the world model.

    On a hard validation error (unknown ID, missing required field,
    invalid property path) we raise :class:`ModelRetry` with a
    structured failure message. PydanticAI surfaces that message back
    to the LLM as a tool retry, so the agent uses its
    ``output_retries`` budget to actually fix the bad output rather
    than letting the caller see it. This is the only point where the
    PromptedOutput path can recover from hallucinated IDs \u2014 the schema
    only lives in the prompt, so the model provider never enforces it.
    """
    if world_state is None:
        # Without a world model we have nothing to validate IDs against.
        return

    @agent.output_validator
    def _validate(_ctx, raw_output: Any) -> Any:
        try:
            parsed = _interpret_agent_output(
                raw_output, query_type=query_type, constrained=constrained,
            )
        except Exception as exc:
            raise ModelRetry(
                f"Could not interpret structured output: {exc!r}. "
                "Re-emit a JSON object that strictly matches the schema."
            ) from exc

        errors = _validate_parsed_query(parsed, world_state)
        hard = [e for e in errors if e.severity == "error"]
        if hard:
            # The fuzzy-repair / general-fallback layer can still
            # rescue this on the caller side, but giving the LLM one
            # chance to self-correct produces dramatically better
            # results than always falling back. Keep the message
            # short so it fits comfortably in the next prompt.
            details = "; ".join(f"{e.field}: {e.message}" for e in hard[:5])
            raise ModelRetry(
                "Structured output failed validation against the world "
                f"model: {details}. Use ONLY IDs that appear in the "
                "graph summary above and use the dotted "
                "<ID>.<property> form for intervention keys."
            )
        return raw_output


def _interpret_agent_output(
    raw_output: Any,
    *,
    query_type: Optional[str],
    constrained: bool,
) -> ParsedQuery:
    """Normalise the agent's structured output into a ``ParsedQuery``.

    Handles three cases:
      \u2022 Constrained dynamic model \u2192 fold into ParsedQuery via
        :func:`_normalise_dynamic_to_parsed` (also forces query_type).
      \u2022 Plain ParsedQuery from a typed prompt \u2192 optionally override
        query_type if the LLM misclassified despite the system prompt.
      \u2022 Plain ParsedQuery from the legacy auto-classify path \u2192 return
        as-is.
    """
    if constrained and query_type is not None and not isinstance(raw_output, ParsedQuery):
        return _normalise_dynamic_to_parsed(raw_output, query_type)

    parsed: ParsedQuery = raw_output
    if query_type is not None and parsed.query_type != query_type:
        logger.info(
            "[QueryParser] Overriding LLM query_type %r → %r",
            parsed.query_type, query_type,
        )
        parsed = parsed.model_copy(update={"query_type": query_type})
    return parsed


def _finalise_parse(
    natural_language: str,
    parsed: ParsedQuery,
    world_state: Optional[WorldStateV1],
) -> QueryParseResult:
    """Run validation + fallback + concrete query construction."""
    errors = _validate_parsed_query(parsed, world_state)
    has_hard_errors = any(e.severity == "error" for e in errors)

    if has_hard_errors:
        logger.warning(
            "[QueryParser] Validation failed with %d error(s) — attempting fallback: %s",
            len(errors),
            "; ".join(e.message for e in errors),
        )
        return _apply_fallback(natural_language, parsed, errors, world_state)

    query = _build_query(parsed, natural_language=natural_language)
    logger.info("[QueryParser] Resolved to %s query", parsed.query_type)
    return QueryParseResult(
        query=query,
        parsed=parsed,
        validation_errors=errors,  # may contain warnings
        is_valid=True,
    )


def parse_query(
    natural_language: str,
    *,
    query_type: Optional[str] = None,
    world_state: Optional[WorldStateV1] = None,
    config: Optional[QueryParsingConfig] = None,
) -> QueryParseResult:
    """Parse a natural-language request into a structured pipeline query.

    Parameters
    ----------
    natural_language : str
        The user's free-form request.
    query_type : str or None
        When provided, the query type is fixed (e.g. ``"observation"``,
        ``"intervention"``) and the LLM only needs to resolve IDs and
        extract type-specific fields. Must be one of
        :data:`QUERY_TYPES`. When ``None``, the LLM also classifies
        the query type (legacy behaviour).

        For the four "graph-binding" query types (``intervention``,
        ``counterfactual``, ``directive``, ``interrogate``), passing
        a ``world_state`` switches the structured output schema to a
        per-call dynamic Pydantic model whose ID-bearing fields are
        constrained to ``Literal[<valid IDs>]``. This forces the LLM
        to emit only IDs that exist in the graph.
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

    system_prompt = _resolve_system_prompt(query_type)
    output_model, constrained = _select_output_model(query_type, world_state)
    user_message = _build_user_message(
        natural_language, world_state, constrained=constrained,
    )

    # ``PromptedOutput`` keeps the dynamic ``Literal`` / ``anyOf``
    # schemas out of the model provider's native ``format`` field.
    # Recent Ollama builds reject ``anyOf`` (used for Optional fields
    # and the ``target_vector_id`` enum-or-null shape) with
    # ``invalid JSON schema in format``. PromptedOutput injects the
    # schema into the prompt and validates the JSON in Python, so
    # arbitrary Pydantic schemas work on any backend.
    agent: Agent[None, Any] = Agent(
        model,
        system_prompt=system_prompt,
        output_type=PromptedOutput(output_model),
        retries=cfg.output_retries,
    )
    _attach_output_validator(
        agent,
        query_type=query_type,
        constrained=constrained,
        world_state=world_state,
    )

    logger.info(
        "[QueryParser] Classifying (constrained=%s, type=%s): %s",
        constrained, query_type, natural_language[:120],
    )

    result = agent.run_sync(
        user_message,
        model_settings={
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        },
    )
    log_agent_output(logger, "QueryParser", result.output)
    parsed = _interpret_agent_output(
        result.output, query_type=query_type, constrained=constrained,
    )
    return _finalise_parse(natural_language, parsed, world_state)


async def parse_query_async(
    natural_language: str,
    *,
    query_type: Optional[str] = None,
    world_state: Optional[WorldStateV1] = None,
    config: Optional[QueryParsingConfig] = None,
) -> QueryParseResult:
    """Async version of :func:`parse_query`."""
    cfg = config or QueryParsingConfig()
    model = _resolve_model(cfg.model)

    system_prompt = _resolve_system_prompt(query_type)
    output_model, constrained = _select_output_model(query_type, world_state)
    user_message = _build_user_message(
        natural_language, world_state, constrained=constrained,
    )

    # See sync ``parse_query`` for why PromptedOutput is used here.
    agent: Agent[None, Any] = Agent(
        model,
        system_prompt=system_prompt,
        output_type=PromptedOutput(output_model),
        retries=cfg.output_retries,
    )
    _attach_output_validator(
        agent,
        query_type=query_type,
        constrained=constrained,
        world_state=world_state,
    )

    logger.info(
        "[QueryParser] Classifying async (constrained=%s, type=%s): %s",
        constrained, query_type, natural_language[:120],
    )

    result = await agent.run(
        user_message,
        model_settings={
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        },
    )
    log_agent_output(logger, "QueryParser", result.output)
    parsed = _interpret_agent_output(
        result.output, query_type=query_type, constrained=constrained,
    )
    return _finalise_parse(natural_language, parsed, world_state)
