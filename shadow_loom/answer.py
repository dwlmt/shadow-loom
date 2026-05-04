# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LLM-backed Q&A for ``general`` and ``interrogate`` queries.

Both query types return a graph slice from
:func:`shadow_loom.narrative_physics.calculate_narrative_physics` but
*do not* generate prose. Without an LLM step they bottom out at the
chat card's "General completed." placeholder, which gives the user no
real answer.

This module compresses the world state into a token-budget-friendly
context block and asks the configured LLM to answer the user's
question, returning a structured ``AnswerCard`` that the chat /
Answer panel can render directly (claim · evidence · confidence ·
caveats).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput

from shadow_loom.generation import GenerationConfig
from shadow_loom.models import WorldStateV1
from shadow_loom.settings import resolve_model as _resolve_model

logger = logging.getLogger(__name__)


# =====================================================================
# Output model
# =====================================================================


class AnswerCard(BaseModel):
    """Structured Q&A answer surfaced in the chat / Answer panel."""

    answer: str = Field(
        description=(
            "Direct, plain-language answer to the user's question. "
            "Reference characters, events, and locations by their human "
            "names rather than by ID."
        ),
    )
    evidence_node_ids: List[str] = Field(
        default_factory=list,
        description=(
            "World-state node ids (ENT_*, EVT_*, OBJ_*, LOC_*) that "
            "support the answer. Pull only ids that actually appear in "
            "the supplied context."
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How well the world state supports the answer. 1.0 means "
            "directly stated; 0.5 means inferred; <0.3 means speculative."
        ),
    )
    caveats: List[str] = Field(
        default_factory=list,
        description=(
            "Any reasons the answer is partial, uncertain, or "
            "unanswerable from the available world state."
        ),
    )


# =====================================================================
# World-state compression
# =====================================================================


def _compress_world_state(
    physics_state: Dict[str, Any] | None,
    *,
    max_entities: int = 60,
    max_events: int = 80,
    max_locations: int = 30,
    max_channels: int = 20,
) -> str:
    """Render a compact, LLM-friendly summary of the omniscient graph.

    ``physics_state`` is the dict returned by
    :func:`shadow_loom.extract_graph.extract_full_world_state` — a
    serialised :class:`WorldStateV1`, optionally time-sliced.
    """
    if not physics_state:
        return "(no world state available)"

    lines: List[str] = []

    entities = physics_state.get("entities", {}) or {}
    if entities:
        lines.append("## Entities")
        for ent_id, ent in list(entities.items())[:max_entities]:
            name = ent.get("name", ent_id)
            status = ent.get("status", "unknown")
            loc = ent.get("location_id", "?")
            consts = ent.get("constants") or []
            const_str = f" [{', '.join(consts)}]" if consts else ""
            lines.append(
                f"- `{ent_id}` {name} — status={status}, "
                f"loc={loc}{const_str}"
            )
        if len(entities) > max_entities:
            lines.append(f"  …(+{len(entities) - max_entities} more entities)")

    locations = physics_state.get("locations", {}) or {}
    if locations:
        lines.append("\n## Locations")
        for loc_id, loc in list(locations.items())[:max_locations]:
            name = loc.get("name", loc_id)
            lines.append(f"- `{loc_id}` {name}")
        if len(locations) > max_locations:
            lines.append(
                f"  …(+{len(locations) - max_locations} more locations)"
            )

    objects = physics_state.get("objects", {}) or {}
    if objects:
        lines.append("\n## Objects")
        for obj_id, obj in list(objects.items())[:max_entities]:
            name = obj.get("name", obj_id)
            owner = obj.get("owner_id") or "—"
            lines.append(f"- `{obj_id}` {name} (owner={owner})")

    events = physics_state.get("events", []) or []
    if events:
        # Sort by fabula_time so the LLM gets chronological order.
        sorted_events = sorted(
            events, key=lambda e: e.get("fabula_time", 0),
        )
        recent = sorted_events[-max_events:]
        lines.append("\n## Events (chronological)")
        for evt in recent:
            eid = evt.get("id", "?")
            ft = evt.get("fabula_time", "?")
            etype = evt.get("event_type", "?")
            actors = ",".join(evt.get("actor_ids") or []) or "—"
            targets = ",".join(evt.get("target_ids") or []) or "—"
            desc = (evt.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 160:
                desc = desc[:157] + "…"
            content = evt.get("content")
            content_str = ""
            if content:
                c = str(content).replace("\n", " ")
                if len(c) > 120:
                    c = c[:117] + "…"
                content_str = f" content=\"{c}\""
            lines.append(
                f"- T={ft} `{eid}` ({etype}) actors=[{actors}] "
                f"targets=[{targets}]{content_str} — {desc}"
            )
        if len(events) > max_events:
            lines.append(
                f"  …(+{len(events) - max_events} earlier events omitted)"
            )

    causal = physics_state.get("causal_topology", []) or []
    if causal:
        lines.append("\n## Causal edges (cause → effect)")
        for ce in causal[: max_events]:
            cause = ce.get("source_id") or "?"
            effect = ce.get("target_id") or "?"
            kind = ce.get("causality_type") or "causes"
            trait = ce.get("trait_target")
            delta = ce.get("trait_delta")
            tail = ""
            if trait:
                if delta is not None:
                    tail = f" [{trait}{float(delta):+.2f}]"
                else:
                    tail = f" [{trait}]"
            lines.append(f"- {cause} —[{kind}]→ {effect}{tail}")

    social = physics_state.get("social_topology", []) or []
    if social:
        lines.append("\n## Social relationships")
        for rel in social[: max_entities]:
            a = rel.get("source_entity_id") or "?"
            b = rel.get("target_entity_id") or "?"
            metrics = rel.get("metrics") or {}
            metric_parts = []
            for axis, m in metrics.items():
                if isinstance(m, dict) and "value" in m:
                    metric_parts.append(f"{axis}={float(m['value']):+.2f}")
            metric_str = (
                f" ({', '.join(metric_parts)})" if metric_parts else ""
            )
            lines.append(f"- {a} → {b}{metric_str}")

    spatial = physics_state.get("spatial_topology", []) or []
    if spatial:
        lines.append("\n## Spatial connections")
        for se in spatial[: max_locations]:
            a = se.get("source_id") or "?"
            b = se.get("target_id") or "?"
            locked = " [LOCKED]" if se.get("is_locked") else ""
            lines.append(f"- {a} ↔ {b}{locked}")

    channels = physics_state.get("channels", {}) or {}
    if channels:
        lines.append("\n## Information channels")
        for cid, ch in list(channels.items())[:max_channels]:
            medium = ch.get("medium", "?")
            parts = ",".join(ch.get("participant_ids") or []) or "—"
            lines.append(f"- `{cid}` {medium} participants=[{parts}]")

    return "\n".join(lines)


# =====================================================================
# Agent
# =====================================================================


_SYSTEM_PROMPT_GENERAL = """\
You are the Shadow Loom narrative analyst. You answer questions about
a structured story world that has been extracted into an Aspectual
Modal World Network (AMWN) graph: entities (characters/groups),
locations, objects, events (in fabula time), causal edges between
events, social relationships, spatial connections, and information
channels.

You will be given:
  1. The user's question.
  2. The omniscient world-state slice (entities, events, edges, etc.)
     known at the current temporal anchor.

Rules:
  • Answer ONLY from the supplied world state. Do not invent characters,
    events, or relationships that are not present.
  • If the answer is not deducible, say so plainly and lower confidence.
  • Reference characters, events, and locations by their human names
    in the prose answer; list the exact node ids in evidence_node_ids.
  • Length should match the question. Use a single sentence for a
    factual lookup; use several short paragraphs (with line breaks)
    for synthesis questions that span multiple entities, events, or
    causal chains. Markdown bullet points are welcome when the
    answer is genuinely a list. Do not pad — add a sentence only
    when it carries new information.
  • Caveats should call out missing information, ambiguity, or
    inferences that go beyond what is stated.
"""


_SYSTEM_PROMPT_INTERROGATE = """\
You are the Shadow Loom causal interrogator. The user is asking a
diagnostic question (e.g. "why does X happen?", "what does C know?",
"is there a causal path from A to B?").

You will be given:
  1. The user's question.
  2. The omniscient world-state slice (entities, events, causal edges,
     beliefs, channels) known at the current temporal anchor.
  3. Whether causal proof is required.

Rules:
  • When asked "why" or "what caused", trace through the causal_topology
    edges and recent events to construct a causal chain. Walk the
    chain explicitly — "A → B → C because …" — rather than
    collapsing the answer to a single conclusion.
  • When asked "what does X know", consult X's beliefs and the
    information channels they participate in. Distinguish between
    direct knowledge, inference, and what X is unaware of.
  • When asked about relationships or spatial reachability, walk the
    social_topology / spatial_topology edges.
  • If require_proof is true, only assert claims you can back with at
    least one explicit edge or event in the supplied data.
  • Return the supporting node ids in evidence_node_ids so the UI can
    highlight them.
  • Length is set by the question. Diagnostic answers can be
    multi-paragraph when the causal chain is long; structure them
    with markdown line breaks or bullet points so the user can
    follow each step. Avoid both one-line shrugs and unnecessary
    padding.
  • If the world state does not support an answer, say so plainly,
    lower confidence to <=0.3, and add a caveat naming the missing
    information.
"""


def _build_answer_agent(
    config: GenerationConfig,
    *,
    query_type: str,
) -> Agent[None, AnswerCard]:
    """Construct the Q&A agent for ``general`` / ``interrogate``."""
    system_prompt = (
        _SYSTEM_PROMPT_INTERROGATE
        if query_type == "interrogate"
        else _SYSTEM_PROMPT_GENERAL
    )
    agent: Agent[None, AnswerCard] = Agent(
        _resolve_model(config.model),
        output_type=NativeOutput(AnswerCard),
        system_prompt=system_prompt,
        retries=config.output_retries,
    )
    return agent


# =====================================================================
# Public API
# =====================================================================


def answer_question(
    question: str,
    physics_state: Dict[str, Any] | None,
    *,
    query_type: str = "general",
    require_proof: bool = False,
    world_state: Optional[WorldStateV1] = None,  # noqa: ARG001 — reserved
    config: Optional[GenerationConfig] = None,
) -> AnswerCard:
    """Answer a Q&A question using the supplied world-state slice.

    Returns an :class:`AnswerCard` even on failure (with a low
    confidence and a caveat) so callers never have to deal with
    ``None``.
    """
    config = config or GenerationConfig()
    if not (question or "").strip():
        return AnswerCard(
            answer="(no question supplied)",
            confidence=0.0,
            caveats=["Empty question."],
        )

    context_block = _compress_world_state(physics_state)
    user_msg_parts: List[str] = [
        f"Question: {question.strip()}",
        f"Query type: {query_type}",
    ]
    if query_type == "interrogate":
        user_msg_parts.append(
            f"Require causal proof: {'yes' if require_proof else 'no'}"
        )
    user_msg_parts.extend([
        "",
        "World state at current temporal anchor:",
        context_block,
    ])
    user_msg = "\n".join(user_msg_parts)

    agent = _build_answer_agent(config, query_type=query_type)
    try:
        result = agent.run_sync(user_msg)
        card = result.output
        logger.info(
            "[Answer] q_type=%s conf=%.2f answer_len=%d evidence=%d",
            query_type,
            card.confidence,
            len(card.answer or ""),
            len(card.evidence_node_ids or []),
        )
        return card
    except Exception as exc:  # noqa: BLE001
        logger.exception("[Answer] LLM call failed")
        return AnswerCard(
            answer=(
                "I was unable to answer that question because the "
                "language model call failed. The world state may be "
                "too large to fit in the model's context, or the "
                "model is unavailable."
            ),
            confidence=0.0,
            caveats=[f"LLM error: {exc}"],
        )
