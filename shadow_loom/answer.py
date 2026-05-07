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
from typing import Any, Dict, List, Literal, Optional

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
    branch_world_id: Literal["factual", "shadow"] = "factual",
) -> str:
    """Render a compact, LLM-friendly summary of the omniscient graph.

    ``physics_state`` is the dict returned by
    :func:`shadow_loom.extract_graph.extract_full_world_state` — a
    serialised :class:`WorldStateV1`, optionally time-sliced.

    When ``branch_world_id`` is ``shadow`` the compressor surfaces
    each event/edge/channel/snapshot's own ``world_id`` tag inline so
    the answering LLM can disambiguate factual ancestors from active
    shadow content. For factual queries the tag is omitted (every
    surviving entry is factual by construction).
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
            # Traits (≥0.15 from neutral) — without these the LLM
            # has nothing to consult when asked "is X brave / angry /
            # naive". WorldStateV1 nests traits as ``{name: {value,
            # inertia, evidence_strength}}``; flat scalars are
            # tolerated as a legacy shape.
            traits = ent.get("traits") or {}
            shown_traits: List[str] = []
            for tname, tv in traits.items():
                if isinstance(tv, dict):
                    val = tv.get("value")
                else:
                    val = tv
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(fval - 0.5) >= 0.15 or abs(fval) >= 0.15:
                    shown_traits.append(f"{tname}={fval:+.2f}")
            if shown_traits:
                lines.append(f"    traits: {', '.join(shown_traits[:12])}")
            # Beliefs — what this entity thinks is true about other
            # nodes. Critical for "what does X know" / "what does X
            # think Y did" questions; the system prompt already
            # promises this is available.
            beliefs = ent.get("beliefs") or []
            if beliefs:
                shown_beliefs: List[str] = []
                for b in beliefs[:6]:
                    if not isinstance(b, dict):
                        continue
                    tgt = b.get("target_id", "?")
                    state = (b.get("perceived_state") or "").strip()
                    if len(state) > 80:
                        state = state[:77] + "…"
                    conf = b.get("confidence")
                    conf_str = (
                        f" (conf={float(conf):.2f})"
                        if isinstance(conf, (int, float)) else ""
                    )
                    shown_beliefs.append(f"re {tgt}: '{state}'{conf_str}")
                if shown_beliefs:
                    lines.append(
                        "    beliefs: " + " | ".join(shown_beliefs)
                    )
                if len(beliefs) > 6:
                    lines.append(f"    …(+{len(beliefs) - 6} more beliefs)")
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
            wid_tag = ""
            if branch_world_id == "shadow":
                ewid = evt.get("world_id", "factual")
                wid_tag = f" [{ewid}]"
            lines.append(
                f"- T={ft} `{eid}` ({etype}){wid_tag} actors=[{actors}] "
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
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{ce.get('world_id', 'factual')}]"
            lines.append(f"- {cause} —[{kind}]→ {effect}{wid_tag}{tail}")

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
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{rel.get('world_id', 'factual')}]"
            lines.append(f"- {a} → {b}{wid_tag}{metric_str}")

    spatial = physics_state.get("spatial_topology", []) or []
    if spatial:
        lines.append("\n## Spatial connections")
        for se in spatial[: max_locations]:
            a = se.get("source_id") or "?"
            b = se.get("target_id") or "?"
            locked = " [LOCKED]" if se.get("is_locked") else ""
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{se.get('world_id', 'factual')}]"
            lines.append(f"- {a} ↔ {b}{wid_tag}{locked}")

    # Surface utterance content + truth_value separately. Without
    # this an interrogator asking "did X tell Y the truth about Z"
    # gets no signal — the events list above only carries
    # ``description``, while the actual quoted ``content`` and the
    # ``truth_value`` (true / false / performative) live on the
    # utterance event itself.
    utterances = [
        e for e in (physics_state.get("events", []) or [])
        if e.get("event_type") == "utterance"
    ]
    if utterances:
        utterances.sort(key=lambda e: e.get("fabula_time", 0))
        lines.append("\n## Utterances (dialogue, with truth_value)")
        for u in utterances[-max_events:]:
            uid = u.get("id", "?")
            ft = u.get("fabula_time", "?")
            speaker = u.get("speaker_id") or (u.get("actor_ids") or ["?"])[0]
            addressees = ",".join(u.get("addressee_ids") or []) or "—"
            tv = u.get("truth_value")
            tv_str = f" truth={tv}" if tv else ""
            content = (u.get("content") or "").strip().replace("\n", " ")
            if len(content) > 200:
                content = content[:197] + "…"
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{u.get('world_id', 'factual')}]"
            lines.append(
                f"- T={ft} `{uid}`{wid_tag} {speaker} → [{addressees}]"
                f"{tv_str}: {content}"
            )

    world_traits = physics_state.get("world_traits", {}) or {}
    if world_traits:
        lines.append("\n## World traits (global forces)")
        for wid, wt in list(world_traits.items())[:max_channels]:
            name = wt.get("name", wid)
            mag = wt.get("magnitude") or {}
            mval = mag.get("value") if isinstance(mag, dict) else mag
            mag_str = (
                f" mag={float(mval):.2f}"
                if isinstance(mval, (int, float)) else ""
            )
            domains = ", ".join(wt.get("affected_domains") or [])
            domains_str = f" domains=[{domains}]" if domains else ""
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{wt.get('world_id', 'factual')}]"
            lines.append(f"- `{wid}` {name}{wid_tag}{mag_str}{domains_str}")

    channels = physics_state.get("channels", {}) or {}
    if channels:
        lines.append("\n## Information channels")
        for cid, ch in list(channels.items())[:max_channels]:
            medium = ch.get("medium", "?")
            parts = ",".join(ch.get("participant_ids") or []) or "—"
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{ch.get('world_id', 'factual')}]"
            lines.append(f"- `{cid}` {medium}{wid_tag} participants=[{parts}]")

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
  2. The active AMWN branch (factual mainline vs a shadow fork) the
     question is being asked about. When the active branch is
     `shadow`, you may also be given a FACTUAL MAINLINE block as
     background contrast and a STORY SO FAR block of prose written
     on the active branch — both are background context, not
     authoritative. The world-state slice IS authoritative.
  3. The omniscient world-state slice (entities, events, edges, etc.)
     known at the current temporal anchor on the active branch.

Rules:
  • Answer ONLY from the supplied world state. Do not invent characters,
    events, or relationships that are not present.
  • The active branch is the source of truth for the answer. When on a
    shadow fork, do NOT default back to canonical / factual outcomes
    that the shadow has overwritten — answer from the shadow state.
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
  2. The active AMWN branch (factual mainline vs a shadow fork) the
     diagnostic is being run against. Causal proofs are valid only
     within the active branch — do NOT walk through edges tagged for
     a different branch when reaching for evidence.
  3. The omniscient world-state slice (entities, events, causal edges,
     beliefs, channels) known at the current temporal anchor on the
     active branch.
  4. Whether causal proof is required.

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
    least one explicit edge or event in the supplied data, and only
    use edges that belong to the active branch.
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


_SYSTEM_PROMPT_INTERVENTION = """\
You are the Shadow Loom Pearl-Rung-2 (do-operator) analyst. The user
asked a Rung-2 question — they want to know what the world looks like
*under a forced surgery* on its present state. The world-state slice
you are given is the post-do sandbox produced by the causal physics
engine (NOT the factual mainline).

You will be given:
  1. The user's question.
  2. The active AMWN branch and the omniscient post-do world-state
     slice. Treat this slice as the ground truth for everything the
     question asks "now".
  3. Phase-7 RUNG-2 SURGERY METADATA when the parser produced typed
     ``do_targets``. The metadata names the *kind* of surgery and the
     concrete payload — DoEvent / DoProposition / DoBelief / DoConcern
     / DoTrait — and lists the propositions, beliefs, and concerns
     whose values shifted relative to the factual world.

Rules:
  • Answer ONLY from the supplied (post-do) world state. The factual
    mainline is background contrast; do NOT default to it.
  • Match the surgery's epistemic / ontic register:
      - DoProposition  → "Under the clamp that PROP X is true, …" (ontic).
      - DoBelief       → "From holder H's clamped belief …" (epistemic;
        the world may be unchanged but H's beliefs were forced).
      - DoConcern      → "With holder H's concern C clamped to
        salience S, …" (motivational; reweighs disposition, not facts).
      - DoTrait        → "With H's trait T clamped to V, …".
      - DoEvent        → "Under do(E={occurred|prevented}), …".
  • When the surgery is vacuous (Rule 3 pruned target_node_ids) say so
    plainly and lower confidence; do NOT invent downstream ripples.
  • When typed AFFECTED PROPOSITIONS / BELIEFS / CONCERNS are listed,
    foreground them in the answer rather than leading with low-stake
    surface state changes.
  • Never name the rung level or the words "do-operator" in the
    rendered prose. Use natural conditional language ("Suppose…",
    "If we force…", "Under that clamp,…").
  • Reference characters and events by human names in the prose;
    list exact node ids (and PROP_/CCN_ ids) in evidence_node_ids.
"""


_SYSTEM_PROMPT_COUNTERFACTUAL = """\
You are the Shadow Loom Pearl-Rung-3 (counterfactual) analyst. The
user asked a Rung-3 question — they want to know what *would have*
happened had the past been different. The world-state slice you are
given is the post-abduction, post-prediction sandbox after a
historical surgery (NOT the factual mainline).

You will be given:
  1. The user's question.
  2. The active AMWN branch and the omniscient counterfactual
     world-state slice (the simulated branch).
  3. A FACTUAL MAINLINE contrast block (when available) so you can
     diff actual vs counterfactual.
  4. Phase-7 RUNG-3 SURGERY METADATA when the parser produced typed
     ``historical_do_targets``. The metadata names the *kind* of
     historical surgery and lists the propositions, beliefs, and
     concerns that flipped between actual and counterfactual.
  5. A NARRATIVE FORM tag when one was inferred (tragic / comic /
     ironic / neutral) — apply the matching closing register.

Rules:
  • Answer ONLY from the counterfactual world state for what *would*
    happen; cite the factual mainline only when contrasting.
  • Match the surgery's epistemic / ontic register:
      - DoProposition  → "Had it been the case that PROP X = T, …".
      - DoBelief       → "Had H believed otherwise about PROP X, …"
        (epistemic — Romeo not believing Juliet dead, etc.).
      - DoConcern      → "Without H's concern C, …" (motivational —
        Roese commission/omission frame).
      - DoTrait        → "Had H been less/more T, …".
      - DoEvent        → "Had E not occurred (or had it gone
        differently), …".
  • Apply the narrative-form hedge:
      - tragic   → close with an "and yet" register; foreground regret.
      - comic    → close with an "and so" register; foreground relief.
      - ironic   → "as if to mock" — same magnitude, rearranged
        polarities.
      - neutral  → "though it would have made no difference".
  • Surface the AFFECTED PROPOSITIONS / BELIEFS / CONCERNS as the
    causal mechanism of the counterfactual outcome.
  • If the abduction did not yield enough to answer, say so plainly,
    lower confidence to <=0.3, and add a caveat.
  • Never name the rung level or "abduction" / "do-operator" in the
    rendered prose. Use natural subjunctive language.
  • Reference by human names in prose; list ids (incl. PROP_/CCN_) in
    evidence_node_ids.
"""


def _build_answer_agent(
    config: GenerationConfig,
    *,
    query_type: str,
) -> Agent[None, AnswerCard]:
    """Construct the Q&A agent for the given query type."""
    if query_type == "interrogate":
        system_prompt = _SYSTEM_PROMPT_INTERROGATE
    elif query_type == "intervention":
        system_prompt = _SYSTEM_PROMPT_INTERVENTION
    elif query_type == "counterfactual":
        system_prompt = _SYSTEM_PROMPT_COUNTERFACTUAL
    else:
        system_prompt = _SYSTEM_PROMPT_GENERAL
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
    branch_world_id: Literal["factual", "shadow"] = "factual",
    branch_label: Optional[str] = None,
    factual_contrast_summary: Optional[str] = None,
    preceding_prose: Optional[str] = None,
    narrative_style: Optional[Any] = None,
    do_targets: Optional[List[Dict[str, Any]]] = None,
    affected_propositions: Optional[List[str]] = None,
    affected_beliefs: Optional[List[str]] = None,
    affected_concerns: Optional[List[str]] = None,
    tragedy_form: Optional[str] = None,
) -> AnswerCard:
    """Answer a Q&A question using the supplied world-state slice.

    Returns an :class:`AnswerCard` even on failure (with a low
    confidence and a caveat) so callers never have to deal with
    ``None``.

    ``branch_world_id`` / ``branch_label`` / ``factual_contrast_summary``
    / ``preceding_prose`` carry the same AMWN branch context that the
    generation pipeline threads onto a CreativeBrief, so the Q&A
    answers stay consistent with whichever branch the user is
    currently exploring.

    ``narrative_style`` (a :class:`shadow_loom.models.NarrativeStyle`
    when populated) lets the answer agent mirror the source register
    — primarily so character names, place names, and tonal diction
    in the ``answer`` field match the world the user is exploring.
    The ``AnswerCard`` is structured output, so the influence is
    bounded to the ``answer`` and ``caveats`` strings; we do not
    enforce a word budget.
    """
    config = config or GenerationConfig()
    if not (question or "").strip():
        return AnswerCard(
            answer="(no question supplied)",
            confidence=0.0,
            caveats=["Empty question."],
        )

    context_block = _compress_world_state(
        physics_state, branch_world_id=branch_world_id,
    )
    user_msg_parts: List[str] = [
        f"Question: {question.strip()}",
        f"Query type: {query_type}",
        f"Active AMWN branch: {branch_world_id}"
        + (f" (label: {branch_label})" if branch_label else ""),
    ]
    if query_type == "interrogate":
        user_msg_parts.append(
            f"Require causal proof: {'yes' if require_proof else 'no'}"
        )

    # Phase-9: surface typed Pearl-rung surgery metadata so the
    # intervention / counterfactual answer agents can match the right
    # epistemic / ontic register and apply the narrative-form hedge.
    # Caller is expected to forward these from the rung-2 / rung-3
    # ``calculate_narrative_physics`` result dict (Phase-7 keys).
    if query_type in ("intervention", "counterfactual"):
        rung_label = "RUNG-2" if query_type == "intervention" else "RUNG-3"
        if do_targets:
            user_msg_parts.extend([
                "",
                f"=== {rung_label} SURGERY METADATA (typed do_targets) ===",
            ])
            for t in do_targets:
                kind = t.get("target_kind", "?")
                # Compact one-line summary per target so the LLM can
                # see the discriminator + the kind-specific payload.
                payload_keys = [k for k in t.keys() if k != "target_kind"]
                fields = ", ".join(
                    f"{k}={t[k]!r}" for k in payload_keys if t.get(k) is not None
                )
                user_msg_parts.append(f"  - {kind}: {fields}")
        if affected_propositions:
            user_msg_parts.append(
                "AFFECTED PROPOSITIONS (truth flipped under the surgery): "
                + ", ".join(affected_propositions)
            )
        if affected_beliefs:
            user_msg_parts.append(
                "AFFECTED BELIEFS (confidence shifted under the surgery): "
                + ", ".join(affected_beliefs)
            )
        if affected_concerns:
            user_msg_parts.append(
                "AFFECTED CONCERNS (satisfaction or salience shifted): "
                + ", ".join(affected_concerns)
            )
        if tragedy_form:
            user_msg_parts.append(
                f"NARRATIVE FORM: {tragedy_form} \u2014 apply the matching "
                "closing register per the system prompt."
            )
    if branch_world_id == "shadow" and factual_contrast_summary:
        user_msg_parts.extend([
            "",
            "=== FACTUAL MAINLINE AT SAME HORIZON (background contrast) ===",
            factual_contrast_summary.strip(),
            "(The active branch is a shadow fork — the prose above is "
            "what happened on canon. Use it only to disambiguate; do "
            "not assume the shadow branch follows it.)",
        ])
    if preceding_prose:
        user_msg_parts.extend([
            "",
            "=== STORY SO FAR (prior prose on this branch) ===",
            preceding_prose.strip(),
        ])
    if narrative_style is not None:
        # Surface the source register so the LLM uses the same
        # diction / formality the rest of the pipeline mirrors. We
        # deliberately do NOT enforce target_word_min/max — the
        # AnswerCard is a structured response, not a prose chunk.
        ns_lines = ["", "=== SOURCE REGISTER (for tone / diction only) ==="]
        fmt = getattr(narrative_style, "format", None)
        voice = getattr(narrative_style, "voice", None)
        density = getattr(narrative_style, "prose_density", None)
        exemplar = getattr(narrative_style, "style_exemplar", None)
        if fmt:
            ns_lines.append(f"  format: {fmt}")
        if voice:
            ns_lines.append(f"  voice: {voice}")
        if density:
            ns_lines.append(f"  prose density: {density}")
        if exemplar:
            ns_lines.append(f"  exemplar: {str(exemplar)[:600]}")
        user_msg_parts.extend(ns_lines)
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
