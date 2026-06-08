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
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput

from shadow_loom.generation import GenerationConfig
from shadow_loom.models import WorldStateV1
from shadow_loom.settings import resolve_model as _resolve_model, get_settings as _get_settings

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


def _coerce_sandbox_to_world_shape(
    physics_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate ``nx.node_link_data(sandbox)`` to WorldStateV1 shape.

    Rung-2 (intervention) and Rung-3 (counterfactual) physics emit
    ``physics_state = nx.node_link_data(self.sandbox)`` — a
    ``{"nodes": [...], "edges"|"links": [...], "graph": {...}}``
    payload. The Q&A compressor reads from the WorldStateV1 dict shape
    (``entities`` keyed by id, ``events`` as a list, ``social_topology``
    as a list of relationship edges, etc.). Without this translation
    the shadow-branch Q&A prompt was emitted with an empty world-state
    block.

    No-op when ``physics_state`` already looks like a WorldStateV1
    dump (no ``nodes`` key, or no ``edges``/``links`` bucket).
    """
    if not isinstance(physics_state, dict):
        return physics_state
    if "nodes" not in physics_state:
        return physics_state
    edges_bucket = (
        physics_state.get("edges")
        if "edges" in physics_state
        else physics_state.get("links")
    )
    if edges_bucket is None:
        return physics_state

    entities: Dict[str, Dict[str, Any]] = {}
    objects: Dict[str, Dict[str, Any]] = {}
    locations: Dict[str, Dict[str, Any]] = {}
    world_traits: Dict[str, Dict[str, Any]] = {}
    channels: Dict[str, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []
    for n in physics_state.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if not nid:
            continue
        nt = n.get("node_type")
        node = {k: v for k, v in n.items() if k != "node_type"}
        if nt == "Entity":
            entities[nid] = node
        elif nt == "NarrativeObject":
            objects[nid] = node
        elif nt == "Location":
            locations[nid] = node
        elif nt == "WorldTrait":
            world_traits[nid] = node
        elif nt == "Channel":
            channels[nid] = node
        elif nt == "EventNode":
            events.append(node)

    social: List[Dict[str, Any]] = []
    spatial: List[Dict[str, Any]] = []
    causal: List[Dict[str, Any]] = []
    for link in edges_bucket or []:
        if not isinstance(link, dict):
            continue
        et = link.get("edge_type")
        src = link.get("source")
        tgt = link.get("target")
        if et == "relationship":
            rel = dict(link)
            rel.setdefault("source_entity_id", src)
            rel.setdefault("target_entity_id", tgt)
            social.append(rel)
        elif et == "connected_to":
            se = dict(link)
            se.setdefault("source_id", src)
            se.setdefault("target_id", tgt)
            spatial.append(se)
        elif et == "causal":
            ce = dict(link)
            ce.setdefault("source_id", src)
            ce.setdefault("target_id", tgt)
            causal.append(ce)

    propositions = (
        (physics_state.get("graph") or {}).get("propositions") or []
    )

    return {
        "entities": entities,
        "objects": objects,
        "locations": locations,
        "world_traits": world_traits,
        "channels": channels,
        "events": events,
        "social_topology": social,
        "spatial_topology": spatial,
        "causal_topology": causal,
        "propositions": propositions,
    }


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

    # Rung-2/3 callers pass ``physics_state = nx.node_link_data(sandbox)``
    # which is shaped ``{"nodes": [...], "edges": [...]}`` (NetworkX
    # 3.4+; ``"links"`` on older releases). The compressor below
    # expects the WorldStateV1 shape (``entities``, ``events``,
    # ``social_topology``, ``spatial_topology``, ``channels``,
    # ``world_traits``, ``propositions``). Without this translation
    # the shadow-branch Q&A prompt's "World state at current temporal
    # anchor:" block came out empty, blinding the answer agent to
    # the very do-surgery the user was asking about.
    physics_state = _coerce_sandbox_to_world_shape(physics_state)

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
            # On a shadow read the projected ``entities`` dict layers
            # shadow clones over the factual baseline; tag each row
            # so the Q&A LLM can tell a do(·)-modified entity apart
            # from a factual passthrough (matches the tagging the
            # events / edges / channels / world-traits sections do).
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{ent.get('world_id', 'factual')}]"
            lines.append(
                f"- `{ent_id}` {name}{wid_tag} — status={status}, "
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
            # Concerns — standing fears/desires referencing a
            # proposition. Critical for "what does X want / fear",
            # "why is X motivated to do Y", and any rung-1 question
            # that turns on motivational state. Mirrors the
            # rung-2/3 do_target_context's concern coverage so
            # interrogation has parity with intervention reasoning.
            concerns = ent.get("concerns") or []
            if concerns:
                shown_concerns: List[str] = []
                for c in concerns[:6]:
                    if not isinstance(c, dict):
                        continue
                    ccn = c.get("concern_id", "?")
                    pol = c.get("polarity", "?")
                    pid = c.get("proposition_id", "?")
                    sal = c.get("salience")
                    sal_str = (
                        f" sal={float(sal):.2f}"
                        if isinstance(sal, (int, float)) else ""
                    )
                    kind = c.get("kind")
                    kind_str = f" kind={kind}" if kind else ""
                    shown_concerns.append(
                        f"{ccn}({pol} PROP {pid}{sal_str}{kind_str})"
                    )
                if shown_concerns:
                    lines.append(
                        "    concerns: " + " | ".join(shown_concerns)
                    )
                if len(concerns) > 6:
                    lines.append(f"    …(+{len(concerns) - 6} more concerns)")
            # State-timeline trajectory — the chain of history that
            # tells the interrogator HOW this entity arrived at its
            # current traits / location / beliefs. Without this an
            # interrogation answer to "how did X end up here" or
            # "what changed X" has to guess from events alone. Mirror
            # of the rung-2/3 trait/world_trait history sections.
            timeline = ent.get("state_timeline") or []
            if timeline:
                shown_snaps: List[str] = []
                for snap in timeline[-4:]:
                    if not isinstance(snap, dict):
                        continue
                    ft = snap.get("fabula_time", "?")
                    trig = snap.get("triggered_by")
                    sloc = snap.get("location_id")
                    bits = [f"t={ft}"]
                    if sloc:
                        bits.append(f"loc={sloc}")
                    if trig:
                        bits.append(f"via={trig}")
                    shown_snaps.append("[" + ", ".join(bits) + "]")
                if shown_snaps:
                    lines.append(
                        "    history: " + " \u2192 ".join(shown_snaps)
                    )
                if len(timeline) > 4:
                    lines.append(
                        f"    …(+{len(timeline) - 4} earlier snapshots)"
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
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{obj.get('world_id', 'factual')}]"
            lines.append(f"- `{obj_id}` {name}{wid_tag} (owner={owner})")

    events = physics_state.get("events", []) or []
    if events:
        # Spread across the full timeline: first quarter for opening
        # context + last three quarters for recent state. Without this
        # a head-only tail slice blinds the answer agent to events from
        # the beginning of the story when interventions / counterfactuals
        # target early acts.
        sorted_events = sorted(
            events, key=lambda e: e.get("fabula_time", 0),
        )
        if len(sorted_events) <= max_events:
            spread = sorted_events
        else:
            n_head = max(1, max_events // 4)
            n_tail = max_events - n_head
            spread = sorted_events[:n_head] + sorted_events[len(sorted_events) - n_tail:]
        lines.append("\n## Events (chronological)")
        if len(sorted_events) > max_events:
            lines.append(
                f"  [{len(sorted_events)} total; showing {len(spread)} "
                f"(first {n_head} + last {n_tail}) — increase max_events for more]"
            )
        for evt in spread:
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
            # Always label the truth status so the LLM consuming this
            # block never treats utterance content as ground truth.
            # When the extractor left ``truth_value`` unset we mark it
            # explicitly as ``unknown`` rather than silently omitting
            # the field \u2014 omitting reads as "this is a fact".
            tv_str = f" truth={tv or 'unknown'}"
            content = (u.get("content") or "").strip().replace("\n", " ")
            if len(content) > 200:
                content = content[:197] + "…"
            wid_tag = ""
            if branch_world_id == "shadow":
                wid_tag = f" [{u.get('world_id', 'factual')}]"
            # R19-M9: surface the channel an utterance routed over
            # so the answer agent can reason about channel reach /
            # severance instead of inferring face-to-face by
            # default.
            via_ch = u.get("via_channel_id")
            ch_tag = f" via=`{via_ch}`" if via_ch else ""
            lines.append(
                f"- T={ft} `{uid}`{wid_tag} {speaker} → [{addressees}]"
                f"{tv_str}{ch_tag}: {content}"
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
            # R19-M9: include the channel's availability window so
            # the answer agent doesn't cite a channel as evidence
            # outside its established/terminated fabula range.
            est = ch.get("established_at_fabula")
            term = ch.get("terminated_at_fabula")
            win_tag = ""
            if est is not None or term is not None:
                win_tag = (
                    f" window=[{est if est is not None else '?'}."
                    f".{term if term is not None else '∞'}]"
                )
            lines.append(f"- `{cid}` {medium}{wid_tag}{win_tag} participants=[{parts}]")

    # Propositions — the catalogue of structured factual claims.
    # ``truth_at_fabula`` is the physics commit log: the answer LLM
    # MUST treat ``False`` commits as factually NOT the case (even if
    # one or more characters still believe them) and ``True`` commits
    # as established fact. Without this section the Interrogator /
    # General Q&A surfaces only events + beliefs and silently misses
    # the propositional ground truth.
    propositions = physics_state.get("propositions", None)
    if propositions:
        if isinstance(propositions, dict):
            prop_iter = list(propositions.items())
        else:
            prop_iter = [
                (p.get("id") or p.get("proposition_id") or "?", p)
                for p in propositions
                if isinstance(p, dict)
            ]
        if prop_iter:
            lines.append("\n## Propositions (factual claims, with commit log)")
            for pid, p in prop_iter[:max_events]:
                desc = (p.get("description") or "").strip().replace("\n", " ")
                if len(desc) > 140:
                    desc = desc[:137] + "…"
                truth_map = p.get("truth_at_fabula") or {}
                if truth_map:
                    items = sorted(
                        ((int(t), bool(v)) for t, v in truth_map.items()),
                        key=lambda kv: kv[0],
                    )
                    truth_str = ", ".join(
                        f"T={t}:{'TRUE' if v else 'FALSE'}" for t, v in items
                    )
                    truth_part = f" truth=[{truth_str}]"
                else:
                    truth_part = " truth=[uncommitted]"
                # Tag proposition rows with their AMWN ``world_id`` on
                # shadow reads so the LLM can tell a shadow-clone
                # commit (the do-surgery's truth on this branch) apart
                # from a factual passthrough (still authoritative
                # because the shadow merge did not split it).
                wid_tag = ""
                if branch_world_id == "shadow":
                    wid_tag = f" [{p.get('world_id', 'factual')}]"
                lines.append(f"- `{pid}`{wid_tag}{truth_part} \u2014 {desc}")
            if len(prop_iter) > max_events:
                lines.append(
                    f"  …(+{len(prop_iter) - max_events} more propositions)"
                )

    # Negative facts — events the physics tags as NOT occurring and
    # propositions whose latest commit is FALSE. Surfaced as a
    # dedicated callout so the answer LLM cannot accidentally answer
    # "yes, X happened" about an event the world records as prevented,
    # or assert a falsified proposition as fact.
    prevented_events = [
        e for e in (physics_state.get("events", []) or [])
        if (e.get("event_type") or "") in {"prevented", "never_happened", "removed"}
    ]
    false_props: List[str] = []
    if propositions:
        if isinstance(propositions, dict):
            _piter = list(propositions.values())
        else:
            _piter = [p for p in propositions if isinstance(p, dict)]
        for p in _piter:
            truth_map = p.get("truth_at_fabula") or {}
            if not truth_map:
                continue
            items = sorted(
                ((int(t), bool(v)) for t, v in truth_map.items()),
                key=lambda kv: kv[0],
            )
            if items and items[-1][1] is False:
                pid = p.get("id") or p.get("proposition_id") or "?"
                desc = (p.get("description") or "").strip().replace("\n", " ")
                if len(desc) > 120:
                    desc = desc[:117] + "…"
                false_props.append(f"`{pid}` (false @ T={items[-1][0]}) — {desc}")
    if prevented_events or false_props:
        lines.append(
            "\n## NEGATIVE FACTS — these did NOT occur / are NOT true"
        )
        lines.append(
            "  Treat the items below as authoritative non-events. The "
            "answer must not assert them as having happened or being "
            "true. Characters may still *believe* a false proposition "
            "(belief ≠ fact); answer that distinction explicitly when "
            "asked, but do not enact the proposition as fact."
        )
        for evt in prevented_events[:max_events]:
            eid = evt.get("id", "?")
            ft = evt.get("fabula_time", "?")
            etype = evt.get("event_type", "?")
            desc = (evt.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 120:
                desc = desc[:117] + "…"
            lines.append(f"- T={ft} `{eid}` [{etype}] — {desc}")
        for fp in false_props[:max_events]:
            lines.append(f"- {fp}")

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
  • The world state has a NEGATIVE FACTS section listing events the
    physics tags as `prevented` / `never_happened` / `removed` and
    propositions whose latest commit in `truth_at_fabula` is FALSE.
    Treat these as authoritative non-events: do NOT assert any of
    them as having occurred or being true. A character may still
    *believe* a false proposition; surface that distinction
    explicitly when relevant (e.g. "Macbeth believes Banquo is
    dead" vs "Banquo is alive"), but do not enact the proposition
    as fact in the answer.
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
  • The world state has a NEGATIVE FACTS section listing events the
    physics tags as `prevented` / `never_happened` / `removed` and
    propositions whose latest commit is FALSE. Do NOT walk through
    a prevented event as if it occurred, do NOT cite a falsified
    proposition as a cause, and do NOT include their node ids in
    evidence_node_ids as supporting evidence for a positive claim.
    A causal chain that depends on a prevented event collapses —
    say so plainly.
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
     / DoTrait / DoNarrativeObject / DoWorldTrait / DoChannel /
     DoRelationship / DoCausalEdge / DoSpatialEdge / DoEntityDelete /
     DoObjectDelete — and lists the propositions, beliefs, concerns,
     objects, world-traits, edges, and excisions whose values shifted
     relative to the factual world.

Rules:
  • Answer ONLY from the supplied (post-do) world state. The factual
    mainline is background contrast; do NOT default to it.
  • Match the surgery's epistemic / ontic register:
      - DoProposition       → "Under the clamp that PROP X is true, …" (ontic).
      - DoBelief            → "From holder H's clamped belief …" (epistemic;
        the world may be unchanged but H's beliefs were forced).
      - DoConcern           → "With holder H's concern C clamped to
        salience S, …" (motivational; reweighs disposition, not facts).
      - DoTrait             → "With H's trait T clamped to V, …".
      - DoEvent             → "Under do(E={occurred|prevented}), …". When
        ``new_at_location_id`` is set, render as "Under do(E moved
        to LOC X), …" — the event still happens but at a
        different location, dragging its actors with it. When
        ``new_fabula_time`` is set, render as "Under do(E shifted
        to t=N), …".
      - DoNarrativeObject   → "With OBJ X clamped to {location|owner|
        property}, …" (object relocation / ownership / properties).
      - DoWorldTrait        → "Under the ambient clamp WORLD_T = V, …"
        (storm-intensity, regime-grip, season — the global force layer).
      - DoCausalEdge        → "With the causal link E_A→E_B {severed|
        added}, …" (re-write the storyworld's mechanism, not its
        propositions).
      - DoSpatialEdge       → "With LOC_A→LOC_B {severed|locked|
        unlocked}, …" (topology re-write — a closed passage, a
        new shortcut, a locked door).
      - DoChannel           → "With channel CHN_K {activated|
        deactivated|retuned}, …" (epistemic carrier severed / opened).
      - DoRelationship      → "With H↔T's {affinity|power|fear} clamped
        to V, …".
      - DoEntityDelete      → "As if ENT_X had never existed, …"
        (existence excision — cascades remove their social/belief/
        causal footprint; surface the excision as the mechanism).
      - DoObjectDelete      → "As if OBJ_X had never existed, …".
  • When the surgery is vacuous (Rule 3 pruned target_node_ids) say so
    plainly and lower confidence; do NOT invent downstream ripples.
  • When typed AFFECTED PROPOSITIONS / BELIEFS / CONCERNS / OBJECTS /
    WORLD TRAITS / EDGES / EVENTS / ENTITY DELETIONS / OBJECT
    DELETIONS are listed, foreground them in the answer rather than
    leading with low-stake surface state changes.
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
     historical surgery and lists the propositions, beliefs, concerns,
     objects, world-traits, edges, and excisions that flipped between
     actual and counterfactual.
  5. A NARRATIVE FORM tag when one was inferred (tragic / comic /
     ironic / neutral) — apply the matching closing register.

Rules:
  • Answer ONLY from the counterfactual world state for what *would*
    happen; cite the factual mainline only when contrasting.
  • Match the surgery's epistemic / ontic register:
      - DoProposition       → "Had it been the case that PROP X = T, …".
      - DoBelief            → "Had H believed otherwise about PROP X, …"
        (epistemic — Romeo not believing Juliet dead, etc.).
      - DoConcern           → "Without H's concern C, …" (motivational —
        Roese commission/omission frame).
      - DoTrait             → "Had H been less/more T, …".
      - DoEvent             → "Had E not occurred (or had it gone
        differently), …". When ``new_at_location_id`` is set,
        render as "Had E happened at LOC X instead, …". When
        ``new_fabula_time`` is set, render as "Had E happened at
        t=N instead, …".
      - DoNarrativeObject   → "Had OBJ X been at LOC Y / owned by H
        instead, …".
      - DoWorldTrait        → "Had the ambient WORLD_T been V instead, …"
        (storm calmed, regime weakened, winter delayed).
      - DoCausalEdge        → "Had E_A not led to E_B, …" / "Had E_A
        caused E_B, …" (mechanism counterfactual).
      - DoSpatialEdge       → "Had LOC_A been reachable from LOC_B, …"
        / "Had the passage been locked, …".
      - DoChannel           → "Had channel CHN_K been open/severed, …".
      - DoRelationship      → "Had H↔T's {affinity|power|fear} been V, …".
      - DoEntityDelete      → "Had ENT_X never existed, …" (the
        excision is the lever; cascade-reasoning is the mechanism).
      - DoObjectDelete      → "Had OBJ_X never existed, …".
  • Apply the narrative-form hedge:
      - tragic   → close with an "and yet" register; foreground regret.
      - comic    → close with an "and so" register; foreground relief.
      - ironic   → "as if to mock" — same magnitude, rearranged
        polarities.
      - neutral  → "though it would have made no difference".
  • Surface the AFFECTED PROPOSITIONS / BELIEFS / CONCERNS / OBJECTS /
    WORLD TRAITS / EDGES / EVENTS / ENTITY DELETIONS / OBJECT
    DELETIONS as the causal mechanism of the counterfactual outcome.
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
        _resolve_model(config.model, stage="generation"),
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
    world_state: Optional[WorldStateV1] = None,
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
    affected_objects: Optional[List[str]] = None,
    affected_world_traits: Optional[List[str]] = None,
    affected_edges: Optional[List[str]] = None,
    affected_entity_deletes: Optional[List[str]] = None,
    affected_object_deletes: Optional[List[str]] = None,
    affected_events: Optional[List[str]] = None,
    tragedy_form: Optional[str] = None,
    # Phase-10: downstream consequence cascades. The intervention /
    # counterfactual Q&A path historically only saw affected-id lists,
    # so the agent could not enumerate cascading consequences. Mirror
    # the engine-emitted mutation logs the generation brief now
    # carries so the AnswerCard cites the actual propagation chain
    # rather than a one-line surgery summary.
    mutations: Optional[List[Dict[str, Any]]] = None,
    social_mutations: Optional[List[Dict[str, Any]]] = None,
    proposition_mutations: Optional[List[Dict[str, Any]]] = None,
    belief_mutations: Optional[List[Dict[str, Any]]] = None,
    concern_mutations: Optional[List[Dict[str, Any]]] = None,
    # Round-5 audit: object / world-trait / edge mutation streams
    # produced by ``_typed_target_payload`` were silently dropped at
    # the Q&A renderer hand-off. Accept them so prop relocations,
    # ambient-force shifts, and topology surgeries reach the answer
    # prompt instead of vanishing between physics and renderer.
    object_mutations: Optional[List[Dict[str, Any]]] = None,
    world_trait_mutations: Optional[List[Dict[str, Any]]] = None,
    edge_mutations: Optional[List[Dict[str, Any]]] = None,
    entity_delete_mutations: Optional[List[Dict[str, Any]]] = None,
    object_delete_mutations: Optional[List[Dict[str, Any]]] = None,
    event_mutations: Optional[List[Dict[str, Any]]] = None,
    blocked: Optional[List[Dict[str, Any]]] = None,
    causal_chain: Optional[List[str]] = None,
    # 2026-05-29 deep-audit HIGH-1: the pipeline computes deterministic
    # posterior diagnostics and world-schema warnings on every rung-1
    # call (``physics_result["posterior_warnings"]`` /
    # ``physics_result["world_schema_warnings"]``), but the call site
    # at ``shadow_loom/pipeline.py`` previously only forwarded
    # ``physics_state``. The warnings stayed buried inside the physics
    # result and never reached the answer-LLM, so the AnswerCard could
    # confidently contradict the engine's own diagnostics. Accept them
    # here so the rung-1 / rung-2 / rung-3 answer prompts can surface
    # them in a dedicated DIAGNOSTICS block.
    posterior_warnings: Optional[List[str]] = None,
    world_schema_warnings: Optional[List[str]] = None,
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

    _gs = _get_settings().generation
    context_block = _compress_world_state(
        physics_state,
        branch_world_id=branch_world_id,
        max_entities=_gs.answer_max_entities,
        max_events=_gs.answer_max_events,
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

    # 2026-05-29 deep-audit HIGH-1: surface deterministic posterior +
    # schema diagnostics so the answer agent cannot contradict them.
    # These come from the pipeline's rung-1 posterior computer and
    # world-schema auditor; both produce structured warning strings
    # that the renderer / auditor already see. Mirroring them on the
    # Q&A prompt keeps all three surfaces (renderer, auditor, Q&A)
    # consistent.
    if posterior_warnings:
        user_msg_parts.append("")
        user_msg_parts.append("=== POSTERIOR DIAGNOSTICS (deterministic) ===")
        for _w in posterior_warnings[:50]:
            user_msg_parts.append(f"  - {_w}")
        user_msg_parts.append(
            "Treat each diagnostic above as authoritative; do not "
            "contradict it in the answer. If the diagnostic narrows "
            "or rules out a hypothesis the question raises, reflect "
            "that in the answer + caveats."
        )
    if world_schema_warnings:
        user_msg_parts.append("")
        user_msg_parts.append("=== WORLD-SCHEMA WARNINGS (engine audit) ===")
        for _w in world_schema_warnings[:50]:
            user_msg_parts.append(f"  - {_w}")
        user_msg_parts.append(
            "These warnings describe structural gaps in the world "
            "state (missing referents, orphaned ids, malformed "
            "truth-keys). If a warning intersects the question's "
            "subject matter, mention the uncertainty in caveats "
            "rather than guessing past it."
        )

    # Phase-9: surface typed Pearl-rung surgery metadata so the
    # intervention / counterfactual answer agents can match the right
    # epistemic / ontic register and apply the narrative-form hedge.
    # Caller is expected to forward these from the rung-2 / rung-3
    # ``calculate_narrative_physics`` result dict (Phase-7 keys).
    if query_type in ("intervention", "counterfactual"):
        from shadow_loom.generation import (
            _resolve_affected_descriptions,
            _annotate_ids,
            _format_trait_mutation_lines,
            _format_social_mutation_lines,
            _format_proposition_mutation_lines,
            _format_belief_mutation_lines,
            _format_concern_mutation_lines,
            _format_object_mutation_lines,
            _format_world_trait_mutation_lines,
            _format_edge_mutation_lines,
            _format_entity_delete_mutation_lines,
            _format_object_delete_mutation_lines,
            _format_event_mutation_lines,
            _format_blocked_propagation_lines,
            _format_do_target_causal_context,
        )
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
                # Rich causal-neighbourhood block for every do-target
                # kind so the answer agent can reason about the
                # precursor chain / belief network / trait history
                # around the clamp rather than treat the do-target
                # as an opaque id. Mirror of the renderer / auditor
                # ``do_target_context`` surface; the helper dispatches
                # by ``target_kind`` so every clamp surface (event,
                # proposition, belief, concern, trait, world_trait,
                # channel, relationship, causal_edge, spatial_edge,
                # object) is covered.
                if world_state is not None:
                    _de = SimpleNamespace(**t)
                    _ctx = _format_do_target_causal_context(world_state, _de)
                    if _ctx:
                        user_msg_parts.append("    " + _ctx.replace("\n", "\n    "))
        if affected_propositions:
            _prop_descs = _resolve_affected_descriptions(world_state, affected_propositions, "prop") if world_state else []
            user_msg_parts.append(
                "AFFECTED PROPOSITIONS (truth flipped under the surgery): "
                + _annotate_ids(affected_propositions, _prop_descs)
            )
        if affected_beliefs:
            _belief_descs = _resolve_affected_descriptions(world_state, affected_beliefs, "belief") if world_state else []
            user_msg_parts.append(
                "AFFECTED BELIEFS (confidence shifted under the surgery): "
                + _annotate_ids(affected_beliefs, _belief_descs)
            )
        if affected_concerns:
            _concern_descs = _resolve_affected_descriptions(world_state, affected_concerns, "concern") if world_state else []
            user_msg_parts.append(
                "AFFECTED CONCERNS (satisfaction or salience shifted): "
                + _annotate_ids(affected_concerns, _concern_descs)
            )
        if affected_objects:
            user_msg_parts.append(
                "AFFECTED OBJECTS (relocated / owner-changed / property-set): "
                + ", ".join(affected_objects[:20])
            )
        if affected_world_traits:
            user_msg_parts.append(
                "AFFECTED WORLD TRAITS (ambient-force values clamped): "
                + ", ".join(affected_world_traits[:20])
            )
        if affected_edges:
            user_msg_parts.append(
                "AFFECTED EDGES (topology rewrites \u2014 edge_type:action:source\u2192target): "
                + ", ".join(affected_edges[:20])
            )
        if affected_entity_deletes:
            user_msg_parts.append(
                "AFFECTED ENTITY DELETIONS (existence-counterfactual excisions): "
                + ", ".join(affected_entity_deletes[:20])
            )
        if affected_object_deletes:
            user_msg_parts.append(
                "AFFECTED OBJECT DELETIONS (existence-counterfactual excisions): "
                + ", ".join(affected_object_deletes[:20])
            )
        if affected_events:
            user_msg_parts.append(
                "AFFECTED EVENTS (relocated / time-shifted \u2014 event_id:kind): "
                + ", ".join(affected_events[:20])
            )
        if tragedy_form:
            user_msg_parts.append(
                f"NARRATIVE FORM: {tragedy_form} \u2014 apply the matching "
                "closing register per the system prompt."
            )
        # Phase-10: surface the engine-emitted downstream cascades so
        # the answer agent can enumerate the actual propagation rather
        # than guess "this surgery would affect X". Without this the
        # AnswerCard.answer landed short and missed cascading effects
        # the engine had already computed.
        cascade_sections = [
            ("DOWNSTREAM TRAIT CASCADES",
             _format_trait_mutation_lines(mutations)),
            ("DOWNSTREAM RELATIONSHIP CASCADES",
             _format_social_mutation_lines(social_mutations)),
            ("PROPOSITION CASCADES",
             _format_proposition_mutation_lines(proposition_mutations)),
            ("BELIEF CASCADES",
             _format_belief_mutation_lines(belief_mutations)),
            ("CONCERN CASCADES",
             _format_concern_mutation_lines(concern_mutations)),
            ("OBJECT CASCADES",
             _format_object_mutation_lines(object_mutations)),
            ("WORLD-TRAIT CASCADES",
             _format_world_trait_mutation_lines(world_trait_mutations)),
            ("EDGE CASCADES",
             _format_edge_mutation_lines(edge_mutations)),
            ("ENTITY DELETION CASCADES",
             _format_entity_delete_mutation_lines(entity_delete_mutations)),
            ("OBJECT DELETION CASCADES",
             _format_object_delete_mutation_lines(object_delete_mutations)),
            ("EVENT CASCADES",
             _format_event_mutation_lines(event_mutations)),
            ("BLOCKED PROPAGATIONS",
             _format_blocked_propagation_lines(blocked)),
        ]
        for header, bullets in cascade_sections:
            if bullets:
                user_msg_parts.append(f"{header}:")
                user_msg_parts.extend(bullets)
        if causal_chain:
            _chain_descs = _resolve_affected_descriptions(world_state, causal_chain, "event") if world_state else []
            user_msg_parts.append(
                "CAUSAL CHAIN (events through which the surgery "
                "propagates): " + " \u2192 ".join(
                    f'{eid} ("{d}")' if d and d != eid else eid
                    for eid, d in zip(
                        causal_chain,
                        _chain_descs if len(_chain_descs) == len(causal_chain) else causal_chain,
                    )
                )
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
        # Round-7 audit: enforce token budget by truncating preceding prose
        _pp_max = _get_settings().generation.preceding_prose_max_chars
        _pp = preceding_prose.strip()
        if len(_pp) > _pp_max:
            _pp = "\u2026" + _pp[-_pp_max:]
        # On a shadow fork the joined prose tail is a mix of factual
        # ancestors (the canon the fork branched from) and shadow
        # continuation. Without a banner the answer LLM treats every
        # paragraph as authoritative "this branch" history and ends
        # up asserting factual-only events (e.g. Mrs Coady's heart
        # attack) that the shadow physics state has explicitly
        # suppressed. Each block is already marked with its
        # ``(factual: …)`` / ``(shadow: …)`` provenance by
        # ``_gather_preceding_prose``; the wrapper here just promotes
        # that distinction into an explicit precedence rule.
        if branch_world_id == "shadow":
            user_msg_parts.extend([
                "",
                "=== STORY SO FAR (mixed: factual canon + shadow fork tail) ===",
                _pp,
                "(The blocks tagged ``(factual: …)`` are the canon the "
                "shadow fork diverges FROM — they describe what would "
                "have happened on the mainline, NOT what is true on this "
                "branch. Where the WORLD STATE block contradicts a "
                "factual prose detail (a suppressed event, a flipped "
                "proposition, a missing snapshot), the WORLD STATE is "
                "authoritative; treat the conflicting factual prose as "
                "superseded by the intervention. The blocks tagged "
                "``(shadow: …)`` are this fork's own continuation and "
                "remain in force.)",
            ])
        else:
            user_msg_parts.extend([
                "",
                "=== STORY SO FAR (prior prose on this branch) ===",
                _pp,
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

    # Emit the full Q&A prompt at DEBUG so the answering agent's
    # instructions can be analysed alongside the renderer / auditor
    # / evaluator prompts. Wrapped with BEGIN/END markers for
    # unambiguous extraction from the log stream. INFO would expose
    # world context / branch contrast / prior prose to shared log
    # aggregators (data-exposure risk — round-3 audit).
    import os
    if os.environ.get("SECURE_DEBUG_PROMPT") == "1":
        logger.debug(
            "[Answer] Q&A prompt (q_type=%s, %d chars):\n"
            "========== BEGIN ANSWER PROMPT ==========\n%s\n"
            "========== END ANSWER PROMPT ==========",
            query_type,
            len(user_msg),
            user_msg,
        )
    logger.info(
        "[Answer] Q&A prompt prepared (q_type=%s, %d chars)",
        query_type, len(user_msg),
    )

    agent = _build_answer_agent(config, query_type=query_type)
    try:
        # Round-12 R12-02: bound the answer-agent call so a hung
        # provider can't pin the request handler indefinitely. The
        # caller fails fast with TimeoutError; the underlying HTTP
        # request keeps running on the worker thread until the
        # provider closes it. Timeout is tunable via the env var
        # ``SHADOW_LOOM_LLM_TIMEOUT_S`` (default 120s) shared with
        # query_parsing.
        import concurrent.futures as _cf
        import os as _os

        try:
            _timeout = float(_os.environ.get("SHADOW_LOOM_LLM_TIMEOUT_S", "120"))
        except (TypeError, ValueError):
            _timeout = 120.0

        with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
            _fut = _pool.submit(
                agent.run_sync,
                user_msg,
                model_settings={
                    "max_tokens": config.answer_max_tokens,
                },
            )
            try:
                result = _fut.result(timeout=_timeout)
            except _cf.TimeoutError as exc:
                logger.warning(
                    "[Answer] agent.run_sync exceeded %.1fs timeout", _timeout,
                )
                raise TimeoutError(
                    f"Answer LLM call did not complete within {_timeout:.0f}s"
                ) from exc
        card = result.output
        # Validate that ``evidence_node_ids`` actually resolve to
        # nodes the answer agent could plausibly cite. ``physics_state``
        # is the projection we just rendered into the prompt; node
        # ids that don't appear there are model hallucinations.
        # Filter them out and surface a caveat so the chat card /
        # auditor / require_proof gate see only grounded evidence.
        evidence = list(card.evidence_node_ids or [])
        if evidence and isinstance(physics_state, dict):
            known_ids: set = set()
            # NB: the projected ``physics_state`` keys vary with the
            # extractor (full vs ego vs interrogate) and Q&A query type.
            # Cover both the legacy keys and the canonical world-state
            # shape: ``entities`` (not "characters"), ``social_topology``
            # (not "relationships"), ``causal_topology``, plus the
            # affect layer (propositions / beliefs / concerns) and
            # topology lists. Missing a key here causes valid evidence
            # ids to be falsely dropped as hallucinations.
            for collection_key in (
                "events", "entities", "characters", "locations", "objects",
                "propositions", "beliefs", "concerns", "world_traits",
                "channels", "relationships", "social_topology",
                "causal_topology", "spatial_topology",
            ):
                collection = physics_state.get(collection_key)
                if isinstance(collection, dict):
                    known_ids.update(str(k) for k in collection.keys())
                elif isinstance(collection, list):
                    for entry in collection:
                        if isinstance(entry, dict):
                            # Pick up any of the id-shaped fields the
                            # entry exposes. Topology entries carry
                            # source_id / target_id rather than a
                            # primary id; without harvesting them
                            # the evidence-grounding check rejects
                            # all social / spatial / causal arrow
                            # evidence as hallucinated.
                            for id_key in (
                                "id", "node_id", "event_id",
                                "source_id", "target_id",
                                "source_entity_id", "target_entity_id",
                                "channel_id", "proposition_id",
                                "belief_id", "concern_id",
                                "world_trait_id", "object_id",
                            ):
                                if id_key in entry and entry[id_key]:
                                    known_ids.add(str(entry[id_key]))
            # R20: also fold in canonical ``world_state`` ids when
            # available. Counterfactual / intervention answer paths
            # often render a *thin* shadow projection that omits
            # spectator entities, ambient events, and unchanged
            # propositions — the agent then cites a perfectly valid
            # canonical id and the grounding check drops it as
            # "unsupported", collapsing the AnswerCard confidence
            # to ~0. We trust the parent world_state as ground truth
            # and let the branch-tombstone subtraction below remove
            # any ids the shadow has explicitly excised.
            if world_state is not None:
                for ent in (getattr(world_state, "entities", None) or {}).keys():
                    known_ids.add(str(ent))
                for ev in (getattr(world_state, "events", None) or []):
                    eid = getattr(ev, "id", None) or getattr(ev, "event_id", None)
                    if eid:
                        known_ids.add(str(eid))
                for loc in (getattr(world_state, "locations", None) or {}).keys():
                    known_ids.add(str(loc))
                for obj in (getattr(world_state, "objects", None) or {}).keys():
                    known_ids.add(str(obj))
                for ch in (getattr(world_state, "channels", None) or {}).keys():
                    known_ids.add(str(ch))
                # 2026-05-29 round-3 HIGH: ``WorldStateV1.propositions`` is
                # a ``List[Proposition]`` (not a dict); ``beliefs`` /
                # ``concerns`` do NOT live on WorldStateV1 at all
                # (they hang off ``Entity.beliefs`` /
                # ``Entity.concerns``). The pre-fix ``.keys()`` loops
                # would either ``AttributeError`` against any world
                # with non-empty propositions (Death on the Nile,
                # Macbeth, every shipped example) or silently
                # short-circuit through ``{}`` and leave the
                # corresponding id namespace ungrounded \u2014 so a
                # perfectly valid PROP_/CCN_ citation in
                # ``evidence_node_ids`` was demoted to "unsupported"
                # and the AnswerCard confidence dropped to ~0.
                for p in (getattr(world_state, "propositions", None) or []):
                    pid = getattr(p, "proposition_id", None) or getattr(p, "id", None)
                    if pid:
                        known_ids.add(str(pid))
                for _ent in (getattr(world_state, "entities", None) or {}).values():
                    for _b in (getattr(_ent, "beliefs", None) or []):
                        _bpid = getattr(_b, "proposition_id", None)
                        if _bpid:
                            known_ids.add(str(_bpid))
                        _btgt = getattr(_b, "target_id", None)
                        if _btgt:
                            known_ids.add(str(_btgt))
                    for _c in (getattr(_ent, "concerns", None) or []):
                        _cid = getattr(_c, "concern_id", None)
                        if _cid:
                            known_ids.add(str(_cid))
                        _cpid = getattr(_c, "proposition_id", None)
                        if _cpid:
                            known_ids.add(str(_cpid))
                for wt in (getattr(world_state, "world_traits", None) or {}).keys():
                    known_ids.add(str(wt))
            unsupported = [eid for eid in evidence if str(eid) not in known_ids]
            # R19-H8: when answering on a shadow branch, subtract any
            # ids the branch has tombstoned. ``physics_state`` may
            # have been built from a partially-projected world (R19-H14
            # is still extending ``projected_for_branch`` to events /
            # channels / topologies) so canonical-list-derived ids
            # can survive into ``known_ids`` even when the shadow has
            # already deleted them. Treat tombstoned ids as
            # ungrounded so cited evidence cannot point at things
            # the branch claims no longer exist.
            if (
                world_state is not None
                and branch_world_id == "shadow"
                and branch_label
            ):
                tombstoned: set = set()
                for sidecar_name in (
                    "shadow_removed_entity_ids",
                    "shadow_removed_object_ids",
                    "shadow_removed_channel_ids",
                    "shadow_removed_event_ids",
                    "shadow_removed_location_ids",
                ):
                    sidecar = getattr(world_state, sidecar_name, None) or {}
                    for rid in (sidecar.get(branch_label) or []):
                        tombstoned.add(str(rid))
                if tombstoned:
                    known_ids -= tombstoned
                    # Recompute ``unsupported`` so the caveat / drop
                    # logic below sees the post-tombstone view.
                    unsupported = [
                        eid for eid in evidence if str(eid) not in known_ids
                    ]
            # R19-H7: previously the "Only filter when we actually
            # know the id space" guard caused the hallucination
            # filter to **bypass entirely** whenever the projection
            # was empty or in an unrecognised shape. That meant a
            # model could fabricate arbitrary EVT/ENT ids on a
            # degenerate physics_state and ``require_proof`` would
            # accept them. We now treat *no known ids* as
            # *no grounded ids* and drop every evidence id, so the
            # downstream ``require_proof`` gate fails hard instead
            # of trusting fabricated provenance.
            if unsupported:
                if known_ids:
                    supported = [eid for eid in evidence if str(eid) in known_ids]
                else:
                    supported = []
                caveats = list(card.caveats or [])
                if known_ids:
                    caveats.append(
                        "Some evidence node ids returned by the answer agent "
                        "did not resolve in the projected world state and "
                        "were dropped: "
                        + ", ".join(str(u) for u in unsupported[:8])
                        + ("…" if len(unsupported) > 8 else "")
                    )
                else:
                    caveats.append(
                        "Projected world state exposed no resolvable ids; "
                        "all evidence ids returned by the answer agent "
                        "were treated as ungrounded and dropped."
                    )
                # Down-rank confidence proportional to how many ids
                # were unsupported.
                hallucination_ratio = (
                    len(unsupported) / max(1, len(evidence))
                )
                card = AnswerCard(
                    answer=card.answer,
                    confidence=max(
                        0.0,
                        float(card.confidence or 0.0)
                        * (1.0 - hallucination_ratio),
                    ),
                    caveats=caveats,
                    evidence_node_ids=supported,
                )
                logger.warning(
                    "[Answer] Dropped %d/%d unsupported evidence ids",
                    len(unsupported), len(evidence),
                )
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
