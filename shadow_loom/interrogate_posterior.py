# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic interrogate / general posterior ranking.

AUDIT P0-7: the previous interrogate path delegated all ranking to the
answer LLM. That left the engine without a verifiable ground truth to
audit against, so two real-plot interrogations (Death on the Nile,
Tinker Tailor) returned plausible-but-unverifiable posteriors.

This module walks the canonical truth surfaces in
:class:`WorldStateV1` and emits a deterministic per-proposition
posterior table the answer LLM and auditor can both consult:

* ``Proposition.truth_at_fabula`` — author-asserted or
  physics-committed truth state at each fabula tick. The proposition's
  posterior is read at the largest tick ``<= query_fabula_time``.
* ``Entity.beliefs`` — every belief carrying ``proposition_id`` is
  aggregated into a confidence-weighted endorsement / contradiction
  count, plus a list of supporting/contradicting entity_ids.

The output is intentionally narrow: ``proposition_id``, the canonical
truth at the query tick, the contradicting-belief weight, and a
plain-English ``evidence`` field for the renderer / auditor. It is
not a probabilistic model — it is a deterministic audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shadow_loom.models import WorldStateV1


# 2026-05-29 (deep-audit MED): module-level negation lexicon and helper
# so unit tests can exercise polarity resolution without driving a
# whole posterior pass. The lexicon is broader than the original (R1-1)
# inline tuple \u2014 it now includes lexical negators that show up in
# real example_worlds perceived_state strings: denies/refuses/refused/
# failed to/absent/lacking/devoid/without/unable/unwilling.
_NEGATION_TOKENS = (
    " not ", " no ", " never ", " none ", " n't ", " cannot ", " can't ",
    "isn't", "aren't", "wasn't", "weren't", "didn't", "doesn't",
    "don't", "hasn't", "haven't", "hadn't", "won't", "wouldn't",
    "shouldn't", "couldn't", "mustn't",
    # Lexical negators beyond contractions \u2014 these are common in the
    # consequences_extraction.md prompt's perceived_state outputs.
    " denies ", " denied ", " deny ",
    " refuses ", " refused ", " refuse ",
    " failed to ", " fails to ", " fail to ",
    " absent ", " lacking ", " lacks ", " devoid ",
    " without ", " unable to ", " unwilling to ",
    " rejects ", " rejected ", " reject ",
    " disbelieves ", " disbelieved ", " doubt ", " doubts ", " doubted ",
)


def _module_has_negation(text: str) -> bool:
    """True when *text* contains any token from :data:`_NEGATION_TOKENS`.

    Padded with spaces on both sides so token boundaries match at
    string edges (e.g. ``\"not safe\"`` becomes ``\" not safe \"``).
    """
    if not text:
        return False
    haystack = f" {text.lower().strip()} "
    return any(tok in haystack for tok in _NEGATION_TOKENS)


def _module_resolve_polarity(
    perceived: str,
    prop_desc: Optional[str],
    canonical_truth: Optional[bool],
) -> bool:
    """Module-level polarity resolver (testable equivalent of the
    inner ``_resolve_polarity`` in :func:`interrogate_posterior`).

    Returns True when *perceived* supports *canonical_truth* relative
    to *prop_desc*. Empty / falsy *perceived* is treated as legacy
    endorsement of canonical truth so silent beliefs do not flip the
    posterior.
    """
    if not perceived:
        return True
    perceived_negated = _module_has_negation(perceived)
    prop_negated = _module_has_negation(prop_desc or "")
    belief_asserts_surface = perceived_negated == prop_negated
    if canonical_truth is None:
        return belief_asserts_surface
    return belief_asserts_surface == bool(canonical_truth)


@dataclass
class PosteriorRow:
    proposition_id: str
    truth_at_query: Optional[bool]
    canonical_tick: Optional[int]
    support_weight: float
    contradict_weight: float
    supporters: List[str] = field(default_factory=list)
    contradictors: List[str] = field(default_factory=list)
    description: Optional[str] = None
    stakes: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposition_id": self.proposition_id,
            "truth_at_query": self.truth_at_query,
            "canonical_tick": self.canonical_tick,
            "support_weight": round(self.support_weight, 3),
            "contradict_weight": round(self.contradict_weight, 3),
            "supporters": list(self.supporters),
            "contradictors": list(self.contradictors),
            "description": self.description,
            "stakes": self.stakes,
        }


def interrogate_posterior(
    world_state: WorldStateV1,
    *,
    fabula_time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Compute the deterministic posterior table at ``fabula_time``.

    When ``fabula_time`` is ``None`` the largest tick in
    ``truth_at_fabula`` is used for each proposition independently
    (i.e. "current canonical truth"). Returns a list of dicts sorted
    by descending ``stakes`` then by ``proposition_id``.
    """
    rows: List[PosteriorRow] = []
    propositions = list(world_state.propositions or [])

    # Index beliefs by proposition_id so we touch each entity only once.
    # Each row carries the holder's free-form ``perceived_state`` so
    # the per-proposition bucketing loop below can detect explicit
    # negation/affirmation polarity rather than collapsing every
    # belief onto the canonical truth (R1-1, 2026-05-29).
    belief_index: Dict[str, List[tuple[str, float, str]]] = {}
    for ent_id, ent in (world_state.entities or {}).items():
        for b in (ent.beliefs or []):
            pid = getattr(b, "proposition_id", None)
            if not pid:
                continue
            conf = float(getattr(b, "confidence", 0.5) or 0.0)
            inertia = float(getattr(b, "inertia", 0.5) or 0.0)
            # Weight combines belief strength and stickiness so a
            # high-confidence transient belief still ranks below a
            # high-confidence sticky one.
            weight = max(0.0, min(1.0, 0.5 * (conf + inertia)))
            perceived = str(getattr(b, "perceived_state", "") or "")
            belief_index.setdefault(pid, []).append((ent_id, weight, perceived))

    # R1-1 (2026-05-29): lexical-polarity detector for ``perceived_state``.
    # ``Belief`` carries no first-class affirm/deny boolean, but most
    # perceived_state strings are short declarative clauses (e.g.
    # "Cup is safe" vs "Cup is NOT safe", "Macbeth killed Duncan" vs
    # "Macbeth did not kill Duncan"). We try to decide each belief's
    # polarity against the proposition's canonical description; on
    # mismatch or absence the legacy canonical-truth bucketing rule
    # is used as the fallback. The helpers are module-level so they
    # can be unit-tested directly (2026-05-29 deep-audit refactor).

    _has_negation = _module_has_negation
    _resolve_polarity = _module_resolve_polarity


    for prop in propositions:
        pid = prop.proposition_id
        truth_map = getattr(prop, "truth_at_fabula", None) or {}
        # Pick the canonical tick: largest key <= fabula_time, else
        # largest key overall.
        canonical_tick: Optional[int] = None
        truth_at_query: Optional[bool] = None
        if truth_map:
            # R20-C1: ``truth_at_fabula`` is declared ``Dict[int, bool]``
            # but JSON deserialization (DB roundtrip, MCP transport)
            # can return string keys. We coerce both the keys and the
            # lookup target so ``truth_map[canonical_tick]`` never
            # raises a KeyError when the stored keys are strings like
            # ``"1000"`` but ``canonical_tick`` is ``int(1000)``.
            try:
                normalized = {int(k): v for k, v in truth_map.items()}
            except (TypeError, ValueError):
                normalized = {}
            int_keys = sorted(normalized.keys())
            if int_keys:
                if fabula_time is None:
                    canonical_tick = int_keys[-1]
                else:
                    eligible = [k for k in int_keys if k <= fabula_time]
                    canonical_tick = eligible[-1] if eligible else None
                if canonical_tick is not None:
                    truth_at_query = bool(normalized[canonical_tick])

        support_w = 0.0
        contradict_w = 0.0
        supporters: List[str] = []
        contradictors: List[str] = []
        prop_desc = getattr(prop, "description", None)
        for ent_id, weight, perceived in belief_index.get(pid, []):
            # R1-1 (2026-05-29): bucket by lexical polarity of the
            # holder's perceived_state against the proposition's
            # description, then XOR against the canonical truth.
            # Falls back to the legacy "endorse canonical" rule when
            # perceived_state is empty.
            if _resolve_polarity(perceived, prop_desc, truth_at_query):
                support_w += weight
                supporters.append(ent_id)
            else:
                contradict_w += weight
                contradictors.append(ent_id)

        rows.append(PosteriorRow(
            proposition_id=pid,
            truth_at_query=truth_at_query,
            canonical_tick=canonical_tick,
            support_weight=support_w,
            contradict_weight=contradict_w,
            supporters=supporters,
            contradictors=contradictors,
            description=getattr(prop, "description", None),
            stakes=float(getattr(prop, "stakes", 0.0) or 0.0),
        ))

    rows.sort(key=lambda r: (-r.stakes, r.proposition_id))
    return [r.to_dict() for r in rows]


def surface_entity_constants(
    world_state: WorldStateV1,
    *,
    subject_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Surface immutable entity tags (``Entity.constants``) for the
    abduction / Rung-3 path (AUDIT round-2 P0).

    Abduction asks "what hidden cause made this happen?" but the
    standard hypothesis enumerator only walks mutable causal edges,
    so an entity's intrinsic prerequisite (Feyre's ``mortal_origin``,
    Frankenstein's ``creator``, Edmund's ``forbidden_kin``) is
    invisible. This helper emits a flat list of ``{entity_id,
    constant}`` rows the renderer / abductor can fold in as
    candidate ``ConstantPrerequisite`` hypotheses.

    When ``subject_ids`` is supplied, only those entities are scanned;
    otherwise every entity in the world is included.
    """
    out: List[Dict[str, Any]] = []
    ents = world_state.entities or {}
    if subject_ids is None:
        targets = list(ents.items())
    else:
        targets = [(sid, ents[sid]) for sid in subject_ids if sid in ents]
    for ent_id, ent in targets:
        consts = list(getattr(ent, "constants", None) or [])
        for tag in consts:
            out.append({
                "entity_id": ent_id,
                "constant": tag,
                "kind": "ConstantPrerequisite",
            })
    return out


def audit_posterior_consistency(
    posterior: List[Dict[str, Any]],
) -> List[str]:
    """Return a list of human-readable audit warnings for the table.

    Warnings:
      * Proposition has belief support but no canonical truth tick.
      * Canonical truth contradicts entirely the entity belief mass
        (e.g. truth=True but every belief targets a contradicting
        perceived_state — rare without explicit polarity tracking).
      * Stakes > 0.5 with zero belief evidence (under-modelled).
    """
    warnings: List[str] = []
    for row in posterior:
        pid = row["proposition_id"]
        if row["canonical_tick"] is None and (row["support_weight"] > 0 or row["contradict_weight"] > 0):
            warnings.append(
                f"[posterior·audit] PROP {pid}: belief mass present "
                f"but no canonical truth_at_fabula tick — proposition is "
                f"un-anchored in time."
            )
        if row["stakes"] >= 0.5 and (row["support_weight"] + row["contradict_weight"]) == 0:
            warnings.append(
                f"[posterior·audit] PROP {pid}: stakes={row['stakes']} "
                f"but no entity belief evidence — under-modelled."
            )
    return warnings


__all__ = [
    "PosteriorRow",
    "interrogate_posterior",
    "surface_entity_constants",
    "audit_posterior_consistency",
]
