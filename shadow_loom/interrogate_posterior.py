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
    belief_index: Dict[str, List[tuple[str, float, bool]]] = {}
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
            # Polarity convention: a belief is "supporting" the
            # proposition's canonical truth. If the canonical truth at
            # query is False, the same belief becomes "contradicting".
            # We resolve polarity per-proposition below.
            belief_index.setdefault(pid, []).append((ent_id, weight, True))

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
        for ent_id, weight, _polarity in belief_index.get(pid, []):
            # A belief endorses the proposition's perceived_state.
            # Without explicit polarity on Belief, we treat the
            # belief as endorsing whatever the proposition's
            # canonical truth currently is — flipping the
            # contributor bucket if the canonical truth is False.
            if truth_at_query is True or truth_at_query is None:
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
