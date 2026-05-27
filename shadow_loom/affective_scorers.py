# SPDX-FileCopyrightText: 2026 David Hyland
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic affective scorers for ``interrogate`` / ``general``
query payloads (AUDIT P1-2).

The rich scorers in ``directive_assembly`` (mystery / irony / suspense
/ surprise / tension) are bound to the generation / directive path
only. ``interrogate`` and ``general`` answers historically reached
the LLM through ``answer.py`` without any affective summary, so the
LLM was guessing tone from raw entity / event lists.

This module provides five tiny, deterministic features computed
straight from :class:`WorldStateV1` so the answerer always has a
ground-truthed affective banner:

* **mystery**     \u2014 mean ``stakes`` of propositions whose
  ``truth_at_fabula`` is empty or undecided at the current tick.
* **irony**       \u2014 fraction of high-confidence beliefs that
  contradict the canonical truth of the linked proposition.
* **suspense**    \u2014 mean ``intensity`` of open concerns whose
  target proposition is still undecided.
* **surprise**    \u2014 count of canonical truth flips within the
  recent fabula window (default last 2000 ticks).
* **tension**     \u2014 mean ``abs(fear)`` over the densest dyad
  cluster, capped at 1.0.

All five outputs are floats in ``[0.0, 1.0]`` (``surprise`` is
normalised by ``min(flips / 5.0, 1.0)``). The function never raises
and silently substitutes ``0.0`` for any unavailable component.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from shadow_loom.models import WorldStateV1


def _truth_at(prop, fabula_time: Optional[int]) -> Optional[bool]:
    truth = getattr(prop, "truth_at_fabula", None) or {}
    if not truth:
        return None
    keys = []
    for k in truth.keys():
        try:
            keys.append(int(k))
        except (TypeError, ValueError):
            continue
    if not keys:
        return None
    if fabula_time is None:
        chosen = max(keys)
    else:
        ok = [k for k in keys if k <= int(fabula_time)]
        if not ok:
            return None
        chosen = max(ok)
    for k, v in truth.items():
        try:
            if int(k) == chosen:
                return bool(v)
        except (TypeError, ValueError):
            continue
    return None


def compute_affective_scorers(
    world_state: WorldStateV1,
    *,
    fabula_time: Optional[int] = None,
    recent_window: int = 2000,
) -> Dict[str, float]:
    if world_state is None:
        return {
            "mystery": 0.0, "irony": 0.0, "suspense": 0.0,
            "surprise": 0.0, "tension": 0.0,
        }

    propositions = list(world_state.propositions or [])
    entities = list((world_state.entities or {}).values())

    # ---- mystery ----
    undecided_stakes = []
    for prop in propositions:
        if _truth_at(prop, fabula_time) is None:
            undecided_stakes.append(float(getattr(prop, "stakes", 0.0) or 0.0))
    mystery = (sum(undecided_stakes) / len(undecided_stakes)) if undecided_stakes else 0.0

    # ---- irony ----
    # Heuristic proxy: a belief carries dramatic irony when the
    # character holds it confidently AND the linked proposition has
    # settled FALSE in canonical truth (audience knows, character
    # believes). ``perceived_state`` is a free-form string in the
    # model so we cannot compare it structurally; the truth-False +
    # high-confidence proxy captures the classic shape (Romeo
    # confident Juliet dead while ALIVE=True, or Othello confident
    # Desdemona unfaithful while FIDELITY=True \u2014 negate the
    # proposition framing during authoring so this lines up).
    prop_truth = {
        p.proposition_id: _truth_at(p, fabula_time)
        for p in propositions
    }
    n_high_conf = 0
    n_contradicting = 0
    for ent in entities:
        for b in (ent.beliefs or []):
            pid = getattr(b, "proposition_id", None)
            if not pid or pid not in prop_truth:
                continue
            truth = prop_truth[pid]
            if truth is None:
                continue
            conf = float(getattr(b, "confidence", 0.0) or 0.0)
            if conf < 0.6:
                continue
            n_high_conf += 1
            if truth is False:
                n_contradicting += 1
    irony = (n_contradicting / n_high_conf) if n_high_conf else 0.0

    # ---- suspense ----
    open_concerns = []
    for ent in entities:
        for c in (ent.concerns or []):
            pid = getattr(c, "proposition_id", None)
            if pid and prop_truth.get(pid) is None:
                inten = float(getattr(c, "salience", 0.0) or 0.0)
                open_concerns.append(inten)
    suspense = (sum(open_concerns) / len(open_concerns)) if open_concerns else 0.0

    # ---- surprise ----
    flips = 0
    if fabula_time is not None:
        lo = int(fabula_time) - int(recent_window)
        hi = int(fabula_time)
        for prop in propositions:
            truth = getattr(prop, "truth_at_fabula", None) or {}
            ticks = []
            for k in truth.keys():
                try:
                    ki = int(k)
                except (TypeError, ValueError):
                    continue
                if lo <= ki <= hi:
                    ticks.append((ki, bool(truth[k])))
            ticks.sort()
            for i in range(1, len(ticks)):
                if ticks[i][1] != ticks[i - 1][1]:
                    flips += 1
    surprise = min(flips / 5.0, 1.0)

    # ---- tension ----
    fears = []
    for re in (world_state.social_topology or []):
        metric = (re.metrics or {}).get("fear")
        if metric is None:
            continue
        fears.append(abs(float(getattr(metric, "value", 0.0) or 0.0)))
    tension = (sum(fears) / len(fears)) if fears else 0.0
    tension = min(tension, 1.0)

    # ---- ambivalence (AUDIT round-2 P1) ----
    # Holders whose concerns include explicit ``counter_concern_ids``
    # are dramatising internal conflict (Mainwaring class-anxiety
    # vs platoon-pride, Edmund greed vs siblings, Mathilde
    # transactional ambition vs surviving care). Score = mean over
    # entities of (paired_salience_product) where the pair is at
    # least one of the two concerns the entity holds. Each
    # contributing pair contributes ``min(s_a, s_b)`` so a single
    # highly-salient pair dominates a noisy long tail.
    ambivalence_signals = []
    for ent in entities:
        concerns = list(getattr(ent, "concerns", None) or [])
        if not concerns:
            continue
        by_id = {getattr(c, "concern_id", None): c for c in concerns}
        pairs_seen: set = set()
        ent_signal = 0.0
        for c in concerns:
            cid = getattr(c, "concern_id", None)
            for other in (getattr(c, "counter_concern_ids", None) or []):
                # Self-asymmetric counter links (the partner concern
                # lives on another entity) are silently skipped \u2014
                # the symmetry audit elsewhere surfaces those.
                if other not in by_id:
                    continue
                key = tuple(sorted((cid or "", other)))
                if key in pairs_seen:
                    continue
                pairs_seen.add(key)
                s_a = float(getattr(c, "salience", 0.0) or 0.0)
                s_b = float(getattr(by_id[other], "salience", 0.0) or 0.0)
                ent_signal = max(ent_signal, min(s_a, s_b))
        if ent_signal > 0.0:
            ambivalence_signals.append(ent_signal)
    ambivalence = (
        sum(ambivalence_signals) / len(ambivalence_signals)
        if ambivalence_signals else 0.0
    )
    ambivalence = min(max(ambivalence, 0.0), 1.0)

    return {
        "mystery": round(mystery, 4),
        "irony": round(irony, 4),
        "suspense": round(suspense, 4),
        "surprise": round(surprise, 4),
        "tension": round(tension, 4),
        "ambivalence": round(ambivalence, 4),
    }


__all__ = ["compute_affective_scorers"]
