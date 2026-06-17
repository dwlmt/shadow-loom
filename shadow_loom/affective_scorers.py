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
* **ambivalence** \u2014 mean over entities of the strongest pair of
  co-active opposing concerns (``min`` of the two saliences), capped
  at 1.0.

All six outputs are floats in ``[0.0, 1.0]`` (``surprise`` is
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
    recent_window: Optional[int] = None,
) -> Dict[str, float]:
    if world_state is None:
        return {
            "mystery": 0.0, "irony": 0.0, "suspense": 0.0,
            "surprise": 0.0, "tension": 0.0, "ambivalence": 0.0,
        }

    # R19-L7: pull previously-hardcoded tunables from
    # DirectiveAssemblySettings so deployments can tune
    # recent-window / surprise normalisation / confidence cutoff via
    # env vars (``DIRECTIVE_ASSEMBLY_SCORER_*``) instead of forking
    # this module.
    try:
        from shadow_loom.settings import get_settings as _get_settings
        _da = _get_settings().directive_assembly
        _recent_window = int(recent_window if recent_window is not None
                             else _da.scorer_recent_window)
        _surprise_norm = float(_da.scorer_surprise_flip_norm)
        _min_conf = float(_da.scorer_min_belief_confidence)
    except Exception:
        _recent_window = int(recent_window if recent_window is not None
                             else 2000)
        _surprise_norm = 5.0
        _min_conf = 0.6

    propositions = list(world_state.propositions or [])
    entities = list((world_state.entities or {}).values())

    # R19-H15 / R19-M12: when ``fabula_time`` is supplied, beliefs
    # and concerns must be replayed through their state_timelines so
    # the scorer reads the entity's epistemic state *as of* that
    # moment, not the union of every historical snapshot. Without
    # this, R18-12 belief invalidations and concern activation
    # windows are silently ignored \u2014 closed concerns still
    # raise suspense, retracted beliefs still raise dramatic irony.
    from shadow_loom.models import (
        reconstruct_entity_at as _reconstruct_entity_at,
        reconstruct_concern_at as _reconstruct_concern_at,
    )

    def _effective_beliefs(ent):
        if fabula_time is None:
            return list(getattr(ent, "beliefs", None) or [])
        try:
            reconstructed = _reconstruct_entity_at(ent, int(fabula_time))
        except Exception:
            return list(getattr(ent, "beliefs", None) or [])
        # Wrap each dict so callers can still use attribute access
        # via ``getattr(b, 'confidence', ...)`` / ``b.get(...)``.
        return list(reconstructed.get("beliefs", []) or [])

    def _effective_concerns(ent):
        """Return active concerns at ``fabula_time`` with replayed salience.

        Inactive concerns (closed activation window) are dropped so
        suspense / ambivalence don't accumulate signal from concerns
        the world has already resolved.
        """
        raw = list(getattr(ent, "concerns", None) or [])
        if fabula_time is None:
            return raw
        out = []
        ft = int(fabula_time)
        for c in raw:
            try:
                replayed = _reconstruct_concern_at(c, ft)
            except Exception:
                out.append(c)
                continue
            if not replayed.get("active", True):
                continue
            # Build a lightweight view object so existing
            # ``getattr(c, "salience", ...)`` / ``counter_concern_ids``
            # access keeps working without rewriting the loops below.
            class _ConcernView:
                pass
            view = _ConcernView()
            view.concern_id = getattr(c, "concern_id", None)
            view.proposition_id = getattr(c, "proposition_id", None)
            view.salience = replayed.get("salience", getattr(c, "salience", 0.0))
            view.polarity = replayed.get("polarity", getattr(c, "polarity", None))
            view.counter_concern_ids = replayed.get(
                "counter_concern_ids",
                getattr(c, "counter_concern_ids", None) or [],
            )
            view.kind = replayed.get("kind", getattr(c, "kind", None))
            out.append(view)
        return out

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
        for b in _effective_beliefs(ent):
            # Replayed beliefs are dicts; raw beliefs are Belief models.
            if isinstance(b, dict):
                pid = b.get("proposition_id")
                conf_raw = b.get("confidence", 0.0)
            else:
                pid = getattr(b, "proposition_id", None)
                conf_raw = getattr(b, "confidence", 0.0)
            if not pid or pid not in prop_truth:
                continue
            truth = prop_truth[pid]
            if truth is None:
                continue
            conf = float(conf_raw or 0.0)
            if conf < _min_conf:
                continue
            n_high_conf += 1
            if truth is False:
                n_contradicting += 1
    irony = (n_contradicting / n_high_conf) if n_high_conf else 0.0

    # ---- suspense ----
    open_concerns = []
    for ent in entities:
        for c in _effective_concerns(ent):
            pid = getattr(c, "proposition_id", None)
            # Mirror the irony guard: a concern is "open" only when its
            # proposition exists AND is undecided. A concern pointing at
            # a proposition absent from the world is a dangling ref (the
            # schema auditor flags it), not suspense.
            if pid and pid in prop_truth and prop_truth[pid] is None:
                inten = float(getattr(c, "salience", 0.0) or 0.0)
                open_concerns.append(inten)
    suspense = (sum(open_concerns) / len(open_concerns)) if open_concerns else 0.0

    # ---- surprise ----
    flips = 0
    if fabula_time is not None:
        lo = int(fabula_time) - int(_recent_window)
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
    surprise = min(flips / _surprise_norm, 1.0)

    # ---- tension ----
    # R-2026-06-06: when ``fabula_time`` is supplied, fear must be
    # read *as of* that tick, not at the edge's latest value —
    # otherwise a time-sliced tension score leaks relationship state
    # from later in the story. Mirror the belief / concern replay the
    # other scorers do, via ``reconstruct_relationship_at``.
    _causal_edges = list(world_state.causal_topology or [])
    _events = list(world_state.events or [])
    fears = []
    for re in (world_state.social_topology or []):
        if fabula_time is not None:
            try:
                from shadow_loom.models import (
                    reconstruct_relationship_at as _reconstruct_rel_at,
                )
                rolled = _reconstruct_rel_at(
                    re, int(fabula_time),
                    causal_edges=_causal_edges, events=_events,
                )
                if "fear" not in rolled:
                    continue
                fears.append(abs(float(rolled["fear"])))
                continue
            except Exception:
                pass
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
        concerns = list(_effective_concerns(ent))
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
