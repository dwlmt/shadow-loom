# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Affective-curve evolution regression tests.

The UI's danger / conflict / narrative_tension / power-dynamic
gauges are derived from per-axis ``RelationshipMetric`` values
aggregated over relationship edges (see
``shadow_loom_ui.viz_helpers._compute_affective_scores_uncached``).
Two failure modes silently flatten those curves at fixture-extraction
time, long before any prose is generated, and they apply uniformly
to **every** relationship axis — not just ``fear``:

1. **Observed-but-static axes.** A dyad that declares
   ``RelationshipMetric(observed=True)`` for any axis (``affinity``,
   ``fear``, ``power_dynamic``) but is never touched by any
   ``mutation_social`` causal edge with that ``trait_target``
   contributes the same constant value at every snapshot. If every
   observed axis of a kind in a fixture is static, the corresponding
   gauge curve cannot evolve over fabula time.

2. **Story-with-threat-but-zero-fear-mutations.** A fixture that
   contains threat / violence / intimidation events but routes ALL
   its ``mutation_social`` edges through the ``affinity`` axis
   leaves ``danger`` pinned at the baseline forever, regardless of
   how many ``observed=True`` fear axes exist. (This was the
   original *A Fish Called Wanda* failure: 14 social mutations, all
   on ``affinity``; danger flat at 0.20.) The same pathology can
   appear on ``power_dynamic`` (e.g. a court-intrigue plot whose
   only social mutations target affinity).

These tests lock both invariants at fixture-load time.

Worlds with no antagonistic charge whatsoever (``brief_encounter``,
``persuasion``) are explicitly exempted — flat danger is the correct
reading of their threat arc, not a bug.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import example_worlds
from shadow_loom.models import WorldStateV1


# Stories that are intentionally fear-free — domestic / romance dramas
# whose narrative arc has no physical threat component. A flat danger
# curve at zero is the correct reading for these, not a regression.
NO_FEAR_FIXTURES = {
    "brief_encounter",
    "persuasion",
}


# Fixtures extracted under the prior version of the ingestion prompts
# (before the per-axis ``mutation_social`` requirement was strengthened
# in ``physics_extraction.md``). They contain ``observed=True`` fear
# baselines that are never mutated by any causal edge — the failure
# mode that produces flat danger curves on the UI gauges.
#
# Each entry here is a regression debt scheduled for the next
# re-ingestion pass. New fixtures (or re-ingested versions of these)
# MUST satisfy the per-axis mutation invariant; this set only
# grandfathers the historical extractions so the test gates *new*
# regressions without forcing a bulk rewrite of all 16 fixtures.
#
# To clear an entry: re-extract the world (or hand-author fear-mutation
# edges) so every observed fear axis is touched by at least one
# ``mutation_social`` edge with ``trait_target="fear"``, then remove
# the world name from this set.
KNOWN_FLAT_FEAR_FIXTURES: set[str] = set()


def _all_fixture_worlds() -> list[tuple[str, WorldStateV1]]:
    out: list[tuple[str, WorldStateV1]] = []
    for mod in pkgutil.iter_modules(example_worlds.__path__):
        if mod.name.startswith("_"):
            continue
        m = importlib.import_module(f"example_worlds.{mod.name}")
        ws = getattr(m, "world_state", None)
        if isinstance(ws, WorldStateV1):
            out.append((mod.name, ws))
    return out


WORLDS = _all_fixture_worlds()
WORLD_IDS = [name for name, _ in WORLDS]

# Canonical relationship axes — kept in sync with
# ``shadow_loom.models._REL_METRIC_NAMES``.
RELATIONSHIP_AXES: tuple[str, ...] = ("affinity", "fear", "power_dynamic")


def _observed_static_dyads(
    ws: WorldStateV1, axis: str,
) -> list[tuple[str, str, float]]:
    """Return every (source, target, value) dyad whose ``axis`` is
    declared ``observed=True`` with a non-zero magnitude AND is never
    touched by any ``mutation_social`` edge whose ``trait_target``
    equals ``axis``. These are the dyads that would render flat on
    the corresponding gauge.
    """
    observed: set[tuple[str, str, float]] = set()
    for r in ws.social_topology:
        m = r.metrics.get(axis)
        if m is None or not m.observed:
            continue
        # Magnitude — affinity / power_dynamic are signed in [-1, 1],
        # fear is non-negative in [0, 1]. A measured-zero baseline
        # is a legitimate "no signal" reading and is allowed to stay
        # flat (no contribution to the rising/falling shape anyway).
        if abs(m.value) <= 1e-6:
            continue
        observed.add(
            (r.source_entity_id, r.target_entity_id, m.value),
        )

    mutated: set[tuple[str, str]] = set()
    for c in ws.causal_topology:
        if c.causality_type != "mutation_social":
            continue
        if c.trait_target != axis:
            continue
        if c.target_id and c.rel_counterpart_id:
            mutated.add((c.target_id, c.rel_counterpart_id))

    return [
        (src, tgt, val)
        for (src, tgt, val) in observed
        if (src, tgt) not in mutated
    ]


@pytest.mark.parametrize("name,ws", WORLDS, ids=WORLD_IDS)
def test_observed_fear_axes_have_at_least_one_mutation(name, ws):
    """Every observed fear dyad MUST be touched by at least one
    ``mutation_social`` causal edge with ``trait_target="fear"``.

    A dyad that declares ``observed=True`` for ``fear`` but is never
    mutated produces a constant contribution to the danger mean at
    every snapshot — *visible* on the gauge but *flat* across the
    timeline. This is almost always a fixture-extraction oversight
    (the LLM marked the axis observed but never wove fear-shifting
    events into the causal topology).

    Allowed exceptions:
      * Worlds in ``NO_FEAR_FIXTURES`` (intentionally fear-free).
      * Worlds in ``KNOWN_FLAT_FEAR_FIXTURES`` (extracted under the
        prior prompt; scheduled for re-ingestion).
      * Dyads whose fear baseline is 0.0 — a pre-declared "no fear
        between these two" can legitimately stay flat.
    """
    if name in NO_FEAR_FIXTURES:
        pytest.skip(f"{name!r} is intentionally fear-free")
    if name in KNOWN_FLAT_FEAR_FIXTURES:
        pytest.xfail(
            f"{name!r}: extracted under the prior ingestion prompt; "
            f"scheduled for re-ingestion to add per-axis "
            f"mutation_social fear edges"
        )

    static = _observed_static_dyads(ws, "fear")
    if static:
        report = "\n".join(
            f"  {src} → {tgt}  fear.value={val:.2f} (no mutation_social fear edge)"
            for src, tgt, val in sorted(static)
        )
        pytest.fail(
            f"{name!r}: {len(static)} observed fear axis/axes have a "
            f"non-zero baseline but are never mutated by any "
            f"mutation_social causal edge. The danger gauge will read "
            f"these as constant contributions and the threat arc will "
            f"appear flat across the fabula timeline.\n"
            f"Either (a) add a mutation_social edge with "
            f"trait_target='fear' wired to the relevant threat event, "
            f"(b) drop the baseline to 0.0 (measured-zero), or (c) set "
            f"observed=False to declare this axis genuinely unmeasured.\n"
            f"Static dyads:\n{report}"
        )


# Fixtures grandfathered for the broader "all axes" invariant. These
# extractions predate the per-axis mutation requirement; almost all
# of them have observed affinity / power_dynamic baselines but route
# 100% of their ``mutation_social`` edges through ``affinity``,
# leaving the other axes flat. Bulk re-extraction is out of scope —
# the set narrows as fixtures are re-ingested under the strengthened
# prompts.
#
# Empty by default at the moment of introduction; populated lazily by
# the test runner discovering historical pre-existing static dyads.
# To clear an entry: emit ``mutation_social`` edges for every
# observed axis on the fixture's relationship edges, then remove the
# world name from this set.
KNOWN_FLAT_AXIS_FIXTURES: dict[str, set[str]] = {
    "affinity": set(),
    "power_dynamic": set(),
}


@pytest.mark.parametrize("name,ws", WORLDS, ids=WORLD_IDS)
@pytest.mark.parametrize("axis", ["affinity", "power_dynamic"])
def test_observed_axes_have_at_least_one_mutation(axis, name, ws):
    """Per-axis generalisation of the fear test, applied to the
    remaining two relationship axes (``affinity`` and
    ``power_dynamic``). The same pathology — observed baseline,
    zero mutation edges → flat gauge — afflicts every axis equally.

    Worlds in ``NO_FEAR_FIXTURES`` are still run for ``affinity`` and
    ``power_dynamic`` (a romance with no fear arc still has affinity
    movement). Worlds in ``KNOWN_FLAT_AXIS_FIXTURES[axis]`` are
    grandfathered and ``xfail``ed pending re-ingestion.
    """
    if name in KNOWN_FLAT_AXIS_FIXTURES.get(axis, set()):
        pytest.xfail(
            f"{name!r}: extracted under the prior ingestion prompt; "
            f"scheduled for re-ingestion to add per-axis "
            f"mutation_social {axis} edges"
        )

    static = _observed_static_dyads(ws, axis)
    if static:
        report = "\n".join(
            f"  {src} → {tgt}  {axis}.value={val:+.2f} "
            f"(no mutation_social {axis} edge)"
            for src, tgt, val in sorted(static)
        )
        pytest.fail(
            f"{name!r}: {len(static)} observed {axis} axis/axes have "
            f"a non-zero baseline but are never mutated by any "
            f"mutation_social causal edge. The corresponding gauge "
            f"will read these as constant contributions and the "
            f"{axis} arc will appear flat across the fabula timeline.\n"
            f"Either (a) add a mutation_social edge with "
            f"trait_target={axis!r} wired to the relevant event, "
            f"(b) drop the baseline to 0.0 (measured-zero), or (c) "
            f"set observed=False to declare this axis genuinely "
            f"unmeasured.\nStatic dyads:\n{report}"
        )


@pytest.mark.parametrize("name,ws", WORLDS, ids=WORLD_IDS)
def test_threat_stories_have_at_least_one_fear_mutation(name, ws):
    """Stories that contain physical threat events MUST also contain
    at least one ``mutation_social`` edge with ``trait_target="fear"``.

    The original *A Fish Called Wanda* regression: the fixture had
    Otto torturing Ken with the fish, shooting the cupboard Archie was
    hiding in, and being run over by a steamroller — but every one of
    its 14 ``mutation_social`` edges targeted ``affinity``, so the
    danger curve was a flat line at the baseline. This test prevents
    a future fixture from making the same mistake by checking the
    *causal topology balance*: if any social mutation exists at all,
    fear (and ideally power_dynamic) should be represented.
    """
    if name in NO_FEAR_FIXTURES:
        pytest.skip(f"{name!r} is intentionally fear-free")

    social_mutations = [
        c for c in ws.causal_topology
        if c.causality_type == "mutation_social"
    ]
    if not social_mutations:
        pytest.skip(f"{name!r} has no mutation_social edges at all")

    by_axis: dict[str, int] = {}
    for c in social_mutations:
        if c.trait_target:
            by_axis[c.trait_target] = by_axis.get(c.trait_target, 0) + 1

    if "fear" not in by_axis:
        pytest.fail(
            f"{name!r}: the fixture has {len(social_mutations)} "
            f"mutation_social edges but NONE target the 'fear' axis "
            f"(by_axis={by_axis}). The danger gauge cannot evolve over "
            f"fabula time. If the story genuinely contains no threat "
            f"dynamics, add the world name to NO_FEAR_FIXTURES; "
            f"otherwise add at least one mutation_social edge with "
            f"trait_target='fear' wired to the relevant threat event."
        )
