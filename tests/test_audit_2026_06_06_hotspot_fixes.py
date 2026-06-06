# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-06-06 hotspot-audit fixes.

Three independent findings from the deeper hotspot pass:

  1. ``_post_pass_dedup_near_duplicate_events`` rewrote loser->keeper ids in
     causal_topology, proposition.referent_ids, belief.acquired_via_event_id
     and the various ``triggered_by`` fields, but NOT ``EventNode.target_ids``
     (which can hold EVT_ utterance referents) nor ``Belief.target_id`` (which
     can be an EVT_ id). Collapsing a duplicate event therefore left dangling
     references to the now-deleted loser id.

  2. ``build_intervention_brief`` built its InterventionMechanism block by
     iterating the legacy ``query.interventions`` dotted-key dict. A typed-only
     ``InterventionQuery(do_targets=[...])`` (interventions={}) — the shape MCP
     and fixtures author — produced an empty mechanism block, so the renderer
     and auditor never saw the old_state->new_state + inertia struggle for any
     typed surgery. The ``_backfill_typed_do_targets`` validator only lifts
     legacy->typed, never the reverse.

  3. ``RelationshipMetric.inertia`` was unconstrained while its siblings
     (TraitVector.inertia / Belief.inertia) are bounded [0,1]; an out-of-range
     value would silently disable or always-fire the inertia gate.
"""
from __future__ import annotations

import importlib

import pydantic
import pytest

from shadow_loom.models import (
    Belief,
    Entity,
    EventNode,
    RelationshipMetric,
    WorldStateV1,
)
from shadow_loom.settings import CausalPhysicsSettings
from shadow_loom.ingestion import _post_pass_dedup_near_duplicate_events
from shadow_loom.generation import build_intervention_brief, _do_target_to_dotted_kv
from shadow_loom.query_models import (
    InterventionQuery,
    DoTrait,
    DoRelationship,
    DoEvent,
    DoProposition,
    DoWorldTrait,
)


# ---------------------------------------------------------------------------
# 1. Event-dedup must remap EventNode.target_ids and Belief.target_id.
# ---------------------------------------------------------------------------
class TestDedupRemapsEvtReferences:

    @staticmethod
    def _world_with_duplicate_referenced_by_evt_and_belief() -> WorldStateV1:
        keeper = EventNode(
            id="EVT_KEEP",
            description="the longer more informative description of the deed",
            event_type="outcome", fabula_time=100, syuzhet_index=100,
            actor_ids=["ENT_A"], target_ids=[],
        )
        loser = EventNode(
            id="EVT_LOSE", description="short",
            event_type="outcome", fabula_time=100, syuzhet_index=101,
            actor_ids=["ENT_A"], target_ids=[],
        )
        # A surviving utterance that references the loser via target_ids.
        referer = EventNode(
            id="EVT_UTT", description="someone speaks about the deed",
            event_type="utterance", fabula_time=200, syuzhet_index=200,
            speaker_id="ENT_A", addressee_ids=["ENT_B"], at_location_id="LOC_X",
            target_ids=["EVT_LOSE"],
        )
        ent_a = Entity(
            name="A", location_id="LOC_X", status="healthy", traits={},
            beliefs=[Belief(
                target_id="EVT_LOSE", perceived_state="knows of the deed",
                confidence=0.9, inertia=0.5,
            )],
        )
        ent_b = Entity(name="B", location_id="LOC_X", status="healthy", traits={})
        return WorldStateV1(
            locations={}, objects={},
            entities={"ENT_A": ent_a, "ENT_B": ent_b},
            events=[keeper, loser, referer], causal_topology=[],
        )

    def test_evt_target_ids_remapped_to_keeper(self):
        out = _post_pass_dedup_near_duplicate_events(
            self._world_with_duplicate_referenced_by_evt_and_belief(), [],
        )
        utt = next(e for e in out.events if e.id == "EVT_UTT")
        assert utt.target_ids == ["EVT_KEEP"], (
            "EVT_UTT.target_ids still points at the collapsed loser id"
        )

    def test_belief_target_id_remapped_to_keeper(self):
        out = _post_pass_dedup_near_duplicate_events(
            self._world_with_duplicate_referenced_by_evt_and_belief(), [],
        )
        assert out.entities["ENT_A"].beliefs[0].target_id == "EVT_KEEP", (
            "Belief.target_id still points at the collapsed loser id"
        )

    def test_loser_event_is_actually_collapsed(self):
        out = _post_pass_dedup_near_duplicate_events(
            self._world_with_duplicate_referenced_by_evt_and_belief(), [],
        )
        ids = {e.id for e in out.events}
        assert "EVT_LOSE" not in ids and "EVT_KEEP" in ids


# ---------------------------------------------------------------------------
# 2. Intervention brief mechanism block must cover typed-only do_targets.
# ---------------------------------------------------------------------------
class TestInterventionBriefTypedMechanisms:

    def test_reverse_mapper_covers_clean_kinds(self):
        assert _do_target_to_dotted_kv(
            DoTrait(holder_id="ENT_A", trait_name="ambition", value=0.9)
        ) == ("ENT_A.traits.ambition", 0.9)
        assert _do_target_to_dotted_kv(
            DoRelationship(source_entity_id="ENT_A", target_entity_id="ENT_B",
                           metric="affinity", value=-0.5)
        ) == ("ENT_A.relationships.ENT_B.affinity", -0.5)
        assert _do_target_to_dotted_kv(
            DoEvent(event_id="EVT_X", occurred=False)
        ) == ("EVT_X.event_type", "averted")
        assert _do_target_to_dotted_kv(
            DoProposition(proposition_id="PROP_X", truth=True)
        ) == ("PROP_X.truth", True)
        assert _do_target_to_dotted_kv(
            DoWorldTrait(world_trait_id="WORLD_W", value=0.3)
        ) == ("WORLD_W.value", 0.3)

    def test_typed_only_trait_query_produces_mechanism(self):
        ws = importlib.import_module("example_worlds.macbeth").world_state
        ent = ws.entities["ENT_MACBETH"]
        trait_name = next(iter(ent.traits.keys()))
        q = InterventionQuery(
            do_targets=[DoTrait(holder_id="ENT_MACBETH",
                                trait_name=trait_name, value=0.95)],
            original_query="raise ambition",
        )
        assert q.interventions == {}, "fixture must exercise the typed-only path"
        brief = build_intervention_brief(q, {}, ws)
        nodes = {(m.node_id, m.new_state) for m in brief.intervention_mechanisms}
        assert ("ENT_MACBETH", "0.95") in nodes, (
            "typed-only DoTrait produced no InterventionMechanism block"
        )

    def test_legacy_dict_path_still_produces_mechanism(self):
        ws = importlib.import_module("example_worlds.macbeth").world_state
        ent = ws.entities["ENT_MACBETH"]
        trait_name = next(iter(ent.traits.keys()))
        # Legacy dotted-key shape; validator backfills do_targets but keeps
        # the dict populated, so the mechanism source stays the legacy dict.
        q = InterventionQuery(
            interventions={f"ENT_MACBETH.traits.{trait_name}": 0.95},
            original_query="raise ambition",
        )
        brief = build_intervention_brief(q, {}, ws)
        assert any(
            m.node_id == "ENT_MACBETH" for m in brief.intervention_mechanisms
        )


# ---------------------------------------------------------------------------
# 3. RelationshipMetric.inertia must be bounded [0, 1].
# ---------------------------------------------------------------------------
class TestRelationshipMetricInertiaBounded:

    def test_rejects_out_of_range_inertia(self):
        with pytest.raises(pydantic.ValidationError):
            RelationshipMetric(value=0.2, inertia=5.0)
        with pytest.raises(pydantic.ValidationError):
            RelationshipMetric(value=0.2, inertia=-0.1)

    def test_accepts_in_range_inertia(self):
        assert RelationshipMetric(value=0.2, inertia=0.3).inertia == 0.3
        assert RelationshipMetric(value=0.2, inertia=0.0).inertia == 0.0
        assert RelationshipMetric(value=0.2, inertia=1.0).inertia == 1.0


# ---------------------------------------------------------------------------
# 4. Monte-Carlo is on by default but SEEDED, so results are stable.
# ---------------------------------------------------------------------------
class TestMonteCarloStableByDefault:
    """Monte-Carlo stays on by default (samples=24) but ``monte_carlo_seed``
    defaults to a fixed int so engine distributions — and the plausibility /
    vacuity verdicts derived from them — are reproducible run-to-run rather
    than flipping near a decision boundary."""

    def test_seed_default_is_fixed_int(self):
        s = CausalPhysicsSettings()
        assert s.monte_carlo_samples > 0, "MC must stay on by default"
        assert isinstance(s.monte_carlo_seed, int), (
            "monte_carlo_seed must default to a fixed int so MC is stable; "
            "a None default reintroduces run-to-run verdict flipping"
        )

    def test_engine_distributions_are_reproducible(self):
        # Lazy import: the helper lives in the causal-physics test module.
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from test_causal_physics import _make_minimal_world, _build_sandbox
        from shadow_loom.causal_physics import CausalPhysicsEngine
        from shadow_loom.settings import get_settings

        # The suite-wide autouse fixture zeroes monte_carlo_samples for
        # determinism; re-enable the shipped defaults (samples=24, seed=0)
        # here so we actually exercise the seeded Monte-Carlo path. The
        # fixture restores the prior value on teardown.
        physics = get_settings().physics
        physics.monte_carlo_samples = 24
        physics.monte_carlo_seed = 0

        def _run():
            ws = _make_minimal_world()
            sandbox = _build_sandbox(ws, ["ENT_ALICE", "ENT_BOB"])
            engine = CausalPhysicsEngine(sandbox, ws)
            r = engine.execute(rung=2, interventions={"ENT_ALICE.anger": 0.95})
            return {
                node: {t: (d.mean, d.p5, d.p50, d.p95) for t, d in traits.items()}
                for node, traits in r.trait_distributions.items()
            }

        first, second = _run(), _run()
        assert first, "expected a Monte-Carlo trait distribution (MC on by default)"
        assert first == second, (
            "Monte-Carlo results differ across identical runs — the default "
            "seed is not stabilising the sampler"
        )
