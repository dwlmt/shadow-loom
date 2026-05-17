# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 0 — DoTarget discriminated union tests.

Covers construction, discriminator dispatch, round-trip serialisation,
and the legacy intervention-dict migration adapter.
"""
import pytest
from pydantic import ValidationError

from shadow_loom.query_models import (
    DoEvent,
    DoProposition,
    DoBelief,
    DoConcern,
    DoTrait,
    InterventionQuery,
    CounterfactualQuery,
)


class TestDoTargetConstruction:
    def test_do_event(self):
        t = DoEvent(event_id="EVT_DUNCAN_MURDER", occurred=False)
        assert t.target_kind == "event"
        assert t.event_id == "EVT_DUNCAN_MURDER"
        assert t.occurred is False

    def test_do_proposition(self):
        t = DoProposition(
            proposition_id="PROP_BANQUO_LINE_KINGS",
            truth=True,
            fabula_time=10000,
            propagate_to_beliefs=True,
        )
        assert t.target_kind == "proposition"
        assert t.truth is True
        assert t.propagate_to_beliefs is True

    def test_do_belief(self):
        t = DoBelief(
            holder_id="ENT_MACDUFF",
            target_id="ENT_MACBETH",
            confidence=0.0,
            proposition_id="PROP_MACBETH_FOUL_PLAY",
        )
        assert t.target_kind == "belief"
        assert t.confidence == 0.0
        assert 0.0 <= t.confidence <= 1.0

    def test_do_belief_confidence_clamped(self):
        with pytest.raises(ValidationError):
            DoBelief(holder_id="ENT_X", target_id="ENT_Y", confidence=1.5)

    def test_do_concern(self):
        t = DoConcern(
            holder_id="ENT_LADY_MACBETH",
            concern_id="CCN_LADY_DESIRES_CROWN",
            salience=0.0,
            active=False,
        )
        assert t.target_kind == "concern"
        assert t.salience == 0.0
        assert t.active is False
        # polarity left unset → None
        assert t.polarity is None

    def test_do_trait(self):
        t = DoTrait(holder_id="ENT_MACBETH", trait_name="ambition", value=0.0)
        assert t.target_kind == "trait"
        assert t.trait_name == "ambition"


class TestDiscriminatedDispatch:
    def test_intervention_query_accepts_mixed_targets(self):
        q = InterventionQuery(
            do_targets=[
                DoEvent(event_id="EVT_X", occurred=False),
                DoProposition(proposition_id="PROP_Y", truth=True),
                DoBelief(holder_id="ENT_A", target_id="ENT_B"),
                DoConcern(holder_id="ENT_A", concern_id="CCN_Z"),
                DoTrait(holder_id="ENT_A", trait_name="ambition", value=0.5),
            ],
            original_query="mixed",
        )
        assert [t.target_kind for t in q.do_targets] == [
            "event", "proposition", "belief", "concern", "trait",
        ]

    def test_intervention_query_default_lists_empty(self):
        q = InterventionQuery(original_query="empty")
        assert q.do_targets == []
        assert q.interventions == {}

    def test_counterfactual_query_accepts_mixed_targets(self):
        q = CounterfactualQuery(
            historical_do_targets=[
                DoBelief(holder_id="ENT_ROMEO", target_id="ENT_JULIET",
                         perceived_state="alive", confidence=1.0),
            ],
            evidence_node_ids=["EVT_TOMB_SCENE"],
            original_query="cf",
        )
        assert q.historical_do_targets[0].target_kind == "belief"


class TestRoundTrip:
    @pytest.mark.parametrize("target", [
        DoEvent(event_id="EVT_A", occurred=False),
        DoProposition(proposition_id="PROP_A", truth=True),
        DoBelief(holder_id="ENT_A", target_id="ENT_B", confidence=0.3),
        DoConcern(holder_id="ENT_A", concern_id="CCN_A", polarity="fear", salience=0.7),
        DoTrait(holder_id="ENT_A", trait_name="ambition", value=0.4, inertia=0.8),
    ])
    def test_each_variant_round_trips(self, target):
        q = InterventionQuery(do_targets=[target], original_query="rt")
        restored = InterventionQuery.model_validate_json(q.model_dump_json())
        assert restored.do_targets[0].target_kind == target.target_kind
        assert restored.do_targets[0].model_dump() == target.model_dump()

    def test_discriminator_picks_correct_variant_from_dict(self):
        q = InterventionQuery.model_validate({
            "original_query": "x",
            "do_targets": [
                {"target_kind": "concern", "holder_id": "ENT_A",
                 "concern_id": "CCN_A", "polarity": "desire"},
            ],
        })
        assert isinstance(q.do_targets[0], DoConcern)
        assert q.do_targets[0].polarity == "desire"


class TestLegacyMigration:
    def test_intervention_legacy_dict_lifts_to_do_events(self):
        # The model_validator on InterventionQuery auto-coerces the
        # legacy ``interventions`` dict into typed ``do_targets`` at
        # construction time — no explicit migration call required.
        q = InterventionQuery(
            interventions={"EVT_FOO": "averted", "EVT_BAR": True, "EVT_BAZ": False},
            original_query="legacy",
        )
        kinds = {(t.event_id, t.occurred) for t in q.do_targets}
        assert kinds == {("EVT_FOO", False), ("EVT_BAR", True), ("EVT_BAZ", False)}

    def test_intervention_typed_targets_take_priority(self):
        existing = DoEvent(event_id="EVT_X", occurred=False)
        q = InterventionQuery(
            do_targets=[existing],
            interventions={"EVT_OTHER": "averted"},
            original_query="both",
        )
        # Typed list already populated → validator is a no-op.
        assert len(q.do_targets) == 1
        assert q.do_targets[0].event_id == "EVT_X"

    def test_counterfactual_legacy_dict_lifts(self):
        q = CounterfactualQuery(
            historical_interventions={"EVT_GUARD_DUTY": "averted"},
            evidence_node_ids=["EVT_DUNCAN_FOUND"],
            original_query="legacy cf",
        )
        assert len(q.historical_do_targets) == 1
        assert q.historical_do_targets[0].event_id == "EVT_GUARD_DUTY"
        assert q.historical_do_targets[0].occurred is False

    def test_migration_idempotent(self):
        # Construct once, then re-validate the dumped JSON; the typed
        # list must remain stable across the round-trip.
        q = InterventionQuery(interventions={"EVT_A": "averted"}, original_query="x")
        first = list(q.do_targets)
        restored = InterventionQuery.model_validate_json(q.model_dump_json())
        assert restored.do_targets == first

    def test_empty_legacy_dict_yields_empty_targets(self):
        q = InterventionQuery(original_query="x")
        assert q.do_targets == []
