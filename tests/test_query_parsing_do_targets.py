# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 6 — query parser DoTarget surface.

Covers the deterministic parts of the parser do-target wiring:

  * ``_collect_typed_ids`` exposes ``proposition_ids`` and ``concern_ids``.
  * ``_do_target_items_to_typed`` lifts flat dict records into the typed
    :class:`DoTarget` discriminated union, dropping malformed entries.
  * ``_build_query`` (the ParsedQuery → UserRequest
    converter) populates ``InterventionQuery.do_targets`` and
    ``CounterfactualQuery.historical_do_targets`` when the parser
    emitted Phase-6 ``do_targets`` items.

The dynamic-schema construction itself depends on a live LLM run so is
not exercised here; it's covered by the existing ``test_query_parsing``
agent integration tests.
"""
import pytest

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, Belief, Concern, Proposition,
    TraitVector, CausalEdge,
)
from shadow_loom.query_models import (
    DoEvent, DoTrait, DoBelief, DoConcern, DoProposition,
    InterventionQuery, CounterfactualQuery,
)
from shadow_loom.query_parsing import (
    ParsedQuery,
    _collect_typed_ids,
    _do_target_items_to_typed,
    _build_do_target_item_model,
    _build_query,
)


def _make_world() -> WorldStateV1:
    return WorldStateV1(
        locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
        objects={},
        entities={
            "ENT_MACBETH": Entity(
                id="ENT_MACBETH", name="Macbeth", location_id="LOC_A",
                status="healthy",
                traits={"ambition": TraitVector(value=0.6, inertia=0.2)},
                concerns=[
                    Concern(
                        concern_id="CCN_LADY_DESIRES_CROWN",
                        proposition_id="PROP_BANQUO_LINE_KINGS",
                        polarity="desire", kind="ambition",
                        salience=0.9,
                    ),
                ],
            ),
            "ENT_MACDUFF": Entity(
                id="ENT_MACDUFF", name="Macduff", location_id="LOC_A",
                status="healthy",
                traits={},
                beliefs=[
                    Belief(
                        target_id="ENT_MACBETH",
                        perceived_state="Macbeth's grief is feigned",
                        confidence=0.7,
                        evidence_strength="moderate",
                        proposition_id="PROP_MACBETH_FOUL_PLAY",
                        inertia=0.2,
                    ),
                ],
            ),
        },
        events=[
            EventNode(id="EVT_MURDER", fabula_time=10, syuzhet_index=10,
                      event_type="choice", actor_ids=["ENT_MACBETH"],
                      target_ids=["ENT_MACDUFF"],
                      description="The murder."),
        ],
        causal_topology=[],
        spatial_topology=[],
        social_topology=[],
        propositions=[
            Proposition(
                proposition_id="PROP_BANQUO_LINE_KINGS",
                description="Banquo's line shall be kings.",
                kind="event_occurs",
                referent_ids=["ENT_MACBETH"],
                truth_at_fabula={1: True},
            ),
            Proposition(
                proposition_id="PROP_MACBETH_FOUL_PLAY",
                description="Macbeth committed foul play.",
                kind="event_occurs",
                referent_ids=["ENT_MACBETH"],
                truth_at_fabula={10: True},
            ),
        ],
    )


class TestCollectTypedIds:
    def test_proposition_ids_collected(self):
        ws = _make_world()
        typed = _collect_typed_ids(ws)
        assert "PROP_BANQUO_LINE_KINGS" in typed["proposition_ids"]
        assert "PROP_MACBETH_FOUL_PLAY" in typed["proposition_ids"]

    def test_concern_ids_collected_across_entities(self):
        ws = _make_world()
        typed = _collect_typed_ids(ws)
        assert "CCN_LADY_DESIRES_CROWN" in typed["concern_ids"]


class TestDoTargetItemModel:
    def test_model_builds_with_world(self):
        ws = _make_world()
        Model = _build_do_target_item_model(ws)
        # Build one of each kind to confirm the schema accepts all five.
        Model(target_kind="event", event_id="EVT_MURDER", occurred=False)
        Model(target_kind="trait", entity_id="ENT_MACBETH",
              trait_name="ambition", trait_value=0.0)
        Model(target_kind="belief", holder_id="ENT_MACDUFF",
              target_id="ENT_MACBETH",
              proposition_id="PROP_MACBETH_FOUL_PLAY", confidence=0.0)
        Model(target_kind="concern", holder_id="ENT_MACBETH",
              concern_id="CCN_LADY_DESIRES_CROWN",
              salience=0.0)
        Model(target_kind="proposition",
              proposition_id="PROP_BANQUO_LINE_KINGS", truth=True)


class TestDoTargetItemsToTyped:
    def test_lift_event(self):
        items = [{"target_kind": "event", "event_id": "EVT_MURDER",
                  "occurred": False}]
        result = _do_target_items_to_typed(items)
        assert len(result) == 1
        assert isinstance(result[0], DoEvent)
        assert result[0].event_id == "EVT_MURDER"
        assert result[0].occurred is False

    def test_lift_trait(self):
        items = [{"target_kind": "trait", "entity_id": "ENT_MACBETH",
                  "trait_name": "ambition", "trait_value": 0.0}]
        result = _do_target_items_to_typed(items)
        assert isinstance(result[0], DoTrait)
        assert result[0].value == 0.0

    def test_lift_belief(self):
        items = [{"target_kind": "belief", "holder_id": "ENT_MACDUFF",
                  "target_id": "ENT_MACBETH",
                  "proposition_id": "PROP_MACBETH_FOUL_PLAY",
                  "confidence": 0.0}]
        result = _do_target_items_to_typed(items)
        assert isinstance(result[0], DoBelief)
        assert result[0].confidence == 0.0

    def test_lift_concern(self):
        items = [{"target_kind": "concern",
                  "holder_id": "ENT_MACBETH",
                  "concern_id": "CCN_LADY_DESIRES_CROWN",
                  "salience": 0.0}]
        result = _do_target_items_to_typed(items)
        assert isinstance(result[0], DoConcern)
        assert result[0].salience == 0.0

    def test_lift_proposition(self):
        items = [{"target_kind": "proposition",
                  "proposition_id": "PROP_BANQUO_LINE_KINGS",
                  "truth": True, "propagate_to_beliefs": False}]
        result = _do_target_items_to_typed(items)
        assert isinstance(result[0], DoProposition)
        assert result[0].truth is True
        assert result[0].propagate_to_beliefs is False

    def test_drops_items_missing_required_fields(self):
        # Belief without proposition_id → dropped.
        items = [
            {"target_kind": "belief", "holder_id": "ENT_MACDUFF",
             "target_id": "ENT_MACBETH"},
            {"target_kind": "event", "event_id": "EVT_MURDER",
             "occurred": True},
        ]
        result = _do_target_items_to_typed(items)
        assert len(result) == 1
        assert isinstance(result[0], DoEvent)

    def test_drops_unknown_kinds(self):
        items = [{"target_kind": "spaceship", "name": "USS Enterprise"}]
        assert _do_target_items_to_typed(items) == []

    def test_empty_input(self):
        assert _do_target_items_to_typed([]) == []
        assert _do_target_items_to_typed(None) == []

    def test_mixed_kinds(self):
        items = [
            {"target_kind": "proposition",
             "proposition_id": "PROP_BANQUO_LINE_KINGS", "truth": True},
            {"target_kind": "concern",
             "holder_id": "ENT_MACBETH",
             "concern_id": "CCN_LADY_DESIRES_CROWN", "salience": 0.0},
        ]
        result = _do_target_items_to_typed(items)
        assert len(result) == 2
        kinds = {r.target_kind for r in result}
        assert kinds == {"proposition", "concern"}


class TestParsedQueryToTypedDoTargets:
    def test_intervention_with_proposition_do_target(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="Suppose Banquo's line really would inherit.",
            interventions={"ENT_MACBETH.status": "healthy"},
            do_targets=[{"target_kind": "proposition",
                         "proposition_id": "PROP_BANQUO_LINE_KINGS",
                         "truth": True}],
        )
        req = _build_query(parsed, "Suppose Banquo's line inherits.")
        assert isinstance(req, InterventionQuery)
        assert len(req.do_targets) == 1
        assert isinstance(req.do_targets[0], DoProposition)
        assert req.do_targets[0].proposition_id == "PROP_BANQUO_LINE_KINGS"

    def test_intervention_with_belief_do_target(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="If Macduff believed Macbeth's grief was sincere.",
            interventions={"ENT_MACDUFF.status": "healthy"},
            do_targets=[{"target_kind": "belief",
                         "holder_id": "ENT_MACDUFF",
                         "target_id": "ENT_MACBETH",
                         "proposition_id": "PROP_MACBETH_FOUL_PLAY",
                         "confidence": 0.0}],
        )
        req = _build_query(parsed, "If Macduff trusted Macbeth.")
        assert isinstance(req.do_targets[0], DoBelief)
        assert req.do_targets[0].confidence == 0.0

    def test_intervention_with_concern_do_target(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="If Lady Macbeth had no ambition.",
            interventions={"ENT_MACBETH.status": "healthy"},
            do_targets=[{"target_kind": "concern",
                         "holder_id": "ENT_MACBETH",
                         "concern_id": "CCN_LADY_DESIRES_CROWN",
                         "salience": 0.0}],
        )
        req = _build_query(parsed, "If Lady Macbeth had no ambition.")
        assert isinstance(req.do_targets[0], DoConcern)
        assert req.do_targets[0].salience == 0.0

    def test_counterfactual_lifts_to_historical_do_targets(self):
        parsed = ParsedQuery(
            query_type="counterfactual",
            reasoning="What if Macbeth had refused.",
            historical_interventions={"EVT_MURDER.event_type": "prevented"},
            do_targets=[{"target_kind": "event",
                         "event_id": "EVT_MURDER", "occurred": False}],
        )
        req = _build_query(parsed, "What if Macbeth refused?")
        assert isinstance(req, CounterfactualQuery)
        assert len(req.historical_do_targets) == 1
        assert isinstance(req.historical_do_targets[0], DoEvent)
        assert req.historical_do_targets[0].occurred is False

    def test_intervention_without_do_targets_falls_back_to_legacy(self):
        parsed = ParsedQuery(
            query_type="intervention",
            reasoning="Plain state change.",
            interventions={"ENT_MACBETH.status": "dead"},
        )
        req = _build_query(parsed, "Kill Macbeth.")
        assert isinstance(req, InterventionQuery)
        # Phase-6 list is empty…
        assert req.do_targets == []
        # …but legacy dotted dict survives for downstream coercion.
        assert "ENT_MACBETH.status" in req.interventions
