# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Round-trip coverage for legacy dotted-key ⇄ typed ``DoTarget`` coercion.

Locks in option (c) from the typed-do_targets migration: any single-key
encodable surgery the engine emits via
:func:`shadow_loom.narrative_physics._lift_do_targets_to_legacy_dict`
must coerce back to the original typed shape via
:func:`shadow_loom.query_models._coerce_legacy_dict`, and the
auto-validators on :class:`InterventionQuery` / :class:`CounterfactualQuery`
must populate the typed list at construction time so every downstream
consumer sees it without having to call a coerce helper explicitly.
"""

from shadow_loom.query_models import (
    CounterfactualQuery,
    DoEvent,
    DoNarrativeObject,
    DoProposition,
    DoTrait,
    DoWorldTrait,
    InterventionQuery,
    _coerce_legacy_dict,
)
from shadow_loom.narrative_physics import _lift_do_targets_to_legacy_dict


def _roundtrip(targets):
    """Lift typed → legacy dict → typed list."""
    return _coerce_legacy_dict(_lift_do_targets_to_legacy_dict(targets))


def test_event_prevented_roundtrip():
    typed = [DoEvent(event_id="EVT_KILL_DUNCAN", occurred=False)]
    out = _roundtrip(typed)
    assert len(out) == 1
    assert isinstance(out[0], DoEvent)
    assert out[0].event_id == "EVT_KILL_DUNCAN"
    assert out[0].occurred is False


def test_event_occurred_roundtrip():
    typed = [DoEvent(event_id="EVT_FLEE", occurred=True)]
    out = _roundtrip(typed)
    assert isinstance(out[0], DoEvent)
    assert out[0].occurred is True


def test_proposition_truth_roundtrip():
    out = _coerce_legacy_dict({"PROP_KING_DEAD.truth": True})
    assert len(out) == 1
    assert isinstance(out[0], DoProposition)
    assert out[0].proposition_id == "PROP_KING_DEAD"
    assert out[0].truth is True


def test_world_trait_value_roundtrip():
    out = _coerce_legacy_dict({"WORLD_STORM.value": 0.75})
    assert len(out) == 1
    assert isinstance(out[0], DoWorldTrait)
    assert out[0].world_trait_id == "WORLD_STORM"
    assert out[0].value == 0.75


def test_object_location_roundtrip():
    out = _coerce_legacy_dict({"OBJ_DAGGER.location_id": "LOC_CHAMBER"})
    assert len(out) == 1
    assert isinstance(out[0], DoNarrativeObject)
    assert out[0].object_id == "OBJ_DAGGER"
    assert out[0].new_location_id == "LOC_CHAMBER"


def test_object_owner_null_roundtrip():
    out = _coerce_legacy_dict({"OBJ_DAGGER.owner_id": None})
    assert len(out) == 1
    assert isinstance(out[0], DoNarrativeObject)
    assert out[0].set_owner_null is True


def test_entity_trait_roundtrip():
    typed = [DoTrait(holder_id="ENT_MACBETH", trait_name="ambition", value=0.9)]
    out = _roundtrip(typed)
    assert len(out) == 1
    assert isinstance(out[0], DoTrait)
    assert out[0].holder_id == "ENT_MACBETH"
    assert out[0].trait_name == "ambition"
    assert out[0].value == 0.9


def test_intervention_query_auto_coerces_legacy_dict():
    q = InterventionQuery(interventions={
        "EVT_KILL_DUNCAN.event_type": "prevented",
        "PROP_KING_DEAD.truth": False,
    })
    kinds = {type(t).__name__ for t in q.do_targets}
    assert kinds == {"DoEvent", "DoProposition"}


def test_counterfactual_query_auto_coerces_legacy_dict():
    q = CounterfactualQuery(
        evidence_node_ids=["EVT_NOW"],
        historical_interventions={
            "EVT_KILL_DUNCAN.event_type": "prevented",
            "ENT_MACBETH.traits.ambition": 0.1,
        },
    )
    kinds = {type(t).__name__ for t in q.historical_do_targets}
    assert kinds == {"DoEvent", "DoTrait"}


def test_typed_do_targets_take_precedence():
    """When typed already present, validator must not overwrite."""
    typed = [DoEvent(event_id="EVT_X", occurred=True)]
    q = InterventionQuery(
        do_targets=typed,
        interventions={"EVT_Y.event_type": "prevented"},
    )
    assert q.do_targets == typed


def test_belief_and_concern_skipped_in_legacy_coercion():
    """DoBelief / DoConcern need pair info absent from a single dotted key.
    They must stay absent rather than be invented from a partial encoding."""
    out = _coerce_legacy_dict({
        "BEL_ARBITRARY.value": True,
        "CCN_ARBITRARY.value": 0.5,
    })
    assert out == []


# ---------------------------------------------------------------------
# Step 1 extension: lift parity for Prop / WorldTrait / NarrativeObject
# ---------------------------------------------------------------------


def test_proposition_lift_roundtrip():
    typed = [DoProposition(proposition_id="PROP_KING_DEAD", truth=True)]
    out = _roundtrip(typed)
    assert len(out) == 1
    assert isinstance(out[0], DoProposition)
    assert out[0].proposition_id == "PROP_KING_DEAD"
    assert out[0].truth is True


def test_world_trait_lift_roundtrip():
    typed = [DoWorldTrait(world_trait_id="WORLD_STORM", value=0.42)]
    out = _roundtrip(typed)
    assert len(out) == 1
    assert isinstance(out[0], DoWorldTrait)
    assert out[0].world_trait_id == "WORLD_STORM"
    assert out[0].value == 0.42


def test_narrative_object_location_lift_roundtrip():
    typed = [DoNarrativeObject(
        object_id="OBJ_DAGGER", new_location_id="LOC_CHAMBER",
    )]
    out = _roundtrip(typed)
    assert len(out) == 1
    assert isinstance(out[0], DoNarrativeObject)
    assert out[0].object_id == "OBJ_DAGGER"
    assert out[0].new_location_id == "LOC_CHAMBER"


def test_narrative_object_owner_null_lift_roundtrip():
    typed = [DoNarrativeObject(object_id="OBJ_DAGGER", set_owner_null=True)]
    out = _roundtrip(typed)
    assert len(out) == 1
    assert isinstance(out[0], DoNarrativeObject)
    assert out[0].set_owner_null is True


def test_narrative_object_combined_location_and_owner_lift_roundtrip():
    typed = [DoNarrativeObject(
        object_id="OBJ_RING",
        new_location_id="LOC_RIVER",
        new_owner_id="ENT_FRODO",
    )]
    legacy = _lift_do_targets_to_legacy_dict(typed)
    # Two distinct dotted keys, one per mutation.
    assert set(legacy.keys()) == {"OBJ_RING.location_id", "OBJ_RING.owner_id"}
    coerced = _coerce_legacy_dict(legacy)
    # Coerce yields one DoNarrativeObject per key. We only assert the
    # mutation surface remains complete after the round-trip.
    assert {t.object_id for t in coerced} == {"OBJ_RING"}
    locs = {t.new_location_id for t in coerced if t.new_location_id}
    owns = {t.new_owner_id for t in coerced if t.new_owner_id}
    assert locs == {"LOC_RIVER"}
    assert owns == {"ENT_FRODO"}


# ---------------------------------------------------------------------
# Step 2: validator warns on unhandled legacy prefixes
# ---------------------------------------------------------------------


def test_intervention_query_warns_on_belief_prefix():
    import pytest
    with pytest.warns(UserWarning, match="BEL_"):
        q = InterventionQuery(interventions={"BEL_ARBITRARY.value": True})
    # Coercer still returns an empty typed list \u2014 the warning is the
    # only signal that the legacy dict could not be honoured.
    assert q.do_targets == []


def test_intervention_query_warns_on_causal_edge_prefix():
    import pytest
    with pytest.warns(UserWarning, match="CAUSAL_EDGE_"):
        InterventionQuery(interventions={
            "CAUSAL_EDGE_X.strength": 0.5,
            "EVT_DUNCAN.event_type": "prevented",  # supported, no warn-trigger
        })


def test_counterfactual_query_warns_on_concern_prefix():
    import pytest
    with pytest.warns(UserWarning, match="CCN_"):
        CounterfactualQuery(
            evidence_node_ids=["EVT_NOW"],
            historical_interventions={"CCN_RAGE.salience": 0.9},
        )


def test_intervention_query_no_warning_for_supported_prefixes():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail
        InterventionQuery(interventions={
            "EVT_DUNCAN.event_type": "prevented",
            "PROP_KING_DEAD.truth": True,
            "WORLD_STORM.value": 0.5,
            "OBJ_DAGGER.location_id": "LOC_CHAMBER",
            "ENT_MACBETH.traits.ambition": 0.9,
        })


def test_intervention_query_strict_mode_raises_on_unencodable_prefix(monkeypatch):
    """``SHADOW_LOOM_STRICT_LEGACY_PREFIXES=1`` upgrades the soft warning
    to a hard ``ValueError`` so CI can fail loudly on silently-dropped
    surgeries."""
    import pytest
    from shadow_loom import query_models
    monkeypatch.setenv("SHADOW_LOOM_STRICT_LEGACY_PREFIXES", "1")
    query_models._refresh_strict_mode_from_env()
    try:
        with pytest.raises(ValueError, match="BEL_"):
            InterventionQuery(interventions={"BEL_ARBITRARY.value": True})
    finally:
        monkeypatch.delenv("SHADOW_LOOM_STRICT_LEGACY_PREFIXES", raising=False)
        query_models._refresh_strict_mode_from_env()


def test_counterfactual_query_strict_mode_raises(monkeypatch):
    import pytest
    from shadow_loom import query_models
    monkeypatch.setenv("SHADOW_LOOM_STRICT_LEGACY_PREFIXES", "true")
    query_models._refresh_strict_mode_from_env()
    try:
        with pytest.raises(ValueError, match="REL_"):
            CounterfactualQuery(
                evidence_node_ids=["EVT_NOW"],
                historical_interventions={"REL_X.strength": 0.5},
            )
    finally:
        monkeypatch.delenv("SHADOW_LOOM_STRICT_LEGACY_PREFIXES", raising=False)
        query_models._refresh_strict_mode_from_env()


def test_intervention_query_strict_mode_disabled_by_default(monkeypatch):
    """Without the env var, the warning path remains the default."""
    import pytest
    from shadow_loom import query_models
    monkeypatch.delenv("SHADOW_LOOM_STRICT_LEGACY_PREFIXES", raising=False)
    query_models._refresh_strict_mode_from_env()
    with pytest.warns(UserWarning, match="BEL_"):
        InterventionQuery(interventions={"BEL_X.value": True})


# ---------------------------------------------------------------------
# validate_assignment: post-construction field reassignment re-fires
# the typed-do_targets backfill validator
# ---------------------------------------------------------------------


def test_intervention_query_reassigning_interventions_refires_validator():
    q = InterventionQuery()
    assert q.do_targets == []
    # Whole-field reassignment triggers validate_assignment, which in
    # turn re-runs the @model_validator(mode="after") backfill.
    q.interventions = {"EVT_X.event_type": "prevented"}
    assert len(q.do_targets) == 1
    assert isinstance(q.do_targets[0], DoEvent)
    assert q.do_targets[0].occurred is False


def test_counterfactual_query_reassigning_historical_interventions_refires_validator():
    q = CounterfactualQuery(evidence_node_ids=["EVT_NOW"])
    assert q.historical_do_targets == []
    q.historical_interventions = {"PROP_X.truth": True}
    assert len(q.historical_do_targets) == 1
    assert isinstance(q.historical_do_targets[0], DoProposition)


# ---------------------------------------------------------------------
# Per-instance warning dedup: validate_assignment fires the validator
# on every field write, but the unhandled-prefix warning must only
# emit once per (instance, offender-set) combination.
# ---------------------------------------------------------------------


def test_unhandled_prefix_warning_deduped_across_unrelated_field_writes():
    """Setting unrelated fields must not re-emit the prefix warning.

    With ``validate_assignment=True`` the model_validator re-fires on
    every field write, so without per-instance dedup a caller filling
    in N fields would emit the same warning N times.
    """
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = InterventionQuery(interventions={"BEL_X.value": True})
        q.target_node_ids = ["EVT_NOW"]
        q.original_query = "something"
    # Exactly one warning across construction + two unrelated assigns.
    prefix_warnings = [w for w in caught if "BEL_" in str(w.message)]
    assert len(prefix_warnings) == 1


def test_unhandled_prefix_warning_re_emits_when_offenders_change():
    """Reassigning ``interventions`` to a NEW unencodable payload must
    fire a fresh warning — dedup is keyed on the offender signature,
    not just 'have we ever warned at all'."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = InterventionQuery(interventions={"BEL_X.value": True})
        # Same offender set → deduped.
        q.interventions = {"BEL_X.value": True}
        # New offender set → fresh warning.
        q.interventions = {"CCN_Y.salience": 0.5}
    prefix_warnings = [
        w for w in caught if "BEL_" in str(w.message) or "CCN_" in str(w.message)
    ]
    assert len(prefix_warnings) == 2


# ---------------------------------------------------------------------
# Step 3: validator fires via model_validate / model_validate_json
# ---------------------------------------------------------------------


def test_intervention_query_model_validate_coerces_legacy_dict():
    q = InterventionQuery.model_validate({
        "query_type": "intervention",
        "interventions": {"EVT_KILL_DUNCAN.event_type": "prevented"},
    })
    assert len(q.do_targets) == 1
    assert isinstance(q.do_targets[0], DoEvent)
    assert q.do_targets[0].occurred is False


def test_intervention_query_model_validate_json_coerces_legacy_dict():
    import json
    payload = json.dumps({
        "query_type": "intervention",
        "interventions": {"PROP_KING_DEAD.truth": True},
    })
    q = InterventionQuery.model_validate_json(payload)
    assert len(q.do_targets) == 1
    assert isinstance(q.do_targets[0], DoProposition)
    assert q.do_targets[0].truth is True


def test_counterfactual_query_model_validate_coerces_legacy_dict():
    q = CounterfactualQuery.model_validate({
        "query_type": "counterfactual",
        "evidence_node_ids": ["EVT_NOW"],
        "historical_interventions": {
            "EVT_KILL_DUNCAN.event_type": "prevented",
            "ENT_MACBETH.traits.ambition": 0.1,
        },
    })
    kinds = {type(t).__name__ for t in q.historical_do_targets}
    assert kinds == {"DoEvent", "DoTrait"}


def test_counterfactual_query_model_validate_json_coerces_legacy_dict():
    import json
    payload = json.dumps({
        "query_type": "counterfactual",
        "evidence_node_ids": ["EVT_NOW"],
        "historical_interventions": {"WORLD_STORM.value": 0.9},
    })
    q = CounterfactualQuery.model_validate_json(payload)
    assert len(q.historical_do_targets) == 1
    assert isinstance(q.historical_do_targets[0], DoWorldTrait)


def test_intervention_query_model_validate_does_not_duplicate_typed():
    """When typed targets are supplied, model_validate must not invent
    extras from a co-present legacy dict (precedence is typed-wins)."""
    q = InterventionQuery.model_validate({
        "query_type": "intervention",
        "do_targets": [{
            "target_kind": "event",
            "event_id": "EVT_X",
            "occurred": True,
        }],
        "interventions": {"EVT_Y.event_type": "prevented"},
    })
    assert len(q.do_targets) == 1
    assert q.do_targets[0].event_id == "EVT_X"


# ---------------------------------------------------------------------
# Step 4: _do_target_items_to_typed warns on dropped records
# ---------------------------------------------------------------------


def test_do_target_items_to_typed_warns_on_dropped_record():
    import pytest
    from shadow_loom.query_parsing import _do_target_items_to_typed
    # Missing required ``event_id`` should drop and warn.
    with pytest.warns(UserWarning, match="dropped"):
        out = _do_target_items_to_typed([
            {"target_kind": "event"},  # missing event_id \u2014 dropped
        ])
    assert out == []


def test_do_target_items_to_typed_warns_on_unknown_kind():
    import pytest
    from shadow_loom.query_parsing import _do_target_items_to_typed
    with pytest.warns(UserWarning, match="dropped"):
        _do_target_items_to_typed([
            {"target_kind": "totally_made_up"},
        ])


def test_do_target_items_to_typed_no_warning_on_clean_input():
    import warnings
    from shadow_loom.query_parsing import _do_target_items_to_typed
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail
        out = _do_target_items_to_typed([
            {"target_kind": "event", "event_id": "EVT_X", "occurred": True},
        ])
    assert len(out) == 1
    assert isinstance(out[0], DoEvent)
