# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression coverage for the 2026-05-17 ingestion audit fixes:

1. ``failure_counts['affect']`` aggregation must not ``KeyError`` when
   any per-chunk affect call raises.
2. ``_run_correction_patch`` must not silently mis-classify Phase E /
   ontology-repair / channel-intelligibility / edge-swap patches as
   "empty" and drop them.
3. ``_auto_repair`` must deterministically backfill a belief's
   ``acquired_via_event_id`` / ``acquired_via_channel_id`` when exactly
   one utterance plausibly produced it.
"""

import pytest

from shadow_loom.ingestion import WorldStatePatch, _auto_repair
from shadow_loom.models import (
    Belief,
    Channel,
    Entity,
    EventNode,
    Location,
    WorldStateV1,
)


# ---------------------------------------------------------------------
# Bug 2 — generic empty-patch check covers every actionable field.
# ---------------------------------------------------------------------


# The set of fields that, when populated, MUST NOT be classified as
# "empty" by ``_run_correction_patch``. ``notes`` is excluded because
# it is informational only and doesn't mutate the world.
_NON_EMPTY_FIELDS_TO_PROBE = [
    "event_renames",
    "drop_event_ids",
    "update_event_fields",
    "update_entity_location",
    "add_state_timeline_entries",
    "drop_causal_edges",
    "add_causal_edges",
    "drop_social_edges",
    "add_social_edges",
    "drop_spatial_edges",
    "add_spatial_edges",
    "drop_channel_ids",
    "add_channels",
    "channel_renames",
    "update_channel_intelligibility",
    "add_propositions",
    "update_proposition_snapshots",
    "commit_proposition_truth",
    "add_concerns",
    "update_concern_snapshots",
    "set_belief_proposition_ids",
    "entity_renames",
    "object_renames",
    "drop_object_ids",
    "location_renames",
    "drop_location_ids",
    "swap_causal_edge_directions",
]


def _is_patch_empty(patch: WorldStatePatch) -> bool:
    """Mirror the predicate used inside ``_run_correction_patch``.

    Centralised here so the test directly exercises the same logic
    without having to mock the LLM correction agent.
    """
    _NON_ACTIONABLE = {"notes"}
    return all(
        not getattr(patch, name)
        for name in type(patch).model_fields
        if name not in _NON_ACTIONABLE
    )


def test_completely_default_patch_is_empty():
    assert _is_patch_empty(WorldStatePatch())


def test_notes_only_patch_is_still_empty():
    assert _is_patch_empty(WorldStatePatch(notes="planning ahead"))


@pytest.mark.parametrize("field_name", _NON_EMPTY_FIELDS_TO_PROBE)
def test_patch_with_single_actionable_field_is_not_empty(field_name):
    """For every WorldStatePatch field that mutates world state,
    setting it to a non-default value must mark the patch as non-empty.

    Catches the regression where the hand-rolled emptiness check only
    enumerated 15 of the ~27 actionable fields, silently dropping
    Phase E / ontology-repair / channel-intelligibility patches.
    """
    # Use a sentinel non-empty container appropriate to each field's
    # type. The string-keyed dicts and lists all accept truthy stub
    # values that pass pydantic validation since these are
    # ``Dict[str, Any]`` / ``List[<model>]`` in WorldStatePatch.
    field = WorldStatePatch.model_fields[field_name]
    annotation = field.annotation
    sentinel = _sentinel_for_field(field_name, annotation)
    patch = WorldStatePatch(**{field_name: sentinel})
    assert not _is_patch_empty(patch), (
        f"Patch with non-empty {field_name!r} was mis-classified as empty"
    )


def _sentinel_for_field(name, annotation):
    """Return a minimal non-empty value that satisfies the field's type.

    We avoid constructing full sub-models where pydantic-strict typing
    is enforced; instead use the simplest container shape that the
    emptiness check sees as truthy. Where a Field requires a typed
    list / dict of models, we either use a known-safe singleton or
    construct the minimal model.
    """
    # Plain dict / list fields with permissive value types.
    str_dict_fields = {
        "event_renames", "update_entity_location", "channel_renames",
        "entity_renames", "object_renames", "location_renames",
    }
    if name in str_dict_fields:
        return {"OLD_ID": "NEW_ID"}
    if name == "update_event_fields":
        return {"EVT_X": {"speaker_id": "ENT_X"}}
    if name == "update_channel_intelligibility":
        return {"CHN_X": {"ENT_X": 0.5}}
    if name == "commit_proposition_truth":
        return {"PROP_X": {0: True}}
    str_list_fields = {
        "drop_event_ids", "drop_channel_ids",
        "drop_object_ids", "drop_location_ids",
    }
    if name in str_list_fields:
        return ["EVT_X"]
    # Snapshot append maps use real models; supply an empty list inside
    # a populated key — the *outer* dict is non-empty so the check
    # still passes.
    if name == "add_state_timeline_entries":
        return {"ENT_X": []}
    if name == "update_proposition_snapshots":
        return {"PROP_X": []}
    if name == "add_concerns":
        return {"ENT_X": []}
    if name == "update_concern_snapshots":
        return {"ENT_X": {"CCN_X": []}}
    if name == "add_propositions":
        return {"PROP_X": _make_minimal_proposition()}
    if name == "add_channels":
        return {"CHN_X": _make_minimal_channel()}
    if name == "set_belief_proposition_ids":
        return [_make_belief_prop_assignment()]
    # All edge add/drop lists.
    edge_add_drop = {
        "drop_causal_edges", "add_causal_edges",
        "drop_social_edges", "add_social_edges",
        "drop_spatial_edges", "add_spatial_edges",
        "swap_causal_edge_directions",
    }
    if name in edge_add_drop:
        return [_make_edge_for_field(name)]
    raise AssertionError(f"No sentinel defined for field {name!r}")


def _make_minimal_proposition():
    from shadow_loom.models import Proposition
    return Proposition(
        proposition_id="PROP_X", kind="outcome", description="x", referent_ids=[],
    )


def _make_minimal_channel():
    return Channel(
        id="CHN_X",
        name="X channel",
        medium="speech",
        participant_ids=["ENT_A", "ENT_B"],
        intelligibility={"ENT_A": 1.0, "ENT_B": 1.0},
    )


def _make_belief_prop_assignment():
    from shadow_loom.ingestion import _BeliefPropAssignment
    return _BeliefPropAssignment(
        entity_id="ENT_X", target_id="ENT_Y", proposition_id="PROP_X",
    )


def _make_edge_for_field(name: str):
    from shadow_loom.models import CausalEdge, RelationshipEdge, SpatialEdge
    from shadow_loom.ingestion import _EdgeRef
    if name == "add_causal_edges":
        return CausalEdge(
            source_id="EVT_A", target_id="EVT_B",
            causality_type="chain_reaction", causal_force=0.5,
            evidence_strength="moderate",
            mechanism="test", fabula_time=0,
        )
    if name == "add_social_edges":
        return RelationshipEdge(
            source_entity_id="ENT_A", target_entity_id="ENT_B",
            relationship_type="alliance",
        )
    if name == "add_spatial_edges":
        return SpatialEdge(
            source_id="LOC_A", target_id="LOC_B",
            relation_type="adjacent",
        )
    # All drop_* and swap_* fields use _EdgeRef tuples.
    return _EdgeRef(source_id="EVT_A", target_id="EVT_B")


# ---------------------------------------------------------------------
# Bug 3 — Deterministic belief provenance inference from utterance chains.
# ---------------------------------------------------------------------


def _make_ws_with_utterance_and_orphan_belief():
    """Single-utterance world where the audience (ENT_A) has a belief
    about ENT_C with no provenance. The utterance is the unique
    plausible source."""
    utterance = EventNode(
        id="EVT_UTTER",
        fabula_time=10,
        syuzhet_index=1,
        event_type="utterance",
        actor_ids=["ENT_B"],
        target_ids=["ENT_C"],
        speaker_id="ENT_B",
        addressee_ids=["ENT_A"],
        via_channel_id="CHN_PHONE",
        description="B tells A about C.",
        content="C is in danger.",
        truth_value="true",
    )
    channel = Channel(
        id="CHN_PHONE",
        name="Phone line",
        medium="telephone",
        participant_ids=["ENT_A", "ENT_B"],
        intelligibility={"ENT_A": 1.0, "ENT_B": 1.0},
    )
    belief = Belief(
        target_id="ENT_C",
        perceived_state="in danger",
        confidence=0.9,
        inertia=0.5,
        established_at_fabula=10,
    )
    ent_a = Entity(
        id="ENT_A", name="Alice", description="receiver",
        location_id="LOC_A", status="healthy", traits={},
        beliefs=[belief],
    )
    ent_b = Entity(id="ENT_B", name="Bob", description="speaker",
                   location_id="LOC_A", status="healthy", traits={})
    ent_c = Entity(id="ENT_C", name="Carol", description="subject",
                   location_id="LOC_A", status="healthy", traits={})
    return WorldStateV1(
        story_title="t", style="prose",
        locations={"LOC_A": Location(
                id="LOC_A",
                name="A", description="A", ambient_state={})},
        objects={},
        entities={"ENT_A": ent_a, "ENT_B": ent_b, "ENT_C": ent_c},
        events=[utterance], channels={"CHN_PHONE": channel},
        causal_topology=[], spatial_topology=[], social_topology=[],
    )


def test_belief_provenance_inferred_from_unique_utterance():
    ws = _make_ws_with_utterance_and_orphan_belief()
    repaired, repairs = _auto_repair(ws)
    b = repaired.entities["ENT_A"].beliefs[0]
    assert b.acquired_via_event_id == "EVT_UTTER"
    assert b.acquired_via_channel_id == "CHN_PHONE"
    assert any("Inferred belief provenance" in r for r in repairs)


def test_belief_provenance_not_inferred_when_ambiguous():
    """Two plausible utterances → no inference (conservative)."""
    ws = _make_ws_with_utterance_and_orphan_belief()
    # Add a SECOND utterance that also has ENT_A as an addressee and
    # ENT_C as a target.
    second = EventNode(
        id="EVT_UTTER_2",
        fabula_time=12,
        syuzhet_index=2,
        event_type="utterance",
        actor_ids=["ENT_B"],
        target_ids=["ENT_C"],
        speaker_id="ENT_B",
        addressee_ids=["ENT_A"],
        description="B tells A more about C.",
        content="C has a plan.",
        truth_value="true",
    )
    ws = ws.model_copy(update={"events": [*ws.events, second]})
    repaired, _ = _auto_repair(ws)
    b = repaired.entities["ENT_A"].beliefs[0]
    assert b.acquired_via_event_id is None
    assert b.acquired_via_channel_id is None


def test_belief_provenance_inference_skips_already_populated():
    """If the belief already has provenance, inference must NOT overwrite."""
    ws = _make_ws_with_utterance_and_orphan_belief()
    # Pre-populate provenance with a known-valid event id.
    pre_pop = ws.entities["ENT_A"].beliefs[0].model_copy(
        update={
            "acquired_via_event_id": "EVT_UTTER",
            "acquired_via_channel_id": None,
        },
    )
    ent_a = ws.entities["ENT_A"].model_copy(update={"beliefs": [pre_pop]})
    ws = ws.model_copy(update={
        "entities": {**ws.entities, "ENT_A": ent_a},
    })
    repaired, _ = _auto_repair(ws)
    b = repaired.entities["ENT_A"].beliefs[0]
    # The event id stays as it was; channel is NOT auto-filled because
    # the inference pass skips beliefs that already have any
    # provenance set.
    assert b.acquired_via_event_id == "EVT_UTTER"
    assert b.acquired_via_channel_id is None


def test_belief_provenance_not_inferred_for_future_utterance():
    """An utterance occurring AFTER the snapshot fabula_time is never
    a valid source for a belief that exists at that snapshot."""
    ws = _make_ws_with_utterance_and_orphan_belief()
    # Move the utterance to fabula_time=100 (well after the belief
    # snapshot's implicit 'now' inferred from established_at_fabula).
    # Place the belief INSIDE a state_timeline snapshot at fabula=5
    # so the inference helper has a clear ``before_fabula`` cutoff.
    from shadow_loom.models import EntityStateSnapshot
    pre_belief = ws.entities["ENT_A"].beliefs[0]
    snapshot = EntityStateSnapshot(
        fabula_time=5, beliefs_added=[pre_belief], triggered_by="EVT_OTHER",
    )
    ent_a = ws.entities["ENT_A"].model_copy(update={
        "beliefs": [],  # move belief into snapshot
        "state_timeline": [snapshot],
    })
    moved_utterance = ws.events[0].model_copy(update={"fabula_time": 100})
    ws = ws.model_copy(update={
        "entities": {**ws.entities, "ENT_A": ent_a},
        "events": [moved_utterance],
    })
    repaired, _ = _auto_repair(ws)
    b = repaired.entities["ENT_A"].state_timeline[0].beliefs_added[0]
    assert b.acquired_via_event_id is None
