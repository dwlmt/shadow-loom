# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression coverage for the generic empty-patch check in
``_run_correction_patch``.

The previous hand-rolled emptiness check enumerated only the 15 legacy
``WorldStatePatch`` fields, silently mis-classifying patches that
touched ONLY Phase E ops (propositions / concerns /
belief-prop links), ontology repairs (entity / object / location
renames + drops), channel-intelligibility tweaks, or causal-edge
direction swaps. Those patches were dropped with a ``"empty_patch"``
log line and never applied.

These tests assert that the generic ``model_fields``-driven check
treats every actionable field as a real signal.
"""

from shadow_loom.ingestion import WorldStatePatch
from shadow_loom.models import Concern, Proposition


# The check inside ``_run_correction_patch`` (shadow_loom/ingestion.py
# ~L17190). Mirrored here so we test the predicate directly without
# spinning up the full correction-agent LLM pipeline.
_NON_ACTIONABLE_PATCH_FIELDS = {"notes"}


def _is_empty_patch(patch: WorldStatePatch) -> bool:
    return all(
        not getattr(patch, name)
        for name in type(patch).model_fields
        if name not in _NON_ACTIONABLE_PATCH_FIELDS
    )


def test_truly_empty_patch_is_empty():
    assert _is_empty_patch(WorldStatePatch()) is True


def test_notes_only_patch_is_still_empty():
    assert _is_empty_patch(WorldStatePatch(notes="just a comment")) is True


# --- Phase E ops (previously misclassified) ----------------------------


def test_add_propositions_only_is_non_empty():
    patch = WorldStatePatch(add_propositions={
        "PROP_X": Proposition(
            proposition_id="PROP_X", kind="outcome", description="x is x",
        ),
    })
    assert _is_empty_patch(patch) is False


def test_commit_proposition_truth_only_is_non_empty():
    patch = WorldStatePatch(commit_proposition_truth={
        "PROP_X": {100: True},
    })
    assert _is_empty_patch(patch) is False


def test_update_proposition_snapshots_only_is_non_empty():
    # Pass an opaque list element \u2014 the empty check only inspects truthiness.
    patch = WorldStatePatch.model_validate({
        "update_proposition_snapshots": {"PROP_X": [{"fabula_time": 1}]},
    })
    assert _is_empty_patch(patch) is False


def test_add_concerns_only_is_non_empty():
    patch = WorldStatePatch(add_concerns={
        "ENT_MACBETH": [Concern(
            concern_id="CCN_X", proposition_id="PROP_X", polarity="desire",
        )],
    })
    assert _is_empty_patch(patch) is False


def test_update_concern_snapshots_only_is_non_empty():
    patch = WorldStatePatch.model_validate({
        "update_concern_snapshots": {
            "ENT_MACBETH": {"CCN_X": [{"fabula_time": 1}]},
        },
    })
    assert _is_empty_patch(patch) is False


def test_set_belief_proposition_ids_only_is_non_empty():
    patch = WorldStatePatch.model_validate({
        "set_belief_proposition_ids": [
            {"entity_id": "ENT_A", "target_id": "ENT_B", "proposition_id": "PROP_X"},
        ],
    })
    assert _is_empty_patch(patch) is False


# --- Ontology repairs (previously misclassified) -----------------------


def test_entity_renames_only_is_non_empty():
    patch = WorldStatePatch(entity_renames={"ENT_OLD": "ENT_NEW"})
    assert _is_empty_patch(patch) is False


def test_object_renames_only_is_non_empty():
    patch = WorldStatePatch(object_renames={"OBJ_OLD": "OBJ_NEW"})
    assert _is_empty_patch(patch) is False


def test_drop_object_ids_only_is_non_empty():
    patch = WorldStatePatch(drop_object_ids=["OBJ_X"])
    assert _is_empty_patch(patch) is False


def test_location_renames_only_is_non_empty():
    patch = WorldStatePatch(location_renames={"LOC_OLD": "LOC_NEW"})
    assert _is_empty_patch(patch) is False


def test_drop_location_ids_only_is_non_empty():
    patch = WorldStatePatch(drop_location_ids=["LOC_X"])
    assert _is_empty_patch(patch) is False


# --- Channel intelligibility + edge swaps (previously misclassified) ---


def test_update_channel_intelligibility_only_is_non_empty():
    patch = WorldStatePatch(update_channel_intelligibility={
        "CHN_X": {"ENT_A": 0.3},
    })
    assert _is_empty_patch(patch) is False


def test_swap_causal_edge_directions_only_is_non_empty():
    patch = WorldStatePatch.model_validate({
        "swap_causal_edge_directions": [
            {"source_id": "EVT_A", "target_id": "EVT_B"},
        ],
    })
    assert _is_empty_patch(patch) is False


# --- Legacy fields (regression: should still be non-empty) -------------


def test_event_renames_only_is_non_empty():
    patch = WorldStatePatch(event_renames={"EVT_OLD": "EVT_NEW"})
    assert _is_empty_patch(patch) is False


def test_drop_event_ids_only_is_non_empty():
    patch = WorldStatePatch(drop_event_ids=["EVT_X"])
    assert _is_empty_patch(patch) is False
