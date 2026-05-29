"""CC-1 (2026-05-29) — contract test for branch-payload ↔ branch-model parity.

The ``_build_downstream_cascade_payload`` helper in ``generation.py``
emits a dict of pre-formatted cascade lines that is then unpacked into
:class:`CounterfactualBranch`, :class:`InterventionBranch`, and
:class:`ThreatProximity`. Pydantic silently drops keys that aren't
declared as model fields, so if a new cascade dimension is added to
the payload without also being declared on every consumer model, the
data is lost between physics and renderer/auditor with no exception.

The original ``event_cascade_detail`` bug (R2-2) was exactly this:
the payload bundled it, but no branch model declared the field, so
the renderer was told "no event cascade" even when chain_reaction
descendants had been pruned by ``causal_physics`` Step B.6.

This test locks the contract: every key returned by
``_build_downstream_cascade_payload`` must be a declared field on
all three downstream models.
"""

from shadow_loom.generation import _build_downstream_cascade_payload
from shadow_loom.directive_assembly import (
    CounterfactualBranch,
    InterventionBranch,
    ThreatProximity,
)


def test_cascade_payload_keys_are_branch_model_fields():
    payload = _build_downstream_cascade_payload(
        mutations=[],
        social_mutations=[],
        proposition_mutations=[],
        belief_mutations=[],
        concern_mutations=[],
        object_mutations=[],
        world_trait_mutations=[],
        edge_mutations=[],
        entity_delete_mutations=[],
        object_delete_mutations=[],
        event_mutations=[],
        blocked=[],
    )
    payload_keys = set(payload.keys())

    for model in (CounterfactualBranch, InterventionBranch, ThreatProximity):
        model_fields = set(model.model_fields.keys())
        missing = payload_keys - model_fields
        assert not missing, (
            f"{model.__name__} is missing branch-model fields for "
            f"cascade payload keys: {sorted(missing)}. Add them as "
            f"``List[str] = Field(default_factory=list)`` so the data "
            f"isn't silently dropped by pydantic."
        )


def test_event_cascade_detail_specifically_present():
    """Regression guard for the R2-2 bug — keep this even if the
    generic contract test above already covers it, because this
    field is the canary for chain_reaction prune cascades reaching
    the renderer.
    """
    payload = _build_downstream_cascade_payload(event_mutations=[])
    assert "event_cascade_detail" in payload
    for model in (CounterfactualBranch, InterventionBranch, ThreatProximity):
        assert "event_cascade_detail" in model.model_fields, (
            f"{model.__name__}.event_cascade_detail missing \u2014 "
            f"chain_reaction descendant prune cascades will be lost."
        )
