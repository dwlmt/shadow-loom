# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lifecycle-closure regression tests.

Pins the dynamic-element lifecycle invariants exercised end-to-end by
the ingestion post-passes and the entity replay path:

* Concern materialised closure (desire->False / fear->True) is treated
  the same as realised closure (close + low-salience snapshot) AND
  caps ``activation_fabula_window`` so post-resolution affect
  detectors operate on the event, not the standing concern.
* Belief invalidation is fine-grained: when a contradicted belief
  carries a ``proposition_id``, the post-pass writes a composite
  ``"target_id::PROP_..."`` invalidation key so co-located beliefs
  (same target, different proposition) are not over-invalidated.
* The replay merger in :func:`reconstruct_entity_state` honours both
  the legacy bare ``target_id`` and the new composite forms.
* Utterance/channel temporal validity flags utterances routed
  through a channel after its ``terminated_at_fabula``.
"""
from __future__ import annotations

from shadow_loom.models import (
    WorldStateV1, Location, Entity, EventNode, Channel, Proposition,
    Belief, Concern, ConcernSnapshot, EntityStateSnapshot, TraitVector,
    RelationshipEdge, RelationshipMetric,
    reconstruct_entity_at,
)
from shadow_loom.ingestion import (
    _post_pass_close_resolved_concerns,
    _post_pass_invalidate_contradicted_beliefs,
    _validate_time_ordering,
)
from shadow_loom.extract_graph import (
    _time_slice_relationship_at,
    extract_full_world_state,
)


def _bare_world() -> WorldStateV1:
    return WorldStateV1(
        locations={"LOC_A": Location(name="A", description="A", ambient_state={})},
        objects={},
        entities={},
        events=[],
        channels={},
        propositions=[],
        social_topology=[], spatial_topology=[], causal_topology=[],
        world_traits={},
    )


# ------------------------------------------------------- concern closure
class TestConcernMaterialisedClosure:
    def _world_with_concern(self, polarity: str, truth: bool) -> WorldStateV1:
        ws = _bare_world()
        ws.propositions = [
            Proposition(
                proposition_id="PROP_X", kind="outcome",
                referent_ids=["ENT_OWNER"], description="X happens",
                truth_at_fabula={1500: truth},
            )
        ]
        ws.entities = {
            "ENT_OWNER": Entity(
                id="ENT_OWNER", name="Owner", location_id="LOC_A",
                status="healthy", traits={}, beliefs=[],
                concerns=[Concern(
                    concern_id="CCN_X",
                    proposition_id="PROP_X",
                    polarity=polarity,
                    salience=0.8,
                )],
            ),
        }
        return ws

    def test_realised_desire_closes(self):
        """Desire->True realises the concern; standard closure."""
        ws = self._world_with_concern("desire", True)
        out = _post_pass_close_resolved_concerns(ws, [])
        c = out.entities["ENT_OWNER"].concerns[0]
        assert c.state_timeline, "expected a closing snapshot"
        assert c.state_timeline[-1].fabula_time == 1500
        assert c.state_timeline[-1].salience == 0.0

    def test_materialised_fear_closes_and_caps_window(self):
        """Fear->True MATERIALISES the concern. Previously the
        post-pass left such concerns standing — the bug. The fix
        closes them AND caps ``activation_fabula_window`` so
        post-resolution grief/rage detectors use the event, not the
        standing fear.
        """
        ws = self._world_with_concern("fear", True)
        out = _post_pass_close_resolved_concerns(ws, [])
        c = out.entities["ENT_OWNER"].concerns[0]
        assert c.state_timeline, "fear-materialised must produce a closing snapshot"
        assert c.state_timeline[-1].fabula_time == 1500
        assert c.activation_fabula_window is not None
        assert c.activation_fabula_window[1] == 1500, \
            "materialised closure must clamp the activation window upper bound"

    def test_materialised_desire_closes_and_caps_window(self):
        """Desire->False also materialises (as loss)."""
        ws = self._world_with_concern("desire", False)
        out = _post_pass_close_resolved_concerns(ws, [])
        c = out.entities["ENT_OWNER"].concerns[0]
        assert c.state_timeline, "desire-materialised must close the concern"
        assert c.activation_fabula_window is not None
        assert c.activation_fabula_window[1] == 1500


# ------------------------------------------------- belief invalidation
class TestBeliefInvalidationGranularity:
    def test_composite_key_used_when_proposition_known(self):
        ws = _bare_world()
        ws.propositions = [
            Proposition(
                proposition_id="PROP_DEAD", kind="outcome",
                referent_ids=["ENT_TARGET"], description="dead",
                truth_at_fabula={100: True},
            ),
            Proposition(
                proposition_id="PROP_TRAITOR", kind="trait_holds",
                referent_ids=["ENT_TARGET"], description="traitor",
                truth_at_fabula={},
            ),
        ]
        ws.entities = {
            "ENT_TARGET": Entity(
                id="ENT_TARGET", name="T", location_id="LOC_A",
                status="healthy", traits={}, beliefs=[], concerns=[],
            ),
            "ENT_HOLDER": Entity(
                id="ENT_HOLDER", name="H", location_id="LOC_A",
                status="healthy", traits={}, concerns=[],
                # Two beliefs about ENT_TARGET — only the "alive"
                # belief is contradicted by PROP_DEAD committing
                # True (low confidence + true commit = contradiction
                # rule).
                beliefs=[
                    Belief(target_id="ENT_TARGET", perceived_state="alive",
                           confidence=0.2, inertia=0.5, established_at_fabula=0,
                           proposition_id="PROP_DEAD"),
                    Belief(target_id="ENT_TARGET", perceived_state="trustworthy",
                           confidence=0.8, inertia=0.5, established_at_fabula=0,
                           proposition_id="PROP_TRAITOR"),
                ],
            ),
        }
        out = _post_pass_invalidate_contradicted_beliefs(ws, [])
        snaps = out.entities["ENT_HOLDER"].state_timeline
        # Exactly one invalidation snapshot, keyed by composite form.
        invs = [s for s in snaps if s.beliefs_invalidated]
        assert len(invs) == 1
        assert invs[0].beliefs_invalidated == ["ENT_TARGET::PROP_DEAD"]


class TestReplayHonoursCompositeInvalidation:
    def test_composite_drops_only_matching_belief(self):
        b_dead = Belief(target_id="ENT_T", perceived_state="alive",
                        confidence=0.8, inertia=0.5, established_at_fabula=0,
                        proposition_id="PROP_DEAD")
        b_traitor = Belief(target_id="ENT_T", perceived_state="loyal",
                           confidence=0.8, inertia=0.5, established_at_fabula=0,
                           proposition_id="PROP_TRAITOR")
        holder = Entity(
            id="ENT_H", name="H", location_id="LOC_A",
            status="healthy", traits={}, concerns=[],
            beliefs=[b_dead, b_traitor],
            state_timeline=[
                EntityStateSnapshot(
                    fabula_time=100,
                    beliefs_invalidated=["ENT_T::PROP_DEAD"],
                ),
            ],
        )
        result = reconstruct_entity_at(holder, fabula_time=200)
        # Only the dead belief is removed; traitor belief survives.
        kept_props = sorted(
            (b.get("proposition_id") if isinstance(b, dict) else b.proposition_id)
            for b in result["beliefs"]
        )
        assert kept_props == ["PROP_TRAITOR"]

    def test_legacy_bare_target_drops_all(self):
        """Backward-compat: a legacy bare target_id entry still
        drops every belief about that target.
        """
        b1 = Belief(target_id="ENT_T", perceived_state="x", confidence=0.5,
                    inertia=0.5, established_at_fabula=0, proposition_id="PROP_1")
        b2 = Belief(target_id="ENT_T", perceived_state="y", confidence=0.5,
                    inertia=0.5, established_at_fabula=0, proposition_id="PROP_2")
        holder = Entity(
            id="ENT_H", name="H", location_id="LOC_A",
            status="healthy", traits={}, concerns=[],
            beliefs=[b1, b2],
            state_timeline=[
                EntityStateSnapshot(
                    fabula_time=100,
                    beliefs_invalidated=["ENT_T"],  # legacy form
                ),
            ],
        )
        result = reconstruct_entity_at(holder, fabula_time=200)
        assert result["beliefs"] == []


# -------------------------------------- utterance/channel temporal check
class TestUtteranceChannelTemporalValidity:
    def test_utterance_after_channel_terminated_is_error(self):
        ws = _bare_world()
        ws.entities = {
            "ENT_A": Entity(id="ENT_A", name="A", location_id="LOC_A",
                            status="healthy", traits={}, beliefs=[], concerns=[]),
            "ENT_B": Entity(id="ENT_B", name="B", location_id="LOC_A",
                            status="healthy", traits={}, beliefs=[], concerns=[]),
        }
        ws.channels = {
            "CHN_X": Channel(
                id="CHN_X", name="x", medium="voice",
                participant_ids=["ENT_A", "ENT_B"],
                directionality="duplex",
                intelligibility={},
                established_at_fabula=0,
                terminated_at_fabula=100,
            ),
        }
        ws.events = [
            EventNode(
                id="EVT_LATE", fabula_time=200, syuzhet_index=0,
                event_type="utterance",
                actor_ids=["ENT_A"], target_ids=[],
                description="A speaks after the channel is dead",
                content="hi", speaker_id="ENT_A",
                addressee_ids=["ENT_B"], via_channel_id="CHN_X",
                truth_value="true",
            ),
        ]
        issues = _validate_time_ordering(ws)
        bad = [
            i for i in issues
            if i.category == "temporal" and i.severity == "error"
            and "terminated_at_fabula" in i.detail
        ]
        assert bad, "expected an error flagging utterance-after-channel-termination"


# ---------------------------------------- relationship lifecycle field
class TestRelationshipLifecycle:
    def _ws_with_rel(
        self, established: int = None, ended: int = None,
        metric_ts: int = 0,
    ) -> WorldStateV1:
        ws = _bare_world()
        ws.entities = {
            "ENT_A": Entity(id="ENT_A", name="A", location_id="LOC_A",
                            status="healthy", traits={}, beliefs=[], concerns=[]),
            "ENT_B": Entity(id="ENT_B", name="B", location_id="LOC_A",
                            status="healthy", traits={}, beliefs=[], concerns=[]),
        }
        # ``_validate_time_ordering`` short-circuits on an empty event
        # list — add one anchor event so the relationship-lifecycle
        # branch actually runs.
        ws.events = [
            EventNode(
                id="EVT_ANCHOR", fabula_time=0, syuzhet_index=0,
                event_type="outcome",
                actor_ids=["ENT_A"], target_ids=[],
                description="anchor", content="anchor",
            ),
        ]
        ws.social_topology = [
            RelationshipEdge(
                source_entity_id="ENT_A", target_entity_id="ENT_B",
                established_at_fabula=established,
                ended_at_fabula=ended,
                metrics={"affinity": RelationshipMetric(
                    value=0.5, last_updated_fabula=metric_ts,
                )},
            ),
        ]
        return ws

    def test_time_slice_drops_ended_relationship(self):
        ws = self._ws_with_rel(established=0, ended=100, metric_ts=50)
        # At t=50 (alive): edge present
        out_alive = _time_slice_relationship_at(ws.social_topology[0], 50)
        assert out_alive is not None
        # At t=200 (post-ended): edge dropped
        out_dead = _time_slice_relationship_at(ws.social_topology[0], 200)
        assert out_dead is None

    def test_time_slice_drops_pre_established_relationship(self):
        ws = self._ws_with_rel(established=100, metric_ts=120)
        out = _time_slice_relationship_at(ws.social_topology[0], 50)
        assert out is None

    def test_validator_flags_ended_before_established(self):
        ws = self._ws_with_rel(established=200, ended=100)
        issues = _validate_time_ordering(ws)
        bad = [
            i for i in issues
            if "ended_at_fabula" in i.detail and "established_at_fabula" in i.detail
        ]
        assert bad

    def test_validator_flags_metric_after_ended(self):
        ws = self._ws_with_rel(established=0, ended=100, metric_ts=200)
        issues = _validate_time_ordering(ws)
        bad = [
            i for i in issues
            if "severed relationship" in i.detail.lower() or "after ended_at" in i.detail.lower()
        ]
        assert bad


# ---------------------------------------- superseded-event filter
class TestSupersededEventFilter:
    def test_omniscient_extract_drops_superseded_when_successor_present(self):
        ws = _bare_world()
        ws.events = [
            EventNode(
                id="EVT_OLD", fabula_time=100, syuzhet_index=0,
                event_type="outcome", actor_ids=["ENT_A"], target_ids=[],
                description="old", content="old",
                superseded_by_event_id="EVT_NEW",
            ),
            EventNode(
                id="EVT_NEW", fabula_time=100, syuzhet_index=1,
                event_type="outcome", actor_ids=["ENT_A"], target_ids=[],
                description="new", content="new",
            ),
        ]
        ws.entities = {
            "ENT_A": Entity(id="ENT_A", name="A", location_id="LOC_A",
                            status="healthy", traits={}, beliefs=[], concerns=[]),
        }
        out = extract_full_world_state(ws)
        ids = {e["id"] for e in out["events"]}
        assert "EVT_NEW" in ids
        assert "EVT_OLD" not in ids


# ---------------------------------------- hidden-channel time-slice
class TestHiddenChannelTimeSlice:
    def test_terminated_channel_not_surfaced_post_termination(self):
        from shadow_loom.directive_assembly import compute_hidden_channels_for
        ws = _bare_world()
        ws.entities = {
            "ENT_A": Entity(id="ENT_A", name="A", location_id="LOC_A",
                            status="healthy", traits={}, beliefs=[], concerns=[]),
            "ENT_B": Entity(id="ENT_B", name="B", location_id="LOC_A",
                            status="healthy", traits={}, beliefs=[], concerns=[]),
        }
        ws.channels = {
            "CHN_DEAD": Channel(
                id="CHN_DEAD", name="dead", medium="voice",
                participant_ids=["ENT_A", "ENT_B"],
                directionality="duplex",
                intelligibility={"ENT_A": 0.0},  # would normally surface
                established_at_fabula=0,
                terminated_at_fabula=100,
            ),
        }
        # An event reveals fabula time 200 (post-termination) at syuzhet=5.
        ws.events = [
            EventNode(
                id="EVT_VISIBLE", fabula_time=200, syuzhet_index=5,
                event_type="outcome", actor_ids=["ENT_A"], target_ids=[],
                description="x", content="x",
            ),
        ]
        hidden = compute_hidden_channels_for(ws, syuzhet_anchor=10)
        ch_kinds = [h for h in hidden if h.kind == "channel" and h.channel_id == "CHN_DEAD"]
        assert ch_kinds == [], (
            "Terminated channel must not surface as hidden capability "
            "once the source text has reached/passed termination."
        )
