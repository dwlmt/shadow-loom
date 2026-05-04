# SPDX-FileCopyrightText: 2026 David Rae Wilmot
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integration tests for the channel/belief/utterance integration work.

Covers behaviours added across P0-P3:

* Parser: counterfactual surgery may target ``CHN_*`` and ``EVT_*`` (utterance)
  fields including ``status`` and ``truth_value``.
* Instantiator: utterance events from the ego-payload land in the sandbox;
  ``communicating_with: []`` prunes beliefs whose provenance points at the
  severed pair.
* Causal physics: destructive surgery on a channel or utterance prunes
  beliefs whose ``acquired_via_*`` provenance has been invalidated, and
  surfaces ``pruned_beliefs_count`` / ``pruned_utterance_event_ids`` /
  ``disabled_channel_ids`` on the result.
* AMWN: ``Channel`` is built as a first-class graph node with edges to
  participants and per-utterance speaker→utterance→channel→addressee
  routing.
* Generation: when ``brief.hidden_channels`` is non-empty, the rendering
  prompt contains the ``UTTERANCE & CHANNEL FIDELITY`` block.
* Auditor: deterministic withheld-utterance leak pre-check raises a typed
  ``withheld_utterance_leak`` violation when prose echoes future utterance
  ``content``.
* Directive assembly: ``HiddenChannel.unintelligible_for`` is populated from
  the channel's intelligibility map and the configured
  ``intelligibility_threshold``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import List

import pytest

from shadow_loom.models import (
    WorldStateV1,
    Location,
    Entity,
    EventNode,
    Channel,
    Belief,
    TraitVector,
)


# ============================================================
# Shared fixture: tiny world with one channel, one utterance,
# and one belief acquired via that utterance/channel.
# ============================================================
def _make_channel_world() -> WorldStateV1:
    """Two-entity world with a single utterance over a single channel.

    Alice tells Bob (via CHN_PHONE) that "the safe is empty".
    Bob holds a belief tagged with ``acquired_via_event_id="EVT_UTT"``
    and ``acquired_via_channel_id="CHN_PHONE"``.
    """
    return WorldStateV1(
        locations={
            "LOC_HOME": Location(name="Home", description="Home", ambient_state={}),
        },
        objects={},
        entities={
            "ENT_ALICE": Entity(
                id="ENT_ALICE",
                name="Alice",
                location_id="LOC_HOME",
                status="healthy",
                traits={"trust": TraitVector(value=0.5, inertia=0.2)},
            ),
            "ENT_BOB": Entity(
                id="ENT_BOB",
                name="Bob",
                location_id="LOC_HOME",
                status="healthy",
                traits={"trust": TraitVector(value=0.5, inertia=0.2)},
                beliefs=[
                    Belief(
                        target_id="OBJ_SAFE",
                        perceived_state="The safe is empty",
                        confidence=0.9,
                        inertia=0.3,
                        established_at_fabula=10,
                        acquired_via_event_id="EVT_UTT",
                        acquired_via_channel_id="CHN_PHONE",
                    ),
                ],
            ),
        },
        events=[
            EventNode(
                id="EVT_UTT",
                fabula_time=10,
                syuzhet_index=1,
                event_type="utterance",
                actor_ids=["ENT_ALICE"],
                target_ids=[],
                speaker_id="ENT_ALICE",
                addressee_ids=["ENT_BOB"],
                via_channel_id="CHN_PHONE",
                content="The safe is empty",
                truth_value="true",
                description="Alice tells Bob the safe is empty",
            ),
        ],
        causal_topology=[],
        channels={
            "CHN_PHONE": Channel(
                id="CHN_PHONE",
                name="phone line",
                medium="telephone",
                participant_ids=["ENT_ALICE", "ENT_BOB"],
                directionality="duplex",
                intelligibility={"ENT_BOB": 1.0, "ENT_ALICE": 1.0},
                established_at_fabula=0,
                evidence_strength="strong",
            ),
        },
    )


# ============================================================
# 1. Parser: channel/utterance counterfactuals are accepted.
# ============================================================
class TestParserChannelCounterfactuals:
    def test_channel_id_appears_in_typed_collection(self):
        from shadow_loom.query_parsing import _collect_typed_ids

        ws = _make_channel_world()
        typed = _collect_typed_ids(ws)
        assert "CHN_PHONE" in typed["channel_ids"]
        assert "EVT_UTT" in typed["event_ids"]
        assert "EVT_UTT" in typed.get("utterance_event_ids", [])

    def test_counterfactual_dynamic_model_accepts_channel_target(self):
        from shadow_loom.query_parsing import _build_counterfactual_dynamic_model

        ws = _make_channel_world()
        Model = _build_counterfactual_dynamic_model(ws)
        # Construct via dict — should validate without error.
        instance = Model(
            reasoning="Test channel surgery interpretation.",
            historical_interventions=[
                {
                    "target_id": "CHN_PHONE",
                    "property": "status",
                    "value": "severed",
                }
            ],
            forward_deltas=[],
        )
        assert instance.historical_interventions[0].target_id == "CHN_PHONE"

    def test_counterfactual_dynamic_model_accepts_utterance_truth_flip(self):
        from shadow_loom.query_parsing import _build_counterfactual_dynamic_model

        ws = _make_channel_world()
        Model = _build_counterfactual_dynamic_model(ws)
        instance = Model(
            reasoning="Test utterance truth flip.",
            historical_interventions=[
                {
                    "target_id": "EVT_UTT",
                    "property": "truth_value",
                    "value": "false",
                }
            ],
            forward_deltas=[],
        )
        assert instance.historical_interventions[0].property == "truth_value"


# ============================================================
# 2. Causal physics: provenance prune on destructive surgery.
# ============================================================
class TestCausalPhysicsProvenancePrune:
    def _collect(self, interventions):
        from shadow_loom.causal_physics import CausalPhysicsEngine

        # Use a bare engine just to call the helper. Settings come from the
        # default config — we only exercise the static intervention parser.
        eng = CausalPhysicsEngine.__new__(CausalPhysicsEngine)
        return eng._collect_provenance_invalidations(interventions)

    def test_severed_channel_collected(self):
        evts, chans = self._collect({"CHN_PHONE.status": "severed"})
        assert chans == {"CHN_PHONE"}
        assert evts == set()

    def test_disabled_status_collected(self):
        _, chans = self._collect({"CHN_PHONE.status": "disabled"})
        assert chans == {"CHN_PHONE"}

    def test_empty_participants_collected(self):
        _, chans = self._collect({"CHN_PHONE.participant_ids": []})
        assert chans == {"CHN_PHONE"}

    def test_legacy_chan_prefix_also_collected(self):
        # The parser docstring historically said CHAN_*; both prefixes must
        # be honoured so older queries don't silently no-op.
        _, chans = self._collect({"CHAN_OLD.status": "severed"})
        assert chans == {"CHAN_OLD"}

    def test_false_utterance_collected(self):
        evts, chans = self._collect({"EVT_UTT.truth_value": "false"})
        assert evts == {"EVT_UTT"}
        assert chans == set()

    def test_performative_utterance_collected(self):
        evts, _ = self._collect({"EVT_UTT.truth_value": "performative"})
        assert evts == {"EVT_UTT"}

    def test_prevented_event_collected(self):
        evts, _ = self._collect({"EVT_UTT.event_type": "prevented"})
        assert evts == {"EVT_UTT"}

    def test_non_destructive_relabel_does_not_collect(self):
        # Relabelling an outcome is not destructive — no provenance prune.
        evts, chans = self._collect({"EVT_UTT.outcome": "reworded"})
        assert evts == set() and chans == set()


# ============================================================
# 3. CausalPhysicsResult exposes new prune fields with defaults.
# ============================================================
class TestCausalPhysicsResultFields:
    def test_default_fields_present_and_empty(self):
        from shadow_loom.causal_physics import CausalPhysicsResult
        import networkx as nx

        # CausalPhysicsResult requires a serialised sandbox_data dict
        # (nx.node_link_data) so the result is JSON-round-trippable
        # through the pipeline / MCP layer.
        result = CausalPhysicsResult(
            physics_state=nx.MultiDiGraph(),
            sandbox_data=nx.node_link_data(nx.MultiDiGraph()),
        )
        assert result.pruned_beliefs_count == 0
        assert result.pruned_utterance_event_ids == []
        assert result.disabled_channel_ids == []


# ============================================================
# 4. AMWN: channel becomes a first-class graph node with
#    speaker→utterance→channel→addressee routing.
# ============================================================
class TestAMWNChannelNode:
    def test_channel_node_and_participant_edges_present(self):
        from shadow_loom.amwn import build_causal_diagram

        ws = _make_channel_world()
        g = build_causal_diagram(ws)
        assert g.has_node("CHN_PHONE")
        # Channel↔participant edges (direction depends on impl; check either way).
        touches_alice = g.has_edge("CHN_PHONE", "ENT_ALICE") or g.has_edge(
            "ENT_ALICE", "CHN_PHONE"
        )
        touches_bob = g.has_edge("CHN_PHONE", "ENT_BOB") or g.has_edge(
            "ENT_BOB", "CHN_PHONE"
        )
        assert touches_alice
        assert touches_bob

    def test_utterance_routed_through_channel(self):
        from shadow_loom.amwn import build_causal_diagram

        ws = _make_channel_world()
        g = build_causal_diagram(ws)
        # Speaker → utterance and utterance → addressee or channel hop.
        # We accept either a direct (speaker, utt) edge or a routing via
        # the channel — the key invariant is that Bob is reachable from
        # Alice through the utterance/channel substructure.
        import networkx as nx
        assert nx.has_path(g, "ENT_ALICE", "ENT_BOB")


# ============================================================
# 5. Generation: UTTERANCE & CHANNEL FIDELITY block.
# ============================================================
class TestGenerationFidelityBlock:
    def test_fidelity_block_present_when_hidden_channels(self):
        from shadow_loom.generation import assemble_rendering_prompt
        from shadow_loom.directive_assembly import CreativeBrief, HiddenChannel

        brief = CreativeBrief(
            target_effect="suspense",
            target_entities=["ENT_ALICE"],
            scene_context={"syuzhet_anchor": 5},
            hidden_channels=[
                HiddenChannel(
                    kind="channel",
                    channel_id="CHN_PHONE",
                    medium="telephone",
                    participant_ids=["ENT_ALICE", "ENT_BOB"],
                    unintelligible_for=["ENT_BOB"],
                )
            ],
        )
        prompt = assemble_rendering_prompt(brief)
        assert "UTTERANCE & CHANNEL FIDELITY" in prompt
        assert "CHN_PHONE" in prompt

    def test_fidelity_block_absent_when_no_hidden_channels(self):
        from shadow_loom.generation import assemble_rendering_prompt
        from shadow_loom.directive_assembly import CreativeBrief

        brief = CreativeBrief(
            target_effect="observation",
            target_entities=["ENT_ALICE"],
            scene_context={},
            hidden_channels=[],
        )
        prompt = assemble_rendering_prompt(brief)
        assert "UTTERANCE & CHANNEL FIDELITY" not in prompt


# ============================================================
# 6. Auditor: typed violations + deterministic leak pre-check.
# ============================================================
class TestAuditorWithheldUtteranceLeak:
    def test_violation_type_extended(self):
        from shadow_loom.auditor import AuditViolation

        # Confirm the literal accepts the new typed values.
        for vt in (
            "withheld_utterance_leak",
            "channel_intelligibility_violation",
            "utterance_truth_contradiction",
            "belief_provenance_contradiction",
        ):
            v = AuditViolation(
                violation_type=vt,
                severity="major",
                description=f"test {vt}",
                feedback=f"test feedback for {vt}",
            )
            assert v.violation_type == vt

    def test_leak_detector_flags_future_utterance_substring(self):
        from shadow_loom.auditor import _withheld_utterance_leak_violations

        ws = _make_channel_world()
        # Add a future utterance whose content (>=12 chars) leaks into prose.
        ws.events.append(
            EventNode(
                id="EVT_FUTURE",
                fabula_time=99,
                syuzhet_index=99,
                event_type="utterance",
                actor_ids=["ENT_ALICE"],
                speaker_id="ENT_ALICE",
                addressee_ids=["ENT_BOB"],
                content="I poisoned the chalice last night",
                truth_value="true",
                description="future confession",
            )
        )
        prose = "She turned and whispered: I poisoned the chalice last night, then walked out."
        violations = _withheld_utterance_leak_violations(prose, ws, syuzhet_anchor=5)
        assert violations, "expected a leak violation"
        assert any(v.violation_type == "withheld_utterance_leak" for v in violations)

    def test_leak_detector_silent_on_past_utterance(self):
        from shadow_loom.auditor import _withheld_utterance_leak_violations

        ws = _make_channel_world()
        # EVT_UTT has syuzhet_index=1; anchor=5 → it's already revealed.
        prose = "Bob remembered: The safe is empty, she had said."
        violations = _withheld_utterance_leak_violations(prose, ws, syuzhet_anchor=5)
        assert not violations, "past utterance should not trigger leak"


# ============================================================
# 7. Directive assembly: unintelligible_for surfaces low-intel
#    participants per the configured threshold.
# ============================================================
class TestHiddenChannelUnintelligibleFor:
    def _hidden(self, ws, anchor):
        from shadow_loom.directive_assembly import DirectiveAssembler
        from shadow_loom.extract_graph import extract_ego_graph_from_memory

        ego = extract_ego_graph_from_memory(ws, ["ENT_ALICE"])
        assembler = DirectiveAssembler(None, ego, ws)
        return assembler.compute_hidden_channels(syuzhet_anchor=anchor)

    def test_unintelligible_participant_listed(self):
        ws = _make_channel_world()
        # Drop Bob's intelligibility below the default 0.3 threshold.
        ws.channels["CHN_PHONE"].intelligibility["ENT_BOB"] = 0.1
        # Add a withheld utterance so the channel surfaces.
        ws.events.append(
            EventNode(
                id="EVT_HIDDEN",
                fabula_time=50,
                syuzhet_index=50,
                event_type="utterance",
                actor_ids=["ENT_ALICE"],
                speaker_id="ENT_ALICE",
                addressee_ids=["ENT_BOB"],
                via_channel_id="CHN_PHONE",
                content="Run, the police are here",
                truth_value="true",
                description="warning",
            )
        )
        # anchor=0 — every utterance on CHN_PHONE is withheld, so the
        # channel itself surfaces as a hidden channel record (kind='channel').
        hidden = self._hidden(ws, anchor=0)
        phone = next((h for h in hidden if h.channel_id == "CHN_PHONE" and h.kind == "channel"), None)
        assert phone is not None, "CHN_PHONE should be surfaced as hidden"
        assert "ENT_BOB" in phone.unintelligible_for

    def test_low_intel_only_channel_still_surfaces(self):
        """Even with no withheld utterance, a low-intelligibility channel
        must surface so the renderer can simulate the comprehension gap."""
        ws = _make_channel_world()
        ws.channels["CHN_PHONE"].intelligibility["ENT_BOB"] = 0.05
        # syuzhet_anchor large enough that EVT_UTT is past — no future utts.
        # The on-page utterance EVT_UTT was via CHN_PHONE so the channel
        # is technically revealed; with a sub-threshold intelligibility,
        # ``compute_hidden_channels`` must still flag the channel as a
        # comprehension-asymmetry signal even though it is on-page.
        # We pre-empt that by removing the on-page utterance to make the
        # "never carries an on-page utterance" branch fire.
        ws.events = [e for e in ws.events if e.id != "EVT_UTT"]
        hidden = self._hidden(ws, anchor=999)
        ids = [h.channel_id for h in hidden if h.kind == "channel"]
        assert "CHN_PHONE" in ids
        phone = next(h for h in hidden if h.channel_id == "CHN_PHONE" and h.kind == "channel")
        assert "ENT_BOB" in phone.unintelligible_for


# ============================================================
# 8. Narrative physics + MCP: prune fields surface in result.
# ============================================================
class TestNarrativePhysicsResultSurfacesPrune:
    def test_intervention_result_has_prune_keys(self):
        from shadow_loom.narrative_physics import calculate_narrative_physics
        from shadow_loom.query_models import InterventionQuery

        ws = _make_channel_world()
        # The pruning surface is wired only through the CausalPhysicsEngine
        # path — the legacy AMWNInstantiator-only path doesn't emit
        # belief/utterance pruning, so engine mode is required here.
        result = calculate_narrative_physics(
            InterventionQuery(
                # Pull Alice into focus so the channel surgery has a
                # concrete sandbox to bind to.
                focus_entity_ids=["ENT_ALICE", "ENT_BOB"],
                interventions={"CHN_PHONE.status": "severed"},
            ),
            ws,
            use_causal_engine=True,
        )
        # The engine result either succeeds (with prune keys) or is
        # flagged as implausible; both shapes are valid responses, but
        # when the surgery binds we must surface the new pruning keys.
        if result.get("status") == "success":
            assert "pruned_beliefs_count" in result
            assert "pruned_utterance_event_ids" in result
            assert "disabled_channel_ids" in result
        else:
            assert result.get("status") == "implausible"
